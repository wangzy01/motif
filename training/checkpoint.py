"""Checkpoint / scheduler / architecture and continuation contracts.

本模块集中训练入口的"状态持久化与一致性校验"逻辑，使 train.py 回到干净状态。
The historical v1 payload remains a valid *init_from* source after its model,
optimizer and architecture contracts pass.  Same-stage resume now requires the
additive ``mot_full_training_v2`` schema because exact continuation also needs
all per-rank RNG streams.  Legacy v1 resume is intentionally rejected; the one
audited A-final exception is explicit ``init_from`` and starts a fresh lineage.

含四类：
  * build_scheduler                : LambdaLR（warmup + constant/cosine）构建
  * _ablation_fingerprint          : 架构/消融开关指纹（F2，mask-only 切换也能挡）
  * _ablation_fingerprint_check    : resume 时指纹比对（旧 ckpt 无指纹则 warn 跳过）
  * _ablation_mismatch_check       : resume 时可训练参数名集合比对
  * save_checkpoint / load_checkpoint : 单进程 + FSDP2（DCP full_state_dict gather）双模

train.py / scripts / experiments 仍以 `from train import ...` 调用：train.py 通过本模块的
import 把这些名字再导出，调用点解析不变。
"""

from __future__ import annotations

import math
import os
import secrets
import stat
from collections.abc import Mapping

import torch

from training.formal_boot import (
    FORMAL_RUN_LINEAGE_KEY,
    normalize_expected_run_lineage_sha256,
    resolve_expected_run_lineage_sha256,
)
from training.rng_state import (
    FULL_CHECKPOINT_SCHEMA,
    collect_rng_state,
    validate_init_from_rng_policy,
    validate_resume_checkpoint,
)


WEIGHTS_EMA_SNAPSHOT_SCHEMA = "mot_weights_ema_snapshot_v1"


def build_scheduler(optimizer, config, max_steps: int):
    warmup = int(config.scheduler.get("num_warmup_steps", 0))

    def lr_lambda(step):
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if str(config.scheduler.get("decay", "constant")) == "cosine":
            progress = (step - warmup) / max(1, max_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def derive_max_steps(config, batch_size, world_size, log=None, cli_max_steps=None) -> int:
    """衔接（gated）：给了 trainer.max_exposure_images 则按【曝光量 / 实际 global_batch】派生 max_steps，
    使阶段步数随卡数自适应（本 trainer 无梯度累积，一次 optimizer step = 一个 global batch）。
    global_batch 优先用显式 trainer.global_batch_size；否则按 batch_size×world_size 现算。
    不给 max_exposure_images → 返回显式 trainer.max_steps（旧行为，字节级零变化）。
    cli_max_steps（命令行 --max-steps，调试/smoke 用）若显式给出则【优先于一切】，并跳过曝光量派生
    （否则在设了 max_exposure_images 的 redesign config 上 --max-steps 会被静默吞掉=footgun）。"""
    _log = log or (lambda m: None)
    max_steps = int(config.trainer.max_steps)
    exp = int(config.trainer.get("max_exposure_images", 0) or 0)
    if cli_max_steps is not None:                          # 显式 --max-steps 优先（调试/smoke 逃生口）
        if exp > 0:
            _log(f"[train] ⚠️ 显式 --max-steps={int(cli_max_steps)} 优先于 max_exposure_images={exp:,}；跳过曝光量派生（调试/smoke）")
        return int(cli_max_steps)
    if exp <= 0:
        return max_steps
    actual_gbs = batch_size * world_size
    explicit_gbs = int(config.trainer.get("global_batch_size", 0) or 0)
    gbs = explicit_gbs or actual_gbs
    if explicit_gbs and explicit_gbs != actual_gbs:
        _log(f"[train] ⚠️ global_batch_size={explicit_gbs} 与实际 batch_size×world={actual_gbs} 不符；"
             f"按显式值派生 max_steps（请确认梯度累积/卡数是否如预期）")
    derived = -(-exp // gbs)                               # ceil(exposure / global_batch)
    _log(f"[train] max_steps={derived} ← 曝光量 {exp:,} / global_batch {gbs} "
         f"（{'显式 global_batch_size' if explicit_gbs else f'batch_size {batch_size}×world {world_size} 现算'}；"
         f"config.max_steps={max_steps} 仅作参考）")
    return derived


# --------------------------------------------------------------------------- #
# checkpoint：只存可训练参数（frozen 塔由 pretrained_path 复原）+ optimizer/scheduler。
# FSDP2 下用 DCP get_state_dict(full_state_dict) 把 sharded DTensor gather 成完整张量。
# --------------------------------------------------------------------------- #
def _ablation_fingerprint(config) -> dict:
    """架构/消融开关指纹：捕获**改变 forward 语义但可能不改参数名**的开关（如
    `text_adapter.bidirectional` 只翻 causal mask），让 resume 能挡住 `_ablation_mismatch_check`
    （只比参数名）漏掉的"同名语义切换"（审计 F2：bidirectional↔causal 静默错续）。"""
    m = config.model
    ta = m.get("text_adapter", {}) or {}
    ls = m.get("long_skip", {}) or {}
    ab = m.get("abs_sincos", {}) or {}
    ul = m.get("und_lora", {}) or {}
    mlf = (ta.get("multi_layer_fusion", {}) or {})  # Ablation 2：嵌在 text_adapter 下
    targets = ul.get("target_modules", ["q", "k", "v", "o"])
    if isinstance(targets, str):
        targets = [t.strip() for t in targets.split(",") if t.strip()]
    return {
        "text_adapter.enabled": bool(ta.get("enabled", False)),
        "text_adapter.bidirectional": bool(ta.get("bidirectional", True)),
        "text_adapter.num_layers": int(ta.get("num_layers", 2)),
        "text_adapter.rope": bool(ta.get("rope", False)),
        "text_adapter.multi_layer_fusion.enabled": bool(mlf.get("enabled", False)),
        "text_adapter.multi_layer_fusion.num_layers": int(mlf.get("num_layers", 12)),
        "und_lora.enabled": bool(ul.get("enabled", False)),
        "und_lora.num_layers": int(ul.get("num_layers", 8)),
        "und_lora.rank": int(ul.get("rank", 32)),
        "und_lora.alpha": float(ul.get("alpha", ul.get("rank", 32))),
        "und_lora.lr_scale": float(ul.get("lr_scale", 1.0)),
        "und_lora.target_modules": ",".join(sorted(str(t) for t in targets)),
        "long_skip.enabled": bool(ls.get("enabled", False)),
        "long_skip.mode": str(ls.get("mode", "A_full")),
        "noise_condition": bool(m.get("noise_condition", True)),
        "abs_sincos.enabled": bool(ab.get("enabled", False)),
        "sandwich_norm": bool(m.get("sandwich_norm", True)),
        "patch_size": int(m.get("patch_size", 1)),
    }


def _ablation_fingerprint_check(saved_fp, config, *, require_exact: bool = False):
    """resume 时比对架构/消融指纹；任何不一致（含 mask-only 切换）→ 清晰报错。
    旧 ckpt 无指纹时跳过（向后兼容，仍由 `_ablation_mismatch_check` 兜参数名）。"""
    if saved_fp is None:
        print("[train] ⚠️ 该 checkpoint 无消融指纹（旧版），跳过指纹校验；仅靠参数名检查。"
              "mask-only 开关（如 text_adapter.bidirectional）切换可能未被挡住，请自行确认。", flush=True)
        return
    cur = _ablation_fingerprint(config)
    if require_exact and set(saved_fp) != set(cur):
        raise RuntimeError(
            "checkpoint ablation fingerprint key closure mismatch: "
            f"checkpoint_only={sorted(set(saved_fp) - set(cur))}, "
            f"current_only={sorted(set(cur) - set(saved_fp))}"
        )
    # 向后兼容：旧 ckpt 的指纹可能没有后加的键（如 text_adapter.multi_layer_fusion.*）；
    # 缺键时跳过该键（不当作不一致），避免误挡 resume —— 实际参数集变动仍由 _ablation_mismatch_check 兜底。
    diffs = {k: (saved_fp[k], cur[k]) for k in cur if k in saved_fp and saved_fp[k] != cur[k]}
    if diffs:
        lines = "\n".join(f"    {k}: checkpoint={a!r} vs 当前={b!r}" for k, (a, b) in diffs.items())
        raise RuntimeError(
            "checkpoint 与当前 config 的架构/消融指纹不一致，无法 resume（即使参数名相同，"
            "forward 语义已变，如 bidirectional↔causal）：\n" + lines +
            "\n  请用与该 checkpoint 一致的开关续训，或换新的 checkpoint_dir 从头训。"
        )


def _ablation_mismatch_check(saved_keys, model):
    """Require exact trainable-key closure before any permissive DCP load.

    DCP uses ``strict=False`` because the frozen tower is reconstructed from its
    pretrained source, but that must not make missing *trainable* tensors legal.
    The old implementation only noticed three ablation-name substrings and could
    silently leave an ordinary new trainable tensor randomly initialized.
    """
    cur_keys = {n for n, p in model.model.named_parameters() if p.requires_grad}
    saved = set(saved_keys)
    unexpected = sorted(saved - cur_keys)
    missing = sorted(cur_keys - saved)
    if unexpected or missing:
        raise RuntimeError(
            "checkpoint trainable-key closure mismatch; refusing permissive state load."
            f"\n  checkpoint_only({len(unexpected)}): {unexpected[:8]}"
            f"\n  current_only({len(missing)}): {missing[:8]}"
        )


def _current_optimizer_param_names(model, optimizer) -> tuple[list[list[str]], dict[int, str]]:
    named = {id(p): n for n, p in model.model.named_parameters() if p.requires_grad}
    groups: list[list[str]] = []
    seen: set[int] = set()
    for group_index, group in enumerate(optimizer.param_groups):
        names: list[str] = []
        for param in group.get("params", []):
            pid = id(param)
            if pid not in named:
                raise RuntimeError(
                    f"optimizer group {group_index} contains a non-trainable or foreign parameter"
                )
            if pid in seen:
                raise RuntimeError(f"optimizer parameter {named[pid]!r} appears more than once")
            seen.add(pid)
            names.append(named[pid])
        groups.append(names)
    if seen != set(named):
        missing = sorted(named[pid] for pid in set(named) - seen)
        raise RuntimeError(f"optimizer omits trainable parameters: {missing[:8]}")
    return groups, named


def _optimizer_state_integrity_check(saved_optimizer, trainable_state, model, optimizer) -> None:
    """Validate complete Adam state and map it to every current trainable tensor."""
    if not isinstance(saved_optimizer, Mapping):
        raise RuntimeError("checkpoint optimizer must be a mapping")
    if not isinstance(trainable_state, Mapping) or not trainable_state:
        raise RuntimeError("checkpoint trainable_state must be a non-empty mapping")
    state = saved_optimizer.get("state")
    groups = saved_optimizer.get("param_groups")
    if not isinstance(state, Mapping):
        raise RuntimeError("checkpoint optimizer.state must be a mapping")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("checkpoint optimizer.param_groups must be a non-empty list")

    current_groups, current_name_by_id = _current_optimizer_param_names(model, optimizer)
    if len(groups) != len(current_groups):
        raise RuntimeError(
            f"optimizer param-group count mismatch: checkpoint={len(groups)}, current={len(current_groups)}"
        )
    flat_saved_ids = []
    id_to_name = {}
    for group_index, (saved_group, current_names) in enumerate(zip(groups, current_groups)):
        if not isinstance(saved_group, Mapping) or not isinstance(saved_group.get("params"), list):
            raise RuntimeError(f"checkpoint optimizer param_group[{group_index}] is malformed")
        saved_ids = list(saved_group["params"])
        if len(saved_ids) != len(current_names):
            raise RuntimeError(
                f"optimizer group {group_index} size mismatch: "
                f"checkpoint={len(saved_ids)}, current={len(current_names)}"
            )
        if all(isinstance(saved_id, str) for saved_id in saved_ids) and set(saved_ids) != set(current_names):
            raise RuntimeError(
                f"optimizer FQN membership mismatch in group {group_index}: "
                f"checkpoint_only={sorted(set(saved_ids) - set(current_names))[:8]}, "
                f"current_only={sorted(set(current_names) - set(saved_ids))[:8]}"
            )
        for saved_id, current_name in zip(saved_ids, current_names):
            try:
                duplicate = saved_id in id_to_name
            except TypeError as exc:
                raise RuntimeError("checkpoint optimizer parameter ids must be hashable") from exc
            if duplicate:
                raise RuntimeError(f"checkpoint optimizer parameter id duplicated: {saved_id!r}")
            id_to_name[saved_id] = current_name
            flat_saved_ids.append(saved_id)

    if set(state) != set(flat_saved_ids):
        missing_state = [x for x in flat_saved_ids if x not in state]
        unexpected_state = [x for x in state if x not in id_to_name]
        raise RuntimeError(
            "checkpoint optimizer state coverage mismatch: "
            f"missing={missing_state[:8]}, unexpected={unexpected_state[:8]}"
        )
    trainable_names = set(trainable_state)
    if all(isinstance(x, str) for x in flat_saved_ids) and set(flat_saved_ids) != trainable_names:
        raise RuntimeError(
            "checkpoint optimizer FQN set differs from trainable_state: "
            f"optimizer_only={sorted(set(flat_saved_ids) - trainable_names)[:8]}, "
            f"trainable_only={sorted(trainable_names - set(flat_saved_ids))[:8]}"
        )
    if all(isinstance(x, str) for x in flat_saved_ids):
        # DCP full optimizer state uses trainable FQNs.  Bind by identity, not
        # incidental param-group order.
        id_to_name = {saved_id: saved_id for saved_id in flat_saved_ids}
    if len(flat_saved_ids) != len(trainable_names):
        raise RuntimeError(
            f"checkpoint optimizer parameter count {len(flat_saved_ids)} != "
            f"trainable tensor count {len(trainable_names)}"
        )

    for saved_id in flat_saved_ids:
        name = id_to_name[saved_id]
        record = state[saved_id]
        if not isinstance(record, Mapping):
            raise RuntimeError(f"optimizer state for {name!r} must be a mapping")
        required = {"step", "exp_avg", "exp_avg_sq"}
        if not required.issubset(record):
            raise RuntimeError(
                f"optimizer state for {name!r} lacks {sorted(required - set(record))}"
            )
        step = record["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1 or not bool(torch.isfinite(step).all()):
                raise RuntimeError(f"optimizer step for {name!r} must be one finite scalar")
        elif not isinstance(step, (int, float)) or not math.isfinite(float(step)):
            raise RuntimeError(f"optimizer step for {name!r} must be finite")
        expected_shape = tuple(trainable_state[name].shape)
        for key in ("exp_avg", "exp_avg_sq"):
            value = record[key]
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"optimizer {key} for {name!r} must be a tensor")
            if tuple(value.shape) != expected_shape:
                raise RuntimeError(
                    f"optimizer {key} shape mismatch for {name!r}: "
                    f"checkpoint={tuple(value.shape)}, parameter={expected_shape}"
                )
            if value.dtype != torch.float32:
                raise RuntimeError(
                    f"optimizer {key} dtype mismatch for {name!r}: expected=float32, got={value.dtype}"
                )


def _runtime_optimizer_integrity_check(model, optimizer) -> None:
    groups, name_by_id = _current_optimizer_param_names(model, optimizer)
    del groups
    state_ids = {id(param) for param in optimizer.state}
    if state_ids != set(name_by_id):
        missing = sorted(name_by_id[pid] for pid in set(name_by_id) - state_ids)
        raise RuntimeError(f"loaded optimizer has incomplete runtime state: missing={missing[:8]}")
    for param, record in optimizer.state.items():
        name = name_by_id[id(param)]
        if not isinstance(record, Mapping):
            raise RuntimeError(f"loaded optimizer state for {name!r} is malformed")
        required = {"step", "exp_avg", "exp_avg_sq"}
        if not required.issubset(record):
            raise RuntimeError(
                f"loaded optimizer state for {name!r} lacks {sorted(required - set(record))}"
            )
        step = record["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1 or not bool(torch.isfinite(step).all()):
                raise RuntimeError(f"loaded optimizer step for {name!r} is invalid")
        elif not isinstance(step, (int, float)) or not math.isfinite(float(step)):
            raise RuntimeError(f"loaded optimizer step for {name!r} is invalid")
        for key in ("exp_avg", "exp_avg_sq"):
            value = record.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(param.shape):
                raise RuntimeError(f"loaded optimizer {key} shape mismatch for {name!r}")
            if value.dtype != torch.float32:
                raise RuntimeError(f"loaded optimizer {key} dtype mismatch for {name!r}: {value.dtype}")


def _ema_state_integrity_check(ema_state, trainable_state, model) -> None:
    """EMA is part of the full-state continuation contract, never best-effort."""
    runtime_ema = getattr(model, "_ema", None)
    expects_ema = runtime_ema is not None
    if not expects_ema:
        if ema_state is not None:
            raise RuntimeError("checkpoint contains EMA but current run has EMA disabled")
        return
    if not isinstance(ema_state, Mapping):
        raise RuntimeError("EMA-enabled run requires checkpoint ema mapping")
    expected = set(trainable_state)
    if set(ema_state) != expected:
        raise RuntimeError(
            "checkpoint EMA key closure mismatch: "
            f"checkpoint_only={sorted(set(ema_state) - expected)[:8]}, "
            f"trainable_only={sorted(expected - set(ema_state))[:8]}"
        )
    for name in expected:
        value = ema_state[name]
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"checkpoint EMA {name!r} must be a tensor")
        if tuple(value.shape) != tuple(trainable_state[name].shape):
            raise RuntimeError(
                f"checkpoint EMA shape mismatch for {name!r}: "
                f"ema={tuple(value.shape)}, raw={tuple(trainable_state[name].shape)}"
            )
        if value.dtype != torch.float32:
            raise RuntimeError(
                f"checkpoint EMA dtype mismatch for {name!r}: expected=float32, got={value.dtype}"
            )


def _runtime_ema_integrity_check(model) -> None:
    runtime_ema = getattr(model, "_ema", None)
    if runtime_ema is None:
        return
    trainable = {n: p for n, p in model.model.named_parameters() if p.requires_grad}
    if set(runtime_ema) != set(trainable):
        raise RuntimeError(
            "loaded runtime EMA key closure mismatch: "
            f"ema_only={sorted(set(runtime_ema) - set(trainable))[:8]}, "
            f"trainable_only={sorted(set(trainable) - set(runtime_ema))[:8]}"
        )
    for name, value in runtime_ema.items():
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(trainable[name].shape):
            raise RuntimeError(f"loaded runtime EMA shape mismatch for {name!r}")
        if value.dtype != torch.float32:
            raise RuntimeError(f"loaded runtime EMA dtype mismatch for {name!r}: {value.dtype}")


def _scheduler_state_integrity_check(scheduler_state, global_step, optimizer_state) -> None:
    if not isinstance(scheduler_state, Mapping):
        raise RuntimeError("checkpoint scheduler must be a mapping")
    step = int(global_step)
    if scheduler_state.get("last_epoch") != step:
        raise RuntimeError(
            f"scheduler last_epoch/global_step mismatch: "
            f"last_epoch={scheduler_state.get('last_epoch')!r}, global_step={step}"
        )
    if scheduler_state.get("_step_count") != step + 1:
        raise RuntimeError(
            f"scheduler _step_count/global_step mismatch: "
            f"_step_count={scheduler_state.get('_step_count')!r}, expected={step + 1}"
        )
    optimizer_groups = optimizer_state.get("param_groups") if isinstance(optimizer_state, Mapping) else None
    if not isinstance(optimizer_groups, list) or not optimizer_groups:
        raise RuntimeError("cannot bind scheduler to malformed optimizer param_groups")
    group_count = len(optimizer_groups)
    for key in ("base_lrs", "_last_lr"):
        values = scheduler_state.get(key)
        if not isinstance(values, list) or len(values) != group_count:
            raise RuntimeError(
                f"scheduler {key} group count mismatch: expected={group_count}, "
                f"got={len(values) if isinstance(values, list) else type(values).__name__}"
            )
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
            raise RuntimeError(f"scheduler {key} must contain finite numeric values")
    for index, (group, last_lr) in enumerate(zip(optimizer_groups, scheduler_state["_last_lr"])):
        lr = group.get("lr") if isinstance(group, Mapping) else None
        if not isinstance(lr, (int, float)) or float(lr) != float(last_lr):
            raise RuntimeError(
                f"optimizer/scheduler lr mismatch at group {index}: optimizer={lr!r}, "
                f"scheduler={last_lr!r}"
            )


def _runtime_scheduler_integrity_check(scheduler, optimizer, global_step) -> None:
    state = scheduler.state_dict()
    optimizer_state = optimizer.state_dict()
    _scheduler_state_integrity_check(state, global_step, optimizer_state)


def _raise_synchronized_validation_error(error, *, is_dist: bool, context: str) -> None:
    """Make every rank fail before any following model-state collective."""
    if is_dist:
        import torch.distributed as dist
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(
            gathered,
            None if error is None else f"{type(error).__name__}: {error}",
        )
        failures = [(rank, value) for rank, value in enumerate(gathered) if value is not None]
        if failures:
            message = f"{context} validation failed across ranks: {failures}"
            if error is not None:
                raise RuntimeError(message) from error
            raise RuntimeError(message)
    elif error is not None:
        raise error


def sync_metrics_before_checkpoint(metrics_file, *, rank: int, is_dist: bool) -> None:
    """Durably commit rank-0 metrics before any checkpoint collective."""
    flush_error = None
    if rank == 0:
        try:
            if metrics_file is None:
                raise RuntimeError("rank 0 metrics handle is missing")
            metrics_file.flush()
            os.fsync(metrics_file.fileno())
        except BaseException as exc:
            flush_error = exc
    _raise_synchronized_validation_error(
        flush_error,
        is_dist=is_dist,
        context="metrics durable checkpoint boundary",
    )


def _fsync_directory(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_regular_file(path: str, *, context: str) -> None:
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"{context} is not an ordinary file: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"{context} changed identity before fsync: {path}")
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_exclusive_torch_tmp(payload, target: str, *, context: str) -> str:
    """Serialize to an unpredictable create-once file and fsync its bytes."""
    directory = os.path.dirname(target)
    stem = os.path.basename(target)
    for _ in range(16):
        tmp = os.path.join(
            directory,
            f".{stem}.tmp.{os.getpid()}.{secrets.token_hex(16)}",
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            fd = os.open(tmp, flags, 0o600)
        except FileExistsError:
            continue
        try:
            with os.fdopen(fd, "wb", closefd=False) as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(fd)
        _fsync_regular_file(tmp, context=context)
        return tmp
    raise RuntimeError(f"could not allocate exclusive {context} in {directory}")


def should_write_final_checkpoint(global_step: int, last_checkpoint_step: int | None) -> bool:
    """Avoid serializing the same terminal step twice.

    A duplicate final save would replace last.pt with a new inode after the
    milestone hard link was minted, making the correct permanent anchor appear
    stale.  Skipping it also avoids a second 26GB write.
    """
    return last_checkpoint_step is None or int(last_checkpoint_step) != int(global_step)


def _validate_snapshot_tensor_mapping(actual, expected, *, context: str) -> None:
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        raise RuntimeError(f"{context} must be a tensor mapping")
    if set(actual) != set(expected):
        raise RuntimeError(
            f"{context} key closure mismatch: "
            f"snapshot_only={sorted(set(actual) - set(expected))[:8]}, "
            f"last_only={sorted(set(expected) - set(actual))[:8]}"
        )
    for name in expected:
        saved = actual[name]
        source = expected[name]
        if not isinstance(saved, torch.Tensor) or not isinstance(source, torch.Tensor):
            raise RuntimeError(f"{context}[{name!r}] must be a tensor")
        if tuple(saved.shape) != tuple(source.shape) or saved.dtype != source.dtype:
            raise RuntimeError(
                f"{context}[{name!r}] metadata mismatch: "
                f"snapshot={tuple(saved.shape)}/{saved.dtype}, "
                f"last={tuple(source.shape)}/{source.dtype}"
            )
        if not torch.equal(saved, source):
            raise RuntimeError(f"{context}[{name!r}] value mismatch")


def _validate_weights_ema_snapshot(snapshot: str, full_payload) -> None:
    """Prove that an existing compact snapshot is exactly derived from last.pt.

    Loading with mmap keeps this generic compatibility path bounded in resident
    memory.  The formal Stage512A control uses ``full`` hard links instead, so
    this expensive tensor-by-tensor proof is not on its normal save path.
    """
    saved = torch.load(snapshot, map_location="cpu", weights_only=True, mmap=True)
    expected_top = {
        "snapshot_schema",
        "trainable_state",
        "ema",
        "global_step",
        "ablation_fp",
        FORMAL_RUN_LINEAGE_KEY,
    }
    if not isinstance(saved, Mapping) or set(saved) != expected_top:
        actual_keys = set(saved) if isinstance(saved, Mapping) else set()
        raise RuntimeError(
            "weights_ema periodic snapshot top-level closure mismatch: "
            f"snapshot_only={sorted(actual_keys - expected_top)}, "
            f"required_only={sorted(expected_top - actual_keys)}"
        )
    if saved.get("snapshot_schema") != WEIGHTS_EMA_SNAPSHOT_SCHEMA:
        raise RuntimeError(
            "weights_ema periodic snapshot schema mismatch: "
            f"{saved.get('snapshot_schema')!r}"
        )
    expected_step = full_payload["global_step"]
    actual_step = saved.get("global_step")
    if isinstance(actual_step, bool) or not isinstance(actual_step, int):
        raise RuntimeError("weights_ema periodic snapshot global_step must be int")
    if actual_step != expected_step:
        raise RuntimeError(
            "weights_ema periodic snapshot global_step mismatch: "
            f"snapshot={actual_step}, last={expected_step}"
        )
    if saved.get("ablation_fp") != full_payload["ablation_fp"]:
        raise RuntimeError("weights_ema periodic snapshot ablation fingerprint mismatch")
    if saved.get(FORMAL_RUN_LINEAGE_KEY) != full_payload.get(FORMAL_RUN_LINEAGE_KEY):
        raise RuntimeError("weights_ema periodic snapshot formal run lineage mismatch")
    _validate_snapshot_tensor_mapping(
        saved.get("trainable_state"),
        full_payload["trainable_state"],
        context="weights_ema periodic snapshot trainable_state",
    )
    expected_ema = full_payload["ema"]
    actual_ema = saved.get("ema")
    if expected_ema is None:
        if actual_ema is not None:
            raise RuntimeError("weights_ema periodic snapshot unexpectedly contains EMA")
    else:
        _validate_snapshot_tensor_mapping(
            actual_ema,
            expected_ema,
            context="weights_ema periodic snapshot ema",
        )


def _validate_historical_full_snapshots(
    path: str,
    current_payload,
    config,
    model,
    optimizer,
    *,
    rank: int,
    is_dist: bool,
) -> None:
    """Validate every permanent milestone that must predate current last.pt."""
    if config is None or str(
        config.trainer.get("checkpoint_keep_format", "weights_ema")
    ) != "full":
        return
    keep_every = int(config.trainer.get("checkpoint_keep_every", 0))
    keep_max = int(config.trainer.get("checkpoint_keep_max", 0))
    current_step = int(current_payload["global_step"])
    if keep_every <= 0 or keep_max <= 0 or current_step <= keep_every:
        return
    directory = os.path.dirname(path)
    expected_steps = [
        keep_every * index
        for index in range(1, keep_max + 1)
        if keep_every * index < current_step
    ]
    for milestone in expected_steps:
        snapshot = os.path.join(directory, f"step_{milestone}.pt")
        if not os.path.lexists(snapshot):
            raise RuntimeError(f"required permanent full snapshot is missing: {snapshot}")
        value = os.lstat(snapshot)
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
            raise RuntimeError(f"required permanent full snapshot is not ordinary: {snapshot}")
        if value.st_mode & 0o222:
            raise RuntimeError(f"required permanent full snapshot is not read-only: {snapshot}")
        anchor = torch.load(
            snapshot, map_location="cpu", weights_only=True, mmap=True
        )
        if not isinstance(anchor, Mapping) or set(anchor) != set(current_payload):
            raise RuntimeError(
                f"permanent full snapshot top-level closure mismatch: {snapshot}"
            )
        if anchor.get("checkpoint_schema") != FULL_CHECKPOINT_SCHEMA:
            raise RuntimeError(f"permanent full snapshot schema mismatch: {snapshot}")
        if anchor.get("global_step") != milestone:
            raise RuntimeError(
                f"permanent full snapshot step mismatch: path={milestone}, "
                f"payload={anchor.get('global_step')!r}"
            )
        if anchor.get(FORMAL_RUN_LINEAGE_KEY) != current_payload.get(
            FORMAL_RUN_LINEAGE_KEY
        ):
            raise RuntimeError(f"permanent full snapshot lineage mismatch: {snapshot}")
        if anchor.get("ablation_fp") != current_payload.get("ablation_fp"):
            raise RuntimeError(
                f"permanent full snapshot ablation fingerprint mismatch: {snapshot}"
            )
        validate_resume_checkpoint(
            anchor,
            model,
            rank=rank,
            is_dist=is_dist,
        )
        _ablation_mismatch_check(anchor["trainable_state"].keys(), model)
        _ablation_fingerprint_check(
            anchor.get("ablation_fp"), config, require_exact=True
        )
        _optimizer_state_integrity_check(
            anchor["optimizer"], anchor["trainable_state"], model, optimizer
        )
        _ema_state_integrity_check(
            anchor.get("ema"), anchor["trainable_state"], model
        )
        _scheduler_state_integrity_check(
            anchor.get("scheduler"), milestone, anchor["optimizer"]
        )
        del anchor


def _publish_periodic_snapshot(path, full_payload, config) -> str | None:
    """Idempotently publish/repair the configured snapshot for this payload."""
    if config is None:
        return None
    step = int(full_payload["global_step"])
    keep_every = int(config.trainer.get("checkpoint_keep_every", 0))
    keep_max = int(config.trainer.get("checkpoint_keep_max", 0))
    if keep_every <= 0 or keep_max <= 0 or step <= 0 or step % keep_every != 0:
        return None
    directory = os.path.dirname(path)
    snapshot = os.path.join(directory, f"step_{step}.pt")
    fmt = str(config.trainer.get("checkpoint_keep_format", "weights_ema"))
    if fmt not in {"full", "weights_ema"}:
        raise RuntimeError(f"unknown checkpoint_keep_format={fmt!r}")
    snapshots_before = sorted(
        (f for f in os.listdir(directory) if f.startswith("step_") and f.endswith(".pt")),
        key=lambda f: int(f[len("step_"):-len(".pt")]),
    )
    if fmt == "full":
        if len(snapshots_before) > keep_max:
            raise RuntimeError(
                f"full periodic snapshot retention already exceeds keep_max={keep_max}; "
                "permanent full anchors are never deleted"
            )
        if not os.path.lexists(snapshot) and len(snapshots_before) >= keep_max:
            raise RuntimeError(
                f"full periodic snapshot retention is full at keep_max={keep_max}; "
                f"refusing to delete a permanent anchor or publish {snapshot}"
            )
    if os.path.lexists(snapshot):
        if os.path.islink(snapshot) or not os.path.isfile(snapshot):
            raise RuntimeError(f"periodic snapshot target is not a regular file: {snapshot}")
        if fmt == "full":
            if not os.path.samefile(path, snapshot):
                raise RuntimeError(
                    "full periodic snapshot is not the same inode as validated last.pt; "
                    f"refusing stale or copied target: {snapshot}"
                )
            value = os.lstat(snapshot)
            if value.st_mode & 0o222:
                if stat.S_IMODE(value.st_mode) != 0o600 or value.st_nlink != 2:
                    raise RuntimeError(
                        "full periodic snapshot has an unexpected writable mode/link count: "
                        f"mode={oct(stat.S_IMODE(value.st_mode))}, nlink={value.st_nlink}"
                    )
                os.chmod(snapshot, 0o444)
                _fsync_regular_file(snapshot, context="repaired full periodic snapshot")
                _fsync_directory(directory)
        else:
            if os.lstat(snapshot).st_mode & 0o222:
                raise RuntimeError(f"periodic snapshot target is not read-only: {snapshot}")
            _validate_weights_ema_snapshot(snapshot, full_payload)
    else:
        if fmt == "full":
            os.link(path, snapshot)
            if not os.path.samefile(path, snapshot):
                raise RuntimeError(
                    f"full periodic snapshot hard-link verification failed: {snapshot}"
                )
            os.chmod(snapshot, 0o444)
            _fsync_regular_file(snapshot, context="full periodic snapshot")
            _fsync_directory(directory)
        elif fmt == "weights_ema":
            tmp = _write_exclusive_torch_tmp(
                {
                    "snapshot_schema": WEIGHTS_EMA_SNAPSHOT_SCHEMA,
                    "trainable_state": full_payload["trainable_state"],
                    "ema": full_payload["ema"],
                    "global_step": step,
                    "ablation_fp": full_payload["ablation_fp"],
                    FORMAL_RUN_LINEAGE_KEY: full_payload.get(FORMAL_RUN_LINEAGE_KEY),
                },
                snapshot,
                context="weights_ema periodic snapshot temporary",
            )
            os.replace(tmp, snapshot)
            os.chmod(snapshot, 0o444)
            _fsync_regular_file(snapshot, context="weights_ema periodic snapshot")
            _fsync_directory(directory)
            _validate_weights_ema_snapshot(snapshot, full_payload)

    snapshots = sorted(
        (f for f in os.listdir(directory) if f.startswith("step_") and f.endswith(".pt")),
        key=lambda f: int(f[len("step_"):-len(".pt")]),
    )
    if fmt == "weights_ema":
        removed = False
        for old in snapshots[:-keep_max]:
            os.remove(os.path.join(directory, old))
            removed = True
        if removed:
            _fsync_directory(directory)
    elif len(snapshots) > keep_max:
        raise RuntimeError(
            f"full periodic snapshot count {len(snapshots)} exceeds keep_max={keep_max}"
        )
    return snapshot


def save_checkpoint(path, model, optimizer, scheduler, global_step, is_dist, rank, config=None):
    formal_run_lineage = resolve_expected_run_lineage_sha256(is_dist=is_dist)
    trainable_names = {n for n, p in model.model.named_parameters() if p.requires_grad}
    if is_dist:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
        # 集合操作：所有 rank 都要调（gather sharded → full），仅 rank0 落盘。
        msd, osd = get_state_dict(
            model.model, optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        trainable = {n: v for n, v in msd.items() if n in trainable_names}
        opt_state = osd
    else:
        trainable = {n: p.detach().cpu() for n, p in model.model.named_parameters() if p.requires_grad}
        opt_state = optimizer.state_dict()
    # EMA gather（FSDP2 下 full_tensor 是 collective，须在 rank0 落盘前所有 rank 调）
    ema_state = model.ema_state_dict() if getattr(model, "_ema", None) else None
    # RNG is also an all-rank collective and must happen before nonzero ranks
    # return.  Only rank0 receives the envelope to serialize.
    rng_state = collect_rng_state(model, rank=rank, is_dist=is_dist)
    validation_error = None
    if rank == 0:
        try:
            _ablation_mismatch_check(trainable.keys(), model)
            _optimizer_state_integrity_check(opt_state, trainable, model, optimizer)
            _ema_state_integrity_check(ema_state, trainable, model)
            _scheduler_state_integrity_check(
                scheduler.state_dict(), global_step, opt_state
            )
        except BaseException as exc:
            validation_error = exc
    _raise_synchronized_validation_error(
        validation_error,
        is_dist=is_dist,
        context="checkpoint save payload",
    )
    write_error = None
    if rank == 0:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            directory = os.path.dirname(path)
            if os.path.lexists(path) and (
                os.path.islink(path) or not os.path.isfile(path)
            ):
                raise RuntimeError(f"checkpoint target is not a regular file: {path}")
            payload = {
                "checkpoint_schema": FULL_CHECKPOINT_SCHEMA,
                "trainable_state": trainable,
                "optimizer": opt_state,
                "ema": ema_state,
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "ablation_fp": _ablation_fingerprint(config) if config is not None else None,
                "rng_state": rng_state,
                FORMAL_RUN_LINEAGE_KEY: formal_run_lineage,
            }
            tmp = _write_exclusive_torch_tmp(
                payload, path, context="checkpoint temporary"
            )
            os.replace(tmp, path)
            _fsync_regular_file(path, context="checkpoint last.pt")
            _fsync_directory(directory)
            _publish_periodic_snapshot(path, payload, config)
        except BaseException as exc:
            write_error = exc
    # Nonzero ranks wait for durable publication instead of racing into the
    # next FSDP collective.  A rank0 disk error therefore fails every rank.
    _raise_synchronized_validation_error(
        write_error,
        is_dist=is_dist,
        context="checkpoint durable publication",
    )


def load_checkpoint(path, model, optimizer, scheduler, is_dist, config=None) -> tuple[int, dict]:
    """Load deterministic training state and return RNG for the late commit point.

    RNG is validated here but deliberately not restored yet: reconcile, WandB
    and other resume-only startup work runs afterwards and may consume global
    RNG.  train.py commits the returned state immediately before first batch.
    """
    expected_formal_run_lineage = resolve_expected_run_lineage_sha256(is_dist=is_dist)
    rank = torch.distributed.get_rank() if is_dist else 0
    ckpt = None
    pending_rng_state = None
    validation_error = None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        pending_rng_state = validate_resume_checkpoint(
            ckpt,
            model,
            rank=rank,
            is_dist=is_dist,
        )
        saved_formal_run_lineage = ckpt.get(FORMAL_RUN_LINEAGE_KEY)
        if saved_formal_run_lineage is not None:
            normalize_expected_run_lineage_sha256(saved_formal_run_lineage)
        # Symmetric binding: a formal checkpoint cannot be resumed by merely
        # omitting the expected lineage environment.  Non-formal compatibility
        # is the None/None case (including pre-lineage v2 checkpoints).
        if saved_formal_run_lineage != expected_formal_run_lineage:
            raise RuntimeError(
                "formal run lineage mismatch: "
                f"checkpoint={saved_formal_run_lineage!r}, "
                f"expected={expected_formal_run_lineage!r}"
            )
        _ablation_mismatch_check(ckpt["trainable_state"].keys(), model)
        if config is not None:
            if ckpt.get("ablation_fp") is None:
                raise RuntimeError("v2 resume checkpoint lacks required ablation_fp")
            _ablation_fingerprint_check(
                ckpt.get("ablation_fp"), config, require_exact=True
            )
        _optimizer_state_integrity_check(
            ckpt["optimizer"], ckpt["trainable_state"], model, optimizer
        )
        _ema_state_integrity_check(ckpt.get("ema"), ckpt["trainable_state"], model)
        _scheduler_state_integrity_check(
            ckpt.get("scheduler"), ckpt["global_step"], ckpt["optimizer"]
        )
        if isinstance(ckpt["global_step"], bool) or not isinstance(ckpt["global_step"], int):
            raise RuntimeError("checkpoint global_step must be int")
        if ckpt["global_step"] < 0:
            raise RuntimeError(f"checkpoint global_step must be non-negative: {ckpt['global_step']}")
        if rank == 0:
            _validate_historical_full_snapshots(
                path,
                ckpt,
                config,
                model,
                optimizer,
                rank=rank,
                is_dist=is_dist,
            )
    except BaseException as exc:
        validation_error = exc
    _raise_synchronized_validation_error(
        validation_error,
        is_dist=is_dist,
        context="checkpoint resume pre-load",
    )
    assert ckpt is not None and pending_rng_state is not None
    repair_error = None
    if rank == 0:
        try:
            repaired = _publish_periodic_snapshot(path, ckpt, config)
            if repaired is not None:
                print(
                    f"[checkpoint] periodic snapshot present/repaired: {repaired}",
                    flush=True,
                )
        except BaseException as exc:
            repair_error = exc
    _raise_synchronized_validation_error(
        repair_error,
        is_dist=is_dist,
        context="checkpoint periodic snapshot startup repair",
    )
    if is_dist:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions, set_model_state_dict, set_optimizer_state_dict)
        opts = StateDictOptions(full_state_dict=True, strict=False)
        set_model_state_dict(model.model, ckpt["trainable_state"], options=opts)
        set_optimizer_state_dict(model.model, optimizer, optim_state_dict=ckpt["optimizer"], options=opts)
    else:
        model.model.load_state_dict(ckpt["trainable_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
    post_error = None
    try:
        if getattr(model, "_ema", None) is not None:
            model.load_ema_state_dict(ckpt.get("ema"))
        scheduler.load_state_dict(ckpt["scheduler"])
        _runtime_optimizer_integrity_check(model, optimizer)
        _runtime_ema_integrity_check(model)
        _runtime_scheduler_integrity_check(scheduler, optimizer, ckpt["global_step"])
    except BaseException as exc:
        post_error = exc
    _raise_synchronized_validation_error(
        post_error,
        is_dist=is_dist,
        context="checkpoint resume post-load",
    )
    return int(ckpt["global_step"]), pending_rng_state


def init_from_checkpoint(path, model, optimizer, is_dist, config=None, load_optimizer=True) -> str:
    """跨分辨率【阶段切换】初始化（如 256→512a）：从上一阶段的 final ckpt 载入【可训练权重
    (+ 可选 optimizer 动量 + EMA)】，但**不读 global_step、不载 scheduler** —— 让新阶段用自己的
    scheduler 从 step0 重新 warmup（设计 §6：继承 optimizer + const LR + 换分辨率加几百步 re-warmup）。

    与 load_checkpoint（同分辨率续训/崩溃恢复，完整恢复 step+scheduler，逐字不变）**严格区分**：
    本函数由 train.py 在 NOT(本地 out_dir/last.pt 存在 AND resume) 时调用 —— 即「本地无 last.pt（首次启动新阶段，
    常态）」或「本地有 last.pt 但 resume=false（显式弃进度、从上阶段重来；train.py 此时会大声告警）」。
    resume=true 时本地 last.pt 优先走 load_checkpoint，新阶段 retry 不会被本函数重置进度。

    指纹/参数名校验与 resume 一致：arch/消融不匹配仍硬报错（如把 B1 ckpt 喂给 text_adapter.enabled=false
    的 config）；分辨率不在指纹内（_ablation_fingerprint 只含 arch 开关），故 256↔512 同 arch 通过。
    optimizer 动量维度只与参数 shape 绑定（256→512 可训参数 shape 不变，仅 latent token 数变），可安全继承。
    **只继承动量(state)、不继承超参**（param_groups 的 lr/betas/eps/wd 恢复为本阶段值，防首步绕过 re-warmup）；
    缺 ablation_fp 默认硬报错（除非 config.trainer.allow_unsafe_init_from_no_fp=true）。
    """
    validation_error = None
    rng_policy = None
    ckpt = None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        _ablation_mismatch_check(ckpt["trainable_state"].keys(), model)
        if config is None:
            raise RuntimeError("init_from requires config for architecture and RNG policy checks")
        rng_policy = validate_init_from_rng_policy(ckpt, config)
        # init_from 比 resume 更严：缺 ablation_fp 默认【硬报错】。跨 run 初始化比同 run resume 更易指错，
        # 且 mask-only 差异(如 text_adapter.bidirectional)不改参数名、_ablation_mismatch_check 挡不住 → 会静默错初始化。
        saved_fp = ckpt.get("ablation_fp")
        if saved_fp is None and not bool(config.trainer.get("allow_unsafe_init_from_no_fp", False)):
            raise RuntimeError(
                "init_from 的 checkpoint 无 ablation_fp（旧版/weights-only/外部转换 ckpt）。init_from 跨 run，"
                "mask-only 开关(如 text_adapter.bidirectional)不改参数名 → 可能静默错初始化。"
                "确认 arch 与本 run 完全一致后，显式设 trainer.allow_unsafe_init_from_no_fp=true 跳过本检查。")
        _ablation_fingerprint_check(
            saved_fp, config, require_exact=True
        )  # audited A-final carries the complete current fingerprint closure
        if load_optimizer:
            _optimizer_state_integrity_check(
                ckpt["optimizer"], ckpt["trainable_state"], model, optimizer
            )
        _ema_state_integrity_check(ckpt.get("ema"), ckpt["trainable_state"], model)
    except BaseException as exc:
        validation_error = exc
    _raise_synchronized_validation_error(
        validation_error,
        is_dist=is_dist,
        context="init_from pre-load",
    )
    assert ckpt is not None and rng_policy is not None
    # 继承 optimizer 时【只继承动量(state)，不继承超参(param_groups 的 lr/betas/eps/wd)】：
    # snapshot 当前(本阶段 config/scheduler 设定)的超参，load 后恢复 —— 否则首步 optimizer.step() 会用上一阶段
    # ckpt 的满 lr（绕过本阶段 re-warmup）并沿用旧 betas/wd（审计 P0，已 toy 实证）。仅改 param_groups 标量，不碰 sharded state。
    saved_hp = ([{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]
                if load_optimizer else None)
    if is_dist:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions, set_model_state_dict, set_optimizer_state_dict)
        opts = StateDictOptions(full_state_dict=True, strict=False)
        set_model_state_dict(model.model, ckpt["trainable_state"], options=opts)
        if load_optimizer:
            set_optimizer_state_dict(model.model, optimizer, optim_state_dict=ckpt["optimizer"], options=opts)
    else:
        model.model.load_state_dict(ckpt["trainable_state"], strict=False)
        if load_optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
    post_error = None
    try:
        if load_optimizer:                                 # 恢复本阶段超参（动量已从 ckpt 继承）
            if len(optimizer.param_groups) != len(saved_hp):
                raise RuntimeError(
                    "optimizer param-group count changed during init_from load: "
                    f"before={len(saved_hp)}, after={len(optimizer.param_groups)}"
                )
            for g, hp in zip(optimizer.param_groups, saved_hp):
                g.update(hp)
        if getattr(model, "_ema", None) is not None:
            model.load_ema_state_dict(ckpt.get("ema"))
        # Deliberately do not load source scheduler/global_step: this stage has
        # step=0 and a fresh warmup scheduler.
        if load_optimizer:
            _runtime_optimizer_integrity_check(model, optimizer)
        _runtime_ema_integrity_check(model)
    except BaseException as exc:
        post_error = exc
    _raise_synchronized_validation_error(
        post_error,
        is_dist=is_dist,
        context="init_from post-load",
    )
    # Source RNG is never restored across a stage boundary.
    return rng_policy
