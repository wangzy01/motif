"""checkpoint / lr scheduler / 消融指纹（迁自 train.py，纯搬家，行为逐字不变）。

本模块集中训练入口的"状态持久化与一致性校验"逻辑，使 train.py 回到干净状态。
**🔴硬约束：checkpoint 序列化结构、指纹计算方式、resume 加载逻辑必须逐字不变**——
直播任务（v1/B1/ab1/ab2/ab3）写出的 checkpoint 将来要能被这份重构后的代码原样 resume。
绝不改 state_dict 结构、键名、指纹哈希算法、DTensor gather/scatter 流程；仅做搬家 + import。

含四类：
  * build_scheduler                : LambdaLR（warmup + constant/cosine）构建
  * _ablation_fingerprint          : 架构/消融开关指纹（F2，mask-only 切换也能挡）
  * _ablation_fingerprint_check    : resume 时指纹比对（旧 ckpt 无指纹则 warn 跳过）
  * _ablation_mismatch_check       : resume 时可训练参数名集合比对
  * save_checkpoint / load_checkpoint : 单进程 + FSDP2（DCP full_state_dict gather）双模

train.py / scripts / experiments 仍以 `from train import ...` 调用：train.py 通过本模块的
import 把这些名字再导出，调用点解析不变。
"""

import math
import os

import torch


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


def _ablation_fingerprint_check(saved_fp, config):
    """resume 时比对架构/消融指纹；任何不一致（含 mask-only 切换）→ 清晰报错。
    旧 ckpt 无指纹时跳过（向后兼容，仍由 `_ablation_mismatch_check` 兜参数名）。"""
    if saved_fp is None:
        print("[train] ⚠️ 该 checkpoint 无消融指纹（旧版），跳过指纹校验；仅靠参数名检查。"
              "mask-only 开关（如 text_adapter.bidirectional）切换可能未被挡住，请自行确认。", flush=True)
        return
    cur = _ablation_fingerprint(config)
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
    """切换消融/架构开关后可训练参数集会变，旧 ckpt 无法 resume —— 给清晰报错。"""
    cur_keys = {n for n, p in model.model.named_parameters() if p.requires_grad}
    saved = set(saved_keys)
    ablation = ("_moe_gen", "time_embed", "lora_und")
    unexpected = [k for k in saved - cur_keys if any(a in k for a in ablation)]
    missing = [k for k in cur_keys - saved if any(a in k for a in ablation)]
    if unexpected or missing:
        raise RuntimeError(
            "checkpoint 与当前 config 的消融/架构开关不一致，无法 resume（model 与 optimizer 参数集都对不上）。"
            f"\n  checkpoint 多出: {unexpected[:4]}\n  当前模型多出: {missing[:4]}"
            "\n  请用与该 checkpoint 一致的开关续训，或换新的 checkpoint_dir 从头训。"
        )
    extra = [k for k in saved - cur_keys if k not in unexpected]
    assert not extra, f"checkpoint 有未知参数（非消融）: {extra[:5]}"


def save_checkpoint(path, model, optimizer, scheduler, global_step, is_dist, rank, config=None):
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
    if rank != 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(
        {"trainable_state": trainable, "optimizer": opt_state, "ema": ema_state,
         "scheduler": scheduler.state_dict(), "global_step": global_step,
         "ablation_fp": _ablation_fingerprint(config) if config is not None else None},
        tmp,
    )
    os.replace(tmp, path)
    # —— 周期性步号快照保留（默认安全：未设 keep_every/keep_max 则不启用，行为与旧版完全一致）——
    keep_every = int(config.trainer.get("checkpoint_keep_every", 0)) if config is not None else 0
    keep_max = int(config.trainer.get("checkpoint_keep_max", 0)) if config is not None else 0
    if keep_every > 0 and keep_max > 0 and global_step > 0 and global_step % keep_every == 0:
        d = os.path.dirname(path)
        snap = os.path.join(d, f"step_{global_step}.pt")
        fmt = str(config.trainer.get("checkpoint_keep_format", "weights_ema")) if config is not None else "weights_ema"
        if not os.path.exists(snap):
            if fmt == "full":
                os.link(path, snap)                       # 硬链接当前 last.pt（full 可续训；瞬时零额外空间，last.pt 之后被覆盖也不影响快照）
            else:                                          # weights_ema：仅权重+EMA（无 optimizer，约半大小），供 eval/出图/回滚；最新可续训副本始终是 last.pt
                stmp = snap + ".tmp"
                torch.save(
                    {"trainable_state": trainable, "ema": ema_state, "global_step": global_step,
                     "ablation_fp": _ablation_fingerprint(config) if config is not None else None},
                    stmp,
                )
                os.replace(stmp, snap)
        snaps = sorted(
            (f for f in os.listdir(d) if f.startswith("step_") and f.endswith(".pt")),
            key=lambda f: int(f[len("step_"):-len(".pt")]),
        )
        for old in snaps[:-keep_max]:                     # 删最旧，仅保留最近 keep_max 个步号快照
            os.remove(os.path.join(d, old))


def load_checkpoint(path, model, optimizer, scheduler, is_dist, config=None) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    _ablation_mismatch_check(ckpt["trainable_state"].keys(), model)
    if config is not None:                                 # F2：架构/消融指纹比对（挡 mask-only 切换）
        _ablation_fingerprint_check(ckpt.get("ablation_fp"), config)
    if is_dist:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions, set_model_state_dict, set_optimizer_state_dict)
        opts = StateDictOptions(full_state_dict=True, strict=False)
        set_model_state_dict(model.model, ckpt["trainable_state"], options=opts)
        set_optimizer_state_dict(model.model, optimizer, optim_state_dict=ckpt["optimizer"], options=opts)
    else:
        model.model.load_state_dict(ckpt["trainable_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
    if getattr(model, "_ema", None) is not None:
        model.load_ema_state_dict(ckpt.get("ema"))
    scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["global_step"])


def init_from_checkpoint(path, model, optimizer, is_dist, config=None, load_optimizer=True) -> None:
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
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    _ablation_mismatch_check(ckpt["trainable_state"].keys(), model)
    if config is not None:
        # init_from 比 resume 更严：缺 ablation_fp 默认【硬报错】。跨 run 初始化比同 run resume 更易指错，
        # 且 mask-only 差异(如 text_adapter.bidirectional)不改参数名、_ablation_mismatch_check 挡不住 → 会静默错初始化。
        saved_fp = ckpt.get("ablation_fp")
        if saved_fp is None and not bool(config.trainer.get("allow_unsafe_init_from_no_fp", False)):
            raise RuntimeError(
                "init_from 的 checkpoint 无 ablation_fp（旧版/weights-only/外部转换 ckpt）。init_from 跨 run，"
                "mask-only 开关(如 text_adapter.bidirectional)不改参数名 → 可能静默错初始化。"
                "确认 arch 与本 run 完全一致后，显式设 trainer.allow_unsafe_init_from_no_fp=true 跳过本检查。")
        _ablation_fingerprint_check(saved_fp, config)      # 有指纹则照常硬校验（不一致 raise）
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
    if load_optimizer:                                     # 恢复本阶段超参（动量已从 ckpt 继承）
        for g, hp in zip(optimizer.param_groups, saved_hp):
            g.update(hp)
    if getattr(model, "_ema", None) is not None and ckpt.get("ema") is not None:
        model.load_ema_state_dict(ckpt.get("ema"))
    # 故意不调用 scheduler.load_state_dict、不返回 global_step：新阶段 step 从 0 起、scheduler 全新 re-warmup。
