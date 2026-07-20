#!/usr/bin/env python3
"""Moment-backed GPU postflight for the fresh 7.18 Stage256 smoke-v3 checkpoint.

The v2 three-step smoke proved that exact parameter-versus-EMA inequality is a
bad activation proxy for unit-scale RMSNorm weights behind two zero-init
openers.  This verifier retains every architecture, checkpoint, EMA and finite
state gate from v2, binds the independent v2 CPU-forensics receipt, and proves
the short-run learning paths with Adam moment state where an exact parameter
delta is not numerically required.

Evidence outputs are create-once.  Diagnostic, full-state and forward-probe
reports may be written with ``status=FAIL`` for forensics, but ``--output`` is
created only when every gate passes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


EXPECTED = {
    "architecture_sha256": "d6db3855f1bd561ee21f31238b559dbff7adc65149a32025785cc259863d9abf",
    "ablation_sha256": "70546507a768a9d01705b8118b6e3cfea3f055f22db5e9bcdfb3e25e7d022642",
    "config_sha256": "70c8b4cdadfd17af29e087a9718958bc3e5655feff7afaf8d48ed7bd32bef00a",
    "eval_helper_sha256": "8d106c39914083a87b0a372b56c2bf41477ee0279ea0f51e5daa844a38cd736a",
    "forensics_receipt_sha256": "391d9fa31a99a9176367c65226f3966ba031c12b8e9262b6c9e8d1603707a994",
    "forensics_checkpoint_sha256": "010f754fabc59734e459b6224af692675c9562798d7d69f45097301a24a23480",
    "trainable_count": 483,
    "trainable_keyset_sha256": "6c61b43d30a8bde2f13fc8a91e30a8a0de271934ba5c3fd72df392cd027a4424",
    "frozen_missing_count": 310,
    "frozen_missing_keyset_sha256": "d380e45c9d5d53c7f9629c612b8a2adcd0bc075854931c583fe1570683511378",
    "parameter_counts_millions": [3366.0, 1645.5, 1720.6],
}


GROUP_PREDICATES: dict[str, Callable[[str], bool]] = {
    "outer_proj_out": lambda key: key in {"proj_out.weight", "proj_out.bias"},
    "adapter_out_proj": lambda key: "text_adapter_moe_gen.out_proj" in key,
    "adapter_blocks": lambda key: "text_adapter_moe_gen.blocks" in key,
    "adapter_out_norm": lambda key: "text_adapter_moe_gen.out_norm" in key,
    "mlf_gate": lambda key: key.endswith("multi_layer_fusion_moe_gen.gate"),
    "mlf_logits": lambda key: key.endswith("multi_layer_fusion_moe_gen.logits"),
    "text_kv": lambda key: (
        "text_k_proj_moe_gen" in key or "text_v_proj_moe_gen" in key
    ),
    "text_k_norm": lambda key: "text_k_norm_moe_gen" in key,
}

GROUP_COUNTS = {
    "outer_proj_out": 2,
    "adapter_out_proj": 1,
    "adapter_blocks": 22,
    "adapter_out_norm": 1,
    "mlf_gate": 1,
    "mlf_logits": 1,
    "text_kv": 56,
    "text_k_norm": 28,
}

DELTA_ALL_GROUPS = {
    "outer_proj_out",
    "adapter_out_proj",
    "mlf_gate",
    "mlf_logits",
    "text_kv",
}

ADAPTER_NORM_SUFFIXES = (
    ".ln1.weight",
    ".ln2.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def keyset_sha256(keys: set[str] | list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(keys)) + "\n").encode()).hexdigest()


def regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return value


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("stage256_two_final_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: dict[str, Any]) -> str:
    if not path.is_absolute():
        raise ValueError(f"output path must be absolute: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing smoke-v3 postflight overwrite: {path}")
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing smoke-v3 postflight overwrite: {path}")
    os.link(temporary, path)
    os.unlink(temporary)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(raw).hexdigest()


def stable_stat(path: Path) -> dict[str, int]:
    if path.is_symlink():
        raise RuntimeError(f"checkpoint must not be a symlink: {path}")
    value = path.stat()
    if not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise RuntimeError(f"checkpoint must be a nonempty regular file: {path}")
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def tensor_metrics(value: Any) -> dict[str, Any]:
    import torch

    if not torch.is_tensor(value):
        return {"is_tensor": False, "finite": False}
    finite = bool(torch.isfinite(value).all())
    work = value.detach().float()
    result = {
        "is_tensor": True,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": finite,
        "elements": int(value.numel()),
        "nonzero_elements": int(torch.count_nonzero(work).item()),
        "l2": float(work.norm().item()) if finite else None,
        "max_abs": (
            float(work.abs().max().item())
            if finite and work.numel()
            else (0.0 if finite else None)
        ),
    }
    del work
    return result


def delta_metrics(current: Any, average: Any) -> dict[str, Any]:
    import torch

    if not torch.is_tensor(current) or not torch.is_tensor(average):
        return {"available": False, "shape_dtype_match": False, "finite": False}
    shape_dtype_match = current.shape == average.shape and current.dtype == average.dtype
    if not shape_dtype_match:
        return {"available": True, "shape_dtype_match": False, "finite": False}
    delta = current.detach().float() - average.detach().float()
    finite = bool(torch.isfinite(delta).all())
    result = {
        "available": True,
        "shape_dtype_match": True,
        "finite": finite,
        "elements": int(delta.numel()),
        "changed_elements": int(torch.count_nonzero(delta).item()),
        "l2": float(delta.norm().item()) if finite else None,
        "max_abs": (
            float(delta.abs().max().item())
            if finite and delta.numel()
            else (0.0 if finite else None)
        ),
    }
    del delta
    return result


def exact_optimizer_step(value: Any, expected: int) -> tuple[bool, int | None]:
    import torch

    if torch.is_tensor(value):
        if value.numel() != 1 or not bool(torch.isfinite(value).all()):
            return False, None
        numeric = float(value.item())
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return False, None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return False, None
    integer = int(numeric)
    return integer == expected, integer


def validate_forensics_receipt(path: Path) -> dict[str, Any]:
    receipt_sha = sha256_file(path)
    if receipt_sha != EXPECTED["forensics_receipt_sha256"]:
        raise RuntimeError(
            f"activation-forensics receipt SHA drift: {receipt_sha}"
        )
    value = load_json(path, "activation-forensics receipt")
    diagnosis = value.get("diagnosis")
    structural = value.get("structural_identity")
    checkpoint = value.get("checkpoint")
    if not isinstance(diagnosis, dict) or not isinstance(structural, dict) or not isinstance(checkpoint, dict):
        raise RuntimeError("activation-forensics receipt lacks required evidence objects")
    required_true = (
        "all_14_adapter_dense_parameters_changed",
        "exact_norm_only_parameter_gap",
        "all_adapter_block_optimizer_moments_nonzero",
        "all_reported_group_optimizer_moments_nonzero",
        "all_zero_init_openers_changed",
        "structural_identity_pass",
        "old_parameter_delta_proxy_false_negative_candidate",
    )
    if (
        value.get("schema_version") != "stage256_20260718_smoke_v2_activation_forensics_v1"
        or value.get("status") != "PASS"
        or value.get("evidence_only_not_smoke_success") is not True
        or int(value.get("global_step", -1)) != 3
        or int(value.get("trainable_count", -1)) != EXPECTED["trainable_count"]
        or int(value.get("optimizer_state_count", -1)) != EXPECTED["trainable_count"]
        or value.get("keyset_sha256") != EXPECTED["trainable_keyset_sha256"]
        or value.get("ablation_fingerprint_sha256") != EXPECTED["ablation_sha256"]
        or checkpoint.get("sha256") != EXPECTED["forensics_checkpoint_sha256"]
        or diagnosis.get("recommended_action") != "REVIEW_MOMENT_BACKED_V3_GATE"
        or not all(diagnosis.get(key) is True for key in required_true)
    ):
        raise RuntimeError("activation-forensics receipt does not authorize moment-backed v3")
    if not all(value is True for value in structural.values() if isinstance(value, bool)):
        raise RuntimeError("activation-forensics structural identity contains a failed boolean")
    return {
        "path": str(path),
        "sha256": receipt_sha,
        "schema_version": value["schema_version"],
        "source_checkpoint_sha256": checkpoint["sha256"],
        "diagnosis": {key: diagnosis[key] for key in (*required_true, "recommended_action")},
    }


def run_float32_forward_probe(
    transformer: Any,
    *,
    weights_src: str,
    require_nonzero_branch: bool,
) -> dict[str, Any]:
    """Run the actual checkpoint modules and inspect pre-residual branches.

    ``transformer`` is the real CUDA model built by the frozen eval helper.  It
    is first probed with EMA weights and then, after a strict trainable-state
    load, with the v3 trainable weights.  The branch tensors are inspected
    directly; subtracting a tiny branch from a large residual is never a gate.
    """
    import torch

    adapter = getattr(transformer, "text_adapter_moe_gen", None)
    fusion = getattr(transformer, "multi_layer_fusion_moe_gen", None)
    if adapter is None or fusion is None:
        raise RuntimeError("actual checkpoint model lacks adapter or MLF module")
    parameters = list(adapter.parameters()) + list(fusion.parameters())
    if not parameters:
        raise RuntimeError("actual checkpoint adapter/MLF modules have no parameters")
    device = parameters[0].device
    source_parameter_dtypes = sorted({str(parameter.dtype) for parameter in parameters})
    if any(parameter.device != device for parameter in parameters):
        raise RuntimeError("actual checkpoint adapter/MLF parameters span devices")

    # The v3 contract is explicitly about the FP32 branch before residual
    # addition.  Cast the *actual loaded modules* in place; loading the
    # trainable state after the EMA probe will then cast those checkpoint
    # tensors into the same FP32 destination modules.
    adapter.float().eval()
    fusion.float().eval()
    execution_parameter_dtypes = sorted({
        str(parameter.dtype)
        for parameter in list(adapter.parameters()) + list(fusion.parameters())
    })

    block_keys = sorted(
        name for name, _parameter in adapter.named_parameters()
        if name.startswith("blocks.")
    )
    causal_structure = (
        len(adapter.blocks) == 2
        and all(bool(getattr(block, "causal", False)) for block in adapter.blocks)
    )
    hidden_size = int(adapter.out_proj.in_features)
    sequence = 7
    split = 4

    rng_devices = []
    if device.type == "cuda":
        rng_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(20260718)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(20260718)
        hidden = torch.randn(
            1, sequence, hidden_size, device=device, dtype=torch.float32
        )
        changed_future = hidden.clone()
        changed_future[:, split:] += 3.0 * torch.randn_like(changed_future[:, split:])
        mask = torch.ones(1, sequence, device=device, dtype=torch.bool)

        hook_rows: list[dict[str, Any]] = []
        handles = []
        for index, block in enumerate(adapter.blocks):
            def hook(_module, _inputs, output, *, block_index=index):
                hook_rows.append({
                    "block": block_index,
                    "shape": list(output.shape),
                    "dtype": str(output.dtype),
                    "finite": bool(torch.isfinite(output).all()),
                })
            handles.append(block.register_forward_hook(hook))

        def adapter_branch(value):
            branch_hidden = value
            for branch_block in adapter.blocks:
                branch_hidden = branch_block(branch_hidden, key_padding_mask=mask)
            return adapter.out_proj(adapter.out_norm(branch_hidden))

        with torch.no_grad():
            try:
                base_branch = adapter_branch(hidden)
            finally:
                for handle in handles:
                    handle.remove()
            future_branch = adapter_branch(changed_future)
            prefix_max_delta = float(
                (base_branch[:, :split] - future_branch[:, :split]).abs().max().item()
            )
            future_max_delta = float(
                (base_branch[:, split:] - future_branch[:, split:]).abs().max().item()
            )

            layer_count = int(getattr(fusion, "num_layers", 0))
            fusion_hidden = [
                torch.randn(
                    1, sequence, hidden_size, device=device, dtype=torch.float32
                )
                for _ in range(layer_count)
            ]
            fusion_last = fusion_hidden[-1]
            fusion_weights = torch.softmax(fusion.logits.float(), dim=0)
            elmo = sum(
                fusion_weights[index] * fusion_hidden[index]
                for index in range(layer_count)
            )
            fusion_branch = fusion.gate.float() * (elmo - fusion_last)

    adapter_branch_nonzero = int(torch.count_nonzero(base_branch).item()) > 0
    fusion_branch_nonzero = int(torch.count_nonzero(fusion_branch).item()) > 0
    checks = {
        "actual_checkpoint_modules": True,
        "float32_execution": (
            base_branch.dtype == future_branch.dtype == fusion_branch.dtype == torch.float32
            and execution_parameter_dtypes == ["torch.float32"]
        ),
        "causal_two_block_structure": causal_structure,
        "adapter_block_parameter_count_22": len(block_keys) == 22,
        "two_block_hooks_finite_shape": (
            len(hook_rows) == 2
            and [row["block"] for row in hook_rows] == [0, 1]
            and all(
                row["shape"] == [1, sequence, hidden_size]
                and row["dtype"] == "torch.float32"
                and row["finite"]
                for row in hook_rows
            )
        ),
        "adapter_branch_finite_shape": (
            list(base_branch.shape) == [1, sequence, hidden_size]
            and bool(torch.isfinite(base_branch).all())
        ),
        "mlf_12_layer_structure": layer_count == 12,
        "mlf_branch_finite_shape": (
            list(fusion_branch.shape) == [1, sequence, hidden_size]
            and bool(torch.isfinite(fusion_branch).all())
        ),
        "causal_prefix_independent_of_future": prefix_max_delta <= 1.0e-5,
        "trainable_adapter_branch_nonzero": (
            adapter_branch_nonzero if require_nonzero_branch else True
        ),
        "trainable_mlf_branch_nonzero": (
            fusion_branch_nonzero if require_nonzero_branch else True
        ),
        "trainable_future_branch_sensitive": (
            math.isfinite(future_max_delta) and future_max_delta > 0.0
            if require_nonzero_branch
            else True
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "weights_src": weights_src,
        "nonzero_branch_required": require_nonzero_branch,
        "device": str(device),
        "source_parameter_dtypes": source_parameter_dtypes,
        "execution_parameter_dtypes": execution_parameter_dtypes,
        "hidden_size": hidden_size,
        "checks": checks,
        "adapter": {
            "block_parameter_keys": block_keys,
            "hook_rows": hook_rows,
            "branch_shape": list(base_branch.shape),
            "branch_dtype": str(base_branch.dtype),
            "branch_finite": bool(torch.isfinite(base_branch).all()),
            "branch_nonzero_elements": int(torch.count_nonzero(base_branch).item()),
            "branch_l2": float(base_branch.float().norm().item()),
            "prefix_max_delta": prefix_max_delta,
            "future_max_delta": future_max_delta,
            "future_branch_sensitive": future_max_delta > 0.0,
        },
        "mlf": {
            "layer_count": layer_count,
            "branch_shape": list(fusion_branch.shape),
            "branch_dtype": str(fusion_branch.dtype),
            "branch_finite": bool(torch.isfinite(fusion_branch).all()),
            "branch_nonzero_elements": int(torch.count_nonzero(fusion_branch).item()),
            "branch_l2": float(fusion_branch.float().norm().item()),
        },
        "method": (
            "Actual checkpoint modules; pre-residual branches only: "
            "adapter.out_proj(adapter.out_norm(blocks(h))) and "
            "gate*(elmo-h_last). EMA requires finite shape/causal execution but "
            "not nonzero effect; trainable_state requires both branches nonzero."
        ),
    }


def collect_activation_diagnostics(
    trainable_state: dict[str, Any],
    ema: dict[str, Any],
    optimizer: dict[str, Any],
    scheduler: Any,
    expected_step: int,
) -> dict[str, Any]:
    import torch

    opt_state = optimizer.get("state") if isinstance(optimizer, dict) else None
    if not isinstance(opt_state, dict):
        opt_state = {}
    state_keys, ema_keys, opt_keys = set(trainable_state), set(ema), set(opt_state)
    union_keys = sorted(state_keys | ema_keys | opt_keys)
    structural = {
        "state_count": len(state_keys),
        "ema_count": len(ema_keys),
        "optimizer_state_count": len(opt_keys),
        "state_ema_keyset_equal": state_keys == ema_keys,
        "state_optimizer_keyset_equal": state_keys == opt_keys,
        "trainable_keyset_sha256": keyset_sha256(state_keys),
        "expected_trainable_count": len(state_keys) == EXPECTED["trainable_count"],
        "expected_trainable_keyset": keyset_sha256(state_keys) == EXPECTED["trainable_keyset_sha256"],
        "state_missing_from_ema": sorted(state_keys - ema_keys),
        "state_missing_from_optimizer": sorted(state_keys - opt_keys),
        "ema_extra_vs_state": sorted(ema_keys - state_keys),
        "optimizer_extra_vs_state": sorted(opt_keys - state_keys),
    }
    param_groups = optimizer.get("param_groups") if isinstance(optimizer, dict) else None
    param_group_rows = []
    if isinstance(param_groups, list):
        for raw_group in param_groups:
            group = raw_group if isinstance(raw_group, dict) else {}
            names = [str(value) for value in group.get("params", [])]
            try:
                betas = [float(value) for value in group.get("betas", ())]
                numeric = {
                    "weight_decay": float(group.get("weight_decay")),
                    "lr": float(group.get("lr")),
                    "initial_lr": float(group.get("initial_lr")),
                    "eps": float(group.get("eps")),
                }
                numeric_finite = all(math.isfinite(value) for value in (*numeric.values(), *betas))
            except (TypeError, ValueError, OverflowError):
                betas, numeric, numeric_finite = [], {}, False
            param_group_rows.append({
                "parameter_count": len(names),
                "parameter_keyset_sha256": keyset_sha256(names),
                "betas": betas,
                "numeric_finite": numeric_finite,
                **numeric,
            })
    optimizer_hparams_pass = (
        len(param_group_rows) == 1
        and param_group_rows[0].get("numeric_finite") is True
        and param_group_rows[0].get("parameter_count") == EXPECTED["trainable_count"]
        and param_group_rows[0].get("parameter_keyset_sha256") == EXPECTED["trainable_keyset_sha256"]
        and param_group_rows[0].get("weight_decay") == 0.0
        and param_group_rows[0].get("betas") == [0.9, 0.95]
        and param_group_rows[0].get("eps") == 1.0e-8
        and param_group_rows[0].get("initial_lr") == 2.121e-4
        and param_group_rows[0].get("lr") == 8.484e-7
    )
    scheduler_evidence = {
        "is_dict": isinstance(scheduler, dict),
        "last_epoch": scheduler.get("last_epoch") if isinstance(scheduler, dict) else None,
        "base_lrs": scheduler.get("base_lrs") if isinstance(scheduler, dict) else None,
        "last_lrs": scheduler.get("_last_lr") if isinstance(scheduler, dict) else None,
    }
    try:
        scheduler_pass = (
            isinstance(scheduler, dict)
            and int(scheduler.get("last_epoch", -1)) == expected_step
            and [float(value) for value in scheduler.get("base_lrs", [])] == [2.121e-4]
            and [float(value) for value in scheduler.get("_last_lr", [])] == [8.484e-7]
        )
    except (TypeError, ValueError, OverflowError):
        scheduler_pass = False
    structural["optimizer_param_groups"] = param_group_rows
    structural["expected_optimizer_hparams"] = optimizer_hparams_pass
    structural["scheduler"] = scheduler_evidence
    structural["expected_scheduler_state"] = scheduler_pass
    groups: dict[str, Any] = {}
    for group, predicate in GROUP_PREDICATES.items():
        selected = [key for key in union_keys if predicate(key)]
        rows = []
        for key in selected:
            current = trainable_state.get(key)
            average = ema.get(key)
            record = opt_state.get(key)
            parameter = tensor_metrics(current)
            average_metrics = tensor_metrics(average)
            delta = delta_metrics(current, average)
            record_valid = isinstance(record, dict) and {
                "step", "exp_avg", "exp_avg_sq"
            }.issubset(record)
            first = record.get("exp_avg") if record_valid else None
            second = record.get("exp_avg_sq") if record_valid else None
            first_metrics = tensor_metrics(first)
            second_metrics = tensor_metrics(second)
            shape_dtype_match = (
                torch.is_tensor(current)
                and torch.is_tensor(first)
                and torch.is_tensor(second)
                and current.shape == first.shape == second.shape
                and current.dtype == first.dtype == second.dtype
            )
            step_ok, step_value = exact_optimizer_step(
                record.get("step") if record_valid else None, expected_step
            )
            moments_nonzero = (
                record_valid
                and shape_dtype_match
                and first_metrics.get("finite") is True
                and second_metrics.get("finite") is True
                and first_metrics.get("nonzero_elements", 0) > 0
                and second_metrics.get("nonzero_elements", 0) > 0
            )
            base_pass = (
                parameter.get("finite") is True
                and average_metrics.get("finite") is True
                and delta.get("finite") is True
                and delta.get("shape_dtype_match") is True
                and step_ok
                and moments_nonzero
            )
            rows.append({
                "key": key,
                "parameter": parameter,
                "ema": average_metrics,
                "trainable_ema_delta": delta,
                "optimizer_record_valid": record_valid,
                "optimizer_shape_dtype_match": shape_dtype_match,
                "optimizer_step": step_value,
                "optimizer_step_exact": step_ok,
                "exp_avg": first_metrics,
                "exp_avg_sq": second_metrics,
                "optimizer_moments_nonzero": moments_nonzero,
                "base_pass": base_pass,
            })
        changed_keys = [
            row["key"]
            for row in rows
            if row["trainable_ema_delta"].get("finite") is True
            and row["trainable_ema_delta"].get("changed_elements", 0) > 0
        ]
        required_delta_keys: list[str]
        if group == "adapter_blocks":
            required_delta_keys = sorted(
                key for key in selected if not key.endswith(ADAPTER_NORM_SUFFIXES)
            )
        elif group in DELTA_ALL_GROUPS:
            required_delta_keys = sorted(selected)
        else:
            required_delta_keys = []
        missing_delta_keys = sorted(set(required_delta_keys) - set(changed_keys))
        expected_count = GROUP_COUNTS[group]
        gate = (
            len(selected) == expected_count
            and all(row["base_pass"] for row in rows)
            and not missing_delta_keys
        )
        if group == "adapter_blocks":
            gate = gate and len(required_delta_keys) == 14
        groups[group] = {
            "status": "PASS" if gate else "FAIL",
            "key_count": len(selected),
            "expected_key_count": expected_count,
            "changed_key_count": len(changed_keys),
            "changed_keys": sorted(changed_keys),
            "required_delta_key_count": len(required_delta_keys),
            "required_delta_keys": required_delta_keys,
            "missing_required_delta_keys": missing_delta_keys,
            "moment_backed_key_count": sum(row["optimizer_moments_nonzero"] for row in rows),
            "step_exact_key_count": sum(row["optimizer_step_exact"] for row in rows),
            "base_pass_key_count": sum(row["base_pass"] for row in rows),
            "rows": rows,
        }
    structural_pass = (
        structural["state_ema_keyset_equal"]
        and structural["state_optimizer_keyset_equal"]
        and structural["expected_trainable_count"]
        and structural["expected_trainable_keyset"]
        and structural["expected_optimizer_hparams"]
        and structural["expected_scheduler_state"]
    )
    checks = {
        "structural_identity": structural_pass,
        **{f"group_{name}": value["status"] == "PASS" for name, value in groups.items()},
    }
    return {
        "schema_version": "stage256_20260718_smoke_v3_activation_diagnostics_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "expected_step": expected_step,
        "checks": checks,
        "structural_identity": structural,
        "groups": groups,
        "gate_semantics": {
            "delta_required_for_all": sorted(DELTA_ALL_GROUPS),
            "adapter_blocks": "all 14 dense tensors require delta; all 22 require finite step-3 Adam moments",
            "adapter_out_norm": "finite step-3 Adam moments; exact parameter delta is diagnostic only",
            "text_k_norm": "all 28 require finite step-3 Adam moments; exact parameter delta is diagnostic only",
        },
    }


def collect_full_state_audit(
    trainable_state: dict[str, Any],
    ema: dict[str, Any],
    expected_step: int,
    checkpoint_step: int,
) -> dict[str, Any]:
    import torch

    state_keys, ema_keys = set(trainable_state), set(ema)
    failures = []
    all_shapes_match = state_keys == ema_keys
    all_finite = state_keys == ema_keys
    dtype_counts: dict[str, int] = {}
    if state_keys == ema_keys:
        for key in sorted(state_keys):
            current, average = trainable_state[key], ema[key]
            valid_tensor = torch.is_tensor(current) and torch.is_tensor(average)
            shape_match = valid_tensor and current.shape == average.shape
            dtype_match = valid_tensor and current.dtype == average.dtype
            current_finite = valid_tensor and bool(torch.isfinite(current).all())
            ema_finite = valid_tensor and bool(torch.isfinite(average).all())
            all_shapes_match = all_shapes_match and valid_tensor and shape_match and dtype_match
            all_finite = all_finite and current_finite and ema_finite
            if valid_tensor:
                dtype_counts[str(current.dtype)] = dtype_counts.get(str(current.dtype), 0) + 1
            if not (valid_tensor and shape_match and dtype_match and current_finite and ema_finite):
                failures.append({
                    "key": key,
                    "valid_tensor": valid_tensor,
                    "shape_match": shape_match,
                    "dtype_match": dtype_match,
                    "current_finite": current_finite,
                    "ema_finite": ema_finite,
                })
    checks = {
        "global_step": checkpoint_step == expected_step == 3,
        "state_ema_keyset_equal": state_keys == ema_keys,
        "trainable_count_483": len(state_keys) == EXPECTED["trainable_count"],
        "ema_count_483": len(ema_keys) == EXPECTED["trainable_count"],
        "keyset_sha256": keyset_sha256(state_keys) == EXPECTED["trainable_keyset_sha256"],
        "all_shapes_dtypes_finite": all_shapes_match and all_finite,
    }
    return {
        "schema_version": "stage256_20260718_smoke_v3_full_state_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "global_step": checkpoint_step,
        "expected_step": expected_step,
        "trainable_count": len(state_keys),
        "ema_count": len(ema_keys),
        "keyset_sha256": keyset_sha256(state_keys),
        "all_shapes_match": all_shapes_match,
        "all_finite": all_finite,
        "dtype_counts": dtype_counts,
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
    }


def write_failure_diagnostic_once(
    path: Path,
    *,
    phase: str,
    error: BaseException,
    evidence: dict[str, Any] | None = None,
) -> None:
    """Best-effort create-once failure evidence for an early hard gate."""
    if path.exists() or path.is_symlink():
        return
    value = {
        "schema_version": "stage256_20260718_smoke_v3_activation_diagnostics_v1",
        "status": "FAIL",
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "evidence": evidence or {},
    }
    try:
        atomic_json(path, value)
    except Exception as writer_error:
        print(
            json.dumps({
                "status": "FAIL",
                "phase": phase,
                "diagnostic_writer_error": repr(writer_error),
                "original_error": repr(error),
            }, sort_keys=True),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--eval-helper", type=Path, required=True)
    parser.add_argument("--forensics-receipt", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument("--full-state-output", type=Path, required=True)
    parser.add_argument("--forward-probe-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify(args: argparse.Namespace) -> int:

    if args.expected_step != 3:
        raise ValueError("formal smoke-v3 contract requires exactly three completed steps")
    output_paths = [
        args.diagnostic_output,
        args.full_state_output,
        args.forward_probe_output,
        args.output,
    ]
    if len({str(path) for path in output_paths}) != len(output_paths):
        raise ValueError("smoke-v3 evidence and success outputs must be distinct")
    for path in output_paths:
        if not path.is_absolute():
            raise ValueError(f"output must be absolute: {path}")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"smoke-v3 output already exists: {path}")

    regular_file(args.config, "config")
    regular_file(args.checkpoint, "checkpoint")
    regular_file(args.eval_helper, "eval helper")
    if sha256_file(args.config) != EXPECTED["config_sha256"]:
        raise RuntimeError("smoke-v3 config SHA drift")
    if sha256_file(args.eval_helper) != EXPECTED["eval_helper_sha256"]:
        raise RuntimeError("smoke-v3 eval-helper SHA drift")
    forensics = validate_forensics_receipt(args.forensics_receipt)

    checkpoint_stat_before = stable_stat(args.checkpoint)
    helper = load_module(args.eval_helper)
    relative_config = str(args.config.resolve().relative_to(args.repo_root.resolve()))
    run_spec = {
        "name": "7.18 Stage256 W8 smoke v3",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": args.expected_step,
        "config": relative_config,
        "config_sha256": EXPECTED["config_sha256"],
        "overrides": [],
        "architecture_expect": {
            "model.patch_size": 1,
            "model.text_max_length": 384,
            "model.sandwich_norm": True,
            "model.noise_condition": True,
            "model.long_skip.enabled": False,
            "model.text_adapter.enabled": True,
            "model.text_adapter.bidirectional": False,
            "model.text_adapter.multi_layer_fusion.enabled": True,
            "model.text_adapter.multi_layer_fusion.num_layers": 12,
            "model.und_lora.enabled": False,
            "model.abs_sincos.enabled": False,
        },
    }
    checkpoint_file, checkpoint_identity = helper.stable_open_and_hash(args.checkpoint)
    try:
        model, checkpoint_meta, config_identity, audit = helper.build_model(
            args.repo_root, run_spec, checkpoint_file
        )
    finally:
        checkpoint_file.close()

    architecture_checks = {
        "architecture_fingerprint": audit["architecture_fingerprint"]["sha256"] == EXPECTED["architecture_sha256"],
        "ablation_fingerprint": audit["checkpoint_ablation_fingerprint"]["sha256"] == EXPECTED["ablation_sha256"],
        "trainable_count": audit["trainable_key_count"] == EXPECTED["trainable_count"],
        "trainable_keyset": audit["trainable_keyset_sha256"] == EXPECTED["trainable_keyset_sha256"],
        "frozen_missing_count": audit["allowed_frozen_missing_count"] == EXPECTED["frozen_missing_count"],
        "frozen_missing_keyset": audit["frozen_missing_keyset_sha256"] == EXPECTED["frozen_missing_keyset_sha256"],
        "weights_src_ema": audit.get("weights_src") == "ema",
        "checkpoint_step": int(checkpoint_meta.get("global_step", -1)) == args.expected_step,
        "config_identity": config_identity.get("sha256") == EXPECTED["config_sha256"],
    }
    total = sum(parameter.numel() for parameter in model.model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.model.parameters()
        if parameter.requires_grad
    )
    frozen = total - trainable
    rounded = [round(value / 1e6, 1) for value in (total, trainable, frozen)]
    architecture_checks["parameter_counts"] = rounded == EXPECTED["parameter_counts_millions"]
    if not all(architecture_checks.values()):
        raise RuntimeError(
            "smoke-v3 architecture/checkpoint identity gate failed: "
            + json.dumps(architecture_checks, sort_keys=True)
        )
    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trainable_state = checkpoint.get("trainable_state")
    ema = checkpoint.get("ema")
    optimizer = checkpoint.get("optimizer")
    scheduler = checkpoint.get("scheduler")
    if (
        not isinstance(trainable_state, dict)
        or not isinstance(ema, dict)
        or not isinstance(optimizer, dict)
        or not isinstance(scheduler, dict)
    ):
        raise RuntimeError(
            "smoke-v3 checkpoint lacks trainable_state/EMA/optimizer/scheduler dictionaries"
        )
    checkpoint_step = int(checkpoint.get("global_step", -1))

    def failed_probe(weights_src: str, error: BaseException) -> dict[str, Any]:
        return {
            "status": "FAIL",
            "weights_src": weights_src,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }

    try:
        ema_probe = run_float32_forward_probe(
            model.model,
            weights_src="ema",
            require_nonzero_branch=False,
        )
    except Exception as error:
        ema_probe = failed_probe("ema", error)

    model_state_keys = set(model.model.state_dict())
    expected_frozen_missing = model_state_keys - set(trainable_state)
    strict_load = {
        "expected_frozen_missing_count": len(expected_frozen_missing),
        "expected_frozen_missing_keyset_sha256": keyset_sha256(expected_frozen_missing),
    }
    try:
        missing, unexpected = model.model.load_state_dict(trainable_state, strict=False)
        missing_set, unexpected_set = set(missing), set(unexpected)
        strict_load.update({
            "missing_count": len(missing_set),
            "missing_keyset_sha256": keyset_sha256(missing_set),
            "unexpected_count": len(unexpected_set),
            "unexpected_keys": sorted(unexpected_set),
            "missing_equals_expected_frozen": missing_set == expected_frozen_missing,
            "expected_frozen_contract": (
                len(expected_frozen_missing) == EXPECTED["frozen_missing_count"]
                and keyset_sha256(expected_frozen_missing)
                == EXPECTED["frozen_missing_keyset_sha256"]
            ),
        })
        strict_load["status"] = (
            "PASS"
            if not unexpected_set
            and strict_load["missing_equals_expected_frozen"]
            and strict_load["expected_frozen_contract"]
            else "FAIL"
        )
        if strict_load["status"] != "PASS":
            raise RuntimeError(
                "strict trainable-state load did not equal the frozen-missing contract: "
                + json.dumps(strict_load, sort_keys=True)
            )
        current_probe = run_float32_forward_probe(
            model.model,
            weights_src="trainable_state",
            require_nonzero_branch=True,
        )
    except Exception as error:
        strict_load.setdefault("status", "FAIL")
        strict_load.setdefault("error_type", type(error).__name__)
        strict_load.setdefault("error", str(error))
        current_probe = failed_probe("trainable_state", error)

    def probe_check(probe: dict[str, Any], name: str) -> bool:
        checks = probe.get("checks")
        return isinstance(checks, dict) and checks.get(name) is True

    adapter_blocks_executed = len(
        current_probe.get("adapter", {}).get("hook_rows", [])
    )
    causal_prefix_independent = (
        probe_check(ema_probe, "causal_prefix_independent_of_future")
        and probe_check(current_probe, "causal_prefix_independent_of_future")
    )
    trainable_all_finite = (
        probe_check(current_probe, "two_block_hooks_finite_shape")
        and probe_check(current_probe, "adapter_branch_finite_shape")
        and probe_check(current_probe, "mlf_branch_finite_shape")
    )
    ema_all_finite = (
        probe_check(ema_probe, "two_block_hooks_finite_shape")
        and probe_check(ema_probe, "adapter_branch_finite_shape")
        and probe_check(ema_probe, "mlf_branch_finite_shape")
    )
    forward_checks = {
        "ema_execution": ema_probe.get("status") == "PASS",
        "strict_trainable_state_load": strict_load.get("status") == "PASS",
        "trainable_execution": current_probe.get("status") == "PASS",
        "adapter_blocks_executed_2": adapter_blocks_executed == 2,
        "causal_prefix_independent": causal_prefix_independent,
        "trainable_fp32_pre_residual_finite": trainable_all_finite,
        "trainable_adapter_branch_nonzero": probe_check(
            current_probe, "trainable_adapter_branch_nonzero"
        ),
        "trainable_mlf_branch_nonzero": probe_check(
            current_probe, "trainable_mlf_branch_nonzero"
        ),
        "trainable_future_branch_sensitive": probe_check(
            current_probe, "trainable_future_branch_sensitive"
        ),
        "ema_finite_shape": ema_all_finite,
    }
    forward_probe = {
        "schema_version": "stage256_20260718_smoke_v3_forward_probe_v1",
        "status": "PASS" if all(forward_checks.values()) else "FAIL",
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "forensics_receipt_sha256": forensics["sha256"],
        "checks": forward_checks,
        "adapter_blocks_executed": adapter_blocks_executed,
        "causal_prefix_independent": causal_prefix_independent,
        "strict_trainable_state_load": strict_load,
        "trainable_fp32_pre_residual": {
            "all_finite": trainable_all_finite,
            "adapter_nonzero": probe_check(
                current_probe, "trainable_adapter_branch_nonzero"
            ),
            "mlf_nonzero": probe_check(
                current_probe, "trainable_mlf_branch_nonzero"
            ),
            "adapter": current_probe.get("adapter"),
            "mlf": current_probe.get("mlf"),
        },
        "ema": {
            "all_finite": ema_all_finite,
            "shapes_match": ema_all_finite,
            "adapter": ema_probe.get("adapter"),
            "mlf": ema_probe.get("mlf"),
        },
        "details": {
            "ema": ema_probe,
            "trainable_state": current_probe,
        },
    }
    forward_probe_sha = atomic_json(args.forward_probe_output, forward_probe)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    diagnostics = collect_activation_diagnostics(
        trainable_state, ema, optimizer, scheduler, args.expected_step
    )
    adapter_group = diagnostics["groups"]["adapter_blocks"]
    adapter_rows = adapter_group["rows"]
    adapter_norm_keys = sorted(
        row["key"]
        for row in adapter_rows
        if row["key"].endswith(ADAPTER_NORM_SUFFIXES)
    )
    adapter_unchanged_keys = sorted(
        row["key"]
        for row in adapter_rows
        if row["trainable_ema_delta"].get("finite") is True
        and row["trainable_ema_delta"].get("changed_elements") == 0
    )
    checkpoint_step_pass = checkpoint_step == args.expected_step == 3
    diagnostics["checks"]["checkpoint_global_step"] = checkpoint_step_pass
    if not checkpoint_step_pass:
        diagnostics["status"] = "FAIL"
    diagnostics.update({
        "checkpoint": checkpoint_identity,
        "checkpoint_step": checkpoint_step,
        "global_step": checkpoint_step,
        "forensics": forensics,
        "forensics_receipt_sha256": forensics["sha256"],
        "adapter_blocks": {
            "key_count": adapter_group["key_count"],
            "dense_changed_key_count": (
                adapter_group["required_delta_key_count"]
                - len(adapter_group["missing_required_delta_keys"])
            ),
            "optimizer_moment_nonzero_key_count": adapter_group[
                "moment_backed_key_count"
            ],
            "norm_key_count": len(adapter_norm_keys),
            "norm_delta_reported_count": sum(
                row["trainable_ema_delta"].get("finite") is True
                for row in adapter_rows
                if row["key"] in adapter_norm_keys
            ),
            "unexpected_unchanged_dense_count": len(
                adapter_group["missing_required_delta_keys"]
            ),
            "unchanged_keys": adapter_unchanged_keys,
            "exact_norm_only_parameter_gap": (
                adapter_unchanged_keys == adapter_norm_keys
            ),
        },
        "adapter_out_norm": {
            "key_count": diagnostics["groups"]["adapter_out_norm"]["key_count"],
            "optimizer_moment_nonzero_key_count": diagnostics["groups"][
                "adapter_out_norm"
            ]["moment_backed_key_count"],
        },
        "text_k_norm": {
            "key_count": diagnostics["groups"]["text_k_norm"]["key_count"],
            "optimizer_moment_nonzero_key_count": diagnostics["groups"][
                "text_k_norm"
            ]["moment_backed_key_count"],
        },
    })
    diagnostic_sha = atomic_json(args.diagnostic_output, diagnostics)

    full_state = collect_full_state_audit(
        trainable_state, ema, args.expected_step, checkpoint_step
    )
    full_state.update({
        "checkpoint": checkpoint_identity,
        "forensics_receipt_sha256": forensics["sha256"],
    })
    full_state_sha = atomic_json(args.full_state_output, full_state)

    checkpoint_stat_after = stable_stat(args.checkpoint)
    all_pass = (
        diagnostics["status"] == "PASS"
        and full_state["status"] == "PASS"
        and forward_probe["status"] == "PASS"
        and checkpoint_stat_after == checkpoint_stat_before
    )
    summary = {
        "activation": diagnostics["status"],
        "full_state": full_state["status"],
        "forward_probe": forward_probe["status"],
        "checkpoint_identity_stable": checkpoint_stat_after == checkpoint_stat_before,
    }
    if not all_pass:
        print(json.dumps({"status": "FAIL", "summary": summary}, sort_keys=True), flush=True)
        raise RuntimeError(
            "smoke-v3 moment-backed checkpoint gate failed; detailed create-once evidence was preserved"
        )

    success = {
        "schema_version": "stage256_20260718_smoke_checkpoint_gate_v3",
        "status": "PASS",
        "expected_step": args.expected_step,
        "checkpoint": checkpoint_identity,
        "checkpoint_meta": checkpoint_meta,
        "config_identity": config_identity,
        "load_audit": audit,
        "architecture_checks": architecture_checks,
        "parameter_counts": {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "rounded_millions": rounded,
        },
        "forensics": forensics,
        "forensics_receipt_sha256": forensics["sha256"],
        "activation_diagnostic_sha256": diagnostic_sha,
        "full_state_audit_sha256": full_state_sha,
        "forward_probe_sha256": forward_probe_sha,
        "evidence": {
            "activation_diagnostics": {
                "path": str(args.diagnostic_output),
                "sha256": diagnostic_sha,
            },
            "full_state": {
                "path": str(args.full_state_output),
                "sha256": full_state_sha,
            },
            "forward_probe": {
                "path": str(args.forward_probe_output),
                "sha256": forward_probe_sha,
            },
        },
        "summary": summary,
    }
    output_sha = atomic_json(args.output, success)
    print(json.dumps({
        "status": "PASS",
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "architecture_sha256": audit["architecture_fingerprint"]["sha256"],
        "trainable_keys": audit["trainable_key_count"],
        "groups": {
            name: {
                "keys": group["key_count"],
                "changed": group["changed_key_count"],
                "moments": group["moment_backed_key_count"],
            }
            for name, group in diagnostics["groups"].items()
        },
        "output_sha256": output_sha,
    }, sort_keys=True), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    try:
        return verify(args)
    except Exception as error:
        write_failure_diagnostic_once(
            args.diagnostic_output,
            phase="verification",
            error=error,
            evidence={
                "checkpoint": str(args.checkpoint),
                "expected_step": args.expected_step,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
