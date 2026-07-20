#!/usr/bin/env python3
"""GPU postflight for the three-step 7.18 Stage256 W8 smoke checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path


EXPECTED = {
    "architecture_sha256": "d6db3855f1bd561ee21f31238b559dbff7adc65149a32025785cc259863d9abf",
    "ablation_sha256": "70546507a768a9d01705b8118b6e3cfea3f055f22db5e9bcdfb3e25e7d022642",
    "trainable_count": 483,
    "trainable_keyset_sha256": "6c61b43d30a8bde2f13fc8a91e30a8a0de271934ba5c3fd72df392cd027a4424",
    "frozen_missing_count": 310,
    "frozen_missing_keyset_sha256": "d380e45c9d5d53c7f9629c612b8a2adcd0bc075854931c583fe1570683511378",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("stage256_two_final_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_and_changed(current, ema) -> tuple[bool, float, float]:
    import torch

    if current.shape != ema.shape:
        raise RuntimeError(
            f"trainable/EMA tensor shape mismatch: {current.shape} vs {ema.shape}"
        )
    finite = bool(torch.isfinite(current).all() and torch.isfinite(ema).all())
    current_norm = float(current.float().norm().item())
    delta = float((current.float() - ema.float()).norm().item())
    return finite, current_norm, delta


def atomic_json(path: Path, value: dict) -> None:
    if not path.is_absolute():
        raise ValueError("output path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists():
        raise FileExistsError(f"refusing smoke-postflight overwrite: {path}")
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=3)
    parser.add_argument("--eval-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_step != 3:
        raise ValueError("formal smoke contract requires exactly three completed steps")

    helper = load_module(args.eval_helper)
    config_sha = sha256_file(args.config)
    relative_config = str(args.config.resolve().relative_to(args.repo_root.resolve()))
    run_spec = {
        "name": "7.18 Stage256 W8 smoke",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": args.expected_step,
        "config": relative_config,
        "config_sha256": config_sha,
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

    if audit["architecture_fingerprint"]["sha256"] != EXPECTED["architecture_sha256"]:
        raise RuntimeError("smoke architecture fingerprint differs from previous-best topology")
    if audit["checkpoint_ablation_fingerprint"]["sha256"] != EXPECTED["ablation_sha256"]:
        raise RuntimeError("smoke checkpoint ablation fingerprint drift")
    if (
        audit["trainable_key_count"] != EXPECTED["trainable_count"]
        or audit["trainable_keyset_sha256"] != EXPECTED["trainable_keyset_sha256"]
        or audit["allowed_frozen_missing_count"] != EXPECTED["frozen_missing_count"]
        or audit["frozen_missing_keyset_sha256"] != EXPECTED["frozen_missing_keyset_sha256"]
    ):
        raise RuntimeError("smoke trainable/frozen key identity differs from previous-best topology")
    total = sum(parameter.numel() for parameter in model.model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.model.parameters() if parameter.requires_grad
    )
    frozen = total - trainable
    rounded = [round(value / 1e6, 1) for value in (total, trainable, frozen)]
    if rounded != [3366.0, 1645.5, 1720.6]:
        raise RuntimeError(f"parameter-count fingerprint drift: {rounded}")

    # Free the GPU model before the CPU checkpoint postflight; the second load
    # is deliberate so the canonical eval helper remains unmodified.
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        raise
    trainable_state = checkpoint.get("trainable_state")
    ema = checkpoint.get("ema")
    if not isinstance(trainable_state, dict) or not isinstance(ema, dict):
        raise RuntimeError("smoke checkpoint lacks trainable_state/EMA dictionaries")
    if set(trainable_state) != set(ema):
        raise RuntimeError("smoke checkpoint trainable_state and EMA keys differ")

    group_predicates = {
        "adapter_out_proj": lambda key: "text_adapter_moe_gen.out_proj" in key,
        "adapter_blocks": lambda key: "text_adapter_moe_gen.blocks" in key,
        "mlf_gate": lambda key: key.endswith("multi_layer_fusion_moe_gen.gate"),
        "mlf_logits": lambda key: key.endswith("multi_layer_fusion_moe_gen.logits"),
        "text_kv": lambda key: (
            "text_k_proj_moe_gen" in key or "text_v_proj_moe_gen" in key
        ),
    }
    groups: dict[str, dict] = {}
    for group, predicate in group_predicates.items():
        keys = sorted(key for key in trainable_state if predicate(key))
        if not keys:
            raise RuntimeError(f"smoke checkpoint has no keys for group {group}")
        rows = []
        for key in keys:
            finite, norm, delta = finite_and_changed(trainable_state[key], ema[key])
            rows.append({"key": key, "finite": finite, "norm": norm, "trainable_ema_delta": delta})
        if not all(row["finite"] for row in rows):
            raise RuntimeError(f"non-finite tensor in group {group}")
        changed = sum(row["trainable_ema_delta"] > 0.0 for row in rows)
        minimum_changed = len(rows) if group in {"adapter_out_proj", "mlf_gate", "mlf_logits"} else math.ceil(0.9 * len(rows))
        if changed < minimum_changed:
            raise RuntimeError(
                f"group {group} did not prove learning activation after three steps: "
                f"changed={changed}/{len(rows)}, required={minimum_changed}"
            )
        if group in {"adapter_out_proj", "mlf_gate", "mlf_logits"} and not all(
            row["norm"] > 0.0 for row in rows
        ):
            raise RuntimeError(f"zero-init group {group} remained exactly zero after smoke")
        groups[group] = {
            "key_count": len(rows),
            "changed_key_count": changed,
            "minimum_changed": minimum_changed,
            "rows": rows,
        }

    report = {
        "schema_version": "stage256_20260718_smoke_checkpoint_gate_v1",
        "status": "PASS",
        "checkpoint": checkpoint_identity,
        "checkpoint_meta": checkpoint_meta,
        "config_identity": config_identity,
        "load_audit": audit,
        "parameter_counts": {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "rounded_millions": rounded,
        },
        "learning_activation": groups,
    }
    atomic_json(args.output, report)
    print(json.dumps({
        "status": "PASS",
        "checkpoint_step": checkpoint_meta["global_step"],
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "architecture_sha256": audit["architecture_fingerprint"]["sha256"],
        "trainable_keys": audit["trainable_key_count"],
        "groups": {key: value["changed_key_count"] for key, value in groups.items()},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
