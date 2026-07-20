#!/usr/bin/env python3
"""Fail-closed static gate for the approved 2026-07-18 Stage256 retrain.

This gate is intentionally cheap: it checks configuration and JCC contracts,
not the large overlay/plan artifacts.  Large-artifact gates run in their own
CPU JCC jobs and publish separate success receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


EXPECTED_SAMPLING_SHA256 = (
    "07a0b46fa9ebb840ecf7f97cc7fccc0ec68e44c47368ae13c450bcfe81a4fd48"
)
EXPECTED_PREVIOUS_CONFIG_SHA256 = (
    "d0f5012be059f2cb98f78de0d1c15ac1f5d7ca1c9358370845f237d10fba1e72"
)
EXPECTED_CHECKPOINT_DIR = Path(
    "/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/"
    "stage256_20260718_causalmlf_offline_exact_v1"
)
EXPECTED_PLAN = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_arb_stage256_20260718_semantic_v1/w64_prod/plan"
)
EXPECTED_OVERLAY = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_arb_stage256_20260718_semantic_v1/overlay"
)
EXPECTED_STORE = (
    "stage256_20260718.offline_semantic_dataset.OfflineSemanticExactDataset"
)
EXPECTED_MODEL_VALUES = {
    "patch_size": 1,
    "time_gap": 2048,
    "timestep_multiplier": 1.0,
    "text_max_length": 384,
    "cfg_dropout_prob": 0.1,
    "sandwich_norm": True,
    "noise_condition": True,
}
EXPECTED_DIFFS = {
    "dataset.offline_plan_root",
    "dataset.offline_semantic_overlay_root",
    "dataset.store_cls",
    "model.text_adapter.bidirectional",
    "model.text_adapter.enabled",
    "model.text_adapter.multi_layer_fusion.enabled",
    "model.text_adapter.rope",
    "optimizer.params.lr",
    "trainer.checkpoint_dir",
    "trainer.max_exposure_images",
}
SHA_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        result.update(flatten(value[key], path))
    return result


def require(mapping: dict[str, Any], dotted: str, expected: Any) -> None:
    cursor: Any = mapping
    for token in dotted.split("."):
        if not isinstance(cursor, dict) or token not in cursor:
            raise ValueError(f"missing required config key {dotted}")
        cursor = cursor[token]
    if cursor != expected:
        raise ValueError(
            f"config drift at {dotted}: expected={expected!r}, actual={cursor!r}"
        )


def validate_config(
    config_path: Path,
    baseline_path: Path,
    previous_path: Path,
    sampling_path: Path,
) -> dict[str, Any]:
    config = read_yaml(config_path)
    baseline = read_yaml(baseline_path)
    previous_sha = sha256_file(previous_path)
    if previous_sha != EXPECTED_PREVIOUS_CONFIG_SHA256:
        raise ValueError(
            "previous-best config identity drift: "
            f"expected={EXPECTED_PREVIOUS_CONFIG_SHA256}, actual={previous_sha}"
        )

    left, right = flatten(baseline), flatten(config)
    diff = {
        key: {"baseline": left.get(key, "<MISSING>"), "new": right.get(key, "<MISSING>")}
        for key in sorted(set(left) | set(right))
        if left.get(key, "<MISSING>") != right.get(key, "<MISSING>")
    }
    if set(diff) != EXPECTED_DIFFS:
        raise ValueError(
            "new config differs from offline-exact baseline outside the approved allowlist: "
            f"missing={sorted(EXPECTED_DIFFS-set(diff))}, "
            f"unexpected={sorted(set(diff)-EXPECTED_DIFFS)}"
        )

    for key, expected in EXPECTED_MODEL_VALUES.items():
        require(config, f"model.{key}", expected)
    for dotted, expected in {
        "model.text_adapter.enabled": True,
        "model.text_adapter.num_layers": 2,
        "model.text_adapter.bidirectional": False,
        "model.text_adapter.rope": False,
        "model.text_adapter.multi_layer_fusion.enabled": True,
        "model.text_adapter.multi_layer_fusion.num_layers": 12,
        "model.long_skip.enabled": False,
        "model.long_skip.mode": "A_full",
        "model.und_lora.enabled": False,
        "model.abs_sincos.enabled": False,
        "dataset.store_cls": EXPECTED_STORE,
        "dataset.offline_plan_root": str(EXPECTED_PLAN),
        "dataset.offline_semantic_overlay_root": str(EXPECTED_OVERLAY),
        "dataset.multi_ar.enabled": True,
        "dataset.multi_ar.assignment_mode": "offline_exact_v1",
        "dataset.multi_ar.target_area": 65536,
        "dataset.multi_ar.adaptive_no_upscale": True,
        "dataset.multi_ar.square_only_domains": ["synthetic", "textatlas"],
        "optimizer.params.lr": 0.0002121,
        "trainer.batch_size": 36,
        "trainer.max_exposure_images": 384000768,
        "trainer.checkpoint_dir": str(EXPECTED_CHECKPOINT_DIR),
        "trainer.resume": True,
        "trainer.ema.enabled": True,
        "trainer.ema.decay": 0.9999,
    }.items():
        require(config, dotted, expected)
    trainer = config.get("trainer", {})
    if "init_from" in trainer or "init_from" in config:
        raise ValueError("fresh run must not contain init_from")

    sampling_sha = sha256_file(sampling_path)
    if sampling_sha != EXPECTED_SAMPLING_SHA256:
        raise ValueError(
            f"sampling config SHA drift: expected={EXPECTED_SAMPLING_SHA256}, "
            f"actual={sampling_sha}"
        )
    sampling = read_json(sampling_path)
    if sampling["domain_mix"]["256"] != {
        "anime": 0.27,
        "monet_real": 0.405,
        "gpic": 0.259,
        "synthetic": 0.051,
        "textatlas": 0.015,
    }:
        raise ValueError("Stage256 domain mix drift")
    caption = sampling["anime"]["caption"]
    if caption.get("require_nl_stages") != ["256"]:
        raise ValueError("sampling config no longer requires Stage256 NL")
    routes = caption["stage_route_weights_has_nl"]["256"]
    if routes != {"rich_nl": 50, "nl": 30, "nl_short": 20, "tag_medium": 0}:
        raise ValueError("Stage256 anime caption routes drift")
    anime_sampling = sampling["anime"]["sampling"]
    if (
        anime_sampling["priority"].get("max_repeat") != 4
        or anime_sampling["priority"].get("max_priority") != 3.0
    ):
        raise ValueError("anime priority/repeat contract drift")

    return {
        "config_sha256": sha256_file(config_path),
        "baseline_sha256": sha256_file(baseline_path),
        "previous_config_sha256": previous_sha,
        "sampling_sha256": sampling_sha,
        "approved_structured_diff": diff,
    }


def validate_job(path: Path, *, workers: int, gpus: int, formal: bool) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    value = read_yaml(path)
    expected = {
        "numWorkers": workers,
        "gpuPerWorker": gpus,
        "maxRetries": 0,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError(f"{path}: {key} must equal {wanted}, got {value.get(key)!r}")
    command = value.get("command")
    if not isinstance(command, str) or not command:
        raise ValueError(f"{path}: missing command")
    required = [
        "stage256_20260718_causalmlf_offline_exact_v1.yaml",
        "trainer.init_from",
        "maxRetries",
    ]
    # The launch must explicitly prove init_from is absent.  The token appears
    # in the negative preflight, never as a training override.
    for token in required:
        if token not in raw:
            raise ValueError(f"{path}: required contract token absent: {token}")
    forbidden_overrides = (
        "model.text_adapter.enabled=",
        "model.text_adapter.bidirectional=",
        "model.text_adapter.multi_layer_fusion.enabled=",
        "optimizer.params.lr=",
    )
    for token in forbidden_overrides:
        if token in command:
            raise ValueError(f"{path}: formal architecture/LR dotlist override forbidden: {token}")
    if formal:
        if (workers, gpus) != (8, 8):
            raise AssertionError("formal topology validator misuse")
        for token in (
            "NNODES=8",
            "nproc-per-node 8",
            "384000768",
            str(EXPECTED_CHECKPOINT_DIR),
            "PLAN_AUDIT_SUCCESS.json",
            "OVERLAY_VERIFY_SUCCESS.json",
        ):
            if token not in raw:
                raise ValueError(f"{path}: formal token absent: {token}")
    else:
        if workers * gpus > 8:
            raise ValueError(f"{path}: smoke exceeds the eight-GPU test cap")
        for token in ("--max-steps 3", "trainer.resume=false"):
            if token not in raw:
                raise ValueError(f"{path}: smoke token absent: {token}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "name": value.get("name"),
        **expected,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise ValueError("output must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite static-gate output: {path}")
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--previous-config", type=Path, required=True)
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--smoke-job", type=Path, required=True)
    parser.add_argument("--formal-job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema_version": "stage256_20260718_static_gate_v1",
        "status": "PASS",
        "config": validate_config(
            args.config, args.baseline_config, args.previous_config, args.sampling
        ),
        "smoke_job": validate_job(args.smoke_job, workers=1, gpus=8, formal=False),
        "formal_job": validate_job(args.formal_job, workers=8, gpus=8, formal=True),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
