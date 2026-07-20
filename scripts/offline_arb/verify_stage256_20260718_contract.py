#!/usr/bin/env python3
"""Fail-closed static/release checks for the 7.18 Stage256 launch.

This checker intentionally reads only control-plane files.  Full catalog,
sidecar, plan-block and GPIC verification belongs to the CPU JCC build/freeze
jobs; GPU jobs consume their immutable success receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


RUN_ID = "stage256_20260718_causalmlf_offline_exact_v1"
RELEASE_ROOT = (
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_arb_stage256_20260718_semantic_v1"
)
CHECKPOINT_ROOT = (
    "/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/"
    "stage256_20260718_causalmlf_offline_exact_v1"
)
STORE_CLS = (
    "stage256_20260718.offline_semantic_dataset.OfflineSemanticExactDataset"
)
ARCHITECTURE_FINGERPRINT = {
    "text_adapter.enabled": True,
    "text_adapter.bidirectional": False,
    "text_adapter.num_layers": 2,
    "text_adapter.rope": False,
    "text_adapter.multi_layer_fusion.enabled": True,
    "text_adapter.multi_layer_fusion.num_layers": 12,
    "und_lora.enabled": False,
    "und_lora.num_layers": 8,
    "und_lora.rank": 32,
    "und_lora.alpha": 32.0,
    "und_lora.lr_scale": 0.3,
    "und_lora.target_modules": "k,o,q,v",
    "long_skip.enabled": False,
    "long_skip.mode": "A_full",
    "noise_condition": True,
    "abs_sincos.enabled": False,
    "sandwich_norm": True,
    "patch_size": 1,
}
ARCHITECTURE_FINGERPRINT_JSON = json.dumps(
    ARCHITECTURE_FINGERPRINT, sort_keys=True, separators=(",", ":")
)
ARCHITECTURE_FINGERPRINT_SHA256 = hashlib.sha256(
    ARCHITECTURE_FINGERPRINT_JSON.encode("utf-8")
).hexdigest()
EXPECTED_ARCHITECTURE_FINGERPRINT_SHA256 = (
    "70546507a768a9d01705b8118b6e3cfea3f055f22db5e9bcdfb3e25e7d022642"
)

ALLOWED_BASELINE_DIFFS = {
    "model.text_adapter.enabled",
    "model.text_adapter.bidirectional",
    "model.text_adapter.rope",
    "model.text_adapter.multi_layer_fusion.enabled",
    "dataset.store_cls",
    "dataset.offline_plan_root",
    "dataset.offline_semantic_overlay_root",
    "optimizer.params.lr",
    "trainer.checkpoint_dir",
    "trainer.max_exposure_images",
}

EXPECTED_JOBS = {
    "offline_arb_stage256_20260718_overlay_build.yaml": (1, 0, 0),
    "offline_arb_stage256_20260718_overlay_verify.yaml": (1, 0, 0),
    "offline_arb_stage256_20260718_selection_plans.yaml": (1, 0, 0),
    "offline_arb_stage256_20260718_semantic_materialize.yaml": (1, 0, 0),
    "offline_arb_stage256_20260718_semantic_validate.yaml": (1, 0, 0),
    "offline_arb_stage256_20260718_freeze.yaml": (1, 0, 0),
    "offline_arb_stage256_20260718_final_static.yaml": (1, 0, 0),
    "offline_arb_stage256_20260718_final_health.yaml": (1, 0, 0),
    "offline_exact_stage256_20260718_smoke_1x8.yaml": (1, 8, 0),
    "offline_exact_stage256_20260718_FORMAL_8x8.yaml": (8, 8, 0),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a mapping in {path}")
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(child, path))
    return result


def _config_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    adapter = model["text_adapter"]
    mlf = adapter["multi_layer_fusion"]
    und = model["und_lora"]
    targets = und.get("target_modules", ["q", "k", "v", "o"])
    if isinstance(targets, str):
        targets = [item.strip() for item in targets.split(",") if item.strip()]
    return {
        "text_adapter.enabled": bool(adapter.get("enabled", False)),
        "text_adapter.bidirectional": bool(adapter.get("bidirectional", True)),
        "text_adapter.num_layers": int(adapter.get("num_layers", 2)),
        "text_adapter.rope": bool(adapter.get("rope", False)),
        "text_adapter.multi_layer_fusion.enabled": bool(mlf.get("enabled", False)),
        "text_adapter.multi_layer_fusion.num_layers": int(mlf.get("num_layers", 12)),
        "und_lora.enabled": bool(und.get("enabled", False)),
        "und_lora.num_layers": int(und.get("num_layers", 8)),
        "und_lora.rank": int(und.get("rank", 32)),
        "und_lora.alpha": float(und.get("alpha", und.get("rank", 32))),
        "und_lora.lr_scale": float(und.get("lr_scale", 1.0)),
        "und_lora.target_modules": ",".join(sorted(str(item) for item in targets)),
        "long_skip.enabled": bool(model["long_skip"].get("enabled", False)),
        "long_skip.mode": str(model["long_skip"].get("mode", "A_full")),
        "noise_condition": bool(model.get("noise_condition", True)),
        "abs_sincos.enabled": bool(model["abs_sincos"].get("enabled", False)),
        "sandwich_norm": bool(model.get("sandwich_norm", True)),
        "patch_size": int(model.get("patch_size", 1)),
    }


def validate_config(config_path: Path, baseline_path: Path | None) -> dict[str, Any]:
    config = _load_yaml(config_path)
    model = config["model"]
    dataset = config["dataset"]
    trainer = config["trainer"]
    optimizer = config["optimizer"]["params"]

    fingerprint = _config_fingerprint(config)
    if fingerprint != ARCHITECTURE_FINGERPRINT:
        raise RuntimeError(
            "architecture fingerprint mismatch:\n"
            + json.dumps(fingerprint, indent=2, sort_keys=True)
        )
    if ARCHITECTURE_FINGERPRINT_SHA256 != EXPECTED_ARCHITECTURE_FINGERPRINT_SHA256:
        raise RuntimeError("source-code architecture fingerprint constant is inconsistent")
    if "init_from" in trainer:
        raise RuntimeError("fresh Stage256 config must not contain trainer.init_from")
    if trainer.get("resume") is not True:
        raise RuntimeError("trainer.resume must remain true for same-namespace recovery")
    if trainer.get("checkpoint_dir") != CHECKPOINT_ROOT:
        raise RuntimeError("checkpoint namespace drift")
    if dataset.get("store_cls") != STORE_CLS:
        raise RuntimeError("semantic dataset class drift")
    if dataset.get("offline_plan_root") != f"{RELEASE_ROOT}/w64_prod/plan":
        raise RuntimeError("formal semantic plan root drift")
    if dataset.get("offline_semantic_overlay_root") != f"{RELEASE_ROOT}/overlay":
        raise RuntimeError("semantic overlay root drift")
    if dataset.get("multi_ar", {}).get("enabled") is not True:
        raise RuntimeError("multi-AR must stay enabled")
    if dataset.get("multi_ar", {}).get("assignment_mode") != "offline_exact_v1":
        raise RuntimeError("offline-exact assignment mode drift")
    if int(trainer.get("batch_size", -1)) != 36:
        raise RuntimeError("per-rank batch must equal 36")
    if int(trainer.get("max_exposure_images", -1)) != 384_000_768:
        raise RuntimeError("formal exposure budget drift")
    if abs(float(optimizer.get("lr", -1.0)) - 2.121e-4) > 1e-15:
        raise RuntimeError("learning rate drift")
    if model.get("pretrained_path") != "/fsx/models/Qwen3-VL-2B-Instruct":
        raise RuntimeError("pretrained component root drift")

    changed: list[str] = []
    if baseline_path is not None:
        baseline = _flatten(_load_yaml(baseline_path))
        candidate = _flatten(config)
        for key in sorted(set(baseline) | set(candidate)):
            if baseline.get(key, object()) != candidate.get(key, object()):
                changed.append(key)
        unexpected = sorted(set(changed) - ALLOWED_BASELINE_DIFFS)
        missing = sorted(ALLOWED_BASELINE_DIFFS - set(changed))
        if unexpected or missing:
            raise RuntimeError(
                f"baseline diff contract failed; unexpected={unexpected}, missing={missing}"
            )

    return {
        "architecture_fingerprint": fingerprint,
        "architecture_fingerprint_json": ARCHITECTURE_FINGERPRINT_JSON,
        "architecture_fingerprint_sha256": ARCHITECTURE_FINGERPRINT_SHA256,
        "baseline_changed_keys": changed,
    }


def validate_jobs(jobs_dir: Path) -> dict[str, Any]:
    checked: dict[str, Any] = {}
    for filename, (workers, gpus, retries) in EXPECTED_JOBS.items():
        path = jobs_dir / filename
        value = _load_yaml(path)
        observed = (
            int(value.get("numWorkers", -1)),
            int(value.get("gpuPerWorker", -1)),
            int(value.get("maxRetries", -1)),
        )
        if observed != (workers, gpus, retries):
            raise RuntimeError(
                f"topology/retry drift in {filename}: {observed} != {(workers, gpus, retries)}"
            )
        if gpus == 0 and int(value.get("cpuPerWorker", 0)) < 16:
            raise RuntimeError(f"CPU JCC must request >=16 CPU: {filename}")
        command = value.get("command")
        if not isinstance(command, str) or "set -" not in command:
            raise RuntimeError(f"malformed command in {filename}")
        if "maxRetries: 0" not in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"maxRetries=0 must be explicit: {filename}")
        if gpus > 0:
            for forbidden in (
                "--create-freeze",
                "--build-overlay",
                "materialize_stage256_semantic_plan.py",
                "materialize_semantic_plan.py",
                "verify_stage256_semantic_overlay.py",
                "validate_stage256_semantic_overlay.py",
                "audit_semantic_plan.py",
            ):
                if forbidden in command:
                    raise RuntimeError(
                        f"heavy CPU operation leaked into GPU job {filename}: {forbidden}"
                    )
            common_tokens = (
                "EXPECTED_ARCH_FP_SHA256=70546507a768a9d01705b8118b6e3cfea3f055f22db5e9bcdfb3e25e7d022642",
                "offline_gpic_blob_v1",
                "MANIFEST_SHA256=a18ba1f1250e3b00abdc1051dc71e08a254b8855d5b61b9975e88d2e6ae2ee05",
            )
            for token in common_tokens:
                if token not in command:
                    raise RuntimeError(f"GPU safety binding missing from {filename}: {token}")
            required_receipt = (
                "FINAL_HEALTH_SUCCESS.json"
                if filename.endswith("FORMAL_8x8.yaml")
                else "FREEZE_VERIFY_SUCCESS.json"
            )
            if required_receipt not in command:
                raise RuntimeError(
                    f"GPU prerequisite receipt missing from {filename}: {required_receipt}"
                )
        checked[filename] = {
            "name": value.get("name"),
            "numWorkers": workers,
            "gpuPerWorker": gpus,
            "maxRetries": retries,
        }
    return checked


def _walk_scalars(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_scalars(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_scalars(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def validate_receipts(receipts: list[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    saw_tag_only_proof = False
    for path in receipts:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"receipt missing/non-regular: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "PASS":
            raise RuntimeError(f"receipt status is not PASS: {path}")
        if value.get("run_id", RUN_ID) != RUN_ID:
            raise RuntimeError(f"receipt run_id drift: {path}")
        for key, scalar in _walk_scalars(value):
            leaf = key.rsplit(".", 1)[-1].lower()
            if leaf in {
                "tag_only_exposure",
                "tag_only_exposures",
                "planned_tag_only_exposure",
                "runtime_tag_only_exposure",
                "tag_only_selected",
                "anime_tag_only_exposures",
            }:
                if int(scalar) != 0:
                    raise RuntimeError(f"non-zero tag-only exposure in {path}: {key}={scalar}")
                saw_tag_only_proof = True
        result[str(path)] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    if not saw_tag_only_proof:
        raise RuntimeError("no receipt supplied an explicit tag-only exposure=0 proof")
    return result


def _atomic_create_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace create-once receipt: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o444)
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--jobs-dir", type=Path)
    parser.add_argument("--receipt", type=Path, action="append", default=[])
    parser.add_argument("--emit", type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "status": "PASS",
        "run_id": RUN_ID,
        "config": validate_config(args.config, args.baseline_config),
    }
    if args.jobs_dir is not None:
        report["jobs"] = validate_jobs(args.jobs_dir)
    if args.receipt:
        report["receipts"] = validate_receipts(args.receipt)
    report["contract_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if args.emit is not None:
        _atomic_create_json(args.emit, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
