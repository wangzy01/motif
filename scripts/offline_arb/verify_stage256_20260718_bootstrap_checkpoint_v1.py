#!/usr/bin/env python3
"""CPU-only deep validator for the formal-v2 step-256 bootstrap checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

import torch
import yaml


EXPECTED_CONFIG_SHA256 = "70c8b4cdadfd17af29e087a9718958bc3e5655feff7afaf8d48ed7bd32bef00a"
EXPECTED_V1_YAML_SHA256 = "24eb86fed868f9ceaa15f434bdaf7428a9722ea24240dfad529ae4a183663e93"
EXPECTED_TRAINABLE_COUNT = 483
EXPECTED_KEYSET_SHA256 = "6c61b43d30a8bde2f13fc8a91e30a8a0de271934ba5c3fd72df392cd027a4424"
EXPECTED_STEP = 256
BASE_LR = 2.121e-4
WARMUP_STEPS = 1000
EXPECTED_ABLATION = {
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
BOOT_NAME = "evan-offarb-s256-20260718-causalmlf-boot256-8x8-v2"
RESUME_NAME = "evan-offarb-s256-20260718-causalmlf-8x8-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v1-yaml", type=Path, required=True)
    parser.add_argument("--boot-yaml", type=Path, required=True)
    parser.add_argument("--resume-yaml", type=Path, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    parser.add_argument("--plan-audit", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--final-health", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--static-contract", type=Path, required=True)
    parser.add_argument("--validator-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def regular(path: Path, *, nonempty: bool = True) -> os.stat_result:
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise RuntimeError(f"not a regular non-symlink file: {path}")
    if nonempty and value.st_size <= 0:
        raise RuntimeError(f"empty file: {path}")
    return value


def stat_identity(path: Path) -> dict[str, int]:
    value = regular(path)
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(16 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def keyset_sha256(keys: Any) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(str(key) for key in keys)) + "\n").encode("utf-8")
    ).hexdigest()


def finite_tensor(value: Any, *, shape: tuple[int, ...] | None = None) -> bool:
    if not isinstance(value, torch.Tensor):
        return False
    if shape is not None and tuple(value.shape) != shape:
        return False
    return bool(torch.isfinite(value).all().item())


def exact_step(value: Any, expected: int) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or not finite_tensor(value):
            return False
        value = value.item()
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(numeric) and numeric.is_integer() and int(numeric) == expected


def check_job_yaml(path: Path, expected_name: str, *, boot: bool) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    value = yaml.safe_load(raw)
    expected_resources = {
        "name": expected_name,
        "numWorkers": 8,
        "gpuPerWorker": 8,
        "cpuPerWorker": 80,
        "memGBPerWorker": 900,
        "maxRetries": 0,
    }
    observed = {key: value.get(key) for key in expected_resources}
    if observed != expected_resources:
        raise RuntimeError(f"job resource/name mismatch for {path}: {observed}")
    command = str(value.get("command", ""))
    required = [
        "trainer.checkpoint_dir=\"$CK\"",
        "stage256_20260718_causalmlf_offline_exact_v2",
        "tag_only_exposure",
        "FORMAL_CONTROL_PLANE_PASS",
    ]
    if boot:
        required.extend(["--max-steps 256", "formal_v2_boot_attempts"])
        if "OFFLINE_ARB_AUTO_RESUME_RECONCILE=1" in command:
            raise RuntimeError("bootstrap must not enable resume reconciliation")
    else:
        required.extend(
            [
                "OFFLINE_ARB_AUTO_RESUME_RECONCILE=1",
                "BOOT256_CHECKPOINT_SUCCESS.json",
                "formal_v2_resume_attempts",
            ]
        )
        if "--max-steps 256" in command:
            raise RuntimeError("main resume YAML must retain the full formal horizon")
    missing = [token for token in required if token not in command]
    if missing:
        raise RuntimeError(f"missing command contracts in {path}: {missing}")
    return {"sha256": sha256_file(path), "resources": observed}


def validate_metrics(path: Path) -> dict[str, Any]:
    before = stat_identity(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise RuntimeError(f"partial metrics tail at row {line_number}")
            value = json.loads(line)
            if type(value.get("step")) is not int or value["step"] != line_number:
                raise RuntimeError(f"metrics step mismatch at row {line_number}")
            for key, item in value.items():
                if key == "step":
                    continue
                if isinstance(item, (int, float)) and not math.isfinite(float(item)):
                    raise RuntimeError(f"non-finite metrics {key} at step {line_number}")
            rows.append(value)
    if len(rows) != EXPECTED_STEP:
        raise RuntimeError(f"expected {EXPECTED_STEP} metrics rows, found {len(rows)}")
    snapshots = [row for row in rows if "buf_semantic_anime_tag_only_records" in row]
    expected_snapshot_steps = {1, 50, 100, 150, 200, 250}
    if {int(row["step"]) for row in snapshots} != expected_snapshot_steps:
        raise RuntimeError("dataline snapshot steps mismatch")
    zero_keys = (
        "buf_semantic_anime_tag_only_records",
        "buf_semantic_tag_fallback_records",
        "buf_gpic_blob_source_fallback_items",
        "gpic_allrank_sum_source_fallback_items",
        "gpic_allrank_sum_integrity_failures",
        "gpic_allrank_sum_pread_errors",
        "gpic_allrank_sum_short_read_failures",
        "gpic_allrank_health_snapshot_errors",
    )
    for row in snapshots:
        for key in zero_keys:
            if float(row.get(key, -1)) != 0.0:
                raise RuntimeError(f"nonzero/missing {key} at step {row['step']}")
    if max(float(row.get("buf_semantic_anime_nl_records", 0)) for row in snapshots) <= 0:
        raise RuntimeError("runtime semantic anime NL evidence is empty")
    if stat_identity(path) != before:
        raise RuntimeError("metrics changed during validation")
    return {
        "identity": before,
        "sha256": sha256_file(path),
        "first_step": 1,
        "last_step": EXPECTED_STEP,
        "rows": len(rows),
        "last": rows[-1],
        "snapshot_steps": sorted(expected_snapshot_steps),
    }


def validate_checkpoint(path: Path) -> dict[str, Any]:
    before = stat_identity(path)
    if path.with_name(path.name + ".tmp").exists():
        raise RuntimeError("checkpoint temporary file is present")
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("global_step", -1)) != EXPECTED_STEP:
        raise RuntimeError("checkpoint global_step mismatch")
    if checkpoint.get("ablation_fp") != EXPECTED_ABLATION:
        raise RuntimeError(f"checkpoint ablation fingerprint mismatch: {checkpoint.get('ablation_fp')}")
    state = checkpoint.get("trainable_state")
    ema = checkpoint.get("ema")
    optimizer = checkpoint.get("optimizer")
    scheduler = checkpoint.get("scheduler")
    if not all(isinstance(value, dict) for value in (state, ema, optimizer, scheduler)):
        raise RuntimeError("checkpoint is missing state/EMA/optimizer/scheduler dictionaries")
    keys = set(state)
    opt_state = optimizer.get("state")
    if not isinstance(opt_state, dict):
        raise RuntimeError("optimizer state is not a dictionary")
    if not (
        len(keys) == EXPECTED_TRAINABLE_COUNT
        and keyset_sha256(keys) == EXPECTED_KEYSET_SHA256
        and set(ema) == keys
        and set(opt_state) == keys
    ):
        raise RuntimeError("model/EMA/optimizer keyset contract mismatch")
    nonzero_moment_count = 0
    for key in sorted(keys):
        parameter = state[key]
        average = ema[key]
        record = opt_state[key]
        shape = tuple(parameter.shape) if isinstance(parameter, torch.Tensor) else None
        if shape is None or not finite_tensor(parameter) or not finite_tensor(average, shape=shape):
            raise RuntimeError(f"non-finite/shape mismatch model or EMA tensor: {key}")
        if not isinstance(record, dict) or not exact_step(record.get("step"), EXPECTED_STEP):
            raise RuntimeError(f"optimizer step mismatch: {key}")
        first, second = record.get("exp_avg"), record.get("exp_avg_sq")
        if not finite_tensor(first, shape=shape) or not finite_tensor(second, shape=shape):
            raise RuntimeError(f"optimizer moment finite/shape mismatch: {key}")
        if bool(torch.count_nonzero(first).item()) and bool(torch.count_nonzero(second).item()):
            nonzero_moment_count += 1
    if nonzero_moment_count != EXPECTED_TRAINABLE_COUNT:
        raise RuntimeError(
            f"optimizer nonzero moment coverage {nonzero_moment_count}/{EXPECTED_TRAINABLE_COUNT}"
        )
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise RuntimeError("optimizer param-group contract mismatch")
    group = groups[0]
    if set(str(value) for value in group.get("params", [])) != keys:
        raise RuntimeError("optimizer param-group keyset mismatch")
    expected_lr = BASE_LR * (EXPECTED_STEP + 1) / WARMUP_STEPS
    if not (
        [float(value) for value in group.get("betas", [])] == [0.9, 0.95]
        and float(group.get("weight_decay")) == 0.0
        and float(group.get("eps")) == 1.0e-8
        and math.isclose(float(group.get("initial_lr")), BASE_LR, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(float(group.get("lr")), expected_lr, rel_tol=0.0, abs_tol=1e-15)
        and int(scheduler.get("last_epoch", -1)) == EXPECTED_STEP
        and [float(value) for value in scheduler.get("base_lrs", [])] == [BASE_LR]
        and len(scheduler.get("_last_lr", [])) == 1
        and math.isclose(float(scheduler["_last_lr"][0]), expected_lr, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise RuntimeError("optimizer/scheduler hyperparameter state mismatch")
    after = stat_identity(path)
    if after != before:
        raise RuntimeError("checkpoint changed during full validation")
    return {
        "identity": before,
        "sha256": digest,
        "global_step": EXPECTED_STEP,
        "trainable_count": len(keys),
        "trainable_keyset_sha256": keyset_sha256(keys),
        "ema_count": len(ema),
        "optimizer_state_count": len(opt_state),
        "optimizer_nonzero_moment_count": nonzero_moment_count,
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "scheduler_last_lr": float(scheduler["_last_lr"][0]),
    }


def discover_boot_logs(root: Path) -> dict[str, Any]:
    logs = sorted(root.glob(f"train.{BOOT_NAME}-*.n*.log"))
    pattern = re.compile(r"^train\.(.+)\.n([0-7])\.[0-9]{8}T[0-9]{6}Z\.log$")
    rows = []
    for path in logs:
        match = pattern.match(path.name)
        if match:
            rows.append((match.group(1), int(match.group(2)), path))
    jobtags = {jobtag for jobtag, _, _ in rows}
    if len(jobtags) != 1 or {rank for _, rank, _ in rows} != set(range(8)) or len(rows) != 8:
        raise RuntimeError(f"boot train-log identity/rank mismatch: {rows}")
    jobtag = next(iter(jobtags))
    rank0 = next(path for _, rank, path in rows if rank == 0)
    rank0_text = rank0.read_text(encoding="utf-8", errors="replace")
    if f"[train] done at step {EXPECTED_STEP};" not in rank0_text:
        raise RuntimeError("rank0 boot log lacks exact terminal training summary")
    forbidden = ("FATAL:", "Traceback (most recent call last)", "CUDA out of memory", "non-finite", "NCCL error")
    for _, _, path in rows:
        text = path.read_text(encoding="utf-8", errors="replace")
        hit = [token for token in forbidden if token in text]
        if hit:
            raise RuntimeError(f"forbidden log token in {path}: {hit}")
    return {
        "jobtag": jobtag,
        "ranks": list(range(8)),
        "rank0_log": str(rank0),
        "rank0_log_sha256": sha256_file(rank0),
    }


def json_status(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        raise RuntimeError(f"upstream receipt is not PASS: {path}")
    return value


def publish_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing output overwrite: {path}")
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    inputs = (
        args.checkpoint,
        args.metrics,
        args.config,
        args.v1_yaml,
        args.boot_yaml,
        args.resume_yaml,
        args.addendum,
        args.plan_audit,
        args.overlay,
        args.freeze,
        args.final_health,
        args.static,
        args.static_contract,
        args.validator_yaml,
    )
    for path in inputs:
        regular(path)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"validator output already exists: {args.output}")
    if sha256_file(args.config) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("frozen config SHA drift")
    if sha256_file(args.v1_yaml) != EXPECTED_V1_YAML_SHA256:
        raise RuntimeError("formal-v1 YAML SHA drift")
    plan_audit = json_status(args.plan_audit)
    overlay = json_status(args.overlay)
    final_health = json_status(args.final_health)
    static_receipt = json_status(args.static)
    static_contract = json_status(args.static_contract)
    if not (
        int(plan_audit["formal_prefix"]["steps"]) == 166667
        and int(plan_audit["formal_prefix"]["exposures"]) == 384000768
        and int(plan_audit["full_stage"]["anime_tag_only_exposures"]) == 0
        and int(plan_audit["multi_ar"]["active_native_bucket_count"]) == 20
        and int(overlay["checks"]["anime_full_sidecar_strict_nl"]["tag_only_selected"]) == 0
        and int(final_health["tag_only_exposure"]) == 0
        and static_receipt["formal_job"]["sha256"] == EXPECTED_V1_YAML_SHA256
        and static_contract.get("status") == "PASS"
    ):
        raise RuntimeError("frozen semantic/static upstream contract mismatch")
    boot_yaml = check_job_yaml(args.boot_yaml, BOOT_NAME, boot=True)
    resume_yaml = check_job_yaml(args.resume_yaml, RESUME_NAME, boot=False)
    metrics = validate_metrics(args.metrics)
    logs = discover_boot_logs(args.result_root)
    checkpoint = validate_checkpoint(args.checkpoint)
    receipt = {
        "schema": "stage256_20260718_formal_v2_bootstrap_checkpoint_v1",
        "status": "PASS",
        "expected_step": EXPECTED_STEP,
        "exact_target_step": 166667,
        "exact_target_exposures": 384000768,
        "world_size": 64,
        "global_batch": 2304,
        "tag_only_exposure": 0,
        "checkpoint": {"path": str(args.checkpoint.resolve()), **checkpoint},
        "metrics": {"path": str(args.metrics.resolve()), **metrics},
        "boot_logs": logs,
        "config": {"path": str(args.config.resolve()), "sha256": EXPECTED_CONFIG_SHA256},
        "formal_v1_yaml_sha256": EXPECTED_V1_YAML_SHA256,
        "boot_yaml": boot_yaml,
        "resume_yaml": resume_yaml,
        "addendum_sha256": sha256_file(args.addendum),
        "validator_yaml_sha256": sha256_file(args.validator_yaml),
        "freeze_sha256": sha256_file(args.freeze),
        "plan_audit_sha256": sha256_file(args.plan_audit),
        "overlay_sha256": sha256_file(args.overlay),
        "final_health_sha256": sha256_file(args.final_health),
        "static_sha256": sha256_file(args.static),
        "static_contract_sha256": sha256_file(args.static_contract),
    }
    publish_exclusive(args.output, receipt)
    print(json.dumps({"status": "PASS", "output": str(args.output), "checkpoint_sha256": checkpoint["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
