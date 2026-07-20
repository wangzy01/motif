#!/usr/bin/env python3
"""CPU-only deep validator for the Stage256 formal-v3 recovery chain."""

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
EXPECTED_V2_YAML_SHA256 = "2b170af876aa0f1cd3c4e8e4fd84c05b4a902f371b4064a0abdb1760b163ed5e"
EXPECTED_BOOT_RECEIPT_SHA256 = "8fc7f3c1d2cef92566456aeadaf30399ec56b07033ffa04c1688a5dabdc634af"
EXPECTED_BOOT_CHECKPOINT_SHA256 = "61a9cccf75a5d9e6aef72d8404964bf705c7c727773459140fec7486dfb15e7a"
EXPECTED_TRAINABLE_COUNT = 483
EXPECTED_KEYSET_SHA256 = "6c61b43d30a8bde2f13fc8a91e30a8a0de271934ba5c3fd72df392cd027a4424"
EXPECTED_V2_JOBTAG = "evan-offarb-s256-20260718-causalmlf-8x8-v2-kbc7"
RECOVER_BASE = "evan-offarb-s256-20260718-causalmlf-recover384-8x8-v3"
MAIN_BASE = "evan-offarb-s256-20260718-causalmlf-8x8-v3"
PRE_VALIDATOR_BASE = "evan-offarb-s256-20260718-recover-preflight-v3"
POST_VALIDATOR_BASE = "evan-offarb-s256-20260718-recover384-validate-v3"
BOOT_STEP = 256
FAILED_METRICS_STEP = 430
RECOVER_STEP = 384
FINAL_STEP = 166667
EXACT_EXPOSURES = 384000768
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
ZERO_METRIC_KEYS = (
    "buf_semantic_anime_tag_only_records",
    "buf_semantic_tag_fallback_records",
    "buf_gpic_blob_source_fallback_items",
    "gpic_allrank_sum_source_fallback_items",
    "gpic_allrank_sum_integrity_failures",
    "gpic_allrank_sum_pread_errors",
    "gpic_allrank_sum_short_read_failures",
    "gpic_allrank_health_snapshot_errors",
)
FORBIDDEN_LOG_TOKENS = (
    "FATAL:",
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "non-finite",
    "NCCL error",
    "NCCL failure",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pre", "post"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v1-yaml", type=Path, required=True)
    parser.add_argument("--v2-yaml", type=Path, required=True)
    parser.add_argument("--recover-yaml", type=Path, required=True)
    parser.add_argument("--main-yaml", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    parser.add_argument("--boot-receipt", type=Path, required=True)
    parser.add_argument("--pre-receipt", type=Path)
    parser.add_argument("--pre-validator-yaml", type=Path, required=True)
    parser.add_argument("--post-validator-yaml", type=Path, required=True)
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
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def keyset_sha256(keys: Any) -> str:
    raw = ("\n".join(sorted(str(key) for key in keys)) + "\n").encode()
    return sha256_bytes(raw)


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


def stable_json(path: Path) -> dict[str, Any]:
    before = stat_identity(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if stat_identity(path) != before:
        raise RuntimeError(f"JSON changed while reading: {path}")
    return value


def validate_checkpoint(path: Path, expected_step: int) -> dict[str, Any]:
    before = stat_identity(path)
    if path.with_name(path.name + ".tmp").exists():
        raise RuntimeError("checkpoint temporary exists")
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("global_step", -1)) != expected_step:
        raise RuntimeError("checkpoint global_step mismatch")
    if checkpoint.get("ablation_fp") != EXPECTED_ABLATION:
        raise RuntimeError("checkpoint ablation fingerprint mismatch")
    state = checkpoint.get("trainable_state")
    ema = checkpoint.get("ema")
    optimizer = checkpoint.get("optimizer")
    scheduler = checkpoint.get("scheduler")
    if not all(isinstance(value, dict) for value in (state, ema, optimizer, scheduler)):
        raise RuntimeError("checkpoint state dictionaries missing")
    keys = set(state)
    opt_state = optimizer.get("state")
    if not isinstance(opt_state, dict):
        raise RuntimeError("optimizer state missing")
    if not (
        len(keys) == EXPECTED_TRAINABLE_COUNT
        and keyset_sha256(keys) == EXPECTED_KEYSET_SHA256
        and set(ema) == keys
        and set(opt_state) == keys
    ):
        raise RuntimeError("model/EMA/optimizer keyset mismatch")
    nonzero = 0
    for key in sorted(keys):
        parameter = state[key]
        average = ema[key]
        record = opt_state[key]
        shape = tuple(parameter.shape) if isinstance(parameter, torch.Tensor) else None
        if shape is None or not finite_tensor(parameter) or not finite_tensor(average, shape=shape):
            raise RuntimeError(f"model/EMA finite or shape mismatch: {key}")
        if not isinstance(record, dict) or not exact_step(record.get("step"), expected_step):
            raise RuntimeError(f"optimizer step mismatch: {key}")
        first, second = record.get("exp_avg"), record.get("exp_avg_sq")
        if not finite_tensor(first, shape=shape) or not finite_tensor(second, shape=shape):
            raise RuntimeError(f"optimizer moment finite/shape mismatch: {key}")
        if bool(torch.count_nonzero(first).item()) and bool(torch.count_nonzero(second).item()):
            nonzero += 1
    if nonzero != EXPECTED_TRAINABLE_COUNT:
        raise RuntimeError(f"optimizer nonzero coverage {nonzero}/{EXPECTED_TRAINABLE_COUNT}")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise RuntimeError("optimizer param-group mismatch")
    group = groups[0]
    expected_lr = BASE_LR * (expected_step + 1) / WARMUP_STEPS
    if not (
        set(str(value) for value in group.get("params", [])) == keys
        and [float(value) for value in group.get("betas", [])] == [0.9, 0.95]
        and float(group.get("weight_decay")) == 0.0
        and float(group.get("eps")) == 1.0e-8
        and math.isclose(float(group.get("initial_lr")), BASE_LR, rel_tol=0, abs_tol=1e-15)
        and math.isclose(float(group.get("lr")), expected_lr, rel_tol=0, abs_tol=1e-15)
        and int(scheduler.get("last_epoch", -1)) == expected_step
        and [float(value) for value in scheduler.get("base_lrs", [])] == [BASE_LR]
        and len(scheduler.get("_last_lr", [])) == 1
        and math.isclose(float(scheduler["_last_lr"][0]), expected_lr, rel_tol=0, abs_tol=1e-15)
    ):
        raise RuntimeError("optimizer/scheduler hyperparameter mismatch")
    if stat_identity(path) != before:
        raise RuntimeError("checkpoint changed during validation")
    return {
        "identity": before,
        "sha256": digest,
        "global_step": expected_step,
        "trainable_count": len(keys),
        "trainable_keyset_sha256": keyset_sha256(keys),
        "ema_count": len(ema),
        "optimizer_state_count": len(opt_state),
        "optimizer_nonzero_moment_count": nonzero,
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "scheduler_last_lr": float(scheduler["_last_lr"][0]),
    }


def validate_metrics(
    path: Path, expected_last: int, boot_metrics_sha256: str
) -> dict[str, Any]:
    before = stat_identity(path)
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if any(not line.endswith(b"\n") for line in lines):
        raise RuntimeError("metrics has partial tail")
    rows: list[dict[str, Any]] = []
    for expected, line in enumerate(lines, 1):
        value = json.loads(line)
        if type(value.get("step")) is not int or value["step"] != expected:
            raise RuntimeError(f"metrics step mismatch at row {expected}")
        for key, item in value.items():
            if key != "step" and isinstance(item, (int, float)) and not math.isfinite(float(item)):
                raise RuntimeError(f"non-finite metrics {key} at step {expected}")
        rows.append(value)
    if len(rows) != expected_last:
        raise RuntimeError(f"expected metrics 1..{expected_last}, found {len(rows)}")
    prefix = b"".join(lines[:BOOT_STEP])
    if sha256_bytes(prefix) != boot_metrics_sha256:
        raise RuntimeError("metrics 1..256 prefix differs from bootstrap receipt")
    snapshots = [row for row in rows if "buf_semantic_anime_tag_only_records" in row]
    expected_snapshots = {1, *range(50, expected_last + 1, 50)}
    if {int(row["step"]) for row in snapshots} != expected_snapshots:
        raise RuntimeError("runtime semantic snapshot steps mismatch")
    for row in snapshots:
        for key in ZERO_METRIC_KEYS:
            if float(row.get(key, -1)) != 0.0:
                raise RuntimeError(f"nonzero/missing {key} at step {row['step']}")
    if max(float(row.get("buf_semantic_anime_nl_records", 0)) for row in snapshots) <= 0:
        raise RuntimeError("runtime anime NL evidence empty")
    if stat_identity(path) != before:
        raise RuntimeError("metrics changed during validation")
    return {
        "identity": before,
        "sha256": sha256_bytes(raw),
        "prefix_1_256_sha256": sha256_bytes(prefix),
        "rows": len(rows),
        "first_step": 1,
        "last_step": expected_last,
        "snapshot_steps": sorted(expected_snapshots),
        "last": rows[-1],
    }


def check_gpu_yaml(path: Path, expected_name: str, *, recover: bool) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    value = yaml.safe_load(raw)
    expected = {
        "name": expected_name,
        "numWorkers": 8,
        "gpuPerWorker": 8,
        "cpuPerWorker": 80,
        "memGBPerWorker": 900,
        "maxRetries": 0,
    }
    observed = {key: value.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"GPU YAML resource mismatch: {path}: {observed}")
    command = str(value.get("command", ""))
    required = (
        "scripts/offline_arb/run_stage256_20260718_formal_v3.sh",
        "recover384" if recover else "main",
    )
    missing = [token for token in required if token not in command]
    if missing:
        raise RuntimeError(f"GPU YAML missing contracts {missing}: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "resources": observed}


def check_runner(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    required = (
        "recover384|main",
        "PRE_RECOVER256_SUCCESS.json",
        "RECOVER384_CHECKPOINT_SUCCESS.json",
        "formal_v3_recover384_attempts",
        "formal_v3_resume_attempts",
        "OFFLINE_ARB_AUTO_RESUME_RECONCILE=1",
        "--max-steps 384",
        "trainer.checkpoint_dir=\"$CK\"",
        "dataset.gpic_payload_backend=offline_gpic_blob_v1",
    )
    missing = [token for token in required if token not in raw]
    if missing:
        raise RuntimeError(f"formal-v3 runner missing contracts: {missing}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def check_cpu_yaml(path: Path, expected_name: str, phase: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "name": expected_name,
        "numWorkers": 1,
        "gpuPerWorker": 0,
        "cpuPerWorker": 32,
        "memGBPerWorker": 384,
        "maxRetries": 0,
    }
    observed = {key: value.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"CPU YAML resource mismatch: {path}: {observed}")
    command = str(value.get("command", ""))
    if f"--phase {phase}" not in command:
        raise RuntimeError(f"CPU YAML phase mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "resources": observed}


def validate_attempt(attempt: Path, jobtag: str) -> dict[str, Any]:
    if not attempt.is_dir() or attempt.is_symlink():
        raise RuntimeError(f"attempt directory missing/unsafe: {attempt}")
    fresh = []
    for rank in range(8):
        path = attempt / f"n{rank}.fresh"
        regular(path)
        fields = path.read_text(encoding="utf-8").strip().split("|")
        if len(fields) != 5 or fields[0] != "READY" or fields[1] != jobtag or int(fields[2]) != rank:
            raise RuntimeError(f"fresh marker mismatch: {path}: {fields}")
        fresh.append({"rank": rank, "path": str(path), "sha256": sha256_file(path)})
    namespace = regular(attempt / "namespace.pass")
    namespace_fields = (attempt / "namespace.pass").read_text().strip().split("|")
    if len(namespace_fields) not in (4, 5) or namespace_fields[:2] != ["PASS", jobtag]:
        raise RuntimeError("namespace.pass mismatch")
    regular(attempt / "rdzv.addr")
    rdzv_fields = (attempt / "rdzv.addr").read_text().strip().split("|")
    if len(rdzv_fields) not in (4, 5) or rdzv_fields[2] != jobtag:
        raise RuntimeError("rdzv mismatch")
    gpic_paths = sorted(attempt.glob("gpicpreflight.n*.json"))
    if len(gpic_paths) != 8:
        raise RuntimeError("expected eight GPIC receipts")
    gpic = []
    for path in gpic_paths:
        value = stable_json(path)
        rank = int(value.get("node_rank", -1))
        if value.get("status") != "PASS" or value.get("jobtag") != jobtag:
            raise RuntimeError(f"GPIC receipt mismatch: {path}")
        gpic.append({"rank": rank, "path": str(path), "sha256": sha256_file(path)})
    if {row["rank"] for row in gpic} != set(range(8)):
        raise RuntimeError("GPIC rank set mismatch")
    return {
        "path": str(attempt.resolve()),
        "fresh": fresh,
        "namespace_sha256": sha256_file(attempt / "namespace.pass"),
        "rdzv_sha256": sha256_file(attempt / "rdzv.addr"),
        "gpic": sorted(gpic, key=lambda row: row["rank"]),
    }


def discover_logs(result_root: Path, jobtag: str, *, post: bool) -> dict[str, Any]:
    logs = sorted(result_root.glob(f"train.{jobtag}.n*.log"))
    pattern = re.compile(rf"^train\.{re.escape(jobtag)}\.n([0-7])\.[0-9]{{8}}T[0-9]{{6}}Z\.log$")
    rows = []
    for path in logs:
        match = pattern.match(path.name)
        if match:
            rows.append((int(match.group(1)), path))
    if len(rows) != 8 or {rank for rank, _ in rows} != set(range(8)):
        raise RuntimeError(f"train log rank mismatch for {jobtag}: {rows}")
    rank0 = next(path for rank, path in rows if rank == 0)
    text0 = rank0.read_text(encoding="utf-8", errors="replace")
    if f"at step {BOOT_STEP}" not in text0 or "[train] resumed from " not in text0:
        raise RuntimeError("rank0 lacks exact step-256 resume")
    if post:
        if '"mode": "trim_to_checkpoint"' not in text0:
            raise RuntimeError("RECOVER384 log lacks trim reconciliation")
        if f"[train] done at step {RECOVER_STEP};" not in text0:
            raise RuntimeError("RECOVER384 log lacks terminal step384")
    else:
        if '"mode": "already_at_checkpoint"' not in text0:
            raise RuntimeError("formal-v2 log lacks exact-boundary reconciliation")
        if f"[train] step {FAILED_METRICS_STEP}/{FINAL_STEP}" not in text0:
            raise RuntimeError("formal-v2 log lacks terminal observed step430")
    log_rows = []
    for rank, path in rows:
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [token for token in FORBIDDEN_LOG_TOKENS if token in text]
        if hits:
            raise RuntimeError(f"forbidden tokens in {path}: {hits}")
        log_rows.append({"rank": rank, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    nodemon = sorted(result_root.glob(f"nodemon.{jobtag}.n*.log"))
    if len(nodemon) != 8:
        raise RuntimeError("expected eight nodemon logs")
    return {
        "rank0": str(rank0),
        "rank0_sha256": sha256_file(rank0),
        "train_logs": sorted(log_rows, key=lambda row: row["rank"]),
        "nodemon": [{"path": str(path), "sha256": sha256_file(path), "identity": stat_identity(path)} for path in nodemon],
    }


def discover_recover_jobtag(result_root: Path) -> str:
    pattern = re.compile(rf"^train\.({re.escape(RECOVER_BASE)}-[A-Za-z0-9]+)\.n[0-7]\.[0-9]{{8}}T[0-9]{{6}}Z\.log$")
    jobtags = set()
    for path in result_root.glob(f"train.{RECOVER_BASE}-*.n*.log"):
        match = pattern.match(path.name)
        if match:
            jobtags.add(match.group(1))
    if len(jobtags) != 1:
        raise RuntimeError(f"expected one RECOVER384 actual jobtag, found {sorted(jobtags)}")
    return next(iter(jobtags))


def validate_reconcile(
    checkpoint: Path, jobtag: str, pre_receipt_sha256: str
) -> dict[str, Any]:
    root = checkpoint.parent / "auto_resume_reconcile_v1"
    attempt = root / "attempts" / f"{jobtag}.{pre_receipt_sha256}"
    receipt_path = attempt / "receipt.json"
    success_path = attempt / "_SUCCESS"
    receipt = stable_json(receipt_path)
    if not (
        receipt.get("status") == "PASS"
        and receipt.get("mode") == "trim_to_checkpoint"
        and int(receipt.get("resume_step", receipt.get("active_metrics", {}).get("last_step", -1))) == BOOT_STEP
        and int(receipt["metrics_before"]["last_step"]) == FAILED_METRICS_STEP
        and int(receipt["discarded_tail"]["first_step"]) == BOOT_STEP + 1
        and int(receipt["discarded_tail"]["last_step"]) == FAILED_METRICS_STEP
        and int(receipt["active_metrics"]["last_step"]) == BOOT_STEP
        and receipt["bindings"]["jobtag"] == jobtag
        and receipt["bindings"]["attempt_nonce"] == pre_receipt_sha256
    ):
        raise RuntimeError("auto-resume reconciliation receipt mismatch")
    regular(success_path)
    guard_path = Path(receipt["guard_path"])
    guard = stable_json(guard_path)
    if guard.get("status") != "REPLAY_AUTHORIZED" or guard["bindings"]["jobtag"] != jobtag:
        raise RuntimeError("auto-resume replay guard mismatch")
    for path in (
        Path(receipt["metrics_before"]["backup_path"]),
        Path(receipt["discarded_tail"]["path"]),
    ):
        regular(path)
    return {
        "attempt_path": str(attempt.resolve()),
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "success_sha256": sha256_file(success_path),
        "guard_path": str(guard_path.resolve()),
        "guard_sha256": sha256_file(guard_path),
        "metrics_before_last_step": FAILED_METRICS_STEP,
        "discarded_rows": FAILED_METRICS_STEP - BOOT_STEP,
    }


def publish_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing output overwrite: {path}")
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    required = [
        args.checkpoint,
        args.metrics,
        args.config,
        args.v1_yaml,
        args.v2_yaml,
        args.recover_yaml,
        args.main_yaml,
        args.runner,
        args.addendum,
        args.boot_receipt,
        args.pre_validator_yaml,
        args.post_validator_yaml,
        Path(__file__),
    ]
    if args.phase == "post":
        if args.pre_receipt is None:
            raise RuntimeError("--pre-receipt is required for post phase")
        required.append(args.pre_receipt)
    for path in required:
        regular(path)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    if sha256_file(args.config) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("config SHA drift")
    if sha256_file(args.v1_yaml) != EXPECTED_V1_YAML_SHA256:
        raise RuntimeError("formal-v1 YAML SHA drift")
    if sha256_file(args.v2_yaml) != EXPECTED_V2_YAML_SHA256:
        raise RuntimeError("formal-v2 YAML SHA drift")
    if sha256_file(args.boot_receipt) != EXPECTED_BOOT_RECEIPT_SHA256:
        raise RuntimeError("bootstrap receipt SHA drift")
    boot = stable_json(args.boot_receipt)
    if not (
        boot.get("status") == "PASS"
        and boot.get("schema") == "stage256_20260718_formal_v2_bootstrap_checkpoint_v1"
        and int(boot.get("expected_step", -1)) == BOOT_STEP
        and int(boot.get("tag_only_exposure", -1)) == 0
        and boot["checkpoint"]["sha256"] == EXPECTED_BOOT_CHECKPOINT_SHA256
    ):
        raise RuntimeError("bootstrap receipt content mismatch")
    recover_yaml = check_gpu_yaml(args.recover_yaml, RECOVER_BASE, recover=True)
    main_yaml = check_gpu_yaml(args.main_yaml, MAIN_BASE, recover=False)
    runner = check_runner(args.runner)
    pre_validator_yaml = check_cpu_yaml(args.pre_validator_yaml, PRE_VALIDATOR_BASE, "pre")
    post_validator_yaml = check_cpu_yaml(args.post_validator_yaml, POST_VALIDATOR_BASE, "post")
    expected_checkpoint_step = BOOT_STEP if args.phase == "pre" else RECOVER_STEP
    expected_metrics_step = FAILED_METRICS_STEP if args.phase == "pre" else RECOVER_STEP
    checkpoint = validate_checkpoint(args.checkpoint, expected_checkpoint_step)
    metrics = validate_metrics(args.metrics, expected_metrics_step, boot["metrics"]["sha256"])
    if args.phase == "pre":
        if checkpoint["sha256"] != EXPECTED_BOOT_CHECKPOINT_SHA256:
            raise RuntimeError("pre-recovery checkpoint SHA differs from validated bootstrap")
        if checkpoint["identity"] != boot["checkpoint"]["identity"]:
            raise RuntimeError("pre-recovery checkpoint identity differs from bootstrap receipt")
        attempt = validate_attempt(
            args.release_root / "formal_v2_resume_attempts" / EXPECTED_V2_JOBTAG,
            EXPECTED_V2_JOBTAG,
        )
        logs = discover_logs(args.result_root, EXPECTED_V2_JOBTAG, post=False)
        replay_root = args.checkpoint.parent / "auto_resume_reconcile_v1"
        if replay_root.exists() or replay_root.is_symlink():
            raise RuntimeError(f"step-256 replay guard/evidence root is already present: {replay_root}")
        source = {
            "jobtag": EXPECTED_V2_JOBTAG,
            "terminal_jcc_status_required": "Stopped",
            "attempt": attempt,
            "logs": logs,
            "observed_last_step": FAILED_METRICS_STEP,
            "disposition": "INFRASTRUCTURE_STOP_SAFE_TO_REPLAY_FROM_VALIDATED_STEP256",
        }
        receipt = {
            "schema": "stage256_20260718_formal_v3_pre_recover256_v1",
            "status": "PASS",
            "phase": "pre",
            "expected_checkpoint_step": BOOT_STEP,
            "observed_metrics_last_step": FAILED_METRICS_STEP,
            "recover_target_step": RECOVER_STEP,
            "exact_target_step": FINAL_STEP,
            "exact_target_exposures": EXACT_EXPOSURES,
            "world_size": 64,
            "global_batch": 2304,
            "tag_only_exposure": 0,
            "checkpoint": {"path": str(args.checkpoint.resolve()), **checkpoint},
            "metrics": {"path": str(args.metrics.resolve()), **metrics},
            "source": source,
            "replay_guard_precondition": {"path": str(replay_root), "state": "ABSENT"},
        }
    else:
        assert args.pre_receipt is not None
        pre = stable_json(args.pre_receipt)
        pre_sha = sha256_file(args.pre_receipt)
        if not (
            pre.get("status") == "PASS"
            and pre.get("schema") == "stage256_20260718_formal_v3_pre_recover256_v1"
            and int(pre.get("expected_checkpoint_step", -1)) == BOOT_STEP
            and int(pre.get("recover_target_step", -1)) == RECOVER_STEP
            and int(pre.get("tag_only_exposure", -1)) == 0
        ):
            raise RuntimeError("pre-recovery receipt mismatch")
        jobtag = discover_recover_jobtag(args.result_root)
        attempt = validate_attempt(
            args.release_root / "formal_v3_recover384_attempts" / jobtag,
            jobtag,
        )
        logs = discover_logs(args.result_root, jobtag, post=True)
        reconcile = validate_reconcile(args.checkpoint, jobtag, pre_sha)
        receipt = {
            "schema": "stage256_20260718_formal_v3_recover384_checkpoint_v1",
            "status": "PASS",
            "phase": "post",
            "expected_checkpoint_step": RECOVER_STEP,
            "exact_target_step": FINAL_STEP,
            "exact_target_exposures": EXACT_EXPOSURES,
            "world_size": 64,
            "global_batch": 2304,
            "tag_only_exposure": 0,
            "checkpoint": {"path": str(args.checkpoint.resolve()), **checkpoint},
            "metrics": {"path": str(args.metrics.resolve()), **metrics},
            "pre_receipt": {"path": str(args.pre_receipt.resolve()), "sha256": pre_sha},
            "recovery": {"jobtag": jobtag, "attempt": attempt, "logs": logs, "reconcile": reconcile},
        }
    receipt.update(
        {
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "formal_v1_yaml_sha256": EXPECTED_V1_YAML_SHA256,
            "formal_v2_yaml_sha256": EXPECTED_V2_YAML_SHA256,
            "boot_receipt_sha256": EXPECTED_BOOT_RECEIPT_SHA256,
            "recover_yaml": recover_yaml,
            "main_yaml": main_yaml,
            "runner": runner,
            "addendum_sha256": sha256_file(args.addendum),
            "validator_script_sha256": sha256_file(Path(__file__)),
            "pre_validator_yaml": pre_validator_yaml,
            "post_validator_yaml": post_validator_yaml,
        }
    )
    publish_exclusive(args.output, receipt)
    print(json.dumps({"status": "PASS", "phase": args.phase, "output": str(args.output), "checkpoint_sha256": checkpoint["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
