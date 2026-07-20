#!/usr/bin/env python3
"""CPU-only gate for resuming Stage256 after the requested main-v3 pause."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_stage256_20260718_formal_v3_recovery as v3  # noqa: E402

from training.auto_resume_reconcile import _checkpoint_identity  # noqa: E402


V3_JOBTAG = "evan-offarb-s256-20260718-causalmlf-8x8-v3-qpvb"
V4_MAIN_BASE = "evan-offarb-s256-20260718-causalmlf-8x8-v4"
V4_PREFLIGHT_BASE = "evan-offarb-s256-20260718-pause-preflight-v4"
V3_RECEIPT_SHA256 = "c6d705cb68fb79ff2f4fbde5871113bb9cb7bb23418d559a6a873aa4e19b9db0"
V3_MAIN_YAML_SHA256 = "471c38da71040be72ba859a0276ed5272860fb3d69c6b2145df70fedd04dad5e"
CHECKPOINT_SHA256 = "6705da4257e2e306737457c5a8c255cdacf6b40b2494a59f6a4dab6392cc7dd6"
CHECKPOINT_DIGEST = "f439d53f7350b3096049186ea45bd347c0c0b899e162c4513e88e48972322e21"
CHECKPOINT_STEP = 384
METRICS_STEP = 3000
FINAL_STEP = 166667
EXACT_EXPOSURES = 384000768
NOT_BEFORE_UNIX = 1784448600
FORBIDDEN_LOG_TOKENS = v3.FORBIDDEN_LOG_TOKENS + ("marker collision",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v3-main-yaml", type=Path, required=True)
    parser.add_argument("--v4-main-yaml", type=Path, required=True)
    parser.add_argument("--v4-preflight-yaml", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    parser.add_argument("--v3-receipt", type=Path, required=True)
    parser.add_argument("--boot-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def regular(path: Path) -> os.stat_result:
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise RuntimeError(f"not a regular non-symlink file: {path}")
    if value.st_size <= 0:
        raise RuntimeError(f"empty file: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    before = v3.stat_identity(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if v3.stat_identity(path) != before:
        raise RuntimeError(f"JSON changed while reading: {path}")
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def check_gpu_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "name": V4_MAIN_BASE,
        "numWorkers": 8,
        "gpuPerWorker": 8,
        "cpuPerWorker": 80,
        "memGBPerWorker": 900,
        "maxRetries": 0,
    }
    observed = {key: value.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"v4 GPU YAML mismatch: {observed}")
    command = str(value.get("command", ""))
    for token in ("run_stage256_20260718_formal_v4.sh", " main"):
        if token not in command:
            raise RuntimeError(f"v4 GPU YAML lacks {token!r}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "resources": observed}


def check_cpu_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "name": V4_PREFLIGHT_BASE,
        "numWorkers": 1,
        "gpuPerWorker": 0,
        "cpuPerWorker": 32,
        "memGBPerWorker": 384,
        "maxRetries": 0,
    }
    observed = {key: value.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"v4 CPU YAML mismatch: {observed}")
    if "verify_stage256_20260718_formal_v4_pause.py" not in str(value.get("command", "")):
        raise RuntimeError("v4 CPU YAML lacks validator invocation")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "resources": observed}


def check_runner(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    required = (
        "PRE_RESUME384_SUCCESS.json",
        "formal_v4_resume_attempts",
        "OFFLINE_ARB_AUTO_RESUME_RECONCILE=1",
        "EXPECTED_CHECKPOINT_STEP=384",
        "EXPECTED_METRICS_STEP=3000",
        "NOT_BEFORE_UNIX=1784448600",
        "trainer.checkpoint_dir=\"$CK\"",
        "dataset.gpic_payload_backend=offline_gpic_blob_v1",
    )
    missing = [token for token in required if token not in raw]
    if missing:
        raise RuntimeError(f"v4 runner lacks contracts: {missing}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def discover_v3_logs(result_root: Path) -> dict[str, Any]:
    pattern = re.compile(
        rf"^train\.{re.escape(V3_JOBTAG)}\.n([0-7])\.[0-9]{{8}}T[0-9]{{6}}Z\.log$"
    )
    rows: list[tuple[int, Path]] = []
    for path in result_root.glob(f"train.{V3_JOBTAG}.n*.log"):
        match = pattern.match(path.name)
        if match:
            rows.append((int(match.group(1)), path))
    if len(rows) != 8 or {rank for rank, _ in rows} != set(range(8)):
        raise RuntimeError(f"main-v3 train-log rank mismatch: {rows}")
    rank0 = next(path for rank, path in rows if rank == 0)
    rank0_text = rank0.read_text(encoding="utf-8", errors="replace")
    for token in (
        '"mode": "already_at_checkpoint"',
        "[train] resumed from ",
        " at step 384",
        f"[train] step {METRICS_STEP}/{FINAL_STEP}",
    ):
        if token not in rank0_text:
            raise RuntimeError(f"main-v3 rank0 log lacks {token!r}")
    logs = []
    for rank, path in sorted(rows):
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [token for token in FORBIDDEN_LOG_TOKENS if token in text]
        if hits:
            raise RuntimeError(f"forbidden main-v3 log tokens in {path}: {hits}")
        logs.append(
            {
                "rank": rank,
                "path": str(path.resolve()),
                "identity": v3.stat_identity(path),
                "sha256": sha256_file(path),
            }
        )
    monitors = []
    for rank in range(8):
        matches = sorted(result_root.glob(f"nodemon.{V3_JOBTAG}.n{rank}.*.log"))
        if len(matches) != 1:
            raise RuntimeError(f"main-v3 nodemon rank {rank} mismatch: {matches}")
        path = matches[0]
        text = path.read_text(encoding="utf-8", errors="replace")
        if "oom " in text.lower() or "avg10=0.00" not in text:
            raise RuntimeError(f"main-v3 nodemon health mismatch: {path}")
        monitors.append(
            {
                "rank": rank,
                "path": str(path.resolve()),
                "identity": v3.stat_identity(path),
                "sha256": sha256_file(path),
            }
        )
    return {
        "rank0": str(rank0.resolve()),
        "train_logs": logs,
        "node_monitors": monitors,
    }


def publish_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing v4 receipt overwrite: {path}")
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    required = (
        args.checkpoint,
        args.metrics,
        args.config,
        args.v3_main_yaml,
        args.v4_main_yaml,
        args.v4_preflight_yaml,
        args.runner,
        args.addendum,
        args.v3_receipt,
        args.boot_receipt,
        Path(__file__),
        Path(v3.__file__),
    )
    for path in required:
        regular(path)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"v4 output already exists: {args.output}")
    if sha256_file(args.config) != v3.EXPECTED_CONFIG_SHA256:
        raise RuntimeError("config SHA drift")
    if sha256_file(args.v3_main_yaml) != V3_MAIN_YAML_SHA256:
        raise RuntimeError("main-v3 YAML SHA drift")
    if sha256_file(args.v3_receipt) != V3_RECEIPT_SHA256:
        raise RuntimeError("main-v3 source receipt SHA drift")
    if sha256_file(args.boot_receipt) != v3.EXPECTED_BOOT_RECEIPT_SHA256:
        raise RuntimeError("bootstrap receipt SHA drift")

    v3_receipt = load_json(args.v3_receipt)
    boot_receipt = load_json(args.boot_receipt)
    if not (
        v3_receipt.get("status") == "PASS"
        and v3_receipt.get("schema") == "stage256_20260718_formal_v3_recover384_checkpoint_v1"
        and int(v3_receipt.get("expected_checkpoint_step", -1)) == CHECKPOINT_STEP
        and int(v3_receipt.get("tag_only_exposure", -1)) == 0
        and int(v3_receipt.get("world_size", -1)) == 64
        and int(v3_receipt.get("global_batch", -1)) == 2304
        and int(v3_receipt.get("exact_target_step", -1)) == FINAL_STEP
        and int(v3_receipt.get("exact_target_exposures", -1)) == EXACT_EXPOSURES
    ):
        raise RuntimeError("main-v3 source receipt content mismatch")
    if not (
        boot_receipt.get("status") == "PASS"
        and boot_receipt.get("schema") == "stage256_20260718_formal_v2_bootstrap_checkpoint_v1"
    ):
        raise RuntimeError("bootstrap receipt content mismatch")

    checkpoint = v3.validate_checkpoint(args.checkpoint, CHECKPOINT_STEP)
    if checkpoint["sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("step-384 checkpoint SHA mismatch")
    if checkpoint["sha256"] != v3_receipt["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint SHA differs from v3 receipt")
    if checkpoint["identity"] != v3_receipt["checkpoint"]["identity"]:
        raise RuntimeError("checkpoint identity differs from v3 receipt")

    metrics = v3.validate_metrics(
        args.metrics, METRICS_STEP, boot_receipt["metrics"]["sha256"]
    )
    raw_metrics = args.metrics.read_bytes()
    lines = raw_metrics.splitlines(keepends=True)
    prefix_384_sha256 = hashlib.sha256(b"".join(lines[:CHECKPOINT_STEP])).hexdigest()
    if prefix_384_sha256 != v3_receipt["metrics"]["sha256"]:
        raise RuntimeError("metrics 1..384 prefix differs from v3 receipt")
    if any(float(metrics["last"].get(key, -1)) != 0.0 for key in v3.ZERO_METRIC_KEYS):
        raise RuntimeError("step-3000 semantic/GPIC snapshot is nonzero or incomplete")

    attempt = v3.validate_attempt(
        args.release_root / "formal_v3_resume_attempts" / V3_JOBTAG, V3_JOBTAG
    )
    logs = discover_v3_logs(args.result_root)
    sampled_identity = _checkpoint_identity(args.checkpoint, CHECKPOINT_STEP)
    if sampled_identity["identity_digest"] != CHECKPOINT_DIGEST:
        raise RuntimeError("step-384 checkpoint identity digest mismatch")
    claim = args.checkpoint.parent / "auto_resume_reconcile_v1" / "claims" / f"{CHECKPOINT_DIGEST}.json"
    if claim.exists() or claim.is_symlink():
        raise RuntimeError(f"step-384 replay claim already exists: {claim}")

    main_yaml = check_gpu_yaml(args.v4_main_yaml)
    preflight_yaml = check_cpu_yaml(args.v4_preflight_yaml)
    runner = check_runner(args.runner)
    receipt = {
        "schema": "stage256_20260718_formal_v4_pre_resume384_v1",
        "status": "PASS",
        "expected_checkpoint_step": CHECKPOINT_STEP,
        "observed_metrics_last_step": METRICS_STEP,
        "exact_target_step": FINAL_STEP,
        "exact_target_exposures": EXACT_EXPOSURES,
        "world_size": 64,
        "global_batch": 2304,
        "tag_only_exposure": 0,
        "not_before": {
            "iso_cst": "2026-07-19T16:10:00+08:00",
            "unix": NOT_BEFORE_UNIX,
        },
        "checkpoint": {"path": str(args.checkpoint.resolve()), **checkpoint},
        "metrics": {
            "path": str(args.metrics.resolve()),
            **metrics,
            "prefix_1_384_sha256": prefix_384_sha256,
        },
        "uncheckpointed_tail": {
            "first_step": CHECKPOINT_STEP + 1,
            "last_step": METRICS_STEP,
            "rows": METRICS_STEP - CHECKPOINT_STEP,
            "disposition": "PRESERVE_TRIM_AND_REPLAY_FROM_VALIDATED_STEP384",
        },
        "source": {
            "jobtag": V3_JOBTAG,
            "terminal_jcc_status_required_before_submit": "Stopped",
            "attempt": attempt,
            "logs": logs,
        },
        "replay_guard_precondition": {
            "checkpoint_identity_digest": CHECKPOINT_DIGEST,
            "path": str(claim),
            "state": "ABSENT",
        },
        "v3_receipt": {
            "path": str(args.v3_receipt.resolve()),
            "sha256": V3_RECEIPT_SHA256,
        },
        "config_sha256": v3.EXPECTED_CONFIG_SHA256,
        "v3_main_yaml": {
            "path": str(args.v3_main_yaml.resolve()),
            "sha256": V3_MAIN_YAML_SHA256,
        },
        "main_yaml": main_yaml,
        "preflight_yaml": preflight_yaml,
        "runner": runner,
        "addendum_sha256": sha256_file(args.addendum),
        "validator_script_sha256": sha256_file(Path(__file__)),
        "v3_validator_script_sha256": sha256_file(Path(v3.__file__)),
    }
    publish_exclusive(args.output, receipt)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output),
                "checkpoint_sha256": checkpoint["sha256"],
                "metrics_last_step": METRICS_STEP,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
