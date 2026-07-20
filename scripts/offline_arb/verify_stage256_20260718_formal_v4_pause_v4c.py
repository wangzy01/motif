#!/usr/bin/env python3
"""Narrow log-audit correction wrapper for the formal-v4 CPU preflight."""

from __future__ import annotations

import re
import stat
from pathlib import Path
from typing import Any

import yaml

import verify_stage256_20260718_formal_v4_pause as implementation


def ordinary_identity(path: Path, *, nonempty: bool) -> dict[str, int]:
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise RuntimeError(f"not an ordinary non-symlink file: {path}")
    if nonempty and value.st_size <= 0:
        raise RuntimeError(f"required nonempty file is empty: {path}")
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def discover_v3_logs(result_root: Path) -> dict[str, Any]:
    jobtag = implementation.V3_JOBTAG
    pattern = re.compile(
        rf"^train\.{re.escape(jobtag)}\.n([0-7])\.[0-9]{{8}}T[0-9]{{6}}Z\.log$"
    )
    rows: list[tuple[int, Path]] = []
    for path in result_root.glob(f"train.{jobtag}.n*.log"):
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
        f"[train] step {implementation.METRICS_STEP}/{implementation.FINAL_STEP}",
    ):
        if token not in rank0_text:
            raise RuntimeError(f"main-v3 rank0 log lacks {token!r}")
    logs = []
    for rank, path in sorted(rows):
        identity = ordinary_identity(path, nonempty=(rank == 0))
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [
            token for token in implementation.FORBIDDEN_LOG_TOKENS if token in text
        ]
        if hits:
            raise RuntimeError(f"forbidden main-v3 log tokens in {path}: {hits}")
        logs.append(
            {
                "rank": rank,
                "path": str(path.resolve()),
                "identity": identity,
                "sha256": implementation.sha256_file(path),
            }
        )
    monitors = []
    for rank in range(8):
        matches = sorted(result_root.glob(f"nodemon.{jobtag}.n{rank}.*.log"))
        if len(matches) != 1:
            raise RuntimeError(f"main-v3 nodemon rank {rank} mismatch: {matches}")
        path = matches[0]
        identity = ordinary_identity(path, nonempty=True)
        text = path.read_text(encoding="utf-8", errors="replace")
        if "oom " in text.lower() or "avg10=0.00" not in text:
            raise RuntimeError(f"main-v3 nodemon health mismatch: {path}")
        monitors.append(
            {
                "rank": rank,
                "path": str(path.resolve()),
                "identity": identity,
                "sha256": implementation.sha256_file(path),
            }
        )
    return {
        "rank0": str(rank0.resolve()),
        "train_logs": logs,
        "node_monitors": monitors,
        "nonzero_rank_empty_stdout_allowed": True,
    }


def check_gpu_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "name": "evan-offarb-s256-20260718-causalmlf-8x8-v4d",
        "numWorkers": 8,
        "gpuPerWorker": 8,
        "cpuPerWorker": 80,
        "memGBPerWorker": 900,
        "maxRetries": 0,
    }
    observed = {key: value.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"v4d GPU YAML mismatch: {observed}")
    command = str(value.get("command", ""))
    for token in ("run_stage256_20260718_formal_v4d.sh", " main"):
        if token not in command:
            raise RuntimeError(f"v4d GPU YAML lacks {token!r}")
    return {
        "path": str(path.resolve()),
        "sha256": implementation.sha256_file(path),
        "resources": observed,
    }


def check_runner(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    required = (
        "PRE_RESUME384_SUCCESS.json",
        "formal_v4d_resume_attempts",
        "OFFLINE_ARB_AUTO_RESUME_RECONCILE=1",
        "EXPECTED_CHECKPOINT_STEP=384",
        "EXPECTED_METRICS_STEP=3000",
        "NOT_BEFORE_UNIX=1784442982",
        "trainer.checkpoint_dir=\"$CK\"",
        "dataset.gpic_payload_backend=offline_gpic_blob_v1",
    )
    missing = [token for token in required if token not in raw]
    if missing:
        raise RuntimeError(f"v4d runner lacks contracts: {missing}")
    return {"path": str(path.resolve()), "sha256": implementation.sha256_file(path)}


implementation.V4_PREFLIGHT_BASE = "evan-offarb-s256-20260718-pause-preflight-v4c"
implementation.V4_MAIN_BASE = "evan-offarb-s256-20260718-causalmlf-8x8-v4d"
implementation.NOT_BEFORE_UNIX = 1784442982
implementation.discover_v3_logs = discover_v3_logs
implementation.check_gpu_yaml = check_gpu_yaml
implementation.check_runner = check_runner


if __name__ == "__main__":
    raise SystemExit(implementation.main())
