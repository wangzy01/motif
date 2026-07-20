#!/usr/bin/env python3
"""Strict per-arm analyzer for the CPU W16 GPIC L16/L8/L8/L16 selector."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from training.dataline_offline.planner_v1 import OfflinePlan


MASTER_SCHEMA = "offline_arb_w16_gpic_abba_master_aggregate_v1"
PREFLIGHT_SCHEMA = "offline_arb_w16_gpic_abba_preflight_v1"
ARM_CONTRACT_SCHEMA = "offline_arb_w16_gpic_abba_arm_contract_v1"
ARM_AGGREGATE_SCHEMA = "offline_arb_w16_gpic_abba_arm_aggregate_v1"
NODE_MANIFEST_SCHEMA = "offline_arb_w16_gpic_abba_node_manifest_v1"
ANALYSIS_SCHEMA = "offline_arb_w16_gpic_abba_analysis_v1"
TELEMETRY_SCHEMA = "offline_arb_io_profile_v1"
WORLD_SIZE = 16
LOCAL_WORLD_SIZE = 8
NODES = 2
PER_RANK_BS = 36
PREFETCH_STEPS = 4
SLICE_STEPS = 192
STEADY_OFFSET = 40
TARGET_SECONDS_PER_STEP = 2.592
CAPACITY_HEADROOM_RATIO = 1.0 / 0.9
WAIT_MEAN_MAX_S = TARGET_SECONDS_PER_STEP * 0.9
WAIT_P95_MAX_S = TARGET_SECONDS_PER_STEP * 2.0
WAIT_MAX_S = TARGET_SECONDS_PER_STEP * 4.0
WAIT_OVER_TARGET_FRACTION_MAX = 0.10
WINDOW_MEAN_MAX_S = PREFETCH_STEPS * TARGET_SECONDS_PER_STEP * 0.9
WINDOW_TAIL_MAX_S = PREFETCH_STEPS * TARGET_SECONDS_PER_STEP * 2.0
RUNNER_ELAPSED_PER_STEP_MAX_S = 2.75
PAIR_DRIFT_MAX = 0.05
WAIT_LOG_THRESHOLD_S = 0.000001
FRESH_MARKER_MAX_AGE_S = 1800
FRESH_MARKER_MAX_FUTURE_S = 30
EXPECTED_PLAN_ID = "stage256_offline_exact_v1_w16_gpuval"
EXPECTED_DOMAINS = ("anime", "monet_real", "gpic", "synthetic", "textatlas")
ARMS = (
    ("l16_a", 16, 0, 192),
    ("l8_a", 8, 192, 384),
    ("l8_b", 8, 384, 576),
    ("l16_b", 16, 576, 768),
)
THREAD_GROUPS = {"L8": ("l8_a", "l8_b"), "L16": ("l16_a", "l16_b")}
GATE_RELATIVE_FILES = (
    "scripts/offline_arb/run_w16_gpic_abba_raw_gate.py",
    "jobs/offline_arb_w16_gpic_abba_raw_gate.yaml",
    "scripts/offline_arb/analyze_w16_gpic_abba_raw_gate.py",
    "jobs/offline_arb_w16_gpic_abba_analyze.yaml",
)
PYTHON = Path("/fsx/evan/mot_code/venv-gpu/bin/python")
ALLOWED_RECORD_TYPES = {"identity", "window", "wait", "state", "closure"}
SHA_RE = re.compile(r"[0-9a-f]{64}")
RANK_FILE_RE = re.compile(r"io_profile\.rank(?P<rank>[0-9]{4})\.jsonl")
STEP_LINE_RE = re.compile(
    r"W16_ABBA_RAW_STEP_RETIRED arm=(?P<arm>[a-z0-9_]+) rank=(?P<rank>[0-9]+) "
    r"step=(?P<step>[0-9]+) relative_step=(?P<relative>[0-9]+) "
    r"items=(?P<items>[0-9]+) raw_bytes=(?P<raw_bytes>[0-9]+) "
    r"wait_s=(?P<wait_s>[0-9]+\.[0-9]{9}) scheduled_until=(?P<scheduled>[0-9]+)"
)
PASS_LINE_RE = re.compile(
    r"W16_ABBA_RAW_RANK_PASS arm=(?P<arm>[a-z0-9_]+) rank=(?P<rank>[0-9]+) "
    r"start_step=(?P<start>[0-9]+) stop_step=(?P<stop>[0-9]+) "
    r"retired_steps=(?P<retired>[0-9]+) per_rank_batch_size=(?P<bs>[0-9]+) "
    r"raw_items=(?P<items>[0-9]+) raw_bytes=(?P<raw_bytes>[0-9]+) "
    r"wait_total_s=(?P<wait_total>[0-9]+\.[0-9]{9}) "
    r"wait_max_s=(?P<wait_max>[0-9]+\.[0-9]{9}) "
    r"elapsed_s=(?P<elapsed>[0-9]+\.[0-9]{9}) decode_calls=(?P<decode>[0-9]+)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_attempt_nonce(run_id: str, jobtag: str) -> str:
    return canonical_sha256(
        {
            "schema_version": "offline_arb_w16_gpic_abba_attempt_nonce_v1",
            "run_id": run_id,
            "jobtag": jobtag,
        }
    )


def expected_arm_nonce(
    attempt: str, arm: str, threads: int, start: int, stop: int
) -> str:
    return canonical_sha256(
        {
            "schema_version": "offline_arb_w16_gpic_abba_arm_nonce_v1",
            "attempt_nonce": attempt,
            "arm": arm,
            "threads": threads,
            "start_step": start,
            "stop_step": stop,
        }
    )


def validate_marker_time(marker_raw: str, contract_created_at: int, label: str) -> int:
    if not marker_raw.isdigit():
        raise ValueError(f"{label} timestamp malformed")
    marker_created_at = int(marker_raw)
    age = contract_created_at - marker_created_at
    if age < -FRESH_MARKER_MAX_FUTURE_S or age > FRESH_MARKER_MAX_AGE_S:
        raise ValueError(f"{label} content timestamp is stale: age={age}")
    return marker_created_at


def require_file(path: Path, label: str, *, nonempty: bool = True) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or (nonempty and path.stat().st_size <= 0)
    ):
        raise ValueError(f"{label} missing/empty/non-regular/symlink: {path}")


def require_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} missing/non-directory/symlink: {path}")


def read_object(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def strict_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{label} is not finite/nonnegative")
    return result


def exact_json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact_json_equal(left, right) for left, right in zip(actual, expected)
        )
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            exact_json_equal(actual[key], expected[key]) for key in expected
        )
    return bool(actual == expected)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate_sha_artifact(path: Path, declaration: Any, label: str, *, named: bool) -> str:
    if not isinstance(declaration, dict):
        raise ValueError(f"{label} declaration malformed")
    expected_keys = {"name", "bytes", "sha256"} if named else {"path", "bytes", "sha256"}
    if set(declaration) != expected_keys:
        raise ValueError(f"{label} declaration key set drift")
    declared_bytes = strict_int(declaration["bytes"], f"{label}.bytes", minimum=0)
    require_file(path, label, nonempty=declared_bytes > 0)
    if named:
        if declaration["name"] != path.name:
            raise ValueError(f"{label} name mismatch")
    elif declaration["path"] != str(path.resolve(strict=True)):
        raise ValueError(f"{label} canonical path mismatch")
    if declared_bytes != path.stat().st_size:
        raise ValueError(f"{label} size mismatch")
    expected_sha = declaration["sha256"]
    if not isinstance(expected_sha, str) or SHA_RE.fullmatch(expected_sha) is None:
        raise ValueError(f"{label} SHA malformed")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise ValueError(f"{label} SHA mismatch")
    return actual


def fingerprint(path: Path, *, directory: bool = False) -> dict[str, int]:
    if path.is_symlink() or (not path.is_dir() if directory else not path.is_file()):
        raise ValueError(f"artifact type/symlink drift: {path}")
    metadata = path.lstat()
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "size_bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
    }


def build_animefreq_ledger(root: Path) -> dict[str, Any]:
    require_directory(root, "anime-frequency root")
    header = read_object(root / "header.json", "anime-frequency header")
    declared = header.get("files")
    if (
        header.get("schema_version") != "offline_arb_animefreq_v1"
        or not isinstance(declared, dict)
        or len(declared) != 17
        or header.get("shard_count") != len(declared)
    ):
        raise ValueError("anime-frequency header/file-count contract drift")
    if {path.name for path in root.iterdir()} != {"header.json", "_SUCCESS", *declared}:
        raise ValueError("anime-frequency root exact-set drift")
    files = {}
    for name in sorted(declared):
        path = root / name
        require_file(path, f"anime-frequency file {name}")
        digest = sha256_file(path)
        entry = declared[name]
        if not isinstance(entry, dict) or entry.get("sha256") != digest:
            raise ValueError(f"anime-frequency SHA/header mismatch: {name}")
        files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "fingerprint": fingerprint(path),
        }
    success = root / "_SUCCESS"
    require_file(success, "anime-frequency _SUCCESS", nonempty=False)
    return {
        "schema_version": "offline_arb_w16_animefreq_ledger_v1",
        "root": str(root),
        "root_fingerprint": fingerprint(root, directory=True),
        "header_sha256": sha256_file(root / "header.json"),
        "success": {
            "path": str(success),
            "bytes": success.stat().st_size,
            "sha256": sha256_file(success),
            "fingerprint": fingerprint(success),
        },
        "files": files,
    }


def validate_sidecar_baseline(root: Path, baseline: Any) -> dict[str, Any]:
    require_directory(root, "sidecar baseline root")
    if not isinstance(baseline, dict) or baseline.get("root") != str(root):
        raise ValueError("sidecar baseline identity malformed")
    files = baseline.get("files")
    directories = baseline.get("directories")
    if not isinstance(files, dict) or not isinstance(directories, dict):
        raise ValueError("sidecar baseline ledger malformed")
    domain_names = set(directories) - {"root"}
    if {path.name for path in root.iterdir()} != {"header.json", "_SUCCESS", *domain_names}:
        raise ValueError("sidecar root exact-set changed")
    live_directories = {}
    for label, declaration in directories.items():
        path = root if label == "root" else root / label
        if not isinstance(declaration, dict) or declaration.get("path") != str(path):
            raise ValueError(f"sidecar directory declaration malformed: {label}")
        current = fingerprint(path, directory=True)
        if declaration.get("fingerprint") != current:
            raise ValueError(f"sidecar directory fingerprint changed: {label}")
        live_directories[label] = {"path": str(path), "fingerprint": current}
    expected_by_domain: dict[str, set[str]] = {domain: set() for domain in domain_names}
    live_files = {}
    for name, declaration in sorted(files.items()):
        if not isinstance(name, str) or not isinstance(declaration, dict):
            raise ValueError("sidecar file ledger entry malformed")
        path = root / name
        if declaration.get("path") != str(path) or path.parent.name not in expected_by_domain:
            raise ValueError(f"sidecar file canonical path drift: {name}")
        require_file(path, f"sidecar ledger file {name}")
        current = fingerprint(path)
        if (
            declaration.get("bytes") != path.stat().st_size
            or declaration.get("fingerprint") != current
            or not isinstance(declaration.get("sha256"), str)
            or SHA_RE.fullmatch(declaration["sha256"]) is None
        ):
            raise ValueError(f"sidecar file size/fingerprint/SHA drift: {name}")
        expected_by_domain[path.parent.name].add(path.name)
        live_files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": declaration["sha256"],
            "fingerprint": current,
        }
    for domain, names in expected_by_domain.items():
        if {path.name for path in (root / domain).iterdir()} != names:
            raise ValueError(f"sidecar domain exact-set changed: {domain}")
    if len(live_files) != baseline.get("file_count") or sum(
        entry["bytes"] for entry in live_files.values()
    ) != baseline.get("total_bytes"):
        raise ValueError("sidecar live ledger count/bytes do not close")
    return {
        "directories": live_directories,
        "files": live_files,
        "file_count": len(live_files),
        "total_bytes": sum(entry["bytes"] for entry in live_files.values()),
    }


def validate_ready_snapshot(value: Any, label: str, domains: set[str]) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} ready snapshot malformed")
    scheduled = strict_int(value.get("scheduled_depth_steps"), f"{label}.scheduled", minimum=0)
    contiguous = strict_int(value.get("contiguous_ready_steps"), f"{label}.contiguous", minimum=0)
    if contiguous > scheduled:
        raise ValueError(f"{label} contiguous depth exceeds scheduled depth")
    ready = value.get("domain_ready_items")
    if not isinstance(ready, dict) or set(ready) != domains:
        raise ValueError(f"{label} domain-ready set mismatch")
    for domain, count in ready.items():
        strict_int(count, f"{label}.{domain}", minimum=0)


def validate_image_detail(
    detail: Any, *, domain: str, n: int, image_s: float, workers: int, label: str
) -> None:
    if not isinstance(detail, dict):
        raise ValueError(f"{label} image detail malformed")
    kind = detail.get("kind")
    if domain == "gpic":
        if kind != "gpic_in_memory" or detail.get("files") != n or detail.get("workers") != workers:
            raise ValueError(f"{label} GPIC detail mismatch")
        unique = strict_int(detail.get("unique_files"), f"{label}.unique_files", minimum=1)
        if unique > n:
            raise ValueError(f"{label} GPIC unique count exceeds n")
        strict_int(detail.get("bytes"), f"{label}.bytes", minimum=1)
        required = ("open", "fstat", "path_stat", "read", "file_total")
        maximum_key = "max_file_s"
    elif domain in {"monet_real", "synthetic"}:
        if kind != "coalesced_tar":
            raise ValueError(f"{label} tar detail kind mismatch")
        unique_tars = strict_int(detail.get("unique_tars"), f"{label}.unique_tars", minimum=1)
        requests = strict_int(detail.get("locator_requests"), f"{label}.requests", minimum=1)
        unique_locators = strict_int(
            detail.get("unique_locators"), f"{label}.unique_locators", minimum=1
        )
        runs = strict_int(detail.get("runs"), f"{label}.runs", minimum=1)
        requested = strict_int(detail.get("requested_bytes"), f"{label}.requested", minimum=1)
        member = strict_int(detail.get("member_bytes"), f"{label}.member", minimum=1)
        actual = strict_int(detail.get("actual_read_bytes"), f"{label}.actual", minimum=1)
        if (
            requests != n
            or unique_tars > unique_locators
            or unique_locators > requests
            or runs > unique_locators
            or member < requested
            or actual < requested
            or detail.get("workers") != min(8, unique_tars)
        ):
            raise ValueError(f"{label} tar accounting mismatch")
        required = ("open", "seek", "read", "tar_total")
        maximum_key = "max_tar_s"
    elif domain == "anime":
        if kind != "anime_fetch" or detail.get("files") != n:
            raise ValueError(f"{label} anime detail mismatch")
        return
    elif domain == "textatlas":
        if kind != "textatlas_parquet" or detail.get("files") != n:
            raise ValueError(f"{label} textatlas detail mismatch")
        return
    else:
        raise ValueError(f"unsupported domain: {domain}")
    thread_sum = detail.get("thread_sum_s")
    operation_max = detail.get("operation_max_s")
    if not isinstance(thread_sum, dict) or not isinstance(operation_max, dict):
        raise ValueError(f"{label} timing maps malformed")
    for key in required:
        total = number(thread_sum.get(key), f"{label}.thread_sum.{key}")
        maximum = number(operation_max.get(key), f"{label}.operation_max.{key}")
        if maximum > total + 1e-9:
            raise ValueError(f"{label} operation max exceeds sum")
    wall = number(detail.get("wall_s"), f"{label}.wall", positive=True)
    number(detail.get(maximum_key), f"{label}.{maximum_key}")
    number(detail.get("effective_concurrency"), f"{label}.effective_concurrency")
    if wall > image_s + 0.01:
        raise ValueError(f"{label} detail wall exceeds image time")


class PlanWindowCounts:
    def __init__(self, plan: OfflinePlan):
        self.plan = plan
        self.domains = tuple(str(value) for value in plan.header["domains"])
        totals = [
            sum(int(row[index]) for row in plan.header["N"])
            for index in range(len(self.domains))
        ]
        running = 0
        self.ends: list[int] = []
        for total in totals:
            running += total
            self.ends.append(running)
        if running != int(plan.header["catalog_size"]):
            raise ValueError("plan domain ranges do not close")
        self.cache: dict[tuple[int, int], dict[str, int]] = {}

    def get(self, start: int, rank: int) -> dict[str, int]:
        key = (start, rank)
        if key in self.cache:
            return self.cache[key]
        if start < 0 or start % PREFETCH_STEPS:
            raise ValueError(f"invalid plan window start {start}")
        counts = {domain: 0 for domain in self.domains}
        for step in range(start, start + PREFETCH_STEPS):
            _bucket, rows = self.plan.batch(step, rank)
            for raw in rows.tolist():
                row_id = int(raw)
                domain_index = bisect.bisect_right(self.ends, row_id)
                if row_id < 0 or domain_index >= len(self.domains):
                    raise ValueError("plan row id outside domain ranges")
                counts[self.domains[domain_index]] += 1
        if sum(counts.values()) != PREFETCH_STEPS * PER_RANK_BS:
            raise ValueError("plan window counts do not close")
        self.cache[key] = counts
        return counts


def parse_runner_log(
    path: Path, *, arm: str, node_rank: int, start: int, stop: int
) -> dict[int, dict[str, Any]]:
    require_file(path, f"{arm} node {node_rank} runner log")
    ranks = range(node_rank * LOCAL_WORLD_SIZE, (node_rank + 1) * LOCAL_WORLD_SIZE)
    steps: dict[int, list[dict[str, Any]]] = {rank: [] for rank in ranks}
    passes: dict[int, list[dict[str, Any]]] = {rank: [] for rank in ranks}
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            step_match = STEP_LINE_RE.fullmatch(line)
            if step_match is not None:
                rank = int(step_match.group("rank"))
                if rank not in steps or step_match.group("arm") != arm:
                    raise ValueError(f"{arm} node {node_rank} foreign step sentinel")
                steps[rank].append(
                    {
                        "line": line_number,
                        "step": int(step_match.group("step")),
                        "relative": int(step_match.group("relative")),
                        "items": int(step_match.group("items")),
                        "raw_bytes": int(step_match.group("raw_bytes")),
                        "wait_s": float(step_match.group("wait_s")),
                        "scheduled": int(step_match.group("scheduled")),
                    }
                )
                continue
            pass_match = PASS_LINE_RE.fullmatch(line)
            if pass_match is not None:
                rank = int(pass_match.group("rank"))
                if rank not in passes or pass_match.group("arm") != arm:
                    raise ValueError(f"{arm} node {node_rank} foreign PASS sentinel")
                passes[rank].append(
                    {
                        "start": int(pass_match.group("start")),
                        "stop": int(pass_match.group("stop")),
                        "retired": int(pass_match.group("retired")),
                        "bs": int(pass_match.group("bs")),
                        "items": int(pass_match.group("items")),
                        "raw_bytes": int(pass_match.group("raw_bytes")),
                        "wait_total": float(pass_match.group("wait_total")),
                        "wait_max": float(pass_match.group("wait_max")),
                        "elapsed": float(pass_match.group("elapsed")),
                        "decode": int(pass_match.group("decode")),
                    }
                )
                continue
            if "W16_ABBA_RAW_STEP_RETIRED" in line or "W16_ABBA_RAW_RANK_PASS" in line:
                raise ValueError(f"{arm} malformed/interleaved sentinel at line {line_number}")
    result: dict[int, dict[str, Any]] = {}
    for rank in ranks:
        records = steps[rank]
        if len(records) != SLICE_STEPS:
            raise ValueError(f"{arm} rank {rank} step count mismatch")
        for relative, record in enumerate(records):
            step = start + relative
            expected_scheduled = min(
                stop, start + PREFETCH_STEPS * (((relative + 1) // PREFETCH_STEPS) + 2)
            )
            if (
                record["step"] != step
                or record["relative"] != relative
                or record["items"] != PER_RANK_BS
                or record["raw_bytes"] <= 0
                or record["scheduled"] != expected_scheduled
            ):
                raise ValueError(f"{arm} rank {rank} step contract mismatch at {step}")
        if len(passes[rank]) != 1:
            raise ValueError(f"{arm} rank {rank} must have exactly one PASS")
        summary = passes[rank][0]
        waits = [record["wait_s"] for record in records]
        if (
            summary["start"] != start
            or summary["stop"] != stop
            or summary["retired"] != SLICE_STEPS
            or summary["bs"] != PER_RANK_BS
            or summary["items"] != SLICE_STEPS * PER_RANK_BS
            or summary["raw_bytes"] != sum(record["raw_bytes"] for record in records)
            or summary["decode"] != 0
            or summary["elapsed"] <= 0
            or summary["elapsed"] + 0.05 < sum(waits)
            or abs(summary["wait_total"] - sum(waits)) > 0.01
            or abs(summary["wait_max"] - max(waits)) > 0.01
        ):
            raise ValueError(f"{arm} rank {rank} PASS accounting mismatch")
        result[rank] = {"steps": records, "summary": summary}
    return result


def load_telemetry(
    path: Path,
    *,
    arm: str,
    rank: int,
    threads: int,
    start: int,
    stop: int,
    nonce: str,
    attempt_nonce: str,
    host: str,
    jobtag: str,
    plan_id: str,
    plan_hash: str,
    catalog_hash: str,
    sidecar_header: str,
    sidecar_release: str,
    domains: set[str],
    runner_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    records = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith("\n") or not raw.strip():
                raise ValueError(f"{arm} rank {rank} blank/unterminated telemetry")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{arm} rank {rank} telemetry record malformed")
            sequence = strict_int(value.get("sequence"), "telemetry sequence", minimum=0)
            record_rank = strict_int(value.get("rank"), "telemetry rank", minimum=0)
            if (
                value.get("schema_version") != TELEMETRY_SCHEMA
                or sequence != line_number - 1
                or record_rank != rank
                or value.get("record_type") not in ALLOWED_RECORD_TYPES
            ):
                raise ValueError(f"{arm} rank {rank} telemetry schema/sequence drift")
            records.append(value)
    identities = [value for value in records if value["record_type"] == "identity"]
    closures = [value for value in records if value["record_type"] == "closure"]
    if len(identities) != 1 or identities[0] is not records[0]:
        raise ValueError(f"{arm} rank {rank} identity is not unique first record")
    if len(closures) != 1 or closures[0] is not records[-1]:
        raise ValueError(f"{arm} rank {rank} closure is not unique last record")
    node_rank = rank // LOCAL_WORLD_SIZE
    telemetry_root = (
        f"/local/evan/offline_arb_w16_gpic_abba/{jobtag}/"
        f"{attempt_nonce}/{arm}/{nonce}/node{node_rank}"
    )
    identity = identities[0]
    actual_root = identity.get("telemetry_root")
    if actual_root != telemetry_root:
        raise ValueError(f"{arm} rank {rank} telemetry root mismatch: {telemetry_root}")
    expected_identity = {
        "world_size": WORLD_SIZE,
        "local_rank": rank % LOCAL_WORLD_SIZE,
        "local_world_size": LOCAL_WORLD_SIZE,
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "catalog_hash": catalog_hash,
        "sidecar_header_sha256": sidecar_header,
        "sidecar_release_digest": sidecar_release,
        "attempt_jobtag": jobtag,
        "attempt_nonce": nonce,
        "attempt_host": host,
        "telemetry_root": actual_root,
        "telemetry_path": f"{actual_root}/io_profile.rank{rank:04d}.jsonl",
        "prefetch_steps": PREFETCH_STEPS,
        "gpic_threads_per_node": threads,
        "gpic_workers_per_rank": threads // LOCAL_WORLD_SIZE,
        "tar_workers_per_rank": 8,
        "gpic_payload_in_memory": True,
    }
    for key, wanted in expected_identity.items():
        if not exact_json_equal(identity.get(key), wanted):
            raise ValueError(f"{arm} rank {rank} identity {key} mismatch")
    strict_int(identity.get("pid"), f"{arm} rank {rank} pid", minimum=1)
    closure = closures[0]
    before = len(records) - 1
    if (
        strict_int(closure.get("next_step"), "closure next_step", minimum=0) != stop
        or strict_int(closure.get("records_before_closure"), "writer records", minimum=0)
        != before
        or strict_int(closure.get("records_enqueued_before_closure"), "producer records", minimum=0)
        != before
        or closure.get("telemetry_fatal") is not False
        or strict_int(closure.get("dropped_emits"), "dropped emits", minimum=0) != 0
        or strict_int(closure.get("runtime_errors"), "runtime errors", minimum=0) != 0
        or strict_int(closure.get("rejected_emits_before_closure"), "rejected emits", minimum=0)
        != 0
        or strict_int(
            closure.get("rejected_emits_observed_by_writer"), "writer rejected", minimum=0
        )
        != 0
    ):
        raise ValueError(f"{arm} rank {rank} closure accounting mismatch")
    number(closure.get("producer_enqueue_seconds_before_closure"), "producer enqueue")
    number(closure.get("writer_seconds_before_closure"), "writer seconds")

    window_starts = set(range(start, stop, PREFETCH_STEPS))
    windows: dict[tuple[int, str], dict[str, Any]] = {}
    states = []
    waits: dict[int, dict[str, Any]] = {}
    for value in records[1:-1]:
        record_type = value["record_type"]
        if record_type == "window":
            window_start = strict_int(value.get("start"), "window start", minimum=0)
            window_stop = strict_int(value.get("stop"), "window stop", minimum=1)
            domain = value.get("domain")
            n = strict_int(value.get("n"), "window n", minimum=1)
            if (
                domain not in domains
                or window_start not in window_starts
                or window_stop != window_start + PREFETCH_STEPS
                or value.get("caption_backend") != "sidecar"
                or value.get("caption_workers") != 16
            ):
                raise ValueError(f"{arm} rank {rank} window identity mismatch")
            key = (window_start, str(domain))
            if key in windows:
                raise ValueError(f"{arm} rank {rank} duplicate window {key}")
            catalog_s = number(value.get("catalog_s"), "window catalog")
            caption_s = number(value.get("caption_s"), "window caption")
            image_s = number(value.get("image_s"), "window image")
            total_s = number(value.get("total_s"), "window total", positive=True)
            if total_s + 1e-6 < catalog_s + caption_s + image_s:
                raise ValueError(f"{arm} rank {rank} window components exceed total")
            sources = value.get("sub_source_counts")
            if not isinstance(sources, dict):
                raise ValueError("window source counts malformed")
            if any(not isinstance(source, str) or not source for source in sources):
                raise ValueError("window source-count key malformed")
            source_total = sum(
                strict_int(count, "window source count", minimum=1)
                for count in sources.values()
            )
            if domain in {"monet_real", "gpic", "synthetic"} and source_total != n:
                raise ValueError("window source counts do not close")
            validate_image_detail(
                value.get("image_detail"),
                domain=str(domain),
                n=n,
                image_s=image_s,
                workers=threads // LOCAL_WORLD_SIZE,
                label=f"{arm} rank {rank} {key}",
            )
            windows[key] = value
        elif record_type == "state":
            consumed = strict_int(value.get("consumed_step"), "state consumed", minimum=0)
            if strict_int(value.get("next_step"), "state next", minimum=1) != consumed + 1:
                raise ValueError("state next-step mismatch")
            validate_ready_snapshot(value, f"{arm} rank {rank} state", domains)
            states.append(consumed)
        elif record_type == "wait":
            step = strict_int(value.get("step"), "wait step", minimum=0)
            wait_s = number(value.get("condition_wait_s"), "condition wait", positive=True)
            if step < start or step >= stop or step in waits or wait_s <= WAIT_LOG_THRESHOLD_S:
                raise ValueError("wait step/threshold/duplicate mismatch")
            strict_int(value.get("condition_wakeups"), "wait wakeups", minimum=1)
            expected_items = strict_int(value.get("expected_items"), "wait expected", minimum=1)
            entry_ready = strict_int(value.get("entry_ready_items"), "wait entry", minimum=0)
            exit_ready = strict_int(value.get("exit_ready_items"), "wait exit", minimum=0)
            if expected_items != PER_RANK_BS or entry_ready > PER_RANK_BS or exit_ready != PER_RANK_BS:
                raise ValueError("wait item accounting mismatch")
            relative = step - start
            if wait_s > float(runner_steps[relative]["wait_s"]) + 0.05:
                raise ValueError("telemetry wait exceeds runner wait")
            validate_ready_snapshot(value.get("entry"), "wait entry snapshot", domains)
            validate_ready_snapshot(value.get("exit"), "wait exit snapshot", domains)
            waits[step] = value
    if states != list(range(start, stop)):
        raise ValueError(f"{arm} rank {rank} state sequence is not exact slice")
    if list(waits) != sorted(waits):
        raise ValueError(f"{arm} rank {rank} wait sequence unordered")
    return {
        "record_counts": dict(sorted(Counter(value["record_type"] for value in records).items())),
        "windows": windows,
        "waits": waits,
    }


def add_gate(
    gates: list[dict[str, Any]], name: str, value: float, operator: str, threshold: float
) -> None:
    if operator not in {"<=", ">="}:
        raise ValueError(f"unsupported gate operator: {operator}")
    passed = value <= threshold if operator == "<=" else value >= threshold
    gates.append(
        {"name": name, "value": value, "operator": operator, "threshold": threshold, "pass": passed}
    )


def positive_pair_drift(left: float, right: float) -> float:
    left_value = number(left, "pair drift left", positive=True)
    right_value = number(right, "pair drift right", positive=True)
    return abs(left_value - right_value) / max(left_value, right_value)


def analyze_arm(
    *,
    arm_spec: tuple[str, int, int, int],
    arm_aggregate_path: Path,
    arm_aggregate: dict[str, Any],
    preflight: dict[str, Any],
    plan: OfflinePlan,
    counts: PlanWindowCounts,
    evidence_root: Path,
    closure: dict[str, tuple[Path, str]],
) -> dict[str, Any]:
    arm, threads, start, stop = arm_spec
    expected_arm = {
        "schema_version": ARM_AGGREGATE_SCHEMA,
        "status": "PASS",
        "jobtag": preflight["jobtag"],
        "attempt_nonce": preflight["attempt_nonce"],
        "arm": arm,
        "threads": threads,
        "start_step": start,
        "stop_step": stop,
        "world_size": WORLD_SIZE,
        "local_world_size": LOCAL_WORLD_SIZE,
        "per_rank_bs": PER_RANK_BS,
        "prefetch_steps": PREFETCH_STEPS,
        "preflight_sha256": sha256_file(Path(preflight["_path"])),
        "animefreq_ledger_sha256": canonical_sha256(preflight["animefreq_ledger"]),
        "sidecar_baseline_sha256": canonical_sha256(preflight["sidecar_baseline"]),
        "sidecar_full_verify": preflight["sidecar_full_verify"],
        "sidecar_header_sha256": preflight["sidecar_header_sha256"],
        "sidecar_release_digest": preflight["sidecar_release_digest"],
        "sidecar_snapshot_fingerprints_sha256": preflight[
            "sidecar_snapshot_fingerprints_sha256"
        ],
    }
    for key, wanted in expected_arm.items():
        if not exact_json_equal(arm_aggregate.get(key), wanted):
            raise ValueError(f"{arm} aggregate {key} mismatch")
    if set(arm_aggregate) != set(expected_arm) | {
        "arm_nonce",
        "arm_contract_sha256",
        "nodes",
    }:
        raise ValueError(f"{arm} aggregate exact key set mismatch")
    nonce = arm_aggregate.get("arm_nonce")
    if nonce != expected_arm_nonce(preflight["attempt_nonce"], arm, threads, start, stop):
        raise ValueError(f"{arm} nonce mismatch")
    arm_contract_path = Path(preflight["_path"]).parent / f"arm_contract.{preflight['jobtag']}.{arm}.json"
    arm_contract = read_object(arm_contract_path, f"{arm} contract")
    expected_contract = {
        "schema_version": ARM_CONTRACT_SCHEMA,
        "status": "READY",
        "jobtag": preflight["jobtag"],
        "arm": arm,
        "attempt_nonce": preflight["attempt_nonce"],
        "arm_nonce": nonce,
        "threads": threads,
        "start_step": start,
        "stop_step": stop,
    }
    for key, wanted in expected_contract.items():
        if not exact_json_equal(arm_contract.get(key), wanted):
            raise ValueError(f"{arm} contract {key} mismatch")
    if set(arm_contract) != set(expected_contract) | {
        "created_at_unix",
        "preflight",
        "fresh_markers",
    }:
        raise ValueError(f"{arm} contract exact key set mismatch")
    arm_contract_created_at = strict_int(
        arm_contract.get("created_at_unix"), f"{arm} contract creation time", minimum=1
    )
    contract_preflight = arm_contract.get("preflight")
    if (
        not isinstance(contract_preflight, dict)
        or set(contract_preflight) != {"path", "sha256"}
        or contract_preflight.get("path") != preflight["_path"]
        or contract_preflight.get("sha256") != sha256_file(Path(preflight["_path"]))
    ):
        raise ValueError(f"{arm} contract/preflight binding mismatch")
    fresh_markers = arm_contract.get("fresh_markers")
    if not isinstance(fresh_markers, list) or len(fresh_markers) != NODES:
        raise ValueError(f"{arm} fresh-marker declarations malformed")
    for node_rank in range(NODES):
        marker = Path(preflight["_path"]).parent / (
            f"arm_fresh.{preflight['jobtag']}.{arm}.n{node_rank}"
        )
        require_file(marker, f"{arm} node {node_rank} fresh marker")
        declaration = fresh_markers[node_rank]
        marker_text = marker.read_bytes().decode("utf-8", errors="strict")
        fields = marker_text[:-1].split("|") if marker_text.endswith("\n") else []
        if (
            not isinstance(declaration, dict)
            or set(declaration) != {"path", "sha256"}
            or declaration.get("path") != str(marker)
            or declaration.get("sha256") != sha256_file(marker)
            or marker_text.count("\n") != 1
            or "\r" in marker_text
            or len(fields) != 8
            or fields[:6]
            != [
                "READY",
                preflight["jobtag"],
                arm,
                preflight["attempt_nonce"],
                nonce,
                str(node_rank),
            ]
            or fields[7] != f"{preflight['jobtag']}-{node_rank}"
        ):
            raise ValueError(f"{arm} node {node_rank} fresh marker mismatch")
        validate_marker_time(
            fields[6], arm_contract_created_at, f"{arm} node {node_rank} fresh marker"
        )
        closure[f"{arm}:fresh_marker:{node_rank}"] = (marker, sha256_file(marker))
    if arm_aggregate.get("arm_contract_sha256") != sha256_file(arm_contract_path):
        raise ValueError(f"{arm} aggregate/contract SHA mismatch")
    closure[f"{arm}:aggregate"] = (arm_aggregate_path, sha256_file(arm_aggregate_path))
    closure[f"{arm}:contract"] = (arm_contract_path, sha256_file(arm_contract_path))

    nodes = arm_aggregate.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != NODES:
        raise ValueError(f"{arm} must bind exactly two nodes")
    rank_paths: dict[int, Path] = {}
    rank_shas: dict[int, str] = {}
    rank_hosts: dict[int, str] = {}
    runner: dict[int, dict[str, Any]] = {}
    node_evidence = []
    arm_root = evidence_root / arm
    require_directory(arm_root, f"{arm} evidence root")
    current_sidecar_ledger_sha = canonical_sha256(
        validate_sidecar_baseline(
            Path(preflight["sidecar_baseline"]["root"]),
            preflight["sidecar_baseline"],
        )
    )
    for node_rank in range(NODES):
        matches = [
            node
            for node in nodes
            if isinstance(node, dict) and exact_json_equal(node.get("node_rank"), node_rank)
        ]
        if len(matches) != 1:
            raise ValueError(f"{arm} node {node_rank} aggregate entry mismatch")
        node = matches[0]
        if set(node) != {
            "node_rank",
            "manifest",
            "manifest_sha256",
            "ready",
            "status",
            "run_rc",
            "collect_rc",
            "rank_files",
            "runner_log_sha256",
            "sidecar_control_sha256",
            "sidecar_ledger_before_sha256",
            "sidecar_ledger_after_sha256",
        }:
            raise ValueError(f"{arm} node {node_rank} aggregate exact key set mismatch")
        node_root = arm_root / f"node{node_rank}"
        require_directory(node_root, f"{arm} node {node_rank} root")
        manifest_path = node_root / "node_manifest.json"
        if node.get("manifest") != str(manifest_path) or node.get("manifest_sha256") != sha256_file(
            manifest_path
        ):
            raise ValueError(f"{arm} node {node_rank} manifest binding mismatch")
        for key, wanted in (
            ("status", "PASS"),
            ("run_rc", 0),
            ("collect_rc", 0),
            ("rank_files", LOCAL_WORLD_SIZE),
        ):
            if not exact_json_equal(node.get(key), wanted):
                raise ValueError(f"{arm} node {node_rank} aggregate close mismatch")
        manifest = read_object(manifest_path, f"{arm} node {node_rank} manifest")
        expected_manifest = {
            "schema_version": NODE_MANIFEST_SCHEMA,
            "status": "PASS",
            "error": "",
            "jobtag": preflight["jobtag"],
            "attempt_nonce": preflight["attempt_nonce"],
            "arm": arm,
            "arm_nonce": nonce,
            "threads": threads,
            "start_step": start,
            "stop_step": stop,
            "node_rank": node_rank,
            "host": f"{preflight['jobtag']}-{node_rank}",
            "world_size": WORLD_SIZE,
            "local_world_size": LOCAL_WORLD_SIZE,
            "per_rank_bs": PER_RANK_BS,
            "prefetch_steps": PREFETCH_STEPS,
            "expected_global_ranks": list(
                range(node_rank * LOCAL_WORLD_SIZE, (node_rank + 1) * LOCAL_WORLD_SIZE)
            ),
            "run_rc": 0,
            "collect_rc": 0,
            "preflight_sha256": sha256_file(Path(preflight["_path"])),
            "arm_contract_sha256": sha256_file(arm_contract_path),
            "artifacts": preflight["artifacts"],
            "plan": preflight["plan"],
            "animefreq_ledger_sha256": canonical_sha256(preflight["animefreq_ledger"]),
            "sidecar_baseline_sha256": canonical_sha256(preflight["sidecar_baseline"]),
            "sidecar_full_verify": preflight["sidecar_full_verify"],
            "sidecar_header_sha256": preflight["sidecar_header_sha256"],
            "sidecar_release_digest": preflight["sidecar_release_digest"],
            "sidecar_snapshot_fingerprints_sha256": preflight[
                "sidecar_snapshot_fingerprints_sha256"
            ],
        }
        for key, wanted in expected_manifest.items():
            if not exact_json_equal(manifest.get(key), wanted):
                raise ValueError(f"{arm} node {node_rank} manifest {key} mismatch")
        if set(manifest) != set(expected_manifest) | {
            "runner_log",
            "files",
            "sidecar_control",
            "sidecar_ledger_before",
            "sidecar_ledger_after",
        }:
            raise ValueError(f"{arm} node {node_rank} manifest exact key set mismatch")
        result_root = Path(preflight["_path"]).parent
        control_path = result_root / (
            f"sidecarcontrol.{preflight['jobtag']}.{arm}.n{node_rank}.json"
        )
        control_sha = validate_sha_artifact(
            control_path,
            manifest.get("sidecar_control"),
            f"{arm} node {node_rank} sidecar control",
            named=False,
        )
        control = read_object(control_path, f"{arm} node {node_rank} sidecar control")
        if (
            control.get("status") != "PASS"
            or control.get("mode") != "verify-sidecar-control"
            or control.get("header_sha256") != preflight["sidecar_header_sha256"]
            or control.get("release_digest") != preflight["sidecar_release_digest"]
            or control.get("catalog_hash") != preflight["plan"]["catalog_hash"]
            or control.get("header_file_sha256")
            != preflight["artifacts"]["sidecar_header"]["sha256"]
            or control.get("success_file_sha256")
            != preflight["artifacts"]["sidecar_success"]["sha256"]
            or control.get("catalog_header_sha256")
            != preflight["artifacts"]["catalog_header"]["sha256"]
        ):
            raise ValueError(f"{arm} node {node_rank} sidecar control mismatch")
        closure[f"{arm}:sidecar_control:{node_rank}"] = (control_path, control_sha)
        for phase in ("before", "after"):
            declaration = manifest.get(f"sidecar_ledger_{phase}")
            ledger_path = result_root / (
                f"sidecarledger.{preflight['jobtag']}.{arm}.n{node_rank}.{phase}.json"
            )
            ledger_sha = validate_sha_artifact(
                ledger_path,
                declaration,
                f"{arm} node {node_rank} sidecar ledger {phase}",
                named=False,
            )
            ledger = read_object(
                ledger_path, f"{arm} node {node_rank} sidecar ledger {phase}"
            )
            expected_ledger = {
                "schema_version": "offline_arb_w16_sidecar_arm_ledger_v1",
                "status": "PASS",
                "jobtag": preflight["jobtag"],
                "attempt_nonce": preflight["attempt_nonce"],
                "arm": arm,
                "arm_nonce": nonce,
                "node_rank": node_rank,
                "host": f"{preflight['jobtag']}-{node_rank}",
                "phase": phase,
                "sidecar_full_verify_sha256": preflight["sidecar_full_verify"]["sha256"],
                "baseline_sha256": canonical_sha256(preflight["sidecar_baseline"]),
                "live_ledger_sha256": current_sidecar_ledger_sha,
                "file_count": preflight["sidecar_baseline"]["file_count"],
                "total_bytes": preflight["sidecar_baseline"]["total_bytes"],
            }
            for key, wanted in expected_ledger.items():
                if not exact_json_equal(ledger.get(key), wanted):
                    raise ValueError(
                        f"{arm} node {node_rank} sidecar ledger {phase} {key} mismatch"
                    )
            if set(ledger) != set(expected_ledger) | {"created_at_unix"}:
                raise ValueError(
                    f"{arm} node {node_rank} sidecar ledger {phase} key set mismatch"
                )
            strict_int(
                ledger.get("created_at_unix"),
                f"{arm} node {node_rank} sidecar ledger time",
                minimum=1,
            )
            if node.get(f"sidecar_ledger_{phase}_sha256") != ledger_sha:
                raise ValueError(
                    f"{arm} node {node_rank} aggregate sidecar ledger {phase} mismatch"
                )
            closure[f"{arm}:sidecar_ledger:{node_rank}:{phase}"] = (
                ledger_path,
                ledger_sha,
            )
        if node.get("sidecar_control_sha256") != control_sha:
            raise ValueError(f"{arm} node {node_rank} aggregate sidecar control mismatch")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != LOCAL_WORLD_SIZE:
            raise ValueError(f"{arm} node {node_rank} rank-file declaration mismatch")
        expected_ranks = set(
            range(node_rank * LOCAL_WORLD_SIZE, (node_rank + 1) * LOCAL_WORLD_SIZE)
        )
        observed_ranks = set()
        expected_names = {"node_manifest.json", "runner.log"}
        for entry in files:
            name = entry.get("name") if isinstance(entry, dict) else None
            match = RANK_FILE_RE.fullmatch(name) if isinstance(name, str) else None
            if match is None:
                raise ValueError(f"{arm} node {node_rank} rank filename malformed")
            rank = int(match.group("rank"))
            path = node_root / name
            digest = validate_sha_artifact(path, entry, f"{arm} rank {rank} telemetry", named=True)
            observed_ranks.add(rank)
            rank_paths[rank] = path
            rank_shas[rank] = digest
            rank_hosts[rank] = manifest["host"]
            expected_names.add(name)
        if observed_ranks != expected_ranks:
            raise ValueError(f"{arm} node {node_rank} rank set mismatch")
        runner_entry = manifest.get("runner_log")
        runner_path = node_root / "runner.log"
        runner_sha = validate_sha_artifact(
            runner_path, runner_entry, f"{arm} node {node_rank} runner log", named=True
        )
        if node.get("runner_log_sha256") != runner_sha:
            raise ValueError(f"{arm} node {node_rank} runner SHA mismatch")
        if {path.name for path in node_root.iterdir()} != expected_names:
            raise ValueError(f"{arm} node {node_rank} exact artifact set mismatch")
        parsed = parse_runner_log(
            runner_path, arm=arm, node_rank=node_rank, start=start, stop=stop
        )
        if set(runner).intersection(parsed):
            raise ValueError(f"{arm} duplicate runner ranks")
        runner.update(parsed)
        closure[f"{arm}:node_manifest:{node_rank}"] = (
            manifest_path,
            sha256_file(manifest_path),
        )
        closure[f"{arm}:runner_log:{node_rank}"] = (runner_path, runner_sha)
        ready_path = Path(preflight["_path"]).parent / (
            f"arm_ready.{preflight['jobtag']}.{arm}.n{node_rank}"
        )
        require_file(ready_path, f"{arm} node {node_rank} ready marker")
        ready_decl = node.get("ready")
        if (
            not isinstance(ready_decl, dict)
            or set(ready_decl) != {"path", "sha256"}
            or ready_decl.get("path") != str(ready_path)
            or ready_decl.get("sha256") != sha256_file(ready_path)
        ):
            raise ValueError(f"{arm} node {node_rank} aggregate ready binding mismatch")
        ready_fields = ready_path.read_text(encoding="utf-8").strip().split("|")
        if ready_fields != [
            "PRESERVED",
            preflight["jobtag"],
            arm,
            nonce,
            str(node_rank),
            sha256_file(manifest_path),
            "PASS",
        ]:
            raise ValueError(f"{arm} node {node_rank} ready marker mismatch")
        closure[f"{arm}:ready_marker:{node_rank}"] = (
            ready_path,
            sha256_file(ready_path),
        )
        node_evidence.append(
            {
                "node_rank": node_rank,
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "runner_log": str(runner_path),
                "runner_log_sha256": runner_sha,
            }
        )
    if set(rank_paths) != set(range(WORLD_SIZE)) or set(runner) != set(range(WORLD_SIZE)):
        raise ValueError(f"{arm} global rank evidence not exact 0..15")
    if {path.name for path in arm_root.iterdir()} != {
        "node0",
        "node1",
        "arm_aggregate.json",
    }:
        raise ValueError(f"{arm} evidence root exact-set mismatch")

    domains = tuple(str(value) for value in plan.header["domains"])
    telemetry: dict[int, dict[str, Any]] = {}
    for rank in range(WORLD_SIZE):
        telemetry[rank] = load_telemetry(
            rank_paths[rank],
            arm=arm,
            rank=rank,
            threads=threads,
            start=start,
            stop=stop,
            nonce=nonce,
            attempt_nonce=preflight["attempt_nonce"],
            host=rank_hosts[rank],
            jobtag=preflight["jobtag"],
            plan_id=str(plan.header["plan_id"]),
            plan_hash=str(plan.header["plan_hash"]),
            catalog_hash=str(plan.header["catalog_hash"]),
            sidecar_header=preflight["sidecar_header_sha256"],
            sidecar_release=preflight["sidecar_release_digest"],
            domains=set(domains),
            runner_steps=runner[rank]["steps"],
        )
        expected_window_keys: set[tuple[int, str]] = set()
        for window_start in range(start, stop, PREFETCH_STEPS):
            expected = counts.get(window_start, rank)
            for domain in domains:
                if expected[domain] > 0:
                    expected_window_keys.add((window_start, domain))
                record = telemetry[rank]["windows"].get((window_start, domain))
                if (record is not None) != (expected[domain] > 0):
                    raise ValueError(f"{arm} rank {rank} window presence/plan mismatch")
                if record is not None and record.get("n") != expected[domain]:
                    raise ValueError(f"{arm} rank {rank} window n/plan mismatch")
        if set(telemetry[rank]["windows"]) != expected_window_keys:
            raise ValueError(f"{arm} rank {rank} telemetry window exact set mismatch")
        closure[f"{arm}:rank_telemetry:{rank}"] = (rank_paths[rank], rank_shas[rank])

    steady_starts = range(start + STEADY_OFFSET, stop, PREFETCH_STEPS)
    target_window_seconds = len(steady_starts) * PREFETCH_STEPS * TARGET_SECONDS_PER_STEP
    critical_windows = []
    for window_start in steady_starts:
        candidates = [
            (float(record["total_s"]), rank, domain)
            for rank in range(WORLD_SIZE)
            for domain in domains
            if (record := telemetry[rank]["windows"].get((window_start, domain))) is not None
        ]
        seconds, critical_rank, critical_domain = max(candidates)
        critical_windows.append(
            {
                "start": window_start,
                "critical_s": seconds,
                "rank": critical_rank,
                "domain": critical_domain,
            }
        )
    critical_values = [item["critical_s"] for item in critical_windows]
    critical_summary = {
        "windows": len(critical_values),
        "mean_s": statistics.fmean(critical_values),
        "median_s": statistics.median(critical_values),
        "p95_s": percentile(critical_values, 0.95),
        "max_s": max(critical_values),
        "critical_windows": critical_windows,
    }
    rank_domain_capacity: dict[str, dict[str, float]] = {domain: {} for domain in domains}
    rank_domain_rate: dict[str, dict[str, float]] = {domain: {} for domain in domains}
    rank_domain_required: dict[str, dict[str, float]] = {domain: {} for domain in domains}
    for rank in range(WORLD_SIZE):
        for domain in domains:
            images = 0
            seconds = 0.0
            for window_start in steady_starts:
                count = counts.get(window_start, rank)[domain]
                if count:
                    images += count
                    seconds += float(telemetry[rank]["windows"][(window_start, domain)]["total_s"])
            if images <= 0 or seconds <= 0:
                raise ValueError(f"{arm} rank {rank} lacks capacity evidence for {domain}")
            observed = images / seconds
            required = images / target_window_seconds
            rank_domain_rate[domain][str(rank)] = observed
            rank_domain_required[domain][str(rank)] = required
            rank_domain_capacity[domain][str(rank)] = observed / required
    domain_min_capacity = {
        domain: min(values.values()) for domain, values in rank_domain_capacity.items()
    }
    critical_waits = []
    for step in range(start + STEADY_OFFSET, stop):
        relative = step - start
        values = [runner[rank]["steps"][relative]["wait_s"] for rank in range(WORLD_SIZE)]
        maximum = max(values)
        critical_waits.append(
            {"step": step, "critical_wait_s": maximum, "rank": values.index(maximum)}
        )
    wait_values = [item["critical_wait_s"] for item in critical_waits]
    wait_summary = {
        "steps": len(wait_values),
        "mean_s": statistics.fmean(wait_values),
        "median_s": statistics.median(wait_values),
        "p95_s": percentile(wait_values, 0.95),
        "max_s": max(wait_values),
        "over_target_count": sum(value > TARGET_SECONDS_PER_STEP for value in wait_values),
        "over_target_fraction": sum(value > TARGET_SECONDS_PER_STEP for value in wait_values)
        / len(wait_values),
        "critical_steps": critical_waits,
    }
    elapsed_per_step = {
        str(rank): runner[rank]["summary"]["elapsed"] / SLICE_STEPS
        for rank in range(WORLD_SIZE)
    }
    gates: list[dict[str, Any]] = []
    add_gate(gates, "global_wait_mean_s", wait_summary["mean_s"], "<=", WAIT_MEAN_MAX_S)
    add_gate(gates, "global_wait_p95_s", wait_summary["p95_s"], "<=", WAIT_P95_MAX_S)
    add_gate(gates, "global_wait_max_s", wait_summary["max_s"], "<=", WAIT_MAX_S)
    add_gate(
        gates,
        "global_wait_over_target_fraction",
        wait_summary["over_target_fraction"],
        "<=",
        WAIT_OVER_TARGET_FRACTION_MAX,
    )
    add_gate(gates, "critical_window_mean_s", critical_summary["mean_s"], "<=", WINDOW_MEAN_MAX_S)
    add_gate(gates, "critical_window_p95_s", critical_summary["p95_s"], "<=", WINDOW_TAIL_MAX_S)
    add_gate(gates, "critical_window_max_s", critical_summary["max_s"], "<=", WINDOW_TAIL_MAX_S)
    add_gate(
        gates,
        "max_rank_runner_elapsed_per_step_s",
        max(elapsed_per_step.values()),
        "<=",
        RUNNER_ELAPSED_PER_STEP_MAX_S,
    )
    for domain in domains:
        add_gate(
            gates,
            f"{domain}_min_rank_capacity_ratio",
            domain_min_capacity[domain],
            ">=",
            CAPACITY_HEADROOM_RATIO,
        )
    failed = [gate["name"] for gate in gates if not gate["pass"]]
    return {
        "status": "PASS" if not failed else "NO_GO",
        "arm": arm,
        "threads": threads,
        "start_step": start,
        "stop_step": stop,
        "steady_start_step": start + STEADY_OFFSET,
        "failed_gates": failed,
        "gates": gates,
        "global_critical_wait": wait_summary,
        "combined_critical_windows": critical_summary,
        "domain_min_rank_capacity_ratio": domain_min_capacity,
        "rank_domain_capacity_ratio": rank_domain_capacity,
        "rank_domain_img_s": rank_domain_rate,
        "rank_domain_required_img_s": rank_domain_required,
        "rank_runner_elapsed_per_step_s": elapsed_per_step,
        "telemetry_record_counts": {
            str(rank): telemetry[rank]["record_counts"] for rank in range(WORLD_SIZE)
        },
        "evidence": {
            "arm_aggregate": str(arm_aggregate_path),
            "arm_aggregate_sha256": sha256_file(arm_aggregate_path),
            "node_manifests": node_evidence,
            "rank_telemetry_sha256": {str(rank): rank_shas[rank] for rank in range(WORLD_SIZE)},
        },
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    master_path = Path(args.master)
    repo = Path(args.repo).resolve(strict=True)
    plan_root = Path(args.plan)
    master = read_object(master_path, "ABBA master aggregate")
    result_root = master_path.parent
    require_directory(result_root, "ABBA result root")
    if result_root.parent.name != args.run_id or master_path.name != f"abba.{result_root.name}.aggregate.json":
        raise ValueError("ABBA master path is not canonical for run-id/jobtag")
    expected_master = {
        "schema_version": MASTER_SCHEMA,
        "status": "PASS",
        "run_id": args.run_id,
        "jobtag": result_root.name,
        "world_size": WORLD_SIZE,
        "local_world_size": LOCAL_WORLD_SIZE,
        "nodes": NODES,
        "per_rank_bs": PER_RANK_BS,
        "prefetch_steps": PREFETCH_STEPS,
    }
    for key, wanted in expected_master.items():
        if not exact_json_equal(master.get(key), wanted):
            raise ValueError(f"ABBA master {key} mismatch")
    if set(master) != set(expected_master) | {
        "attempt_nonce",
        "preflight",
        "sidecar_full_post",
        "arms",
    }:
        raise ValueError("ABBA master exact key set mismatch")
    attempt_nonce = master.get("attempt_nonce")
    if attempt_nonce != expected_attempt_nonce(args.run_id, result_root.name):
        raise ValueError("ABBA master attempt nonce mismatch")
    preflight_decl = master.get("preflight")
    preflight_path = result_root / f"preflight.{result_root.name}.contract.json"
    if (
        not isinstance(preflight_decl, dict)
        or preflight_decl.get("path") != str(preflight_path)
        or preflight_decl.get("sha256") != sha256_file(preflight_path)
    ):
        raise ValueError("ABBA master/preflight binding mismatch")
    preflight = read_object(preflight_path, "ABBA preflight contract")
    preflight_keys = {
        "schema_version",
        "status",
        "jobtag",
        "attempt_nonce",
        "created_at_unix",
        "master_addr",
        "sidecar_full_verify",
        "artifacts",
        "animefreq_ledger",
        "plan",
        "sidecar_baseline",
        "sidecar_header_sha256",
        "sidecar_release_digest",
        "sidecar_snapshot_fingerprints_sha256",
        "fresh_markers",
        "arms",
    }
    if set(preflight) != preflight_keys:
        raise ValueError("ABBA preflight exact key set mismatch")
    preflight["_path"] = str(preflight_path)
    expected_preflight = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "PASS",
        "jobtag": result_root.name,
        "attempt_nonce": attempt_nonce,
        "arms": [
            {"arm": arm, "threads": threads, "start_step": start, "stop_step": stop}
            for arm, threads, start, stop in ARMS
        ],
    }
    for key, wanted in expected_preflight.items():
        if not exact_json_equal(preflight.get(key), wanted):
            raise ValueError(f"ABBA preflight {key} mismatch")
    preflight_created_at = strict_int(
        preflight.get("created_at_unix"), "preflight creation time", minimum=1
    )
    if not isinstance(preflight.get("master_addr"), str) or not preflight["master_addr"]:
        raise ValueError("ABBA preflight master address malformed")
    for label in (
        "sidecar_header_sha256",
        "sidecar_release_digest",
        "sidecar_snapshot_fingerprints_sha256",
    ):
        if not isinstance(preflight.get(label), str) or SHA_RE.fullmatch(preflight[label]) is None:
            raise ValueError(f"preflight {label} malformed")
    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("preflight artifacts malformed")
    expected_paths = {
        "runner": repo / GATE_RELATIVE_FILES[0],
        "capture_yaml": repo / GATE_RELATIVE_FILES[1],
        "analyzer": repo / GATE_RELATIVE_FILES[2],
        "analyze_yaml": repo / GATE_RELATIVE_FILES[3],
        "dataset_source": repo / "training/dataline_offline/offline_dataset.py",
        "planner_source": repo / "training/dataline_offline/planner_v1.py",
        "sidecar_runtime_source": repo / "training/dataline_offline/caption_sidecar.py",
        "complete_dataset_source": repo / "training/dataline/complete_dataset.py",
        "dataline_config_source": repo / "training/dataline/config.py",
        "image_fetch_source": repo / "training/dataline/image_fetch.py",
        "manifest_index_source": repo / "training/dataline/manifest_index.py",
        "multi_ar_source": repo / "training/dataline/multi_ar.py",
        "native_safety_source": repo / "training/dataline/native_safety.py",
        "pil_safety_source": repo / "training/dataline/pil_safety.py",
        "caption_build_source": repo / "training/dataline/caption_build.py",
        "sampler_source": repo / "training/dataline/sampler.py",
        "sidecar_contract_source": repo / "scripts/offline_arb/caption_sidecar_contract.py",
        "sidecar_verify_source": repo / "scripts/offline_arb/make_freeze_manifest.py",
        "sampling": Path(
            "/fsx/evan/mot_code/data_analysis/configs/t5_sampling_redesign_v1.json"
        ),
        "config": repo / "configs/stage256_offline_exact_v1.yaml",
        "plan_header": plan_root / "header.json",
        "plan_checksums": plan_root / "checksums.json",
        "catalog_header": Path(args.catalog) / "header.json",
        "animefreq_header": Path(args.animefreq) / "header.json",
        "sidecar_header": Path(args.sidecar) / "header.json",
        "sidecar_success": Path(args.sidecar) / "_SUCCESS",
        "python_interpreter": PYTHON.resolve(strict=True),
    }
    if set(artifacts) != set(expected_paths):
        raise ValueError("preflight live artifact exact key set mismatch")
    closure: dict[str, tuple[Path, str]] = {
        "master": (master_path, sha256_file(master_path)),
        "preflight": (preflight_path, sha256_file(preflight_path)),
    }
    for label, path in expected_paths.items():
        digest = validate_sha_artifact(path, artifacts.get(label), f"preflight {label}", named=False)
        closure[f"artifact:{label}"] = (path, digest)
    if Path(sys.executable).resolve(strict=True) != expected_paths["python_interpreter"]:
        raise ValueError("analyzer is not running through the preflight-bound Python target")
    animefreq_ledger = preflight.get("animefreq_ledger")
    live_animefreq = build_animefreq_ledger(Path(args.animefreq))
    if not exact_json_equal(animefreq_ledger, live_animefreq):
        raise ValueError("preflight/analyzer anime-frequency ledger mismatch")
    sidecar_baseline = preflight.get("sidecar_baseline")
    validate_sidecar_baseline(Path(args.sidecar), sidecar_baseline)

    full_verify = preflight.get("sidecar_full_verify")
    verify_path = result_root / f"sidecarfull.{result_root.name}.json"
    if (
        not isinstance(full_verify, dict)
        or set(full_verify) != {"path", "sha256"}
        or full_verify.get("path") != str(verify_path)
        or full_verify.get("sha256") != sha256_file(verify_path)
    ):
        raise ValueError("preflight full-sidecar declaration mismatch")
    verify = read_object(verify_path, "capture full-sidecar verification JSON")
    if (
        verify.get("status") != "PASS"
        or verify.get("mode") != "verify-sidecar"
        or verify.get("root") != str(Path(args.sidecar).resolve(strict=True))
        or verify.get("header_sha256") != preflight["sidecar_header_sha256"]
        or verify.get("release_digest") != preflight["sidecar_release_digest"]
        or verify.get("snapshot_fingerprints_sha256")
        != preflight["sidecar_snapshot_fingerprints_sha256"]
        or verify.get("header_file_sha256") != artifacts["sidecar_header"]["sha256"]
        or verify.get("success_file_sha256") != artifacts["sidecar_success"]["sha256"]
        or verify.get("catalog_header_sha256") != artifacts["catalog_header"]["sha256"]
        or strict_int(verify.get("file_count"), "full sidecar file count", minimum=1) <= 0
        or strict_int(verify.get("total_bytes"), "full sidecar bytes", minimum=1) <= 0
    ):
        raise ValueError("capture full-sidecar verification evidence mismatch")
    if (
        not isinstance(sidecar_baseline, dict)
        or sidecar_baseline.get("files") != verify.get("files")
        or sidecar_baseline.get("directories") != verify.get("directories")
        or sidecar_baseline.get("file_count") != verify.get("file_count")
        or sidecar_baseline.get("total_bytes") != verify.get("total_bytes")
        or sidecar_baseline.get("header_sha256") != verify.get("header_sha256")
        or sidecar_baseline.get("release_digest") != verify.get("release_digest")
    ):
        raise ValueError("capture full-sidecar/baseline ledger mismatch")
    closure["sidecar_full_verify"] = (verify_path, sha256_file(verify_path))

    post_decl = master.get("sidecar_full_post")
    post_path = result_root / f"sidecarfull.post.{result_root.name}.json"
    if (
        not isinstance(post_decl, dict)
        or set(post_decl) != {"path", "sha256"}
        or post_decl.get("path") != str(post_path)
        or post_decl.get("sha256") != sha256_file(post_path)
    ):
        raise ValueError("master post-sidecar declaration mismatch")
    post_verify = read_object(post_path, "post-ABBA full sidecar verification")
    if post_verify != verify:
        raise ValueError("post-ABBA full sidecar verification differs from preflight")
    closure["sidecar_full_post"] = (post_path, sha256_file(post_path))

    from scripts.offline_arb.make_freeze_manifest import validate_sidecar

    analysis_full = {"status": "PASS", "mode": "verify-sidecar", **validate_sidecar(
        Path(args.sidecar), Path(args.catalog), full=True
    )}
    if analysis_full != verify:
        raise ValueError("analysis-time full sidecar verification differs from capture")
    analysis_full_path = result_root / f"sidecarfull.analysis.{result_root.name}.json"
    atomic_write(analysis_full_path, analysis_full)
    closure["sidecar_full_analysis"] = (
        analysis_full_path,
        sha256_file(analysis_full_path),
    )

    fresh_declarations = preflight.get("fresh_markers")
    if not isinstance(fresh_declarations, list) or len(fresh_declarations) != NODES:
        raise ValueError("preflight fresh-marker declarations malformed")
    for node_rank in range(NODES):
        marker = result_root / f"livepreflight.{result_root.name}.n{node_rank}.fresh"
        require_file(marker, f"fresh marker node {node_rank}")
        declaration = fresh_declarations[node_rank]
        marker_text = marker.read_bytes().decode("utf-8", errors="strict")
        fields = marker_text[:-1].split("|") if marker_text.endswith("\n") else []
        if (
            not isinstance(declaration, dict)
            or set(declaration) != {"path", "sha256"}
            or declaration.get("path") != str(marker)
            or declaration.get("sha256") != sha256_file(marker)
            or marker_text.count("\n") != 1
            or "\r" in marker_text
            or len(fields) != 6
            or fields[:4]
            != ["READY", result_root.name, attempt_nonce, str(node_rank)]
            or fields[5] != f"{result_root.name}-{node_rank}"
        ):
            raise ValueError("fresh preflight marker identity mismatch")
        validate_marker_time(
            fields[4], preflight_created_at, f"preflight node {node_rank} fresh marker"
        )
        closure[f"fresh_marker:{node_rank}"] = (marker, sha256_file(marker))

    plan = OfflinePlan(plan_root, world_size=WORLD_SIZE, per_rank_bs=PER_RANK_BS, verify_checksums=True)
    if (
        plan.header.get("plan_id") != EXPECTED_PLAN_ID
        or tuple(plan.header.get("domains", [])) != EXPECTED_DOMAINS
        or int(plan.config.total_steps) < ARMS[-1][3]
        or verify.get("catalog_hash") != plan.header.get("catalog_hash")
    ):
        raise ValueError("W16 plan identity/domain/coverage mismatch")
    plan_decl = preflight.get("plan")
    if not isinstance(plan_decl, dict):
        raise ValueError("preflight plan declaration malformed")
    for key, wanted in (
        ("plan_id", plan.header["plan_id"]),
        ("plan_hash", plan.header["plan_hash"]),
        ("catalog_hash", plan.header["catalog_hash"]),
        ("world_size", WORLD_SIZE),
        ("per_rank_bs", PER_RANK_BS),
        ("prefetch_steps", PREFETCH_STEPS),
    ):
        if not exact_json_equal(plan_decl.get(key), wanted):
            raise ValueError(f"preflight plan {key} mismatch")
    counts = PlanWindowCounts(plan)

    master_arms = master.get("arms")
    if not isinstance(master_arms, list) or len(master_arms) != len(ARMS):
        raise ValueError("master arm evidence count mismatch")
    evidence_root = result_root / f"evidence.{result_root.name}"
    require_directory(evidence_root, "ABBA evidence root")
    arm_results: dict[str, dict[str, Any]] = {}
    for arm_spec in ARMS:
        arm = arm_spec[0]
        matches = [item for item in master_arms if isinstance(item, dict) and item.get("arm") == arm]
        if len(matches) != 1:
            raise ValueError(f"master arm {arm} missing/duplicated")
        declaration = matches[0]
        if set(declaration) != {"arm", "path", "sha256", "status"}:
            raise ValueError(f"master arm {arm} declaration key set mismatch")
        aggregate_path = evidence_root / arm / "arm_aggregate.json"
        if (
            declaration.get("path") != str(aggregate_path)
            or declaration.get("sha256") != sha256_file(aggregate_path)
            or declaration.get("status") != "PASS"
        ):
            raise ValueError(f"master arm {arm} binding mismatch")
        aggregate = read_object(aggregate_path, f"{arm} aggregate")
        arm_results[arm] = analyze_arm(
            arm_spec=arm_spec,
            arm_aggregate_path=aggregate_path,
            arm_aggregate=aggregate,
            preflight=preflight,
            plan=plan,
            counts=counts,
            evidence_root=evidence_root,
            closure=closure,
        )
    if {path.name for path in evidence_root.iterdir()} != {arm for arm, *_ in ARMS}:
        raise ValueError("ABBA evidence root arm exact-set mismatch")

    used_blocks = plan_decl.get("used_blocks")
    if not isinstance(used_blocks, dict) or not used_blocks:
        raise ValueError("preflight used plan blocks malformed")
    expected_verified = sorted(
        {
            step // plan.config.block_steps
            for _arm, _threads, start, stop in ARMS
            for step in range(start, stop)
        }
    )
    if sorted(plan._verified_blocks) != expected_verified:
        raise ValueError("analyzer plan verified-block set mismatch")
    for block_index in expected_verified:
        relative = f"block_{block_index:06d}.npz"
        block_path = plan_root / "blocks" / relative
        digest = validate_sha_artifact(
            block_path, used_blocks.get(relative), f"used plan block {relative}", named=False
        )
        if plan.checksums.get(relative) != digest:
            raise ValueError("plan reader/preflight block SHA mismatch")
        closure[f"plan_block:{relative}"] = (block_path, digest)

    qualifications = {}
    for group, arms in THREAD_GROUPS.items():
        left, right = arms
        left_capacity = arm_results[left]["domain_min_rank_capacity_ratio"]["gpic"]
        right_capacity = arm_results[right]["domain_min_rank_capacity_ratio"]["gpic"]
        drift = positive_pair_drift(left_capacity, right_capacity)
        stable = drift <= PAIR_DRIFT_MAX
        arms_pass = all(arm_results[arm]["status"] == "PASS" for arm in arms)
        qualifications[group] = {
            "arms": list(arms),
            "arm_statuses": {arm: arm_results[arm]["status"] for arm in arms},
            "gpic_min_rank_capacity_ratio": {
                left: left_capacity,
                right: right_capacity,
            },
            "pair_drift_formula": "abs(a-b)/max(a,b)",
            "pair_drift": drift,
            "pair_drift_max": PAIR_DRIFT_MAX,
            "stable": stable,
            "inconclusive": not stable,
            "arms_pass": arms_pass,
            "qualified": arms_pass and stable,
            "worst_gpic_min_rank_capacity_ratio": min(left_capacity, right_capacity),
        }
    candidates = [group for group, value in qualifications.items() if value["qualified"]]
    inconclusive_groups = [
        group for group, value in qualifications.items() if value["inconclusive"]
    ]
    status = "PASS" if candidates else "NO_GO"
    decision = (
        "SELECTABLE"
        if candidates
        else "INCONCLUSIVE"
        if inconclusive_groups
        else "NO_GO"
    )

    for label, (path, expected_sha) in closure.items():
        require_file(path, f"final closure {label}")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"final immutable-evidence closure mismatch: {label}")
    if {path.name for path in evidence_root.iterdir()} != {arm for arm, *_ in ARMS}:
        raise ValueError("final ABBA evidence-root exact-set rescan mismatch")
    for arm, *_ in ARMS:
        arm_root = evidence_root / arm
        if {path.name for path in arm_root.iterdir()} != {
            "node0",
            "node1",
            "arm_aggregate.json",
        }:
            raise ValueError(f"final {arm} exact-set rescan mismatch")
        for node_rank in range(NODES):
            node_root = arm_root / f"node{node_rank}"
            expected_node = {
                "node_manifest.json",
                "runner.log",
                *{
                    f"io_profile.rank{rank:04d}.jsonl"
                    for rank in range(
                        node_rank * LOCAL_WORLD_SIZE,
                        (node_rank + 1) * LOCAL_WORLD_SIZE,
                    )
                },
            }
            if {path.name for path in node_root.iterdir()} != expected_node:
                raise ValueError(f"final {arm} node {node_rank} exact-set rescan mismatch")
    validate_sidecar_baseline(Path(args.sidecar), sidecar_baseline)
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "status": status,
        "decision": decision,
        "selection_candidates": candidates,
        "inconclusive_groups": inconclusive_groups,
        "thread_qualification": qualifications,
        "failed_arms": [arm for arm, value in arm_results.items() if value["status"] != "PASS"],
        "identity": {
            "run_id": args.run_id,
            "jobtag": result_root.name,
            "attempt_nonce": attempt_nonce,
            "plan_id": plan.header["plan_id"],
            "plan_hash": plan.header["plan_hash"],
            "catalog_hash": plan.header["catalog_hash"],
            "world_size": WORLD_SIZE,
            "local_world_size": LOCAL_WORLD_SIZE,
            "per_rank_bs": PER_RANK_BS,
            "prefetch_steps": PREFETCH_STEPS,
            "target_seconds_per_step": TARGET_SECONDS_PER_STEP,
        },
        "thresholds": {
            "capacity_headroom_ratio": CAPACITY_HEADROOM_RATIO,
            "global_wait_mean_max_s": WAIT_MEAN_MAX_S,
            "global_wait_p95_max_s": WAIT_P95_MAX_S,
            "global_wait_max_s": WAIT_MAX_S,
            "global_wait_over_target_fraction_max": WAIT_OVER_TARGET_FRACTION_MAX,
            "critical_window_mean_max_s": WINDOW_MEAN_MAX_S,
            "critical_window_tail_max_s": WINDOW_TAIL_MAX_S,
            "runner_elapsed_per_step_max_s": RUNNER_ELAPSED_PER_STEP_MAX_S,
            "pair_drift_max": PAIR_DRIFT_MAX,
        },
        "arms": arm_results,
        "evidence": {
            "master": str(master_path),
            "master_sha256": sha256_file(master_path),
            "preflight": str(preflight_path),
            "preflight_sha256": sha256_file(preflight_path),
            "sidecar_full_verify": str(verify_path),
            "sidecar_full_verify_sha256": sha256_file(verify_path),
            "sidecar_full_post": str(post_path),
            "sidecar_full_post_sha256": sha256_file(post_path),
            "sidecar_full_analysis": str(analysis_full_path),
            "sidecar_full_analysis_sha256": sha256_file(analysis_full_path),
            "animefreq_ledger_sha256": canonical_sha256(animefreq_ledger),
            "sidecar_baseline_sha256": canonical_sha256(sidecar_baseline),
            "final_closure_artifacts": len(closure),
        },
        "scope": (
            "CPU raw-supply selector only. A qualifying thread count still requires semantic "
            "coverage for that count and a clean 2x8 GPU validation before FORMAL."
        ),
    }


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite analysis output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--animefreq", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = analyze(args)
    except BaseException as exc:
        result = {
            "schema_version": ANALYSIS_SCHEMA,
            "status": "FATAL",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if args.output:
            try:
                atomic_write(Path(args.output), result)
            except BaseException as output_exc:
                print(f"FATAL ABBA analyzer output failure: {output_exc}", file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    if args.output:
        atomic_write(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
