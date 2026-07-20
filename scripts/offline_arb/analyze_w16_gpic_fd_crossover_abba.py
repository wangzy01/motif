#!/usr/bin/env python3
"""Analyze the W16 strict_path/fd_stable crossover ABBA capture.

The r4 analyzer remains the strict closure and absolute-performance engine.  A
mode-aware compatibility layer validates (then removes only for the r4 engine)
the extension fields.  This analyzer additionally requires exact same-slice
rank/step/item/domain/raw-byte ledgers, fd syscall accounting, two independently
passing fd arms, <=5% fd drift, and <=10% strict control drift.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import re
import statistics
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Mapping

from scripts.offline_arb import analyze_w16_gpic_abba_raw_gate as core


RUN_ID = "stage256_offline_exact_w16_gpic_fd_crossover_abba_20260712r1"
ANALYSIS_SCHEMA = "offline_arb_w16_gpic_fd_crossover_analysis_v1"
WORLD_SIZE = 16
LOCAL_WORLD_SIZE = 8
PER_RANK_BS = 36
PREFETCH_STEPS = 4
SLICE_STEPS = 192
FD_PAIR_DRIFT_MAX = 0.05
STRICT_ENV_DRIFT_MAX = 0.10
GPUVAL_CANDIDATE_MIN_GPIC_SPEEDUP = 1.50
ARMS = (
    ("strict_x", 16, 0, 192),
    ("fd_x", 16, 0, 192),
    ("fd_y", 16, 192, 384),
    ("strict_y", 16, 192, 384),
)
MODE_BY_ARM = {
    "strict_x": "strict_path",
    "fd_x": "fd_stable",
    "fd_y": "fd_stable",
    "strict_y": "strict_path",
}
CROSSOVER_PAIRS = (("strict_x", "fd_x"), ("fd_y", "strict_y"))
GATE_RELATIVE_FILES = (
    "scripts/offline_arb/run_w16_gpic_fd_crossover_abba.py",
    "jobs/offline_arb_w16_gpic_fd_crossover_abba.yaml",
    "scripts/offline_arb/analyze_w16_gpic_fd_crossover_abba.py",
    "jobs/offline_arb_w16_gpic_fd_crossover_abba_analyze.yaml",
)
EXTRA_ARTIFACT_PATHS = {
    "crossover_core_runner_source": "scripts/offline_arb/run_w16_gpic_abba_raw_gate.py",
    "crossover_core_analyzer_source": "scripts/offline_arb/analyze_w16_gpic_abba_raw_gate.py",
}
CUSTOM_TO_CORE_SCHEMA = {
    "offline_arb_w16_gpic_fd_crossover_master_aggregate_v1": core.MASTER_SCHEMA,
    "offline_arb_w16_gpic_fd_crossover_preflight_v1": core.PREFLIGHT_SCHEMA,
    "offline_arb_w16_gpic_fd_crossover_arm_contract_v1": core.ARM_CONTRACT_SCHEMA,
    "offline_arb_w16_gpic_fd_crossover_arm_aggregate_v1": core.ARM_AGGREGATE_SCHEMA,
    "offline_arb_w16_gpic_fd_crossover_node_manifest_v1": core.NODE_MANIFEST_SCHEMA,
}
SHA_RE = re.compile(r"[0-9a-f]{64}")
STEP_LINE_RE = re.compile(
    r"W16_FD_CROSSOVER_STEP_RETIRED arm=(?P<arm>[a-z0-9_]+) "
    r"mode=(?P<mode>strict_path|fd_stable) rank=(?P<rank>[0-9]+) "
    r"step=(?P<step>[0-9]+) relative_step=(?P<relative>[0-9]+) "
    r"items=(?P<items>[0-9]+) raw_bytes=(?P<raw_bytes>[0-9]+) "
    r"wait_s=(?P<wait_s>[0-9]+\.[0-9]{9}) scheduled_until=(?P<scheduled>[0-9]+) "
    r"item_ledger_sha256=(?P<ledger>[0-9a-f]{64})"
)
PASS_LINE_RE = re.compile(
    r"W16_FD_CROSSOVER_RANK_PASS arm=(?P<arm>[a-z0-9_]+) "
    r"mode=(?P<mode>strict_path|fd_stable) rank=(?P<rank>[0-9]+) "
    r"start_step=(?P<start>[0-9]+) stop_step=(?P<stop>[0-9]+) "
    r"retired_steps=(?P<retired>[0-9]+) per_rank_batch_size=(?P<bs>[0-9]+) "
    r"raw_items=(?P<items>[0-9]+) raw_bytes=(?P<raw_bytes>[0-9]+) "
    r"wait_total_s=(?P<wait_total>[0-9]+\.[0-9]{9}) "
    r"wait_max_s=(?P<wait_max>[0-9]+\.[0-9]{9}) "
    r"elapsed_s=(?P<elapsed>[0-9]+\.[0-9]{9}) decode_calls=(?P<decode>[0-9]+) "
    r"step_ledger_sha256=(?P<ledger>[0-9a-f]{64}) "
    r"plan_raw_ledger_schema=(?P<ledger_schema>[a-z0-9_]+)"
)

_ORIGINAL_READ_OBJECT = core.read_object
_ORIGINAL_LOAD_TELEMETRY = core.load_telemetry
_RUNNER_EVIDENCE: dict[tuple[str, int], dict[str, Any]] = {}
_TELEMETRY_AUDIT: dict[tuple[str, int], dict[str, Any]] = {}


def expected_analysis_attempt_nonce(
    run_id: str, source_jobtag: str, analyzer_jobtag: str
) -> str:
    return core.canonical_sha256(
        {
            "schema_version": "offline_arb_w16_gpic_fd_crossover_analysis_attempt_v1",
            "run_id": run_id,
            "source_jobtag": source_jobtag,
            "analyzer_jobtag": analyzer_jobtag,
        }
    )


def analysis_output_name(
    source_jobtag: str, analyzer_jobtag: str, attempt_nonce: str
) -> str:
    for label, value in (
        ("source_jobtag", source_jobtag),
        ("analyzer_jobtag", analyzer_jobtag),
        ("attempt_nonce", attempt_nonce),
    ):
        if (
            re.fullmatch(r"[A-Za-z0-9_.-]+", value or "") is None
            or value in {".", ".."}
            or Path(value).name != value
        ):
            raise ValueError(f"analysis {label} is malformed")
    if SHA_RE.fullmatch(attempt_nonce) is None:
        raise ValueError("analysis attempt nonce is not SHA256-shaped")
    name = (
        "fdx."
        f"analyzer-{analyzer_jobtag}.attempt-{attempt_nonce}.json"
    )
    if len((name + ".attestation.json").encode("utf-8")) > 255:
        raise ValueError("analysis output/attestation filename exceeds NAME_MAX")
    return name


def _validate_analysis_attempt_identity(args: argparse.Namespace) -> Path:
    master_path = Path(args.master)
    source_jobtag = master_path.parent.name
    expected_nonce = expected_analysis_attempt_nonce(
        args.run_id, source_jobtag, args.analyzer_jobtag
    )
    if args.analysis_attempt_nonce != expected_nonce:
        raise ValueError("analysis attempt nonce identity mismatch")
    run_root = master_path.parent.parent
    expected_root = (
        run_root
        / "_analysis"
        / source_jobtag
        / args.analyzer_jobtag
        / expected_nonce
    )
    supplied_root = Path(args.analysis_attempt_root)
    hierarchy = (
        run_root,
        run_root / "_analysis",
        run_root / "_analysis" / source_jobtag,
        run_root / "_analysis" / source_jobtag / args.analyzer_jobtag,
        supplied_root,
    )
    if any(path.is_symlink() or not path.is_dir() for path in hierarchy):
        raise ValueError("analysis attempt hierarchy contains a symlink/non-directory")
    if (
        supplied_root != expected_root
        or supplied_root.resolve(strict=True) != expected_root.resolve(strict=True)
    ):
        raise ValueError("analysis attempt root is not the canonical real namespace")
    expected_output = supplied_root / analysis_output_name(
        source_jobtag, args.analyzer_jobtag, expected_nonce
    )
    if args.output is None or Path(args.output) != expected_output:
        raise ValueError("analysis output path is not canonical for attempt identity")
    entries = list(supplied_root.iterdir())
    if (
        len(entries) != 1
        or entries[0].name != "capture_input_attestation.json"
        or entries[0].is_symlink()
        or not entries[0].is_file()
        or entries[0].stat().st_size <= 0
    ):
        raise ValueError("analysis attempt namespace is not fresh/exact before analyzer")
    return supplied_root


def mode_for_arm(arm: str) -> str:
    try:
        return MODE_BY_ARM[arm]
    except KeyError as exc:
        raise ValueError(f"unknown crossover arm: {arm!r}") from exc


def expected_arm_nonce(
    attempt: str, arm: str, threads: int, start: int, stop: int
) -> str:
    return core.canonical_sha256(
        {
            "schema_version": "offline_arb_w16_gpic_fd_crossover_arm_nonce_v1",
            "attempt_nonce": attempt,
            "arm": arm,
            "gpic_source_consistency_mode": mode_for_arm(arm),
            "threads": threads,
            "start_step": start,
            "stop_step": stop,
        }
    )


def _strip_extra_artifacts(value: dict[str, Any]) -> None:
    artifacts = value.get("artifacts")
    if isinstance(artifacts, dict):
        for label in EXTRA_ARTIFACT_PATHS:
            artifacts.pop(label, None)


def compatibility_read_object(path: Path, label: str) -> dict[str, Any]:
    """Validate extension identity before producing an r4-compatible copy."""
    raw = _ORIGINAL_READ_OBJECT(path, label)
    schema = raw.get("schema_version")
    if schema not in CUSTOM_TO_CORE_SCHEMA:
        return raw
    value = copy.deepcopy(raw)
    if schema == "offline_arb_w16_gpic_fd_crossover_master_aggregate_v1":
        if (
            value.get("gpic_source_consistency_mode_by_arm") != MODE_BY_ARM
            or value.get("crossover_pairs") != [list(pair) for pair in CROSSOVER_PAIRS]
            or value.get("semantic_external_gate") != "PENDING"
            or value.get("production_selection_qualified") is not False
        ):
            raise ValueError("crossover master mode/pair/production contract mismatch")
        value.pop("gpic_source_consistency_mode_by_arm")
        value.pop("crossover_pairs")
        value.pop("semantic_external_gate")
        value.pop("production_selection_qualified")
    elif schema == "offline_arb_w16_gpic_fd_crossover_preflight_v1":
        arms = value.get("arms")
        expected = [
            {
                "arm": arm,
                "gpic_source_consistency_mode": mode_for_arm(arm),
                "threads": threads,
                "start_step": start,
                "stop_step": stop,
            }
            for arm, threads, start, stop in ARMS
        ]
        if arms != expected or value.get("crossover_pairs") != [list(pair) for pair in CROSSOVER_PAIRS]:
            raise ValueError("crossover preflight mode/pair contract mismatch")
        if value.get("gpic_threads_resolution") != {
            "configured_per_node": 16,
            "required_environment_value": "16",
            "arm_argument_values": [16, 16, 16, 16],
            "expected_local_world_size": 8,
            "expected_workers_per_rank": 2,
        }:
            raise ValueError("crossover preflight GPIC thread-resolution mismatch")
        value["arms"] = [
            {key: item[key] for key in ("arm", "threads", "start_step", "stop_step")}
            for item in arms
        ]
        value.pop("crossover_pairs")
        value.pop("gpic_threads_resolution")
        _strip_extra_artifacts(value)
    else:
        arm = value.get("arm")
        if (
            not isinstance(arm, str)
            or value.get("gpic_source_consistency_mode") != mode_for_arm(arm)
        ):
            raise ValueError(f"{label} explicit source-consistency mode mismatch")
        value.pop("gpic_source_consistency_mode")
        if schema == "offline_arb_w16_gpic_fd_crossover_node_manifest_v1":
            _strip_extra_artifacts(value)
    value["schema_version"] = CUSTOM_TO_CORE_SCHEMA[schema]
    return value


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite/nonnegative")
    return result


def parse_runner_log(
    path: Path, *, arm: str, node_rank: int, start: int, stop: int
) -> dict[int, dict[str, Any]]:
    core.require_file(path, f"{arm} node {node_rank} runner log")
    mode = mode_for_arm(arm)
    ranks = range(node_rank * LOCAL_WORLD_SIZE, (node_rank + 1) * LOCAL_WORLD_SIZE)
    steps: dict[int, list[dict[str, Any]]] = {rank: [] for rank in ranks}
    passes: dict[int, list[dict[str, Any]]] = {rank: [] for rank in ranks}
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            match = STEP_LINE_RE.fullmatch(line)
            if match is not None:
                rank = int(match.group("rank"))
                if rank not in steps or match.group("arm") != arm or match.group("mode") != mode:
                    raise ValueError(f"{arm} node {node_rank} foreign step sentinel")
                steps[rank].append(
                    {
                        "line": line_number,
                        "step": int(match.group("step")),
                        "relative": int(match.group("relative")),
                        "items": int(match.group("items")),
                        "raw_bytes": int(match.group("raw_bytes")),
                        "wait_s": float(match.group("wait_s")),
                        "scheduled": int(match.group("scheduled")),
                        "item_ledger_sha256": match.group("ledger"),
                    }
                )
                continue
            match = PASS_LINE_RE.fullmatch(line)
            if match is not None:
                rank = int(match.group("rank"))
                if rank not in passes or match.group("arm") != arm or match.group("mode") != mode:
                    raise ValueError(f"{arm} node {node_rank} foreign PASS sentinel")
                passes[rank].append(
                    {
                        "start": int(match.group("start")),
                        "stop": int(match.group("stop")),
                        "retired": int(match.group("retired")),
                        "bs": int(match.group("bs")),
                        "items": int(match.group("items")),
                        "raw_bytes": int(match.group("raw_bytes")),
                        "wait_total": float(match.group("wait_total")),
                        "wait_max": float(match.group("wait_max")),
                        "elapsed": float(match.group("elapsed")),
                        "decode": int(match.group("decode")),
                        "step_ledger_sha256": match.group("ledger"),
                        "ledger_schema": match.group("ledger_schema"),
                    }
                )
                continue
            if "W16_FD_CROSSOVER_STEP_RETIRED" in line or "W16_FD_CROSSOVER_RANK_PASS" in line:
                raise ValueError(f"{arm} malformed/interleaved sentinel at line {line_number}")
    result: dict[int, dict[str, Any]] = {}
    for rank in ranks:
        records = steps[rank]
        if len(records) != SLICE_STEPS:
            raise ValueError(f"{arm} rank {rank} step count mismatch")
        for relative, record in enumerate(records):
            expected_step = start + relative
            expected_scheduled = min(
                stop,
                start + PREFETCH_STEPS * (((relative + 1) // PREFETCH_STEPS) + 2),
            )
            if (
                record["step"] != expected_step
                or record["relative"] != relative
                or record["items"] != PER_RANK_BS
                or record["raw_bytes"] <= 0
                or record["scheduled"] != expected_scheduled
            ):
                raise ValueError(f"{arm} rank {rank} step contract mismatch at {expected_step}")
        if len(passes[rank]) != 1:
            raise ValueError(f"{arm} rank {rank} must have exactly one PASS")
        summary = passes[rank][0]
        waits = [record["wait_s"] for record in records]
        expected_overall_ledger = core.canonical_sha256(
            {
                "schema_version": "offline_arb_w16_fd_crossover_step_ledger_v1",
                "rank": rank,
                "start_step": start,
                "stop_step": stop,
                "step_hashes": [
                    {"step": record["step"], "sha256": record["item_ledger_sha256"]}
                    for record in records
                ],
            }
        )
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
            or summary["step_ledger_sha256"] != expected_overall_ledger
            or summary["ledger_schema"] != "plan_identity_plus_raw_byte_length_v1"
        ):
            raise ValueError(f"{arm} rank {rank} PASS accounting/ledger mismatch")
        result[rank] = {"steps": records, "summary": summary}
        _RUNNER_EVIDENCE[(arm, rank)] = result[rank]
    return result


def load_telemetry(path: Path, **kwargs: Any) -> dict[str, Any]:
    arm = str(kwargs["arm"])
    rank = int(kwargs["rank"])
    mode = mode_for_arm(arm)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith("\n") or not raw.strip():
                raise ValueError(f"{arm} rank {rank} telemetry line {line_number} malformed")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{arm} rank {rank} telemetry record is not an object")
            records.append(value)
    identities = [item for item in records if item.get("record_type") == "identity"]
    if len(identities) != 1 or identities[0] is not records[0]:
        raise ValueError(f"{arm} rank {rank} identity is not unique first")
    if identities[0].get("gpic_source_consistency_mode") != mode:
        raise ValueError(f"{arm} rank {rank} telemetry identity mode mismatch")
    if (
        identities[0].get("gpic_threads_per_node") != 16
        or identities[0].get("gpic_workers_per_rank") != 2
        or kwargs.get("threads") != 16
    ):
        raise ValueError(f"{arm} rank {rank} GPIC thread-resolution identity mismatch")
    gpic_windows = 0
    files_total = 0
    fstat_calls_total = 0
    path_stat_calls_total = 0
    counter_contract_exact = True
    path_stat_timing_zero = True
    for record in records:
        if record.get("record_type") != "window" or record.get("domain") != "gpic":
            continue
        detail = record.get("image_detail")
        if not isinstance(detail, dict):
            raise ValueError(f"{arm} rank {rank} GPIC image_detail malformed")
        if detail.get("gpic_source_consistency_mode") != mode:
            raise ValueError(f"{arm} rank {rank} GPIC detail mode mismatch")
        files = _strict_int(detail.get("files"), "GPIC files", minimum=1)
        fstat_calls = _strict_int(detail.get("fstat_calls"), "GPIC fstat_calls")
        path_stat_calls = _strict_int(detail.get("path_stat_calls"), "GPIC path_stat_calls")
        thread_sum = detail.get("thread_sum_s")
        operation_max = detail.get("operation_max_s")
        if not isinstance(thread_sum, dict) or not isinstance(operation_max, dict):
            raise ValueError("GPIC timing maps malformed")
        path_sum = _finite_number(thread_sum.get("path_stat"), "GPIC path_stat sum")
        path_max = _finite_number(operation_max.get("path_stat"), "GPIC path_stat max")
        expected_path_calls = 0 if mode == "fd_stable" else 2 * files
        counter_contract_exact = counter_contract_exact and (
            fstat_calls == 2 * files and path_stat_calls == expected_path_calls
        )
        if mode == "fd_stable":
            path_stat_timing_zero = path_stat_timing_zero and path_sum == 0.0 and path_max == 0.0
        gpic_windows += 1
        files_total += files
        fstat_calls_total += fstat_calls
        path_stat_calls_total += path_stat_calls
    if gpic_windows <= 0:
        raise ValueError(f"{arm} rank {rank} has no GPIC telemetry windows")
    _TELEMETRY_AUDIT[(arm, rank)] = {
        "mode": mode,
        "gpic_threads_per_node": 16,
        "gpic_workers_per_rank": 2,
        "gpic_windows": gpic_windows,
        "files": files_total,
        "fstat_calls": fstat_calls_total,
        "path_stat_calls": path_stat_calls_total,
        "counter_contract_exact": counter_contract_exact,
        "path_stat_timing_zero": path_stat_timing_zero,
    }
    return _ORIGINAL_LOAD_TELEMETRY(path, **kwargs)


def _validate_extra_artifacts(
    master_path: Path, repo: Path
) -> dict[str, tuple[Path, str]]:
    master = _ORIGINAL_READ_OBJECT(master_path, "raw crossover master")
    if master.get("schema_version") != "offline_arb_w16_gpic_fd_crossover_master_aggregate_v1":
        raise ValueError("raw crossover master schema mismatch")
    preflight_ref = master.get("preflight")
    if not isinstance(preflight_ref, dict):
        raise ValueError("raw crossover master preflight reference malformed")
    preflight_path = Path(str(preflight_ref.get("path", "")))
    if preflight_ref.get("sha256") != core.sha256_file(preflight_path):
        raise ValueError("raw crossover master/preflight SHA mismatch")
    preflight = _ORIGINAL_READ_OBJECT(preflight_path, "raw crossover preflight")
    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("raw crossover preflight artifacts malformed")
    closure: dict[str, tuple[Path, str]] = {}
    for label, relative in EXTRA_ARTIFACT_PATHS.items():
        path = repo / relative
        declaration = artifacts.get(label)
        if not isinstance(declaration, dict) or set(declaration) != {"path", "bytes", "sha256"}:
            raise ValueError(f"extra artifact declaration malformed: {label}")
        core.require_file(path, f"extra artifact {label}")
        digest = core.sha256_file(path)
        if (
            declaration.get("path") != str(path.resolve(strict=True))
            or declaration.get("bytes") != path.stat().st_size
            or declaration.get("sha256") != digest
        ):
            raise ValueError(f"extra artifact capture/live mismatch: {label}")
        closure[label] = (path, digest)
    return closure


def _pair_raw_exact(left: str, right: str) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for rank in range(WORLD_SIZE):
        left_value = _RUNNER_EVIDENCE.get((left, rank))
        right_value = _RUNNER_EVIDENCE.get((right, rank))
        if not isinstance(left_value, dict) or not isinstance(right_value, dict):
            raise ValueError(f"crossover parser evidence missing for rank {rank}: {left}/{right}")
        left_steps = left_value["steps"]
        right_steps = right_value["steps"]
        if len(left_steps) != len(right_steps):
            mismatches.append({"rank": rank, "reason": "step_count"})
            continue
        for relative, (lhs, rhs) in enumerate(zip(left_steps, right_steps)):
            if (
                lhs["relative"] != rhs["relative"]
                or lhs["items"] != rhs["items"]
                or lhs["raw_bytes"] != rhs["raw_bytes"]
                or lhs["item_ledger_sha256"] != rhs["item_ledger_sha256"]
            ):
                mismatches.append(
                    {
                        "rank": rank,
                        "relative_step": relative,
                        "left_raw_bytes": lhs["raw_bytes"],
                        "right_raw_bytes": rhs["raw_bytes"],
                        "left_ledger_sha256": lhs["item_ledger_sha256"],
                        "right_ledger_sha256": rhs["item_ledger_sha256"],
                    }
                )
                if len(mismatches) >= 16:
                    break
        if left_value["summary"]["raw_bytes"] != right_value["summary"]["raw_bytes"]:
            mismatches.append({"rank": rank, "reason": "rank_raw_bytes"})
    return {
        "left": left,
        "right": right,
        "ranks": WORLD_SIZE,
        "steps_per_rank": SLICE_STEPS,
        "ledger_schema": "plan_identity_plus_raw_byte_length_v1",
        "exact": not mismatches,
        "mismatches": mismatches,
    }


def _positive_drift(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        raise ValueError("capacity drift inputs must be positive")
    return abs(left - right) / max(left, right)


def _positive_speedup(numerator: float, denominator: float, label: str) -> float:
    if (
        isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, (int, float))
        or not isinstance(denominator, (int, float))
        or not math.isfinite(float(numerator))
        or not math.isfinite(float(denominator))
        or numerator <= 0
        or denominator <= 0
    ):
        raise ValueError(f"{label} speedup inputs must be finite/positive")
    return float(numerator) / float(denominator)


def _crossover_performance(
    arms: Mapping[str, dict[str, Any]], strict_arm: str, fd_arm: str
) -> dict[str, Any]:
    strict_value = arms[strict_arm]
    fd_value = arms[fd_arm]
    strict_capacity = float(
        strict_value["domain_min_rank_capacity_ratio"]["gpic"]
    )
    fd_capacity = float(fd_value["domain_min_rank_capacity_ratio"]["gpic"])
    strict_elapsed = strict_value["rank_runner_elapsed_per_step_s"]
    fd_elapsed = fd_value["rank_runner_elapsed_per_step_s"]
    expected_ranks = {str(rank) for rank in range(WORLD_SIZE)}
    if set(strict_elapsed) != expected_ranks or set(fd_elapsed) != expected_ranks:
        raise ValueError(f"{strict_arm}/{fd_arm} runner rank set mismatch")
    rank_speedups = {
        str(rank): _positive_speedup(
            float(strict_elapsed[str(rank)]),
            float(fd_elapsed[str(rank)]),
            f"{strict_arm}/{fd_arm} rank {rank} elapsed",
        )
        for rank in range(WORLD_SIZE)
    }
    strict_wait = strict_value["global_critical_wait"]
    fd_wait = fd_value["global_critical_wait"]
    strict_window = strict_value["combined_critical_windows"]
    fd_window = fd_value["combined_critical_windows"]
    return {
        "strict_arm": strict_arm,
        "fd_arm": fd_arm,
        "gpic_min_rank_capacity_ratio": {
            "strict": strict_capacity,
            "fd": fd_capacity,
        },
        "gpic_capacity_speedup_fd_over_strict": _positive_speedup(
            fd_capacity, strict_capacity, f"{strict_arm}/{fd_arm} GPIC capacity"
        ),
        "runner_elapsed_speedup_strict_over_fd": {
            "per_rank": rank_speedups,
            "min": min(rank_speedups.values()),
            "median": statistics.median(rank_speedups.values()),
            "mean": statistics.fmean(rank_speedups.values()),
        },
        "global_critical_wait_speedup_strict_over_fd": {
            key: _positive_speedup(
                float(strict_wait[key]),
                float(fd_wait[key]),
                f"{strict_arm}/{fd_arm} wait {key}",
            )
            for key in ("mean_s", "p95_s", "max_s")
        },
        "combined_critical_window_speedup_strict_over_fd": {
            key: _positive_speedup(
                float(strict_window[key]),
                float(fd_window[key]),
                f"{strict_arm}/{fd_arm} window {key}",
            )
            for key in ("mean_s", "p95_s", "max_s")
        },
    }


def _install_core_compatibility(analysis_attempt_root: Path) -> None:
    core.ARMS = ARMS
    core.THREAD_GROUPS = {"FD": ("fd_x", "fd_y"), "STRICT": ("strict_x", "strict_y")}
    core.GATE_RELATIVE_FILES = GATE_RELATIVE_FILES
    core.read_object = compatibility_read_object
    core.expected_arm_nonce = expected_arm_nonce
    core.parse_runner_log = parse_runner_log
    core.load_telemetry = load_telemetry
    # The reviewed r4 analyzer writes one analysis-time sidecar snapshot beside
    # the immutable capture.  Compile the same function with exactly that one
    # destination redirected into this analyzer-attempt namespace.  Any source
    # drift that changes the audited statement is fatal instead of silently
    # falling back to capture mutation.
    source = textwrap.dedent(inspect.getsource(core.analyze))
    needle = (
        'analysis_full_path = result_root / '
        'f"sidecarfull.analysis.{result_root.name}.json"'
    )
    replacement = (
        'analysis_full_path = _CROSSOVER_ANALYSIS_ATTEMPT_ROOT / '
        'f"sidecarfull.analysis.{result_root.name}.json"'
    )
    if source.count(needle) != 1:
        raise ValueError("r4 analyzer analysis-sidecar redirect statement drifted")
    transformed = source.replace(needle, replacement)
    core.__dict__["_CROSSOVER_ANALYSIS_ATTEMPT_ROOT"] = analysis_attempt_root
    namespace: dict[str, Any] = {}
    exec(
        compile(transformed, str(Path(core.__file__).resolve(strict=True)), "exec"),
        core.__dict__,
        namespace,
    )
    core.analyze = namespace["analyze"]


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_id != RUN_ID:
        raise ValueError(f"crossover run-id mismatch: {args.run_id!r}")
    _RUNNER_EVIDENCE.clear()
    _TELEMETRY_AUDIT.clear()
    analysis_attempt_root = _validate_analysis_attempt_identity(args)
    master_path = Path(args.master)
    repo = Path(args.repo).resolve(strict=True)
    extra_closure = _validate_extra_artifacts(master_path, repo)
    _install_core_compatibility(analysis_attempt_root)
    result = core.analyze(args)

    pair_results = {
        f"{left}_vs_{right}": _pair_raw_exact(left, right)
        for left, right in CROSSOVER_PAIRS
    }
    raw_exact = all(value["exact"] for value in pair_results.values())
    expected_keys = {(arm, rank) for arm, *_ in ARMS for rank in range(WORLD_SIZE)}
    if set(_TELEMETRY_AUDIT) != expected_keys:
        raise ValueError("mode telemetry audit rank/arm exact set mismatch")
    telemetry_by_arm: dict[str, Any] = {}
    for arm, *_ in ARMS:
        ranks = {str(rank): _TELEMETRY_AUDIT[(arm, rank)] for rank in range(WORLD_SIZE)}
        telemetry_by_arm[arm] = {
            "mode": mode_for_arm(arm),
            "gpic_threads_per_node": 16,
            "gpic_workers_per_rank": 2,
            "gpic_windows": sum(value["gpic_windows"] for value in ranks.values()),
            "files": sum(value["files"] for value in ranks.values()),
            "fstat_calls": sum(value["fstat_calls"] for value in ranks.values()),
            "path_stat_calls": sum(value["path_stat_calls"] for value in ranks.values()),
            "counter_contract_exact": all(value["counter_contract_exact"] for value in ranks.values()),
            "path_stat_timing_zero": all(value["path_stat_timing_zero"] for value in ranks.values()),
            "ranks": ranks,
        }
        result["arms"][arm]["gpic_source_consistency_mode"] = mode_for_arm(arm)

    fd_left = result["arms"]["fd_x"]["domain_min_rank_capacity_ratio"]["gpic"]
    fd_right = result["arms"]["fd_y"]["domain_min_rank_capacity_ratio"]["gpic"]
    strict_left = result["arms"]["strict_x"]["domain_min_rank_capacity_ratio"]["gpic"]
    strict_right = result["arms"]["strict_y"]["domain_min_rank_capacity_ratio"]["gpic"]
    fd_drift = _positive_drift(fd_left, fd_right)
    strict_drift = _positive_drift(strict_left, strict_right)
    fd_absolute_pass = all(result["arms"][arm]["status"] == "PASS" for arm in ("fd_x", "fd_y"))
    fd_counter_pass = all(
        telemetry_by_arm[arm]["counter_contract_exact"]
        and telemetry_by_arm[arm]["path_stat_calls"] == 0
        and telemetry_by_arm[arm]["path_stat_timing_zero"]
        and telemetry_by_arm[arm]["fstat_calls"] == 2 * telemetry_by_arm[arm]["files"]
        for arm in ("fd_x", "fd_y")
    )
    strict_counter_pass = all(
        telemetry_by_arm[arm]["counter_contract_exact"]
        and telemetry_by_arm[arm]["path_stat_calls"] == 2 * telemetry_by_arm[arm]["files"]
        and telemetry_by_arm[arm]["fstat_calls"] == 2 * telemetry_by_arm[arm]["files"]
        for arm in ("strict_x", "strict_y")
    )
    strict_stable = strict_drift <= STRICT_ENV_DRIFT_MAX and strict_counter_pass
    fd_stable = fd_drift <= FD_PAIR_DRIFT_MAX
    selectable = strict_stable and fd_absolute_pass and fd_stable and fd_counter_pass and raw_exact
    crossover_performance = {
        "x": _crossover_performance(result["arms"], "strict_x", "fd_x"),
        "y": _crossover_performance(result["arms"], "strict_y", "fd_y"),
    }
    capacity_speedups = {
        label: value["gpic_capacity_speedup_fd_over_strict"]
        for label, value in crossover_performance.items()
    }
    min_capacity_speedup = min(capacity_speedups.values())
    capacity_speedup_drift = _positive_drift(
        capacity_speedups["x"], capacity_speedups["y"]
    )
    min_runner_speedup = min(
        value["runner_elapsed_speedup_strict_over_fd"]["min"]
        for value in crossover_performance.values()
    )
    min_wait_mean_speedup = min(
        value["global_critical_wait_speedup_strict_over_fd"]["mean_s"]
        for value in crossover_performance.values()
    )
    min_window_mean_speedup = min(
        value["combined_critical_window_speedup_strict_over_fd"]["mean_s"]
        for value in crossover_performance.values()
    )
    gpuval_candidate = (
        raw_exact
        and fd_counter_pass
        and strict_counter_pass
        and min_capacity_speedup >= GPUVAL_CANDIDATE_MIN_GPIC_SPEEDUP
    )
    decision = (
        "SELECT_FD_STABLE"
        if selectable
        else "NO_GO"
        if not raw_exact or not fd_counter_pass
        else "INCONCLUSIVE"
        if not strict_stable
        else "NO_GO"
    )

    result.update(
        {
            "schema_version": ANALYSIS_SCHEMA,
            "status": "PASS" if selectable else "NO_GO",
            "decision": decision,
            "semantic_external_gate": "PENDING",
            "production_selection_qualified": False,
            "selection_candidates": ["fd_stable"] if selectable else [],
            "inconclusive_groups": ["strict_environment"] if decision == "INCONCLUSIVE" else [],
            "crossover_raw_exact": {
                "status": "PASS" if raw_exact else "NO_GO",
                "pairs": pair_results,
            },
            "crossover_performance": {
                "pairs": crossover_performance,
                "gpic_capacity_speedup_fd_over_strict": capacity_speedups,
                "min_gpic_capacity_speedup": min_capacity_speedup,
                "gpic_capacity_speedup_pair_drift": capacity_speedup_drift,
                "min_runner_elapsed_speedup": min_runner_speedup,
                "min_global_critical_wait_mean_speedup": min_wait_mean_speedup,
                "min_combined_critical_window_mean_speedup": min_window_mean_speedup,
            },
            "gpuval_diagnostic": {
                "status": "GPUVAL_CANDIDATE" if gpuval_candidate else "NOT_CANDIDATE",
                "production_selection_qualified": False,
                "semantic_external_gate": "PENDING",
                "minimum_required_gpic_capacity_speedup": GPUVAL_CANDIDATE_MIN_GPIC_SPEEDUP,
                "raw_exact": raw_exact,
                "fd_counter_contract_pass": fd_counter_pass,
                "strict_counter_contract_pass": strict_counter_pass,
                "both_slices_significantly_faster": (
                    min_capacity_speedup >= GPUVAL_CANDIDATE_MIN_GPIC_SPEEDUP
                ),
                "min_observed_gpic_capacity_speedup": min_capacity_speedup,
                "meaning": (
                    "Diagnostic permission to run the separate semantic gate and 2x8 GPUVAL; "
                    "never a FORMAL/production selector and never a substitute for absolute gates."
                ),
            },
            "mode_telemetry": telemetry_by_arm,
            "qualification": {
                "fd_arms_absolute_pass": fd_absolute_pass,
                "fd_gpic_min_rank_capacity_ratio": {"fd_x": fd_left, "fd_y": fd_right},
                "fd_pair_drift": fd_drift,
                "fd_pair_drift_max": FD_PAIR_DRIFT_MAX,
                "fd_pair_stable": fd_stable,
                "fd_counter_contract_pass": fd_counter_pass,
                "strict_gpic_min_rank_capacity_ratio": {
                    "strict_x": strict_left,
                    "strict_y": strict_right,
                },
                "strict_environment_drift": strict_drift,
                "strict_environment_drift_max": STRICT_ENV_DRIFT_MAX,
                "strict_counter_contract_pass": strict_counter_pass,
                "strict_environment_stable": strict_stable,
                "same_slice_plan_domain_raw_bytes_exact": raw_exact,
                "selectable": selectable,
            },
        }
    )
    result.pop("thread_qualification", None)
    result["thresholds"]["fd_pair_drift_max"] = FD_PAIR_DRIFT_MAX
    result["thresholds"]["strict_environment_drift_max"] = STRICT_ENV_DRIFT_MAX
    result["thresholds"]["gpuval_candidate_min_gpic_speedup"] = (
        GPUVAL_CANDIDATE_MIN_GPIC_SPEEDUP
    )
    result["identity"]["gpic_threads_per_node"] = 16
    result["identity"]["mode_by_arm"] = dict(MODE_BY_ARM)
    result["identity"]["analyzer_jobtag"] = args.analyzer_jobtag
    result["identity"]["analysis_attempt_nonce"] = args.analysis_attempt_nonce
    result["identity"]["analysis_attempt_root"] = str(analysis_attempt_root)
    result["scope"] = (
        "CPU raw-supply crossover only. Raw equality binds ordered plan identity/domain and "
        "raw byte length without payload hashing in the timed interval; payload-content "
        "semantic equivalence remains a separate semantic gate before GPU/formal use."
    )
    for label, (path, expected_sha) in extra_closure.items():
        core.require_file(path, f"final extra closure {label}")
        if core.sha256_file(path) != expected_sha:
            raise ValueError(f"extra crossover dependency changed during analysis: {label}")
    result["evidence"]["crossover_extension_artifacts"] = {
        label: {"path": str(path), "sha256": digest}
        for label, (path, digest) in extra_closure.items()
    }
    return result


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    core.atomic_write(path, value)


def _output_attestation_path(output: Path) -> Path:
    return output.with_name(output.name + ".attestation.json")


def _live_artifact_postcheck(args: argparse.Namespace) -> dict[str, Any]:
    master_path = Path(args.master)
    raw_master = _ORIGINAL_READ_OBJECT(master_path, "attestation master")
    preflight_ref = raw_master.get("preflight")
    if not isinstance(preflight_ref, dict):
        raise ValueError("attestation master/preflight reference malformed")
    preflight_path = Path(str(preflight_ref.get("path", "")))
    raw_preflight = _ORIGINAL_READ_OBJECT(preflight_path, "attestation preflight")
    artifacts = raw_preflight.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("attestation preflight artifacts malformed")
    result: dict[str, Any] = {}
    for label, declaration in sorted(artifacts.items()):
        if not isinstance(declaration, dict) or set(declaration) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ValueError(f"attestation artifact declaration malformed: {label}")
        path = Path(str(declaration["path"]))
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"attestation artifact is not a real regular file: {label}")
        digest = core.sha256_file(path)
        if path.stat().st_size != declaration["bytes"] or digest != declaration["sha256"]:
            raise ValueError(f"attestation live artifact drift: {label}")
        result[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    return result


def _write_output_and_attestation(
    args: argparse.Namespace, result: dict[str, Any]
) -> Path:
    if args.output is None:
        raise ValueError("crossover analyzer requires an output for detached attestation")
    output = Path(args.output)
    attestation_path = _output_attestation_path(output)
    evidence = result.setdefault("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("analysis evidence must be an object")
    evidence["analysis_output_attestation_expected"] = {
        "path": str(attestation_path),
        "schema_version": "offline_arb_w16_gpic_fd_crossover_output_attestation_v1",
    }
    atomic_write(output, result)
    master_path = Path(args.master)
    capture_input_attestation = Path(args.analysis_attempt_root) / "capture_input_attestation.json"
    if (
        capture_input_attestation.is_symlink()
        or not capture_input_attestation.is_file()
        or capture_input_attestation.stat().st_size <= 0
    ):
        raise ValueError("capture input attestation is missing/empty/non-regular")
    live_artifacts = _live_artifact_postcheck(args)
    attestation = {
        "schema_version": "offline_arb_w16_gpic_fd_crossover_output_attestation_v1",
        "status": result.get("status"),
        "decision": result.get("decision"),
        "run_id": args.run_id,
        "source_jobtag": master_path.parent.name,
        "analyzer_jobtag": args.analyzer_jobtag,
        "analysis_attempt_nonce": args.analysis_attempt_nonce,
        "analysis_attempt_root": args.analysis_attempt_root,
        "created_at_unix": int(time.time()),
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": core.sha256_file(output),
        },
        "capture_master": {
            "path": str(master_path),
            "bytes": master_path.stat().st_size,
            "sha256": core.sha256_file(master_path),
        },
        "capture_input_attestation": {
            "path": str(capture_input_attestation),
            "bytes": capture_input_attestation.stat().st_size,
            "sha256": core.sha256_file(capture_input_attestation),
        },
        "live_artifact_postcheck": live_artifacts,
        "live_artifact_postcheck_sha256": core.canonical_sha256(live_artifacts),
    }
    atomic_write(attestation_path, attestation)
    return attestation_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--animefreq", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--analysis-attempt-root", required=True)
    parser.add_argument("--analyzer-jobtag", required=True)
    parser.add_argument("--analysis-attempt-nonce", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = analyze(args)
    except BaseException as exc:
        result = {
            "schema_version": ANALYSIS_SCHEMA,
            "status": "FATAL",
            "semantic_external_gate": "PENDING",
            "production_selection_qualified": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if args.output:
            try:
                _write_output_and_attestation(args, result)
            except BaseException as output_exc:
                print(f"FATAL crossover analyzer output failure: {output_exc}", file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    _write_output_and_attestation(args, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
