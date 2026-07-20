#!/usr/bin/env python3
"""Capture a W16 strict_path/fd_stable crossover ABBA on two CPU nodes.

The already-reviewed r4 W16 runner remains the orchestration/closure kernel and
is itself a preflight-bound artifact.  This extension changes only the arm
matrix and adds mode-explicit contracts, sentinels, telemetry expectations,
manifests, and aggregates.  Timed work never hashes image payloads: each step's
ordered plan identity and raw byte lengths are hashed only after the arm timer
has stopped.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import stat
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.offline_arb import run_w16_gpic_abba_raw_gate as core


REPO = Path("/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop")
RUN_ID = "stage256_offline_exact_w16_gpic_fd_crossover_abba_20260712r1"
RUNNER_SELF = REPO / "scripts/offline_arb/run_w16_gpic_fd_crossover_abba.py"
CAPTURE_YAML = REPO / "jobs/offline_arb_w16_gpic_fd_crossover_abba.yaml"
ANALYZER = REPO / "scripts/offline_arb/analyze_w16_gpic_fd_crossover_abba.py"
ANALYZE_YAML = REPO / "jobs/offline_arb_w16_gpic_fd_crossover_abba_analyze.yaml"
CORE_RUNNER = REPO / "scripts/offline_arb/run_w16_gpic_abba_raw_gate.py"
CORE_ANALYZER = REPO / "scripts/offline_arb/analyze_w16_gpic_abba_raw_gate.py"
LOCAL_PUBLICATION_BASE = Path("/local/evan/offline_arb_w16_gpic_abba")

WORLD_SIZE = 16
LOCAL_WORLD_SIZE = 8
PER_RANK_BS = 36
PREFETCH_STEPS = 4
SLICE_STEPS = 192
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
STEP_PREFIX = "W16_FD_CROSSOVER_STEP_RETIRED "
PASS_PREFIX = "W16_FD_CROSSOVER_RANK_PASS "
OLD_PASS_PREFIX = "W16_ABBA_RAW_RANK_PASS "
ARM_IN_LINE_RE = re.compile(r"(?:^| )arm=([a-z0-9_]+)(?: |$)")

_ORIGINAL_LIVE_ARTIFACTS = core._live_artifacts
_ORIGINAL_DATASET_KWARGS = core._dataset_kwargs
_ORIGINAL_CREATE_PREFLIGHT = core._create_preflight
_ORIGINAL_VALIDATE_PREFLIGHT = core._validate_preflight
_ORIGINAL_ARM_CONTRACT = core._arm_contract
_ORIGINAL_VALIDATE_ARM_CONTRACT = core._validate_arm_contract
_ORIGINAL_PRESERVE_NODE_ARM = core._preserve_node_arm
_ORIGINAL_CREATE_ARM_AGGREGATE = core._create_arm_aggregate
_ORIGINAL_ATOMIC_WRITE_JSON = core.atomic_write_json
_RANK_RESULTS: dict[str, dict[str, Any]] = {}


def mode_for_arm(arm: str) -> str:
    try:
        return MODE_BY_ARM[arm]
    except KeyError as exc:
        raise RuntimeError(f"unknown crossover arm: {arm!r}") from exc


def capture_attempt_nonce(jobtag: str) -> str:
    return core.canonical_sha256(
        {
            "schema_version": "offline_arb_w16_gpic_abba_attempt_nonce_v1",
            "run_id": RUN_ID,
            "jobtag": jobtag,
        }
    )


def arm_nonce(attempt: str, arm: str, threads: int, start: int, stop: int) -> str:
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


def _atomic_sentinel(line: str) -> None:
    if (
        not isinstance(line, str)
        or not line
        or "\n" in line
        or "\r" in line
        or not line.startswith((STEP_PREFIX, PASS_PREFIX))
    ):
        raise RuntimeError("crossover rank sentinel is malformed")
    payload = (line + "\n").encode("utf-8", errors="strict")
    descriptor = sys.stdout.fileno()
    if not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
        raise RuntimeError("crossover rank sentinel stdout is not a pipe")
    pipe_buf = int(os.fpathconf(descriptor, "PC_PIPE_BUF"))
    if pipe_buf <= 0 or len(payload) > pipe_buf:
        raise RuntimeError(
            f"crossover sentinel exceeds PIPE_BUF: bytes={len(payload)} pipe_buf={pipe_buf}"
        )
    written = os.write(descriptor, payload)
    if written != len(payload):
        raise RuntimeError(
            f"partial crossover sentinel write: wrote={written} expected={len(payload)}"
        )


def write_rank_sentinel(line: str) -> None:
    """Accept native extension lines and mode-augment the core PASS summary."""
    if line.startswith(STEP_PREFIX) or line.startswith(PASS_PREFIX):
        _atomic_sentinel(line)
        return
    if not line.startswith(OLD_PASS_PREFIX):
        raise RuntimeError("unexpected sentinel requested by crossover core")
    match = ARM_IN_LINE_RE.search(line)
    if match is None:
        raise RuntimeError("core PASS sentinel lacks arm identity")
    arm = match.group(1)
    result = _RANK_RESULTS.get(arm)
    if not isinstance(result, dict):
        raise RuntimeError(f"crossover PASS lacks timed result for {arm}")
    translated = PASS_PREFIX + line[len(OLD_PASS_PREFIX) :]
    translated = translated.replace(
        f"arm={arm} ",
        f"arm={arm} mode={mode_for_arm(arm)} ",
        1,
    )
    translated += (
        f" step_ledger_sha256={result['step_ledger_sha256']}"
        f" plan_raw_ledger_schema={result['plan_raw_ledger_schema']}"
    )
    _atomic_sentinel(translated)


def _dataset_kwargs(args: argparse.Namespace, rank: int) -> dict[str, Any]:
    kwargs = _ORIGINAL_DATASET_KWARGS(args, rank)
    configured_threads = kwargs.get("gpic_threads_per_node")
    environment_threads = os.environ.get("OFFLINE_ARB_GPIC_THREADS_PER_NODE")
    if (
        type(configured_threads) is not int
        or configured_threads != 16
        or args.gpic_threads_per_node != 16
        or environment_threads != "16"
    ):
        raise RuntimeError(
            "crossover GPIC thread resolution drift: "
            f"config={configured_threads!r} args={args.gpic_threads_per_node!r} "
            f"env={environment_threads!r}"
        )
    mode = mode_for_arm(args.arm)
    kwargs["gpic_source_consistency_mode"] = mode
    return kwargs


def _run_raw_arm(dataset: Any, args: argparse.Namespace, rank: int) -> dict[str, Any]:
    missing = [
        name
        for name in core.REQUIRED_RAW_API
        if not callable(getattr(dataset, name, None))
    ]
    if missing:
        raise RuntimeError(f"offline raw-supply API drifted; missing={missing}")
    mode = mode_for_arm(args.arm)
    if getattr(dataset, "_gpic_source_consistency_mode", None) != mode:
        raise AssertionError("dataset GPIC source-consistency mode drift")
    if int(dataset.plan_header["per_rank_bs"]) != PER_RANK_BS:
        raise AssertionError("plan per-rank batch size drift")
    if int(dataset.plan_header["total_steps"]) < args.stop_step:
        raise AssertionError("plan does not cover requested crossover arm")
    if int(dataset.prefetch_steps) != PREFETCH_STEPS:
        raise AssertionError("prefetch depth drift")
    if int(dataset._gpic_threads_per_node) != args.gpic_threads_per_node:
        raise AssertionError("GPIC node thread identity drift")
    expected_workers = args.gpic_threads_per_node // LOCAL_WORLD_SIZE
    if int(dataset._gpic_workers) != expected_workers:
        raise AssertionError("GPIC per-rank worker identity drift")
    if dataset._gpic_payload_in_memory is not True:
        raise AssertionError("crossover raw gate requires in-memory GPIC payloads")

    original_enqueue = dataset._enqueue_telemetry
    emitted: set[tuple[int, str]] = set()
    duplicates: set[tuple[int, str]] = set()
    window_lock = threading.Lock()

    def observed_enqueue(record: Any) -> bool:
        accepted = original_enqueue(record)
        if accepted and isinstance(record, dict) and record.get("record_type") == "window":
            key = (int(record.get("start", -1)), str(record.get("domain")))
            with window_lock:
                if key in emitted:
                    duplicates.add(key)
                emitted.add(key)
        return accepted

    dataset._enqueue_telemetry = observed_enqueue
    dataset.set_start_step(args.start_step)
    core.dist.barrier()
    dataset._start_pipeline()
    if int(dataset._scheduled_until) != min(args.stop_step, args.start_step + 8):
        raise AssertionError("initial bounded prefetch drift")

    bytes_by_domain: Counter[str] = Counter()
    items_by_domain: Counter[str] = Counter()
    total_raw_bytes = 0
    total_items = 0
    total_wait_s = 0.0
    max_wait_s = 0.0
    expected_windows: set[tuple[int, str]] = set()
    step_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for relative_step, step in enumerate(range(args.start_step, args.stop_step)):
        expected_bucket, expected_items = dataset._plan_items_for_step(step)
        window_start = args.start_step + (
            (step - args.start_step) // PREFETCH_STEPS
        ) * PREFETCH_STEPS
        expected_windows.update(
            (window_start, str(item.domain)) for item in expected_items
        )
        if (
            len(expected_items) != PER_RANK_BS
            or dataset._step_bucket.get(step) != expected_bucket
        ):
            raise AssertionError(f"rank {rank} step {step}: scheduled identity mismatch")
        core.dist.barrier()
        wait_started = time.perf_counter()
        prepared = dataset._wait_for_step(step)
        wait_s = time.perf_counter() - wait_started
        if len(prepared) != PER_RANK_BS:
            raise AssertionError(f"rank {rank} step {step}: item count mismatch")
        step_raw_bytes = 0
        item_ledger: list[list[Any]] = []
        for slot, (actual, expected) in enumerate(zip(prepared, expected_items)):
            planned = core._planned_identity(expected)
            if core._planned_identity(actual.planned) != planned or planned[1] != slot:
                raise AssertionError(
                    f"rank {rank} step {step} slot {slot}: prepared/plan identity mismatch"
                )
            catalog = actual.catalog
            if (
                int(catalog.get("row_id", -1)) != int(expected.row_id)
                or str(catalog.get("domain")) != str(expected.domain)
                or int(catalog.get("bucket_idx", -1)) != int(expected.bucket_idx)
                or int(catalog.get("width", 0)) <= 0
                or int(catalog.get("height", 0)) <= 0
            ):
                raise AssertionError(f"rank {rank} step {step} slot {slot}: catalog drift")
            if actual.local_path is not None or not isinstance(actual.raw, bytes):
                raise AssertionError("crossover arm left the formal in-memory raw path")
            raw = dataset._raw_for_item(actual)
            if not isinstance(raw, bytes) or not raw:
                raise AssertionError("crossover raw payload is empty/non-bytes")
            size = len(raw)
            # No payload digest is computed inside the timed interval.  The exact
            # ordered ledger binds every plan item/domain and its raw byte length.
            item_ledger.append([slot, *planned, size])
            step_raw_bytes += size
            total_raw_bytes += size
            total_items += 1
            domain = str(expected.domain)
            items_by_domain[domain] += 1
            bytes_by_domain[domain] += size
            del raw
        dataset._retire_step(step, prepared)
        if step in dataset._ready or step in dataset._expected or step in dataset._step_bucket:
            raise AssertionError(f"rank {rank} step {step}: retire left live state")
        dataset._next_step = step + 1
        if dataset._next_step < args.stop_step:
            core._maintain_bounded_prefetch(dataset, args.stop_step)
        dataset._refresh_metrics(plan_step=step)
        dataset._emit_state_record(step)
        total_wait_s += wait_s
        max_wait_s = max(max_wait_s, wait_s)
        step_records.append(
            {
                "step": step,
                "relative_step": relative_step,
                "items": len(prepared),
                "raw_bytes": step_raw_bytes,
                "wait_s": wait_s,
                "scheduled_until": int(dataset._scheduled_until),
                "item_ledger": item_ledger,
            }
        )
    if int(dataset._next_step) != args.stop_step or int(dataset._scheduled_until) != args.stop_step:
        raise AssertionError("crossover arm final step/schedule boundary mismatch")
    timed_elapsed_s = time.perf_counter() - started

    telemetry_deadline = time.monotonic() + 60.0
    while True:
        with window_lock:
            observed = set(emitted)
            repeated = set(duplicates)
        if repeated:
            raise AssertionError(f"duplicate telemetry windows: {sorted(repeated)}")
        if not observed.issubset(expected_windows):
            raise AssertionError(f"unexpected telemetry windows: {sorted(observed - expected_windows)}")
        if observed == expected_windows:
            break
        if time.monotonic() >= telemetry_deadline:
            raise TimeoutError(
                "timed out waiting for exact terminal telemetry windows: "
                f"missing={sorted(expected_windows - observed)}"
            )
        time.sleep(0.01)
    dataset._enqueue_telemetry = original_enqueue

    expected_total = SLICE_STEPS * PER_RANK_BS
    if total_items != expected_total:
        raise AssertionError("crossover raw-item total mismatch")
    ledger_schema = "plan_identity_plus_raw_byte_length_v1"
    step_hashes: list[dict[str, Any]] = []
    for record in step_records:
        ledger_hash = core.canonical_sha256(
            {
                "schema_version": ledger_schema,
                "rank": rank,
                "step": record["step"],
                "items": record.pop("item_ledger"),
            }
        )
        record["item_ledger_sha256"] = ledger_hash
        step_hashes.append({"step": record["step"], "sha256": ledger_hash})
    overall_ledger = core.canonical_sha256(
        {
            "schema_version": "offline_arb_w16_fd_crossover_step_ledger_v1",
            "rank": rank,
            "start_step": args.start_step,
            "stop_step": args.stop_step,
            "step_hashes": step_hashes,
        }
    )
    # Sentinel formatting and hashing happen strictly after timed_elapsed_s.
    for record in step_records:
        write_rank_sentinel(
            STEP_PREFIX
            +
            f"arm={args.arm} mode={mode} rank={rank} step={record['step']} "
            f"relative_step={record['relative_step']} items={record['items']} "
            f"raw_bytes={record['raw_bytes']} wait_s={record['wait_s']:.9f} "
            f"scheduled_until={record['scheduled_until']} "
            f"item_ledger_sha256={record['item_ledger_sha256']}"
        )
    result = {
        "rank": rank,
        "mode": mode,
        "raw_items": total_items,
        "raw_bytes": total_raw_bytes,
        "items_by_domain": dict(sorted(items_by_domain.items())),
        "bytes_by_domain": dict(sorted(bytes_by_domain.items())),
        "wait_total_s": total_wait_s,
        "wait_max_s": max_wait_s,
        "elapsed_s": timed_elapsed_s,
        "step_ledger_sha256": overall_ledger,
        "plan_raw_ledger_schema": ledger_schema,
    }
    _RANK_RESULTS[args.arm] = result
    return result


def _live_artifacts() -> dict[str, dict[str, Any]]:
    artifacts = _ORIGINAL_LIVE_ARTIFACTS()
    for label, path in (
        ("crossover_core_runner_source", CORE_RUNNER),
        ("crossover_core_analyzer_source", CORE_ANALYZER),
    ):
        artifacts[label] = {
            "path": str(core.require_file(path, label).resolve(strict=True)),
            "bytes": path.stat().st_size,
            "sha256": core.sha256_file(path),
        }
    return artifacts


def _create_preflight(result_root: Path, jobtag: str, markers: list[Path]) -> dict[str, Any]:
    value = _ORIGINAL_CREATE_PREFLIGHT(result_root, jobtag, markers)
    cfg = core.OmegaConf.load(core.require_file(core.CONFIG, "crossover config"))
    configured_threads = cfg.dataset.get("gpic_threads_per_node")
    if type(configured_threads) is not int or configured_threads != 16:
        raise RuntimeError("crossover config must freeze gpic_threads_per_node=16")
    value["schema_version"] = "offline_arb_w16_gpic_fd_crossover_preflight_v1"
    value["arms"] = [
        {
            "arm": arm,
            "gpic_source_consistency_mode": mode_for_arm(arm),
            "threads": threads,
            "start_step": start,
            "stop_step": stop,
        }
        for arm, threads, start, stop in ARMS
    ]
    value["crossover_pairs"] = [["strict_x", "fd_x"], ["fd_y", "strict_y"]]
    value["gpic_threads_resolution"] = {
        "configured_per_node": 16,
        "required_environment_value": "16",
        "arm_argument_values": [threads for _arm, threads, _start, _stop in ARMS],
        "expected_local_world_size": LOCAL_WORLD_SIZE,
        "expected_workers_per_rank": 2,
    }
    return value


def _validate_preflight(value: dict[str, Any], jobtag: str) -> None:
    expected_arms = [
        {
            "arm": arm,
            "gpic_source_consistency_mode": mode_for_arm(arm),
            "threads": threads,
            "start_step": start,
            "stop_step": stop,
        }
        for arm, threads, start, stop in ARMS
    ]
    if (
        value.get("schema_version") != "offline_arb_w16_gpic_fd_crossover_preflight_v1"
        or value.get("arms") != expected_arms
        or value.get("crossover_pairs") != [["strict_x", "fd_x"], ["fd_y", "strict_y"]]
        or value.get("gpic_threads_resolution")
        != {
            "configured_per_node": 16,
            "required_environment_value": "16",
            "arm_argument_values": [16, 16, 16, 16],
            "expected_local_world_size": 8,
            "expected_workers_per_rank": 2,
        }
    ):
        raise RuntimeError("shared crossover preflight mode contract malformed")
    shadow = copy.deepcopy(value)
    shadow["schema_version"] = "offline_arb_w16_gpic_abba_preflight_v1"
    shadow["arms"] = [
        {key: item[key] for key in ("arm", "threads", "start_step", "stop_step")}
        for item in shadow["arms"]
    ]
    shadow.pop("crossover_pairs")
    shadow.pop("gpic_threads_resolution")
    _ORIGINAL_VALIDATE_PREFLIGHT(shadow, jobtag)


def _arm_contract(*args: Any, **kwargs: Any) -> dict[str, Any]:
    value = _ORIGINAL_ARM_CONTRACT(*args, **kwargs)
    arm = str(value["arm"])
    value["schema_version"] = "offline_arb_w16_gpic_fd_crossover_arm_contract_v1"
    value["gpic_source_consistency_mode"] = mode_for_arm(arm)
    return value


def _validate_arm_contract(value: dict[str, Any], **kwargs: Any) -> None:
    arm = str(kwargs["arm"])
    if (
        value.get("schema_version")
        != "offline_arb_w16_gpic_fd_crossover_arm_contract_v1"
        or value.get("gpic_source_consistency_mode") != mode_for_arm(arm)
    ):
        raise RuntimeError(f"crossover arm mode contract mismatch: {arm}")
    shadow = copy.deepcopy(value)
    shadow["schema_version"] = "offline_arb_w16_gpic_abba_arm_contract_v1"
    shadow.pop("gpic_source_consistency_mode")
    _ORIGINAL_VALIDATE_ARM_CONTRACT(shadow, **kwargs)


def _final_publication_value(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Transform selected core objects before their one and only publication."""
    if not isinstance(value, dict):
        raise RuntimeError("crossover JSON publication value must be an object")
    schema = value.get("schema_version")
    selected_schemas = {
        "offline_arb_w16_gpic_abba_node_manifest_v1",
        "offline_arb_w16_gpic_abba_arm_aggregate_v1",
        "offline_arb_w16_gpic_abba_master_aggregate_v1",
    }
    if schema not in selected_schemas:
        publication_target_name = (
            path.name in {"node_manifest.json", "arm_aggregate.json"}
            or re.fullmatch(r"abba\.[A-Za-z0-9_.-]+\.aggregate\.json", path.name)
            is not None
        )
        if publication_target_name:
            raise RuntimeError("publication target received an unexpected source schema")
        return value
    published = copy.deepcopy(value)
    jobtag = published.get("jobtag")
    attempt = published.get("attempt_nonce")
    if (
        not isinstance(jobtag, str)
        or core.SAFE_COMPONENT_RE.fullmatch(jobtag) is None
        or jobtag in {".", ".."}
        or Path(jobtag).name != jobtag
        or not isinstance(attempt, str)
        or core.SHA_RE.fullmatch(attempt) is None
        or attempt != capture_attempt_nonce(jobtag)
    ):
        raise RuntimeError("publication jobtag/attempt identity is malformed")
    if schema == "offline_arb_w16_gpic_abba_node_manifest_v1":
        arm = str(published.get("arm", ""))
        spec = next((item for item in ARMS if item[0] == arm), None)
        node_rank = published.get("node_rank")
        arm_nonce_value = published.get("arm_nonce")
        if (
            spec is None
            or type(node_rank) is not int
            or node_rank not in range(core.NODES)
            or not isinstance(arm_nonce_value, str)
            or core.SHA_RE.fullmatch(arm_nonce_value) is None
            or (
                published.get("threads"),
                published.get("start_step"),
                published.get("stop_step"),
            )
            != spec[1:]
            or arm_nonce_value != arm_nonce(attempt, *spec)
        ):
            raise RuntimeError("node-manifest arm/topology/nonce identity is malformed")
        expected_path = (
            LOCAL_PUBLICATION_BASE
            / jobtag
            / attempt
            / arm
            / arm_nonce_value
            / f"node{node_rank}"
            / "node_manifest.json"
        )
        if path != expected_path:
            raise RuntimeError("node-manifest schema requested on a foreign canonical path")
        published["schema_version"] = (
            "offline_arb_w16_gpic_fd_crossover_node_manifest_v1"
        )
        published["gpic_source_consistency_mode"] = mode_for_arm(arm)
    elif schema == "offline_arb_w16_gpic_abba_arm_aggregate_v1":
        arm = str(published.get("arm", ""))
        spec = next((item for item in ARMS if item[0] == arm), None)
        arm_nonce_value = published.get("arm_nonce")
        if (
            spec is None
            or not isinstance(arm_nonce_value, str)
            or core.SHA_RE.fullmatch(arm_nonce_value) is None
            or (
                published.get("threads"),
                published.get("start_step"),
                published.get("stop_step"),
            )
            != spec[1:]
            or arm_nonce_value != arm_nonce(attempt, *spec)
        ):
            raise RuntimeError("arm-aggregate topology/nonce identity is malformed")
        expected_path = (
            core.BASE
            / RUN_ID
            / jobtag
            / f"evidence.{jobtag}"
            / arm
            / "arm_aggregate.json"
        )
        if path != expected_path:
            raise RuntimeError("arm-aggregate schema requested on a foreign canonical path")
        published["schema_version"] = (
            "offline_arb_w16_gpic_fd_crossover_arm_aggregate_v1"
        )
        published["gpic_source_consistency_mode"] = mode_for_arm(arm)
    elif schema == "offline_arb_w16_gpic_abba_master_aggregate_v1":
        expected_path = core.BASE / RUN_ID / jobtag / f"abba.{jobtag}.aggregate.json"
        arms = published.get("arms")
        if published.get("run_id") != RUN_ID or path != expected_path:
            raise RuntimeError("master schema requested on a foreign target/identity")
        if not isinstance(arms, list) or len(arms) != len(ARMS):
            raise RuntimeError("master arm publication set is malformed")
        for declaration, spec in zip(arms, ARMS):
            arm = spec[0]
            expected_aggregate = (
                core.BASE
                / RUN_ID
                / jobtag
                / f"evidence.{jobtag}"
                / arm
                / "arm_aggregate.json"
            )
            if (
                not isinstance(declaration, dict)
                or set(declaration) != {"arm", "path", "sha256", "status"}
                or declaration.get("arm") != arm
                or declaration.get("path") != str(expected_aggregate)
                or not isinstance(declaration.get("sha256"), str)
                or core.SHA_RE.fullmatch(declaration["sha256"]) is None
                or declaration.get("status") != "PASS"
            ):
                raise RuntimeError(f"master arm publication identity is malformed: {arm}")
        published["schema_version"] = (
            "offline_arb_w16_gpic_fd_crossover_master_aggregate_v1"
        )
        published["gpic_source_consistency_mode_by_arm"] = dict(MODE_BY_ARM)
        published["crossover_pairs"] = [
            ["strict_x", "fd_x"],
            ["fd_y", "strict_y"],
        ]
        published["semantic_external_gate"] = "PENDING"
        published["production_selection_qualified"] = False
    return published


def _publication_atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    final_value = _final_publication_value(path, value)
    _ORIGINAL_ATOMIC_WRITE_JSON(path, final_value)


def _preserve_node_arm(**kwargs: Any) -> tuple[Path, dict[str, Any]]:
    path, value = _ORIGINAL_PRESERVE_NODE_ARM(**kwargs)
    arm = str(value["arm"])
    if (
        value.get("schema_version")
        != "offline_arb_w16_gpic_fd_crossover_node_manifest_v1"
        or value.get("gpic_source_consistency_mode") != mode_for_arm(arm)
    ):
        raise RuntimeError(f"crossover node manifest was not final on first publish: {arm}")
    return path, value


def _create_arm_aggregate(*args: Any, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
    path, _core_value = _ORIGINAL_CREATE_ARM_AGGREGATE(*args, **kwargs)
    value = core.read_object(path, "first-published crossover arm aggregate")
    arm = str(value["arm"])
    if (
        value.get("schema_version")
        != "offline_arb_w16_gpic_fd_crossover_arm_aggregate_v1"
        or value.get("gpic_source_consistency_mode") != mode_for_arm(arm)
    ):
        raise RuntimeError(f"crossover arm aggregate was not final on first publish: {arm}")
    return path, value


def _install_extension() -> None:
    core.RUN_ID = RUN_ID
    core.RUNNER_SELF = RUNNER_SELF
    core.CAPTURE_YAML = CAPTURE_YAML
    core.ANALYZER = ANALYZER
    core.ANALYZE_YAML = ANALYZE_YAML
    core.ARMS = ARMS
    core.GATE_FILES = (RUNNER_SELF, CAPTURE_YAML, ANALYZER, ANALYZE_YAML)
    core.PRECHECK_FILES = core.GATE_FILES + (
        CORE_RUNNER,
        CORE_ANALYZER,
        core.DATASET_SOURCE,
        core.PLANNER_SOURCE,
        core.SIDECAR_RUNTIME_SOURCE,
        core.COMPLETE_DATASET_SOURCE,
        core.DATALINE_CONFIG_SOURCE,
        core.IMAGE_FETCH_SOURCE,
        core.MANIFEST_INDEX_SOURCE,
        core.MULTI_AR_SOURCE,
        core.NATIVE_SAFETY_SOURCE,
        core.PIL_SAFETY_SOURCE,
        core.CAPTION_BUILD_SOURCE,
        core.SAMPLER_SOURCE,
        core.SIDECAR_CONTRACT_SOURCE,
        core.SIDECAR_VERIFY_SOURCE,
        core.CONFIG,
    )
    core.arm_nonce = arm_nonce
    core.write_rank_sentinel = write_rank_sentinel
    core._dataset_kwargs = _dataset_kwargs
    core._run_raw_arm = _run_raw_arm
    core._live_artifacts = _live_artifacts
    core._create_preflight = _create_preflight
    core._validate_preflight = _validate_preflight
    core._arm_contract = _arm_contract
    core._validate_arm_contract = _validate_arm_contract
    core._preserve_node_arm = _preserve_node_arm
    core._create_arm_aggregate = _create_arm_aggregate
    core.atomic_write_json = _publication_atomic_write_json


def _finalize_master() -> None:
    host, jobtag, node_rank = core._host_identity()
    del host
    if node_rank != 0:
        return
    path = core.BASE / RUN_ID / jobtag / f"abba.{jobtag}.aggregate.json"
    value = core.read_object(path, "crossover master aggregate")
    if value.get("status") != "PASS" or value.get("run_id") != RUN_ID:
        raise RuntimeError("crossover master is not a bound PASS")
    if (
        value.get("schema_version")
        != "offline_arb_w16_gpic_fd_crossover_master_aggregate_v1"
        or value.get("gpic_source_consistency_mode_by_arm") != MODE_BY_ARM
        or value.get("semantic_external_gate") != "PENDING"
        or value.get("production_selection_qualified") is not False
    ):
        raise RuntimeError("crossover master was not final on first publish")
    print(
        f"W16_GPIC_FD_CROSSOVER_CAPTURE_PASS result={path} sha256={core.sha256_file(path)}",
        flush=True,
    )


def main() -> int:
    _install_extension()
    if os.environ.get("OFFLINE_ARB_GPIC_THREADS_PER_NODE") != "16":
        raise RuntimeError(
            "crossover process requires OFFLINE_ARB_GPIC_THREADS_PER_NODE=16"
        )
    rc = core.main()
    if rc == 0 and len(sys.argv) >= 2 and sys.argv[1] == "node-wrapper":
        _finalize_master()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
