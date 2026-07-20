#!/usr/bin/env python3
"""Capture a balanced W16 L16/L8/L8/L16 raw-supply selector on two CPU nodes.

The same immutable-attempt file serves two roles.  JCC invokes ``node-wrapper``
once on each node; the wrapper performs the shared live-SHA/sidecar preflight and starts
four fresh 8-rank gloo torchrun processes.  Torchrun invokes ``rank-worker`` to
consume the exact plan slice through the loader's raw publication boundary.
No image decode, resize, tensor conversion, model, optimizer, or checkpoint is
allowed in this diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import torch.distributed as dist
from omegaconf import OmegaConf


REPO = Path("/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop")
BASE = Path("/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop")
RUN_ID = "stage256_offline_exact_w16_gpic_abba_raw_20260712r4"
PLAN = BASE / "offline_arb_plans_v1/w16_gpuval/plan"
CATALOG = BASE / "offline_arb_catalog_v1"
ANIME_FREQ = BASE / "offline_arb_animefreq_v1"
SIDECAR = BASE / "offline_arb_caption_sidecar_v1"
CONFIG = REPO / "configs/stage256_offline_exact_v1.yaml"
DATASET_SOURCE = REPO / "training/dataline_offline/offline_dataset.py"
PLANNER_SOURCE = REPO / "training/dataline_offline/planner_v1.py"
SIDECAR_RUNTIME_SOURCE = REPO / "training/dataline_offline/caption_sidecar.py"
COMPLETE_DATASET_SOURCE = REPO / "training/dataline/complete_dataset.py"
DATALINE_CONFIG_SOURCE = REPO / "training/dataline/config.py"
IMAGE_FETCH_SOURCE = REPO / "training/dataline/image_fetch.py"
MANIFEST_INDEX_SOURCE = REPO / "training/dataline/manifest_index.py"
MULTI_AR_SOURCE = REPO / "training/dataline/multi_ar.py"
NATIVE_SAFETY_SOURCE = REPO / "training/dataline/native_safety.py"
PIL_SAFETY_SOURCE = REPO / "training/dataline/pil_safety.py"
CAPTION_BUILD_SOURCE = REPO / "training/dataline/caption_build.py"
SAMPLER_SOURCE = REPO / "training/dataline/sampler.py"
SIDECAR_CONTRACT_SOURCE = REPO / "scripts/offline_arb/caption_sidecar_contract.py"
SIDECAR_VERIFY_SOURCE = REPO / "scripts/offline_arb/make_freeze_manifest.py"
SAMPLING = Path("/fsx/evan/mot_code/data_analysis/configs/t5_sampling_redesign_v1.json")
RUNNER_SELF = REPO / "scripts/offline_arb/run_w16_gpic_abba_raw_gate.py"
CAPTURE_YAML = REPO / "jobs/offline_arb_w16_gpic_abba_raw_gate.yaml"
ANALYZER = REPO / "scripts/offline_arb/analyze_w16_gpic_abba_raw_gate.py"
ANALYZE_YAML = REPO / "jobs/offline_arb_w16_gpic_abba_analyze.yaml"
PYTHON = Path("/fsx/evan/mot_code/venv-gpu/bin/python")

WORLD_SIZE = 16
LOCAL_WORLD_SIZE = 8
NODES = 2
PER_RANK_BS = 36
PREFETCH_STEPS = 4
SLICE_STEPS = 192
ARMS = (
    ("l16_a", 16, 0, 192),
    ("l8_a", 8, 192, 384),
    ("l8_b", 8, 384, 576),
    ("l16_b", 16, 576, 768),
)
GATE_FILES = (
    RUNNER_SELF,
    CAPTURE_YAML,
    ANALYZER,
    ANALYZE_YAML,
)
PRECHECK_FILES = GATE_FILES + (
    DATASET_SOURCE,
    PLANNER_SOURCE,
    SIDECAR_RUNTIME_SOURCE,
    COMPLETE_DATASET_SOURCE,
    DATALINE_CONFIG_SOURCE,
    IMAGE_FETCH_SOURCE,
    MANIFEST_INDEX_SOURCE,
    MULTI_AR_SOURCE,
    NATIVE_SAFETY_SOURCE,
    PIL_SAFETY_SOURCE,
    CAPTION_BUILD_SOURCE,
    SAMPLER_SOURCE,
    SIDECAR_CONTRACT_SOURCE,
    SIDECAR_VERIFY_SOURCE,
    CONFIG,
)
SHA_RE = re.compile(r"[0-9a-f]{64}")
SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9_.-]+")
REQUIRED_RAW_API = (
    "_start_pipeline",
    "_schedule_window",
    "_plan_items_for_step",
    "_wait_for_step",
    "_raw_for_item",
    "_retire_step",
    "_refresh_metrics",
    "_emit_state_record",
    "close",
)
RANK_SENTINEL_PREFIXES = (
    "W16_ABBA_RAW_STEP_RETIRED ",
    "W16_ABBA_RAW_RANK_PASS ",
)
FRESH_MARKER_MAX_AGE_S = 1800
FRESH_MARKER_MAX_FUTURE_S = 30


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


def attempt_nonce(jobtag: str) -> str:
    return canonical_sha256(
        {
            "schema_version": "offline_arb_w16_gpic_abba_attempt_nonce_v1",
            "run_id": RUN_ID,
            "jobtag": jobtag,
        }
    )


def arm_nonce(
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


def require_file(path: Path, label: str, *, nonempty: bool = True) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} missing/non-regular/symlink: {path}")
    if nonempty and path.stat().st_size <= 0:
        raise RuntimeError(f"{label} is empty: {path}")
    return path


def resolve_python_interpreter(path: Path) -> Path:
    """Resolve the venv entry while keeping every other artifact symlink-forbidden."""
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"python interpreter cannot be strictly resolved: {path}") from exc
    metadata = resolved.lstat()
    if resolved.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"python interpreter target is not a real regular file: {resolved}")
    if metadata.st_size <= 0 or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"python interpreter target is empty/non-executable: {resolved}")
    return resolved


def python_interpreter_artifact(path: Path) -> dict[str, Any]:
    resolved = resolve_python_interpreter(path)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_rank_sentinel(line: str) -> None:
    """Emit one rank evidence line with one checked PIPE_BUF-bounded write."""
    if (
        not isinstance(line, str)
        or not line
        or "\n" in line
        or "\r" in line
        or not line.startswith(RANK_SENTINEL_PREFIXES)
    ):
        raise RuntimeError("rank sentinel is malformed")
    payload = (line + "\n").encode("utf-8", errors="strict")
    descriptor = sys.stdout.fileno()
    if not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
        raise RuntimeError("rank sentinel stdout is not a pipe")
    try:
        pipe_buf = int(os.fpathconf(descriptor, "PC_PIPE_BUF"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot determine rank sentinel PIPE_BUF") from exc
    if pipe_buf <= 0 or len(payload) > pipe_buf:
        raise RuntimeError(
            f"rank sentinel exceeds PIPE_BUF: bytes={len(payload)} pipe_buf={pipe_buf}"
        )
    try:
        written = os.write(descriptor, payload)
    except OSError as exc:
        raise RuntimeError("atomic rank sentinel write failed") from exc
    if written != len(payload):
        raise RuntimeError(
            f"partial atomic rank sentinel write: wrote={written} expected={len(payload)}"
        )


def validate_fresh_marker_declarations(
    declarations: Any,
    marker_paths: list[Path],
    expected_identities: list[tuple[str, ...]],
    *,
    contract_created_at: Any,
    label: str,
) -> None:
    if (
        not isinstance(contract_created_at, int)
        or isinstance(contract_created_at, bool)
        or contract_created_at <= 0
        or not isinstance(declarations, list)
        or len(declarations) != len(marker_paths)
        or len(expected_identities) != len(marker_paths)
    ):
        raise RuntimeError(f"{label} fresh-marker contract malformed")
    for node_rank, (path, expected_identity) in enumerate(
        zip(marker_paths, expected_identities)
    ):
        require_file(path, f"{label} node {node_rank} fresh marker")
        declaration = declarations[node_rank]
        if (
            not isinstance(declaration, dict)
            or set(declaration) != {"path", "sha256"}
            or declaration.get("path") != str(path)
            or declaration.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"{label} node {node_rank} fresh-marker SHA/path mismatch")
        try:
            text = path.read_bytes().decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"{label} node {node_rank} fresh marker is not UTF-8") from exc
        if not text.endswith("\n") or text.count("\n") != 1 or "\r" in text:
            raise RuntimeError(f"{label} node {node_rank} fresh-marker framing mismatch")
        fields = text[:-1].split("|")
        if (
            len(fields) != len(expected_identity) + 2
            or tuple(fields[: len(expected_identity)]) != expected_identity
            or fields[-1] != f"{expected_identity[1]}-{node_rank}"
            or not fields[-2].isdigit()
        ):
            raise RuntimeError(f"{label} node {node_rank} fresh-marker identity mismatch")
        marker_created_at = int(fields[-2])
        age = contract_created_at - marker_created_at
        if age < -FRESH_MARKER_MAX_FUTURE_S or age > FRESH_MARKER_MAX_AGE_S:
            raise RuntimeError(
                f"{label} node {node_rank} fresh-marker content timestamp is stale: age={age}"
            )


def require_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} missing/non-directory/symlink: {path}")
    return path


def read_object(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite atomic artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"stale atomic temporary exists: {temporary}")
    try:
        with temporary.open("xb", buffering=0) as handle:
            handle.write(payload)
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, payload)


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_s: float,
    label: str,
    interval_s: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval_s)
    raise TimeoutError(f"timed out waiting for {label}")


def artifact(path: Path, name: str | None = None) -> dict[str, Any]:
    require_file(path, f"artifact {path}")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"artifact is not a regular file: {path}")
    return {
        "name": path.name if name is None else name,
        "bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def validate_artifact(path: Path, declaration: Any, label: str) -> None:
    if not isinstance(declaration, dict) or set(declaration) != {"name", "bytes", "sha256"}:
        raise RuntimeError(f"{label} declaration malformed")
    require_file(path, label)
    if declaration["name"] != path.name or declaration["bytes"] != path.stat().st_size:
        raise RuntimeError(f"{label} name/size mismatch")
    if declaration["sha256"] != sha256_file(path):
        raise RuntimeError(f"{label} SHA mismatch")


def _strict_env_int(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.isdigit():
        raise RuntimeError(f"{name} must be an unsigned integer, got {raw!r}")
    return int(raw)


def _dataset_kwargs(args: argparse.Namespace, rank: int) -> dict[str, Any]:
    cfg = OmegaConf.load(require_file(Path(args.config), "config"))
    value = OmegaConf.to_container(cfg.dataset, resolve=True)
    if not isinstance(value, dict):
        raise RuntimeError("config.dataset must resolve to a mapping")
    if Path(str(value.get("sampling_config", ""))).resolve(strict=True) != SAMPLING:
        raise RuntimeError("config sampling path differs from ABBA-bound sampling JSON")
    kwargs = dict(value)
    kwargs.pop("store_cls", None)
    multi_ar = dict(kwargs.get("multi_ar") or {})
    multi_ar["assignment_mode"] = "offline_exact_v1"
    kwargs.update(
        {
            "offline_plan_root": str(require_directory(Path(args.plan_root), "plan root")),
            "offline_catalog_root": str(
                require_directory(Path(args.catalog_root), "catalog root")
            ),
            "offline_anime_freq_root": str(
                require_directory(Path(args.anime_freq_root), "anime-frequency root")
            ),
            "offline_caption_sidecar_root": str(
                require_directory(Path(args.sidecar_root), "caption-sidecar root")
            ),
            "rank": rank,
            "world_size": WORLD_SIZE,
            "stage": "256",
            "multi_ar": multi_ar,
            "tokenizer_path": None,
        }
    )
    return kwargs


def _planned_identity(item: Any) -> tuple[int, int, int, str, int, int]:
    return (
        int(item.step),
        int(item.slot),
        int(item.row_id),
        str(item.domain),
        int(item.domain_idx),
        int(item.bucket_idx),
    )


def _maintain_bounded_prefetch(dataset: Any, stop_step: int) -> None:
    while (
        int(dataset._scheduled_until) < stop_step
        and int(dataset._scheduled_until) - int(dataset._next_step)
        <= int(dataset.prefetch_steps)
    ):
        dataset._schedule_window(int(dataset._scheduled_until))
    if int(dataset._scheduled_until) > stop_step:
        raise AssertionError(
            f"prefetch escaped arm: scheduled_until={dataset._scheduled_until}, stop={stop_step}"
        )


def _run_raw_arm(dataset: Any, args: argparse.Namespace, rank: int) -> dict[str, Any]:
    missing = [name for name in REQUIRED_RAW_API if not callable(getattr(dataset, name, None))]
    if missing:
        raise RuntimeError(f"offline raw-supply API drifted; missing={missing}")
    if int(dataset.plan_header["per_rank_bs"]) != PER_RANK_BS:
        raise AssertionError("plan per-rank batch size drift")
    if int(dataset.plan_header["total_steps"]) < args.stop_step:
        raise AssertionError("plan does not cover requested ABBA arm")
    if int(dataset.prefetch_steps) != PREFETCH_STEPS:
        raise AssertionError("prefetch depth drift")
    if int(dataset._gpic_threads_per_node) != args.gpic_threads_per_node:
        raise AssertionError("GPIC node thread identity drift")
    expected_workers = args.gpic_threads_per_node // LOCAL_WORLD_SIZE
    if int(dataset._gpic_workers) != expected_workers:
        raise AssertionError("GPIC per-rank worker identity drift")
    if dataset._gpic_payload_in_memory is not True:
        raise AssertionError("ABBA raw gate requires gpic_payload_in_memory=true")

    original_enqueue = dataset._enqueue_telemetry
    emitted_window_keys: set[tuple[int, str]] = set()
    duplicate_window_keys: set[tuple[int, str]] = set()
    window_lock = threading.Lock()

    def observed_enqueue(record: Any) -> bool:
        accepted = original_enqueue(record)
        if accepted and isinstance(record, dict) and record.get("record_type") == "window":
            key = (int(record.get("start", -1)), str(record.get("domain")))
            with window_lock:
                if key in emitted_window_keys:
                    duplicate_window_keys.add(key)
                emitted_window_keys.add(key)
        return accepted

    dataset._enqueue_telemetry = observed_enqueue
    dataset.set_start_step(args.start_step)
    dist.barrier()
    dataset._start_pipeline()
    if int(dataset._scheduled_until) != min(args.stop_step, args.start_step + 8):
        raise AssertionError("initial bounded prefetch drift")

    bytes_by_domain: Counter[str] = Counter()
    items_by_domain: Counter[str] = Counter()
    total_raw_bytes = 0
    total_items = 0
    total_wait_s = 0.0
    max_wait_s = 0.0
    expected_window_keys: set[tuple[int, str]] = set()
    started = time.perf_counter()
    for relative_step, step in enumerate(range(args.start_step, args.stop_step)):
        expected_bucket, expected_items = dataset._plan_items_for_step(step)
        window_start = args.start_step + (
            (step - args.start_step) // PREFETCH_STEPS
        ) * PREFETCH_STEPS
        expected_window_keys.update(
            (window_start, str(item.domain)) for item in expected_items
        )
        if len(expected_items) != PER_RANK_BS or dataset._step_bucket.get(step) != expected_bucket:
            raise AssertionError(f"rank {rank} step {step}: scheduled plan identity mismatch")
        dist.barrier()
        wait_started = time.perf_counter()
        prepared = dataset._wait_for_step(step)
        wait_s = time.perf_counter() - wait_started
        if len(prepared) != PER_RANK_BS:
            raise AssertionError(f"rank {rank} step {step}: prepared item count mismatch")
        step_raw_bytes = 0
        for slot, (actual, expected) in enumerate(zip(prepared, expected_items)):
            if _planned_identity(actual.planned) != _planned_identity(expected):
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
                raise AssertionError("ABBA arm left the formal in-memory raw path")
            raw = dataset._raw_for_item(actual)
            if not isinstance(raw, bytes) or not raw:
                raise AssertionError("ABBA raw payload is empty/non-bytes")
            size = len(raw)
            step_raw_bytes += size
            total_raw_bytes += size
            total_items += 1
            items_by_domain[str(expected.domain)] += 1
            bytes_by_domain[str(expected.domain)] += size
            del raw
        dataset._retire_step(step, prepared)
        if step in dataset._ready or step in dataset._expected or step in dataset._step_bucket:
            raise AssertionError(f"rank {rank} step {step}: retire left live state")
        dataset._next_step = step + 1
        if dataset._next_step < args.stop_step:
            _maintain_bounded_prefetch(dataset, args.stop_step)
        dataset._refresh_metrics(plan_step=step)
        dataset._emit_state_record(step)
        total_wait_s += wait_s
        max_wait_s = max(max_wait_s, wait_s)
        write_rank_sentinel(
            "W16_ABBA_RAW_STEP_RETIRED "
            f"arm={args.arm} rank={rank} step={step} relative_step={relative_step} "
            f"items={len(prepared)} raw_bytes={step_raw_bytes} wait_s={wait_s:.9f} "
            f"scheduled_until={int(dataset._scheduled_until)}"
        )
    if int(dataset._next_step) != args.stop_step or int(dataset._scheduled_until) != args.stop_step:
        raise AssertionError("ABBA arm final step/schedule boundary mismatch")
    telemetry_deadline = time.monotonic() + 60.0
    while True:
        with window_lock:
            observed = set(emitted_window_keys)
            duplicates = set(duplicate_window_keys)
        if duplicates:
            raise AssertionError(f"duplicate telemetry windows emitted: {sorted(duplicates)}")
        if not observed.issubset(expected_window_keys):
            raise AssertionError(
                f"unexpected telemetry windows emitted: {sorted(observed - expected_window_keys)}"
            )
        if observed == expected_window_keys:
            break
        if time.monotonic() >= telemetry_deadline:
            raise TimeoutError(
                "timed out waiting for exact terminal telemetry windows: "
                f"missing={sorted(expected_window_keys - observed)}"
            )
        time.sleep(0.01)
    dataset._enqueue_telemetry = original_enqueue
    expected_items_total = SLICE_STEPS * PER_RANK_BS
    if total_items != expected_items_total:
        raise AssertionError("ABBA raw-item total mismatch")
    return {
        "rank": rank,
        "raw_items": total_items,
        "raw_bytes": total_raw_bytes,
        "items_by_domain": dict(sorted(items_by_domain.items())),
        "bytes_by_domain": dict(sorted(bytes_by_domain.items())),
        "wait_total_s": total_wait_s,
        "wait_max_s": max_wait_s,
        "elapsed_s": time.perf_counter() - started,
    }


def _assert_closed_evidence(dataset: Any, rank: int) -> None:
    if dataset._telemetry_enabled is not True or dataset._telemetry_closed is not True:
        raise AssertionError(f"rank {rank}: dataset telemetry did not close")
    writer = dataset._telemetry_thread
    if writer is None or writer.is_alive() or dataset._telemetry_writer_error is not None:
        raise AssertionError(f"rank {rank}: telemetry writer close failed")
    if (
        int(dataset._telemetry_dropped_count) != 0
        or int(dataset._telemetry_runtime_error_count) != 0
        or int(dataset._telemetry_rejected_count) != 0
        or dataset._threads
    ):
        raise AssertionError(f"rank {rank}: supply/telemetry close is not clean")


def _assert_rank_runtime_inputs(preflight: dict[str, Any]) -> None:
    _assert_artifacts_unchanged(preflight["artifacts"])
    _assert_bound_python_interpreter(preflight["artifacts"])
    _assert_animefreq_unchanged(preflight["animefreq_ledger"])
    _validate_sidecar_ledger(preflight["sidecar_baseline"])


def rank_worker(args: argparse.Namespace) -> int:
    preflight_path = Path(args.preflight)
    require_file(preflight_path, "rank preflight contract")
    if sha256_file(preflight_path) != args.expected_preflight_sha256:
        raise RuntimeError("rank preflight differs from wrapper-bound SHA")
    preflight = read_object(preflight_path, "rank preflight contract")
    attempt_jobtag = os.environ.get("OFFLINE_ARB_ATTEMPT_JOBTAG", "")
    expected_preflight_path = BASE / RUN_ID / attempt_jobtag / (
        f"preflight.{attempt_jobtag}.contract.json"
    )
    if preflight_path != expected_preflight_path:
        raise RuntimeError("rank preflight path is not canonical for attempt")
    _validate_preflight(preflight, attempt_jobtag)
    _assert_rank_runtime_inputs(preflight)
    for path, expected, label in (
        (RUNNER_SELF, args.expected_runner_sha256, "runner"),
        (DATASET_SOURCE, args.expected_dataset_sha256, "dataset source"),
        (PLANNER_SOURCE, args.expected_planner_sha256, "planner source"),
        (
            SIDECAR_RUNTIME_SOURCE,
            args.expected_sidecar_runtime_sha256,
            "sidecar runtime source",
        ),
        (Path(args.config), args.expected_config_sha256, "config"),
        (Path(args.plan_root) / "header.json", args.expected_plan_header_sha256, "plan header"),
        (
            Path(args.plan_root) / "checksums.json",
            args.expected_plan_checksums_sha256,
            "plan checksums",
        ),
        (
            Path(args.catalog_root) / "header.json",
            args.expected_catalog_header_sha256,
            "catalog header",
        ),
        (
            Path(args.anime_freq_root) / "header.json",
            args.expected_animefreq_header_sha256,
            "anime-frequency header",
        ),
        (
            Path(args.sidecar_root) / "header.json",
            args.expected_sidecar_header_file_sha256,
            "sidecar header file",
        ),
    ):
        require_file(path, label)
        if sha256_file(path) != expected:
            raise RuntimeError(f"{label} differs from wrapper-bound SHA")
    if dist.is_initialized():
        raise RuntimeError("process group initialized before ABBA rank worker")
    for supplied, canonical, label in (
        (Path(args.config), CONFIG, "config"),
        (Path(args.plan_root), PLAN, "plan"),
        (Path(args.catalog_root), CATALOG, "catalog"),
        (Path(args.anime_freq_root), ANIME_FREQ, "anime-frequency"),
        (Path(args.sidecar_root), SIDECAR, "sidecar"),
    ):
        if supplied.resolve(strict=True) != canonical.resolve(strict=True):
            raise RuntimeError(f"{label} path differs from wrapper-bound canonical path")
    dist.init_process_group(
        backend="gloo", timeout=timedelta(seconds=args.process_group_timeout_s)
    )
    dataset = None
    rank = dist.get_rank()
    try:
        local_rank = _strict_env_int("LOCAL_RANK")
        local_world = _strict_env_int("LOCAL_WORLD_SIZE")
        node_rank = _strict_env_int("OFFLINE_ARB_NODE_RANK")
        if (
            dist.get_world_size() != WORLD_SIZE
            or dist.get_backend() != "gloo"
            or local_world != LOCAL_WORLD_SIZE
            or rank != node_rank * LOCAL_WORLD_SIZE + local_rank
        ):
            raise RuntimeError("ABBA rank topology/backend mapping drift")
        import training.dataline_offline.offline_dataset as offline_module
        import training.dataline_offline.caption_sidecar as sidecar_runtime_module
        import training.dataline_offline.planner_v1 as planner_module
        import training.dataline.complete_dataset as complete_dataset_module
        import training.dataline.config as dataline_config_module
        import training.dataline.image_fetch as image_fetch_module
        import training.dataline.manifest_index as manifest_index_module
        import training.dataline.multi_ar as multi_ar_module
        import training.dataline.native_safety as native_safety_module
        import training.dataline.pil_safety as pil_safety_module
        import training.dataline.caption_build as caption_build_module
        import training.dataline.sampler as sampler_module

        module_path = Path(offline_module.__file__).resolve(strict=True)
        if module_path != DATASET_SOURCE or sha256_file(module_path) != args.expected_dataset_sha256:
            raise RuntimeError("imported dataset module differs from wrapper-bound source")
        for module, path, expected, label in (
            (planner_module, PLANNER_SOURCE, args.expected_planner_sha256, "planner"),
            (
                sidecar_runtime_module,
                SIDECAR_RUNTIME_SOURCE,
                args.expected_sidecar_runtime_sha256,
                "sidecar runtime",
            ),
            (
                complete_dataset_module,
                COMPLETE_DATASET_SOURCE,
                preflight["artifacts"]["complete_dataset_source"]["sha256"],
                "complete dataset",
            ),
            (
                dataline_config_module,
                DATALINE_CONFIG_SOURCE,
                preflight["artifacts"]["dataline_config_source"]["sha256"],
                "dataline config",
            ),
            (
                image_fetch_module,
                IMAGE_FETCH_SOURCE,
                preflight["artifacts"]["image_fetch_source"]["sha256"],
                "image fetch",
            ),
            (
                manifest_index_module,
                MANIFEST_INDEX_SOURCE,
                preflight["artifacts"]["manifest_index_source"]["sha256"],
                "manifest index",
            ),
            (
                multi_ar_module,
                MULTI_AR_SOURCE,
                preflight["artifacts"]["multi_ar_source"]["sha256"],
                "multi-AR",
            ),
            (
                native_safety_module,
                NATIVE_SAFETY_SOURCE,
                preflight["artifacts"]["native_safety_source"]["sha256"],
                "native safety",
            ),
            (
                pil_safety_module,
                PIL_SAFETY_SOURCE,
                preflight["artifacts"]["pil_safety_source"]["sha256"],
                "PIL safety",
            ),
            (
                caption_build_module,
                CAPTION_BUILD_SOURCE,
                preflight["artifacts"]["caption_build_source"]["sha256"],
                "caption build",
            ),
            (
                sampler_module,
                SAMPLER_SOURCE,
                preflight["artifacts"]["sampler_source"]["sha256"],
                "sampler",
            ),
        ):
            module_path = Path(module.__file__).resolve(strict=True)
            if module_path != path or sha256_file(module_path) != expected:
                raise RuntimeError(f"imported {label} module differs from wrapper-bound source")
        _assert_rank_runtime_inputs(preflight)

        def decode_is_forbidden(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("decode/resize/tensor conversion is forbidden in ABBA raw gate")

        offline_module.decode_rgb = decode_is_forbidden
        offline_module.resize_u8_to_bucket = decode_is_forbidden
        offline_module.uint8_to_norm_tensor = decode_is_forbidden
        dataset = offline_module.OfflineExactDataset(**_dataset_kwargs(args, rank))
        result = _run_raw_arm(dataset, args, rank)
        _assert_rank_runtime_inputs(preflight)
        dataset.close()
        _assert_closed_evidence(dataset, rank)
        dataset = None
        dist.barrier()
        write_rank_sentinel(
            "W16_ABBA_RAW_RANK_PASS "
            f"arm={args.arm} rank={rank} start_step={args.start_step} "
            f"stop_step={args.stop_step} retired_steps={SLICE_STEPS} "
            f"per_rank_batch_size={PER_RANK_BS} raw_items={result['raw_items']} "
            f"raw_bytes={result['raw_bytes']} wait_total_s={result['wait_total_s']:.9f} "
            f"wait_max_s={result['wait_max_s']:.9f} "
            f"elapsed_s={result['elapsed_s']:.9f} decode_calls=0"
        )
        return 0
    finally:
        try:
            if dataset is not None:
                dataset.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def _host_identity() -> tuple[str, str, int]:
    host = socket.gethostname()
    if "-" not in host:
        raise RuntimeError(f"cannot derive JCC node rank from hostname: {host}")
    jobtag, rank_raw = host.rsplit("-", 1)
    if SAFE_COMPONENT_RE.fullmatch(jobtag) is None or rank_raw not in {"0", "1"}:
        raise RuntimeError(f"malformed JCC hostname: {host}")
    node_rank = int(rank_raw)
    if host != f"{jobtag}-{node_rank}":
        raise RuntimeError("hostname/jobtag/node mapping drift")
    return host, jobtag, node_rank


def _live_artifacts() -> dict[str, dict[str, Any]]:
    paths = {
        "runner": RUNNER_SELF,
        "capture_yaml": CAPTURE_YAML,
        "analyzer": ANALYZER,
        "analyze_yaml": ANALYZE_YAML,
        "dataset_source": DATASET_SOURCE,
        "planner_source": PLANNER_SOURCE,
        "sidecar_runtime_source": SIDECAR_RUNTIME_SOURCE,
        "complete_dataset_source": COMPLETE_DATASET_SOURCE,
        "dataline_config_source": DATALINE_CONFIG_SOURCE,
        "image_fetch_source": IMAGE_FETCH_SOURCE,
        "manifest_index_source": MANIFEST_INDEX_SOURCE,
        "multi_ar_source": MULTI_AR_SOURCE,
        "native_safety_source": NATIVE_SAFETY_SOURCE,
        "pil_safety_source": PIL_SAFETY_SOURCE,
        "caption_build_source": CAPTION_BUILD_SOURCE,
        "sampler_source": SAMPLER_SOURCE,
        "sidecar_contract_source": SIDECAR_CONTRACT_SOURCE,
        "sidecar_verify_source": SIDECAR_VERIFY_SOURCE,
        "sampling": SAMPLING,
        "config": CONFIG,
        "plan_header": PLAN / "header.json",
        "plan_checksums": PLAN / "checksums.json",
        "catalog_header": CATALOG / "header.json",
        "animefreq_header": ANIME_FREQ / "header.json",
        "sidecar_header": SIDECAR / "header.json",
        "sidecar_success": SIDECAR / "_SUCCESS",
    }
    artifacts = {
        label: {
            "path": str(require_file(path, label).resolve(strict=True)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for label, path in paths.items()
    }
    artifacts["python_interpreter"] = python_interpreter_artifact(PYTHON)
    return artifacts


def _assert_artifacts_unchanged(expected: dict[str, Any]) -> None:
    live = _live_artifacts()
    if live != expected:
        raise RuntimeError("ABBA source/control artifacts changed during attempt")


def _assert_bound_python_interpreter(artifacts: dict[str, Any]) -> None:
    expected = artifacts.get("python_interpreter")
    live = python_interpreter_artifact(PYTHON)
    running = resolve_python_interpreter(Path(sys.executable))
    if expected != live or running != Path(live["path"]):
        raise RuntimeError("running Python interpreter differs from preflight-bound real target")


def _fingerprint(path: Path, *, directory: bool = False) -> dict[str, int]:
    if path.is_symlink() or (not path.is_dir() if directory else not path.is_file()):
        raise RuntimeError(f"artifact type/symlink drift: {path}")
    metadata = path.lstat()
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "size_bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
    }


def _build_animefreq_ledger() -> dict[str, Any]:
    root = require_directory(ANIME_FREQ, "anime-frequency root")
    header = read_object(root / "header.json", "anime-frequency header")
    declared = header.get("files")
    if (
        header.get("schema_version") != "offline_arb_animefreq_v1"
        or not isinstance(declared, dict)
        or len(declared) != 17
        or header.get("shard_count") != len(declared)
    ):
        raise RuntimeError("anime-frequency header/file-count contract drift")
    expected_names = {"header.json", "_SUCCESS", *declared}
    observed_names = {path.name for path in root.iterdir()}
    if observed_names != expected_names:
        raise RuntimeError("anime-frequency root exact-set drift")
    files: dict[str, Any] = {}
    for name in sorted(declared):
        if SAFE_COMPONENT_RE.fullmatch(name) is None or not name.endswith(".json.gz"):
            raise RuntimeError(f"anime-frequency filename malformed: {name}")
        path = root / name
        require_file(path, f"anime-frequency file {name}")
        digest = sha256_file(path)
        entry = declared[name]
        if not isinstance(entry, dict) or entry.get("sha256") != digest:
            raise RuntimeError(f"anime-frequency SHA/header mismatch: {name}")
        files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "fingerprint": _fingerprint(path),
        }
    success = root / "_SUCCESS"
    if success.is_symlink() or not success.is_file():
        raise RuntimeError("anime-frequency _SUCCESS missing/non-regular/symlink")
    return {
        "schema_version": "offline_arb_w16_animefreq_ledger_v1",
        "root": str(root),
        "root_fingerprint": _fingerprint(root, directory=True),
        "header_sha256": sha256_file(root / "header.json"),
        "success": {
            "path": str(success),
            "bytes": success.stat().st_size,
            "sha256": sha256_file(success),
            "fingerprint": _fingerprint(success),
        },
        "files": files,
    }


def _assert_animefreq_unchanged(expected: dict[str, Any]) -> None:
    if _build_animefreq_ledger() != expected:
        raise RuntimeError("anime-frequency exact ledger changed during attempt")


def _full_sidecar_snapshot() -> dict[str, Any]:
    from scripts.offline_arb.make_freeze_manifest import validate_sidecar

    value = validate_sidecar(SIDECAR, CATALOG, full=True)
    if not isinstance(value, dict):
        raise RuntimeError("full sidecar verifier returned a non-object")
    return {"status": "PASS", "mode": "verify-sidecar", **value}


def _sidecar_baseline(full: dict[str, Any]) -> dict[str, Any]:
    files = full.get("files")
    directories = full.get("directories")
    if (
        full.get("status") != "PASS"
        or full.get("mode") != "verify-sidecar"
        or full.get("root") != str(SIDECAR)
        or not isinstance(files, dict)
        or not isinstance(directories, dict)
        or len(files) != full.get("file_count")
        or sum(int(entry.get("bytes", -1)) for entry in files.values())
        != full.get("total_bytes")
    ):
        raise RuntimeError("full sidecar ledger summary malformed")
    baseline = {
        "schema_version": "offline_arb_w16_sidecar_baseline_v1",
        "root": str(SIDECAR),
        "directories": directories,
        "files": files,
        "file_count": full["file_count"],
        "total_bytes": full["total_bytes"],
        "header_sha256": full["header_sha256"],
        "release_digest": full["release_digest"],
    }
    _validate_sidecar_ledger(baseline)
    return baseline


def _validate_sidecar_ledger(baseline: dict[str, Any]) -> dict[str, Any]:
    root = require_directory(SIDECAR, "sidecar ledger root")
    files = baseline.get("files")
    directories = baseline.get("directories")
    if not isinstance(files, dict) or not isinstance(directories, dict):
        raise RuntimeError("sidecar baseline ledger malformed")
    domain_names = set(directories) - {"root"}
    expected_root = {"header.json", "_SUCCESS", *domain_names}
    if {path.name for path in root.iterdir()} != expected_root:
        raise RuntimeError("sidecar root exact-set changed")
    live_directories: dict[str, Any] = {}
    for label, declaration in directories.items():
        path = root if label == "root" else root / label
        if not isinstance(declaration, dict) or declaration.get("path") != str(path):
            raise RuntimeError(f"sidecar directory declaration malformed: {label}")
        fingerprint = _fingerprint(path, directory=True)
        if declaration.get("fingerprint") != fingerprint:
            raise RuntimeError(f"sidecar directory fingerprint changed: {label}")
        live_directories[label] = {"path": str(path), "fingerprint": fingerprint}
    expected_by_domain: dict[str, set[str]] = {domain: set() for domain in domain_names}
    live_files: dict[str, Any] = {}
    for name, declaration in sorted(files.items()):
        if not isinstance(name, str) or not isinstance(declaration, dict):
            raise RuntimeError("sidecar file ledger entry malformed")
        path = SIDECAR / name
        if declaration.get("path") != str(path) or path.parent.name not in expected_by_domain:
            raise RuntimeError(f"sidecar file canonical path drift: {name}")
        require_file(path, f"sidecar ledger file {name}")
        fingerprint = _fingerprint(path)
        if (
            declaration.get("bytes") != path.stat().st_size
            or declaration.get("fingerprint") != fingerprint
            or SHA_RE.fullmatch(str(declaration.get("sha256") or "")) is None
        ):
            raise RuntimeError(f"sidecar file size/fingerprint/SHA declaration drift: {name}")
        expected_by_domain[path.parent.name].add(path.name)
        live_files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": declaration["sha256"],
            "fingerprint": fingerprint,
        }
    for domain, expected_names in expected_by_domain.items():
        domain_root = root / domain
        if {path.name for path in domain_root.iterdir()} != expected_names:
            raise RuntimeError(f"sidecar domain exact-set changed: {domain}")
    if len(live_files) != baseline.get("file_count") or sum(
        entry["bytes"] for entry in live_files.values()
    ) != baseline.get("total_bytes"):
        raise RuntimeError("sidecar live ledger count/bytes do not close")
    return {
        "directories": live_directories,
        "files": live_files,
        "file_count": len(live_files),
        "total_bytes": sum(entry["bytes"] for entry in live_files.values()),
    }


def _write_sidecar_ledger_evidence(
    *,
    result_root: Path,
    preflight: dict[str, Any],
    arm_contract: dict[str, Any],
    node_rank: int,
    phase: str,
) -> dict[str, Any]:
    if phase not in {"before", "after"}:
        raise RuntimeError(f"unsupported sidecar ledger phase: {phase}")
    live = _validate_sidecar_ledger(preflight["sidecar_baseline"])
    value = {
        "schema_version": "offline_arb_w16_sidecar_arm_ledger_v1",
        "status": "PASS",
        "jobtag": preflight["jobtag"],
        "attempt_nonce": preflight["attempt_nonce"],
        "arm": arm_contract["arm"],
        "arm_nonce": arm_contract["arm_nonce"],
        "node_rank": node_rank,
        "host": f"{preflight['jobtag']}-{node_rank}",
        "phase": phase,
        "sidecar_full_verify_sha256": preflight["sidecar_full_verify"]["sha256"],
        "baseline_sha256": canonical_sha256(preflight["sidecar_baseline"]),
        "live_ledger_sha256": canonical_sha256(live),
        "file_count": live["file_count"],
        "total_bytes": live["total_bytes"],
        "created_at_unix": int(time.time()),
    }
    path = result_root / (
        f"sidecarledger.{preflight['jobtag']}.{arm_contract['arm']}."
        f"n{node_rank}.{phase}.json"
    )
    atomic_write_json(path, value)
    return {"path": str(path), "sha256": sha256_file(path)}


def _master_address() -> str:
    output = subprocess.check_output(["hostname", "-i"], text=True).strip().split()
    if not output:
        raise RuntimeError("hostname -i returned no rendezvous address")
    return output[0]


def _create_preflight(result_root: Path, jobtag: str, markers: list[Path]) -> dict[str, Any]:
    initial_artifacts = _live_artifacts()
    animefreq_ledger = _build_animefreq_ledger()
    verify_path = result_root / f"sidecarfull.{jobtag}.json"
    atomic_write_json(verify_path, _full_sidecar_snapshot())
    verify = read_object(verify_path, "full sidecar preflight JSON")
    if verify.get("status") != "PASS" or verify.get("mode") != "verify-sidecar":
        raise RuntimeError("full sidecar preflight did not verify")
    _assert_artifacts_unchanged(initial_artifacts)

    from training.dataline_offline.planner_v1 import OfflinePlan

    plan = OfflinePlan(PLAN, world_size=WORLD_SIZE, per_rank_bs=PER_RANK_BS, verify_checksums=True)
    plan.verify_all_checksums()
    if int(plan.config.total_steps) < ARMS[-1][3]:
        raise RuntimeError("W16 plan does not cover all ABBA slices")
    header = read_object(PLAN / "header.json", "W16 plan header")
    checksums = read_object(PLAN / "checksums.json", "W16 plan checksums")
    if checksums.get("header") != sha256_file(PLAN / "header.json"):
        raise RuntimeError("W16 plan header/checksums binding mismatch")
    used_blocks = sorted(
        {
            step // int(plan.config.block_steps)
            for _arm, _threads, start, stop in ARMS
            for step in range(start, stop)
        }
    )
    block_artifacts: dict[str, Any] = {}
    for block_index in used_blocks:
        relative = f"block_{block_index:06d}.npz"
        path = PLAN / "blocks" / relative
        digest = sha256_file(require_file(path, f"W16 plan block {relative}"))
        if plan.checksums.get(relative) != digest:
            raise RuntimeError(f"W16 plan used-block checksum mismatch: {relative}")
        block_artifacts[relative] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    sidecar_header = read_object(SIDECAR / "header.json", "sidecar header")
    nonce = attempt_nonce(jobtag)
    contract = {
        "schema_version": "offline_arb_w16_gpic_abba_preflight_v1",
        "status": "PASS",
        "jobtag": jobtag,
        "attempt_nonce": nonce,
        "created_at_unix": int(time.time()),
        "master_addr": _master_address(),
        "sidecar_full_verify": {
            "path": str(verify_path),
            "sha256": sha256_file(verify_path),
        },
        "artifacts": initial_artifacts,
        "animefreq_ledger": animefreq_ledger,
        "plan": {
            "plan_id": header.get("plan_id"),
            "plan_hash": header.get("plan_hash"),
            "catalog_hash": header.get("catalog_hash"),
            "world_size": header.get("world_size"),
            "per_rank_bs": header.get("per_rank_bs"),
            "prefetch_steps": PREFETCH_STEPS,
            "used_blocks": block_artifacts,
        },
        "sidecar_header_sha256": sidecar_header.get("header_sha256"),
        "sidecar_release_digest": verify["release_digest"],
        "sidecar_snapshot_fingerprints_sha256": verify[
            "snapshot_fingerprints_sha256"
        ],
        "fresh_markers": [
            {"path": str(path), "sha256": sha256_file(path)} for path in markers
        ],
        "arms": [
            {"arm": arm, "threads": threads, "start_step": start, "stop_step": stop}
            for arm, threads, start, stop in ARMS
        ],
    }
    contract["sidecar_baseline"] = _sidecar_baseline(verify)
    for label in (
        "sidecar_header_sha256",
        "sidecar_release_digest",
        "sidecar_snapshot_fingerprints_sha256",
    ):
        if SHA_RE.fullmatch(str(contract[label] or "")) is None:
            raise RuntimeError(f"{label} identity malformed")
    if (
        verify.get("header_sha256") != contract["sidecar_header_sha256"]
        or verify.get("catalog_hash") != header.get("catalog_hash")
    ):
        raise RuntimeError("full sidecar/W16 plan identity mismatch")
    _assert_artifacts_unchanged(initial_artifacts)
    _assert_animefreq_unchanged(animefreq_ledger)
    _validate_sidecar_ledger(contract["sidecar_baseline"])
    return contract


def _validate_preflight(value: dict[str, Any], jobtag: str) -> None:
    if (
        value.get("schema_version") != "offline_arb_w16_gpic_abba_preflight_v1"
        or value.get("status") != "PASS"
        or value.get("jobtag") != jobtag
        or value.get("attempt_nonce") != attempt_nonce(jobtag)
        or value.get("arms")
        != [
            {"arm": arm, "threads": threads, "start_step": start, "stop_step": stop}
            for arm, threads, start, stop in ARMS
        ]
    ):
        raise RuntimeError("shared ABBA preflight contract malformed")
    if not isinstance(value.get("master_addr"), str) or not value["master_addr"]:
        raise RuntimeError("shared ABBA master address malformed")
    preflight_marker_paths = [
        BASE / RUN_ID / jobtag / f"livepreflight.{jobtag}.n{node_rank}.fresh"
        for node_rank in range(NODES)
    ]
    validate_fresh_marker_declarations(
        value.get("fresh_markers"),
        preflight_marker_paths,
        [
            ("READY", jobtag, value["attempt_nonce"], str(node_rank))
            for node_rank in range(NODES)
        ],
        contract_created_at=value.get("created_at_unix"),
        label="preflight",
    )
    for label in (
        "sidecar_header_sha256",
        "sidecar_release_digest",
        "sidecar_snapshot_fingerprints_sha256",
    ):
        if SHA_RE.fullmatch(str(value.get(label) or "")) is None:
            raise RuntimeError(f"shared ABBA {label} malformed")
    full_verify = value.get("sidecar_full_verify")
    if (
        not isinstance(full_verify, dict)
        or set(full_verify) != {"path", "sha256"}
        or full_verify.get("path")
        != str(BASE / RUN_ID / jobtag / f"sidecarfull.{jobtag}.json")
        or full_verify.get("sha256")
        != sha256_file(Path(str(full_verify.get("path", ""))))
    ):
        raise RuntimeError("shared ABBA full-sidecar evidence malformed")
    _assert_artifacts_unchanged(value["artifacts"])
    _assert_animefreq_unchanged(value["animefreq_ledger"])
    _validate_sidecar_ledger(value["sidecar_baseline"])


def _sidecar_control(
    result_root: Path, jobtag: str, arm: str, node_rank: int
) -> dict[str, Any]:
    path = result_root / f"sidecarcontrol.{jobtag}.{arm}.n{node_rank}.json"
    completed = subprocess.run(
        [
            str(PYTHON),
            str(SIDECAR_VERIFY_SOURCE),
            "--verify-sidecar-control",
            "--sidecar-root",
            str(SIDECAR),
            "--catalog-root",
            str(CATALOG),
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=3600,
    )
    atomic_write_text(path, completed.stdout)
    value = read_object(path, "sidecar control")
    if value.get("status") != "PASS" or value.get("mode") != "verify-sidecar-control":
        raise RuntimeError("sidecar control failed")
    return {"path": str(path), "sha256": sha256_file(path), "value": value}


def _arm_contract(
    result_root: Path,
    jobtag: str,
    preflight_path: Path,
    arm: str,
    threads: int,
    start: int,
    stop: int,
    markers: list[Path],
    attempt: str,
) -> dict[str, Any]:
    nonce = arm_nonce(attempt, arm, threads, start, stop)
    return {
        "schema_version": "offline_arb_w16_gpic_abba_arm_contract_v1",
        "status": "READY",
        "jobtag": jobtag,
        "arm": arm,
        "attempt_nonce": attempt,
        "arm_nonce": nonce,
        "threads": threads,
        "start_step": start,
        "stop_step": stop,
        "created_at_unix": int(time.time()),
        "preflight": {"path": str(preflight_path), "sha256": sha256_file(preflight_path)},
        "fresh_markers": [
            {"path": str(path), "sha256": sha256_file(path)} for path in markers
        ],
    }


def _validate_arm_contract(
    value: dict[str, Any],
    *,
    jobtag: str,
    arm: str,
    threads: int,
    start: int,
    stop: int,
    preflight_path: Path,
    attempt: str,
) -> None:
    nonce = arm_nonce(attempt, arm, threads, start, stop)
    expected = {
        "schema_version": "offline_arb_w16_gpic_abba_arm_contract_v1",
        "status": "READY",
        "jobtag": jobtag,
        "arm": arm,
        "attempt_nonce": attempt,
        "arm_nonce": nonce,
        "threads": threads,
        "start_step": start,
        "stop_step": stop,
    }
    if set(value) != set(expected) | {"created_at_unix", "preflight", "fresh_markers"}:
        raise RuntimeError(f"arm contract exact key set mismatch: {arm}")
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        raise RuntimeError(f"arm contract identity mismatch: {arm}")
    preflight = value.get("preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("path") != str(preflight_path)
        or preflight.get("sha256") != sha256_file(preflight_path)
    ):
        raise RuntimeError(f"arm/preflight binding mismatch: {arm}")
    marker_paths = [
        preflight_path.parent / f"arm_fresh.{jobtag}.{arm}.n{node_rank}"
        for node_rank in range(NODES)
    ]
    validate_fresh_marker_declarations(
        value.get("fresh_markers"),
        marker_paths,
        [
            ("READY", jobtag, arm, attempt, nonce, str(node_rank))
            for node_rank in range(NODES)
        ],
        contract_created_at=value.get("created_at_unix"),
        label=f"arm {arm}",
    )


def _stream_subprocess(command: list[str], env: dict[str, str], log_path: Path) -> int:
    with log_path.open("xb", buffering=0) as raw_log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
        )
        assert process.stdout is not None
        for block in iter(lambda: process.stdout.read(1 << 16), b""):
            raw_log.write(block)
            os.write(sys.stdout.fileno(), block)
        rc = process.wait()
        os.fsync(raw_log.fileno())
    return rc


def _copy_and_fsync(source: Path, destination: Path) -> None:
    require_file(source, f"copy source {source}")
    with source.open("rb", buffering=0) as reader, destination.open("xb", buffering=0) as writer:
        shutil.copyfileobj(reader, writer, length=1 << 20)
        os.fsync(writer.fileno())


def _preserve_node_arm(
    *,
    result_root: Path,
    evidence_root: Path,
    local_root: Path,
    runner_log: Path,
    preflight: dict[str, Any],
    arm_contract: dict[str, Any],
    host: str,
    node_rank: int,
    run_rc: int,
    sidecar_control: dict[str, Any],
    sidecar_ledger_before: dict[str, Any],
    sidecar_ledger_after: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    rank_start = node_rank * LOCAL_WORLD_SIZE
    expected_names = {
        f"io_profile.rank{rank:04d}.jsonl"
        for rank in range(rank_start, rank_start + LOCAL_WORLD_SIZE)
    }
    observed_entries = list(local_root.iterdir()) if local_root.is_dir() else []
    observed_names = {path.name for path in observed_entries}
    valid_files = [
        path
        for path in observed_entries
        if path.name in expected_names and path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    ]
    collect_ok = (
        observed_names == expected_names
        and len(valid_files) == LOCAL_WORLD_SIZE
        and runner_log.is_file()
        and not runner_log.is_symlink()
        and runner_log.stat().st_size > 0
    )
    files = [artifact(path) for path in sorted(valid_files)]
    manifest = {
        "schema_version": "offline_arb_w16_gpic_abba_node_manifest_v1",
        "status": "PASS" if run_rc == 0 and collect_ok else "FAIL",
        "error": "" if run_rc == 0 and collect_ok else "runner_or_exact_set_failed",
        "jobtag": preflight["jobtag"],
        "attempt_nonce": preflight["attempt_nonce"],
        "arm": arm_contract["arm"],
        "arm_nonce": arm_contract["arm_nonce"],
        "threads": arm_contract["threads"],
        "start_step": arm_contract["start_step"],
        "stop_step": arm_contract["stop_step"],
        "node_rank": node_rank,
        "host": host,
        "world_size": WORLD_SIZE,
        "local_world_size": LOCAL_WORLD_SIZE,
        "per_rank_bs": PER_RANK_BS,
        "prefetch_steps": PREFETCH_STEPS,
        "expected_global_ranks": list(range(rank_start, rank_start + LOCAL_WORLD_SIZE)),
        "run_rc": run_rc,
        "collect_rc": 0 if collect_ok else 46,
        "preflight_sha256": sha256_file(
            result_root / f"preflight.{preflight['jobtag']}.contract.json"
        ),
        "arm_contract_sha256": sha256_file(
            result_root / f"arm_contract.{preflight['jobtag']}.{arm_contract['arm']}.json"
        ),
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
        "sidecar_control": {
            "path": sidecar_control["path"],
            "sha256": sidecar_control["sha256"],
        },
        "sidecar_ledger_before": sidecar_ledger_before,
        "sidecar_ledger_after": sidecar_ledger_after,
        "runner_log": artifact(runner_log, "runner.log"),
        "files": files,
    }
    local_manifest = local_root / "node_manifest.json"
    atomic_write_json(local_manifest, manifest)

    arm_root = evidence_root / arm_contract["arm"]
    arm_root.mkdir(parents=True, exist_ok=True)
    final_node = arm_root / f"node{node_rank}"
    stage_node = arm_root / f".node{node_rank}.{host}.tmp"
    if final_node.exists() or final_node.is_symlink() or stage_node.exists() or stage_node.is_symlink():
        raise RuntimeError(f"arm evidence destination already exists: {final_node}")
    stage_node.mkdir(mode=0o700)
    for path in sorted(valid_files):
        _copy_and_fsync(path, stage_node / path.name)
    _copy_and_fsync(runner_log, stage_node / "runner.log")
    _copy_and_fsync(local_manifest, stage_node / "node_manifest.json")
    fsync_directory(stage_node)
    os.replace(stage_node, final_node)
    fsync_directory(arm_root)

    copied_manifest = read_object(final_node / "node_manifest.json", "copied node manifest")
    expected_copied = {"node_manifest.json", "runner.log"} | {
        item["name"] for item in copied_manifest["files"]
    }
    if {path.name for path in final_node.iterdir()} != expected_copied:
        raise RuntimeError("copied arm node exact-set mismatch")
    for item in copied_manifest["files"]:
        validate_artifact(final_node / item["name"], item, "copied rank telemetry")
    validate_artifact(final_node / "runner.log", copied_manifest["runner_log"], "copied runner log")
    return final_node / "node_manifest.json", copied_manifest


def _create_arm_aggregate(
    result_root: Path,
    evidence_root: Path,
    preflight: dict[str, Any],
    arm_contract: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    nodes = []
    for node_rank in range(NODES):
        manifest_path = evidence_root / arm_contract["arm"] / f"node{node_rank}/node_manifest.json"
        manifest = read_object(manifest_path, f"arm node {node_rank} manifest")
        manifest_sha = sha256_file(manifest_path)
        ready_path = result_root / (
            f"arm_ready.{preflight['jobtag']}.{arm_contract['arm']}.n{node_rank}"
        )
        require_file(ready_path, f"arm node {node_rank} ready marker")
        ready_fields = ready_path.read_text(encoding="utf-8").strip().split("|")
        expected_ready = [
            "PRESERVED",
            preflight["jobtag"],
            arm_contract["arm"],
            arm_contract["arm_nonce"],
            str(node_rank),
            manifest_sha,
            str(manifest.get("status")),
        ]
        if ready_fields != expected_ready:
            raise RuntimeError(f"arm node {node_rank} ready/manifest binding mismatch")
        nodes.append(
            {
                "node_rank": node_rank,
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "ready": {"path": str(ready_path), "sha256": sha256_file(ready_path)},
                "status": manifest.get("status"),
                "run_rc": manifest.get("run_rc"),
                "collect_rc": manifest.get("collect_rc"),
                "rank_files": len(manifest.get("files", [])),
                "runner_log_sha256": (
                    manifest.get("runner_log", {}).get("sha256")
                    if isinstance(manifest.get("runner_log"), dict)
                    else None
                ),
                "sidecar_control_sha256": manifest.get("sidecar_control", {}).get(
                    "sha256"
                ),
                "sidecar_ledger_before_sha256": manifest.get(
                    "sidecar_ledger_before", {}
                ).get("sha256"),
                "sidecar_ledger_after_sha256": manifest.get(
                    "sidecar_ledger_after", {}
                ).get("sha256"),
            }
        )
    status = "PASS" if all(
        node["status"] == "PASS"
        and node["run_rc"] == 0
        and node["collect_rc"] == 0
        and node["rank_files"] == LOCAL_WORLD_SIZE
        for node in nodes
    ) else "FAIL"
    value = {
        "schema_version": "offline_arb_w16_gpic_abba_arm_aggregate_v1",
        "status": status,
        "jobtag": preflight["jobtag"],
        "attempt_nonce": preflight["attempt_nonce"],
        "arm": arm_contract["arm"],
        "arm_nonce": arm_contract["arm_nonce"],
        "threads": arm_contract["threads"],
        "start_step": arm_contract["start_step"],
        "stop_step": arm_contract["stop_step"],
        "world_size": WORLD_SIZE,
        "local_world_size": LOCAL_WORLD_SIZE,
        "per_rank_bs": PER_RANK_BS,
        "prefetch_steps": PREFETCH_STEPS,
        "preflight_sha256": sha256_file(
            result_root / f"preflight.{preflight['jobtag']}.contract.json"
        ),
        "animefreq_ledger_sha256": canonical_sha256(preflight["animefreq_ledger"]),
        "sidecar_baseline_sha256": canonical_sha256(preflight["sidecar_baseline"]),
        "arm_contract_sha256": sha256_file(
            result_root / f"arm_contract.{preflight['jobtag']}.{arm_contract['arm']}.json"
        ),
        "sidecar_full_verify": preflight["sidecar_full_verify"],
        "sidecar_header_sha256": preflight["sidecar_header_sha256"],
        "sidecar_release_digest": preflight["sidecar_release_digest"],
        "sidecar_snapshot_fingerprints_sha256": preflight[
            "sidecar_snapshot_fingerprints_sha256"
        ],
        "nodes": nodes,
    }
    path = evidence_root / arm_contract["arm"] / "arm_aggregate.json"
    atomic_write_json(path, value)
    return path, value


def node_wrapper(_args: argparse.Namespace) -> int:
    host, jobtag, node_rank = _host_identity()
    if os.environ.get("OFFLINE_ARB_ABBA_RUN_ID") != RUN_ID:
        raise RuntimeError("capture YAML/runner RUN_ID binding mismatch")
    python_interpreter = resolve_python_interpreter(PYTHON)
    if resolve_python_interpreter(Path(sys.executable)) != python_interpreter:
        raise RuntimeError("node wrapper is not running through the resolved Python target")
    for path in (
        PLAN / "header.json",
        PLAN / "checksums.json",
        CATALOG / "header.json",
        ANIME_FREQ / "header.json",
        SIDECAR / "header.json",
        *PRECHECK_FILES,
    ):
        require_file(path, "ABBA input")
    require_directory(PLAN, "W16 plan")
    require_directory(PLAN / "blocks", "W16 plan blocks")
    result_root = BASE / RUN_ID / jobtag
    result_root.mkdir(parents=True, exist_ok=True)
    evidence_root = result_root / f"evidence.{jobtag}"
    evidence_root.mkdir(exist_ok=True)

    expected_attempt_nonce = attempt_nonce(jobtag)
    fresh_marker = result_root / f"livepreflight.{jobtag}.n{node_rank}.fresh"
    atomic_write_text(
        fresh_marker,
        f"READY|{jobtag}|{expected_attempt_nonce}|{node_rank}|{int(time.time())}|{host}\n",
    )
    marker_paths = [
        result_root / f"livepreflight.{jobtag}.n{rank}.fresh" for rank in range(NODES)
    ]
    if node_rank == 0:
        wait_until(
            lambda: all(path.is_file() and not path.is_symlink() for path in marker_paths),
            timeout_s=1800,
            label="two fresh live-SHA preflight markers",
        )
        preflight = _create_preflight(result_root, jobtag, marker_paths)
        preflight_path = result_root / f"preflight.{jobtag}.contract.json"
        atomic_write_json(preflight_path, preflight)
    else:
        preflight_path = result_root / f"preflight.{jobtag}.contract.json"
    wait_until(
        lambda: preflight_path.is_file()
        and not preflight_path.is_symlink(),
        timeout_s=18_000,
        label="shared ABBA preflight contract",
    )
    preflight = read_object(preflight_path, "shared ABBA preflight")
    _validate_preflight(preflight, jobtag)

    arm_evidence: list[dict[str, Any]] = []
    for arm_index, (arm, threads, start, stop) in enumerate(ARMS):
        _assert_artifacts_unchanged(preflight["artifacts"])
        expected_arm_nonce = arm_nonce(
            preflight["attempt_nonce"], arm, threads, start, stop
        )
        marker = result_root / f"arm_fresh.{jobtag}.{arm}.n{node_rank}"
        atomic_write_text(
            marker,
            f"READY|{jobtag}|{arm}|{preflight['attempt_nonce']}|{expected_arm_nonce}|"
            f"{node_rank}|{int(time.time())}|{host}\n",
        )
        markers = [
            result_root / f"arm_fresh.{jobtag}.{arm}.n{rank}" for rank in range(NODES)
        ]
        contract_path = result_root / f"arm_contract.{jobtag}.{arm}.json"
        if node_rank == 0:
            wait_until(
                lambda: all(path.is_file() and not path.is_symlink() for path in markers),
                timeout_s=1800,
                label=f"{arm} fresh markers",
            )
            atomic_write_json(
                contract_path,
                _arm_contract(
                    result_root,
                    jobtag,
                    preflight_path,
                    arm,
                    threads,
                    start,
                    stop,
                    markers,
                    preflight["attempt_nonce"],
                ),
            )
        wait_until(
            lambda: contract_path.is_file()
            and not contract_path.is_symlink(),
            timeout_s=3600,
            label=f"{arm} contract",
        )
        arm_value = read_object(contract_path, f"{arm} contract")
        _validate_arm_contract(
            arm_value,
            jobtag=jobtag,
            arm=arm,
            threads=threads,
            start=start,
            stop=stop,
            preflight_path=preflight_path,
            attempt=preflight["attempt_nonce"],
        )
        _assert_rank_runtime_inputs(preflight)
        control = _sidecar_control(result_root, jobtag, arm, node_rank)
        if (
            control["value"].get("header_sha256") != preflight["sidecar_header_sha256"]
            or control["value"].get("release_digest")
            != preflight["sidecar_release_digest"]
            or control["value"].get("catalog_hash") != preflight["plan"]["catalog_hash"]
            or control["value"].get("header_file_sha256")
            != preflight["artifacts"]["sidecar_header"]["sha256"]
            or control["value"].get("success_file_sha256")
            != preflight["artifacts"]["sidecar_success"]["sha256"]
            or control["value"].get("catalog_header_sha256")
            != preflight["artifacts"]["catalog_header"]["sha256"]
        ):
            raise RuntimeError("per-arm/node sidecar control differs from preflight")
        ledger_before = _write_sidecar_ledger_evidence(
            result_root=result_root,
            preflight=preflight,
            arm_contract=arm_value,
            node_rank=node_rank,
            phase="before",
        )
        local_root = Path(
            f"/local/evan/offline_arb_w16_gpic_abba/{jobtag}/"
            f"{preflight['attempt_nonce']}/{arm}/{arm_value['arm_nonce']}/node{node_rank}"
        )
        if local_root.exists() or local_root.is_symlink():
            raise RuntimeError(f"arm local telemetry root already exists: {local_root}")
        local_root.mkdir(parents=True, mode=0o700)
        env = dict(os.environ)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "OFFLINE_ARB_GPIC_THREADS_PER_NODE": str(threads),
                "OFFLINE_ARB_PREFETCH_STEPS": str(PREFETCH_STEPS),
                "OFFLINE_ARB_PROFILE_WINDOWS": "0",
                "OFFLINE_ARB_WAIT_LOG_THRESHOLD_S": "0.000001",
                "OFFLINE_ARB_IO_STATE_EVERY_STEPS": "1",
                "OFFLINE_ARB_IO_TELEMETRY_DIR": str(local_root),
                "OFFLINE_ARB_ATTEMPT_JOBTAG": jobtag,
                "OFFLINE_ARB_ATTEMPT_NONCE": arm_value["arm_nonce"],
                "OFFLINE_ARB_ATTEMPT_HOST": host,
                "OFFLINE_ARB_SIDECAR_HEADER_SHA256": preflight["sidecar_header_sha256"],
                "OFFLINE_ARB_SIDECAR_RELEASE_DIGEST": preflight["sidecar_release_digest"],
                "OFFLINE_ARB_NODE_RANK": str(node_rank),
            }
        )
        log_path = result_root / f"rawgate.{jobtag}.{arm}.n{node_rank}.log"
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=60s",
            "21600s",
            str(PYTHON),
            "-m",
            "torch.distributed.run",
            "--nnodes",
            str(NODES),
            "--nproc-per-node",
            str(LOCAL_WORLD_SIZE),
            "--node-rank",
            str(node_rank),
            "--master-addr",
            preflight["master_addr"],
            "--master-port",
            str(29831 + arm_index),
            str(RUNNER_SELF),
            "rank-worker",
            "--arm",
            arm,
            "--gpic-threads-per-node",
            str(threads),
            "--start-step",
            str(start),
            "--stop-step",
            str(stop),
            "--config",
            str(CONFIG),
            "--plan-root",
            str(PLAN),
            "--catalog-root",
            str(CATALOG),
            "--anime-freq-root",
            str(ANIME_FREQ),
            "--sidecar-root",
            str(SIDECAR),
            "--preflight",
            str(preflight_path),
            "--expected-preflight-sha256",
            sha256_file(preflight_path),
            "--expected-runner-sha256",
            preflight["artifacts"]["runner"]["sha256"],
            "--expected-dataset-sha256",
            preflight["artifacts"]["dataset_source"]["sha256"],
            "--expected-planner-sha256",
            preflight["artifacts"]["planner_source"]["sha256"],
            "--expected-sidecar-runtime-sha256",
            preflight["artifacts"]["sidecar_runtime_source"]["sha256"],
            "--expected-config-sha256",
            preflight["artifacts"]["config"]["sha256"],
            "--expected-plan-header-sha256",
            preflight["artifacts"]["plan_header"]["sha256"],
            "--expected-plan-checksums-sha256",
            preflight["artifacts"]["plan_checksums"]["sha256"],
            "--expected-catalog-header-sha256",
            preflight["artifacts"]["catalog_header"]["sha256"],
            "--expected-animefreq-header-sha256",
            preflight["artifacts"]["animefreq_header"]["sha256"],
            "--expected-sidecar-header-file-sha256",
            preflight["artifacts"]["sidecar_header"]["sha256"],
        ]
        run_rc = _stream_subprocess(command, env, log_path)
        ledger_after = _write_sidecar_ledger_evidence(
            result_root=result_root,
            preflight=preflight,
            arm_contract=arm_value,
            node_rank=node_rank,
            phase="after",
        )
        manifest_path, manifest = _preserve_node_arm(
            result_root=result_root,
            evidence_root=evidence_root,
            local_root=local_root,
            runner_log=log_path,
            preflight=preflight,
            arm_contract=arm_value,
            host=host,
            node_rank=node_rank,
            run_rc=run_rc,
            sidecar_control=control,
            sidecar_ledger_before=ledger_before,
            sidecar_ledger_after=ledger_after,
        )
        _assert_rank_runtime_inputs(preflight)
        ready_path = result_root / f"arm_ready.{jobtag}.{arm}.n{node_rank}"
        atomic_write_text(
            ready_path,
            f"PRESERVED|{jobtag}|{arm}|{arm_value['arm_nonce']}|{node_rank}|"
            f"{sha256_file(manifest_path)}|{manifest['status']}\n",
        )
        ready_paths = [
            result_root / f"arm_ready.{jobtag}.{arm}.n{rank}" for rank in range(NODES)
        ]
        wait_until(
            lambda: all(path.is_file() and not path.is_symlink() for path in ready_paths),
            timeout_s=3600,
            label=f"{arm} node evidence",
        )
        aggregate_path = evidence_root / arm / "arm_aggregate.json"
        if node_rank == 0:
            aggregate_path, aggregate = _create_arm_aggregate(
                result_root, evidence_root, preflight, arm_value
            )
        wait_until(
            lambda: aggregate_path.is_file() and not aggregate_path.is_symlink(),
            timeout_s=3600,
            label=f"{arm} aggregate",
        )
        aggregate = read_object(aggregate_path, f"{arm} aggregate")
        arm_evidence.append(
            {
                "arm": arm,
                "path": str(aggregate_path),
                "sha256": sha256_file(aggregate_path),
                "status": aggregate.get("status"),
            }
        )
        if aggregate.get("status") != "PASS":
            raise RuntimeError(f"ABBA capture arm failed: {arm}")

    sidecar_post_path = result_root / f"sidecarfull.post.{jobtag}.json"
    if node_rank == 0:
        _assert_rank_runtime_inputs(preflight)
        initial_full = read_object(
            Path(preflight["sidecar_full_verify"]["path"]),
            "initial full sidecar verification",
        )
        post_full = _full_sidecar_snapshot()
        if post_full != initial_full:
            raise RuntimeError("post-ABBA full sidecar verification differs from preflight")
        atomic_write_json(sidecar_post_path, post_full)
    wait_until(
        lambda: sidecar_post_path.is_file() and not sidecar_post_path.is_symlink(),
        timeout_s=18_000,
        label="post-ABBA full sidecar verification",
    )
    post_full = read_object(sidecar_post_path, "post-ABBA full sidecar verification")
    if post_full != read_object(
        Path(preflight["sidecar_full_verify"]["path"]),
        "initial full sidecar verification",
    ):
        raise RuntimeError("shared post-ABBA sidecar verification mismatch")
    sidecar_post = {"path": str(sidecar_post_path), "sha256": sha256_file(sidecar_post_path)}

    master_path = result_root / f"abba.{jobtag}.aggregate.json"
    if node_rank == 0:
        master = {
            "schema_version": "offline_arb_w16_gpic_abba_master_aggregate_v1",
            "status": "PASS" if all(item["status"] == "PASS" for item in arm_evidence) else "FAIL",
            "run_id": RUN_ID,
            "jobtag": jobtag,
            "attempt_nonce": preflight["attempt_nonce"],
            "world_size": WORLD_SIZE,
            "local_world_size": LOCAL_WORLD_SIZE,
            "nodes": NODES,
            "per_rank_bs": PER_RANK_BS,
            "prefetch_steps": PREFETCH_STEPS,
            "preflight": {"path": str(preflight_path), "sha256": sha256_file(preflight_path)},
            "sidecar_full_post": sidecar_post,
            "arms": arm_evidence,
        }
        atomic_write_json(master_path, master)
    wait_until(
        lambda: master_path.is_file() and not master_path.is_symlink(),
        timeout_s=3600,
        label="ABBA master aggregate",
    )
    master = read_object(master_path, "ABBA master aggregate")
    if master.get("status") != "PASS" or master.get("arms") != arm_evidence:
        raise RuntimeError("ABBA master aggregate is not bound PASS")
    print(
        f"W16_GPIC_ABBA_CAPTURE_PASS result={master_path} sha256={sha256_file(master_path)}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("node-wrapper")
    worker = subparsers.add_parser("rank-worker")
    worker.add_argument("--arm", required=True, choices=[arm for arm, *_ in ARMS])
    worker.add_argument("--gpic-threads-per-node", type=int, required=True)
    worker.add_argument("--start-step", type=int, required=True)
    worker.add_argument("--stop-step", type=int, required=True)
    worker.add_argument("--config", required=True)
    worker.add_argument("--plan-root", required=True)
    worker.add_argument("--catalog-root", required=True)
    worker.add_argument("--anime-freq-root", required=True)
    worker.add_argument("--sidecar-root", required=True)
    worker.add_argument("--preflight", required=True)
    worker.add_argument("--expected-preflight-sha256", required=True)
    worker.add_argument("--expected-runner-sha256", required=True)
    worker.add_argument("--expected-dataset-sha256", required=True)
    worker.add_argument("--expected-planner-sha256", required=True)
    worker.add_argument("--expected-sidecar-runtime-sha256", required=True)
    worker.add_argument("--expected-config-sha256", required=True)
    worker.add_argument("--expected-plan-header-sha256", required=True)
    worker.add_argument("--expected-plan-checksums-sha256", required=True)
    worker.add_argument("--expected-catalog-header-sha256", required=True)
    worker.add_argument("--expected-animefreq-header-sha256", required=True)
    worker.add_argument("--expected-sidecar-header-file-sha256", required=True)
    worker.add_argument("--process-group-timeout-s", type=int, default=3600)
    args = parser.parse_args()
    if args.mode == "rank-worker":
        expected = next(item for item in ARMS if item[0] == args.arm)
        if (args.gpic_threads_per_node, args.start_step, args.stop_step) != expected[1:]:
            parser.error("rank-worker arm/thread/slice identity mismatch")
        if args.stop_step - args.start_step != SLICE_STEPS:
            parser.error("rank-worker slice must contain exactly 192 steps")
        for name in (
            "expected_runner_sha256",
            "expected_dataset_sha256",
            "expected_planner_sha256",
            "expected_sidecar_runtime_sha256",
            "expected_config_sha256",
            "expected_plan_header_sha256",
            "expected_plan_checksums_sha256",
            "expected_catalog_header_sha256",
            "expected_animefreq_header_sha256",
            "expected_sidecar_header_file_sha256",
            "expected_preflight_sha256",
        ):
            if SHA_RE.fullmatch(getattr(args, name)) is None:
                parser.error(f"--{name.replace('_', '-')} must be a lowercase SHA-256")
        if args.process_group_timeout_s <= 0:
            parser.error("--process-group-timeout-s must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "rank-worker":
        return rank_worker(args)
    return node_wrapper(args)


if __name__ == "__main__":
    raise SystemExit(main())
