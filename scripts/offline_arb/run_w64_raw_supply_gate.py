#!/usr/bin/env python3
"""Exercise the W64 OFFLINE_ARB supply path without decoding images.

The wrapper supplies an attempt-bound runtime contract.  Every rank re-hashes
that complete contract before import, after import, before the measured run and
after close.  The gate waits for the exact terminal telemetry-window set before
closing, so a successful capture cannot silently omit late window records.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import stat
import threading
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch.distributed as dist
from omegaconf import OmegaConf


EXPECTED_WORLD_SIZE = 64
EXPECTED_LOCAL_WORLD_SIZE = 8
START_STEP = 0
STOP_STEP = 192
PREFETCH_STEPS = 4
RUNTIME_CONTRACT_SCHEMA = "offline_arb_w64_runtime_contract_v1"
GPIC_CONSISTENCY_MODES = frozenset({"strict_path", "fd_stable"})
SHA_RE = re.compile(r"[0-9a-f]{64}")
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
RUNTIME_MODULES = {
    "caption_contract_source": (
        "scripts.offline_arb.caption_sidecar_contract",
        "scripts/offline_arb/caption_sidecar_contract.py",
    ),
    "training_package_source": (
        "training",
        "training/__init__.py",
    ),
    "dataline_package_source": (
        "training.dataline",
        "training/dataline/__init__.py",
    ),
    "offline_package_source": (
        "training.dataline_offline",
        "training/dataline_offline/__init__.py",
    ),
    "dataset_source": (
        "training.dataline_offline.offline_dataset",
        "training/dataline_offline/offline_dataset.py",
    ),
    "planner_source": (
        "training.dataline_offline.planner_v1",
        "training/dataline_offline/planner_v1.py",
    ),
    "sidecar_runtime_source": (
        "training.dataline_offline.caption_sidecar",
        "training/dataline_offline/caption_sidecar.py",
    ),
    "complete_dataset_source": (
        "training.dataline.complete_dataset",
        "training/dataline/complete_dataset.py",
    ),
    "dataline_config_source": (
        "training.dataline.config",
        "training/dataline/config.py",
    ),
    "image_fetch_source": (
        "training.dataline.image_fetch",
        "training/dataline/image_fetch.py",
    ),
    "manifest_index_source": (
        "training.dataline.manifest_index",
        "training/dataline/manifest_index.py",
    ),
    "multi_ar_source": (
        "training.dataline.multi_ar",
        "training/dataline/multi_ar.py",
    ),
    "native_safety_source": (
        "training.dataline.native_safety",
        "training/dataline/native_safety.py",
    ),
    "pil_safety_source": (
        "training.dataline.pil_safety",
        "training/dataline/pil_safety.py",
    ),
    "caption_build_source": (
        "training.dataline.caption_build",
        "training/dataline/caption_build.py",
    ),
    "sampler_source": (
        "training.dataline.sampler",
        "training/dataline/sampler.py",
    ),
}
RUNTIME_ARTIFACT_KEYS = frozenset(
    {
        "runner",
        "capture_yaml",
        "config",
        "freeze_manifest",
        "preflight_verify",
        "plan_header",
        "plan_checksums",
        "catalog_header",
        "animefreq_header",
        "sidecar_header",
        "sampling",
        "freeze_builder_source",
        *RUNTIME_MODULES,
    }
)
EMPTY_RUNTIME_ARTIFACT_KEYS = frozenset(
    {"training_package_source", "dataline_package_source"}
)
RUNTIME_NAMESPACE_PACKAGES = {
    "scripts": "scripts",
    "scripts.offline_arb": "scripts/offline_arb",
}


def _strict_env_int(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.isdigit():
        raise RuntimeError(f"{name} must be an unsigned integer, got {raw!r}")
    return int(raw)


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    resolved = path.expanduser().resolve(strict=True)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or not resolved.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return resolved


def _real_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise RuntimeError(f"{label} must be a real directory: {path}")
    return resolved


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _sha256_file(path: Path) -> str:
    before = path.lstat()
    before_identity = _stat_identity(before)
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise RuntimeError(f"snapshot source is not a regular non-symlink: {path}")
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        opened = os.fstat(handle.fileno())
        opened_identity = _stat_identity(opened)
        if opened_identity != before_identity:
            raise RuntimeError(f"snapshot source changed while opening: {path}")
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
        closed = os.fstat(handle.fileno())
        closed_identity = _stat_identity(closed)
    after = path.lstat()
    after_identity = _stat_identity(after)
    if closed_identity != before_identity or after_identity != before_identity:
        raise RuntimeError(f"snapshot source changed while hashing: {path}")
    return digest.hexdigest()


def _read_snapshot(path: Path, label: str) -> tuple[bytes, str]:
    canonical = _regular_file(path, label)
    before = canonical.lstat()
    identity = _stat_identity(before)
    with canonical.open("rb", buffering=0) as handle:
        if _stat_identity(os.fstat(handle.fileno())) != identity:
            raise RuntimeError(f"{label} changed while opening")
        payload = handle.read()
        closed = os.fstat(handle.fileno())
    after = canonical.lstat()
    for value in (closed, after):
        if _stat_identity(value) != identity:
            raise RuntimeError(f"{label} changed while reading")
    return payload, hashlib.sha256(payload).hexdigest()


def _artifact_entry(
    path: Path,
    entry: Any,
    label: str,
    *,
    allow_empty: bool = False,
) -> Path:
    if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
        raise RuntimeError(f"{label} declaration is malformed")
    canonical = _regular_file(path, label)
    if entry.get("path") != str(canonical):
        raise RuntimeError(f"{label} canonical path differs from runtime contract")
    if (
        type(entry.get("bytes")) is not int
        or entry["bytes"] < 0
        or (entry["bytes"] == 0 and not allow_empty)
    ):
        raise RuntimeError(f"{label} byte declaration is malformed")
    if canonical.stat().st_size != entry["bytes"]:
        raise RuntimeError(f"{label} byte size differs from runtime contract")
    expected_sha = entry.get("sha256")
    if not isinstance(expected_sha, str) or SHA_RE.fullmatch(expected_sha) is None:
        raise RuntimeError(f"{label} SHA declaration is malformed")
    if _sha256_file(canonical) != expected_sha:
        raise RuntimeError(f"{label} SHA differs from runtime contract")
    return canonical


def _expected_artifact_paths(args: argparse.Namespace, contract: dict[str, Any]) -> dict[str, Path]:
    repo = Path(__file__).resolve(strict=True).parents[2]
    jobtag = contract["jobtag"]
    contract_root = Path(args.runtime_contract).resolve(strict=True).parent
    result = {
        "runner": Path(__file__),
        "capture_yaml": repo / "jobs/offline_arb_w64_raw_supply_gate.yaml",
        "config": Path(args.config),
        "freeze_manifest": Path(args.freeze),
        "preflight_verify": contract_root / f"freezepreflight.{jobtag}.json",
        "plan_header": Path(args.plan_root) / "header.json",
        "plan_checksums": Path(args.plan_root) / "checksums.json",
        "catalog_header": Path(args.catalog_root) / "header.json",
        "animefreq_header": Path(args.anime_freq_root) / "header.json",
        "sidecar_header": Path(args.sidecar_root) / "header.json",
        "sampling": Path(args.sampling),
        "freeze_builder_source": repo / "scripts/offline_arb/make_freeze_manifest.py",
    }
    result.update({key: repo / relative for key, (_, relative) in RUNTIME_MODULES.items()})
    return result


def _load_runtime_contract(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    contract_path = _regular_file(Path(args.runtime_contract), "runtime contract")
    contract_raw, contract_sha = _read_snapshot(contract_path, "runtime contract")
    if contract_sha != args.expected_runtime_contract_sha256:
        raise RuntimeError("runtime contract SHA differs from wrapper-bound identity")
    try:
        value = json.loads(contract_raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read runtime contract: {exc}") from exc
    expected_keys = {
        "schema_version",
        "run_id",
        "jobtag",
        "attempt_nonce",
        "gpic_source_consistency_mode",
        "gpic_threads_per_node",
        "freeze_release_digest",
        "freeze_sha256",
        "preflight_verify_sha256",
        "artifacts",
        "fresh_markers",
        "namespace_packages",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError("runtime contract key set drift")
    if value.get("schema_version") != RUNTIME_CONTRACT_SCHEMA:
        raise RuntimeError("runtime contract schema drift")
    for key in ("attempt_nonce", "freeze_release_digest", "freeze_sha256", "preflight_verify_sha256"):
        if not isinstance(value.get(key), str) or SHA_RE.fullmatch(value[key]) is None:
            raise RuntimeError(f"runtime contract {key} is malformed")
    jobtag = value.get("jobtag")
    if (
        not isinstance(jobtag, str)
        or jobtag in {"", ".", ".."}
        or re.fullmatch(r"[A-Za-z0-9_.-]+", jobtag) is None
        or Path(jobtag).name != jobtag
    ):
        raise RuntimeError("runtime contract jobtag is malformed")
    if value.get("gpic_source_consistency_mode") not in GPIC_CONSISTENCY_MODES:
        raise RuntimeError("runtime contract GPIC consistency mode is unsupported")
    if value["gpic_source_consistency_mode"] != args.expected_gpic_source_consistency_mode:
        raise RuntimeError("runtime contract GPIC consistency mode differs from wrapper")
    if (
        type(value.get("gpic_threads_per_node")) is not int
        or value["gpic_threads_per_node"] != 16
        or value["gpic_threads_per_node"] != args.expected_gpic_threads_per_node
        or os.environ.get("OFFLINE_ARB_GPIC_THREADS_PER_NODE")
        != str(value["gpic_threads_per_node"])
    ):
        raise RuntimeError("runtime contract/configured/environment GPIC thread identity drift")
    if os.environ.get("OFFLINE_ARB_ATTEMPT_JOBTAG") != jobtag:
        raise RuntimeError("runtime contract jobtag differs from attempt environment")
    if os.environ.get("OFFLINE_ARB_ATTEMPT_NONCE") != value["attempt_nonce"]:
        raise RuntimeError("runtime contract nonce differs from attempt environment")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != RUNTIME_ARTIFACT_KEYS:
        raise RuntimeError("runtime contract artifact set drift")
    namespace_packages = value.get("namespace_packages")
    expected_namespaces = {
        name: str((Path(__file__).resolve(strict=True).parents[2] / relative).resolve(strict=True))
        for name, relative in RUNTIME_NAMESPACE_PACKAGES.items()
    }
    if namespace_packages != expected_namespaces:
        raise RuntimeError("runtime contract namespace-package closure drift")
    expected_paths = _expected_artifact_paths(args, value)
    for key in sorted(RUNTIME_ARTIFACT_KEYS):
        _artifact_entry(
            expected_paths[key],
            artifacts[key],
            f"runtime artifact {key}",
            allow_empty=key in EMPTY_RUNTIME_ARTIFACT_KEYS,
        )
    if artifacts["runner"]["sha256"] != args.expected_runner_sha256:
        raise RuntimeError("runtime contract runner SHA differs from wrapper")
    if artifacts["freeze_manifest"]["sha256"] != value["freeze_sha256"]:
        raise RuntimeError("runtime contract freeze SHA does not close")
    if artifacts["preflight_verify"]["sha256"] != value["preflight_verify_sha256"]:
        raise RuntimeError("runtime contract preflight SHA does not close")
    markers = value.get("fresh_markers")
    if not isinstance(markers, dict) or set(markers) != {str(index) for index in range(8)}:
        raise RuntimeError("runtime contract fresh-marker set drift")
    for node, entry in markers.items():
        marker = contract_path.parent / f"freezepreflight.{jobtag}.n{node}.fresh"
        _artifact_entry(marker, entry, f"fresh marker node {node}")
    return value, contract_sha


def _assert_runtime_artifacts(
    args: argparse.Namespace,
    contract: dict[str, Any],
    contract_sha: str,
    phase: str,
    *,
    rank: int | None = None,
) -> None:
    contract_path = _regular_file(Path(args.runtime_contract), "runtime contract")
    if _sha256_file(contract_path) != contract_sha:
        raise RuntimeError(f"runtime contract changed during {phase}")
    for key, path in _expected_artifact_paths(args, contract).items():
        _artifact_entry(
            path,
            contract["artifacts"][key],
            f"{phase} artifact {key}",
            allow_empty=key in EMPTY_RUNTIME_ARTIFACT_KEYS,
        )
    prefix = "wrapper" if rank is None else str(rank)
    print(
        f"W64_RUNTIME_CLOSURE rank={prefix} phase={phase} contract_sha256={contract_sha}",
        flush=True,
    )


def _assert_imported_modules(contract: dict[str, Any]) -> Any:
    modules: dict[str, Any] = {}
    for module_name, expected_root in contract["namespace_packages"].items():
        module = importlib.import_module(module_name)
        search_locations = getattr(module.__spec__, "submodule_search_locations", None)
        resolved_locations = (
            {str(Path(location).resolve(strict=True)) for location in search_locations}
            if search_locations is not None
            else set()
        )
        if getattr(module, "__file__", None) is not None or resolved_locations != {
            expected_root
        }:
            raise RuntimeError(f"namespace package closure drift: {module_name}")
    for key, (module_name, _) in RUNTIME_MODULES.items():
        module = importlib.import_module(module_name)
        expected = contract["artifacts"][key]
        module_path = Path(module.__file__).resolve(strict=True)
        if module_path != Path(expected["path"]) or _sha256_file(module_path) != expected["sha256"]:
            raise RuntimeError(f"imported module differs from runtime contract: {module_name}")
        modules[key] = module
    return modules["dataset_source"]


def _dataset_kwargs(
    args: argparse.Namespace,
    rank: int,
    expected_consistency_mode: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    config_path = _regular_file(args.config, "config")
    config_raw, config_sha = _read_snapshot(config_path, "config")
    if config_sha != contract["artifacts"]["config"]["sha256"]:
        raise RuntimeError("config snapshot differs from runtime contract")
    cfg = OmegaConf.create(config_raw.decode("utf-8"))
    value = OmegaConf.to_container(cfg.dataset, resolve=True)
    if not isinstance(value, dict):
        raise RuntimeError("config.dataset must resolve to a mapping")
    kwargs = dict(value)
    kwargs.pop("store_cls", None)
    configured_mode = kwargs.get("gpic_source_consistency_mode")
    if configured_mode != expected_consistency_mode:
        raise RuntimeError(
            "configured GPIC consistency mode differs from runtime contract: "
            f"configured={configured_mode!r}, expected={expected_consistency_mode!r}"
        )
    configured_threads = kwargs.get("gpic_threads_per_node")
    if configured_threads != contract["gpic_threads_per_node"]:
        raise RuntimeError(
            "configured GPIC threads-per-node differs from runtime contract: "
            f"configured={configured_threads!r}, expected={contract['gpic_threads_per_node']!r}"
        )
    multi_ar = dict(kwargs.get("multi_ar") or {})
    multi_ar["assignment_mode"] = "offline_exact_v1"
    kwargs.update(
        {
            "offline_plan_root": str(_real_directory(args.plan_root, "plan root")),
            "offline_catalog_root": str(_real_directory(args.catalog_root, "catalog root")),
            "offline_anime_freq_root": str(
                _real_directory(args.anime_freq_root, "anime frequency root")
            ),
            "offline_caption_sidecar_root": str(
                _real_directory(args.sidecar_root, "caption sidecar root")
            ),
            "rank": rank,
            "world_size": EXPECTED_WORLD_SIZE,
            "stage": "256",
            "multi_ar": multi_ar,
            # The raw gate deliberately stops before tokenization/decode.
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


def _maintain_bounded_prefetch(dataset: Any) -> None:
    while (
        int(dataset._scheduled_until) < STOP_STEP
        and int(dataset._scheduled_until) - int(dataset._next_step)
        <= int(dataset.prefetch_steps)
    ):
        dataset._schedule_window(int(dataset._scheduled_until))
    if int(dataset._scheduled_until) > STOP_STEP:
        raise AssertionError(
            f"prefetch escaped raw gate: scheduled_until={dataset._scheduled_until}"
        )


def _run_raw_gate(
    dataset: Any,
    rank: int,
    expected_consistency_mode: str,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_RAW_API if not callable(getattr(dataset, name, None))]
    if missing:
        raise RuntimeError(f"offline raw-supply API drifted; missing={missing}")
    first = int(dataset.plan_header["materialization"]["first_materialized_step"])
    total = int(dataset.plan_header["total_steps"])
    batch_size = int(dataset.plan_header["per_rank_bs"])
    if first != START_STEP or total < STOP_STEP or batch_size <= 0:
        raise AssertionError("W64 plan materialization/batch contract drift")
    if int(dataset.prefetch_steps) != PREFETCH_STEPS:
        raise AssertionError("W64 prefetch depth drift")
    if dataset._gpic_payload_in_memory is not True:
        raise AssertionError("W64 raw gate requires formal gpic_payload_in_memory=true")
    if getattr(dataset, "_gpic_source_consistency_mode", None) != expected_consistency_mode:
        raise AssertionError("W64 runtime GPIC consistency mode drift")
    if int(getattr(dataset, "_gpic_threads_per_node", -1)) != 16:
        raise AssertionError("W64 runtime GPIC threads-per-node drift")

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
    expected_window_keys: set[tuple[int, str]] = set()
    dataset.set_start_step(START_STEP)
    dist.barrier()
    dataset._start_pipeline()
    if int(dataset._scheduled_until) > STOP_STEP:
        raise AssertionError("initial prefetch escaped W64 raw gate")

    bytes_by_domain: Counter[str] = Counter()
    items_by_domain: Counter[str] = Counter()
    total_raw_bytes = 0
    total_items = 0
    total_wait_s = 0.0
    max_wait_s = 0.0
    retired_steps = 0
    gate_started = time.perf_counter()
    for step in range(START_STEP, STOP_STEP):
        expected_bucket, expected_items = dataset._plan_items_for_step(step)
        window_start = START_STEP + ((step - START_STEP) // PREFETCH_STEPS) * PREFETCH_STEPS
        expected_window_keys.update(
            (window_start, str(item.domain)) for item in expected_items
        )
        if len(expected_items) != batch_size:
            raise AssertionError(f"rank {rank} step {step}: plan slot count drift")
        if dataset._step_bucket.get(step) != expected_bucket:
            raise AssertionError(f"rank {rank} step {step}: scheduled bucket drift")

        dist.barrier()
        wait_started = time.perf_counter()
        prepared = dataset._wait_for_step(step)
        wait_s = time.perf_counter() - wait_started
        total_wait_s += wait_s
        max_wait_s = max(max_wait_s, wait_s)
        if len(prepared) != batch_size:
            raise AssertionError(f"rank {rank} step {step}: ready slot count drift")

        step_raw_bytes = 0
        for slot, (actual, expected) in enumerate(zip(prepared, expected_items)):
            if _planned_identity(actual.planned) != _planned_identity(expected):
                raise AssertionError(f"rank {rank} step {step} slot {slot}: plan identity drift")
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
                raise AssertionError("formal in-memory raw contract drift")
            raw = dataset._raw_for_item(actual)
            if not isinstance(raw, bytes) or not raw:
                raise AssertionError("raw payload is empty/non-bytes")
            raw_bytes = len(raw)
            step_raw_bytes += raw_bytes
            total_raw_bytes += raw_bytes
            total_items += 1
            items_by_domain[str(expected.domain)] += 1
            bytes_by_domain[str(expected.domain)] += raw_bytes
            del raw

        dataset._retire_step(step, prepared)
        if step in dataset._ready or step in dataset._expected or step in dataset._step_bucket:
            raise AssertionError(f"rank {rank} step {step}: retire left live step state")
        with dataset._condition:
            negative_domains = {
                domain: int(dataset._domain_ready[domain])
                for domain in dataset.domains
                if int(dataset._domain_ready[domain]) < 0
            }
        if negative_domains:
            raise AssertionError(f"rank {rank} step {step}: negative ready counters")
        dataset._next_step = step + 1
        if dataset._next_step < STOP_STEP:
            _maintain_bounded_prefetch(dataset)
        dataset._refresh_metrics(plan_step=step)
        dataset._emit_state_record(step)
        retired_steps += 1
        print(
            "W64_RAW_STEP_RETIRED "
            f"rank={rank} step={step} items={len(prepared)} raw_bytes={step_raw_bytes} "
            f"wait_s={wait_s:.9f} scheduled_until={int(dataset._scheduled_until)}",
            flush=True,
        )

    if retired_steps != STOP_STEP - START_STEP:
        raise AssertionError("retired-step total drift")
    if total_items != (STOP_STEP - START_STEP) * batch_size:
        raise AssertionError("raw-item total drift")
    if int(dataset._next_step) != STOP_STEP or int(dataset._scheduled_until) != STOP_STEP:
        raise AssertionError("terminal step/schedule boundary drift")

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
    return {
        "rank": rank,
        "start_step": START_STEP,
        "stop_step": STOP_STEP,
        "retired_steps": retired_steps,
        "per_rank_batch_size": batch_size,
        "raw_items": total_items,
        "raw_bytes": total_raw_bytes,
        "items_by_domain": dict(sorted(items_by_domain.items())),
        "bytes_by_domain": dict(sorted(bytes_by_domain.items())),
        "wait_total_s": total_wait_s,
        "wait_max_s": max_wait_s,
        "elapsed_s": time.perf_counter() - gate_started,
        "decode_calls": 0,
        "telemetry_windows": len(expected_window_keys),
        "gpic_source_consistency_mode": expected_consistency_mode,
        "gpic_threads_per_node": int(dataset._gpic_threads_per_node),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--anime-freq-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--expected-runtime-contract-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-gpic-source-consistency-mode", required=True)
    parser.add_argument("--expected-gpic-threads-per-node", type=int, required=True)
    parser.add_argument("--verify-runtime-contract-only", action="store_true")
    parser.add_argument("--process-group-timeout-s", type=int, default=3600)
    args = parser.parse_args()
    if args.process_group_timeout_s <= 0:
        parser.error("--process-group-timeout-s must be positive")
    for name in ("expected_runtime_contract_sha256", "expected_runner_sha256"):
        if SHA_RE.fullmatch(getattr(args, name)) is None:
            parser.error(f"--{name.replace('_', '-')} must be a lowercase SHA-256")
    if args.expected_gpic_source_consistency_mode not in GPIC_CONSISTENCY_MODES:
        parser.error("unsupported --expected-gpic-source-consistency-mode")
    if args.expected_gpic_threads_per_node != 16:
        parser.error("--expected-gpic-threads-per-node must be 16")
    return args


def main() -> int:
    args = parse_args()
    contract, contract_sha = _load_runtime_contract(args)
    _assert_runtime_artifacts(args, contract, contract_sha, "preflight")
    if args.verify_runtime_contract_only:
        print(
            "W64_RUNTIME_CONTRACT_PASS "
            f"sha256={contract_sha} mode={contract['gpic_source_consistency_mode']} "
            f"gpic_threads_per_node={contract['gpic_threads_per_node']}",
            flush=True,
        )
        return 0
    if dist.is_initialized():
        raise RuntimeError("process group was unexpectedly initialized before raw gate")
    dist.init_process_group(
        backend="gloo",
        timeout=timedelta(seconds=args.process_group_timeout_s),
    )
    dataset = None
    rank = dist.get_rank()
    try:
        world_size = dist.get_world_size()
        local_rank = _strict_env_int("LOCAL_RANK")
        local_world_size = _strict_env_int("LOCAL_WORLD_SIZE")
        node_rank = _strict_env_int("OFFLINE_ARB_NODE_RANK")
        if dist.get_backend() != "gloo" or world_size != EXPECTED_WORLD_SIZE:
            raise RuntimeError("raw gate backend/world-size drift")
        if local_world_size != EXPECTED_LOCAL_WORLD_SIZE:
            raise RuntimeError("raw gate local-world-size drift")
        expected_rank = node_rank * EXPECTED_LOCAL_WORLD_SIZE + local_rank
        if not 0 <= local_rank < EXPECTED_LOCAL_WORLD_SIZE or rank != expected_rank:
            raise RuntimeError("raw gate global/local rank mapping drift")

        _assert_runtime_artifacts(args, contract, contract_sha, "before_import", rank=rank)
        offline_module = _assert_imported_modules(contract)
        _assert_runtime_artifacts(args, contract, contract_sha, "after_import", rank=rank)

        def _decode_is_forbidden(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("decode/resize/tensor conversion must never run in raw gate")

        offline_module.decode_rgb = _decode_is_forbidden
        offline_module.resize_u8_to_bucket = _decode_is_forbidden
        offline_module.uint8_to_norm_tensor = _decode_is_forbidden
        mode = contract["gpic_source_consistency_mode"]
        kwargs = _dataset_kwargs(args, rank, mode, contract)
        dataset = offline_module.OfflineExactDataset(**kwargs)
        if getattr(dataset, "_gpic_source_consistency_mode", None) != mode:
            raise RuntimeError("constructed dataset GPIC consistency mode differs from contract")
        _assert_runtime_artifacts(args, contract, contract_sha, "before_run", rank=rank)
        result = _run_raw_gate(dataset, rank, mode)
        _assert_runtime_artifacts(args, contract, contract_sha, "after_run", rank=rank)
        dataset.close()
        _assert_closed_evidence(dataset, rank)
        _assert_runtime_artifacts(args, contract, contract_sha, "after_close", rank=rank)
        dataset = None
        dist.barrier()
        print(
            "W64_RAW_RANK_PASS "
            f"rank={result['rank']} start_step={result['start_step']} "
            f"stop_step={result['stop_step']} retired_steps={result['retired_steps']} "
            f"per_rank_batch_size={result['per_rank_batch_size']} "
            f"raw_items={result['raw_items']} raw_bytes={result['raw_bytes']} "
            f"wait_total_s={result['wait_total_s']:.9f} "
            f"wait_max_s={result['wait_max_s']:.9f} "
            f"elapsed_s={result['elapsed_s']:.9f} decode_calls={result['decode_calls']} "
            f"telemetry_windows={result['telemetry_windows']} "
            f"gpic_source_consistency_mode={result['gpic_source_consistency_mode']} "
            f"gpic_threads_per_node={result['gpic_threads_per_node']}",
            flush=True,
        )
        return 0
    finally:
        try:
            if dataset is not None:
                dataset.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
