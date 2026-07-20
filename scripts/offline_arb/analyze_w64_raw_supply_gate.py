#!/usr/bin/env python3
"""Strict evidence and capacity gate for the 64-rank W64 raw-supply capture."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib
import json
import math
import os
import re
import secrets
import stat
import statistics
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf
from training.dataline_offline.planner_v1 import OfflinePlan


TELEMETRY_SCHEMA = "offline_arb_io_profile_v1"
AGGREGATE_SCHEMA = "offline_arb_w64_raw_supply_aggregate_v1"
MANIFEST_SCHEMA = "offline_arb_w64_raw_supply_node_manifest_v1"
ANALYSIS_SCHEMA = "offline_arb_w64_raw_supply_analysis_v1"
RUNTIME_CONTRACT_SCHEMA = "offline_arb_w64_runtime_contract_v1"
FREEZE_SCHEMA = "offline_arb_freeze_v3_caption_sidecar"
WORLD_SIZE = 64
LOCAL_WORLD_SIZE = 8
NODES = 8
PER_RANK_BS = 36
EXPECTED_PLAN_ID = "stage256_offline_exact_v1_w64_prod"
EXPECTED_DOMAINS = ("anime", "monet_real", "gpic", "synthetic", "textatlas")
PREFETCH_STEPS = 4
START_STEP = 0
STOP_STEP = 192
STEADY_STEPS = range(40, STOP_STEP)
STEADY_WINDOW_STARTS = range(40, STOP_STEP, PREFETCH_STEPS)
ALL_WINDOW_STARTS = range(START_STEP, STOP_STEP, PREFETCH_STEPS)
TARGET_SECONDS_PER_STEP = 2.592
CAPACITY_HEADROOM_RATIO = 1.0 / 0.9
WAIT_MEAN_MAX_S = TARGET_SECONDS_PER_STEP * 0.9
WAIT_P95_MAX_S = TARGET_SECONDS_PER_STEP * 2.0
WAIT_MAX_S = TARGET_SECONDS_PER_STEP * 4.0
WAIT_OVER_TARGET_FRACTION_MAX = 0.10
WINDOW_MEAN_MAX_S = PREFETCH_STEPS * TARGET_SECONDS_PER_STEP * 0.9
WINDOW_TAIL_MAX_S = PREFETCH_STEPS * TARGET_SECONDS_PER_STEP * 2.0
RUNNER_ELAPSED_PER_STEP_MAX_S = 2.75
FORMAL_STEPS = 166_667
WAIT_LOG_THRESHOLD_S = 0.000001
ALLOWED_RECORD_TYPES = {"identity", "window", "wait", "state", "closure"}
FREEZE_RELEASE_KEYS = (
    "schema_version",
    "identities",
    "inputs",
    "plan_blocks",
    "catalog",
    "animefreq",
    "caption_sidecar",
    "training_sources",
)
FREEZE_TOP_LEVEL_KEYS = set(FREEZE_RELEASE_KEYS) | {"frozen_at", "git_head_at_freeze"}
FROZEN_GATE_FILES = (
    "scripts/offline_arb/run_w64_raw_supply_gate.py",
    "jobs/offline_arb_w64_raw_supply_gate.yaml",
    "scripts/offline_arb/analyze_w64_raw_supply_gate.py",
    "jobs/offline_arb_w64_raw_supply_analyze.yaml",
)
GPIC_CONSISTENCY_MODES = frozenset({"strict_path", "fd_stable"})
RUNTIME_MODULE_FILES = {
    "caption_contract_source": "scripts/offline_arb/caption_sidecar_contract.py",
    "training_package_source": "training/__init__.py",
    "dataline_package_source": "training/dataline/__init__.py",
    "offline_package_source": "training/dataline_offline/__init__.py",
    "dataset_source": "training/dataline_offline/offline_dataset.py",
    "planner_source": "training/dataline_offline/planner_v1.py",
    "sidecar_runtime_source": "training/dataline_offline/caption_sidecar.py",
    "complete_dataset_source": "training/dataline/complete_dataset.py",
    "dataline_config_source": "training/dataline/config.py",
    "image_fetch_source": "training/dataline/image_fetch.py",
    "manifest_index_source": "training/dataline/manifest_index.py",
    "multi_ar_source": "training/dataline/multi_ar.py",
    "native_safety_source": "training/dataline/native_safety.py",
    "pil_safety_source": "training/dataline/pil_safety.py",
    "caption_build_source": "training/dataline/caption_build.py",
    "sampler_source": "training/dataline/sampler.py",
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
        *RUNTIME_MODULE_FILES,
    }
)
EMPTY_RUNTIME_ARTIFACT_KEYS = frozenset(
    {"training_package_source", "dataline_package_source"}
)
RUNTIME_NAMESPACE_PACKAGES = {
    "scripts": "scripts",
    "scripts.offline_arb": "scripts/offline_arb",
}
SHA_RE = re.compile(r"[0-9a-f]{64}")
RANK_FILE_RE = re.compile(r"io_profile\.rank(?P<rank>[0-9]{4})\.jsonl")
STEP_LINE_RE = re.compile(
    r"W64_RAW_STEP_RETIRED rank=(?P<rank>[0-9]+) step=(?P<step>[0-9]+) "
    r"items=(?P<items>[0-9]+) raw_bytes=(?P<raw_bytes>[0-9]+) "
    r"wait_s=(?P<wait_s>[0-9]+\.[0-9]{9}) "
    r"scheduled_until=(?P<scheduled>[0-9]+)"
)
PASS_LINE_RE = re.compile(
    r"W64_RAW_RANK_PASS rank=(?P<rank>[0-9]+) start_step=(?P<start>[0-9]+) "
    r"stop_step=(?P<stop>[0-9]+) retired_steps=(?P<retired>[0-9]+) "
    r"per_rank_batch_size=(?P<bs>[0-9]+) raw_items=(?P<items>[0-9]+) "
    r"raw_bytes=(?P<raw_bytes>[0-9]+) wait_total_s=(?P<wait_total>[0-9]+\.[0-9]{9}) "
    r"wait_max_s=(?P<wait_max>[0-9]+\.[0-9]{9}) "
    r"elapsed_s=(?P<elapsed>[0-9]+\.[0-9]{9}) decode_calls=(?P<decode>[0-9]+) "
    r"telemetry_windows=(?P<telemetry_windows>[0-9]+) "
    r"gpic_source_consistency_mode=(?P<gpic_mode>strict_path|fd_stable) "
    r"gpic_threads_per_node=(?P<gpic_threads>[0-9]+)"
)


def sha256_file(path: Path) -> str:
    before = path.lstat()
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError(f"snapshot source is not a regular non-symlink: {path}")
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        opened = os.fstat(handle.fileno())
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != identity:
            raise ValueError(f"snapshot source changed while opening: {path}")
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
        closed = os.fstat(handle.fileno())
    after = path.lstat()
    for value in (closed, after):
        if (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        ) != identity:
            raise ValueError(f"snapshot source changed while hashing: {path}")
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


def require_regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} missing/non-regular/symlink: {path}")


def require_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} missing/non-directory/symlink: {path}")


def read_object_snapshot(path: Path, label: str) -> tuple[dict[str, Any], str]:
    require_regular(path, label)
    before = path.lstat()
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    try:
        with path.open("rb", buffering=0) as handle:
            opened = os.fstat(handle.fileno())
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            ) != identity:
                raise ValueError(f"{label} changed while opening: {path}")
            payload = handle.read()
            closed = os.fstat(handle.fileno())
        after = path.lstat()
        for metadata in (closed, after):
            if (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            ) != identity:
                raise ValueError(f"{label} changed while reading: {path}")
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    return read_object_snapshot(path, label)[0]


def strict_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value}")
    return value


def number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        qualifier = "positive finite" if positive else "nonnegative finite"
        raise ValueError(f"{label} must be {qualifier}, got {value!r}")
    return result


def exact_json_equal(actual: Any, expected: Any) -> bool:
    """JSON equality that never treats bool as integer and recurses into containers."""

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


def validated_jobtag(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or value in {"", ".", ".."}
        or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None
        or Path(value).name != value
    ):
        raise ValueError(f"{label} must be a safe non-dot basename, got {value!r}")
    return value


def validated_analysis_output(args: argparse.Namespace) -> tuple[Path, str]:
    source_jobtag = validated_jobtag(args.source_jobtag, "explicit source jobtag")
    run_id = validated_jobtag(args.run_id, "run id")
    if not args.output:
        raise ValueError("W64 analyzer requires an explicit unique output path")
    supplied = Path(args.output)
    if not supplied.is_absolute():
        raise ValueError("W64 analysis output path must be absolute")
    parent = supplied.parent.resolve(strict=True)
    require_directory(parent, "W64 analysis output parent")
    require_directory(parent.parent, "W64 analysis run root")
    prefix = f"w64_raw_supply_analysis.{source_jobtag}."
    if (
        parent.name != source_jobtag
        or parent.parent.name != run_id
        or supplied != parent / supplied.name
        or not supplied.name.startswith(prefix)
        or not supplied.name.endswith(".json")
        or Path(supplied.name).name != supplied.name
    ):
        raise ValueError("W64 analysis output path is non-canonical/unsafe")
    analysis_jobtag = supplied.name[len(prefix) : -len(".json")]
    validated_jobtag(analysis_jobtag, "analysis output jobtag")
    return supplied, analysis_jobtag


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def artifact_matches(
    path: Path,
    entry: Any,
    label: str,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(entry, dict):
        raise ValueError(f"{label} artifact declaration is not an object")
    require_regular(path, label)
    size = strict_int(
        entry.get("bytes"),
        f"{label}.bytes",
        minimum=0 if allow_empty else 1,
    )
    if path.stat().st_size != size:
        raise ValueError(f"{label} byte-size mismatch")
    expected = entry.get("sha256")
    if not isinstance(expected, str) or SHA_RE.fullmatch(expected) is None:
        raise ValueError(f"{label} SHA declaration is malformed")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch")
    return actual


def frozen_artifact_matches(
    path: Path,
    entry: Any,
    label: str,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(entry, dict):
        raise ValueError(f"{label} freeze declaration is not an object")
    require_regular(path, label)
    canonical = path.resolve(strict=True)
    if entry.get("path") != str(canonical):
        raise ValueError(f"{label} frozen path mismatch")
    return artifact_matches(path, entry, label, allow_empty=allow_empty)


def validate_frozen_gate_files(
    freeze_path: Path,
    *,
    repo: Path,
    runner: Path,
    capture_yaml: Path,
    analyze_yaml: Path,
) -> dict[str, Any]:
    freeze, freeze_snapshot_sha = read_object_snapshot(freeze_path, "fresh freeze")
    if set(freeze) != FREEZE_TOP_LEVEL_KEYS or freeze.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("fresh freeze schema mismatch")
    if not isinstance(freeze.get("frozen_at"), str) or not isinstance(
        freeze.get("git_head_at_freeze"), str
    ):
        raise ValueError("fresh freeze timestamp/git identity is malformed")
    release = canonical_sha256({key: freeze[key] for key in FREEZE_RELEASE_KEYS})
    training_sources = freeze.get("training_sources")
    files = training_sources.get("files") if isinstance(training_sources, dict) else None
    if not isinstance(files, dict):
        raise ValueError("freeze training_sources.files is malformed")
    discovery = training_sources.get("discovery")
    explicit = discovery.get("explicit") if isinstance(discovery, dict) else None
    if not isinstance(explicit, list) or not set(FROZEN_GATE_FILES).issubset(explicit):
        raise ValueError("fresh freeze explicit source set does not include every W64 gate file")
    if training_sources.get("root") != str(repo):
        raise ValueError("fresh freeze training source root does not match canonical repository")
    live_paths = {
        FROZEN_GATE_FILES[0]: runner,
        FROZEN_GATE_FILES[1]: capture_yaml,
        FROZEN_GATE_FILES[2]: Path(__file__).resolve(strict=True),
        FROZEN_GATE_FILES[3]: analyze_yaml,
    }
    artifacts: dict[str, Any] = {}
    for relative, supplied in live_paths.items():
        canonical = (repo / relative).resolve(strict=True)
        if supplied.resolve(strict=True) != canonical:
            raise ValueError(f"gate path is not canonical for {relative}: {supplied}")
        entry = files.get(relative)
        if not isinstance(entry, dict):
            raise ValueError(f"fresh freeze does not bind gate file: {relative}")
        expected_path = str(canonical)
        if entry.get("path") != expected_path:
            raise ValueError(f"freeze path mismatch for {relative}")
        actual_sha = artifact_matches(canonical, entry, f"frozen {relative}")
        artifacts[relative] = {
            "path": expected_path,
            "sha256": actual_sha,
            "bytes": canonical.stat().st_size,
        }
    return {
        "freeze": freeze,
        "freeze_sha256": freeze_snapshot_sha,
        "release_digest": release,
        "gate_files": artifacts,
    }


def validate_full_freeze_verify(
    path: Path,
    *,
    frozen: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    value, value_sha = read_object_snapshot(path, label)
    expected_keys = {
        "status",
        "mode",
        "frozen_at",
        "release_digest",
        "freeze_sha256",
        "freeze_file",
        "snapshot_fingerprints_sha256",
        "header_sha256",
        "header_file_sha256",
        "success_file_sha256",
        "catalog_hash",
        "catalog_header_sha256",
        "file_count",
        "total_bytes",
        "sidecar_release_digest",
        "training_sources",
        "catalog_files",
        "plan_blocks",
    }
    if set(value) != expected_keys:
        raise ValueError(f"{label} key set drift")
    freeze_value = frozen["freeze"]
    if (
        value.get("status") != "PASS"
        or value.get("mode") != "verify-freeze"
        or value.get("frozen_at") != freeze_value["frozen_at"]
        or value.get("release_digest") != frozen["release_digest"]
        or value.get("freeze_sha256") != frozen["freeze_sha256"]
    ):
        raise ValueError(f"{label} identity/status mismatch")
    for key in ("snapshot_fingerprints_sha256", "header_sha256", "header_file_sha256", "success_file_sha256", "catalog_hash", "catalog_header_sha256", "sidecar_release_digest"):
        if not isinstance(value.get(key), str) or SHA_RE.fullmatch(value[key]) is None:
            raise ValueError(f"{label} {key} is malformed")
    freeze_file = value.get("freeze_file")
    if not isinstance(freeze_file, dict):
        raise ValueError(f"{label} freeze_file is malformed")
    frozen_artifact_matches(
        Path(freeze_file.get("path", "")),
        freeze_file,
        f"{label} freeze file",
    )
    expected_counts = {
        "training_sources": len(freeze_value["training_sources"]["files"]),
        "catalog_files": len(freeze_value["catalog"]["files"]),
        "plan_blocks": len(freeze_value["plan_blocks"]["files"]),
        "file_count": freeze_value["caption_sidecar"]["file_count"],
        "total_bytes": freeze_value["caption_sidecar"]["total_bytes"],
    }
    for key, wanted in expected_counts.items():
        if strict_int(value.get(key), f"{label}.{key}", minimum=1) != wanted:
            raise ValueError(f"{label} {key} mismatch")
    return {"path": str(path), "sha256": value_sha, "value": value}


def validate_runtime_contract(
    path: Path,
    *,
    entry: Any,
    aggregate_sha256: str,
    args: argparse.Namespace,
    frozen: dict[str, Any],
    jobtag: str,
    nonce: str,
) -> dict[str, Any]:
    contract_sha = artifact_matches(path, entry, "W64 runtime contract")
    if contract_sha != aggregate_sha256:
        raise ValueError("aggregate runtime-contract SHA does not close")
    value, value_sha = read_object_snapshot(path, "W64 runtime contract")
    if value_sha != contract_sha:
        raise ValueError("runtime contract changed between artifact binding and parse")
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
    if set(value) != expected_keys or value.get("schema_version") != RUNTIME_CONTRACT_SCHEMA:
        raise ValueError("runtime contract schema/key set drift")
    mode = value.get("gpic_source_consistency_mode")
    if mode not in GPIC_CONSISTENCY_MODES:
        raise ValueError("runtime contract GPIC consistency mode is unsupported")
    if type(value.get("gpic_threads_per_node")) is not int or value["gpic_threads_per_node"] != 16:
        raise ValueError("runtime contract GPIC threads-per-node must be 16")
    if (
        value.get("run_id") != args.run_id
        or value.get("jobtag") != jobtag
        or value.get("attempt_nonce") != nonce
        or value.get("freeze_release_digest") != frozen["release_digest"]
        or value.get("freeze_sha256") != frozen["freeze_sha256"]
    ):
        raise ValueError("runtime contract attempt/freeze identity mismatch")
    repo = Path(args.repo).resolve(strict=True)
    expected_namespaces = {
        name: str((repo / relative).resolve(strict=True))
        for name, relative in RUNTIME_NAMESPACE_PACKAGES.items()
    }
    if value.get("namespace_packages") != expected_namespaces:
        raise ValueError("runtime contract namespace-package closure drift")
    for module_name, expected_root in expected_namespaces.items():
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
            raise ValueError(f"runtime namespace package closure drift: {module_name}")
    result_root = path.parent
    expected_paths = {
        "runner": Path(args.runner),
        "capture_yaml": Path(args.capture_yaml),
        "config": Path(args.config),
        "freeze_manifest": Path(args.freeze),
        "preflight_verify": result_root / f"freezepreflight.{jobtag}.json",
        "plan_header": Path(args.plan) / "header.json",
        "plan_checksums": Path(args.plan) / "checksums.json",
        "catalog_header": Path(args.catalog_root) / "header.json",
        "animefreq_header": Path(args.anime_freq_root) / "header.json",
        "sidecar_header": Path(args.sidecar_root) / "header.json",
        "sampling": Path(args.sampling),
        "freeze_builder_source": repo / "scripts/offline_arb/make_freeze_manifest.py",
    }
    expected_paths.update(
        {key: repo / relative for key, relative in RUNTIME_MODULE_FILES.items()}
    )
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != RUNTIME_ARTIFACT_KEYS:
        raise ValueError("runtime contract artifact set drift")
    artifact_evidence: dict[str, dict[str, Any]] = {}
    frozen_sources = frozen["freeze"]["training_sources"]["files"]
    for key, expected_path in expected_paths.items():
        canonical = expected_path.resolve(strict=True)
        live_sha = frozen_artifact_matches(
            canonical,
            artifacts.get(key),
            f"runtime contract artifact {key}",
            allow_empty=key in EMPTY_RUNTIME_ARTIFACT_KEYS,
        )
        artifact_evidence[key] = {
            "path": str(canonical),
            "bytes": canonical.stat().st_size,
            "sha256": live_sha,
        }
        try:
            relative = canonical.relative_to(repo).as_posix()
        except ValueError:
            continue
        frozen_entry = frozen_sources.get(relative)
        if not isinstance(frozen_entry, dict) or any(
            frozen_entry.get(field) != artifacts[key].get(field)
            for field in ("path", "bytes", "sha256")
        ):
            raise ValueError(f"runtime contract/freeze source mismatch: {relative}")
    if artifacts["preflight_verify"]["sha256"] != value.get("preflight_verify_sha256"):
        raise ValueError("runtime contract preflight verification SHA mismatch")
    markers = value.get("fresh_markers")
    if not isinstance(markers, dict) or set(markers) != {str(node) for node in range(NODES)}:
        raise ValueError("runtime contract fresh-marker set drift")
    marker_evidence = []
    marker_times = []
    for node in range(NODES):
        marker_path = result_root / f"freezepreflight.{jobtag}.n{node}.fresh"
        marker_sha = frozen_artifact_matches(
            marker_path,
            markers[str(node)],
            f"runtime contract fresh marker node {node}",
        )
        fields = marker_path.read_text(encoding="ascii").strip().split("|")
        if (
            len(fields) != 6
            or fields[:3] != ["READY", jobtag, str(node)]
            or fields[4] != f"{jobtag}-{node}"
            or not fields[3].isdigit()
            or SHA_RE.fullmatch(fields[5]) is None
        ):
            raise ValueError(f"runtime contract fresh marker node {node} malformed")
        marker_times.append(int(fields[3]))
        marker_evidence.append({"path": str(marker_path), "sha256": marker_sha})
    if max(marker_times) - min(marker_times) >= 2700:
        raise ValueError("runtime contract fresh-marker barrier span is too large")
    config_path = Path(args.config)
    config_before = config_path.lstat()
    config_identity = (
        config_before.st_dev,
        config_before.st_ino,
        config_before.st_size,
        config_before.st_mtime_ns,
        config_before.st_ctime_ns,
    )
    with config_path.open("rb", buffering=0) as handle:
        config_opened = os.fstat(handle.fileno())
        if (
            config_opened.st_dev,
            config_opened.st_ino,
            config_opened.st_size,
            config_opened.st_mtime_ns,
            config_opened.st_ctime_ns,
        ) != config_identity:
            raise ValueError("runtime config changed while opening")
        config_payload = handle.read()
        config_closed = os.fstat(handle.fileno())
    config_after = config_path.lstat()
    for metadata in (config_closed, config_after):
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ) != config_identity:
            raise ValueError("runtime config changed while reading")
    if hashlib.sha256(config_payload).hexdigest() != artifacts["config"]["sha256"]:
        raise ValueError("runtime config snapshot differs from contract")
    configured_dataset = OmegaConf.create(config_payload.decode("utf-8")).dataset
    configured = configured_dataset.gpic_source_consistency_mode
    if str(configured) != mode:
        raise ValueError("runtime contract/config GPIC consistency mode mismatch")
    configured_threads = configured_dataset.gpic_threads_per_node
    if int(configured_threads) != value["gpic_threads_per_node"]:
        raise ValueError("runtime contract/config GPIC threads-per-node mismatch")
    return {
        "path": str(path),
        "sha256": contract_sha,
        "value": value,
        "mode": mode,
        "artifacts": artifact_evidence,
        "fresh_markers": marker_evidence,
    }


def validate_preflight(
    result_root: Path,
    *,
    jobtag: str,
    nonce: str,
    freeze_sha256: str,
    release_digest: str,
    runner_sha256: str,
    yaml_sha256: str,
    runtime_contract: dict[str, Any],
) -> dict[str, Any]:
    pass_path = result_root / f"freezepreflight.{jobtag}.pass"
    json_path = result_root / f"freezepreflight.{jobtag}.json"
    require_regular(pass_path, "attempt freeze PASS")
    require_regular(json_path, "attempt freeze JSON")
    fields = pass_path.read_text(encoding="ascii").strip().split("|")
    if len(fields) != 16 or fields[:3] != ["PASS", jobtag, nonce]:
        raise ValueError("attempt freeze PASS identity/field count mismatch")
    timestamp = fields[3]
    if not timestamp.isdigit():
        raise ValueError("attempt freeze PASS timestamp is malformed")
    pass_timestamp = int(timestamp)
    for value in fields[4:15]:
        if SHA_RE.fullmatch(value) is None:
            raise ValueError("attempt freeze PASS contains malformed SHA")
    (
        header_sha,
        header_file_sha,
        success_file_sha,
        catalog_header_sha,
        sidecar_release,
        pass_release,
        json_sha,
        pass_freeze_sha,
        pass_runner_sha,
        pass_yaml_sha,
        pass_contract_sha,
        pass_gpic_mode,
    ) = fields[4:]
    if (
        pass_release != release_digest
        or pass_freeze_sha != freeze_sha256
        or pass_runner_sha != runner_sha256
        or pass_yaml_sha != yaml_sha256
        or pass_contract_sha != runtime_contract["sha256"]
        or pass_gpic_mode != runtime_contract["mode"]
        or sha256_file(json_path) != json_sha
    ):
        raise ValueError("attempt freeze PASS/live artifact binding mismatch")
    value, parsed_json_sha = read_object_snapshot(json_path, "attempt freeze JSON")
    if parsed_json_sha != json_sha:
        raise ValueError("attempt freeze JSON changed between PASS binding and parse")
    expected = {
        "status": "PASS",
        "mode": "verify-freeze",
        "header_sha256": header_sha,
        "header_file_sha256": header_file_sha,
        "success_file_sha256": success_file_sha,
        "catalog_header_sha256": catalog_header_sha,
        "sidecar_release_digest": sidecar_release,
        "release_digest": release_digest,
        "freeze_sha256": freeze_sha256,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError(f"attempt freeze JSON {key} mismatch")
    fresh_markers = runtime_contract["fresh_markers"]
    controls = []
    for node in range(NODES):
        marker_path = Path(fresh_markers[node]["path"])
        marker_fields = marker_path.read_text(encoding="ascii").strip().split("|")
        if len(marker_fields) != 6 or marker_fields[:3] != ["READY", jobtag, str(node)]:
            raise ValueError(f"node {node} fresh preflight marker identity mismatch")
        marker_timestamp = marker_fields[3]
        if (
            not marker_timestamp.isdigit()
            or marker_fields[4] != f"{jobtag}-{node}"
            or int(marker_timestamp) > pass_timestamp
            or SHA_RE.fullmatch(marker_fields[5]) is None
        ):
            raise ValueError(f"node {node} fresh preflight marker timing/host mismatch")
        control_path = result_root / f"sidecarcontrol.{jobtag}.n{node}.json"
        control = read_object(control_path, f"node {node} sidecar control")
        for key, wanted in (
            ("status", "PASS"),
            ("mode", "verify-sidecar-control"),
            ("header_sha256", header_sha),
            ("header_file_sha256", header_file_sha),
            ("success_file_sha256", success_file_sha),
            ("catalog_header_sha256", catalog_header_sha),
            ("release_digest", sidecar_release),
        ):
            if control.get(key) != wanted:
                raise ValueError(f"node {node} sidecar control {key} mismatch")
        controls.append({"path": str(control_path), "sha256": sha256_file(control_path)})
    return {
        "sidecar_header_sha256": header_sha,
        "sidecar_release_digest": sidecar_release,
        "preflight_json_sha256": json_sha,
        "pass": {"path": str(pass_path), "sha256": sha256_file(pass_path)},
        "json": {"path": str(json_path), "sha256": json_sha},
        "fresh_markers": fresh_markers,
        "controls": controls,
    }


def validate_aggregate(
    aggregate_path: Path,
    *,
    run_id: str,
    freeze: dict[str, Any],
    runner_sha256: str,
    yaml_sha256: str,
) -> dict[str, Any]:
    aggregate, aggregate_sha = read_object_snapshot(aggregate_path, "W64 aggregate")
    if aggregate_path.parent.parent.name != run_id:
        raise ValueError("aggregate parent is not bound to run-id")
    if aggregate.get("schema_version") != AGGREGATE_SCHEMA or aggregate.get("status") != "PASS":
        raise ValueError("W64 aggregate schema/status is not PASS")
    expected_keys = {
        "schema_version",
        "status",
        "jobtag",
        "nonce",
        "freeze_release_digest",
        "freeze_sha256",
        "sidecar_header_sha256",
        "sidecar_release_digest",
        "runner_sha256",
        "yaml_sha256",
        "runtime_contract",
        "runtime_contract_sha256",
        "gpic_source_consistency_mode",
        "gpic_threads_per_node",
        "preflight_verify",
        "postflight_verify",
        "closure_markers",
        "preflight_pass",
        "sidecar_controls",
        "preservation_markers",
        "expected_nodes",
        "expected_ranks",
        "start_step",
        "stop_step",
        "nodes",
    }
    if set(aggregate) != expected_keys:
        raise ValueError("W64 aggregate key set drift")
    jobtag = validated_jobtag(aggregate.get("jobtag"), "aggregate jobtag")
    nonce = aggregate.get("nonce")
    if not isinstance(nonce, str) or SHA_RE.fullmatch(nonce) is None:
        raise ValueError("aggregate nonce is malformed")
    if aggregate_path.name != f"telemetry.{jobtag}.aggregate.json":
        raise ValueError("aggregate filename/jobtag mismatch")
    expected_top = {
        "freeze_release_digest": freeze["release_digest"],
        "freeze_sha256": freeze["freeze_sha256"],
        "runner_sha256": runner_sha256,
        "yaml_sha256": yaml_sha256,
        "expected_nodes": NODES,
        "expected_ranks": WORLD_SIZE,
        "start_step": START_STEP,
        "stop_step": STOP_STEP,
    }
    for key, wanted in expected_top.items():
        if not exact_json_equal(aggregate.get(key), wanted):
            raise ValueError(f"aggregate {key} mismatch")
    sidecar_header = aggregate.get("sidecar_header_sha256")
    sidecar_release = aggregate.get("sidecar_release_digest")
    if SHA_RE.fullmatch(str(sidecar_header or "")) is None or SHA_RE.fullmatch(
        str(sidecar_release or "")
    ) is None:
        raise ValueError("aggregate sidecar identity is malformed")
    runtime_contract_sha = aggregate.get("runtime_contract_sha256")
    gpic_mode = aggregate.get("gpic_source_consistency_mode")
    gpic_threads = aggregate.get("gpic_threads_per_node")
    if not isinstance(runtime_contract_sha, str) or SHA_RE.fullmatch(runtime_contract_sha) is None:
        raise ValueError("aggregate runtime-contract SHA is malformed")
    if gpic_mode not in GPIC_CONSISTENCY_MODES:
        raise ValueError("aggregate GPIC consistency mode is unsupported")
    if type(gpic_threads) is not int or gpic_threads != 16:
        raise ValueError("aggregate GPIC threads-per-node identity drift")
    for key in ("runtime_contract", "preflight_verify", "postflight_verify", "preflight_pass"):
        entry = aggregate.get(key)
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError(f"aggregate {key} artifact declaration is malformed")
    closure_markers = aggregate.get("closure_markers")
    if not isinstance(closure_markers, list) or len(closure_markers) != NODES:
        raise ValueError("aggregate final closure-marker set is malformed")
    if any(
        not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}
        for entry in closure_markers
    ):
        raise ValueError("aggregate final closure-marker artifact is malformed")
    for key in ("sidecar_controls", "preservation_markers"):
        entries = aggregate.get(key)
        if not isinstance(entries, list) or len(entries) != NODES or any(
            not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}
            for entry in entries
        ):
            raise ValueError(f"aggregate {key} artifact set is malformed")
    nodes = aggregate.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != NODES:
        raise ValueError("aggregate must declare exactly eight nodes")
    return {
        "value": aggregate,
        "jobtag": jobtag,
        "nonce": nonce,
        "sidecar_header_sha256": sidecar_header,
        "sidecar_release_digest": sidecar_release,
        "runtime_contract_sha256": runtime_contract_sha,
        "gpic_source_consistency_mode": gpic_mode,
        "gpic_threads_per_node": gpic_threads,
        "aggregate_sha256": aggregate_sha,
    }


def parse_runner_log(
    path: Path,
    node_rank: int,
    gpic_source_consistency_mode: str,
) -> dict[int, dict[str, Any]]:
    require_regular(path, f"node {node_rank} runner log")
    steps: dict[int, list[dict[str, Any]]] = {
        rank: [] for rank in range(node_rank * LOCAL_WORLD_SIZE, (node_rank + 1) * LOCAL_WORLD_SIZE)
    }
    passes: dict[int, list[dict[str, Any]]] = {rank: [] for rank in steps}
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            step_match = STEP_LINE_RE.fullmatch(line)
            if step_match is not None:
                rank = int(step_match.group("rank"))
                if rank not in steps:
                    raise ValueError(f"node {node_rank} log contains foreign step rank {rank}")
                steps[rank].append(
                    {
                        "line": line_number,
                        "step": int(step_match.group("step")),
                        "items": int(step_match.group("items")),
                        "raw_bytes": int(step_match.group("raw_bytes")),
                        "wait_s": float(step_match.group("wait_s")),
                        "scheduled_until": int(step_match.group("scheduled")),
                    }
                )
                continue
            pass_match = PASS_LINE_RE.fullmatch(line)
            if pass_match is not None:
                rank = int(pass_match.group("rank"))
                if rank not in passes:
                    raise ValueError(f"node {node_rank} log contains foreign PASS rank {rank}")
                passes[rank].append(
                    {
                        "line": line_number,
                        "start_step": int(pass_match.group("start")),
                        "stop_step": int(pass_match.group("stop")),
                        "retired_steps": int(pass_match.group("retired")),
                        "per_rank_bs": int(pass_match.group("bs")),
                        "raw_items": int(pass_match.group("items")),
                        "raw_bytes": int(pass_match.group("raw_bytes")),
                        "wait_total_s": float(pass_match.group("wait_total")),
                        "wait_max_s": float(pass_match.group("wait_max")),
                        "elapsed_s": float(pass_match.group("elapsed")),
                        "decode_calls": int(pass_match.group("decode")),
                        "telemetry_windows": int(pass_match.group("telemetry_windows")),
                        "gpic_source_consistency_mode": pass_match.group("gpic_mode"),
                        "gpic_threads_per_node": int(pass_match.group("gpic_threads")),
                    }
                )
                continue
            if "W64_RAW_STEP_RETIRED" in line or "W64_RAW_RANK_PASS" in line:
                raise ValueError(
                    f"node {node_rank} malformed/interleaved runner sentinel line {line_number}"
                )
    result: dict[int, dict[str, Any]] = {}
    for rank in steps:
        records = steps[rank]
        if len(records) != STOP_STEP or [item["step"] for item in records] != list(
            range(START_STEP, STOP_STEP)
        ):
            raise ValueError(f"rank {rank} runner steps are not exact 0..191")
        for record in records:
            step = record["step"]
            expected_scheduled = min(
                STOP_STEP,
                PREFETCH_STEPS * (((step + 1) // PREFETCH_STEPS) + 2),
            )
            if (
                record["items"] != PER_RANK_BS
                or record["raw_bytes"] <= 0
                or record["scheduled_until"] != expected_scheduled
            ):
                raise ValueError(f"rank {rank} runner step {step} contract mismatch")
        if len(passes[rank]) != 1:
            raise ValueError(f"rank {rank} must have exactly one runner PASS")
        summary = passes[rank][0]
        waits = [item["wait_s"] for item in records]
        raw_bytes = sum(item["raw_bytes"] for item in records)
        if (
            summary["start_step"] != START_STEP
            or summary["stop_step"] != STOP_STEP
            or summary["retired_steps"] != STOP_STEP
            or summary["per_rank_bs"] != PER_RANK_BS
            or summary["raw_items"] != STOP_STEP * PER_RANK_BS
            or summary["raw_bytes"] != raw_bytes
            or summary["decode_calls"] != 0
            or summary["telemetry_windows"] <= 0
            or summary["gpic_source_consistency_mode"]
            != gpic_source_consistency_mode
            or summary["gpic_threads_per_node"] != 16
            or summary["elapsed_s"] <= 0
            or summary["elapsed_s"] + 0.05 < sum(waits)
            or abs(summary["wait_total_s"] - sum(waits)) > 0.01
            or abs(summary["wait_max_s"] - max(waits)) > 0.01
        ):
            raise ValueError(f"rank {rank} runner PASS accounting mismatch")
        result[rank] = {"steps": records, "summary": summary}
    return result


def validate_ready_snapshot(value: Any, label: str, domains: set[str]) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
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
    detail: Any,
    *,
    domain: str,
    n: int,
    image_s: float,
    label: str,
    gpic_source_consistency_mode: str,
) -> None:
    if not isinstance(detail, dict):
        raise ValueError(f"{label}.image_detail must be an object")
    kind = detail.get("kind")
    if domain == "gpic":
        if kind != "gpic_in_memory" or detail.get("files") != n or detail.get("workers") != 2:
            raise ValueError(f"{label} GPIC detail mismatch")
        if detail.get("gpic_source_consistency_mode") != gpic_source_consistency_mode:
            raise ValueError(f"{label} GPIC consistency mode mismatch")
        fstat_calls = strict_int(detail.get("fstat_calls"), f"{label}.fstat_calls", minimum=0)
        path_stat_calls = strict_int(
            detail.get("path_stat_calls"), f"{label}.path_stat_calls", minimum=0
        )
        expected_path_stat_calls = 2 * n if gpic_source_consistency_mode == "strict_path" else 0
        if fstat_calls != 2 * n or path_stat_calls != expected_path_stat_calls:
            raise ValueError(f"{label} GPIC consistency syscall accounting mismatch")
        unique = strict_int(detail.get("unique_files"), f"{label}.unique_files", minimum=1)
        if unique > n:
            raise ValueError(f"{label} GPIC unique file count exceeds n")
        strict_int(detail.get("bytes"), f"{label}.bytes", minimum=1)
        required = ("open", "fstat", "path_stat", "read", "file_total")
        maximum_key = "max_file_s"
    elif domain in {"monet_real", "synthetic"}:
        if kind != "coalesced_tar":
            raise ValueError(f"{label} tar detail kind mismatch")
        unique_tars = strict_int(detail.get("unique_tars"), f"{label}.unique_tars", minimum=1)
        locator_requests = strict_int(
            detail.get("locator_requests"), f"{label}.locator_requests", minimum=1
        )
        unique_locators = strict_int(
            detail.get("unique_locators"), f"{label}.unique_locators", minimum=1
        )
        runs = strict_int(detail.get("runs"), f"{label}.runs", minimum=1)
        requested = strict_int(detail.get("requested_bytes"), f"{label}.requested", minimum=1)
        member = strict_int(detail.get("member_bytes"), f"{label}.member", minimum=1)
        actual = strict_int(detail.get("actual_read_bytes"), f"{label}.actual", minimum=1)
        if (
            locator_requests != n
            or unique_tars > unique_locators
            or unique_locators > locator_requests
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
        raise ValueError(f"{label} unsupported domain {domain!r}")
    thread_sum = detail.get("thread_sum_s")
    operation_max = detail.get("operation_max_s")
    if not isinstance(thread_sum, dict) or not isinstance(operation_max, dict):
        raise ValueError(f"{label} detailed timing maps are malformed")
    for key in required:
        total = number(thread_sum.get(key), f"{label}.thread_sum.{key}")
        maximum = number(operation_max.get(key), f"{label}.operation_max.{key}")
        if maximum > total + 1e-9:
            raise ValueError(f"{label} operation max exceeds thread sum")
    wall_s = number(detail.get("wall_s"), f"{label}.wall_s", positive=True)
    number(detail.get(maximum_key), f"{label}.{maximum_key}")
    number(detail.get("effective_concurrency"), f"{label}.effective_concurrency")
    if wall_s > image_s + 0.01:
        raise ValueError(f"{label} image detail wall exceeds outer image time")


def load_rank_telemetry(
    path: Path,
    rank: int,
    *,
    plan_id: str,
    plan_hash: str,
    catalog_hash: str,
    jobtag: str,
    nonce: str,
    host: str,
    sidecar_header_sha256: str,
    sidecar_release_digest: str,
    domains: set[str],
    runner_steps: list[dict[str, Any]],
    gpic_source_consistency_mode: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith("\n") or not raw.strip():
                raise ValueError(f"rank {rank} telemetry line {line_number} is blank/unterminated")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"rank {rank} malformed JSONL line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"rank {rank} telemetry line {line_number} is not an object")
            if value.get("schema_version") != TELEMETRY_SCHEMA:
                raise ValueError(f"rank {rank} telemetry schema drift")
            sequence = strict_int(
                value.get("sequence"), f"rank {rank} telemetry sequence", minimum=0
            )
            record_rank = strict_int(
                value.get("rank"), f"rank {rank} telemetry record rank", minimum=0
            )
            if sequence != line_number - 1:
                raise ValueError(f"rank {rank} telemetry sequence gap")
            if value.get("record_type") not in ALLOWED_RECORD_TYPES or record_rank != rank:
                raise ValueError(f"rank {rank} telemetry record identity/type drift")
            records.append(value)
    if not records:
        raise ValueError(f"rank {rank} telemetry is empty")
    identities = [value for value in records if value["record_type"] == "identity"]
    closures = [value for value in records if value["record_type"] == "closure"]
    if len(identities) != 1 or identities[0] is not records[0]:
        raise ValueError(f"rank {rank} identity is not unique first record")
    if len(closures) != 1 or closures[0] is not records[-1]:
        raise ValueError(f"rank {rank} closure is not unique last record")
    expected_identity = {
        "world_size": WORLD_SIZE,
        "local_rank": rank % LOCAL_WORLD_SIZE,
        "local_world_size": LOCAL_WORLD_SIZE,
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "catalog_hash": catalog_hash,
        "sidecar_header_sha256": sidecar_header_sha256,
        "sidecar_release_digest": sidecar_release_digest,
        "attempt_jobtag": jobtag,
        "attempt_nonce": nonce,
        "attempt_host": host,
        "telemetry_root": (
            f"/local/evan/offline_arb_w64_raw/{jobtag}/{nonce}/node{rank // LOCAL_WORLD_SIZE}"
        ),
        "telemetry_path": (
            f"/local/evan/offline_arb_w64_raw/{jobtag}/{nonce}/node{rank // LOCAL_WORLD_SIZE}/"
            f"io_profile.rank{rank:04d}.jsonl"
        ),
        "prefetch_steps": PREFETCH_STEPS,
        "gpic_threads_per_node": 16,
        "gpic_workers_per_rank": 2,
        "tar_workers_per_rank": 8,
        "gpic_payload_in_memory": True,
        "gpic_source_consistency_mode": gpic_source_consistency_mode,
    }
    identity = identities[0]
    for key, wanted in expected_identity.items():
        if not exact_json_equal(identity.get(key), wanted):
            raise ValueError(f"rank {rank} identity {key} mismatch")
    strict_int(identity.get("pid"), f"rank {rank} pid", minimum=1)
    closure = closures[0]
    if (
        strict_int(
            closure.get("next_step"), f"rank {rank} closure next_step", minimum=0
        )
        != STOP_STEP
    ):
        raise ValueError(f"rank {rank} closure next_step mismatch")
    before = len(records) - 1
    if (
        strict_int(closure.get("records_before_closure"), "writer records", minimum=0) != before
        or strict_int(closure.get("records_enqueued_before_closure"), "producer records", minimum=0)
        != before
        or closure.get("telemetry_fatal") is not False
        or strict_int(closure.get("dropped_emits"), "dropped emits", minimum=0) != 0
        or strict_int(closure.get("runtime_errors"), "runtime errors", minimum=0) != 0
        or strict_int(closure.get("rejected_emits_before_closure"), "rejected emits", minimum=0)
        != 0
        or strict_int(
            closure.get("rejected_emits_observed_by_writer"), "writer rejected emits", minimum=0
        )
        != 0
    ):
        raise ValueError(f"rank {rank} closure accounting/fatal mismatch")
    number(closure.get("producer_enqueue_seconds_before_closure"), "producer enqueue seconds")
    number(closure.get("writer_seconds_before_closure"), "writer seconds")

    windows: dict[tuple[int, str], dict[str, Any]] = {}
    states: list[int] = []
    waits: dict[int, dict[str, Any]] = {}
    for value in records[1:-1]:
        record_type = value["record_type"]
        if record_type == "window":
            start = strict_int(value.get("start"), f"rank {rank} window start", minimum=0)
            stop = strict_int(value.get("stop"), f"rank {rank} window stop", minimum=1)
            domain = value.get("domain")
            n = strict_int(value.get("n"), f"rank {rank} window n", minimum=1)
            if domain not in domains or start not in ALL_WINDOW_STARTS or stop != start + PREFETCH_STEPS:
                raise ValueError(f"rank {rank} window identity/range mismatch")
            if value.get("caption_backend") != "sidecar" or value.get("caption_workers") != 16:
                raise ValueError(f"rank {rank} window caption backend/workers mismatch")
            key = (start, str(domain))
            if key in windows:
                raise ValueError(f"rank {rank} duplicate window {key}")
            catalog_s = number(value.get("catalog_s"), f"rank {rank} {key} catalog_s")
            caption_s = number(value.get("caption_s"), f"rank {rank} {key} caption_s")
            image_s = number(value.get("image_s"), f"rank {rank} {key} image_s")
            total_s = number(value.get("total_s"), f"rank {rank} {key} total_s", positive=True)
            if total_s + 1e-6 < catalog_s + caption_s + image_s:
                raise ValueError(f"rank {rank} {key} component timing exceeds total")
            source_counts = value.get("sub_source_counts")
            if not isinstance(source_counts, dict):
                raise ValueError(f"rank {rank} {key} source counts malformed")
            source_total = 0
            for source, count in source_counts.items():
                if not isinstance(source, str):
                    raise ValueError(f"rank {rank} {key} source key malformed")
                source_total += strict_int(count, f"rank {rank} {key} source count", minimum=1)
            if domain in {"monet_real", "gpic", "synthetic"} and source_total != n:
                raise ValueError(f"rank {rank} {key} source counts do not close")
            validate_image_detail(
                value.get("image_detail"),
                domain=str(domain),
                n=n,
                image_s=image_s,
                label=f"rank {rank} {key}",
                gpic_source_consistency_mode=gpic_source_consistency_mode,
            )
            windows[key] = value
        elif record_type == "state":
            consumed = strict_int(value.get("consumed_step"), f"rank {rank} state", minimum=0)
            if strict_int(
                value.get("next_step"), f"rank {rank} state next_step", minimum=1
            ) != consumed + 1:
                raise ValueError(f"rank {rank} state next_step mismatch")
            validate_ready_snapshot(value, f"rank {rank} state {consumed}", domains)
            states.append(consumed)
        elif record_type == "wait":
            step = strict_int(value.get("step"), f"rank {rank} wait step", minimum=0)
            wait_s = number(value.get("condition_wait_s"), f"rank {rank} wait seconds", positive=True)
            if step >= STOP_STEP or step in waits or wait_s <= WAIT_LOG_THRESHOLD_S:
                raise ValueError(f"rank {rank} wait threshold/step/duplicate mismatch")
            strict_int(value.get("condition_wakeups"), f"rank {rank} wakeups", minimum=1)
            expected_items = strict_int(value.get("expected_items"), "wait expected", minimum=1)
            entry_ready = strict_int(value.get("entry_ready_items"), "wait entry", minimum=0)
            exit_ready = strict_int(value.get("exit_ready_items"), "wait exit", minimum=0)
            if expected_items != PER_RANK_BS or entry_ready > PER_RANK_BS or exit_ready != PER_RANK_BS:
                raise ValueError(f"rank {rank} wait item accounting mismatch")
            if wait_s > float(runner_steps[step]["wait_s"]) + 0.05:
                raise ValueError(f"rank {rank} telemetry wait exceeds runner wait")
            validate_ready_snapshot(value.get("entry"), f"rank {rank} wait entry snapshot", domains)
            validate_ready_snapshot(value.get("exit"), f"rank {rank} wait exit snapshot", domains)
            waits[step] = value
    if states != list(range(START_STEP, STOP_STEP)):
        raise ValueError(f"rank {rank} state sequence is not exact 0..191")
    if list(waits) != sorted(waits):
        raise ValueError(f"rank {rank} wait records are not step ordered")
    return {
        "records": len(records),
        "record_counts": dict(sorted(Counter(value["record_type"] for value in records).items())),
        "windows": windows,
        "waits": waits,
    }


class PlanWindowCounts:
    def __init__(self, plan: OfflinePlan):
        self.plan = plan
        self.domains = tuple(str(value) for value in plan.header["domains"])
        totals = [
            sum(int(row[index]) for row in plan.header["N"])
            for index in range(len(self.domains))
        ]
        running = 0
        self.domain_ends: list[int] = []
        for total in totals:
            running += total
            self.domain_ends.append(running)
        if running != int(plan.header["catalog_size"]):
            raise ValueError("plan domain ranges do not close over catalog")
        self.cache: dict[tuple[int, int], dict[str, int]] = {}

    def get(self, start: int, rank: int) -> dict[str, int]:
        key = (start, rank)
        if key in self.cache:
            return self.cache[key]
        if start not in ALL_WINDOW_STARTS:
            raise ValueError(f"invalid gate plan window start {start}")
        counts = {domain: 0 for domain in self.domains}
        for step in range(start, start + PREFETCH_STEPS):
            _bucket, row_ids = self.plan.batch(step, rank)
            for raw in row_ids.tolist():
                row_id = int(raw)
                domain_index = bisect.bisect_right(self.domain_ends, row_id)
                if row_id < 0 or domain_index >= len(self.domains):
                    raise ValueError(f"plan row id outside domain ranges: {row_id}")
                counts[self.domains[domain_index]] += 1
        if sum(counts.values()) != PREFETCH_STEPS * PER_RANK_BS:
            raise ValueError(f"plan counts do not close for rank {rank} window {start}")
        self.cache[key] = counts
        return counts


def add_gate(
    gates: list[dict[str, Any]],
    name: str,
    value: float,
    operator: str,
    threshold: float,
) -> None:
    if operator == "<=":
        passed = value <= threshold
    elif operator == ">=":
        passed = value >= threshold
    else:
        raise AssertionError(operator)
    gates.append(
        {
            "name": name,
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "pass": passed,
        }
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    source_jobtag = validated_jobtag(args.source_jobtag, "explicit source jobtag")
    run_id = validated_jobtag(args.run_id, "run id")
    aggregate_supplied = Path(args.aggregate)
    require_regular(aggregate_supplied, "W64 aggregate")
    aggregate_path = aggregate_supplied.resolve(strict=True)
    if not aggregate_supplied.is_absolute() or aggregate_supplied != aggregate_path:
        raise ValueError("W64 aggregate path must be absolute, canonical, and escape-free")
    result_root = aggregate_path.parent
    run_root = result_root.parent
    require_directory(run_root, "W64 run root")
    require_directory(result_root, "W64 result root")
    if run_root.resolve(strict=True) != run_root or result_root.resolve(strict=True) != result_root:
        raise ValueError("W64 run/result root must be canonical and non-symlinked")
    if run_root.name != run_id or result_root.name != source_jobtag:
        raise ValueError("W64 aggregate canonical run/source hierarchy mismatch")
    expected_aggregate = result_root / f"telemetry.{source_jobtag}.aggregate.json"
    if aggregate_path != expected_aggregate:
        raise ValueError("W64 aggregate path is not canonical for explicit source jobtag")
    analysis_verify_supplied = Path(args.analysis_freeze_verify)
    require_regular(analysis_verify_supplied, "analysis-time full freeze verify")
    analysis_verify_path = analysis_verify_supplied.resolve(strict=True)
    verify_prefix = f"analysis_freeze_verify.{source_jobtag}."
    if (
        not analysis_verify_supplied.is_absolute()
        or analysis_verify_supplied != analysis_verify_path
        or analysis_verify_path.parent != result_root
        or not analysis_verify_path.name.startswith(verify_prefix)
        or not analysis_verify_path.name.endswith(".pre.json")
        or Path(analysis_verify_path.name).name != analysis_verify_path.name
    ):
        raise ValueError("analysis-time freeze verify path is non-canonical/unsafe")
    verify_analysis_jobtag = analysis_verify_path.name[
        len(verify_prefix) : -len(".pre.json")
    ]
    validated_jobtag(verify_analysis_jobtag, "analysis freeze-verify jobtag")
    output_supplied, analysis_jobtag = validated_analysis_output(args)
    if output_supplied.parent != result_root:
        raise ValueError("analysis output and aggregate source roots differ")
    if analysis_jobtag != verify_analysis_jobtag:
        raise ValueError("analysis output/full-freeze jobtags differ")
    plan_root = Path(args.plan)
    freeze_path = Path(args.freeze)
    repo = Path(args.repo).resolve(strict=True)
    runner_path = Path(args.runner)
    capture_yaml_path = Path(args.capture_yaml)
    analyze_yaml_path = Path(args.analyze_yaml)
    frozen = validate_frozen_gate_files(
        freeze_path,
        repo=repo,
        runner=runner_path,
        capture_yaml=capture_yaml_path,
        analyze_yaml=analyze_yaml_path,
    )
    runner_sha = frozen["gate_files"][FROZEN_GATE_FILES[0]]["sha256"]
    capture_yaml_sha = frozen["gate_files"][FROZEN_GATE_FILES[1]]["sha256"]
    aggregate_data = validate_aggregate(
        aggregate_path,
        run_id=args.run_id,
        freeze=frozen,
        runner_sha256=runner_sha,
        yaml_sha256=capture_yaml_sha,
    )
    if aggregate_data["jobtag"] != source_jobtag:
        raise ValueError("aggregate jobtag differs from explicit source jobtag")
    aggregate_value = aggregate_data["value"]
    contract_path = result_root / f"runtime_contract.{aggregate_data['jobtag']}.json"
    if aggregate_value["runtime_contract"].get("path") != str(
        contract_path.resolve(strict=True)
    ):
        raise ValueError("aggregate runtime-contract path is non-canonical")
    runtime_contract = validate_runtime_contract(
        contract_path,
        entry=aggregate_value["runtime_contract"],
        aggregate_sha256=aggregate_data["runtime_contract_sha256"],
        args=args,
        frozen=frozen,
        jobtag=aggregate_data["jobtag"],
        nonce=aggregate_data["nonce"],
    )
    planner_module = sys.modules.get(OfflinePlan.__module__)
    planner_entry = runtime_contract["artifacts"]["planner_source"]
    if (
        planner_module is None
        or not isinstance(getattr(planner_module, "__file__", None), str)
        or Path(planner_module.__file__).resolve(strict=True)
        != Path(planner_entry["path"])
        or sha256_file(Path(planner_module.__file__).resolve(strict=True))
        != planner_entry["sha256"]
    ):
        raise ValueError("analysis imported planner differs from runtime contract")
    if runtime_contract["mode"] != aggregate_data["gpic_source_consistency_mode"]:
        raise ValueError("runtime-contract/aggregate GPIC consistency mode mismatch")
    if runtime_contract["value"]["gpic_threads_per_node"] != aggregate_data["gpic_threads_per_node"]:
        raise ValueError("runtime-contract/aggregate GPIC threads-per-node mismatch")
    capture_pre_path = result_root / f"freezepreflight.{aggregate_data['jobtag']}.json"
    capture_post_path = result_root / f"freezepost.{aggregate_data['jobtag']}.json"
    for key, path in (
        ("preflight_verify", capture_pre_path),
        ("postflight_verify", capture_post_path),
    ):
        if aggregate_value[key].get("path") != str(path.resolve(strict=True)):
            raise ValueError(f"aggregate {key} path is non-canonical")
        artifact_matches(path, aggregate_value[key], f"aggregate {key}")
    capture_pre_verify = validate_full_freeze_verify(
        capture_pre_path, frozen=frozen, label="capture preflight full freeze verify"
    )
    capture_post_verify = validate_full_freeze_verify(
        capture_post_path, frozen=frozen, label="capture postflight full freeze verify"
    )
    analysis_verify = validate_full_freeze_verify(
        analysis_verify_path, frozen=frozen, label="analysis-time full freeze verify"
    )
    if not exact_json_equal(
        capture_pre_verify["value"], capture_post_verify["value"]
    ) or not exact_json_equal(capture_pre_verify["value"], analysis_verify["value"]):
        raise ValueError("capture/analysis complete freeze verification results differ")
    closure_marker_evidence = []
    for node, entry in enumerate(aggregate_value["closure_markers"]):
        marker_path = result_root / f"closure.{aggregate_data['jobtag']}.n{node}.ready"
        marker_sha = frozen_artifact_matches(
            marker_path, entry, f"final closure marker node {node}"
        )
        fields = marker_path.read_text(encoding="ascii").strip().split("|")
        if fields != [
            "CLOSED",
            aggregate_data["jobtag"],
            aggregate_data["nonce"],
            str(node),
            runtime_contract["sha256"],
            capture_post_verify["sha256"],
        ]:
            raise ValueError(f"final closure marker node {node} content mismatch")
        closure_marker_evidence.append({"path": str(marker_path), "sha256": marker_sha})
    preflight = validate_preflight(
        result_root,
        jobtag=aggregate_data["jobtag"],
        nonce=aggregate_data["nonce"],
        freeze_sha256=frozen["freeze_sha256"],
        release_digest=frozen["release_digest"],
        runner_sha256=runner_sha,
        yaml_sha256=capture_yaml_sha,
        runtime_contract=runtime_contract,
    )
    if (
        preflight["sidecar_header_sha256"] != aggregate_data["sidecar_header_sha256"]
        or preflight["sidecar_release_digest"] != aggregate_data["sidecar_release_digest"]
    ):
        raise ValueError("preflight/aggregate sidecar identity mismatch")
    if aggregate_value["preflight_pass"].get("path") != preflight["pass"]["path"]:
        raise ValueError("aggregate preflight-PASS path mismatch")
    aggregate_pass_sha = artifact_matches(
        Path(preflight["pass"]["path"]),
        aggregate_value["preflight_pass"],
        "aggregate preflight PASS",
    )
    if aggregate_pass_sha != preflight["pass"]["sha256"]:
        raise ValueError("aggregate/preflight PASS SHA mismatch")
    for node, (control, entry) in enumerate(
        zip(preflight["controls"], aggregate_value["sidecar_controls"])
    ):
        if entry.get("path") != control["path"]:
            raise ValueError(f"aggregate sidecar-control path mismatch node {node}")
        if artifact_matches(
            Path(control["path"]), entry, f"aggregate sidecar control node {node}"
        ) != control["sha256"]:
            raise ValueError(f"aggregate sidecar-control SHA mismatch node {node}")

    require_directory(plan_root, "W64 plan root")
    require_directory(plan_root / "blocks", "W64 plan blocks root")
    freeze_value = frozen["freeze"]
    freeze_inputs = freeze_value.get("inputs")
    if not isinstance(freeze_inputs, dict):
        raise ValueError("fresh freeze inputs section is malformed")
    plan_header_path = plan_root / "header.json"
    plan_checksums_path = plan_root / "checksums.json"
    plan_header_sha = frozen_artifact_matches(
        plan_header_path, freeze_inputs.get("plan_header"), "freeze-bound plan header"
    )
    plan_checksums_sha = frozen_artifact_matches(
        plan_checksums_path,
        freeze_inputs.get("plan_checksums"),
        "freeze-bound plan checksums",
    )
    plan = OfflinePlan(
        plan_root,
        world_size=WORLD_SIZE,
        per_rank_bs=PER_RANK_BS,
        verify_checksums=True,
    )
    if plan.config.total_steps < FORMAL_STEPS:
        raise ValueError(f"W64 plan has fewer than formal {FORMAL_STEPS} steps")
    plan_id = str(plan.header["plan_id"])
    plan_hash = str(plan.header["plan_hash"])
    catalog_hash = str(plan.header["catalog_hash"])
    domains = tuple(str(value) for value in plan.header["domains"])
    if plan_id != EXPECTED_PLAN_ID or domains != EXPECTED_DOMAINS:
        raise ValueError("W64 production plan id/domain order mismatch")
    freeze_identities = freeze_value.get("identities")
    freeze_sidecar = freeze_value.get("caption_sidecar")
    if not isinstance(freeze_identities, dict) or not isinstance(freeze_sidecar, dict):
        raise ValueError("fresh freeze identity/sidecar sections are malformed")
    expected_freeze_identities = {
        "plan_id": plan_id,
        "catalog_hash": catalog_hash,
        "caption_sidecar_header_sha256": aggregate_data["sidecar_header_sha256"],
        "caption_sidecar_release_digest": aggregate_data["sidecar_release_digest"],
    }
    for key, wanted in expected_freeze_identities.items():
        if not exact_json_equal(freeze_identities.get(key), wanted):
            raise ValueError(f"fresh freeze identity {key} mismatch")
    if (
        freeze_sidecar.get("header_sha256") != aggregate_data["sidecar_header_sha256"]
        or freeze_sidecar.get("release_digest")
        != aggregate_data["sidecar_release_digest"]
        or freeze_sidecar.get("catalog_hash") != catalog_hash
    ):
        raise ValueError("fresh freeze caption-sidecar identity mismatch")
    expected_counts = PlanWindowCounts(plan)

    nodes = aggregate_data["value"]["nodes"]
    evidence_root = result_root / f"telemetry.{aggregate_data['jobtag']}"
    require_directory(evidence_root, "W64 evidence root")
    rank_paths: dict[int, Path] = {}
    rank_hosts: dict[int, str] = {}
    runner_data: dict[int, dict[str, Any]] = {}
    rank_telemetry_sha256: dict[int, str] = {}
    manifest_evidence = []
    for node_rank in range(NODES):
        matches = [
            value
            for value in nodes
            if isinstance(value, dict) and value.get("node_rank") == node_rank
        ]
        if len(matches) != 1:
            raise ValueError(f"aggregate node {node_rank} missing/duplicated")
        node = matches[0]
        expected_node_keys = {
            "node_rank",
            "manifest",
            "manifest_sha256",
            "status",
            "run_rc",
            "tee_rc",
            "collect_rc",
            "runtime_post_rc",
            "rank_files",
            "runner_log_sha256",
        }
        if set(node) != expected_node_keys:
            raise ValueError(f"aggregate node {node_rank} key set drift")
        node_root = evidence_root / f"node{node_rank}"
        require_directory(node_root, f"node {node_rank} evidence directory")
        manifest_path = node_root / "node_manifest.json"
        if node.get("manifest") != str(manifest_path):
            raise ValueError(f"node {node_rank} aggregate manifest path mismatch")
        if any(
            not exact_json_equal(node.get(key), wanted)
            for key, wanted in (
                ("node_rank", node_rank),
                ("status", "PASS"),
                ("run_rc", 0),
                ("tee_rc", 0),
                ("collect_rc", 0),
                ("runtime_post_rc", 0),
                ("rank_files", LOCAL_WORLD_SIZE),
            )
        ):
            raise ValueError(f"node {node_rank} aggregate close is not clean")
        manifest = read_object(manifest_path, f"node {node_rank} manifest")
        manifest_sha = sha256_file(manifest_path)
        if node.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"node {node_rank} aggregate manifest SHA mismatch")
        preservation_path = result_root / (
            f"telemetry.{aggregate_data['jobtag']}.n{node_rank}.ready"
        )
        preservation_entry = aggregate_value["preservation_markers"][node_rank]
        if preservation_entry.get("path") != str(preservation_path.resolve(strict=True)):
            raise ValueError(f"node {node_rank} preservation-marker path mismatch")
        artifact_matches(
            preservation_path,
            preservation_entry,
            f"node {node_rank} preservation marker",
        )
        preservation_fields = preservation_path.read_text(encoding="ascii").strip().split("|")
        expected_preservation_fields = [
            "PRESERVED",
            aggregate_data["jobtag"],
            aggregate_data["nonce"],
            str(node_rank),
            manifest_sha,
            runtime_contract["sha256"],
            "0",
            "0",
            "0",
            "0",
        ]
        if preservation_fields != expected_preservation_fields:
            raise ValueError(f"node {node_rank} preservation-marker content mismatch")
        expected_manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "PASS",
            "error": "",
            "jobtag": aggregate_data["jobtag"],
            "nonce": aggregate_data["nonce"],
            "node_rank": node_rank,
            "host": f"{aggregate_data['jobtag']}-{node_rank}",
            "run_rc": 0,
            "tee_rc": 0,
            "collect_rc": 0,
            "runtime_post_rc": 0,
            "world_size": WORLD_SIZE,
            "local_world_size": LOCAL_WORLD_SIZE,
            "start_step": START_STEP,
            "stop_step": STOP_STEP,
            "expected_global_ranks": list(
                range(node_rank * LOCAL_WORLD_SIZE, (node_rank + 1) * LOCAL_WORLD_SIZE)
            ),
            "freeze_release_digest": frozen["release_digest"],
            "freeze_sha256": frozen["freeze_sha256"],
            "sidecar_header_sha256": aggregate_data["sidecar_header_sha256"],
            "sidecar_release_digest": aggregate_data["sidecar_release_digest"],
            "runner_sha256": runner_sha,
            "yaml_sha256": capture_yaml_sha,
            "runtime_contract_sha256": runtime_contract["sha256"],
            "gpic_source_consistency_mode": runtime_contract["mode"],
            "gpic_threads_per_node": runtime_contract["value"]["gpic_threads_per_node"],
        }
        if set(manifest) != set(expected_manifest) | {"runner_log", "files"}:
            raise ValueError(f"node {node_rank} manifest key set drift")
        for key, wanted in expected_manifest.items():
            if not exact_json_equal(manifest.get(key), wanted):
                raise ValueError(f"node {node_rank} manifest {key} mismatch")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != LOCAL_WORLD_SIZE:
            raise ValueError(f"node {node_rank} manifest must declare eight rank files")
        expected_ranks = set(
            range(node_rank * LOCAL_WORLD_SIZE, (node_rank + 1) * LOCAL_WORLD_SIZE)
        )
        observed_ranks = set()
        expected_names = {"node_manifest.json", "runner.log"}
        for entry in files:
            if not isinstance(entry, dict):
                raise ValueError(f"node {node_rank} rank artifact declaration malformed")
            if set(entry) != {"name", "bytes", "sha256"}:
                raise ValueError(f"node {node_rank} rank artifact key set drift")
            name = entry.get("name")
            match = RANK_FILE_RE.fullmatch(name) if isinstance(name, str) else None
            if match is None:
                raise ValueError(f"node {node_rank} unsafe rank filename {name!r}")
            rank = int(match.group("rank"))
            if rank in observed_ranks:
                raise ValueError(f"duplicate telemetry rank {rank}")
            path = manifest_path.parent / name
            rank_sha = artifact_matches(path, entry, f"rank {rank} telemetry")
            observed_ranks.add(rank)
            rank_paths[rank] = path
            rank_hosts[rank] = manifest["host"]
            rank_telemetry_sha256[rank] = rank_sha
            expected_names.add(name)
        if observed_ranks != expected_ranks:
            raise ValueError(f"node {node_rank} telemetry rank set mismatch")
        runner_log = manifest.get("runner_log")
        if (
            not isinstance(runner_log, dict)
            or set(runner_log) != {"name", "bytes", "sha256"}
            or runner_log.get("name") != "runner.log"
        ):
            raise ValueError(f"node {node_rank} manifest runner-log declaration mismatch")
        runner_log_path = manifest_path.parent / "runner.log"
        runner_log_sha = artifact_matches(
            runner_log_path, runner_log, f"node {node_rank} runner log"
        )
        if node.get("runner_log_sha256") != runner_log_sha:
            raise ValueError(f"node {node_rank} aggregate runner-log SHA mismatch")
        observed_names = {path.name for path in manifest_path.parent.iterdir()}
        if observed_names != expected_names:
            raise ValueError(f"node {node_rank} atomic evidence exact-set mismatch")
        parsed_log = parse_runner_log(
            runner_log_path,
            node_rank,
            runtime_contract["mode"],
        )
        overlap = set(runner_data).intersection(parsed_log)
        if overlap:
            raise ValueError(f"duplicate runner ranks across nodes: {sorted(overlap)}")
        runner_data.update(parsed_log)
        manifest_evidence.append(
            {
                "node_rank": node_rank,
                "path": str(manifest_path),
                "sha256": manifest_sha,
                "runner_log": str(runner_log_path),
                "runner_log_sha256": runner_log_sha,
                "preservation_marker": str(preservation_path),
                "preservation_marker_sha256": preservation_entry["sha256"],
            }
        )
    if set(rank_paths) != set(range(WORLD_SIZE)) or set(runner_data) != set(range(WORLD_SIZE)):
        raise ValueError("W64 evidence does not cover global ranks 0..63 exactly")
    observed_node_entries = {path.name for path in evidence_root.iterdir()}
    expected_node_entries = {f"node{node_rank}" for node_rank in range(NODES)}
    if observed_node_entries != expected_node_entries:
        raise ValueError("W64 evidence root does not contain exactly node0..node7")

    telemetry: dict[int, dict[str, Any]] = {}
    for rank in range(WORLD_SIZE):
        telemetry[rank] = load_rank_telemetry(
            rank_paths[rank],
            rank,
            plan_id=plan_id,
            plan_hash=plan_hash,
            catalog_hash=catalog_hash,
            jobtag=aggregate_data["jobtag"],
            nonce=aggregate_data["nonce"],
            host=rank_hosts[rank],
            sidecar_header_sha256=aggregate_data["sidecar_header_sha256"],
            sidecar_release_digest=aggregate_data["sidecar_release_digest"],
            domains=set(domains),
            runner_steps=runner_data[rank]["steps"],
            gpic_source_consistency_mode=runtime_contract["mode"],
        )
        for start in ALL_WINDOW_STARTS:
            counts = expected_counts.get(start, rank)
            for domain in domains:
                record = telemetry[rank]["windows"].get((start, domain))
                if (record is not None) != (counts[domain] > 0):
                    raise ValueError(
                        f"rank {rank} window {(start, domain)} presence/plan mismatch"
                    )
                if record is not None and record.get("n") != counts[domain]:
                    raise ValueError(f"rank {rank} window {(start, domain)} n/plan mismatch")
        if len(telemetry[rank]["windows"]) != sum(
            expected_counts.get(start, rank)[domain] > 0
            for start in ALL_WINDOW_STARTS
            for domain in domains
        ):
            raise ValueError(f"rank {rank} telemetry contains unexpected windows")
        if runner_data[rank]["summary"]["telemetry_windows"] != len(
            telemetry[rank]["windows"]
        ):
            raise ValueError(f"rank {rank} runner/telemetry terminal window count mismatch")

    critical_windows = []
    for start in STEADY_WINDOW_STARTS:
        candidates = []
        for rank in range(WORLD_SIZE):
            for domain in domains:
                record = telemetry[rank]["windows"].get((start, domain))
                if record is not None:
                    candidates.append((float(record["total_s"]), rank, domain))
        if not candidates:
            raise ValueError(f"no critical-window candidates for start {start}")
        seconds, rank, domain = max(candidates)
        critical_windows.append(
            {"start": start, "critical_s": seconds, "rank": rank, "domain": domain}
        )
    critical_values = [item["critical_s"] for item in critical_windows]
    critical_summary = {
        "windows": len(critical_windows),
        "mean_s": statistics.fmean(critical_values),
        "median_s": statistics.median(critical_values),
        "p95_s": percentile(critical_values, 0.95),
        "max_s": max(critical_values),
        "critical_windows": critical_windows,
    }

    rank_domain_rates: dict[str, dict[str, float]] = {domain: {} for domain in domains}
    rank_domain_required: dict[str, dict[str, float]] = {domain: {} for domain in domains}
    rank_domain_capacity: dict[str, dict[str, float]] = {domain: {} for domain in domains}
    target_window_seconds = (
        len(STEADY_WINDOW_STARTS) * PREFETCH_STEPS * TARGET_SECONDS_PER_STEP
    )
    for rank in range(WORLD_SIZE):
        for domain in domains:
            images = 0
            seconds = 0.0
            for start in STEADY_WINDOW_STARTS:
                count = expected_counts.get(start, rank)[domain]
                if count:
                    images += count
                    seconds += float(telemetry[rank]["windows"][(start, domain)]["total_s"])
            if images <= 0 or seconds <= 0:
                raise ValueError(f"rank {rank} lacks steady capacity evidence for {domain}")
            observed = images / seconds
            required = images / target_window_seconds
            rank_domain_rates[domain][str(rank)] = observed
            rank_domain_required[domain][str(rank)] = required
            rank_domain_capacity[domain][str(rank)] = observed / required
    domain_min_capacity = {
        domain: min(values.values()) for domain, values in rank_domain_capacity.items()
    }

    critical_waits = []
    for step in STEADY_STEPS:
        candidates = [runner_data[rank]["steps"][step]["wait_s"] for rank in range(WORLD_SIZE)]
        maximum = max(candidates)
        critical_waits.append(
            {"step": step, "critical_wait_s": maximum, "rank": candidates.index(maximum)}
        )
    wait_values = [item["critical_wait_s"] for item in critical_waits]
    wait_over_target = sum(value > TARGET_SECONDS_PER_STEP for value in wait_values)
    wait_summary = {
        "steps": len(wait_values),
        "mean_s": statistics.fmean(wait_values),
        "median_s": statistics.median(wait_values),
        "p95_s": percentile(wait_values, 0.95),
        "max_s": max(wait_values),
        "over_target_count": wait_over_target,
        "over_target_fraction": wait_over_target / len(wait_values),
        "critical_steps": critical_waits,
    }
    elapsed_per_step = {
        str(rank): runner_data[rank]["summary"]["elapsed_s"] / STOP_STEP
        for rank in range(WORLD_SIZE)
    }
    max_elapsed_per_step = max(elapsed_per_step.values())

    plan_blocks = freeze_value.get("plan_blocks")
    plan_block_files = plan_blocks.get("files") if isinstance(plan_blocks, dict) else None
    if not isinstance(plan_block_files, dict):
        raise ValueError("fresh freeze plan-block file section is malformed")
    verified_blocks = sorted(plan._verified_blocks)
    expected_block_indices = sorted(
        {step // plan.config.block_steps for step in range(START_STEP, STOP_STEP)}
    )
    if verified_blocks != expected_block_indices:
        raise ValueError(
            "plan-derived W64 count block set mismatch: "
            f"verified={verified_blocks}, expected={expected_block_indices}"
        )
    plan_block_evidence: dict[str, dict[str, Any]] = {}
    for block_index in verified_blocks:
        relative, block_path = plan._block_path(block_index)
        block_sha = frozen_artifact_matches(
            block_path,
            plan_block_files.get(relative),
            f"freeze-bound plan block {relative}",
        )
        if plan.checksums.get(relative) != block_sha:
            raise ValueError(f"plan reader/checksums/freeze disagree for {relative}")
        plan_block_evidence[relative] = {
            "path": str(block_path),
            "sha256": block_sha,
            "bytes": block_path.stat().st_size,
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
    add_gate(
        gates,
        "critical_window_mean_s",
        critical_summary["mean_s"],
        "<=",
        WINDOW_MEAN_MAX_S,
    )
    add_gate(
        gates,
        "critical_window_p95_s",
        critical_summary["p95_s"],
        "<=",
        WINDOW_TAIL_MAX_S,
    )
    add_gate(
        gates,
        "critical_window_max_s",
        critical_summary["max_s"],
        "<=",
        WINDOW_TAIL_MAX_S,
    )
    add_gate(
        gates,
        "max_rank_runner_elapsed_per_step_s",
        max_elapsed_per_step,
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
    status = "PASS" if not failed else "NO_GO"

    # Final immutable-evidence close: every byte-bearing artifact used above is
    # re-opened only after all calculations have finished.  A mutation during
    # analysis is FATAL rather than a performance NO_GO.
    closure_artifacts: dict[str, tuple[Path, str]] = {
        "aggregate": (aggregate_path, aggregate_data["aggregate_sha256"]),
        "fresh_freeze": (freeze_path, frozen["freeze_sha256"]),
        "preflight_pass": (
            Path(preflight["pass"]["path"]),
            preflight["pass"]["sha256"],
        ),
        "preflight_json": (
            Path(preflight["json"]["path"]),
            preflight["json"]["sha256"],
        ),
        "plan_header": (plan_header_path, plan_header_sha),
        "plan_checksums": (plan_checksums_path, plan_checksums_sha),
        "runtime_contract": (
            Path(runtime_contract["path"]),
            runtime_contract["sha256"],
        ),
        "capture_postflight_full_freeze_verify": (
            Path(capture_post_verify["path"]),
            capture_post_verify["sha256"],
        ),
        "analysis_time_full_freeze_verify": (
            Path(analysis_verify["path"]),
            analysis_verify["sha256"],
        ),
    }
    for relative, entry in frozen["gate_files"].items():
        closure_artifacts[f"gate:{relative}"] = (Path(entry["path"]), entry["sha256"])
    for name, entry in runtime_contract["artifacts"].items():
        closure_artifacts[f"runtime:{name}"] = (Path(entry["path"]), entry["sha256"])
    for control in preflight["controls"]:
        closure_artifacts[f"sidecar_control:{control['path']}"] = (
            Path(control["path"]),
            control["sha256"],
        )
    for marker in preflight["fresh_markers"]:
        closure_artifacts[f"fresh_marker:{marker['path']}"] = (
            Path(marker["path"]),
            marker["sha256"],
        )
    for node, marker in enumerate(closure_marker_evidence):
        closure_artifacts[f"final_closure_marker:{node}"] = (
            Path(marker["path"]),
            marker["sha256"],
        )
    for item in manifest_evidence:
        closure_artifacts[f"node_manifest:{item['node_rank']}"] = (
            Path(item["path"]),
            item["sha256"],
        )
        closure_artifacts[f"runner_log:{item['node_rank']}"] = (
            Path(item["runner_log"]),
            item["runner_log_sha256"],
        )
        closure_artifacts[f"preservation_marker:{item['node_rank']}"] = (
            Path(item["preservation_marker"]),
            item["preservation_marker_sha256"],
        )
    for rank, path in rank_paths.items():
        closure_artifacts[f"rank_telemetry:{rank}"] = (
            path,
            rank_telemetry_sha256[rank],
        )
    for relative, entry in plan_block_evidence.items():
        closure_artifacts[f"plan_block:{relative}"] = (
            Path(entry["path"]),
            entry["sha256"],
        )
    for label, (path, expected_sha) in closure_artifacts.items():
        require_regular(path, f"final closure {label}")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"final immutable-evidence closure mismatch: {label}")
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "status": status,
        "failed_gates": failed,
        "reasons": (
            ["all_w64_raw_supply_evidence_and_capacity_gates_pass"]
            if not failed
            else [f"strict_gate_failed:{name}" for name in failed]
        ),
        "identity": {
            "run_id": args.run_id,
            "jobtag": aggregate_data["jobtag"],
            "nonce": aggregate_data["nonce"],
            "plan_id": plan_id,
            "plan_hash": plan_hash,
            "catalog_hash": catalog_hash,
            "sidecar_header_sha256": aggregate_data["sidecar_header_sha256"],
            "sidecar_release_digest": aggregate_data["sidecar_release_digest"],
            "freeze_sha256": frozen["freeze_sha256"],
            "freeze_release_digest": frozen["release_digest"],
            "runtime_contract_sha256": runtime_contract["sha256"],
            "gpic_source_consistency_mode": runtime_contract["mode"],
            "gpic_threads_per_node": runtime_contract["value"]["gpic_threads_per_node"],
            "world_size": WORLD_SIZE,
            "local_world_size": LOCAL_WORLD_SIZE,
            "per_rank_bs": PER_RANK_BS,
            "start_step": START_STEP,
            "stop_step": STOP_STEP,
            "steady_steps": [STEADY_STEPS.start, STEADY_STEPS.stop - 1],
            "steady_window_starts": [
                STEADY_WINDOW_STARTS.start,
                STEADY_WINDOW_STARTS.stop - PREFETCH_STEPS,
                PREFETCH_STEPS,
            ],
            "target_seconds_per_step": TARGET_SECONDS_PER_STEP,
            "formal_reference_steps": FORMAL_STEPS,
            "formal_reference_samples": FORMAL_STEPS * WORLD_SIZE * PER_RANK_BS,
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
        },
        "gates": gates,
        "global_critical_wait": wait_summary,
        "combined_critical_windows": critical_summary,
        "domain_min_rank_capacity_ratio": domain_min_capacity,
        "rank_domain_capacity_ratio": rank_domain_capacity,
        "rank_domain_img_s": rank_domain_rates,
        "rank_domain_required_img_s": rank_domain_required,
        "rank_runner_elapsed_per_step_s": elapsed_per_step,
        "raw_supply_projected_days_reference": (
            max_elapsed_per_step * FORMAL_STEPS / 86_400.0
        ),
        "telemetry_record_counts": {
            str(rank): telemetry[rank]["record_counts"] for rank in range(WORLD_SIZE)
        },
        "evidence": {
            "aggregate": str(aggregate_path),
            "aggregate_sha256": aggregate_data["aggregate_sha256"],
            "fresh_freeze": str(freeze_path),
            "fresh_freeze_sha256": frozen["freeze_sha256"],
            "runtime_contract": {
                "path": runtime_contract["path"],
                "sha256": runtime_contract["sha256"],
                "artifacts": runtime_contract["artifacts"],
            },
            "capture_preflight_full_freeze_verify": {
                "path": capture_pre_verify["path"],
                "sha256": capture_pre_verify["sha256"],
            },
            "capture_postflight_full_freeze_verify": {
                "path": capture_post_verify["path"],
                "sha256": capture_post_verify["sha256"],
            },
            "analysis_time_full_freeze_verify": {
                "path": analysis_verify["path"],
                "sha256": analysis_verify["sha256"],
                "training_sources": analysis_verify["value"]["training_sources"],
                "catalog_files": analysis_verify["value"]["catalog_files"],
                "plan_blocks": analysis_verify["value"]["plan_blocks"],
            },
            "frozen_gate_files": frozen["gate_files"],
            "node_manifests": manifest_evidence,
            "sidecar_controls": preflight["controls"],
            "fresh_markers": preflight["fresh_markers"],
            "final_closure_markers": closure_marker_evidence,
            "preflight_pass": preflight["pass"],
            "preflight_json": preflight["json"],
            "plan_header_sha256": plan_header_sha,
            "plan_checksums_sha256": plan_checksums_sha,
            "verified_plan_blocks": plan_block_evidence,
            "rank_telemetry_sha256": {
                str(rank): rank_telemetry_sha256[rank] for rank in range(WORLD_SIZE)
            },
            "final_closure_artifacts": len(closure_artifacts),
        },
        "scope": (
            "Raw supply only: this gate proves 64-rank plan-bound byte supply capacity, "
            "not decode/model/FSDP/checkpoint correctness; strict 2x8 GPUVAL remains mandatory."
        ),
    }


def atomic_publish_bytes(path: Path, payload: bytes, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite {label}: {path}")
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(16)}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ValueError(f"refusing to overwrite {label}: {path}") from exc
        temporary.unlink()
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


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_publish_bytes(path, payload, label="analysis output")


def atomic_no_clobber_self_test(root: Path) -> None:
    if root.exists() or root.is_symlink():
        raise RuntimeError(f"atomic self-test root already exists: {root}")
    root.mkdir(parents=True)
    target = root / "winner.json"
    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, str]] = []
    outcome_lock = threading.Lock()

    def writer(name: str) -> None:
        value = {"schema_version": "w64_atomic_no_clobber_self_test_v1", "writer": name}
        barrier.wait()
        try:
            atomic_write(target, value)
            outcome = ("success", name)
        except ValueError:
            outcome = ("exists", name)
        with outcome_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=writer, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    successes = [name for status, name in outcomes if status == "success"]
    rejected = [name for status, name in outcomes if status == "exists"]
    if len(successes) != 1 or len(rejected) != 1:
        raise RuntimeError(f"atomic no-clobber race accounting mismatch: {outcomes}")
    winner_bytes = target.read_bytes()
    winner = json.loads(winner_bytes.decode("utf-8"))
    if winner.get("writer") != successes[0]:
        raise RuntimeError("atomic no-clobber published content does not match winner")
    try:
        atomic_write(target, {"writer": "late-overwrite"})
    except ValueError:
        pass
    else:
        raise RuntimeError("atomic no-clobber accepted a late overwrite")
    if target.read_bytes() != winner_bytes:
        raise RuntimeError("atomic no-clobber winner changed after rejected overwrite")
    leftovers = {path.name for path in root.iterdir()}
    if leftovers != {target.name}:
        raise RuntimeError(f"atomic no-clobber left temporary artifacts: {leftovers}")
    target.unlink()
    root.rmdir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--analysis-freeze-verify", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--capture-yaml", required=True)
    parser.add_argument("--analyze-yaml", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--catalog-root", required=True)
    parser.add_argument("--anime-freq-root", required=True)
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--sampling", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-jobtag", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    safe_output: Path | None = None
    try:
        safe_output, _analysis_jobtag = validated_analysis_output(args)
        result = analyze(args)
    except BaseException as exc:
        result = {
            "schema_version": ANALYSIS_SCHEMA,
            "status": "FATAL",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if safe_output is not None:
            try:
                atomic_write(safe_output, result)
            except BaseException as output_exc:
                print(
                    f"FATAL W64 analyzer output failure: {type(output_exc).__name__}: {output_exc}",
                    file=sys.stderr,
                )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    if safe_output is not None:
        atomic_write(safe_output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
