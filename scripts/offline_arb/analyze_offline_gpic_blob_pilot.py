#!/usr/bin/env python3
"""Independently close an ``offline_gpic_blob_v1`` pilot evidence set.

The analyzer reports measurements and byte-equivalence only.  It never marks a
pilot production-qualified and intentionally has no automatic training switch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SHA256 = set("0123456789abcdef")
FORMAL_GLOBAL_GPIC_OBJECTS_PER_SECOND = 230.0
FORMAL_NODE_COUNT = 8


class AnalysisError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256


def manifest_hash(value: dict[str, Any]) -> str:
    core = dict(value)
    core.pop("manifest_sha256", None)
    return canonical_sha(core)


def report_hash(value: dict[str, Any]) -> str:
    core = dict(value)
    core.pop("report_sha256", None)
    return canonical_sha(core)


def receipt_hash(value: dict[str, Any]) -> str:
    core = dict(value)
    core.pop("receipt_sha256", None)
    return canonical_sha(core)


def validate_runner_job_binding(
    value: Any,
    *,
    label: str,
    runner_path: Path,
    runner_sha256: str,
    job_spec_path: Path,
    job_spec_sha256: str,
) -> None:
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} code binding is missing")
    try:
        recorded_runner = Path(value["runner_path"]).resolve(strict=True)
        recorded_job = Path(value["job_spec_path"]).resolve(strict=True)
    except Exception as exc:
        raise AnalysisError(f"{label} code paths are invalid") from exc
    if (
        recorded_runner != runner_path
        or value.get("runner_sha256") != runner_sha256
        or recorded_job != job_spec_path
        or value.get("job_spec_sha256") != job_spec_sha256
    ):
        raise AnalysisError(f"{label} runner/job-spec binding drift")


def load_json(path: Path) -> tuple[dict[str, Any], str]:
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if not path.is_file() or path.is_symlink() or identity_before != identity_after or len(raw) != before.st_size:
        raise AnalysisError(f"unstable/non-regular JSON input: {path}")
    try:
        value = json.loads(raw)
    except Exception as exc:
        raise AnalysisError(f"invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"JSON input is not an object: {path}")
    if raw != canonical_bytes(value):
        raise AnalysisError(f"JSON evidence is not canonical: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def stable_file_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    count = 0
    with path.open("rb", buffering=0) as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            count += len(block)
            digest.update(block)
    after = path.stat()
    if (
        path.is_symlink()
        or not path.is_file()
        or count != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    ):
        raise AnalysisError(f"unstable analyzer code input: {path}")
    return digest.hexdigest()


def labeled_path(raw: str) -> tuple[str, Path]:
    label, separator, path = raw.partition("=")
    if not separator or not label or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in label):
        raise argparse.ArgumentTypeError("expected LABEL=/absolute/path")
    value = Path(path)
    if not value.is_absolute():
        raise argparse.ArgumentTypeError("labeled path must be absolute")
    return label, value


def exact_pair(raw: str) -> tuple[str, str]:
    left, separator, right = raw.partition("=")
    if not separator or not left or not right:
        raise argparse.ArgumentTypeError("expected LEFT=RIGHT")
    return left, right


def load_artifact(root: Path) -> tuple[dict[str, Any], str]:
    root = root.resolve(strict=True)
    manifest, raw_sha = load_json(root / "manifest.json")
    if manifest.get("schema_version") != "offline_gpic_blob_v1_pilot":
        raise AnalysisError("artifact manifest schema drift")
    checksum = manifest.get("manifest_sha256")
    if not valid_sha(checksum) or manifest_hash(manifest) != checksum:
        raise AnalysisError("artifact manifest self-hash drift")
    success = (root / "_SUCCESS").read_bytes()
    if success not in {checksum.encode("ascii"), (checksum + "\n").encode("ascii")}:
        raise AnalysisError("artifact _SUCCESS/manifest mismatch")
    if manifest.get("pilot_only") is not True or manifest.get("production_selection_qualified") is not False:
        raise AnalysisError("artifact pilot qualification flags drift")
    return manifest, raw_sha


def file_identity_vector(manifest: dict[str, Any]) -> dict[str, Any]:
    files = manifest["files"]
    return {
        "selection": {key: files["selection"][key] for key in ("bytes", "sha256")},
        "index": {key: files["index"][key] for key in ("bytes", "sha256")},
        "blobs": [
            {key: blob[key] for key in ("blob_id", "bytes", "sha256")}
            for blob in files["blobs"]
        ],
    }


def summarize_timed_reports(values: list[dict[str, Any]]) -> dict[str, Any]:
    arms = [repeat for value in values for repeat in value["repeats"]]
    rates = sorted(float(arm["objects_per_second"]) for arm in arms)
    bandwidth = sorted(float(arm["mib_per_second"]) for arm in arms)
    p50 = sorted(float(arm["per_pread"]["p50_ms"]) for arm in arms)
    p95 = sorted(float(arm["per_pread"]["p95_ms"]) for arm in arms)
    p99 = sorted(float(arm["per_pread"]["p99_ms"]) for arm in arms)
    amplification = {float(arm["userspace_read_amplification"]) for arm in arms}
    if amplification != {1.0}:
        raise AnalysisError("userspace replay amplification is not exactly 1.0")

    def median(items: list[float]) -> float:
        middle = len(items) // 2
        return items[middle] if len(items) % 2 else (items[middle - 1] + items[middle]) / 2

    median_rate = median(rates)
    return {
        "arm_count": len(arms),
        "objects_per_second_median": median_rate,
        "objects_per_second_min": rates[0],
        "mib_per_second_median": median(bandwidth),
        "pread_p50_ms_median_across_arms": median(p50),
        "pread_p95_ms_median_across_arms": median(p95),
        "pread_p99_ms_median_across_arms": median(p99),
        "userspace_read_amplification": 1.0,
        "physical_backend_read_amplification_observable": False,
        "capacity_interpretation_forbidden": True,
    }


def summarize_single_capacity(value: dict[str, Any]) -> dict[str, Any]:
    repeats = value.get("repeats")
    if not isinstance(repeats, list) or len(repeats) != 1:
        raise AnalysisError("capacity authority must contain exactly one timed repeat")
    arm = repeats[0]
    if arm.get("userspace_read_amplification") != 1.0:
        raise AnalysisError("capacity authority userspace amplification drift")
    rate = float(arm["objects_per_second"])
    return {
        "objects": arm["objects"],
        "objects_per_second_exact_first_cold_arm": rate,
        "mib_per_second_exact_first_cold_arm": float(arm["mib_per_second"]),
        "pread_p50_ms_exact_first_cold_arm": float(arm["per_pread"]["p50_ms"]),
        "pread_p95_ms_exact_first_cold_arm": float(arm["per_pread"]["p95_ms"]),
        "pread_p99_ms_exact_first_cold_arm": float(arm["per_pread"]["p99_ms"]),
        "userspace_read_amplification": 1.0,
        "physical_backend_read_amplification_observable": False,
        "formal_required_objects_per_second_per_node": (
            FORMAL_GLOBAL_GPIC_OBJECTS_PER_SECOND / FORMAL_NODE_COUNT
        ),
        "capacity_multiple_vs_formal_per_node": (
            rate
            / (FORMAL_GLOBAL_GPIC_OBJECTS_PER_SECOND / FORMAL_NODE_COUNT)
        ),
        "capacity_arms_averaged": False,
        "eight_node_linear_scale_is_assumed": False,
    }


def summarize_replays(values: list[dict[str, Any]]) -> dict[str, Any]:
    authorities = [value for value in values if value.get("arm_role") == "capacity_authority"]
    integrity = [value for value in values if value.get("arm_role") == "integrity_warm"]
    if len(authorities) != 1:
        raise AnalysisError(
            f"artifact must have exactly one capacity authority, got {len(authorities)}"
        )
    if len(integrity) != 1:
        raise AnalysisError(
            f"artifact must have exactly one warm integrity arm, got {len(integrity)}"
        )
    authority = authorities[0]
    if (
        authority.get("arm_sequence_index") != 0
        or authority.get("selection_mode") != "cold_unique_region"
        or authority.get("capacity_authority") is not True
        or len(authority.get("repeats", [])) != 1
    ):
        raise AnalysisError("capacity authority is not the unique sequence-0 cold arm")
    warm = integrity[0]
    if (
        warm.get("arm_sequence_index") != 1
        or warm.get("selection_mode") != "full_coverage"
        or warm.get("capacity_authority") is not False
        or len(warm.get("repeats", [])) != 1
    ):
        raise AnalysisError("warm integrity arm is not the sequence-1 full pass")
    try:
        authority_time = datetime.fromisoformat(authority["generated_at"])
        warm_time = datetime.fromisoformat(warm["generated_at"])
    except Exception as exc:
        raise AnalysisError("replay generated_at timestamp is malformed") from exc
    if authority_time >= warm_time:
        raise AnalysisError("full-coverage integrity pass did not follow the cold authority")
    return {
        "capacity_authority": {
            "report_sha256": authority["report_sha256"],
            "selection": authority["replay_selection"],
            "production_capacity_status": authority["production_capacity_status"],
            "measurement": summarize_single_capacity(authority),
        },
        "warm_integrity": {
            "report_sha256": warm["report_sha256"],
            "selection": warm["replay_selection"],
            "measurement": summarize_timed_reports([warm]),
            "excluded_from_capacity_summary": True,
        },
        "capacity_arms_averaged": False,
    }


def validate_pilot_shape(
    *,
    source: dict[str, Any],
    source_raw_sha256: str,
    artifacts: dict[str, dict[str, Any]],
    artifact_raw_sha256: dict[str, str],
    artifact_roots: dict[str, Path],
    replays_by_artifact: dict[str, list[dict[str, Any]]],
    replay_inputs: list[dict[str, Any]],
    build_receipt: dict[str, Any] | None,
    s3_build_receipt: dict[str, Any] | None,
) -> None:
    expected_workers = [8, 16, 24, 32, 48, 48, 32, 24, 16, 8]
    arms = source.get("arms")
    if (
        source.get("status") != "PASS"
        or not isinstance(arms, list)
        or len(arms) != 10
        or [arm.get("arm_index") for arm in arms] != list(range(10))
        or [arm.get("workers") for arm in arms] != expected_workers
        or any(arm.get("objects") != 4000 for arm in arms)
        or any(arm.get("selection", {}).get("sample_rows") != 4000 for arm in arms)
        or source.get("selection_union", {}).get("sample_rows") != 40_000
        or source.get("samples_mutually_exclusive") is not True
        or source.get("excluded_build_selection_sample_rows") != 100_000
        or source.get("excluded_build_selection_overlap_rows") != 0
        or source.get("bidirectional_thread_order") is not True
        or source.get("source_order") != "path"
        or source.get("max_arm_byte_imbalance_allowed") != 1.10
        or not isinstance(source.get("arm_bytes_max_over_min"), (int, float))
        or source["arm_bytes_max_over_min"] > 1.10
        or set(source.get("median_objects_per_second_by_threads", {}))
        != {"8", "16", "24", "32", "48"}
        or source.get("recommended_threads_by_median") not in {8, 16, 24, 32, 48}
    ):
        raise AnalysisError("formal pilot source-scan shape drift")

    has_s3 = any(value.get("backend") == "mountpoint_s3" for value in artifacts.values())
    expected_labels = {"full100k", "fsx_s0b1", "fsx_s0b4"}
    if has_s3:
        expected_labels.add("s3_s0b4")
    if set(artifacts) != expected_labels:
        raise AnalysisError(
            f"formal pilot artifact label set drift: {sorted(artifacts)}"
        )
    shapes = {
        "full100k": ("fsx_posix", 100_000, 4 * (1 << 30)),
        "fsx_s0b1": ("fsx_posix", 20_000, 1 * (1 << 30)),
        "fsx_s0b4": ("fsx_posix", 20_000, 4 * (1 << 30)),
    }
    if has_s3:
        shapes["s3_s0b4"] = ("mountpoint_s3", 20_000, 4 * (1 << 30))
    source_runtime = source.get("runtime")
    fsx_builder_runtimes: list[dict[str, Any]] = []
    for label, (backend, rows, blob_bytes) in shapes.items():
        manifest = artifacts[label]
        if (
            manifest.get("backend") != backend
            or manifest.get("selection", {}).get("sample_rows") != rows
            or manifest.get("layout", {}).get("blob_target_bytes") != blob_bytes
            or manifest.get("layout", {}).get("stripe", {}).get("count") != 0
            or manifest.get("source_read", {}).get("source_order") != "path"
            or manifest.get("source_read", {}).get("workers")
            != source.get("recommended_threads_by_median")
        ):
            raise AnalysisError(f"formal pilot artifact shape drift: {label}")
        if backend == "fsx_posix":
            if manifest.get("builder_runtime") != source_runtime:
                raise AnalysisError(f"source/FSx builder runtime drift: {label}")
            fsx_builder_runtimes.append(manifest["builder_runtime"])
        else:
            if manifest.get("source_read", {}).get("max_total_source_bytes") != 4 * (1 << 30):
                raise AnalysisError("S3 pilot total-source-byte cap drift")
    if not fsx_builder_runtimes or any(
        value != fsx_builder_runtimes[0] for value in fsx_builder_runtimes
    ):
        raise AnalysisError("FSx artifacts were not built in one exact runtime")

    if build_receipt is None:
        raise AnalysisError("formal pilot build receipt is required")
    if (
        build_receipt.get("source_scan_raw_sha256") != source_raw_sha256
        or build_receipt.get("source_scan_report_sha256")
        != source.get("report_sha256")
        or build_receipt.get("selected_source_threads")
        != source.get("recommended_threads_by_median")
        or build_receipt.get("stripe_scope") != "default_stripe_count_0_only"
    ):
        raise AnalysisError("build receipt source/stripe binding drift")
    receipt_artifacts = build_receipt.get("artifacts")
    if not isinstance(receipt_artifacts, list) or len(receipt_artifacts) != 3:
        raise AnalysisError("build receipt must bind exactly three FSx artifacts")
    receipt_by_root = {}
    for entry in receipt_artifacts:
        if not isinstance(entry, dict):
            raise AnalysisError("malformed build receipt artifact entry")
        root = Path(entry.get("root", "")).resolve(strict=True)
        if root in receipt_by_root:
            raise AnalysisError("duplicate build receipt artifact root")
        receipt_by_root[root] = entry
    expected_fsx_roots = {
        artifact_roots[label] for label in ("full100k", "fsx_s0b1", "fsx_s0b4")
    }
    fsx_jobtags = {root.parent.name for root in expected_fsx_roots}
    if set(receipt_by_root) != expected_fsx_roots:
        raise AnalysisError("build receipt artifact root set drift")
    if len(fsx_jobtags) != 1 or build_receipt.get("jobtag") not in fsx_jobtags:
        raise AnalysisError("build receipt jobtag/artifact namespace drift")
    if type(build_receipt.get("lfs_available")) is not bool:
        raise AnalysisError("build receipt lfs availability type drift")
    for label in ("full100k", "fsx_s0b1", "fsx_s0b4"):
        entry = receipt_by_root[artifact_roots[label]]
        if (
            entry.get("manifest_raw_sha256") != artifact_raw_sha256[label]
            or entry.get("manifest_sha256") != artifacts[label]["manifest_sha256"]
            or entry.get("sample_rows") != artifacts[label]["selection"]["sample_rows"]
        ):
            raise AnalysisError(f"build receipt artifact binding drift: {label}")

    if has_s3:
        if s3_build_receipt is None:
            raise AnalysisError("S3 artifact present without S3 build receipt")
        fsx_root = artifact_roots["fsx_s0b4"]
        s3_root = artifact_roots["s3_s0b4"]
        if (
            s3_build_receipt.get("status") != "BYTE_IDENTICAL"
            or Path(s3_build_receipt.get("fsx_root", "")).resolve(strict=True)
            != fsx_root
            or Path(s3_build_receipt.get("s3_root", "")).resolve(strict=True)
            != s3_root
            or s3_build_receipt.get("fsx_manifest_raw_sha256")
            != artifact_raw_sha256["fsx_s0b4"]
            or s3_build_receipt.get("s3_manifest_raw_sha256")
            != artifact_raw_sha256["s3_s0b4"]
            or s3_build_receipt.get("fsx_manifest_sha256")
            != artifacts["fsx_s0b4"]["manifest_sha256"]
            or s3_build_receipt.get("s3_manifest_sha256")
            != artifacts["s3_s0b4"]["manifest_sha256"]
            or s3_build_receipt.get("source_build_jobtag") != fsx_root.parent.name
            or s3_build_receipt.get("s3_build_jobtag") != s3_root.parent.name
        ):
            raise AnalysisError("S3 build receipt binding drift")
    elif s3_build_receipt is not None:
        raise AnalysisError("S3 build receipt supplied without an S3 artifact")

    expected_replay_labels = {
        f"{label}_{suffix}"
        for label in expected_labels
        for suffix in ("capacity", "integrity")
    }
    if {entry["label"] for entry in replay_inputs} != expected_replay_labels:
        raise AnalysisError("formal pilot replay label set drift")
    replay_runtimes: list[dict[str, Any]] = []
    for label, values in replays_by_artifact.items():
        if len(values) != 2:
            raise AnalysisError(f"artifact replay count drift: {label}")
        summarize_replays(values)
        for value in values:
            if value.get("workers") != 16 or len(value.get("repeats", [])) != 1:
                raise AnalysisError(f"formal pilot replay worker/repeat drift: {label}")
            if value.get("artifact_sample_rows") != artifacts[label]["selection"]["sample_rows"]:
                raise AnalysisError(f"formal pilot replay sample binding drift: {label}")
            replay_runtimes.append(value.get("replay_runtime"))
        authority = next(value for value in values if value["arm_role"] == "capacity_authority")
        if authority.get("replay_selection", {}).get("selected_segments", 0) < 64:
            raise AnalysisError(f"cold-region authority too small: {label}")
    if not replay_runtimes or any(
        value != replay_runtimes[0] for value in replay_runtimes
    ):
        raise AnalysisError("replay reports were not captured in one exact runtime")


def validate_live_jobtag_assignments(
    *,
    build_job_path: Path,
    s3_build_job_path: Path | None,
    replay_job_path: Path,
    analyze_job_path: Path,
    artifacts: dict[str, dict[str, Any]],
    artifact_roots: dict[str, Path],
    replay_inputs: list[dict[str, Any]],
) -> None:
    def assignment(path: Path, name: str, expected: str) -> None:
        text = path.read_text(encoding="utf-8")
        matches = re.findall(rf"(?m)^\s{{2}}{re.escape(name)}=([^\r\n]*)$", text)
        if matches != [expected]:
            raise AnalysisError(
                f"live {path.name} assignment drift: {name}={matches!r}, expected={expected!r}"
            )

    fsx_jobtags = {
        artifact_roots[label].parent.name
        for label, manifest in artifacts.items()
        if manifest.get("backend") == "fsx_posix"
    }
    if len(fsx_jobtags) != 1:
        raise AnalysisError("cannot derive one explicit FSx build jobtag")
    fsx_jobtag = next(iter(fsx_jobtags))
    replay_jobtags = {Path(entry["path"]).resolve(strict=True).parent.name for entry in replay_inputs}
    if len(replay_jobtags) != 1:
        raise AnalysisError("cannot derive one explicit replay jobtag")
    replay_jobtag = next(iter(replay_jobtags))
    has_s3 = "s3_s0b4" in artifacts
    assignment(replay_job_path, "SOURCE_BUILD_JOBTAG", fsx_jobtag)
    assignment(replay_job_path, "INCLUDE_S3", "true" if has_s3 else "false")
    assignment(analyze_job_path, "SOURCE_BUILD_JOBTAG", fsx_jobtag)
    assignment(analyze_job_path, "SOURCE_REPLAY_JOBTAG", replay_jobtag)
    assignment(analyze_job_path, "INCLUDE_S3", "true" if has_s3 else "false")
    if has_s3:
        if s3_build_job_path is None:
            raise AnalysisError("S3 artifact has no live S3 build job path")
        s3_jobtag = artifact_roots["s3_s0b4"].parent.name
        assignment(s3_build_job_path, "ENABLE_S3_WRITE", "true")
        assignment(s3_build_job_path, "SOURCE_BUILD_JOBTAG", fsx_jobtag)
        assignment(replay_job_path, "SOURCE_S3_BUILD_JOBTAG", s3_jobtag)
        assignment(analyze_job_path, "SOURCE_S3_BUILD_JOBTAG", s3_jobtag)
    else:
        assignment(replay_job_path, "SOURCE_S3_BUILD_JOBTAG", "")
        assignment(analyze_job_path, "SOURCE_S3_BUILD_JOBTAG", "")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    closure: dict[Path, str] = {}

    def bind_file(path: Path) -> tuple[Path, str]:
        resolved = path.resolve(strict=True)
        checksum = stable_file_sha256(resolved)
        previous = closure.setdefault(resolved, checksum)
        if previous != checksum:
            raise AnalysisError(f"evidence file changed during initial binding: {resolved}")
        return resolved, checksum

    analyzer_path = Path(__file__).resolve(strict=True)
    analyzer_path, analyzer_sha = bind_file(analyzer_path)
    job_spec_path, analyzer_job_sha = bind_file(args.job_spec)
    runner_path, live_runner_sha = bind_file(args.runner)
    build_job_path, live_build_job_sha = bind_file(args.build_job_spec)
    replay_job_path, live_replay_job_sha = bind_file(args.replay_job_spec)
    if args.s3_build_job_spec is not None:
        s3_build_job_path, live_s3_build_job_sha = bind_file(args.s3_build_job_spec)
    else:
        s3_build_job_path = None
        live_s3_build_job_sha = None
    code = {
        "analyzer_path": str(analyzer_path),
        "analyzer_sha256": analyzer_sha,
        "job_spec_path": str(job_spec_path),
        "job_spec_sha256": analyzer_job_sha,
        "runner_path": str(runner_path),
        "runner_sha256": live_runner_sha,
        "build_job_spec_path": str(build_job_path),
        "build_job_spec_sha256": live_build_job_sha,
        "replay_job_spec_path": str(replay_job_path),
        "replay_job_spec_sha256": live_replay_job_sha,
        "s3_build_job_spec_path": (
            str(s3_build_job_path) if s3_build_job_path is not None else None
        ),
        "s3_build_job_spec_sha256": live_s3_build_job_sha,
    }
    source, source_raw_sha = load_json(args.source_scan)
    source_path, rebound_source_sha = bind_file(args.source_scan)
    if rebound_source_sha != source_raw_sha:
        raise AnalysisError("source-scan raw SHA changed during binding")
    build_receipt = None
    build_receipt_raw_sha = None
    if args.build_receipt is not None:
        build_receipt, build_receipt_raw_sha = load_json(args.build_receipt)
        _, rebound = bind_file(args.build_receipt)
        if rebound != build_receipt_raw_sha:
            raise AnalysisError("build receipt changed during binding")
        if (
            build_receipt.get("schema_version")
            != "offline_gpic_blob_build_receipt_v1"
            or not valid_sha(build_receipt.get("receipt_sha256"))
            or receipt_hash(build_receipt) != build_receipt["receipt_sha256"]
        ):
            raise AnalysisError("build receipt self-hash/schema drift")
    s3_build_receipt = None
    s3_build_receipt_raw_sha = None
    if args.s3_build_receipt is not None:
        s3_build_receipt, s3_build_receipt_raw_sha = load_json(args.s3_build_receipt)
        _, rebound = bind_file(args.s3_build_receipt)
        if rebound != s3_build_receipt_raw_sha:
            raise AnalysisError("S3 build receipt changed during binding")
        if (
            s3_build_receipt.get("schema_version")
            != "offline_gpic_blob_s3_build_receipt_v1"
            or not valid_sha(s3_build_receipt.get("receipt_sha256"))
            or receipt_hash(s3_build_receipt)
            != s3_build_receipt["receipt_sha256"]
        ):
            raise AnalysisError("S3 build receipt self-hash/schema drift")
    source_checksum = source.get("report_sha256")
    if (
        source.get("schema_version") != "offline_gpic_blob_source_scan_v1"
        or source.get("status") != "PASS"
        or not valid_sha(source_checksum)
        or report_hash(source) != source_checksum
        or source.get("pilot_only") is not True
        or source.get("production_selection_qualified") is not False
    ):
        raise AnalysisError("source-scan report contract drift")
    validate_runner_job_binding(
        source.get("code"),
        label="source scan",
        runner_path=runner_path,
        runner_sha256=live_runner_sha,
        job_spec_path=build_job_path,
        job_spec_sha256=live_build_job_sha,
    )

    artifacts: dict[str, dict[str, Any]] = {}
    artifact_raw_sha: dict[str, str] = {}
    artifact_roots: dict[str, Path] = {}
    for label, root in args.artifact:
        if label in artifacts:
            raise AnalysisError(f"duplicate artifact label: {label}")
        manifest, raw_sha = load_artifact(root)
        manifest_path, rebound_manifest_sha = bind_file(root / "manifest.json")
        if rebound_manifest_sha != raw_sha:
            raise AnalysisError(f"artifact manifest changed during binding: {label}")
        bind_file(root / "_SUCCESS")
        artifacts[label] = manifest
        artifact_raw_sha[label] = raw_sha
        artifact_roots[label] = root.resolve(strict=True)
    for label, manifest in artifacts.items():
        if manifest["backend"] == "fsx_posix":
            expected_job_path = build_job_path
            expected_job_sha = live_build_job_sha
        elif manifest["backend"] == "mountpoint_s3":
            if s3_build_job_path is None or live_s3_build_job_sha is None:
                raise AnalysisError("S3 artifact present without live S3 build job spec")
            expected_job_path = s3_build_job_path
            expected_job_sha = live_s3_build_job_sha
        else:
            raise AnalysisError(f"unsupported artifact backend: {label}")
        validate_runner_job_binding(
            manifest.get("code"),
            label=f"artifact {label}",
            runner_path=runner_path,
            runner_sha256=live_runner_sha,
            job_spec_path=expected_job_path,
            job_spec_sha256=expected_job_sha,
        )
        if (
            manifest["catalog"]["catalog_hash"] != source["catalog"]["catalog_hash"]
            or manifest["catalog"]["gpic"]["sha256"]
            != source["catalog"]["gpic"]["sha256"]
        ):
            raise AnalysisError(f"source-scan/artifact catalog binding drift: {label}")

    replays_by_artifact: dict[str, list[dict[str, Any]]] = {label: [] for label in artifacts}
    replay_inputs: list[dict[str, Any]] = []
    for replay_label, path in args.replay:
        value, raw_sha = load_json(path)
        _, rebound_replay_sha = bind_file(path)
        if rebound_replay_sha != raw_sha:
            raise AnalysisError(f"replay report changed during binding: {replay_label}")
        checksum = value.get("report_sha256")
        if (
            value.get("schema_version") != "offline_gpic_blob_replay_v1"
            or not valid_sha(checksum)
            or report_hash(value) != checksum
            or value.get("pilot_only") is not True
            or value.get("production_selection_qualified") is not False
        ):
            raise AnalysisError(f"replay report contract drift: {replay_label}")
        validate_runner_job_binding(
            value.get("code"),
            label=f"replay {replay_label}",
            runner_path=runner_path,
            runner_sha256=live_runner_sha,
            job_spec_path=replay_job_path,
            job_spec_sha256=live_replay_job_sha,
        )
        if (
            value.get("pre_timing_blob_sha256_verified") is not False
            or value.get("pre_timing_validation")
            != "manifest_index_selection_sha256_and_blob_size_only"
        ):
            raise AnalysisError(f"replay cache-integrity contract drift: {replay_label}")
        if value.get("direct_io") != {
            "o_direct_used": False,
            "o_direct_required": False,
            "status": "optional_diagnostic_not_yet_jcc_validated",
        }:
            raise AnalysisError(f"unvalidated O_DIRECT became a replay dependency: {replay_label}")
        role = value.get("arm_role")
        mode = value.get("selection_mode")
        selection_contract = value.get("replay_selection")
        if not isinstance(selection_contract, dict):
            raise AnalysisError(f"replay selection contract missing: {replay_label}")
        if role == "capacity_authority":
            if (
                value.get("capacity_authority") is not True
                or value.get("arm_sequence_index") != 0
                or mode != "cold_unique_region"
                or value.get("timed_content_validation")
                != "every_selected_cold_region_segment_exact_pread_and_sha256"
                or selection_contract.get("cold_unique_region") is not True
                or selection_contract.get("full_coverage") is not False
                or selection_contract.get("region_bytes") != 16 * (1 << 20)
                or selection_contract.get("potential_readahead_guard_bytes")
                != 8 * (1 << 20)
                or selection_contract.get("get_start_alignment_bytes")
                != 8 * (1 << 20)
                or selection_contract.get("max_complete_segments_per_region") != 1
                or selection_contract.get("selected_region_count")
                != selection_contract.get("selected_segments")
            ):
                raise AnalysisError(f"cold capacity contract drift: {replay_label}")
        elif role == "integrity_warm":
            if (
                value.get("capacity_authority") is not False
                or value.get("arm_sequence_index") != 1
                or mode != "full_coverage"
                or value.get("timed_content_validation")
                != "every_index_segment_exact_pread_and_sha256"
                or selection_contract.get("cold_unique_region") is not False
                or selection_contract.get("full_coverage") is not True
            ):
                raise AnalysisError(f"warm integrity contract drift: {replay_label}")
        else:
            raise AnalysisError(f"unknown replay arm role: {replay_label}")
        matches = [
            label
            for label, root in artifact_roots.items()
            if Path(value.get("artifact_root", "")).resolve(strict=True) == root
        ]
        if len(matches) != 1:
            raise AnalysisError(f"replay does not bind exactly one artifact: {replay_label}")
        artifact_label = matches[0]
        if value.get("artifact_manifest_sha256") != artifacts[artifact_label]["manifest_sha256"]:
            raise AnalysisError(f"replay/artifact manifest mismatch: {replay_label}")
        if value.get("backend") != artifacts[artifact_label]["backend"]:
            raise AnalysisError(f"replay/artifact backend mismatch: {replay_label}")
        expected_production_status = (
            "INCONCLUSIVE_FOR_PRODUCTION"
            if artifacts[artifact_label]["backend"] == "mountpoint_s3"
            and artifacts[artifact_label]["selection"]["sample_rows"] < 100_000
            else "PILOT_DIAGNOSTIC_ONLY"
        )
        if value.get("production_capacity_status") != expected_production_status:
            raise AnalysisError(f"replay production-status drift: {replay_label}")
        if args.require_distinct_runtime_hosts:
            build_runtime = artifacts[artifact_label].get("builder_runtime", {})
            replay_runtime = value.get("replay_runtime", {})
            if (
                not isinstance(build_runtime, dict)
                or not isinstance(replay_runtime, dict)
                or not build_runtime.get("hostname")
                or not replay_runtime.get("hostname")
                or build_runtime["hostname"] == replay_runtime["hostname"]
            ):
                raise AnalysisError(f"builder/replay pod separation not proven: {replay_label}")
            if (
                artifacts[artifact_label].get("code", {}).get("job_spec_sha256")
                == value.get("code", {}).get("job_spec_sha256")
            ):
                raise AnalysisError(f"builder/replay job-spec separation not proven: {replay_label}")
            build_boot = build_runtime.get("boot_id_sha256")
            replay_boot = replay_runtime.get("boot_id_sha256")
            if (
                not valid_sha(build_boot)
                or not valid_sha(replay_boot)
                or build_runtime.get("boot_id_raw_published") is not False
                or replay_runtime.get("boot_id_raw_published") is not False
                or build_boot == replay_boot
            ):
                raise AnalysisError(
                    f"builder/replay physical cache-host separation not proven: {replay_label}"
                )
            build_node = build_runtime.get("physical_node_name_sha256")
            replay_node = replay_runtime.get("physical_node_name_sha256")
            if build_node and replay_node and build_node == replay_node:
                raise AnalysisError(f"builder/replay reused the observed physical node: {replay_label}")
        for repeat in value.get("repeats", []):
            expected_objects = (
                artifacts[artifact_label]["selection"]["sample_rows"]
                if mode == "full_coverage"
                else selection_contract.get("selected_segments")
            )
            if (
                repeat.get("objects") != expected_objects
                or repeat.get("requested_bytes") != repeat.get("returned_bytes")
                or repeat.get("userspace_read_amplification") != 1.0
            ):
                raise AnalysisError(f"replay accounting drift: {replay_label}")
        if not value.get("repeats"):
            raise AnalysisError(f"replay has no timed arms: {replay_label}")
        replays_by_artifact[artifact_label].append(value)
        replay_inputs.append(
            {
                "label": replay_label,
                "path": str(path),
                "raw_sha256": raw_sha,
                "report_sha256": checksum,
                "artifact": artifact_label,
            }
        )
    missing_replay = sorted(label for label, values in replays_by_artifact.items() if not values)
    if missing_replay:
        raise AnalysisError(f"artifacts without random-pread replay: {missing_replay}")
    if args.enforce_pilot_shape:
        validate_pilot_shape(
            source=source,
            source_raw_sha256=source_raw_sha,
            artifacts=artifacts,
            artifact_raw_sha256=artifact_raw_sha,
            artifact_roots=artifact_roots,
            replays_by_artifact=replays_by_artifact,
            replay_inputs=replay_inputs,
            build_receipt=build_receipt,
            s3_build_receipt=s3_build_receipt,
        )
        validate_live_jobtag_assignments(
            build_job_path=build_job_path,
            s3_build_job_path=s3_build_job_path,
            replay_job_path=replay_job_path,
            analyze_job_path=job_spec_path,
            artifacts=artifacts,
            artifact_roots=artifact_roots,
            replay_inputs=replay_inputs,
        )

    exact_results: list[dict[str, Any]] = []
    for left, right in args.exact_pair:
        if left not in artifacts or right not in artifacts or left == right:
            raise AnalysisError(f"invalid exact pair: {left}={right}")
        left_manifest = artifacts[left]
        right_manifest = artifacts[right]
        if left_manifest["backend"] == right_manifest["backend"]:
            raise AnalysisError(f"exact payload pair must compare distinct backends: {left}={right}")
        bindings = {
            "catalog_hash": (
                left_manifest["catalog"]["catalog_hash"],
                right_manifest["catalog"]["catalog_hash"],
            ),
            "gpic_catalog_sha256": (
                left_manifest["catalog"]["gpic"]["sha256"],
                right_manifest["catalog"]["gpic"]["sha256"],
            ),
            "selection_sha256": (
                left_manifest["selection"]["selection_sha256"],
                right_manifest["selection"]["selection_sha256"],
            ),
            "sample_rows": (
                left_manifest["selection"]["sample_rows"],
                right_manifest["selection"]["sample_rows"],
            ),
            "blob_target_bytes": (
                left_manifest["layout"]["blob_target_bytes"],
                right_manifest["layout"]["blob_target_bytes"],
            ),
            "content_files": (file_identity_vector(left_manifest), file_identity_vector(right_manifest)),
        }
        unequal = sorted(key for key, (left_value, right_value) in bindings.items() if left_value != right_value)
        if unequal:
            raise AnalysisError(f"exact-payload pair differs: {left}={right}: {unequal}")
        exact_results.append(
            {
                "left": left,
                "right": right,
                "status": "BYTE_IDENTICAL",
                "left_backend": left_manifest["backend"],
                "right_backend": right_manifest["backend"],
                "content_files": file_identity_vector(left_manifest),
            }
        )
    if args.enforce_pilot_shape:
        has_s3_artifact = any(
            value.get("backend") == "mountpoint_s3" for value in artifacts.values()
        )
        if has_s3_artifact and (
            len(exact_results) != 1
            or {exact_results[0]["left"], exact_results[0]["right"]}
            != {"fsx_s0b4", "s3_s0b4"}
        ):
            raise AnalysisError("formal S3 pilot requires one exact FSx/S3 payload pair")
        if not has_s3_artifact and exact_results:
            raise AnalysisError("FSx-only formal pilot must not claim an S3 exact pair")

    artifact_summaries: dict[str, Any] = {}
    has_inconclusive_s3 = False
    for label, manifest in artifacts.items():
        replay_summary = summarize_replays(replays_by_artifact[label])
        if (
            manifest["backend"] == "mountpoint_s3"
            and manifest["selection"]["sample_rows"] < 100_000
        ):
            has_inconclusive_s3 = True
            if (
                replay_summary["capacity_authority"]["production_capacity_status"]
                != "INCONCLUSIVE_FOR_PRODUCTION"
            ):
                raise AnalysisError("20k S3 capacity was incorrectly qualified")
        artifact_summaries[label] = {
            "root": str(artifact_roots[label]),
            "manifest_raw_sha256": artifact_raw_sha[label],
            "manifest_sha256": manifest["manifest_sha256"],
            "backend": manifest["backend"],
            "sample_rows": manifest["selection"]["sample_rows"],
            "unique_shards": manifest["selection"]["unique_shards"],
            "extension_counts": manifest["selection"]["extension_counts"],
            "blob_count": len(manifest["files"]["blobs"]),
            "blob_bytes": sum(item["bytes"] for item in manifest["files"]["blobs"]),
            "source_build_objects_per_second": manifest["source_read"]["objects_per_second"],
            "source_build_rate_is_replay_authority": False,
            "replay": replay_summary,
        }

    output = {
        "schema_version": "offline_gpic_blob_pilot_analysis_v1",
        "status": (
            "INCONCLUSIVE_FOR_PRODUCTION"
            if has_inconclusive_s3
            else "DIAGNOSTIC_COMPLETE"
        ),
        "pilot_only": True,
        "production_selection_qualified": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope_note": (
            "This closes pilot byte identity and userspace pread measurements only; "
            "it does not authorize a loader change, full repack, or formal training."
        ),
        "s3_20k_policy": (
            "feasibility_or_reject_only; at least 100k cold-region S3 evidence is required "
            "before any production-capacity candidacy"
        ),
        "distinct_runtime_hosts_required": args.require_distinct_runtime_hosts,
        "cache_host_separation_proof": (
            "distinct_sha256_of_kernel_boot_id_per_builder_replay_pair"
            if args.require_distinct_runtime_hosts
            else "not_required_for_synthetic_static_analysis"
        ),
        "source_scan": {
            "path": str(args.source_scan),
            "raw_sha256": source_raw_sha,
            "report_sha256": source_checksum,
            "recommended_threads_by_median": source["recommended_threads_by_median"],
            "median_objects_per_second_by_threads": source["median_objects_per_second_by_threads"],
            "full_catalog_hours_one_worker_idealized": source["full_catalog_hours_one_worker_idealized"],
            "full_catalog_hours_eight_workers_idealized": source["full_catalog_hours_eight_workers_idealized"],
        },
        "artifacts": artifact_summaries,
        "exact_payload_pairs": exact_results,
        "replay_inputs": replay_inputs,
        "receipts": {
            "build": (
                {
                    "path": str(args.build_receipt),
                    "raw_sha256": build_receipt_raw_sha,
                    "receipt_sha256": build_receipt["receipt_sha256"],
                }
                if build_receipt is not None
                else None
            ),
            "s3_build": (
                {
                    "path": str(args.s3_build_receipt),
                    "raw_sha256": s3_build_receipt_raw_sha,
                    "receipt_sha256": s3_build_receipt["receipt_sha256"],
                }
                if s3_build_receipt is not None
                else None
            ),
        },
        "limitations": [
            "Only the unique first cold-region arm is capacity authority; later full-coverage reads are warm integrity evidence and are never averaged into capacity.",
            "A separate JCC pod and distinct hashed kernel boot-id are enforced; raw boot-id is never published.",
            "userspace read amplification is exact returned/requested bytes; physical backend fetch amplification is not observable here.",
            "the 20k writable-S3 artifact is feasibility/reject-only; a separate >=100k S3 cold-region run is required before S3 can become a production-capacity candidate.",
            "small-layout build rates may be source-cache-warm because their 20k selection is a subset of the 100k build; only disjoint source-scan arms rank source concurrency.",
            "source-scan object payloads are disjoint, but shared mount metadata caches cannot be forcibly cleared and remain a stated limitation.",
        ],
        "code": code,
        "evidence_closure": [
            {"path": str(path), "sha256": checksum}
            for path, checksum in sorted(closure.items(), key=lambda item: str(item[0]))
        ],
    }
    for path, checksum in sorted(closure.items(), key=lambda item: str(item[0])):
        if stable_file_sha256(path) != checksum:
            raise AnalysisError(f"bound evidence changed during analysis closure: {path}")
    output["analysis_sha256"] = canonical_sha(output)
    destination = args.output.absolute()
    if not str(destination).startswith("/fsx/"):
        raise AnalysisError("analysis evidence output must be on FSx")
    if destination.exists() or destination.is_symlink():
        raise AnalysisError("analysis output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp.{os.getpid()}"
    try:
        with temporary.open("xb") as handle:
            payload = canonical_bytes(output)
            written = handle.write(payload)
            if written != len(payload):
                raise AnalysisError("short analysis output write")
            handle.flush()
            os.fsync(handle.fileno())
        # Same-directory link publication is atomic and refuses an existing
        # destination; analysis evidence is always placed on FSx, never S3.
        os.link(temporary, destination)
        temporary.unlink()
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"OFFLINE_GPIC_BLOB_PILOT_ANALYSIS_PASS output={destination}", flush=True)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scan", type=Path, required=True)
    parser.add_argument("--artifact", type=labeled_path, action="append", required=True)
    parser.add_argument("--replay", type=labeled_path, action="append", required=True)
    parser.add_argument("--exact-pair", type=exact_pair, action="append", default=[])
    parser.add_argument("--require-distinct-runtime-hosts", action="store_true")
    parser.add_argument("--enforce-pilot-shape", action="store_true")
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--build-job-spec", type=Path, required=True)
    parser.add_argument("--s3-build-job-spec", type=Path)
    parser.add_argument("--replay-job-spec", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--s3-build-receipt", type=Path)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        analyze(parse_args())
    except BaseException as exc:
        print(
            f"OFFLINE_GPIC_BLOB_PILOT_ANALYSIS_FATAL class={type(exc).__name__}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise SystemExit(1)
