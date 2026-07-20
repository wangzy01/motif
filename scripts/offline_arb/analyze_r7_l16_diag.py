#!/usr/bin/env python3
"""Strict evidence and performance gate for the r7-L16 2x8 diagnostic."""

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
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from training.dataline_offline.planner_v1 import OfflinePlan


TELEMETRY_SCHEMA = "offline_arb_io_profile_v1"
AGGREGATE_SCHEMA = "offline_arb_r7_l16_telemetry_aggregate_v1"
MANIFEST_SCHEMA = "offline_arb_r7_l16_node_manifest_v1"
ANALYSIS_SCHEMA = "offline_arb_r7_l16_analysis_v1"
WORLD_SIZE = 16
LOCAL_WORLD_SIZE = 8
PREFETCH_STEPS = 4
MAX_STEPS = 192
STEADY_METRIC_STEPS = range(41, 193)
STEADY_WINDOW_STARTS = range(40, 192, 4)
FORMAL_STEPS = 166_667
TARGET_SECONDS_PER_STEP = 2.592
MIN_CAPACITY_RATIO = 1.0 / 0.9
ALLOWED_RECORD_TYPES = {
    "identity",
    "window",
    "wait",
    "state",
    "closure",
}
SHA_RE = re.compile(r"[0-9a-f]{64}")
RANK_FILE_RE = re.compile(r"io_profile\.rank(?P<rank>[0-9]{4})\.jsonl")
TRAIN_LOG_RE = re.compile(
    r"train\.(?P<jobtag>[A-Za-z0-9_.-]+)\.rR7\.n0\."
    r"(?P<stamp>[0-9]{8}T[0-9]{6}Z)\.log"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing, non-regular, or a symlink: {path}")


def read_object(path: Path, label: str) -> dict[str, Any]:
    require_regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


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


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    iterations = [float(row["iter_s"]) for row in rows]
    waits = [float(row["data_wait_s"]) for row in rows]
    wait_fractions = [float(row["data_wait_frac"]) for row in rows]
    result = {
        "rows": len(rows),
        "step_min": min(int(row["step"]) for row in rows),
        "step_max": max(int(row["step"]) for row in rows),
        "iter_mean_s": statistics.fmean(iterations),
        "iter_median_s": statistics.median(iterations),
        "iter_p95_s": percentile(iterations, 0.95),
        "iter_max_s": max(iterations),
        "iter_gt_5s": sum(value > 5.0 for value in iterations),
        "data_wait_mean_s": statistics.fmean(waits),
        "data_wait_p95_s": percentile(waits, 0.95),
        "data_wait_frac_mean": statistics.fmean(wait_fractions),
        "data_wait_time_weighted_frac": sum(waits) / sum(iterations),
    }
    result["iter_gt_5s_fraction"] = result["iter_gt_5s"] / result["rows"]
    result["projected_days_384m_samples"] = (
        result["iter_mean_s"] * FORMAL_STEPS / 86_400.0
    )
    return result


def load_metrics(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_regular(path, "rank-0 metrics")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError(f"blank metrics line {line_number}")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed metrics JSON line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"metrics line {line_number} is not an object")
            step = strict_int(value.get("step"), f"metrics[{line_number}].step", minimum=1)
            iter_s = number(value.get("iter_s"), f"metrics[{line_number}].iter_s", positive=True)
            wait_s = number(value.get("data_wait_s"), f"metrics[{line_number}].data_wait_s")
            wait_frac = number(
                value.get("data_wait_frac"),
                f"metrics[{line_number}].data_wait_frac",
            )
            if wait_s > iter_s + 0.01 or wait_frac > 1.0:
                raise ValueError(f"impossible data-wait values at metrics line {line_number}")
            if abs(wait_frac - wait_s / iter_s) > 0.02:
                raise ValueError(f"inconsistent data-wait fraction at metrics line {line_number}")
            rows.append(
                {
                    "step": step,
                    "iter_s": iter_s,
                    "data_wait_s": wait_s,
                    "data_wait_frac": wait_frac,
                }
            )
    observed = [row["step"] for row in rows]
    expected = list(range(1, MAX_STEPS + 1))
    if observed != expected:
        raise ValueError(
            f"metrics must be exact steps 1..{MAX_STEPS}; "
            f"rows={len(observed)}, first={observed[:3]}, last={observed[-3:]}"
        )
    steady = [row for row in rows if row["step"] in STEADY_METRIC_STEPS]
    if [row["step"] for row in steady] != list(STEADY_METRIC_STEPS):
        raise ValueError("steady metrics do not exactly cover steps 41..192")
    return rows, metric_summary(steady)


def validate_evidence(aggregate_path: Path) -> dict[str, Any]:
    aggregate = read_object(aggregate_path, "dual-node aggregate")
    if aggregate.get("schema_version") != AGGREGATE_SCHEMA:
        raise ValueError("unsupported telemetry aggregate schema")
    if aggregate.get("status") != "PASS":
        raise ValueError(f"telemetry aggregate is not PASS: {aggregate.get('status')!r}")
    jobtag = aggregate.get("jobtag")
    nonce = aggregate.get("nonce")
    if not isinstance(jobtag, str) or not jobtag or "/" in jobtag:
        raise ValueError(f"invalid aggregate jobtag: {jobtag!r}")
    if not isinstance(nonce, str) or SHA_RE.fullmatch(nonce) is None:
        raise ValueError("invalid aggregate nonce")
    if aggregate_path.name != f"telemetry.{jobtag}.aggregate.json":
        raise ValueError("aggregate filename is not bound to its JOBTAG")
    if aggregate.get("expected_nodes") != 2 or aggregate.get("expected_ranks") != WORLD_SIZE:
        raise ValueError("aggregate topology is not exactly two nodes / sixteen ranks")
    nodes = aggregate.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 2:
        raise ValueError("aggregate must bind exactly two node manifests")

    evidence_root = aggregate_path.parent / f"telemetry.{jobtag}"
    rank_paths: dict[int, Path] = {}
    rank_hosts: dict[int, str] = {}
    manifests: list[dict[str, Any]] = []
    for expected_node in range(2):
        matches = [value for value in nodes if isinstance(value, dict) and value.get("node_rank") == expected_node]
        if len(matches) != 1:
            raise ValueError(f"aggregate node {expected_node} entry is missing/duplicated")
        node = matches[0]
        expected_manifest = evidence_root / f"node{expected_node}" / "node_manifest.json"
        declared_manifest = node.get("manifest")
        if not isinstance(declared_manifest, str) or Path(declared_manifest) != expected_manifest:
            raise ValueError(f"node {expected_node} manifest path is not canonical")
        require_regular(expected_manifest, f"node {expected_node} manifest")
        manifest_sha = sha256_file(expected_manifest)
        if node.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"node {expected_node} aggregate/manifest SHA mismatch")
        if (
            node.get("status") != "PASS"
            or node.get("train_rc") != 0
            or node.get("tee_rc") != 0
            or node.get("collect_rc") != 0
            or node.get("rank_files") != LOCAL_WORLD_SIZE
        ):
            raise ValueError(f"node {expected_node} aggregate close is not clean")

        manifest = read_object(expected_manifest, f"node {expected_node} manifest")
        if (
            manifest.get("schema_version") != MANIFEST_SCHEMA
            or manifest.get("status") != "PASS"
            or manifest.get("jobtag") != jobtag
            or manifest.get("nonce") != nonce
            or manifest.get("node_rank") != expected_node
            or manifest.get("train_rc") != 0
            or manifest.get("tee_rc") != 0
            or manifest.get("expected_rank_files") != LOCAL_WORLD_SIZE
        ):
            raise ValueError(f"node {expected_node} manifest identity/status mismatch")
        host = manifest.get("host")
        if (
            not isinstance(host, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+", host) is None
            or not host.startswith(f"{jobtag}-")
        ):
            raise ValueError(f"node {expected_node} manifest host is malformed")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != LOCAL_WORLD_SIZE:
            raise ValueError(f"node {expected_node} manifest must declare eight files")
        expected_ranks = set(
            range(expected_node * LOCAL_WORLD_SIZE, (expected_node + 1) * LOCAL_WORLD_SIZE)
        )
        observed_ranks: set[int] = set()
        for entry in files:
            if not isinstance(entry, dict):
                raise ValueError(f"node {expected_node} manifest file entry is malformed")
            name = entry.get("name")
            match = RANK_FILE_RE.fullmatch(name) if isinstance(name, str) else None
            if match is None:
                raise ValueError(f"unsafe/unexpected rank telemetry filename: {name!r}")
            rank = int(match.group("rank"))
            if rank in observed_ranks:
                raise ValueError(f"duplicate rank telemetry file for rank {rank}")
            path = expected_manifest.parent / name
            require_regular(path, f"rank {rank} telemetry")
            size = strict_int(entry.get("bytes"), f"rank {rank} manifest bytes", minimum=1)
            if path.stat().st_size != size:
                raise ValueError(f"rank {rank} telemetry byte-size mismatch")
            expected_sha = entry.get("sha256")
            if not isinstance(expected_sha, str) or SHA_RE.fullmatch(expected_sha) is None:
                raise ValueError(f"rank {rank} telemetry manifest SHA is malformed")
            if sha256_file(path) != expected_sha:
                raise ValueError(f"rank {rank} telemetry SHA mismatch")
            observed_ranks.add(rank)
            rank_paths[rank] = path
            rank_hosts[rank] = host
        if observed_ranks != expected_ranks:
            raise ValueError(
                f"node {expected_node} rank set mismatch: {sorted(observed_ranks)}"
            )
        manifests.append(
            {
                "node_rank": expected_node,
                "path": str(expected_manifest),
                "sha256": manifest_sha,
            }
        )
    if set(rank_paths) != set(range(WORLD_SIZE)) or set(rank_hosts) != set(range(WORLD_SIZE)):
        raise ValueError("dual-node evidence does not cover global ranks 0..15 exactly")

    preflight_pass = aggregate_path.parent / f"sidecarpreflight.{jobtag}.pass"
    require_regular(preflight_pass, "attempt-bound sidecar PASS")
    pass_fields = preflight_pass.read_text(encoding="ascii").strip().split("|")
    if (
        len(pass_fields) != 10
        or pass_fields[0] != "PASS"
        or pass_fields[1] != jobtag
        or pass_fields[2] != nonce
    ):
        raise ValueError("sidecar PASS is not bound to aggregate JOBTAG/nonce")
    for label, digest in zip(
        (
            "sidecar_header",
            "sidecar_header_file",
            "sidecar_success_file",
            "catalog_header",
            "sidecar_release",
            "preflight_json",
        ),
        pass_fields[4:],
    ):
        if SHA_RE.fullmatch(digest) is None:
            raise ValueError(f"sidecar PASS {label} digest is malformed")
    preflight_json = aggregate_path.parent / f"sidecarpreflight.{jobtag}.json"
    require_regular(preflight_json, "attempt-bound sidecar preflight JSON")
    if sha256_file(preflight_json) != pass_fields[9]:
        raise ValueError("sidecar preflight JSON SHA differs from PASS")
    preflight = read_object(preflight_json, "attempt-bound sidecar preflight JSON")
    if (
        preflight.get("status") != "PASS"
        or preflight.get("mode") != "verify-sidecar"
        or preflight.get("header_sha256") != pass_fields[4]
        or preflight.get("release_digest") != pass_fields[8]
    ):
        raise ValueError("sidecar preflight JSON/PASS identity mismatch")
    return {
        "jobtag": jobtag,
        "nonce": nonce,
        "aggregate_sha256": sha256_file(aggregate_path),
        "manifests": manifests,
        "rank_paths": rank_paths,
        "rank_hosts": rank_hosts,
        "sidecar_header_sha256": pass_fields[4],
        "sidecar_release_digest": pass_fields[8],
    }


def validate_image_detail(
    detail: Any,
    *,
    domain: str,
    n: int,
    image_s: float,
    label: str,
) -> None:
    if not isinstance(detail, dict):
        raise ValueError(f"{label}.image_detail must be an object")
    kind = detail.get("kind")
    if domain == "gpic":
        if kind != "gpic_in_memory" or detail.get("files") != n:
            raise ValueError(f"{label} GPIC detail kind/file count mismatch")
        unique_files = strict_int(
            detail.get("unique_files"), f"{label}.unique_files", minimum=1
        )
        if unique_files > n:
            raise ValueError(f"{label} GPIC unique_files exceeds files")
        strict_int(detail.get("bytes"), f"{label}.bytes", minimum=1)
        if detail.get("workers") != 2:
            raise ValueError(f"{label} GPIC must use exactly two workers per rank")
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
        member_bytes = strict_int(
            detail.get("member_bytes"), f"{label}.member_bytes", minimum=1
        )
        requested = strict_int(
            detail.get("requested_bytes"), f"{label}.requested_bytes", minimum=1
        )
        actual = strict_int(
            detail.get("actual_read_bytes"), f"{label}.actual_read_bytes", minimum=1
        )
        if (
            locator_requests != n
            or unique_tars > unique_locators
            or unique_locators > locator_requests
            or runs > unique_locators
            or member_bytes < requested
            or actual < requested
        ):
            raise ValueError(f"{label} impossible coalesced-tar accounting")
        expected_workers = min(8, unique_tars)
        if detail.get("workers") != expected_workers:
            raise ValueError(f"{label} tar worker accounting mismatch")
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
    if not isinstance(thread_sum, dict):
        raise ValueError(f"{label}.thread_sum_s must be an object")
    for key in required:
        number(thread_sum.get(key), f"{label}.thread_sum_s.{key}")
    operation_max = detail.get("operation_max_s")
    if not isinstance(operation_max, dict):
        raise ValueError(f"{label}.operation_max_s must be an object")
    for key in required:
        maximum = number(operation_max.get(key), f"{label}.operation_max_s.{key}")
        if maximum > float(thread_sum[key]) + 1e-9:
            raise ValueError(f"{label} operation max exceeds thread sum for {key}")
    wall_s = number(detail.get("wall_s"), f"{label}.wall_s", positive=True)
    number(detail.get(maximum_key), f"{label}.{maximum_key}")
    number(detail.get("effective_concurrency"), f"{label}.effective_concurrency")
    if wall_s > image_s + 0.01:
        raise ValueError(f"{label} detailed image wall exceeds outer image time")


def validate_ready_snapshot(value: Any, label: str, domains: set[str]) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    scheduled = strict_int(
        value.get("scheduled_depth_steps"), f"{label}.scheduled_depth_steps", minimum=0
    )
    contiguous = strict_int(
        value.get("contiguous_ready_steps"), f"{label}.contiguous_ready_steps", minimum=0
    )
    if contiguous > scheduled:
        raise ValueError(f"{label} contiguous-ready depth exceeds scheduled depth")
    ready = value.get("domain_ready_items")
    if not isinstance(ready, dict) or set(ready) != domains:
        raise ValueError(f"{label}.domain_ready_items domain set mismatch")
    for domain, count in ready.items():
        strict_int(count, f"{label}.domain_ready_items.{domain}", minimum=0)


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
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith("\n"):
                raise ValueError(f"rank {rank} telemetry has an unterminated final line")
            if not raw.strip():
                raise ValueError(f"rank {rank} telemetry blank line {line_number}")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"rank {rank} malformed telemetry line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"rank {rank} telemetry line {line_number} is not an object")
            if value.get("schema_version") != TELEMETRY_SCHEMA:
                raise ValueError(f"rank {rank} telemetry schema drift at line {line_number}")
            if value.get("sequence") != line_number - 1:
                raise ValueError(f"rank {rank} telemetry sequence gap at line {line_number}")
            if value.get("record_type") not in ALLOWED_RECORD_TYPES:
                raise ValueError(f"rank {rank} unknown telemetry record type at line {line_number}")
            if value.get("rank") != rank:
                raise ValueError(f"rank {rank} telemetry identity drift at line {line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"rank {rank} telemetry is empty")
    identities = [value for value in records if value["record_type"] == "identity"]
    closures = [value for value in records if value["record_type"] == "closure"]
    if len(identities) != 1 or identities[0] is not records[0]:
        raise ValueError(f"rank {rank} identity must be the unique first record")
    if len(closures) != 1 or closures[0] is not records[-1]:
        raise ValueError(f"rank {rank} closure must be the unique final record")
    identity = identities[0]
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
            f"/local/evan/offline_arb_r7_l16/{jobtag}/{nonce}/"
            f"node{rank // LOCAL_WORLD_SIZE}"
        ),
        "telemetry_path": (
            f"/local/evan/offline_arb_r7_l16/{jobtag}/{nonce}/"
            f"node{rank // LOCAL_WORLD_SIZE}/io_profile.rank{rank:04d}.jsonl"
        ),
        "prefetch_steps": PREFETCH_STEPS,
        "gpic_threads_per_node": 16,
        "gpic_workers_per_rank": 2,
        "tar_workers_per_rank": 8,
        "gpic_payload_in_memory": True,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(
                f"rank {rank} identity {key}={identity.get(key)!r}, expected={expected!r}"
            )
    strict_int(identity.get("pid"), f"rank {rank} pid", minimum=1)
    closure = closures[0]
    if closure.get("next_step") != MAX_STEPS:
        raise ValueError(f"rank {rank} closure next_step is not {MAX_STEPS}")
    before = len(records) - 1
    writer_before = strict_int(
        closure.get("records_before_closure"),
        f"rank {rank} writer records before closure",
        minimum=0,
    )
    producer_before = strict_int(
        closure.get("records_enqueued_before_closure"),
        f"rank {rank} producer records before closure",
        minimum=0,
    )
    if (
        writer_before != before
        or producer_before != before
    ):
        raise ValueError(f"rank {rank} closure record-count accounting mismatch")
    if closure.get("telemetry_fatal") is not False:
        raise ValueError(f"rank {rank} closure marks telemetry FATAL")
    dropped_emits = strict_int(
        closure.get("dropped_emits"), f"rank {rank} dropped emits", minimum=0
    )
    runtime_errors = strict_int(
        closure.get("runtime_errors"), f"rank {rank} runtime errors", minimum=0
    )
    if dropped_emits != 0 or runtime_errors != 0:
        raise ValueError(f"rank {rank} closure reports dropped/error telemetry")
    rejected_before = strict_int(
        closure.get("rejected_emits_before_closure"),
        f"rank {rank} rejected emits before closure",
        minimum=0,
    )
    rejected_by_writer = strict_int(
        closure.get("rejected_emits_observed_by_writer"),
        f"rank {rank} rejected emits observed by writer",
        minimum=0,
    )
    if (
        rejected_before != 0
        or rejected_by_writer != 0
        or rejected_before != rejected_by_writer
    ):
        raise ValueError(f"rank {rank} closure reports rejected telemetry")
    number(
        closure.get("producer_enqueue_seconds_before_closure"),
        f"rank {rank} producer enqueue seconds",
    )
    number(
        closure.get("writer_seconds_before_closure"),
        f"rank {rank} writer seconds",
    )

    windows: dict[tuple[int, str], dict[str, Any]] = {}
    for value in records:
        record_type = value["record_type"]
        if record_type == "window":
            start = strict_int(value.get("start"), f"rank {rank} window start", minimum=0)
            stop = strict_int(value.get("stop"), f"rank {rank} window stop", minimum=1)
            n = strict_int(value.get("n"), f"rank {rank} window n", minimum=1)
            domain = value.get("domain")
            if not isinstance(domain, str) or domain not in domains:
                raise ValueError(f"rank {rank} window domain is invalid: {domain!r}")
            if value.get("caption_backend") != "sidecar" or value.get("caption_workers") != 16:
                raise ValueError(f"rank {rank} window caption backend/worker drift")
            if start % PREFETCH_STEPS or stop != start + PREFETCH_STEPS:
                raise ValueError(f"rank {rank} non-canonical window [{start},{stop})")
            key = (start, domain)
            if key in windows:
                raise ValueError(f"rank {rank} duplicate window {key}")
            catalog_s = number(value.get("catalog_s"), f"rank {rank} {key} catalog_s")
            caption_s = number(value.get("caption_s"), f"rank {rank} {key} caption_s")
            image_s = number(value.get("image_s"), f"rank {rank} {key} image_s")
            total_s = number(value.get("total_s"), f"rank {rank} {key} total_s", positive=True)
            if total_s + 1e-6 < catalog_s + caption_s + image_s:
                raise ValueError(f"rank {rank} {key} component times exceed total")
            source_counts = value.get("sub_source_counts")
            if not isinstance(source_counts, dict):
                raise ValueError(f"rank {rank} {key} sub_source_counts is malformed")
            source_total = 0
            for source, count in source_counts.items():
                if not isinstance(source, str):
                    raise ValueError(f"rank {rank} {key} sub_source key is not a string")
                source_total += strict_int(count, f"rank {rank} {key} sub_source count", minimum=1)
            if domain in {"monet_real", "gpic", "synthetic"} and source_total != n:
                raise ValueError(f"rank {rank} {key} sub_source counts do not sum to n")
            validate_image_detail(
                value.get("image_detail"),
                domain=domain,
                n=n,
                image_s=image_s,
                label=f"rank {rank} {key}",
            )
            windows[key] = value
        elif record_type == "state":
            consumed = strict_int(
                value.get("consumed_step"), f"rank {rank} state step", minimum=0
            )
            if value.get("next_step") != consumed + 1:
                raise ValueError(f"rank {rank} state next_step drift")
            validate_ready_snapshot(value, f"rank {rank} state {consumed}", domains)
        elif record_type == "wait":
            wait_seconds = number(
                value.get("condition_wait_s"), f"rank {rank} wait seconds", positive=True
            )
            if wait_seconds <= 2.0:
                raise ValueError(f"rank {rank} emitted a wait record at/below threshold")
            strict_int(
                value.get("condition_wakeups"),
                f"rank {rank} wait wakeups",
                minimum=1,
            )
            expected_items = strict_int(
                value.get("expected_items"), f"rank {rank} wait expected_items", minimum=1
            )
            entry_ready = strict_int(
                value.get("entry_ready_items"), f"rank {rank} wait entry_ready", minimum=0
            )
            exit_ready = strict_int(
                value.get("exit_ready_items"), f"rank {rank} wait exit_ready", minimum=0
            )
            if expected_items != 36 or entry_ready > expected_items or exit_ready != expected_items:
                raise ValueError(f"rank {rank} wait ready-item accounting mismatch")
            validate_ready_snapshot(value.get("entry"), f"rank {rank} wait entry", domains)
            validate_ready_snapshot(value.get("exit"), f"rank {rank} wait exit", domains)

    states = [value["consumed_step"] for value in records if value["record_type"] == "state"]
    if states != list(range(3, MAX_STEPS, 4)):
        raise ValueError(f"rank {rank} state records are not exact consumed steps 3,7,..,191")
    return {
        "records": len(records),
        "record_counts": dict(sorted(Counter(value["record_type"] for value in records).items())),
        "windows": windows,
        "wait_records": [value for value in records if value["record_type"] == "wait"],
    }


class PlanWindowCounts:
    """Independently decode domain counts from plan row IDs, never telemetry n."""

    def __init__(self, plan: OfflinePlan):
        self.plan = plan
        self.domains = tuple(str(value) for value in plan.header["domains"])
        totals = [
            sum(int(row[d]) for row in plan.header["N"])
            for d in range(len(self.domains))
        ]
        running = 0
        self.domain_ends: list[int] = []
        for value in totals:
            running += value
            self.domain_ends.append(running)
        if running != int(plan.header["catalog_size"]):
            raise ValueError("independent plan domain ranges do not close over catalog_size")
        self.cache: dict[tuple[int, int], dict[str, int]] = {}

    def get(self, start: int, rank: int) -> dict[str, int]:
        key = (start, rank)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if start < 0 or start % PREFETCH_STEPS or start + PREFETCH_STEPS > self.plan.config.total_steps:
            raise ValueError(f"invalid plan window [{start},{start + PREFETCH_STEPS})")
        counts = {domain: 0 for domain in self.domains}
        for step in range(start, start + PREFETCH_STEPS):
            _bucket, row_ids = self.plan.batch(step, rank)
            for raw in row_ids.tolist():
                row_id = int(raw)
                domain_index = bisect.bisect_right(self.domain_ends, row_id)
                if domain_index >= len(self.domains) or row_id < 0:
                    raise ValueError(f"plan row_id outside independent domain ranges: {row_id}")
                counts[self.domains[domain_index]] += 1
        if sum(counts.values()) != PREFETCH_STEPS * self.plan.config.per_rank_bs:
            raise ValueError(f"independent plan count does not close for start={start}, rank={rank}")
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
    aggregate_path = Path(args.aggregate)
    metrics_path = Path(args.metrics)
    train_log_path = Path(args.train_log)
    plan_root = Path(args.plan)
    evidence = validate_evidence(aggregate_path)
    if aggregate_path.parent.parent.name != args.run_id:
        raise ValueError("aggregate parent is not bound to --run-id")
    if metrics_path != aggregate_path.parent / "ckpt" / "metrics.jsonl":
        raise ValueError("metrics path is not canonical for this aggregate")
    log_match = TRAIN_LOG_RE.fullmatch(train_log_path.name)
    if (
        train_log_path.parent != aggregate_path.parent
        or log_match is None
        or log_match.group("jobtag") != evidence["jobtag"]
    ):
        raise ValueError("node-0 train log path/name is not bound to aggregate JOBTAG")
    require_regular(train_log_path, "node-0 training log")
    train_log = train_log_path.read_text(encoding="utf-8", errors="strict")
    for token in (
        "[train] step 192/192 ",
        "[train] save_final=false",
        "[train] done at step 192",
        "[train] skip_distributed_teardown=true",
        "[train] force_os_exit_after_train=true",
    ):
        if token not in train_log:
            raise ValueError(f"node-0 training log lacks exact completion token: {token!r}")
    if "OFFLINE_ARB_WINDOW_PROFILE" in train_log:
        raise ValueError("legacy PROFILE_WINDOWS stdout appeared in r7-L16 log")

    plan = OfflinePlan(plan_root, world_size=WORLD_SIZE, per_rank_bs=36, verify_checksums=True)
    if plan.config.total_steps < MAX_STEPS:
        raise ValueError("plan has fewer than 192 steps")
    plan_id = str(plan.header["plan_id"])
    plan_hash = str(plan.header["plan_hash"])
    catalog_hash = str(plan.header["catalog_hash"])
    domains = tuple(str(value) for value in plan.header["domains"])
    expected_counts = PlanWindowCounts(plan)

    telemetry: dict[int, dict[str, Any]] = {}
    for rank, path in sorted(evidence["rank_paths"].items()):
        telemetry[rank] = load_rank_telemetry(
            path,
            rank,
            plan_id=plan_id,
            plan_hash=plan_hash,
            catalog_hash=catalog_hash,
            jobtag=evidence["jobtag"],
            nonce=evidence["nonce"],
            host=evidence["rank_hosts"][rank],
            sidecar_header_sha256=evidence["sidecar_header_sha256"],
            sidecar_release_digest=evidence["sidecar_release_digest"],
            domains=set(domains),
        )

    # Validate every published window, including harmless lookahead windows,
    # against plan-derived counts.  Then prove the full steady window grid.
    for rank, rank_data in telemetry.items():
        for (start, domain), record in rank_data["windows"].items():
            expected = expected_counts.get(start, rank)[domain]
            if record["n"] != expected or expected <= 0:
                raise ValueError(
                    f"rank {rank} window {(start, domain)} telemetry n={record['n']} "
                    f"but independent plan n={expected}"
                )
        for start in STEADY_WINDOW_STARTS:
            counts = expected_counts.get(start, rank)
            for domain in domains:
                present = (start, domain) in rank_data["windows"]
                if present != (counts[domain] > 0):
                    raise ValueError(
                        f"rank {rank} steady window {(start, domain)} presence={present}, "
                        f"independent plan n={counts[domain]}"
                    )

    combined: list[dict[str, Any]] = []
    rank_domain_rates: dict[str, dict[str, float]] = {domain: {} for domain in domains}
    rank_domain_required_rates: dict[str, dict[str, float]] = {
        domain: {} for domain in domains
    }
    rank_domain_capacity_ratios: dict[str, dict[str, float]] = {
        domain: {} for domain in domains
    }
    for start in STEADY_WINDOW_STARTS:
        candidates: list[tuple[float, int, str]] = []
        for rank in range(WORLD_SIZE):
            for domain in domains:
                record = telemetry[rank]["windows"].get((start, domain))
                if record is not None:
                    candidates.append((float(record["total_s"]), rank, domain))
        if not candidates:
            raise ValueError(f"no telemetry candidates for steady window start={start}")
        critical_s, critical_rank, critical_domain = max(candidates)
        combined.append(
            {
                "start": start,
                "critical_s": critical_s,
                "rank": critical_rank,
                "domain": critical_domain,
            }
        )
    critical_values = [value["critical_s"] for value in combined]
    combined_summary = {
        "windows": len(combined),
        "start_min": combined[0]["start"],
        "start_max": combined[-1]["start"],
        "mean_s": statistics.fmean(critical_values),
        "median_s": statistics.median(critical_values),
        "p95_s": percentile(critical_values, 0.95),
        "max_s": max(critical_values),
        "critical_windows": combined,
    }

    for rank in range(WORLD_SIZE):
        for domain in domains:
            images = 0
            seconds = 0.0
            for start in STEADY_WINDOW_STARTS:
                count = expected_counts.get(start, rank)[domain]
                if count:
                    record = telemetry[rank]["windows"][(start, domain)]
                    images += count
                    seconds += float(record["total_s"])
            if images <= 0 or seconds <= 0:
                raise ValueError(f"rank {rank} has no steady throughput evidence for {domain}")
            observed_rate = images / seconds
            target_seconds = (
                len(STEADY_WINDOW_STARTS)
                * PREFETCH_STEPS
                * TARGET_SECONDS_PER_STEP
            )
            required_rate = images / target_seconds
            rank_domain_rates[domain][str(rank)] = observed_rate
            rank_domain_required_rates[domain][str(rank)] = required_rate
            rank_domain_capacity_ratios[domain][str(rank)] = (
                observed_rate / required_rate
            )
    domain_min_rates = {
        domain: min(values.values()) for domain, values in rank_domain_rates.items()
    }
    domain_min_capacity_ratios = {
        domain: min(values.values())
        for domain, values in rank_domain_capacity_ratios.items()
    }

    _all_metrics, steady_metrics = load_metrics(metrics_path)
    gates: list[dict[str, Any]] = []
    add_gate(gates, "metrics_iter_mean_s", steady_metrics["iter_mean_s"], "<=", 2.75)
    add_gate(gates, "metrics_iter_median_s", steady_metrics["iter_median_s"], "<=", 2.70)
    add_gate(gates, "metrics_iter_p95_s", steady_metrics["iter_p95_s"], "<=", 3.50)
    add_gate(
        gates,
        "metrics_iter_max_s",
        steady_metrics["iter_max_s"],
        "<=",
        5.0,
    )
    add_gate(
        gates,
        "metrics_iter_gt_5s_count",
        steady_metrics["iter_gt_5s"],
        "<=",
        0.0,
    )
    add_gate(
        gates,
        "metrics_data_wait_time_weighted_frac",
        steady_metrics["data_wait_time_weighted_frac"],
        "<=",
        0.18,
    )
    add_gate(gates, "combined_critical_mean_s", combined_summary["mean_s"], "<=", 9.3312)
    add_gate(gates, "combined_critical_p95_s", combined_summary["p95_s"], "<=", 20.736)
    add_gate(gates, "combined_critical_max_s", combined_summary["max_s"], "<=", 20.736)
    add_gate(
        gates,
        "gpic_min_rank_capacity_ratio",
        domain_min_capacity_ratios["gpic"],
        ">=",
        MIN_CAPACITY_RATIO,
    )
    add_gate(
        gates,
        "monet_real_min_rank_capacity_ratio",
        domain_min_capacity_ratios["monet_real"],
        ">=",
        MIN_CAPACITY_RATIO,
    )
    failed = [gate["name"] for gate in gates if not gate["pass"]]
    status = "PASS" if not failed else "NO_GO"

    wait_diagnostics = {
        str(rank): {
            "wait_records": len(telemetry[rank]["wait_records"]),
            "wait_seconds": sum(
                float(value["condition_wait_s"])
                for value in telemetry[rank]["wait_records"]
            ),
        }
        for rank in range(WORLD_SIZE)
    }
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "status": status,
        "failed_gates": failed,
        "reasons": (
            ["all_r7_l16_strict_gates_pass"]
            if not failed
            else [f"strict_gate_failed:{name}" for name in failed]
        ),
        "identity": {
            "run_id": args.run_id,
            "jobtag": evidence["jobtag"],
            "nonce": evidence["nonce"],
            "plan_id": plan_id,
            "plan_hash": plan_hash,
            "catalog_hash": catalog_hash,
            "sidecar_header_sha256": evidence["sidecar_header_sha256"],
            "sidecar_release_digest": evidence["sidecar_release_digest"],
            "world_size": WORLD_SIZE,
            "per_rank_bs": 36,
            "max_steps": MAX_STEPS,
            "gpic_threads_per_node": 16,
            "prefetch_steps": PREFETCH_STEPS,
            "profile_windows": False,
            "steady_metric_steps": [41, 192],
            "steady_window_starts": [40, 188, 4],
        },
        "gates": gates,
        "metrics_41_192": steady_metrics,
        "rank0_wait_fraction_reference": {
            "value": steady_metrics["data_wait_time_weighted_frac"],
            "reference_max": 0.18,
            "within_reference": steady_metrics["data_wait_time_weighted_frac"] <= 0.18,
            "hard_gate": True,
            "reason": "time-weighted rank-0 data wait is a strict wall-clock gate",
        },
        "combined_critical_windows": combined_summary,
        "domain_min_rank_img_s": domain_min_rates,
        "rank_domain_img_s": rank_domain_rates,
        "rank_domain_required_img_s": rank_domain_required_rates,
        "domain_min_rank_capacity_ratio": domain_min_capacity_ratios,
        "rank_domain_capacity_ratio": rank_domain_capacity_ratios,
        "wait_diagnostics": wait_diagnostics,
        "telemetry_record_counts": {
            str(rank): telemetry[rank]["record_counts"] for rank in range(WORLD_SIZE)
        },
        "evidence": {
            "aggregate": str(aggregate_path),
            "aggregate_sha256": evidence["aggregate_sha256"],
            "node_manifests": evidence["manifests"],
            "metrics": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "train_log": str(train_log_path),
            "train_log_sha256": sha256_file(train_log_path),
            "plan_header_sha256": sha256_file(plan_root / "header.json"),
            "plan_checksums_sha256": sha256_file(plan_root / "checksums.json"),
        },
        "measurement_caveat": (
            "Detailed local telemetry adds timing/aggregation/JSONL overhead. A near-boundary "
            "result must be confirmed by a telemetry-off run before FORMAL authorization."
        ),
    }


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite analysis output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"stale analysis temporary exists: {temporary}")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--plan", required=True)
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
                print(
                    f"FATAL r7-L16 analyzer output failure: {type(output_exc).__name__}: {output_exc}",
                    file=sys.stderr,
                )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    if args.output:
        atomic_write(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
