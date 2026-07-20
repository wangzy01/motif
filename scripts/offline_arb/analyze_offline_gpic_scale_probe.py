#!/usr/bin/env python3
"""Analyze GPIC source -> local staging -> Mountpoint-S3 scale evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from scripts.offline_arb import run_offline_gpic_blob_pilot as pilot_core
from scripts.offline_arb import run_offline_gpic_scale_probe as probe


ANALYSIS_SCHEMA = "offline_gpic_scale_analysis_v1"
MIB = 1 << 20


class AnalysisError(RuntimeError):
    """Evidence is incomplete, inconsistent, or unsafe to use."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return probe.canonical_json_bytes(value)


def sha256_file(path: Path, block_bytes: int = 8 * MIB) -> str:
    return probe.sha256_file(path, block_bytes)


def percentile(values: Sequence[float], quantile: float) -> float:
    require(bool(values), "empty percentile population")
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    probe.require_regular_file(path)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except Exception as exc:
        raise AnalysisError(f"invalid JSON: {path}") from exc
    require(isinstance(value, dict), f"JSON object required: {path}")
    require(raw == canonical_json_bytes(value), f"non-canonical JSON: {path}")
    return value, raw


def validate_self_hash(value: dict[str, Any], field: str, label: str) -> None:
    require(value.get(field) == probe.self_hash(value, field), f"{label} self-hash drift")


def exact_names(root: Path, expected: set[str], label: str) -> None:
    actual = {path.name for path in root.iterdir()}
    require(actual == expected, f"{label} exact-file set drift: {sorted(actual ^ expected)}")
    for path in root.iterdir():
        if path.name.startswith("blob-"):
            probe.require_regular_file(path)


def current_code_binding(args: argparse.Namespace) -> dict[str, Any]:
    analyzer = Path(__file__).resolve(strict=True)
    return {
        "analyzer_path": str(analyzer),
        "analyzer_sha256": sha256_file(analyzer),
        "analyzer_job_spec_path": str(args.job_spec),
        "analyzer_job_spec_sha256": sha256_file(args.job_spec),
        "probe_runner_path": str(args.probe_runner),
        "probe_runner_sha256": sha256_file(args.probe_runner),
        "probe_job_spec_path": str(args.probe_job_spec),
        "probe_job_spec_sha256": sha256_file(args.probe_job_spec),
        "publisher_core_path": str(Path(pilot_core.__file__).resolve(strict=True)),
        "publisher_core_sha256": sha256_file(Path(pilot_core.__file__).resolve(strict=True)),
    }


def validate_probe_code(recorded: dict[str, Any], current: dict[str, Any]) -> None:
    mapping = {
        "runner_path": "probe_runner_path",
        "runner_sha256": "probe_runner_sha256",
        "job_spec_path": "probe_job_spec_path",
        "job_spec_sha256": "probe_job_spec_sha256",
        "publisher_core_path": "publisher_core_path",
        "publisher_core_sha256": "publisher_core_sha256",
    }
    for recorded_key, current_key in mapping.items():
        require(
            recorded.get(recorded_key) == current.get(current_key),
            f"probe code binding drift: {recorded_key}",
        )


def parse_telemetry(path: Path, binding: dict[str, Any]) -> dict[str, Any]:
    require(path.stat().st_size == binding.get("bytes"), "telemetry byte binding drift")
    require(sha256_file(path) == binding.get("sha256"), "telemetry SHA-256 drift")
    events: list[dict[str, Any]] = []
    previous_total: dict[str, int] | None = None
    previous_elapsed = -1.0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            try:
                value = json.loads(raw)
            except Exception as exc:
                raise AnalysisError(f"telemetry JSON error line={line_number}") from exc
            require(raw == canonical_json_bytes(value), f"telemetry canonical drift line={line_number}")
            require(value.get("schema_version") == probe.SCHEMA_VERSION, "telemetry schema drift")
            elapsed = value.get("elapsed_seconds")
            require(isinstance(elapsed, (int, float)) and elapsed >= previous_elapsed, "telemetry time regression")
            previous_elapsed = float(elapsed)
            phase = value.get("phase")
            reason = value.get("reason")
            require(isinstance(phase, str) and reason in {"interval", "phase_start", "phase_end"}, "telemetry event drift")
            for key in ("phase_counters", "total_counters"):
                counters = value.get(key)
                require(isinstance(counters, dict) and set(counters) == set(probe.Telemetry.FIELDS), "telemetry counter schema drift")
                require(all(type(item) is int and item >= 0 for item in counters.values()), "telemetry counter type drift")
            total = value["total_counters"]
            if previous_total is not None:
                require(
                    all(total[key] >= previous_total[key] for key in probe.Telemetry.FIELDS),
                    "telemetry total counter regression",
                )
            previous_total = total
            process = value.get("process")
            require(isinstance(process, dict), "telemetry process stats missing")
            for key in ("rss_bytes", "max_rss_bytes", "fd_count", "cpu_user_seconds", "cpu_system_seconds"):
                require(isinstance(process.get(key), (int, float)) and process[key] >= 0, f"telemetry {key} drift")
            events.append(value)
    require(len(events) >= 2, "telemetry event population too small")
    for phase, _, _, _ in probe.PHASES:
        reasons = [event["reason"] for event in events if event["phase"] == phase]
        require(
            len(reasons) >= 2
            and reasons[0] == "phase_start"
            and reasons[-1] == "phase_end"
            and reasons.count("phase_start") == 1
            and reasons.count("phase_end") == 1
            and all(reason == "interval" for reason in reasons[1:-1]),
            f"telemetry phase event sequence drift: {phase}",
        )
    return {"events": events, "event_count": len(events)}


def phase_series(
    events: Sequence[dict[str, Any]], phase: str
) -> list[tuple[float, float]]:
    points = [
        (
            float(event["phase_elapsed_seconds"]),
            float(event["phase_counters"]["objects"]),
        )
        for event in events
        if event.get("phase") == phase
        and event.get("reason") in {"phase_start", "interval", "phase_end"}
    ]
    points.sort()
    require(len(points) >= 2 and points[0][1] == 0, f"phase series is incomplete: {phase}")
    if points[0][0] > 0:
        points.insert(0, (0.0, 0.0))
    require(
        all(points[index][0] >= points[index - 1][0] and points[index][1] >= points[index - 1][1] for index in range(1, len(points))),
        f"phase series regressed: {phase}",
    )
    return points


def interpolate_series(points: Sequence[tuple[float, float]], when: float) -> float:
    if when <= points[0][0]:
        return points[0][1]
    if when >= points[-1][0]:
        return points[-1][1]
    low = 0
    high = len(points) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if points[middle][0] <= when:
            low = middle
        else:
            high = middle
    left_t, left_value = points[low]
    right_t, right_value = points[high]
    if right_t == left_t:
        return right_value
    fraction = (when - left_t) / (right_t - left_t)
    return left_value + fraction * (right_value - left_value)


def aligned_window_rates(
    series: Sequence[Sequence[tuple[float, float]]], window_seconds: float
) -> list[float]:
    require(bool(series), "aligned window series population is empty")
    stop = max(points[-1][0] for points in series)
    if stop < window_seconds:
        return []
    knots = {0.0, stop, window_seconds}
    for points in series:
        for when, _ in points:
            if 0.0 <= when <= stop:
                knots.add(when)
            shifted = when + window_seconds
            if 0.0 <= shifted <= stop:
                knots.add(shifted)
    rates: list[float] = []
    for right in sorted(knots):
        if right < window_seconds:
            continue
        left = right - window_seconds
        delta_objects = sum(
            interpolate_series(points, right) - interpolate_series(points, left)
            for points in series
        )
        require(delta_objects >= -1e-9, "aligned window object counter regressed")
        rates.append(max(0.0, delta_objects) / window_seconds)
    return rates


def window_rates(
    events: Sequence[dict[str, Any]], phase: str, window_seconds: float
) -> list[float]:
    return aligned_window_rates([phase_series(events, phase)], window_seconds)


def verify_selection(
    *,
    root: Path,
    receipt: dict[str, Any],
    catalog: pilot_core.CatalogBinding,
    worker_id: int,
    args: argparse.Namespace,
) -> tuple[list[probe.Item], list[probe.Item], list[probe.Item]]:
    main_positions, before_positions, after_positions = probe.expected_position_sets(
        total_rows=catalog.gpic_rows,
        worker_id=worker_id,
        workers=args.expected_workers,
        main_rows=args.expected_main_rows,
        control_rows_per_arm=args.expected_control_rows_per_arm,
        exclude_build_rows=args.expected_exclude_build_rows,
        exclude_scan_rows=args.expected_exclude_scan_rows,
    )
    main = probe.select_catalog_items(catalog, main_positions, source_prefix=args.expected_source_prefix)
    before = (
        probe.select_catalog_items(catalog, before_positions, source_prefix=args.expected_source_prefix)
        if before_positions
        else []
    )
    after = (
        probe.select_catalog_items(catalog, after_positions, source_prefix=args.expected_source_prefix)
        if after_positions
        else []
    )
    controls = sorted(before + after, key=lambda item: item.local_index)
    binding = receipt["files"]["selection"]
    path = root / binding["name"]
    require(path.name == "selection.bin", "selection filename drift")
    require(path.stat().st_size == binding["bytes"], "selection byte binding drift")
    digest = hashlib.sha256()
    expected = [(0, item) for item in main] + [(1, item) for item in controls]
    with path.open("rb", buffering=0) as handle:
        for kind, item in expected:
            record = handle.read(probe.SELECTION_STRUCT.size)
            require(len(record) == probe.SELECTION_STRUCT.size, "selection record hole")
            digest.update(record)
            actual_kind, local, row_id, path_sha = probe.SELECTION_STRUCT.unpack(record)
            require(actual_kind == kind, "selection kind drift")
            require(local == item.local_index and row_id == item.row_id, "selection row identity drift")
            require(path_sha == hashlib.sha256(item.image_path.encode("utf-8")).digest(), "selection path identity drift")
        require(handle.read(1) == b"", "selection tail bytes")
    require(digest.hexdigest() == binding["sha256"], "selection content SHA drift")
    require(binding["records"] == len(expected), "selection record count drift")
    require(receipt["main_selection_sha256"] == probe.selection_digest(main), "main selection digest drift")
    expected_control_digest = probe.selection_digest(controls) if controls else None
    require(receipt["control_selection_sha256"] == expected_control_digest, "control selection digest drift")
    return main, before, after


def read_exact(handle: Any, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        block = handle.read(size - len(output))
        if not block:
            raise AnalysisError("blob segment ended early")
        output.extend(block)
    return bytes(output)


def verify_index_and_blobs(
    *,
    root: Path,
    s3_worker_root: Path,
    receipt: dict[str, Any],
    combined_items: Sequence[probe.Item],
    verify_content: bool,
) -> dict[str, Any]:
    index_binding = receipt["files"]["index"]
    index_path = root / index_binding["name"]
    require(index_path.name == "index.bin", "index filename drift")
    require(index_path.stat().st_size == index_binding["bytes"], "index byte binding drift")
    require(index_binding["records"] == len(combined_items), "index record-count drift")
    require(index_binding["bytes"] == len(combined_items) * probe.INDEX_STRUCT.size, "index fixed width drift")
    blobs = receipt["s3_blobs"]
    require(isinstance(blobs, list) and blobs, "blob manifest missing")
    for blob_id, binding in enumerate(blobs):
        require(binding.get("blob_id") == blob_id, "blob id sequence drift")
        require(binding.get("name") == f"blob-{blob_id:06d}.bin", "blob filename drift")
        path = s3_worker_root / binding["name"]
        require(binding.get("key") == str(path), "S3 blob key binding drift")
        require(path.stat().st_size == binding["bytes"], "blob byte binding drift")
    index_digest = hashlib.sha256()
    current_blob_id = -1
    blob_handle: Any = None
    blob_digest: Any = None
    blob_offset = 0
    verified_bytes = 0

    def close_blob() -> None:
        nonlocal blob_handle, blob_digest, blob_offset
        if blob_handle is None:
            return
        binding = blobs[current_blob_id]
        require(blob_offset == binding["bytes"], "blob segment coverage hole/tail")
        if verify_content:
            require(blob_digest.hexdigest() == binding["sha256"], "blob content SHA drift")
        blob_handle.close()
        blob_handle = None
        blob_digest = None
        blob_offset = 0

    try:
        with index_path.open("rb", buffering=0) as index_handle:
            for expected_item in combined_items:
                record = index_handle.read(probe.INDEX_STRUCT.size)
                require(len(record) == probe.INDEX_STRUCT.size, "index record hole")
                index_digest.update(record)
                row_id, blob_id, offset, length, source_sha = probe.INDEX_STRUCT.unpack(record)
                require(row_id == expected_item.row_id, "index row/path-order identity drift")
                require(type(length) is int and 0 < length <= 0xFFFFFFFF, "index length drift")
                if blob_id != current_blob_id:
                    close_blob()
                    require(blob_id == current_blob_id + 1 and blob_id < len(blobs), "blob transition drift")
                    current_blob_id = blob_id
                    blob_handle = (s3_worker_root / blobs[blob_id]["name"]).open("rb", buffering=0)
                    blob_digest = hashlib.sha256()
                    blob_offset = 0
                require(offset == blob_offset, "blob offset gap/overlap")
                if verify_content:
                    payload = read_exact(blob_handle, length)
                    blob_digest.update(payload)
                    require(hashlib.sha256(payload).digest() == source_sha, "blob segment/source SHA drift")
                else:
                    blob_handle.seek(length, os.SEEK_CUR)
                blob_offset += length
                verified_bytes += length
            require(index_handle.read(1) == b"", "index tail bytes")
        close_blob()
    finally:
        if blob_handle is not None:
            blob_handle.close()
    require(current_blob_id + 1 == len(blobs), "unreferenced blob file")
    require(index_digest.hexdigest() == index_binding["sha256"], "index SHA drift")
    require(verified_bytes == sum(item["bytes"] for item in blobs), "blob byte closure drift")
    return {"records": len(combined_items), "bytes": verified_bytes, "content_verified": verify_content}


def validate_phase_reports(
    receipt: dict[str, Any],
    *,
    worker_id: int,
    main: Sequence[probe.Item],
    before: Sequence[probe.Item],
    after: Sequence[probe.Item],
) -> tuple[dict[str, dict[str, Any]], list[probe.Item]]:
    reports = receipt.get("phase_reports")
    require(isinstance(reports, list) and [item.get("name") for item in reports] == [item[0] for item in probe.PHASES], "phase order drift")
    by_name = {item["name"]: item for item in reports}
    expected_counts = {
        "single_before": len(before) if worker_id == 0 else 0,
        "source_a": len(main[0::4]),
        "combined_b1": len(main[1::4]),
        "combined_b2": len(main[2::4]),
        "source_a2": len(main[3::4]),
        "single_after": len(after) if worker_id == 0 else 0,
    }
    for name, role, combined, _ in probe.PHASES:
        value = by_name[name]
        require(value.get("role") == role and value.get("combined") is combined, f"phase role drift {name}")
        require(value.get("objects") == expected_counts[name], f"phase object count drift {name}")
        elapsed = value.get("elapsed_seconds")
        require(isinstance(elapsed, (int, float)) and elapsed > 0, f"phase elapsed drift {name}")
        require(isinstance(value.get("source_bytes"), int) and value["source_bytes"] >= expected_counts[name], f"phase bytes drift {name}")
        counters = value.get("telemetry_counters")
        require(isinstance(counters, dict) and counters.get("objects") == expected_counts[name], f"phase telemetry object closure {name}")
        require(counters.get("source_bytes") == value["source_bytes"], f"phase telemetry byte closure {name}")
        require(counters.get("source_attempts", 0) >= expected_counts[name], f"phase attempts closure {name}")
        histogram = value.get("attempt_histogram")
        require(isinstance(histogram, dict), f"phase attempt histogram missing {name}")
        parsed_histogram = {int(key): count for key, count in histogram.items()}
        require(
            all(attempt >= 1 and type(count) is int and count >= 0 for attempt, count in parsed_histogram.items())
            and sum(parsed_histogram.values()) == expected_counts[name]
            and sum(attempt * count for attempt, count in parsed_histogram.items())
            == counters["source_attempts"],
            f"phase attempt histogram closure {name}",
        )
        require(value.get("source_errors") == counters.get("source_errors"), f"phase error closure {name}")
        if not combined:
            require(
                counters.get("staged_bytes") == 0
                and counters.get("s3_write_bytes") == 0,
                f"source-only phase wrote staging/S3 bytes {name}",
            )
        else:
            require(counters.get("staged_bytes") == value["source_bytes"], f"combined staging/source byte drift {name}")
    require(
        sum(by_name[name]["telemetry_counters"]["s3_write_bytes"] for name in ("combined_b1", "combined_b2"))
        == sum(by_name[name]["source_bytes"] for name in ("combined_b1", "combined_b2")),
        "combined Mountpoint-S3 write/source byte closure drift",
    )
    combined_items = sorted(main[1::4], key=lambda item: (item.image_path, item.row_id)) + sorted(
        main[2::4], key=lambda item: (item.image_path, item.row_id)
    )
    return by_name, combined_items


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    current_code = current_code_binding(args)
    topology_evidence = probe.load_topology_evidence(
        args.topology_summary,
        expected_jobtag=args.topology_jobtag,
        expected_raw_sha256=args.topology_summary_raw_sha256,
    )
    root = args.artifact_root
    coord = args.coord_root
    s3_root = args.s3_root
    probe.require_regular_dir(root)
    probe.require_regular_dir(coord)
    probe.require_regular_dir(s3_root)
    require(
        root.name == args.jobtag
        and coord.name == args.jobtag
        and s3_root.name == args.jobtag,
        "jobtag/root basename drift",
    )
    s3_init, s3_init_raw = strict_json(s3_root / "_INIT.json")
    require(
        s3_init.get("schema_version") == probe.S3_INIT_SCHEMA,
        "S3 initialization schema drift",
    )
    validate_self_hash(s3_init, "init_sha256", "S3 initialization")
    require(
        s3_init.get("jobtag") == args.jobtag
        and s3_init.get("workers") == args.expected_workers
        and s3_init.get("root") == str(s3_root)
        and s3_init.get("topology_evidence") == topology_evidence,
        "S3 initialization identity drift",
    )
    validate_probe_code(s3_init.get("code", {}), current_code)
    require(
        s3_init
        == probe.make_s3_init(
            jobtag=args.jobtag,
            workers=args.expected_workers,
            s3_job_root=s3_root,
            code=s3_init["code"],
            topology_evidence=topology_evidence,
        ),
        "S3 initialization exact binding drift",
    )
    coord_init, coord_init_raw = strict_json(coord / "_INIT.json")
    require(
        coord_init.get("schema_version") == probe.COORD_INIT_SCHEMA,
        "FSx initialization schema drift",
    )
    validate_self_hash(coord_init, "coord_init_sha256", "FSx initialization")
    expected_coord_init = probe.make_coord_init(
        jobtag=args.jobtag,
        workers=args.expected_workers,
        s3_job_root=s3_root,
        code=s3_init["code"],
        topology_evidence=topology_evidence,
        s3_init=s3_init,
        s3_init_raw=s3_init_raw,
    )
    require(
        coord_init == expected_coord_init
        and coord_init_raw == canonical_json_bytes(expected_coord_init),
        "FSx/S3 initialization binding drift",
    )
    top, top_raw = strict_json(root / "job_receipt.json")
    require(top.get("schema_version") == probe.JOB_RECEIPT_SCHEMA, "job receipt schema drift")
    validate_self_hash(top, "job_receipt_sha256", "job receipt")
    require((root / "_SUCCESS").read_text(encoding="ascii").strip() == top["job_receipt_sha256"], "job success marker drift")
    require(top.get("jobtag") == args.jobtag and top.get("worker_count") == args.expected_workers, "job receipt identity drift")
    validate_probe_code(top.get("code", {}), current_code)
    s3_top, s3_top_raw = strict_json(s3_root / "manifest.json")
    require(
        s3_top.get("schema_version")
        == "offline_gpic_scale_s3_job_manifest_v1",
        "S3 job manifest schema drift",
    )
    validate_self_hash(s3_top, "manifest_sha256", "S3 job manifest")
    require(
        s3_top.get("jobtag") == args.jobtag
        and s3_top.get("root") == str(s3_root)
        and s3_top.get("worker_count") == args.expected_workers,
        "S3 job manifest identity drift",
    )
    validate_probe_code(s3_top.get("code", {}), current_code)
    require(
        (s3_root / "_SUCCESS").read_text(encoding="ascii").strip()
        == s3_top["manifest_sha256"],
        "S3 job success marker drift",
    )
    require(
        top.get("s3_job_root") == str(s3_root)
        and top.get("s3_job_manifest_raw_sha256")
        == hashlib.sha256(s3_top_raw).hexdigest()
        and top.get("s3_job_manifest_sha256") == s3_top["manifest_sha256"]
        and top.get("s3_payload_bytes") == s3_top.get("bytes"),
        "FSx/S3 top-level binding drift",
    )
    require(
        top.get("s3_init_raw_sha256")
        == hashlib.sha256(s3_init_raw).hexdigest()
        and top.get("s3_init_sha256") == s3_init["init_sha256"]
        and s3_top.get("s3_init_raw_sha256")
        == hashlib.sha256(s3_init_raw).hexdigest()
        and s3_top.get("s3_init_sha256") == s3_init["init_sha256"],
        "top-level S3 initialization binding drift",
    )
    approval, approval_raw = strict_json(coord / "GLOBAL_APPROVED.json")
    require(approval.get("schema_version") == probe.GLOBAL_APPROVAL_SCHEMA, "global approval schema drift")
    validate_self_hash(approval, "approval_sha256", "global approval")
    require(approval.get("jobtag") == args.jobtag and approval.get("workers") == args.expected_workers, "global approval identity drift")
    require(hashlib.sha256(approval_raw).hexdigest() == top.get("global_approval_raw_sha256"), "top/approval raw binding drift")
    require(
        s3_top.get("global_approval_raw_sha256")
        == hashlib.sha256(approval_raw).hexdigest(),
        "S3 top/approval binding drift",
    )
    require(approval.get("main_rows_per_worker") == args.expected_main_rows, "approval main-row shape drift")
    require(
        approval.get("topology_evidence") == topology_evidence,
        "approval/local-topology evidence binding drift",
    )
    require(
        approval.get("s3_init_raw_sha256")
        == hashlib.sha256(s3_init_raw).hexdigest()
        and approval.get("s3_init_sha256") == s3_init["init_sha256"],
        "approval/S3 initialization binding drift",
    )
    local_topologies = approval.get("local_topologies")
    require(
        isinstance(local_topologies, list)
        and len(local_topologies) == args.expected_workers
        and approval.get("local_free_floor_bytes")
        == args.local_free_floor_bytes
        and approval.get("full_partition_local_budget_bytes")
        == args.full_partition_local_budget_bytes
        and approval.get("full_partition_local_estimate_bytes_x1_2", 1 << 100)
        <= args.full_partition_local_budget_bytes,
        "approval local topology/budget closure drift",
    )
    for worker_id, topology in enumerate(local_topologies):
        require(
            isinstance(topology, dict)
            and type(topology.get("st_dev")) is int
            and topology.get("free_bytes", -1) >= args.local_free_floor_bytes
            and (
                not args.production_gate
                or (
                    topology.get("fstype") == "ext4"
                    and topology.get("mount_target") == "/local/evan"
                    and topology.get("total_bytes", 0)
                    >= topology_evidence["total_bytes_min"]
                )
            ),
            f"approval live local topology drift worker={worker_id}",
        )
    require(approval.get("main_rows_union") == args.expected_workers * args.expected_main_rows, "approval union-row closure drift")
    require(approval.get("pilot_exclusion_union_rows") == args.expected_exclude_build_rows + args.expected_exclude_scan_rows, "approval exclusion drift")
    require(approval.get("byte_imbalance_cap") == args.max_byte_imbalance, "approval imbalance cap drift")
    require(approval.get("main_byte_max_over_min") <= args.max_byte_imbalance, "main-byte imbalance gate failed")
    require(approval.get("combined_byte_max_over_min") <= args.max_byte_imbalance, "combined-byte imbalance gate failed")
    require(
        approval.get("s3_payload_hard_cap_bytes")
        == args.expected_workers * args.max_s3_bytes_per_worker
        and approval.get("s3_payload_bytes_observed")
        <= approval["s3_payload_hard_cap_bytes"],
        "combined S3 payload cap closure failed",
    )
    require(
        s3_top.get("bytes") == approval["s3_payload_bytes_observed"],
        "S3 top/approval payload-byte closure drift",
    )
    require(
        approval.get("role_bytes_per_object_max_over_min")
        <= args.max_byte_imbalance,
        "source-only/combined bytes-per-object representativeness gate failed",
    )
    require(
        approval.get("all_role_bytes_per_object_max_over_min")
        <= args.max_byte_imbalance,
        "main/control bytes-per-object representativeness gate failed",
    )
    worker_bindings = top.get("workers")
    require(isinstance(worker_bindings, list) and len(worker_bindings) == args.expected_workers, "top worker binding count drift")
    require([item.get("worker_id") for item in worker_bindings] == list(range(args.expected_workers)), "top worker id sequence drift")
    s3_worker_bindings = s3_top.get("workers")
    require(
        isinstance(s3_worker_bindings, list)
        and [item.get("worker_id") for item in s3_worker_bindings]
        == list(range(args.expected_workers)),
        "S3 top worker sequence drift",
    )
    require(
        s3_top.get("bytes")
        == sum(value.get("bytes", -1) for value in s3_worker_bindings),
        "S3 top worker-byte closure drift",
    )
    exact_names(
        root,
        {"job_receipt.json", "_SUCCESS", *[f"worker-{rank}" for rank in range(args.expected_workers)]},
        "job root",
    )
    exact_names(
        s3_root,
        {
            "_INIT.json",
            "manifest.json",
            "_SUCCESS",
            *[f"worker-{rank}" for rank in range(args.expected_workers)],
        },
        "S3 job root",
    )
    validate_probe_code(approval.get("code", {}), current_code)
    precommit_bindings = approval.get("precommits")
    require(
        isinstance(precommit_bindings, list)
        and [item.get("worker_id") for item in precommit_bindings]
        == list(range(args.expected_workers)),
        "approval precommit sequence drift",
    )
    for worker_id, binding in enumerate(precommit_bindings):
        path = coord / f"precommit.worker-{worker_id}.json"
        require(binding.get("path") == str(path), f"approval precommit path drift {worker_id}")
        value, raw = strict_json(path)
        validate_self_hash(value, "precommit_sha256", f"precommit {worker_id}")
        require(value.get("jobtag") == args.jobtag and value.get("worker_id") == worker_id, f"precommit identity drift {worker_id}")
        require(hashlib.sha256(raw).hexdigest() == binding.get("raw_sha256"), f"precommit raw binding drift {worker_id}")
        require(value["precommit_sha256"] == binding.get("precommit_sha256"), f"precommit self binding drift {worker_id}")
        require(
            value.get("s3_init_raw_sha256")
            == hashlib.sha256(s3_init_raw).hexdigest()
            and value.get("s3_init_sha256") == s3_init["init_sha256"],
            f"precommit/S3 initialization binding drift {worker_id}",
        )

    # Reject a cache-host collision before reading/hashing any large S3 blob.
    preflight_runtimes: list[dict[str, Any]] = []
    for worker_id, binding in enumerate(worker_bindings):
        worker_root = root / f"worker-{worker_id}"
        receipt, receipt_raw = strict_json(worker_root / "receipt.json")
        validate_self_hash(receipt, "receipt_sha256", f"worker {worker_id}")
        require(
            hashlib.sha256(receipt_raw).hexdigest()
            == binding.get("receipt_raw_sha256")
            and receipt.get("worker_id") == worker_id
            and receipt.get("jobtag") == args.jobtag,
            f"preflight worker receipt binding drift {worker_id}",
        )
        runtime = receipt.get("runtime")
        require(
            isinstance(runtime, dict)
            and runtime.get("boot_id_raw_published") is False
            and isinstance(runtime.get("boot_id_sha256"), str)
            and len(runtime["boot_id_sha256"]) == 64,
            f"preflight runtime binding drift {worker_id}",
        )
        preflight_runtimes.append(runtime)
    physical_nodes = {item["boot_id_sha256"] for item in preflight_runtimes}
    require(
        len(physical_nodes) == args.expected_physical_nodes,
        f"physical-node count drift: {len(physical_nodes)}",
    )
    analyzer_runtime = pilot_core.runtime_identity()
    if args.production_gate:
        require(
            str(args.s3_root).startswith("/s3/evan/")
            or str(args.s3_root).startswith("/s3-pixai-test-uw2/evan/"),
            "production S3 root is outside approved Evan mounts",
        )
        require(
            analyzer_runtime["boot_id_sha256"] not in physical_nodes,
            "analyzer must verify S3 payloads from a fresh physical node",
        )

    catalog = pilot_core.load_catalog(args.catalog_root, verify_gpic_sha256=args.verify_catalog_sha256)
    phase_by_worker: list[dict[str, dict[str, Any]]] = []
    telemetry_by_worker: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    source_config_authority: dict[str, Any] | None = None
    fsx_evidence_bytes_observed = 0
    expected_arm_rows = args.expected_main_rows // 4
    for worker_id, binding in enumerate(worker_bindings):
        status, status_raw = strict_json(coord / f"published.worker-{worker_id}.json")
        require(
            status.get("jobtag") == args.jobtag
            and status.get("worker_id") == worker_id
            and hashlib.sha256(status_raw).hexdigest()
            == binding.get("published_status_raw_sha256")
            and status.get("publication_method") == binding.get("publication_method"),
            f"published-status binding drift {worker_id}",
        )
        worker_root = root / f"worker-{worker_id}"
        s3_worker_root = s3_root / f"worker-{worker_id}"
        probe.require_regular_dir(s3_worker_root)
        s3_manifest, s3_manifest_raw = strict_json(
            s3_worker_root / "manifest.json"
        )
        require(
            s3_manifest.get("schema_version")
            == "offline_gpic_scale_s3_worker_manifest_v1",
            f"S3 worker manifest schema drift {worker_id}",
        )
        validate_self_hash(
            s3_manifest, "manifest_sha256", f"S3 worker {worker_id}"
        )
        require(
            s3_manifest.get("jobtag") == args.jobtag
            and s3_manifest.get("worker_id") == worker_id
            and s3_manifest.get("workers") == args.expected_workers
            and s3_manifest.get("root") == str(s3_worker_root)
            and s3_manifest.get("bytes")
            == sum(value["bytes"] for value in s3_manifest.get("blobs", [])),
            f"S3 worker manifest identity drift {worker_id}",
        )
        validate_probe_code(s3_manifest.get("code", {}), current_code)
        require(
            (s3_worker_root / "_SUCCESS").read_text(encoding="ascii").strip()
            == s3_manifest["manifest_sha256"],
            f"S3 worker marker drift {worker_id}",
        )
        s3_binding = s3_worker_bindings[worker_id]
        require(
            s3_binding.get("root") == str(s3_worker_root)
            and s3_binding.get("manifest_raw_sha256")
            == hashlib.sha256(s3_manifest_raw).hexdigest()
            and s3_binding.get("manifest_sha256")
            == s3_manifest["manifest_sha256"]
            and status.get("s3_worker_manifest_raw_sha256")
            == hashlib.sha256(s3_manifest_raw).hexdigest()
            and status.get("s3_worker_manifest_sha256")
            == s3_manifest["manifest_sha256"],
            f"S3 worker binding drift {worker_id}",
        )
        require(
            binding.get("s3_root") == str(s3_worker_root)
            and binding.get("s3_manifest_raw_sha256")
            == hashlib.sha256(s3_manifest_raw).hexdigest()
            and binding.get("s3_manifest_sha256")
            == s3_manifest["manifest_sha256"]
            and binding.get("s3_bytes") == s3_manifest["bytes"],
            f"FSx top/S3 worker binding drift {worker_id}",
        )
        require(
            s3_manifest.get("global_approval_raw_sha256")
            == hashlib.sha256(approval_raw).hexdigest()
            and s3_manifest.get("global_approval_sha256")
            == approval["approval_sha256"],
            f"S3 worker approval binding drift {worker_id}",
        )
        require(
            s3_manifest.get("s3_init_raw_sha256")
            == hashlib.sha256(s3_init_raw).hexdigest()
            and s3_manifest.get("s3_init_sha256")
            == s3_init["init_sha256"]
            and status.get("s3_init_raw_sha256")
            == hashlib.sha256(s3_init_raw).hexdigest()
            and status.get("s3_init_sha256") == s3_init["init_sha256"],
            f"worker/S3 initialization binding drift {worker_id}",
        )
        require(binding.get("root") == str(worker_root), f"top worker root drift {worker_id}")
        probe.require_regular_dir(worker_root)
        receipt, receipt_raw = strict_json(worker_root / "receipt.json")
        require(receipt.get("schema_version") == probe.WORKER_RECEIPT_SCHEMA, f"worker schema drift {worker_id}")
        validate_self_hash(receipt, "receipt_sha256", f"worker {worker_id}")
        require((worker_root / "_SUCCESS").read_text(encoding="ascii").strip() == receipt["receipt_sha256"], f"worker marker drift {worker_id}")
        require(hashlib.sha256(receipt_raw).hexdigest() == binding.get("receipt_raw_sha256"), f"top/raw worker binding drift {worker_id}")
        require(receipt["receipt_sha256"] == binding.get("receipt_sha256"), f"top/self worker binding drift {worker_id}")
        require(receipt.get("jobtag") == args.jobtag and receipt.get("worker_id") == worker_id and receipt.get("workers") == args.expected_workers, f"worker identity drift {worker_id}")
        validate_probe_code(receipt.get("code", {}), current_code)
        require(receipt.get("global_approval_raw_sha256") == hashlib.sha256(approval_raw).hexdigest(), f"worker approval binding drift {worker_id}")
        require(receipt.get("global_approval_sha256") == approval["approval_sha256"], f"worker approval self binding drift {worker_id}")
        require(
            receipt.get("storage", {}).get("s3_init_raw_sha256")
            == hashlib.sha256(s3_init_raw).hexdigest()
            and receipt.get("storage", {}).get("s3_init_sha256")
            == s3_init["init_sha256"],
            f"receipt/S3 initialization binding drift {worker_id}",
        )
        require(receipt.get("source_prefix") == args.expected_source_prefix, f"source prefix drift {worker_id}")
        require(
            receipt.get("topology_evidence") == topology_evidence,
            f"receipt/local-topology evidence binding drift {worker_id}",
        )
        require(
            receipt.get("local_topology_live") == local_topologies[worker_id],
            f"receipt/live local topology binding drift {worker_id}",
        )
        require(receipt.get("partition") == "catalog_local_index_mod_workers", f"partition drift {worker_id}")
        require(receipt.get("source_order") == "path_lexicographic_within_phase", f"source order drift {worker_id}")
        require(receipt.get("main_rows") == args.expected_main_rows, f"worker main rows drift {worker_id}")
        require(receipt.get("sample_rows_per_worker_per_arm") == expected_arm_rows, f"per-arm sample rows drift {worker_id}")
        require(receipt.get("main_total_rows_all_workers") == args.expected_workers * args.expected_main_rows, f"main total rows drift {worker_id}")
        require(receipt.get("combined_rows_per_worker") == 2 * expected_arm_rows, f"combined worker rows drift {worker_id}")
        require(receipt.get("combined_total_rows_all_workers") == 2 * expected_arm_rows * args.expected_workers, f"combined total rows drift {worker_id}")
        require(receipt.get("single_control_total_rows") == 2 * args.expected_control_rows_per_arm, f"single control total rows drift {worker_id}")
        require(
            receipt.get("all_source_rows_including_controls")
            == args.expected_workers * args.expected_main_rows
            + 2 * args.expected_control_rows_per_arm,
            f"all-source total rows drift {worker_id}",
        )
        require(receipt.get("control_rows") == (2 * args.expected_control_rows_per_arm if worker_id == 0 else 0), f"control rows drift {worker_id}")
        require(
            receipt.get("pilot_exclusion")
            == {
                "build_rows": args.expected_exclude_build_rows,
                "scan_rows": args.expected_exclude_scan_rows,
                "union_rows": args.expected_exclude_build_rows
                + args.expected_exclude_scan_rows,
            },
            f"prior-pilot exclusion receipt drift {worker_id}",
        )
        if args.production_gate:
            require(receipt.get("production_shape") is True, f"production-shape flag absent worker={worker_id}")
        source_config = receipt.get("source")
        require(
            isinstance(source_config, dict)
            and type(source_config.get("threads")) is int
            and source_config["threads"] > 0
            and type(source_config.get("attempt_limit")) is int
            and source_config["attempt_limit"] > 0
            and type(source_config.get("block_bytes")) is int
            and source_config["block_bytes"] > 0,
            f"source configuration drift worker={worker_id}",
        )
        if source_config_authority is None:
            source_config_authority = source_config
        else:
            require(source_config == source_config_authority, f"cross-worker source configuration drift {worker_id}")
        layout = receipt.get("layout")
        require(
            isinstance(layout, dict)
            and layout.get("index_struct") == probe.INDEX_STRUCT.format
            and layout.get("index_record_bytes") == probe.INDEX_STRUCT.size
            and layout.get("selection_struct") == probe.SELECTION_STRUCT.format
            and layout.get("selection_record_bytes") == probe.SELECTION_STRUCT.size
            and layout.get("compression") == "none",
            f"layout contract drift worker={worker_id}",
        )
        if args.production_gate:
            require(layout.get("blob_target_bytes") == probe.PRODUCTION_BLOB_BYTES, f"production blob target drift worker={worker_id}")
            require(
                layout.get("max_s3_bytes_per_worker")
                == probe.PRODUCTION_MAX_S3_BYTES_PER_WORKER
                and layout.get("max_s3_bytes_all_workers")
                == args.expected_workers
                * probe.PRODUCTION_MAX_S3_BYTES_PER_WORKER
                and layout.get("max_local_staging_bytes_per_worker")
                == probe.PRODUCTION_MAX_LOCAL_STAGING_BYTES_PER_WORKER
                and layout.get("max_fsx_evidence_bytes_per_worker")
                == probe.PRODUCTION_MAX_FSX_EVIDENCE_BYTES_PER_WORKER
                and layout.get("local_free_floor_bytes")
                == probe.PRODUCTION_LOCAL_FREE_FLOOR_BYTES
                and layout.get("full_partition_local_budget_bytes")
                == probe.PRODUCTION_FULL_PARTITION_LOCAL_BUDGET_BYTES,
                f"production storage cap drift worker={worker_id}",
            )
        receipt_catalog = receipt.get("catalog", {})
        require(receipt_catalog.get("catalog_hash") == catalog.catalog_hash, f"catalog hash drift worker={worker_id}")
        require(receipt_catalog.get("gpic", {}).get("rows") == catalog.gpic_rows, f"catalog rows drift worker={worker_id}")
        files = receipt.get("files")
        require(isinstance(files, dict) and set(files) == {"selection", "index", "telemetry"}, f"file map drift worker={worker_id}")
        blobs = receipt.get("s3_blobs")
        require(isinstance(blobs, list) and blobs == s3_manifest.get("blobs"), f"S3 blob manifest drift worker={worker_id}")
        blob_names = [item.get("name") for item in blobs]
        exact_names(worker_root, {"receipt.json", "_SUCCESS", "selection.bin", "index.bin", "telemetry.jsonl"}, f"worker {worker_id}")
        exact_names(s3_worker_root, {"manifest.json", "_SUCCESS", *blob_names}, f"S3 worker {worker_id}")
        main, before, after = verify_selection(
            root=worker_root,
            receipt=receipt,
            catalog=catalog,
            worker_id=worker_id,
            args=args,
        )
        phases, combined_items = validate_phase_reports(
            receipt, worker_id=worker_id, main=main, before=before, after=after
        )
        verification = verify_index_and_blobs(
            root=worker_root,
            s3_worker_root=s3_worker_root,
            receipt=receipt,
            combined_items=combined_items,
            verify_content=args.verify_blob_content,
        )
        telemetry = parse_telemetry(worker_root / "telemetry.jsonl", files["telemetry"])
        for phase_name, report in phases.items():
            endings = [
                event
                for event in telemetry["events"]
                if event["phase"] == phase_name
                and event["reason"] == "phase_end"
            ]
            require(
                len(endings) == 1
                and endings[0]["phase_counters"]
                == report["telemetry_counters"],
                f"telemetry/receipt phase-end closure drift worker={worker_id} phase={phase_name}",
            )
        runtime = receipt.get("runtime")
        require(isinstance(runtime, dict) and runtime.get("boot_id_raw_published") is False, f"runtime boot binding drift worker={worker_id}")
        require(isinstance(runtime.get("boot_id_sha256"), str) and len(runtime["boot_id_sha256"]) == 64, f"runtime boot hash missing worker={worker_id}")
        if args.production_gate:
            require(runtime.get("hostname") == f"{args.jobtag}-{worker_id}", f"runtime hostname/jobtag drift worker={worker_id}")
        resources = receipt.get("resources")
        require(
            isinstance(resources, dict)
            and set(resources) == {"before_phases", "after_phases"},
            f"storage resource snapshots missing worker={worker_id}",
        )
        for point in ("before_phases", "after_phases"):
            require(
                set(resources[point])
                == {"fsx_space", "local_space", "s3_mount_space"},
                f"storage-space snapshot set drift worker={worker_id} point={point}",
            )
            for space_name, space in resources[point].items():
                require(
                    isinstance(space, dict)
                    and space.get("status") in {"known", "unknown"},
                    f"space status drift worker={worker_id} point={point} name={space_name}",
                )
                if space["status"] == "known":
                    require(
                        all(type(space.get(key)) is int and space[key] >= 0 for key in ("total_bytes", "used_bytes", "free_bytes")),
                        f"space values drift worker={worker_id} point={point} name={space_name}",
                    )
                else:
                    require(isinstance(space.get("reason"), str), f"space unknown reason missing worker={worker_id}")
        storage = receipt.get("storage")
        require(
            isinstance(storage, dict)
            and storage.get("payload_backend") == "mountpoint_s3"
            and storage.get("local_staging_empty_and_removed") is True
            and storage.get("s3_worker_root") == str(s3_worker_root)
            and storage.get("s3_worker_manifest_raw_sha256")
            == hashlib.sha256(s3_manifest_raw).hexdigest()
            and storage.get("s3_worker_manifest_sha256")
            == s3_manifest["manifest_sha256"]
            and storage.get("s3_payload_bytes") == s3_manifest["bytes"]
            and storage.get("local_staging_peak_bytes")
            <= args.max_local_staging_bytes_per_worker
            and storage.get("fsx_compact_evidence_bytes")
            <= args.max_fsx_evidence_bytes_per_worker,
            f"storage contract drift worker={worker_id}",
        )
        fsx_evidence_bytes_observed += storage["fsx_compact_evidence_bytes"]
        lustre = receipt.get("lustre")
        require(
            isinstance(lustre, dict) and lustre.get("status") in {"known", "unknown"},
            f"Lustre getstripe status drift worker={worker_id}",
        )
        if lustre["status"] == "known":
            require(
                isinstance(lustre.get("files"), list)
                and len(lustre["files"]) == 3,
                f"Lustre getstripe file closure drift worker={worker_id}",
            )
        else:
            require(isinstance(lustre.get("reason"), str), f"Lustre unknown reason missing worker={worker_id}")
        phase_by_worker.append(phases)
        telemetry_by_worker.append(telemetry)
        runtimes.append(runtime)
        resource_rows.append({"worker_id": worker_id, "resources": resources, "lustre": lustre})
        verification_rows.append({"worker_id": worker_id, **verification})
    pilot_core.validate_catalog_closure(catalog)
    require(
        runtimes == preflight_runtimes,
        "worker runtime changed between preflight and full verification",
    )

    source_phase_rates: list[float] = []
    combined_phase_rates: list[float] = []
    single_rates: list[float] = []
    skew_ratios: list[float] = []
    phase_aggregate: dict[str, Any] = {}
    for name in ("source_a", "combined_b1", "combined_b2", "source_a2"):
        reports = [worker[name] for worker in phase_by_worker]
        aggregate_objects = sum(item["objects"] for item in reports)
        aggregate_bytes = sum(item["source_bytes"] for item in reports)
        aggregate_elapsed = max(float(item["elapsed_seconds"]) for item in reports)
        rate = aggregate_objects / aggregate_elapsed
        node_rates = [item["objects"] / item["elapsed_seconds"] for item in reports]
        skew = max(node_rates) / min(node_rates)
        skew_ratios.append(skew)
        phase_aggregate[name] = {
            "sample_rows_per_worker": reports[0]["objects"],
            "total_rows": aggregate_objects,
            "source_bytes": aggregate_bytes,
            "aggregate_elapsed_seconds": aggregate_elapsed,
            "aggregate_objects_per_second": rate,
            "node_fast_rate_over_slow_rate": skew,
            "node_slow_time_over_fast_time": skew,
        }
        (combined_phase_rates if name.startswith("combined") else source_phase_rates).append(rate)
    for name in ("single_before", "single_after"):
        report = phase_by_worker[0][name]
        single_rates.append(report["objects"] / report["elapsed_seconds"])
        phase_aggregate[name] = {
            "sample_rows_worker0": report["objects"],
            "total_rows": report["objects"],
            "objects_per_second": single_rates[-1],
        }
    source_rate = sum(
        phase_aggregate[name]["total_rows"] for name in ("source_a", "source_a2")
    ) / sum(
        phase_aggregate[name]["aggregate_elapsed_seconds"]
        for name in ("source_a", "source_a2")
    )
    combined_rate = sum(
        phase_aggregate[name]["total_rows"]
        for name in ("combined_b1", "combined_b2")
    ) / sum(
        phase_aggregate[name]["aggregate_elapsed_seconds"]
        for name in ("combined_b1", "combined_b2")
    )
    single_rate = sum(
        phase_aggregate[name]["total_rows"]
        for name in ("single_before", "single_after")
    ) / sum(
        phase_by_worker[0][name]["elapsed_seconds"]
        for name in ("single_before", "single_after")
    )
    e8 = source_rate / (args.expected_workers * single_rate)
    max_skew = max(skew_ratios)
    combined_over_source = combined_rate / source_rate

    five_minute: dict[str, Any] = {}
    aggregate_window_rates_by_role: dict[str, list[float]] = {
        "source_only_8": [],
        "combined_8": [],
        "single_control": [],
    }
    missing_windows: list[str] = []
    for name, role, _, _ in probe.PHASES:
        ranks = [0] if role == "single_control" else list(range(args.expected_workers))
        node_p10s: list[float] = []
        counts: dict[str, int] = {}
        aligned_series: list[list[tuple[float, float]]] = []
        for rank in ranks:
            rates = window_rates(
                telemetry_by_worker[rank]["events"], name, args.window_seconds
            )
            counts[str(rank)] = len(rates)
            if not rates:
                missing_windows.append(f"{name}:worker-{rank}")
            else:
                node_p10s.append(percentile(rates, 0.10))
            aligned_series.append(
                phase_series(telemetry_by_worker[rank]["events"], name)
            )
        aggregate_rates = aligned_window_rates(
            aligned_series, args.window_seconds
        )
        aggregate_p10 = (
            percentile(aggregate_rates, 0.10) if aggregate_rates else None
        )
        if aggregate_p10 is None:
            missing_windows.append(f"{name}:aggregate")
        five_minute[name] = {
            "window_counts_by_worker": counts,
            "node_p10_objects_per_second": node_p10s,
            "aggregate_aligned_window_count": len(aggregate_rates),
            "aggregate_aligned_p10_objects_per_second": aggregate_p10,
        }
        aggregate_window_rates_by_role[role].extend(aggregate_rates)
    source_window_p10 = (
        percentile(aggregate_window_rates_by_role["source_only_8"], 0.10)
        if aggregate_window_rates_by_role["source_only_8"]
        else None
    )
    combined_window_p10 = (
        percentile(aggregate_window_rates_by_role["combined_8"], 0.10)
        if aggregate_window_rates_by_role["combined_8"]
        else None
    )
    single_window_p10 = (
        percentile(aggregate_window_rates_by_role["single_control"], 0.10)
        if aggregate_window_rates_by_role["single_control"]
        else None
    )

    io_shares: list[float] = []
    io_breakdown: list[dict[str, Any]] = []
    io_keys = (
        "local_stage_write_ns",
        "local_stage_close_ns",
        "local_stage_fsync_ns",
        "local_cleanup_ns",
        "s3_copy_read_ns",
        "s3_write_ns",
        "s3_fsync_ns",
        "index_write_ns",
        "index_fsync_ns",
    )
    for worker_id, phases in enumerate(phase_by_worker):
        for name in ("combined_b1", "combined_b2"):
            report = phases[name]
            counters = report["telemetry_counters"]
            io_ns = sum(counters[key] for key in io_keys)
            wall_ns = report["elapsed_seconds"] * 1e9
            io_shares.append(io_ns / wall_ns)
            io_breakdown.append(
                {
                    "worker_id": worker_id,
                    "phase": name,
                    "wall_seconds": report["elapsed_seconds"],
                    "source_open_seconds": counters["source_open_ns"] / 1e9,
                    "source_read_hash_seconds": counters["source_read_hash_ns"] / 1e9,
                    "ordered_hol_seconds": counters["ordered_hol_ns"] / 1e9,
                    "transport_seconds_by_component": {
                        key.removesuffix("_ns"): counters[key] / 1e9
                        for key in io_keys
                    },
                    "transport_wall_share": io_ns / wall_ns,
                    "staged_bytes": counters["staged_bytes"],
                    "s3_write_bytes": counters["s3_write_bytes"],
                }
            )
    projected_write_fsync_eta_share = max(io_shares)
    total_source_attempts = sum(
        phases[name]["telemetry_counters"]["source_attempts"]
        for phases in phase_by_worker
        for name, _, _, _ in probe.PHASES
    )
    total_source_errors = sum(
        phases[name]["telemetry_counters"]["source_errors"]
        for phases in phase_by_worker
        for name, _, _, _ in probe.PHASES
    )
    e8_gate = e8 >= args.e8_min
    e8_target = e8 >= args.e8_target
    skew_gate = max_skew <= args.max_node_skew
    combined_ratio_gate = combined_over_source >= args.min_combined_over_source
    io_share_gate = projected_write_fsync_eta_share <= args.max_write_fsync_eta_share
    transport_gate = combined_ratio_gate or io_share_gate
    windows_gate = not missing_windows or args.allow_missing_windows
    positive_window_gate = (
        combined_window_p10 is not None and combined_window_p10 > 0
    ) or args.allow_missing_windows
    hard_pass = (
        e8_gate
        and skew_gate
        and transport_gate
        and windows_gate
        and positive_window_gate
    )
    if not hard_pass:
        status = "FAIL"
    elif not e8_target:
        status = "PASS_GATE_BELOW_TARGET_CONFIRMATION_RECOMMENDED"
    else:
        status = "PASS"
    eta_rate = min(
        combined_rate,
        combined_window_p10 if combined_window_p10 is not None else combined_rate,
    )
    t_full_seconds = (
        catalog.gpic_rows / eta_rate * args.eta_safety_factor
        if eta_rate > 0
        else None
    )
    t_75_seconds = (
        catalog.gpic_rows
        * args.partial_fraction
        / eta_rate
        * args.eta_safety_factor
        if eta_rate > 0
        else None
    )
    total_main_rows = args.expected_workers * args.expected_main_rows
    total_combined_rows = total_main_rows // 2
    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "generated_at": probe.utc_now(),
        "status": status,
        "jobtag": args.jobtag,
        "evidence": {
            "artifact_root": str(root),
            "coord_root": str(coord),
            "s3_root": str(s3_root),
            "s3_init_raw_sha256": hashlib.sha256(s3_init_raw).hexdigest(),
            "s3_init_sha256": s3_init["init_sha256"],
            "coord_init_raw_sha256": hashlib.sha256(
                coord_init_raw
            ).hexdigest(),
            "coord_init_sha256": coord_init["coord_init_sha256"],
            "s3_job_manifest_raw_sha256": hashlib.sha256(
                s3_top_raw
            ).hexdigest(),
            "s3_job_manifest_sha256": s3_top["manifest_sha256"],
            "job_receipt_raw_sha256": hashlib.sha256(top_raw).hexdigest(),
            "job_receipt_sha256": top["job_receipt_sha256"],
            "global_approval_raw_sha256": hashlib.sha256(approval_raw).hexdigest(),
            "global_approval_sha256": approval["approval_sha256"],
            "code": current_code,
            "worker_receipts": worker_bindings,
            "physical_node_count": len(physical_nodes),
            "analyzer_runtime": analyzer_runtime,
            "analyzer_boot_distinct_from_builders": (
                analyzer_runtime["boot_id_sha256"] not in physical_nodes
            ),
            "boot_id_raw_published": False,
            "blob_content_verified": args.verify_blob_content,
            "catalog_sha256_verified": args.verify_catalog_sha256,
            "source_config": source_config_authority,
            "topology_evidence": topology_evidence,
        },
        "shape": {
            "workers": args.expected_workers,
            "sample_rows_per_worker_per_arm": expected_arm_rows,
            "main_rows_per_worker": args.expected_main_rows,
            "main_total_rows": total_main_rows,
            "combined_rows_per_worker": args.expected_main_rows // 2,
            "combined_total_rows": total_combined_rows,
            "s3_payload_bytes_observed": approval[
                "s3_payload_bytes_observed"
            ],
            "s3_payload_hard_cap_bytes": approval[
                "s3_payload_hard_cap_bytes"
            ],
            "fsx_compact_evidence_hard_cap_bytes": (
                args.expected_workers * args.max_fsx_evidence_bytes_per_worker
            ),
            "fsx_compact_evidence_bytes_observed": fsx_evidence_bytes_observed,
            "local_rolling_staging_hard_cap_bytes_per_worker": (
                args.max_local_staging_bytes_per_worker
            ),
            "full_partition_local_estimate_bytes_x1_2": approval[
                "full_partition_local_estimate_bytes_x1_2"
            ],
            "full_partition_local_budget_bytes": approval[
                "full_partition_local_budget_bytes"
            ],
            "local_free_floor_bytes": approval["local_free_floor_bytes"],
            "single_control_rows_per_arm": args.expected_control_rows_per_arm,
            "single_control_total_rows": 2 * args.expected_control_rows_per_arm,
            "all_source_rows_including_controls": (
                total_main_rows + 2 * args.expected_control_rows_per_arm
            ),
            "arm_order": [item[0] for item in probe.PHASES],
            "partition": "catalog_local_index_mod_workers",
            "source_order": "path_lexicographic_within_phase",
            "excluded_prior_pilot_rows": args.expected_exclude_build_rows + args.expected_exclude_scan_rows,
        },
        "metrics": {
            "E8": e8,
            "E8_gate_min": args.e8_min,
            "E8_target": args.e8_target,
            "source_only_8_objects_per_second": source_rate,
            "single_node_source_only_objects_per_second": single_rate,
            "combined_8_objects_per_second": combined_rate,
            "combined_over_source_only": combined_over_source,
            "node_fast_rate_over_slow_rate_max": max_skew,
            "node_slow_time_over_fast_time_max": max_skew,
            "five_minute_source_aggregate_p10_objects_per_second": source_window_p10,
            "five_minute_combined_aggregate_p10_objects_per_second": combined_window_p10,
            "five_minute_single_node_p10_objects_per_second": single_window_p10,
            "projected_write_fsync_eta_share_max": projected_write_fsync_eta_share,
            "source_only_bytes_per_object": approval["source_only_bytes_per_object"],
            "combined_bytes_per_object": approval["combined_bytes_per_object"],
            "role_bytes_per_object_max_over_min": approval[
                "role_bytes_per_object_max_over_min"
            ],
            "control_before_bytes_per_object": approval[
                "control_before_bytes_per_object"
            ],
            "control_after_bytes_per_object": approval[
                "control_after_bytes_per_object"
            ],
            "all_role_bytes_per_object_max_over_min": approval[
                "all_role_bytes_per_object_max_over_min"
            ],
            "source_attempts_total": total_source_attempts,
            "source_errors_total": total_source_errors,
            "phase_aggregate": phase_aggregate,
            "five_minute_windows": five_minute,
            "window_seconds": args.window_seconds,
            "per_worker_phase_io_breakdown": io_breakdown,
        },
        "eta": {
            "rate_authority": (
                "min_combined_two_phase_total_rate_and_pooled_aligned_5min_p10"
            ),
            "rate_objects_per_second": eta_rate,
            "safety_factor": args.eta_safety_factor,
            "catalog_rows_full": catalog.gpic_rows,
            "partial_fraction": args.partial_fraction,
            "T_full_seconds_x1_2": t_full_seconds,
            "T_full_hours_x1_2": (
                t_full_seconds / 3600.0 if t_full_seconds is not None else None
            ),
            "T_75_seconds_x1_2": t_75_seconds,
            "T_75_hours_x1_2": (
                t_75_seconds / 3600.0 if t_75_seconds is not None else None
            ),
        },
        "gates": {
            "E8_at_least_0_75": e8_gate,
            "E8_target_0_85": e8_target,
            "node_skew_at_most_1_25": skew_gate,
            "combined_at_least_source_0_85": combined_ratio_gate,
            "write_fsync_projected_eta_share_at_most_0_15": io_share_gate,
            "combined_or_io_tradeoff": transport_gate,
            "zero_source_errors_in_probe": total_source_errors == 0,
            "all_required_5min_windows_present": not missing_windows,
            "positive_combined_5min_p10": positive_window_gate,
            "window_requirement_bypassed_nonproduction": (
                bool(missing_windows) and args.allow_missing_windows
            ),
            "missing_5min_windows": missing_windows,
            "hard_pass": hard_pass,
        },
        "worker_resources_and_lustre": resource_rows,
        "worker_blob_verification": verification_rows,
        "next_step": (
            "full_s3_repack_eta_ready"
            if status == "PASS"
            else "larger_confirmation_requires_explicit_storage_backend_decision"
            if status == "PASS_GATE_BELOW_TARGET_CONFIRMATION_RECOMMENDED"
            else "do_not_authorize_repack_from_this_probe"
        ),
    }
    result["analysis_sha256"] = probe.self_hash(result, "analysis_sha256")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--coord-root", type=Path, required=True)
    parser.add_argument("--s3-root", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument(
        "--topology-summary",
        type=Path,
        default=probe.PRODUCTION_TOPOLOGY_SUMMARY,
    )
    parser.add_argument(
        "--topology-jobtag", default=probe.PRODUCTION_TOPOLOGY_JOBTAG
    )
    parser.add_argument(
        "--topology-summary-raw-sha256",
        default=probe.PRODUCTION_TOPOLOGY_RAW_SHA256,
    )
    parser.add_argument("--jobtag", required=True)
    parser.add_argument("--probe-runner", type=Path, required=True)
    parser.add_argument("--probe-job-spec", type=Path, required=True)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-workers", type=int, default=probe.PRODUCTION_WORKERS)
    parser.add_argument("--expected-physical-nodes", type=int, default=probe.PRODUCTION_WORKERS)
    parser.add_argument("--expected-main-rows", type=int, default=probe.PRODUCTION_MAIN_ROWS)
    parser.add_argument("--expected-control-rows-per-arm", type=int, default=probe.PRODUCTION_CONTROL_ROWS_PER_ARM)
    parser.add_argument("--expected-exclude-build-rows", type=int, default=probe.PRODUCTION_EXCLUDE_BUILD_ROWS)
    parser.add_argument("--expected-exclude-scan-rows", type=int, default=probe.PRODUCTION_EXCLUDE_SCAN_ROWS)
    parser.add_argument("--expected-source-prefix", default=probe.PRODUCTION_SOURCE_PREFIX)
    parser.add_argument("--max-byte-imbalance", type=float, default=1.10)
    parser.add_argument(
        "--max-s3-bytes-per-worker",
        type=int,
        default=probe.PRODUCTION_MAX_S3_BYTES_PER_WORKER,
    )
    parser.add_argument(
        "--max-local-staging-bytes-per-worker",
        type=int,
        default=probe.PRODUCTION_MAX_LOCAL_STAGING_BYTES_PER_WORKER,
    )
    parser.add_argument(
        "--max-fsx-evidence-bytes-per-worker",
        type=int,
        default=probe.PRODUCTION_MAX_FSX_EVIDENCE_BYTES_PER_WORKER,
    )
    parser.add_argument(
        "--local-free-floor-bytes",
        type=int,
        default=probe.PRODUCTION_LOCAL_FREE_FLOOR_BYTES,
    )
    parser.add_argument(
        "--full-partition-local-budget-bytes",
        type=int,
        default=probe.PRODUCTION_FULL_PARTITION_LOCAL_BUDGET_BYTES,
    )
    parser.add_argument("--e8-min", type=float, default=0.75)
    parser.add_argument("--e8-target", type=float, default=0.85)
    parser.add_argument("--max-node-skew", type=float, default=1.25)
    parser.add_argument("--min-combined-over-source", type=float, default=0.85)
    parser.add_argument("--max-write-fsync-eta-share", type=float, default=0.15)
    parser.add_argument("--eta-safety-factor", type=float, default=1.20)
    parser.add_argument("--partial-fraction", type=float, default=0.75)
    parser.add_argument("--window-seconds", type=float, default=300.0)
    parser.add_argument("--allow-missing-windows", action="store_true")
    parser.add_argument("--verify-catalog-sha256", action="store_true")
    parser.add_argument("--verify-blob-content", action="store_true")
    parser.add_argument("--production-gate", action="store_true")
    args = parser.parse_args(argv)
    for label in (
        "artifact_root",
        "coord_root",
        "s3_root",
        "catalog_root",
        "topology_summary",
        "probe_runner",
        "probe_job_spec",
        "job_spec",
    ):
        setattr(args, label, getattr(args, label).expanduser().resolve(strict=True))
    args.output = args.output.expanduser().absolute()
    require(not args.output.exists() and not args.output.is_symlink(), "analysis output namespace is not fresh")
    require(args.output.parent.exists() and str(args.output).startswith("/fsx/"), "analysis output must target an existing FSx directory")
    require(args.expected_main_rows % 4 == 0, "main rows must split exactly into four arms")
    require(1 <= args.expected_physical_nodes <= args.expected_workers, "expected physical-node count drift")
    require(args.partial_fraction == 0.75 and args.eta_safety_factor == 1.20, "ETA authority constants drift")
    require(args.window_seconds > 0, "window seconds must be positive")
    if args.production_gate:
        require(args.expected_workers == probe.PRODUCTION_WORKERS, "production worker count drift")
        require(args.expected_physical_nodes == probe.PRODUCTION_WORKERS, "production physical-node count drift")
        require(args.expected_main_rows == probe.PRODUCTION_MAIN_ROWS, "production main rows drift")
        require(args.expected_control_rows_per_arm == probe.PRODUCTION_CONTROL_ROWS_PER_ARM, "production control rows drift")
        require(
            args.expected_exclude_build_rows == probe.PRODUCTION_EXCLUDE_BUILD_ROWS
            and args.expected_exclude_scan_rows == probe.PRODUCTION_EXCLUDE_SCAN_ROWS,
            "production prior-pilot exclusion shape drift",
        )
        require(args.expected_source_prefix == probe.PRODUCTION_SOURCE_PREFIX, "production source prefix drift")
        require(
            args.topology_summary == probe.PRODUCTION_TOPOLOGY_SUMMARY
            and args.topology_jobtag == probe.PRODUCTION_TOPOLOGY_JOBTAG
            and args.topology_summary_raw_sha256
            == probe.PRODUCTION_TOPOLOGY_RAW_SHA256,
            "production topology evidence binding drift",
        )
        require(args.max_byte_imbalance == 1.10, "production byte-imbalance gate drift")
        require(
            args.max_s3_bytes_per_worker
            == probe.PRODUCTION_MAX_S3_BYTES_PER_WORKER
            and args.max_local_staging_bytes_per_worker
            == probe.PRODUCTION_MAX_LOCAL_STAGING_BYTES_PER_WORKER
            and args.max_fsx_evidence_bytes_per_worker
            == probe.PRODUCTION_MAX_FSX_EVIDENCE_BYTES_PER_WORKER,
            "production storage caps drift",
        )
        require(
            args.local_free_floor_bytes
            == probe.PRODUCTION_LOCAL_FREE_FLOOR_BYTES
            and args.full_partition_local_budget_bytes
            == probe.PRODUCTION_FULL_PARTITION_LOCAL_BUDGET_BYTES,
            "production local full-partition budget drift",
        )
        require(args.e8_min == 0.75 and args.e8_target == 0.85, "production E8 gates drift")
        require(args.max_node_skew == 1.25, "production node-skew gate drift")
        require(args.min_combined_over_source == 0.85, "production combined/source gate drift")
        require(args.max_write_fsync_eta_share == 0.15, "production write/fsync gate drift")
        require(args.verify_catalog_sha256 and args.verify_blob_content, "production analyzer requires full catalog/blob verification")
        require(args.window_seconds == 300.0 and not args.allow_missing_windows, "production analyzer requires strict five-minute windows")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(args)
    raw = canonical_json_bytes(result)
    pilot_core.write_once_bytes(args.output, raw, fsync_file=True)
    pilot_core.fsync_dir(args.output.parent)
    eta_hours = result["eta"]["T_full_hours_x1_2"]
    eta_text = f"{eta_hours:.3f}" if eta_hours is not None else "unavailable"
    print(
        f"OFFLINE_GPIC_SCALE_ANALYSIS_{result['status']} output={args.output} "
        f"E8={result['metrics']['E8']:.6f} "
        f"T_full_h={eta_text}",
        flush=True,
    )
    return 0 if result["status"] != "FAIL" else 3


if __name__ == "__main__":
    raise SystemExit(main())
