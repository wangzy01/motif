#!/usr/bin/env python3
"""Fresh eight-host full verifier and success-last GPIC publication."""

from __future__ import annotations

import argparse
import hashlib
import mmap
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from scripts.offline_arb import build_offline_gpic_s3_production as builder
from scripts.offline_arb import offline_gpic_s3_production_common as common
from scripts.offline_arb import offline_gpic_tar_sequential as tar_source
from scripts.offline_arb import run_offline_gpic_blob_pilot as pilot_core


def wait_for_paths(
    paths: Sequence[Path], *, coord_root: Path, timeout_seconds: float, label: str
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        failures = sorted(coord_root.glob("FAIL.*.json"))
        if failures:
            raise common.ProductionError(
                f"peer failure while waiting for {label}: {failures[0].name}"
            )
        if all(path.exists() for path in paths):
            return
        if time.monotonic() >= deadline:
            raise common.ProductionError(f"timed out waiting for {label}")
        time.sleep(0.5)


def initialize_verifier(args: argparse.Namespace, code: dict[str, Any]) -> None:
    init = {
        "schema_version": common.PRODUCTION_SCHEMA,
        "build_jobtag": args.build_jobtag,
        "verifier_jobtag": args.verifier_jobtag,
        "workers": args.workers,
        "artifact_root": str(args.verifier_artifact_root),
        "coord_root": str(args.verifier_coord_root),
        "code": code,
    }
    init["verifier_init_sha256"] = common.self_hash(
        init, "verifier_init_sha256"
    )
    common.require_regular_dir(args.verifier_artifact_root.parent)
    common.require_regular_dir(args.verifier_coord_root.parent)
    if args.worker_id == 0:
        for path in (args.verifier_artifact_root, args.verifier_coord_root):
            if path.exists() or path.is_symlink():
                raise common.ProductionError(
                    f"verifier namespace is not fresh: {path}"
                )
        args.verifier_artifact_root.mkdir(mode=0o700)
        (args.verifier_artifact_root / "workers").mkdir(mode=0o700)
        args.verifier_coord_root.mkdir(mode=0o700)
        common.write_fsx_json_once(
            args.verifier_coord_root / "_INIT.json", init
        )
        pilot_core.fsync_dir(args.verifier_coord_root)
    common.wait_for_exact_file(
        args.verifier_coord_root / "_INIT.json",
        timeout_seconds=args.barrier_timeout_seconds,
        failure_root=(
            args.verifier_coord_root
            if args.verifier_coord_root.exists()
            else None
        ),
        label="verifier initialization",
    )
    observed, raw = common.strict_json(args.verifier_coord_root / "_INIT.json")
    common.validate_self_hash(
        observed, "verifier_init_sha256", "verifier initialization"
    )
    if observed != init or raw != common.canonical_json_bytes(init):
        raise common.ProductionError("verifier initialization binding drift")
    worker_root = (
        args.verifier_artifact_root / "workers" / f"worker-{args.worker_id}"
    )
    if worker_root.exists() or worker_root.is_symlink():
        raise common.ProductionError("verifier worker namespace is not fresh")
    worker_root.mkdir(mode=0o700)
    args.verifier_worker_root = worker_root


def load_build_evidence(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
]:
    common.assert_no_root_success(args.s3_root)
    build_manifest, build_raw = common.strict_json(
        args.build_artifact_root / "build_manifest.json"
    )
    common.validate_self_hash(
        build_manifest, "build_manifest_sha256", "build manifest"
    )
    build_complete, complete_raw = common.strict_json(
        args.build_artifact_root / "build_complete.json"
    )
    common.validate_self_hash(
        build_complete, "build_complete_sha256", "build complete"
    )
    marker = (
        args.build_artifact_root / "_BUILD_COMPLETE"
    ).read_text(encoding="ascii").strip()
    if (
        build_manifest.get("schema_version") != common.BUILD_MANIFEST_SCHEMA
        or build_manifest.get("status") != "BUILD_COMPLETE_NOT_YET_PRODUCTION"
        or build_manifest.get("build_jobtag") != args.build_jobtag
        or build_manifest.get("s3_root") != str(args.s3_root)
        or build_manifest.get("artifact_root") != str(args.build_artifact_root)
        or build_manifest.get("coord_root") != str(args.build_coord_root)
        or build_manifest.get("worker_count") != args.workers
        or build_manifest.get("rows") != args.gpic_rows
        or build_manifest.get("partition_mode") != args.partition_mode
        or build_manifest.get("source_authority", {}).get("mode")
        != args.source_mode
        or build_complete.get("schema_version") != common.BUILD_COMPLETE_SCHEMA
        or build_complete.get("build_jobtag") != args.build_jobtag
        or build_complete.get("build_manifest_raw_sha256")
        != hashlib.sha256(build_raw).hexdigest()
        or build_complete.get("build_manifest_sha256")
        != build_manifest["build_manifest_sha256"]
        or build_complete.get("dense_index_sha256")
        != build_manifest.get("dense_index", {}).get("sha256")
        or build_complete.get("dense_index_bytes")
        != build_manifest.get("dense_index", {}).get("bytes")
        or marker != build_complete["build_complete_sha256"]
    ):
        raise common.ProductionError("build-complete evidence binding drift")
    common.validate_code_snapshot(build_manifest.get("builder_code", {}))
    common.validate_code_snapshot(build_manifest.get("coordinator_code", {}))
    partition_plan, plan_raw = common.strict_json(
        args.build_coord_root / "partition_plan.json"
    )
    if (
        build_manifest.get("partition_plan_raw_sha256")
        != hashlib.sha256(plan_raw).hexdigest()
        or build_manifest.get("partition_plan_sha256")
        != partition_plan.get("partition_plan_sha256")
    ):
        raise common.ProductionError("build/partition-plan binding drift")
    s3_init, s3_init_raw = common.strict_json(
        args.s3_root / common.S3_INIT_RELATIVE_PATH
    )
    common.validate_self_hash(s3_init, "build_init_sha256", "S3 build init")
    if (
        build_manifest.get("build_init_raw_sha256")
        != hashlib.sha256(s3_init_raw).hexdigest()
        or build_manifest.get("build_init_sha256")
        != s3_init["build_init_sha256"]
        or s3_init.get("partition_plan_raw_sha256")
        != hashlib.sha256(plan_raw).hexdigest()
        or s3_init.get("partition_plan_sha256")
        != partition_plan.get("partition_plan_sha256")
    ):
        raise common.ProductionError("build/S3-init/partition binding drift")
    limits = build_manifest.get("limits")
    if not isinstance(limits, dict) or limits != s3_init.get("limits"):
        raise common.ProductionError("build/S3-init production limits drift")
    if (
        limits.get("max_source_archive_bytes_per_worker")
        != args.max_source_archive_bytes_per_worker
    ):
        raise common.ProductionError("verifier source archive-byte cap drift")
    if args.production_gate:
        common.validate_production_limits(limits)
    if args.source_mode == "tar_sequential":
        full_binding = build_manifest.get("full_tar_mapping")
        full_path = (
            args.build_coord_root
            / "tar_mapping_authority"
            / "full_mapping.json"
        )
        if not isinstance(full_binding, dict):
            raise common.ProductionError("build manifest lacks full tar mapping")
        full_mapping, full_raw = common.strict_json(full_path)
        tar_source.validate_published_full_mapping(
            full_path,
            expected_raw_sha256=hashlib.sha256(full_raw).hexdigest(),
            expected_self_sha256=full_mapping.get("full_mapping_sha256", ""),
        )
        source_authority = build_manifest.get("source_authority")
        if (
            not isinstance(source_authority, dict)
            or build_manifest.get("tar_workers")
            != source_authority.get("tar_workers")
            or full_binding.get("path") != str(full_path)
            or full_binding.get("bytes") != len(full_raw)
            or full_binding.get("raw_sha256")
            != hashlib.sha256(full_raw).hexdigest()
            or full_binding.get("full_mapping_sha256")
            != full_mapping.get("full_mapping_sha256")
            or full_mapping.get("sample_mapping")
            != source_authority.get("mapping")
            or full_mapping.get("payload_reads_before_authority") != 0
            or full_mapping.get("max_source_archive_bytes_per_worker")
            != args.max_source_archive_bytes_per_worker
            or full_mapping.get(
                "source_archive_byte_imbalance_max_over_min", float("inf")
            )
            > common.PRODUCTION_BYTE_IMBALANCE_CAP
            or max(
                full_mapping.get(
                    "worker_source_archive_bytes",
                    [args.max_source_archive_bytes_per_worker + 1],
                )
            )
            > args.max_source_archive_bytes_per_worker
        ):
            raise common.ProductionError("build/full tar mapping closure drift")
    elif build_manifest.get("full_tar_mapping") is not None:
        raise common.ProductionError("small-files build contains tar mapping")
    return (
        build_manifest,
        build_raw,
        build_complete,
        complete_raw,
        partition_plan,
        plan_raw,
        s3_init,
        s3_init_raw,
    )


def approve_fresh_hosts(
    args: argparse.Namespace,
    *,
    code: dict[str, Any],
    build_manifest: dict[str, Any],
    build_raw: bytes,
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], bytes]:
    preflight = {
        "schema_version": common.PRODUCTION_SCHEMA,
        "build_jobtag": args.build_jobtag,
        "verifier_jobtag": args.verifier_jobtag,
        "worker_id": args.worker_id,
        "workers": args.workers,
        "runtime": runtime,
        "code": code,
        "build_manifest_raw_sha256": hashlib.sha256(build_raw).hexdigest(),
        "build_manifest_sha256": build_manifest["build_manifest_sha256"],
    }
    preflight["preflight_sha256"] = common.self_hash(
        preflight, "preflight_sha256"
    )
    own_path = args.verifier_coord_root / f"preflight.worker-{args.worker_id}.json"
    own_raw = common.write_fsx_json_once(own_path, preflight)
    paths = [
        args.verifier_coord_root / f"preflight.worker-{rank}.json"
        for rank in range(args.workers)
    ]
    wait_for_paths(
        paths,
        coord_root=args.verifier_coord_root,
        timeout_seconds=args.barrier_timeout_seconds,
        label="verifier physical-host preflights",
    )
    approval_path = args.verifier_coord_root / "VERIFY_APPROVED.json"
    if args.worker_id == 0:
        values: list[dict[str, Any]] = []
        bindings: list[dict[str, Any]] = []
        for rank, path in enumerate(paths):
            value, raw = common.strict_json(path)
            common.validate_self_hash(
                value, "preflight_sha256", f"verifier preflight {rank}"
            )
            if (
                value.get("build_jobtag") != args.build_jobtag
                or value.get("verifier_jobtag") != args.verifier_jobtag
                or value.get("worker_id") != rank
                or value.get("workers") != args.workers
                or value.get("code") != code
                or value.get("build_manifest_raw_sha256")
                != hashlib.sha256(build_raw).hexdigest()
                or value.get("build_manifest_sha256")
                != build_manifest["build_manifest_sha256"]
            ):
                raise common.ProductionError(
                    f"verifier preflight binding drift worker={rank}"
                )
            values.append(value)
            bindings.append(
                {
                    "worker_id": rank,
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "preflight_sha256": value["preflight_sha256"],
                    "runtime": value["runtime"],
                }
            )
        verifier_boots = common.stable_unique_boot_hashes(
            [value["runtime"] for value in values], "verifier"
        )
        builder_boots = set(build_manifest.get("builder_boot_id_sha256", []))
        if (
            (args.production_gate and len(builder_boots) != args.workers)
            or (args.production_gate and len(verifier_boots) != args.workers)
            or (args.production_gate and not verifier_boots.isdisjoint(builder_boots))
        ):
            raise common.ProductionError(
                "fresh verifier physical-host separation gate failed"
            )
        approval = {
            "schema_version": common.PRODUCTION_SCHEMA,
            "generated_at": common.utc_now(),
            "status": "VERIFY_APPROVED",
            "build_jobtag": args.build_jobtag,
            "verifier_jobtag": args.verifier_jobtag,
            "workers": bindings,
            "worker_count": args.workers,
            "builder_boot_id_sha256": sorted(builder_boots),
            "verifier_boot_id_sha256": sorted(verifier_boots),
            "physical_hosts_disjoint": verifier_boots.isdisjoint(builder_boots),
            "code": code,
            "build_manifest_raw_sha256": hashlib.sha256(build_raw).hexdigest(),
            "build_manifest_sha256": build_manifest["build_manifest_sha256"],
            "publication_semantics": "no_blob_read_before_this_fresh_host_gate",
        }
        approval["approval_sha256"] = common.self_hash(
            approval, "approval_sha256"
        )
        common.write_fsx_json_once(approval_path, approval)
        pilot_core.fsync_dir(args.verifier_coord_root)
    common.wait_for_exact_file(
        approval_path,
        timeout_seconds=args.barrier_timeout_seconds,
        failure_root=args.verifier_coord_root,
        label="fresh verifier approval",
    )
    approval, approval_raw = common.strict_json(approval_path)
    common.validate_self_hash(approval, "approval_sha256", "verify approval")
    if (
        approval.get("status") != "VERIFY_APPROVED"
        or approval.get("build_jobtag") != args.build_jobtag
        or approval.get("verifier_jobtag") != args.verifier_jobtag
        or approval.get("worker_count") != args.workers
        or approval.get("code") != code
        or approval.get("build_manifest_raw_sha256")
        != hashlib.sha256(build_raw).hexdigest()
        or approval.get("build_manifest_sha256")
        != build_manifest["build_manifest_sha256"]
        or approval["workers"][args.worker_id].get("raw_sha256")
        != hashlib.sha256(own_raw).hexdigest()
    ):
        raise common.ProductionError("fresh verifier approval binding drift")
    return approval, approval_raw


def verify_worker_payloads(
    args: argparse.Namespace,
    *,
    build_manifest: dict[str, Any],
    dense_path: Path,
) -> dict[str, Any]:
    rank = args.worker_id
    worker_binding = build_manifest["workers"][rank]
    blobs = [
        value
        for value in build_manifest["blobs"]
        if value.get("worker_id") == rank
    ]
    if not blobs or worker_binding.get("blob_count") != len(blobs):
        raise common.ProductionError("verifier blob assignment count drift")
    if common.file_binding(dense_path) != build_manifest.get("dense_index"):
        raise common.ProductionError("verifier dense-index file binding drift")
    s3_worker_root = args.s3_root / "workers" / f"worker-{rank}"
    common.exact_names(
        s3_worker_root,
        {value["name"] for value in blobs},
        f"verifier S3 worker {rank}",
    )
    dense_fd = os.open(dense_path, os.O_RDONLY)
    dense = mmap.mmap(dense_fd, 0, access=mmap.ACCESS_READ)
    verified_blobs: list[dict[str, Any]] = []
    rows = 0
    payload_bytes = 0
    sidecar_path = (
        args.build_artifact_root / "workers" / f"worker-{rank}" / "blob_order.bin"
    )
    try:
        with sidecar_path.open("rb", buffering=0) as sidecar:
            for sequence, blob in enumerate(blobs):
                expected_blob_id = common.global_blob_id(rank, sequence, args.workers)
                if (
                    blob.get("sequence") != sequence
                    or blob.get("blob_id") != expected_blob_id
                    or blob.get("name") != common.blob_name(expected_blob_id)
                ):
                    raise common.ProductionError(
                        f"verifier blob sequence drift worker={rank} sequence={sequence}"
                    )
                path = s3_worker_root / blob["name"]
                before = common.require_regular_file(path)
                whole = hashlib.sha256()
                offset = 0
                with path.open("rb", buffering=0) as handle:
                    for segment_index in range(blob["segments"]):
                        record = common.read_exact(
                            sidecar,
                            common.RECORD_STRUCT.size,
                            label=(
                                f"verifier sidecar worker={rank} blob={expected_blob_id} "
                                f"segment={segment_index}"
                            ),
                        )
                        row_id, blob_id, record_offset, length, checksum = (
                            common.RECORD_STRUCT.unpack(record)
                        )
                        local_index = row_id - build_manifest["catalog"]["gpic"][
                            "row_base"
                        ]
                        if (
                            blob_id != expected_blob_id
                            or record_offset != offset
                            or length <= 0
                            or local_index < 0
                            or local_index >= args.gpic_rows
                            or bytes(
                                dense[
                                    local_index * common.RECORD_STRUCT.size :
                                    (local_index + 1) * common.RECORD_STRUCT.size
                                ]
                            )
                            != record
                        ):
                            raise common.ProductionError(
                                f"verifier segment/index binding drift row_id={row_id}"
                            )
                        payload = common.read_exact(
                            handle,
                            length,
                            label=f"S3 segment row_id={row_id}",
                        )
                        if hashlib.sha256(payload).digest() != checksum:
                            raise common.ProductionError(
                                f"S3 segment SHA mismatch row_id={row_id}"
                            )
                        whole.update(payload)
                        offset += length
                        rows += 1
                        payload_bytes += length
                    if handle.read(1):
                        raise common.ProductionError(
                            f"S3 blob has extra tail blob={expected_blob_id}"
                        )
                after = common.require_regular_file(path)
                if (
                    before.st_size != blob["bytes"]
                    or after.st_size != blob["bytes"]
                    or offset != blob["bytes"]
                    or whole.hexdigest() != blob["sha256"]
                ):
                    raise common.ProductionError(
                        f"whole S3 blob closure drift blob={expected_blob_id}"
                    )
                verified_blobs.append(
                    {
                        "worker_id": rank,
                        "sequence": sequence,
                        "blob_id": expected_blob_id,
                        "key": str(path),
                        "bytes": blob["bytes"],
                        "segments": blob["segments"],
                        "sha256": whole.hexdigest(),
                    }
                )
            if sidecar.read(1):
                raise common.ProductionError(
                    f"verifier sidecar has extra tail worker={rank}"
                )
    finally:
        dense.close()
        os.close(dense_fd)
    if (
        rows != worker_binding["rows"]
        or payload_bytes != worker_binding["payload_bytes"]
    ):
        raise common.ProductionError("verifier worker row/byte closure drift")
    return {
        "rows": rows,
        "payload_bytes": payload_bytes,
        "blobs": verified_blobs,
    }


def publish_worker_report(
    args: argparse.Namespace,
    *,
    code: dict[str, Any],
    runtime: dict[str, Any],
    build_manifest: dict[str, Any],
    build_raw: bytes,
    approval: dict[str, Any],
    approval_raw: bytes,
    verified: dict[str, Any],
    elapsed_seconds: float,
) -> tuple[dict[str, Any], bytes]:
    report = {
        "schema_version": common.VERIFIER_WORKER_SCHEMA,
        "generated_at": common.utc_now(),
        "build_jobtag": args.build_jobtag,
        "verifier_jobtag": args.verifier_jobtag,
        "worker_id": args.worker_id,
        "workers": args.workers,
        "runtime": runtime,
        "code": code,
        "build_manifest_raw_sha256": hashlib.sha256(build_raw).hexdigest(),
        "build_manifest_sha256": build_manifest["build_manifest_sha256"],
        "verify_approval_raw_sha256": hashlib.sha256(approval_raw).hexdigest(),
        "verify_approval_sha256": approval["approval_sha256"],
        "rows": verified["rows"],
        "payload_bytes": verified["payload_bytes"],
        "blobs": verified["blobs"],
        "elapsed_seconds": elapsed_seconds,
        "integrity_errors": 0,
        "short_reads": 0,
        "fallbacks": 0,
    }
    report["report_sha256"] = common.self_hash(report, "report_sha256")
    raw = common.write_fsx_json_once(
        args.verifier_worker_root / "report.json", report
    )
    common.write_fsx_bytes_once(
        args.verifier_worker_root / "_SUCCESS",
        (report["report_sha256"] + "\n").encode("ascii"),
    )
    pilot_core.fsync_dir(args.verifier_worker_root)
    status = {
        "schema_version": common.PRODUCTION_SCHEMA,
        "build_jobtag": args.build_jobtag,
        "verifier_jobtag": args.verifier_jobtag,
        "worker_id": args.worker_id,
        "report_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "report_sha256": report["report_sha256"],
        "rows": report["rows"],
        "payload_bytes": report["payload_bytes"],
    }
    common.write_fsx_json_once(
        args.verifier_coord_root / f"verified.worker-{args.worker_id}.json",
        status,
    )
    pilot_core.fsync_dir(args.verifier_coord_root)
    return report, raw


def close_and_publish(
    args: argparse.Namespace,
    *,
    code: dict[str, Any],
    build_manifest: dict[str, Any],
    build_raw: bytes,
    build_complete: dict[str, Any],
    complete_raw: bytes,
    partition_plan: dict[str, Any],
    plan_raw: bytes,
    s3_init: dict[str, Any],
    s3_init_raw: bytes,
    approval: dict[str, Any],
    approval_raw: bytes,
) -> None:
    statuses = [
        args.verifier_coord_root / f"verified.worker-{rank}.json"
        for rank in range(args.workers)
    ]
    wait_for_paths(
        statuses,
        coord_root=args.verifier_coord_root,
        timeout_seconds=args.barrier_timeout_seconds,
        label="full verifier reports",
    )
    if args.worker_id != 0:
        common.wait_for_exact_file(
            args.s3_root / "_SUCCESS",
            timeout_seconds=args.barrier_timeout_seconds,
            failure_root=args.verifier_coord_root,
            label="production root success",
        )
        return
    reports: list[dict[str, Any]] = []
    report_bindings: list[dict[str, Any]] = []
    all_blobs: list[dict[str, Any]] = []
    keys: set[str] = set()
    for rank, status_path in enumerate(statuses):
        status, status_raw = common.strict_json(status_path)
        root = args.verifier_artifact_root / "workers" / f"worker-{rank}"
        common.exact_names(root, {"report.json", "_SUCCESS"}, f"verifier {rank}")
        report, report_raw = common.strict_json(root / "report.json")
        common.validate_self_hash(report, "report_sha256", f"verify report {rank}")
        marker = (root / "_SUCCESS").read_text(encoding="ascii").strip()
        if (
            status_raw != common.canonical_json_bytes(status)
            or status.get("worker_id") != rank
            or status.get("report_raw_sha256")
            != hashlib.sha256(report_raw).hexdigest()
            or status.get("report_sha256") != report["report_sha256"]
            or marker != report["report_sha256"]
            or report.get("schema_version") != common.VERIFIER_WORKER_SCHEMA
            or report.get("build_jobtag") != args.build_jobtag
            or report.get("verifier_jobtag") != args.verifier_jobtag
            or report.get("worker_id") != rank
            or report.get("workers") != args.workers
            or report.get("code") != code
            or report.get("build_manifest_raw_sha256")
            != hashlib.sha256(build_raw).hexdigest()
            or report.get("build_manifest_sha256")
            != build_manifest["build_manifest_sha256"]
            or report.get("verify_approval_raw_sha256")
            != hashlib.sha256(approval_raw).hexdigest()
            or report.get("verify_approval_sha256")
            != approval["approval_sha256"]
            or report.get("integrity_errors") != 0
            or report.get("short_reads") != 0
            or report.get("fallbacks") != 0
        ):
            raise common.ProductionError(
                f"full verifier report binding drift worker={rank}"
            )
        for blob in report.get("blobs", []):
            key = blob.get("key")
            if key in keys:
                raise common.ProductionError("full verifier duplicate blob key")
            keys.add(key)
            all_blobs.append(blob)
        reports.append(report)
        report_bindings.append(
            {
                "worker_id": rank,
                "root": str(root),
                "report_raw_sha256": hashlib.sha256(report_raw).hexdigest(),
                "report_sha256": report["report_sha256"],
                "rows": report["rows"],
                "payload_bytes": report["payload_bytes"],
                "blob_count": len(report["blobs"]),
                "runtime": report["runtime"],
            }
        )
    expected_keys = {value["key"] for value in build_manifest["blobs"]}
    if (
        keys != expected_keys
        or len(all_blobs) != build_manifest["blob_count"]
        or sum(value["rows"] for value in reports) != args.gpic_rows
        or sum(value["payload_bytes"] for value in reports)
        != build_manifest["payload_bytes"]
    ):
        raise common.ProductionError("full verifier global row/byte/key union drift")
    verifier_boots = common.stable_unique_boot_hashes(
        [value["runtime"] for value in reports], "verifier report"
    )
    if sorted(verifier_boots) != approval["verifier_boot_id_sha256"]:
        raise common.ProductionError("verifier report/preflight host drift")
    common.exact_names(
        args.s3_root,
        {common.S3_INIT_RELATIVE_PATH, "controls", "workers"},
        "prepublication S3 root",
    )
    common.exact_names(args.s3_controls_root, set(), "prepublication controls")
    summary = {
        "schema_version": common.VERIFIER_SUMMARY_SCHEMA,
        "generated_at": common.utc_now(),
        "status": "PASS",
        "build_jobtag": args.build_jobtag,
        "verifier_jobtag": args.verifier_jobtag,
        "workers": report_bindings,
        "worker_count": args.workers,
        "rows": args.gpic_rows,
        "payload_bytes": build_manifest["payload_bytes"],
        "blob_count": build_manifest["blob_count"],
        "keys_sha256": hashlib.sha256(
            common.canonical_json_bytes(sorted(keys))
        ).hexdigest(),
        "builder_boot_id_sha256": build_manifest["builder_boot_id_sha256"],
        "verifier_boot_id_sha256": sorted(verifier_boots),
        "physical_hosts_disjoint": approval["physical_hosts_disjoint"],
        "build_manifest_raw_sha256": hashlib.sha256(build_raw).hexdigest(),
        "build_manifest_sha256": build_manifest["build_manifest_sha256"],
        "build_complete_raw_sha256": hashlib.sha256(complete_raw).hexdigest(),
        "build_complete_sha256": build_complete["build_complete_sha256"],
        "verify_approval_raw_sha256": hashlib.sha256(approval_raw).hexdigest(),
        "verify_approval_sha256": approval["approval_sha256"],
        "code": code,
        "integrity_errors": 0,
        "short_reads": 0,
        "fallbacks": 0,
    }
    summary["verifier_summary_sha256"] = common.self_hash(
        summary, "verifier_summary_sha256"
    )
    summary_raw = common.write_fsx_json_once(
        args.verifier_artifact_root / "verifier_summary.json", summary
    )
    dense_path = args.build_artifact_root / "dense_index.bin"
    release_storage = {
        common.RELEASE_STORAGE_S3_ROOT_FIELD: str(args.s3_root),
        common.RELEASE_STORAGE_INDEX_PATH_FIELD: str(dense_path),
        common.RELEASE_STORAGE_S3_INIT_FIELD: {
            common.RELEASE_STORAGE_S3_INIT_PATH_FIELD: common.S3_INIT_RELATIVE_PATH,
            common.RELEASE_STORAGE_S3_INIT_BYTES_FIELD: len(s3_init_raw),
            common.RELEASE_STORAGE_S3_INIT_RAW_SHA256_FIELD: hashlib.sha256(
                s3_init_raw
            ).hexdigest(),
        },
    }
    canonical = {
        "schema_version": common.CANONICAL_MANIFEST_SCHEMA,
        "generated_at": common.utc_now(),
        "status": "PRODUCTION_VERIFIED",
        "build_jobtag": args.build_jobtag,
        "verifier_jobtag": args.verifier_jobtag,
        common.RELEASE_FIELD: {
            common.RELEASE_STORAGE_FIELD: release_storage,
        },
        "catalog": build_manifest["catalog"],
        "partition_mode": build_manifest["partition_mode"],
        "partition_plan_raw_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "partition_plan_sha256": partition_plan["partition_plan_sha256"],
        "source_authority": build_manifest["source_authority"],
        "limits": build_manifest["limits"],
        "tar_workers": build_manifest.get("tar_workers"),
        "full_tar_mapping": build_manifest.get("full_tar_mapping"),
        "dense_index": build_manifest["dense_index"],
        "record_struct": common.RECORD_STRUCT.format,
        "record_bytes": common.RECORD_STRUCT.size,
        "rows": args.gpic_rows,
        "payload_bytes": build_manifest["payload_bytes"],
        "blob_count": build_manifest["blob_count"],
        "blobs": build_manifest["blobs"],
        "workers": build_manifest["workers"],
        "build_manifest_raw_sha256": hashlib.sha256(build_raw).hexdigest(),
        "build_manifest_sha256": build_manifest["build_manifest_sha256"],
        "build_complete_raw_sha256": hashlib.sha256(complete_raw).hexdigest(),
        "build_complete_sha256": build_complete["build_complete_sha256"],
        "verifier_summary_raw_sha256": hashlib.sha256(summary_raw).hexdigest(),
        "verifier_summary_sha256": summary["verifier_summary_sha256"],
        "s3_init_sha256": s3_init["build_init_sha256"],
        "verifier_code": code,
        "publication_semantics": (
            "canonical_manifest_then_fsx_success_then_single_s3_root_success_absolute_last"
        ),
    }
    canonical["manifest_sha256"] = common.self_hash(
        canonical, "manifest_sha256"
    )
    canonical_raw = common.canonical_json_bytes(canonical)
    marker_raw = (canonical["manifest_sha256"] + "\n").encode("ascii")
    projected_s3_bytes = (
        build_manifest["payload_bytes"]
        + len(s3_init_raw)
        + len(canonical_raw)
        + len(marker_raw)
    )
    if projected_s3_bytes > args.max_s3_total_bytes:
        raise common.ProductionError("final exact S3 total cap exceeded")
    common.write_fsx_bytes_once(
        args.verifier_artifact_root / "canonical_manifest.json", canonical_raw
    )
    pilot_core.fsync_dir(args.verifier_artifact_root)
    fsx_bytes = common.fsx_tree_bytes(
        [
            args.build_artifact_root,
            args.build_coord_root,
            args.verifier_artifact_root,
            args.verifier_coord_root,
        ]
    )
    if fsx_bytes + len(marker_raw) > args.max_fsx_incremental_bytes:
        raise common.ProductionError("final production FSx cap exceeded")
    common.write_fsx_bytes_once(
        args.verifier_artifact_root / "_SUCCESS", marker_raw
    )
    pilot_core.fsync_dir(args.verifier_artifact_root)
    common.write_s3_bytes_once(args.s3_controls_root / "manifest.json", canonical_raw)
    common.exact_names(
        args.s3_controls_root,
        {"manifest.json"},
        "committed controls",
    )
    # Absolute final storage mutation.  No evidence write is permitted below.
    common.write_s3_bytes_once(args.s3_root / "_SUCCESS", marker_raw)


def run(args: argparse.Namespace) -> None:
    code = common.code_snapshot(
        runner=Path(__file__),
        job_spec=args.job_spec,
        extra_files={
            "builder_runner": args.builder_runner,
            "builder_job_spec": args.builder_job_spec,
            "coordinator_runner": args.coordinator_runner,
            "coordinator_job_spec": args.coordinator_job_spec,
            "tar_source": Path(tar_source.__file__),
        },
    )
    common.require_regular_dir(args.build_artifact_root)
    common.require_regular_dir(args.build_coord_root)
    common.require_regular_dir(args.s3_root)
    if not args.s3_controls_root.exists():
        args.s3_controls_root.mkdir(mode=0o700)
    common.exact_names(
        args.s3_root,
        {common.S3_INIT_RELATIVE_PATH, "controls", "workers"},
        "verifier preflight S3 root",
    )
    common.exact_names(args.s3_controls_root, set(), "verifier preflight controls")
    initialize_verifier(args, code)
    (
        build_manifest,
        build_raw,
        build_complete,
        complete_raw,
        partition_plan,
        plan_raw,
        s3_init,
        s3_init_raw,
    ) = load_build_evidence(args)
    catalog = common.load_production_catalog(
        args.catalog_root, production_gate=args.production_gate
    )
    builder.validate_partition_plan(
        partition_plan,
        build_jobtag=args.build_jobtag,
        partition_mode=args.partition_mode,
        workers=args.workers,
        catalog=catalog,
        source_prefix=args.source_prefix,
        production_gate=args.production_gate,
    )
    runtime = common.runtime_identity()
    approval, approval_raw = approve_fresh_hosts(
        args,
        code=code,
        build_manifest=build_manifest,
        build_raw=build_raw,
        runtime=runtime,
    )
    try:
        started = time.monotonic()
        verified = verify_worker_payloads(
            args,
            build_manifest=build_manifest,
            dense_path=args.build_artifact_root / "dense_index.bin",
        )
        common.validate_catalog_closure(catalog)
        common.validate_code_snapshot(code)
        publish_worker_report(
            args,
            code=code,
            runtime=runtime,
            build_manifest=build_manifest,
            build_raw=build_raw,
            approval=approval,
            approval_raw=approval_raw,
            verified=verified,
            elapsed_seconds=time.monotonic() - started,
        )
        close_and_publish(
            args,
            code=code,
            build_manifest=build_manifest,
            build_raw=build_raw,
            build_complete=build_complete,
            complete_raw=complete_raw,
            partition_plan=partition_plan,
            plan_raw=plan_raw,
            s3_init=s3_init,
            s3_init_raw=s3_init_raw,
            approval=approval,
            approval_raw=approval_raw,
        )
        marker = (args.s3_root / "_SUCCESS").read_text(encoding="ascii").strip()
        print(
            "OFFLINE_GPIC_S3_PRODUCTION_VERIFY_PASS "
            f"build_jobtag={args.build_jobtag} verifier_jobtag={args.verifier_jobtag} "
            f"worker={args.worker_id} manifest_sha256={marker}",
            flush=True,
        )
    except BaseException as exc:
        if isinstance(exc, pilot_core.CommittedSyncUnconfirmed) or common.success_or_unknown(
            args.s3_root / "_SUCCESS"
        ):
            print(
                "OFFLINE_GPIC_S3_PRODUCTION_RELEASE_COMMITTED_SYNC_UNCONFIRMED",
                file=sys.stderr,
                flush=True,
            )
        else:
            try:
                common.write_failure(
                    args.verifier_coord_root,
                    build_jobtag=args.build_jobtag,
                    actor="verifier",
                    actor_id=args.worker_id,
                    exc=exc,
                )
            except BaseException:
                pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-artifact-root", type=Path, required=True)
    parser.add_argument("--build-coord-root", type=Path, required=True)
    parser.add_argument("--verifier-artifact-prefix", type=Path, required=True)
    parser.add_argument("--verifier-coord-prefix", type=Path, required=True)
    parser.add_argument("--s3-root", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--build-jobtag", required=True)
    parser.add_argument("--verifier-jobtag", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--workers", type=int, default=common.PRODUCTION_WORKERS)
    parser.add_argument("--gpic-rows", type=int, default=common.PRODUCTION_GPIC_ROWS)
    parser.add_argument(
        "--partition-mode",
        choices=common.PRODUCTION_PARTITION_MODES,
        required=True,
    )
    parser.add_argument("--source-prefix", default=common.PRODUCTION_SOURCE_PREFIX)
    parser.add_argument(
        "--source-mode", choices=common.PRODUCTION_SOURCE_MODES, required=True
    )
    parser.add_argument(
        "--max-s3-total-bytes",
        type=int,
        default=common.PRODUCTION_MAX_S3_TOTAL_BYTES,
    )
    parser.add_argument(
        "--max-fsx-incremental-bytes",
        type=int,
        default=common.PRODUCTION_MAX_FSX_INCREMENTAL_BYTES,
    )
    parser.add_argument(
        "--max-source-archive-bytes-per-worker",
        type=int,
        default=common.PRODUCTION_MAX_SOURCE_ARCHIVE_BYTES_PER_WORKER,
    )
    parser.add_argument("--builder-runner", type=Path, required=True)
    parser.add_argument("--builder-job-spec", type=Path, required=True)
    parser.add_argument("--coordinator-runner", type=Path, required=True)
    parser.add_argument("--coordinator-job-spec", type=Path, required=True)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--barrier-timeout-seconds", type=float, default=43_200.0)
    parser.add_argument("--verify-authority", required=True)
    parser.add_argument("--production-gate", action="store_true")
    args = parser.parse_args(argv)
    for label in (
        "build_artifact_root",
        "build_coord_root",
        "verifier_artifact_prefix",
        "verifier_coord_prefix",
        "s3_root",
        "catalog_root",
        "builder_runner",
        "builder_job_spec",
        "coordinator_runner",
        "coordinator_job_spec",
        "job_spec",
    ):
        setattr(args, label, getattr(args, label).expanduser().resolve(strict=True))
    if args.verify_authority != common.VERIFY_AUTHORITY:
        raise common.ProductionError("full verification authority is absent")
    for label in ("build_jobtag", "verifier_jobtag"):
        value = getattr(args, label)
        if (
            not value
            or Path(value).name != value
            or value in {".", "..", "latest"}
        ):
            raise common.ProductionError(f"{label} is malformed/non-exact")
    if args.build_jobtag == args.verifier_jobtag:
        raise common.ProductionError("build/verifier jobtags must be independent")
    if (
        args.build_artifact_root.name != args.build_jobtag
        or args.build_coord_root.name != args.build_jobtag
        or args.s3_root.name != args.build_jobtag
        or not (0 <= args.worker_id < args.workers)
    ):
        raise common.ProductionError("verifier exact build root/worker drift")
    if not str(args.build_artifact_root).startswith("/fsx/") or not str(
        args.verifier_artifact_prefix
    ).startswith("/fsx/"):
        raise common.ProductionError("verifier FSx roots escaped /fsx")
    if (
        args.barrier_timeout_seconds <= 0
        or args.max_source_archive_bytes_per_worker < 1
    ):
        raise common.ProductionError("verifier barrier timeout must be positive")
    if args.production_gate and (
        args.workers != common.PRODUCTION_WORKERS
        or args.gpic_rows != common.PRODUCTION_GPIC_ROWS
        or args.source_prefix != common.PRODUCTION_SOURCE_PREFIX
        or args.max_s3_total_bytes != common.PRODUCTION_MAX_S3_TOTAL_BYTES
        or args.max_source_archive_bytes_per_worker
        != common.PRODUCTION_MAX_SOURCE_ARCHIVE_BYTES_PER_WORKER
        or args.max_fsx_incremental_bytes
        != common.PRODUCTION_MAX_FSX_INCREMENTAL_BYTES
        or not str(args.s3_root).startswith(
            "/s3/evan/artifacts/offline_gpic_blob_v1/"
        )
    ):
        raise common.ProductionError("frozen production verifier constants drift")
    args.verifier_artifact_root = (
        args.verifier_artifact_prefix
        / args.build_jobtag
        / args.verifier_jobtag
    )
    args.verifier_coord_root = (
        args.verifier_coord_prefix / args.build_jobtag / args.verifier_jobtag
    )
    args.s3_controls_root = args.s3_root / "controls"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
