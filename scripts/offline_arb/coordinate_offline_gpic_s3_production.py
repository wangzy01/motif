#!/usr/bin/env python3
"""Close eight GPIC production workers and merge the exact dense index.

This is Phase C only.  It publishes durable build-complete evidence on FSx but
is forbidden from creating the production S3 root ``_SUCCESS``.
"""

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


def validate_build_initialization(
    args: argparse.Namespace,
    *,
    expected_code: dict[str, Any],
    catalog_manifest: dict[str, Any],
    topology_evidence: dict[str, Any],
    limits: dict[str, Any],
    partition_plan: dict[str, Any],
    partition_plan_raw: bytes,
) -> tuple[dict[str, Any], bytes]:
    observed, observed_raw = common.strict_json(
        args.s3_root / common.S3_INIT_RELATIVE_PATH
    )
    source_authority = observed.get("source_authority")
    if (
        not isinstance(source_authority, dict)
        or source_authority.get("mode") != args.source_mode
    ):
        raise common.ProductionError("coordinator source authority/mode drift")
    expected = common.make_build_init(
        build_jobtag=args.build_jobtag,
        workers=args.workers,
        s3_root=args.s3_root,
        artifact_root=args.artifact_root,
        coord_root=args.coord_root,
        code=expected_code,
        catalog=catalog_manifest,
        topology_evidence=topology_evidence,
        limits=limits,
        partition_plan=partition_plan,
        partition_plan_raw=partition_plan_raw,
        source_authority=source_authority,
    )
    if observed != expected or observed_raw != common.canonical_json_bytes(expected):
        raise common.ProductionError("coordinator S3 init exact binding drift")
    value, raw = observed, observed_raw
    coord, coord_raw = common.strict_json(args.coord_root / "_INIT.json")
    expected_coord = common.make_coord_init(build_init=value, build_init_raw=raw)
    common.validate_self_hash(coord, "coord_init_sha256", "coord initialization")
    if coord != expected_coord or coord_raw != common.canonical_json_bytes(
        expected_coord
    ):
        raise common.ProductionError("coordinator FSx/S3 init binding drift")
    return value, raw


def validate_approval(
    args: argparse.Namespace,
    *,
    build_init: dict[str, Any],
    build_init_raw: bytes,
    builder_code: dict[str, Any],
    catalog_manifest: dict[str, Any],
    partition_plan: dict[str, Any],
    partition_plan_raw: bytes,
) -> tuple[dict[str, Any], bytes]:
    approval, raw = common.strict_json(args.coord_root / "UPLOAD_APPROVED.json")
    common.validate_self_hash(approval, "approval_sha256", "upload approval")
    if (
        approval.get("status") != "UPLOAD_APPROVED"
        or approval.get("build_jobtag") != args.build_jobtag
        or approval.get("worker_count") != args.workers
        or approval.get("rows") != args.gpic_rows
        or approval.get("payload_bytes", args.max_s3_payload_bytes + 1)
        > args.max_s3_payload_bytes
        or approval.get("payload_bytes", 0)
        + approval.get("s3_controls_reserve_bytes", 0)
        > args.max_s3_total_bytes
        or approval.get("byte_imbalance_cap") != args.max_byte_imbalance
        or approval.get("byte_imbalance_max_over_min", float("inf"))
        > args.max_byte_imbalance
        or approval.get("code") != builder_code
        or approval.get("catalog") != catalog_manifest
        or approval.get("build_init_raw_sha256")
        != hashlib.sha256(build_init_raw).hexdigest()
        or approval.get("build_init_sha256")
        != build_init["build_init_sha256"]
        or approval.get("partition_mode") != args.partition_mode
        or approval.get("partition_plan_raw_sha256")
        != hashlib.sha256(partition_plan_raw).hexdigest()
        or approval.get("partition_plan_sha256")
        != partition_plan["partition_plan_sha256"]
    ):
        raise common.ProductionError("upload approval closure drift")
    source_authority = build_init.get("source_authority")
    if not isinstance(source_authority, dict):
        raise common.ProductionError("upload approval source authority is malformed")
    if args.source_mode == "tar_sequential":
        full_binding = approval.get("full_tar_mapping")
        full_path = args.coord_root / "tar_mapping_authority" / "full_mapping.json"
        if not isinstance(full_binding, dict):
            raise common.ProductionError("upload approval lacks full tar mapping")
        full_mapping, full_raw = common.strict_json(full_path)
        tar_source.validate_published_full_mapping(
            full_path,
            expected_raw_sha256=hashlib.sha256(full_raw).hexdigest(),
            expected_self_sha256=full_mapping.get("full_mapping_sha256", ""),
        )
        if (
            approval.get("tar_workers") != source_authority.get("tar_workers")
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
            != build_init["limits"][
                "max_source_archive_bytes_per_worker"
            ]
            or full_mapping.get(
                "source_archive_byte_imbalance_max_over_min", float("inf")
            )
            > common.PRODUCTION_BYTE_IMBALANCE_CAP
            or max(
                full_mapping.get("worker_source_archive_bytes", [
                    build_init["limits"][
                        "max_source_archive_bytes_per_worker"
                    ]
                    + 1
                ])
            )
            > build_init["limits"]["max_source_archive_bytes_per_worker"]
        ):
            raise common.ProductionError("upload approval tar mapping closure drift")
    elif approval.get("full_tar_mapping") is not None:
        raise common.ProductionError("small-files approval contains tar mapping")
    workers = approval.get("workers")
    if (
        not isinstance(workers, list)
        or [value.get("worker_id") for value in workers]
        != list(range(args.workers))
    ):
        raise common.ProductionError("upload approval worker sequence drift")
    return approval, raw


def validate_workers(
    args: argparse.Namespace,
    *,
    build_init: dict[str, Any],
    build_init_raw: bytes,
    builder_code: dict[str, Any],
    catalog: pilot_core.CatalogBinding,
    catalog_manifest: dict[str, Any],
    topology_evidence: dict[str, Any],
    approval: dict[str, Any],
    approval_raw: bytes,
    partition_plan: dict[str, Any],
    partition_plan_raw: bytes,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    receipt_bindings: list[dict[str, Any]] = []
    all_blobs: list[dict[str, Any]] = []
    all_keys: set[str] = set()
    payloads: list[int] = []
    runtimes: list[dict[str, Any]] = []
    for rank in range(args.workers):
        root = args.fsx_workers_root / f"worker-{rank}"
        expected_names = {
            "row_fragment.bin",
            "blob_order.bin",
            "local_manifest.json",
            "preupload.json",
            "_PREUPLOAD_READY",
            "receipt.json",
            "_SUCCESS",
        }
        if args.source_mode == "tar_sequential":
            expected_names.add("tar_telemetry.jsonl")
        common.exact_names(
            root,
            expected_names,
            f"completed builder worker {rank}",
        )
        receipt, receipt_raw = common.strict_json(root / "receipt.json")
        common.validate_self_hash(receipt, "receipt_sha256", f"receipt {rank}")
        marker = (root / "_SUCCESS").read_text(encoding="ascii").strip()
        status, status_raw = common.strict_json(
            args.coord_root / f"uploaded.worker-{rank}.json"
        )
        preupload, preupload_raw = common.strict_json(root / "preupload.json")
        common.validate_self_hash(
            preupload, "preupload_sha256", f"preupload {rank}"
        )
        local_manifest, local_raw = common.strict_json(root / "local_manifest.json")
        common.validate_self_hash(
            local_manifest, "local_manifest_sha256", f"local manifest {rank}"
        )
        if (
            receipt.get("schema_version") != common.WORKER_RECEIPT_SCHEMA
            or marker != receipt["receipt_sha256"]
            or receipt.get("build_jobtag") != args.build_jobtag
            or receipt.get("worker_id") != rank
            or receipt.get("workers") != args.workers
            or receipt.get("root") != str(root)
            or receipt.get("s3_worker_root")
            != str(args.s3_workers_root / f"worker-{rank}")
            or receipt.get("rows") != partition_plan["worker_rows"][rank]
            or receipt.get("code") != builder_code
            or receipt.get("catalog") != catalog_manifest
            or receipt.get("topology_evidence") != topology_evidence
            or receipt.get("build_init_raw_sha256")
            != hashlib.sha256(build_init_raw).hexdigest()
            or receipt.get("build_init_sha256")
            != build_init["build_init_sha256"]
            or receipt.get("partition_mode") != args.partition_mode
            or receipt.get("partition_plan_raw_sha256")
            != hashlib.sha256(partition_plan_raw).hexdigest()
            or receipt.get("partition_plan_sha256")
            != partition_plan["partition_plan_sha256"]
            or receipt.get("upload_approval_raw_sha256")
            != hashlib.sha256(approval_raw).hexdigest()
            or receipt.get("upload_approval_sha256")
            != approval["approval_sha256"]
            or receipt.get("preupload_raw_sha256")
            != hashlib.sha256(preupload_raw).hexdigest()
            or receipt.get("preupload_sha256")
            != preupload["preupload_sha256"]
            or receipt.get("local_manifest_raw_sha256")
            != hashlib.sha256(local_raw).hexdigest()
            or receipt.get("local_manifest_sha256")
            != local_manifest["local_manifest_sha256"]
            or receipt.get("tar_telemetry")
            != local_manifest.get("tar_telemetry")
            or receipt.get("tar_mapping_evidence")
            != local_manifest.get("tar_mapping_evidence")
            or status.get("build_jobtag") != args.build_jobtag
            or status.get("worker_id") != rank
            or status.get("receipt_raw_sha256")
            != hashlib.sha256(receipt_raw).hexdigest()
            or status.get("receipt_sha256") != receipt["receipt_sha256"]
            or status.get("rows") != receipt["rows"]
            or status.get("payload_bytes") != receipt["payload_bytes"]
        ):
            raise common.ProductionError(
                f"completed worker evidence binding drift worker={rank}"
            )
        if common.canonical_json_bytes(status) != status_raw:
            raise common.ProductionError(f"uploaded status canonical drift {rank}")
        for label in ("row_fragment", "blob_order"):
            binding = receipt.get(label)
            if (
                not isinstance(binding, dict)
                or binding.get("bytes")
                != partition_plan["worker_rows"][rank]
                * common.RECORD_STRUCT.size
                or common.file_binding(root / binding.get("name", "")) != binding
            ):
                raise common.ProductionError(
                    f"worker fragment binding drift worker={rank} file={label}"
                )
        if args.source_mode == "tar_sequential":
            telemetry = receipt.get("tar_telemetry")
            mapping_evidence = receipt.get("tar_mapping_evidence")
            if (
                not isinstance(telemetry, dict)
                or not isinstance(telemetry.get("file"), dict)
                or common.file_binding(root / "tar_telemetry.jsonl")
                != telemetry["file"]
                or telemetry.get("rows")
                != partition_plan["worker_rows"][rank]
                or telemetry.get("max_source_archive_bytes_per_worker")
                != build_init["limits"][
                    "max_source_archive_bytes_per_worker"
                ]
                or telemetry.get("max_selected_bytes_per_tar")
                != build_init["source_authority"].get(
                    "max_selected_bytes_per_tar"
                )
                or not isinstance(mapping_evidence, dict)
                or mapping_evidence.get("full")
                != approval.get("full_tar_mapping")
                or receipt.get("source", {}).get("tar_workers")
                != approval.get("tar_workers")
                or receipt.get("source", {}).get(
                    "max_source_archive_bytes_per_worker"
                )
                != build_init["limits"][
                    "max_source_archive_bytes_per_worker"
                ]
                or receipt.get("source", {}).get(
                    "max_selected_bytes_per_tar"
                )
                != build_init["source_authority"].get(
                    "max_selected_bytes_per_tar"
                )
            ):
                raise common.ProductionError(
                    f"worker tar evidence closure drift worker={rank}"
                )
        blobs = receipt.get("blobs")
        expected_public_blobs = [
            {key: item for key, item in blob.items() if key != "local_path"}
            for blob in local_manifest.get("blobs", [])
        ]
        if not isinstance(blobs, list) or blobs != expected_public_blobs:
            raise common.ProductionError(f"worker blob list drift worker={rank}")
        s3_worker = args.s3_workers_root / f"worker-{rank}"
        common.exact_names(
            s3_worker,
            {value["name"] for value in blobs},
            f"S3 production worker {rank}",
        )
        worker_bytes = 0
        worker_rows = 0
        for sequence, blob in enumerate(blobs):
            blob_id = common.global_blob_id(rank, sequence, args.workers)
            name = common.blob_name(blob_id)
            key = str(s3_worker / name)
            if (
                blob.get("sequence") != sequence
                or blob.get("blob_id") != blob_id
                or blob.get("name") != name
                or blob.get("key") != key
                or not common.is_sha256(blob.get("sha256"))
                or type(blob.get("bytes")) is not int
                or blob["bytes"] <= 0
                or type(blob.get("segments")) is not int
                or blob["segments"] <= 0
                or key in all_keys
                or common.require_regular_file(Path(key)).st_size != blob["bytes"]
            ):
                raise common.ProductionError(
                    f"S3 blob identity/size drift worker={rank} sequence={sequence}"
                )
            all_keys.add(key)
            all_blobs.append({**blob, "worker_id": rank})
            worker_bytes += blob["bytes"]
            worker_rows += blob["segments"]
        if (
            worker_bytes != receipt.get("payload_bytes")
            or worker_rows != partition_plan["worker_rows"][rank]
            or worker_bytes > args.max_s3_bytes_per_worker
        ):
            raise common.ProductionError(
                f"worker byte/row union drift worker={rank}"
            )
        receipt_bindings.append(
            {
                "worker_id": rank,
                "root": str(root),
                "receipt_raw_sha256": hashlib.sha256(receipt_raw).hexdigest(),
                "receipt_sha256": receipt["receipt_sha256"],
                "row_fragment": receipt["row_fragment"],
                "blob_order": receipt["blob_order"],
                "rows": receipt["rows"],
                "payload_bytes": receipt["payload_bytes"],
                "blob_count": len(blobs),
                "runtime": receipt["runtime"],
                "tar_telemetry": receipt.get("tar_telemetry"),
                "tar_mapping_worker": (
                    receipt.get("tar_mapping_evidence", {}).get("worker")
                    if isinstance(receipt.get("tar_mapping_evidence"), dict)
                    else None
                ),
            }
        )
        payloads.append(worker_bytes)
        runtimes.append(receipt["runtime"])
    if (
        len(all_keys) != len(all_blobs)
        or sum(value["rows"] for value in receipt_bindings) != args.gpic_rows
        or sum(payloads) != approval["payload_bytes"]
        or sum(payloads) > args.max_s3_payload_bytes
        or max(payloads) / min(payloads) > args.max_byte_imbalance
    ):
        raise common.ProductionError("global worker union/cap closure failed")
    boot_hashes = common.stable_unique_boot_hashes(runtimes, "builder")
    if args.production_gate and len(boot_hashes) != args.workers:
        raise common.ProductionError("builder physical-host count drift")
    return receipt_bindings, all_blobs, boot_hashes


def merge_dense_index(
    args: argparse.Namespace,
    *,
    catalog: pilot_core.CatalogBinding,
    blobs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    path = args.artifact_root / "dense_index.bin"
    if path.exists() or path.is_symlink():
        raise common.ProductionError("dense index destination is not fresh")
    blob_ids = {value["blob_id"] for value in blobs}
    handles = [
        (args.fsx_workers_root / f"worker-{rank}" / "row_fragment.bin").open(
            "rb", buffering=0
        )
        for rank in range(args.workers)
    ]
    digest = hashlib.sha256()
    written = 0
    output_buffer = bytearray()
    try:
        current: list[bytes | None] = []
        for rank, handle in enumerate(handles):
            record = handle.read(common.RECORD_STRUCT.size)
            if len(record) != common.RECORD_STRUCT.size:
                raise common.ProductionError(
                    f"worker row fragment empty/short rank={rank}"
                )
            current.append(record)
        with path.open("xb", buffering=0) as output:
            for slot in range(args.gpic_rows):
                expected_row_id = catalog.gpic_row_base + slot
                owners = [
                    rank
                    for rank, record_value in enumerate(current)
                    if record_value is not None
                    and common.RECORD_STRUCT.unpack(record_value)[0]
                    == expected_row_id
                ]
                if len(owners) != 1:
                    raise common.ProductionError(
                        f"dense fragment missing/duplicate explicit row slot={slot}"
                    )
                rank = owners[0]
                record = current[rank]
                if record is None:
                    raise common.ProductionError("dense merge internal owner hole")
                row_id, blob_id, _offset, length, checksum = (
                    common.RECORD_STRUCT.unpack(record)
                )
                if (
                    row_id != expected_row_id
                    or blob_id % args.workers != rank
                    or blob_id not in blob_ids
                    or length <= 0
                    or not any(checksum)
                ):
                    raise common.ProductionError(
                        f"dense index explicit row/blob drift slot={slot}"
                    )
                output_buffer.extend(record)
                digest.update(record)
                written += len(record)
                if len(output_buffer) >= 8 * common.MIB:
                    common.write_all(output, output_buffer, label="dense_index.bin")
                    output_buffer.clear()
                next_record = handles[rank].read(common.RECORD_STRUCT.size)
                if not next_record:
                    current[rank] = None
                elif len(next_record) != common.RECORD_STRUCT.size:
                    raise common.ProductionError(
                        f"worker row fragment short tail rank={rank}"
                    )
                else:
                    next_row_id = common.RECORD_STRUCT.unpack(next_record)[0]
                    if next_row_id <= row_id:
                        raise common.ProductionError(
                            f"worker row fragment order regressed rank={rank}"
                        )
                    current[rank] = next_record
            if output_buffer:
                common.write_all(output, output_buffer, label="dense_index.bin")
                output_buffer.clear()
            output.flush()
            os.fsync(output.fileno())
        for rank, handle in enumerate(handles):
            if current[rank] is not None or handle.read(1):
                raise common.ProductionError(
                    f"worker row fragment has tail after dense merge rank={rank}"
                )
    finally:
        for handle in handles:
            handle.close()
    expected_bytes = args.gpic_rows * common.RECORD_STRUCT.size
    binding = common.file_binding(path)
    if (
        written != expected_bytes
        or binding["bytes"] != expected_bytes
        or binding["sha256"] != digest.hexdigest()
        or (args.production_gate and expected_bytes != common.PRODUCTION_DENSE_INDEX_BYTES)
    ):
        raise common.ProductionError("dense index exact byte/SHA closure failed")
    pilot_core.fsync_dir(args.artifact_root)
    return binding


def validate_sidecars_against_dense(
    args: argparse.Namespace,
    *,
    catalog: pilot_core.CatalogBinding,
    worker_bindings: Sequence[dict[str, Any]],
    blobs: Sequence[dict[str, Any]],
) -> None:
    dense_fd = os.open(args.artifact_root / "dense_index.bin", os.O_RDONLY)
    dense = mmap.mmap(dense_fd, 0, access=mmap.ACCESS_READ)
    try:
        for rank, worker in enumerate(worker_bindings):
            worker_blobs = [
                value for value in blobs if value.get("worker_id") == rank
            ]
            by_id = {value["blob_id"]: value for value in worker_blobs}
            current_blob = -1
            current_end = 0
            current_segments = 0
            rows = 0
            sidecar = (
                args.fsx_workers_root / f"worker-{rank}" / "blob_order.bin"
            )
            with sidecar.open("rb", buffering=0) as handle:
                while True:
                    record = handle.read(common.RECORD_STRUCT.size)
                    if not record:
                        break
                    if len(record) != common.RECORD_STRUCT.size:
                        raise common.ProductionError(
                            f"blob-order short tail worker={rank}"
                        )
                    row_id, blob_id, offset, length, checksum = (
                        common.RECORD_STRUCT.unpack(record)
                    )
                    local_index = row_id - catalog.gpic_row_base
                    if (
                        local_index < 0
                        or local_index >= args.gpic_rows
                        or bytes(
                            dense[
                                local_index * common.RECORD_STRUCT.size :
                                (local_index + 1) * common.RECORD_STRUCT.size
                            ]
                        )
                        != record
                        or length <= 0
                        or not any(checksum)
                    ):
                        raise common.ProductionError(
                            f"sidecar/dense explicit record drift worker={rank} row={row_id}"
                        )
                    if blob_id != current_blob:
                        if current_blob != -1:
                            previous = by_id[current_blob]
                            if (
                                current_end != previous["bytes"]
                                or current_segments != previous["segments"]
                            ):
                                raise common.ProductionError(
                                    f"sidecar blob closure drift blob={current_blob}"
                                )
                        expected_sequence = len(
                            [
                                value
                                for value in worker_blobs
                                if value["blob_id"] < blob_id
                            ]
                        )
                        if (
                            blob_id
                            != common.global_blob_id(rank, expected_sequence, args.workers)
                            or blob_id not in by_id
                            or offset != 0
                        ):
                            raise common.ProductionError(
                                f"sidecar blob transition drift worker={rank} blob={blob_id}"
                            )
                        current_blob = blob_id
                        current_end = 0
                        current_segments = 0
                    if offset != current_end:
                        raise common.ProductionError(
                            f"sidecar gap/overlap worker={rank} blob={blob_id}"
                        )
                    current_end += length
                    current_segments += 1
                    rows += 1
            if current_blob == -1:
                raise common.ProductionError(f"empty sidecar worker={rank}")
            final = by_id[current_blob]
            if (
                current_end != final["bytes"]
                or current_segments != final["segments"]
                or rows != worker["rows"]
            ):
                raise common.ProductionError(
                    f"final sidecar closure drift worker={rank}"
                )
    finally:
        dense.close()
        os.close(dense_fd)


def run(args: argparse.Namespace) -> None:
    coordinator_code = common.code_snapshot(
        runner=Path(__file__),
        job_spec=args.job_spec,
        extra_files={
            "builder_runner": args.builder_runner,
            "builder_job_spec": args.builder_job_spec,
            "tar_source": Path(tar_source.__file__),
        },
    )
    common.require_regular_dir(args.artifact_root)
    common.require_regular_dir(args.coord_root)
    common.require_regular_dir(args.s3_root)
    if not args.s3_controls_root.exists():
        args.s3_controls_root.mkdir(mode=0o700)
    common.exact_names(
        args.s3_root,
        {common.S3_INIT_RELATIVE_PATH, "controls", "workers"},
        "pre-verification S3 root",
    )
    common.exact_names(args.s3_controls_root, set(), "pre-verification controls")
    common.exact_names(
        args.s3_workers_root,
        {f"worker-{rank}" for rank in range(args.workers)},
        "S3 workers root",
    )
    common.assert_no_root_success(args.s3_root)
    topology_evidence = common.load_topology_evidence(
        args.topology_summary,
        expected_jobtag=args.topology_jobtag,
        expected_raw_sha256=args.topology_summary_raw_sha256,
    )
    catalog = common.load_production_catalog(
        args.catalog_root, production_gate=args.production_gate
    )
    catalog_value = common.catalog_manifest(catalog)
    if catalog.gpic_rows != args.gpic_rows:
        raise common.ProductionError("coordinator catalog row count drift")
    builder_code = common.code_snapshot(
        runner=args.builder_runner,
        job_spec=args.builder_job_spec,
        extra_files={"tar_source": Path(tar_source.__file__)},
    )
    common.validate_code_snapshot(builder_code)
    limits = common.production_limits(
        workers=args.workers,
        gpic_rows=args.gpic_rows,
        blob_target_bytes=args.blob_target_bytes,
        max_local_bytes_per_worker=args.max_local_bytes_per_worker,
        max_source_archive_bytes_per_worker=(
            args.max_source_archive_bytes_per_worker
        ),
        max_s3_bytes_per_worker=args.max_s3_bytes_per_worker,
        max_s3_payload_bytes=args.max_s3_payload_bytes,
        max_s3_total_bytes=args.max_s3_total_bytes,
        s3_controls_reserve_bytes=args.s3_controls_reserve_bytes,
        max_fsx_incremental_bytes=args.max_fsx_incremental_bytes,
    )
    if args.production_gate:
        common.validate_production_limits(limits)
    partition_plan, partition_plan_raw = common.strict_json(
        args.coord_root / "partition_plan.json"
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
    build_init, build_init_raw = validate_build_initialization(
        args,
        expected_code=builder_code,
        catalog_manifest=catalog_value,
        topology_evidence=topology_evidence,
        limits=limits,
        partition_plan=partition_plan,
        partition_plan_raw=partition_plan_raw,
    )
    approval, approval_raw = validate_approval(
        args,
        build_init=build_init,
        build_init_raw=build_init_raw,
        builder_code=builder_code,
        catalog_manifest=catalog_value,
        partition_plan=partition_plan,
        partition_plan_raw=partition_plan_raw,
    )
    try:
        expected_statuses = [
            args.coord_root / f"uploaded.worker-{rank}.json"
            for rank in range(args.workers)
        ]
        builder.wait_for_paths(
            expected_statuses,
            coord_root=args.coord_root,
            timeout_seconds=args.wait_timeout_seconds,
            label="uploaded worker receipts",
        )
        worker_bindings, blobs, builder_boot_hashes = validate_workers(
            args,
            build_init=build_init,
            build_init_raw=build_init_raw,
            builder_code=builder_code,
            catalog=catalog,
            catalog_manifest=catalog_value,
            topology_evidence=topology_evidence,
            approval=approval,
            approval_raw=approval_raw,
            partition_plan=partition_plan,
            partition_plan_raw=partition_plan_raw,
        )
        blobs = sorted(blobs, key=lambda value: value["blob_id"])
        if [value["blob_id"] for value in blobs] != sorted(
            {value["blob_id"] for value in blobs}
        ):
            raise common.ProductionError(
                "global blob IDs are not strictly increasing and unique"
            )
        dense = merge_dense_index(args, catalog=catalog, blobs=blobs)
        validate_sidecars_against_dense(
            args,
            catalog=catalog,
            worker_bindings=worker_bindings,
            blobs=blobs,
        )
        common.validate_catalog_closure(catalog)
        common.validate_code_snapshot(builder_code)
        common.validate_code_snapshot(coordinator_code)
        fsx_before_manifest = common.fsx_tree_bytes(
            [args.artifact_root, args.coord_root]
        )
        if fsx_before_manifest > args.max_fsx_incremental_bytes:
            raise common.ProductionError("FSx cap exceeded before build manifest")
        keys = sorted(value["key"] for value in blobs)
        manifest = {
            "schema_version": common.BUILD_MANIFEST_SCHEMA,
            "generated_at": common.utc_now(),
            "status": "BUILD_COMPLETE_NOT_YET_PRODUCTION",
            "build_jobtag": args.build_jobtag,
            "artifact_root": str(args.artifact_root),
            "coord_root": str(args.coord_root),
            "s3_root": str(args.s3_root),
            "workers": worker_bindings,
            "worker_count": args.workers,
            "builder_boot_id_sha256": sorted(builder_boot_hashes),
            "rows": args.gpic_rows,
            "payload_bytes": sum(value["bytes"] for value in blobs),
            "blob_count": len(blobs),
            "blobs": blobs,
            "keys_sha256": hashlib.sha256(
                common.canonical_json_bytes(keys)
            ).hexdigest(),
            "dense_index": dense,
            "record_struct": common.RECORD_STRUCT.format,
            "record_bytes": common.RECORD_STRUCT.size,
            "partition_mode": args.partition_mode,
            "partition_plan_raw_sha256": hashlib.sha256(
                partition_plan_raw
            ).hexdigest(),
            "partition_plan_sha256": partition_plan[
                "partition_plan_sha256"
            ],
            "source_authority": build_init["source_authority"],
            "tar_workers": approval.get("tar_workers"),
            "full_tar_mapping": approval.get("full_tar_mapping"),
            "catalog": catalog_value,
            "topology_evidence": topology_evidence,
            "builder_code": builder_code,
            "coordinator_code": coordinator_code,
            "upload_approval_raw_sha256": hashlib.sha256(
                approval_raw
            ).hexdigest(),
            "upload_approval_sha256": approval["approval_sha256"],
            **common.build_init_binding(build_init, build_init_raw),
            "limits": limits,
            "fsx_bytes_before_manifest": fsx_before_manifest,
            "publication_semantics": (
                "build_complete_without_s3_root_success_full_fresh_verifier_required"
            ),
        }
        manifest["build_manifest_sha256"] = common.self_hash(
            manifest, "build_manifest_sha256"
        )
        manifest_raw = common.write_fsx_json_once(
            args.artifact_root / "build_manifest.json", manifest
        )
        complete = {
            "schema_version": common.BUILD_COMPLETE_SCHEMA,
            "generated_at": common.utc_now(),
            "status": "BUILD_COMPLETE_NOT_YET_PRODUCTION",
            "build_jobtag": args.build_jobtag,
            "build_manifest_raw_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "build_manifest_sha256": manifest["build_manifest_sha256"],
            "dense_index_sha256": dense["sha256"],
            "dense_index_bytes": dense["bytes"],
            "rows": args.gpic_rows,
            "payload_bytes": manifest["payload_bytes"],
            "blob_count": manifest["blob_count"],
            "coordinator_code": coordinator_code,
            **common.build_init_binding(build_init, build_init_raw),
        }
        complete["build_complete_sha256"] = common.self_hash(
            complete, "build_complete_sha256"
        )
        complete_raw = common.write_fsx_json_once(
            args.artifact_root / "build_complete.json", complete
        )
        common.write_fsx_bytes_once(
            args.artifact_root / "_BUILD_COMPLETE",
            (complete["build_complete_sha256"] + "\n").encode("ascii"),
        )
        pilot_core.fsync_dir(args.artifact_root)
        final_fsx_bytes = common.fsx_tree_bytes(
            [args.artifact_root, args.coord_root]
        )
        if final_fsx_bytes > args.max_fsx_incremental_bytes:
            raise pilot_core.CommittedSyncUnconfirmed(
                "build-complete marker visible but final FSx cap recheck failed"
            )
        common.assert_no_root_success(args.s3_root)
        print(
            "OFFLINE_GPIC_S3_PRODUCTION_BUILD_COMPLETE "
            f"build_jobtag={args.build_jobtag} rows={args.gpic_rows} "
            f"bytes={manifest['payload_bytes']} manifest_raw_sha256="
            f"{hashlib.sha256(manifest_raw).hexdigest()} "
            f"complete_raw_sha256={hashlib.sha256(complete_raw).hexdigest()}",
            flush=True,
        )
    except BaseException as exc:
        if isinstance(exc, pilot_core.CommittedSyncUnconfirmed) or common.success_or_unknown(
            args.artifact_root / "_BUILD_COMPLETE"
        ):
            print(
                "OFFLINE_GPIC_S3_PRODUCTION_BUILD_COMMITTED_SYNC_UNCONFIRMED",
                file=sys.stderr,
                flush=True,
            )
        else:
            try:
                common.write_failure(
                    args.coord_root,
                    build_jobtag=args.build_jobtag,
                    actor="coordinator",
                    actor_id=0,
                    exc=exc,
                )
            except BaseException:
                pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--coord-root", type=Path, required=True)
    parser.add_argument("--s3-root", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument(
        "--topology-summary", type=Path, default=common.PRODUCTION_TOPOLOGY_SUMMARY
    )
    parser.add_argument(
        "--topology-jobtag", default=common.PRODUCTION_TOPOLOGY_JOBTAG
    )
    parser.add_argument(
        "--topology-summary-raw-sha256",
        default=common.PRODUCTION_TOPOLOGY_RAW_SHA256,
    )
    parser.add_argument("--build-jobtag", required=True)
    parser.add_argument("--builder-runner", type=Path, required=True)
    parser.add_argument("--builder-job-spec", type=Path, required=True)
    parser.add_argument("--job-spec", type=Path, required=True)
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
        "--blob-target-bytes",
        type=int,
        default=common.PRODUCTION_BLOB_TARGET_BYTES,
    )
    parser.add_argument(
        "--max-local-bytes-per-worker",
        type=int,
        default=common.PRODUCTION_MAX_LOCAL_BYTES_PER_WORKER,
    )
    parser.add_argument(
        "--max-source-archive-bytes-per-worker",
        type=int,
        default=common.PRODUCTION_MAX_SOURCE_ARCHIVE_BYTES_PER_WORKER,
    )
    parser.add_argument(
        "--max-s3-bytes-per-worker",
        type=int,
        default=common.PRODUCTION_MAX_S3_BYTES_PER_WORKER,
    )
    parser.add_argument(
        "--max-s3-payload-bytes",
        type=int,
        default=common.PRODUCTION_MAX_S3_PAYLOAD_BYTES,
    )
    parser.add_argument(
        "--max-s3-total-bytes",
        type=int,
        default=common.PRODUCTION_MAX_S3_TOTAL_BYTES,
    )
    parser.add_argument(
        "--s3-controls-reserve-bytes",
        type=int,
        default=common.PRODUCTION_S3_CONTROLS_RESERVE_BYTES,
    )
    parser.add_argument(
        "--max-fsx-incremental-bytes",
        type=int,
        default=common.PRODUCTION_MAX_FSX_INCREMENTAL_BYTES,
    )
    parser.add_argument(
        "--max-byte-imbalance",
        type=float,
        default=common.PRODUCTION_BYTE_IMBALANCE_CAP,
    )
    parser.add_argument("--wait-timeout-seconds", type=float, default=21_600.0)
    parser.add_argument("--production-gate", action="store_true")
    args = parser.parse_args(argv)
    for label in (
        "artifact_root",
        "coord_root",
        "s3_root",
        "catalog_root",
        "topology_summary",
        "builder_runner",
        "builder_job_spec",
        "job_spec",
    ):
        setattr(args, label, getattr(args, label).expanduser().resolve(strict=True))
    if (
        not args.build_jobtag
        or Path(args.build_jobtag).name != args.build_jobtag
        or args.build_jobtag in {".", "..", "latest"}
        or args.artifact_root.name != args.build_jobtag
        or args.coord_root.name != args.build_jobtag
        or args.s3_root.name != args.build_jobtag
    ):
        raise common.ProductionError("coordinator exact build jobtag/root drift")
    if not str(args.artifact_root).startswith("/fsx/") or not str(
        args.coord_root
    ).startswith("/fsx/"):
        raise common.ProductionError("coordinator FSx root escaped /fsx")
    if (
        args.wait_timeout_seconds <= 0
        or args.max_source_archive_bytes_per_worker < 1
    ):
        raise common.ProductionError("coordinator wait timeout must be positive")
    if args.production_gate and (
        args.workers != common.PRODUCTION_WORKERS
        or args.gpic_rows != common.PRODUCTION_GPIC_ROWS
        or args.source_prefix != common.PRODUCTION_SOURCE_PREFIX
        or args.blob_target_bytes != common.PRODUCTION_BLOB_TARGET_BYTES
        or args.max_local_bytes_per_worker
        != common.PRODUCTION_MAX_LOCAL_BYTES_PER_WORKER
        or args.max_source_archive_bytes_per_worker
        != common.PRODUCTION_MAX_SOURCE_ARCHIVE_BYTES_PER_WORKER
        or args.max_s3_bytes_per_worker
        != common.PRODUCTION_MAX_S3_BYTES_PER_WORKER
        or args.max_s3_payload_bytes != common.PRODUCTION_MAX_S3_PAYLOAD_BYTES
        or args.max_s3_total_bytes != common.PRODUCTION_MAX_S3_TOTAL_BYTES
        or args.s3_controls_reserve_bytes
        != common.PRODUCTION_S3_CONTROLS_RESERVE_BYTES
        or args.max_fsx_incremental_bytes
        != common.PRODUCTION_MAX_FSX_INCREMENTAL_BYTES
        or args.max_byte_imbalance != common.PRODUCTION_BYTE_IMBALANCE_CAP
        or args.topology_summary != common.PRODUCTION_TOPOLOGY_SUMMARY
        or args.topology_jobtag != common.PRODUCTION_TOPOLOGY_JOBTAG
        or args.topology_summary_raw_sha256
        != common.PRODUCTION_TOPOLOGY_RAW_SHA256
        or not str(args.s3_root).startswith(
            "/s3/evan/artifacts/offline_gpic_blob_v1/"
        )
    ):
        raise common.ProductionError("frozen production coordinator constants drift")
    args.fsx_workers_root = args.artifact_root / "workers"
    args.s3_controls_root = args.s3_root / "controls"
    args.s3_workers_root = args.s3_root / "workers"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
