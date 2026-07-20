#!/usr/bin/env python3
"""No-network synthetic and negative gate for GPIC S3 production."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from scripts.offline_arb import build_offline_gpic_s3_production as builder
from scripts.offline_arb import offline_gpic_s3_production_common as common
from scripts.offline_arb import run_offline_gpic_blob_pilot as pilot_core
from scripts.offline_arb import verify_offline_gpic_s3_production as verifier


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_failure(label: str, function: Any, *args: Any, **kwargs: Any) -> None:
    try:
        function(*args, **kwargs)
    except Exception:
        return
    raise AssertionError(f"negative unexpectedly passed: {label}")


def create_catalog(root: Path, *, rows: int = 512, shards: int = 32) -> tuple[Path, str]:
    source = root / "source"
    source.mkdir(parents=True)
    row_base = 73
    paths: list[str] = []
    for local_index in range(rows):
        shard_id = local_index % shards
        shard = source / f"{shard_id:05d}"
        shard.mkdir(exist_ok=True)
        extension = ".jpg" if local_index % 2 == 0 else ".png"
        path = shard / f"{local_index:064x}{extension}"
        payload = hashlib.sha256(f"prod-static-{local_index}".encode()).digest() * 32
        path.write_bytes(payload)
        paths.append(str(path))
    schema = pa.schema(
        [
            pa.field("row_id", pa.int64(), nullable=False),
            pa.field("image_path", pa.string(), nullable=False),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array(range(row_base, row_base + rows), type=pa.int64()),
            pa.array(paths, type=pa.string()),
        ],
        schema=schema,
    )
    catalog = root / "catalog"
    catalog.mkdir()
    parquet = catalog / "gpic.parquet"
    pq.write_table(table, parquet, row_group_size=37, compression="zstd")
    raw = parquet.read_bytes()
    header = {
        "schema_version": pilot_core.CATALOG_SCHEMA_VERSION,
        "domain_order": ["prelude", "gpic"],
        "row_count": row_base + rows,
        "catalog_hash": hashlib.sha256(b"production-static-catalog").hexdigest(),
        "files": {
            "prelude": {"rows": row_base},
            "gpic": {
                "name": parquet.name,
                "rows": rows,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "schema": str(schema),
            },
        },
    }
    (catalog / "header.json").write_bytes(common.canonical_json_bytes(header))
    return catalog, str(source) + "/"


def create_tar_source_authorities(
    root: Path, *, catalog: Path, source_prefix: str
) -> tuple[Path, Path, str, Path, str, Path, str]:
    binding = common.load_production_catalog(catalog, production_gate=False)
    tar_root = root / "tar-source"
    tar_root.mkdir()
    source_root = Path(source_prefix)
    shard_ids: list[int] = []
    for shard_dir in sorted(source_root.iterdir(), key=lambda path: path.name):
        require(shard_dir.is_dir(), "synthetic source shard is not a directory")
        shard_id = int(shard_dir.name)
        shard_ids.append(shard_id)
        tar_path = tar_root / f"gpic_train_{shard_id:05d}.tar"
        with tarfile.open(
            tar_path, mode="w", format=tarfile.USTAR_FORMAT
        ) as archive:
            for image_path in sorted(
                shard_dir.iterdir(), key=lambda path: path.name
            ):
                for name, payload in (
                    (image_path.stem + ".json", b"{}"),
                    (image_path.name, image_path.read_bytes()),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mtime = 0
                    info.mode = 0o600
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(payload))
    inventory = {
        "schema_version": "offline_gpic_tar_inventory_v1",
        "tar_root": str(tar_root),
        "expected_tar_count": len(shard_ids),
        "observed_tar_count": len(shard_ids),
        "exact_tar_name_set": True,
        "missing_tar_names": [],
        "unexpected_tar_names": [],
        "duplicate_tar_name_count": 0,
        "non_tar_entries": [],
    }
    inventory["inventory_sha256"] = common.self_hash(
        inventory, "inventory_sha256"
    )
    sample = {
        "schema_version": "offline_gpic_tar_catalog_sample_v1",
        "catalog_total_rows": binding.gpic_rows,
        "catalog_rows_scanned": binding.gpic_rows,
        "catalog_total_row_groups": 1,
        "catalog_row_groups_scanned": 1,
        "samples_per_tar": 1,
        "sample_rows": len(shard_ids),
        "sample_image_paths_unique": len(shard_ids),
    }
    sample["sample_sha256"] = common.self_hash(sample, "sample_sha256")
    tar_results: list[dict[str, Any]] = []
    for shard_id in shard_ids:
        value = {
            "tar_shard": shard_id,
            "pairing_valid": True,
            "exact_payload_equivalence": True,
            "missing_member_basenames": [],
        }
        value["tar_evidence_sha256"] = common.self_hash(
            value, "tar_evidence_sha256"
        )
        tar_results.append(value)
    report = {
        "schema_version": "offline_gpic_tar_mapping_report_v1",
        "status": "PASS",
        "read_only_probe": True,
        "jobtag": "synthetic-tar-mapping-authority",
        "tar_root": str(tar_root),
        "source_prefix": source_prefix,
        "catalog_hash": binding.catalog_hash,
        "gpic_rows": binding.gpic_rows,
        "gpic_row_base": binding.gpic_row_base,
        "gpic_sha256": binding.gpic_sha256,
        "tar_shards": shard_ids,
        "tar_member_name_prefixes": [""],
        "exact_catalog_tar_payload_equivalence": True,
        "all_sampled_tars_pairing_valid": True,
        "all_catalog_samples_found_in_derived_tar": True,
        "sample_comparisons": len(shard_ids),
        "byte_equal_comparisons": len(shard_ids),
        "sha256_equal_comparisons": len(shard_ids),
        "inventory": inventory,
        "sample": sample,
        "tar_results": tar_results,
        "writes": {"s3_writes": 0},
    }
    report["report_sha256"] = common.self_hash(report, "report_sha256")
    mapping_root = root / "mapping-authority"
    mapping_root.mkdir()
    report_path = mapping_root / "summary.json"
    report_raw = common.canonical_json_bytes(report)
    report_path.write_bytes(report_raw)
    (mapping_root / "_COMPLETE").write_bytes(
        (hashlib.sha256(report_raw).hexdigest() + "\n").encode("ascii")
    )
    quick = {
        "schema_version": "offline_gpic_tar_seq_quick_summary_v1",
        "jobtag": common.PRODUCTION_TAR_QUICK_JOBTAG,
        "distinct_boots": 8,
        "sha_aggregate_bytes_per_second": 5_000_000_000,
        "sha_slowest_bytes_per_second": 600_000_000,
        "list_aggregate_bytes_per_second": 5_000_000_000,
        "list_slowest_bytes_per_second": 600_000_000,
        "rows": [{"worker_id": rank} for rank in range(8)],
    }
    quick_path = root / "synthetic_tar_quick.json"
    quick_raw = common.canonical_json_bytes(quick)
    quick_path.write_bytes(quick_raw)
    parser_root = root / "parser-authority"
    parser_root.mkdir()
    parser_summary = {
        "schema_version": "offline_gpic_tar_parser_gate_summary_v1",
        "status": "PASS",
        "recommendation": "PASS_FOR_BOUNDED_PRODUCTION_INTEGRATION",
        "recommended_tar_workers": 4,
        "jobtag": "synthetic-tar-parser-gate",
        "workers": 8,
        "distinct_physical_boots": 8,
        "tars_per_worker": 8,
        "unique_tar_shards": 64,
        "shards_disjoint": True,
        "semantic_digest_equal_across_1_4_8_per_worker": True,
        "aggregate": [
            {
                "tar_workers": workers,
                "projected_8node_archive_bytes_per_second": 2_000_000_000,
                "estimated_full_archive_scan_hours": 1.0,
                "max_peak_rss_bytes": 1_000_000_000,
            }
            for workers in (1, 4, 8)
        ],
        "thresholds": {
            "min_projected_aggregate_bytes_per_second": 1_000_000_000,
            "max_projected_scan_hours": 4.0,
            "max_peak_rss_bytes": 32 * (1 << 30),
        },
        "code": {"files": {}},
    }
    parser_summary["summary_sha256"] = common.self_hash(
        parser_summary, "summary_sha256"
    )
    parser_path = parser_root / "summary.json"
    parser_raw = common.canonical_json_bytes(parser_summary)
    parser_path.write_bytes(parser_raw)
    (parser_root / "_COMPLETE").write_bytes(
        (hashlib.sha256(parser_raw).hexdigest() + "\n").encode("ascii")
    )
    return (
        tar_root,
        report_path,
        hashlib.sha256(report_raw).hexdigest(),
        quick_path,
        hashlib.sha256(quick_raw).hexdigest(),
        parser_path,
        hashlib.sha256(parser_raw).hexdigest(),
    )


def run_parallel(commands: Sequence[Sequence[str]], *, success: bool) -> list[str]:
    processes = [
        subprocess.Popen(
            list(command),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for command in commands
    ]
    outputs: list[str] = []
    codes: list[int] = []
    try:
        for process in processes:
            output, _ = process.communicate(timeout=300)
            outputs.append(output)
            codes.append(process.returncode)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
    if success:
        require(codes == [0] * len(commands), f"parallel task failed codes={codes}\n{outputs}")
    else:
        require(any(code != 0 for code in codes), "negative parallel task passed")
    return outputs


def parser_gate_negatives(path: Path, raw_sha256: str) -> None:
    expect_failure(
        "parser gate wrong raw SHA",
        builder.load_tar_parser_gate_authority,
        path,
        expected_raw_sha256="0" * 64,
        requested_tar_workers=4,
        production_gate=False,
    )
    original, _ = common.strict_json(path)
    for label, field, replacement in (
        ("recommended workers", "recommended_tar_workers", 8),
        ("physical workers", "workers", 7),
    ):
        root = path.parent.parent / f"parser-negative-{field}"
        root.mkdir()
        value = dict(original)
        value[field] = replacement
        value["summary_sha256"] = common.self_hash(value, "summary_sha256")
        candidate = root / "summary.json"
        raw = common.canonical_json_bytes(value)
        candidate.write_bytes(raw)
        candidate_sha = hashlib.sha256(raw).hexdigest()
        (root / "_COMPLETE").write_bytes(
            (candidate_sha + "\n").encode("ascii")
        )
        expect_failure(
            f"parser gate {label}",
            builder.load_tar_parser_gate_authority,
            candidate,
            expected_raw_sha256=candidate_sha,
            requested_tar_workers=4,
            production_gate=False,
        )


def builder_command(
    *,
    runner: Path,
    job_spec: Path,
    catalog: Path,
    source_prefix: str,
    root: Path,
    jobtag: str,
    worker_id: int,
    source_mode: str = "small_files",
    tar_root: Path | None = None,
    tar_mapping: Path | None = None,
    tar_mapping_raw_sha256: str | None = None,
    tar_quick: Path | None = None,
    tar_quick_raw_sha256: str | None = None,
    tar_parser_gate: Path | None = None,
    tar_parser_gate_raw_sha256: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(runner),
        "--catalog-root",
        str(catalog),
        "--artifact-prefix",
        str(root / "artifacts"),
        "--coord-prefix",
        str(root / "coord"),
        "--local-prefix",
        str(root / "local"),
        "--s3-prefix",
        str(root / "s3"),
        "--build-jobtag",
        jobtag,
        "--worker-id",
        str(worker_id),
        "--workers",
        "8",
        "--partition-mode",
        "shard_balanced_rows",
        "--source-mode",
        source_mode,
        "--source-prefix",
        source_prefix,
        "--source-threads",
        "2",
        "--source-attempts",
        "2",
        "--source-block-bytes",
        "512",
        "--retry-base-seconds",
        "0.001",
        "--blob-target-bytes",
        "16384",
        "--max-local-bytes-per-worker",
        "50000000",
        "--max-source-archive-bytes-per-worker",
        "50000000",
        "--max-s3-bytes-per-worker",
        "3000000",
        "--max-s3-payload-bytes",
        "19000000",
        "--max-s3-total-bytes",
        "20000000",
        "--s3-controls-reserve-bytes",
        "1000000",
        "--max-fsx-incremental-bytes",
        "50000000",
        "--max-byte-imbalance",
        "1.10",
        "--s3-block-bytes",
        "4096",
        "--barrier-timeout-seconds",
        "120",
        "--write-authority",
        common.WRITE_AUTHORITY,
        "--job-spec",
        str(job_spec),
        "--cleanup-local-after-receipt",
    ]
    if source_mode == "tar_sequential":
        require(
            tar_root is not None
            and tar_mapping is not None
            and tar_mapping_raw_sha256 is not None
            and tar_quick is not None
            and tar_quick_raw_sha256 is not None
            and tar_parser_gate is not None
            and tar_parser_gate_raw_sha256 is not None,
            "synthetic tar builder authority is incomplete",
        )
        for flag, value in (
            ("--source-threads", "1"),
            ("--source-attempts", "1"),
            ("--source-block-bytes", "1048576"),
            ("--retry-base-seconds", "0"),
            ("--max-source-archive-bytes-per-worker", "60000000"),
        ):
            command[command.index(flag) + 1] = value
        command.extend(
            [
                "--tar-workers",
                "4",
                "--tar-root",
                str(tar_root),
                "--tar-mapping-manifest",
                str(tar_mapping),
                "--tar-mapping-raw-sha256",
                tar_mapping_raw_sha256,
                "--tar-quick-summary",
                str(tar_quick),
                "--tar-quick-raw-sha256",
                tar_quick_raw_sha256,
                "--tar-parser-gate-summary",
                str(tar_parser_gate),
                "--tar-parser-gate-raw-sha256",
                tar_parser_gate_raw_sha256,
            ]
        )
    return command


def coordinator_command(
    *,
    runner: Path,
    builder_runner: Path,
    builder_job_spec: Path,
    job_spec: Path,
    catalog: Path,
    source_prefix: str,
    root: Path,
    jobtag: str,
    source_mode: str = "small_files",
) -> list[str]:
    return [
        sys.executable,
        str(runner),
        "--artifact-root",
        str(root / "artifacts" / jobtag),
        "--coord-root",
        str(root / "coord" / jobtag),
        "--s3-root",
        str(root / "s3" / jobtag),
        "--catalog-root",
        str(catalog),
        "--build-jobtag",
        jobtag,
        "--builder-runner",
        str(builder_runner),
        "--builder-job-spec",
        str(builder_job_spec),
        "--job-spec",
        str(job_spec),
        "--workers",
        "8",
        "--gpic-rows",
        "512",
        "--partition-mode",
        "shard_balanced_rows",
        "--source-mode",
        source_mode,
        "--source-prefix",
        source_prefix,
        "--blob-target-bytes",
        "16384",
        "--max-local-bytes-per-worker",
        "50000000",
        "--max-source-archive-bytes-per-worker",
        "60000000" if source_mode == "tar_sequential" else "50000000",
        "--max-s3-bytes-per-worker",
        "3000000",
        "--max-s3-payload-bytes",
        "19000000",
        "--max-s3-total-bytes",
        "20000000",
        "--s3-controls-reserve-bytes",
        "1000000",
        "--max-fsx-incremental-bytes",
        "50000000",
        "--max-byte-imbalance",
        "1.10",
        "--wait-timeout-seconds",
        "120",
    ]


def verifier_command(
    *,
    runner: Path,
    builder_runner: Path,
    coordinator_runner: Path,
    builder_job_spec: Path,
    coordinator_job_spec: Path,
    job_spec: Path,
    catalog: Path,
    source_prefix: str,
    root: Path,
    build_jobtag: str,
    verifier_jobtag: str,
    worker_id: int,
    source_mode: str = "small_files",
) -> list[str]:
    return [
        sys.executable,
        str(runner),
        "--build-artifact-root",
        str(root / "artifacts" / build_jobtag),
        "--build-coord-root",
        str(root / "coord" / build_jobtag),
        "--verifier-artifact-prefix",
        str(root / "verify-artifacts"),
        "--verifier-coord-prefix",
        str(root / "verify-coord"),
        "--s3-root",
        str(root / "s3" / build_jobtag),
        "--catalog-root",
        str(catalog),
        "--build-jobtag",
        build_jobtag,
        "--verifier-jobtag",
        verifier_jobtag,
        "--worker-id",
        str(worker_id),
        "--workers",
        "8",
        "--gpic-rows",
        "512",
        "--partition-mode",
        "shard_balanced_rows",
        "--source-mode",
        source_mode,
        "--source-prefix",
        source_prefix,
        "--max-s3-total-bytes",
        "20000000",
        "--max-source-archive-bytes-per-worker",
        "60000000" if source_mode == "tar_sequential" else "50000000",
        "--max-fsx-incremental-bytes",
        "50000000",
        "--builder-runner",
        str(builder_runner),
        "--builder-job-spec",
        str(builder_job_spec),
        "--coordinator-runner",
        str(coordinator_runner),
        "--coordinator-job-spec",
        str(coordinator_job_spec),
        "--job-spec",
        str(job_spec),
        "--barrier-timeout-seconds",
        "120",
        "--verify-authority",
        common.VERIFY_AUTHORITY,
    ]


def prepare_roots(root: Path) -> None:
    for name in (
        "artifacts",
        "coord",
        "local",
        "s3",
        "verify-artifacts",
        "verify-coord",
    ):
        (root / name).mkdir(parents=True)


def validate_yaml(paths: Sequence[Path]) -> None:
    values = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    for path, value in zip(paths, values):
        require(value.get("gpuPerWorker") == 0, f"GPU drift: {path.name}")
        require(value.get("cpuPerWorker") == 16, f"CPU16 drift: {path.name}")
        require(value.get("maxRetries") == 0, f"retry drift: {path.name}")
        command = value.get("command")
        require(isinstance(command, str), f"missing command: {path.name}")
        subprocess.run(["bash", "-n"], input=command, text=True, check=True)
        for forbidden in ("set -x", "set -eux", "pip install", "latest --"):
            require(forbidden not in command, f"forbidden token {forbidden}: {path.name}")
    require(values[0].get("numWorkers") == 8, "builder must have eight workers")
    require(values[2].get("numWorkers") == 8, "verifier must have eight workers")
    build_command = values[0]["command"]
    for token in (
        "PARTITION_MODE=shard_modulo",
        "SOURCE_MODE=tar_sequential",
        "TAR_WORKERS=8",
        "TAR_SOURCE=",
        "TAR_MAPPING=",
        "TAR_QUICK=",
        "TAR_PARSER_GATE=",
        "--tar-parser-gate-summary",
        "--tar-parser-gate-raw-sha256 a8f8b02d",
        "EXPECTED_RUNNER_SHA=6f015558",
        "EXPECTED_COMMON_SHA=1f31ee29",
        "EXPECTED_TAR_SOURCE_SHA=5a32fbec",
        "EXPECTED_PILOT_SHA=3a391784",
        "EXPECTED_SCALE_SHA=11459bfd",
        "sha256sum \"$RUNNER\"",
        "sha256sum \"$COMMON\"",
        "sha256sum \"$TAR_SOURCE\"",
        "sha256sum \"$PILOT\"",
        "sha256sum \"$SCALE\"",
        "--source-attempts 1",
        "--retry-base-seconds 0",
        "--max-local-bytes-per-worker 1000000000000",
        "--max-source-archive-bytes-per-worker 1800000000000",
        "--max-s3-payload-bytes 3199000000000",
        "--max-s3-total-bytes 3200000000000",
        "--max-fsx-incremental-bytes 50000000000",
        "--production-gate",
    ):
        require(token in build_command, f"production builder token missing: {token}")


def pure_negatives(root: Path) -> None:
    telemetry_line = builder.canonical_json_line({"record": 1})
    require(
        telemetry_line == b'{"record":1}\n'
        and len(telemetry_line.splitlines()) == 1,
        "tar telemetry must encode exactly one canonical record per line",
    )
    cross_host_file = root / "cross-host-source-identity.bin"
    cross_host_file.write_bytes(b"portable-fuse-identity")
    metadata = common.require_regular_file(cross_host_file)
    observed = pilot_core.FileIdentity.from_stat(metadata)
    portable = {
        "path": str(cross_host_file),
        "bytes": observed.size,
        "identity": {
            "dev": observed.dev + 101,
            "ino": observed.ino + 103,
            "size": observed.size,
            "mtime_ns": observed.mtime_ns,
            "ctime_ns": observed.ctime_ns,
        },
    }
    require(
        builder.validate_cross_host_source_archive_binding(
            portable, cross_host_file, metadata
        )
        == observed.size,
        "cross-host dev/ino differences must be portable",
    )
    for field in ("size", "mtime_ns", "ctime_ns"):
        drifted = {**portable, "identity": dict(portable["identity"])}
        drifted["identity"][field] += 1
        expect_failure(
            f"cross-host stable identity {field}",
            builder.validate_cross_host_source_archive_binding,
            drifted,
            cross_host_file,
            metadata,
        )
    drifted_bytes = {**portable, "bytes": portable["bytes"] + 1}
    expect_failure(
        "cross-host source archive bytes",
        builder.validate_cross_host_source_archive_binding,
        drifted_bytes,
        cross_host_file,
        metadata,
    )
    require(common.RECORD_STRUCT.format == "<qIQI32s", "record ABI drift")
    require(common.RECORD_STRUCT.size == 56, "record size drift")
    require(
        common.PRODUCTION_GPIC_ROWS * common.RECORD_STRUCT.size
        == common.PRODUCTION_DENSE_INDEX_BYTES,
        "dense-index production byte closure drift",
    )
    release = {
        common.RELEASE_STORAGE_S3_ROOT_FIELD: "/s3/root",
        common.RELEASE_STORAGE_INDEX_PATH_FIELD: "/fsx/index.bin",
        common.RELEASE_STORAGE_S3_INIT_FIELD: {
            common.RELEASE_STORAGE_S3_INIT_PATH_FIELD: "_INIT.json",
            common.RELEASE_STORAGE_S3_INIT_BYTES_FIELD: 1,
            common.RELEASE_STORAGE_S3_INIT_RAW_SHA256_FIELD: "0" * 64,
        },
    }
    require(
        set(release) == {"s3_root", "index_path", "s3_init"}
        and set(release["s3_init"]) == {"path", "bytes", "raw_sha256"},
        "release.storage exact schema drift",
    )
    upload_root = root / "upload-negatives"
    upload_root.mkdir()
    source = upload_root / "source.bin"
    source.write_bytes(b"immutable" * 128)
    destination = upload_root / "existing.bin"
    destination.write_bytes(b"old")
    expect_failure(
        "S3 upload no-clobber",
        common.upload_file_once,
        source,
        destination,
        expected_bytes=source.stat().st_size,
        expected_sha256=common.sha256_file(source),
        block_bytes=64,
    )
    require(destination.read_bytes() == b"old", "no-clobber modified destination")
    unknown = upload_root / "unknown.bin"
    real_write_all = common.write_all

    def write_then_fail(handle: Any, payload: bytes, *, label: str) -> None:
        real_write_all(handle, payload, label=label)
        raise OSError("synthetic post-write failure")

    try:
        common.write_all = write_then_fail
        expect_failure(
            "visible upload error is committed/unknown",
            common.upload_file_once,
            source,
            unknown,
            expected_bytes=source.stat().st_size,
            expected_sha256=common.sha256_file(source),
            block_bytes=source.stat().st_size,
        )
    finally:
        common.write_all = real_write_all
    require(unknown.exists(), "unknown-state negative did not expose final key")


def execute_pipeline(
    *,
    root: Path,
    catalog: Path,
    source_prefix: str,
    build_jobtag: str,
    verifier_jobtag: str,
    builder_runner: Path,
    coordinator_runner: Path,
    verifier_runner: Path,
    builder_job: Path,
    coordinator_job: Path,
    verifier_job: Path,
    corrupt: bool,
    source_mode: str = "small_files",
    tar_root: Path | None = None,
    tar_mapping: Path | None = None,
    tar_mapping_raw_sha256: str | None = None,
    tar_quick: Path | None = None,
    tar_quick_raw_sha256: str | None = None,
    tar_parser_gate: Path | None = None,
    tar_parser_gate_raw_sha256: str | None = None,
) -> None:
    prepare_roots(root)
    # The submitter owns only the source-build parent; verifier rank 0 owns the
    # fresh verifier-jobtag leaf below it.
    (root / "verify-artifacts" / build_jobtag).mkdir(mode=0o700)
    (root / "verify-coord" / build_jobtag).mkdir(mode=0o700)
    build_commands = [
        builder_command(
            runner=builder_runner,
            job_spec=builder_job,
            catalog=catalog,
            source_prefix=source_prefix,
            root=root,
            jobtag=build_jobtag,
            worker_id=rank,
            source_mode=source_mode,
            tar_root=tar_root,
            tar_mapping=tar_mapping,
            tar_mapping_raw_sha256=tar_mapping_raw_sha256,
            tar_quick=tar_quick,
            tar_quick_raw_sha256=tar_quick_raw_sha256,
            tar_parser_gate=tar_parser_gate,
            tar_parser_gate_raw_sha256=tar_parser_gate_raw_sha256,
        )
        for rank in range(8)
    ]
    run_parallel(build_commands, success=True)
    s3_root = root / "s3" / build_jobtag
    require(not (s3_root / "_SUCCESS").exists(), "builder published root success")
    completed = subprocess.run(
        coordinator_command(
            runner=coordinator_runner,
            builder_runner=builder_runner,
            builder_job_spec=builder_job,
            job_spec=coordinator_job,
            catalog=catalog,
            source_prefix=source_prefix,
            root=root,
            jobtag=build_jobtag,
            source_mode=source_mode,
        ),
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    require(completed.returncode == 0, f"coordinator failed\n{completed.stdout}\n{completed.stderr}")
    require(not (s3_root / "_SUCCESS").exists(), "coordinator published root success")
    if source_mode == "tar_sequential":
        coord_root = root / "coord" / build_jobtag
        full_path = coord_root / "tar_mapping_authority" / "full_mapping.json"
        full_mapping, full_raw = common.strict_json(full_path)
        full_marker = (
            coord_root / "tar_mapping_authority" / "_COMPLETE"
        ).read_bytes()
        require(
            full_marker
            == (hashlib.sha256(full_raw).hexdigest() + "\n").encode("ascii")
            and full_mapping["payload_reads_before_authority"] == 0,
            "tar full authority was not success-last before payload reads",
        )
        build_init, _ = common.strict_json(s3_root / "_INIT.json")
        require(
            build_init["limits"]["max_local_bytes_per_worker"] == 50_000_000
            and build_init["limits"][
                "max_source_archive_bytes_per_worker"
            ]
            == 60_000_000,
            "source archive-read and local-footprint caps are not independent",
        )
        approval, _ = common.strict_json(coord_root / "UPLOAD_APPROVED.json")
        require(
            approval["full_tar_mapping"]["raw_sha256"]
            == hashlib.sha256(full_raw).hexdigest(),
            "upload approval does not bind full tar authority",
        )
        for rank in range(8):
            worker_root = root / "artifacts" / build_jobtag / "workers" / f"worker-{rank}"
            receipt, _ = common.strict_json(worker_root / "receipt.json")
            preupload, _ = common.strict_json(worker_root / "preupload.json")
            local_manifest, _ = common.strict_json(
                worker_root / "local_manifest.json"
            )
            worker_mapping, _ = common.strict_json(
                coord_root / "tar_mapping_workers" / f"worker-{rank}.json"
            )
            telemetry_raw = (worker_root / "tar_telemetry.jsonl").read_bytes()
            telemetry_lines = telemetry_raw.splitlines(keepends=True)
            require(
                preupload["source_thread_authority"]
                == "not_applicable_tar_sequential"
                and receipt["source"]["attempts"] == 1
                and receipt["source"]["tar_workers"] == 4
                and receipt["source"]["max_selected_bytes_per_tar"]
                == 3_000_000
                and receipt["tar_mapping_evidence"]["full"]
                == approval["full_tar_mapping"]
                and local_manifest["tar_telemetry"]["archive_bytes"]
                == worker_mapping["source_archive_bytes"]
                and local_manifest["tar_telemetry"][
                    "max_selected_bytes_per_tar"
                ]
                == 3_000_000
                and len(telemetry_lines)
                == local_manifest["tar_telemetry"]["records"],
                f"tar builder glue closure drift worker={rank}",
            )
            for line in telemetry_lines:
                value = json.loads(line)
                require(
                    builder.canonical_json_line(value) == line,
                    f"tar telemetry is not one canonical record/line worker={rank}",
                )
    if corrupt:
        build_manifest = json.loads(
            (root / "artifacts" / build_jobtag / "build_manifest.json").read_bytes()
        )
        first = Path(build_manifest["blobs"][0]["key"])
        raw = bytearray(first.read_bytes())
        raw[0] ^= 0xFF
        first.write_bytes(raw)
    verify_commands = [
        verifier_command(
            runner=verifier_runner,
            builder_runner=builder_runner,
            coordinator_runner=coordinator_runner,
            builder_job_spec=builder_job,
            coordinator_job_spec=coordinator_job,
            job_spec=verifier_job,
            catalog=catalog,
            source_prefix=source_prefix,
            root=root,
            build_jobtag=build_jobtag,
            verifier_jobtag=verifier_jobtag,
            worker_id=rank,
            source_mode=source_mode,
        )
        for rank in range(8)
    ]
    run_parallel(verify_commands, success=not corrupt)
    if corrupt:
        require(not (s3_root / "_SUCCESS").exists(), "corrupt payload published success")
        return
    marker = (s3_root / "_SUCCESS").read_text(encoding="ascii").strip()
    common.exact_names(
        s3_root,
        {"_INIT.json", "controls", "workers", "_SUCCESS"},
        "published static root",
    )
    common.exact_names(
        s3_root / "controls", {"manifest.json"}, "published controls"
    )
    canonical, canonical_raw = common.strict_json(s3_root / "controls" / "manifest.json")
    common.validate_self_hash(canonical, "manifest_sha256", "canonical manifest")
    require(marker == canonical["manifest_sha256"], "root marker/manifest drift")
    storage = canonical[common.RELEASE_FIELD][common.RELEASE_STORAGE_FIELD]
    require(
        set(storage) == {"s3_root", "index_path", "s3_init"}
        and set(storage["s3_init"]) == {"path", "bytes", "raw_sha256"}
        and storage["s3_init"]["path"] == "_INIT.json",
        "published release.storage exact schema drift",
    )
    fsx_manifest = (
        root
        / "verify-artifacts"
        / build_jobtag
        / verifier_jobtag
        / "canonical_manifest.json"
    ).read_bytes()
    require(fsx_manifest == canonical_raw, "FSx/S3 canonical manifest byte drift")
    fsx_release_root = (
        root / "verify-artifacts" / build_jobtag / verifier_jobtag
    )
    require(
        (fsx_release_root / "_SUCCESS").read_text(encoding="ascii").strip()
        == canonical["manifest_sha256"],
        "FSx release marker/manifest drift",
    )
    blob_ids = [value["blob_id"] for value in canonical["blobs"]]
    require(blob_ids == sorted(set(blob_ids)), "canonical blob IDs not strict sorted")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--builder-job-spec", type=Path, required=True)
    parser.add_argument("--coordinator-job-spec", type=Path, required=True)
    parser.add_argument("--verifier-job-spec", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().absolute()
    root.mkdir(parents=True)
    builder_runner = Path(builder.__file__).resolve(strict=True)
    coordinator_runner = Path(
        __import__(
            "scripts.offline_arb.coordinate_offline_gpic_s3_production",
            fromlist=["dummy"],
        ).__file__
    ).resolve(strict=True)
    verifier_runner = Path(verifier.__file__).resolve(strict=True)
    validate_yaml(
        [
            args.builder_job_spec,
            args.coordinator_job_spec,
            args.verifier_job_spec,
            args.job_spec,
        ]
    )
    pure_negatives(root)
    inputs = root / "inputs"
    inputs.mkdir()
    catalog, source_prefix = create_catalog(inputs)
    (
        tar_root,
        tar_mapping,
        tar_mapping_raw_sha256,
        tar_quick,
        tar_quick_raw_sha256,
        tar_parser_gate,
        tar_parser_gate_raw_sha256,
    ) = create_tar_source_authorities(
        inputs, catalog=catalog, source_prefix=source_prefix
    )
    parser_gate_negatives(tar_parser_gate, tar_parser_gate_raw_sha256)
    execute_pipeline(
        root=root / "good",
        catalog=catalog,
        source_prefix=source_prefix,
        build_jobtag="synthetic-production-build-good",
        verifier_jobtag="synthetic-production-verify-good",
        builder_runner=builder_runner,
        coordinator_runner=coordinator_runner,
        verifier_runner=verifier_runner,
        builder_job=args.builder_job_spec,
        coordinator_job=args.coordinator_job_spec,
        verifier_job=args.verifier_job_spec,
        corrupt=False,
    )
    execute_pipeline(
        root=root / "corrupt",
        catalog=catalog,
        source_prefix=source_prefix,
        build_jobtag="synthetic-production-build-corrupt",
        verifier_jobtag="synthetic-production-verify-corrupt",
        builder_runner=builder_runner,
        coordinator_runner=coordinator_runner,
        verifier_runner=verifier_runner,
        builder_job=args.builder_job_spec,
        coordinator_job=args.coordinator_job_spec,
        verifier_job=args.verifier_job_spec,
        corrupt=True,
    )
    execute_pipeline(
        root=root / "tar-good",
        catalog=catalog,
        source_prefix=source_prefix,
        build_jobtag="synthetic-production-tar-build-good",
        verifier_jobtag="synthetic-production-tar-verify-good",
        builder_runner=builder_runner,
        coordinator_runner=coordinator_runner,
        verifier_runner=verifier_runner,
        builder_job=args.builder_job_spec,
        coordinator_job=args.coordinator_job_spec,
        verifier_job=args.verifier_job_spec,
        corrupt=False,
        source_mode="tar_sequential",
        tar_root=tar_root,
        tar_mapping=tar_mapping,
        tar_mapping_raw_sha256=tar_mapping_raw_sha256,
        tar_quick=tar_quick,
        tar_quick_raw_sha256=tar_quick_raw_sha256,
        tar_parser_gate=tar_parser_gate,
        tar_parser_gate_raw_sha256=tar_parser_gate_raw_sha256,
    )
    print("OFFLINE_GPIC_S3_PRODUCTION_STATIC_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
