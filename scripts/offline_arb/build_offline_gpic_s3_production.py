#!/usr/bin/env python3
"""Build all GPIC payload blobs into a fresh immutable S3 namespace.

Eight CPU workers own disjoint five-digit source shards through one explicit,
prescanned partition plan.  Physical blobs remain path-sorted, while compact
row fragments are restored to explicit global-row order for the dense merge.
Every worker finishes its complete node-local partition before a global
pre-upload approval can be issued.  The builder publishes worker receipts and
compact FSx evidence but never publishes the production root ``_SUCCESS``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow.parquet as pq

from scripts.offline_arb import offline_gpic_s3_production_common as common
from scripts.offline_arb import offline_gpic_tar_sequential as tar_source
from scripts.offline_arb import run_offline_gpic_blob_pilot as pilot_core


@dataclasses.dataclass(frozen=True)
class Item:
    local_index: int
    row_id: int
    image_path: str


@dataclasses.dataclass(frozen=True)
class SourceRead:
    item: Item
    payload: bytes
    sha256_raw: bytes
    attempts: int
    errors: int
    elapsed_ns: int


def canonical_json_line(value: dict[str, Any]) -> bytes:
    raw = common.canonical_json_bytes(value)
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise common.ProductionError("canonical JSONL record is not exactly one line")
    return raw


def validate_cross_host_source_archive_binding(
    source_archive: dict[str, Any],
    expected_tar_path: Path,
    observer_metadata: os.stat_result,
) -> int:
    """Validate only portable identity fields across distinct FUSE mounts."""

    identity = source_archive.get("identity")
    expected_keys = {"dev", "ino", "size", "mtime_ns", "ctime_ns"}
    if (
        set(source_archive) != {"path", "bytes", "identity"}
        or not isinstance(identity, dict)
        or set(identity) != expected_keys
        or any(type(identity[key]) is not int for key in expected_keys)
        or identity["dev"] < 0
        or identity["ino"] < 0
        or identity["size"] <= 0
        or identity["mtime_ns"] < 0
        or identity["ctime_ns"] < 0
        or type(source_archive.get("bytes")) is not int
        or source_archive["bytes"] <= 0
    ):
        raise common.ProductionError("malformed cross-host source archive identity")
    observer = pilot_core.FileIdentity.from_stat(observer_metadata)
    if (
        source_archive["path"] != str(expected_tar_path)
        or source_archive["bytes"] != observer.size
        or identity["size"] != observer.size
        or identity["mtime_ns"] != observer.mtime_ns
        or identity["ctime_ns"] != observer.ctime_ns
    ):
        raise common.ProductionError("cross-host source archive stable identity drift")
    return observer.size


def extract_shard_id(image_path: str, source_prefix: str) -> int:
    if not image_path.startswith(source_prefix):
        raise common.ProductionError("GPIC path escaped the frozen source prefix")
    relative = image_path[len(source_prefix) :]
    shard, separator, tail = relative.partition("/")
    if (
        separator != "/"
        or len(shard) != 5
        or not shard.isascii()
        or not shard.isdigit()
        or not tail
    ):
        raise common.ProductionError(f"malformed GPIC shard path: {image_path}")
    return int(shard)


def make_partition_plan(
    *,
    catalog: pilot_core.CatalogBinding,
    source_prefix: str,
    workers: int,
    partition_mode: str,
    production_gate: bool,
) -> dict[str, Any]:
    if partition_mode not in common.PRODUCTION_PARTITION_MODES:
        raise common.ProductionError("unsupported explicit partition mode")
    counts: dict[int, int] = {}
    parquet = pq.ParquetFile(catalog.gpic_path)
    group_start = 0
    for group_index in range(parquet.num_row_groups):
        table = parquet.read_row_group(
            group_index,
            columns=["row_id", "image_path"],
            use_threads=False,
        ).combine_chunks()
        row_ids = table.column("row_id")
        paths = table.column("image_path")
        for offset in range(table.num_rows):
            local_index = group_start + offset
            row_id = row_ids[offset].as_py()
            image_path = paths[offset].as_py()
            if (
                type(row_id) is not int
                or row_id != catalog.gpic_row_base + local_index
                or not isinstance(image_path, str)
            ):
                raise common.ProductionError(
                    f"partition prescan catalog identity drift local={local_index}"
                )
            shard = extract_shard_id(image_path, source_prefix)
            counts[shard] = counts.get(shard, 0) + 1
        group_start += table.num_rows
    if group_start != catalog.gpic_rows or sum(counts.values()) != catalog.gpic_rows:
        raise common.ProductionError("partition prescan row union drift")
    if production_gate and sorted(counts) != list(range(8000)):
        raise common.ProductionError("production GPIC shard set is not exactly 00000..07999")
    owners: dict[int, int] = {}
    worker_rows = [0] * workers
    worker_shards = [0] * workers
    if partition_mode == "shard_modulo":
        for shard in sorted(counts):
            owner = shard % workers
            owners[shard] = owner
            worker_rows[owner] += counts[shard]
            worker_shards[owner] += 1
    else:
        for shard, rows in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            owner = min(range(workers), key=lambda rank: (worker_rows[rank], rank))
            owners[shard] = owner
            worker_rows[owner] += rows
            worker_shards[owner] += 1
    if (
        min(worker_rows) <= 0
        or sum(worker_rows) != catalog.gpic_rows
        or sum(worker_shards) != len(counts)
        or max(worker_rows) / min(worker_rows) > common.PRODUCTION_BYTE_IMBALANCE_CAP
    ):
        raise common.ProductionError("partition prescan row balance/union gate failed")
    plan = {
        "schema_version": common.PARTITION_PLAN_SCHEMA,
        "build_jobtag": None,
        "partition_mode": partition_mode,
        "workers": workers,
        "rows": catalog.gpic_rows,
        "source_prefix": source_prefix,
        "catalog_hash": catalog.catalog_hash,
        "gpic_sha256": catalog.gpic_sha256,
        "shards": [
            {
                "shard_id": shard,
                "name": f"{shard:05d}",
                "rows": counts[shard],
                "worker_id": owners[shard],
            }
            for shard in sorted(counts)
        ],
        "worker_rows": worker_rows,
        "worker_shards": worker_shards,
        "row_imbalance_max_over_min": max(worker_rows) / min(worker_rows),
        "row_imbalance_cap": common.PRODUCTION_BYTE_IMBALANCE_CAP,
        "ownership": "each_exact_five_digit_source_shard_has_one_worker",
    }
    return plan


def finalize_partition_plan(plan: dict[str, Any], build_jobtag: str) -> dict[str, Any]:
    value = dict(plan)
    value["build_jobtag"] = build_jobtag
    value["partition_plan_sha256"] = common.self_hash(
        value, "partition_plan_sha256"
    )
    return value


def validate_partition_plan(
    value: dict[str, Any],
    *,
    build_jobtag: str,
    partition_mode: str,
    workers: int,
    catalog: pilot_core.CatalogBinding,
    source_prefix: str,
    production_gate: bool,
) -> dict[int, int]:
    common.validate_self_hash(value, "partition_plan_sha256", "partition plan")
    shards = value.get("shards")
    worker_rows = value.get("worker_rows")
    if (
        value.get("schema_version") != common.PARTITION_PLAN_SCHEMA
        or value.get("build_jobtag") != build_jobtag
        or value.get("partition_mode") != partition_mode
        or value.get("workers") != workers
        or value.get("rows") != catalog.gpic_rows
        or value.get("source_prefix") != source_prefix
        or value.get("catalog_hash") != catalog.catalog_hash
        or value.get("gpic_sha256") != catalog.gpic_sha256
        or not isinstance(shards, list)
        or not isinstance(worker_rows, list)
        or len(worker_rows) != workers
        or min(worker_rows, default=0) <= 0
        or sum(worker_rows) != catalog.gpic_rows
        or value.get("row_imbalance_max_over_min", float("inf"))
        > value.get("row_imbalance_cap", 0)
        or value.get("row_imbalance_cap")
        != common.PRODUCTION_BYTE_IMBALANCE_CAP
    ):
        raise common.ProductionError("partition plan top-level binding drift")
    owners: dict[int, int] = {}
    recomputed_rows = [0] * workers
    for expected_position, entry in enumerate(shards):
        shard = entry.get("shard_id") if isinstance(entry, dict) else None
        owner = entry.get("worker_id") if isinstance(entry, dict) else None
        rows = entry.get("rows") if isinstance(entry, dict) else None
        if (
            type(shard) is not int
            or shard != expected_position
            or entry.get("name") != f"{shard:05d}"
            or type(owner) is not int
            or not (0 <= owner < workers)
            or type(rows) is not int
            or rows <= 0
            or shard in owners
        ):
            raise common.ProductionError(
                f"partition plan shard binding drift position={expected_position}"
            )
        owners[shard] = owner
        recomputed_rows[owner] += rows
    if recomputed_rows != worker_rows:
        raise common.ProductionError("partition plan worker-row closure drift")
    if production_gate and len(owners) != 8000:
        raise common.ProductionError("production partition plan shard count drift")
    return owners


def load_tar_parser_gate_authority(
    path: Path,
    *,
    expected_raw_sha256: str,
    requested_tar_workers: int,
    production_gate: bool,
) -> dict[str, Any]:
    value, raw = common.strict_json(path)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    complete_path = path.parent / "_COMPLETE"
    complete_before = common.require_regular_file(complete_path)
    complete_raw = complete_path.read_bytes()
    complete_after = common.require_regular_file(complete_path)
    common.validate_self_hash(value, "summary_sha256", "tar parser gate")
    thresholds = value.get("thresholds")
    aggregate = value.get("aggregate")
    code_files = value.get("code", {}).get("files")
    if not isinstance(aggregate, list) or not isinstance(thresholds, dict):
        raise common.ProductionError("tar parser gate metrics are malformed")
    if [entry.get("tar_workers") for entry in aggregate] != [1, 4, 8]:
        raise common.ProductionError("tar parser gate arm sequence drift")
    recommended = next(
        (
            entry
            for entry in aggregate
            if entry.get("tar_workers") == requested_tar_workers
        ),
        None,
    )
    if (
        raw_sha256 != expected_raw_sha256
        or complete_before.st_size != 65
        or pilot_core.FileIdentity.from_stat(complete_before)
        != pilot_core.FileIdentity.from_stat(complete_after)
        or complete_raw != (raw_sha256 + "\n").encode("ascii")
        or value.get("schema_version")
        != "offline_gpic_tar_parser_gate_summary_v1"
        or value.get("status") != "PASS"
        or value.get("recommendation")
        != "PASS_FOR_BOUNDED_PRODUCTION_INTEGRATION"
        or value.get("recommended_tar_workers") != requested_tar_workers
        or value.get("workers") != 8
        or value.get("distinct_physical_boots") != 8
        or value.get("tars_per_worker") != 8
        or value.get("unique_tar_shards") != 64
        or value.get("shards_disjoint") is not True
        or value.get("semantic_digest_equal_across_1_4_8_per_worker") is not True
        or not isinstance(recommended, dict)
        or recommended.get("projected_8node_archive_bytes_per_second", 0)
        < thresholds.get("min_projected_aggregate_bytes_per_second", float("inf"))
        or recommended.get("estimated_full_archive_scan_hours", float("inf"))
        > thresholds.get("max_projected_scan_hours", 0)
        or recommended.get("max_peak_rss_bytes", float("inf"))
        > thresholds.get("max_peak_rss_bytes", 0)
    ):
        raise common.ProductionError("tar parser gate authority did not pass")
    if production_gate and (
        path != common.PRODUCTION_TAR_PARSER_GATE_SUMMARY
        or expected_raw_sha256
        != common.PRODUCTION_TAR_PARSER_GATE_RAW_SHA256
        or value.get("jobtag") != common.PRODUCTION_TAR_PARSER_GATE_JOBTAG
        or value.get("summary_sha256")
        != common.PRODUCTION_TAR_PARSER_GATE_SELF_SHA256
        or hashlib.sha256(complete_raw).hexdigest()
        != common.PRODUCTION_TAR_PARSER_GATE_COMPLETE_RAW_SHA256
        or requested_tar_workers != common.PRODUCTION_TAR_WORKERS
        or not isinstance(code_files, dict)
        or {
            label: binding.get("sha256")
            for label, binding in code_files.items()
            if isinstance(binding, dict)
        }
        != common.PRODUCTION_TAR_PARSER_GATE_CODE_SHA256
    ):
        raise common.ProductionError("frozen production tar parser authority drift")
    return {
        "path": str(path),
        "bytes": len(raw),
        "raw_sha256": raw_sha256,
        "summary_sha256": value["summary_sha256"],
        "complete_path": str(complete_path),
        "complete_bytes": len(complete_raw),
        "complete_raw_sha256": hashlib.sha256(complete_raw).hexdigest(),
        "jobtag": value.get("jobtag"),
        "recommended_tar_workers": requested_tar_workers,
        "distinct_physical_boots": value["distinct_physical_boots"],
        "semantic_digest_equal_across_1_4_8_per_worker": True,
        "recommended_metrics": recommended,
        "thresholds": thresholds,
    }


def load_source_authority(
    args: argparse.Namespace, catalog: pilot_core.CatalogBinding
) -> dict[str, Any]:
    if args.source_mode == "small_files":
        if (
            args.tar_mapping_manifest is not None
            or args.tar_mapping_raw_sha256
            or args.tar_quick_summary is not None
            or args.tar_quick_raw_sha256
            or args.tar_parser_gate_summary is not None
            or args.tar_parser_gate_raw_sha256
        ):
            raise common.ProductionError(
                "small-files mode must not bind tar authorities"
            )
        if args.source_thread_authority is None:
            thread_authority = {
                "status": "SYNTHETIC_ONLY",
                "threads": args.source_threads,
                "path": None,
                "raw_sha256": None,
            }
        else:
            thread_value, thread_raw = common.strict_json(
                args.source_thread_authority
            )
            if (
                hashlib.sha256(thread_raw).hexdigest()
                != args.source_thread_authority_raw_sha256
            ):
                raise common.ProductionError("source-thread authority raw SHA drift")
            candidates = {
                value
                for value in (
                    thread_value.get("selected_source_threads"),
                    thread_value.get("source_config", {}).get("threads")
                    if isinstance(thread_value.get("source_config"), dict)
                    else None,
                    thread_value.get("evidence", {})
                    .get("source_config", {})
                    .get("threads")
                    if isinstance(thread_value.get("evidence"), dict)
                    and isinstance(
                        thread_value.get("evidence", {}).get("source_config"),
                        dict,
                    )
                    else None,
                )
                if type(value) is int
            }
            if thread_value.get("status") != "PASS" or candidates != {
                args.source_threads
            }:
                raise common.ProductionError(
                    "source-thread analyzer authority did not approve this value"
                )
            thread_authority = {
                "status": "PASS",
                "threads": args.source_threads,
                "path": str(args.source_thread_authority),
                "raw_sha256": hashlib.sha256(thread_raw).hexdigest(),
            }
        return {
            "mode": "small_files",
            "source_prefix": args.source_prefix,
            "mapping_manifest": None,
            "production_authorized": False,
            "thread_authority": thread_authority,
        }
    if args.source_thread_authority is not None:
        raise common.ProductionError(
            "tar-sequential mode must not bind small-file thread authority"
        )
    if args.tar_workers not in common.PRODUCTION_ALLOWED_TAR_WORKERS:
        raise common.ProductionError(
            "tar worker count is outside the bounded-probe candidate set"
        )
    if (
        args.tar_mapping_manifest is None
        or not common.is_sha256(args.tar_mapping_raw_sha256)
        or args.tar_quick_summary is None
        or not common.is_sha256(args.tar_quick_raw_sha256)
        or args.tar_parser_gate_summary is None
        or not common.is_sha256(args.tar_parser_gate_raw_sha256)
    ):
        raise common.ProductionError(
            "tar-sequential mode requires exact mapping and throughput authorities"
        )
    sample_binding = tar_source.load_sample_mapping_authority(
        args.tar_mapping_manifest,
        expected_raw_sha256=args.tar_mapping_raw_sha256,
        catalog_hash=catalog.catalog_hash,
        gpic_rows=catalog.gpic_rows,
        row_base=catalog.gpic_row_base,
        gpic_sha256=catalog.gpic_sha256,
        tar_root=args.tar_root,
        source_prefix=args.source_prefix,
        production_gate=args.production_gate,
    )
    if args.production_gate and (
        sample_binding.get("report_sha256")
        != common.PRODUCTION_TAR_MAPPING_SELF_SHA256
    ):
        raise common.ProductionError("frozen tar mapping report self-hash drift")
    quick, quick_raw = common.strict_json(args.tar_quick_summary)
    if (
        hashlib.sha256(quick_raw).hexdigest() != args.tar_quick_raw_sha256
        or quick.get("schema_version")
        != "offline_gpic_tar_seq_quick_summary_v1"
        or quick.get("jobtag") != common.PRODUCTION_TAR_QUICK_JOBTAG
        or quick.get("distinct_boots") != 8
        or quick.get("sha_aggregate_bytes_per_second", 0) < 4_000_000_000
        or quick.get("sha_slowest_bytes_per_second", 0) < 500_000_000
        or quick.get("list_aggregate_bytes_per_second", 0) < 4_000_000_000
        or quick.get("list_slowest_bytes_per_second", 0) < 500_000_000
        or not isinstance(quick.get("rows"), list)
        or len(quick["rows"]) != 8
    ):
        raise common.ProductionError("tar throughput authority binding drift")
    parser_gate = load_tar_parser_gate_authority(
        args.tar_parser_gate_summary,
        expected_raw_sha256=args.tar_parser_gate_raw_sha256,
        requested_tar_workers=args.tar_workers,
        production_gate=args.production_gate,
    )
    effective_tar_selected_cap = min(
        common.PRODUCTION_TAR_MAX_SELECTED_BYTES_PER_TAR,
        args.max_s3_bytes_per_worker,
    )
    if args.production_gate and (
        effective_tar_selected_cap
        != common.PRODUCTION_TAR_MAX_SELECTED_BYTES_PER_TAR
    ):
        raise common.ProductionError(
            "production per-tar selected-payload cap drift"
        )
    return {
        "mode": "tar_sequential",
        "production_authorized": True,
        "attempts": 1,
        "threads": 1,
        "tar_workers": args.tar_workers,
        "read_buffer_bytes": common.PRODUCTION_TAR_READ_BUFFER_BYTES,
        "io_block_bytes": common.PRODUCTION_TAR_IO_BLOCK_BYTES,
        "max_selected_bytes_per_tar": effective_tar_selected_cap,
        "max_source_archive_bytes_per_worker": (
            args.max_source_archive_bytes_per_worker
        ),
        "tar_root": str(args.tar_root),
        "mapping": sample_binding,
        "parser_gate": parser_gate,
        "throughput": {
            "path": str(args.tar_quick_summary),
            "bytes": len(quick_raw),
            "raw_sha256": hashlib.sha256(quick_raw).hexdigest(),
            "sha_aggregate_bytes_per_second": quick[
                "sha_aggregate_bytes_per_second"
            ],
            "sha_slowest_bytes_per_second": quick[
                "sha_slowest_bytes_per_second"
            ],
            "list_aggregate_bytes_per_second": quick[
                "list_aggregate_bytes_per_second"
            ],
        },
    }


def source_read(
    item: Item,
    *,
    attempts: int,
    block_bytes: int,
    retry_base_seconds: float,
) -> SourceRead:
    started = time.perf_counter_ns()
    errors = 0
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        descriptor: int | None = None
        try:
            descriptor = os.open(
                item.image_path,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
            before = pilot_core.FileIdentity.from_stat(os.fstat(descriptor))
            if before.size <= 0 or before.size > 0xFFFFFFFF:
                raise common.ProductionError(
                    f"invalid GPIC source size row_id={item.row_id}"
                )
            digest = hashlib.sha256()
            payload = bytearray()
            while True:
                block = os.read(descriptor, block_bytes)
                if not block:
                    break
                digest.update(block)
                payload.extend(block)
            after = pilot_core.FileIdentity.from_stat(os.fstat(descriptor))
            if before != after or len(payload) != before.size:
                raise common.ProductionError(
                    f"changed/short GPIC source row_id={item.row_id}"
                )
            return SourceRead(
                item=item,
                payload=bytes(payload),
                sha256_raw=digest.digest(),
                attempts=attempt,
                errors=errors,
                elapsed_ns=time.perf_counter_ns() - started,
            )
        except common.ProductionError:
            raise
        except OSError as exc:
            errors += 1
            last_error = exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if attempt < attempts:
            time.sleep(retry_base_seconds * (2 ** (attempt - 1)))
    raise common.ProductionError(
        f"source read exhausted same-path retries row_id={item.row_id}"
    ) from last_error


def ordered_source_reads(
    items: Iterator[Item],
    *,
    threads: int,
    attempts: int,
    block_bytes: int,
    retry_base_seconds: float,
) -> Iterator[SourceRead]:
    if threads < 1:
        raise common.ProductionError("source thread count must be positive")
    pending: deque[concurrent.futures.Future[SourceRead]] = deque()
    max_pending = max(threads, 2 * threads)
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        for item in items:
            pending.append(
                pool.submit(
                    source_read,
                    item,
                    attempts=attempts,
                    block_bytes=block_bytes,
                    retry_base_seconds=retry_base_seconds,
                )
            )
            if len(pending) >= max_pending:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def materialize_partition(
    *,
    catalog: pilot_core.CatalogBinding,
    worker_id: int,
    shard_owners: dict[int, int],
    expected_rows: int,
    source_prefix: str,
    database_path: Path,
) -> dict[str, Any]:
    connection = sqlite3.connect(str(database_path))
    count = 0
    path_digest = hashlib.sha256()
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-262144")
        connection.execute(
            "CREATE TABLE rows ("
            "local_index INTEGER PRIMARY KEY, "
            "row_id INTEGER NOT NULL UNIQUE, "
            "shard_id INTEGER NOT NULL, "
            "image_path TEXT NOT NULL)"
        )
        parquet = pq.ParquetFile(catalog.gpic_path)
        group_start = 0
        for group_index in range(parquet.num_row_groups):
            table = parquet.read_row_group(
                group_index,
                columns=["row_id", "image_path"],
                use_threads=False,
            ).combine_chunks()
            row_ids = table.column("row_id")
            paths = table.column("image_path")
            batch: list[tuple[int, int, int, str]] = []
            for offset in range(table.num_rows):
                local_index = group_start + offset
                row_id = row_ids[offset].as_py()
                image_path = paths[offset].as_py()
                if (
                    type(row_id) is not int
                    or row_id != catalog.gpic_row_base + local_index
                ):
                    raise common.ProductionError(
                        f"catalog row identity drift local={local_index}"
                    )
                if (
                    not isinstance(image_path, str)
                    or not image_path.startswith(source_prefix)
                    or not Path(image_path).is_absolute()
                    or Path(image_path).suffix.lower() not in {".jpg", ".png"}
                    or ".." in Path(image_path).parts
                ):
                    raise common.ProductionError(
                        f"catalog GPIC path drift row_id={row_id}"
                    )
                shard = extract_shard_id(image_path, source_prefix)
                if shard_owners.get(shard) != worker_id:
                    continue
                batch.append((local_index, row_id, shard, image_path))
                encoded = image_path.encode("utf-8")
                path_digest.update(
                    common.RECORD_STRUCT.pack(
                        row_id,
                        0,
                        local_index,
                        len(encoded),
                        hashlib.sha256(encoded).digest(),
                    )
                )
                count += 1
            connection.executemany(
                "INSERT INTO rows(local_index,row_id,shard_id,image_path) "
                "VALUES (?,?,?,?)",
                batch,
            )
            connection.commit()
            group_start += table.num_rows
        if group_start != catalog.gpic_rows or count != expected_rows:
            raise common.ProductionError(
                f"partition materialization count drift count={count} expected={expected_rows}"
            )
        database_count = connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
        minmax = connection.execute(
            "SELECT MIN(local_index),MAX(local_index),MIN(row_id),MAX(row_id) FROM rows"
        ).fetchone()
        if (
            database_count != expected_rows
            or type(minmax[0]) is not int
            or type(minmax[1]) is not int
            or minmax[0] < 0
            or minmax[1] >= catalog.gpic_rows
            or minmax[2] != catalog.gpic_row_base + minmax[0]
            or minmax[3] != catalog.gpic_row_base + minmax[1]
        ):
            raise common.ProductionError("materialized partition boundary drift")
        connection.execute(
            "CREATE INDEX rows_path_order ON rows(image_path COLLATE BINARY,row_id)"
        )
        connection.execute(
            "CREATE INDEX rows_shard_order ON rows(shard_id,image_path COLLATE BINARY)"
        )
        connection.commit()
    finally:
        connection.close()
    with database_path.open("rb", buffering=0) as handle:
        os.fsync(handle.fileno())
    pilot_core.fsync_dir(database_path.parent)
    binding = common.file_binding(database_path)
    return {
        **binding,
        "rows": count,
        "catalog_order_path_binding_sha256": path_digest.hexdigest(),
        "sort_order": "image_path_binary_then_row_id",
    }


def iter_sorted_items(database_path: Path) -> Iterator[Item]:
    uri = f"file:{database_path}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        cursor = connection.execute(
            "SELECT local_index,row_id,image_path FROM rows "
            "ORDER BY image_path COLLATE BINARY,row_id"
        )
        previous: tuple[str, int] | None = None
        while True:
            rows = cursor.fetchmany(8192)
            if not rows:
                break
            for local_index, row_id, image_path in rows:
                key = (image_path, row_id)
                if previous is not None and key < previous:
                    raise common.ProductionError("SQLite path order regressed")
                previous = key
                yield Item(local_index, row_id, image_path)
    finally:
        connection.close()


class LocalPartitionWriter:
    def __init__(
        self,
        *,
        root: Path,
        s3_worker_root: Path,
        worker_id: int,
        workers: int,
        row_base: int,
        gpic_rows: int,
        expected_rows: int,
        blob_target_bytes: int,
        max_payload_bytes: int,
    ):
        self.root = root
        self.blob_root = root / "blobs"
        self.blob_root.mkdir(mode=0o700)
        self.s3_worker_root = s3_worker_root
        self.worker_id = worker_id
        self.workers = workers
        self.row_base = row_base
        self.gpic_rows = gpic_rows
        self.expected_rows = expected_rows
        self.blob_target_bytes = blob_target_bytes
        self.max_payload_bytes = max_payload_bytes
        self.row_dense_work_path = root / "row_dense_work.bin"
        self.blob_order_path = root / "blob_order.bin"
        self.row_fd = os.open(
            self.row_dense_work_path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        fragment_bytes = gpic_rows * common.RECORD_STRUCT.size
        try:
            os.posix_fallocate(self.row_fd, 0, fragment_bytes)
        except BaseException:
            os.close(self.row_fd)
            raise
        self.order_handle = self.blob_order_path.open("xb", buffering=0)
        self.handle: Any = None
        self.digest: Any = None
        self.current_path: Path | None = None
        self.current_blob_id = -1
        self.current_sequence = -1
        self.current_bytes = 0
        self.current_segments = 0
        self.total_bytes = 0
        self.total_rows = 0
        self.blobs: list[dict[str, Any]] = []

    def _open_blob(self) -> None:
        self.current_sequence += 1
        self.current_blob_id = common.global_blob_id(
            self.worker_id, self.current_sequence, self.workers
        )
        name = common.blob_name(self.current_blob_id)
        self.current_path = self.blob_root / name
        self.handle = self.current_path.open("xb", buffering=0)
        self.digest = hashlib.sha256()
        self.current_bytes = 0
        self.current_segments = 0

    def _close_blob(self) -> None:
        if self.handle is None or self.current_path is None or self.digest is None:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.handle = None
        binding = common.file_binding(self.current_path)
        if (
            binding["bytes"] != self.current_bytes
            or binding["sha256"] != self.digest.hexdigest()
            or self.current_bytes <= 0
            or self.current_segments <= 0
        ):
            raise common.ProductionError(
                f"closed local blob fingerprint drift: {self.current_path.name}"
            )
        self.blobs.append(
            {
                "sequence": self.current_sequence,
                "blob_id": self.current_blob_id,
                "name": self.current_path.name,
                "local_path": str(self.current_path),
                "key": str(self.s3_worker_root / self.current_path.name),
                "bytes": self.current_bytes,
                "sha256": binding["sha256"],
                "segments": self.current_segments,
            }
        )
        self.current_path = None
        self.digest = None

    def append(self, result: SourceRead) -> None:
        payload_bytes = len(result.payload)
        if payload_bytes <= 0 or payload_bytes > 0xFFFFFFFF:
            raise common.ProductionError("segment length escaped uint32")
        if self.handle is None:
            self._open_blob()
        elif self.current_bytes and (
            self.current_bytes + payload_bytes > self.blob_target_bytes
        ):
            self._close_blob()
            self._open_blob()
        if self.total_bytes + payload_bytes > self.max_payload_bytes:
            raise common.ProductionError("worker S3 payload hard cap exceeded")
        expected_row = self.row_base + result.item.local_index
        if (
            result.item.row_id != expected_row
            or result.item.local_index < 0
            or result.item.local_index >= self.gpic_rows
        ):
            raise common.ProductionError("source result row binding drift")
        offset = self.current_bytes
        common.write_all(
            self.handle,
            result.payload,
            label=f"local blob {self.current_blob_id}",
        )
        self.digest.update(result.payload)
        record = common.RECORD_STRUCT.pack(
            result.item.row_id,
            self.current_blob_id,
            offset,
            payload_bytes,
            result.sha256_raw,
        )
        common.pwrite_all(
            self.row_fd,
            record,
            result.item.local_index * common.RECORD_STRUCT.size,
            label="row_dense_work.bin",
        )
        common.write_all(self.order_handle, record, label="blob_order.bin")
        self.current_bytes += payload_bytes
        self.current_segments += 1
        self.total_bytes += payload_bytes
        self.total_rows += 1

    def close(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self._close_blob()
        self.order_handle.flush()
        os.fsync(self.order_handle.fileno())
        self.order_handle.close()
        os.fsync(self.row_fd)
        os.close(self.row_fd)
        if self.total_rows != self.expected_rows or not self.blobs:
            raise common.ProductionError("local partition row/blob closure failed")
        pilot_core.fsync_dir(self.blob_root)
        pilot_core.fsync_dir(self.root)
        order_binding = common.file_binding(self.blob_order_path)
        expected_fragment_bytes = self.expected_rows * common.RECORD_STRUCT.size
        if order_binding["bytes"] != expected_fragment_bytes:
            raise common.ProductionError("worker fragment exact-size drift")
        if common.require_regular_file(self.row_dense_work_path).st_size != (
            self.gpic_rows * common.RECORD_STRUCT.size
        ):
            raise common.ProductionError("dense work fragment exact-size drift")
        return self.blobs, order_binding

    def abort(self) -> None:
        for handle in (self.handle, self.order_handle):
            if handle is not None:
                try:
                    handle.close()
                except BaseException:
                    pass
        try:
            os.close(self.row_fd)
        except BaseException:
            pass


def validate_local_fragments(
    *,
    database_path: Path,
    row_dense_work_path: Path,
    row_fragment_path: Path,
    blob_order_path: Path,
    blobs: Sequence[dict[str, Any]],
    row_base: int,
    worker_id: int,
    workers: int,
    expected_rows: int,
) -> dict[str, Any]:
    by_id = {value["blob_id"]: value for value in blobs}
    if len(by_id) != len(blobs):
        raise common.ProductionError("duplicate blob ID in local manifest")
    dense_slots = (
        common.require_regular_file(row_dense_work_path).st_size
        // common.RECORD_STRUCT.size
    )
    dense_fd = os.open(row_dense_work_path, os.O_RDONLY)
    try:
        current_blob = -1
        current_end = 0
        current_segments = 0
        with blob_order_path.open("rb", buffering=0) as order_handle:
            for order_index in range(expected_rows):
                record = common.read_exact(
                    order_handle,
                    common.RECORD_STRUCT.size,
                    label=f"blob-order record {order_index}",
                )
                row_id, blob_id, offset, length, checksum = (
                    common.RECORD_STRUCT.unpack(record)
                )
                local_index = row_id - row_base
                if (
                    local_index < 0
                    or local_index >= dense_slots
                    or length <= 0
                    or not any(checksum)
                ):
                    raise common.ProductionError(
                        f"blob-order row binding drift order={order_index}"
                    )
                dense_record = common.pread_exact(
                    dense_fd,
                    common.RECORD_STRUCT.size,
                    local_index * common.RECORD_STRUCT.size,
                    label=f"dense work slot {local_index}",
                )
                if dense_record != record:
                    raise common.ProductionError(
                        f"dense/blob-order fragment mismatch local={local_index}"
                    )
                if blob_id != current_blob:
                    if current_blob != -1:
                        previous = by_id[current_blob]
                        if (
                            current_end != previous["bytes"]
                            or current_segments != previous["segments"]
                        ):
                            raise common.ProductionError(
                                f"blob segment closure drift blob={current_blob}"
                            )
                    expected_sequence = 0 if current_blob == -1 else (
                        (current_blob - worker_id) // workers + 1
                    )
                    expected_blob = common.global_blob_id(
                        worker_id, expected_sequence, workers
                    )
                    if blob_id != expected_blob or offset != 0 or blob_id not in by_id:
                        raise common.ProductionError(
                            f"blob transition drift blob={blob_id}"
                        )
                    current_blob = blob_id
                    current_end = 0
                    current_segments = 0
                if offset != current_end:
                    raise common.ProductionError(
                        f"blob offset gap/overlap blob={blob_id}"
                    )
                current_end += length
                current_segments += 1
            if order_handle.read(1):
                raise common.ProductionError("blob-order fragment has an extra tail")
        if current_blob == -1:
            raise common.ProductionError("blob-order fragment is empty")
        final = by_id[current_blob]
        if current_end != final["bytes"] or current_segments != final["segments"]:
            raise common.ProductionError("final blob segment closure drift")
        if len(blobs) != ((current_blob - worker_id) // workers) + 1:
            raise common.ProductionError("unreferenced local blob")
    finally:
        os.close(dense_fd)

    # Compact the globally-slotted local work file into a worker fragment that
    # is strictly increasing by explicit global row ID.  This remains valid for
    # either shard partition policy and lets Phase C perform a k-way exact merge.
    connection = sqlite3.connect(f"file:{database_path}?mode=ro&immutable=1", uri=True)
    dense_fd = os.open(row_dense_work_path, os.O_RDONLY)
    count = 0
    previous_local = -1
    digest = hashlib.sha256()
    try:
        with row_fragment_path.open("xb", buffering=0) as output:
            cursor = connection.execute(
                "SELECT local_index,row_id FROM rows ORDER BY local_index"
            )
            while True:
                rows = cursor.fetchmany(8192)
                if not rows:
                    break
                for local_index, expected_row_id in rows:
                    if local_index <= previous_local:
                        raise common.ProductionError(
                            "compact row-fragment local order regressed"
                        )
                    record = common.pread_exact(
                        dense_fd,
                        common.RECORD_STRUCT.size,
                        local_index * common.RECORD_STRUCT.size,
                        label=f"compact dense slot {local_index}",
                    )
                    row_id, blob_id, _offset, length, checksum = (
                        common.RECORD_STRUCT.unpack(record)
                    )
                    if (
                        row_id != expected_row_id
                        or row_id != row_base + local_index
                        or blob_id not in by_id
                        or blob_id % workers != worker_id
                        or length <= 0
                        or not any(checksum)
                    ):
                        raise common.ProductionError(
                            f"compact row fragment hole/rebinding local={local_index}"
                        )
                    common.write_all(output, record, label="row_fragment.bin")
                    digest.update(record)
                    count += 1
                    previous_local = local_index
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(dense_fd)
        connection.close()
    if count != expected_rows:
        raise common.ProductionError("compact row-fragment row count drift")
    binding = common.file_binding(row_fragment_path)
    if (
        binding["bytes"] != expected_rows * common.RECORD_STRUCT.size
        or binding["sha256"] != digest.hexdigest()
    ):
        raise common.ProductionError("compact row-fragment byte/SHA drift")
    row_dense_work_path.unlink()
    pilot_core.fsync_dir(row_fragment_path.parent)
    return binding


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


def establish_tar_mapping_authority(
    args: argparse.Namespace,
    *,
    catalog: pilot_core.CatalogBinding,
    shard_owners: dict[int, int],
    expected_rows: int,
    partition_plan: dict[str, Any],
    partition_plan_raw: bytes,
    source_authority: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish the exact global structural mapping before any tar payload read."""

    worker_root = args.coord_root / "tar_mapping_workers"
    authority_root = args.coord_root / "tar_mapping_authority"
    common.require_regular_dir(worker_root)
    common.require_regular_dir(authority_root)
    worker_mapping = tar_source.scan_worker_mapping(
        iter_sorted_items(args.local_root / "rows.sqlite3"),
        build_jobtag=args.build_jobtag,
        worker_id=args.worker_id,
        workers=args.workers,
        source_prefix=args.source_prefix,
        row_base=catalog.gpic_row_base,
        gpic_rows=catalog.gpic_rows,
        expected_rows=expected_rows,
        shard_owners=shard_owners,
        catalog_hash=catalog.catalog_hash,
        gpic_sha256=catalog.gpic_sha256,
        partition_plan_raw_sha256=hashlib.sha256(
            partition_plan_raw
        ).hexdigest(),
        partition_plan_sha256=partition_plan["partition_plan_sha256"],
    )
    source_archive_bytes = 0
    for shard_entry in worker_mapping["shards"]:
        tar_path = args.tar_root / shard_entry["tar_name"]
        metadata = common.require_regular_file(tar_path)
        if metadata.st_size <= 0:
            raise common.ProductionError("empty worker-owned source tar")
        identity = pilot_core.FileIdentity.from_stat(metadata)
        shard_entry["source_archive"] = {
            "path": str(tar_path),
            "bytes": metadata.st_size,
            "identity": dataclasses.asdict(identity),
        }
        source_archive_bytes += metadata.st_size
    if source_archive_bytes > args.max_source_archive_bytes_per_worker:
        raise common.ProductionError(
            "worker source archive-byte cap exceeded before payload reads"
        )
    worker_mapping["source_archive_count"] = len(worker_mapping["shards"])
    worker_mapping["source_archive_bytes"] = source_archive_bytes
    worker_mapping["max_source_archive_bytes_per_worker"] = (
        args.max_source_archive_bytes_per_worker
    )
    worker_mapping["source_archive_stat_before_payload"] = True
    worker_mapping["worker_mapping_sha256"] = common.self_hash(
        worker_mapping, "worker_mapping_sha256"
    )
    worker_path = worker_root / f"worker-{args.worker_id}.json"
    worker_raw = common.write_fsx_json_once(worker_path, worker_mapping)
    worker_raw_sha256 = hashlib.sha256(worker_raw).hexdigest()
    common.write_fsx_bytes_once(
        worker_root / f"worker-{args.worker_id}.ready",
        (worker_raw_sha256 + "\n").encode("ascii"),
    )
    pilot_core.fsync_dir(worker_root)
    ready_paths = [
        worker_root / f"worker-{rank}.ready" for rank in range(args.workers)
    ]
    wait_for_paths(
        ready_paths,
        coord_root=args.coord_root,
        timeout_seconds=args.barrier_timeout_seconds,
        label="tar worker structural mapping authorities",
    )
    full_path = authority_root / "full_mapping.json"
    complete_path = authority_root / "_COMPLETE"
    if args.worker_id == 0:
        common.exact_names(
            worker_root,
            {
                name
                for rank in range(args.workers)
                for name in (f"worker-{rank}.json", f"worker-{rank}.ready")
            },
            "tar worker mapping authority root",
        )
        worker_mappings: list[dict[str, Any]] = []
        worker_source_archive_bytes: list[int] = []
        for rank in range(args.workers):
            path = worker_root / f"worker-{rank}.json"
            value, raw = common.strict_json(path)
            marker = common.require_regular_file(
                worker_root / f"worker-{rank}.ready"
            )
            marker_raw = (
                worker_root / f"worker-{rank}.ready"
            ).read_bytes()
            archive_bytes = 0
            shards = value.get("shards")
            if not isinstance(shards, list):
                raise common.ProductionError(
                    f"tar worker shard inventory missing worker={rank}"
                )
            for shard_entry in shards:
                if not isinstance(shard_entry, dict):
                    raise common.ProductionError("malformed tar shard inventory")
                shard_id = shard_entry.get("shard_id")
                source_archive = shard_entry.get("source_archive")
                expected_tar_path = (
                    args.tar_root / tar_source.expected_tar_name(shard_id)
                    if type(shard_id) is int
                    else None
                )
                if not isinstance(source_archive, dict) or expected_tar_path is None:
                    raise common.ProductionError("malformed source archive binding")
                metadata = common.require_regular_file(expected_tar_path)
                try:
                    archive_bytes += validate_cross_host_source_archive_binding(
                        source_archive, expected_tar_path, metadata
                    )
                except common.ProductionError as exc:
                    raise common.ProductionError(
                        f"source archive stat binding drift shard={shard_id}"
                    ) from exc
            if (
                marker.st_size != 65
                or marker_raw
                != (hashlib.sha256(raw).hexdigest() + "\n").encode("ascii")
                or value.get("build_jobtag") != args.build_jobtag
                or value.get("worker_id") != rank
                or value.get("rows") != partition_plan["worker_rows"][rank]
                or value.get("catalog_hash") != catalog.catalog_hash
                or value.get("gpic_sha256") != catalog.gpic_sha256
                or value.get("partition_plan_raw_sha256")
                != hashlib.sha256(partition_plan_raw).hexdigest()
                or value.get("partition_plan_sha256")
                != partition_plan["partition_plan_sha256"]
                or value.get("source_archive_count") != len(shards)
                or value.get("source_archive_bytes") != archive_bytes
                or archive_bytes > args.max_source_archive_bytes_per_worker
                or value.get("max_source_archive_bytes_per_worker")
                != args.max_source_archive_bytes_per_worker
                or value.get("source_archive_stat_before_payload") is not True
            ):
                raise common.ProductionError(
                    f"tar worker mapping publication drift worker={rank}"
                )
            tar_source.validate_worker_mapping(
                value,
                worker_id=rank,
                workers=args.workers,
                source_prefix=args.source_prefix,
                row_base=catalog.gpic_row_base,
                gpic_rows=catalog.gpic_rows,
                expected_rows=partition_plan["worker_rows"][rank],
                shard_owners=shard_owners,
            )
            worker_mappings.append(value)
            worker_source_archive_bytes.append(archive_bytes)
        source_archive_imbalance = max(worker_source_archive_bytes) / min(
            worker_source_archive_bytes
        )
        if source_archive_imbalance > args.max_byte_imbalance:
            raise common.ProductionError(
                "global source archive-byte imbalance gate failed"
            )
        full_mapping = tar_source.make_full_mapping_authority(
            worker_mappings,
            build_jobtag=args.build_jobtag,
            workers=args.workers,
            shard_owners=shard_owners,
            source_prefix=args.source_prefix,
            tar_root=args.tar_root,
            row_base=catalog.gpic_row_base,
            gpic_rows=catalog.gpic_rows,
            catalog_hash=catalog.catalog_hash,
            gpic_sha256=catalog.gpic_sha256,
            partition_plan_raw_sha256=hashlib.sha256(
                partition_plan_raw
            ).hexdigest(),
            partition_plan_sha256=partition_plan["partition_plan_sha256"],
            sample_binding=source_authority["mapping"],
            production_gate=args.production_gate,
        )
        full_mapping["source_archive_bytes"] = sum(
            worker_source_archive_bytes
        )
        full_mapping["worker_source_archive_bytes"] = (
            worker_source_archive_bytes
        )
        full_mapping["source_archive_byte_imbalance_max_over_min"] = (
            source_archive_imbalance
        )
        full_mapping["source_archive_byte_imbalance_cap"] = (
            args.max_byte_imbalance
        )
        full_mapping["max_source_archive_bytes_per_worker"] = (
            args.max_source_archive_bytes_per_worker
        )
        full_mapping["full_mapping_sha256"] = common.self_hash(
            full_mapping, "full_mapping_sha256"
        )
        full_raw = common.write_fsx_json_once(full_path, full_mapping)
        full_raw_sha256 = hashlib.sha256(full_raw).hexdigest()
        common.write_fsx_bytes_once(
            complete_path, (full_raw_sha256 + "\n").encode("ascii")
        )
        pilot_core.fsync_dir(authority_root)
    common.wait_for_exact_file(
        complete_path,
        timeout_seconds=args.barrier_timeout_seconds,
        failure_root=args.coord_root,
        label="global tar structural mapping authority",
    )
    common.exact_names(
        authority_root,
        {"full_mapping.json", "_COMPLETE"},
        "global tar mapping authority root",
    )
    full_mapping, full_raw = common.strict_json(full_path)
    full_raw_sha256 = hashlib.sha256(full_raw).hexdigest()
    tar_source.validate_published_full_mapping(
        full_path,
        expected_raw_sha256=full_raw_sha256,
        expected_self_sha256=full_mapping.get("full_mapping_sha256", ""),
    )
    if (
        full_mapping.get("build_jobtag") != args.build_jobtag
        or full_mapping.get("workers") != args.workers
        or full_mapping.get("rows") != catalog.gpic_rows
        or full_mapping.get("catalog_hash") != catalog.catalog_hash
        or full_mapping.get("gpic_sha256") != catalog.gpic_sha256
        or full_mapping.get("partition_plan_raw_sha256")
        != hashlib.sha256(partition_plan_raw).hexdigest()
        or full_mapping.get("partition_plan_sha256")
        != partition_plan["partition_plan_sha256"]
        or full_mapping.get("sample_mapping") != source_authority["mapping"]
        or full_mapping.get("shard_count") != len(shard_owners)
        or full_mapping.get("payload_reads_before_authority") != 0
        or full_mapping.get("max_source_archive_bytes_per_worker")
        != args.max_source_archive_bytes_per_worker
        or full_mapping.get(
            "source_archive_byte_imbalance_max_over_min", float("inf")
        )
        > args.max_byte_imbalance
        or full_mapping.get("source_archive_byte_imbalance_cap")
        != args.max_byte_imbalance
        or not isinstance(full_mapping.get("worker_source_archive_bytes"), list)
        or len(full_mapping["worker_source_archive_bytes"]) != args.workers
        or max(full_mapping["worker_source_archive_bytes"], default=0)
        > args.max_source_archive_bytes_per_worker
        or sum(full_mapping["worker_source_archive_bytes"])
        != full_mapping.get("source_archive_bytes")
    ):
        raise common.ProductionError("global tar mapping authority binding drift")
    common.require_regular_file(complete_path)
    complete_raw = complete_path.read_bytes()
    if complete_raw != (full_raw_sha256 + "\n").encode("ascii"):
        raise common.ProductionError("global tar mapping completion marker drift")
    evidence = {
        "full": {
            "path": str(full_path),
            "bytes": len(full_raw),
            "raw_sha256": full_raw_sha256,
            "full_mapping_sha256": full_mapping["full_mapping_sha256"],
            "complete_path": str(complete_path),
            "complete_bytes": len(complete_raw),
            "complete_raw_sha256": hashlib.sha256(complete_raw).hexdigest(),
        },
        "worker": {
            "path": str(worker_path),
            "bytes": len(worker_raw),
            "raw_sha256": worker_raw_sha256,
            "worker_mapping_sha256": worker_mapping["worker_mapping_sha256"],
        },
    }
    return worker_mapping, evidence


def initialize_build(
    args: argparse.Namespace,
    *,
    code: dict[str, Any],
    catalog: pilot_core.CatalogBinding,
    catalog_manifest: dict[str, Any],
    topology_evidence: dict[str, Any],
    limits: dict[str, Any],
    source_authority: dict[str, Any],
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes, dict[int, int]]:
    s3_init_path = args.s3_root / common.S3_INIT_RELATIVE_PATH
    if args.worker_id == 0:
        for path in (args.artifact_root, args.coord_root, args.s3_root):
            if path.exists() or path.is_symlink():
                raise common.ProductionError(
                    f"production build namespace is not fresh: {path}"
                )
        args.artifact_root.mkdir(mode=0o700)
        args.fsx_workers_root.mkdir(mode=0o700)
        args.coord_root.mkdir(mode=0o700)
        args.s3_root.mkdir(mode=0o700)
        args.s3_controls_root.mkdir(mode=0o700)
        args.s3_workers_root.mkdir(mode=0o700)
        if args.source_mode == "tar_sequential":
            (args.coord_root / "tar_mapping_workers").mkdir(mode=0o700)
            (args.coord_root / "tar_mapping_authority").mkdir(mode=0o700)
        plan = finalize_partition_plan(
            make_partition_plan(
                catalog=catalog,
                source_prefix=args.source_prefix,
                workers=args.workers,
                partition_mode=args.partition_mode,
                production_gate=args.production_gate,
            ),
            args.build_jobtag,
        )
        partition_plan_raw = common.canonical_json_bytes(plan)
        expected = common.make_build_init(
            build_jobtag=args.build_jobtag,
            workers=args.workers,
            s3_root=args.s3_root,
            artifact_root=args.artifact_root,
            coord_root=args.coord_root,
            code=code,
            catalog=catalog_manifest,
            topology_evidence=topology_evidence,
            limits=limits,
            partition_plan=plan,
            partition_plan_raw=partition_plan_raw,
            source_authority=source_authority,
        )
        expected_raw = common.canonical_json_bytes(expected)
        common.write_s3_bytes_once(s3_init_path, expected_raw)
        common.write_fsx_bytes_once(
            args.coord_root / "partition_plan.json", partition_plan_raw
        )
        common.write_fsx_json_once(
            args.coord_root / "_INIT.json",
            common.make_coord_init(
                build_init=expected,
                build_init_raw=expected_raw,
            ),
        )
        pilot_core.fsync_dir(args.coord_root)
    common.wait_for_exact_file(
        args.coord_root / "_INIT.json",
        timeout_seconds=args.barrier_timeout_seconds,
        failure_root=args.coord_root if args.coord_root.exists() else None,
        label="FSx build initialization object",
    )
    common.wait_for_exact_file(
        s3_init_path,
        timeout_seconds=args.barrier_timeout_seconds,
        failure_root=args.coord_root,
        label="S3 build initialization object",
    )
    build_init_observed, build_init_raw_observed = common.strict_json(s3_init_path)
    common.validate_self_hash(
        build_init_observed, "build_init_sha256", "build initialization"
    )
    partition_plan, partition_plan_raw = common.strict_json(
        args.coord_root / "partition_plan.json"
    )
    shard_owners = validate_partition_plan(
        partition_plan,
        build_jobtag=args.build_jobtag,
        partition_mode=args.partition_mode,
        workers=args.workers,
        catalog=catalog,
        source_prefix=args.source_prefix,
        production_gate=args.production_gate,
    )
    expected = common.make_build_init(
        build_jobtag=args.build_jobtag,
        workers=args.workers,
        s3_root=args.s3_root,
        artifact_root=args.artifact_root,
        coord_root=args.coord_root,
        code=code,
        catalog=catalog_manifest,
        topology_evidence=topology_evidence,
        limits=limits,
        partition_plan=partition_plan,
        partition_plan_raw=partition_plan_raw,
        source_authority=source_authority,
    )
    if (
        build_init_observed != expected
        or build_init_raw_observed != common.canonical_json_bytes(expected)
    ):
        raise common.ProductionError("S3 initialization exact binding drift")
    build_init = build_init_observed
    build_init_raw = build_init_raw_observed
    coord_init, coord_init_raw = common.strict_json(args.coord_root / "_INIT.json")
    expected_coord = common.make_coord_init(
        build_init=build_init, build_init_raw=build_init_raw
    )
    common.validate_self_hash(coord_init, "coord_init_sha256", "coord initialization")
    if coord_init != expected_coord or coord_init_raw != common.canonical_json_bytes(
        expected_coord
    ):
        raise common.ProductionError("FSx/S3 initialization binding drift")
    common.require_regular_dir(args.artifact_root)
    common.require_regular_dir(args.fsx_workers_root)
    common.require_regular_dir(args.coord_root)
    if args.source_mode == "tar_sequential":
        common.require_regular_dir(args.coord_root / "tar_mapping_workers")
        common.require_regular_dir(args.coord_root / "tar_mapping_authority")
    common.require_regular_dir(args.s3_root)
    if not args.s3_controls_root.exists():
        args.s3_controls_root.mkdir(mode=0o700)
    common.require_regular_dir(args.s3_controls_root)
    if not args.s3_workers_root.exists():
        args.s3_workers_root.mkdir(mode=0o700)
    common.require_regular_dir(args.s3_workers_root)
    if args.s3_worker_root.exists() or args.s3_worker_root.is_symlink():
        raise common.ProductionError("S3 worker namespace is not fresh")
    args.s3_worker_root.mkdir(mode=0o700)
    if args.fsx_worker_root.exists() or args.fsx_worker_root.is_symlink():
        raise common.ProductionError("FSx worker namespace is not fresh")
    args.fsx_worker_root.mkdir(mode=0o700)
    if args.local_root.exists() or args.local_root.is_symlink():
        raise common.ProductionError("local production namespace is not fresh")
    args.local_root.mkdir(parents=True, mode=0o700)
    common.assert_no_root_success(args.s3_root)
    return (
        build_init,
        build_init_raw,
        partition_plan,
        partition_plan_raw,
        shard_owners,
    )


def publish_preupload(
    args: argparse.Namespace,
    *,
    code: dict[str, Any],
    catalog_manifest: dict[str, Any],
    topology_evidence: dict[str, Any],
    local_topology: dict[str, Any],
    runtime: dict[str, Any],
    build_init: dict[str, Any],
    build_init_raw: bytes,
    partition_plan: dict[str, Any],
    partition_plan_raw: bytes,
    local_manifest: dict[str, Any],
    local_manifest_raw: bytes,
) -> tuple[dict[str, Any], bytes]:
    files = [
        (args.local_root / "row_fragment.bin", "row_fragment.bin"),
        (args.local_root / "blob_order.bin", "blob_order.bin"),
        (args.local_root / "local_manifest.json", "local_manifest.json"),
    ]
    if args.source_mode == "tar_sequential":
        files.append(
            (args.local_root / "tar_telemetry.jsonl", "tar_telemetry.jsonl")
        )
    for source, name in files:
        common.copy_file_fsync_once(
            source,
            args.fsx_worker_root / name,
            block_bytes=common.PRODUCTION_S3_BLOCK_BYTES,
        )
    preupload = {
        "schema_version": common.PRODUCTION_SCHEMA,
        "generated_at": common.utc_now(),
        "build_jobtag": args.build_jobtag,
        "worker_id": args.worker_id,
        "workers": args.workers,
        "root": str(args.fsx_worker_root),
        "s3_worker_root": str(args.s3_worker_root),
        "code": code,
        "catalog": catalog_manifest,
        "topology_evidence": topology_evidence,
        "local_topology": local_topology,
        "runtime": runtime,
        **common.build_init_binding(build_init, build_init_raw),
        "partition_mode": partition_plan["partition_mode"],
        "partition_plan_raw_sha256": hashlib.sha256(
            partition_plan_raw
        ).hexdigest(),
        "partition_plan_sha256": partition_plan["partition_plan_sha256"],
        "local_manifest_raw_sha256": hashlib.sha256(local_manifest_raw).hexdigest(),
        "local_manifest_sha256": local_manifest["local_manifest_sha256"],
        "rows": local_manifest["rows"],
        "payload_bytes": local_manifest["payload_bytes"],
        "blob_count": len(local_manifest["blobs"]),
        "row_fragment": common.file_binding(
            args.fsx_worker_root / "row_fragment.bin"
        ),
        "blob_order": common.file_binding(args.fsx_worker_root / "blob_order.bin"),
        "tar_telemetry": local_manifest.get("tar_telemetry"),
        "tar_mapping_evidence": local_manifest.get("tar_mapping_evidence"),
        "source_threads": args.source_threads,
        "tar_workers": args.tar_workers,
        "source_mode": args.source_mode,
        "source_thread_authority": (
            "not_applicable_tar_sequential"
            if args.source_mode == "tar_sequential"
            else "scale_analyzer_result_parameter"
        ),
        "limits": build_init["limits"],
    }
    preupload["preupload_sha256"] = common.self_hash(
        preupload, "preupload_sha256"
    )
    raw = common.write_fsx_json_once(
        args.fsx_worker_root / "preupload.json", preupload
    )
    common.write_fsx_bytes_once(
        args.fsx_worker_root / "_PREUPLOAD_READY",
        (preupload["preupload_sha256"] + "\n").encode("ascii"),
    )
    pilot_core.fsync_dir(args.fsx_worker_root)
    return preupload, raw


def approve_uploads(
    args: argparse.Namespace,
    *,
    code: dict[str, Any],
    catalog_manifest: dict[str, Any],
    topology_evidence: dict[str, Any],
    build_init: dict[str, Any],
    build_init_raw: bytes,
    partition_plan: dict[str, Any],
    partition_plan_raw: bytes,
) -> tuple[dict[str, Any], bytes]:
    paths = [
        args.fsx_workers_root / f"worker-{rank}" / "_PREUPLOAD_READY"
        for rank in range(args.workers)
    ]
    wait_for_paths(
        paths,
        coord_root=args.coord_root,
        timeout_seconds=args.barrier_timeout_seconds,
        label="pre-upload worker evidence",
    )
    approval_path = args.coord_root / "UPLOAD_APPROVED.json"
    if args.worker_id == 0:
        workers: list[dict[str, Any]] = []
        keys: set[str] = set()
        payloads: list[int] = []
        runtimes: list[dict[str, Any]] = []
        global_tar_mapping: dict[str, Any] | None = None
        for rank in range(args.workers):
            root = args.fsx_workers_root / f"worker-{rank}"
            expected_names = {
                "row_fragment.bin",
                "blob_order.bin",
                "local_manifest.json",
                "preupload.json",
                "_PREUPLOAD_READY",
            }
            if args.source_mode == "tar_sequential":
                expected_names.add("tar_telemetry.jsonl")
            common.exact_names(
                root,
                expected_names,
                f"pre-upload worker {rank}",
            )
            value, raw = common.strict_json(root / "preupload.json")
            common.validate_self_hash(value, "preupload_sha256", f"preupload {rank}")
            marker = (root / "_PREUPLOAD_READY").read_text(encoding="ascii").strip()
            local_manifest, local_raw = common.strict_json(root / "local_manifest.json")
            common.validate_self_hash(
                local_manifest, "local_manifest_sha256", f"local manifest {rank}"
            )
            if (
                marker != value["preupload_sha256"]
                or value.get("build_jobtag") != args.build_jobtag
                or value.get("worker_id") != rank
                or value.get("workers") != args.workers
                or value.get("root") != str(root)
                or value.get("s3_worker_root")
                != str(args.s3_workers_root / f"worker-{rank}")
                or value.get("code") != code
                or value.get("catalog") != catalog_manifest
                or value.get("topology_evidence") != topology_evidence
                or value.get("limits") != build_init["limits"]
                or value.get("build_init_raw_sha256")
                != hashlib.sha256(build_init_raw).hexdigest()
                or value.get("build_init_sha256")
                != build_init["build_init_sha256"]
                or value.get("partition_mode") != args.partition_mode
                or value.get("partition_plan_raw_sha256")
                != hashlib.sha256(partition_plan_raw).hexdigest()
                or value.get("partition_plan_sha256")
                != partition_plan["partition_plan_sha256"]
                or value.get("local_manifest_raw_sha256")
                != hashlib.sha256(local_raw).hexdigest()
                or value.get("local_manifest_sha256")
                != local_manifest["local_manifest_sha256"]
                or value.get("rows")
                != partition_plan["worker_rows"][rank]
                or local_manifest.get("rows")
                != partition_plan["worker_rows"][rank]
                or value.get("payload_bytes")
                != local_manifest.get("payload_bytes")
                or value.get("blob_count") != len(local_manifest.get("blobs", []))
                or value.get("source_threads") != args.source_threads
                or value.get("tar_workers") != args.tar_workers
                or value.get("source_mode") != args.source_mode
                or value.get("source_thread_authority")
                != (
                    "not_applicable_tar_sequential"
                    if args.source_mode == "tar_sequential"
                    else "scale_analyzer_result_parameter"
                )
                or value.get("tar_telemetry")
                != local_manifest.get("tar_telemetry")
                or value.get("tar_mapping_evidence")
                != local_manifest.get("tar_mapping_evidence")
                or local_manifest.get("build_jobtag") != args.build_jobtag
                or local_manifest.get("worker_id") != rank
                or local_manifest.get("workers") != args.workers
                or local_manifest.get("code") != code
                or local_manifest.get("catalog") != catalog_manifest
                or local_manifest.get("build_init_raw_sha256")
                != hashlib.sha256(build_init_raw).hexdigest()
                or local_manifest.get("build_init_sha256")
                != build_init["build_init_sha256"]
                or local_manifest.get("partition_mode") != args.partition_mode
                or local_manifest.get("partition_plan_raw_sha256")
                != hashlib.sha256(partition_plan_raw).hexdigest()
                or local_manifest.get("partition_plan_sha256")
                != partition_plan["partition_plan_sha256"]
            ):
                raise common.ProductionError(
                    f"pre-upload identity binding drift worker={rank}"
                )
            for file_key in ("row_fragment", "blob_order"):
                binding = value[file_key]
                path = root / binding["name"]
                if common.file_binding(path) != binding:
                    raise common.ProductionError(
                        f"pre-upload fragment drift worker={rank} file={file_key}"
                    )
                if binding["bytes"] != (
                    partition_plan["worker_rows"][rank]
                    * common.RECORD_STRUCT.size
                ):
                    raise common.ProductionError(
                        f"pre-upload fragment size drift worker={rank}"
                    )
            if args.source_mode == "tar_sequential":
                telemetry = local_manifest.get("tar_telemetry")
                mapping_evidence = local_manifest.get("tar_mapping_evidence")
                if (
                    not isinstance(telemetry, dict)
                    or not isinstance(telemetry.get("file"), dict)
                    or common.file_binding(root / "tar_telemetry.jsonl")
                    != telemetry["file"]
                    or telemetry.get("rows")
                    != partition_plan["worker_rows"][rank]
                    or telemetry.get("records")
                    != partition_plan["worker_shards"][rank]
                    or telemetry.get("tar_workers") != args.tar_workers
                    or telemetry.get("archive_bytes", 0)
                    > args.max_source_archive_bytes_per_worker
                    or telemetry.get("max_source_archive_bytes_per_worker")
                    != args.max_source_archive_bytes_per_worker
                    or telemetry.get("attempts") != 1
                    or telemetry.get("errors") != 0
                    or not isinstance(mapping_evidence, dict)
                    or not isinstance(mapping_evidence.get("full"), dict)
                    or not isinstance(mapping_evidence.get("worker"), dict)
                ):
                    raise common.ProductionError(
                        f"tar telemetry/mapping evidence drift worker={rank}"
                    )
                full = mapping_evidence["full"]
                worker_mapping = mapping_evidence["worker"]
                if global_tar_mapping is None:
                    global_tar_mapping = full
                elif global_tar_mapping != full:
                    raise common.ProductionError(
                        "workers disagree on global tar mapping authority"
                    )
                expected_worker_path = (
                    args.coord_root / "tar_mapping_workers" / f"worker-{rank}.json"
                )
                expected_full_path = (
                    args.coord_root
                    / "tar_mapping_authority"
                    / "full_mapping.json"
                )
                worker_file = common.file_binding(expected_worker_path)
                full_file = common.file_binding(expected_full_path)
                if (
                    worker_mapping.get("path") != str(expected_worker_path)
                    or worker_mapping.get("bytes") != worker_file["bytes"]
                    or worker_mapping.get("raw_sha256")
                    != worker_file["sha256"]
                    or full.get("path") != str(expected_full_path)
                    or full.get("bytes") != full_file["bytes"]
                    or full.get("raw_sha256") != full_file["sha256"]
                ):
                    raise common.ProductionError(
                        f"tar mapping file binding drift worker={rank}"
                    )
            elif (
                local_manifest.get("tar_telemetry") is not None
                or local_manifest.get("tar_mapping_evidence") is not None
            ):
                raise common.ProductionError(
                    "small-files worker unexpectedly published tar evidence"
                )
            worker_keys: list[str] = []
            worker_bytes = 0
            worker_segments = 0
            blobs = local_manifest.get("blobs")
            if not isinstance(blobs, list) or not blobs:
                raise common.ProductionError(f"empty blob list worker={rank}")
            for sequence, blob in enumerate(blobs):
                blob_id = common.global_blob_id(rank, sequence, args.workers)
                name = common.blob_name(blob_id)
                key = str(args.s3_workers_root / f"worker-{rank}" / name)
                if (
                    blob.get("sequence") != sequence
                    or blob.get("blob_id") != blob_id
                    or blob.get("name") != name
                    or blob.get("key") != key
                    or type(blob.get("bytes")) is not int
                    or blob["bytes"] <= 0
                    or not common.is_sha256(blob.get("sha256"))
                    or type(blob.get("segments")) is not int
                    or blob["segments"] <= 0
                    or key in keys
                ):
                    raise common.ProductionError(
                        f"pre-upload blob/key drift worker={rank} sequence={sequence}"
                    )
                keys.add(key)
                worker_keys.append(key)
                worker_bytes += blob["bytes"]
                worker_segments += blob["segments"]
                intended_path = Path(key)
                if intended_path.exists() or intended_path.is_symlink():
                    raise common.ProductionError(
                        f"blob appeared before global approval: {key}"
                    )
            if (
                worker_bytes != value["payload_bytes"]
                or worker_segments != partition_plan["worker_rows"][rank]
                or worker_bytes > args.max_s3_bytes_per_worker
            ):
                raise common.ProductionError(
                    f"pre-upload byte/row closure drift worker={rank}"
                )
            workers.append(
                {
                    "worker_id": rank,
                    "preupload_raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "preupload_sha256": value["preupload_sha256"],
                    "local_manifest_raw_sha256": hashlib.sha256(local_raw).hexdigest(),
                    "local_manifest_sha256": local_manifest["local_manifest_sha256"],
                    "rows": value["rows"],
                    "payload_bytes": worker_bytes,
                    "blob_count": len(blobs),
                    "keys": worker_keys,
                    "runtime": value["runtime"],
                    "tar_telemetry": local_manifest.get("tar_telemetry"),
                    "tar_mapping_worker": (
                        local_manifest.get("tar_mapping_evidence", {}).get("worker")
                        if isinstance(
                            local_manifest.get("tar_mapping_evidence"), dict
                        )
                        else None
                    ),
                }
            )
            payloads.append(worker_bytes)
            runtimes.append(value["runtime"])
        total_payload = sum(payloads)
        if (
            total_payload > args.max_s3_payload_bytes
            or total_payload + args.s3_controls_reserve_bytes
            > args.max_s3_total_bytes
            or max(payloads) / min(payloads) > args.max_byte_imbalance
            or len(keys) != sum(value["blob_count"] for value in workers)
            or sum(value["rows"] for value in workers)
            != sum(partition_plan["worker_rows"])
        ):
            raise common.ProductionError("global pre-upload cap/union gate failed")
        boot_hashes = common.stable_unique_boot_hashes(runtimes, "builder")
        if args.production_gate and len(boot_hashes) != args.workers:
            raise common.ProductionError("builders are not on eight physical hosts")
        fsx_bytes = common.fsx_tree_bytes([args.artifact_root, args.coord_root])
        if fsx_bytes > args.max_fsx_incremental_bytes:
            raise common.ProductionError("pre-upload FSx artifact cap exceeded")
        approval = {
            "schema_version": common.PRODUCTION_SCHEMA,
            "generated_at": common.utc_now(),
            "status": "UPLOAD_APPROVED",
            "build_jobtag": args.build_jobtag,
            "workers": workers,
            "worker_count": args.workers,
            "rows": sum(partition_plan["worker_rows"]),
            "payload_bytes": total_payload,
            "blob_count": len(keys),
            "keys_sha256": hashlib.sha256(
                common.canonical_json_bytes(sorted(keys))
            ).hexdigest(),
            "byte_imbalance_max_over_min": max(payloads) / min(payloads),
            "byte_imbalance_cap": args.max_byte_imbalance,
            "row_imbalance_max_over_min": partition_plan[
                "row_imbalance_max_over_min"
            ],
            "row_imbalance_cap": partition_plan["row_imbalance_cap"],
            "max_s3_payload_bytes": args.max_s3_payload_bytes,
            "max_s3_total_bytes": args.max_s3_total_bytes,
            "s3_controls_reserve_bytes": args.s3_controls_reserve_bytes,
            "fsx_bytes_at_approval": fsx_bytes,
            "max_fsx_incremental_bytes": args.max_fsx_incremental_bytes,
            "builder_boot_id_sha256": sorted(boot_hashes),
            "partition_mode": args.partition_mode,
            "partition_plan_raw_sha256": hashlib.sha256(
                partition_plan_raw
            ).hexdigest(),
            "partition_plan_sha256": partition_plan["partition_plan_sha256"],
            "tar_workers": args.tar_workers,
            "full_tar_mapping": global_tar_mapping,
            "code": code,
            "catalog": catalog_manifest,
            "topology_evidence": topology_evidence,
            **common.build_init_binding(build_init, build_init_raw),
            "publication_semantics": "zero_blob_uploads_before_this_approval",
        }
        approval["approval_sha256"] = common.self_hash(
            approval, "approval_sha256"
        )
        common.write_fsx_json_once(approval_path, approval)
        pilot_core.fsync_dir(args.coord_root)
    common.wait_for_exact_file(
        approval_path,
        timeout_seconds=args.barrier_timeout_seconds,
        failure_root=args.coord_root,
        label="global upload approval",
    )
    approval, approval_raw = common.strict_json(approval_path)
    common.validate_self_hash(approval, "approval_sha256", "upload approval")
    own_local_manifest, _ = common.strict_json(
        args.fsx_worker_root / "local_manifest.json"
    )
    own_mapping_evidence = own_local_manifest.get("tar_mapping_evidence")
    expected_full_tar_mapping = (
        own_mapping_evidence.get("full")
        if isinstance(own_mapping_evidence, dict)
        else None
    )
    if (
        approval.get("status") != "UPLOAD_APPROVED"
        or approval.get("build_jobtag") != args.build_jobtag
        or approval.get("worker_count") != args.workers
        or approval.get("rows") != sum(partition_plan["worker_rows"])
        or approval.get("payload_bytes", args.max_s3_payload_bytes + 1)
        > args.max_s3_payload_bytes
        or approval.get("byte_imbalance_max_over_min", float("inf"))
        > args.max_byte_imbalance
        or approval.get("code") != code
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
        or approval.get("tar_workers") != args.tar_workers
        or approval.get("full_tar_mapping") != expected_full_tar_mapping
    ):
        raise common.ProductionError("global upload approval binding drift")
    own = approval["workers"][args.worker_id]
    own_preupload_raw = (
        args.fsx_worker_root / "preupload.json"
    ).read_bytes()
    if (
        own.get("worker_id") != args.worker_id
        or own.get("preupload_raw_sha256")
        != hashlib.sha256(own_preupload_raw).hexdigest()
    ):
        raise common.ProductionError("upload approval does not bind this worker")
    return approval, approval_raw


def publish_worker_receipt(
    args: argparse.Namespace,
    *,
    code: dict[str, Any],
    catalog_manifest: dict[str, Any],
    topology_evidence: dict[str, Any],
    local_topology: dict[str, Any],
    runtime: dict[str, Any],
    build_init: dict[str, Any],
    build_init_raw: bytes,
    partition_plan: dict[str, Any],
    partition_plan_raw: bytes,
    local_manifest: dict[str, Any],
    local_manifest_raw: bytes,
    preupload: dict[str, Any],
    preupload_raw: bytes,
    approval: dict[str, Any],
    approval_raw: bytes,
    build_elapsed_seconds: float,
    upload_elapsed_seconds: float,
) -> tuple[dict[str, Any], bytes]:
    public_blobs = [
        {key: value for key, value in blob.items() if key != "local_path"}
        for blob in local_manifest["blobs"]
    ]
    common.exact_names(
        args.s3_worker_root,
        {blob["name"] for blob in local_manifest["blobs"]},
        f"S3 worker {args.worker_id} before receipt",
    )
    for blob in local_manifest["blobs"]:
        metadata = common.require_regular_file(args.s3_worker_root / blob["name"])
        if metadata.st_size != blob["bytes"]:
            raise common.ProductionError(
                f"S3 visible size drift blob={blob['blob_id']}"
            )
    receipt = {
        "schema_version": common.WORKER_RECEIPT_SCHEMA,
        "generated_at": common.utc_now(),
        "build_jobtag": args.build_jobtag,
        "worker_id": args.worker_id,
        "workers": args.workers,
        "root": str(args.fsx_worker_root),
        "s3_worker_root": str(args.s3_worker_root),
        "rows": local_manifest["rows"],
        "payload_bytes": local_manifest["payload_bytes"],
        "blobs": public_blobs,
        "row_fragment": preupload["row_fragment"],
        "blob_order": preupload["blob_order"],
        "tar_telemetry": local_manifest.get("tar_telemetry"),
        "tar_mapping_evidence": local_manifest.get("tar_mapping_evidence"),
        "local_manifest_raw_sha256": hashlib.sha256(local_manifest_raw).hexdigest(),
        "local_manifest_sha256": local_manifest["local_manifest_sha256"],
        "preupload_raw_sha256": hashlib.sha256(preupload_raw).hexdigest(),
        "preupload_sha256": preupload["preupload_sha256"],
        "upload_approval_raw_sha256": hashlib.sha256(approval_raw).hexdigest(),
        "upload_approval_sha256": approval["approval_sha256"],
        "code": code,
        "catalog": catalog_manifest,
        "topology_evidence": topology_evidence,
        "local_topology": local_topology,
        "runtime": runtime,
        **common.build_init_binding(build_init, build_init_raw),
        "partition_mode": partition_plan["partition_mode"],
        "partition_plan_raw_sha256": hashlib.sha256(
            partition_plan_raw
        ).hexdigest(),
        "partition_plan_sha256": partition_plan["partition_plan_sha256"],
        "timing": {
            "local_build_seconds": build_elapsed_seconds,
            "s3_upload_seconds": upload_elapsed_seconds,
        },
        "source": {
            "mode": args.source_mode,
            "threads": args.source_threads,
            "tar_workers": args.tar_workers,
            "attempts": args.source_attempts,
            "block_bytes": args.source_block_bytes,
            "tar_read_buffer_bytes": common.PRODUCTION_TAR_READ_BUFFER_BYTES,
            "tar_io_block_bytes": common.PRODUCTION_TAR_IO_BLOCK_BYTES,
            "max_selected_bytes_per_tar": build_init["source_authority"].get(
                "max_selected_bytes_per_tar"
            ),
            "max_source_archive_bytes_per_worker": (
                args.max_source_archive_bytes_per_worker
            ),
            "fallbacks": 0,
        },
        "publication_semantics": (
            "worker_receipt_success_last_no_root_success"
        ),
    }
    receipt["receipt_sha256"] = common.self_hash(receipt, "receipt_sha256")
    raw = common.write_fsx_json_once(args.fsx_worker_root / "receipt.json", receipt)
    common.write_fsx_bytes_once(
        args.fsx_worker_root / "_SUCCESS",
        (receipt["receipt_sha256"] + "\n").encode("ascii"),
    )
    pilot_core.fsync_dir(args.fsx_worker_root)
    return receipt, raw


def run(args: argparse.Namespace) -> None:
    code = common.code_snapshot(
        runner=Path(__file__),
        job_spec=args.job_spec,
        extra_files={"tar_source": Path(tar_source.__file__)},
    )
    topology_evidence = common.load_topology_evidence(
        args.topology_summary,
        expected_jobtag=args.topology_jobtag,
        expected_raw_sha256=args.topology_summary_raw_sha256,
    )
    catalog = common.load_production_catalog(
        args.catalog_root, production_gate=args.production_gate
    )
    catalog_value = common.catalog_manifest(catalog)
    source_authority = load_source_authority(args, catalog)
    if args.production_gate and not source_authority.get("production_authorized"):
        raise common.ProductionError(
            "small-file production source is paused pending official-tar mapping pilot"
        )
    limits = common.production_limits(
        workers=args.workers,
        gpic_rows=catalog.gpic_rows,
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
    local_topology = common.local_mount_binding(
        args.local_prefix, production_gate=args.production_gate
    )
    runtime = common.runtime_identity()
    (
        build_init,
        build_init_raw,
        partition_plan,
        partition_plan_raw,
        shard_owners,
    ) = initialize_build(
        args,
        code=code,
        catalog=catalog,
        catalog_manifest=catalog_value,
        topology_evidence=topology_evidence,
        limits=limits,
        source_authority=source_authority,
    )
    expected_rows = partition_plan["worker_rows"][args.worker_id]
    writer: LocalPartitionWriter | None = None
    try:
        build_started = time.monotonic()
        database_binding = materialize_partition(
            catalog=catalog,
            worker_id=args.worker_id,
            shard_owners=shard_owners,
            expected_rows=expected_rows,
            source_prefix=args.source_prefix,
            database_path=args.local_root / "rows.sqlite3",
        )
        tar_worker_mapping: dict[str, Any] | None = None
        tar_mapping_evidence: dict[str, Any] | None = None
        if args.source_mode == "tar_sequential":
            tar_worker_mapping, tar_mapping_evidence = (
                establish_tar_mapping_authority(
                    args,
                    catalog=catalog,
                    shard_owners=shard_owners,
                    expected_rows=expected_rows,
                    partition_plan=partition_plan,
                    partition_plan_raw=partition_plan_raw,
                    source_authority=source_authority,
                )
            )
        writer = LocalPartitionWriter(
            root=args.local_root,
            s3_worker_root=args.s3_worker_root,
            worker_id=args.worker_id,
            workers=args.workers,
            row_base=catalog.gpic_row_base,
            gpic_rows=catalog.gpic_rows,
            expected_rows=expected_rows,
            blob_target_bytes=args.blob_target_bytes,
            max_payload_bytes=args.max_s3_bytes_per_worker,
        )
        source_attempts = 0
        source_errors = 0
        source_elapsed_ns = 0
        tar_telemetry: dict[str, Any] | None = None

        def append_source_results(source_results: Iterator[Any]) -> None:
            nonlocal source_attempts, source_errors, source_elapsed_ns
            assert writer is not None
            for result in source_results:
                writer.append(result)
                source_attempts += result.attempts
                source_errors += result.errors
                source_elapsed_ns += result.elapsed_ns

        if args.source_mode == "tar_sequential":
            assert tar_worker_mapping is not None
            tar_archive_authorities = {
                entry["shard_id"]: entry["source_archive"]
                for entry in tar_worker_mapping["shards"]
            }
            telemetry_path = args.local_root / "tar_telemetry.jsonl"
            telemetry_digest = hashlib.sha256()
            telemetry_records = 0
            telemetry_archive_bytes = 0
            telemetry_selected_bytes = 0
            telemetry_rows = 0
            telemetry_elapsed_ns = 0
            previous_tar_shard = -1
            with telemetry_path.open("xb", buffering=0) as telemetry_handle:

                def record_tar_telemetry(value: dict[str, Any]) -> None:
                    nonlocal telemetry_records
                    nonlocal telemetry_archive_bytes
                    nonlocal telemetry_selected_bytes
                    nonlocal telemetry_rows
                    nonlocal telemetry_elapsed_ns
                    nonlocal previous_tar_shard
                    common.validate_self_hash(
                        value, "tar_telemetry_sha256", "tar telemetry"
                    )
                    tar_shard = value.get("tar_shard")
                    archive_authority = tar_archive_authorities.get(tar_shard)
                    if (
                        value.get("schema_version")
                        != tar_source.TAR_TELEMETRY_SCHEMA
                        or value.get("worker_id") != args.worker_id
                        or value.get("workers") != args.workers
                        or type(tar_shard) is not int
                        or tar_shard <= previous_tar_shard
                        or shard_owners.get(tar_shard) != args.worker_id
                        or not isinstance(archive_authority, dict)
                        or value.get("tar_name")
                        != tar_source.expected_tar_name(tar_shard)
                        or value.get("tar_path")
                        != archive_authority.get("path")
                        or value.get("archive_bytes")
                        != archive_authority.get("bytes")
                        or value.get("source_identity")
                        != archive_authority.get("identity")
                        or value.get("archive_bytes") != value.get(
                            "archive_bytes_read"
                        )
                        or type(value.get("appended_rows")) is not int
                        or value["appended_rows"] <= 0
                        or type(value.get("appended_bytes")) is not int
                        or value["appended_bytes"] <= 0
                        or value.get("attempts") != 1
                        or value.get("errors") != 0
                        or value.get("extra_local_chunk_bytes") != 0
                        or value.get("fsx_writes") != 0
                        or value.get("s3_writes") != 0
                    ):
                        raise common.ProductionError(
                            "tar per-archive telemetry binding drift"
                        )
                    raw = canonical_json_line(value)
                    common.write_all(
                        telemetry_handle, raw, label="tar_telemetry.jsonl"
                    )
                    telemetry_digest.update(raw)
                    telemetry_records += 1
                    telemetry_archive_bytes += value["archive_bytes"]
                    telemetry_selected_bytes += value["appended_bytes"]
                    telemetry_rows += value["appended_rows"]
                    telemetry_elapsed_ns += value["elapsed_ns"]
                    previous_tar_shard = tar_shard

                append_source_results(
                    tar_source.iter_tar_sequential_reads(
                        iter_sorted_items(args.local_root / "rows.sqlite3"),
                        worker_mapping=tar_worker_mapping,
                        worker_id=args.worker_id,
                        workers=args.workers,
                        shard_owners=shard_owners,
                        source_prefix=args.source_prefix,
                        tar_root=args.tar_root,
                        row_base=catalog.gpic_row_base,
                        gpic_rows=catalog.gpic_rows,
                        expected_rows=expected_rows,
                        attempts=args.source_attempts,
                        tar_workers=args.tar_workers,
                        max_pending_tars=args.tar_workers,
                        read_buffer_bytes=common.PRODUCTION_TAR_READ_BUFFER_BYTES,
                        io_block_bytes=common.PRODUCTION_TAR_IO_BLOCK_BYTES,
                        max_selected_bytes=args.max_s3_bytes_per_worker,
                        max_archive_bytes=(
                            args.max_source_archive_bytes_per_worker
                        ),
                        max_selected_bytes_per_tar=(
                            source_authority["max_selected_bytes_per_tar"]
                        ),
                        telemetry_callback=record_tar_telemetry,
                    )
                )
                telemetry_handle.flush()
                os.fsync(telemetry_handle.fileno())
            telemetry_binding = common.file_binding(telemetry_path)
            if (
                telemetry_binding["sha256"] != telemetry_digest.hexdigest()
                or telemetry_records != tar_worker_mapping["shard_count"]
                or telemetry_rows != expected_rows
                or telemetry_selected_bytes != writer.total_bytes
                or telemetry_archive_bytes
                != tar_worker_mapping["source_archive_bytes"]
                or telemetry_archive_bytes
                > args.max_source_archive_bytes_per_worker
                or source_attempts != expected_rows
                or source_errors != 0
            ):
                raise common.ProductionError("tar telemetry/source closure drift")
            tar_telemetry = {
                "file": telemetry_binding,
                "records": telemetry_records,
                "archive_bytes": telemetry_archive_bytes,
                "selected_bytes": telemetry_selected_bytes,
                "rows": telemetry_rows,
                "elapsed_ns_sum": telemetry_elapsed_ns,
                "tar_workers": args.tar_workers,
                "max_pending_tars": args.tar_workers,
                "read_buffer_bytes": common.PRODUCTION_TAR_READ_BUFFER_BYTES,
                "io_block_bytes": common.PRODUCTION_TAR_IO_BLOCK_BYTES,
                "max_selected_bytes_per_tar": (
                    source_authority["max_selected_bytes_per_tar"]
                ),
                "max_source_archive_bytes_per_worker": (
                    args.max_source_archive_bytes_per_worker
                ),
                "extra_local_chunk_bytes": 0,
                "attempts": 1,
                "errors": 0,
            }
        else:
            append_source_results(
                ordered_source_reads(
                    iter_sorted_items(args.local_root / "rows.sqlite3"),
                    threads=args.source_threads,
                    attempts=args.source_attempts,
                    block_bytes=args.source_block_bytes,
                    retry_base_seconds=args.retry_base_seconds,
                )
            )
        blobs, order_binding = writer.close()
        writer = None
        row_binding = validate_local_fragments(
            database_path=args.local_root / "rows.sqlite3",
            row_dense_work_path=args.local_root / "row_dense_work.bin",
            row_fragment_path=args.local_root / "row_fragment.bin",
            blob_order_path=args.local_root / "blob_order.bin",
            blobs=blobs,
            row_base=catalog.gpic_row_base,
            worker_id=args.worker_id,
            workers=args.workers,
            expected_rows=expected_rows,
        )
        local_manifest = {
            "schema_version": common.LOCAL_MANIFEST_SCHEMA,
            "generated_at": common.utc_now(),
            "build_jobtag": args.build_jobtag,
            "worker_id": args.worker_id,
            "workers": args.workers,
            "rows": expected_rows,
            "payload_bytes": sum(value["bytes"] for value in blobs),
            "blobs": blobs,
            "row_fragment": row_binding,
            "blob_order": order_binding,
            "materialization_database": database_binding,
            "source_attempts": source_attempts,
            "source_errors": source_errors,
            "source_elapsed_ns_sum": source_elapsed_ns,
            "tar_telemetry": tar_telemetry,
            "tar_mapping_evidence": tar_mapping_evidence,
            "source_order": (
                "tar_shard_ascending_then_archive_member_order"
                if args.source_mode == "tar_sequential"
                else "image_path_binary_then_row_id"
            ),
            "source_mode": args.source_mode,
            "dense_slot_order": "explicit_row_id_sorted_within_shard_partition",
            "partition_mode": args.partition_mode,
            "partition_plan_raw_sha256": hashlib.sha256(
                partition_plan_raw
            ).hexdigest(),
            "partition_plan_sha256": partition_plan["partition_plan_sha256"],
            "record_struct": common.RECORD_STRUCT.format,
            "record_bytes": common.RECORD_STRUCT.size,
            "code": code,
            "catalog": catalog_value,
            **common.build_init_binding(build_init, build_init_raw),
        }
        local_manifest["local_manifest_sha256"] = common.self_hash(
            local_manifest, "local_manifest_sha256"
        )
        local_manifest_raw = common.canonical_json_bytes(local_manifest)
        with (args.local_root / "local_manifest.json").open(
            "xb", buffering=0
        ) as handle:
            common.write_all(handle, local_manifest_raw, label="local_manifest.json")
            handle.flush()
            os.fsync(handle.fileno())
        pilot_core.fsync_dir(args.local_root)
        local_bytes = common.fsx_tree_bytes([args.local_root])
        if local_bytes > args.max_local_bytes_per_worker:
            raise common.ProductionError("complete local partition hard cap exceeded")
        build_elapsed = time.monotonic() - build_started
        preupload, preupload_raw = publish_preupload(
            args,
            code=code,
            catalog_manifest=catalog_value,
            topology_evidence=topology_evidence,
            local_topology=local_topology,
            runtime=runtime,
            build_init=build_init,
            build_init_raw=build_init_raw,
            partition_plan=partition_plan,
            partition_plan_raw=partition_plan_raw,
            local_manifest=local_manifest,
            local_manifest_raw=local_manifest_raw,
        )
        # This global approval is the linear gate: no blob upload occurs above.
        approval, approval_raw = approve_uploads(
            args,
            code=code,
            catalog_manifest=catalog_value,
            topology_evidence=topology_evidence,
            build_init=build_init,
            build_init_raw=build_init_raw,
            partition_plan=partition_plan,
            partition_plan_raw=partition_plan_raw,
        )
        common.assert_no_root_success(args.s3_root)
        upload_started = time.monotonic()
        for blob in local_manifest["blobs"]:
            common.upload_file_once(
                Path(blob["local_path"]),
                args.s3_worker_root / blob["name"],
                expected_bytes=blob["bytes"],
                expected_sha256=blob["sha256"],
                block_bytes=args.s3_block_bytes,
            )
        upload_elapsed = time.monotonic() - upload_started
        common.validate_catalog_closure(catalog)
        common.validate_code_snapshot(code)
        receipt, receipt_raw = publish_worker_receipt(
            args,
            code=code,
            catalog_manifest=catalog_value,
            topology_evidence=topology_evidence,
            local_topology=local_topology,
            runtime=runtime,
            build_init=build_init,
            build_init_raw=build_init_raw,
            partition_plan=partition_plan,
            partition_plan_raw=partition_plan_raw,
            local_manifest=local_manifest,
            local_manifest_raw=local_manifest_raw,
            preupload=preupload,
            preupload_raw=preupload_raw,
            approval=approval,
            approval_raw=approval_raw,
            build_elapsed_seconds=build_elapsed,
            upload_elapsed_seconds=upload_elapsed,
        )
        uploaded = {
            "schema_version": common.PRODUCTION_SCHEMA,
            "build_jobtag": args.build_jobtag,
            "worker_id": args.worker_id,
            "receipt_raw_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "receipt_sha256": receipt["receipt_sha256"],
            "rows": receipt["rows"],
            "payload_bytes": receipt["payload_bytes"],
            **common.build_init_binding(build_init, build_init_raw),
            "partition_plan_sha256": partition_plan["partition_plan_sha256"],
        }
        common.write_fsx_json_once(
            args.coord_root / f"uploaded.worker-{args.worker_id}.json", uploaded
        )
        pilot_core.fsync_dir(args.coord_root)
        common.assert_no_root_success(args.s3_root)
        if args.cleanup_local_after_receipt:
            common.remove_tree_after_commit(args.local_root)
        print(
            "OFFLINE_GPIC_S3_PRODUCTION_WORKER_PASS "
            f"build_jobtag={args.build_jobtag} worker={args.worker_id} "
            f"rows={receipt['rows']} bytes={receipt['payload_bytes']}",
            flush=True,
        )
    except BaseException as exc:
        if writer is not None:
            writer.abort()
        marker = args.fsx_worker_root / "_SUCCESS"
        if isinstance(exc, pilot_core.CommittedSyncUnconfirmed) or common.success_or_unknown(
            marker
        ):
            print(
                "OFFLINE_GPIC_S3_PRODUCTION_COMMITTED_SYNC_UNCONFIRMED "
                f"worker={args.worker_id}",
                file=sys.stderr,
                flush=True,
            )
        else:
            try:
                common.write_failure(
                    args.coord_root,
                    build_jobtag=args.build_jobtag,
                    actor="builder",
                    actor_id=args.worker_id,
                    exc=exc,
                )
            except BaseException:
                pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--artifact-prefix", type=Path, required=True)
    parser.add_argument("--coord-prefix", type=Path, required=True)
    parser.add_argument("--local-prefix", type=Path, required=True)
    parser.add_argument("--s3-prefix", type=Path, required=True)
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
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--workers", type=int, default=common.PRODUCTION_WORKERS)
    parser.add_argument(
        "--partition-mode",
        choices=common.PRODUCTION_PARTITION_MODES,
        required=True,
    )
    parser.add_argument("--source-prefix", default=common.PRODUCTION_SOURCE_PREFIX)
    parser.add_argument(
        "--source-mode", choices=common.PRODUCTION_SOURCE_MODES, required=True
    )
    parser.add_argument("--tar-mapping-manifest", type=Path)
    parser.add_argument("--tar-mapping-raw-sha256")
    parser.add_argument("--tar-quick-summary", type=Path)
    parser.add_argument("--tar-quick-raw-sha256")
    parser.add_argument("--tar-parser-gate-summary", type=Path)
    parser.add_argument("--tar-parser-gate-raw-sha256")
    parser.add_argument("--tar-root", type=Path)
    parser.add_argument(
        "--tar-workers", type=int, default=common.PRODUCTION_TAR_WORKERS
    )
    parser.add_argument("--source-thread-authority", type=Path)
    parser.add_argument("--source-thread-authority-raw-sha256")
    parser.add_argument(
        "--source-threads", type=int, default=common.PRODUCTION_SOURCE_THREADS
    )
    parser.add_argument(
        "--source-attempts", type=int, default=common.PRODUCTION_SOURCE_ATTEMPTS
    )
    parser.add_argument(
        "--source-block-bytes",
        type=int,
        default=common.PRODUCTION_SOURCE_BLOCK_BYTES,
    )
    parser.add_argument("--retry-base-seconds", type=float, default=0.25)
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
    parser.add_argument(
        "--s3-block-bytes", type=int, default=common.PRODUCTION_S3_BLOCK_BYTES
    )
    parser.add_argument("--barrier-timeout-seconds", type=float, default=21_600.0)
    parser.add_argument("--write-authority", required=True)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--cleanup-local-after-receipt", action="store_true")
    parser.add_argument("--production-gate", action="store_true")
    args = parser.parse_args(argv)
    for label in (
        "catalog_root",
        "artifact_prefix",
        "coord_prefix",
        "local_prefix",
        "s3_prefix",
        "topology_summary",
        "job_spec",
    ):
        setattr(args, label, getattr(args, label).expanduser().resolve(strict=True))
    if args.tar_mapping_manifest is not None:
        args.tar_mapping_manifest = (
            args.tar_mapping_manifest.expanduser().resolve(strict=True)
        )
    if args.tar_quick_summary is not None:
        args.tar_quick_summary = args.tar_quick_summary.expanduser().resolve(
            strict=True
        )
    if args.tar_parser_gate_summary is not None:
        args.tar_parser_gate_summary = (
            args.tar_parser_gate_summary.expanduser().resolve(strict=True)
        )
    if args.tar_root is not None:
        args.tar_root = args.tar_root.expanduser().resolve(strict=True)
    if args.source_thread_authority is not None:
        args.source_thread_authority = (
            args.source_thread_authority.expanduser().resolve(strict=True)
        )
        if not common.is_sha256(args.source_thread_authority_raw_sha256):
            raise common.ProductionError(
                "source-thread authority raw SHA is malformed"
            )
    elif args.source_thread_authority_raw_sha256 is not None:
        raise common.ProductionError(
            "source-thread authority SHA was supplied without a file"
        )
    if args.write_authority != common.WRITE_AUTHORITY:
        raise common.ProductionError("production write authority is absent")
    if (
        not args.build_jobtag
        or Path(args.build_jobtag).name != args.build_jobtag
        or args.build_jobtag in {".", "..", "latest"}
    ):
        raise common.ProductionError("build jobtag is malformed/non-exact")
    if not (0 <= args.worker_id < args.workers) or args.workers < 1:
        raise common.ProductionError("worker identity is malformed")
    if args.source_threads < 1:
        raise common.ProductionError("source threads must be positive")
    if not (1.0 <= args.max_byte_imbalance <= 1.10):
        raise common.ProductionError("byte imbalance cap must be in [1.0,1.10]")
    for label in (
        "source_attempts",
        "source_block_bytes",
        "blob_target_bytes",
        "max_local_bytes_per_worker",
        "max_source_archive_bytes_per_worker",
        "max_s3_bytes_per_worker",
        "max_s3_payload_bytes",
        "max_s3_total_bytes",
        "s3_controls_reserve_bytes",
        "max_fsx_incremental_bytes",
        "s3_block_bytes",
    ):
        if getattr(args, label) < 1:
            raise common.ProductionError(f"{label} must be positive")
    if args.retry_base_seconds < 0 or args.barrier_timeout_seconds <= 0:
        raise common.ProductionError("retry/barrier interval is malformed")
    if not args.source_prefix.startswith("/") or not args.source_prefix.endswith("/"):
        raise common.ProductionError("source prefix must be an absolute directory prefix")
    if not str(args.artifact_prefix).startswith("/fsx/") or not str(
        args.coord_prefix
    ).startswith("/fsx/"):
        raise common.ProductionError("production FSx prefixes escaped /fsx")
    if args.artifact_prefix == args.coord_prefix:
        raise common.ProductionError("artifact/coord prefixes must differ")
    if args.production_gate:
        if (
            args.workers != common.PRODUCTION_WORKERS
            or args.partition_mode != "shard_modulo"
            or args.source_mode != "tar_sequential"
            or args.source_threads != 1
            or args.tar_workers not in common.PRODUCTION_ALLOWED_TAR_WORKERS
            or args.source_attempts != common.PRODUCTION_SOURCE_ATTEMPTS
            or args.retry_base_seconds != 0.0
            or args.source_block_bytes != common.PRODUCTION_SOURCE_BLOCK_BYTES
            or args.source_prefix != common.PRODUCTION_SOURCE_PREFIX
            or args.blob_target_bytes != common.PRODUCTION_BLOB_TARGET_BYTES
            or args.max_local_bytes_per_worker
            != common.PRODUCTION_MAX_LOCAL_BYTES_PER_WORKER
            or args.max_source_archive_bytes_per_worker
            != common.PRODUCTION_MAX_SOURCE_ARCHIVE_BYTES_PER_WORKER
            or args.max_s3_bytes_per_worker
            != common.PRODUCTION_MAX_S3_BYTES_PER_WORKER
            or args.max_s3_payload_bytes
            != common.PRODUCTION_MAX_S3_PAYLOAD_BYTES
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
            or args.tar_mapping_manifest != common.PRODUCTION_TAR_MAPPING_REPORT
            or args.tar_mapping_raw_sha256
            != common.PRODUCTION_TAR_MAPPING_RAW_SHA256
            or args.tar_quick_summary != common.PRODUCTION_TAR_QUICK_SUMMARY
            or args.tar_quick_raw_sha256
            != common.PRODUCTION_TAR_QUICK_RAW_SHA256
            or args.tar_parser_gate_summary
            != common.PRODUCTION_TAR_PARSER_GATE_SUMMARY
            or args.tar_parser_gate_raw_sha256
            != common.PRODUCTION_TAR_PARSER_GATE_RAW_SHA256
            or args.tar_root != common.PRODUCTION_TAR_ROOT
            or args.source_thread_authority is not None
            or not str(args.local_prefix).startswith("/local/evan/")
            or not str(args.s3_prefix).startswith(
                "/s3/evan/artifacts/offline_gpic_blob_v1"
            )
        ):
            raise common.ProductionError("frozen production builder constants drift")
    args.artifact_root = args.artifact_prefix / args.build_jobtag
    args.fsx_workers_root = args.artifact_root / "workers"
    args.fsx_worker_root = args.fsx_workers_root / f"worker-{args.worker_id}"
    args.coord_root = args.coord_prefix / args.build_jobtag
    args.local_root = (
        args.local_prefix / args.build_jobtag / f"worker-{args.worker_id}"
    )
    args.s3_root = args.s3_prefix / args.build_jobtag
    args.s3_controls_root = args.s3_root / "controls"
    args.s3_workers_root = args.s3_root / "workers"
    args.s3_worker_root = args.s3_workers_root / f"worker-{args.worker_id}"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
