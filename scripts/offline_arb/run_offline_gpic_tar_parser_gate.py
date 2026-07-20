#!/usr/bin/env python3
"""Eight-node read-only real-data gate for ``offline_gpic_tar_sequential``.

Every physical node owns eight disjoint real tar shards and replays the same
selection with 1, 4, and 8 bounded tar readers.  The arm order is rotated by
rank so the shared source sees a constant mixed load.  Only compact canonical
evidence is written, under one fresh FSx jobtag namespace.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import os
import platform
import re
import resource
import struct
import time
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.offline_arb import offline_gpic_s3_production_common as common
from scripts.offline_arb import offline_gpic_tar_sequential as tar_source
from scripts.offline_arb import run_offline_gpic_blob_pilot as pilot_core


SCHEMA_VERSION = "offline_gpic_tar_parser_gate_v1"
RECEIPT_SCHEMA = "offline_gpic_tar_parser_gate_worker_v1"
SUMMARY_SCHEMA = "offline_gpic_tar_parser_gate_summary_v1"
INIT_SCHEMA = "offline_gpic_tar_parser_gate_init_v1"
READ_AUTHORITY = "offline-gpic-tar-parser-gate-read-only-v1"
PRODUCTION_WORKERS = 8
TARS_PER_WORKER = 8
ARM_WORKERS = (1, 4, 8)
SOURCE_PREFIX = tar_source.PRODUCTION_SOURCE_PREFIX
TAR_ROOT = tar_source.PRODUCTION_TAR_ROOT
GPIC_ROWS = 20_000_000
FULL_ARCHIVE_BYTES_ESTIMATE = 12_796_000_000_000
MIN_PROJECTED_AGGREGATE_BPS = 1_000_000_000.0
MAX_PROJECTED_SCAN_HOURS = 4.0
MAX_PEAK_RSS_BYTES = 32 * (1 << 30)
MAX_SELECTED_BYTES = 16 * (1 << 30)
MAX_SELECTED_BYTES_PER_TAR = 1_000_000_000
SEMANTIC_PREFIX = struct.Struct("<qqII")


class GateError(RuntimeError):
    """A parser-gate contract failed."""


@dataclasses.dataclass(frozen=True)
class Item:
    local_index: int
    row_id: int
    image_path: str


def utc_now() -> str:
    return common.utc_now()


def selected_shards(worker_id: int) -> tuple[int, ...]:
    if type(worker_id) is not int or not (0 <= worker_id < PRODUCTION_WORKERS):
        raise GateError("invalid parser-gate worker ID")
    base = worker_id * 1_000
    values = tuple(base + 73 + (113 * index) for index in range(TARS_PER_WORKER))
    if values[-1] >= (worker_id + 1) * 1_000 or values[-1] >= 8_000:
        raise GateError("deterministic tar selection escaped worker stratum")
    return values


def selection_plan(jobtag: str) -> dict[str, Any]:
    value = {
        "schema_version": "offline_gpic_tar_parser_gate_selection_v1",
        "jobtag": jobtag,
        "workers": PRODUCTION_WORKERS,
        "tars_per_worker": TARS_PER_WORKER,
        "source_prefix": SOURCE_PREFIX,
        "tar_root": str(TAR_ROOT),
        "shards_by_worker": [
            {
                "worker_id": rank,
                "shards": list(selected_shards(rank)),
                "arm_order": list(rotated_arm_order(rank)),
            }
            for rank in range(PRODUCTION_WORKERS)
        ],
        "ownership": "64 deterministic tar shards; disjoint by 1000-shard stratum",
    }
    value["selection_plan_sha256"] = common.self_hash(
        value, "selection_plan_sha256"
    )
    return value


def rotated_arm_order(worker_id: int) -> tuple[int, ...]:
    offset = worker_id % len(ARM_WORKERS)
    return ARM_WORKERS[offset:] + ARM_WORKERS[:offset]


def sha256_file(path: Path) -> str:
    return common.sha256_file(path)


def code_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    files = {
        "runner": Path(__file__).resolve(strict=True),
        "tar_source": Path(tar_source.__file__).resolve(strict=True),
        "common": Path(common.__file__).resolve(strict=True),
        "pilot_core": Path(pilot_core.__file__).resolve(strict=True),
        "job_spec": args.job_spec.resolve(strict=True),
    }
    value = {
        "files": {
            name: {
                "path": str(path),
                "bytes": common.require_regular_file(path).st_size,
                "sha256": sha256_file(path),
            }
            for name, path in sorted(files.items())
        },
        "python": platform.python_version(),
        "pyarrow": pa.__version__,
    }
    if (
        value["files"]["runner"]["sha256"] != args.expected_runner_sha256
        or value["files"]["tar_source"]["sha256"]
        != args.expected_tar_source_sha256
    ):
        raise GateError("frozen runner/tar-source SHA drift")
    return value


def validate_code_snapshot(value: dict[str, Any]) -> None:
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise GateError("code snapshot is malformed")
    for name, binding in files.items():
        if (
            not isinstance(binding, dict)
            or not common.is_sha256(binding.get("sha256"))
            or sha256_file(Path(binding.get("path", ""))) != binding["sha256"]
        ):
            raise GateError(f"code changed during parser gate: {name}")


def runtime_identity(worker_id: int) -> dict[str, Any]:
    boot_raw = Path("/proc/sys/kernel/random/boot_id").read_bytes()
    return {
        "host": platform.node(),
        "worker_id": worker_id,
        "boot_id_sha256": hashlib.sha256(boot_raw).hexdigest(),
        "pid": os.getpid(),
        "kernel": platform.release(),
    }


def wait_for(path: Path, *, timeout_seconds: float, label: str) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file() and not path.is_symlink():
            return
        time.sleep(0.2)
    raise GateError(f"timeout waiting for {label}: {path}")


def barrier(
    root: Path,
    *,
    label: str,
    worker_id: int,
    workers: int,
    timeout_seconds: float,
) -> None:
    barrier_root = root / "barriers" / label
    barrier_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    common.write_fsx_bytes_once(
        barrier_root / f"ready-{worker_id}", b"ready\n"
    )
    go = barrier_root / "GO"
    if worker_id == 0:
        deadline = time.monotonic() + timeout_seconds
        expected = {f"ready-{rank}" for rank in range(workers)}
        while time.monotonic() < deadline:
            observed = {
                path.name
                for path in barrier_root.iterdir()
                if path.is_file() and not path.is_symlink() and path.name.startswith("ready-")
            }
            if observed == expected:
                common.write_fsx_bytes_once(go, b"go\n")
                break
            time.sleep(0.2)
        else:
            raise GateError(f"barrier participant timeout: {label}")
    wait_for(go, timeout_seconds=timeout_seconds, label=f"barrier {label}")
    if go.read_bytes() != b"go\n":
        raise GateError(f"barrier marker drift: {label}")


def initialize_output(
    args: argparse.Namespace,
    *,
    code: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    root = args.output_prefix / args.jobtag
    init_path = root / "init.json"
    if args.worker_id == 0:
        if root.exists() or root.is_symlink():
            raise GateError("parser-gate output namespace is not fresh")
        root.mkdir(mode=0o700)
        (root / "workers").mkdir(mode=0o700)
        (root / "barriers").mkdir(mode=0o700)
        init = {
            "schema_version": INIT_SCHEMA,
            "generated_at": utc_now(),
            "jobtag": args.jobtag,
            "workers": args.workers,
            "tars_per_worker": args.tars_per_worker,
            "arm_workers": list(ARM_WORKERS),
            "code": code,
            "selection_plan": plan,
            "sample_report": {
                "path": str(args.sample_report),
                "raw_sha256": args.sample_report_raw_sha256,
            },
            "read_authority": args.read_authority,
            "writes": {
                "fsx_root": str(root),
                "source_writes": 0,
                "s3_writes": 0,
            },
        }
        init["init_sha256"] = common.self_hash(init, "init_sha256")
        common.write_fsx_json_once(init_path, init)
    wait_for(init_path, timeout_seconds=args.barrier_timeout_seconds, label="init")
    init, _ = common.strict_json(init_path)
    common.validate_self_hash(init, "init_sha256", "parser gate init")
    if (
        init.get("schema_version") != INIT_SCHEMA
        or init.get("jobtag") != args.jobtag
        or init.get("workers") != args.workers
        or init.get("tars_per_worker") != args.tars_per_worker
        or init.get("arm_workers") != list(ARM_WORKERS)
        or init.get("code") != code
        or init.get("selection_plan") != plan
        or init.get("read_authority") != args.read_authority
        or init.get("writes", {}).get("fsx_root") != str(root)
        or init.get("writes", {}).get("source_writes") != 0
        or init.get("writes", {}).get("s3_writes") != 0
    ):
        raise GateError("parser-gate init binding drift")
    return root, init


def select_catalog_items(
    catalog: pilot_core.CatalogBinding,
    *,
    shards: Sequence[int],
) -> tuple[list[Item], dict[str, Any]]:
    prefixes = [f"{SOURCE_PREFIX}{shard:05d}/" for shard in shards]
    pattern = "^(?:" + "|".join(re.escape(value) for value in prefixes) + ")"
    parquet = pq.ParquetFile(catalog.gpic_path)
    items: list[Item] = []
    counts = {shard: 0 for shard in shards}
    group_start = 0
    started = time.perf_counter()
    for group_index in range(parquet.num_row_groups):
        table = parquet.read_row_group(
            group_index,
            columns=["row_id", "image_path"],
            use_threads=False,
        ).combine_chunks()
        paths = table.column("image_path")
        row_ids = table.column("row_id")
        mask = pc.match_substring_regex(paths, pattern=pattern)
        offsets = pc.indices_nonzero(mask).to_pylist()
        for raw_offset in offsets:
            offset = int(raw_offset)
            local_index = group_start + offset
            row_id = row_ids[offset].as_py()
            image_path = paths[offset].as_py()
            if type(row_id) is not int or row_id != catalog.gpic_row_base + local_index:
                raise GateError("selected catalog row identity drift")
            shard, _, _ = tar_source.canonical_catalog_target(
                image_path, source_prefix=SOURCE_PREFIX
            )
            if shard not in counts:
                raise GateError("catalog regex selected a foreign shard")
            counts[shard] += 1
            items.append(Item(local_index, row_id, image_path))
        group_start += table.num_rows
    if (
        group_start != catalog.gpic_rows
        or any(count < 1 for count in counts.values())
        or sum(counts.values()) != len(items)
    ):
        raise GateError("selected catalog shard/row closure failed")
    items.sort(key=lambda item: item.image_path.encode("ascii"))
    digest = hashlib.sha256()
    for item in items:
        raw = item.image_path.encode("ascii")
        digest.update(struct.pack("<qqI", item.local_index, item.row_id, len(raw)))
        digest.update(raw)
    return items, {
        "rows": len(items),
        "counts_by_shard": [
            {"shard_id": shard, "rows": counts[shard]} for shard in shards
        ],
        "selection_sha256": digest.hexdigest(),
        "catalog_rows_scanned": group_start,
        "catalog_row_groups_scanned": parquet.num_row_groups,
        "elapsed_seconds": time.perf_counter() - started,
    }


def semantic_digest_update(digest: Any, result: tar_source.TarSourceRead) -> None:
    member = Path(result.item.image_path).name.encode("ascii")
    digest.update(
        SEMANTIC_PREFIX.pack(
            result.item.local_index,
            result.item.row_id,
            len(member),
            len(result.payload),
        )
    )
    digest.update(member)
    digest.update(result.sha256_raw)


def run_arm(
    *,
    items: Sequence[Item],
    worker_mapping: dict[str, Any],
    worker_id: int,
    workers: int,
    shard_owners: dict[int, int],
    catalog: pilot_core.CatalogBinding,
    tar_root: Path,
    tar_workers: int,
    max_archive_bytes: int,
) -> dict[str, Any]:
    telemetry: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    seen_rows: set[int] = set()
    rows = 0
    selected_bytes = 0
    attempts = 0
    errors = 0
    first_keys: list[str] = []
    last_keys: list[str] = []
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    started_ns = time.perf_counter_ns()
    source = tar_source.iter_tar_sequential_reads(
        iter(items),
        worker_mapping=worker_mapping,
        worker_id=worker_id,
        workers=workers,
        shard_owners=shard_owners,
        source_prefix=SOURCE_PREFIX,
        tar_root=tar_root,
        row_base=catalog.gpic_row_base,
        gpic_rows=catalog.gpic_rows,
        expected_rows=len(items),
        attempts=1,
        tar_workers=tar_workers,
        max_pending_tars=tar_workers,
        read_buffer_bytes=tar_source.DEFAULT_READ_BUFFER_BYTES,
        io_block_bytes=tar_source.DEFAULT_IO_BLOCK_BYTES,
        max_selected_bytes=MAX_SELECTED_BYTES,
        max_archive_bytes=max_archive_bytes,
        max_selected_bytes_per_tar=MAX_SELECTED_BYTES_PER_TAR,
        telemetry_callback=telemetry.append,
    )
    for result in source:
        if (
            result.item.row_id in seen_rows
            or result.attempts != 1
            or result.errors != 0
            or len(result.payload) < 1
            or result.sha256_raw != hashlib.sha256(result.payload).digest()
        ):
            raise GateError("parser arm SourceRead integrity/uniqueness drift")
        seen_rows.add(result.item.row_id)
        semantic_digest_update(digest, result)
        key = f"{result.item.row_id}:{Path(result.item.image_path).name}"
        if len(first_keys) < 8:
            first_keys.append(key)
        last_keys.append(key)
        if len(last_keys) > 8:
            del last_keys[0]
        rows += 1
        selected_bytes += len(result.payload)
        attempts += result.attempts
        errors += result.errors
    del source
    gc.collect()
    elapsed_ns = time.perf_counter_ns() - started_ns
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    expected_shards = [entry["shard_id"] for entry in worker_mapping["shards"]]
    if (
        rows != len(items)
        or attempts != rows
        or errors != 0
        or len(seen_rows) != rows
        or [value.get("tar_shard") for value in telemetry] != expected_shards
        or sum(value.get("archive_bytes", -1) for value in telemetry)
        != max_archive_bytes
        or sum(value.get("archive_bytes_read", -1) for value in telemetry)
        != max_archive_bytes
        or sum(value.get("appended_rows", -1) for value in telemetry) != rows
        or sum(value.get("appended_bytes", -1) for value in telemetry)
        != selected_bytes
        or any(
            value.get("tar_telemetry_sha256")
            != common.self_hash(value, "tar_telemetry_sha256")
            for value in telemetry
        )
    ):
        raise GateError("parser arm rows/bytes/tar telemetry closure failed")
    return {
        "tar_workers": tar_workers,
        "rows": rows,
        "selected_bytes": selected_bytes,
        "archive_bytes": max_archive_bytes,
        "tar_count": len(telemetry),
        "attempts": attempts,
        "errors": errors,
        "semantic_sha256": digest.hexdigest(),
        "first_keys": first_keys,
        "last_keys": last_keys,
        "elapsed_ns": elapsed_ns,
        "archive_bytes_per_second": max_archive_bytes / (elapsed_ns / 1e9),
        "selected_rows_per_second": rows / (elapsed_ns / 1e9),
        "selected_bytes_per_second": selected_bytes / (elapsed_ns / 1e9),
        "peak_rss_bytes_before": rss_before,
        "peak_rss_bytes_after": rss_after,
        "per_tar_telemetry_sha256": hashlib.sha256(
            common.canonical_json_bytes(telemetry)
        ).hexdigest(),
        "per_tar": telemetry,
    }


def publish_summary(
    args: argparse.Namespace,
    *,
    root: Path,
    init: dict[str, Any],
    code: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    receipts: list[dict[str, Any]] = []
    for rank in range(args.workers):
        value, _ = common.strict_json(root / "workers" / f"worker-{rank}.json")
        common.validate_self_hash(value, "receipt_sha256", f"parser receipt {rank}")
        if (
            value.get("schema_version") != RECEIPT_SCHEMA
            or value.get("status") != "PASS"
            or value.get("jobtag") != args.jobtag
            or value.get("worker_id") != rank
            or value.get("workers") != args.workers
            or value.get("code") != code
            or value.get("selection_plan_sha256")
            != plan["selection_plan_sha256"]
            or value.get("init_sha256") != init["init_sha256"]
        ):
            raise GateError(f"parser worker receipt binding drift rank={rank}")
        receipts.append(value)
    boots = [value["runtime"]["boot_id_sha256"] for value in receipts]
    shard_sets = [set(value["shards"]) for value in receipts]
    shard_union = set().union(*shard_sets)
    disjoint = sum(len(value) for value in shard_sets) == len(shard_union)
    semantic_equal_per_worker = all(
        len({arm["semantic_sha256"] for arm in receipt["arms"]}) == 1
        and len({arm["rows"] for arm in receipt["arms"]}) == 1
        and len({arm["selected_bytes"] for arm in receipt["arms"]}) == 1
        and {arm["tar_workers"] for arm in receipt["arms"]} == set(ARM_WORKERS)
        for receipt in receipts
    )
    aggregate: list[dict[str, Any]] = []
    for tar_workers in ARM_WORKERS:
        arms = [
            next(
                arm for arm in receipt["arms"] if arm["tar_workers"] == tar_workers
            )
            for receipt in receipts
        ]
        rates = [arm["archive_bytes_per_second"] for arm in arms]
        projected = sum(rates)
        aggregate.append(
            {
                "tar_workers": tar_workers,
                "projected_8node_archive_bytes_per_second": projected,
                "slowest_node_archive_bytes_per_second": min(rates),
                "fastest_node_archive_bytes_per_second": max(rates),
                "estimated_full_archive_scan_hours": (
                    FULL_ARCHIVE_BYTES_ESTIMATE / projected / 3600
                ),
                "max_peak_rss_bytes": max(
                    arm["peak_rss_bytes_after"] for arm in arms
                ),
                "rows": sum(arm["rows"] for arm in arms),
                "selected_bytes": sum(arm["selected_bytes"] for arm in arms),
                "archive_bytes": sum(arm["archive_bytes"] for arm in arms),
            }
        )
    best_rate = max(
        value["projected_8node_archive_bytes_per_second"] for value in aggregate
    )
    eligible = [
        value
        for value in aggregate
        if value["projected_8node_archive_bytes_per_second"] >= 0.95 * best_rate
    ]
    recommendation = min(eligible, key=lambda value: value["tar_workers"])
    closure = (
        len(set(boots)) == args.workers
        and disjoint
        and len(shard_union) == args.workers * args.tars_per_worker
        and semantic_equal_per_worker
        and all(
            value["shards"] == list(selected_shards(value["worker_id"]))
            for value in receipts
        )
        and all(
            value["peak_rss_bytes"] <= MAX_PEAK_RSS_BYTES for value in receipts
        )
    )
    performance = (
        recommendation["projected_8node_archive_bytes_per_second"]
        >= MIN_PROJECTED_AGGREGATE_BPS
        and recommendation["estimated_full_archive_scan_hours"]
        <= MAX_PROJECTED_SCAN_HOURS
        and recommendation["max_peak_rss_bytes"] <= MAX_PEAK_RSS_BYTES
    )
    status = "PASS" if closure and performance else "FAIL"
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "generated_at": utc_now(),
        "status": status,
        "recommendation": (
            "PASS_FOR_BOUNDED_PRODUCTION_INTEGRATION"
            if status == "PASS"
            else "BLOCK_PRODUCTION_TAR_ADAPTER"
        ),
        "recommended_tar_workers": recommendation["tar_workers"],
        "jobtag": args.jobtag,
        "workers": args.workers,
        "distinct_physical_boots": len(set(boots)),
        "tars_per_worker": args.tars_per_worker,
        "unique_tar_shards": len(shard_union),
        "shards_disjoint": disjoint,
        "semantic_digest_equal_across_1_4_8_per_worker": semantic_equal_per_worker,
        "aggregate": aggregate,
        "thresholds": {
            "min_projected_aggregate_bytes_per_second": MIN_PROJECTED_AGGREGATE_BPS,
            "max_projected_scan_hours": MAX_PROJECTED_SCAN_HOURS,
            "max_peak_rss_bytes": MAX_PEAK_RSS_BYTES,
            "rate_tie_fraction_choose_lower_workers": 0.95,
        },
        "measurement_semantics": (
            "each node replays the same 8 tars; arm order rotates by rank; "
            "projected aggregate is the sum of eight per-node rates under mixed "
            "simultaneous source load"
        ),
        "production_build_submission": "BLOCKED_BY_THIS_GATE_AND_SEPARATE_FULL_AUTHORITY_GATES",
        "selection_plan": plan,
        "init_sha256": init["init_sha256"],
        "code": code,
        "receipts": receipts,
    }
    summary["summary_sha256"] = common.self_hash(summary, "summary_sha256")
    raw = common.write_fsx_json_once(root / "summary.json", summary)
    raw_sha = hashlib.sha256(raw).hexdigest()
    common.write_fsx_bytes_once(
        root / "_COMPLETE", (raw_sha + "\n").encode("ascii")
    )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    code = code_snapshot(args)
    plan = selection_plan(args.jobtag)
    root, init = initialize_output(args, code=code, plan=plan)
    barrier(
        root,
        label="initialized",
        worker_id=args.worker_id,
        workers=args.workers,
        timeout_seconds=args.barrier_timeout_seconds,
    )
    catalog = pilot_core.load_catalog(args.catalog_root, verify_gpic_sha256=False)
    if catalog.gpic_rows != GPIC_ROWS:
        raise GateError("production catalog row count drift")
    sample_binding = tar_source.load_sample_mapping_authority(
        args.sample_report,
        expected_raw_sha256=args.sample_report_raw_sha256,
        catalog_hash=catalog.catalog_hash,
        gpic_rows=catalog.gpic_rows,
        row_base=catalog.gpic_row_base,
        gpic_sha256=catalog.gpic_sha256,
        tar_root=args.tar_root,
        source_prefix=SOURCE_PREFIX,
        production_gate=True,
    )
    shards = selected_shards(args.worker_id)
    items, selection = select_catalog_items(catalog, shards=shards)
    owners = {
        shard: rank
        for rank in range(args.workers)
        for shard in selected_shards(rank)
    }
    plan_raw_sha = hashlib.sha256(common.canonical_json_bytes(plan)).hexdigest()
    worker_mapping = tar_source.scan_worker_mapping(
        iter(items),
        build_jobtag=args.jobtag,
        worker_id=args.worker_id,
        workers=args.workers,
        source_prefix=SOURCE_PREFIX,
        row_base=catalog.gpic_row_base,
        gpic_rows=catalog.gpic_rows,
        expected_rows=len(items),
        shard_owners=owners,
        catalog_hash=catalog.catalog_hash,
        gpic_sha256=catalog.gpic_sha256,
        partition_plan_raw_sha256=plan_raw_sha,
        partition_plan_sha256=plan["selection_plan_sha256"],
    )
    archive_bytes = 0
    archive_bindings: list[dict[str, Any]] = []
    for shard in shards:
        path = args.tar_root / tar_source.expected_tar_name(shard)
        identity = pilot_core.FileIdentity.from_stat(common.require_regular_file(path))
        archive_bytes += identity.size
        archive_bindings.append(
            {
                "shard_id": shard,
                "name": path.name,
                "bytes": identity.size,
                "identity": dataclasses.asdict(identity),
            }
        )
    barrier(
        root,
        label="catalog_mapping_ready",
        worker_id=args.worker_id,
        workers=args.workers,
        timeout_seconds=args.barrier_timeout_seconds,
    )
    arms: list[dict[str, Any]] = []
    for round_index, tar_workers in enumerate(rotated_arm_order(args.worker_id)):
        barrier(
            root,
            label=f"round_{round_index}_start",
            worker_id=args.worker_id,
            workers=args.workers,
            timeout_seconds=args.barrier_timeout_seconds,
        )
        arm = run_arm(
            items=items,
            worker_mapping=worker_mapping,
            worker_id=args.worker_id,
            workers=args.workers,
            shard_owners=owners,
            catalog=catalog,
            tar_root=args.tar_root,
            tar_workers=tar_workers,
            max_archive_bytes=archive_bytes,
        )
        arm["execution_round"] = round_index
        arms.append(arm)
        barrier(
            root,
            label=f"round_{round_index}_complete",
            worker_id=args.worker_id,
            workers=args.workers,
            timeout_seconds=args.barrier_timeout_seconds,
        )
    semantic_values = {arm["semantic_sha256"] for arm in arms}
    row_values = {arm["rows"] for arm in arms}
    selected_byte_values = {arm["selected_bytes"] for arm in arms}
    if (
        len(semantic_values) != 1
        or row_values != {len(items)}
        or len(selected_byte_values) != 1
        or {arm["tar_workers"] for arm in arms} != set(ARM_WORKERS)
    ):
        raise GateError("1/4/8 semantic replay equality failed")
    pilot_core.validate_catalog_closure(catalog)
    validate_code_snapshot(code)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "generated_at": utc_now(),
        "status": "PASS",
        "jobtag": args.jobtag,
        "worker_id": args.worker_id,
        "workers": args.workers,
        "runtime": runtime_identity(args.worker_id),
        "shards": list(shards),
        "archive_bytes": archive_bytes,
        "archives": archive_bindings,
        "catalog_selection": selection,
        "worker_mapping": worker_mapping,
        "sample_mapping": sample_binding,
        "arms": arms,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "selection_plan_sha256": plan["selection_plan_sha256"],
        "init_sha256": init["init_sha256"],
        "code": code,
        "writes": {
            "fsx_receipt": str(root / "workers" / f"worker-{args.worker_id}.json"),
            "source_writes": 0,
            "s3_writes": 0,
        },
    }
    receipt["receipt_sha256"] = common.self_hash(receipt, "receipt_sha256")
    common.write_fsx_json_once(
        root / "workers" / f"worker-{args.worker_id}.json", receipt
    )
    barrier(
        root,
        label="receipts",
        worker_id=args.worker_id,
        workers=args.workers,
        timeout_seconds=args.barrier_timeout_seconds,
    )
    if args.worker_id == 0:
        summary = publish_summary(
            args, root=root, init=init, code=code, plan=plan
        )
    else:
        summary = {}
    wait_for(
        root / "_COMPLETE",
        timeout_seconds=args.barrier_timeout_seconds,
        label="summary COMPLETE",
    )
    final, raw = common.strict_json(root / "summary.json")
    common.validate_self_hash(final, "summary_sha256", "parser gate summary")
    marker = (root / "_COMPLETE").read_bytes()
    if (
        marker != (hashlib.sha256(raw).hexdigest() + "\n").encode("ascii")
        or final.get("status") != "PASS"
        or final.get("recommendation") != "PASS_FOR_BOUNDED_PRODUCTION_INTEGRATION"
    ):
        raise GateError("parser gate final recommendation is not PASS")
    validate_code_snapshot(code)
    print(
        "OFFLINE_GPIC_TAR_PARSER_GATE_PASS "
        f"worker={args.worker_id} recommended={final['recommended_tar_workers']} "
        f"summary={root / 'summary.json'}",
        flush=True,
    )
    return summary if args.worker_id == 0 else final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--tar-root", type=Path, required=True)
    parser.add_argument("--sample-report", type=Path, required=True)
    parser.add_argument("--sample-report-raw-sha256", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--jobtag", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--tars-per-worker", type=int, required=True)
    parser.add_argument("--barrier-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--read-authority", required=True)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-tar-source-sha256", required=True)
    args = parser.parse_args(argv)
    for name in ("catalog_root", "tar_root", "sample_report", "output_prefix", "job_spec"):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    if (
        args.read_authority != READ_AUTHORITY
        or args.workers != PRODUCTION_WORKERS
        or args.tars_per_worker != TARS_PER_WORKER
        or not (0 <= args.worker_id < args.workers)
        or args.tar_root != TAR_ROOT
        or not common.is_sha256(args.sample_report_raw_sha256)
        or not common.is_sha256(args.expected_runner_sha256)
        or not common.is_sha256(args.expected_tar_source_sha256)
        or not str(args.output_prefix).startswith("/fsx/")
        or args.barrier_timeout_seconds <= 0
        or not args.jobtag
        or Path(args.jobtag).name != args.jobtag
    ):
        raise GateError("production parser-gate invocation shape drift")
    if args.output_prefix == args.catalog_root or args.output_prefix == args.tar_root:
        raise GateError("parser-gate output overlaps an input root")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
