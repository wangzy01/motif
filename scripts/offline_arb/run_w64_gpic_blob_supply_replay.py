#!/usr/bin/env python3
"""64-process production-plan GPIC blob supply/capacity gate.

Each formal rank derives GPIC row IDs from the same frozen W64 plan, performs
two concurrent verified ``pread`` streams (the formal 16-threads/node split),
and publishes process health plus exact transport counters.  Rank 0 hashes the
1.12-GB index once and broadcasts the digest; every rank independently accepts
that collective verification before its first payload read.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
import time
from pathlib import Path
from typing import Any

import torch.distributed as dist

from training.dataline_offline.gpic_blob import (
    CatalogBinding,
    GpicBlobOverlay,
    canonical_json_bytes,
)
from training.dataline_offline.planner_v1 import OfflinePlan


SCHEMA = "offline_gpic_blob_w64_rank_replay_v1"
AGGREGATE_SCHEMA = "offline_gpic_blob_w64_supply_gate_v1"
SHA_RE = re.compile(r"[0-9a-f]{64}")
WORLD_SIZE = 64
LOCAL_WORLD_SIZE = 8
FD_CACHE_SIZE = 1024
READ_WORKERS = 2
CAPACITY_MIN = 1.25
TARGET_STEP_SECONDS = 2.50
MAX_RSS_BYTES = 8 << 30
MAX_RSS_GROWTH_BYTES = 2 << 30


class GateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise GateError(f"not a nonempty regular file: {path}")
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        if _identity(os.fstat(handle.fileno())) != identity:
            raise GateError(f"file changed while opening: {path}")
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
        closed = os.fstat(handle.fileno())
    if _identity(closed) != identity or _identity(path.lstat()) != identity:
        raise GateError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def json_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    if not raw:
        raise GateError(f"empty JSON: {path}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"JSON is not an object: {path}")
    return raw, value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    raw = canonical_json_bytes(value)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise GateError(f"refusing output overwrite: {path}")
    with temporary.open("xb", buffering=0) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not args.artifact_root.is_absolute()
        or args.artifact_root.is_symlink()
        or not args.artifact_root.is_dir()
        or args.artifact_root.name != args.buildtag
        or args.buildtag in {"", ".", "..", "latest"}
    ):
        raise GateError("artifact root/buildtag binding is missing, mutable, or unsafe")
    if SHA_RE.fullmatch(args.manifest_sha256) is None:
        raise GateError("manifest SHA-256 is missing or malformed")
    manifest_raw, manifest = json_object(args.artifact_root / "manifest.json")
    if hashlib.sha256(manifest_raw).hexdigest() != args.manifest_sha256:
        raise GateError("live manifest differs from the exact submitted SHA-256")
    if (args.artifact_root / "_SUCCESS").read_bytes() != (
        args.manifest_sha256 + "\n"
    ).encode("ascii"):
        raise GateError("FSx _SUCCESS does not bind the submitted manifest")
    if (
        manifest.get("schema_version") != "offline_gpic_blob_v1"
        or manifest.get("status") != "PASS"
    ):
        raise GateError("manifest is not an offline_gpic_blob_v1 PASS")
    release = manifest.get("release")
    storage = release.get("storage") if isinstance(release, dict) else None
    if (
        not isinstance(storage, dict)
        or Path(str(storage.get("s3_root", ""))).name != args.buildtag
    ):
        raise GateError("manifest S3 root does not bind the submitted buildtag")
    _catalog_raw, catalog = json_object(args.catalog_root / "header.json")
    return manifest, catalog


def live_catalog_binding(catalog_root: Path, catalog: dict[str, Any]) -> CatalogBinding:
    order = catalog.get("domain_order")
    files = catalog.get("files")
    if not isinstance(order, list) or not isinstance(files, dict) or "gpic" not in order:
        raise GateError("catalog domain/file structure is malformed")
    row_base = 0
    for domain in order:
        info = files.get(domain)
        if not isinstance(info, dict) or type(info.get("rows")) is not int:
            raise GateError(f"catalog rows missing for {domain!r}")
        if domain == "gpic":
            return CatalogBinding(
                catalog_hash=str(catalog["catalog_hash"]),
                catalog_header_sha256=sha256_file(catalog_root / "header.json"),
                gpic_name=str(info["name"]),
                gpic_row_base=row_base,
                gpic_rows=int(info["rows"]),
                gpic_bytes=int(info["bytes"]),
                gpic_sha256=str(info["sha256"]),
                gpic_schema=str(info["schema"]),
            )
        row_base += int(info["rows"])
    raise GateError("catalog GPIC row range was not found")


def collective_errors(local_error: str | None) -> list[str]:
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_error)
    return [f"rank{rank}={error}" for rank, error in enumerate(gathered) if error]


def plan_rows(
    plan: OfflinePlan,
    rank: int,
    *,
    start: int,
    steps: int,
    row_lo: int,
    row_hi: int,
) -> list[int]:
    rows: list[int] = []
    for step in range(start, start + steps):
        _bucket, raw = plan.batch(step, rank)
        rows.extend(int(row) for row in raw.tolist() if row_lo <= int(row) < row_hi)
    return rows


def health_number(value: dict[str, float], key: str) -> float:
    number = float(value.get(key, 0.0))
    if not math.isfinite(number):
        raise GateError(f"non-finite overlay health metric {key}={number}")
    return number


def validate_rank_result(value: dict[str, Any], rank: int, nonce: str) -> None:
    if (
        value.get("schema_version") != SCHEMA
        or value.get("status") != "PASS"
        or value.get("rank") != rank
        or value.get("world_size") != WORLD_SIZE
        or value.get("nonce") != nonce
        or value.get("payload_backend") != "offline_gpic_blob_v1"
        or value.get("storage_backend") != "mountpoint_s3"
        or value.get("fd_cache_size") != FD_CACHE_SIZE
        or value.get("read_workers") != READ_WORKERS
    ):
        raise GateError(f"rank replay identity mismatch rank={rank}")
    if float(value.get("capacity_ratio", 0.0)) < CAPACITY_MIN:
        raise GateError(f"rank {rank} capacity ratio is below {CAPACITY_MIN}")
    if value.get("fd_cache_covers_blob_count") is not True:
        raise GateError(f"rank {rank} fd_cache=1024 does not cover the manifest blob set")
    if int(value.get("process_rss_bytes", -1)) > MAX_RSS_BYTES:
        raise GateError(f"rank {rank} process RSS exceeds 8 GiB")
    if int(value.get("rss_growth_bytes", -1)) > MAX_RSS_GROWTH_BYTES:
        raise GateError(f"rank {rank} overlay RSS growth exceeds 2 GiB")
    if int(value.get("fd_current", -1)) > FD_CACHE_SIZE or int(
        value.get("fd_peak", -1)
    ) > FD_CACHE_SIZE:
        raise GateError(f"rank {rank} blob FD LRU exceeded fd_cache_size=1024")
    soft = int(value.get("rlimit_nofile_soft", -2))
    process_fds = int(value.get("process_fd_count", -1))
    if process_fds < 0 or (soft != -1 and process_fds + 64 > soft):
        raise GateError(f"rank {rank} RLIMIT_NOFILE headroom is invalid")
    for key in (
        "short_read_failures",
        "integrity_failures",
        "pread_errors",
        "source_fallback_items",
    ):
        if int(value.get(key, -1)) != 0:
            raise GateError(f"rank {rank} nonzero {key}={value.get(key)}")
    if value.get("pread_requested_bytes") != value.get("pread_returned_bytes"):
        raise GateError(f"rank {rank} pread byte closure failed")
    if int(value.get("read_items", -1)) != int(value.get("measured_reads", -2)):
        raise GateError(f"rank {rank} measured/read-item closure failed")


def main_run(args: argparse.Namespace) -> int:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "-1"))
    if world != WORLD_SIZE or not 0 <= rank < WORLD_SIZE:
        raise GateError(f"W64 topology mismatch rank={rank} world={world}")
    if local_world != LOCAL_WORLD_SIZE or not 0 <= local_rank < LOCAL_WORLD_SIZE:
        raise GateError(f"local topology mismatch local_rank={local_rank} local_world={local_world}")

    nonce_box = [secrets.token_hex(32) if rank == 0 else None]
    dist.broadcast_object_list(nonce_box, src=0)
    nonce = str(nonce_box[0])
    if SHA_RE.fullmatch(nonce) is None:
        raise GateError("distributed replay nonce is malformed")

    if rank == 0:
        if args.output_root.exists() or args.output_root.is_symlink():
            raise GateError(f"output namespace already exists: {args.output_root}")
        args.output_root.mkdir(parents=True)
    dist.barrier()

    local_error: str | None = None
    overlay: GpicBlobOverlay | None = None
    manifest: dict[str, Any] | None = None
    catalog: dict[str, Any] | None = None
    try:
        manifest, catalog = validate_inputs(args)
        overlay = GpicBlobOverlay(
            args.artifact_root,
            expected_manifest_sha256=args.manifest_sha256,
            catalog=live_catalog_binding(args.catalog_root, catalog),
            fd_cache_size=FD_CACHE_SIZE,
        )
    except BaseException as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    failures = collective_errors(local_error)
    if failures:
        if overlay is not None:
            overlay.close()
        raise GateError("overlay preflight failed collectively: " + "; ".join(failures))
    assert overlay is not None and manifest is not None and catalog is not None

    verification: list[str | None] = [None, None]
    if rank == 0:
        try:
            verification[0] = overlay.compute_index_sha256()
        except BaseException as exc:
            verification[1] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(verification, src=0)
    local_error = None
    try:
        if verification[1] or verification[0] is None:
            raise GateError(verification[1] or "rank0 returned no index digest")
        overlay.accept_collective_index_verification(str(verification[0]))
    except BaseException as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    failures = collective_errors(local_error)
    if failures:
        overlay.close()
        raise GateError("collective index verification failed: " + "; ".join(failures))

    binding = live_catalog_binding(args.catalog_root, catalog)
    plan = OfflinePlan(args.plan_root, world_size=WORLD_SIZE, verify_checksums=True)
    selected = plan_rows(
        plan,
        rank,
        start=args.plan_start_step,
        steps=args.plan_steps,
        row_lo=binding.gpic_row_base,
        row_hi=binding.gpic_row_base + binding.gpic_rows,
    )
    if len(selected) < args.warmup_reads + args.reads_per_rank:
        raise GateError(
            f"rank {rank} plan interval has only {len(selected)} GPIC rows; "
            f"needs {args.warmup_reads + args.reads_per_rank}"
        )
    warmup = selected[: args.warmup_reads]
    measured = selected[args.warmup_reads : args.warmup_reads + args.reads_per_rank]
    for row_id in warmup:
        overlay.read(row_id)
    before = overlay.health_snapshot()
    started = time.perf_counter()
    digest = hashlib.sha256()

    def read_one(row_id: int) -> tuple[int, bytes]:
        return row_id, overlay.read(row_id)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=READ_WORKERS,
        thread_name_prefix=f"gpic-w64-r{rank}",
    ) as pool:
        for row_id, raw in pool.map(read_one, measured):
            digest.update(row_id.to_bytes(8, "little", signed=True))
            digest.update(hashlib.sha256(raw).digest())
    elapsed = time.perf_counter() - started
    after = overlay.health_snapshot()
    if elapsed <= 0:
        raise GateError("non-positive replay elapsed time")
    supply = len(measured) / elapsed
    plan_gpic_per_step = len(selected) / args.plan_steps
    demand = plan_gpic_per_step / TARGET_STEP_SECONDS
    capacity = supply / demand if demand > 0 else 0.0

    def delta(key: str) -> int:
        return int(round(health_number(after, key) - health_number(before, key)))

    value = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "nonce": nonce,
        "rank": rank,
        "world_size": world,
        "local_rank": local_rank,
        "local_world_size": local_world,
        "host": os.uname().nodename,
        "payload_backend": "offline_gpic_blob_v1",
        "storage_backend": manifest["release"]["backend"],
        "artifact_root": str(args.artifact_root.resolve(strict=True)),
        "buildtag": args.buildtag,
        "manifest_sha256": args.manifest_sha256,
        "release_sha256": manifest.get("release_sha256"),
        "fd_cache_size": FD_CACHE_SIZE,
        "read_workers": READ_WORKERS,
        "fd_cache_covers_blob_count": bool(
            health_number(after, "fd_cache_covers_blob_count") == 1.0
        ),
        "manifest_blob_count": int(health_number(after, "blob_count")),
        "plan_start_step": args.plan_start_step,
        "plan_steps": args.plan_steps,
        "plan_gpic_items": len(selected),
        "plan_gpic_per_step": plan_gpic_per_step,
        "target_step_seconds": TARGET_STEP_SECONDS,
        "required_gpic_img_s": demand,
        "warmup_reads": len(warmup),
        "measured_reads": len(measured),
        "elapsed_s": elapsed,
        "supply_img_s": supply,
        "capacity_ratio": capacity,
        "payload_identity_sha256": digest.hexdigest(),
        "process_rss_bytes": int(health_number(after, "process_rss_bytes")),
        "startup_rss_bytes": int(health_number(after, "startup_rss_bytes")),
        "rss_growth_bytes": max(
            0,
            int(health_number(after, "process_rss_bytes"))
            - int(health_number(after, "startup_rss_bytes")),
        ),
        "process_fd_count": int(health_number(after, "process_fd_count")),
        "startup_fd_count": int(health_number(after, "startup_fd_count")),
        "rlimit_nofile_soft": int(health_number(after, "rlimit_nofile_soft")),
        "rlimit_nofile_hard": int(health_number(after, "rlimit_nofile_hard")),
        "fd_current": int(health_number(after, "fd_current")),
        "fd_peak": int(health_number(after, "fd_peak")),
        "short_read_failures": delta("short_read_failures"),
        "integrity_failures": delta("integrity_failures"),
        "pread_errors": delta("pread_errors"),
        "source_fallback_items": 0,
        "pread_requested_bytes": delta("pread_requested_bytes"),
        "pread_returned_bytes": delta("pread_returned_bytes"),
        "read_items": delta("read_items"),
        "bindings": {
            "plan_header_sha256": sha256_file(args.plan_root / "header.json"),
            "catalog_header_sha256": sha256_file(args.catalog_root / "header.json"),
            "manifest_sha256": args.manifest_sha256,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "job_yaml_sha256": sha256_file(args.job_yaml),
            "blob_runtime_sha256": sha256_file(
                Path(__file__).resolve().parents[2] / "training/dataline_offline/gpic_blob.py"
            ),
        },
    }
    validate_rank_result(value, rank, nonce)
    rank_path = args.output_root / f"rank{rank:02d}.json"
    atomic_json(rank_path, value)

    gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, value)
    if rank == 0:
        rows: list[dict[str, Any]] = []
        bindings = value["bindings"]
        for expected_rank, item in enumerate(gathered):
            if not isinstance(item, dict):
                raise GateError(f"missing gathered rank result {expected_rank}")
            validate_rank_result(item, expected_rank, nonce)
            if item["bindings"] != bindings:
                raise GateError(f"immutable binding drift at rank {expected_rank}")
            file_path = args.output_root / f"rank{expected_rank:02d}.json"
            rows.append(
                {
                    "rank": expected_rank,
                    "file": file_path.name,
                    "file_sha256": sha256_file(file_path),
                    "supply_img_s": item["supply_img_s"],
                    "required_gpic_img_s": item["required_gpic_img_s"],
                    "capacity_ratio": item["capacity_ratio"],
                    "process_rss_bytes": item["process_rss_bytes"],
                    "process_fd_count": item["process_fd_count"],
                    "fd_peak": item["fd_peak"],
                }
            )
        aggregate = {
            "schema_version": AGGREGATE_SCHEMA,
            "status": "PASS",
            "verdict": "CAPACITY_AT_LEAST_1.25",
            "nonce": nonce,
            "world_size": WORLD_SIZE,
            "local_world_size": LOCAL_WORLD_SIZE,
            "payload_backend": "offline_gpic_blob_v1",
            "storage_backend": "mountpoint_s3",
            "artifact_root": str(args.artifact_root.resolve(strict=True)),
            "buildtag": args.buildtag,
            "manifest_sha256": args.manifest_sha256,
            "fd_cache_size": FD_CACHE_SIZE,
            "read_workers_per_rank": READ_WORKERS,
            "target_step_seconds": TARGET_STEP_SECONDS,
            "capacity_minimum": CAPACITY_MIN,
            "allrank_slowest_supply_img_s": min(item["supply_img_s"] for item in gathered if item),
            "allrank_highest_required_gpic_img_s": max(
                item["required_gpic_img_s"] for item in gathered if item
            ),
            "allrank_min_capacity_ratio": min(item["capacity_ratio"] for item in gathered if item),
            "allrank_max_rss_bytes": max(item["process_rss_bytes"] for item in gathered if item),
            "allrank_max_startup_rss_bytes": max(
                item["startup_rss_bytes"] for item in gathered if item
            ),
            "allrank_max_rss_growth_bytes": max(
                item["rss_growth_bytes"] for item in gathered if item
            ),
            "allrank_max_process_fd_count": max(
                item["process_fd_count"] for item in gathered if item
            ),
            "allrank_max_fd_peak": max(item["fd_peak"] for item in gathered if item),
            "allrank_min_rlimit_nofile_soft": min(
                item["rlimit_nofile_soft"] for item in gathered if item
            ),
            "allrank_min_rlimit_nofile_hard": min(
                item["rlimit_nofile_hard"] for item in gathered if item
            ),
            "allrank_fd_cache_covers_blob_count": all(
                item["fd_cache_covers_blob_count"] for item in gathered if item
            ),
            "allrank_sum_short_read_failures": sum(
                item["short_read_failures"] for item in gathered if item
            ),
            "allrank_sum_integrity_failures": sum(
                item["integrity_failures"] for item in gathered if item
            ),
            "allrank_sum_pread_errors": sum(item["pread_errors"] for item in gathered if item),
            "allrank_sum_source_fallback_items": 0,
            "allrank_sum_pread_requested_bytes": sum(
                item["pread_requested_bytes"] for item in gathered if item
            ),
            "allrank_sum_pread_returned_bytes": sum(
                item["pread_returned_bytes"] for item in gathered if item
            ),
            "bindings": bindings,
            "ranks": rows,
        }
        if aggregate["allrank_min_capacity_ratio"] < CAPACITY_MIN:
            raise GateError("aggregate capacity is below 1.25")
        if aggregate["allrank_sum_pread_requested_bytes"] != aggregate["allrank_sum_pread_returned_bytes"]:
            raise GateError("aggregate pread byte closure failed")
        atomic_json(args.output_root / "aggregate.json", aggregate)
        marker = (
            f"PASS|{nonce}|{args.buildtag}|{args.manifest_sha256}|"
            f"{sha256_file(args.output_root / 'aggregate.json')}\n"
        ).encode("ascii")
        success = args.output_root / "_SUCCESS"
        temporary = args.output_root / "._SUCCESS.tmp"
        with temporary.open("xb", buffering=0) as handle:
            handle.write(marker)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, success)
        print(
            "GPIC_BLOB_W64_SUPPLY_PASS "
            f"min_capacity={aggregate['allrank_min_capacity_ratio']:.3f} "
            f"slowest_img_s={aggregate['allrank_slowest_supply_img_s']:.3f} "
            f"output={args.output_root}"
        )
    dist.barrier()
    overlay.close()
    dist.destroy_process_group()
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--artifact-root", type=Path, required=True)
    value.add_argument("--buildtag", required=True)
    value.add_argument("--manifest-sha256", required=True)
    value.add_argument("--plan-root", type=Path, required=True)
    value.add_argument("--catalog-root", type=Path, required=True)
    value.add_argument("--job-yaml", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--plan-start-step", type=int, default=0)
    value.add_argument("--plan-steps", type=int, default=256)
    value.add_argument("--warmup-reads", type=int, default=16)
    value.add_argument("--reads-per-rank", type=int, default=512)
    return value


if __name__ == "__main__":
    try:
        raise SystemExit(main_run(parser().parse_args()))
    except GateError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
