#!/usr/bin/env python3
"""Eight-node GPIC source -> local staging -> Mountpoint-S3 scale probe.

This program is intentionally separate from ``run_offline_gpic_blob_pilot``.
It measures scale-out, not loader semantics, and never mutates a production
catalog or an existing pilot namespace.  The production shape is:

* eight CPU16 JCC workers;
* 400,000 mutually exclusive main rows per worker, partitioned by
  ``catalog_local_index % 8``;
* disjoint A/B/B/A arms (source-only / source+blob / source+blob / source-only)
  ordered by source path within every worker; and
* two additional, disjoint worker-0 source-only controls bracketing the
  eight-worker arms.  They provide the one-node denominator for E8 without
  re-reading an object used by another arm.

All inter-worker synchronization and compact evidence are on FSx.  Blob bytes
are staged one file at a time under ``/local/evan``, fully hashed/fsynced, then
copied sequentially to worker-exclusive Mountpoint-S3 keys.  The coordinator
writes the S3 job ``_SUCCESS`` last, after all eight worker manifests close.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import platform
import resource
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow.parquet as pq

from scripts.offline_arb import run_offline_gpic_blob_pilot as pilot_core


SCHEMA_VERSION = "offline_gpic_scale_probe_v1"
S3_INIT_SCHEMA = "offline_gpic_scale_s3_init_v1"
COORD_INIT_SCHEMA = "offline_gpic_scale_coord_init_v1"
WORKER_RECEIPT_SCHEMA = "offline_gpic_scale_worker_receipt_v1"
JOB_RECEIPT_SCHEMA = "offline_gpic_scale_job_receipt_v1"
GLOBAL_APPROVAL_SCHEMA = "offline_gpic_scale_global_approval_v1"
PRODUCTION_WORKERS = 8
PRODUCTION_MAIN_ROWS = 400_000
PRODUCTION_CONTROL_ROWS_PER_ARM = 60_000
PRODUCTION_EXCLUDE_BUILD_ROWS = 100_000
PRODUCTION_EXCLUDE_SCAN_ROWS = 40_000
PRODUCTION_BLOB_BYTES = 4 * (1 << 30)
# Decimal-byte cap keeps the eight-worker payload peak below 250 GB while
# leaving modest headroom over the measured ~28.9 GB/worker quick sample.
PRODUCTION_MAX_S3_BYTES_PER_WORKER = 31_000_000_000
PRODUCTION_MAX_LOCAL_STAGING_BYTES_PER_WORKER = 8 * (1 << 30)
PRODUCTION_MAX_FSX_EVIDENCE_BYTES_PER_WORKER = 128 * (1 << 20)
PRODUCTION_LOCAL_FREE_FLOOR_BYTES = 1_000_000_000_000
PRODUCTION_FULL_PARTITION_LOCAL_BUDGET_BYTES = 500_000_000_000
PRODUCTION_TOPOLOGY_JOBTAG = "evan-mot-crop-offlinearb-local-topology-94ns"
PRODUCTION_TOPOLOGY_RAW_SHA256 = (
    "27c48e9d6438d992fa850a0da9192c7043ccb6714e65ffa33460a755b4c5d73a"
)
PRODUCTION_TOPOLOGY_SUMMARY = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_arb_local_topology_v1/"
    f"{PRODUCTION_TOPOLOGY_JOBTAG}/summary.json"
)
PRODUCTION_SOURCE_PREFIX = pilot_core.SOURCE_PREFIX
WRITE_AUTHORITY = "offline-gpic-scale-probe-v1-write-authorized"
INDEX_STRUCT = pilot_core.INDEX_STRUCT
SELECTION_STRUCT = struct.Struct("<B7xqq32s")
MIB = 1 << 20
PHASES = (
    ("single_before", "single_control", False, 0),
    ("source_a", "source_only_8", False, 0),
    ("combined_b1", "combined_8", True, 1),
    ("combined_b2", "combined_8", True, 2),
    ("source_a2", "source_only_8", False, 3),
    ("single_after", "single_control", False, 1),
)


class ScaleProbeError(RuntimeError):
    """Fail-closed scale-probe contract error."""


@dataclasses.dataclass(frozen=True)
class Item:
    local_index: int
    row_id: int
    image_path: str


@dataclasses.dataclass(frozen=True)
class SourceResult:
    item: Item
    payload: bytes
    sha256_raw: bytes
    source_open_ns: int
    source_read_hash_ns: int
    source_total_ns: int
    attempts: int
    errors: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
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


def self_hash(value: dict[str, Any], field: str) -> str:
    core = dict(value)
    core.pop(field, None)
    return hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def sha256_file(path: Path, block_bytes: int = 8 * MIB) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require_regular_file(path: Path) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ScaleProbeError(f"missing regular file: {path}") from exc
    if not path.is_file() or path.is_symlink() or value.st_nlink < 1:
        raise ScaleProbeError(f"unsafe non-regular file: {path}")
    return value


def require_regular_dir(path: Path) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ScaleProbeError(f"missing directory: {path}") from exc
    if not path.is_dir() or path.is_symlink() or value.st_nlink < 1:
        raise ScaleProbeError(f"unsafe non-directory: {path}")
    return value


def write_once_atomic_bytes(
    path: Path, raw: bytes, *, fsync_file: bool = True
) -> None:
    temporary = path.parent / f".{path.name}.work.{uuid.uuid4().hex}"
    pilot_core.write_once_bytes(temporary, raw, fsync_file=fsync_file)
    pilot_core.fsync_dir(path.parent)
    try:
        pilot_core.publish_file_noreplace(temporary, path)
    except BaseException as exc:
        try:
            path.lstat()
        except FileNotFoundError:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        except BaseException as inspect_exc:
            raise pilot_core.CommittedSyncUnconfirmed(
                f"atomic file publication state is unknowable: {path.name}"
            ) from inspect_exc
        # The final name is visible.  It may be our complete inode after a
        # link+fsync error or a no-clobber collision; either case is a monotonic
        # committed/unknown state and must never be followed by FAIL evidence.
        raise pilot_core.CommittedSyncUnconfirmed(
            f"atomic file final name became visible while publish reported failure: {path.name}"
        ) from exc
    try:
        pilot_core.fsync_dir(path.parent)
    except BaseException as exc:
        raise pilot_core.CommittedSyncUnconfirmed(
            f"atomic file became visible but parent sync is unconfirmed: {path.name}"
        ) from exc


def write_once_json(path: Path, value: Any, *, fsync_file: bool = True) -> bytes:
    raw = canonical_json_bytes(value)
    write_once_atomic_bytes(path, raw, fsync_file=fsync_file)
    return raw


def write_fd_all(descriptor: int, payload: bytes, *, label: str) -> None:
    if type(descriptor) is not int or descriptor < 0 or not payload:
        raise ScaleProbeError(f"invalid fd write request: {label}")
    view = memoryview(payload)
    while view:
        try:
            written = os.write(descriptor, view)
        except InterruptedError:
            continue
        if written <= 0 or written > len(view):
            raise ScaleProbeError(f"short/zero fd write: {label}")
        view = view[written:]


def code_snapshot(job_spec: Path) -> dict[str, Any]:
    runner = Path(__file__).resolve(strict=True)
    core = Path(pilot_core.__file__).resolve(strict=True)
    job_spec = job_spec.expanduser().resolve(strict=True)
    return {
        "runner_path": str(runner),
        "runner_sha256": sha256_file(runner),
        "publisher_core_path": str(core),
        "publisher_core_sha256": sha256_file(core),
        "job_spec_path": str(job_spec),
        "job_spec_sha256": sha256_file(job_spec),
        "python": platform.python_version(),
        "index_struct": INDEX_STRUCT.format,
        "index_record_bytes": INDEX_STRUCT.size,
        "selection_struct": SELECTION_STRUCT.format,
        "selection_record_bytes": SELECTION_STRUCT.size,
    }


def validate_code_snapshot(value: dict[str, Any]) -> None:
    for prefix in ("runner", "publisher_core", "job_spec"):
        path = Path(value[f"{prefix}_path"])
        if sha256_file(path) != value[f"{prefix}_sha256"]:
            raise ScaleProbeError(f"{prefix} changed during scale probe")


def stratified_positions(total: int, count: int) -> list[int]:
    if type(total) is not int or type(count) is not int or not (0 < count <= total):
        raise ScaleProbeError(f"invalid stratified shape total={total} count={count}")
    values = [((2 * index + 1) * total) // (2 * count) for index in range(count)]
    if values != sorted(set(values)):
        raise ScaleProbeError("stratified positions are not strictly unique")
    return values


def partition_size(total_rows: int, worker_id: int, workers: int) -> int:
    if not (0 <= worker_id < workers) or total_rows <= worker_id:
        raise ScaleProbeError("invalid modulo partition")
    return ((total_rows - 1 - worker_id) // workers) + 1


def map_allowed_rank(rank: int, total_ordinals: int, excluded_ordinals: Sequence[int]) -> int:
    """Map an allowed-sequence rank to its original partition ordinal."""
    if rank < 0 or rank >= total_ordinals - len(excluded_ordinals):
        raise ScaleProbeError("allowed rank is out of range")
    value = rank
    while True:
        updated = rank + bisect.bisect_right(excluded_ordinals, value)
        if updated == value:
            break
        value = updated
    if value >= total_ordinals or (
        bisect.bisect_left(excluded_ordinals, value) < len(excluded_ordinals)
        and excluded_ordinals[bisect.bisect_left(excluded_ordinals, value)] == value
    ):
        raise ScaleProbeError("allowed-rank inversion did not close")
    return value


def select_partition_positions(
    *,
    total_rows: int,
    worker_id: int,
    workers: int,
    count: int,
    excluded_positions: Sequence[int],
) -> list[int]:
    part_rows = partition_size(total_rows, worker_id, workers)
    excluded_ordinals = sorted(
        {
            (position - worker_id) // workers
            for position in excluded_positions
            if position % workers == worker_id
        }
    )
    if any(value < 0 or value >= part_rows for value in excluded_ordinals):
        raise ScaleProbeError("excluded ordinal escaped its partition")
    allowed = part_rows - len(excluded_ordinals)
    ranks = stratified_positions(allowed, count)
    positions = [
        worker_id + workers * map_allowed_rank(rank, part_rows, excluded_ordinals)
        for rank in ranks
    ]
    if (
        len(positions) != count
        or positions != sorted(set(positions))
        or any(position % workers != worker_id for position in positions)
        or set(positions).intersection(excluded_positions)
    ):
        raise ScaleProbeError("partition selection closure failed")
    return positions


def expected_position_sets(
    *,
    total_rows: int,
    worker_id: int,
    workers: int,
    main_rows: int,
    control_rows_per_arm: int,
    exclude_build_rows: int,
    exclude_scan_rows: int,
) -> tuple[list[int], list[int], list[int]]:
    excluded_build = stratified_positions(total_rows, exclude_build_rows)
    excluded_scan = stratified_positions(total_rows, exclude_scan_rows)
    excluded = sorted(set(excluded_build).union(excluded_scan))
    if len(excluded) != exclude_build_rows + exclude_scan_rows:
        raise ScaleProbeError("pilot exclusion sets overlap; expected exact 140k-style union")
    main = select_partition_positions(
        total_rows=total_rows,
        worker_id=worker_id,
        workers=workers,
        count=main_rows,
        excluded_positions=excluded,
    )
    controls: list[int] = []
    if worker_id == 0 and control_rows_per_arm:
        controls = select_partition_positions(
            total_rows=total_rows,
            worker_id=worker_id,
            workers=workers,
            count=2 * control_rows_per_arm,
            excluded_positions=sorted(set(excluded).union(main)),
        )
    control_before = controls[0::2]
    control_after = controls[1::2]
    expected_controls = control_rows_per_arm if worker_id == 0 else 0
    if len(control_before) != expected_controls or len(control_after) != expected_controls:
        raise ScaleProbeError("single-node control split did not close")
    return main, control_before, control_after


def select_catalog_items(
    catalog: pilot_core.CatalogBinding,
    positions: Sequence[int],
    *,
    source_prefix: str,
) -> list[Item]:
    if not positions or list(positions) != sorted(set(positions)):
        raise ScaleProbeError("catalog target positions are malformed")
    parquet = pq.ParquetFile(catalog.gpic_path)
    selected: list[Item | None] = [None] * len(positions)
    cursor = 0
    group_start = 0
    for group_index in range(parquet.num_row_groups):
        group_rows = parquet.metadata.row_group(group_index).num_rows
        group_stop = group_start + group_rows
        targets: list[tuple[int, int]] = []
        while cursor < len(positions) and positions[cursor] < group_stop:
            if positions[cursor] < group_start:
                raise ScaleProbeError("catalog row-group mapping regressed")
            targets.append((cursor, positions[cursor] - group_start))
            cursor += 1
        if targets:
            table = parquet.read_row_group(
                group_index, columns=["row_id", "image_path"], use_threads=False
            ).combine_chunks()
            row_ids = table.column("row_id")
            paths = table.column("image_path")
            for output_index, row_offset in targets:
                local = positions[output_index]
                row_id = row_ids[row_offset].as_py()
                image_path = paths[row_offset].as_py()
                if type(row_id) is not int or row_id != catalog.gpic_row_base + local:
                    raise ScaleProbeError(f"catalog row identity drift local={local}")
                if not isinstance(image_path, str) or not image_path.startswith(source_prefix):
                    raise ScaleProbeError(f"catalog source-prefix drift row_id={row_id}")
                suffix = Path(image_path).suffix.lower()
                if suffix not in {".jpg", ".png"}:
                    raise ScaleProbeError(f"catalog source extension drift row_id={row_id}")
                selected[output_index] = Item(local, row_id, image_path)
        group_start = group_stop
    if group_start != catalog.gpic_rows or cursor != len(positions) or any(
        value is None for value in selected
    ):
        raise ScaleProbeError("catalog selection contains a hole")
    return [value for value in selected if value is not None]


def selection_digest(items: Sequence[Item]) -> str:
    digest = hashlib.sha256()
    for item in items:
        path = item.image_path.encode("utf-8")
        digest.update(struct.pack("<qqI", item.local_index, item.row_id, len(path)))
        digest.update(path)
    return digest.hexdigest()


def write_selection_file(path: Path, main: Sequence[Item], controls: Sequence[Item]) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("xb", buffering=0) as handle:
        for kind, items in ((0, main), (1, controls)):
            for item in items:
                record = SELECTION_STRUCT.pack(
                    kind,
                    item.local_index,
                    item.row_id,
                    hashlib.sha256(item.image_path.encode("utf-8")).digest(),
                )
                pilot_core.write_all(handle, record, label="selection.bin")
                digest.update(record)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "records": len(main) + len(controls),
        "main_records": len(main),
        "control_records": len(controls),
    }


class Telemetry:
    FIELDS = (
        "objects",
        "source_bytes",
        "source_attempts",
        "source_errors",
        "source_open_ns",
        "source_read_hash_ns",
        "source_total_ns",
        "ordered_hol_ns",
        "local_stage_write_ns",
        "local_stage_close_ns",
        "local_stage_fsync_ns",
        "local_cleanup_ns",
        "s3_copy_read_ns",
        "s3_write_ns",
        "s3_fsync_ns",
        "staged_bytes",
        "s3_write_bytes",
        "index_write_ns",
        "index_fsync_ns",
    )

    def __init__(self, path: Path, interval_seconds: float):
        self.path = path
        self.interval_seconds = interval_seconds
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread_error: BaseException | None = None
        self._started_ns = time.perf_counter_ns()
        self._phase = "initializing"
        self._phase_started_ns = self._started_ns
        self._phase_counters = {field: 0 for field in self.FIELDS}
        self._total_counters = {field: 0 for field in self.FIELDS}
        self._thread = threading.Thread(target=self._ticker, name="scale-telemetry", daemon=True)
        self._thread.start()

    def _process_stats(self) -> dict[str, Any]:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_bytes: int | None = None
        try:
            fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
            rss_bytes = int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
        try:
            fd_count: int | None = len(list(Path("/proc/self/fd").iterdir()))
        except OSError:
            fd_count = None
        return {
            "rss_bytes": rss_bytes,
            "max_rss_bytes": int(usage.ru_maxrss) * 1024,
            "fd_count": fd_count,
            "cpu_user_seconds": usage.ru_utime,
            "cpu_system_seconds": usage.ru_stime,
        }

    def _write_event_locked(self, reason: str) -> None:
        now_ns = time.perf_counter_ns()
        event = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": utc_now(),
            "reason": reason,
            "phase": self._phase,
            "elapsed_seconds": (now_ns - self._started_ns) / 1e9,
            "phase_elapsed_seconds": (now_ns - self._phase_started_ns) / 1e9,
            "phase_counters": dict(self._phase_counters),
            "total_counters": dict(self._total_counters),
            "process": self._process_stats(),
        }
        write_fd_all(
            self._fd,
            canonical_json_bytes(event),
            label="telemetry.jsonl",
        )

    def _write_event(self, reason: str) -> None:
        with self._lock:
            self._write_event_locked(reason)

    def _ticker(self) -> None:
        try:
            while not self._stop.wait(self.interval_seconds):
                self._write_event("interval")
        except BaseException as exc:
            self._thread_error = exc
            self._stop.set()

    def _raise_thread_error(self) -> None:
        if self._thread_error is not None:
            raise ScaleProbeError("telemetry background thread failed") from self._thread_error

    def begin_phase(self, phase: str) -> None:
        self._raise_thread_error()
        with self._lock:
            self._phase = phase
            self._phase_started_ns = time.perf_counter_ns()
            self._phase_counters = {field: 0 for field in self.FIELDS}
            self._write_event_locked("phase_start")

    def end_phase(self) -> None:
        self._raise_thread_error()
        with self._lock:
            self._write_event_locked("phase_end")
            previous = self._phase
            self._phase = f"barrier_after:{previous}"
            self._phase_started_ns = time.perf_counter_ns()
            self._phase_counters = {field: 0 for field in self.FIELDS}

    def add(self, **values: int) -> None:
        self._raise_thread_error()
        with self._lock:
            for key, value in values.items():
                if key not in self._phase_counters or type(value) is not int or value < 0:
                    raise ScaleProbeError(f"invalid telemetry increment {key}={value!r}")
                self._phase_counters[key] += value
                self._total_counters[key] += value

    def phase_snapshot(self) -> dict[str, int]:
        self._raise_thread_error()
        with self._lock:
            return dict(self._phase_counters)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds + 1.0))
        if self._thread.is_alive():
            raise ScaleProbeError("telemetry thread did not stop")
        error = self._thread_error
        try:
            if error is None:
                with self._lock:
                    self._write_event_locked("phase_end")
            os.fsync(self._fd)
        finally:
            os.close(self._fd)
        if error is not None:
            raise ScaleProbeError("telemetry background thread failed") from error


def _read_source(
    item: Item,
    attempts: int,
    block_bytes: int,
    retry_base_seconds: float,
) -> SourceResult:
    last_error: BaseException | None = None
    errors = 0
    cumulative_open_ns = 0
    cumulative_read_hash_ns = 0
    all_started_ns = time.perf_counter_ns()
    for attempt in range(1, attempts + 1):
        opened_ns = time.perf_counter_ns()
        descriptor: int | None = None
        open_done_ns: int | None = None
        try:
            descriptor = os.open(item.image_path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            open_done_ns = time.perf_counter_ns()
            cumulative_open_ns += open_done_ns - opened_ns
            before = pilot_core.FileIdentity.from_stat(os.fstat(descriptor))
            if before.size <= 0 or before.size > 0xFFFFFFFF:
                raise ScaleProbeError(f"invalid GPIC source size row_id={item.row_id}")
            digest = hashlib.sha256()
            payload = bytearray()
            while True:
                block = os.read(descriptor, block_bytes)
                if not block:
                    break
                digest.update(block)
                payload.extend(block)
            after = pilot_core.FileIdentity.from_stat(os.fstat(descriptor))
            finished_ns = time.perf_counter_ns()
            cumulative_read_hash_ns += finished_ns - open_done_ns
            if before != after or len(payload) != before.size:
                raise ScaleProbeError(f"GPIC source identity/length drift row_id={item.row_id}")
            return SourceResult(
                item=item,
                payload=bytes(payload),
                sha256_raw=digest.digest(),
                source_open_ns=cumulative_open_ns,
                source_read_hash_ns=cumulative_read_hash_ns,
                source_total_ns=finished_ns - all_started_ns,
                attempts=attempt,
                errors=errors,
            )
        except ScaleProbeError:
            raise
        except OSError as exc:
            failed_ns = time.perf_counter_ns()
            if open_done_ns is None:
                cumulative_open_ns += failed_ns - opened_ns
            else:
                cumulative_read_hash_ns += failed_ns - open_done_ns
            errors += 1
            last_error = exc
            if attempt < attempts:
                time.sleep(retry_base_seconds * (2 ** (attempt - 1)))
        finally:
            if descriptor is not None:
                os.close(descriptor)
    raise ScaleProbeError(
        f"GPIC source retries exhausted row_id={item.row_id} class={type(last_error).__name__}"
    ) from last_error


def ordered_reads(
    items: Sequence[Item],
    *,
    workers: int,
    attempts: int,
    block_bytes: int,
    retry_base_seconds: float,
) -> Iterator[tuple[SourceResult, int]]:
    capacity = max(workers, workers * 2)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="gpic-scale-source"
    ) as executor:
        futures: dict[int, concurrent.futures.Future[SourceResult]] = {}
        submitted = 0

        def submit(index: int) -> None:
            futures[index] = executor.submit(
                _read_source,
                items[index],
                attempts,
                block_bytes,
                retry_base_seconds,
            )

        while submitted < min(len(items), capacity):
            submit(submitted)
            submitted += 1
        try:
            for index in range(len(items)):
                before_wait = time.perf_counter_ns()
                result = futures.pop(index).result()
                hol_ns = time.perf_counter_ns() - before_wait
                yield result, hol_ns
                if submitted < len(items):
                    submit(submitted)
                    submitted += 1
        except BaseException:
            for future in futures.values():
                future.cancel()
            raise


class StagedS3BlobWriter:
    def __init__(
        self,
        local_root: Path,
        s3_worker_root: Path,
        target_bytes: int,
        telemetry: Telemetry,
        max_s3_total_bytes: int,
        max_local_staging_bytes: int,
    ):
        self.local_root = local_root
        self.s3_worker_root = s3_worker_root
        self.target_bytes = target_bytes
        self.telemetry = telemetry
        self.max_s3_total_bytes = max_s3_total_bytes
        self.max_local_staging_bytes = max_local_staging_bytes
        self.total_bytes = 0
        self.local_peak_bytes = 0
        self.blob_id = -1
        self.handle: Any = None
        self.offset = 0
        self.digest: Any = None
        self.files: list[dict[str, Any]] = []

    def _open_next(self) -> None:
        self.blob_id += 1
        path = self.local_root / f"blob-{self.blob_id:06d}.bin"
        self.handle = path.open("xb", buffering=0)
        self.offset = 0
        self.digest = hashlib.sha256()

    def _close_current(self) -> None:
        if self.handle is None:
            return
        path = Path(self.handle.name)
        close_started = time.perf_counter_ns()
        self.handle.flush()
        close_ns = time.perf_counter_ns() - close_started
        fsync_started = time.perf_counter_ns()
        os.fsync(self.handle.fileno())
        fsync_ns = time.perf_counter_ns() - fsync_started
        self.handle.close()
        stat = require_regular_file(path)
        if stat.st_size != self.offset:
            raise ScaleProbeError("local staging blob size drift after close")
        local_sha256 = self.digest.hexdigest()
        destination = self.s3_worker_root / path.name
        if destination.exists() or destination.is_symlink():
            raise ScaleProbeError(f"S3 blob key is not fresh: {destination}")
        source_digest = hashlib.sha256()
        s3_digest = hashlib.sha256()
        copied = 0
        read_ns = 0
        write_ns = 0
        s3_handle: Any = None
        source_handle: Any = None
        try:
            source_handle = path.open("rb", buffering=0)
            s3_handle = destination.open("xb", buffering=0)
            while True:
                before_read = time.perf_counter_ns()
                block = source_handle.read(MIB)
                read_ns += time.perf_counter_ns() - before_read
                if not block:
                    break
                source_digest.update(block)
                before_write = time.perf_counter_ns()
                view = memoryview(block)
                while view:
                    written = s3_handle.write(view)
                    if written is None or written <= 0:
                        raise ScaleProbeError("short Mountpoint-S3 blob write")
                    s3_digest.update(view[:written])
                    copied += written
                    view = view[written:]
                write_ns += time.perf_counter_ns() - before_write
            source_after = pilot_core.FileIdentity.from_stat(os.fstat(source_handle.fileno()))
            if (
                source_after != pilot_core.FileIdentity.from_stat(stat)
                or source_after.size != self.offset
            ):
                raise ScaleProbeError("local staging blob identity drift during S3 copy")
            s3_handle.flush()
            before_s3_fsync = time.perf_counter_ns()
            os.fsync(s3_handle.fileno())
            s3_fsync_ns = time.perf_counter_ns() - before_s3_fsync
            s3_handle.close()
            s3_handle = None
        finally:
            if source_handle is not None:
                source_handle.close()
            if s3_handle is not None:
                s3_handle.close()
        if (
            copied != self.offset
            or source_digest.hexdigest() != local_sha256
            or s3_digest.hexdigest() != local_sha256
        ):
            raise ScaleProbeError("local staging/S3 copy byte or SHA closure drift")
        destination_stat = require_regular_file(destination)
        if destination_stat.st_size != self.offset:
            raise ScaleProbeError("Mountpoint-S3 blob stat size drift after fsync")
        self.files.append(
            {
                "name": path.name,
                "key": str(destination),
                "blob_id": self.blob_id,
                "bytes": self.offset,
                "sha256": local_sha256,
            }
        )
        before_cleanup = time.perf_counter_ns()
        path.unlink()
        pilot_core.fsync_dir(self.local_root)
        cleanup_ns = time.perf_counter_ns() - before_cleanup
        self.telemetry.add(
            local_stage_close_ns=close_ns,
            local_stage_fsync_ns=fsync_ns,
            local_cleanup_ns=cleanup_ns,
            s3_copy_read_ns=read_ns,
            s3_write_ns=write_ns,
            s3_fsync_ns=s3_fsync_ns,
            s3_write_bytes=self.offset,
        )
        self.handle = None
        self.digest = None

    def append(self, payload: bytes) -> tuple[int, int]:
        if not payload:
            raise ScaleProbeError("empty payload")
        if self.total_bytes + len(payload) > self.max_s3_total_bytes:
            raise ScaleProbeError(
                "combined S3 payload exceeds the per-worker hard cap "
                f"cap={self.max_s3_total_bytes}"
            )
        if self.handle is None:
            self._open_next()
        elif self.offset and self.offset + len(payload) > self.target_bytes:
            self._close_current()
            self._open_next()
        assert self.handle is not None and self.digest is not None
        if self.offset + len(payload) > self.max_local_staging_bytes:
            raise ScaleProbeError("local staging blob exceeds its hard cap")
        start = self.offset
        before = time.perf_counter_ns()
        view = memoryview(payload)
        while view:
            written = self.handle.write(view)
            if written is None or written <= 0:
                raise ScaleProbeError("short blob write")
            self.digest.update(view[:written])
            self.offset += written
            view = view[written:]
        elapsed = time.perf_counter_ns() - before
        self.local_peak_bytes = max(self.local_peak_bytes, self.offset)
        self.telemetry.add(
            local_stage_write_ns=elapsed,
            staged_bytes=len(payload),
        )
        self.total_bytes += len(payload)
        return self.blob_id, start

    def close(self) -> list[dict[str, Any]]:
        self._close_current()
        if not self.files:
            raise ScaleProbeError("combined arms produced no S3 blob")
        return self.files

    def abort(self) -> None:
        if self.handle is not None:
            try:
                self.handle.close()
            finally:
                self.handle = None


def space_snapshot(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        return {
            "status": "known",
            "path": str(path),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    except OSError as exc:
        return {
            "status": "unknown",
            "path": str(path),
            "reason": type(exc).__name__,
        }


def process_stats(local_path: Path, s3_path: Path) -> dict[str, Any]:
    return {
        "fsx_space": space_snapshot(Path("/fsx")),
        "local_space": space_snapshot(local_path),
        "s3_mount_space": space_snapshot(s3_path),
    }


def local_mount_binding(path: Path, *, require_ext4: bool) -> dict[str, Any]:
    executable = shutil.which("findmnt")
    if executable is None:
        raise ScaleProbeError("findmnt is required to bind /local/evan topology")
    completed = subprocess.run(
        [executable, "-n", "-o", "FSTYPE,SOURCE,TARGET", "--target", str(path)],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ScaleProbeError("findmnt failed for local staging mount")
    fields = completed.stdout.strip().split(maxsplit=2)
    if len(fields) != 3:
        raise ScaleProbeError("findmnt local staging output is malformed")
    fstype, source, target = fields
    if require_ext4 and (fstype != "ext4" or target != "/local/evan"):
        raise ScaleProbeError(
            f"production local staging topology drift fstype={fstype} target={target}"
        )
    metadata = require_regular_dir(path)
    space = space_snapshot(path)
    if space.get("status") != "known":
        raise ScaleProbeError("local staging free space is unavailable")
    return {
        "path": str(path),
        "fstype": fstype,
        "mount_target": target,
        "mount_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "st_dev": int(metadata.st_dev),
        "total_bytes": space["total_bytes"],
        "free_bytes": space["free_bytes"],
    }


def load_topology_evidence(
    path: Path, *, expected_jobtag: str, expected_raw_sha256: str
) -> dict[str, Any]:
    require_regular_file(path)
    raw = path.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if raw_sha256 != expected_raw_sha256:
        raise ScaleProbeError("local topology summary raw SHA-256 drift")
    value = json.loads(raw)
    if raw != canonical_json_bytes(value):
        raise ScaleProbeError("local topology summary is not canonical JSON")
    if (
        value.get("schema_version")
        != "offline_arb_local_topology_summary_v1"
        or value.get("status") != "PASS"
        or value.get("jobtag") != expected_jobtag
        or value.get("workers") != PRODUCTION_WORKERS
        or value.get("local_is_node_scoped_by_boot_group") is not True
        or value.get("local_is_shared_across_all_workers") is not False
        or value.get("safe_staging_bytes_min", 0)
        < PRODUCTION_FULL_PARTITION_LOCAL_BUDGET_BYTES
        or value.get("summary_sha256")
        != self_hash(value, "summary_sha256")
    ):
        raise ScaleProbeError("local topology summary contract drift")
    return {
        "path": str(path),
        "jobtag": expected_jobtag,
        "raw_sha256": raw_sha256,
        "summary_sha256": value.get("summary_sha256"),
        "worker_reports_sha256": value.get("worker_reports_sha256"),
        "safe_staging_bytes_min": value["safe_staging_bytes_min"],
        "total_bytes_min": value["total_bytes_min"],
        "available_bytes_min": value["available_bytes_min"],
    }


def lustre_binding(paths: Sequence[Path]) -> dict[str, Any]:
    executable = shutil.which("lfs")
    if executable is None:
        return {"status": "unknown", "reason": "lfs_not_found"}
    results: list[dict[str, Any]] = []
    for path in paths:
        row: dict[str, Any] = {"name": path.name}
        for label, flag in (("stripe_count", "-c"), ("stripe_size", "-S"), ("stripe_index", "-i")):
            try:
                completed = subprocess.run(
                    [executable, "getstripe", flag, str(path)],
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return {
                    "status": "unknown",
                    "reason": f"getstripe_{label}_{type(exc).__name__}",
                }
            if completed.returncode != 0:
                return {
                    "status": "unknown",
                    "reason": f"getstripe_{label}_rc_{completed.returncode}",
                }
            value = completed.stdout.strip()
            if not value:
                return {"status": "unknown", "reason": f"getstripe_{label}_empty"}
            row[label] = value
        results.append(row)
    return {"status": "known", "files": results}


def barrier(
    coord_root: Path,
    *,
    name: str,
    worker_id: int,
    workers: int,
    jobtag: str,
    timeout_seconds: float,
) -> None:
    root = coord_root / "barriers" / name
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    ready = {
        "schema_version": SCHEMA_VERSION,
        "jobtag": jobtag,
        "barrier": name,
        "worker_id": worker_id,
        "timestamp": utc_now(),
    }
    path = root / f"ready.worker-{worker_id}.json"
    if not path.exists():
        write_once_json(path, ready)
        pilot_core.fsync_dir(root)
    deadline = time.monotonic() + timeout_seconds
    go = root / "_GO.json"
    while True:
        failures = sorted(coord_root.glob("FAIL.*.json"))
        if failures:
            raise ScaleProbeError(f"peer failure observed at barrier {name}: {failures[0].name}")
        if go.exists():
            raw = go.read_bytes()
            value = json.loads(raw)
            if value != {
                "schema_version": SCHEMA_VERSION,
                "jobtag": jobtag,
                "barrier": name,
                "workers": workers,
            } or raw != canonical_json_bytes(value):
                raise ScaleProbeError(f"barrier GO binding drift: {name}")
            return
        if worker_id == 0:
            ready_paths = [root / f"ready.worker-{rank}.json" for rank in range(workers)]
            if all(item.exists() for item in ready_paths):
                for rank, item in enumerate(ready_paths):
                    value = json.loads(item.read_bytes())
                    if value.get("jobtag") != jobtag or value.get("barrier") != name or value.get("worker_id") != rank:
                        raise ScaleProbeError(f"barrier ready binding drift: {item}")
                write_once_json(
                    go,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "jobtag": jobtag,
                        "barrier": name,
                        "workers": workers,
                    },
                )
                pilot_core.fsync_dir(root)
                continue
        if time.monotonic() >= deadline:
            raise ScaleProbeError(f"barrier timed out: {name}")
        time.sleep(0.5)


def make_s3_init(
    *,
    jobtag: str,
    workers: int,
    s3_job_root: Path,
    code: dict[str, Any],
    topology_evidence: dict[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": S3_INIT_SCHEMA,
        "jobtag": jobtag,
        "workers": workers,
        "root": str(s3_job_root),
        "code": code,
        "topology_evidence": topology_evidence,
    }
    value["init_sha256"] = self_hash(value, "init_sha256")
    return value


def load_and_validate_s3_init(
    path: Path,
    *,
    jobtag: str,
    workers: int,
    s3_job_root: Path,
    code: dict[str, Any],
    topology_evidence: dict[str, Any],
) -> tuple[dict[str, Any], bytes]:
    before = require_regular_file(path)
    raw = path.read_bytes()
    after = require_regular_file(path)
    if before.st_size != len(raw) or after.st_size != len(raw):
        raise ScaleProbeError("S3 initialization object size drift")
    try:
        value = json.loads(raw)
    except Exception as exc:
        raise ScaleProbeError("S3 initialization object is invalid JSON") from exc
    expected = make_s3_init(
        jobtag=jobtag,
        workers=workers,
        s3_job_root=s3_job_root,
        code=code,
        topology_evidence=topology_evidence,
    )
    if value != expected or raw != canonical_json_bytes(expected):
        raise ScaleProbeError("S3 initialization object binding drift")
    return value, raw


def s3_init_binding(value: dict[str, Any], raw: bytes) -> dict[str, str]:
    if value.get("init_sha256") != self_hash(value, "init_sha256"):
        raise ScaleProbeError("S3 initialization self-hash drift")
    return {
        "s3_init_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "s3_init_sha256": value["init_sha256"],
    }


def make_coord_init(
    *,
    jobtag: str,
    workers: int,
    s3_job_root: Path,
    code: dict[str, Any],
    topology_evidence: dict[str, Any],
    s3_init: dict[str, Any],
    s3_init_raw: bytes,
) -> dict[str, Any]:
    value = {
        "schema_version": COORD_INIT_SCHEMA,
        "jobtag": jobtag,
        "workers": workers,
        "code": code,
        "topology_evidence": topology_evidence,
        "s3_job_root": str(s3_job_root),
        "s3_init_key": str(s3_job_root / "_INIT.json"),
        **s3_init_binding(s3_init, s3_init_raw),
    }
    value["coord_init_sha256"] = self_hash(value, "coord_init_sha256")
    return value


def initialize_roots(
    args: argparse.Namespace,
    code: dict[str, Any],
    topology_evidence: dict[str, Any],
) -> tuple[dict[str, Any], bytes]:
    s3_init_path = args.s3_job_root / "_INIT.json"
    if args.worker_id == 0:
        for path in (args.artifact_root, args.coord_root, args.s3_job_root):
            if path.exists() or path.is_symlink():
                raise ScaleProbeError(f"scale-probe namespace is not fresh: {path}")
        args.artifact_root.mkdir(parents=False, mode=0o700)
        args.coord_root.mkdir(parents=False, mode=0o700)
        args.s3_job_root.mkdir(parents=False, mode=0o700)
        s3_init = make_s3_init(
            jobtag=args.jobtag,
            workers=args.workers,
            s3_job_root=args.s3_job_root,
            code=code,
            topology_evidence=topology_evidence,
        )
        s3_init_raw = canonical_json_bytes(s3_init)
        # Mountpoint-S3 directories are not durable coordination objects and an
        # empty mkdir is not expected to appear in a peer's mount.  Publish a
        # real, immutable key first; only then expose the FSx initialization.
        write_s3_key_once(s3_init_path, s3_init_raw)
        write_once_json(
            args.coord_root / "_INIT.json",
            make_coord_init(
                jobtag=args.jobtag,
                workers=args.workers,
                s3_job_root=args.s3_job_root,
                code=code,
                topology_evidence=topology_evidence,
                s3_init=s3_init,
                s3_init_raw=s3_init_raw,
            ),
        )
        pilot_core.fsync_dir(args.coord_root)
    deadline = time.monotonic() + args.barrier_timeout_seconds
    while True:
        try:
            s3_init_path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ScaleProbeError(
                "unable to inspect S3 initialization object"
            ) from exc
        else:
            s3_init, s3_init_raw = load_and_validate_s3_init(
                s3_init_path,
                jobtag=args.jobtag,
                workers=args.workers,
                s3_job_root=args.s3_job_root,
                code=code,
                topology_evidence=topology_evidence,
            )
            break
        if time.monotonic() >= deadline:
            raise ScaleProbeError("timed out waiting for S3 initialization object")
        time.sleep(0.5)

    init_path = args.coord_root / "_INIT.json"
    while not init_path.exists():
        if time.monotonic() >= deadline:
            raise ScaleProbeError("timed out waiting for FSx initialization object")
        time.sleep(0.5)
    init_raw = init_path.read_bytes()
    try:
        init_value = json.loads(init_raw)
    except Exception as exc:
        raise ScaleProbeError("FSx initialization object is invalid JSON") from exc
    expected_coord_init = make_coord_init(
        jobtag=args.jobtag,
        workers=args.workers,
        s3_job_root=args.s3_job_root,
        code=code,
        topology_evidence=topology_evidence,
        s3_init=s3_init,
        s3_init_raw=s3_init_raw,
    )
    if init_value != expected_coord_init or init_raw != canonical_json_bytes(
        expected_coord_init
    ):
        raise ScaleProbeError("FSx initialization object binding drift")
    require_regular_dir(args.artifact_root)
    require_regular_dir(args.coord_root)
    require_regular_dir(args.s3_job_root)
    if args.s3_worker_root.exists() or args.s3_worker_root.is_symlink():
        raise ScaleProbeError("S3 worker namespace is not fresh")
    args.s3_worker_root.mkdir(mode=0o700)
    if args.local_root.exists() or args.local_root.is_symlink():
        raise ScaleProbeError("local staging namespace is not fresh")
    args.local_root.mkdir(parents=True, mode=0o700)
    return s3_init, s3_init_raw


def wait_for_files(
    paths: Sequence[Path], coord_root: Path, timeout_seconds: float, label: str
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not all(path.exists() for path in paths):
        failures = sorted(coord_root.glob("FAIL.*.json"))
        if failures:
            raise ScaleProbeError(f"peer failure while waiting for {label}: {failures[0].name}")
        if time.monotonic() >= deadline:
            raise ScaleProbeError(f"timed out waiting for {label}")
        time.sleep(0.5)


def execute_phase(
    *,
    name: str,
    role: str,
    items: Sequence[Item],
    combined: bool,
    args: argparse.Namespace,
    telemetry: Telemetry,
    blob_writer: StagedS3BlobWriter,
    index_handle: Any,
    index_digest: Any,
) -> dict[str, Any]:
    telemetry.begin_phase(name)
    started = time.perf_counter_ns()
    bytes_read = 0
    attempts = Counter()
    errors = 0
    ordered_items = sorted(items, key=lambda item: (item.image_path, item.row_id))
    ordered_digest = hashlib.sha256()
    for result, hol_ns in ordered_reads(
        ordered_items,
        workers=args.source_threads,
        attempts=args.source_attempts,
        block_bytes=args.source_block_bytes,
        retry_base_seconds=args.retry_base_seconds,
    ):
        telemetry.add(
            objects=1,
            source_bytes=len(result.payload),
            source_attempts=result.attempts,
            source_errors=result.errors,
            source_open_ns=result.source_open_ns,
            source_read_hash_ns=result.source_read_hash_ns,
            source_total_ns=result.source_total_ns,
            ordered_hol_ns=hol_ns,
        )
        bytes_read += len(result.payload)
        attempts[result.attempts] += 1
        errors += result.errors
        ordered_digest.update(struct.pack("<qqI", result.item.local_index, result.item.row_id, len(result.payload)))
        ordered_digest.update(result.sha256_raw)
        if combined:
            blob_id, offset = blob_writer.append(result.payload)
            record = INDEX_STRUCT.pack(
                result.item.row_id,
                blob_id,
                offset,
                len(result.payload),
                result.sha256_raw,
            )
            before = time.perf_counter_ns()
            pilot_core.write_all(index_handle, record, label="index.bin")
            elapsed = time.perf_counter_ns() - before
            telemetry.add(index_write_ns=elapsed)
            index_digest.update(record)
    elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
    counters = telemetry.phase_snapshot()
    if counters["objects"] != len(items) or counters["source_bytes"] != bytes_read:
        raise ScaleProbeError(f"phase telemetry closure failed: {name}")
    return {
        "name": name,
        "role": role,
        "combined": combined,
        "objects": len(items),
        "source_bytes": bytes_read,
        "elapsed_seconds": elapsed_seconds,
        "objects_per_second": (len(items) / elapsed_seconds) if items else 0.0,
        "mib_per_second": (bytes_read / MIB / elapsed_seconds) if items else 0.0,
        "attempt_histogram": {str(key): value for key, value in sorted(attempts.items())},
        "source_errors": errors,
        "path_order_row_length_content_sha256": ordered_digest.hexdigest(),
        "telemetry_counters": counters,
    }


def precommit_and_approve(
    args: argparse.Namespace,
    *,
    code: dict[str, Any],
    catalog: pilot_core.CatalogBinding,
    selection_file: dict[str, Any],
    phase_reports: Sequence[dict[str, Any]],
    s3_blobs: Sequence[dict[str, Any]],
    local_topology: dict[str, Any],
    topology_evidence: dict[str, Any],
    s3_init: dict[str, Any],
    s3_init_raw: bytes,
    active_root: Path,
) -> tuple[dict[str, Any], bytes]:
    main_phases = [value for value in phase_reports if value["role"] in {"source_only_8", "combined_8"}]
    source_only = [value for value in phase_reports if value["role"] == "source_only_8"]
    combined = [value for value in phase_reports if value["role"] == "combined_8"]
    single_before = next(value for value in phase_reports if value["name"] == "single_before")
    single_after = next(value for value in phase_reports if value["name"] == "single_after")
    precommit = {
        "schema_version": SCHEMA_VERSION,
        "jobtag": args.jobtag,
        "worker_id": args.worker_id,
        "workers": args.workers,
        "active_root": str(active_root),
        "code": code,
        "catalog_hash": catalog.catalog_hash,
        "catalog_rows": catalog.gpic_rows,
        "gpic_row_base": catalog.gpic_row_base,
        "selection_file": selection_file,
        "main_rows": args.main_rows,
        "control_rows": 2 * args.control_rows_per_arm if args.worker_id == 0 else 0,
        "main_source_bytes": sum(value["source_bytes"] for value in main_phases),
        "source_only_main_bytes": sum(value["source_bytes"] for value in source_only),
        "source_only_main_objects": sum(value["objects"] for value in source_only),
        "combined_source_bytes": sum(value["source_bytes"] for value in combined),
        "combined_source_objects": sum(value["objects"] for value in combined),
        "max_s3_bytes_per_worker": args.max_s3_bytes_per_worker,
        "single_before_bytes": single_before["source_bytes"],
        "single_before_objects": single_before["objects"],
        "single_after_bytes": single_after["source_bytes"],
        "single_after_objects": single_after["objects"],
        "phase_names": [value["name"] for value in phase_reports],
        "s3_worker_root": str(args.s3_worker_root),
        "s3_blobs": list(s3_blobs),
        "s3_blob_bytes": sum(value["bytes"] for value in s3_blobs),
        "local_topology": local_topology,
        "topology_evidence": topology_evidence,
        **s3_init_binding(s3_init, s3_init_raw),
        "local_free_floor_bytes": args.local_free_floor_bytes,
        "full_partition_local_budget_bytes": (
            args.full_partition_local_budget_bytes
        ),
    }
    precommit["precommit_sha256"] = self_hash(precommit, "precommit_sha256")
    precommit_path = args.coord_root / f"precommit.worker-{args.worker_id}.json"
    precommit_raw = write_once_json(precommit_path, precommit)
    pilot_core.fsync_dir(args.coord_root)
    paths = [args.coord_root / f"precommit.worker-{rank}.json" for rank in range(args.workers)]
    wait_for_files(paths, args.coord_root, args.barrier_timeout_seconds, "precommit receipts")
    approval_path = args.coord_root / "GLOBAL_APPROVED.json"
    if args.worker_id == 0 and not approval_path.exists():
        values: list[dict[str, Any]] = []
        bindings: list[dict[str, Any]] = []
        main_bytes: list[int] = []
        source_only_bytes: list[int] = []
        source_only_objects: list[int] = []
        combined_bytes: list[int] = []
        combined_objects: list[int] = []
        for rank, path in enumerate(paths):
            raw = path.read_bytes()
            value = json.loads(raw)
            if raw != canonical_json_bytes(value) or value.get("precommit_sha256") != self_hash(value, "precommit_sha256"):
                raise ScaleProbeError(f"precommit self-hash drift worker={rank}")
            if (
                value.get("jobtag") != args.jobtag
                or value.get("worker_id") != rank
                or value.get("workers") != args.workers
                or value.get("code") != code
                or value.get("catalog_hash") != catalog.catalog_hash
                or value.get("catalog_rows") != catalog.gpic_rows
                or value.get("main_rows") != args.main_rows
                or value.get("control_rows") != (2 * args.control_rows_per_arm if rank == 0 else 0)
                or value.get("source_only_main_objects") != args.main_rows // 2
                or value.get("combined_source_objects") != args.main_rows // 2
                or value.get("max_s3_bytes_per_worker")
                != args.max_s3_bytes_per_worker
                or value.get("combined_source_bytes", -1)
                > args.max_s3_bytes_per_worker
                or value.get("s3_worker_root")
                != str(args.s3_job_root / f"worker-{rank}")
                or value.get("s3_blob_bytes")
                != value.get("combined_source_bytes")
                or value.get("local_free_floor_bytes")
                != args.local_free_floor_bytes
                or value.get("full_partition_local_budget_bytes")
                != args.full_partition_local_budget_bytes
                or value.get("topology_evidence") != topology_evidence
                or value.get("s3_init_raw_sha256")
                != hashlib.sha256(s3_init_raw).hexdigest()
                or value.get("s3_init_sha256") != s3_init["init_sha256"]
                or value.get("single_before_objects")
                != (args.control_rows_per_arm if rank == 0 else 0)
                or value.get("single_after_objects")
                != (args.control_rows_per_arm if rank == 0 else 0)
                or value.get("main_source_bytes")
                != value.get("source_only_main_bytes", -1)
                + value.get("combined_source_bytes", -1)
            ):
                raise ScaleProbeError(f"precommit identity drift worker={rank}")
            s3_blobs = value.get("s3_blobs")
            if not isinstance(s3_blobs, list) or not s3_blobs:
                raise ScaleProbeError(f"S3 blob manifest missing worker={rank}")
            for blob_id, blob in enumerate(s3_blobs):
                expected_name = f"blob-{blob_id:06d}.bin"
                expected_key = str(args.s3_job_root / f"worker-{rank}" / expected_name)
                if (
                    blob.get("blob_id") != blob_id
                    or blob.get("name") != expected_name
                    or blob.get("key") != expected_key
                    or type(blob.get("bytes")) is not int
                    or blob["bytes"] <= 0
                    or not pilot_core.is_sha256(blob.get("sha256"))
                ):
                    raise ScaleProbeError(f"S3 blob binding drift worker={rank} blob={blob_id}")
                metadata = require_regular_file(Path(expected_key))
                if metadata.st_size != blob["bytes"]:
                    raise ScaleProbeError(f"S3 blob stat drift worker={rank} blob={blob_id}")
            topology = value.get("local_topology")
            if (
                not isinstance(topology, dict)
                or type(topology.get("st_dev")) is not int
                or type(topology.get("free_bytes")) is not int
                or topology["free_bytes"] < args.local_free_floor_bytes
                or (
                    args.production_shape
                    and (
                        topology.get("fstype") != "ext4"
                        or topology.get("mount_target") != "/local/evan"
                        or topology.get("total_bytes", 0)
                        < topology_evidence["total_bytes_min"]
                    )
                )
            ):
                raise ScaleProbeError(
                    f"local staging topology/free floor drift worker={rank}"
                )
            expected_main, before, after = expected_position_sets(
                total_rows=catalog.gpic_rows,
                worker_id=rank,
                workers=args.workers,
                main_rows=args.main_rows,
                control_rows_per_arm=args.control_rows_per_arm,
                exclude_build_rows=args.exclude_build_rows,
                exclude_scan_rows=args.exclude_scan_rows,
            )
            selection_path = Path(value["active_root"]) / value["selection_file"]["name"]
            if sha256_file(selection_path) != value["selection_file"]["sha256"]:
                raise ScaleProbeError(f"selection-file hash drift worker={rank}")
            expected_positions = [(0, item) for item in expected_main] + [
                (1, item) for item in sorted(before + after)
            ]
            with selection_path.open("rb", buffering=0) as handle:
                for kind, local in expected_positions:
                    record = handle.read(SELECTION_STRUCT.size)
                    if len(record) != SELECTION_STRUCT.size:
                        raise ScaleProbeError(f"selection-file hole worker={rank}")
                    actual_kind, actual_local, row_id, _ = SELECTION_STRUCT.unpack(record)
                    if actual_kind != kind or actual_local != local or row_id != catalog.gpic_row_base + local:
                        raise ScaleProbeError(f"selection identity drift worker={rank}")
                if handle.read(1):
                    raise ScaleProbeError(f"selection-file tail worker={rank}")
            values.append(value)
            bindings.append(
                {
                    "worker_id": rank,
                    "path": str(path),
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "precommit_sha256": value["precommit_sha256"],
                }
            )
            main_bytes.append(value["main_source_bytes"])
            source_only_bytes.append(value["source_only_main_bytes"])
            source_only_objects.append(value["source_only_main_objects"])
            combined_bytes.append(value["combined_source_bytes"])
            combined_objects.append(value["combined_source_objects"])
        if (
            min(main_bytes) <= 0
            or min(source_only_bytes) <= 0
            or min(combined_bytes) <= 0
            or min(source_only_objects) <= 0
            or min(combined_objects) <= 0
        ):
            raise ScaleProbeError("precommit byte totals are empty")
        main_ratio = max(main_bytes) / min(main_bytes)
        combined_ratio = max(combined_bytes) / min(combined_bytes)
        if main_ratio > args.max_byte_imbalance or combined_ratio > args.max_byte_imbalance:
            raise ScaleProbeError(
                f"cross-worker byte imbalance exceeds cap main={main_ratio:.6f} combined={combined_ratio:.6f}"
            )
        source_only_bytes_per_object = sum(source_only_bytes) / sum(source_only_objects)
        combined_bytes_per_object = sum(combined_bytes) / sum(combined_objects)
        role_bytes_per_object_ratio = max(
            source_only_bytes_per_object, combined_bytes_per_object
        ) / min(source_only_bytes_per_object, combined_bytes_per_object)
        if role_bytes_per_object_ratio > args.max_byte_imbalance:
            raise ScaleProbeError(
                "source-only/combined sample byte representativeness exceeds cap "
                f"ratio={role_bytes_per_object_ratio:.6f}"
            )
        worker0 = values[0]
        control_before_bytes_per_object = (
            worker0["single_before_bytes"] / worker0["single_before_objects"]
        )
        control_after_bytes_per_object = (
            worker0["single_after_bytes"] / worker0["single_after_objects"]
        )
        all_role_bpo = [
            source_only_bytes_per_object,
            combined_bytes_per_object,
            control_before_bytes_per_object,
            control_after_bytes_per_object,
        ]
        all_role_ratio = max(all_role_bpo) / min(all_role_bpo)
        if all_role_ratio > args.max_byte_imbalance:
            raise ScaleProbeError(
                "main/control role byte representativeness exceeds cap "
                f"ratio={all_role_ratio:.6f}"
            )
        full_partition_rows = max(
            partition_size(catalog.gpic_rows, rank, args.workers)
            for rank in range(args.workers)
        )
        full_partition_estimate = int(
            (sum(main_bytes) / (args.workers * args.main_rows))
            * full_partition_rows
            * 1.20
        )
        if full_partition_estimate > args.full_partition_local_budget_bytes:
            raise ScaleProbeError(
                "sample-derived full-partition local staging estimate exceeds budget"
            )
        approval = {
            "schema_version": GLOBAL_APPROVAL_SCHEMA,
            "generated_at": utc_now(),
            "jobtag": args.jobtag,
            "workers": args.workers,
            "main_rows_per_worker": args.main_rows,
            "main_rows_union": args.workers * args.main_rows,
            "partition": "catalog_local_index_mod_workers",
            "pilot_exclusion_union_rows": args.exclude_build_rows + args.exclude_scan_rows,
            "byte_imbalance_cap": args.max_byte_imbalance,
            "main_byte_max_over_min": main_ratio,
            "combined_byte_max_over_min": combined_ratio,
            "s3_payload_bytes_observed": sum(combined_bytes),
            "s3_payload_hard_cap_bytes": (
                args.workers * args.max_s3_bytes_per_worker
            ),
            "source_only_bytes_per_object": source_only_bytes_per_object,
            "combined_bytes_per_object": combined_bytes_per_object,
            "role_bytes_per_object_max_over_min": role_bytes_per_object_ratio,
            "control_before_bytes_per_object": control_before_bytes_per_object,
            "control_after_bytes_per_object": control_after_bytes_per_object,
            "all_role_bytes_per_object_max_over_min": all_role_ratio,
            "local_free_floor_bytes": args.local_free_floor_bytes,
            "full_partition_rows_max": full_partition_rows,
            "full_partition_local_estimate_bytes_x1_2": full_partition_estimate,
            "full_partition_local_budget_bytes": (
                args.full_partition_local_budget_bytes
            ),
            "local_topologies": [item["local_topology"] for item in values],
            "topology_evidence": topology_evidence,
            **s3_init_binding(s3_init, s3_init_raw),
            "precommits": bindings,
            "code": code,
        }
        approval["approval_sha256"] = self_hash(approval, "approval_sha256")
        write_once_json(approval_path, approval)
        pilot_core.fsync_dir(args.coord_root)
    wait_for_files([approval_path], args.coord_root, args.barrier_timeout_seconds, "global approval")
    raw = approval_path.read_bytes()
    value = json.loads(raw)
    if (
        raw != canonical_json_bytes(value)
        or value.get("approval_sha256") != self_hash(value, "approval_sha256")
        or value.get("jobtag") != args.jobtag
        or value.get("workers") != args.workers
        or value.get("main_rows_union") != args.workers * args.main_rows
        or value.get("code") != code
        or value.get("s3_init_raw_sha256")
        != hashlib.sha256(s3_init_raw).hexdigest()
        or value.get("s3_init_sha256") != s3_init["init_sha256"]
    ):
        raise ScaleProbeError("global approval binding drift")
    binding = value["precommits"][args.worker_id]
    if binding.get("raw_sha256") != hashlib.sha256(precommit_raw).hexdigest():
        raise ScaleProbeError("global approval does not bind local precommit")
    return value, raw


def write_s3_key_once(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ScaleProbeError(f"Mountpoint-S3 final key is not fresh: {path}")
    handle: Any = None
    try:
        handle = path.open("xb", buffering=0)
        pilot_core.write_all(handle, payload, label=f"S3 key {path.name}")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
    except BaseException as exc:
        if handle is not None:
            try:
                handle.close()
            except BaseException:
                pass
        try:
            path.lstat()
        except FileNotFoundError:
            raise
        except BaseException as inspect_exc:
            raise pilot_core.CommittedSyncUnconfirmed(
                f"S3 key visibility is unknowable after write error: {path.name}"
            ) from inspect_exc
        raise pilot_core.CommittedSyncUnconfirmed(
            f"S3 key became visible while write reported failure: {path.name}"
        ) from exc
    metadata = require_regular_file(path)
    if metadata.st_size != len(payload):
        raise pilot_core.CommittedSyncUnconfirmed(
            f"S3 key size is unconfirmed after fsync: {path.name}"
        )


def publish_s3_worker_manifest(
    args: argparse.Namespace,
    *,
    code: dict[str, Any],
    catalog: pilot_core.CatalogBinding,
    approval: dict[str, Any],
    approval_raw: bytes,
    blobs: Sequence[dict[str, Any]],
    s3_init: dict[str, Any],
    s3_init_raw: bytes,
) -> tuple[dict[str, Any], bytes]:
    manifest = {
        "schema_version": "offline_gpic_scale_s3_worker_manifest_v1",
        "generated_at": utc_now(),
        "jobtag": args.jobtag,
        "worker_id": args.worker_id,
        "workers": args.workers,
        "root": str(args.s3_worker_root),
        "catalog_hash": catalog.catalog_hash,
        "catalog_rows": catalog.gpic_rows,
        "global_approval_raw_sha256": hashlib.sha256(approval_raw).hexdigest(),
        "global_approval_sha256": approval["approval_sha256"],
        **s3_init_binding(s3_init, s3_init_raw),
        "blobs": list(blobs),
        "bytes": sum(value["bytes"] for value in blobs),
        "code": code,
        "publication_semantics": (
            "mountpoint_s3_write_once_worker_keys_file_fsync_manifest_then_success_last"
        ),
    }
    manifest["manifest_sha256"] = self_hash(manifest, "manifest_sha256")
    raw = canonical_json_bytes(manifest)
    write_s3_key_once(args.s3_worker_root / "manifest.json", raw)
    write_s3_key_once(
        args.s3_worker_root / "_SUCCESS",
        (manifest["manifest_sha256"] + "\n").encode("ascii"),
    )
    return manifest, raw


def publish_worker(
    args: argparse.Namespace,
    *,
    active_root: Path,
    receipt: dict[str, Any],
) -> tuple[str, bytes]:
    receipt["receipt_sha256"] = self_hash(receipt, "receipt_sha256")
    receipt_raw = write_once_json(active_root / "receipt.json", receipt)
    compact_bytes = sum(
        require_regular_file(path).st_size for path in active_root.iterdir()
    )
    if compact_bytes > args.max_fsx_evidence_bytes_per_worker:
        raise ScaleProbeError("published FSx worker evidence exceeds its hard cap")
    pilot_core.write_once_bytes(
        active_root / "_SUCCESS",
        (receipt["receipt_sha256"] + "\n").encode("ascii"),
        fsync_file=True,
    )
    pilot_core.fsync_dir(active_root)
    output = args.artifact_root / f"worker-{args.worker_id}"
    if output.exists() or output.is_symlink():
        raise ScaleProbeError("worker output appeared before publication")
    method = pilot_core.publish_directory_noreplace(active_root, output, success_last=True)
    return method, receipt_raw


def close_job(
    args: argparse.Namespace,
    code: dict[str, Any],
    s3_init: dict[str, Any],
    s3_init_raw: bytes,
) -> None:
    paths = [args.coord_root / f"published.worker-{rank}.json" for rank in range(args.workers)]
    wait_for_files(paths, args.coord_root, args.barrier_timeout_seconds, "worker publications")
    top_success = args.artifact_root / "_SUCCESS"
    if args.worker_id == 0 and not top_success.exists():
        workers: list[dict[str, Any]] = []
        for rank, status_path in enumerate(paths):
            status_raw = status_path.read_bytes()
            status = json.loads(status_raw)
            root = args.artifact_root / f"worker-{rank}"
            receipt_raw = (root / "receipt.json").read_bytes()
            receipt = json.loads(receipt_raw)
            marker = (root / "_SUCCESS").read_text(encoding="ascii").strip()
            s3_root = args.s3_job_root / f"worker-{rank}"
            s3_manifest_raw = (s3_root / "manifest.json").read_bytes()
            s3_manifest = json.loads(s3_manifest_raw)
            s3_marker = (s3_root / "_SUCCESS").read_text(encoding="ascii").strip()
            if (
                status_raw != canonical_json_bytes(status)
                or status.get("jobtag") != args.jobtag
                or status.get("worker_id") != rank
                or status.get("worker_receipt_raw_sha256") != hashlib.sha256(receipt_raw).hexdigest()
                or receipt_raw != canonical_json_bytes(receipt)
                or receipt.get("receipt_sha256") != self_hash(receipt, "receipt_sha256")
                or marker != receipt["receipt_sha256"]
                or s3_manifest_raw != canonical_json_bytes(s3_manifest)
                or s3_manifest.get("manifest_sha256")
                != self_hash(s3_manifest, "manifest_sha256")
                or s3_marker != s3_manifest["manifest_sha256"]
                or status.get("s3_worker_manifest_raw_sha256")
                != hashlib.sha256(s3_manifest_raw).hexdigest()
                or status.get("s3_worker_manifest_sha256")
                != s3_manifest["manifest_sha256"]
                or status.get("s3_init_raw_sha256")
                != hashlib.sha256(s3_init_raw).hexdigest()
                or status.get("s3_init_sha256") != s3_init["init_sha256"]
                or s3_manifest.get("s3_init_raw_sha256")
                != hashlib.sha256(s3_init_raw).hexdigest()
                or s3_manifest.get("s3_init_sha256")
                != s3_init["init_sha256"]
                or receipt.get("storage", {}).get("s3_init_raw_sha256")
                != hashlib.sha256(s3_init_raw).hexdigest()
                or receipt.get("storage", {}).get("s3_init_sha256")
                != s3_init["init_sha256"]
            ):
                raise ScaleProbeError(f"published worker binding drift worker={rank}")
            workers.append(
                {
                    "worker_id": rank,
                    "root": str(root),
                    "receipt_raw_sha256": hashlib.sha256(receipt_raw).hexdigest(),
                    "receipt_sha256": receipt["receipt_sha256"],
                    "published_status_raw_sha256": hashlib.sha256(status_raw).hexdigest(),
                    "publication_method": status["publication_method"],
                    "s3_root": str(s3_root),
                    "s3_manifest_raw_sha256": hashlib.sha256(
                        s3_manifest_raw
                    ).hexdigest(),
                    "s3_manifest_sha256": s3_manifest["manifest_sha256"],
                    "s3_bytes": s3_manifest["bytes"],
                }
            )
        approval_raw = (args.coord_root / "GLOBAL_APPROVED.json").read_bytes()
        s3_top = {
            "schema_version": "offline_gpic_scale_s3_job_manifest_v1",
            "generated_at": utc_now(),
            "jobtag": args.jobtag,
            "root": str(args.s3_job_root),
            "worker_count": args.workers,
            "workers": [
                {
                    "worker_id": value["worker_id"],
                    "root": value["s3_root"],
                    "manifest_raw_sha256": value["s3_manifest_raw_sha256"],
                    "manifest_sha256": value["s3_manifest_sha256"],
                    "bytes": value["s3_bytes"],
                }
                for value in workers
            ],
            "bytes": sum(value["s3_bytes"] for value in workers),
            "global_approval_raw_sha256": hashlib.sha256(approval_raw).hexdigest(),
            **s3_init_binding(s3_init, s3_init_raw),
            "code": code,
            "publication_semantics": (
                "mountpoint_s3_worker_successes_then_job_manifest_and_success_last"
            ),
        }
        s3_top["manifest_sha256"] = self_hash(s3_top, "manifest_sha256")
        s3_top_raw = canonical_json_bytes(s3_top)
        write_s3_key_once(args.s3_job_root / "manifest.json", s3_top_raw)
        write_s3_key_once(
            args.s3_job_root / "_SUCCESS",
            (s3_top["manifest_sha256"] + "\n").encode("ascii"),
        )
        top = {
            "schema_version": JOB_RECEIPT_SCHEMA,
            "generated_at": utc_now(),
            "jobtag": args.jobtag,
            "workers": workers,
            "worker_count": args.workers,
            "global_approval_raw_sha256": hashlib.sha256(approval_raw).hexdigest(),
            **s3_init_binding(s3_init, s3_init_raw),
            "s3_job_root": str(args.s3_job_root),
            "s3_job_manifest_raw_sha256": hashlib.sha256(s3_top_raw).hexdigest(),
            "s3_job_manifest_sha256": s3_top["manifest_sha256"],
            "s3_payload_bytes": s3_top["bytes"],
            "code": code,
        }
        top["job_receipt_sha256"] = self_hash(top, "job_receipt_sha256")
        write_once_json(args.artifact_root / "job_receipt.json", top)
        pilot_core.fsync_dir(args.artifact_root)
        write_once_atomic_bytes(
            top_success,
            (top["job_receipt_sha256"] + "\n").encode("ascii"),
            fsync_file=True,
        )
        pilot_core.fsync_dir(args.artifact_root)
        pilot_core.fsync_dir(args.artifact_root.parent)
    wait_for_files([top_success], args.coord_root, args.barrier_timeout_seconds, "job publication")


def publication_marker_committed_or_unknown(args: argparse.Namespace) -> bool:
    committed_or_unknown = False
    for marker in (
        args.artifact_root / f"worker-{args.worker_id}" / "_SUCCESS",
        args.s3_worker_root / "_SUCCESS",
        args.s3_job_root / "_SUCCESS",
    ):
        try:
            marker.lstat()
            committed_or_unknown = True
        except FileNotFoundError:
            pass
        except BaseException:
            committed_or_unknown = True
    return committed_or_unknown


def run(args: argparse.Namespace) -> None:
    code = code_snapshot(args.job_spec)
    topology_evidence = load_topology_evidence(
        args.topology_summary,
        expected_jobtag=args.topology_jobtag,
        expected_raw_sha256=args.topology_summary_raw_sha256,
    )
    s3_init, s3_init_raw = initialize_roots(args, code, topology_evidence)
    local_topology = local_mount_binding(
        args.local_prefix, require_ext4=args.production_shape
    )
    if (
        local_topology["free_bytes"] < args.local_free_floor_bytes
        or local_topology["total_bytes"]
        < args.full_partition_local_budget_bytes
    ):
        raise ScaleProbeError("live local topology does not meet free/budget floor")
    barrier(
        args.coord_root,
        name="initialized",
        worker_id=args.worker_id,
        workers=args.workers,
        jobtag=args.jobtag,
        timeout_seconds=args.barrier_timeout_seconds,
    )
    catalog = pilot_core.load_catalog(args.catalog_root, verify_gpic_sha256=False)
    main_positions, control_before_positions, control_after_positions = expected_position_sets(
        total_rows=catalog.gpic_rows,
        worker_id=args.worker_id,
        workers=args.workers,
        main_rows=args.main_rows,
        control_rows_per_arm=args.control_rows_per_arm,
        exclude_build_rows=args.exclude_build_rows,
        exclude_scan_rows=args.exclude_scan_rows,
    )
    main = select_catalog_items(catalog, main_positions, source_prefix=args.source_prefix)
    control_before = (
        select_catalog_items(catalog, control_before_positions, source_prefix=args.source_prefix)
        if control_before_positions
        else []
    )
    control_after = (
        select_catalog_items(catalog, control_after_positions, source_prefix=args.source_prefix)
        if control_after_positions
        else []
    )
    controls = sorted(control_before + control_after, key=lambda item: item.local_index)
    active_root = args.artifact_root / f".worker-{args.worker_id}.work.{uuid.uuid4().hex}"
    active_root.mkdir(mode=0o700)
    telemetry = Telemetry(active_root / "telemetry.jsonl", args.telemetry_interval_seconds)
    resources_at_start = process_stats(args.local_root, args.s3_job_root)
    writer = StagedS3BlobWriter(
        args.local_root,
        args.s3_worker_root,
        args.blob_target_bytes,
        telemetry,
        args.max_s3_bytes_per_worker,
        args.max_local_staging_bytes_per_worker,
    )
    index_path = active_root / "index.bin"
    index_handle = index_path.open("xb", buffering=0)
    index_digest = hashlib.sha256()
    phase_reports: list[dict[str, Any]] = []
    selection_file: dict[str, Any] | None = None
    blobs: list[dict[str, Any]] | None = None
    index_closed = False
    try:
        selection_file = write_selection_file(active_root / "selection.bin", main, controls)
        main_arms = [main[index::4] for index in range(4)]
        for name, role, combined, arm in PHASES:
            barrier(
                args.coord_root,
                name=f"{name}.start",
                worker_id=args.worker_id,
                workers=args.workers,
                jobtag=args.jobtag,
                timeout_seconds=args.barrier_timeout_seconds,
            )
            if role == "single_control":
                items = (control_before if arm == 0 else control_after) if args.worker_id == 0 else []
            else:
                items = main_arms[arm]
            report = execute_phase(
                name=name,
                role=role,
                items=items,
                combined=combined,
                args=args,
                telemetry=telemetry,
                blob_writer=writer,
                index_handle=index_handle,
                index_digest=index_digest,
            )
            if name == "combined_b2":
                finalize_started = time.perf_counter_ns()
                blobs = writer.close()
                index_handle.flush()
                before_fsync = time.perf_counter_ns()
                os.fsync(index_handle.fileno())
                index_fsync_ns = time.perf_counter_ns() - before_fsync
                telemetry.add(index_fsync_ns=index_fsync_ns)
                index_handle.close()
                index_closed = True
                report["elapsed_seconds"] += (time.perf_counter_ns() - finalize_started) / 1e9
                report["objects_per_second"] = report["objects"] / report["elapsed_seconds"]
                report["mib_per_second"] = report["source_bytes"] / MIB / report["elapsed_seconds"]
                report["telemetry_counters"] = telemetry.phase_snapshot()
            telemetry.end_phase()
            phase_reports.append(report)
            barrier(
                args.coord_root,
                name=f"{name}.end",
                worker_id=args.worker_id,
                workers=args.workers,
                jobtag=args.jobtag,
                timeout_seconds=args.barrier_timeout_seconds,
            )
        if blobs is None or not index_closed:
            raise ScaleProbeError("combined blob/index finalization did not run in B2")
        if any(args.local_root.iterdir()):
            raise ScaleProbeError("local staging root is not empty after S3 upload")
        args.local_root.rmdir()
        telemetry.close()
        index_file = {
            "name": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": index_digest.hexdigest(),
            "records": sum(value["objects"] for value in phase_reports if value["combined"]),
        }
        if index_file["bytes"] != index_file["records"] * INDEX_STRUCT.size:
            raise ScaleProbeError("fixed-width index did not close")
        telemetry_file = {
            "name": "telemetry.jsonl",
            "bytes": (active_root / "telemetry.jsonl").stat().st_size,
            "sha256": sha256_file(active_root / "telemetry.jsonl"),
        }
        fsx_evidence_bytes = (
            selection_file["bytes"]
            + index_file["bytes"]
            + telemetry_file["bytes"]
        )
        if fsx_evidence_bytes > args.max_fsx_evidence_bytes_per_worker:
            raise ScaleProbeError(
                "compact FSx evidence exceeds the per-worker hard cap"
            )
        approval, approval_raw = precommit_and_approve(
            args,
            code=code,
            catalog=catalog,
            selection_file=selection_file,
            phase_reports=phase_reports,
            s3_blobs=blobs,
            local_topology=local_topology,
            topology_evidence=topology_evidence,
            s3_init=s3_init,
            s3_init_raw=s3_init_raw,
            active_root=active_root,
        )
        pilot_core.validate_catalog_closure(catalog)
        validate_code_snapshot(code)
        s3_manifest, s3_manifest_raw = publish_s3_worker_manifest(
            args,
            code=code,
            catalog=catalog,
            approval=approval,
            approval_raw=approval_raw,
            blobs=blobs,
            s3_init=s3_init,
            s3_init_raw=s3_init_raw,
        )
        fsx_paths = [
            active_root / selection_file["name"],
            index_path,
            active_root / telemetry_file["name"],
        ]
        receipt = {
            "schema_version": WORKER_RECEIPT_SCHEMA,
            "generated_at": utc_now(),
            "jobtag": args.jobtag,
            "worker_id": args.worker_id,
            "workers": args.workers,
            "production_shape": args.production_shape,
            "shape_mode": (
                "quick_100k_per_arm"
                if args.production_shape
                else "synthetic_or_custom"
            ),
            "partition": "catalog_local_index_mod_workers",
            "source_order": "path_lexicographic_within_phase",
            "source_prefix": args.source_prefix,
            "main_rows": len(main),
            "control_rows": len(controls),
            "sample_rows_per_worker_per_arm": len(main) // 4,
            "main_total_rows_all_workers": args.workers * len(main),
            "combined_rows_per_worker": len(main) // 2,
            "combined_total_rows_all_workers": args.workers * len(main) // 2,
            "single_control_total_rows": 2 * args.control_rows_per_arm,
            "all_source_rows_including_controls": (
                args.workers * len(main) + 2 * args.control_rows_per_arm
            ),
            "main_selection_sha256": selection_digest(main),
            "control_selection_sha256": selection_digest(controls) if controls else None,
            "pilot_exclusion": {
                "build_rows": args.exclude_build_rows,
                "scan_rows": args.exclude_scan_rows,
                "union_rows": args.exclude_build_rows + args.exclude_scan_rows,
            },
            "catalog": pilot_core.catalog_manifest(catalog),
            "code": code,
            "runtime": pilot_core.runtime_identity(),
            "topology_evidence": topology_evidence,
            "local_topology_live": local_topology,
            "resources": {
                "before_phases": resources_at_start,
                "after_phases": process_stats(args.local_prefix, args.s3_job_root),
            },
            "lustre": lustre_binding(fsx_paths),
            "global_approval_raw_sha256": hashlib.sha256(approval_raw).hexdigest(),
            "global_approval_sha256": approval["approval_sha256"],
            "phase_reports": phase_reports,
            "source": {
                "threads": args.source_threads,
                "attempt_limit": args.source_attempts,
                "block_bytes": args.source_block_bytes,
            },
            "layout": {
                "blob_target_bytes": args.blob_target_bytes,
                "max_s3_bytes_per_worker": args.max_s3_bytes_per_worker,
                "max_s3_bytes_all_workers": (
                    args.workers * args.max_s3_bytes_per_worker
                ),
                "max_local_staging_bytes_per_worker": (
                    args.max_local_staging_bytes_per_worker
                ),
                "max_fsx_evidence_bytes_per_worker": (
                    args.max_fsx_evidence_bytes_per_worker
                ),
                "local_free_floor_bytes": args.local_free_floor_bytes,
                "full_partition_local_budget_bytes": (
                    args.full_partition_local_budget_bytes
                ),
                "index_struct": INDEX_STRUCT.format,
                "index_record_bytes": INDEX_STRUCT.size,
                "selection_struct": SELECTION_STRUCT.format,
                "selection_record_bytes": SELECTION_STRUCT.size,
                "compression": "none",
            },
            "storage": {
                "payload_backend": "mountpoint_s3",
                "local_staging_root": str(args.local_root),
                "local_staging_empty_and_removed": True,
                "local_staging_peak_bytes": writer.local_peak_bytes,
                "fsx_compact_evidence_bytes": fsx_evidence_bytes,
                "s3_worker_root": str(args.s3_worker_root),
                "s3_worker_manifest_raw_sha256": hashlib.sha256(
                    s3_manifest_raw
                ).hexdigest(),
                "s3_worker_manifest_sha256": s3_manifest["manifest_sha256"],
                "s3_payload_bytes": s3_manifest["bytes"],
                **s3_init_binding(s3_init, s3_init_raw),
            },
            "files": {
                "selection": selection_file,
                "index": index_file,
                "telemetry": telemetry_file,
            },
            "s3_blobs": blobs,
            "publication_semantics": (
                "s3_worker_manifest_success_last_then_fsx_compact_receipt_noreplace"
            ),
        }
        method, receipt_raw = publish_worker(args, active_root=active_root, receipt=receipt)
        published = {
            "schema_version": SCHEMA_VERSION,
            "jobtag": args.jobtag,
            "worker_id": args.worker_id,
            "worker_receipt_raw_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "worker_receipt_sha256": receipt["receipt_sha256"],
            "publication_method": method,
            "s3_worker_manifest_raw_sha256": hashlib.sha256(
                s3_manifest_raw
            ).hexdigest(),
            "s3_worker_manifest_sha256": s3_manifest["manifest_sha256"],
            **s3_init_binding(s3_init, s3_init_raw),
        }
        write_once_json(args.coord_root / f"published.worker-{args.worker_id}.json", published)
        pilot_core.fsync_dir(args.coord_root)
        close_job(args, code, s3_init, s3_init_raw)
        print(
            f"OFFLINE_GPIC_SCALE_WORKER_PASS jobtag={args.jobtag} worker={args.worker_id} "
            f"root={args.artifact_root / f'worker-{args.worker_id}'}",
            flush=True,
        )
    except BaseException as exc:
        try:
            if not index_handle.closed:
                index_handle.close()
            writer.abort()
        except BaseException:
            pass
        marker_state_committed_or_unknown = publication_marker_committed_or_unknown(
            args
        )
        if (
            isinstance(exc, pilot_core.CommittedSyncUnconfirmed)
            or marker_state_committed_or_unknown
        ):
            print(
                f"OFFLINE_GPIC_SCALE_COMMITTED_SYNC_UNCONFIRMED worker={args.worker_id}",
                file=sys.stderr,
                flush=True,
            )
        else:
            try:
                failure = {
                    "schema_version": SCHEMA_VERSION,
                    "jobtag": args.jobtag,
                    "worker_id": args.worker_id,
                    "exception_class": type(exc).__name__,
                    "timestamp": utc_now(),
                }
                write_once_json(
                    args.coord_root / f"FAIL.worker-{args.worker_id}.{uuid.uuid4().hex}.json",
                    failure,
                )
                pilot_core.fsync_dir(args.coord_root)
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
        "--topology-summary", type=Path, default=PRODUCTION_TOPOLOGY_SUMMARY
    )
    parser.add_argument(
        "--topology-jobtag", default=PRODUCTION_TOPOLOGY_JOBTAG
    )
    parser.add_argument(
        "--topology-summary-raw-sha256",
        default=PRODUCTION_TOPOLOGY_RAW_SHA256,
    )
    parser.add_argument("--jobtag", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--workers", type=int, default=PRODUCTION_WORKERS)
    parser.add_argument("--main-rows", type=int, default=PRODUCTION_MAIN_ROWS)
    parser.add_argument("--control-rows-per-arm", type=int, default=PRODUCTION_CONTROL_ROWS_PER_ARM)
    parser.add_argument("--exclude-build-rows", type=int, default=PRODUCTION_EXCLUDE_BUILD_ROWS)
    parser.add_argument("--exclude-scan-rows", type=int, default=PRODUCTION_EXCLUDE_SCAN_ROWS)
    parser.add_argument("--source-prefix", default=PRODUCTION_SOURCE_PREFIX)
    parser.add_argument("--source-threads", type=int, default=32)
    parser.add_argument("--source-attempts", type=int, default=3)
    parser.add_argument("--source-block-bytes", type=int, default=1 << 20)
    parser.add_argument("--retry-base-seconds", type=float, default=0.25)
    parser.add_argument("--blob-target-bytes", type=int, default=PRODUCTION_BLOB_BYTES)
    parser.add_argument(
        "--max-s3-bytes-per-worker",
        type=int,
        default=PRODUCTION_MAX_S3_BYTES_PER_WORKER,
    )
    parser.add_argument(
        "--max-local-staging-bytes-per-worker",
        type=int,
        default=PRODUCTION_MAX_LOCAL_STAGING_BYTES_PER_WORKER,
    )
    parser.add_argument(
        "--max-fsx-evidence-bytes-per-worker",
        type=int,
        default=PRODUCTION_MAX_FSX_EVIDENCE_BYTES_PER_WORKER,
    )
    parser.add_argument(
        "--local-free-floor-bytes",
        type=int,
        default=PRODUCTION_LOCAL_FREE_FLOOR_BYTES,
    )
    parser.add_argument(
        "--full-partition-local-budget-bytes",
        type=int,
        default=PRODUCTION_FULL_PARTITION_LOCAL_BUDGET_BYTES,
    )
    parser.add_argument("--telemetry-interval-seconds", type=float, default=60.0)
    parser.add_argument("--barrier-timeout-seconds", type=float, default=10_800.0)
    parser.add_argument("--max-byte-imbalance", type=float, default=1.10)
    parser.add_argument("--write-authority", required=True)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--production-shape", action="store_true")
    args = parser.parse_args(argv)
    args.artifact_prefix = args.artifact_prefix.expanduser().resolve(strict=True)
    args.coord_prefix = args.coord_prefix.expanduser().resolve(strict=True)
    args.local_prefix = args.local_prefix.expanduser().resolve(strict=True)
    args.s3_prefix = args.s3_prefix.expanduser().resolve(strict=True)
    args.topology_summary = args.topology_summary.expanduser().resolve(strict=True)
    args.catalog_root = args.catalog_root.expanduser().resolve(strict=True)
    args.job_spec = args.job_spec.expanduser().resolve(strict=True)
    if args.write_authority != WRITE_AUTHORITY:
        raise ScaleProbeError("explicit scale-probe write authority is absent")
    if not str(args.artifact_prefix).startswith("/fsx/") or not str(args.coord_prefix).startswith("/fsx/"):
        raise ScaleProbeError("scale-probe outputs must be on FSx")
    if args.artifact_prefix == args.coord_prefix:
        raise ScaleProbeError("artifact and coordination prefixes must differ")
    if args.production_shape and not str(args.local_prefix).startswith("/local/evan"):
        raise ScaleProbeError("production local staging must be under /local/evan")
    if args.production_shape and not (
        str(args.s3_prefix).startswith("/s3/evan/")
        or str(args.s3_prefix).startswith("/s3-pixai-test-uw2/evan/")
    ):
        raise ScaleProbeError("production S3 prefix is not an approved Evan mount")
    if not args.jobtag or Path(args.jobtag).name != args.jobtag or args.jobtag in {".", ".."}:
        raise ScaleProbeError("jobtag is malformed")
    if not (0 <= args.worker_id < args.workers) or args.workers < 1:
        raise ScaleProbeError("worker identity is malformed")
    for label in (
        "main_rows",
        "exclude_build_rows",
        "exclude_scan_rows",
        "source_threads",
        "source_attempts",
        "source_block_bytes",
        "blob_target_bytes",
        "max_s3_bytes_per_worker",
        "max_local_staging_bytes_per_worker",
        "max_fsx_evidence_bytes_per_worker",
        "local_free_floor_bytes",
        "full_partition_local_budget_bytes",
    ):
        if getattr(args, label) < 1:
            raise ScaleProbeError(f"{label} must be positive")
    if args.control_rows_per_arm < 1:
        raise ScaleProbeError("control_rows_per_arm must be positive")
    if args.telemetry_interval_seconds <= 0 or args.barrier_timeout_seconds <= 0:
        raise ScaleProbeError("telemetry/barrier interval must be positive")
    if not (1.0 <= args.max_byte_imbalance <= 1.10):
        raise ScaleProbeError("byte imbalance cap must be in [1.0, 1.10]")
    if not args.source_prefix.startswith("/") or not args.source_prefix.endswith("/"):
        raise ScaleProbeError("source prefix must be an absolute directory prefix")
    if args.production_shape and (
        args.workers != PRODUCTION_WORKERS
        or args.main_rows != PRODUCTION_MAIN_ROWS
        or args.control_rows_per_arm != PRODUCTION_CONTROL_ROWS_PER_ARM
        or args.exclude_build_rows != PRODUCTION_EXCLUDE_BUILD_ROWS
        or args.exclude_scan_rows != PRODUCTION_EXCLUDE_SCAN_ROWS
        or args.blob_target_bytes != PRODUCTION_BLOB_BYTES
        or args.max_s3_bytes_per_worker != PRODUCTION_MAX_S3_BYTES_PER_WORKER
        or args.max_local_staging_bytes_per_worker
        != PRODUCTION_MAX_LOCAL_STAGING_BYTES_PER_WORKER
        or args.max_fsx_evidence_bytes_per_worker
        != PRODUCTION_MAX_FSX_EVIDENCE_BYTES_PER_WORKER
        or args.local_free_floor_bytes != PRODUCTION_LOCAL_FREE_FLOOR_BYTES
        or args.full_partition_local_budget_bytes
        != PRODUCTION_FULL_PARTITION_LOCAL_BUDGET_BYTES
        or args.topology_jobtag != PRODUCTION_TOPOLOGY_JOBTAG
        or args.topology_summary_raw_sha256
        != PRODUCTION_TOPOLOGY_RAW_SHA256
        or args.topology_summary != PRODUCTION_TOPOLOGY_SUMMARY
        or args.source_prefix != PRODUCTION_SOURCE_PREFIX
        or args.telemetry_interval_seconds != 60.0
        or args.max_byte_imbalance != 1.10
    ):
        raise ScaleProbeError("production-shape constants drift")
    args.artifact_root = args.artifact_prefix / args.jobtag
    args.coord_root = args.coord_prefix / args.jobtag
    args.local_root = args.local_prefix / args.jobtag / f"worker-{args.worker_id}"
    args.s3_job_root = args.s3_prefix / args.jobtag
    args.s3_worker_root = args.s3_job_root / f"worker-{args.worker_id}"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
