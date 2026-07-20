#!/usr/bin/env python3
"""Build and replay a bounded ``offline_gpic_blob_v1`` pilot.

This is deliberately not wired into the training loader.  It answers two
questions with immutable evidence:

* how fast a CPU16 JCC worker can read a catalog-stratified GPIC sample from
  the source mount with different concurrency levels; and
* how fast the exact same concatenated bytes can be randomly ``pread`` from
  a Lustre target and from a writable Mountpoint-S3 target.

The POSIX publisher uses fsync plus renameat2(NOREPLACE) when supported; on
Lustre EINVAL/unsupported it atomically claims the directory name, hard-links
regular files without replacement, and links ``_SUCCESS`` last.
The Mountpoint-S3 publisher never renames or directory-fsyncs: it creates a
fresh, caller-authorized namespace, sequentially writes and file-fsyncs each
final key once, and writes ``_SUCCESS`` last.  A failed S3 build can leave
objects only under that unique pilot namespace and can never look published.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import dataclasses
import errno
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import random
import re
import queue
import shutil
import stat
import struct
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow
import pyarrow.parquet as pq


SCHEMA_VERSION = "offline_gpic_blob_v1_pilot"
LAYOUT_VERSION = "offline_gpic_blob_index_le_v1"
CATALOG_SCHEMA_VERSION = "offline_arb_catalog_v1"
SELECTION_METHOD = "catalog_stratified_midpoint_v1"
INDEX_STRUCT = struct.Struct("<qIQI32s")
INDEX_RECORD_BYTES = INDEX_STRUCT.size
SOURCE_PREFIX = "/s3-mewtant-raw-datasets/stanford-vision-lab/gpic_umziped/images/train/"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SHARD_RE = re.compile(r"[0-9]{5}")
BOOT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
MIB = 1 << 20
REPLAY_COLD_REGION_BYTES = 16 * MIB
REPLAY_POTENTIAL_READAHEAD_BYTES = 8 * MIB
AT_FDCWD = -100
RENAME_NOREPLACE = 1
DEFAULT_CATALOG_ROOT = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_arb_catalog_v1"
)
WRITE_AUTHORITY = "offline-gpic-blob-v1-pilot-write-authorized"
_FALLBACK_BEFORE_DIRECTORY_CLAIM_HOOK: Any = None
_POST_COMMIT_DURABILITY_HOOK: Any = None


class PilotError(RuntimeError):
    """Fail-closed pilot contract error."""


class CommittedSyncUnconfirmed(PilotError):
    """Publication marker is visible, but post-commit completion is unconfirmed.

    This state is monotonic: callers must never remove ``_SUCCESS`` or add a
    failure marker/failed-root after observing this exception.
    """


@dataclasses.dataclass(frozen=True)
class FileIdentity:
    dev: int
    ino: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "FileIdentity":
        return cls(
            dev=int(value.st_dev),
            ino=int(value.st_ino),
            size=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
        )


@dataclasses.dataclass(frozen=True)
class CatalogBinding:
    root: Path
    header_path: Path
    gpic_path: Path
    header_raw_sha256: str
    catalog_hash: str
    catalog_rows: int
    gpic_rows: int
    gpic_row_base: int
    gpic_bytes: int
    gpic_sha256: str
    gpic_schema: str
    header_identity: FileIdentity
    gpic_identity: FileIdentity


@dataclasses.dataclass(frozen=True)
class SelectedItem:
    catalog_local_index: int
    row_id: int
    image_path: str
    shard: str
    extension: str


@dataclasses.dataclass(frozen=True)
class SourceRead:
    item: SelectedItem
    payload: bytes
    sha256_raw: bytes
    duration_ns: int
    attempts: int


@dataclasses.dataclass(frozen=True)
class IndexEntry:
    row_id: int
    blob_id: int
    offset: int
    length: int
    sha256_raw: bytes


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strict_int(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PilotError(f"{label} must be an integer >= {minimum}: {value!r}")
    return value


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


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


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def manifest_self_hash(value: dict[str, Any]) -> str:
    core = dict(value)
    core.pop("manifest_sha256", None)
    return canonical_json_sha256(core)


def regular_file(path: Path) -> os.stat_result:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise PilotError(f"not a regular file: {path}")
    return value


def regular_dir(path: Path) -> os.stat_result:
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode):
        raise PilotError(f"not a real directory: {path}")
    return value


def sha256_file(path: Path, block_bytes: int = 8 * MIB) -> str:
    before = FileIdentity.from_stat(regular_file(path))
    digest = hashlib.sha256()
    count = 0
    with path.open("rb", buffering=0) as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            count += len(block)
            digest.update(block)
    after = FileIdentity.from_stat(regular_file(path))
    if before != after or count != before.size:
        raise PilotError(f"file changed or short-read while hashing: {path}")
    return digest.hexdigest()


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish without the overwrite semantics of ``os.rename``."""
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS), str(destination))
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    result = function(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _rename_noreplace_fallback_allowed(exc: OSError) -> bool:
    return exc.errno in {
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
        errno.ENOTSUP,
    }


def publish_file_noreplace(source: Path, destination: Path) -> str:
    """Publish one regular file atomically without ever replacing a name."""
    regular_file(source)
    if source.parent.resolve(strict=True) != destination.parent.resolve(strict=True):
        raise PilotError("file no-replace publication requires one parent directory")
    try:
        rename_noreplace(source, destination)
        return "renameat2_noreplace"
    except OSError as exc:
        if not _rename_noreplace_fallback_allowed(exc):
            raise
    # link(2) creates the destination name atomically and returns EEXIST if a
    # concurrent publisher won.  It therefore preserves no-clobber on Lustre
    # filesystems that reject renameat2(RENAME_NOREPLACE).
    os.link(source, destination, follow_symlinks=False)
    fsync_dir(destination.parent)
    source.unlink()
    fsync_dir(destination.parent)
    return "hardlink_noreplace"


def _directory_inode(path: Path) -> tuple[int, int]:
    metadata = regular_dir(path)
    return int(metadata.st_dev), int(metadata.st_ino)


def _mark_directory_claim_failed_before_commit(destination: Path) -> None:
    """Best-effort failure marker, callable only while _SUCCESS is absent."""
    try:
        marker = destination / "_PUBLISH_FAILED"
        if not marker.exists() and not marker.is_symlink():
            write_once_bytes(marker, b"fail-closed\n", fsync_file=True)
        fsync_dir(destination)
    except BaseException:
        pass


def _raise_if_marker_visible_or_unknown(
    destination: Path,
    *,
    message: str,
    cause: BaseException,
) -> None:
    """Return only when marker absence is positively established."""
    try:
        (destination / "_SUCCESS").lstat()
    except FileNotFoundError:
        return
    except BaseException as inspect_exc:
        raise CommittedSyncUnconfirmed(
            f"{message}; _SUCCESS absence could not be established"
        ) from inspect_exc
    raise CommittedSyncUnconfirmed(message) from cause


def publish_directory_noreplace(
    source: Path,
    destination: Path,
    *,
    success_last: bool,
) -> str:
    """Publish a flat directory without overwriting a concurrent destination.

    Fast path is renameat2(NOREPLACE).  Lustre may return EINVAL for that
    operation.  The fallback atomically claims the destination with mkdir,
    hard-links every regular file with EEXIST/no-overwrite semantics, and links
    ``_SUCCESS`` last.  Before that final link the visible directory is never a
    published artifact under this repository's reader contract.
    """
    regular_dir(source)
    source_parent = source.parent.resolve(strict=True)
    destination_parent = destination.parent.resolve(strict=True)
    if source_parent != destination_parent:
        raise PilotError("directory publication requires source/destination siblings")
    entries = sorted(source.iterdir(), key=lambda path: path.name)
    if not entries or any(path.is_symlink() for path in entries):
        raise PilotError("publication source is empty or contains a symlink")
    for path in entries:
        regular_file(path)
    success_source = source / "_SUCCESS"
    if success_last and success_source not in entries:
        raise PilotError("successful directory publication lacks _SUCCESS")
    # Failed-root publication is never allowed to use the directory-rename fast
    # path: source may still contain _SUCCESS from the abandoned active tree.
    if success_last:
        source_inode = _directory_inode(source)
        expected_source_names = {path.name for path in entries}
        fsync_dir(source)
        fsync_dir(source.parent)
        if _directory_inode(source) != source_inode:
            raise PilotError("source inode changed before rename publication")
        if {path.name for path in source.iterdir()} != expected_source_names:
            raise PilotError("source file set changed before rename publication")
        try:
            rename_noreplace(source, destination)
        except OSError as exc:
            _raise_if_marker_visible_or_unknown(
                destination,
                message="rename reported failure after publication marker became visible",
                cause=exc,
            )
            if not _rename_noreplace_fallback_allowed(exc):
                raise
        else:
            # rename is the linearization point.  Nothing after this branch may
            # retract _SUCCESS or create failure evidence.
            try:
                hook = _POST_COMMIT_DURABILITY_HOOK
                if hook is not None:
                    hook(destination, "renameat2_noreplace")
                fsync_dir(destination)
                fsync_dir(destination.parent)
                if _directory_inode(destination) != source_inode:
                    raise PilotError("committed rename destination inode changed")
                if {
                    path.name for path in destination.iterdir()
                } != expected_source_names:
                    raise PilotError("committed rename destination file set drift")
            except BaseException as exc:
                raise CommittedSyncUnconfirmed(
                    "renameat2 publication committed but post-commit sync/recheck failed"
                ) from exc
            return "renameat2_noreplace"

    hook = _FALLBACK_BEFORE_DIRECTORY_CLAIM_HOOK
    if hook is not None:
        hook(destination)
    # mkdir is the atomic, no-overwrite claim.  A concurrent destination causes
    # FileExistsError here and is never removed or renamed over.
    destination.mkdir(mode=0o700, exist_ok=False)
    claim_inode = _directory_inode(destination)
    publish_entries = (
        entries if success_last else [path for path in entries if path.name != "_SUCCESS"]
    )
    ordered = [path for path in publish_entries if path.name != "_SUCCESS"]
    expected_before_success = {path.name for path in ordered}
    try:
        for path in ordered:
            if _directory_inode(destination) != claim_inode:
                raise PilotError("destination claim inode changed during publication")
            os.link(path, destination / path.name, follow_symlinks=False)
        # Every validation and data/metadata durability operation precedes the
        # marker.  Before _SUCCESS, failures may still be marked fail-closed.
        if _directory_inode(destination) != claim_inode:
            raise PilotError("destination claim inode changed before precommit sync")
        if {path.name for path in destination.iterdir()} != expected_before_success:
            raise PilotError("destination changed before publication marker")
        fsync_dir(destination)
        fsync_dir(destination.parent)
        if _directory_inode(destination) != claim_inode:
            raise PilotError("destination claim inode changed after precommit sync")
        if {path.name for path in destination.iterdir()} != expected_before_success:
            raise PilotError("destination changed after precommit sync")
    except BaseException:
        _mark_directory_claim_failed_before_commit(destination)
        raise

    if success_last:
        try:
            os.link(
                success_source,
                destination / "_SUCCESS",
                follow_symlinks=False,
            )
        except BaseException as exc:
            # A signal or transport error may be reported after link(2)'s
            # namespace mutation.  Add failure evidence only after positively
            # proving that _SUCCESS is absent.
            _raise_if_marker_visible_or_unknown(
                destination,
                message="hardlink reported failure after publication marker became visible",
                cause=exc,
            )
            _mark_directory_claim_failed_before_commit(destination)
            raise
        # _SUCCESS link above is the sole linearization point.  From here on the
        # state is monotonic: never unlink it and never add _PUBLISH_FAILED.
        try:
            hook = _POST_COMMIT_DURABILITY_HOOK
            if hook is not None:
                hook(destination, "claim_directory_hardlink_files_success_last")
            fsync_dir(destination)
            fsync_dir(destination.parent)
            expected_final = {path.name for path in publish_entries}
            if _directory_inode(destination) != claim_inode:
                raise PilotError("committed destination inode changed")
            if {path.name for path in destination.iterdir()} != expected_final:
                raise PilotError("committed directory file set drift")
        except BaseException as exc:
            raise CommittedSyncUnconfirmed(
                "hardlink publication committed but post-commit sync/recheck failed"
            ) from exc
    else:
        # No success marker exists in a failed-root.  Close durability normally.
        fsync_dir(destination)
        fsync_dir(destination.parent)

    # The destination now owns hard links to the exact immutable files.  Source
    # cleanup is post-publication housekeeping and cannot affect destination.
    try:
        for path in entries:
            path.unlink()
        source.rmdir()
        fsync_dir(source.parent)
    except BaseException as exc:
        print(
            f"OFFLINE_GPIC_BLOB_PUBLISH_CLEANUP_WARNING class={type(exc).__name__}",
            file=sys.stderr,
            flush=True,
        )
    return (
        "claim_directory_hardlink_files_success_last"
        if success_last
        else "claim_directory_hardlink_files_failed_root_no_success"
    )


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    if not str(path.absolute()).startswith("/fsx/"):
        raise PilotError("atomic evidence publication is restricted to FSx")
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    try:
        with temporary.open("xb", buffering=0) as handle:
            write_all(handle, payload, label=str(temporary))
            handle.flush()
            os.fsync(handle.fileno())
        publish_file_noreplace(temporary, path)
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_once_bytes(path: Path, payload: bytes, *, fsync_file: bool) -> None:
    with path.open("xb", buffering=0) as handle:
        write_all(handle, payload, label=str(path))
        handle.flush()
        # Mountpoint-S3 commits a sequentially-created object through the file
        # handle.  We therefore sync every final object before close as well;
        # what we deliberately never do for that backend is directory fsync or
        # rename-based publication.
        if fsync_file:
            os.fsync(handle.fileno())


def write_all(handle: Any, payload: bytes | memoryview, *, label: str) -> None:
    view = memoryview(payload)
    while view:
        written = handle.write(view)
        if written is None or written <= 0:
            raise PilotError(f"short write: {label}")
        view = view[written:]


def write_json_once(path: Path, value: Any, *, fsync_file: bool) -> None:
    write_once_bytes(path, canonical_json_bytes(value), fsync_file=fsync_file)


def percentile(values: Sequence[int | float], quantile: float) -> float:
    if not values:
        raise PilotError("cannot calculate percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_summary_ns(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        raise PilotError("empty timing population")
    return {
        "count": len(values),
        "min_ms": min(values) / 1e6,
        "p50_ms": percentile(values, 0.50) / 1e6,
        "p95_ms": percentile(values, 0.95) / 1e6,
        "p99_ms": percentile(values, 0.99) / 1e6,
        "max_ms": max(values) / 1e6,
    }


def snapshot_code(job_spec: Path) -> dict[str, Any]:
    runner = Path(__file__).resolve(strict=True)
    job_spec = job_spec.expanduser().resolve(strict=True)
    return {
        "runner_path": str(runner),
        "runner_sha256": sha256_file(runner),
        "job_spec_path": str(job_spec),
        "job_spec_sha256": sha256_file(job_spec),
        "python": platform.python_version(),
        "pyarrow": pyarrow.__version__,
        "index_struct": INDEX_STRUCT.format,
        "index_record_bytes": INDEX_RECORD_BYTES,
    }


def runtime_identity() -> dict[str, Any]:
    node_name = None
    node_source = None
    for key in ("JCC_NODE_NAME", "K8S_NODE_NAME", "NODE_NAME"):
        value = os.environ.get(key)
        if value:
            node_name = value
            node_source = key
            break
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    try:
        boot_id_first = boot_path.read_text(encoding="ascii").strip().lower()
        boot_id_second = boot_path.read_text(encoding="ascii").strip().lower()
    except OSError as exc:
        raise PilotError("kernel boot-id is unavailable for cache-host binding") from exc
    if boot_id_first != boot_id_second or BOOT_ID_RE.fullmatch(boot_id_first) is None:
        raise PilotError("kernel boot-id is unstable or malformed")
    # Never publish the raw host identifier.  The hash is sufficient to prove
    # builder/replay physical-cache separation without leaking the UUID.
    boot_id_sha256 = hashlib.sha256(boot_id_first.encode("ascii")).hexdigest()
    return {
        "hostname": platform.node(),
        "jcc_task_name": os.environ.get("JCC_TASK_NAME"),
        "jcc_task_instance": os.environ.get("JCC_TASK_INSTANCE"),
        "physical_node_name_sha256": (
            hashlib.sha256(node_name.encode("utf-8")).hexdigest()
            if node_name is not None
            else None
        ),
        "physical_node_name_source": node_source,
        "boot_id_sha256": boot_id_sha256,
        "boot_id_raw_published": False,
    }


def validate_code_snapshot(snapshot: dict[str, Any]) -> None:
    for prefix in ("runner", "job_spec"):
        path = Path(snapshot[f"{prefix}_path"])
        if sha256_file(path) != snapshot[f"{prefix}_sha256"]:
            raise PilotError(f"{prefix} changed during pilot")


def load_catalog(root: Path, *, verify_gpic_sha256: bool) -> CatalogBinding:
    root = root.expanduser().resolve(strict=True)
    regular_dir(root)
    header_path = root / "header.json"
    header_before = FileIdentity.from_stat(regular_file(header_path))
    header_raw = header_path.read_bytes()
    header_after = FileIdentity.from_stat(regular_file(header_path))
    if header_before != header_after or len(header_raw) != header_before.size:
        raise PilotError("catalog header changed while reading")
    try:
        header = json.loads(header_raw)
    except Exception as exc:
        raise PilotError("catalog header is invalid JSON") from exc
    if not isinstance(header, dict) or header.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise PilotError("catalog schema_version drift")
    order = header.get("domain_order")
    files = header.get("files")
    if not isinstance(order, list) or not isinstance(files, dict) or "gpic" not in order:
        raise PilotError("catalog domain/file mapping is malformed")
    cursor = 0
    gpic_entry: dict[str, Any] | None = None
    gpic_row_base = -1
    for name in order:
        entry = files.get(name)
        if not isinstance(entry, dict):
            raise PilotError(f"catalog file entry missing: {name}")
        rows = strict_int(entry.get("rows"), f"catalog files.{name}.rows", 1)
        if name == "gpic":
            gpic_entry = entry
            gpic_row_base = cursor
        cursor += rows
    if header.get("row_count") != cursor or gpic_entry is None:
        raise PilotError("catalog row-count closure drift")
    catalog_hash = header.get("catalog_hash")
    if not is_sha256(catalog_hash):
        raise PilotError("catalog_hash is malformed")
    name = gpic_entry.get("name")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".parquet"):
        raise PilotError("gpic catalog filename is malformed")
    gpic_path = root / name
    gpic_identity = FileIdentity.from_stat(regular_file(gpic_path))
    gpic_bytes = strict_int(gpic_entry.get("bytes"), "gpic catalog bytes", 1)
    gpic_rows = strict_int(gpic_entry.get("rows"), "gpic catalog rows", 1)
    gpic_sha = gpic_entry.get("sha256")
    gpic_schema = gpic_entry.get("schema")
    if (
        gpic_identity.size != gpic_bytes
        or not is_sha256(gpic_sha)
        or not isinstance(gpic_schema, str)
        or not gpic_schema
    ):
        raise PilotError("gpic catalog file binding is malformed")
    if verify_gpic_sha256 and sha256_file(gpic_path) != gpic_sha:
        raise PilotError("gpic catalog SHA-256 mismatch")
    parquet = pq.ParquetFile(gpic_path)
    required = {"row_id", "image_path"}
    if parquet.metadata.num_rows != gpic_rows or not required.issubset(
        parquet.schema_arrow.names
    ):
        raise PilotError("gpic parquet footer/schema drift")
    if str(parquet.schema_arrow) != gpic_schema:
        raise PilotError("gpic parquet schema text drift")
    return CatalogBinding(
        root=root,
        header_path=header_path,
        gpic_path=gpic_path,
        header_raw_sha256=hashlib.sha256(header_raw).hexdigest(),
        catalog_hash=catalog_hash,
        catalog_rows=cursor,
        gpic_rows=gpic_rows,
        gpic_row_base=gpic_row_base,
        gpic_bytes=gpic_bytes,
        gpic_sha256=gpic_sha,
        gpic_schema=gpic_schema,
        header_identity=header_after,
        gpic_identity=gpic_identity,
    )


def validate_catalog_closure(binding: CatalogBinding) -> None:
    if FileIdentity.from_stat(regular_file(binding.header_path)) != binding.header_identity:
        raise PilotError("catalog header identity changed during pilot")
    if FileIdentity.from_stat(regular_file(binding.gpic_path)) != binding.gpic_identity:
        raise PilotError("gpic catalog identity changed during pilot")
    if hashlib.sha256(binding.header_path.read_bytes()).hexdigest() != binding.header_raw_sha256:
        raise PilotError("catalog header content changed during pilot")


def catalog_manifest(binding: CatalogBinding) -> dict[str, Any]:
    return {
        "root": str(binding.root),
        "header": {
            "name": binding.header_path.name,
            "bytes": binding.header_identity.size,
            "raw_sha256": binding.header_raw_sha256,
            "identity": dataclasses.asdict(binding.header_identity),
        },
        "catalog_hash": binding.catalog_hash,
        "catalog_rows": binding.catalog_rows,
        "gpic": {
            "name": binding.gpic_path.name,
            "bytes": binding.gpic_bytes,
            "sha256": binding.gpic_sha256,
            "rows": binding.gpic_rows,
            "row_base": binding.gpic_row_base,
            "schema": binding.gpic_schema,
            "identity": dataclasses.asdict(binding.gpic_identity),
        },
    }


def stratified_positions(total_rows: int, sample_rows: int) -> list[int]:
    strict_int(total_rows, "total_rows", 1)
    strict_int(sample_rows, "sample_rows", 1)
    if sample_rows > total_rows:
        raise PilotError("sample_rows exceeds catalog rows")
    positions = [((2 * index + 1) * total_rows) // (2 * sample_rows) for index in range(sample_rows)]
    if len(set(positions)) != sample_rows or positions != sorted(positions):
        raise PilotError("stratified position construction is not unique/ordered")
    return positions


def select_catalog_items(
    binding: CatalogBinding,
    *,
    sample_rows: int,
    coverage_intervals: int,
    min_unique_shards: int,
) -> tuple[list[SelectedItem], dict[str, Any]]:
    positions = stratified_positions(binding.gpic_rows, sample_rows)
    return select_catalog_positions(
        binding,
        positions=positions,
        coverage_intervals=coverage_intervals,
        min_unique_shards=min_unique_shards,
        method=SELECTION_METHOD,
    )


def summarize_selection(
    binding: CatalogBinding,
    items: Sequence[SelectedItem],
    *,
    coverage_intervals: int,
    min_unique_shards: int,
    method: str,
) -> dict[str, Any]:
    if not items:
        raise PilotError("cannot summarize an empty catalog selection")
    positions = [item.catalog_local_index for item in items]
    if positions != sorted(set(positions)):
        raise PilotError("selection positions are not unique and ordered")
    shard_counts = Counter(item.shard for item in items)
    extension_counts = Counter(item.extension for item in items)
    interval_counts = Counter(
        min(
            coverage_intervals - 1,
            (item.catalog_local_index * coverage_intervals) // binding.gpic_rows,
        )
        for item in items
    )
    if len(shard_counts) < min_unique_shards:
        raise PilotError(
            f"selection shard coverage too small: {len(shard_counts)} < {min_unique_shards}"
        )
    if set(extension_counts) != {".jpg", ".png"}:
        raise PilotError("selection does not cover both JPG and PNG")
    if len(interval_counts) != coverage_intervals:
        raise PilotError("selection does not cover every requested catalog interval")

    digest = hashlib.sha256()
    for item in items:
        raw_path = item.image_path.encode("utf-8")
        digest.update(struct.pack("<qqI", item.catalog_local_index, item.row_id, len(raw_path)))
        digest.update(raw_path)
    return {
        "method": method,
        "sample_rows": len(items),
        "catalog_local_first": items[0].catalog_local_index,
        "catalog_local_last": items[-1].catalog_local_index,
        "row_id_first": items[0].row_id,
        "row_id_last": items[-1].row_id,
        "selection_sha256": digest.hexdigest(),
        "coverage_intervals": coverage_intervals,
        "covered_intervals": len(interval_counts),
        "interval_count_min": min(interval_counts.values()),
        "interval_count_max": max(interval_counts.values()),
        "unique_shards": len(shard_counts),
        "min_unique_shards_required": min_unique_shards,
        "extension_counts": dict(sorted(extension_counts.items())),
    }


def select_catalog_positions(
    binding: CatalogBinding,
    *,
    positions: Sequence[int],
    coverage_intervals: int,
    min_unique_shards: int,
    method: str,
) -> tuple[list[SelectedItem], dict[str, Any]]:
    positions = list(positions)
    if (
        not positions
        or positions != sorted(set(positions))
        or positions[0] < 0
        or positions[-1] >= binding.gpic_rows
    ):
        raise PilotError("explicit catalog positions are malformed")
    parquet = pq.ParquetFile(binding.gpic_path)
    by_group: dict[int, list[tuple[int, int]]] = {}
    position_cursor = 0
    group_start = 0
    for group_index in range(parquet.num_row_groups):
        group_rows = parquet.metadata.row_group(group_index).num_rows
        group_stop = group_start + group_rows
        while position_cursor < len(positions) and positions[position_cursor] < group_stop:
            position = positions[position_cursor]
            if position < group_start:
                raise PilotError("catalog position/group mapping regressed")
            by_group.setdefault(group_index, []).append((position_cursor, position - group_start))
            position_cursor += 1
        group_start = group_stop
    if group_start != binding.gpic_rows or position_cursor != len(positions):
        raise PilotError("catalog row-group coverage does not close")

    selected: list[SelectedItem | None] = [None] * len(positions)
    for group_index, targets in by_group.items():
        table = parquet.read_row_group(
            group_index, columns=["row_id", "image_path"], use_threads=False
        ).combine_chunks()
        row_ids = table.column("row_id")
        paths = table.column("image_path")
        for selection_index, row_offset in targets:
            row_id = row_ids[row_offset].as_py()
            image_path = paths[row_offset].as_py()
            position = positions[selection_index]
            expected_row_id = binding.gpic_row_base + position
            if type(row_id) is not int or row_id != expected_row_id:
                raise PilotError(f"catalog row_id drift at local position {position}")
            if not isinstance(image_path, str) or not image_path.startswith(SOURCE_PREFIX):
                raise PilotError(f"catalog GPIC path prefix drift at row_id={row_id}")
            path = Path(image_path)
            shard = path.parent.name
            extension = path.suffix.lower()
            if SHARD_RE.fullmatch(shard) is None or extension not in {".jpg", ".png"}:
                raise PilotError(f"catalog GPIC shard/extension drift at row_id={row_id}")
            selected[selection_index] = SelectedItem(
                catalog_local_index=position,
                row_id=row_id,
                image_path=image_path,
                shard=shard,
                extension=extension,
            )
    if any(item is None for item in selected):
        raise PilotError("catalog selection contains holes")
    items = [item for item in selected if item is not None]
    summary = summarize_selection(
        binding,
        items,
        coverage_intervals=coverage_intervals,
        min_unique_shards=min_unique_shards,
        method=method,
    )
    return items, summary


def selection_jsonl(items: Sequence[SelectedItem]) -> bytes:
    output = bytearray()
    for item in items:
        output.extend(
            canonical_json_bytes(
                {
                    "catalog_local_index": item.catalog_local_index,
                    "image_path": item.image_path,
                    "row_id": item.row_id,
                }
            )
        )
    return bytes(output)


def source_ordered_items(
    items: Sequence[SelectedItem], mode: str
) -> list[SelectedItem]:
    if mode == "catalog":
        return list(items)
    if mode == "path":
        return sorted(items, key=lambda item: (item.image_path, item.row_id))
    raise PilotError(f"unsupported source order: {mode!r}")


def _read_source_once(item: SelectedItem, block_bytes: int) -> SourceRead:
    started = time.perf_counter_ns()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(item.image_path, flags)
    try:
        before = FileIdentity.from_stat(os.fstat(descriptor))
        if before.size <= 0 or before.size > 0xFFFFFFFF:
            raise PilotError(f"invalid GPIC source size row_id={item.row_id}")
        blocks: list[bytes] = []
        count = 0
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, block_bytes)
            if not block:
                break
            count += len(block)
            digest.update(block)
            blocks.append(block)
        after = FileIdentity.from_stat(os.fstat(descriptor))
        if before != after:
            raise PilotError(f"GPIC fd identity changed row_id={item.row_id}")
        if count != before.size:
            raise PilotError(
                f"GPIC full-read length mismatch row_id={item.row_id}: {count} != {before.size}"
            )
        payload = b"".join(blocks)
        return SourceRead(
            item=item,
            payload=payload,
            sha256_raw=digest.digest(),
            duration_ns=time.perf_counter_ns() - started,
            attempts=1,
        )
    finally:
        os.close(descriptor)


def read_source_stable(
    item: SelectedItem,
    *,
    attempts: int,
    block_bytes: int,
    retry_base_seconds: float,
) -> SourceRead:
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            value = _read_source_once(item, block_bytes)
            return dataclasses.replace(value, attempts=attempt)
        except PilotError:
            # Identity, empty/short-read, and size contract failures are data
            # integrity failures, not transient transport failures.
            raise
        except OSError as exc:
            last = exc
            if attempt < attempts:
                time.sleep(retry_base_seconds * (2 ** (attempt - 1)))
    raise PilotError(
        f"GPIC source read exhausted retries row_id={item.row_id}, class={type(last).__name__}"
    ) from last


def ordered_parallel_reads(
    items: Sequence[SelectedItem],
    *,
    workers: int,
    attempts: int,
    block_bytes: int,
    retry_base_seconds: float,
) -> Iterator[SourceRead]:
    if workers < 1:
        raise PilotError("workers must be positive")
    capacity = max(workers, workers * 2)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="gpic-blob-source"
    ) as executor:
        futures: dict[int, concurrent.futures.Future[SourceRead]] = {}
        submit_cursor = 0

        def submit_one(index: int) -> None:
            futures[index] = executor.submit(
                read_source_stable,
                items[index],
                attempts=attempts,
                block_bytes=block_bytes,
                retry_base_seconds=retry_base_seconds,
            )

        while submit_cursor < min(len(items), capacity):
            submit_one(submit_cursor)
            submit_cursor += 1
        try:
            for index in range(len(items)):
                future = futures.pop(index)
                yield future.result()
                if submit_cursor < len(items):
                    submit_one(submit_cursor)
                    submit_cursor += 1
        except BaseException:
            for future in futures.values():
                future.cancel()
            raise


def validate_output_authority(
    output_root: Path,
    allowed_output_prefix: Path,
    backend: str,
    write_authority: str,
) -> tuple[Path, Path]:
    if write_authority != WRITE_AUTHORITY:
        raise PilotError("explicit pilot write authority token is absent")
    output_root = output_root.expanduser().absolute()
    prefix = allowed_output_prefix.expanduser().resolve(strict=True)
    regular_dir(prefix)
    try:
        output_root.relative_to(prefix)
    except ValueError as exc:
        raise PilotError("output root is outside the caller-authorized prefix") from exc
    if output_root == prefix or len(output_root.relative_to(prefix).parts) < 2:
        raise PilotError("output root must be at least two components below its allowed prefix")
    if backend == "fsx_posix" and not str(prefix).startswith("/fsx/"):
        raise PilotError("fsx_posix prefix must be under /fsx")
    if backend == "mountpoint_s3" and not (
        str(prefix).startswith("/s3/evan/")
        or str(prefix).startswith("/s3-pixai-test-uw2/evan/")
    ):
        raise PilotError("mountpoint_s3 prefix is not an approved writable Evan mount")
    if output_root.exists() or output_root.is_symlink():
        raise PilotError(f"output namespace already exists: {output_root}")
    cursor = prefix
    for component in output_root.relative_to(prefix).parts[:-1]:
        cursor = cursor / component
        if cursor.exists() or cursor.is_symlink():
            regular_dir(cursor)
    return output_root, prefix


class BlobWriter:
    def __init__(self, root: Path, target_bytes: int, fsync_files: bool):
        self.root = root
        self.target_bytes = target_bytes
        self.fsync_files = fsync_files
        self.blob_id = -1
        self.handle: Any = None
        self.offset = 0
        self.digest: hashlib._Hash | None = None
        self.files: list[dict[str, Any]] = []

    def _open_next(self) -> None:
        self.blob_id += 1
        if self.blob_id > 0xFFFFFFFF:
            raise PilotError("blob_id exceeds uint32")
        path = self.root / f"blob-{self.blob_id:06d}.bin"
        self.handle = path.open("xb", buffering=0)
        self.offset = 0
        self.digest = hashlib.sha256()

    def _close_current(self) -> None:
        if self.handle is None or self.digest is None:
            return
        path = Path(self.handle.name)
        self.handle.flush()
        if self.fsync_files:
            os.fsync(self.handle.fileno())
        self.handle.close()
        metadata = regular_file(path)
        if metadata.st_size != self.offset:
            raise PilotError(f"blob size drift after close: {path}")
        self.files.append(
            {
                "name": path.name,
                "blob_id": self.blob_id,
                "bytes": self.offset,
                "sha256": self.digest.hexdigest(),
            }
        )
        self.handle = None
        self.digest = None

    def append(self, payload: bytes) -> tuple[int, int]:
        if not payload:
            raise PilotError("refusing to append an empty payload")
        if self.handle is None:
            self._open_next()
        elif self.offset > 0 and self.offset + len(payload) > self.target_bytes:
            self._close_current()
            self._open_next()
        assert self.handle is not None and self.digest is not None
        start = self.offset
        view = memoryview(payload)
        while view:
            written = self.handle.write(view)
            if written is None or written <= 0:
                raise PilotError("short blob write")
            self.digest.update(view[:written])
            self.offset += written
            view = view[written:]
        return self.blob_id, start

    def close(self) -> list[dict[str, Any]]:
        self._close_current()
        if not self.files:
            raise PilotError("blob writer produced no files")
        return self.files

    def abort(self) -> None:
        if self.handle is not None:
            try:
                self.handle.close()
            finally:
                self.handle = None


def set_lustre_stripe(root: Path, count: int, size_mib: int) -> dict[str, Any]:
    if count == 0:
        return {"count": 0, "size_mib": 0, "configured": False}
    if count not in {1, 4, 8} or size_mib < 1:
        raise PilotError("unsupported pilot Lustre stripe configuration")
    command = ["lfs", "setstripe", "-c", str(count), "-S", f"{size_mib}m", str(root)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise PilotError(f"lfs setstripe failed rc={completed.returncode}")
    return {"count": count, "size_mib": size_mib, "configured": True}


def file_binding(path: Path, sha256: str | None = None) -> dict[str, Any]:
    metadata = regular_file(path)
    return {
        "name": path.name,
        "bytes": int(metadata.st_size),
        "sha256": sha256 if sha256 is not None else sha256_file(path),
    }


def build_pilot(args: argparse.Namespace) -> dict[str, Any]:
    output_root, _ = validate_output_authority(
        args.output_root,
        args.allowed_output_prefix,
        args.backend,
        args.write_authority,
    )
    code = snapshot_code(args.job_spec)
    catalog = load_catalog(args.catalog_root, verify_gpic_sha256=True)
    items, selection = select_catalog_items(
        catalog,
        sample_rows=args.sample_rows,
        coverage_intervals=args.coverage_intervals,
        min_unique_shards=args.min_unique_shards,
    )
    nonce = uuid.uuid4().hex
    is_fsx = args.backend == "fsx_posix"
    if not is_fsx and args.sample_rows > 20_000:
        raise PilotError("Mountpoint-S3 pilot is hard-limited to 20,000 catalog rows")
    if not is_fsx and args.blob_target_bytes > 8 * (1 << 30):
        raise PilotError("Mountpoint-S3 pilot blob target exceeds the 8-GiB safety cap")
    if not is_fsx and not (1 <= args.max_total_source_bytes <= 8 * (1 << 30)):
        raise PilotError("Mountpoint-S3 pilot requires a total-source-byte cap <= 8 GiB")
    active_root = (
        output_root.parent / f".{output_root.name}.work.{nonce}" if is_fsx else output_root
    )
    if active_root.exists() or active_root.is_symlink():
        raise PilotError("active pilot namespace is not fresh")
    active_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    stripe: dict[str, Any] = {"count": 0, "size_mib": 0, "configured": False}
    writer: BlobWriter | None = None
    publication_marker_committed = False
    try:
        if is_fsx:
            stripe = set_lustre_stripe(active_root, args.stripe_count, args.stripe_size_mib)
        elif args.stripe_count != 0:
            raise PilotError("Mountpoint-S3 build must use stripe_count=0")

        selection_raw = selection_jsonl(items)
        write_once_bytes(
            active_root / "selection.jsonl", selection_raw, fsync_file=True
        )
        selection_file = file_binding(
            active_root / "selection.jsonl", hashlib.sha256(selection_raw).hexdigest()
        )

        index_path = active_root / "index.bin"
        writer = BlobWriter(active_root, args.blob_target_bytes, fsync_files=True)
        index_entries: list[IndexEntry] = []
        source_latencies: list[int] = []
        source_attempts = Counter()
        total_source_bytes = 0
        aggregate_content = hashlib.sha256()
        started_ns = time.perf_counter_ns()
        try:
            for result in ordered_parallel_reads(
                source_ordered_items(items, args.source_order),
                workers=args.threads,
                attempts=args.source_attempts,
                block_bytes=args.source_block_bytes,
                retry_base_seconds=args.retry_base_seconds,
            ):
                if (
                    args.max_total_source_bytes > 0
                    and total_source_bytes + len(result.payload)
                    > args.max_total_source_bytes
                ):
                    raise PilotError("pilot source bytes exceed the explicit output cap")
                blob_id, offset = writer.append(result.payload)
                index_entries.append(
                    IndexEntry(
                        row_id=result.item.row_id,
                        blob_id=blob_id,
                        offset=offset,
                        length=len(result.payload),
                        sha256_raw=result.sha256_raw,
                    )
                )
                aggregate_content.update(struct.pack("<qI", result.item.row_id, len(result.payload)))
                aggregate_content.update(result.sha256_raw)
                source_latencies.append(result.duration_ns)
                source_attempts[result.attempts] += 1
                total_source_bytes += len(result.payload)
            blob_files = writer.close()
            writer = None
            index_entries.sort(key=lambda entry: entry.row_id)
            if [entry.row_id for entry in index_entries] != [item.row_id for item in items]:
                raise PilotError("source-order build did not close over the selected row_ids")
            index_digest = hashlib.sha256()
            with index_path.open("xb", buffering=0) as index_handle:
                for entry in index_entries:
                    record = INDEX_STRUCT.pack(
                        entry.row_id,
                        entry.blob_id,
                        entry.offset,
                        entry.length,
                        entry.sha256_raw,
                    )
                    write_all(index_handle, record, label="index.bin")
                    index_digest.update(record)
                index_handle.flush()
                os.fsync(index_handle.fileno())
        except BaseException:
            if writer is not None:
                writer.abort()
            raise
        finished_ns = time.perf_counter_ns()

        index_file = file_binding(index_path, index_digest.hexdigest())
        if index_file["bytes"] != len(items) * INDEX_RECORD_BYTES:
            raise PilotError("fixed-width index byte count does not close")
        if sum(entry["bytes"] for entry in blob_files) != total_source_bytes:
            raise PilotError("blob bytes do not equal source bytes")
        validate_catalog_closure(catalog)
        validate_code_snapshot(code)
        elapsed_s = (finished_ns - started_ns) / 1e9
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "layout_version": LAYOUT_VERSION,
            "pilot_only": True,
            "production_selection_qualified": False,
            "generated_at": utc_now(),
            "run_nonce": nonce,
            "backend": args.backend,
            "publication_semantics": (
                "posix_renameat2_noreplace_or_mkdir_claim_hardlink_files_success_last"
                if is_fsx
                else "mountpoint_s3_write_once_final_keys_file_fsync_success_last_no_rename_no_dir_fsync"
            ),
            "catalog": catalog_manifest(catalog),
            "selection": selection,
            "layout": {
                "endianness": "little",
                "record_struct": INDEX_STRUCT.format,
                "record_bytes": INDEX_RECORD_BYTES,
                "fields": ["row_id:i64", "blob_id:u32", "offset:u64", "length:u32", "sha256:32B"],
                "blob_target_bytes": args.blob_target_bytes,
                "segments_contiguous_without_padding": True,
                "blob_segment_order": (
                    "source_path_lexicographic" if args.source_order == "path" else "catalog_row_id"
                ),
                "index_order": "row_id_ascending",
                "compression": "none",
                "stripe": stripe,
            },
            "files": {
                "selection": selection_file,
                "index": index_file,
                "blobs": blob_files,
            },
            "source_read": {
                "workers": args.threads,
                "source_order": args.source_order,
                "attempt_limit": args.source_attempts,
                "attempt_histogram": {str(key): value for key, value in sorted(source_attempts.items())},
                "bytes": total_source_bytes,
                "max_total_source_bytes": args.max_total_source_bytes,
                "objects": len(items),
                "elapsed_seconds": elapsed_s,
                "objects_per_second": len(items) / elapsed_s,
                "mib_per_second": total_source_bytes / MIB / elapsed_s,
                "per_object": timing_summary_ns(source_latencies),
                "aggregate_row_length_sha256_sha256": aggregate_content.hexdigest(),
            },
            "code": code,
            "builder_runtime": runtime_identity(),
        }
        manifest["manifest_sha256"] = manifest_self_hash(manifest)
        write_json_once(active_root / "manifest.json", manifest, fsync_file=True)
        write_once_bytes(
            active_root / "_SUCCESS",
            (manifest["manifest_sha256"] + "\n").encode("ascii"),
            fsync_file=True,
        )
        if is_fsx:
            fsync_dir(active_root)
            if output_root.exists() or output_root.is_symlink():
                raise PilotError("output appeared before atomic publication")
            publish_method = publish_directory_noreplace(
                active_root,
                output_root,
                success_last=True,
            )
            publication_marker_committed = True
            print(
                f"OFFLINE_GPIC_BLOB_FSX_PUBLISH method={publish_method}",
                flush=True,
            )
        else:
            # For Mountpoint-S3 the write-once final _SUCCESS key is the
            # publication linearization point.
            publication_marker_committed = True
        print(
            f"OFFLINE_GPIC_BLOB_PILOT_BUILD_PASS backend={args.backend} "
            f"root={output_root} rows={len(items)} img_s={len(items)/elapsed_s:.3f}",
            flush=True,
        )
        return manifest
    except CommittedSyncUnconfirmed:
        # The destination publication marker is already visible.  This is not
        # an ordinary build failure: creating _FAILED.json, _PUBLISH_FAILED, or
        # a *.FAILED.* root would contradict the monotonic publication state.
        print(
            "OFFLINE_GPIC_BLOB_PILOT_COMMITTED_SYNC_UNCONFIRMED "
            f"backend={args.backend} root={output_root}",
            file=sys.stderr,
            flush=True,
        )
        raise
    except BaseException as exc:
        marker_visible = False
        try:
            (output_root / "_SUCCESS").lstat()
            marker_visible = True
        except FileNotFoundError:
            pass
        except BaseException as inspect_exc:
            raise CommittedSyncUnconfirmed(
                "exception observed while publication marker state was unknowable"
            ) from inspect_exc
        if publication_marker_committed or marker_visible:
            # This also covers a marker write that became visible but reported
            # an error, plus any later logging/housekeeping exception.  Never
            # turn a visible success marker into contradictory failure evidence.
            raise CommittedSyncUnconfirmed(
                "exception observed after publication marker became visible"
            ) from exc
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "FAILED",
            "generated_at": utc_now(),
            "run_nonce": nonce,
            "backend": args.backend,
            "exception_class": type(exc).__name__,
        }
        try:
            write_json_once(active_root / "_FAILED.json", failure, fsync_file=True)
            if is_fsx:
                fsync_dir(active_root)
                failed_root = output_root.parent / f"{output_root.name}.FAILED.{nonce}"
                if not failed_root.exists() and not failed_root.is_symlink():
                    publish_directory_noreplace(
                        active_root,
                        failed_root,
                        success_last=False,
                    )
        except BaseException:
            pass
        raise


def source_scan(args: argparse.Namespace) -> dict[str, Any]:
    code = snapshot_code(args.job_spec)
    catalog = load_catalog(args.catalog_root, verify_gpic_sha256=True)
    threads = [int(token) for token in args.threads_matrix.split(",")]
    if not threads or len(set(threads)) != len(threads) or any(value < 1 or value > 128 for value in threads):
        raise PilotError("threads_matrix must contain unique integers in [1,128]")
    arm_threads = threads + list(reversed(threads)) if args.abba else threads
    combined_positions = stratified_positions(
        catalog.gpic_rows, args.sample_rows * len(arm_threads)
    )
    excluded_positions = (
        set(stratified_positions(catalog.gpic_rows, args.exclude_sample_rows))
        if args.exclude_sample_rows
        else set()
    )
    overlap = excluded_positions.intersection(combined_positions)
    if overlap:
        raise PilotError(
            "source-scan sample overlaps the explicitly excluded build selection: "
            f"count={len(overlap)}"
        )
    combined_items, combined_selection = select_catalog_positions(
        catalog,
        positions=combined_positions,
        coverage_intervals=args.coverage_intervals,
        min_unique_shards=args.min_unique_shards,
        method="catalog_stratified_midpoint_disjoint_arm_union_v1",
    )
    arm_items = [combined_items[index::len(arm_threads)] for index in range(len(arm_threads))]
    arm_selections = [
        summarize_selection(
            catalog,
            values,
            coverage_intervals=args.coverage_intervals,
            min_unique_shards=args.min_unique_shards,
            method="catalog_stratified_midpoint_disjoint_arm_v1",
        )
        for values in arm_items
    ]
    all_scan_rows = [item.row_id for values in arm_items for item in values]
    if len(set(all_scan_rows)) != len(all_scan_rows):
        raise PilotError("source-scan arm samples are not mutually exclusive")
    arms: list[dict[str, Any]] = []
    for arm_index, (workers, items, selection) in enumerate(
        zip(arm_threads, arm_items, arm_selections)
    ):
        latencies: list[int] = []
        attempts = Counter()
        byte_count = 0
        content_digest = hashlib.sha256()
        started = time.perf_counter_ns()
        for result in ordered_parallel_reads(
            source_ordered_items(items, args.source_order),
            workers=workers,
            attempts=args.source_attempts,
            block_bytes=args.source_block_bytes,
            retry_base_seconds=args.retry_base_seconds,
        ):
            byte_count += len(result.payload)
            latencies.append(result.duration_ns)
            attempts[result.attempts] += 1
            content_digest.update(struct.pack("<qI", result.item.row_id, len(result.payload)))
            content_digest.update(result.sha256_raw)
        elapsed_s = (time.perf_counter_ns() - started) / 1e9
        arm = {
            "arm_index": arm_index,
            "workers": workers,
            "objects": len(items),
            "bytes": byte_count,
            "elapsed_seconds": elapsed_s,
            "objects_per_second": len(items) / elapsed_s,
            "mib_per_second": byte_count / MIB / elapsed_s,
            "per_object": timing_summary_ns(latencies),
            "attempt_histogram": {str(key): value for key, value in sorted(attempts.items())},
            "aggregate_row_length_sha256_sha256": content_digest.hexdigest(),
            "selection": selection,
        }
        arms.append(arm)
        print(
            f"OFFLINE_GPIC_SOURCE_SCAN_ARM index={arm_index} threads={workers} "
            f"img_s={arm['objects_per_second']:.3f}",
            flush=True,
        )
    content_digests = {arm["aggregate_row_length_sha256_sha256"] for arm in arms}
    if len(content_digests) != len(arms):
        raise PilotError("source-scan disjoint arms unexpectedly share a content digest")
    byte_counts = [arm["bytes"] for arm in arms]
    byte_imbalance = max(byte_counts) / min(byte_counts)
    byte_balanced = byte_imbalance <= args.max_byte_imbalance
    by_thread: dict[int, list[float]] = {}
    for arm in arms:
        by_thread.setdefault(arm["workers"], []).append(arm["objects_per_second"])
    medians = {key: percentile(value, 0.50) for key, value in by_thread.items()}
    recommended = (
        min(
            (key for key, value in medians.items() if value == max(medians.values())),
            default=threads[0],
        )
        if byte_balanced
        else None
    )
    validate_catalog_closure(catalog)
    validate_code_snapshot(code)
    report = {
        "schema_version": "offline_gpic_blob_source_scan_v1",
        "status": "PASS" if byte_balanced else "INCONCLUSIVE",
        "status_reason": (
            "disjoint_arm_bytes_balanced"
            if byte_balanced
            else "disjoint_arm_bytes_exceed_max_imbalance"
        ),
        "pilot_only": True,
        "production_selection_qualified": False,
        "generated_at": utc_now(),
        "catalog": catalog_manifest(catalog),
        "selection_union": combined_selection,
        "samples_mutually_exclusive": True,
        "excluded_build_selection_sample_rows": args.exclude_sample_rows,
        "excluded_build_selection_overlap_rows": 0,
        "bidirectional_thread_order": args.abba,
        "arm_bytes_max_over_min": byte_imbalance,
        "max_arm_byte_imbalance_allowed": args.max_byte_imbalance,
        "source_order": args.source_order,
        "arms": arms,
        "median_objects_per_second_by_threads": {str(key): medians[key] for key in sorted(medians)},
        "recommended_threads_by_median": recommended,
        "full_catalog_hours_one_worker_idealized": (
            catalog.gpic_rows / max(medians.values()) / 3600 if byte_balanced else None
        ),
        "full_catalog_hours_eight_workers_idealized": (
            catalog.gpic_rows / max(medians.values()) / 3600 / 8 if byte_balanced else None
        ),
        "code": code,
        "runtime": runtime_identity(),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise PilotError("source-scan output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(output, canonical_json_bytes(report))
    print(
        f"OFFLINE_GPIC_SOURCE_SCAN_{report['status']} output={output} "
        f"recommended_threads={recommended}",
        flush=True,
    )
    return report


def load_published_manifest(root: Path) -> tuple[dict[str, Any], bytes]:
    root = root.expanduser().resolve(strict=True)
    regular_dir(root)
    manifest_path = root / "manifest.json"
    success_path = root / "_SUCCESS"
    regular_file(manifest_path)
    regular_file(success_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise PilotError("published manifest schema drift")
    if raw != canonical_json_bytes(manifest):
        raise PilotError("published manifest is not canonical JSON")
    checksum = manifest.get("manifest_sha256")
    if not is_sha256(checksum) or manifest_self_hash(manifest) != checksum:
        raise PilotError("published manifest self-hash drift")
    if success_path.read_bytes() not in {
        checksum.encode("ascii"),
        (checksum + "\n").encode("ascii"),
    }:
        raise PilotError("_SUCCESS does not bind the manifest")
    return manifest, raw


def validate_artifact_files(
    root: Path,
    manifest: dict[str, Any],
    *,
    verify_blob_sha256: bool = True,
) -> list[IndexEntry]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise PilotError("manifest files mapping missing")
    selection = files.get("selection")
    index = files.get("index")
    blobs = files.get("blobs")
    if not isinstance(selection, dict) or not isinstance(index, dict) or not isinstance(blobs, list):
        raise PilotError("manifest file bindings malformed")
    expected_names = {"manifest.json", "_SUCCESS", selection.get("name"), index.get("name")}
    for blob_id, entry in enumerate(blobs):
        if not isinstance(entry, dict) or entry.get("blob_id") != blob_id:
            raise PilotError("blob manifest sequence drift")
        expected_names.add(entry.get("name"))
    if None in expected_names:
        raise PilotError("manifest contains a missing filename")
    names = [selection.get("name"), index.get("name"), *[entry.get("name") for entry in blobs]]
    if any(
        not isinstance(name, str) or Path(name).name != name or not name
        for name in names
    ) or len(set(names)) != len(names):
        raise PilotError("manifest filenames are malformed or duplicated")
    live_names = {path.name for path in root.iterdir()}
    if live_names != expected_names:
        raise PilotError(
            f"published file set drift missing={sorted(expected_names-live_names)} "
            f"extra={sorted(live_names-expected_names)}"
        )
    for entry in [selection, index]:
        path = root / entry["name"]
        metadata = regular_file(path)
        if metadata.st_size != entry.get("bytes") or sha256_file(path) != entry.get("sha256"):
            raise PilotError(f"published file binding drift: {path.name}")
    for entry in blobs:
        path = root / entry["name"]
        metadata = regular_file(path)
        if metadata.st_size != entry.get("bytes"):
            raise PilotError(f"published blob size drift: {path.name}")
        if verify_blob_sha256 and sha256_file(path) != entry.get("sha256"):
            raise PilotError(f"published blob SHA-256 drift: {path.name}")

    raw_selection = (root / selection["name"]).read_bytes()
    selection_rows: list[tuple[int, int, str]] = []
    selection_digest = hashlib.sha256()
    for line_number, line in enumerate(raw_selection.splitlines(), start=1):
        try:
            value = json.loads(line)
        except Exception as exc:
            raise PilotError(f"invalid selection JSONL at line {line_number}") from exc
        if not isinstance(value, dict) or set(value) != {
            "catalog_local_index",
            "image_path",
            "row_id",
        }:
            raise PilotError(f"selection JSONL schema drift at line {line_number}")
        local_index = value["catalog_local_index"]
        row_id = value["row_id"]
        image_path = value["image_path"]
        if type(local_index) is not int or type(row_id) is not int or not isinstance(image_path, str):
            raise PilotError(f"selection JSONL type drift at line {line_number}")
        if line + b"\n" != canonical_json_bytes(value):
            raise PilotError(f"selection JSONL is not canonical at line {line_number}")
        raw_path = image_path.encode("utf-8")
        selection_digest.update(struct.pack("<qqI", local_index, row_id, len(raw_path)))
        selection_digest.update(raw_path)
        selection_rows.append((local_index, row_id, image_path))
    if (
        len(selection_rows) != manifest.get("selection", {}).get("sample_rows")
        or [row[0] for row in selection_rows] != sorted({row[0] for row in selection_rows})
        or selection_digest.hexdigest()
        != manifest.get("selection", {}).get("selection_sha256")
    ):
        raise PilotError("selection JSONL coverage/digest drift")

    raw_index = (root / index["name"]).read_bytes()
    if len(raw_index) % INDEX_RECORD_BYTES:
        raise PilotError("index file is not fixed-width")
    entries = [
        IndexEntry(*INDEX_STRUCT.unpack_from(raw_index, offset))
        for offset in range(0, len(raw_index), INDEX_RECORD_BYTES)
    ]
    if len(entries) != manifest.get("selection", {}).get("sample_rows"):
        raise PilotError("index row count/selection mismatch")
    if [entry.row_id for entry in entries] != sorted({entry.row_id for entry in entries}):
        raise PilotError("index row_id order/uniqueness drift")
    if [entry.row_id for entry in entries] != [row[1] for row in selection_rows]:
        raise PilotError("index row_ids do not match the audit selection")
    by_blob: list[list[IndexEntry]] = [[] for _ in blobs]
    for entry in entries:
        if entry.blob_id >= len(blobs) or entry.length <= 0:
            raise PilotError("index blob_id/length out of range")
        by_blob[entry.blob_id].append(entry)
    expected_offsets = [0] * len(blobs)
    for blob_id, blob_entries in enumerate(by_blob):
        for entry in sorted(blob_entries, key=lambda value: value.offset):
            if entry.offset != expected_offsets[blob_id]:
                raise PilotError("index segments are not contiguous and padding-free")
            expected_offsets[blob_id] += entry.length
    if expected_offsets != [entry["bytes"] for entry in blobs]:
        raise PilotError("index offsets do not close at blob sizes")
    return entries


def select_replay_entries(
    entries: Sequence[IndexEntry],
    manifest: dict[str, Any],
    *,
    mode: str,
    seed: int,
    min_cold_regions: int,
) -> tuple[list[IndexEntry], dict[str, Any]]:
    if mode == "full_coverage":
        return list(entries), {
            "mode": mode,
            "artifact_segments": len(entries),
            "selected_segments": len(entries),
            "full_coverage": True,
            "cold_unique_region": False,
        }
    if mode != "cold_unique_region":
        raise PilotError(f"unsupported replay selection mode: {mode!r}")
    candidates: dict[tuple[int, int], tuple[bytes, IndexEntry]] = {}
    eligible_segments = 0
    for entry in entries:
        region_index = entry.offset // REPLAY_COLD_REGION_BYTES
        region_start = region_index * REPLAY_COLD_REGION_BYTES
        region_stop = region_start + REPLAY_COLD_REGION_BYTES
        potential_stop = (
            entry.offset + REPLAY_POTENTIAL_READAHEAD_BYTES + entry.length
        )
        aligned_get_start = (
            entry.offset // REPLAY_POTENTIAL_READAHEAD_BYTES
        ) * REPLAY_POTENTIAL_READAHEAD_BYTES
        aligned_potential_stop = (
            aligned_get_start + REPLAY_POTENTIAL_READAHEAD_BYTES + entry.length
        )
        if (
            potential_stop > region_stop
            or aligned_get_start < region_start
            or aligned_potential_stop > region_stop
        ):
            continue
        eligible_segments += 1
        score = hashlib.sha256(
            struct.pack(
                "<QqIQI",
                seed & 0xFFFFFFFFFFFFFFFF,
                entry.row_id,
                entry.blob_id,
                entry.offset,
                entry.length,
            )
        ).digest()
        key = (entry.blob_id, region_index)
        previous = candidates.get(key)
        if previous is None or score < previous[0]:
            candidates[key] = (score, entry)
    selected = sorted((value[1] for value in candidates.values()), key=lambda entry: entry.row_id)
    if len(selected) < min_cold_regions:
        raise PilotError(
            f"cold-unique-region population too small: {len(selected)} < {min_cold_regions}"
        )
    keys = [(entry.blob_id, entry.offset // REPLAY_COLD_REGION_BYTES) for entry in selected]
    if len(keys) != len(set(keys)):
        raise PilotError("cold replay selected more than one segment in a guard region")
    for entry in selected:
        region_stop = (
            entry.offset // REPLAY_COLD_REGION_BYTES + 1
        ) * REPLAY_COLD_REGION_BYTES
        if (
            entry.offset + REPLAY_POTENTIAL_READAHEAD_BYTES + entry.length
            > region_stop
        ):
            raise PilotError("cold replay potential read range crosses its guard region")
        aligned_get_start = (
            entry.offset // REPLAY_POTENTIAL_READAHEAD_BYTES
        ) * REPLAY_POTENTIAL_READAHEAD_BYTES
        if (
            aligned_get_start
            < (entry.offset // REPLAY_COLD_REGION_BYTES) * REPLAY_COLD_REGION_BYTES
            or aligned_get_start
            + REPLAY_POTENTIAL_READAHEAD_BYTES
            + entry.length
            > region_stop
        ):
            raise PilotError(
                "cold replay 8-MiB-aligned GET range crosses its 16-MiB guard region"
            )
    digest = hashlib.sha256()
    for entry in selected:
        digest.update(
            INDEX_STRUCT.pack(
                entry.row_id,
                entry.blob_id,
                entry.offset,
                entry.length,
                entry.sha256_raw,
            )
        )
    return selected, {
        "mode": mode,
        "artifact_segments": len(entries),
        "eligible_segments": eligible_segments,
        "selected_segments": len(selected),
        "selected_region_count": len(keys),
        "region_bytes": REPLAY_COLD_REGION_BYTES,
        "potential_readahead_guard_bytes": REPLAY_POTENTIAL_READAHEAD_BYTES,
        "max_complete_segments_per_region": 1,
        "potential_range_contract": (
            "both_[offset,offset+8MiB+length)_and_"
            "[floor8MiB(offset),floor8MiB(offset)+8MiB+length)_"
            "remain_within_one_16MiB_region"
        ),
        "get_start_alignment_bytes": REPLAY_POTENTIAL_READAHEAD_BYTES,
        "selection_sha256": digest.hexdigest(),
        "full_coverage": False,
        "cold_unique_region": True,
    }


def _replay_worker(
    worker_id: int,
    blob_paths: list[str],
    requests: list[IndexEntry],
    ready: Any,
    start: Any,
    result: Any,
) -> None:
    descriptors: list[int] = []
    sent_ready = False
    try:
        identities: list[FileIdentity] = []
        for path in blob_paths:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            descriptors.append(descriptor)
            identities.append(FileIdentity.from_stat(os.fstat(descriptor)))
        ready.put((worker_id, "READY"))
        sent_ready = True
        if not start.wait(timeout=300):
            raise PilotError("replay start barrier timeout")
        latencies: list[int] = []
        requested_bytes = 0
        returned_bytes = 0
        for entry in requests:
            before = time.perf_counter_ns()
            payload = os.pread(descriptors[entry.blob_id], entry.length, entry.offset)
            elapsed = time.perf_counter_ns() - before
            requested_bytes += entry.length
            returned_bytes += len(payload)
            if len(payload) != entry.length or hashlib.sha256(payload).digest() != entry.sha256_raw:
                raise PilotError(f"random pread verification failed row_id={entry.row_id}")
            latencies.append(elapsed)
        for descriptor, identity in zip(descriptors, identities):
            if FileIdentity.from_stat(os.fstat(descriptor)) != identity:
                raise PilotError("blob fd identity changed during replay")
        result.put(
            {
                "worker_id": worker_id,
                "status": "PASS",
                "objects": len(requests),
                "requested_bytes": requested_bytes,
                "returned_bytes": returned_bytes,
                "latencies_ns": latencies,
            }
        )
    except BaseException as exc:
        if not sent_ready:
            ready.put((worker_id, "FAIL"))
        result.put(
            {
                "worker_id": worker_id,
                "status": "FAIL",
                "exception_class": type(exc).__name__,
            }
        )
    finally:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


def replay_once(
    root: Path,
    manifest: dict[str, Any],
    entries: list[IndexEntry],
    *,
    workers: int,
    seed: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    shuffled = list(entries)
    random.Random(seed).shuffle(shuffled)
    chunks = [shuffled[index::workers] for index in range(workers)]
    blob_paths = [str(root / entry["name"]) for entry in manifest["files"]["blobs"]]
    context = mp.get_context("spawn")
    ready = context.Queue()
    result = context.Queue()
    start = context.Event()
    processes = [
        context.Process(
            target=_replay_worker,
            args=(index, blob_paths, chunks[index], ready, start, result),
        )
        for index in range(workers)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + timeout_seconds
    try:
        ready_ids: set[int] = set()
        while len(ready_ids) < workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PilotError("workers did not reach replay barrier")
            try:
                worker_id, status = ready.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                continue
            if status != "READY":
                raise PilotError(f"replay worker failed before barrier worker={worker_id}")
            if worker_id in ready_ids:
                raise PilotError("invalid replay readiness message")
            ready_ids.add(worker_id)
        started_ns = time.perf_counter_ns()
        start.set()
        results: list[dict[str, Any]] = []
        while len(results) < workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PilotError("replay result timeout")
            try:
                results.append(result.get(timeout=min(remaining, 5.0)))
            except queue.Empty:
                continue
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        finished_ns = time.perf_counter_ns()
        if any(process.exitcode != 0 for process in processes):
            raise PilotError("replay worker process exited nonzero")
        if any(value.get("status") != "PASS" for value in results):
            classes = sorted(
                {value.get("exception_class", "unknown") for value in results if value.get("status") != "PASS"}
            )
            raise PilotError(f"replay worker verification failed classes={classes}")
        if {value.get("worker_id") for value in results} != set(range(workers)):
            raise PilotError("replay worker result identity drift")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    elapsed_s = (finished_ns - started_ns) / 1e9
    latencies = [item for value in results for item in value["latencies_ns"]]
    requested = sum(value["requested_bytes"] for value in results)
    returned = sum(value["returned_bytes"] for value in results)
    if len(latencies) != len(entries) or requested != returned:
        raise PilotError("replay accounting does not close")
    return {
        "seed": seed,
        "workers": workers,
        "objects": len(entries),
        "requested_bytes": requested,
        "returned_bytes": returned,
        "elapsed_seconds": elapsed_s,
        "objects_per_second": len(entries) / elapsed_s,
        "mib_per_second": returned / MIB / elapsed_s,
        "userspace_read_amplification": returned / requested,
        "physical_backend_read_amplification_observable": False,
        "per_pread": timing_summary_ns(latencies),
    }


def replay_pilot(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser().resolve(strict=True)
    code = snapshot_code(args.job_spec)
    manifest, manifest_raw = load_published_manifest(root)
    # Do not sequentially hash blobs before a replay arm: that would populate
    # FSx/Mountpoint caches.  The first cold-region arm verifies every selected
    # guard-region segment; the later integrity arm verifies every index segment.
    artifact_entries = validate_artifact_files(
        root, manifest, verify_blob_sha256=False
    )
    entries, replay_selection = select_replay_entries(
        artifact_entries,
        manifest,
        mode=args.selection_mode,
        seed=args.seed,
        min_cold_regions=args.min_cold_regions,
    )
    controls_before = {
        "manifest_raw_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "success_raw_sha256": hashlib.sha256((root / "_SUCCESS").read_bytes()).hexdigest(),
    }
    repeats = [
        replay_once(
            root,
            manifest,
            entries,
            workers=args.workers,
            seed=args.seed + repeat,
            timeout_seconds=args.timeout_seconds,
        )
        for repeat in range(args.repeats)
    ]
    closing_manifest, closing_raw = load_published_manifest(root)
    if closing_manifest != manifest or hashlib.sha256(closing_raw).hexdigest() != controls_before["manifest_raw_sha256"]:
        raise PilotError("artifact manifest changed during replay")
    if hashlib.sha256((root / "_SUCCESS").read_bytes()).hexdigest() != controls_before["success_raw_sha256"]:
        raise PilotError("artifact success marker changed during replay")
    validate_code_snapshot(code)
    report = {
        "schema_version": "offline_gpic_blob_replay_v1",
        "pilot_only": True,
        "production_selection_qualified": False,
        "generated_at": utc_now(),
        "artifact_root": str(root),
        "artifact_manifest_sha256": manifest["manifest_sha256"],
        "backend": manifest["backend"],
        "workers": args.workers,
        "arm_role": args.arm_role,
        "arm_sequence_index": args.arm_sequence_index,
        "capacity_authority": args.arm_role == "capacity_authority",
        "selection_mode": args.selection_mode,
        "replay_selection": replay_selection,
        "artifact_sample_rows": manifest["selection"]["sample_rows"],
        "persistent_blob_fds_per_worker": len(manifest["files"]["blobs"]),
        "direct_io": {
            "o_direct_used": False,
            "o_direct_required": False,
            "status": "optional_diagnostic_not_yet_jcc_validated",
        },
        "access_order": (
            "deterministic_full-coverage_random_permutation"
            if args.selection_mode == "full_coverage"
            else "deterministic_cold-unique-region_random_permutation"
        ),
        "pre_timing_blob_sha256_verified": False,
        "pre_timing_validation": "manifest_index_selection_sha256_and_blob_size_only",
        "timed_content_validation": (
            "every_index_segment_exact_pread_and_sha256"
            if args.selection_mode == "full_coverage"
            else "every_selected_cold_region_segment_exact_pread_and_sha256"
        ),
        "production_capacity_status": (
            "INCONCLUSIVE_FOR_PRODUCTION"
            if manifest["backend"] == "mountpoint_s3"
            and manifest["selection"]["sample_rows"] < 100_000
            else "PILOT_DIAGNOSTIC_ONLY"
        ),
        "repeats": repeats,
        "code": code,
        "replay_runtime": runtime_identity(),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise PilotError("replay output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(output, canonical_json_bytes(report))
    print(
        f"OFFLINE_GPIC_BLOB_REPLAY_PASS backend={manifest['backend']} output={output} "
        f"img_s={percentile([x['objects_per_second'] for x in repeats],0.5):.3f}",
        flush=True,
    )
    return report


def add_catalog_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--sample-rows", type=int, required=True)
    parser.add_argument("--coverage-intervals", type=int, default=256)
    parser.add_argument("--min-unique-shards", type=int, default=7900)
    parser.add_argument("--source-attempts", type=int, default=3)
    parser.add_argument("--source-block-bytes", type=int, default=MIB)
    parser.add_argument("--retry-base-seconds", type=float, default=0.05)
    parser.add_argument("--source-order", choices=("path", "catalog"), default="path")
    parser.add_argument("--job-spec", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("source-scan")
    add_catalog_args(scan)
    scan.add_argument("--threads-matrix", required=True)
    scan.add_argument("--abba", action="store_true")
    scan.add_argument("--exclude-sample-rows", type=int, default=0)
    scan.add_argument("--max-byte-imbalance", type=float, default=1.25)
    scan.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser("build")
    add_catalog_args(build)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--allowed-output-prefix", type=Path, required=True)
    build.add_argument("--backend", choices=("fsx_posix", "mountpoint_s3"), required=True)
    build.add_argument("--write-authority", required=True)
    build.add_argument("--threads", type=int, required=True)
    build.add_argument("--blob-target-bytes", type=int, required=True)
    build.add_argument("--max-total-source-bytes", type=int, default=0)
    build.add_argument("--stripe-count", type=int, default=0)
    build.add_argument("--stripe-size-mib", type=int, default=64)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--root", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--workers", type=int, default=16)
    replay.add_argument("--repeats", type=int, default=1)
    replay.add_argument("--seed", type=int, default=20260712)
    replay.add_argument("--timeout-seconds", type=int, default=3600)
    replay.add_argument(
        "--selection-mode",
        choices=("cold_unique_region", "full_coverage"),
        required=True,
    )
    replay.add_argument(
        "--arm-role",
        choices=("capacity_authority", "integrity_warm"),
        required=True,
    )
    replay.add_argument("--arm-sequence-index", type=int, required=True)
    replay.add_argument("--min-cold-regions", type=int, default=64)
    replay.add_argument("--job-spec", type=Path, required=True)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.command in {"source-scan", "build"}:
        strict_int(args.sample_rows, "sample_rows", 1)
        strict_int(args.coverage_intervals, "coverage_intervals", 1)
        strict_int(args.min_unique_shards, "min_unique_shards", 1)
        strict_int(args.source_attempts, "source_attempts", 1)
        strict_int(args.source_block_bytes, "source_block_bytes", 1)
        if args.sample_rows < args.coverage_intervals:
            raise PilotError("sample_rows must cover every requested interval")
        if args.min_unique_shards > args.sample_rows:
            raise PilotError("min_unique_shards exceeds sample_rows")
        if not math.isfinite(args.retry_base_seconds) or args.retry_base_seconds < 0:
            raise PilotError("retry_base_seconds must be finite and nonnegative")
    if args.command == "source-scan":
        strict_int(args.exclude_sample_rows, "exclude_sample_rows", 0)
        if not math.isfinite(args.max_byte_imbalance) or args.max_byte_imbalance < 1.0:
            raise PilotError("max_byte_imbalance must be finite and >= 1")
    if args.command == "build":
        strict_int(args.threads, "threads", 1)
        strict_int(args.blob_target_bytes, "blob_target_bytes", 1)
        strict_int(args.max_total_source_bytes, "max_total_source_bytes", 0)
        strict_int(args.stripe_count, "stripe_count", 0)
        strict_int(args.stripe_size_mib, "stripe_size_mib", 1)
        if args.backend == "mountpoint_s3" and not (
            1 <= args.max_total_source_bytes <= 8 * (1 << 30)
        ):
            raise PilotError(
                "Mountpoint-S3 build requires max_total_source_bytes in [1,8GiB]"
            )
    if args.command == "replay":
        strict_int(args.workers, "workers", 1)
        strict_int(args.repeats, "repeats", 1)
        strict_int(args.timeout_seconds, "timeout_seconds", 1)
        strict_int(args.arm_sequence_index, "arm_sequence_index", 0)
        strict_int(args.min_cold_regions, "min_cold_regions", 1)
        if args.arm_role == "capacity_authority" and (
            args.selection_mode != "cold_unique_region"
            or args.arm_sequence_index != 0
            or args.repeats != 1
        ):
            raise PilotError(
                "capacity_authority must be the unique sequence-0, one-repeat cold_unique_region arm"
            )
        if args.arm_role == "integrity_warm" and (
            args.selection_mode != "full_coverage"
            or args.arm_sequence_index < 1
        ):
            raise PilotError(
                "integrity_warm must be a post-authority full_coverage arm"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if args.command == "source-scan":
        report = source_scan(args)
        if report["status"] != "PASS":
            return 3
    elif args.command == "build":
        build_pilot(args)
    elif args.command == "replay":
        replay_pilot(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except SystemExit:
        # Preserve argparse's intentional exit status (for example --help).
        raise
    except KeyboardInterrupt:
        print(
            "OFFLINE_GPIC_BLOB_PILOT_FATAL class=KeyboardInterrupt",
            file=sys.stderr,
            flush=True,
        )
        exit_code = 130
    except CommittedSyncUnconfirmed:
        print(
            "OFFLINE_GPIC_BLOB_PILOT_COMMITTED_SYNC_UNCONFIRMED_FATAL",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        exit_code = 5
    except BaseException as exc:
        print(
            f"OFFLINE_GPIC_BLOB_PILOT_FATAL class={type(exc).__name__}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        exit_code = 1
    raise SystemExit(exit_code)
