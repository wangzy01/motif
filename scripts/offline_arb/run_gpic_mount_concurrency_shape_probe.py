#!/usr/bin/env python3
"""Cold-within-capture GPIC mount concurrency-shape diagnostic.

This is deliberately independent from the production loader.  It compares a
single shared thread pool with the current eight-process/two-thread shape while
holding the node-wide source-object contract constant:

* every physical image is used by exactly one timed cell;
* source files are opened directly, fully read, and fstat-identified before and
  after the read; empty/short/changed/non-regular inputs are fatal;
* the source-read timer stops before a separate atomic /local write + file and
  directory fsync + exact reread verification phase starts;
* the frozen 20k cache is reconciled against the formal GPIC catalog, then a
  sparse sample is drawn from every catalog row group with all frozen-cache UIDs
  excluded before any timed source I/O;
* artifacts contain only aggregates and digests, never source paths or keys.

The A16/B16 comparison is an AB...BA crossover.  A8/A24/A32 are curve points in
the middle.  Historical T2 p1x32=88.2259 img/s is reported only as a warm-loop
upper-bound reference; it is not a capacity gate.  The capacity gate is the
formal 28.8 img/s/node demand times 1.25 = 36.0 img/s/node.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import queue
import re
import secrets
import stat
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "offline_arb_gpic_mount_concurrency_shape_v1"
RUN_ID = "offline_arb_gpic_mount_concurrency_shape_20260712r1"
SOURCE_CACHE_BYTES = 6_220_043
SOURCE_CACHE_SHA256 = "3c45ba6e3898b546dbc191e6b708e0bffc7047fd059708b7af8cc0e190cfa298"
SOURCE_CACHE_ITEMS = 20_000
FORMAL_SAMPLE_POOL_ITEMS = 20_000
FORMAL_CANDIDATES_PER_ROW_GROUP = 70
CATALOG_HEADER_BYTES = 95_794
CATALOG_HEADER_SHA256 = "a06567fe3359efa730a0cd099ee5245b039c1648502d304f2e35f4487727a507"
CATALOG_GPIC_BYTES = 2_286_611_460
CATALOG_GPIC_SHA256 = "bf2883d3665d8730c157d32180a81c3ad888b219b8f58c0e049ae7317cf4b0d8"
EXPECTED_SOURCE_ROOT = "/s3-mewtant-raw-datasets/stanford-vision-lab/gpic_umziped"
EXPECTED_CATALOG_SCHEMA = "offline_arb_catalog_v1"
EXPECTED_CATALOG_ROWS = 20_000_000
SELECTION_SEED = "offline-arb-gpic-concurrency-shape-20260712r1"
PER_CELL_ITEMS = 2_800
CAPTURE_SELECTED_ITEMS = PER_CELL_ITEMS * 7
SOURCE_CHUNK_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_CELL_PAYLOAD_BYTES = 8 * 1024 * 1024 * 1024
FORMAL_DEMAND_IMG_S = 28.8
CAPACITY_MARGIN = 1.25
CAPACITY_GATE_IMG_S = FORMAL_DEMAND_IMG_S * CAPACITY_MARGIN
HISTORICAL_WARM_LOOP_IMG_S = 88.22592592592592
LOCAL_WORLD_SIZE = 8
SAFE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")
HEX64_RE = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")

# AB...BA controls monotonic mount drift for the architecture comparison.
CELL_SPECS: tuple[tuple[str, str, int, int], ...] = (
    ("a16_x", "shared_thread_pool", 1, 16),
    ("b16_x", "eight_fragmented_processes", 8, 2),
    ("a08_mid", "shared_thread_pool", 1, 8),
    ("a24_mid", "shared_thread_pool", 1, 24),
    ("a32_mid", "shared_thread_pool", 1, 32),
    ("b16_y", "eight_fragmented_processes", 8, 2),
    ("a16_y", "shared_thread_pool", 1, 16),
)


class ProbeError(RuntimeError):
    """Expected fail-closed error whose message is a path-free class token."""


@dataclass(frozen=True)
class SourceItem:
    uid: str
    src_root: str
    src_shard: str
    src_key: str
    ext: str
    width: int = 0
    height: int = 0

    def source_path(self) -> str:
        return os.path.join(
            self.src_root, "images", "train", self.src_shard, self.src_key + self.ext
        )

    def anonymous_id(self) -> str:
        return canonical_sha256(
            {
                "schema": "gpic_source_item_v1",
                "uid": self.uid,
                "src_root": self.src_root,
                "src_shard": self.src_shard,
                "src_key": self.src_key,
                "ext": self.ext,
            }
        )


@dataclass
class SourceRead:
    item: SourceItem
    payload: bytes
    elapsed_ns: int
    identity_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = SOURCE_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def require_regular_input(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    try:
        if path.is_symlink():
            raise ProbeError("input_symlink")
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ProbeError("input_not_regular")
        if before.st_size != expected_bytes:
            raise ProbeError("input_size_drift")
        observed = sha256_file(path)
        after = path.stat()
    except ProbeError:
        raise
    except OSError as exc:
        raise ProbeError(error_class(exc, "input_io")) from None
    if file_identity(before) != file_identity(after):
        raise ProbeError("input_identity_changed")
    if observed != expected_sha256:
        raise ProbeError("input_sha256_drift")


def error_class(exc: BaseException, phase: str) -> str:
    if isinstance(exc, ProbeError):
        token = str(exc)
        if re.fullmatch(r"[a-z0-9_]{1,96}", token):
            return token
        return "probe_error"
    if isinstance(exc, OSError):
        errno = exc.errno if isinstance(exc.errno, int) and exc.errno >= 0 else 0
        return f"{phase}_oserror_e{errno}"
    return f"{phase}_{type(exc).__name__.lower()}"


def atomic_publish_json(path: Path, value: Any) -> None:
    raw = canonical_json_bytes(value)
    if path.exists() or path.is_symlink():
        raise ProbeError("artifact_clobber")
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    descriptor = -1
    try:
        descriptor = os.open(
            tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ProbeError("artifact_short_write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(tmp, path, follow_symlinks=False)
        tmp.unlink()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def artifact_attestation(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if file_identity(before) != file_identity(after):
        raise ProbeError("artifact_identity_changed")
    return {"bytes": int(after.st_size), "sha256": digest}


def load_cache(path: Path) -> list[SourceItem]:
    require_regular_input(path, SOURCE_CACHE_BYTES, SOURCE_CACHE_SHA256)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ProbeError("cache_json_invalid") from None
    if not isinstance(value, dict) or set(value) != {"items", "resolve_errors"}:
        raise ProbeError("cache_schema_drift")
    if value["resolve_errors"] != 0 or not isinstance(value["items"], list):
        raise ProbeError("cache_resolution_drift")
    if len(value["items"]) != SOURCE_CACHE_ITEMS:
        raise ProbeError("cache_item_count_drift")
    items: list[SourceItem] = []
    physical: set[tuple[str, str, str, str]] = set()
    uids: set[str] = set()
    for raw in value["items"]:
        if not isinstance(raw, dict) or set(raw) != {
            "uid", "src_root", "src_shard", "src_key", "ext"
        }:
            raise ProbeError("cache_item_schema_drift")
        if raw["src_root"] != EXPECTED_SOURCE_ROOT:
            raise ProbeError("cache_source_root_drift")
        shard = raw["src_shard"]
        key = raw["src_key"]
        ext = raw["ext"]
        uid = raw["uid"]
        if not isinstance(shard, str) or re.fullmatch(r"[0-9]{5}", shard) is None:
            raise ProbeError("cache_shard_invalid")
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ProbeError("cache_key_invalid")
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ProbeError("cache_extension_invalid")
        if uid != "gpic:" + key:
            raise ProbeError("cache_uid_invalid")
        identity = (EXPECTED_SOURCE_ROOT, shard, key, ext)
        if identity in physical or uid in uids:
            raise ProbeError("cache_duplicate_source")
        physical.add(identity)
        uids.add(uid)
        items.append(SourceItem(uid, EXPECTED_SOURCE_ROOT, shard, key, ext))
    return items


def validate_catalog_header(header_path: Path, catalog_path: Path) -> dict[str, Any]:
    require_regular_input(header_path, CATALOG_HEADER_BYTES, CATALOG_HEADER_SHA256)
    try:
        header = json.loads(header_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ProbeError("catalog_header_json_invalid") from None
    if header.get("schema_version") != EXPECTED_CATALOG_SCHEMA:
        raise ProbeError("catalog_header_schema_drift")
    catalog_files = header.get("files")
    entry = catalog_files.get("gpic") if isinstance(catalog_files, dict) else None
    if not isinstance(entry, dict):
        raise ProbeError("catalog_gpic_entry_missing")
    if (
        entry.get("name") != "gpic.parquet"
        or entry.get("bytes") != CATALOG_GPIC_BYTES
        or entry.get("sha256") != CATALOG_GPIC_SHA256
        or entry.get("rows") != EXPECTED_CATALOG_ROWS
        or entry.get("compression") != "zstd"
    ):
        raise ProbeError("catalog_gpic_entry_drift")
    require_regular_input(catalog_path, CATALOG_GPIC_BYTES, CATALOG_GPIC_SHA256)
    return {
        "header": {"bytes": CATALOG_HEADER_BYTES, "sha256": CATALOG_HEADER_SHA256},
        "gpic": {"bytes": CATALOG_GPIC_BYTES, "sha256": CATALOG_GPIC_SHA256},
    }


def catalog_source_item(uid: Any, width: Any, height: Any, image_path: Any) -> SourceItem:
    if type(uid) is not str or type(image_path) is not str:
        raise ProbeError("catalog_source_identity_invalid")
    pattern = re.compile(
        re.escape(EXPECTED_SOURCE_ROOT)
        + r"/images/train/([0-9]{5})/([0-9a-f]{64})(\.(?:jpg|jpeg|png|webp))"
    )
    match = pattern.fullmatch(image_path)
    if match is None:
        raise ProbeError("catalog_source_path_invalid")
    shard, key, ext = match.groups()
    if uid != "gpic:" + key:
        raise ProbeError("catalog_source_uid_invalid")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ProbeError("catalog_dimension_invalid")
    return SourceItem(uid, EXPECTED_SOURCE_ROOT, shard, key, ext, width, height)


def build_formal_unseen_sample(
    cache_items: Sequence[SourceItem], catalog_path: Path
) -> dict[str, Any]:
    """Build a sparse formal-catalog sample excluding every frozen-cache UID.

    The catalog is still streamed in full so all 20k frozen cache identities are
    reconciled exactly.  Timed candidates, however, are deterministic sparse rows
    from every row group and have zero UID overlap with that repeatedly-used cache.
    """
    try:
        import pyarrow.parquet as pq
    except Exception:
        raise ProbeError("pyarrow_unavailable") from None
    wanted = {item.uid: item for item in cache_items}
    if len(wanted) != SOURCE_CACHE_ITEMS:
        raise ProbeError("cache_uid_set_drift")
    found_cache: set[str] = set()
    selected: list[SourceItem] = []
    row_group_counts: dict[int, int] = {}
    try:
        parquet = pq.ParquetFile(catalog_path)
        if parquet.metadata.num_rows != EXPECTED_CATALOG_ROWS:
            raise ProbeError("catalog_row_count_drift")
        required = ("uid", "width", "height", "image_path")
        if not set(required).issubset(set(parquet.schema.names)):
            raise ProbeError("catalog_column_drift")
        row_groups = parquet.metadata.num_row_groups
        base, remainder = divmod(FORMAL_SAMPLE_POOL_ITEMS, row_groups)
        if base <= 0 or base + 1 > FORMAL_CANDIDATES_PER_ROW_GROUP:
            raise ProbeError("formal_sample_density_drift")
        extra_seed = int.from_bytes(
            hashlib.sha256((SELECTION_SEED + ":extra-row-groups").encode("ascii")).digest()[:8],
            "big",
        )
        import random

        extra_groups = set(random.Random(extra_seed).sample(range(row_groups), remainder))
        for row_group in range(row_groups):
            table = parquet.read_row_group(
                row_group, columns=["uid", "width", "height", "image_path"], use_threads=False
            )
            uid_values = table.column("uid").to_pylist()
            width_column = table.column("width")
            height_column = table.column("height")
            path_column = table.column("image_path")
            for position, uid in enumerate(uid_values):
                cached = wanted.get(uid)
                if cached is None:
                    continue
                if uid in found_cache:
                    raise ProbeError("catalog_cache_uid_duplicate")
                if path_column[position].as_py() != cached.source_path():
                    raise ProbeError("catalog_cache_path_mismatch")
                found_cache.add(uid)

            wanted_here = base + int(row_group in extra_groups)
            position_seed = int.from_bytes(
                hashlib.sha256(
                    (SELECTION_SEED + f":row-group:{row_group}").encode("ascii")
                ).digest()[:8],
                "big",
            )
            candidate_positions = random.Random(position_seed).sample(
                range(table.num_rows), min(FORMAL_CANDIDATES_PER_ROW_GROUP, table.num_rows)
            )
            accepted = 0
            for position in candidate_positions:
                uid = uid_values[position]
                if uid in wanted:
                    continue
                item = catalog_source_item(
                    uid,
                    width_column[position].as_py(),
                    height_column[position].as_py(),
                    path_column[position].as_py(),
                )
                selected.append(item)
                accepted += 1
                if accepted == wanted_here:
                    break
            if accepted != wanted_here:
                raise ProbeError("formal_sample_row_group_short")
            row_group_counts[row_group] = accepted
    except ProbeError:
        raise
    except Exception as exc:
        raise ProbeError(error_class(exc, "catalog_scan")) from None
    if found_cache != set(wanted):
        raise ProbeError("catalog_cache_membership_incomplete")
    if len(selected) != FORMAL_SAMPLE_POOL_ITEMS:
        raise ProbeError("formal_sample_count_drift")
    anonymous = [item.anonymous_id() for item in selected]
    if len(set(anonymous)) != len(anonymous):
        raise ProbeError("formal_sample_duplicate_source")
    if {item.uid for item in selected} & set(wanted):
        raise ProbeError("formal_sample_cache_overlap")
    return {
        "items": selected,
        "attestation": {
            "formal_catalog_rows": EXPECTED_CATALOG_ROWS,
            "formal_catalog_row_groups": row_groups,
            "frozen_cache_items": len(cache_items),
            "frozen_cache_matched_once": len(found_cache),
            "formal_sample_pool_items": len(selected),
            "formal_sample_frozen_cache_uid_overlap": 0,
            "candidates_per_row_group_budget": FORMAL_CANDIDATES_PER_ROW_GROUP,
            "selected_per_row_group_min": min(row_group_counts.values()),
            "selected_per_row_group_max": max(row_group_counts.values()),
            "row_group_distribution_sha256": canonical_sha256(sorted(row_group_counts.items())),
            "formal_sample_sha256": canonical_sha256(
                sorted((item.anonymous_id(), item.width, item.height) for item in selected)
            ),
        },
    }


def area_bin(item: SourceItem) -> int:
    return int(math.floor(math.log2(max(1, item.width * item.height))))


def build_balanced_cells(items: Sequence[SourceItem]) -> dict[str, list[SourceItem]]:
    """Balance format, pixel-area proxy and shard across seven equal cells.

    Source byte size cannot be known without touching mount metadata or content.
    Pixel area is therefore the pre-I/O size proxy; exact byte balance is checked
    post-read and can make the final verdict inconclusive.
    """
    if len(items) != FORMAL_SAMPLE_POOL_ITEMS:
        raise ProbeError("selection_input_count_drift")
    ordered = sorted(
        items,
        key=lambda item: (
            item.ext,
            area_bin(item),
            item.src_shard,
            hashlib.sha256((SELECTION_SEED + item.uid).encode("ascii")).digest(),
        ),
    )
    cell_names = [spec[0] for spec in CELL_SPECS]
    provisional: dict[str, list[SourceItem]] = {name: [] for name in cell_names}
    for block_start in range(0, len(ordered), len(cell_names)):
        block = ordered[block_start : block_start + len(cell_names)]
        rotation = (block_start // len(cell_names)) % len(cell_names)
        for offset, item in enumerate(block):
            name = cell_names[(rotation + offset) % len(cell_names)]
            provisional[name].append(item)
    # Equal first-touch panels are more important than consuming the whole cache.
    # Deterministic hash trimming disperses the 400 holdouts across strata instead
    # of dropping a suffix (the list above is deliberately stratum-sorted).
    cells = {
        name: sorted(
            provisional[name],
            key=lambda item: hashlib.sha256(
                (SELECTION_SEED + ":trim:" + name + ":" + item.uid).encode("ascii")
            ).digest(),
        )[:PER_CELL_ITEMS]
        for name in cell_names
    }
    if any(len(value) != PER_CELL_ITEMS for value in cells.values()):
        raise ProbeError("selection_cell_count_drift")
    anonymous = [item.anonymous_id() for value in cells.values() for item in value]
    if len(anonymous) != CAPTURE_SELECTED_ITEMS or len(set(anonymous)) != len(anonymous):
        raise ProbeError("selection_physical_overlap")
    return cells


def cell_selection_summary(items: Sequence[SourceItem]) -> dict[str, Any]:
    ext_counts = Counter(item.ext for item in items)
    area_counts = Counter(str(area_bin(item)) for item in items)
    shard_counts = Counter(item.src_shard for item in items)
    return {
        "items": len(items),
        "selection_sha256": canonical_sha256([item.anonymous_id() for item in items]),
        "format_counts": dict(sorted(ext_counts.items())),
        "area_proxy_bin_counts": dict(sorted(area_counts.items())),
        "unique_shards": len(shard_counts),
        "max_items_per_shard": max(shard_counts.values(), default=0),
        "shard_distribution_sha256": canonical_sha256(sorted(shard_counts.items())),
    }


def exact_source_read(item: SourceItem) -> SourceRead:
    descriptor = -1
    started_ns = time.perf_counter_ns()
    try:
        descriptor = os.open(
            item.source_path(), os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProbeError("source_not_regular")
        if before.st_size <= 0:
            raise ProbeError("source_empty")
        if before.st_size > MAX_SOURCE_BYTES:
            raise ProbeError("source_size_over_limit")
        remaining = int(before.st_size)
        chunks: list[bytes] = []
        while remaining:
            block = os.read(descriptor, min(SOURCE_CHUNK_BYTES, remaining))
            if not block:
                raise ProbeError("source_short_read")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ProbeError("source_grew_during_read")
        after = os.fstat(descriptor)
        if file_identity(before) != file_identity(after):
            raise ProbeError("source_identity_changed")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ProbeError("source_length_mismatch")
        elapsed_ns = time.perf_counter_ns() - started_ns
        return SourceRead(
            item=item,
            payload=payload,
            elapsed_ns=elapsed_ns,
            identity_sha256=canonical_sha256(file_identity(before)),
        )
    except ProbeError:
        raise
    except BaseException as exc:
        raise ProbeError(error_class(exc, "source_read")) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise ProbeError("local_short_write")
        view = view[written:]


def exact_local_stage(read: SourceRead, local_dir: str, ordinal: int) -> int:
    started_ns = time.perf_counter_ns()
    directory = Path(local_dir)
    name = f"item-{ordinal:06d}.bin"
    destination = directory / name
    temporary = directory / (
        f".{name}.tmp.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}"
    )
    descriptor = -1
    directory_fd = -1
    verify_fd = -1
    try:
        if destination.exists() or destination.is_symlink():
            raise ProbeError("local_destination_clobber")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        write_all(descriptor, read.payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, destination)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        os.fsync(directory_fd)
        verify_fd = os.open(destination, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(verify_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != len(read.payload):
            raise ProbeError("local_identity_invalid")
        remaining = int(before.st_size)
        chunks: list[bytes] = []
        while remaining:
            block = os.read(verify_fd, min(SOURCE_CHUNK_BYTES, remaining))
            if not block:
                raise ProbeError("local_short_read")
            chunks.append(block)
            remaining -= len(block)
        if os.read(verify_fd, 1):
            raise ProbeError("local_grew_during_read")
        after = os.fstat(verify_fd)
        if file_identity(before) != file_identity(after):
            raise ProbeError("local_identity_changed")
        if b"".join(chunks) != read.payload:
            raise ProbeError("local_payload_mismatch")
        os.close(verify_fd)
        verify_fd = -1
        destination.unlink()
        os.fsync(directory_fd)
        return time.perf_counter_ns() - started_ns
    except ProbeError:
        raise
    except BaseException as exc:
        raise ProbeError(error_class(exc, "local_stage")) from None
    finally:
        for fd in (verify_fd, descriptor, directory_fd):
            if fd >= 0:
                os.close(fd)
        for path in (temporary, destination):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def quantiles_ns(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise ProbeError("latency_series_empty")
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
        return round(ordered[index] / 1_000_000.0, 3)

    return {
        "n": len(ordered),
        "p50_ms": pick(0.50),
        "p90_ms": pick(0.90),
        "p95_ms": pick(0.95),
        "p99_ms": pick(0.99),
        "max_ms": round(ordered[-1] / 1_000_000.0, 3),
    }


def summarize_source(reads: Sequence[SourceRead], wall_ns: int) -> dict[str, Any]:
    if wall_ns <= 0 or not reads:
        raise ProbeError("source_window_invalid")
    total_bytes = sum(len(read.payload) for read in reads)
    ledger = [
        (
            read.item.anonymous_id(),
            len(read.payload),
            read.identity_sha256,
            read.elapsed_ns,
        )
        for read in reads
    ]
    seconds = wall_ns / 1_000_000_000.0
    return {
        "files": len(reads),
        "bytes": total_bytes,
        "wall_seconds": round(seconds, 6),
        "files_per_second": len(reads) / seconds,
        "mib_per_second": total_bytes / (1024 * 1024) / seconds,
        "mean_bytes": total_bytes / len(reads),
        "effective_concurrency": sum(read.elapsed_ns for read in reads) / wall_ns,
        "latency": quantiles_ns([read.elapsed_ns for read in reads]),
        "exact_read_ledger_sha256": canonical_sha256(ledger),
    }


def summarize_local(reads: Sequence[SourceRead], latencies: Sequence[int], wall_ns: int) -> dict[str, Any]:
    if wall_ns <= 0 or len(reads) != len(latencies):
        raise ProbeError("local_window_invalid")
    total_bytes = sum(len(read.payload) for read in reads)
    seconds = wall_ns / 1_000_000_000.0
    return {
        "files": len(reads),
        "bytes": total_bytes,
        "wall_seconds": round(seconds, 6),
        "files_per_second": len(reads) / seconds,
        "mib_per_second": total_bytes / (1024 * 1024) / seconds,
        "latency": quantiles_ns(latencies),
        "verified_files": len(latencies),
        "contract": "atomic_replace_file_fsync_dir_fsync_exact_reread",
    }


def read_source_panel(
    items: Sequence[SourceItem], concurrency: int, byte_budget: int
) -> list[SourceRead]:
    if concurrency <= 0 or byte_budget <= 0:
        raise ProbeError("source_panel_contract_invalid")
    state = {"bytes": 0}
    lock = threading.Lock()

    def bounded_read(item: SourceItem) -> SourceRead:
        result = exact_source_read(item)
        with lock:
            state["bytes"] += len(result.payload)
            if state["bytes"] > byte_budget:
                raise ProbeError("source_panel_byte_budget_exceeded")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(bounded_read, items))


def run_shared_cell(items: Sequence[SourceItem], concurrency: int, local_dir: Path) -> dict[str, Any]:
    local_dir.mkdir(mode=0o700, parents=True)
    source_started = time.perf_counter_ns()
    try:
        reads = read_source_panel(items, concurrency, MAX_CELL_PAYLOAD_BYTES)
    except BaseException as exc:
        raise ProbeError(error_class(exc, "shared_source")) from None
    source_wall = time.perf_counter_ns() - source_started
    stage_started = time.perf_counter_ns()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            latencies = list(
                executor.map(
                    lambda pair: exact_local_stage(pair[1], str(local_dir), pair[0]),
                    enumerate(reads),
                )
            )
    except BaseException as exc:
        raise ProbeError(error_class(exc, "shared_stage")) from None
    stage_wall = time.perf_counter_ns() - stage_started
    return {
        "mount_full_read": summarize_source(reads, source_wall),
        "local_atomic_stage_verify": summarize_local(reads, latencies, stage_wall),
    }


def _fragment_worker(
    worker_id: int,
    items: list[SourceItem],
    threads: int,
    local_dir: str,
    source_start: Any,
    stage_start: Any,
    messages: Any,
) -> None:
    try:
        Path(local_dir).mkdir(mode=0o700, parents=True)
        messages.put(("ready", worker_id, None))
        if not source_start.wait(120):
            raise ProbeError("source_start_timeout")
        source_started = time.perf_counter_ns()
        reads = read_source_panel(items, threads, MAX_CELL_PAYLOAD_BYTES // LOCAL_WORLD_SIZE)
        source_ended = time.perf_counter_ns()
        source_wall = source_ended - source_started
        source_summary = summarize_source(reads, source_wall)
        source_summary["_panel_end_ns"] = source_ended
        messages.put(("source_done", worker_id, source_summary))
        if not stage_start.wait(120):
            raise ProbeError("stage_start_timeout")
        stage_started = time.perf_counter_ns()
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            latencies = list(
                executor.map(
                    lambda pair: exact_local_stage(pair[1], local_dir, pair[0]),
                    enumerate(reads),
                )
            )
        stage_ended = time.perf_counter_ns()
        stage_wall = stage_ended - stage_started
        local_summary = summarize_local(reads, latencies, stage_wall)
        local_summary["_panel_end_ns"] = stage_ended
        messages.put(("stage_done", worker_id, local_summary))
    except BaseException as exc:
        messages.put(("failure", worker_id, error_class(exc, "fragment_worker")))
        raise SystemExit(3) from None


def collect_messages(
    messages: Any,
    expected_kind: str,
    expected_workers: set[int],
    timeout_s: float,
) -> dict[int, dict[str, Any] | None]:
    deadline = time.monotonic() + timeout_s
    values: dict[int, dict[str, Any] | None] = {}
    while set(values) != expected_workers:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProbeError(f"{expected_kind}_timeout")
        try:
            kind, worker_id, payload = messages.get(timeout=min(5.0, remaining))
        except queue.Empty:
            continue
        if kind == "failure":
            raise ProbeError(str(payload) if isinstance(payload, str) else "fragment_failure")
        if kind != expected_kind or worker_id not in expected_workers or worker_id in values:
            raise ProbeError("fragment_message_contract_drift")
        values[worker_id] = payload
    return values


def terminate_processes(processes: Sequence[Any]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=15)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def aggregate_fragment_phase(
    per_worker: Mapping[int, Mapping[str, Any]], wall_ns: int, phase: str
) -> dict[str, Any]:
    if wall_ns <= 0 or set(per_worker) != set(range(LOCAL_WORLD_SIZE)):
        raise ProbeError("fragment_phase_closure_drift")
    files = sum(int(value["files"]) for value in per_worker.values())
    bytes_count = sum(int(value["bytes"]) for value in per_worker.values())
    seconds = wall_ns / 1_000_000_000.0
    result: dict[str, Any] = {
        "files": files,
        "bytes": bytes_count,
        "wall_seconds": round(seconds, 6),
        "files_per_second": files / seconds,
        "mib_per_second": bytes_count / (1024 * 1024) / seconds,
        "per_worker_files": [int(per_worker[i]["files"]) for i in range(LOCAL_WORLD_SIZE)],
        "per_worker_wall_seconds": [
            float(per_worker[i]["wall_seconds"]) for i in range(LOCAL_WORLD_SIZE)
        ],
    }
    if phase == "source":
        result.update(
            {
                "mean_bytes": bytes_count / files,
                "effective_concurrency": sum(
                    float(value["effective_concurrency"]) * float(value["wall_seconds"])
                    for value in per_worker.values()
                )
                / seconds,
                "exact_read_ledger_sha256": canonical_sha256(
                    [per_worker[i]["exact_read_ledger_sha256"] for i in range(LOCAL_WORLD_SIZE)]
                ),
                "per_worker_latency": [per_worker[i]["latency"] for i in range(LOCAL_WORLD_SIZE)],
            }
        )
    else:
        result.update(
            {
                "verified_files": sum(
                    int(value["verified_files"]) for value in per_worker.values()
                ),
                "contract": "atomic_replace_file_fsync_dir_fsync_exact_reread",
                "per_worker_latency": [per_worker[i]["latency"] for i in range(LOCAL_WORLD_SIZE)],
            }
        )
    return result


def run_fragmented_cell(items: Sequence[SourceItem], threads: int, local_dir: Path) -> dict[str, Any]:
    if len(items) < LOCAL_WORLD_SIZE or threads != 2:
        raise ProbeError("fragment_topology_drift")
    local_dir.mkdir(mode=0o700, parents=True)
    context = multiprocessing.get_context("spawn")
    source_start = context.Event()
    stage_start = context.Event()
    messages = context.Queue()
    slices = [list(items[worker::LOCAL_WORLD_SIZE]) for worker in range(LOCAL_WORLD_SIZE)]
    processes = [
        context.Process(
            target=_fragment_worker,
            args=(
                worker,
                slices[worker],
                threads,
                str(local_dir / f"worker-{worker:02d}"),
                source_start,
                stage_start,
                messages,
            ),
        )
        for worker in range(LOCAL_WORLD_SIZE)
    ]
    try:
        for process in processes:
            process.start()
        expected = set(range(LOCAL_WORLD_SIZE))
        collect_messages(messages, "ready", expected, 120)
        source_started = time.perf_counter_ns()
        source_start.set()
        source_values = collect_messages(messages, "source_done", expected, 1800)
        source_wall = max(int(value["_panel_end_ns"]) for value in source_values.values()) - source_started
        if source_wall <= 0:
            raise ProbeError("fragment_source_clock_drift")
        stage_started = time.perf_counter_ns()
        stage_start.set()
        stage_values = collect_messages(messages, "stage_done", expected, 1800)
        stage_wall = max(int(value["_panel_end_ns"]) for value in stage_values.values()) - stage_started
        if stage_wall <= 0:
            raise ProbeError("fragment_stage_clock_drift")
        for process in processes:
            process.join(timeout=30)
        if any(process.is_alive() or process.exitcode != 0 for process in processes):
            raise ProbeError("fragment_process_exit_drift")
    except BaseException as exc:
        terminate_processes(processes)
        raise ProbeError(error_class(exc, "fragment_parent")) from None
    finally:
        messages.close()
        messages.join_thread()
    return {
        "mount_full_read": aggregate_fragment_phase(source_values, source_wall, "source"),
        "local_atomic_stage_verify": aggregate_fragment_phase(stage_values, stage_wall, "stage"),
    }


def distribution_balance(cell_summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    byte_totals = {
        name: int(value["mount_full_read"]["bytes"]) for name, value in cell_summaries.items()
    }
    means = list(byte_totals.values())
    ratio = max(means) / min(means) if means and min(means) > 0 else math.inf
    return {
        "exact_byte_totals_by_cell": byte_totals,
        "max_to_min_ratio": ratio,
        "threshold_max_to_min": 1.10,
        "pass": ratio <= 1.10,
        "pre_io_strata": ["format", "pixel_area_log2_proxy", "source_shard"],
        "note": "exact source byte size is intentionally learned only by first-touch full read",
    }


def final_decision(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    speeds = {
        name: float(value["mount_full_read"]["files_per_second"])
        for name, value in results.items()
    }
    current_fragmented_pass = min(speeds[name] for name in ("b16_x", "b16_y")) >= CAPACITY_GATE_IMG_S
    shared16_pass = min(speeds[name] for name in ("a16_x", "a16_y")) >= CAPACITY_GATE_IMG_S
    curve_cells = ("a08_mid", "a24_mid", "a32_mid")
    curve_best_cell = max(curve_cells, key=lambda name: speeds[name])
    curve_best_pass = speeds[curve_best_cell] >= CAPACITY_GATE_IMG_S
    a16_geomean = math.sqrt(speeds["a16_x"] * speeds["a16_y"])
    b16_geomean = math.sqrt(speeds["b16_x"] * speeds["b16_y"])
    balance = distribution_balance(results)
    if not balance["pass"]:
        status = "INCONCLUSIVE_UNBALANCED_BYTES"
        recommendation = "repeat_with_rebalanced_formal_unseen_panels"
    elif current_fragmented_pass:
        status = "COMPLETE_CURRENT_FRAGMENTED_PASS"
        recommendation = "current_fragmented_shape_meets_source_only_capacity_gate"
    elif shared16_pass:
        status = "COMPLETE_WITH_SHARED16_CANDIDATE"
        recommendation = "validate_shared_node_dispatcher_in_loader_without_promotion_from_this_probe"
    elif curve_best_pass:
        status = "COMPLETE_WITH_SHARED_CURVE_CANDIDATE"
        recommendation = "validate_best_shared_pool_curve_point_in_loader_without_promotion_from_this_probe"
    else:
        status = "COMPLETE_NO_CAPACITY_CANDIDATE"
        recommendation = "neither_current_nor_shared_pool_source_shape_meets_capacity_gate"
    return {
        "status": status,
        "recommendation": recommendation,
        "diagnostic_only": True,
        "production_promotion_authorized": False,
        "primary_gate_scope": "mount_full_read_source_only",
        "local_stage_is_gate": False,
        "local_stage_note": (
            "per-image file fsync plus directory fsync plus exact reread is deliberately more "
            "conservative than a production rolling cache and is diagnostic only"
        ),
        "formal_demand_img_s_per_node": FORMAL_DEMAND_IMG_S,
        "margin_multiplier": CAPACITY_MARGIN,
        "cold_first_touch_gate_img_s_per_node": CAPACITY_GATE_IMG_S,
        "a16_crossover_geomean_img_s": a16_geomean,
        "b16_crossover_geomean_img_s": b16_geomean,
        "fragmented_to_shared_ratio": b16_geomean / a16_geomean,
        "current_fragmented_status": {
            "status": "PASS" if current_fragmented_pass else "FAIL",
            "cells": ["b16_x", "b16_y"],
            "both_positions_pass": current_fragmented_pass,
        },
        "shared16_candidate_status": {
            "status": "PASS" if shared16_pass else "FAIL",
            "cells": ["a16_x", "a16_y"],
            "both_positions_pass": shared16_pass,
        },
        "curve_best_status": {
            "status": "PASS" if curve_best_pass else "FAIL",
            "cell": curve_best_cell,
            "files_per_second": speeds[curve_best_cell],
        },
        "per_cell_capacity_pass": {
            name: speed >= CAPACITY_GATE_IMG_S for name, speed in speeds.items()
        },
        "best_shared_cell": max(
            (name for name in speeds if name.startswith("a")), key=lambda name: speeds[name]
        ),
        "historical_t2_warm_loop_upper_bound_img_s": HISTORICAL_WARM_LOOP_IMG_S,
        "a32_mount_only_to_historical_warm_source_plus_write_ratio": (
            speeds["a32_mid"] / HISTORICAL_WARM_LOOP_IMG_S
        ),
        "historical_reference_is_gate": False,
        "historical_reference_comparison_is_apples_to_apples": False,
        "within_capture_source_objects_are_disjoint": True,
        "absolute_node_cache_coldness_claimed": False,
        "balance": balance,
    }


def assert_artifact_safe(raw: bytes, items: Sequence[SourceItem]) -> None:
    forbidden = (
        EXPECTED_SOURCE_ROOT.encode("ascii"),
        b"/images/train/",
        b"s3://",
        b'"uid"',
        b'"src_key"',
        b'"src_shard"',
        b'"image_path"',
    )
    if any(token in raw for token in forbidden):
        raise ProbeError("artifact_source_path_leak")
    raw_keys = {item.src_key.encode("ascii") for item in items}
    for token in HEX64_RE.findall(raw):
        if token in raw_keys:
            raise ProbeError("artifact_source_key_leak")


def verify_frozen_attestations(attestations: Mapping[str, Mapping[str, Any]]) -> None:
    for value in attestations.values():
        path = Path(str(value["_path"]))
        try:
            observed = artifact_attestation(path)
        except (OSError, ProbeError) as exc:
            raise ProbeError(error_class(exc, "attestation_verify")) from None
        if observed != {"bytes": value["bytes"], "sha256": value["sha256"]}:
            raise ProbeError("attestation_drift")


def capture(args: argparse.Namespace) -> Path:
    if args.run_id != RUN_ID:
        raise ProbeError("run_id_drift")
    if SAFE_TOKEN_RE.fullmatch(args.jobtag) is None:
        raise ProbeError("jobtag_invalid")
    cache_path = Path(args.cache).absolute()
    header_path = Path(args.catalog_header).absolute()
    catalog_path = Path(args.catalog_gpic).absolute()
    runner_path = Path(args.runner).absolute()
    yaml_path = Path(args.capture_yaml).absolute()
    require_regular_input(runner_path, args.runner_bytes, args.runner_sha256)
    require_regular_input(yaml_path, args.capture_yaml_bytes, args.capture_yaml_sha256)
    items = load_cache(cache_path)
    catalog_attestation = validate_catalog_header(header_path, catalog_path)
    reconciliation = build_formal_unseen_sample(items, catalog_path)
    items = reconciliation["items"]
    cells = build_balanced_cells(items)

    output_parent = Path(args.output_parent).absolute()
    run_root = output_parent / RUN_ID
    job_root = run_root / args.jobtag
    attempt_nonce = secrets.token_hex(32)
    attempt_root = job_root / attempt_nonce
    try:
        attempt_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        raise ProbeError(error_class(exc, "output_create")) from None
    local_root = Path(args.local_root).absolute()
    try:
        local_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        raise ProbeError(error_class(exc, "local_create")) from None

    frozen = {
        "cache": {
            "_path": str(cache_path),
            "bytes": SOURCE_CACHE_BYTES,
            "sha256": SOURCE_CACHE_SHA256,
        },
        "catalog_header": {
            "_path": str(header_path),
            "bytes": CATALOG_HEADER_BYTES,
            "sha256": CATALOG_HEADER_SHA256,
        },
        "catalog_gpic": {
            "_path": str(catalog_path),
            "bytes": CATALOG_GPIC_BYTES,
            "sha256": CATALOG_GPIC_SHA256,
        },
        "runner": {
            "_path": str(runner_path),
            "bytes": args.runner_bytes,
            "sha256": args.runner_sha256,
        },
        "capture_yaml": {
            "_path": str(yaml_path),
            "bytes": args.capture_yaml_bytes,
            "sha256": args.capture_yaml_sha256,
        },
    }
    public_frozen = {
        name: {"bytes": value["bytes"], "sha256": value["sha256"]}
        for name, value in frozen.items()
    }
    selection_summaries = {name: cell_selection_summary(value) for name, value in cells.items()}
    contract = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY",
        "run_id": RUN_ID,
        "attempt_nonce": attempt_nonce,
        "input_attestations": public_frozen,
        "catalog_attestation": catalog_attestation,
        "catalog_cache_reconciliation": reconciliation["attestation"],
        "selection": {
            "source_cache_items": SOURCE_CACHE_ITEMS,
            "formal_unseen_sample_pool_items": FORMAL_SAMPLE_POOL_ITEMS,
            "formal_sample_frozen_cache_uid_overlap": 0,
            "selected_items": CAPTURE_SELECTED_ITEMS,
            "holdout_items": FORMAL_SAMPLE_POOL_ITEMS - CAPTURE_SELECTED_ITEMS,
            "physical_overlap_across_cells": 0,
            "cell_items": PER_CELL_ITEMS,
            "cell_summaries": selection_summaries,
            "selection_sha256": canonical_sha256(
                {name: summary["selection_sha256"] for name, summary in selection_summaries.items()}
            ),
        },
        "cell_order": [spec[0] for spec in CELL_SPECS],
        "architecture_contract": [
            {"cell": name, "shape": shape, "processes": procs, "threads_per_process": threads}
            for name, shape, procs, threads in CELL_SPECS
        ],
        "source_contract": "open_nofollow_fstat_full_read_eof_fstat_same_identity",
        "source_payload_memory_budget_bytes_per_cell": MAX_CELL_PAYLOAD_BYTES,
        "local_contract": "atomic_replace_file_fsync_dir_fsync_exact_reread",
        "source_timer_excludes_local_stage": True,
        "within_capture_first_touch": True,
        "known_historical_frozen_cache_overlap": 0,
        "sample_strategy": "sparse_every_formal_catalog_row_group_excluding_frozen_cache",
        "absolute_node_cache_coldness_claimed": False,
        "capacity_gate_img_s_per_node": CAPACITY_GATE_IMG_S,
        "historical_warm_reference_is_gate": False,
        "diagnostic_only": True,
        "production_promotion_authorized": False,
    }
    atomic_publish_json(attempt_root / "contract.json", contract)

    results: dict[str, dict[str, Any]] = {}
    for name, shape, processes, threads in CELL_SPECS:
        if processes == 1:
            result = run_shared_cell(cells[name], threads, local_root / name)
        else:
            result = run_fragmented_cell(cells[name], threads, local_root / name)
        if result["mount_full_read"]["files"] != PER_CELL_ITEMS:
            raise ProbeError("cell_source_count_drift")
        if result["local_atomic_stage_verify"]["verified_files"] != PER_CELL_ITEMS:
            raise ProbeError("cell_stage_count_drift")
        two_phase_seconds = (
            float(result["mount_full_read"]["wall_seconds"])
            + float(result["local_atomic_stage_verify"]["wall_seconds"])
        )
        result["serialized_two_phase_total"] = {
            "wall_seconds": round(two_phase_seconds, 6),
            "files_per_second": PER_CELL_ITEMS / two_phase_seconds,
            "note": "diagnostic source phase then stronger durable-local verification phase; not pipelined",
        }
        results[name] = {
            "shape": shape,
            "processes": processes,
            "threads_per_process": threads,
            "selection": selection_summaries[name],
            **result,
        }
        print(
            "GPIC_CONCURRENCY_SHAPE_CELL_PASS "
            f"cell={name} files={PER_CELL_ITEMS} "
            f"source_img_s={result['mount_full_read']['files_per_second']:.3f} "
            f"stage_img_s={result['local_atomic_stage_verify']['files_per_second']:.3f}",
            flush=True,
        )

    decision = final_decision(results)
    result_document = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "attempt_nonce": attempt_nonce,
        "results": results,
        "decision": decision,
    }
    atomic_publish_json(attempt_root / "result.json", result_document)
    verify_frozen_attestations(frozen)
    for path in (attempt_root / "contract.json", attempt_root / "result.json"):
        assert_artifact_safe(path.read_bytes(), items)
    manifest = {
        "schema_version": "offline_arb_gpic_mount_concurrency_shape_manifest_v1",
        "status": "SEALED",
        "attempt_nonce": attempt_nonce,
        "files": {
            "contract.json": artifact_attestation(attempt_root / "contract.json"),
            "result.json": artifact_attestation(attempt_root / "result.json"),
        },
    }
    atomic_publish_json(attempt_root / "manifest.json", manifest)
    assert_artifact_safe((attempt_root / "manifest.json").read_bytes(), items)
    complete = {
        "schema_version": "offline_arb_gpic_mount_concurrency_shape_complete_v1",
        "status": decision["status"],
        "attempt_nonce": attempt_nonce,
        "manifest": artifact_attestation(attempt_root / "manifest.json"),
    }
    atomic_publish_json(attempt_root / "complete.json", complete)
    assert_artifact_safe((attempt_root / "complete.json").read_bytes(), items)
    verify_frozen_attestations(frozen)
    try:
        for directory in sorted(local_root.rglob("*"), reverse=True):
            if directory.is_dir():
                directory.rmdir()
        local_root.rmdir()
    except OSError:
        raise ProbeError("local_cleanup_incomplete") from None
    return attempt_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("capture",))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--catalog-header", required=True)
    parser.add_argument("--catalog-gpic", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--runner-bytes", required=True, type=int)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--capture-yaml", required=True)
    parser.add_argument("--capture-yaml-bytes", required=True, type=int)
    parser.add_argument("--capture-yaml-sha256", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--jobtag", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = capture(args)
    except BaseException as exc:
        print(
            "GPIC_CONCURRENCY_SHAPE_FAIL "
            f"error_class={error_class(exc, 'capture')}",
            flush=True,
        )
        return 3
    print(
        "GPIC_CONCURRENCY_SHAPE_COMPLETE "
        f"jobtag={args.jobtag} attempt={root.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
