#!/usr/bin/env python3
"""Build the five-domain typed caption sidecar aligned to global catalog row_id.

The build is source-centric and externally partitioned.  Catalog requests are
hash-partitioned by ``(source parquet path, source row group)``; every requested
source row group is therefore decoded exactly once per domain build.  Each
partition is sorted by global ``row_id`` and the sorted runs are merged into
fixed 65,536-row parquet files with exact 256-row row groups.

Completed domains are recoverable independently under the content-addressed
work root.  The formal output root is created only after all five domains have
matching source/full-reread typed digests.  Publication hard-links the verified
domain files into a temporary sibling tree and atomically renames that complete
tree, containing only five domain directories, ``header.json``, and
``_SUCCESS``, into place.

Use ``--diagnostic-rows-per-domain`` for a bounded real-source exercise.  A
diagnostic output root must differ from the canonical formal root.  ``--self-test``
runs a tiny typed null/list/unicode parquet round trip and does not read the
production catalog.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Direct script execution places scripts/offline_arb on sys.path.  Import from
# that directory first, while retaining the package form for module execution.
try:
    from caption_sidecar_contract import (
        ArrowIpcSemanticDigester,
        BUILDER_BUILD_ID,
        CAPTION_COLUMNS,
        CAPTION_SCHEMA_CONTRACT_SHA256,
        CAPTION_SCHEMAS,
        DOMAINS,
        FILE_ROWS,
        ROW_GROUP_ROWS,
        ROW_ORDER,
        SCHEMA_VERSION,
        SEMANTIC_DIGEST_ALGORITHM,
        canonical_json_sha256,
        canonical_schema,
        header_self_hash,
        sidecar_schema,
    )
except ImportError:  # pragma: no cover - exercised by ``python -m``
    from scripts.offline_arb.caption_sidecar_contract import (
        ArrowIpcSemanticDigester,
        BUILDER_BUILD_ID,
        CAPTION_COLUMNS,
        CAPTION_SCHEMA_CONTRACT_SHA256,
        CAPTION_SCHEMAS,
        DOMAINS,
        FILE_ROWS,
        ROW_GROUP_ROWS,
        ROW_ORDER,
        SCHEMA_VERSION,
        SEMANTIC_DIGEST_ALGORITHM,
        canonical_json_sha256,
        canonical_schema,
        header_self_hash,
        sidecar_schema,
    )


BASE = Path("/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop")
DEFAULT_CATALOG_ROOT = BASE / "offline_arb_catalog_v1"
DEFAULT_OUTPUT_ROOT = BASE / "offline_arb_caption_sidecar_v1"
DEFAULT_WORK_ROOT = BASE / ".offline_arb_caption_sidecar_v1.work"
CATALOG_SCHEMA_VERSION = "offline_arb_catalog_v1"
REQUEST_PARTITIONS = 256
REQUEST_ROW_GROUP_ROWS = 8_192
VALUE_ROW_GROUP_ROWS = 8_192
MERGE_BATCH_ROWS = 4_096
MIB = 1 << 20
SHA256_RE = re.compile(r"[0-9a-f]{64}")

REQUEST_SCHEMA = pa.schema(
    [
        pa.field("row_id", pa.int64(), nullable=False),
        pa.field("caption_parquet_path", pa.string(), nullable=False),
        pa.field("caption_row_group", pa.int32(), nullable=False),
        pa.field("caption_row_in_group", pa.int32(), nullable=False),
    ]
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def regular_file(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"artifact is not a regular file: {path}")
    return metadata


def regular_dir(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"artifact is not a real directory: {path}")
    return metadata


def stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    try:
        with temporary.open("x", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def read_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    regular_file(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    before = regular_file(path)
    digest = hashlib.sha256()
    bytes_read = 0
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 * MIB), b""):
            digest.update(block)
            bytes_read += len(block)
    after = regular_file(path)
    if stat_identity(before) != stat_identity(after):
        raise RuntimeError(f"file changed while hashing: {path}")
    if bytes_read != after.st_size:
        raise RuntimeError(
            f"short file hash read: {path}: read={bytes_read}, size={after.st_size}"
        )
    return digest.hexdigest()


@dataclass(frozen=True)
class CatalogDomain:
    name: str
    path: Path
    file_name: str
    row_base: int
    row_count: int
    size_bytes: int
    sha256: str
    schema: str


@dataclass(frozen=True)
class CatalogSnapshot:
    root: Path
    header_path: Path
    header: dict[str, Any]
    header_sha256: str
    catalog_hash: str
    domains: dict[str, CatalogDomain]


@dataclass(frozen=True)
class RequestPartition:
    index: int
    path: Path
    rows: int


@dataclass(frozen=True)
class ValuePartition:
    index: int
    path: Path
    rows: int
    source_identities: dict[str, tuple[int, int, int, int, int]]


def load_catalog_snapshot(root: Path) -> CatalogSnapshot:
    root = root.expanduser().resolve(strict=True)
    regular_dir(root)
    header_path = root / "header.json"
    header, header_sha256 = read_json_snapshot(header_path)
    if header.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported catalog schema: {header.get('schema_version')!r}"
        )
    if header.get("domain_order") != list(DOMAINS):
        raise ValueError(
            f"catalog domain_order drift: {header.get('domain_order')!r}"
        )
    catalog_hash = header.get("catalog_hash")
    if not is_sha256(catalog_hash):
        raise ValueError(f"catalog_hash is malformed: {catalog_hash!r}")
    raw_files = header.get("files")
    if not isinstance(raw_files, dict) or set(raw_files) != set(DOMAINS):
        raise ValueError("catalog files mapping does not contain exactly five domains")

    cursor = 0
    domains: dict[str, CatalogDomain] = {}
    for domain in DOMAINS:
        entry = raw_files[domain]
        if not isinstance(entry, dict):
            raise ValueError(f"catalog files.{domain} must be an object")
        name = entry.get("name")
        if (
            not isinstance(name, str)
            or not name.endswith(".parquet")
            or Path(name).name != name
        ):
            raise ValueError(f"catalog files.{domain}.name is malformed: {name!r}")
        rows = strict_int(entry.get("rows"), f"catalog files.{domain}.rows", minimum=1)
        size_bytes = strict_int(
            entry.get("bytes"), f"catalog files.{domain}.bytes", minimum=1
        )
        checksum = entry.get("sha256")
        if not is_sha256(checksum):
            raise ValueError(f"catalog files.{domain}.sha256 is malformed")
        schema = entry.get("schema")
        if not isinstance(schema, str) or not schema:
            raise ValueError(f"catalog files.{domain}.schema is missing")
        path = root / name
        metadata = regular_file(path)
        if metadata.st_size != size_bytes:
            raise ValueError(
                f"catalog file size drift {path}: {metadata.st_size} != {size_bytes}"
            )
        domains[domain] = CatalogDomain(
            name=domain,
            path=path,
            file_name=name,
            row_base=cursor,
            row_count=rows,
            size_bytes=size_bytes,
            sha256=checksum,
            schema=schema,
        )
        cursor += rows
    if header.get("row_count") != cursor:
        raise ValueError(f"catalog row_count drift: {header.get('row_count')!r} != {cursor}")
    return CatalogSnapshot(
        root=root,
        header_path=header_path,
        header=header,
        header_sha256=header_sha256,
        catalog_hash=catalog_hash,
        domains=domains,
    )


def verify_catalog_file(
    spec: CatalogDomain, *, hash_bytes: bool = True
) -> tuple[int, int, int, int, int]:
    before = regular_file(spec.path)
    if before.st_size != spec.size_bytes:
        raise ValueError(f"catalog file size drift: {spec.path}")
    if hash_bytes:
        checksum = sha256_file(spec.path)
        if checksum != spec.sha256:
            raise ValueError(
                f"catalog file SHA-256 drift {spec.path}: {checksum} != {spec.sha256}"
            )
    parquet = pq.ParquetFile(spec.path)
    if parquet.metadata.num_rows != spec.row_count:
        raise ValueError(
            f"catalog footer rows drift {spec.path}: "
            f"{parquet.metadata.num_rows} != {spec.row_count}"
        )
    required = REQUEST_SCHEMA.names
    if any(name not in parquet.schema_arrow.names for name in required):
        raise ValueError(f"catalog locator schema is incomplete: {spec.path}")
    selected = pa.schema([parquet.schema_arrow.field(name) for name in required])
    if not canonical_schema(selected).equals(REQUEST_SCHEMA, check_metadata=False):
        raise ValueError(
            f"catalog locator typed schema drift {spec.path}:\n{selected}\n!=\n{REQUEST_SCHEMA}"
        )
    after = regular_file(spec.path)
    if stat_identity(before) != stat_identity(after):
        raise RuntimeError(f"catalog changed during identity validation: {spec.path}")
    return stat_identity(after)


def request_partition_index(path: str, row_group: int, partitions: int) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"cap-side-req-v1")
    digest.update(path.encode("utf-8"))
    digest.update(b"\0")
    digest.update(struct.pack(">q", row_group))
    return int.from_bytes(digest.digest(), "big") % partitions


def partition_catalog_requests(
    spec: CatalogDomain,
    *,
    effective_rows: int,
    partitions: int,
    request_dir: Path,
    catalog_identity: tuple[int, int, int, int, int],
) -> list[RequestPartition]:
    request_dir.mkdir(parents=True, exist_ok=False)
    parquet = pq.ParquetFile(spec.path)
    writers: dict[int, pq.ParquetWriter] = {}
    paths: dict[int, Path] = {}
    counts: dict[int, int] = {}
    partition_cache: dict[tuple[str, int], int] = {}
    next_row_id = spec.row_base
    remaining = effective_rows
    try:
        batches = parquet.iter_batches(
            batch_size=65_536,
            columns=REQUEST_SCHEMA.names,
            use_threads=False,
        )
        for raw_batch in batches:
            if remaining <= 0:
                break
            batch = raw_batch.slice(0, min(raw_batch.num_rows, remaining))
            table = pa.Table.from_batches([batch]).cast(REQUEST_SCHEMA, safe=True)
            row_ids = table.column("row_id").combine_chunks().to_numpy(
                zero_copy_only=False
            )
            expected = np.arange(
                next_row_id, next_row_id + table.num_rows, dtype=np.int64
            )
            if not np.array_equal(row_ids, expected):
                raise ValueError(
                    f"catalog global row_id gap/reorder in {spec.name} at {next_row_id}"
                )
            source_paths = table.column("caption_parquet_path").to_pylist()
            source_groups = table.column("caption_row_group").to_pylist()
            source_rows = table.column("caption_row_in_group").to_pylist()
            by_partition: dict[int, list[int]] = {}
            for offset, (source_path, source_group, source_row) in enumerate(
                zip(source_paths, source_groups, source_rows)
            ):
                if (
                    not isinstance(source_path, str)
                    or not source_path
                    or type(source_group) is not int
                    or source_group < 0
                    or type(source_row) is not int
                    or source_row < 0
                ):
                    raise ValueError(
                        f"malformed catalog caption locator row_id={next_row_id + offset}"
                    )
                key = (source_path, source_group)
                part = partition_cache.get(key)
                if part is None:
                    part = request_partition_index(source_path, source_group, partitions)
                    partition_cache[key] = part
                by_partition.setdefault(part, []).append(offset)
            for part, offsets in by_partition.items():
                selected = table.take(pa.array(offsets, type=pa.int64()))
                writer = writers.get(part)
                if writer is None:
                    path = request_dir / f"part-{part:04d}.parquet"
                    writer = pq.ParquetWriter(
                        path,
                        REQUEST_SCHEMA,
                        compression="zstd",
                        use_dictionary=True,
                        write_statistics=True,
                    )
                    writers[part] = writer
                    paths[part] = path
                    counts[part] = 0
                writer.write_table(selected, row_group_size=REQUEST_ROW_GROUP_ROWS)
                counts[part] += selected.num_rows
            next_row_id += table.num_rows
            remaining -= table.num_rows
            if (next_row_id - spec.row_base) % 1_048_576 < table.num_rows:
                print(
                    f"[sidecar:{spec.name}] catalog requests "
                    f"{next_row_id - spec.row_base}/{effective_rows}",
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()

    if remaining != 0 or next_row_id != spec.row_base + effective_rows:
        raise ValueError(
            f"catalog request scan short for {spec.name}: "
            f"rows={next_row_id-spec.row_base}/{effective_rows}"
        )
    if stat_identity(regular_file(spec.path)) != catalog_identity:
        raise RuntimeError(f"catalog changed while partitioning requests: {spec.path}")
    products = [
        RequestPartition(index=part, path=paths[part], rows=counts[part])
        for part in sorted(paths)
    ]
    if sum(item.rows for item in products) != effective_rows:
        raise AssertionError(f"request partition count drift for {spec.name}")
    return products


def validate_source_schema(
    path: Path,
    parquet: pq.ParquetFile,
    expected: pa.Schema,
) -> None:
    incompatible: list[str] = []
    for expected_field in expected:
        try:
            live = parquet.schema_arrow.field(expected_field.name)
        except KeyError as exc:
            raise ValueError(
                f"source caption column missing in {path}: {expected_field.name}"
            ) from exc
        if live.type == expected_field.type:
            continue
        if expected_field.nullable and pa.types.is_null(live.type):
            continue
        incompatible.append(
            f"{expected_field.name}: live={live.type}, expected={expected_field.type}"
        )
    if incompatible:
        raise ValueError(
            f"source typed schema drift in {path}: " + "; ".join(incompatible)
        )


def build_value_partition(
    request: RequestPartition,
    *,
    source_schema: pa.Schema,
    output_schema: pa.Schema,
    value_dir: Path,
) -> ValuePartition:
    request_table = pq.read_table(
        request.path,
        columns=REQUEST_SCHEMA.names,
        use_threads=False,
    ).cast(REQUEST_SCHEMA, safe=True)
    if request_table.num_rows != request.rows:
        raise ValueError(f"request partition row count drift: {request.path}")
    row_ids = request_table.column("row_id").to_pylist()
    source_paths = request_table.column("caption_parquet_path").to_pylist()
    source_groups = request_table.column("caption_row_group").to_pylist()
    source_rows = request_table.column("caption_row_in_group").to_pylist()
    grouped: dict[tuple[str, int], list[int]] = {}
    for index, (path, row_group) in enumerate(zip(source_paths, source_groups)):
        grouped.setdefault((str(path), int(row_group)), []).append(index)

    pieces: list[pa.Table] = []
    current_path: Path | None = None
    current_parquet: pq.ParquetFile | None = None
    current_identity: tuple[int, int, int, int, int] | None = None
    source_identities: dict[str, tuple[int, int, int, int, int]] = {}

    def close_current() -> None:
        if current_path is not None and current_identity is not None:
            if stat_identity(regular_file(current_path)) != current_identity:
                raise RuntimeError(f"caption source changed while reading: {current_path}")

    for (path_text, row_group), indices in sorted(grouped.items()):
        path = Path(path_text)
        if current_path != path:
            close_current()
            metadata = regular_file(path)
            current_identity = stat_identity(metadata)
            current_path = path
            current_parquet = pq.ParquetFile(path)
            validate_source_schema(path, current_parquet, source_schema)
            previous_identity = source_identities.setdefault(
                str(path), current_identity
            )
            if previous_identity != current_identity:
                raise RuntimeError(
                    f"caption source identity changed within partition: {path}"
                )
        assert current_parquet is not None
        if row_group < 0 or row_group >= current_parquet.num_row_groups:
            raise ValueError(f"source row group out of range: {path}:{row_group}")
        source = current_parquet.read_row_group(
            row_group,
            columns=source_schema.names,
            use_threads=False,
        )
        try:
            # A parquet shard whose values are all null may decode a physical
            # Arrow ``null`` column.  Promote it explicitly to the frozen
            # nullable logical schema before any take/merge/digest operation.
            source = source.cast(source_schema, safe=True)
        except Exception as exc:
            raise ValueError(
                f"cannot safely cast source caption schema {path}:{row_group} "
                f"to canonical {source_schema}: {exc}"
            ) from exc
        within = [int(source_rows[index]) for index in indices]
        if any(value < 0 or value >= source.num_rows for value in within):
            raise ValueError(f"source row offset out of range: {path}:{row_group}")
        selected = source.take(pa.array(within, type=pa.int64()))
        arrays: list[pa.Array | pa.ChunkedArray] = [
            pa.array([int(row_ids[index]) for index in indices], type=pa.int64())
        ]
        arrays.extend(selected.column(name) for name in source_schema.names)
        pieces.append(pa.Table.from_arrays(arrays, schema=output_schema))
    close_current()
    if not pieces:
        raise ValueError(f"empty request partition: {request.path}")
    table = pa.concat_tables(pieces)
    order = pc.sort_indices(table, sort_keys=[("row_id", "ascending")])
    table = table.take(order)
    sorted_ids = table.column("row_id").combine_chunks().to_numpy(zero_copy_only=False)
    if table.num_rows != request.rows or (
        table.num_rows > 1 and np.any(sorted_ids[1:] <= sorted_ids[:-1])
    ):
        raise ValueError(f"value partition row_id duplicate/count drift: {request.path}")
    value_dir.mkdir(parents=True, exist_ok=True)
    path = value_dir / f"part-{request.index:04d}.parquet"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
            row_group_size=VALUE_ROW_GROUP_ROWS,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return ValuePartition(
        index=request.index,
        path=path,
        rows=table.num_rows,
        source_identities=source_identities,
    )


def freeze_source_snapshot(
    identities: dict[str, tuple[int, int, int, int, int]],
) -> dict[str, Any]:
    if not identities:
        raise ValueError("source snapshot is empty")
    validate_inprocess_source_identities(identities)
    entries: list[dict[str, Any]] = []
    for path_text in sorted(identities):
        expected = identities[path_text]
        entries.append(
            {
                "path": path_text,
                "bytes": expected[2],
                "mtime_ns": expected[3],
            }
        )
    return {
        "file_count": len(entries),
        "identity_sha256": canonical_json_sha256(entries),
        "files": entries,
    }


def validate_inprocess_source_identities(
    identities: dict[str, tuple[int, int, int, int, int]],
) -> None:
    if not identities:
        raise ValueError("in-process source identity set is empty")
    for path_text, expected in identities.items():
        live = stat_identity(regular_file(Path(path_text)))
        if live != expected:
            raise RuntimeError(
                f"caption source changed in process: {path_text}: "
                f"live={live}, expected={expected}"
            )


def validate_source_snapshot(raw: Any, *, domain: str) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"completed source snapshot missing: {domain}")
    files = raw.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"completed source snapshot file list malformed: {domain}")
    if raw.get("file_count") != len(files):
        raise ValueError(f"completed source snapshot count drift: {domain}")
    if raw.get("identity_sha256") != canonical_json_sha256(files):
        raise ValueError(f"completed source snapshot digest drift: {domain}")
    previous_path: str | None = None
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError(f"completed source snapshot entry malformed: {domain}")
        path_text = entry.get("path")
        if (
            not isinstance(path_text, str)
            or not path_text
            or (previous_path is not None and path_text <= previous_path)
        ):
            raise ValueError(f"completed source snapshot path order drift: {domain}")
        if set(entry) != {"path", "bytes", "mtime_ns"}:
            raise ValueError(
                f"completed source snapshot contains non-portable fields: {domain}"
            )
        expected = (
            strict_int(entry.get("bytes"), f"{domain}.source.bytes", minimum=1),
            strict_int(entry.get("mtime_ns"), f"{domain}.source.mtime_ns"),
        )
        metadata = regular_file(Path(path_text))
        live = (int(metadata.st_size), int(metadata.st_mtime_ns))
        if live != expected:
            raise RuntimeError(
                f"completed caption source snapshot changed for {domain}: "
                f"{path_text}: live={live}, expected={expected}"
            )
        previous_path = path_text


class SortedPartitionCursor:
    def __init__(self, item: ValuePartition, schema: pa.Schema) -> None:
        self.item = item
        self.schema = schema
        self.parquet = pq.ParquetFile(item.path)
        if not canonical_schema(self.parquet.schema_arrow).equals(
            schema, check_metadata=False
        ):
            raise ValueError(f"value partition schema drift: {item.path}")
        if self.parquet.metadata.num_rows != item.rows:
            raise ValueError(f"value partition footer row drift: {item.path}")
        self.iterator = self.parquet.iter_batches(
            batch_size=MERGE_BATCH_ROWS,
            columns=schema.names,
            use_threads=False,
        )
        self.current: pa.RecordBatch | None = None
        self.last_seen: int | None = None
        self.rows_seen = 0

    def _ensure(self) -> bool:
        if self.current is not None:
            return True
        try:
            batch = next(self.iterator)
        except StopIteration:
            return False
        normalized = canonical_schema(batch.schema)
        if not normalized.equals(self.schema, check_metadata=False):
            raise ValueError(f"value partition decoded schema drift: {self.item.path}")
        ids = batch.column(0).to_numpy(zero_copy_only=False)
        if len(ids) == 0 or (len(ids) > 1 and np.any(ids[1:] <= ids[:-1])):
            raise ValueError(f"value partition is not strictly sorted: {self.item.path}")
        if self.last_seen is not None and int(ids[0]) <= self.last_seen:
            raise ValueError(f"value partition cross-batch order drift: {self.item.path}")
        self.last_seen = int(ids[-1])
        self.current = batch
        return True

    def take_before(self, row_stop: int) -> list[pa.Table]:
        pieces: list[pa.Table] = []
        while self._ensure():
            assert self.current is not None
            ids = self.current.column(0).to_numpy(zero_copy_only=False)
            cut = int(np.searchsorted(ids, row_stop, side="left"))
            if cut:
                piece = self.current.slice(0, cut)
                pieces.append(pa.Table.from_batches([piece], schema=self.schema))
                self.rows_seen += cut
                self.current = (
                    self.current.slice(cut) if cut < self.current.num_rows else None
                )
            if cut < len(ids):
                break
        return pieces

    def assert_exhausted(self) -> None:
        if self._ensure():
            raise ValueError(f"unconsumed value rows remain: {self.item.path}")
        if self.rows_seen != self.item.rows:
            raise ValueError(
                f"value partition consumed rows {self.rows_seen} != {self.item.rows}: "
                f"{self.item.path}"
            )


def parquet_file_entry(
    path: Path,
    *,
    domain: str,
    index: int,
    row_start: int,
    row_stop: int,
    schema: pa.Schema,
) -> dict[str, Any]:
    metadata = regular_file(path)
    parquet = pq.ParquetFile(path)
    rows = row_stop - row_start
    expected_row_groups = math.ceil(rows / ROW_GROUP_ROWS)
    if parquet.metadata.num_rows != rows or parquet.num_row_groups != expected_row_groups:
        raise ValueError(f"sidecar footer layout drift: {path}")
    live_schema = canonical_schema(parquet.schema_arrow)
    if not live_schema.equals(schema, check_metadata=False):
        raise ValueError(f"sidecar parquet schema drift: {path}")
    for row_group in range(parquet.num_row_groups):
        expected_rows = min(
            ROW_GROUP_ROWS, row_stop - (row_start + row_group * ROW_GROUP_ROWS)
        )
        if parquet.metadata.row_group(row_group).num_rows != expected_rows:
            raise ValueError(f"sidecar row-group size drift: {path}:{row_group}")
    return {
        "name": f"{domain}/part-{index:05d}.parquet",
        "row_start": row_start,
        "row_stop": row_stop,
        "rows": rows,
        "bytes": int(metadata.st_size),
        "sha256": sha256_file(path),
        "row_groups": expected_row_groups,
        "schema": str(parquet.schema_arrow),
    }


def merge_value_partitions(
    domain: str,
    *,
    row_base: int,
    row_count: int,
    values: Sequence[ValuePartition],
    schema: pa.Schema,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    if sum(item.rows for item in values) != row_count:
        raise ValueError(f"value partition total rows drift for {domain}")
    cursors = [SortedPartitionCursor(item, schema) for item in values]
    digester = ArrowIpcSemanticDigester(
        domain=domain,
        row_base=row_base,
        row_count=row_count,
        schema=schema,
    )
    files: list[dict[str, Any]] = []
    file_count = math.ceil(row_count / FILE_ROWS)
    for index in range(file_count):
        row_start = row_base + index * FILE_ROWS
        row_stop = min(row_base + row_count, row_start + FILE_ROWS)
        pieces: list[pa.Table] = []
        for cursor in cursors:
            pieces.extend(cursor.take_before(row_stop))
        if not pieces:
            raise ValueError(f"no merged values for {domain} [{row_start},{row_stop})")
        table = pa.concat_tables(pieces)
        order = pc.sort_indices(table, sort_keys=[("row_id", "ascending")])
        table = table.take(order)
        expected_rows = row_stop - row_start
        row_ids = table.column("row_id").combine_chunks().to_pylist()
        if table.num_rows != expected_rows or row_ids != list(range(row_start, row_stop)):
            raise ValueError(
                f"global row_id coverage drift for {domain} [{row_start},{row_stop})"
            )
        for offset in range(0, table.num_rows, ROW_GROUP_ROWS):
            digester.update(table.slice(offset, min(ROW_GROUP_ROWS, table.num_rows - offset)))

        path = output_dir / f"part-{index:05d}.parquet"
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
        try:
            pq.write_table(
                table,
                temporary,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
                row_group_size=ROW_GROUP_ROWS,
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        files.append(
            parquet_file_entry(
                path,
                domain=domain,
                index=index,
                row_start=row_start,
                row_stop=row_stop,
                schema=schema,
            )
        )
        if (index + 1) % 32 == 0 or index + 1 == file_count:
            print(
                f"[sidecar:{domain}] final files {index + 1}/{file_count}",
                flush=True,
            )
    for cursor in cursors:
        cursor.assert_exhausted()
    return files, digester.finish()


def full_reread_digest(
    domain: str,
    *,
    row_base: int,
    row_count: int,
    files: Sequence[dict[str, Any]],
    directory: Path,
    schema: pa.Schema,
) -> str:
    digester = ArrowIpcSemanticDigester(
        domain=domain,
        row_base=row_base,
        row_count=row_count,
        schema=schema,
    )
    cursor = row_base
    for index, entry in enumerate(files):
        path = directory / f"part-{index:05d}.parquet"
        before = regular_file(path)
        if before.st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"sidecar file identity drift before reread: {path}")
        parquet = pq.ParquetFile(path)
        if str(parquet.schema_arrow) != entry["schema"]:
            raise ValueError(f"sidecar header/footer schema drift: {path}")
        if not parquet.schema_arrow.equals(schema, check_metadata=True):
            raise ValueError(f"sidecar typed schema drift on reread: {path}")
        if parquet.metadata.num_rows != entry["rows"]:
            raise ValueError(f"sidecar footer row count drift on reread: {path}")
        if parquet.num_row_groups != entry["row_groups"]:
            raise ValueError(f"sidecar footer row-group drift on reread: {path}")
        for row_group in range(parquet.num_row_groups):
            expected_rows = min(
                ROW_GROUP_ROWS,
                int(entry["row_stop"]) - (int(entry["row_start"]) + row_group * ROW_GROUP_ROWS),
            )
            if parquet.metadata.row_group(row_group).num_rows != expected_rows:
                raise ValueError(f"sidecar row-group layout drift: {path}:{row_group}")
            table = parquet.read_row_group(
                row_group,
                columns=schema.names,
                use_threads=False,
            )
            if not table.schema.equals(schema, check_metadata=True):
                raise ValueError(f"sidecar decoded schema drift: {path}:{row_group}")
            ids = table.column("row_id").combine_chunks().to_pylist()
            if ids != list(range(cursor, cursor + expected_rows)):
                raise ValueError(f"sidecar decoded row_id drift: {path}:{row_group}")
            digester.update(table)
            cursor += expected_rows
        after = regular_file(path)
        if stat_identity(before) != stat_identity(after):
            raise RuntimeError(f"sidecar changed during full reread: {path}")
    if cursor != row_base + row_count:
        raise ValueError(f"sidecar full reread row count drift for {domain}")
    return digester.finish()


def workspace_for(
    work_root: Path,
    catalog_hash: str,
    diagnostic_rows: int | None,
) -> Path:
    mode = "formal" if diagnostic_rows is None else f"diag-{diagnostic_rows}"
    return (
        work_root
        / catalog_hash
        / BUILDER_BUILD_ID
        / CAPTION_SCHEMA_CONTRACT_SHA256
        / mode
    )


def domain_marker_hash(marker: dict[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in marker.items() if key != "marker_sha256"}
    )


def completed_domain(
    snapshot: CatalogSnapshot,
    domain: str,
    *,
    workspace: Path,
    effective_rows: int,
    hash_files: bool,
) -> dict[str, Any] | None:
    marker_path = workspace / "markers" / f"{domain}.json"
    directory = workspace / "complete" / domain
    if not marker_path.is_file() or not directory.is_dir():
        return None
    marker, _raw_sha = read_json_snapshot(marker_path)
    if marker.get("marker_sha256") != domain_marker_hash(marker):
        raise ValueError(f"completed domain marker self-hash drift: {marker_path}")
    spec = snapshot.domains[domain]
    expected = {
        "schema_version": SCHEMA_VERSION,
        "semantic_digest_algorithm": SEMANTIC_DIGEST_ALGORITHM,
        "builder_build_id": BUILDER_BUILD_ID,
        "caption_schema_contract_sha256": CAPTION_SCHEMA_CONTRACT_SHA256,
        "catalog_hash": snapshot.catalog_hash,
        "catalog_header_sha256": snapshot.header_sha256,
        "domain": domain,
        "row_base": spec.row_base,
        "row_count": effective_rows,
        "columns": list(CAPTION_COLUMNS[domain]),
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(
                f"completed marker contract drift {domain}.{key}: "
                f"{marker.get(key)!r} != {value!r}"
            )
    catalog_file = marker.get("catalog_file")
    expected_catalog_file = {
        "name": spec.file_name,
        "row_base": spec.row_base,
        "rows": spec.row_count,
        "bytes": spec.size_bytes,
        "sha256": spec.sha256,
        "schema": spec.schema,
    }
    if catalog_file != expected_catalog_file:
        raise ValueError(f"completed marker catalog binding drift: {domain}")
    source_digest = marker.get("source_semantics_sha256")
    decoded_digest = marker.get("decoded_sidecar_semantics_sha256")
    if not is_sha256(source_digest) or source_digest != decoded_digest:
        raise ValueError(f"completed marker semantic digest drift: {domain}")
    validate_source_snapshot(marker.get("source_snapshot"), domain=domain)
    files = marker.get("files")
    expected_count = math.ceil(effective_rows / FILE_ROWS)
    if not isinstance(files, list) or len(files) != expected_count:
        raise ValueError(f"completed marker file count drift: {domain}")
    expected_names = {f"part-{index:05d}.parquet" for index in range(expected_count)}
    live_entries = list(directory.iterdir())
    for path in live_entries:
        regular_file(path)
    live_names = {path.name for path in live_entries}
    if live_names != expected_names:
        raise ValueError(f"completed domain exact file set drift: {directory}")
    expected_schema = sidecar_schema(
        CAPTION_SCHEMAS[domain], CAPTION_COLUMNS[domain]
    )
    build = marker.get("build")
    if (
        not isinstance(build, dict)
        or build.get("source_schema") != str(CAPTION_SCHEMAS[domain])
        or build.get("sidecar_schema") != str(expected_schema)
    ):
        raise ValueError(f"completed marker canonical schema binding drift: {domain}")
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise ValueError(f"completed file entry malformed: {domain}/{index}")
        expected_name = f"{domain}/part-{index:05d}.parquet"
        if entry.get("name") != expected_name:
            raise ValueError(f"completed file name drift: {domain}/{index}")
        row_start = spec.row_base + index * FILE_ROWS
        row_stop = min(spec.row_base + effective_rows, row_start + FILE_ROWS)
        if (
            entry.get("row_start"),
            entry.get("row_stop"),
            entry.get("rows"),
        ) != (row_start, row_stop, row_stop - row_start):
            raise ValueError(f"completed file row range drift: {domain}/{index}")
        path = directory / f"part-{index:05d}.parquet"
        metadata = regular_file(path)
        if metadata.st_size != entry.get("bytes"):
            raise ValueError(f"completed file byte size drift: {path}")
        if hash_files and sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"completed file SHA-256 drift: {path}")
        parquet = pq.ParquetFile(path)
        expected_row_groups = math.ceil((row_stop - row_start) / ROW_GROUP_ROWS)
        if (
            parquet.metadata.num_rows != entry["rows"]
            or parquet.num_row_groups != expected_row_groups
            or entry.get("row_groups") != expected_row_groups
            or not parquet.schema_arrow.equals(expected_schema, check_metadata=True)
            or str(parquet.schema_arrow) != str(expected_schema)
            or entry.get("schema") != str(expected_schema)
        ):
            raise ValueError(f"completed parquet footer drift: {path}")
        physical_cursor = row_start
        for row_group in range(parquet.num_row_groups):
            expected_group_rows = min(
                ROW_GROUP_ROWS, row_stop - physical_cursor
            )
            if parquet.metadata.row_group(row_group).num_rows != expected_group_rows:
                raise ValueError(
                    f"completed parquet row-group size drift: {path}:{row_group}"
                )
            if hash_files:
                row_ids = parquet.read_row_group(
                    row_group, columns=["row_id"], use_threads=False
                ).column("row_id").to_pylist()
                if row_ids != list(
                    range(physical_cursor, physical_cursor + expected_group_rows)
                ):
                    raise ValueError(
                        f"completed parquet physical row_id drift: {path}:{row_group}"
                    )
            physical_cursor += expected_group_rows
        if physical_cursor != row_stop:
            raise ValueError(f"completed parquet physical row count drift: {path}")
    return marker


def publish_completed_domain(
    workspace: Path,
    domain: str,
    source_directory: Path,
    marker: dict[str, Any],
    source_identities: dict[str, tuple[int, int, int, int, int]],
) -> None:
    complete_root = workspace / "complete"
    marker_root = workspace / "markers"
    orphan_root = workspace / "orphans"
    complete_root.mkdir(parents=True, exist_ok=True)
    marker_root.mkdir(parents=True, exist_ok=True)
    orphan_root.mkdir(parents=True, exist_ok=True)
    destination = complete_root / domain
    if destination.exists():
        orphan = orphan_root / f"{domain}.{time.time_ns()}"
        os.rename(destination, orphan)
    os.rename(source_directory, destination)
    fsync_dir(complete_root)
    # This is the final source stat barrier immediately before the recoverable
    # marker becomes visible; it closes the gap after the staged directory move.
    validate_inprocess_source_identities(source_identities)
    validate_source_snapshot(marker.get("source_snapshot"), domain=domain)
    marker["marker_sha256"] = domain_marker_hash(marker)
    atomic_write_json(marker_root / f"{domain}.json", marker)


def build_domain(
    snapshot: CatalogSnapshot,
    domain: str,
    *,
    workspace: Path,
    diagnostic_rows: int | None,
    request_partitions: int,
    source_workers: int,
    keep_temporary: bool,
) -> dict[str, Any]:
    spec = snapshot.domains[domain]
    effective_rows = (
        spec.row_count
        if diagnostic_rows is None
        else min(spec.row_count, diagnostic_rows)
    )
    locks = workspace / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{domain}.lock").open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = completed_domain(
            snapshot,
            domain,
            workspace=workspace,
            effective_rows=effective_rows,
            hash_files=True,
        )
        if existing is not None:
            print(f"[sidecar:{domain}] completed domain is reusable", flush=True)
            return existing

        started = time.perf_counter()
        temporary_root = workspace / "tmp" / f"{domain}.{os.getpid()}.{time.time_ns()}"
        temporary_root.mkdir(parents=True, exist_ok=False)
        try:
            catalog_identity = verify_catalog_file(
                spec, hash_bytes=diagnostic_rows is None
            )
            requests = partition_catalog_requests(
                spec,
                effective_rows=effective_rows,
                partitions=request_partitions,
                request_dir=temporary_root / "requests",
                catalog_identity=catalog_identity,
            )
            source_schema = CAPTION_SCHEMAS[domain]
            output_schema = sidecar_schema(source_schema, CAPTION_COLUMNS[domain])
            values_dir = temporary_root / "values"
            values: list[ValuePartition] = []
            domain_source_identities: dict[
                str, tuple[int, int, int, int, int]
            ] = {}
            print(
                f"[sidecar:{domain}] source partitions={len(requests)} "
                f"workers={source_workers}",
                flush=True,
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(source_workers, len(requests)),
                thread_name_prefix=f"caption-sidecar-{domain}",
            ) as pool:
                futures = {
                    pool.submit(
                        build_value_partition,
                        request,
                        source_schema=source_schema,
                        output_schema=output_schema,
                        value_dir=values_dir,
                    ): request
                    for request in requests
                }
                for done, future in enumerate(
                    concurrent.futures.as_completed(futures), 1
                ):
                    request = futures[future]
                    try:
                        value_partition = future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"source partition failed {domain}/{request.index}: {exc}"
                        ) from exc
                    for path_text, identity in value_partition.source_identities.items():
                        previous = domain_source_identities.setdefault(
                            path_text, identity
                        )
                        if previous != identity:
                            raise RuntimeError(
                                f"caption source mixed snapshots across partitions for "
                                f"{domain}: {path_text}: {previous} != {identity}"
                            )
                    values.append(value_partition)
                    if done % 8 == 0 or done == len(futures):
                        print(
                            f"[sidecar:{domain}] source partitions {done}/{len(futures)}",
                            flush=True,
                        )
            values.sort(key=lambda item: item.index)
            staged_domain = temporary_root / "staged" / domain
            staged_domain.parent.mkdir(parents=True, exist_ok=True)
            files, source_digest = merge_value_partitions(
                domain,
                row_base=spec.row_base,
                row_count=effective_rows,
                values=values,
                schema=output_schema,
                output_dir=staged_domain,
            )
            decoded_digest = full_reread_digest(
                domain,
                row_base=spec.row_base,
                row_count=effective_rows,
                files=files,
                directory=staged_domain,
                schema=output_schema,
            )
            if source_digest != decoded_digest:
                raise ValueError(
                    f"source/full-reread semantic digest mismatch for {domain}: "
                    f"{source_digest} != {decoded_digest}"
                )
            if stat_identity(regular_file(spec.path)) != catalog_identity:
                raise RuntimeError(f"catalog changed during domain build: {spec.path}")
            if sha256_file(snapshot.header_path) != snapshot.header_sha256:
                raise RuntimeError("catalog header changed during domain build")
            # Final source restat occurs after every source partition, merge,
            # sidecar write, and full semantic reread, immediately before the
            # recoverable domain marker is made visible.
            source_snapshot = freeze_source_snapshot(domain_source_identities)
            marker: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "semantic_digest_algorithm": SEMANTIC_DIGEST_ALGORITHM,
                "builder_build_id": BUILDER_BUILD_ID,
                "caption_schema_contract_sha256": CAPTION_SCHEMA_CONTRACT_SHA256,
                "catalog_hash": snapshot.catalog_hash,
                "catalog_header_sha256": snapshot.header_sha256,
                "domain": domain,
                "row_base": spec.row_base,
                "row_count": effective_rows,
                "columns": list(CAPTION_COLUMNS[domain]),
                "source_semantics_sha256": source_digest,
                "decoded_sidecar_semantics_sha256": decoded_digest,
                "source_snapshot": source_snapshot,
                "files": files,
                "catalog_file": {
                    "name": spec.file_name,
                    "row_base": spec.row_base,
                    "rows": spec.row_count,
                    "bytes": spec.size_bytes,
                    "sha256": spec.sha256,
                    "schema": spec.schema,
                },
                "build": {
                    "diagnostic_rows_per_domain": diagnostic_rows,
                    "catalog_file_sha256_live_verified": diagnostic_rows is None,
                    "request_partitions": request_partitions,
                    "nonempty_request_partitions": len(requests),
                    "source_workers": source_workers,
                    "source_schema": str(source_schema),
                    "sidecar_schema": str(output_schema),
                    "duration_seconds": time.perf_counter() - started,
                },
            }
            publish_completed_domain(
                workspace,
                domain,
                staged_domain,
                marker,
                domain_source_identities,
            )
            completed = completed_domain(
                snapshot,
                domain,
                workspace=workspace,
                effective_rows=effective_rows,
                hash_files=True,
            )
            if completed is None:
                raise AssertionError(f"completed domain marker vanished: {domain}")
            print(
                f"[sidecar:{domain}] PASS rows={effective_rows} "
                f"digest={source_digest}",
                flush=True,
            )
            return completed
        finally:
            if not keep_temporary and temporary_root.exists():
                shutil.rmtree(temporary_root)


def exact_root_entries(root: Path, markers: dict[str, dict[str, Any]]) -> None:
    regular_dir(root)
    expected_top = {"header.json", "_SUCCESS", *DOMAINS}
    top_entries = list(root.iterdir())
    live_top = {path.name for path in top_entries}
    if live_top != expected_top:
        raise ValueError(
            f"published root entries drift: missing={sorted(expected_top-live_top)}, "
            f"unexpected={sorted(live_top-expected_top)}"
        )
    regular_file(root / "header.json")
    regular_file(root / "_SUCCESS")
    for domain in DOMAINS:
        directory = root / domain
        regular_dir(directory)
        expected = {Path(item["name"]).name for item in markers[domain]["files"]}
        entries = list(directory.iterdir())
        for path in entries:
            regular_file(path)
        live = {path.name for path in entries}
        if live != expected:
            raise ValueError(f"published domain file set drift: {domain}")


def validate_fingerprint_closure(
    fingerprints: dict[Path, tuple[str, tuple[int, int, int, int, int]]],
) -> None:
    for path, (kind, expected_identity) in sorted(
        fingerprints.items(), key=lambda item: str(item[0])
    ):
        metadata = regular_dir(path) if kind == "dir" else regular_file(path)
        live_identity = stat_identity(metadata)
        if live_identity != expected_identity:
            raise RuntimeError(
                f"filesystem fingerprint changed during closure: {path}: "
                f"live={live_identity}, expected={expected_identity}"
            )


def validate_existing_output_root(
    root: Path,
    *,
    expected_header: dict[str, Any],
    markers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Fully validate an already-visible root before treating it as success."""

    fingerprints: dict[
        Path, tuple[str, tuple[int, int, int, int, int]]
    ] = {}

    def remember(path: Path, kind: str) -> None:
        metadata = regular_dir(path) if kind == "dir" else regular_file(path)
        identity = stat_identity(metadata)
        previous = fingerprints.setdefault(path, (kind, identity))
        if previous != (kind, identity):
            raise RuntimeError(f"published root path changed during capture: {path}")

    remember(root, "dir")
    remember(root / "header.json", "file")
    remember(root / "_SUCCESS", "file")
    for domain in DOMAINS:
        remember(root / domain, "dir")
    exact_root_entries(root, markers)
    header, header_raw_sha256 = read_json_snapshot(root / "header.json")
    if set(header) != set(expected_header) or not isinstance(
        header.get("generated_at"), str
    ):
        raise ValueError(f"existing sidecar header key set drift: {root}")
    header_sha256 = header.get("header_sha256")
    if (
        not is_sha256(header_sha256)
        or header_self_hash(header) != header_sha256
        or header_sha256 != expected_header.get("header_sha256")
    ):
        raise ValueError(f"existing sidecar header identity drift: {root}")
    success_bytes = (root / "_SUCCESS").read_bytes()
    if success_bytes not in {
        header_sha256.encode("ascii"),
        (header_sha256 + "\n").encode("ascii"),
    }:
        raise ValueError(f"existing sidecar _SUCCESS/header drift: {root}")

    for domain in DOMAINS:
        domain_header = header.get("domains", {}).get(domain)
        if domain_header != expected_header["domains"][domain]:
            raise ValueError(f"existing sidecar domain header drift: {domain}")
        schema = sidecar_schema(
            CAPTION_SCHEMAS[domain], CAPTION_COLUMNS[domain]
        )
        cursor = int(domain_header["row_base"])
        expected_stop = cursor + int(domain_header["row_count"])
        for index, entry in enumerate(domain_header["files"]):
            path = root / str(entry["name"])
            remember(path, "file")
            before = regular_file(path)
            if before.st_size != entry.get("bytes"):
                raise ValueError(f"existing sidecar byte-size drift: {path}")
            if sha256_file(path) != entry.get("sha256"):
                raise ValueError(f"existing sidecar SHA-256 drift: {path}")
            parquet = pq.ParquetFile(path)
            row_start = int(entry["row_start"])
            row_stop = int(entry["row_stop"])
            rows = row_stop - row_start
            row_groups = math.ceil(rows / ROW_GROUP_ROWS)
            if (
                row_start != cursor
                or entry.get("rows") != rows
                or entry.get("row_groups") != row_groups
                or parquet.metadata.num_rows != rows
                or parquet.num_row_groups != row_groups
                or not parquet.schema_arrow.equals(schema, check_metadata=True)
                or str(parquet.schema_arrow) != str(schema)
                or entry.get("schema") != str(schema)
            ):
                raise ValueError(
                    f"existing sidecar footer/schema/layout drift: {domain}/{index}"
                )
            for row_group in range(parquet.num_row_groups):
                expected_rows = min(ROW_GROUP_ROWS, row_stop - cursor)
                if parquet.metadata.row_group(row_group).num_rows != expected_rows:
                    raise ValueError(
                        f"existing sidecar row-group size drift: {path}:{row_group}"
                    )
                row_ids = parquet.read_row_group(
                    row_group, columns=["row_id"], use_threads=False
                ).column("row_id").to_pylist()
                if row_ids != list(range(cursor, cursor + expected_rows)):
                    raise ValueError(
                        f"existing sidecar physical row_id drift: {path}:{row_group}"
                    )
                cursor += expected_rows
            after = regular_file(path)
            if stat_identity(before) != stat_identity(after):
                raise RuntimeError(f"existing sidecar changed during validation: {path}")
        if cursor != expected_stop:
            raise ValueError(f"existing sidecar domain coverage drift: {domain}")

    # Full-root closure: controls are decoded again, the exact directory set is
    # enumerated again, then every file/directory fingerprint collected above is
    # restatted with dev/inode/size/mtime/ctime as the final filesystem action.
    closing_header, closing_header_raw_sha256 = read_json_snapshot(
        root / "header.json"
    )
    closing_success_bytes = (root / "_SUCCESS").read_bytes()
    if (
        closing_header_raw_sha256 != header_raw_sha256
        or closing_header != header
        or header_self_hash(closing_header) != header_sha256
        or closing_success_bytes != success_bytes
    ):
        raise RuntimeError(f"published control files changed during closure: {root}")
    exact_root_entries(root, markers)
    validate_fingerprint_closure(fingerprints)
    return header


def publish_sidecar(
    snapshot: CatalogSnapshot,
    *,
    workspace: Path,
    output_root: Path,
    diagnostic_rows: int | None,
) -> dict[str, Any] | None:
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / "finalize.lock").open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if sha256_file(snapshot.header_path) != snapshot.header_sha256:
            raise RuntimeError("catalog header changed before sidecar publication")
        fast_missing = [
            domain
            for domain in DOMAINS
            if not (workspace / "markers" / f"{domain}.json").is_file()
            or not (workspace / "complete" / domain).is_dir()
        ]
        if fast_missing:
            print(
                f"[sidecar] publication deferred; incomplete domains={fast_missing}",
                flush=True,
            )
            return None
        markers: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for domain in DOMAINS:
            spec = snapshot.domains[domain]
            effective_rows = (
                spec.row_count
                if diagnostic_rows is None
                else min(spec.row_count, diagnostic_rows)
            )
            marker = completed_domain(
                snapshot,
                domain,
                workspace=workspace,
                effective_rows=effective_rows,
                hash_files=True,
            )
            if marker is None:
                missing.append(domain)
            else:
                markers[domain] = marker
        if missing:
            print(
                f"[sidecar] publication deferred; incomplete domains={missing}",
                flush=True,
            )
            return None

        # Rebind all five live catalog files immediately before formal
        # publication.  Diagnostic mode is intentionally bounded and validates
        # footer/stat identity without hashing every full production catalog.
        for domain in DOMAINS:
            verify_catalog_file(
                snapshot.domains[domain], hash_bytes=diagnostic_rows is None
            )
        if sha256_file(snapshot.header_path) != snapshot.header_sha256:
            raise RuntimeError("catalog header changed while finalizing sidecar")

        output_root = output_root.expanduser().resolve(strict=False)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        publish_root = output_root.parent / (
            f".{output_root.name}.publish.{os.getpid()}.{time.time_ns()}"
        )
        publish_root.mkdir()
        try:
            for domain in DOMAINS:
                source_dir = workspace / "complete" / domain
                destination_dir = publish_root / domain
                destination_dir.mkdir()
                for item in markers[domain]["files"]:
                    source = source_dir / Path(item["name"]).name
                    destination = destination_dir / source.name
                    try:
                        os.link(source, destination, follow_symlinks=False)
                    except OSError as exc:
                        raise RuntimeError(
                            "sidecar publication requires work/output roots on the same "
                            f"filesystem for verified hard links: {source} -> {destination}: {exc}"
                        ) from exc
                fsync_dir(destination_dir)

            domains_header = {
                domain: {
                    "row_base": markers[domain]["row_base"],
                    "row_count": markers[domain]["row_count"],
                    "columns": markers[domain]["columns"],
                    "source_semantics_sha256": markers[domain][
                        "source_semantics_sha256"
                    ],
                    "decoded_sidecar_semantics_sha256": markers[domain][
                        "decoded_sidecar_semantics_sha256"
                    ],
                    "files": markers[domain]["files"],
                }
                for domain in DOMAINS
            }
            header: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "semantic_digest_algorithm": SEMANTIC_DIGEST_ALGORITHM,
                "builder_build_id": BUILDER_BUILD_ID,
                "caption_schema_contract_sha256": CAPTION_SCHEMA_CONTRACT_SHA256,
                "catalog_hash": snapshot.catalog_hash,
                "catalog_header_sha256": snapshot.header_sha256,
                "row_order": ROW_ORDER,
                "file_rows": FILE_ROWS,
                "row_group_rows": ROW_GROUP_ROWS,
                "domain_order": list(DOMAINS),
                "total_rows": sum(value["row_count"] for value in domains_header.values()),
                "diagnostic_run": diagnostic_rows is not None,
                "diagnostic_rows_per_domain": diagnostic_rows,
                "catalog_files": {
                    domain: markers[domain]["catalog_file"] for domain in DOMAINS
                },
                "domains": domains_header,
                "generated_at": utc_now(),
            }
            header["header_sha256"] = header_self_hash(header)
            atomic_write_json(publish_root / "header.json", header)
            atomic_write_text(
                publish_root / "_SUCCESS", header["header_sha256"] + "\n", encoding="ascii"
            )
            exact_root_entries(publish_root, markers)
            fsync_dir(publish_root)

            if output_root.exists():
                try:
                    existing = validate_existing_output_root(
                        output_root,
                        expected_header=header,
                        markers=markers,
                    )
                except Exception as exc:
                    raise FileExistsError(
                        f"refusing to reuse or replace invalid existing sidecar root "
                        f"{output_root}: {exc}"
                    ) from exc
                if existing.get("header_sha256") == header["header_sha256"]:
                    print(
                        f"[sidecar] identical output fully revalidated: {output_root}",
                        flush=True,
                    )
                    return existing
                raise AssertionError("validated existing sidecar identity changed")
            os.rename(publish_root, output_root)
            fsync_dir(output_root.parent)
            published = validate_existing_output_root(
                output_root,
                expected_header=header,
                markers=markers,
            )
            print(
                f"[sidecar] PUBLISHED root={output_root} "
                f"header_sha256={header['header_sha256']}",
                flush=True,
            )
            return published
        finally:
            if publish_root.exists():
                shutil.rmtree(publish_root)


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="offline-caption-sidecar-selftest-") as raw:
        root = Path(raw)

        def digest_round_trip(domain: str, table: pa.Table, name: str) -> str:
            rows = table.num_rows
            source = ArrowIpcSemanticDigester(
                domain=domain,
                row_base=int(table.column("row_id")[0].as_py()),
                row_count=rows,
                schema=table.schema,
            )
            for offset in range(0, rows, ROW_GROUP_ROWS):
                source.update(
                    table.slice(offset, min(ROW_GROUP_ROWS, rows - offset))
                )
            source_digest = source.finish()
            path = root / name
            pq.write_table(
                table,
                path,
                compression="zstd",
                row_group_size=ROW_GROUP_ROWS,
                use_dictionary=True,
            )
            parquet = pq.ParquetFile(path)
            if not parquet.schema_arrow.equals(table.schema, check_metadata=True):
                raise ValueError(f"self-test parquet schema drift: {name}")
            decoded = ArrowIpcSemanticDigester(
                domain=domain,
                row_base=int(table.column("row_id")[0].as_py()),
                row_count=rows,
                schema=table.schema,
            )
            for row_group in range(parquet.num_row_groups):
                decoded.update(parquet.read_row_group(row_group, use_threads=False))
            decoded_digest = decoded.finish()
            if source_digest != decoded_digest:
                raise ValueError(
                    f"self-test semantic digest mismatch {name}: "
                    f"{source_digest} != {decoded_digest}"
                )
            return source_digest

        # Exercise the frozen anime large_string/large_list schema, Unicode,
        # null lists, null list elements, and a non-zero IPC slice offset.
        anime_rows = 300
        anime_source_schema = CAPTION_SCHEMAS["anime"]
        anime_arrays: list[pa.Array] = []
        for field in anime_source_schema:
            if pa.types.is_large_list(field.type):
                values = [
                    None
                    if index % 13 == 0
                    else [f"tag-{index}", None if index % 17 == 0 else "猫"]
                    for index in range(anime_rows)
                ]
            else:
                values = [
                    None if index % 11 == 0 else f"{field.name}-{index}-雪"
                    for index in range(anime_rows)
                ]
            anime_arrays.append(pa.array(values, type=field.type))
        anime_schema = sidecar_schema(
            anime_source_schema, CAPTION_COLUMNS["anime"]
        )
        anime_table = pa.Table.from_arrays(
            [pa.array(range(anime_rows), type=pa.int64()), *anime_arrays],
            schema=anime_schema,
        )
        anime_digest = digest_round_trip(
            "anime", anime_table, "anime_typed.parquet"
        )

        # Reproduce the production failure mode: the first source shard has a
        # physical Arrow null field, followed by a normal nullable string shard.
        # Both must become the same canonical nullable string before merge.
        null_source_path = root / "gpic_source_00_all_null.parquet"
        mixed_source_path = root / "gpic_source_01_mixed_string.parquet"
        wrong_source_path = root / "gpic_source_02_wrong_int.parquet"
        pq.write_table(
            pa.Table.from_arrays(
                [pa.nulls(3)],
                schema=pa.schema([pa.field("caption", pa.null(), nullable=True)]),
            ),
            null_source_path,
        )
        pq.write_table(
            pa.Table.from_arrays(
                [pa.array([None, "mixed-雪", "tail"], type=pa.string())],
                schema=CAPTION_SCHEMAS["gpic"],
            ),
            mixed_source_path,
        )
        pq.write_table(
            pa.Table.from_arrays(
                [pa.array([1], type=pa.int64())],
                schema=pa.schema([pa.field("caption", pa.int64())]),
            ),
            wrong_source_path,
        )
        canonical_source_parts: list[pa.Table] = []
        for source_path in (null_source_path, mixed_source_path):
            source_parquet = pq.ParquetFile(source_path)
            validate_source_schema(
                source_path, source_parquet, CAPTION_SCHEMAS["gpic"]
            )
            decoded_source = source_parquet.read_row_group(
                0, columns=["caption"], use_threads=False
            ).cast(CAPTION_SCHEMAS["gpic"], safe=True)
            if not decoded_source.schema.equals(
                CAPTION_SCHEMAS["gpic"], check_metadata=True
            ):
                raise ValueError("self-test physical null was not canonically typed")
            canonical_source_parts.append(decoded_source)
        if not pa.types.is_null(
            pq.ParquetFile(null_source_path).schema_arrow.field("caption").type
        ):
            raise ValueError("self-test failed to create a physical null source")
        if canonical_source_parts[0].column("caption").null_count != 3:
            raise ValueError("self-test lost physical null values during safe cast")
        try:
            wrong_parquet = pq.ParquetFile(wrong_source_path)
            validate_source_schema(
                wrong_source_path, wrong_parquet, CAPTION_SCHEMAS["gpic"]
            )
        except ValueError:
            pass
        else:
            raise ValueError("self-test accepted incompatible int64 caption source")

        canonical_values = pa.concat_tables(canonical_source_parts)
        gpic_schema = sidecar_schema(
            CAPTION_SCHEMAS["gpic"], CAPTION_COLUMNS["gpic"]
        )
        gpic_table = pa.Table.from_arrays(
            [
                pa.array(range(canonical_values.num_rows), type=pa.int64()),
                canonical_values.column("caption").combine_chunks(),
            ],
            schema=gpic_schema,
        )
        if gpic_table.column("caption").type != pa.string():
            raise ValueError("self-test sidecar caption did not retain string type")
        gpic_digest = digest_round_trip(
            "gpic", gpic_table, "gpic_null_promoted.parquet"
        )

        # Exercise the exact dev/inode/size/mtime/ctime closure primitive used
        # after full existing-root and newly-published-root verification.
        closure_probe = root / "closure_probe.txt"
        closure_probe.write_text("stable", encoding="ascii")
        closure_fingerprints = {
            closure_probe: ("file", stat_identity(regular_file(closure_probe)))
        }
        validate_fingerprint_closure(closure_fingerprints)
        closure_probe.write_text("tampered-content", encoding="ascii")
        try:
            validate_fingerprint_closure(closure_fingerprints)
        except RuntimeError:
            closure_tamper_rejected = True
        else:
            raise ValueError("self-test accepted a closure fingerprint mutation")

    sample_header = {
        "schema_version": SCHEMA_VERSION,
        "builder_build_id": BUILDER_BUILD_ID,
        "caption_schema_contract_sha256": CAPTION_SCHEMA_CONTRACT_SHA256,
        "generated_at": "ignored",
        "value": 1,
    }
    first = header_self_hash(sample_header)
    sample_header["generated_at"] = "also-ignored"
    sample_header["header_sha256"] = first
    if header_self_hash(sample_header) != first:
        raise ValueError("self-test header self-hash exclusion drift")
    print(
        json.dumps(
            {
                "status": "PASS",
                "builder_build_id": BUILDER_BUILD_ID,
                "caption_schema_contract_sha256": CAPTION_SCHEMA_CONTRACT_SHA256,
                "semantic_digest_algorithm": SEMANTIC_DIGEST_ALGORITHM,
                "anime_rows": anime_rows,
                "anime_digest": anime_digest,
                "gpic_promoted_rows": gpic_table.num_rows,
                "gpic_digest": gpic_digest,
                "physical_null_first_source": True,
                "mixed_null_string_second_source": True,
                "closure_tamper_rejected": closure_tamper_rejected,
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument(
        "--domain",
        action="append",
        choices=DOMAINS,
        help="build one domain; repeat for multiple (default: all five)",
    )
    parser.add_argument(
        "--domain-workers",
        type=int,
        default=1,
        help="number of selected domains to build concurrently in this process",
    )
    parser.add_argument("--source-workers", type=int, default=8)
    parser.add_argument("--request-partitions", type=int, default=REQUEST_PARTITIONS)
    parser.add_argument(
        "--diagnostic-rows-per-domain",
        type=int,
        default=None,
        help="build only the first N global-catalog rows of each selected domain",
    )
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="write recoverable domain products but never attempt five-domain publication",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="publish already completed domain products without building",
    )
    parser.add_argument("--keep-temporary", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("domain_workers", "source_workers", "request_partitions"):
        value = getattr(args, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (
        args.diagnostic_rows_per_domain is not None
        and args.diagnostic_rows_per_domain <= 0
    ):
        raise ValueError("--diagnostic-rows-per-domain must be positive")
    selected = tuple(args.domain or DOMAINS)
    if len(set(selected)) != len(selected):
        raise ValueError(f"duplicate --domain values: {selected!r}")
    if args.finalize_only and args.no_finalize:
        raise ValueError("--finalize-only and --no-finalize are mutually exclusive")
    output = args.output_root.expanduser().resolve(strict=False)
    if (
        args.diagnostic_rows_per_domain is not None
        and output == DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    ):
        raise ValueError(
            "diagnostic mode refuses the canonical formal output root; pass a distinct "
            "--output-root"
        )


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    if args.self_test:
        return run_self_test()
    snapshot = load_catalog_snapshot(args.catalog_root)
    workspace = workspace_for(
        args.work_root.expanduser().resolve(strict=False),
        snapshot.catalog_hash,
        args.diagnostic_rows_per_domain,
    )
    workspace.mkdir(parents=True, exist_ok=True)
    if not args.finalize_only:
        selected = tuple(args.domain or DOMAINS)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.domain_workers, len(selected)),
            thread_name_prefix="caption-sidecar-domain",
        ) as pool:
            futures = {
                pool.submit(
                    build_domain,
                    snapshot,
                    domain,
                    workspace=workspace,
                    diagnostic_rows=args.diagnostic_rows_per_domain,
                    request_partitions=args.request_partitions,
                    source_workers=args.source_workers,
                    keep_temporary=args.keep_temporary,
                ): domain
                for domain in selected
            }
            for future in concurrent.futures.as_completed(futures):
                domain = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"domain build failed for {domain}: {exc}") from exc
    if not args.no_finalize:
        publish_sidecar(
            snapshot,
            workspace=workspace,
            output_root=args.output_root,
            diagnostic_rows=args.diagnostic_rows_per_domain,
        )
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except Exception as exc:
        print(
            f"FATAL caption sidecar builder: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
