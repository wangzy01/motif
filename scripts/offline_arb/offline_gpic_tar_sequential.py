#!/usr/bin/env python3
"""Fail-closed sequential GPIC tar source for the production blob builder.

The adapter deliberately has no publication surface.  It validates a frozen
sample mapping authority, produces an exact structural mapping authority from
the catalog-derived worker rows, and streams worker-owned tar shards directly
into the existing final-blob writer.  It never creates per-tar payload chunks.
"""

from __future__ import annotations

import dataclasses
import concurrent.futures
import hashlib
import io
import os
import struct
import time
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

from scripts.offline_arb import offline_gpic_s3_production_common as common
from scripts.offline_arb import run_offline_gpic_blob_pilot as pilot_core


SAMPLE_REPORT_SCHEMA = "offline_gpic_tar_mapping_report_v1"
SAMPLE_BINDING_SCHEMA = "offline_gpic_tar_sample_authority_binding_v1"
WORKER_MAPPING_SCHEMA = "offline_gpic_tar_worker_mapping_v1"
FULL_MAPPING_SCHEMA = "offline_gpic_tar_full_mapping_authority_v1"
TAR_TELEMETRY_SCHEMA = "offline_gpic_tar_sequential_telemetry_v1"
TRAIN_TAR_COUNT = 8_000
PRODUCTION_GPIC_ROWS = 20_000_000
PRODUCTION_SAMPLE_SHARDS = (
    0,
    1,
    7,
    31,
    127,
    511,
    1_023,
    2_047,
    3_071,
    4_095,
    5_119,
    6_143,
    7_001,
    7_500,
    7_900,
    7_999,
)
PRODUCTION_TAR_ROOT = Path(
    "/s3-mewtant-raw-datasets/stanford-vision-lab/gpic/train"
)
PRODUCTION_SOURCE_PREFIX = (
    "/s3-mewtant-raw-datasets/stanford-vision-lab/"
    "gpic_umziped/images/train/"
)
IMAGE_SUFFIXES = frozenset({".jpg", ".png"})
TAR_BLOCK_BYTES = 512
DEFAULT_READ_BUFFER_BYTES = 16 * (1 << 20)
DEFAULT_IO_BLOCK_BYTES = 8 * (1 << 20)
MAX_INDEXED_MEMBER_BYTES = 0xFFFFFFFF
MAPPING_RECORD_STRUCT = struct.Struct("<IqqI")
MAPPING_RECORD_ENCODING = (
    "little-endian <uint32 shard,int64 local_index,int64 row_id,uint32 "
    "member_utf8_bytes> followed by canonical member UTF-8"
)


class ItemLike(Protocol):
    local_index: int
    row_id: int
    image_path: str


@dataclasses.dataclass(frozen=True)
class TarSourceRead:
    """Attribute-compatible with ``build_offline...SourceRead``."""

    item: ItemLike
    payload: bytes
    sha256_raw: bytes
    attempts: int
    errors: int
    elapsed_ns: int


@dataclasses.dataclass(frozen=True)
class _Target:
    item: ItemLike
    shard: int
    member: str
    member_raw: bytes


def _strict_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise common.ProductionError(f"{label} must be a positive integer")
    return value


def _strict_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise common.ProductionError(f"{label} must be a nonnegative integer")
    return value


def _file_identity(value: os.stat_result) -> pilot_core.FileIdentity:
    return pilot_core.FileIdentity.from_stat(value)


def expected_tar_name(shard: int) -> str:
    if type(shard) is not int or not (0 <= shard < TRAIN_TAR_COUNT):
        raise common.ProductionError("tar shard escaped 00000..07999")
    return f"gpic_train_{shard:05d}.tar"


def canonical_catalog_target(
    image_path: str, *, source_prefix: str
) -> tuple[int, str, bytes]:
    """Map one catalog path to its exact, non-normalized tar member key."""

    if (
        not isinstance(image_path, str)
        or not image_path.startswith(source_prefix)
        or not source_prefix.startswith("/")
        or not source_prefix.endswith("/")
    ):
        raise common.ProductionError("catalog target escaped frozen source prefix")
    relative = image_path[len(source_prefix) :]
    pieces = relative.split("/")
    if len(pieces) != 2:
        raise common.ProductionError("catalog target is not shard/basename shaped")
    shard_raw, member = pieces
    if (
        len(shard_raw) != 5
        or not shard_raw.isascii()
        or not shard_raw.isdigit()
    ):
        raise common.ProductionError("catalog target shard is not five decimal digits")
    shard = int(shard_raw)
    if not (0 <= shard < TRAIN_TAR_COUNT):
        raise common.ProductionError("catalog target shard is out of range")
    member_raw = _canonical_member(member, suffixes=IMAGE_SUFFIXES)
    return shard, member, member_raw


def _canonical_member(name: str, *, suffixes: frozenset[str]) -> bytes:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise common.ProductionError("unsafe tar member name")
    pure = PurePosixPath(name)
    if (
        pure.name != name
        or name in {".", ".."}
        or pure.suffix not in suffixes
        or not name.isascii()
    ):
        raise common.ProductionError(f"non-canonical tar member name: {name!r}")
    raw = name.encode("ascii")
    if len(raw) > 100:
        raise common.ProductionError("tar member exceeds frozen ustar name field")
    return raw


def _mapping_record(target: _Target) -> bytes:
    return MAPPING_RECORD_STRUCT.pack(
        target.shard,
        target.item.local_index,
        target.item.row_id,
        len(target.member_raw),
    ) + target.member_raw


def _validated_targets(
    items: Iterable[ItemLike],
    *,
    source_prefix: str,
    row_base: int,
    gpic_rows: int,
    worker_id: int,
    shard_owners: Mapping[int, int],
) -> Iterator[_Target]:
    previous: tuple[int, bytes] | None = None
    for position, item in enumerate(items):
        if (
            type(getattr(item, "local_index", None)) is not int
            or type(getattr(item, "row_id", None)) is not int
            or not isinstance(getattr(item, "image_path", None), str)
        ):
            raise common.ProductionError(
                f"malformed catalog target at worker position={position}"
            )
        if (
            item.local_index < 0
            or item.local_index >= gpic_rows
            or item.row_id != row_base + item.local_index
        ):
            raise common.ProductionError(
                f"catalog row-id binding drift local={item.local_index}"
            )
        shard, member, member_raw = canonical_catalog_target(
            item.image_path, source_prefix=source_prefix
        )
        if shard_owners.get(shard) != worker_id:
            raise common.ProductionError(
                f"worker {worker_id} does not own tar shard {shard:05d}"
            )
        key = (shard, member_raw)
        if previous is not None and key <= previous:
            problem = "duplicate" if key == previous else "order regression"
            raise common.ProductionError(
                f"catalog (shard,canonical member) {problem}: {shard:05d}/{member}"
            )
        previous = key
        yield _Target(item=item, shard=shard, member=member, member_raw=member_raw)


def scan_worker_mapping(
    items: Iterable[ItemLike],
    *,
    build_jobtag: str,
    worker_id: int,
    workers: int,
    source_prefix: str,
    row_base: int,
    gpic_rows: int,
    expected_rows: int,
    shard_owners: Mapping[int, int],
    catalog_hash: str,
    gpic_sha256: str,
    partition_plan_raw_sha256: str,
    partition_plan_sha256: str,
) -> dict[str, Any]:
    """Full catalog structural scan; this must complete before payload reads."""

    _strict_nonnegative_int(worker_id, "worker_id")
    _strict_positive_int(workers, "workers")
    _strict_positive_int(gpic_rows, "gpic_rows")
    _strict_positive_int(expected_rows, "expected_rows")
    if worker_id >= workers or not build_jobtag:
        raise common.ProductionError("worker/build identity is malformed")
    for label, value in (
        ("catalog_hash", catalog_hash),
        ("gpic_sha256", gpic_sha256),
        ("partition_plan_raw_sha256", partition_plan_raw_sha256),
        ("partition_plan_sha256", partition_plan_sha256),
    ):
        if not common.is_sha256(value):
            raise common.ProductionError(f"{label} is not SHA-256")
    expected_shards = sorted(
        shard for shard, owner in shard_owners.items() if owner == worker_id
    )
    if not expected_shards:
        raise common.ProductionError("worker owns no tar shards")
    total_digest = hashlib.sha256()
    shard_digest = hashlib.sha256()
    shard_rows = 0
    total_rows = 0
    current_shard: int | None = None
    shard_values: list[dict[str, Any]] = []
    seen_local_indices = bytearray((gpic_rows + 7) // 8)

    def finish_shard() -> None:
        nonlocal shard_digest, shard_rows
        if current_shard is None:
            return
        if shard_rows < 1:
            raise common.ProductionError("empty owned tar shard")
        shard_values.append(
            {
                "shard_id": current_shard,
                "tar_name": expected_tar_name(current_shard),
                "rows": shard_rows,
                "mapping_sha256": shard_digest.hexdigest(),
            }
        )
        shard_digest = hashlib.sha256()
        shard_rows = 0

    for target in _validated_targets(
        items,
        source_prefix=source_prefix,
        row_base=row_base,
        gpic_rows=gpic_rows,
        worker_id=worker_id,
        shard_owners=shard_owners,
    ):
        byte_index, bit_index = divmod(target.item.local_index, 8)
        mask = 1 << bit_index
        if seen_local_indices[byte_index] & mask:
            raise common.ProductionError(
                f"duplicate explicit catalog row local={target.item.local_index}"
            )
        seen_local_indices[byte_index] |= mask
        if current_shard != target.shard:
            finish_shard()
            current_shard = target.shard
        record = _mapping_record(target)
        total_digest.update(record)
        shard_digest.update(record)
        shard_rows += 1
        total_rows += 1
    finish_shard()
    observed_shards = [value["shard_id"] for value in shard_values]
    if total_rows != expected_rows or observed_shards != expected_shards:
        raise common.ProductionError(
            "worker full structural mapping row/shard closure failed"
        )
    value = {
        "schema_version": WORKER_MAPPING_SCHEMA,
        "status": "PASS",
        "build_jobtag": build_jobtag,
        "worker_id": worker_id,
        "workers": workers,
        "source_prefix": source_prefix,
        "row_base": row_base,
        "gpic_rows": gpic_rows,
        "rows": total_rows,
        "unique_mapping_keys": total_rows,
        "catalog_hash": catalog_hash,
        "gpic_sha256": gpic_sha256,
        "partition_plan_raw_sha256": partition_plan_raw_sha256,
        "partition_plan_sha256": partition_plan_sha256,
        "mapping_key": "(tar_shard,canonical_member_basename)",
        "mapping_record_encoding": MAPPING_RECORD_ENCODING,
        "mapping_order": "tar_shard_ascending_then_member_ascii_binary",
        "mapping_sha256": total_digest.hexdigest(),
        "shard_count": len(shard_values),
        "shards": shard_values,
        "ownership": "every listed source tar shard is exclusively worker-owned",
        "payload_reads_before_authority": 0,
    }
    value["worker_mapping_sha256"] = common.self_hash(
        value, "worker_mapping_sha256"
    )
    return value


def validate_worker_mapping(
    value: dict[str, Any],
    *,
    worker_id: int,
    workers: int,
    source_prefix: str,
    row_base: int,
    gpic_rows: int,
    expected_rows: int,
    shard_owners: Mapping[int, int],
) -> None:
    common.validate_self_hash(value, "worker_mapping_sha256", "worker mapping")
    shards = value.get("shards")
    expected_shards = sorted(
        shard for shard, owner in shard_owners.items() if owner == worker_id
    )
    if (
        value.get("schema_version") != WORKER_MAPPING_SCHEMA
        or value.get("status") != "PASS"
        or value.get("worker_id") != worker_id
        or value.get("workers") != workers
        or value.get("source_prefix") != source_prefix
        or value.get("row_base") != row_base
        or value.get("gpic_rows") != gpic_rows
        or value.get("rows") != expected_rows
        or value.get("unique_mapping_keys") != expected_rows
        or value.get("mapping_record_encoding") != MAPPING_RECORD_ENCODING
        or not common.is_sha256(value.get("mapping_sha256"))
        or not isinstance(shards, list)
        or value.get("shard_count") != len(expected_shards)
        or value.get("payload_reads_before_authority") != 0
    ):
        raise common.ProductionError("worker mapping authority binding drift")
    observed: list[int] = []
    rows = 0
    for entry in shards:
        if not isinstance(entry, dict):
            raise common.ProductionError("malformed worker shard authority")
        shard = entry.get("shard_id")
        count = entry.get("rows")
        if (
            type(shard) is not int
            or type(count) is not int
            or count < 1
            or entry.get("tar_name") != expected_tar_name(shard)
            or not common.is_sha256(entry.get("mapping_sha256"))
        ):
            raise common.ProductionError("worker shard authority binding drift")
        observed.append(shard)
        rows += count
    if observed != expected_shards or rows != expected_rows:
        raise common.ProductionError("worker mapping authority row/shard union drift")


def load_sample_mapping_authority(
    report_path: Path,
    *,
    expected_raw_sha256: str,
    catalog_hash: str,
    gpic_rows: int,
    row_base: int,
    gpic_sha256: str,
    tar_root: Path,
    source_prefix: str,
    production_gate: bool,
) -> dict[str, Any]:
    """Validate canonical raw bytes, report self-hash, and sibling marker."""

    report_path = report_path.resolve(strict=True)
    marker_path = report_path.parent / "_COMPLETE"
    report, raw = common.strict_json(report_path)
    raw_sha = hashlib.sha256(raw).hexdigest()
    marker_before = common.require_regular_file(marker_path)
    marker_raw = marker_path.read_bytes()
    marker_after = common.require_regular_file(marker_path)
    if (
        not common.is_sha256(expected_raw_sha256)
        or raw_sha != expected_raw_sha256
        or marker_raw != (expected_raw_sha256 + "\n").encode("ascii")
        or _file_identity(marker_before) != _file_identity(marker_after)
    ):
        raise common.ProductionError("sample mapping raw SHA/_COMPLETE drift")
    common.validate_self_hash(report, "report_sha256", "sample mapping report")
    inventory = report.get("inventory")
    sample = report.get("sample")
    tar_results = report.get("tar_results")
    if not isinstance(inventory, dict) or not isinstance(sample, dict):
        raise common.ProductionError("sample mapping evidence is malformed")
    common.validate_self_hash(inventory, "inventory_sha256", "tar inventory")
    common.validate_self_hash(sample, "sample_sha256", "catalog tar sample")
    if (
        report.get("schema_version") != SAMPLE_REPORT_SCHEMA
        or report.get("status") != "PASS"
        or report.get("read_only_probe") is not True
        or report.get("tar_root") != str(tar_root)
        or report.get("source_prefix") != source_prefix
        or report.get("catalog_hash") != catalog_hash
        or report.get("gpic_rows") != gpic_rows
        or report.get("gpic_row_base") != row_base
        or report.get("gpic_sha256") != gpic_sha256
        or report.get("exact_catalog_tar_payload_equivalence") is not True
        or report.get("all_sampled_tars_pairing_valid") is not True
        or report.get("all_catalog_samples_found_in_derived_tar") is not True
        or report.get("tar_member_name_prefixes") != [""]
        or report.get("writes", {}).get("s3_writes") != 0
        or inventory.get("exact_tar_name_set") is not True
        or inventory.get("missing_tar_names") != []
        or inventory.get("unexpected_tar_names") != []
        or inventory.get("duplicate_tar_name_count") != 0
        or inventory.get("tar_root") != str(tar_root)
        or sample.get("catalog_total_rows") != gpic_rows
        or sample.get("catalog_rows_scanned") != gpic_rows
        or not isinstance(tar_results, list)
    ):
        raise common.ProductionError("sample mapping authority semantic drift")
    for result in tar_results:
        if not isinstance(result, dict):
            raise common.ProductionError("malformed sampled tar result")
        common.validate_self_hash(result, "tar_evidence_sha256", "sampled tar")
        if (
            result.get("pairing_valid") is not True
            or result.get("exact_payload_equivalence") is not True
            or result.get("missing_member_basenames") != []
        ):
            raise common.ProductionError("sampled tar equivalence drift")
    if production_gate and (
        tar_root != PRODUCTION_TAR_ROOT
        or source_prefix != PRODUCTION_SOURCE_PREFIX
        or gpic_rows != PRODUCTION_GPIC_ROWS
        or inventory.get("expected_tar_count") != TRAIN_TAR_COUNT
        or inventory.get("observed_tar_count") != TRAIN_TAR_COUNT
        or tuple(report.get("tar_shards", ())) != PRODUCTION_SAMPLE_SHARDS
        or report.get("sample_comparisons") != 256
        or report.get("byte_equal_comparisons") != 256
        or report.get("sha256_equal_comparisons") != 256
        or sample.get("samples_per_tar") != 16
        or sample.get("sample_rows") != 256
        or sample.get("sample_image_paths_unique") != 256
        or sample.get("catalog_total_row_groups") != 314
        or sample.get("catalog_row_groups_scanned") != 314
        or len(tar_results) != 16
    ):
        raise common.ProductionError("production sample mapping shape drift")
    binding = {
        "schema_version": SAMPLE_BINDING_SCHEMA,
        "path": str(report_path),
        "bytes": len(raw),
        "raw_sha256": raw_sha,
        "report_sha256": report["report_sha256"],
        "complete_path": str(marker_path),
        "complete_bytes": len(marker_raw),
        "complete_raw_sha256": hashlib.sha256(marker_raw).hexdigest(),
        "inventory_sha256": inventory["inventory_sha256"],
        "sample_sha256": sample["sample_sha256"],
        "catalog_hash": catalog_hash,
        "gpic_rows": gpic_rows,
        "row_base": row_base,
        "gpic_sha256": gpic_sha256,
        "tar_root": str(tar_root),
        "source_prefix": source_prefix,
        "sample_comparisons": report["sample_comparisons"],
        "byte_equal_comparisons": report["byte_equal_comparisons"],
        "sha256_equal_comparisons": report["sha256_equal_comparisons"],
    }
    binding["sample_binding_sha256"] = common.self_hash(
        binding, "sample_binding_sha256"
    )
    return binding


def make_full_mapping_authority(
    worker_mappings: Sequence[dict[str, Any]],
    *,
    build_jobtag: str,
    workers: int,
    shard_owners: Mapping[int, int],
    source_prefix: str,
    tar_root: Path,
    row_base: int,
    gpic_rows: int,
    catalog_hash: str,
    gpic_sha256: str,
    partition_plan_raw_sha256: str,
    partition_plan_sha256: str,
    sample_binding: dict[str, Any],
    production_gate: bool,
) -> dict[str, Any]:
    """Merge worker scans into the exact, pre-payload global authority."""

    common.validate_self_hash(
        sample_binding, "sample_binding_sha256", "sample mapping binding"
    )
    if len(worker_mappings) != workers:
        raise common.ProductionError("worker mapping authority count drift")
    ordered = sorted(worker_mappings, key=lambda item: item.get("worker_id", -1))
    shard_entries: dict[int, dict[str, Any]] = {}
    rows = 0
    worker_bindings: list[dict[str, Any]] = []
    for worker_id, value in enumerate(ordered):
        expected_rows = sum(
            entry.get("rows", 0)
            for entry in value.get("shards", [])
            if isinstance(entry, dict)
        )
        validate_worker_mapping(
            value,
            worker_id=worker_id,
            workers=workers,
            source_prefix=source_prefix,
            row_base=row_base,
            gpic_rows=gpic_rows,
            expected_rows=expected_rows,
            shard_owners=shard_owners,
        )
        if (
            value.get("build_jobtag") != build_jobtag
            or value.get("catalog_hash") != catalog_hash
            or value.get("gpic_sha256") != gpic_sha256
            or value.get("partition_plan_raw_sha256")
            != partition_plan_raw_sha256
            or value.get("partition_plan_sha256") != partition_plan_sha256
        ):
            raise common.ProductionError("worker mapping global binding drift")
        for entry in value["shards"]:
            shard = entry["shard_id"]
            if shard in shard_entries or shard_owners.get(shard) != worker_id:
                raise common.ProductionError("global tar shard ownership overlap")
            shard_entries[shard] = {
                **entry,
                "worker_id": worker_id,
            }
        rows += value["rows"]
        worker_bindings.append(
            {
                "worker_id": worker_id,
                "rows": value["rows"],
                "shard_count": value["shard_count"],
                "mapping_sha256": value["mapping_sha256"],
                "worker_mapping_sha256": value["worker_mapping_sha256"],
            }
        )
    expected_shards = sorted(shard_owners)
    if sorted(shard_entries) != expected_shards or rows != gpic_rows:
        raise common.ProductionError("global structural mapping union drift")
    if production_gate and (
        workers != 8
        or gpic_rows != PRODUCTION_GPIC_ROWS
        or expected_shards != list(range(TRAIN_TAR_COUNT))
        or tar_root != PRODUCTION_TAR_ROOT
        or source_prefix != PRODUCTION_SOURCE_PREFIX
    ):
        raise common.ProductionError("production full mapping shape drift")
    shard_union = [shard_entries[shard] for shard in expected_shards]
    union_sha = hashlib.sha256(
        common.canonical_json_bytes(shard_union)
    ).hexdigest()
    value = {
        "schema_version": FULL_MAPPING_SCHEMA,
        "status": "PASS",
        "generated_at": common.utc_now(),
        "build_jobtag": build_jobtag,
        "workers": workers,
        "tar_root": str(tar_root),
        "source_prefix": source_prefix,
        "row_base": row_base,
        "gpic_rows": gpic_rows,
        "rows": rows,
        "unique_mapping_keys": rows,
        "catalog_hash": catalog_hash,
        "gpic_sha256": gpic_sha256,
        "partition_plan_raw_sha256": partition_plan_raw_sha256,
        "partition_plan_sha256": partition_plan_sha256,
        "sample_mapping": sample_binding,
        "mapping_key": "(tar_shard,canonical_member_basename)",
        "mapping_record_encoding": MAPPING_RECORD_ENCODING,
        "mapping_union_sha256": union_sha,
        "shard_count": len(shard_union),
        "shards": shard_union,
        "worker_mappings": worker_bindings,
        "payload_reads_before_authority": 0,
        "publication_order": "canonical_authority_then_COMPLETE_then_payload_scan",
    }
    value["full_mapping_sha256"] = common.self_hash(
        value, "full_mapping_sha256"
    )
    return value


def validate_published_full_mapping(
    path: Path,
    *,
    expected_raw_sha256: str,
    expected_self_sha256: str,
) -> dict[str, Any]:
    """Load an immutable full authority with an exact sibling ``_COMPLETE``."""

    path = path.resolve(strict=True)
    value, raw = common.strict_json(path)
    common.validate_self_hash(value, "full_mapping_sha256", "full tar mapping")
    raw_sha = hashlib.sha256(raw).hexdigest()
    marker = path.parent / "_COMPLETE"
    marker_before = common.require_regular_file(marker)
    marker_raw = marker.read_bytes()
    marker_after = common.require_regular_file(marker)
    if (
        not common.is_sha256(expected_raw_sha256)
        or not common.is_sha256(expected_self_sha256)
        or raw_sha != expected_raw_sha256
        or value.get("full_mapping_sha256") != expected_self_sha256
        or marker_raw != (expected_raw_sha256 + "\n").encode("ascii")
        or _file_identity(marker_before) != _file_identity(marker_after)
        or value.get("schema_version") != FULL_MAPPING_SCHEMA
        or value.get("status") != "PASS"
        or value.get("payload_reads_before_authority") != 0
    ):
        raise common.ProductionError("published full tar mapping authority drift")
    return value


def _parse_octal(field: bytes, label: str) -> int:
    if not field or field[0] & 0x80:
        raise common.ProductionError(f"unsupported tar {label} encoding")
    stripped = field.rstrip(b"\x00 ").lstrip(b" ")
    if not stripped:
        return 0
    if any(byte < ord("0") or byte > ord("7") for byte in stripped):
        raise common.ProductionError(f"invalid tar {label} octal field")
    return int(stripped, 8)


def _parse_header(header: bytes) -> tuple[str, int]:
    if len(header) != TAR_BLOCK_BYTES or header == bytes(TAR_BLOCK_BYTES):
        raise common.ProductionError("invalid nonzero tar header request")
    expected = _parse_octal(header[148:156], "checksum")
    observed = sum(header[:148]) + (8 * ord(" ")) + sum(header[156:])
    if expected != observed:
        raise common.ProductionError("tar header checksum mismatch")
    if header[257:263] != b"ustar\x00" or header[263:265] != b"00":
        raise common.ProductionError("tar header escaped frozen POSIX ustar format")
    if header[345:500].rstrip(b"\x00"):
        raise common.ProductionError("tar header unexpectedly uses a path prefix")
    name_field = header[:100]
    name_raw, separator, tail = name_field.partition(b"\x00")
    if separator and tail.rstrip(b"\x00"):
        raise common.ProductionError("tar name field has bytes after NUL")
    try:
        name = name_raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise common.ProductionError("tar member name is not ASCII") from exc
    if header[156:157] not in {b"0", b"\x00"}:
        raise common.ProductionError("non-regular tar member is forbidden")
    size = _parse_octal(header[124:136], "size")
    if size < 1:
        raise common.ProductionError("empty tar member is forbidden")
    return name, size


class _ExactReader:
    def __init__(self, handle: io.BufferedReader, *, block_bytes: int):
        self.handle = handle
        self.scratch = bytearray(block_bytes)
        self.bytes_read = 0

    def into(self, target: memoryview, *, label: str) -> None:
        cursor = 0
        while cursor < len(target):
            try:
                count = self.handle.readinto(target[cursor:])
            except InterruptedError:
                continue
            except OSError as exc:
                raise common.ProductionError(f"tar read error: {label}") from exc
            if count is None or count <= 0:
                raise common.ProductionError(f"short tar read: {label}")
            cursor += count
            self.bytes_read += count

    def exact(self, length: int, *, label: str) -> bytes:
        target = bytearray(length)
        self.into(memoryview(target), label=label)
        return bytes(target)

    def discard(self, length: int, *, label: str, require_zero: bool = False) -> None:
        remaining = length
        while remaining:
            count = min(remaining, len(self.scratch))
            view = memoryview(self.scratch)[:count]
            self.into(view, label=label)
            if require_zero and any(view):
                raise common.ProductionError(f"nonzero tar end padding: {label}")
            remaining -= count


def _group_targets(
    targets: Iterable[_Target],
) -> Iterator[tuple[int, list[_Target]]]:
    current: int | None = None
    values: list[_Target] = []
    for target in targets:
        if current is None:
            current = target.shard
        if target.shard != current:
            yield current, values
            current = target.shard
            values = []
        values.append(target)
    if current is not None:
        yield current, values


@dataclasses.dataclass(frozen=True)
class _TarBatch:
    shard: int
    results: tuple[TarSourceRead, ...]
    telemetry: dict[str, Any]
    archive_bytes: int
    selected_bytes: int


def _read_one_tar_batch(
    *,
    shard: int,
    group: Sequence[_Target],
    authority: dict[str, Any],
    tar_path: Path,
    scheduled_identity: pilot_core.FileIdentity,
    read_buffer_bytes: int,
    io_block_bytes: int,
    max_selected_bytes_per_tar: int,
    worker_id: int,
    workers: int,
) -> _TarBatch:
    """Read one complete tar.  It has no retry and no write side effect."""

    targets = {target.member: target for target in group}
    if len(targets) != len(group):
        raise common.ProductionError("duplicate selected target within tar batch")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    buffered: io.BufferedReader | None = None
    started_ns = time.perf_counter_ns()
    selected_bytes = 0
    member_count = 0
    json_count = 0
    image_count = 0
    pair_count = 0
    pending_json_stem: str | None = None
    selected_seen: set[str] = set()
    results: list[TarSourceRead] = []
    try:
        try:
            descriptor = os.open(tar_path, flags)
        except OSError as exc:
            raise common.ProductionError(
                f"cannot open worker-owned tar shard={shard:05d}"
            ) from exc
        opened = os.fstat(descriptor)
        identity = _file_identity(opened)
        if (
            identity != scheduled_identity
            or opened.st_size <= 0
            or opened.st_size % TAR_BLOCK_BYTES
        ):
            raise common.ProductionError("tar identity/size shape changed while opening")
        raw = os.fdopen(descriptor, "rb", buffering=0, closefd=False)
        buffered = io.BufferedReader(raw, buffer_size=read_buffer_bytes)
        reader = _ExactReader(buffered, block_bytes=io_block_bytes)
        while True:
            header = reader.exact(TAR_BLOCK_BYTES, label="tar header")
            if header == bytes(TAR_BLOCK_BYTES):
                second = reader.exact(
                    TAR_BLOCK_BYTES, label="second tar end header"
                )
                if second != bytes(TAR_BLOCK_BYTES):
                    raise common.ProductionError(
                        "tar archive lacks two consecutive zero end blocks"
                    )
                trailing = opened.st_size - reader.bytes_read
                if trailing < 0:
                    raise common.ProductionError("tar bytes exceed fstat size")
                reader.discard(
                    trailing,
                    label="tar trailing record padding",
                    require_zero=True,
                )
                break
            name, member_bytes = _parse_header(header)
            member_count += 1
            if pending_json_stem is None:
                _canonical_member(name, suffixes=frozenset({".json"}))
                pending_json_stem = PurePosixPath(name).stem
                json_count += 1
                target: _Target | None = None
            else:
                _canonical_member(name, suffixes=IMAGE_SUFFIXES)
                if PurePosixPath(name).stem != pending_json_stem:
                    raise common.ProductionError(
                        "tar JSON/image canonical stem pairing drift"
                    )
                pending_json_stem = None
                image_count += 1
                pair_count += 1
                if name in selected_seen:
                    raise common.ProductionError(
                        "selected image member appears more than once in tar"
                    )
                target = targets.pop(name, None)
                if target is not None:
                    selected_seen.add(name)
            read_started_ns = time.perf_counter_ns()
            if target is None:
                reader.discard(member_bytes, label=f"tar member {name}")
                payload: bytes | None = None
            else:
                if member_bytes > MAX_INDEXED_MEMBER_BYTES:
                    raise common.ProductionError(
                        "selected tar member escaped uint32 index length"
                    )
                if selected_bytes + member_bytes > max_selected_bytes_per_tar:
                    raise common.ProductionError(
                        "single tar selected-payload memory cap exceeded"
                    )
                payload_work = bytearray(member_bytes)
                reader.into(
                    memoryview(payload_work), label=f"selected tar member {name}"
                )
                payload = bytes(payload_work)
            padding = (-member_bytes) % TAR_BLOCK_BYTES
            if padding:
                reader.discard(padding, label=f"tar member padding {name}")
            if target is not None:
                assert payload is not None
                results.append(
                    TarSourceRead(
                        item=target.item,
                        payload=payload,
                        sha256_raw=hashlib.sha256(payload).digest(),
                        attempts=1,
                        errors=0,
                        elapsed_ns=time.perf_counter_ns() - read_started_ns,
                    )
                )
                selected_bytes += member_bytes
        if pending_json_stem is not None or json_count != image_count:
            raise common.ProductionError("tar ends with an unpaired JSON member")
        if targets or len(results) != authority["rows"]:
            raise common.ProductionError(
                f"selected targets not found exactly once shard={shard:05d}"
            )
        if reader.bytes_read != opened.st_size:
            raise common.ProductionError("tar full-byte scan closure failed")
        after = os.fstat(descriptor)
        path_after = common.require_regular_file(tar_path)
        if identity != _file_identity(after) or identity != _file_identity(path_after):
            raise common.ProductionError("tar identity changed during scan")
        elapsed_ns = time.perf_counter_ns() - started_ns
        telemetry = {
            "schema_version": TAR_TELEMETRY_SCHEMA,
            "worker_id": worker_id,
            "workers": workers,
            "tar_shard": shard,
            "tar_name": tar_path.name,
            "tar_path": str(tar_path),
            "archive_bytes": opened.st_size,
            "archive_bytes_read": reader.bytes_read,
            "appended_rows": len(results),
            "appended_bytes": selected_bytes,
            "header_count": member_count,
            "checksum_validated_headers": member_count,
            "regular_member_count": member_count,
            "json_member_count": json_count,
            "image_member_count": image_count,
            "json_image_pair_count": pair_count,
            "missing_selected_targets": 0,
            "duplicate_selected_targets": 0,
            "attempts": 1,
            "errors": 0,
            "read_buffer_bytes": read_buffer_bytes,
            "io_block_bytes": io_block_bytes,
            "elapsed_ns": elapsed_ns,
            "archive_bytes_per_second": opened.st_size / (elapsed_ns / 1e9),
            "source_identity": dataclasses.asdict(identity),
            "extra_local_chunk_bytes": 0,
            "fsx_writes": 0,
            "s3_writes": 0,
            "stream_order": "tar_shard_ascending_then_archive_member_order",
        }
        telemetry["tar_telemetry_sha256"] = common.self_hash(
            telemetry, "tar_telemetry_sha256"
        )
        return _TarBatch(
            shard=shard,
            results=tuple(results),
            telemetry=telemetry,
            archive_bytes=opened.st_size,
            selected_bytes=selected_bytes,
        )
    except common.ProductionError:
        raise
    except (OSError, ValueError) as exc:
        raise common.ProductionError(
            f"fatal tar sequential scan error shard={shard:05d}"
        ) from exc
    finally:
        if buffered is not None:
            try:
                buffered.close()
            except BaseException:
                pass
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException:
                pass


def iter_tar_sequential_reads(
    items: Iterable[ItemLike],
    *,
    worker_mapping: dict[str, Any],
    worker_id: int,
    workers: int,
    shard_owners: Mapping[int, int],
    source_prefix: str,
    tar_root: Path,
    row_base: int,
    gpic_rows: int,
    expected_rows: int,
    attempts: int,
    tar_workers: int = 1,
    max_pending_tars: int | None = None,
    read_buffer_bytes: int = DEFAULT_READ_BUFFER_BYTES,
    io_block_bytes: int = DEFAULT_IO_BLOCK_BYTES,
    max_selected_bytes: int,
    max_archive_bytes: int,
    max_selected_bytes_per_tar: int = 1_000_000_000,
    telemetry_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[TarSourceRead]:
    """Bounded concurrent reads, ordered by shard then archive member.

    Each future owns one complete tar and at most
    ``max_selected_bytes_per_tar`` memory.  The main thread consumes futures in
    submission (ascending shard) order, so concurrency cannot change blob or
    dense-index semantics.
    """

    if attempts != 1:
        raise common.ProductionError("tar_sequential requires attempts=1 exactly")
    _strict_positive_int(read_buffer_bytes, "read_buffer_bytes")
    _strict_positive_int(io_block_bytes, "io_block_bytes")
    _strict_positive_int(max_selected_bytes, "max_selected_bytes")
    _strict_positive_int(max_archive_bytes, "max_archive_bytes")
    _strict_positive_int(max_selected_bytes_per_tar, "max_selected_bytes_per_tar")
    _strict_positive_int(tar_workers, "tar_workers")
    if max_pending_tars is None:
        max_pending_tars = tar_workers
    _strict_positive_int(max_pending_tars, "max_pending_tars")
    if (
        read_buffer_bytes < (1 << 20)
        or read_buffer_bytes > 256 * (1 << 20)
        or io_block_bytes < TAR_BLOCK_BYTES
        or io_block_bytes > 64 * (1 << 20)
        or io_block_bytes % TAR_BLOCK_BYTES
        or tar_workers > 16
        or max_pending_tars > tar_workers
        or max_selected_bytes_per_tar > max_selected_bytes
    ):
        raise common.ProductionError("tar sequential buffer shape is unsafe")
    validate_worker_mapping(
        worker_mapping,
        worker_id=worker_id,
        workers=workers,
        source_prefix=source_prefix,
        row_base=row_base,
        gpic_rows=gpic_rows,
        expected_rows=expected_rows,
        shard_owners=shard_owners,
    )
    tar_root = tar_root.resolve(strict=True)
    common.require_regular_dir(tar_root)
    authority_entries = {value["shard_id"]: value for value in worker_mapping["shards"]}
    total_mapping_digest = hashlib.sha256()
    total_rows = 0
    total_selected_bytes = 0
    total_archive_bytes = 0
    observed_shards: list[int] = []
    validated = _validated_targets(
        items,
        source_prefix=source_prefix,
        row_base=row_base,
        gpic_rows=gpic_rows,
        worker_id=worker_id,
        shard_owners=shard_owners,
    )
    groups = iter(_group_targets(validated))
    pending: deque[tuple[int, concurrent.futures.Future[_TarBatch]]] = deque()
    scheduled_archive_bytes = 0

    def submit_next(
        pool: concurrent.futures.ThreadPoolExecutor,
    ) -> bool:
        nonlocal scheduled_archive_bytes
        try:
            shard, group = next(groups)
        except StopIteration:
            return False
        authority = authority_entries.get(shard)
        group_digest = hashlib.sha256()
        for target in group:
            record = _mapping_record(target)
            group_digest.update(record)
            total_mapping_digest.update(record)
        if (
            authority is None
            or authority.get("rows") != len(group)
            or authority.get("mapping_sha256") != group_digest.hexdigest()
        ):
            raise common.ProductionError(
                f"tar reader/structural authority drift shard={shard:05d}"
            )
        observed_shards.append(shard)
        tar_path = tar_root / expected_tar_name(shard)
        path_before = common.require_regular_file(tar_path)
        if path_before.st_size <= 0:
            raise common.ProductionError("empty GPIC tar archive")
        if scheduled_archive_bytes + path_before.st_size > max_archive_bytes:
            raise common.ProductionError("worker source archive-byte cap exceeded")
        scheduled_archive_bytes += path_before.st_size
        pending.append(
            (
                shard,
                pool.submit(
                    _read_one_tar_batch,
                    shard=shard,
                    group=tuple(group),
                    authority=authority,
                    tar_path=tar_path,
                    scheduled_identity=_file_identity(path_before),
                    read_buffer_bytes=read_buffer_bytes,
                    io_block_bytes=io_block_bytes,
                    max_selected_bytes_per_tar=max_selected_bytes_per_tar,
                    worker_id=worker_id,
                    workers=workers,
                ),
            )
        )
        return True

    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=tar_workers, thread_name_prefix="gpic-tar-sequential"
    )
    try:
        while len(pending) < max_pending_tars and submit_next(pool):
            pass
        while pending:
            expected_shard, future = pending.popleft()
            batch = future.result()
            if batch.shard != expected_shard:
                raise common.ProductionError("ordered tar future identity drift")
            if total_selected_bytes + batch.selected_bytes > max_selected_bytes:
                raise common.ProductionError("worker selected payload-byte cap exceeded")
            total_rows += len(batch.results)
            total_selected_bytes += batch.selected_bytes
            total_archive_bytes += batch.archive_bytes
            for result in batch.results:
                yield result
            if telemetry_callback is not None:
                telemetry_callback(batch.telemetry)
            while len(pending) < max_pending_tars and submit_next(pool):
                pass
    finally:
        for _, future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    expected_shards = [entry["shard_id"] for entry in worker_mapping["shards"]]
    if (
        observed_shards != expected_shards
        or total_rows != expected_rows
        or total_mapping_digest.hexdigest() != worker_mapping["mapping_sha256"]
        or total_selected_bytes > max_selected_bytes
        or total_archive_bytes > max_archive_bytes
    ):
        raise common.ProductionError("tar sequential worker closure failed")
