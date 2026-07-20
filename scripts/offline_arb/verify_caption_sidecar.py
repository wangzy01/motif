#!/usr/bin/env python3
"""Gate an OFFLINE_ARB typed-caption sidecar against its frozen contract.

The builder, runtime reader, and this validator share only the immutable wire
contract and semantic-digest codec.  Header/layout/file/footer/row/source checks
remain independently implemented here.  Every selected sidecar row is decoded;
comparison with original caption sources is deliberately typed and sampled,
not an independent full-source digest.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import hashlib
import json
import random
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.offline_arb.caption_sidecar_contract import (
    ArrowIpcSemanticDigester,
    BUILDER_BUILD_ID,
    CAPTION_COLUMNS,
    CAPTION_SCHEMAS,
    CAPTION_SCHEMA_CONTRACT_SHA256,
    DOMAINS,
    FILE_ROWS,
    ROW_GROUP_ROWS,
    ROW_ORDER,
    SCHEMA_VERSION,
    SEMANTIC_DIGEST_ALGORITHM,
    canonical_json_sha256,
    canonical_schema,
    header_self_hash,
)


HEADER_NAME = "header.json"
SUCCESS_NAME = "_SUCCESS"
BASE = Path("/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop")
DEFAULT_CATALOG_ROOT = BASE / "offline_arb_catalog_v1"
DEFAULT_SIDECAR_ROOT = BASE / "offline_arb_caption_sidecar_v1"
# Filled only from the target JCC image via --print-known-answer, then frozen.
# The normal --self-test intentionally fails closed until this becomes 64 hex.
KNOWN_ANSWER_EXPECTED_SHA256 = (
    "9bd23247c702ea0b2f3c104059cb64cbb9cbf15fc8685b67b09b394b1c9dc72d"
)
HEADER_KEYS = {
    "schema_version",
    "semantic_digest_algorithm",
    "builder_build_id",
    "caption_schema_contract_sha256",
    "catalog_hash",
    "catalog_header_sha256",
    "row_order",
    "file_rows",
    "row_group_rows",
    "domain_order",
    "total_rows",
    "diagnostic_run",
    "diagnostic_rows_per_domain",
    "catalog_files",
    "domains",
    "generated_at",
    "header_sha256",
}
DOMAIN_KEYS = {
    "row_base",
    "row_count",
    "columns",
    "source_semantics_sha256",
    "decoded_sidecar_semantics_sha256",
    "files",
}
FILE_KEYS = {
    "name",
    "row_start",
    "row_stop",
    "rows",
    "bytes",
    "sha256",
    "row_groups",
    "schema",
}
CATALOG_FILE_KEYS = {"name", "row_base", "rows", "bytes", "sha256", "schema"}


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def regular_metadata(path: Path) -> Any:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"artifact is not a regular file: {path}")
    return metadata


def read_object(path: Path) -> dict[str, Any]:
    regular_metadata(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_descriptor(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


@dataclass(frozen=True)
class SidecarFile:
    domain: str
    index: int
    name: str
    path: Path
    row_start: int
    row_stop: int
    rows: int
    size_bytes: int
    sha256: str
    row_groups: int
    schema: str


@dataclass(frozen=True)
class CatalogDomain:
    name: str
    row_base: int
    row_count: int
    path: Path
    size_bytes: int
    sha256: str
    schema: str


@dataclass(frozen=True)
class DomainSpec:
    name: str
    row_base: int
    row_count: int
    columns: tuple[str, ...]
    source_semantics_sha256: str
    decoded_sidecar_semantics_sha256: str
    files: tuple[SidecarFile, ...]
    catalog: CatalogDomain


@dataclass(frozen=True)
class FileFingerprint:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "inode": self.inode,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, raw: Any, label: str) -> "FileFingerprint":
        if not isinstance(raw, dict) or set(raw) != {
            "device",
            "inode",
            "size_bytes",
            "mtime_ns",
            "ctime_ns",
            "sha256",
        }:
            raise ValueError(f"malformed fingerprint: {label}")
        values = {
            key: strict_int(raw.get(key), f"{label}.{key}")
            for key in ("device", "inode", "size_bytes", "mtime_ns", "ctime_ns")
        }
        checksum = raw.get("sha256")
        if not is_sha256(checksum):
            raise ValueError(f"malformed fingerprint SHA-256: {label}")
        return cls(**values, sha256=checksum)


def expected_sidecar_schema(domain: str) -> pa.Schema:
    return pa.schema(
        [
            pa.field("row_id", pa.int64(), nullable=False),
            *list(CAPTION_SCHEMAS[domain]),
        ]
    )


def metadata_identity(metadata: Any) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def metadata_identity_dict(metadata: Any) -> dict[str, int]:
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "size_bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
    }


def validate_root(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} is not a real directory: {path}")


def catalog_domains(
    catalog_root: Path, catalog_header: dict[str, Any]
) -> dict[str, CatalogDomain]:
    if catalog_header.get("schema_version") != "offline_arb_catalog_v1":
        raise ValueError("unsupported catalog schema")
    order = catalog_header.get("domain_order")
    files = catalog_header.get("files")
    if order != list(DOMAINS):
        raise ValueError(
            f"catalog domain order drift: {order!r} != {list(DOMAINS)!r}"
        )
    if not isinstance(files, dict) or set(files) != set(DOMAINS):
        raise ValueError("catalog file domain set is malformed")

    cursor = 0
    result: dict[str, CatalogDomain] = {}
    for domain in DOMAINS:
        entry = files[domain]
        if not isinstance(entry, dict):
            raise ValueError(f"catalog files.{domain} must be an object")
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
        name = entry.get("name")
        if (
            not isinstance(name, str)
            or not name.endswith(".parquet")
            or Path(name).name != name
        ):
            raise ValueError(f"catalog files.{domain}.name is malformed: {name!r}")
        result[domain] = CatalogDomain(
            name=domain,
            row_base=cursor,
            row_count=rows,
            path=catalog_root / name,
            size_bytes=size_bytes,
            sha256=checksum,
            schema=schema,
        )
        cursor += rows
    if catalog_header.get("row_count") != cursor:
        raise ValueError(
            f"catalog row_count mismatch: {catalog_header.get('row_count')!r} != {cursor}"
        )
    return result


def parse_domain(
    root: Path,
    domain: str,
    raw: Any,
    catalog: CatalogDomain,
) -> DomainSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"sidecar domains.{domain} must be an object")
    if set(raw) != DOMAIN_KEYS:
        raise ValueError(
            f"sidecar domains.{domain} key set drift: "
            f"missing={sorted(DOMAIN_KEYS-set(raw))}, "
            f"unexpected={sorted(set(raw)-DOMAIN_KEYS)}"
        )
    row_base = strict_int(raw.get("row_base"), f"domains.{domain}.row_base")
    row_count = strict_int(
        raw.get("row_count"), f"domains.{domain}.row_count", minimum=1
    )
    if (row_base, row_count) != (catalog.row_base, catalog.row_count):
        raise ValueError(
            f"sidecar/catalog row range mismatch for {domain}: "
            f"sidecar=[{row_base},{row_base + row_count}), "
            f"catalog=[{catalog.row_base},{catalog.row_base + catalog.row_count})"
        )

    columns = raw.get("columns")
    expected_columns = CAPTION_COLUMNS[domain]
    if not isinstance(columns, list) or tuple(columns) != expected_columns:
        raise ValueError(
            f"sidecar column contract mismatch for {domain}: "
            f"{columns!r} != {list(expected_columns)!r}"
        )
    if len(set(columns)) != len(columns):
        raise ValueError(f"sidecar columns contain duplicates for {domain}")

    source_digest = raw.get("source_semantics_sha256")
    decoded_digest = raw.get("decoded_sidecar_semantics_sha256")
    if not is_sha256(source_digest) or not is_sha256(decoded_digest):
        raise ValueError(f"sidecar semantic digest is malformed for {domain}")
    if source_digest != decoded_digest:
        raise ValueError(
            f"source_semantics_sha256 != decoded_sidecar_semantics_sha256 for {domain}"
        )

    raw_files = raw.get("files")
    expected_file_count = (row_count + FILE_ROWS - 1) // FILE_ROWS
    if not isinstance(raw_files, list) or len(raw_files) != expected_file_count:
        raise ValueError(
            f"sidecar file count mismatch for {domain}: "
            f"{len(raw_files) if isinstance(raw_files, list) else raw_files!r} "
            f"!= {expected_file_count}"
        )

    files: list[SidecarFile] = []
    for index, entry in enumerate(raw_files):
        if not isinstance(entry, dict):
            raise ValueError(f"sidecar file entry is malformed: {domain}/{index}")
        if set(entry) != FILE_KEYS:
            raise ValueError(
                f"sidecar file key set drift {domain}/{index}: "
                f"missing={sorted(FILE_KEYS-set(entry))}, "
                f"unexpected={sorted(set(entry)-FILE_KEYS)}"
            )
        expected_name = f"{domain}/part-{index:05d}.parquet"
        if entry.get("name") != expected_name:
            raise ValueError(
                f"sidecar ordered file name mismatch: {entry.get('name')!r} "
                f"!= {expected_name!r}"
            )
        relative = Path(expected_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe sidecar file name: {expected_name!r}")
        path = root / relative
        if path.parent.parent != root or path.parent.name != domain:
            raise ValueError(f"unsafe sidecar file path: {path}")

        row_start = row_base + index * FILE_ROWS
        row_stop = min(row_base + row_count, row_start + FILE_ROWS)
        rows = row_stop - row_start
        declared = (
            strict_int(entry.get("row_start"), f"{expected_name}.row_start"),
            strict_int(entry.get("row_stop"), f"{expected_name}.row_stop", minimum=1),
            strict_int(entry.get("rows"), f"{expected_name}.rows", minimum=1),
        )
        if declared != (row_start, row_stop, rows):
            raise ValueError(
                f"sidecar declared row range mismatch for {expected_name}: "
                f"{declared!r} != {(row_start, row_stop, rows)!r}"
            )
        size_bytes = strict_int(
            entry.get("bytes"), f"{expected_name}.bytes", minimum=1
        )
        checksum = entry.get("sha256")
        if not is_sha256(checksum):
            raise ValueError(f"sidecar SHA-256 is malformed: {expected_name}")
        row_groups = (rows + ROW_GROUP_ROWS - 1) // ROW_GROUP_ROWS
        if entry.get("row_groups") != row_groups:
            raise ValueError(f"sidecar row-group count mismatch: {expected_name}")
        schema = entry.get("schema")
        if not isinstance(schema, str) or not schema:
            raise ValueError(f"sidecar declared schema is missing: {expected_name}")
        files.append(
            SidecarFile(
                domain=domain,
                index=index,
                name=expected_name,
                path=path,
                row_start=row_start,
                row_stop=row_stop,
                rows=rows,
                size_bytes=size_bytes,
                sha256=checksum,
                row_groups=row_groups,
                schema=schema,
            )
        )

    return DomainSpec(
        name=domain,
        row_base=row_base,
        row_count=row_count,
        columns=expected_columns,
        source_semantics_sha256=source_digest,
        decoded_sidecar_semantics_sha256=decoded_digest,
        files=tuple(files),
        catalog=catalog,
    )


def exact_sidecar_file_set(root: Path, domains: dict[str, DomainSpec]) -> None:
    expected = {HEADER_NAME, SUCCESS_NAME}
    expected.update(file.name for spec in domains.values() for file in spec.files)
    live: set[str] = set()
    live_directories: set[str] = set()
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            live_directories.add(path.relative_to(root).as_posix())
            continue
        relative = path.relative_to(root).as_posix()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"non-regular entry in sidecar root: {relative}")
        live.add(relative)
    if live != expected:
        missing = sorted(expected - live)
        added = sorted(live - expected)
        raise ValueError(
            f"sidecar exact file-set mismatch: missing={missing[:20]}, "
            f"unexpected={added[:20]}"
        )
    if live_directories != set(DOMAINS):
        raise ValueError(
            f"sidecar exact directory-set mismatch: "
            f"missing={sorted(set(DOMAINS)-live_directories)}, "
            f"unexpected={sorted(live_directories-set(DOMAINS))}"
        )


def validate_header(
    root: Path,
    catalog_root: Path,
) -> tuple[dict[str, Any], dict[str, DomainSpec]]:
    validate_root(root, "sidecar root")
    validate_root(catalog_root, "catalog root")
    header_path = root / HEADER_NAME
    success_path = root / SUCCESS_NAME
    header = read_object(header_path)
    if set(header) != HEADER_KEYS:
        raise ValueError(
            f"sidecar header key set drift: missing={sorted(HEADER_KEYS-set(header))}, "
            f"unexpected={sorted(set(header)-HEADER_KEYS)}"
        )
    if header.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported sidecar schema: {header.get('schema_version')!r}")
    if header.get("row_order") != ROW_ORDER:
        raise ValueError(f"unsupported sidecar row order: {header.get('row_order')!r}")
    if header.get("file_rows") != FILE_ROWS:
        raise ValueError(f"sidecar file_rows must be {FILE_ROWS}")
    if header.get("row_group_rows") != ROW_GROUP_ROWS:
        raise ValueError(f"sidecar row_group_rows must be {ROW_GROUP_ROWS}")
    if header.get("semantic_digest_algorithm") != SEMANTIC_DIGEST_ALGORITHM:
        raise ValueError(
            f"unsupported semantic digest: {header.get('semantic_digest_algorithm')!r}"
        )
    if header.get("builder_build_id") != BUILDER_BUILD_ID:
        raise ValueError(
            f"unsupported sidecar builder: {header.get('builder_build_id')!r}"
        )
    if header.get("caption_schema_contract_sha256") != CAPTION_SCHEMA_CONTRACT_SHA256:
        raise ValueError("sidecar caption schema contract SHA-256 mismatch")
    if header.get("domain_order") != list(DOMAINS):
        raise ValueError(f"sidecar domain_order drift: {header.get('domain_order')!r}")
    if header.get("diagnostic_run") is not False:
        raise ValueError("formal sidecar gate refuses diagnostic_run artifacts")
    if header.get("diagnostic_rows_per_domain") is not None:
        raise ValueError("formal sidecar diagnostic_rows_per_domain must be null")
    if not isinstance(header.get("generated_at"), str) or not header["generated_at"]:
        raise ValueError("sidecar generated_at is missing")

    header_sha256 = header.get("header_sha256")
    if not is_sha256(header_sha256) or header_self_hash(header) != header_sha256:
        raise ValueError("sidecar header self-hash mismatch")
    regular_metadata(success_path)
    success_bytes = success_path.read_bytes()
    if success_bytes not in {
        header_sha256.encode("ascii"),
        (header_sha256 + "\n").encode("ascii"),
    }:
        raise ValueError("sidecar _SUCCESS is not the one-line header SHA-256")

    catalog_header_path = catalog_root / HEADER_NAME
    catalog_header = read_object(catalog_header_path)
    live_catalog_header_sha256 = sha256_file(catalog_header_path)
    if header.get("catalog_header_sha256") != live_catalog_header_sha256:
        raise ValueError(
            "sidecar/catalog header SHA-256 mismatch: "
            f"{header.get('catalog_header_sha256')!r} != {live_catalog_header_sha256}"
        )
    catalog_hash = catalog_header.get("catalog_hash")
    if not is_sha256(catalog_hash) or header.get("catalog_hash") != catalog_hash:
        raise ValueError("sidecar/catalog catalog_hash mismatch")
    catalogs = catalog_domains(catalog_root, catalog_header)
    total_rows = sum(spec.row_count for spec in catalogs.values())
    if header.get("total_rows") != total_rows:
        raise ValueError(
            f"sidecar total_rows mismatch: {header.get('total_rows')!r} != {total_rows}"
        )
    raw_catalog_files = header.get("catalog_files")
    if not isinstance(raw_catalog_files, dict) or set(raw_catalog_files) != set(DOMAINS):
        raise ValueError("sidecar catalog_files domain set is malformed")
    for domain in DOMAINS:
        raw = raw_catalog_files[domain]
        if not isinstance(raw, dict) or set(raw) != CATALOG_FILE_KEYS:
            raise ValueError(f"sidecar catalog_files.{domain} key set is malformed")
        catalog = catalogs[domain]
        expected = {
            "name": catalog.path.name,
            "row_base": catalog.row_base,
            "rows": catalog.row_count,
            "bytes": catalog.size_bytes,
            "sha256": catalog.sha256,
            "schema": catalog.schema,
        }
        if raw != expected:
            raise ValueError(
                f"sidecar catalog_files.{domain} binding mismatch: {raw!r} != {expected!r}"
            )

    raw_domains = header.get("domains")
    if not isinstance(raw_domains, dict) or set(raw_domains) != set(DOMAINS):
        raise ValueError(
            f"sidecar domain set mismatch: "
            f"{sorted(raw_domains) if isinstance(raw_domains, dict) else raw_domains!r}"
        )
    domains = {
        domain: parse_domain(root, domain, raw_domains[domain], catalogs[domain])
        for domain in DOMAINS
    }
    exact_sidecar_file_set(root, domains)
    return header, domains


def revalidate_control_snapshot(
    root: Path,
    catalog_root: Path,
    header: dict[str, Any],
    domains: dict[str, DomainSpec],
) -> None:
    if read_object(root / HEADER_NAME) != header:
        raise ValueError("sidecar header changed during validation")
    success = (root / SUCCESS_NAME).read_bytes()
    expected_success = (str(header["header_sha256"]) + "\n").encode("ascii")
    if success != expected_success:
        raise ValueError("sidecar _SUCCESS changed during validation")
    if sha256_file(catalog_root / HEADER_NAME) != header["catalog_header_sha256"]:
        raise ValueError("catalog header changed during sidecar validation")
    exact_sidecar_file_set(root, domains)


def validate_file_hash(item: SidecarFile) -> tuple[str, FileFingerprint]:
    metadata = regular_metadata(item.path)
    if metadata.st_size != item.size_bytes:
        raise ValueError(
            f"sidecar file byte-size mismatch {item.name}: "
            f"{metadata.st_size} != {item.size_bytes}"
        )
    checksum = sha256_file(item.path)
    if checksum != item.sha256:
        raise ValueError(
            f"sidecar file SHA-256 mismatch {item.name}: {checksum} != {item.sha256}"
        )
    after = regular_metadata(item.path)
    before_identity = metadata_identity(metadata)
    after_identity = metadata_identity(after)
    if before_identity != after_identity:
        raise ValueError(f"sidecar file changed while hashing: {item.name}")
    return (
        item.name,
        FileFingerprint(
            device=int(after.st_dev),
            inode=int(after.st_ino),
            size_bytes=int(after.st_size),
            mtime_ns=int(after.st_mtime_ns),
            ctime_ns=int(after.st_ctime_ns),
            sha256=checksum,
        ),
    )


def validate_catalog_file(spec: CatalogDomain) -> tuple[str, FileFingerprint]:
    metadata = regular_metadata(spec.path)
    if metadata.st_size != spec.size_bytes:
        raise ValueError(
            f"catalog file byte-size mismatch {spec.name}: "
            f"{metadata.st_size} != {spec.size_bytes}"
        )
    checksum = sha256_file(spec.path)
    if checksum != spec.sha256:
        raise ValueError(
            f"catalog file SHA-256 mismatch {spec.name}: {checksum} != {spec.sha256}"
        )
    parquet = pq.ParquetFile(spec.path)
    if parquet.metadata.num_rows != spec.row_count:
        raise ValueError(f"catalog footer row count mismatch: {spec.name}")
    if str(parquet.schema_arrow) != spec.schema:
        raise ValueError(f"catalog footer/header schema mismatch: {spec.name}")
    after = regular_metadata(spec.path)
    before_identity = metadata_identity(metadata)
    after_identity = metadata_identity(after)
    if before_identity != after_identity:
        raise ValueError(f"catalog file changed while hashing: {spec.name}")
    return (
        spec.name,
        FileFingerprint(
            device=int(after.st_dev),
            inode=int(after.st_ino),
            size_bytes=int(after.st_size),
            mtime_ns=int(after.st_mtime_ns),
            ctime_ns=int(after.st_ctime_ns),
            sha256=checksum,
        ),
    )


def selected_row_ids(row_base: int, row_count: int, count: int, seed: int) -> set[int]:
    take = min(count, row_count)
    rng = random.Random(seed)
    result = set(rng.sample(range(row_base, row_base + row_count), take))
    result.update((row_base, row_base + row_count // 2, row_base + row_count - 1))
    return result


def first_true_index(mask: pa.Array | pa.ChunkedArray) -> int | None:
    indices = pc.indices_nonzero(pc.fill_null(mask, False))
    return int(indices[0].as_py()) if len(indices) else None


def recursively_contains_unicode(value: Any) -> bool:
    if isinstance(value, str):
        return any(ord(character) > 127 for character in value)
    if isinstance(value, (list, tuple)):
        return any(recursively_contains_unicode(item) for item in value)
    if isinstance(value, dict):
        return any(recursively_contains_unicode(item) for item in value.values())
    return False


def witness_offset(array: pa.ChunkedArray, kind: str) -> int | None:
    if kind == "null":
        return first_true_index(pc.is_null(array))
    if kind == "unicode":
        for index, value in enumerate(array.to_pylist()):
            if recursively_contains_unicode(value):
                return index
        return None
    if kind == "list" and (
        pa.types.is_list(array.type)
        or pa.types.is_large_list(array.type)
        or pa.types.is_fixed_size_list(array.type)
    ):
        for index, value in enumerate(array.to_pylist()):
            if value is not None and len(value) > 0:
                return index
    return None


def capture_row(table: pa.Table, offset: int, schema: pa.Schema) -> dict[str, Any]:
    value = table.select(schema.names).slice(offset, 1).to_pylist()[0]
    return {name: value[name] for name in schema.names}


def validate_schema(domain: str, spec: DomainSpec) -> pa.Schema:
    first = pq.ParquetFile(spec.files[0].path).schema_arrow
    expected = expected_sidecar_schema(domain)
    if first.names != expected.names:
        raise ValueError(
            f"sidecar schema names mismatch for {domain}: "
            f"{first.names!r} != {expected.names!r}"
        )
    normalized = canonical_schema(first)
    if not first.equals(normalized, check_metadata=True):
        raise ValueError(f"sidecar schema contains non-canonical metadata for {domain}")
    if not first.equals(expected, check_metadata=True):
        raise ValueError(
            f"sidecar frozen typed schema mismatch for {domain}:\n{first}\n!=\n{expected}"
        )
    return expected


def fingerprint_matches(path: Path, expected: FileFingerprint) -> bool:
    metadata = regular_metadata(path)
    return metadata_identity(metadata) == (
        expected.device,
        expected.inode,
        expected.size_bytes,
        expected.mtime_ns,
        expected.ctime_ns,
    )


def iter_and_validate_sidecar(
    spec: DomainSpec,
    fingerprints: dict[str, FileFingerprint],
    sample_ids: set[int],
) -> tuple[
    str,
    pa.Schema,
    int,
    int,
    int,
    dict[int, dict[str, Any]],
    dict[str, int],
]:
    schema = validate_schema(spec.name, spec)
    digester = ArrowIpcSemanticDigester(
        domain=spec.name,
        row_base=spec.row_base,
        row_count=spec.row_count,
        schema=schema,
    )
    cursor = spec.row_base
    physical_rows = 0
    row_groups = 0
    max_row_group_rows = 0
    samples: dict[int, dict[str, Any]] = {}
    witnesses: dict[str, int] = {}
    sample_order = sorted(sample_ids)

    for item in spec.files:
        fingerprint = fingerprints[item.name]
        if not fingerprint_matches(item.path, fingerprint):
            raise ValueError(f"sidecar file changed after hashing: {item.name}")
        parquet = pq.ParquetFile(item.path)
        live_schema = parquet.schema_arrow
        if not live_schema.equals(schema, check_metadata=True):
            raise ValueError(f"typed sidecar schema drift: {item.name}")
        if str(live_schema) != item.schema:
            raise ValueError(f"sidecar footer/header schema mismatch: {item.name}")
        if parquet.metadata.num_rows != item.rows:
            raise ValueError(f"sidecar footer row count mismatch: {item.name}")
        if parquet.num_row_groups != item.row_groups:
            raise ValueError(f"sidecar footer row-group count mismatch: {item.name}")

        file_rows = 0
        for row_group in range(parquet.num_row_groups):
            expected_start = item.row_start + row_group * ROW_GROUP_ROWS
            expected_rows = min(ROW_GROUP_ROWS, item.row_stop - expected_start)
            metadata = parquet.metadata.row_group(row_group)
            rows = int(metadata.num_rows)
            if rows != expected_rows or rows <= 0 or rows > ROW_GROUP_ROWS:
                raise ValueError(
                    f"sidecar row-group layout mismatch {item.name}:{row_group}: "
                    f"{rows} != {expected_rows}"
                )
            table = parquet.read_row_group(
                row_group, columns=schema.names, use_threads=False
            )
            if not table.schema.equals(schema, check_metadata=True):
                raise ValueError(f"decoded sidecar schema drift: {item.name}:{row_group}")
            if table.num_rows != rows or table.column("row_id").null_count:
                raise ValueError(f"decoded sidecar row layout mismatch: {item.name}:{row_group}")
            lo = bisect.bisect_left(sample_order, cursor)
            hi = bisect.bisect_left(sample_order, cursor + rows)
            for row_id in sample_order[lo:hi]:
                samples[row_id] = capture_row(table, row_id - cursor, schema)

            # One real-data witness per special value shape is sufficient: every
            # sampled row compares all typed columns.  Do not regex-scan every
            # string column for the entire 384M-row artifact after one witness.
            witnessed_kinds = {
                key.rsplit(":", 1)[1] for key in witnesses
            }
            for kind in ("null", "unicode", "list"):
                if kind in witnessed_kinds:
                    continue
                for field in schema:
                    if field.name == "row_id":
                        continue
                    array = table.column(field.name)
                    key = f"{field.name}:{kind}"
                    offset = witness_offset(array, kind)
                    if offset is not None:
                        witness_row = cursor + offset
                        witnesses[key] = witness_row
                        samples[witness_row] = capture_row(table, offset, schema)
                        break

            try:
                # The shared codec also independently enforces the exact global
                # row_id sequence for every fixed 256-row logical batch.
                digester.update(table)
            except Exception as exc:
                raise ValueError(
                    f"sidecar semantic batch mismatch at {item.name}:{row_group}: {exc}"
                ) from exc
            cursor += rows
            file_rows += rows
            physical_rows += rows
            row_groups += 1
            max_row_group_rows = max(max_row_group_rows, rows)
        if file_rows != item.rows or cursor != item.row_stop:
            raise ValueError(f"sidecar physical/declared row range mismatch: {item.name}")
        if not fingerprint_matches(item.path, fingerprint):
            raise ValueError(f"sidecar file changed while decoding: {item.name}")
        if (item.index + 1) % 32 == 0 or item.index + 1 == len(spec.files):
            print(
                f"[caption-sidecar-validate:{spec.name}] decoded files "
                f"{item.index + 1}/{len(spec.files)} rows={physical_rows}/{spec.row_count}",
                file=sys.stderr,
                flush=True,
            )

    if cursor != spec.row_base + spec.row_count:
        raise ValueError(
            f"sidecar physical rows stop at {cursor}, "
            f"expected {spec.row_base + spec.row_count}"
        )
    missing_samples = sample_ids - samples.keys()
    if missing_samples:
        raise ValueError(
            f"sidecar failed to capture sampled rows: {sorted(missing_samples)[:10]}"
        )
    return (
        digester.finish(),
        schema,
        physical_rows,
        row_groups,
        max_row_group_rows,
        samples,
        witnesses,
    )


def catalog_sample_locators(
    catalog_path: Path,
    domain_base: int,
    domain_count: int,
    sample_ids: Iterable[int],
) -> dict[int, tuple[str, int, int]]:
    regular_metadata(catalog_path)
    parquet = pq.ParquetFile(catalog_path)
    required = (
        "row_id",
        "caption_parquet_path",
        "caption_row_group",
        "caption_row_in_group",
    )
    if any(name not in parquet.schema_arrow.names for name in required):
        raise ValueError(f"catalog locator schema is incomplete: {catalog_path}")
    expected_types = {
        "row_id": pa.int64(),
        "caption_parquet_path": pa.string(),
        "caption_row_group": pa.int32(),
        "caption_row_in_group": pa.int32(),
    }
    wrong_types = {
        name: str(parquet.schema_arrow.field(name).type)
        for name, expected in expected_types.items()
        if parquet.schema_arrow.field(name).type != expected
    }
    if wrong_types:
        raise ValueError(f"catalog locator typed schema drift: {wrong_types}")
    if parquet.metadata.num_rows != domain_count:
        raise ValueError(f"catalog domain row count drift: {catalog_path}")
    ends: list[int] = []
    cursor = 0
    for index in range(parquet.num_row_groups):
        rows = int(parquet.metadata.row_group(index).num_rows)
        if rows <= 0:
            raise ValueError(f"catalog has empty row group: {catalog_path}:{index}")
        cursor += rows
        ends.append(cursor)
    if cursor != domain_count:
        raise ValueError(f"catalog row-group total mismatch: {catalog_path}")

    grouped: dict[int, list[int]] = {}
    for row_id in sorted(sample_ids):
        local = row_id - domain_base
        if local < 0 or local >= domain_count:
            raise ValueError(f"sample row_id outside catalog domain: {row_id}")
        row_group = bisect.bisect_right(ends, local)
        grouped.setdefault(row_group, []).append(row_id)

    result: dict[int, tuple[str, int, int]] = {}
    starts = [0, *ends[:-1]]
    for row_group, row_ids in sorted(grouped.items()):
        table = parquet.read_row_group(
            row_group, columns=list(required), use_threads=False
        )
        for row_id in row_ids:
            offset = row_id - domain_base - starts[row_group]
            row = table.slice(offset, 1).to_pylist()[0]
            if row.get("row_id") != row_id:
                raise ValueError(f"catalog physical row_id mismatch for sample {row_id}")
            path = row.get("caption_parquet_path")
            source_row_group = row.get("caption_row_group")
            source_row_in_group = row.get("caption_row_in_group")
            if (
                not isinstance(path, str)
                or not path
                or type(source_row_group) is not int
                or source_row_group < 0
                or type(source_row_in_group) is not int
                or source_row_in_group < 0
            ):
                raise ValueError(f"catalog source locator is malformed for row_id={row_id}")
            result[row_id] = (path, source_row_group, source_row_in_group)
    return result


def validate_source_physical_schema(
    path: Path,
    live_schema: pa.Schema,
    expected_schema: pa.Schema,
) -> None:
    fields = {field.name: field for field in live_schema}
    incompatible: list[str] = []
    for expected in expected_schema:
        live = fields.get(expected.name)
        if live is None:
            incompatible.append(f"{expected.name}: missing")
        elif live.type == expected.type:
            continue
        elif expected.nullable and pa.types.is_null(live.type):
            continue
        else:
            incompatible.append(
                f"{expected.name}: live={live.type}/{live.nullable}, "
                f"expected={expected.type}/{expected.nullable}"
            )
    if incompatible:
        raise ValueError(
            f"source typed schema mismatch {path}: " + "; ".join(incompatible)
        )


def canonical_source_row_group(
    path: Path,
    parquet: pq.ParquetFile,
    row_group: int,
    expected_schema: pa.Schema,
) -> pa.Table:
    validate_source_physical_schema(path, parquet.schema_arrow, expected_schema)
    if row_group < 0 or row_group >= parquet.num_row_groups:
        raise ValueError(f"source sample row group out of range: {path}:{row_group}")
    table = parquet.read_row_group(
        row_group, columns=expected_schema.names, use_threads=False
    )
    try:
        table = table.cast(expected_schema, safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError) as exc:
        raise ValueError(
            f"cannot safely cast sampled source {path}:{row_group} "
            f"to frozen schema: {exc}"
        ) from exc
    if not table.schema.equals(expected_schema, check_metadata=True):
        raise ValueError(f"sampled source canonical schema drift: {path}:{row_group}")
    return table


def source_sample_values(
    locators: dict[int, tuple[str, int, int]],
    schema: pa.Schema,
    workers: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[int, list[tuple[int, int]]]] = {}
    for row_id, (path, row_group, row_in_group) in locators.items():
        grouped.setdefault(path, {}).setdefault(row_group, []).append(
            (row_id, row_in_group)
        )
    expected_schema = pa.schema(list(schema)[1:])

    def read_one(
        item: tuple[str, dict[int, list[tuple[int, int]]]]
    ) -> tuple[dict[int, dict[str, Any]], str, dict[str, int]]:
        path_text, requests_by_group = item
        path = Path(path_text)
        before = regular_metadata(path)
        parquet = pq.ParquetFile(path)
        output: dict[int, dict[str, Any]] = {}
        for row_group, requests in sorted(requests_by_group.items()):
            table = canonical_source_row_group(
                path, parquet, row_group, expected_schema
            )
            for row_id, within in requests:
                if within < 0 or within >= table.num_rows:
                    raise ValueError(
                        f"source sample row offset out of range: "
                        f"{path}:{row_group}:{within}"
                    )
                values = table.slice(within, 1).to_pylist()[0]
                output[row_id] = {"row_id": row_id, **values}
        after = regular_metadata(path)
        if metadata_identity(before) != metadata_identity(after):
            raise ValueError(f"source file changed while sampling: {path}")
        return output, path_text, metadata_identity_dict(after)

    result: dict[int, dict[str, Any]] = {}
    source_identities: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, max(1, len(grouped))),
        thread_name_prefix="caption-sidecar-source-sample",
    ) as pool:
        for partial, path, identity in pool.map(read_one, sorted(grouped.items())):
            overlap = result.keys() & partial.keys()
            if overlap:
                raise ValueError(f"duplicate source sample rows: {sorted(overlap)}")
            result.update(partial)
            source_identities[path] = identity
    return result, source_identities


def verify_source_samples(
    spec: DomainSpec,
    catalog_fingerprint: FileFingerprint,
    sidecar_samples: dict[int, dict[str, Any]],
    schema: pa.Schema,
    workers: int,
) -> dict[str, Any]:
    if not fingerprint_matches(spec.catalog.path, catalog_fingerprint):
        raise ValueError(f"catalog file changed after hashing: {spec.name}")
    locators = catalog_sample_locators(
        spec.catalog.path,
        spec.row_base,
        spec.row_count,
        sidecar_samples,
    )
    sources, source_identities = source_sample_values(locators, schema, workers)
    if not fingerprint_matches(spec.catalog.path, catalog_fingerprint):
        raise ValueError(f"catalog file changed while sampling: {spec.name}")
    if sources.keys() != sidecar_samples.keys():
        raise ValueError("source/sidecar sampled row set mismatch")
    for row_id in sorted(sidecar_samples):
        if sources[row_id] != sidecar_samples[row_id]:
            raise ValueError(
                f"typed source/sidecar sampled value mismatch row_id={row_id}: "
                f"source={sources[row_id]!r}, sidecar={sidecar_samples[row_id]!r}"
            )
    return {
        "rows": len(sidecar_samples),
        "source_files": len({path for path, _rg, _within in locators.values()}),
        "source_row_groups": len(
            {(path, row_group) for path, row_group, _within in locators.values()}
        ),
        "source_identity_scope": "sampled_files_stat_before_and_after_no_full_sha256",
        "source_identities": source_identities,
        "sampled_row_ids_sha256": canonical_json_sha256(sorted(sidecar_samples)),
    }


def validate_domain(
    spec: DomainSpec,
    fingerprints: dict[str, FileFingerprint],
    catalog_fingerprint: FileFingerprint,
    *,
    sample_rows: int,
    seed: int,
    sample_workers: int,
) -> dict[str, Any]:
    requested_samples = selected_row_ids(
        spec.row_base, spec.row_count, sample_rows, seed
    )
    (
        decoded_digest,
        schema,
        physical_rows,
        row_groups,
        max_row_group_rows,
        samples,
        witnesses,
    ) = iter_and_validate_sidecar(spec, fingerprints, requested_samples)
    if decoded_digest != spec.decoded_sidecar_semantics_sha256:
        raise ValueError(
            f"decoded sidecar digest mismatch for {spec.name}: "
            f"live={decoded_digest}, header={spec.decoded_sidecar_semantics_sha256}"
        )
    if decoded_digest != spec.source_semantics_sha256:
        raise ValueError(
            f"decoded sidecar digest != source semantics digest for {spec.name}"
        )
    sample_report = verify_source_samples(
        spec,
        catalog_fingerprint,
        samples,
        schema,
        sample_workers,
    )
    sample_report.update(
        {
            "fixed_seed": seed,
            "random_rows_requested": len(requested_samples),
            "witnesses": dict(sorted(witnesses.items())),
            "witness_kinds": sorted({key.rsplit(":", 1)[1] for key in witnesses}),
        }
    )
    return {
        "status": "PASS",
        "row_base": spec.row_base,
        "row_count": spec.row_count,
        "files": len(spec.files),
        "bytes": sum(item.size_bytes for item in spec.files),
        "rows": physical_rows,
        "row_groups": row_groups,
        "max_row_group_rows": max_row_group_rows,
        "catalog_file_sha256": spec.catalog.sha256,
        "semantic_digest_sha256": decoded_digest,
        "typed_schema": schema_descriptor(schema),
        "source_sample": sample_report,
    }


def evidence_context(args: argparse.Namespace) -> dict[str, Any]:
    if not is_sha256(args.run_nonce):
        raise ValueError("--run-nonce must be a lowercase 64-hex value")
    if (
        not isinstance(args.jobtag, str)
        or not args.jobtag
        or "/" in args.jobtag
        or ".." in args.jobtag
    ):
        raise ValueError("--jobtag is malformed")
    if type(args.worker_index) is not int or args.worker_index < 0:
        raise ValueError("--worker-index must be a non-negative integer")
    if not isinstance(args.evidence_host, str) or not args.evidence_host:
        raise ValueError("--evidence-host is required")
    return {
        "run_nonce": args.run_nonce,
        "jobtag": args.jobtag,
        "worker_index": args.worker_index,
        "host": args.evidence_host,
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    context = evidence_context(args)
    root = args.root.expanduser()
    catalog_root = args.catalog_root.expanduser()
    header, all_domains = validate_header(root, catalog_root)
    print(
        f"[caption-sidecar-validate] header/file-set PASS "
        f"header_sha256={header['header_sha256']}",
        file=sys.stderr,
        flush=True,
    )
    selected_domains = tuple(args.domain or DOMAINS)
    if len(set(selected_domains)) != len(selected_domains):
        raise ValueError(f"duplicate --domain values: {selected_domains!r}")
    selected_files = [
        item for domain in selected_domains for item in all_domains[domain].files
    ]
    hashed: dict[str, FileFingerprint] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.hash_workers, max(1, len(selected_files))),
        thread_name_prefix="caption-sidecar-sha256",
    ) as pool:
        futures = {
            pool.submit(validate_file_hash, item): item for item in selected_files
        }
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name, fingerprint = future.result()
            hashed[name] = fingerprint
            if done % 128 == 0 or done == len(futures):
                print(
                    f"[caption-sidecar-validate] hashed sidecar files "
                    f"{done}/{len(futures)}",
                    file=sys.stderr,
                    flush=True,
                )
    catalog_hashed: dict[str, FileFingerprint] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.hash_workers, len(selected_domains)),
        thread_name_prefix="caption-catalog-sha256",
    ) as pool:
        futures = {
            pool.submit(validate_catalog_file, all_domains[domain].catalog): domain
            for domain in selected_domains
        }
        for future in concurrent.futures.as_completed(futures):
            name, fingerprint = future.result()
            catalog_hashed[name] = fingerprint
            print(
                f"[caption-sidecar-validate] catalog identity PASS domain={name}",
                file=sys.stderr,
                flush=True,
            )

    reports: dict[str, Any] = {}
    for domain in selected_domains:
        print(
            f"[caption-sidecar-validate:{domain}] full decode start",
            file=sys.stderr,
            flush=True,
        )
        domain_seed = args.seed + DOMAINS.index(domain) * 1_000_003
        reports[domain] = validate_domain(
            all_domains[domain],
            hashed,
            catalog_hashed[domain],
            sample_rows=args.sample_rows,
            seed=domain_seed,
            sample_workers=args.sample_workers,
        )
        print(
            f"[caption-sidecar-validate:{domain}] full decode/source sample PASS",
            file=sys.stderr,
            flush=True,
        )
    witness_kinds = sorted(
        {
            kind
            for report in reports.values()
            for kind in report["source_sample"]["witness_kinds"]
        }
    )
    revalidate_control_snapshot(root, catalog_root, header, all_domains)
    return {
        "status": "PASS",
        "schema_version": "offline_arb_caption_sidecar_validation_v1",
        "evidence_scope": (
            "full_sidecar_structural_byte_and_digest_self_consistency_plus_"
            "deterministic_sampled_typed_source_equality_not_full_source_digest"
        ),
        "run_context": context,
        "sidecar_schema_version": SCHEMA_VERSION,
        "builder_build_id": BUILDER_BUILD_ID,
        "caption_schema_contract_sha256": CAPTION_SCHEMA_CONTRACT_SHA256,
        "sidecar_root": str(root),
        "header_sha256": header["header_sha256"],
        "catalog_hash": header["catalog_hash"],
        "catalog_header_sha256": header["catalog_header_sha256"],
        "semantic_digest_algorithm": SEMANTIC_DIGEST_ALGORITHM,
        "validated_domains": list(selected_domains),
        "files": sum(report["files"] for report in reports.values()),
        "bytes": sum(report["bytes"] for report in reports.values()),
        "rows": sum(report["rows"] for report in reports.values()),
        "witness_kinds": witness_kinds,
        "sidecar_fingerprints": {
            name: fingerprint.as_dict() for name, fingerprint in sorted(hashed.items())
        },
        "catalog_fingerprints": {
            name: fingerprint.as_dict()
            for name, fingerprint in sorted(catalog_hashed.items())
        },
        "domains": reports,
    }


def load_validation_result(path: Path) -> dict[str, Any]:
    result = read_object(path)
    if result.get("status") != "PASS" or result.get("schema_version") != (
        "offline_arb_caption_sidecar_validation_v1"
    ):
        raise ValueError(f"not a PASS caption-sidecar validation result: {path}")
    return result


def validate_result_context(
    result: dict[str, Any], expected: dict[str, Any], path: Path
) -> None:
    if result.get("run_context") != expected:
        raise ValueError(
            f"validation result run context mismatch {path}: "
            f"{result.get('run_context')!r} != {expected!r}"
        )


def result_fingerprints(
    result: dict[str, Any],
    field: str,
    expected_names: set[str],
) -> dict[str, FileFingerprint]:
    raw = result.get(field)
    if not isinstance(raw, dict) or set(raw) != expected_names:
        raise ValueError(
            f"validation result {field} set mismatch: "
            f"missing={sorted(expected_names-set(raw) if isinstance(raw, dict) else expected_names)[:20]}, "
            f"unexpected={sorted(set(raw)-expected_names)[:20] if isinstance(raw, dict) else []}"
        )
    return {
        name: FileFingerprint.from_dict(value, f"{field}.{name}")
        for name, value in raw.items()
    }


def run_result_revalidation(args: argparse.Namespace) -> dict[str, Any]:
    context = evidence_context(args)
    result_path = args.revalidate_result.expanduser()
    result = load_validation_result(result_path)
    validate_result_context(result, context, result_path)
    root = args.root.expanduser()
    catalog_root = args.catalog_root.expanduser()
    header, domains = validate_header(root, catalog_root)
    if result.get("header_sha256") != header["header_sha256"]:
        raise ValueError("validation result/header identity mismatch")
    selected = result.get("validated_domains")
    if (
        not isinstance(selected, list)
        or not selected
        or any(domain not in DOMAINS for domain in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("validation result selected domain set is malformed")
    expected_files = {
        item.name for domain in selected for item in domains[domain].files
    }
    fingerprints = result_fingerprints(
        result, "sidecar_fingerprints", expected_files
    )
    for name, fingerprint in fingerprints.items():
        if not fingerprint_matches(root / name, fingerprint):
            raise ValueError(f"post-barrier sidecar fingerprint drift: {name}")
    catalog_fingerprints = result_fingerprints(
        result, "catalog_fingerprints", set(selected)
    )
    for domain, fingerprint in catalog_fingerprints.items():
        if not fingerprint_matches(domains[domain].catalog.path, fingerprint):
            raise ValueError(f"post-barrier catalog fingerprint drift: {domain}")
    revalidate_control_snapshot(root, catalog_root, header, domains)
    output = dict(result)
    output["post_barrier_revalidation"] = {
        "status": "PASS",
        "run_nonce": context["run_nonce"],
        "worker_index": context["worker_index"],
        "sidecar_files": len(fingerprints),
        "catalog_files": len(catalog_fingerprints),
        "identity_fields": [
            "device",
            "inode",
            "size_bytes",
            "mtime_ns",
            "ctime_ns",
        ],
        "control_snapshot_revalidated": True,
    }
    return output


def run_full_root_closure(args: argparse.Namespace) -> dict[str, Any]:
    context = evidence_context(args)
    if context["worker_index"] != 0:
        raise ValueError("full-root closure must run on worker index 0")
    paths = [path.expanduser() for path in args.close_results]
    results = [load_validation_result(path) for path in paths]
    worker_indices: set[int] = set()
    for path, result in zip(paths, results):
        result_context = result.get("run_context")
        if not isinstance(result_context, dict):
            raise ValueError(f"missing result run context: {path}")
        if (
            result_context.get("run_nonce") != context["run_nonce"]
            or result_context.get("jobtag") != context["jobtag"]
        ):
            raise ValueError(f"full-root closure run identity mismatch: {path}")
        worker_index = result_context.get("worker_index")
        if type(worker_index) is not int:
            raise ValueError(f"malformed worker index: {path}")
        worker_indices.add(worker_index)
        post = result.get("post_barrier_revalidation")
        if (
            not isinstance(post, dict)
            or post.get("status") != "PASS"
            or post.get("run_nonce") != context["run_nonce"]
        ):
            raise ValueError(f"missing post-barrier PASS evidence: {path}")
    if worker_indices != {0, 1}:
        raise ValueError(f"full-root closure worker set mismatch: {worker_indices}")

    root = args.root.expanduser()
    catalog_root = args.catalog_root.expanduser()
    header, domains = validate_header(root, catalog_root)
    for path, result in zip(paths, results):
        if (
            result.get("header_sha256") != header["header_sha256"]
            or result.get("catalog_hash") != header["catalog_hash"]
            or result.get("catalog_header_sha256")
            != header["catalog_header_sha256"]
            or result.get("builder_build_id") != BUILDER_BUILD_ID
            or result.get("caption_schema_contract_sha256")
            != CAPTION_SCHEMA_CONTRACT_SHA256
        ):
            raise ValueError(f"full-root closure artifact identity mismatch: {path}")
    selected = [
        domain for result in results for domain in result.get("validated_domains", [])
    ]
    if len(selected) != len(set(selected)) or set(selected) != set(DOMAINS):
        raise ValueError(f"full-root closure five-domain coverage mismatch: {selected}")

    all_fingerprints: dict[str, FileFingerprint] = {}
    catalog_fingerprints: dict[str, FileFingerprint] = {}
    for result in results:
        result_domains = result["validated_domains"]
        expected_files = {
            item.name for domain in result_domains for item in domains[domain].files
        }
        partial = result_fingerprints(
            result, "sidecar_fingerprints", expected_files
        )
        overlap = set(all_fingerprints) & set(partial)
        if overlap:
            raise ValueError(f"duplicate closure sidecar fingerprints: {sorted(overlap)}")
        all_fingerprints.update(partial)
        partial_catalog = result_fingerprints(
            result, "catalog_fingerprints", set(result_domains)
        )
        catalog_overlap = set(catalog_fingerprints) & set(partial_catalog)
        if catalog_overlap:
            raise ValueError(
                f"duplicate closure catalog fingerprints: {sorted(catalog_overlap)}"
            )
        catalog_fingerprints.update(partial_catalog)

    file_specs = {
        item.name: item for domain in DOMAINS for item in domains[domain].files
    }
    if set(all_fingerprints) != set(file_specs):
        raise ValueError("full-root closure fingerprint set is incomplete")
    if set(catalog_fingerprints) != set(DOMAINS):
        raise ValueError("full-root closure catalog fingerprint set is incomplete")

    hashes: dict[str, str] = {}
    closure_fingerprints: dict[str, FileFingerprint] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.hash_workers, len(file_specs)),
        thread_name_prefix="caption-sidecar-full-root-closure",
    ) as pool:
        futures = {
            pool.submit(validate_file_hash, item): name
            for name, item in file_specs.items()
        }
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name, live_fingerprint = future.result()
            hashes[name] = live_fingerprint.sha256
            closure_fingerprints[name] = live_fingerprint
            if done % 128 == 0 or done == len(futures):
                print(
                    f"[caption-sidecar-closure] hashed files {done}/{len(futures)}",
                    file=sys.stderr,
                    flush=True,
                )

    # Worker fingerprints were intentionally rechecked on their originating
    # nodes after the barrier.  Some shared filesystems expose node-local st_dev
    # or inode values, so worker 0 establishes a fresh all-root identity epoch
    # around its own full hash instead of comparing cross-node stat numbers.
    catalog_hashes: dict[str, str] = {}
    closure_catalog_fingerprints: dict[str, FileFingerprint] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.hash_workers, len(DOMAINS)),
        thread_name_prefix="caption-catalog-full-root-closure",
    ) as pool:
        futures = {
            pool.submit(validate_catalog_file, domains[domain].catalog): domain
            for domain in DOMAINS
        }
        for future in concurrent.futures.as_completed(futures):
            domain, fingerprint = future.result()
            catalog_hashes[domain] = fingerprint.sha256
            closure_catalog_fingerprints[domain] = fingerprint

    for name, fingerprint in closure_fingerprints.items():
        if not fingerprint_matches(file_specs[name].path, fingerprint):
            raise ValueError(f"full-root post-hash fingerprint drift: {name}")
    for domain, fingerprint in closure_catalog_fingerprints.items():
        if not fingerprint_matches(domains[domain].catalog.path, fingerprint):
            raise ValueError(f"full-root post-hash catalog drift: {domain}")
    revalidate_control_snapshot(root, catalog_root, header, domains)
    return {
        "status": "PASS",
        "schema_version": "offline_arb_caption_sidecar_full_root_closure_v1",
        "run_context": context,
        "header_sha256": header["header_sha256"],
        "catalog_hash": header["catalog_hash"],
        "builder_build_id": BUILDER_BUILD_ID,
        "caption_schema_contract_sha256": CAPTION_SCHEMA_CONTRACT_SHA256,
        "worker_indices": sorted(worker_indices),
        "validated_domains": list(DOMAINS),
        "files": len(hashes),
        "bytes": sum(item.size_bytes for item in file_specs.values()),
        "file_sha256_manifest": canonical_json_sha256(dict(sorted(hashes.items()))),
        "catalog_sha256_manifest": canonical_json_sha256(
            dict(sorted(catalog_hashes.items()))
        ),
        "worker_fingerprint_manifests": {
            str(result["run_context"]["worker_index"]): canonical_json_sha256(
                result["sidecar_fingerprints"]
            )
            for result in results
        },
        "identity_fields": [
            "device",
            "inode",
            "size_bytes",
            "mtime_ns",
            "ctime_ns",
        ],
        "worker_result_sha256": {
            str(result["run_context"]["worker_index"]): sha256_file(path)
            for path, result in zip(paths, results)
        },
        "control_snapshot_revalidated_after_full_hash": True,
    }


def digest_fixture(table: pa.Table, *, row_base: int) -> str:
    digester = ArrowIpcSemanticDigester(
        domain="anime",
        row_base=row_base,
        row_count=table.num_rows,
        schema=table.schema,
    )
    for offset in range(0, table.num_rows, ROW_GROUP_ROWS):
        digester.update(table.slice(offset, min(ROW_GROUP_ROWS, table.num_rows - offset)))
    return digester.finish()


def known_answer_digest() -> str:
    row_base = 41
    schema = expected_sidecar_schema("gpic")
    table = pa.Table.from_pydict(
        {
            "row_id": [41, 42, 43, 44],
            "caption": ["alpha", None, "雪-café", ""],
        },
        schema=schema,
    )
    digester = ArrowIpcSemanticDigester(
        domain="gpic",
        row_base=row_base,
        row_count=table.num_rows,
        schema=schema,
    )
    digester.update(table)
    return digester.finish()


def run_self_test() -> dict[str, Any]:
    row_base = 17
    rows = 777
    schema = pa.schema(
        [
            pa.field("row_id", pa.int64(), nullable=False),
            pa.field("caption", pa.string(), nullable=True),
            pa.field("tags", pa.list_(pa.string()), nullable=True),
        ]
    )
    table = pa.Table.from_pydict(
        {
            "row_id": list(range(row_base, row_base + rows)),
            "caption": [
                None
                if index % 19 == 0
                else ("雪とcafé" if index == 257 else f"row-{index}")
                for index in range(rows)
            ],
            "tags": [
                None if index % 23 == 0 else [f"tag-{index % 7}", "猫"]
                for index in range(rows)
            ],
        },
        schema=schema,
    )
    one = digest_fixture(table, row_base=row_base)
    chunked = pa.concat_tables(
        [table.slice(0, 3), table.slice(3, 511), table.slice(514)]
    )
    many = digest_fixture(chunked, row_base=row_base)
    if one != many:
        raise AssertionError("semantic digest depends on Arrow chunk boundaries")
    mutated_caption = pa.array(
        ["mutated", *table.column("caption").slice(1).to_pylist()],
        type=pa.string(),
    )
    mutated = table.set_column(1, schema.field("caption"), mutated_caption)
    changed = digest_fixture(mutated, row_base=row_base)
    if changed == one:
        raise AssertionError("semantic digest failed to detect a typed value mutation")
    if witness_offset(table.column("tags"), "list") is None:
        raise AssertionError("non-empty list witness was not detected")
    if witness_offset(table.column("tags"), "unicode") is None:
        raise AssertionError("recursive list unicode witness was not detected")

    expected_source = pa.schema(
        [
            pa.field("caption", pa.string(), nullable=True),
            pa.field("tags", pa.list_(pa.string()), nullable=True),
        ]
    )
    with tempfile.TemporaryDirectory(prefix="caption-sidecar-validator-selftest-") as raw:
        root = Path(raw)
        physical_null = pa.Table.from_arrays(
            [
                pa.nulls(3),
                pa.nulls(3),
            ],
            names=["caption", "tags"],
        )
        null_path = root / "physical-null.parquet"
        pq.write_table(physical_null, null_path, row_group_size=3)
        null_parquet = pq.ParquetFile(null_path)
        canonical_null = canonical_source_row_group(
            null_path, null_parquet, 0, expected_source
        )
        if not canonical_null.schema.equals(expected_source, check_metadata=True):
            raise AssertionError("physical null source was not cast to canonical schema")
        if canonical_null.to_pylist() != [
            {"caption": None, "tags": None},
            {"caption": None, "tags": None},
            {"caption": None, "tags": None},
        ]:
            raise AssertionError("physical null safe cast changed values")

        wrong_path = root / "wrong-int.parquet"
        pq.write_table(
            pa.table(
                {
                    "caption": pa.array([1, 2, 3], type=pa.int64()),
                    "tags": pa.nulls(3),
                }
            ),
            wrong_path,
            row_group_size=3,
        )
        wrong_rejected = False
        try:
            canonical_source_row_group(
                wrong_path, pq.ParquetFile(wrong_path), 0, expected_source
            )
        except ValueError:
            wrong_rejected = True
        if not wrong_rejected:
            raise AssertionError("illegal physical int->string source type was accepted")

        nonnullable = pa.schema(
            [
                pa.field("caption", pa.string(), nullable=False),
                pa.field("tags", pa.list_(pa.string()), nullable=True),
            ]
        )
        null_nonnullable_rejected = False
        try:
            validate_source_physical_schema(
                null_path, null_parquet.schema_arrow, nonnullable
            )
        except ValueError:
            null_nonnullable_rejected = True
        if not null_nonnullable_rejected:
            raise AssertionError("physical null was accepted for a non-nullable field")

    known_answer = known_answer_digest()
    if not is_sha256(KNOWN_ANSWER_EXPECTED_SHA256):
        raise AssertionError(
            "KNOWN_ANSWER_EXPECTED_SHA256 is not frozen; run --print-known-answer "
            "on the target JCC image, patch the 64-hex value, then rerun self-test"
        )
    if known_answer != KNOWN_ANSWER_EXPECTED_SHA256:
        raise AssertionError(
            f"semantic digest known-answer drift: "
            f"{known_answer} != {KNOWN_ANSWER_EXPECTED_SHA256}"
        )
    return {
        "status": "PASS",
        "schema_version": "offline_arb_caption_sidecar_validator_selftest_v1",
        "rows": rows,
        "digest": one,
        "known_answer_digest": known_answer,
        "covers": [
            "physical-null-to-string",
            "physical-null-to-list<string>",
            "illegal-type-rejection",
            "nonnullable-null-rejection",
            "nonempty-list",
            "recursive-unicode",
            "chunk-boundary-independence",
        ],
    }


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_SIDECAR_ROOT)
    parser.add_argument("--domain", action="append", choices=list(DOMAINS))
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--sample-rows", type=positive_int, default=64)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--hash-workers", type=positive_int, default=8)
    parser.add_argument("--sample-workers", type=positive_int, default=8)
    parser.add_argument("--run-nonce")
    parser.add_argument("--jobtag")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--evidence-host")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--print-known-answer", action="store_true")
    mode.add_argument("--revalidate-result", type=Path)
    mode.add_argument("--close-results", type=Path, nargs=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.print_known_answer:
            sys.stdout.write(known_answer_digest() + "\n")
            sys.stdout.flush()
            return 0
        if args.self_test:
            result = run_self_test()
        elif args.revalidate_result is not None:
            result = run_result_revalidation(args)
        elif args.close_results is not None:
            result = run_full_root_closure(args)
        else:
            result = run_validation(args)
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return 0
    except Exception as exc:
        print(
            f"FATAL caption sidecar validation: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
