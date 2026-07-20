"""Strict reader for the catalog-row-aligned OFFLINE_ARB caption sidecar.

The sidecar changes only where raw caption fields are read.  Every returned row
is addressed by the immutable global catalog ``row_id`` and is checked against
the physical row stored in parquet before it can reach caption augmentation.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import re
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.offline_arb.caption_sidecar_contract import (
    BUILDER_BUILD_ID,
    CAPTION_SCHEMA_CONTRACT_SHA256,
    CAPTION_SCHEMAS,
    DOMAINS,
    FILE_ROWS,
    ROW_GROUP_ROWS,
    ROW_ORDER,
    SCHEMA_VERSION,
    SEMANTIC_DIGEST_ALGORITHM,
    canonical_schema,
    sidecar_schema,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_HEADER_KEYS = frozenset(
    {
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
)
_DOMAIN_KEYS = frozenset(
    {
        "row_base",
        "row_count",
        "columns",
        "source_semantics_sha256",
        "decoded_sidecar_semantics_sha256",
        "files",
    }
)
_FILE_KEYS = frozenset(
    {
        "name",
        "row_start",
        "row_stop",
        "rows",
        "bytes",
        "sha256",
        "row_groups",
        "schema",
    }
)
_CATALOG_FILE_KEYS = frozenset(
    {"name", "row_base", "rows", "bytes", "sha256", "schema"}
)
_CATALOG_HEADER_BINDING_KEYS = frozenset(
    {"name", "rows", "bytes", "sha256", "schema"}
)


class CaptionSidecarError(RuntimeError):
    """The frozen sidecar cannot serve the requested exact catalog rows."""


@dataclass(frozen=True)
class _FileSpec:
    domain: str
    index: int
    path: Path
    row_start: int
    row_stop: int
    rows: int
    size_bytes: int
    sha256: str
    row_groups: int
    schema: pa.Schema
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _DomainSpec:
    name: str
    row_base: int
    row_count: int
    columns: tuple[str, ...]
    source_semantics_sha256: str
    decoded_sidecar_semantics_sha256: str
    files: tuple[_FileSpec, ...]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CaptionSidecarError(
            f"{label} must be an integer >= {minimum}, got {value!r}"
        )
    return value


def _require_exact_object_keys(
    value: Any,
    expected_keys: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CaptionSidecarError(f"{label} must be an object")
    live_keys = set(value)
    if live_keys != expected_keys:
        raise CaptionSidecarError(
            f"{label} key set mismatch: "
            f"missing={sorted(expected_keys-live_keys)}, "
            f"added={sorted(live_keys-expected_keys)}"
        )
    return value


def _read_object(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise CaptionSidecarError(f"JSON artifact is not a regular file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except CaptionSidecarError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptionSidecarError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptionSidecarError(f"expected JSON object: {path}")
    return value


def _require_real_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CaptionSidecarError(f"cannot stat {label} {path}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise CaptionSidecarError(f"{label} is not a real directory: {path}")


def _directory_names(path: Path, label: str) -> set[str]:
    try:
        return {child.name for child in path.iterdir()}
    except OSError as exc:
        raise CaptionSidecarError(f"cannot list {label} {path}: {exc}") from exc


def _catalog_file_binding(value: Any, label: str) -> dict[str, Any]:
    raw = _require_exact_object_keys(value, _CATALOG_FILE_KEYS, label)
    name = raw["name"]
    if (
        not isinstance(name, str)
        or not name.endswith(".parquet")
        or Path(name).name != name
    ):
        raise CaptionSidecarError(f"{label}.name is malformed: {name!r}")
    row_base = _strict_int(raw["row_base"], f"{label}.row_base")
    rows = _strict_int(raw["rows"], f"{label}.rows", minimum=1)
    size_bytes = _strict_int(raw["bytes"], f"{label}.bytes", minimum=1)
    sha256 = raw["sha256"]
    if not _is_sha256(sha256):
        raise CaptionSidecarError(f"{label}.sha256 is malformed: {sha256!r}")
    schema = raw["schema"]
    if not isinstance(schema, str) or not schema.strip():
        raise CaptionSidecarError(f"{label}.schema is missing")
    return {
        "name": name,
        "row_base": row_base,
        "rows": rows,
        "bytes": size_bytes,
        "sha256": sha256,
        "schema": schema,
    }


def _catalog_header_file_binding(
    value: Any,
    label: str,
    *,
    row_base: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CaptionSidecarError(f"{label} must be an object")
    missing = _CATALOG_HEADER_BINDING_KEYS - set(value)
    if missing:
        raise CaptionSidecarError(
            f"{label} lacks sidecar binding fields: {sorted(missing)}"
        )
    return _catalog_file_binding(
        {
            "name": value["name"],
            "row_base": row_base,
            "rows": value["rows"],
            "bytes": value["bytes"],
            "sha256": value["sha256"],
            "schema": value["schema"],
        },
        label,
    )


class CaptionSidecarStore:
    """Read typed caption rows through one executor shared by all dataset domains."""

    def __init__(
        self,
        root: str | Path,
        *,
        catalog_hash: str,
        catalog_header_sha256: str,
        expected_domains: Mapping[str, Mapping[str, Any]],
        workers: int = 16,
    ) -> None:
        self.root = Path(root).expanduser()
        if not self.root.is_absolute():
            raise CaptionSidecarError(
                f"caption sidecar root must be absolute: {self.root}"
            )
        _require_real_directory(self.root, "caption sidecar root")
        if type(workers) is not int or workers <= 0:
            raise CaptionSidecarError(
                f"caption sidecar workers must be a positive integer, got {workers!r}"
            )
        if not _is_sha256(catalog_hash) or not _is_sha256(catalog_header_sha256):
            raise CaptionSidecarError("runtime catalog identities are malformed")
        expected_domain_order = tuple(expected_domains)
        canonical_domain_order = tuple(DOMAINS)
        if expected_domain_order != canonical_domain_order:
            raise CaptionSidecarError(
                "runtime caption sidecar domain order/set is not canonical: "
                f"runtime={list(expected_domain_order)!r}, "
                f"canonical={list(canonical_domain_order)!r}"
            )
        expected_total_rows = sum(
            _strict_int(
                expected_domains[domain].get("row_count"),
                f"runtime domains.{domain}.row_count",
                minimum=1,
            )
            for domain in expected_domain_order
        )

        header_path = self.root / "header.json"
        header = _read_object(header_path)
        _require_exact_object_keys(header, _HEADER_KEYS, "caption sidecar header")
        if header.get("schema_version") != SCHEMA_VERSION:
            raise CaptionSidecarError(
                f"unsupported caption sidecar schema: {header.get('schema_version')!r}"
            )
        header_sha256 = header.get("header_sha256")
        content = {
            key: value
            for key, value in header.items()
            if key not in {"header_sha256", "generated_at"}
        }
        if not _is_sha256(header_sha256) or _canonical_sha256(content) != header_sha256:
            raise CaptionSidecarError("caption sidecar header self-hash mismatch")
        success_path = self.root / "_SUCCESS"
        try:
            success_metadata = success_path.lstat()
            if not stat.S_ISREG(success_metadata.st_mode):
                raise CaptionSidecarError(
                    f"caption sidecar _SUCCESS is not a regular file: {success_path}"
                )
            success = success_path.read_text(encoding="ascii")
        except CaptionSidecarError:
            raise
        except OSError as exc:
            raise CaptionSidecarError(
                f"cannot read caption sidecar _SUCCESS {success_path}: {exc}"
            ) from exc
        if success not in {header_sha256, f"{header_sha256}\n"}:
            raise CaptionSidecarError("caption sidecar _SUCCESS/header hash mismatch")
        if header.get("catalog_hash") != catalog_hash:
            raise CaptionSidecarError(
                "caption sidecar was built from a different catalog: "
                f"sidecar={header.get('catalog_hash')!r}, runtime={catalog_hash!r}"
            )
        if header.get("catalog_header_sha256") != catalog_header_sha256:
            raise CaptionSidecarError(
                "caption sidecar catalog header SHA-256 mismatch: "
                f"sidecar={header.get('catalog_header_sha256')!r}, "
                f"runtime={catalog_header_sha256!r}"
            )
        if header.get("row_order") != ROW_ORDER:
            raise CaptionSidecarError(
                f"unsupported caption sidecar row order: {header.get('row_order')!r}"
            )
        if header.get("file_rows") != FILE_ROWS:
            raise CaptionSidecarError(
                f"caption sidecar file_rows must be {FILE_ROWS}: {header.get('file_rows')!r}"
            )
        if header.get("row_group_rows") != ROW_GROUP_ROWS:
            raise CaptionSidecarError(
                "caption sidecar row_group_rows must be "
                f"{ROW_GROUP_ROWS}: {header.get('row_group_rows')!r}"
            )
        if header.get("semantic_digest_algorithm") != SEMANTIC_DIGEST_ALGORITHM:
            raise CaptionSidecarError(
                "caption sidecar semantic_digest_algorithm must be "
                f"{SEMANTIC_DIGEST_ALGORITHM!r}: "
                f"{header.get('semantic_digest_algorithm')!r}"
            )
        if header.get("builder_build_id") != BUILDER_BUILD_ID:
            raise CaptionSidecarError(
                "caption sidecar builder_build_id must be "
                f"{BUILDER_BUILD_ID!r}: {header.get('builder_build_id')!r}"
            )
        if (
            header.get("caption_schema_contract_sha256")
            != CAPTION_SCHEMA_CONTRACT_SHA256
        ):
            raise CaptionSidecarError(
                "caption sidecar schema contract SHA-256 mismatch: "
                f"sidecar={header.get('caption_schema_contract_sha256')!r}, "
                f"runtime={CAPTION_SCHEMA_CONTRACT_SHA256!r}"
            )
        if header.get("diagnostic_run") is not False:
            raise CaptionSidecarError(
                "formal runtime refuses caption sidecar diagnostic_run artifacts"
            )
        if (
            "diagnostic_rows_per_domain" not in header
            or header["diagnostic_rows_per_domain"] is not None
        ):
            raise CaptionSidecarError(
                "formal runtime requires diagnostic_rows_per_domain to be null"
            )
        if header.get("domain_order") != list(expected_domain_order):
            raise CaptionSidecarError(
                "caption sidecar domain_order mismatch: "
                f"sidecar={header.get('domain_order')!r}, "
                f"runtime={list(expected_domain_order)!r}"
            )
        total_rows = _strict_int(
            header.get("total_rows"), "caption sidecar total_rows", minimum=1
        )
        if total_rows != expected_total_rows:
            raise CaptionSidecarError(
                "caption sidecar total_rows mismatch: "
                f"sidecar={total_rows}, runtime={expected_total_rows}"
            )
        generated_at = header.get("generated_at")
        if not isinstance(generated_at, str) or not generated_at.strip():
            raise CaptionSidecarError(
                "caption sidecar generated_at must be a non-empty string"
            )

        raw_catalog_files = header.get("catalog_files")
        if (
            not isinstance(raw_catalog_files, dict)
            or set(raw_catalog_files) != set(expected_domain_order)
        ):
            raise CaptionSidecarError(
                "caption sidecar catalog_files domain set mismatch: "
                f"sidecar={sorted(raw_catalog_files) if isinstance(raw_catalog_files, dict) else raw_catalog_files!r}, "
                f"runtime={list(expected_domain_order)!r}"
            )
        for domain, expected in expected_domains.items():
            expected_base = _strict_int(
                expected.get("row_base"), f"runtime domains.{domain}.row_base"
            )
            expected_count = _strict_int(
                expected.get("row_count"),
                f"runtime domains.{domain}.row_count",
                minimum=1,
            )
            expected_binding = _catalog_header_file_binding(
                expected.get("catalog_file"),
                f"runtime domains.{domain}.catalog_file",
                row_base=expected_base,
            )
            if (
                expected_binding["row_base"],
                expected_binding["rows"],
            ) != (expected_base, expected_count):
                raise CaptionSidecarError(
                    f"runtime catalog file row binding mismatch for {domain}"
                )
            sidecar_binding = _catalog_file_binding(
                raw_catalog_files[domain], f"catalog_files.{domain}"
            )
            if sidecar_binding != expected_binding:
                raise CaptionSidecarError(
                    f"caption sidecar/catalog file binding mismatch for {domain}: "
                    f"sidecar={sidecar_binding!r}, runtime={expected_binding!r}"
                )

        raw_domains = header.get("domains")
        if not isinstance(raw_domains, dict) or set(raw_domains) != set(expected_domains):
            raise CaptionSidecarError(
                "caption sidecar domain set mismatch: "
                f"sidecar={sorted(raw_domains) if isinstance(raw_domains, dict) else raw_domains!r}, "
                f"runtime={sorted(expected_domains)}"
            )
        domains: dict[str, _DomainSpec] = {}
        for domain, expected in expected_domains.items():
            domains[domain] = self._parse_domain(domain, raw_domains[domain], expected)
        expected_root_entries = {"header.json", "_SUCCESS", *expected_domains}
        live_root_entries = _directory_names(self.root, "caption sidecar root")
        if live_root_entries != expected_root_entries:
            raise CaptionSidecarError(
                "caption sidecar root entry set mismatch: "
                f"missing={sorted(expected_root_entries-live_root_entries)}, "
                f"added={sorted(live_root_entries-expected_root_entries)}"
            )
        self.header = header
        self.header_sha256 = header_sha256
        self.catalog_hash = catalog_hash
        self.workers = workers
        self._domains = domains
        self._state_lock = threading.Lock()
        self._closed = False
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="offline-caption-sidecar",
        )

    def _parse_domain(
        self,
        domain: str,
        raw: Any,
        expected: Mapping[str, Any],
    ) -> _DomainSpec:
        raw = _require_exact_object_keys(
            raw, _DOMAIN_KEYS, f"caption sidecar domain {domain}"
        )
        row_base = _strict_int(raw.get("row_base"), f"domains.{domain}.row_base")
        row_count = _strict_int(
            raw.get("row_count"), f"domains.{domain}.row_count", minimum=1
        )
        expected_base = _strict_int(
            expected.get("row_base"), f"runtime domains.{domain}.row_base"
        )
        expected_count = _strict_int(
            expected.get("row_count"),
            f"runtime domains.{domain}.row_count",
            minimum=1,
        )
        if (row_base, row_count) != (expected_base, expected_count):
            raise CaptionSidecarError(
                f"caption sidecar row range mismatch for {domain}: "
                f"sidecar=[{row_base},{row_base + row_count}), "
                f"runtime=[{expected_base},{expected_base + expected_count})"
            )
        raw_columns = raw.get("columns")
        expected_columns = tuple(str(value) for value in expected.get("columns", ()))
        canonical_caption_schema = CAPTION_SCHEMAS[domain]
        canonical_columns = tuple(canonical_caption_schema.names)
        if expected_columns != canonical_columns:
            raise CaptionSidecarError(
                f"runtime caption column contract is not canonical for {domain}: "
                f"runtime={list(expected_columns)!r}, "
                f"canonical={list(canonical_columns)!r}"
            )
        canonical_file_schema = sidecar_schema(
            canonical_caption_schema, canonical_columns
        )
        if (
            not isinstance(raw_columns, list)
            or any(not isinstance(value, str) or not value for value in raw_columns)
            or len(set(raw_columns)) != len(raw_columns)
            or tuple(raw_columns) != expected_columns
        ):
            raise CaptionSidecarError(
                f"caption sidecar column contract mismatch for {domain}: "
                f"sidecar={raw_columns!r}, runtime={list(expected_columns)!r}"
            )
        source_digest = raw.get("source_semantics_sha256")
        decoded_digest = raw.get("decoded_sidecar_semantics_sha256")
        if (
            not _is_sha256(source_digest)
            or not _is_sha256(decoded_digest)
            or source_digest != decoded_digest
        ):
            raise CaptionSidecarError(
                f"caption sidecar semantic digest mismatch for {domain}: "
                f"source={source_digest!r}, decoded={decoded_digest!r}"
            )

        raw_files = raw.get("files")
        expected_file_count = math.ceil(row_count / FILE_ROWS)
        if not isinstance(raw_files, list) or len(raw_files) != expected_file_count:
            raise CaptionSidecarError(
                f"caption sidecar file count mismatch for {domain}: "
                f"{len(raw_files) if isinstance(raw_files, list) else raw_files!r} "
                f"!= {expected_file_count}"
            )
        files: list[_FileSpec] = []
        domain_root = self.root / domain
        _require_real_directory(domain_root, f"caption sidecar domain {domain}")
        for index, file_raw in enumerate(raw_files):
            expected_name = f"{domain}/part-{index:05d}.parquet"
            file_raw = _require_exact_object_keys(
                file_raw, _FILE_KEYS, f"caption sidecar file {expected_name}"
            )
            name = file_raw.get("name")
            if name != expected_name:
                raise CaptionSidecarError(
                    f"caption sidecar file name mismatch: {name!r} != {expected_name!r}"
                )
            path = self.root / expected_name
            if path.parent.parent != self.root or path.parent.name != domain:
                raise CaptionSidecarError(f"unsafe caption sidecar path: {path}")
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise CaptionSidecarError(f"caption sidecar file missing: {path}: {exc}") from exc
            if not stat.S_ISREG(metadata.st_mode):
                raise CaptionSidecarError(
                    f"caption sidecar file is not regular or is a symlink: {path}"
                )
            row_start = row_base + index * FILE_ROWS
            row_stop = min(row_base + row_count, row_start + FILE_ROWS)
            rows = row_stop - row_start
            declared_row_start = _strict_int(
                file_raw.get("row_start"), f"{expected_name}.row_start"
            )
            declared_row_stop = _strict_int(
                file_raw.get("row_stop"), f"{expected_name}.row_stop", minimum=1
            )
            declared_rows = _strict_int(
                file_raw.get("rows"), f"{expected_name}.rows", minimum=1
            )
            if (declared_row_start, declared_row_stop, declared_rows) != (
                row_start,
                row_stop,
                rows,
            ):
                raise CaptionSidecarError(
                    f"caption sidecar file row layout mismatch: {expected_name}"
                )
            size_bytes = _strict_int(
                file_raw.get("bytes"), f"{expected_name}.bytes", minimum=1
            )
            if metadata.st_size != size_bytes:
                raise CaptionSidecarError(
                    f"caption sidecar file byte-size mismatch: {path}: "
                    f"{metadata.st_size} != {size_bytes}"
                )
            sha256 = file_raw.get("sha256")
            if not _is_sha256(sha256):
                raise CaptionSidecarError(
                    f"caption sidecar file SHA-256 is malformed: {expected_name}"
                )
            row_groups = math.ceil(rows / ROW_GROUP_ROWS)
            if file_raw.get("row_groups") != row_groups:
                raise CaptionSidecarError(
                    f"caption sidecar row-group count mismatch: {expected_name}"
                )
            declared_schema = file_raw.get("schema")
            if declared_schema != str(canonical_file_schema):
                raise CaptionSidecarError(
                    f"caption sidecar declared schema is not canonical: "
                    f"{expected_name}: {declared_schema!r} != "
                    f"{str(canonical_file_schema)!r}"
                )
            files.append(
                _FileSpec(
                    domain=domain,
                    index=index,
                    path=path,
                    row_start=row_start,
                    row_stop=row_stop,
                    rows=rows,
                    size_bytes=size_bytes,
                    sha256=sha256,
                    row_groups=row_groups,
                    schema=canonical_file_schema,
                    device=metadata.st_dev,
                    inode=metadata.st_ino,
                    mtime_ns=metadata.st_mtime_ns,
                    ctime_ns=metadata.st_ctime_ns,
                )
            )
        expected_file_names = {spec.path.name for spec in files}
        live_file_names = _directory_names(
            domain_root, f"caption sidecar domain {domain}"
        )
        if live_file_names != expected_file_names:
            raise CaptionSidecarError(
                f"caption sidecar file set mismatch for {domain}: "
                f"missing={sorted(expected_file_names-live_file_names)}, "
                f"added={sorted(live_file_names-expected_file_names)}"
            )
        return _DomainSpec(
            name=domain,
            row_base=row_base,
            row_count=row_count,
            columns=expected_columns,
            source_semantics_sha256=source_digest,
            decoded_sidecar_semantics_sha256=decoded_digest,
            files=tuple(files),
        )

    def _read_file_group(
        self,
        spec: _FileSpec,
        columns: tuple[str, ...],
        row_group: int,
        requests: Sequence[tuple[int, int]],
    ) -> dict[int, dict[str, Any]]:
        try:
            before = spec.path.lstat()
            expected_identity = (
                spec.device,
                spec.inode,
                spec.size_bytes,
                spec.mtime_ns,
                spec.ctime_ns,
            )
            live_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            if not stat.S_ISREG(before.st_mode) or live_identity != expected_identity:
                raise CaptionSidecarError(
                    f"caption sidecar file identity changed: {spec.path}: "
                    f"live={live_identity!r}, expected={expected_identity!r}"
                )
            parquet = pq.ParquetFile(spec.path)
            if (
                parquet.metadata.num_rows != spec.rows
                or parquet.num_row_groups != spec.row_groups
            ):
                raise CaptionSidecarError(
                    f"caption sidecar footer layout mismatch: {spec.path}"
                )
            schema = parquet.schema_arrow
            normalized_schema = canonical_schema(schema)
            if (
                not schema.equals(normalized_schema, check_metadata=True)
                or not schema.equals(spec.schema, check_metadata=True)
                or str(schema) != str(spec.schema)
            ):
                raise CaptionSidecarError(
                    f"caption sidecar parquet schema mismatch: {spec.path}: "
                    f"live={schema!s}, expected={spec.schema!s}"
                )
            expected_names = spec.schema.names

            if row_group < 0 or row_group >= spec.row_groups:
                raise CaptionSidecarError(
                    f"caption sidecar row group out of range: {spec.path}:{row_group}"
                )
            expected_start = spec.row_start + row_group * ROW_GROUP_ROWS
            expected_rows = min(ROW_GROUP_ROWS, spec.row_stop - expected_start)
            metadata = parquet.metadata.row_group(row_group)
            if metadata.num_rows != expected_rows:
                raise CaptionSidecarError(
                    f"caption sidecar row-group size mismatch: "
                    f"{spec.path}:{row_group}: {metadata.num_rows} != {expected_rows}"
                )
            table = parquet.read_row_group(
                row_group,
                columns=expected_names,
                use_threads=False,
            )
            decoded_schema = table.schema
            normalized_decoded_schema = canonical_schema(decoded_schema)
            if (
                not decoded_schema.equals(
                    normalized_decoded_schema, check_metadata=True
                )
                or not decoded_schema.equals(spec.schema, check_metadata=True)
                or str(decoded_schema) != str(spec.schema)
            ):
                raise CaptionSidecarError(
                    f"caption sidecar decoded schema mismatch: "
                    f"{spec.path}:{row_group}"
                )
            if table.num_rows != expected_rows or table.column("row_id").null_count:
                raise CaptionSidecarError(
                    f"caption sidecar decoded row-group layout mismatch: "
                    f"{spec.path}:{row_group}"
                )
            row_ids = table.column("row_id").to_pylist()
            expected_row_ids = list(
                range(expected_start, expected_start + expected_rows)
            )
            if row_ids != expected_row_ids:
                raise CaptionSidecarError(
                    f"caption sidecar row_id sequence mismatch: {spec.path}:{row_group}"
                )
            result: dict[int, dict[str, Any]] = {}
            for row_id, offset in requests:
                if offset < 0 or offset >= expected_rows:
                    raise CaptionSidecarError(
                        f"caption sidecar row offset out of range: "
                        f"row_id={row_id}, file={spec.path}, rg={row_group}, offset={offset}"
                    )
                raw = table.slice(offset, 1).to_pylist()[0]
                if raw.get("row_id") != row_id:
                    raise CaptionSidecarError(
                        f"caption sidecar requested/stored row_id mismatch: "
                        f"requested={row_id}, stored={raw.get('row_id')!r}"
                    )
                result[row_id] = {column: raw.get(column) for column in columns}
            after = spec.path.lstat()
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if not stat.S_ISREG(after.st_mode) or after_identity != expected_identity:
                raise CaptionSidecarError(
                    f"caption sidecar file changed while reading: {spec.path}"
                )
            return result
        except CaptionSidecarError:
            raise
        except Exception as exc:
            raise CaptionSidecarError(
                f"cannot read caption sidecar file {spec.path}: {type(exc).__name__}: {exc}"
            ) from exc

    def get_many(
        self,
        domain: str,
        row_ids: Sequence[int],
    ) -> dict[int, dict[str, Any]]:
        spec = self._domains.get(domain)
        if spec is None:
            raise CaptionSidecarError(f"unknown caption sidecar domain: {domain!r}")
        unique: list[int] = []
        seen: set[int] = set()
        for raw in row_ids:
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise CaptionSidecarError(
                    f"caption sidecar row_id must be an integer: {raw!r}"
                )
            row_id = int(raw)
            if row_id < spec.row_base or row_id >= spec.row_base + spec.row_count:
                raise CaptionSidecarError(
                    f"caption sidecar row_id {row_id} is outside {domain} range "
                    f"[{spec.row_base},{spec.row_base + spec.row_count})"
                )
            if row_id not in seen:
                seen.add(row_id)
                unique.append(row_id)
        unique.sort()
        if not unique:
            with self._state_lock:
                if self._closed:
                    raise CaptionSidecarError("caption sidecar store is closed")
            return {}

        by_group: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for row_id in unique:
            local = row_id - spec.row_base
            file_index = local // FILE_ROWS
            within_file = local % FILE_ROWS
            row_group = within_file // ROW_GROUP_ROWS
            offset = within_file % ROW_GROUP_ROWS
            by_group.setdefault((file_index, row_group), []).append((row_id, offset))

        with self._state_lock:
            if self._closed:
                raise CaptionSidecarError("caption sidecar store is closed")
            futures = {
                self._executor.submit(
                    self._read_file_group,
                    spec.files[file_index],
                    spec.columns,
                    row_group,
                    requests,
                ): (file_index, row_group)
                for (file_index, row_group), requests in sorted(by_group.items())
            }
        result: dict[int, dict[str, Any]] = {}
        for future in concurrent.futures.as_completed(futures):
            file_index, row_group = futures[future]
            try:
                rows = future.result()
            except Exception as exc:
                if isinstance(exc, CaptionSidecarError):
                    raise
                raise CaptionSidecarError(
                    f"caption sidecar worker failed for {domain} "
                    f"file {file_index} row group {row_group}: {exc}"
                ) from exc
            overlap = set(result).intersection(rows)
            if overlap:
                raise CaptionSidecarError(
                    f"caption sidecar returned duplicate row_ids: {sorted(overlap)}"
                )
            result.update(rows)
        if set(result) != set(unique):
            raise CaptionSidecarError(
                f"caption sidecar result set mismatch for {domain}: "
                f"missing={sorted(set(unique)-set(result))}, "
                f"added={sorted(set(result)-set(unique))}"
            )
        return result

    def close(self, *, wait: bool = True) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def __del__(self) -> None:
        try:
            self.close(wait=False)
        except Exception:
            pass


__all__ = ["CaptionSidecarError", "CaptionSidecarStore"]
