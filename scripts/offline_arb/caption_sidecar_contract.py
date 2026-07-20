"""Shared immutable contract for OFFLINE_ARB caption sidecar artifacts.

The builder and the independent validator import this module so that semantic
digests cannot silently drift between two implementations.  Runtime does not
recompute these full-data digests; it verifies the frozen header/layout and
checks physical ``row_id`` values on every requested row group.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable
from typing import Any

import pyarrow as pa


SCHEMA_VERSION = "offline_arb_caption_sidecar_v1"
SEMANTIC_DIGEST_ALGORITHM = "arrow_ipc_record_batch_v1"
BUILDER_BUILD_ID = "offline_arb_caption_sidecar_builder_v3_portable_closure"
ROW_ORDER = "global_catalog_row_id"
FILE_ROWS = 65_536
ROW_GROUP_ROWS = 256
DOMAINS = ("anime", "monet_real", "gpic", "synthetic", "textatlas")

# Canonical nullable logical schemas are part of the v1 wire contract.  They do
# not depend on whichever source locator happens to be visited first: parquet
# may legitimately expose an all-null physical column as Arrow ``null`` in one
# shard and its logical string type in another.  Such a physical null column is
# safely promoted to the canonical nullable type by the builder.
_ANIME_LARGE_STRING_COLUMNS = (
    "refined_description",
    "medium_description",
    "short_description",
)
_ANIME_LARGE_LIST_COLUMNS = (
    "tags_general",
    "tags_artist",
    "tags_character",
    "tags_copyright",
    "tags_meta",
)
_ANIME_TRAILING_LARGE_STRING_COLUMNS = (
    "foreground",
    "background",
    "style",
    "image_type",
    "camera_and_lens_effects",
)


def _nullable_string_schema(columns: Iterable[str]) -> pa.Schema:
    return pa.schema([pa.field(name, pa.string(), nullable=True) for name in columns])


CAPTION_SCHEMAS: dict[str, pa.Schema] = {
    "anime": pa.schema(
        [
            *(
                pa.field(name, pa.large_string(), nullable=True)
                for name in _ANIME_LARGE_STRING_COLUMNS
            ),
            *(
                pa.field(
                    name,
                    pa.large_list(
                        pa.field("element", pa.large_string(), nullable=True)
                    ),
                    nullable=True,
                )
                for name in _ANIME_LARGE_LIST_COLUMNS
            ),
            *(
                pa.field(name, pa.large_string(), nullable=True)
                for name in _ANIME_TRAILING_LARGE_STRING_COLUMNS
            ),
        ]
    ),
    "monet_real": _nullable_string_schema(
        (
            "caption_gemini-2.5-flash-lite",
            "caption_sharegpt4v-7b",
            "caption_internvl-3-8b",
            "caption_florence-2-large",
            "caption_original",
        )
    ),
    "gpic": _nullable_string_schema(("caption",)),
    "synthetic": _nullable_string_schema(("caption_original",)),
    "textatlas": _nullable_string_schema(
        ("caption1", "caption2", "caption3", "caption4", "caption5")
    ),
}

# These are source columns only.  Anime ``rating_bucket`` and ``year`` remain
# catalog fields and are deliberately not duplicated in the sidecar.
CAPTION_COLUMNS: dict[str, tuple[str, ...]] = {
    domain: tuple(schema.names) for domain, schema in CAPTION_SCHEMAS.items()
}

_DIGEST_MAGIC = b"offline_arb_caption_sidecar:arrow_ipc_record_batch_v1\0"


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _caption_schema_contract_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "semantic_digest_algorithm": SEMANTIC_DIGEST_ALGORITHM,
        "row_order": ROW_ORDER,
        "file_rows": FILE_ROWS,
        "row_group_rows": ROW_GROUP_ROWS,
        "domains": list(DOMAINS),
        "schemas": {
            domain: [
                {"name": "row_id", "type": "int64", "nullable": False},
                *[
                    {
                        "name": field.name,
                        "type": str(field.type),
                        "nullable": bool(field.nullable),
                    }
                    for field in CAPTION_SCHEMAS[domain]
                ],
            ]
            for domain in DOMAINS
        },
    }


CAPTION_SCHEMA_CONTRACT_SHA256 = canonical_json_sha256(
    _caption_schema_contract_payload()
)


def header_self_hash(header: dict[str, Any]) -> str:
    """Hash the immutable header payload.

    ``generated_at`` is evidence only and ``header_sha256`` is self-referential,
    so both are excluded exactly as required by the runtime reader.
    """

    return canonical_json_sha256(
        {
            key: value
            for key, value in header.items()
            if key not in {"generated_at", "header_sha256"}
        }
    )


def _without_nested_metadata(dtype: pa.DataType) -> pa.DataType:
    """Preserve logical Arrow types while removing non-semantic field metadata."""

    if pa.types.is_list(dtype):
        value = dtype.value_field
        return pa.list_(
            pa.field(
                value.name,
                _without_nested_metadata(value.type),
                nullable=value.nullable,
            )
        )
    if pa.types.is_large_list(dtype):
        value = dtype.value_field
        return pa.large_list(
            pa.field(
                value.name,
                _without_nested_metadata(value.type),
                nullable=value.nullable,
            )
        )
    if pa.types.is_fixed_size_list(dtype):
        value = dtype.value_field
        return pa.list_(
            pa.field(
                value.name,
                _without_nested_metadata(value.type),
                nullable=value.nullable,
            ),
            list_size=dtype.list_size,
        )
    if pa.types.is_struct(dtype):
        return pa.struct(
            [
                pa.field(
                    field.name,
                    _without_nested_metadata(field.type),
                    nullable=field.nullable,
                )
                for field in dtype
            ]
        )
    if pa.types.is_dictionary(dtype):
        return pa.dictionary(
            _without_nested_metadata(dtype.index_type),
            _without_nested_metadata(dtype.value_type),
            ordered=dtype.ordered,
        )
    # Caption sources currently use strings and (nested) lists of strings.  All
    # remaining Arrow scalar types are already metadata-free DataType values.
    return dtype


def canonical_schema(fields: Iterable[pa.Field]) -> pa.Schema:
    return pa.schema(
        [
            pa.field(
                field.name,
                _without_nested_metadata(field.type),
                nullable=field.nullable,
            )
            for field in fields
        ]
    )


def sidecar_schema(source_schema: pa.Schema, columns: Iterable[str]) -> pa.Schema:
    columns_tuple = tuple(columns)
    if source_schema.names != list(columns_tuple):
        raise ValueError(
            f"source schema names {source_schema.names!r} != {list(columns_tuple)!r}"
        )
    caption_fields = canonical_schema(source_schema).names
    if caption_fields != list(columns_tuple):  # defensive: names must not mutate
        raise AssertionError("canonical schema changed field names")
    normalized_source = canonical_schema(source_schema)
    return pa.schema(
        [pa.field("row_id", pa.int64(), nullable=False), *list(normalized_source)]
    )


class ArrowIpcSemanticDigester:
    """SHA-256 over fixed 256-row canonical Arrow IPC record-batch frames.

    Framing is intentionally independent of parquet encoding, file boundaries,
    source read order, and Arrow chunking:

    * magic bytes;
    * uint32-BE domain UTF-8 length + bytes;
    * uint64-BE schema IPC length + ``schema.serialize()`` bytes;
    * uint64-BE ``row_base`` and ``row_count``;
    * for every logical row-group-sized batch in global-row order: uint32-BE
      row count, uint64-BE IPC frame length, then ``RecordBatch.serialize()``;
    * terminal marker plus uint64-BE observed rows and batch count.

    Every non-tail call must contain exactly ``ROW_GROUP_ROWS`` rows.  Thus a
    source-side merge and a full parquet row-group reread produce byte-identical
    digest input even when their physical parquet encodings differ.
    """

    def __init__(
        self,
        *,
        domain: str,
        row_base: int,
        row_count: int,
        schema: pa.Schema,
    ) -> None:
        if domain not in DOMAINS:
            raise ValueError(f"unsupported domain: {domain!r}")
        if isinstance(row_base, bool) or not isinstance(row_base, int) or row_base < 0:
            raise ValueError(f"invalid row_base: {row_base!r}")
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
            raise ValueError(f"invalid row_count: {row_count!r}")
        self.domain = domain
        self.row_base = row_base
        self.row_count = row_count
        self.schema = canonical_schema(schema)
        self._rows = 0
        self._batches = 0
        self._finished = False
        self._digest = hashlib.sha256()
        self._digest.update(_DIGEST_MAGIC)
        domain_bytes = domain.encode("utf-8")
        self._digest.update(struct.pack(">I", len(domain_bytes)))
        self._digest.update(domain_bytes)
        schema_bytes = self.schema.serialize().to_pybytes()
        self._digest.update(struct.pack(">Q", len(schema_bytes)))
        self._digest.update(schema_bytes)
        self._digest.update(struct.pack(">QQ", row_base, row_count))

    @property
    def rows(self) -> int:
        return self._rows

    def update(self, table: pa.Table) -> None:
        if self._finished:
            raise RuntimeError("semantic digester is already finalized")
        remaining = self.row_count - self._rows
        expected_rows = min(ROW_GROUP_ROWS, remaining)
        if table.num_rows != expected_rows:
            raise ValueError(
                f"semantic batch rows {table.num_rows} != expected {expected_rows} "
                f"at domain offset {self._rows}"
            )
        normalized_schema = canonical_schema(table.schema)
        if not normalized_schema.equals(self.schema, check_metadata=False):
            raise ValueError(
                f"semantic batch schema drift:\n{normalized_schema}\n!=\n{self.schema}"
            )
        expected_start = self.row_base + self._rows
        row_ids = table.column("row_id").combine_chunks().to_pylist()
        if row_ids != list(range(expected_start, expected_start + expected_rows)):
            raise ValueError(
                f"semantic batch row_id drift at [{expected_start},"
                f"{expected_start + expected_rows})"
            )
        arrays = [table.column(index).combine_chunks() for index in range(table.num_columns)]
        batch = pa.RecordBatch.from_arrays(arrays, schema=self.schema)
        frame = batch.serialize().to_pybytes()
        self._digest.update(struct.pack(">IQ", batch.num_rows, len(frame)))
        self._digest.update(frame)
        self._rows += batch.num_rows
        self._batches += 1

    def finish(self) -> str:
        if self._finished:
            raise RuntimeError("semantic digester is already finalized")
        if self._rows != self.row_count:
            raise ValueError(
                f"semantic digest row count {self._rows} != expected {self.row_count}"
            )
        self._digest.update(b"END\0")
        self._digest.update(struct.pack(">QQ", self._rows, self._batches))
        self._finished = True
        return self._digest.hexdigest()


__all__ = [
    "ArrowIpcSemanticDigester",
    "BUILDER_BUILD_ID",
    "CAPTION_COLUMNS",
    "CAPTION_SCHEMAS",
    "CAPTION_SCHEMA_CONTRACT_SHA256",
    "DOMAINS",
    "FILE_ROWS",
    "ROW_GROUP_ROWS",
    "ROW_ORDER",
    "SCHEMA_VERSION",
    "SEMANTIC_DIGEST_ALGORITHM",
    "canonical_json_sha256",
    "canonical_schema",
    "header_self_hash",
    "sidecar_schema",
]
