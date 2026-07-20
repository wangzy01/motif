#!/usr/bin/env python3
"""Inventory caption parquet sources referenced by one OFFLINE_ARB catalog domain.

The catalog scan reads only ``caption_parquet_path`` and aggregates counts inside
Arrow one row group at a time.  Source inspection uses ``stat`` plus parquet footer
metadata only; caption payload columns are never scanned.

The final report is emitted as one JSON document on stdout.  Progress and failures
go to stderr so a JCC wrapper can safely capture stdout as the durable artifact.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import stat
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq


BASE = Path("/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop")
DEFAULT_CATALOG_ROOT = BASE / "offline_arb_catalog_v1"
DOMAINS = ("monet_real", "gpic")
CAPTION_COLUMNS = {
    "monet_real": (
        "caption_gemini-2.5-flash-lite",
        "caption_sharegpt4v-7b",
        "caption_internvl-3-8b",
        "caption_florence-2-large",
        "caption_original",
    ),
    "gpic": ("caption",),
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integer, got {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected positive integer, got {value}")
    return value


def scan_catalog_source_counts(
    catalog_path: Path,
    expected_rows: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Return exact path -> catalog-row counts using only one catalog column."""

    before = catalog_path.stat()
    parquet = pq.ParquetFile(catalog_path)
    field = parquet.schema_arrow.field("caption_parquet_path")
    if str(field.type) != "string" or field.nullable:
        raise ValueError(
            f"catalog caption_parquet_path must be non-null string: {catalog_path}"
        )
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"catalog row count mismatch: footer={parquet.metadata.num_rows}, "
            f"header={expected_rows}"
        )

    counts: Counter[str] = Counter()
    for row_group in range(parquet.num_row_groups):
        values = parquet.read_row_group(
            row_group,
            columns=["caption_parquet_path"],
            use_threads=False,
        ).column("caption_parquet_path")
        if values.null_count:
            raise ValueError(
                f"catalog caption_parquet_path has nulls in row group {row_group}"
            )
        value_counts = pc.value_counts(values.combine_chunks())
        paths = value_counts.field(0).to_pylist()
        row_counts = value_counts.field(1).to_pylist()
        for raw_path, raw_count in zip(paths, row_counts):
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(
                    f"invalid catalog caption source in row group {row_group}: "
                    f"{raw_path!r}"
                )
            path = Path(raw_path)
            if not path.is_absolute():
                raise ValueError(f"catalog caption source is not absolute: {raw_path!r}")
            count = int(raw_count)
            if count <= 0:
                raise ValueError(
                    f"invalid catalog source row count for {raw_path}: {count}"
                )
            counts[raw_path] += count

    after = catalog_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"catalog changed while scanning: {catalog_path}")
    if sum(counts.values()) != expected_rows:
        raise RuntimeError(
            f"catalog source counts do not close: {sum(counts.values())} != "
            f"{expected_rows}"
        )
    if not counts:
        raise RuntimeError(f"catalog has no caption sources: {catalog_path}")

    return dict(sorted(counts.items())), {
        "row_groups": int(parquet.num_row_groups),
        "rows": int(parquet.metadata.num_rows),
        "bytes": int(after.st_size),
    }


def inspect_source(
    source_path: str,
    catalog_rows: int,
    caption_columns: tuple[str, ...],
    include_metadata: bool,
) -> tuple[str, dict[str, Any]]:
    path = Path(source_path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"caption source is not a regular file: {path}")
    result: dict[str, Any] = {
        "path": source_path,
        "parent": str(path.parent),
        "name": path.name,
        "bytes": int(before.st_size),
        "catalog_rows": int(catalog_rows),
    }

    if include_metadata:
        parquet = pq.ParquetFile(path)
        schema_names = list(parquet.schema.names)
        missing = sorted(set(caption_columns) - set(schema_names))
        if missing:
            raise ValueError(f"caption source {path} is missing columns {missing}")
        column_indices = [schema_names.index(name) for name in caption_columns]
        compressed = 0
        uncompressed = 0
        row_group_rows: list[int] = []
        for row_group in range(parquet.num_row_groups):
            metadata = parquet.metadata.row_group(row_group)
            row_group_rows.append(int(metadata.num_rows))
            for column_index in column_indices:
                column = metadata.column(column_index)
                compressed += int(column.total_compressed_size)
                uncompressed += int(column.total_uncompressed_size)
        if sum(row_group_rows) != parquet.metadata.num_rows:
            raise RuntimeError(f"source row-group accounting drift: {path}")
        result["parquet"] = {
            "rows": int(parquet.metadata.num_rows),
            "row_groups": int(parquet.num_row_groups),
            "row_group_rows_min": min(row_group_rows) if row_group_rows else 0,
            "row_group_rows_max": max(row_group_rows) if row_group_rows else 0,
            "caption_columns": list(caption_columns),
            "caption_compressed_bytes": compressed,
            "caption_uncompressed_bytes": uncompressed,
        }

    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"caption source changed during inventory: {path}")
    return source_path, result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    catalog_root = args.catalog_root.expanduser()
    header_path = catalog_root / "header.json"
    header = read_object(header_path)
    if header.get("schema_version") != "offline_arb_catalog_v1":
        raise ValueError(f"unsupported catalog schema: {header.get('schema_version')!r}")
    catalog_hash = header.get("catalog_hash")
    if (
        not isinstance(catalog_hash, str)
        or len(catalog_hash) != 64
        or any(character not in "0123456789abcdef" for character in catalog_hash)
    ):
        raise ValueError("catalog header has invalid catalog_hash")
    declared = header.get("files", {}).get(args.domain)
    if not isinstance(declared, dict):
        raise ValueError(f"catalog header lacks files.{args.domain}")
    name = declared.get("name")
    rows = declared.get("rows")
    declared_sha = declared.get("sha256")
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or type(rows) is not int
        or rows <= 0
        or not isinstance(declared_sha, str)
        or len(declared_sha) != 64
    ):
        raise ValueError(f"malformed catalog file declaration for {args.domain}")
    catalog_path = catalog_root / name
    if not catalog_path.is_file():
        raise FileNotFoundError(catalog_path)

    print(
        f"scanning {args.domain} catalog caption_parquet_path only: {catalog_path}",
        file=sys.stderr,
        flush=True,
    )
    source_counts, scan = scan_catalog_source_counts(catalog_path, rows)
    caption_columns = CAPTION_COLUMNS[args.domain]
    source_results: dict[str, dict[str, Any]] = {}
    workers = min(args.metadata_workers, len(source_counts))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=f"caption-inventory-{args.domain}",
    ) as pool:
        futures = {
            pool.submit(
                inspect_source,
                path,
                count,
                caption_columns,
                not args.skip_source_metadata,
            ): path
            for path, count in source_counts.items()
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            path, result = future.result()
            source_results[path] = result
            completed += 1
            if completed == 1 or completed % 32 == 0 or completed == len(futures):
                print(
                    f"[{args.domain}] source footers {completed}/{len(futures)}",
                    file=sys.stderr,
                    flush=True,
                )

    ordered_sources = [source_results[path] for path in sorted(source_results)]
    total_bytes = sum(int(source["bytes"]) for source in ordered_sources)
    max_bytes = max(int(source["bytes"]) for source in ordered_sources)
    parent_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "bytes": 0, "catalog_rows": 0}
    )
    for source in ordered_sources:
        parent = parent_stats[str(source["parent"])]
        parent["files"] += 1
        parent["bytes"] += int(source["bytes"])
        parent["catalog_rows"] += int(source["catalog_rows"])

    metadata_summary: dict[str, Any] | None = None
    if not args.skip_source_metadata:
        metadata_summary = {
            "source_rows": sum(int(source["parquet"]["rows"]) for source in ordered_sources),
            "source_row_groups": sum(
                int(source["parquet"]["row_groups"]) for source in ordered_sources
            ),
            "caption_compressed_bytes": sum(
                int(source["parquet"]["caption_compressed_bytes"])
                for source in ordered_sources
            ),
            "caption_uncompressed_bytes": sum(
                int(source["parquet"]["caption_uncompressed_bytes"])
                for source in ordered_sources
            ),
            "row_group_rows_min": min(
                int(source["parquet"]["row_group_rows_min"])
                for source in ordered_sources
            ),
            "row_group_rows_max": max(
                int(source["parquet"]["row_group_rows_max"])
                for source in ordered_sources
            ),
        }

    return {
        "schema_version": "offline_arb_caption_source_inventory_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain": args.domain,
        "catalog": {
            "root": str(catalog_root),
            "path": str(catalog_path),
            "catalog_hash": catalog_hash,
            "declared_sha256": declared_sha,
            **scan,
        },
        "caption_columns": list(caption_columns),
        "source_set_sha256": canonical_sha256(sorted(source_counts)),
        "source_count": len(ordered_sources),
        "source_total_bytes": total_bytes,
        "source_max_bytes": max_bytes,
        "full_stage_required_bytes_total_plus_max_plus_32gib": (
            total_bytes + max_bytes + 32 * (1 << 30)
        ),
        "catalog_rows_by_source_total": sum(source_counts.values()),
        "parents": {key: parent_stats[key] for key in sorted(parent_stats)},
        "metadata_summary": metadata_summary,
        "sources": ordered_sources,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=DOMAINS)
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=DEFAULT_CATALOG_ROOT,
    )
    parser.add_argument("--metadata-workers", type=require_positive_int, default=16)
    parser.add_argument(
        "--skip-source-metadata",
        action="store_true",
        help="stat sources but do not open parquet footers",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args)
        json.dump(report, sys.stdout, ensure_ascii=False, sort_keys=True, indent=2)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return 0
    except Exception as exc:
        print(
            f"FATAL caption source inventory ({args.domain}): "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
