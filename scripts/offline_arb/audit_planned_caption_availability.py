#!/usr/bin/env python3
"""Audit caption availability for every slot consumed by the formal plan.

This is a read-only incident diagnostic.  It first finds catalog rows for which
the runtime ``_select_band`` availability set is empty, then scans the requested
plan prefix and records their exact exposure count and first/last coordinate.
For planned MONET defects it also reads the frozen caption sidecar and reports
which of the five physical caption fields are actually non-empty.

The script deliberately does not propose or apply runtime fallback semantics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.offline_arb.caption_sidecar_contract import CAPTION_SCHEMAS, DOMAINS
from training.dataline_offline.caption_sidecar import CaptionSidecarStore


CAPTION_FIELDS = (
    "caption_gemini-2.5-flash-lite",
    "caption_sharegpt4v-7b",
    "caption_internvl-3-8b",
    "caption_florence-2-large",
    "caption_original",
)
BAND_COLUMNS = tuple(f"band_captioner_t{i}" for i in range(1, 5))
AUDITED_DOMAINS = ("monet_real", "gpic")
BLOCK_RE = re.compile(r"block_(\d{6})\.npz")


class AuditError(RuntimeError):
    pass


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AuditError(f"{label} must be a positive integer, got {value!r}")
    return int(value)


def _catalog_ranges(header: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    order = header.get("domain_order")
    files = header.get("files")
    if order != list(DOMAINS) or not isinstance(files, dict):
        raise AuditError("catalog domain order/files contract drift")
    cursor = 0
    result: dict[str, tuple[int, int]] = {}
    for domain in DOMAINS:
        entry = files.get(domain)
        if not isinstance(entry, dict):
            raise AuditError(f"catalog files.{domain} missing")
        rows = _positive_int(entry.get("rows"), f"catalog files.{domain}.rows")
        result[domain] = (cursor, rows)
        cursor += rows
    if cursor != header.get("row_count"):
        raise AuditError(f"catalog row_count drift: ranges={cursor}, header={header.get('row_count')!r}")
    return result


def _catalog_unavailable_rows(
    catalog_root: Path,
    catalog_header: Mapping[str, Any],
    ranges: Mapping[str, tuple[int, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    detail_columns = [
        "row_id",
        "uid",
        "bucket_idx",
        "width",
        "height",
        "src_key",
        "sub_source",
        "caption_type",
        "caption_tier",
        "served_mask",
        *BAND_COLUMNS,
        "base_weight",
    ]
    numeric_columns = ["row_id", "served_mask", *BAND_COLUMNS]

    for domain in AUDITED_DOMAINS:
        file_entry = catalog_header["files"][domain]
        path = catalog_root / str(file_entry["name"])
        parquet = pq.ParquetFile(path)
        expected_base, expected_rows = ranges[domain]
        unavailable_count = 0
        served_code_mismatch_count = 0
        observed_rows = 0
        domain_rows: list[dict[str, Any]] = []

        for row_group in range(parquet.metadata.num_row_groups):
            numeric = parquet.read_row_group(
                row_group, columns=numeric_columns, use_threads=False
            )
            row_ids = numeric.column("row_id").to_numpy(zero_copy_only=False)
            served = numeric.column("served_mask").to_numpy(zero_copy_only=False).astype(
                np.uint8, copy=False
            )
            codes = np.stack(
                [
                    numeric.column(column).to_numpy(zero_copy_only=False)
                    for column in BAND_COLUMNS
                ],
                axis=1,
            )
            bits = np.left_shift(np.uint8(1), np.arange(4, dtype=np.uint8))
            derived = (((codes > 0).astype(np.uint8) * bits).sum(axis=1)).astype(np.uint8)
            served_code_mismatch_count += int(np.count_nonzero(served != derived))
            available = np.any((codes > 0) & ((served[:, None] & bits) != 0), axis=1)
            bad_offsets = np.flatnonzero(~available)
            observed_rows += int(numeric.num_rows)
            if bad_offsets.size == 0:
                continue

            detail = parquet.read_row_group(
                row_group, columns=detail_columns, use_threads=False
            )
            for offset in bad_offsets.tolist():
                row = detail.slice(int(offset), 1).to_pylist()[0]
                row["domain"] = domain
                row["catalog_row_group"] = int(row_group)
                row["catalog_row_in_group"] = int(offset)
                row["available_bands"] = [
                    index
                    for index, column in enumerate(BAND_COLUMNS)
                    if int(row[column]) > 0 and ((int(row["served_mask"]) >> index) & 1)
                ]
                if row["available_bands"]:
                    raise AuditError(f"internal availability audit error: {row!r}")
                domain_rows.append(row)

        if observed_rows != expected_rows:
            raise AuditError(
                f"catalog row-count drift for {domain}: {observed_rows} != {expected_rows}"
            )
        expected_ids = [int(row["row_id"]) for row in domain_rows]
        if any(row_id < expected_base or row_id >= expected_base + expected_rows for row_id in expected_ids):
            raise AuditError(f"catalog unavailable row outside {domain} range")
        if len(expected_ids) != len(set(expected_ids)):
            raise AuditError(f"duplicate unavailable catalog row_id in {domain}")
        unavailable_count = len(domain_rows)
        summary[domain] = {
            "catalog_rows": expected_rows,
            "unavailable_unique_rows": unavailable_count,
            "unavailable_fraction": unavailable_count / expected_rows,
            "served_code_mismatch_count": served_code_mismatch_count,
        }
        all_rows.extend(domain_rows)

    all_rows.sort(key=lambda row: int(row["row_id"]))
    return all_rows, summary


def _block_index(path: Path) -> int:
    match = BLOCK_RE.fullmatch(path.name)
    if match is None:
        raise AuditError(f"unexpected plan block name: {path}")
    return int(match.group(1))


def _scan_block(
    path: Path,
    *,
    block_steps: int,
    requested_steps: int,
    world_size: int,
    per_rank_bs: int,
    bad_ids: np.ndarray,
) -> dict[str, Any]:
    index = _block_index(path)
    start = index * block_steps
    stop = min(start + block_steps, requested_steps)
    take = stop - start
    if take <= 0:
        return {"index": index, "steps": 0, "counts": {}, "first": {}, "last": {}}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"bucket_of_step", "item_row", "domain_counts"}:
                raise AuditError(f"plan block dataset set drift: {path}: {archive.files!r}")
            full_rows = archive["item_row"]
            full_buckets = archive["bucket_of_step"]
            full_domain_counts = archive["domain_counts"]
            if (
                full_rows.ndim != 3
                or full_rows.shape[0] < take
                or full_rows.shape[1:] != (world_size, per_rank_bs)
            ):
                raise AuditError(
                    f"plan block item_row shape drift: {path}: {full_rows.shape}"
                )
            full_steps = int(full_rows.shape[0])
            if full_rows.dtype != np.dtype(np.int64):
                raise AuditError(
                    f"plan block item_row dtype drift: {path}: {full_rows.dtype}"
                )
            if full_buckets.shape != (full_steps,) or full_buckets.dtype != np.dtype(np.int16):
                raise AuditError(
                    "plan block bucket_of_step shape/dtype drift: "
                    f"{path}: shape={full_buckets.shape}, dtype={full_buckets.dtype}"
                )
            if (
                full_domain_counts.shape != (full_steps, len(DOMAINS))
                or full_domain_counts.dtype != np.dtype(np.int32)
            ):
                raise AuditError(
                    "plan block domain_counts shape/dtype drift: "
                    f"{path}: shape={full_domain_counts.shape}, "
                    f"dtype={full_domain_counts.dtype}"
                )
            rows = np.asarray(full_rows[:take])
            if rows.shape != (take, world_size, per_rank_bs):
                raise AuditError(
                    f"plan block item_row truncated slice: {path}: {rows.shape}"
                )
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError(f"cannot read plan block {path}: {exc}") from exc

    if bad_ids.size == 0:
        return {
            "index": index,
            "steps": take,
            "counts": {},
            "first": {},
            "last": {},
        }
    positions = np.searchsorted(bad_ids, rows)
    in_range = positions < bad_ids.size
    safe_positions = np.minimum(positions, bad_ids.size - 1)
    matched = in_range & (bad_ids[safe_positions] == rows)
    coordinates = np.argwhere(matched)
    counts: Counter[int] = Counter()
    first: dict[int, tuple[int, int, int]] = {}
    last: dict[int, tuple[int, int, int]] = {}
    for local_step, rank, slot in coordinates.tolist():
        row_id = int(rows[local_step, rank, slot])
        coordinate = (start + int(local_step), int(rank), int(slot))
        counts[row_id] += 1
        first.setdefault(row_id, coordinate)
        last[row_id] = coordinate
    return {
        "index": index,
        "steps": take,
        "counts": dict(counts),
        "first": first,
        "last": last,
    }


def _scan_plan(
    plan_root: Path,
    plan_header: Mapping[str, Any],
    *,
    requested_steps: int,
    workers: int,
    bad_row_ids: Iterable[int],
) -> dict[str, Any]:
    total_steps = _positive_int(plan_header.get("total_steps"), "plan total_steps")
    block_steps = _positive_int(plan_header.get("block_steps"), "plan block_steps")
    world_size = _positive_int(plan_header.get("world_size"), "plan world_size")
    per_rank_bs = _positive_int(plan_header.get("per_rank_bs"), "plan per_rank_bs")
    if requested_steps > total_steps:
        raise AuditError(f"requested steps {requested_steps} exceed plan total_steps {total_steps}")
    bad_ids = np.array(sorted(set(int(value) for value in bad_row_ids)), dtype=np.int64)

    needed_blocks = (requested_steps + block_steps - 1) // block_steps
    paths = [plan_root / "blocks" / f"block_{index:06d}.npz" for index in range(needed_blocks)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise AuditError(f"missing required plan blocks: {missing[:8]!r}")

    counts: Counter[int] = Counter()
    first: dict[int, tuple[int, int, int]] = {}
    last: dict[int, tuple[int, int, int]] = {}
    scanned_steps = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _scan_block,
                path,
                block_steps=block_steps,
                requested_steps=requested_steps,
                world_size=world_size,
                per_rank_bs=per_rank_bs,
                bad_ids=bad_ids,
            )
            for path in paths
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            scanned_steps += int(result["steps"])
            counts.update({int(key): int(value) for key, value in result["counts"].items()})
            for raw_row_id, coordinate in result["first"].items():
                row_id = int(raw_row_id)
                value = tuple(int(item) for item in coordinate)
                if row_id not in first or value < first[row_id]:
                    first[row_id] = value
            for raw_row_id, coordinate in result["last"].items():
                row_id = int(raw_row_id)
                value = tuple(int(item) for item in coordinate)
                if row_id not in last or value > last[row_id]:
                    last[row_id] = value
    if scanned_steps != requested_steps:
        raise AuditError(f"plan scan step-count drift: {scanned_steps} != {requested_steps}")

    return {
        "steps": requested_steps,
        "world_size": world_size,
        "per_rank_bs": per_rank_bs,
        "global_batch_size": world_size * per_rank_bs,
        "exposures": requested_steps * world_size * per_rank_bs,
        "blocks": needed_blocks,
        "workers": workers,
        "counts": dict(counts),
        "first": first,
        "last": last,
    }


def _open_sidecar(
    sidecar_root: Path,
    catalog_root: Path,
    catalog_header: Mapping[str, Any],
    ranges: Mapping[str, tuple[int, int]],
    workers: int,
) -> CaptionSidecarStore:
    expected = {
        domain: {
            "row_base": ranges[domain][0],
            "row_count": ranges[domain][1],
            "columns": tuple(CAPTION_SCHEMAS[domain].names),
            "catalog_file": catalog_header["files"][domain],
        }
        for domain in DOMAINS
    }
    return CaptionSidecarStore(
        sidecar_root,
        catalog_hash=str(catalog_header["catalog_hash"]),
        catalog_header_sha256=_sha256(catalog_root / "header.json"),
        expected_domains=expected,
        workers=workers,
    )


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sidecar_report(
    store: CaptionSidecarStore,
    planned_bad_monet: list[dict[str, Any]],
) -> dict[str, Any]:
    row_ids = [int(row["row_id"]) for row in planned_bad_monet]
    rows = store.get_many("monet_real", row_ids) if row_ids else {}
    unique_masks: Counter[str] = Counter()
    exposure_masks: Counter[str] = Counter()
    fields_unique = Counter({field: 0 for field in CAPTION_FIELDS})
    fields_exposure = Counter({field: 0 for field in CAPTION_FIELDS})
    details = []
    for catalog_row in planned_bad_monet:
        row_id = int(catalog_row["row_id"])
        values = rows[row_id]
        flags = [_nonempty(values.get(field)) for field in CAPTION_FIELDS]
        mask = sum((1 << index) for index, flag in enumerate(flags) if flag)
        key = f"0b{mask:05b}"
        exposures = int(catalog_row["plan_exposures"])
        unique_masks[key] += 1
        exposure_masks[key] += exposures
        for field, flag in zip(CAPTION_FIELDS, flags):
            if flag:
                fields_unique[field] += 1
                fields_exposure[field] += exposures
        details.append(
            {
                "row_id": row_id,
                "uid": catalog_row["uid"],
                "plan_exposures": exposures,
                "nonempty_mask": key,
                "nonempty_fields": [
                    field for field, flag in zip(CAPTION_FIELDS, flags) if flag
                ],
                "field_states": {
                    field: (
                        "nonempty"
                        if flag
                        else "null"
                        if values.get(field) is None
                        else "empty_or_whitespace"
                    )
                    for field, flag in zip(CAPTION_FIELDS, flags)
                },
            }
        )
    return {
        "field_order": list(CAPTION_FIELDS),
        "mask_definition": "bit i means field_order[i] is a non-empty stripped string",
        "planned_unavailable_unique_rows": len(details),
        "planned_unavailable_exposures": sum(item["plan_exposures"] for item in details),
        "unique_row_mask_distribution": dict(sorted(unique_masks.items())),
        "exposure_weighted_mask_distribution": dict(sorted(exposure_masks.items())),
        "nonempty_unique_rows_by_field": dict(fields_unique),
        "nonempty_exposures_by_field": dict(fields_exposure),
        "rows": details,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", required=True, type=Path)
    parser.add_argument("--catalog-root", required=True, type=Path)
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--known-answer-uid", default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.workers <= 0:
        parser.error("--steps and --workers must be positive")
    return args


def main() -> int:
    args = _args()
    plan_root = args.plan_root.expanduser().resolve(strict=True)
    catalog_root = args.catalog_root.expanduser().resolve(strict=True)
    sidecar_root = args.sidecar_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    for root, label in (
        (plan_root, "plan root"),
        (catalog_root, "catalog root"),
        (sidecar_root, "sidecar root"),
    ):
        if not root.is_dir():
            raise AuditError(f"{label} is not a directory: {root}")
    if output.exists():
        raise AuditError(f"refusing to overwrite output: {output}")

    plan_header_path = plan_root / "header.json"
    catalog_header_path = catalog_root / "header.json"
    sidecar_header_path = sidecar_root / "header.json"
    plan_header = _strict_json(plan_header_path)
    catalog_header = _strict_json(catalog_header_path)
    ranges = _catalog_ranges(catalog_header)
    unavailable, catalog_summary = _catalog_unavailable_rows(
        catalog_root, catalog_header, ranges
    )
    scan = _scan_plan(
        plan_root,
        plan_header,
        requested_steps=args.steps,
        workers=args.workers,
        bad_row_ids=[int(row["row_id"]) for row in unavailable],
    )

    by_id = {int(row["row_id"]): row for row in unavailable}
    for row_id, row in by_id.items():
        row["plan_exposures"] = int(scan["counts"].get(row_id, 0))
        first = scan["first"].get(row_id)
        last = scan["last"].get(row_id)
        row["first_plan_coordinate"] = (
            {"step": first[0], "rank": first[1], "slot": first[2]} if first else None
        )
        row["last_plan_coordinate"] = (
            {"step": last[0], "rank": last[1], "slot": last[2]} if last else None
        )

    planned = [row for row in unavailable if int(row["plan_exposures"]) > 0]
    planned.sort(
        key=lambda row: (
            int(row["first_plan_coordinate"]["step"]),
            int(row["first_plan_coordinate"]["rank"]),
            int(row["first_plan_coordinate"]["slot"]),
        )
    )
    store = _open_sidecar(
        sidecar_root, catalog_root, catalog_header, ranges, args.workers
    )
    try:
        sidecar = _sidecar_report(
            store, [row for row in planned if row["domain"] == "monet_real"]
        )
    finally:
        store.close(wait=True)

    known_answer = None
    if args.known_answer_uid is not None:
        matches = [row for row in unavailable if row["uid"] == args.known_answer_uid]
        if len(matches) != 1:
            raise AuditError(
                f"known-answer UID must identify exactly one unavailable catalog row: "
                f"{args.known_answer_uid!r}, matches={len(matches)}"
            )
        match = matches[0]
        sidecar_match = [row for row in sidecar["rows"] if row["row_id"] == match["row_id"]]
        if int(match["plan_exposures"]) <= 0 or len(sidecar_match) != 1:
            raise AuditError("known-answer row is not exposed with a sidecar record")
        known_answer = {
            "uid": match["uid"],
            "row_id": match["row_id"],
            "plan_exposures": match["plan_exposures"],
            "first_plan_coordinate": match["first_plan_coordinate"],
            "sidecar_nonempty_mask": sidecar_match[0]["nonempty_mask"],
            "pass": sidecar_match[0]["nonempty_mask"] == "0b00000",
        }
        if not known_answer["pass"]:
            raise AuditError(f"known-answer sidecar unexpectedly has a non-empty caption: {known_answer}")

    domain_plan_summary = {}
    for domain in AUDITED_DOMAINS:
        rows = [row for row in unavailable if row["domain"] == domain]
        hit = [row for row in rows if int(row["plan_exposures"]) > 0]
        first_coordinate = min(
            (row["first_plan_coordinate"] for row in hit),
            key=lambda value: (
                int(value["step"]), int(value["rank"]), int(value["slot"])
            ),
            default=None,
        )
        domain_plan_summary[domain] = {
            "catalog_unavailable_unique_rows": len(rows),
            "planned_unavailable_unique_rows": len(hit),
            "planned_unavailable_exposures": sum(int(row["plan_exposures"]) for row in hit),
            "first_plan_coordinate": first_coordinate,
        }

    result = {
        "status": "PASS",
        "audit": "offline_arb_planned_caption_availability_v1",
        "read_only": True,
        "provenance": {
            "plan_root": str(plan_root),
            "plan_id": plan_header.get("plan_id"),
            "plan_hash": plan_header.get("plan_hash"),
            "plan_header_sha256": _sha256(plan_header_path),
            "catalog_root": str(catalog_root),
            "catalog_hash": catalog_header.get("catalog_hash"),
            "catalog_header_sha256": _sha256(catalog_header_path),
            "sidecar_root": str(sidecar_root),
            "sidecar_header_sha256": _sha256(sidecar_header_path),
        },
        "scan": {key: value for key, value in scan.items() if key not in {"counts", "first", "last"}},
        "catalog_summary": catalog_summary,
        "plan_summary": domain_plan_summary,
        "known_answer": known_answer,
        "monet_sidecar": sidecar,
        "unavailable_rows": unavailable,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({
        "status": "PASS",
        "output": str(output),
        "scanned_exposures": scan["exposures"],
        "plan_summary": domain_plan_summary,
        "known_answer": known_answer,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
