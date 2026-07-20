#!/usr/bin/env python3
"""Read-only sharded audit of planned TextAtlas payload byte integrity.

Only catalog rows present in the supplied plan bitmap are inspected.  Source
Parquet files are assigned deterministically across shards; every selected row
is checked for the incident signature (exactly 32768 bytes and no JPEG EOI),
and signature matches receive a strict Pillow decode diagnostic.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


class AuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"JSON root is not an object: {path}")
    return value


def _assigned(path: str, shard_index: int, shard_count: int) -> bool:
    token = hashlib.sha256(path.encode("utf-8")).digest()
    return int.from_bytes(token[:8], "big") % shard_count == shard_index


def _catalog_tasks(
    catalog_path: Path,
    bitmap: np.ndarray,
    *,
    textatlas_base: int,
    shard_index: int,
    shard_count: int,
) -> tuple[dict[str, dict[int, list[tuple[int, int, str, str]]]], dict[str, int]]:
    parquet = pq.ParquetFile(catalog_path)
    tasks: dict[str, dict[int, list[tuple[int, int, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    catalog_rows = 0
    selected_plan_rows = 0
    assigned_plan_rows = 0
    assigned_paths: set[str] = set()
    all_paths: set[str] = set()
    columns = [
        "row_id",
        "uid",
        "image_path",
        "image_parquet_path",
        "image_row_group",
        "image_row_in_group",
    ]
    for row_group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(row_group, columns=columns, use_threads=False)
        rows = table.to_pylist()
        catalog_rows += len(rows)
        for row in rows:
            row_id = int(row["row_id"])
            local = row_id - textatlas_base
            if local < 0 or local >= bitmap.size:
                raise AuditError(f"catalog TextAtlas row outside bitmap: {row_id}")
            path = str(row["image_parquet_path"])
            all_paths.add(path)
            if not bool(bitmap[local]):
                continue
            selected_plan_rows += 1
            if not _assigned(path, shard_index, shard_count):
                continue
            assigned_plan_rows += 1
            assigned_paths.add(path)
            source_group = int(row["image_row_group"])
            source_row = int(row["image_row_in_group"])
            tasks[path][source_group].append(
                (source_row, row_id, str(row["uid"]), str(row["image_path"]))
            )
    return dict(tasks), {
        "catalog_rows": catalog_rows,
        "selected_plan_unique_rows_all_shards": selected_plan_rows,
        "assigned_plan_unique_rows": assigned_plan_rows,
        "source_files_all_shards": len(all_paths),
        "assigned_source_files": len(assigned_paths),
        "assigned_source_row_groups": sum(len(value) for value in tasks.values()),
    }


def _strict_pillow(raw: bytes) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image_format = image.format
            image_size = [int(image.size[0]), int(image.size[1])]
            image_mode = image.mode
            # Match runtime decode_rgb rather than Pillow.verify(): conversion
            # forces the decoder to consume the full compressed pixel stream.
            rgb = image.convert("RGB")
            rgb.load()
        return {
            "strict_decode_ok": True,
            "format": image_format,
            "decoded_size": image_size,
            "mode": image_mode,
            "error_type": None,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - audit records exact decoder state
        return {
            "strict_decode_ok": False,
            "format": None,
            "decoded_size": None,
            "mode": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _scan_source_file(
    path: str,
    groups: dict[int, list[tuple[int, int, str, str]]],
    signature_mode: str,
) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    checked = 0
    exact_32768 = 0
    exact_32768_with_eoi = 0
    jpeg_rows = 0
    jpeg_missing_eoi = 0
    strict_decode_rows = 0
    strict_decode_failures = 0
    candidates: list[dict[str, Any]] = []
    for source_group in sorted(groups):
        if source_group < 0 or source_group >= parquet.metadata.num_row_groups:
            raise AuditError(f"source row-group outside file: {path}#{source_group}")
        table = parquet.read_row_group(
            source_group, columns=["image", "image_path"], use_threads=False
        ).combine_chunks()
        image_struct = table.column("image").chunk(0)
        image_bytes = image_struct.field("bytes")
        image_paths = table.column("image_path").chunk(0)
        for source_row, row_id, uid, expected_image_path in groups[source_group]:
            if source_row < 0 or source_row >= table.num_rows:
                raise AuditError(
                    f"source row outside row-group: {path}#{source_group}:{source_row}"
                )
            observed_image_path = image_paths[source_row].as_py()
            if str(observed_image_path) != expected_image_path:
                raise AuditError(
                    f"image_path locator mismatch row_id={row_id}: "
                    f"catalog={expected_image_path!r}, source={observed_image_path!r}"
                )
            raw = image_bytes[source_row].as_py()
            if not isinstance(raw, bytes) or not raw:
                raise AuditError(f"empty/non-bytes source payload row_id={row_id}")
            checked += 1
            has_jpeg_soi = raw.startswith(b"\xff\xd8\xff")
            has_eoi = raw.endswith(b"\xff\xd9")
            if has_jpeg_soi:
                jpeg_rows += 1
                if not has_eoi:
                    jpeg_missing_eoi += 1
            if len(raw) == 32768:
                exact_32768 += 1
                if has_eoi:
                    exact_32768_with_eoi += 1
            if signature_mode == "exact_32768_missing_eoi":
                selected = len(raw) == 32768 and not has_eoi
            elif signature_mode == "jpeg_missing_eoi":
                selected = has_jpeg_soi and not has_eoi
            elif signature_mode == "strict_decode_failure_all":
                strict_decode_rows += 1
                diagnostic = _strict_pillow(raw)
                selected = not bool(diagnostic["strict_decode_ok"])
                if selected:
                    strict_decode_failures += 1
            else:
                raise AuditError(f"unknown signature mode: {signature_mode}")
            if not selected:
                continue
            if signature_mode != "strict_decode_failure_all":
                diagnostic = _strict_pillow(raw)
            candidates.append(
                {
                    "row_id": row_id,
                    "uid": uid,
                    "image_path": expected_image_path,
                    "image_parquet_path": path,
                    "image_row_group": source_group,
                    "image_row_in_group": source_row,
                    "payload_bytes": len(raw),
                    "payload_sha256": hashlib.sha256(raw).hexdigest(),
                    "head_hex": raw[:16].hex(),
                    "tail_hex": raw[-16:].hex(),
                    "has_jpeg_soi": has_jpeg_soi,
                    "has_jpeg_eoi": has_eoi,
                    **diagnostic,
                }
            )
    return {
        "path": path,
        "checked_plan_unique_rows": checked,
        "exact_32768_rows": exact_32768,
        "exact_32768_with_eoi_rows": exact_32768_with_eoi,
        "jpeg_rows": jpeg_rows,
        "jpeg_missing_eoi_rows": jpeg_missing_eoi,
        "strict_decode_rows": strict_decode_rows,
        "strict_decode_failures": strict_decode_failures,
        "signature_candidates": candidates,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", required=True, type=Path)
    parser.add_argument("--plan-json", required=True, type=Path)
    parser.add_argument("--plan-bitmap", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--signature-mode",
        choices=(
            "exact_32768_missing_eoi",
            "jpeg_missing_eoi",
            "strict_decode_failure_all",
        ),
        default="exact_32768_missing_eoi",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if not 0 <= args.shard_index < args.shard_count:
        raise AuditError("invalid shard index/count")
    if args.workers <= 0:
        raise AuditError("workers must be positive")
    catalog_root = args.catalog_root.expanduser().resolve(strict=True)
    plan_json_path = args.plan_json.expanduser().resolve(strict=True)
    plan_bitmap_path = args.plan_bitmap.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise AuditError(f"refusing to overwrite output: {output}")
    plan_json = _json(plan_json_path)
    if plan_json.get("status") != "PASS" or plan_json.get("audit") != "offline_arb_textatlas_plan_rows_v1":
        raise AuditError("plan audit identity/status drift")
    if _sha256(plan_bitmap_path) != plan_json["bitmap"]["sha256"]:
        raise AuditError("plan bitmap SHA drift")
    with np.load(plan_bitmap_path, allow_pickle=False) as archive:
        if "textatlas_total" not in archive.files:
            raise AuditError("plan bitmap array missing")
        bitmap = np.asarray(archive["textatlas_total"], dtype=np.bool_)
        textatlas_base = int(np.asarray(archive["textatlas_base"])[0])
        textatlas_rows = int(np.asarray(archive["textatlas_rows"])[0])
    if bitmap.shape != (textatlas_rows,):
        raise AuditError(f"plan bitmap shape drift: {bitmap.shape}")
    if textatlas_base != int(plan_json["catalog"]["textatlas_row_base"]):
        raise AuditError("plan bitmap/catalog base drift")
    catalog_header_path = catalog_root / "header.json"
    catalog_header = _json(catalog_header_path)
    if _sha256(catalog_header_path) != plan_json["catalog"]["header_sha256"]:
        raise AuditError("live catalog header SHA drift")
    catalog_path = catalog_root / str(catalog_header["files"]["textatlas"]["name"])
    tasks, inventory = _catalog_tasks(
        catalog_path,
        bitmap,
        textatlas_base=textatlas_base,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(
        json.dumps(
            {"phase": "catalog_complete", "shard": args.shard_index, **inventory},
            sort_keys=True,
        ),
        flush=True,
    )

    results: list[dict[str, Any]] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_scan_source_file, path, groups, args.signature_mode): path
            for path, groups in tasks.items()
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 5 == 0 or completed == len(futures):
                checked = sum(item["checked_plan_unique_rows"] for item in results)
                candidates = sum(len(item["signature_candidates"]) for item in results)
                print(
                    json.dumps(
                        {
                            "phase": "source_progress",
                            "shard": args.shard_index,
                            "files_complete": completed,
                            "files_total": len(futures),
                            "rows_checked": checked,
                            "signature_candidates": candidates,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    results.sort(key=lambda item: item["path"])
    checked = sum(item["checked_plan_unique_rows"] for item in results)
    if checked != inventory["assigned_plan_unique_rows"]:
        raise AuditError(
            f"source row coverage drift: {checked} != {inventory['assigned_plan_unique_rows']}"
        )
    candidates = [
        candidate
        for item in results
        for candidate in item["signature_candidates"]
    ]
    candidates.sort(key=lambda item: int(item["row_id"]))
    result = {
        "status": "PASS",
        "audit": "offline_arb_textatlas_planned_payloads_v2",
        "read_only": True,
        "signature_mode": args.signature_mode,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "workers": args.workers,
        "provenance": {
            "catalog_header_sha256": _sha256(catalog_header_path),
            "catalog_textatlas_sha256": catalog_header["files"]["textatlas"]["sha256"],
            "plan_json": str(plan_json_path),
            "plan_json_sha256": _sha256(plan_json_path),
            "plan_bitmap": str(plan_bitmap_path),
            "plan_bitmap_sha256": _sha256(plan_bitmap_path),
        },
        "inventory": inventory,
        "summary": {
            "source_files_checked": len(results),
            "source_row_groups_checked": inventory["assigned_source_row_groups"],
            "planned_unique_rows_checked": checked,
            "exact_32768_rows": sum(item["exact_32768_rows"] for item in results),
            "exact_32768_with_eoi_rows": sum(
                item["exact_32768_with_eoi_rows"] for item in results
            ),
            "jpeg_rows": sum(item["jpeg_rows"] for item in results),
            "jpeg_missing_eoi_rows": sum(
                item["jpeg_missing_eoi_rows"] for item in results
            ),
            "strict_decode_rows": sum(item["strict_decode_rows"] for item in results),
            "strict_decode_failures": sum(
                item["strict_decode_failures"] for item in results
            ),
            "signature_candidates": len(candidates),
            "strict_decode_failures_among_candidates": sum(
                not bool(item["strict_decode_ok"]) for item in candidates
            ),
        },
        "signature_candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({"status": "PASS", "output": str(output), **result["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
