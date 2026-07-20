#!/usr/bin/env python3
"""Read-only inventory of TextAtlas rows exposed by an offline plan.

The output bitmap is indexed by the frozen TextAtlas catalog-local row number.
It is an audit accelerator only; neither the plan nor any training artifact is
modified.  Coordinates use the plan/runtime's zero-based ``step`` convention.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np


BLOCK_RE = re.compile(r"block_(\d{6})\.npz")
DOMAIN_ORDER = ("anime", "monet_real", "gpic", "synthetic", "textatlas")


class AuditError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_block(
    path: Path,
    *,
    block_steps: int,
    total_steps: int,
    formal_steps: int,
    world_size: int,
    per_rank_bs: int,
    textatlas_base: int,
    textatlas_rows: int,
    target_row_id: int,
) -> dict[str, Any]:
    match = BLOCK_RE.fullmatch(path.name)
    if match is None:
        raise AuditError(f"bad block name: {path}")
    block_index = int(match.group(1))
    start = block_index * block_steps
    take = min(block_steps, total_steps - start)
    if take <= 0:
        raise AuditError(f"empty requested block: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"bucket_of_step", "item_row", "domain_counts"}:
            raise AuditError(f"block array set drift: {path}: {archive.files!r}")
        rows = np.asarray(archive["item_row"][:take], dtype=np.int64)
        buckets = np.asarray(archive["bucket_of_step"][:take])
    if rows.shape != (take, world_size, per_rank_bs):
        raise AuditError(f"item_row shape drift: {path}: {rows.shape}")
    if buckets.shape != (take,):
        raise AuditError(f"bucket_of_step shape drift: {path}: {buckets.shape}")

    text_mask = (rows >= textatlas_base) & (
        rows < textatlas_base + textatlas_rows
    )
    total_offsets = np.unique(rows[text_mask] - textatlas_base).astype(
        np.int32, copy=False
    )
    formal_take = max(0, min(take, formal_steps - start))
    if formal_take:
        formal_rows = rows[:formal_take]
        formal_mask = (formal_rows >= textatlas_base) & (
            formal_rows < textatlas_base + textatlas_rows
        )
        formal_offsets = np.unique(
            formal_rows[formal_mask] - textatlas_base
        ).astype(np.int32, copy=False)
        formal_exposures = int(np.count_nonzero(formal_mask))
    else:
        formal_offsets = np.empty(0, dtype=np.int32)
        formal_exposures = 0

    coordinates = []
    for local_step, rank, slot in np.argwhere(rows == target_row_id).tolist():
        step = start + int(local_step)
        coordinates.append(
            {
                "step": step,
                "rank": int(rank),
                "slot": int(slot),
                "bucket_idx": int(buckets[local_step]),
                "within_formal_166667": step < formal_steps,
            }
        )
    return {
        "block_index": block_index,
        "steps": take,
        "formal_steps": formal_take,
        "textatlas_total_exposures": int(np.count_nonzero(text_mask)),
        "textatlas_formal_exposures": formal_exposures,
        "total_offsets": total_offsets,
        "formal_offsets": formal_offsets,
        "target_coordinates": coordinates,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", required=True, type=Path)
    parser.add_argument("--catalog-root", required=True, type=Path)
    parser.add_argument("--formal-steps", type=int, default=166667)
    parser.add_argument("--target-row-id", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-bitmap", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _args()
    plan_root = args.plan_root.expanduser().resolve(strict=True)
    catalog_root = args.catalog_root.expanduser().resolve(strict=True)
    plan_header_path = plan_root / "header.json"
    catalog_header_path = catalog_root / "header.json"
    plan = _json(plan_header_path)
    catalog = _json(catalog_header_path)
    if tuple(catalog.get("domain_order", ())) != DOMAIN_ORDER:
        raise AuditError("catalog domain_order drift")
    if int(plan.get("world_size", 0)) != 64 or int(plan.get("per_rank_bs", 0)) != 36:
        raise AuditError("expected frozen w64/per-rank-bs36 plan")
    total_steps = int(plan["total_steps"])
    block_steps = int(plan["block_steps"])
    formal_steps = int(args.formal_steps)
    if not 0 < formal_steps <= total_steps:
        raise AuditError("formal step bound is outside the plan")
    cursor = 0
    ranges: dict[str, tuple[int, int]] = {}
    for domain in DOMAIN_ORDER:
        rows = int(catalog["files"][domain]["rows"])
        ranges[domain] = (cursor, rows)
        cursor += rows
    if cursor != int(catalog["row_count"]):
        raise AuditError("catalog row range drift")
    textatlas_base, textatlas_rows = ranges["textatlas"]
    if not textatlas_base <= args.target_row_id < textatlas_base + textatlas_rows:
        raise AuditError("target row is not TextAtlas")

    needed_blocks = (total_steps + block_steps - 1) // block_steps
    paths = [
        plan_root / "blocks" / f"block_{index:06d}.npz"
        for index in range(needed_blocks)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise AuditError(f"missing plan blocks: {missing[:8]!r}")

    total_bitmap = np.zeros(textatlas_rows, dtype=np.bool_)
    formal_bitmap = np.zeros(textatlas_rows, dtype=np.bool_)
    target_coordinates: list[dict[str, Any]] = []
    scanned_steps = 0
    scanned_formal_steps = 0
    total_exposures = 0
    formal_exposures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _scan_block,
                path,
                block_steps=block_steps,
                total_steps=total_steps,
                formal_steps=formal_steps,
                world_size=int(plan["world_size"]),
                per_rank_bs=int(plan["per_rank_bs"]),
                textatlas_base=textatlas_base,
                textatlas_rows=textatlas_rows,
                target_row_id=int(args.target_row_id),
            )
            for path in paths
        ]
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            scanned_steps += int(item["steps"])
            scanned_formal_steps += int(item["formal_steps"])
            total_exposures += int(item["textatlas_total_exposures"])
            formal_exposures += int(item["textatlas_formal_exposures"])
            total_bitmap[item["total_offsets"]] = True
            formal_bitmap[item["formal_offsets"]] = True
            target_coordinates.extend(item["target_coordinates"])
    if scanned_steps != total_steps or scanned_formal_steps != formal_steps:
        raise AuditError(
            f"step coverage drift total={scanned_steps}/{total_steps}, "
            f"formal={scanned_formal_steps}/{formal_steps}"
        )
    target_coordinates.sort(key=lambda x: (x["step"], x["rank"], x["slot"]))

    output_json = args.output_json.expanduser().resolve(strict=False)
    output_bitmap = args.output_bitmap.expanduser().resolve(strict=False)
    for path in (output_json, output_bitmap):
        if path.exists() or path.is_symlink():
            raise AuditError(f"refusing to overwrite output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    bitmap_tmp = output_bitmap.with_name(output_bitmap.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(
        bitmap_tmp,
        textatlas_total=total_bitmap,
        textatlas_formal=formal_bitmap,
        textatlas_base=np.array([textatlas_base], dtype=np.int64),
        textatlas_rows=np.array([textatlas_rows], dtype=np.int64),
        total_steps=np.array([total_steps], dtype=np.int64),
        formal_steps=np.array([formal_steps], dtype=np.int64),
    )
    os.replace(bitmap_tmp, output_bitmap)
    bitmap_sha = _sha256(output_bitmap)
    result = {
        "status": "PASS",
        "audit": "offline_arb_textatlas_plan_rows_v1",
        "read_only": True,
        "plan": {
            "root": str(plan_root),
            "plan_id": plan.get("plan_id"),
            "plan_hash": plan.get("plan_hash"),
            "header_sha256": _sha256(plan_header_path),
            "total_steps": total_steps,
            "formal_steps": formal_steps,
            "world_size": int(plan["world_size"]),
            "per_rank_bs": int(plan["per_rank_bs"]),
        },
        "catalog": {
            "root": str(catalog_root),
            "catalog_hash": catalog.get("catalog_hash"),
            "header_sha256": _sha256(catalog_header_path),
            "textatlas_row_base": textatlas_base,
            "textatlas_rows": textatlas_rows,
        },
        "textatlas": {
            "total_plan_exposures": total_exposures,
            "formal_plan_exposures": formal_exposures,
            "total_plan_unique_rows": int(np.count_nonzero(total_bitmap)),
            "formal_plan_unique_rows": int(np.count_nonzero(formal_bitmap)),
        },
        "target": {
            "row_id": int(args.target_row_id),
            "total_plan_exposures": len(target_coordinates),
            "formal_plan_exposures": sum(
                int(item["within_formal_166667"]) for item in target_coordinates
            ),
            "coordinates": target_coordinates,
        },
        "bitmap": {"path": str(output_bitmap), "sha256": bitmap_sha},
    }
    json_tmp = output_json.with_name(output_json.name + f".tmp.{os.getpid()}")
    json_tmp.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(json_tmp, output_json)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
