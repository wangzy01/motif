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
LEGACY_COORDINATE_SCHEMA = "stage512a_w48_v1"
COORDINATE_SCHEMA_FIELDS = {
    LEGACY_COORDINATE_SCHEMA: "within_formal_166667",
    "generic_v1": "within_formal",
}
COORDINATE_SCHEMA_AUDIT_NAMES = {
    LEGACY_COORDINATE_SCHEMA: "offline_arb_textatlas_plan_rows_v1",
    "generic_v1": "offline_arb_textatlas_plan_rows_generic_v1",
}


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


def _resolve_within_formal_field(
    coordinate_schema: str,
    explicit_field: str | None,
) -> str:
    try:
        expected_field = COORDINATE_SCHEMA_FIELDS[coordinate_schema]
    except KeyError as exc:
        raise AuditError(
            f"unsupported coordinate schema: {coordinate_schema!r}"
        ) from exc
    if explicit_field is not None and explicit_field != expected_field:
        raise AuditError(
            "within-formal field does not match coordinate schema: "
            f"schema={coordinate_schema!r}, "
            f"expected={expected_field!r}, observed={explicit_field!r}"
        )
    return expected_field


def _coordinate_record(
    *,
    step: int,
    rank: int,
    slot: int,
    bucket_idx: int,
    formal_steps: int,
    within_formal_field: str,
) -> dict[str, Any]:
    return {
        "step": step,
        "rank": rank,
        "slot": slot,
        "bucket_idx": bucket_idx,
        within_formal_field: step < formal_steps,
    }


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
    target_row_id: int | None,
    within_formal_field: str = "within_formal_166667",
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
        rows = np.asarray(archive["item_row"])
        buckets = np.asarray(archive["bucket_of_step"])
        domain_counts = np.asarray(archive["domain_counts"])
    expected_shapes = {
        "item_row": (take, world_size, per_rank_bs),
        "bucket_of_step": (take,),
        "domain_counts": (take, len(DOMAIN_ORDER)),
    }
    actual_shapes = {
        "item_row": rows.shape,
        "bucket_of_step": buckets.shape,
        "domain_counts": domain_counts.shape,
    }
    if actual_shapes != expected_shapes:
        raise AuditError(
            f"plan block array shape drift: {path}: "
            f"actual={actual_shapes!r}, expected={expected_shapes!r}"
        )
    expected_dtypes = {
        "item_row": np.dtype(np.int64),
        "bucket_of_step": np.dtype(np.int16),
        "domain_counts": np.dtype(np.int32),
    }
    actual_dtypes = {
        "item_row": rows.dtype,
        "bucket_of_step": buckets.dtype,
        "domain_counts": domain_counts.dtype,
    }
    if actual_dtypes != expected_dtypes:
        raise AuditError(
            f"plan block array dtype drift: {path}: "
            f"actual={actual_dtypes!r}, expected={expected_dtypes!r}"
        )

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
    target_hits = (
        np.argwhere(rows == target_row_id).tolist()
        if target_row_id is not None
        else []
    )
    for local_step, rank, slot in target_hits:
        step = start + int(local_step)
        coordinates.append(
            _coordinate_record(
                step=step,
                rank=int(rank),
                slot=int(slot),
                bucket_idx=int(buckets[local_step]),
                formal_steps=formal_steps,
                within_formal_field=within_formal_field,
            )
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
    parser.add_argument("--target-row-id", type=int, default=None)
    parser.add_argument("--expected-world-size", type=int, default=64)
    parser.add_argument("--expected-per-rank-bs", type=int, default=36)
    parser.add_argument("--expected-stage", default=None)
    parser.add_argument("--expected-image-size", type=int, default=None)
    parser.add_argument("--expected-eligibility", default=None)
    parser.add_argument(
        "--expected-requires-nl", choices=("true", "false"), default=None
    )
    parser.add_argument(
        "--expected-anime-nl-filter-applied",
        choices=("true", "false"),
        default=None,
    )
    parser.add_argument("--expected-plan-id-prefix", default=None)
    parser.add_argument(
        "--coordinate-schema",
        choices=tuple(COORDINATE_SCHEMA_FIELDS),
        default=LEGACY_COORDINATE_SCHEMA,
        help=(
            "coordinate evidence schema; the default preserves the frozen W48 "
            "field name"
        ),
    )
    parser.add_argument(
        "--within-formal-field",
        choices=tuple(COORDINATE_SCHEMA_FIELDS.values()),
        default=None,
        help=(
            "optional explicit field assertion; it must match the selected "
            "coordinate schema"
        ),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-bitmap", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _args()
    within_formal_field = _resolve_within_formal_field(
        args.coordinate_schema, args.within_formal_field
    )
    plan_root = args.plan_root.expanduser().resolve(strict=True)
    catalog_root = args.catalog_root.expanduser().resolve(strict=True)
    plan_header_path = plan_root / "header.json"
    catalog_header_path = catalog_root / "header.json"
    plan = _json(plan_header_path)
    catalog = _json(catalog_header_path)
    if tuple(catalog.get("domain_order", ())) != DOMAIN_ORDER:
        raise AuditError("catalog domain_order drift")
    if int(plan.get("world_size", 0)) != args.expected_world_size or int(
        plan.get("per_rank_bs", 0)
    ) != args.expected_per_rank_bs:
        raise AuditError(
            "plan topology mismatch: "
            f"actual=({plan.get('world_size')},{plan.get('per_rank_bs')}), "
            f"expected=({args.expected_world_size},{args.expected_per_rank_bs})"
        )
    expected_identity_raw = {
        "stage": args.expected_stage,
        "image_size": args.expected_image_size,
        "eligibility": args.expected_eligibility,
        "requires_nl": args.expected_requires_nl,
        "anime_nl_filter_applied": args.expected_anime_nl_filter_applied,
    }
    supplied = [value is not None for value in expected_identity_raw.values()]
    if any(supplied) and not all(supplied):
        raise AuditError(
            "stage identity expectations are all-or-none; provide all five fields"
        )
    if all(supplied):
        expected_identity = {
            "stage": args.expected_stage,
            "image_size": args.expected_image_size,
            "eligibility": args.expected_eligibility,
            "requires_nl": args.expected_requires_nl == "true",
            "anime_nl_filter_applied": (
                args.expected_anime_nl_filter_applied == "true"
            ),
        }
        for label, document in (("plan", plan), ("catalog", catalog)):
            actual = {key: document.get(key) for key in expected_identity}
            if actual != expected_identity:
                raise AuditError(
                    f"{label} stage identity mismatch: actual={actual!r}, "
                    f"expected={expected_identity!r}"
                )
    if args.expected_plan_id_prefix is not None and not str(
        plan.get("plan_id", "")
    ).startswith(args.expected_plan_id_prefix):
        raise AuditError(
            f"plan_id={plan.get('plan_id')!r} lacks expected prefix "
            f"{args.expected_plan_id_prefix!r}"
        )
    if plan.get("catalog_hash") != catalog.get("catalog_hash"):
        raise AuditError("plan/catalog catalog_hash mismatch")
    if plan.get("assignment_hash") != catalog.get("assignment_hash"):
        raise AuditError("plan/catalog assignment_hash mismatch")
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
    if args.target_row_id is not None and not (
        textatlas_base <= args.target_row_id < textatlas_base + textatlas_rows
    ):
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
                target_row_id=(
                    int(args.target_row_id)
                    if args.target_row_id is not None
                    else None
                ),
                within_formal_field=within_formal_field,
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
        "audit": COORDINATE_SCHEMA_AUDIT_NAMES[args.coordinate_schema],
        "read_only": True,
        "plan": {
            "root": str(plan_root),
            "plan_id": plan.get("plan_id"),
            "plan_hash": plan.get("plan_hash"),
            "catalog_hash": plan.get("catalog_hash"),
            "assignment_hash": plan.get("assignment_hash"),
            "header_sha256": _sha256(plan_header_path),
            "total_steps": total_steps,
            "formal_steps": formal_steps,
            "world_size": int(plan["world_size"]),
            "per_rank_bs": int(plan["per_rank_bs"]),
        },
        "catalog": {
            "root": str(catalog_root),
            "catalog_hash": catalog.get("catalog_hash"),
            "assignment_hash": catalog.get("assignment_hash"),
            "header_sha256": _sha256(catalog_header_path),
            "textatlas_row_base": textatlas_base,
            "textatlas_rows": textatlas_rows,
        },
        "stage_identity": {
            key: plan.get(key)
            for key in (
                "stage",
                "image_size",
                "eligibility",
                "requires_nl",
                "anime_nl_filter_applied",
            )
        },
        "textatlas": {
            "total_plan_exposures": total_exposures,
            "formal_plan_exposures": formal_exposures,
            "total_plan_unique_rows": int(np.count_nonzero(total_bitmap)),
            "formal_plan_unique_rows": int(np.count_nonzero(formal_bitmap)),
        },
        "target": {
            "row_id": (
                int(args.target_row_id)
                if args.target_row_id is not None
                else None
            ),
            "total_plan_exposures": len(target_coordinates),
            "formal_plan_exposures": sum(
                int(item[within_formal_field]) for item in target_coordinates
            ),
            "coordinates": target_coordinates,
        },
        "bitmap": {"path": str(output_bitmap), "sha256": bitmap_sha},
    }
    if args.coordinate_schema != LEGACY_COORDINATE_SCHEMA:
        result["coordinate_schema"] = {
            "name": args.coordinate_schema,
            "within_formal_field": within_formal_field,
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
