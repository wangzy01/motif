#!/usr/bin/env python3
"""Finalize actual-production TextAtlas decode reclassification evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class FinalizeError(RuntimeError):
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
        raise FinalizeError(f"JSON root is not an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--neutral-suggestion", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--image-fetch", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--historical-progress-step", type=int, default=52940)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.expanduser().resolve(strict=True)
    audit_path = args.audit.expanduser().resolve(strict=True)
    neutral_path = args.neutral_suggestion.expanduser().resolve(strict=True)
    bindings_path = args.bindings.expanduser().resolve(strict=True)
    image_fetch_path = args.image_fetch.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FinalizeError(f"refusing to overwrite output: {output}")
    shard_paths = [root / f"runtime_recheck_shard_{index}.json" for index in range(args.shard_count)]
    shards = [_json(path) for path in shard_paths]
    if any(item.get("status") != "PASS" for item in shards):
        raise FinalizeError("runtime recheck shard status drift")
    if sorted(int(item["shard_index"]) for item in shards) != list(range(args.shard_count)):
        raise FinalizeError("runtime recheck shard index coverage drift")

    audit = _json(audit_path)
    neutral = _json(neutral_path)
    bindings = _json(bindings_path)
    expected_ids = {int(item["row_id"]) for item in audit["candidates"]}
    rows = [item for shard in shards for item in shard["rows"]]
    observed_ids = [int(item["row_id"]) for item in rows]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
        raise FinalizeError("runtime recheck row coverage drift")
    if any(
        item["provenance"]["integrity_audit_sha256"] != _sha256(audit_path)
        or item["provenance"]["repaired_bindings_sha256"] != _sha256(bindings_path)
        or item["provenance"]["production_image_fetch_sha256"] != _sha256(image_fetch_path)
        for item in shards
    ):
        raise FinalizeError("runtime recheck provenance drift")
    if any(item["image_max_pixels_after_production_import"] is not None for item in rows):
        raise FinalizeError("production Pillow MAX_IMAGE_PIXELS policy not disabled")

    by_id = {int(item["row_id"]): item for item in rows}
    candidate_by_id = {int(item["row_id"]): item for item in audit["candidates"]}
    bound_ids = {int(item["row_id"]) for item in bindings["rows"]}
    success_ids = {
        row_id for row_id, item in by_id.items() if bool(item["raw_runtime_decode_ok"])
    }
    failure_ids = set(by_id) - success_ids
    if failure_ids != bound_ids or len(bound_ids) != 3:
        raise FinalizeError(
            f"runtime failure/repair partition drift failures={sorted(failure_ids)},"
            f"bindings={sorted(bound_ids)}"
        )
    for row_id in success_ids:
        item = by_id[row_id]
        candidate = candidate_by_id[row_id]
        if (
            item["raw_decoded_mode"] != "RGB"
            or not item["pixel_load_succeeded"]
            or item["raw_decoded_size"]
            != [int(candidate["catalog_width"]), int(candidate["catalog_height"])]
            or item.get("repair") is not None
        ):
            raise FinalizeError(f"runtime success evidence drift row_id={row_id}")
    for row_id in failure_ids:
        item = by_id[row_id]
        if item["raw_runtime_error_type"] != "OSError" or not bool(
            item.get("repair", {}).get("ok")
        ):
            raise FinalizeError(f"runtime repair evidence drift row_id={row_id}")

    classifications = []
    historical_success_rows = 0
    for candidate in audit["candidates"]:
        row_id = int(candidate["row_id"])
        runtime = by_id[row_id]
        first = candidate["first_formal_coordinate"]
        before_progress = bool(
            first is not None and int(first["step"]) < args.historical_progress_step
        )
        if row_id in success_ids and before_progress:
            historical_success_rows += 1
        classifications.append(
            {
                "row_id": row_id,
                "classification": (
                    "production_decode_success_no_exception"
                    if row_id in success_ids
                    else "production_decode_failure_explicit_eoi_repair"
                ),
                "raw_bytes": runtime["raw_bytes"],
                "raw_sha256": runtime["raw_sha256"],
                "runtime_decoded_size": runtime["raw_decoded_size"],
                "runtime_error_type": runtime["raw_runtime_error_type"],
                "runtime_error": runtime["raw_runtime_error"],
                "pixel_load_succeeded": runtime["pixel_load_succeeded"],
                "repair": runtime["repair"],
                "formal_plan_exposures": candidate["formal_plan_exposures"],
                "total_plan_exposures": candidate["total_plan_exposures"],
                "first_formal_coordinate": first,
                "historically_before_observed_progress": before_progress,
                "formal_coordinates": [
                    coordinate
                    for coordinate in candidate["coordinates"]
                    if coordinate["within_formal_166667"]
                ],
                "ru_maxrss_kib": runtime["ru_maxrss_kib"],
                "elapsed_seconds": runtime["elapsed_seconds"],
            }
        )

    result = {
        "status": "PASS",
        "audit": "offline_arb_textatlas_runtime_policy_reclassification_v1",
        "decision": {
            "actual_production_decode_failure_rows": len(failure_ids),
            "explicit_jpeg_eoi_repair_rows": len(bound_ids),
            "production_decode_success_rows": len(success_ids),
            "neutral_skip_rows_required": 0,
            "neutral_skip_suggestion_status": "INVALIDATED_FALSE_POSITIVE_DO_NOT_ACTIVATE",
        },
        "reason": (
            "The broad audit used Pillow's default decompression-bomb threshold; "
            "production image_fetch disables that threshold before decode_rgb. "
            "All 40 affected rows completed actual RGB conversion and pixel load."
        ),
        "runtime_policy": {
            "production_image_fetch_path": str(image_fetch_path),
            "production_image_fetch_sha256": _sha256(image_fetch_path),
            "image_max_pixels_after_production_import": None,
            "decoder": "training.dataline.image_fetch.decode_rgb",
            "per_candidate_fresh_process": True,
        },
        "historical_cross_check": {
            "observed_formal_progress_step": args.historical_progress_step,
            "runtime_success_rows_first_seen_before_progress": historical_success_rows,
        },
        "provenance": {
            "integrity_audit_path": str(audit_path),
            "integrity_audit_sha256": _sha256(audit_path),
            "invalidated_neutral_suggestion_path": str(neutral_path),
            "invalidated_neutral_suggestion_sha256": _sha256(neutral_path),
            "repaired_bindings_path": str(bindings_path),
            "repaired_bindings_sha256": _sha256(bindings_path),
            "plan": audit["provenance"],
            "runtime_recheck_shards": [
                {"path": str(path), "sha256": _sha256(path)} for path in shard_paths
            ],
        },
        "summary": {
            "candidate_rows": len(rows),
            "formal_candidate_exposures": sum(
                int(item["formal_plan_exposures"]) for item in audit["candidates"]
            ),
            "runtime_success_rows": len(success_ids),
            "runtime_failure_rows": len(failure_ids),
            "repair_success_rows": len(bound_ids),
            "max_child_ru_maxrss_kib": max(int(item["ru_maxrss_kib"]) for item in rows),
        },
        "rows": classifications,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({"status": "PASS", **result["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
