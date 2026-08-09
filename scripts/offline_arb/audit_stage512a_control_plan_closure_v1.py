#!/usr/bin/env python3
"""Full post-build closure for the fresh-seed Stage512a control plan.

This is intentionally stricter than the generic planner validator.  Besides
scanning every slot, it proves that the new plan changed only its deterministic
draw identity, derives the exact coordinates of the reviewed Anime payload,
counts the three TextAtlas JPEG-repair rows, and proves that all six captionless
MONET rows remain unreachable under the weighted stream.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import stat
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow.parquet as pq

from training.dataline_offline.anime_native_upsample_exceptions_weighted_v1 import (
    ANIME_NATIVE_UPSAMPLE_EXCEPTIONS,
)
from training.dataline_offline.caption_exceptions_v1 import (
    STAGE512A_CAPTIONLESS_MONET_ROWS,
)
from training.dataline_offline.image_exceptions_v1 import (
    STAGE512A_WEIGHTED_TEXTATLAS_JPEG_EOI_REPAIRS,
)


EXPECTED_PLAN_ID = (
    "stage512a_offline_exact_v2_continue_control_rngfix_v1_w64_256m_seed20260710"
)
EXPECTED_CONTRACT_SHA256 = (
    "c228f8b8400c4f96033431816b650d6e7093b3bf8e164018d22e694c60b96c38"
)
EXPECTED_CATALOG_HASH = (
    "370b1679e8c5670338a03ba4c6972cdbf69fa5068fc35108bf646b2d64642331"
)
EXPECTED_DOMAIN_TOTALS = {
    "anime": 94_720_000,
    "monet_real": 102_400_000,
    "gpic": 33_280_000,
    "synthetic": 12_800_000,
    "textatlas": 12_800_000,
}
EXPECTED_DOMAIN_MIX = {
    "anime": "0.37",
    "monet_real": "0.4",
    "gpic": "0.13",
    "synthetic": "0.05",
    "textatlas": "0.05",
}
EXPECTED_TEXTATLAS_EXPOSURES = {
    68_973_396: 3,
    69_010_633: 3,
    70_063_978: 3,
}
EXPECTED_ANIME_EXPOSURES = {11_163_647: 6}
BLOCK_DATASETS = {
    "bucket_of_step",
    "item_row",
    "caption_band",
    "domain_counts",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"JSON root is not an object: {path}")
    return value


def require_mode(path: Path, mode: int, *, regular: bool | None = None) -> None:
    info = path.lstat()
    if path.is_symlink() or stat.S_IMODE(info.st_mode) != mode:
        raise AssertionError(f"unsafe mode/symlink: {path}")
    if regular is True and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
        raise AssertionError(f"unsafe regular file identity: {path}")
    if regular is False and not stat.S_ISDIR(info.st_mode):
        raise AssertionError(f"not a directory: {path}")


def domain_bounds(catalog_header: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    cursor = 0
    result: dict[str, tuple[int, int]] = {}
    for domain in catalog_header["domain_order"]:
        count = int(catalog_header["files"][domain]["rows"])
        result[str(domain)] = (cursor, cursor + count)
        cursor += count
    return result


def catalog_record(
    catalog_root: Path,
    bounds: Mapping[str, tuple[int, int]],
    row_id: int,
    columns: tuple[str, ...],
) -> tuple[str, dict[str, Any]]:
    domain = next(
        name for name, (lo, hi) in bounds.items() if lo <= row_id < hi
    )
    lo, _ = bounds[domain]
    local_row = row_id - lo
    reader = pq.ParquetFile(catalog_root / f"{domain}.parquet")
    cursor = 0
    for group in range(reader.num_row_groups):
        count = reader.metadata.row_group(group).num_rows
        if local_row < cursor + count:
            offset = local_row - cursor
            table = reader.read_row_group(group, columns=list(columns))
            record = {
                name: table[name][offset].as_py()
                for name in columns
            }
            return domain, record
        cursor += count
    raise AssertionError(f"catalog row outside parquet: {row_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-root", required=True, type=Path)
    parser.add_argument("--old-plan-root", required=True, type=Path)
    parser.add_argument("--catalog-root", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-plan-id", default=EXPECTED_PLAN_ID)
    args = parser.parse_args()

    if args.out.exists() or args.out.is_symlink():
        raise SystemExit(f"refusing to overwrite {args.out}")

    require_mode(args.plan_root, 0o555, regular=False)
    require_mode(args.plan_root / "blocks", 0o555, regular=False)
    if {path.name for path in args.plan_root.iterdir()} != {
        "header.json", "checksums.json", "blocks"
    }:
        raise AssertionError("plan root member-set drift")
    for name in ("header.json", "checksums.json"):
        require_mode(args.plan_root / name, 0o444, regular=True)

    header = load_json(args.plan_root / "header.json")
    checksums = load_json(args.plan_root / "checksums.json")
    old_header = load_json(args.old_plan_root / "header.json")
    catalog_header = load_json(args.catalog_root / "header.json")

    assert args.expected_plan_id == EXPECTED_PLAN_ID
    assert header["plan_id"] == args.expected_plan_id
    assert header["seed"] == 20260710
    assert header["stage"] == "512a" and header["image_size"] == 512
    assert header["format_version"] == 2
    assert header["block_steps"] == 1024
    assert (header["world_size"], header["per_rank_bs"]) == (64, 32)
    assert (header["total_steps"], header["total_exposures"]) == (
        125_000, 256_000_000
    )
    assert header["weighted_contract_sha256"] == EXPECTED_CONTRACT_SHA256
    assert sha256_file(args.contract) == EXPECTED_CONTRACT_SHA256
    assert header["catalog_hash"] == EXPECTED_CATALOG_HASH
    assert catalog_header["catalog_hash"] == EXPECTED_CATALOG_HASH
    assert header["domain_mix"] == EXPECTED_DOMAIN_MIX
    assert header["domain_totals"] == EXPECTED_DOMAIN_TOTALS
    assert header["square_only_domains"] == ["synthetic", "textatlas"]
    assert sum(int(value) > 0 for value in header["S"]) == 30
    assert sha256_file(args.catalog_root / "header.json") == (
        "51a5f0904987a751b54842633d2968ea7f6485307850858d9472319c15748d1d"
    )
    assert old_header["plan_id"] == "stage512a_offline_exact_v2_w64_256m_weighted_v1"
    assert old_header["plan_hash"] == (
        "7295ca937d5002a0cf5e207c1d466c5992352e395adc9ae1c33f3cb616f675db"
    )

    # These fields define the A data/geometry control.  The producer source and
    # deterministic plan identity are intentionally different; these are not.
    invariant_fields = (
        "stage", "image_size", "eligibility", "requires_nl",
        "anime_nl_filter_applied", "world_size", "per_rank_bs",
        "total_steps", "total_exposures", "domains", "domain_mix",
        "domain_totals", "D", "N", "N_total", "R_max", "buckets",
        "square_bucket_idx", "square_only_domains", "catalog_hash",
        "upstream_catalog_hash", "assignment_hash", "upstream_assignment_hash",
        "weighted_contract", "weighted_contract_sha256",
    )
    differences = {
        field: {"old": old_header.get(field), "new": header.get(field)}
        for field in invariant_fields
        if old_header.get(field) != header.get(field)
    }
    assert not differences, f"old/new A control invariants drifted: {sorted(differences)}"

    bounds = domain_bounds(catalog_header)
    text_repairs = dict(STAGE512A_WEIGHTED_TEXTATLAS_JPEG_EOI_REPAIRS)
    anime_exceptions = dict(ANIME_NATIVE_UPSAMPLE_EXCEPTIONS)
    captionless = dict(STAGE512A_CAPTIONLESS_MONET_ROWS)
    assert len(text_repairs) == 3 and len(anime_exceptions) == 1
    assert len(captionless) == 6

    # Revalidate that the frozen physical exception definitions still name the
    # exact rows in the unchanged A catalog.  Coordinates are deliberately not
    # reused here; they are derived from the fresh plan below.
    catalog_evidence: dict[str, Any] = {}
    for row_id, repair in text_repairs.items():
        columns = (
            "row_id", "uid", "domain", "bucket_idx", "width", "height",
            "image_path", "image_parquet_path", "image_row_group",
            "image_row_in_group",
        )
        domain, record = catalog_record(args.catalog_root, bounds, row_id, columns)
        expected = {
            "row_id": repair.row_id, "uid": repair.uid, "domain": "textatlas",
            "bucket_idx": repair.bucket_idx, "width": repair.width,
            "height": repair.height, "image_path": repair.image_path,
            "image_parquet_path": repair.image_parquet_path,
            "image_row_group": repair.image_row_group,
            "image_row_in_group": repair.image_row_in_group,
        }
        assert domain == "textatlas" and record == expected
        catalog_evidence[str(row_id)] = record

    for row_id, exception in anime_exceptions.items():
        columns = (
            "row_id", "uid", "domain", "bucket_idx", "width", "height",
            "key", "fetch_url", "cluster", "rating_bucket", "year",
            "caption_parquet_path", "caption_row_group", "caption_row_in_group",
        )
        domain, record = catalog_record(args.catalog_root, bounds, row_id, columns)
        expected = {
            "row_id": exception.row_id, "uid": exception.uid,
            "domain": exception.domain, "bucket_idx": exception.bucket_idx,
            "width": exception.catalog_width, "height": exception.catalog_height,
            "key": exception.key, "fetch_url": exception.fetch_url,
            "cluster": exception.cluster, "rating_bucket": exception.rating_bucket,
            "year": exception.year,
            "caption_parquet_path": exception.caption_parquet_path,
            "caption_row_group": exception.caption_row_group,
            "caption_row_in_group": exception.caption_row_in_group,
        }
        assert domain == "anime" and record == expected
        catalog_evidence[str(row_id)] = record

    for row_id, uid in captionless.items():
        domain, record = catalog_record(
            args.catalog_root, bounds, row_id, ("row_id", "uid", "domain")
        )
        assert domain == "monet_real"
        assert record == {"row_id": row_id, "uid": uid, "domain": "monet_real"}
        catalog_evidence[str(row_id)] = record

    target_ids = set(text_repairs) | set(anime_exceptions) | set(captionless)
    hits: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    realized_domains = np.zeros(len(header["domains"]), dtype=np.int64)
    realized_buckets = np.zeros(len(header["buckets"]), dtype=np.int64)
    total_steps = 0
    total_refs = 0
    first_block = int(header["materialization"]["first_materialized_block"])
    num_blocks = int(header["num_blocks"])
    expected_names = {
        f"block_{index:06d}.npz" for index in range(first_block, num_blocks)
    }
    assert set(checksums) == {"header", "blocks"}
    assert set(checksums["blocks"]) == expected_names
    assert {path.name for path in (args.plan_root / "blocks").iterdir()} == expected_names
    assert sha256_file(args.plan_root / "header.json") == checksums["header"]

    real_bounds = [bounds["monet_real"], bounds["gpic"]]
    for index in range(first_block, num_blocks):
        name = f"block_{index:06d}.npz"
        path = args.plan_root / "blocks" / name
        require_mode(path, 0o444, regular=True)
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == checksums["blocks"][name]
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            assert set(archive.files) == BLOCK_DATASETS
            buckets = np.asarray(archive["bucket_of_step"])
            rows = np.asarray(archive["item_row"])
            bands = np.asarray(archive["caption_band"])
            domain_counts = np.asarray(archive["domain_counts"])
        expected_len = min(1024, 125_000 - index * 1024)
        assert buckets.shape == (expected_len,) and buckets.dtype == np.int16
        assert rows.shape == (expected_len, 64, 32) and rows.dtype == np.int64
        assert bands.shape == rows.shape and bands.dtype == np.int8
        assert domain_counts.shape == (expected_len, 5) and domain_counts.dtype == np.int32
        assert np.all((bands >= -1) & (bands <= 3))
        real_mask = np.zeros(rows.shape, dtype=bool)
        for lo, hi in real_bounds:
            real_mask |= (rows >= lo) & (rows < hi)
        assert np.all(bands[real_mask] >= 0)
        assert np.all(bands[~real_mask] == -1)
        assert np.all(domain_counts.sum(axis=1, dtype=np.int64) == 2048)
        realized_domains += domain_counts.sum(axis=0, dtype=np.int64)
        realized_buckets += np.bincount(
            buckets.astype(np.int64), minlength=len(realized_buckets)
        )
        total_steps += expected_len
        total_refs += rows.size

        mask = np.isin(rows, np.fromiter(target_ids, dtype=np.int64))
        for local_step, rank, slot in np.argwhere(mask):
            row_id = int(rows[local_step, rank, slot])
            step = index * 1024 + int(local_step)
            assert int(buckets[local_step]) == int(catalog_evidence[str(row_id)].get("bucket_idx", 182))
            assert int(bands[local_step, rank, slot]) == -1
            hits[row_id].append((step, int(rank), int(slot)))

    assert total_steps == 125_000 and total_refs == 256_000_000
    assert realized_domains.tolist() == [
        EXPECTED_DOMAIN_TOTALS[name] for name in header["domains"]
    ]
    assert realized_buckets.tolist() == header["S"]
    assert all(hits[row_id] for row_id in text_repairs)
    assert all(hits[row_id] for row_id in anime_exceptions)
    assert all(not hits[row_id] for row_id in captionless)
    assert {
        row_id: len(hits[row_id]) for row_id in text_repairs
    } == EXPECTED_TEXTATLAS_EXPOSURES
    assert {
        row_id: len(hits[row_id]) for row_id in anime_exceptions
    } == EXPECTED_ANIME_EXPOSURES
    for row_id, coordinates in hits.items():
        assert len(coordinates) == len(set(coordinates))
        assert len({step for step, _, _ in coordinates}) == len(coordinates)

    plan_files = [
        args.plan_root / "header.json",
        args.plan_root / "checksums.json",
        *sorted((args.plan_root / "blocks").iterdir()),
    ]
    text_payloads = {
        str(row_id): {
            **dataclasses.asdict(repair),
            "realized_exposure_count": len(hits[row_id]),
            "realized_coordinates": [list(item) for item in hits[row_id]],
        }
        for row_id, repair in text_repairs.items()
    }
    anime_payloads = {}
    for row_id, exception in anime_exceptions.items():
        physical = dataclasses.asdict(exception)
        physical.pop("coordinates")
        anime_payloads[str(row_id)] = {
            **physical,
            "realized_exposure_count": len(hits[row_id]),
            "realized_coordinates": [list(item) for item in hits[row_id]],
        }

    report = {
        "schema": "stage512a_continue_control_plan_closure_v1",
        "status": "PASS",
        "plan_root": str(args.plan_root),
        "plan_id": header["plan_id"],
        "plan_hash": header["plan_hash"],
        "header_sha256": sha256_file(args.plan_root / "header.json"),
        "checksums_sha256": sha256_file(args.plan_root / "checksums.json"),
        "block_manifest_sha256": header["hashes"]["blocks_manifest_sha256"],
        "plan_file_count": len(plan_files),
        "plan_bytes": sum(path.stat().st_size for path in plan_files),
        "old_plan_root": str(args.old_plan_root),
        "old_plan_hash": old_header["plan_hash"],
        "control_invariant_fields": list(invariant_fields),
        "control_invariant_sha256": canonical_sha256(
            {field: header.get(field) for field in invariant_fields}
        ),
        "fresh_identity": {
            "seed": 20260710,
            "interpretation": "preregistered_fresh_seed_not_sequence_paired_to_stage512b",
        },
        "realized_domain_exposures": dict(zip(header["domains"], realized_domains.tolist())),
        "kept_bucket_count": sum(int(value) > 0 for value in realized_buckets),
        "textatlas_jpeg_repairs": text_payloads,
        "anime_native_upsample": anime_payloads,
        "captionless_monet_zero_hit_rows": {
            str(row_id): {"uid": uid, "realized_exposure_count": len(hits[row_id])}
            for row_id, uid in captionless.items()
        },
        "catalog_evidence": catalog_evidence,
        "source_authorities": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "image_exceptions_v1.py": sha256_file(
                Path(__file__).resolve().parents[2]
                / "training/dataline_offline/image_exceptions_v1.py"
            ),
            "anime_native_upsample_exceptions_weighted_v1.py": sha256_file(
                Path(__file__).resolve().parents[2]
                / "training/dataline_offline/anime_native_upsample_exceptions_weighted_v1.py"
            ),
            "caption_exceptions_v1.py": sha256_file(
                Path(__file__).resolve().parents[2]
                / "training/dataline_offline/caption_exceptions_v1.py"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(args.out, 0o444)
    print(
        "STAGE512A_CONTROL_PLAN_CLOSURE_PASS",
        header["plan_hash"],
        "anime_coords=",
        {row_id: len(hits[row_id]) for row_id in anime_exceptions},
        "textatlas_counts=",
        {row_id: len(hits[row_id]) for row_id in text_repairs},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
