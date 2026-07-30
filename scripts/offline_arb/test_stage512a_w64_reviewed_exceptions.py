#!/usr/bin/env python3
"""JCC CPU preflight for the reviewed stage512a W64 exceptions.

The default path validates immutable documents and only the plan blocks that
contain the 43 reviewed TextAtlas candidates.  ``--full-plan-count`` streams
all 128 W64 blocks once and independently closes the six Monet FORMAL counts.
``--actual-source`` also checks the six catalog rows and three physical JPEG
payloads.  There is no fixture fallback in either mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from scripts.offline_arb.finalize_stage512a_w64_exception_gate import (
    CATALOG_HEADER_SHA256,
    FORMAL_STEPS,
    GateError,
    MONET_UIDS,
    PER_RANK_BS,
    PLAN_HEADER_SHA256,
    PLAN_HORIZON,
    TEXTATLAS_REPAIRS,
    WORLD_SIZE,
    validate_gate_documents,
)
from scripts.offline_arb.stage512a_w64_exception_contract import (
    BLOCK_STEPS,
    CATALOG_HASH,
    GLOBAL_BATCH,
    NUM_BLOCKS,
    PLAN_HASH,
    PLAN_ID,
    finalized_contract,
)
from training.dataline.image_fetch import decode_rgb
from training.dataline_offline.caption_exceptions_v1 import (
    STAGE512A_W64_CAPTIONLESS_MONET_ROWS,
    STAGE512A_W64_CATALOG_HASH,
    STAGE512A_W64_CATALOG_HEADER_SHA256,
    STAGE512A_W64_ELIGIBILITY,
    STAGE512A_W64_PLAN_HEADER_SHA256,
    STAGE512A_W64_PLAN_ID,
    STAGE512A_W64_STAGE,
    bind_caption_exceptions,
)
from training.dataline_offline.caption_sidecar import CaptionSidecarStore
from training.dataline_offline.image_exceptions_v1 import (
    STAGE512A_W64_PLAN_HASH,
    STAGE512A_W64_TEXTATLAS_JPEG_EOI_REPAIRS,
    ImageExceptionMismatchError,
    apply_textatlas_jpeg_eoi_repair,
    bind_image_exceptions,
)
from training.dataline_offline.offline_dataset import _CAPTION_SOURCE_COLUMNS


BASE = Path("/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop")
DEFAULT_PLAN_ROOT = BASE / "offline_arb_plans_stage512a_w64_v2/w64_prod/plan"
DEFAULT_CATALOG_ROOT = BASE / "offline_arb_catalog_stage512a_v1"
DEFAULT_SIDECAR_HEADER = BASE / "offline_arb_caption_sidecar_stage512a_v1/header.json"
DEFAULT_AUDIT_ROOT = BASE / "offline_arb_exceptions_audit_stage512a_w64_v1"
DEFAULT_REBIND_ROOT = (
    BASE / "offline_arb_textatlas_catalog_evidence_rebind_stage512a_w64_v1"
)
PLAN_BLOCK_MEMBERS = frozenset({"bucket_of_step", "item_row", "domain_counts"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def plan_block_arrays(
    archive: Any, *, block_index: int
) -> tuple[np.ndarray, np.ndarray]:
    require(set(archive.files) == PLAN_BLOCK_MEMBERS, f"block member drift: {block_index}")
    items = np.asarray(archive["item_row"])
    buckets = np.asarray(archive["bucket_of_step"])
    counts = np.asarray(archive["domain_counts"])
    require(
        items.dtype == np.dtype(np.int64)
        and items.shape == (BLOCK_STEPS, WORLD_SIZE, PER_RANK_BS),
        f"block item_row contract drift: {block_index}",
    )
    require(
        buckets.dtype == np.dtype(np.int16) and buckets.shape == (BLOCK_STEPS,),
        f"block bucket contract drift: {block_index}",
    )
    require(
        counts.dtype == np.dtype(np.int32) and counts.shape == (BLOCK_STEPS, 5),
        f"block domain_counts contract drift: {block_index}",
    )
    return items, buckets


def domain_layout(plan: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    domains = tuple(str(item) for item in plan["domains"])
    matrix = np.asarray(plan["N"], dtype=np.int64)
    require(matrix.shape == (len(plan["buckets"]), len(domains)), "plan N shape drift")
    totals = tuple(int(item) for item in plan["N_total"])
    require(tuple(matrix.sum(axis=0).tolist()) == totals, "plan N/N_total drift")
    row_base = 0
    result: dict[str, tuple[int, int]] = {}
    for domain, rows in zip(domains, totals, strict=True):
        result[domain] = (row_base, rows)
        row_base += rows
    require(row_base == int(plan["catalog_size"]), "plan catalog_size drift")
    return result


def verify_rebind_coordinates(
    *,
    plan_root: Path,
    checksums: Mapping[str, Any],
    rebind: Mapping[str, Any],
) -> None:
    grouped: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    seen: set[tuple[int, int, int, int]] = set()
    for row in rebind["rows"]:
        row_id = int(row["row_id"])
        for coordinate in row["coordinates"]:
            identity = (
                row_id,
                int(coordinate["step"]),
                int(coordinate["rank"]),
                int(coordinate["slot"]),
            )
            require(identity not in seen, f"duplicate W64 coordinate: {identity}")
            seen.add(identity)
            grouped[identity[1] // BLOCK_STEPS].append((row_id, coordinate))
    block_hashes = checksums["blocks"]
    for block_index, entries in sorted(grouped.items()):
        name = f"block_{block_index:06d}.npz"
        path = plan_root / "blocks" / name
        require(sha256(path) == block_hashes[name], f"plan block SHA drift: {name}")
        with np.load(path, allow_pickle=False) as archive:
            items, buckets = plan_block_arrays(archive, block_index=block_index)
            for row_id, coordinate in entries:
                step = int(coordinate["step"])
                rank = int(coordinate["rank"])
                slot = int(coordinate["slot"])
                require(
                    int(items[step % BLOCK_STEPS, rank, slot]) == row_id,
                    f"plan coordinate row drift: row={row_id},coordinate={coordinate}",
                )
                require(
                    int(buckets[step % BLOCK_STEPS]) == int(coordinate["bucket_idx"]),
                    f"plan coordinate bucket drift: row={row_id}",
                )
                require(
                    int(coordinate["plan_index"])
                    == step * GLOBAL_BATCH + rank * PER_RANK_BS + slot,
                    f"plan_index drift: row={row_id}",
                )


def check_full_monet_counts(
    *, plan_root: Path, checksums: Mapping[str, Any], expected: Mapping[int, int]
) -> None:
    targets = np.asarray(sorted(expected), dtype=np.int64)
    counts = {int(row_id): 0 for row_id in targets.tolist()}
    block_hashes = checksums["blocks"]
    for block_index in range(NUM_BLOCKS):
        name = f"block_{block_index:06d}.npz"
        path = plan_root / "blocks" / name
        require(sha256(path) == block_hashes[name], f"full scan block SHA drift: {name}")
        with np.load(path, allow_pickle=False) as archive:
            items, _ = plan_block_arrays(archive, block_index=block_index)
        block_start = block_index * BLOCK_STEPS
        formal_here = max(0, min(BLOCK_STEPS, FORMAL_STEPS - block_start))
        if not formal_here:
            continue
        selected = items[:formal_here].reshape(-1)
        selected = selected[np.isin(selected, targets)]
        row_ids, row_counts = np.unique(selected, return_counts=True)
        for row_id, count in zip(row_ids.tolist(), row_counts.tolist(), strict=True):
            counts[int(row_id)] += int(count)
    require(counts == dict(expected), f"full W64 Monet count drift: {counts!r}")


def catalog_row(
    *,
    catalog_root: Path,
    catalog: Mapping[str, Any],
    layout: Mapping[str, tuple[int, int]],
    domain: str,
    row_id: int,
    columns: list[str],
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    base, count = layout[domain]
    require(base <= row_id < base + count, f"row {row_id} outside {domain}")
    path = catalog_root / str(catalog["files"][domain]["name"])
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == count, f"catalog row count drift: {domain}")
    local = row_id - base
    start = 0
    for group in range(parquet.metadata.num_row_groups):
        rows = int(parquet.metadata.row_group(group).num_rows)
        if local < start + rows:
            result = (
                parquet.read_row_group(group, columns=columns, use_threads=False)
                .slice(local - start, 1)
                .to_pylist()[0]
            )
            require(result.get("row_id") == row_id, "catalog physical row drift")
            require(result.get("domain") == domain, "catalog physical domain drift")
            return result
        start += rows
    raise AssertionError(f"cannot locate catalog row {row_id}/{domain}")


def check_actual_source(
    *,
    plan: Mapping[str, Any],
    catalog_root: Path,
    catalog: Mapping[str, Any],
    sidecar_root: Path,
) -> None:
    import pyarrow.parquet as pq

    layout = domain_layout(plan)

    def decode_domain(row_id: int) -> str:
        for domain, (base, count) in layout.items():
            if base <= row_id < base + count:
                return domain
        raise AssertionError(f"row outside plan catalog: {row_id}")

    caption_bound = bind_caption_exceptions(
        stage=STAGE512A_W64_STAGE,
        eligibility=STAGE512A_W64_ELIGIBILITY,
        plan_id=STAGE512A_W64_PLAN_ID,
        catalog_hash=STAGE512A_W64_CATALOG_HASH,
        plan_header_sha256=STAGE512A_W64_PLAN_HEADER_SHA256,
        catalog_header_sha256=STAGE512A_W64_CATALOG_HEADER_SHA256,
        decode_domain=decode_domain,
    )
    require(dict(caption_bound) == MONET_UIDS, "W64 caption bind drift")
    expected_domains = {
        domain: {
            "row_base": layout[domain][0],
            "row_count": layout[domain][1],
            "columns": _CAPTION_SOURCE_COLUMNS[domain],
            "catalog_file": catalog["files"][domain],
        }
        for domain in plan["domains"]
    }
    sidecar = CaptionSidecarStore(
        sidecar_root,
        catalog_hash=CATALOG_HASH,
        catalog_header_sha256=CATALOG_HEADER_SHA256,
        expected_domains=expected_domains,
        workers=6,
    )
    try:
        sidecar_rows = sidecar.get_many("monet_real", list(caption_bound))
    finally:
        sidecar.close()
    monet_columns = ["row_id", "uid", "domain", "served_mask", "bucket_idx"]
    for row_id, uid in caption_bound.items():
        row = catalog_row(
            catalog_root=catalog_root,
            catalog=catalog,
            layout=layout,
            domain="monet_real",
            row_id=row_id,
            columns=monet_columns,
        )
        require(row["uid"] == uid, f"Monet UID drift: {row_id}")
        values = sidecar_rows[row_id]
        require(
            set(values) == set(_CAPTION_SOURCE_COLUMNS["monet_real"])
            and all(value is None for value in values.values()),
            f"Monet sidecar row is not exactly null: {row_id}/{values!r}",
        )

    image_bound = bind_image_exceptions(
        stage="512a",
        eligibility="elig_512",
        plan_id=PLAN_ID,
        plan_hash=PLAN_HASH,
        catalog_hash=CATALOG_HASH,
        plan_header_sha256=PLAN_HEADER_SHA256,
        catalog_header_sha256=CATALOG_HEADER_SHA256,
        decode_domain=decode_domain,
    )
    require(
        dict(image_bound) == dict(STAGE512A_W64_TEXTATLAS_JPEG_EOI_REPAIRS),
        "W64 image bind drift",
    )
    columns = [
        "row_id",
        "uid",
        "domain",
        "bucket_idx",
        "width",
        "height",
        "image_path",
        "image_parquet_path",
        "image_row_group",
        "image_row_in_group",
    ]
    for row_id, repair in image_bound.items():
        row = catalog_row(
            catalog_root=catalog_root,
            catalog=catalog,
            layout=layout,
            domain="textatlas",
            row_id=row_id,
            columns=columns,
        )
        source = pq.ParquetFile(repair.image_parquet_path).read_row_group(
            repair.image_row_group,
            columns=["image", "image_path"],
            use_threads=False,
        )
        source_row = source.slice(repair.image_row_in_group, 1).to_pylist()[0]
        require(source_row["image_path"] == repair.image_path, f"source locator drift: {row_id}")
        raw = bytes(source_row["image"]["bytes"])
        require(decode_rgb(raw) is None, f"reviewed raw JPEG unexpectedly decodes: {row_id}")
        repaired, expected_pixel_sha = apply_textatlas_jpeg_eoi_repair(
            repair, domain="textatlas", row=row, raw=raw
        )
        require(repaired == raw + b"\xff\xd9", f"repair is not exact append-EOI: {row_id}")
        image = decode_rgb(repaired)
        require(image is not None and image.mode == "RGB", f"repaired decode failed: {row_id}")
        require(image.size == (repair.width, repair.height), f"repaired size drift: {row_id}")
        require(
            hashlib.sha256(image.tobytes()).hexdigest()
            == expected_pixel_sha
            == repair.pixel_sha256,
            f"repaired pixel SHA drift: {row_id}",
        )
        bad_row = dict(row)
        bad_row["uid"] = "TextAtlas:wrong.jpg"
        try:
            apply_textatlas_jpeg_eoi_repair(
                repair, domain="textatlas", row=bad_row, raw=raw
            )
        except ImageExceptionMismatchError:
            pass
        else:
            raise AssertionError(f"repair accepted wrong catalog identity: {row_id}")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--sidecar-header", type=Path, default=DEFAULT_SIDECAR_HEADER)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--rebind-root", type=Path, default=DEFAULT_REBIND_ROOT)
    parser.add_argument("--actual-source", action="store_true")
    parser.add_argument("--full-plan-count", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    contract = finalized_contract()
    plan_header = args.plan_root / "header.json"
    plan_checksums = args.plan_root / "checksums.json"
    catalog_header = args.catalog_root / "header.json"
    caption_path = args.audit_root / "caption_availability.json"
    caption_horizon_path = args.audit_root / "caption_availability_horizon.json"
    plan_rows_path = args.audit_root / "plan_rows.json"
    plan_bitmap_path = args.audit_root / "plan_rows_bitmap.npz"
    rebind_path = args.rebind_root / "textatlas_catalog_evidence_rebind.json"
    receipt_path = args.rebind_root / "rebind_receipt.json"
    success_path = args.rebind_root / "_SUCCESS"

    require(sha256(plan_header) == PLAN_HEADER_SHA256, "W64 plan header SHA drift")
    require(
        sha256(plan_checksums) == contract["hashes"]["plan_checksums_sha256"],
        "W64 plan checksums SHA drift",
    )
    require(sha256(catalog_header) == CATALOG_HEADER_SHA256, "catalog header SHA drift")
    require(sha256(caption_path) == contract["hashes"]["caption_audit_sha256"], "caption FORMAL SHA drift")
    require(sha256(caption_horizon_path) == contract["hashes"]["caption_horizon_audit_sha256"], "caption horizon SHA drift")
    require(sha256(plan_rows_path) == contract["hashes"]["plan_rows_sha256"], "plan-row SHA drift")
    require(sha256(plan_bitmap_path) == contract["hashes"]["plan_bitmap_sha256"], "plan bitmap SHA drift")
    require(sha256(rebind_path) == contract["hashes"]["textatlas_rebind_sha256"], "rebind SHA drift")
    require(sha256(receipt_path) == contract["hashes"]["textatlas_rebind_receipt_sha256"], "rebind receipt SHA drift")

    plan = json_object(plan_header)
    checksums = json_object(plan_checksums)
    catalog = json_object(catalog_header)
    caption = json_object(caption_path)
    caption_horizon = json_object(caption_horizon_path)
    plan_rows = json_object(plan_rows_path)
    rebind = json_object(rebind_path)
    receipt = json_object(receipt_path)
    marker = success_path.read_bytes()
    try:
        result = validate_gate_documents(
            plan,
            catalog,
            caption,
            caption_horizon,
            plan_rows,
            rebind,
            receipt,
            marker,
            plan_header_sha256=sha256(plan_header),
            catalog_header_sha256=sha256(catalog_header),
            plan_bitmap_sha256=sha256(plan_bitmap_path),
            rebind_path=rebind_path,
            rebind_sha256=sha256(rebind_path),
            rebind_receipt_path=receipt_path,
            rebind_receipt_sha256=sha256(receipt_path),
        )
    except GateError as exc:
        raise AssertionError(str(exc)) from exc
    require(plan.get("plan_id") == PLAN_ID and plan.get("plan_hash") == PLAN_HASH, "W64 plan identity drift")
    require(plan.get("world_size") == WORLD_SIZE and plan.get("per_rank_bs") == PER_RANK_BS, "W64 topology drift")
    require(plan.get("total_steps") == PLAN_HORIZON, "W64 horizon drift")
    verify_rebind_coordinates(plan_root=args.plan_root, checksums=checksums, rebind=rebind)
    if args.full_plan_count:
        check_full_monet_counts(
            plan_root=args.plan_root,
            checksums=checksums,
            expected=contract["monet_formal_exposures"],
        )
    if args.actual_source:
        check_actual_source(
            plan=plan,
            catalog_root=args.catalog_root,
            catalog=catalog,
            sidecar_root=args.sidecar_header.parent,
        )
    print(
        "STAGE512A_W64_REVIEWED_EXCEPTIONS_PASS "
        f"formal_steps={FORMAL_STEPS} horizon={PLAN_HORIZON} "
        f"monet={sum(result['caption_formal_counts'].values())} "
        f"textatlas={result['textatlas_counts']['candidate_formal_exposures']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
