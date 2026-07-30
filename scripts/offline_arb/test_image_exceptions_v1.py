#!/usr/bin/env python3
"""Known-answer and negative tests for frozen image-byte exceptions v1."""

from __future__ import annotations

import hashlib
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageFile

from training.dataline_offline.image_exceptions_v1 import (
    ALLOWLIST_SUGGESTION_SHA256,
    CATALOG_HASH,
    CATALOG_HEADER_SHA256,
    ELIGIBILITY,
    INTEGRITY_AUDIT_SHA256,
    PLAN_HASH,
    PLAN_HEADER_SHA256,
    PLAN_ID,
    STAGE,
    REPAIRED_PAYLOAD_BINDINGS_SHA256,
    TEXTATLAS_JPEG_EOI_REPAIRS,
    ImageExceptionBindingError,
    ImageExceptionMismatchError,
    apply_textatlas_jpeg_eoi_repair,
    bind_image_exceptions,
)
from training.dataline_offline.offline_dataset import (
    OfflineExactDataset,
    OfflineLoaderError,
    _PlannedItem,
    _PreparedItem,
    _VirtualRowDecoder,
)


BASE = Path("/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop")
PLAN_HEADER = BASE / "offline_arb_plans_v1/w64_prod/plan/header.json"
CATALOG_ROOT = BASE / "offline_arb_catalog_v1"
CATALOG_HEADER = CATALOG_ROOT / "header.json"
INTEGRITY_ROOT = BASE / "offline_arb_textatlas_full_strict_decode_audit_v1"
INTEGRITY_AUDIT = INTEGRITY_ROOT / "integrity_audit.json"
ALLOWLIST_SUGGESTION = INTEGRITY_ROOT / "frozen_allowlist_suggestion.json"
REPAIRED_PAYLOAD_BINDINGS = INTEGRITY_ROOT / "repaired_payload_bindings.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_error(fn, text: str) -> None:
    try:
        fn()
    except (
        ImageExceptionBindingError,
        ImageExceptionMismatchError,
        OfflineLoaderError,
    ) as exc:
        assert text in str(exc), (text, str(exc))
    else:
        raise AssertionError(f"expected error containing {text!r}")


def _locate(parquet: pq.ParquetFile, row: int) -> tuple[int, int]:
    base = 0
    for group in range(parquet.metadata.num_row_groups):
        rows = parquet.metadata.row_group(group).num_rows
        if row < base + rows:
            return group, row - base
        base += rows
    raise AssertionError((row, base))


def _catalog_row(
    parquet: pq.ParquetFile, row_id: int, textatlas_base: int
) -> dict:
    group, within = _locate(parquet, row_id - textatlas_base)
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
    return (
        parquet.read_row_group(group, columns=columns, use_threads=False)
        .slice(within, 1)
        .to_pylist()[0]
    )


def _source_bytes(row: dict) -> bytes:
    parquet = pq.ParquetFile(row["image_parquet_path"])
    table = parquet.read_row_group(
        int(row["image_row_group"]),
        columns=["image", "image_path"],
        use_threads=False,
    )
    value = table.slice(int(row["image_row_in_group"]), 1).to_pylist()[0]
    assert value["image_path"] == row["image_path"]
    return bytes(value["image"]["bytes"])


def _truncated_pixel_sha(raw: bytes) -> tuple[tuple[int, int], str]:
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert("RGB")
            pixels = np.asarray(rgb, dtype=np.uint8).copy()
            return rgb.size, hashlib.sha256(pixels.tobytes()).hexdigest()
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def main() -> int:
    plan = json.loads(PLAN_HEADER.read_text(encoding="utf-8"))
    catalog_header = json.loads(CATALOG_HEADER.read_text(encoding="utf-8"))
    assert _sha256(PLAN_HEADER) == PLAN_HEADER_SHA256
    assert _sha256(CATALOG_HEADER) == CATALOG_HEADER_SHA256
    assert _sha256(INTEGRITY_AUDIT) == INTEGRITY_AUDIT_SHA256
    assert _sha256(ALLOWLIST_SUGGESTION) == ALLOWLIST_SUGGESTION_SHA256
    assert (
        _sha256(REPAIRED_PAYLOAD_BINDINGS)
        == REPAIRED_PAYLOAD_BINDINGS_SHA256
    )
    assert plan["plan_id"] == PLAN_ID
    assert plan["plan_hash"] == PLAN_HASH
    assert plan["catalog_hash"] == catalog_header["catalog_hash"] == CATALOG_HASH

    decoder = _VirtualRowDecoder(plan["domains"], plan["N"])
    decode_domain = lambda row_id: plan["domains"][decoder.decode(row_id)[0]]
    bound = bind_image_exceptions(
        stage=STAGE,
        eligibility=ELIGIBILITY,
        plan_id=PLAN_ID,
        plan_hash=PLAN_HASH,
        catalog_hash=CATALOG_HASH,
        plan_header_sha256=PLAN_HEADER_SHA256,
        catalog_header_sha256=CATALOG_HEADER_SHA256,
        decode_domain=decode_domain,
    )
    assert dict(bound) == dict(TEXTATLAS_JPEG_EOI_REPAIRS)
    assert bound, "the production exception table must not be empty"
    audit = json.loads(INTEGRITY_AUDIT.read_text(encoding="utf-8"))
    suggestion = json.loads(ALLOWLIST_SUGGESTION.read_text(encoding="utf-8"))
    repaired_bindings = json.loads(
        REPAIRED_PAYLOAD_BINDINGS.read_text(encoding="utf-8")
    )
    assert audit["status"] == "PASS"
    assert audit["scope"].startswith("all 4,687,038 planned TextAtlas rows")
    assert audit["summary"]["raw_strict_decode_failures"] == 43
    assert audit["summary"]["raw_strict_failure_error_types"] == {
        "DecompressionBombError": 40,
        "OSError": 3,
    }
    assert audit["summary"]["eligible_explicit_allowlist_rows"] == len(bound)
    # Production image_fetch disables Pillow's decompression-bomb threshold;
    # the 40 audit-only bomb candidates must never become runtime exceptions.
    assert Image.MAX_IMAGE_PIXELS is None
    assert suggestion["generic_signature_fallback_forbidden"] is True
    assert len(suggestion["rows"]) == len(bound)
    assert repaired_bindings["status"] == "PASS"
    assert len(repaired_bindings["rows"]) == len(bound)
    suggested = {int(row["row_id"]): row for row in suggestion["rows"]}
    assert set(suggested) == set(bound)
    repaired_by_id = {
        int(row["row_id"]): row for row in repaired_bindings["rows"]
    }
    assert set(repaired_by_id) == set(bound)
    for row_id, repair in bound.items():
        source = suggested[row_id]
        assert source["uid"] == repair.uid
        assert int(source["bucket_idx"]) == repair.bucket_idx
        assert int(source["catalog_width"]) == repair.width
        assert int(source["catalog_height"]) == repair.height
        assert source["image_path"] == repair.image_path
        assert source["image_parquet_path"] == repair.image_parquet_path
        assert int(source["image_row_group"]) == repair.image_row_group
        assert int(source["image_row_in_group"]) == repair.image_row_in_group
        assert int(source["payload_bytes"]) == repair.raw_bytes
        assert source["payload_sha256"] == repair.raw_sha256
        assert source["appended_bytes_hex"] == "ffd9"
        assert source["expected_rgb_sha256"] == repair.pixel_sha256
        verified = repaired_by_id[row_id]
        assert int(verified["repaired_bytes"]) == repair.repaired_bytes
        assert verified["repaired_payload_sha256"] == repair.repaired_sha256
        assert verified["repaired_rgb_sha256"] == repair.pixel_sha256

    unrelated = bind_image_exceptions(
        stage="512a",
        eligibility="elig_512",
        plan_id="synthetic-test-plan",
        plan_hash="3" * 64,
        catalog_hash="0" * 64,
        plan_header_sha256="1" * 64,
        catalog_header_sha256="2" * 64,
        decode_domain=lambda _row_id: "gpic",
    )
    assert not unrelated
    _expect_error(
        lambda: bind_image_exceptions(
            stage=STAGE,
            eligibility=ELIGIBILITY,
            plan_id=PLAN_ID,
            plan_hash=PLAN_HASH,
            catalog_hash="0" * 64,
            plan_header_sha256=PLAN_HEADER_SHA256,
            catalog_header_sha256=CATALOG_HEADER_SHA256,
            decode_domain=lambda _row_id: "textatlas",
        ),
        "binding mismatch",
    )
    _expect_error(
        lambda: bind_image_exceptions(
            stage=STAGE,
            eligibility=ELIGIBILITY,
            plan_id=PLAN_ID,
            plan_hash=PLAN_HASH,
            catalog_hash=CATALOG_HASH,
            plan_header_sha256=PLAN_HEADER_SHA256,
            catalog_header_sha256=CATALOG_HEADER_SHA256,
            decode_domain=lambda _row_id: "gpic",
        ),
        "not textatlas",
    )
    _expect_error(
        lambda: bind_image_exceptions(
            stage="512a",
            eligibility="elig_512",
            plan_id=PLAN_ID,
            plan_hash=PLAN_HASH,
            catalog_hash=CATALOG_HASH,
            plan_header_sha256=PLAN_HEADER_SHA256,
            catalog_header_sha256=CATALOG_HEADER_SHA256,
            decode_domain=lambda _row_id: "textatlas",
        ),
        "binding mismatch",
    )

    textatlas_idx = plan["domains"].index("textatlas")
    textatlas_base = int(sum(sum(row[:textatlas_idx]) for row in plan["N"]))
    catalog_file = CATALOG_ROOT / catalog_header["files"]["textatlas"]["name"]
    catalog = pq.ParquetFile(catalog_file)
    dataset = object.__new__(OfflineExactDataset)
    dataset.buckets = tuple(tuple(int(v) for v in pair) for pair in plan["buckets"])
    dataset._image_repairs = bound
    dataset._anime_native_upsample_exceptions = {}
    dataset._crop_v_anchor = {}
    dataset.rank = 0

    positives = 0
    for row_id, repair in bound.items():
        row = _catalog_row(catalog, row_id, textatlas_base)
        assert row["row_id"] == row_id and row["domain"] == "textatlas"
        raw = _source_bytes(row)
        repaired, expected_pixel_sha = apply_textatlas_jpeg_eoi_repair(
            repair, domain="textatlas", row=row, raw=raw
        )
        assert len(raw) == repair.raw_bytes
        assert hashlib.sha256(raw).hexdigest() == repair.raw_sha256
        assert len(repaired) == repair.repaired_bytes
        assert hashlib.sha256(repaired).hexdigest() == repair.repaired_sha256
        assert repaired == raw + b"\xff\xd9"
        with Image.open(io.BytesIO(repaired)) as image:
            rgb = image.convert("RGB")
            strict_pixels = np.asarray(rgb, dtype=np.uint8).copy()
        strict_pixel_sha = hashlib.sha256(strict_pixels.tobytes()).hexdigest()
        truncated_size, truncated_pixel_sha = _truncated_pixel_sha(raw)
        assert strict_pixel_sha == truncated_pixel_sha == expected_pixel_sha
        assert tuple(truncated_size) == (int(row["width"]), int(row["height"]))

        planned = _PlannedItem(
            step=0,
            slot=0,
            row_id=row_id,
            domain="textatlas",
            domain_idx=textatlas_idx,
            bucket_idx=int(row["bucket_idx"]),
            caption_band=-1,
        )
        prepared = _PreparedItem(planned=planned, catalog=row, caption_record={}, raw=raw)
        target_h, target_w = dataset.buckets[planned.bucket_idx]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bucket = dataset._bucket_u8_for_item(prepared)
        assert bucket.dtype == np.uint8 and bucket.shape == (target_h, target_w, 3)
        assert len(caught) == 1
        assert "frozen TextAtlas JPEG EOI repair" in str(caught[0].message)
        positives += 1

    first_id, first = next(iter(bound.items()))
    first_row = _catalog_row(catalog, first_id, textatlas_base)
    first_raw = _source_bytes(first_row)
    for field, value in (
        ("uid", "TextVisionBlend:wrong.jpg"),
        ("bucket_idx", (int(first_row["bucket_idx"]) + 1) % len(plan["buckets"])),
        ("image_row_in_group", int(first_row["image_row_in_group"]) + 1),
    ):
        drift = dict(first_row)
        drift[field] = value
        _expect_error(
            lambda drift=drift: apply_textatlas_jpeg_eoi_repair(
                first, domain="textatlas", row=drift, raw=first_raw
            ),
            "catalog binding mismatch",
        )
    _expect_error(
        lambda: apply_textatlas_jpeg_eoi_repair(
            first, domain="gpic", row=first_row, raw=first_raw
        ),
        "catalog binding mismatch",
    )
    _expect_error(
        lambda: apply_textatlas_jpeg_eoi_repair(
            first, domain="textatlas", row=first_row, raw=first_raw[:-1]
        ),
        "payload binding mismatch",
    )
    _expect_error(
        lambda: apply_textatlas_jpeg_eoi_repair(
            first, domain="textatlas", row=first_row, raw=first_raw + b"\xff\xd9"
        ),
        "payload binding mismatch",
    )
    print(
        "IMAGE_EXCEPTIONS_V1_PASS "
        f"positives={positives} negatives=8 pixel_equivalent={positives}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
