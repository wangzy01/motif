"""Frozen, production-only image-byte exceptions for OFFLINE_ARB_V1."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping


PLAN_ID = "stage256_offline_exact_v1_w64_prod"
PLAN_HASH = "a187b1eb2f99f1b1c4c860deed369d2c04bc2a997e12249dd9566ccf3962f263"
CATALOG_HASH = "b5f83014e4eab304460336adff217c35e993a2d892a5d889f69f87013c65f212"
PLAN_HEADER_SHA256 = "e4ce718bc29e823a6ba44f8e592f6b8fb446273a5c6908eec87c5e194e3da9a9"
CATALOG_HEADER_SHA256 = "a06567fe3359efa730a0cd099ee5245b039c1648502d304f2e35f4487727a507"
INTEGRITY_AUDIT_SHA256 = "35c45f47e7a4af55da3eec22d5d4be5306f87ae589d5fd1d994aef6c346bdeac"
ALLOWLIST_SUGGESTION_SHA256 = (
    "3ca13f35569cd31ea64d2fa7f3b332465c42f98407a405129031b73783c01868"
)
REPAIRED_PAYLOAD_BINDINGS_SHA256 = (
    "e932637fe2a21227f12bbbb26d7f57021fe52d2ffda0adb1f08e60375119d1fa"
)


@dataclass(frozen=True)
class TextAtlasJpegEoiRepair:
    row_id: int
    uid: str
    bucket_idx: int
    width: int
    height: int
    image_path: str
    image_parquet_path: str
    image_row_group: int
    image_row_in_group: int
    raw_bytes: int
    raw_sha256: str
    repaired_bytes: int
    repaired_sha256: str
    pixel_sha256: str


TEXTATLAS_JPEG_EOI_REPAIRS: Mapping[int, TextAtlasJpegEoiRepair] = MappingProxyType(
    {
        94_682_834: TextAtlasJpegEoiRepair(
            row_id=94_682_834,
            uid="TextVisionBlend:716f0b1cc7134b84bd6a098a55a70425.jpg",
            bucket_idx=48,
            width=512,
            height=512,
            image_path="716f0b1cc7134b84bd6a098a55a70425.jpg",
            image_parquet_path=(
                "/s3/evan/textatlas5m/TextVisionBlend/"
                "train-00009-of-00022.parquet"
            ),
            image_row_group=192,
            image_row_in_group=4,
            raw_bytes=32_768,
            raw_sha256="1b9bec49cef9e0b04709c2447712ef9b4a3825d18d60a000facc4e3631d5363c",
            repaired_bytes=32_770,
            repaired_sha256="03e276bb716de0d10da58fef1b22dd6f8458972012cd053d2ba6c2c6cf8a9767",
            pixel_sha256="bbb0b6cf52ed0430b7d52ebf5bf578420098ba0a54fdd65be6822024ae52089a",
        ),
        93_569_671: TextAtlasJpegEoiRepair(
            row_id=93_569_671,
            uid="PPT2Structured:986e2d6790064c0d9db822e405cfd804.jpg",
            bucket_idx=48,
            width=3_600,
            height=2_700,
            image_path="986e2d6790064c0d9db822e405cfd804.jpg",
            image_parquet_path=(
                "/s3/evan/textatlas5m/PPT2Structured/"
                "train-00021-of-00036.parquet"
            ),
            image_row_group=13,
            image_row_in_group=27,
            raw_bytes=1_036_288,
            raw_sha256="2a84efd18e5c66e72cfb26f34c3195db6ed97becd02633ab7c2bd8449dcc45d2",
            repaired_bytes=1_036_290,
            repaired_sha256="0522d5ea8705494f9c8ebfb8442bacc888d6916dbc5659e22e8cde8f2ed7fc4a",
            pixel_sha256="120051cba9c51f1ad620acaf49a3ca3105845d60a947a5ef9f9a6ff69d76f79f",
        ),
        93_606_908: TextAtlasJpegEoiRepair(
            row_id=93_606_908,
            uid="PPT2Structured:fbc3c5a38a12459b92ed77870494e609.jpg",
            bucket_idx=48,
            width=3_600,
            height=2_700,
            image_path="fbc3c5a38a12459b92ed77870494e609.jpg",
            image_parquet_path=(
                "/s3/evan/textatlas5m/PPT2Structured/"
                "train-00035-of-00036.parquet"
            ),
            image_row_group=10,
            image_row_in_group=78,
            raw_bytes=786_432,
            raw_sha256="bbd73e4673977154cb76665bdbb29ca43227d710b23e77a4c2a10348883fd460",
            repaired_bytes=786_434,
            repaired_sha256="e38ba338c5c12c453e1a17fd57bdac536f165f175f0d1c3c181925bb39b3a361",
            pixel_sha256="7b44b9a43b7f221fff237018c312d696dd03cc9c1854585a770c9ba9ab6643aa",
        ),
    }
)


class ImageExceptionBindingError(ValueError):
    pass


class ImageExceptionMismatchError(ValueError):
    pass


def bind_image_exceptions(
    *,
    plan_id: str,
    plan_hash: str,
    catalog_hash: str,
    plan_header_sha256: str,
    catalog_header_sha256: str,
    decode_domain: Callable[[int], str],
) -> Mapping[int, TextAtlasJpegEoiRepair]:
    actual = (
        plan_id,
        plan_hash,
        catalog_hash,
        plan_header_sha256,
        catalog_header_sha256,
    )
    expected = (
        PLAN_ID,
        PLAN_HASH,
        CATALOG_HASH,
        PLAN_HEADER_SHA256,
        CATALOG_HEADER_SHA256,
    )
    production_signal = plan_id == PLAN_ID or plan_header_sha256 == PLAN_HEADER_SHA256
    if not production_signal:
        return MappingProxyType({})
    if actual != expected:
        raise ImageExceptionBindingError(
            "image exception production binding mismatch: "
            f"actual={actual!r}, expected={expected!r}"
        )
    for row_id, repair in TEXTATLAS_JPEG_EOI_REPAIRS.items():
        domain = decode_domain(row_id)
        if domain != "textatlas":
            raise ImageExceptionBindingError(
                f"image exception row {row_id}/{repair.uid} decodes to "
                f"{domain!r}, not textatlas"
            )
    return TEXTATLAS_JPEG_EOI_REPAIRS


def apply_textatlas_jpeg_eoi_repair(
    repair: TextAtlasJpegEoiRepair,
    *,
    domain: str,
    row: Mapping[str, Any],
    raw: bytes,
) -> tuple[bytes, str]:
    actual_fields = {
        "row_id": int(row.get("row_id", -1)),
        "uid": str(row.get("uid")),
        "bucket_idx": int(row.get("bucket_idx", -1)),
        "width": int(row.get("width", -1)),
        "height": int(row.get("height", -1)),
        "image_path": str(row.get("image_path")),
        "image_parquet_path": str(row.get("image_parquet_path")),
        "image_row_group": int(row.get("image_row_group", -1)),
        "image_row_in_group": int(row.get("image_row_in_group", -1)),
    }
    expected_fields = {
        "row_id": repair.row_id,
        "uid": repair.uid,
        "bucket_idx": repair.bucket_idx,
        "width": repair.width,
        "height": repair.height,
        "image_path": repair.image_path,
        "image_parquet_path": repair.image_parquet_path,
        "image_row_group": repair.image_row_group,
        "image_row_in_group": repair.image_row_in_group,
    }
    if domain != "textatlas" or actual_fields != expected_fields:
        raise ImageExceptionMismatchError(
            "textatlas JPEG repair catalog binding mismatch: "
            f"domain={domain!r}, actual={actual_fields!r}, expected={expected_fields!r}"
        )
    raw_sha = hashlib.sha256(raw).hexdigest()
    if (
        len(raw) != repair.raw_bytes
        or raw_sha != repair.raw_sha256
        or not raw.startswith(b"\xff\xd8\xff")
        or raw.endswith(b"\xff\xd9")
    ):
        raise ImageExceptionMismatchError(
            "textatlas JPEG repair payload binding mismatch: "
            f"row_id={repair.row_id},bytes={len(raw)},sha256={raw_sha},"
            f"head={raw[:3].hex()},tail={raw[-2:].hex()}"
        )
    repaired = raw + b"\xff\xd9"
    repaired_sha = hashlib.sha256(repaired).hexdigest()
    if len(repaired) != repair.repaired_bytes or repaired_sha != repair.repaired_sha256:
        raise ImageExceptionMismatchError(
            "textatlas JPEG repaired payload identity mismatch: "
            f"row_id={repair.row_id},bytes={len(repaired)},sha256={repaired_sha}"
        )
    return repaired, repair.pixel_sha256


__all__ = [
    "ALLOWLIST_SUGGESTION_SHA256",
    "CATALOG_HASH",
    "CATALOG_HEADER_SHA256",
    "INTEGRITY_AUDIT_SHA256",
    "ImageExceptionBindingError",
    "ImageExceptionMismatchError",
    "PLAN_HEADER_SHA256",
    "PLAN_HASH",
    "PLAN_ID",
    "REPAIRED_PAYLOAD_BINDINGS_SHA256",
    "TEXTATLAS_JPEG_EOI_REPAIRS",
    "TextAtlasJpegEoiRepair",
    "apply_textatlas_jpeg_eoi_repair",
    "bind_image_exceptions",
]
