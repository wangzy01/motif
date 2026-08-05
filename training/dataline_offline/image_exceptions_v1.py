"""Frozen, production-only image-byte exceptions for exact offline plans.

The public unprefixed constants remain the released stage256 ABI and the
existing ``STAGE512A_*`` constants remain the released W48 ABI.  Stage512a W64
has a separately named production identity while retaining the same catalog
rows and exact physical payload/pixel identities for its three reviewed JPEGs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping


PLAN_ID = "stage256_offline_exact_v1_w64_prod"
STAGE = "256"
ELIGIBILITY = "elig_256"
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


# Stage512a W48 production profile.  Evidence remains separately named so the
# suggestion/review chain cannot be mistaken for the legacy stage256 audit.
STAGE512A_PLAN_ID = "stage512a_offline_exact_v2_w48_prod"
STAGE512A_STAGE = "512a"
STAGE512A_ELIGIBILITY = "elig_512"
STAGE512A_PLAN_HASH = (
    "72363201789fc72803219604887095d287b5d52bbfc415a64744e84f99899a8a"
)
STAGE512A_CATALOG_HASH = (
    "370b1679e8c5670338a03ba4c6972cdbf69fa5068fc35108bf646b2d64642331"
)
STAGE512A_PLAN_HEADER_SHA256 = (
    "939df84dd066d93c36cc10606a57e1dfc4e9d4c174f0b658c45ab2cad35cbd0d"
)
STAGE512A_CATALOG_HEADER_SHA256 = (
    "51a5f0904987a751b54842633d2968ea7f6485307850858d9472319c15748d1d"
)
STAGE512A_INTEGRITY_AUDIT_SHA256 = (
    "9d2b42eed098a1808589f418402cbfa19ffa5278852fa40d166ed31d89450a6f"
)
STAGE512A_ALLOWLIST_SUGGESTION_SHA256 = (
    "29fdce7368823a943a688434be095996d26ad4309594762f3ae9408150d2b932"
)
STAGE512A_REPAIRED_PAYLOAD_BINDINGS_SHA256 = (
    "0450ba03cfa6dfca1a2b1b3cacc27c6c6d50ae40a4eda31a7589d95913a923d5"
)
STAGE512A_FINAL_REVIEW_SHA256 = (
    "d1734949e27c4ef8cdc037e9c37dbece1ad463d2f75f01139093189f57f9c4c3"
)

STAGE512A_TEXTATLAS_JPEG_EOI_REPAIRS: Mapping[
    int, TextAtlasJpegEoiRepair
] = MappingProxyType(
    {
        68_973_396: TextAtlasJpegEoiRepair(
            row_id=68_973_396,
            uid="PPT2Structured:986e2d6790064c0d9db822e405cfd804.jpg",
            bucket_idx=182,
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
        69_010_633: TextAtlasJpegEoiRepair(
            row_id=69_010_633,
            uid="PPT2Structured:fbc3c5a38a12459b92ed77870494e609.jpg",
            bucket_idx=182,
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
        70_063_978: TextAtlasJpegEoiRepair(
            row_id=70_063_978,
            uid="TextVisionBlend:716f0b1cc7134b84bd6a098a55a70425.jpg",
            bucket_idx=182,
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
    }
)
STAGE512A_TEXTATLAS_FORMAL_EXPOSURES: Mapping[int, int] = MappingProxyType(
    {68_973_396: 4, 69_010_633: 3, 70_063_978: 3}
)
STAGE512A_TEXTATLAS_TOTAL_FORMAL_EXPOSURES = 10

# Stage512a W64 production profile.  The catalog and reviewed JPEG payloads are
# unchanged from W48.  W64 plan exposure counts and evidence SHAs are not
# runtime inputs; the W64 reviewed gate keeps them fail-closed until its plan
# audit/rebind artifacts have been published.
STAGE512A_W64_PLAN_ID = "stage512a_offline_exact_v2_w64_prod"
STAGE512A_W64_STAGE = "512a"
STAGE512A_W64_ELIGIBILITY = "elig_512"
STAGE512A_W64_PLAN_HASH = (
    "37b30d902bd46c5d6c1fe26d3b69e13d5d03d847ec3fea1a8bccdf1651c85097"
)
STAGE512A_W64_CATALOG_HASH = STAGE512A_CATALOG_HASH
STAGE512A_W64_PLAN_HEADER_SHA256 = (
    "b69e8a20331a1abb13905cbc4cf4710abb4d7d2a592a0d7a3dfcfe3db0201c75"
)
STAGE512A_W64_CATALOG_HEADER_SHA256 = STAGE512A_CATALOG_HEADER_SHA256

STAGE512A_W64_TEXTATLAS_JPEG_EOI_REPAIRS: Mapping[
    int, TextAtlasJpegEoiRepair
] = MappingProxyType(dict(STAGE512A_TEXTATLAS_JPEG_EOI_REPAIRS))


# Stage512a 2026-07-22 weighted W64 production profile.  The catalog and the
# three byte/pixel-reviewed JPEG repairs are identical to the prior W64
# release; only the immutable plan identity and observed exposure counts differ.
STAGE512A_WEIGHTED_PLAN_ID = "stage512a_offline_exact_v2_w64_256m_weighted_v1"
STAGE512A_WEIGHTED_STAGE = "512a"
STAGE512A_WEIGHTED_ELIGIBILITY = "elig_512"
STAGE512A_WEIGHTED_PLAN_HASH = (
    "7295ca937d5002a0cf5e207c1d466c5992352e395adc9ae1c33f3cb616f675db"
)
STAGE512A_WEIGHTED_CATALOG_HASH = STAGE512A_W64_CATALOG_HASH
STAGE512A_WEIGHTED_PLAN_HEADER_SHA256 = (
    "dc994b05147e33ee7515a986b3edb48e42c777b9257bc663dfa78c35a77a892d"
)
STAGE512A_WEIGHTED_CATALOG_HEADER_SHA256 = STAGE512A_W64_CATALOG_HEADER_SHA256
STAGE512A_WEIGHTED_TEXTATLAS_JPEG_EOI_REPAIRS: Mapping[
    int, TextAtlasJpegEoiRepair
] = MappingProxyType(dict(STAGE512A_W64_TEXTATLAS_JPEG_EOI_REPAIRS))
STAGE512A_WEIGHTED_TEXTATLAS_FORMAL_EXPOSURES: Mapping[int, int] = MappingProxyType(
    {68_973_396: 3, 69_010_633: 4, 70_063_978: 3}
)
STAGE512A_WEIGHTED_TEXTATLAS_TOTAL_FORMAL_EXPOSURES = 10


# Stage512b anchors_v1 production profile.  The three reviewed JPEGs carry over --
# same source bytes, same reviewed repairs, same pixel identities -- but the Stage512b
# catalog RENUMBERED the rows (85.67M textatlas base instead of 93.2M) and re-bucketed
# them under the 13-anchor scheme, and this registry is keyed by row_id.  Nobody re-keyed
# it, so bind_image_exceptions() found no Stage512b profile and returned the empty map:
# the formal run then died at plan step 5746 on row 89,134,630 (the PPT2Structured image
# below), whose truncation had been known, reviewed and repaired since stage256.
# Every byte and pixel identity here was re-measured against the Stage512b catalog and
# matched the frozen constants exactly before this profile was added (2026-08-05).
STAGE512B_ANCHORS_PLAN_ID = "stage512b_offline_exact_v2_w64_512m_anchors_v1"
STAGE512B_ANCHORS_STAGE = "512b"
STAGE512B_ANCHORS_ELIGIBILITY = "elig_512"
STAGE512B_ANCHORS_PLAN_HASH = (
    "50b5ae2bb2a952b18355d5d5876ceac8c4d8086ae72f872b306a8167f84fd685"
)
STAGE512B_ANCHORS_CATALOG_HASH = (
    "b91e4906df9a1f8b039effa69df83a0555310d964bf22d6e1da7c13e9c781b91"
)
STAGE512B_ANCHORS_PLAN_HEADER_SHA256 = (
    "5cd978e8b8392ff379c0e2961f2d515279b052cf1ab766a097fc15922349c806"
)
STAGE512B_ANCHORS_CATALOG_HEADER_SHA256 = (
    "4ab78bb39715cf596362a8e68eed0e4b77abdfbf1f44f06bbcca454aee366ed8"
)


def _rekeyed_stage512b_repair(
    old_row_id: int, new_row_id: int, new_bucket_idx: int
) -> TextAtlasJpegEoiRepair:
    """The Stage512b identity of a reviewed repair: new row_id and anchor bucket, every
    byte- and pixel-level field UNCHANGED from the frozen stage256 review."""
    import dataclasses
    base = TEXTATLAS_JPEG_EOI_REPAIRS[old_row_id]
    return dataclasses.replace(base, row_id=new_row_id, bucket_idx=new_bucket_idx)


STAGE512B_ANCHORS_TEXTATLAS_JPEG_EOI_REPAIRS: Mapping[
    int, TextAtlasJpegEoiRepair
] = MappingProxyType(
    {
        88_575_641: _rekeyed_stage512b_repair(94_682_834, 88_575_641, 0),
        89_134_630: _rekeyed_stage512b_repair(93_569_671, 89_134_630, 1),
        89_161_949: _rekeyed_stage512b_repair(93_606_908, 89_161_949, 1),
    }
)


class ImageExceptionBindingError(ValueError):
    pass


class ImageExceptionMismatchError(ValueError):
    pass


_EMPTY_EXCEPTIONS: Mapping[int, TextAtlasJpegEoiRepair] = MappingProxyType({})

# Layout mirrors bind_image_exceptions arguments and ends with its returned
# mapping.  plan_id, plan_hash, and plan-header SHA are independently unique
# production signals; mixing signals from two profiles is always fatal.
_PROFILES = (
    (
        STAGE,
        ELIGIBILITY,
        PLAN_ID,
        PLAN_HASH,
        CATALOG_HASH,
        PLAN_HEADER_SHA256,
        CATALOG_HEADER_SHA256,
        TEXTATLAS_JPEG_EOI_REPAIRS,
    ),
    (
        STAGE512A_STAGE,
        STAGE512A_ELIGIBILITY,
        STAGE512A_PLAN_ID,
        STAGE512A_PLAN_HASH,
        STAGE512A_CATALOG_HASH,
        STAGE512A_PLAN_HEADER_SHA256,
        STAGE512A_CATALOG_HEADER_SHA256,
        STAGE512A_TEXTATLAS_JPEG_EOI_REPAIRS,
    ),
    (
        STAGE512A_W64_STAGE,
        STAGE512A_W64_ELIGIBILITY,
        STAGE512A_W64_PLAN_ID,
        STAGE512A_W64_PLAN_HASH,
        STAGE512A_W64_CATALOG_HASH,
        STAGE512A_W64_PLAN_HEADER_SHA256,
        STAGE512A_W64_CATALOG_HEADER_SHA256,
        STAGE512A_W64_TEXTATLAS_JPEG_EOI_REPAIRS,
    ),
    (
        STAGE512A_WEIGHTED_STAGE,
        STAGE512A_WEIGHTED_ELIGIBILITY,
        STAGE512A_WEIGHTED_PLAN_ID,
        STAGE512A_WEIGHTED_PLAN_HASH,
        STAGE512A_WEIGHTED_CATALOG_HASH,
        STAGE512A_WEIGHTED_PLAN_HEADER_SHA256,
        STAGE512A_WEIGHTED_CATALOG_HEADER_SHA256,
        STAGE512A_WEIGHTED_TEXTATLAS_JPEG_EOI_REPAIRS,
    ),
    (
        STAGE512B_ANCHORS_STAGE,
        STAGE512B_ANCHORS_ELIGIBILITY,
        STAGE512B_ANCHORS_PLAN_ID,
        STAGE512B_ANCHORS_PLAN_HASH,
        STAGE512B_ANCHORS_CATALOG_HASH,
        STAGE512B_ANCHORS_PLAN_HEADER_SHA256,
        STAGE512B_ANCHORS_CATALOG_HEADER_SHA256,
        STAGE512B_ANCHORS_TEXTATLAS_JPEG_EOI_REPAIRS,
    ),
)


def bind_image_exceptions(
    *,
    stage: str,
    eligibility: str,
    plan_id: str,
    plan_hash: str,
    catalog_hash: str,
    plan_header_sha256: str,
    catalog_header_sha256: str,
    decode_domain: Callable[[int], str],
) -> Mapping[int, TextAtlasJpegEoiRepair]:
    actual = (
        stage,
        eligibility,
        plan_id,
        plan_hash,
        catalog_hash,
        plan_header_sha256,
        catalog_header_sha256,
    )
    matches = tuple(
        profile
        for profile in _PROFILES
        if (
            plan_id == profile[2]
            or plan_hash == profile[3]
            or plan_header_sha256 == profile[5]
        )
    )
    # Catalogs may be shared by diagnostic plans, so catalog identity alone is
    # deliberately not an activation signal.
    if not matches:
        return _EMPTY_EXCEPTIONS
    if len(matches) != 1:
        raise ImageExceptionBindingError(
            "image exception production signals select conflicting profiles"
        )
    profile = matches[0]
    expected = profile[:7]
    if actual != expected:
        raise ImageExceptionBindingError(
            "image exception production binding mismatch: "
            f"actual={actual!r}, expected={expected!r}"
        )
    repairs = profile[7]
    for row_id, repair in repairs.items():
        domain = decode_domain(row_id)
        if domain != "textatlas":
            raise ImageExceptionBindingError(
                f"image exception row {row_id}/{repair.uid} decodes to "
                f"{domain!r}, not textatlas"
            )
    return repairs


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
    "ELIGIBILITY",
    "INTEGRITY_AUDIT_SHA256",
    "ImageExceptionBindingError",
    "ImageExceptionMismatchError",
    "PLAN_HEADER_SHA256",
    "PLAN_HASH",
    "PLAN_ID",
    "STAGE",
    "REPAIRED_PAYLOAD_BINDINGS_SHA256",
    "STAGE512A_ALLOWLIST_SUGGESTION_SHA256",
    "STAGE512A_CATALOG_HASH",
    "STAGE512A_CATALOG_HEADER_SHA256",
    "STAGE512A_ELIGIBILITY",
    "STAGE512A_FINAL_REVIEW_SHA256",
    "STAGE512A_INTEGRITY_AUDIT_SHA256",
    "STAGE512A_PLAN_HASH",
    "STAGE512A_PLAN_HEADER_SHA256",
    "STAGE512A_PLAN_ID",
    "STAGE512A_REPAIRED_PAYLOAD_BINDINGS_SHA256",
    "STAGE512A_STAGE",
    "STAGE512A_TEXTATLAS_FORMAL_EXPOSURES",
    "STAGE512A_TEXTATLAS_JPEG_EOI_REPAIRS",
    "STAGE512A_TEXTATLAS_TOTAL_FORMAL_EXPOSURES",
    "STAGE512A_W64_CATALOG_HASH",
    "STAGE512A_W64_CATALOG_HEADER_SHA256",
    "STAGE512A_W64_ELIGIBILITY",
    "STAGE512A_W64_PLAN_HASH",
    "STAGE512A_W64_PLAN_HEADER_SHA256",
    "STAGE512A_W64_PLAN_ID",
    "STAGE512A_W64_STAGE",
    "STAGE512A_W64_TEXTATLAS_JPEG_EOI_REPAIRS",
    "TEXTATLAS_JPEG_EOI_REPAIRS",
    "TextAtlasJpegEoiRepair",
    "apply_textatlas_jpeg_eoi_repair",
    "bind_image_exceptions",
]
