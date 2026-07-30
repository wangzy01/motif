"""Reviewed weighted-plan Anime native-upsample exception.

The mapping is enabled only for one exact plan/catalog/revalidation identity.
Every coordinate, catalog field, raw payload, decoded RGB, transform, and output
is pinned.  Rows outside the mapping retain the loader's fail-closed policy.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True, order=True)
class AnimePlanCoordinate:
    step: int
    rank: int
    slot: int


@dataclass(frozen=True)
class AnimeNativeUpsampleException:
    row_id: int
    coordinates: frozenset[AnimePlanCoordinate]
    domain: str
    uid: str
    key: str
    fetch_url: str
    key_decoded: str
    source_chunk: str
    source_offset: int
    bucket_idx: int
    catalog_width: int
    catalog_height: int
    cluster: int
    rating_bucket: str
    year: int
    caption_parquet_path: str
    caption_row_group: int
    caption_row_in_group: int
    raw_bytes: int
    raw_sha256: str
    decoded_width: int
    decoded_height: int
    decoded_rgb_sha256: str
    target_width: int
    target_height: int
    crop_v_anchor: float
    output_rgb_sha256: str


STAGE = "512a"
ELIGIBILITY = "elig_512"
PLAN_ID = "stage512a_offline_exact_v2_w64_256m_weighted_v1"
PLAN_HASH = "7295ca937d5002a0cf5e207c1d466c5992352e395adc9ae1c33f3cb616f675db"
CATALOG_HASH = "370b1679e8c5670338a03ba4c6972cdbf69fa5068fc35108bf646b2d64642331"
PLAN_HEADER_SHA256 = "dc994b05147e33ee7515a986b3edb48e42c777b9257bc663dfa78c35a77a892d"
CATALOG_HEADER_SHA256 = "51a5f0904987a751b54842633d2968ea7f6485307850858d9472319c15748d1d"
REVALIDATION_REPORT_SHA256 = "dd1d5ffc9a519a3c5522fb4296fba2e9f9788efbcac6a79f0d73e484cd0ee26c"
FORMAL_STEP_STOP = 125_000
WORLD_SIZE = 64
PER_RANK_BATCH = 32

_KEY = "a2Vtb25vX2NodW5rXzAwMDEyJDE4NTMxMTQzNjgkMzMxNzA"
_URL = "https://img.metanomaly.co/r2/" + _KEY
_COORDINATES = ((6032, 52, 0), (38063, 9, 1), (73429, 12, 2), (93835, 41, 0), (113350, 7, 2))
EXPOSURE_COUNT = len(_COORDINATES)

ANIME_NATIVE_UPSAMPLE_EXCEPTIONS: Mapping[int, AnimeNativeUpsampleException] = (
    MappingProxyType(
        {
            11_163_647: AnimeNativeUpsampleException(
                row_id=11_163_647,
                coordinates=frozenset(
                    AnimePlanCoordinate(step, rank, slot)
                    for step, rank, slot in _COORDINATES
                ),
                domain="anime",
                uid="anime:" + _KEY,
                key=_KEY,
                fetch_url=_URL,
                key_decoded="kemono_chunk_00012$1853114368$33170",
                source_chunk="kemono_chunk_00012",
                source_offset=1_853_114_368,
                bucket_idx=182,
                catalog_width=600,
                catalog_height=600,
                cluster=1_179,
                rating_bucket="sensitive",
                year=2_022,
                caption_parquet_path=(
                    "/s3/evan/anime_core_ditv04_ct_soft/data/00000012.parquet"
                ),
                caption_row_group=3,
                caption_row_in_group=100_655,
                raw_bytes=33_170,
                raw_sha256=(
                    "694c10f206645325d965a3743f959f17f5061ece114277d9dfa84ea1039e13b5"
                ),
                decoded_width=363,
                decoded_height=450,
                decoded_rgb_sha256=(
                    "0410e541e2da917ae4a22946ea70306a74805cc274fbb231d74ab853d3e68097"
                ),
                target_width=512,
                target_height=512,
                crop_v_anchor=0.15,
                output_rgb_sha256=(
                    "27b07f2cf60a891fd4816fff80ccdd26eb2478ab60e58fac41bcce7fb89bb3aa"
                ),
            )
        }
    )
)


class AnimeNativeUpsampleBindingError(ValueError):
    pass


class AnimeNativeUpsampleMismatchError(ValueError):
    pass


_EMPTY: Mapping[int, AnimeNativeUpsampleException] = MappingProxyType({})
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")


def _strict_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise AnimeNativeUpsampleMismatchError(
            f"anime native-upsample {name} must be exact int"
        )
    return value


def _strict_float(value: Any, name: str) -> float:
    if type(value) is not float:
        raise AnimeNativeUpsampleMismatchError(
            f"anime native-upsample {name} must be exact float"
        )
    return value


def _validate_definition(
    mapping_key: int, exception: AnimeNativeUpsampleException
) -> None:
    if type(mapping_key) is not int or mapping_key != exception.row_id:
        raise AnimeNativeUpsampleBindingError("anime exception row-id drift")
    if (
        not exception.coordinates
        or len(exception.coordinates) != EXPOSURE_COUNT
        or any(
            type(item) is not AnimePlanCoordinate
            or not 0 <= item.step < FORMAL_STEP_STOP
            or not 0 <= item.rank < WORLD_SIZE
            or not 0 <= item.slot < PER_RANK_BATCH
            for item in exception.coordinates
        )
    ):
        raise AnimeNativeUpsampleBindingError("anime exception coordinate drift")
    if exception.domain != "anime" or exception.uid != "anime:" + exception.key:
        raise AnimeNativeUpsampleBindingError("anime exception domain/key drift")
    if exception.fetch_url != "https://img.metanomaly.co/r2/" + exception.key:
        raise AnimeNativeUpsampleBindingError("anime exception URL drift")
    padding = "=" * ((4 - len(exception.key) % 4) % 4)
    decoded = base64.b64decode(
        exception.key + padding, altchars=b"-_", validate=True
    ).decode("utf-8")
    if decoded != exception.key_decoded:
        raise AnimeNativeUpsampleBindingError("anime exception key decode drift")
    chunk, offset, byte_count = decoded.split("$")
    if (
        chunk != exception.source_chunk
        or offset != str(exception.source_offset)
        or byte_count != str(exception.raw_bytes)
    ):
        raise AnimeNativeUpsampleBindingError("anime exception source locator drift")
    for name in (
        "raw_sha256",
        "decoded_rgb_sha256",
        "output_rgb_sha256",
    ):
        if _LOWER_SHA256.fullmatch(getattr(exception, name)) is None:
            raise AnimeNativeUpsampleBindingError(f"invalid SHA-256: {name}")
    if (
        exception.decoded_width >= exception.target_width
        and exception.decoded_height >= exception.target_height
    ):
        raise AnimeNativeUpsampleBindingError("exception does not require upsampling")


def bind_anime_native_upsample_exceptions(
    *,
    stage: str,
    eligibility: str,
    plan_id: str,
    plan_hash: str,
    catalog_hash: str,
    plan_header_sha256: str,
    catalog_header_sha256: str,
    decode_domain: Callable[[int], str],
) -> Mapping[int, AnimeNativeUpsampleException]:
    actual = (
        stage,
        eligibility,
        plan_id,
        plan_hash,
        catalog_hash,
        plan_header_sha256,
        catalog_header_sha256,
    )
    expected = (
        STAGE,
        ELIGIBILITY,
        PLAN_ID,
        PLAN_HASH,
        CATALOG_HASH,
        PLAN_HEADER_SHA256,
        CATALOG_HEADER_SHA256,
    )
    selected = (
        plan_id == PLAN_ID
        or plan_hash == PLAN_HASH
        or plan_header_sha256 == PLAN_HEADER_SHA256
    )
    if not selected:
        return _EMPTY
    if actual != expected:
        raise AnimeNativeUpsampleBindingError(
            f"anime native-upsample production binding mismatch: {actual!r}"
        )
    if (
        _LOWER_SHA256.fullmatch(REVALIDATION_REPORT_SHA256) is None
        or EXPOSURE_COUNT < 1
    ):
        raise AnimeNativeUpsampleBindingError("revalidation authority is incomplete")
    coordinates = [
        item
        for exception in ANIME_NATIVE_UPSAMPLE_EXCEPTIONS.values()
        for item in exception.coordinates
    ]
    if len(coordinates) != len(set(coordinates)) or len(coordinates) != EXPOSURE_COUNT:
        raise AnimeNativeUpsampleBindingError("coordinate closure drift")
    for row_id, exception in ANIME_NATIVE_UPSAMPLE_EXCEPTIONS.items():
        _validate_definition(row_id, exception)
        if decode_domain(row_id) != "anime":
            raise AnimeNativeUpsampleBindingError("exception row domain drift")
    return ANIME_NATIVE_UPSAMPLE_EXCEPTIONS


def apply_anime_native_upsample_exception(
    exception: AnimeNativeUpsampleException,
    *,
    step: int,
    rank: int,
    slot: int,
    domain: str,
    row: Mapping[str, Any],
    raw: bytes,
    decoded_rgb: np.ndarray,
    target_height: int,
    target_width: int,
    crop_v_anchor: float,
    resize_u8_to_bucket: Callable[[np.ndarray, int, int, float], np.ndarray],
) -> np.ndarray:
    _validate_definition(exception.row_id, exception)
    coordinate = AnimePlanCoordinate(
        _strict_int(step, "step"),
        _strict_int(rank, "rank"),
        _strict_int(slot, "slot"),
    )
    if coordinate not in exception.coordinates:
        raise AnimeNativeUpsampleMismatchError(
            f"anime native-upsample coordinate mismatch: {coordinate!r}"
        )
    expected_catalog = {
        "row_id": exception.row_id,
        "uid": exception.uid,
        "domain": exception.domain,
        "bucket_idx": exception.bucket_idx,
        "width": exception.catalog_width,
        "height": exception.catalog_height,
        "key": exception.key,
        "fetch_url": exception.fetch_url,
        "cluster": exception.cluster,
        "rating_bucket": exception.rating_bucket,
        "year": exception.year,
        "caption_parquet_path": exception.caption_parquet_path,
        "caption_row_group": exception.caption_row_group,
        "caption_row_in_group": exception.caption_row_in_group,
    }
    actual_catalog: dict[str, Any] = {}
    for name, expected_value in expected_catalog.items():
        actual_value = row.get(name)
        if type(actual_value) is not type(expected_value):
            raise AnimeNativeUpsampleMismatchError(
                f"anime native-upsample catalog field type drift: {name}"
            )
        actual_catalog[name] = actual_value
    if domain != exception.domain or actual_catalog != expected_catalog:
        raise AnimeNativeUpsampleMismatchError(
            "anime native-upsample catalog binding mismatch"
        )
    if type(raw) is not bytes:
        raise AnimeNativeUpsampleMismatchError("raw payload is not exact bytes")
    raw_sha = hashlib.sha256(raw).hexdigest()
    if len(raw) != exception.raw_bytes or raw_sha != exception.raw_sha256:
        raise AnimeNativeUpsampleMismatchError(
            f"anime native-upsample raw payload drift: {len(raw)}/{raw_sha}"
        )
    if type(decoded_rgb) is not np.ndarray or decoded_rgb.dtype != np.uint8:
        raise AnimeNativeUpsampleMismatchError("decoded RGB type drift")
    decoded_sha = hashlib.sha256(decoded_rgb.tobytes(order="C")).hexdigest()
    if (
        decoded_rgb.shape
        != (exception.decoded_height, exception.decoded_width, 3)
        or decoded_sha != exception.decoded_rgb_sha256
    ):
        raise AnimeNativeUpsampleMismatchError(
            f"anime native-upsample decoded RGB drift: {decoded_rgb.shape}/{decoded_sha}"
        )
    target_h = _strict_int(target_height, "target_height")
    target_w = _strict_int(target_width, "target_width")
    anchor = _strict_float(crop_v_anchor, "crop_v_anchor")
    if (
        target_h != exception.target_height
        or target_w != exception.target_width
        or anchor != exception.crop_v_anchor
    ):
        raise AnimeNativeUpsampleMismatchError("target transform drift")
    output = resize_u8_to_bucket(decoded_rgb, target_h, target_w, anchor)
    output_sha = hashlib.sha256(output.tobytes(order="C")).hexdigest()
    if (
        type(output) is not np.ndarray
        or output.dtype != np.uint8
        or output.shape != (target_h, target_w, 3)
        or output_sha != exception.output_rgb_sha256
    ):
        raise AnimeNativeUpsampleMismatchError(
            f"anime native-upsample output drift: {output.shape}/{output_sha}"
        )
    return output


__all__ = [
    "ANIME_NATIVE_UPSAMPLE_EXCEPTIONS",
    "AnimeNativeUpsampleBindingError",
    "AnimeNativeUpsampleException",
    "AnimeNativeUpsampleMismatchError",
    "AnimePlanCoordinate",
    "EXPOSURE_COUNT",
    "REVALIDATION_REPORT_SHA256",
    "apply_anime_native_upsample_exception",
    "bind_anime_native_upsample_exceptions",
]
