"""Stage512b binding of the reviewed weighted-plan Anime native-upsample exception.

The image is the one Stage512a already reviewed and excepted
(anime_native_upsample_exceptions_weighted_v1._KEY): catalog metadata says 600x600, the
authentic payload decodes to 363x450, and covering a 512x512 anchor therefore requires a
1.4105x upsample that the anchors gate otherwise forbids. The Stage512b catalog
renumbered the row (11,163,647 -> 534,123) and re-bucketed it under the 13-anchor scheme
(182 -> 0), and this registry -- like the textatlas JPEG repairs before it -- is keyed by
row_id: nobody re-keyed it, the binding fell back to the empty map, and the formal run
died at plan step 11,270 on the row's first of eight appearances (2026-08-05 05:54 UTC).

Every byte- and pixel-level constant below is the 512a review value, and each was
re-measured on 2026-08-05 before this module was written: the payload fetched from the
live URL matched raw_sha256, decode matched decoded_rgb_sha256 at 450x363x3, and
resize_u8_to_bucket(decoded, 512, 512, 0.15) matched output_rgb_sha256 exactly. The
Stage512b anchor 0 is 512x512 and the anime crop_v_anchor is 0.15, so the pinned
transform is bit-identical to the reviewed one.

The eight coordinates are plan indices measured by scanning all 245 plan blocks for
row 534,123 (zero occurrences before 11,270 -- consistent with the run dying on the
first). Waived exposure: 8 of 512,000,000 = 1.56e-8, recorded as an explicit waiver in
docs/STAGE512B_DECISIONS.md. Every other below-anchor row remains fail-closed.

A separate module rather than an entry in the 512a table because that module's closure
checks pin EXPOSURE_COUNT and FORMAL_STEP_STOP=125,000; appending a 250k-horizon entry
would either trip them or require loosening reviewed invariants.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from training.dataline_offline.anime_native_upsample_exceptions_weighted_v1 import (
    ANIME_NATIVE_UPSAMPLE_EXCEPTIONS as _STAGE512A_EXCEPTIONS,
    AnimeNativeUpsampleBindingError,
    AnimeNativeUpsampleException,
    AnimePlanCoordinate,
)

STAGE = "512b"
ELIGIBILITY = "elig_512"
PLAN_ID = "stage512b_offline_exact_v2_w64_512m_anchors_v1"
PLAN_HASH = "50b5ae2bb2a952b18355d5d5876ceac8c4d8086ae72f872b306a8167f84fd685"
CATALOG_HASH = "b91e4906df9a1f8b039effa69df83a0555310d964bf22d6e1da7c13e9c781b91"
PLAN_HEADER_SHA256 = "5cd978e8b8392ff379c0e2961f2d515279b052cf1ab766a097fc15922349c806"
CATALOG_HEADER_SHA256 = "4ab78bb39715cf596362a8e68eed0e4b77abdfbf1f44f06bbcca454aee366ed8"
FORMAL_STEP_STOP = 250_000
WORLD_SIZE = 64
PER_RANK_BATCH = 32

_SOURCE = _STAGE512A_EXCEPTIONS[11_163_647]

# Plan indices (0-based, the loader's planned.step), from a full scan of all 245 blocks.
_COORDINATES = (
    (11_270, 30, 3),
    (46_194, 53, 1),
    (97_356, 38, 0),
    (116_413, 48, 1),
    (151_593, 17, 0),
    (184_345, 57, 3),
    (209_406, 44, 0),
    (236_229, 21, 1),
)
EXPOSURE_COUNT = len(_COORDINATES)

STAGE512B_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS: Mapping[
    int, AnimeNativeUpsampleException
] = MappingProxyType(
    {
        534_123: AnimeNativeUpsampleException(
            row_id=534_123,
            coordinates=frozenset(
                AnimePlanCoordinate(step, rank, slot)
                for step, rank, slot in _COORDINATES
            ),
            domain=_SOURCE.domain,
            uid=_SOURCE.uid,
            key=_SOURCE.key,
            fetch_url=_SOURCE.fetch_url,
            key_decoded=_SOURCE.key_decoded,
            source_chunk=_SOURCE.source_chunk,
            source_offset=_SOURCE.source_offset,
            # The 13-anchor scheme's square anchor; the ONLY field besides row_id and
            # the coordinates that differs from the reviewed 512a entry.
            bucket_idx=0,
            catalog_width=_SOURCE.catalog_width,
            catalog_height=_SOURCE.catalog_height,
            cluster=_SOURCE.cluster,
            rating_bucket=_SOURCE.rating_bucket,
            year=_SOURCE.year,
            caption_parquet_path=_SOURCE.caption_parquet_path,
            caption_row_group=_SOURCE.caption_row_group,
            caption_row_in_group=_SOURCE.caption_row_in_group,
            raw_bytes=_SOURCE.raw_bytes,
            raw_sha256=_SOURCE.raw_sha256,
            decoded_width=_SOURCE.decoded_width,
            decoded_height=_SOURCE.decoded_height,
            decoded_rgb_sha256=_SOURCE.decoded_rgb_sha256,
            target_width=_SOURCE.target_width,
            target_height=_SOURCE.target_height,
            crop_v_anchor=_SOURCE.crop_v_anchor,
            output_rgb_sha256=_SOURCE.output_rgb_sha256,
        )
    }
)


_EMPTY: Mapping[int, AnimeNativeUpsampleException] = MappingProxyType({})


def bind_stage512b_anime_native_upsample_exceptions(
    *,
    stage: str,
    eligibility: str,
    plan_id: str,
    plan_hash: str,
    catalog_hash: str,
    plan_header_sha256: str,
    catalog_header_sha256: str,
    decode_domain,
) -> Mapping[int, AnimeNativeUpsampleException]:
    """Same activation contract as the 512a bind: selected by plan identity, and once
    selected every identity field must match exactly or the bind fails closed. A separate
    function because the 512a module's closure checks pin ITS horizon and exposure count
    as module globals."""
    selected = (
        plan_id == PLAN_ID
        or plan_hash == PLAN_HASH
        or plan_header_sha256 == PLAN_HEADER_SHA256
    )
    if not selected:
        return _EMPTY
    actual = (stage, eligibility, plan_id, plan_hash, catalog_hash,
              plan_header_sha256, catalog_header_sha256)
    expected = (STAGE, ELIGIBILITY, PLAN_ID, PLAN_HASH, CATALOG_HASH,
                PLAN_HEADER_SHA256, CATALOG_HEADER_SHA256)
    if actual != expected:
        raise AnimeNativeUpsampleBindingError(
            f"stage512b anime native-upsample production binding mismatch: {actual!r}"
        )
    coordinates = [
        item
        for exception in STAGE512B_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS.values()
        for item in exception.coordinates
    ]
    if len(coordinates) != len(set(coordinates)) or len(coordinates) != EXPOSURE_COUNT:
        raise AnimeNativeUpsampleBindingError("stage512b coordinate closure drift")
    for row_id, exception in STAGE512B_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS.items():
        if exception.row_id != row_id:
            raise AnimeNativeUpsampleBindingError("stage512b exception row-id drift")
        for item in exception.coordinates:
            if not (0 <= item.step < FORMAL_STEP_STOP
                    and 0 <= item.rank < WORLD_SIZE
                    and 0 <= item.slot < PER_RANK_BATCH):
                raise AnimeNativeUpsampleBindingError(
                    "stage512b exception coordinate out of range"
                )
        if exception.decoded_width >= exception.target_width and \
           exception.decoded_height >= exception.target_height:
            raise AnimeNativeUpsampleBindingError(
                "stage512b exception does not require upsampling"
            )
        if decode_domain(row_id) != "anime":
            raise AnimeNativeUpsampleBindingError("stage512b exception row domain drift")
    return STAGE512B_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS
