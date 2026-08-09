"""Fresh Stage512a-control binding for the reviewed Anime upsample waiver.

The physical row and transform were already reviewed for the released weighted
Stage512a run.  This module changes only the immutable plan identity and the six
coordinates found by the sealed full-256M control-plan scan.  A separate module
keeps the old five-coordinate ABI and its default closure untouched.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Callable, Mapping

from training.dataline_offline.anime_native_upsample_exceptions_weighted_v1 import (
    ANIME_NATIVE_UPSAMPLE_EXCEPTIONS as _STAGE512A_WEIGHTED_EXCEPTIONS,
    AnimeNativeUpsampleBindingError,
    AnimeNativeUpsampleException,
    AnimePlanCoordinate,
    _validate_definition,
)


STAGE = "512a"
ELIGIBILITY = "elig_512"
PLAN_ID = (
    "stage512a_offline_exact_v2_continue_control_rngfix_v1_"
    "w64_256m_seed20260710"
)
PLAN_HASH = "fbb5fcbd91d80daff8dfe89693f38eb3fa6ddeb657d01198df7fcac9cdcd039c"
CATALOG_HASH = "370b1679e8c5670338a03ba4c6972cdbf69fa5068fc35108bf646b2d64642331"
PLAN_HEADER_SHA256 = (
    "5fbe7ca546d3e77ef37d95bf52c7ded2f9741d0b4377fbadb8b059e9b9a1dce0"
)
CATALOG_HEADER_SHA256 = (
    "51a5f0904987a751b54842633d2968ea7f6485307850858d9472319c15748d1d"
)
PLAN_CLOSURE_REPORT_SHA256 = (
    "cb92872fc8f13cae10bfafaeb3238c9cc0159bf0c31a096e85ee9b188eb53104"
)
CANONICAL_PLAN_RECEIPT_SHA256 = (
    "5178207100c8af20230489dac88b9118665c93abb33ab5c97b6ccc532b2ea6ca"
)
FORMAL_STEP_STOP = 125_000
WORLD_SIZE = 64
PER_RANK_BATCH = 32

_COORDINATES = (
    (11_486, 35, 1),
    (50_429, 7, 2),
    (60_035, 46, 0),
    (95_244, 16, 1),
    (119_964, 33, 0),
    (121_904, 58, 1),
)
EXPOSURE_COUNT = len(_COORDINATES)
CLOSURE = (EXPOSURE_COUNT, FORMAL_STEP_STOP, WORLD_SIZE, PER_RANK_BATCH)

_SOURCE = _STAGE512A_WEIGHTED_EXCEPTIONS[11_163_647]
STAGE512A_CONTROL_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS: Mapping[
    int, AnimeNativeUpsampleException
] = MappingProxyType(
    {
        11_163_647: dataclasses.replace(
            _SOURCE,
            coordinates=frozenset(
                AnimePlanCoordinate(step, rank, slot)
                for step, rank, slot in _COORDINATES
            ),
        )
    }
)

_EMPTY: Mapping[int, AnimeNativeUpsampleException] = MappingProxyType({})


def bind_stage512a_control_anime_native_upsample_exceptions(
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
    """Bind only the exact fresh control plan; partial identity matches are fatal."""

    selected = (
        plan_id == PLAN_ID
        or plan_hash == PLAN_HASH
        or plan_header_sha256 == PLAN_HEADER_SHA256
    )
    if not selected:
        return _EMPTY
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
    if actual != expected:
        raise AnimeNativeUpsampleBindingError(
            "stage512a-control anime native-upsample production binding mismatch: "
            f"{actual!r}"
        )
    coordinates = [
        coordinate
        for exception in STAGE512A_CONTROL_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS.values()
        for coordinate in exception.coordinates
    ]
    if len(coordinates) != EXPOSURE_COUNT or len(coordinates) != len(set(coordinates)):
        raise AnimeNativeUpsampleBindingError(
            "stage512a-control anime coordinate closure drift"
        )
    for row_id, exception in (
        STAGE512A_CONTROL_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS.items()
    ):
        _validate_definition(row_id, exception, CLOSURE)
        if decode_domain(row_id) != "anime":
            raise AnimeNativeUpsampleBindingError(
                "stage512a-control anime exception row domain drift"
            )
    return STAGE512A_CONTROL_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS


__all__ = [
    "CANONICAL_PLAN_RECEIPT_SHA256",
    "CATALOG_HASH",
    "CATALOG_HEADER_SHA256",
    "CLOSURE",
    "ELIGIBILITY",
    "EXPOSURE_COUNT",
    "FORMAL_STEP_STOP",
    "PER_RANK_BATCH",
    "PLAN_CLOSURE_REPORT_SHA256",
    "PLAN_HASH",
    "PLAN_HEADER_SHA256",
    "PLAN_ID",
    "STAGE",
    "STAGE512A_CONTROL_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS",
    "WORLD_SIZE",
    "bind_stage512a_control_anime_native_upsample_exceptions",
]
