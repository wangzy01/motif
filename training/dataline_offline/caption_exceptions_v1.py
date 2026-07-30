"""Frozen, production-only caption exceptions for exact offline plans.

The original stage256 profile and its public constants are an ABI: they remain
byte-for-byte values from the 384M-slot audit.  Stage512a W48 is also a frozen
ABI.  The W64 formal plan therefore has a third, separately named profile even
though it deliberately reuses the exact W48 catalog rows.  No profile activates
unless an exact production-plan identity and every catalog binding match.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Mapping


PLAN_ID = "stage256_offline_exact_v1_w64_prod"
STAGE = "256"
ELIGIBILITY = "elig_256"
CATALOG_HASH = "b5f83014e4eab304460336adff217c35e993a2d892a5d889f69f87013c65f212"
PLAN_HEADER_SHA256 = "e4ce718bc29e823a6ba44f8e592f6b8fb446273a5c6908eec87c5e194e3da9a9"
CATALOG_HEADER_SHA256 = "a06567fe3359efa730a0cd099ee5245b039c1648502d304f2e35f4487727a507"
AUDIT_JSON_SHA256 = "102895529c7427fb675d29d59eb66e0c39c29b63465df538bcbfa819153b639e"

CAPTIONLESS_MONET_ROWS: Mapping[int, str] = MappingProxyType(
    {
        18_907_181: "monet:058_6f9a1ddf56d9e468",
        23_881_582: "monet:023_a11e3e8b8dc53b6a",
        23_881_911: "monet:023_aaa4de30fe312a90",
        31_217_524: "monet:023_6af995e4741f26cd",
        31_221_132: "monet:023_df29c87ed5939c37",
        32_217_542: "monet:058_9ca806c37710fd48",
        35_492_824: "monet:056_dbaeba5eada7a499",
        40_078_044: "monet:008_b7c97448d7784e9a",
    }
)

# Stage512a W48 production profile.  These constants are deliberately not
# aliases of the legacy names above: callers that import the old names must
# continue to observe the released stage256 ABI.
STAGE512A_PLAN_ID = "stage512a_offline_exact_v2_w48_prod"
STAGE512A_STAGE = "512a"
STAGE512A_ELIGIBILITY = "elig_512"
STAGE512A_CATALOG_HASH = (
    "370b1679e8c5670338a03ba4c6972cdbf69fa5068fc35108bf646b2d64642331"
)
STAGE512A_PLAN_HEADER_SHA256 = (
    "939df84dd066d93c36cc10606a57e1dfc4e9d4c174f0b658c45ab2cad35cbd0d"
)
STAGE512A_CATALOG_HEADER_SHA256 = (
    "51a5f0904987a751b54842633d2968ea7f6485307850858d9472319c15748d1d"
)
STAGE512A_SIDECAR_HEADER_SHA256 = (
    "72fe0c4d5b5258a1a6320a6ab57f422cd815dd26a8a4a29594fefdd7abfa7224"
)
STAGE512A_AUDIT_JSON_SHA256 = (
    "9c1d809365de3badefe011d192b197615ff13504dbe470a2c4387636f358b0ad"
)

STAGE512A_CAPTIONLESS_MONET_ROWS: Mapping[int, str] = MappingProxyType(
    {
        19_760_606: "monet:058_6f9a1ddf56d9e468",
        22_783_601: "monet:023_a11e3e8b8dc53b6a",
        29_778_338: "monet:023_6af995e4741f26cd",
        29_779_948: "monet:023_df29c87ed5939c37",
        30_281_293: "monet:058_9ca806c37710fd48",
        35_574_104: "monet:056_dbaeba5eada7a499",
    }
)
STAGE512A_CAPTIONLESS_MONET_EXPOSURES: Mapping[int, int] = MappingProxyType(
    {
        19_760_606: 3,
        22_783_601: 3,
        29_778_338: 4,
        29_779_948: 3,
        30_281_293: 3,
        35_574_104: 3,
    }
)
STAGE512A_CAPTIONLESS_MONET_TOTAL_EXPOSURES = 19

# Stage512a W64 production profile.  The catalog is unchanged from W48, so the
# reviewed row/UID policy is identical; only the exact production-plan signals
# differ.  Exposure counts are evidence, not runtime activation inputs, and are
# intentionally kept in the W64 gate contract until the W64 plan audit exists.
STAGE512A_W64_PLAN_ID = "stage512a_offline_exact_v2_w64_prod"
STAGE512A_W64_STAGE = "512a"
STAGE512A_W64_ELIGIBILITY = "elig_512"
STAGE512A_W64_CATALOG_HASH = STAGE512A_CATALOG_HASH
STAGE512A_W64_PLAN_HEADER_SHA256 = (
    "b69e8a20331a1abb13905cbc4cf4710abb4d7d2a592a0d7a3dfcfe3db0201c75"
)
STAGE512A_W64_CATALOG_HEADER_SHA256 = STAGE512A_CATALOG_HEADER_SHA256

STAGE512A_W64_CAPTIONLESS_MONET_ROWS: Mapping[int, str] = MappingProxyType(
    {
        19_760_606: "monet:058_6f9a1ddf56d9e468",
        22_783_601: "monet:023_a11e3e8b8dc53b6a",
        29_778_338: "monet:023_6af995e4741f26cd",
        29_779_948: "monet:023_df29c87ed5939c37",
        30_281_293: "monet:058_9ca806c37710fd48",
        35_574_104: "monet:056_dbaeba5eada7a499",
    }
)

# ``_MONET_SRC_COL`` has six codes because code 7 reuses Florence's physical
# field for the first-sentence variant.  All six decoded values must be empty.
MONET_CAPTION_CODES = frozenset({1, 2, 3, 4, 5, 7})


class CaptionExceptionBindingError(ValueError):
    pass


_EMPTY_EXCEPTIONS: Mapping[int, str] = MappingProxyType({})

# Tuple layout intentionally mirrors the public bind arguments, followed by the
# immutable row map.  Only plan_id/header are activation signals: diagnostic
# plans are allowed to reuse a production catalog without inheriting policy.
_PROFILES = (
    (
        STAGE,
        ELIGIBILITY,
        PLAN_ID,
        CATALOG_HASH,
        PLAN_HEADER_SHA256,
        CATALOG_HEADER_SHA256,
        CAPTIONLESS_MONET_ROWS,
    ),
    (
        STAGE512A_STAGE,
        STAGE512A_ELIGIBILITY,
        STAGE512A_PLAN_ID,
        STAGE512A_CATALOG_HASH,
        STAGE512A_PLAN_HEADER_SHA256,
        STAGE512A_CATALOG_HEADER_SHA256,
        STAGE512A_CAPTIONLESS_MONET_ROWS,
    ),
    (
        STAGE512A_W64_STAGE,
        STAGE512A_W64_ELIGIBILITY,
        STAGE512A_W64_PLAN_ID,
        STAGE512A_W64_CATALOG_HASH,
        STAGE512A_W64_PLAN_HEADER_SHA256,
        STAGE512A_W64_CATALOG_HEADER_SHA256,
        STAGE512A_W64_CAPTIONLESS_MONET_ROWS,
    ),
)


def bind_caption_exceptions(
    *,
    stage: str,
    eligibility: str,
    plan_id: str,
    catalog_hash: str,
    plan_header_sha256: str,
    catalog_header_sha256: str,
    decode_domain: Callable[[int], str],
) -> Mapping[int, str]:
    """Return the frozen exceptions only for their exact production snapshot.

    Unrelated diagnostic plans get no exceptions.  A partial identity match is
    treated as stale/tampered production state and fails closed.
    """

    actual = (
        stage,
        eligibility,
        plan_id,
        catalog_hash,
        plan_header_sha256,
        catalog_header_sha256,
    )
    matches = tuple(
        profile
        for profile in _PROFILES
        if plan_id == profile[2] or plan_header_sha256 == profile[4]
    )
    # Diagnostic/smoke plans intentionally reuse immutable production
    # catalogs.  Catalog identity alone is therefore not an activation signal.
    if not matches:
        return _EMPTY_EXCEPTIONS
    if len(matches) != 1:
        raise CaptionExceptionBindingError(
            "caption exception production signals select conflicting profiles"
        )
    profile = matches[0]
    expected = profile[:6]
    if actual != expected:
        raise CaptionExceptionBindingError(
            "caption exception production binding mismatch: "
            f"actual={actual!r}, expected={expected!r}"
        )
    rows = profile[6]
    for row_id, uid in rows.items():
        domain = decode_domain(row_id)
        if domain != "monet_real":
            raise CaptionExceptionBindingError(
                f"caption exception row {row_id}/{uid} decodes to {domain!r}, not monet_real"
            )
    return rows


__all__ = [
    "AUDIT_JSON_SHA256",
    "CAPTIONLESS_MONET_ROWS",
    "CATALOG_HASH",
    "CATALOG_HEADER_SHA256",
    "ELIGIBILITY",
    "CaptionExceptionBindingError",
    "MONET_CAPTION_CODES",
    "PLAN_HEADER_SHA256",
    "PLAN_ID",
    "STAGE",
    "STAGE512A_AUDIT_JSON_SHA256",
    "STAGE512A_CAPTIONLESS_MONET_EXPOSURES",
    "STAGE512A_CAPTIONLESS_MONET_ROWS",
    "STAGE512A_CAPTIONLESS_MONET_TOTAL_EXPOSURES",
    "STAGE512A_CATALOG_HASH",
    "STAGE512A_CATALOG_HEADER_SHA256",
    "STAGE512A_ELIGIBILITY",
    "STAGE512A_PLAN_HEADER_SHA256",
    "STAGE512A_PLAN_ID",
    "STAGE512A_SIDECAR_HEADER_SHA256",
    "STAGE512A_STAGE",
    "STAGE512A_W64_CAPTIONLESS_MONET_ROWS",
    "STAGE512A_W64_CATALOG_HASH",
    "STAGE512A_W64_CATALOG_HEADER_SHA256",
    "STAGE512A_W64_ELIGIBILITY",
    "STAGE512A_W64_PLAN_HEADER_SHA256",
    "STAGE512A_W64_PLAN_ID",
    "STAGE512A_W64_STAGE",
    "bind_caption_exceptions",
]
