"""Frozen, production-only caption exceptions for OFFLINE_ARB_V1.

These eight rows are the complete result of the 384M-slot read-only audit
bound below.  The module does not make an exception active unless every live
production plan/catalog identity matches and every listed row decodes to
``monet_real``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Mapping


PLAN_ID = "stage256_offline_exact_v1_w64_prod"
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

# ``_MONET_SRC_COL`` has six codes because code 7 reuses Florence's physical
# field for the first-sentence variant.  All six decoded values must be empty.
MONET_CAPTION_CODES = frozenset({1, 2, 3, 4, 5, 7})


class CaptionExceptionBindingError(ValueError):
    pass


def bind_caption_exceptions(
    *,
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

    actual = (plan_id, catalog_hash, plan_header_sha256, catalog_header_sha256)
    expected = (PLAN_ID, CATALOG_HASH, PLAN_HEADER_SHA256, CATALOG_HEADER_SHA256)
    # Diagnostic/smoke plans intentionally reuse the immutable production
    # catalog.  Only a production *plan* identity may activate this policy;
    # catalog identity alone must leave unrelated plans exception-free.
    production_signal = plan_id == PLAN_ID or plan_header_sha256 == PLAN_HEADER_SHA256
    if not production_signal:
        return MappingProxyType({})
    if actual != expected:
        raise CaptionExceptionBindingError(
            "caption exception production binding mismatch: "
            f"actual={actual!r}, expected={expected!r}"
        )
    for row_id, uid in CAPTIONLESS_MONET_ROWS.items():
        domain = decode_domain(row_id)
        if domain != "monet_real":
            raise CaptionExceptionBindingError(
                f"caption exception row {row_id}/{uid} decodes to {domain!r}, not monet_real"
            )
    return CAPTIONLESS_MONET_ROWS


__all__ = [
    "AUDIT_JSON_SHA256",
    "CAPTIONLESS_MONET_ROWS",
    "CATALOG_HASH",
    "CATALOG_HEADER_SHA256",
    "CaptionExceptionBindingError",
    "MONET_CAPTION_CODES",
    "PLAN_HEADER_SHA256",
    "PLAN_ID",
    "bind_caption_exceptions",
]
