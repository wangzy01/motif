"""Single fill-point for the stage512a W64 reviewed-exception evidence.

The values below are bound to the immutable W64 plan/audit and catalog-level
TextAtlas rebind artifacts.  They are intentionally independent of the W48
counts.  Every W64 publisher or actual-source preflight calls
:func:`finalized_contract` before it can claim authority.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any, Mapping


PLAN_ID = "stage512a_offline_exact_v2_w64_prod"
PLAN_HASH = "37b30d902bd46c5d6c1fe26d3b69e13d5d03d847ec3fea1a8bccdf1651c85097"
PLAN_HEADER_SHA256 = (
    "b69e8a20331a1abb13905cbc4cf4710abb4d7d2a592a0d7a3dfcfe3db0201c75"
)
CATALOG_HASH = "370b1679e8c5670338a03ba4c6972cdbf69fa5068fc35108bf646b2d64642331"
CATALOG_HEADER_SHA256 = (
    "51a5f0904987a751b54842633d2968ea7f6485307850858d9472319c15748d1d"
)
ASSIGNMENT_HASH = (
    "1b7da5dd6863df7cd0ec102ad7ef717cdb80aec4b40bfa354bed5dedaa6d30c8"
)
SIDECAR_HEADER_SHA256 = (
    "72fe0c4d5b5258a1a6320a6ab57f422cd815dd26a8a4a29594fefdd7abfa7224"
)

PLAN_HORIZON = 131_072
FORMAL_STEPS = 125_000
WORLD_SIZE = 64
PER_RANK_BS = 32
GLOBAL_BATCH = 2_048
FORMAL_EXPOSURES = 256_000_000
BLOCK_STEPS = 1_024
NUM_BLOCKS = 128

MONET_ROWS = (
    19_760_606,
    22_783_601,
    29_778_338,
    29_779_948,
    30_281_293,
    35_574_104,
)
TEXTATLAS_REPAIR_ROWS = (68_973_396, 69_010_633, 70_063_978)

# Exact immutable artifact digests.
PLAN_CHECKSUMS_SHA256 = (
    "7f5663eac314798cbeca6e48725f4f4739f96fef80b12470e6dabd080ba93ba2"
)
CAPTION_AUDIT_SHA256 = (
    "70d93e400641efde1175bda34dc5e98fe36d4efe178551726534caea5130c70d"
)
CAPTION_HORIZON_AUDIT_SHA256 = (
    "f1949ba0e9cda18a95695437a4db33625c5b45cb8ed9b4bd2d339ae0fd716a7e"
)
PLAN_ROWS_SHA256 = (
    "cbfebdb64dd25c5eedd2564f07a02b07bd1ea144fc3c13ee7af5893391470f99"
)
PLAN_BITMAP_SHA256 = (
    "1c9d1c6935ec19f83029968fbc2dbf5f471975269cc116d1772aede9fd1aa4bd"
)
TEXTATLAS_REBIND_SHA256 = (
    "d6dd59144eb8864a1ce15003fd0a4a8eb6684f5dc9aeaed4dc97d20c92786494"
)
TEXTATLAS_REBIND_RECEIPT_SHA256 = (
    "bc49d926e14d76e688bba456e7c55b98465587b0e49659b584df40ba46fb740d"
)

# Exact W64 FORMAL-prefix exposure maps.  Do not copy W48 counts: the topology,
# seed namespace and prefix length changed.
MONET_FORMAL_EXPOSURES: Mapping[int, Any] = MappingProxyType(
    {
        19_760_606: 3,
        22_783_601: 4,
        29_778_338: 3,
        29_779_948: 3,
        30_281_293: 3,
        35_574_104: 3,
    }
)
MONET_TOTAL_EXPOSURES: Mapping[int, Any] = MappingProxyType(
    {
        19_760_606: 3,
        22_783_601: 4,
        29_778_338: 4,
        29_779_948: 3,
        30_281_293: 3,
        35_574_104: 3,
    }
)
TEXTATLAS_FORMAL_EXPOSURES: Mapping[int, Any] = MappingProxyType(
    {
        68_973_396: 3,
        69_010_633: 3,
        70_063_978: 3,
    }
)
TEXTATLAS_TOTAL_EXPOSURES: Mapping[int, Any] = MappingProxyType(
    {
        68_973_396: 4,
        69_010_633: 3,
        70_063_978: 3,
    }
)

# Aggregate exposure counts over all 43 catalog-level strict-decode candidates
# and their reviewed 40-success/3-repair partition.
TEXTATLAS_CANDIDATE_FORMAL_EXPOSURES: Any = 134
TEXTATLAS_CANDIDATE_TOTAL_EXPOSURES: Any = 137
TEXTATLAS_PRODUCTION_SUCCESS_FORMAL_EXPOSURES: Any = 125
TEXTATLAS_PRODUCTION_SUCCESS_TOTAL_EXPOSURES: Any = 127


class W64ExceptionContractPendingError(RuntimeError):
    """Raised when any W64 audit-derived value is invalid or inconsistent."""


def _sha(value: Any, label: str, pending: list[str]) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        pending.append(label)
        return ""
    return value


def _count(value: Any, label: str, pending: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        pending.append(label)
        return 0
    return value


def _count_map(
    value: Mapping[int, Any],
    expected_rows: tuple[int, ...],
    label: str,
    pending: list[str],
) -> dict[int, int]:
    if set(value) != set(expected_rows):
        pending.append(f"{label}.row_set")
    return {
        row_id: _count(value.get(row_id), f"{label}[{row_id}]", pending)
        for row_id in expected_rows
    }


def finalized_contract() -> dict[str, Any]:
    """Return normalized evidence values or fail with every unfinished field."""

    pending: list[str] = []
    hashes = {
        "plan_checksums_sha256": _sha(
            PLAN_CHECKSUMS_SHA256, "plan_checksums_sha256", pending
        ),
        "caption_audit_sha256": _sha(
            CAPTION_AUDIT_SHA256, "caption_audit_sha256", pending
        ),
        "caption_horizon_audit_sha256": _sha(
            CAPTION_HORIZON_AUDIT_SHA256,
            "caption_horizon_audit_sha256",
            pending,
        ),
        "plan_rows_sha256": _sha(
            PLAN_ROWS_SHA256, "plan_rows_sha256", pending
        ),
        "plan_bitmap_sha256": _sha(
            PLAN_BITMAP_SHA256, "plan_bitmap_sha256", pending
        ),
        "textatlas_rebind_sha256": _sha(
            TEXTATLAS_REBIND_SHA256,
            "textatlas_rebind_sha256",
            pending,
        ),
        "textatlas_rebind_receipt_sha256": _sha(
            TEXTATLAS_REBIND_RECEIPT_SHA256,
            "textatlas_rebind_receipt_sha256",
            pending,
        ),
    }
    monet_formal = _count_map(
        MONET_FORMAL_EXPOSURES,
        MONET_ROWS,
        "monet_formal_exposures",
        pending,
    )
    monet_total = _count_map(
        MONET_TOTAL_EXPOSURES,
        MONET_ROWS,
        "monet_total_exposures",
        pending,
    )
    textatlas_formal = _count_map(
        TEXTATLAS_FORMAL_EXPOSURES,
        TEXTATLAS_REPAIR_ROWS,
        "textatlas_formal_exposures",
        pending,
    )
    textatlas_total = _count_map(
        TEXTATLAS_TOTAL_EXPOSURES,
        TEXTATLAS_REPAIR_ROWS,
        "textatlas_total_exposures",
        pending,
    )
    aggregate = {
        "candidate_formal": _count(
            TEXTATLAS_CANDIDATE_FORMAL_EXPOSURES,
            "textatlas_candidate_formal_exposures",
            pending,
        ),
        "candidate_total": _count(
            TEXTATLAS_CANDIDATE_TOTAL_EXPOSURES,
            "textatlas_candidate_total_exposures",
            pending,
        ),
        "production_success_formal": _count(
            TEXTATLAS_PRODUCTION_SUCCESS_FORMAL_EXPOSURES,
            "textatlas_production_success_formal_exposures",
            pending,
        ),
        "production_success_total": _count(
            TEXTATLAS_PRODUCTION_SUCCESS_TOTAL_EXPOSURES,
            "textatlas_production_success_total_exposures",
            pending,
        ),
    }
    if pending:
        raise W64ExceptionContractPendingError(
            "W64 exception evidence is not finalized: " + ", ".join(sorted(pending))
        )

    if any(monet_formal[row] > monet_total[row] for row in MONET_ROWS):
        raise W64ExceptionContractPendingError(
            "W64 Monet FORMAL exposure exceeds horizon exposure"
        )

    repaired_formal = sum(textatlas_formal.values())
    repaired_total = sum(textatlas_total.values())
    if any(textatlas_formal[row] > textatlas_total[row] for row in TEXTATLAS_REPAIR_ROWS):
        raise W64ExceptionContractPendingError(
            "W64 TextAtlas FORMAL exposure exceeds horizon exposure"
        )
    if aggregate["candidate_formal"] != (
        aggregate["production_success_formal"] + repaired_formal
    ):
        raise W64ExceptionContractPendingError(
            "W64 TextAtlas FORMAL candidate partition does not close"
        )
    if aggregate["candidate_total"] != (
        aggregate["production_success_total"] + repaired_total
    ):
        raise W64ExceptionContractPendingError(
            "W64 TextAtlas horizon candidate partition does not close"
        )
    if aggregate["candidate_formal"] > aggregate["candidate_total"]:
        raise W64ExceptionContractPendingError(
            "W64 TextAtlas FORMAL aggregate exceeds horizon aggregate"
        )
    if aggregate["production_success_formal"] > aggregate["production_success_total"]:
        raise W64ExceptionContractPendingError(
            "W64 TextAtlas production-success FORMAL aggregate exceeds horizon aggregate"
        )

    # These are the exact runtime-active projection consumed by the final gate.
    # The rebind document itself deliberately calls the three repair rows
    # ``review`` rows until this gate has closed runtime/source provenance.
    gate_counts = {
        "monet_empty_prompt_rows": len(MONET_ROWS),
        "monet_empty_prompt_formal_exposures": sum(monet_formal.values()),
        "monet_empty_prompt_horizon_exposures": sum(monet_total.values()),
        "textatlas_raw_candidate_rows": 43,
        "textatlas_raw_candidate_formal_exposures": aggregate[
            "candidate_formal"
        ],
        "textatlas_raw_candidate_horizon_exposures": aggregate[
            "candidate_total"
        ],
        "textatlas_production_decode_success_rows": 40,
        "textatlas_production_decode_success_formal_exposures": aggregate[
            "production_success_formal"
        ],
        "textatlas_production_decode_success_horizon_exposures": aggregate[
            "production_success_total"
        ],
        "textatlas_repaired_rows": len(TEXTATLAS_REPAIR_ROWS),
        "textatlas_repaired_formal_exposures": repaired_formal,
        "textatlas_repaired_horizon_exposures": repaired_total,
    }

    return {
        "hashes": hashes,
        "monet_formal_exposures": monet_formal,
        "monet_total_exposures": monet_total,
        "textatlas_formal_exposures": textatlas_formal,
        "textatlas_total_exposures": textatlas_total,
        "textatlas_aggregate_exposures": aggregate,
        "gate_counts": gate_counts,
    }


__all__ = [
    "ASSIGNMENT_HASH",
    "BLOCK_STEPS",
    "CAPTION_AUDIT_SHA256",
    "CAPTION_HORIZON_AUDIT_SHA256",
    "CATALOG_HASH",
    "CATALOG_HEADER_SHA256",
    "FORMAL_EXPOSURES",
    "FORMAL_STEPS",
    "GLOBAL_BATCH",
    "MONET_ROWS",
    "MONET_FORMAL_EXPOSURES",
    "MONET_TOTAL_EXPOSURES",
    "NUM_BLOCKS",
    "PER_RANK_BS",
    "PLAN_HASH",
    "PLAN_CHECKSUMS_SHA256",
    "PLAN_HEADER_SHA256",
    "PLAN_HORIZON",
    "PLAN_ID",
    "PLAN_BITMAP_SHA256",
    "PLAN_ROWS_SHA256",
    "SIDECAR_HEADER_SHA256",
    "TEXTATLAS_REPAIR_ROWS",
    "TEXTATLAS_REBIND_RECEIPT_SHA256",
    "TEXTATLAS_REBIND_SHA256",
    "WORLD_SIZE",
    "W64ExceptionContractPendingError",
    "finalized_contract",
]
