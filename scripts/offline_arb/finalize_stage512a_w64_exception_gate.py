#!/usr/bin/env python3
"""Publish the exact reviewed stage512a W64 exception launch authority.

This is additive to the released W48 gate.  It accepts only the exact W64
plan, both W64 caption scans, the generic-coordinate plan-row audit, and the
catalog-level TextAtlas evidence rebind.  The legacy W48 stage1/review chain is
authority only through that rebind's immutable closure; it is never relabelled
as W64 evidence here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.offline_arb.immutable_review_evidence import (
    EvidencePublishError,
    STAGE512A_W64_REVIEWED_GATE_BINDING_LABELS,
    absolute_lexical,
    load_stage512a_write_producer_provenance,
    read_stable_bytes,
    require_real_directory,
    require_real_file,
    validate_plan_coordinates,
    validate_sha256,
)
from scripts.offline_arb.stage512a_w64_exception_contract import (
    ASSIGNMENT_HASH,
    BLOCK_STEPS,
    CATALOG_HASH,
    CATALOG_HEADER_SHA256,
    FORMAL_EXPOSURES,
    FORMAL_STEPS,
    GLOBAL_BATCH,
    MONET_ROWS,
    NUM_BLOCKS,
    PER_RANK_BS,
    PLAN_HASH,
    PLAN_HEADER_SHA256,
    PLAN_HORIZON,
    PLAN_ID,
    SIDECAR_HEADER_SHA256,
    TEXTATLAS_REPAIR_ROWS,
    WORLD_SIZE,
    finalized_contract,
)


IDENTITY = {
    "stage": "512a",
    "image_size": 512,
    "eligibility": "elig_512",
    "requires_nl": False,
    "anime_nl_filter_applied": False,
}
DOMAINS = ("anime", "monet_real", "gpic", "synthetic", "textatlas")
CATALOG_ROWS = 70_367_899
TEXTATLAS_ROWS = 4_036_998
TEXTATLAS_ROW_BASE = 66_330_901
SCHEMA = "offline_arb_stage512a_w64_reviewed_exception_gate_v1"
POLICY = "reviewed_exact_stage512a_w64_exceptions_v1"
MONET_POLICY = "exact_rows_empty_prompt_v1"
TEXTATLAS_POLICY = "append_exactly_one_jpeg_eoi_v1"

# Keep all rebind summary spelling in one place.  The rebind publisher is being
# hardened independently; if one of its descriptive field names changes, this
# is the sole name-routing table that must be reviewed and adjusted here.
REBIND_SUMMARY_FIELDS = {
    "candidate_rows": "candidate_rows",
    "planned_candidate_rows": "w64_planned_candidate_rows",
    "formal_candidate_rows": "w64_formal_candidate_rows",
    "candidate_horizon_exposures": "w64_total_candidate_exposures",
    "candidate_formal_exposures": "w64_formal_candidate_exposures",
    "production_success_rows": "production_success_rows",
    "production_failure_rows": "production_failure_rows",
    "pixel_equivalent_repair_rows": (
        "pixel_equivalent_eoi_repair_review_rows"
    ),
    "textatlas_audited_candidate_rows": "textatlas_audited_candidate_rows",
    "textatlas_formal_candidate_rows": "textatlas_formal_candidate_rows",
    "textatlas_candidate_formal_exposures": (
        "textatlas_raw_candidate_formal_exposures"
    ),
    "textatlas_candidate_horizon_exposures": (
        "textatlas_raw_candidate_horizon_exposures"
    ),
    "textatlas_production_success_rows": (
        "textatlas_production_decode_success_rows"
    ),
    "textatlas_production_success_formal_exposures": (
        "textatlas_production_decode_success_formal_exposures"
    ),
    "textatlas_production_success_horizon_exposures": (
        "textatlas_production_decode_success_horizon_exposures"
    ),
    "textatlas_repair_policy_rows": "textatlas_repair_review_rows",
    "textatlas_repaired_formal_rows": "textatlas_repair_review_formal_rows",
    "textatlas_repaired_formal_exposures": (
        "textatlas_repair_review_formal_exposures"
    ),
    "textatlas_repaired_horizon_exposures": (
        "textatlas_repair_review_horizon_exposures"
    ),
    "runtime_policy_activated": "runtime_policy_activated",
}

BASE = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop"
)
REPO = Path("/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop_stage512a_w64")
EXPECTED_OUTPUT = (
    BASE / "offline_arb_exceptions_stage512a_w64_reviewed_v1/FORMAL_PASS.json"
)
EXPECTED_TEMPLATE = REPO / "jobs/offline_arb_stage512a_w64_reviewed_exception_gate.yaml"
EXPECTED_FINALIZER = REPO / "scripts/offline_arb/finalize_stage512a_w64_exception_gate.py"
EXPECTED_HELPER = REPO / "scripts/offline_arb/immutable_review_evidence.py"
EXPECTED_RENDERER = REPO / "scripts/offline_arb/render_stage512a_write_job.py"
EXPECTED_IMAGE = (
    "471112743779.dkr.ecr.us-west-2.amazonaws.com/"
    "autobuild:py312-torch211-cu129"
)

PLAN_ROOT = BASE / "offline_arb_plans_stage512a_w64_v2/w64_prod/plan"
CATALOG_ROOT = BASE / "offline_arb_catalog_stage512a_v1"
SIDECAR_ROOT = BASE / "offline_arb_caption_sidecar_stage512a_v1"
AUDIT_ROOT = BASE / "offline_arb_exceptions_audit_stage512a_w64_v1"
REBIND_ROOT = BASE / "offline_arb_textatlas_catalog_evidence_rebind_stage512a_w64_v1"
EXPECTED_PRODUCER_BINDINGS: dict[str, Path] = {
    "plan_header": PLAN_ROOT / "header.json",
    "plan_checksums": PLAN_ROOT / "checksums.json",
    "catalog_header": CATALOG_ROOT / "header.json",
    "sidecar_header": SIDECAR_ROOT / "header.json",
    "caption_audit": AUDIT_ROOT / "caption_availability.json",
    "caption_horizon_audit": AUDIT_ROOT / "caption_availability_horizon.json",
    "plan_rows": AUDIT_ROOT / "plan_rows.json",
    "plan_bitmap": AUDIT_ROOT / "plan_rows_bitmap.npz",
    "textatlas_rebind": REBIND_ROOT / "textatlas_catalog_evidence_rebind.json",
    "textatlas_rebind_receipt": REBIND_ROOT / "rebind_receipt.json",
    "textatlas_rebind_success": REBIND_ROOT / "_SUCCESS",
    "w64_exception_contract": (
        REPO / "scripts/offline_arb/stage512a_w64_exception_contract.py"
    ),
    "offline_dataset_runtime": REPO / "training/dataline_offline/offline_dataset.py",
    "caption_runtime": REPO / "training/dataline_offline/caption_exceptions_v1.py",
    "image_runtime": REPO / "training/dataline_offline/image_exceptions_v1.py",
    "image_fetch_runtime": REPO / "training/dataline/image_fetch.py",
    "targeted_test_runtime": (
        REPO / "scripts/offline_arb/test_stage512a_w64_reviewed_exceptions.py"
    ),
    "gate_finalizer": EXPECTED_FINALIZER,
    "gate_evidence_helper": EXPECTED_HELPER,
    "write_renderer": EXPECTED_RENDERER,
}

MONET_UIDS = {
    19_760_606: "monet:058_6f9a1ddf56d9e468",
    22_783_601: "monet:023_a11e3e8b8dc53b6a",
    29_778_338: "monet:023_6af995e4741f26cd",
    29_779_948: "monet:023_df29c87ed5939c37",
    30_281_293: "monet:058_9ca806c37710fd48",
    35_574_104: "monet:056_dbaeba5eada7a499",
}

TEXTATLAS_REPAIRS: dict[int, dict[str, Any]] = {
    68_973_396: {
        "uid": "PPT2Structured:986e2d6790064c0d9db822e405cfd804.jpg",
        "bucket_idx": 182,
        "width": 3_600,
        "height": 2_700,
        "image_path": "986e2d6790064c0d9db822e405cfd804.jpg",
        "image_parquet_path": (
            "/s3/evan/textatlas5m/PPT2Structured/train-00021-of-00036.parquet"
        ),
        "image_row_group": 13,
        "image_row_in_group": 27,
        "raw_bytes": 1_036_288,
        "raw_sha256": "2a84efd18e5c66e72cfb26f34c3195db6ed97becd02633ab7c2bd8449dcc45d2",
        "repaired_bytes": 1_036_290,
        "repaired_sha256": "0522d5ea8705494f9c8ebfb8442bacc888d6916dbc5659e22e8cde8f2ed7fc4a",
        "pixel_sha256": "120051cba9c51f1ad620acaf49a3ca3105845d60a947a5ef9f9a6ff69d76f79f",
    },
    69_010_633: {
        "uid": "PPT2Structured:fbc3c5a38a12459b92ed77870494e609.jpg",
        "bucket_idx": 182,
        "width": 3_600,
        "height": 2_700,
        "image_path": "fbc3c5a38a12459b92ed77870494e609.jpg",
        "image_parquet_path": (
            "/s3/evan/textatlas5m/PPT2Structured/train-00035-of-00036.parquet"
        ),
        "image_row_group": 10,
        "image_row_in_group": 78,
        "raw_bytes": 786_432,
        "raw_sha256": "bbd73e4673977154cb76665bdbb29ca43227d710b23e77a4c2a10348883fd460",
        "repaired_bytes": 786_434,
        "repaired_sha256": "e38ba338c5c12c453e1a17fd57bdac536f165f175f0d1c3c181925bb39b3a361",
        "pixel_sha256": "7b44b9a43b7f221fff237018c312d696dd03cc9c1854585a770c9ba9ab6643aa",
    },
    70_063_978: {
        "uid": "TextVisionBlend:716f0b1cc7134b84bd6a098a55a70425.jpg",
        "bucket_idx": 182,
        "width": 512,
        "height": 512,
        "image_path": "716f0b1cc7134b84bd6a098a55a70425.jpg",
        "image_parquet_path": (
            "/s3/evan/textatlas5m/TextVisionBlend/train-00009-of-00022.parquet"
        ),
        "image_row_group": 192,
        "image_row_in_group": 4,
        "raw_bytes": 32_768,
        "raw_sha256": "1b9bec49cef9e0b04709c2447712ef9b4a3825d18d60a000facc4e3631d5363c",
        "repaired_bytes": 32_770,
        "repaired_sha256": "03e276bb716de0d10da58fef1b22dd6f8458972012cd053d2ba6c2c6cf8a9767",
        "pixel_sha256": "bbb0b6cf52ed0430b7d52ebf5bf578420098ba0a54fdd65be6822024ae52089a",
    },
}


class GateError(RuntimeError):
    pass


def _required(value: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in value:
        raise GateError(f"required field missing: {label}.{key}")
    return value[key]


def _mapping(value: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    observed = _required(value, key, label)
    if not isinstance(observed, Mapping):
        raise GateError(f"{label}.{key} must be an object")
    return observed


def _list(value: Mapping[str, Any], key: str, label: str) -> list[Any]:
    observed = _required(value, key, label)
    if not isinstance(observed, list):
        raise GateError(f"{label}.{key} must be a list")
    return observed


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GateError(f"{label} must be an integer >= {minimum}")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise GateError(f"{label} must be a lowercase SHA-256")
    try:
        return validate_sha256(value, label)
    except EvidencePublishError as exc:
        raise GateError(str(exc)) from exc


def _json_snapshot(
    path: Path, label: str, expected_sha256: str | None = None
) -> tuple[dict[str, Any], dict[str, str], bytes]:
    real = require_real_file(path, label)
    raw = read_stable_bytes(real, label)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise GateError(
            f"{label} SHA drift: observed={digest},expected={expected_sha256}"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} JSON root must be an object")
    return value, {"path": str(real), "sha256": digest}, raw


def _file_snapshot(
    path: Path, label: str, expected_sha256: str | None = None
) -> tuple[dict[str, str], bytes]:
    real = require_real_file(path, label)
    raw = read_stable_bytes(real, label)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise GateError(
            f"{label} SHA drift: observed={digest},expected={expected_sha256}"
        )
    return {"path": str(real), "sha256": digest}, raw


def _identity(value: Mapping[str, Any], label: str) -> None:
    observed = {key: _required(value, key, label) for key in IDENTITY}
    if observed != IDENTITY:
        raise GateError(f"{label} stage identity drift: {observed!r}")


def _validate_plan_catalog(
    plan: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    _identity(plan, "plan")
    _identity(catalog, "catalog")
    expected_plan = {
        "format_version": 2,
        "plan_id": PLAN_ID,
        "plan_hash": PLAN_HASH,
        "catalog_hash": CATALOG_HASH,
        "assignment_hash": ASSIGNMENT_HASH,
        "total_steps": PLAN_HORIZON,
        "block_steps": BLOCK_STEPS,
        "num_blocks": NUM_BLOCKS,
        "world_size": WORLD_SIZE,
        "per_rank_bs": PER_RANK_BS,
        "G": GLOBAL_BATCH,
        "global_batch_size": GLOBAL_BATCH,
        "total_exposures": PLAN_HORIZON * GLOBAL_BATCH,
    }
    for key, expected in expected_plan.items():
        if _required(plan, key, "plan") != expected:
            raise GateError(f"plan {key} drift")
    if _list(plan, "domains", "plan") != list(DOMAINS):
        raise GateError("plan domains/order drift")
    raw_domain_exposures = _list(plan, "D", "plan")
    if len(raw_domain_exposures) != len(DOMAINS) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in raw_domain_exposures
    ):
        raise GateError("plan D must contain five non-negative integer exposures")
    if sum(raw_domain_exposures) != PLAN_HORIZON * GLOBAL_BATCH:
        raise GateError("plan D does not cover the complete materialized horizon")
    domain_totals = _mapping(plan, "domain_totals", "plan")
    if set(domain_totals) != set(DOMAINS):
        raise GateError("plan domain_totals domain set drift")
    domain_exposures = {
        domain: _integer(
            _required(domain_totals, domain, "plan.domain_totals"),
            f"plan.domain_totals.{domain}",
        )
        for domain in DOMAINS
    }
    if [domain_exposures[domain] for domain in DOMAINS] != raw_domain_exposures:
        raise GateError("plan D/domain_totals drift")
    if _required(catalog, "catalog_hash", "catalog") != CATALOG_HASH:
        raise GateError("catalog hash drift")
    if _required(catalog, "assignment_hash", "catalog") != ASSIGNMENT_HASH:
        raise GateError("catalog assignment drift")
    if _integer(_required(catalog, "row_count", "catalog"), "catalog.row_count") != CATALOG_ROWS:
        raise GateError("catalog row count drift")
    if _list(catalog, "domain_order", "catalog") != list(DOMAINS):
        raise GateError("catalog domain order drift")
    files = _mapping(catalog, "files", "catalog")
    if set(files) != set(DOMAINS):
        raise GateError("catalog domain file set drift")
    catalog_file_rows = {
        domain: _integer(
            _required(_mapping(files, domain, "catalog.files"), "rows", f"catalog.files.{domain}"),
            f"catalog.files.{domain}.rows",
            minimum=1,
        )
        for domain in DOMAINS
    }
    if sum(catalog_file_rows.values()) != CATALOG_ROWS:
        raise GateError("catalog domain row counts do not close")
    if catalog_file_rows["textatlas"] != TEXTATLAS_ROWS:
        raise GateError("catalog TextAtlas row count drift")
    return {
        "topology": {
            "plan_horizon": PLAN_HORIZON,
            "formal_steps": FORMAL_STEPS,
            "world_size": WORLD_SIZE,
            "per_rank_bs": PER_RANK_BS,
            "global_batch": GLOBAL_BATCH,
            "formal_exposures": FORMAL_EXPOSURES,
        },
        "domain_exposures": domain_exposures,
    }


def _validate_caption_audit(
    value: Mapping[str, Any],
    *,
    steps: int,
    expected_exposures: Mapping[int, int],
    label: str,
) -> dict[int, int]:
    if (
        _required(value, "status", label) != "PASS"
        or _required(value, "audit", label)
        != "offline_arb_planned_caption_availability_v1"
        or _required(value, "read_only", label) is not True
    ):
        raise GateError(f"{label} status/schema/read_only drift")
    provenance = _mapping(value, "provenance", label)
    expected_provenance = {
        "plan_id": PLAN_ID,
        "plan_hash": PLAN_HASH,
        "plan_header_sha256": PLAN_HEADER_SHA256,
        "catalog_hash": CATALOG_HASH,
        "catalog_header_sha256": CATALOG_HEADER_SHA256,
        "sidecar_header_sha256": SIDECAR_HEADER_SHA256,
    }
    for key, expected in expected_provenance.items():
        if _required(provenance, key, f"{label}.provenance") != expected:
            raise GateError(f"{label} {key} binding drift")
    scan = _mapping(value, "scan", label)
    expected_scan = {
        "steps": steps,
        "world_size": WORLD_SIZE,
        "per_rank_bs": PER_RANK_BS,
        "global_batch_size": GLOBAL_BATCH,
        "exposures": steps * GLOBAL_BATCH,
        "blocks": (steps + BLOCK_STEPS - 1) // BLOCK_STEPS,
    }
    for key, expected in expected_scan.items():
        if _integer(_required(scan, key, f"{label}.scan"), f"{label}.scan.{key}") != expected:
            raise GateError(f"{label} scan {key} drift")
    _integer(_required(scan, "workers", f"{label}.scan"), f"{label}.scan.workers", minimum=1)
    rows = _list(value, "unavailable_rows", label)
    observed: dict[int, int] = {}
    observed_uids: dict[int, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise GateError(f"{label}.unavailable_rows[{index}] is not an object")
        row_id = _integer(row.get("row_id"), f"{label} row_id")
        if row.get("domain") != "monet_real" or row_id in observed:
            raise GateError(f"{label} unavailable row set drift")
        observed[row_id] = _integer(row.get("plan_exposures"), f"{label} exposures")
        observed_uids[row_id] = str(row.get("uid"))
        if row.get("served_mask") != 0 or row.get("available_bands") != []:
            raise GateError(f"{label} reviewed Monet row is no longer captionless")
    if observed != dict(expected_exposures) or observed_uids != MONET_UIDS:
        raise GateError(f"{label} exact Monet row/exposure policy drift")
    if set(observed) != set(MONET_ROWS):
        raise GateError(f"{label} Monet row set differs from contract")
    return observed


def _validate_plan_rows(
    value: Mapping[str, Any], bitmap_sha256: str
) -> dict[str, Any]:
    if (
        value.get("status") != "PASS"
        or value.get("audit") != "offline_arb_textatlas_plan_rows_generic_v1"
        or value.get("read_only") is not True
        or value.get("coordinate_schema")
        != {"name": "generic_v1", "within_formal_field": "within_formal"}
    ):
        raise GateError("W64 plan-row audit status/schema drift")
    _identity(_mapping(value, "stage_identity", "plan_rows"), "plan_rows.stage_identity")
    plan = _mapping(value, "plan", "plan_rows")
    expected_plan = {
        "plan_id": PLAN_ID,
        "plan_hash": PLAN_HASH,
        "header_sha256": PLAN_HEADER_SHA256,
        "catalog_hash": CATALOG_HASH,
        "assignment_hash": ASSIGNMENT_HASH,
        "total_steps": PLAN_HORIZON,
        "formal_steps": FORMAL_STEPS,
        "world_size": WORLD_SIZE,
        "per_rank_bs": PER_RANK_BS,
    }
    for key, expected in expected_plan.items():
        if plan.get(key) != expected:
            raise GateError(f"W64 plan-row audit plan.{key} drift")
    catalog = _mapping(value, "catalog", "plan_rows")
    expected_catalog = {
        "catalog_hash": CATALOG_HASH,
        "assignment_hash": ASSIGNMENT_HASH,
        "header_sha256": CATALOG_HEADER_SHA256,
        "textatlas_row_base": TEXTATLAS_ROW_BASE,
        "textatlas_rows": TEXTATLAS_ROWS,
    }
    for key, expected in expected_catalog.items():
        if catalog.get(key) != expected:
            raise GateError(f"W64 plan-row audit catalog.{key} drift")
    textatlas = _mapping(value, "textatlas", "plan_rows")
    expected_textatlas = {
        "formal_plan_exposures": 12_799_845,
        "formal_plan_unique_rows": TEXTATLAS_ROWS,
        "total_plan_exposures": 13_421_773,
        "total_plan_unique_rows": TEXTATLAS_ROWS,
    }
    if dict(textatlas) != expected_textatlas:
        raise GateError("W64 plan-row TextAtlas coverage drift")
    bitmap = _mapping(value, "bitmap", "plan_rows")
    if bitmap.get("sha256") != bitmap_sha256:
        raise GateError("W64 plan-row embedded bitmap SHA drift")
    # The frozen part1 document records its pre-publication work path.  The
    # canonical bitmap is independently read and SHA-bound by the gate; path
    # identity is intentionally not rewritten after publication.
    if not isinstance(bitmap.get("path"), str) or not bitmap.get("path"):
        raise GateError("W64 plan-row bitmap provenance path is missing")
    return expected_textatlas


def _reject_legacy_coordinates(value: Any, label: str = "rebind") -> None:
    if isinstance(value, Mapping):
        if "within_formal_166667" in value:
            raise GateError(f"{label} leaked legacy within_formal_166667")
        for key, child in value.items():
            _reject_legacy_coordinates(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_legacy_coordinates(child, f"{label}[{index}]")


def _validate_repair_projection(row: Mapping[str, Any], row_id: int) -> None:
    expected = TEXTATLAS_REPAIRS[row_id]
    locator = _mapping(row, "locator", f"rebind row {row_id}")
    observed = {
        "uid": row.get("uid"),
        "bucket_idx": row.get("bucket_idx"),
        "width": row.get("catalog_width"),
        "height": row.get("catalog_height"),
        "image_path": locator.get("image_path"),
        "image_parquet_path": locator.get("image_parquet_path"),
        "image_row_group": locator.get("image_row_group"),
        "image_row_in_group": locator.get("image_row_in_group"),
        "raw_bytes": row.get("payload_bytes"),
        "raw_sha256": row.get("payload_sha256"),
    }
    for key in observed:
        if observed[key] != expected[key]:
            raise GateError(f"W64 repair row {row_id} {key} drift")
    review = _mapping(row, "legacy_production_decoder_review", f"rebind row {row_id}")
    if (
        review.get("classification")
        != "repair_candidate_pixel_equivalent_review_only"
        or review.get("pixel_load_succeeded") is not False
        or review.get("runtime_policy_activated") is not False
    ):
        raise GateError(f"W64 repair row {row_id} production review drift")
    repair = _mapping(review, "repair_candidate", f"rebind row {row_id}.review")
    expected_repair = {
        "repaired_bytes": expected["repaired_bytes"],
        "repaired_payload_sha256": expected["repaired_sha256"],
        "repaired_rgb_sha256": expected["pixel_sha256"],
        "repaired_decoded_mode": "RGB",
        "repaired_decoded_size": [expected["width"], expected["height"]],
        "pixel_equivalent_to_stage1_expected": True,
        "runtime_policy_activated": False,
        "status": "PIXEL_EQUIVALENT_REVIEW_ONLY",
    }
    if dict(repair) != expected_repair:
        raise GateError(f"W64 repair row {row_id} pixel-equivalence drift")


def _validate_rebind(
    value: Mapping[str, Any],
    receipt: Mapping[str, Any],
    marker_raw: bytes,
    *,
    evidence_path: Path,
    evidence_sha256: str,
    receipt_path: Path,
    receipt_sha256: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        value.get("status") != "REVIEW_EVIDENCE_PASS"
        or value.get("audit")
        != "offline_arb_textatlas_catalog_evidence_rebind_w64_v1"
        or value.get("read_only") is not True
        or value.get("activation_status") != "EVIDENCE_ONLY_NOT_RUNTIME_ACTIVE"
        or value.get("runtime_policy_activated") is not False
    ):
        raise GateError("W64 TextAtlas rebind status/schema drift")
    _identity(_mapping(value, "stage_identity", "rebind"), "rebind.stage_identity")
    plan_contract = _mapping(value, "plan_contract", "rebind")
    expected_plan_contract = {
        "plan_id": PLAN_ID,
        "plan_hash": PLAN_HASH,
        "plan_horizon": PLAN_HORIZON,
        "formal_steps": FORMAL_STEPS,
        "world_size": WORLD_SIZE,
        "per_rank_bs": PER_RANK_BS,
        "global_batch": GLOBAL_BATCH,
        "formal_exposures": FORMAL_EXPOSURES,
    }
    for key, expected in expected_plan_contract.items():
        if plan_contract.get(key) != expected:
            raise GateError(f"W64 TextAtlas rebind plan_contract.{key} drift")
    if plan_contract.get("coordinate_schema") != {
        "formal_field": "within_formal",
        "plan_index_formula": "step*2048+rank*32+slot",
        "zero_based": True,
    }:
        raise GateError("W64 TextAtlas rebind coordinate schema drift")
    source = _mapping(value, "source_evidence", "rebind")
    w64_plan = _mapping(source, "w64_plan", "rebind.source_evidence")
    if {
        "header_sha256": w64_plan.get("header_sha256"),
        "plan_hash": w64_plan.get("plan_hash"),
        "checksums_sha256": w64_plan.get("checksums_sha256"),
    } != {
        "header_sha256": PLAN_HEADER_SHA256,
        "plan_hash": PLAN_HASH,
        "checksums_sha256": contract["hashes"]["plan_checksums_sha256"],
    }:
        raise GateError("W64 TextAtlas rebind plan provenance drift")
    plan_audit = _mapping(source, "w64_plan_audit", "rebind.source_evidence")
    if {
        "plan_rows_sha256": plan_audit.get("plan_rows_sha256"),
        "bitmap_sha256": plan_audit.get("bitmap_sha256"),
    } != {
        "plan_rows_sha256": contract["hashes"]["plan_rows_sha256"],
        "bitmap_sha256": contract["hashes"]["plan_bitmap_sha256"],
    }:
        raise GateError("W64 TextAtlas rebind plan-audit provenance drift")

    rows = _list(value, "rows", "rebind")
    if len(rows) != 43 or any(not isinstance(row, Mapping) for row in rows):
        raise GateError("W64 TextAtlas rebind must contain exactly 43 rows")
    seen: set[int] = set()
    formal_sum = 0
    total_sum = 0
    success_formal = 0
    success_total = 0
    repaired_formal: dict[int, int] = {}
    repaired_total: dict[int, int] = {}
    reviewed_repairs: list[dict[str, Any]] = []
    success_rows = 0
    repair_rows = 0
    for index, row in enumerate(rows):
        row_id = _integer(row.get("row_id"), f"rebind.rows[{index}].row_id")
        if (
            row_id in seen
            or row.get("domain") != "textatlas"
            or row.get("runtime_policy_activated") is not False
            or row.get("activation_status") != "EVIDENCE_ONLY_NOT_RUNTIME_ACTIVE"
        ):
            raise GateError("W64 TextAtlas rebind row identity/uniqueness drift")
        seen.add(row_id)
        strict_raw = _mapping(row, "strict_raw", f"rebind row {row_id}")
        if strict_raw.get("ok") is not False:
            raise GateError(f"W64 TextAtlas row {row_id} is no longer a strict failure")
        eligible = row.get("eligible_for_explicit_append_eoi_allowlist")
        if not isinstance(eligible, bool) or eligible is not (
            row_id in TEXTATLAS_REPAIR_ROWS
        ):
            raise GateError(f"W64 TextAtlas row {row_id} allowlist partition drift")
        bucket_idx = _integer(row.get("bucket_idx"), f"rebind row {row_id}.bucket")
        try:
            coordinates, formal = validate_plan_coordinates(
                row.get("coordinates"),
                bucket_idx=bucket_idx,
                label=f"W64 TextAtlas row {row_id}",
                plan_horizon=PLAN_HORIZON,
                formal_steps=FORMAL_STEPS,
                world_size=WORLD_SIZE,
                per_rank_bs=PER_RANK_BS,
                global_batch=GLOBAL_BATCH,
                within_formal_field="within_formal",
            )
        except EvidencePublishError as exc:
            raise GateError(str(exc)) from exc
        for coordinate in coordinates:
            expected_plan_index = (
                int(coordinate["step"]) * GLOBAL_BATCH
                + int(coordinate["rank"]) * PER_RANK_BS
                + int(coordinate["slot"])
            )
            if coordinate.get("plan_index") != expected_plan_index:
                raise GateError(
                    f"W64 TextAtlas row {row_id} coordinate plan_index drift"
                )
        if row.get("formal_coordinates") != formal:
            raise GateError(f"W64 TextAtlas row {row_id} formal coordinate drift")
        if not formal or row.get("first_formal_coordinate") != formal[0]:
            raise GateError(f"W64 TextAtlas row {row_id} first formal coordinate drift")
        if row.get("total_plan_exposures") != len(coordinates):
            raise GateError(f"W64 TextAtlas row {row_id} total exposure drift")
        if row.get("formal_plan_exposures") != len(formal):
            raise GateError(f"W64 TextAtlas row {row_id} formal exposure drift")
        formal_sum += len(formal)
        total_sum += len(coordinates)
        review = _mapping(row, "legacy_production_decoder_review", f"rebind row {row_id}")
        classification = review.get("classification")
        if row_id in TEXTATLAS_REPAIR_ROWS:
            _validate_repair_projection(row, row_id)
            expected_repair = TEXTATLAS_REPAIRS[row_id]
            locator = _mapping(row, "locator", f"rebind row {row_id}")
            reviewed_repairs.append(
                {
                    "row_id": row_id,
                    "uid": expected_repair["uid"],
                    "bucket_idx": expected_repair["bucket_idx"],
                    "width": expected_repair["width"],
                    "height": expected_repair["height"],
                    "image_path": locator["image_path"],
                    "image_parquet_path": locator["image_parquet_path"],
                    "image_row_group": locator["image_row_group"],
                    "image_row_in_group": locator["image_row_in_group"],
                    "raw_bytes": expected_repair["raw_bytes"],
                    "raw_sha256": expected_repair["raw_sha256"],
                    "appended_bytes_hex": "ffd9",
                    "repaired_bytes": expected_repair["repaired_bytes"],
                    "repaired_payload_sha256": expected_repair["repaired_sha256"],
                    "repaired_rgb_sha256": expected_repair["pixel_sha256"],
                    "formal_plan_exposures": len(formal),
                    "total_plan_exposures": len(coordinates),
                }
            )
            repaired_formal[row_id] = len(formal)
            repaired_total[row_id] = len(coordinates)
            repair_rows += 1
        elif classification == "production_decode_success_no_exception":
            if (
                review.get("pixel_load_succeeded") is not True
                or review.get("runtime_policy_activated") is not False
                or review.get("repair_candidate") is not None
            ):
                raise GateError(f"W64 production-success row {row_id} review drift")
            success_formal += len(formal)
            success_total += len(coordinates)
            success_rows += 1
        else:
            raise GateError(f"W64 TextAtlas row {row_id} has unreviewed classification")
    if repair_rows != 3 or success_rows != 40 or set(repaired_formal) != set(TEXTATLAS_REPAIR_ROWS):
        raise GateError("W64 TextAtlas reviewed 40/3 partition drift")
    expected_aggregate = contract["textatlas_aggregate_exposures"]
    if {
        "candidate_formal": formal_sum,
        "candidate_total": total_sum,
        "production_success_formal": success_formal,
        "production_success_total": success_total,
    } != dict(expected_aggregate):
        raise GateError("W64 TextAtlas aggregate exposure counts drift")
    if repaired_formal != contract["textatlas_formal_exposures"]:
        raise GateError("W64 TextAtlas repair FORMAL exposure map drift")
    if repaired_total != contract["textatlas_total_exposures"]:
        raise GateError("W64 TextAtlas repair horizon exposure map drift")
    summary = _mapping(value, "summary", "rebind")
    expected_summary_by_semantic = {
        "candidate_rows": 43,
        "planned_candidate_rows": 43,
        "formal_candidate_rows": 43,
        "candidate_horizon_exposures": total_sum,
        "candidate_formal_exposures": formal_sum,
        "production_success_rows": 40,
        "production_failure_rows": 3,
        "pixel_equivalent_repair_rows": 3,
        "textatlas_audited_candidate_rows": 43,
        "textatlas_formal_candidate_rows": 43,
        "textatlas_candidate_formal_exposures": formal_sum,
        "textatlas_candidate_horizon_exposures": total_sum,
        "textatlas_production_success_rows": 40,
        "textatlas_production_success_formal_exposures": success_formal,
        "textatlas_production_success_horizon_exposures": success_total,
        "textatlas_repair_policy_rows": 3,
        "textatlas_repaired_formal_rows": 3,
        "textatlas_repaired_formal_exposures": sum(repaired_formal.values()),
        "textatlas_repaired_horizon_exposures": sum(repaired_total.values()),
        "runtime_policy_activated": False,
    }
    expected_summary = {
        REBIND_SUMMARY_FIELDS[semantic]: expected
        for semantic, expected in expected_summary_by_semantic.items()
    }
    if set(REBIND_SUMMARY_FIELDS) != set(expected_summary_by_semantic):
        raise GateError("internal W64 rebind summary semantic contract drift")
    if dict(summary) != expected_summary:
        raise GateError("W64 TextAtlas rebind summary drift")
    _reject_legacy_coordinates(value)

    if (
        receipt.get("status") != "REVIEW_EVIDENCE_PASS"
        or receipt.get("schema_version")
        != "stage512a_textatlas_w64_catalog_rebind_receipt_v1"
        or receipt.get("read_only") is not True
        or receipt.get("activation_status") != "EVIDENCE_ONLY_NOT_RUNTIME_ACTIVE"
        or receipt.get("runtime_policy_activated") is not False
        or receipt.get("stage_identity") != IDENTITY
        or receipt.get("inputs") != source
        or receipt.get("summary") != summary
    ):
        raise GateError("W64 TextAtlas rebind receipt drift")
    expected_artifact = [{
        "name": evidence_path.name,
        "path": str(evidence_path),
        "sha256": evidence_sha256,
        "status": "REVIEW_EVIDENCE_PASS",
    }]
    if receipt.get("artifacts") != expected_artifact:
        raise GateError("W64 TextAtlas rebind receipt artifact binding drift")
    if marker_raw != (
        f"PASS|EVIDENCE_ONLY_NOT_RUNTIME_ACTIVE|{receipt_sha256}\n".encode("ascii")
    ):
        raise GateError("W64 TextAtlas rebind marker/receipt drift")
    if evidence_path.parent != receipt_path.parent:
        raise GateError("W64 TextAtlas rebind artifact/receipt directory drift")
    return {
        "candidate_formal_exposures": formal_sum,
        "candidate_total_exposures": total_sum,
        "production_success_formal_exposures": success_formal,
        "production_success_total_exposures": success_total,
        "repair_formal_exposures": repaired_formal,
        "repair_total_exposures": repaired_total,
        "reviewed_repairs": sorted(
            reviewed_repairs, key=lambda item: int(item["row_id"])
        ),
    }


def _load_source_module(raw: bytes, path: str, suffix: str) -> types.ModuleType:
    name = f"_stage512a_w64_gate_{suffix}_{hashlib.sha256(raw).hexdigest()}"
    module = types.ModuleType(name)
    module.__file__ = path
    sys.modules[name] = module
    try:
        exec(compile(raw, path, "exec", dont_inherit=True, optimize=0), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


def _validate_runtime_profiles(caption: types.ModuleType, image: types.ModuleType) -> None:
    if (
        caption.STAGE512A_W64_PLAN_ID != PLAN_ID
        or caption.STAGE512A_W64_PLAN_HEADER_SHA256 != PLAN_HEADER_SHA256
        or caption.STAGE512A_W64_CATALOG_HASH != CATALOG_HASH
        or caption.STAGE512A_W64_CATALOG_HEADER_SHA256 != CATALOG_HEADER_SHA256
        or dict(caption.STAGE512A_W64_CAPTIONLESS_MONET_ROWS) != MONET_UIDS
    ):
        raise GateError("W64 caption runtime constants drift")
    if (
        image.STAGE512A_W64_PLAN_ID != PLAN_ID
        or image.STAGE512A_W64_PLAN_HASH != PLAN_HASH
        or image.STAGE512A_W64_PLAN_HEADER_SHA256 != PLAN_HEADER_SHA256
        or image.STAGE512A_W64_CATALOG_HASH != CATALOG_HASH
        or image.STAGE512A_W64_CATALOG_HEADER_SHA256 != CATALOG_HEADER_SHA256
        or set(image.STAGE512A_W64_TEXTATLAS_JPEG_EOI_REPAIRS)
        != set(TEXTATLAS_REPAIR_ROWS)
    ):
        raise GateError("W64 image runtime constants drift")
    for row_id, expected in TEXTATLAS_REPAIRS.items():
        repair = image.STAGE512A_W64_TEXTATLAS_JPEG_EOI_REPAIRS[row_id]
        observed = {
            "uid": repair.uid,
            "bucket_idx": repair.bucket_idx,
            "width": repair.width,
            "height": repair.height,
            "image_path": repair.image_path,
            "image_parquet_path": repair.image_parquet_path,
            "image_row_group": repair.image_row_group,
            "image_row_in_group": repair.image_row_in_group,
            "raw_bytes": repair.raw_bytes,
            "raw_sha256": repair.raw_sha256,
            "repaired_bytes": repair.repaired_bytes,
            "repaired_sha256": repair.repaired_sha256,
            "pixel_sha256": repair.pixel_sha256,
        }
        if observed != expected:
            raise GateError(f"W64 image runtime repair identity drift: {row_id}")

    def domain(row_id: int) -> str:
        if row_id in MONET_UIDS:
            return "monet_real"
        if row_id in TEXTATLAS_REPAIRS:
            return "textatlas"
        raise GateError(f"unexpected runtime-reviewed row {row_id}")

    caption_bound = caption.bind_caption_exceptions(
        stage="512a",
        eligibility="elig_512",
        plan_id=PLAN_ID,
        catalog_hash=CATALOG_HASH,
        plan_header_sha256=PLAN_HEADER_SHA256,
        catalog_header_sha256=CATALOG_HEADER_SHA256,
        decode_domain=domain,
    )
    image_bound = image.bind_image_exceptions(
        stage="512a",
        eligibility="elig_512",
        plan_id=PLAN_ID,
        plan_hash=PLAN_HASH,
        catalog_hash=CATALOG_HASH,
        plan_header_sha256=PLAN_HEADER_SHA256,
        catalog_header_sha256=CATALOG_HEADER_SHA256,
        decode_domain=domain,
    )
    if dict(caption_bound) != MONET_UIDS or set(image_bound) != set(TEXTATLAS_REPAIRS):
        raise GateError("exact W64 runtime profile did not activate")

    # W16 GPUVAL shares the catalog but must not activate either production
    # profile.  Catalog identity alone is never an activation signal.
    if caption.bind_caption_exceptions(
        stage="512a",
        eligibility="elig_512",
        plan_id="stage512a_offline_exact_v2_w16_gpuval",
        catalog_hash=CATALOG_HASH,
        plan_header_sha256="1" * 64,
        catalog_header_sha256=CATALOG_HEADER_SHA256,
        decode_domain=domain,
    ):
        raise GateError("W16 GPUVAL activated caption production exceptions")
    if image.bind_image_exceptions(
        stage="512a",
        eligibility="elig_512",
        plan_id="stage512a_offline_exact_v2_w16_gpuval",
        plan_hash="2" * 64,
        catalog_hash=CATALOG_HASH,
        plan_header_sha256="3" * 64,
        catalog_header_sha256=CATALOG_HEADER_SHA256,
        decode_domain=domain,
    ):
        raise GateError("W16 GPUVAL activated image production exceptions")

    try:
        caption.bind_caption_exceptions(
            stage="512a",
            eligibility="elig_512",
            plan_id=caption.STAGE512A_PLAN_ID,
            catalog_hash=CATALOG_HASH,
            plan_header_sha256=PLAN_HEADER_SHA256,
            catalog_header_sha256=CATALOG_HEADER_SHA256,
            decode_domain=domain,
        )
    except caption.CaptionExceptionBindingError as exc:
        if "conflicting profiles" not in str(exc):
            raise GateError("caption mixed-profile failure reason drift") from exc
    else:
        raise GateError("W48/W64 caption signal mix did not fail closed")
    try:
        image.bind_image_exceptions(
            stage="512a",
            eligibility="elig_512",
            plan_id=image.STAGE512A_PLAN_ID,
            plan_hash=PLAN_HASH,
            catalog_hash=CATALOG_HASH,
            plan_header_sha256=image.STAGE512A_PLAN_HEADER_SHA256,
            catalog_header_sha256=CATALOG_HEADER_SHA256,
            decode_domain=domain,
        )
    except image.ImageExceptionBindingError as exc:
        if "conflicting profiles" not in str(exc):
            raise GateError("image mixed-profile failure reason drift") from exc
    else:
        raise GateError("W48/W64 image signal mix did not fail closed")


def _validate_runtime_sources(
    args: argparse.Namespace,
) -> dict[str, dict[str, str]]:
    inputs = {
        "offline_dataset": (
            args.offline_dataset_runtime,
            args.expected_offline_dataset_runtime_sha256,
        ),
        "caption_exceptions": (
            args.caption_runtime,
            args.expected_caption_runtime_sha256,
        ),
        "image_exceptions": (
            args.image_runtime,
            args.expected_image_runtime_sha256,
        ),
        "image_fetch": (
            args.image_fetch_runtime,
            args.expected_image_fetch_runtime_sha256,
        ),
        "targeted_test": (
            args.targeted_test_runtime,
            args.expected_targeted_test_runtime_sha256,
        ),
    }
    artifacts: dict[str, dict[str, str]] = {}
    raws: dict[str, bytes] = {}
    for label, (path, expected) in inputs.items():
        expected_sha = _sha(expected, f"expected {label} SHA")
        artifacts[label], raws[label] = _file_snapshot(path, label, expected_sha)
    caption = _load_source_module(
        raws["caption_exceptions"], artifacts["caption_exceptions"]["path"], "caption"
    )
    image = _load_source_module(
        raws["image_exceptions"], artifacts["image_exceptions"]["path"], "image"
    )
    _validate_runtime_profiles(caption, image)
    if b"bind_caption_exceptions(" not in raws["offline_dataset"] or b"bind_image_exceptions(" not in raws["offline_dataset"]:
        raise GateError("offline dataset no longer binds reviewed exception profiles")
    return artifacts


def _load_bound_contract(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, str]]:
    expected_sha = _sha(
        args.expected_w64_exception_contract_sha256,
        "expected W64 exception contract SHA",
    )
    artifact, raw = _file_snapshot(
        args.w64_exception_contract,
        "W64 exception contract",
        expected_sha,
    )
    module = _load_source_module(raw, artifact["path"], "contract")
    known = {
        "PLAN_ID": PLAN_ID,
        "PLAN_HASH": PLAN_HASH,
        "PLAN_HEADER_SHA256": PLAN_HEADER_SHA256,
        "CATALOG_HASH": CATALOG_HASH,
        "CATALOG_HEADER_SHA256": CATALOG_HEADER_SHA256,
        "ASSIGNMENT_HASH": ASSIGNMENT_HASH,
        "SIDECAR_HEADER_SHA256": SIDECAR_HEADER_SHA256,
        "PLAN_HORIZON": PLAN_HORIZON,
        "FORMAL_STEPS": FORMAL_STEPS,
        "WORLD_SIZE": WORLD_SIZE,
        "PER_RANK_BS": PER_RANK_BS,
        "GLOBAL_BATCH": GLOBAL_BATCH,
    }
    for name, expected in known.items():
        if getattr(module, name, None) != expected:
            raise GateError(f"bound W64 exception contract {name} drift")
    try:
        value = module.finalized_contract()
    except Exception as exc:  # fail closed across a separately loaded module
        raise GateError(f"bound W64 exception contract is not finalized: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError("bound W64 exception contract returned a non-object")
    return value, artifact


def validate_gate_documents(
    plan: Mapping[str, Any],
    catalog: Mapping[str, Any],
    caption_formal: Mapping[str, Any],
    caption_horizon: Mapping[str, Any],
    plan_rows: Mapping[str, Any],
    rebind: Mapping[str, Any],
    rebind_receipt: Mapping[str, Any],
    rebind_marker_raw: bytes,
    *,
    plan_header_sha256: str,
    catalog_header_sha256: str,
    plan_bitmap_sha256: str,
    rebind_path: Path,
    rebind_sha256: str,
    rebind_receipt_path: Path,
    rebind_receipt_sha256: str,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate small documents; retained as a unit-test API."""

    contract = finalized_contract() if contract is None else contract
    if plan_header_sha256 != PLAN_HEADER_SHA256:
        raise GateError("W64 plan header SHA drift")
    if catalog_header_sha256 != CATALOG_HEADER_SHA256:
        raise GateError("stage512a catalog header SHA drift")
    if plan_bitmap_sha256 != contract["hashes"]["plan_bitmap_sha256"]:
        raise GateError("W64 plan bitmap SHA drift")
    plan_catalog = _validate_plan_catalog(plan, catalog)
    caption_formal_counts = _validate_caption_audit(
        caption_formal,
        steps=FORMAL_STEPS,
        expected_exposures=contract["monet_formal_exposures"],
        label="caption_formal",
    )
    caption_total_counts = _validate_caption_audit(
        caption_horizon,
        steps=PLAN_HORIZON,
        expected_exposures=contract["monet_total_exposures"],
        label="caption_horizon",
    )
    textatlas_coverage = _validate_plan_rows(plan_rows, plan_bitmap_sha256)
    textatlas_counts = _validate_rebind(
        rebind,
        rebind_receipt,
        rebind_marker_raw,
        evidence_path=rebind_path,
        evidence_sha256=rebind_sha256,
        receipt_path=rebind_receipt_path,
        receipt_sha256=rebind_receipt_sha256,
        contract=contract,
    )
    return {
        "topology": plan_catalog["topology"],
        "domain_exposures": plan_catalog["domain_exposures"],
        "caption_formal_counts": caption_formal_counts,
        "caption_total_counts": caption_total_counts,
        "textatlas_coverage": textatlas_coverage,
        "textatlas_counts": textatlas_counts,
    }


def _publish_exclusive_readonly(path: Path, raw: bytes) -> None:
    output = absolute_lexical(path)
    if output != EXPECTED_OUTPUT:
        raise GateError(
            f"gate output must be exact canonical W64 path {EXPECTED_OUTPUT}, got {output}"
        )
    parent = output.parent
    base = require_real_directory(parent.parent, "W64 gate output namespace base")
    if parent.parent != base or Path(parent.name).name != parent.name:
        raise GateError("W64 gate namespace must be a direct normalized child")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    base_before = os.lstat(base)
    base_fd = os.open(base, flags)
    parent_fd: int | None = None
    descriptor: int | None = None
    created_parent = False
    try:
        base_identity = (base_before.st_dev, base_before.st_ino, base_before.st_mode)
        current_base = os.fstat(base_fd)
        if base_identity != (current_base.st_dev, current_base.st_ino, current_base.st_mode):
            raise GateError("W64 gate base changed before publication")
        try:
            os.stat(parent.name, dir_fd=base_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise GateError("canonical W64 reviewed-gate namespace already exists")
        os.mkdir(parent.name, 0o700, dir_fd=base_fd)
        created_parent = True
        parent_fd = os.open(parent.name, flags, dir_fd=base_fd)
        if os.listdir(parent_fd):
            raise GateError("new W64 reviewed-gate namespace is not empty")
        output_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            output_flags |= os.O_NOFOLLOW
        descriptor = os.open(output.name, output_flags, 0o400, dir_fd=parent_fd)
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise GateError("short write publishing W64 exception gate")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        named = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino, metadata.st_size)
            != (named.st_dev, named.st_ino, named.st_size)
        ):
            raise GateError("published W64 gate type/mode/inode drift")
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed = b""
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += chunk
        if observed != raw:
            raise GateError("published W64 gate readback drift")
        if set(os.listdir(parent_fd)) != {"FORMAL_PASS.json"}:
            raise GateError("published W64 gate namespace member drift")
        os.fchmod(parent_fd, 0o555)
        os.fsync(parent_fd)
        os.fsync(base_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created_parent and descriptor is None:
            try:
                if parent_fd is not None and not os.listdir(parent_fd):
                    os.rmdir(parent.name, dir_fd=base_fd)
            except OSError:
                pass
        if parent_fd is not None:
            os.close(parent_fd)
        os.close(base_fd)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-header", required=True, type=Path)
    parser.add_argument("--plan-checksums", required=True, type=Path)
    parser.add_argument("--catalog-header", required=True, type=Path)
    parser.add_argument("--sidecar-header", required=True, type=Path)
    parser.add_argument("--caption-audit", required=True, type=Path)
    parser.add_argument("--caption-horizon-audit", required=True, type=Path)
    parser.add_argument("--plan-rows", required=True, type=Path)
    parser.add_argument("--plan-bitmap", required=True, type=Path)
    parser.add_argument("--textatlas-rebind", required=True, type=Path)
    parser.add_argument("--textatlas-rebind-receipt", required=True, type=Path)
    parser.add_argument("--textatlas-rebind-success", required=True, type=Path)
    parser.add_argument("--w64-exception-contract", required=True, type=Path)
    parser.add_argument("--expected-w64-exception-contract-sha256", required=True)
    parser.add_argument("--offline-dataset-runtime", required=True, type=Path)
    parser.add_argument("--expected-offline-dataset-runtime-sha256", required=True)
    parser.add_argument("--caption-runtime", required=True, type=Path)
    parser.add_argument("--expected-caption-runtime-sha256", required=True)
    parser.add_argument("--image-runtime", required=True, type=Path)
    parser.add_argument("--expected-image-runtime-sha256", required=True)
    parser.add_argument("--image-fetch-runtime", required=True, type=Path)
    parser.add_argument("--expected-image-fetch-runtime-sha256", required=True)
    parser.add_argument("--targeted-test-runtime", required=True, type=Path)
    parser.add_argument("--expected-targeted-test-runtime-sha256", required=True)
    parser.add_argument("--producer-template", required=True, type=Path)
    parser.add_argument("--producer-finalizer", required=True, type=Path)
    parser.add_argument("--producer-evidence-helper", required=True, type=Path)
    parser.add_argument("--producer-renderer", required=True, type=Path)
    parser.add_argument("--producer-rendered-yaml", required=True, type=Path)
    parser.add_argument("--producer-launch-token", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    if sys.flags.optimize != 0 or os.environ.get("PYTHONOPTIMIZE") is not None:
        raise GateError("optimized Python is forbidden for the W64 exception gate")
    args = _args()
    contract, contract_artifact = _load_bound_contract(args)

    plan, plan_artifact, _ = _json_snapshot(
        args.plan_header, "W64 plan header", PLAN_HEADER_SHA256
    )
    checksums, checksums_artifact, _ = _json_snapshot(
        args.plan_checksums,
        "W64 plan checksums",
        contract["hashes"]["plan_checksums_sha256"],
    )
    if checksums.get("header") != PLAN_HEADER_SHA256:
        raise GateError("W64 checksums/header closure drift")
    blocks = checksums.get("blocks")
    if not isinstance(blocks, Mapping) or set(blocks) != {
        f"block_{index:06d}.npz" for index in range(NUM_BLOCKS)
    } or any(not isinstance(value, str) or _sha(value, "W64 block SHA") != value for value in blocks.values()):
        raise GateError("W64 checksums block set/SHA drift")
    catalog, catalog_artifact, _ = _json_snapshot(
        args.catalog_header, "stage512a catalog header", CATALOG_HEADER_SHA256
    )
    _sidecar, sidecar_artifact, _ = _json_snapshot(
        args.sidecar_header, "stage512a caption sidecar header", SIDECAR_HEADER_SHA256
    )
    caption, caption_artifact, _ = _json_snapshot(
        args.caption_audit,
        "W64 caption FORMAL audit",
        contract["hashes"]["caption_audit_sha256"],
    )
    caption_horizon, caption_horizon_artifact, _ = _json_snapshot(
        args.caption_horizon_audit,
        "W64 caption horizon audit",
        contract["hashes"]["caption_horizon_audit_sha256"],
    )
    plan_rows, plan_rows_artifact, _ = _json_snapshot(
        args.plan_rows,
        "W64 plan-row audit",
        contract["hashes"]["plan_rows_sha256"],
    )
    plan_bitmap_artifact, _ = _file_snapshot(
        args.plan_bitmap,
        "W64 plan-row bitmap",
        contract["hashes"]["plan_bitmap_sha256"],
    )
    rebind, rebind_artifact, _ = _json_snapshot(
        args.textatlas_rebind,
        "W64 TextAtlas rebind",
        contract["hashes"]["textatlas_rebind_sha256"],
    )
    rebind_receipt, rebind_receipt_artifact, _ = _json_snapshot(
        args.textatlas_rebind_receipt,
        "W64 TextAtlas rebind receipt",
        contract["hashes"]["textatlas_rebind_receipt_sha256"],
    )
    rebind_success_artifact, rebind_marker_raw = _file_snapshot(
        args.textatlas_rebind_success, "W64 TextAtlas rebind success"
    )
    result = validate_gate_documents(
        plan,
        catalog,
        caption,
        caption_horizon,
        plan_rows,
        rebind,
        rebind_receipt,
        rebind_marker_raw,
        plan_header_sha256=plan_artifact["sha256"],
        catalog_header_sha256=catalog_artifact["sha256"],
        plan_bitmap_sha256=plan_bitmap_artifact["sha256"],
        rebind_path=Path(rebind_artifact["path"]),
        rebind_sha256=rebind_artifact["sha256"],
        rebind_receipt_path=Path(rebind_receipt_artifact["path"]),
        rebind_receipt_sha256=rebind_receipt_artifact["sha256"],
        contract=contract,
    )
    runtime = _validate_runtime_sources(args)
    producer_bindings = {
        "plan_header": args.plan_header,
        "plan_checksums": args.plan_checksums,
        "catalog_header": args.catalog_header,
        "sidecar_header": args.sidecar_header,
        "caption_audit": args.caption_audit,
        "caption_horizon_audit": args.caption_horizon_audit,
        "plan_rows": args.plan_rows,
        "plan_bitmap": args.plan_bitmap,
        "textatlas_rebind": args.textatlas_rebind,
        "textatlas_rebind_receipt": args.textatlas_rebind_receipt,
        "textatlas_rebind_success": args.textatlas_rebind_success,
        "w64_exception_contract": args.w64_exception_contract,
        "offline_dataset_runtime": args.offline_dataset_runtime,
        "caption_runtime": args.caption_runtime,
        "image_runtime": args.image_runtime,
        "image_fetch_runtime": args.image_fetch_runtime,
        "targeted_test_runtime": args.targeted_test_runtime,
        "gate_finalizer": args.producer_finalizer,
        "gate_evidence_helper": args.producer_evidence_helper,
        "write_renderer": args.producer_renderer,
    }
    if absolute_lexical(args.producer_template) != EXPECTED_TEMPLATE:
        raise GateError("W64 reviewed-gate producer template path drift")
    if set(producer_bindings) != set(EXPECTED_PRODUCER_BINDINGS):
        raise GateError("W64 reviewed-gate producer binding label contract drift")
    for label, observed in producer_bindings.items():
        expected = EXPECTED_PRODUCER_BINDINGS[label]
        if absolute_lexical(observed) != expected:
            raise GateError(
                f"W64 reviewed-gate producer binding path drift for {label}: "
                f"observed={observed},expected={expected}"
            )
    try:
        producer = load_stage512a_write_producer_provenance(
            args.producer_rendered_yaml,
            expected_launch_token=args.producer_launch_token,
            expected_template=EXPECTED_TEMPLATE,
            expected_finalizer=EXPECTED_FINALIZER,
            expected_evidence_helper=EXPECTED_HELPER,
            expected_renderer=EXPECTED_RENDERER,
            expected_container_image=EXPECTED_IMAGE,
            expected_bindings=EXPECTED_PRODUCER_BINDINGS,
            expected_binding_labels=STAGE512A_W64_REVIEWED_GATE_BINDING_LABELS,
        )
    except EvidencePublishError as exc:
        raise GateError(str(exc)) from exc
    monet_rows = [
        {
            "domain": "monet_real",
            "row_id": row_id,
            "uid": MONET_UIDS[row_id],
            "formal_plan_exposures": result["caption_formal_counts"][row_id],
            "total_plan_exposures": result["caption_total_counts"][row_id],
        }
        for row_id in MONET_ROWS
    ]
    # The rebind artifact intentionally describes its three candidates as
    # ``repair_review`` rows.  Only this final gate, after runtime-source and
    # producer provenance close, activates them and projects ``repaired``
    # counters for the existing FORMAL/GPUVAL consumer ABI.
    counts = {
        "monet_empty_prompt_rows": len(MONET_ROWS),
        "monet_empty_prompt_formal_exposures": sum(
            result["caption_formal_counts"].values()
        ),
        "monet_empty_prompt_horizon_exposures": sum(
            result["caption_total_counts"].values()
        ),
        "textatlas_raw_candidate_rows": 43,
        "textatlas_raw_candidate_formal_exposures": result["textatlas_counts"][
            "candidate_formal_exposures"
        ],
        "textatlas_raw_candidate_horizon_exposures": result["textatlas_counts"][
            "candidate_total_exposures"
        ],
        "textatlas_production_decode_success_rows": 40,
        "textatlas_production_decode_success_formal_exposures": result[
            "textatlas_counts"
        ]["production_success_formal_exposures"],
        "textatlas_production_decode_success_horizon_exposures": result[
            "textatlas_counts"
        ]["production_success_total_exposures"],
        "textatlas_repaired_rows": 3,
        "textatlas_repaired_formal_exposures": sum(
            result["textatlas_counts"]["repair_formal_exposures"].values()
        ),
        "textatlas_repaired_horizon_exposures": sum(
            result["textatlas_counts"]["repair_total_exposures"].values()
        ),
    }
    if counts != contract["gate_counts"]:
        raise GateError("W64 reviewed-exception exact gate count closure drift")
    output = {
        "status": "PASS",
        "schema": SCHEMA,
        "policy": POLICY,
        "stage_identity": dict(IDENTITY),
        "plan_id": PLAN_ID,
        "plan_hash": PLAN_HASH,
        "catalog_hash": CATALOG_HASH,
        "assignment_hash": ASSIGNMENT_HASH,
        "plan_contract": result["topology"],
        "plan_domain_exposures": result["domain_exposures"],
        "plan_header_sha256": plan_artifact["sha256"],
        "catalog_header_sha256": catalog_artifact["sha256"],
        "producer": producer,
        "evidence": {
            "plan_header": plan_artifact,
            "plan_checksums": checksums_artifact,
            "catalog_header": catalog_artifact,
            "sidecar_header": sidecar_artifact,
            "caption_audit": caption_artifact,
            "caption_horizon_audit": caption_horizon_artifact,
            "plan_rows": plan_rows_artifact,
            "plan_bitmap": plan_bitmap_artifact,
            "textatlas_rebind": rebind_artifact,
            "textatlas_rebind_receipt": rebind_receipt_artifact,
            "textatlas_rebind_success": rebind_success_artifact,
            "w64_exception_contract": contract_artifact,
        },
        "runtime": runtime,
        "reviewed_exceptions": {
            "monet_empty_prompt": {
                "policy": MONET_POLICY,
                "unique_rows": len(monet_rows),
                "formal_exposures": counts[
                    "monet_empty_prompt_formal_exposures"
                ],
                "rows": monet_rows,
            },
            "textatlas_jpeg_eoi": {
                "policy": TEXTATLAS_POLICY,
                "raw_candidate_rows": counts["textatlas_raw_candidate_rows"],
                "raw_candidate_formal_exposures": counts[
                    "textatlas_raw_candidate_formal_exposures"
                ],
                "production_decode_success_rows": counts[
                    "textatlas_production_decode_success_rows"
                ],
                "production_decode_success_formal_exposures": counts[
                    "textatlas_production_decode_success_formal_exposures"
                ],
                "repaired_rows": counts["textatlas_repaired_rows"],
                "repaired_formal_exposures": counts[
                    "textatlas_repaired_formal_exposures"
                ],
                "repaired_horizon_exposures": counts[
                    "textatlas_repaired_horizon_exposures"
                ],
                "rows": result["textatlas_counts"]["reviewed_repairs"],
            },
        },
        "counts": counts,
    }
    raw = (
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish_exclusive_readonly(args.output, raw)
    print(
        json.dumps(
            {
                "status": "PASS",
                "schema": SCHEMA,
                "output": str(args.output),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "counts": output["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
