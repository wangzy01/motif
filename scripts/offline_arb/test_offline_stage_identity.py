#!/usr/bin/env python3
"""Lightweight regression tests for OFFLINE_ARB per-stage artifact identity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dataline_offline.offline_dataset import (  # noqa: E402
    OfflineLoaderError,
    _OFFLINE_STAGE_IDENTITIES,
    _assignment_identity_payload,
    _canonical_sha256,
    _resolve_offline_stage_identity,
    _validate_artifact_stage_identity,
    _validate_plan_stage_identity,
)


LEGACY_STAGE256_CATALOG_HASH = (
    "b5f83014e4eab304460336adff217c35e993a2d892a5d889f69f87013c65f212"
)
LEGACY_STAGE256_ASSIGNMENT_HASH = (
    "10bd55e0a37aacef35583c91846984171c6ebdfa88ee8351087bb937cc498cc4"
)
LEGACY_STAGE256_PLAN_ID = "stage256_offline_exact_v1_w64_prod"


def _expect_error(fragment: str, fn: Callable[[], Any]) -> None:
    try:
        fn()
    except OfflineLoaderError as exc:
        if fragment not in str(exc):
            raise AssertionError(
                f"expected error containing {fragment!r}, got {str(exc)!r}"
            ) from exc
    else:
        raise AssertionError(f"expected OfflineLoaderError containing {fragment!r}")


def _assignment_source() -> dict[str, Any]:
    return {
        "bucket_table_hash": "a" * 64,
        "floor": 0.005,
        "kept_bucket_indices": [0, 2],
        "fold_map": [0, 0, 2],
        "square_only_domains": ["synthetic", "textatlas"],
        "square_bucket_idx": 0,
    }


def _test_runtime_contracts() -> None:
    legacy = _resolve_offline_stage_identity(
        "256", None, {"assignment_mode": "offline_exact_v1"}
    )
    if (
        legacy.eligibility != "elig_256"
        or legacy.assignment_eligibility != "phase3a_stage256_v1"
    ):
        raise AssertionError("stage256 public/legacy eligibility identities leaked")
    explicit_256 = _resolve_offline_stage_identity(
        "256", 256, {"target_area": 256 * 256}
    )
    if explicit_256 != legacy:
        raise AssertionError("legacy and explicit stage256 contracts differ")
    stage512a = _resolve_offline_stage_identity(
        "512a", 512, {"target_area": 512 * 512}
    )
    if (
        stage512a.eligibility != "elig_512"
        or stage512a.assignment_eligibility != "elig_512"
    ):
        raise AssertionError("stage512a eligibility contract drift")

    # 512b used to be the "unknown stage" case here and this asserted that resolving it
    # raised "no frozen artifact contract".  It became a real stage in 9c573bf and the
    # assertion has been failing ever since, unnoticed because no job runs this file --
    # in a suite whose entire subject is the mechanism that keeps a 13-anchor build away
    # from the frozen 361-bucket Stage512a artifacts.  Assert the contract instead.
    stage512b = _resolve_offline_stage_identity(
        "512b", 512, {"target_area": 512 * 512, "bucket_mode": "anchors_v1"}
    )
    if (
        stage512b.eligibility != "elig_512"
        or stage512b.assignment_eligibility != "elig_512"
    ):
        raise AssertionError("stage512b eligibility contract drift")
    if stage512b.legacy_artifacts_allowed:
        raise AssertionError("stage512b must never be allowed to touch legacy artifacts")
    if stage512b.requires_nl or stage512b.anime_nl_filter_applied:
        raise AssertionError("stage512b must not inherit the 256 anime-NL filter")
    if stage512b == stage512a:
        raise AssertionError("stage512b and stage512a identities must be distinguishable")
    # The size guards are what stop a 512b run from being pointed at a 256 catalog.
    _expect_error(
        "requires explicit dataset.image_size=512",
        lambda: _resolve_offline_stage_identity("512b", None, {"target_area": 512 * 512}),
    )
    _expect_error(
        "requires dataset.image_size=512",
        lambda: _resolve_offline_stage_identity("512b", 256, {"target_area": 512 * 512}),
    )
    # A genuinely unknown stage must still be refused, which is what the old 512b
    # assertion was really testing.
    _expect_error(
        "no frozen artifact contract",
        lambda: _resolve_offline_stage_identity("1024", 1024, {"target_area": 1024 * 1024}),
    )
    _expect_error(
        "requires explicit dataset.image_size=512",
        lambda: _resolve_offline_stage_identity(
            "512a", None, {"target_area": 512 * 512}
        ),
    )
    _expect_error(
        "requires dataset.image_size=512",
        lambda: _resolve_offline_stage_identity(
            "512a", 256, {"target_area": 512 * 512}
        ),
    )
    _expect_error(
        "requires dataset.image_size=512",
        lambda: _resolve_offline_stage_identity(
            "512a", True, {"target_area": 512 * 512}
        ),
    )
    _expect_error(
        "requires dataset.multi_ar.target_area=262144",
        lambda: _resolve_offline_stage_identity("512a", 512, {}),
    )
    _expect_error(
        "requires dataset.multi_ar.target_area=262144",
        lambda: _resolve_offline_stage_identity(
            "512a", 512, {"target_area": 256 * 256}
        ),
    )


def _test_header_contracts() -> None:
    stage256 = _OFFLINE_STAGE_IDENTITIES["256"]
    stage512a = _OFFLINE_STAGE_IDENTITIES["512a"]

    if _validate_artifact_stage_identity(
        {},
        artifact="legacy plan",
        identity=stage256,
        requires_nl=True,
    ):
        raise AssertionError("legacy stage256 plan was marked explicit")
    if _validate_plan_stage_identity(
        {"format_version": 1},
        identity=stage256,
        requires_nl=True,
    ):
        raise AssertionError("legacy stage256 format-v1 plan was marked explicit")
    _expect_error(
        "runtime stage '256' requires requires_nl=True",
        lambda: _validate_artifact_stage_identity(
            {},
            artifact="legacy stage256 plan",
            identity=stage256,
            requires_nl=False,
        ),
    )
    _expect_error(
        "lacks required stage/eligibility identity",
        lambda: _validate_artifact_stage_identity(
            {},
            artifact="stage512a plan",
            identity=stage512a,
            requires_nl=False,
        ),
    )

    explicit_256 = {
        "stage": "256",
        "image_size": 256,
        "eligibility": "elig_256",
        "requires_nl": True,
        "anime_nl_filter_applied": False,
    }
    if not _validate_artifact_stage_identity(
        explicit_256,
        artifact="stage256 catalog",
        identity=stage256,
        requires_nl=True,
    ):
        raise AssertionError("explicit stage256 catalog was not explicit")
    _expect_error(
        "eligibility mismatch",
        lambda: _validate_artifact_stage_identity(
            {**explicit_256, "eligibility": "phase3a_stage256_v1"},
            artifact="stage256 catalog",
            identity=stage256,
            requires_nl=True,
        ),
    )

    explicit_512a = {
        "stage": "512a",
        "image_size": 512,
        "eligibility": "elig_512",
        "requires_nl": False,
        "anime_nl_filter_applied": False,
    }
    if not _validate_artifact_stage_identity(
        explicit_512a,
        artifact="stage512a catalog",
        identity=stage512a,
        requires_nl=False,
    ):
        raise AssertionError("explicit stage512a catalog was not explicit")
    if not _validate_artifact_stage_identity(
        explicit_512a,
        artifact="stage512a plan",
        identity=stage512a,
        requires_nl=False,
    ):
        raise AssertionError("explicit stage512a plan was not explicit")
    if not _validate_plan_stage_identity(
        {**explicit_512a, "format_version": 2},
        identity=stage512a,
        requires_nl=False,
    ):
        raise AssertionError("stage512a format-v2 plan was not explicit")
    _expect_error(
        "plan stage identity/format mismatch",
        lambda: _validate_plan_stage_identity(
            {**explicit_512a, "format_version": 1},
            identity=stage512a,
            requires_nl=False,
        ),
    )
    _expect_error(
        "plan stage identity/format mismatch",
        lambda: _validate_plan_stage_identity(
            {"format_version": 2},
            identity=stage256,
            requires_nl=True,
        ),
    )
    _expect_error(
        "stage mismatch",
        lambda: _validate_artifact_stage_identity(
            explicit_256,
            artifact="stage256 catalog",
            identity=stage512a,
            requires_nl=False,
        ),
    )
    _expect_error(
        "eligibility mismatch",
        lambda: _validate_artifact_stage_identity(
            {**explicit_512a, "eligibility": "elig_256"},
            artifact="mixed catalog",
            identity=stage512a,
            requires_nl=False,
        ),
    )
    _expect_error(
        "requires_nl mismatch",
        lambda: _validate_artifact_stage_identity(
            {**explicit_512a, "requires_nl": True},
            artifact="mixed catalog",
            identity=stage512a,
            requires_nl=False,
        ),
    )
    _expect_error(
        "anime_nl_filter_applied mismatch",
        lambda: _validate_artifact_stage_identity(
            {**explicit_512a, "anime_nl_filter_applied": True},
            artifact="mixed catalog",
            identity=stage512a,
            requires_nl=False,
        ),
    )
    _expect_error(
        "partial stage identity",
        lambda: _validate_artifact_stage_identity(
            {"stage": "512a"},
            artifact="partial plan",
            identity=stage512a,
            requires_nl=False,
        ),
    )


def _test_assignment_contracts() -> None:
    source = _assignment_source()
    stage256 = _assignment_identity_payload(
        source,
        identity=_OFFLINE_STAGE_IDENTITIES["256"],
        requires_nl=True,
    )
    expected_256 = {
        "bucket_table_hash": "a" * 64,
        "floor": 0.005,
        "kept": [0, 2],
        "fold_map": [0, 0, 2],
        "square_only_domains": ["synthetic", "textatlas"],
        "square_bucket_idx": 0,
        "assignment": "phase3a_vectorized_with_production_scalar_fallback_v1",
        "eligibility": "phase3a_stage256_v1",
    }
    if stage256 != expected_256:
        raise AssertionError(f"stage256 assignment ABI drift: {stage256!r}")

    stage512a = _assignment_identity_payload(
        source,
        identity=_OFFLINE_STAGE_IDENTITIES["512a"],
        requires_nl=False,
    )
    expected_512a = {
        "bucket_table_hash": "a" * 64,
        "floor": 0.005,
        "kept": [0, 2],
        "fold_map": [0, 0, 2],
        "square_only_domains": ["synthetic", "textatlas"],
        "square_bucket_idx": 0,
        "assignment": "phase3a_vectorized_with_production_scalar_fallback_v1",
        "stage": "512a",
        "image_size": 512,
        "eligibility": "elig_512",
        "requires_nl": False,
    }
    if stage512a != expected_512a:
        raise AssertionError(f"stage512a assignment ABI drift: {stage512a!r}")
    if _canonical_sha256(stage256) == _canonical_sha256(stage512a):
        raise AssertionError("stage256/stage512a assignment identities collided")


def _test_legacy_artifacts(catalog_path: Path, plan_path: Path) -> None:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    stage256 = _OFFLINE_STAGE_IDENTITIES["256"]
    if catalog.get("catalog_hash") != LEGACY_STAGE256_CATALOG_HASH:
        raise AssertionError("legacy stage256 catalog hash drift")
    if catalog.get("assignment_hash") != LEGACY_STAGE256_ASSIGNMENT_HASH:
        raise AssertionError("legacy stage256 assignment hash drift")
    if plan.get("plan_id") != LEGACY_STAGE256_PLAN_ID:
        raise AssertionError("legacy stage256 plan ID drift")
    if plan.get("catalog_hash") != LEGACY_STAGE256_CATALOG_HASH:
        raise AssertionError("legacy stage256 plan/catalog binding drift")
    if plan.get("assignment_hash") != LEGACY_STAGE256_ASSIGNMENT_HASH:
        raise AssertionError("legacy stage256 plan/assignment binding drift")
    if _validate_artifact_stage_identity(
        catalog,
        artifact="legacy catalog",
        identity=stage256,
        requires_nl=True,
    ):
        raise AssertionError("legacy stage256 catalog unexpectedly became explicit")
    if _validate_plan_stage_identity(
        plan,
        identity=stage256,
        requires_nl=True,
    ):
        raise AssertionError("legacy stage256 plan unexpectedly became explicit")
    rebuilt = _canonical_sha256(
        _assignment_identity_payload(
            catalog,
            identity=stage256,
            requires_nl=True,
        )
    )
    if rebuilt != LEGACY_STAGE256_ASSIGNMENT_HASH:
        raise AssertionError(
            f"legacy stage256 assignment reconstruction drift: {rebuilt}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-catalog-header", type=Path)
    parser.add_argument("--legacy-plan-header", type=Path)
    args = parser.parse_args()
    if (args.legacy_catalog_header is None) != (args.legacy_plan_header is None):
        parser.error("legacy catalog/plan headers must be supplied together")
    return args


def main() -> int:
    args = _parse_args()
    _test_runtime_contracts()
    _test_header_contracts()
    _test_assignment_contracts()
    legacy_checked = args.legacy_catalog_header is not None
    if legacy_checked:
        _test_legacy_artifacts(
            args.legacy_catalog_header.resolve(strict=True),
            args.legacy_plan_header.resolve(strict=True),
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "runtime_contracts": True,
                "header_negative_matrix": True,
                "assignment_abi": True,
                "legacy_stage256_artifacts_checked": legacy_checked,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
