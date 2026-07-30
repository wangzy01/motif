#!/usr/bin/env python3
"""Lightweight regression tests for legacy planner v1 and stage512a v2 identity."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.offline_arb.make_planner_input import (
    _stable_sampling_sha256,
    _validate_stage512a_catalog_sampling,
)
from training.dataline_offline.planner_v1 import (
    OfflinePlan,
    PlannerInputError,
    _normalize_input,
    build_plan,
    build_plan_from_file,
)


V1_CONFIG_SHA256 = "cace8c7a2c240644de7a82ca327f69ad21b5bf7db401bd5b1f86958b0732b101"
V1_PLAN_SHA256 = "ac5878be02196bccfe2c9e67337f95ac9ef43a881f35449e1b1d94b434a04174"
V1_BLOCKS_SHA256 = "86a3e01327488191a7ef79b4fc251d6ef83479882809d49fcb4f0a52b388a95c"


def base_input(version: int) -> dict:
    size = 256 if version == 1 else 512
    raw = {
        "format_version": version,
        "plan_id": "compat_v1" if version == 1 else "stage512a_offline_exact_v2_unit",
        "seed": 7,
        "world_size": 1,
        "per_rank_bs": 2,
        "total_steps": 4,
        "domains": ["anime"],
        "domain_mix": {"anime": 1.0},
        "square_only_domains": [],
        "buckets": [[size, size]],
        "square_bucket_idx": 0,
        "N": [[8]],
        "R_max": {"anime": 2},
        "epsilon_jitter": False,
        "block_steps": 2,
        "catalog_hash": "a" * 64,
        "assignment_hash": "b" * 64,
    }
    if version == 2:
        raw.update(
            {
                "stage": "512a",
                "image_size": 512,
                "eligibility": "elig_512",
                "requires_nl": False,
                "anime_nl_filter_applied": False,
            }
        )
    return raw


def expect_input_error(raw: dict, needle: str) -> None:
    try:
        _normalize_input(raw)
    except PlannerInputError as exc:
        assert needle in str(exc), (needle, str(exc))
    else:
        raise AssertionError(f"expected PlannerInputError containing {needle!r}")


def expect_system_exit(callback, needle: str) -> None:
    try:
        callback()
    except SystemExit as exc:
        assert needle in str(exc), (needle, str(exc))
    else:
        raise AssertionError(f"expected SystemExit containing {needle!r}")


def main() -> int:
    v1 = _normalize_input(base_input(1))
    assert v1.config_sha256 == V1_CONFIG_SHA256
    assert (
        v1.stage
        is v1.image_size
        is v1.eligibility
        is v1.requires_nl
        is v1.anime_nl_filter_applied
        is None
    )

    v2_raw = base_input(2)
    v2 = _normalize_input(v2_raw)
    assert (v2.stage, v2.image_size, v2.eligibility, v2.requires_nl) == (
        "512a",
        512,
        "elig_512",
        False,
    )

    for missing in (
        "stage",
        "image_size",
        "eligibility",
        "requires_nl",
        "anime_nl_filter_applied",
    ):
        bad = copy.deepcopy(v2_raw)
        del bad[missing]
        expect_input_error(bad, "missing required")
    for key, value, needle in (
        ("stage", "512", "canonical"),
        ("image_size", 256, "stage/image_size"),
        ("eligibility", "elig_256", "derived from image_size"),
        ("requires_nl", True, "stage NL policy"),
        ("anime_nl_filter_applied", True, "stage NL policy"),
        ("catalog_hash", "not-a-sha", "lowercase SHA-256"),
        ("plan_id", "stage256_offline_exact_v2_wrong", "must start with"),
    ):
        bad = copy.deepcopy(v2_raw)
        bad[key] = value
        expect_input_error(bad, needle)

    explicit_256_v2 = copy.deepcopy(v2_raw)
    explicit_256_v2.update(
        {
            "plan_id": "stage256_offline_exact_v2_unit",
            "stage": "256",
            "image_size": 256,
            "eligibility": "elig_256",
            "requires_nl": True,
            "anime_nl_filter_applied": False,
            "buckets": [[256, 256]],
        }
    )
    explicit_256 = _normalize_input(explicit_256_v2)
    assert (explicit_256.requires_nl, explicit_256.anime_nl_filter_applied) == (
        True,
        False,
    )
    bad_256 = copy.deepcopy(explicit_256_v2)
    bad_256["anime_nl_filter_applied"] = True
    expect_input_error(bad_256, "stage NL policy")

    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="stage512a_planner_v2.") as td:
        root = Path(td)
        sampling_path = root / "sampling.json"
        sampling_path.write_text('{"fixture":"sampling-v1"}\n', encoding="utf-8")
        sampling_sha256, sampling_fingerprint = _stable_sampling_sha256(
            sampling_path
        )
        stage512a_mix = {
            "anime": 0.37,
            "monet_real": 0.40,
            "gpic": 0.13,
            "synthetic": 0.05,
            "textatlas": 0.05,
        }
        catalog_sampling = {
            "inputs": {
                "sampling_config_sha256": sampling_sha256,
                "domain_mix": dict(stage512a_mix),
            }
        }
        _validate_stage512a_catalog_sampling(
            catalog_sampling, stage512a_mix, sampling_sha256
        )

        bad_sampling_sha = copy.deepcopy(catalog_sampling)
        bad_sampling_sha["inputs"]["sampling_config_sha256"] = "f" * 64
        expect_system_exit(
            lambda: _validate_stage512a_catalog_sampling(
                bad_sampling_sha, stage512a_mix, sampling_sha256
            ),
            "SHA-256 differs",
        )
        bad_sampling_mix = copy.deepcopy(catalog_sampling)
        bad_sampling_mix["inputs"]["domain_mix"]["anime"] = 0.36
        expect_system_exit(
            lambda: _validate_stage512a_catalog_sampling(
                bad_sampling_mix, stage512a_mix, sampling_sha256
            ),
            "domain_mix differs",
        )
        sampling_path.write_text(
            '{"fixture":"sampling-v2-with-different-size"}\n', encoding="utf-8"
        )
        expect_system_exit(
            lambda: _stable_sampling_sha256(
                sampling_path, expected_fingerprint=sampling_fingerprint
            ),
            "fingerprint changed",
        )

        v1_plan = root / "v1"
        v1_header = build_plan(v1, v1_plan)
        assert v1_header["format_version"] == 1
        assert v1_header["plan_hash"] == V1_PLAN_SHA256
        assert v1_header["hashes"]["blocks_manifest_sha256"] == V1_BLOCKS_SHA256
        assert not any(
            key in v1_header
            for key in (
                "stage",
                "image_size",
                "eligibility",
                "requires_nl",
                "anime_nl_filter_applied",
            )
        )

        v2_input = root / "planner_input.json"
        v2_input.write_text(json.dumps(v2_raw, indent=1) + "\n", encoding="utf-8")
        v2_plan = root / "v2"
        v2_header = build_plan_from_file(v2_input, v2_plan)
        for key in (
            "stage",
            "image_size",
            "eligibility",
            "requires_nl",
            "anime_nl_filter_applied",
        ):
            assert v2_header[key] == v2_raw[key]
        reader = OfflinePlan(v2_plan, expected_input=v2_raw, verify_checksums=True)
        reader.verify_all_checksums()

        report = root / "validate.json"
        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts/offline_arb/validate_plan.py"),
                "--input",
                str(v2_input),
                "--plan",
                str(v2_plan),
                "--report",
                str(report),
            ],
            check=True,
        )
        result = json.loads(report.read_text(encoding="utf-8"))
        assert result["overall"] == "PASS", result

    print(
        "STAGE512A_PLANNER_V2_PASS "
        f"v1_config={V1_CONFIG_SHA256[:12]} v2_config={v2.config_sha256[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
