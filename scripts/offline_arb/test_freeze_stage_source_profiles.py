#!/usr/bin/env python3
"""Small fail-closed tests for freeze stage profiles and GPIC binding."""

import hashlib
import json
import tempfile
import stat
from pathlib import Path

import scripts.offline_arb.make_freeze_manifest as freeze_module
from scripts.offline_arb.make_freeze_manifest import (
    CONTRACT_KEYS,
    SOURCE_EXPLICIT,
    SOURCE_EXPLICIT_STAGE512A,
    STAGE512A_EXCEPTION_COUNTS,
    STAGE512A_EXCEPTION_POLICY,
    STAGE512A_EXCEPTION_PRODUCER_IMAGE,
    STAGE512A_EXCEPTION_PRODUCER_SOURCES,
    STAGE512A_EXCEPTION_RUNTIME_SOURCES,
    STAGE512A_EXCEPTION_SCHEMA,
    SnapshotTracker,
    atomic_write_manifest,
    load_stage_contract,
    source_explicit_for_stage,
    validate_external_bindings,
    validate_stage512a_catalog_sampling,
)


def expect_value_error(callback, needle: str) -> None:
    try:
        callback()
    except ValueError as exc:
        assert needle in str(exc), (needle, str(exc))
    else:
        raise AssertionError(f"expected ValueError containing {needle!r}")


def main() -> int:
    assert source_explicit_for_stage("256") is SOURCE_EXPLICIT
    assert source_explicit_for_stage("512a") is SOURCE_EXPLICIT_STAGE512A

    assert SOURCE_EXPLICIT == (
        "train.py",
        "configs/stage256_offline_exact_v1.yaml",
        "jobs/offline_exact_256_2x8_gpuval.yaml",
        "jobs/offline_exact_256_8x8_FORMAL.yaml",
        "jobs/offline_arb_freeze_v3.yaml",
        "jobs/offline_arb_w64_raw_supply_gate.yaml",
        "jobs/offline_arb_w64_raw_supply_analyze.yaml",
        "scripts/offline_arb/make_freeze_manifest.py",
        "scripts/offline_arb/verify_offline_gpic_compact_controls.py",
        "scripts/offline_arb/run_w64_raw_supply_gate.py",
        "scripts/offline_arb/analyze_w64_raw_supply_gate.py",
        "scripts/offline_arb/caption_sidecar_contract.py",
        "scripts/offline_arb/build_caption_sidecar.py",
        "scripts/offline_arb/verify_caption_sidecar.py",
        "scripts/offline_arb/test_caption_sidecar_runtime.py",
        "training/dataline_offline/caption_sidecar.py",
        "training/dataline_offline/offline_dataset.py",
        "training/config_schema.py",
    )
    assert SOURCE_EXPLICIT_STAGE512A == (
        "train.py",
        "configs/stage512a_offline_exact_w64_v1.yaml",
        "configs/stage256_offline_exact_v1.yaml",
        "jobs/offline_arb_phase5_catalog_stage512a.yaml",
        "jobs/offline_arb_phase5_plans_stage512a.yaml",
        "jobs/offline_arb_animefreq_stage512a.yaml",
        "jobs/offline_arb_caption_sidecar_build_stage512a.yaml",
        "jobs/offline_arb_exceptions_audit_stage512a_w64.yaml",
        "jobs/offline_arb_textatlas_catalog_evidence_rebind_w64_TEMPLATE.yaml",
        "jobs/offline_arb_stage256_completion_gate_for_stage512a.yaml",
        "jobs/offline_arb_stage512a_w64_reviewed_exception_gate.yaml",
        "jobs/offline_arb_freeze_v4_stage512a_w64_reviewed.yaml",
        "jobs/offline_exact_512a_2x8_GPUVAL_TEMPLATE.yaml",
        "jobs/offline_exact_512a_8x8_FORMAL_TEMPLATE.yaml",
        "scripts/offline_arb/make_freeze_manifest.py",
        "scripts/offline_arb/test_freeze_stage_source_profiles.py",
        "scripts/offline_arb/phase5_build_catalog.py",
        "scripts/offline_arb/make_planner_input.py",
        "scripts/offline_arb/build_plan.py",
        "scripts/offline_arb/validate_plan.py",
        "scripts/offline_arb/precompute_anime_freq.py",
        "scripts/offline_arb/test_animefreq_input_contract.py",
        "scripts/offline_arb/render_stage512a_write_job.py",
        "scripts/offline_arb/test_stage512a_write_job_renderer.py",
        "scripts/offline_arb/caption_sidecar_contract.py",
        "scripts/offline_arb/build_caption_sidecar.py",
        "scripts/offline_arb/verify_caption_sidecar.py",
        "scripts/offline_arb/test_caption_sidecar_runtime.py",
        "scripts/offline_arb/audit_planned_caption_availability.py",
        "scripts/offline_arb/audit_textatlas_plan_rows.py",
        "scripts/offline_arb/rebind_textatlas_catalog_evidence_w64.py",
        "scripts/offline_arb/test_rebind_textatlas_catalog_evidence_w64.py",
        "scripts/offline_arb/immutable_review_evidence.py",
        "scripts/offline_arb/stage512a_w64_exception_contract.py",
        "scripts/offline_arb/finalize_stage512a_w64_exception_gate.py",
        "scripts/offline_arb/test_stage512a_w64_exception_gate_contract.py",
        "scripts/offline_arb/test_stage512a_w64_runtime_profiles.py",
        "scripts/offline_arb/test_stage512a_w64_reviewed_exceptions.py",
        "scripts/offline_arb/completion_publication.py",
        "scripts/offline_arb/test_completion_publication.py",
        "scripts/offline_arb/verify_stage256_completion_for_stage512a.py",
        "scripts/offline_arb/test_stage256_completion_contract.py",
        "scripts/offline_arb/stage512a_launch_guard.py",
        "scripts/offline_arb/render_stage512a_formal_launch.py",
        "scripts/offline_arb/test_stage512a_launch_guard.py",
        "scripts/offline_arb/stage512a_gpuval_guard.py",
        "scripts/offline_arb/render_stage512a_gpuval.py",
        "scripts/offline_arb/test_stage512a_gpuval_contract.py",
        "training/dataline_offline/caption_sidecar.py",
        "training/dataline_offline/offline_dataset.py",
        "training/dataline_offline/caption_exceptions_v1.py",
        "training/dataline_offline/image_exceptions_v1.py",
        "training/dataline/image_fetch.py",
        "training/config_schema.py",
    )
    stage256 = set(source_explicit_for_stage("256"))
    stage512a = set(source_explicit_for_stage("512a"))
    assert len(SOURCE_EXPLICIT_STAGE512A) == len(stage512a)
    for forbidden in (
        "jobs/offline_exact_256_2x8_gpuval.yaml",
        "jobs/offline_exact_256_8x8_FORMAL.yaml",
        "jobs/offline_arb_freeze_v3.yaml",
        "jobs/offline_arb_freeze_v3_stage512a.yaml",
        "jobs/offline_arb_w64_raw_supply_gate.yaml",
        "jobs/offline_arb_w64_raw_supply_analyze.yaml",
        "configs/stage512a_offline_exact_v2.yaml",
        "jobs/offline_arb_exceptions_audit_stage512a.yaml",
        "jobs/offline_arb_stage512a_reviewed_exception_gate.yaml",
        "jobs/offline_arb_freeze_v4_stage512a_reviewed.yaml",
        "jobs/offline_exact_512a_6x8_FORMAL_TEMPLATE.yaml",
        "scripts/offline_arb/finalize_stage512a_exception_gate.py",
        "scripts/offline_arb/test_stage512a_exception_gate_contract.py",
        "scripts/offline_arb/test_stage512a_reviewed_exceptions.py",
    ):
        assert forbidden not in stage512a, forbidden
    for stage512a_only in (
        "configs/stage512a_offline_exact_w64_v1.yaml",
        "jobs/offline_arb_stage512a_w64_reviewed_exception_gate.yaml",
        "jobs/offline_arb_freeze_v4_stage512a_w64_reviewed.yaml",
        "jobs/offline_exact_512a_2x8_GPUVAL_TEMPLATE.yaml",
        "jobs/offline_exact_512a_8x8_FORMAL_TEMPLATE.yaml",
        "scripts/offline_arb/test_animefreq_input_contract.py",
        "scripts/offline_arb/render_stage512a_write_job.py",
        "scripts/offline_arb/test_stage512a_write_job_renderer.py",
        "scripts/offline_arb/stage512a_gpuval_guard.py",
        "scripts/offline_arb/render_stage512a_gpuval.py",
        "scripts/offline_arb/test_stage512a_gpuval_contract.py",
        "scripts/offline_arb/test_stage512a_launch_guard.py",
        "scripts/offline_arb/stage512a_w64_exception_contract.py",
        "scripts/offline_arb/finalize_stage512a_w64_exception_gate.py",
        "scripts/offline_arb/test_stage512a_w64_reviewed_exceptions.py",
    ):
        assert stage512a_only not in stage256, stage512a_only

    assert STAGE512A_EXCEPTION_SCHEMA == (
        "offline_arb_stage512a_w64_reviewed_exception_gate_v1"
    )
    assert STAGE512A_EXCEPTION_POLICY == (
        "reviewed_exact_stage512a_w64_exceptions_v1"
    )
    assert STAGE512A_EXCEPTION_COUNTS == {
        "monet_empty_prompt_rows": 6,
        "monet_empty_prompt_formal_exposures": 19,
        "monet_empty_prompt_horizon_exposures": 20,
        "textatlas_raw_candidate_rows": 43,
        "textatlas_raw_candidate_formal_exposures": 134,
        "textatlas_raw_candidate_horizon_exposures": 137,
        "textatlas_production_decode_success_rows": 40,
        "textatlas_production_decode_success_formal_exposures": 125,
        "textatlas_production_decode_success_horizon_exposures": 127,
        "textatlas_repaired_rows": 3,
        "textatlas_repaired_formal_exposures": 9,
        "textatlas_repaired_horizon_exposures": 10,
    }
    assert freeze_module.STAGE512A_REVIEWED_GATE_PATH == (
        freeze_module.BASE
        / "offline_arb_exceptions_stage512a_w64_reviewed_v1/FORMAL_PASS.json"
    )
    assert freeze_module.STAGE512A_REVIEWED_FREEZE_BASENAME == (
        "FREEZE_MANIFEST_STAGE512A_W64_REVIEWED_EXCEPTIONS_V1.json"
    )
    assert freeze_module.STAGE512A_REVIEWED_FREEZE_PATH == (
        freeze_module.BASE
        / "offline_arb_plans_stage512a_w64_v2"
        / freeze_module.STAGE512A_REVIEWED_FREEZE_BASENAME
    )
    gate_authority = freeze_module.STAGE512A_REVIEWED_GATE_SHA256
    assert isinstance(gate_authority, str)
    pending_gate_authority = (
        gate_authority.startswith("__STAGE512A_")
        and gate_authority.endswith("_PENDING__")
    )
    assert pending_gate_authority or (
        len(gate_authority) == 64
        and set(gate_authority) <= set("0123456789abcdef")
    )

    for unsupported in ("", "512", "512A", "512b"):
        try:
            source_explicit_for_stage(unsupported)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsupported stage fell back to a source profile: {unsupported!r}")

    with tempfile.TemporaryDirectory(prefix="freeze-exclusive-publish-") as raw_root:
        root = Path(raw_root).resolve(strict=True)
        output = root / "FREEZE.json"
        atomic_write_manifest(output, {"status": "PASS"})
        assert stat.S_IMODE(output.stat().st_mode) == 0o444
        expect_value_error(
            lambda: atomic_write_manifest(output, {"status": "REPLACED"}),
            "already exists",
        )
        real_parent = root / "real"
        real_parent.mkdir()
        symlink_parent = root / "linked"
        symlink_parent.symlink_to(real_parent, target_is_directory=True)
        expect_value_error(
            lambda: atomic_write_manifest(
                symlink_parent / "FREEZE.json", {"status": "PASS"}
            ),
            "existing real directory",
        )

    identity = {
        "stage": "512a",
        "image_size": 512,
        "eligibility": "elig_512",
        "requires_nl": False,
        "anime_nl_filter_applied": False,
    }
    contract = {**identity, "legacy_stage256_headers": False}
    catalog_hash = "a" * 64
    catalog_header_sha256 = "b" * 64
    plan_header_sha256 = "1" * 64
    plan_header = {
        "plan_id": "stage512a_offline_exact_v2_w64_prod",
        "plan_hash": "2" * 64,
        "assignment_hash": "3" * 64,
    }
    catalog_header = {"catalog_hash": catalog_hash}
    with tempfile.TemporaryDirectory(prefix="freeze-gpic-binding-") as raw_root:
        root = Path(raw_root).resolve(strict=True)
        repo = root / "repo"
        runtime: dict[str, dict[str, str]] = {}
        for label, relative in STAGE512A_EXCEPTION_RUNTIME_SOURCES.items():
            source = repo / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# {label}\n", encoding="utf-8")
            runtime[label] = {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        producer_artifacts: dict[str, dict[str, object]] = {}
        for label, relative in STAGE512A_EXCEPTION_PRODUCER_SOURCES.items():
            source = repo / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# producer {label}\n", encoding="utf-8")
            producer_artifacts[label] = {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        rendered_yaml = root / "rendered-reviewed-gate.yaml"
        rendered_yaml.write_text("name: reviewed-gate\n", encoding="utf-8")
        rendered_yaml.chmod(0o444)
        render_payload_sha = "4" * 64
        producer = {
            "schema": "stage512a_reviewed_gate_producer_provenance_v1",
            "job_name": "stage512a-reviewed-gate-test",
            "container_image": STAGE512A_EXCEPTION_PRODUCER_IMAGE,
            "launch_token": f"STAGE512A_WRITE_V1_{render_payload_sha}",
            "render_payload_sha256": render_payload_sha,
            "rendered_yaml": {
                "path": str(rendered_yaml),
                "bytes": rendered_yaml.stat().st_size,
                "sha256": hashlib.sha256(rendered_yaml.read_bytes()).hexdigest(),
            },
            **producer_artifacts,
        }
        artifact_root = root / "immutable-build-0001"
        artifact_root.mkdir()
        release_path = root / "FORMAL_RELEASE.json"
        exception_parent = root / "reviewed-gate"
        exception_parent.mkdir()
        exception_path = exception_parent / "FORMAL_PASS.json"
        positive = {
            "status": "PASS",
            "stage_identity": identity,
            "catalog_hash": catalog_hash,
            "catalog_header_sha256": catalog_header_sha256,
            "artifact_root": str(artifact_root),
            "manifest_sha256": "c" * 64,
            "release_digest": "d" * 64,
        }
        positive_exception = {
            "status": "PASS",
            "schema": STAGE512A_EXCEPTION_SCHEMA,
            "policy": STAGE512A_EXCEPTION_POLICY,
            "stage_identity": identity,
            **plan_header,
            "catalog_hash": catalog_hash,
            "plan_header_sha256": plan_header_sha256,
            "catalog_header_sha256": catalog_header_sha256,
            "counts": STAGE512A_EXCEPTION_COUNTS,
            "runtime": runtime,
            "producer": producer,
        }
        exception_path.write_text(
            json.dumps(positive_exception) + "\n", encoding="utf-8"
        )
        exception_path.chmod(0o444)
        exception_parent.chmod(0o555)
        original_gate_path = freeze_module.STAGE512A_REVIEWED_GATE_PATH
        original_gate_sha256 = freeze_module.STAGE512A_REVIEWED_GATE_SHA256
        freeze_module.STAGE512A_REVIEWED_GATE_PATH = exception_path
        freeze_module.STAGE512A_REVIEWED_GATE_SHA256 = hashlib.sha256(
            exception_path.read_bytes()
        ).hexdigest()

        def write_exception(value: dict) -> None:
            exception_path.chmod(0o644)
            try:
                exception_path.write_text(
                    json.dumps(value) + "\n", encoding="utf-8"
                )
            finally:
                exception_path.chmod(0o444)
            freeze_module.STAGE512A_REVIEWED_GATE_SHA256 = hashlib.sha256(
                exception_path.read_bytes()
            ).hexdigest()

        def write(value: dict) -> None:
            release_path.write_text(json.dumps(value) + "\n", encoding="utf-8")

        def expect_failure(value: dict, label: str) -> None:
            write(value)
            try:
                validate_external_bindings(
                    {
                        "gpic_payload": release_path,
                        "reviewed_exception_gate": exception_path,
                    },
                    contract,
                    plan_header,
                    catalog_header,
                    plan_header_sha256,
                    catalog_header_sha256,
                    repo,
                    SnapshotTracker(),
                )
            except ValueError:
                pass
            else:
                raise AssertionError(f"malformed GPIC binding passed: {label}")

        write(positive)
        validate_external_bindings(
            {
                "gpic_payload": release_path,
                "reviewed_exception_gate": exception_path,
            },
            contract,
            plan_header,
            catalog_header,
            plan_header_sha256,
            catalog_header_sha256,
            repo,
            SnapshotTracker(),
        )
        sealed_gate_sha256 = freeze_module.STAGE512A_REVIEWED_GATE_SHA256
        freeze_module.STAGE512A_REVIEWED_GATE_SHA256 = "0" * 64
        expect_value_error(
            lambda: validate_external_bindings(
                {
                    "gpic_payload": release_path,
                    "reviewed_exception_gate": exception_path,
                },
                contract,
                plan_header,
                catalog_header,
                plan_header_sha256,
                catalog_header_sha256,
                repo,
                SnapshotTracker(),
            ),
            "differs from the sealed W64 authority",
        )
        freeze_module.STAGE512A_REVIEWED_GATE_SHA256 = sealed_gate_sha256
        try:
            validate_external_bindings(
                {}, contract, plan_header, catalog_header, plan_header_sha256,
                catalog_header_sha256, repo, SnapshotTracker()
            )
        except ValueError:
            pass
        else:
            raise AssertionError("stage512a freeze accepted a missing GPIC binding")

        for key, bad_value in (
            ("status", "FAIL"),
            ("stage_identity", {**identity, "requires_nl": True}),
            ("catalog_hash", "e" * 64),
            ("catalog_header_sha256", "f" * 64),
            ("manifest_sha256", "not-a-sha"),
            ("release_digest", "not-a-sha"),
        ):
            expect_failure({**positive, key: bad_value}, key)

        bad_exception = {
            **positive_exception,
            "counts": {
                **STAGE512A_EXCEPTION_COUNTS,
                "textatlas_repaired_rows": 4,
            },
        }
        write_exception(bad_exception)
        write(positive)
        expect_value_error(
            lambda: validate_external_bindings(
                {
                    "gpic_payload": release_path,
                    "reviewed_exception_gate": exception_path,
                },
                contract,
                plan_header,
                catalog_header,
                plan_header_sha256,
                catalog_header_sha256,
                repo,
                SnapshotTracker(),
            ),
            "exact count contract",
        )
        bad_producer_exception = {
            **positive_exception,
            "producer": {
                **producer,
                "renderer": {
                    **producer["renderer"],
                    "sha256": "5" * 64,
                },
            },
        }
        write_exception(bad_producer_exception)
        expect_value_error(
            lambda: validate_external_bindings(
                {
                    "gpic_payload": release_path,
                    "reviewed_exception_gate": exception_path,
                },
                contract,
                plan_header,
                catalog_header,
                plan_header_sha256,
                catalog_header_sha256,
                repo,
                SnapshotTracker(),
            ),
            "producer.renderer byte identity mismatch",
        )
        assert set(identity) == set(CONTRACT_KEYS)
        freeze_module.STAGE512A_REVIEWED_GATE_PATH = original_gate_path
        freeze_module.STAGE512A_REVIEWED_GATE_SHA256 = original_gate_sha256

    with tempfile.TemporaryDirectory(prefix="freeze-stage512a-sampling-") as raw_root:
        root = Path(raw_root).resolve(strict=True)
        sampling = root / "sampling.json"
        sampling_value = {
            "anime": {"caption": {"require_nl_stages": []}},
            "domain_mix": {
                "512a": {
                    "anime": 0.37,
                    "monet_real": 0.40,
                    "gpic": 0.13,
                    "synthetic": 0.05,
                    "textatlas": 0.05,
                }
            },
        }
        sampling.write_text(json.dumps(sampling_value) + "\n", encoding="utf-8")
        config = root / "stage512a.yaml"
        config.write_text(
            "dataset:\n"
            "  stage: 512a\n"
            "  image_size: 512\n"
            f"  sampling_config: {sampling}\n",
            encoding="utf-8",
        )
        tracker = SnapshotTracker()
        sampling_contract, sampling_evidence = load_stage_contract(
            config, sampling, tracker
        )
        assert sampling_contract == identity
        assert sampling_evidence == {
            "sha256": hashlib.sha256(sampling.read_bytes()).hexdigest(),
            "domain_mix": sampling_value["domain_mix"]["512a"],
        }
        tracker.close()

        catalog_sampling = {
            "inputs": {
                "sampling_config_sha256": sampling_evidence["sha256"],
                "domain_mix": dict(sampling_evidence["domain_mix"]),
            }
        }
        plan_sampling = {
            "domain_mix": {
                "anime": "0.37",
                "monet_real": "0.4",
                "gpic": "0.13",
                "synthetic": "0.05",
                "textatlas": "0.05",
            },
            "sampling_config_sha256": sampling_evidence["sha256"],
        }
        validate_stage512a_catalog_sampling(
            plan_sampling, catalog_sampling, sampling_contract, sampling_evidence
        )
        plan_without_sampling_sha = dict(plan_sampling)
        plan_without_sampling_sha.pop("sampling_config_sha256")
        validate_stage512a_catalog_sampling(
            plan_without_sampling_sha,
            catalog_sampling,
            sampling_contract,
            sampling_evidence,
        )

        bad_sampling_sha = json.loads(json.dumps(catalog_sampling))
        bad_sampling_sha["inputs"]["sampling_config_sha256"] = "f" * 64
        expect_value_error(
            lambda: validate_stage512a_catalog_sampling(
                plan_sampling,
                bad_sampling_sha,
                sampling_contract,
                sampling_evidence,
            ),
            "sampling SHA-256 differs",
        )
        bad_sampling_mix = json.loads(json.dumps(catalog_sampling))
        bad_sampling_mix["inputs"]["domain_mix"]["anime"] = 0.36
        expect_value_error(
            lambda: validate_stage512a_catalog_sampling(
                plan_sampling,
                bad_sampling_mix,
                sampling_contract,
                sampling_evidence,
            ),
            "domain_mix differs",
        )
        bad_plan_mix = json.loads(json.dumps(plan_sampling))
        bad_plan_mix["domain_mix"]["anime"] = "0.36"
        expect_value_error(
            lambda: validate_stage512a_catalog_sampling(
                bad_plan_mix,
                catalog_sampling,
                sampling_contract,
                sampling_evidence,
            ),
            "plan domain_mix differs",
        )
        bad_plan_sha = json.loads(json.dumps(plan_sampling))
        bad_plan_sha["sampling_config_sha256"] = "e" * 64
        expect_value_error(
            lambda: validate_stage512a_catalog_sampling(
                bad_plan_sha,
                catalog_sampling,
                sampling_contract,
                sampling_evidence,
            ),
            "plan sampling SHA-256 differs",
        )
        validate_stage512a_catalog_sampling(
            {}, {}, {"stage": "256"}, None
        )

    print("FREEZE_STAGE_SOURCE_PROFILES_AND_GPIC_BINDING_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
