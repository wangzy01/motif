#!/usr/bin/env python3
"""Fail-closed static/artifact preflight for the Stage512a RNG-fix control.

This intentionally does no GPU work and creates no files.  A FORMAL pod runs it
before rendezvous.  The fresh plan remains additionally protected by the runtime
WeightedOfflinePlan reader and plan-specific exception closure tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

from omegaconf import OmegaConf

from training.dataline_offline.weighted_plan_reader_v1 import WeightedOfflinePlan


DATA_ROOT = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop"
)
CONFIG_DEFAULT = Path("configs/stage512a_continue_control_rngfix_v1.yaml")
REFERENCE_CONFIG = Path("configs/stage512a_20260722_weighted_w64_v1.yaml")
EXPECTED_REFERENCE_CONFIG_SHA = (
    "611635fc13b04c6d62c20875eff541736afbe6e35cc17c1f81caafb7f823a380"
)
EXPECTED_CKPT = Path(
    "/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/"
    "stage512a_continue_control_rngfix_v1"
)
EXPECTED_INIT = Path(
    "/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/"
    "stage512a_continue_control_rngfix_v1_inputs/a_final_step125000.last.pt"
)
EXPECTED_INIT_SHA = (
    "957bfcf12b89c4042a32056566012a2451e72aa0b2706b0ca51808ab79155c31"
)
EXPECTED_CATALOG = DATA_ROOT / "offline_arb_catalog_stage512a_v1"
EXPECTED_CATALOG_HASH = (
    "370b1679e8c5670338a03ba4c6972cdbf69fa5068fc35108bf646b2d64642331"
)
EXPECTED_PLAN = DATA_ROOT / (
    "offline_arb_plan_stage512a_continue_control_rngfix_v1_"
    "w64_256m_seed20260710_20260809"
)
EXPECTED_PLAN_ID = (
    "stage512a_offline_exact_v2_continue_control_rngfix_v1_"
    "w64_256m_seed20260710"
)
EXPECTED_PLAN_HASH = (
    "fbb5fcbd91d80daff8dfe89693f38eb3fa6ddeb657d01198df7fcac9cdcd039c"
)
EXPECTED_PLAN_HEADER_SHA = (
    "5fbe7ca546d3e77ef37d95bf52c7ded2f9741d0b4377fbadb8b059e9b9a1dce0"
)
EXPECTED_PLAN_CHECKSUMS_SHA = (
    "aa8283436369f3ddc2f2c7648c28f56cb8b8619b2e87b660f7b2ca29847fc8f0"
)
PLAN_AUDIT_ROOT = Path(
    "/fsx/evan/audits/stage512a_continue_control_rngfix_v1/"
    "plan_closure_v1_20260809"
)
EXPECTED_CANONICAL_PLAN_RECEIPT_SHA = (
    "5178207100c8af20230489dac88b9118665c93abb33ab5c97b6ccc532b2ea6ca"
)
EXPECTED_PLAN_CLOSURE_SHA = (
    "cb92872fc8f13cae10bfafaeb3238c9cc0159bf0c31a096e85ee9b188eb53104"
)
OLD_CONSUMED_PLAN_HASH = (
    "7295ca937d5002a0cf5e207c1d466c5992352e395adc9ae1c33f3cb616f675db"
)
OLD_CONSUMED_PLAN = DATA_ROOT / (
    "offline_arb_plan_stage512a_w64_256m_weighted_v1_20260722"
)
EXPECTED_S_SHA = "8ea5aeee31c1e1cdcb6cc86da98b7036ab974721c7fdba2aa7d1de14fffede8b"
EXPECTED_C_SHA = "e9ebcb74e03a33b1541b592f608e6e2433c7254108b209a75b54cd59f49eddae"
EXPECTED_D_SHA = "0ee380eba2863b3b78efa4d21306ef14f98c874d328edc9d934868e139ccc222"
EXPECTED_REAL_BAND_SHA = (
    "86e6d269459d07ed47814c8e103cb021ee1a13a1df028dd6fb09457998771727"
)
EXPECTED_SAMPLING = Path(
    "/fsx/evan/mot_code/data_analysis/configs/t5_sampling_redesign_v1.json"
)
EXPECTED_CONTRACT_SHA = (
    "c228f8b8400c4f96033431816b650d6e7093b3bf8e164018d22e694c60b96c38"
)
EXPECTED_BLOB = (
    DATA_ROOT
    / "offline_gpic_blob_stage512a_v1"
    / "stage512a-gpic-payload-reuse-v1"
)
EXPECTED_BLOB_MANIFEST_SHA = (
    "c9365753356f942caca1ea622c788ae5d05db320ecfa5804595b25ac8af3a648"
)
EXPECTED_GPIC_DENSE_INDEX_SHA = (
    "37f1a2bca7a7236c13f0d757547175a654647ea2f81de526a553f113bac7bded"
)
EXPECTED_GPIC_REUSE_PROOF_SHA = (
    "fac2e0bb040be6c393d5934309d4b2b90d809dd8de7f0b71554487dec2736dc8"
)
EXPECTED_GPIC_VERIFIER_SHA = (
    "75cc61664107e45371d452eda60b10ff4123c907a5e1997dfbcacb1f63417f69"
)
ASSET_REPORT = DATA_ROOT / "stage512a_continue_asset_rehash_20260809v2/REPORT.json"
EXPECTED_ASSET_REPORT_SHA = (
    "71c69d5dd56e5e1ac09ca7f3fbd9f380d6ce3bdbc5e328d3a7111a8da675041a"
)
REGISTRY_ROOT = Path(
    "/fsx/evan/audits/stage512a_continue_control_rngfix_v1/"
    "registry_closure_v2_20260809"
)
EXPECTED_REGISTRY_RECEIPT_SHA = (
    "5eb979532d44f2abeaf91d5640f883b19ec2b2d8eb8810819f66fad77b521451"
)
EXPECTED_REGISTRY_SUCCESS_SHA = (
    "1f5766e38ce5476ea6b26655808f0934872a8cdda104026d092ba95979bf2425"
)

# Every effective config leaf not listed here must equal the historical A config.
# This is intentionally a closed allowlist: changing EXPECTED_COMMIT cannot silently
# turn a future config edit into a new scientific variable.
ALLOWED_CONFIG_DELTAS = frozenset(
    {
        "dataset.offline_plan_root",
        "dataset.offline_plan_hash",
        "dataset.gpic_threads_per_node",
        "dataset.gpic_payload_backend",
        "dataset.offline_gpic_blob_root",
        "dataset.gpic_blob_manifest_sha256",
        "dataset.gpic_blob_fd_cache_size",
        "model.rank_independent_seed",
        "trainer.checkpoint_steps",
        "trainer.checkpoint_keep_every",
        "trainer.checkpoint_keep_format",
        "trainer.checkpoint_dir",
        # Operational fail-closed fence; historical A predates the formal
        # launcher and therefore has no equivalent leaf.
        "trainer.formal_launch_required",
        "trainer.init_from",
        "trainer.init_from_rng_policy",
        "trainer.gc_pause_probe",
    }
)

SMALL_FILE_HASHES = {
    EXPECTED_SAMPLING: "07a0b46fa9ebb840ecf7f97cc7fccc0ec68e44c47368ae13c450bcfe81a4fd48",
    EXPECTED_CATALOG / "header.json": "51a5f0904987a751b54842633d2968ea7f6485307850858d9472319c15748d1d",
    DATA_ROOT / "offline_arb_anime_priority_stage512a_weighted_v1_20260722/header.json":
        "93e23f11ae58acdef4830f6c24d1824e52ec686f52db0a7a8361127e68e4c536",
    DATA_ROOT / "offline_arb_anime_priority_stage512a_weighted_v1_20260722/anime_priority.parquet":
        "0a86f7195a8c87975a37414010c1ead52e6c626d05b115adeeac862f93f5ed56",
    DATA_ROOT / "offline_arb_animefreq_stage512a_v1/header.json":
        "4c2cc28f7b161b9f2a037d66c8f54405521f542ea81dfa1b0198e697cf7c15d4",
    DATA_ROOT / "offline_arb_caption_sidecar_stage512a_v1/header.json":
        "72fe0c4d5b5258a1a6320a6ab57f422cd815dd26a8a4a29594fefdd7abfa7224",
    EXPECTED_BLOB / "manifest.json": EXPECTED_BLOB_MANIFEST_SHA,
    EXPECTED_PLAN / "header.json": EXPECTED_PLAN_HEADER_SHA,
    EXPECTED_PLAN / "checksums.json": EXPECTED_PLAN_CHECKSUMS_SHA,
    PLAN_AUDIT_ROOT / "CANONICAL_PLAN_RECEIPT.json":
        EXPECTED_CANONICAL_PLAN_RECEIPT_SHA,
    PLAN_AUDIT_ROOT / "CLOSURE_REPORT.json": EXPECTED_PLAN_CLOSURE_SHA,
    PLAN_AUDIT_ROOT / "_SUCCESS":
        "0f83e65b88dc2c138c36cc1a80ddb84c937c8fca899d7bbf1c90d60b6c9a285b",
    ASSET_REPORT: EXPECTED_ASSET_REPORT_SHA,
    ASSET_REPORT.parent / "_SUCCESS":
        "179538528f0b72c23761ca0bde36893260b2c81419fb7b48af2a109236c6f265",
    EXPECTED_BLOB / "dense_index.bin": EXPECTED_GPIC_DENSE_INDEX_SHA,
    EXPECTED_BLOB / "payload_reuse_proof.json": EXPECTED_GPIC_REUSE_PROOF_SHA,
    EXPECTED_BLOB / "verifier_report.json": EXPECTED_GPIC_VERIFIER_SHA,
    REGISTRY_ROOT / "REGISTRY_CLOSURE.json": EXPECTED_REGISTRY_RECEIPT_SHA,
    REGISTRY_ROOT / "_SUCCESS": EXPECTED_REGISTRY_SUCCESS_SHA,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def strict_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise AssertionError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def strict_json_bytes(raw: bytes, label: str) -> dict:
    value = json.loads(raw, object_pairs_hook=strict_object_pairs)
    if not isinstance(value, dict):
        raise AssertionError(f"JSON authority is not an object: {label}")
    return value


def ordinary_file(path: Path) -> None:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or path.is_symlink() or value.st_size <= 0:
        raise AssertionError(f"not an ordinary nonempty file: {path}")


def sealed_file(path: Path) -> None:
    ordinary_file(path)
    if path.stat().st_mode & 0o222:
        raise AssertionError(f"launch authority must be read-only: {path}")


def load_bound_json(path: Path, expected_sha: str, *, sealed: bool = True) -> dict:
    (sealed_file if sealed else ordinary_file)(path)
    got = sha256_file(path)
    if got != expected_sha:
        raise AssertionError(f"authority SHA drift: {path}: {got} != {expected_sha}")
    return strict_json_bytes(path.read_bytes(), str(path))


def config_leaf_deltas(reference: object, candidate: object, prefix: str = "") -> set[str]:
    if type(reference) is not type(candidate):
        return {prefix}
    if isinstance(reference, dict) and isinstance(candidate, dict):
        result: set[str] = set()
        for key in sorted(set(reference) | set(candidate)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in reference or key not in candidate:
                result.add(path)
            else:
                result |= config_leaf_deltas(reference[key], candidate[key], path)
        return result
    if isinstance(reference, list) and isinstance(candidate, list):
        return set() if reference == candidate else {prefix}
    return set() if reference == candidate else {prefix}


def validate(config_path: Path, contract_path: Path,
             release_manifest_path: Path) -> dict[str, object]:
    ordinary_file(config_path)
    ordinary_file(REFERENCE_CONFIG)
    reference_config_sha = sha256_file(REFERENCE_CONFIG)
    assert reference_config_sha == EXPECTED_REFERENCE_CONFIG_SHA, (
        "historical A authority drift: "
        f"{reference_config_sha} != {EXPECTED_REFERENCE_CONFIG_SHA}"
    )
    ordinary_file(contract_path)
    contract_sha = sha256_file(contract_path)
    assert contract_sha == EXPECTED_CONTRACT_SHA
    for path, expected in SMALL_FILE_HASHES.items():
        ordinary_file(path)
        got = sha256_file(path)
        assert got == expected, f"asset SHA drift: {path}: {got} != {expected}"
    for path in (
        PLAN_AUDIT_ROOT / "CANONICAL_PLAN_RECEIPT.json",
        PLAN_AUDIT_ROOT / "CLOSURE_REPORT.json",
        PLAN_AUDIT_ROOT / "_SUCCESS",
        ASSET_REPORT,
        ASSET_REPORT.parent / "_SUCCESS",
        REGISTRY_ROOT / "REGISTRY_CLOSURE.json",
        REGISTRY_ROOT / "_SUCCESS",
    ):
        sealed_file(path)

    cfg = OmegaConf.load(config_path)
    reference = OmegaConf.to_container(OmegaConf.load(REFERENCE_CONFIG), resolve=True)
    candidate = OmegaConf.to_container(cfg, resolve=True)
    deltas = config_leaf_deltas(reference, candidate)
    assert deltas == ALLOWED_CONFIG_DELTAS, (
        "scientific-control delta set drift: "
        f"actual={sorted(deltas)}, expected={sorted(ALLOWED_CONFIG_DELTAS)}"
    )
    d = cfg.dataset
    t = cfg.trainer
    assert str(d.stage) == "512a"
    assert Path(str(d.sampling_config)) == EXPECTED_SAMPLING
    assert Path(str(d.offline_catalog_root)) == EXPECTED_CATALOG
    assert Path(str(d.offline_plan_root)) == EXPECTED_PLAN
    assert str(d.offline_plan_hash) == EXPECTED_PLAN_HASH
    assert str(d.offline_weighted_contract_sha256) == EXPECTED_CONTRACT_SHA
    assert Path(str(d.offline_anime_freq_root)) == DATA_ROOT / "offline_arb_animefreq_stage512a_v1"
    assert Path(str(d.offline_caption_sidecar_root)) == DATA_ROOT / "offline_arb_caption_sidecar_stage512a_v1"
    assert "offline_textatlas_gt_root" not in d

    plan_reader = WeightedOfflinePlan(
        EXPECTED_PLAN,
        world_size=64,
        per_rank_bs=32,
        expected_plan_hash=str(d.offline_plan_hash),
        expected_contract_sha256=EXPECTED_CONTRACT_SHA,
        verify_checksums=True,
    )
    plan = plan_reader.header
    old_plan = strict_json_bytes(
        (OLD_CONSUMED_PLAN / "header.json").read_bytes(), "old A plan header"
    )
    catalog = strict_json_bytes(
        (EXPECTED_CATALOG / "header.json").read_bytes(), "catalog header"
    )
    assert catalog["catalog_hash"] == EXPECTED_CATALOG_HASH
    assert plan["plan_id"] == EXPECTED_PLAN_ID
    assert plan["plan_hash"] == EXPECTED_PLAN_HASH
    assert plan["plan_hash"] != OLD_CONSUMED_PLAN_HASH
    assert plan["upstream_catalog_hash"] == EXPECTED_CATALOG_HASH
    assert plan["weighted_contract_sha256"] == EXPECTED_CONTRACT_SHA
    for key, expected_sha in (
        ("S", EXPECTED_S_SHA), ("C", EXPECTED_C_SHA), ("D", EXPECTED_D_SHA),
    ):
        assert plan[key] == old_plan[key], f"old/new {key} control invariant drift"
        assert canonical_json_sha(plan[key]) == expected_sha
    real_band = plan["materialization"]["realized_real_band_exposures"]
    assert real_band == old_plan["materialization"]["realized_real_band_exposures"]
    assert real_band == [24_422_401, 36_633_599, 20_352_002, 54_271_998]
    assert canonical_json_sha(real_band) == EXPECTED_REAL_BAND_SHA
    pi = plan["planner_input"]
    assert int(pi["seed"]) == 20260710
    assert int(pi["total_steps"]) == 125_000
    assert int(pi["world_size"]) == 64 and int(pi["per_rank_bs"]) == 32
    assert pi["domain_mix"] == {
        "anime": "0.37",
        "monet_real": "0.4",
        "gpic": "0.13",
        "synthetic": "0.05",
        "textatlas": "0.05",
    }
    assert pi["square_only_domains"] == ["synthetic", "textatlas"]
    assert list(plan["materialization"]["realized_domain_exposures"]) == [
        94_720_000,
        102_400_000,
        33_280_000,
        12_800_000,
        12_800_000,
    ]

    canonical = load_bound_json(
        PLAN_AUDIT_ROOT / "CANONICAL_PLAN_RECEIPT.json",
        EXPECTED_CANONICAL_PLAN_RECEIPT_SHA,
    )
    assert canonical["schema"] == "stage512a_continue_control_plan_canonical_receipt_v1"
    assert canonical["status"] == "PASS"
    assert canonical["canonical_plan_root"] == str(EXPECTED_PLAN)
    assert canonical["plan_id"] == EXPECTED_PLAN_ID
    assert canonical["plan_hash"] == EXPECTED_PLAN_HASH
    assert canonical["header_sha256"] == EXPECTED_PLAN_HEADER_SHA
    assert canonical["checksums_sha256"] == EXPECTED_PLAN_CHECKSUMS_SHA
    assert canonical["weighted_contract_sha256"] == EXPECTED_CONTRACT_SHA
    assert bool(canonical["deterministic_double_build"])
    assert bool(canonical["runtime_reader_all_checksums"])
    assert int(canonical["total_steps"]) == 125_000
    assert int(canonical["total_exposures"]) == 256_000_000
    assert int(canonical["world_size"]) == 64
    assert int(canonical["per_rank_bs"]) == 32
    assert canonical["full_256m_closure"]["sha256"] == EXPECTED_PLAN_CLOSURE_SHA

    closure = load_bound_json(
        PLAN_AUDIT_ROOT / "CLOSURE_REPORT.json", EXPECTED_PLAN_CLOSURE_SHA
    )
    assert closure["schema"] == "stage512a_continue_control_plan_closure_v1"
    assert closure["status"] == "PASS"
    assert closure["plan_root"] == str(EXPECTED_PLAN)
    assert closure["plan_id"] == EXPECTED_PLAN_ID
    assert closure["plan_hash"] == EXPECTED_PLAN_HASH
    assert closure["header_sha256"] == EXPECTED_PLAN_HEADER_SHA
    assert closure["checksums_sha256"] == EXPECTED_PLAN_CHECKSUMS_SHA
    assert closure["kept_bucket_count"] == 30
    assert {
        int(row): int(value["realized_exposure_count"])
        for row, value in closure["textatlas_jpeg_repairs"].items()
    } == {68_973_396: 3, 69_010_633: 3, 70_063_978: 3}
    anime = closure["anime_native_upsample"]["11163647"]
    assert int(anime["realized_exposure_count"]) == 6
    assert anime["realized_coordinates"] == [
        [11_486, 35, 1], [50_429, 7, 2], [60_035, 46, 0],
        [95_244, 16, 1], [119_964, 33, 0], [121_904, 58, 1],
    ]
    assert {
        int(row): int(value["realized_exposure_count"])
        for row, value in closure["captionless_monet_zero_hit_rows"].items()
    } == {
        19_760_606: 0, 22_783_601: 0, 29_778_338: 0,
        29_779_948: 0, 30_281_293: 0, 35_574_104: 0,
    }

    mar = d.multi_ar
    assert str(mar.assignment_mode) == "offline_weighted_band_v1"
    assert str(mar.bucket_mode) == "adaptive"
    assert float(mar.bucket_prob_floor) == 0.005
    assert list(mar.square_only_domains) == ["synthetic", "textatlas"]
    assert int(mar.target_area) == 262_144 and int(mar.divisor) == 16
    assert bool(mar.adaptive_no_upscale)
    assert float(mar.adaptive_min_area_ratio) == 0.75
    assert float(mar.adaptive_max_aspect) == 4.0

    assert str(d.gpic_payload_backend) == "offline_gpic_blob_v1"
    assert Path(str(d.offline_gpic_blob_root)) == EXPECTED_BLOB
    assert str(d.gpic_blob_manifest_sha256) == EXPECTED_BLOB_MANIFEST_SHA
    assert int(d.gpic_threads_per_node) == 64
    assert int(d.gpic_blob_fd_cache_size) == 680
    blob = strict_json_bytes(
        (EXPECTED_BLOB / "manifest.json").read_bytes(), "GPIC manifest"
    )
    assert blob["release"]["catalog"]["catalog_hash"] == EXPECTED_CATALOG_HASH
    assert int(blob["release"]["row_count"]) == 4_239_912

    reuse = load_bound_json(
        EXPECTED_BLOB / "payload_reuse_proof.json",
        EXPECTED_GPIC_REUSE_PROOF_SHA,
        sealed=False,
    )
    assert reuse["schema_version"] == "offline_gpic_payload_reuse_proof_v1"
    assert reuse["status"] == "PASS"
    assert reuse["target_catalog"]["catalog_hash"] == EXPECTED_CATALOG_HASH
    assert int(reuse["target_row_count"]) == 4_239_912
    for key in (
        "duplicate_source_image_path", "duplicate_source_uid",
        "duplicate_target_image_path", "duplicate_target_uid",
        "image_path_mismatches", "locator_mismatches", "missing_uid",
        "segment_binding_mismatches", "source_index_row_mismatches",
        "target_index_row_mismatches",
    ):
        assert int(reuse[key]) == 0, f"GPIC payload-reuse mismatch: {key}"
    assert reuse["target_dense_index"]["sha256"] == EXPECTED_GPIC_DENSE_INDEX_SHA
    verifier = load_bound_json(
        EXPECTED_BLOB / "verifier_report.json", EXPECTED_GPIC_VERIFIER_SHA,
        sealed=False,
    )
    assert verifier["schema_version"] == "offline_gpic_blob_verify_v1"
    assert verifier["status"] == "PASS"
    assert verifier["catalog_hash"] == EXPECTED_CATALOG_HASH
    assert bool(verifier["full_blob_sha256_verified"])
    assert bool(verifier["full_segment_sha256_verified"])
    assert int(verifier["full_index_records_verified"]) == 4_239_912
    assert int(verifier["row_count"]) == 4_239_912
    assert int(verifier["blob_count"]) == 680
    assert verifier["index_sha256"] == EXPECTED_GPIC_DENSE_INDEX_SHA

    asset = load_bound_json(ASSET_REPORT, EXPECTED_ASSET_REPORT_SHA)
    assert asset["schema"] == "stage512a_continue_asset_rehash_preflight_v2"
    assert asset["status"] == "PASS"
    verdict = asset["launch_blocking_verdict"]
    assert verdict["a_final_full_sha256"] == "PASS"
    assert verdict["asset_integrity"] == "PASS"
    assert verdict["caption_sidecar_full_physical_closure"] == "PASS"
    assert verdict["gpic_release_chain"] == "PASS"
    checkpoint = asset["checkpoint"]
    assert checkpoint["status"] == "PASS"
    assert checkpoint["full_sha256"] == EXPECTED_INIT_SHA
    assert int(checkpoint["global_step"]) == 125_000
    assert checkpoint["raw_ema_optimizer_counts"] == {
        "raw": 483, "ema": 483, "optimizer": 483,
    }
    aliases = [Path(item["path"]) for item in checkpoint["aliases"]]
    assert len(aliases) == 4 and EXPECTED_INIT in aliases
    alias_stats = []
    for alias in aliases:
        ordinary_file(alias)
        value = alias.stat()
        assert not (value.st_mode & 0o222), f"A-final hardlink is writable: {alias}"
        assert value.st_size == 26_328_353_144
        assert value.st_nlink == 4
        alias_stats.append((value.st_dev, value.st_ino))
    assert len(set(alias_stats)) == 1, "A-final four-point hardlink closure drift"
    init_parent = EXPECTED_INIT.parent.stat()
    assert not (init_parent.st_mode & 0o222), "A-final alias parent must be read-only"
    asset_gpic = asset["gpic"]
    assert asset_gpic["status"] == "PASS"
    assert asset_gpic["prior_full_blob_sha256_verified"] is True
    assert asset_gpic["prior_full_segment_sha256_verified"] is True
    assert int(asset_gpic["metadata_blob_count"]) == 680
    assert asset_gpic["metadata_missing"] == []
    assert asset_gpic["metadata_size_mismatch"] == []
    assert asset_gpic["dense_index"]["sha256"] == EXPECTED_GPIC_DENSE_INDEX_SHA
    assert asset_gpic["control_files"]["payload_reuse_proof.json"]["sha256"] == (
        EXPECTED_GPIC_REUSE_PROOF_SHA
    )
    assert asset_gpic["control_files"]["verifier_report.json"]["sha256"] == (
        EXPECTED_GPIC_VERIFIER_SHA
    )

    registry = load_bound_json(
        REGISTRY_ROOT / "REGISTRY_CLOSURE.json", EXPECTED_REGISTRY_RECEIPT_SHA
    )
    assert registry["schema"] == "stage512a_continue_control_registry_closure_v2"
    assert registry["status"] == "PASS"
    assert registry["plan"]["root"] == str(EXPECTED_PLAN)
    assert registry["plan"]["plan_id"] == EXPECTED_PLAN_ID
    assert registry["plan"]["plan_hash"] == EXPECTED_PLAN_HASH
    assert registry["plan"]["header_sha256"] == EXPECTED_PLAN_HEADER_SHA
    assert registry["plan"]["checksums_sha256"] == EXPECTED_PLAN_CHECKSUMS_SHA
    assert registry["plan"]["s_sha256"] == EXPECTED_S_SHA
    assert registry["plan"]["c_sha256"] == EXPECTED_C_SHA
    assert registry["plan"]["d_sha256"] == EXPECTED_D_SHA
    scan = registry["scan"]
    assert bool(scan["all_block_checksums_verified"])
    assert int(scan["verified_blocks"]) == 123
    assert int(scan["scanned_exposures"]) == 256_000_000
    assert int(scan["unserved_real_band_selected"]) == 0
    assert scan["real_band_join_coverage"] == {
        "coverage": "100%", "duplicate_join": 0,
        "joined_exposures": 135_680_000, "missing_join": 0,
        "selected_exposures": 135_680_000,
    }
    assert scan["real_band_exposures"]["combined"] == [
        24_422_401, 36_633_599, 20_352_002, 54_271_998,
    ]
    assert registry["registries"]["image"]["bound_row_ids"] == [
        68_973_396, 69_010_633, 70_063_978,
    ]
    assert registry["registries"]["anime"]["bound_row_ids"] == [11_163_647]
    assert registry["registries"]["anime"]["closure"] == [6, 125_000, 64, 32]
    assert registry["registries"]["caption"]["bound_row_ids"] == []
    negative = registry["registries"]["negative_identity_checks"]
    assert negative["status"] == "PASS" and int(negative["count"]) == 27
    for relative, expected_sha in registry["source_authorities"]["files"].items():
        source = Path(relative)
        ordinary_file(source)
        assert sha256_file(source) == expected_sha, f"registry source drift: {relative}"
    assert set(registry["source_authorities"]["files"]) == {
        "configs/stage512a_continue_control_registry_closure_v2_policy.json",
        "provenance/jcc_jobs/20260809_stage512a_control_registry_closure_v2.yaml",
        "scripts/offline_arb/audit_stage512a_control_registry_closure_v2.py",
        "scripts/offline_arb/immutable_review_evidence.py",
        "training/dataline_offline/anime_native_upsample_exceptions_control_rngfix_v1.py",
        "training/dataline_offline/caption_exceptions_v1.py",
        "training/dataline_offline/image_exceptions_v1.py",
        "training/dataline_offline/offline_dataset.py",
        "training/dataline_offline/stage512a_control_registry_closure_v1.py",
    }
    from training.dataline_offline.stage512a_control_registry_closure_v1 import (
        validate_runtime_bindings,
        validate_stage512a_control_registry_closure,
    )
    validate_stage512a_control_registry_closure()
    validate_runtime_bindings()

    release_manifest = load_bound_json(
        release_manifest_path, sha256_file(release_manifest_path)
    )
    assert release_manifest["schema"] == "stage512a_control_resolved_release_v1"
    assert release_manifest["status"] == "PASS"
    fixed = release_manifest["fixed_identities"]
    assert fixed["plan_hash"] == EXPECTED_PLAN_HASH
    assert fixed["plan_header_sha256"] == EXPECTED_PLAN_HEADER_SHA
    assert fixed["plan_checksums_sha256"] == EXPECTED_PLAN_CHECKSUMS_SHA
    assert fixed["reference_config_sha256"] == EXPECTED_REFERENCE_CONFIG_SHA
    assert fixed["canonical_plan_receipt_sha256"] == EXPECTED_CANONICAL_PLAN_RECEIPT_SHA
    assert fixed["plan_closure_sha256"] == EXPECTED_PLAN_CLOSURE_SHA
    assert fixed["asset_report_sha256"] == EXPECTED_ASSET_REPORT_SHA
    assert release_manifest["evidence"]["registry_closure"]["sha256"] == (
        EXPECTED_REGISTRY_RECEIPT_SHA
    )

    assert float(cfg.model.cfg_dropout_prob) == 0.1
    assert cfg.model.rank_independent_seed is False
    assert float(cfg.model.text_rich_cfg_dropout_prob) == 0.0
    assert bool(cfg.model.text_adapter.enabled)
    assert int(cfg.model.text_adapter.num_layers) == 2
    assert not bool(cfg.model.text_adapter.bidirectional)
    assert bool(cfg.model.text_adapter.multi_layer_fusion.enabled)
    assert int(cfg.model.text_adapter.multi_layer_fusion.num_layers) == 12
    assert not bool(cfg.model.long_skip.enabled)
    assert float(cfg.sigma_sampling.shift) == 1.0
    assert float(cfg.optimizer.params.lr) == 1.732051e-4
    assert list(cfg.optimizer.params.betas) == [0.9, 0.95]
    assert float(cfg.optimizer.params.weight_decay) == 0.0
    assert int(cfg.scheduler.num_warmup_steps) == 500
    assert str(cfg.scheduler.decay) == "constant"

    assert int(t.seed) == 42 and int(t.batch_size) == 32
    assert t.formal_launch_required is True
    assert int(t.max_steps) == 125_000
    assert int(t.max_exposure_images) == 256_000_000
    assert Path(str(t.checkpoint_dir)) == EXPECTED_CKPT
    assert bool(t.resume) and bool(t.init_load_optimizer)
    assert Path(str(t.init_from)) == EXPECTED_INIT
    assert str(t.init_from_rng_policy) == "legacy_missing_rng"
    assert int(t.checkpoint_steps) == 1_500
    assert int(t.checkpoint_keep_every) == 30_000
    assert int(t.checkpoint_keep_max) == 4
    assert str(t.checkpoint_keep_format) == "full"
    assert bool(t.ema.enabled) and float(t.ema.decay) == 0.9999
    assert bool(t.gc_pause_probe)
    snapshots = list(range(int(t.checkpoint_keep_every), int(t.max_steps) + 1,
                           int(t.checkpoint_keep_every)))
    assert snapshots == [30_000, 60_000, 90_000, 120_000]
    assert int(t.checkpoint_keep_every) % int(t.checkpoint_steps) == 0
    assert len(snapshots) <= int(t.checkpoint_keep_max)

    return {
        "status": "PASS",
        "schema": "stage512a_continue_control_rngfix_preflight_v1",
        "config_sha256": sha256_file(config_path),
        "catalog_hash": EXPECTED_CATALOG_HASH,
        "plan_id": plan["plan_id"],
        "plan_hash": plan["plan_hash"],
        "plan_header_sha256": EXPECTED_PLAN_HEADER_SHA,
        "plan_checksums_sha256": EXPECTED_PLAN_CHECKSUMS_SHA,
        "canonical_plan_receipt_sha256": EXPECTED_CANONICAL_PLAN_RECEIPT_SHA,
        "plan_closure_sha256": EXPECTED_PLAN_CLOSURE_SHA,
        "S_sha256": EXPECTED_S_SHA,
        "C_sha256": EXPECTED_C_SHA,
        "D_sha256": EXPECTED_D_SHA,
        "realized_real_band_exposures_sha256": EXPECTED_REAL_BAND_SHA,
        "asset_report_sha256": EXPECTED_ASSET_REPORT_SHA,
        "reference_config_sha256": EXPECTED_REFERENCE_CONFIG_SHA,
        "contract_sha256": contract_sha,
        "gpic_manifest_sha256": EXPECTED_BLOB_MANIFEST_SHA,
        "gpic_dense_index_sha256": EXPECTED_GPIC_DENSE_INDEX_SHA,
        "gpic_payload_reuse_proof_sha256": EXPECTED_GPIC_REUSE_PROOF_SHA,
        "gpic_verifier_report_sha256": EXPECTED_GPIC_VERIFIER_SHA,
        "registry_closure_v2_sha256": EXPECTED_REGISTRY_RECEIPT_SHA,
        "registry_closure_v2_success_sha256": EXPECTED_REGISTRY_SUCCESS_SHA,
        "checkpoint_dir": str(EXPECTED_CKPT),
        "init_from": str(EXPECTED_INIT),
        "init_from_expected_sha256": EXPECTED_INIT_SHA,
        "permanent_full_state_steps": snapshots,
        "canonical_config_delta_allowlist": sorted(ALLOWED_CONFIG_DELTAS),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "provenance/imported_control_scripts/weighted_contract_stage512a_v1.json"
        ),
    )
    parser.add_argument("--release-manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate(args.config, args.contract, args.release_manifest)
    except Exception as exc:  # noqa: BLE001 - CLI must turn every drift into a hard fail.
        print(f"STAGE512A_CONTROL_PREFLIGHT_FAIL {type(exc).__name__}: {exc}")
        return 1
    print("STAGE512A_CONTROL_PREFLIGHT_PASS " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
