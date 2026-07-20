#!/usr/bin/env python3
"""Create or verify the immutable 7.18 Stage256 semantic-data release.

The manifest closes every new large artifact and every new runtime source file.
It transitively binds the already-published physical catalog/sidecar/source
release through its independently verified upstream freeze.  GPIC data keeps
its canonical immutable buildtag/manifest identity and is additionally checked
per node by the existing compact preflight in GPU jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


SCHEMA = "stage256_20260718_semantic_release_freeze_v1"
RUN_ID = "stage256_20260718_causalmlf_offline_exact_v1"
ARCH_FP = "70546507a768a9d01705b8118b6e3cfea3f055f22db5e9bcdfb3e25e7d022642"
EXACT_STEPS = 166_667
EXACT_EXPOSURES = 384_000_768
EXPECTED_GPICSHA = "a18ba1f1250e3b00abdc1051dc71e08a254b8855d5b61b9975e88d2e6ae2ee05"
EXPECTED_BUILD_TAG = "evan-mot-crop-offlinearb-gpic-s3-production-build-zg8g"
EXPECTED_NATIVE_BUCKET_INDICES = [
    25, 27, 29, 31, 33, 35, 38, 40, 45, 48,
    49, 54, 56, 59, 61, 63, 65, 67, 69, 71,
]
EXPECTED_DOMAIN_TOTALS = {
    "anime": 103_680_207,
    "monet_real": 155_520_311,
    "gpic": 99_456_199,
    "synthetic": 19_584_039,
    "textatlas": 5_760_012,
}
SHA_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace create-once output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary output collision: {temporary}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def regular_file(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"required path is not a regular non-symlink file: {path}")
    return resolved


def regular_tree(path: Path) -> tuple[Path, list[Path]]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_dir():
        raise ValueError(f"required path is not a regular non-symlink directory: {path}")
    files: list[Path] = []
    for child in sorted(resolved.rglob("*")):
        if child.is_symlink():
            raise ValueError(f"symlink forbidden in frozen tree: {child}")
        if child.is_file():
            files.append(child)
        elif not child.is_dir():
            raise ValueError(f"non-file object forbidden in frozen tree: {child}")
    if not files:
        raise ValueError(f"frozen tree is empty: {resolved}")
    return resolved, files


def dataset_source_files(path: Path) -> tuple[Path, list[Path]]:
    """Require and freeze exactly the three authoritative source files."""

    root = path.resolve(strict=True)
    if path.is_symlink() or not root.is_dir():
        raise ValueError(f"dataset package is not a regular directory: {path}")
    expected = {"__init__.py", "semantic_plan.py", "offline_semantic_dataset.py"}
    actual_sources: set[str] = set()
    for child in root.iterdir():
        if child.is_symlink():
            raise ValueError(f"symlink forbidden in dataset package: {child}")
        if not child.is_file() or child.name not in expected:
            raise ValueError(f"unexpected dataset-package source/object: {child}")
        actual_sources.add(child.name)
    if actual_sources != expected:
        raise ValueError(
            f"dataset-package source set drift: missing={sorted(expected-actual_sources)}, "
            f"extra={sorted(actual_sources-expected)}"
        )
    return root, [root / name for name in sorted(expected)]


def file_entry(path: Path) -> dict[str, Any]:
    target = regular_file(path)
    stat = target.stat()
    return {
        "path": str(target),
        "bytes": stat.st_size,
        "sha256": sha256_file(target),
    }


def tree_entry(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root, files = regular_tree(path)
    entries = []
    for child in files:
        stat = child.stat()
        entries.append(
            {
                "relative": child.relative_to(root).as_posix(),
                "bytes": stat.st_size,
                "sha256": sha256_file(child),
            }
        )
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"root": str(root), "file_count": len(entries), "manifest_sha256": digest}, entries


def require_pass(path: Path) -> dict[str, Any]:
    value = read_json(regular_file(path))
    if value.get("status") != "PASS":
        raise ValueError(f"required PASS receipt is not PASS: {path}")
    return value


def validate_control_contract(
    release_root: Path, config: Path, formal_plan: Path, upstream_verify: Path
) -> dict[str, Any]:
    config_value = yaml.safe_load(regular_file(config).read_text(encoding="utf-8"))
    if "init_from" in config_value.get("trainer", {}):
        raise ValueError("fresh config contains trainer.init_from")
    if int(config_value["trainer"]["max_exposure_images"]) != EXACT_EXPOSURES:
        raise ValueError("config exact exposure drift")
    if config_value["dataset"]["offline_plan_root"] != str(formal_plan.resolve()):
        raise ValueError("config formal semantic-plan root drift")
    if config_value["dataset"]["offline_semantic_overlay_root"] != str(
        (release_root / "overlay").resolve()
    ):
        raise ValueError("config overlay root drift")

    overlay_header = read_json(regular_file(release_root / "overlay" / "header.json"))
    if overlay_header.get("semantic_assertions") != {
        "anime_all_rows_strip_nonempty_nl": True,
        "anime_tag_only_rows": 0,
    }:
        raise ValueError("overlay strict-NL/tag-only assertion drift")
    overlay_validation = require_pass(release_root / "OVERLAY_VERIFY_SUCCESS.json")
    if overlay_validation.get("overlay_header_sha256") != sha256_file(
        release_root / "overlay" / "header.json"
    ):
        raise ValueError("overlay validation receipt/header binding drift")

    audit = require_pass(formal_plan / "PLAN_AUDIT_SUCCESS.json")
    if int(audit["formal_prefix"]["steps"]) != EXACT_STEPS:
        raise ValueError("formal audit step drift")
    if int(audit["formal_prefix"]["exposures"]) != EXACT_EXPOSURES:
        raise ValueError("formal audit exact-exposure drift")
    if int(audit["full_stage"]["anime_tag_only_exposures"]) != 0:
        raise ValueError("formal audit has nonzero anime tag-only exposure")
    observed_domain_totals = {
        str(name): int(value)
        for name, value in audit["formal_prefix"]["domain_totals"].items()
    }
    if observed_domain_totals != EXPECTED_DOMAIN_TOTALS:
        raise ValueError(
            f"formal audit domain-total drift: {observed_domain_totals}"
        )
    if sum(observed_domain_totals.values()) != EXACT_EXPOSURES:
        raise ValueError("formal audit domain totals do not close exact exposure")
    if {
        str(name): int(value)
        for name, value in audit["full_stage"]["domain_totals"].items()
    } != observed_domain_totals:
        raise ValueError("full-stage and formal-prefix domain totals differ")
    if int(audit["multi_ar"]["active_native_bucket_count"]) != 20:
        raise ValueError("formal audit active native bucket count drift")
    observed_indices = [
        int(item["bucket_idx"])
        for item in audit["multi_ar"]["active_native_buckets"]
    ]
    if observed_indices != EXPECTED_NATIVE_BUCKET_INDICES:
        raise ValueError(
            f"formal active native bucket indices drift: {observed_indices}"
        )
    plan_header = read_json(regular_file(formal_plan / "header.json"))
    catalog_hash = str(plan_header.get("catalog_hash", ""))
    assignment_hash = str(plan_header.get("assignment_hash", ""))
    if not SHA_RE.fullmatch(catalog_hash) or not SHA_RE.fullmatch(assignment_hash):
        raise ValueError("formal plan catalog/assignment identity is invalid")
    if plan_header.get("upstream_catalog_hash") != catalog_hash:
        raise ValueError("formal plan upstream catalog identity drift")
    if plan_header.get("upstream_assignment_hash") != assignment_hash:
        raise ValueError("formal plan upstream assignment identity drift")
    plan_hashes = plan_header.get("hashes", {})
    if plan_hashes.get("catalog_sha256") != catalog_hash:
        raise ValueError("formal plan hashes.catalog_sha256 drift")
    if plan_hashes.get("assignment_sha256") != assignment_hash:
        raise ValueError("formal plan hashes.assignment_sha256 drift")
    upstream = require_pass(upstream_verify)
    if upstream.get("mode") != "verify-freeze":
        raise ValueError("upstream verification is not verify-freeze mode")
    return {
        "tag_only_exposure": 0,
        "active_native_buckets": 20,
        "active_native_bucket_indices": EXPECTED_NATIVE_BUCKET_INDICES,
        "domain_totals": EXPECTED_DOMAIN_TOTALS,
        "legacy_loader_catalog_identity": {
            "catalog_hash": catalog_hash,
            "assignment_hash": assignment_hash,
        },
        "formal_steps": EXACT_STEPS,
        "formal_exposures": EXACT_EXPOSURES,
    }


def create(args: argparse.Namespace) -> dict[str, Any]:
    release_root = args.release_root.resolve(strict=True)
    if args.gpic_manifest_sha256 != EXPECTED_GPICSHA:
        raise ValueError("GPIC manifest SHA drift")
    if args.gpic_artifact_root.name != EXPECTED_BUILD_TAG:
        raise ValueError("GPIC immutable buildtag drift")
    if not args.gpic_artifact_root.resolve(strict=True).is_dir():
        raise ValueError("GPIC artifact root missing")
    if regular_file(args.upstream_freeze).stat().st_size <= 0:
        raise ValueError("upstream freeze is empty")

    assertions = validate_control_contract(
        release_root, args.config, args.formal_plan_root, args.upstream_verify
    )
    tree_paths = {
        "overlay": args.overlay_root,
        "w8_selection_plan": release_root / "w8_smoke" / "selection_plan",
        "w64_selection_plan": release_root / "w64_prod" / "selection_plan",
        "w8_semantic_plan": args.smoke_plan_root,
        "w64_semantic_plan": args.formal_plan_root,
    }
    trees: dict[str, Any] = {}
    tree_files: dict[str, Any] = {}
    for name, path in tree_paths.items():
        info, entries = tree_entry(path)
        trees[name] = info
        tree_files[name] = entries

    repo_root = args.config.resolve(strict=True).parents[1]
    script_names = (
        "audit_semantic_plan.py",
        "build_stage256_semantic_overlay.py",
        "freeze_stage256_20260718_release.py",
        "make_stage256_semantic_planner_input.py",
        "materialize_semantic_plan.py",
        "validate_stage256_20260718_static.py",
        "validate_stage256_semantic_overlay.py",
        "verify_stage256_20260718_contract.py",
        "verify_stage256_20260718_smoke_checkpoint.py",
    )
    files = {
        "config": file_entry(args.config),
        "upstream_freeze": file_entry(args.upstream_freeze),
        "upstream_verify": file_entry(args.upstream_verify),
    }
    dataset_root, dataset_sources = dataset_source_files(args.dataset_package)
    for source in dataset_sources:
        files[f"dataset_source:{source.name}"] = file_entry(source)
    for receipt in (
        "OVERLAY_VERIFY_SUCCESS.json",
        "SELECTION_PLANS_SUCCESS.json",
        "SEMANTIC_MATERIALIZE_SUCCESS.json",
        "SEMANTIC_VALIDATE_SUCCESS.json",
    ):
        files[f"receipt:{receipt}"] = file_entry(release_root / receipt)
    for tag in ("w8_smoke", "w64_prod"):
        for filename in (
            "planner_input.json",
            "planner_input_binding.json",
            "selection_validate_report.json",
        ):
            files[f"selection_control:{tag}/{filename}"] = file_entry(
                release_root / tag / filename
            )
    for name in script_names:
        files[f"script:{name}"] = file_entry(repo_root / "scripts" / "offline_arb" / name)

    manifest = {
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_root": str(release_root),
        "architecture_fingerprint_sha256": ARCH_FP,
        "exact_training": {
            "world_size": 64,
            "per_rank_bs": 36,
            "steps": EXACT_STEPS,
            "exposures": EXACT_EXPOSURES,
        },
        "assertions": assertions,
        "gpic": {
            "artifact_root": str(args.gpic_artifact_root.resolve(strict=True)),
            "buildtag": EXPECTED_BUILD_TAG,
            "manifest_sha256": EXPECTED_GPICSHA,
        },
        "dataset_package_contract": {
            "root": str(dataset_root),
            "sources": [source.name for source in dataset_sources],
            "additional_objects": "forbidden",
        },
        "files": files,
        "trees": trees,
        "tree_files": tree_files,
    }
    atomic_json(args.manifest, manifest)
    manifest_sha = sha256_file(args.manifest)
    receipt = {
        "schema_version": "stage256_20260718_freeze_create_success_v1",
        "status": "PASS",
        "run_id": RUN_ID,
        "freeze_sha256": manifest_sha,
        "tag_only_exposure": 0,
        "formal_steps": EXACT_STEPS,
        "formal_exposures": EXACT_EXPOSURES,
        "file_count": len(files) + sum(info["file_count"] for info in trees.values()),
    }
    atomic_json(args.output, receipt)
    return receipt


def verify_file(entry: dict[str, Any], label: str) -> None:
    path = regular_file(Path(entry["path"]))
    if path.stat().st_size != int(entry["bytes"]):
        raise ValueError(f"frozen file size drift: {label}: {path}")
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise ValueError(f"frozen file SHA drift: {label}: {path}")


def verify_tree(info: dict[str, Any], entries: list[dict[str, Any]], label: str) -> None:
    root, current = regular_tree(Path(info["root"]))
    current_relatives = [item.relative_to(root).as_posix() for item in current]
    expected_relatives = [item["relative"] for item in entries]
    if current_relatives != expected_relatives:
        raise ValueError(f"frozen tree file set drift: {label}")
    if int(info["file_count"]) != len(entries):
        raise ValueError(f"frozen tree file count drift: {label}")
    manifest_sha = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest_sha != info["manifest_sha256"]:
        raise ValueError(f"frozen tree manifest self hash drift: {label}")
    for entry, path in zip(entries, current):
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"frozen tree file size drift: {label}/{entry['relative']}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"frozen tree file SHA drift: {label}/{entry['relative']}")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = regular_file(args.manifest)
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA or manifest.get("run_id") != RUN_ID:
        raise ValueError("freeze schema/run identity drift")
    if manifest.get("architecture_fingerprint_sha256") != ARCH_FP:
        raise ValueError("freeze architecture fingerprint drift")
    if manifest.get("release_root") != str(args.release_root.resolve(strict=True)):
        raise ValueError("freeze release root drift")
    exact = manifest.get("exact_training")
    if exact != {
        "world_size": 64,
        "per_rank_bs": 36,
        "steps": EXACT_STEPS,
        "exposures": EXACT_EXPOSURES,
    }:
        raise ValueError("freeze exact training contract drift")
    gpic = manifest.get("gpic")
    if not isinstance(gpic, dict) or gpic.get("buildtag") != EXPECTED_BUILD_TAG or gpic.get(
        "manifest_sha256"
    ) != EXPECTED_GPICSHA:
        raise ValueError("freeze GPIC contract drift")
    if not Path(str(gpic["artifact_root"])).resolve(strict=True).is_dir():
        raise ValueError("frozen GPIC artifact root missing")
    if manifest.get("assertions", {}).get("tag_only_exposure") != 0:
        raise ValueError("freeze tag-only assertion drift")
    if manifest.get("assertions", {}).get("active_native_bucket_indices") != EXPECTED_NATIVE_BUCKET_INDICES:
        raise ValueError("freeze native bucket-index assertion drift")
    if manifest.get("assertions", {}).get("domain_totals") != EXPECTED_DOMAIN_TOTALS:
        raise ValueError("freeze domain-total assertion drift")
    catalog_identity = manifest.get("assertions", {}).get(
        "legacy_loader_catalog_identity"
    )
    if not isinstance(catalog_identity, dict) or set(catalog_identity) != {
        "catalog_hash",
        "assignment_hash",
    }:
        raise ValueError("freeze legacy-loader catalog identity is missing")
    if not all(SHA_RE.fullmatch(str(value)) for value in catalog_identity.values()):
        raise ValueError("freeze legacy-loader catalog identity is invalid")
    dataset_contract = manifest.get("dataset_package_contract")
    if not isinstance(dataset_contract, dict):
        raise ValueError("freeze dataset package contract missing")
    _dataset_root, dataset_sources = dataset_source_files(Path(dataset_contract["root"]))
    if [source.name for source in dataset_sources] != dataset_contract.get("sources"):
        raise ValueError("freeze dataset source set drift")
    if dataset_contract.get("additional_objects") != "forbidden":
        raise ValueError("freeze dataset additional-object contract drift")

    for label, entry in sorted(manifest["files"].items()):
        verify_file(entry, label)
    for label, info in sorted(manifest["trees"].items()):
        verify_tree(info, manifest["tree_files"][label], label)

    manifest_sha = sha256_file(manifest_path)
    receipt = {
        "schema_version": "stage256_20260718_freeze_verify_success_v1",
        "status": "PASS",
        "run_id": RUN_ID,
        "freeze_sha256": manifest_sha,
        "tag_only_exposure": 0,
        "formal_steps": EXACT_STEPS,
        "formal_exposures": EXACT_EXPOSURES,
        "gpic_manifest_sha256": EXPECTED_GPICSHA,
    }
    atomic_json(args.output, receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-package", type=Path)
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--smoke-plan-root", type=Path)
    parser.add_argument("--formal-plan-root", type=Path)
    parser.add_argument("--upstream-freeze", type=Path)
    parser.add_argument("--upstream-verify", type=Path)
    parser.add_argument("--gpic-artifact-root", type=Path)
    parser.add_argument("--gpic-manifest-sha256")
    args = parser.parse_args(argv)
    if args.create:
        required = (
            "config",
            "dataset_package",
            "overlay_root",
            "smoke_plan_root",
            "formal_plan_root",
            "upstream_freeze",
            "upstream_verify",
            "gpic_artifact_root",
            "gpic_manifest_sha256",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"--create missing: {', '.join(missing)}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = create(args) if args.create else verify(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
