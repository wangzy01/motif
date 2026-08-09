#!/usr/bin/env python3
"""Create an immutable code worktree and externally resolved FORMAL bundle.

The tracked JCC YAML is a template.  It deliberately cannot contain the commit
that contains itself.  This renderer runs *after* the final commit and copies
all external launch gates into a create-once, read-only bundle.  The submitted
YAML and its manifest therefore have stable, non-self-referential identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any


PLACEHOLDERS = {
    "__FORMAL_REPO__",
    "__EXPECTED_COMMIT__",
    "__EXPECTED_CONFIG_SHA256__",
    "__RESOLVED_SPEC_PATH__",
    "__RELEASE_MANIFEST_PATH__",
    "__TRAINING_LINEAGE_PATH__",
    "__TRAINING_LINEAGE_SHA256__",
}
EVIDENCE_KEYS = (
    "registry_closure", "rng_gpuval", "a_final_load_smoke", "b120_handoff",
)

FIXED_IDENTITIES = {
    "plan_id": "stage512a_offline_exact_v2_continue_control_rngfix_v1_w64_256m_seed20260710",
    "plan_root": "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_plan_stage512a_continue_control_rngfix_v1_w64_256m_seed20260710_20260809",
    "plan_hash": "fbb5fcbd91d80daff8dfe89693f38eb3fa6ddeb657d01198df7fcac9cdcd039c",
    "plan_header_sha256": "5fbe7ca546d3e77ef37d95bf52c7ded2f9741d0b4377fbadb8b059e9b9a1dce0",
    "plan_checksums_sha256": "aa8283436369f3ddc2f2c7648c28f56cb8b8619b2e87b660f7b2ca29847fc8f0",
    "plan_block_manifest_sha256": "6fd84c224493daefc990bfb661966b748433ee0cb631d1cf4bb6a3b8809ff78c",
    "catalog_hash": "370b1679e8c5670338a03ba4c6972cdbf69fa5068fc35108bf646b2d64642331",
    "catalog_header_sha256": "51a5f0904987a751b54842633d2968ea7f6485307850858d9472319c15748d1d",
    "weighted_contract_sha256": "c228f8b8400c4f96033431816b650d6e7093b3bf8e164018d22e694c60b96c38",
    "reference_config_sha256": "611635fc13b04c6d62c20875eff541736afbe6e35cc17c1f81caafb7f823a380",
    "canonical_plan_receipt_sha256": "5178207100c8af20230489dac88b9118665c93abb33ab5c97b6ccc532b2ea6ca",
    "plan_closure_sha256": "cb92872fc8f13cae10bfafaeb3238c9cc0159bf0c31a096e85ee9b188eb53104",
    "asset_report_sha256": "71c69d5dd56e5e1ac09ca7f3fbd9f380d6ce3bdbc5e328d3a7111a8da675041a",
    "init_from_sha256": "957bfcf12b89c4042a32056566012a2451e72aa0b2706b0ca51808ab79155c31",
    "gpic_manifest_sha256": "c9365753356f942caca1ea622c788ae5d05db320ecfa5804595b25ac8af3a648",
    "gpic_dense_index_sha256": "37f1a2bca7a7236c13f0d757547175a654647ea2f81de526a553f113bac7bded",
    "gpic_payload_reuse_proof_sha256": "fac2e0bb040be6c393d5934309d4b2b90d809dd8de7f0b71554487dec2736dc8",
    "gpic_verifier_report_sha256": "75cc61664107e45371d452eda60b10ff4123c907a5e1997dfbcacb1f63417f69",
    "registry_closure_v2_sha256": "5eb979532d44f2abeaf91d5640f883b19ec2b2d8eb8810819f66fad77b521451",
    "registry_closure_v2_success_sha256": "1f5766e38ce5476ea6b26655808f0934872a8cdda104026d092ba95979bf2425",
    "registry_policy_sha256": "2fe5fe43a758a0a85eb9ac42dc48999822489f3423dc212916206f56a284e08a",
    "registry_verifier_sha256": "4935b4d84ca4e4efdb88c938b9246e1e6dbe86ce471f1d05d96f32fd7e378ba1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ordinary_readonly(path: Path) -> None:
    value = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise RuntimeError(f"not an ordinary nonempty authority: {path}")
    if value.st_mode & 0o222:
        raise RuntimeError(f"launch authority must be read-only: {path}")


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def atomic_create(path: Path, data: bytes, mode: int = 0o444) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def freeze_release(source_repo: Path, release_repo: Path, commit: str) -> None:
    if release_repo.exists():
        raise RuntimeError(f"create-once release path already exists: {release_repo}")
    if run("git", "rev-parse", "HEAD", cwd=source_repo) != commit:
        raise RuntimeError("source HEAD differs from requested release commit")
    if run("git", "status", "--porcelain=v1", "--untracked-files=all", cwd=source_repo):
        raise RuntimeError("source worktree is not clean")
    release_repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["git", "worktree", "add", "--detach", str(release_repo), commit],
        cwd=source_repo,
    )
    if run("git", "rev-parse", "HEAD", cwd=release_repo) != commit:
        raise RuntimeError("release worktree commit mismatch")
    if run("git", "status", "--porcelain=v1", "--untracked-files=all", cwd=release_repo):
        raise RuntimeError("new release worktree is not clean")
    for root, dirs, files in os.walk(release_repo):
        for name in files:
            path = Path(root) / name
            os.chmod(path, path.stat().st_mode & ~0o222)
        for name in dirs:
            path = Path(root) / name
            os.chmod(path, path.stat().st_mode & ~0o222)
    os.chmod(release_repo, release_repo.stat().st_mode & ~0o222)


def parse_evidence(values: list[str]) -> dict[str, tuple[Path, str]]:
    parsed: dict[str, tuple[Path, str]] = {}
    for value in values:
        try:
            key, raw_path, expected_sha = value.split("=", 2)
        except ValueError as exc:
            raise RuntimeError("--evidence must be KEY=PATH=SHA256") from exc
        if key in parsed or key not in EVIDENCE_KEYS:
            raise RuntimeError(f"unexpected/duplicate evidence key: {key}")
        if len(expected_sha) != 64:
            raise RuntimeError(f"malformed evidence SHA: {key}")
        parsed[key] = (Path(raw_path), expected_sha)
    if set(parsed) != set(EVIDENCE_KEYS):
        raise RuntimeError(f"evidence set must be exactly {EVIDENCE_KEYS}")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--release-repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--evidence", action="append", default=[])
    args = parser.parse_args()
    if len(args.commit) != 40 or any(ch not in "0123456789abcdef" for ch in args.commit):
        raise RuntimeError("--commit must be a full lowercase git SHA")
    template_rel = Path(
        "provenance/jcc_jobs/20260809_stage512a_continue_control_rngfix_FORMAL_w64.yaml"
    )
    template = args.source_repo / template_rel
    if args.bundle_dir.exists():
        raise RuntimeError(f"create-once bundle already exists: {args.bundle_dir}")
    evidence = parse_evidence(args.evidence)
    for key, (path, expected_sha) in evidence.items():
        ordinary_readonly(path)
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f"evidence SHA mismatch: {key}")
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or value.get("status") != "PASS":
            raise RuntimeError(f"evidence is not a top-level PASS receipt: {key}")

    freeze_release(args.source_repo, args.release_repo, args.commit)
    template = args.release_repo / template_rel
    config = args.release_repo / "configs/stage512a_continue_control_rngfix_v1.yaml"
    config_sha = sha256_file(config)
    tree = run("git", "rev-parse", "HEAD^{tree}", cwd=args.release_repo)

    args.bundle_dir.mkdir(parents=True)
    evidence_dir = args.bundle_dir / "evidence"
    evidence_dir.mkdir()
    frozen_evidence: dict[str, dict[str, str]] = {}
    for key, (source, expected_sha) in evidence.items():
        destination = evidence_dir / f"{key}.json"
        atomic_create(destination, source.read_bytes())
        if sha256_file(destination) != expected_sha:
            raise RuntimeError(f"frozen evidence copy mismatch: {key}")
        frozen_evidence[key] = {"path": str(destination), "sha256": expected_sha}
    os.chmod(evidence_dir, 0o555)

    lineage_path = args.bundle_dir / "TRAINING_LINEAGE.json"
    lineage: dict[str, Any] = {
        "schema": "stage512a_control_training_lineage_v1",
        "status": "PASS",
        "run_id": "stage512a_continue_control_rngfix_v1",
        "release_commit": args.commit,
        "release_tree": tree,
        "config_sha256": config_sha,
        "checkpoint_dir": (
            "/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/"
            "stage512a_continue_control_rngfix_v1"
        ),
        "init_from": (
            "/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/"
            "stage512a_continue_control_rngfix_v1_inputs/a_final_step125000.last.pt"
        ),
        "world_size": 64,
        "per_rank_batch": 32,
        "global_batch": 2048,
        "fixed_identities": FIXED_IDENTITIES,
        "evidence": frozen_evidence,
        "rng_stream_contract": {
            "schema": "mot_rng_state_v1",
            "rank_independent_seed": False,
            "attempt_identity_excluded": True,
        },
    }
    atomic_create(
        lineage_path,
        (json.dumps(lineage, sort_keys=True, indent=1) + "\n").encode(),
    )
    lineage_sha = sha256_file(lineage_path)

    spec_path = args.bundle_dir / "resolved_FORMAL_w64.yaml"
    manifest_path = args.bundle_dir / "RELEASE_MANIFEST.json"
    substitutions = {
        "__FORMAL_REPO__": str(args.release_repo),
        "__EXPECTED_COMMIT__": args.commit,
        "__EXPECTED_CONFIG_SHA256__": config_sha,
        "__RESOLVED_SPEC_PATH__": str(spec_path),
        "__RELEASE_MANIFEST_PATH__": str(manifest_path),
        "__TRAINING_LINEAGE_PATH__": str(lineage_path),
        "__TRAINING_LINEAGE_SHA256__": lineage_sha,
    }
    rendered = template.read_text()
    for placeholder, replacement in substitutions.items():
        if rendered.count(placeholder) != 1:
            raise RuntimeError(f"template must contain exactly one {placeholder}")
        rendered = rendered.replace(placeholder, replacement)
    if any(item in rendered for item in PLACEHOLDERS) or "__FORMAL_" in rendered:
        raise RuntimeError("resolved spec retains a launch placeholder")
    atomic_create(spec_path, rendered.encode())

    manifest = {
        "schema": "stage512a_control_resolved_release_v1",
        "status": "PASS",
        "release_repo": str(args.release_repo),
        "commit": args.commit,
        "tree": tree,
        "template": {"path": str(template), "sha256": sha256_file(template)},
        "config": {"path": str(config), "sha256": config_sha},
        "resolved_spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "training_lineage": {"path": str(lineage_path), "sha256": lineage_sha},
        "fixed_identities": FIXED_IDENTITIES,
        "evidence": frozen_evidence,
    }
    atomic_create(
        manifest_path,
        (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode(),
    )
    os.chmod(args.bundle_dir, 0o555)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
