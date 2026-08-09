#!/usr/bin/env python3
"""Render the commit-bound 2-GPU A-final load smoke-v3 JCC spec.

This is deliberately separate from the generic release freezer.  It proves the
exact final Git blobs used by the loader smoke, resolves every template field
once, and emits a sealed receipt that Phase B can validate independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
from pathlib import Path
from typing import Any, NoReturn


SCHEMA = "stage512a_control_a_final_load_smoke_v3_render_v1"
CONFIG_REL = "configs/stage512a_continue_control_rngfix_v1.yaml"
PROBE_REL = "scripts/stage512a_control_a_final_load_smoke_v3.py"
VALIDATOR_REL = "scripts/validate_stage512a_control_a_final_load_smoke_v3.py"
TEMPLATE_REL = (
    "provenance/jcc_jobs/20260809_stage512a_control_a_final_load_smoke_v3.yaml"
)
RENDERER_REL = "scripts/render_stage512a_control_a_final_load_smoke_v3.py"
TEST_REL = "scripts/test_stage512a_control_a_final_load_smoke_v3.py"
RECEIPT_SCHEMA_REL = (
    "provenance/stage512a_control_a_final_load_smoke_v3.schema.json"
)
CORE_RELS = (
    "train.py",
    "training/checkpoint.py",
    "training/rng_state.py",
    "training/formal_boot.py",
    "training/config_schema.py",
    "networks/model_mot.py",
    "networks/vae.py",
)
RUNTIME_RELS = CORE_RELS + (CONFIG_REL, PROBE_REL, VALIDATOR_REL, TEMPLATE_REL)
DEDICATED_RELS = RUNTIME_RELS + (RENDERER_REL, TEST_REL, RECEIPT_SCHEMA_REL)
PLACEHOLDER_RE = re.compile(r"PLACEHOLDER_[A-Z0-9_]+")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class RenderError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise RenderError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_ordinary(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        fail(f"cannot lstat {path}: {exc}")
    require(stat.S_ISREG(metadata.st_mode) and not path.is_symlink(),
            f"expected ordinary non-symlink file: {path}")
    raw = path.read_bytes()
    after = path.lstat()
    require(
        (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while reading: {path}",
    )
    return raw


def sha256_file(path: Path) -> str:
    return sha256_bytes(read_ordinary(path))


def no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            fail(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def strict_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=no_duplicate_object,
            parse_constant=lambda token: fail(f"{label}: forbidden constant {token}"),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        fail(f"{label}: invalid strict JSON: {exc}")
    require(type(value) is dict, f"{label}: root must be an object")
    return value


def git_blob_raw_sha256(repo: Path, commit: str, relative_path: str) -> str:
    spec = f"{commit}:{relative_path}"
    object_id = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--verify", spec], text=True
    ).strip()
    object_type = subprocess.check_output(
        ["git", "-C", str(repo), "cat-file", "-t", object_id], text=True
    ).strip()
    require(object_type == "blob", f"final object is not blob: {relative_path}")
    raw = subprocess.check_output(
        ["git", "-C", str(repo), "cat-file", "blob", object_id]
    )
    return sha256_bytes(raw)


def verify_git_blob_map(
    repo: Path, commit: str, relative_paths: tuple[str, ...]
) -> dict[str, str]:
    require(len(relative_paths) == len(set(relative_paths)), "duplicate source path")
    result: dict[str, str] = {}
    for relative_path in relative_paths:
        runtime_path = repo / relative_path
        blob_sha = git_blob_raw_sha256(repo, commit, relative_path)
        worktree_sha = sha256_file(runtime_path)
        require(
            worktree_sha == blob_sha,
            f"worktree differs from final Git blob: {relative_path}; "
            f"worktree={worktree_sha}, blob={blob_sha}",
        )
        result[relative_path] = blob_sha
    return result


def canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def validate_phase_a(
    *,
    release_repo: Path,
    commit: str,
    phase_a_manifest: Path,
    expected_phase_a_manifest_sha256: str,
    training_lineage: Path,
    expected_training_lineage_sha256: str,
    config_sha256: str,
) -> None:
    manifest_raw = read_ordinary(phase_a_manifest)
    lineage_raw = read_ordinary(training_lineage)
    require(
        sha256_bytes(manifest_raw) == expected_phase_a_manifest_sha256,
        "Phase-A manifest SHA drift",
    )
    require(
        sha256_bytes(lineage_raw) == expected_training_lineage_sha256,
        "training-lineage SHA drift",
    )
    success_path = phase_a_manifest.parent / "_PHASE_A_SUCCESS"
    require(
        read_ordinary(success_path)
        == (expected_phase_a_manifest_sha256 + "\n").encode("ascii"),
        "Phase-A success marker does not bind manifest",
    )
    manifest = strict_json(manifest_raw, "Phase-A manifest")
    require(
        set(manifest)
        == {
            "schema", "status", "release_repo", "commit", "tree", "config",
            "evaluation_policy", "template", "training_lineage",
        },
        "Phase-A manifest key closure drift",
    )
    require(
        manifest["schema"] == "stage512a_control_phase_a_release_v1"
        and manifest["status"] == "PASS"
        and manifest["release_repo"] == str(release_repo)
        and manifest["commit"] == commit,
        "Phase-A release identity drift",
    )
    require(
        manifest["config"]
        == {"path": str(release_repo / CONFIG_REL), "sha256": config_sha256},
        "Phase-A config binding drift",
    )
    require(
        manifest["training_lineage"]
        == {"path": str(training_lineage), "sha256": expected_training_lineage_sha256},
        "Phase-A training-lineage binding drift",
    )
    lineage = strict_json(lineage_raw, "training lineage")
    require(
        lineage.get("schema") == "stage512a_control_training_lineage_v1"
        and lineage.get("status") == "PASS"
        and lineage.get("run_id") == "stage512a_continue_control_rngfix_v1",
        "training-lineage identity drift",
    )
    release = lineage.get("release")
    require(type(release) is dict, "training-lineage release record missing")
    require(
        release.get("repo") == str(release_repo)
        and release.get("commit") == commit
        and release.get("config_path") == str(release_repo / CONFIG_REL)
        and release.get("config_sha256") == config_sha256,
        "training-lineage release binding drift",
    )


def substitutions(
    *,
    release_repo: Path,
    commit: str,
    phase_a_manifest: Path,
    phase_a_manifest_sha256: str,
    training_lineage: Path,
    training_lineage_sha256: str,
    smoke_output: Path,
    source_map: dict[str, str],
) -> dict[str, str]:
    return {
        "PLACEHOLDER_RELEASE_REPO_PATH": str(release_repo),
        "PLACEHOLDER_RELEASE_CONFIG_PATH": str(release_repo / CONFIG_REL),
        "PLACEHOLDER_PHASE_A_MANIFEST_PATH": str(phase_a_manifest),
        "PLACEHOLDER_PHASE_A_MANIFEST_SHA256": phase_a_manifest_sha256,
        "PLACEHOLDER_CANONICAL_TRAINING_LINEAGE_PATH": str(training_lineage),
        "PLACEHOLDER_CANONICAL_TRAINING_LINEAGE_SHA256": training_lineage_sha256,
        "PLACEHOLDER_RELEASE_PROBE_PATH": str(release_repo / PROBE_REL),
        "PLACEHOLDER_PROBE_SHA256": source_map[PROBE_REL],
        "PLACEHOLDER_RELEASE_VALIDATOR_PATH": str(release_repo / VALIDATOR_REL),
        "PLACEHOLDER_VALIDATOR_SHA256": source_map[VALIDATOR_REL],
        "PLACEHOLDER_RELEASE_JOB_TEMPLATE_PATH": str(release_repo / TEMPLATE_REL),
        "PLACEHOLDER_JOB_TEMPLATE_SHA256": source_map[TEMPLATE_REL],
        "PLACEHOLDER_EXPECTED_COMMIT": commit,
        "PLACEHOLDER_TRAIN_PY_GIT_BLOB_RAW_SHA256": source_map["train.py"],
        "PLACEHOLDER_TRAINING_CHECKPOINT_PY_GIT_BLOB_RAW_SHA256": source_map[
            "training/checkpoint.py"
        ],
        "PLACEHOLDER_TRAINING_RNG_STATE_PY_GIT_BLOB_RAW_SHA256": source_map[
            "training/rng_state.py"
        ],
        "PLACEHOLDER_TRAINING_FORMAL_BOOT_PY_GIT_BLOB_RAW_SHA256": source_map[
            "training/formal_boot.py"
        ],
        "PLACEHOLDER_TRAINING_CONFIG_SCHEMA_PY_GIT_BLOB_RAW_SHA256": source_map[
            "training/config_schema.py"
        ],
        "PLACEHOLDER_NETWORKS_MODEL_MOT_PY_GIT_BLOB_RAW_SHA256": source_map[
            "networks/model_mot.py"
        ],
        "PLACEHOLDER_NETWORKS_VAE_PY_GIT_BLOB_RAW_SHA256": source_map[
            "networks/vae.py"
        ],
        "PLACEHOLDER_CONFIG_SHA256": source_map[CONFIG_REL],
        "PLACEHOLDER_CREATE_ONCE_OUTPUT": str(smoke_output),
    }


def render_template(template_raw: bytes, replacements: dict[str, str]) -> bytes:
    try:
        text = template_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(f"job template is not UTF-8: {exc}")
    found = PLACEHOLDER_RE.findall(text)
    require(
        len(found) == len(set(found)) == 22,
        f"job template placeholder occurrence closure drift: {found}",
    )
    require(set(found) == set(replacements), "job template placeholder key drift")
    for placeholder, value in replacements.items():
        require(
            value and "\x00" not in value and "\n" not in value and "\r" not in value,
            f"unsafe replacement for {placeholder}",
        )
        text = text.replace(placeholder, value)
    require(PLACEHOLDER_RE.search(text) is None, "unresolved job placeholder remains")
    return text.encode("utf-8")


def atomic_file(path: Path, raw: bytes, mode: int = 0o444) -> None:
    require(path.is_absolute(), f"output path must be absolute: {path}")
    require(not path.exists() and not path.is_symlink(), f"output exists: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            require(written > 0, f"short write to {path}")
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, mode)
    finally:
        os.close(fd)


def publish_receipt(output: Path, value: dict[str, Any]) -> str:
    require(output.is_absolute(), "render receipt output must be absolute")
    require(not output.exists() and not output.is_symlink(), f"output exists: {output}")
    staging = output.with_name(
        f".{output.name}.staging.{os.getpid()}.{secrets.token_hex(8)}"
    )
    staging.mkdir(mode=0o700, parents=False)
    report_raw = canonical_json_bytes(value)
    report_sha = sha256_bytes(report_raw)
    atomic_file(staging / "REPORT.json", report_raw)
    atomic_file(staging / "_SUCCESS", (report_sha + "\n").encode("ascii"))
    os.chmod(staging, 0o555)
    os.rename(staging, output)
    return report_sha


def render(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.release_repo.resolve()
    require(args.release_repo.is_absolute() and repo == args.release_repo,
            "release repo must be canonical absolute path")
    require(COMMIT_RE.fullmatch(args.expected_commit) is not None,
            "expected commit must be full lowercase 40-hex")
    for value, label in (
        (args.expected_phase_a_manifest_sha256, "Phase-A manifest"),
        (args.expected_training_lineage_sha256, "training lineage"),
    ):
        require(SHA_RE.fullmatch(value) is not None and value != "0" * 64,
                f"{label} SHA must be nonzero lowercase SHA-256")
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    status_lines = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).splitlines()
    require(head == args.expected_commit, "release HEAD drift")
    require(status_lines == [], f"release worktree is not clean: {status_lines[:12]}")
    dedicated_map = verify_git_blob_map(repo, head, DEDICATED_RELS)
    runtime_map = {path: dedicated_map[path] for path in RUNTIME_RELS}
    schema = strict_json(read_ordinary(repo / RECEIPT_SCHEMA_REL), "receipt schema")
    require(
        schema.get("properties", {}).get("schema", {}).get("const") == SCHEMA,
        "receipt schema identity drift",
    )
    validate_phase_a(
        release_repo=repo,
        commit=head,
        phase_a_manifest=args.phase_a_manifest,
        expected_phase_a_manifest_sha256=args.expected_phase_a_manifest_sha256,
        training_lineage=args.training_lineage,
        expected_training_lineage_sha256=args.expected_training_lineage_sha256,
        config_sha256=runtime_map[CONFIG_REL],
    )
    replacements = substitutions(
        release_repo=repo,
        commit=head,
        phase_a_manifest=args.phase_a_manifest,
        phase_a_manifest_sha256=args.expected_phase_a_manifest_sha256,
        training_lineage=args.training_lineage,
        training_lineage_sha256=args.expected_training_lineage_sha256,
        smoke_output=args.smoke_output,
        source_map=runtime_map,
    )
    resolved_raw = render_template(read_ordinary(repo / TEMPLATE_REL), replacements)
    resolved_sha = sha256_bytes(resolved_raw)
    value = {
        "schema": SCHEMA,
        "status": "PASS",
        "release": {
            "repo": str(repo),
            "commit": head,
            "git_status_porcelain": status_lines,
            "config_sha256": runtime_map[CONFIG_REL],
        },
        "phase_a": {
            "manifest_path": str(args.phase_a_manifest),
            "manifest_sha256": args.expected_phase_a_manifest_sha256,
            "training_lineage_path": str(args.training_lineage),
            "training_lineage_sha256": args.expected_training_lineage_sha256,
        },
        "artifacts": {
            "template_path": str(repo / TEMPLATE_REL),
            "template_sha256": runtime_map[TEMPLATE_REL],
            "resolved_spec_path": str(args.resolved_spec_output),
            "resolved_spec_sha256": resolved_sha,
            "smoke_output_path": str(args.smoke_output),
            "placeholder_count": 22,
        },
        "runtime_git_blob_raw_sha256": runtime_map,
        "dedicated_asset_git_blob_raw_sha256": dedicated_map,
        "semantic_contract": {
            "world_size": 2,
            "trainable_raw_ema_adam_count": 483,
            "adam_step": 291667,
            "stage_local_step": 0,
            "scheduler_last_epoch": 0,
            "scheduler_step_count": 1,
            "warmup_steps": 500,
            "base_lr": 1.732051e-4,
            "first_optimizer_update_lr": 3.464102e-7,
            "step1_logged_lr": 6.928204e-7,
            "branch_resolver": "training.formal_boot.resolve_checkpoint_boot_branch",
        },
    }
    atomic_file(args.resolved_spec_output, resolved_raw)
    report_sha = publish_receipt(args.render_receipt_output, value)
    value["render_receipt_sha256"] = report_sha
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--expected-phase-a-manifest-sha256", required=True)
    parser.add_argument("--training-lineage", type=Path, required=True)
    parser.add_argument("--expected-training-lineage-sha256", required=True)
    parser.add_argument("--smoke-output", type=Path, required=True)
    parser.add_argument("--resolved-spec-output", type=Path, required=True)
    parser.add_argument("--render-receipt-output", type=Path, required=True)
    return parser


def main() -> int:
    try:
        value = render(build_parser().parse_args())
    except (RenderError, OSError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
