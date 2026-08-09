#!/usr/bin/env python3
"""Filesystem protocol for the Stage512a-control FORMAL launcher.

The shell launcher owns the lifetime ``flock``.  This module makes the state
transitions underneath that lease explicit and testable: fresh/continue mode,
pre-checkpoint evidence preservation, generation-scoped node handshakes, and
the append-only FORMAL_RUN identity chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import time
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_RECEIPT_SCHEMA = "stage512a_continue_control_rngfix_formal_run_v2"
RELEASE_MANIFEST_SCHEMA = "stage512a_control_resolved_release_v1"
ARCHIVE_SCHEMA = "stage512a_control_precheckpoint_retry_archive_v1"
INVITE_SCHEMA = "stage512a_control_generation_invite_v1"
HELLO_SCHEMA = "stage512a_control_node_hello_v1"
RDZV_SCHEMA = "stage512a_control_generation_rdzv_v1"


class ProtocolError(RuntimeError):
    """A fail-closed launch protocol violation."""


def strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ProtocolError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ordinary_file(path: Path, *, nonempty: bool = True) -> os.stat_result:
    value = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(value.st_mode):
        raise ProtocolError(f"not an ordinary file: {path}")
    if nonempty and value.st_size <= 0:
        raise ProtocolError(f"empty authority file: {path}")
    return value


def readonly_file(path: Path) -> os.stat_result:
    value = ordinary_file(path)
    if value.st_mode & 0o222:
        raise ProtocolError(f"authority is writable: {path}")
    return value


def read_json(path: Path, *, readonly: bool = False) -> dict[str, Any]:
    (readonly_file if readonly else ordinary_file)(path)
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=strict_object_pairs)
    except Exception as exc:  # noqa: BLE001
        raise ProtocolError(f"invalid JSON authority {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON authority is not an object: {path}")
    return value


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: dict[str, Any], *, replace: bool, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=1) + "\n").encode()
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
    finally:
        os.close(fd)
    if replace:
        os.replace(tmp, path)
    else:
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            tmp.unlink()
            raise ProtocolError(f"create-once authority already exists: {path}") from exc
        tmp.unlink()
    fsync_dir(path.parent)


def check_token(value: str, label: str) -> None:
    if not TOKEN_RE.fullmatch(value):
        raise ProtocolError(f"malformed {label}: {value!r}")


def validate_release_manifest(path: Path, repo: Path, commit: str, spec: Path) -> dict[str, Any]:
    manifest = read_json(path, readonly=True)
    if manifest.get("schema") != RELEASE_MANIFEST_SCHEMA or manifest.get("status") != "PASS":
        raise ProtocolError("resolved release manifest schema/status mismatch")
    if manifest.get("release_repo") != str(repo) or manifest.get("commit") != commit:
        raise ProtocolError("resolved release repo/commit mismatch")
    if manifest.get("resolved_spec", {}).get("path") != str(spec):
        raise ProtocolError("resolved spec path mismatch")
    got_spec = sha256_file(spec)
    if manifest["resolved_spec"].get("sha256") != got_spec:
        raise ProtocolError("resolved spec SHA mismatch")
    cfg = repo / "configs/stage512a_continue_control_rngfix_v1.yaml"
    if manifest.get("config", {}).get("path") != str(cfg):
        raise ProtocolError("resolved config path mismatch")
    if manifest["config"].get("sha256") != sha256_file(cfg):
        raise ProtocolError("resolved config SHA mismatch")
    for key, item in manifest.get("evidence", {}).items():
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ProtocolError(f"malformed release evidence entry: {key}")
        evidence_path = Path(item["path"])
        readonly_file(evidence_path)
        if sha256_file(evidence_path) != item["sha256"]:
            raise ProtocolError(f"release evidence SHA mismatch: {key}")
    required = {"registry_closure", "rng_gpuval", "a_final_load_smoke", "b120_handoff"}
    if set(manifest.get("evidence", {})) != required:
        raise ProtocolError("resolved release evidence set is not closed")
    lineage = manifest.get("training_lineage", {})
    if set(lineage) != {"path", "sha256"}:
        raise ProtocolError("resolved training lineage binding is malformed")
    lineage_path = Path(lineage["path"])
    readonly_file(lineage_path)
    if sha256_file(lineage_path) != lineage["sha256"]:
        raise ProtocolError("training lineage SHA mismatch")
    lineage_value = read_json(lineage_path, readonly=True)
    if lineage_value.get("schema") != "stage512a_control_training_lineage_v1":
        raise ProtocolError("training lineage schema mismatch")
    if lineage_value.get("status") != "PASS":
        raise ProtocolError("training lineage status mismatch")
    forbidden = {"run_mode", "attempt_jobtag", "generation", "node_nonce", "reconcile_nonce"}
    if forbidden.intersection(lineage_value):
        raise ProtocolError("attempt-local field leaked into stable training lineage")
    return manifest


def expected_run_identity(manifest_path: Path, manifest: dict[str, Any], run_id: str,
                          checkpoint_dir: Path) -> dict[str, Any]:
    fixed = manifest["fixed_identities"]
    return {
        "run_id": run_id,
        "checkpoint_dir": str(checkpoint_dir),
        "commit": manifest["commit"],
        "config_sha256": manifest["config"]["sha256"],
        "config_path": manifest["config"]["path"],
        "release_manifest_path": str(manifest_path),
        "release_manifest_sha256": sha256_file(manifest_path),
        "training_lineage_sha256": manifest["training_lineage"]["sha256"],
        "plan_hash": fixed["plan_hash"],
        "plan_id": fixed["plan_id"],
        "plan_root": fixed["plan_root"],
        "plan_header_sha256": fixed["plan_header_sha256"],
        "plan_checksums_sha256": fixed["plan_checksums_sha256"],
        "catalog_hash": fixed["catalog_hash"],
        "catalog_header_sha256": fixed["catalog_header_sha256"],
        "weighted_contract_sha256": fixed["weighted_contract_sha256"],
        "init_from_sha256": fixed["init_from_sha256"],
        "reference_config_sha256": fixed["reference_config_sha256"],
        "canonical_plan_receipt_sha256": fixed["canonical_plan_receipt_sha256"],
        "plan_closure_sha256": fixed["plan_closure_sha256"],
        "asset_report_sha256": fixed["asset_report_sha256"],
        "gpic_payload_reuse_proof_sha256": fixed["gpic_payload_reuse_proof_sha256"],
        "gpic_verifier_report_sha256": fixed["gpic_verifier_report_sha256"],
        "registry_closure_sha256": manifest["evidence"]["registry_closure"]["sha256"],
        "registry_closure_path": manifest["evidence"]["registry_closure"]["path"],
        "rng_gpuval_sha256": manifest["evidence"]["rng_gpuval"]["sha256"],
        "rng_gpuval_path": manifest["evidence"]["rng_gpuval"]["path"],
        "a_final_load_smoke_sha256": manifest["evidence"]["a_final_load_smoke"]["sha256"],
        "a_final_load_smoke_path": manifest["evidence"]["a_final_load_smoke"]["path"],
        "b120_handoff_sha256": manifest["evidence"]["b120_handoff"]["sha256"],
        "b120_handoff_path": manifest["evidence"]["b120_handoff"]["path"],
        "world_size": 64,
        "global_batch": 2048,
        "max_steps": 125_000,
        "max_exposure_images": 256_000_000,
        "permanent_full_state_steps": [30_000, 60_000, 90_000, 120_000],
        "checkpoint_schema": "mot_full_training_v2",
        "rng_schema": "mot_rng_state_v1",
        "init_from_rng_policy": "legacy_missing_rng",
    }


def validate_first_run_receipt(checkpoint_dir: Path, expected: dict[str, Any]) -> None:
    receipt_path = checkpoint_dir / "FORMAL_RUN.json"
    value = read_json(receipt_path)
    if value.get("schema") != RUN_RECEIPT_SCHEMA:
        raise ProtocolError("FORMAL_RUN schema mismatch")
    attempts = value.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ProtocolError("FORMAL_RUN has no first attempt")
    first = attempts[0]
    if not isinstance(first, dict) or first.get("run_mode") != "fresh":
        raise ProtocolError("FORMAL_RUN first attempt was not a fresh boot")
    for key, wanted in expected.items():
        if first.get(key) != wanted:
            raise ProtocolError(f"FORMAL_RUN first-attempt identity drift: {key}")
    attempt_fields = {
        "run_mode", "attempt_jobtag", "generation", "reconcile_nonce",
        "precheckpoint_archive_receipt", "created_unix_ns",
        "eval_cfg_scales", "eval_is_launch_blocker",
    }
    if set(first) != set(expected) | attempt_fields:
        raise ProtocolError("FORMAL_RUN first-attempt key set drift")
    check_token(str(first["generation"]), "first-attempt generation")
    check_token(str(first["reconcile_nonce"]), "first-attempt reconcile nonce")


def _validate_checkpoint_root(checkpoint_dir: Path, run_id: str) -> tuple[bool, bool]:
    if checkpoint_dir.is_symlink():
        raise ProtocolError("checkpoint root may not be a symlink")
    if not checkpoint_dir.exists():
        return False, False
    if not checkpoint_dir.is_dir():
        raise ProtocolError("checkpoint root exists but is not a directory")
    owner_path = checkpoint_dir / ".jobtag"
    ordinary_file(owner_path)
    owner = owner_path.read_text().strip()
    if owner != run_id:
        raise ProtocolError(f"checkpoint root owner mismatch: {owner!r}")
    last = checkpoint_dir / "last.pt"
    if last.exists() or last.is_symlink():
        ordinary_file(last)
        return True, True
    return True, False


def probe_mode(checkpoint_dir: Path, run_mode: str, run_id: str,
               expected_identity: dict[str, Any]) -> str:
    if run_mode not in {"fresh", "continue"}:
        raise ProtocolError("RUN_MODE must be fresh or continue")
    exists, has_last = _validate_checkpoint_root(checkpoint_dir, run_id)
    if not exists:
        if run_mode == "continue":
            raise ProtocolError("continue checkpoint root does not exist")
        return "fresh"
    if has_last:
        validate_first_run_receipt(checkpoint_dir, expected_identity)
        return "continue"
    if run_mode == "continue":
        raise ProtocolError("continue checkpoint root has no last.pt")
    receipt = checkpoint_dir / "FORMAL_RUN.json"
    if receipt.exists() or receipt.is_symlink():
        validate_first_run_receipt(checkpoint_dir, expected_identity)
    return "fresh"


def _metrics_summary(path: Path) -> dict[str, Any]:
    lines = 0
    parsed = 0
    invalid = 0
    last_step: int | None = None
    with path.open("rb") as handle:
        for raw in handle:
            lines += 1
            try:
                value = json.loads(raw)
                step = value.get("step") if isinstance(value, dict) else None
                if isinstance(step, int):
                    parsed += 1
                    last_step = step
            except Exception:  # noqa: BLE001 - partial tails are evidence, not fatal.
                invalid += 1
    return {"line_count": lines, "parsed_step_lines": parsed,
            "invalid_json_lines": invalid, "last_step": last_step}


def archive_precheckpoint_residuals(checkpoint_dir: Path, run_id: str, jobtag: str,
                                    generation: str) -> Path | None:
    evidence_root = checkpoint_dir / "evidence" / "precheckpoint_retry_archive"
    if evidence_root.exists():
        if evidence_root.is_symlink() or not evidence_root.is_dir():
            raise ProtocolError("precheckpoint archive root is ambiguous")
        for child in evidence_root.iterdir():
            if child.is_symlink() or not child.is_dir():
                raise ProtocolError(f"ambiguous precheckpoint archive entry: {child}")
            receipt = child / "RECEIPT.json"
            if not receipt.exists():
                raise ProtocolError(f"incomplete prior precheckpoint archive: {child}")
            readonly_file(receipt)

    candidates: list[Path] = []
    for child in checkpoint_dir.iterdir():
        name = child.name
        is_metrics = name == "metrics.jsonl" or (
            name.startswith("metrics.") and name.endswith(".jsonl")
        )
        is_tmp = ".tmp" in name or name.endswith(".partial")
        if not (is_metrics or is_tmp):
            continue
        if child.is_symlink() or not child.is_file():
            raise ProtocolError(f"ambiguous residual artifact: {child}")
        candidates.append(child)
    if not candidates:
        return None

    evidence_root.mkdir(parents=True, exist_ok=True)
    archive_dir = evidence_root / f"generation-{generation}"
    os.mkdir(archive_dir)
    payload_dir = archive_dir / "payload"
    os.mkdir(payload_dir)
    records: list[dict[str, Any]] = []
    for source in sorted(candidates, key=lambda item: item.name):
        destination = payload_dir / source.name
        os.replace(source, destination)
        item: dict[str, Any] = {
            "name": source.name,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
        if source.name.endswith(".jsonl"):
            item.update(_metrics_summary(destination))
        records.append(item)
    fsync_dir(payload_dir)
    receipt = {
        "schema": ARCHIVE_SCHEMA,
        "status": "PASS",
        "run_id": run_id,
        "attempt_jobtag": jobtag,
        "generation": generation,
        "reason": "same-owner fresh retry before first full checkpoint",
        "last_pt_present": False,
        "exclusive_flock_held_by_caller": True,
        "files": records,
    }
    atomic_json(archive_dir / "RECEIPT.json", receipt, replace=False)
    fsync_dir(archive_dir)
    return archive_dir / "RECEIPT.json"


def resolve_mode(checkpoint_dir: Path, run_mode: str, run_id: str, jobtag: str,
                 generation: str, expected_identity: dict[str, Any]) -> tuple[str, Path | None]:
    effective = probe_mode(checkpoint_dir, run_mode, run_id, expected_identity)
    if not checkpoint_dir.exists():
        try:
            os.mkdir(checkpoint_dir)
        except FileExistsError as exc:
            raise ProtocolError("fresh checkpoint-directory atomic claim lost") from exc
        owner = checkpoint_dir / ".jobtag"
        with owner.open("x") as handle:
            handle.write(run_id + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        fsync_dir(checkpoint_dir)
    archive = None
    if effective == "fresh":
        archive = archive_precheckpoint_residuals(
            checkpoint_dir, run_id, jobtag, generation
        )
    return effective, archive


def mint_generation(coord_root: Path, run_id: str, jobtag: str, commit: str,
                    config_sha: str, node_nonce: str) -> tuple[str, Path]:
    check_token(node_nonce, "node nonce")
    generations = coord_root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        generation = secrets.token_hex(32)
        generation_dir = generations / generation
        try:
            os.mkdir(generation_dir)
            break
        except FileExistsError:
            continue
    else:
        raise ProtocolError("could not mint a unique generation directory")
    invite = {
        "schema": INVITE_SCHEMA,
        "run_id": run_id,
        "attempt_jobtag": jobtag,
        "generation": generation,
        "commit": commit,
        "config_sha256": config_sha,
        "rank0_node_nonce": node_nonce,
    }
    atomic_json(generation_dir / "INVITE.json", invite, replace=False)
    atomic_json(coord_root / f"active.{jobtag}.json", invite, replace=True)
    return generation, generation_dir


def _validate_invite(invite: dict[str, Any], *, run_id: str, jobtag: str,
                     commit: str, config_sha: str) -> tuple[str, Path]:
    if invite.get("schema") != INVITE_SCHEMA:
        raise ProtocolError("invite schema mismatch")
    expected = {
        "run_id": run_id, "attempt_jobtag": jobtag,
        "commit": commit, "config_sha256": config_sha,
    }
    for key, value in expected.items():
        if invite.get(key) != value:
            raise ProtocolError(f"invite identity mismatch: {key}")
    generation = invite.get("generation", "")
    check_token(generation, "generation")
    return generation, Path(generation)


def write_hello(generation_dir: Path, rank: int, node_nonce: str, run_id: str,
                jobtag: str, generation: str) -> bool:
    check_token(node_nonce, "node nonce")
    check_token(generation, "generation")
    hello_path = generation_dir / f"hello.n{rank}.json"
    hello = {
        "schema": HELLO_SCHEMA, "run_id": run_id,
        "attempt_jobtag": jobtag, "generation": generation,
        "node_rank": rank, "node_nonce": node_nonce,
    }
    if hello_path.exists() or hello_path.is_symlink():
        try:
            return read_json(hello_path) == hello
        except ProtocolError:
            return False
    try:
        atomic_json(hello_path, hello, replace=False)
    except ProtocolError:
        return read_json(hello_path) == hello
    return True


def _read_hello(path: Path, rank: int, run_id: str, jobtag: str,
                generation: str) -> str:
    hello = read_json(path)
    expected = {
        "schema": HELLO_SCHEMA, "run_id": run_id,
        "attempt_jobtag": jobtag, "generation": generation, "node_rank": rank,
    }
    for key, value in expected.items():
        if hello.get(key) != value:
            raise ProtocolError(f"node hello mismatch rank={rank}: {key}")
    nonce = hello.get("node_nonce", "")
    check_token(nonce, f"rank {rank} node nonce")
    return nonce


def collect_rdzv(generation_dir: Path, world_nodes: int, run_id: str, jobtag: str,
                  generation: str, commit: str, config_sha: str, mode: str,
                  master_addr: str, lineage_sha: str, timeout_s: int) -> dict[str, Any]:
    check_token(lineage_sha, "training lineage sha")
    deadline = time.monotonic() + timeout_s
    nonce_map: dict[str, str] = {}
    while time.monotonic() < deadline:
        nonce_map = {}
        try:
            for rank in range(world_nodes):
                nonce_map[str(rank)] = _read_hello(
                    generation_dir / f"hello.n{rank}.json", rank,
                    run_id, jobtag, generation,
                )
        except (FileNotFoundError, ProtocolError):
            time.sleep(1)
            continue
        extras = [
            item.name for item in generation_dir.glob("hello.n*.json")
            if item.name not in {f"hello.n{rank}.json" for rank in range(world_nodes)}
        ]
        if extras:
            raise ProtocolError(f"unexpected/duplicate node hellos: {sorted(extras)}")
        if len(set(nonce_map.values())) != world_nodes:
            raise ProtocolError("node nonces are not unique across all eight nodes")
        break
    else:
        raise ProtocolError(f"only {len(nonce_map)}/{world_nodes} valid node hellos")
    reconcile_nonce = secrets.token_hex(32)
    receipt = {
        "schema": RDZV_SCHEMA, "status": "READY", "run_id": run_id,
        "attempt_jobtag": jobtag, "attempt_tag": f"{jobtag}-{generation[:16]}",
        "generation": generation, "commit": commit,
        "config_sha256": config_sha, "effective_mode": mode,
        "training_lineage_sha256": lineage_sha,
        "master_addr": master_addr, "reconcile_nonce": reconcile_nonce,
        "node_nonce_map": nonce_map, "world_nodes": world_nodes,
    }
    atomic_json(generation_dir / "RDZV.json", receipt, replace=False)
    return receipt


def validate_rdzv(value: dict[str, Any], *, rank: int, node_nonce: str,
                  run_id: str, jobtag: str, commit: str, config_sha: str,
                  generation: str, world_nodes: int, lineage_sha: str) -> None:
    expected = {
        "schema": RDZV_SCHEMA, "status": "READY", "run_id": run_id,
        "attempt_jobtag": jobtag, "generation": generation,
        "commit": commit, "config_sha256": config_sha,
        "training_lineage_sha256": lineage_sha,
        "world_nodes": world_nodes,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ProtocolError(f"RDZV identity mismatch: {key}")
    if value.get("effective_mode") not in {"fresh", "continue"}:
        raise ProtocolError("RDZV mode invalid")
    nonce_map = value.get("node_nonce_map")
    expected_keys = {str(item) for item in range(world_nodes)}
    if not isinstance(nonce_map, dict) or set(nonce_map) != expected_keys:
        raise ProtocolError("RDZV nonce map is not exact")
    if len(set(nonce_map.values())) != world_nodes:
        raise ProtocolError("RDZV nonce map is not unique")
    if nonce_map.get(str(rank)) != node_nonce:
        raise ProtocolError("RDZV does not bind this node's current nonce")
    check_token(value.get("reconcile_nonce", ""), "common reconcile nonce")
    if not value.get("master_addr"):
        raise ProtocolError("RDZV master address missing")


def await_rdzv(coord_root: Path, rank: int, node_nonce: str, run_id: str,
               jobtag: str, commit: str, config_sha: str, world_nodes: int,
               lineage_sha: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    active = coord_root / f"active.{jobtag}.json"
    while time.monotonic() < deadline:
        try:
            invite = read_json(active)
            generation, _ = _validate_invite(
                invite, run_id=run_id, jobtag=jobtag,
                commit=commit, config_sha=config_sha,
            )
            generation_dir = coord_root / "generations" / generation
            sealed_invite = read_json(generation_dir / "INVITE.json")
            if sealed_invite != invite:
                raise ProtocolError("active invite differs from sealed generation invite")
            if not write_hello(generation_dir, rank, node_nonce, run_id, jobtag, generation):
                time.sleep(1)
                continue
            rdzv_path = generation_dir / "RDZV.json"
            if rdzv_path.exists():
                rdzv = read_json(rdzv_path)
                validate_rdzv(
                    rdzv, rank=rank, node_nonce=node_nonce, run_id=run_id,
                    jobtag=jobtag, commit=commit, config_sha=config_sha,
                    generation=generation, world_nodes=world_nodes,
                    lineage_sha=lineage_sha,
                )
                return rdzv
        except (FileNotFoundError, ProtocolError):
            pass
        time.sleep(1)
    raise ProtocolError("generation rendezvous timeout")


def append_formal_receipt(checkpoint_dir: Path, expected: dict[str, Any], mode: str,
                          jobtag: str, generation: str, reconcile_nonce: str,
                          archive_receipt: str | None) -> None:
    check_token(generation, "generation")
    check_token(reconcile_nonce, "reconcile nonce")
    out = checkpoint_dir / "FORMAL_RUN.json"
    attempt = dict(expected)
    attempt.update({
        "run_mode": mode, "attempt_jobtag": jobtag,
        "generation": generation, "reconcile_nonce": reconcile_nonce,
        "precheckpoint_archive_receipt": archive_receipt,
        "created_unix_ns": time.time_ns(),
        "eval_cfg_scales": [4, 8],
        "eval_is_launch_blocker": False,
    })
    if out.exists():
        prior = read_json(out)
        if prior.get("schema") != RUN_RECEIPT_SCHEMA:
            raise ProtocolError("cannot append to foreign FORMAL_RUN schema")
        attempts = prior.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ProtocolError("malformed prior FORMAL_RUN attempts")
        validate_first_run_receipt(checkpoint_dir, expected)
    else:
        if mode != "fresh":
            raise ProtocolError("first FORMAL_RUN attempt must be fresh")
        attempts = []
    atomic_json(out, {"schema": RUN_RECEIPT_SCHEMA, "attempts": attempts + [attempt]},
                replace=out.exists(), mode=0o644)


def _manifest_and_expected(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(args.release_manifest)
    manifest = validate_release_manifest(
        manifest_path, Path(args.repo), args.commit, Path(args.resolved_spec)
    )
    expected = expected_run_identity(
        manifest_path, manifest, args.run_id, Path(args.checkpoint_dir)
    )
    return manifest, expected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--release-manifest", required=True)
    common.add_argument("--repo", required=True)
    common.add_argument("--commit", required=True)
    common.add_argument("--resolved-spec", required=True)
    common.add_argument("--run-id", required=True)
    common.add_argument("--checkpoint-dir", required=True)

    sub.add_parser("verify-release", parents=[common])
    token = sub.add_parser("token")
    token.add_argument("--bytes", type=int, default=32)
    probe = sub.add_parser("probe-mode", parents=[common])
    probe.add_argument("--run-mode", required=True)
    resolve = sub.add_parser("resolve-mode", parents=[common])
    resolve.add_argument("--run-mode", required=True)
    resolve.add_argument("--jobtag", required=True)
    resolve.add_argument("--generation", required=True)
    mint = sub.add_parser("mint-generation")
    for item in ("coord-root", "run-id", "jobtag", "commit", "config-sha", "node-nonce"):
        mint.add_argument(f"--{item}", required=True)
    hello = sub.add_parser("write-hello")
    for item in ("generation-dir", "node-nonce", "run-id", "jobtag", "generation"):
        hello.add_argument(f"--{item}", required=True)
    hello.add_argument("--rank", type=int, required=True)
    collect = sub.add_parser("collect-rdzv")
    for item in ("generation-dir", "run-id", "jobtag", "generation", "commit",
                 "config-sha", "mode", "master-addr", "lineage-sha"):
        collect.add_argument(f"--{item}", required=True)
    collect.add_argument("--world-nodes", type=int, default=8)
    collect.add_argument("--timeout-s", type=int, default=3600)
    await_parser = sub.add_parser("await-rdzv")
    for item in ("coord-root", "node-nonce", "run-id", "jobtag", "commit", "config-sha",
                 "lineage-sha"):
        await_parser.add_argument(f"--{item}", required=True)
    await_parser.add_argument("--rank", type=int, required=True)
    await_parser.add_argument("--world-nodes", type=int, default=8)
    await_parser.add_argument("--timeout-s", type=int, default=3600)
    receipt = sub.add_parser("append-receipt", parents=[common])
    for item in ("mode", "jobtag", "generation", "reconcile-nonce"):
        receipt.add_argument(f"--{item}", required=True)
    receipt.add_argument("--archive-receipt", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "token":
            print(secrets.token_hex(args.bytes))
        elif args.command == "verify-release":
            _manifest_and_expected(args)
            print("FORMAL_RELEASE_MANIFEST_PASS")
        elif args.command == "probe-mode":
            _, expected = _manifest_and_expected(args)
            print(probe_mode(Path(args.checkpoint_dir), args.run_mode, args.run_id, expected))
        elif args.command == "resolve-mode":
            _, expected = _manifest_and_expected(args)
            mode, archive = resolve_mode(
                Path(args.checkpoint_dir), args.run_mode, args.run_id,
                args.jobtag, args.generation, expected,
            )
            print(mode + "\t" + (str(archive) if archive else "-"))
        elif args.command == "mint-generation":
            generation, generation_dir = mint_generation(
                Path(args.coord_root), args.run_id, args.jobtag, args.commit,
                args.config_sha, args.node_nonce,
            )
            print(generation + "\t" + str(generation_dir))
        elif args.command == "write-hello":
            if not write_hello(
                Path(args.generation_dir), args.rank, args.node_nonce,
                args.run_id, args.jobtag, args.generation,
            ):
                raise ProtocolError("generation hello conflicts with an existing node nonce")
        elif args.command == "collect-rdzv":
            value = collect_rdzv(
                Path(args.generation_dir), args.world_nodes, args.run_id,
                args.jobtag, args.generation, args.commit, args.config_sha,
                args.mode, args.master_addr, args.lineage_sha, args.timeout_s,
            )
            print(json.dumps(value, sort_keys=True))
        elif args.command == "await-rdzv":
            value = await_rdzv(
                Path(args.coord_root), args.rank, args.node_nonce, args.run_id,
                args.jobtag, args.commit, args.config_sha, args.world_nodes,
                args.lineage_sha, args.timeout_s,
            )
            print(json.dumps(value, sort_keys=True))
        elif args.command == "append-receipt":
            _, expected = _manifest_and_expected(args)
            append_formal_receipt(
                Path(args.checkpoint_dir), expected, args.mode, args.jobtag,
                args.generation, args.reconcile_nonce,
                None if args.archive_receipt == "-" else args.archive_receipt,
            )
        else:  # pragma: no cover
            raise ProtocolError(f"unhandled command: {args.command}")
    except (ProtocolError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"FORMAL_PROTOCOL_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
