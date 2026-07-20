#!/usr/bin/env python3
"""Capture and aggregate the strict_path/fd_stable semantic telemetry gate.

The capture deliberately runs each telemetry state in a new interpreter.  The
four resulting arms are therefore:

* strict_path, telemetry off (plain reader)
* strict_path, telemetry on  (instrumented reader)
* fd_stable, telemetry off    (plain reader)
* fd_stable, telemetry on     (instrumented reader)

Only the explicit GPIC source-consistency key may differ between the two config
snapshots.  Every arm consumes the same plan prefix with four simulated ranks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_RUN = "offline_arb_gpic_fd_semantic_run_v1"
SCHEMA_STATE = "offline_arb_gpic_fd_semantic_state_v1"
SCHEMA_ARM = "offline_arb_gpic_fd_semantic_arm_v1"
SCHEMA_TELEMETRY = "offline_arb_gpic_fd_semantic_telemetry_v1"
SCHEMA_AGGREGATE = "offline_arb_gpic_fd_semantic_aggregate_v1"
SCHEMA_RUN_CLAIM = "offline_arb_gpic_fd_semantic_run_claim_v1"
SCHEMA_ARM_CLAIM = "offline_arb_gpic_fd_semantic_arm_claim_v1"
SCHEMA_NEGATIVE_CLAIM = "offline_arb_gpic_fd_semantic_negative_claim_v1"
SCHEMA_AGGREGATE_CLAIM = "offline_arb_gpic_fd_semantic_aggregate_claim_v1"
SCHEMA_CAPTURE_MANIFEST = "offline_arb_gpic_fd_semantic_capture_manifest_v1"
TELEMETRY_SCHEMA = "offline_arb_io_profile_v1"
MODES = ("strict_path", "fd_stable")
STATES = ("off", "on")
HASH_KEYS = (
    "trace_sha256",
    "prompts_sha256",
    "pixels_sha256",
    "cfg_keep_sha256",
)
SMOKE_IDENTITY_KEYS = (
    "plan_id",
    "plan_hash",
    "catalog_hash",
    "assignment_hash",
    "semantic_hash_schema",
    "ranks",
    "per_rank_bs",
    "start_step",
    "steps",
    "total_images",
)
RANKS = 4
STEPS = 12
START_STEP = 0
PREFETCH_STEPS = 4
GPIC_THREADS = 16
REQUIRED_CONSUMED_WINDOW_STARTS = (0, 4, 8)
HEX64 = re.compile(r"[0-9a-f]{64}").fullmatch
SAFE_TOKEN = re.compile(r"[A-Za-z0-9_.-]+").fullmatch

# These are the executable sources whose live bytes must remain unchanged from
# run initialization through aggregation.  A byte-identical copy is retained
# with the evidence so a later independent audit never trusts mutable live code.
RUNTIME_FILES = (
    "jobs/offline_arb_gpic_fd_semantic_gate.yaml",
    "scripts/offline_arb/run_gpic_fd_semantic_gate.py",
    "scripts/offline_arb/run_gpic_fd_contract_negatives.py",
    "scripts/offline_arb/smoke_gpic_fd_semantic_loader.py",
    "scripts/offline_arb/smoke_offline_loader.py",
    "scripts/offline_arb/make_freeze_manifest.py",
    "scripts/offline_arb/caption_sidecar_contract.py",
    "training/dataline_offline/offline_dataset.py",
    "training/dataline_offline/planner_v1.py",
    "training/dataline_offline/caption_sidecar.py",
    "training/dataline/complete_dataset.py",
    "training/dataline/config.py",
    "training/dataline/image_fetch.py",
    "training/dataline/manifest_index.py",
    "training/dataline/multi_ar.py",
    "training/dataline/native_safety.py",
    "training/dataline/pil_safety.py",
    "training/config_schema.py",
    "train.py",
)


class GateError(RuntimeError):
    pass


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def regular_bytes(path: Path, *, nonempty: bool = True) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise GateError(f"missing file: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise GateError(f"expected non-symlink regular file: {path}")
    if nonempty and metadata.st_size <= 0:
        raise GateError(f"expected non-empty file: {path}")
    raw = path.read_bytes()
    after = path.lstat()
    before_identity = (
        metadata.st_mode,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    after_identity = (
        after.st_mode,
        after.st_dev,
        after.st_ino,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if len(raw) != metadata.st_size or before_identity != after_identity:
        raise GateError(f"file size changed while reading: {path}")
    return raw


def strict_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = regular_bytes(path)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise GateError(f"invalid strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"top-level JSON must be an object: {path}")
    return raw, value


def write_all(handle: Any, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = handle.write(view)
        if not written:
            raise OSError("short write")
        view = view[written:]


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, raw: bytes) -> None:
    """Durably publish one ordinary final file without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    real_directory(path.parent, "output parent")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    if temporary.exists() or temporary.is_symlink():
        raise GateError(f"refusing stale output path: {path}")
    with temporary.open("xb", buffering=0) as handle:
        write_all(handle, raw)
        os.fsync(handle.fileno())
    publish_temp_noreplace(temporary, path)


def publish_temp_noreplace(temporary: Path, path: Path) -> None:
    """Hard-link a fully fsynced same-directory temporary into its final name."""

    if temporary.parent != path.parent:
        raise GateError("temporary/final publish parents differ")
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as exc:
        temporary.unlink(missing_ok=True)
        raise GateError(f"refusing to replace published output: {path}") from exc
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    try:
        temporary.unlink()
    except OSError as exc:
        raise GateError(f"cannot remove publish temporary {temporary}: {exc}") from exc
    fsync_directory(path.parent)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_bytes(path, raw)


def exclusive_claim(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Create and fsync a persistent O_EXCL ownership claim."""

    path.parent.mkdir(parents=True, exist_ok=True)
    real_directory(path.parent, "claim parent")
    payload = dict(value)
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise GateError(f"ownership claim already exists: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb", buffering=0, closefd=True) as handle:
            write_all(handle, raw)
            os.fsync(handle.fileno())
    except BaseException:
        # A partial persistent claim is deliberately not removed: a retry must
        # use a new attempt namespace rather than guessing ownership.
        raise
    fsync_directory(path.parent)
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def metadata_identity(metadata: os.stat_result) -> dict[str, int]:
    return {
        "st_mode": metadata.st_mode,
        "st_dev": metadata.st_dev,
        "st_ino": metadata.st_ino,
        "st_nlink": metadata.st_nlink,
        "st_size": metadata.st_size,
        "st_mtime_ns": metadata.st_mtime_ns,
        "st_ctime_ns": metadata.st_ctime_ns,
    }


def forbidden_artifact_name(name: str) -> bool:
    lowered = name.lower()
    return (
        name.startswith("failed")
        or ".failed" in lowered
        or name.endswith(".tmp")
        or ".tmp." in name
        or name.startswith(".tmp")
    )


def scan_artifact_tree(root: Path) -> dict[str, Any]:
    """Hash a non-symlink artifact tree and reject transient/failure residue."""

    real_directory(root, "artifact root")
    directories: dict[str, dict[str, int]] = {}
    files: dict[str, dict[str, Any]] = {}
    for current_raw, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_raw)
        relative_dir = "." if current == root else current.relative_to(root).as_posix()
        current_meta = current.lstat()
        if not stat.S_ISDIR(current_meta.st_mode) or current.is_symlink():
            raise GateError(f"artifact directory is not a real directory: {current}")
        directories[relative_dir] = metadata_identity(current_meta)
        dirnames.sort()
        filenames.sort()
        for name in dirnames:
            if forbidden_artifact_name(name):
                raise GateError(f"forbidden transient/failure directory: {current / name}")
            child = current / name
            child_meta = child.lstat()
            if not stat.S_ISDIR(child_meta.st_mode) or child.is_symlink():
                raise GateError(f"artifact child directory is a symlink/non-directory: {child}")
        for name in filenames:
            if forbidden_artifact_name(name):
                raise GateError(f"forbidden transient/failure file: {current / name}")
            child = current / name
            raw = regular_bytes(child)
            relative = child.relative_to(root).as_posix()
            files[relative] = {
                **metadata_identity(child.lstat()),
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }
    return {"directories": directories, "files": files}


def expected_artifact_directories() -> set[str]:
    result = {".", "input_snapshots", "runtime_sources"}
    for mode in MODES:
        result.add(f"telemetry.{mode}.on")
    for relative in RUNTIME_FILES:
        parent = Path("runtime_sources") / relative
        for candidate in parent.parents:
            value = candidate.as_posix()
            if value == ".":
                break
            result.add(value)
    return result


def expected_artifact_files(
    hosts: Mapping[str, str], *, phase: str
) -> set[str]:
    if set(hosts) != set(MODES):
        raise GateError("host mapping does not cover both modes")
    files = {
        "claim.run.json",
        "run.json",
        "config.strict_path.yaml",
        "config.fd_stable.yaml",
        "input_snapshots/plan.header.json",
        "input_snapshots/catalog.header.json",
        "input_snapshots/sidecar.header.json",
        "input_snapshots/sidecar.control.initial.json",
        "input_snapshots/sampling_config.json",
        "claim.negative.json",
        "negative_contract.json",
        "claim.aggregate.json",
    }
    files.update(f"runtime_sources/{relative}" for relative in RUNTIME_FILES)
    for mode in MODES:
        files.add(f"arm.{mode}.json")
        for state in STATES:
            files.add(f"claim.arm.{mode}.{state}.json")
            files.add(f"state.{mode}.{state}.json")
            files.add(f"smoke.{mode}.{state}.{hosts[mode]}.log")
        files.add(f"telemetry.{mode}.on/manifest.json")
        files.update(
            f"telemetry.{mode}.on/io_profile.rank{rank:04d}.jsonl"
            for rank in range(RANKS)
        )
    if phase == "pre_manifest":
        pass
    elif phase in {"before_aggregate", "final"}:
        files.add("capture_manifest.json")
        files.add("input_snapshots/sidecar.control.final.json")
        if phase == "final":
            files.add("aggregate.json")
    else:
        raise GateError(f"unknown artifact phase: {phase}")
    return files


def assert_artifact_tree_exact(
    root: Path, hosts: Mapping[str, str], *, phase: str
) -> dict[str, Any]:
    tree = scan_artifact_tree(root)
    expected_files = expected_artifact_files(hosts, phase=phase)
    expected_dirs = expected_artifact_directories()
    observed_files = set(tree["files"])
    observed_dirs = set(tree["directories"])
    if observed_files != expected_files or observed_dirs != expected_dirs:
        raise GateError(
            "artifact tree exact-set mismatch "
            f"phase={phase} missing_files={sorted(expected_files - observed_files)} "
            f"extra_files={sorted(observed_files - expected_files)} "
            f"missing_dirs={sorted(expected_dirs - observed_dirs)} "
            f"extra_dirs={sorted(observed_dirs - expected_dirs)}"
        )
    return tree


def real_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise GateError(f"missing {label}: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise GateError(f"{label} must be a non-symlink directory: {path}")


def file_identity(path: Path, raw: bytes) -> dict[str, Any]:
    metadata = path.lstat()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "st_dev": metadata.st_dev,
        "st_ino": metadata.st_ino,
        "st_mtime_ns": metadata.st_mtime_ns,
        "st_ctime_ns": metadata.st_ctime_ns,
    }


def validate_token(value: str, label: str) -> None:
    if SAFE_TOKEN(value) is None:
        raise GateError(f"malformed {label}: {value!r}")


def validate_hex(value: Any, label: str) -> str:
    result = str(value or "")
    if HEX64(result) is None:
        raise GateError(f"malformed {label}: {value!r}")
    return result


def config_values(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise GateError(f"invalid YAML {label}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("dataset"), dict):
        raise GateError(f"{label} lacks a dataset mapping")
    return value


def configs_differ_only_by_mode(strict_raw: bytes, fd_raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    strict_value = config_values(strict_raw, "strict config")
    fd_value = config_values(fd_raw, "fd config")
    strict_dataset = strict_value["dataset"]
    fd_dataset = fd_value["dataset"]
    if strict_dataset.get("gpic_source_consistency_mode") != "strict_path":
        raise GateError("strict config mode is not explicit strict_path")
    if fd_dataset.get("gpic_source_consistency_mode") != "fd_stable":
        raise GateError("fd config mode is not explicit fd_stable")
    if strict_dataset.get("gpic_payload_in_memory") is not True or fd_dataset.get(
        "gpic_payload_in_memory"
    ) is not True:
        raise GateError("both configs must use direct in-memory GPIC payloads")
    if strict_dataset.get("gpic_threads_per_node") != GPIC_THREADS or fd_dataset.get(
        "gpic_threads_per_node"
    ) != GPIC_THREADS:
        raise GateError("both configs must explicitly freeze gpic_threads_per_node=16")
    normalized = json.loads(json.dumps(fd_value))
    normalized["dataset"]["gpic_source_consistency_mode"] = "strict_path"
    if normalized != strict_value:
        raise GateError("config snapshots differ beyond gpic_source_consistency_mode")
    return strict_value, fd_value


def snapshot_runtime_sources(repo: Path, root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    snapshot_root = root / "runtime_sources"
    snapshot_root.mkdir(mode=0o700)
    for relative in RUNTIME_FILES:
        live = repo / relative
        raw = regular_bytes(live)
        destination = snapshot_root / relative
        atomic_bytes(destination, raw)
        identity = file_identity(live, raw)
        identity["snapshot_path"] = str(destination)
        result[relative] = identity
    return result


def validate_sidecar_control(
    value: Mapping[str, Any], *, header_sha: str, label: str
) -> str:
    release = validate_hex(value.get("release_digest"), f"{label} release_digest")
    if (
        value.get("status") != "PASS"
        or value.get("mode") != "verify-sidecar-control"
        or value.get("header_sha256") != header_sha
    ):
        raise GateError(f"{label} sidecar-control identity mismatch")
    return release


def load_run(root: Path) -> dict[str, Any]:
    _raw, run = strict_json(root / "run.json")
    if run.get("schema_version") != SCHEMA_RUN:
        raise GateError("run schema mismatch")
    validate_token(str(run.get("jobtag", "")), "run jobtag")
    validate_hex(run.get("run_nonce"), "run nonce")
    if run.get("modes") != list(MODES) or run.get("states") != list(STATES):
        raise GateError("run mode/state matrix mismatch")
    if run.get("ranks") != RANKS or run.get("steps") != STEPS or run.get(
        "start_step"
    ) != START_STEP:
        raise GateError("run rank/step contract mismatch")
    claim_raw, claim = strict_json(root / "claim.run.json")
    expected_claim_evidence = {
        "path": str(root / "claim.run.json"),
        "bytes": len(claim_raw),
        "sha256": sha256_bytes(claim_raw),
    }
    if (
        set(claim)
        != {
            "schema_version",
            "jobtag",
            "run_nonce",
            "worker_index",
            "host",
            "root",
            "pid",
            "created_time_ns",
        }
        or claim.get("schema_version") != SCHEMA_RUN_CLAIM
        or claim.get("jobtag") != run.get("jobtag")
        or claim.get("run_nonce") != run.get("run_nonce")
        or claim.get("worker_index") != 0
        or claim.get("host") != run.get("worker0_host")
        or claim.get("root") != str(root)
        or type(claim.get("pid")) is not int
        or claim["pid"] <= 0
        or type(claim.get("created_time_ns")) is not int
        or run.get("claims") != {"run": expected_claim_evidence}
    ):
        raise GateError("run ownership claim mismatch")
    return run


def verify_live_bindings(root: Path, run: Mapping[str, Any]) -> None:
    repo = Path(str(run["repo_root"]))
    real_directory(repo, "repo root")
    sources = run.get("runtime_sources")
    if not isinstance(sources, dict) or set(sources) != set(RUNTIME_FILES):
        raise GateError("runtime source manifest file set mismatch")
    for relative in RUNTIME_FILES:
        evidence = sources[relative]
        if not isinstance(evidence, dict):
            raise GateError(f"runtime source evidence is not an object: {relative}")
        live = repo / relative
        raw = regular_bytes(live)
        metadata = live.lstat()
        expected = {
            "path": str(live),
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "st_dev": metadata.st_dev,
            "st_ino": metadata.st_ino,
            "st_mtime_ns": metadata.st_mtime_ns,
            "st_ctime_ns": metadata.st_ctime_ns,
            "snapshot_path": str(root / "runtime_sources" / relative),
        }
        if evidence != expected:
            raise GateError(f"live runtime source changed: {relative}")
        snapshot_raw = regular_bytes(Path(expected["snapshot_path"]))
        if snapshot_raw != raw:
            raise GateError(f"runtime source snapshot mismatch: {relative}")

    snapshots = run.get("config_snapshots")
    if not isinstance(snapshots, dict) or set(snapshots) != set(MODES):
        raise GateError("config snapshot manifest mismatch")
    config_raw: dict[str, bytes] = {}
    for mode in MODES:
        evidence = snapshots[mode]
        path = root / f"config.{mode}.yaml"
        raw = regular_bytes(path)
        if evidence != {
            "path": str(path),
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        }:
            raise GateError(f"config snapshot evidence mismatch: {mode}")
        config_raw[mode] = raw
    configs_differ_only_by_mode(config_raw["strict_path"], config_raw["fd_stable"])

    formal = Path(str(run["formal_config"]["path"]))
    formal_raw = regular_bytes(formal)
    if file_identity(formal, formal_raw) != run["formal_config"]:
        raise GateError("live formal config changed")
    if formal_raw != config_raw["strict_path"]:
        raise GateError("strict snapshot no longer equals formal config")

    bindings = run.get("bindings")
    if not isinstance(bindings, dict):
        raise GateError("run bindings missing")
    for key, filename in (
        ("plan_header", "plan.header.json"),
        ("catalog_header", "catalog.header.json"),
        ("sidecar_header", "sidecar.header.json"),
        ("sampling_config", "sampling_config.json"),
    ):
        evidence = bindings.get(key)
        if not isinstance(evidence, dict):
            raise GateError(f"missing binding: {key}")
        live = Path(str(evidence.get("path", "")))
        live_raw = regular_bytes(live)
        snapshot_raw = regular_bytes(root / "input_snapshots" / filename)
        if (
            len(live_raw) != evidence.get("bytes")
            or sha256_bytes(live_raw) != evidence.get("sha256")
            or live_raw != snapshot_raw
        ):
            raise GateError(f"live input binding changed: {key}")
    sidecar_root = Path(str(run["sidecar_root"]))
    success = regular_bytes(sidecar_root / "_SUCCESS").decode("ascii")
    header_sha = bindings["sidecar_header"].get("declared_header_sha256")
    if success not in {header_sha, f"{header_sha}\n"}:
        raise GateError("sidecar _SUCCESS identity mismatch")


def command_init(args: argparse.Namespace) -> None:
    root = args.root
    repo = args.repo
    validate_token(args.jobtag, "jobtag")
    validate_token(args.host, "worker0 host")
    if args.host != f"{args.jobtag}-0":
        raise GateError("worker0 host/jobtag mapping mismatch")
    real_directory(repo, "repo root")
    for path, label in (
        (args.plan_root, "plan root"),
        (args.catalog_root, "catalog root"),
        (args.sidecar_root, "sidecar root"),
        (args.anime_freq_root, "anime frequency root"),
    ):
        real_directory(path, label)
    real_directory(root.parent, "run parent")
    nonce = os.urandom(32).hex()
    created_time_ns = time.time_ns()
    try:
        os.mkdir(root, 0o700)
    except FileExistsError as exc:
        raise GateError(f"run root already claimed: {root}") from exc
    fsync_directory(root.parent)
    run_claim = exclusive_claim(
        root / "claim.run.json",
        {
            "schema_version": SCHEMA_RUN_CLAIM,
            "jobtag": args.jobtag,
            "run_nonce": nonce,
            "worker_index": 0,
            "host": args.host,
            "root": str(root),
            "pid": os.getpid(),
            "created_time_ns": created_time_ns,
        },
    )

    formal_raw = regular_bytes(args.formal_config)
    strict_marker = b"  gpic_source_consistency_mode: strict_path\n"
    if formal_raw.count(strict_marker) != 1:
        raise GateError("formal config must contain exactly one explicit strict_path line")
    fd_raw = formal_raw.replace(
        strict_marker,
        b"  gpic_source_consistency_mode: fd_stable\n",
        1,
    )
    strict_value, fd_value = configs_differ_only_by_mode(formal_raw, fd_raw)
    configured_sidecar = Path(str(strict_value["dataset"].get("offline_caption_sidecar_root", "")))
    if configured_sidecar != args.sidecar_root:
        raise GateError("formal config sidecar root differs from gate sidecar root")
    sampling_path = Path(str(strict_value["dataset"].get("sampling_config", "")))
    sampling_raw = regular_bytes(sampling_path)

    atomic_bytes(root / "config.strict_path.yaml", formal_raw)
    atomic_bytes(root / "config.fd_stable.yaml", fd_raw)
    input_root = root / "input_snapshots"
    input_root.mkdir(mode=0o700)

    plan_raw, plan = strict_json(args.plan_root / "header.json")
    catalog_raw, catalog = strict_json(args.catalog_root / "header.json")
    sidecar_raw, sidecar = strict_json(args.sidecar_root / "header.json")
    control_raw, control = strict_json(args.sidecar_control)
    plan_hash = validate_hex(plan.get("plan_hash"), "plan hash")
    catalog_hash = validate_hex(catalog.get("catalog_hash"), "catalog hash")
    assignment_hash = validate_hex(catalog.get("assignment_hash"), "assignment hash")
    sidecar_header_sha = validate_hex(sidecar.get("header_sha256"), "sidecar header SHA")
    sidecar_release = validate_sidecar_control(
        control, header_sha=sidecar_header_sha, label="initial"
    )
    success = regular_bytes(args.sidecar_root / "_SUCCESS").decode("ascii")
    if success not in {sidecar_header_sha, f"{sidecar_header_sha}\n"}:
        raise GateError("sidecar header/_SUCCESS identity mismatch")
    for destination, raw in (
        (input_root / "plan.header.json", plan_raw),
        (input_root / "catalog.header.json", catalog_raw),
        (input_root / "sidecar.header.json", sidecar_raw),
        (input_root / "sidecar.control.initial.json", control_raw),
        (input_root / "sampling_config.json", sampling_raw),
    ):
        atomic_bytes(destination, raw)

    runtime_sources = snapshot_runtime_sources(repo, root)
    run = {
        "schema_version": SCHEMA_RUN,
        "jobtag": args.jobtag,
        "run_nonce": nonce,
        "worker0_host": args.host,
        "created_time_ns": created_time_ns,
        "repo_root": str(repo),
        "formal_config": file_identity(args.formal_config, formal_raw),
        "plan_root": str(args.plan_root),
        "catalog_root": str(args.catalog_root),
        "sidecar_root": str(args.sidecar_root),
        "anime_freq_root": str(args.anime_freq_root),
        "ranks": RANKS,
        "steps": STEPS,
        "start_step": START_STEP,
        "prefetch_steps": PREFETCH_STEPS,
        "gpic_threads_per_node": GPIC_THREADS,
        "modes": list(MODES),
        "states": list(STATES),
        "hash_keys": list(HASH_KEYS),
        "config_snapshots": {
            "strict_path": {
                "path": str(root / "config.strict_path.yaml"),
                "bytes": len(formal_raw),
                "sha256": sha256_bytes(formal_raw),
            },
            "fd_stable": {
                "path": str(root / "config.fd_stable.yaml"),
                "bytes": len(fd_raw),
                "sha256": sha256_bytes(fd_raw),
            },
        },
        "effective_seed": int(strict_value["dataset"].get("seed", 0)),
        "bindings": {
            "plan_header": {
                "path": str(args.plan_root / "header.json"),
                "bytes": len(plan_raw),
                "sha256": sha256_bytes(plan_raw),
                "plan_id": plan.get("plan_id"),
                "plan_hash": plan_hash,
            },
            "catalog_header": {
                "path": str(args.catalog_root / "header.json"),
                "bytes": len(catalog_raw),
                "sha256": sha256_bytes(catalog_raw),
                "catalog_hash": catalog_hash,
                "assignment_hash": assignment_hash,
            },
            "sidecar_header": {
                "path": str(args.sidecar_root / "header.json"),
                "bytes": len(sidecar_raw),
                "sha256": sha256_bytes(sidecar_raw),
                "declared_header_sha256": sidecar_header_sha,
                "release_digest": sidecar_release,
            },
            "sampling_config": {
                "path": str(sampling_path),
                "bytes": len(sampling_raw),
                "sha256": sha256_bytes(sampling_raw),
            },
        },
        "runtime_sources": runtime_sources,
        "claims": {"run": run_claim},
    }
    atomic_json(root / "run.json", run)
    verify_live_bindings(root, load_run(root))
    print(
        f"GPIC_FD_SEMANTIC_INIT_PASS root={root} nonce={nonce} "
        f"plan_hash={plan_hash} catalog_hash={catalog_hash}",
        flush=True,
    )


def parse_smoke_log(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateError(f"smoke log is not UTF-8: {path}") from exc
    if "OFFLINE_ARB_WINDOW_PROFILE" in text:
        raise GateError(f"legacy profile output present: {path}")
    decoder = json.JSONDecoder(
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token}")
        )
    )
    candidates: list[dict[str, Any]] = []
    for match in re.finditer(r"(?m)^\{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start() :])
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("status") == "PASS" and isinstance(
            value.get("per_rank"), dict
        ):
            candidates.append(value)
    if len(candidates) != 1:
        raise GateError(f"expected exactly one smoke result in {path}, found {len(candidates)}")
    smoke = candidates[0]
    if (
        smoke.get("ranks") != RANKS
        or smoke.get("steps") != STEPS
        or smoke.get("start_step") != START_STEP
        or set(smoke.get("per_rank", {})) != {str(rank) for rank in range(RANKS)}
    ):
        raise GateError(f"smoke rank/step identity mismatch: {path}")
    expected_items = RANKS * STEPS * smoke.get("per_rank_bs", -1)
    if smoke.get("assertions", {}).get("item_trace_exact") != expected_items:
        raise GateError(f"smoke exact trace assertion mismatch: {path}")
    for rank, rank_value in smoke["per_rank"].items():
        if not isinstance(rank_value, dict):
            raise GateError(f"smoke rank evidence is not an object rank={rank}")
        for key in HASH_KEYS:
            validate_hex(rank_value.get(key), f"smoke {key} rank={rank}")
    return smoke


def parse_runtime_identities(
    raw: bytes, path: Path, mode: str, state_token: str
) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateError(f"smoke log is not UTF-8: {path}") from exc
    prefix = "GPIC_FD_RUNTIME_IDENTITY "
    values: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            value = json.loads(
                line[len(prefix) :],
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON token {token}")
                ),
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise GateError(f"invalid GPIC runtime identity line in {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise GateError(f"GPIC runtime identity is not an object in {path}")
        values.append(value)
    if len(values) != RANKS:
        raise GateError(f"expected {RANKS} runtime identities in {path}, found {len(values)}")
    by_rank = {value.get("rank"): value for value in values}
    if set(by_rank) != set(range(RANKS)) or len(by_rank) != len(values):
        raise GateError(f"runtime identity rank set/uniqueness mismatch in {path}")
    pids: set[int] = set()
    result = []
    for rank in range(RANKS):
        value = by_rank[rank]
        expected = {
            "rank": rank,
            "world_size": RANKS,
            "local_rank": rank,
            "local_world_size": RANKS,
            "gpic_source_consistency_mode": mode,
            "gpic_threads_per_node": GPIC_THREADS,
            "gpic_workers_per_rank": GPIC_THREADS // RANKS,
            "gpic_payload_in_memory": True,
            "attempt_state_token": state_token,
        }
        for key, wanted in expected.items():
            if value.get(key) != wanted:
                raise GateError(
                    f"runtime identity drift mode={mode} rank={rank} key={key}: "
                    f"{value.get(key)!r} != {wanted!r}"
                )
        if type(value.get("pid")) is not int or value["pid"] <= 0 or value["pid"] in pids:
            raise GateError(f"runtime identity PID invalid/duplicate mode={mode} rank={rank}")
        pids.add(value["pid"])
        result.append(value)
    return result


def proc_start_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError as exc:
        raise GateError(f"cannot read process start identity pid={pid}: {exc}") from exc
    close = raw.rfind(")")
    if close < 0:
        raise GateError(f"malformed /proc stat for pid={pid}")
    fields_from_state = raw[close + 2 :].split()
    # fields_from_state[0] is field 3 (state); field 22 (starttime) is index 19.
    if len(fields_from_state) <= 19:
        raise GateError(f"short /proc stat for pid={pid}")
    try:
        value = int(fields_from_state[19])
    except ValueError as exc:
        raise GateError(f"non-integer process starttime pid={pid}") from exc
    if value <= 0:
        raise GateError(f"non-positive process starttime pid={pid}")
    return value


def finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise GateError(f"{label} is not finite/non-negative: {value!r}")
    return result


def validate_telemetry_records(
    raw: bytes,
    *,
    rank: int,
    mode: str,
    run: Mapping[str, Any],
    host: str,
    local_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not raw.endswith(b"\n"):
        raise GateError(f"telemetry rank file lacks final newline rank={rank}")
    try:
        lines = raw.decode("utf-8").splitlines()
        records = [
            json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON token {token}")
                ),
            )
            for line in lines
        ]
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise GateError(f"invalid telemetry JSONL rank={rank}: {exc}") from exc
    if not records:
        raise GateError(f"empty telemetry rank file rank={rank}")
    for sequence, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != TELEMETRY_SCHEMA
            or record.get("sequence") != sequence
            or record.get("rank") != rank
        ):
            raise GateError(f"telemetry sequence/schema/rank drift rank={rank} seq={sequence}")
    identities = [value for value in records if value.get("record_type") == "identity"]
    closures = [value for value in records if value.get("record_type") == "closure"]
    if len(identities) != 1 or identities[0] is not records[0]:
        raise GateError(f"identity is not unique first record rank={rank}")
    if len(closures) != 1 or closures[0] is not records[-1]:
        raise GateError(f"closure is not unique last record rank={rank}")
    identity = identities[0]
    expected_identity = {
        "world_size": RANKS,
        "local_rank": rank,
        "local_world_size": RANKS,
        "plan_id": run["bindings"]["plan_header"]["plan_id"],
        "plan_hash": run["bindings"]["plan_header"]["plan_hash"],
        "catalog_hash": run["bindings"]["catalog_header"]["catalog_hash"],
        "sidecar_header_sha256": run["bindings"]["sidecar_header"][
            "declared_header_sha256"
        ],
        "sidecar_release_digest": run["bindings"]["sidecar_header"]["release_digest"],
        "attempt_jobtag": run["jobtag"],
        "attempt_nonce": run["run_nonce"],
        "attempt_host": host,
        "telemetry_root": str(local_root),
        "telemetry_path": str(local_root / f"io_profile.rank{rank:04d}.jsonl"),
        "prefetch_steps": PREFETCH_STEPS,
        "gpic_threads_per_node": GPIC_THREADS,
        "gpic_workers_per_rank": GPIC_THREADS // RANKS,
        "tar_workers_per_rank": 8,
        "gpic_payload_in_memory": True,
        "gpic_source_consistency_mode": mode,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise GateError(
                f"telemetry identity drift rank={rank} key={key}: "
                f"{identity.get(key)!r} != {expected!r}"
            )
    if type(identity.get("pid")) is not int or identity["pid"] <= 0:
        raise GateError(f"telemetry identity PID invalid rank={rank}")
    closure = closures[0]
    before = len(records) - 1
    if (
        closure.get("next_step") != STEPS
        or closure.get("records_before_closure") != before
        or closure.get("records_enqueued_before_closure") != before
        or closure.get("telemetry_fatal") is not False
        or closure.get("dropped_emits") != 0
        or closure.get("runtime_errors") != 0
        or closure.get("rejected_emits_before_closure") != 0
        or closure.get("rejected_emits_observed_by_writer") != 0
    ):
        raise GateError(f"telemetry closure accounting drift rank={rank}")

    gpic_windows: list[dict[str, Any]] = []
    observed_tokens: set[tuple[int, int]] = set()
    total_files = total_bytes = total_fstat = total_path_stat = 0
    path_stat_sum = path_stat_max = 0.0
    for record in records:
        if record.get("record_type") != "window" or record.get("domain") != "gpic":
            continue
        start, stop, count = record.get("start"), record.get("stop"), record.get("n")
        if (
            type(start) is not int
            or type(stop) is not int
            or type(count) is not int
            or start < 0
            or stop != start + PREFETCH_STEPS
            or count <= 0
        ):
            raise GateError(f"malformed GPIC window rank={rank}: {record}")
        token = (start, stop)
        if token in observed_tokens:
            raise GateError(f"duplicate GPIC window rank={rank} token={token}")
        observed_tokens.add(token)
        detail = record.get("image_detail")
        if not isinstance(detail, dict):
            raise GateError(f"missing GPIC image_detail rank={rank} token={token}")
        files = detail.get("files")
        unique_files = detail.get("unique_files")
        byte_count = detail.get("bytes")
        fstat_calls = detail.get("fstat_calls")
        path_stat_calls = detail.get("path_stat_calls")
        workers = detail.get("workers")
        expected_path_calls = 2 * files if mode == "strict_path" and type(files) is int else 0
        if (
            detail.get("kind") != "gpic_in_memory"
            or detail.get("gpic_source_consistency_mode") != mode
            or type(files) is not int
            or files != count
            or type(unique_files) is not int
            or unique_files <= 0
            or unique_files > files
            or type(byte_count) is not int
            or byte_count <= 0
            or fstat_calls != 2 * files
            or path_stat_calls != expected_path_calls
            or workers != min(GPIC_THREADS // RANKS, files)
            or sum(record.get("sub_source_counts", {}).values()) != count
        ):
            raise GateError(f"GPIC window consumption/counter drift rank={rank} token={token}")
        thread_sum = detail.get("thread_sum_s")
        operation_max = detail.get("operation_max_s")
        if not isinstance(thread_sum, dict) or not isinstance(operation_max, dict):
            raise GateError(f"GPIC timing maps missing rank={rank} token={token}")
        for key in ("open", "fstat", "path_stat", "read", "file_total"):
            finite_nonnegative(thread_sum.get(key), f"thread_sum.{key}")
            finite_nonnegative(operation_max.get(key), f"operation_max.{key}")
        current_path_sum = float(thread_sum["path_stat"])
        current_path_max = float(operation_max["path_stat"])
        if mode == "fd_stable" and (current_path_sum != 0.0 or current_path_max != 0.0):
            raise GateError(f"fd_stable recorded nonzero path_stat timing rank={rank} token={token}")
        total_files += files
        total_bytes += byte_count
        total_fstat += fstat_calls
        total_path_stat += path_stat_calls
        path_stat_sum += current_path_sum
        path_stat_max = max(path_stat_max, current_path_max)
        gpic_windows.append(
            {
                "start": start,
                "stop": stop,
                "files": files,
                "bytes": byte_count,
                "fstat_calls": fstat_calls,
                "path_stat_calls": path_stat_calls,
                "workers": workers,
                "path_stat_thread_sum_s": current_path_sum,
                "path_stat_operation_max_s": current_path_max,
            }
        )
    missing_consumed = [
        start
        for start in REQUIRED_CONSUMED_WINDOW_STARTS
        if (start, start + PREFETCH_STEPS) not in observed_tokens
    ]
    if missing_consumed or total_files <= 0 or total_bytes <= 0:
        raise GateError(
            f"telemetry did not prove consumed GPIC windows rank={rank}: missing={missing_consumed}"
        )
    summary = {
        "rank": rank,
        "records": len(records),
        "windows": sum(value.get("record_type") == "window" for value in records),
        "gpic_windows": gpic_windows,
        "gpic_window_count": len(gpic_windows),
        "gpic_files": total_files,
        "gpic_bytes": total_bytes,
        "fstat_calls": total_fstat,
        "path_stat_calls": total_path_stat,
        "path_stat_thread_sum_s": path_stat_sum,
        "path_stat_operation_max_s": path_stat_max,
        "required_consumed_window_starts": list(REQUIRED_CONSUMED_WINDOW_STARTS),
        "closure_zero_loss": True,
    }
    return identity, closure, summary


def preserve_and_validate_telemetry(
    root: Path,
    local_root: Path,
    *,
    mode: str,
    run: Mapping[str, Any],
    host: str,
) -> dict[str, Any]:
    _claim_raw, claim = strict_json(root / f"claim.arm.{mode}.on.json")
    if (
        set(claim)
        != {
            "schema_version",
            "jobtag",
            "run_nonce",
            "mode",
            "telemetry_state",
            "worker_index",
            "host",
            "pid",
            "created_time_ns",
        }
        or claim.get("schema_version") != SCHEMA_ARM_CLAIM
        or claim.get("jobtag") != run["jobtag"]
        or claim.get("run_nonce") != run["run_nonce"]
        or claim.get("mode") != mode
        or claim.get("telemetry_state") != "on"
        or claim.get("worker_index") != MODES.index(mode)
        or claim.get("host") != host
    ):
        raise GateError(f"telemetry writer lacks matching arm claim mode={mode}")
    real_directory(local_root, "local telemetry root")
    expected_names = {f"io_profile.rank{rank:04d}.jsonl" for rank in range(RANKS)}
    if {path.name for path in local_root.iterdir()} != expected_names:
        raise GateError("local telemetry rank file set mismatch")
    final = root / f"telemetry.{mode}.on"
    stage = root / f".telemetry.{mode}.on.{host}.tmp"
    if final.exists() or final.is_symlink() or stage.exists() or stage.is_symlink():
        raise GateError(f"stale shared telemetry destination for mode={mode}")
    stage.mkdir(mode=0o700)
    files: list[dict[str, Any]] = []
    pids: set[int] = set()
    for rank in range(RANKS):
        local = local_root / f"io_profile.rank{rank:04d}.jsonl"
        raw = regular_bytes(local)
        identity, closure, summary = validate_telemetry_records(
            raw,
            rank=rank,
            mode=mode,
            run=run,
            host=host,
            local_root=local_root,
        )
        if identity["pid"] in pids:
            raise GateError("telemetry rank processes did not have distinct PIDs")
        pids.add(identity["pid"])
        shared = stage / local.name
        atomic_bytes(shared, raw)
        files.append(
            {
                "rank": rank,
                "local_path": str(local),
                "shared_path": str(final / local.name),
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
                "identity": identity,
                "closure": closure,
                "summary": summary,
            }
        )
    manifest = {
        "status": "PASS",
        "schema_version": SCHEMA_TELEMETRY,
        "jobtag": run["jobtag"],
        "run_nonce": run["run_nonce"],
        "host": host,
        "mode": mode,
        "state": "on",
        "local_root": str(local_root),
        "shared_root": str(final),
        "rank_files": RANKS,
        "files": files,
    }
    atomic_json(stage / "manifest.json", manifest)
    directory_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if final.exists() or final.is_symlink():
        raise GateError(f"telemetry final directory already exists mode={mode}")
    os.rename(stage, final)
    fsync_directory(root)
    manifest_path = final / "manifest.json"
    manifest_raw = regular_bytes(manifest_path)
    return {
        "status": "PASS",
        "enabled": True,
        "schema_version": SCHEMA_TELEMETRY,
        "manifest": str(manifest_path),
        "manifest_bytes": len(manifest_raw),
        "manifest_sha256": sha256_bytes(manifest_raw),
        "shared_root": str(final),
        "rank_files": RANKS,
        "gpic_consumption_proven_per_rank": True,
        "files": files,
    }


def run_smoke_state(
    args: argparse.Namespace,
    *,
    run: Mapping[str, Any],
    mode: str,
    state: str,
    claim_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    telemetry_enabled = state == "on"
    state_token = sha256_bytes(
        f"{run['run_nonce']}\0{mode}\0{state}".encode("utf-8")
    )
    root = args.root
    host = args.host
    local_base = args.local_root / f"{mode}_{state}"
    if local_base.exists() or local_base.is_symlink():
        raise GateError(f"stale local state root: {local_base}")
    local_base.mkdir(parents=True, mode=0o700)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPYCACHEPREFIX": str(local_base / "pycache"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "ARROW_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OFFLINE_ARB_STEP_TIMEOUT_S": "1800",
            "OFFLINE_ARB_PREFETCH_STEPS": str(PREFETCH_STEPS),
            "OFFLINE_ARB_PROFILE_WINDOWS": "0",
            "OFFLINE_ARB_WAIT_LOG_THRESHOLD_S": "2",
            "OFFLINE_ARB_IO_STATE_EVERY_STEPS": "4",
            "OFFLINE_ARB_GPIC_THREADS_PER_NODE": str(GPIC_THREADS),
            "OFFLINE_ARB_LOCAL_ROOT": str(local_base / "payload"),
            "OFFLINE_ARB_SMOKE_DATASET_CONFIG": str(root / f"config.{mode}.yaml"),
            "GPIC_FD_EXPECTED_MODE": mode,
            "GPIC_FD_EXPECTED_THREADS": str(GPIC_THREADS),
            "GPIC_FD_EXPECTED_WORKERS_PER_RANK": str(GPIC_THREADS // RANKS),
            "GPIC_FD_STATE_TOKEN": state_token,
        }
    )
    for key in (
        "OFFLINE_ARB_IO_TELEMETRY_DIR",
        "OFFLINE_ARB_ATTEMPT_JOBTAG",
        "OFFLINE_ARB_ATTEMPT_NONCE",
        "OFFLINE_ARB_ATTEMPT_HOST",
        "OFFLINE_ARB_SIDECAR_HEADER_SHA256",
        "OFFLINE_ARB_SIDECAR_RELEASE_DIGEST",
        "OFFLINE_ARB_ANIME_CAPTION_ROOT",
        "OFFLINE_ARB_ANIME_CAPTION_WORKERS",
    ):
        env.pop(key, None)
    telemetry_root = local_base / "telemetry"
    if telemetry_enabled:
        telemetry_root.mkdir(mode=0o700)
        env.update(
            {
                "OFFLINE_ARB_IO_TELEMETRY_DIR": str(telemetry_root),
                "OFFLINE_ARB_ATTEMPT_JOBTAG": str(run["jobtag"]),
                "OFFLINE_ARB_ATTEMPT_NONCE": str(run["run_nonce"]),
                "OFFLINE_ARB_ATTEMPT_HOST": host,
                "OFFLINE_ARB_SIDECAR_HEADER_SHA256": str(
                    run["bindings"]["sidecar_header"]["declared_header_sha256"]
                ),
                "OFFLINE_ARB_SIDECAR_RELEASE_DIGEST": str(
                    run["bindings"]["sidecar_header"]["release_digest"]
                ),
            }
        )
    command = [
        str(args.python),
        str(args.repo / "scripts/offline_arb/smoke_gpic_fd_semantic_loader.py"),
        "--plan-root",
        str(run["plan_root"]),
        "--catalog-root",
        str(run["catalog_root"]),
        "--anime-freq-root",
        str(run["anime_freq_root"]),
        "--ranks",
        str(RANKS),
        "--steps",
        str(STEPS),
        "--start-step",
        str(START_STEP),
        "--worker-timeout-s",
        "7100",
    ]
    log_final = root / f"smoke.{mode}.{state}.{host}.log"
    log_tmp = root / f".smoke.{mode}.{state}.{host}.{os.getpid()}.log.tmp"
    if log_final.exists() or log_final.is_symlink() or log_tmp.exists() or log_tmp.is_symlink():
        raise GateError(f"stale smoke log path mode={mode} state={state}")
    started_ns = time.time_ns()
    with log_tmp.open("xb", buffering=0) as handle:
        process = subprocess.Popen(
            command,
            cwd=args.repo,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        process_starttime_ticks = proc_start_ticks(process.pid)
        try:
            rc = process.wait(timeout=7260)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
            raise GateError(f"smoke timed out mode={mode} state={state}")
        handle.flush()
        os.fsync(handle.fileno())
    publish_temp_noreplace(log_tmp, log_final)
    finished_ns = time.time_ns()
    log_raw = regular_bytes(log_final)
    if rc != 0:
        raise GateError(f"smoke failed mode={mode} state={state} rc={rc}; log={log_final}")
    smoke = parse_smoke_log(log_raw, log_final)
    runtime_identities = parse_runtime_identities(log_raw, log_final, mode, state_token)
    if (
        smoke.get("plan_hash") != run["bindings"]["plan_header"]["plan_hash"]
        or smoke.get("catalog_hash") != run["bindings"]["catalog_header"]["catalog_hash"]
        or smoke.get("assignment_hash")
        != run["bindings"]["catalog_header"]["assignment_hash"]
    ):
        raise GateError(f"smoke plan/catalog binding drift mode={mode} state={state}")
    if telemetry_enabled:
        telemetry = preserve_and_validate_telemetry(
            root,
            telemetry_root,
            mode=mode,
            run=run,
            host=host,
        )
        telemetry_pids = {
            evidence["rank"]: evidence["identity"]["pid"]
            for evidence in telemetry["files"]
        }
        runtime_pids = {value["rank"]: value["pid"] for value in runtime_identities}
        if telemetry_pids != runtime_pids:
            raise GateError(f"runtime/telemetry rank PID identity drift mode={mode}")
    else:
        if telemetry_root.exists() or telemetry_root.is_symlink():
            raise GateError(f"telemetry-off state created telemetry root mode={mode}")
        telemetry = {"status": "OFF", "enabled": False, "rank_files": 0}
    value = {
        "status": "PASS",
        "schema_version": SCHEMA_STATE,
        "jobtag": run["jobtag"],
        "run_nonce": run["run_nonce"],
        "host": host,
        "worker_index": args.worker_index,
        "mode": mode,
        "telemetry_state": state,
        "telemetry_enabled": telemetry_enabled,
        "attempt_state_token": state_token,
        "claim": dict(claim_evidence),
        "config": run["config_snapshots"][mode],
        "fresh_process": {
            "pid": process.pid,
            "started_time_ns": started_ns,
            "finished_time_ns": finished_ns,
            "returncode": rc,
            "proc_starttime_ticks": process_starttime_ticks,
            "python_invocation": str(args.python),
            "python_resolved": str(args.python.resolve(strict=True)),
        },
        "log": {
            "path": str(log_final),
            "bytes": len(log_raw),
            "sha256": sha256_bytes(log_raw),
        },
        "smoke": smoke,
        "runtime_identities": runtime_identities,
        "telemetry": telemetry,
    }
    atomic_json(root / f"state.{mode}.{state}.json", value)
    return value


def command_arm(args: argparse.Namespace) -> None:
    validate_token(args.host, "host")
    if args.mode not in MODES:
        raise GateError(f"unsupported mode: {args.mode}")
    expected_worker = MODES.index(args.mode)
    if args.worker_index != expected_worker:
        raise GateError(
            f"mode/worker mismatch: mode={args.mode} worker={args.worker_index} expected={expected_worker}"
        )
    if args.host != f"{args.jobtag}-{args.worker_index}":
        raise GateError("arm host/jobtag/worker mapping mismatch")
    run = load_run(args.root)
    if run["jobtag"] != args.jobtag:
        raise GateError("arm jobtag differs from run")
    verify_live_bindings(args.root, run)
    python_raw = regular_bytes(args.python.resolve(strict=True))
    arm_claims: dict[str, dict[str, Any]] = {}
    states = []
    for state in STATES:
        arm_claims[state] = exclusive_claim(
            args.root / f"claim.arm.{args.mode}.{state}.json",
            {
                "schema_version": SCHEMA_ARM_CLAIM,
                "jobtag": run["jobtag"],
                "run_nonce": run["run_nonce"],
                "mode": args.mode,
                "telemetry_state": state,
                "worker_index": args.worker_index,
                "host": args.host,
                "pid": os.getpid(),
                "created_time_ns": time.time_ns(),
            },
        )
        states.append(
            run_smoke_state(
                args,
                run=run,
                mode=args.mode,
                state=state,
                claim_evidence=arm_claims[state],
            )
        )
    process_identities = {
        (
            state["fresh_process"]["pid"],
            state["fresh_process"]["proc_starttime_ticks"],
        )
        for state in states
    }
    if len(process_identities) != len(STATES):
        raise GateError("telemetry off/on did not use distinct fresh process identities")
    off_smoke, on_smoke = (state["smoke"] for state in states)
    for key in SMOKE_IDENTITY_KEYS:
        if off_smoke.get(key) != on_smoke.get(key):
            raise GateError(f"plain/instrumented smoke identity drift mode={args.mode} key={key}")
    matched: dict[str, dict[str, str]] = {}
    for rank in (str(value) for value in range(RANKS)):
        matched[rank] = {}
        for key in HASH_KEYS:
            left = off_smoke["per_rank"][rank][key]
            right = on_smoke["per_rank"][rank][key]
            if left != right:
                raise GateError(
                    f"plain/instrumented semantic drift mode={args.mode} rank={rank} key={key}"
                )
            matched[rank][key] = left
    verify_live_bindings(args.root, run)
    arm = {
        "status": "PASS",
        "schema_version": SCHEMA_ARM,
        "jobtag": run["jobtag"],
        "run_nonce": run["run_nonce"],
        "host": args.host,
        "worker_index": args.worker_index,
        "mode": args.mode,
        "states": list(STATES),
        "state_files": {
            state: str(args.root / f"state.{args.mode}.{state}.json") for state in STATES
        },
        "plain_instrumented_exact": True,
        "matched_hashes": matched,
        "claims": arm_claims,
        "python": {
            "invocation": str(args.python),
            "resolved": str(args.python.resolve(strict=True)),
            "resolved_bytes": len(python_raw),
            "resolved_sha256": sha256_bytes(python_raw),
        },
    }
    atomic_json(args.root / f"arm.{args.mode}.json", arm)
    print(
        f"GPIC_FD_SEMANTIC_ARM_PASS mode={args.mode} host={args.host} "
        f"result={args.root / f'arm.{args.mode}.json'}",
        flush=True,
    )


def validate_state_artifact(
    root: Path,
    run: Mapping[str, Any],
    *,
    mode: str,
    state: str,
) -> dict[str, Any]:
    _raw, value = strict_json(root / f"state.{mode}.{state}.json")
    enabled = state == "on"
    if (
        set(value)
        != {
            "status",
            "schema_version",
            "jobtag",
            "run_nonce",
            "host",
            "worker_index",
            "mode",
            "telemetry_state",
            "telemetry_enabled",
            "attempt_state_token",
            "claim",
            "config",
            "fresh_process",
            "log",
            "smoke",
            "runtime_identities",
            "telemetry",
        }
        or value.get("status") != "PASS"
        or value.get("schema_version") != SCHEMA_STATE
        or value.get("jobtag") != run["jobtag"]
        or value.get("run_nonce") != run["run_nonce"]
        or value.get("worker_index") != MODES.index(mode)
        or value.get("mode") != mode
        or value.get("telemetry_state") != state
        or value.get("telemetry_enabled") is not enabled
        or value.get("config") != run["config_snapshots"][mode]
        or value.get("claim")
        != {
            "path": str(root / f"claim.arm.{mode}.{state}.json"),
            "bytes": len(regular_bytes(root / f"claim.arm.{mode}.{state}.json")),
            "sha256": sha256_bytes(
                regular_bytes(root / f"claim.arm.{mode}.{state}.json")
            ),
        }
        or not value.get("host")
    ):
        raise GateError(f"state artifact identity mismatch mode={mode} state={state}")
    process = value.get("fresh_process")
    if (
        not isinstance(process, dict)
        or type(process.get("pid")) is not int
        or process["pid"] <= 0
        or type(process.get("started_time_ns")) is not int
        or type(process.get("finished_time_ns")) is not int
        or process["finished_time_ns"] <= process["started_time_ns"]
        or process.get("returncode") != 0
        or type(process.get("proc_starttime_ticks")) is not int
        or process["proc_starttime_ticks"] <= 0
    ):
        raise GateError(f"fresh process evidence invalid mode={mode} state={state}")
    log = value.get("log")
    if not isinstance(log, dict):
        raise GateError(f"state log evidence missing mode={mode} state={state}")
    log_raw = regular_bytes(Path(str(log.get("path", ""))))
    if len(log_raw) != log.get("bytes") or sha256_bytes(log_raw) != log.get("sha256"):
        raise GateError(f"state log evidence changed mode={mode} state={state}")
    smoke = parse_smoke_log(log_raw, Path(log["path"]))
    if smoke != value.get("smoke"):
        raise GateError(f"embedded smoke differs from log mode={mode} state={state}")
    expected_token = sha256_bytes(
        f"{run['run_nonce']}\0{mode}\0{state}".encode("utf-8")
    )
    if value.get("attempt_state_token") != expected_token:
        raise GateError(f"state attempt token mismatch mode={mode} state={state}")
    runtime_identities = parse_runtime_identities(
        log_raw, Path(log["path"]), mode, expected_token
    )
    if value.get("runtime_identities") != runtime_identities:
        raise GateError(f"embedded runtime identities differ from log mode={mode} state={state}")
    if enabled:
        telemetry = value.get("telemetry")
        if not isinstance(telemetry, dict) or telemetry.get("enabled") is not True:
            raise GateError(f"telemetry-on evidence missing mode={mode}")
        manifest_path = Path(str(telemetry.get("manifest", "")))
        manifest_raw, manifest = strict_json(manifest_path)
        expected_root = root / f"telemetry.{mode}.on"
        if (
            set(telemetry)
            != {
                "status",
                "enabled",
                "schema_version",
                "manifest",
                "manifest_bytes",
                "manifest_sha256",
                "shared_root",
                "rank_files",
                "gpic_consumption_proven_per_rank",
                "files",
            }
            or set(manifest)
            != {
                "status",
                "schema_version",
                "jobtag",
                "run_nonce",
                "host",
                "mode",
                "state",
                "local_root",
                "shared_root",
                "rank_files",
                "files",
            }
            or telemetry.get("status") != "PASS"
            or telemetry.get("enabled") is not True
            or telemetry.get("schema_version") != SCHEMA_TELEMETRY
            or telemetry.get("manifest") != str(manifest_path)
            or manifest_path != expected_root / "manifest.json"
            or telemetry.get("manifest_bytes") != len(manifest_raw)
            or telemetry.get("manifest_sha256") != sha256_bytes(manifest_raw)
            or telemetry.get("shared_root") != str(expected_root)
            or telemetry.get("rank_files") != RANKS
            or telemetry.get("gpic_consumption_proven_per_rank") is not True
            or manifest.get("status") != "PASS"
            or manifest.get("schema_version") != SCHEMA_TELEMETRY
            or manifest.get("jobtag") != run["jobtag"]
            or manifest.get("run_nonce") != run["run_nonce"]
            or manifest.get("host") != value["host"]
            or manifest.get("mode") != mode
            or manifest.get("state") != "on"
            or manifest.get("rank_files") != RANKS
            or manifest.get("shared_root") != str(expected_root)
            or len(manifest.get("files", [])) != RANKS
            or manifest.get("files") != telemetry.get("files")
        ):
            raise GateError(f"telemetry manifest identity mismatch mode={mode}")
        expected_entries = {"manifest.json"} | {
            f"io_profile.rank{rank:04d}.jsonl" for rank in range(RANKS)
        }
        if {path.name for path in expected_root.iterdir()} != expected_entries:
            raise GateError(f"shared telemetry file set mismatch mode={mode}")
        for rank, evidence in enumerate(manifest.get("files", [])):
            shared = expected_root / f"io_profile.rank{rank:04d}.jsonl"
            raw = regular_bytes(shared)
            identity, closure, summary = validate_telemetry_records(
                raw,
                rank=rank,
                mode=mode,
                run=run,
                host=value["host"],
                local_root=Path(manifest["local_root"]),
            )
            if evidence != {
                "rank": rank,
                "local_path": str(Path(manifest["local_root"]) / shared.name),
                "shared_path": str(shared),
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
                "identity": identity,
                "closure": closure,
                "summary": summary,
            }:
                raise GateError(f"telemetry rank evidence mismatch mode={mode} rank={rank}")
        if {
            evidence["rank"]: evidence["identity"]["pid"]
            for evidence in manifest["files"]
        } != {item["rank"]: item["pid"] for item in runtime_identities}:
            raise GateError(f"runtime/telemetry PID binding mismatch mode={mode}")
    elif value.get("telemetry") != {"status": "OFF", "enabled": False, "rank_files": 0}:
        raise GateError(f"telemetry-off evidence drift mode={mode}")
    return value


def command_verify(args: argparse.Namespace) -> None:
    run = load_run(args.root)
    if run["jobtag"] != args.jobtag:
        raise GateError("verify jobtag differs from run")
    verify_live_bindings(args.root, run)
    print(
        f"GPIC_FD_SEMANTIC_VERIFY_PASS root={args.root} nonce={run['run_nonce']}",
        flush=True,
    )


def command_aggregate(args: argparse.Namespace) -> None:
    run = load_run(args.root)
    validate_token(args.host, "aggregate host")
    if (
        run["jobtag"] != args.jobtag
        or args.worker_index != 0
        or args.host != run["worker0_host"]
    ):
        raise GateError("aggregate identity differs from worker0 run contract")
    verify_live_bindings(args.root, run)
    aggregate_claim = exclusive_claim(
        args.root / "claim.aggregate.json",
        {
            "schema_version": SCHEMA_AGGREGATE_CLAIM,
            "jobtag": run["jobtag"],
            "run_nonce": run["run_nonce"],
            "worker_index": args.worker_index,
            "host": args.host,
            "pid": os.getpid(),
            "created_time_ns": time.time_ns(),
        },
    )
    final_control_raw, final_control = strict_json(args.final_sidecar_control)
    release = validate_sidecar_control(
        final_control,
        header_sha=run["bindings"]["sidecar_header"]["declared_header_sha256"],
        label="final",
    )
    if release != run["bindings"]["sidecar_header"]["release_digest"]:
        raise GateError("sidecar release changed during semantic gate")

    states = {
        mode: {
            state: validate_state_artifact(args.root, run, mode=mode, state=state)
            for state in STATES
        }
        for mode in MODES
    }
    arms: dict[str, Any] = {}
    for mode in MODES:
        _arm_raw, arm = strict_json(args.root / f"arm.{mode}.json")
        expected_arm_claims: dict[str, dict[str, Any]] = {}
        for state in STATES:
            claim_path = args.root / f"claim.arm.{mode}.{state}.json"
            claim_raw, claim = strict_json(claim_path)
            expected_arm_claims[state] = {
                "path": str(claim_path),
                "bytes": len(claim_raw),
                "sha256": sha256_bytes(claim_raw),
            }
            if (
                set(claim)
                != {
                    "schema_version",
                    "jobtag",
                    "run_nonce",
                    "mode",
                    "telemetry_state",
                    "worker_index",
                    "host",
                    "pid",
                    "created_time_ns",
                }
                or claim.get("schema_version") != SCHEMA_ARM_CLAIM
                or claim.get("jobtag") != run["jobtag"]
                or claim.get("run_nonce") != run["run_nonce"]
                or claim.get("mode") != mode
                or claim.get("telemetry_state") != state
                or claim.get("worker_index") != MODES.index(mode)
                or claim.get("host") != arm.get("host")
                or type(claim.get("pid")) is not int
                or claim["pid"] <= 0
                or type(claim.get("created_time_ns")) is not int
            ):
                raise GateError(
                    f"arm ownership claim mismatch mode={mode} state={state}"
                )
        expected_state_files = {
            state: str(args.root / f"state.{mode}.{state}.json") for state in STATES
        }
        expected_matched = {
            str(rank): {
                key: states[mode]["off"]["smoke"]["per_rank"][str(rank)][key]
                for key in HASH_KEYS
            }
            for rank in range(RANKS)
        }
        python_evidence = arm.get("python")
        if (
            set(arm)
            != {
                "status",
                "schema_version",
                "jobtag",
                "run_nonce",
                "host",
                "worker_index",
                "mode",
                "states",
                "state_files",
                "plain_instrumented_exact",
                "matched_hashes",
                "claims",
                "python",
            }
            or arm.get("status") != "PASS"
            or arm.get("schema_version") != SCHEMA_ARM
            or arm.get("jobtag") != run["jobtag"]
            or arm.get("run_nonce") != run["run_nonce"]
            or arm.get("worker_index") != MODES.index(mode)
            or arm.get("mode") != mode
            or arm.get("states") != list(STATES)
            or arm.get("state_files") != expected_state_files
            or arm.get("plain_instrumented_exact") is not True
            or arm.get("matched_hashes") != expected_matched
            or arm.get("host") != states[mode]["off"]["host"]
            or states[mode]["off"]["host"] != states[mode]["on"]["host"]
            or arm.get("claims") != expected_arm_claims
            or not isinstance(python_evidence, dict)
            or type(python_evidence.get("resolved_bytes")) is not int
            or python_evidence["resolved_bytes"] <= 0
            or HEX64(str(python_evidence.get("resolved_sha256", ""))) is None
            or any(
                state["fresh_process"].get("python_invocation")
                != python_evidence.get("invocation")
                or state["fresh_process"].get("python_resolved")
                != python_evidence.get("resolved")
                for state in states[mode].values()
            )
        ):
            raise GateError(f"arm artifact identity mismatch mode={mode}")
        if (
            states[mode]["off"]["fresh_process"]["pid"],
            states[mode]["off"]["fresh_process"]["proc_starttime_ticks"],
        ) == (
            states[mode]["on"]["fresh_process"]["pid"],
            states[mode]["on"]["fresh_process"]["proc_starttime_ticks"],
        ):
            raise GateError(f"mode did not use fresh telemetry processes: {mode}")
        arms[mode] = arm

    baseline = states["strict_path"]["off"]["smoke"]
    comparisons: dict[str, dict[str, str]] = {}
    for mode in MODES:
        for state in STATES:
            smoke = states[mode][state]["smoke"]
            for key in SMOKE_IDENTITY_KEYS:
                if smoke.get(key) != baseline.get(key):
                    raise GateError(f"four-arm smoke identity drift mode={mode} state={state} key={key}")
    for rank in (str(value) for value in range(RANKS)):
        comparisons[rank] = {}
        for key in HASH_KEYS:
            hashes = [
                states[mode][state]["smoke"]["per_rank"][rank][key]
                for mode in MODES
                for state in STATES
            ]
            if len(set(hashes)) != 1:
                raise GateError(f"four-arm semantic drift rank={rank} key={key}: {hashes}")
            comparisons[rank][key] = hashes[0]

    negative_raw, negative = strict_json(args.root / "negative_contract.json")
    negative_claim_raw, negative_claim = strict_json(args.root / "claim.negative.json")
    expected_negative_claim = {
        "path": str(args.root / "claim.negative.json"),
        "bytes": len(negative_claim_raw),
        "sha256": sha256_bytes(negative_claim_raw),
    }
    expected_negative = {
        "normal_success": 4,
        "short_read_rejected": 4,
        "in_place_rejected": 4,
        "namespace_strict_rejected": 2,
        "namespace_fd_old_inode": 2,
        "invalid_mode_rejected": 5,
    }
    if (
        negative.get("status") != "PASS"
        or negative.get("schema_version") != "offline_arb_gpic_fd_negative_contract_v1"
        or negative.get("jobtag") != run["jobtag"]
        or negative.get("run_nonce") != run["run_nonce"]
        or negative.get("worker0_host") != run["worker0_host"]
        or negative.get("claim") != expected_negative_claim
        or negative.get("summary") != expected_negative
        or negative.get("case_count") != sum(expected_negative.values())
        or set(negative_claim)
        != {
            "schema_version",
            "jobtag",
            "run_nonce",
            "worker_index",
            "host",
            "pid",
            "created_time_ns",
        }
        or negative_claim.get("schema_version") != SCHEMA_NEGATIVE_CLAIM
        or negative_claim.get("jobtag") != run["jobtag"]
        or negative_claim.get("run_nonce") != run["run_nonce"]
        or negative_claim.get("worker_index") != 0
        or negative_claim.get("host") != run["worker0_host"]
        or type(negative_claim.get("pid")) is not int
        or negative_claim["pid"] <= 0
        or type(negative_claim.get("created_time_ns")) is not int
    ):
        raise GateError("negative contract evidence mismatch")
    verify_live_bindings(args.root, run)
    hosts = {mode: states[mode]["off"]["host"] for mode in MODES}
    pre_tree = assert_artifact_tree_exact(args.root, hosts, phase="pre_manifest")
    capture_manifest = {
        "status": "PASS",
        "schema_version": SCHEMA_CAPTURE_MANIFEST,
        "jobtag": run["jobtag"],
        "run_nonce": run["run_nonce"],
        "worker0_host": run["worker0_host"],
        "aggregate_claim": aggregate_claim,
        "pre_aggregate_tree": pre_tree,
        "pre_aggregate_file_set": sorted(expected_artifact_files(hosts, phase="pre_manifest")),
        "pre_aggregate_directory_set": sorted(expected_artifact_directories()),
        "mutable_after_manifest_directories": [".", "input_snapshots"],
        "expected_post_manifest_additions": [
            "capture_manifest.json",
            "input_snapshots/sidecar.control.final.json",
            "aggregate.json",
        ],
        "forbidden_final_artifacts": ["symlink", "tmp", "failure", "extra"],
    }
    atomic_json(args.root / "capture_manifest.json", capture_manifest)
    capture_manifest_raw = regular_bytes(args.root / "capture_manifest.json")
    atomic_bytes(
        args.root / "input_snapshots" / "sidecar.control.final.json",
        final_control_raw,
    )
    assert_artifact_tree_exact(args.root, hosts, phase="before_aggregate")
    verify_live_bindings(args.root, run)
    aggregate = {
        "status": "PASS",
        "schema_version": SCHEMA_AGGREGATE,
        "jobtag": run["jobtag"],
        "run_nonce": run["run_nonce"],
        "modes": list(MODES),
        "telemetry_states": list(STATES),
        "four_fresh_process_arms": True,
        "config_only_delta": "dataset.gpic_source_consistency_mode",
        "ranks": RANKS,
        "steps": STEPS,
        "start_step": START_STEP,
        "hash_keys": list(HASH_KEYS),
        "per_rank_exact_hashes": comparisons,
        "plain_instrumented_exact_per_mode": True,
        "strict_fd_exact": True,
        "runtime_identity_contract": {
            "arms_checked": 4,
            "ranks_per_arm": RANKS,
            "gpic_threads_per_node": GPIC_THREADS,
            "gpic_workers_per_rank": GPIC_THREADS // RANKS,
            "gpic_payload_in_memory": True,
            "mode_matches_arm": True,
        },
        "telemetry_contract": {
            "strict_path": {
                "identity_mode": "strict_path",
                "fstat_calls_per_file": 2,
                "path_stat_calls_per_file": 2,
            },
            "fd_stable": {
                "identity_mode": "fd_stable",
                "fstat_calls_per_file": 2,
                "path_stat_calls_per_file": 0,
                "path_stat_time_and_max_zero": True,
            },
            "gpic_consumption_proven_per_rank": True,
            "image_detail_workers_rule": "min(4,files);files>=4=>4",
            "telemetry_fatal": False,
            "dropped_emits": 0,
            "runtime_errors": 0,
            "rejected_emits": 0,
        },
        "negative_contract": {
            "path": str(args.root / "negative_contract.json"),
            "bytes": len(negative_raw),
            "sha256": sha256_bytes(negative_raw),
            "claim": expected_negative_claim,
            "summary": expected_negative,
        },
        "claims": {
            "run": run["claims"]["run"],
            "arms": {mode: arms[mode]["claims"] for mode in MODES},
            "negative": expected_negative_claim,
            "aggregate": aggregate_claim,
        },
        "capture_manifest": {
            "path": str(args.root / "capture_manifest.json"),
            "bytes": len(capture_manifest_raw),
            "sha256": sha256_bytes(capture_manifest_raw),
            "schema_version": SCHEMA_CAPTURE_MANIFEST,
        },
        "artifact_tree_contract": {
            "final_files": sorted(expected_artifact_files(hosts, phase="final")),
            "final_directories": sorted(expected_artifact_directories()),
            "exact_set": True,
            "symlinks_allowed": False,
            "tmp_allowed": False,
            "failure_markers_allowed": False,
            "extra_artifacts_allowed": False,
        },
        "sidecar_release_unchanged": True,
        "runtime_sources_unchanged": True,
        "arm_files": {
            mode: str(args.root / f"arm.{mode}.json") for mode in MODES
        },
        "interpretation": (
            "On the fixed 4-rank/12-step plan prefix, strict_path and fd_stable, "
            "with telemetry both disabled and enabled in fresh processes, emitted "
            "identical trace, prompt, pixel, and cfg_keep hashes. Telemetry proves "
            "the consumed GPIC windows used the selected mode and exact syscall contract."
        ),
    }
    atomic_json(args.root / "aggregate.json", aggregate)
    assert_artifact_tree_exact(args.root, hosts, phase="final")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(
        f"GPIC_FD_SEMANTIC_AGGREGATE_PASS result={args.root / 'aggregate.json'}",
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    subparsers = value.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--repo", type=Path, required=True)
    init.add_argument("--jobtag", required=True)
    init.add_argument("--host", required=True)
    init.add_argument("--formal-config", type=Path, required=True)
    init.add_argument("--plan-root", type=Path, required=True)
    init.add_argument("--catalog-root", type=Path, required=True)
    init.add_argument("--sidecar-root", type=Path, required=True)
    init.add_argument("--anime-freq-root", type=Path, required=True)
    init.add_argument("--sidecar-control", type=Path, required=True)
    init.set_defaults(handler=command_init)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--jobtag", required=True)
    verify.set_defaults(handler=command_verify)

    arm = subparsers.add_parser("arm")
    arm.add_argument("--root", type=Path, required=True)
    arm.add_argument("--repo", type=Path, required=True)
    arm.add_argument("--jobtag", required=True)
    arm.add_argument("--host", required=True)
    arm.add_argument("--worker-index", type=int, required=True)
    arm.add_argument("--mode", choices=MODES, required=True)
    arm.add_argument("--python", type=Path, required=True)
    arm.add_argument("--local-root", type=Path, required=True)
    arm.set_defaults(handler=command_arm)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--root", type=Path, required=True)
    aggregate.add_argument("--jobtag", required=True)
    aggregate.add_argument("--host", required=True)
    aggregate.add_argument("--worker-index", type=int, required=True)
    aggregate.add_argument("--final-sidecar-control", type=Path, required=True)
    aggregate.set_defaults(handler=command_aggregate)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        args.handler(args)
    except GateError as exc:
        print(f"FATAL GPIC fd semantic gate: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
