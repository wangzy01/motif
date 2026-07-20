#!/usr/bin/env python3
"""Independent read-only audit for a completed GPIC fd semantic capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

import yaml


RUN_SCHEMA = "offline_arb_gpic_fd_semantic_run_v1"
STATE_SCHEMA = "offline_arb_gpic_fd_semantic_state_v1"
ARM_SCHEMA = "offline_arb_gpic_fd_semantic_arm_v1"
TELEMETRY_MANIFEST_SCHEMA = "offline_arb_gpic_fd_semantic_telemetry_v1"
TELEMETRY_RECORD_SCHEMA = "offline_arb_io_profile_v1"
AGGREGATE_SCHEMA = "offline_arb_gpic_fd_semantic_aggregate_v1"
NEGATIVE_SCHEMA = "offline_arb_gpic_fd_negative_contract_v1"
AUDIT_SCHEMA = "offline_arb_gpic_fd_semantic_independent_audit_v1"
RUN_CLAIM_SCHEMA = "offline_arb_gpic_fd_semantic_run_claim_v1"
ARM_CLAIM_SCHEMA = "offline_arb_gpic_fd_semantic_arm_claim_v1"
NEGATIVE_CLAIM_SCHEMA = "offline_arb_gpic_fd_semantic_negative_claim_v1"
AGGREGATE_CLAIM_SCHEMA = "offline_arb_gpic_fd_semantic_aggregate_claim_v1"
CAPTURE_MANIFEST_SCHEMA = "offline_arb_gpic_fd_semantic_capture_manifest_v1"
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
REQUIRED_GPIC_STARTS = (0, 4, 8)
HEX64 = re.compile(r"[0-9a-f]{64}").fullmatch
SAFE_TOKEN = re.compile(r"[A-Za-z0-9_.-]+").fullmatch
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


class AuditError(RuntimeError):
    pass


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


IDENTITY_KEYS = frozenset(
    {"st_mode", "st_dev", "st_ino", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"}
)


def forbidden_artifact_name(name: str) -> bool:
    lowered = name.lower()
    return (
        name.startswith("failed")
        or ".failed" in lowered
        or name.endswith(".tmp")
        or ".tmp." in name
        or name.startswith(".tmp")
    )


class SourceTreeFreeze:
    """Immutable cached byte view plus before/after filesystem identity guard."""

    def __init__(self, root: Path):
        self.root = Path(os.path.abspath(root))
        self.directories: dict[str, dict[str, int]] = {}
        self.files: dict[str, dict[str, Any]] = {}
        self.bytes_by_relative: dict[str, bytes] = {}
        self._capture()
        # Catch replacement or additions racing the initial full-tree read.
        self.verify_unchanged()

    def _relative(self, path: Path) -> str | None:
        absolute = Path(os.path.abspath(path))
        try:
            relative = absolute.relative_to(self.root)
        except ValueError:
            return None
        return relative.as_posix() if relative.as_posix() != "." else "."

    def _capture(self) -> None:
        try:
            root_meta = self.root.lstat()
        except FileNotFoundError as exc:
            raise AuditError(f"missing source root: {self.root}") from exc
        if not stat.S_ISDIR(root_meta.st_mode) or self.root.is_symlink():
            raise AuditError(f"source root is not a real directory: {self.root}")
        for current_raw, dirnames, filenames in os.walk(
            self.root, topdown=True, followlinks=False
        ):
            current = Path(current_raw)
            relative_dir = "." if current == self.root else current.relative_to(self.root).as_posix()
            current_meta = current.lstat()
            if not stat.S_ISDIR(current_meta.st_mode) or current.is_symlink():
                raise AuditError(f"source tree contains symlink/non-directory: {current}")
            self.directories[relative_dir] = metadata_identity(current_meta)
            dirnames.sort()
            filenames.sort()
            for name in dirnames:
                child = current / name
                if forbidden_artifact_name(name):
                    raise AuditError(f"source tree contains tmp/failure directory: {child}")
                child_meta = child.lstat()
                if not stat.S_ISDIR(child_meta.st_mode) or child.is_symlink():
                    raise AuditError(f"source tree contains symlink/non-directory: {child}")
            for name in filenames:
                child = current / name
                if forbidden_artifact_name(name):
                    raise AuditError(f"source tree contains tmp/failure file: {child}")
                before = child.lstat()
                if not stat.S_ISREG(before.st_mode) or child.is_symlink() or before.st_size <= 0:
                    raise AuditError(f"source tree contains symlink/invalid file: {child}")
                raw = child.read_bytes()
                after = child.lstat()
                if metadata_identity(before) != metadata_identity(after) or len(raw) != before.st_size:
                    raise AuditError(f"source file changed during initial freeze: {child}")
                relative = child.relative_to(self.root).as_posix()
                self.bytes_by_relative[relative] = raw
                self.files[relative] = {
                    **metadata_identity(after),
                    "bytes": len(raw),
                    "sha256": digest(raw),
                }

    def bytes_for(self, path: Path) -> bytes | None:
        relative = self._relative(path)
        if relative is None:
            return None
        if relative not in self.bytes_by_relative:
            raise AuditError(f"path is absent from frozen source file set: {path}")
        return self.bytes_by_relative[relative]

    def directory_present(self, path: Path) -> bool | None:
        relative = self._relative(path)
        if relative is None:
            return None
        return relative in self.directories

    def child_file_names(self, directory: Path) -> set[str]:
        relative = self._relative(directory)
        if relative is None or relative not in self.directories:
            raise AuditError(f"directory absent from frozen source tree: {directory}")
        prefix = "" if relative == "." else f"{relative}/"
        result = set()
        for name in self.files:
            if name.startswith(prefix):
                suffix = name[len(prefix) :]
                if "/" not in suffix:
                    result.add(suffix)
        return result

    def assert_exact(self, files: set[str], directories: set[str]) -> None:
        observed_files = set(self.files)
        observed_dirs = set(self.directories)
        if observed_files != files or observed_dirs != directories:
            raise AuditError(
                "frozen source exact-set mismatch "
                f"missing_files={sorted(files - observed_files)} "
                f"extra_files={sorted(observed_files - files)} "
                f"missing_dirs={sorted(directories - observed_dirs)} "
                f"extra_dirs={sorted(observed_dirs - directories)}"
            )

    def verify_unchanged(self) -> None:
        observed_dirs: dict[str, dict[str, int]] = {}
        observed_files: dict[str, dict[str, Any]] = {}
        try:
            for current_raw, dirnames, filenames in os.walk(
                self.root, topdown=True, followlinks=False
            ):
                current = Path(current_raw)
                relative_dir = "." if current == self.root else current.relative_to(self.root).as_posix()
                current_meta = current.lstat()
                if not stat.S_ISDIR(current_meta.st_mode) or current.is_symlink():
                    raise AuditError(f"source directory replaced during audit: {current}")
                observed_dirs[relative_dir] = metadata_identity(current_meta)
                dirnames.sort()
                filenames.sort()
                for name in dirnames:
                    child = current / name
                    if forbidden_artifact_name(name) or child.is_symlink():
                        raise AuditError(f"source tree gained tmp/failure/symlink directory: {child}")
                    if not stat.S_ISDIR(child.lstat().st_mode):
                        raise AuditError(f"source tree directory changed type: {child}")
                for name in filenames:
                    child = current / name
                    if forbidden_artifact_name(name):
                        raise AuditError(f"source tree gained tmp/failure file: {child}")
                    before = child.lstat()
                    if not stat.S_ISREG(before.st_mode) or child.is_symlink() or before.st_size <= 0:
                        raise AuditError(f"source tree gained symlink/invalid file: {child}")
                    raw = child.read_bytes()
                    after = child.lstat()
                    if metadata_identity(before) != metadata_identity(after):
                        raise AuditError(f"source file changed during final read: {child}")
                    relative = child.relative_to(self.root).as_posix()
                    observed_files[relative] = {
                        **metadata_identity(after),
                        "bytes": len(raw),
                        "sha256": digest(raw),
                    }
        except FileNotFoundError as exc:
            raise AuditError(f"source tree entry disappeared during audit: {exc}") from exc
        if observed_dirs != self.directories or observed_files != self.files:
            raise AuditError("source tree bytes/hash/fstat identity or exact set changed during audit")


ACTIVE_SOURCE_FREEZE: SourceTreeFreeze | None = None


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_hex(value: Any, label: str) -> str:
    result = str(value or "")
    if HEX64(result) is None:
        raise AuditError(f"malformed {label}: {value!r}")
    return result


def regular_bytes(path: Path) -> bytes:
    if ACTIVE_SOURCE_FREEZE is not None:
        cached = ACTIVE_SOURCE_FREEZE.bytes_for(path)
        if cached is not None:
            return cached
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise AuditError(f"missing file: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink() or metadata.st_size <= 0:
        raise AuditError(f"expected non-empty non-symlink regular file: {path}")
    raw = path.read_bytes()
    after = path.lstat()
    if (
        len(raw) != metadata.st_size
        or metadata_identity(metadata) != metadata_identity(after)
    ):
        raise AuditError(f"file size changed while reading: {path}")
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
        raise AuditError(f"invalid strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"top-level JSON is not an object: {path}")
    return raw, value


def real_directory(path: Path, label: str) -> None:
    if ACTIVE_SOURCE_FREEZE is not None:
        cached = ACTIVE_SOURCE_FREEZE.directory_present(path)
        if cached is not None:
            if not cached:
                raise AuditError(f"{label} is absent from frozen source directories: {path}")
            return
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise AuditError(f"missing {label}: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise AuditError(f"{label} is not a non-symlink directory: {path}")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    real_directory(path.parent, "audit attempt root")
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
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("xb", buffering=0) as handle:
        view = memoryview(raw)
        while view:
            written = handle.write(view)
            if not written:
                raise OSError("short audit write")
            view = view[written:]
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as exc:
        temporary.unlink(missing_ok=True)
        raise AuditError(f"refusing to overwrite audit output: {path}") from exc
    try:
        temporary.unlink()
    except OSError as exc:
        raise AuditError(f"cannot remove audit output temporary: {temporary}") from exc
    descriptor = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_smoke_log(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError(f"smoke log is not UTF-8: {path}") from exc
    if "OFFLINE_ARB_WINDOW_PROFILE" in text:
        raise AuditError(f"legacy profile record found in smoke log: {path}")
    decoder = json.JSONDecoder(
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token}")
        )
    )
    values: list[dict[str, Any]] = []
    for match in re.finditer(r"(?m)^\{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start() :])
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("status") == "PASS" and isinstance(
            value.get("per_rank"), dict
        ):
            values.append(value)
    if len(values) != 1:
        raise AuditError(f"expected exactly one smoke JSON in {path}, found {len(values)}")
    return values[0]


def parse_runtime_identities(
    raw: bytes, path: Path, mode: str, state_token: str
) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError(f"smoke log is not UTF-8: {path}") from exc
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
            raise AuditError(f"invalid runtime identity record in {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise AuditError(f"runtime identity is not an object in {path}")
        values.append(value)
    if len(values) != RANKS:
        raise AuditError(f"expected {RANKS} runtime identities in {path}, found {len(values)}")
    by_rank = {value.get("rank"): value for value in values}
    if set(by_rank) != set(range(RANKS)) or len(by_rank) != len(values):
        raise AuditError(f"runtime identity rank set/uniqueness mismatch in {path}")
    pids: set[int] = set()
    ordered = []
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
                raise AuditError(f"runtime identity drift mode={mode} rank={rank} key={key}")
        if type(value.get("pid")) is not int or value["pid"] <= 0 or value["pid"] in pids:
            raise AuditError(f"runtime identity PID invalid/duplicate mode={mode} rank={rank}")
        pids.add(value["pid"])
        ordered.append(value)
    return ordered


def finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise AuditError(f"{label} is not finite/non-negative")
    return result


def expected_source_directories() -> set[str]:
    result = {".", "input_snapshots", "runtime_sources"}
    result.update(f"telemetry.{mode}.on" for mode in MODES)
    for relative in RUNTIME_FILES:
        parent = Path("runtime_sources") / relative
        for candidate in parent.parents:
            value = candidate.as_posix()
            if value == ".":
                break
            result.add(value)
    return result


def expected_source_files(hosts: Mapping[str, str]) -> set[str]:
    if set(hosts) != set(MODES):
        raise AuditError("source host mapping does not cover both modes")
    files = {
        "claim.run.json",
        "run.json",
        "config.strict_path.yaml",
        "config.fd_stable.yaml",
        "input_snapshots/plan.header.json",
        "input_snapshots/catalog.header.json",
        "input_snapshots/sidecar.header.json",
        "input_snapshots/sidecar.control.initial.json",
        "input_snapshots/sidecar.control.final.json",
        "input_snapshots/sampling_config.json",
        "claim.negative.json",
        "negative_contract.json",
        "claim.aggregate.json",
        "capture_manifest.json",
        "aggregate.json",
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
    return files


def file_evidence(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path), "bytes": len(raw), "sha256": digest(raw)}


def audit_run_claim(root: Path, run: Mapping[str, Any]) -> dict[str, Any]:
    raw, claim = strict_json(root / "claim.run.json")
    evidence = file_evidence(root / "claim.run.json", raw)
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
        or claim.get("schema_version") != RUN_CLAIM_SCHEMA
        or claim.get("jobtag") != run.get("jobtag")
        or claim.get("run_nonce") != run.get("run_nonce")
        or claim.get("worker_index") != 0
        or claim.get("host") != run.get("worker0_host")
        or claim.get("root") != str(root)
        or type(claim.get("pid")) is not int
        or claim["pid"] <= 0
        or type(claim.get("created_time_ns")) is not int
        or run.get("claims") != {"run": evidence}
    ):
        raise AuditError("run O_EXCL claim mismatch")
    return evidence


def audit_capture_manifest(
    root: Path,
    run: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    hosts: Mapping[str, str],
    aggregate_claim_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if ACTIVE_SOURCE_FREEZE is None:
        raise AuditError("source tree freeze missing during capture-manifest audit")
    raw, manifest = strict_json(root / "capture_manifest.json")
    expected_manifest_evidence = {
        "path": str(root / "capture_manifest.json"),
        "bytes": len(raw),
        "sha256": digest(raw),
        "schema_version": CAPTURE_MANIFEST_SCHEMA,
    }
    final_files = expected_source_files(hosts)
    post_additions = {
        "capture_manifest.json",
        "input_snapshots/sidecar.control.final.json",
        "aggregate.json",
    }
    pre_files = final_files - post_additions
    expected_dirs = expected_source_directories()
    tree = manifest.get("pre_aggregate_tree")
    if (
        set(manifest)
        != {
            "status",
            "schema_version",
            "jobtag",
            "run_nonce",
            "worker0_host",
            "aggregate_claim",
            "pre_aggregate_tree",
            "pre_aggregate_file_set",
            "pre_aggregate_directory_set",
            "mutable_after_manifest_directories",
            "expected_post_manifest_additions",
            "forbidden_final_artifacts",
        }
        or manifest.get("status") != "PASS"
        or manifest.get("schema_version") != CAPTURE_MANIFEST_SCHEMA
        or manifest.get("jobtag") != run["jobtag"]
        or manifest.get("run_nonce") != run["run_nonce"]
        or manifest.get("worker0_host") != run["worker0_host"]
        or manifest.get("aggregate_claim") != aggregate_claim_evidence
        or manifest.get("pre_aggregate_file_set") != sorted(pre_files)
        or manifest.get("pre_aggregate_directory_set") != sorted(expected_dirs)
        or manifest.get("mutable_after_manifest_directories")
        != [".", "input_snapshots"]
        or manifest.get("expected_post_manifest_additions")
        != [
            "capture_manifest.json",
            "input_snapshots/sidecar.control.final.json",
            "aggregate.json",
        ]
        or manifest.get("forbidden_final_artifacts")
        != ["symlink", "tmp", "failure", "extra"]
        or aggregate.get("capture_manifest") != expected_manifest_evidence
        or not isinstance(tree, dict)
        or set(tree) != {"files", "directories"}
        or set(tree.get("files", {})) != pre_files
        or set(tree.get("directories", {})) != expected_dirs
    ):
        raise AuditError("capture manifest schema/exact-set binding mismatch")
    for relative in pre_files:
        if tree["files"][relative] != ACTIVE_SOURCE_FREEZE.files[relative]:
            raise AuditError(f"pre-aggregate file identity changed: {relative}")
    for relative in expected_dirs - {".", "input_snapshots"}:
        if tree["directories"][relative] != ACTIVE_SOURCE_FREEZE.directories[relative]:
            raise AuditError(f"pre-aggregate directory identity changed: {relative}")
    for relative in (".", "input_snapshots"):
        evidence = tree["directories"].get(relative)
        if (
            not isinstance(evidence, dict)
            or set(evidence) != IDENTITY_KEYS
            or any(type(value) is not int for value in evidence.values())
        ):
            raise AuditError(f"malformed mutable pre-aggregate directory identity: {relative}")
    expected_tree_contract = {
        "final_files": sorted(final_files),
        "final_directories": sorted(expected_dirs),
        "exact_set": True,
        "symlinks_allowed": False,
        "tmp_allowed": False,
        "failure_markers_allowed": False,
        "extra_artifacts_allowed": False,
    }
    if aggregate.get("artifact_tree_contract") != expected_tree_contract:
        raise AuditError("aggregate final artifact-tree contract mismatch")
    return expected_manifest_evidence


def audit_configs(root: Path, run: Mapping[str, Any]) -> dict[str, str]:
    raw_by_mode: dict[str, bytes] = {}
    value_by_mode: dict[str, dict[str, Any]] = {}
    snapshots = run.get("config_snapshots")
    if not isinstance(snapshots, dict) or set(snapshots) != set(MODES):
        raise AuditError("config snapshot manifest file set mismatch")
    for mode in MODES:
        path = root / f"config.{mode}.yaml"
        raw = regular_bytes(path)
        evidence = snapshots[mode]
        if evidence != {"path": str(path), "bytes": len(raw), "sha256": digest(raw)}:
            raise AuditError(f"config snapshot evidence mismatch: {mode}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise AuditError(f"invalid config YAML mode={mode}: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("dataset"), dict):
            raise AuditError(f"config lacks dataset mapping mode={mode}")
        if value["dataset"].get("gpic_source_consistency_mode") != mode:
            raise AuditError(f"config mode identity mismatch: {mode}")
        if value["dataset"].get("gpic_payload_in_memory") is not True:
            raise AuditError(f"config direct GPIC payload disabled: {mode}")
        if value["dataset"].get("gpic_threads_per_node") != GPIC_THREADS:
            raise AuditError(f"config explicit GPIC thread count drift: {mode}")
        raw_by_mode[mode] = raw
        value_by_mode[mode] = value
    normalized = json.loads(json.dumps(value_by_mode["fd_stable"]))
    normalized["dataset"]["gpic_source_consistency_mode"] = "strict_path"
    if normalized != value_by_mode["strict_path"]:
        raise AuditError("strict/fd configs differ beyond the selected mode")
    formal = run.get("formal_config")
    if (
        not isinstance(formal, dict)
        or formal.get("bytes") != len(raw_by_mode["strict_path"])
        or formal.get("sha256") != digest(raw_by_mode["strict_path"])
    ):
        raise AuditError("formal config evidence differs from strict snapshot")
    return {mode: digest(raw_by_mode[mode]) for mode in MODES}


def audit_runtime_sources(root: Path, run: Mapping[str, Any]) -> dict[str, str]:
    sources = run.get("runtime_sources")
    if not isinstance(sources, dict) or set(sources) != set(RUNTIME_FILES):
        raise AuditError("runtime source manifest file set mismatch")
    result: dict[str, str] = {}
    for relative in RUNTIME_FILES:
        evidence = sources[relative]
        snapshot = root / "runtime_sources" / relative
        raw = regular_bytes(snapshot)
        if (
            not isinstance(evidence, dict)
            or evidence.get("snapshot_path") != str(snapshot)
            or evidence.get("bytes") != len(raw)
            or evidence.get("sha256") != digest(raw)
        ):
            raise AuditError(f"runtime source snapshot mismatch: {relative}")
        result[relative] = digest(raw)
    if ACTIVE_SOURCE_FREEZE is None:
        raise AuditError("source tree was not frozen before runtime-source audit")
    observed_files = {
        relative.removeprefix("runtime_sources/")
        for relative in ACTIVE_SOURCE_FREEZE.files
        if relative.startswith("runtime_sources/")
    }
    if observed_files != set(RUNTIME_FILES):
        raise AuditError("runtime source snapshot directory has extra/missing files")
    return result


def audit_input_bindings(root: Path, run: Mapping[str, Any]) -> dict[str, str]:
    bindings = run.get("bindings")
    if not isinstance(bindings, dict):
        raise AuditError("input bindings missing")
    result: dict[str, str] = {}
    for key, filename in (
        ("plan_header", "plan.header.json"),
        ("catalog_header", "catalog.header.json"),
        ("sidecar_header", "sidecar.header.json"),
        ("sampling_config", "sampling_config.json"),
    ):
        raw = regular_bytes(root / "input_snapshots" / filename)
        evidence = bindings.get(key)
        if (
            not isinstance(evidence, dict)
            or evidence.get("bytes") != len(raw)
            or evidence.get("sha256") != digest(raw)
        ):
            raise AuditError(f"input snapshot binding mismatch: {key}")
        result[key] = digest(raw)
    _plan_raw, plan = strict_json(root / "input_snapshots" / "plan.header.json")
    _catalog_raw, catalog = strict_json(root / "input_snapshots" / "catalog.header.json")
    _sidecar_raw, sidecar = strict_json(root / "input_snapshots" / "sidecar.header.json")
    _initial_raw, initial = strict_json(
        root / "input_snapshots" / "sidecar.control.initial.json"
    )
    _final_raw, final = strict_json(root / "input_snapshots" / "sidecar.control.final.json")
    plan_hash = validate_hex(plan.get("plan_hash"), "plan hash")
    catalog_hash = validate_hex(catalog.get("catalog_hash"), "catalog hash")
    assignment_hash = validate_hex(catalog.get("assignment_hash"), "assignment hash")
    header_sha = validate_hex(sidecar.get("header_sha256"), "sidecar header SHA")
    release = validate_hex(initial.get("release_digest"), "initial release digest")
    if (
        bindings["plan_header"].get("plan_id") != plan.get("plan_id")
        or bindings["plan_header"].get("plan_hash") != plan_hash
        or bindings["catalog_header"].get("catalog_hash") != catalog_hash
        or bindings["catalog_header"].get("assignment_hash") != assignment_hash
        or bindings["sidecar_header"].get("declared_header_sha256") != header_sha
        or bindings["sidecar_header"].get("release_digest") != release
    ):
        raise AuditError("declared header identities differ from snapshots")
    for label, control in (("initial", initial), ("final", final)):
        if (
            control.get("status") != "PASS"
            or control.get("mode") != "verify-sidecar-control"
            or control.get("header_sha256") != header_sha
            or control.get("release_digest") != release
        ):
            raise AuditError(f"{label} sidecar control mismatch")
    return result


def audit_telemetry_rank(
    raw: bytes,
    *,
    rank: int,
    mode: str,
    host: str,
    local_root: Path,
    run: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not raw.endswith(b"\n"):
        raise AuditError(f"telemetry lacks final newline mode={mode} rank={rank}")
    try:
        records = [
            json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON token {token}")
                ),
            )
            for line in raw.decode("utf-8").splitlines()
        ]
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid telemetry JSONL mode={mode} rank={rank}: {exc}") from exc
    if not records:
        raise AuditError(f"empty telemetry mode={mode} rank={rank}")
    for sequence, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != TELEMETRY_RECORD_SCHEMA
            or record.get("sequence") != sequence
            or record.get("rank") != rank
        ):
            raise AuditError(f"telemetry sequence/schema/rank drift mode={mode} rank={rank}")
    identities = [value for value in records if value.get("record_type") == "identity"]
    closures = [value for value in records if value.get("record_type") == "closure"]
    if len(identities) != 1 or identities[0] is not records[0]:
        raise AuditError(f"telemetry identity order drift mode={mode} rank={rank}")
    if len(closures) != 1 or closures[0] is not records[-1]:
        raise AuditError(f"telemetry closure order drift mode={mode} rank={rank}")
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
            raise AuditError(f"telemetry identity drift mode={mode} rank={rank} key={key}")
    if type(identity.get("pid")) is not int or identity["pid"] <= 0:
        raise AuditError(f"invalid rank process PID mode={mode} rank={rank}")
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
        raise AuditError(f"telemetry closure accounting drift mode={mode} rank={rank}")

    tokens: set[tuple[int, int]] = set()
    windows: list[dict[str, Any]] = []
    totals = {"files": 0, "bytes": 0, "fstat_calls": 0, "path_stat_calls": 0}
    path_sum = path_max = 0.0
    for record in records:
        if record.get("record_type") != "window" or record.get("domain") != "gpic":
            continue
        start, stop, count = record.get("start"), record.get("stop"), record.get("n")
        if (
            type(start) is not int
            or type(stop) is not int
            or type(count) is not int
            or stop != start + PREFETCH_STEPS
            or count <= 0
        ):
            raise AuditError(f"malformed GPIC window mode={mode} rank={rank}")
        token = (start, stop)
        if token in tokens:
            raise AuditError(f"duplicate GPIC window mode={mode} rank={rank} token={token}")
        tokens.add(token)
        detail = record.get("image_detail")
        if not isinstance(detail, dict):
            raise AuditError(f"missing GPIC detail mode={mode} rank={rank}")
        files, byte_count = detail.get("files"), detail.get("bytes")
        expected_path_calls = 2 * files if mode == "strict_path" and type(files) is int else 0
        if (
            detail.get("kind") != "gpic_in_memory"
            or detail.get("gpic_source_consistency_mode") != mode
            or type(files) is not int
            or files != count
            or type(detail.get("unique_files")) is not int
            or not 0 < detail["unique_files"] <= files
            or type(byte_count) is not int
            or byte_count <= 0
            or detail.get("fstat_calls") != 2 * files
            or detail.get("path_stat_calls") != expected_path_calls
            or detail.get("workers") != min(GPIC_THREADS // RANKS, files)
            or not isinstance(record.get("sub_source_counts"), dict)
            or sum(record["sub_source_counts"].values()) != count
        ):
            raise AuditError(f"GPIC consumption/counter drift mode={mode} rank={rank}")
        thread_sum, operation_max = detail.get("thread_sum_s"), detail.get("operation_max_s")
        if not isinstance(thread_sum, dict) or not isinstance(operation_max, dict):
            raise AuditError(f"GPIC timing maps missing mode={mode} rank={rank}")
        for key in ("open", "fstat", "path_stat", "read", "file_total"):
            finite_nonnegative(thread_sum.get(key), f"{mode}.rank{rank}.sum.{key}")
            finite_nonnegative(operation_max.get(key), f"{mode}.rank{rank}.max.{key}")
        current_sum = float(thread_sum["path_stat"])
        current_max = float(operation_max["path_stat"])
        if mode == "fd_stable" and (current_sum != 0.0 or current_max != 0.0):
            raise AuditError(f"fd_stable path-stat timing nonzero rank={rank}")
        totals["files"] += files
        totals["bytes"] += byte_count
        totals["fstat_calls"] += detail["fstat_calls"]
        totals["path_stat_calls"] += detail["path_stat_calls"]
        path_sum += current_sum
        path_max = max(path_max, current_max)
        windows.append(
            {
                "start": start,
                "stop": stop,
                "files": files,
                "bytes": byte_count,
                "fstat_calls": detail["fstat_calls"],
                "path_stat_calls": detail["path_stat_calls"],
                "workers": detail["workers"],
                "path_stat_thread_sum_s": current_sum,
                "path_stat_operation_max_s": current_max,
            }
        )
    missing = [start for start in REQUIRED_GPIC_STARTS if (start, start + 4) not in tokens]
    if missing or totals["files"] <= 0 or totals["bytes"] <= 0:
        raise AuditError(f"consumed GPIC window proof missing mode={mode} rank={rank}: {missing}")
    summary = {
        "rank": rank,
        "records": len(records),
        "windows": sum(record.get("record_type") == "window" for record in records),
        "gpic_windows": windows,
        "gpic_window_count": len(windows),
        "gpic_files": totals["files"],
        "gpic_bytes": totals["bytes"],
        "fstat_calls": totals["fstat_calls"],
        "path_stat_calls": totals["path_stat_calls"],
        "path_stat_thread_sum_s": path_sum,
        "path_stat_operation_max_s": path_max,
        "required_consumed_window_starts": list(REQUIRED_GPIC_STARTS),
        "closure_zero_loss": True,
    }
    return identity, closure, summary


def audit_state(
    root: Path,
    run: Mapping[str, Any],
    *,
    mode: str,
    state: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _state_raw, value = strict_json(root / f"state.{mode}.{state}.json")
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
        or value.get("schema_version") != STATE_SCHEMA
        or value.get("jobtag") != run["jobtag"]
        or value.get("run_nonce") != run["run_nonce"]
        or value.get("worker_index") != MODES.index(mode)
        or value.get("mode") != mode
        or value.get("telemetry_state") != state
        or value.get("telemetry_enabled") is not enabled
        or value.get("config") != run["config_snapshots"][mode]
        or value.get("claim")
        != file_evidence(
            root / f"claim.arm.{mode}.{state}.json",
            regular_bytes(root / f"claim.arm.{mode}.{state}.json"),
        )
        or not isinstance(value.get("host"), str)
        or SAFE_TOKEN(value["host"]) is None
    ):
        raise AuditError(f"state identity mismatch mode={mode} state={state}")
    process = value.get("fresh_process")
    expected_state_token = digest(
        f"{run['run_nonce']}\0{mode}\0{state}".encode("utf-8")
    )
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
        or value.get("attempt_state_token") != expected_state_token
    ):
        raise AuditError(f"fresh process attestation invalid mode={mode} state={state}")
    log = value.get("log")
    if not isinstance(log, dict):
        raise AuditError(f"log evidence missing mode={mode} state={state}")
    log_path = Path(str(log.get("path", "")))
    log_raw = regular_bytes(log_path)
    if len(log_raw) != log.get("bytes") or digest(log_raw) != log.get("sha256"):
        raise AuditError(f"log byte evidence mismatch mode={mode} state={state}")
    smoke = parse_smoke_log(log_raw, log_path)
    if smoke != value.get("smoke"):
        raise AuditError(f"embedded smoke differs from log mode={mode} state={state}")
    runtime_identities = parse_runtime_identities(
        log_raw, log_path, mode, expected_state_token
    )
    if value.get("runtime_identities") != runtime_identities:
        raise AuditError(f"embedded runtime identity differs from log mode={mode} state={state}")
    per_rank_bs = smoke.get("per_rank_bs")
    expected_items = RANKS * STEPS * per_rank_bs if type(per_rank_bs) is int else -1
    if (
        smoke.get("status") != "PASS"
        or smoke.get("ranks") != RANKS
        or smoke.get("steps") != STEPS
        or smoke.get("start_step") != START_STEP
        or smoke.get("plan_hash") != run["bindings"]["plan_header"]["plan_hash"]
        or smoke.get("catalog_hash") != run["bindings"]["catalog_header"]["catalog_hash"]
        or smoke.get("assignment_hash")
        != run["bindings"]["catalog_header"]["assignment_hash"]
        or smoke.get("assertions", {}).get("item_trace_exact") != expected_items
        or set(smoke.get("per_rank", {})) != {str(rank) for rank in range(RANKS)}
    ):
        raise AuditError(f"smoke exact-plan evidence mismatch mode={mode} state={state}")
    for rank in (str(value) for value in range(RANKS)):
        for key in HASH_KEYS:
            validate_hex(smoke["per_rank"][rank].get(key), f"{mode}/{state}/{rank}/{key}")

    telemetry_summary: dict[str, Any] = {"enabled": False}
    if not enabled:
        if value.get("telemetry") != {"status": "OFF", "enabled": False, "rank_files": 0}:
            raise AuditError(f"telemetry-off evidence drift mode={mode}")
    else:
        telemetry = value.get("telemetry")
        expected_root = root / f"telemetry.{mode}.on"
        manifest_path = expected_root / "manifest.json"
        manifest_raw, manifest = strict_json(manifest_path)
        if (
            not isinstance(telemetry, dict)
            or set(telemetry)
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
            or telemetry.get("schema_version") != TELEMETRY_MANIFEST_SCHEMA
            or telemetry.get("manifest") != str(manifest_path)
            or telemetry.get("manifest_bytes") != len(manifest_raw)
            or telemetry.get("manifest_sha256") != digest(manifest_raw)
            or telemetry.get("shared_root") != str(expected_root)
            or telemetry.get("rank_files") != RANKS
            or telemetry.get("gpic_consumption_proven_per_rank") is not True
            or manifest.get("status") != "PASS"
            or manifest.get("schema_version") != TELEMETRY_MANIFEST_SCHEMA
            or manifest.get("jobtag") != run["jobtag"]
            or manifest.get("run_nonce") != run["run_nonce"]
            or manifest.get("host") != value["host"]
            or manifest.get("mode") != mode
            or manifest.get("state") != "on"
            or manifest.get("rank_files") != RANKS
            or manifest.get("shared_root") != str(expected_root)
            or manifest.get("files") != telemetry.get("files")
        ):
            raise AuditError(f"telemetry manifest identity mismatch mode={mode}")
        real_directory(expected_root, f"shared telemetry {mode}")
        expected_entries = {"manifest.json"} | {
            f"io_profile.rank{rank:04d}.jsonl" for rank in range(RANKS)
        }
        if ACTIVE_SOURCE_FREEZE is None or ACTIVE_SOURCE_FREEZE.child_file_names(
            expected_root
        ) != expected_entries:
            raise AuditError(f"telemetry file set mismatch mode={mode}")
        if len(manifest.get("files", [])) != RANKS:
            raise AuditError(f"telemetry rank evidence count mismatch mode={mode}")
        pids: set[int] = set()
        summaries = []
        local_root = Path(str(manifest.get("local_root", "")))
        for rank, evidence in enumerate(manifest["files"]):
            shared = expected_root / f"io_profile.rank{rank:04d}.jsonl"
            raw = regular_bytes(shared)
            identity, closure, summary = audit_telemetry_rank(
                raw,
                rank=rank,
                mode=mode,
                host=value["host"],
                local_root=local_root,
                run=run,
            )
            if identity["pid"] in pids:
                raise AuditError(f"duplicate rank PID in telemetry mode={mode}")
            pids.add(identity["pid"])
            expected_evidence = {
                "rank": rank,
                "local_path": str(local_root / shared.name),
                "shared_path": str(shared),
                "bytes": len(raw),
                "sha256": digest(raw),
                "identity": identity,
                "closure": closure,
                "summary": summary,
            }
            if evidence != expected_evidence:
                raise AuditError(f"telemetry file evidence mismatch mode={mode} rank={rank}")
            summaries.append(summary)
        if {
            evidence["rank"]: evidence["identity"]["pid"]
            for evidence in manifest["files"]
        } != {item["rank"]: item["pid"] for item in runtime_identities}:
            raise AuditError(f"runtime/telemetry PID identity mismatch mode={mode}")
        telemetry_summary = {
            "enabled": True,
            "manifest": str(manifest_path),
            "manifest_sha256": digest(manifest_raw),
            "rank_summaries": summaries,
            "distinct_rank_processes": RANKS,
        }
    return value, telemetry_summary


def audit_negative(
    root: Path, run: Mapping[str, Any]
) -> tuple[dict[str, int], dict[str, Any], dict[str, Any]]:
    raw, value = strict_json(root / "negative_contract.json")
    claim_raw, claim = strict_json(root / "claim.negative.json")
    claim_evidence = file_evidence(root / "claim.negative.json", claim_raw)
    negative_evidence = {
        "path": str(root / "negative_contract.json"),
        "bytes": len(raw),
        "sha256": digest(raw),
    }
    expected_summary = {
        "normal_success": 4,
        "short_read_rejected": 4,
        "in_place_rejected": 4,
        "namespace_strict_rejected": 2,
        "namespace_fd_old_inode": 2,
        "invalid_mode_rejected": 5,
    }
    cases = value.get("cases")
    if (
        set(value)
        != {
            "status",
            "schema_version",
            "jobtag",
            "run_nonce",
            "worker0_host",
            "claim",
            "case_count",
            "summary",
            "cases",
        }
        or value.get("status") != "PASS"
        or value.get("schema_version") != NEGATIVE_SCHEMA
        or value.get("jobtag") != run["jobtag"]
        or value.get("run_nonce") != run["run_nonce"]
        or value.get("worker0_host") != run["worker0_host"]
        or value.get("claim") != claim_evidence
        or value.get("summary") != expected_summary
        or value.get("case_count") != sum(expected_summary.values())
        or not isinstance(cases, list)
        or len(cases) != sum(expected_summary.values())
        or set(claim)
        != {
            "schema_version",
            "jobtag",
            "run_nonce",
            "worker_index",
            "host",
            "pid",
            "created_time_ns",
        }
        or claim.get("schema_version") != NEGATIVE_CLAIM_SCHEMA
        or claim.get("jobtag") != run["jobtag"]
        or claim.get("run_nonce") != run["run_nonce"]
        or claim.get("worker_index") != 0
        or claim.get("host") != run["worker0_host"]
        or type(claim.get("pid")) is not int
        or claim["pid"] <= 0
        or type(claim.get("created_time_ns")) is not int
    ):
        raise AuditError("negative contract identity/summary mismatch")
    by_name = {case.get("case"): case for case in cases if isinstance(case, dict)}
    if len(by_name) != len(cases):
        raise AuditError("negative contract case names are missing/duplicate")
    expected_normal_payloads = {
        "0:0": digest(b"gpic-normal-a\x00\xff"),
        "0:1": digest(("GPIC/fd/语义/🌙".encode("utf-8") + b"\x00") * 11),
    }
    expected_old_namespace = digest(b"namespace-opened-old-inode\x00")
    expected_new_namespace = digest(b"namespace-visible-new-inode\xff")
    for mode in MODES:
        for telemetry, suffix in ((False, "plain"), (True, "instrumented")):
            normal = by_name.get(f"normal.{mode}.{suffix}")
            short = by_name.get(f"short_read.{mode}.{suffix}")
            in_place = by_name.get(f"in_place.{mode}.{suffix}")
            namespace = by_name.get(f"namespace.{mode}.{suffix}")
            expected_path = 4 if mode == "strict_path" else 0
            if (
                not isinstance(normal, dict)
                or normal.get("outcome") != "PASS"
                or normal.get("mode") != mode
                or normal.get("telemetry") is not telemetry
                or normal.get("files") != 2
                or normal.get("workers") != (2 if telemetry else None)
                or normal.get("fstat_calls") != 4
                or normal.get("path_stat_calls") != expected_path
                or normal.get("path_stat_timing_zero")
                is not (mode == "fd_stable" and telemetry)
                or set(normal.get("payload_sha256", {})) != {"0:0", "0:1"}
                or normal.get("payload_sha256") != expected_normal_payloads
                or not all(
                    HEX64(str(item or ""))
                    for item in normal.get("payload_sha256", {}).values()
                )
            ):
                raise AuditError(f"negative normal case mismatch mode={mode} suffix={suffix}")
            for label, case, category, substring in (
                ("short", short, "short_read", "short gpic source read"),
                (
                    "in_place",
                    in_place,
                    "in_place_identity_change",
                    "gpic source changed while reading",
                ),
            ):
                if (
                    not isinstance(case, dict)
                    or case.get("outcome") != "EXPECTED_REJECTION"
                    or case.get("error_type") != "OfflineLoaderError"
                    or case.get("mode") != mode
                    or case.get("telemetry") is not telemetry
                    or case.get("error_category") != category
                    or case.get("required_substring") != substring
                    or HEX64(str(case.get("error_sha256", ""))) is None
                ):
                    raise AuditError(f"negative {label} case mismatch mode={mode} suffix={suffix}")
            if mode == "strict_path":
                if (
                    not isinstance(namespace, dict)
                    or namespace.get("outcome") != "EXPECTED_REJECTION"
                    or namespace.get("error_type") != "OfflineLoaderError"
                    or namespace.get("mode") != mode
                    or namespace.get("telemetry") is not telemetry
                    or namespace.get("error_category") != "namespace_replacement"
                    or namespace.get("required_substring")
                    != "gpic source changed while reading"
                    or HEX64(str(namespace.get("error_sha256", ""))) is None
                ):
                    raise AuditError(f"strict namespace case mismatch suffix={suffix}")
            else:
                if (
                    not isinstance(namespace, dict)
                    or namespace.get("outcome") != "PASS_OLD_OPENED_INODE"
                    or namespace.get("mode") != mode
                    or namespace.get("telemetry") is not telemetry
                    or namespace.get("payload_sha256")
                    != namespace.get("expected_old_sha256")
                    or namespace.get("payload_sha256") != expected_old_namespace
                    or namespace.get("fstat_calls") != 2
                    or namespace.get("path_stat_calls") != 0
                    or namespace.get("workers") != (1 if telemetry else None)
                ):
                    raise AuditError(f"fd namespace case mismatch suffix={suffix}")
            if (
                not isinstance(namespace, dict)
                or namespace.get("visible_after_sha256") != expected_new_namespace
            ):
                raise AuditError(f"namespace visibility evidence missing mode={mode} suffix={suffix}")
    invalid_inputs = ("", "STRICT_PATH", "fd", True, None)
    for index, invalid_value in enumerate(invalid_inputs):
        case = by_name.get(f"invalid_mode.{index}")
        if (
            not isinstance(case, dict)
            or case.get("outcome") != "EXPECTED_REJECTION"
            or case.get("error_type") != "OfflineLoaderError"
            or case.get("error_category") != "invalid_mode"
            or case.get("required_substring")
            != "gpic_source_consistency_mode must be exactly one of"
            or case.get("input_repr") != repr(invalid_value)
        ):
            raise AuditError(f"invalid-mode rejection mismatch index={index}")
    # Plain/instrumented and strict/fd normal fixtures are byte-identical.
    normal_hashes = [
        by_name[f"normal.{mode}.{suffix}"]["payload_sha256"]
        for mode in MODES
        for suffix in ("plain", "instrumented")
    ]
    if normal_hashes[1:] != normal_hashes[:-1]:
        raise AuditError("negative normal payload bytes differ across four arms")
    return expected_summary, claim_evidence, negative_evidence


def run(args: argparse.Namespace) -> None:
    global ACTIVE_SOURCE_FREEZE
    if (
        SAFE_TOKEN(str(args.audit_host or "")) is None
        or args.output.name != "independent_audit.json"
        or args.output.parent.name != args.audit_host
        or args.source_root.name != args.source_jobtag
    ):
        raise AuditError("audit/source attempt namespace identity mismatch")
    try:
        Path(os.path.abspath(args.output)).relative_to(
            Path(os.path.abspath(args.source_root))
        )
    except ValueError:
        pass
    else:
        raise AuditError("audit output must not be inside the frozen source tree")
    auditor_path = Path(__file__).resolve(strict=True)
    auditor_raw = regular_bytes(auditor_path)
    auditor_identity = metadata_identity(auditor_path.lstat())
    audit_job_path = auditor_path.parents[2] / "jobs/offline_arb_gpic_fd_semantic_independent_audit.yaml"
    audit_job_raw = regular_bytes(audit_job_path)
    audit_job_identity = metadata_identity(audit_job_path.lstat())
    real_directory(args.source_root, "source attempt root")
    if ACTIVE_SOURCE_FREEZE is not None:
        raise AuditError("source tree freeze was already initialized")
    source_freeze = SourceTreeFreeze(args.source_root)
    ACTIVE_SOURCE_FREEZE = source_freeze
    run_raw, run = strict_json(args.source_root / "run.json")
    aggregate_raw, aggregate = strict_json(args.source_root / "aggregate.json")
    if (
        run.get("schema_version") != RUN_SCHEMA
        or run.get("jobtag") != args.source_jobtag
        or SAFE_TOKEN(args.source_jobtag) is None
        or HEX64(str(run.get("run_nonce", ""))) is None
        or run.get("modes") != list(MODES)
        or run.get("states") != list(STATES)
        or run.get("hash_keys") != list(HASH_KEYS)
        or run.get("ranks") != RANKS
        or run.get("steps") != STEPS
        or run.get("start_step") != START_STEP
        or run.get("prefetch_steps") != PREFETCH_STEPS
        or run.get("gpic_threads_per_node") != GPIC_THREADS
    ):
        raise AuditError("run identity/schema/matrix mismatch")
    host_hints: dict[str, str] = {}
    for mode in MODES:
        _off_raw, off_hint = strict_json(args.source_root / f"state.{mode}.off.json")
        _on_raw, on_hint = strict_json(args.source_root / f"state.{mode}.on.json")
        host = off_hint.get("host")
        if (
            not isinstance(host, str)
            or SAFE_TOKEN(host) is None
            or on_hint.get("host") != host
        ):
            raise AuditError(f"cannot derive exact-set host for mode={mode}")
        host_hints[mode] = host
        if host != f"{run['jobtag']}-{MODES.index(mode)}":
            raise AuditError(f"source host/jobtag/worker mapping drift mode={mode}")
    if run.get("worker0_host") != host_hints["strict_path"]:
        raise AuditError("run worker0 host differs from strict_path worker host")
    source_freeze.assert_exact(
        expected_source_files(host_hints), expected_source_directories()
    )
    run_claim_evidence = audit_run_claim(args.source_root, run)
    aggregate_claim_raw, aggregate_claim = strict_json(
        args.source_root / "claim.aggregate.json"
    )
    aggregate_claim_evidence = file_evidence(
        args.source_root / "claim.aggregate.json", aggregate_claim_raw
    )
    if (
        set(aggregate_claim)
        != {
            "schema_version",
            "jobtag",
            "run_nonce",
            "worker_index",
            "host",
            "pid",
            "created_time_ns",
        }
        or aggregate_claim.get("schema_version") != AGGREGATE_CLAIM_SCHEMA
        or aggregate_claim.get("jobtag") != run["jobtag"]
        or aggregate_claim.get("run_nonce") != run["run_nonce"]
        or aggregate_claim.get("worker_index") != 0
        or aggregate_claim.get("host") != run["worker0_host"]
        or type(aggregate_claim.get("pid")) is not int
        or aggregate_claim["pid"] <= 0
        or type(aggregate_claim.get("created_time_ns")) is not int
    ):
        raise AuditError("aggregate O_EXCL claim mismatch")
    config_shas = audit_configs(args.source_root, run)
    runtime_shas = audit_runtime_sources(args.source_root, run)
    binding_shas = audit_input_bindings(args.source_root, run)

    states: dict[str, dict[str, dict[str, Any]]] = {}
    telemetry_summary: dict[str, Any] = {}
    arm_claim_evidences: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        states[mode] = {}
        for state in STATES:
            value, telemetry = audit_state(
                args.source_root, run, mode=mode, state=state
            )
            states[mode][state] = value
            if state == "on":
                telemetry_summary[mode] = telemetry
        if (
            states[mode]["off"]["fresh_process"]["pid"],
            states[mode]["off"]["fresh_process"]["proc_starttime_ticks"],
        ) == (
            states[mode]["on"]["fresh_process"]["pid"],
            states[mode]["on"]["fresh_process"]["proc_starttime_ticks"],
        ):
            raise AuditError(f"telemetry states reused process mode={mode}")
        _arm_raw, arm = strict_json(args.source_root / f"arm.{mode}.json")
        mode_claim_evidences: dict[str, dict[str, Any]] = {}
        for state in STATES:
            claim_path = args.source_root / f"claim.arm.{mode}.{state}.json"
            claim_raw, claim = strict_json(claim_path)
            mode_claim_evidences[state] = file_evidence(claim_path, claim_raw)
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
                or claim.get("schema_version") != ARM_CLAIM_SCHEMA
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
                raise AuditError(
                    f"arm claim mismatch mode={mode} state={state}"
                )
        expected_state_files = {
            state: str(args.source_root / f"state.{mode}.{state}.json") for state in STATES
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
            or arm.get("schema_version") != ARM_SCHEMA
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
            or arm.get("claims") != mode_claim_evidences
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
            raise AuditError(f"arm identity mismatch mode={mode}")
        arm_claim_evidences[mode] = mode_claim_evidences

    baseline = states["strict_path"]["off"]["smoke"]
    exact_hashes: dict[str, dict[str, str]] = {}
    for mode in MODES:
        for state in STATES:
            smoke = states[mode][state]["smoke"]
            for key in SMOKE_IDENTITY_KEYS:
                if smoke.get(key) != baseline.get(key):
                    raise AuditError(f"smoke identity differs mode={mode} state={state} key={key}")
    for rank in (str(value) for value in range(RANKS)):
        exact_hashes[rank] = {}
        for key in HASH_KEYS:
            hashes = [
                states[mode][state]["smoke"]["per_rank"][rank][key]
                for mode in MODES
                for state in STATES
            ]
            if len(set(hashes)) != 1:
                raise AuditError(f"four-arm exact semantic hash drift rank={rank} key={key}")
            exact_hashes[rank][key] = hashes[0]
    negative_summary, negative_claim_evidence, negative_file_evidence = audit_negative(
        args.source_root, run
    )
    capture_manifest_evidence = audit_capture_manifest(
        args.source_root,
        run,
        aggregate,
        host_hints,
        aggregate_claim_evidence,
    )

    expected_telemetry_contract = {
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
    }
    expected_runtime_contract = {
        "arms_checked": 4,
        "ranks_per_arm": RANKS,
        "gpic_threads_per_node": GPIC_THREADS,
        "gpic_workers_per_rank": GPIC_THREADS // RANKS,
        "gpic_payload_in_memory": True,
        "mode_matches_arm": True,
    }
    expected_negative_contract = {
        **negative_file_evidence,
        "claim": negative_claim_evidence,
        "summary": negative_summary,
    }
    expected_claims = {
        "run": run_claim_evidence,
        "arms": arm_claim_evidences,
        "negative": negative_claim_evidence,
        "aggregate": aggregate_claim_evidence,
    }
    expected_arm_files = {
        mode: str(args.source_root / f"arm.{mode}.json") for mode in MODES
    }
    expected_interpretation = (
        "On the fixed 4-rank/12-step plan prefix, strict_path and fd_stable, "
        "with telemetry both disabled and enabled in fresh processes, emitted "
        "identical trace, prompt, pixel, and cfg_keep hashes. Telemetry proves "
        "the consumed GPIC windows used the selected mode and exact syscall contract."
    )
    expected_aggregate_keys = {
        "status",
        "schema_version",
        "jobtag",
        "run_nonce",
        "modes",
        "telemetry_states",
        "four_fresh_process_arms",
        "config_only_delta",
        "ranks",
        "steps",
        "start_step",
        "hash_keys",
        "per_rank_exact_hashes",
        "plain_instrumented_exact_per_mode",
        "strict_fd_exact",
        "runtime_identity_contract",
        "telemetry_contract",
        "negative_contract",
        "claims",
        "capture_manifest",
        "artifact_tree_contract",
        "sidecar_release_unchanged",
        "runtime_sources_unchanged",
        "arm_files",
        "interpretation",
    }
    if (
        set(aggregate) != expected_aggregate_keys
        or aggregate.get("status") != "PASS"
        or aggregate.get("schema_version") != AGGREGATE_SCHEMA
        or aggregate.get("jobtag") != run["jobtag"]
        or aggregate.get("run_nonce") != run["run_nonce"]
        or aggregate.get("modes") != list(MODES)
        or aggregate.get("telemetry_states") != list(STATES)
        or aggregate.get("four_fresh_process_arms") is not True
        or aggregate.get("config_only_delta")
        != "dataset.gpic_source_consistency_mode"
        or aggregate.get("ranks") != RANKS
        or aggregate.get("steps") != STEPS
        or aggregate.get("start_step") != START_STEP
        or aggregate.get("hash_keys") != list(HASH_KEYS)
        or aggregate.get("per_rank_exact_hashes") != exact_hashes
        or aggregate.get("plain_instrumented_exact_per_mode") is not True
        or aggregate.get("strict_fd_exact") is not True
        or aggregate.get("runtime_identity_contract") != expected_runtime_contract
        or aggregate.get("telemetry_contract") != expected_telemetry_contract
        or aggregate.get("negative_contract") != expected_negative_contract
        or aggregate.get("claims") != expected_claims
        or aggregate.get("capture_manifest") != capture_manifest_evidence
        or aggregate.get("sidecar_release_unchanged") is not True
        or aggregate.get("runtime_sources_unchanged") is not True
        or aggregate.get("arm_files") != expected_arm_files
        or aggregate.get("interpretation") != expected_interpretation
    ):
        raise AuditError("aggregate differs from independent recomputation")

    audit = {
        "status": "PASS",
        "schema_version": AUDIT_SCHEMA,
        "audit_attempt_host": args.audit_host,
        "source_jobtag": run["jobtag"],
        "source_run_nonce": run["run_nonce"],
        "source_root": str(args.source_root),
        "audit_runtime": {
            "script_path": str(auditor_path),
            "script_bytes": len(auditor_raw),
            "script_sha256": digest(auditor_raw),
            "script_fstat_identity": auditor_identity,
            "job_path": str(audit_job_path),
            "job_bytes": len(audit_job_raw),
            "job_sha256": digest(audit_job_raw),
            "job_fstat_identity": audit_job_identity,
        },
        "source_run": {
            "bytes": len(run_raw),
            "sha256": digest(run_raw),
        },
        "source_aggregate": {
            "bytes": len(aggregate_raw),
            "sha256": digest(aggregate_raw),
        },
        "source_tree_freeze": {
            "file_count": len(source_freeze.files),
            "directory_count": len(source_freeze.directories),
            "exact_file_set": sorted(source_freeze.files),
            "exact_directory_set": sorted(source_freeze.directories),
            "bytes_hash_fstat_identity_frozen": True,
            "final_rewalk_rehash_required": True,
        },
        "claims": expected_claims,
        "capture_manifest": capture_manifest_evidence,
        "config_snapshot_sha256": config_shas,
        "runtime_source_sha256": runtime_shas,
        "input_snapshot_sha256": binding_shas,
        "four_fresh_process_arms": True,
        "same_plan_rank_step_slot": {
            "plan_id": baseline["plan_id"],
            "plan_hash": baseline["plan_hash"],
            "assignment_hash": baseline["assignment_hash"],
            "ranks": RANKS,
            "start_step": START_STEP,
            "steps": STEPS,
            "per_rank_bs": baseline["per_rank_bs"],
            "item_trace_exact_per_arm": RANKS * STEPS * baseline["per_rank_bs"],
        },
        "per_rank_exact_hashes": exact_hashes,
        "matched_hash_keys": list(HASH_KEYS),
        "telemetry": telemetry_summary,
        "runtime_identity_contract": expected_runtime_contract,
        "telemetry_contract": expected_telemetry_contract,
        "negative_contract_summary": negative_summary,
        "verdict": "PASS",
    }
    source_freeze.verify_unchanged()
    if (
        regular_bytes(auditor_path) != auditor_raw
        or metadata_identity(auditor_path.lstat()) != auditor_identity
        or regular_bytes(audit_job_path) != audit_job_raw
        or metadata_identity(audit_job_path.lstat()) != audit_job_identity
    ):
        raise AuditError("independent auditor source/job changed during audit")
    atomic_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"GPIC_FD_SEMANTIC_INDEPENDENT_AUDIT_PASS result={args.output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-jobtag", required=True)
    parser.add_argument("--audit-host", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except AuditError as exc:
        print(f"FATAL independent GPIC fd semantic audit: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
