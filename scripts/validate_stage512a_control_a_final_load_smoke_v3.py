#!/usr/bin/env python3
"""Fail-closed validator for the sealed A-final -> A-control smoke-v3 receipt.

This validator does not import training code and does not trust values inferred
from the report itself.  All external authority identities are supplied on the
command line.  Dynamic PyTorch scheduler/optimizer dictionaries are permitted
only inside their explicitly designated fields; their boot-critical semantics
are still checked here.
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
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn


SCHEMA = "stage512a_control_a_final_load_smoke_v3"
RENDER_SCHEMA = "stage512a_control_a_final_load_smoke_v3_render_v1"
CONFIG_REL = "configs/stage512a_continue_control_rngfix_v1.yaml"
PROBE_REL = "scripts/stage512a_control_a_final_load_smoke_v3.py"
VALIDATOR_REL = "scripts/validate_stage512a_control_a_final_load_smoke_v3.py"
TEMPLATE_REL = (
    "provenance/jcc_jobs/20260809_stage512a_control_a_final_load_smoke_v3.yaml"
)
RENDERER_REL = "scripts/render_stage512a_control_a_final_load_smoke_v3.py"
TEST_REL = "scripts/test_stage512a_control_a_final_load_smoke_v3.py"
RENDER_SCHEMA_REL = (
    "provenance/stage512a_control_a_final_load_smoke_v3.schema.json"
)
REPORT_NAME = "REPORT.json"
SUCCESS_NAME = "_SUCCESS"
EXPECTED_WORLD_SIZE = 2
EXPECTED_TRAINABLE_COUNT = 483
EXPECTED_OPTIMIZER_STEP = 291667
EXPECTED_SOURCE_STEP = 125000
EXPECTED_BASE_LR = 1.732051e-4
EXPECTED_FIRST_OPTIMIZER_UPDATE_LR = EXPECTED_BASE_LR / 500.0
EXPECTED_STEP1_LOGGED_LR = EXPECTED_BASE_LR * 2.0 / 500.0
EXPECTED_CHECKPOINT_BYTES = 26_328_353_144
EXPECTED_CHECKPOINT_SHA256 = (
    "957bfcf12b89c4042a32056566012a2451e72aa0b2706b0ca51808ab79155c31"
)
EXPECTED_KEYSET_SHA256 = (
    "6c61b43d30a8bde2f13fc8a91e30a8a0de271934ba5c3fd72df392cd027a4424"
)
EXPECTED_ABLATION_FINGERPRINT_SHA256 = (
    "bc452da33095f73230c05c9eeee2e0d027295bd48813ea69f05bc1c159b69895"
)
EXPECTED_LEGACY_TOP_LEVEL_KEYS = [
    "ablation_fp",
    "ema",
    "global_step",
    "optimizer",
    "scheduler",
    "trainable_state",
]
EXPECTED_NEGATIVE_CONTROL_NEEDLES = {
    "ordinary_current_only_trainable": "trainable-key closure mismatch",
    "optimizer_missing_fqn_state": "state coverage mismatch",
    "optimizer_extra_param_group": "param-group count mismatch",
    "ema_missing_fqn": "EMA key closure mismatch",
}
EXPECTED_CORE_SOURCE_KEYS = {
    "train.py",
    "training/checkpoint.py",
    "training/rng_state.py",
    "training/formal_boot.py",
    "training/config_schema.py",
    "networks/model_mot.py",
    "networks/vae.py",
}
EXPECTED_CORE_SOURCE_ARGUMENTS = {
    "train.py": "expected_train_py_git_blob_raw_sha256",
    "training/checkpoint.py": (
        "expected_training_checkpoint_py_git_blob_raw_sha256"
    ),
    "training/rng_state.py": "expected_training_rng_state_py_git_blob_raw_sha256",
    "training/formal_boot.py": (
        "expected_training_formal_boot_py_git_blob_raw_sha256"
    ),
    "training/config_schema.py": (
        "expected_training_config_schema_py_git_blob_raw_sha256"
    ),
    "networks/model_mot.py": (
        "expected_networks_model_mot_py_git_blob_raw_sha256"
    ),
    "networks/vae.py": "expected_networks_vae_py_git_blob_raw_sha256",
}
RUNTIME_GIT_BLOB_PATHS = set(EXPECTED_CORE_SOURCE_KEYS) | {
    CONFIG_REL,
    PROBE_REL,
    VALIDATOR_REL,
    TEMPLATE_REL,
}
DEDICATED_GIT_BLOB_PATHS = RUNTIME_GIT_BLOB_PATHS | {
    RENDERER_REL,
    TEST_REL,
    RENDER_SCHEMA_REL,
}
TOP_LEVEL_KEYS = {
    "schema",
    "status",
    "verdict",
    "authority",
    "tested_final_git_blob_raw_sha256",
    "checkpoint",
    "world_size",
    "boot",
    "closure",
    "scheduler",
    "rng",
    "negative_controls",
    "elapsed_s",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
DEVICE_INODE_RE = re.compile(r"([1-9][0-9]*):([1-9][0-9]*)\Z")
DTYPE_RE = re.compile(r"torch\.[A-Za-z0-9_]+\Z")
MAX_REPORT_BYTES = 4 << 20


class ReceiptValidationError(RuntimeError):
    """A stable, user-facing validation failure."""


def fail(message: str) -> NoReturn:
    raise ReceiptValidationError(message)


def exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if type(value) is not dict:
        fail(f"{where}: expected object, got {type(value).__name__}")
    actual = set(value)
    if actual != expected:
        fail(
            f"{where}: key closure mismatch; "
            f"missing={sorted(expected - actual)!r}, unknown={sorted(actual - expected)!r}"
        )
    return value


def require_exact(value: Any, expected: Any, where: str) -> None:
    if type(value) is not type(expected) or value != expected:
        fail(f"{where}: expected {expected!r}, got {value!r}")


def require_int(value: Any, expected: int, where: str) -> None:
    if type(value) is not int or value != expected:
        fail(f"{where}: expected integer {expected}, got {value!r}")


def require_true(value: Any, where: str) -> None:
    if value is not True:
        fail(f"{where}: expected true, got {value!r}")


def require_false(value: Any, where: str) -> None:
    if value is not False:
        fail(f"{where}: expected false, got {value!r}")


def require_sha256(value: Any, where: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        fail(f"{where}: expected one lowercase SHA-256, got {value!r}")
    if value == "0" * 64:
        fail(f"{where}: all-zero SHA-256 is forbidden")
    return value


def require_commit(value: Any, where: str) -> str:
    if type(value) is not str or GIT_COMMIT_RE.fullmatch(value) is None:
        fail(f"{where}: expected a full lowercase Git object id, got {value!r}")
    if set(value) == {"0"}:
        fail(f"{where}: all-zero Git object id is forbidden")
    return value


def require_abs_posix_path(value: Any, where: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        fail(f"{where}: expected a nonempty absolute POSIX path")
    if not value.startswith("/") or value.startswith("//"):
        fail(f"{where}: expected a single-root absolute POSIX path, got {value!r}")
    if str(PurePosixPath(value)) != value:
        fail(f"{where}: path is not lexically canonical, got {value!r}")
    return value


def expected_final_git_blob_raw_sha256(
    args: argparse.Namespace,
) -> dict[str, str]:
    expected = {
        relative_path: getattr(args, argument_attribute)
        for relative_path, argument_attribute in EXPECTED_CORE_SOURCE_ARGUMENTS.items()
    }
    release_repo = PurePosixPath(args.expected_release_repo)
    asset_bindings = (
        ("config", args.expected_config_path, args.expected_config_sha256),
        ("probe", args.expected_probe_path, args.expected_probe_sha256),
        ("validator", args.expected_validator_path, args.expected_validator_sha256),
        (
            "job template",
            args.expected_job_template_path,
            args.expected_job_template_sha256,
        ),
    )
    for label, absolute_path, expected_sha256 in asset_bindings:
        try:
            relative_path = PurePosixPath(absolute_path).relative_to(release_repo)
        except ValueError as exc:
            fail(
                f"CLI expected {label} path must be inside the final release repo: "
                f"{absolute_path!r}"
            )
        rendered = str(relative_path)
        if not rendered or rendered == "." or rendered.startswith("../"):
            fail(f"CLI expected {label} path is not a release file: {absolute_path!r}")
        if rendered in expected:
            fail(f"CLI final Git-blob map has duplicate path {rendered!r}")
        expected[rendered] = expected_sha256
    if len(expected) != 11 or not EXPECTED_CORE_SOURCE_KEYS.issubset(expected):
        fail(
            "CLI final Git-blob map closure is not exactly seven core sources plus "
            "config/probe/validator/template"
        )
    return expected


def require_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{where}: expected a finite number, got {value!r}")
    rendered = float(value)
    if not math.isfinite(rendered):
        fail(f"{where}: expected a finite number, got {value!r}")
    return rendered


def require_number_equal(value: Any, expected: float, where: str) -> None:
    actual = require_number(value, where)
    if actual != float(expected):
        fail(f"{where}: expected {expected!r}, got {value!r}")


def require_close(value: Any, expected: float, where: str) -> None:
    actual = require_number(value, where)
    tolerance = max(1e-18, abs(expected) * 1e-12)
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=tolerance):
        fail(f"{where}: expected {expected!r} within {tolerance:g}, got {actual!r}")


def require_single_number_list(value: Any, where: str) -> float:
    if type(value) is not list or len(value) != 1:
        fail(f"{where}: expected a one-element numeric list, got {value!r}")
    return require_number(value[0], f"{where}[0]")


def reject_nonfinite_json(value: Any, where: str = "report") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            fail(f"{where}: non-finite JSON number is forbidden")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            reject_nonfinite_json(item, f"{where}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                fail(f"{where}: non-string JSON object key is forbidden")
            reject_nonfinite_json(item, f"{where}.{key}")
        return
    fail(f"{where}: unsupported JSON value type {type(value).__name__}")


def no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"JSON duplicate key is forbidden: {key!r}")
        result[key] = value
    return result


def reject_json_constant(token: str) -> NoReturn:
    fail(f"non-standard JSON constant is forbidden: {token}")


def read_fd_limited(fd: int, maximum: int, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        block = os.read(fd, min(1 << 20, maximum + 1 - total))
        if not block:
            break
        chunks.append(block)
        total += len(block)
        if total > maximum:
            fail(f"{label}: exceeds maximum allowed size {maximum}")
    return b"".join(chunks)


def read_sealed_receipt(report_dir: str) -> tuple[bytes, bytes]:
    absolute = os.path.abspath(report_dir)
    try:
        before = os.lstat(absolute)
    except OSError as exc:
        fail(f"report directory lstat failed: {exc}")
    if not stat.S_ISDIR(before.st_mode):
        fail("report directory must be an ordinary directory, not a symlink or file")
    if stat.S_IMODE(before.st_mode) != 0o555:
        fail(
            "report directory mode must be exactly 0555, got "
            f"{stat.S_IMODE(before.st_mode):04o}"
        )

    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        directory_fd = os.open(absolute, directory_flags)
    except OSError as exc:
        fail(f"report directory secure open failed: {exc}")

    try:
        opened = os.fstat(directory_fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            fail("report directory changed between lstat and open")
        entries = set(os.listdir(directory_fd))
        expected_entries = {REPORT_NAME, SUCCESS_NAME}
        if entries != expected_entries:
            fail(
                "report directory entry closure mismatch; "
                f"missing={sorted(expected_entries - entries)!r}, "
                f"unknown={sorted(entries - expected_entries)!r}"
            )

        payloads: dict[str, bytes] = {}
        file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        for name, maximum in ((REPORT_NAME, MAX_REPORT_BYTES), (SUCCESS_NAME, 65)):
            try:
                fd = os.open(name, file_flags, dir_fd=directory_fd)
            except OSError as exc:
                fail(f"{name}: secure open failed: {exc}")
            try:
                metadata = os.fstat(fd)
                if not stat.S_ISREG(metadata.st_mode):
                    fail(f"{name}: must be an ordinary regular file")
                if stat.S_IMODE(metadata.st_mode) != 0o444:
                    fail(
                        f"{name}: mode must be exactly 0444, got "
                        f"{stat.S_IMODE(metadata.st_mode):04o}"
                    )
                payloads[name] = read_fd_limited(fd, maximum, name)
            finally:
                os.close(fd)

        after = os.fstat(directory_fd)
        if (after.st_dev, after.st_ino, after.st_mode) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
        ):
            fail("report directory identity or mode changed during validation")
    finally:
        os.close(directory_fd)

    return payloads[REPORT_NAME], payloads[SUCCESS_NAME]


def parse_and_bind_report(
    report_bytes: bytes,
    success_bytes: bytes,
    expected_report_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    if not report_bytes or not report_bytes.endswith(b"\n"):
        fail("REPORT.json must be nonempty and newline-terminated")
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    expected_success = (report_sha256 + "\n").encode("ascii")
    if success_bytes != expected_success:
        fail(
            "_SUCCESS does not exactly bind REPORT.json; "
            f"computed={report_sha256}, raw_success={success_bytes!r}"
        )
    if expected_report_sha256 is not None and report_sha256 != expected_report_sha256:
        fail(
            "REPORT.json SHA-256 differs from the externally expected value; "
            f"actual={report_sha256}, expected={expected_report_sha256}"
        )
    try:
        text = report_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(f"REPORT.json is not strict UTF-8: {exc}")
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=no_duplicate_object,
            parse_constant=reject_json_constant,
        )
    except ReceiptValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        fail(f"REPORT.json is not valid strict JSON: {exc}")
    if type(parsed) is not dict:
        fail("REPORT.json root must be an object")
    reject_nonfinite_json(parsed)
    return parsed, report_sha256


def validate_authority(report: dict[str, Any], args: argparse.Namespace) -> None:
    authority = exact_keys(
        report["authority"],
        {
            "release_repo",
            "final_commit",
            "git_status_porcelain",
            "config_path",
            "config_sha256",
            "probe_path",
            "probe_sha256",
            "validator_path",
            "validator_sha256",
            "job_template_path",
            "job_template_sha256",
            "phase_a_manifest_path",
            "phase_a_manifest_sha256",
            "canonical_training_lineage_path",
            "canonical_training_lineage_sha256",
            "canonical_training_lineage_claimed",
        },
        "authority",
    )
    bindings = (
        ("release_repo", args.expected_release_repo),
        ("final_commit", args.expected_commit),
        ("config_path", args.expected_config_path),
        ("config_sha256", args.expected_config_sha256),
        ("probe_path", args.expected_probe_path),
        ("probe_sha256", args.expected_probe_sha256),
        ("validator_path", args.expected_validator_path),
        ("validator_sha256", args.expected_validator_sha256),
        ("job_template_path", args.expected_job_template_path),
        ("job_template_sha256", args.expected_job_template_sha256),
        ("phase_a_manifest_path", args.expected_phase_a_manifest_path),
        ("phase_a_manifest_sha256", args.expected_phase_a_manifest_sha256),
        (
            "canonical_training_lineage_path",
            args.expected_canonical_training_lineage_path,
        ),
        (
            "canonical_training_lineage_sha256",
            args.expected_canonical_training_lineage_sha256,
        ),
    )
    for field, expected in bindings:
        require_exact(authority[field], expected, f"authority.{field}")
    require_true(
        authority["canonical_training_lineage_claimed"],
        "authority.canonical_training_lineage_claimed",
    )
    if authority["git_status_porcelain"] != [] or type(authority["git_status_porcelain"]) is not list:
        fail("authority.git_status_porcelain: final tested worktree was not clean")

    expected_source_map = expected_final_git_blob_raw_sha256(args)
    tested = exact_keys(
        report["tested_final_git_blob_raw_sha256"],
        set(expected_source_map),
        "tested_final_git_blob_raw_sha256",
    )
    for key, value in tested.items():
        require_sha256(value, f"tested_final_git_blob_raw_sha256.{key}")
        require_exact(
            value,
            expected_source_map[key],
            f"tested_final_git_blob_raw_sha256.{key}",
        )


def validate_checkpoint(report: dict[str, Any], args: argparse.Namespace) -> None:
    checkpoint = exact_keys(
        report["checkpoint"],
        {
            "alias_path",
            "original_path",
            "full_sha256",
            "asset_reseal_report_sha256",
            "device_inode",
            "bytes",
            "mode",
            "samefile_with_original",
            "source_global_step",
            "source_scheduler_last_epoch",
            "trainable_keyset_sha256",
            "ablation_fingerprint_sha256",
            "legacy_top_level_keys",
        },
        "checkpoint",
    )
    require_exact(
        checkpoint["alias_path"], args.expected_checkpoint_path, "checkpoint.alias_path"
    )
    require_exact(
        checkpoint["original_path"],
        args.expected_original_checkpoint_path,
        "checkpoint.original_path",
    )
    require_exact(
        checkpoint["full_sha256"],
        args.expected_checkpoint_sha256,
        "checkpoint.full_sha256",
    )
    require_exact(
        checkpoint["asset_reseal_report_sha256"],
        args.expected_asset_reseal_report_sha256,
        "checkpoint.asset_reseal_report_sha256",
    )
    if type(checkpoint["device_inode"]) is not str or DEVICE_INODE_RE.fullmatch(
        checkpoint["device_inode"]
    ) is None:
        fail(
            "checkpoint.device_inode: expected launcher-bound positive 'device:inode', "
            f"got {checkpoint['device_inode']!r}"
        )
    require_int(checkpoint["bytes"], EXPECTED_CHECKPOINT_BYTES, "checkpoint.bytes")
    require_exact(checkpoint["mode"], "0444", "checkpoint.mode")
    require_true(checkpoint["samefile_with_original"], "checkpoint.samefile_with_original")
    require_int(
        checkpoint["source_global_step"],
        EXPECTED_SOURCE_STEP,
        "checkpoint.source_global_step",
    )
    require_int(
        checkpoint["source_scheduler_last_epoch"],
        EXPECTED_SOURCE_STEP,
        "checkpoint.source_scheduler_last_epoch",
    )
    require_exact(
        checkpoint["trainable_keyset_sha256"],
        EXPECTED_KEYSET_SHA256,
        "checkpoint.trainable_keyset_sha256",
    )
    require_exact(
        checkpoint["ablation_fingerprint_sha256"],
        args.expected_ablation_fingerprint_sha256,
        "checkpoint.ablation_fingerprint_sha256",
    )
    require_exact(
        checkpoint["legacy_top_level_keys"],
        EXPECTED_LEGACY_TOP_LEVEL_KEYS,
        "checkpoint.legacy_top_level_keys",
    )


def validate_boot(report: dict[str, Any]) -> None:
    require_int(report["world_size"], EXPECTED_WORLD_SIZE, "world_size")
    boot = exact_keys(
        report["boot"],
        {
            "mode",
            "stage_step",
            "checkpoint_dir",
            "checkpoint_dir_absent",
            "local_last_absent",
            "resolved_train_branch",
            "branch_resolver",
            "config_schema_warning_count",
        },
        "boot",
    )
    require_exact(boot["mode"], "fresh", "boot.mode")
    require_int(boot["stage_step"], 0, "boot.stage_step")
    require_abs_posix_path(boot["checkpoint_dir"], "boot.checkpoint_dir")
    require_true(boot["checkpoint_dir_absent"], "boot.checkpoint_dir_absent")
    require_true(boot["local_last_absent"], "boot.local_last_absent")
    require_exact(boot["resolved_train_branch"], "fresh", "boot.resolved_train_branch")
    require_exact(
        boot["branch_resolver"],
        "training.formal_boot.resolve_checkpoint_boot_branch",
        "boot.branch_resolver",
    )
    require_int(
        boot["config_schema_warning_count"], 0, "boot.config_schema_warning_count"
    )


def require_shape(value: Any, where: str, *, scalar_allowed: bool) -> list[int]:
    if type(value) is not list:
        fail(f"{where}: shape must be a JSON list, got {value!r}")
    if not value and not scalar_allowed:
        fail(f"{where}: scalar shape is not allowed here")
    for index, dimension in enumerate(value):
        if type(dimension) is not int or dimension <= 0:
            fail(f"{where}[{index}]: shape dimension must be a positive integer")
    return value


def validate_tensor_detail(value: Any, where: str, *, scalar_allowed: bool) -> None:
    detail = exact_keys(
        value,
        {"global_shape", "local_shape", "dtype", "exact", "max_abs"},
        where,
    )
    require_shape(detail["global_shape"], f"{where}.global_shape", scalar_allowed=scalar_allowed)
    require_shape(detail["local_shape"], f"{where}.local_shape", scalar_allowed=scalar_allowed)
    if type(detail["dtype"]) is not str or DTYPE_RE.fullmatch(detail["dtype"]) is None:
        fail(f"{where}.dtype: malformed torch dtype {detail['dtype']!r}")
    require_true(detail["exact"], f"{where}.exact")
    require_number_equal(detail["max_abs"], 0.0, f"{where}.max_abs")


def validate_closure(report: dict[str, Any]) -> None:
    closure = exact_keys(
        report["closure"],
        {
            "raw_count",
            "ema_count",
            "optimizer_count",
            "optimizer_param_group_count",
            "adam_step_values",
            "exact_per_rank",
            "ranks_completed",
            "size_stratified_details",
        },
        "closure",
    )
    for field in ("raw_count", "ema_count", "optimizer_count"):
        require_int(closure[field], EXPECTED_TRAINABLE_COUNT, f"closure.{field}")
    require_int(
        closure["optimizer_param_group_count"], 1, "closure.optimizer_param_group_count"
    )
    steps = closure["adam_step_values"]
    if type(steps) is not list or len(steps) != 1:
        fail(f"closure.adam_step_values: expected one value, got {steps!r}")
    require_int(steps[0], EXPECTED_OPTIMIZER_STEP, "closure.adam_step_values[0]")
    exact_per_rank = exact_keys(
        closure["exact_per_rank"],
        {"raw", "ema", "adam_step", "adam_exp_avg", "adam_exp_avg_sq"},
        "closure.exact_per_rank",
    )
    for field in exact_per_rank:
        require_int(
            exact_per_rank[field],
            EXPECTED_TRAINABLE_COUNT,
            f"closure.exact_per_rank.{field}",
        )
    require_int(
        closure["ranks_completed"], EXPECTED_WORLD_SIZE, "closure.ranks_completed"
    )

    details = closure["size_stratified_details"]
    if type(details) is not dict or len(details) != 3:
        fail(
            "closure.size_stratified_details: expected exactly three FQN entries, "
            f"got {type(details).__name__} with "
            f"{len(details) if type(details) is dict else 'unknown'} entries"
        )
    expected_detail_keys = {
        "raw",
        "ema",
        "adam_step",
        "adam_exp_avg",
        "adam_exp_avg_sq",
        "adam_step_value",
    }
    for fqn, value in details.items():
        if type(fqn) is not str or not fqn or fqn.strip() != fqn or "." not in fqn:
            fail(f"closure.size_stratified_details: malformed FQN {fqn!r}")
        item = exact_keys(value, expected_detail_keys, f"closure.size_stratified_details[{fqn!r}]")
        prefix = f"closure.size_stratified_details[{fqn!r}]"
        validate_tensor_detail(item["raw"], f"{prefix}.raw", scalar_allowed=False)
        validate_tensor_detail(item["ema"], f"{prefix}.ema", scalar_allowed=False)
        validate_tensor_detail(item["adam_step"], f"{prefix}.adam_step", scalar_allowed=True)
        validate_tensor_detail(
            item["adam_exp_avg"], f"{prefix}.adam_exp_avg", scalar_allowed=False
        )
        validate_tensor_detail(
            item["adam_exp_avg_sq"], f"{prefix}.adam_exp_avg_sq", scalar_allowed=False
        )
        require_number_equal(
            item["adam_step_value"],
            EXPECTED_OPTIMIZER_STEP,
            f"{prefix}.adam_step_value",
        )
        raw_global = item["raw"]["global_shape"]
        raw_local = item["raw"]["local_shape"]
        for peer in ("ema", "adam_exp_avg", "adam_exp_avg_sq"):
            require_exact(
                item[peer]["global_shape"], raw_global, f"{prefix}.{peer}.global_shape"
            )
            require_exact(
                item[peer]["local_shape"], raw_local, f"{prefix}.{peer}.local_shape"
            )


def validate_scheduler(report: dict[str, Any]) -> None:
    scheduler = exact_keys(
        report["scheduler"],
        {
            "source_ignored",
            "state_unchanged_by_init",
            "warmup_steps",
            "first_optimizer_update_lr",
            "step1_logged_lr",
            "fresh_state",
            "optimizer_hparams_preserved",
            "optimizer_group_hparams",
        },
        "scheduler",
    )
    require_true(scheduler["source_ignored"], "scheduler.source_ignored")
    require_true(
        scheduler["state_unchanged_by_init"], "scheduler.state_unchanged_by_init"
    )
    require_int(scheduler["warmup_steps"], 500, "scheduler.warmup_steps")
    first_lr = require_single_number_list(
        scheduler["first_optimizer_update_lr"], "scheduler.first_optimizer_update_lr"
    )
    step1_lr = require_single_number_list(
        scheduler["step1_logged_lr"], "scheduler.step1_logged_lr"
    )
    if first_lr <= 0.0:
        fail("scheduler.first_optimizer_update_lr[0]: must be positive")
    require_close(
        first_lr,
        EXPECTED_FIRST_OPTIMIZER_UPDATE_LR,
        "scheduler.first_optimizer_update_lr[0]",
    )
    require_close(
        step1_lr,
        EXPECTED_STEP1_LOGGED_LR,
        "scheduler.step1_logged_lr[0]",
    )
    require_close(step1_lr, first_lr * 2.0, "scheduler.step1_logged_lr[0]")

    fresh_state = scheduler["fresh_state"]
    if type(fresh_state) is not dict or not fresh_state:
        fail("scheduler.fresh_state: expected a nonempty PyTorch scheduler state object")
    for required in ("last_epoch", "_step_count", "_last_lr", "base_lrs"):
        if required not in fresh_state:
            fail(f"scheduler.fresh_state: missing boot-critical key {required!r}")
    require_int(fresh_state["last_epoch"], 0, "scheduler.fresh_state.last_epoch")
    require_int(fresh_state["_step_count"], 1, "scheduler.fresh_state._step_count")
    fresh_last_lr = require_single_number_list(
        fresh_state["_last_lr"], "scheduler.fresh_state._last_lr"
    )
    require_close(fresh_last_lr, first_lr, "scheduler.fresh_state._last_lr[0]")
    base_lr = require_single_number_list(
        fresh_state["base_lrs"], "scheduler.fresh_state.base_lrs"
    )
    require_close(base_lr, EXPECTED_BASE_LR, "scheduler.fresh_state.base_lrs[0]")
    require_close(base_lr, first_lr * 500.0, "scheduler.fresh_state.base_lrs[0]")

    require_true(
        scheduler["optimizer_hparams_preserved"],
        "scheduler.optimizer_hparams_preserved",
    )
    groups = scheduler["optimizer_group_hparams"]
    if type(groups) is not list or len(groups) != 1 or type(groups[0]) is not dict or not groups[0]:
        fail("scheduler.optimizer_group_hparams: expected one nonempty parameter group")
    group = groups[0]
    for required in ("lr", "initial_lr"):
        if required not in group:
            fail(f"scheduler.optimizer_group_hparams[0]: missing {required!r}")
    require_close(group["lr"], first_lr, "scheduler.optimizer_group_hparams[0].lr")
    require_close(
        group["initial_lr"], base_lr, "scheduler.optimizer_group_hparams[0].initial_lr"
    )


def validate_rng(report: dict[str, Any]) -> None:
    rng = exact_keys(
        report["rng"],
        {
            "init_from_policy",
            "source_rng_restored",
            "all_stream_states_unchanged_by_init",
            "pre_vae_unique",
            "post_vae_unique",
            "pre_vae_sha256_by_rank",
            "post_vae_sha256_by_rank",
            "stream_state_sha256_rank0",
        },
        "rng",
    )
    require_exact(rng["init_from_policy"], "legacy_missing_rng", "rng.init_from_policy")
    require_false(rng["source_rng_restored"], "rng.source_rng_restored")
    require_true(
        rng["all_stream_states_unchanged_by_init"],
        "rng.all_stream_states_unchanged_by_init",
    )
    require_int(rng["pre_vae_unique"], EXPECTED_WORLD_SIZE, "rng.pre_vae_unique")
    require_int(rng["post_vae_unique"], EXPECTED_WORLD_SIZE, "rng.post_vae_unique")
    before = rng["pre_vae_sha256_by_rank"]
    after = rng["post_vae_sha256_by_rank"]
    if type(before) is not list or len(before) != EXPECTED_WORLD_SIZE:
        fail("rng.pre_vae_sha256_by_rank: expected exactly two rank-ordered digests")
    if type(after) is not list or len(after) != EXPECTED_WORLD_SIZE:
        fail("rng.post_vae_sha256_by_rank: expected exactly two rank-ordered digests")
    for index, digest in enumerate(before):
        require_sha256(digest, f"rng.pre_vae_sha256_by_rank[{index}]")
    for index, digest in enumerate(after):
        require_sha256(digest, f"rng.post_vae_sha256_by_rank[{index}]")
    if len(set(before)) != EXPECTED_WORLD_SIZE:
        fail("rng.pre_vae_sha256_by_rank: rank RNG states are not 2/2 unique")
    require_exact(after, before, "rng.post_vae_sha256_by_rank")

    streams = rng["stream_state_sha256_rank0"]
    if type(streams) is not dict or not streams:
        fail("rng.stream_state_sha256_rank0: expected a nonempty stream digest map")
    if "vae" not in streams:
        fail("rng.stream_state_sha256_rank0: missing required 'vae' stream")
    for name, digest in streams.items():
        if type(name) is not str or not name or name.strip() != name:
            fail(f"rng.stream_state_sha256_rank0: malformed stream name {name!r}")
        require_sha256(digest, f"rng.stream_state_sha256_rank0.{name}")
    require_exact(streams["vae"], before[0], "rng.stream_state_sha256_rank0.vae")


def validate_negative_controls(report: dict[str, Any]) -> None:
    controls = exact_keys(
        report["negative_controls"],
        set(EXPECTED_NEGATIVE_CONTROL_NEEDLES),
        "negative_controls",
    )
    for label, needle in EXPECTED_NEGATIVE_CONTROL_NEEDLES.items():
        rendered = controls[label]
        if type(rendered) is not str or needle not in rendered:
            fail(
                f"negative_controls.{label}: expected captured failure containing "
                f"{needle!r}, got {rendered!r}"
            )
        if ":" not in rendered:
            fail(f"negative_controls.{label}: captured exception lacks its type prefix")


def ordinary_file_bytes(path: Path, *, exact_mode: int | None = None) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        fail(f"{path}: lstat failed: {exc}")
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        fail(f"{path}: expected ordinary non-symlink file")
    if exact_mode is not None and stat.S_IMODE(before.st_mode) != exact_mode:
        fail(
            f"{path}: expected mode {exact_mode:04o}, got "
            f"{stat.S_IMODE(before.st_mode):04o}"
        )
    try:
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        fail(f"{path}: read failed: {exc}")
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(after) != identity(before):
        fail(f"{path}: file changed while reading")
    return raw


def strict_json_bytes(raw: bytes, where: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(f"{where}: not strict UTF-8: {exc}")
    try:
        value = json.loads(
            text,
            object_pairs_hook=no_duplicate_object,
            parse_constant=reject_json_constant,
        )
    except ReceiptValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        fail(f"{where}: invalid strict JSON: {exc}")
    if type(value) is not dict:
        fail(f"{where}: root must be an object")
    reject_nonfinite_json(value, where)
    return value


def final_git_blob_map(
    repo: Path, commit: str, relative_paths: set[str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative_path in sorted(relative_paths):
        source_path = repo / relative_path
        raw_file = ordinary_file_bytes(source_path)
        try:
            object_id = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(repo),
                    "rev-parse",
                    "--verify",
                    f"{commit}:{relative_path}",
                ],
                text=True,
            ).strip()
            object_type = subprocess.check_output(
                ["git", "-C", str(repo), "cat-file", "-t", object_id],
                text=True,
            ).strip()
            raw_blob = subprocess.check_output(
                ["git", "-C", str(repo), "cat-file", "blob", object_id]
            )
        except subprocess.CalledProcessError as exc:
            fail(f"Git blob lookup failed for {relative_path}: {exc}")
        if object_type != "blob":
            fail(f"final Git object is not a blob: {relative_path}")
        file_sha = hashlib.sha256(raw_file).hexdigest()
        blob_sha = hashlib.sha256(raw_blob).hexdigest()
        if file_sha != blob_sha:
            fail(
                f"worktree differs from final Git blob for {relative_path}: "
                f"worktree={file_sha}, blob={blob_sha}"
            )
        result[relative_path] = blob_sha
    return result


def render_substitutions(
    *,
    repo: Path,
    commit: str,
    phase_a_manifest_path: str,
    phase_a_manifest_sha256: str,
    training_lineage_path: str,
    training_lineage_sha256: str,
    smoke_output_path: str,
    runtime_map: dict[str, str],
) -> dict[str, str]:
    return {
        "PLACEHOLDER_RELEASE_REPO_PATH": str(repo),
        "PLACEHOLDER_RELEASE_CONFIG_PATH": str(repo / CONFIG_REL),
        "PLACEHOLDER_PHASE_A_MANIFEST_PATH": phase_a_manifest_path,
        "PLACEHOLDER_PHASE_A_MANIFEST_SHA256": phase_a_manifest_sha256,
        "PLACEHOLDER_CANONICAL_TRAINING_LINEAGE_PATH": training_lineage_path,
        "PLACEHOLDER_CANONICAL_TRAINING_LINEAGE_SHA256": training_lineage_sha256,
        "PLACEHOLDER_RELEASE_PROBE_PATH": str(repo / PROBE_REL),
        "PLACEHOLDER_PROBE_SHA256": runtime_map[PROBE_REL],
        "PLACEHOLDER_RELEASE_VALIDATOR_PATH": str(repo / VALIDATOR_REL),
        "PLACEHOLDER_VALIDATOR_SHA256": runtime_map[VALIDATOR_REL],
        "PLACEHOLDER_RELEASE_JOB_TEMPLATE_PATH": str(repo / TEMPLATE_REL),
        "PLACEHOLDER_JOB_TEMPLATE_SHA256": runtime_map[TEMPLATE_REL],
        "PLACEHOLDER_EXPECTED_COMMIT": commit,
        "PLACEHOLDER_TRAIN_PY_GIT_BLOB_RAW_SHA256": runtime_map["train.py"],
        "PLACEHOLDER_TRAINING_CHECKPOINT_PY_GIT_BLOB_RAW_SHA256": runtime_map[
            "training/checkpoint.py"
        ],
        "PLACEHOLDER_TRAINING_RNG_STATE_PY_GIT_BLOB_RAW_SHA256": runtime_map[
            "training/rng_state.py"
        ],
        "PLACEHOLDER_TRAINING_FORMAL_BOOT_PY_GIT_BLOB_RAW_SHA256": runtime_map[
            "training/formal_boot.py"
        ],
        "PLACEHOLDER_TRAINING_CONFIG_SCHEMA_PY_GIT_BLOB_RAW_SHA256": runtime_map[
            "training/config_schema.py"
        ],
        "PLACEHOLDER_NETWORKS_MODEL_MOT_PY_GIT_BLOB_RAW_SHA256": runtime_map[
            "networks/model_mot.py"
        ],
        "PLACEHOLDER_NETWORKS_VAE_PY_GIT_BLOB_RAW_SHA256": runtime_map[
            "networks/vae.py"
        ],
        "PLACEHOLDER_CONFIG_SHA256": runtime_map[CONFIG_REL],
        "PLACEHOLDER_CREATE_ONCE_OUTPUT": smoke_output_path,
    }


def validate_render_receipt(
    receipt: dict[str, Any], args: argparse.Namespace
) -> None:
    exact_keys(
        receipt,
        {
            "schema",
            "status",
            "release",
            "phase_a",
            "artifacts",
            "runtime_git_blob_raw_sha256",
            "dedicated_asset_git_blob_raw_sha256",
            "semantic_contract",
        },
        "render receipt",
    )
    require_exact(receipt["schema"], RENDER_SCHEMA, "render receipt.schema")
    require_exact(receipt["status"], "PASS", "render receipt.status")
    release = exact_keys(
        receipt["release"],
        {"repo", "commit", "git_status_porcelain", "config_sha256"},
        "render receipt.release",
    )
    require_exact(release["repo"], args.expected_release_repo, "render release.repo")
    require_exact(release["commit"], args.expected_commit, "render release.commit")
    require_exact(release["git_status_porcelain"], [], "render release status")

    repo = Path(args.expected_release_repo)
    try:
        actual_head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        actual_status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        ).splitlines()
    except subprocess.CalledProcessError as exc:
        fail(f"release Git inspection failed: {exc}")
    require_exact(actual_head, args.expected_commit, "actual release HEAD")
    require_exact(actual_status, [], "actual release status")
    dedicated_map = final_git_blob_map(repo, actual_head, DEDICATED_GIT_BLOB_PATHS)
    runtime_map = {path: dedicated_map[path] for path in sorted(RUNTIME_GIT_BLOB_PATHS)}
    require_exact(
        receipt["runtime_git_blob_raw_sha256"], runtime_map, "render runtime blob map"
    )
    require_exact(
        receipt["dedicated_asset_git_blob_raw_sha256"],
        dedicated_map,
        "render dedicated blob map",
    )
    require_exact(release["config_sha256"], runtime_map[CONFIG_REL], "render config SHA")

    schema_value = strict_json_bytes(
        ordinary_file_bytes(repo / RENDER_SCHEMA_REL), "render receipt JSON schema"
    )
    try:
        schema_const = schema_value["properties"]["schema"]["const"]
    except (KeyError, TypeError):
        fail("render receipt JSON schema lacks schema const")
    require_exact(schema_const, RENDER_SCHEMA, "render JSON schema identity")

    phase_a = exact_keys(
        receipt["phase_a"],
        {
            "manifest_path",
            "manifest_sha256",
            "training_lineage_path",
            "training_lineage_sha256",
        },
        "render phase_a",
    )
    phase_bindings = (
        ("manifest_path", args.expected_phase_a_manifest_path),
        ("manifest_sha256", args.expected_phase_a_manifest_sha256),
        ("training_lineage_path", args.expected_training_lineage_path),
        ("training_lineage_sha256", args.expected_training_lineage_sha256),
    )
    for field, expected in phase_bindings:
        require_exact(phase_a[field], expected, f"render phase_a.{field}")
    require_exact(
        hashlib.sha256(
            ordinary_file_bytes(Path(args.expected_phase_a_manifest_path))
        ).hexdigest(),
        args.expected_phase_a_manifest_sha256,
        "actual Phase-A manifest SHA",
    )
    require_exact(
        hashlib.sha256(
            ordinary_file_bytes(Path(args.expected_training_lineage_path))
        ).hexdigest(),
        args.expected_training_lineage_sha256,
        "actual training-lineage SHA",
    )

    artifacts = exact_keys(
        receipt["artifacts"],
        {
            "template_path",
            "template_sha256",
            "resolved_spec_path",
            "resolved_spec_sha256",
            "smoke_output_path",
            "placeholder_count",
        },
        "render artifacts",
    )
    require_exact(artifacts["template_path"], str(repo / TEMPLATE_REL), "template path")
    require_exact(artifacts["template_sha256"], runtime_map[TEMPLATE_REL], "template SHA")
    require_exact(
        artifacts["resolved_spec_path"],
        args.expected_resolved_spec_path,
        "resolved spec path",
    )
    require_exact(
        artifacts["smoke_output_path"],
        args.expected_smoke_output_path,
        "smoke output path",
    )
    require_int(artifacts["placeholder_count"], 22, "placeholder_count")
    substitutions = render_substitutions(
        repo=repo,
        commit=actual_head,
        phase_a_manifest_path=args.expected_phase_a_manifest_path,
        phase_a_manifest_sha256=args.expected_phase_a_manifest_sha256,
        training_lineage_path=args.expected_training_lineage_path,
        training_lineage_sha256=args.expected_training_lineage_sha256,
        smoke_output_path=args.expected_smoke_output_path,
        runtime_map=runtime_map,
    )
    template_raw = ordinary_file_bytes(repo / TEMPLATE_REL)
    try:
        expected_text = template_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(f"render template not UTF-8: {exc}")
    found = re.findall(r"PLACEHOLDER_[A-Z0-9_]+", expected_text)
    if len(found) != len(set(found)) or set(found) != set(substitutions):
        fail("render template placeholder closure drift")
    for placeholder, replacement in substitutions.items():
        expected_text = expected_text.replace(placeholder, replacement)
    if re.search(r"PLACEHOLDER_[A-Z0-9_]+", expected_text):
        fail("resolved spec still contains placeholder")
    expected_resolved = expected_text.encode("utf-8")
    actual_resolved = ordinary_file_bytes(
        Path(args.expected_resolved_spec_path), exact_mode=0o444
    )
    require_exact(actual_resolved, expected_resolved, "resolved spec exact bytes")
    require_exact(
        artifacts["resolved_spec_sha256"],
        hashlib.sha256(expected_resolved).hexdigest(),
        "resolved spec SHA",
    )
    expected_semantic_contract = {
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
    }
    require_exact(
        receipt["semantic_contract"],
        expected_semantic_contract,
        "render semantic contract",
    )


def validate_report(report: dict[str, Any], args: argparse.Namespace) -> None:
    exact_keys(report, TOP_LEVEL_KEYS, "report")
    require_exact(report["schema"], SCHEMA, "schema")
    require_exact(report["status"], "PASS", "status")
    require_exact(report["verdict"], "PASS", "verdict")
    elapsed = require_number(report["elapsed_s"], "elapsed_s")
    if elapsed <= 0.0:
        fail(f"elapsed_s: expected a positive duration, got {elapsed!r}")
    validate_authority(report, args)
    validate_checkpoint(report, args)
    validate_boot(report)
    validate_closure(report)
    validate_scheduler(report)
    validate_rng(report)
    validate_negative_controls(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a sealed Stage512A-final init FSDP2 smoke-v3 receipt."
    )
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--expected-release-repo", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-train-py-git-blob-raw-sha256", required=True)
    parser.add_argument(
        "--expected-training-checkpoint-py-git-blob-raw-sha256", required=True
    )
    parser.add_argument(
        "--expected-training-rng-state-py-git-blob-raw-sha256", required=True
    )
    parser.add_argument(
        "--expected-training-formal-boot-py-git-blob-raw-sha256", required=True
    )
    parser.add_argument(
        "--expected-training-config-schema-py-git-blob-raw-sha256", required=True
    )
    parser.add_argument(
        "--expected-networks-model-mot-py-git-blob-raw-sha256", required=True
    )
    parser.add_argument(
        "--expected-networks-vae-py-git-blob-raw-sha256", required=True
    )
    parser.add_argument("--expected-config-path", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-probe-path", required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-validator-path", required=True)
    parser.add_argument("--expected-validator-sha256", required=True)
    parser.add_argument("--expected-job-template-path", required=True)
    parser.add_argument("--expected-job-template-sha256", required=True)
    parser.add_argument("--expected-phase-a-manifest-path", required=True)
    parser.add_argument("--expected-phase-a-manifest-sha256", required=True)
    parser.add_argument("--expected-canonical-training-lineage-path", required=True)
    parser.add_argument("--expected-canonical-training-lineage-sha256", required=True)
    parser.add_argument("--expected-checkpoint-path", required=True)
    parser.add_argument("--expected-original-checkpoint-path", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-ablation-fingerprint-sha256", required=True)
    parser.add_argument("--expected-asset-reseal-report-sha256", required=True)
    parser.add_argument(
        "--expected-report-sha256",
        help="Optional independent REPORT.json digest; _SUCCESS is always checked.",
    )
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    for field in (
        "expected_release_repo",
        "expected_config_path",
        "expected_probe_path",
        "expected_validator_path",
        "expected_job_template_path",
        "expected_phase_a_manifest_path",
        "expected_canonical_training_lineage_path",
        "expected_checkpoint_path",
        "expected_original_checkpoint_path",
    ):
        require_abs_posix_path(getattr(args, field), f"CLI --{field.replace('_', '-')}")
    require_commit(args.expected_commit, "CLI --expected-commit")
    for field in (
        "expected_train_py_git_blob_raw_sha256",
        "expected_training_checkpoint_py_git_blob_raw_sha256",
        "expected_training_rng_state_py_git_blob_raw_sha256",
        "expected_training_formal_boot_py_git_blob_raw_sha256",
        "expected_training_config_schema_py_git_blob_raw_sha256",
        "expected_networks_model_mot_py_git_blob_raw_sha256",
        "expected_networks_vae_py_git_blob_raw_sha256",
        "expected_config_sha256",
        "expected_probe_sha256",
        "expected_validator_sha256",
        "expected_job_template_sha256",
        "expected_phase_a_manifest_sha256",
        "expected_canonical_training_lineage_sha256",
        "expected_checkpoint_sha256",
        "expected_ablation_fingerprint_sha256",
        "expected_asset_reseal_report_sha256",
    ):
        require_sha256(getattr(args, field), f"CLI --{field.replace('_', '-')}")
    if args.expected_report_sha256 is not None:
        require_sha256(args.expected_report_sha256, "CLI --expected-report-sha256")
    if args.expected_checkpoint_path == args.expected_original_checkpoint_path:
        fail("CLI checkpoint alias and original paths must be distinct path strings")
    require_exact(
        args.expected_checkpoint_sha256,
        EXPECTED_CHECKPOINT_SHA256,
        "CLI --expected-checkpoint-sha256",
    )
    require_exact(
        args.expected_ablation_fingerprint_sha256,
        EXPECTED_ABLATION_FINGERPRINT_SHA256,
        "CLI --expected-ablation-fingerprint-sha256",
    )
    expected_final_git_blob_raw_sha256(args)


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_cli(args)
        report_bytes, success_bytes = read_sealed_receipt(args.report_dir)
        report, report_sha256 = parse_and_bind_report(
            report_bytes, success_bytes, args.expected_report_sha256
        )
        validate_report(report, args)
    except ReceiptValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: sealed A-final init smoke v3 receipt {report_sha256}")
    return 0


def build_render_receipt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently validate a smoke-v3 renderer receipt/spec."
    )
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--expected-release-repo", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-phase-a-manifest-path", required=True)
    parser.add_argument("--expected-phase-a-manifest-sha256", required=True)
    parser.add_argument("--expected-training-lineage-path", required=True)
    parser.add_argument("--expected-training-lineage-sha256", required=True)
    parser.add_argument("--expected-resolved-spec-path", required=True)
    parser.add_argument("--expected-smoke-output-path", required=True)
    parser.add_argument("--expected-report-sha256")
    return parser


def render_receipt_main(argv: list[str]) -> int:
    args = build_render_receipt_parser().parse_args(argv)
    try:
        for field in (
            "expected_release_repo",
            "expected_phase_a_manifest_path",
            "expected_training_lineage_path",
            "expected_resolved_spec_path",
            "expected_smoke_output_path",
        ):
            require_abs_posix_path(
                getattr(args, field), f"CLI --{field.replace('_', '-')}"
            )
        commit = require_commit(args.expected_commit, "CLI --expected-commit")
        if len(commit) != 40:
            fail("renderer receipt requires a 40-hex Git commit")
        for field in (
            "expected_phase_a_manifest_sha256",
            "expected_training_lineage_sha256",
        ):
            require_sha256(getattr(args, field), f"CLI --{field.replace('_', '-')}")
        if args.expected_report_sha256 is not None:
            require_sha256(args.expected_report_sha256, "CLI --expected-report-sha256")
        report_bytes, success_bytes = read_sealed_receipt(args.report_dir)
        receipt, report_sha256 = parse_and_bind_report(
            report_bytes, success_bytes, args.expected_report_sha256
        )
        validate_render_receipt(receipt, args)
    except ReceiptValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: sealed smoke-v3 render receipt {report_sha256}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render-receipt":
        raise SystemExit(render_receipt_main(sys.argv[2:]))
    raise SystemExit(main())
