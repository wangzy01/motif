"""Fail-closed formal launch mode assertion (environment-gated)."""

from __future__ import annotations

import os
import re

import torch.distributed as dist


EXPECTED_BOOT_MODE_ENV = "MOT_EXPECTED_BOOT_MODE"
EXPECTED_RUN_LINEAGE_ENV = "MOT_EXPECTED_RUN_LINEAGE_SHA256"
FORMAL_RUN_LINEAGE_KEY = "formal_run_lineage_sha256"
VALID_BOOT_MODES = ("fresh", "continue")


def normalize_expected_run_lineage_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(
            f"{EXPECTED_RUN_LINEAGE_ENV} must be exactly 64 lowercase hex characters"
        )
    return value


def resolve_expected_run_lineage_sha256(*, is_dist: bool) -> str | None:
    """Validate and synchronize the immutable formal-run identity.

    The value is attempt-invariant: retries may change pod/job identities, but
    they must never change the code/config/data/init lineage bound to last.pt.
    Invalid rank-local values still participate in the object collective so a
    single bad rank cannot strand its peers.
    """
    if is_dist and not dist.is_initialized():
        raise RuntimeError("run-lineage synchronization requires an initialized process group")
    error = None
    value = None
    try:
        value = normalize_expected_run_lineage_sha256(
            os.environ.get(EXPECTED_RUN_LINEAGE_ENV)
        )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    local = {"value": value, "error": error}
    if is_dist:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    failures = [
        (rank, item["error"])
        for rank, item in enumerate(gathered)
        if item["error"] is not None
    ]
    if failures:
        raise RuntimeError(f"formal run-lineage environment validation failed: {failures}")
    values = [item["value"] for item in gathered]
    if any(item != values[0] for item in values[1:]):
        raise RuntimeError(
            f"{EXPECTED_RUN_LINEAGE_ENV} differs across ranks: {values}"
        )
    return values[0]


def verify_formal_launch_requirement(
    *,
    required: bool,
    expected_boot_mode: str | None,
    expected_run_lineage: str | None,
) -> None:
    """Fail closed before output/model setup when a formal launch is required."""

    if not isinstance(required, bool):
        raise RuntimeError("trainer.formal_launch_required must be a boolean")
    if expected_boot_mode is not None and expected_boot_mode not in VALID_BOOT_MODES:
        raise RuntimeError(
            f"{EXPECTED_BOOT_MODE_ENV} must be one of {VALID_BOOT_MODES}, "
            f"got {expected_boot_mode!r}"
        )
    normalize_expected_run_lineage_sha256(expected_run_lineage)
    missing = []
    if required and expected_boot_mode is None:
        missing.append(EXPECTED_BOOT_MODE_ENV)
    if (required or expected_boot_mode is not None) and expected_run_lineage is None:
        missing.append(EXPECTED_RUN_LINEAGE_ENV)
    if missing:
        raise RuntimeError(
            "formal launch identity is required but missing: " + ", ".join(missing)
        )


def resolve_checkpoint_boot_branch(
    checkpoint_path: str,
    *,
    resume: bool,
    init_from: str | None,
    expected_mode: str | None,
    is_dist: bool,
) -> str:
    """Select fresh/continue before any state-loading collective.

    Every rank publishes both its filesystem observation and selected branch.
    This prevents a transient/shared-filesystem visibility split from sending
    some ranks into DCP resume while peers enter init_from collectives.
    """
    if is_dist and not dist.is_initialized():
        raise RuntimeError("boot-branch synchronization requires an initialized process group")
    error = None
    exists = False
    branch = "none"
    try:
        if expected_mode is not None and expected_mode not in VALID_BOOT_MODES:
            raise RuntimeError(
                f"{EXPECTED_BOOT_MODE_ENV} must be one of {VALID_BOOT_MODES}, "
                f"got {expected_mode!r}"
            )
        exists = os.path.exists(checkpoint_path)
        if exists and bool(resume):
            branch = "continue"
        elif init_from:
            branch = "fresh"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    local = {"exists": exists, "branch": branch, "error": error}
    if is_dist:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    failures = [
        (rank, item["error"])
        for rank, item in enumerate(gathered)
        if item["error"] is not None
    ]
    if failures:
        raise RuntimeError(f"checkpoint boot-branch validation failed: {failures}")
    observations = [(item["exists"], item["branch"]) for item in gathered]
    if any(item != observations[0] for item in observations[1:]):
        raise RuntimeError(
            "checkpoint visibility/boot branch differs across ranks: "
            f"{observations}"
        )
    branch = observations[0][1]
    if expected_mode is not None and branch != expected_mode:
        raise RuntimeError(
            f"formal boot branch mismatch before checkpoint load: "
            f"expected={expected_mode}, actual={branch}, "
            f"last_exists={observations[0][0]}"
        )
    return branch


def verify_expected_boot_mode(
    expected: str,
    *,
    actual: str,
    global_step: int,
    init_from: str | None,
    checkpoint_path: str,
) -> str:
    expected = str(expected)
    if expected not in VALID_BOOT_MODES:
        raise RuntimeError(
            f"{EXPECTED_BOOT_MODE_ENV} must be one of {VALID_BOOT_MODES}, got {expected!r}"
        )
    if actual != expected:
        raise RuntimeError(
            f"formal boot mode mismatch: expected={expected}, actual={actual}"
        )
    step = int(global_step)
    if expected == "fresh":
        if not init_from or step != 0:
            raise RuntimeError(
                f"formal fresh boot requires init_from and step=0; init_from={init_from!r}, step={step}"
            )
        return f"[formal-boot] verified mode=fresh init_from={init_from} step=0"
    if step <= 0:
        raise RuntimeError(f"formal continue boot requires checkpoint step>0; step={step}")
    return f"[formal-boot] verified mode=continue checkpoint={checkpoint_path} step={step}"
