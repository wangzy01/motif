"""Fail-closed metrics reconciliation for bounded checkpoint replay.

This module is deliberately independent of torch/distributed.  ``train.py``
calls it on rank 0 only, after every rank has loaded the same checkpoint and
before the dataset start step or metrics append handle is created.

The feature is disabled unless ``OFFLINE_ARB_AUTO_RESUME_RECONCILE=1``.  A
metrics stream already ending at the checkpoint is a true no-op: it neither
creates evidence nor consumes the one-replay guard.  When metrics are ahead of
the checkpoint, the complete stream and discarded tail are preserved before
an atomic prefix cutover.  A checkpoint-identity guard permits that replay
only once, so a deterministic crash cannot be replayed indefinitely.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, Callable


ENABLE_ENV = "OFFLINE_ARB_AUTO_RESUME_RECONCILE"
JOBTAG_ENV = "OFFLINE_ARB_ATTEMPT_JOBTAG"
NONCE_ENV = "OFFLINE_ARB_ATTEMPT_NONCE"
FREEZE_SHA_ENV = "OFFLINE_ARB_AUTO_RESUME_FREEZE_SHA256"
PLAN_HEADER_SHA_ENV = "OFFLINE_ARB_AUTO_RESUME_PLAN_HEADER_SHA256"
SCHEMA_VERSION = "offline_arb_auto_resume_reconcile_v1"
_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAMPLE_BYTES = 1 << 20


class AutoResumeReconcileError(RuntimeError):
    """The active stream cannot be reconciled without semantic risk."""


class AutoResumeReplayLimitError(AutoResumeReconcileError):
    """This exact checkpoint has already consumed its one replay."""


def auto_resume_reconcile_enabled(environ: dict[str, str] | None = None) -> bool:
    """Return the strict, explicit feature state.

    Unset and ``0`` mean disabled.  Any other value except ``1`` is rejected
    instead of being interpreted via Python truthiness.
    """

    env = os.environ if environ is None else environ
    raw = env.get(ENABLE_ENV, "0")
    if raw not in ("0", "1"):
        raise AutoResumeReconcileError(
            f"{ENABLE_ENV} must be exactly 0 or 1, got {raw!r}"
        )
    return raw == "1"


def _bindings_from_env(environ: dict[str, str]) -> dict[str, str]:
    jobtag = environ.get(JOBTAG_ENV, "")
    nonce = environ.get(NONCE_ENV, "")
    freeze_sha = environ.get(FREEZE_SHA_ENV, "")
    plan_header_sha = environ.get(PLAN_HEADER_SHA_ENV, "")
    if _SAFE_NAME_RE.fullmatch(jobtag) is None or jobtag in (".", ".."):
        raise AutoResumeReconcileError(f"{JOBTAG_ENV} is missing or malformed")
    for name, value in (
        (NONCE_ENV, nonce),
        (FREEZE_SHA_ENV, freeze_sha),
        (PLAN_HEADER_SHA_ENV, plan_header_sha),
    ):
        if _SHA256_RE.fullmatch(value) is None:
            raise AutoResumeReconcileError(
                f"{name} must be a lowercase SHA-256 token"
            )
    return {
        "jobtag": jobtag,
        "attempt_nonce": nonce,
        "freeze_sha256": freeze_sha,
        "plan_header_sha256": plan_header_sha,
    }


def _regular_identity(path: Path, *, allow_empty: bool) -> dict[str, int]:
    try:
        value = path.lstat()
    except OSError as exc:
        raise AutoResumeReconcileError(f"cannot stat {path}: {exc}") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise AutoResumeReconcileError(f"not an ordinary non-symlink file: {path}")
    if not allow_empty and value.st_size <= 0:
        raise AutoResumeReconcileError(f"file is empty: {path}")
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _read_stable(path: Path, *, allow_empty: bool) -> tuple[bytes, dict[str, int]]:
    before = _regular_identity(path, allow_empty=allow_empty)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AutoResumeReconcileError(f"cannot read {path}: {exc}") from exc
    after = _regular_identity(path, allow_empty=allow_empty)
    if before != after or len(raw) != before["bytes"]:
        raise AutoResumeReconcileError(f"file changed while reading: {path}")
    return raw, before


def _checkpoint_identity(path: Path, checkpoint_step: int) -> dict[str, Any]:
    if type(checkpoint_step) is not int or checkpoint_step < 0:
        raise AutoResumeReconcileError(
            f"checkpoint_step must be a nonnegative int, got {checkpoint_step!r}"
        )
    before = _regular_identity(path, allow_empty=False)
    sample_hash = hashlib.sha256()
    try:
        with path.open("rb", buffering=0) as handle:
            head = handle.read(min(_SAMPLE_BYTES, before["bytes"]))
            sample_hash.update(head)
            if before["bytes"] > _SAMPLE_BYTES:
                handle.seek(max(0, before["bytes"] - _SAMPLE_BYTES))
                sample_hash.update(handle.read(_SAMPLE_BYTES))
    except OSError as exc:
        raise AutoResumeReconcileError(f"cannot sample checkpoint {path}: {exc}") from exc
    after = _regular_identity(path, allow_empty=False)
    if before != after:
        raise AutoResumeReconcileError(f"checkpoint changed while sampling: {path}")
    identity: dict[str, Any] = {
        **before,
        "path": str(path.resolve(strict=True)),
        "global_step": checkpoint_step,
        "head_tail_sample_sha256": sample_hash.hexdigest(),
    }
    digest_raw = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    identity["identity_digest"] = hashlib.sha256(digest_raw).hexdigest()
    return identity


def _parse_metrics(raw: bytes) -> list[bytes]:
    if not raw:
        return []
    lines = raw.splitlines(keepends=True)
    if any(not line.endswith(b"\n") for line in lines):
        raise AutoResumeReconcileError("metrics stream has a partial/non-newline tail")
    for expected_step, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AutoResumeReconcileError(
                f"metrics JSON decode failed at row {expected_step}: {exc}"
            ) from exc
        if not isinstance(value, dict) or type(value.get("step")) is not int:
            raise AutoResumeReconcileError(
                f"metrics row {expected_step} has no exact integer step"
            )
        if value["step"] != expected_step:
            raise AutoResumeReconcileError(
                "metrics steps are not exactly contiguous from 1: "
                f"row={expected_step}, step={value['step']!r}"
            )
    return lines


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise AutoResumeReconcileError(f"cannot fsync directory {path}: {exc}") from exc


def _mkdir(
    path: Path, *, parents: bool = False, allow_existing: bool = True
) -> None:
    try:
        path.mkdir(mode=0o700, parents=parents, exist_ok=False)
    except FileExistsError as exc:
        if not allow_existing:
            raise AutoResumeReconcileError(
                f"evidence directory already exists: {path}"
            ) from exc
        if not path.is_dir() or path.is_symlink():
            raise AutoResumeReconcileError(f"unsafe evidence directory: {path}")
        return
    except OSError as exc:
        raise AutoResumeReconcileError(f"cannot create directory {path}: {exc}") from exc
    _fsync_dir(path.parent)


def _atomic_exclusive(path: Path, raw: bytes) -> None:
    """Publish a fully fsynced file without ever replacing an existing path."""

    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise AutoResumeReconcileError(f"refusing evidence overwrite: {path}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            handle = os.fdopen(descriptor, "wb", buffering=0)
            descriptor = -1
            with handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.link(temporary, path)
        os.unlink(temporary)
        _fsync_dir(path.parent)
    except FileExistsError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise AutoResumeReconcileError(f"refusing evidence overwrite: {path}") from exc
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise AutoResumeReconcileError(f"cannot publish {path}: {exc}") from exc


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _claim_guard(path: Path, value: dict[str, Any]) -> None:
    """Create the replay guard directly and durably with O_EXCL.

    A crash during this tiny window intentionally leaves a conservative guard;
    automatic retry then stops for human audit instead of risking a second
    replay of the same deterministic failure.
    """

    raw = _json_bytes(value)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise AutoResumeReplayLimitError(
            f"checkpoint replay already consumed; guard exists: {path}"
        ) from exc
    except OSError as exc:
        raise AutoResumeReconcileError(f"cannot claim replay guard {path}: {exc}") from exc
    try:
        handle = os.fdopen(descriptor, "wb", buffering=0)
        descriptor = -1
        with handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_dir(path.parent)


def reconcile_metrics_for_resume(
    *,
    checkpoint_path: str | os.PathLike[str],
    metrics_path: str | os.PathLike[str],
    checkpoint_step: int,
    evidence_root: str | os.PathLike[str] | None = None,
    environ: dict[str, str] | None = None,
    _fault_inject: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Reconcile a strict metrics stream to a loaded checkpoint on rank 0.

    Returns ``{"mode": "disabled"}`` when the gate is off and
    ``{"mode": "already_at_checkpoint"}`` without filesystem mutation when
    the stream already ends exactly at ``checkpoint_step``.
    """

    env = dict(os.environ if environ is None else environ)
    if not auto_resume_reconcile_enabled(env):
        return {"mode": "disabled", "checkpoint_step": checkpoint_step}
    bindings = _bindings_from_env(env)
    checkpoint = Path(checkpoint_path)
    metrics = Path(metrics_path)
    checkpoint_identity = _checkpoint_identity(checkpoint, checkpoint_step)
    if not metrics.exists() and not metrics.is_symlink():
        if checkpoint_step == 0:
            return {"mode": "already_at_checkpoint", "checkpoint_step": 0}
        raise AutoResumeReconcileError(
            f"metrics stream is missing for checkpoint step {checkpoint_step}: {metrics}"
        )
    metrics_raw, metrics_identity = _read_stable(metrics, allow_empty=True)
    lines = _parse_metrics(metrics_raw)
    last_step = len(lines)
    if last_step < checkpoint_step:
        raise AutoResumeReconcileError(
            f"metrics end before checkpoint: metrics={last_step}, checkpoint={checkpoint_step}"
        )
    if last_step == checkpoint_step:
        return {
            "mode": "already_at_checkpoint",
            "checkpoint_step": checkpoint_step,
            "metrics_sha256": hashlib.sha256(metrics_raw).hexdigest(),
            "checkpoint_identity_digest": checkpoint_identity["identity_digest"],
        }

    root = (
        Path(evidence_root)
        if evidence_root is not None
        else checkpoint.parent / "auto_resume_reconcile_v1"
    )
    _mkdir(root, parents=True)
    claims = root / "claims"
    attempts = root / "attempts"
    _mkdir(claims)
    _mkdir(attempts)
    digest = checkpoint_identity["identity_digest"]
    guard_path = claims / f"{digest}.json"
    if guard_path.exists() or guard_path.is_symlink():
        raise AutoResumeReplayLimitError(
            f"checkpoint replay already consumed; guard exists: {guard_path}"
        )

    # JCC retries retain the same job name/hostname prefix.  The preflight
    # nonce is regenerated for every launch, so bind the evidence namespace to
    # both values: a later retry from a *new* checkpoint must not collide with
    # an earlier launch's receipt, while the checkpoint-digest guard above
    # still blocks replaying the same deterministic failure twice.
    attempt_token = f"{bindings['jobtag']}.{bindings['attempt_nonce']}"
    attempt_dir = attempts / attempt_token
    if attempt_dir.exists() or attempt_dir.is_symlink():
        raise AutoResumeReconcileError(
            f"attempt evidence namespace already exists: {attempt_dir}"
        )
    _mkdir(attempt_dir, allow_existing=False)

    prefix_raw = b"".join(lines[:checkpoint_step])
    tail_raw = b"".join(lines[checkpoint_step:])
    before_sha = hashlib.sha256(metrics_raw).hexdigest()
    prefix_sha = hashlib.sha256(prefix_raw).hexdigest()
    tail_sha = hashlib.sha256(tail_raw).hexdigest()
    backup_path = attempt_dir / f"metrics.steps1-{last_step}.jsonl"
    tail_path = attempt_dir / (
        f"metrics.discarded.steps{checkpoint_step + 1}-{last_step}.jsonl"
    )
    _atomic_exclusive(backup_path, metrics_raw)
    _atomic_exclusive(tail_path, tail_raw)
    if _fault_inject is not None:
        _fault_inject("after_backups")

    # Inputs must still be the exact files sampled above before replay is
    # authorized.  The checkpoint sample identity is re-established as well.
    if _regular_identity(metrics, allow_empty=True) != metrics_identity:
        raise AutoResumeReconcileError("metrics changed before replay guard claim")
    if _checkpoint_identity(checkpoint, checkpoint_step) != checkpoint_identity:
        raise AutoResumeReconcileError("checkpoint changed before replay guard claim")

    guard = {
        "schema_version": SCHEMA_VERSION,
        "status": "REPLAY_AUTHORIZED",
        "bindings": bindings,
        "checkpoint": checkpoint_identity,
        "metrics_before_sha256": before_sha,
        "metrics_before_last_step": last_step,
        "resume_step": checkpoint_step,
        "attempt_receipt": str((attempt_dir / "receipt.json").resolve(strict=False)),
        "created_unix_ns": time.time_ns(),
    }
    _claim_guard(guard_path, guard)
    if _fault_inject is not None:
        _fault_inject("after_guard")

    temporary = metrics.with_name(
        f".{metrics.name}.auto-resume.{attempt_token}.tmp"
    )
    if temporary.exists() or temporary.is_symlink():
        raise AutoResumeReconcileError(f"stale metrics cutover temporary: {temporary}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            handle = os.fdopen(descriptor, "wb", buffering=0)
            descriptor = -1
            with handle:
                handle.write(prefix_raw)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if _regular_identity(metrics, allow_empty=True) != metrics_identity:
            raise AutoResumeReconcileError("metrics changed before atomic cutover")
        if _checkpoint_identity(checkpoint, checkpoint_step) != checkpoint_identity:
            raise AutoResumeReconcileError("checkpoint changed before atomic cutover")
        os.replace(temporary, metrics)
        _fsync_dir(metrics.parent)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    if _fault_inject is not None:
        _fault_inject("after_cutover")

    active_raw, active_identity = _read_stable(metrics, allow_empty=True)
    active_lines = _parse_metrics(active_raw)
    if active_raw != prefix_raw or len(active_lines) != checkpoint_step:
        raise AutoResumeReconcileError("active metrics verification failed after cutover")
    backup_raw, backup_identity = _read_stable(backup_path, allow_empty=True)
    saved_tail_raw, tail_identity = _read_stable(tail_path, allow_empty=True)
    if backup_raw != metrics_raw or saved_tail_raw != tail_raw:
        raise AutoResumeReconcileError("metrics evidence verification failed")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "mode": "trim_to_checkpoint",
        "bindings": bindings,
        "checkpoint": checkpoint_identity,
        "metrics_before": {
            **metrics_identity,
            "rows": last_step,
            "last_step": last_step,
            "sha256": before_sha,
            "backup_path": str(backup_path.resolve(strict=True)),
            "backup_identity": backup_identity,
        },
        "discarded_tail": {
            **tail_identity,
            "rows": last_step - checkpoint_step,
            "first_step": checkpoint_step + 1,
            "last_step": last_step,
            "sha256": tail_sha,
            "path": str(tail_path.resolve(strict=True)),
        },
        "active_metrics": {
            **active_identity,
            "rows": checkpoint_step,
            "last_step": checkpoint_step,
            "sha256": prefix_sha,
            "path": str(metrics.resolve(strict=True)),
        },
        "guard_path": str(guard_path.resolve(strict=True)),
        "completed_unix_ns": time.time_ns(),
    }
    receipt_path = attempt_dir / "receipt.json"
    receipt_raw = _json_bytes(receipt)
    _atomic_exclusive(receipt_path, receipt_raw)
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    success_path = attempt_dir / "_SUCCESS"
    _atomic_exclusive(
        success_path,
        (
            f"PASS|{bindings['jobtag']}|{digest}|{checkpoint_step}|"
            f"{last_step}|{receipt_sha}\n"
        ).encode("ascii"),
    )
    _fsync_dir(attempt_dir)
    return {
        "mode": "trim_to_checkpoint",
        "checkpoint_step": checkpoint_step,
        "discarded_rows": last_step - checkpoint_step,
        "metrics_before_last_step": last_step,
        "checkpoint_identity_digest": digest,
        "receipt_path": str(receipt_path.resolve(strict=True)),
        "receipt_sha256": receipt_sha,
    }
