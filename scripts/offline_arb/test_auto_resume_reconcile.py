#!/usr/bin/env python3
"""CPU-only known-answer and fault-injection tests for auto resume."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from training.auto_resume_reconcile import (
    AutoResumeReconcileError,
    AutoResumeReplayLimitError,
    auto_resume_reconcile_enabled,
    reconcile_metrics_for_resume,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def bindings(jobtag: str, *, nonce: str = SHA_A) -> dict[str, str]:
    return {
        "OFFLINE_ARB_AUTO_RESUME_RECONCILE": "1",
        "OFFLINE_ARB_ATTEMPT_JOBTAG": jobtag,
        "OFFLINE_ARB_ATTEMPT_NONCE": nonce,
        "OFFLINE_ARB_AUTO_RESUME_FREEZE_SHA256": SHA_B,
        "OFFLINE_ARB_AUTO_RESUME_PLAN_HEADER_SHA256": SHA_C,
    }


def metrics_bytes(last_step: int) -> bytes:
    return b"".join(
        json.dumps(
            {"step": step, "loss": round(step / 100.0, 4)},
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
        for step in range(1, last_step + 1)
    )


def setup_case(root: Path, *, checkpoint_step: int, metrics_last: int) -> tuple[Path, Path]:
    checkpoint = root / "last.pt"
    metrics = root / "metrics.jsonl"
    checkpoint.write_bytes(b"checkpoint-known-answer\0" + bytes(range(256)) * 9000)
    metrics.write_bytes(metrics_bytes(metrics_last))
    return checkpoint, metrics


def expect(exc_type, function, contains: str | None = None):
    try:
        function()
    except exc_type as exc:
        if contains is not None:
            assert contains in str(exc), (contains, str(exc))
        return exc
    raise AssertionError(f"expected {exc_type.__name__}")


def test_enable_gate() -> None:
    assert auto_resume_reconcile_enabled({}) is False
    assert auto_resume_reconcile_enabled(
        {"OFFLINE_ARB_AUTO_RESUME_RECONCILE": "0"}
    ) is False
    assert auto_resume_reconcile_enabled(
        {"OFFLINE_ARB_AUTO_RESUME_RECONCILE": "1"}
    ) is True
    expect(
        AutoResumeReconcileError,
        lambda: auto_resume_reconcile_enabled(
            {"OFFLINE_ARB_AUTO_RESUME_RECONCILE": "true"}
        ),
        "exactly 0 or 1",
    )


def test_disabled_does_not_touch_inputs(root: Path) -> None:
    result = reconcile_metrics_for_resume(
        checkpoint_path=root / "missing.pt",
        metrics_path=root / "missing.jsonl",
        checkpoint_step=123,
        environ={"OFFLINE_ARB_AUTO_RESUME_RECONCILE": "0"},
    )
    assert result == {"mode": "disabled", "checkpoint_step": 123}
    assert list(root.iterdir()) == []


def test_exact_boundary_is_true_noop(root: Path) -> None:
    checkpoint, metrics = setup_case(root, checkpoint_step=5, metrics_last=5)
    evidence = root / "evidence"
    before = metrics.read_bytes()
    before_stat = metrics.stat()
    result = reconcile_metrics_for_resume(
        checkpoint_path=checkpoint,
        metrics_path=metrics,
        checkpoint_step=5,
        evidence_root=evidence,
        environ=bindings("formal-noop"),
    )
    assert result["mode"] == "already_at_checkpoint"
    assert result["metrics_sha256"] == hashlib.sha256(before).hexdigest()
    assert metrics.read_bytes() == before
    after_stat = metrics.stat()
    assert (after_stat.st_ino, after_stat.st_size, after_stat.st_mtime_ns) == (
        before_stat.st_ino,
        before_stat.st_size,
        before_stat.st_mtime_ns,
    )
    assert not evidence.exists(), "exact boundary must not consume replay evidence"


def test_trim_evidence_and_same_checkpoint_guard(root: Path) -> None:
    checkpoint, metrics = setup_case(root, checkpoint_step=5, metrics_last=8)
    evidence = root / "evidence"
    original = metrics.read_bytes()
    result = reconcile_metrics_for_resume(
        checkpoint_path=checkpoint,
        metrics_path=metrics,
        checkpoint_step=5,
        evidence_root=evidence,
        environ=bindings("formal-retry-one"),
    )
    assert result["mode"] == "trim_to_checkpoint"
    assert result["discarded_rows"] == 3
    assert metrics.read_bytes() == metrics_bytes(5)
    receipt_path = Path(result["receipt_path"])
    receipt_raw = receipt_path.read_bytes()
    assert hashlib.sha256(receipt_raw).hexdigest() == result["receipt_sha256"]
    receipt = json.loads(receipt_raw)
    assert receipt["schema_version"] == "offline_arb_auto_resume_reconcile_v1"
    assert receipt["status"] == "PASS"
    assert receipt["checkpoint"]["global_step"] == 5
    assert receipt["metrics_before"]["rows"] == 8
    assert receipt["metrics_before"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert receipt["discarded_tail"]["rows"] == 3
    assert receipt["discarded_tail"]["sha256"] == hashlib.sha256(
        b"".join(original.splitlines(keepends=True)[5:])
    ).hexdigest()
    assert Path(receipt["metrics_before"]["backup_path"]).read_bytes() == original
    assert Path(receipt["discarded_tail"]["path"]).read_bytes() == b"".join(
        original.splitlines(keepends=True)[5:]
    )
    success = receipt_path.with_name("_SUCCESS").read_text(encoding="ascii")
    assert success.startswith("PASS|formal-retry-one|")

    # Reproduce the same deterministic post-checkpoint failure.  The unchanged
    # checkpoint identity has already consumed its single real replay.
    metrics.write_bytes(metrics_bytes(8))
    expect(
        AutoResumeReplayLimitError,
        lambda: reconcile_metrics_for_resume(
            checkpoint_path=checkpoint,
            metrics_path=metrics,
            checkpoint_step=5,
            evidence_root=evidence,
            environ=bindings("formal-retry-two"),
        ),
        "already consumed",
    )
    assert metrics.read_bytes() == metrics_bytes(8)
    assert not (evidence / "attempts" / "formal-retry-two").exists()


def test_new_checkpoint_identity_gets_new_guard(root: Path) -> None:
    checkpoint, metrics = setup_case(root, checkpoint_step=5, metrics_last=7)
    evidence = root / "evidence"
    first = reconcile_metrics_for_resume(
        checkpoint_path=checkpoint,
        metrics_path=metrics,
        checkpoint_step=5,
        evidence_root=evidence,
        environ=bindings("formal-retry-job", nonce=SHA_A),
    )
    checkpoint.write_bytes(checkpoint.read_bytes() + b"new-checkpoint-step-6")
    metrics.write_bytes(metrics_bytes(8))
    second = reconcile_metrics_for_resume(
        checkpoint_path=checkpoint,
        metrics_path=metrics,
        checkpoint_step=6,
        evidence_root=evidence,
        environ=bindings("formal-retry-job", nonce="d" * 64),
    )
    assert first["checkpoint_identity_digest"] != second["checkpoint_identity_digest"]
    assert metrics.read_bytes() == metrics_bytes(6)
    assert len(list((evidence / "claims").glob("*.json"))) == 2
    attempt_dirs = sorted(path.name for path in (evidence / "attempts").iterdir())
    assert attempt_dirs == [
        f"formal-retry-job.{SHA_A}",
        f"formal-retry-job.{'d' * 64}",
    ]


def test_fault_before_guard_is_retryable(root: Path) -> None:
    checkpoint, metrics = setup_case(root, checkpoint_step=5, metrics_last=8)
    evidence = root / "evidence"

    def fail(stage: str) -> None:
        if stage == "after_backups":
            raise RuntimeError("injected-before-guard")

    expect(
        RuntimeError,
        lambda: reconcile_metrics_for_resume(
            checkpoint_path=checkpoint,
            metrics_path=metrics,
            checkpoint_step=5,
            evidence_root=evidence,
            environ=bindings("fault-before-guard"),
            _fault_inject=fail,
        ),
        "injected-before-guard",
    )
    assert metrics.read_bytes() == metrics_bytes(8)
    assert list((evidence / "claims").iterdir()) == []
    result = reconcile_metrics_for_resume(
        checkpoint_path=checkpoint,
        metrics_path=metrics,
        checkpoint_step=5,
        evidence_root=evidence,
        environ=bindings("fault-before-guard-retry"),
    )
    assert result["mode"] == "trim_to_checkpoint"


def test_fault_after_guard_stops_replay(root: Path) -> None:
    checkpoint, metrics = setup_case(root, checkpoint_step=5, metrics_last=8)
    evidence = root / "evidence"

    def fail(stage: str) -> None:
        if stage == "after_guard":
            raise RuntimeError("injected-after-guard")

    expect(
        RuntimeError,
        lambda: reconcile_metrics_for_resume(
            checkpoint_path=checkpoint,
            metrics_path=metrics,
            checkpoint_step=5,
            evidence_root=evidence,
            environ=bindings("fault-after-guard"),
            _fault_inject=fail,
        ),
        "injected-after-guard",
    )
    assert metrics.read_bytes() == metrics_bytes(8)
    assert len(list((evidence / "claims").glob("*.json"))) == 1
    expect(
        AutoResumeReplayLimitError,
        lambda: reconcile_metrics_for_resume(
            checkpoint_path=checkpoint,
            metrics_path=metrics,
            checkpoint_step=5,
            evidence_root=evidence,
            environ=bindings("fault-after-guard-retry"),
        ),
    )


def test_input_fail_closed(root: Path) -> None:
    checkpoint, metrics = setup_case(root, checkpoint_step=5, metrics_last=4)
    expect(
        AutoResumeReconcileError,
        lambda: reconcile_metrics_for_resume(
            checkpoint_path=checkpoint,
            metrics_path=metrics,
            checkpoint_step=5,
            environ=bindings("metrics-behind"),
        ),
        "end before checkpoint",
    )
    metrics.write_bytes(metrics_bytes(7).replace(b'"step": 4', b'"step": 44'))
    expect(
        AutoResumeReconcileError,
        lambda: reconcile_metrics_for_resume(
            checkpoint_path=checkpoint,
            metrics_path=metrics,
            checkpoint_step=5,
            environ=bindings("metrics-gap"),
        ),
        "not exactly contiguous",
    )
    metrics.write_bytes(metrics_bytes(7)[:-1])
    expect(
        AutoResumeReconcileError,
        lambda: reconcile_metrics_for_resume(
            checkpoint_path=checkpoint,
            metrics_path=metrics,
            checkpoint_step=5,
            environ=bindings("metrics-partial"),
        ),
        "partial/non-newline",
    )
    metrics.unlink()
    expect(
        AutoResumeReconcileError,
        lambda: reconcile_metrics_for_resume(
            checkpoint_path=checkpoint,
            metrics_path=metrics,
            checkpoint_step=5,
            environ=bindings("metrics-missing"),
        ),
        "metrics stream is missing",
    )


def test_checkpoint_change_before_guard_is_detected(root: Path) -> None:
    checkpoint, metrics = setup_case(root, checkpoint_step=5, metrics_last=8)
    evidence = root / "evidence"

    def mutate(stage: str) -> None:
        if stage == "after_backups":
            checkpoint.write_bytes(checkpoint.read_bytes() + b"drift")

    expect(
        AutoResumeReconcileError,
        lambda: reconcile_metrics_for_resume(
            checkpoint_path=checkpoint,
            metrics_path=metrics,
            checkpoint_step=5,
            evidence_root=evidence,
            environ=bindings("checkpoint-drift"),
            _fault_inject=mutate,
        ),
        "checkpoint changed",
    )
    assert metrics.read_bytes() == metrics_bytes(8)
    assert list((evidence / "claims").iterdir()) == []


def run() -> None:
    test_enable_gate()
    tests = (
        test_disabled_does_not_touch_inputs,
        test_exact_boundary_is_true_noop,
        test_trim_evidence_and_same_checkpoint_guard,
        test_new_checkpoint_identity_gets_new_guard,
        test_fault_before_guard_is_retryable,
        test_fault_after_guard_stops_replay,
        test_input_fail_closed,
        test_checkpoint_change_before_guard_is_detected,
    )
    for test in tests:
        with tempfile.TemporaryDirectory(prefix=f"{test.__name__}.") as directory:
            test(Path(directory))
        print(f"PASS {test.__name__}", flush=True)
    print("OFFLINE_ARB_AUTO_RESUME_RECONCILE_TEST_PASS", flush=True)


if __name__ == "__main__":
    run()
