#!/usr/bin/env python3
"""Fault-injection tests for Lustre-safe completion-root publication."""

from __future__ import annotations

import multiprocessing
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Callable

from scripts.offline_arb import completion_publication as publication


PAYLOADS = {
    "FORMAL_PASS.json": b'{"schema":"stage256_completion_for_stage512a_v2","status":"PASS"}\n',
    "FREEZE_VERIFY.json": b'{"mode":"verify-freeze","status":"PASS"}\n',
}


def expect_publish_error(call: Callable[[], object], needle: str) -> str:
    try:
        call()
    except publication.CompletionPublishError as exc:
        message = str(exc)
        assert needle in message, (needle, message)
        return message
    raise AssertionError(f"expected CompletionPublishError containing {needle!r}")


def assert_exact(root: Path, payloads: dict[str, bytes] = PAYLOADS) -> None:
    report = publication.validate_completion_directory(
        root, expected_payloads=payloads
    )
    assert report["canonical_root"] == str(root)
    assert stat.S_IMODE(root.lstat().st_mode) == 0o555
    assert set(report["members"]) == publication.COMPLETION_MEMBERS
    for name, raw in payloads.items():
        member = root / name
        metadata = member.lstat()
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o444
        assert metadata.st_nlink == 1
        assert member.read_bytes() == raw


def write_manual_root(
    root: Path,
    payloads: dict[str, bytes],
    *,
    root_mode: int = 0o555,
) -> None:
    root.mkdir(mode=0o700)
    for name, raw in payloads.items():
        path = root / name
        path.write_bytes(raw)
        os.chmod(path, 0o444)
    os.chmod(root, root_mode)


def _race_worker(
    root: str,
    barrier: multiprocessing.synchronize.Barrier,
    queue: multiprocessing.queues.Queue,
) -> None:
    try:
        barrier.wait(timeout=20)
        report = publication.publish_completion_directory(Path(root), PAYLOADS)
        queue.put(("ok", report["publication_status"]))
    except BaseException as exc:
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def main() -> int:
    # Fresh exclusive claim, exact commit, and mutation-free retry adoption.
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        report = publication.publish_completion_directory(root, PAYLOADS)
        assert report["publication_status"] == "published"
        assert report["recovered"] is False
        assert_exact(root)
        adopted = publication.publish_completion_directory(root, PAYLOADS)
        assert adopted["publication_status"] == "adopted_existing_exact"
        assert adopted["recovered"] is True
        assert_exact(root)

    # A mode-0700 exact prefix is recoverable.  FORMAL_PASS is linked last and
    # root mode 0555 is the only accepted commit state.
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        write_manual_root(
            root,
            {"FREEZE_VERIFY.json": PAYLOADS["FREEZE_VERIFY.json"]},
            root_mode=0o700,
        )
        before_inode = root.stat().st_ino
        expect_publish_error(
            lambda: publication.validate_completion_directory(
                root, expected_payloads=PAYLOADS
            ),
            "committed mode 0555",
        )
        recovered = publication.publish_completion_directory(root, PAYLOADS)
        assert recovered["publication_status"] == "recovered_uncommitted_exact"
        assert root.stat().st_ino == before_inode
        assert_exact(root)

    # Empty reservations and recognized deterministic short temporaries are
    # recoverable because no authoritative final member is wrong.
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        root.mkdir(mode=0o700)
        temp_name = publication._temporary_name(
            "FREEZE_VERIFY.json", PAYLOADS["FREEZE_VERIFY.json"]
        )
        (root / temp_name).write_bytes(b"killed-short-write")
        os.chmod(root / temp_name, 0o444)
        recovered = publication.publish_completion_directory(root, PAYLOADS)
        assert recovered["publication_status"] == "recovered_uncommitted_exact"
        assert_exact(root)

    # A failure after the prerequisite member link leaves a recoverable 0700
    # prefix.  It is preserved, then an exact retry completes it in place.
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        original_publish_member = publication._publish_or_verify_member
        calls = 0

        def fail_after_first(directory_fd: int, name: str, raw: bytes) -> None:
            nonlocal calls
            original_publish_member(directory_fd, name, raw)
            calls += 1
            if calls == 1:
                raise OSError("injected crash after prerequisite link")

        publication._publish_or_verify_member = fail_after_first
        try:
            expect_publish_error(
                lambda: publication.publish_completion_directory(root, PAYLOADS),
                "failed before commit",
            )
        finally:
            publication._publish_or_verify_member = original_publish_member
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert set(path.name for path in root.iterdir()) == {"FREEZE_VERIFY.json"}
        recovered = publication.publish_completion_directory(root, PAYLOADS)
        assert recovered["publication_status"] == "recovered_uncommitted_exact"
        assert_exact(root)

    # If root fsync reports an error after fchmod committed 0555, exact
    # revalidation recovers without ever changing canonical state back.
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        original_fsync = publication.os.fsync
        injected = False

        def fail_first_committed_root_fsync(descriptor: int) -> None:
            nonlocal injected
            metadata = os.fstat(descriptor)
            if (
                not injected
                and stat.S_ISDIR(metadata.st_mode)
                and stat.S_IMODE(metadata.st_mode) == 0o555
                and metadata.st_ino == root.stat().st_ino
            ):
                injected = True
                raise OSError("injected committed-root fsync failure")
            original_fsync(descriptor)

        publication.os.fsync = fail_first_committed_root_fsync
        try:
            recovered = publication.publish_completion_directory(root, PAYLOADS)
        finally:
            publication.os.fsync = original_fsync
        assert injected
        assert recovered["publication_status"] == "recovered_after_commit_exception"
        assert_exact(root)

    # Wrong or structurally impossible uncommitted roots are preserved and
    # never auto-repaired.
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        wrong = b'{"status":"WRONG"}\n'
        write_manual_root(
            root, {"FREEZE_VERIFY.json": wrong}, root_mode=0o700
        )
        inode = (root / "FREEZE_VERIFY.json").stat().st_ino
        expect_publish_error(
            lambda: publication.publish_completion_directory(root, PAYLOADS),
            "differs in bytes/mode/link closure",
        )
        assert (root / "FREEZE_VERIFY.json").read_bytes() == wrong
        assert (root / "FREEZE_VERIFY.json").stat().st_ino == inode
        assert stat.S_IMODE(root.stat().st_mode) == 0o700

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        write_manual_root(
            root,
            {"FORMAL_PASS.json": PAYLOADS["FORMAL_PASS.json"]},
            root_mode=0o700,
        )
        expect_publish_error(
            lambda: publication.publish_completion_directory(root, PAYLOADS),
            "commit member exists before",
        )
        assert set(path.name for path in root.iterdir()) == {"FORMAL_PASS.json"}

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        root.mkdir(mode=0o700)
        (root / "UNKNOWN").write_bytes(b"preserve\n")
        unknown_inode = (root / "UNKNOWN").stat().st_ino
        expect_publish_error(
            lambda: publication.publish_completion_directory(root, PAYLOADS),
            "unknown members",
        )
        assert (root / "UNKNOWN").stat().st_ino == unknown_inode

    # Committed partial/different roots are mutation-free blockers.
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        different = dict(PAYLOADS)
        different["FORMAL_PASS.json"] = b'{"status":"DIFFERENT"}\n'
        write_manual_root(root, different)
        expect_publish_error(
            lambda: publication.publish_completion_directory(root, PAYLOADS),
            "differs from expected bytes",
        )
        assert (root / "FORMAL_PASS.json").read_bytes() == different[
            "FORMAL_PASS.json"
        ]

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        write_manual_root(root, {"FORMAL_PASS.json": PAYLOADS["FORMAL_PASS.json"]})
        expect_publish_error(
            lambda: publication.publish_completion_directory(root, PAYLOADS),
            "member set drift",
        )
        assert (root / "FORMAL_PASS.json").is_file()

    # Legacy staging debris blocks both publication and adoption.
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        publication.publish_completion_directory(root, PAYLOADS)
        conflict = parent / ".completion-v2.staging.conflict"
        conflict.mkdir()
        expect_publish_error(
            lambda: publication.publish_completion_directory(root, PAYLOADS),
            "legacy completion staging conflict",
        )
        publication.validate_completion_directory(
            root, expected_payloads=PAYLOADS, reject_staging_conflicts=False
        )
        conflict.rmdir()
        assert_exact(root)

    # Namespace aliases and traversal spellings are rejected before claim.
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        real_parent = parent / "real"
        real_parent.mkdir()
        alias_parent = parent / "alias"
        alias_parent.symlink_to(real_parent, target_is_directory=True)
        expect_publish_error(
            lambda: publication.publish_completion_directory(
                alias_parent / "completion-v2", PAYLOADS
            ),
            "symlink alias",
        )
        child = real_parent / "child"
        child.mkdir()
        traversal = Path(f"{child}/../completion-v2")
        expect_publish_error(
            lambda: publication.publish_completion_directory(traversal, PAYLOADS),
            "must not contain '..'",
        )
        assert not (real_parent / "completion-v2").exists()

    # Canonical/member symlinks, extras, and hard-linked members are rejected.
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        target = parent / "target"
        write_manual_root(target, PAYLOADS)
        root = parent / "completion-v2"
        root.symlink_to(target, target_is_directory=True)
        expect_publish_error(
            lambda: publication.publish_completion_directory(root, PAYLOADS),
            "cannot open real child directory",
        )

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        publication.publish_completion_directory(root, PAYLOADS)
        os.chmod(root, 0o755)
        (root / "EXTRA").write_text("unexpected\n", encoding="utf-8")
        os.chmod(root, 0o555)
        expect_publish_error(
            lambda: publication.validate_completion_directory(root),
            "member set drift",
        )

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        publication.publish_completion_directory(root, PAYLOADS)
        alias = parent / "proof-hardlink-alias"
        os.link(root / "FORMAL_PASS.json", alias)
        expect_publish_error(
            lambda: publication.validate_completion_directory(root),
            "completion member closure failed",
        )

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        target = parent / "target-member"
        target.write_bytes(PAYLOADS["FORMAL_PASS.json"])
        root = parent / "completion-v2"
        root.mkdir(mode=0o700)
        (root / "FORMAL_PASS.json").symlink_to(target)
        freeze = root / "FREEZE_VERIFY.json"
        freeze.write_bytes(PAYLOADS["FREEZE_VERIFY.json"])
        os.chmod(freeze, 0o444)
        os.chmod(root, 0o555)
        expect_publish_error(
            lambda: publication.validate_completion_directory(root),
            "not a regular file",
        )
        os.chmod(root, 0o755)

    # Stable reads reject member/root metadata mutation even when bytes match.
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        publication.publish_completion_directory(root, PAYLOADS)
        original_read_fd = publication._read_fd
        read_finished = threading.Event()
        mutation_finished = threading.Event()
        outcome: list[BaseException] = []
        first_read = True

        def delayed_read(descriptor: int) -> bytes:
            nonlocal first_read
            raw = original_read_fd(descriptor)
            if first_read:
                first_read = False
                read_finished.set()
                assert mutation_finished.wait(timeout=10)
            return raw

        def validate_in_thread() -> None:
            try:
                publication.validate_completion_directory(
                    root, expected_payloads=PAYLOADS
                )
            except BaseException as exc:
                outcome.append(exc)

        publication._read_fd = delayed_read
        try:
            validator = threading.Thread(target=validate_in_thread)
            validator.start()
            assert read_finished.wait(timeout=10)
            member = root / "FORMAL_PASS.json"
            metadata = member.stat()
            os.utime(
                member,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 10_000_000),
            )
            mutation_finished.set()
            validator.join(timeout=10)
            assert not validator.is_alive()
        finally:
            mutation_finished.set()
            publication._read_fd = original_read_fd
        assert len(outcome) == 1 and isinstance(
            outcome[0], publication.CompletionPublishError
        ), outcome
        assert "changed during stable read" in str(outcome[0]), outcome[0]

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        publication.publish_completion_directory(root, PAYLOADS)
        original_read_fd = publication._read_fd
        mutated_root = False

        def mutate_root_after_read(descriptor: int) -> bytes:
            nonlocal mutated_root
            raw = original_read_fd(descriptor)
            if not mutated_root:
                mutated_root = True
                metadata = root.stat()
                os.utime(
                    root,
                    ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 10_000_000),
                )
            return raw

        publication._read_fd = mutate_root_after_read
        try:
            expect_publish_error(
                lambda: publication.validate_completion_directory(
                    root, expected_payloads=PAYLOADS
                ),
                "root changed during stable validation",
            )
        finally:
            publication._read_fd = original_read_fd

    # Four independent processes race through the real mkdir/link/flock/mode
    # protocol.  Every successful report must converge on one exact root.
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        root = parent / "completion-v2"
        context = multiprocessing.get_context("fork")
        barrier = context.Barrier(4)
        queue = context.Queue()
        workers = [
            context.Process(
                target=_race_worker,
                args=(str(root), barrier, queue),
            )
            for _ in range(4)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30)
            assert not worker.is_alive()
            assert worker.exitcode == 0
        results = [queue.get(timeout=5) for _ in workers]
        assert all(kind == "ok" for kind, _ in results), results
        assert_exact(root)
        assert not any(".tmp." in path.name for path in root.iterdir())

    print(
        "COMPLETION_PUBLICATION_PASS protocol=lustre_mkdir_link_mode_commit_v1 "
        "exact_members=2 root_mode=0555 member_mode=0444 nlink=1 "
        "fresh=PASS exact_adoption=PASS partial_exact_recovery=PASS "
        "wrong_unknown_impossible=REJECTED committed_partial_different=REJECTED "
        "post_commit_recovery=PASS traversal_alias_symlink=REJECTED "
        "stable_member_root_fingerprints=PASS concurrent_exact=PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
