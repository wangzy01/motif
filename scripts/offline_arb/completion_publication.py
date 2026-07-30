#!/usr/bin/env python3
"""Publish and validate the immutable stage256 completion root on Lustre.

FSx for Lustre 2.15 rejects directory ``RENAME_NOREPLACE``.  Publication
therefore uses a recoverable, no-clobber state machine whose namespace
operations are all relative to stable directory file descriptors:

* ``mkdirat`` reserves the canonical basename as a mode-0700 directory;
* deterministic temporary files are sealed and installed with ``linkat``;
* ``FREEZE_VERIFY.json`` is installed before the commit member
  ``FORMAL_PASS.json``;
* an exact mode-0700 prefix may be recovered, while any wrong/unknown state is
  preserved and rejected; and
* ``fchmod(0700 -> 0555)`` is the accepted-state commit bit.

Consumers accept only a real mode-0555 directory containing exactly two
regular mode-0444, single-link members.  A committed prior publication may be
adopted only when its bytes and stable fingerprints match exactly.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any


COMPLETION_MEMBERS = frozenset({"FORMAL_PASS.json", "FREEZE_VERIFY.json"})
COMMIT_MEMBER = "FORMAL_PASS.json"
RECOVERY_ROOT_MODE = 0o700
COMMITTED_ROOT_MODE = 0o555
MEMBER_MODE = 0o444


class CompletionPublishError(RuntimeError):
    pass


def json_document_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    """Stable-read identity, including content/namespace change clocks."""

    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _object_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    """Identity that survives the intentional root chmod at commit."""

    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        stat.S_IFMT(metadata.st_mode),
    )


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    return b"".join(chunks)


def _validate_entry_name(name: str, *, label: str) -> None:
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
        or os.path.altsep is not None
        and os.path.altsep in name
    ):
        raise CompletionPublishError(f"invalid {label} basename: {name!r}")


def _normalized_real_root(canonical_root: Path) -> tuple[Path, Path, str]:
    raw = os.fspath(canonical_root)
    if not os.path.isabs(raw):
        raise CompletionPublishError("completion root must be absolute")
    candidate = Path(raw)
    if ".." in candidate.parts:
        raise CompletionPublishError("completion root must not contain '..'")
    normalized = os.path.normpath(raw)
    if normalized != raw:
        raise CompletionPublishError("completion root must be lexically normalized")
    root = Path(normalized)
    basename = root.name
    _validate_entry_name(basename, label="completion root")
    parent = root.parent
    try:
        real_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise CompletionPublishError(
            f"cannot resolve completion parent directory {parent}: {exc}"
        ) from exc
    if real_parent != parent:
        raise CompletionPublishError(
            f"completion parent must be its real path, not a symlink alias: {parent}"
        )
    return root, parent, basename


def _directory_open_flags() -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _open_real_directory(path: Path) -> tuple[int, os.stat_result]:
    try:
        descriptor = os.open(path, _directory_open_flags())
    except OSError as exc:
        raise CompletionPublishError(f"cannot open real directory {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _fingerprint(opened) != _fingerprint(named)
        ):
            raise CompletionPublishError(
                f"directory pathname/inode mismatch or symlink: {path}"
            )
        return descriptor, opened
    except Exception:
        os.close(descriptor)
        raise


def _open_child_directory(parent_fd: int, name: str) -> tuple[int, os.stat_result]:
    _validate_entry_name(name, label="directory")
    try:
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise CompletionPublishError(
            f"cannot open real child directory {name!r}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _fingerprint(opened) != _fingerprint(named)
        ):
            raise CompletionPublishError(
                f"child directory pathname/inode mismatch or symlink: {name!r}"
            )
        return descriptor, opened
    except Exception:
        os.close(descriptor)
        raise


def _assert_parent_anchor(
    parent_fd: int, parent_path: Path, parent_initial: os.stat_result
) -> None:
    opened = os.fstat(parent_fd)
    try:
        named = parent_path.lstat()
        resolved = parent_path.resolve(strict=True)
    except OSError as exc:
        raise CompletionPublishError(
            f"completion parent path changed during operation: {parent_path}: {exc}"
        ) from exc
    if (
        resolved != parent_path
        or not stat.S_ISDIR(opened.st_mode)
        or _object_identity(opened) != _object_identity(named)
        or _object_identity(opened) != _object_identity(parent_initial)
    ):
        raise CompletionPublishError(
            f"completion parent anchor changed during operation: {parent_path}"
        )


def _entry_stat(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _staging_prefix(canonical_basename: str) -> str:
    """Legacy prefix retained only to fail closed on old transaction debris."""

    return f".{canonical_basename}.staging."


def _require_no_legacy_staging(parent_fd: int, canonical_basename: str) -> None:
    prefix = _staging_prefix(canonical_basename)
    conflicts = sorted(name for name in os.listdir(parent_fd) if name.startswith(prefix))
    if conflicts:
        raise CompletionPublishError(
            "legacy completion staging conflict prevents publication/adoption: "
            + ", ".join(conflicts)
        )


def _stable_regular_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise CompletionPublishError(f"{label} is not a regular file")
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        opened = os.fstat(descriptor)
        if _fingerprint(before) != _fingerprint(opened):
            raise CompletionPublishError(f"{label} changed before open")
        raw = _read_fd(descriptor)
        after_fd = os.fstat(descriptor)
        after_name = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not (
            _fingerprint(before)
            == _fingerprint(after_fd)
            == _fingerprint(after_name)
        ):
            raise CompletionPublishError(f"{label} changed during stable read")
        return raw, after_fd
    except OSError as exc:
        raise CompletionPublishError(f"cannot read {label}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_final_member(
    directory_fd: int,
    name: str,
    raw: bytes,
    *,
    allow_linked_temporary: bool = False,
) -> dict[str, Any]:
    observed, metadata = _stable_regular_at(
        directory_fd, name, label=f"completion member {name}"
    )
    allowed_links = {1, 2} if allow_linked_temporary else {1}
    if (
        observed != raw
        or stat.S_IMODE(metadata.st_mode) != MEMBER_MODE
        or metadata.st_nlink not in allowed_links
    ):
        raise CompletionPublishError(
            f"completion member differs in bytes/mode/link closure: {name}"
        )
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mode": "0444",
        "nlink": 1,
    }


def _temporary_name(name: str, raw: bytes) -> str:
    if name not in COMPLETION_MEMBERS:
        raise CompletionPublishError(f"invalid completion member: {name!r}")
    return f".{name}.tmp.{hashlib.sha256(raw).hexdigest()}"


def _write_temporary(directory_fd: int, name: str, raw: bytes) -> os.stat_result:
    temporary = _temporary_name(name, raw)
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o400, dir_fd=directory_fd)
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise CompletionPublishError(
                    f"short write for completion temporary {temporary}"
                )
            offset += written
        os.fchmod(descriptor, MEMBER_MODE)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        named = os.stat(temporary, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != MEMBER_MODE
            or opened.st_nlink != 1
            or _fingerprint(opened) != _fingerprint(named)
            or _read_fd(descriptor) != raw
        ):
            raise CompletionPublishError(
                f"completion temporary did not seal exactly: {temporary}"
            )
        after = os.fstat(descriptor)
        named_after = os.stat(temporary, dir_fd=directory_fd, follow_symlinks=False)
        if not (
            _fingerprint(opened) == _fingerprint(after) == _fingerprint(named_after)
        ):
            raise CompletionPublishError(
                f"completion temporary changed during readback: {temporary}"
            )
        return after
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _unlink_known_temporary(
    directory_fd: int,
    temporary: str,
    *,
    expected_identity: os.stat_result | None = None,
) -> None:
    current = _entry_stat(directory_fd, temporary)
    if current is None:
        return
    if not stat.S_ISREG(current.st_mode):
        raise CompletionPublishError(
            f"recognized completion temporary is not regular: {temporary}"
        )
    if expected_identity is not None and _object_identity(current) != _object_identity(
        expected_identity
    ):
        raise CompletionPublishError(
            f"completion temporary inode changed before cleanup: {temporary}"
        )
    os.unlink(temporary, dir_fd=directory_fd)
    os.fsync(directory_fd)


def _inventory_recovery_root(
    directory_fd: int,
    payloads: Mapping[str, bytes],
) -> None:
    expected_names = set(COMPLETION_MEMBERS)
    temporary_names = {
        _temporary_name(name, raw): name for name, raw in payloads.items()
    }
    observed = set(os.listdir(directory_fd))
    unknown = observed - expected_names - set(temporary_names)
    if unknown:
        raise CompletionPublishError(
            f"uncommitted completion root contains unknown members: {sorted(unknown)}"
        )
    final_names = observed & expected_names
    if COMMIT_MEMBER in final_names and final_names != expected_names:
        raise CompletionPublishError(
            "completion commit member exists before the exact prerequisite set"
        )

    # Validate every authoritative final name before removing any recognized,
    # explicitly non-authoritative temporary.
    for name in sorted(final_names):
        _require_final_member(
            directory_fd,
            name,
            payloads[name],
            allow_linked_temporary=True,
        )
    for temporary in sorted(observed & set(temporary_names)):
        _stable_regular_at(
            directory_fd,
            temporary,
            label=f"recognized completion temporary {temporary}",
        )
    for temporary in sorted(observed & set(temporary_names)):
        _unlink_known_temporary(directory_fd, temporary)

    for name in sorted(final_names):
        _require_final_member(directory_fd, name, payloads[name])


def _publish_or_verify_member(directory_fd: int, name: str, raw: bytes) -> None:
    existing = _entry_stat(directory_fd, name)
    if existing is not None:
        _require_final_member(directory_fd, name, raw)
        return

    temporary = _temporary_name(name, raw)
    try:
        temporary_metadata = _write_temporary(directory_fd, name, raw)
    except FileExistsError:
        # A cooperating writer can leave only this deterministic temporary.
        # Revalidate it and let the caller's retry path resolve the race.
        observed, temporary_metadata = _stable_regular_at(
            directory_fd,
            temporary,
            label=f"concurrent completion temporary {temporary}",
        )
        if observed != raw:
            raise CompletionPublishError(
                f"concurrent completion temporary bytes differ: {temporary}"
            )
    try:
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            _require_final_member(
                directory_fd, name, raw, allow_linked_temporary=True
            )
        os.fsync(directory_fd)
    finally:
        _unlink_known_temporary(
            directory_fd,
            temporary,
            expected_identity=temporary_metadata,
        )
    _require_final_member(directory_fd, name, raw)


def _validate_completion_at(
    parent_fd: int,
    canonical_basename: str,
    *,
    expected_payloads: Mapping[str, bytes] | None,
) -> dict[str, Any]:
    if expected_payloads is not None and set(expected_payloads) != COMPLETION_MEMBERS:
        raise CompletionPublishError("expected completion payload set is not exact")
    parent_metadata = os.fstat(parent_fd)
    root_fd, root_initial = _open_child_directory(parent_fd, canonical_basename)
    try:
        if (
            root_initial.st_dev != parent_metadata.st_dev
            or stat.S_IMODE(root_initial.st_mode) != COMMITTED_ROOT_MODE
        ):
            raise CompletionPublishError(
                "completion root must be same-device committed mode 0555 directory"
            )
        members = set(os.listdir(root_fd))
        if members != COMPLETION_MEMBERS:
            raise CompletionPublishError(
                f"completion root member set drift: {sorted(members)}"
            )
        report: dict[str, Any] = {}
        for name in sorted(COMPLETION_MEMBERS):
            raw, metadata = _stable_regular_at(
                root_fd, name, label=f"completion member {name}"
            )
            if (
                metadata.st_dev != root_initial.st_dev
                or stat.S_IMODE(metadata.st_mode) != MEMBER_MODE
                or metadata.st_nlink != 1
            ):
                raise CompletionPublishError(
                    f"completion member closure failed: {name}"
                )
            if expected_payloads is not None and raw != expected_payloads[name]:
                raise CompletionPublishError(
                    f"completion member differs from expected bytes: {name}"
                )
            report[name] = {
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "mode": "0444",
                "nlink": 1,
            }

        root_after = os.fstat(root_fd)
        named_after = os.stat(
            canonical_basename, dir_fd=parent_fd, follow_symlinks=False
        )
        if not (
            _fingerprint(root_initial)
            == _fingerprint(root_after)
            == _fingerprint(named_after)
        ):
            raise CompletionPublishError(
                "completion root changed during stable validation"
            )
        return {"root_mode": "0555", "members": report}
    finally:
        os.close(root_fd)


def _publication_report(
    report: Mapping[str, Any], *, status: str, canonical_root: Path
) -> dict[str, Any]:
    return {
        **report,
        "canonical_root": str(canonical_root),
        "publication_status": status,
        "recovered": status != "published",
    }


def validate_completion_directory(
    canonical_root: Path,
    *,
    expected_payloads: Mapping[str, bytes] | None = None,
    reject_staging_conflicts: bool = True,
) -> dict[str, Any]:
    """Require exact root/member/mode/nlink/stable-read closure."""

    root, parent, basename = _normalized_real_root(canonical_root)
    parent_fd, parent_initial = _open_real_directory(parent)
    try:
        report = _validate_completion_at(
            parent_fd, basename, expected_payloads=expected_payloads
        )
        if reject_staging_conflicts:
            _require_no_legacy_staging(parent_fd, basename)
        _assert_parent_anchor(parent_fd, parent, parent_initial)
        return {**report, "canonical_root": str(root)}
    finally:
        os.close(parent_fd)


def publish_completion_directory(
    canonical_root: Path,
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    """Publish exact proof bytes with exclusive claim and crash recovery.

    Mode 0700 is explicitly uncommitted.  A retry may fill only an exact prefix
    with no unknown names and no out-of-order commit member.  Mode 0555 is
    immutable publication state and is never changed by this helper.
    """

    if set(payloads) != COMPLETION_MEMBERS:
        raise CompletionPublishError(
            f"completion payload set must be exact: {sorted(COMPLETION_MEMBERS)}"
        )
    frozen_payloads = {name: bytes(payloads[name]) for name in COMPLETION_MEMBERS}
    if any(not raw for raw in frozen_payloads.values()):
        raise CompletionPublishError("completion payloads must be non-empty")

    root, parent, basename = _normalized_real_root(canonical_root)
    parent_fd, parent_initial = _open_real_directory(parent)
    root_fd = -1
    created = False
    commit_attempted = False
    try:
        _require_no_legacy_staging(parent_fd, basename)
        try:
            os.mkdir(basename, RECOVERY_ROOT_MODE, dir_fd=parent_fd)
            created = True
            os.fsync(parent_fd)
        except FileExistsError:
            created = False

        root_fd, root_initial = _open_child_directory(parent_fd, basename)
        if root_initial.st_dev != parent_initial.st_dev:
            raise CompletionPublishError(
                "completion root must be on the same device as its parent"
            )

        # Advisory serialization avoids deterministic-temp churn between
        # cooperating retries.  Integrity still comes from mkdir/link/mode and
        # stable validation, never from the advisory lock itself.
        fcntl.flock(root_fd, fcntl.LOCK_EX)
        opened = os.fstat(root_fd)
        named = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
        if _object_identity(opened) != _object_identity(named):
            raise CompletionPublishError(
                "completion root identity changed after acquiring publication lock"
            )
        mode = stat.S_IMODE(opened.st_mode)
        if mode == COMMITTED_ROOT_MODE:
            report = _validate_completion_at(
                parent_fd, basename, expected_payloads=frozen_payloads
            )
            _require_no_legacy_staging(parent_fd, basename)
            _assert_parent_anchor(parent_fd, parent, parent_initial)
            return _publication_report(
                report, status="adopted_existing_exact", canonical_root=root
            )
        if mode != RECOVERY_ROOT_MODE:
            raise CompletionPublishError(
                f"uncommitted completion root mode is not 0700: {mode:04o}"
            )

        _inventory_recovery_root(root_fd, frozen_payloads)
        for name in sorted(COMPLETION_MEMBERS - {COMMIT_MEMBER}):
            _publish_or_verify_member(root_fd, name, frozen_payloads[name])
        _publish_or_verify_member(
            root_fd, COMMIT_MEMBER, frozen_payloads[COMMIT_MEMBER]
        )
        if set(os.listdir(root_fd)) != COMPLETION_MEMBERS:
            raise CompletionPublishError(
                "uncommitted completion member set is not exact before commit"
            )
        for name in sorted(COMPLETION_MEMBERS):
            _require_final_member(root_fd, name, frozen_payloads[name])
        root_before_commit = os.fstat(root_fd)
        named_before_commit = os.stat(
            basename, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            _object_identity(root_before_commit)
            != _object_identity(named_before_commit)
            or stat.S_IMODE(root_before_commit.st_mode) != RECOVERY_ROOT_MODE
        ):
            raise CompletionPublishError(
                "completion root identity/mode drifted before commit"
            )
        os.fsync(root_fd)
        commit_attempted = True
        os.fchmod(root_fd, COMMITTED_ROOT_MODE)
        os.fsync(root_fd)
        os.fsync(parent_fd)
        report = _validate_completion_at(
            parent_fd, basename, expected_payloads=frozen_payloads
        )
        _require_no_legacy_staging(parent_fd, basename)
        _assert_parent_anchor(parent_fd, parent, parent_initial)
        status = "published" if created else "recovered_uncommitted_exact"
        return _publication_report(report, status=status, canonical_root=root)
    except Exception as exc:
        # A chmod/fsync/validation call can report failure after the commit bit
        # became durable.  Recover only by proving the full committed state;
        # never mutate or remove a canonical root here.
        if commit_attempted:
            try:
                os.fsync(parent_fd)
                report = _validate_completion_at(
                    parent_fd, basename, expected_payloads=frozen_payloads
                )
                _require_no_legacy_staging(parent_fd, basename)
                _assert_parent_anchor(parent_fd, parent, parent_initial)
                return _publication_report(
                    report,
                    status="recovered_after_commit_exception",
                    canonical_root=root,
                )
            except Exception as recovery_exc:
                raise CompletionPublishError(
                    "publication raised after commit attempt; canonical was "
                    "preserved but exact committed recovery failed: "
                    f"{recovery_exc}"
                ) from exc
        if isinstance(exc, CompletionPublishError):
            raise
        raise CompletionPublishError(
            "completion publication failed before commit; canonical recovery "
            f"state was preserved: {exc}"
        ) from exc
    finally:
        if root_fd >= 0:
            os.close(root_fd)
        os.close(parent_fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    report = validate_completion_directory(args.root)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
