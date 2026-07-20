#!/usr/bin/env python3
"""Publish a read-only FSx compatibility view for the verified GPIC release.

This program never writes below the canonical S3 root.  It copies the already
verified dense index on FSx, derives the legacy runtime controls with the pure
projection helper, and atomically renames one complete six-file directory.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
import uuid
from pathlib import Path
from typing import Any

from training.dataline_offline import gpic_release_view as view


VIEW_FILES = {
    "canonical_manifest.json",
    "verifier_summary.json",
    "dense_index.bin",
    "manifest.json",
    "verifier_report.json",
    "_SUCCESS",
}
VERIFIER_ROOT_FILES = {
    "canonical_manifest.json",
    "verifier_summary.json",
    "workers",
    "_SUCCESS",
}


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _regular(path: Path, label: str, *, nonempty: bool = True) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise view.ReleaseViewError(f"cannot stat {label}: {path}: {exc}") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise view.ReleaseViewError(f"{label} must be a regular non-symlink file: {path}")
    if nonempty and metadata.st_size <= 0:
        raise view.ReleaseViewError(f"{label} must be nonempty: {path}")
    return metadata


def _directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise view.ReleaseViewError(f"{label} must be an absolute non-symlink directory: {path}")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise view.ReleaseViewError(f"cannot resolve {label}: {path}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or not resolved.is_dir() or str(resolved) != str(path):
        raise view.ReleaseViewError(f"{label} must be a normalized real directory: {path}")
    return resolved


def _exact_names(path: Path, expected: set[str], label: str) -> None:
    _directory(path, label)
    try:
        actual = {item.name for item in path.iterdir()}
    except OSError as exc:
        raise view.ReleaseViewError(f"cannot enumerate {label}: {path}: {exc}") from exc
    if actual != expected:
        raise view.ReleaseViewError(
            f"{label} exact-set drift: missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )


def _bound_bytes(path: Path, label: str) -> bytes:
    before = _regular(path, label)
    identity = _identity(before)
    try:
        with path.open("rb", buffering=0) as handle:
            if _identity(os.fstat(handle.fileno())) != identity:
                raise view.ReleaseViewError(f"{label} changed while opening: {path}")
            raw = handle.read()
            closed = os.fstat(handle.fileno())
    except view.ReleaseViewError:
        raise
    except OSError as exc:
        raise view.ReleaseViewError(f"cannot read {label}: {path}: {exc}") from exc
    if _identity(closed) != identity or _identity(_regular(path, label)) != identity:
        raise view.ReleaseViewError(f"{label} changed while reading: {path}")
    return raw


def _write_once(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise view.ReleaseViewError(f"refusing compatibility-view overwrite: {path}")
    try:
        with path.open("xb", buffering=0) as handle:
            cursor = 0
            while cursor < len(raw):
                written = handle.write(raw[cursor:])
                if written is None or written <= 0:
                    raise view.ReleaseViewError(f"short compatibility-view write: {path}")
                cursor += written
            handle.flush()
            os.fsync(handle.fileno())
    except view.ReleaseViewError:
        raise
    except OSError as exc:
        raise view.ReleaseViewError(f"cannot write compatibility-view file {path}: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_and_hash(source: Path, destination: Path, block_bytes: int = 16 << 20) -> tuple[int, str]:
    before = _regular(source, "canonical dense index")
    source_identity = _identity(before)
    if destination.exists() or destination.is_symlink():
        raise view.ReleaseViewError(f"refusing dense-index destination overwrite: {destination}")
    digest = hashlib.sha256()
    total = 0
    try:
        with source.open("rb", buffering=0) as reader, destination.open("xb", buffering=0) as writer:
            if _identity(os.fstat(reader.fileno())) != source_identity:
                raise view.ReleaseViewError("canonical dense index changed while opening")
            while True:
                block = reader.read(block_bytes)
                if not block:
                    break
                digest.update(block)
                cursor = 0
                while cursor < len(block):
                    written = writer.write(block[cursor:])
                    if written is None or written <= 0:
                        raise view.ReleaseViewError("short dense-index copy write")
                    cursor += written
                total += len(block)
            reader_closed = os.fstat(reader.fileno())
            writer.flush()
            os.fsync(writer.fileno())
    except view.ReleaseViewError:
        raise
    except OSError as exc:
        raise view.ReleaseViewError(f"dense-index copy failed: {exc}") from exc
    if (
        _identity(reader_closed) != source_identity
        or _identity(_regular(source, "canonical dense index")) != source_identity
        or _regular(destination, "copied dense index").st_size != total
    ):
        raise view.ReleaseViewError("dense index changed or was truncated during copy")
    return total, digest.hexdigest()


def _sha256_file(path: Path, block_bytes: int = 16 << 20) -> tuple[int, str]:
    before = _regular(path, "compatibility-view dense index")
    identity = _identity(before)
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb", buffering=0) as handle:
            if _identity(os.fstat(handle.fileno())) != identity:
                raise view.ReleaseViewError("compatibility-view index changed while opening")
            while True:
                block = handle.read(block_bytes)
                if not block:
                    break
                digest.update(block)
                total += len(block)
            closed = os.fstat(handle.fileno())
    except view.ReleaseViewError:
        raise
    except OSError as exc:
        raise view.ReleaseViewError(f"cannot hash compatibility-view index: {exc}") from exc
    if _identity(closed) != identity or _identity(_regular(path, "compatibility-view dense index")) != identity:
        raise view.ReleaseViewError("compatibility-view index changed while hashing")
    return total, digest.hexdigest()


def _validate_verifier_root(root: Path, canonical_self: str) -> None:
    _exact_names(root, VERIFIER_ROOT_FILES, "verified FSx root")
    workers = root / "workers"
    _exact_names(workers, {f"worker-{rank}" for rank in range(8)}, "verified worker root")
    for rank in range(8):
        worker = workers / f"worker-{rank}"
        _exact_names(worker, {"report.json", "_SUCCESS"}, f"verified worker {rank}")
        _regular(worker / "report.json", f"verified worker {rank} report")
        _regular(worker / "_SUCCESS", f"verified worker {rank} marker")
    if _bound_bytes(root / "_SUCCESS", "verified FSx _SUCCESS") != (canonical_self + "\n").encode("ascii"):
        raise view.ReleaseViewError("verified FSx _SUCCESS does not bind canonical self-hash")


def _validate_s3_plane(canonical: dict[str, Any], canonical_raw: bytes) -> None:
    storage = canonical["release"]["storage"]
    s3_root = Path(storage["s3_root"])
    _exact_names(s3_root, {"_INIT.json", "_SUCCESS", "controls", "workers"}, "canonical S3 root")
    controls = s3_root / "controls"
    _exact_names(controls, {"manifest.json"}, "canonical S3 controls")
    workers = s3_root / "workers"
    _exact_names(workers, {f"worker-{rank}" for rank in range(8)}, "canonical S3 workers")
    init_raw = _bound_bytes(s3_root / "_INIT.json", "canonical S3 _INIT")
    init = storage["s3_init"]
    if len(init_raw) != init["bytes"] or hashlib.sha256(init_raw).hexdigest() != init["raw_sha256"]:
        raise view.ReleaseViewError("canonical S3 _INIT raw binding mismatch")
    if _bound_bytes(controls / "manifest.json", "canonical S3 manifest") != canonical_raw:
        raise view.ReleaseViewError("canonical FSx/S3 manifest raw bytes differ")
    if _bound_bytes(s3_root / "_SUCCESS", "canonical S3 _SUCCESS") != (
        canonical["manifest_sha256"] + "\n"
    ).encode("ascii"):
        raise view.ReleaseViewError("canonical S3 _SUCCESS does not bind canonical self-hash")
    expected_by_worker: list[dict[str, int]] = [dict() for _ in range(8)]
    for blob in canonical["blobs"]:
        expected_by_worker[blob["worker_id"]][blob["name"]] = blob["bytes"]
    for worker_id, expected in enumerate(expected_by_worker):
        worker_root = workers / f"worker-{worker_id}"
        _exact_names(worker_root, set(expected), f"canonical S3 worker {worker_id}")
        for name, expected_bytes in expected.items():
            metadata = _regular(worker_root / name, f"canonical S3 blob {name}")
            if metadata.st_size != expected_bytes:
                raise view.ReleaseViewError(f"canonical S3 blob size drift: {name}")


def _expected_documents(
    canonical: dict[str, Any], summary: dict[str, Any], summary_raw: bytes, final: Path
) -> tuple[bytes, bytes]:
    manifest, report = view.project_runtime_manifest(
        canonical, summary, final, summary_raw_bytes=summary_raw
    )
    return view.canonical_json_bytes(manifest), view.canonical_json_bytes(report)


def _verify_existing(
    final: Path,
    canonical_raw: bytes,
    summary_raw: bytes,
    manifest_raw: bytes,
    report_raw: bytes,
    expected_index_bytes: int,
    expected_index_sha256: str,
) -> str:
    _exact_names(final, VIEW_FILES, "runtime compatibility view")
    expected_small = {
        "canonical_manifest.json": canonical_raw,
        "verifier_summary.json": summary_raw,
        "manifest.json": manifest_raw,
        "verifier_report.json": report_raw,
    }
    for name, expected in expected_small.items():
        if _bound_bytes(final / name, f"runtime compatibility {name}") != expected:
            raise view.ReleaseViewError(f"existing runtime compatibility {name} differs")
    adapter_raw_sha = hashlib.sha256(manifest_raw).hexdigest()
    if _bound_bytes(final / "_SUCCESS", "runtime compatibility _SUCCESS") != (
        adapter_raw_sha + "\n"
    ).encode("ascii"):
        raise view.ReleaseViewError("runtime compatibility _SUCCESS differs")
    index_bytes, index_sha = _sha256_file(final / "dense_index.bin")
    if index_bytes != expected_index_bytes or index_sha != expected_index_sha256:
        raise view.ReleaseViewError("runtime compatibility dense-index hash/size differs")
    return adapter_raw_sha


def _check_expected(args: argparse.Namespace, canonical: dict[str, Any], summary: dict[str, Any], canonical_raw: bytes, summary_raw: bytes) -> None:
    checks: list[tuple[Any, Any, str]] = [
        (canonical["build_jobtag"], args.buildtag, "build jobtag"),
        (canonical["verifier_jobtag"], args.verifier_jobtag, "verifier jobtag"),
        (canonical["manifest_sha256"], args.canonical_self_sha256, "canonical self SHA-256"),
        (hashlib.sha256(canonical_raw).hexdigest(), args.canonical_raw_sha256, "canonical raw SHA-256"),
        (summary["verifier_summary_sha256"], args.summary_self_sha256, "summary self SHA-256"),
        (hashlib.sha256(summary_raw).hexdigest(), args.summary_raw_sha256, "summary raw SHA-256"),
        (canonical["dense_index"]["sha256"], args.index_sha256, "dense-index SHA-256"),
        (canonical["dense_index"]["bytes"], args.index_bytes, "dense-index bytes"),
        (canonical["rows"], args.rows, "rows"),
        (canonical["blob_count"], args.blob_count, "blob count"),
        (canonical["payload_bytes"], args.payload_bytes, "payload bytes"),
        (canonical["release"]["storage"]["s3_root"], str(args.s3_root), "S3 root"),
    ]
    for actual, expected, label in checks:
        if actual != expected:
            raise view.ReleaseViewError(f"frozen {label} mismatch: expected={expected!r}, actual={actual!r}")


def publish(args: argparse.Namespace, *, require_fsx: bool = True) -> tuple[Path, str, bool]:
    verified_root = _directory(args.verified_root, "verified FSx release root")
    if verified_root.name != args.verifier_jobtag or verified_root.parent.name != args.buildtag:
        raise view.ReleaseViewError("verified FSx root does not bind build/verifier jobtags")
    canonical_raw = _bound_bytes(verified_root / "canonical_manifest.json", "verified canonical manifest")
    summary_raw = _bound_bytes(verified_root / "verifier_summary.json", "verified verifier summary")
    canonical = view.parse_production_json(canonical_raw, "verified canonical manifest")
    summary = view.parse_production_json(summary_raw, "verified verifier summary")
    view.validate_canonical_manifest(canonical, canonical_raw)
    view.validate_verifier_summary(canonical, summary, summary_raw)
    _check_expected(args, canonical, summary, canonical_raw, summary_raw)
    _validate_verifier_root(verified_root, canonical["manifest_sha256"])
    _validate_s3_plane(canonical, canonical_raw)

    base = args.view_base
    if not base.is_absolute() or str(base) != str(Path(str(base))):
        raise view.ReleaseViewError("compatibility-view base must be a normalized absolute path")
    if require_fsx and not str(base).startswith("/fsx/"):
        raise view.ReleaseViewError("production compatibility-view base must stay below /fsx")
    if base.exists() or base.is_symlink():
        _directory(base, "compatibility-view base")
    else:
        parent = _directory(base.parent, "compatibility-view base parent")
        if base.parent != parent:
            raise view.ReleaseViewError("compatibility-view base parent path drift")
        base.mkdir(mode=0o700)
        _fsync_directory(parent)
    final = base / args.buildtag
    manifest_raw, report_raw = _expected_documents(canonical, summary, summary_raw, final)
    if final.exists() or final.is_symlink():
        if final.is_symlink():
            raise view.ReleaseViewError("existing compatibility view is a symlink")
        adapter_sha = _verify_existing(
            final, canonical_raw, summary_raw, manifest_raw, report_raw,
            args.index_bytes, args.index_sha256,
        )
        _validate_s3_plane(canonical, canonical_raw)
        return final, adapter_sha, True

    projected_bytes = (
        args.index_bytes + len(canonical_raw) + len(summary_raw)
        + len(manifest_raw) + len(report_raw) + 65
    )
    if projected_bytes > args.max_view_bytes:
        raise view.ReleaseViewError("compatibility view exceeds its frozen FSx byte budget")
    temp = base / f".{args.buildtag}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    temp.mkdir(mode=0o700)
    try:
        index_bytes, index_sha = _copy_and_hash(
            Path(canonical["release"]["storage"]["index_path"]), temp / "dense_index.bin"
        )
        if index_bytes != args.index_bytes or index_sha != args.index_sha256:
            raise view.ReleaseViewError("copied dense-index hash/size differs from frozen binding")
        _write_once(temp / "canonical_manifest.json", canonical_raw)
        _write_once(temp / "verifier_summary.json", summary_raw)
        _write_once(temp / "verifier_report.json", report_raw)
        _write_once(temp / "manifest.json", manifest_raw)
        adapter_sha = hashlib.sha256(manifest_raw).hexdigest()
        # This is deliberately the final write in the private publication tree.
        _write_once(temp / "_SUCCESS", (adapter_sha + "\n").encode("ascii"))
        _fsync_directory(temp)
        _exact_names(temp, VIEW_FILES, "private runtime compatibility view")
        try:
            os.rename(temp, final)
        except OSError as exc:
            if not final.exists() or final.is_symlink():
                raise view.ReleaseViewError(f"atomic compatibility-view rename failed: {exc}") from exc
            shutil.rmtree(temp)
            adapter_sha = _verify_existing(
                final, canonical_raw, summary_raw, manifest_raw, report_raw,
                args.index_bytes, args.index_sha256,
            )
            _validate_s3_plane(canonical, canonical_raw)
            return final, adapter_sha, True
        _fsync_directory(base)
        # The copy stream already hashed every byte; after the same-filesystem
        # directory rename, stat the published file without a redundant 1.12GB pass.
        if _regular(final / "dense_index.bin", "published dense index").st_size != args.index_bytes:
            raise view.ReleaseViewError("published dense-index size drift after rename")
        _validate_s3_plane(canonical, canonical_raw)
        return final, adapter_sha, False
    except Exception:
        if temp.exists() and not temp.is_symlink() and temp.parent == base and temp.name.startswith(f".{args.buildtag}.tmp."):
            shutil.rmtree(temp)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-root", type=Path, required=True)
    parser.add_argument("--view-base", type=Path, required=True)
    parser.add_argument("--s3-root", type=Path, required=True)
    parser.add_argument("--buildtag", required=True)
    parser.add_argument("--verifier-jobtag", required=True)
    parser.add_argument("--canonical-self-sha256", required=True)
    parser.add_argument("--canonical-raw-sha256", required=True)
    parser.add_argument("--summary-self-sha256", required=True)
    parser.add_argument("--summary-raw-sha256", required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--index-bytes", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--blob-count", type=int, required=True)
    parser.add_argument("--payload-bytes", type=int, required=True)
    parser.add_argument("--max-view-bytes", type=int, default=2_000_000_000)
    args = parser.parse_args(argv)
    for name in (
        "canonical_self_sha256", "canonical_raw_sha256", "summary_self_sha256",
        "summary_raw_sha256", "index_sha256",
    ):
        value = getattr(args, name)
        if view.SHA256_RE.fullmatch(value) is None:
            parser.error(f"--{name.replace('_', '-')} must be a lowercase SHA-256")
    for name in ("index_bytes", "rows", "blob_count", "payload_bytes", "max_view_bytes"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        final, adapter_sha, idempotent = publish(args)
    except view.ReleaseViewError as exc:
        print(f"OFFLINE_GPIC_RUNTIME_VIEW_FAIL {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        "OFFLINE_GPIC_RUNTIME_VIEW_PASS "
        f"root={final} adapter_raw_sha256={adapter_sha} "
        f"canonical_self_sha256={args.canonical_self_sha256} idempotent={idempotent}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
