#!/usr/bin/env python3
"""Atomically stage the frozen anime caption parquet set onto node-local storage.

Successful stdout contains exactly one line: the content-addressed final root.
All diagnostics go to stderr so shell command substitution is safe.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any


GIB = 1 << 30
LOCAL_ROOT = Path("/local")
SOURCE_PARENT = Path("/s3/evan/anime_core_ditv04_ct_soft/data")
FILE_COUNT = 17
READY_SCHEMA = "offline_arb_anime_caption_local_ready_v1"
READY_NAME = "READY.json"
READY_KEYS = {
    "schema_version",
    "manifest_sha256",
    "files",
    "total_bytes",
    "staged_seconds",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def require_regular_file(path: Path, label: str) -> os.stat_result:
    metadata = lstat_or_none(path)
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} is missing, not regular, or a symlink: {path}")
    return metadata


def require_real_directory(path: Path, label: str) -> os.stat_result:
    metadata = lstat_or_none(path)
    if metadata is None or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{label} is missing, not a directory, or a symlink: {path}")
    return metadata


def prepare_base_root(requested: Path) -> Path:
    if not requested.is_absolute():
        raise ValueError(f"--base-root must be absolute: {requested}")
    require_real_directory(LOCAL_ROOT, "node-local root")
    try:
        requested.relative_to(LOCAL_ROOT)
    except ValueError as exc:
        raise ValueError(f"--base-root must be below {LOCAL_ROOT}: {requested}") from exc
    if requested == LOCAL_ROOT:
        raise ValueError(f"--base-root must be a child of {LOCAL_ROOT}, not {LOCAL_ROOT} itself")

    # Resolve before creating anything so an existing symlink in any parent cannot
    # redirect a nominal /local path onto shared storage.  Resolve again after mkdir
    # to close the normal non-adversarial creation race.
    if requested.resolve(strict=False) != requested:
        raise ValueError(f"--base-root must be a canonical real path without symlinks: {requested}")
    requested.mkdir(parents=True, exist_ok=True)
    require_real_directory(requested, "anime caption base root")
    if requested.resolve(strict=True) != requested:
        raise ValueError(f"--base-root resolved through a symlink: {requested}")
    return requested


def parse_entry(name: str, entry: Any) -> tuple[Path, int, str]:
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.endswith(".parquet")
        or not isinstance(entry, dict)
        or set(entry) != {"source_path", "bytes", "sha256"}
    ):
        raise ValueError(f"invalid manifest file entry: {name!r}")
    raw_source = entry.get("source_path")
    size = entry.get("bytes")
    digest = entry.get("sha256")
    if not isinstance(raw_source, str):
        raise ValueError(f"invalid source path declaration for {name}: {raw_source!r}")
    source = Path(raw_source)
    if source.parent != SOURCE_PARENT or source.name != name:
        raise ValueError(f"invalid source declaration for {name}: {source}")
    if type(size) is not int or size <= 0:
        raise ValueError(f"invalid source byte declaration for {name}: {size!r}")
    if not is_sha256(digest):
        raise ValueError(f"invalid SHA-256 declaration for {name}: {digest!r}")
    return source, size, digest


def load_manifest(path: Path) -> dict[str, Any]:
    require_regular_file(path, "anime caption manifest")
    header = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(header, dict) or header.get("schema_version") != "offline_arb_anime_caption_source_v1":
        raise ValueError(f"unsupported anime caption manifest: {path}")
    frozen = header.get("manifest_sha256")
    content = {
        key: value
        for key, value in header.items()
        if key not in {"manifest_sha256", "generated_at"}
    }
    if not is_sha256(frozen) or canonical_sha256(content) != frozen:
        raise ValueError("anime caption manifest self-hash mismatch")
    if header.get("source_parent") != str(SOURCE_PARENT):
        raise ValueError(
            f"anime caption manifest source_parent must be {SOURCE_PARENT}: "
            f"{header.get('source_parent')!r}"
        )
    for identity in (
        "catalog_hash",
        "animefreq_header_sha256",
        "catalog_caption_source_set_sha256",
        "catalog_anime_sha256",
    ):
        if not is_sha256(header.get(identity)):
            raise ValueError(f"anime caption manifest has invalid {identity}")
    files = header.get("files")
    if (
        not isinstance(files, dict)
        or len(files) != FILE_COUNT
        or type(header.get("file_count")) is not int
        or header.get("file_count") != FILE_COUNT
    ):
        raise ValueError(f"anime caption manifest must contain exactly {FILE_COUNT} files")
    parsed = {name: parse_entry(name, entry) for name, entry in files.items()}
    total_bytes = header.get("total_bytes")
    if type(total_bytes) is not int or total_bytes <= 0:
        raise ValueError("anime caption manifest total_bytes must be a positive integer")
    if sum(size for _source, size, _digest in parsed.values()) != total_bytes:
        raise ValueError("anime caption manifest byte accounting mismatch")
    source_set_sha = canonical_sha256(sorted(str(source) for source, _size, _sha in parsed.values()))
    if source_set_sha != header["catalog_caption_source_set_sha256"]:
        raise ValueError("anime caption manifest catalog source-set identity mismatch")
    return header


def validate_source_online(source: Path, expected_size: int) -> os.stat_result:
    metadata = require_regular_file(source, "anime caption source")
    if metadata.st_size != expected_size:
        raise RuntimeError(
            f"anime caption source size mismatch: {source}: "
            f"{metadata.st_size} != {expected_size}"
        )
    return metadata


def copy_and_hash(name: str, entry: Any, destination_root: Path) -> None:
    source, expected_size, expected_sha = parse_entry(name, entry)
    destination = destination_root / name
    source_before = validate_source_online(source, expected_size)
    digest = hashlib.sha256()
    written = 0
    with source.open("rb", buffering=0) as src, destination.open("xb", buffering=0) as dst:
        while True:
            block = src.read(8 << 20)
            if not block:
                break
            dst.write(block)
            digest.update(block)
            written += len(block)
        dst.flush()
        os.fsync(dst.fileno())
    source_after = require_regular_file(source, "anime caption source")
    if (source_before.st_size, source_before.st_mtime_ns) != (
        source_after.st_size,
        source_after.st_mtime_ns,
    ):
        raise RuntimeError(f"source changed while copying: {source}")
    if written != expected_size or digest.hexdigest() != expected_sha:
        raise RuntimeError(
            f"copied bytes/hash mismatch for {name}: {written}/{digest.hexdigest()}"
        )


def validate_directory_entries(root: Path, expected_names: set[str]) -> None:
    require_real_directory(root, "local anime caption root")
    live_names = {path.name for path in root.iterdir()}
    if live_names != expected_names:
        raise RuntimeError(
            "local anime caption directory set mismatch: "
            f"missing={sorted(expected_names-live_names)}, "
            f"added={sorted(live_names-expected_names)}"
        )
    for name in sorted(expected_names):
        require_regular_file(root / name, "local anime caption artifact")


def validate_local(
    root: Path,
    header: dict[str, Any],
    workers: int,
    *,
    extra_names: set[str] | None = None,
) -> None:
    files = header["files"]
    validate_directory_entries(root, set(files) | set(extra_names or ()))

    def check(name: str) -> None:
        _source, expected_size, expected_sha = parse_entry(name, files[name])
        path = root / name
        before = require_regular_file(path, "local anime caption parquet")
        if before.st_size != expected_size:
            raise RuntimeError(f"local size mismatch for {name}")
        live_sha = file_sha256(path)
        after = require_regular_file(path, "local anime caption parquet")
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"local anime caption changed while hashing: {path}")
        if live_sha != expected_sha:
            raise RuntimeError(f"local hash mismatch for {name}: {live_sha} != {expected_sha}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(check, sorted(files)))


def validate_ready(root: Path, header: dict[str, Any]) -> dict[str, Any]:
    validate_directory_entries(root, set(header["files"]) | {READY_NAME})
    ready_path = root / READY_NAME
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if not isinstance(ready, dict) or set(ready) != READY_KEYS:
        raise RuntimeError(f"published READY has an invalid schema: {ready_path}")
    if ready.get("schema_version") != READY_SCHEMA:
        raise RuntimeError(f"published READY schema version mismatch: {ready_path}")
    if ready.get("manifest_sha256") != header["manifest_sha256"]:
        raise RuntimeError(f"published READY manifest mismatch: {ready_path}")
    if type(ready.get("files")) is not int or ready.get("files") != FILE_COUNT:
        raise RuntimeError(f"published READY file count mismatch: {ready_path}")
    if (
        type(ready.get("total_bytes")) is not int
        or ready.get("total_bytes") != header["total_bytes"]
    ):
        raise RuntimeError(f"published READY byte count mismatch: {ready_path}")
    staged_seconds = ready.get("staged_seconds")
    if (
        isinstance(staged_seconds, bool)
        or not isinstance(staged_seconds, (int, float))
        or not math.isfinite(staged_seconds)
        or staged_seconds < 0
    ):
        raise RuntimeError(f"published READY staged_seconds is invalid: {ready_path}")
    return ready


def remove_path_no_follow(path: Path) -> bool:
    metadata = lstat_or_none(path)
    if metadata is None:
        return False
    if stat.S_ISDIR(metadata.st_mode):
        if os.path.ismount(path):
            raise RuntimeError(f"refusing to remove mounted local path: {path}")
        shutil.rmtree(path)
    else:
        path.unlink()
    if lstat_or_none(path) is not None:
        raise RuntimeError(f"failed to remove local path: {path}")
    return True


def discard_invalid_final(final_root: Path, base_root: Path, manifest_sha: str) -> None:
    if final_root.parent != base_root or final_root.name != manifest_sha:
        raise RuntimeError(f"refusing to discard path outside the expected content root: {final_root}")
    if lstat_or_none(final_root) is None:
        return
    quarantine = base_root / (
        f".corrupt.{manifest_sha}.{os.getpid()}.{time.time_ns()}"
    )
    if lstat_or_none(quarantine) is not None:
        raise RuntimeError(f"local quarantine path unexpectedly exists: {quarantine}")
    os.replace(final_root, quarantine)
    fsync_dir(base_root)
    remove_path_no_follow(quarantine)
    fsync_dir(base_root)


def clean_stale_local_paths(base_root: Path, manifest_sha: str) -> None:
    removed = False
    for prefix in (f".tmp.{manifest_sha}.", f".corrupt.{manifest_sha}."):
        for stale in sorted(base_root.glob(prefix + "*")):
            if stale.parent != base_root or not stale.name.startswith(prefix):
                raise RuntimeError(f"refusing unsafe stale cleanup candidate: {stale}")
            removed = remove_path_no_follow(stale) or removed
    if removed:
        fsync_dir(base_root)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path("/local/evan/offline_arb_anime_caption_v1"),
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    try:
        if args.workers not in (1, 2):
            raise ValueError("--workers must be 1 or 2")
        header = load_manifest(args.manifest)
        manifest_sha = header["manifest_sha256"]
        success = args.manifest.parent / "_SUCCESS"
        require_regular_file(success, "anime caption manifest _SUCCESS")
        if success.read_text(encoding="ascii").strip() != manifest_sha:
            raise ValueError("source manifest _SUCCESS is missing or mismatched")
        base_root = prepare_base_root(args.base_root)
        final_root = base_root / manifest_sha
        lock_path = base_root / f".{manifest_sha}.lock"
        lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        lock_fd = os.open(lock_path, lock_flags, 0o600)
        with os.fdopen(lock_fd, "r+b", buffering=0) as lock_handle:
            if not stat.S_ISREG(os.fstat(lock_handle.fileno()).st_mode):
                raise RuntimeError(f"manifest lock is not a regular file: {lock_path}")
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if lstat_or_none(final_root) is not None:
                try:
                    validate_ready(final_root, header)
                    validate_local(
                        final_root,
                        header,
                        args.workers,
                        extra_names={READY_NAME},
                    )
                except Exception as exc:
                    print(
                        f"discarding invalid published anime caption root {final_root}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    discard_invalid_final(final_root, base_root, manifest_sha)
                else:
                    print(str(final_root))
                    return 0

            clean_stale_local_paths(base_root, manifest_sha)

            total = int(header["total_bytes"])
            maximum = max(int(entry["bytes"]) for entry in header["files"].values())
            required = total + maximum + 32 * GIB
            free = shutil.disk_usage(base_root).free
            if free < required:
                raise RuntimeError(
                    f"insufficient /local for anime captions: free={free}, required={required}"
                )
            tmp_root = base_root / f".tmp.{manifest_sha}.{os.getpid()}.{time.time_ns()}"
            tmp_root.mkdir(mode=0o700)
            try:
                print(
                    f"staging 17 anime caption parquets: bytes={total} workers={args.workers} "
                    f"free={free} tmp={tmp_root}",
                    file=sys.stderr,
                    flush=True,
                )
                started = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = [
                        pool.submit(copy_and_hash, name, entry, tmp_root)
                        for name, entry in sorted(header["files"].items())
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        future.result()
                validate_local(tmp_root, header, args.workers)
                ready = {
                    "schema_version": READY_SCHEMA,
                    "manifest_sha256": manifest_sha,
                    "files": FILE_COUNT,
                    "total_bytes": total,
                    "staged_seconds": round(time.monotonic() - started, 3),
                }
                ready_path_tmp = tmp_root / READY_NAME
                ready_path_tmp.write_text(
                    json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                with ready_path_tmp.open("rb") as handle:
                    os.fsync(handle.fileno())
                validate_ready(tmp_root, header)
                fsync_dir(tmp_root)
                os.replace(tmp_root, final_root)
                fsync_dir(base_root)
                print(
                    f"anime caption staging complete in {ready['staged_seconds']}s: {final_root}",
                    file=sys.stderr,
                    flush=True,
                )
            except BaseException:
                try:
                    if remove_path_no_follow(tmp_root):
                        fsync_dir(base_root)
                except Exception as cleanup_exc:
                    print(
                        f"FATAL failed to clean anime caption temp root {tmp_root}: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                raise
        print(str(final_root))
        return 0
    except Exception as exc:
        print(f"FATAL anime caption staging: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
