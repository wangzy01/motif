"""Fail-closed runtime reader for the immutable ``offline_gpic_blob_v1`` release.

The production release keeps its small control plane and fixed-width index on
FSx, while the image blobs live below one mountpoint-S3 directory.  A catalog
global GPIC ``row_id`` addresses exactly one 56-byte little-endian index record::

    <qIQI32s  ==  row_id:i64, blob_id:u32, offset:u64,
                    length:u32, segment_sha256:32B

This module deliberately has no source-file fallback.  Missing controls, an
unverified index, an invalid locator, a short pread, a changed open file, or a
segment digest mismatch are all fatal.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import resource
import stat
import struct
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from training.dataline_offline.gpic_release_view import (
    ReleaseViewError,
    parse_production_json,
    project_release,
    validate_canonical_manifest,
    validate_verifier_summary,
)


SCHEMA_VERSION = "offline_gpic_blob_v1"
VERIFY_SCHEMA_VERSION = "offline_gpic_blob_verify_v1"
LAYOUT_VERSION = "offline_gpic_blob_index_le_v1"
BACKEND = "mountpoint_s3"
SELECTION_MODE = "full_catalog"
INDEX_STRUCT = struct.Struct("<qIQI32s")
INDEX_RECORD_BYTES = INDEX_STRUCT.size
MAX_SEGMENT_BYTES = 256 << 20
SHA256_RE = re.compile(r"[0-9a-f]{64}")

_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "generated_at",
    "release",
    "release_sha256",
    "verification",
    "manifest_sha256",
}
_RELEASE_KEYS = {
    "backend",
    "catalog",
    "layout",
    "storage",
    "index",
    "blobs",
    "row_count",
    "payload_bytes",
    "blob_set_sha256",
}
_CATALOG_KEYS = {"catalog_hash", "catalog_header_sha256", "gpic_file"}
_GPIC_FILE_KEYS = {"name", "row_base", "rows", "bytes", "sha256", "schema"}
_LAYOUT_KEYS = {
    "version",
    "record_struct",
    "record_bytes",
    "endianness",
    "index_order",
    "row_id_namespace",
    "selection_mode",
    "compression",
    "max_segment_bytes",
}
_STORAGE_KEYS = {"s3_root", "index_path", "s3_init"}
_S3_INIT_KEYS = {"path", "bytes", "raw_sha256"}
_FILE_KEYS = {"bytes", "sha256"}
_BLOB_KEYS = {"blob_id", "name", "bytes", "sha256"}
_VERIFICATION_BINDING_KEYS = {"path", "bytes", "sha256", "schema_version"}
_VERIFY_REPORT_KEYS = {
    "schema_version",
    "status",
    "generated_at",
    "release_sha256",
    "catalog_hash",
    "index_sha256",
    "index_bytes",
    "row_id_start",
    "row_id_stop_exclusive",
    "row_count",
    "blob_count",
    "payload_bytes",
    "full_index_records_verified",
    "full_segment_sha256_verified",
    "full_blob_sha256_verified",
    "report_sha256",
}


class GpicBlobError(RuntimeError):
    """The immutable GPIC blob release cannot serve the requested exact row."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def manifest_self_hash(value: Mapping[str, Any]) -> str:
    core = dict(value)
    core.pop("manifest_sha256", None)
    return canonical_json_sha256(core)


def report_self_hash(value: Mapping[str, Any]) -> str:
    core = dict(value)
    core.pop("report_sha256", None)
    return canonical_json_sha256(core)


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    before = _regular_metadata(path, "SHA input")
    identity = _identity(before)
    try:
        with path.open("rb", buffering=0) as handle:
            if _identity(os.fstat(handle.fileno())) != identity:
                raise GpicBlobError(f"file changed while opening for SHA-256: {path}")
            for block in iter(lambda: handle.read(block_bytes), b""):
                digest.update(block)
            closed = os.fstat(handle.fileno())
    except GpicBlobError:
        raise
    except OSError as exc:
        raise GpicBlobError(f"cannot hash {path}: {exc}") from exc
    after = _regular_metadata(path, "SHA input")
    if _identity(closed) != identity or _identity(after) != identity:
        raise GpicBlobError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _sha256_open_fd(
    descriptor: int,
    expected_identity: tuple[int, int, int, int, int],
    *,
    label: str,
    block_bytes: int = 8 << 20,
) -> str:
    """Hash the already-open inode that backs an mmap, without path reopen."""

    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if _identity(before) != expected_identity:
            raise GpicBlobError(f"{label} FD changed before SHA-256")
        size = int(before.st_size)
        offset = 0
        while offset < size:
            wanted = min(block_bytes, size - offset)
            try:
                block = os.pread(descriptor, wanted, offset)
            except InterruptedError:
                continue
            if not block:
                raise GpicBlobError(
                    f"short {label} FD read while hashing: wanted={size}, got={offset}"
                )
            digest.update(block)
            offset += len(block)
        after = os.fstat(descriptor)
    except GpicBlobError:
        raise
    except OSError as exc:
        raise GpicBlobError(f"cannot hash open {label} FD: {exc}") from exc
    if _identity(after) != expected_identity:
        raise GpicBlobError(f"{label} FD changed while hashing")
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _strict_int(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise GpicBlobError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        live = set(value) if isinstance(value, Mapping) else set()
        raise GpicBlobError(
            f"{label} key set drift: missing={sorted(expected-live)}, "
            f"unexpected={sorted(live-expected)}"
        )
    return value


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _regular_metadata(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise GpicBlobError(f"cannot stat {label} {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise GpicBlobError(f"{label} must be a regular non-symlink file: {path}")
    return metadata


def _real_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise GpicBlobError(f"{label} must be an absolute non-symlink directory: {path}")
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as exc:
        raise GpicBlobError(f"cannot resolve {label} {path}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or not resolved.is_dir():
        raise GpicBlobError(f"{label} is not a real directory: {path}")
    return resolved


def _read_bound_bytes(path: Path, label: str) -> tuple[bytes, str]:
    before = _regular_metadata(path, label)
    identity = _identity(before)
    try:
        with path.open("rb", buffering=0) as handle:
            if _identity(os.fstat(handle.fileno())) != identity:
                raise GpicBlobError(f"{label} changed while opening: {path}")
            payload = handle.read()
            closed = os.fstat(handle.fileno())
    except GpicBlobError:
        raise
    except OSError as exc:
        raise GpicBlobError(f"cannot read {label} {path}: {exc}") from exc
    after = _regular_metadata(path, label)
    if _identity(closed) != identity or _identity(after) != identity:
        raise GpicBlobError(f"{label} changed while reading: {path}")
    return payload, hashlib.sha256(payload).hexdigest()


def _parse_canonical_json(raw: bytes, path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GpicBlobError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GpicBlobError(f"{label} must contain one JSON object: {path}")
    if canonical_json_bytes(value) != raw:
        raise GpicBlobError(f"{label} is not canonical JSON: {path}")
    return value


def _blob_relative_parts(raw: Any, label: str) -> tuple[str, str, str]:
    """Validate the frozen ``workers/worker-N/blob-*.bin`` key grammar.

    The allow-list deliberately excludes backslashes, percent escapes, Unicode,
    empty/dot components, and every spelling that ``Path`` would normalize.  It
    therefore remains safe to feed the returned components to ``openat`` without
    relying on string-prefix containment checks.
    """

    if not isinstance(raw, str) or not raw or "\\" in raw or "%" in raw:
        raise GpicBlobError(f"{label} is not a canonical blob key: {raw!r}")
    if raw.startswith("/") or raw.endswith("/") or "//" in raw:
        raise GpicBlobError(f"{label} is not a canonical relative blob key: {raw!r}")
    parts = tuple(raw.split("/"))
    if (
        len(parts) != 3
        or any(part in {"", ".", ".."} for part in parts)
        or parts[0] != "workers"
        or re.fullmatch(r"worker-[0-7]", parts[1]) is None
        or re.fullmatch(r"blob-[0-9]+\.bin", parts[2]) is None
        or "/".join(parts) != raw
    ):
        raise GpicBlobError(
            f"{label} must match workers/worker-[0-7]/blob-[0-9]+.bin, got {raw!r}"
        )
    return parts  # type: ignore[return-value]


def _process_rss_bytes() -> int:
    try:
        statm = Path("/proc/self/statm").read_text(encoding="ascii").split()
        if len(statm) < 2:
            return -1
        return int(statm[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return -1


def _process_fd_count() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


@dataclass(frozen=True)
class CatalogBinding:
    catalog_hash: str
    catalog_header_sha256: str
    gpic_name: str
    gpic_row_base: int
    gpic_rows: int
    gpic_bytes: int
    gpic_sha256: str
    gpic_schema: str


@dataclass(frozen=True)
class BlobBinding:
    blob_id: int
    name: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class IndexEntry:
    row_id: int
    blob_id: int
    offset: int
    length: int
    sha256_raw: bytes


@dataclass
class _FDEntry:
    blob: BlobBinding
    path: Path
    descriptor: int
    identity: tuple[int, int, int, int, int]
    leases: int = 0


class _FDLease:
    def __init__(self, cache: "_BlobFDCache", entry: _FDEntry):
        self._cache = cache
        self.entry = entry
        self._released = False

    def __enter__(self) -> _FDEntry:
        return self.entry

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._cache.release(self.entry)


class _BlobFDCache:
    """Process-local LRU whose leases prevent close/pread eviction races."""

    def __init__(
        self,
        root: Path,
        blobs: Mapping[int, BlobBinding],
        max_open: int,
        acquire_timeout_s: float = 60.0,
    ):
        self.root = root
        self.blobs = dict(blobs)
        self.max_open = _strict_int(max_open, "gpic_blob_fd_cache_size", 1)
        if not isinstance(acquire_timeout_s, (int, float)) or acquire_timeout_s <= 0:
            raise GpicBlobError("FD acquire timeout must be positive")
        self.acquire_timeout_s = float(acquire_timeout_s)
        self._condition = threading.Condition()
        self._entries: "OrderedDict[int, _FDEntry]" = OrderedDict()
        self._closed = False
        self._metrics: Counter[str] = Counter()
        root_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            root_before = self.root.lstat()
        except OSError as exc:
            raise GpicBlobError(
                f"cannot stat GPIC S3 blob root {self.root}: {exc}"
            ) from exc
        if not stat.S_ISDIR(root_before.st_mode):
            raise GpicBlobError(f"GPIC S3 blob root is not a directory: {self.root}")
        try:
            self._root_descriptor = os.open(self.root, root_flags)
        except OSError as exc:
            raise GpicBlobError(f"cannot open GPIC S3 blob root {self.root}: {exc}") from exc
        root_opened = os.fstat(self._root_descriptor)
        if not stat.S_ISDIR(root_opened.st_mode) or _identity(root_opened) != _identity(
            root_before
        ):
            os.close(self._root_descriptor)
            self._root_descriptor = -1
            raise GpicBlobError(
                f"GPIC S3 blob root changed while opening: {self.root}"
            )

    def _open_locked(self, blob: BlobBinding) -> _FDEntry:
        parts = _blob_relative_parts(blob.name, f"blob {blob.blob_id} name")
        path = self.root.joinpath(*parts)
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_flags = (
            file_flags
            | getattr(os, "O_DIRECTORY", 0)
        )
        parent_descriptor = self._root_descriptor
        opened_directories: list[int] = []
        descriptor: int | None = None
        try:
            # Traverse every directory component using no-follow openat.  The
            # lstat/fstat identity pair rejects both pre-existing symlinks and a
            # directory replacement between validation and open.
            for component in parts[:-1]:
                before = os.stat(
                    component, dir_fd=parent_descriptor, follow_symlinks=False
                )
                if not stat.S_ISDIR(before.st_mode):
                    raise GpicBlobError(
                        f"blob parent component is not a real directory: {path}"
                    )
                child_descriptor = os.open(
                    component, directory_flags, dir_fd=parent_descriptor
                )
                opened_directories.append(child_descriptor)
                opened = os.fstat(child_descriptor)
                if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != _identity(
                    before
                ):
                    raise GpicBlobError(
                        f"blob parent component changed while opening: {path}"
                    )
                parent_descriptor = child_descriptor

            leaf_before = os.stat(
                parts[-1], dir_fd=parent_descriptor, follow_symlinks=False
            )
            if not stat.S_ISREG(leaf_before.st_mode):
                raise GpicBlobError(f"S3 blob is not a regular file: {path}")
            descriptor = os.open(parts[-1], file_flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise GpicBlobError(
                f"cannot open S3 blob_id={blob.blob_id}, path={path}: {exc}"
            ) from exc
        finally:
            for directory_descriptor in reversed(opened_directories):
                os.close(directory_descriptor)
        try:
            assert descriptor is not None
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or _identity(metadata) != _identity(leaf_before)
                or int(metadata.st_size) != blob.bytes
            ):
                raise GpicBlobError(
                    f"S3 blob identity/size mismatch blob_id={blob.blob_id}: "
                    f"expected_bytes={blob.bytes}, actual_bytes={metadata.st_size}"
                )
            entry = _FDEntry(blob, path, descriptor, _identity(metadata), leases=0)
        except BaseException:
            os.close(descriptor)
            raise
        self._metrics["fd_opens"] += 1
        self._metrics["fd_current"] += 1
        self._metrics["fd_peak"] = max(
            self._metrics["fd_peak"], self._metrics["fd_current"]
        )
        return entry

    def acquire(self, blob_id: int) -> _FDLease:
        blob = self.blobs.get(blob_id)
        if blob is None:
            raise GpicBlobError(f"index references undeclared blob_id={blob_id}")
        deadline = time.monotonic() + self.acquire_timeout_s
        with self._condition:
            while True:
                if self._closed:
                    raise GpicBlobError("GPIC blob FD cache is closed")
                entry = self._entries.get(blob_id)
                if entry is not None:
                    entry.leases += 1
                    self._entries.move_to_end(blob_id)
                    self._metrics["fd_hits"] += 1
                    return _FDLease(self, entry)

                if len(self._entries) >= self.max_open:
                    evict_id = next(
                        (
                            current_id
                            for current_id, current in self._entries.items()
                            if current.leases == 0
                        ),
                        None,
                    )
                    if evict_id is not None:
                        evicted = self._entries.pop(evict_id)
                        os.close(evicted.descriptor)
                        self._metrics["fd_evictions"] += 1
                        self._metrics["fd_current"] -= 1
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            self._metrics["fd_acquire_timeouts"] += 1
                            raise GpicBlobError(
                                "timed out waiting for an idle GPIC blob FD cache entry"
                            )
                        self._condition.wait(timeout=min(1.0, remaining))
                        continue

                entry = self._open_locked(blob)
                entry.leases = 1
                self._entries[blob_id] = entry
                self._metrics["fd_misses"] += 1
                return _FDLease(self, entry)

    def release(self, entry: _FDEntry) -> None:
        with self._condition:
            current = self._entries.get(entry.blob.blob_id)
            if current is not entry or entry.leases <= 0:
                raise GpicBlobError("GPIC blob FD lease accounting drift")
            entry.leases -= 1
            self._condition.notify_all()

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            result = {key: int(value) for key, value in self._metrics.items()}
            result["fd_current"] = len(self._entries)
            result["fd_active_leases"] = sum(entry.leases for entry in self._entries.values())
            result["fd_cache_size"] = self.max_open
            return result

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            active = {
                blob_id: entry.leases
                for blob_id, entry in self._entries.items()
                if entry.leases
            }
            if active:
                raise GpicBlobError(
                    f"cannot close GPIC blob FD cache with active leases: {active}"
                )
            self._closed = True
            for entry in self._entries.values():
                os.close(entry.descriptor)
            if self._root_descriptor >= 0:
                os.close(self._root_descriptor)
                self._root_descriptor = -1
            self._metrics["fd_closes"] += len(self._entries)
            self._metrics["fd_current"] = 0
            self._entries.clear()
            self._condition.notify_all()


class GpicBlobOverlay:
    """Validated, full-catalog, random-pread GPIC transport."""

    def __init__(
        self,
        control_root: str | os.PathLike[str],
        *,
        expected_manifest_sha256: str,
        catalog: CatalogBinding,
        fd_cache_size: int = 64,
    ):
        if not _is_sha256(expected_manifest_sha256):
            raise GpicBlobError("expected GPIC blob manifest SHA-256 is malformed")
        self.fd_cache_size = _strict_int(
            fd_cache_size, "gpic_blob_fd_cache_size", 1
        )
        self.control_root = _real_directory(Path(control_root).expanduser(), "GPIC control root")
        self.manifest_path = self.control_root / "manifest.json"
        self.success_path = self.control_root / "_SUCCESS"
        self._closed = False
        self._index_verified = False
        self._metrics_lock = threading.Lock()
        self._metrics: Counter[str] = Counter()
        self._index_fd: int | None = None
        self._index_mmap: mmap.mmap | None = None
        self._fd_cache: _BlobFDCache | None = None

        manifest_raw, manifest_raw_sha = _read_bound_bytes(
            self.manifest_path, "GPIC manifest"
        )
        if manifest_raw_sha != expected_manifest_sha256:
            raise GpicBlobError(
                "GPIC manifest SHA differs from the config-bound identity: "
                f"expected={expected_manifest_sha256}, actual={manifest_raw_sha}"
            )
        manifest = _parse_canonical_json(
            manifest_raw, self.manifest_path, "GPIC manifest"
        )
        _exact_keys(manifest, _MANIFEST_KEYS, "GPIC manifest")
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("status") != "PASS"
            or not isinstance(manifest.get("generated_at"), str)
            or not manifest["generated_at"]
        ):
            raise GpicBlobError("GPIC manifest identity/status is malformed")
        if manifest.get("manifest_sha256") != manifest_self_hash(manifest):
            raise GpicBlobError("GPIC manifest self-hash mismatch")
        fsx_success_raw, _fsx_success_sha = _read_bound_bytes(
            self.success_path, "GPIC FSx _SUCCESS"
        )
        marker_payload = (manifest_raw_sha + "\n").encode("ascii")
        if fsx_success_raw != marker_payload:
            raise GpicBlobError(
                "GPIC FSx _SUCCESS does not bind the exact manifest file SHA-256"
            )

        # The FSx view is an immutable compatibility projection.  Its local
        # copies of the production canonical manifest and verifier summary are
        # part of the trust root, not optional provenance.  Validate them before
        # opening either the dense index or any S3 blob.
        self.canonical_manifest_path = self.control_root / "canonical_manifest.json"
        canonical_raw, canonical_raw_sha = _read_bound_bytes(
            self.canonical_manifest_path, "GPIC production canonical manifest"
        )
        try:
            canonical = parse_production_json(
                canonical_raw, "GPIC production canonical manifest"
            )
            validate_canonical_manifest(canonical, canonical_raw)
        except ReleaseViewError as exc:
            raise GpicBlobError(
                f"invalid GPIC production canonical manifest: {exc}"
            ) from exc
        canonical_self_sha = canonical["manifest_sha256"]
        if canonical["build_jobtag"] != self.control_root.name:
            raise GpicBlobError(
                "GPIC production buildtag must equal the FSx view basename"
            )

        self.production_verifier_summary_path = (
            self.control_root / "verifier_summary.json"
        )
        summary_raw, _summary_raw_sha = _read_bound_bytes(
            self.production_verifier_summary_path,
            "GPIC production verifier summary",
        )
        try:
            summary = parse_production_json(
                summary_raw, "GPIC production verifier summary"
            )
            validate_verifier_summary(canonical, summary, summary_raw)
        except ReleaseViewError as exc:
            raise GpicBlobError(
                f"invalid GPIC production verifier summary: {exc}"
            ) from exc

        release = _exact_keys(manifest.get("release"), _RELEASE_KEYS, "GPIC release")
        try:
            projected_release = project_release(canonical, self.control_root)
        except (ReleaseViewError, ValueError, TypeError, KeyError) as exc:
            raise GpicBlobError(
                f"cannot project GPIC production canonical release: {exc}"
            ) from exc
        if projected_release != release:
            raise GpicBlobError(
                "GPIC compatibility release differs from the canonical projection"
            )
        release_sha = canonical_json_sha256(release)
        if manifest.get("release_sha256") != release_sha:
            raise GpicBlobError("GPIC release SHA-256 mismatch")
        if release.get("backend") != BACKEND:
            raise GpicBlobError(
                f"GPIC release backend must be {BACKEND!r}, got {release.get('backend')!r}"
            )
        catalog_raw = _exact_keys(release.get("catalog"), _CATALOG_KEYS, "GPIC catalog binding")
        gpic_file = _exact_keys(
            catalog_raw.get("gpic_file"), _GPIC_FILE_KEYS, "GPIC catalog file binding"
        )
        expected_catalog = {
            "catalog_hash": catalog.catalog_hash,
            "catalog_header_sha256": catalog.catalog_header_sha256,
            "gpic_file": {
                "name": catalog.gpic_name,
                "row_base": catalog.gpic_row_base,
                "rows": catalog.gpic_rows,
                "bytes": catalog.gpic_bytes,
                "sha256": catalog.gpic_sha256,
                "schema": catalog.gpic_schema,
            },
        }
        if dict(catalog_raw) != expected_catalog:
            raise GpicBlobError("GPIC blob release was built from a different catalog")

        layout = _exact_keys(release.get("layout"), _LAYOUT_KEYS, "GPIC layout")
        expected_layout = {
            "version": LAYOUT_VERSION,
            "record_struct": INDEX_STRUCT.format,
            "record_bytes": INDEX_RECORD_BYTES,
            "endianness": "little",
            "index_order": "row_id_ascending",
            "row_id_namespace": "catalog_global",
            "selection_mode": SELECTION_MODE,
            "compression": "none",
            "max_segment_bytes": MAX_SEGMENT_BYTES,
        }
        if dict(layout) != expected_layout:
            raise GpicBlobError(f"unsupported GPIC index layout: {layout}")

        row_count = _strict_int(release.get("row_count"), "GPIC release row_count", 1)
        if row_count != catalog.gpic_rows:
            raise GpicBlobError("GPIC release row_count differs from the catalog")
        index_binding = _exact_keys(release.get("index"), _FILE_KEYS, "GPIC index binding")
        index_bytes = _strict_int(index_binding.get("bytes"), "GPIC index bytes", 1)
        index_sha = index_binding.get("sha256")
        if not _is_sha256(index_sha):
            raise GpicBlobError("GPIC index SHA-256 is malformed")
        expected_index_bytes = row_count * INDEX_RECORD_BYTES
        if index_bytes != expected_index_bytes:
            raise GpicBlobError(
                f"GPIC index byte count must be rows*56: {index_bytes} != {expected_index_bytes}"
            )

        blobs_raw = release.get("blobs")
        if not isinstance(blobs_raw, list) or not blobs_raw:
            raise GpicBlobError("GPIC release blobs must be a non-empty list")
        blobs: dict[int, BlobBinding] = {}
        names: set[str] = set()
        canonical_blobs: list[dict[str, Any]] = []
        previous_blob_id = -1
        for position, raw in enumerate(blobs_raw):
            entry = _exact_keys(raw, _BLOB_KEYS, f"GPIC blob position {position}")
            blob_id = _strict_int(entry.get("blob_id"), "GPIC blob_id", 0)
            if blob_id > 0xFFFFFFFF:
                raise GpicBlobError("GPIC blob_id exceeds the uint32 index field")
            name = entry.get("name")
            blob_bytes = _strict_int(entry.get("bytes"), "GPIC blob bytes", 1)
            checksum = entry.get("sha256")
            if blob_id <= previous_blob_id:
                raise GpicBlobError(
                    "GPIC blob IDs must be unique and strictly ascending in manifest order"
                )
            previous_blob_id = blob_id
            _blob_relative_parts(name, f"GPIC blob {blob_id} name")
            expected_name = (
                f"workers/worker-{blob_id % 8}/blob-{blob_id:08d}.bin"
            )
            if name != expected_name:
                raise GpicBlobError(
                    f"GPIC blob name/ID binding mismatch: expected={expected_name!r}, "
                    f"actual={name!r}"
                )
            if name in names or not _is_sha256(checksum):
                raise GpicBlobError(f"duplicate name or malformed SHA for GPIC blob {blob_id}")
            names.add(name)
            canonical_blob = {
                "blob_id": blob_id,
                "name": name,
                "bytes": blob_bytes,
                "sha256": checksum,
            }
            canonical_blobs.append(canonical_blob)
            blobs[blob_id] = BlobBinding(blob_id, name, blob_bytes, checksum)
        payload_bytes = _strict_int(
            release.get("payload_bytes"), "GPIC payload bytes", 1
        )
        if payload_bytes != sum(blob.bytes for blob in blobs.values()):
            raise GpicBlobError("GPIC payload bytes do not close over declared blobs")
        if release.get("blob_set_sha256") != canonical_json_sha256(canonical_blobs):
            raise GpicBlobError("GPIC blob-set digest mismatch")

        try:
            nofile_soft, nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        except (OSError, ValueError) as exc:
            raise GpicBlobError(f"cannot read RLIMIT_NOFILE: {exc}") from exc
        startup_fd_count = _process_fd_count()
        startup_rss_bytes = _process_rss_bytes()
        if startup_fd_count < 0 or startup_rss_bytes < 0:
            raise GpicBlobError(
                "cannot read startup process FD/RSS health from /proc"
            )
        fd_reserve = 64
        required_soft_limit = startup_fd_count + self.fd_cache_size + fd_reserve
        if (
            nofile_soft != resource.RLIM_INFINITY
            and required_soft_limit > int(nofile_soft)
        ):
            raise GpicBlobError(
                "gpic_blob_fd_cache_size exceeds RLIMIT_NOFILE headroom: "
                f"startup_fds={startup_fd_count}, cache={self.fd_cache_size}, "
                f"reserve={fd_reserve}, soft_limit={nofile_soft}"
            )
        self.blob_count = len(blobs)
        self.rlimit_nofile_soft = int(nofile_soft)
        self.rlimit_nofile_hard = int(nofile_hard)
        self.startup_fd_count = startup_fd_count
        self.startup_rss_bytes = startup_rss_bytes
        self.fd_reserve = fd_reserve
        self.fd_cache_covers_blob_count = self.fd_cache_size >= self.blob_count

        storage = _exact_keys(release.get("storage"), _STORAGE_KEYS, "GPIC storage")
        s3_root_raw, index_path_raw = storage.get("s3_root"), storage.get("index_path")
        s3_init_binding = _exact_keys(
            storage.get("s3_init"), _S3_INIT_KEYS, "GPIC S3 init binding"
        )
        canonical_release = _exact_keys(
            canonical.get("release"), {"storage"}, "GPIC canonical release"
        )
        canonical_storage = _exact_keys(
            canonical_release.get("storage"),
            _STORAGE_KEYS,
            "GPIC canonical storage",
        )
        canonical_index_path_raw = canonical_storage.get("index_path")
        if (
            canonical_storage.get("s3_root") != s3_root_raw
            or canonical_storage.get("s3_init") != s3_init_binding
        ):
            raise GpicBlobError(
                "GPIC view and canonical storage differ outside index_path"
            )
        if (
            not isinstance(canonical_index_path_raw, str)
            or not Path(canonical_index_path_raw).is_absolute()
        ):
            raise GpicBlobError("GPIC canonical storage.index_path must be absolute")
        if not isinstance(s3_root_raw, str) or not Path(s3_root_raw).is_absolute():
            raise GpicBlobError("GPIC storage.s3_root must be absolute")
        if not isinstance(index_path_raw, str) or not Path(index_path_raw).is_absolute():
            raise GpicBlobError("GPIC storage.index_path must be absolute")
        if s3_init_binding.get("path") != "_INIT.json":
            raise GpicBlobError("GPIC storage.s3_init.path must be exactly '_INIT.json'")
        s3_init_bytes = _strict_int(
            s3_init_binding.get("bytes"), "GPIC S3 init bytes", 1
        )
        s3_init_sha = s3_init_binding.get("raw_sha256")
        if not _is_sha256(s3_init_sha):
            raise GpicBlobError("GPIC S3 init raw SHA-256 is malformed")
        self.s3_root = _real_directory(Path(s3_root_raw), "GPIC S3 blob root")
        if str(self.s3_root) != s3_root_raw:
            raise GpicBlobError("GPIC storage.s3_root must be canonical and non-symlink")
        if self.s3_root.name != canonical["build_jobtag"]:
            raise GpicBlobError(
                "GPIC production buildtag must equal the S3 root basename"
            )
        try:
            s3_root_names = {path.name for path in self.s3_root.iterdir()}
        except OSError as exc:
            raise GpicBlobError(f"cannot enumerate GPIC S3 root: {exc}") from exc
        expected_s3_root_names = {"_INIT.json", "_SUCCESS", "workers", "controls"}
        if s3_root_names != expected_s3_root_names:
            raise GpicBlobError(
                "GPIC S3 root exact-set drift: "
                f"missing={sorted(expected_s3_root_names-s3_root_names)}, "
                f"unexpected={sorted(s3_root_names-expected_s3_root_names)}"
            )
        s3_init_raw, s3_init_raw_sha = _read_bound_bytes(
            self.s3_root / "_INIT.json", "GPIC S3 _INIT.json"
        )
        if len(s3_init_raw) != s3_init_bytes or s3_init_raw_sha != s3_init_sha:
            raise GpicBlobError("GPIC S3 _INIT.json file binding mismatch")
        self.s3_workers_root = _real_directory(
            self.s3_root / "workers", "GPIC S3 workers root"
        )
        expected_worker_names = {f"worker-{worker}" for worker in range(8)}
        try:
            live_worker_names = {path.name for path in self.s3_workers_root.iterdir()}
        except OSError as exc:
            raise GpicBlobError(f"cannot enumerate GPIC S3 workers root: {exc}") from exc
        if live_worker_names != expected_worker_names:
            raise GpicBlobError(
                "GPIC S3 workers exact-set drift: "
                f"missing={sorted(expected_worker_names-live_worker_names)}, "
                f"unexpected={sorted(live_worker_names-expected_worker_names)}"
            )
        blobs_by_name = {blob.name: blob for blob in blobs.values()}
        for worker in range(8):
            worker_name = f"worker-{worker}"
            worker_root = _real_directory(
                self.s3_workers_root / worker_name,
                f"GPIC S3 {worker_name}",
            )
            expected_blob_names = {
                Path(name).name
                for name in blobs_by_name
                if name.startswith(f"workers/{worker_name}/")
            }
            try:
                live_blob_names = {path.name for path in worker_root.iterdir()}
            except OSError as exc:
                raise GpicBlobError(
                    f"cannot enumerate GPIC S3 {worker_name}: {exc}"
                ) from exc
            if live_blob_names != expected_blob_names:
                raise GpicBlobError(
                    f"GPIC S3 {worker_name} blob exact-set drift: "
                    f"missing={sorted(expected_blob_names-live_blob_names)[:8]}, "
                    f"unexpected={sorted(live_blob_names-expected_blob_names)[:8]}"
                )
            for blob_name in live_blob_names:
                binding = blobs_by_name[f"workers/{worker_name}/{blob_name}"]
                metadata = _regular_metadata(
                    worker_root / blob_name, f"GPIC S3 blob {binding.blob_id}"
                )
                if int(metadata.st_size) != binding.bytes:
                    raise GpicBlobError(
                        f"GPIC S3 blob size drift blob_id={binding.blob_id}: "
                        f"expected={binding.bytes}, actual={metadata.st_size}"
                    )
        # The release has two independent publication planes.  The immutable S3
        # copy must be byte-identical to the production canonical copied into
        # the FSx view; it intentionally differs from the derived adapter
        # manifest consumed above.
        self.s3_control_root = _real_directory(
            self.s3_root / "controls", "GPIC S3 control root"
        )
        s3_manifest_path = self.s3_control_root / "manifest.json"
        s3_success_path = self.s3_root / "_SUCCESS"
        s3_manifest_raw, s3_manifest_raw_sha = _read_bound_bytes(
            s3_manifest_path, "GPIC S3 manifest"
        )
        s3_success_raw, _s3_success_sha = _read_bound_bytes(
            s3_success_path, "GPIC S3 _SUCCESS"
        )
        if (
            s3_manifest_raw != canonical_raw
            or s3_manifest_raw_sha != canonical_raw_sha
        ):
            raise GpicBlobError(
                "GPIC local and S3 production canonical manifest raw bytes differ"
            )
        canonical_marker_payload = (canonical_self_sha + "\n").encode("ascii")
        if s3_success_raw != canonical_marker_payload:
            raise GpicBlobError(
                "GPIC S3 _SUCCESS does not bind the canonical manifest self-hash"
            )
        try:
            s3_control_names = {path.name for path in self.s3_control_root.iterdir()}
        except OSError as exc:
            raise GpicBlobError(f"cannot enumerate GPIC S3 control root: {exc}") from exc
        if s3_control_names != {"manifest.json"}:
            raise GpicBlobError(
                "GPIC S3 control root exact-set drift: "
                f"missing={sorted({'manifest.json'}-s3_control_names)}, "
                f"unexpected={sorted(s3_control_names-{'manifest.json'})}"
            )
        index_declared = Path(index_path_raw)
        _regular_metadata(index_declared, "GPIC index")
        self.index_path = index_declared.resolve(strict=True)
        if str(self.index_path) != index_path_raw:
            raise GpicBlobError("GPIC storage.index_path must be canonical and non-symlink")
        if self.index_path.parent != self.control_root:
            raise GpicBlobError("GPIC index must be a direct child of the FSx control root")
        index_metadata = _regular_metadata(self.index_path, "GPIC index")
        if int(index_metadata.st_size) != index_bytes:
            raise GpicBlobError("GPIC live index size differs from the manifest")
        verification = _exact_keys(
            manifest.get("verification"),
            _VERIFICATION_BINDING_KEYS,
            "GPIC verification binding",
        )
        report_path_raw = verification.get("path")
        if not isinstance(report_path_raw, str) or not Path(report_path_raw).is_absolute():
            raise GpicBlobError("GPIC verifier report path must be absolute")
        report_declared = Path(report_path_raw)
        _regular_metadata(report_declared, "GPIC verifier report")
        self.verifier_report_path = report_declared.resolve(strict=True)
        if str(self.verifier_report_path) != report_path_raw:
            raise GpicBlobError(
                "GPIC verifier report path must be canonical and non-symlink"
            )
        if self.verifier_report_path.parent != self.control_root:
            raise GpicBlobError(
                "GPIC verifier report must be a direct child of the FSx control root"
            )
        report_raw, report_raw_sha = _read_bound_bytes(
            self.verifier_report_path, "GPIC verifier report"
        )
        if (
            verification.get("schema_version") != VERIFY_SCHEMA_VERSION
            or _strict_int(verification.get("bytes"), "GPIC verifier bytes", 1)
            != len(report_raw)
            or verification.get("sha256") != report_raw_sha
        ):
            raise GpicBlobError("GPIC verifier report file binding mismatch")
        report = _parse_canonical_json(
            report_raw, self.verifier_report_path, "GPIC verifier report"
        )
        _exact_keys(report, _VERIFY_REPORT_KEYS, "GPIC verifier report")
        expected_report = {
            "schema_version": VERIFY_SCHEMA_VERSION,
            "status": "PASS",
            "release_sha256": release_sha,
            "catalog_hash": catalog.catalog_hash,
            "index_sha256": index_sha,
            "index_bytes": index_bytes,
            "row_id_start": catalog.gpic_row_base,
            "row_id_stop_exclusive": catalog.gpic_row_base + catalog.gpic_rows,
            "row_count": catalog.gpic_rows,
            "blob_count": len(blobs),
            "payload_bytes": payload_bytes,
            "full_index_records_verified": catalog.gpic_rows,
            "full_segment_sha256_verified": True,
            "full_blob_sha256_verified": True,
        }
        for key, expected in expected_report.items():
            if report.get(key) != expected:
                raise GpicBlobError(
                    f"GPIC verifier report mismatch for {key}: "
                    f"expected={expected!r}, actual={report.get(key)!r}"
                )
        if (
            not isinstance(report.get("generated_at"), str)
            or not report["generated_at"]
            or report.get("report_sha256") != report_self_hash(report)
        ):
            raise GpicBlobError("GPIC verifier report self-hash/status mismatch")

        expected_control_names = {
            "manifest.json",
            "canonical_manifest.json",
            "verifier_summary.json",
            "_SUCCESS",
            self.index_path.name,
            self.verifier_report_path.name,
        }
        if len(expected_control_names) != 6:
            raise GpicBlobError(
                "GPIC index/verifier names collide with immutable control names"
            )
        try:
            live_control_names = {path.name for path in self.control_root.iterdir()}
        except OSError as exc:
            raise GpicBlobError(f"cannot enumerate GPIC control root: {exc}") from exc
        if live_control_names != expected_control_names:
            raise GpicBlobError(
                "GPIC FSx control root exact-set drift: "
                f"missing={sorted(expected_control_names-live_control_names)}, "
                f"unexpected={sorted(live_control_names-expected_control_names)}"
            )

        self.manifest = manifest
        self.manifest_sha256 = manifest_raw_sha
        self.canonical_manifest = canonical
        self.canonical_manifest_sha256 = str(canonical_self_sha)
        self.canonical_manifest_raw_sha256 = canonical_raw_sha
        self.production_verifier_summary = summary
        self.release_sha256 = release_sha
        self.index_sha256 = str(index_sha)
        self.catalog = catalog
        self.row_id_start = catalog.gpic_row_base
        self.row_id_stop = catalog.gpic_row_base + catalog.gpic_rows
        self.row_count = catalog.gpic_rows
        self.blobs = blobs
        self.payload_bytes = payload_bytes

        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            self._index_fd = os.open(self.index_path, flags)
            opened = os.fstat(self._index_fd)
            if _identity(opened) != _identity(index_metadata):
                raise GpicBlobError("GPIC index changed while opening mmap")
            self._index_identity = _identity(opened)
            self._index_mmap = mmap.mmap(
                self._index_fd, index_bytes, access=mmap.ACCESS_READ
            )
            self._fd_cache = _BlobFDCache(
                self.s3_root, self.blobs, max_open=self.fd_cache_size
            )
        except BaseException:
            self._close_partial()
            raise

    def _close_partial(self) -> None:
        if self._fd_cache is not None:
            self._fd_cache.close()
        if self._index_mmap is not None:
            self._index_mmap.close()
            self._index_mmap = None
        if self._index_fd is not None:
            os.close(self._index_fd)
            self._index_fd = None

    def compute_index_sha256(self) -> str:
        if self._closed:
            raise GpicBlobError("cannot verify a closed GPIC overlay")
        if self._index_fd is None:
            raise GpicBlobError("cannot verify a closed GPIC index FD")
        return _sha256_open_fd(
            self._index_fd,
            self._index_identity,
            label="GPIC mmap index",
        )

    def accept_collective_index_verification(self, actual_sha256: str) -> None:
        if actual_sha256 != self.index_sha256:
            raise GpicBlobError(
                "GPIC index SHA-256 mismatch: "
                f"expected={self.index_sha256}, actual={actual_sha256}"
            )
        if self._index_fd is None or _identity(os.fstat(self._index_fd)) != self._index_identity:
            raise GpicBlobError("GPIC mmap index identity changed during SHA verification")
        live_index = _regular_metadata(self.index_path, "GPIC live index")
        if _identity(live_index) != self._index_identity:
            raise GpicBlobError(
                "GPIC live index path was replaced during mmap SHA verification"
            )
        # Boundary records cheaply close the global-row namespace before any
        # supply thread is allowed to run. Every interior record is checked on use.
        self._entry_for_row(self.row_id_start)
        self._entry_for_row(self.row_id_stop - 1)
        self._index_verified = True

    def verify_index_sha256(self) -> str:
        actual = self.compute_index_sha256()
        self.accept_collective_index_verification(actual)
        return actual

    def _entry_for_row(self, row_id: int) -> IndexEntry:
        if type(row_id) is not int or row_id < self.row_id_start or row_id >= self.row_id_stop:
            raise GpicBlobError(
                f"GPIC global row_id {row_id!r} outside "
                f"[{self.row_id_start},{self.row_id_stop})"
            )
        mapping = self._index_mmap
        if mapping is None:
            raise GpicBlobError("GPIC index mmap is closed")
        local_row = row_id - self.row_id_start
        position = local_row * INDEX_RECORD_BYTES
        try:
            record = INDEX_STRUCT.unpack_from(mapping, position)
        except (ValueError, struct.error) as exc:
            raise GpicBlobError(
                f"cannot unpack GPIC index record row_id={row_id}: {exc}"
            ) from exc
        entry = IndexEntry(*record)
        if entry.row_id != row_id:
            raise GpicBlobError(
                f"GPIC index row identity mismatch: requested={row_id}, record={entry.row_id}"
            )
        blob = self.blobs.get(entry.blob_id)
        if blob is None:
            raise GpicBlobError(
                f"GPIC full-catalog index has missing/invalid blob_id={entry.blob_id} "
                f"at row_id={row_id}"
            )
        if (
            entry.length <= 0
            or entry.length > MAX_SEGMENT_BYTES
            or entry.offset + entry.length > blob.bytes
        ):
            raise GpicBlobError(
                f"GPIC index range is invalid row_id={row_id}, blob_id={entry.blob_id}, "
                f"offset={entry.offset}, length={entry.length}, blob_bytes={blob.bytes}"
            )
        if entry.sha256_raw == b"\x00" * 32:
            raise GpicBlobError(
                f"GPIC full-catalog index contains a zero/sentinel SHA at row_id={row_id}"
            )
        return entry

    def _pread_exact(self, descriptor: int, length: int, offset: int, row_id: int) -> bytes:
        parts: list[bytes] = []
        consumed = 0
        with self._metrics_lock:
            self._metrics["pread_calls"] += 1
            self._metrics["pread_requested_bytes"] += length
        while consumed < length:
            try:
                block = os.pread(descriptor, length - consumed, offset + consumed)
            except InterruptedError:
                with self._metrics_lock:
                    self._metrics["pread_eintr_retries"] += 1
                continue
            except OSError as exc:
                with self._metrics_lock:
                    self._metrics["pread_returned_bytes"] += consumed
                    self._metrics["pread_errors"] += 1
                raise GpicBlobError(
                    f"GPIC blob pread failed row_id={row_id}, "
                    f"offset={offset + consumed}: {exc}"
                ) from exc
            if not block:
                with self._metrics_lock:
                    self._metrics["pread_returned_bytes"] += consumed
                    self._metrics["short_read_failures"] += 1
                raise GpicBlobError(
                    f"short GPIC blob pread row_id={row_id}: wanted={length}, got={consumed}"
                )
            parts.append(block)
            consumed += len(block)
        with self._metrics_lock:
            self._metrics["pread_returned_bytes"] += consumed
        return b"".join(parts)

    def read(self, row_id: int) -> bytes:
        if self._closed:
            raise GpicBlobError("GPIC blob overlay is closed")
        if not self._index_verified:
            raise GpicBlobError("GPIC index was not SHA-verified before first read")
        started = time.perf_counter()
        entry = self._entry_for_row(row_id)
        cache = self._fd_cache
        if cache is None:
            raise GpicBlobError("GPIC blob FD cache is closed")
        try:
            with cache.acquire(entry.blob_id) as opened:
                before = os.fstat(opened.descriptor)
                if _identity(before) != opened.identity:
                    raise GpicBlobError(
                        f"GPIC blob FD changed before pread row_id={row_id}, "
                        f"blob_id={entry.blob_id}"
                    )
                raw = self._pread_exact(
                    opened.descriptor, entry.length, entry.offset, row_id
                )
                after = os.fstat(opened.descriptor)
                if _identity(after) != opened.identity:
                    raise GpicBlobError(
                        f"GPIC blob FD changed during pread row_id={row_id}, "
                        f"blob_id={entry.blob_id}"
                    )
            if len(raw) != entry.length:
                raise GpicBlobError(
                    f"GPIC blob read length mismatch row_id={row_id}: "
                    f"{len(raw)} != {entry.length}"
                )
            actual_sha = hashlib.sha256(raw).digest()
            if actual_sha != entry.sha256_raw:
                raise GpicBlobError(
                    f"GPIC segment SHA-256 mismatch row_id={row_id}, blob_id={entry.blob_id}"
                )
        except BaseException:
            with self._metrics_lock:
                self._metrics["integrity_failures"] += 1
            raise
        elapsed = time.perf_counter() - started
        with self._metrics_lock:
            self._metrics["read_items"] += 1
            self._metrics["read_bytes"] += len(raw)
            self._metrics["read_seconds_micro"] += int(round(elapsed * 1_000_000))
        return raw

    def snapshot(self) -> dict[str, float]:
        with self._metrics_lock:
            result: dict[str, float] = {
                key: float(value) for key, value in self._metrics.items()
            }
        cache = self._fd_cache
        if cache is not None:
            result.update({key: float(value) for key, value in cache.snapshot().items()})
        requested = result.get("pread_requested_bytes", 0.0)
        returned = result.get("pread_returned_bytes", 0.0)
        result["read_amplification"] = returned / requested if requested > 0 else 0.0
        result["manifest_bound"] = 1.0
        result["index_verified"] = 1.0 if self._index_verified else 0.0
        return result

    def health_snapshot(self) -> dict[str, float]:
        """Low-frequency process health used by the all-rank train monitor."""

        result = self.snapshot()
        result["process_rss_bytes"] = float(_process_rss_bytes())
        result["process_fd_count"] = float(_process_fd_count())
        result["startup_rss_bytes"] = float(self.startup_rss_bytes)
        result["startup_fd_count"] = float(self.startup_fd_count)
        result["rlimit_nofile_soft"] = float(self.rlimit_nofile_soft)
        result["rlimit_nofile_hard"] = float(self.rlimit_nofile_hard)
        result["fd_reserve"] = float(self.fd_reserve)
        result["blob_count"] = float(self.blob_count)
        result["fd_cache_covers_blob_count"] = (
            1.0 if self.fd_cache_covers_blob_count else 0.0
        )
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._close_partial()
        self._closed = True


__all__ = [
    "BACKEND",
    "CatalogBinding",
    "GpicBlobError",
    "GpicBlobOverlay",
    "INDEX_RECORD_BYTES",
    "INDEX_STRUCT",
    "LAYOUT_VERSION",
    "MAX_SEGMENT_BYTES",
    "SCHEMA_VERSION",
    "SELECTION_MODE",
    "VERIFY_SCHEMA_VERSION",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "manifest_self_hash",
    "report_self_hash",
]
