#!/usr/bin/env python3
"""Lightweight positive/negative tests for ``offline_gpic_blob_v1`` runtime.

The test builds only three tiny rows in a temporary POSIX directory.  It never
touches the production catalog, S3 mounts, FSx artifacts, or GPUs.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import training.dataline_offline.gpic_blob as gpic_blob_module

from training.dataline_offline.gpic_blob import (
    BACKEND,
    INDEX_RECORD_BYTES,
    INDEX_STRUCT,
    LAYOUT_VERSION,
    MAX_SEGMENT_BYTES,
    SCHEMA_VERSION,
    SELECTION_MODE,
    VERIFY_SCHEMA_VERSION,
    CatalogBinding,
    GpicBlobError,
    GpicBlobOverlay,
    canonical_json_bytes,
    canonical_json_sha256,
    manifest_self_hash,
    report_self_hash,
    _blob_relative_parts,
    _identity,
)
from training.dataline_offline.gpic_release_view import (
    parse_production_json,
    production_json_bytes,
    production_json_sha256,
    production_self_hash,
    project_release,
)


CATALOG = CatalogBinding(
    catalog_hash="1" * 64,
    catalog_header_sha256="2" * 64,
    gpic_name="gpic.parquet",
    gpic_row_base=100,
    gpic_rows=3,
    gpic_bytes=1234,
    gpic_sha256="3" * 64,
    gpic_schema="row_id: int64 not null\nimage_path: string not null",
)
PAYLOADS = (b"alpha-image", b"beta-image-payload", b"gamma")
BUILDTAG = "test-gpic-production-build"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes) -> None:
    path.write_bytes(raw)


def _s3_root(parent: Path) -> Path:
    return parent / "s3" / BUILDTAG


def _rewrite_production_canonical(
    parent: Path, control: Path, mutate
) -> None:
    local_path = control / "canonical_manifest.json"
    canonical = parse_production_json(
        local_path.read_bytes(), "runtime negative canonical"
    )
    mutate(canonical)
    canonical["manifest_sha256"] = production_self_hash(
        canonical, "manifest_sha256"
    )
    raw = production_json_bytes(canonical)
    local_path.write_bytes(raw)
    s3_root = _s3_root(parent)
    (s3_root / "controls/manifest.json").write_bytes(raw)
    (s3_root / "_SUCCESS").write_text(
        canonical["manifest_sha256"] + "\n", encoding="ascii"
    )


def _publish(
    parent: Path,
    *,
    row_ids: tuple[int, int, int] = (100, 101, 102),
    blob_ids: tuple[int, int, int] = (0, 0, 1),
    manifest_blob_ids: tuple[int, int] = (0, 1),
    manifest_blob_names: tuple[str, str] | None = None,
    segment_shas: tuple[bytes, bytes, bytes] | None = None,
    locator_lengths: tuple[int, int, int] | None = None,
) -> tuple[Path, str]:
    assert len(manifest_blob_ids) == 2
    if manifest_blob_names is None:
        manifest_blob_names = tuple(
            f"workers/worker-{blob_id % 8}/blob-{blob_id:08d}.bin"
            for blob_id in manifest_blob_ids
        )
    control = parent / "view" / BUILDTAG
    s3_root = parent / "s3" / BUILDTAG
    source_root = parent / "source"
    control.mkdir(parents=True)
    s3_root.mkdir(parents=True)
    source_root.mkdir()
    workers_root = s3_root / "workers"
    workers_root.mkdir()
    for worker in range(8):
        (workers_root / f"worker-{worker}").mkdir()
    s3_controls = s3_root / "controls"
    s3_controls.mkdir()
    init_raw = canonical_json_bytes(
        {"schema_version": "offline_gpic_blob_init_test_v1", "status": "CLAIMED"}
    )
    _write(s3_root / "_INIT.json", init_raw)

    blob_payloads = (PAYLOADS[0] + PAYLOADS[1], PAYLOADS[2])
    adapter_blobs = []
    for blob_id, name, payload in zip(
        manifest_blob_ids, manifest_blob_names, blob_payloads
    ):
        _write(s3_root / name, payload)
        adapter_blobs.append(
            {
                "blob_id": blob_id,
                "name": name,
                "bytes": len(payload),
                "sha256": _sha(payload),
            }
        )

    default_shas = tuple(hashlib.sha256(payload).digest() for payload in PAYLOADS)
    shas = segment_shas if segment_shas is not None else default_shas
    lengths = locator_lengths if locator_lengths is not None else tuple(map(len, PAYLOADS))
    records = (
        INDEX_STRUCT.pack(row_ids[0], blob_ids[0], 0, lengths[0], shas[0]),
        INDEX_STRUCT.pack(
            row_ids[1], blob_ids[1], len(PAYLOADS[0]), lengths[1], shas[1]
        ),
        INDEX_STRUCT.pack(row_ids[2], blob_ids[2], 0, lengths[2], shas[2]),
    )
    index_raw = b"".join(records)
    source_index_path = source_root / "dense_index.bin"
    index_path = control / "dense_index.bin"
    _write(source_index_path, index_raw)
    _write(index_path, index_raw)

    canonical_blobs = [
        {
            "blob_id": blob_id,
            "bytes": len(payload),
            "key": str(
                s3_root
                / "workers"
                / f"worker-{blob_id % 8}"
                / f"blob-{blob_id:08d}.bin"
            ),
            "name": f"blob-{blob_id:08d}.bin",
            "segments": 2 if blob_id == 0 else 1,
            "sequence": blob_id // 8,
            "sha256": _sha(payload),
            "worker_id": blob_id % 8,
        }
        for blob_id, payload in enumerate(blob_payloads)
    ]
    canonical_workers = []
    for worker_id in range(8):
        owned = [blob for blob in canonical_blobs if blob["worker_id"] == worker_id]
        canonical_workers.append(
            {
                "worker_id": worker_id,
                "blob_count": len(owned),
                "rows": sum(int(blob["segments"]) for blob in owned),
                "payload_bytes": sum(int(blob["bytes"]) for blob in owned),
            }
        )
    fixed_hashes = {
        "build_manifest_raw_sha256": _sha(b"build-manifest-raw"),
        "build_manifest_sha256": _sha(b"build-manifest-self"),
        "build_complete_raw_sha256": _sha(b"build-complete-raw"),
        "build_complete_sha256": _sha(b"build-complete-self"),
    }
    verifier_code = {"record_struct": INDEX_STRUCT.format, "record_bytes": 56}
    summary: dict[str, Any] = {
        "schema_version": "offline_gpic_s3_production_verifier_summary_v1",
        "generated_at": "2026-07-12T00:00:00+00:00",
        "status": "PASS",
        "build_jobtag": BUILDTAG,
        "verifier_jobtag": "test-gpic-production-verify",
        "workers": [
            {
                **worker,
                "report_raw_sha256": _sha(
                    f"worker-{worker['worker_id']}-raw".encode("ascii")
                ),
                "report_sha256": _sha(
                    f"worker-{worker['worker_id']}-self".encode("ascii")
                ),
            }
            for worker in canonical_workers
        ],
        "worker_count": 8,
        "rows": CATALOG.gpic_rows,
        "payload_bytes": sum(len(payload) for payload in blob_payloads),
        "blob_count": len(canonical_blobs),
        "keys_sha256": production_json_sha256(
            sorted(str(blob["key"]) for blob in canonical_blobs)
        ),
        "builder_boot_id_sha256": sorted(
            _sha(f"builder-boot-{worker}".encode("ascii")) for worker in range(8)
        ),
        "verifier_boot_id_sha256": sorted(
            _sha(f"verifier-boot-{worker}".encode("ascii")) for worker in range(8)
        ),
        "physical_hosts_disjoint": True,
        **fixed_hashes,
        "verify_approval_raw_sha256": _sha(b"verify-approval-raw"),
        "verify_approval_sha256": _sha(b"verify-approval-self"),
        "code": verifier_code,
        "integrity_errors": 0,
        "short_reads": 0,
        "fallbacks": 0,
    }
    summary["verifier_summary_sha256"] = production_self_hash(
        summary, "verifier_summary_sha256"
    )
    summary_raw = production_json_bytes(summary)

    canonical: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": "2026-07-12T00:00:01+00:00",
        "status": "PRODUCTION_VERIFIED",
        "build_jobtag": BUILDTAG,
        "verifier_jobtag": summary["verifier_jobtag"],
        "release": {
            "storage": {
                "s3_root": str(s3_root.resolve(strict=True)),
                "index_path": str(source_index_path.resolve(strict=True)),
                "s3_init": {
                    "path": "_INIT.json",
                    "bytes": len(init_raw),
                    "raw_sha256": _sha(init_raw),
                },
            }
        },
        "catalog": {
            "catalog_hash": CATALOG.catalog_hash,
            "catalog_rows": CATALOG.gpic_row_base + CATALOG.gpic_rows,
            "gpic": {
                "name": CATALOG.gpic_name,
                "row_base": CATALOG.gpic_row_base,
                "rows": CATALOG.gpic_rows,
                "bytes": CATALOG.gpic_bytes,
                "sha256": CATALOG.gpic_sha256,
                "schema": CATALOG.gpic_schema,
                "identity": {
                    "ctime_ns": 1,
                    "dev": 1,
                    "ino": 1,
                    "mtime_ns": 1,
                    "size": CATALOG.gpic_bytes,
                },
            },
            "header": {
                "bytes": 1,
                "identity": {
                    "ctime_ns": 1,
                    "dev": 1,
                    "ino": 2,
                    "mtime_ns": 1,
                    "size": 1,
                },
                "name": "header.json",
                "raw_sha256": CATALOG.catalog_header_sha256,
            },
            "root": str(parent.resolve(strict=True)),
        },
        "partition_mode": "shard_modulo",
        "partition_plan_raw_sha256": _sha(b"partition-plan-raw"),
        "partition_plan_sha256": _sha(b"partition-plan-self"),
        "source_authority": {},
        "limits": {
            "workers": 8,
            "gpic_rows": CATALOG.gpic_rows,
            "dense_index_bytes": len(index_raw),
            "record_bytes": INDEX_RECORD_BYTES,
        },
        "tar_workers": 8,
        "full_tar_mapping": {},
        "dense_index": {
            "bytes": len(index_raw),
            "name": "dense_index.bin",
            "sha256": _sha(index_raw),
        },
        "record_struct": INDEX_STRUCT.format,
        "record_bytes": INDEX_RECORD_BYTES,
        "rows": CATALOG.gpic_rows,
        "payload_bytes": sum(len(payload) for payload in blob_payloads),
        "blob_count": len(canonical_blobs),
        "blobs": canonical_blobs,
        "workers": canonical_workers,
        **fixed_hashes,
        "verifier_summary_raw_sha256": _sha(summary_raw),
        "verifier_summary_sha256": summary["verifier_summary_sha256"],
        "s3_init_sha256": _sha(b"s3-init-self"),
        "verifier_code": verifier_code,
        "publication_semantics": (
            "canonical_manifest_then_fsx_success_then_single_s3_root_success_absolute_last"
        ),
    }
    canonical["manifest_sha256"] = production_self_hash(
        canonical, "manifest_sha256"
    )
    canonical_raw = production_json_bytes(canonical)

    release = project_release(canonical, control)
    if (
        manifest_blob_ids != (0, 1)
        or manifest_blob_names
        != (
            "workers/worker-0/blob-00000000.bin",
            "workers/worker-1/blob-00000001.bin",
        )
    ):
        release["blobs"] = adapter_blobs
        release["payload_bytes"] = sum(
            int(blob["bytes"]) for blob in adapter_blobs
        )
        release["blob_set_sha256"] = canonical_json_sha256(adapter_blobs)
    release_sha = canonical_json_sha256(release)
    report: dict[str, Any] = {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "status": "PASS",
        "generated_at": "2026-07-12T00:00:00+00:00",
        "release_sha256": release_sha,
        "catalog_hash": CATALOG.catalog_hash,
        "index_sha256": _sha(index_raw),
        "index_bytes": len(index_raw),
        "row_id_start": CATALOG.gpic_row_base,
        "row_id_stop_exclusive": CATALOG.gpic_row_base + CATALOG.gpic_rows,
        "row_count": CATALOG.gpic_rows,
        "blob_count": len(adapter_blobs),
        "payload_bytes": sum(len(payload) for payload in blob_payloads),
        "full_index_records_verified": CATALOG.gpic_rows,
        "full_segment_sha256_verified": True,
        "full_blob_sha256_verified": True,
    }
    report["report_sha256"] = report_self_hash(report)
    report_raw = canonical_json_bytes(report)
    report_path = control / "verifier_report.json"
    _write(report_path, report_raw)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "generated_at": "2026-07-12T00:00:01+00:00",
        "release": release,
        "release_sha256": release_sha,
        "verification": {
            "path": str(report_path.resolve(strict=True)),
            "bytes": len(report_raw),
            "sha256": _sha(report_raw),
            "schema_version": VERIFY_SCHEMA_VERSION,
        },
    }
    manifest["manifest_sha256"] = manifest_self_hash(manifest)
    manifest_raw = canonical_json_bytes(manifest)
    manifest_sha = _sha(manifest_raw)
    _write(control / "manifest.json", manifest_raw)
    _write(control / "canonical_manifest.json", canonical_raw)
    _write(control / "verifier_summary.json", summary_raw)
    _write(control / "_SUCCESS", (manifest_sha + "\n").encode("ascii"))
    _write(s3_controls / "manifest.json", canonical_raw)
    _write(
        s3_root / "_SUCCESS",
        (canonical["manifest_sha256"] + "\n").encode("ascii"),
    )
    return control, manifest_sha


def _expect_error(label: str, callback, needle: str | None = None) -> None:
    try:
        callback()
    except GpicBlobError as exc:
        if needle is not None and needle not in str(exc):
            raise AssertionError(f"{label}: wrong error: {exc}") from exc
    else:
        raise AssertionError(f"{label}: expected GpicBlobError")


def _open(root: Path, manifest_sha: str, cache_size: int = 1) -> GpicBlobOverlay:
    return GpicBlobOverlay(
        root,
        expected_manifest_sha256=manifest_sha,
        catalog=CATALOG,
        fd_cache_size=cache_size,
    )


def positive_and_lru() -> None:
    with tempfile.TemporaryDirectory(prefix="gpic-blob-positive.") as raw:
        root, manifest_sha = _publish(Path(raw))
        overlay = _open(root, manifest_sha, cache_size=1)
        _expect_error(
            "read-before-index-verification",
            lambda: overlay.read(100),
            "not SHA-verified",
        )
        overlay.verify_index_sha256()
        assert [overlay.read(row) for row in (100, 101, 102)] == list(PAYLOADS)
        assert overlay.read(100) == PAYLOADS[0]
        metrics = overlay.snapshot()
        assert metrics["read_items"] == 4
        assert metrics["read_amplification"] == 1.0
        assert metrics["fd_evictions"] >= 1
        health = overlay.health_snapshot()
        assert health["process_rss_bytes"] > 0
        assert health["process_fd_count"] > 0
        assert health["startup_rss_bytes"] > 0
        assert health["startup_fd_count"] > 0
        assert health["rlimit_nofile_soft"] != 0
        assert health["blob_count"] == 2
        assert health["fd_cache_size"] == 1
        assert health["fd_cache_covers_blob_count"] == 0

        # The same process-level cache is shared safely by concurrent callers.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(overlay.read, (100, 102, 101, 102)))
        assert values == [PAYLOADS[0], PAYLOADS[2], PAYLOADS[1], PAYLOADS[2]]

        cache = overlay._fd_cache  # noqa: SLF001 - deliberate lease negative
        assert cache is not None
        lease = cache.acquire(0)
        _expect_error("close-with-active-lease", overlay.close, "active leases")
        lease.release()
        overlay.close()
        assert overlay.snapshot()["fd_current"] == 0
        _expect_error("read-after-close", lambda: overlay.read(100), "closed")

def negative_manifest_and_success() -> None:
    with tempfile.TemporaryDirectory(prefix="gpic-blob-rlimit.") as raw:
        root, manifest_sha = _publish(Path(raw))
        real_getrlimit = gpic_blob_module.resource.getrlimit
        gpic_blob_module.resource.getrlimit = lambda _which: (8, 8)
        try:
            _expect_error(
                "fd-cache-rlimit-headroom",
                lambda: _open(root, manifest_sha),
                "RLIMIT_NOFILE headroom",
            )
        finally:
            gpic_blob_module.resource.getrlimit = real_getrlimit

    with tempfile.TemporaryDirectory(prefix="gpic-blob-controls.") as raw:
        root, manifest_sha = _publish(Path(raw))
        _expect_error(
            "wrong-config-manifest-sha",
            lambda: _open(root, "f" * 64),
            "config-bound",
        )
        (root / "_SUCCESS").write_text("0" * 64 + "\n", encoding="ascii")
        _expect_error(
            "success-does-not-bind-manifest",
            lambda: _open(root, manifest_sha),
            "FSx _SUCCESS",
        )

    with tempfile.TemporaryDirectory(prefix="gpic-blob-s3-marker.") as raw:
        root, manifest_sha = _publish(Path(raw))
        (_s3_root(Path(raw)) / "_SUCCESS").write_text(
            "0" * 64 + "\n", encoding="ascii"
        )
        _expect_error(
            "s3-success-does-not-bind-manifest",
            lambda: _open(root, manifest_sha),
            "S3 _SUCCESS",
        )

    with tempfile.TemporaryDirectory(prefix="gpic-blob-s3-manifest.") as raw:
        root, manifest_sha = _publish(Path(raw))
        s3_manifest = _s3_root(Path(raw)) / "controls/manifest.json"
        s3_manifest.write_bytes(s3_manifest.read_bytes() + b"\n")
        _expect_error(
            "s3-manifest-raw-drift",
            lambda: _open(root, manifest_sha),
            "canonical manifest raw bytes differ",
        )

    with tempfile.TemporaryDirectory(prefix="gpic-blob-projection-blob.") as raw:
        parent = Path(raw)
        root, manifest_sha = _publish(parent)
        _rewrite_production_canonical(
            parent,
            root,
            lambda canonical: canonical["blobs"][0].__setitem__(
                "sha256", "f" * 64
            ),
        )
        _expect_error(
            "canonical-blob-projection-drift",
            lambda: _open(root, manifest_sha),
            "canonical projection",
        )

    with tempfile.TemporaryDirectory(prefix="gpic-blob-projection-identity.") as raw:
        parent = Path(raw)
        root, manifest_sha = _publish(parent)
        _rewrite_production_canonical(
            parent,
            root,
            lambda canonical: canonical["catalog"].__setitem__(
                "catalog_hash", "4" * 64
            ),
        )
        _expect_error(
            "canonical-identity-projection-drift",
            lambda: _open(root, manifest_sha),
            "canonical projection",
        )

    with tempfile.TemporaryDirectory(prefix="gpic-blob-s3-init.") as raw:
        root, manifest_sha = _publish(Path(raw))
        (_s3_root(Path(raw)) / "_INIT.json").write_bytes(b"changed")
        _expect_error(
            "s3-init-file-binding",
            lambda: _open(root, manifest_sha),
            "_INIT.json file binding mismatch",
        )


def negative_index_row_and_sha() -> None:
    with tempfile.TemporaryDirectory(prefix="gpic-blob-duplicate-id.") as raw:
        root, manifest_sha = _publish(
            Path(raw), manifest_blob_ids=(0, 0)
        )
        _expect_error(
            "duplicate-manifest-blob-id",
            lambda: _open(root, manifest_sha),
            "canonical projection",
        )

    with tempfile.TemporaryDirectory(prefix="gpic-blob-wrong-worker.") as raw:
        root, manifest_sha = _publish(
            Path(raw),
            manifest_blob_names=(
                "workers/worker-1/blob-00000000.bin",
                "workers/worker-1/blob-00000001.bin",
            ),
        )
        _expect_error(
            "manifest-blob-wrong-worker",
            lambda: _open(root, manifest_sha),
            "canonical projection",
        )

    with tempfile.TemporaryDirectory(prefix="gpic-blob-wrong-key.") as raw:
        root, manifest_sha = _publish(
            Path(raw),
            manifest_blob_names=(
                "workers/worker-0/blob-00000002.bin",
                "workers/worker-1/blob-00000001.bin",
            ),
        )
        _expect_error(
            "manifest-blob-wrong-key",
            lambda: _open(root, manifest_sha),
            "canonical projection",
        )

    with tempfile.TemporaryDirectory(prefix="gpic-blob-rowid.") as raw:
        root, manifest_sha = _publish(Path(raw), row_ids=(999, 101, 102))
        overlay = _open(root, manifest_sha)
        _expect_error(
            "explicit-row-id-mismatch",
            overlay.verify_index_sha256,
            "row identity mismatch",
        )
        overlay.close()

    with tempfile.TemporaryDirectory(prefix="gpic-blob-index-replace.") as raw:
        root, manifest_sha = _publish(Path(raw))
        overlay = _open(root, manifest_sha)
        index_path = root / "dense_index.bin"
        replacement = root / "replacement.bin"
        replacement.write_bytes(index_path.read_bytes())
        os.replace(replacement, index_path)
        # POSIX updates the unlinked old inode's ctime, so the first FD-identity
        # guard already rejects it.  Rebind the test's expected FD metadata to
        # exercise the independent live-path identity guard as well (relevant to
        # filesystems whose replacement does not surface through FD ctime).
        assert overlay._index_fd is not None  # noqa: SLF001
        overlay._index_identity = _identity(  # noqa: SLF001
            os.fstat(overlay._index_fd)  # noqa: SLF001
        )
        _expect_error(
            "mmap-index-path-replacement",
            overlay.verify_index_sha256,
            "live index path was replaced",
        )
        overlay.close()

    with tempfile.TemporaryDirectory(prefix="gpic-blob-full-sentinel.") as raw:
        zero_shas = list(hashlib.sha256(payload).digest() for payload in PAYLOADS)
        zero_shas[0] = b"\x00" * 32
        root, manifest_sha = _publish(
            Path(raw),
            blob_ids=(0xFFFFFFFF, 0, 1),
            segment_shas=tuple(zero_shas),
        )
        overlay = _open(root, manifest_sha)
        _expect_error(
            "full-catalog-sentinel",
            overlay.verify_index_sha256,
            "missing/invalid blob_id",
        )
        overlay.close()

    with tempfile.TemporaryDirectory(prefix="gpic-blob-segment-sha.") as raw:
        bad = list(hashlib.sha256(payload).digest() for payload in PAYLOADS)
        bad[1] = b"\x99" * 32
        root, manifest_sha = _publish(Path(raw), segment_shas=tuple(bad))
        overlay = _open(root, manifest_sha)
        overlay.verify_index_sha256()
        _expect_error(
            "segment-sha-mismatch",
            lambda: overlay.read(101),
            "segment SHA-256 mismatch",
        )
        assert overlay.snapshot()["integrity_failures"] == 1
        overlay.close()

    with tempfile.TemporaryDirectory(prefix="gpic-blob-short-pread.") as raw:
        root, manifest_sha = _publish(Path(raw))
        overlay = _open(root, manifest_sha)
        overlay.verify_index_sha256()
        real_pread = os.pread
        os.pread = lambda _fd, _length, _offset: b""  # type: ignore[assignment]
        try:
            _expect_error(
                "short-pread",
                lambda: overlay.read(100),
                "short GPIC blob pread",
            )
        finally:
            os.pread = real_pread  # type: ignore[assignment]
        metrics = overlay.snapshot()
        assert metrics["short_read_failures"] == 1
        assert metrics["pread_requested_bytes"] == len(PAYLOADS[0])
        assert metrics.get("pread_returned_bytes", 0) == 0
        overlay.close()


def negative_locator_missing_and_exact_set() -> None:
    for unsafe in (
        "/workers/worker-0/blob-00000000.bin",
        "workers//worker-0/blob-00000000.bin",
        "workers/../worker-0/blob-00000000.bin",
        "workers/worker-0/%2e%2e",
        "workers\\worker-0\\blob-00000000.bin",
        "workers/worker-0/bl\N{LATIN SMALL LETTER O WITH DIAERESIS}b-00000000.bin",
    ):
        _expect_error(
            f"unsafe-relative-key-{unsafe!r}",
            lambda unsafe=unsafe: _blob_relative_parts(unsafe, "negative key"),
        )

    with tempfile.TemporaryDirectory(prefix="gpic-blob-truncated-index.") as raw:
        root, manifest_sha = _publish(Path(raw))
        index_path = root / "dense_index.bin"
        index_path.write_bytes(index_path.read_bytes()[:-1])
        _expect_error(
            "truncated-index",
            lambda: _open(root, manifest_sha),
            "live index size",
        )

    with tempfile.TemporaryDirectory(prefix="gpic-blob-range.") as raw:
        root, manifest_sha = _publish(
            Path(raw), locator_lengths=(len(PAYLOADS[0]), 999, len(PAYLOADS[2]))
        )
        overlay = _open(root, manifest_sha)
        overlay.verify_index_sha256()
        _expect_error("locator-out-of-range", lambda: overlay.read(101), "range is invalid")
        overlay.close()

    with tempfile.TemporaryDirectory(prefix="gpic-blob-missing.") as raw:
        root, manifest_sha = _publish(Path(raw))
        (
            _s3_root(Path(raw))
            / "workers/worker-1/blob-00000001.bin"
        ).unlink()
        _expect_error(
            "missing-blob",
            lambda: _open(root, manifest_sha),
            "blob exact-set drift",
        )

    with tempfile.TemporaryDirectory(prefix="gpic-blob-extra-control.") as raw:
        root, manifest_sha = _publish(Path(raw))
        (root / "unexpected.txt").write_text("not immutable", encoding="utf-8")
        _expect_error(
            "control-exact-set",
            lambda: _open(root, manifest_sha),
            "exact-set drift",
        )

    with tempfile.TemporaryDirectory(prefix="gpic-blob-extra-s3-root.") as raw:
        root, manifest_sha = _publish(Path(raw))
        (_s3_root(Path(raw)) / "unexpected.bin").write_bytes(b"unexpected")
        _expect_error(
            "s3-root-exact-set",
            lambda: _open(root, manifest_sha),
            "S3 root exact-set drift",
        )

    with tempfile.TemporaryDirectory(prefix="gpic-blob-symlink-leaf.") as raw:
        root, manifest_sha = _publish(Path(raw))
        overlay = _open(root, manifest_sha)
        overlay.verify_index_sha256()
        leaf = _s3_root(Path(raw)) / "workers/worker-0/blob-00000000.bin"
        outside = Path(raw) / "outside-blob.bin"
        outside.write_bytes(leaf.read_bytes())
        leaf.unlink()
        leaf.symlink_to(outside)
        _expect_error(
            "s3-blob-symlink-leaf",
            lambda: overlay.read(100),
            "not a regular file",
        )
        overlay.close()

    with tempfile.TemporaryDirectory(prefix="gpic-blob-symlink-parent.") as raw:
        root, manifest_sha = _publish(Path(raw))
        overlay = _open(root, manifest_sha)
        overlay.verify_index_sha256()
        worker = _s3_root(Path(raw)) / "workers/worker-0"
        outside_worker = Path(raw) / "outside-worker-0"
        worker.rename(outside_worker)
        worker.symlink_to(outside_worker, target_is_directory=True)
        _expect_error(
            "s3-blob-symlink-parent",
            lambda: overlay.read(100),
            "not a real directory",
        )
        overlay.close()


def negative_catalog_and_token_closure() -> None:
    with tempfile.TemporaryDirectory(prefix="gpic-blob-catalog.") as raw:
        root, manifest_sha = _publish(Path(raw))
        wrong = CatalogBinding(**{**CATALOG.__dict__, "catalog_hash": "4" * 64})
        _expect_error(
            "catalog-binding",
            lambda: GpicBlobOverlay(
                root,
                expected_manifest_sha256=manifest_sha,
                catalog=wrong,
                fd_cache_size=1,
            ),
            "different catalog",
        )

    # Importing the production loader here also proves the new transport module
    # is import-closed with its torch/pyarrow dependencies.
    from training.dataline_offline.offline_dataset import (
        OfflineExactDataset,
        OfflineLoaderError,
        _PlannedItem,
    )

    item = _PlannedItem(0, 0, 100, "gpic", 2, 0)
    try:
        OfflineExactDataset._payload_token_closure(
            [(item, {})], {}, "negative-test"
        )
    except OfflineLoaderError as exc:
        assert "token closure mismatch" in str(exc)
    else:
        raise AssertionError("missing-token closure did not fail")

    class FailingOverlay:
        manifest_sha256 = "a" * 64
        release_sha256 = "b" * 64

        @staticmethod
        def snapshot() -> dict[str, float]:
            return {}

        @staticmethod
        def read(_row_id: int) -> bytes:
            raise GpicBlobError("injected selected-blob failure")

    dataset = object.__new__(OfflineExactDataset)
    dataset._gpic_payload_backend = "offline_gpic_blob_v1"
    dataset._gpic_blob = FailingOverlay()
    dataset._gpic_workers = 1
    dataset.rank = 0
    source_called = {"value": False}

    def forbidden_source(*_args, **_kwargs):
        source_called["value"] = True
        return {}

    dataset._gpic_payloads_in_memory = forbidden_source
    try:
        dataset._image_payloads("gpic", [(item, {})], generation=1)
    except OfflineLoaderError as exc:
        assert "injected selected-blob failure" in str(exc)
    else:
        raise AssertionError("injected selected-blob failure did not propagate")
    assert source_called["value"] is False, "full blob failure silently fell back to source"

    # A peer preflight failure is gathered before any rank can enter the
    # rank-0-only index broadcast.  Likewise, a local post-broadcast accept
    # failure is gathered so healthy peers do not continue into training.
    import torch.distributed as torch_dist

    distributed_dataset = object.__new__(OfflineExactDataset)
    distributed_dataset.rank = 0
    distributed_dataset.world_size = 2
    originals = {
        name: getattr(torch_dist, name)
        for name in (
            "is_available",
            "is_initialized",
            "get_rank",
            "get_world_size",
            "all_gather_object",
            "broadcast_object_list",
        )
    }
    calls = {"gather": 0, "broadcast": 0}

    def fake_gather(output, status):
        calls["gather"] += 1
        output[0] = status
        output[1] = "injected peer preflight failure"

    def fake_broadcast(_status, src):
        assert src == 0
        calls["broadcast"] += 1

    try:
        torch_dist.is_available = lambda: True
        torch_dist.is_initialized = lambda: True
        torch_dist.get_rank = lambda: 0
        torch_dist.get_world_size = lambda: 2
        torch_dist.all_gather_object = fake_gather
        torch_dist.broadcast_object_list = fake_broadcast
        errors = distributed_dataset._collective_gpic_preflight_errors(None)
        assert errors == ["rank1=injected peer preflight failure"]

        class AcceptFailOverlay:
            @staticmethod
            def compute_index_sha256() -> str:
                return "a" * 64

            @staticmethod
            def accept_collective_index_verification(_actual: str) -> None:
                raise GpicBlobError("injected mmap identity replacement")

        distributed_dataset._gpic_blob = AcceptFailOverlay()
        try:
            distributed_dataset._verify_gpic_blob_index()
        except OfflineLoaderError as exc:
            assert "failed collectively" in str(exc)
            assert "injected mmap identity replacement" in str(exc)
        else:
            raise AssertionError("collective index accept failure did not propagate")
        assert calls == {"gather": 2, "broadcast": 1}
    finally:
        for name, value in originals.items():
            setattr(torch_dist, name, value)


def main() -> int:
    assert INDEX_RECORD_BYTES == 56
    positive_and_lru()
    negative_manifest_and_success()
    negative_index_row_and_sha()
    negative_locator_missing_and_exact_set()
    negative_catalog_and_token_closure()
    print(
        "GPIC_BLOB_RUNTIME_TEST_PASS cases=34 record_bytes=56 "
        "dual_marker=1 canonical_projection=1 mmap_fd_hash=1 distributed_failfast=1 "
        "rlimit_fd_gate=1 source_fallback=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
