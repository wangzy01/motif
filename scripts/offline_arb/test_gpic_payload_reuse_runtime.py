#!/usr/bin/env python3
"""Tiny positive/negative tests for the explicit GPIC payload-reuse branch."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from scripts.offline_arb import test_gpic_blob_runtime as legacy
from training.dataline_offline.gpic_blob import (
    INDEX_STRUCT,
    REUSE_PROOF_SCHEMA_VERSION,
    REUSE_SCHEMA_VERSION,
    VERIFY_SCHEMA_VERSION,
    CatalogBinding,
    GpicBlobError,
    GpicBlobOverlay,
    canonical_json_bytes,
    canonical_json_sha256,
    manifest_self_hash,
    report_self_hash,
    reuse_proof_self_hash,
)
from training.dataline_offline.gpic_release_view import (
    parse_production_json,
    project_release,
    project_runtime_manifest,
)


TARGET = CatalogBinding(
    catalog_hash="4" * 64,
    catalog_header_sha256="5" * 64,
    gpic_name="gpic.parquet",
    gpic_row_base=200,
    gpic_rows=3,
    gpic_bytes=4321,
    gpic_sha256="6" * 64,
    gpic_schema=legacy.CATALOG.gpic_schema,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_file_binding(name: str, raw: bytes, self_sha: str) -> dict[str, Any]:
    return {"name": name, "bytes": len(raw), "raw_sha256": sha(raw), "self_sha256": self_sha}


def publish_reuse(parent: Path) -> tuple[Path, str, Path, str]:
    legacy_root, legacy_manifest_sha = legacy._publish(parent)  # noqa: SLF001
    canonical_raw = (legacy_root / "canonical_manifest.json").read_bytes()
    summary_raw = (legacy_root / "verifier_summary.json").read_bytes()
    canonical = parse_production_json(canonical_raw, "tiny source canonical")
    summary = parse_production_json(summary_raw, "tiny source summary")

    # Explicit legacy regression: the unchanged pure projection still emits
    # exactly the six-file fixture's existing manifest and verifier report.
    projected_manifest, projected_report = project_runtime_manifest(
        canonical, summary, legacy_root, summary_raw_bytes=summary_raw
    )
    require(canonical_json_bytes(projected_manifest) == (legacy_root / "manifest.json").read_bytes(), "legacy manifest projection bytes changed")
    require(canonical_json_bytes(projected_report) == (legacy_root / "verifier_report.json").read_bytes(), "legacy verifier projection bytes changed")

    reuse_root = parent / "reuse" / "stage512a-gpic-payload-reuse-v1"
    reuse_root.mkdir(parents=True)
    old_records = (legacy_root / "dense_index.bin").read_bytes()
    records = []
    for local, row_id in enumerate(range(TARGET.gpic_row_base, TARGET.gpic_row_base + TARGET.gpic_rows)):
        old = INDEX_STRUCT.unpack_from(old_records, local * INDEX_STRUCT.size)
        records.append(INDEX_STRUCT.pack(row_id, *old[1:]))
    index_raw = b"".join(records)
    (reuse_root / "dense_index.bin").write_bytes(index_raw)

    source_release = project_release(canonical, Path("/") / legacy.BUILDTAG)
    release = copy.deepcopy(source_release)
    release["catalog"] = {
        "catalog_hash": TARGET.catalog_hash,
        "catalog_header_sha256": TARGET.catalog_header_sha256,
        "gpic_file": {
            "name": TARGET.gpic_name,
            "row_base": TARGET.gpic_row_base,
            "rows": TARGET.gpic_rows,
            "bytes": TARGET.gpic_bytes,
            "sha256": TARGET.gpic_sha256,
            "schema": TARGET.gpic_schema,
        },
    }
    release["storage"]["index_path"] = str(reuse_root / "dense_index.bin")
    release["index"] = {"bytes": len(index_raw), "sha256": sha(index_raw)}
    release["row_count"] = TARGET.gpic_rows
    release_sha = canonical_json_sha256(release)

    proof: dict[str, Any] = {
        "schema_version": REUSE_PROOF_SCHEMA_VERSION,
        "status": "PASS",
        "generated_at": "2026-07-15T00:00:00+00:00",
        "source_canonical_manifest": source_file_binding("source_canonical_manifest.json", canonical_raw, canonical["manifest_sha256"]),
        "source_verifier_summary": source_file_binding("source_verifier_summary.json", summary_raw, summary["verifier_summary_sha256"]),
        "source_catalog": source_release["catalog"],
        "source_dense_index": source_release["index"],
        "source_s3_root": source_release["storage"]["s3_root"],
        "source_blob_set_sha256": source_release["blob_set_sha256"],
        "source_payload_bytes": source_release["payload_bytes"],
        "target_catalog": release["catalog"],
        "target_dense_index": release["index"],
        "target_release_sha256": release_sha,
        "target_row_id_start": TARGET.gpic_row_base,
        "target_row_id_stop_exclusive": TARGET.gpic_row_base + TARGET.gpic_rows,
        "target_row_count": TARGET.gpic_rows,
        "join_key": ["uid", "image_path"],
        "source_rows_scanned": legacy.CATALOG.gpic_rows,
        "target_rows_verified": TARGET.gpic_rows,
        "missing_uid": 0,
        "duplicate_source_uid": 0,
        "duplicate_source_image_path": 0,
        "duplicate_target_uid": 0,
        "duplicate_target_image_path": 0,
        "image_path_mismatches": 0,
        "source_index_row_mismatches": 0,
        "target_index_row_mismatches": 0,
        "locator_mismatches": 0,
        "segment_binding_mismatches": 0,
        "producer_receipt_sha256": "7" * 64,
        "verifier_code_sha256": "8" * 64,
    }
    proof["proof_sha256"] = reuse_proof_self_hash(proof)
    proof_raw = canonical_json_bytes(proof)

    report: dict[str, Any] = {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "status": "PASS",
        "generated_at": "2026-07-15T00:00:00+00:00",
        "release_sha256": release_sha,
        "catalog_hash": TARGET.catalog_hash,
        "index_sha256": sha(index_raw),
        "index_bytes": len(index_raw),
        "row_id_start": TARGET.gpic_row_base,
        "row_id_stop_exclusive": TARGET.gpic_row_base + TARGET.gpic_rows,
        "row_count": TARGET.gpic_rows,
        "blob_count": len(release["blobs"]),
        "payload_bytes": release["payload_bytes"],
        "full_index_records_verified": TARGET.gpic_rows,
        "full_segment_sha256_verified": True,
        "full_blob_sha256_verified": True,
    }
    report["report_sha256"] = report_self_hash(report)
    report_raw = canonical_json_bytes(report)
    (reuse_root / "verifier_report.json").write_bytes(report_raw)
    (reuse_root / "payload_reuse_proof.json").write_bytes(proof_raw)
    (reuse_root / "source_canonical_manifest.json").write_bytes(canonical_raw)
    (reuse_root / "source_verifier_summary.json").write_bytes(summary_raw)

    manifest: dict[str, Any] = {
        "schema_version": REUSE_SCHEMA_VERSION,
        "status": "PASS",
        "generated_at": "2026-07-15T00:00:00+00:00",
        "release": release,
        "release_sha256": release_sha,
        "verification": {"path": str(reuse_root / "verifier_report.json"), "bytes": len(report_raw), "sha256": sha(report_raw), "schema_version": VERIFY_SCHEMA_VERSION},
        "payload_reuse": {"path": str(reuse_root / "payload_reuse_proof.json"), "bytes": len(proof_raw), "sha256": sha(proof_raw), "schema_version": REUSE_PROOF_SCHEMA_VERSION},
    }
    manifest["manifest_sha256"] = manifest_self_hash(manifest)
    manifest_raw = canonical_json_bytes(manifest)
    manifest_raw_sha = sha(manifest_raw)
    (reuse_root / "manifest.json").write_bytes(manifest_raw)
    (reuse_root / "_SUCCESS").write_text(manifest_raw_sha + "\n", encoding="ascii")
    return reuse_root, manifest_raw_sha, legacy_root, legacy_manifest_sha


def expect_error(callback, needle: str) -> None:
    try:
        callback()
    except GpicBlobError as exc:
        require(needle in str(exc), f"wrong failure: {exc}")
    else:
        raise AssertionError("expected GpicBlobError")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="gpic-reuse-runtime.") as raw:
        reuse_root, reuse_sha, legacy_root, legacy_sha = publish_reuse(Path(raw))
        legacy_overlay = GpicBlobOverlay(legacy_root, expected_manifest_sha256=legacy_sha, catalog=legacy.CATALOG, fd_cache_size=1)
        legacy_overlay.verify_index_sha256()
        require([legacy_overlay.read(row) for row in (100, 101, 102)] == list(legacy.PAYLOADS), "legacy payload regression")
        legacy_overlay.close()

        overlay = GpicBlobOverlay(reuse_root, expected_manifest_sha256=reuse_sha, catalog=TARGET, fd_cache_size=1)
        overlay.verify_index_sha256()
        require([overlay.read(row) for row in (200, 201, 202)] == list(legacy.PAYLOADS), "reuse payload mismatch")
        require(overlay.payload_reuse_proof is not None, "reuse proof not retained")
        overlay.close()

    with tempfile.TemporaryDirectory(prefix="gpic-reuse-negative.") as raw:
        reuse_root, reuse_sha, _legacy_root, _legacy_sha = publish_reuse(Path(raw))
        proof_path = reuse_root / "payload_reuse_proof.json"
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        proof["locator_mismatches"] = 1
        proof["proof_sha256"] = reuse_proof_self_hash(proof)
        proof_raw = canonical_json_bytes(proof)
        proof_path.write_bytes(proof_raw)
        manifest_path = reuse_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["payload_reuse"]["bytes"] = len(proof_raw)
        manifest["payload_reuse"]["sha256"] = sha(proof_raw)
        manifest["manifest_sha256"] = manifest_self_hash(manifest)
        manifest_raw = canonical_json_bytes(manifest)
        manifest_path.write_bytes(manifest_raw)
        mutated_sha = sha(manifest_raw)
        (reuse_root / "_SUCCESS").write_text(mutated_sha + "\n", encoding="ascii")
        expect_error(
            lambda: GpicBlobOverlay(reuse_root, expected_manifest_sha256=mutated_sha, catalog=TARGET, fd_cache_size=1),
            "nonzero locator_mismatches",
        )
    print("GPIC_PAYLOAD_REUSE_RUNTIME_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
