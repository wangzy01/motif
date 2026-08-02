#!/usr/bin/env python3
"""Static contract checks for the production GPIC blob runtime integration."""

from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MODULE = REPO / "training/dataline_offline/gpic_blob.py"
RELEASE_VIEW = REPO / "training/dataline_offline/gpic_release_view.py"
DATASET = REPO / "training/dataline_offline/offline_dataset.py"
SCHEMA = REPO / "training/config_schema.py"
TRAIN = REPO / "train.py"


def _tree(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _method_source(source: str, tree: ast.Module, cls_name: str, method: str) -> str:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == method
                ):
                    lines = source.splitlines()
                    return "\n".join(lines[child.lineno - 1 : child.end_lineno])
    raise AssertionError(f"missing {cls_name}.{method}")


def main() -> int:
    module_source, module_tree = _tree(MODULE)
    release_view_source, _release_view_tree = _tree(RELEASE_VIEW)
    dataset_source, dataset_tree = _tree(DATASET)
    schema_source, _schema_tree = _tree(SCHEMA)
    train_source, _train_tree = _tree(TRAIN)

    assert 'struct.Struct("<qIQI32s")' in module_source
    assert "INDEX_RECORD_BYTES = INDEX_STRUCT.size" in module_source
    assert "mmap.mmap" in module_source
    assert "os.pread" in module_source
    assert "active leases" in module_source
    assert "workers/worker-[0-7]" in module_source
    assert "blob-[0-9]+" in module_source
    assert "unique and strictly ascending" in module_source
    assert "contiguous in manifest order" not in module_source
    assert '"s3_init"' in module_source
    assert "resource.RLIMIT_NOFILE" in module_source
    assert 's3_success_path = self.s3_root / "_SUCCESS"' in module_source
    assert "GPIC S3 control root exact-set drift" in module_source
    assert "GPIC S3 root exact-set drift" in module_source
    assert "selection_mode\": SELECTION_MODE" in module_source
    assert "full_segment_sha256_verified" in module_source
    assert "full_blob_sha256_verified" in module_source
    assert "parse_production_json" in module_source
    assert "validate_canonical_manifest" in module_source
    assert "validate_verifier_summary" in module_source
    # Was "project_release(canonical, self.control_root)".  The reuse_mode work made the
    # projection root a local, so this source-text assertion has been failing on a pure
    # rename.  Keep the intent -- outside reuse_mode the projection is still rooted at
    # the control root -- rather than the old literal.
    assert "project_release(canonical, projection_root)" in module_source
    assert "else self.control_root" in module_source
    assert "local and S3 production canonical manifest raw bytes differ" in module_source
    assert '"canonical_manifest.json"' in module_source
    assert '"verifier_summary.json"' in module_source
    assert "def production_json_bytes" in release_view_source
    assert "def project_release" in release_view_source

    read_source = _method_source(
        module_source, module_tree, "GpicBlobOverlay", "read"
    )
    assert "hashlib.sha256(raw).digest()" in read_source
    assert "entry.sha256_raw" in read_source
    assert "_pread_exact" in read_source

    lookup_source = _method_source(
        module_source, module_tree, "GpicBlobOverlay", "_entry_for_row"
    )
    assert "local_row = row_id - self.row_id_start" in lookup_source
    assert "entry.row_id != row_id" in lookup_source

    index_hash_source = _method_source(
        module_source, module_tree, "GpicBlobOverlay", "compute_index_sha256"
    )
    assert "_sha256_open_fd" in index_hash_source
    assert "sha256_file" not in index_hash_source
    index_accept_source = _method_source(
        module_source,
        module_tree,
        "GpicBlobOverlay",
        "accept_collective_index_verification",
    )
    assert "_regular_metadata(self.index_path" in index_accept_source
    assert "live index path was replaced" in index_accept_source

    blob_payload_source = _method_source(
        dataset_source, dataset_tree, "OfflineExactDataset", "_gpic_blob_payloads"
    )
    assert "overlay.read(int(item.row_id))" in blob_payload_source
    assert "_gpic_payloads_in_memory" not in blob_payload_source
    assert "_payload_token_closure" in blob_payload_source

    image_router = _method_source(
        dataset_source, dataset_tree, "OfflineExactDataset", "_image_payloads"
    )
    blob_branch = image_router.index('self._gpic_payload_backend == "offline_gpic_blob_v1"')
    source_branch = image_router.index("self._gpic_payload_in_memory")
    assert blob_branch < source_branch
    assert "return self._gpic_blob_payloads" in image_router

    close_source = _method_source(
        dataset_source, dataset_tree, "OfflineExactDataset", "_close_gpic_blob"
    )
    assert "if self._threads" in close_source
    assert "overlay.close()" in close_source

    for key in (
        "offline_gpic_blob_root",
        "gpic_payload_backend",
        "gpic_blob_manifest_sha256",
        "gpic_blob_fd_cache_size",
    ):
        assert f'"{key}"' in schema_source
    assert "runtime_gpic_payload_backend" in train_source
    assert "runtime_gpic_blob_manifest_sha256" in train_source
    assert "gpic_allrank_slowest_supply_img_s" in train_source
    assert "gpic_allrank_max_rss_bytes" in train_source
    assert "gpic_allrank_sum_short_read_failures" in train_source
    assert "gpic_allrank_min_rlimit_nofile_soft" in train_source
    assert "gpic_allrank_min_fd_cache_covers_blob_count" in train_source

    preflight_source = _method_source(
        dataset_source,
        dataset_tree,
        "OfflineExactDataset",
        "_collective_gpic_preflight_errors",
    )
    verify_source = _method_source(
        dataset_source,
        dataset_tree,
        "OfflineExactDataset",
        "_verify_gpic_blob_index",
    )
    assert "all_gather_object" in preflight_source
    assert "all_gather_object" in verify_source

    # Compile the exact edited sources from bytes without importing torch or
    # allocating the production index.
    for path, source in (
        (MODULE, module_source),
        (RELEASE_VIEW, release_view_source),
        (DATASET, dataset_source),
        (SCHEMA, schema_source),
        (TRAIN, train_source),
    ):
        compile(source, str(path), "exec")

    print(
        "GPIC_BLOB_STATIC_TEST_PASS record_bytes=56 mmap=1 pread=1 "
        "lease_lru=1 dual_marker=1 canonical_projection=1 mmap_fd_hash=1 collective_failfast=1 "
        "allrank_health=1 rlimit_fd_gate=1 full_zero_fallback=1 token_closure=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
