#!/usr/bin/env python3
"""Synthetic, no-network red-team gate for the GPIC blob pilot."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.offline_arb import analyze_offline_gpic_blob_pilot as analyzer
from scripts.offline_arb import run_offline_gpic_blob_pilot as runner


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_failure(label: str, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except Exception:
        return
    raise AssertionError(f"negative case unexpectedly passed: {label}")


def create_catalog(root: Path, rows: int = 64) -> tuple[Path, str]:
    source_root = root / "source" / "train"
    source_root.mkdir(parents=True)
    paths: list[str] = []
    row_ids = list(range(1000, 1000 + rows))
    for index in range(rows):
        shard = f"{(index + index // 8) % 8:05d}"
        extension = ".jpg" if (index // 8) % 2 == 0 else ".png"
        path = source_root / shard / f"{index:064x}{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (f"gpic-pilot-{index}-月".encode("utf-8") + bytes([index % 251])) * (5 + index % 7)
        path.write_bytes(payload)
        paths.append(str(path))
    schema = pa.schema(
        [
            pa.field("row_id", pa.int64(), nullable=False),
            pa.field("image_path", pa.string(), nullable=False),
        ]
    )
    table = pa.Table.from_arrays(
        [pa.array(row_ids, type=pa.int64()), pa.array(paths, type=pa.string())],
        schema=schema,
    )
    catalog = root / "catalog"
    catalog.mkdir()
    parquet_path = catalog / "gpic.parquet"
    pq.write_table(table, parquet_path, row_group_size=11, compression="zstd")
    parquet_raw = parquet_path.read_bytes()
    header = {
        "schema_version": runner.CATALOG_SCHEMA_VERSION,
        "domain_order": ["prelude", "gpic"],
        "row_count": 1000 + rows,
        "catalog_hash": "1" * 64,
        "files": {
            "prelude": {"rows": 1000},
            "gpic": {
                "name": parquet_path.name,
                "rows": rows,
                "bytes": len(parquet_raw),
                "sha256": hashlib.sha256(parquet_raw).hexdigest(),
                "schema": str(schema),
            }
        },
    }
    (catalog / "header.json").write_text(
        json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return catalog, str(source_root) + "/"


def build_args(
    catalog: Path,
    job_spec: Path,
    prefix: Path,
    output: Path,
    backend: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=output,
        allowed_output_prefix=prefix,
        backend=backend,
        write_authority=runner.WRITE_AUTHORITY,
        job_spec=job_spec,
        catalog_root=catalog,
        sample_rows=64,
        coverage_intervals=8,
        min_unique_shards=8,
        threads=4,
        source_attempts=2,
        source_block_bytes=31,
        source_order="path",
        retry_base_seconds=0.001,
        blob_target_bytes=1024,
        max_total_source_bytes=(1 << 20) if backend == "mountpoint_s3" else 0,
        stripe_count=0,
        stripe_size_mib=64,
    )


def replay_args(
    root: Path,
    output: Path,
    job_spec: Path,
    *,
    selection_mode: str,
    arm_role: str,
    arm_sequence_index: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        root=root,
        output=output,
        workers=4,
        repeats=1,
        seed=1729,
        timeout_seconds=120,
        selection_mode=selection_mode,
        arm_role=arm_role,
        arm_sequence_index=arm_sequence_index,
        min_cold_regions=1,
        job_spec=job_spec,
    )


def reseal(root: Path, manifest: dict) -> None:
    manifest["manifest_sha256"] = runner.manifest_self_hash(manifest)
    (root / "manifest.json").write_bytes(runner.canonical_json_bytes(manifest))
    (root / "_SUCCESS").write_text(manifest["manifest_sha256"] + "\n", encoding="ascii")


def run(root: Path, job_spec: Path) -> None:
    require(str(root).startswith("/fsx/"), "static root must live on FSx")
    require(not root.exists(), "static namespace must be fresh")
    root.mkdir(parents=True)
    build_spec = root / "synthetic_build_job.yaml"
    replay_spec = root / "synthetic_replay_job.yaml"
    build_spec.write_bytes(job_spec.read_bytes() + b"\n# synthetic-build-binding\n")
    replay_spec.write_bytes(job_spec.read_bytes() + b"\n# synthetic-replay-binding\n")
    catalog, synthetic_prefix = create_catalog(root)
    original_source_prefix = runner.SOURCE_PREFIX
    runner.SOURCE_PREFIX = synthetic_prefix
    try:
        binding = runner.load_catalog(catalog, verify_gpic_sha256=True)
        positions = runner.stratified_positions(binding.gpic_rows, 17)
        require(len(positions) == 17 and positions == sorted(set(positions)), "selection positions drift")
        items, summary = runner.select_catalog_items(
            binding, sample_rows=64, coverage_intervals=8, min_unique_shards=8
        )
        require(len(items) == 64, "synthetic selection row count drift")
        require(summary["unique_shards"] == 8, "synthetic shard coverage drift")
        require(set(summary["extension_counts"]) == {".jpg", ".png"}, "format coverage drift")

        prefix = root / "artifacts"
        prefix.mkdir()
        fsx_root = prefix / "fsx" / "layout"
        original_rename = runner.rename_noreplace

        def force_lustre_einval(*_args, **_kwargs):
            raise OSError(errno.EINVAL, "synthetic Lustre RENAME_NOREPLACE rejection")

        runner.rename_noreplace = force_lustre_einval
        try:
            fsx_manifest = runner.build_pilot(
                build_args(catalog, build_spec, prefix, fsx_root, "fsx_posix")
            )
        finally:
            runner.rename_noreplace = original_rename
        require(fsx_root.is_dir() and (fsx_root / "_SUCCESS").is_file(), "FSx publish missing")
        require(not fsx_root.is_symlink(), "FSx fallback published a symlink")
        require(
            fsx_manifest["publication_semantics"]
            == "posix_renameat2_noreplace_or_mkdir_claim_hardlink_files_success_last",
            "FSx fallback publication contract drift",
        )
        require(
            runner.is_sha256(fsx_manifest["builder_runtime"].get("boot_id_sha256"))
            and fsx_manifest["builder_runtime"].get("boot_id_raw_published") is False
            and "boot_id" not in fsx_manifest["builder_runtime"],
            "builder hashed boot-id contract drift",
        )

        file_fallback_root = root / "file_fallback"
        file_fallback_root.mkdir()
        file_output = file_fallback_root / "evidence.json"
        runner.rename_noreplace = force_lustre_einval
        try:
            runner.atomic_write_bytes(file_output, b"hardlink-fallback\n")
        finally:
            runner.rename_noreplace = original_rename
        require(
            file_output.read_bytes() == b"hardlink-fallback\n",
            "file hardlink no-replace fallback drift",
        )

        claim_root = root / "claim_race"
        claim_root.mkdir()
        claim_source = claim_root / "active"
        claim_destination = claim_root / "published"
        claim_source.mkdir()
        (claim_source / "payload.bin").write_bytes(b"ours")
        (claim_source / "_SUCCESS").write_text("ours\n", encoding="ascii")

        def concurrent_destination(destination: Path) -> None:
            destination.mkdir()
            (destination / "competitor.bin").write_bytes(b"competitor")

        runner.rename_noreplace = force_lustre_einval
        runner._FALLBACK_BEFORE_DIRECTORY_CLAIM_HOOK = concurrent_destination
        try:
            expect_failure(
                "concurrent destination claim",
                runner.publish_directory_noreplace,
                claim_source,
                claim_destination,
                success_last=True,
            )
        finally:
            runner._FALLBACK_BEFORE_DIRECTORY_CLAIM_HOOK = None
            runner.rename_noreplace = original_rename
        require(
            (claim_destination / "competitor.bin").read_bytes() == b"competitor"
            and not (claim_destination / "_SUCCESS").exists()
            and (claim_source / "payload.bin").read_bytes() == b"ours",
            "concurrent destination was overwritten or source was consumed",
        )

        # Once the fallback links _SUCCESS, even a post-commit durability
        # failure must leave a monotonic published state and must not let a
        # downstream wrapper write its success receipt.
        committed_output = prefix / "committed_sync" / "layout"
        committed_receipt = root / "committed_sync_receipt_must_not_exist.json"

        def fail_after_success_link(_destination: Path, method: str) -> None:
            require(
                method == "claim_directory_hardlink_files_success_last",
                "post-commit fallback hook observed wrong method",
            )
            raise OSError(errno.EIO, "synthetic post-link durability failure")

        committed_sync_caught = False
        runner.rename_noreplace = force_lustre_einval
        runner._POST_COMMIT_DURABILITY_HOOK = fail_after_success_link
        try:
            try:
                committed_manifest = runner.build_pilot(
                    build_args(
                        catalog,
                        build_spec,
                        prefix,
                        committed_output,
                        "fsx_posix",
                    )
                )
                committed_receipt.write_bytes(
                    runner.canonical_json_bytes(committed_manifest)
                )
            except runner.CommittedSyncUnconfirmed:
                committed_sync_caught = True
        finally:
            runner._POST_COMMIT_DURABILITY_HOOK = None
            runner.rename_noreplace = original_rename
        committed_work_roots = list(
            committed_output.parent.glob(f".{committed_output.name}.work.*")
        )
        require(committed_sync_caught, "post-link failure lost dedicated committed state")
        require(
            (committed_output / "_SUCCESS").is_file()
            and not (committed_output / "_PUBLISH_FAILED").exists()
            and not (committed_output / "_FAILED.json").exists()
            and not list(
                committed_output.parent.glob(f"{committed_output.name}.FAILED.*")
            )
            and not committed_receipt.exists()
            and len(committed_work_roots) == 1
            and not (committed_work_roots[0] / "_PUBLISH_FAILED").exists()
            and not (committed_work_roots[0] / "_FAILED.json").exists(),
            "post-link committed state was retracted or contradicted",
        )

        # Model the ambiguous syscall boundary directly: link(2) creates the
        # marker name, then the wrapper reports EIO.  The exception path must
        # observe the now-visible marker and must not add _PUBLISH_FAILED.
        ambiguous_root = root / "link_created_then_error"
        ambiguous_root.mkdir()
        ambiguous_source = ambiguous_root / "active"
        ambiguous_destination = ambiguous_root / "published"
        ambiguous_source.mkdir()
        (ambiguous_source / "payload.bin").write_bytes(b"ambiguous-link")
        (ambiguous_source / "_SUCCESS").write_text("linked\n", encoding="ascii")
        original_os_link = runner.os.link
        marker_link_errors = 0

        def link_then_report_eio(*args, **kwargs):
            nonlocal marker_link_errors
            result = original_os_link(*args, **kwargs)
            if Path(args[1]).name == "_SUCCESS":
                marker_link_errors += 1
                raise OSError(errno.EIO, "synthetic EIO after successful marker link")
            return result

        ambiguous_sync_caught = False
        runner.rename_noreplace = force_lustre_einval
        runner.os.link = link_then_report_eio
        try:
            try:
                runner.publish_directory_noreplace(
                    ambiguous_source,
                    ambiguous_destination,
                    success_last=True,
                )
            except runner.CommittedSyncUnconfirmed:
                ambiguous_sync_caught = True
        finally:
            runner.os.link = original_os_link
            runner.rename_noreplace = original_rename
        require(
            ambiguous_sync_caught
            and marker_link_errors == 1
            and (ambiguous_destination / "_SUCCESS").is_file()
            and not (ambiguous_destination / "_PUBLISH_FAILED").exists(),
            "link-created-then-EIO path contradicted its visible marker",
        )

        # Prove that the rename fast path fsyncs the destination parent after
        # rename and classifies a failure there as committed/unconfirmed.
        fast_root = root / "rename_parent_sync"
        fast_root.mkdir()
        fast_source = fast_root / "active"
        fast_destination = fast_root / "published"
        fast_source.mkdir()
        (fast_source / "payload.bin").write_bytes(b"rename-fast-path")
        (fast_source / "_SUCCESS").write_text("fast\n", encoding="ascii")
        original_fsync_dir = runner.fsync_dir
        rename_calls = 0
        parent_postcommit_failures = 0

        def force_rename_success(source: Path, destination: Path) -> None:
            nonlocal rename_calls
            rename_calls += 1
            source.rename(destination)

        def fail_renamed_parent_sync(path: Path) -> None:
            nonlocal parent_postcommit_failures
            path = Path(path)
            if path == fast_root and fast_destination.exists():
                parent_postcommit_failures += 1
                raise OSError(errno.EIO, "synthetic renamed-parent fsync failure")
            original_fsync_dir(path)

        fast_sync_caught = False
        runner.rename_noreplace = force_rename_success
        runner.fsync_dir = fail_renamed_parent_sync
        try:
            try:
                runner.publish_directory_noreplace(
                    fast_source,
                    fast_destination,
                    success_last=True,
                )
            except runner.CommittedSyncUnconfirmed:
                fast_sync_caught = True
        finally:
            runner.fsync_dir = original_fsync_dir
            runner.rename_noreplace = original_rename
        require(
            fast_sync_caught
            and rename_calls == 1
            and parent_postcommit_failures == 1
            and not fast_source.exists()
            and (fast_destination / "_SUCCESS").is_file()
            and not (fast_destination / "_PUBLISH_FAILED").exists(),
            "rename parent-sync failure did not preserve committed marker state",
        )

        # A failure-evidence root may inherit an active source tree containing
        # _SUCCESS.  success_last=False must bypass even an available rename
        # fast path and filter that marker from the claimed destination.
        failed_root_case = root / "failed_root_no_success"
        failed_root_case.mkdir()
        failed_source = failed_root_case / "active"
        failed_destination = failed_root_case / "published_failed"
        failed_source.mkdir()
        (failed_source / "payload.bin").write_bytes(b"failed-payload")
        (failed_source / "_FAILED.json").write_text("{}\n", encoding="ascii")
        (failed_source / "_SUCCESS").write_text("must-filter\n", encoding="ascii")
        forbidden_failed_root_rename_calls = 0

        def available_but_forbidden_failed_root_rename(
            source: Path, destination: Path
        ) -> None:
            nonlocal forbidden_failed_root_rename_calls
            forbidden_failed_root_rename_calls += 1
            source.rename(destination)

        runner.rename_noreplace = available_but_forbidden_failed_root_rename
        try:
            failed_method = runner.publish_directory_noreplace(
                failed_source,
                failed_destination,
                success_last=False,
            )
        finally:
            runner.rename_noreplace = original_rename
        require(
            forbidden_failed_root_rename_calls == 0
            and failed_method
            == "claim_directory_hardlink_files_failed_root_no_success"
            and {path.name for path in failed_destination.iterdir()}
            == {"payload.bin", "_FAILED.json"}
            and not (failed_destination / "_SUCCESS").exists(),
            "failed-root publication used rename or leaked _SUCCESS",
        )

        # Exercise the object-store publication branch on a synthetic FSx path.
        # Any rename or directory fsync in this branch is a static-gate failure.
        s3_root = prefix / "s3_sim" / "layout"
        original_authority = runner.validate_output_authority
        original_publish = runner.publish_directory_noreplace
        original_fsync_dir = runner.fsync_dir

        def simulated_authority(output_root, allowed_output_prefix, backend, write_authority):
            require(backend == "mountpoint_s3", "wrong simulated backend")
            require(write_authority == runner.WRITE_AUTHORITY, "write token drift")
            return Path(output_root), Path(allowed_output_prefix)

        def forbidden_publish(*_args, **_kwargs):
            raise AssertionError("Mountpoint-S3 branch attempted POSIX directory publication")

        def forbidden_dir_fsync(*_args, **_kwargs):
            raise AssertionError("Mountpoint-S3 branch attempted directory fsync")

        runner.validate_output_authority = simulated_authority
        runner.publish_directory_noreplace = forbidden_publish
        runner.fsync_dir = forbidden_dir_fsync
        try:
            s3_manifest = runner.build_pilot(
                build_args(catalog, build_spec, prefix, s3_root, "mountpoint_s3")
            )
        finally:
            runner.validate_output_authority = original_authority
            runner.publish_directory_noreplace = original_publish
            runner.fsync_dir = original_fsync_dir
        require(s3_root.is_dir() and (s3_root / "_SUCCESS").is_file(), "S3-sim publish missing")
        require(
            analyzer.file_identity_vector(fsx_manifest) == analyzer.file_identity_vector(s3_manifest),
            "FSx/S3-sim exact payload drift",
        )

        reports = root / "reports"
        reports.mkdir()
        fsx_replay = reports / "fsx_replay.json"
        s3_replay = reports / "s3_replay.json"
        fsx_integrity = reports / "fsx_integrity.json"
        s3_integrity = reports / "s3_integrity.json"
        runner.replay_pilot(
            replay_args(
                fsx_root,
                fsx_replay,
                replay_spec,
                selection_mode="cold_unique_region",
                arm_role="capacity_authority",
                arm_sequence_index=0,
            )
        )

        runner.replay_pilot(
            replay_args(
                s3_root,
                s3_replay,
                replay_spec,
                selection_mode="cold_unique_region",
                arm_role="capacity_authority",
                arm_sequence_index=0,
            )
        )

        fsx_capacity_value = json.loads(fsx_replay.read_text(encoding="utf-8"))
        cold = fsx_capacity_value["replay_selection"]
        require(
            cold["region_bytes"] == 16 * (1 << 20)
            and cold["potential_readahead_guard_bytes"] == 8 * (1 << 20)
            and cold["get_start_alignment_bytes"] == 8 * (1 << 20)
            and cold["selected_region_count"] == cold["selected_segments"],
            "cold-region replay contract drift",
        )
        fsx_live_manifest, _ = runner.load_published_manifest(fsx_root)
        fsx_entries = runner.validate_artifact_files(
            fsx_root, fsx_live_manifest, verify_blob_sha256=False
        )
        selected_entries, _ = runner.select_replay_entries(
            fsx_entries,
            fsx_live_manifest,
            mode="cold_unique_region",
            seed=1729,
            min_cold_regions=1,
        )
        seen_regions = set()
        for entry in selected_entries:
            region_start = (entry.offset // (16 * (1 << 20))) * (16 * (1 << 20))
            region_stop = region_start + 16 * (1 << 20)
            aligned_start = (entry.offset // (8 * (1 << 20))) * (8 * (1 << 20))
            require((entry.blob_id, region_start) not in seen_regions, "duplicate cold region")
            seen_regions.add((entry.blob_id, region_start))
            require(
                entry.offset + 8 * (1 << 20) + entry.length <= region_stop
                and aligned_start >= region_start
                and aligned_start + 8 * (1 << 20) + entry.length <= region_stop,
                "cold-region aligned potential GET crosses 16MiB boundary",
            )
        runner.replay_pilot(
            replay_args(
                fsx_root,
                fsx_integrity,
                replay_spec,
                selection_mode="full_coverage",
                arm_role="integrity_warm",
                arm_sequence_index=1,
            )
        )
        runner.replay_pilot(
            replay_args(
                s3_root,
                s3_integrity,
                replay_spec,
                selection_mode="full_coverage",
                arm_role="integrity_warm",
                arm_sequence_index=1,
            )
        )
        fsx_integrity_value = json.loads(fsx_integrity.read_text(encoding="utf-8"))
        expect_failure(
            "duplicate capacity authorities",
            analyzer.summarize_replays,
            [fsx_capacity_value, fsx_capacity_value, fsx_integrity_value],
        )
        warm_before_cold = json.loads(json.dumps(fsx_integrity_value))
        warm_before_cold["generated_at"] = "2000-01-01T00:00:00+00:00"
        expect_failure(
            "warm integrity before cold authority",
            analyzer.summarize_replays,
            [fsx_capacity_value, warm_before_cold],
        )

        scan_output = reports / "source_scan.json"
        runner.source_scan(
            SimpleNamespace(
                job_spec=build_spec,
                catalog_root=catalog,
                sample_rows=8,
                coverage_intervals=8,
                min_unique_shards=4,
                threads_matrix="1,2",
                abba=True,
                exclude_sample_rows=0,
                max_byte_imbalance=2.0,
                source_attempts=2,
                source_block_bytes=31,
                source_order="path",
                retry_base_seconds=0.001,
                output=scan_output,
            )
        )
        fake_replay_path = reports / "same_boot_fake_replay.json"
        fake_replay = json.loads(fsx_replay.read_text(encoding="utf-8"))
        fake_replay["replay_runtime"]["hostname"] = "synthetic-distinct-pod"
        fake_replay.pop("report_sha256", None)
        fake_replay["report_sha256"] = runner.canonical_json_sha256(fake_replay)
        fake_replay_path.write_bytes(runner.canonical_json_bytes(fake_replay))
        expect_failure(
            "same physical cache host",
            analyzer.analyze,
            SimpleNamespace(
                source_scan=scan_output,
                artifact=[("fsx", fsx_root)],
                replay=[
                    ("fsx_fake", fake_replay_path),
                    ("fsx_integrity", fsx_integrity),
                ],
                exact_pair=[],
                require_distinct_runtime_hosts=True,
                enforce_pilot_shape=False,
                runner=Path(runner.__file__),
                build_job_spec=build_spec,
                s3_build_job_spec=build_spec,
                replay_job_spec=replay_spec,
                build_receipt=None,
                s3_build_receipt=None,
                job_spec=job_spec,
                output=reports / "same_boot_must_not_publish.json",
            ),
        )
        bad_runner_replay_path = reports / "bad_runner_replay.json"
        bad_runner_replay = json.loads(fsx_replay.read_text(encoding="utf-8"))
        bad_runner_replay["code"]["runner_sha256"] = "e" * 64
        bad_runner_replay.pop("report_sha256", None)
        bad_runner_replay["report_sha256"] = runner.canonical_json_sha256(
            bad_runner_replay
        )
        bad_runner_replay_path.write_bytes(
            runner.canonical_json_bytes(bad_runner_replay)
        )
        expect_failure(
            "replay/live runner SHA mismatch",
            analyzer.analyze,
            SimpleNamespace(
                source_scan=scan_output,
                artifact=[("fsx", fsx_root)],
                replay=[
                    ("fsx_capacity", bad_runner_replay_path),
                    ("fsx_integrity", fsx_integrity),
                ],
                exact_pair=[],
                require_distinct_runtime_hosts=False,
                enforce_pilot_shape=False,
                runner=Path(runner.__file__),
                build_job_spec=build_spec,
                s3_build_job_spec=None,
                replay_job_spec=replay_spec,
                build_receipt=None,
                s3_build_receipt=None,
                job_spec=job_spec,
                output=reports / "bad_runner_must_not_publish.json",
            ),
        )
        bad_receipt_path = reports / "bad_receipt.json"
        bad_receipt_path.write_text(
            json.dumps(
                {
                    "schema_version": "offline_gpic_blob_build_receipt_v1",
                    "receipt_sha256": "0" * 64,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_failure(
            "receipt self hash",
            analyzer.analyze,
            SimpleNamespace(
                source_scan=scan_output,
                artifact=[("fsx", fsx_root)],
                replay=[
                    ("fsx_capacity", fsx_replay),
                    ("fsx_integrity", fsx_integrity),
                ],
                exact_pair=[],
                require_distinct_runtime_hosts=False,
                enforce_pilot_shape=False,
                runner=Path(runner.__file__),
                build_job_spec=build_spec,
                s3_build_job_spec=None,
                replay_job_spec=replay_spec,
                build_receipt=bad_receipt_path,
                s3_build_receipt=None,
                job_spec=job_spec,
                output=reports / "bad_receipt_must_not_publish.json",
            ),
        )
        analysis_output = reports / "analysis.json"
        analysis = analyzer.analyze(
            SimpleNamespace(
                source_scan=scan_output,
                artifact=[("fsx", fsx_root), ("s3", s3_root)],
                replay=[
                    ("fsx_capacity", fsx_replay),
                    ("fsx_integrity", fsx_integrity),
                    ("s3_capacity", s3_replay),
                    ("s3_integrity", s3_integrity),
                ],
                exact_pair=[("fsx", "s3")],
                require_distinct_runtime_hosts=False,
                enforce_pilot_shape=False,
                runner=Path(runner.__file__),
                build_job_spec=build_spec,
                s3_build_job_spec=build_spec,
                replay_job_spec=replay_spec,
                build_receipt=None,
                s3_build_receipt=None,
                job_spec=job_spec,
                output=analysis_output,
            )
        )
        require(
            analysis["status"] == "INCONCLUSIVE_FOR_PRODUCTION",
            "20k-or-smaller S3 analysis was not kept inconclusive",
        )
        require(
            all(
                value["replay"]["capacity_arms_averaged"] is False
                and value["replay"]["warm_integrity"]["excluded_from_capacity_summary"]
                is True
                for value in analysis["artifacts"].values()
            ),
            "warm integrity leaked into capacity authority",
        )
        require(
            analysis["artifacts"]["s3"]["replay"]["capacity_authority"]
            ["production_capacity_status"]
            == "INCONCLUSIVE_FOR_PRODUCTION",
            "small S3 capacity was overqualified",
        )
        require(analysis["production_selection_qualified"] is False, "pilot was overqualified")

        expect_failure(
            "no clobber",
            runner.build_pilot,
            build_args(catalog, build_spec, prefix, fsx_root, "fsx_posix"),
        )
        outside = root / "outside"
        expect_failure(
            "output authority",
            runner.validate_output_authority,
            outside,
            prefix,
            "fsx_posix",
            runner.WRITE_AUTHORITY,
        )

        corrupt_blob = root / "corrupt_blob"
        shutil.copytree(fsx_root, corrupt_blob)
        corrupt_manifest, _ = runner.load_published_manifest(corrupt_blob)
        blob_path = corrupt_blob / corrupt_manifest["files"]["blobs"][0]["name"]
        raw = bytearray(blob_path.read_bytes())
        raw[len(raw) // 2] ^= 1
        blob_path.write_bytes(raw)
        expect_failure(
            "blob sha corruption",
            runner.validate_artifact_files,
            corrupt_blob,
            corrupt_manifest,
        )

        bad_offset = root / "bad_offset"
        shutil.copytree(fsx_root, bad_offset)
        bad_manifest, _ = runner.load_published_manifest(bad_offset)
        index_path = bad_offset / bad_manifest["files"]["index"]["name"]
        index_raw = bytearray(index_path.read_bytes())
        row_id, blob_id, offset, length, checksum = runner.INDEX_STRUCT.unpack_from(index_raw, 0)
        runner.INDEX_STRUCT.pack_into(index_raw, 0, row_id, blob_id, offset + 1, length, checksum)
        index_path.write_bytes(index_raw)
        bad_manifest["files"]["index"]["sha256"] = hashlib.sha256(index_raw).hexdigest()
        bad_manifest["files"]["index"]["bytes"] = len(index_raw)
        reseal(bad_offset, bad_manifest)
        reloaded, _ = runner.load_published_manifest(bad_offset)
        expect_failure(
            "index overlap/gap",
            runner.validate_artifact_files,
            bad_offset,
            reloaded,
        )

        bad_success = root / "bad_success"
        shutil.copytree(fsx_root, bad_success)
        (bad_success / "_SUCCESS").write_text("0" * 64 + "\n", encoding="ascii")
        expect_failure("success mismatch", runner.load_published_manifest, bad_success)

        require(
            "mountpoint_s3_write_once_final_keys_file_fsync_success_last_no_rename_no_dir_fsync"
            == s3_manifest["publication_semantics"],
            "S3 publication contract label drift",
        )
    finally:
        runner.SOURCE_PREFIX = original_source_prefix
    print("OFFLINE_GPIC_BLOB_PILOT_STATIC_PASS cases=28", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-spec", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.root, arguments.job_spec)
