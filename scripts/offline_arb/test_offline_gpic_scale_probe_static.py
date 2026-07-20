#!/usr/bin/env python3
"""Synthetic/no-network and negative gate for the GPIC scale probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from scripts.offline_arb import analyze_offline_gpic_scale_probe as analyzer
from scripts.offline_arb import run_offline_gpic_blob_pilot as pilot_core
from scripts.offline_arb import run_offline_gpic_scale_probe as probe


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_failure(
    case_label: str, function: Any, *args: Any, **kwargs: Any
) -> None:
    try:
        function(*args, **kwargs)
    except Exception:
        return
    raise AssertionError(f"negative unexpectedly passed: {case_label}")


def create_catalog(root: Path, *, rows: int, imbalanced: bool) -> tuple[Path, str]:
    source = root / "source"
    source.mkdir(parents=True)
    paths: list[str] = []
    row_base = 101
    for local in range(rows):
        shard = source / f"{local % 31:05d}"
        shard.mkdir(exist_ok=True)
        extension = ".jpg" if (local // 8) % 2 == 0 else ".png"
        path = shard / f"{local:064x}{extension}"
        multiplier = 2 if imbalanced and local % 8 == 7 else 1
        payload = hashlib.sha256(f"scale-static-{local}".encode()).digest() * (128 * multiplier)
        path.write_bytes(payload)
        paths.append(str(path))
    schema = pa.schema(
        [
            pa.field("row_id", pa.int64(), nullable=False),
            pa.field("image_path", pa.string(), nullable=False),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array(range(row_base, row_base + rows), type=pa.int64()),
            pa.array(paths, type=pa.string()),
        ],
        schema=schema,
    )
    catalog = root / "catalog"
    catalog.mkdir()
    parquet = catalog / "gpic.parquet"
    pq.write_table(table, parquet, row_group_size=37, compression="zstd")
    raw = parquet.read_bytes()
    header = {
        "schema_version": pilot_core.CATALOG_SCHEMA_VERSION,
        "domain_order": ["prelude", "gpic"],
        "row_count": row_base + rows,
        "catalog_hash": hashlib.sha256(b"synthetic-scale-catalog" + bytes([imbalanced])).hexdigest(),
        "files": {
            "prelude": {"rows": row_base},
            "gpic": {
                "name": parquet.name,
                "rows": rows,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "schema": str(schema),
            },
        },
    }
    (catalog / "header.json").write_text(
        json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return catalog, str(source) + "/"


def make_combined_role_larger(source_prefix: str, *, rows: int) -> None:
    source = Path(source_prefix)
    for worker in range(8):
        main, _, _ = probe.expected_position_sets(
            total_rows=rows,
            worker_id=worker,
            workers=8,
            main_rows=32,
            control_rows_per_arm=8,
            exclude_build_rows=16,
            exclude_scan_rows=8,
        )
        for local in main[1::4] + main[2::4]:
            extension = ".jpg" if (local // 8) % 2 == 0 else ".png"
            path = source / f"{local % 31:05d}" / f"{local:064x}{extension}"
            payload = path.read_bytes()
            path.write_bytes(payload + payload)


def make_control_before_larger(source_prefix: str, *, rows: int) -> None:
    source = Path(source_prefix)
    _, before, _ = probe.expected_position_sets(
        total_rows=rows,
        worker_id=0,
        workers=8,
        main_rows=32,
        control_rows_per_arm=8,
        exclude_build_rows=16,
        exclude_scan_rows=8,
    )
    for local in before:
        extension = ".jpg" if (local // 8) % 2 == 0 else ".png"
        path = source / f"{local % 31:05d}" / f"{local:064x}{extension}"
        payload = path.read_bytes()
        path.write_bytes(payload + payload)


def runner_command(
    *,
    runner: Path,
    catalog: Path,
    source_prefix: str,
    artifact_prefix: Path,
    coord_prefix: Path,
    local_prefix: Path,
    s3_prefix: Path,
    jobtag: str,
    worker_id: int,
    job_spec: Path,
) -> list[str]:
    return [
        sys.executable,
        str(runner),
        "--catalog-root",
        str(catalog),
        "--artifact-prefix",
        str(artifact_prefix),
        "--coord-prefix",
        str(coord_prefix),
        "--local-prefix",
        str(local_prefix),
        "--s3-prefix",
        str(s3_prefix),
        "--jobtag",
        jobtag,
        "--worker-id",
        str(worker_id),
        "--workers",
        "8",
        "--main-rows",
        "32",
        "--control-rows-per-arm",
        "8",
        "--exclude-build-rows",
        "16",
        "--exclude-scan-rows",
        "8",
        "--source-prefix",
        source_prefix,
        "--source-threads",
        "2",
        "--source-attempts",
        "2",
        "--source-block-bytes",
        "1024",
        "--retry-base-seconds",
        "0.001",
        "--blob-target-bytes",
        "32768",
        "--telemetry-interval-seconds",
        "0.01",
        "--barrier-timeout-seconds",
        "180",
        "--max-byte-imbalance",
        "1.10",
        "--write-authority",
        probe.WRITE_AUTHORITY,
        "--job-spec",
        str(job_spec),
    ]


def run_workers(commands: Sequence[Sequence[str]], *, expect_success: bool) -> list[str]:
    processes = [
        subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for command in commands
    ]
    outputs: list[str] = []
    returncodes: list[int] = []
    try:
        for process in processes:
            output, _ = process.communicate(timeout=300)
            outputs.append(output)
            returncodes.append(process.returncode)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
    if expect_success:
        require(returncodes == [0] * len(commands), f"synthetic workers failed: {returncodes}\n{outputs}")
    else:
        require(any(code != 0 for code in returncodes), "negative workers unexpectedly all passed")
    return outputs


def analyze_command(
    *,
    analyzer_path: Path,
    artifact_root: Path,
    coord_root: Path,
    s3_root: Path,
    catalog: Path,
    source_prefix: str,
    jobtag: str,
    runner: Path,
    probe_job_spec: Path,
    analyzer_job_spec: Path,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(analyzer_path),
        "--artifact-root",
        str(artifact_root),
        "--coord-root",
        str(coord_root),
        "--s3-root",
        str(s3_root),
        "--catalog-root",
        str(catalog),
        "--jobtag",
        jobtag,
        "--probe-runner",
        str(runner),
        "--probe-job-spec",
        str(probe_job_spec),
        "--job-spec",
        str(analyzer_job_spec),
        "--output",
        str(output),
        "--expected-workers",
        "8",
        "--expected-physical-nodes",
        "1",
        "--expected-main-rows",
        "32",
        "--expected-control-rows-per-arm",
        "8",
        "--expected-exclude-build-rows",
        "16",
        "--expected-exclude-scan-rows",
        "8",
        "--expected-source-prefix",
        source_prefix,
        "--max-byte-imbalance",
        "1.10",
        "--e8-min",
        "0.0",
        "--e8-target",
        "0.0",
        "--max-node-skew",
        "1000",
        "--min-combined-over-source",
        "0.0",
        "--max-write-fsync-eta-share",
        "1000",
        "--eta-safety-factor",
        "1.20",
        "--partial-fraction",
        "0.75",
        "--window-seconds",
        "0.01",
        "--allow-missing-windows",
        "--verify-catalog-sha256",
        "--verify-blob-content",
    ]


def validate_yaml(probe_job: Path, analyze_job: Path, static_job: Path) -> None:
    values = {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in (probe_job, analyze_job, static_job)
    }
    for name, value in values.items():
        require(value.get("gpuPerWorker") == 0 and value.get("cpuPerWorker") == 16, f"CPU16/no-GPU drift: {name}")
        require(value.get("maxRetries") == 0, f"retry drift: {name}")
        command = value.get("command")
        require(isinstance(command, str), f"command missing: {name}")
        subprocess.run(["bash", "-n"], input=command, text=True, check=True)
        for forbidden in ("set -x", "set -eux", "pip install", "find ", "latest --"):
            require(forbidden not in command, f"forbidden token {forbidden!r}: {name}")
    probe_command = values[probe_job.name]["command"]
    require(values[probe_job.name].get("numWorkers") == 8, "probe worker count drift")
    for token in (
        "--main-rows 400000",
        "--control-rows-per-arm 60000",
        "--exclude-build-rows 100000",
        "--exclude-scan-rows 40000",
        "--blob-target-bytes 4294967296",
        "--max-s3-bytes-per-worker 31000000000",
        "--max-local-staging-bytes-per-worker 8589934592",
        "--max-fsx-evidence-bytes-per-worker 134217728",
        "--local-free-floor-bytes 1000000000000",
        "--full-partition-local-budget-bytes 500000000000",
        "--local-prefix \"$LOCAL_PREFIX\"",
        "--s3-prefix \"$S3_PREFIX\"",
        "--topology-jobtag evan-mot-crop-offlinearb-local-topology-94ns",
        "--topology-summary-raw-sha256 27c48e9d6438d992fa850a0da9192c7043ccb6714e65ffa33460a755b4c5d73a",
        "--telemetry-interval-seconds 60",
        "--max-byte-imbalance 1.10",
        "--production-shape",
    ):
        require(token in probe_command, f"probe production token missing: {token}")
    analyze_command_text = values[analyze_job.name]["command"]
    for token in (
        "SOURCE_PROBE_JOBTAG=",
        "--expected-main-rows 400000",
        "--expected-physical-nodes 8",
        "--max-s3-bytes-per-worker 31000000000",
        "--max-local-staging-bytes-per-worker 8589934592",
        "--max-fsx-evidence-bytes-per-worker 134217728",
        "--topology-jobtag evan-mot-crop-offlinearb-local-topology-94ns",
        "--topology-summary-raw-sha256 27c48e9d6438d992fa850a0da9192c7043ccb6714e65ffa33460a755b4c5d73a",
        "--s3-root \"$S3_ROOT\"",
        "--local-free-floor-bytes 1000000000000",
        "--full-partition-local-budget-bytes 500000000000",
        "--verify-catalog-sha256",
        "--verify-blob-content",
        "--production-gate",
        "--e8-min 0.75",
        "--e8-target 0.85",
        "--max-node-skew 1.25",
        "--min-combined-over-source 0.85",
        "--max-write-fsync-eta-share 0.15",
    ):
        require(token in analyze_command_text, f"analyzer production token missing: {token}")
    require("find " not in analyze_command_text and "sort -V" not in analyze_command_text, "analyzer contains auto-discovery")


def pure_negatives(
    root: Path, probe_job_spec: Path, analyzer_job_spec: Path
) -> None:
    main_sets: list[set[int]] = []
    excluded = set(probe.stratified_positions(1024, 16)) | set(probe.stratified_positions(1024, 8))
    for worker in range(8):
        main, before, after = probe.expected_position_sets(
            total_rows=1024,
            worker_id=worker,
            workers=8,
            main_rows=32,
            control_rows_per_arm=8,
            exclude_build_rows=16,
            exclude_scan_rows=8,
        )
        require(len(main) == 32 and len(set(main)) == 32, "main selection hole/duplicate")
        require(all(value % 8 == worker for value in main), "modulo partition drift")
        require(not set(main).intersection(excluded), "prior pilot exclusion drift")
        require(not set(main).intersection(before + after), "control/main overlap")
        main_sets.append(set(main))
    require(len(set().union(*main_sets)) == 256, "cross-worker selection overlap")
    expect_failure("allowed rank negative", probe.map_allowed_rank, -1, 10, [])

    fd_path = root / "fd-short-write.bin"
    descriptor = os.open(fd_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    real_os_write = probe.os.write
    calls = 0

    def interrupted_then_short(fd: int, payload: bytes | memoryview) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError()
        return real_os_write(fd, payload[: max(1, len(payload) // 2)])

    try:
        probe.os.write = interrupted_then_short
        probe.write_fd_all(descriptor, b"strict-fd-write", label="static-short")
    finally:
        probe.os.write = real_os_write
        os.close(descriptor)
    require(fd_path.read_bytes() == b"strict-fd-write", "fd EINTR/short-write loop lost bytes")

    zero_path = root / "fd-zero-write.bin"
    descriptor = os.open(zero_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        probe.os.write = lambda _fd, _payload: 0
        expect_failure(
            "zero fd write",
            probe.write_fd_all,
            descriptor,
            b"must-fail",
            label="static-zero",
        )
    finally:
        probe.os.write = real_os_write
        os.close(descriptor)

    telemetry = probe.Telemetry(root / "thread-error-telemetry.jsonl", 0.02)
    try:
        def fail_telemetry_write(fd: int, payload: bytes | memoryview) -> int:
            if fd == telemetry._fd:
                raise OSError("synthetic telemetry write failure")
            return real_os_write(fd, payload)

        probe.os.write = fail_telemetry_write
        time.sleep(0.08)
    finally:
        probe.os.write = real_os_write
    require(telemetry._thread_error is not None, "telemetry ticker did not retain its exception")
    expect_failure("telemetry thread exception propagation", telemetry.close)
    plateau_rates = analyzer.aligned_window_rates(
        [[(0.0, 0.0), (10.0, 10.0), (310.0, 10.0), (400.0, 20.0)]],
        300.0,
    )
    require(0.0 in plateau_rates, "zero-throughput five-minute plateau was discarded")
    require(
        analyzer.percentile([100.0] * 10 + [200.0] * 10, 0.10) == 100.0,
        "pooled cross-phase p10 was not computed over raw aligned windows",
    )
    marker_root = root / "s3-marker-monotonic"
    (marker_root / "s3" / "worker-0").mkdir(parents=True)
    (marker_root / "fsx").mkdir()
    (marker_root / "s3" / "worker-0" / "_SUCCESS").write_text(
        "committed\n", encoding="ascii"
    )
    require(
        probe.publication_marker_committed_or_unknown(
            SimpleNamespace(
                artifact_root=marker_root / "fsx",
                s3_worker_root=marker_root / "s3" / "worker-0",
                s3_job_root=marker_root / "s3",
                worker_id=0,
            )
        ),
        "visible S3 worker success was not treated as monotonic commit",
    )
    expect_failure(
        "receipt self-hash mismatch",
        analyzer.validate_self_hash,
        {"receipt_sha256": "0" * 64, "worker_id": 0},
        "receipt_sha256",
        "synthetic receipt",
    )

    publish_root = root / "publish-negative"
    publish_root.mkdir()
    source = publish_root / "source"
    destination = publish_root / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "payload").write_bytes(b"x")
    (source / "_SUCCESS").write_text("0" * 64 + "\n", encoding="ascii")
    expect_failure(
        "publication no-clobber",
        pilot_core.publish_directory_noreplace,
        source,
        destination,
        success_last=True,
    )
    existing_s3_key = publish_root / "existing-s3-key.bin"
    existing_s3_key.write_bytes(b"old")
    expect_failure(
        "Mountpoint-S3 final-key no-clobber",
        probe.write_s3_key_once,
        existing_s3_key,
        b"new",
    )
    require(
        existing_s3_key.read_bytes() == b"old",
        "Mountpoint-S3 no-clobber negative modified existing key",
    )

    # Approximate two independent Mountpoint views: mkdir in mount A is not
    # copied to mount B, while a concrete _INIT.json object is propagated and
    # becomes the only cross-mount coordination authority.
    multi_mount = root / "multi-mount-s3-init"
    mount_a = multi_mount / "mount-a"
    mount_b = multi_mount / "mount-b"
    mount_a.mkdir(parents=True)
    mount_b.mkdir()
    jobtag = "synthetic-cross-mount-init"
    view_a = mount_a / jobtag
    view_b = mount_b / jobtag
    view_a.mkdir()
    require(
        not view_b.exists(),
        "synthetic empty S3 directory leaked across independent mount views",
    )
    logical_s3_root = Path("/s3/evan/static-contract") / jobtag
    init_code = {"runner_sha256": "1" * 64, "job_spec_sha256": "2" * 64}
    init_topology = {"jobtag": "topology-static", "raw_sha256": "3" * 64}
    s3_init = probe.make_s3_init(
        jobtag=jobtag,
        workers=8,
        s3_job_root=logical_s3_root,
        code=init_code,
        topology_evidence=init_topology,
    )
    s3_init_raw = probe.canonical_json_bytes(s3_init)
    probe.write_s3_key_once(view_a / "_INIT.json", s3_init_raw)
    view_b.mkdir()
    probe.write_s3_key_once(view_b / "_INIT.json", s3_init_raw)
    observed, observed_raw = probe.load_and_validate_s3_init(
        view_b / "_INIT.json",
        jobtag=jobtag,
        workers=8,
        s3_job_root=logical_s3_root,
        code=init_code,
        topology_evidence=init_topology,
    )
    require(
        observed == s3_init and observed_raw == s3_init_raw,
        "concrete S3 initialization object did not bridge mount views",
    )
    coord_init = probe.make_coord_init(
        jobtag=jobtag,
        workers=8,
        s3_job_root=logical_s3_root,
        code=init_code,
        topology_evidence=init_topology,
        s3_init=observed,
        s3_init_raw=observed_raw,
    )
    require(
        coord_init["s3_init_raw_sha256"]
        == hashlib.sha256(s3_init_raw).hexdigest()
        and coord_init["s3_init_sha256"] == s3_init["init_sha256"],
        "FSx initialization did not bind the concrete S3 object",
    )
    wrong_init = dict(s3_init)
    wrong_init["workers"] = 7
    wrong_init["init_sha256"] = probe.self_hash(wrong_init, "init_sha256")
    wrong_path = view_b / "wrong_INIT.json"
    probe.write_s3_key_once(wrong_path, probe.canonical_json_bytes(wrong_init))
    expect_failure(
        "cross-mount S3 initialization binding drift",
        probe.load_and_validate_s3_init,
        wrong_path,
        jobtag=jobtag,
        workers=8,
        s3_job_root=logical_s3_root,
        code=init_code,
        topology_evidence=init_topology,
    )

    parser_args = [
        "--catalog-root",
        str(root),
        "--artifact-prefix",
        str(root),
        "--coord-prefix",
        str(publish_root),
        "--local-prefix",
        str(root),
        "--s3-prefix",
        str(publish_root),
        "--jobtag",
        "bad-shape",
        "--worker-id",
        "0",
        "--workers",
        "8",
        "--main-rows",
        "32",
        "--write-authority",
        probe.WRITE_AUTHORITY,
        "--job-spec",
        str(probe_job_spec),
        "--production-shape",
    ]
    expect_failure("production shape drift", probe.parse_args, parser_args)
    expect_failure(
        "production analyzer gate relaxation",
        analyzer.parse_args,
        [
            "--artifact-root",
            str(root),
            "--coord-root",
            str(publish_root),
            "--s3-root",
            str(root),
            "--catalog-root",
            str(root),
            "--jobtag",
            "bad-gate",
            "--probe-runner",
            str(Path(probe.__file__).resolve(strict=True)),
            "--probe-job-spec",
            str(probe_job_spec),
            "--job-spec",
            str(analyzer_job_spec),
            "--output",
            str(root / "bad-gate-analysis.json"),
            "--e8-min",
            "0.70",
            "--verify-catalog-sha256",
            "--verify-blob-content",
            "--production-gate",
        ],
    )

    telemetry = root / "bad-telemetry.jsonl"
    bad = {
        "schema_version": probe.SCHEMA_VERSION,
        "timestamp": probe.utc_now(),
        "reason": "interval",
        "phase": "source_a",
        "elapsed_seconds": 1.0,
        "phase_elapsed_seconds": 1.0,
        "phase_counters": {key: 0 for key in probe.Telemetry.FIELDS},
        "total_counters": {key: 0 for key in probe.Telemetry.FIELDS},
        "process": {"rss_bytes": 1, "max_rss_bytes": 1, "fd_count": 1, "cpu_user_seconds": 0.0, "cpu_system_seconds": 0.0},
    }
    telemetry.write_bytes(probe.canonical_json_bytes(bad) + b"not-json\n")
    expect_failure(
        "telemetry malformed line",
        analyzer.parse_telemetry,
        telemetry,
        {"bytes": telemetry.stat().st_size, "sha256": probe.sha256_file(telemetry)},
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--probe-job-spec", type=Path, required=True)
    parser.add_argument("--analyzer-job-spec", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().absolute()
    root.mkdir(parents=True, mode=0o700)
    runner = Path(probe.__file__).resolve(strict=True)
    analyzer_path = Path(analyzer.__file__).resolve(strict=True)
    validate_yaml(args.probe_job_spec, args.analyzer_job_spec, args.job_spec)
    pure_negatives(root, args.probe_job_spec, args.analyzer_job_spec)

    good = root / "good"
    good.mkdir()
    catalog, source_prefix = create_catalog(good, rows=1024, imbalanced=False)
    artifact_prefix = good / "artifacts"
    coord_prefix = good / "coord"
    local_prefix = good / "local"
    s3_prefix = good / "s3"
    artifact_prefix.mkdir()
    coord_prefix.mkdir()
    local_prefix.mkdir()
    s3_prefix.mkdir()
    jobtag = "synthetic-eight-worker-pass"
    commands = [
        runner_command(
            runner=runner,
            catalog=catalog,
            source_prefix=source_prefix,
            artifact_prefix=artifact_prefix,
            coord_prefix=coord_prefix,
            local_prefix=local_prefix,
            s3_prefix=s3_prefix,
            jobtag=jobtag,
            worker_id=worker,
            job_spec=args.probe_job_spec,
        )
        for worker in range(8)
    ]
    run_workers(commands, expect_success=True)
    analysis_output = good / "analysis.json"
    completed = subprocess.run(
        analyze_command(
            analyzer_path=analyzer_path,
            artifact_root=artifact_prefix / jobtag,
            coord_root=coord_prefix / jobtag,
            s3_root=s3_prefix / jobtag,
            catalog=catalog,
            source_prefix=source_prefix,
            jobtag=jobtag,
            runner=runner,
            probe_job_spec=args.probe_job_spec,
            analyzer_job_spec=args.analyzer_job_spec,
            output=analysis_output,
        ),
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    require(completed.returncode == 0, f"synthetic analyzer failed\n{completed.stdout}\n{completed.stderr}")
    result = json.loads(analysis_output.read_bytes())
    require(result.get("status") == "PASS", "synthetic analysis did not pass")
    require(result["shape"]["sample_rows_per_worker_per_arm"] == 8, "synthetic per-arm shape drift")
    require(result["shape"]["main_total_rows"] == 256, "synthetic total shape drift")
    require(result["shape"]["combined_total_rows"] == 128, "synthetic combined shape drift")
    require(result["shape"]["all_source_rows_including_controls"] == 272, "synthetic all-source shape drift")
    require(result["evidence"]["physical_node_count"] == 1, "synthetic boot binding drift")

    bad = root / "imbalanced"
    bad.mkdir()
    bad_catalog, bad_prefix = create_catalog(bad, rows=1024, imbalanced=True)
    bad_artifacts = bad / "artifacts"
    bad_coord = bad / "coord"
    bad_local = bad / "local"
    bad_s3 = bad / "s3"
    bad_artifacts.mkdir()
    bad_coord.mkdir()
    bad_local.mkdir()
    bad_s3.mkdir()
    bad_jobtag = "synthetic-byte-imbalance-reject"
    bad_commands = [
        runner_command(
            runner=runner,
            catalog=bad_catalog,
            source_prefix=bad_prefix,
            artifact_prefix=bad_artifacts,
            coord_prefix=bad_coord,
            local_prefix=bad_local,
            s3_prefix=bad_s3,
            jobtag=bad_jobtag,
            worker_id=worker,
            job_spec=args.probe_job_spec,
        )
        for worker in range(8)
    ]
    run_workers(bad_commands, expect_success=False)
    require(not (bad_artifacts / bad_jobtag / "_SUCCESS").exists(), "byte-imbalance negative published job success")
    require(not (bad_coord / bad_jobtag / "GLOBAL_APPROVED.json").exists(), "byte-imbalance negative published global approval")

    role_bad = root / "role-imbalanced"
    role_bad.mkdir()
    role_catalog, role_prefix = create_catalog(role_bad, rows=1024, imbalanced=False)
    make_combined_role_larger(role_prefix, rows=1024)
    role_artifacts = role_bad / "artifacts"
    role_coord = role_bad / "coord"
    role_local = role_bad / "local"
    role_s3 = role_bad / "s3"
    role_artifacts.mkdir()
    role_coord.mkdir()
    role_local.mkdir()
    role_s3.mkdir()
    role_jobtag = "synthetic-role-byte-imbalance-reject"
    role_commands = [
        runner_command(
            runner=runner,
            catalog=role_catalog,
            source_prefix=role_prefix,
            artifact_prefix=role_artifacts,
            coord_prefix=role_coord,
            local_prefix=role_local,
            s3_prefix=role_s3,
            jobtag=role_jobtag,
            worker_id=worker,
            job_spec=args.probe_job_spec,
        )
        for worker in range(8)
    ]
    run_workers(role_commands, expect_success=False)
    require(not (role_artifacts / role_jobtag / "_SUCCESS").exists(), "role-imbalance negative published job success")
    require(not (role_coord / role_jobtag / "GLOBAL_APPROVED.json").exists(), "role-imbalance negative published global approval")

    control_bad = root / "control-imbalanced"
    control_bad.mkdir()
    control_catalog, control_prefix = create_catalog(
        control_bad, rows=1024, imbalanced=False
    )
    make_control_before_larger(control_prefix, rows=1024)
    control_artifacts = control_bad / "artifacts"
    control_coord = control_bad / "coord"
    control_local = control_bad / "local"
    control_s3 = control_bad / "s3"
    control_artifacts.mkdir()
    control_coord.mkdir()
    control_local.mkdir()
    control_s3.mkdir()
    control_jobtag = "synthetic-control-byte-imbalance-reject"
    control_commands = [
        runner_command(
            runner=runner,
            catalog=control_catalog,
            source_prefix=control_prefix,
            artifact_prefix=control_artifacts,
            coord_prefix=control_coord,
            local_prefix=control_local,
            s3_prefix=control_s3,
            jobtag=control_jobtag,
            worker_id=worker,
            job_spec=args.probe_job_spec,
        )
        for worker in range(8)
    ]
    run_workers(control_commands, expect_success=False)
    require(not (control_artifacts / control_jobtag / "_SUCCESS").exists(), "control-imbalance negative published job success")
    require(not (control_coord / control_jobtag / "GLOBAL_APPROVED.json").exists(), "control-imbalance negative published global approval")

    print("OFFLINE_GPIC_SCALE_STATIC_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
