#!/usr/bin/env python3
"""Shared fail-closed contracts for the full GPIC S3 production artifact.

This module is intentionally separate from the frozen bounded pilot and scale
probe.  It reuses their already-audited low-level publication and catalog
helpers, while giving the production build its own schemas and namespaces.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import struct
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

from scripts.offline_arb import run_offline_gpic_blob_pilot as pilot_core
from scripts.offline_arb import run_offline_gpic_scale_probe as scale_core


PRODUCTION_SCHEMA = "offline_gpic_s3_production_v1"
BUILD_INIT_SCHEMA = "offline_gpic_s3_production_build_init_v1"
PARTITION_PLAN_SCHEMA = "offline_gpic_s3_production_partition_plan_v1"
COORD_INIT_SCHEMA = "offline_gpic_s3_production_coord_init_v1"
LOCAL_MANIFEST_SCHEMA = "offline_gpic_s3_production_local_manifest_v1"
WORKER_RECEIPT_SCHEMA = "offline_gpic_s3_production_worker_receipt_v1"
BUILD_MANIFEST_SCHEMA = "offline_gpic_s3_production_build_manifest_v1"
BUILD_COMPLETE_SCHEMA = "offline_gpic_s3_production_build_complete_v1"
VERIFIER_WORKER_SCHEMA = "offline_gpic_s3_production_verifier_worker_v1"
VERIFIER_SUMMARY_SCHEMA = "offline_gpic_s3_production_verifier_summary_v1"
CANONICAL_MANIFEST_SCHEMA = "offline_gpic_blob_v1"

PRODUCTION_WORKERS = 8
PRODUCTION_GPIC_ROWS = 20_000_000
PRODUCTION_ROWS_PER_WORKER = 2_500_000
PRODUCTION_BLOB_TARGET_BYTES = 4 * (1 << 30)
# Node-local footprint ceiling for direct tar streaming into final 4-GiB
# blobs.  The production path creates zero per-tar payload chunks.
PRODUCTION_MAX_LOCAL_BYTES_PER_WORKER = 1_000_000_000_000
# This is a source-read budget, not node-local storage.  Every selected worker
# scans its complete worker-owned tar archives, including unselected JSON and
# tar padding, so it must remain independent of the /local footprint cap.
PRODUCTION_MAX_SOURCE_ARCHIVE_BYTES_PER_WORKER = 1_800_000_000_000
PRODUCTION_MAX_S3_TOTAL_BYTES = 3_200_000_000_000
PRODUCTION_S3_CONTROLS_RESERVE_BYTES = 1_000_000_000
PRODUCTION_MAX_S3_PAYLOAD_BYTES = (
    PRODUCTION_MAX_S3_TOTAL_BYTES - PRODUCTION_S3_CONTROLS_RESERVE_BYTES
)
PRODUCTION_MAX_S3_BYTES_PER_WORKER = (
    PRODUCTION_MAX_S3_PAYLOAD_BYTES // PRODUCTION_WORKERS
)
PRODUCTION_MAX_FSX_INCREMENTAL_BYTES = 50_000_000_000
PRODUCTION_DENSE_INDEX_BYTES = 1_120_000_000
PRODUCTION_SOURCE_THREADS = 1
PRODUCTION_ALLOWED_SOURCE_THREADS = (24, 32)
PRODUCTION_BYTE_IMBALANCE_CAP = 1.10
PRODUCTION_PARTITION_MODES = ("shard_modulo", "shard_balanced_rows")
PRODUCTION_SOURCE_MODES = ("small_files", "tar_sequential")
PRODUCTION_ALLOWED_TAR_WORKERS = (1, 4, 8)
PRODUCTION_TAR_WORKERS = 8
PRODUCTION_TAR_READ_BUFFER_BYTES = 16 * (1 << 20)
PRODUCTION_TAR_IO_BLOCK_BYTES = 8 * (1 << 20)
PRODUCTION_TAR_MAX_SELECTED_BYTES_PER_TAR = 1_000_000_000
# Production is tar-sequential: a stream error is fatal and never retried.
PRODUCTION_SOURCE_ATTEMPTS = 1
PRODUCTION_SOURCE_BLOCK_BYTES = 1 << 20
PRODUCTION_S3_BLOCK_BYTES = 8 * (1 << 20)
PRODUCTION_TOPOLOGY_JOBTAG = scale_core.PRODUCTION_TOPOLOGY_JOBTAG
PRODUCTION_TOPOLOGY_RAW_SHA256 = scale_core.PRODUCTION_TOPOLOGY_RAW_SHA256
PRODUCTION_TOPOLOGY_SUMMARY = scale_core.PRODUCTION_TOPOLOGY_SUMMARY
PRODUCTION_SOURCE_PREFIX = scale_core.PRODUCTION_SOURCE_PREFIX
PRODUCTION_TAR_MAPPING_JOBTAG = (
    "evan-mot-crop-offlinearb-gpic-tar-mapping-probe-wvsq"
)
PRODUCTION_TAR_MAPPING_RAW_SHA256 = (
    "11f680e771b8f636d3125dbce095b3b780fc29b30d9f102091febc9a0393dc92"
)
PRODUCTION_TAR_MAPPING_SELF_SHA256 = (
    "d7d0733f2f70aa9896b78cb3b9edcee7adfb5317b570ccf5ecda45cebb2961c4"
)
PRODUCTION_TAR_MAPPING_REPORT = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_gpic_tar_mapping_probe_20260712v1/"
    f"{PRODUCTION_TAR_MAPPING_JOBTAG}/report.json"
)
PRODUCTION_TAR_QUICK_JOBTAG = "evan-mot-crop-offlinearb-gpic-tar-seq-quick-vpfq"
PRODUCTION_TAR_QUICK_RAW_SHA256 = (
    "8b5c31121e8a7c7240744396fa56a0950fa9d2f90cb93a5f4e513859031e29d0"
)
PRODUCTION_TAR_QUICK_SUMMARY = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_gpic_tar_seq_quick_v1/"
    f"{PRODUCTION_TAR_QUICK_JOBTAG}/summary.json"
)
PRODUCTION_TAR_PARSER_GATE_JOBTAG = (
    "evan-mot-crop-offlinearb-gpic-tar-parser-gate-p5gm"
)
PRODUCTION_TAR_PARSER_GATE_RAW_SHA256 = (
    "a8f8b02db06cadffab80a73f2a898880ae1da0c35d2fe602e500d6495484ef9e"
)
PRODUCTION_TAR_PARSER_GATE_SELF_SHA256 = (
    "f0d724fe08a919736e4ba51b72ab1071ba5bdf932e3be77fd4ae53214bab324f"
)
PRODUCTION_TAR_PARSER_GATE_COMPLETE_RAW_SHA256 = (
    "4252e395c512aa45628da0ffeb97212e910fd4c04242b2f84f119e21e53060be"
)
PRODUCTION_TAR_PARSER_GATE_SUMMARY = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_gpic_tar_parser_gate_20260712v1/"
    f"{PRODUCTION_TAR_PARSER_GATE_JOBTAG}/summary.json"
)
PRODUCTION_TAR_PARSER_GATE_CODE_SHA256 = {
    "common": "e47896a0c29456025cdaaabaa6ba1ef7d157333d3869579fbc63806a72a83f87",
    "job_spec": "7a8800e9590d4e164b1ecffb098a5e8c6ac56bb6b41baa93783bb989dd8f477d",
    "pilot_core": "3a3917844cdd1688e635eef5c3f1de6699de86ab4a45e42f2012cb52de61e137",
    "runner": "a5f0222fb5044572c9140297470e8389f306bfd593132e77ed7ef672e7fdc41f",
    "tar_source": "5a32fbec3fa8b164f8e2ca4192cd861da5206cf8cfe7e9dd91269b8b7a5bd06e",
}
PRODUCTION_TAR_ROOT = Path(
    "/s3-mewtant-raw-datasets/stanford-vision-lab/gpic/train"
)
WRITE_AUTHORITY = "offline-gpic-s3-production-v1-write-authorized"
VERIFY_AUTHORITY = "offline-gpic-s3-production-v1-verify-authorized"
RECORD_STRUCT = struct.Struct("<qIQI32s")
MIB = 1 << 20
RELEASE_FIELD = "release"
RELEASE_STORAGE_FIELD = "storage"
RELEASE_STORAGE_S3_ROOT_FIELD = "s3_root"
RELEASE_STORAGE_INDEX_PATH_FIELD = "index_path"
RELEASE_STORAGE_S3_INIT_FIELD = "s3_init"
RELEASE_STORAGE_S3_INIT_PATH_FIELD = "path"
RELEASE_STORAGE_S3_INIT_BYTES_FIELD = "bytes"
RELEASE_STORAGE_S3_INIT_RAW_SHA256_FIELD = "raw_sha256"
S3_INIT_RELATIVE_PATH = "_INIT.json"

if RECORD_STRUCT.size != 56:
    raise RuntimeError("production record struct must be exactly 56 bytes")
if PRODUCTION_GPIC_ROWS * RECORD_STRUCT.size != PRODUCTION_DENSE_INDEX_BYTES:
    raise RuntimeError("production dense-index size constant is inconsistent")


class ProductionError(RuntimeError):
    """A production invariant, integrity check, or publication gate failed."""


def utc_now() -> str:
    return pilot_core.utc_now()


def canonical_json_bytes(value: Any) -> bytes:
    return pilot_core.canonical_json_bytes(value)


def self_hash(value: dict[str, Any], field: str) -> str:
    core = dict(value)
    core.pop(field, None)
    return hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def is_sha256(value: Any) -> bool:
    return pilot_core.is_sha256(value)


def require_regular_file(path: Path) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ProductionError(f"missing regular file: {path}") from exc
    if not path.is_file() or path.is_symlink() or value.st_nlink < 1:
        raise ProductionError(f"unsafe non-regular file: {path}")
    return value


def require_regular_dir(path: Path) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ProductionError(f"missing directory: {path}") from exc
    if not path.is_dir() or path.is_symlink() or value.st_nlink < 1:
        raise ProductionError(f"unsafe non-directory: {path}")
    return value


def sha256_file(path: Path, block_bytes: int = PRODUCTION_S3_BLOCK_BYTES) -> str:
    try:
        return pilot_core.sha256_file(path, block_bytes)
    except Exception as exc:
        raise ProductionError(f"file SHA-256 closure failed: {path}") from exc


def strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    before = require_regular_file(path)
    raw = path.read_bytes()
    after = require_regular_file(path)
    if before.st_size != len(raw) or after.st_size != len(raw):
        raise ProductionError(f"JSON file size drift: {path}")
    try:
        def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            output: dict[str, Any] = {}
            for key, item in pairs:
                if key in output:
                    raise ProductionError(f"duplicate JSON key {key!r}: {path}")
                output[key] = item
            return output

        value = json.loads(raw, object_pairs_hook=reject_duplicate_pairs)
    except Exception as exc:
        raise ProductionError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ProductionError(f"non-canonical JSON object: {path}")
    return value, raw


def validate_self_hash(value: dict[str, Any], field: str, label: str) -> None:
    if value.get(field) != self_hash(value, field):
        raise ProductionError(f"{label} self-hash drift")


def write_fsx_bytes_once(path: Path, payload: bytes) -> None:
    if not str(path.absolute()).startswith("/fsx/"):
        raise ProductionError(f"FSx publication escaped /fsx: {path}")
    scale_core.write_once_atomic_bytes(path, payload, fsync_file=True)


def write_fsx_json_once(path: Path, value: Any) -> bytes:
    raw = canonical_json_bytes(value)
    write_fsx_bytes_once(path, raw)
    return raw


def write_s3_bytes_once(path: Path, payload: bytes) -> None:
    scale_core.write_s3_key_once(path, payload)


def exact_names(root: Path, expected: set[str], label: str) -> None:
    require_regular_dir(root)
    actual = {path.name for path in root.iterdir()}
    if actual != expected:
        raise ProductionError(
            f"{label} exact name set drift: {sorted(actual ^ expected)}"
        )


def write_all(handle: Any, payload: bytes | memoryview, *, label: str) -> None:
    try:
        pilot_core.write_all(handle, payload, label=label)
    except Exception as exc:
        raise ProductionError(f"short write: {label}") from exc


def pwrite_all(descriptor: int, payload: bytes, offset: int, *, label: str) -> None:
    if descriptor < 0 or offset < 0 or not payload:
        raise ProductionError(f"invalid pwrite request: {label}")
    view = memoryview(payload)
    cursor = offset
    while view:
        try:
            written = os.pwrite(descriptor, view, cursor)
        except InterruptedError:
            continue
        if written <= 0 or written > len(view):
            raise ProductionError(f"short/zero pwrite: {label}")
        cursor += written
        view = view[written:]


def read_exact(handle: Any, length: int, *, label: str) -> bytes:
    if type(length) is not int or length < 0:
        raise ProductionError(f"invalid exact-read length: {label}")
    output = bytearray()
    while len(output) < length:
        block = handle.read(length - len(output))
        if not block:
            raise ProductionError(f"short read: {label}")
        output.extend(block)
    return bytes(output)


def pread_exact(descriptor: int, length: int, offset: int, *, label: str) -> bytes:
    if descriptor < 0 or length < 0 or offset < 0:
        raise ProductionError(f"invalid pread request: {label}")
    output = bytearray()
    while len(output) < length:
        try:
            block = os.pread(descriptor, length - len(output), offset + len(output))
        except InterruptedError:
            continue
        if not block:
            raise ProductionError(f"short pread: {label}")
        output.extend(block)
    return bytes(output)


def file_binding(path: Path) -> dict[str, Any]:
    metadata = require_regular_file(path)
    return {
        "name": path.name,
        "bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def code_snapshot(
    *, runner: Path, job_spec: Path, extra_files: dict[str, Path] | None = None
) -> dict[str, Any]:
    files = {
        "runner": runner.resolve(strict=True),
        "job_spec": job_spec.resolve(strict=True),
        "common": Path(__file__).resolve(strict=True),
        "pilot_core": Path(pilot_core.__file__).resolve(strict=True),
        "scale_core": Path(scale_core.__file__).resolve(strict=True),
    }
    if extra_files:
        for label, path in extra_files.items():
            if label in files or not label:
                raise ProductionError(f"duplicate/empty code binding label: {label}")
            files[label] = path.resolve(strict=True)
    return {
        "files": {
            label: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for label, path in sorted(files.items())
        },
        "python": platform.python_version(),
        "record_struct": RECORD_STRUCT.format,
        "record_bytes": RECORD_STRUCT.size,
    }


def validate_code_snapshot(value: dict[str, Any]) -> None:
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise ProductionError("code snapshot file map is missing")
    for label, binding in files.items():
        if not isinstance(binding, dict) or not is_sha256(binding.get("sha256")):
            raise ProductionError(f"malformed code binding: {label}")
        path = Path(binding.get("path", ""))
        if sha256_file(path) != binding["sha256"]:
            raise ProductionError(f"code changed during production task: {label}")
    if (
        value.get("record_struct") != RECORD_STRUCT.format
        or value.get("record_bytes") != RECORD_STRUCT.size
    ):
        raise ProductionError("production record ABI drift")


def load_production_catalog(
    root: Path, *, production_gate: bool
) -> pilot_core.CatalogBinding:
    try:
        binding = pilot_core.load_catalog(root, verify_gpic_sha256=True)
    except Exception as exc:
        raise ProductionError("catalog binding/complete SHA verification failed") from exc
    if production_gate and binding.gpic_rows != PRODUCTION_GPIC_ROWS:
        raise ProductionError("production GPIC row count drift")
    if binding.gpic_rows % PRODUCTION_WORKERS != 0:
        raise ProductionError("GPIC rows do not divide exactly across workers")
    return binding


def catalog_manifest(binding: pilot_core.CatalogBinding) -> dict[str, Any]:
    return pilot_core.catalog_manifest(binding)


def validate_catalog_closure(binding: pilot_core.CatalogBinding) -> None:
    try:
        pilot_core.validate_catalog_closure(binding)
    except Exception as exc:
        raise ProductionError("catalog identity changed during production task") from exc


def load_topology_evidence(
    path: Path, *, expected_jobtag: str, expected_raw_sha256: str
) -> dict[str, Any]:
    try:
        return scale_core.load_topology_evidence(
            path,
            expected_jobtag=expected_jobtag,
            expected_raw_sha256=expected_raw_sha256,
        )
    except Exception as exc:
        raise ProductionError("local topology evidence binding failed") from exc


def local_mount_binding(path: Path, *, production_gate: bool) -> dict[str, Any]:
    try:
        value = scale_core.local_mount_binding(path, require_ext4=production_gate)
    except Exception as exc:
        raise ProductionError("live local mount binding failed") from exc
    if production_gate and (
        value.get("mount_target") != "/local/evan"
        or value.get("fstype") != "ext4"
        or value.get("total_bytes", 0) < PRODUCTION_MAX_LOCAL_BYTES_PER_WORKER
        or value.get("free_bytes", 0) < PRODUCTION_MAX_LOCAL_BYTES_PER_WORKER
    ):
        raise ProductionError("live /local/evan production capacity drift")
    return value


def runtime_identity() -> dict[str, Any]:
    try:
        return pilot_core.runtime_identity()
    except Exception as exc:
        raise ProductionError("runtime physical-host binding failed") from exc


def make_build_init(
    *,
    build_jobtag: str,
    workers: int,
    s3_root: Path,
    artifact_root: Path,
    coord_root: Path,
    code: dict[str, Any],
    catalog: dict[str, Any],
    topology_evidence: dict[str, Any],
    limits: dict[str, Any],
    partition_plan: dict[str, Any],
    partition_plan_raw: bytes,
    source_authority: dict[str, Any],
) -> dict[str, Any]:
    validate_self_hash(
        partition_plan, "partition_plan_sha256", "partition plan"
    )
    if partition_plan_raw != canonical_json_bytes(partition_plan):
        raise ProductionError("partition plan raw/canonical drift")
    value = {
        "schema_version": BUILD_INIT_SCHEMA,
        "build_jobtag": build_jobtag,
        "workers": workers,
        "s3_root": str(s3_root),
        "artifact_root": str(artifact_root),
        "coord_root": str(coord_root),
        "code": code,
        "catalog": catalog,
        "topology_evidence": topology_evidence,
        "limits": limits,
        "partition_mode": partition_plan["partition_mode"],
        "partition_plan_raw_sha256": hashlib.sha256(
            partition_plan_raw
        ).hexdigest(),
        "partition_plan_sha256": partition_plan["partition_plan_sha256"],
        "source_authority": source_authority,
    }
    value["build_init_sha256"] = self_hash(value, "build_init_sha256")
    return value


def build_init_binding(value: dict[str, Any], raw: bytes) -> dict[str, str]:
    validate_self_hash(value, "build_init_sha256", "build initialization")
    if raw != canonical_json_bytes(value):
        raise ProductionError("build initialization raw/canonical drift")
    return {
        "build_init_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "build_init_sha256": value["build_init_sha256"],
    }


def make_coord_init(
    *, build_init: dict[str, Any], build_init_raw: bytes
) -> dict[str, Any]:
    value = {
        "schema_version": COORD_INIT_SCHEMA,
        "build_jobtag": build_init["build_jobtag"],
        "s3_build_init_key": str(
            Path(build_init["s3_root"]) / S3_INIT_RELATIVE_PATH
        ),
        **build_init_binding(build_init, build_init_raw),
    }
    value["coord_init_sha256"] = self_hash(value, "coord_init_sha256")
    return value


def wait_for_exact_file(
    path: Path,
    *,
    timeout_seconds: float,
    failure_root: Path | None = None,
    label: str,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ProductionError(f"unable to inspect {label}") from exc
        else:
            return
        if failure_root is not None:
            try:
                failures = sorted(failure_root.glob("FAIL.*.json"))
            except OSError:
                failures = []
            if failures:
                raise ProductionError(
                    f"peer failure while waiting for {label}: {failures[0].name}"
                )
        if time.monotonic() >= deadline:
            raise ProductionError(f"timed out waiting for {label}")
        time.sleep(0.5)


def validate_build_init(
    path: Path,
    *,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], bytes]:
    value, raw = strict_json(path)
    validate_self_hash(value, "build_init_sha256", "build initialization")
    if value != expected or raw != canonical_json_bytes(expected):
        raise ProductionError("build initialization exact binding drift")
    return value, raw


def production_limits(
    *,
    workers: int,
    gpic_rows: int,
    blob_target_bytes: int,
    max_local_bytes_per_worker: int,
    max_source_archive_bytes_per_worker: int,
    max_s3_bytes_per_worker: int,
    max_s3_payload_bytes: int,
    max_s3_total_bytes: int,
    s3_controls_reserve_bytes: int,
    max_fsx_incremental_bytes: int,
) -> dict[str, Any]:
    return {
        "workers": workers,
        "gpic_rows": gpic_rows,
        "nominal_rows_per_worker": gpic_rows // workers,
        "record_bytes": RECORD_STRUCT.size,
        "dense_index_bytes": gpic_rows * RECORD_STRUCT.size,
        "blob_target_bytes": blob_target_bytes,
        "max_local_bytes_per_worker": max_local_bytes_per_worker,
        "max_source_archive_bytes_per_worker": (
            max_source_archive_bytes_per_worker
        ),
        "max_s3_bytes_per_worker": max_s3_bytes_per_worker,
        "max_s3_payload_bytes": max_s3_payload_bytes,
        "max_s3_total_bytes": max_s3_total_bytes,
        "s3_controls_reserve_bytes": s3_controls_reserve_bytes,
        "max_fsx_incremental_bytes": max_fsx_incremental_bytes,
    }


def validate_production_limits(value: dict[str, Any]) -> None:
    expected = production_limits(
        workers=PRODUCTION_WORKERS,
        gpic_rows=PRODUCTION_GPIC_ROWS,
        blob_target_bytes=PRODUCTION_BLOB_TARGET_BYTES,
        max_local_bytes_per_worker=PRODUCTION_MAX_LOCAL_BYTES_PER_WORKER,
        max_source_archive_bytes_per_worker=(
            PRODUCTION_MAX_SOURCE_ARCHIVE_BYTES_PER_WORKER
        ),
        max_s3_bytes_per_worker=PRODUCTION_MAX_S3_BYTES_PER_WORKER,
        max_s3_payload_bytes=PRODUCTION_MAX_S3_PAYLOAD_BYTES,
        max_s3_total_bytes=PRODUCTION_MAX_S3_TOTAL_BYTES,
        s3_controls_reserve_bytes=PRODUCTION_S3_CONTROLS_RESERVE_BYTES,
        max_fsx_incremental_bytes=PRODUCTION_MAX_FSX_INCREMENTAL_BYTES,
    )
    if value != expected:
        raise ProductionError("frozen production storage/shape limits drift")


def global_blob_id(worker_id: int, sequence: int, workers: int) -> int:
    if not (0 <= worker_id < workers) or sequence < 0:
        raise ProductionError("invalid worker/blob sequence")
    value = worker_id + workers * sequence
    if value > 0xFFFFFFFF:
        raise ProductionError("global blob id exceeds uint32")
    return value


def blob_name(blob_id: int) -> str:
    if type(blob_id) is not int or not (0 <= blob_id <= 0xFFFFFFFF):
        raise ProductionError("invalid blob id")
    return f"blob-{blob_id:08d}.bin"


def upload_file_once(
    local_path: Path,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    block_bytes: int = PRODUCTION_S3_BLOCK_BYTES,
) -> None:
    metadata = require_regular_file(local_path)
    if metadata.st_size != expected_bytes or not is_sha256(expected_sha256):
        raise ProductionError(f"local upload binding drift: {local_path.name}")
    # The plan deliberately requires a complete local fingerprint pass before
    # every upload attempt.  This is separate from the upload stream hash.
    if sha256_file(local_path, block_bytes) != expected_sha256:
        raise ProductionError(f"local blob fingerprint drift: {local_path.name}")
    if destination.exists() or destination.is_symlink():
        raise ProductionError(f"S3 destination key is not fresh: {destination}")
    source: Any = None
    output: Any = None
    try:
        source = local_path.open("rb", buffering=0)
        output = destination.open("xb", buffering=0)
        digest = hashlib.sha256()
        copied = 0
        while True:
            block = source.read(block_bytes)
            if not block:
                break
            digest.update(block)
            write_all(output, block, label=f"S3 upload {destination.name}")
            copied += len(block)
        if copied != expected_bytes or digest.hexdigest() != expected_sha256:
            raise ProductionError(f"upload stream fingerprint drift: {local_path.name}")
        output.flush()
        os.fsync(output.fileno())
        output.close()
        output = None
        source.close()
        source = None
    except BaseException as exc:
        for handle in (output, source):
            if handle is not None:
                try:
                    handle.close()
                except BaseException:
                    pass
        try:
            destination.lstat()
        except FileNotFoundError:
            raise
        except BaseException as inspect_exc:
            raise pilot_core.CommittedSyncUnconfirmed(
                f"S3 upload state is unknowable: {destination.name}"
            ) from inspect_exc
        raise pilot_core.CommittedSyncUnconfirmed(
            f"S3 key became visible while upload reported failure: {destination.name}"
        ) from exc
    visible = require_regular_file(destination)
    if visible.st_size != expected_bytes:
        raise pilot_core.CommittedSyncUnconfirmed(
            f"S3 key size is unconfirmed after fsync: {destination.name}"
        )


def fsx_tree_bytes(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        if path.is_symlink():
            raise ProductionError(f"FSx evidence contains a symlink: {path}")
        if path.is_file():
            total += require_regular_file(path).st_size
        elif path.is_dir():
            total += fsx_tree_bytes(path.iterdir())
        else:
            raise ProductionError(f"unsafe FSx evidence entry: {path}")
    return total


def copy_file_fsync_once(source: Path, destination: Path, *, block_bytes: int) -> None:
    source_binding = file_binding(source)
    if destination.exists() or destination.is_symlink():
        raise ProductionError(f"copy destination is not fresh: {destination}")
    with source.open("rb", buffering=0) as input_handle, destination.open(
        "xb", buffering=0
    ) as output_handle:
        digest = hashlib.sha256()
        count = 0
        while True:
            block = input_handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
            write_all(output_handle, block, label=f"copy {destination.name}")
            count += len(block)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    if (
        count != source_binding["bytes"]
        or digest.hexdigest() != source_binding["sha256"]
        or file_binding(destination) != {
            **source_binding,
            "name": destination.name,
        }
    ):
        raise ProductionError(f"copied file fingerprint drift: {destination}")


def write_failure(
    coord_root: Path,
    *,
    build_jobtag: str,
    actor: str,
    actor_id: int,
    exc: BaseException,
) -> None:
    value = {
        "schema_version": PRODUCTION_SCHEMA,
        "build_jobtag": build_jobtag,
        "actor": actor,
        "actor_id": actor_id,
        "exception_class": type(exc).__name__,
        "timestamp": utc_now(),
    }
    write_fsx_json_once(
        coord_root / f"FAIL.{actor}-{actor_id}.{uuid.uuid4().hex}.json", value
    )


def success_or_unknown(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False
    except BaseException:
        return True


def remove_tree_after_commit(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise ProductionError(f"unsafe cleanup root: {path}")
    shutil.rmtree(path)


def assert_no_root_success(s3_root: Path) -> None:
    marker = s3_root / "_SUCCESS"
    try:
        marker.lstat()
    except FileNotFoundError:
        return
    except BaseException as exc:
        raise pilot_core.CommittedSyncUnconfirmed(
            "root success-marker absence is unknowable during build"
        ) from exc
    raise ProductionError("builder/coordinator must not write root _SUCCESS")


def stable_unique_boot_hashes(runtimes: Sequence[dict[str, Any]], label: str) -> set[str]:
    values: set[str] = set()
    for index, runtime in enumerate(runtimes):
        value = runtime.get("boot_id_sha256") if isinstance(runtime, dict) else None
        if (
            not is_sha256(value)
            or runtime.get("boot_id_raw_published") is not False
        ):
            raise ProductionError(f"{label} runtime binding drift index={index}")
        values.add(value)
    return values
