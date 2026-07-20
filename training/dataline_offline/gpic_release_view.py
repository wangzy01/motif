"""Deterministic projection of a production GPIC release into runtime controls.

The authoritative publication is the verifier's ``canonical_manifest.json``
plus ``verifier_summary.json``.  The existing training reader consumes an
older, smaller seven-field manifest and verifier-report schema.  This module
is the single pure projection between those representations.

It performs no filesystem writes.  Callers are expected to compare the
projected canonical JSON bytes with the files in a compatibility view before
using the view.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


CANONICAL_SCHEMA = "offline_gpic_blob_v1"
CANONICAL_STATUS = "PRODUCTION_VERIFIED"
SUMMARY_SCHEMA = "offline_gpic_s3_production_verifier_summary_v1"
RUNTIME_STATUS = "PASS"
VERIFY_SCHEMA = "offline_gpic_blob_verify_v1"
BACKEND = "mountpoint_s3"
LAYOUT_VERSION = "offline_gpic_blob_index_le_v1"
SELECTION_MODE = "full_catalog"
RECORD_STRUCT = "<qIQI32s"
RECORD_BYTES = 56
WORKERS = 8
MAX_SEGMENT_BYTES = 256 << 20
SHA256_RE = re.compile(r"[0-9a-f]{64}")
JOBTAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

CANONICAL_KEYS = {
    "blob_count",
    "blobs",
    "build_complete_raw_sha256",
    "build_complete_sha256",
    "build_jobtag",
    "build_manifest_raw_sha256",
    "build_manifest_sha256",
    "catalog",
    "dense_index",
    "full_tar_mapping",
    "generated_at",
    "limits",
    "manifest_sha256",
    "partition_mode",
    "partition_plan_raw_sha256",
    "partition_plan_sha256",
    "payload_bytes",
    "publication_semantics",
    "record_bytes",
    "record_struct",
    "release",
    "rows",
    "s3_init_sha256",
    "schema_version",
    "source_authority",
    "status",
    "tar_workers",
    "verifier_code",
    "verifier_jobtag",
    "verifier_summary_raw_sha256",
    "verifier_summary_sha256",
    "workers",
}
SUMMARY_KEYS = {
    "blob_count",
    "build_complete_raw_sha256",
    "build_complete_sha256",
    "build_jobtag",
    "build_manifest_raw_sha256",
    "build_manifest_sha256",
    "builder_boot_id_sha256",
    "code",
    "fallbacks",
    "generated_at",
    "integrity_errors",
    "keys_sha256",
    "payload_bytes",
    "physical_hosts_disjoint",
    "rows",
    "schema_version",
    "short_reads",
    "status",
    "verifier_boot_id_sha256",
    "verifier_jobtag",
    "verifier_summary_sha256",
    "verify_approval_raw_sha256",
    "verify_approval_sha256",
    "worker_count",
    "workers",
}
RUNTIME_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "generated_at",
    "release",
    "release_sha256",
    "verification",
    "manifest_sha256",
}
RUNTIME_RELEASE_KEYS = {
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
RUNTIME_REPORT_KEYS = {
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


class ReleaseViewError(RuntimeError):
    """The canonical production release cannot form an exact runtime view."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReleaseViewError(f"value is not canonical-JSON encodable: {exc}") from exc


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def production_json_bytes(value: Any) -> bytes:
    """Production canonical JSON: compact encoding plus exactly one LF."""

    return canonical_json_bytes(value) + b"\n"


def production_json_sha256(value: Any) -> str:
    return hashlib.sha256(production_json_bytes(value)).hexdigest()


def self_hash(value: Mapping[str, Any], field: str) -> str:
    core = dict(value)
    core.pop(field, None)
    return canonical_json_sha256(core)


def production_self_hash(value: Mapping[str, Any], field: str) -> str:
    core = dict(value)
    core.pop(field, None)
    return production_json_sha256(core)


def parse_canonical_json(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseViewError(f"{label} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except ReleaseViewError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseViewError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ReleaseViewError(f"{label} must be one canonical JSON object")
    return value


def parse_production_json(raw: bytes, label: str) -> dict[str, Any]:
    """Parse the production format, rejecting missing/multiple trailing LFs."""

    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ReleaseViewError(f"{label} must end in exactly one LF")
    value = parse_canonical_json(raw[:-1], label)
    if production_json_bytes(value) != raw:
        raise ReleaseViewError(f"{label} production canonical bytes differ")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseViewError(f"{label} must be an object")
    return value


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    result = _mapping(value, label)
    actual = set(result)
    if actual != expected:
        raise ReleaseViewError(
            f"{label} key-set drift: missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )
    return result


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReleaseViewError(f"{label} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ReleaseViewError(f"{label} must be a lowercase SHA-256")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseViewError(f"{label} must be a nonempty string")
    return value


def _jobtag(value: Any, label: str) -> str:
    result = _nonempty_string(value, label)
    if (
        JOBTAG_RE.fullmatch(result) is None
        or result in {".", "..", "latest"}
        or "/" in result
    ):
        raise ReleaseViewError(f"{label} is not a frozen literal jobtag")
    return result


def _absolute_path(value: Any, label: str) -> Path:
    raw = _nonempty_string(value, label)
    path = Path(raw)
    if not path.is_absolute() or str(path) != raw or ".." in path.parts:
        raise ReleaseViewError(f"{label} must be a normalized absolute path")
    return path


def _validate_canonical_catalog(canonical: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    catalog = _exact_keys(
        canonical.get("catalog"),
        {"catalog_hash", "catalog_rows", "gpic", "header", "root"},
        "canonical catalog",
    )
    _sha256(catalog.get("catalog_hash"), "catalog hash")
    catalog_rows = _integer(catalog.get("catalog_rows"), "catalog rows", 1)
    _absolute_path(catalog.get("root"), "catalog root")
    gpic = _exact_keys(
        catalog.get("gpic"),
        {"bytes", "identity", "name", "row_base", "rows", "schema", "sha256"},
        "canonical GPIC catalog file",
    )
    if gpic.get("name") != "gpic.parquet":
        raise ReleaseViewError("canonical GPIC catalog filename drift")
    _integer(gpic.get("bytes"), "GPIC catalog bytes", 1)
    row_base = _integer(gpic.get("row_base"), "GPIC row base")
    rows = _integer(gpic.get("rows"), "GPIC rows", 1)
    _nonempty_string(gpic.get("schema"), "GPIC schema")
    _sha256(gpic.get("sha256"), "GPIC parquet SHA-256")
    if row_base + rows > catalog_rows:
        raise ReleaseViewError("GPIC catalog row interval exceeds catalog rows")
    identity = _exact_keys(
        gpic.get("identity"),
        {"ctime_ns", "dev", "ino", "mtime_ns", "size"},
        "GPIC source identity",
    )
    for key in ("ctime_ns", "dev", "ino", "mtime_ns", "size"):
        _integer(identity.get(key), f"GPIC identity {key}")
    if identity.get("size") != gpic.get("bytes"):
        raise ReleaseViewError("GPIC identity size differs from catalog bytes")
    header = _exact_keys(
        catalog.get("header"),
        {"bytes", "identity", "name", "raw_sha256"},
        "canonical catalog header",
    )
    if header.get("name") != "header.json":
        raise ReleaseViewError("catalog header filename drift")
    _integer(header.get("bytes"), "catalog header bytes", 1)
    _sha256(header.get("raw_sha256"), "catalog header raw SHA-256")
    header_identity = _exact_keys(
        header.get("identity"),
        {"ctime_ns", "dev", "ino", "mtime_ns", "size"},
        "catalog header identity",
    )
    for key in ("ctime_ns", "dev", "ino", "mtime_ns", "size"):
        _integer(header_identity.get(key), f"catalog header identity {key}")
    if header_identity.get("size") != header.get("bytes"):
        raise ReleaseViewError("catalog header identity size differs")
    return catalog, gpic


def validate_canonical_manifest(
    canonical: Mapping[str, Any], raw_bytes: bytes | None = None
) -> Mapping[str, Any]:
    """Strictly validate the frozen production manifest and its self-hash."""

    value = _exact_keys(canonical, CANONICAL_KEYS, "canonical manifest")
    if raw_bytes is not None and production_json_bytes(value) != raw_bytes:
        raise ReleaseViewError("canonical manifest raw bytes are not canonical/exact")
    if (
        value.get("schema_version") != CANONICAL_SCHEMA
        or value.get("status") != CANONICAL_STATUS
        or value.get("publication_semantics")
        != "canonical_manifest_then_fsx_success_then_single_s3_root_success_absolute_last"
    ):
        raise ReleaseViewError("canonical schema/status/publication semantics drift")
    _nonempty_string(value.get("generated_at"), "canonical generated_at")
    buildtag = _jobtag(value.get("build_jobtag"), "canonical build jobtag")
    _jobtag(value.get("verifier_jobtag"), "canonical verifier jobtag")
    if value.get("manifest_sha256") != production_self_hash(value, "manifest_sha256"):
        raise ReleaseViewError("canonical manifest self-hash mismatch")
    for key in (
        "build_complete_raw_sha256",
        "build_complete_sha256",
        "build_manifest_raw_sha256",
        "build_manifest_sha256",
        "partition_plan_raw_sha256",
        "partition_plan_sha256",
        "s3_init_sha256",
        "verifier_summary_raw_sha256",
        "verifier_summary_sha256",
    ):
        _sha256(value.get(key), f"canonical {key}")
    if value.get("record_struct") != RECORD_STRUCT or value.get("record_bytes") != RECORD_BYTES:
        raise ReleaseViewError("canonical dense-index record layout drift")
    if value.get("partition_mode") != "shard_modulo" or value.get("tar_workers") != WORKERS:
        raise ReleaseViewError("canonical partition/tar-worker mode drift")
    _mapping(value.get("source_authority"), "canonical source authority")
    _mapping(value.get("full_tar_mapping"), "canonical full tar mapping")
    _mapping(value.get("verifier_code"), "canonical verifier code")

    catalog, gpic = _validate_canonical_catalog(value)
    rows = _integer(value.get("rows"), "canonical rows", 1)
    if rows != gpic.get("rows"):
        raise ReleaseViewError("canonical rows differ from GPIC catalog rows")
    payload_bytes = _integer(value.get("payload_bytes"), "canonical payload bytes", 1)
    blob_count = _integer(value.get("blob_count"), "canonical blob count", 1)
    dense = _exact_keys(
        value.get("dense_index"), {"bytes", "name", "sha256"}, "dense index binding"
    )
    if dense.get("name") != "dense_index.bin":
        raise ReleaseViewError("dense-index filename drift")
    if dense.get("bytes") != rows * RECORD_BYTES:
        raise ReleaseViewError("dense-index bytes must equal rows*56")
    _sha256(dense.get("sha256"), "dense-index SHA-256")
    limits = _mapping(value.get("limits"), "canonical limits")
    if (
        limits.get("workers") != WORKERS
        or limits.get("gpic_rows") != rows
        or limits.get("dense_index_bytes") != dense.get("bytes")
        or limits.get("record_bytes") != RECORD_BYTES
    ):
        raise ReleaseViewError("canonical production limits do not bind the release")

    release = _exact_keys(value.get("release"), {"storage"}, "canonical release")
    storage = _exact_keys(
        release.get("storage"), {"index_path", "s3_init", "s3_root"}, "canonical storage"
    )
    s3_root = _absolute_path(storage.get("s3_root"), "canonical S3 root")
    if s3_root.name != buildtag:
        raise ReleaseViewError("canonical S3 root basename differs from build jobtag")
    source_index = _absolute_path(storage.get("index_path"), "canonical source index")
    if source_index.name != "dense_index.bin":
        raise ReleaseViewError("canonical source index filename drift")
    s3_init = _exact_keys(
        storage.get("s3_init"), {"bytes", "path", "raw_sha256"}, "canonical S3 init"
    )
    if s3_init.get("path") != "_INIT.json":
        raise ReleaseViewError("canonical S3 init path drift")
    _integer(s3_init.get("bytes"), "canonical S3 init bytes", 1)
    _sha256(s3_init.get("raw_sha256"), "canonical S3 init raw SHA-256")

    raw_blobs = value.get("blobs")
    if not isinstance(raw_blobs, list) or len(raw_blobs) != blob_count:
        raise ReleaseViewError("canonical blob list/count mismatch")
    payload_sum = 0
    segment_sum = 0
    worker_payload = [0] * WORKERS
    worker_rows = [0] * WORKERS
    worker_blobs = [0] * WORKERS
    for position, raw_blob in enumerate(raw_blobs):
        blob = _exact_keys(
            raw_blob,
            {"blob_id", "bytes", "key", "name", "segments", "sequence", "sha256", "worker_id"},
            f"canonical blob {position}",
        )
        blob_id = _integer(blob.get("blob_id"), f"blob {position} id")
        if blob_id != position or blob_id > 0xFFFFFFFF:
            raise ReleaseViewError("canonical blob IDs must be contiguous manifest positions")
        worker_id = _integer(blob.get("worker_id"), f"blob {blob_id} worker")
        sequence = _integer(blob.get("sequence"), f"blob {blob_id} sequence")
        expected_name = f"blob-{blob_id:08d}.bin"
        if (
            worker_id != blob_id % WORKERS
            or sequence != blob_id // WORKERS
            or blob.get("name") != expected_name
        ):
            raise ReleaseViewError(f"canonical blob worker/sequence/name drift at {blob_id}")
        expected_key = str(s3_root / "workers" / f"worker-{worker_id}" / expected_name)
        if blob.get("key") != expected_key:
            raise ReleaseViewError(f"canonical blob key drift at {blob_id}")
        blob_bytes = _integer(blob.get("bytes"), f"blob {blob_id} bytes", 1)
        segments = _integer(blob.get("segments"), f"blob {blob_id} segments", 1)
        _sha256(blob.get("sha256"), f"blob {blob_id} SHA-256")
        payload_sum += blob_bytes
        segment_sum += segments
        worker_payload[worker_id] += blob_bytes
        worker_rows[worker_id] += segments
        worker_blobs[worker_id] += 1
    if payload_sum != payload_bytes or segment_sum != rows:
        raise ReleaseViewError("canonical blob byte/segment closure mismatch")

    canonical_workers = value.get("workers")
    if not isinstance(canonical_workers, list) or len(canonical_workers) != WORKERS:
        raise ReleaseViewError("canonical worker list must contain exactly eight workers")
    for worker_id, raw_worker in enumerate(canonical_workers):
        worker = _mapping(raw_worker, f"canonical worker {worker_id}")
        if (
            worker.get("worker_id") != worker_id
            or worker.get("blob_count") != worker_blobs[worker_id]
            or worker.get("rows") != worker_rows[worker_id]
            or worker.get("payload_bytes") != worker_payload[worker_id]
        ):
            raise ReleaseViewError(f"canonical worker {worker_id} closure mismatch")
    if catalog.get("catalog_hash") is None:  # keeps type narrowing explicit
        raise ReleaseViewError("canonical catalog hash missing")
    return value


def validate_verifier_summary(
    canonical: Mapping[str, Any],
    summary: Mapping[str, Any],
    raw_bytes: bytes | None = None,
) -> Mapping[str, Any]:
    """Validate the production verifier receipt and its canonical binding."""

    canonical_value = validate_canonical_manifest(canonical)
    value = _exact_keys(summary, SUMMARY_KEYS, "verifier summary")
    if raw_bytes is not None and production_json_bytes(value) != raw_bytes:
        raise ReleaseViewError("verifier summary raw bytes are not canonical/exact")
    if value.get("schema_version") != SUMMARY_SCHEMA or value.get("status") != RUNTIME_STATUS:
        raise ReleaseViewError("verifier summary schema/status drift")
    _nonempty_string(value.get("generated_at"), "verifier summary generated_at")
    summary_self = _sha256(value.get("verifier_summary_sha256"), "summary self SHA-256")
    if summary_self != production_self_hash(value, "verifier_summary_sha256"):
        raise ReleaseViewError("verifier summary self-hash mismatch")
    if canonical_value.get("verifier_summary_sha256") != summary_self:
        raise ReleaseViewError("canonical manifest does not bind summary self-hash")
    if raw_bytes is not None and canonical_value.get("verifier_summary_raw_sha256") != hashlib.sha256(raw_bytes).hexdigest():
        raise ReleaseViewError("canonical manifest does not bind summary raw bytes")
    for key in (
        "build_complete_raw_sha256",
        "build_complete_sha256",
        "build_manifest_raw_sha256",
        "build_manifest_sha256",
    ):
        if value.get(key) != canonical_value.get(key):
            raise ReleaseViewError(f"verifier summary/canonical {key} mismatch")
    if (
        value.get("build_jobtag") != canonical_value.get("build_jobtag")
        or value.get("verifier_jobtag") != canonical_value.get("verifier_jobtag")
        or value.get("rows") != canonical_value.get("rows")
        or value.get("payload_bytes") != canonical_value.get("payload_bytes")
        or value.get("blob_count") != canonical_value.get("blob_count")
        or value.get("worker_count") != WORKERS
        or value.get("physical_hosts_disjoint") is not True
        or value.get("integrity_errors") != 0
        or value.get("short_reads") != 0
        or value.get("fallbacks") != 0
        or value.get("code") != canonical_value.get("verifier_code")
    ):
        raise ReleaseViewError("verifier summary proof/identity closure mismatch")
    for key in ("keys_sha256", "verify_approval_raw_sha256", "verify_approval_sha256"):
        _sha256(value.get(key), f"verifier summary {key}")
    expected_keys_sha = production_json_sha256(
        sorted(str(blob["key"]) for blob in canonical_value["blobs"])
    )
    if value.get("keys_sha256") != expected_keys_sha:
        raise ReleaseViewError("verifier summary key-set digest mismatch")
    builder_boots = value.get("builder_boot_id_sha256")
    verifier_boots = value.get("verifier_boot_id_sha256")
    for boots, label in ((builder_boots, "builder"), (verifier_boots, "verifier")):
        if not isinstance(boots, list) or len(boots) != WORKERS:
            raise ReleaseViewError(f"{label} boot set must contain exactly eight hashes")
        for checksum in boots:
            _sha256(checksum, f"{label} boot hash")
        if len(set(boots)) != WORKERS or boots != sorted(boots):
            raise ReleaseViewError(f"{label} boot hashes must be unique and sorted")
    if set(builder_boots) & set(verifier_boots):
        raise ReleaseViewError("builder and verifier physical boot sets overlap")

    workers = value.get("workers")
    if not isinstance(workers, list) or len(workers) != WORKERS:
        raise ReleaseViewError("verifier summary worker list must contain eight workers")
    canonical_workers = canonical_value["workers"]
    for worker_id, raw_worker in enumerate(workers):
        worker = _mapping(raw_worker, f"verifier summary worker {worker_id}")
        canonical_worker = _mapping(canonical_workers[worker_id], f"canonical worker {worker_id}")
        if (
            worker.get("worker_id") != worker_id
            or worker.get("rows") != canonical_worker.get("rows")
            or worker.get("payload_bytes") != canonical_worker.get("payload_bytes")
            or worker.get("blob_count") != canonical_worker.get("blob_count")
        ):
            raise ReleaseViewError(f"verifier summary worker {worker_id} closure mismatch")
        _sha256(worker.get("report_raw_sha256"), f"worker {worker_id} report raw SHA-256")
        _sha256(worker.get("report_sha256"), f"worker {worker_id} report self SHA-256")
    return value


def project_release(canonical: Mapping[str, Any], control_root: str | Path) -> dict[str, Any]:
    """Project a fully validated production canonical into the old release schema."""

    value = validate_canonical_manifest(canonical)
    root = _absolute_path(str(control_root), "runtime compatibility control root")
    buildtag = str(value["build_jobtag"])
    if root.name != buildtag:
        raise ReleaseViewError("runtime compatibility root basename must equal build jobtag")
    catalog = _mapping(value["catalog"], "canonical catalog")
    gpic = _mapping(catalog["gpic"], "canonical GPIC file")
    dense = _mapping(value["dense_index"], "canonical dense index")
    storage = _mapping(_mapping(value["release"], "canonical release")["storage"], "canonical storage")
    projected_blobs = [
        {
            "blob_id": int(blob["blob_id"]),
            "name": (
                f"workers/worker-{int(blob['worker_id'])}/"
                f"blob-{int(blob['blob_id']):08d}.bin"
            ),
            "bytes": int(blob["bytes"]),
            "sha256": str(blob["sha256"]),
        }
        for blob in value["blobs"]
    ]
    release = {
        "backend": BACKEND,
        "catalog": {
            "catalog_hash": catalog["catalog_hash"],
            "catalog_header_sha256": _mapping(catalog["header"], "catalog header")["raw_sha256"],
            "gpic_file": {
                "name": gpic["name"],
                "row_base": gpic["row_base"],
                "rows": gpic["rows"],
                "bytes": gpic["bytes"],
                "sha256": gpic["sha256"],
                "schema": gpic["schema"],
            },
        },
        "layout": {
            "version": LAYOUT_VERSION,
            "record_struct": RECORD_STRUCT,
            "record_bytes": RECORD_BYTES,
            "endianness": "little",
            "index_order": "row_id_ascending",
            "row_id_namespace": "catalog_global",
            "selection_mode": SELECTION_MODE,
            "compression": "none",
            "max_segment_bytes": MAX_SEGMENT_BYTES,
        },
        "storage": {
            "s3_root": storage["s3_root"],
            "index_path": str(root / "dense_index.bin"),
            "s3_init": dict(_mapping(storage["s3_init"], "canonical S3 init")),
        },
        "index": {"bytes": dense["bytes"], "sha256": dense["sha256"]},
        "blobs": projected_blobs,
        "row_count": value["rows"],
        "payload_bytes": value["payload_bytes"],
        "blob_set_sha256": canonical_json_sha256(projected_blobs),
    }
    _exact_keys(release, RUNTIME_RELEASE_KEYS, "projected runtime release")
    return release


def project_verifier_report(
    canonical: Mapping[str, Any],
    summary: Mapping[str, Any],
    control_root: str | Path,
    *,
    summary_raw_bytes: bytes | None = None,
) -> dict[str, Any]:
    value = validate_canonical_manifest(canonical)
    verified = validate_verifier_summary(value, summary, summary_raw_bytes)
    release = project_release(value, control_root)
    gpic = _mapping(_mapping(value["catalog"], "canonical catalog")["gpic"], "canonical GPIC")
    report: dict[str, Any] = {
        "schema_version": VERIFY_SCHEMA,
        "status": RUNTIME_STATUS,
        "generated_at": verified["generated_at"],
        "release_sha256": canonical_json_sha256(release),
        "catalog_hash": _mapping(value["catalog"], "canonical catalog")["catalog_hash"],
        "index_sha256": _mapping(value["dense_index"], "dense index")["sha256"],
        "index_bytes": _mapping(value["dense_index"], "dense index")["bytes"],
        "row_id_start": gpic["row_base"],
        "row_id_stop_exclusive": int(gpic["row_base"]) + int(gpic["rows"]),
        "row_count": value["rows"],
        "blob_count": value["blob_count"],
        "payload_bytes": value["payload_bytes"],
        "full_index_records_verified": value["rows"],
        "full_segment_sha256_verified": True,
        "full_blob_sha256_verified": True,
    }
    report["report_sha256"] = self_hash(report, "report_sha256")
    _exact_keys(report, RUNTIME_REPORT_KEYS, "projected verifier report")
    return report


def project_runtime_manifest(
    canonical: Mapping[str, Any],
    summary: Mapping[str, Any],
    control_root: str | Path,
    *,
    summary_raw_bytes: bytes | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return deterministic ``(manifest, verifier_report)`` old-schema objects."""

    value = validate_canonical_manifest(canonical)
    release = project_release(value, control_root)
    report = project_verifier_report(
        value, summary, control_root, summary_raw_bytes=summary_raw_bytes
    )
    report_raw = canonical_json_bytes(report)
    root = _absolute_path(str(control_root), "runtime compatibility control root")
    manifest: dict[str, Any] = {
        "schema_version": CANONICAL_SCHEMA,
        "status": RUNTIME_STATUS,
        "generated_at": value["generated_at"],
        "release": release,
        "release_sha256": canonical_json_sha256(release),
        "verification": {
            "path": str(root / "verifier_report.json"),
            "bytes": len(report_raw),
            "sha256": hashlib.sha256(report_raw).hexdigest(),
            "schema_version": VERIFY_SCHEMA,
        },
    }
    manifest["manifest_sha256"] = self_hash(manifest, "manifest_sha256")
    _exact_keys(manifest, RUNTIME_MANIFEST_KEYS, "projected runtime manifest")
    return manifest, report


__all__ = [
    "CANONICAL_KEYS",
    "RUNTIME_MANIFEST_KEYS",
    "RUNTIME_RELEASE_KEYS",
    "RUNTIME_REPORT_KEYS",
    "ReleaseViewError",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "parse_canonical_json",
    "parse_production_json",
    "production_json_bytes",
    "production_json_sha256",
    "production_self_hash",
    "project_release",
    "project_runtime_manifest",
    "project_verifier_report",
    "self_hash",
    "validate_canonical_manifest",
    "validate_verifier_summary",
]
