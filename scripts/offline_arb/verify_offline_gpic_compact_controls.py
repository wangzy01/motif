#!/usr/bin/env python3
"""Compact per-node GPIC publication preflight (no full index/blob rehash).

The one-time publisher verifier remains authoritative for payload completeness.
This launch-time check reparses and fingerprints both publication planes, the
verifier receipts, catalog binding, and copied index path/size before GPU
allocation.  Runtime then performs the collective index SHA and per-segment
SHA checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

from training.dataline_offline.gpic_release_view import (
    RUNTIME_MANIFEST_KEYS,
    RUNTIME_RELEASE_KEYS,
    RUNTIME_REPORT_KEYS,
    ReleaseViewError,
    parse_production_json,
    project_release,
    validate_canonical_manifest,
    validate_verifier_summary,
)


SHA_RE = re.compile(r"[0-9a-f]{64}")
SCHEMA = "offline_gpic_blob_compact_preflight_v1"
FSX_VIEW_NAMES = {
    "manifest.json",
    "verifier_report.json",
    "dense_index.bin",
    "canonical_manifest.json",
    "verifier_summary.json",
    "_SUCCESS",
}


class PreflightError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def bound_bytes(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise PreflightError(f"{label} is not a nonempty regular file: {path}")
    identity = ident(before)
    with path.open("rb", buffering=0) as handle:
        if ident(os.fstat(handle.fileno())) != identity:
            raise PreflightError(f"{label} changed while opening: {path}")
        raw = handle.read()
        closed = os.fstat(handle.fileno())
    if ident(closed) != identity or ident(path.lstat()) != identity:
        raise PreflightError(f"{label} changed while reading: {path}")
    return raw, before


def ident(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def json_from(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise PreflightError(f"{label} has duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except PreflightError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} is not a JSON object")
    return value


def sha(path: Path) -> str:
    raw, _metadata = bound_bytes(path, str(path))
    return hashlib.sha256(raw).hexdigest()


def real_dir(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise PreflightError(f"{label} is not an absolute non-symlink directory")
    metadata = path.lstat()
    resolved = path.resolve(strict=True)
    if not stat.S_ISDIR(metadata.st_mode) or not resolved.is_dir():
        raise PreflightError(f"{label} is not a directory: {path}")
    if resolved != path:
        raise PreflightError(f"{label} is not a normalized real path: {path}")
    return resolved


def exact_names(directory: Path, expected: set[str], label: str) -> None:
    actual = {path.name for path in directory.iterdir()}
    if actual != expected:
        raise PreflightError(
            f"{label} exact set drift: missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    raw = canonical(value)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise PreflightError(f"refusing compact preflight output overwrite: {path}")
    with temporary.open("xb", buffering=0) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    if (
        not args.artifact_root.is_absolute()
        or args.artifact_root.is_symlink()
        or args.artifact_root.name != args.buildtag
        or args.buildtag in {"", ".", "..", "latest"}
        or SHA_RE.fullmatch(args.manifest_sha256) is None
    ):
        raise PreflightError("artifact/buildtag/manifest submission binding is malformed")
    root = real_dir(args.artifact_root, "FSx control root")
    exact_names(root, FSX_VIEW_NAMES, "FSx compatibility view")

    # The submitted SHA remains the exact raw identity of the small adapter
    # manifest.  It intentionally is not the production canonical identity.
    manifest_raw, _manifest_metadata = bound_bytes(root / "manifest.json", "FSx manifest")
    if hashlib.sha256(manifest_raw).hexdigest() != args.manifest_sha256:
        raise PreflightError("FSx manifest differs from submitted SHA-256")
    adapter_marker = (args.manifest_sha256 + "\n").encode("ascii")
    if bound_bytes(root / "_SUCCESS", "FSx _SUCCESS")[0] != adapter_marker:
        raise PreflightError("FSx _SUCCESS does not bind adapter manifest raw SHA-256")
    manifest = json_from(manifest_raw, "FSx manifest")
    if canonical(manifest) != manifest_raw:
        raise PreflightError("FSx manifest is not canonical JSON")
    if set(manifest) != RUNTIME_MANIFEST_KEYS:
        raise PreflightError("adapter manifest exact schema key-set drift")
    manifest_core = dict(manifest)
    manifest_self = manifest_core.pop("manifest_sha256", None)
    if (
        SHA_RE.fullmatch(str(manifest_self)) is None
        or manifest_self != hashlib.sha256(canonical(manifest_core)).hexdigest()
    ):
        raise PreflightError("manifest self-hash differs")
    if (
        manifest.get("schema_version") != "offline_gpic_blob_v1"
        or manifest.get("status") != "PASS"
        or not isinstance(manifest.get("generated_at"), str)
        or not manifest["generated_at"]
        or SHA_RE.fullmatch(str(manifest.get("release_sha256", ""))) is None
    ):
        raise PreflightError("manifest schema/status/release identity is malformed")
    release = manifest.get("release")
    if (
        not isinstance(release, dict)
        or set(release) != RUNTIME_RELEASE_KEYS
        or release.get("backend") != "mountpoint_s3"
    ):
        raise PreflightError("manifest backend is not mountpoint_s3")
    if manifest.get("release_sha256") != hashlib.sha256(canonical(release)).hexdigest():
        raise PreflightError("manifest release SHA-256 differs")
    blobs = release.get("blobs")
    if not isinstance(blobs, list) or not blobs or len(blobs) > 1024:
        raise PreflightError("manifest blob set is empty or exceeds fd_cache_size=1024")
    storage = release.get("storage")
    if not isinstance(storage, dict) or set(storage) != {
        "s3_root",
        "index_path",
        "s3_init",
    }:
        raise PreflightError("manifest storage binding is malformed")

    # The compatibility view must carry byte-exact local copies of both
    # production controls.  Their JSON and self-hashes use the production
    # convention (compact JSON plus exactly one trailing LF).
    canonical_path = root / "canonical_manifest.json"
    canonical_raw, _canonical_metadata = bound_bytes(
        canonical_path, "FSx production canonical manifest"
    )
    try:
        production = parse_production_json(
            canonical_raw, "FSx production canonical manifest"
        )
        validate_canonical_manifest(production, canonical_raw)
    except ReleaseViewError as exc:
        raise PreflightError(f"invalid FSx production canonical manifest: {exc}") from exc
    production_self = str(production["manifest_sha256"])
    if (
        production.get("build_jobtag") != args.buildtag
        or canonical_path.parent != root
    ):
        raise PreflightError("production canonical buildtag/view binding differs")

    summary_path = root / "verifier_summary.json"
    summary_raw, _summary_metadata = bound_bytes(
        summary_path, "FSx production verifier summary"
    )
    try:
        summary = parse_production_json(
            summary_raw, "FSx production verifier summary"
        )
        validate_verifier_summary(production, summary, summary_raw)
    except ReleaseViewError as exc:
        raise PreflightError(f"invalid FSx production verifier summary: {exc}") from exc

    try:
        projected_release = project_release(production, root)
    except (ReleaseViewError, ValueError, TypeError, KeyError) as exc:
        raise PreflightError(f"cannot project production canonical release: {exc}") from exc
    if projected_release != release:
        raise PreflightError(
            "adapter release differs from exact production canonical projection"
        )

    production_storage = production["release"]["storage"]
    if (
        production_storage.get("s3_root") != storage.get("s3_root")
        or production_storage.get("s3_init") != storage.get("s3_init")
    ):
        raise PreflightError("adapter and production S3 storage bindings differ")
    s3_root_raw = production_storage.get("s3_root")
    if not isinstance(s3_root_raw, str):
        raise PreflightError("production canonical S3 root is malformed")
    s3_root = real_dir(Path(s3_root_raw), "S3 root")
    if str(s3_root) != s3_root_raw:
        raise PreflightError("production canonical S3 root is not normalized")
    if s3_root.name != args.buildtag:
        raise PreflightError("S3 root does not bind buildtag")
    exact_names(
        s3_root,
        {
            "_INIT.json",
            "_SUCCESS",
            "controls",
            "workers",
        },
        "S3 root",
    )
    controls = real_dir(s3_root / "controls", "S3 controls")
    exact_names(controls, {"manifest.json"}, "S3 controls")
    s3_manifest_raw, _s3_manifest_metadata = bound_bytes(
        controls / "manifest.json", "S3 production canonical manifest"
    )
    try:
        s3_production = parse_production_json(
            s3_manifest_raw, "S3 production canonical manifest"
        )
        validate_canonical_manifest(s3_production, s3_manifest_raw)
    except ReleaseViewError as exc:
        raise PreflightError(f"invalid S3 production canonical manifest: {exc}") from exc
    if (
        s3_manifest_raw != canonical_raw
        or s3_production != production
        or s3_production.get("build_jobtag") != args.buildtag
        or s3_production["release"]["storage"].get("s3_root") != str(s3_root)
    ):
        raise PreflightError("FSx and S3 production canonical manifests differ")
    canonical_marker = (production_self + "\n").encode("ascii")
    if bound_bytes(s3_root / "_SUCCESS", "S3 _SUCCESS")[0] != canonical_marker:
        raise PreflightError(
            "S3 _SUCCESS does not bind production canonical self-hash"
        )
    init_binding = storage.get("s3_init")
    if (
        not isinstance(init_binding, dict)
        or set(init_binding) != {"path", "bytes", "raw_sha256"}
        or init_binding.get("path") != "_INIT.json"
    ):
        raise PreflightError("S3 init binding is malformed")
    init_raw, _init_metadata = bound_bytes(s3_root / "_INIT.json", "S3 init")
    if (
        len(init_raw) != init_binding.get("bytes")
        or hashlib.sha256(init_raw).hexdigest() != init_binding.get("raw_sha256")
    ):
        raise PreflightError("S3 init live fingerprint differs")
    workers = real_dir(s3_root / "workers", "S3 workers")
    expected_workers = {f"worker-{index}" for index in range(8)}
    exact_names(workers, expected_workers, "S3 worker directory")
    for worker in expected_workers:
        real_dir(workers / worker, f"S3 {worker}")

    if not args.output.is_absolute() or args.output.is_symlink():
        raise PreflightError("compact preflight output path must be absolute/non-symlink")
    output_parent = real_dir(args.output.parent, "compact preflight output parent")
    output = output_parent / args.output.name
    if output != args.output or root == output_parent or root in output_parent.parents:
        raise PreflightError("compact preflight output must be outside immutable FSx view")
    if s3_root == output_parent or s3_root in output_parent.parents:
        raise PreflightError("compact preflight output must not write below S3 root")

    index_raw = storage.get("index_path")
    index = Path(str(index_raw))
    index_meta = index.lstat()
    index_binding = release.get("index")
    if (
        not isinstance(index_raw, str)
        or index != root / "dense_index.bin"
        or str(index) != index_raw
        or index.is_symlink()
        or not stat.S_ISREG(index_meta.st_mode)
        or not isinstance(index_binding, dict)
        or set(index_binding) != {"bytes", "sha256"}
        or index_meta.st_size != index_binding.get("bytes")
        or SHA_RE.fullmatch(str(index_binding.get("sha256", ""))) is None
    ):
        raise PreflightError("copied index direct-child/size/manifest binding differs")
    verification = manifest.get("verification")
    if not isinstance(verification, dict) or set(verification) != {
        "path",
        "bytes",
        "sha256",
        "schema_version",
    }:
        raise PreflightError("verifier receipt binding is malformed")
    report_path_raw = verification.get("path")
    report_path = Path(str(report_path_raw))
    if (
        not isinstance(report_path_raw, str)
        or report_path != root / "verifier_report.json"
        or str(report_path) != report_path_raw
    ):
        raise PreflightError("verifier receipt is not the exact view direct child")
    report_raw, _report_metadata = bound_bytes(report_path, "verifier receipt")
    if (
        verification.get("schema_version") != "offline_gpic_blob_verify_v1"
        or len(report_raw) != verification.get("bytes")
        or hashlib.sha256(report_raw).hexdigest() != verification.get("sha256")
    ):
        raise PreflightError("verifier receipt live fingerprint differs")
    report = json_from(report_raw, "verifier receipt")
    if canonical(report) != report_raw:
        raise PreflightError("verifier receipt is not canonical JSON")
    if set(report) != RUNTIME_REPORT_KEYS:
        raise PreflightError("verifier receipt exact schema key-set drift")
    report_core = dict(report)
    report_self = report_core.pop("report_sha256", None)
    if report_self != hashlib.sha256(canonical(report_core)).hexdigest():
        raise PreflightError("verifier receipt self-hash differs")
    release_catalog = release["catalog"]
    release_gpic = release_catalog["gpic_file"]
    expected_report = {
        "schema_version": "offline_gpic_blob_verify_v1",
        "status": "PASS",
        "release_sha256": manifest.get("release_sha256"),
        "catalog_hash": release_catalog["catalog_hash"],
        "index_sha256": index_binding["sha256"],
        "index_bytes": index_binding["bytes"],
        "row_id_start": release_gpic["row_base"],
        "row_id_stop_exclusive": release_gpic["row_base"] + release_gpic["rows"],
        "row_count": release.get("row_count"),
        "blob_count": len(blobs),
        "payload_bytes": release.get("payload_bytes"),
        "full_index_records_verified": release.get("row_count"),
        "full_segment_sha256_verified": True,
        "full_blob_sha256_verified": True,
    }
    if (
        not isinstance(report.get("generated_at"), str)
        or not report["generated_at"]
        or any(report.get(key) != expected for key, expected in expected_report.items())
    ):
        raise PreflightError("verifier receipt does not prove full publication")

    catalog_raw, _catalog_metadata = bound_bytes(
        args.catalog_root / "header.json", "catalog header"
    )
    catalog = json_from(catalog_raw, "catalog header")
    manifest_catalog = release.get("catalog")
    if (
        not isinstance(manifest_catalog, dict)
        or manifest_catalog.get("catalog_hash") != catalog.get("catalog_hash")
        or manifest_catalog.get("catalog_header_sha256")
        != hashlib.sha256(catalog_raw).hexdigest()
    ):
        raise PreflightError("catalog header differs from artifact release")
    domain_order = catalog.get("domain_order")
    live_files = catalog.get("files")
    if not isinstance(domain_order, list) or not isinstance(live_files, dict):
        raise PreflightError("catalog domain/file structure is malformed")
    row_base = 0
    for domain in domain_order:
        info = live_files.get(domain)
        if not isinstance(info, dict) or type(info.get("rows")) is not int:
            raise PreflightError(f"catalog rows missing for {domain!r}")
        if domain == "gpic":
            expected_gpic = {
                "name": info.get("name"),
                "row_base": row_base,
                "rows": info.get("rows"),
                "bytes": info.get("bytes"),
                "sha256": info.get("sha256"),
                "schema": info.get("schema"),
            }
            if manifest_catalog.get("gpic_file") != expected_gpic:
                raise PreflightError("live GPIC catalog file binding differs from release")
            break
        row_base += int(info["rows"])
    else:
        raise PreflightError("catalog domain_order has no GPIC domain")
    bindings = {
        "base_config_sha256": sha(args.config),
        "plan_header_sha256": sha(args.plan_root / "header.json"),
        "catalog_header_sha256": hashlib.sha256(catalog_raw).hexdigest(),
        "manifest_sha256": args.manifest_sha256,
        "adapter_manifest_self_sha256": str(manifest_self),
        "adapter_release_sha256": manifest["release_sha256"],
        "production_canonical_raw_sha256": hashlib.sha256(canonical_raw).hexdigest(),
        "production_canonical_self_sha256": production_self,
        "production_verifier_summary_raw_sha256": hashlib.sha256(summary_raw).hexdigest(),
        "production_verifier_summary_self_sha256": summary[
            "verifier_summary_sha256"
        ],
        "job_yaml_sha256": sha(args.job_yaml),
        "compact_preflight_sha256": sha(Path(__file__).resolve()),
    }
    value = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "jobtag": args.jobtag,
        "node_rank": args.node_rank,
        "host": os.uname().nodename,
        "artifact_root": str(root),
        "buildtag": args.buildtag,
        "manifest_sha256": args.manifest_sha256,
        "release_sha256": manifest["release_sha256"],
        "adapter_manifest": {
            "path": str(root / "manifest.json"),
            "raw_sha256": args.manifest_sha256,
            "self_sha256": manifest_self,
            "release_sha256": manifest["release_sha256"],
        },
        "production_canonical": {
            "path": str(canonical_path),
            "raw_sha256": hashlib.sha256(canonical_raw).hexdigest(),
            "self_sha256": production_self,
            "build_jobtag": production["build_jobtag"],
            "verifier_jobtag": production["verifier_jobtag"],
            "s3_raw_sha256": hashlib.sha256(s3_manifest_raw).hexdigest(),
        },
        "production_verifier_summary": {
            "path": str(summary_path),
            "raw_sha256": hashlib.sha256(summary_raw).hexdigest(),
            "self_sha256": summary["verifier_summary_sha256"],
            "build_jobtag": summary["build_jobtag"],
            "verifier_jobtag": summary["verifier_jobtag"],
        },
        "backend": "mountpoint_s3",
        "s3_root": str(s3_root),
        "index": {
            "path": str(index.resolve(strict=True)),
            "bytes": index_meta.st_size,
            "declared_sha256": index_binding["sha256"],
            "inode": index_meta.st_ino,
            "device": index_meta.st_dev,
            "content_hash_deferred": "runtime rank0 collective verification",
        },
        "verifier_receipt_sha256": hashlib.sha256(report_raw).hexdigest(),
        "manifest_blob_count": len(blobs),
        "fd_cache_size": 1024,
        "full_publication_verified_once": True,
        "bindings": bindings,
    }
    atomic_json(output, value)
    print(
        f"GPIC_BLOB_COMPACT_PREFLIGHT_PASS node={args.node_rank} "
        f"adapter_manifest={args.manifest_sha256} "
        f"production_manifest={production_self} index_bytes={index_meta.st_size}"
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--artifact-root", type=Path, required=True)
    value.add_argument("--buildtag", required=True)
    value.add_argument("--manifest-sha256", required=True)
    value.add_argument("--catalog-root", type=Path, required=True)
    value.add_argument("--plan-root", type=Path, required=True)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--job-yaml", type=Path, required=True)
    value.add_argument("--jobtag", required=True)
    value.add_argument("--node-rank", type=int, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except (PreflightError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
