#!/usr/bin/env python3
"""Focused positive/negative contract test for the compact GPIC preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.offline_arb import verify_offline_gpic_compact_controls as compact
from training.dataline_offline.gpic_release_view import (
    canonical_json_bytes,
    production_json_bytes,
    production_json_sha256,
    production_self_hash,
    project_runtime_manifest,
)


BUILDTAG = "offline-gpic-compact-static-build-v1"
VERIFIER_JOBTAG = "offline-gpic-compact-static-verify-v1"
GENERATED_AT = "2026-07-12T00:00:00Z"


def digest(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def fixture(base: Path) -> tuple[argparse.Namespace, dict[str, Any]]:
    view = base / "view" / BUILDTAG
    s3_root = base / "s3" / BUILDTAG
    catalog_root = base / "catalog"
    plan_root = base / "plan"
    output_root = base / "output"
    for path in (view, catalog_root, plan_root, output_root):
        path.mkdir(parents=True, exist_ok=False)
    for worker_id in range(8):
        (s3_root / "workers" / f"worker-{worker_id}").mkdir(
            parents=True, exist_ok=False
        )

    init_raw = b'{"schema_version":"static_s3_init_v1"}\n'
    write(s3_root / "_INIT.json", init_raw)
    init_binding = {
        "path": "_INIT.json",
        "bytes": len(init_raw),
        "raw_sha256": digest(init_raw),
    }

    gpic_rows = 8
    dense_raw = b"D" * (gpic_rows * 56)
    catalog_file = {
        "name": "gpic.parquet",
        "rows": gpic_rows,
        "bytes": 123,
        "sha256": digest("gpic-parquet"),
        "schema": "static-gpic-schema-v1",
    }
    catalog_header = {
        "catalog_hash": digest("catalog"),
        "domain_order": ["gpic"],
        "files": {"gpic": catalog_file},
    }
    catalog_raw = canonical_json_bytes(catalog_header)
    write(catalog_root / "header.json", catalog_raw)

    blobs: list[dict[str, Any]] = []
    canonical_workers: list[dict[str, Any]] = []
    summary_workers: list[dict[str, Any]] = []
    for blob_id in range(8):
        blob_bytes = 100 + blob_id
        name = f"blob-{blob_id:08d}.bin"
        blobs.append(
            {
                "blob_id": blob_id,
                "bytes": blob_bytes,
                "key": str(s3_root / "workers" / f"worker-{blob_id}" / name),
                "name": name,
                "segments": 1,
                "sequence": 0,
                "sha256": digest(f"blob-{blob_id}"),
                "worker_id": blob_id,
            }
        )
        closure = {
            "worker_id": blob_id,
            "rows": 1,
            "payload_bytes": blob_bytes,
            "blob_count": 1,
        }
        canonical_workers.append(dict(closure))
        summary_workers.append(
            {
                **closure,
                "report_raw_sha256": digest(f"report-raw-{blob_id}"),
                "report_sha256": digest(f"report-self-{blob_id}"),
            }
        )

    verifier_code = {"runner": digest("verifier-runner")}
    common_bindings = {
        "build_manifest_raw_sha256": digest("build-manifest-raw"),
        "build_manifest_sha256": digest("build-manifest-self"),
        "build_complete_raw_sha256": digest("build-complete-raw"),
        "build_complete_sha256": digest("build-complete-self"),
    }
    builder_boots = sorted(digest(f"builder-boot-{index}") for index in range(8))
    verifier_boots = sorted(digest(f"verifier-boot-{index}") for index in range(8))
    summary: dict[str, Any] = {
        "blob_count": len(blobs),
        **common_bindings,
        "build_jobtag": BUILDTAG,
        "builder_boot_id_sha256": builder_boots,
        "code": verifier_code,
        "fallbacks": 0,
        "generated_at": GENERATED_AT,
        "integrity_errors": 0,
        "keys_sha256": production_json_sha256(sorted(blob["key"] for blob in blobs)),
        "payload_bytes": sum(blob["bytes"] for blob in blobs),
        "physical_hosts_disjoint": True,
        "rows": gpic_rows,
        "schema_version": "offline_gpic_s3_production_verifier_summary_v1",
        "short_reads": 0,
        "status": "PASS",
        "verifier_boot_id_sha256": verifier_boots,
        "verifier_jobtag": VERIFIER_JOBTAG,
        "verify_approval_raw_sha256": digest("approval-raw"),
        "verify_approval_sha256": digest("approval-self"),
        "worker_count": 8,
        "workers": summary_workers,
    }
    summary["verifier_summary_sha256"] = production_self_hash(
        summary, "verifier_summary_sha256"
    )
    summary_raw = production_json_bytes(summary)

    identity = {"ctime_ns": 1, "dev": 1, "ino": 1, "mtime_ns": 1, "size": 123}
    header_identity = {
        "ctime_ns": 1,
        "dev": 1,
        "ino": 2,
        "mtime_ns": 1,
        "size": len(catalog_raw),
    }
    canonical: dict[str, Any] = {
        "blob_count": len(blobs),
        "blobs": blobs,
        **common_bindings,
        "build_jobtag": BUILDTAG,
        "catalog": {
            "catalog_hash": catalog_header["catalog_hash"],
            "catalog_rows": gpic_rows,
            "gpic": {
                **catalog_file,
                "identity": identity,
                "row_base": 0,
            },
            "header": {
                "bytes": len(catalog_raw),
                "identity": header_identity,
                "name": "header.json",
                "raw_sha256": digest(catalog_raw),
            },
            "root": str(catalog_root),
        },
        "dense_index": {
            "bytes": len(dense_raw),
            "name": "dense_index.bin",
            "sha256": digest(dense_raw),
        },
        "full_tar_mapping": {},
        "generated_at": GENERATED_AT,
        "limits": {
            "workers": 8,
            "gpic_rows": gpic_rows,
            "dense_index_bytes": len(dense_raw),
            "record_bytes": 56,
        },
        "partition_mode": "shard_modulo",
        "partition_plan_raw_sha256": digest("partition-plan-raw"),
        "partition_plan_sha256": digest("partition-plan-self"),
        "payload_bytes": summary["payload_bytes"],
        "publication_semantics": (
            "canonical_manifest_then_fsx_success_then_single_s3_root_success_absolute_last"
        ),
        "record_bytes": 56,
        "record_struct": "<qIQI32s",
        "release": {
            "storage": {
                "index_path": str(base / "production-source" / "dense_index.bin"),
                "s3_init": init_binding,
                "s3_root": str(s3_root),
            }
        },
        "rows": gpic_rows,
        "s3_init_sha256": digest("s3-init-self"),
        "schema_version": "offline_gpic_blob_v1",
        "source_authority": {},
        "status": "PRODUCTION_VERIFIED",
        "tar_workers": 8,
        "verifier_code": verifier_code,
        "verifier_jobtag": VERIFIER_JOBTAG,
        "verifier_summary_raw_sha256": digest(summary_raw),
        "verifier_summary_sha256": summary["verifier_summary_sha256"],
        "workers": canonical_workers,
    }
    canonical["manifest_sha256"] = production_self_hash(
        canonical, "manifest_sha256"
    )
    canonical_raw = production_json_bytes(canonical)

    adapter, report = project_runtime_manifest(
        canonical,
        summary,
        view,
        summary_raw_bytes=summary_raw,
    )
    adapter_raw = canonical_json_bytes(adapter)
    report_raw = canonical_json_bytes(report)
    adapter_raw_sha = digest(adapter_raw)
    write(view / "manifest.json", adapter_raw)
    write(view / "verifier_report.json", report_raw)
    write(view / "dense_index.bin", dense_raw)
    write(view / "canonical_manifest.json", canonical_raw)
    write(view / "verifier_summary.json", summary_raw)
    write(view / "_SUCCESS", (adapter_raw_sha + "\n").encode("ascii"))

    write(s3_root / "controls" / "manifest.json", canonical_raw)
    write(
        s3_root / "_SUCCESS",
        (str(canonical["manifest_sha256"]) + "\n").encode("ascii"),
    )
    write(plan_root / "header.json", b'{"static":true}')
    config = base / "config.yaml"
    job_yaml = base / "job.yaml"
    write(config, b"static: true\n")
    write(job_yaml, b"static: true\n")
    args = argparse.Namespace(
        artifact_root=view,
        buildtag=BUILDTAG,
        manifest_sha256=adapter_raw_sha,
        catalog_root=catalog_root,
        plan_root=plan_root,
        config=config,
        job_yaml=job_yaml,
        jobtag="offline-gpic-compact-static-run-v1",
        node_rank=0,
        output=output_root / "preflight.json",
    )
    return args, canonical


def expect_failure(args: argparse.Namespace, needle: str) -> None:
    try:
        compact.run(args)
    except compact.PreflightError as exc:
        assert needle in str(exc), (needle, str(exc))
    else:
        raise AssertionError(f"compact preflight unexpectedly accepted {needle}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="offline-gpic-compact-static-") as raw:
        base = Path(raw).resolve()

        positive_args, positive_canonical = fixture(base / "positive")
        compact.run(positive_args)
        result = json.loads(positive_args.output.read_bytes())
        assert result["status"] == "PASS"
        assert result["adapter_manifest"]["raw_sha256"] == positive_args.manifest_sha256
        assert (
            result["production_canonical"]["self_sha256"]
            == positive_canonical["manifest_sha256"]
        )

        drift_args, drift_canonical = fixture(base / "canonical-blob-drift")
        drift_canonical["blobs"][0]["sha256"] = digest("canonical-blob-drift")
        drift_canonical["manifest_sha256"] = production_self_hash(
            drift_canonical, "manifest_sha256"
        )
        drift_raw = production_json_bytes(drift_canonical)
        write(drift_args.artifact_root / "canonical_manifest.json", drift_raw)
        s3_root = Path(drift_canonical["release"]["storage"]["s3_root"])
        write(s3_root / "controls" / "manifest.json", drift_raw)
        write(
            s3_root / "_SUCCESS",
            (str(drift_canonical["manifest_sha256"]) + "\n").encode("ascii"),
        )
        expect_failure(
            drift_args,
            "adapter release differs from exact production canonical projection",
        )

        marker_args, marker_canonical = fixture(base / "marker-drift")
        marker_s3_root = Path(marker_canonical["release"]["storage"]["s3_root"])
        write(marker_s3_root / "_SUCCESS", (digest("wrong-marker") + "\n").encode())
        expect_failure(
            marker_args,
            "S3 _SUCCESS does not bind production canonical self-hash",
        )

    print(
        "OFFLINE_GPIC_COMPACT_CONTROLS_STATIC_PASS "
        "exact_six=1 adapter_raw_marker=1 canonical_blob_drift_rejected=1 "
        "canonical_marker_drift_rejected=1 s3_payload_reads=0 s3_writes=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
