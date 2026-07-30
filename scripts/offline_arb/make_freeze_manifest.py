#!/usr/bin/env python3
"""Create and verify the immutable OFFLINE_ARB_V1 FORMAL freeze.

The default mode creates freeze v3 from one closed filesystem snapshot.  The
read-only modes are used by GPUVAL and FORMAL startup:

* ``--verify-sidecar-control`` validates only the small immutable controls;
* ``--verify-sidecar`` hashes the exact five-domain sidecar and validates every
  parquet footer without decoding caption values; and
* ``--verify-freeze`` hashes every frozen semantic input and compares it with
  freeze v3.

Every expensive path is intended for a 16-CPU JCC worker.  Five-component
fingerprints (device, inode, size, mtime, ctime) are captured around each read
and uniformly restated after all hashes, control reparses and exact-set checks.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import threading
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from scripts.offline_arb.caption_sidecar_contract import (
    BUILDER_BUILD_ID,
    CAPTION_COLUMNS,
    CAPTION_SCHEMA_CONTRACT_SHA256,
    CAPTION_SCHEMAS,
    DOMAINS,
    FILE_ROWS,
    ROW_GROUP_ROWS,
    ROW_ORDER,
    SCHEMA_VERSION,
    SEMANTIC_DIGEST_ALGORITHM,
    canonical_json_sha256,
    header_self_hash,
    sidecar_schema,
)


REPO = Path("/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop")
BASE = Path("/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop")
OUT = BASE / "offline_arb_plans_v1" / "FREEZE_MANIFEST.json"
PLAN = BASE / "offline_arb_plans_v1/w64_prod/plan"
CATALOG = BASE / "offline_arb_catalog_v1"
SAMPLING = Path("/fsx/evan/mot_code/data_analysis/configs/t5_sampling_redesign_v1.json")
ANIME_FREQ = BASE / "offline_arb_animefreq_v1"
SIDECAR = BASE / "offline_arb_caption_sidecar_v1"
DEFAULT_CONFIG_REL = Path("configs/stage256_offline_exact_v1.yaml")
CONTRACT_KEYS = (
    "stage",
    "image_size",
    "eligibility",
    "requires_nl",
    "anime_nl_filter_applied",
)
LEGACY_STAGE256_ASSIGNMENT = (
    "10bd55e0a37aacef35583c91846984171c6ebdfa88ee8351087bb937cc498cc4"
)
STAGE256_FREEZE_SCHEMA = "offline_arb_freeze_v3_caption_sidecar"
STAGE512A_FREEZE_SCHEMA = "offline_arb_freeze_v4_stage512a_reviewed_exceptions"
STAGE512A_EXCEPTION_SCHEMA = (
    "offline_arb_stage512a_w64_reviewed_exception_gate_v1"
)
STAGE512A_EXCEPTION_POLICY = "reviewed_exact_stage512a_w64_exceptions_v1"
STAGE512A_EXCEPTION_COUNTS = {
    "monet_empty_prompt_rows": 6,
    "monet_empty_prompt_formal_exposures": 19,
    "monet_empty_prompt_horizon_exposures": 20,
    "textatlas_raw_candidate_rows": 43,
    "textatlas_raw_candidate_formal_exposures": 134,
    "textatlas_raw_candidate_horizon_exposures": 137,
    "textatlas_production_decode_success_rows": 40,
    "textatlas_production_decode_success_formal_exposures": 125,
    "textatlas_production_decode_success_horizon_exposures": 127,
    "textatlas_repaired_rows": 3,
    "textatlas_repaired_formal_exposures": 9,
    "textatlas_repaired_horizon_exposures": 10,
}
STAGE512A_EXCEPTION_RUNTIME_SOURCES = {
    "offline_dataset": "training/dataline_offline/offline_dataset.py",
    "caption_exceptions": "training/dataline_offline/caption_exceptions_v1.py",
    "image_exceptions": "training/dataline_offline/image_exceptions_v1.py",
    "image_fetch": "training/dataline/image_fetch.py",
    "targeted_test": (
        "scripts/offline_arb/test_stage512a_w64_reviewed_exceptions.py"
    ),
}
STAGE512A_EXCEPTION_PRODUCER_SOURCES = {
    "template": "jobs/offline_arb_stage512a_w64_reviewed_exception_gate.yaml",
    "finalizer": (
        "scripts/offline_arb/finalize_stage512a_w64_exception_gate.py"
    ),
    "immutable_evidence_helper": "scripts/offline_arb/immutable_review_evidence.py",
    "renderer": "scripts/offline_arb/render_stage512a_write_job.py",
}
STAGE512A_EXCEPTION_PRODUCER_IMAGE = (
    "471112743779.dkr.ecr.us-west-2.amazonaws.com/"
    "autobuild:py312-torch211-cu129"
)
STAGE512A_REVIEWED_GATE_PATH = (
    BASE
    / "offline_arb_exceptions_stage512a_w64_reviewed_v1/FORMAL_PASS.json"
)
STAGE512A_REVIEWED_GATE_SHA256 = (
    "c3bae0569ce7b9ca0de824f683a8adfd5011d0d0e0088fe19d36b61c70a1829c"
)
STAGE512A_REVIEWED_FREEZE_BASENAME = (
    "FREEZE_MANIFEST_STAGE512A_W64_REVIEWED_EXCEPTIONS_V1.json"
)
STAGE512A_REVIEWED_FREEZE_PATH = (
    BASE
    / "offline_arb_plans_stage512a_w64_v2"
    / STAGE512A_REVIEWED_FREEZE_BASENAME
)


def validate_stage512a_reviewed_gate_storage(path: Path) -> None:
    """Require the one canonical sealed gate namespace, not a copied PASS."""

    if path != STAGE512A_REVIEWED_GATE_PATH:
        raise ValueError(
            "reviewed_exception_gate must use the exact canonical path: "
            f"{STAGE512A_REVIEWED_GATE_PATH}"
        )
    try:
        file_meta = path.lstat()
        parent_meta = path.parent.lstat()
        canonical_parent = path.parent.resolve(strict=True)
        members = {item.name for item in path.parent.iterdir()}
    except OSError as exc:
        raise ValueError("reviewed_exception_gate sealed namespace is unreadable") from exc
    if (
        not stat.S_ISREG(file_meta.st_mode)
        or file_meta.st_nlink != 1
        or stat.S_IMODE(file_meta.st_mode) != 0o444
    ):
        raise ValueError("reviewed_exception_gate must be one regular 0444 inode")
    if (
        not stat.S_ISDIR(parent_meta.st_mode)
        or canonical_parent != path.parent
        or stat.S_IMODE(parent_meta.st_mode) != 0o555
        or members != {path.name}
    ):
        raise ValueError(
            "reviewed_exception_gate parent must be canonical 0555 with exact members"
        )


def validate_stage512a_reviewed_freeze_storage(
    path: Path, *, require_exists: bool
) -> None:
    """Reject alternate/copy freeze paths and writable released manifests."""

    if path != STAGE512A_REVIEWED_FREEZE_PATH:
        raise ValueError(
            "stage512a reviewed freeze must use the exact canonical path: "
            f"{STAGE512A_REVIEWED_FREEZE_PATH}"
        )
    try:
        parent_meta = path.parent.lstat()
        canonical_parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError("stage512a reviewed freeze parent is unreadable") from exc
    if not stat.S_ISDIR(parent_meta.st_mode) or canonical_parent != path.parent:
        raise ValueError("stage512a reviewed freeze parent must be canonical and real")
    try:
        file_meta = path.lstat()
    except FileNotFoundError:
        if require_exists:
            raise ValueError("stage512a reviewed freeze is missing")
        return
    if not require_exists:
        raise ValueError("stage512a reviewed freeze output already exists")
    if (
        not stat.S_ISREG(file_meta.st_mode)
        or file_meta.st_nlink != 1
        or stat.S_IMODE(file_meta.st_mode) != 0o444
    ):
        raise ValueError("stage512a reviewed freeze must be one regular 0444 inode")

SOURCE_TREES = ("training", "networks")
SOURCE_EXPLICIT = (
    "train.py",
    "configs/stage256_offline_exact_v1.yaml",
    "jobs/offline_exact_256_2x8_gpuval.yaml",
    "jobs/offline_exact_256_8x8_FORMAL.yaml",
    "jobs/offline_arb_freeze_v3.yaml",
    "jobs/offline_arb_w64_raw_supply_gate.yaml",
    "jobs/offline_arb_w64_raw_supply_analyze.yaml",
    "scripts/offline_arb/make_freeze_manifest.py",
    "scripts/offline_arb/verify_offline_gpic_compact_controls.py",
    "scripts/offline_arb/run_w64_raw_supply_gate.py",
    "scripts/offline_arb/analyze_w64_raw_supply_gate.py",
    "scripts/offline_arb/caption_sidecar_contract.py",
    "scripts/offline_arb/build_caption_sidecar.py",
    "scripts/offline_arb/verify_caption_sidecar.py",
    "scripts/offline_arb/test_caption_sidecar_runtime.py",
    "training/dataline_offline/caption_sidecar.py",
    "training/dataline_offline/offline_dataset.py",
    "training/config_schema.py",
)

# Stage256 is a released legacy freeze contract: SOURCE_EXPLICIT and its
# manifest discovery payload above must stay byte-for-byte stable.  Stage512a
# uses a separate fail-closed source profile so it cannot silently freeze the
# stage256 launch/gate set.
SOURCE_EXPLICIT_STAGE512A = (
    "train.py",
    "configs/stage512a_offline_exact_w64_v1.yaml",
    "configs/stage256_offline_exact_v1.yaml",
    "jobs/offline_arb_phase5_catalog_stage512a.yaml",
    "jobs/offline_arb_phase5_plans_stage512a.yaml",
    "jobs/offline_arb_animefreq_stage512a.yaml",
    "jobs/offline_arb_caption_sidecar_build_stage512a.yaml",
    "jobs/offline_arb_exceptions_audit_stage512a_w64.yaml",
    "jobs/offline_arb_textatlas_catalog_evidence_rebind_w64_TEMPLATE.yaml",
    "jobs/offline_arb_stage256_completion_gate_for_stage512a.yaml",
    "jobs/offline_arb_stage512a_w64_reviewed_exception_gate.yaml",
    "jobs/offline_arb_freeze_v4_stage512a_w64_reviewed.yaml",
    "jobs/offline_exact_512a_2x8_GPUVAL_TEMPLATE.yaml",
    "jobs/offline_exact_512a_8x8_FORMAL_TEMPLATE.yaml",
    "scripts/offline_arb/make_freeze_manifest.py",
    "scripts/offline_arb/test_freeze_stage_source_profiles.py",
    "scripts/offline_arb/phase5_build_catalog.py",
    "scripts/offline_arb/make_planner_input.py",
    "scripts/offline_arb/build_plan.py",
    "scripts/offline_arb/validate_plan.py",
    "scripts/offline_arb/precompute_anime_freq.py",
    "scripts/offline_arb/test_animefreq_input_contract.py",
    "scripts/offline_arb/render_stage512a_write_job.py",
    "scripts/offline_arb/test_stage512a_write_job_renderer.py",
    "scripts/offline_arb/caption_sidecar_contract.py",
    "scripts/offline_arb/build_caption_sidecar.py",
    "scripts/offline_arb/verify_caption_sidecar.py",
    "scripts/offline_arb/test_caption_sidecar_runtime.py",
    "scripts/offline_arb/audit_planned_caption_availability.py",
    "scripts/offline_arb/audit_textatlas_plan_rows.py",
    "scripts/offline_arb/rebind_textatlas_catalog_evidence_w64.py",
    "scripts/offline_arb/test_rebind_textatlas_catalog_evidence_w64.py",
    "scripts/offline_arb/immutable_review_evidence.py",
    "scripts/offline_arb/stage512a_w64_exception_contract.py",
    "scripts/offline_arb/finalize_stage512a_w64_exception_gate.py",
    "scripts/offline_arb/test_stage512a_w64_exception_gate_contract.py",
    "scripts/offline_arb/test_stage512a_w64_runtime_profiles.py",
    "scripts/offline_arb/test_stage512a_w64_reviewed_exceptions.py",
    "scripts/offline_arb/completion_publication.py",
    "scripts/offline_arb/test_completion_publication.py",
    "scripts/offline_arb/verify_stage256_completion_for_stage512a.py",
    "scripts/offline_arb/test_stage256_completion_contract.py",
    "scripts/offline_arb/stage512a_launch_guard.py",
    "scripts/offline_arb/render_stage512a_formal_launch.py",
    "scripts/offline_arb/test_stage512a_launch_guard.py",
    "scripts/offline_arb/stage512a_gpuval_guard.py",
    "scripts/offline_arb/render_stage512a_gpuval.py",
    "scripts/offline_arb/test_stage512a_gpuval_contract.py",
    "training/dataline_offline/caption_sidecar.py",
    "training/dataline_offline/offline_dataset.py",
    "training/dataline_offline/caption_exceptions_v1.py",
    "training/dataline_offline/image_exceptions_v1.py",
    "training/dataline/image_fetch.py",
    "training/config_schema.py",
)

HEADER_KEYS = {
    "schema_version",
    "semantic_digest_algorithm",
    "builder_build_id",
    "caption_schema_contract_sha256",
    "catalog_hash",
    "catalog_header_sha256",
    "row_order",
    "file_rows",
    "row_group_rows",
    "domain_order",
    "total_rows",
    "diagnostic_run",
    "diagnostic_rows_per_domain",
    "catalog_files",
    "domains",
    "generated_at",
    "header_sha256",
}
DOMAIN_KEYS = {
    "row_base",
    "row_count",
    "columns",
    "source_semantics_sha256",
    "decoded_sidecar_semantics_sha256",
    "files",
}
FILE_KEYS = {
    "name",
    "row_start",
    "row_stop",
    "rows",
    "bytes",
    "sha256",
    "row_groups",
    "schema",
}
CATALOG_FILE_KEYS = {"name", "row_base", "rows", "bytes", "sha256", "schema"}
ANIMEFREQ_HEADER_KEYS = {
    "catalog_hash",
    "catalog_path",
    "files",
    "generated_at",
    "schema_version",
    "self_check",
    "shard_count",
}
FINGERPRINT_KEYS = {
    "device",
    "inode",
    "size_bytes",
    "mtime_ns",
    "ctime_ns",
}
HASH_WORKERS = 16


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def regular_metadata(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"artifact is not a regular file: {path}")
    return metadata


def directory_metadata(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"artifact is not a real directory: {path}")
    return metadata


def stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def fingerprint(metadata: os.stat_result) -> dict[str, int]:
    values = stat_identity(metadata)
    return {
        "device": values[0],
        "inode": values[1],
        "size_bytes": values[2],
        "mtime_ns": values[3],
        "ctime_ns": values[4],
    }


class SnapshotTracker:
    """Thread-safe file/directory fingerprint and content-hash ledger."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[Path, str, tuple[int, int, int, int, int]]] = {}
        self._hashes: dict[str, str] = {}

    @staticmethod
    def _path(path: Path) -> Path:
        if not path.is_absolute():
            raise ValueError(f"snapshot path must be absolute: {path}")
        return path

    def remember(
        self, path: Path, kind: str, metadata: os.stat_result
    ) -> tuple[int, int, int, int, int]:
        path = self._path(path)
        identity = stat_identity(metadata)
        key = str(path)
        with self._lock:
            previous = self._entries.get(key)
            current = (path, kind, identity)
            if previous is not None and previous != current:
                raise ValueError(f"mixed filesystem snapshot detected: {path}")
            self._entries[key] = current
        return identity

    def remember_directory(self, path: Path) -> None:
        self.remember(path, "directory", directory_metadata(path))

    def read_bytes(self, path: Path) -> bytes:
        before = regular_metadata(path)
        payload = path.read_bytes()
        after = regular_metadata(path)
        if stat_identity(before) != stat_identity(after):
            raise ValueError(f"file changed while reading: {path}")
        self.remember(path, "file", after)
        checksum = hashlib.sha256(payload).hexdigest()
        key = str(path)
        with self._lock:
            previous = self._hashes.get(key)
            if previous is not None and previous != checksum:
                raise ValueError(f"file content changed between reads: {path}")
            self._hashes[key] = checksum
        return payload

    def hash_file(self, path: Path) -> str:
        before = regular_metadata(path)
        key = str(path)
        with self._lock:
            cached = self._hashes.get(key)
        if cached is not None:
            self.remember(path, "file", before)
            return cached
        digest = hashlib.sha256()
        with path.open("rb", buffering=0) as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        checksum = digest.hexdigest()
        after = regular_metadata(path)
        if stat_identity(before) != stat_identity(after):
            raise ValueError(f"file changed while hashing: {path}")
        self.remember(path, "file", after)
        with self._lock:
            previous = self._hashes.get(key)
            if previous is not None and previous != checksum:
                raise ValueError(f"file hash changed within snapshot: {path}")
            self._hashes[key] = checksum
        return checksum

    def artifact(self, path: Path) -> dict[str, Any]:
        checksum = self.hash_file(path)
        metadata = regular_metadata(path)
        self.remember(path, "file", metadata)
        return {
            "path": str(path),
            "bytes": int(metadata.st_size),
            "sha256": checksum,
            "fingerprint": fingerprint(metadata),
        }

    def directory_artifact(self, path: Path) -> dict[str, Any]:
        metadata = directory_metadata(path)
        self.remember(path, "directory", metadata)
        return {"path": str(path), "fingerprint": fingerprint(metadata)}

    def merge(self, other: "SnapshotTracker") -> None:
        with other._lock:
            entries = list(other._entries.values())
            hashes = dict(other._hashes)
        for path, kind, identity in entries:
            metadata = path.lstat()
            if stat_identity(metadata) != identity:
                raise ValueError(f"snapshot changed before ledger merge: {path}")
            self.remember(path, kind, metadata)
        with self._lock:
            for key, checksum in hashes.items():
                previous = self._hashes.get(key)
                if previous is not None and previous != checksum:
                    raise ValueError(f"snapshot hash conflict during merge: {key}")
                self._hashes[key] = checksum

    def close(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
        for path, kind, expected in sorted(entries, key=lambda item: str(item[0])):
            metadata = path.lstat()
            if kind == "file" and not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"tracked file changed type: {path}")
            if kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"tracked directory changed type: {path}")
            if stat_identity(metadata) != expected:
                raise ValueError(f"tracked artifact changed before snapshot close: {path}")

    def evidence_digest(self) -> str:
        with self._lock:
            payload = {
                key: {
                    "kind": kind,
                    "fingerprint": {
                        "device": identity[0],
                        "inode": identity[1],
                        "size_bytes": identity[2],
                        "mtime_ns": identity[3],
                        "ctime_ns": identity[4],
                    },
                    **({"sha256": self._hashes[key]} if key in self._hashes else {}),
                }
                for key, (_path, kind, identity) in sorted(self._entries.items())
            }
        return canonical_json_sha256(payload)


def read_json_snapshot(
    path: Path, tracker: SnapshotTracker
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = tracker.read_bytes(path)
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value, tracker.artifact(path)


def exact_set(label: str, live: Iterable[str], expected: Iterable[str]) -> None:
    live_set, expected_set = set(live), set(expected)
    if live_set != expected_set:
        raise ValueError(
            f"{label} set drift: missing={sorted(expected_set-live_set)[:20]}, "
            f"unexpected={sorted(live_set-expected_set)[:20]}"
        )


def load_stage_contract(
    config: Path, sampling: Path, tracker: SnapshotTracker
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    config_payload = tracker.read_bytes(config)
    try:
        cfg = OmegaConf.load(io.StringIO(config_payload.decode("utf-8")))
    except Exception as exc:
        raise ValueError(f"cannot parse selected config {config}: {exc}") from exc
    stage = OmegaConf.select(cfg, "dataset.stage")
    image_size = OmegaConf.select(cfg, "dataset.image_size")
    sampling_from_config = OmegaConf.select(cfg, "dataset.sampling_config")
    if stage not in ("256", "512a"):
        raise ValueError(f"unsupported selected config stage: {stage!r}")
    if type(image_size) is not int or image_size != {"256": 256, "512a": 512}[stage]:
        raise ValueError(f"selected config stage/image_size mismatch: {stage!r}/{image_size!r}")
    if not isinstance(sampling_from_config, str) or not sampling_from_config:
        raise ValueError("selected config dataset.sampling_config is missing")
    if Path(sampling_from_config).expanduser().resolve(strict=False) != sampling:
        raise ValueError(
            f"selected config sampling path {sampling_from_config} != --sampling {sampling}"
        )
    sampling_value, sampling_artifact = read_json_snapshot(sampling, tracker)
    anime = sampling_value.get("anime")
    caption = anime.get("caption") if isinstance(anime, dict) else None
    require_stages = caption.get("require_nl_stages", []) if isinstance(caption, dict) else []
    if not isinstance(require_stages, list) or any(
        not isinstance(value, str) for value in require_stages
    ):
        raise ValueError("sampling anime.caption.require_nl_stages must be a string list")
    requires_nl = stage in require_stages
    if stage == "512a" and requires_nl:
        raise ValueError("stage512a freeze requires sampling requires_nl=false")
    contract = {
        "stage": stage,
        "image_size": image_size,
        "eligibility": f"elig_{image_size}",
        "requires_nl": requires_nl,
        # Offline catalog construction intentionally does not apply the online
        # anime NL-only filter, including for the legacy stage256 artifact.
        "anime_nl_filter_applied": False,
    }
    sampling_evidence = None
    if stage == "512a":
        domain_mix_root = sampling_value.get("domain_mix")
        domain_mix = (
            domain_mix_root.get(stage) if isinstance(domain_mix_root, dict) else None
        )
        if (
            not isinstance(domain_mix, dict)
            or set(domain_mix) != set(DOMAINS)
            or any(type(domain_mix[domain]) is not float for domain in DOMAINS)
        ):
            raise ValueError("stage512a sampling domain_mix is malformed")
        sampling_evidence = {
            "sha256": sampling_artifact["sha256"],
            "domain_mix": dict(domain_mix),
        }
    return contract, sampling_evidence


def validate_stage512a_catalog_sampling(
    plan_header: dict[str, Any],
    catalog_header: dict[str, Any],
    contract: dict[str, Any],
    sampling_evidence: dict[str, Any] | None,
) -> None:
    """Bind stage512a catalog semantics to the current tracked sampling JSON."""

    if contract["stage"] != "512a":
        return
    if not isinstance(sampling_evidence, dict):
        raise ValueError("stage512a sampling evidence is missing")
    inputs = catalog_header.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("stage512a catalog inputs must be an object")
    if inputs.get("sampling_config_sha256") != sampling_evidence["sha256"]:
        raise ValueError("stage512a catalog sampling SHA-256 differs from current sampling")
    catalog_mix = inputs.get("domain_mix")
    if (
        not isinstance(catalog_mix, dict)
        or set(catalog_mix) != set(DOMAINS)
        or any(type(catalog_mix[domain]) is not float for domain in DOMAINS)
        or catalog_mix != sampling_evidence["domain_mix"]
    ):
        raise ValueError("stage512a catalog domain_mix differs from current sampling")
    expected_plan_mix: dict[str, str] = {}
    for domain in DOMAINS:
        text = format(Decimal(str(sampling_evidence["domain_mix"][domain])), "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        expected_plan_mix[domain] = text or "0"
    if plan_header.get("domain_mix") != expected_plan_mix:
        raise ValueError("stage512a plan domain_mix differs from current sampling")
    if (
        "sampling_config_sha256" in plan_header
        and plan_header["sampling_config_sha256"] != sampling_evidence["sha256"]
    ):
        raise ValueError("stage512a plan sampling SHA-256 differs from current sampling")


def _explicit_header_contract(
    header: dict[str, Any], expected: dict[str, Any], label: str
) -> bool:
    present = {key for key in CONTRACT_KEYS if key in header}
    if not present:
        return False
    if present != set(CONTRACT_KEYS):
        raise ValueError(f"{label} has a partial stage contract: {sorted(present)}")
    for key in CONTRACT_KEYS:
        got, wanted = header[key], expected[key]
        if type(got) is not type(wanted) or got != wanted:
            raise ValueError(f"{label} {key}={got!r} != selected config {wanted!r}")
    return True


def validate_release_contract(
    plan_header: dict[str, Any],
    catalog_header: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    catalog_explicit = _explicit_header_contract(catalog_header, expected, "catalog header")
    plan_explicit = _explicit_header_contract(plan_header, expected, "plan header")
    if catalog_explicit != plan_explicit:
        raise ValueError("plan/catalog stage contract presence differs")
    legacy = not catalog_explicit
    if legacy:
        if expected["stage"] != "256":
            raise ValueError("only verified stage256 artifacts may omit the stage contract")
        if catalog_header.get("assignment_hash") != LEGACY_STAGE256_ASSIGNMENT:
            raise ValueError("legacy stage256 catalog assignment identity is not recognized")
        if (
            plan_header.get("assignment_hash") != LEGACY_STAGE256_ASSIGNMENT
            or plan_header.get("upstream_assignment_hash") != LEGACY_STAGE256_ASSIGNMENT
            or not str(plan_header.get("plan_id", "")).startswith("stage256_")
        ):
            raise ValueError("legacy stage256 plan assignment identity is not recognized")
    return {**expected, "legacy_stage256_headers": legacy}


def validate_external_bindings(
    bindings: dict[str, Path],
    contract: dict[str, Any],
    plan_header: dict[str, Any],
    catalog_header: dict[str, Any],
    plan_header_sha256: str,
    catalog_header_sha256: str,
    repo: Path,
    tracker: SnapshotTracker,
) -> None:
    if contract["stage"] == "512a":
        expected_labels = {"gpic_payload", "reviewed_exception_gate"}
        if set(bindings) != expected_labels:
            raise ValueError(
                "stage512a freeze requires exactly --external-binding "
                "gpic_payload=<JSON> and reviewed_exception_gate=<JSON>; "
                f"actual={sorted(bindings)!r}"
            )
    if "gpic_payload" in bindings:
        value, _artifact = read_json_snapshot(bindings["gpic_payload"], tracker)
        expected_identity = {key: contract[key] for key in CONTRACT_KEYS}
        if value.get("status") != "PASS":
            raise ValueError("gpic_payload status is not PASS")
        if value.get("stage_identity") != expected_identity:
            raise ValueError(
                "gpic_payload stage_identity does not match the selected release contract"
            )
        if value.get("catalog_hash") != catalog_header["catalog_hash"]:
            raise ValueError("gpic_payload catalog_hash does not bind the selected catalog")
        if value.get("catalog_header_sha256") != catalog_header_sha256:
            raise ValueError(
                "gpic_payload catalog_header_sha256 does not bind the live catalog header"
            )
        artifact_root_raw = value.get("artifact_root")
        if not isinstance(artifact_root_raw, str) or not artifact_root_raw:
            raise ValueError("gpic_payload artifact_root is missing or malformed")
        artifact_root = Path(artifact_root_raw)
        if (
            not artifact_root.is_absolute()
            or artifact_root.name in {"latest", "current"}
            or artifact_root.is_symlink()
            or not artifact_root.is_dir()
            or artifact_root.resolve(strict=True) != artifact_root
        ):
            raise ValueError(
                "gpic_payload artifact_root must be an absolute, canonical, "
                "non-symlink immutable directory"
            )
        for key in ("manifest_sha256", "release_digest"):
            if not is_sha256(value.get(key)):
                raise ValueError(f"gpic_payload {key} is not a lowercase SHA-256")

    if "reviewed_exception_gate" in bindings:
        validate_stage512a_reviewed_gate_storage(bindings["reviewed_exception_gate"])
        value, gate_artifact = read_json_snapshot(
            bindings["reviewed_exception_gate"], tracker
        )
        if not is_sha256(STAGE512A_REVIEWED_GATE_SHA256):
            raise ValueError(
                "stage512a W64 reviewed_exception_gate SHA-256 authority "
                "is pending or malformed"
            )
        if gate_artifact["sha256"] != STAGE512A_REVIEWED_GATE_SHA256:
            raise ValueError(
                "reviewed_exception_gate SHA-256 differs from the sealed "
                "W64 authority"
            )
        expected_identity = {key: contract[key] for key in CONTRACT_KEYS}
        if value.get("status") != "PASS":
            raise ValueError("reviewed_exception_gate status is not PASS")
        if value.get("schema") != STAGE512A_EXCEPTION_SCHEMA:
            raise ValueError("reviewed_exception_gate schema mismatch")
        if value.get("policy") != STAGE512A_EXCEPTION_POLICY:
            raise ValueError("reviewed_exception_gate policy mismatch")
        if value.get("stage_identity") != expected_identity:
            raise ValueError("reviewed_exception_gate stage_identity mismatch")
        for key in ("plan_id", "plan_hash", "assignment_hash"):
            if value.get(key) != plan_header.get(key):
                raise ValueError(f"reviewed_exception_gate {key} mismatch")
        if value.get("catalog_hash") != catalog_header.get("catalog_hash"):
            raise ValueError("reviewed_exception_gate catalog_hash mismatch")
        if value.get("plan_header_sha256") != plan_header_sha256:
            raise ValueError("reviewed_exception_gate plan header SHA mismatch")
        if value.get("catalog_header_sha256") != catalog_header_sha256:
            raise ValueError("reviewed_exception_gate catalog header SHA mismatch")
        if value.get("counts") != STAGE512A_EXCEPTION_COUNTS:
            raise ValueError("reviewed_exception_gate exact count contract mismatch")

        runtime = value.get("runtime")
        if not isinstance(runtime, dict) or set(runtime) != set(
            STAGE512A_EXCEPTION_RUNTIME_SOURCES
        ):
            raise ValueError("reviewed_exception_gate runtime source set mismatch")
        for label, relative in STAGE512A_EXCEPTION_RUNTIME_SOURCES.items():
            item = runtime.get(label)
            expected_path = repo / relative
            if not isinstance(item, dict):
                raise ValueError(
                    f"reviewed_exception_gate runtime.{label} must be an object"
                )
            if item.get("path") != str(expected_path):
                raise ValueError(
                    f"reviewed_exception_gate runtime.{label} path mismatch"
                )
            live = tracker.artifact(expected_path)
            if item.get("sha256") != live["sha256"]:
                raise ValueError(
                    f"reviewed_exception_gate runtime.{label} SHA mismatch"
                )

        producer = value.get("producer")
        expected_producer_keys = {
            "schema",
            "job_name",
            "container_image",
            "launch_token",
            "render_payload_sha256",
            "rendered_yaml",
            *STAGE512A_EXCEPTION_PRODUCER_SOURCES,
        }
        if not isinstance(producer, dict) or set(producer) != expected_producer_keys:
            raise ValueError("reviewed_exception_gate producer provenance set mismatch")
        token = producer.get("launch_token")
        payload_sha = producer.get("render_payload_sha256")
        if (
            producer.get("schema")
            != "stage512a_reviewed_gate_producer_provenance_v1"
            or producer.get("container_image")
            != STAGE512A_EXCEPTION_PRODUCER_IMAGE
            or not isinstance(producer.get("job_name"), str)
            or not producer["job_name"]
            or not isinstance(token, str)
            or not token.startswith("STAGE512A_WRITE_V1_")
            or not is_sha256(token.removeprefix("STAGE512A_WRITE_V1_"))
            or not is_sha256(payload_sha)
            or token.removeprefix("STAGE512A_WRITE_V1_") != payload_sha
        ):
            raise ValueError("reviewed_exception_gate producer identity mismatch")

        producer_paths = {
            label: repo / relative
            for label, relative in STAGE512A_EXCEPTION_PRODUCER_SOURCES.items()
        }
        rendered_item = producer.get("rendered_yaml")
        if not isinstance(rendered_item, dict) or set(rendered_item) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ValueError("reviewed_exception_gate producer rendered_yaml malformed")
        rendered_path_raw = rendered_item.get("path")
        rendered_path = Path(rendered_path_raw) if isinstance(rendered_path_raw, str) else Path()
        try:
            rendered_is_canonical = (
                rendered_path.is_absolute()
                and rendered_path.resolve(strict=True) == rendered_path
            )
        except OSError as exc:
            raise ValueError(
                "reviewed_exception_gate rendered YAML path is unreadable"
            ) from exc
        if not rendered_is_canonical:
            raise ValueError("reviewed_exception_gate rendered YAML path is not canonical")
        if stat.S_IMODE(rendered_path.lstat().st_mode) & 0o222:
            raise ValueError("reviewed_exception_gate rendered YAML is writable")
        producer_paths["rendered_yaml"] = rendered_path
        for label, expected_path in producer_paths.items():
            item = producer.get(label)
            if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
                raise ValueError(
                    f"reviewed_exception_gate producer.{label} artifact malformed"
                )
            if item.get("path") != str(expected_path):
                raise ValueError(
                    f"reviewed_exception_gate producer.{label} path mismatch"
                )
            live = tracker.artifact(expected_path)
            if (
                item.get("bytes") != live["bytes"]
                or item.get("sha256") != live["sha256"]
            ):
                raise ValueError(
                    f"reviewed_exception_gate producer.{label} byte identity mismatch"
                )


def expected_inputs(
    repo: Path,
    config: Path,
    plan: Path,
    catalog: Path,
    sampling: Path,
    animefreq: Path,
    sidecar: Path,
    external_bindings: dict[str, Path],
) -> dict[str, Path]:
    inputs = {
        "config": config,
        "plan_header": plan / "header.json",
        "plan_checksums": plan / "checksums.json",
        "catalog_header": catalog / "header.json",
        "sampling_json": sampling,
        "animefreq_header": animefreq / "header.json",
        "animefreq_success": animefreq / "_SUCCESS",
        "sidecar_header": sidecar / "header.json",
        "sidecar_success": sidecar / "_SUCCESS",
    }
    for name, path in sorted(external_bindings.items()):
        inputs[f"external:{name}"] = path
    return inputs


def source_explicit_for_stage(stage: str) -> tuple[str, ...]:
    if stage == "256":
        return SOURCE_EXPLICIT
    if stage == "512a":
        return SOURCE_EXPLICIT_STAGE512A
    raise ValueError(f"no freeze source profile for stage {stage!r}")


def discover_training_sources(repo: Path, stage: str) -> dict[str, Path]:
    explicit = source_explicit_for_stage(stage)
    paths = {repo / relative for relative in explicit}
    for tree in SOURCE_TREES:
        root = repo / tree
        directory_metadata(root)
        paths.update(path for path in root.rglob("*.py") if path.is_file())
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"training source files missing: {missing}")
    return {
        path.relative_to(repo).as_posix(): path
        for path in sorted(paths)
    }


def catalog_domains(catalog_header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if catalog_header.get("schema_version") != "offline_arb_catalog_v1":
        raise ValueError("unsupported catalog schema")
    if catalog_header.get("domain_order") != list(DOMAINS):
        raise ValueError("catalog domain order drift")
    files = catalog_header.get("files")
    if not isinstance(files, dict) or set(files) != set(DOMAINS):
        raise ValueError("catalog domain file set drift")
    cursor = 0
    result: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        raw = files[domain]
        if not isinstance(raw, dict):
            raise ValueError(f"catalog files.{domain} must be an object")
        name = raw.get("name")
        rows = strict_int(raw.get("rows"), f"catalog {domain}.rows", minimum=1)
        size_bytes = strict_int(raw.get("bytes"), f"catalog {domain}.bytes", minimum=1)
        checksum, schema = raw.get("sha256"), raw.get("schema")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".parquet")
            or not is_sha256(checksum)
            or not isinstance(schema, str)
            or not schema
        ):
            raise ValueError(f"malformed catalog file declaration: {domain}")
        result[domain] = {
            "name": name,
            "row_base": cursor,
            "rows": rows,
            "bytes": size_bytes,
            "sha256": checksum,
            "schema": schema,
        }
        cursor += rows
    if catalog_header.get("row_count") != cursor:
        raise ValueError("catalog total row count drift")
    return result


def exact_plan_root(
    plan: Path, block_names: Iterable[str], tracker: SnapshotTracker
) -> None:
    tracker.remember_directory(plan)
    blocks = plan / "blocks"
    tracker.remember_directory(blocks)
    exact_set("plan root", (path.name for path in plan.iterdir()), {
        "header.json", "checksums.json", "blocks"
    })
    exact_set("plan block", (path.name for path in blocks.iterdir()), block_names)
    tracker.remember(plan / "header.json", "file", regular_metadata(plan / "header.json"))
    tracker.remember(
        plan / "checksums.json", "file", regular_metadata(plan / "checksums.json")
    )
    for name in block_names:
        tracker.remember(blocks / name, "file", regular_metadata(blocks / name))


def validate_plan_blocks(
    plan: Path, checksums: dict[str, Any], tracker: SnapshotTracker
) -> dict[str, Any]:
    blocks = checksums.get("blocks")
    if not isinstance(blocks, dict):
        raise ValueError("plan checksums.blocks must be an object")
    for name, expected in blocks.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".npz")
            or not is_sha256(expected)
        ):
            raise ValueError(f"malformed plan block checksum: {name!r}")
    exact_plan_root(plan, blocks, tracker)

    def validate_one(item: tuple[str, str]) -> tuple[str, dict[str, Any]]:
        name, expected = item
        artifact = tracker.artifact(plan / "blocks" / name)
        if artifact["sha256"] != expected:
            raise ValueError(f"plan block hash drift: {name}")
        return name, artifact

    files: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(HASH_WORKERS, max(1, len(blocks))),
        thread_name_prefix="freeze-plan-sha",
    ) as pool:
        for name, artifact in pool.map(validate_one, sorted(blocks.items())):
            files[name] = artifact
    return {
        "root": str(plan),
        "directories": {
            "root": tracker.directory_artifact(plan),
            "blocks": tracker.directory_artifact(plan / "blocks"),
        },
        "files": dict(sorted(files.items())),
    }


def exact_catalog_declared(
    root: Path,
    catalog: dict[str, dict[str, Any]],
    tracker: SnapshotTracker,
) -> None:
    tracker.remember_directory(root)
    tracker.remember(root / "header.json", "file", regular_metadata(root / "header.json"))
    declared = {entry["name"] for entry in catalog.values()}
    if len(declared) != len(DOMAINS):
        raise ValueError("catalog declared parquet names are not unique")
    for name in declared:
        tracker.remember(root / name, "file", regular_metadata(root / name))


def validate_catalog_files(
    root: Path,
    catalog: dict[str, dict[str, Any]],
    tracker: SnapshotTracker,
) -> dict[str, Any]:
    exact_catalog_declared(root, catalog, tracker)

    def validate_one(item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        domain, declared = item
        path = root / declared["name"]
        artifact = tracker.artifact(path)
        if artifact["bytes"] != declared["bytes"] or artifact["sha256"] != declared["sha256"]:
            raise ValueError(f"catalog parquet bytes/SHA drift: {domain}")
        parquet = pq.ParquetFile(path)
        if (
            parquet.metadata.num_rows != declared["rows"]
            or str(parquet.schema_arrow) != declared["schema"]
        ):
            raise ValueError(f"catalog parquet footer drift: {domain}")
        tracker.remember(path, "file", regular_metadata(path))
        return domain, {**declared, "path": str(path), "fingerprint": artifact["fingerprint"]}

    files: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(DOMAINS), thread_name_prefix="freeze-catalog-sha"
    ) as pool:
        for domain, artifact in pool.map(validate_one, sorted(catalog.items())):
            files[domain] = artifact
    return {
        "root": str(root),
        "directory": tracker.directory_artifact(root),
        "files": dict(sorted(files.items())),
    }


def exact_animefreq_root(
    root: Path, shard_names: Iterable[str], tracker: SnapshotTracker
) -> None:
    expected = {"header.json", "_SUCCESS", *set(shard_names)}
    tracker.remember_directory(root)
    exact_set("animefreq root", (path.name for path in root.iterdir()), expected)
    for name in expected:
        tracker.remember(root / name, "file", regular_metadata(root / name))


def validate_animefreq(
    root: Path,
    catalog_root: Path,
    catalog_hash: str,
    tracker: SnapshotTracker,
) -> tuple[dict[str, Any], dict[str, Any]]:
    header, header_artifact = read_json_snapshot(root / "header.json", tracker)
    if set(header) != ANIMEFREQ_HEADER_KEYS:
        raise ValueError("animefreq header key set drift")
    if header.get("schema_version") != "offline_arb_animefreq_v1":
        raise ValueError("unsupported animefreq schema")
    if header.get("catalog_hash") != catalog_hash:
        raise ValueError("animefreq catalog_hash != catalog catalog_hash")
    if header.get("catalog_path") != str(catalog_root / "anime.parquet"):
        raise ValueError("animefreq catalog_path drift")
    if not isinstance(header.get("generated_at"), str) or not header["generated_at"]:
        raise ValueError("animefreq generated_at is empty")
    self_check = header.get("self_check")
    if (
        not isinstance(self_check, dict)
        or set(self_check) != {"pass", "sampled_shards"}
        or self_check.get("pass") is not True
        or not isinstance(self_check.get("sampled_shards"), list)
        or not self_check["sampled_shards"]
        or any(not isinstance(value, str) or not value for value in self_check["sampled_shards"])
    ):
        raise ValueError("animefreq self_check is malformed or not PASS")
    declared = header.get("files")
    if not isinstance(declared, dict) or header.get("shard_count") != 17 or len(declared) != 17:
        raise ValueError("animefreq must declare exactly 17 shards")
    source_shards: set[str] = set()
    for name, entry in declared.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".json.gz")
            or not isinstance(entry, dict)
            or set(entry) != {"sha256", "source_shard"}
            or not is_sha256(entry.get("sha256"))
            or not isinstance(entry.get("source_shard"), str)
            or not entry["source_shard"]
        ):
            raise ValueError(f"invalid animefreq shard declaration: {name!r}")
        source_shards.add(entry["source_shard"])
    if len(source_shards) != 17 or not set(self_check["sampled_shards"]).issubset(source_shards):
        raise ValueError("animefreq source/self-check shard set drift")
    exact_animefreq_root(root, declared, tracker)
    success = tracker.read_bytes(root / "_SUCCESS")
    if success != b"":
        raise ValueError("animefreq _SUCCESS must be the canonical empty marker")

    def validate_one(item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        name, entry = item
        artifact = tracker.artifact(root / name)
        if artifact["sha256"] != entry["sha256"]:
            raise ValueError(f"animefreq shard hash mismatch: {name}")
        return name, artifact

    shards: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(HASH_WORKERS, len(declared)),
        thread_name_prefix="freeze-animefreq-sha",
    ) as pool:
        for name, artifact in pool.map(validate_one, sorted(declared.items())):
            shards[name] = artifact
    return header, {
        "root": str(root),
        "directory": tracker.directory_artifact(root),
        "header": header_artifact,
        "success": tracker.artifact(root / "_SUCCESS"),
        "shards": dict(sorted(shards.items())),
    }


def parse_sidecar_header(
    root: Path, catalog_root: Path, tracker: SnapshotTracker
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, bytes],
]:
    if not root.is_absolute() or not catalog_root.is_absolute():
        raise ValueError("sidecar and catalog roots must be absolute")
    directory_metadata(root)
    directory_metadata(catalog_root)
    paths = {
        "header": root / "header.json",
        "success": root / "_SUCCESS",
        "catalog_header": catalog_root / "header.json",
    }
    controls = {label: tracker.read_bytes(path) for label, path in paths.items()}
    header = json.loads(controls["header"])
    catalog_header = json.loads(controls["catalog_header"])
    if not isinstance(header, dict) or not isinstance(catalog_header, dict):
        raise ValueError("sidecar/catalog header must be JSON objects")
    if set(header) != HEADER_KEYS:
        raise ValueError(
            f"sidecar header key set drift: missing={sorted(HEADER_KEYS-set(header))}, "
            f"unexpected={sorted(set(header)-HEADER_KEYS)}"
        )
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "semantic_digest_algorithm": SEMANTIC_DIGEST_ALGORITHM,
        "builder_build_id": BUILDER_BUILD_ID,
        "caption_schema_contract_sha256": CAPTION_SCHEMA_CONTRACT_SHA256,
        "row_order": ROW_ORDER,
        "file_rows": FILE_ROWS,
        "row_group_rows": ROW_GROUP_ROWS,
        "domain_order": list(DOMAINS),
        "diagnostic_run": False,
        "diagnostic_rows_per_domain": None,
    }
    for key, expected in expected_scalars.items():
        if header.get(key) != expected:
            raise ValueError(
                f"sidecar header contract drift {key}: {header.get(key)!r} != {expected!r}"
            )
    if not isinstance(header.get("generated_at"), str) or not header["generated_at"]:
        raise ValueError("sidecar generated_at is empty")
    semantic_header_sha = header.get("header_sha256")
    if not is_sha256(semantic_header_sha) or header_self_hash(header) != semantic_header_sha:
        raise ValueError("sidecar header self-hash mismatch")
    if controls["success"] != (semantic_header_sha + "\n").encode("ascii"):
        raise ValueError("sidecar _SUCCESS must be exactly header_sha256 plus newline")

    catalog_header_sha = hashlib.sha256(controls["catalog_header"]).hexdigest()
    catalog_hash = catalog_header.get("catalog_hash")
    if (
        not is_sha256(catalog_hash)
        or header.get("catalog_hash") != catalog_hash
        or header.get("catalog_header_sha256") != catalog_header_sha
    ):
        raise ValueError("sidecar/catalog header binding mismatch")
    catalogs = catalog_domains(catalog_header)
    if header.get("total_rows") != sum(value["rows"] for value in catalogs.values()):
        raise ValueError("sidecar total_rows/catalog mismatch")
    raw_catalog_files = header.get("catalog_files")
    if not isinstance(raw_catalog_files, dict) or set(raw_catalog_files) != set(DOMAINS):
        raise ValueError("sidecar catalog_files domain set drift")
    for domain in DOMAINS:
        raw = raw_catalog_files[domain]
        if not isinstance(raw, dict) or set(raw) != CATALOG_FILE_KEYS or raw != catalogs[domain]:
            raise ValueError(f"sidecar catalog binding drift: {domain}")

    raw_domains = header.get("domains")
    if not isinstance(raw_domains, dict) or set(raw_domains) != set(DOMAINS):
        raise ValueError("sidecar domains must contain exactly the five contract domains")
    parsed: dict[str, list[dict[str, Any]]] = {}
    for domain in DOMAINS:
        raw = raw_domains[domain]
        if not isinstance(raw, dict) or set(raw) != DOMAIN_KEYS:
            raise ValueError(f"sidecar domains.{domain} key set drift")
        catalog = catalogs[domain]
        row_base, row_count = catalog["row_base"], catalog["rows"]
        if (raw.get("row_base"), raw.get("row_count")) != (row_base, row_count):
            raise ValueError(f"sidecar/catalog row range mismatch: {domain}")
        if raw.get("columns") != list(CAPTION_COLUMNS[domain]):
            raise ValueError(f"sidecar columns drift: {domain}")
        source_digest = raw.get("source_semantics_sha256")
        decoded_digest = raw.get("decoded_sidecar_semantics_sha256")
        if not is_sha256(source_digest) or source_digest != decoded_digest:
            raise ValueError(f"sidecar source/decoded semantic digest mismatch: {domain}")
        files = raw.get("files")
        expected_count = math.ceil(row_count / FILE_ROWS)
        if not isinstance(files, list) or len(files) != expected_count:
            raise ValueError(f"sidecar file count drift: {domain}")
        expected_schema = str(sidecar_schema(CAPTION_SCHEMAS[domain], CAPTION_COLUMNS[domain]))
        parsed[domain] = []
        for index, entry in enumerate(files):
            if not isinstance(entry, dict) or set(entry) != FILE_KEYS:
                raise ValueError(f"sidecar file entry key drift: {domain}/{index}")
            name = f"{domain}/part-{index:05d}.parquet"
            row_start = row_base + index * FILE_ROWS
            row_stop = min(row_base + row_count, row_start + FILE_ROWS)
            rows = row_stop - row_start
            row_groups = math.ceil(rows / ROW_GROUP_ROWS)
            expected = {
                "name": name,
                "row_start": row_start,
                "row_stop": row_stop,
                "rows": rows,
                "bytes": entry.get("bytes"),
                "sha256": entry.get("sha256"),
                "row_groups": row_groups,
                "schema": expected_schema,
            }
            if entry != expected:
                raise ValueError(f"sidecar file contract/range drift: {name}")
            strict_int(entry["bytes"], f"{name}.bytes", minimum=1)
            if not is_sha256(entry["sha256"]):
                raise ValueError(f"sidecar file SHA malformed: {name}")
            parsed[domain].append(dict(entry))
    return header, parsed, controls


def exact_sidecar_root(
    root: Path,
    parsed: dict[str, list[dict[str, Any]]],
    tracker: SnapshotTracker,
) -> None:
    tracker.remember_directory(root)
    expected_root = {"header.json", "_SUCCESS", *DOMAINS}
    exact_set("sidecar root entry", (path.name for path in root.iterdir()), expected_root)
    tracker.remember(root / "header.json", "file", regular_metadata(root / "header.json"))
    tracker.remember(root / "_SUCCESS", "file", regular_metadata(root / "_SUCCESS"))
    for domain in DOMAINS:
        directory = root / domain
        tracker.remember_directory(directory)
        expected = {Path(entry["name"]).name for entry in parsed[domain]}
        exact_set(
            f"sidecar {domain} file",
            (path.name for path in directory.iterdir()),
            expected,
        )
        for name in expected:
            tracker.remember(directory / name, "file", regular_metadata(directory / name))


def verify_sidecar_footer(
    path: Path, entry: dict[str, Any], expected_schema: pa.Schema
) -> None:
    parquet = pq.ParquetFile(path)
    if (
        str(parquet.schema_arrow) != entry["schema"]
        or not parquet.schema_arrow.equals(expected_schema, check_metadata=True)
    ):
        raise ValueError(f"sidecar footer schema drift: {entry['name']}")
    row_field = parquet.schema_arrow.field("row_id")
    if parquet.schema_arrow.names[0] != "row_id" or row_field.type != pa.int64() or row_field.nullable:
        raise ValueError(f"sidecar row_id must be first, non-null int64: {entry['name']}")
    if parquet.metadata.num_rows != entry["rows"] or parquet.num_row_groups != entry["row_groups"]:
        raise ValueError(f"sidecar footer row count/group drift: {entry['name']}")
    cursor = entry["row_start"]
    for index in range(parquet.num_row_groups):
        expected_rows = min(ROW_GROUP_ROWS, entry["row_stop"] - cursor)
        row_group = parquet.metadata.row_group(index)
        statistics = row_group.column(0).statistics
        if (
            row_group.num_rows != expected_rows
            or statistics is None
            or not statistics.has_min_max
            or int(statistics.min) != cursor
            or int(statistics.max) != cursor + expected_rows - 1
            or statistics.null_count != 0
        ):
            raise ValueError(f"sidecar row-group range/null drift: {entry['name']}:{index}")
        cursor += expected_rows
    if cursor != entry["row_stop"]:
        raise ValueError(f"sidecar footer row range did not reach stop: {entry['name']}")


def sidecar_release_digest(
    root: Path, header: dict[str, Any], controls: dict[str, bytes]
) -> str:
    return canonical_json_sha256(
        {
            "schema_version": "offline_arb_caption_sidecar_release_v1",
            "root": str(root),
            "header_sha256": header["header_sha256"],
            "header_file_sha256": hashlib.sha256(controls["header"]).hexdigest(),
            "success_file_sha256": hashlib.sha256(controls["success"]).hexdigest(),
            "catalog_hash": header["catalog_hash"],
            "catalog_header_sha256": hashlib.sha256(controls["catalog_header"]).hexdigest(),
        }
    )


def validate_sidecar(
    root: Path,
    catalog_root: Path,
    *,
    full: bool,
    parent_tracker: SnapshotTracker | None = None,
) -> dict[str, Any]:
    tracker = SnapshotTracker()
    header, parsed, controls = parse_sidecar_header(root, catalog_root, tracker)
    files: dict[str, dict[str, Any]] = {}
    if full:
        exact_sidecar_root(root, parsed, tracker)

        def validate_one(
            item: tuple[str, dict[str, Any]]
        ) -> tuple[str, dict[str, Any]]:
            domain, entry = item
            path = root / entry["name"]
            artifact = tracker.artifact(path)
            if artifact["bytes"] != entry["bytes"] or artifact["sha256"] != entry["sha256"]:
                raise ValueError(f"sidecar byte-size/SHA drift: {entry['name']}")
            expected_schema = sidecar_schema(CAPTION_SCHEMAS[domain], CAPTION_COLUMNS[domain])
            verify_sidecar_footer(path, entry, expected_schema)
            tracker.remember(path, "file", regular_metadata(path))
            return entry["name"], {
                "path": str(path),
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
                "row_start": entry["row_start"],
                "row_stop": entry["row_stop"],
                "rows": entry["rows"],
                "row_groups": entry["row_groups"],
                "schema": entry["schema"],
                "fingerprint": artifact["fingerprint"],
            }

        work = [(domain, entry) for domain in DOMAINS for entry in parsed[domain]]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(HASH_WORKERS, max(1, len(work))),
            thread_name_prefix="freeze-sidecar-sha-footer",
        ) as pool:
            for name, artifact in pool.map(validate_one, work):
                files[name] = artifact

    # Control reparse precedes the final exact-set and uniform fingerprint close.
    final_header, final_parsed, final_controls = parse_sidecar_header(
        root, catalog_root, tracker
    )
    if final_header != header or final_parsed != parsed or final_controls != controls:
        raise ValueError("sidecar control snapshot changed during preflight")
    if full:
        exact_sidecar_root(root, parsed, tracker)
    tracker.close()
    if parent_tracker is not None:
        parent_tracker.merge(tracker)

    control_artifacts = {
        "header": tracker.artifact(root / "header.json"),
        "success": tracker.artifact(root / "_SUCCESS"),
        "catalog_header": tracker.artifact(catalog_root / "header.json"),
    }
    directory_artifacts = {
        "root": tracker.directory_artifact(root),
        **(
            {domain: tracker.directory_artifact(root / domain) for domain in DOMAINS}
            if full
            else {}
        ),
    }
    tracker.close()
    release_digest = sidecar_release_digest(root, header, controls)
    return {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "header_sha256": header["header_sha256"],
        "header_file_sha256": control_artifacts["header"]["sha256"],
        "success_file_sha256": control_artifacts["success"]["sha256"],
        "catalog_hash": header["catalog_hash"],
        "catalog_header_sha256": control_artifacts["catalog_header"]["sha256"],
        "release_digest": release_digest,
        "controls": control_artifacts,
        "directories": directory_artifacts,
        "files": dict(sorted(files.items())),
        "file_count": sum(len(value) for value in parsed.values()),
        "total_bytes": sum(entry["bytes"] for domain in DOMAINS for entry in parsed[domain]),
        "snapshot_fingerprints_sha256": tracker.evidence_digest(),
    }


def validate_common_headers(
    plan: Path,
    catalog: Path,
    expected_contract: dict[str, Any],
    tracker: SnapshotTracker,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    plan_header, plan_header_artifact = read_json_snapshot(plan / "header.json", tracker)
    plan_checksums, checksums_artifact = read_json_snapshot(plan / "checksums.json", tracker)
    catalog_header, catalog_header_artifact = read_json_snapshot(catalog / "header.json", tracker)
    catalog_hash = catalog_header.get("catalog_hash")
    if not is_sha256(catalog_hash):
        raise ValueError("catalog header catalog_hash is malformed")
    if plan_header.get("upstream_catalog_hash") != catalog_hash:
        raise ValueError("plan/catalog hash binding mismatch")
    assignment_hash = catalog_header.get("assignment_hash")
    if not is_sha256(assignment_hash):
        raise ValueError("catalog header assignment_hash is malformed")
    if (
        plan_header.get("assignment_hash") != assignment_hash
        or plan_header.get("upstream_assignment_hash") != assignment_hash
    ):
        raise ValueError("plan/catalog assignment hash binding mismatch")
    if plan_checksums.get("header") != plan_header_artifact["sha256"]:
        raise ValueError("plan checksums do not bind the live plan header")
    release_contract = validate_release_contract(
        plan_header, catalog_header, expected_contract
    )
    return (
        plan_header,
        plan_checksums,
        catalog_header,
        {
            "plan_header": plan_header_artifact,
            "plan_checksums": checksums_artifact,
            "catalog_header": catalog_header_artifact,
        },
        release_contract,
    )


def collect_release_snapshot(
    *,
    repo: Path,
    config: Path,
    plan: Path,
    catalog: Path,
    sampling: Path,
    animefreq: Path,
    sidecar: Path,
    external_bindings: dict[str, Path],
    tracker: SnapshotTracker,
) -> dict[str, Any]:
    for label, path in {
        "repo": repo,
        "plan": plan,
        "catalog": catalog,
        "animefreq": animefreq,
        "sidecar": sidecar,
    }.items():
        if not path.is_absolute():
            raise ValueError(f"{label} path must be absolute: {path}")
        directory_metadata(path)
    if not sampling.is_absolute():
        raise ValueError(f"sampling path must be absolute: {sampling}")
    if not config.is_absolute():
        raise ValueError(f"config path must be absolute: {config}")
    regular_metadata(config)
    for name, path in external_bindings.items():
        if not path.is_absolute():
            raise ValueError(f"external binding {name} path must be absolute: {path}")
        regular_metadata(path)

    expected_contract, sampling_evidence = load_stage_contract(
        config, sampling, tracker
    )
    (
        plan_header,
        plan_checksums,
        catalog_header,
        header_artifacts,
        release_contract,
    ) = validate_common_headers(
        plan, catalog, expected_contract, tracker
    )
    validate_stage512a_catalog_sampling(
        plan_header, catalog_header, release_contract, sampling_evidence
    )
    validate_external_bindings(
        external_bindings,
        release_contract,
        plan_header,
        catalog_header,
        header_artifacts["plan_header"]["sha256"],
        header_artifacts["catalog_header"]["sha256"],
        repo,
        tracker,
    )
    catalogs = catalog_domains(catalog_header)
    anime_header, anime_section = validate_animefreq(
        animefreq, catalog, catalog_header["catalog_hash"], tracker
    )
    plan_section = validate_plan_blocks(plan, plan_checksums, tracker)
    catalog_section = validate_catalog_files(catalog, catalogs, tracker)
    sidecar_section = validate_sidecar(
        sidecar, catalog, full=True, parent_tracker=tracker
    )
    if sidecar_section["catalog_hash"] != catalog_header["catalog_hash"]:
        raise ValueError("sidecar/catalog identity mismatch")

    source_explicit = source_explicit_for_stage(release_contract["stage"])
    sources = discover_training_sources(repo, release_contract["stage"])
    source_files = {
        relative: tracker.artifact(path) for relative, path in sorted(sources.items())
    }
    input_paths = expected_inputs(
        repo, config, plan, catalog, sampling, animefreq, sidecar, external_bindings
    )
    inputs = {label: tracker.artifact(path) for label, path in sorted(input_paths.items())}
    duplicated_controls = {
        "animefreq_header": anime_section["header"],
        "animefreq_success": anime_section["success"],
        "sidecar_header": sidecar_section["controls"]["header"],
        "sidecar_success": sidecar_section["controls"]["success"],
        "catalog_header": sidecar_section["controls"]["catalog_header"],
    }
    for label, artifact in duplicated_controls.items():
        if inputs[label] != artifact:
            raise ValueError(f"duplicated release control snapshot mismatch: {label}")

    # Reparse every control only after every large hash/footer pass has ended.
    closing_plan_header, _ = read_json_snapshot(plan / "header.json", tracker)
    closing_plan_checksums, _ = read_json_snapshot(plan / "checksums.json", tracker)
    closing_catalog_header, _ = read_json_snapshot(catalog / "header.json", tracker)
    closing_anime_header, _ = read_json_snapshot(animefreq / "header.json", tracker)
    closing_anime_success = tracker.read_bytes(animefreq / "_SUCCESS")
    closing_sidecar_header, closing_sidecar_files, _ = parse_sidecar_header(
        sidecar, catalog, tracker
    )
    if (
        closing_plan_header != plan_header
        or closing_plan_checksums != plan_checksums
        or closing_catalog_header != catalog_header
        or closing_anime_header != anime_header
        or closing_anime_success != b""
        or closing_sidecar_header.get("header_sha256") != sidecar_section["header_sha256"]
        or sum(len(value) for value in closing_sidecar_files.values())
        != sidecar_section["file_count"]
    ):
        raise ValueError("release controls changed during snapshot collection")

    # Final exact-set phase, followed by one uniform five-tuple restat covering
    # controls, directories and every hashed data/source file.
    exact_plan_root(plan, plan_checksums["blocks"], tracker)
    exact_animefreq_root(animefreq, anime_header["files"], tracker)
    exact_catalog_declared(catalog, catalogs, tracker)
    exact_sidecar_root(sidecar, closing_sidecar_files, tracker)
    closing_sources = discover_training_sources(repo, release_contract["stage"])
    exact_set("training source", closing_sources, sources)
    for relative, path in closing_sources.items():
        tracker.remember(path, "file", regular_metadata(path))
    tracker.close()

    identities = {
        "plan_id": plan_header.get("plan_id"),
        "catalog_hash": catalog_header.get("catalog_hash"),
        "assignment_hash": catalog_header.get("assignment_hash"),
        **{key: release_contract[key] for key in CONTRACT_KEYS},
        "caption_sidecar_header_sha256": sidecar_section["header_sha256"],
        "caption_sidecar_release_digest": sidecar_section["release_digest"],
    }
    for key in (
        "plan_id",
        "catalog_hash",
        "assignment_hash",
        "caption_sidecar_header_sha256",
        "caption_sidecar_release_digest",
    ):
        if not identities[key]:
            raise ValueError(f"release identity {key} is empty")
    for key in CONTRACT_KEYS:
        if type(identities[key]) is not type(expected_contract[key]):
            raise ValueError(f"release contract identity {key} has the wrong type")
    return {
        "identities": identities,
        "inputs": inputs,
        "plan_blocks": plan_section,
        "catalog": catalog_section,
        "animefreq": anime_section,
        "caption_sidecar": sidecar_section,
        "training_sources": {
            "root": str(repo),
            "discovery": {
                "explicit": list(source_explicit),
                "python_trees": list(SOURCE_TREES),
            },
            "files": source_files,
        },
        "snapshot_fingerprints_sha256": tracker.evidence_digest(),
        "control_artifacts": header_artifacts,
    }


def build_manifest(
    *,
    repo: Path,
    config: Path,
    plan: Path,
    catalog: Path,
    sampling: Path,
    animefreq: Path,
    sidecar: Path,
    external_bindings: dict[str, Path],
) -> tuple[dict[str, Any], str]:
    tracker = SnapshotTracker()
    git_head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    release = collect_release_snapshot(
        repo=repo,
        config=config,
        plan=plan,
        catalog=catalog,
        sampling=sampling,
        animefreq=animefreq,
        sidecar=sidecar,
        external_bindings=external_bindings,
        tracker=tracker,
    )
    control_artifacts = release.pop("control_artifacts")
    # These are already represented byte-for-byte in inputs; retain the
    # consistency assertion without duplicating another manifest section.
    if any(release["inputs"][label] != artifact for label, artifact in control_artifacts.items()):
        raise ValueError("release control artifact/input mismatch")
    snapshot_digest = release.pop("snapshot_fingerprints_sha256")
    freeze_schema = (
        STAGE512A_FREEZE_SCHEMA
        if release["identities"]["stage"] == "512a"
        else STAGE256_FREEZE_SCHEMA
    )
    manifest = {
        "schema_version": freeze_schema,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_head_at_freeze": git_head,
        **release,
    }
    return manifest, snapshot_digest


def atomic_write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise ValueError(f"freeze output must be absolute: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError(f"freeze output parent must be an existing real directory: {path.parent}")
    if path.parent.resolve(strict=True) != path.parent:
        raise ValueError(f"freeze output parent must be canonical: {path.parent}")
    payload = (
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o444)
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError(f"short freeze write: {path}")
                offset += written
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except FileExistsError as exc:
        raise ValueError(f"immutable freeze output already exists: {path}") from exc
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise ValueError(f"published freeze mode must be 0444: {path}")


def freeze_release_digest(manifest: dict[str, Any]) -> str:
    """Canonical semantic release identity, excluding informational timestamps."""

    keys = (
        "schema_version",
        "identities",
        "inputs",
        "plan_blocks",
        "catalog",
        "animefreq",
        "caption_sidecar",
        "training_sources",
    )
    return canonical_json_sha256({key: manifest[key] for key in keys})


def verify_freeze(
    freeze_path: Path,
    *,
    repo: Path,
    config: Path,
    plan: Path,
    catalog: Path,
    sampling: Path,
    animefreq: Path,
    sidecar: Path,
    external_bindings: dict[str, Path],
) -> dict[str, Any]:
    tracker = SnapshotTracker()
    manifest, freeze_artifact = read_json_snapshot(freeze_path, tracker)
    expected_keys = {
        "schema_version",
        "frozen_at",
        "git_head_at_freeze",
        "identities",
        "inputs",
        "plan_blocks",
        "catalog",
        "animefreq",
        "caption_sidecar",
        "training_sources",
    }
    if set(manifest) != expected_keys:
        raise ValueError(
            f"freeze top-level key set drift: missing={sorted(expected_keys-set(manifest))}, "
            f"unexpected={sorted(set(manifest)-expected_keys)}"
        )
    if manifest.get("schema_version") not in {
        STAGE256_FREEZE_SCHEMA,
        STAGE512A_FREEZE_SCHEMA,
    }:
        raise ValueError(f"unsupported freeze schema: {manifest.get('schema_version')!r}")
    if manifest.get("schema_version") == STAGE512A_FREEZE_SCHEMA:
        validate_stage512a_reviewed_freeze_storage(
            freeze_path, require_exists=True
        )
    if (
        not isinstance(manifest.get("frozen_at"), str)
        or not manifest["frozen_at"]
        or not isinstance(manifest.get("git_head_at_freeze"), str)
        or not manifest["git_head_at_freeze"]
    ):
        raise ValueError("freeze frozen_at/git_head_at_freeze is malformed")

    live = collect_release_snapshot(
        repo=repo,
        config=config,
        plan=plan,
        catalog=catalog,
        sampling=sampling,
        animefreq=animefreq,
        sidecar=sidecar,
        external_bindings=external_bindings,
        tracker=tracker,
    )
    expected_schema = (
        STAGE512A_FREEZE_SCHEMA
        if live["identities"]["stage"] == "512a"
        else STAGE256_FREEZE_SCHEMA
    )
    if manifest.get("schema_version") != expected_schema:
        raise ValueError(
            "freeze schema does not match the selected stage contract: "
            f"expected={expected_schema!r}, actual={manifest.get('schema_version')!r}"
        )
    live.pop("snapshot_fingerprints_sha256")
    control_artifacts = live.pop("control_artifacts")
    if any(live["inputs"][label] != artifact for label, artifact in control_artifacts.items()):
        raise ValueError("live release control artifact/input mismatch")
    # Existing production stage256 freezes predate the five-field identity
    # closure.  They remain verifiable only when the live catalog/plan passed
    # the recognized legacy-stage256 checks above.  New/partial identities are
    # never projected.
    frozen_identities = manifest.get("identities")
    live_identities = live.get("identities")
    if not isinstance(frozen_identities, dict) or not isinstance(live_identities, dict):
        raise ValueError("freeze/live identities must be objects")
    frozen_contract_keys = {key for key in CONTRACT_KEYS if key in frozen_identities}
    if not frozen_contract_keys:
        if (
            live_identities.get("stage") != "256"
            or live_identities.get("assignment_hash") != LEGACY_STAGE256_ASSIGNMENT
        ):
            raise ValueError("contract-less freeze is not a verified legacy stage256 release")
        comparable_identities = {
            key: value for key, value in live_identities.items() if key not in CONTRACT_KEYS
        }
    elif frozen_contract_keys == set(CONTRACT_KEYS):
        comparable_identities = live_identities
    else:
        raise ValueError("frozen identities contain a partial stage contract")
    if frozen_identities != comparable_identities:
        raise ValueError("live release section differs from freeze: identities")

    for section in (
        "inputs",
        "plan_blocks",
        "catalog",
        "animefreq",
        "caption_sidecar",
        "training_sources",
    ):
        if manifest.get(section) != live[section]:
            raise ValueError(f"live release section differs from freeze: {section}")
    tracker.close()
    release_digest = freeze_release_digest(manifest)
    return {
        "status": "PASS",
        "mode": "verify-freeze",
        "frozen_at": manifest["frozen_at"],
        "release_digest": release_digest,
        "freeze_sha256": freeze_artifact["sha256"],
        "freeze_file": freeze_artifact,
        "snapshot_fingerprints_sha256": tracker.evidence_digest(),
        **{
            key: live["caption_sidecar"][key]
            for key in (
                "header_sha256",
                "header_file_sha256",
                "success_file_sha256",
                "catalog_hash",
                "catalog_header_sha256",
                "file_count",
                "total_bytes",
            )
        },
        "sidecar_release_digest": live["caption_sidecar"]["release_digest"],
        "training_sources": len(live["training_sources"]["files"]),
        "catalog_files": len(live["catalog"]["files"]),
        "plan_blocks": len(live["plan_blocks"]["files"]),
    }


def sidecar_summary(value: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "status": "PASS",
        "mode": mode,
        **{
            key: value[key]
            for key in (
                "root",
                "header_sha256",
                "header_file_sha256",
                "success_file_sha256",
                "catalog_hash",
                "catalog_header_sha256",
                "release_digest",
                "file_count",
                "total_bytes",
                "snapshot_fingerprints_sha256",
            )
        },
    }


def parse_external_bindings(values: Iterable[str]) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    for raw in values:
        name, separator, path_raw = raw.partition("=")
        if (
            not separator
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
            or not path_raw
        ):
            raise ValueError(
                f"--external-binding must be NAME=/absolute/path, got {raw!r}"
            )
        if name in bindings:
            raise ValueError(f"duplicate external binding name: {name}")
        path = Path(path_raw).expanduser()
        if not path.is_absolute():
            raise ValueError(f"external binding path must be absolute: {path}")
        bindings[name] = path.resolve(strict=False)
    return bindings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-sidecar-control", action="store_true")
    modes.add_argument("--verify-sidecar", action="store_true")
    modes.add_argument("--verify-freeze", action="store_true")
    parser.add_argument("--freeze", type=Path, default=OUT)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="selected training config; default is <repo>/configs/stage256_offline_exact_v1.yaml",
    )
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--catalog-root", type=Path, default=CATALOG)
    parser.add_argument("--sampling", type=Path, default=SAMPLING)
    parser.add_argument("--animefreq-root", type=Path, default=ANIME_FREQ)
    parser.add_argument("--sidecar-root", type=Path, default=SIDECAR)
    parser.add_argument(
        "--external-binding",
        action="append",
        default=[],
        metavar="NAME=/ABSOLUTE/PATH",
        help="freeze an external stage artifact; stage512a requires gpic_payload=<JSON>",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        args.freeze = args.freeze.expanduser().resolve(strict=False)
        args.repo = args.repo.expanduser().resolve(strict=False)
        if args.config is None:
            args.config = args.repo / DEFAULT_CONFIG_REL
        elif not args.config.is_absolute():
            args.config = args.repo / args.config
        args.config = args.config.expanduser().resolve(strict=False)
        args.plan = args.plan.expanduser().resolve(strict=False)
        args.catalog_root = args.catalog_root.expanduser().resolve(strict=False)
        args.sampling = args.sampling.expanduser().resolve(strict=False)
        args.animefreq_root = args.animefreq_root.expanduser().resolve(strict=False)
        args.sidecar_root = args.sidecar_root.expanduser().resolve(strict=False)
        external_bindings = parse_external_bindings(args.external_binding)
        if args.verify_sidecar_control:
            value = validate_sidecar(args.sidecar_root, args.catalog_root, full=False)
            result = sidecar_summary(value, "verify-sidecar-control")
        elif args.verify_sidecar:
            value = validate_sidecar(args.sidecar_root, args.catalog_root, full=True)
            result = sidecar_summary(value, "verify-sidecar")
        elif args.verify_freeze:
            result = verify_freeze(
                args.freeze,
                repo=args.repo,
                config=args.config,
                plan=args.plan,
                catalog=args.catalog_root,
                sampling=args.sampling,
                animefreq=args.animefreq_root,
                sidecar=args.sidecar_root,
                external_bindings=external_bindings,
            )
        else:
            manifest, source_snapshot_digest = build_manifest(
                repo=args.repo,
                config=args.config,
                plan=args.plan,
                catalog=args.catalog_root,
                sampling=args.sampling,
                animefreq=args.animefreq_root,
                sidecar=args.sidecar_root,
                external_bindings=external_bindings,
            )
            if manifest["identities"]["stage"] == "512a":
                validate_stage512a_reviewed_freeze_storage(
                    args.freeze, require_exists=False
                )
            atomic_write_manifest(args.freeze, manifest)
            freeze_tracker = SnapshotTracker()
            written, freeze_artifact = read_json_snapshot(args.freeze, freeze_tracker)
            if written != manifest:
                raise ValueError("atomically written freeze differs from in-memory manifest")
            freeze_tracker.close()
            result = {
                "status": "PASS",
                "mode": "create-freeze",
                "path": str(args.freeze),
                "frozen_at": manifest["frozen_at"],
                "release_digest": freeze_release_digest(manifest),
                "freeze_sha256": freeze_artifact["sha256"],
                "freeze_file": freeze_artifact,
                "source_snapshot_fingerprints_sha256": source_snapshot_digest,
                "freeze_snapshot_fingerprints_sha256": freeze_tracker.evidence_digest(),
                **manifest["identities"],
                "sidecar_files": manifest["caption_sidecar"]["file_count"],
                "sidecar_bytes": manifest["caption_sidecar"]["total_bytes"],
                "catalog_files": len(manifest["catalog"]["files"]),
                "plan_blocks": len(manifest["plan_blocks"]["files"]),
                "training_sources": len(manifest["training_sources"]["files"]),
            }
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        print(
            f"FATAL freeze/sidecar gate: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
