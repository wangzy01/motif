"""Reader and shared contract helpers for the 2026-07-18 semantic plan.

The legacy :class:`planner_v1.OfflinePlan` remains the authority for the
physical catalog and for the bucket/domain schedule.  A semantic plan changes
only ``item_row``: each value is an old-catalog global row id selected from a
checksum-bound eligibility/weight overlay.  This module deliberately lives in
a new top-level package so the frozen legacy ``training`` source tree is not
modified.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from training.dataline_offline.planner_v1 import OfflinePlan, PlannerInputError


PLAN_FORMAT = "offline_arb_semantic_plan_v1"
PLAN_FORMAT_VERSION = 1
OVERLAY_FORMAT = "offline_arb_semantic_overlay_v1"
OVERLAY_FORMAT_VERSION = 1
ALGORITHM_VERSION = "offline_arb_semantic_materializer_v1_20260718"
QUOTA_ALGORITHM = "capped_float64_largest_remainder_v1"
TICKET_ALGORITHM = "quota_layer_pcg64_boundary_swap_havel_fallback_v1"
BLOCK_FORMAT = "deterministic_stored_npz_npy2_three_datasets_v2"
BLOCK_DATASETS = ("bucket_of_step", "item_row", "domain_counts")
AUDIT_FORMAT = "offline_arb_semantic_plan_audit_v1"
SUCCESS_NAME = "PLAN_AUDIT_SUCCESS.json"
FORMAL_W64_ACTIVE_BUCKET_INDICES = (
    25,
    27,
    29,
    31,
    33,
    35,
    38,
    40,
    45,
    48,
    49,
    54,
    56,
    59,
    61,
    63,
    65,
    67,
    69,
    71,
)
FORMAL_W64_PREFIX_DOMAIN_TOTALS = {
    "anime": 103_680_207,
    "monet_real": 155_520_311,
    "gpic": 99_456_199,
    "synthetic": 19_584_039,
    "textatlas": 5_760_012,
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SemanticPlanError(RuntimeError):
    """A semantic plan or one of its immutable bindings is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticPlanError(f"cannot read JSON object {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise SemanticPlanError(f"expected a JSON object in {source}")
    return value


def write_json_create_once(path: str | os.PathLike[str], value: Any) -> None:
    """Atomically create one JSON file; never replace an existing artifact."""

    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    if target.exists() or temporary.exists():
        raise SemanticPlanError(f"refusing to overwrite create-once file: {target}")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        np.ascontiguousarray(array),
        version=(2, 0),
        allow_pickle=False,
    )
    return buffer.getvalue()


def write_deterministic_npz_create_once(
    path: str | os.PathLike[str], arrays: Sequence[tuple[str, np.ndarray]]
) -> None:
    """Create an uncompressed NPZ without wall-clock ZIP metadata."""

    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    if target.exists() or temporary.exists():
        raise SemanticPlanError(f"refusing to overwrite create-once NPZ: {target}")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="x",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for name, array in arrays:
                info = zipfile.ZipInfo(
                    f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                info.flag_bits = 0
                archive.writestr(
                    info,
                    _npy_bytes(array),
                    compress_type=zipfile.ZIP_STORED,
                )
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SemanticPlanError(f"{label} must be a lowercase SHA-256 string")
    return value


def require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SemanticPlanError(f"{label} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise SemanticPlanError(f"{label} must be >= {minimum}, got {result}")
    return result


def cell_key(bucket: int, domain: int) -> str:
    return f"{int(bucket)}:{int(domain)}"


def overlay_cell_key(bucket: int, domain_name: str) -> str:
    return f"{int(bucket)}:{domain_name}"


def resolve_bound_root(plan_root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise SemanticPlanError(f"{label} must be a non-empty path string")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = plan_root / path
    return path.resolve(strict=False)


def physical_cell_offsets(n_matrix: np.ndarray) -> np.ndarray:
    if n_matrix.ndim != 2 or not np.issubdtype(n_matrix.dtype, np.integer):
        raise SemanticPlanError("N must be a two-dimensional integer matrix")
    offsets = np.zeros_like(n_matrix, dtype=np.int64)
    cursor = 0
    for d in range(n_matrix.shape[1]):
        for b in range(n_matrix.shape[0]):
            offsets[b, d] = cursor
            cursor += int(n_matrix[b, d])
    return offsets


def load_overlay_header(overlay_root: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(overlay_root)
    header = read_json_object(root / "header.json")
    required = {
        "format",
        "format_version",
        "overlay_id",
        "catalog_hash",
        "catalog_header_sha256",
        "sampling_config_sha256",
        "domains",
        "buckets",
        "N_select",
        "excluded_row_ids",
        "semantic_assertions",
        "cells",
    }
    missing = sorted(required - set(header))
    if missing:
        raise SemanticPlanError(f"overlay header is missing keys: {missing}")
    if header["format"] != OVERLAY_FORMAT:
        raise SemanticPlanError(f"unsupported overlay format: {header['format']!r}")
    if require_int(header["format_version"], "overlay.format_version") != OVERLAY_FORMAT_VERSION:
        raise SemanticPlanError("unsupported overlay format_version")
    if not isinstance(header["overlay_id"], str) or not header["overlay_id"]:
        raise SemanticPlanError("overlay_id must be a non-empty string")
    for key in (
        "catalog_hash",
        "catalog_header_sha256",
        "sampling_config_sha256",
    ):
        require_sha256(header[key], f"overlay.{key}")
    domains = header["domains"]
    buckets = header["buckets"]
    if not isinstance(domains, list) or not domains or any(
        not isinstance(item, str) or not item for item in domains
    ):
        raise SemanticPlanError("overlay.domains must be non-empty domain names")
    if len(set(domains)) != len(domains):
        raise SemanticPlanError("overlay.domains contains duplicates")
    if not isinstance(buckets, list) or not buckets:
        raise SemanticPlanError("overlay.buckets must be non-empty")
    parsed_buckets: list[list[int]] = []
    for i, item in enumerate(buckets):
        if not isinstance(item, list) or len(item) != 2:
            raise SemanticPlanError(f"overlay.buckets[{i}] must be [height,width]")
        parsed_buckets.append(
            [
                require_int(item[0], f"overlay.buckets[{i}][0]", minimum=1),
                require_int(item[1], f"overlay.buckets[{i}][1]", minimum=1),
            ]
        )
    if len({tuple(item) for item in parsed_buckets}) != len(parsed_buckets):
        raise SemanticPlanError("overlay.buckets contains duplicates")
    n_select = np.asarray(header["N_select"])
    expected_shape = (len(parsed_buckets), len(domains))
    if n_select.shape != expected_shape or not np.issubdtype(n_select.dtype, np.integer):
        raise SemanticPlanError(
            f"overlay.N_select must be integer matrix {expected_shape}, got {n_select.shape}/{n_select.dtype}"
        )
    if np.any(n_select < 0):
        raise SemanticPlanError("overlay.N_select contains negative values")
    exclusions = header["excluded_row_ids"]
    if not isinstance(exclusions, list):
        raise SemanticPlanError("overlay.excluded_row_ids must be a list")
    parsed_exclusions = [
        require_int(value, f"overlay.excluded_row_ids[{i}]", minimum=0)
        for i, value in enumerate(exclusions)
    ]
    if len(set(parsed_exclusions)) != len(parsed_exclusions):
        raise SemanticPlanError("overlay.excluded_row_ids contains duplicates")
    assertions = header["semantic_assertions"]
    if not isinstance(assertions, Mapping):
        raise SemanticPlanError("overlay.semantic_assertions must be an object")
    if assertions.get("anime_all_rows_strip_nonempty_nl") is not True:
        raise SemanticPlanError(
            "overlay must assert anime_all_rows_strip_nonempty_nl=true"
        )
    if require_int(
        assertions.get("anime_tag_only_rows"),
        "overlay.semantic_assertions.anime_tag_only_rows",
        minimum=0,
    ) != 0:
        raise SemanticPlanError("overlay must assert anime_tag_only_rows=0")
    cells = header["cells"]
    expected_cells = {
        overlay_cell_key(b, domains[d])
        for b in range(len(parsed_buckets))
        for d in range(len(domains))
        if int(n_select[b, d]) > 0
    }
    if not isinstance(cells, Mapping) or set(cells) != expected_cells:
        raise SemanticPlanError(
            "overlay.cells must exactly cover positive N_select cells: "
            f"missing={sorted(expected_cells-set(cells) if isinstance(cells, Mapping) else expected_cells)}, "
            f"extra={sorted(set(cells)-expected_cells) if isinstance(cells, Mapping) else []}"
        )
    for key, info in cells.items():
        if not isinstance(info, Mapping):
            raise SemanticPlanError(f"overlay.cells[{key!r}] must be an object")
        if not {"path", "sha256", "rows"}.issubset(info):
            raise SemanticPlanError(
                f"overlay.cells[{key!r}] must contain path/sha256/rows"
            )
        bucket_text, separator, domain_name = key.partition(":")
        if not separator or domain_name not in domains:
            raise SemanticPlanError(
                f"overlay cell key {key!r} must be '<bucket>:<domain_name>'"
            )
        try:
            b = int(bucket_text)
        except ValueError as exc:
            raise SemanticPlanError(f"overlay cell key {key!r} has invalid bucket") from exc
        d = domains.index(domain_name)
        if b < 0 or b >= len(parsed_buckets):
            raise SemanticPlanError(f"overlay cell key {key!r} bucket is outside table")
        if not isinstance(info["path"], str) or not info["path"]:
            raise SemanticPlanError(f"overlay cell {key} path is invalid")
        require_sha256(info["sha256"], f"overlay.cells[{key}].sha256")
        if require_int(info["rows"], f"overlay.cells[{key}].rows", minimum=1) != int(
            n_select[b, d]
        ):
            raise SemanticPlanError(f"overlay cell {key} row count disagrees with N_select")
    return header


def load_overlay_cell(
    overlay_root: str | os.PathLike[str],
    overlay_header: Mapping[str, Any],
    bucket: int,
    domain: int,
    *,
    verify_checksum: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    domains = overlay_header.get("domains")
    if not isinstance(domains, list) or domain < 0 or domain >= len(domains):
        raise SemanticPlanError(f"overlay domain index outside table: {domain}")
    key = overlay_cell_key(bucket, str(domains[domain]))
    info = overlay_header["cells"].get(key)
    if info is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    path = Path(overlay_root) / str(info["path"])
    if not path.is_file():
        raise SemanticPlanError(f"missing overlay cell file: {path}")
    if verify_checksum:
        actual = sha256_file(path)
        if actual != info["sha256"]:
            raise SemanticPlanError(
                f"overlay cell checksum mismatch for {key}: expected={info['sha256']}, actual={actual}"
            )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not {"row_id", "weight"}.issubset(archive.files):
                raise SemanticPlanError(
                    f"overlay cell {key} must contain row_id/weight"
                )
            row_ids = np.array(archive["row_id"], copy=True)
            weights = np.array(archive["weight"], copy=True)
            for optional_name in sorted(set(archive.files) - {"row_id", "weight"}):
                optional = np.asarray(archive[optional_name])
                if optional.shape != (int(info["rows"]),):
                    raise SemanticPlanError(
                        f"overlay cell {key} optional array {optional_name!r} "
                        f"has shape {optional.shape}, expected {(int(info['rows']),)}"
                    )
                if optional.dtype.hasobject:
                    raise SemanticPlanError(
                        f"overlay cell {key} optional array {optional_name!r} uses object dtype"
                    )
                if np.issubdtype(optional.dtype, np.floating) and np.any(~np.isfinite(optional)):
                    raise SemanticPlanError(
                        f"overlay cell {key} optional array {optional_name!r} contains non-finite values"
                    )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SemanticPlanError(f"cannot load overlay cell {path}: {exc}") from exc
    expected_rows = int(info["rows"])
    if row_ids.dtype != np.dtype(np.int64) or row_ids.shape != (expected_rows,):
        raise SemanticPlanError(
            f"overlay cell {key} row_id has {row_ids.dtype}/{row_ids.shape}, expected int64/{(expected_rows,)}"
        )
    if weights.dtype != np.dtype(np.float64) or weights.shape != (expected_rows,):
        raise SemanticPlanError(
            f"overlay cell {key} weight has {weights.dtype}/{weights.shape}, expected float64/{(expected_rows,)}"
        )
    if expected_rows and np.any(row_ids[1:] <= row_ids[:-1]):
        raise SemanticPlanError(f"overlay cell {key} row_id must be strictly increasing")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise SemanticPlanError(f"overlay cell {key} weights must be finite and positive")
    return row_ids, weights


def binding_for_plan(root: str | os.PathLike[str], plan: OfflinePlan) -> dict[str, Any]:
    resolved = Path(root).resolve(strict=False)
    return {
        "root": str(resolved),
        "plan_id": str(plan.header["plan_id"]),
        "plan_hash": str(plan.header["plan_hash"]),
        "header_sha256": sha256_file(resolved / "header.json"),
        "checksums_sha256": sha256_file(resolved / "checksums.json"),
    }


def _validate_plan_binding(
    semantic_root: Path,
    name: str,
    raw: Any,
    *,
    world_size: int | None = None,
) -> OfflinePlan:
    if not isinstance(raw, Mapping):
        raise SemanticPlanError(f"bindings.{name} must be an object")
    required = {"root", "plan_id", "plan_hash", "header_sha256", "checksums_sha256"}
    if set(raw) != required:
        raise SemanticPlanError(f"bindings.{name} must contain exactly {sorted(required)}")
    root = resolve_bound_root(semantic_root, raw["root"], f"bindings.{name}.root")
    for filename, key in (("header.json", "header_sha256"), ("checksums.json", "checksums_sha256")):
        expected = require_sha256(raw[key], f"bindings.{name}.{key}")
        actual = sha256_file(root / filename)
        if actual != expected:
            raise SemanticPlanError(
                f"{name} {filename} binding mismatch: expected={expected}, actual={actual}"
            )
    try:
        plan = OfflinePlan(
            root,
            world_size=world_size,
            verify_checksums=True,
        )
    except PlannerInputError as exc:
        raise SemanticPlanError(f"invalid bound {name}: {exc}") from exc
    if plan.header["plan_id"] != raw["plan_id"] or plan.header["plan_hash"] != raw["plan_hash"]:
        raise SemanticPlanError(f"{name} plan identity differs from semantic binding")
    return plan


class SemanticOfflinePlan:
    """Checksum-validating reader with the legacy ``OfflinePlan`` batch API."""

    def __init__(
        self,
        plan_root: str | os.PathLike[str],
        *,
        expected_input: Any | None = None,
        world_size: int | None = None,
        per_rank_bs: int | None = None,
        verify_checksums: bool = True,
        require_audit: bool = True,
    ):
        self.plan_root = Path(plan_root).resolve(strict=False)
        if not self.plan_root.is_dir():
            raise SemanticPlanError(f"semantic plan root is not a directory: {self.plan_root}")
        self.header = read_json_object(self.plan_root / "header.json")
        self._validate_header_shape()

        requested_world = (
            None
            if world_size is None
            else require_int(world_size, "world_size", minimum=1)
        )
        requested_bs = (
            None
            if per_rank_bs is None
            else require_int(per_rank_bs, "per_rank_bs", minimum=1)
        )
        if requested_world is not None and requested_world != int(self.header["world_size"]):
            raise SemanticPlanError(
                f"world_size mismatch: plan={self.header['world_size']}, requested={requested_world}"
            )
        if requested_bs is not None and requested_bs != int(self.header["per_rank_bs"]):
            raise SemanticPlanError(
                f"per_rank_bs mismatch: plan={self.header['per_rank_bs']}, requested={requested_bs}"
            )
        if expected_input is not None:
            raise SemanticPlanError(
                "SemanticOfflinePlan does not accept a legacy expected_input; bind the selection plan in header.json"
            )

        bindings = self.header["bindings"]
        self.base_plan = _validate_plan_binding(
            self.plan_root, "base_physical_plan", bindings["base_physical_plan"]
        )
        self.selection_plan = _validate_plan_binding(
            self.plan_root,
            "selection_plan",
            bindings["selection_plan"],
            world_size=int(self.header["world_size"]),
        )
        self._validate_bound_semantics()
        self._validate_checksums_document()
        self.verify_checksums = bool(verify_checksums)
        self._verified_blocks: set[int] = set()
        self._cached_block_index: int | None = None
        self._cached_bucket: np.ndarray | None = None
        self._cached_items: np.ndarray | None = None
        self._cached_domain_counts: np.ndarray | None = None
        self.first_materialized_block = int(
            self.header["materialization"]["first_materialized_block"]
        )
        self.config = SimpleNamespace(
            total_steps=int(self.header["total_steps"]),
            block_steps=int(self.header["block_steps"]),
            world_size=int(self.header["world_size"]),
            per_rank_bs=int(self.header["per_rank_bs"]),
            G=int(self.header["G"]),
            B=len(self.header["buckets"]),
            K=len(self.header["domains"]),
            buckets=tuple(tuple(int(v) for v in hw) for hw in self.header["buckets"]),
            N=np.asarray(self.header["N"], dtype=np.int64),
        )
        if require_audit:
            self._validate_audit_receipt()

    def _validate_header_shape(self) -> None:
        required = {
            "format_version",
            "plan_format",
            "algorithm_version",
            "algorithm_versions",
            "plan_id",
            "plan_hash",
            "world_size",
            "per_rank_bs",
            "total_steps",
            "G",
            "global_batch_size",
            "total_exposures",
            "block_steps",
            "num_blocks",
            "block_datasets",
            "domains",
            "domain_mix",
            "square_only_domains",
            "buckets",
            "square_bucket_idx",
            "N",
            "N_select",
            "R_max",
            "D",
            "S",
            "C",
            "catalog_hash",
            "assignment_hash",
            "upstream_catalog_hash",
            "upstream_assignment_hash",
            "catalog_size",
            "cell_row_offset",
            "excluded_row_ids",
            "audit_contract",
            "materialization",
            "bindings",
            "quota_files",
            "hashes",
        }
        missing = sorted(required - set(self.header))
        if missing:
            raise SemanticPlanError(f"semantic header is missing keys: {missing}")
        if require_int(self.header["format_version"], "format_version") != PLAN_FORMAT_VERSION:
            raise SemanticPlanError("unsupported semantic plan format_version")
        if self.header["plan_format"] != PLAN_FORMAT:
            raise SemanticPlanError(f"unsupported plan_format: {self.header['plan_format']!r}")
        if self.header["algorithm_version"] != ALGORITHM_VERSION:
            raise SemanticPlanError(
                f"algorithm_version mismatch: {self.header['algorithm_version']!r}"
            )
        expected_algorithms = {
            "materializer": ALGORITHM_VERSION,
            "quota": QUOTA_ALGORITHM,
            "ticket_permutation": TICKET_ALGORITHM,
            "block_format": BLOCK_FORMAT,
        }
        if self.header["algorithm_versions"] != expected_algorithms:
            raise SemanticPlanError("algorithm_versions mismatch")
        if self.header["block_datasets"] != list(BLOCK_DATASETS):
            raise SemanticPlanError("block_datasets mismatch")
        require_sha256(self.header["plan_hash"], "plan_hash")
        catalog_hash = require_sha256(self.header["catalog_hash"], "catalog_hash")
        assignment_hash = require_sha256(
            self.header["assignment_hash"], "assignment_hash"
        )
        if require_sha256(
            self.header["upstream_catalog_hash"], "upstream_catalog_hash"
        ) != catalog_hash:
            raise SemanticPlanError("upstream_catalog_hash differs from catalog_hash")
        if require_sha256(
            self.header["upstream_assignment_hash"], "upstream_assignment_hash"
        ) != assignment_hash:
            raise SemanticPlanError(
                "upstream_assignment_hash differs from assignment_hash"
            )
        if not isinstance(self.header["plan_id"], str) or not self.header["plan_id"]:
            raise SemanticPlanError("plan_id must be a non-empty string")
        world = require_int(self.header["world_size"], "world_size", minimum=1)
        per_rank = require_int(self.header["per_rank_bs"], "per_rank_bs", minimum=1)
        total_steps = require_int(self.header["total_steps"], "total_steps", minimum=1)
        block_steps = require_int(self.header["block_steps"], "block_steps", minimum=1)
        expected_num_blocks = (total_steps + block_steps - 1) // block_steps
        if require_int(self.header["num_blocks"], "num_blocks", minimum=1) != expected_num_blocks:
            raise SemanticPlanError("num_blocks mismatch")
        if self.header["G"] != world * per_rank or self.header["global_batch_size"] != world * per_rank:
            raise SemanticPlanError("G/global_batch_size mismatch")
        if self.header["total_exposures"] != total_steps * world * per_rank:
            raise SemanticPlanError("total_exposures mismatch")
        audit_contract = self.header["audit_contract"]
        if not isinstance(audit_contract, Mapping) or set(audit_contract) != {
            "formal_prefix_steps",
            "formal_prefix_exposures",
            "formal_prefix_domain_totals",
            "active_native_buckets",
            "active_native_bucket_indices",
            "excluded_rows",
        }:
            raise SemanticPlanError("audit_contract has the wrong shape")
        prefix_steps = require_int(
            audit_contract["formal_prefix_steps"],
            "audit_contract.formal_prefix_steps",
            minimum=1,
        )
        if prefix_steps > total_steps:
            raise SemanticPlanError("audit formal prefix exceeds total_steps")
        if audit_contract["formal_prefix_exposures"] != prefix_steps * world * per_rank:
            raise SemanticPlanError("audit formal prefix exposure product mismatch")
        prefix_domain_totals = audit_contract["formal_prefix_domain_totals"]
        if not isinstance(prefix_domain_totals, Mapping) or set(prefix_domain_totals) != set(
            self.header["domains"]
        ):
            raise SemanticPlanError(
                "audit formal_prefix_domain_totals must map every domain exactly"
            )
        parsed_prefix_domain_total = 0
        for domain in self.header["domains"]:
            parsed_prefix_domain_total += require_int(
                prefix_domain_totals[domain],
                f"audit_contract.formal_prefix_domain_totals.{domain}",
                minimum=0,
            )
        if parsed_prefix_domain_total != audit_contract["formal_prefix_exposures"]:
            raise SemanticPlanError(
                "audit formal prefix domain totals do not sum to prefix exposures"
            )
        require_int(
            audit_contract["active_native_buckets"],
            "audit_contract.active_native_buckets",
            minimum=1,
        )
        active_indices = audit_contract["active_native_bucket_indices"]
        if not isinstance(active_indices, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in active_indices
        ):
            raise SemanticPlanError(
                "audit_contract.active_native_bucket_indices must be an integer list"
            )
        if active_indices != sorted(set(active_indices)):
            raise SemanticPlanError(
                "audit active native bucket indices must be sorted and unique"
            )
        if len(active_indices) != audit_contract["active_native_buckets"]:
            raise SemanticPlanError("audit active bucket count/list mismatch")
        require_int(
            audit_contract["excluded_rows"],
            "audit_contract.excluded_rows",
            minimum=0,
        )
        domains = self.header["domains"]
        buckets = self.header["buckets"]
        if not isinstance(domains, list) or not domains or len(set(domains)) != len(domains):
            raise SemanticPlanError("domains must be a unique non-empty list")
        if not isinstance(buckets, list) or not buckets:
            raise SemanticPlanError("buckets must be a non-empty list")
        shape = (len(buckets), len(domains))
        for key in ("N", "N_select", "C"):
            value = np.asarray(self.header[key])
            if value.shape != shape or not np.issubdtype(value.dtype, np.integer) or np.any(value < 0):
                raise SemanticPlanError(f"{key} must be a nonnegative integer matrix {shape}")
        if np.any(np.asarray(self.header["N_select"], dtype=np.int64) > np.asarray(self.header["N"], dtype=np.int64)):
            raise SemanticPlanError("N_select cannot exceed physical N")
        if len(self.header["S"]) != len(buckets) or len(self.header["D"]) != len(domains):
            raise SemanticPlanError("D/S dimensions mismatch")
        s_active_indices = [
            index for index, value in enumerate(self.header["S"]) if int(value) > 0
        ]
        if audit_contract["active_native_bucket_indices"] != s_active_indices:
            raise SemanticPlanError("audit active-native-bucket contract disagrees with S")
        if int(self.header["world_size"]) == 64 and s_active_indices != list(
            FORMAL_W64_ACTIVE_BUCKET_INDICES
        ):
            raise SemanticPlanError(
                "formal W64 active bucket indices differ from the frozen Stage256 contract"
            )
        if int(self.header["world_size"]) == 64 and dict(prefix_domain_totals) != dict(
            FORMAL_W64_PREFIX_DOMAIN_TOTALS
        ):
            raise SemanticPlanError(
                "formal W64 prefix domain totals differ from the frozen Stage256 contract"
            )
        if audit_contract["excluded_rows"] != len(self.header["excluded_row_ids"]):
            raise SemanticPlanError("audit excluded-row contract disagrees with header exclusions")
        offsets = physical_cell_offsets(np.asarray(self.header["N"], dtype=np.int64))
        if self.header["cell_row_offset"] != offsets.tolist():
            raise SemanticPlanError("cell_row_offset disagrees with physical N")
        if self.header["catalog_size"] != int(np.asarray(self.header["N"], dtype=np.int64).sum()):
            raise SemanticPlanError("catalog_size disagrees with physical N")
        hashes = self.header["hashes"]
        if not isinstance(hashes, Mapping):
            raise SemanticPlanError("hashes must be an object")
        for key in (
            "blocks_manifest_sha256",
            "quota_manifest_sha256",
            "identity_payload_sha256",
            "catalog_sha256",
            "assignment_sha256",
        ):
            require_sha256(hashes.get(key), f"hashes.{key}")
        if hashes["catalog_sha256"] != catalog_hash:
            raise SemanticPlanError("hashes.catalog_sha256 differs from catalog_hash")
        if hashes["assignment_sha256"] != assignment_hash:
            raise SemanticPlanError(
                "hashes.assignment_sha256 differs from assignment_hash"
            )
        identity = self._identity_payload()
        actual_identity = sha256_bytes(canonical_json_bytes(identity))
        if actual_identity != hashes["identity_payload_sha256"]:
            raise SemanticPlanError("identity payload hash mismatch")
        if actual_identity != self.header["plan_hash"]:
            raise SemanticPlanError("plan_hash must equal identity payload hash")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "plan_format": self.header["plan_format"],
            "algorithm_versions": self.header["algorithm_versions"],
            "plan_id": self.header["plan_id"],
            "world_size": self.header["world_size"],
            "per_rank_bs": self.header["per_rank_bs"],
            "total_steps": self.header["total_steps"],
            "domains": self.header["domains"],
            "buckets": self.header["buckets"],
            "N": self.header["N"],
            "N_select": self.header["N_select"],
            "R_max": self.header["R_max"],
            "D": self.header["D"],
            "S": self.header["S"],
            "C": self.header["C"],
            "excluded_row_ids": self.header["excluded_row_ids"],
            "audit_contract": self.header["audit_contract"],
            "catalog_hash": self.header["catalog_hash"],
            "assignment_hash": self.header["assignment_hash"],
            "upstream_catalog_hash": self.header["upstream_catalog_hash"],
            "upstream_assignment_hash": self.header["upstream_assignment_hash"],
            "bindings": self.header["bindings"],
            "blocks_manifest_sha256": self.header["hashes"]["blocks_manifest_sha256"],
            "quota_manifest_sha256": self.header["hashes"]["quota_manifest_sha256"],
        }

    def _validate_bound_semantics(self) -> None:
        selection = self.selection_plan.header
        base = self.base_plan.header
        exact_selection = {
            "world_size": self.header["world_size"],
            "per_rank_bs": self.header["per_rank_bs"],
            "total_steps": self.header["total_steps"],
            "block_steps": self.header["block_steps"],
            "domains": self.header["domains"],
            "buckets": self.header["buckets"],
            "square_only_domains": self.header["square_only_domains"],
            "square_bucket_idx": self.header["square_bucket_idx"],
            "N": self.header["N_select"],
            "D": self.header["D"],
            "S": self.header["S"],
            "C": self.header["C"],
        }
        for key, expected in exact_selection.items():
            if selection[key] != expected:
                raise SemanticPlanError(f"selection plan {key} differs from semantic header")
        for key in ("domains", "buckets", "square_only_domains", "square_bucket_idx"):
            if base[key] != self.header[key]:
                raise SemanticPlanError(f"base physical plan {key} differs from semantic header")
        if base["N"] != self.header["N"]:
            raise SemanticPlanError("base physical plan N differs from semantic physical N")
        if base["catalog_size"] != self.header["catalog_size"]:
            raise SemanticPlanError("base physical catalog_size mismatch")
        if base["cell_row_offset"] != self.header["cell_row_offset"]:
            raise SemanticPlanError("base physical cell offsets mismatch")
        for label, bound_plan in (
            ("base physical", base),
            ("selection", selection),
        ):
            catalog_expectations = {
                "catalog_hash": self.header["catalog_hash"],
                "upstream_catalog_hash": self.header["upstream_catalog_hash"],
                "assignment_hash": self.header["assignment_hash"],
                "upstream_assignment_hash": self.header[
                    "upstream_assignment_hash"
                ],
            }
            for key, expected in catalog_expectations.items():
                if bound_plan.get(key) != expected:
                    raise SemanticPlanError(
                        f"{label} plan {key} differs from semantic catalog identity"
                    )
            bound_hashes = bound_plan.get("hashes")
            if not isinstance(bound_hashes, Mapping):
                raise SemanticPlanError(f"{label} plan hashes are missing")
            if bound_hashes.get("catalog_sha256") != self.header["catalog_hash"]:
                raise SemanticPlanError(f"{label} plan catalog SHA binding mismatch")
            if (
                bound_hashes.get("assignment_sha256")
                != self.header["assignment_hash"]
            ):
                raise SemanticPlanError(
                    f"{label} plan assignment SHA binding mismatch"
                )
        bindings = self.header["bindings"]
        overlay_binding = bindings.get("overlay")
        if not isinstance(overlay_binding, Mapping) or set(overlay_binding) != {
            "root",
            "header_sha256",
            "overlay_id",
        }:
            raise SemanticPlanError("bindings.overlay has the wrong shape")
        overlay_root = resolve_bound_root(
            self.plan_root, overlay_binding["root"], "bindings.overlay.root"
        )
        expected_overlay_sha = require_sha256(
            overlay_binding["header_sha256"], "bindings.overlay.header_sha256"
        )
        if sha256_file(overlay_root / "header.json") != expected_overlay_sha:
            raise SemanticPlanError("overlay header binding mismatch")
        overlay = load_overlay_header(overlay_root)
        if overlay["overlay_id"] != overlay_binding["overlay_id"]:
            raise SemanticPlanError("overlay_id binding mismatch")
        if overlay["domains"] != self.header["domains"] or overlay["buckets"] != self.header["buckets"]:
            raise SemanticPlanError("overlay domain/bucket table mismatch")
        if overlay["N_select"] != self.header["N_select"]:
            raise SemanticPlanError("overlay N_select mismatch")
        if overlay["excluded_row_ids"] != self.header["excluded_row_ids"]:
            raise SemanticPlanError("overlay exclusions mismatch")
        if overlay["catalog_hash"] != self.header["catalog_hash"]:
            raise SemanticPlanError("overlay catalog_hash mismatch")
        for key in ("catalog_header_sha256", "sampling_config_sha256"):
            if overlay[key] != bindings[key]:
                raise SemanticPlanError(f"overlay {key} differs from semantic binding")
        require_sha256(bindings["catalog_header_sha256"], "bindings.catalog_header_sha256")
        require_sha256(bindings["sampling_config_sha256"], "bindings.sampling_config_sha256")

    def _validate_checksums_document(self) -> None:
        checksums = read_json_object(self.plan_root / "checksums.json")
        if set(checksums) != {"header", "blocks", "quotas"}:
            raise SemanticPlanError("checksums.json must contain exactly header/blocks/quotas")
        expected_header = require_sha256(checksums["header"], "checksums.header")
        actual_header = sha256_file(self.plan_root / "header.json")
        if actual_header != expected_header:
            raise SemanticPlanError(
                f"semantic header checksum mismatch: expected={expected_header}, actual={actual_header}"
            )
        blocks = checksums["blocks"]
        quotas = checksums["quotas"]
        if not isinstance(blocks, Mapping) or not isinstance(quotas, Mapping):
            raise SemanticPlanError("checksums blocks/quotas must be objects")
        first_block = int(self.header["materialization"]["first_materialized_block"])
        expected_blocks = {
            f"block_{i:06d}.npz"
            for i in range(first_block, int(self.header["num_blocks"]))
        }
        if set(blocks) != expected_blocks:
            raise SemanticPlanError("semantic block checksum set mismatch")
        if set(quotas) != set(self.header["quota_files"]):
            raise SemanticPlanError("semantic quota checksum set mismatch")
        for group, subdir in ((blocks, "blocks"), (quotas, "quotas")):
            for relative, checksum in group.items():
                require_sha256(checksum, f"checksums.{subdir}.{relative}")
                if not (self.plan_root / subdir / relative).is_file():
                    raise SemanticPlanError(f"missing semantic {subdir} file: {relative}")
        block_manifest = sha256_bytes(canonical_json_bytes(dict(blocks)))
        quota_manifest = sha256_bytes(canonical_json_bytes(dict(quotas)))
        if block_manifest != self.header["hashes"]["blocks_manifest_sha256"]:
            raise SemanticPlanError("block checksum manifest hash mismatch")
        if quota_manifest != self.header["hashes"]["quota_manifest_sha256"]:
            raise SemanticPlanError("quota checksum manifest hash mismatch")
        for relative, info in self.header["quota_files"].items():
            if not isinstance(info, Mapping) or set(info) != {"bucket", "domain", "rows", "exposures"}:
                raise SemanticPlanError(f"quota_files[{relative!r}] has the wrong shape")
        self.checksums = dict(blocks)
        self.quota_checksums = dict(quotas)

    def _validate_audit_receipt(self) -> None:
        path = self.plan_root / SUCCESS_NAME
        receipt = read_json_object(path)
        if receipt.get("format") != AUDIT_FORMAT or receipt.get("status") != "PASS":
            raise SemanticPlanError(f"{SUCCESS_NAME} is not a PASS receipt")
        bindings = receipt.get("bindings")
        if not isinstance(bindings, Mapping):
            raise SemanticPlanError("audit receipt bindings are missing")
        expected = {
            "plan_hash": self.header["plan_hash"],
            "plan_header_sha256": sha256_file(self.plan_root / "header.json"),
            "plan_checksums_sha256": sha256_file(self.plan_root / "checksums.json"),
        }
        for key, value in expected.items():
            if bindings.get(key) != value:
                raise SemanticPlanError(f"audit receipt {key} binding mismatch")
        assertions = receipt.get("assertions")
        if not isinstance(assertions, Mapping):
            raise SemanticPlanError("audit receipt assertions are missing")
        required_true = (
            "all_block_checksums",
            "global_batch_unique",
            "physical_cell_ranges",
            "overlay_membership",
            "exclusions_zero",
            "repeat_caps",
            "domain_bucket_matrix",
            "native_aspect_ratios",
            "anime_tag_only_zero",
            "exact_full_stage_quota",
            "formal_prefix",
            "formal_prefix_domain_totals",
        )
        for key in required_true:
            if assertions.get(key) is not True:
                raise SemanticPlanError(f"audit receipt assertion {key!r} is not true")
        contract = self.header["audit_contract"]
        formal = receipt.get("formal_prefix")
        full = receipt.get("full_stage")
        multi_ar = receipt.get("multi_ar")
        exclusions = receipt.get("exclusions")
        if not all(isinstance(value, Mapping) for value in (formal, full, multi_ar, exclusions)):
            raise SemanticPlanError("audit receipt result sections are missing")
        receipt_expectations = (
            (formal.get("steps"), contract["formal_prefix_steps"], "formal prefix steps"),
            (
                formal.get("exposures"),
                contract["formal_prefix_exposures"],
                "formal prefix exposures",
            ),
            (
                formal.get("domain_totals"),
                contract["formal_prefix_domain_totals"],
                "formal prefix domain totals",
            ),
            (full.get("steps"), self.header["total_steps"], "full-stage steps"),
            (
                full.get("exposures"),
                self.header["total_exposures"],
                "full-stage exposures",
            ),
            (
                multi_ar.get("active_native_bucket_count"),
                contract["active_native_buckets"],
                "active native buckets",
            ),
            (
                multi_ar.get("active_native_bucket_indices"),
                contract["active_native_bucket_indices"],
                "active native bucket indices",
            ),
            (
                exclusions.get("expected_count"),
                contract["excluded_rows"],
                "excluded row count",
            ),
            (exclusions.get("observed_exposures"), 0, "excluded row exposures"),
            (full.get("anime_tag_only_exposures"), 0, "anime tag-only exposures"),
        )
        for actual, expected_value, label in receipt_expectations:
            if actual != expected_value:
                raise SemanticPlanError(
                    f"audit receipt {label} mismatch: stored={actual!r}, expected={expected_value!r}"
                )

    def _block_path(self, block_index: int) -> tuple[str, Path]:
        filename = f"block_{block_index:06d}.npz"
        return filename, self.plan_root / "blocks" / filename

    def _verify_block_checksum(self, block_index: int) -> None:
        if block_index in self._verified_blocks:
            return
        relative, path = self._block_path(block_index)
        expected = self.checksums[relative]
        actual = sha256_file(path)
        if actual != expected:
            raise SemanticPlanError(
                f"semantic block checksum mismatch for {relative}: expected={expected}, actual={actual}"
            )
        self._verified_blocks.add(block_index)

    def verify_all_checksums(self, *, include_quotas: bool = False) -> None:
        for block_index in range(self.first_materialized_block, int(self.header["num_blocks"])):
            self._verify_block_checksum(block_index)
        if include_quotas:
            for relative, expected in self.quota_checksums.items():
                actual = sha256_file(self.plan_root / "quotas" / relative)
                if actual != expected:
                    raise SemanticPlanError(
                        f"quota checksum mismatch for {relative}: expected={expected}, actual={actual}"
                    )

    def _load_block(self, block_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._cached_block_index == block_index:
            assert self._cached_bucket is not None
            assert self._cached_items is not None
            assert self._cached_domain_counts is not None
            return self._cached_bucket, self._cached_items, self._cached_domain_counts
        if block_index < self.first_materialized_block or block_index >= int(self.header["num_blocks"]):
            raise IndexError(f"block index outside materialized semantic plan: {block_index}")
        if self.verify_checksums:
            self._verify_block_checksum(block_index)
        relative, path = self._block_path(block_index)
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(BLOCK_DATASETS):
                    raise SemanticPlanError(f"unexpected datasets in {relative}: {archive.files}")
                bucket = np.array(archive["bucket_of_step"], copy=True)
                items = np.array(archive["item_row"], copy=True)
                counts = np.array(archive["domain_counts"], copy=True)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise SemanticPlanError(f"cannot load semantic block {path}: {exc}") from exc
        start = block_index * int(self.header["block_steps"])
        expected_steps = min(
            int(self.header["block_steps"]), int(self.header["total_steps"]) - start
        )
        expected_item_shape = (
            expected_steps,
            int(self.header["world_size"]),
            int(self.header["per_rank_bs"]),
        )
        if bucket.dtype != np.dtype(np.int16) or bucket.shape != (expected_steps,):
            raise SemanticPlanError(f"invalid bucket_of_step in {relative}")
        if items.dtype != np.dtype(np.int64) or items.shape != expected_item_shape:
            raise SemanticPlanError(f"invalid item_row in {relative}: {items.dtype}/{items.shape}")
        expected_count_shape = (expected_steps, len(self.header["domains"]))
        if counts.dtype != np.dtype(np.int32) or counts.shape != expected_count_shape:
            raise SemanticPlanError(f"invalid domain_counts in {relative}")
        selection_bucket, _selection_items, selection_counts = self.selection_plan._load_block(  # noqa: SLF001
            block_index
        )
        if not np.array_equal(bucket, selection_bucket) or not np.array_equal(counts, selection_counts):
            raise SemanticPlanError(
                f"semantic schedule differs from selection plan in block {block_index}"
            )
        n_physical = np.asarray(self.header["N"], dtype=np.int64)
        offsets = np.asarray(self.header["cell_row_offset"], dtype=np.int64)
        exclusions = np.asarray(self.header["excluded_row_ids"], dtype=np.int64)
        G = int(self.header["G"])
        for local_step in range(expected_steps):
            b = int(bucket[local_step])
            flat = items[local_step].T.reshape(-1)
            if flat.shape != (G,) or np.unique(flat).size != G:
                raise SemanticPlanError(
                    f"duplicate semantic item in block={block_index},step={local_step}"
                )
            if exclusions.size and np.any(np.isin(flat, exclusions)):
                raise SemanticPlanError(
                    f"excluded row present in block={block_index},step={local_step}"
                )
            boundaries = np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(counts[local_step], dtype=np.int64))
            )
            for d in range(len(self.header["domains"])):
                lo, hi = int(boundaries[d]), int(boundaries[d + 1])
                values = flat[lo:hi]
                row_lo = int(offsets[b, d])
                row_hi = row_lo + int(n_physical[b, d])
                if np.any(values < row_lo) or np.any(values >= row_hi):
                    raise SemanticPlanError(
                        f"physical cell range violation in block={block_index},step={local_step},domain={d}"
                    )
        bucket.setflags(write=False)
        items.setflags(write=False)
        counts.setflags(write=False)
        self._cached_block_index = block_index
        self._cached_bucket = bucket
        self._cached_items = items
        self._cached_domain_counts = counts
        return bucket, items, counts

    def batch(self, step: int, rank: int) -> tuple[tuple[int, int], np.ndarray]:
        step_i = require_int(step, "step", minimum=0)
        rank_i = require_int(rank, "rank", minimum=0)
        if step_i >= int(self.header["total_steps"]):
            raise IndexError(f"step {step_i} outside semantic plan")
        if rank_i >= int(self.header["world_size"]):
            raise IndexError(f"rank {rank_i} outside semantic plan")
        block_index, offset = divmod(step_i, int(self.header["block_steps"]))
        bucket, items, _counts = self._load_block(block_index)
        b = int(bucket[offset])
        return tuple(int(v) for v in self.header["buckets"][b]), items[offset, rank_i]

    def first_step_after(self, completed_optimizer_step: int) -> int | None:
        completed = require_int(completed_optimizer_step, "completed_optimizer_step")
        total = int(self.header["total_steps"])
        if completed < -1 or completed >= total:
            raise SemanticPlanError(
                f"completed_optimizer_step must be in [-1,{total - 1}], got {completed}"
            )
        next_step = completed + 1
        return None if next_step == total else next_step

    def seek_after_completed(self, completed_optimizer_step: int) -> int | None:
        next_step = self.first_step_after(completed_optimizer_step)
        self._cached_block_index = None
        self._cached_bucket = None
        self._cached_items = None
        self._cached_domain_counts = None
        if next_step is not None:
            self._load_block(next_step // int(self.header["block_steps"]))
        return next_step


__all__ = [
    "ALGORITHM_VERSION",
    "AUDIT_FORMAT",
    "BLOCK_DATASETS",
    "BLOCK_FORMAT",
    "FORMAL_W64_ACTIVE_BUCKET_INDICES",
    "FORMAL_W64_PREFIX_DOMAIN_TOTALS",
    "OVERLAY_FORMAT",
    "OVERLAY_FORMAT_VERSION",
    "PLAN_FORMAT",
    "PLAN_FORMAT_VERSION",
    "QUOTA_ALGORITHM",
    "SUCCESS_NAME",
    "SemanticOfflinePlan",
    "SemanticPlanError",
    "TICKET_ALGORITHM",
    "binding_for_plan",
    "canonical_json_bytes",
    "cell_key",
    "load_overlay_cell",
    "load_overlay_header",
    "overlay_cell_key",
    "physical_cell_offsets",
    "read_json_object",
    "require_int",
    "require_sha256",
    "sha256_bytes",
    "sha256_file",
    "write_deterministic_npz_create_once",
    "write_json_create_once",
]
