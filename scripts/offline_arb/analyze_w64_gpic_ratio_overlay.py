#!/usr/bin/env python3
"""Exact plan-only analysis of deterministic GPIC ratio overlays.

For each ``k`` in ``8,10,12,13,14,15,16,17``, a GPIC catalog-local row is
selected iff ``local_row_id % 20 < k``.  The analyzer streams the formal W64
prefix once, measures exact selected exposures and per-node unique objects,
and evaluates absolute cold-source demand in full 4/16/64-step windows whose
starts are anchored every four steps.

Capacity is always normalized to the fixed nominal GPIC demand from the frozen
formal plan.  Coverage percentiles are reported separately and never used as a
substitute for absolute burst demand.  The analyzer reads no image and never
opens the GPIC catalog parquet payload; the parquet is only stat-bound through
the freeze/catalog headers.  This is diagnostic evidence, not a promotion
authority and not a reusable overlay artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from training.dataline_offline.planner_v1 import OfflinePlan, PlannerInputError


SCHEMA_VERSION = "offline_arb_w64_gpic_ratio_overlay_v1"
FORMAL_START_STEP = 0
FORMAL_STOP_STEP = 166_667
EXPECTED_WORLD_SIZE = 64
EXPECTED_PER_RANK_BS = 36
EXPECTED_GLOBAL_BATCH = 2_304
EXPECTED_FORMAL_EXPOSURES = 384_000_768
EXPECTED_FORMAL_GPIC_EXPOSURES = 99_456_552
EXPECTED_PLAN_BLOCKS = 168
EXPECTED_BLOCK_STEPS = 1_024
EXPECTED_GPIC_ROWS = 20_000_000
EXPECTED_GPIC_R_MAX = 20
EXPECTED_FREEZE_SCHEMA = "offline_arb_freeze_v3_caption_sidecar"
EXPECTED_CATALOG_SCHEMA = "offline_arb_catalog_v1"
EXPECTED_PLAN_ID = "stage256_offline_exact_v1_w64_prod"
TARGET_DOMAIN = "gpic"
RANKS_PER_NODE = 8
NODE_COUNT = 8
MODULUS = 20
SELECTION_K = tuple(range(8, 18))
WINDOW_STRIDE_STEPS = 4
WINDOW_HORIZONS = (4, 16, 64)
UPPER_TAIL_POINTS = (("p99", 9_900), ("p99.9", 9_990), ("max", 10_000))
LOWER_COVERAGE_POINTS = (("min", 0), ("p0.1", 10), ("p1", 100))
BASELINE_CAPACITY_NUMERATOR = 787
BASELINE_CAPACITY_DENOMINATOR = 1_000
CAPACITY_GATES = (("minimum", 1_111, 1_000), ("target", 5, 4))
DEFAULT_STORAGE_BYTES = (98_304, 114_688, 131_072, 144_296, 163_840, 196_608)
CENTRAL_OBJECT_BYTES = 144_296
LOCAL_CAPACITY_GIB = (250, 500)


class EvidenceError(RuntimeError):
    """A frozen-input, accounting, or publication contract was violated."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=1024 * 1024) as handle:
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be a JSON object")
    return value


def _require_canonical_file(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise EvidenceError(f"{label} must be an absolute regular non-symlink: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise EvidenceError(f"{label} is not canonical: {path} -> {resolved}")
    return resolved


def _require_canonical_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise EvidenceError(f"{label} must be an absolute directory non-symlink: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise EvidenceError(f"{label} is not canonical: {path} -> {resolved}")
    return resolved


def _fingerprint(path: Path, *, directory: bool = False) -> dict[str, int]:
    value = path.stat(follow_symlinks=False)
    valid = stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)
    if not valid:
        raise EvidenceError(f"unexpected input type: {path}")
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _require_record(
    record: Any,
    path: Path,
    label: str,
    *,
    sha256: str | None = None,
) -> dict[str, int]:
    if not isinstance(record, Mapping):
        raise EvidenceError(f"freeze {label} must be an object")
    if record.get("path") != str(path):
        raise EvidenceError(f"freeze {label} path mismatch")
    live = _fingerprint(path)
    if record.get("fingerprint") != live:
        raise EvidenceError(f"freeze {label} five-field fingerprint mismatch")
    if int(record.get("bytes", -1)) != live["size_bytes"]:
        raise EvidenceError(f"freeze {label} byte count mismatch")
    if sha256 is not None and record.get("sha256") != sha256:
        raise EvidenceError(f"freeze {label} SHA mismatch")
    return live


def _fraction(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise EvidenceError(f"invalid fraction {numerator}/{denominator}")
    divisor = math.gcd(numerator, denominator)
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "reduced_numerator": int(numerator // divisor),
        "reduced_denominator": int(denominator // divisor),
        "decimal": format(numerator / denominator, ".12f"),
    }


def _positive_ratio(numerator: int, denominator: int) -> dict[str, Any]:
    if numerator < 0 or denominator <= 0:
        raise EvidenceError(f"invalid ratio {numerator}/{denominator}")
    divisor = math.gcd(numerator, denominator)
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "reduced_numerator": int(numerator // divisor),
        "reduced_denominator": int(denominator // divisor),
        "decimal": format(numerator / denominator, ".12f"),
    }


def _parse_positive_ints(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(token.strip()) for token in raw.split(",") if token.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def _storage_estimates(unique_objects: int, object_bytes_values: Sequence[int]) -> list[dict[str, Any]]:
    if unique_objects < 0:
        raise EvidenceError("unique object count must be nonnegative")
    rows = []
    for object_bytes in object_bytes_values:
        total = int(unique_objects) * int(object_bytes)
        rows.append(
            {
                "assumed_bytes_per_object": int(object_bytes),
                "is_central_observed_mean": int(object_bytes) == CENTRAL_OBJECT_BYTES,
                "estimated_bytes": total,
                "estimated_decimal_gb": format(total / 1_000_000_000, ".6f"),
                "estimated_gib": format(total / (1 << 30), ".6f"),
                "estimated_decimal_tb": format(total / 1_000_000_000_000, ".6f"),
                "fits_capacity_gib": {
                    str(gib): total <= gib * (1 << 30) for gib in LOCAL_CAPACITY_GIB
                },
            }
        )
    return rows


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, mode)
    try:
        view = memoryview(payload)
        while view:
            wrote = os.write(fd, view)
            if wrote <= 0:
                raise EvidenceError(f"short write: {path}")
            view = view[wrote:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_publish_json(output: Path, value: Mapping[str, Any]) -> str:
    payload = _canonical_bytes(value)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evidence: {output}")
    parent = _require_canonical_directory(output.parent, "output parent")
    temporary = parent / f".{output.name}.tmp.{os.getpid()}.{os.urandom(16).hex()}"
    try:
        _write_exclusive(temporary, payload)
        os.link(temporary, output, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if output.is_symlink() or not output.is_file() or output.read_bytes() != payload:
        raise EvidenceError("published evidence failed byte-exact readback")
    return _sha256_bytes(payload)


def _validate_job_yaml(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    result: dict[str, int] = {}
    for key in ("numWorkers", "gpuPerWorker", "cpuPerWorker"):
        matches = re.findall(rf"(?m)^{re.escape(key)}:\s*([0-9]+)\s*$", text)
        if len(matches) != 1:
            raise EvidenceError(f"job YAML must declare {key} exactly once")
        result[key] = int(matches[0])
    if result != {"numWorkers": 1, "gpuPerWorker": 0, "cpuPerWorker": 16}:
        raise EvidenceError(f"job YAML topology drift: {result}")
    return result


def _assignment_identity_binding(
    *,
    freeze_identities: Mapping[str, Any],
    plan_header: Mapping[str, Any],
    catalog_header: Mapping[str, Any],
) -> dict[str, Any]:
    plan_hashes = plan_header.get("hashes")
    if not isinstance(plan_hashes, Mapping):
        raise EvidenceError("plan header hashes object is missing")
    values = {
        "freeze_identities_assignment_hash": freeze_identities.get("assignment_hash"),
        "plan_header_hashes_assignment_sha256": plan_hashes.get("assignment_sha256"),
        "plan_header_top_level_assignment_hash": plan_header.get("assignment_hash"),
        "catalog_header_assignment_hash": catalog_header.get("assignment_hash"),
    }
    for label, value in values.items():
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
        ):
            raise EvidenceError(f"{label} must be a lowercase SHA-256 identity")
    if len(set(values.values())) != 1:
        raise EvidenceError(f"assignment identity mismatch: {values}")
    assignment_hash = next(iter(values.values()))
    return {
        **values,
        "assignment_hash": assignment_hash,
        "all_four_assignment_identities_equal": True,
        "comparison": "exact_lowercase_sha256_string_equality",
    }


def _bind_frozen_inputs(
    *,
    freeze_path: Path,
    freeze: Mapping[str, Any],
    plan_root: Path,
    header_path: Path,
    checksums_path: Path,
    header: Mapping[str, Any],
    checksums: Mapping[str, Any],
    catalog_header_path: Path,
    catalog_header: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, dict[str, int]], Path, dict[str, int]]:
    if freeze.get("schema_version") != EXPECTED_FREEZE_SCHEMA:
        raise EvidenceError(f"freeze schema must be {EXPECTED_FREEZE_SCHEMA}")
    identities = freeze.get("identities")
    inputs = freeze.get("inputs")
    plan_blocks = freeze.get("plan_blocks")
    catalog = freeze.get("catalog")
    if not all(isinstance(value, Mapping) for value in (identities, inputs, plan_blocks, catalog)):
        raise EvidenceError("freeze is missing identities/inputs/plan_blocks/catalog objects")
    assert isinstance(identities, Mapping) and isinstance(inputs, Mapping)
    assert isinstance(plan_blocks, Mapping) and isinstance(catalog, Mapping)
    if identities.get("plan_id") != EXPECTED_PLAN_ID:
        raise EvidenceError("freeze plan identity drift")
    if identities.get("catalog_hash") != header.get("hashes", {}).get("catalog_sha256"):
        raise EvidenceError("freeze/plan catalog identity mismatch")
    _require_record(
        inputs.get("plan_header"), header_path, "inputs.plan_header", sha256=hashes["plan_header"]
    )
    _require_record(
        inputs.get("plan_checksums"),
        checksums_path,
        "inputs.plan_checksums",
        sha256=hashes["plan_checksums"],
    )
    _require_record(
        inputs.get("catalog_header"),
        catalog_header_path,
        "inputs.catalog_header",
        sha256=hashes["catalog_header"],
    )

    blocks_root = plan_root / "blocks"
    if set(plan_root.iterdir()) != {header_path, checksums_path, blocks_root}:
        raise EvidenceError("plan root entry set drift")
    blocks_root = _require_canonical_directory(blocks_root, "plan blocks")
    directories = plan_blocks.get("directories")
    if not isinstance(directories, Mapping) or set(directories) != {"root", "blocks"}:
        raise EvidenceError("freeze plan directory set drift")
    for name, path in (("root", plan_root), ("blocks", blocks_root)):
        record = directories[name]
        if not isinstance(record, Mapping) or record.get("path") != str(path):
            raise EvidenceError(f"freeze plan directory record mismatch: {name}")
        if record.get("fingerprint") != _fingerprint(path, directory=True):
            raise EvidenceError(f"freeze plan directory fingerprint mismatch: {name}")

    frozen_files = plan_blocks.get("files")
    checksum_blocks = checksums.get("blocks")
    if not isinstance(frozen_files, Mapping) or not isinstance(checksum_blocks, Mapping):
        raise EvidenceError("freeze/checksums block maps are missing")
    if set(frozen_files) != set(checksum_blocks) or len(frozen_files) != EXPECTED_PLAN_BLOCKS:
        raise EvidenceError("freeze/checksums block set is not exactly 168 files")
    if {entry.name for entry in blocks_root.iterdir()} != set(checksum_blocks):
        raise EvidenceError("live block entry set drift")
    snapshots: dict[str, dict[str, int]] = {}
    manifest: list[dict[str, Any]] = []
    for index, name in enumerate(sorted(checksum_blocks)):
        if name != f"block_{index:06d}.npz":
            raise EvidenceError(f"non-contiguous block name: {name}")
        path = _require_canonical_file(blocks_root / name, f"plan block {name}")
        expected_sha = checksum_blocks[name]
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise EvidenceError(f"invalid block SHA record: {name}")
        live = _require_record(
            frozen_files[name], path, f"plan_blocks.files.{name}", sha256=expected_sha
        )
        snapshots[name] = live
        manifest.append(
            {
                "name": name,
                "bytes": live["size_bytes"],
                "sha256": expected_sha,
                "fingerprint": live,
            }
        )

    catalog_directory = catalog.get("directory")
    catalog_files = catalog.get("files")
    if not isinstance(catalog_directory, Mapping) or not isinstance(catalog_files, Mapping):
        raise EvidenceError("freeze catalog directory/files objects are missing")
    catalog_root = _require_canonical_directory(catalog_header_path.parent, "catalog root")
    if catalog_directory.get("path") != str(catalog_root):
        raise EvidenceError("freeze catalog root path mismatch")
    if catalog_directory.get("fingerprint") != _fingerprint(catalog_root, directory=True):
        raise EvidenceError("freeze catalog root fingerprint mismatch")
    gpic_record = catalog_files.get(TARGET_DOMAIN)
    if not isinstance(gpic_record, Mapping):
        raise EvidenceError("freeze GPIC catalog file record is missing")
    gpic_path = _require_canonical_file(Path(str(gpic_record.get("path"))), "GPIC catalog parquet")
    # Deliberately stat-only: the trusted SHA is already frozen, and this
    # analyzer must not read the catalog parquet payload.
    gpic_live = _require_record(
        gpic_record,
        gpic_path,
        "catalog.files.gpic",
        sha256=str(gpic_record.get("sha256")),
    )
    header_gpic = catalog_header.get("files", {}).get(TARGET_DOMAIN)
    if not isinstance(header_gpic, Mapping):
        raise EvidenceError("catalog header GPIC record is missing")
    for key in ("name", "rows", "bytes", "sha256", "schema"):
        if header_gpic.get(key) != gpic_record.get(key):
            raise EvidenceError(f"catalog header/freeze GPIC {key} mismatch")
    if catalog_header.get("schema_version") != EXPECTED_CATALOG_SCHEMA:
        raise EvidenceError("catalog header schema drift")
    if catalog_header.get("catalog_hash") != identities.get("catalog_hash"):
        raise EvidenceError("catalog header/freeze catalog hash mismatch")
    assignment_binding = _assignment_identity_binding(
        freeze_identities=identities,
        plan_header=header,
        catalog_header=catalog_header,
    )

    return (
        {
            "freeze": {
                "path": str(freeze_path),
                "sha256": hashes["freeze"],
                "schema_version": freeze.get("schema_version"),
            },
            "plan_block_count": len(manifest),
            "plan_blocks_manifest_digest_contract": (
                "canonical_json_sorted_list(name,bytes,sha256,fingerprint)"
            ),
            "plan_blocks_manifest_sha256": _sha256_bytes(_canonical_bytes(manifest)),
            "assignment_identity_binding": assignment_binding,
            "catalog": {
                "header_path": str(catalog_header_path),
                "header_sha256": hashes["catalog_header"],
                "catalog_hash": catalog_header.get("catalog_hash"),
                "gpic_parquet": {
                    "path": str(gpic_path),
                    "bytes": gpic_live["size_bytes"],
                    "sha256_from_frozen_catalog": gpic_record.get("sha256"),
                    "fingerprint": gpic_live,
                    "payload_read_by_analyzer": False,
                    "stat_only": True,
                },
            },
        },
        snapshots,
        gpic_path,
        gpic_live,
    )


def _clear_plan_cache(plan: OfflinePlan) -> None:
    plan._cached_block_index = None  # noqa: SLF001
    plan._cached_bucket = None  # noqa: SLF001
    plan._cached_items = None  # noqa: SLF001
    plan._cached_domain_counts = None  # noqa: SLF001


def _capacity_record(
    hot: int,
    total: int,
    *,
    nominal_demand_numerator: int,
    nominal_demand_denominator: int,
) -> dict[str, Any]:
    if total <= 0 or hot < 0 or hot > total:
        raise EvidenceError("invalid hot/total capacity inputs")
    if nominal_demand_numerator <= 0 or nominal_demand_denominator <= 0:
        raise EvidenceError("nominal demand must be a positive rational")
    cold = total - hot
    if cold == 0:
        adjusted: dict[str, Any] = {
            "numerator": None,
            "denominator": None,
            "reduced_numerator": None,
            "reduced_denominator": None,
            "decimal": "infinity",
        }
    else:
        numerator = BASELINE_CAPACITY_NUMERATOR * nominal_demand_numerator
        denominator = BASELINE_CAPACITY_DENOMINATOR * nominal_demand_denominator * cold
        divisor = math.gcd(numerator, denominator)
        adjusted = {
            "numerator": int(numerator),
            "denominator": int(denominator),
            "reduced_numerator": int(numerator // divisor),
            "reduced_denominator": int(denominator // divisor),
            "decimal": format(numerator / denominator, ".12f"),
        }
    gates = {}
    for name, gate_num, gate_den in CAPACITY_GATES:
        lhs = BASELINE_CAPACITY_NUMERATOR * nominal_demand_numerator * gate_den
        rhs = (
            BASELINE_CAPACITY_DENOMINATOR
            * nominal_demand_denominator
            * cold
            * gate_num
        )
        gates[name] = {
            "threshold": format(gate_num / gate_den, ".3f"),
            "pass": bool(cold == 0 or lhs >= rhs),
            "comparison": "exact_integer_cross_multiplication",
            "lhs": int(lhs),
            "rhs": int(rhs),
        }
    return {
        "total_gpic_exposures": int(total),
        "selected_overlay_exposures": int(hot),
        "absolute_cold_source_exposures": int(cold),
        "selected_coverage": _fraction(hot, total),
        "cold_source_fraction": _fraction(cold, total),
        "fixed_nominal_gpic_demand_for_interval": _positive_ratio(
            nominal_demand_numerator, nominal_demand_denominator
        ),
        "observed_total_over_fixed_nominal": _positive_ratio(
            total * nominal_demand_denominator, nominal_demand_numerator
        ),
        "observed_cold_over_fixed_nominal": _positive_ratio(
            cold * nominal_demand_denominator, nominal_demand_numerator
        ),
        "baseline_source_capacity_ratio": "0.787",
        "adjusted_source_capacity_ratio_baseline_times_fixed_nominal_divided_by_absolute_cold": adjusted,
        "capacity_gates": gates,
    }


def _location_digest(
    *,
    k: int,
    horizon: int,
    node: int,
    start: int,
    metric: str,
    point: str,
) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "plan_id": EXPECTED_PLAN_ID,
                "selection": f"gpic_local_row_id_mod_20_lt_{k}",
                "horizon_steps": horizon,
                "node": node,
                "start_step_inclusive": start,
                "end_step_exclusive": start + horizon,
                "metric": metric,
                "point": point,
            }
        )
    )


def _anchored_window_sums(
    values: np.ndarray,
    horizon: int,
    *,
    stride: int = WINDOW_STRIDE_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise EvidenceError("window source must be a one-dimensional integer vector")
    if horizon <= 0 or stride <= 0 or values.size < horizon:
        raise EvidenceError("invalid horizon/stride/source length")
    starts = np.arange(0, values.size - horizon + 1, stride, dtype=np.int64)
    prefix = np.empty(values.size + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(values, dtype=np.int64, out=prefix[1:])
    return starts, prefix[starts + horizon] - prefix[starts]


def _tie_first_index(
    values: np.ndarray,
    target: int,
    nodes: np.ndarray,
    starts: np.ndarray,
) -> int:
    candidates = np.flatnonzero(values == target)
    if candidates.size == 0:
        raise EvidenceError("quantile target has no source row")
    order = np.lexsort((starts[candidates], nodes[candidates]))
    return int(candidates[int(order[0])])


def _context_record(
    *,
    index: int,
    totals: np.ndarray,
    hot: np.ndarray,
    nodes: np.ndarray,
    starts: np.ndarray,
    k: int,
    horizon: int,
    metric: str,
    point: str,
) -> dict[str, Any]:
    total = int(totals[index])
    selected = int(hot[index])
    node = int(nodes[index])
    start = int(starts[index])
    record = _capacity_record(
        selected,
        total,
        nominal_demand_numerator=EXPECTED_FORMAL_GPIC_EXPOSURES * horizon,
        nominal_demand_denominator=FORMAL_STOP_STEP * NODE_COUNT,
    )
    record.update(
        {
            "node": node,
            "start_step_inclusive": start,
            "end_step_exclusive": start + horizon,
            "location_digest_sha256": _location_digest(
                k=k,
                horizon=horizon,
                node=node,
                start=start,
                metric=metric,
                point=point,
            ),
        }
    )
    return record


def _upper_tail_distribution(
    *,
    values: np.ndarray,
    totals: np.ndarray,
    hot: np.ndarray,
    nodes: np.ndarray,
    starts: np.ndarray,
    k: int,
    horizon: int,
    metric: str,
) -> dict[str, Any]:
    arrays = (values, totals, hot, nodes, starts)
    if any(value.ndim != 1 for value in arrays) or len({value.size for value in arrays}) != 1:
        raise EvidenceError("upper-tail arrays have inconsistent shapes")
    if values.size == 0 or np.any(values < 0):
        raise EvidenceError("upper-tail population must be nonempty and nonnegative")
    ordered_values = np.sort(values.astype(np.int64, copy=False), kind="stable")
    population = int(values.size)
    points = {}
    for name, basis_points in UPPER_TAIL_POINTS:
        if basis_points == 10_000:
            target = int(ordered_values[-1])
            nearest_rank = population - 1
        else:
            nearest_rank = max(0, math.ceil(basis_points * population / 10_000) - 1)
            target = int(ordered_values[nearest_rank])
        index = _tie_first_index(values, target, nodes, starts)
        record = _context_record(
            index=index,
            totals=totals,
            hot=hot,
            nodes=nodes,
            starts=starts,
            k=k,
            horizon=horizon,
            metric=metric,
            point=name,
        )
        record.update(
            {
                "metric_value_exposures": target,
                "nearest_rank_zero_based": nearest_rank,
                "nearest_rank_basis_points": basis_points,
                "tie_break": "lowest_node_then_earliest_start_for_equal_metric_value",
            }
        )
        points[name] = record
    return {
        "metric": metric,
        "population": population,
        "quantile_method": "nearest_rank_ceil(q*n)-1_on_absolute_integer_demand",
        "points": points,
    }


def _lower_coverage_distribution(
    *,
    totals: np.ndarray,
    hot: np.ndarray,
    nodes: np.ndarray,
    starts: np.ndarray,
    k: int,
    horizon: int,
) -> dict[str, Any]:
    valid = totals > 0
    if not np.any(valid):
        raise EvidenceError("coverage population has no positive-demand windows")
    vt = totals[valid].astype(np.int64, copy=False)
    vh = hot[valid].astype(np.int64, copy=False)
    vn = nodes[valid].astype(np.int64, copy=False)
    vs = starts[valid].astype(np.int64, copy=False)
    ratios = vh.astype(np.float64) / vt.astype(np.float64)
    order = np.lexsort((vs, vn, ratios))
    population = int(order.size)
    points = {}
    for name, basis_points in LOWER_COVERAGE_POINTS:
        rank = 0 if basis_points == 0 else max(
            0, math.ceil(basis_points * population / 10_000) - 1
        )
        index = int(order[rank])
        record = _context_record(
            index=index,
            totals=vt,
            hot=vh,
            nodes=vn,
            starts=vs,
            k=k,
            horizon=horizon,
            metric="selected_coverage",
            point=name,
        )
        record["nearest_rank_zero_based"] = rank
        points[name] = record
    return {
        "metric": "selected_coverage",
        "population_positive_total_demand": population,
        "zero_total_demand_windows_excluded": int((~valid).sum()),
        "quantile_method": (
            "nearest_rank_lower_tail_on_float64_hot_div_total_then_node_then_start; "
            "denominators<=18432 make distinct-rational ordering safe"
        ),
        "points": points,
    }


def _window_distribution(
    *,
    totals: np.ndarray,
    hot: np.ndarray,
    nodes: np.ndarray,
    starts: np.ndarray,
    k: int,
    horizon: int,
) -> dict[str, Any]:
    arrays = (totals, hot, nodes, starts)
    if any(value.ndim != 1 for value in arrays) or len({value.size for value in arrays}) != 1:
        raise EvidenceError("window arrays have inconsistent shapes")
    if totals.size == 0 or np.any(totals < 0) or np.any(hot < 0) or np.any(hot > totals):
        raise EvidenceError("window total/hot values are invalid")
    maximum = horizon * RANKS_PER_NODE * EXPECTED_PER_RANK_BS
    if np.any(totals > maximum):
        raise EvidenceError("window GPIC demand exceeds node batch-size upper bound")
    cold = totals - hot
    total_tail = _upper_tail_distribution(
        values=totals,
        totals=totals,
        hot=hot,
        nodes=nodes,
        starts=starts,
        k=k,
        horizon=horizon,
        metric="absolute_total_gpic_demand",
    )
    cold_tail = _upper_tail_distribution(
        values=cold,
        totals=totals,
        hot=hot,
        nodes=nodes,
        starts=starts,
        k=k,
        horizon=horizon,
        metric="absolute_cold_source_demand",
    )
    coverage = _lower_coverage_distribution(
        totals=totals,
        hot=hot,
        nodes=nodes,
        starts=starts,
        k=k,
        horizon=horizon,
    )
    return {
        "horizon_steps": horizon,
        "anchor_stride_steps": WINDOW_STRIDE_STEPS,
        "full_windows_only": True,
        "window_population": int(totals.size),
        "capacity_model": (
            "0.787 * fixed nominal GPIC demand for horizon / absolute cold demand; "
            "coverage and total burst are separate context metrics"
        ),
        "absolute_total_gpic_demand": total_tail,
        "absolute_cold_source_demand": cold_tail,
        "selected_coverage_lower_tail_context_only": coverage,
        "coverage_min_location_equals_absolute_cold_max_location": bool(
            coverage["points"]["min"]["node"]
            == cold_tail["points"]["max"]["node"]
            and coverage["points"]["min"]["start_step_inclusive"]
            == cold_tail["points"]["max"]["start_step_inclusive"]
        ),
    }


def _selection_catalog_unique(k: int, row_count: int = EXPECTED_GPIC_ROWS) -> int:
    if k <= 0 or k >= MODULUS or row_count < 0:
        raise EvidenceError("invalid ratio-overlay selection parameters")
    cycles, tail = divmod(row_count, MODULUS)
    return cycles * k + min(k, tail)


def _residue_unique_counts(seen: np.ndarray) -> np.ndarray:
    if seen.dtype != np.bool_ or seen.ndim != 1 or seen.size != EXPECTED_GPIC_ROWS:
        raise EvidenceError("seen bitmap shape/type drift")
    return np.asarray(
        [np.count_nonzero(seen[residue::MODULUS]) for residue in range(MODULUS)],
        dtype=np.int64,
    )


def _self_test(work_root: Path) -> None:
    work_root.mkdir(parents=True, exist_ok=True)
    root = _require_canonical_directory(work_root.resolve(strict=True), "self-test work root")
    if not str(root).startswith("/local/"):
        raise EvidenceError("self-test work root must be under /local")
    work = Path(tempfile.mkdtemp(prefix="w64_gpic_ratio_overlay_selftest.", dir=root))
    try:
        if list(SELECTION_K) != list(range(8, 18)):
            raise EvidenceError("evaluated k set must be every integer from 8 through 17")
        if [_selection_catalog_unique(k, 40) for k in SELECTION_K] != [2 * k for k in SELECTION_K]:
            raise EvidenceError("modulo-selection cardinality self-test failed")

        digest = "a" * 64
        assignment = _assignment_identity_binding(
            freeze_identities={"assignment_hash": digest},
            plan_header={
                "assignment_hash": digest,
                "hashes": {"assignment_sha256": digest},
            },
            catalog_header={"assignment_hash": digest},
        )
        if not assignment["all_four_assignment_identities_equal"]:
            raise EvidenceError("positive assignment-binding self-test failed")
        try:
            _assignment_identity_binding(
                freeze_identities={"assignment_hash": digest},
                plan_header={
                    "assignment_hash": digest,
                    "hashes": {"assignment_sha256": "b" * 64},
                },
                catalog_header={"assignment_hash": digest},
            )
        except EvidenceError:
            pass
        else:
            raise EvidenceError("assignment-mismatch negative self-test did not fail closed")

        values = np.arange(1, 9, dtype=np.int64)
        starts, sums = _anchored_window_sums(values, 4, stride=4)
        if starts.tolist() != [0, 4] or sums.tolist() != [10, 26]:
            raise EvidenceError("anchored-window self-test failed")

        # Window 0 has the worst coverage (0/10), while window 1 has both the
        # largest total burst (100) and largest absolute cold demand (20).
        # A coverage-only model would choose the wrong capacity-worst window.
        totals = np.asarray([10, 100], dtype=np.int64)
        hot = np.asarray([0, 80], dtype=np.int64)
        nodes = np.asarray([0, 0], dtype=np.int64)
        tiny_starts = np.asarray([0, 4], dtype=np.int64)
        distribution = _window_distribution(
            totals=totals,
            hot=hot,
            nodes=nodes,
            starts=tiny_starts,
            k=10,
            horizon=4,
        )
        if (
            distribution["selected_coverage_lower_tail_context_only"]["points"]["min"][
                "start_step_inclusive"
            ]
            != 0
        ):
            raise EvidenceError("coverage-worst self-test location drift")
        if (
            distribution["absolute_total_gpic_demand"]["points"]["max"][
                "start_step_inclusive"
            ]
            != 4
            or distribution["absolute_cold_source_demand"]["points"]["max"][
                "start_step_inclusive"
            ]
            != 4
        ):
            raise EvidenceError("absolute-burst self-test location drift")

        exact_target = _capacity_record(
            1_852,
            5_000,
            nominal_demand_numerator=5_000,
            nominal_demand_denominator=1,
        )
        below_target = _capacity_record(
            1_851,
            5_000,
            nominal_demand_numerator=5_000,
            nominal_demand_denominator=1,
        )
        if (
            exact_target["adjusted_source_capacity_ratio_baseline_times_fixed_nominal_divided_by_absolute_cold"][
                "reduced_numerator"
            ]
            != 5
            or exact_target[
                "adjusted_source_capacity_ratio_baseline_times_fixed_nominal_divided_by_absolute_cold"
            ]["reduced_denominator"]
            != 4
            or not exact_target["capacity_gates"]["target"]["pass"]
            or below_target["capacity_gates"]["target"]["pass"]
        ):
            raise EvidenceError("exact target-gate boundary self-test failed")

        try:
            _window_distribution(
                totals=np.asarray([1], dtype=np.int64),
                hot=np.asarray([2], dtype=np.int64),
                nodes=np.asarray([0], dtype=np.int64),
                starts=np.asarray([0], dtype=np.int64),
                k=10,
                horizon=4,
            )
        except EvidenceError:
            pass
        else:
            raise EvidenceError("hot>total negative self-test did not fail closed")

        publish = work / "publish"
        publish.mkdir()
        target = publish / "evidence.json"
        _atomic_publish_json(target, {"self_test": "PASS"})
        try:
            _atomic_publish_json(target, {"self_test": "MUST_NOT_OVERWRITE"})
        except FileExistsError:
            pass
        else:
            raise EvidenceError("no-clobber self-test did not reject overwrite")
        if _read_json_object(target, "self-test evidence") != {"self_test": "PASS"}:
            raise EvidenceError("no-clobber self-test changed original evidence")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("W64_GPIC_RATIO_OVERLAY_SELF_TEST_PASS cases=8", flush=True)


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    script_path = _require_canonical_file(Path(__file__).resolve(strict=True), "analysis script")
    repo = script_path.parents[2]
    planner_path = _require_canonical_file(
        repo / "training/dataline_offline/planner_v1.py", "planner source"
    )
    job_yaml = _require_canonical_file(args.job_yaml, "job YAML")
    topology = _validate_job_yaml(job_yaml)
    freeze_path = _require_canonical_file(args.freeze, "freeze manifest")
    plan_root = _require_canonical_directory(args.plan_root, "W64 plan root")
    header_path = _require_canonical_file(plan_root / "header.json", "plan header")
    checksums_path = _require_canonical_file(plan_root / "checksums.json", "plan checksums")
    catalog_header_path = _require_canonical_file(args.catalog_header, "catalog header")
    output = args.output
    if not output.is_absolute():
        raise EvidenceError("output must be absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evidence: {output}")
    work_root = args.work_root
    work_root.mkdir(parents=True, exist_ok=True)
    work_root = _require_canonical_directory(work_root.resolve(strict=True), "work root")
    if not str(work_root).startswith("/local/"):
        raise EvidenceError("temporary work root must be under /local")
    work = Path(
        tempfile.mkdtemp(
            prefix=f"w64_gpic_ratio_overlay.{socket.gethostname()}.{os.getpid()}.",
            dir=work_root,
        )
    )

    source_paths = {
        "script": script_path,
        "planner_source": planner_path,
        "job_yaml": job_yaml,
        "freeze": freeze_path,
        "plan_header": header_path,
        "plan_checksums": checksums_path,
        "catalog_header": catalog_header_path,
    }
    hashes_before = {name: _sha256_file(path) for name, path in source_paths.items()}
    header = _read_json_object(header_path, "plan header")
    checksums = _read_json_object(checksums_path, "plan checksums")
    freeze = _read_json_object(freeze_path, "freeze manifest")
    catalog_header = _read_json_object(catalog_header_path, "catalog header")
    binding, block_snapshots, gpic_catalog_path, gpic_catalog_fingerprint = _bind_frozen_inputs(
        freeze_path=freeze_path,
        freeze=freeze,
        plan_root=plan_root,
        header_path=header_path,
        checksums_path=checksums_path,
        header=header,
        checksums=checksums,
        catalog_header_path=catalog_header_path,
        catalog_header=catalog_header,
        hashes=hashes_before,
    )

    try:
        try:
            plan = OfflinePlan(
                plan_root,
                world_size=EXPECTED_WORLD_SIZE,
                per_rank_bs=EXPECTED_PER_RANK_BS,
                verify_checksums=True,
            )
        except (PlannerInputError, OSError, ValueError) as exc:
            raise EvidenceError(f"OfflinePlan validation failed: {exc}") from exc
        config = plan.config
        actual_pin = {
            "plan_id": header.get("plan_id"),
            "world_size": header.get("world_size"),
            "per_rank_bs": header.get("per_rank_bs"),
            "global_batch_size": header.get("global_batch_size"),
            "num_blocks": header.get("num_blocks"),
            "block_steps": header.get("block_steps"),
            "row_scheme": header.get("row_scheme"),
            "row_id_scheme": header.get("row_id_scheme"),
            "item_row_algorithm": header.get("item_row_mapping", {}).get("algorithm"),
            "catalog_global_rows": header.get("item_row_mapping", {}).get("catalog_global_rows"),
        }
        expected_pin = {
            "plan_id": EXPECTED_PLAN_ID,
            "world_size": EXPECTED_WORLD_SIZE,
            "per_rank_bs": EXPECTED_PER_RANK_BS,
            "global_batch_size": EXPECTED_GLOBAL_BATCH,
            "num_blocks": EXPECTED_PLAN_BLOCKS,
            "block_steps": EXPECTED_BLOCK_STEPS,
            "row_scheme": "virtual_catalog_v1",
            "row_id_scheme": "global",
            "item_row_algorithm": "virtual_catalog_v1",
            "catalog_global_rows": True,
        }
        if actual_pin != expected_pin:
            raise EvidenceError(f"W64 plan identity pin mismatch: {actual_pin}")
        if FORMAL_STOP_STEP > config.total_steps or FORMAL_START_STEP != 0:
            raise EvidenceError("formal interval is outside W64 plan")
        if FORMAL_STOP_STEP * EXPECTED_GLOBAL_BATCH != EXPECTED_FORMAL_EXPOSURES:
            raise EvidenceError("formal global exposure constant drift")
        if NODE_COUNT * RANKS_PER_NODE != EXPECTED_WORLD_SIZE:
            raise EvidenceError("node/rank topology constant drift")
        try:
            domain_index = list(config.domains).index(TARGET_DOMAIN)
        except ValueError as exc:
            raise EvidenceError("GPIC is absent from frozen plan domains") from exc
        domain_totals = [int(value) for value in header.get("N_total", [])]
        if len(domain_totals) != len(config.domains):
            raise EvidenceError("plan N_total/domain count mismatch")
        gpic_lo = sum(domain_totals[:domain_index])
        gpic_size = domain_totals[domain_index]
        gpic_hi = gpic_lo + gpic_size
        if gpic_size != EXPECTED_GPIC_ROWS:
            raise EvidenceError(f"GPIC catalog size drift: {gpic_size}")
        if int(header.get("R_max", {}).get(TARGET_DOMAIN, -1)) != EXPECTED_GPIC_R_MAX:
            raise EvidenceError("GPIC R_max drift")
        if header.get("item_row_mapping", {}).get("virtual_contiguous_range") != [
            0,
            sum(domain_totals),
        ]:
            raise EvidenceError("plan virtual row range drift")
        freeze_gpic = freeze["catalog"]["files"][TARGET_DOMAIN]
        if int(freeze_gpic.get("row_base", -1)) != gpic_lo or int(
            freeze_gpic.get("rows", -1)
        ) != gpic_size:
            raise EvidenceError("plan virtual GPIC range disagrees with frozen catalog")

        used_last_block = (FORMAL_STOP_STEP - 1) // config.block_steps
        used_block_count = used_last_block + 1
        expected_tail = FORMAL_STOP_STEP - used_last_block * config.block_steps
        consumed_steps: list[int] = []
        domain_exposures = np.zeros(len(config.domains), dtype=np.int64)
        step_total = np.zeros((FORMAL_STOP_STEP, NODE_COUNT), dtype=np.uint16)
        step_selected = np.zeros(
            (len(SELECTION_K), FORMAL_STOP_STEP, NODE_COUNT), dtype=np.uint16
        )
        node_seen = np.zeros((NODE_COUNT, gpic_size), dtype=np.bool_)

        # One exact formal-prefix pass.  The remaining five non-formal blocks
        # are SHA-verified without materializing their arrays.
        for block_index in range(EXPECTED_PLAN_BLOCKS):
            if block_index > used_last_block:
                plan._verify_block_checksum(block_index)  # noqa: SLF001
                continue
            _bucket, items, domain_counts = plan._load_block(block_index)  # noqa: SLF001
            global_start = block_index * config.block_steps
            take = min(int(items.shape[0]), FORMAL_STOP_STEP - global_start)
            if take <= 0 or items.shape[1:] != (EXPECTED_WORLD_SIZE, EXPECTED_PER_RANK_BS):
                raise EvidenceError(f"plan block shape/formal slice drift at block {block_index}")
            formal_items = items[:take]
            formal_counts = domain_counts[:take]
            domain_exposures += formal_counts.sum(axis=0, dtype=np.int64)
            mask = (formal_items >= gpic_lo) & (formal_items < gpic_hi)
            declared = int(formal_counts[:, domain_index].sum(dtype=np.int64))
            if int(mask.sum(dtype=np.int64)) != declared:
                raise EvidenceError(f"GPIC item/domain_counts mismatch at block {block_index}")
            reshaped_mask = mask.reshape(
                take, NODE_COUNT, RANKS_PER_NODE, EXPECTED_PER_RANK_BS
            )
            step_total[global_start : global_start + take] = reshaped_mask.sum(
                axis=(2, 3), dtype=np.uint16
            )

            local_residue = np.zeros(mask.shape, dtype=np.uint8)
            local_ids = np.ascontiguousarray(formal_items[mask] - gpic_lo, dtype=np.int64)
            if local_ids.size and (int(local_ids.min()) < 0 or int(local_ids.max()) >= gpic_size):
                raise EvidenceError("GPIC local row id escaped catalog range")
            local_residue[mask] = np.remainder(local_ids, MODULUS).astype(np.uint8, copy=False)
            for node in range(NODE_COUNT):
                rank_lo = node * RANKS_PER_NODE
                rank_hi = rank_lo + RANKS_PER_NODE
                node_mask = mask[:, rank_lo:rank_hi, :]
                node_ids = formal_items[:, rank_lo:rank_hi, :][node_mask] - gpic_lo
                if node_ids.size:
                    node_seen[node, node_ids] = True
            for selection_index, k in enumerate(SELECTION_K):
                selected = mask & (local_residue < k)
                reshaped_selected = selected.reshape(
                    take, NODE_COUNT, RANKS_PER_NODE, EXPECTED_PER_RANK_BS
                )
                step_selected[
                    selection_index, global_start : global_start + take
                ] = reshaped_selected.sum(axis=(2, 3), dtype=np.uint16)
                del selected, reshaped_selected
            consumed_steps.append(take)
            del _bucket, items, domain_counts, formal_items, formal_counts
            del mask, reshaped_mask, local_residue, local_ids, node_mask, node_ids
            _clear_plan_cache(plan)

        if len(plan._verified_blocks) != EXPECTED_PLAN_BLOCKS:  # noqa: SLF001
            raise EvidenceError("not all 168 frozen plan blocks were SHA-verified")
        if consumed_steps != [config.block_steps] * (used_block_count - 1) + [expected_tail]:
            raise EvidenceError("formal block consumption shape drift")
        if int(domain_exposures.sum(dtype=np.int64)) != EXPECTED_FORMAL_EXPOSURES:
            raise EvidenceError("formal global exposure total mismatch")
        if int(domain_exposures[domain_index]) != EXPECTED_FORMAL_GPIC_EXPOSURES:
            raise EvidenceError("formal GPIC exposure total mismatch")
        if int(step_total.sum(dtype=np.int64)) != EXPECTED_FORMAL_GPIC_EXPOSURES:
            raise EvidenceError("node/step GPIC exposure total mismatch")
        if np.any(step_selected > step_total[None, :, :]):
            raise EvidenceError("selected step demand exceeds total GPIC demand")
        for earlier, later in zip(step_selected[:-1], step_selected[1:]):
            if np.any(earlier > later):
                raise EvidenceError("ratio-overlay selected demand is not monotone in k")

        global_seen = np.logical_or.reduce(node_seen, axis=0)
        global_residue_unique = _residue_unique_counts(global_seen)
        node_residue_unique = np.stack(
            [_residue_unique_counts(node_seen[node]) for node in range(NODE_COUNT)], axis=0
        )
        global_exposed_unique = int(global_residue_unique.sum(dtype=np.int64))

        selection_results: list[dict[str, Any]] = []
        all_worst_matrix: list[dict[str, Any]] = []
        for selection_index, k in enumerate(SELECTION_K):
            selected_catalog_unique = _selection_catalog_unique(k)
            selected_exposed_unique = int(global_residue_unique[:k].sum(dtype=np.int64))
            selected_exposures = int(step_selected[selection_index].sum(dtype=np.int64))
            per_node = []
            aggregate_arrays: dict[
                int, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]
            ] = {horizon: ([], [], [], []) for horizon in WINDOW_HORIZONS}
            all_worst_checks: list[dict[str, Any]] = []

            global_full_prefix = _capacity_record(
                selected_exposures,
                EXPECTED_FORMAL_GPIC_EXPOSURES,
                nominal_demand_numerator=EXPECTED_FORMAL_GPIC_EXPOSURES,
                nominal_demand_denominator=1,
            )
            all_worst_checks.append(
                {
                    "scope": "global_full_formal_prefix_average",
                    "target_gate_pass": global_full_prefix["capacity_gates"]["target"]["pass"],
                }
            )

            for node in range(NODE_COUNT):
                node_total = int(step_total[:, node].sum(dtype=np.int64))
                node_hot = int(step_selected[selection_index, :, node].sum(dtype=np.int64))
                node_unique = int(node_residue_unique[node, :k].sum(dtype=np.int64))
                node_full_prefix = _capacity_record(
                    node_hot,
                    node_total,
                    nominal_demand_numerator=EXPECTED_FORMAL_GPIC_EXPOSURES,
                    nominal_demand_denominator=NODE_COUNT,
                )
                all_worst_checks.append(
                    {
                        "scope": "node_full_formal_prefix_average",
                        "node": node,
                        "target_gate_pass": node_full_prefix["capacity_gates"]["target"]["pass"],
                    }
                )
                node_record: dict[str, Any] = {
                    "node": node,
                    "global_ranks": list(
                        range(node * RANKS_PER_NODE, (node + 1) * RANKS_PER_NODE)
                    ),
                    "selected_unique_row_ids_needed_by_node": node_unique,
                    "selected_node_storage_estimates": _storage_estimates(
                        node_unique, args.storage_bytes
                    ),
                    "full_formal_prefix_average_capacity": node_full_prefix,
                    "horizons": [],
                }
                for horizon in WINDOW_HORIZONS:
                    starts, totals = _anchored_window_sums(step_total[:, node], horizon)
                    selected_starts, hot_values = _anchored_window_sums(
                        step_selected[selection_index, :, node], horizon
                    )
                    if not np.array_equal(starts, selected_starts):
                        raise EvidenceError("total/selected window anchors disagree")
                    nodes = np.full(starts.shape, node, dtype=np.int64)
                    distribution = _window_distribution(
                        totals=totals,
                        hot=hot_values,
                        nodes=nodes,
                        starts=starts,
                        k=k,
                        horizon=horizon,
                    )
                    distribution.update(
                        {
                            "last_included_start_step": int(starts[-1]),
                            "formal_tail_steps_not_covered_by_last_full_window": (
                                FORMAL_STOP_STEP - int(starts[-1] + horizon)
                            ),
                        }
                    )
                    node_record["horizons"].append(distribution)
                    worst = distribution["absolute_cold_source_demand"]["points"]["max"]
                    all_worst_checks.append(
                        {
                            "scope": "node_horizon_max_absolute_cold",
                            "node": node,
                            "horizon_steps": horizon,
                            "absolute_cold_source_exposures": worst[
                                "absolute_cold_source_exposures"
                            ],
                            "target_gate_pass": worst["capacity_gates"]["target"]["pass"],
                            "location_digest_sha256": worst["location_digest_sha256"],
                        }
                    )
                    totals_list, hot_list, nodes_list, starts_list = aggregate_arrays[horizon]
                    totals_list.append(totals)
                    hot_list.append(hot_values)
                    nodes_list.append(nodes)
                    starts_list.append(starts)
                per_node.append(node_record)

            aggregate_horizons = []
            for horizon in WINDOW_HORIZONS:
                totals_list, hot_list, nodes_list, starts_list = aggregate_arrays[horizon]
                distribution = _window_distribution(
                    totals=np.concatenate(totals_list),
                    hot=np.concatenate(hot_list),
                    nodes=np.concatenate(nodes_list),
                    starts=np.concatenate(starts_list),
                    k=k,
                    horizon=horizon,
                )
                distribution["population_scope"] = "all_8_nodes_each_window_equal_weight"
                aggregate_horizons.append(distribution)

            all_worst_pass = all(bool(row["target_gate_pass"]) for row in all_worst_checks)
            all_worst_record = {
                "k": k,
                "selection_fraction": _fraction(k, MODULUS),
                "definition": (
                    "global and every-node full-prefix averages plus every one of "
                    "8 nodes x 4/16/64-step maximum absolute-cold windows"
                ),
                "check_count": len(all_worst_checks),
                "all_target_1.25_gates_pass": all_worst_pass,
                "failed_checks": [row for row in all_worst_checks if not row["target_gate_pass"]],
            }
            all_worst_matrix.append(all_worst_record)
            selection_results.append(
                {
                    "k": k,
                    "selection_rule": f"gpic_local_row_id % {MODULUS} < {k}",
                    "selection_fraction": _fraction(k, MODULUS),
                    "selected_catalog_unique_row_ids": selected_catalog_unique,
                    "selected_formal_exposed_unique_row_ids": selected_exposed_unique,
                    "formal_unexposed_selected_catalog_row_ids": (
                        selected_catalog_unique - selected_exposed_unique
                    ),
                    "selected_formal_exposures": selected_exposures,
                    "selected_exposure_coverage": _fraction(
                        selected_exposures, EXPECTED_FORMAL_GPIC_EXPOSURES
                    ),
                    "cold_source_formal_exposures": (
                        EXPECTED_FORMAL_GPIC_EXPOSURES - selected_exposures
                    ),
                    "global_overlay_storage_estimates": _storage_estimates(
                        selected_catalog_unique, args.storage_bytes
                    ),
                    "global_full_formal_prefix_average_capacity": global_full_prefix,
                    "per_node": per_node,
                    "aggregate_node_windows": aggregate_horizons,
                    "all_worst_target_gate": all_worst_record,
                }
            )

        minimum = next(
            (row for row in all_worst_matrix if row["all_target_1.25_gates_pass"]), None
        )
        minimum_decision = (
            {
                "found_within_evaluated_ratios": True,
                "k": int(minimum["k"]),
                "selection_fraction": minimum["selection_fraction"],
                "selection_percent": format(100 * int(minimum["k"]) / MODULUS, ".1f"),
            }
            if minimum is not None
            else {
                "found_within_evaluated_ratios": False,
                "k": None,
                "selection_fraction": None,
                "selection_percent": None,
            }
        )

        blocks_root = plan_root / "blocks"
        for name, before in block_snapshots.items():
            if _fingerprint(blocks_root / name) != before:
                raise EvidenceError(f"plan block stat changed during analysis: {name}")
        if _fingerprint(gpic_catalog_path) != gpic_catalog_fingerprint:
            raise EvidenceError("GPIC catalog parquet stat changed during analysis")
        hashes_after = {name: _sha256_file(path) for name, path in source_paths.items()}
        if hashes_after != hashes_before:
            raise EvidenceError("bound source/metadata files changed during analysis")

        result_document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "decision_scope": "diagnostic_only_not_a_FORMAL_promotion_authority",
            "scope": {
                "formal_steps": {
                    "start_inclusive": FORMAL_START_STEP,
                    "stop_exclusive": FORMAL_STOP_STEP,
                    "count": FORMAL_STOP_STEP - FORMAL_START_STEP,
                },
                "world_size": EXPECTED_WORLD_SIZE,
                "node_count": NODE_COUNT,
                "ranks_per_node": RANKS_PER_NODE,
                "per_rank_batch_size": EXPECTED_PER_RANK_BS,
                "global_batch_size": EXPECTED_GLOBAL_BATCH,
                "formal_global_exposures": EXPECTED_FORMAL_EXPOSURES,
                "target_domain": TARGET_DOMAIN,
                "gpic_row_count": EXPECTED_GPIC_ROWS,
                "gpic_virtual_row_range": [gpic_lo, gpic_hi],
                "formal_gpic_exposures": EXPECTED_FORMAL_GPIC_EXPOSURES,
                "formal_gpic_unique_row_ids_exposed": global_exposed_unique,
                "image_payload_reads": 0,
                "catalog_parquet_payload_reads": 0,
                "catalog_parquet_access_mode": "filesystem_metadata_only_no_open_or_payload_read",
                "sample_paths_read": 0,
                "sample_paths_emitted": 0,
                "bitmap_artifacts_published": 0,
            },
            "selection_contract": {
                "local_row_id_namespace": "gpic_global_row_id_minus_frozen_gpic_row_base",
                "modulus": MODULUS,
                "selected_when": "gpic_local_row_id % 20 < k",
                "evaluated_k_in_ascending_order": list(SELECTION_K),
                "selection_is_plan_independent": True,
            },
            "capacity_model_contract": {
                "baseline_source_capacity_ratio": "0.787",
                "fixed_nominal_average_gpic_exposures_per_node_step": _positive_ratio(
                    EXPECTED_FORMAL_GPIC_EXPOSURES, FORMAL_STOP_STEP * NODE_COUNT
                ),
                "adjusted_capacity_formula": (
                    "0.787 * fixed nominal GPIC demand for interval / "
                    "observed absolute cold-source exposures"
                ),
                "gates": {
                    name: format(numerator / denominator, ".3f")
                    for name, numerator, denominator in CAPACITY_GATES
                },
                "gate_comparison": "exact_integer_cross_multiplication",
                "window_horizons_steps": list(WINDOW_HORIZONS),
                "window_anchor_stride_steps": WINDOW_STRIDE_STEPS,
                "absolute_cold_upper_tail_points": [name for name, _ in UPPER_TAIL_POINTS],
                "coverage_is_context_only_not_capacity_authority": True,
                "node_capacity_allocation_assumption": (
                    "the measured 0.787 aggregate baseline is interpreted as eight "
                    "homogeneous nodes with equal per-node source capacity"
                ),
            },
            "minimum_evaluated_ratio_whose_all_worst_cases_pass_target_1.25": minimum_decision,
            "all_worst_target_gate_matrix": all_worst_matrix,
            "ratios": selection_results,
            "formal_prefix_domain_exposures": {
                domain: int(domain_exposures[index])
                for index, domain in enumerate(config.domains)
            },
            "input_binding": {
                "plan_root": str(plan_root),
                "plan_header": {
                    "path": str(header_path),
                    "sha256": hashes_before["plan_header"],
                    "bytes": header_path.stat().st_size,
                },
                "plan_checksums": {
                    "path": str(checksums_path),
                    "sha256": hashes_before["plan_checksums"],
                    "bytes": checksums_path.stat().st_size,
                },
                "all_plan_blocks_sha256_verified": True,
                "all_plan_block_count": EXPECTED_PLAN_BLOCKS,
                "formal_array_blocks_streamed_once": used_block_count,
                "formal_array_block_index_range_inclusive": [0, used_last_block],
                "nonformal_blocks_sha_only_verified": EXPECTED_PLAN_BLOCKS - used_block_count,
                "maximum_materialized_plan_blocks": 1,
                "freeze_and_catalog_binding": binding,
                "source_files": {
                    name: {"path": str(source_paths[name]), "sha256": hashes_before[name]}
                    for name in ("script", "planner_source", "job_yaml")
                },
            },
            "storage_model": {
                "central_object_bytes": CENTRAL_OBJECT_BYTES,
                "storage_byte_sensitivity": list(args.storage_bytes),
                "local_capacity_sensitivity_gib": list(LOCAL_CAPACITY_GIB),
                "excludes": [
                    "blob index bytes",
                    "filesystem metadata",
                    "allocator padding",
                    "publication manifests",
                ],
            },
            "limitations": {
                "not_measured": [
                    "actual selected object sizes",
                    "image reads or decoding",
                    "blob builder throughput",
                    "FSx blob random-read latency",
                    "source-request coalescing",
                    "runtime rank wall-clock interleaving",
                ],
                "ratio_overlay_assumption": (
                    "all selected objects are resident in the overlay before demand; every "
                    "non-selected exposure consumes one source service"
                ),
            },
            "runtime": {
                "hostname": socket.gethostname(),
                "python_executable": str(Path(sys.executable).resolve(strict=True)),
                "python_version": sys.version,
                "numpy_version": np.__version__,
                "os_cpu_count": os.cpu_count(),
                "job_yaml_topology": topology,
                "temporary_root": str(work_root),
                "temporary_artifacts_retained": False,
            },
        }
        output_sha = _atomic_publish_json(output, result_document)
        return result_document, output_sha
    finally:
        shutil.rmtree(work, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plan-root", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--catalog-header", type=Path)
    parser.add_argument("--job-yaml", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--storage-bytes",
        type=_parse_positive_ints,
        default=DEFAULT_STORAGE_BYTES,
        help="comma-separated object-byte sensitivity values; must include 144296",
    )
    args = parser.parse_args(argv)
    if CENTRAL_OBJECT_BYTES not in args.storage_bytes:
        parser.error(f"--storage-bytes must include central observed mean {CENTRAL_OBJECT_BYTES}")
    if not args.self_test:
        missing = [
            name
            for name in ("plan_root", "freeze", "catalog_header", "job_yaml", "output")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"missing required analysis arguments: {missing}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            _self_test(args.work_root)
            return 0
        result, output_sha = analyze(args)
        decision = result["minimum_evaluated_ratio_whose_all_worst_cases_pass_target_1.25"]
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": str(args.output),
                    "output_sha256": output_sha,
                    "gpic_exposures": result["scope"]["formal_gpic_exposures"],
                    "evaluated_k": list(SELECTION_K),
                    "minimum_all_worst_target_k": decision["k"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except (EvidenceError, FileExistsError, OSError, ValueError) as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
