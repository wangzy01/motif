#!/usr/bin/env python3
"""Exact no-image analysis of count-threshold GPIC hotsets in the W64 plan.

The analyzer makes two streaming passes over the frozen plan.  Pass one counts
all formal GPIC row-id exposures and constructs the count>=8/7/6/5 hotsets.
Pass two measures each hotset by node and in 4-step-aligned 4/16/64-step
look-ahead windows.  It never opens the GPIC parquet payload or an image.

This is diagnostic evidence only.  In particular, its adjusted-capacity model
assumes that every selected object is already resident and that every
non-selected exposure is served by the measured source.  It is not a training
promotion authority.
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


SCHEMA_VERSION = "offline_arb_w64_gpic_hotset_v1"
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
RANKS_PER_NODE = 8
NODE_COUNT = 8
TARGET_DOMAIN = "gpic"
HOT_THRESHOLDS = (8, 7, 6, 5)
REUSABLE_BITMAP_THRESHOLD = 6
WINDOW_STRIDE_STEPS = 4
PREFETCH_HORIZON_STEPS = (4, 16, 64)
LOWER_QUANTILE_BASIS_POINTS = (0, 10, 100, 500, 5_000)
LOWER_QUANTILE_NAMES = {
    0: "min",
    10: "p0.1",
    100: "p1",
    500: "p5",
    5_000: "median_lower",
}
DEFAULT_STORAGE_BYTES = (98_304, 114_688, 131_072, 144_296, 163_840, 196_608)
CENTRAL_OBJECT_BYTES = 144_296
LOCAL_CAPACITY_GIB = (250, 500)
BASELINE_CAPACITY_NUMERATOR = 787
BASELINE_CAPACITY_DENOMINATOR = 1_000
CAPACITY_GATES = (("minimum", 1_111, 1_000), ("target", 5, 4))


class EvidenceError(RuntimeError):
    """The frozen-input or accounting contract was not met."""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=1024 * 1024) as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


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


def _fingerprint(path: Path, expect_directory: bool = False) -> dict[str, int]:
    value = path.stat(follow_symlinks=False)
    expected = stat.S_ISDIR(value.st_mode) if expect_directory else stat.S_ISREG(value.st_mode)
    if not expected:
        raise EvidenceError(f"unexpected input type: {path}")
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _require_record(record: Any, path: Path, label: str, *, sha256: str | None = None) -> dict[str, int]:
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
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
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
    return _atomic_publish_bytes(output, _canonical_bytes(value))


def _atomic_publish_bytes(output: Path, payload: bytes) -> str:
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


def _create_exclusive_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise EvidenceError("artifact directory must be absolute")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to reuse artifact directory: {path}")
    parent = _require_canonical_directory(path.parent, "artifact directory parent")
    os.mkdir(path, mode=0o700)
    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return _require_canonical_directory(path, "new artifact directory")


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
    _require_record(inputs.get("plan_header"), header_path, "inputs.plan_header", sha256=hashes["plan_header"])
    _require_record(inputs.get("plan_checksums"), checksums_path, "inputs.plan_checksums", sha256=hashes["plan_checksums"])
    _require_record(inputs.get("catalog_header"), catalog_header_path, "inputs.catalog_header", sha256=hashes["catalog_header"])

    if set(plan_root.iterdir()) != {header_path, checksums_path, plan_root / "blocks"}:
        raise EvidenceError("plan root entry set drift")
    blocks_root = _require_canonical_directory(plan_root / "blocks", "plan blocks")
    directories = plan_blocks.get("directories")
    if not isinstance(directories, Mapping) or set(directories) != {"root", "blocks"}:
        raise EvidenceError("freeze plan directory set drift")
    for name, path in (("root", plan_root), ("blocks", blocks_root)):
        record = directories[name]
        if not isinstance(record, Mapping) or record.get("path") != str(path):
            raise EvidenceError(f"freeze plan directory record mismatch: {name}")
        if record.get("fingerprint") != _fingerprint(path, expect_directory=True):
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
        live = _require_record(frozen_files[name], path, f"plan_blocks.files.{name}", sha256=expected_sha)
        snapshots[name] = live
        manifest.append({"name": name, "bytes": live["size_bytes"], "sha256": expected_sha, "fingerprint": live})

    catalog_directory = catalog.get("directory")
    catalog_files = catalog.get("files")
    if not isinstance(catalog_directory, Mapping) or not isinstance(catalog_files, Mapping):
        raise EvidenceError("freeze catalog directory/files objects are missing")
    catalog_root = _require_canonical_directory(catalog_header_path.parent, "catalog root")
    if catalog_directory.get("path") != str(catalog_root):
        raise EvidenceError("freeze catalog root path mismatch")
    if catalog_directory.get("fingerprint") != _fingerprint(catalog_root, expect_directory=True):
        raise EvidenceError("freeze catalog root fingerprint mismatch")
    gpic_record = catalog_files.get(TARGET_DOMAIN)
    if not isinstance(gpic_record, Mapping):
        raise EvidenceError("freeze GPIC catalog file record is missing")
    gpic_path = _require_canonical_file(Path(str(gpic_record.get("path"))), "GPIC catalog parquet")
    gpic_live = _require_record(gpic_record, gpic_path, "catalog.files.gpic", sha256=str(gpic_record.get("sha256")))
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

    return (
        {
            "freeze": {"path": str(freeze_path), "sha256": hashes["freeze"], "schema_version": freeze.get("schema_version")},
            "plan_block_count": len(manifest),
            "plan_blocks_manifest_digest_contract": "canonical_json_sorted_list(name,bytes,sha256,fingerprint)",
            "plan_blocks_manifest_sha256": _sha256_bytes(_canonical_bytes(manifest)),
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


def _validate_count_histogram(
    counts: np.ndarray,
    *,
    domain_size: int,
    r_max: int,
    expected_exposures: int,
) -> tuple[np.ndarray, int]:
    if counts.dtype != np.uint16 or counts.ndim != 1 or counts.size != domain_size:
        raise EvidenceError("GPIC count vector dtype/shape drift")
    maximum = int(counts.max(initial=0))
    if maximum > r_max:
        raise EvidenceError(f"GPIC row exposure count {maximum} exceeds R_max={r_max}")
    histogram = np.bincount(counts, minlength=r_max + 1)
    if histogram.size != r_max + 1:
        raise EvidenceError("GPIC histogram escaped [0,R_max]")
    if int(histogram.sum(dtype=np.int64)) != domain_size:
        raise EvidenceError("GPIC histogram population is not the 20M catalog rows")
    weighted = int(np.dot(np.arange(r_max + 1, dtype=np.int64), histogram.astype(np.int64)))
    if weighted != expected_exposures:
        raise EvidenceError(f"GPIC weighted histogram exposure mismatch: {weighted} != {expected_exposures}")
    if int(counts.sum(dtype=np.int64)) != weighted:
        raise EvidenceError("GPIC count-vector/histogram weighted sum mismatch")
    return histogram, maximum


def _storage_estimates(unique_objects: int, byte_values: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    for object_bytes in byte_values:
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


def _positive_ratio(numerator: int, denominator: int) -> dict[str, Any]:
    if numerator < 0 or denominator <= 0:
        raise EvidenceError(f"invalid nonnegative ratio {numerator}/{denominator}")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "decimal": format(numerator / denominator, ".12f"),
    }


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
        adjusted: dict[str, Any] = {"numerator": None, "denominator": None, "decimal": "infinity"}
    else:
        numerator = BASELINE_CAPACITY_NUMERATOR * nominal_demand_numerator
        denominator = BASELINE_CAPACITY_DENOMINATOR * nominal_demand_denominator * cold
        adjusted = {
            "numerator": numerator,
            "denominator": denominator,
            "decimal": format(numerator / denominator, ".12f"),
        }
    gates = {}
    for name, gate_num, gate_den in CAPACITY_GATES:
        passes = cold == 0 or (
            BASELINE_CAPACITY_NUMERATOR * nominal_demand_numerator * gate_den
            >= BASELINE_CAPACITY_DENOMINATOR * nominal_demand_denominator * cold * gate_num
        )
        gates[name] = {
            "threshold": format(gate_num / gate_den, ".3f"),
            "pass": bool(passes),
            "comparison": "exact_integer_cross_multiplication",
        }
    return {
        "hot_coverage": _fraction(hot, total),
        "cold_source_fraction": _fraction(cold, total),
        "nominal_average_gpic_demand_exposures_for_interval": _positive_ratio(
            nominal_demand_numerator, nominal_demand_denominator
        ),
        "observed_total_gpic_demand_over_nominal_average": _positive_ratio(
            total * nominal_demand_denominator, nominal_demand_numerator
        ),
        "observed_cold_source_demand_over_nominal_average": _positive_ratio(
            cold * nominal_demand_denominator, nominal_demand_numerator
        ),
        "baseline_source_capacity_ratio": "0.787",
        "adjusted_source_capacity_ratio_baseline_times_nominal_demand_divided_by_observed_cold_demand": adjusted,
        "capacity_gates": gates,
    }


def _location_digest(*, threshold: int, horizon: int, node: int, start: int, end: int) -> str:
    identity = {
        "plan_id": EXPECTED_PLAN_ID,
        "threshold": threshold,
        "horizon_steps": horizon,
        "node": node,
        "start_step_inclusive": start,
        "end_step_exclusive": end,
    }
    return _sha256_bytes(_canonical_bytes(identity))


def _coverage_distribution(
    *,
    totals: np.ndarray,
    hot: np.ndarray,
    nodes: np.ndarray,
    starts: np.ndarray,
    threshold: int,
    horizon: int,
) -> dict[str, Any]:
    arrays = (totals, hot, nodes, starts)
    if any(value.ndim != 1 for value in arrays) or len({value.size for value in arrays}) != 1:
        raise EvidenceError("window distribution arrays have inconsistent shapes")
    if totals.size == 0:
        raise EvidenceError("window distribution is empty")
    if np.any(hot < 0) or np.any(totals < 0) or np.any(hot > totals):
        raise EvidenceError("window hot counts are outside [0,total]")
    if np.any(totals > horizon * RANKS_PER_NODE * EXPECTED_PER_RANK_BS):
        raise EvidenceError("window GPIC demand exceeds the node batch-size upper bound")
    valid = totals > 0
    zero_demand = int((~valid).sum())
    if not np.any(valid):
        raise EvidenceError("all windows have zero GPIC demand")
    vt = totals[valid].astype(np.int64, copy=False)
    vh = hot[valid].astype(np.int64, copy=False)
    vn = nodes[valid].astype(np.int64, copy=False)
    vs = starts[valid].astype(np.int64, copy=False)
    ratios = vh.astype(np.float64) / vt.astype(np.float64)
    order = np.lexsort((vs, vn, ratios))
    nominal_numerator = EXPECTED_FORMAL_GPIC_EXPOSURES * horizon
    nominal_denominator = FORMAL_STOP_STEP * NODE_COUNT
    points: dict[str, Any] = {}
    population = int(order.size)
    for basis_points in LOWER_QUANTILE_BASIS_POINTS:
        rank = 0 if basis_points == 0 else math.ceil(basis_points * population / 10_000) - 1
        index = int(order[rank])
        total = int(vt[index])
        selected = int(vh[index])
        node = int(vn[index])
        start = int(vs[index])
        record = _capacity_record(
            selected,
            total,
            nominal_demand_numerator=nominal_numerator,
            nominal_demand_denominator=nominal_denominator,
        )
        record.update(
            {
                "nearest_rank_zero_based": rank,
                "node": node,
                "start_step_inclusive": start,
                "end_step_exclusive": start + horizon,
                "location_digest_sha256": _location_digest(
                    threshold=threshold, horizon=horizon, node=node, start=start, end=start + horizon
                ),
            }
        )
        points[LOWER_QUANTILE_NAMES[basis_points]] = record
    cold = vt - vh
    # The nominal demand is fixed for a given horizon, so the lowest adjusted
    # capacity is exactly the window with the largest absolute cold demand.
    capacity_order = np.lexsort((vs, vn, -cold))
    worst_index = int(capacity_order[0])
    worst_total = int(vt[worst_index])
    worst_hot = int(vh[worst_index])
    worst_node = int(vn[worst_index])
    worst_start = int(vs[worst_index])
    capacity_worst = _capacity_record(
        worst_hot,
        worst_total,
        nominal_demand_numerator=nominal_numerator,
        nominal_demand_denominator=nominal_denominator,
    )
    capacity_worst.update(
        {
            "node": worst_node,
            "start_step_inclusive": worst_start,
            "end_step_exclusive": worst_start + horizon,
            "location_digest_sha256": _location_digest(
                threshold=threshold,
                horizon=horizon,
                node=worst_node,
                start=worst_start,
                end=worst_start + horizon,
            ),
            "selection_rule": "maximum_absolute_cold_exposures_then_node_then_start",
        }
    )
    return {
        "window_population": int(totals.size),
        "positive_demand_window_population": population,
        "zero_demand_window_population_excluded_from_fraction_quantiles": zero_demand,
        "hot_coverage_lower_tail_quantile_method": "nearest_rank_ceil(q*n)-1_sorted_by_float64_ratio_then_node_then_start; exact ordering because denominators<=horizon*288<=18432 gives distinct-rational separation>2.9e-9",
        "cold_source_mapping": "hot min/p0.1/p1/p5/median correspond to cold max/p99.9/p99/p95/median_upper",
        "points": points,
        "minimum_hot_coverage_window_location_digest_sha256": points["min"]["location_digest_sha256"],
        "minimum_adjusted_capacity_window": capacity_worst,
        "worst_window_location_digest_sha256": capacity_worst["location_digest_sha256"],
        "worst_window_gate_result": capacity_worst["capacity_gates"],
    }


def _window_sums(values: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 1 or values.size != FORMAL_STOP_STEP:
        raise EvidenceError("step vector shape drift")
    starts = np.arange(0, FORMAL_STOP_STEP - horizon + 1, WINDOW_STRIDE_STEPS, dtype=np.int64)
    prefix = np.empty(FORMAL_STOP_STEP + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(values, dtype=np.int64, out=prefix[1:])
    return starts, prefix[starts + horizon] - prefix[starts]


def _self_test(work_root: Path) -> None:
    work_root.mkdir(parents=True, exist_ok=True)
    root = _require_canonical_directory(work_root.resolve(strict=True), "self-test work root")
    if not str(root).startswith("/local/"):
        raise EvidenceError("self-test work root must be under /local")
    work = Path(tempfile.mkdtemp(prefix="w64_gpic_hotset_selftest.", dir=root))
    try:
        counts = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20], dtype=np.uint16)
        histogram, maximum = _validate_count_histogram(counts, domain_size=12, r_max=20, expected_exposures=75)
        if int(histogram.sum()) != 12 or maximum != 20 or int(counts[counts >= 8].sum()) != 47:
            raise EvidenceError("tiny histogram self-test mismatch")
        invalid = counts.copy()
        invalid[0] = 21
        try:
            _validate_count_histogram(invalid, domain_size=12, r_max=20, expected_exposures=96)
        except EvidenceError:
            pass
        else:
            raise EvidenceError("negative R_max self-test did not fail closed")
        totals = np.asarray([8, 10, 12], dtype=np.int64)
        hot = np.asarray([4, 9, 3], dtype=np.int64)
        nodes = np.asarray([0, 0, 1], dtype=np.int64)
        starts = np.asarray([0, 4, 0], dtype=np.int64)
        distribution = _coverage_distribution(
            totals=totals, hot=hot, nodes=nodes, starts=starts, threshold=8, horizon=4
        )
        if distribution["points"]["min"]["hot_coverage"]["numerator"] != 3:
            raise EvidenceError("tiny coverage self-test mismatch")
        separated = _coverage_distribution(
            totals=np.asarray([10, 100], dtype=np.int64),
            hot=np.asarray([0, 80], dtype=np.int64),
            nodes=np.asarray([0, 0], dtype=np.int64),
            starts=np.asarray([0, 4], dtype=np.int64),
            threshold=8,
            horizon=4,
        )
        if separated["points"]["min"]["start_step_inclusive"] != 0:
            raise EvidenceError("minimum-hot-coverage selection self-test mismatch")
        if separated["minimum_adjusted_capacity_window"]["start_step_inclusive"] != 4:
            raise EvidenceError("absolute-cold capacity-worst selection self-test mismatch")
        try:
            _coverage_distribution(
                totals=np.asarray([1]), hot=np.asarray([2]), nodes=np.asarray([0]), starts=np.asarray([0]), threshold=8, horizon=4
            )
        except EvidenceError:
            pass
        else:
            raise EvidenceError("negative hot>total self-test did not fail closed")
        small_all_cold = _capacity_record(
            0, 10, nominal_demand_numerator=100, nominal_demand_denominator=1
        )["adjusted_source_capacity_ratio_baseline_times_nominal_demand_divided_by_observed_cold_demand"]
        large_half_cold = _capacity_record(
            5_000, 10_000, nominal_demand_numerator=100, nominal_demand_denominator=1
        )["adjusted_source_capacity_ratio_baseline_times_nominal_demand_divided_by_observed_cold_demand"]
        if not all(
            isinstance(value, int)
            for value in (
                small_all_cold["numerator"],
                small_all_cold["denominator"],
                large_half_cold["numerator"],
                large_half_cold["denominator"],
            )
        ):
            raise EvidenceError("finite capacity self-test did not return integer rationals")
        if (
            small_all_cold["numerator"] * large_half_cold["denominator"]
            <= large_half_cold["numerator"] * small_all_cold["denominator"]
        ):
            raise EvidenceError("absolute cold-demand capacity normalization self-test failed")
        publish_dir = work / "publish"
        publish_dir.mkdir()
        target = publish_dir / "evidence.json"
        _atomic_publish_json(target, {"self_test": "PASS"})
        try:
            _atomic_publish_json(target, {"self_test": "MUST_NOT_OVERWRITE"})
        except FileExistsError:
            pass
        else:
            raise EvidenceError("atomic no-clobber self-test did not reject overwrite")
        if _read_json_object(target, "self-test evidence") != {"self_test": "PASS"}:
            raise EvidenceError("atomic no-clobber changed the original payload")
        artifact_dir = _create_exclusive_directory(work / "artifacts")
        artifact = artifact_dir / "tiny.packbits"
        if _atomic_publish_bytes(artifact, b"\x05") != hashlib.sha256(b"\x05").hexdigest():
            raise EvidenceError("atomic bitmap self-test SHA mismatch")
        try:
            _create_exclusive_directory(artifact_dir)
        except FileExistsError:
            pass
        else:
            raise EvidenceError("artifact-directory no-clobber self-test did not fail closed")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("W64_GPIC_HOTSET_SELF_TEST_PASS cases=8", flush=True)


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    script_path = _require_canonical_file(Path(__file__).resolve(strict=True), "analysis script")
    repo = script_path.parents[2]
    planner_path = _require_canonical_file(repo / "training/dataline_offline/planner_v1.py", "planner source")
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
    artifact_dir = args.artifact_dir
    if not artifact_dir.is_absolute():
        raise EvidenceError("artifact directory must be absolute")
    if artifact_dir.exists() or artifact_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse artifact directory: {artifact_dir}")
    if artifact_dir.parent != output.parent:
        raise EvidenceError("artifact directory and analysis JSON must share one canonical attempt directory")
    work_root = args.work_root
    work_root.mkdir(parents=True, exist_ok=True)
    work_root = _require_canonical_directory(work_root.resolve(strict=True), "work root")
    if not str(work_root).startswith("/local/"):
        raise EvidenceError("temporary work root must be under /local")
    work = Path(tempfile.mkdtemp(prefix=f"w64_gpic_hotset.{socket.gethostname()}.{os.getpid()}.", dir=work_root))

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
            raise EvidenceError("formal interval is outside the W64 plan")
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
        if header.get("item_row_mapping", {}).get("virtual_contiguous_range") != [0, sum(domain_totals)]:
            raise EvidenceError("plan virtual row range drift")
        freeze_gpic = freeze["catalog"]["files"][TARGET_DOMAIN]
        if int(freeze_gpic.get("row_base", -1)) != gpic_lo or int(freeze_gpic.get("rows", -1)) != gpic_size:
            raise EvidenceError("plan virtual GPIC range disagrees with frozen catalog")

        used_last_block = (FORMAL_STOP_STEP - 1) // config.block_steps
        used_block_count = used_last_block + 1
        counts = np.zeros(gpic_size, dtype=np.uint16)
        domain_exposures_pass1 = np.zeros(len(config.domains), dtype=np.int64)
        consumed_steps: list[int] = []

        # Pass 1: exact per-row exposure counts.
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
            domain_exposures_pass1 += formal_counts.sum(axis=0, dtype=np.int64)
            mask = (formal_items >= gpic_lo) & (formal_items < gpic_hi)
            declared = int(formal_counts[:, domain_index].sum(dtype=np.int64))
            if int(mask.sum(dtype=np.int64)) != declared:
                raise EvidenceError(f"GPIC item/domain_counts mismatch at block {block_index}")
            ids = np.ascontiguousarray(formal_items[mask] - gpic_lo, dtype=np.int64)
            if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= gpic_size):
                raise EvidenceError("GPIC local row id escaped catalog range")
            np.add.at(counts, ids, 1)
            consumed_steps.append(take)
            del _bucket, items, domain_counts, formal_items, formal_counts, mask, ids
            _clear_plan_cache(plan)

        if len(plan._verified_blocks) != EXPECTED_PLAN_BLOCKS:  # noqa: SLF001
            raise EvidenceError("not all 168 frozen plan blocks were SHA-verified")
        expected_tail = FORMAL_STOP_STEP - used_last_block * config.block_steps
        if consumed_steps != [config.block_steps] * (used_block_count - 1) + [expected_tail]:
            raise EvidenceError("formal block consumption shape drift")
        if int(domain_exposures_pass1.sum()) != EXPECTED_FORMAL_EXPOSURES:
            raise EvidenceError("formal global exposure total mismatch")
        if int(domain_exposures_pass1[domain_index]) != EXPECTED_FORMAL_GPIC_EXPOSURES:
            raise EvidenceError("formal GPIC exposure total mismatch")
        histogram, maximum_count = _validate_count_histogram(
            counts,
            domain_size=EXPECTED_GPIC_ROWS,
            r_max=EXPECTED_GPIC_R_MAX,
            expected_exposures=EXPECTED_FORMAL_GPIC_EXPOSURES,
        )

        selected_masks: dict[int, np.ndarray] = {}
        threshold_base: dict[int, dict[str, Any]] = {}
        bitmap_records = []
        reusable_bitmap_payload: bytes | None = None
        for threshold in HOT_THRESHOLDS:
            selected = counts >= threshold
            selected_unique = int(np.count_nonzero(selected))
            selected_exposures = int(counts[selected].sum(dtype=np.int64))
            bitmap = np.packbits(selected, bitorder="little")
            bitmap_sha = hashlib.sha256(memoryview(bitmap)).hexdigest()
            if threshold == REUSABLE_BITMAP_THRESHOLD:
                reusable_bitmap_payload = bitmap.tobytes(order="C")
            selected_masks[threshold] = selected
            record = {
                "count_threshold_inclusive": threshold,
                "selected_unique_row_ids": selected_unique,
                "selected_exposures": selected_exposures,
                "hot_exposure_coverage": _fraction(selected_exposures, EXPECTED_FORMAL_GPIC_EXPOSURES),
                "cold_source_exposures_under_pinned_residency_model": EXPECTED_FORMAL_GPIC_EXPOSURES - selected_exposures,
                "selected_bitmap_encoding": "20M_gpic_local_row_ids_ascending_numpy_packbits_little",
                "selected_bitmap_bytes": int(bitmap.size),
                "selected_bitmap_sha256": bitmap_sha,
                "global_storage_estimates": _storage_estimates(selected_unique, args.storage_bytes),
            }
            threshold_base[threshold] = record
            bitmap_records.append({"threshold": threshold, "sha256": bitmap_sha, "bytes": int(bitmap.size)})
            del bitmap
        if reusable_bitmap_payload is None:
            raise EvidenceError("reusable count>=6 bitmap was not materialized")
        for higher, lower in zip(HOT_THRESHOLDS, HOT_THRESHOLDS[1:]):
            if np.any(selected_masks[higher] & ~selected_masks[lower]):
                raise EvidenceError(f"hotset nesting failed: count>={higher} is not contained in count>={lower}")
            if threshold_base[higher]["selected_unique_row_ids"] > threshold_base[lower]["selected_unique_row_ids"]:
                raise EvidenceError("hotset selected-unique monotonicity failed")
            if threshold_base[higher]["selected_exposures"] > threshold_base[lower]["selected_exposures"]:
                raise EvidenceError("hotset selected-exposure monotonicity failed")

        # Pass 2: exact node/step demand and hot coverage.  No row identifiers
        # are emitted; node_seen is only used for per-node storage sizing.
        step_total = np.zeros((FORMAL_STOP_STEP, NODE_COUNT), dtype=np.uint16)
        step_hot = np.zeros((len(HOT_THRESHOLDS), FORMAL_STOP_STEP, NODE_COUNT), dtype=np.uint16)
        node_seen = np.zeros((NODE_COUNT, gpic_size), dtype=np.bool_)
        domain_exposures_pass2 = np.zeros(len(config.domains), dtype=np.int64)
        for block_index in range(used_block_count):
            _bucket, items, domain_counts = plan._load_block(block_index)  # noqa: SLF001
            global_start = block_index * config.block_steps
            take = min(int(items.shape[0]), FORMAL_STOP_STEP - global_start)
            formal_items = items[:take]
            formal_counts = domain_counts[:take]
            domain_exposures_pass2 += formal_counts.sum(axis=0, dtype=np.int64)
            mask = (formal_items >= gpic_lo) & (formal_items < gpic_hi)
            reshaped_mask = mask.reshape(take, NODE_COUNT, RANKS_PER_NODE, EXPECTED_PER_RANK_BS)
            step_total[global_start : global_start + take] = reshaped_mask.sum(axis=(2, 3), dtype=np.uint16)
            all_ids = np.ascontiguousarray(formal_items[mask] - gpic_lo, dtype=np.int64)
            occurrence_counts = counts[all_ids]
            for node in range(NODE_COUNT):
                rank_lo = node * RANKS_PER_NODE
                rank_hi = rank_lo + RANKS_PER_NODE
                node_mask = mask[:, rank_lo:rank_hi, :]
                node_ids = formal_items[:, rank_lo:rank_hi, :][node_mask] - gpic_lo
                if node_ids.size:
                    node_seen[node, node_ids] = True
            for threshold_index, threshold in enumerate(HOT_THRESHOLDS):
                hot_occurrence = np.zeros(mask.shape, dtype=np.bool_)
                hot_occurrence[mask] = occurrence_counts >= threshold
                reshaped_hot = hot_occurrence.reshape(
                    take, NODE_COUNT, RANKS_PER_NODE, EXPECTED_PER_RANK_BS
                )
                step_hot[threshold_index, global_start : global_start + take] = reshaped_hot.sum(
                    axis=(2, 3), dtype=np.uint16
                )
                del hot_occurrence, reshaped_hot
            del _bucket, items, domain_counts, formal_items, formal_counts, mask, reshaped_mask
            del all_ids, occurrence_counts, node_mask, node_ids
            _clear_plan_cache(plan)

        if not np.array_equal(domain_exposures_pass1, domain_exposures_pass2):
            raise EvidenceError("two plan passes disagree on domain exposures")
        if int(step_total.sum(dtype=np.int64)) != EXPECTED_FORMAL_GPIC_EXPOSURES:
            raise EvidenceError("second-pass node/step GPIC exposure mismatch")
        for threshold_index, threshold in enumerate(HOT_THRESHOLDS):
            if int(step_hot[threshold_index].sum(dtype=np.int64)) != threshold_base[threshold]["selected_exposures"]:
                raise EvidenceError(f"second-pass hot exposure mismatch for threshold {threshold}")

        threshold_results = []
        for threshold_index, threshold in enumerate(HOT_THRESHOLDS):
            per_node = []
            horizon_arrays: dict[int, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]] = {
                horizon: ([], [], [], []) for horizon in PREFETCH_HORIZON_STEPS
            }
            for node in range(NODE_COUNT):
                node_total = int(step_total[:, node].sum(dtype=np.int64))
                node_hot = int(step_hot[threshold_index, :, node].sum(dtype=np.int64))
                node_selected_unique = int(np.count_nonzero(node_seen[node] & selected_masks[threshold]))
                node_record = {
                    "node": node,
                    "global_ranks": list(range(node * RANKS_PER_NODE, (node + 1) * RANKS_PER_NODE)),
                    "full_formal_prefix": {
                        "gpic_exposures": node_total,
                        "hot_exposures": node_hot,
                        "cold_source_exposures": node_total - node_hot,
                        **_capacity_record(
                            node_hot,
                            node_total,
                            nominal_demand_numerator=EXPECTED_FORMAL_GPIC_EXPOSURES,
                            nominal_demand_denominator=NODE_COUNT,
                        ),
                    },
                    "selected_unique_row_ids_needed_by_this_node": node_selected_unique,
                    "selected_node_storage_estimates": _storage_estimates(node_selected_unique, args.storage_bytes),
                    "horizons": [],
                }
                for horizon in PREFETCH_HORIZON_STEPS:
                    starts, totals = _window_sums(step_total[:, node], horizon)
                    hot_starts, hot_values = _window_sums(step_hot[threshold_index, :, node], horizon)
                    if not np.array_equal(starts, hot_starts):
                        raise EvidenceError("hot/total horizon starts disagree")
                    nodes = np.full(starts.shape, node, dtype=np.int64)
                    distribution = _coverage_distribution(
                        totals=totals,
                        hot=hot_values,
                        nodes=nodes,
                        starts=starts,
                        threshold=threshold,
                        horizon=horizon,
                    )
                    distribution.update(
                        {
                            "horizon_steps": horizon,
                            "anchor_stride_steps": WINDOW_STRIDE_STEPS,
                            "full_windows_only": True,
                            "last_included_start_step": int(starts[-1]),
                            "formal_tail_steps_not_covered_by_last_full_window": FORMAL_STOP_STEP - int(starts[-1] + horizon),
                        }
                    )
                    node_record["horizons"].append(distribution)
                    totals_list, hot_list, nodes_list, starts_list = horizon_arrays[horizon]
                    totals_list.append(totals)
                    hot_list.append(hot_values)
                    nodes_list.append(nodes)
                    starts_list.append(starts)
                per_node.append(node_record)

            aggregate_horizons = []
            for horizon in PREFETCH_HORIZON_STEPS:
                totals_list, hot_list, nodes_list, starts_list = horizon_arrays[horizon]
                distribution = _coverage_distribution(
                    totals=np.concatenate(totals_list),
                    hot=np.concatenate(hot_list),
                    nodes=np.concatenate(nodes_list),
                    starts=np.concatenate(starts_list),
                    threshold=threshold,
                    horizon=horizon,
                )
                distribution.update(
                    {
                        "horizon_steps": horizon,
                        "anchor_stride_steps": WINDOW_STRIDE_STEPS,
                        "population_scope": "all_8_nodes_each_window_equal_weight",
                    }
                )
                aggregate_horizons.append(distribution)
            result = dict(threshold_base[threshold])
            result["global_full_formal_prefix_capacity"] = _capacity_record(
                int(result["selected_exposures"]),
                EXPECTED_FORMAL_GPIC_EXPOSURES,
                nominal_demand_numerator=EXPECTED_FORMAL_GPIC_EXPOSURES,
                nominal_demand_denominator=1,
            )
            result["per_node"] = per_node
            result["aggregate_node_windows"] = aggregate_horizons
            threshold_results.append(result)

        global_node_union = np.logical_or.reduce(node_seen, axis=0)
        global_exposed = counts > 0
        if not np.array_equal(global_node_union, global_exposed):
            raise EvidenceError("second-pass node union disagrees with first-pass exposed-row set")
        del global_node_union, global_exposed, node_seen

        blocks_root = plan_root / "blocks"
        for name, before in block_snapshots.items():
            if _fingerprint(blocks_root / name) != before:
                raise EvidenceError(f"plan block stat changed during analysis: {name}")
        if _fingerprint(gpic_catalog_path) != gpic_catalog_fingerprint:
            raise EvidenceError("GPIC catalog parquet stat changed during analysis")
        hashes_after = {name: _sha256_file(path) for name, path in source_paths.items()}
        if hashes_after != hashes_before:
            raise EvidenceError("bound source/metadata files changed during analysis")

        histogram_rows = [
            {"exposure_count": count, "gpic_row_ids": int(histogram[count])}
            for count in range(EXPECTED_GPIC_R_MAX + 1)
        ]
        # Publish the reusable count>=6 bitset before the analysis JSON.  A
        # failed/partial artifact publication can therefore never leave a
        # PASS JSON that points at absent bytes.  The enclosing directory,
        # bitmap, manifest, and final JSON are all no-clobber publications.
        artifact_dir = _create_exclusive_directory(artifact_dir)
        bitmap_path = artifact_dir / "gpic_count_ge_6.packbits"
        bitmap_sha = _atomic_publish_bytes(bitmap_path, reusable_bitmap_payload)
        reusable_record = threshold_base[REUSABLE_BITMAP_THRESHOLD]
        if bitmap_sha != reusable_record["selected_bitmap_sha256"]:
            raise EvidenceError("published reusable bitmap SHA disagrees with in-memory threshold evidence")
        bitmap_manifest = {
            "schema_version": "offline_arb_gpic_hotset_bitmap_v1",
            "status": "PASS",
            "decision_scope": "reusable_builder_input_not_a_FORMAL_promotion_authority",
            "selection": {
                "target_domain": TARGET_DOMAIN,
                "formal_plan_id": EXPECTED_PLAN_ID,
                "formal_steps": {"start_inclusive": 0, "stop_exclusive": FORMAL_STOP_STEP},
                "count_threshold_operator": ">=",
                "count_threshold": REUSABLE_BITMAP_THRESHOLD,
                "selected_unique_row_ids": reusable_record["selected_unique_row_ids"],
                "selected_exposures": reusable_record["selected_exposures"],
            },
            "row_id_namespace": {
                "name": "gpic_catalog_local_row_id_zero_based",
                "local_row_id_base": 0,
                "row_count": EXPECTED_GPIC_ROWS,
                "catalog_global_row_base": gpic_lo,
                "catalog_global_row_stop_exclusive": gpic_hi,
            },
            "bitmap": {
                "file_name": bitmap_path.name,
                "encoding": "numpy.packbits",
                "bitorder": "little",
                "bit_index": "gpic_catalog_local_row_id",
                "selected_bit_value": 1,
                "bits": EXPECTED_GPIC_ROWS,
                "padding_bits": 0,
                "bytes": len(reusable_bitmap_payload),
                "sha256": bitmap_sha,
            },
            "input_binding": {
                "catalog_header_sha256": hashes_before["catalog_header"],
                "catalog_hash": catalog_header.get("catalog_hash"),
                "gpic_catalog_parquet_sha256_from_freeze": binding["catalog"]["gpic_parquet"]["sha256_from_frozen_catalog"],
                "plan_header_sha256": hashes_before["plan_header"],
                "plan_checksums_sha256": hashes_before["plan_checksums"],
                "freeze_sha256": hashes_before["freeze"],
                "producer_script_sha256": hashes_before["script"],
                "producer_job_yaml_sha256": hashes_before["job_yaml"],
            },
            "sample_paths_emitted": 0,
        }
        bitmap_manifest_path = artifact_dir / "manifest.json"
        bitmap_manifest_sha = _atomic_publish_json(bitmap_manifest_path, bitmap_manifest)
        artifact_binding = {
            "directory": str(artifact_dir),
            "directory_created_no_clobber": True,
            "bitmap": {
                "path": str(bitmap_path),
                "bytes": len(reusable_bitmap_payload),
                "sha256": bitmap_sha,
            },
            "manifest": {
                "path": str(bitmap_manifest_path),
                "bytes": bitmap_manifest_path.stat().st_size,
                "sha256": bitmap_manifest_sha,
                "schema_version": bitmap_manifest["schema_version"],
            },
        }
        sorted_bitmap_records = sorted(bitmap_records, key=lambda row: int(row["threshold"]))
        result_document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "decision_scope": "diagnostic_only_not_a_FORMAL_promotion_authority",
            "scope": {
                "formal_steps": {"start_inclusive": 0, "stop_exclusive": FORMAL_STOP_STEP, "count": FORMAL_STOP_STEP},
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
                "image_payload_reads": 0,
                "catalog_parquet_payload_reads": 0,
                "sample_paths_read": 0,
                "sample_paths_emitted": 0,
            },
            "model_contract": {
                "hotset_definition": "global GPIC local row_ids whose exact formal exposure count is >= threshold",
                "hotset_residency_assumption": "all selected objects are resident before demand and remain resident",
                "cold_source_assumption": "every non-selected exposure consumes one source service; transient cold reuse is intentionally not credited",
                "prefetch_horizon_definition": "full forward windows anchored every strict 4 steps; horizon aggregates absolute forward demand but creates no capacity and does not erase burst demand",
                "horizons_steps": list(PREFETCH_HORIZON_STEPS),
                "window_stride_steps": WINDOW_STRIDE_STEPS,
                "baseline_capacity_ratio": "0.787",
                "node_capacity_allocation_assumption": "the measured aggregate baseline is interpreted as 8 homogeneous nodes with equal per-node source capacity; node-window gates are invalid if capacity is materially asymmetric",
                "fixed_nominal_average_gpic_exposures_per_node_step": _positive_ratio(
                    EXPECTED_FORMAL_GPIC_EXPOSURES, FORMAL_STOP_STEP * NODE_COUNT
                ),
                "adjusted_capacity_formula": "0.787 * fixed nominal-average GPIC demand for interval / observed absolute cold-source demand",
                "capacity_gates": {name: format(num / den, ".3f") for name, num, den in CAPACITY_GATES},
            },
            "limitations": {
                "storage_estimates": "object-count times assumed bytes; filesystem/cache metadata and allocator overhead are excluded",
                "central_object_bytes": CENTRAL_OBJECT_BYTES,
                "storage_byte_sensitivity": list(args.storage_bytes),
                "local_capacity_sensitivity_gib": list(LOCAL_CAPACITY_GIB),
                "not_measured": [
                    "actual object bytes by selected row",
                    "image decoding",
                    "cache admission or eviction",
                    "filesystem metadata overhead",
                    "rank wall-clock interleaving",
                    "source request coalescing",
                ],
            },
            "inputs": {
                "plan_root": str(plan_root),
                "plan_header": {"path": str(header_path), "sha256": hashes_before["plan_header"], "bytes": header_path.stat().st_size},
                "plan_checksums": {"path": str(checksums_path), "sha256": hashes_before["plan_checksums"], "bytes": checksums_path.stat().st_size},
                "all_plan_blocks_sha256_verified": True,
                "all_plan_block_count": EXPECTED_PLAN_BLOCKS,
                "formal_array_blocks_streamed_twice": used_block_count,
                "formal_array_block_index_range_inclusive": [0, used_last_block],
                "nonformal_blocks_sha_only_verified": EXPECTED_PLAN_BLOCKS - used_block_count,
                "maximum_materialized_plan_blocks_per_pass": 1,
                "freeze_and_catalog_binding": binding,
                "source_files": {
                    name: {"path": str(source_paths[name]), "sha256": hashes_before[name]}
                    for name in ("script", "planner_source", "job_yaml")
                },
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
            "formal_prefix_domain_exposures": {
                domain: int(domain_exposures_pass1[index]) for index, domain in enumerate(config.domains)
            },
            "gpic_count_histogram": {
                "population_sum_expected_20m": int(histogram.sum(dtype=np.int64)),
                "weighted_exposure_sum_expected_99456552": int(
                    np.dot(np.arange(EXPECTED_GPIC_R_MAX + 1, dtype=np.int64), histogram.astype(np.int64))
                ),
                "maximum_observed_count": maximum_count,
                "R_max": EXPECTED_GPIC_R_MAX,
                "rows": histogram_rows,
            },
            "selected_bitmap_set_digest_contract": "canonical_json_sorted_threshold_order(threshold,sha256,bytes)",
            "selected_bitmap_set_digest_sha256": _sha256_bytes(_canonical_bytes(sorted_bitmap_records)),
            "reusable_count_ge_6_bitmap_artifact": artifact_binding,
            "hotsets": threshold_results,
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
    parser.add_argument("--artifact-dir", type=Path)
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
            name for name in ("plan_root", "freeze", "catalog_header", "job_yaml", "output", "artifact_dir")
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
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": str(args.output),
                    "output_sha256": output_sha,
                    "gpic_exposures": result["scope"]["formal_gpic_exposures"],
                    "threshold_selected_unique": {
                        str(row["count_threshold_inclusive"]): row["selected_unique_row_ids"]
                        for row in result["hotsets"]
                    },
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
