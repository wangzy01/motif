"""Weighted, caption-band-bound Stage512a offline plan materializer.

This module deliberately reuses the reviewed integer domain/AR quota solver
from ``planner_v1`` and replaces only its uniform cell row enumeration.  The
large production build is JCC-only; fixture tests may call the pure helpers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from training.dataline_offline import planner_v1 as base
from training.dataline_offline.weighted_streams_v1 import (
    ALGORITHM as WEIGHTED_STREAM_ALGORITHM,
    BandBoundCellStream,
    MutableSequenceStream,
    WeightedStreamError,
    anime_repeat_sequence,
    collision_reserve_size,
    deterministic_band_schedule,
    segment_label_counts,
    systematic_weighted_sequence,
)


PLAN_FORMAT = "offline_arb_weighted_band_plan_v1"
PLANNER_ALGORITHM_VERSION = "offline_arb_weighted_band_planner_v1_20260722"
CONTRACT_SCHEMA = "stage512a_weighted_plan_contract_v1"
BLOCK_DATASETS = (
    "bucket_of_step",
    "item_row",
    "caption_band",
    "domain_counts",
)
REAL_DOMAINS = frozenset({"monet_real", "gpic"})
EXPECTED_DOMAINS = ("anime", "monet_real", "gpic", "synthetic", "textatlas")
EXPECTED_STREAM_KINDS = {
    "anime": "anime_repeat_v1",
    "monet_real": "positive_base_weight_band_bound_v1",
    "gpic": "positive_base_weight_band_bound_v1",
    "synthetic": "positive_base_weight_full_pool_v1",
    "textatlas": "uniform_cycle_v1",
}


class WeightedPlanError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, nlink_one: bool = True) -> os.stat_result:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or (nlink_one and st.st_nlink != 1):
        raise WeightedPlanError(f"unsafe input file: {path}")
    return st


def _read_json(path: Path) -> dict[str, Any]:
    _regular_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise WeightedPlanError(f"cannot parse JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WeightedPlanError(f"JSON root is not an object: {path}")
    return value


def _lower_sha(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise WeightedPlanError(f"{path} is not a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class WeightedContract:
    raw: dict[str, Any]
    source_path: Path
    source_sha256: str
    catalog_root: Path
    catalog_hash: str
    catalog_header_sha256: str
    anime_priority_root: Path
    anime_priority_header_sha256: str
    anime_priority_parquet_sha256: str
    sampling_config_sha256: str
    band_weights: tuple[Decimal, Decimal, Decimal, Decimal]
    anime_max_repeat: int
    anime_extra_frac: float
    stream_kinds: dict[str, str]


def load_weighted_contract(path: str | os.PathLike[str]) -> WeightedContract:
    source = Path(path).resolve(strict=True)
    raw = _read_json(source)
    required = {
        "schema",
        "stage",
        "image_size",
        "catalog_root",
        "catalog_hash",
        "catalog_header_sha256",
        "anime_priority_root",
        "anime_priority_header_sha256",
        "anime_priority_parquet_sha256",
        "sampling_config_sha256",
        "band_weights",
        "anime_max_repeat",
        "anime_extra_frac",
        "stream_kinds",
    }
    if set(raw) != required:
        raise WeightedPlanError(
            f"weighted contract key-set drift: missing={sorted(required-set(raw))}, "
            f"extra={sorted(set(raw)-required)}"
        )
    if raw["schema"] != CONTRACT_SCHEMA or raw["stage"] != "512a" or raw["image_size"] != 512:
        raise WeightedPlanError("weighted contract stage/schema mismatch")
    catalog_root = Path(str(raw["catalog_root"])).resolve(strict=True)
    anime_root = Path(str(raw["anime_priority_root"])).resolve(strict=True)
    if not catalog_root.is_dir() or catalog_root.is_symlink():
        raise WeightedPlanError("unsafe catalog root")
    if not anime_root.is_dir() or anime_root.is_symlink():
        raise WeightedPlanError("unsafe Anime priority root")
    weights_raw = raw["band_weights"]
    if not isinstance(weights_raw, list) or len(weights_raw) != 4:
        raise WeightedPlanError("band_weights must be a four-element array")
    try:
        band_weights = tuple(Decimal(str(value)) for value in weights_raw)
    except Exception as exc:
        raise WeightedPlanError("invalid band_weights") from exc
    if any(not value.is_finite() or value < 0 for value in band_weights):
        raise WeightedPlanError("band_weights must be finite/non-negative")
    if sum(band_weights, Decimal(0)) != Decimal(1):
        raise WeightedPlanError("band_weights must sum exactly to one")
    max_repeat = raw["anime_max_repeat"]
    extra_frac = raw["anime_extra_frac"]
    if isinstance(max_repeat, bool) or not isinstance(max_repeat, int) or max_repeat < 1:
        raise WeightedPlanError("invalid anime_max_repeat")
    if isinstance(extra_frac, bool) or not isinstance(extra_frac, (int, float)):
        raise WeightedPlanError("invalid anime_extra_frac")
    extra_frac_f = float(extra_frac)
    if not np.isfinite(extra_frac_f) or not (0 <= extra_frac_f <= 1):
        raise WeightedPlanError("anime_extra_frac outside [0,1]")
    stream_kinds = raw["stream_kinds"]
    if stream_kinds != EXPECTED_STREAM_KINDS:
        raise WeightedPlanError(
            f"stream_kinds mismatch: {stream_kinds!r} != {EXPECTED_STREAM_KINDS!r}"
        )
    result = WeightedContract(
        raw=raw,
        source_path=source,
        source_sha256=_sha256_file(source),
        catalog_root=catalog_root,
        catalog_hash=_lower_sha(raw["catalog_hash"], "catalog_hash"),
        catalog_header_sha256=_lower_sha(
            raw["catalog_header_sha256"], "catalog_header_sha256"
        ),
        anime_priority_root=anime_root,
        anime_priority_header_sha256=_lower_sha(
            raw["anime_priority_header_sha256"], "anime_priority_header_sha256"
        ),
        anime_priority_parquet_sha256=_lower_sha(
            raw["anime_priority_parquet_sha256"], "anime_priority_parquet_sha256"
        ),
        sampling_config_sha256=_lower_sha(
            raw["sampling_config_sha256"], "sampling_config_sha256"
        ),
        band_weights=band_weights,  # type: ignore[arg-type]
        anime_max_repeat=max_repeat,
        anime_extra_frac=extra_frac_f,
        stream_kinds=dict(stream_kinds),
    )
    _validate_bound_inputs(result)
    return result


def _validate_bound_inputs(contract: WeightedContract) -> None:
    catalog_header_path = contract.catalog_root / "header.json"
    if _sha256_file(catalog_header_path) != contract.catalog_header_sha256:
        raise WeightedPlanError("catalog header SHA drift")
    catalog_header = _read_json(catalog_header_path)
    if catalog_header.get("catalog_hash") != contract.catalog_hash:
        raise WeightedPlanError("catalog hash drift")
    if catalog_header.get("stage") != "512a" or catalog_header.get("image_size") != 512:
        raise WeightedPlanError("catalog stage identity drift")
    files = catalog_header.get("files")
    if not isinstance(files, dict) or set(files) != set(EXPECTED_DOMAINS):
        raise WeightedPlanError("catalog file authority key-set drift")
    for domain in EXPECTED_DOMAINS:
        record = files[domain]
        if not isinstance(record, dict):
            raise WeightedPlanError(f"catalog file authority is not an object: {domain}")
        expected_name = f"{domain}.parquet"
        if record.get("name") != expected_name:
            raise WeightedPlanError(f"catalog file-name drift: {domain}")
        path = contract.catalog_root / expected_name
        st = _regular_file(path)
        if st.st_size != record.get("bytes"):
            raise WeightedPlanError(f"catalog file-size drift: {domain}")
        expected_sha = _lower_sha(record.get("sha256"), f"files.{domain}.sha256")
        if _sha256_file(path) != expected_sha:
            raise WeightedPlanError(f"catalog file SHA drift: {domain}")
    anime_header_path = contract.anime_priority_root / "header.json"
    if _sha256_file(anime_header_path) != contract.anime_priority_header_sha256:
        raise WeightedPlanError("Anime priority header SHA drift")
    anime_header = _read_json(anime_header_path)
    success = _read_json(contract.anime_priority_root / "_SUCCESS")
    parquet_path = contract.anime_priority_root / "anime_priority.parquet"
    _regular_file(parquet_path)
    if anime_header.get("status") != "PASS" or success.get("status") != "PASS":
        raise WeightedPlanError("Anime priority sidecar is not committed PASS")
    if anime_header.get("catalog", {}).get("catalog_hash") != contract.catalog_hash:
        raise WeightedPlanError("Anime priority/catalog binding drift")
    if anime_header.get("sampling", {}).get("sha256") != contract.sampling_config_sha256:
        raise WeightedPlanError("Anime priority/sampling binding drift")
    if anime_header.get("output", {}).get("sha256") != contract.anime_priority_parquet_sha256:
        raise WeightedPlanError("Anime priority header parquet SHA drift")
    if _sha256_file(parquet_path) != contract.anime_priority_parquet_sha256:
        raise WeightedPlanError("Anime priority parquet SHA drift")


def _load_parquet_columns(path: Path, columns: Sequence[str]) -> dict[str, np.ndarray]:
    _regular_file(path)
    pf = pq.ParquetFile(path)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in columns}
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=list(columns), use_threads=False)
        for name in columns:
            parts[name].append(np.asarray(table[name].to_numpy(zero_copy_only=False)))
    return {
        name: np.concatenate(chunks) if chunks else np.empty(0)
        for name, chunks in parts.items()
    }


class _PlainCellStream:
    def __init__(self, stream: Any):
        self.stream = stream

    def take(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        rows = self.stream.take(count)
        return rows, np.full(count, -1, dtype=np.int8)


@dataclass
class _PreparedStreams:
    cells: dict[tuple[int, int], Any]
    mutable_streams: list[MutableSequenceStream]
    source_stats: dict[str, Any]


def _validate_domain_rows(
    cfg: base.PlannerConfig,
    domain_index: int,
    row_id: np.ndarray,
    bucket: np.ndarray,
) -> None:
    if row_id.dtype != np.dtype(np.int64):
        row_id = row_id.astype(np.int64, copy=False)
    if bucket.shape != row_id.shape or np.any(bucket < 0) or np.any(bucket >= cfg.B):
        raise WeightedPlanError("catalog row/bucket shape or range drift")
    if len(row_id) and np.any(row_id[1:] <= row_id[:-1]):
        raise WeightedPlanError("catalog row IDs are not strictly increasing")
    row_bases = base._cell_row_bases(cfg)
    for b in range(cfg.B):
        rows = row_id[bucket == b]
        expected_n = int(cfg.N[b, domain_index])
        if len(rows) != expected_n:
            raise WeightedPlanError(
                f"catalog N mismatch b={b},d={domain_index}: {len(rows)} != {expected_n}"
            )
        if expected_n:
            expected = np.arange(
                int(row_bases[b, domain_index]),
                int(row_bases[b, domain_index]) + expected_n,
                dtype=np.int64,
            )
            if not np.array_equal(rows, expected):
                raise WeightedPlanError(f"catalog virtual row mapping drift b={b},d={domain_index}")


def _build_streams(
    cfg: base.PlannerConfig,
    contract: WeightedContract,
    C: np.ndarray,
    per_bucket_quotas: Sequence[np.ndarray],
) -> _PreparedStreams:
    if tuple(cfg.domains) != EXPECTED_DOMAINS:
        raise WeightedPlanError(f"unexpected domain order: {cfg.domains!r}")
    if cfg.catalog_hash != contract.catalog_hash:
        raise WeightedPlanError("planner input/catalog contract hash mismatch")
    cells: dict[tuple[int, int], Any] = {}
    mutables: list[MutableSequenceStream] = []
    stats: dict[str, Any] = {"domains": {}}
    row_bases = base._cell_row_bases(cfg)

    # Anime priority sidecar is already catalog-row and bucket bound.
    d = cfg.domains.index("anime")
    anime = _load_parquet_columns(
        contract.anime_priority_root / "anime_priority.parquet",
        ("row_id", "bucket_idx", "priority"),
    )
    anime_rows = anime["row_id"].astype(np.int64, copy=False)
    anime_bucket = anime["bucket_idx"].astype(np.int16, copy=False)
    anime_priority = anime["priority"].astype(np.float64, copy=False)
    _validate_domain_rows(cfg, d, anime_rows, anime_bucket)
    anime_stats: dict[str, Any] = {"cells": {}}
    for b in range(cfg.B):
        total = int(C[b, d])
        if not total:
            continue
        mask = anime_bucket == b
        rows = anime_rows[mask]
        priority = anime_priority[mask]
        max_take = int(per_bucket_quotas[b][:, d].max(initial=0))
        reserve = collision_reserve_size(total, max_take)
        sequence = anime_repeat_sequence(
            rows,
            priority,
            total,
            reserve=reserve,
            max_repeat=contract.anime_max_repeat,
            extra_frac=contract.anime_extra_frac,
            plan_id=cfg.plan_id,
            seed=cfg.seed,
            label="anime",
            parts=(b, d),
        )
        mutable = MutableSequenceStream(sequence, limit=total)
        mutables.append(mutable)
        cells[(b, d)] = _PlainCellStream(mutable)
        anime_stats["cells"][str(b)] = {
            "rows": int(len(rows)),
            "exposures": total,
            "reserve": reserve,
            "max_batch_take": max_take,
        }
    stats["domains"]["anime"] = anime_stats
    del anime, anime_rows, anime_bucket, anime_priority

    for domain in ("monet_real", "gpic"):
        d = cfg.domains.index(domain)
        arrays = _load_parquet_columns(
            contract.catalog_root / f"{domain}.parquet",
            (
                "row_id",
                "bucket_idx",
                "base_weight",
                "served_mask",
                "band_captioner_t1",
                "band_captioner_t2",
                "band_captioner_t3",
                "band_captioner_t4",
            ),
        )
        rows_all = arrays["row_id"].astype(np.int64, copy=False)
        buckets_all = arrays["bucket_idx"].astype(np.int16, copy=False)
        weights_all = arrays["base_weight"].astype(np.float64, copy=False)
        served_all = arrays["served_mask"].astype(np.uint8, copy=False)
        codes_all = np.column_stack(
            [arrays[f"band_captioner_t{i}"].astype(np.int8, copy=False) for i in range(1, 5)]
        )
        _validate_domain_rows(cfg, d, rows_all, buckets_all)
        if not np.all(np.isfinite(weights_all)) or np.any(weights_all < 0):
            raise WeightedPlanError(f"{domain} has invalid base_weight")
        domain_stats: dict[str, Any] = {"cells": {}}
        for b in range(cfg.B):
            total = int(C[b, d])
            if not total:
                continue
            cell = buckets_all == b
            rows = rows_all[cell]
            weights = weights_all[cell]
            served = served_all[cell]
            codes = codes_all[cell]
            positive = weights > 0
            valid_any = positive & np.any(
                ((((served[:, None] >> np.arange(4, dtype=np.uint8)) & 1) != 0) & (codes > 0)),
                axis=1,
            )
            max_take = int(per_bucket_quotas[b][:, d].max(initial=0))
            if int(np.count_nonzero(valid_any)) < max_take:
                raise WeightedPlanError(
                    f"{domain} b={b} cannot provide a unique positive served batch: "
                    f"{np.count_nonzero(valid_any)} < {max_take}"
                )
            band_masks = [
                positive & (((served >> band) & 1) != 0) & (codes[:, band] > 0)
                for band in range(4)
            ]
            available = [bool(np.any(mask)) for mask in band_masks]
            schedule, band_quotas = deterministic_band_schedule(
                total,
                contract.band_weights,
                available,
                plan_id=cfg.plan_id,
                seed=cfg.seed,
                label=domain,
                parts=(b, d),
            )
            step_counts = per_bucket_quotas[b][:, d].astype(np.int64, copy=False)
            per_step_band_counts = segment_label_counts(schedule, step_counts, 4)
            band_max_take = per_step_band_counts.max(axis=0, initial=0)
            for subset_bits in range(1, 1 << 4):
                subset = [band for band in range(4) if (subset_bits >> band) & 1]
                max_demand = int(
                    per_step_band_counts[:, subset].sum(axis=1, dtype=np.int64).max(initial=0)
                )
                eligible_union = np.zeros(len(rows), dtype=bool)
                for band in subset:
                    eligible_union |= band_masks[band]
                capacity = int(np.count_nonzero(eligible_union))
                if capacity < max_demand:
                    raise WeightedPlanError(
                        f"{domain} b={b} violates band Hall capacity for {subset}: "
                        f"{capacity} < {max_demand}"
                    )
            band_streams: dict[int, MutableSequenceStream] = {}
            band_stats: dict[str, Any] = {}
            for band in range(4):
                quota = int(band_quotas[band])
                if not quota:
                    continue
                mask = band_masks[band]
                if int(np.count_nonzero(mask)) < int(band_max_take[band]):
                    raise WeightedPlanError(
                        f"{domain} b={b},band={band} cannot provide a unique batch: "
                        f"{np.count_nonzero(mask)} < {int(band_max_take[band])}"
                    )
                reserve = collision_reserve_size(quota, int(band_max_take[band]))
                sequence = systematic_weighted_sequence(
                    rows[mask],
                    weights[mask],
                    quota,
                    reserve=reserve,
                    plan_id=cfg.plan_id,
                    seed=cfg.seed,
                    label=f"{domain}:band{band}",
                    parts=(b, d, band),
                )
                mutable = MutableSequenceStream(sequence, limit=quota)
                mutables.append(mutable)
                band_streams[band] = mutable
                band_stats[str(band)] = {
                    "eligible_positive_rows": int(np.count_nonzero(mask)),
                    "positive_weight_sum": float(weights[mask].sum(dtype=np.float64)),
                    "exposures": quota,
                    "reserve": reserve,
                    "max_batch_take": int(band_max_take[band]),
                }
            band_order = sorted(
                band_streams,
                key=lambda band: (int(np.count_nonzero(band_masks[band])), band),
            )
            cells[(b, d)] = BandBoundCellStream(
                schedule, band_streams, band_order=band_order
            )
            domain_stats["cells"][str(b)] = {
                "rows": int(len(rows)),
                "positive_rows": int(np.count_nonzero(positive)),
                "positive_served_rows": int(np.count_nonzero(valid_any)),
                "zero_weight_rows": int(np.count_nonzero(weights == 0)),
                "exposures": total,
                "max_batch_take": max_take,
                "band_quotas": band_quotas.tolist(),
                "band_greedy_order": band_order,
                "bands": band_stats,
            }
        stats["domains"][domain] = domain_stats
        del arrays, rows_all, buckets_all, weights_all, served_all, codes_all

    # Synthetic follows the streaming full-pool base_weight rule.
    domain = "synthetic"
    d = cfg.domains.index(domain)
    synthetic = _load_parquet_columns(
        contract.catalog_root / "synthetic.parquet",
        ("row_id", "bucket_idx", "base_weight"),
    )
    rows_all = synthetic["row_id"].astype(np.int64, copy=False)
    buckets_all = synthetic["bucket_idx"].astype(np.int16, copy=False)
    weights_all = synthetic["base_weight"].astype(np.float64, copy=False)
    _validate_domain_rows(cfg, d, rows_all, buckets_all)
    if not np.all(np.isfinite(weights_all)) or np.any(weights_all < 0):
        raise WeightedPlanError("Synthetic has invalid base_weight")
    synthetic_stats: dict[str, Any] = {"cells": {}}
    for b in range(cfg.B):
        total = int(C[b, d])
        if not total:
            continue
        cell = buckets_all == b
        positive = cell & (weights_all > 0)
        max_take = int(per_bucket_quotas[b][:, d].max(initial=0))
        if int(np.count_nonzero(positive)) < max_take:
            raise WeightedPlanError("Synthetic positive-weight cell violates batch uniqueness")
        reserve = collision_reserve_size(total, max_take)
        sequence = systematic_weighted_sequence(
            rows_all[positive],
            weights_all[positive],
            total,
            reserve=reserve,
            plan_id=cfg.plan_id,
            seed=cfg.seed,
            label="synthetic",
            parts=(b, d),
        )
        mutable = MutableSequenceStream(sequence, limit=total)
        mutables.append(mutable)
        cells[(b, d)] = _PlainCellStream(mutable)
        synthetic_stats["cells"][str(b)] = {
            "rows": int(np.count_nonzero(cell)),
            "positive_rows": int(np.count_nonzero(positive)),
            "zero_weight_rows": int(np.count_nonzero(cell & (weights_all == 0))),
            "exposures": total,
            "reserve": reserve,
            "max_batch_take": max_take,
        }
    stats["domains"][domain] = synthetic_stats
    del synthetic, rows_all, buckets_all, weights_all

    # TextAtlas remains uniform and square-only; reuse the reviewed PRP stream.
    d = cfg.domains.index("textatlas")
    base_stats = {"feistel_cycles_created": 0, "cycle_boundary_swaps": 0}
    text_stats: dict[str, Any] = {"cells": {}}
    for b in range(cfg.B):
        total = int(C[b, d])
        if not total:
            continue
        stream = base._CellStream(cfg, b, d, int(row_bases[b, d]), base_stats)
        cells[(b, d)] = _PlainCellStream(stream)
        text_stats["cells"][str(b)] = {
            "rows": int(cfg.N[b, d]),
            "exposures": total,
        }
    text_stats["uniform_feistel_stats"] = base_stats
    stats["domains"]["textatlas"] = text_stats
    return _PreparedStreams(cells, mutables, stats)


def _materialize_blocks(
    output: Path,
    cfg: base.PlannerConfig,
    D: np.ndarray,
    S: np.ndarray,
    C: np.ndarray,
    schedule: np.ndarray,
    per_bucket_quotas: Sequence[np.ndarray],
    start_step: int,
    prepared: _PreparedStreams,
) -> tuple[dict[str, str], dict[str, Any]]:
    blocks_dir = output / "blocks"
    blocks_dir.mkdir(mode=0o700)
    occurrence = np.zeros(cfg.B, dtype=np.int64)
    realized_domains = np.zeros(cfg.K, dtype=np.int64)
    realized_bands = np.zeros(4, dtype=np.int64)
    checksums: dict[str, str] = {}
    num_blocks = (cfg.total_steps + cfg.block_steps - 1) // cfg.block_steps
    first_block_index = start_step // cfg.block_steps
    unique_checks = 0
    rank_checks = 0
    for block_idx in range(num_blocks):
        start = block_idx * cfg.block_steps
        stop = min(cfg.total_steps, start + cfg.block_steps)
        count = stop - start
        write_block = block_idx >= first_block_index
        if write_block:
            bucket_of_step = schedule[start:stop].astype(np.int16, copy=True)
            item_row = np.empty((count, cfg.world_size, cfg.per_rank_bs), dtype=np.int64)
            caption_band = np.empty((count, cfg.world_size, cfg.per_rank_bs), dtype=np.int8)
            domain_counts = np.empty((count, cfg.K), dtype=np.int32)
        for local_step, global_step in enumerate(range(start, stop)):
            b = int(schedule[global_step])
            bucket_occurrence = int(occurrence[b])
            quotas = per_bucket_quotas[b][bucket_occurrence]
            occurrence[b] += 1
            parts: list[np.ndarray] = []
            band_parts: list[np.ndarray] = []
            domain_parts: list[np.ndarray] = []
            for d in range(cfg.K):
                cell_count = int(quotas[d])
                if cell_count:
                    try:
                        rows, bands = prepared.cells[(b, d)].take(cell_count)
                    except (WeightedStreamError, KeyError) as exc:
                        raise WeightedPlanError(
                            f"cell stream failed at step={global_step},b={b},d={d}: {exc}"
                        ) from exc
                    parts.append(rows)
                    band_parts.append(bands)
                    domain_parts.append(np.full(cell_count, d, dtype=np.int32))
                    for band in range(4):
                        realized_bands[band] += int(np.count_nonzero(bands == band))
                realized_domains[d] += cell_count
            flat_rows = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
            flat_bands = np.concatenate(band_parts) if band_parts else np.empty(0, dtype=np.int8)
            flat_domains = np.concatenate(domain_parts) if domain_parts else np.empty(0, dtype=np.int32)
            if flat_rows.shape != (cfg.G,) or flat_bands.shape != (cfg.G,):
                raise WeightedPlanError(f"global batch shape drift at step {global_step}")
            if int(np.unique(flat_rows).size) != cfg.G:
                raise WeightedPlanError(f"duplicate global row at step {global_step}")
            unique_checks += 1
            dealt_domains = flat_domains.reshape(cfg.per_rank_bs, cfg.world_size).T
            for d in range(cfg.K):
                counts = np.count_nonzero(dealt_domains == d, axis=1)
                if int(counts.max()) - int(counts.min()) > 1:
                    raise WeightedPlanError(
                        f"rank domain imbalance step={global_step},domain={d}"
                    )
            rank_checks += 1
            if write_block:
                item_row[local_step] = flat_rows.reshape(cfg.per_rank_bs, cfg.world_size).T
                caption_band[local_step] = flat_bands.reshape(cfg.per_rank_bs, cfg.world_size).T
                domain_counts[local_step] = quotas
        if write_block:
            filename = f"block_{block_idx:06d}.npz"
            path = blocks_dir / filename
            base._write_deterministic_npz(
                path,
                (
                    ("bucket_of_step", bucket_of_step),
                    ("item_row", item_row),
                    ("caption_band", caption_band),
                    ("domain_counts", domain_counts),
                ),
            )
            checksums[filename] = base._file_sha256(path)
    if not np.array_equal(occurrence, S) or not np.array_equal(realized_domains, D):
        raise WeightedPlanError("materialized quota conservation drift")
    collision_swaps = sum(stream.collision_swaps for stream in prepared.mutable_streams)
    materialization = {
        "num_blocks": num_blocks,
        "materialized_num_blocks": len(checksums),
        "requested_start_step": start_step,
        "first_materialized_block": first_block_index,
        "first_materialized_step": first_block_index * cfg.block_steps,
        "realized_domain_exposures": realized_domains.tolist(),
        "realized_bucket_steps": occurrence.tolist(),
        "realized_real_band_exposures": realized_bands.tolist(),
        "global_batches_explicitly_checked_unique": unique_checks,
        "rank_balance_checks": rank_checks,
        "weighted_collision_swaps": collision_swaps,
        # Compatibility/accounting fields for the reused base header builder.
        "feistel_cycles_created": int(
            prepared.source_stats["domains"]["textatlas"]
            .get("uniform_feistel_stats", {})
            .get("feistel_cycles_created", 0)
        ),
        "cycle_boundary_swaps": int(
            prepared.source_stats["domains"]["textatlas"]
            .get("uniform_feistel_stats", {})
            .get("cycle_boundary_swaps", 0)
        ),
    }
    return checksums, materialization


def _weighted_header(
    cfg: base.PlannerConfig,
    contract: WeightedContract,
    D: np.ndarray,
    S: np.ndarray,
    C: np.ndarray,
    solver_record: Mapping[str, Any],
    per_step_repair_moves: int,
    checksums: Mapping[str, str],
    materialization: Mapping[str, Any],
    source_stats: Mapping[str, Any],
) -> dict[str, Any]:
    header = base._make_header(
        cfg,
        D,
        S,
        C,
        solver_record,
        per_step_repair_moves,
        checksums,
        materialization,
    )
    base_identity = str(header["plan_hash"])
    blocks_manifest = str(header["hashes"]["blocks_manifest_sha256"])
    producer_sources = {
        "weighted_plan_v1.py": _sha256_file(Path(__file__).resolve(strict=True)),
        "weighted_streams_v1.py": _sha256_file(
            Path(__file__).with_name("weighted_streams_v1.py").resolve(strict=True)
        ),
        "planner_v1.py": _sha256_file(Path(base.__file__).resolve(strict=True)),
    }
    weighted_binding = {
        "format": PLAN_FORMAT,
        "algorithm": PLANNER_ALGORITHM_VERSION,
        "weighted_streams": WEIGHTED_STREAM_ALGORITHM,
        "base_plan_identity_sha256": base_identity,
        "weighted_contract_sha256": contract.source_sha256,
        "catalog_header_sha256": contract.catalog_header_sha256,
        "anime_priority_header_sha256": contract.anime_priority_header_sha256,
        "anime_priority_parquet_sha256": contract.anime_priority_parquet_sha256,
        "blocks_manifest_sha256": blocks_manifest,
        "block_datasets": list(BLOCK_DATASETS),
        "producer_sources": producer_sources,
    }
    identity = _sha256_bytes(_canonical_bytes(weighted_binding))
    header.update(
        {
            "plan_format": PLAN_FORMAT,
            "planner_algo_version": PLANNER_ALGORITHM_VERSION,
            "algorithm_version": PLANNER_ALGORITHM_VERSION,
            "block_datasets": list(BLOCK_DATASETS),
            "base_plan_identity_sha256": base_identity,
            "weighted_contract": copy.deepcopy(contract.raw),
            "weighted_contract_sha256": contract.source_sha256,
            "weighted_binding": weighted_binding,
            "weighted_source_stats": copy.deepcopy(dict(source_stats)),
            "producer_sources": producer_sources,
            "plan_hash": identity,
        }
    )
    algorithms = dict(header["algorithm_versions"])
    algorithms.update(
        {
            "weighted_plan": PLANNER_ALGORITHM_VERSION,
            "weighted_streams": WEIGHTED_STREAM_ALGORITHM,
            "caption_band": "plan_bound_int8_v1",
            "real_band_allocation": "configured_weights_renorm_supported_largest_remainder_v1",
            "zero_weight": "strict_exclusion_v1",
        }
    )
    header["algorithm_versions"] = algorithms
    header["hashes"]["base_plan_identity_sha256"] = base_identity
    header["hashes"]["weighted_contract_sha256"] = contract.source_sha256
    header["hashes"]["plan_identity_sha256"] = identity
    header["hashes"]["planner_algo_version"] = PLANNER_ALGORITHM_VERSION
    header["config_hash"] = cfg.config_sha256
    return header


def build_weighted_plan(
    planner_input: str | os.PathLike[str],
    weighted_contract: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    start_step: int = 0,
    allow_canary: bool = False,
) -> dict[str, Any]:
    cfg = base.load_planner_input(planner_input)
    contract = load_weighted_contract(weighted_contract)
    if cfg.stage != "512a" or cfg.image_size != 512 or cfg.format_version != 2:
        raise WeightedPlanError("weighted planner requires Stage512a format-v2 input")
    formal = cfg.M == 256_000_000 and cfg.total_steps == 125_000 and cfg.G == 2048
    canary = (
        allow_canary
        and "canary" in cfg.plan_id
        and cfg.M <= 1_000_000
        and cfg.total_steps <= 10_000
        and cfg.world_size <= 16
    )
    if not formal and not canary:
        raise WeightedPlanError(
            f"formal horizon drift: M={cfg.M},steps={cfg.total_steps},G={cfg.G}"
        )
    if start_step < 0 or start_step >= cfg.total_steps:
        raise WeightedPlanError("start_step outside plan")
    output = Path(out_dir)
    staging = output.with_name(output.name + ".work")
    if output.exists() or output.is_symlink():
        raise WeightedPlanError(f"output root already exists: {output}")
    if staging.exists() or staging.is_symlink():
        raise WeightedPlanError(f"staging root already exists: {staging}")

    D, S, C, _Q, solver, _allowed, _caps = base._solve_quotas(cfg)
    schedule = base._bucket_schedule(cfg, S)
    per_bucket_quotas, repair_moves = base._per_step_domain_quotas(cfg, C, S)
    prepared = _build_streams(cfg, contract, C, per_bucket_quotas)

    output.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(staging, 0o700)
    checksums, materialization = _materialize_blocks(
        staging,
        cfg,
        D,
        S,
        C,
        schedule,
        per_bucket_quotas,
        start_step,
        prepared,
    )
    header = _weighted_header(
        cfg,
        contract,
        D,
        S,
        C,
        solver,
        repair_moves,
        checksums,
        materialization,
        prepared.source_stats,
    )
    base._write_json_atomic(staging / "header.json", header)
    base._write_json_atomic(
        staging / "checksums.json",
        {"header": base._file_sha256(staging / "header.json"), "blocks": checksums},
    )
    for path in (staging / "blocks").iterdir():
        os.chmod(path, 0o444)
    os.chmod(staging / "header.json", 0o444)
    os.chmod(staging / "checksums.json", 0o444)
    os.chmod(staging / "blocks", 0o555)
    os.chmod(staging, 0o555)

    # A single, deterministic sibling staging name is the exclusive writer
    # claim.  FSx for Lustre rejects directory RENAME_NOREPLACE, so after that
    # claim we use the POSIX same-filesystem rename: readers either see no
    # canonical root or the completely sealed tree.  Any pre-publication
    # failure deliberately leaves ``.work`` as evidence while keeping the
    # canonical path absent; callers must move to a fresh release identity.
    if output.exists() or output.is_symlink():
        raise WeightedPlanError(f"output root appeared before publish: {output}")
    staged_stat = staging.lstat()
    os.rename(staging, output)
    published_stat = output.lstat()
    if (published_stat.st_dev, published_stat.st_ino) != (
        staged_stat.st_dev,
        staged_stat.st_ino,
    ):
        raise WeightedPlanError("published plan inode drift")
    if stat.S_IMODE(published_stat.st_mode) != 0o555:
        raise WeightedPlanError("published plan root is not sealed 0555")
    return header


__all__ = [
    "BLOCK_DATASETS",
    "CONTRACT_SCHEMA",
    "PLAN_FORMAT",
    "PLANNER_ALGORITHM_VERSION",
    "WeightedContract",
    "WeightedPlanError",
    "build_weighted_plan",
    "load_weighted_contract",
]
