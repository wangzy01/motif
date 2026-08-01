#!/usr/bin/env python3
"""OFFLINE_ARB_V1 Phase-5: materialize the authoritative L1/L2 catalog.

The contract implemented here is docs/OFFLINE_ARB_PLANNER_DESIGN.md section 11:

* enumerate the five production manifest pools row-by-row;
* use the exact Phase-3a eligibility and native-bucket assignment semantics;
* fold the 95-index bucket namespace with ``bucket_prob_floor=0.005`` exactly as
  ``make_planner_input.py --floor 0.005`` (including unconditional square keep);
* materialize rows in ``(domain order, bucket_idx, uid lexicographic)`` order and
  make ``row_id`` the contiguous global physical row number;
* keep caption text out of the catalog.  Large caption fields are addressed by
  ``(parquet_path, row_group, row_in_group)`` locators.

The full build is deliberately external/partitioned.  A 64-process first pass
counts raw cells.  A second per-manifest-shard pass writes compact candidates;
real/synthetic candidates are hash-partitioned by source shard so every source
caption parquet and tar-index row group is scanned once.  Joined rows become
sorted cell runs.  The final merge visits one domain/cell at a time, never the
95M-row catalog as a single in-memory table.

Formal output (the only output root accepted):

  offline_arb_catalog_v1/
    anime.parquet, monet_real.parquet, gpic.parquet,
    synthetic.parquet, textatlas.parquet
    bad_rows.parquet
    N_diff.json
    header.json

With ``--max-shards-per-domain`` every product inserts ``.diag`` before its
extension and therefore cannot overwrite a formal build.

Exit 33 is reserved for a missing tar-index ``_SUCCESS`` marker, matching the
Phase-5 job preflight contract.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import socket
import struct
import sys
import time
import traceback
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from omegaconf import OmegaConf

# Direct ``python scripts/offline_arb/...py`` execution otherwise puts only the
# script directory on sys.path.  Resolve the repository root without relying on
# the job's PYTHONPATH export (the job still sets it explicitly).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dataline.config import load_sampling_config
from training.dataline.manifest_index import (
    MONET_CAPTIONERS,
    _TOK_COL,
    _band_captioner_vectorized,
    list_shards,
)
from training.dataline.multi_ar import (
    assign_adaptive_bucket,
    build_adaptive_buckets,
    nearest_allowed_bucket,
    square_index,
)
from training.dataline import anchors_v1

BUCKET_MODES = ("adaptive", anchors_v1.BUCKET_MODE)
#: The only stage bucket_mode=anchors_v1 may build under. Every stage pins its own
#: canonical output/stats roots, so this is what keeps a 13-anchor build away from the
#: frozen 361-bucket Stage512a artifacts.
ANCHORS_STAGE = "512b"


OUTPUT_BASE = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop"
)
CANONICAL_OUTPUT_DIRS = {
    "256": OUTPUT_BASE / "offline_arb_catalog_v1",
    "512a": OUTPUT_BASE / "offline_arb_catalog_stage512a_v1",
    # Stage512b-real: its own root so a 13-anchor build can never os.replace() the
    # frozen 361-bucket stage512a parquets in place.
    "512b": OUTPUT_BASE / "offline_arb_catalog_stage512b_v1",
}
CANONICAL_OUTPUT_DIR = CANONICAL_OUTPUT_DIRS["256"]
DEFAULT_TAR_INDEX_DIR = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_arb_tarindex_v1"
)
CANONICAL_BUCKET_STATS_DIRS = {
    "256": OUTPUT_BASE / "offline_arb_bucketstats_v1",
    "512a": OUTPUT_BASE / "offline_arb_bucketstats_stage512a_v1",
    "512b": OUTPUT_BASE / "offline_arb_bucketstats_stage512b_v1",
}
DEFAULT_N_MATRIX = CANONICAL_BUCKET_STATS_DIRS["256"] / "N_matrix.parquet"
DEFAULT_BUCKET_REPORT = (
    CANONICAL_BUCKET_STATS_DIRS["256"] / "bucket_decision_report.json"
)
DEFAULT_PIPELINE_CONFIG = "configs/stage256_redesign_naifu_arb_v1.yaml"

DOMAINS = ("anime", "monet_real", "gpic", "synthetic", "textatlas")
REAL_DOMAINS = frozenset(("monet_real", "gpic", "synthetic"))
TAR_DOMAINS = frozenset(("monet_real", "synthetic"))
LEGACY_STAGE256_FLOOR = 0.005
DEFAULT_ROW_GROUP_SIZE = 65_536
DEFAULT_SOURCE_PARTITIONS = 256
TAR_MISS_MAX_FRACTION = 0.001
MIB = 1 << 20
ELIGIBILITY_FALLBACK_DOMAINS = {
    "256": frozenset(("monet_real", "gpic", "synthetic", "textatlas")),
    "512a": frozenset(("anime", "monet_real", "gpic", "synthetic")),
    # 512b: identical to 512a, because the physical schemas are identical -- textatlas is
    # the one manifest that really stores elig_512, so it must NOT be whitelisted for the
    # short-side fallback. The whitelist is not merely indexed: it is evaluated per shard
    # in eligibility_mode_for_schema() and again as a publication gate
    # (eligibility_source_pass), so widening it makes that gate unsatisfiable.
    "512b": frozenset(("anime", "monet_real", "gpic", "synthetic")),
}


class EligibilityContractError(RuntimeError):
    """A manifest omitted the configured eligibility column outside the audit whitelist."""

# Fork workers inherit this read-mostly context.  It avoids serializing bucket
# tables, sampling config, stats, and the tar-index row-group map per task.
_CTX: dict[str, Any] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * MIB), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def product_name(stem: str, extension: str, diagnostic: bool) -> str:
    return f"{stem}{'.diag' if diagnostic else ''}.{extension}"


def validate_output_dir(raw: str, stage: str) -> Path:
    path = Path(raw).expanduser().resolve(strict=False)
    expected = CANONICAL_OUTPUT_DIRS.get(stage)
    if expected is None:
        raise ValueError(f"unsupported offline catalog stage {stage!r}")
    if path != expected:
        raise ValueError(f"--output-dir must be exactly {expected} for stage={stage}; got {path}")
    return path


def _strict_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def load_runtime_config(config_path: str) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    stage_raw = OmegaConf.select(cfg, "dataset.stage")
    if not isinstance(stage_raw, str) or stage_raw not in CANONICAL_OUTPUT_DIRS:
        raise ValueError(f"unsupported dataset.stage {stage_raw!r}")
    stage = stage_raw
    image_size = _strict_positive_int(
        OmegaConf.select(cfg, "dataset.image_size"), "dataset.image_size"
    )
    expected_image_size = {"256": 256, "512a": 512, "512b": 512}[stage]
    if image_size != expected_image_size:
        raise ValueError(
            f"stage={stage} requires dataset.image_size={expected_image_size}, got {image_size}"
        )
    if OmegaConf.select(cfg, "dataset.multi_ar.enabled") is not True:
        raise ValueError("offline catalog requires dataset.multi_ar.enabled=true")
    bucket_mode = OmegaConf.select(cfg, "dataset.multi_ar.bucket_mode")
    if bucket_mode not in BUCKET_MODES:
        raise ValueError(
            f"offline catalog requires multi_ar.bucket_mode in {BUCKET_MODES}, got "
            f"{bucket_mode!r}; refusing legacy fixed ar_list/bucket_probs/"
            "native_bucket_dist configuration"
        )
    required_adaptive = (
        "target_area",
        "divisor",
        "adaptive_no_upscale",
        "adaptive_min_area_ratio",
        "adaptive_max_aspect",
        "square_only_domains",
    )
    missing = [
        key
        for key in required_adaptive
        if OmegaConf.select(cfg, f"dataset.multi_ar.{key}") is None
    ]
    if missing:
        raise ValueError(f"adaptive bucket config is incomplete: missing {missing}")
    mar = cfg.dataset.multi_ar
    mar_params = {
        "target_area": int(mar.target_area),
        "divisor": int(mar.divisor),
        "no_upscale": bool(mar.adaptive_no_upscale),
        "min_area_ratio": float(mar.adaptive_min_area_ratio),
        "max_aspect": float(mar.adaptive_max_aspect),
    }
    if mar_params["target_area"] != image_size * image_size:
        raise ValueError(
            f"stage={stage} image_size={image_size} requires multi_ar.target_area="
            f"{image_size * image_size}, got {mar_params['target_area']}"
        )
    square_only = tuple(str(x) for x in mar.square_only_domains)
    if set(square_only) - set(DOMAINS):
        raise ValueError(f"unknown square_only_domains: {square_only}")
    sampling_path = str(cfg.dataset.sampling_config)
    stats_cache = str(cfg.dataset.stats_cache)
    sampling_cfg = load_sampling_config(sampling_path)
    if stage not in sampling_cfg.domain_mix:
        raise ValueError(f"sampling config has no domain_mix for stage={stage}")
    stage_mix = sampling_cfg.domain_mix[stage]
    if set(stage_mix) != set(DOMAINS):
        raise ValueError(
            f"stage={stage} domain_mix keys must be exactly {list(DOMAINS)}, got {sorted(stage_mix)}"
        )
    domain_mix = {d: float(stage_mix[d]) for d in DOMAINS}
    if not math.isclose(sum(domain_mix.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"stage={stage} domain_mix must sum to 1")
    requires_nl = bool(sampling_cfg.anime_caption.requires_nl(stage))
    if stage in ("512a", "512b") and requires_nl:
        raise ValueError(
            f"stage{stage} offline catalog requires requires_nl=false; refusing tag-only row loss"
        )
    if bucket_mode == anchors_v1.BUCKET_MODE:
        # Positive assertion, not a blacklist. Each stage pins its own canonical
        # output/stats/N-matrix roots, so a 13-anchor build under stage 512a would
        # os.replace() the five frozen 361-bucket Stage512a parquets in place; and stage
        # 256 has target_area 65,536, which the 262,144-area anchor table does not serve.
        if stage != ANCHORS_STAGE:
            raise ValueError(
                f"bucket_mode={anchors_v1.BUCKET_MODE} requires dataset.stage="
                f"{ANCHORS_STAGE!r}, got {stage!r}. Other stages pin canonical output "
                "directories belonging to a different geometry (stage 512a's frozen "
                "361-bucket catalog would be overwritten in place)."
            )
        # Fixed 13-anchor table. anchor_id order IS the bucket index, and it is the
        # identity key every Stage512b audit is written against, so it must not be
        # re-sorted the way build_adaptive_buckets() sorts by aspect ratio.
        buckets = anchors_v1.anchors_hw()
        # Normalized by anchors_v1.params_from_values(), which phase3a also calls, so the
        # `report_config["multi_ar_params"] != runtime["mar_params"]` equality gate
        # compares dicts produced by one expression rather than two transcriptions.
        # The gate is required, not defaulted: it materially changes which rows are
        # admitted (704x352 = 247,808 is itself below 262,144). 0 means off.
        mar_params.update(anchors_v1.params_from_values(
            OmegaConf.select(cfg, "dataset.multi_ar.anchors_area_gate"),
            OmegaConf.select(cfg, "dataset.multi_ar.anchors_crop_loss_max"),
        ))
    else:
        buckets = build_adaptive_buckets(
            mar_params["target_area"],
            mar_params["divisor"],
            min_area_ratio=mar_params["min_area_ratio"],
            max_aspect=mar_params["max_aspect"],
        )
    return {
        "bucket_mode": bucket_mode,
        "stage": stage,
        "image_size": image_size,
        "eligibility": f"elig_{image_size}",
        "requires_nl": requires_nl,
        "anime_elig_always_true": stage == "256",
        "anime_nl_filter_applied": False,
        "eligibility_fallback_domains": ELIGIBILITY_FALLBACK_DOMAINS[stage],
        "pipeline_cfg": cfg,
        "mar_params": mar_params,
        "square_only": square_only,
        "sampling_path": sampling_path,
        "sampling_cfg": sampling_cfg,
        "stats_cache": stats_cache,
        "domain_mix": domain_mix,
        "buckets": buckets,
        "square_idx": square_index(buckets),
    }


def load_stats_cache_read_only(path: str, expected_monet_shards: int) -> dict[str, Any]:
    """Read the production stats cache without invoking its write-capable helper."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    fingerprint = raw.get("_n_shards")
    if fingerprint not in (None, expected_monet_shards):
        raise RuntimeError(
            f"stats cache shard fingerprint {fingerprint} != monet manifest shards "
            f"{expected_monet_shards}: {path}"
        )
    return {
        "source_counts": raw["source_counts"],
        "cluster_sizes": {int(k): int(v) for k, v in raw["cluster_sizes"].items()},
    }


def extreme_aspect_sides(target_area: int, divisor: int, max_aspect: float) -> tuple[int, int]:
    long_side = max(
        divisor, int(math.floor(math.sqrt(target_area * max_aspect) / divisor)) * divisor
    )
    short_side = max(
        divisor, int(math.floor((long_side / max_aspect) / divisor)) * divisor
    )
    return long_side, short_side


def build_bucket_table(buckets: list[tuple[int, int]], divisor: int) -> np.ndarray:
    max_h = max(h for h, _ in buckets)
    max_w = max(w for _, w in buckets)
    table = np.full((max_h // divisor + 2, max_w // divisor + 2), -1, dtype=np.int64)
    for i, (h, w) in enumerate(buckets):
        table[h // divisor, w // divisor] = i
    return table


def require_identity_fold(bucket_mode: str, fold_map: np.ndarray, floor: float) -> None:
    """In anchors mode the fold must be the identity, or refuse to build.

    The fold is exactly the upsampling hole anchors_v1 exists to close.
    `compute_floor_mapping` -> `nearest_allowed_bucket` is an aspect-ratio nearest
    neighbour with **no feasibility test**, so a pruned extent folds onto a taller
    anchor: 704x352 -> 672x384 enlarges by 1.091x, and the extents are precisely the
    anchors most likely to fall under a probability floor. Shipping that would produce a
    catalog whose rows `anchors_v1.verify_binding()` rejects at train time.
    """
    if bucket_mode != anchors_v1.BUCKET_MODE:
        return
    non_identity = [
        (int(i), int(fold_map[i])) for i in range(len(fold_map)) if int(fold_map[i]) != i
    ]
    if non_identity:
        raise ValueError(
            f"bucket_mode={anchors_v1.BUCKET_MODE} requires an identity fold, but "
            f"floor={floor} would remap {non_identity}; the fold has no feasibility test "
            "and would re-introduce upsampling. Use --floor 0."
        )


def vectorized_anchors_assign(w: np.ndarray, h: np.ndarray) -> np.ndarray:
    """anchors_v1 assignment. Returns -1 for rows no anchor can serve.

    This cannot reuse vectorized_adaptive_assign(): that function keys a dense
    (h//divisor, w//divisor) lookup table and falls back to a Python scalar call on a
    miss. With 13 anchors that table is ~99.9% -1, so every single row would take the
    scalar path -- a catalog build over ~1e9 rows that never finishes, with no error.
    """
    params = _CTX["mar_params"]
    aid, _crop_loss = anchors_v1.assign_anchor_batch(
        np.rint(w).astype(np.int64),
        np.rint(h).astype(np.int64),
        crop_loss_max=params["crop_loss_max"],
        area_gate=params["area_gate"],
    )
    return aid


def vectorized_adaptive_assign(w: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Vectorized Phase-3a assignment, with the production scalar as fallback."""
    params = _CTX["mar_params"]
    target_area = params["target_area"]
    divisor = params["divisor"]
    no_upscale = params["no_upscale"]
    max_aspect = params["max_aspect"]
    extreme_long = _CTX["extreme_long"]
    extreme_short = _CTX["extreme_short"]
    table = _CTX["bucket_table"]
    buckets = _CTX["buckets"]

    raw_area = w * h
    scale = np.where(raw_area > target_area, np.sqrt(target_area / raw_area), 1.0)
    fw = w * scale
    fh = h * scale
    if no_upscale:
        ww = np.maximum(divisor, np.floor(fw / divisor).astype(np.int64) * divisor)
        hh = np.maximum(divisor, np.floor(fh / divisor).astype(np.int64) * divisor)
    else:
        ww = np.maximum(divisor, np.round(fw / divisor).astype(np.int64) * divisor)
        hh = np.maximum(divisor, np.round(fh / divisor).astype(np.int64) * divisor)
    aspect = ww / hh
    over = aspect > max_aspect
    under = aspect < (1.0 / max_aspect)
    hh = np.where(over, extreme_short, np.where(under, extreme_long, hh)).astype(np.int64)
    ww = np.where(over, extreme_long, np.where(under, extreme_short, ww)).astype(np.int64)

    hi = hh // divisor
    wi = ww // divisor
    out = np.full(w.shape[0], -1, dtype=np.int64)
    valid = (hi >= 0) & (hi < table.shape[0]) & (wi >= 0) & (wi < table.shape[1])
    out[valid] = table[hi[valid], wi[valid]]
    for i in np.where(out < 0)[0]:
        out[i] = assign_adaptive_bucket(
            int(round(float(w[i]))),
            int(round(float(h[i]))),
            buckets,
            target_area,
            divisor,
            no_upscale,
            max_aspect,
        )
    return out


def _float_np(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    try:
        return column.to_numpy(zero_copy_only=False).astype(np.float64)
    except Exception:  # null/mixed defensive path
        return np.array(
            [float(x) if x is not None else np.nan for x in column.to_pylist()],
            dtype=np.float64,
        )


def _int_np(column: pa.ChunkedArray | pa.Array, null_value: int = -1) -> np.ndarray:
    try:
        return column.to_numpy(zero_copy_only=False).astype(np.int64)
    except Exception:
        return np.array(
            [int(x) if x is not None else null_value for x in column.to_pylist()], dtype=np.int64
        )


def _bool_np(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    try:
        return column.to_numpy(zero_copy_only=False).astype(bool)
    except Exception:
        return np.array([bool(x) if x is not None else False for x in column.to_pylist()])


def eligibility_mode_for_schema(names: set[str], domain: str) -> str:
    if domain == "anime" and _CTX["anime_elig_always_true"]:
        return "stage256_anime_all_true_compatibility"
    if _CTX["eligibility"] in names:
        return "column"
    if domain in _CTX["eligibility_fallback_domains"]:
        return "short_dim_fallback"
    raise EligibilityContractError(
        f"domain={domain} is missing required {_CTX['eligibility']} outside fallback whitelist"
    )


def filter_and_assign(
    table: pa.Table, domain: str, *, folded: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Return selected row positions, width, height, and raw/folded bucket index."""
    names = set(table.column_names)
    w = _float_np(table.column("width"))
    h = _float_np(table.column("height"))
    n = table.num_rows

    if "holdout" in names:
        holdout_keep = _int_np(table.column("holdout")) == 0
    else:
        holdout_keep = np.ones(n, dtype=bool)
    eligibility_column = _CTX["eligibility"]
    anchors_mode = _CTX["bucket_mode"] == anchors_v1.BUCKET_MODE
    # Do not consult the elig_* schema in anchors mode: it is not read, and
    # eligibility_mode_for_schema() would raise EligibilityContractError over a column
    # this path never touches.
    eligibility_mode = (
        "anchors_v1_area_gate_and_feasible_anchor"
        if anchors_mode
        else eligibility_mode_for_schema(names, domain)
    )
    if anchors_mode:
        # The min(w,h) >= image_size gate is an aspect-ratio filter in disguise and is
        # replaced by the source area gate; servability itself is decided by the
        # feasible-anchor test below, which is what actually guarantees no upsampling.
        gate = _CTX["mar_params"]["area_gate"]
        elig = np.ones(n, dtype=bool) if gate is None else (w * h) >= float(gate)
    elif eligibility_mode == "stage256_anime_all_true_compatibility":
        elig = np.ones(n, dtype=bool)
    elif eligibility_mode == "column":
        elig = _bool_np(table.column(eligibility_column))
    else:
        elig = np.minimum(w, h) >= float(_CTX["image_size"])
    wh_valid = np.isfinite(w) & np.isfinite(h) & (w > 0) & (h > 0)
    if anchors_mode:
        # anchors_v1 raises above MAX_DIM so its exact integer arithmetic cannot
        # overflow. Screen those rows out here instead: one absurd manifest row must
        # not abort a multi-hour build with a message naming neither shard nor uid.
        wh_valid &= (w < anchors_v1.MAX_DIM) & (h < anchors_v1.MAX_DIM)
    keep = holdout_keep & elig & wh_valid
    selected = np.flatnonzero(keep)
    wk = w[selected]
    hk = h[selected]
    unservable = 0
    if anchors_mode:
        # Run the real assignment for every domain, including square-only ones: forcing
        # square_idx without a feasibility test would upsample any row that cannot cover
        # 512x512.
        raw_bucket = vectorized_anchors_assign(wk, hk)
        if domain in _CTX["square_only"]:
            raw_bucket = np.where(
                raw_bucket == int(_CTX["square_idx"]), raw_bucket, np.int64(-1)
            )
        servable = raw_bucket >= 0
        unservable = int((~servable).sum())
        if unservable:
            selected = selected[servable]
            wk = wk[servable]
            hk = hk[servable]
            raw_bucket = raw_bucket[servable]
    elif domain in _CTX["square_only"]:
        raw_bucket = np.full(selected.size, _CTX["square_idx"], dtype=np.int64)
    else:
        raw_bucket = vectorized_adaptive_assign(wk, hk)
    bucket = _CTX["fold_map"][raw_bucket] if folded else raw_bucket
    counters = {
        "rows_raw": int(n),
        "rows_holdout_dropped": int((~holdout_keep).sum()),
        "rows_elig_dropped": int((holdout_keep & ~elig).sum()),
        "rows_invalid_wh_dropped": int((holdout_keep & elig & ~wh_valid).sum()),
        "rows_unservable_dropped": unservable,
        "rows_kept": int(selected.size),
        "eligibility_fallback_rows": (
            int(n) if eligibility_mode == "short_dim_fallback" else 0
        ),
    }
    return selected, wk, hk, bucket, counters


def count_manifest_shard(task: tuple[str, str, int]) -> dict[str, Any]:
    domain, shard_path, task_idx = task
    pf = pq.ParquetFile(shard_path)
    names = set(pf.schema_arrow.names)
    eligibility_mode = eligibility_mode_for_schema(names, domain)
    cols = ["width", "height"]
    if "holdout" in names:
        cols.append("holdout")
    eligibility_column = _CTX["eligibility"]
    if not (domain == "anime" and _CTX["anime_elig_always_true"]):
        if eligibility_column in names:
            cols.append(eligibility_column)
    nvec = np.zeros(len(_CTX["buckets"]), dtype=np.int64)
    totals = defaultdict(int)
    totals["eligibility_column_shards"] = int(eligibility_mode == "column")
    totals["eligibility_fallback_shards"] = int(
        eligibility_mode == "short_dim_fallback"
    )
    totals["eligibility_compatibility_shards"] = int(
        eligibility_mode == "stage256_anime_all_true_compatibility"
    )
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=cols, use_threads=False)
        _sel, _w, _h, bucket, counters = filter_and_assign(table, domain, folded=False)
        nvec += np.bincount(bucket, minlength=len(nvec))[: len(nvec)]
        for key, value in counters.items():
            totals[key] += value
    return {
        "domain": domain,
        "shard_path": shard_path,
        "task_idx": task_idx,
        "N": nvec,
        "coverage": dict(totals),
    }


def read_n_matrix(path: Path, buckets: list[tuple[int, int]]) -> np.ndarray:
    table = pq.read_table(path)
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    if len(cols.get("bucket_h", [])) != len(buckets):
        raise RuntimeError(f"N_matrix bucket count mismatch: {path}")
    for i, expected in enumerate(buckets):
        got = (int(cols["bucket_h"][i]), int(cols["bucket_w"][i]))
        if got != expected:
            raise RuntimeError(f"N_matrix bucket {i}: {got} != config {expected}")
    return np.array(
        [[int(cols[f"n_{domain}"][b]) for domain in DOMAINS] for b in range(len(buckets))],
        dtype=np.int64,
    )


def compute_floor_mapping(
    raw_n: np.ndarray,
    buckets: list[tuple[int, int]],
    domain_mix: dict[str, float],
    square_idx: int,
    floor: float,
) -> tuple[set[int], np.ndarray, np.ndarray, np.ndarray]:
    """Exact make_planner_input.py --floor semantics."""
    totals = raw_n.sum(axis=0)
    probs = np.zeros(raw_n.shape[0], dtype=np.float64)
    for j, domain in enumerate(DOMAINS):
        probs += domain_mix[domain] * raw_n[:, j] / max(int(totals[j]), 1)
    kept = {b for b in range(len(buckets)) if probs[b] >= floor}
    kept.add(square_idx)
    fold_map = np.empty(len(buckets), dtype=np.int64)
    folded = np.zeros_like(raw_n)
    frozen_kept = frozenset(kept)
    for b in range(len(buckets)):
        target = b if b in kept else nearest_allowed_bucket(b, buckets, frozen_kept)
        fold_map[b] = target
        folded[target, :] += raw_n[b, :]
    return kept, fold_map, folded, probs


def assignment_hash_payload(
    runtime: dict[str, Any],
    *,
    bucket_table_hash: str,
    floor: float,
    kept: Iterable[int],
    fold_map: np.ndarray,
) -> dict[str, Any]:
    """Return the canonical assignment identity while preserving stage256 bytes."""
    payload: dict[str, Any] = {
        "bucket_table_hash": bucket_table_hash,
        "floor": floor,
        "kept": sorted(int(value) for value in kept),
        "fold_map": [int(value) for value in fold_map.tolist()],
        "square_only_domains": list(runtime["square_only"]),
        "square_bucket_idx": int(runtime["square_idx"]),
        "assignment": (
            "anchors_v1_feasible_argmax_retention_v1"
            if runtime.get("bucket_mode") == anchors_v1.BUCKET_MODE
            else "phase3a_vectorized_with_production_scalar_fallback_v1"
        ),
        # This historical string is part of the stage256 assignment ABI.  It is
        # intentionally distinct from the new top-level artifact eligibility.
        "eligibility": "phase3a_stage256_v1",
    }
    if runtime["stage"] in ("512a", "512b"):
        payload.update(
            {
                "stage": runtime["stage"],
                "image_size": runtime["image_size"],
                "eligibility": runtime["eligibility"],
                "requires_nl": runtime["requires_nl"],
            }
        )
    if runtime.get("bucket_mode") == anchors_v1.BUCKET_MODE:
        # Without this, a legacy-361 catalog and an anchors-13 catalog would differ only
        # in bucket_table_hash, and the mismatch error would give no hint that two
        # incompatible geometries were involved.
        payload["bucket_mode"] = anchors_v1.BUCKET_MODE
        gate = runtime["mar_params"]["area_gate"]
        payload["anchor_identity"] = anchors_v1.identity(
            crop_loss_max=runtime["mar_params"]["crop_loss_max"], area_gate=gate
        )
        payload["source_area_gate"] = None if gate is None else int(gate)
        payload["crop_loss_max"] = float(runtime["mar_params"]["crop_loss_max"])
    return payload


def _contract_fields(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": runtime["stage"],
        "image_size": runtime["image_size"],
        "eligibility": runtime["eligibility"],
        "requires_nl": runtime["requires_nl"],
        "anime_nl_filter_applied": runtime["anime_nl_filter_applied"],
    }


def _validate_stage_input_path(path: Path, expected: Path, label: str, stage: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if resolved != expected:
        raise ValueError(f"{label} must be {expected} for stage={stage}; got {resolved}")
    return resolved


def validate_bucket_report(
    path: Path,
    runtime: dict[str, Any],
    n_matrix_path: Path,
    n_matrix_sha256: str,
    floor: float,
) -> dict[str, Any]:
    """Bind the floor choice to the same stage-specific Phase-3a evidence."""
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict) or report.get("schema_version") != "offline_arb_bucketstats_v1":
        raise ValueError(f"unsupported bucket decision report: {path}")
    if report.get("diagnostic_run") is not False:
        raise ValueError("catalog floor must come from a formal, non-diagnostic bucket report")

    expected_contract = _contract_fields(runtime)
    present = {key for key in expected_contract if key in report}
    legacy_stage256 = not present
    if legacy_stage256:
        if runtime["stage"] != "256":
            raise ValueError("stage512a bucket report is missing the explicit stage contract")
        nested = report.get("config")
        if not isinstance(nested, dict) or nested.get("stage") != "256":
            raise ValueError("legacy stage256 bucket report identity is not verifiable")
    else:
        if present != set(expected_contract):
            raise ValueError(f"bucket report has a partial stage contract: {sorted(present)}")
        for key, expected in expected_contract.items():
            if report.get(key) != expected:
                raise ValueError(
                    f"bucket report {key}={report.get(key)!r} != config {expected!r}"
                )

    report_config = report.get("config")
    if not isinstance(report_config, dict):
        raise ValueError("bucket report config section is missing")
    if report_config.get("multi_ar_params") != runtime["mar_params"]:
        raise ValueError("bucket report/config multi_ar parameters differ")
    if report_config.get("square_only_domains") != list(runtime["square_only"]):
        raise ValueError("bucket report/config square_only_domains differ")
    if report_config.get("domain_mix") != runtime["domain_mix"]:
        raise ValueError("bucket report/config domain_mix differs")
    eligibility_audit = report.get("eligibility_audit")
    if runtime["stage"] in ("512a", "512b"):
        if not isinstance(eligibility_audit, dict) or eligibility_audit.get("pass") is not True:
            raise ValueError(
                f"stage{runtime['stage']} bucket report eligibility audit is missing or failed"
            )
    # The remaining cross-checks describe the elig_*-column / short-side-fallback
    # bookkeeping, which anchors mode does not produce: it reads no elig_* column.
    if runtime["bucket_mode"] != anchors_v1.BUCKET_MODE and runtime["stage"] in (
        "512a",
        "512b",
    ):
        if eligibility_audit.get("fallback_domain_whitelist") != sorted(
            runtime["eligibility_fallback_domains"]
        ):
            raise ValueError("bucket report eligibility fallback whitelist differs")
        coverage = report.get("coverage")
        if not isinstance(coverage, dict):
            raise ValueError("bucket report coverage is missing")
        for domain in DOMAINS:
            row = coverage.get(domain)
            if not isinstance(row, dict):
                raise ValueError(f"bucket report coverage missing domain={domain}")
            if domain in runtime["eligibility_fallback_domains"]:
                valid = (
                    row.get("eligibility_fallback_shards") == row.get("shards_scanned")
                    and row.get("eligibility_column_shards") == 0
                )
            else:
                valid = (
                    row.get("eligibility_column_shards") == row.get("shards_scanned")
                    and row.get("eligibility_fallback_shards") == 0
                )
            if not valid:
                raise ValueError(
                    f"bucket report eligibility source mismatch for {domain}: {row}"
                )
    report_buckets = report.get("buckets")
    expected_buckets = [
        {"idx": i, "h": h, "w": w, "ar": w / h}
        for i, (h, w) in enumerate(runtime["buckets"])
    ]
    if report_buckets != expected_buckets:
        raise ValueError("bucket report/config adaptive bucket table differs")

    matrix = report.get("N_matrix")
    if isinstance(matrix, dict):
        if matrix.get("sha256") != n_matrix_sha256:
            raise ValueError("bucket report N_matrix SHA does not match selected N_matrix")
        report_matrix_path = Path(str(matrix.get("path", ""))).expanduser().resolve(strict=False)
        if report_matrix_path != n_matrix_path:
            raise ValueError("bucket report N_matrix path does not match selected N_matrix")
    elif runtime["stage"] != "256" or not legacy_stage256:
        raise ValueError("bucket report does not bind N_matrix path/SHA")

    candidates = report.get("floors")
    if not isinstance(candidates, list):
        raise ValueError("bucket report floors must be a list")
    matches = [
        entry
        for entry in candidates
        if isinstance(entry, dict)
        and type(entry.get("floor")) in (int, float)
        and float(entry["floor"]) == floor
    ]
    if len(matches) != 1:
        raise ValueError(
            f"selected --floor={floor} must appear exactly once in bucket report; found {len(matches)}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "legacy_stage256_contract": legacy_stage256,
        "selected_floor": floor,
        "selected_floor_report": matches[0],
    }


def matrix_list(matrix: np.ndarray) -> list[list[int]]:
    return [[int(x) for x in row] for row in matrix.tolist()]


def sparse_matrix(matrix: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in range(matrix.shape[0]):
        for j, domain in enumerate(DOMAINS):
            value = int(matrix[b, j])
            if value:
                out.append({"bucket_idx": b, "domain": domain, "value": value})
    return out


def _common_no_row_fields() -> list[pa.Field]:
    return [
        pa.field("uid", pa.string(), nullable=False),
        pa.field("domain", pa.string(), nullable=False),
        pa.field("bucket_idx", pa.int16(), nullable=False),
        pa.field("width", pa.int32(), nullable=False),
        pa.field("height", pa.int32(), nullable=False),
    ]


def _real_condition_fields() -> list[pa.Field]:
    return [
        pa.field("src_key", pa.string(), nullable=False),
        pa.field("sub_source", pa.string(), nullable=False),
        pa.field("cluster_id", pa.int64()),
        pa.field("caption_type", pa.string()),
        pa.field("caption_tier", pa.int8()),
        pa.field("served_mask", pa.uint8(), nullable=False),
        pa.field("band_captioner_t1", pa.int8(), nullable=False),
        pa.field("band_captioner_t2", pa.int8(), nullable=False),
        pa.field("band_captioner_t3", pa.int8(), nullable=False),
        pa.field("band_captioner_t4", pa.int8(), nullable=False),
        pa.field("base_weight", pa.float64(), nullable=False),
    ]


ANIME_SCHEMA_NO_ROW = pa.schema(
    _common_no_row_fields()
    + [
        pa.field("key", pa.string(), nullable=False),
        pa.field("fetch_url", pa.string(), nullable=False),
        pa.field("cluster", pa.int64()),
        pa.field("rating_bucket", pa.string()),
        pa.field("year", pa.int32()),
        pa.field("caption_parquet_path", pa.string(), nullable=False),
        pa.field("caption_row_group", pa.int32(), nullable=False),
        pa.field("caption_row_in_group", pa.int32(), nullable=False),
    ]
)

MONET_SCHEMA_NO_ROW = pa.schema(
    _common_no_row_fields()
    + [
        pa.field("tar_path", pa.string(), nullable=False),
        pa.field("tar_offset", pa.int64(), nullable=False),
        pa.field("tar_size", pa.int64(), nullable=False),
    ]
    + _real_condition_fields()
    + [
        pa.field("caption_parquet_path", pa.string(), nullable=False),
        pa.field("caption_row_group", pa.int32(), nullable=False),
        pa.field("caption_row_in_group", pa.int32(), nullable=False),
    ]
)

GPIC_SCHEMA_NO_ROW = pa.schema(
    _common_no_row_fields()
    + [pa.field("image_path", pa.string(), nullable=False)]
    + _real_condition_fields()
    + [
        pa.field("caption_parquet_path", pa.string(), nullable=False),
        pa.field("caption_row_group", pa.int32(), nullable=False),
        pa.field("caption_row_in_group", pa.int32(), nullable=False),
    ]
)

SYNTH_SCHEMA_NO_ROW = pa.schema(
    _common_no_row_fields()
    + [
        pa.field("tar_path", pa.string(), nullable=False),
        pa.field("tar_offset", pa.int64(), nullable=False),
        pa.field("tar_size", pa.int64(), nullable=False),
    ]
    + _real_condition_fields()
    + [
        pa.field("caption_kind", pa.string(), nullable=False),
        pa.field("caption_parquet_path", pa.string(), nullable=False),
        pa.field("caption_row_group", pa.int32(), nullable=False),
        pa.field("caption_row_in_group", pa.int32(), nullable=False),
    ]
)

TEXTATLAS_SCHEMA_NO_ROW = pa.schema(
    _common_no_row_fields()
    + [
        pa.field("image_path", pa.string(), nullable=False),
        pa.field("image_parquet_path", pa.string(), nullable=False),
        pa.field("image_row_group", pa.int32(), nullable=False),
        pa.field("image_row_in_group", pa.int32(), nullable=False),
        pa.field("caption_parquet_path", pa.string(), nullable=False),
        pa.field("caption_row_group", pa.int32(), nullable=False),
        pa.field("caption_row_in_group", pa.int32(), nullable=False),
    ]
)

FINAL_SCHEMAS_NO_ROW = {
    "anime": ANIME_SCHEMA_NO_ROW,
    "monet_real": MONET_SCHEMA_NO_ROW,
    "gpic": GPIC_SCHEMA_NO_ROW,
    "synthetic": SYNTH_SCHEMA_NO_ROW,
    "textatlas": TEXTATLAS_SCHEMA_NO_ROW,
}
FINAL_SCHEMAS = {
    domain: pa.schema([pa.field("row_id", pa.int64(), nullable=False)] + list(schema))
    for domain, schema in FINAL_SCHEMAS_NO_ROW.items()
}

CANDIDATE_SCHEMA = pa.schema(
    _common_no_row_fields()
    + [
        pa.field("src_root", pa.string(), nullable=False),
        pa.field("src_shard", pa.string(), nullable=False),
    ]
    + _real_condition_fields()
    + [pa.field("source_partition", pa.int16(), nullable=False)]
)

BAD_ROWS_SCHEMA = pa.schema(
    [
        pa.field("uid", pa.string(), nullable=False),
        pa.field("domain", pa.string(), nullable=False),
        pa.field("bucket_idx", pa.int16(), nullable=False),
        pa.field("src_shard", pa.string(), nullable=False),
        pa.field("src_key", pa.string(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
    ]
)


def _take_py(table: pa.Table, name: str, selected: np.ndarray) -> list[Any]:
    return table.column(name).take(pa.array(selected, type=pa.int64())).to_pylist()


def _source_partition(domain: str, src_root: str, src_shard: str) -> int:
    payload = f"{domain}\0{src_root}\0{src_shard}".encode("utf-8")
    return (zlib.crc32(payload) & 0xFFFFFFFF) % int(_CTX["source_partitions"])


def real_derived_columns(
    table: pa.Table, domain: str, selected: np.ndarray
) -> dict[str, list[Any]]:
    cfg = _CTX["sampling_cfg"]
    stats = _CTX["stats"]
    sub_all = np.array(table.column("sub_source").to_pylist(), dtype=object)
    cluster_all = _int_np(table.column("cluster_id"))
    tier_all = _int_np(table.column("caption_tier"))
    tok_arrays = {
        c: _int_np(table.column(_TOK_COL[c])) for c in MONET_CAPTIONERS
    }
    tok_i1 = _int_np(table.column("tok_i1"))
    band = _band_captioner_vectorized(
        tok_arrays,
        tok_i1,
        sub_all,
        tier_all,
        domain == "gpic",
        cfg,
    )[selected]
    served = np.zeros(selected.size, dtype=np.uint8)
    for bi in range(4):
        served |= ((band[:, bi] > 0).astype(np.uint8) << bi)

    sub = sub_all[selected]
    cluster = cluster_all[selected]
    if domain == "gpic":
        source_factor = np.ones(selected.size, dtype=np.float64)
    elif cfg.monet_subsource_explicit_probs and domain == "monet_real":
        explicit = cfg.monet_subsource_explicit_probs
        source_factor = np.array(
            [
                explicit.get(str(s), 0.0)
                / max(int(stats["source_counts"].get(str(s), 1)), 1)
                for s in sub
            ],
            dtype=np.float64,
        )
    else:
        source_factor = np.array(
            [stats["source_counts"].get(str(s), 1) ** (-cfg.source_alpha) for s in sub],
            dtype=np.float64,
        )
    cluster_factor = np.array(
        [
            1.0
            if int(c) == -1
            else stats["cluster_sizes"].get(int(c), 1) ** (-cfg.cluster_beta)
            for c in cluster
        ],
        dtype=np.float64,
    )
    return {
        "sub_source": [str(x) for x in sub.tolist()],
        "cluster_id": [int(x) for x in cluster.tolist()],
        "caption_type": _take_py(table, "caption_type", selected),
        "caption_tier": [int(x) for x in tier_all[selected].tolist()],
        "served_mask": served.tolist(),
        "band_captioner_t1": band[:, 0].astype(np.int8).tolist(),
        "band_captioner_t2": band[:, 1].astype(np.int8).tolist(),
        "band_captioner_t3": band[:, 2].astype(np.int8).tolist(),
        "band_captioner_t4": band[:, 3].astype(np.int8).tolist(),
        "base_weight": (source_factor * cluster_factor).tolist(),
    }


def _required_real_columns(names: set[str], domain: str) -> list[str]:
    required = {
        "uid",
        "sub_source",
        "src_root",
        "src_shard",
        "src_key",
        "width",
        "height",
        "cluster_id",
        "caption_type",
        "caption_tier",
        "holdout",
        "tok_i1",
        *(_TOK_COL[c] for c in MONET_CAPTIONERS),
    }
    missing = required - names
    if missing:
        raise RuntimeError(f"real/synth manifest missing required columns: {sorted(missing)}")
    cols = sorted(required)
    eligibility_mode_for_schema(names, domain)
    eligibility_column = str(_CTX["eligibility"])
    if eligibility_column in names:
        cols.append(eligibility_column)
    return cols


def _write_sorted_cell_runs(
    domain: str,
    token: str,
    tables_by_bucket: dict[int, list[pa.Table]],
    output_dir: Path,
) -> tuple[dict[int, str], list[int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, str] = {}
    counts = [0] * len(_CTX["buckets"])
    schema = FINAL_SCHEMAS_NO_ROW[domain]
    for bucket, pieces in tables_by_bucket.items():
        table = pa.concat_tables(pieces) if len(pieces) > 1 else pieces[0]
        table = table.cast(schema).sort_by([("uid", "ascending")])
        path = output_dir / f"{token}.b{bucket:03d}.parquet"
        pq.write_table(
            table,
            path,
            compression="zstd",
            row_group_size=int(_CTX["row_group_size"]),
            use_dictionary=True,
            write_statistics=True,
        )
        paths[bucket] = str(path)
        counts[bucket] = table.num_rows
    return paths, counts


def _append_by_bucket(
    destination: dict[int, list[pa.Table]], table: pa.Table, bucket: np.ndarray
) -> None:
    for b in np.unique(bucket):
        positions = np.flatnonzero(bucket == b)
        destination[int(b)].append(table.take(pa.array(positions, type=pa.int64())))


def _textatlas_source_positions(
    source_path: str, absolute_rows: np.ndarray, expected_rows: int
) -> tuple[list[int], list[int]]:
    source_pf = pq.ParquetFile(source_path)
    if source_pf.metadata.num_rows != expected_rows:
        raise RuntimeError(
            f"textatlas source/sidecar row mismatch: {source_path} "
            f"{source_pf.metadata.num_rows} != {expected_rows}"
        )
    sizes = np.array(
        [source_pf.metadata.row_group(i).num_rows for i in range(source_pf.num_row_groups)],
        dtype=np.int64,
    )
    ends = np.cumsum(sizes)
    starts = np.concatenate((np.array([0], dtype=np.int64), ends[:-1]))
    rgs = np.searchsorted(ends, absolute_rows, side="right")
    within = absolute_rows - starts[rgs]
    return rgs.astype(np.int32).tolist(), within.astype(np.int32).tolist()


def build_manifest_fragment(task: tuple[str, str, int, str]) -> dict[str, Any]:
    """Second pass: one manifest shard -> candidate or final direct cell runs."""
    domain, shard_path, task_idx, work_root_s = task
    work_root = Path(work_root_s)
    pf = pq.ParquetFile(shard_path)
    names = set(pf.schema_arrow.names)
    token = f"{task_idx:06d}_{hashlib.sha256(shard_path.encode()).hexdigest()[:12]}"
    counts = np.zeros(len(_CTX["buckets"]), dtype=np.int64)
    coverage = defaultdict(int)
    eligibility_mode = eligibility_mode_for_schema(names, domain)
    coverage["eligibility_column_shards"] = int(eligibility_mode == "column")
    coverage["eligibility_fallback_shards"] = int(
        eligibility_mode == "short_dim_fallback"
    )
    coverage["eligibility_compatibility_shards"] = int(
        eligibility_mode == "stage256_anime_all_true_compatibility"
    )

    if domain in REAL_DOMAINS:
        columns = _required_real_columns(names, domain)
        pieces: list[pa.Table] = []
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=columns, use_threads=False)
            selected, wk, hk, bucket, counters = filter_and_assign(table, domain, folded=True)
            for key, value in counters.items():
                coverage[key] += value
            if selected.size == 0:
                continue
            counts += np.bincount(bucket, minlength=len(counts))[: len(counts)]
            uid = [str(x) for x in _take_py(table, "uid", selected)]
            src_root = [str(x) for x in _take_py(table, "src_root", selected)]
            src_shard = [str(x) for x in _take_py(table, "src_shard", selected)]
            src_key = [str(x) for x in _take_py(table, "src_key", selected)]
            partitions: list[int] = []
            partition_cache: dict[tuple[str, str], int] = {}
            for root, shard in zip(src_root, src_shard):
                ref = (root, shard)
                part = partition_cache.get(ref)
                if part is None:
                    part = _source_partition(domain, root, shard)
                    partition_cache[ref] = part
                partitions.append(part)
            data: dict[str, Any] = {
                "uid": uid,
                "domain": [domain] * selected.size,
                "bucket_idx": bucket.astype(np.int16).tolist(),
                "width": np.rint(wk).astype(np.int32).tolist(),
                "height": np.rint(hk).astype(np.int32).tolist(),
                "src_root": src_root,
                "src_shard": src_shard,
                "src_key": src_key,
                **real_derived_columns(table, domain, selected),
                "source_partition": partitions,
            }
            pieces.append(pa.Table.from_pydict(data, schema=CANDIDATE_SCHEMA))
        candidate = pa.concat_tables(pieces) if pieces else pa.Table.from_pylist([], CANDIDATE_SCHEMA)
        out_dir = work_root / "candidates" / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{token}.parquet"
        pq.write_table(
            candidate,
            out_path,
            compression="zstd",
            row_group_size=int(_CTX["row_group_size"]),
            use_dictionary=True,
        )
        return {
            "domain": domain,
            "shard_path": shard_path,
            "candidate_path": str(out_path),
            "cell_paths": {},
            "counts": counts,
            "coverage": dict(coverage),
        }

    tables_by_bucket: dict[int, list[pa.Table]] = defaultdict(list)
    if domain == "anime":
        required = {
            "key",
            "s3key",
            "width",
            "height",
            "cluster_clipl14_level2",
            "rating_bucket",
            "year",
        }
        missing = required - names
        if missing:
            raise RuntimeError(f"anime manifest missing columns: {sorted(missing)}")
        columns = sorted(required)
        eligibility_column = str(_CTX["eligibility"])
        if eligibility_column in names:
            columns.append(eligibility_column)
        rg_starts = np.cumsum(
            [0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups - 1)]
        )
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=columns, use_threads=False)
            selected, wk, hk, bucket, counters = filter_and_assign(table, domain, folded=True)
            for key, value in counters.items():
                coverage[key] += value
            if selected.size == 0:
                continue
            counts += np.bincount(bucket, minlength=len(counts))[: len(counts)]
            keys = [str(x) for x in _take_py(table, "key", selected)]
            data = {
                "uid": [f"anime:{key}" for key in keys],
                "domain": [domain] * selected.size,
                "bucket_idx": bucket.astype(np.int16).tolist(),
                "width": np.rint(wk).astype(np.int32).tolist(),
                "height": np.rint(hk).astype(np.int32).tolist(),
                "key": keys,
                "fetch_url": [str(x) for x in _take_py(table, "s3key", selected)],
                "cluster": _take_py(table, "cluster_clipl14_level2", selected),
                "rating_bucket": _take_py(table, "rating_bucket", selected),
                "year": _take_py(table, "year", selected),
                "caption_parquet_path": [shard_path] * selected.size,
                "caption_row_group": [rg] * selected.size,
                "caption_row_in_group": selected.astype(np.int32).tolist(),
            }
            out = pa.Table.from_pydict(data, schema=ANIME_SCHEMA_NO_ROW)
            _append_by_bucket(tables_by_bucket, out, bucket)
    elif domain == "textatlas":
        required = {"image_path", "uid", "width", "height"}
        missing = required - names
        if missing:
            raise RuntimeError(f"textatlas sidecar missing columns: {sorted(missing)}")
        eligibility_column = str(_CTX["eligibility"])
        columns = sorted(
            required | ({eligibility_column} if eligibility_column in names else set())
        )
        subset = Path(shard_path).parent.name
        shard_name = Path(shard_path).name
        source_path = str(Path(_CTX["textatlas_source_root"]) / subset / shard_name)
        rg_sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
        rg_starts = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(rg_sizes, dtype=np.int64)[:-1])
        )
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=columns, use_threads=False)
            selected, wk, hk, bucket, counters = filter_and_assign(table, domain, folded=True)
            for key, value in counters.items():
                coverage[key] += value
            if selected.size == 0:
                continue
            counts += np.bincount(bucket, minlength=len(counts))[: len(counts)]
            absolute = rg_starts[rg] + selected
            image_rg, image_in = _textatlas_source_positions(
                source_path, absolute, pf.metadata.num_rows
            )
            data = {
                "uid": [str(x) for x in _take_py(table, "uid", selected)],
                "domain": [domain] * selected.size,
                "bucket_idx": bucket.astype(np.int16).tolist(),
                "width": np.rint(wk).astype(np.int32).tolist(),
                "height": np.rint(hk).astype(np.int32).tolist(),
                "image_path": [str(x) for x in _take_py(table, "image_path", selected)],
                "image_parquet_path": [source_path] * selected.size,
                "image_row_group": image_rg,
                "image_row_in_group": image_in,
                "caption_parquet_path": [shard_path] * selected.size,
                "caption_row_group": [rg] * selected.size,
                "caption_row_in_group": selected.astype(np.int32).tolist(),
            }
            out = pa.Table.from_pydict(data, schema=TEXTATLAS_SCHEMA_NO_ROW)
            _append_by_bucket(tables_by_bucket, out, bucket)
    else:  # defensive: DOMAINS is closed
        raise AssertionError(domain)

    paths, written_counts = _write_sorted_cell_runs(
        domain, token, tables_by_bucket, work_root / "cell_runs" / domain
    )
    if written_counts != counts.tolist():
        raise RuntimeError(f"direct cell write count drift in {shard_path}")
    return {
        "domain": domain,
        "shard_path": shard_path,
        "candidate_path": None,
        "cell_paths": paths,
        "counts": counts,
        "coverage": dict(coverage),
    }


def repartition_candidates(
    domain: str,
    candidate_paths: list[str],
    work_root: Path,
    partitions: int,
    row_group_size: int,
) -> dict[int, str]:
    """Hash-partition candidates by source shard with bounded aggregate buffers."""
    out_dir = work_root / "source_partitions" / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[int, pq.ParquetWriter] = {}
    paths: dict[int, Path] = {}
    buffers: dict[int, list[pa.Table]] = defaultdict(list)
    buffered_rows: dict[int, int] = defaultdict(int)
    total_buffered = 0

    def flush(part: int) -> None:
        nonlocal total_buffered
        if not buffers[part]:
            return
        table = pa.concat_tables(buffers[part]) if len(buffers[part]) > 1 else buffers[part][0]
        if part not in writers:
            path = out_dir / f"part_{part:04d}.parquet"
            paths[part] = path
            writers[part] = pq.ParquetWriter(path, CANDIDATE_SCHEMA, compression="zstd")
        writers[part].write_table(table, row_group_size=row_group_size)
        total_buffered -= buffered_rows[part]
        buffers[part].clear()
        buffered_rows[part] = 0

    try:
        for path in candidate_paths:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=row_group_size, use_threads=False):
                table = pa.Table.from_batches([batch], schema=CANDIDATE_SCHEMA)
                part_values = _int_np(table.column("source_partition"))
                for part in np.unique(part_values):
                    selected = np.flatnonzero(part_values == part)
                    piece = table.take(pa.array(selected, type=pa.int64()))
                    p = int(part)
                    buffers[p].append(piece)
                    buffered_rows[p] += piece.num_rows
                    total_buffered += piece.num_rows
                    if buffered_rows[p] >= row_group_size:
                        flush(p)
                if total_buffered >= 4_000_000:
                    for p in list(buffers):
                        flush(p)
        for p in list(buffers):
            flush(p)
    finally:
        for writer in writers.values():
            writer.close()
    return {part: str(path) for part, path in paths.items()}


def build_tar_rowgroup_map(tar_index_dir: Path) -> tuple[dict[tuple[str, str], list[tuple[str, int]]], dict[str, Any]]:
    """Map (domain, src_shard) to exact tar-index row groups using footer stats."""
    success = tar_index_dir / "_SUCCESS"
    if not success.is_file():
        raise FileNotFoundError(success)
    # taridx 分片改造(2026-07-10)后,完成记录分散在 legacy manifest.json(cc12m)与
    # manifest.<tag>.json(a/b/c 分片各写各的)。必须与 phase3a_tar_index.py 的
    # load_all_manifest_records 同语义合并读取;.diag 永不参与。_SUCCESS 已由
    # --verify-only 对全部 9 个子源做过完整性+sha256 gate,这里按 status==ok 取即可。
    manifest_paths = [
        p for p in sorted(tar_index_dir.glob("manifest*.json")) if "diag" not in p.name
    ]
    if not manifest_paths:
        raise RuntimeError(f"no tar-index manifest files under {tar_index_dir}")
    merged: dict[str, dict[str, Any]] = {}
    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        for name, info in (manifest.get("files") or {}).items():
            if not name.endswith(".parquet") or name.endswith(".diag.parquet"):
                continue
            if info.get("status") != "ok" or name in merged:
                continue
            merged[name] = info
    files: list[tuple[str, Path]] = []
    for name, info in sorted(merged.items()):
        path = tar_index_dir / name
        if path.is_file():
            files.append((str(info.get("domain")), path))
    if not files:
        raise RuntimeError(
            f"tar-index manifests have no completed parquet files: "
            f"{[str(p) for p in manifest_paths]}"
        )

    mapping: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    row_groups = 0
    for domain_hint, path in files:
        pf = pq.ParquetFile(path)
        schema_names = pf.schema.names
        src_col_idx = schema_names.index("src_shard")
        for rg in range(pf.num_row_groups):
            row_groups += 1
            stats = pf.metadata.row_group(rg).column(src_col_idx).statistics
            shards: list[str]
            if stats is not None and stats.has_min_max and stats.min == stats.max:
                value = stats.min.decode() if isinstance(stats.min, bytes) else str(stats.min)
                shards = [value]
            else:
                tiny = pf.read_row_group(rg, columns=["src_shard", "domain"], use_threads=False)
                shards = sorted({str(x) for x in tiny.column("src_shard").to_pylist()})
                domains = {str(x) for x in tiny.column("domain").to_pylist()}
                if len(domains) != 1:
                    raise RuntimeError(f"mixed domains in tar-index row group {path}:{rg}")
                domain_hint = next(iter(domains))
            for shard in shards:
                mapping[(domain_hint, shard)].append((str(path), rg))
    # gate 判定实际落在 gate_summary.json(--verify-only 写),不在任何 manifest 里。
    gate_path = tar_index_dir / "gate_summary.json"
    gate: dict[str, Any] | None = None
    if gate_path.is_file():
        with gate_path.open("r", encoding="utf-8") as f:
            gate = json.load(f)
    summary = {
        "manifest_paths": [str(p) for p in manifest_paths],
        "manifest_sha256": {p.name: sha256_file(p) for p in manifest_paths},
        "files": [str(path) for _, path in files],
        "file_count": len(files),
        "row_group_count": row_groups,
        "source_shard_count": len(mapping),
        "gate": gate,
    }
    return dict(mapping), summary


def _caption_source_path(domain: str, src_root: str, src_shard: str) -> Path:
    if domain == "gpic":
        return Path(src_root) / "parquets" / f"gpic_train_{src_shard}.parquet"
    return Path(src_root) / src_shard


def read_caption_source_locator(
    domain: str, src_root: str, src_shard: str, wanted: set[str]
) -> dict[str, tuple[str, int, int, str | None]]:
    path = _caption_source_path(domain, src_root, src_shard)
    pf = pq.ParquetFile(path)
    key_col = "key" if domain == "gpic" else "id"
    columns = [key_col, "s3key"] if domain == "gpic" else [key_col]
    found: dict[str, tuple[str, int, int, str | None]] = {}
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=columns, use_threads=False)
        keys = table.column(key_col).to_pylist()
        s3keys = table.column("s3key").to_pylist() if domain == "gpic" else [None] * len(keys)
        for row_in_group, (key_raw, s3key) in enumerate(zip(keys, s3keys)):
            key = str(key_raw)
            if key not in wanted:
                continue
            if key in found:
                raise RuntimeError(f"duplicate source caption key {domain} {path} {key}")
            image_path = None
            if domain == "gpic":
                ext = os.path.splitext(str(s3key))[1] if s3key else ".jpg"
                if ext and not ext.startswith("."):
                    ext = "." + ext
                image_path = str(Path(src_root) / "images" / "train" / src_shard / f"{key}{ext}")
            found[key] = (str(path), rg, row_in_group, image_path)
    missing = wanted - found.keys()
    if missing:
        sample = sorted(missing)[:10]
        raise RuntimeError(
            f"caption source locator miss domain={domain} source={path} "
            f"missing={len(missing)}/{len(wanted)} sample={sample}"
        )
    return found


def read_tar_locators(
    domain: str, src_shard: str, wanted: set[str]
) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    for path_s, rg in _CTX["tar_rg_map"].get((domain, src_shard), []):
        pf = pq.ParquetFile(path_s)
        table = pf.read_row_group(
            rg, columns=["src_shard", "src_key", "offset", "size", "domain"], use_threads=False
        )
        shards = table.column("src_shard").to_pylist()
        keys = table.column("src_key").to_pylist()
        offsets = table.column("offset").to_pylist()
        sizes = table.column("size").to_pylist()
        domains = table.column("domain").to_pylist()
        for shard_raw, key_raw, offset, size, row_domain in zip(
            shards, keys, offsets, sizes, domains
        ):
            if str(row_domain) != domain or str(shard_raw) != src_shard:
                continue
            key = str(key_raw)
            if key not in wanted:
                continue
            if key in found:
                raise RuntimeError(f"duplicate tar-index key {domain} {src_shard} {key}")
            found[key] = (int(offset), int(size))
    return found


def _candidate_rows(table: pa.Table) -> dict[str, list[Any]]:
    return {name: table.column(name).to_pylist() for name in table.column_names}


def _empty_bad_table() -> pa.Table:
    return pa.Table.from_pylist([], schema=BAD_ROWS_SCHEMA)


def join_source_partition(task: tuple[str, int, str, str]) -> dict[str, Any]:
    """Join one bounded source partition to source-row and tar locators."""
    domain, part, candidate_path, work_root_s = task
    work_root = Path(work_root_s)
    table = pq.read_table(candidate_path)
    rows = _candidate_rows(table)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (root, shard) in enumerate(zip(rows["src_root"], rows["src_shard"])):
        groups[(str(root), str(shard))].append(i)

    final_columns: dict[int, dict[str, list[Any]]] = {
        b: {field.name: [] for field in FINAL_SCHEMAS_NO_ROW[domain]}
        for b in range(len(_CTX["buckets"]))
    }
    bad: dict[str, list[Any]] = {field.name: [] for field in BAD_ROWS_SCHEMA}
    input_count = table.num_rows
    tar_hits = 0
    tar_misses = 0

    condition_names = [field.name for field in _real_condition_fields()]
    common_names = ["uid", "domain", "bucket_idx", "width", "height"]

    for (src_root, src_shard), indices in groups.items():
        wanted = {str(rows["src_key"][i]) for i in indices}
        caption_locs = read_caption_source_locator(domain, src_root, src_shard, wanted)
        tar_locs = read_tar_locators(domain, src_shard, wanted) if domain in TAR_DOMAINS else {}
        tar_path = str(Path(src_root) / Path(src_shard).with_suffix(".tar"))

        for i in indices:
            key = str(rows["src_key"][i])
            bucket = int(rows["bucket_idx"][i])
            if domain in TAR_DOMAINS and key not in tar_locs:
                tar_misses += 1
                bad["uid"].append(str(rows["uid"][i]))
                bad["domain"].append(domain)
                bad["bucket_idx"].append(bucket)
                bad["src_shard"].append(src_shard)
                bad["src_key"].append(key)
                bad["reason"].append("tar_index_miss")
                continue
            target = final_columns[bucket]
            for name in common_names:
                target[name].append(rows[name][i])
            caption_path, caption_rg, caption_in, image_path = caption_locs[key]
            if domain in TAR_DOMAINS:
                offset, size = tar_locs[key]
                if size <= 0 or offset < 0:
                    raise RuntimeError(
                        f"invalid tar locator {domain} {src_shard} {key}: {offset},{size}"
                    )
                tar_hits += 1
                target["tar_path"].append(tar_path)
                target["tar_offset"].append(offset)
                target["tar_size"].append(size)
            else:
                if not image_path:
                    raise RuntimeError(f"gpic image path absent: {src_shard} {key}")
                target["image_path"].append(image_path)
            for name in condition_names:
                target[name].append(rows[name][i])
            if domain == "synthetic":
                target["caption_kind"].append("original")
            target["caption_parquet_path"].append(caption_path)
            target["caption_row_group"].append(caption_rg)
            target["caption_row_in_group"].append(caption_in)

    tables_by_bucket: dict[int, list[pa.Table]] = defaultdict(list)
    for bucket, data in final_columns.items():
        if not data["uid"]:
            continue
        tables_by_bucket[bucket].append(
            pa.Table.from_pydict(data, schema=FINAL_SCHEMAS_NO_ROW[domain])
        )
    token = f"part_{part:04d}"
    paths, counts = _write_sorted_cell_runs(
        domain, token, tables_by_bucket, work_root / "cell_runs" / domain / "joined"
    )
    bad_table = pa.Table.from_pydict(bad, schema=BAD_ROWS_SCHEMA) if bad["uid"] else _empty_bad_table()
    bad_dir = work_root / "bad_parts"
    bad_dir.mkdir(parents=True, exist_ok=True)
    bad_path = bad_dir / f"{domain}_{token}.parquet"
    pq.write_table(bad_table, bad_path, compression="zstd")
    if sum(counts) + tar_misses != input_count:
        raise RuntimeError(f"partition accounting drift {domain} part={part}")
    return {
        "domain": domain,
        "part": part,
        "cell_paths": paths,
        "counts": counts,
        "bad_path": str(bad_path),
        "input_count": input_count,
        "tar_hits": tar_hits,
        "tar_misses": tar_misses,
    }


def consolidate_bad_rows(paths: list[str], output_path: Path, row_group_size: int) -> pa.Table:
    pieces = [pq.read_table(path) for path in paths if pq.ParquetFile(path).metadata.num_rows]
    table = pa.concat_tables(pieces) if pieces else _empty_bad_table()
    if table.num_rows:
        table = table.sort_by([("domain", "ascending"), ("bucket_idx", "ascending"), ("uid", "ascending")])
    pq.write_table(
        table,
        output_path,
        compression="zstd",
        row_group_size=row_group_size,
        use_dictionary=True,
    )
    return table


def merge_domain_cells(
    domain: str,
    runs_by_bucket: dict[int, list[str]],
    output_path: Path,
    start_row_id: int,
    row_group_size: int,
) -> tuple[int, list[int], dict[str, Any]]:
    """Second-level merge: one domain at a time, one cell at a time."""
    writer = pq.ParquetWriter(output_path, FINAL_SCHEMAS[domain], compression="zstd")
    row_id = start_row_id
    counts = [0] * len(_CTX["buckets"])
    max_cell_rows = 0
    max_cell_bucket: int | None = None
    try:
        for bucket in range(len(_CTX["buckets"])):
            paths = sorted(runs_by_bucket.get(bucket, []))
            if not paths:
                continue
            pieces = [pq.read_table(path) for path in paths]
            table = pa.concat_tables(pieces) if len(pieces) > 1 else pieces[0]
            table = table.cast(FINAL_SCHEMAS_NO_ROW[domain]).sort_by([("uid", "ascending")])
            uids = table.column("uid").to_pylist()
            for i in range(1, len(uids)):
                if uids[i] == uids[i - 1]:
                    raise RuntimeError(f"duplicate uid within cell {domain}/{bucket}: {uids[i]}")
            n = table.num_rows
            ids = pa.array(np.arange(row_id, row_id + n, dtype=np.int64), type=pa.int64())
            table = table.add_column(0, "row_id", ids).cast(FINAL_SCHEMAS[domain])
            writer.write_table(table, row_group_size=row_group_size)
            counts[bucket] = n
            row_id += n
            if n > max_cell_rows:
                max_cell_rows = n
                max_cell_bucket = bucket
            del pieces, table, uids, ids
            gc.collect()
    finally:
        writer.close()
    return row_id, counts, {
        "strategy": "domain_then_bucket_cell_merge_v1",
        "largest_materialized_cell_rows": max_cell_rows,
        "largest_materialized_cell_bucket": max_cell_bucket,
        "whole_catalog_materialized": False,
    }


def _uid_partition_write(
    handles: list[Any], buffers: list[bytearray], uid: str, flush_bytes: int = MIB
) -> None:
    raw = uid.encode("utf-8")
    part = (zlib.crc32(raw) & 0xFFFFFFFF) & 0xFF
    buf = buffers[part]
    buf.extend(struct.pack("<I", len(raw)))
    buf.extend(raw)
    if len(buf) >= flush_bytes:
        handles[part].write(buf)
        buf.clear()


def _read_uid_partition(path: Path) -> list[bytes]:
    payload = path.read_bytes()
    view = memoryview(payload)
    out: list[bytes] = []
    pos = 0
    while pos < len(view):
        if pos + 4 > len(view):
            raise RuntimeError(f"truncated uid partition header: {path}")
        (length,) = struct.unpack_from("<I", view, pos)
        pos += 4
        if pos + length > len(view):
            raise RuntimeError(f"truncated uid partition body: {path}")
        out.append(bytes(view[pos : pos + length]))
        pos += length
    return out


def validate_catalog_files(
    domain_paths: dict[str, Path],
    expected_n: np.ndarray,
    uid_work_dir: Path,
) -> dict[str, Any]:
    """Independent streaming validation of the four section-11 invariants."""
    uid_work_dir.mkdir(parents=True, exist_ok=True)
    uid_paths = [uid_work_dir / f"uid_{i:03d}.bin" for i in range(256)]
    handles = [path.open("wb") for path in uid_paths]
    buffers = [bytearray() for _ in range(256)]
    observed = np.zeros_like(expected_n)
    expected_row = 0
    row_contiguous = True
    order_ok = True
    tar_rows = 0
    tar_hits = 0
    previous_domain_idx = -1
    try:
        for domain_idx, domain in enumerate(DOMAINS):
            pf = pq.ParquetFile(domain_paths[domain])
            columns = ["row_id", "uid", "domain", "bucket_idx"]
            if domain in TAR_DOMAINS:
                columns += ["tar_path", "tar_offset", "tar_size"]
            previous_bucket = -1
            previous_uid: str | None = None
            for batch in pf.iter_batches(
                batch_size=DEFAULT_ROW_GROUP_SIZE, columns=columns, use_threads=False
            ):
                ids = _int_np(batch.column("row_id"))
                expected_ids = np.arange(expected_row, expected_row + batch.num_rows, dtype=np.int64)
                if not np.array_equal(ids, expected_ids):
                    row_contiguous = False
                expected_row += batch.num_rows
                uids = batch.column("uid").to_pylist()
                domains = batch.column("domain").to_pylist()
                buckets = _int_np(batch.column("bucket_idx"))
                if any(str(x) != domain for x in domains) or domain_idx < previous_domain_idx:
                    order_ok = False
                previous_domain_idx = domain_idx
                for uid_raw, bucket_raw in zip(uids, buckets):
                    uid = str(uid_raw)
                    bucket = int(bucket_raw)
                    if bucket < previous_bucket or (
                        bucket == previous_bucket and previous_uid is not None and uid < previous_uid
                    ):
                        order_ok = False
                    if bucket != previous_bucket:
                        previous_uid = None
                    previous_bucket = bucket
                    previous_uid = uid
                    observed[bucket, domain_idx] += 1
                    _uid_partition_write(handles, buffers, uid)
                if domain in TAR_DOMAINS:
                    paths = batch.column("tar_path").to_pylist()
                    offsets = _int_np(batch.column("tar_offset"))
                    sizes = _int_np(batch.column("tar_size"))
                    tar_rows += batch.num_rows
                    tar_hits += sum(
                        1
                        for path, offset, size in zip(paths, offsets, sizes)
                        if path and int(offset) >= 0 and int(size) > 0
                    )
    finally:
        for i, handle in enumerate(handles):
            if buffers[i]:
                handle.write(buffers[i])
            handle.close()

    duplicate_samples: list[str] = []
    uid_rows = 0
    for path in uid_paths:
        values = _read_uid_partition(path)
        uid_rows += len(values)
        values.sort()
        for i in range(1, len(values)):
            if values[i] == values[i - 1]:
                if len(duplicate_samples) < 20:
                    duplicate_samples.append(values[i].decode("utf-8", errors="replace"))
        path.unlink(missing_ok=True)
    uid_work_dir.rmdir()

    n_equal = np.array_equal(observed, expected_n)
    uid_unique = not duplicate_samples
    result = {
        "cell_counts": {
            "pass": n_equal,
            "tolerance": 0,
            "expected": matrix_list(expected_n),
            "observed": matrix_list(observed),
            "difference_sparse": sparse_matrix(observed - expected_n),
        },
        "row_id_contiguous": {
            "pass": row_contiguous,
            "row_count": expected_row,
            "first_expected": 0,
            "last_expected": expected_row - 1 if expected_row else None,
        },
        "physical_order": {
            "pass": order_ok,
            "key": ["domain_order", "bucket_idx", "uid_lexicographic"],
        },
        "tar_locator": {
            "pass": tar_hits == tar_rows,
            "catalog_tar_rows": tar_rows,
            "exact_index_join_hits": tar_hits,
            "misses_after_exclusion": tar_rows - tar_hits,
        },
        "uid_unique": {
            "pass": uid_unique,
            "scope": "global_five_domain_exact_raw_uid",
            "rows_checked": uid_rows,
            "duplicate_samples": duplicate_samples,
            "partition_count": 256,
        },
    }
    if not all(
        (
            result["cell_counts"]["pass"],
            result["row_id_contiguous"]["pass"],
            result["physical_order"]["pass"],
            result["tar_locator"]["pass"],
            result["uid_unique"]["pass"],
        )
    ):
        raise RuntimeError(f"catalog self-check failed: {json.dumps(result, ensure_ascii=False)[:4000]}")
    return result


CAPTION_STRATEGY = {
    "anime": {
        "stored_narrow": ["key", "fetch_url", "cluster", "rating_bucket", "year"],
        "source_locator": ["caption_parquet_path", "caption_row_group", "caption_row_in_group"],
        "fetched_large": [
            "refined_description",
            "medium_description",
            "short_description",
            "tags_general",
            "tags_artist",
            "tags_character",
            "tags_copyright",
            "tags_meta",
            "foreground",
            "background",
            "style",
            "image_type",
            "camera_and_lens_effects",
        ],
        "note": "general_freq is intentionally not a row field; adapter must define deterministic shard-frequency semantics",
    },
    "monet_real": {
        "stored_narrow": [
            "caption_type",
            "caption_tier",
            "served_mask",
            "band_captioner_t1..t4",
        ],
        "source_locator": ["caption_parquet_path", "caption_row_group", "caption_row_in_group"],
        "fetched_large": [
            "caption_gemini-2.5-flash-lite",
            "caption_sharegpt4v-7b",
            "caption_internvl-3-8b",
            "caption_florence-2-large",
            "caption_original",
        ],
    },
    "gpic": {
        "stored_narrow": [
            "caption_type",
            "caption_tier",
            "served_mask",
            "band_captioner_t1..t4",
        ],
        "source_locator": ["caption_parquet_path", "caption_row_group", "caption_row_in_group"],
        "fetched_large": ["caption"],
    },
    "synthetic": {
        "stored_narrow": ["caption_kind=original"],
        "source_locator": ["caption_parquet_path", "caption_row_group", "caption_row_in_group"],
        "fetched_large": ["caption_original"],
    },
    "textatlas": {
        "stored_narrow": ["image_path"],
        "source_locator": ["caption_parquet_path", "caption_row_group", "caption_row_in_group"],
        "fetched_large": ["caption1", "caption2", "caption3", "caption4", "caption5"],
        "note": "image and caption row-group locators are independent",
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_PIPELINE_CONFIG)
    ap.add_argument(
        "--stage", choices=tuple(CANONICAL_OUTPUT_DIRS), default=None,
        help="explicit stage guard; required for stage512a",
    )
    ap.add_argument("--n-matrix", default=None)
    ap.add_argument("--bucket-report", default=None)
    ap.add_argument(
        "--floor", type=float, default=None,
        help="selected Phase-3a candidate; stage256 defaults to legacy 0.005, stage512a requires explicit value",
    )
    ap.add_argument("--tar-index-dir", default=str(DEFAULT_TAR_INDEX_DIR))
    ap.add_argument("--output-dir", default=str(CANONICAL_OUTPUT_DIR))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--row-group-size", type=int, default=DEFAULT_ROW_GROUP_SIZE)
    ap.add_argument("--source-partitions", type=int, default=DEFAULT_SOURCE_PARTITIONS)
    ap.add_argument(
        "--max-shards-per-domain",
        type=int,
        default=None,
        help="diagnostic mode: scan only the first N production-enumerated shards per domain; all products get .diag",
    )
    args = ap.parse_args()
    if args.workers <= 0:
        ap.error("--workers must be positive")
    if args.row_group_size <= 0:
        ap.error("--row-group-size must be positive")
    if args.source_partitions <= 0 or args.source_partitions > 32_767:
        ap.error("--source-partitions must be in [1,32767]")
    if args.max_shards_per_domain is not None and args.max_shards_per_domain <= 0:
        ap.error("--max-shards-per-domain must be positive")
    return args


def _merge_coverage(results: Iterable[dict[str, Any]], shards_total: dict[str, int]) -> dict[str, Any]:
    coverage = {
        d: {
            "shards_total": shards_total[d],
            "shards_scanned": 0,
            "rows_raw": 0,
            "rows_holdout_dropped": 0,
            "rows_elig_dropped": 0,
            "rows_invalid_wh_dropped": 0,
            # anchors_v1 drops rows after assignment when no anchor can serve them.
            # Always present (0 in legacy) so the coverage identity is one expression:
            #   rows_raw - holdout - elig - invalid - unservable == rows_kept
            "rows_unservable_dropped": 0,
            "rows_kept": 0,
            "eligibility_column_shards": 0,
            "eligibility_fallback_shards": 0,
            "eligibility_fallback_rows": 0,
            "eligibility_compatibility_shards": 0,
        }
        for d in DOMAINS
    }
    for result in results:
        row = coverage[result["domain"]]
        row["shards_scanned"] += 1
        for key, value in result["coverage"].items():
            row[key] += int(value)
    return coverage


def main() -> int:
    args = parse_args()
    runtime = load_runtime_config(args.config)
    if args.stage is not None and args.stage != runtime["stage"]:
        raise ValueError(
            f"--stage={args.stage} does not match config dataset.stage={runtime['stage']}"
        )
    if runtime["stage"] in ("512a", "512b") and args.stage is None:
        raise ValueError(
            f"stage{runtime['stage']} catalog build requires explicit "
            f"--stage {runtime['stage']}"
        )
    if args.floor is None:
        if runtime["stage"] in ("512a", "512b"):
            raise ValueError(
                f"stage{runtime['stage']} catalog build requires an explicit --floor "
                "chosen from its bucket report (anchors_v1 requires --floor 0)"
            )
        floor = LEGACY_STAGE256_FLOOR
    else:
        floor = float(args.floor)
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("--floor must be finite and non-negative")

    stage = runtime["stage"]
    output_dir = validate_output_dir(args.output_dir, stage)
    stats_root = CANONICAL_BUCKET_STATS_DIRS[stage]
    n_matrix_path = _validate_stage_input_path(
        Path(args.n_matrix) if args.n_matrix else stats_root / "N_matrix.parquet",
        stats_root / "N_matrix.parquet",
        "--n-matrix",
        stage,
    )
    bucket_report_path = _validate_stage_input_path(
        Path(args.bucket_report) if args.bucket_report else stats_root / "bucket_decision_report.json",
        stats_root / "bucket_decision_report.json",
        "--bucket-report",
        stage,
    )
    if not n_matrix_path.is_file():
        raise FileNotFoundError(n_matrix_path)
    if not bucket_report_path.is_file():
        raise FileNotFoundError(bucket_report_path)
    n_matrix_sha256 = sha256_file(n_matrix_path)
    bucket_report_binding = validate_bucket_report(
        bucket_report_path, runtime, n_matrix_path, n_matrix_sha256, floor
    )

    tar_index_dir = Path(args.tar_index_dir).expanduser().resolve(strict=False)
    success_marker = tar_index_dir / "_SUCCESS"
    if not success_marker.is_file():
        print(f"FATAL: tar index incomplete; missing {success_marker}", file=sys.stderr, flush=True)
        return 33

    started = time.perf_counter()
    diagnostic = args.max_shards_per_domain is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = output_dir / ("_work.diag" if diagnostic else "_work")
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)

    sampling_cfg = runtime["sampling_cfg"]
    all_shards = list_shards(sampling_cfg)
    selected_shards = {
        domain: (
            all_shards[domain][: args.max_shards_per_domain]
            if diagnostic
            else all_shards[domain]
        )
        for domain in DOMAINS
    }
    if any(not selected_shards[d] for d in DOMAINS):
        raise RuntimeError("one or more domains has no selected manifest shard")

    stats = load_stats_cache_read_only(runtime["stats_cache"], len(all_shards["monet_real"]))
    extreme_long, extreme_short = extreme_aspect_sides(
        runtime["mar_params"]["target_area"],
        runtime["mar_params"]["divisor"],
        runtime["mar_params"]["max_aspect"],
    )
    _CTX.update(
        {
            **runtime,
            "stats": stats,
            "bucket_table": build_bucket_table(
                runtime["buckets"], runtime["mar_params"]["divisor"]
            ),
            "extreme_long": extreme_long,
            "extreme_short": extreme_short,
            "fold_map": np.arange(len(runtime["buckets"]), dtype=np.int64),
            "source_partitions": args.source_partitions,
            "row_group_size": args.row_group_size,
            "textatlas_source_root": sampling_cfg.textatlas_source_root,
        }
    )

    tasks: list[tuple[str, str, int]] = []
    task_idx = 0
    for domain in DOMAINS:
        for path in selected_shards[domain]:
            tasks.append((domain, path, task_idx))
            task_idx += 1
    print(
        f"[catalog] pass1 count: tasks={len(tasks)} workers={args.workers} diagnostic={diagnostic}",
        flush=True,
    )
    ctx = mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        count_results = list(pool.map(count_manifest_shard, tasks, chunksize=1))

    domain_index = {domain: i for i, domain in enumerate(DOMAINS)}
    raw_actual = np.zeros((len(runtime["buckets"]), len(DOMAINS)), dtype=np.int64)
    for result in count_results:
        raw_actual[:, domain_index[result["domain"]]] += result["N"]

    if diagnostic:
        raw_baseline = raw_actual.copy()
        n_matrix_source = "scan_derived_diagnostic_same_shards"
        external_raw_check = {
            "pass": True,
            "tolerance": 0,
            "skipped_external_full_matrix": True,
            "reason": "diagnostic shard subset cannot be compared to the formal full-pool N_matrix",
        }
    else:
        if not n_matrix_path.is_file():
            raise FileNotFoundError(n_matrix_path)
        raw_baseline = read_n_matrix(n_matrix_path, runtime["buckets"])
        external_raw_check = {
            "pass": bool(np.array_equal(raw_actual, raw_baseline)),
            "tolerance": 0,
            "difference_sparse": sparse_matrix(raw_actual - raw_baseline),
        }
        n_matrix_source = str(n_matrix_path)
        if not external_raw_check["pass"]:
            raise RuntimeError(
                "manifest raw N differs from Phase-3a N_matrix: "
                + json.dumps(external_raw_check, ensure_ascii=False)[:4000]
            )

    kept, fold_map, folded_baseline, raw_probs = compute_floor_mapping(
        raw_baseline,
        runtime["buckets"],
        runtime["domain_mix"],
        runtime["square_idx"],
        floor,
    )
    require_identity_fold(runtime["bucket_mode"], fold_map, floor)
    reported_kept = bucket_report_binding["selected_floor_report"].get(
        "kept_bucket_indices"
    )
    # Phase-3a's informational floor=0 candidate suppresses zero-mass buckets,
    # while make_planner_input's floor=0 semantics leave the full namespace in
    # place.  Positive floors must agree exactly; floor=0 remains separately
    # governed by the C6 evidence in the report.
    if floor > 0.0 and reported_kept != sorted(kept):
        raise RuntimeError(
            "selected bucket report floor produces different kept buckets: "
            f"report={reported_kept}, catalog={sorted(kept)}"
        )
    _CTX["fold_map"] = fold_map
    print(f"[catalog] floor={floor}: kept={len(kept)}/{len(runtime['buckets'])} {sorted(kept)}", flush=True)

    build_tasks = [(d, p, i, str(work_root)) for d, p, i in tasks]
    print(f"[catalog] pass2 materialize per manifest shard: tasks={len(build_tasks)}", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        build_results = []
        for done, result in enumerate(pool.map(build_manifest_fragment, build_tasks, chunksize=1), 1):
            build_results.append(result)
            if done % 50 == 0 or done == len(build_tasks):
                print(f"[catalog] pass2 progress {done}/{len(build_tasks)}", flush=True)

    built_prejoin = np.zeros_like(folded_baseline)
    candidate_paths_by_domain: dict[str, list[str]] = defaultdict(list)
    runs_by_domain_bucket: dict[str, dict[int, list[str]]] = {
        d: defaultdict(list) for d in DOMAINS
    }
    for result in build_results:
        d = result["domain"]
        built_prejoin[:, domain_index[d]] += result["counts"]
        if result["candidate_path"]:
            candidate_paths_by_domain[d].append(result["candidate_path"])
        for bucket, path in result["cell_paths"].items():
            runs_by_domain_bucket[d][int(bucket)].append(path)
    prejoin_check = {
        "pass": bool(np.array_equal(built_prejoin, folded_baseline)),
        "tolerance": 0,
        "expected": matrix_list(folded_baseline),
        "observed": matrix_list(built_prejoin),
        "difference_sparse": sparse_matrix(built_prejoin - folded_baseline),
    }
    if not prejoin_check["pass"]:
        raise RuntimeError(f"pass2 folded N drift: {json.dumps(prejoin_check)[:4000]}")

    tar_rg_map, tar_index_summary = build_tar_rowgroup_map(tar_index_dir)
    _CTX["tar_rg_map"] = tar_rg_map
    partition_paths: dict[str, dict[int, str]] = {}
    for domain in ("monet_real", "gpic", "synthetic"):
        print(f"[catalog] repartition candidates: {domain}", flush=True)
        partition_paths[domain] = repartition_candidates(
            domain,
            candidate_paths_by_domain[domain],
            work_root,
            args.source_partitions,
            args.row_group_size,
        )

    join_tasks: list[tuple[str, int, str, str]] = []
    for domain in ("monet_real", "gpic", "synthetic"):
        for part, path in sorted(partition_paths[domain].items()):
            join_tasks.append((domain, part, path, str(work_root)))
    print(f"[catalog] source/caption/tar joins: partitions={len(join_tasks)}", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        join_results = []
        for done, result in enumerate(pool.map(join_source_partition, join_tasks, chunksize=1), 1):
            join_results.append(result)
            if done % 32 == 0 or done == len(join_tasks):
                print(f"[catalog] join progress {done}/{len(join_tasks)}", flush=True)

    bad_paths: list[str] = []
    bad_n = np.zeros_like(folded_baseline)
    joined_n = np.zeros_like(folded_baseline)
    tar_join_by_domain = {
        d: {"input_rows": 0, "hits": 0, "misses": 0, "miss_fraction": 0.0}
        for d in TAR_DOMAINS
    }
    for result in join_results:
        d = result["domain"]
        joined_n[:, domain_index[d]] += np.array(result["counts"], dtype=np.int64)
        for bucket, path in result["cell_paths"].items():
            runs_by_domain_bucket[d][int(bucket)].append(path)
        bad_paths.append(result["bad_path"])
        if d in TAR_DOMAINS:
            stats_row = tar_join_by_domain[d]
            stats_row["input_rows"] += result["input_count"]
            stats_row["hits"] += result["tar_hits"]
            stats_row["misses"] += result["tar_misses"]

    bad_tmp = work_root / product_name("bad_rows", "parquet", diagnostic)
    bad_table = consolidate_bad_rows(bad_paths, bad_tmp, args.row_group_size)
    if bad_table.num_rows:
        bad_bucket = _int_np(bad_table.column("bucket_idx"))
        bad_domain = bad_table.column("domain").to_pylist()
        for d, b in zip(bad_domain, bad_bucket):
            bad_n[int(b), domain_index[str(d)]] += 1
    for d, row in tar_join_by_domain.items():
        row["miss_fraction"] = row["misses"] / max(row["input_rows"], 1)
        row["within_0p1pct_allowance"] = row["miss_fraction"] <= TAR_MISS_MAX_FRACTION

    adjusted_expected = folded_baseline - bad_n
    for d in REAL_DOMAINS:
        observed = joined_n[:, domain_index[d]]
        expected = adjusted_expected[:, domain_index[d]]
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"joined count drift for {d}: {sparse_matrix((joined_n-adjusted_expected))[:50]}")

    n_diff_report: dict[str, Any] = {
        "schema_version": "offline_arb_catalog_n_diff_v1",
        "diagnostic_run": diagnostic,
        **_contract_fields(runtime),
        "floor": floor,
        "domains": list(DOMAINS),
        "original_planner_N": matrix_list(folded_baseline),
        "tar_index_excluded_N": matrix_list(bad_n),
        "catalog_expected_after_exclusion_N": matrix_list(adjusted_expected),
        "catalog_minus_original_sparse": sparse_matrix(adjusted_expected - folded_baseline),
        "reconciliation_pass": bool(np.array_equal(adjusted_expected + bad_n, folded_baseline)),
        "tar_join_by_domain": tar_join_by_domain,
    }
    n_diff_tmp = work_root / product_name("N_diff", "json", diagnostic)
    atomic_write_json(n_diff_tmp, n_diff_report)

    bad_output = output_dir / product_name("bad_rows", "parquet", diagnostic)
    n_diff_output = output_dir / product_name("N_diff", "json", diagnostic)
    # Publish the exclusion evidence even when the allowance is exceeded; no catalog/header is published.
    shutil.copyfile(bad_tmp, bad_output)
    shutil.copyfile(n_diff_tmp, n_diff_output)
    excessive = [d for d, row in tar_join_by_domain.items() if not row["within_0p1pct_allowance"]]
    if excessive:
        raise RuntimeError(
            f"tar-index misses exceed 0.1% for {excessive}: {json.dumps(tar_join_by_domain)}"
        )

    # Final domain/bucket merge.  Direct-domain runs were added before the joined real runs.
    domain_tmp_paths: dict[str, Path] = {}
    row_cursor = 0
    final_n = np.zeros_like(adjusted_expected)
    merge_stats: dict[str, Any] = {}
    for d in DOMAINS:
        tmp = work_root / product_name(d, "parquet", diagnostic)
        row_cursor, counts, merge_stat = merge_domain_cells(
            d,
            runs_by_domain_bucket[d],
            tmp,
            row_cursor,
            args.row_group_size,
        )
        domain_tmp_paths[d] = tmp
        final_n[:, domain_index[d]] = np.array(counts, dtype=np.int64)
        merge_stats[d] = merge_stat

    if not np.array_equal(final_n, adjusted_expected):
        raise RuntimeError(f"final merge count drift: {sparse_matrix(final_n-adjusted_expected)[:100]}")

    validations = validate_catalog_files(
        domain_tmp_paths, adjusted_expected, work_root / "uid_unique_parts"
    )
    validations["pre_exclusion_manifest_N_vs_make_planner_input"] = prejoin_check
    validations["raw_manifest_N_vs_phase3a_N_matrix"] = external_raw_check
    validations["tar_miss_allowance"] = {
        "pass": not excessive,
        "max_fraction": TAR_MISS_MAX_FRACTION,
        "by_domain": tar_join_by_domain,
        "excluded_rows": int(bad_table.num_rows),
        "reconciliation_pass": n_diff_report["reconciliation_pass"],
    }

    # Hash the exact parquet bytes before publication; os.replace does not change them.
    files: dict[str, Any] = {}
    for d in DOMAINS:
        path = domain_tmp_paths[d]
        pf = pq.ParquetFile(path)
        files[d] = {
            "name": product_name(d, "parquet", diagnostic),
            "rows": pf.metadata.num_rows,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "schema": str(pf.schema_arrow),
            "compression": "zstd",
            "row_group_target_rows": args.row_group_size,
        }
    catalog_hash_payload = {
        "format": "offline_arb_catalog_v1",
        "domains": list(DOMAINS),
        "files": {d: {"rows": files[d]["rows"], "sha256": files[d]["sha256"]} for d in DOMAINS},
        "row_order": ["domain", "bucket_idx", "uid"],
        "floor": floor,
        "bucket_table": [list(x) for x in runtime["buckets"]],
        "kept_bucket_indices": sorted(kept),
    }
    bucket_table_hash = canonical_sha256([list(x) for x in runtime["buckets"]])
    assignment_payload = assignment_hash_payload(
        runtime,
        bucket_table_hash=bucket_table_hash,
        floor=floor,
        kept=kept,
        fold_map=fold_map,
    )
    assignment_hash = canonical_sha256(assignment_payload)
    catalog_hash = canonical_sha256(catalog_hash_payload)
    coverage_summary = _merge_coverage(
        build_results, {d: len(all_shards[d]) for d in DOMAINS}
    )
    anchors_mode_run = runtime["bucket_mode"] == anchors_v1.BUCKET_MODE
    # In anchors mode no elig_* column is read, so the per-domain column/fallback shard
    # bookkeeping describes a rule that was never applied and cannot be a gate.
    eligibility_source_pass = anchors_mode_run or all(
        (
            coverage_summary[domain]["eligibility_fallback_shards"]
            == coverage_summary[domain]["shards_scanned"]
            and coverage_summary[domain]["eligibility_column_shards"] == 0
        )
        if domain in runtime["eligibility_fallback_domains"]
        else (
            coverage_summary[domain]["eligibility_column_shards"]
            == coverage_summary[domain]["shards_scanned"]
            and coverage_summary[domain]["eligibility_fallback_shards"] == 0
        )
        for domain in DOMAINS
        if not (
            runtime["stage"] == "256"
            and domain == "anime"
            and runtime["anime_elig_always_true"]
        )
    )
    if not eligibility_source_pass:
        raise RuntimeError(f"catalog eligibility source audit failed: {coverage_summary}")

    header: dict[str, Any] = {
        "schema_version": "offline_arb_catalog_v1",
        "generated_at": utc_now(),
        "host": socket.gethostname(),
        "diagnostic_run": diagnostic,
        "max_shards_per_domain": args.max_shards_per_domain,
        **_contract_fields(runtime),
        "catalog_hash": catalog_hash,
        "bucket_table_hash": bucket_table_hash,
        "assignment_hash": assignment_hash,
        "floor": floor,
        "domains": list(DOMAINS),
        "domain_order": list(DOMAINS),
        "square_only_domains": list(runtime["square_only"]),
        "buckets": [
            {"idx": i, "h": h, "w": w, "kept": i in kept}
            for i, (h, w) in enumerate(runtime["buckets"])
        ],
        "square_bucket_idx": runtime["square_idx"],
        "kept_bucket_indices": sorted(kept),
        "fold_map": fold_map.tolist(),
        "row_scheme": "virtual_catalog_v1_materialized",
        # Generated, not a literal: "bucket_idx_0_to_94" was already wrong for the
        # 361-bucket stage512a catalogs in production and would be doubly wrong at 13.
        "row_order": [
            "domain_order",
            f"bucket_idx_0_to_{len(runtime['buckets']) - 1}",
            "uid_lexicographic",
        ],
        "row_count": int(row_cursor),
        "row_group_target_rows": args.row_group_size,
        "files": files,
        "N": {
            "source": n_matrix_source,
            "raw_before_floor": matrix_list(raw_baseline),
            "folded_original_planner": matrix_list(folded_baseline),
            "tar_index_excluded": matrix_list(bad_n),
            "materialized_after_exclusion": matrix_list(final_n),
        },
        "coverage": coverage_summary,
        "eligibility_audit": (
            {
                # anchors_v1 reads neither the elig_* column nor the short-side rule, so
                # reporting them here would be a false record of what was applied.
                "pass": eligibility_source_pass,
                "bucket_mode": anchors_v1.BUCKET_MODE,
                "column": None,
                "rule": (
                    "source_area_gate"
                    + (
                        f">={runtime['mar_params']['area_gate']}"
                        if runtime["mar_params"]["area_gate"] is not None
                        else "=disabled"
                    )
                    + " AND a feasible anchor exists (w>=W_k and h>=H_k) with"
                    f" crop_loss<={runtime['mar_params']['crop_loss_max']}"
                ),
                "fallback_domain_whitelist": [],
            }
            if runtime["bucket_mode"] == anchors_v1.BUCKET_MODE
            else {
                "pass": eligibility_source_pass,
                "column": runtime["eligibility"],
                "fallback_rule": f"min(width,height)>={runtime['image_size']}",
                "fallback_domain_whitelist": sorted(
                    runtime["eligibility_fallback_domains"]
                ),
            }
        ),
        "self_checks": validations,
        "caption_strategy": CAPTION_STRATEGY,
        "sort_and_memory": {
            "manifest_workers": args.workers,
            "source_hash_partitions": args.source_partitions,
            "levels": ["per_manifest_or_source_partition_cell_run", "domain_then_bucket_final_merge"],
            "whole_table_loaded": False,
            "per_domain": merge_stats,
        },
        "inputs": {
            "pipeline_config": str(Path(args.config).resolve()),
            "pipeline_config_sha256": sha256_file(Path(args.config)),
            "sampling_config": runtime["sampling_path"],
            "sampling_config_sha256": sha256_file(Path(runtime["sampling_path"])),
            "N_matrix": str(n_matrix_path),
            "N_matrix_sha256": n_matrix_sha256,
            "bucket_decision_report": bucket_report_binding,
            "tar_index": tar_index_summary,
            "multi_ar_params": runtime["mar_params"],
            "domain_mix": runtime["domain_mix"],
        },
        "reports": {
            "bad_rows": product_name("bad_rows", "parquet", diagnostic),
            "N_diff": product_name("N_diff", "json", diagnostic),
        },
        "duration_seconds": time.perf_counter() - started,
    }

    # Publish domain files first, remove all scratch state, and publish the
    # header last.  A visible header therefore means every referenced product
    # passed validation and the builder returned through every fallible cleanup
    # step before the publication marker became visible.
    for d in DOMAINS:
        final_path = output_dir / product_name(d, "parquet", diagnostic)
        os.replace(domain_tmp_paths[d], final_path)
    shutil.rmtree(work_root)
    header_path = output_dir / product_name("header", "json", diagnostic)
    atomic_write_json(header_path, header)

    print("==== OFFLINE_ARB PHASE5 CATALOG SELF-CHECK ====")
    print(
        json.dumps(
            {
                "catalog_hash": catalog_hash,
                "row_count": row_cursor,
                "kept_bucket_indices": sorted(kept),
                "bad_rows": int(bad_table.num_rows),
                "tar_join_by_domain": tar_join_by_domain,
                "self_checks": {k: v.get("pass") for k, v in validations.items() if isinstance(v, dict)},
                "header": str(header_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # authoritative job: fail fast, but retain an actionable traceback
        print(f"FATAL phase5 catalog: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(2)
