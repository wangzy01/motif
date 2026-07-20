#!/usr/bin/env python3
"""Build the create-once Stage256 semantic sampling overlay.

The physical catalog is immutable.  This program only emits, for every
``(bucket_idx, domain)`` cell, a sorted list of physical ``row_id`` values and
their conditional sampling weights.  It restores the useful semantics of the
streaming sampler without restoring streaming itself:

* anime: at least one *stripped, non-empty* NL field, exact legacy priority,
  and ``compute_repeat_counts(max_repeat=4, extra_frac=.25)``;
* MONET/GPIC: ``base_weight > 0`` and ``served_mask > 0``, with the exact
  band-first within-domain row marginal;
* synthetic: positive ``base_weight``;
* TextAtlas: uniform;
* eleven known-bad physical rows are always excluded.

This is deliberately a CPU/JCC program.  It scans tens of GB and must not be
run on a login host for production data.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


FORMAT = "offline_arb_semantic_overlay_v1"
FORMAT_VERSION = 1
DOMAINS = ("anime", "monet_real", "gpic", "synthetic", "textatlas")
BANDS = ("T1", "T2", "T3", "T4")
NL_COLUMNS = ("refined_description", "medium_description", "short_description")
DEFAULT_EXCLUDED = {
    "monet_real": (
        18907181, 23881582, 23881911, 31217524,
        31221132, 32217542, 35492824, 40078044,
    ),
    "textatlas": (94682834, 93569671, 93606908),
}

# Exact constants from training/dataline/manifest_index.py.  The live source
# files are separately SHA-bound in header.json, so a reviewer can compare this
# frozen implementation byte-for-byte with the intended implementation.
CLEAN_BG_TAGS = frozenset({
    "simple_background", "white_background", "solid_background",
    "gradient_background", "transparent_background",
})
PLATFORM_USERNAME_TAGS = frozenset({
    "twitter_username", "patreon_username", "weibo_username", "instagram_username",
    "deviantart_username", "pixiv_username", "fanbox_username", "facebook_username",
    "tumblr_username", "gumroad_username", "bluesky_username", "subscribestar_username",
    "artstation_username", "lofter_username", "skeb_username", "mihoyo_username",
    "ko-fi_username", "newgrounds_username",
})
WATERMARK_SIGNATURE_TAGS = frozenset({
    "watermark", "sample_watermark", "weibo_watermark", "watermark_grid",
    "miyoushe_watermark", "official_watermark", "character_watermark", "alpha_signature",
    "signature", "artist_name", "artist_signature", "character_signature", "dated",
    "copyright_notice", "copyright_name", "too_many_watermarks",
})
LOGO_TAGS = frozenset({
    "logo", "artist_logo", "patreon_logo", "twitter_logo", "copyright_logo",
    "instagram_logo", "facebook_logo", "deviantart_logo", "pixiv_logo",
    "twitter_x_logo", "youtube_logo",
})
VISIBLE_MARK_TAGS = PLATFORM_USERNAME_TAGS | WATERMARK_SIGNATURE_TAGS | LOGO_TAGS
CLEAN_QUALITY_Z_MIN = 0.4
CLEAN_YOLO_AREA_MAX = 0.005
YOLO_COLUMN = "skytnt_yolo_textblock_detection_10_added_text_area_nms"
ANIME_SOURCE_COLUMNS = (
    "ct_weight", "ct_lanes", "quality_z", "hps_v3_score_qwen3vl",
    "risk_bucket", "tags_general", YOLO_COLUMN, *NL_COLUMNS,
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    try:
        with tmp.open("wb") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _as_float(array: pa.Array | pa.ChunkedArray, null_value: float) -> np.ndarray:
    return np.asarray(
        pc.fill_null(array, pa.scalar(null_value, type=array.type)).to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )


def _stripped_nonempty(array: pa.Array | pa.ChunkedArray) -> np.ndarray:
    # utf8_trim_whitespace is the intentional 7.18 strengthening over the old
    # bool-only has_nl flag.  Whitespace-only strings can never fall back to tags.
    trimmed = pc.utf8_trim_whitespace(pc.fill_null(array, ""))
    return np.asarray(
        pc.greater(pc.utf8_length(trimmed), 0).to_numpy(zero_copy_only=False), dtype=bool,
    )


def _lane_bonus(lanes: Iterable[str] | None, exact: dict[str, float], prefixes: dict[str, float]) -> float:
    if not lanes:
        return 0.0
    total = 0.0
    for lane in lanes:
        total += exact.get(lane, 0.0)
        for prefix, bonus in prefixes.items():
            if lane.startswith(prefix):
                total += bonus
    return total


def _composition_clean(tags: Iterable[str] | None, yolo: float, quality_z: float) -> bool:
    values = set(tags or ())
    return (
        bool(values & CLEAN_BG_TAGS)
        and not bool(values & VISIBLE_MARK_TAGS)
        and yolo < CLEAN_YOLO_AREA_MAX
        and quality_z >= CLEAN_QUALITY_Z_MIN
    )


def compute_repeat_counts(
    priority: np.ndarray, max_repeat: int, extra_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact copy of training.dataline.sampler.compute_repeat_counts semantics."""
    n = len(priority)
    counts = np.ones(n, dtype=np.int64)
    if n == 0 or max_repeat <= 1:
        return counts
    score = np.maximum(priority.astype(np.float64) - 1.0, 0.0)
    if not np.any(score > 0):
        return counts
    budget = min(int(round(n * extra_frac)), n * (max_repeat - 1))
    remaining = budget
    while remaining > 0:
        eligible = np.flatnonzero((counts < max_repeat) & (score > 0))
        if eligible.size == 0:
            break
        probs = score[eligible].astype(np.float64, copy=True)
        total = probs.sum()
        if total <= 0:
            break
        probs /= total
        take = min(remaining, int(eligible.size))
        chosen = rng.choice(eligible, size=take, replace=False, p=probs)
        counts[chosen] += 1
        remaining -= take
    return counts


def anime_cell_seed(overlay_id: str, root_seed: int, bucket_idx: int) -> int:
    """Stable, namespace-bound uint64 seed for one anime bucket/cell."""
    payload = f"{overlay_id}\x00{root_seed}\x00{bucket_idx}\x00anime".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def anime_params(sampling: dict[str, Any]) -> dict[str, Any]:
    raw = sampling["anime"]["sampling"]
    priority = raw["priority"]
    quality = priority["quality_bonus"]
    caps = raw["lane_caps"]["risk_bucket_downweight"]
    return {
        "max_priority": float(priority["max_priority"]),
        "max_repeat": int(priority["max_repeat"]),
        "extra_frac": float(priority.get("extra_frac", 0.25)),
        "quality_value": float(quality["value"]),
        # These thresholds are explicit in the authoritative `when` and in
        # manifest_index.py; freeze them rather than trying to parse prose.
        "quality_qz_min": 1.5,
        "quality_hps_min": 0.0,
        "lane_bonus": {k: float(v) for k, v in priority["lane_bonus"].items() if not k.startswith("_")},
        "lane_prefix_bonus": {k: float(v) for k, v in priority["lane_prefix_bonus"].items()},
        "clean_bonus": float(priority["composition_clean_bonus"]["value"]),
        "cap_buckets": frozenset(caps["buckets"]),
        "cap_multiplier": float(caps["priority_multiplier"]),
    }


def _process_anime_source(task: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    """Worker: scan one source parquet, emitting two mmap-friendly npy arrays."""
    source_s, work_s, params = task
    source = Path(source_s)
    work = Path(work_s)
    parquet = pq.ParquetFile(source)
    n = parquet.metadata.num_rows
    priority_path = work / f"{source.stem}.priority.npy"
    nl_path = work / f"{source.stem}.nl.npy"
    priority_out = np.lib.format.open_memmap(priority_path, mode="w+", dtype=np.float64, shape=(n,))
    nl_out = np.lib.format.open_memmap(nl_path, mode="w+", dtype=np.bool_, shape=(n,))
    offsets = [0]
    cursor = 0
    for rg in range(parquet.num_row_groups):
        table = parquet.read_row_group(rg, columns=list(ANIME_SOURCE_COLUMNS), use_threads=False)
        m = table.num_rows
        if table["ct_weight"].null_count:
            # The authoritative vector path converts directly to float64 and
            # therefore does not define a null fallback.
            raise ValueError(f"null ct_weight in {source} row group {rg}")
        ct_weight = np.asarray(table["ct_weight"].to_numpy(zero_copy_only=False), dtype=np.float64)
        qz = _as_float(table["quality_z"], 0.0)
        hps = _as_float(table["hps_v3_score_qwen3vl"], 0.0)
        yolo = _as_float(table[YOLO_COLUMN], math.inf)
        lanes = table["ct_lanes"].to_pylist()
        tags = table["tags_general"].to_pylist()
        risks = table["risk_bucket"].to_pylist()

        value = ct_weight.copy()
        value += np.where(
            (qz >= params["quality_qz_min"]) & (hps >= params["quality_hps_min"]),
            params["quality_value"], 0.0,
        )
        value += np.fromiter(
            (_lane_bonus(x, params["lane_bonus"], params["lane_prefix_bonus"]) for x in lanes),
            dtype=np.float64, count=m,
        )
        clean = np.fromiter(
            (
                _composition_clean(tag_row, yolo[i], qz[i])
                for i, tag_row in enumerate(tags)
            ), dtype=bool, count=m,
        )
        value += np.where(clean, params["clean_bonus"], 0.0)
        value = np.clip(value, 1.0, params["max_priority"])
        capped = np.fromiter((x in params["cap_buckets"] for x in risks), dtype=bool, count=m)
        value = np.where(capped, value * params["cap_multiplier"], value)
        has_nl = np.zeros(m, dtype=bool)
        for column in NL_COLUMNS:
            has_nl |= _stripped_nonempty(table[column])

        priority_out[cursor:cursor + m] = value
        nl_out[cursor:cursor + m] = has_nl
        cursor += m
        offsets.append(cursor)
    priority_out.flush()
    nl_out.flush()
    if cursor != n:
        raise RuntimeError(f"anime source short scan: {source}: {cursor}/{n}")
    return {
        "source_path": str(source),
        "rows": n,
        "row_group_offsets": offsets,
        "priority_path": str(priority_path),
        "nl_path": str(nl_path),
    }


def scan_anime_sources(
    source_manifest: dict[str, Any], work: Path, params: dict[str, Any], workers: int,
) -> dict[str, dict[str, Any]]:
    entries = source_manifest.get("files")
    if not isinstance(entries, dict) or len(entries) != 17:
        raise ValueError("anime source manifest must bind exactly 17 files")
    tasks: list[tuple[str, str, dict[str, Any]]] = []
    for name, entry in sorted(entries.items()):
        source = Path(str(entry.get("source_path", "")))
        if source.name != name or not source.is_file():
            raise FileNotFoundError(f"bad/missing anime source: {name}: {source}")
        if source.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"anime source size drift: {source}")
        tasks.append((str(source), str(work), params))
    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        futures = [pool.submit(_process_anime_source, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results[result["source_path"]] = result
            print(f"[anime-source] {Path(result['source_path']).name}: {result['rows']:,}", flush=True)
    return results


def _arrow_numpy(column: pa.ChunkedArray, dtype: np.dtype) -> np.ndarray:
    return np.asarray(column.combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype)


def load_anime_catalog(
    catalog_path: Path, source_arrays: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = [
        "row_id", "bucket_idx", "caption_parquet_path",
        "caption_row_group", "caption_row_in_group",
    ]
    table = pq.read_table(catalog_path, columns=columns, use_threads=True)
    row_id = _arrow_numpy(table["row_id"], np.int64)
    bucket = _arrow_numpy(table["bucket_idx"], np.int16)
    row_group = _arrow_numpy(table["caption_row_group"], np.int32)
    row_in_group = _arrow_numpy(table["caption_row_in_group"], np.int32)
    if row_id.size and (np.any(np.diff(row_id) != 1) or row_id[0] != 0):
        raise ValueError("anime catalog physical row_id is not contiguous from zero")

    encoded = pc.dictionary_encode(table["caption_parquet_path"].combine_chunks())
    codes = np.asarray(encoded.indices.to_numpy(zero_copy_only=False), dtype=np.int32)
    dictionary = encoded.dictionary.to_pylist()  # exactly 17 objects, never 17M
    priority = np.empty(len(row_id), dtype=np.float64)
    has_nl = np.empty(len(row_id), dtype=bool)
    covered = np.zeros(len(row_id), dtype=bool)
    for code, source in enumerate(dictionary):
        if source not in source_arrays:
            raise ValueError(f"catalog locator source not in frozen manifest: {source}")
        mask = codes == code
        meta = source_arrays[source]
        offsets = np.asarray(meta["row_group_offsets"], dtype=np.int64)
        rg = row_group[mask]
        ri = row_in_group[mask]
        if np.any(rg < 0) or np.any(rg + 1 >= len(offsets)):
            raise ValueError(f"row-group locator out of range: {source}")
        pos = offsets[rg] + ri
        if np.any(ri < 0) or np.any(pos >= offsets[rg + 1]):
            raise ValueError(f"row-in-group locator out of range: {source}")
        src_priority = np.load(meta["priority_path"], mmap_mode="r")
        src_nl = np.load(meta["nl_path"], mmap_mode="r")
        priority[mask] = src_priority[pos]
        has_nl[mask] = src_nl[pos]
        covered[mask] = True
    if not np.all(covered):
        raise RuntimeError("anime catalog locator join was not total")
    return row_id, bucket, priority, has_nl


def load_exclusions(path: Path | None) -> dict[str, tuple[int, ...]]:
    if path is None:
        result = {domain: tuple(rows) for domain, rows in DEFAULT_EXCLUDED.items()}
    else:
        value, _ = json_snapshot(path)
        result = {domain: tuple(int(x) for x in rows) for domain, rows in value.items()}
    expected = {domain: set(rows) for domain, rows in DEFAULT_EXCLUDED.items()}
    actual = {domain: set(result.get(domain, ())) for domain in expected}
    if actual != expected or set(result) - set(expected):
        raise ValueError(f"production exclusion set must be exact: expected={expected}, actual={actual}")
    return {domain: tuple(sorted(rows)) for domain, rows in result.items()}


def normalized(values: np.ndarray, label: str) -> np.ndarray:
    total = float(values.sum(dtype=np.float64))
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"non-positive/invalid cell weight: {label}: {total}")
    result = values.astype(np.float64, copy=False) / total
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(f"invalid normalized weights: {label}")
    return result


def deterministic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a byte-deterministic, uncompressed NPZ without wall-clock ZIP metadata."""
    order = [name for name in ("row_id", "weight", "repeat", "priority") if name in arrays]
    if set(order) != set(arrays):
        raise ValueError(f"unexpected NPZ arrays: {sorted(set(arrays) - set(order))}")
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in order:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            with archive.open(info, mode="w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, arrays[name], allow_pickle=False)


def real_domain_arrays(
    table: pa.Table, domain: str, band_mix: np.ndarray, exclusions: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    row_id = _arrow_numpy(table["row_id"], np.int64)
    bucket = _arrow_numpy(table["bucket_idx"], np.int16)
    base = _arrow_numpy(table["base_weight"], np.float64)
    if domain in ("monet_real", "gpic"):
        served = _arrow_numpy(table["served_mask"], np.uint8)
        keep = np.isfinite(base) & (base > 0) & (served > 0)
    else:
        served = np.zeros(len(row_id), dtype=np.uint8)
        keep = np.isfinite(base) & (base > 0)
    if exclusions:
        keep &= ~np.isin(row_id, np.fromiter(exclusions, dtype=np.int64))

    output_weight = np.empty(int(keep.sum()), dtype=np.float64)
    kept_row = row_id[keep]
    kept_bucket = bucket[keep]
    kept_base = base[keep]
    kept_served = served[keep]
    metrics: dict[int, dict[str, Any]] = {}
    for bidx in np.unique(bucket):
        all_cell = bucket == bidx
        cell = kept_bucket == bidx
        n_selected = int(cell.sum())
        if n_selected == 0:
            metrics[int(bidx)] = {"catalog_rows": int(all_cell.sum()), "selected_rows": 0}
            continue
        if domain in ("monet_real", "gpic"):
            marginal = np.zeros(n_selected, dtype=np.float64)
            active_mass = 0.0
            for band in range(4):
                serves = ((kept_served[cell] >> band) & 1).astype(bool)
                denom = float(kept_base[cell][serves].sum(dtype=np.float64))
                if denom > 0:
                    marginal[serves] += band_mix[band] * kept_base[cell][serves] / denom
                    active_mass += float(band_mix[band])
            if active_mass <= 0:
                raise ValueError(f"no active band in {domain} bucket {bidx}")
            weight = normalized(marginal, f"{domain}:{bidx}")
        else:
            weight = normalized(kept_base[cell], f"{domain}:{bidx}")
        output_weight[cell] = weight
        metrics[int(bidx)] = {
            "catalog_rows": int(all_cell.sum()),
            "selected_rows": n_selected,
            "weight_sum": float(weight.sum(dtype=np.float64)),
        }
    return kept_row, kept_bucket, output_weight, np.ones(len(kept_row), dtype=np.uint8), metrics


def textatlas_arrays(
    table: pa.Table, exclusions: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    row_id = _arrow_numpy(table["row_id"], np.int64)
    bucket = _arrow_numpy(table["bucket_idx"], np.int16)
    keep = ~np.isin(row_id, np.fromiter(exclusions, dtype=np.int64))
    rows, buckets = row_id[keep], bucket[keep]
    weights = np.empty(len(rows), dtype=np.float64)
    metrics: dict[int, dict[str, Any]] = {}
    for bidx in np.unique(bucket):
        all_cell = bucket == bidx
        cell = buckets == bidx
        n = int(cell.sum())
        if n:
            weights[cell] = 1.0 / n
        metrics[int(bidx)] = {
            "catalog_rows": int(all_cell.sum()), "selected_rows": n,
            "weight_sum": float(weights[cell].sum()) if n else 0.0,
        }
    return rows, buckets, weights, np.ones(len(rows), dtype=np.uint8), metrics


def write_cells(
    root: Path, domain: str, row_id: np.ndarray, bucket: np.ndarray,
    weight: np.ndarray, repeat: np.ndarray, priority: np.ndarray | None,
    cells: dict[str, Any], n_select: dict[str, int], metrics: dict[int, dict[str, Any]],
) -> None:
    if not (len(row_id) == len(bucket) == len(weight) == len(repeat)):
        raise ValueError(f"array length mismatch: {domain}")
    for bidx in sorted(int(x) for x in np.unique(bucket)):
        mask = bucket == bidx
        rows = row_id[mask].astype(np.int64, copy=False)
        weights = weight[mask].astype(np.float64, copy=False)
        repeats = repeat[mask].astype(np.uint8, copy=False)
        if len(rows) == 0:
            continue
        if np.any(np.diff(rows) <= 0):
            raise ValueError(f"cell row_id is not strictly increasing: {bidx}:{domain}")
        if abs(float(weights.sum(dtype=np.float64)) - 1.0) > 2e-12:
            raise ValueError(f"cell weights do not sum to one: {bidx}:{domain}")
        rel = Path("cells") / f"b{bidx:03d}_{domain}.npz"
        path = root / rel
        arrays: dict[str, np.ndarray] = {"row_id": rows, "weight": weights, "repeat": repeats}
        if priority is not None:
            arrays["priority"] = priority[mask].astype(np.float64, copy=False)
        deterministic_savez(path, arrays)
        key = f"{bidx}:{domain}"
        entry = {
            "path": rel.as_posix(), "sha256": file_sha256(path),
            "bytes": path.stat().st_size, "rows": len(rows),
            "row_id_min": int(rows[0]), "row_id_max": int(rows[-1]),
            "weight_sum": float(weights.sum(dtype=np.float64)),
            "repeat_sum": int(repeats.sum(dtype=np.int64)),
            "arrays": {name: str(value.dtype) for name, value in arrays.items()},
        }
        entry.update(metrics.get(bidx, {}))
        cells[key] = entry
        n_select[key] = len(rows)


def bind_input_file(path: Path, declared_sha: str | None = None) -> dict[str, Any]:
    digest = file_sha256(path)
    if declared_sha is not None and digest != declared_sha:
        raise ValueError(f"input SHA drift: {path}: {digest}/{declared_sha}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"create-once output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging collision: {staging}")
    staging.mkdir()
    (staging / "cells").mkdir()
    work = staging / "_anime_work"
    work.mkdir()
    try:
        catalog_header, catalog_header_sha = json_snapshot(args.catalog_header)
        sidecar_header, sidecar_header_sha = json_snapshot(args.caption_sidecar_header)
        source_manifest, source_manifest_sha = json_snapshot(args.anime_source_manifest)
        sampling, sampling_sha = json_snapshot(args.sampling_config)
        if catalog_header.get("schema_version") != "offline_arb_catalog_v1":
            raise ValueError("unsupported catalog header")
        if sidecar_header.get("schema_version") != "offline_arb_caption_sidecar_v1":
            raise ValueError("unsupported caption sidecar header")
        if source_manifest.get("schema_version") != "offline_arb_anime_caption_source_v1":
            raise ValueError("unsupported anime source manifest")
        catalog_hash = str(catalog_header["catalog_hash"])
        if sidecar_header.get("catalog_hash") != catalog_hash or source_manifest.get("catalog_hash") != catalog_hash:
            raise ValueError("catalog/sidecar/anime-source binding mismatch")
        if sidecar_header.get("catalog_header_sha256") != catalog_header_sha:
            raise ValueError("caption sidecar binds a different catalog header")
        exclusions = load_exclusions(args.exclusions_json)
        params = anime_params(sampling)
        if params["max_repeat"] != 4 or params["extra_frac"] != 0.25:
            raise ValueError(f"Stage256 contract requires max_repeat=4/extra_frac=.25: {params}")
        route = sampling["anime"]["caption"]["stage_route_weights_has_nl"]["256"]
        if float(route.get("tag_medium", -1)) != 0:
            raise ValueError("sampling config Stage256 tag_medium route is not zero")

        catalog_root = args.catalog_root.resolve()
        catalog_inputs: dict[str, Any] = {}
        for domain in DOMAINS:
            path = catalog_root / f"{domain}.parquet"
            declared = catalog_header["files"][domain]
            catalog_inputs[domain] = bind_input_file(path, declared["sha256"])

        source_arrays = scan_anime_sources(source_manifest, work, params, args.workers)
        anime_row, anime_bucket, anime_priority, anime_nl = load_anime_catalog(
            catalog_root / "anime.parquet", source_arrays,
        )
        selected = anime_nl
        selected_priority = anime_priority[selected]
        selected_rows = anime_row[selected]
        selected_buckets = anime_bucket[selected]
        # Repeat allocation is deliberately local to each physical
        # (bucket,anime) cell.  This makes every cell independently frozen and
        # prevents the population of one aspect ratio from affecting another.
        repeats = np.empty(len(selected_rows), dtype=np.uint8)
        anime_cell_seeds: dict[str, int] = {}
        for bidx in np.unique(selected_buckets):
            cell = selected_buckets == bidx
            cell_seed = anime_cell_seed(args.overlay_id, args.seed, int(bidx))
            anime_cell_seeds[str(int(bidx))] = cell_seed
            repeats[cell] = compute_repeat_counts(
                selected_priority[cell], params["max_repeat"], params["extra_frac"],
                np.random.default_rng(cell_seed),
            ).astype(np.uint8)
        anime_weights = np.empty(len(selected_rows), dtype=np.float64)
        anime_metrics: dict[int, dict[str, Any]] = {}
        for bidx in np.unique(anime_bucket):
            all_cell = anime_bucket == bidx
            cell = selected_buckets == bidx
            if np.any(cell):
                anime_weights[cell] = normalized(repeats[cell].astype(np.float64), f"anime:{bidx}")
            anime_metrics[int(bidx)] = {
                "catalog_rows": int(all_cell.sum()), "selected_rows": int(cell.sum()),
                "strict_nl_rows": int(cell.sum()), "tag_only_rows": 0,
                "repeat_sum": int(repeats[cell].sum(dtype=np.int64)),
            }

        cells: dict[str, Any] = {}
        n_select: dict[str, int] = {}
        write_cells(
            staging, "anime", selected_rows, selected_buckets, anime_weights,
            repeats, selected_priority, cells, n_select, anime_metrics,
        )
        del anime_row, anime_bucket, anime_priority, anime_nl, selected_priority

        band_mix = np.array([float(sampling["band_mix_real"][x]) for x in BANDS], dtype=np.float64)
        if not np.isclose(band_mix.sum(), 1.0):
            raise ValueError(f"band_mix_real does not sum to one: {band_mix}")
        for domain in ("monet_real", "gpic", "synthetic"):
            cols = ["row_id", "bucket_idx", "base_weight"]
            if domain != "synthetic":
                cols.append("served_mask")
            table = pq.read_table(catalog_root / f"{domain}.parquet", columns=cols, use_threads=True)
            arrays = real_domain_arrays(
                table, domain, band_mix, set(exclusions.get(domain, ())),
            )
            write_cells(staging, domain, *arrays[:4], None, cells, n_select, arrays[4])
            del table, arrays

        table = pq.read_table(
            catalog_root / "textatlas.parquet", columns=["row_id", "bucket_idx"], use_threads=True,
        )
        arrays = textatlas_arrays(table, set(exclusions["textatlas"]))
        write_cells(staging, "textatlas", *arrays[:4], None, cells, n_select, arrays[4])
        del table, arrays

        active_buckets = sorted({int(key.split(":", 1)[0]) for key in cells})
        bucket_entries = catalog_header.get("buckets")
        if not isinstance(bucket_entries, list) or len(bucket_entries) != 95:
            raise ValueError("catalog header must provide the complete 95-row bucket table")
        bucket_shapes: list[list[int]] = []
        for expected_idx, entry in enumerate(bucket_entries):
            if int(entry.get("idx", -1)) != expected_idx:
                raise ValueError(f"catalog bucket table index drift at {expected_idx}")
            bucket_shapes.append([int(entry["h"]), int(entry["w"])])
        n_select_matrix = [
            [int(n_select.get(f"{bucket_idx}:{domain}", 0)) for domain in DOMAINS]
            for bucket_idx in range(len(bucket_shapes))
        ]
        excluded_flat = sorted(x for rows in exclusions.values() for x in rows)
        header: dict[str, Any] = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "overlay_id": args.overlay_id,
            "catalog_hash": catalog_hash,
            "catalog_header_sha256": catalog_header_sha,
            "sampling_config_sha256": sampling_sha,
            "caption_sidecar_header_sha256": sidecar_header_sha,
            "anime_source_manifest_sha256": source_manifest_sha,
            "anime_source_content_manifest_sha256": source_manifest.get("manifest_sha256"),
            "domains": list(DOMAINS),
            "buckets": bucket_shapes,
            "active_bucket_indices": active_buckets,
            "N_select": n_select_matrix,
            "N_select_by_cell": n_select,
            "excluded_row_ids": excluded_flat,
            "excluded_row_ids_by_domain": {k: list(v) for k, v in exclusions.items()},
            "cells": cells,
            "seed": int(args.seed),
            "anime_contract": {
                "eligibility": "OR(strip(refined_description),strip(medium_description),strip(short_description)) non-empty",
                "eligibility_version": "stage256_stripped_nonempty_nl_v1",
                "tag_only_exposure": 0,
                "priority_formula": "manifest_index_anime_priority_v1_exact",
                "repeat_formula": "sampler_compute_repeat_counts_v1_exact",
                "max_repeat": params["max_repeat"], "extra_frac": params["extra_frac"],
                "repeat_scope": "independent (bucket_idx,anime) cell",
                "cell_seed_algorithm": "uint64_be(sha256(overlay_id\\0root_seed\\0bucket_idx\\0anime)[:8])",
                "cell_seeds": anime_cell_seeds,
            },
            "semantic_assertions": {
                "anime_all_rows_strip_nonempty_nl": True,
                "anime_tag_only_rows": 0,
            },
            "weight_contract": {
                "normalization": "strictly-positive float64; sum=1 independently in every cell",
                "monet_real": "conditional band-first row marginal from band_mix_real, served_mask, base_weight",
                "gpic": "conditional band-first row marginal from band_mix_real, served_mask, base_weight",
                "synthetic": "base_weight normalized within cell",
                "textatlas": "uniform within cell",
                "anime": "integer repeat normalized within cell",
            },
            "bindings": {
                "catalog_header": bind_input_file(args.catalog_header),
                "caption_sidecar_header": bind_input_file(args.caption_sidecar_header),
                "anime_source_manifest": bind_input_file(args.anime_source_manifest),
                "sampling_config": bind_input_file(args.sampling_config),
                "catalog_files": catalog_inputs,
                "manifest_index_py": bind_input_file(args.manifest_index_py) if args.manifest_index_py else None,
                "sampler_py": bind_input_file(args.sampler_py) if args.sampler_py else None,
            },
        }
        header["content_sha256"] = canonical_sha256(header)
        atomic_json(staging / "header.json", header)
        header_file_sha = file_sha256(staging / "header.json")
        success = {
            "status": "PASS", "format": FORMAT, "overlay_id": args.overlay_id,
            "header_sha256": header_file_sha,
            "content_sha256": header["content_sha256"],
            "cell_count": len(cells), "selected_rows": sum(n_select.values()),
            "eligible_anime_rows": int(len(selected_rows)),
            "tag_only_exposure": 0,
        }
        success["success_sha256"] = canonical_sha256(success)
        atomic_json(staging / "_SUCCESS", success)
        shutil.rmtree(work)
        fsync_dir(staging)
        os.rename(staging, output)
        fsync_dir(output.parent)
        print(json.dumps({"status": "PASS", "output": str(output), **success}, sort_keys=True), flush=True)
        return success
    except BaseException:
        # Preserve partial data for forensics, but never publish it under the
        # canonical create-once leaf and never emit _SUCCESS early.
        print(f"partial staging preserved at {staging}", file=sys.stderr, flush=True)
        raise


def self_test() -> None:
    values = pa.array([None, "", "   ", "\n\t", "actual NL", " \u3000x\u3000 "])
    assert _stripped_nonempty(values).tolist() == [False, False, False, False, True, True]
    p = np.array([1.0, 1.2, 2.0, 3.0])
    a = compute_repeat_counts(p, 4, 0.25, np.random.default_rng(123))
    b = compute_repeat_counts(p, 4, 0.25, np.random.default_rng(123))
    assert np.array_equal(a, b) and int(a.sum()) == 5 and a.min() == 1 and a.max() <= 4
    w = normalized(np.array([1.0, 2.0, 3.0]), "selftest")
    assert np.isclose(w.sum(), 1.0) and np.all(w > 0)
    # Tiny synthetic cell: exercise positive-weight/served filtering, the
    # fixed MONET exclusion, and the band-first row marginal.
    table = pa.table({
        "row_id": pa.array([18907181, 10, 11, 12], type=pa.int64()),
        "bucket_idx": pa.array([33, 33, 33, 33], type=pa.int16()),
        "base_weight": pa.array([9.0, 1.0, 2.0, 1.0], type=pa.float64()),
        "served_mask": pa.array([15, 1, 10, 8], type=pa.uint8()),
    })
    rows, buckets, weights, repeats, metrics = real_domain_arrays(
        table, "monet_real", np.array([.18, .27, .15, .40]), {18907181},
    )
    assert rows.tolist() == [10, 11, 12] and buckets.tolist() == [33, 33, 33]
    raw = np.array([.18, .27 + .40 * 2 / 3, .40 / 3])
    assert np.allclose(weights, raw / raw.sum()) and repeats.tolist() == [1, 1, 1]
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        (tmp / "cells").mkdir()
        cells: dict[str, Any] = {}
        counts: dict[str, int] = {}
        write_cells(
            tmp, "monet_real", rows, buckets, weights, repeats, None,
            cells, counts, metrics,
        )
        assert counts == {"33:monet_real": 3}
        assert file_sha256(tmp / cells["33:monet_real"]["path"]) == cells["33:monet_real"]["sha256"]
        first_sha = cells["33:monet_real"]["sha256"]
        second = tmp / "second.npz"
        deterministic_savez(second, {"row_id": rows, "weight": weights, "repeat": repeats})
        assert file_sha256(second) == first_sha
    print("SELF_TEST_PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--overlay-id")
    parser.add_argument("--catalog-root", type=Path)
    parser.add_argument("--catalog-header", type=Path)
    parser.add_argument("--caption-sidecar-header", type=Path)
    parser.add_argument("--anime-source-manifest", type=Path)
    parser.add_argument("--sampling-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest-index-py", type=Path)
    parser.add_argument("--sampler-py", type=Path)
    parser.add_argument("--exclusions-json", type=Path)
    parser.add_argument("--seed", type=int, default=718256)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.self_test:
        return args
    required = (
        "overlay_id", "catalog_root", "catalog_header", "caption_sidecar_header",
        "anime_source_manifest", "sampling_config", "output_dir",
        "manifest_index_py", "sampler_py",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    if not (1 <= args.workers <= 32):
        parser.error("--workers must be in [1,32]")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL semantic overlay build: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
