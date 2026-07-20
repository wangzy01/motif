#!/usr/bin/env python3
"""OFFLINE_ARB_V1 Phase-1 元数据探针：Task A / D / G。

全部数据源严格只读；程序只写 ``--output-dir``，绝不修改 manifest、图片源或训练 cache。
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import fcntl
import gc
import hashlib
import io
import json
import math
import os
import socket
import struct
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

from training.dataline.image_fetch import (
    fetch_url_bytes,
    monet_tar_path,
    read_gpic_bytes,
    stream_tar_images,
)


SEED = 20260709
MROOT = Path("/s3-mewtant-raw-datasets/jasperai/monet/v1.1.0")
MONET_MANIFEST = Path("/s3-pixai-test-uw2/evan/mot_filtering_v1/t5_manifest/rich_v1/monet")
SYNTH_MANIFEST = Path("/s3-pixai-test-uw2/evan/mot_filtering_v1/t5_manifest/synthetic_v1/monet_synth")
GPIC_MANIFEST = Path("/s3-pixai-test-uw2/evan/mot_filtering_v1/t5_manifest/rich_v1/gpic")
ANIME_RICH = Path("/s3/evan/anime_core_ditv04_ct_soft/data")
ANIME_THIN = Path("/s3-pixai-test-uw2/evan/mot_filtering_v1/t5_manifest/daily_v1/anime")
TEXTATLAS_MANIFEST = Path("/fsx/evan/textatlas_manifest_v1")
TEXTATLAS_SOURCE = Path("/s3/evan/textatlas5m")
OUTPUT_PARENT = Path("/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop")

MONET_SUBSOURCES = (
    "cc12m",
    "commoncatalog-cc-by",
    "coyo",
    "diffusion-aesthetic-4k",
    "laion",
    "megalith10m",
)
SYNTH_SUBSOURCES = ("flux-klein", "flux-schnell", "z-image")
TEXTATLAS_ACTIVE_SUBSETS = (
    "CleanTextSynth",
    "LongWordsSubset-A",
    "LongWordsSubset-M",
    "PPT2Details",
    "PPT2Structured",
    "Paper2Text",
    "StyledTextSynth",
    "TextScenesHQ",
    "TextVisionBlend",
)

EXPECTED_D = {
    "monet_real": {"files": 256, "rows": 40_000_000},
    "synthetic": {"files": 64, "rows": 13_808_747},
    "gpic": {"files": 256, "rows": 20_000_000},
    "anime": {"files": 64, "rows": 16_499_287},
    # Task D uses the stronger all-subset scope; active training is 595 files.
    "textatlas": {"files": 600, "rows": 5_398_826},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)   # dump 中途抛出时清残留 tmp(replace 成功后为 no-op)


def append_summary(output_dir: Path, task: str, verdict: str, numbers: dict[str, Any]) -> None:
    def clean(value: Any) -> str:
        if isinstance(value, float):
            value = f"{value:.6g}"
        return str(value).replace(" ", "_").replace("\n", "_")

    payload = ",".join(f"{k}={clean(v)}" for k, v in numbers.items()) or "none"
    path = output_dir / "pilot_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(f"task={task} verdict={verdict} key_numbers={payload}\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def parquet_files(root: Path) -> list[Path]:
    return sorted(root.glob("*.parquet"))


def nested_parquet_files(root: Path, subsets: Sequence[str] | None = None) -> list[Path]:
    if subsets is None:
        return sorted(root.glob("*/*.parquet"))
    return sorted(p for subset in subsets for p in (root / subset).glob("*.parquet"))


def stable_rank(seed: int, *parts: Any) -> int:
    raw = "\0".join(str(x) for x in (seed, *parts)).encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def spread_files(files: Sequence[Path], max_files: int, seed: int) -> list[Path]:
    """Pick files reproducibly while retaining parent/subset coverage."""
    files = list(files)
    if len(files) <= max_files:
        return files
    by_parent: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        by_parent[p.parent.name].append(p)
    chosen: list[Path] = []
    for parent in sorted(by_parent):
        ranked = sorted(by_parent[parent], key=lambda p: stable_rank(seed, p))
        chosen.append(ranked[0])
    remaining = [p for p in files if p not in set(chosen)]
    remaining.sort(key=lambda p: stable_rank(seed + 1, p))
    return (chosen + remaining)[:max_files]


def row_group_starts(md: pq.FileMetaData) -> tuple[list[int], int]:
    starts: list[int] = []
    total = 0
    for i in range(md.num_row_groups):
        starts.append(total)
        total += md.row_group(i).num_rows
    return starts, total


def sampled_parquet_rows(
    files: Sequence[Path],
    columns: Sequence[str],
    target: int,
    seed: int,
    max_files: int = 32,
) -> list[dict[str, Any]]:
    """Sample exact row positions, reading only the row groups that contain them."""
    if target <= 0:
        return []
    picked_files = spread_files(files, min(max_files, len(files)), seed)
    metadata: list[tuple[Path, pq.ParquetFile, list[int], int]] = []
    for p in picked_files:
        pf = pq.ParquetFile(p)
        starts, rows = row_group_starts(pf.metadata)
        if rows:
            metadata.append((p, pf, starts, rows))
    if not metadata:
        return []

    total_rows = sum(x[3] for x in metadata)
    raw_quotas = [target * x[3] / total_rows for x in metadata]
    quotas = [min(x[3], int(math.floor(q))) for x, q in zip(metadata, raw_quotas)]
    remain = target - sum(quotas)
    order = sorted(
        range(len(metadata)),
        key=lambda i: (raw_quotas[i] - math.floor(raw_quotas[i]), stable_rank(seed, metadata[i][0])),
        reverse=True,
    )
    for i in order:
        if remain <= 0:
            break
        if quotas[i] < metadata[i][3]:
            quotas[i] += 1
            remain -= 1

    out: list[dict[str, Any]] = []
    for file_i, ((path, pf, starts, rows), quota) in enumerate(zip(metadata, quotas)):
        if quota <= 0:
            continue
        rng = np.random.default_rng(seed + file_i * 1_000_003)
        indices = np.sort(rng.choice(rows, size=min(quota, rows), replace=False).astype(np.int64))
        by_rg: dict[int, list[int]] = defaultdict(list)
        for row_idx in indices.tolist():
            rg = bisect.bisect_right(starts, row_idx) - 1
            by_rg[rg].append(row_idx)
        for rg, global_indices in by_rg.items():
            table = pf.read_row_group(rg, columns=list(columns), use_threads=False)
            locals_ = pa.array([x - starts[rg] for x in global_indices], type=pa.int64())
            rows_py = table.take(locals_).to_pylist()
            for global_idx, rec in zip(global_indices, rows_py):
                rec["_manifest_path"] = str(path)
                rec["_row_index"] = int(global_idx)
                out.append(rec)
    out.sort(key=lambda r: stable_rank(seed + 99, r.get("uid", r.get("key", r.get("image_path")))))
    return out[:target]


def stratified_select(rows: Sequence[dict[str, Any]], n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reserve 10% around each 256/512 boundary, then fill ten short-side quantile bins."""
    dedup: dict[str, dict[str, Any]] = {}
    for rec in rows:
        identity = str(rec.get("uid") or rec.get("key") or rec.get("image_path") or (
            rec.get("src_shard"), rec.get("src_key")
        ))
        if rec.get("width") is None or rec.get("height") is None:
            continue
        rec = dict(rec)
        rec["_identity"] = identity
        rec["_short"] = int(min(int(rec["width"]), int(rec["height"])))
        dedup.setdefault(identity, rec)
    pool = list(dedup.values())
    if not pool:
        return [], {"candidate_count": 0, "quantile_edges": [], "boundary_counts": {}}
    n = min(n, len(pool))
    selected: dict[str, dict[str, Any]] = {}
    boundary_counts: dict[str, int] = {}
    boundary_quota = max(1, n // 10)
    for boundary in (256, 512):
        ranked = sorted(
            pool,
            key=lambda r: (abs(r["_short"] - boundary), stable_rank(seed + boundary, r["_identity"])),
        )
        before = len(selected)
        for rec in ranked:
            if len(selected) - before >= boundary_quota or len(selected) >= n:
                break
            if rec["_identity"] not in selected:
                rec = dict(rec)
                rec["_selection_stratum"] = f"boundary_{boundary}"
                selected[rec["_identity"]] = rec
        boundary_counts[str(boundary)] = len(selected) - before

    remaining = [r for r in pool if r["_identity"] not in selected]
    shorts = np.asarray([r["_short"] for r in remaining] or [r["_short"] for r in pool], dtype=np.int64)
    edges = np.quantile(shorts, np.linspace(0.0, 1.0, 11), method="linear").tolist()
    bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in remaining:
        idx = int(np.searchsorted(np.asarray(edges[1:-1]), rec["_short"], side="right"))
        bins[idx].append(rec)
    for idx in bins:
        bins[idx].sort(key=lambda r: stable_rank(seed + idx, r["_identity"]))

    need = n - len(selected)
    active = [idx for idx in range(10) if bins.get(idx)]
    cursor = {idx: 0 for idx in active}
    while need > 0 and active:
        progressed = False
        for idx in list(active):
            pos = cursor[idx]
            if pos >= len(bins[idx]):
                active.remove(idx)
                continue
            rec = dict(bins[idx][pos])
            cursor[idx] += 1
            rec["_selection_stratum"] = f"quantile_{idx}"
            selected[rec["_identity"]] = rec
            need -= 1
            progressed = True
            if need <= 0:
                break
        if not progressed:
            break

    result = list(selected.values())[:n]
    result.sort(key=lambda r: stable_rank(seed + 777, r["_identity"]))
    strata = Counter(r["_selection_stratum"] for r in result)
    return result, {
        "candidate_count": len(pool),
        "quantile_edges": [round(float(x), 3) for x in edges],
        "boundary_counts": boundary_counts,
        "selected_strata": dict(sorted(strata.items())),
    }


def sampled_n_tars(base: Path, n: int, seed: int) -> list[Path]:
    """从物理 shard 全集做稳定 hash 抽样，避免总取 ingest 顺序最前的文件。"""
    found: list[Path] = []
    if not base.is_dir():
        return found
    for group in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name):
        found.extend(group.glob("*.tar"))
    found.sort(key=lambda p: stable_rank(seed, p))
    return found[:n]


def selected_tar_locators(domain: str, seed: int) -> list[str]:
    """Select physical tar shards first; the returned paths are manifest src_shard values."""
    tars: list[Path] = []
    if domain == "monet_real":
        for i, sub in enumerate(MONET_SUBSOURCES):
            count = 2 if sub in ("coyo", "laion") else 1
            tars.extend(sampled_n_tars(MROOT / sub, count, seed + i))
    elif domain == "synthetic":
        for i, sub in enumerate(SYNTH_SUBSOURCES):
            tars.extend(sampled_n_tars(MROOT / "synthetic" / sub, 2, seed + i))
    else:
        raise ValueError(domain)
    rels = [str(p.relative_to(MROOT).with_suffix(".parquet")) for p in tars]
    return rels[:8]


def tar_manifest_candidates(domain: str, target: int, seed: int) -> tuple[list[dict[str, Any]], list[str]]:
    root = MONET_MANIFEST if domain == "monet_real" else SYNTH_MANIFEST
    locators = selected_tar_locators(domain, seed)
    if not locators:
        return [], []
    wanted = pa.array(locators, type=pa.large_string())
    columns = ["uid", "sub_source", "src_root", "src_shard", "src_key", "width", "height"]
    candidates: list[dict[str, Any]] = []
    cap = min(120_000, max(target * 20, target + 100))
    for path in parquet_files(root):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=131_072, columns=columns, use_threads=False):
            mask = pc.is_in(batch.column(batch.schema.get_field_index("src_shard")), value_set=wanted)
            selected = batch.filter(mask)
            if selected.num_rows:
                candidates.extend(selected.to_pylist())
    # 扫完全部 manifest 后再做稳定有界抽样，避免 manifest 前缀偏差。
    if len(candidates) > cap:
        candidates.sort(key=lambda r: stable_rank(seed + 500, r.get("uid"), r.get("src_shard")))
        candidates = candidates[:cap]
    return candidates, locators


def inspect_image(raw: bytes, rec: dict[str, Any]) -> dict[str, Any]:
    out = {
        "uid": rec.get("uid") or (f"anime:{rec['key']}" if rec.get("key") else rec.get("image_path")),
        "locator": {
            k: rec[k]
            for k in (
                "src_root", "src_shard", "src_key", "s3key", "_manifest_path", "_row_index",
                "source_path", "source_row_group", "source_row_group_rows",
            )
            if k in rec
        },
        "sub_source": rec.get("sub_source") or rec.get("dataset") or Path(str(rec.get("_manifest_path", ""))).parent.name,
        "manifest_width": int(rec["width"]),
        "manifest_height": int(rec["height"]),
        "selection_stratum": rec.get("_selection_stratum"),
    }
    try:
        with Image.open(io.BytesIO(raw)) as img:
            decoded = tuple(int(x) for x in img.size)
            try:
                orientation = int(img.getexif().get(274, 1))
            except Exception:  # noqa: BLE001 - malformed EXIF is itself observable
                orientation = -1
        manifest = (out["manifest_width"], out["manifest_height"])
        if decoded == manifest:
            classification = "exact"
        elif decoded == (manifest[1], manifest[0]):
            classification = "transposed"
        else:
            classification = "other_mismatch"
        out.update({
            "decoded_width": decoded[0],
            "decoded_height": decoded[1],
            "exif_orientation": orientation,
            "classification": classification,
        })
    except Exception as exc:  # noqa: BLE001 - corrupt data belongs in the probe result
        out.update({"classification": "decode_error", "error": f"{type(exc).__name__}: {exc}"[:500]})
    return out


def fetch_tar_samples(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_tar: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in samples:
        by_tar[monet_tar_path(str(rec["src_root"]), str(rec["src_shard"]))].append(rec)
    results: list[dict[str, Any]] = []
    for tar_path, rows in sorted(by_tar.items()):
        by_key = {str(r["src_key"]): r for r in rows}
        seen: set[str] = set()
        for key, raw in stream_tar_images(tar_path, set(by_key)):
            seen.add(key)
            results.append(inspect_image(raw, by_key[key]))
        for key in sorted(set(by_key) - seen):
            rec = by_key[key]
            results.append({
                "uid": rec.get("uid"),
                "locator": {"tar_path": tar_path, "src_key": key},
                "manifest_width": int(rec["width"]),
                "manifest_height": int(rec["height"]),
                "selection_stratum": rec.get("_selection_stratum"),
                "classification": "fetch_error",
                "error": "stream_tar_images did not yield the requested key",
            })
    return results


def fetch_anime_samples(samples: Sequence[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    workers = max(1, min(16, workers))

    def fetch(rec: dict[str, Any]) -> bytes | None:
        return fetch_url_bytes(str(rec["s3key"]), retries=3, timeout=20, cache_dir=None)

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for rec, raw in zip(samples, pool.map(fetch, samples)):
            if raw is None:
                results.append({
                    "uid": f"anime:{rec['key']}",
                    "locator": {"s3key": rec["s3key"]},
                    "manifest_width": int(rec["width"]),
                    "manifest_height": int(rec["height"]),
                    "selection_stratum": rec.get("_selection_stratum"),
                    "classification": "fetch_error",
                    "error": "fetch_url_bytes returned None",
                })
            else:
                results.append(inspect_image(raw, rec))
    return results


def fetch_gpic_samples(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for rec in samples:
        raw = read_gpic_bytes(rec["src_root"], rec["src_shard"], rec["src_key"], ext=None)
        if raw is None:
            results.append({
                "uid": rec.get("uid"),
                "locator": {k: rec[k] for k in ("src_root", "src_shard", "src_key")},
                "manifest_width": int(rec["width"]),
                "manifest_height": int(rec["height"]),
                "selection_stratum": rec.get("_selection_stratum"),
                "classification": "fetch_error",
                "error": "read_gpic_bytes returned None",
            })
        else:
            results.append(inspect_image(raw, rec))
    return results


def fetch_textatlas_samples(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sidecar: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in samples:
        by_sidecar[str(rec["_manifest_path"])].append(rec)
    results: list[dict[str, Any]] = []

    def failed(rec: dict[str, Any], source: Path, exc: Exception) -> dict[str, Any]:
        return {
            "uid": rec.get("uid"),
            "locator": {"source": str(source), "row_index": rec.get("_row_index")},
            "manifest_width": int(rec["width"]),
            "manifest_height": int(rec["height"]),
            "selection_stratum": rec.get("_selection_stratum"),
            "classification": "fetch_error",
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }

    for sidecar_s, rows in sorted(by_sidecar.items()):
        sidecar = Path(sidecar_s)
        source = TEXTATLAS_SOURCE / sidecar.parent.name / sidecar.name
        try:
            pf = pq.ParquetFile(source)
            starts, total = row_group_starts(pf.metadata)
        except Exception as exc:  # noqa: BLE001 - file-level open/footer error
            results.extend(failed(rec, source, exc) for rec in rows)
            continue
        by_rg: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for rec in rows:
            row_idx = int(rec["_row_index"])
            if row_idx < 0 or row_idx >= total:
                results.append(failed(rec, source, IndexError(f"sidecar row {row_idx} outside source rows {total}")))
                continue
            by_rg[bisect.bisect_right(starts, row_idx) - 1].append(rec)
        for rg, rg_rows in by_rg.items():
            try:
                table = pf.read_row_group(rg, columns=["image", "image_path"], use_threads=False)
            except Exception as exc:  # noqa: BLE001 - isolate a damaged row group
                results.extend(failed(rec, source, exc) for rec in rg_rows)
                continue
            for rec in rg_rows:
                try:
                    local = int(rec["_row_index"]) - starts[rg]
                    image = table.column("image")[local].as_py()
                    source_image_path = table.column("image_path")[local].as_py()
                    if source_image_path != rec["image_path"]:
                        raise ValueError(
                            f"row identity mismatch sidecar={rec['image_path']!r} source={source_image_path!r}"
                        )
                    raw = image.get("bytes") if isinstance(image, dict) else None
                    if not raw:
                        raise ValueError("image.bytes is empty")
                    rec_with_rg = dict(rec)
                    rec_with_rg.update({
                        "source_path": str(source),
                        "source_row_group": rg,
                        "source_row_group_rows": table.num_rows,
                    })
                    results.append(inspect_image(raw, rec_with_rg))
                except Exception as exc:  # noqa: BLE001 - isolate one corrupt/misaligned item
                    results.append(failed(rec, source, exc))
    return results


def summarize_domain_a(domain: str, requested: int, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(r.get("classification")) for r in records)
    decoded = counts["exact"] + counts["transposed"] + counts["other_mismatch"]
    orientations = Counter(str(r.get("exif_orientation")) for r in records if "exif_orientation" in r)
    orientation_non1 = sum(v for k, v in orientations.items() if k != "1")
    exact_rate = counts["exact"] / decoded if decoded else 0.0
    orientation_rate = orientation_non1 / decoded if decoded else 1.0
    mismatches = counts["transposed"] + counts["other_mismatch"]
    transpose_share = counts["transposed"] / mismatches if mismatches else 0.0
    coverage = decoded / requested if requested else 0.0
    transpose_concentrated = (
        counts["transposed"] >= max(3, math.ceil(decoded * 0.001))
        and transpose_share >= 0.5
    )
    threshold_fail = exact_rate < 0.995 or orientation_rate >= 0.005 or transpose_concentrated
    if threshold_fail:
        verdict = "FAIL"
    elif coverage < 1.0:
        verdict = "WARN"
    else:
        verdict = "PASS"
    summary = {
        "domain": domain,
        "verdict": verdict,
        "requested": requested,
        "records": len(records),
        "decoded": decoded,
        "coverage_rate": coverage,
        "classification_counts": dict(sorted(counts.items())),
        "exact_rate": exact_rate,
        "orientation_non1_rate": orientation_rate,
        "orientation_histogram": dict(sorted(orientations.items())),
        "transpose_share_of_mismatches": transpose_share,
        "transpose_concentrated": transpose_concentrated,
        "sub_source_counts": dict(sorted(Counter(str(r.get("sub_source", "unknown")) for r in records).items())),
    }
    if domain == "textatlas":
        groups: dict[tuple[str, int], int] = {}
        for rec in records:
            locator = rec.get("locator") or {}
            if "source_path" in locator and "source_row_group" in locator:
                groups[(str(locator["source_path"]), int(locator["source_row_group"]))] = int(
                    locator.get("source_row_group_rows", 0)
                )
        loaded_rows = sum(groups.values())
        summary["row_group_io"] = {
            "unique_row_groups_read": len(groups),
            "rows_materialized_by_row_group_reads": loaded_rows,
            "row_materialization_amplification_over_decoded": loaded_rows / decoded if decoded else None,
        }
    return summary


def task_a(args: argparse.Namespace) -> dict[str, Any]:
    requested = args.sample_n if args.sample_n is not None else 2_000
    if requested <= 0:
        raise ValueError("--sample-n must be positive")
    candidate_n = min(100_000, max(requested * 20, requested + 100))
    all_samples: dict[str, list[dict[str, Any]]] = {}
    sampling: dict[str, Any] = {}
    tar_shards: dict[str, list[str]] = {}

    for offset, domain in enumerate(("monet_real", "synthetic")):
        candidates, locators = tar_manifest_candidates(domain, requested, args.seed + offset)
        all_samples[domain], sampling[domain] = stratified_select(candidates, requested, args.seed + offset)
        tar_shards[domain] = locators

    generic_specs = {
        "anime": (parquet_files(ANIME_RICH), ["key", "s3key", "dataset", "width", "height"]),
        "gpic": (parquet_files(GPIC_MANIFEST), ["uid", "sub_source", "src_root", "src_shard", "src_key", "width", "height"]),
        "textatlas": (
            nested_parquet_files(TEXTATLAS_MANIFEST, TEXTATLAS_ACTIVE_SUBSETS),
            ["uid", "image_path", "width", "height"],
        ),
    }
    for offset, (domain, (files, columns)) in enumerate(generic_specs.items(), start=10):
        # 全域按 footer 行数分配随机行；manifest 单 RG 时会窄列扫完整文件，但不会读无关列。
        candidates = sampled_parquet_rows(
            files, columns, candidate_n, args.seed + offset, max_files=len(files)
        )
        all_samples[domain], sampling[domain] = stratified_select(candidates, requested, args.seed + offset)

    per_sample: dict[str, list[dict[str, Any]]] = {
        "monet_real": fetch_tar_samples(all_samples["monet_real"]),
        "synthetic": fetch_tar_samples(all_samples["synthetic"]),
        "anime": fetch_anime_samples(all_samples["anime"], args.fetch_workers),
        "gpic": fetch_gpic_samples(all_samples["gpic"]),
        "textatlas": fetch_textatlas_samples(all_samples["textatlas"]),
    }
    domains = {
        domain: summarize_domain_a(domain, requested, records)
        for domain, records in per_sample.items()
    }
    domain_verdicts = [x["verdict"] for x in domains.values()]
    verdict = "FAIL" if "FAIL" in domain_verdicts else ("WARN" if "WARN" in domain_verdicts else "PASS")
    if verdict == "PASS" and requested < 2_000:
        verdict = "WARN"
    return {
        "task": "A",
        "verdict": verdict,
        "scope": "five_domains",
        "sample_n_per_domain": requested,
        "thresholds": {
            "exact_rate_min": 0.995,
            "orientation_non1_rate_strict_max": 0.005,
            "decoded_coverage_required_for_pass": 1.0,
            "transpose_concentrated": "count>=max(3,ceil(decoded*0.001)) and share_of_mismatches>=0.5",
            "boundary_quota_each": "10% of selected rows nearest min(width,height)=256/512",
            "quantile_bins": 10,
        },
        "tar_shards": tar_shards,
        "sampling": sampling,
        "domains": domains,
        "samples": per_sample,
        "key_numbers": {
            "domains_pass": sum(v == "PASS" for v in domain_verdicts),
            "domains_warn": sum(v == "WARN" for v in domain_verdicts),
            "domains_fail": sum(v == "FAIL" for v in domain_verdicts),
            "decoded": sum(x["decoded"] for x in domains.values()),
        },
    }


def md5_digest(data: bytes) -> bytes:
    try:
        return hashlib.md5(data, usedforsecurity=False).digest()
    except TypeError:  # pragma: no cover - compatibility with non-OpenSSL Python builds
        return hashlib.md5(data).digest()


def hash_uid_file(job: tuple[str, str, int | None, int]) -> tuple[str, np.ndarray, int, int, int]:
    path_s, column, limit, batch_size = job
    pf = pq.ParquetFile(path_s)
    target = pf.metadata.num_rows if limit is None else min(limit, pf.metadata.num_rows)
    hashes = np.empty((target, 2), dtype=np.int64)
    pos = 0
    nulls = 0
    empty = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=[column], use_threads=False):
        values = batch.column(0).to_pylist()
        for value in values:
            if pos >= target:
                break
            if value is None:
                nulls += 1
                raw = b"\x00<NULL_UID>"
            else:
                raw = str(value).encode("utf-8", "surrogatepass")
                if not raw:
                    empty += 1
            hashes[pos] = struct.unpack(">qq", md5_digest(raw))
            pos += 1
        if pos >= target:
            break
    return path_s, hashes[:pos], nulls, empty, pos


def hash_domain(
    domain: str,
    files: Sequence[Path],
    column: str,
    sample_n: int | None,
    workers: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    footer_rows = []
    for p in files:
        footer_rows.append(pq.ParquetFile(p).metadata.num_rows)
    full_rows = int(sum(footer_rows))
    arrays: list[np.ndarray] = []
    nulls = 0
    empty = 0
    rows_hashed = 0
    started = time.perf_counter()

    if sample_n is not None:
        remaining = sample_n
        jobs: list[tuple[str, str, int | None, int]] = []
        for path, rows in zip(files, footer_rows):
            if remaining <= 0:
                break
            take = min(remaining, rows)
            jobs.append((str(path), column, take, batch_size))
            remaining -= take
        results = map(hash_uid_file, jobs)
    else:
        jobs = [(str(path), column, None, batch_size) for path in files]
        max_workers = max(1, min(workers, len(jobs)))
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        results = executor.map(hash_uid_file, jobs, chunksize=1)

    try:
        for _path, arr, nnull, nempty, nrows in results:
            arrays.append(arr)
            nulls += nnull
            empty += nempty
            rows_hashed += nrows
    finally:
        if sample_n is None:
            executor.shutdown(wait=True, cancel_futures=True)

    hashes = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 2), dtype=np.int64)
    elapsed = time.perf_counter() - started
    expected_hashed = full_rows if sample_n is None else min(sample_n, full_rows)
    return hashes, {
        "domain": domain,
        "manifest_files": len(files),
        "footer_rows": full_rows,
        "rows_hashed": rows_hashed,
        "expected_rows_hashed": expected_hashed,
        "null_uid_count": nulls,
        "empty_uid_count": empty,
        "hash_seconds": elapsed,
        "hash_rows_per_second": rows_hashed / elapsed if elapsed else None,
    }


def unique_digest_stats(hashes: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    digest_view = np.ascontiguousarray(hashes).view(np.dtype("V16")).reshape(-1)
    unique, counts = np.unique(digest_view, return_counts=True)
    duplicate_groups = int(np.count_nonzero(counts > 1))
    stats = {
        "unique_uid_digests": int(unique.size),
        "duplicate_extra_rows": int(digest_view.size - unique.size),
        "duplicate_groups": duplicate_groups,
        "max_multiplicity": int(counts.max()) if counts.size else 0,
    }
    return unique, stats


def task_d(args: argparse.Namespace) -> dict[str, Any]:
    if args.sample_n is not None and args.sample_n <= 0:
        raise ValueError("--sample-n must be positive")
    specs = {
        "monet_real": (parquet_files(MONET_MANIFEST), "uid"),
        "synthetic": (parquet_files(SYNTH_MANIFEST), "uid"),
        "gpic": (parquet_files(GPIC_MANIFEST), "uid"),
        # Thin anime manifest has the literal uid column required by Task D.
        "anime": (parquet_files(ANIME_THIN), "uid"),
        # Stronger all-10-subset scope (600), including CoverBook.
        "textatlas": (nested_parquet_files(TEXTATLAS_MANIFEST), "uid"),
    }
    domains: dict[str, dict[str, Any]] = {}
    retained: dict[str, np.ndarray] = {}
    for domain, (files, column) in specs.items():
        hashes, info = hash_domain(
            domain,
            files,
            column,
            args.sample_n,
            args.hash_workers,
            args.batch_size,
        )
        unique, unique_stats = unique_digest_stats(hashes)
        info.update(unique_stats)
        expected = EXPECTED_D[domain]
        info["expected_manifest_files"] = expected["files"]
        info["expected_footer_rows"] = expected["rows"]
        info["complete_manifest"] = len(files) == expected["files"] and info["footer_rows"] == expected["rows"]
        domains[domain] = info
        if domain in ("monet_real", "synthetic"):
            retained[domain] = unique
        del hashes
        if domain not in retained:
            del unique
        gc.collect()

    intersection = np.intersect1d(
        retained["monet_real"], retained["synthetic"], assume_unique=True
    )
    cross_count = int(intersection.size)
    del retained, intersection
    gc.collect()

    defects = []
    for domain, info in domains.items():
        if info["duplicate_extra_rows"]:
            defects.append(f"{domain}:duplicate")
        if info["null_uid_count"] or info["empty_uid_count"]:
            defects.append(f"{domain}:null_or_empty")
        if info["rows_hashed"] != info["expected_rows_hashed"]:
            defects.append(f"{domain}:short_read")
        if not info["complete_manifest"]:
            defects.append(f"{domain}:manifest_count")
    if cross_count:
        defects.append("monet_real_intersect_synthetic")
    if defects:
        verdict = "FAIL"
    elif args.sample_n is not None:
        verdict = "WARN"
    else:
        verdict = "PASS"
    return {
        "task": "D",
        "verdict": verdict,
        "scope": "sampled_per_domain" if args.sample_n is not None else "full",
        "sample_n_per_domain": args.sample_n,
        "hash": "MD5 digest split into two int64 values; np.unique over a contiguous V16 row view",
        "domains": domains,
        "monet_real_synthetic_intersection_count": cross_count,
        "defects": defects,
        "thresholds": {
            "domain_duplicate_extra_rows": 0,
            "monet_real_synthetic_intersection_count": 0,
            "null_or_empty_uid": 0,
            "complete_manifest_required": True,
        },
        "key_numbers": {
            "domains": len(domains),
            "duplicate_extra_rows": sum(x["duplicate_extra_rows"] for x in domains.values()),
            "cross_intersection": cross_count,
            "rows_hashed": sum(x["rows_hashed"] for x in domains.values()),
        },
    }


def image_bytes_column(rg: pq.RowGroupMetaData) -> int | None:
    for i in range(rg.num_columns):
        if rg.column(i).path_in_schema == "image.bytes":
            return i
    return None


def task_g(args: argparse.Namespace) -> dict[str, Any]:
    files_all = nested_parquet_files(TEXTATLAS_SOURCE)
    if args.sample_n is not None:
        if args.sample_n <= 0:
            raise ValueError("--sample-n must be positive")
        files = spread_files(files_all, min(args.sample_n, len(files_all)), args.seed)
    else:
        files = files_all

    row_group_rows: list[int] = []
    row_groups: list[dict[str, Any]] = []
    per_shard: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    offset_ok = 0
    offset_total = 0
    non_tail_anomalies = 0
    total_rows = 0
    total_footer_bytes = 0
    scan_started = time.perf_counter()
    for path in files:
        shard_started = time.perf_counter()
        try:
            pf = pq.ParquetFile(path)
            md = pf.metadata
            footer_loaded = time.perf_counter()
            row_start = 0
            shard_offset_ok = 0
            previous_end: int | None = None
            overlaps = 0
            for rg_i in range(md.num_row_groups):
                rg = md.row_group(rg_i)
                rows = int(rg.num_rows)
                row_group_rows.append(rows)
                if rg_i < md.num_row_groups - 1 and rows != 100:
                    non_tail_anomalies += 1
                col_i = image_bytes_column(rg)
                entry: dict[str, Any] = {
                    "path": str(path),
                    "subset": path.parent.name,
                    "row_group": rg_i,
                    "row_start": row_start,
                    "num_rows": rows,
                }
                offset_total += 1
                if col_i is None:
                    entry["offset_error"] = "image.bytes column missing"
                else:
                    col = rg.column(col_i)
                    raw_file_offset = int(col.file_offset) if col.file_offset is not None else -1
                    compressed_size = int(col.total_compressed_size) if col.total_compressed_size is not None else -1
                    dictionary_offset = int(col.dictionary_page_offset) if col.dictionary_page_offset is not None else -1
                    data_offset = int(col.data_page_offset) if col.data_page_offset is not None else -1
                    starts = [x for x in (dictionary_offset, data_offset) if x >= 0]
                    range_start = min(starts) if starts else -1
                    range_end = range_start + compressed_size if range_start >= 0 else -1
                    valid = range_start >= 0 and compressed_size > 0
                    entry.update({
                        "file_offset_raw": raw_file_offset,
                        "total_compressed_size": compressed_size,
                        "dictionary_page_offset": dictionary_offset,
                        "data_page_offset": data_offset,
                        "range_start": range_start,
                        "range_end": range_end,
                        "file_offset_minus_range_end": raw_file_offset - range_end if range_end >= 0 else None,
                        "file_offset_consistent_with_chunk_end": raw_file_offset < 0 or raw_file_offset >= range_end,
                        "offset_valid": valid,
                    })
                    if previous_end is not None and range_start >= 0 and range_start < previous_end:
                        overlaps += 1
                    if range_end >= 0:
                        previous_end = range_end
                    if valid:
                        offset_ok += 1
                        shard_offset_ok += 1
                row_groups.append(entry)
                row_start += rows
            total_rows += int(md.num_rows)
            total_footer_bytes += int(md.serialized_size)
            per_shard.append({
                "path": str(path),
                "subset": path.parent.name,
                "num_rows": int(md.num_rows),
                "num_row_groups": int(md.num_row_groups),
                "footer_serialized_bytes": int(md.serialized_size),
                "footer_open_seconds": footer_loaded - shard_started,
                "metadata_extract_seconds": time.perf_counter() - footer_loaded,
                "offsets_extracted": shard_offset_ok,
                "overlapping_image_chunks": overlaps,
            })
        except Exception as exc:  # noqa: BLE001 - continue to produce a complete failure inventory
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"[:500]})
    elapsed = time.perf_counter() - scan_started

    hist = Counter(row_group_rows)
    exact100_rate = hist[100] / len(row_group_rows) if row_group_rows else 0.0
    around100_rate = sum(v for k, v in hist.items() if 90 <= k <= 110) / len(row_group_rows) if row_group_rows else 0.0
    offset_rate = offset_ok / offset_total if offset_total else 0.0
    full_scope = args.sample_n is None
    defects = []
    if full_scope and len(files_all) != 600:
        defects.append(f"source_file_count={len(files_all)}")
    if errors:
        defects.append(f"footer_errors={len(errors)}")
    if offset_rate < 1.0:
        defects.append(f"offset_extract_rate={offset_rate:.6f}")
    if exact100_rate < 0.98:
        defects.append(f"exact100_rate={exact100_rate:.6f}")
    if non_tail_anomalies:
        defects.append(f"non_tail_anomalies={non_tail_anomalies}")
    if defects:
        verdict = "FAIL"
    elif not full_scope:
        verdict = "WARN"
    else:
        verdict = "PASS"

    # Two explicit layouts: an incremental 4-byte pair, and a standalone 16-byte mapping.
    cost = {
        "rows": total_rows,
        "row_groups": len(row_group_rows),
        "incremental_rg_id_uint16_plus_row_in_rg_uint16_bytes": total_rows * 4,
        "standalone_global_row_shard_rg_local_16bytes": total_rows * 16,
        "row_group_locator_32bytes": len(row_group_rows) * 32,
        "uint16_capacity": {
            "global_row_group_count_fits": len(row_group_rows) <= np.iinfo(np.uint16).max,
            "max_rows_in_group_fits": (max(row_group_rows) if row_group_rows else 0) <= np.iinfo(np.uint16).max,
        },
        "observed_footer_and_extract_seconds": elapsed,
        "observed_row_groups_per_second": len(row_group_rows) / elapsed if elapsed else None,
        "note": "Raw, uncompressed estimates; excludes Parquet writer time and catalog columns already stored elsewhere.",
    }
    return {
        "task": "G",
        "verdict": verdict,
        "scope": "all_600_source_parquets" if full_scope else "sampled_source_parquets",
        "source_files_discovered": len(files_all),
        "source_files_scanned": len(files),
        "footer_errors": errors,
        "total_rows": total_rows,
        "total_row_groups": len(row_group_rows),
        "footer_serialized_bytes": total_footer_bytes,
        "row_group_rows_histogram": {str(k): v for k, v in sorted(hist.items())},
        "row_group_rows_percentiles": {
            "min": min(row_group_rows) if row_group_rows else None,
            "p50": percentile(row_group_rows, 50),
            "p95": percentile(row_group_rows, 95),
            "p99": percentile(row_group_rows, 99),
            "max": max(row_group_rows) if row_group_rows else None,
        },
        "exact_100_rows_rate": exact100_rate,
        "rows_90_to_110_rate": around100_rate,
        "non_tail_not_100_count": non_tail_anomalies,
        "offset_extract_rate": offset_rate,
        "thresholds": {
            "source_files_full_scope": 600,
            "footer_and_offset_extract_rate": 1.0,
            "exact_100_rows_rate_min": 0.98,
            "non_tail_not_100_count": 0,
        },
        "index_cost_projection": cost,
        "per_shard": per_shard,
        # Literal per-row-group locator evidence; file_offset is retained as raw metadata,
        # while range_start is derived from dictionary/data page offsets.
        "row_groups": row_groups,
        "defects": defects,
        "key_numbers": {
            "files": len(files),
            "row_groups": len(row_group_rows),
            "exact100_rate": exact100_rate,
            "offset_rate": offset_rate,
        },
    }


TASKS = {"A": task_a, "D": task_d, "G": task_g}


def run_one(task: str, args: argparse.Namespace) -> str:
    started_wall = utc_now()
    started = time.perf_counter()
    try:
        result = TASKS[task](args)
    except Exception as exc:  # noqa: BLE001 - a machine-readable failure is better than a lost traceback
        result = {
            "task": task,
            "verdict": "FAIL",
            "scope": "exception",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=30),
            "key_numbers": {"exception": type(exc).__name__},
        }
    result.update({
        "schema_version": "offline_arb_phase1_v1",
        "host": socket.gethostname(),
        "seed": args.seed,
        "started_at": started_wall,
        "completed_at": utc_now(),
        "duration_seconds": time.perf_counter() - started,
    })
    output_dir = Path(args.output_dir)
    atomic_json(output_dir / f"task{task}_result.json", result)
    append_summary(output_dir, task, result["verdict"], result.get("key_numbers", {}))
    print(f"task={task} verdict={result['verdict']} result={output_dir / f'task{task}_result.json'}", flush=True)
    return str(result["verdict"])


def parse_tasks(raw: str) -> list[str]:
    tasks = [x.strip().upper() for x in raw.split(",") if x.strip()]
    invalid = [x for x in tasks if x not in TASKS]
    if not tasks or invalid:
        raise argparse.ArgumentTypeError(f"Task must be a comma-separated subset of A,D,G; invalid={invalid}")
    # D 会起多进程；组合运行时先做 D，避免在 A 的 HTTP/PIL 线程之后 fork。
    requested = set(tasks)
    return [task for task in ("D", "G", "A") if task in requested]


def validate_output_dir(raw: str) -> Path:
    path = Path(raw).expanduser().resolve(strict=False)
    parent = OUTPUT_PARENT.resolve(strict=False)
    if path.parent != parent or not path.name.startswith("offline_arb_phase1_") or len(path.name) <= len("offline_arb_phase1_"):
        raise ValueError(
            f"--output-dir must be {OUTPUT_PARENT}/offline_arb_phase1_${{RUN_TAG}}, got {path}"
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=parse_tasks, help="A,D,G or a comma-separated subset")
    parser.add_argument("--output-dir", required=True, help="Run output directory; the only write destination")
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Scale A rows/domain; cap D rows/domain; cap G source shards. D/G defaults remain full.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fetch-workers", type=int, default=16, help="Task A anime fetch workers (hard-capped at 16)")
    parser.add_argument("--hash-workers", type=int, default=16, help="Task D manifest hashing processes (max 64)")
    parser.add_argument("--batch-size", type=int, default=262_144, help="Task D Arrow batch rows")
    args = parser.parse_args()
    args.hash_workers = max(1, min(64, args.hash_workers))
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        args.output_dir = validate_output_dir(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    verdicts = [run_one(task, args) for task in args.task]
    return 2 if "FAIL" in verdicts else 0


if __name__ == "__main__":
    sys.exit(main())
