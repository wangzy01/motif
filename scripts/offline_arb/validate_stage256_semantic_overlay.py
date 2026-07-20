#!/usr/bin/env python3
"""Independent full validator for the 7.18 Stage256 semantic overlay.

Production mode validates every NPZ byte/hash and every selected row.  In
particular it scans all anime caption-sidecar parts and evaluates the three NL
columns itself with a stripped-non-empty predicate.  It also recomputes all
non-anime weights from the immutable catalog and, when ``--repo-root`` is
provided (required unless explicitly disabled), recomputes anime priority and
repeat assignment through the SHA-bound authoritative implementation.

This is a large-I/O CPU/JCC program, not a login-host command.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


FORMAT = "offline_arb_semantic_overlay_v1"
DOMAINS = ("anime", "monet_real", "gpic", "synthetic", "textatlas")
BANDS = ("T1", "T2", "T3", "T4")
NL_COLUMNS = ("refined_description", "medium_description", "short_description")
EXPECTED_EXCLUDED = {
    "monet_real": {
        18907181, 23881582, 23881911, 31217524,
        31221132, 32217542, 35492824, 40078044,
    },
    "textatlas": {94682834, 93569671, 93606908},
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
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
        raise ValueError(f"expected JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"create-once receipt exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    try:
        with tmp.open("wb") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def stripped_nonempty(array: pa.Array | pa.ChunkedArray) -> np.ndarray:
    trimmed = pc.utf8_trim_whitespace(pc.fill_null(array, ""))
    return np.asarray(
        pc.greater(pc.utf8_length(trimmed), 0).to_numpy(zero_copy_only=False), dtype=bool,
    )


def arrow_numpy(column: pa.ChunkedArray, dtype: np.dtype) -> np.ndarray:
    return np.asarray(column.combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype)


def normalized(value: np.ndarray) -> np.ndarray:
    total = float(value.sum(dtype=np.float64))
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"invalid weight denominator: {total}")
    return value.astype(np.float64, copy=False) / total


def cell_seed(overlay_id: str, root_seed: int, bucket_idx: int) -> int:
    payload = f"{overlay_id}\x00{root_seed}\x00{bucket_idx}\x00anime".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def load_and_validate_cells(root: Path, header: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    cells: dict[str, dict[str, np.ndarray]] = {}
    declared = header.get("cells")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("header.cells is empty")
    exclusions = set(int(x) for x in header["excluded_row_ids"])
    for key, entry in sorted(declared.items()):
        bucket_s, domain = key.split(":", 1)
        bucket = int(bucket_s)
        if domain not in DOMAINS or not (0 <= bucket < 95):
            raise ValueError(f"invalid cell key: {key}")
        path = root / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"cell SHA mismatch: {key}")
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            required = {"row_id", "weight", "repeat"}
            if not required.issubset(names):
                raise ValueError(f"cell arrays missing: {key}: {names}")
            arrays = {name: archive[name].copy() for name in archive.files}
        rows, weight, repeat = arrays["row_id"], arrays["weight"], arrays["repeat"]
        if rows.dtype != np.int64 or weight.dtype != np.float64 or repeat.dtype != np.uint8:
            raise TypeError(f"bad NPZ dtypes: {key}: {rows.dtype}/{weight.dtype}/{repeat.dtype}")
        if not (len(rows) == len(weight) == len(repeat) == int(entry["rows"])):
            raise ValueError(f"cell row count mismatch: {key}")
        if len(rows) == 0 or np.any(np.diff(rows) <= 0):
            raise ValueError(f"cell row ids not strictly increasing/nonempty: {key}")
        if np.any(~np.isfinite(weight)) or np.any(weight <= 0) or abs(float(weight.sum()) - 1.0) > 2e-12:
            raise ValueError(f"invalid cell weights: {key}")
        if np.any(repeat < 1) or (domain == "anime" and np.any(repeat > 4)):
            raise ValueError(f"invalid repeat count: {key}")
        if domain != "anime" and np.any(repeat != 1):
            raise ValueError(f"non-anime repeat is not one: {key}")
        if np.isin(rows, np.fromiter(exclusions, dtype=np.int64)).any():
            raise ValueError(f"excluded row present in cell: {key}")
        if int(rows[0]) != int(entry["row_id_min"]) or int(rows[-1]) != int(entry["row_id_max"]):
            raise ValueError(f"cell range metadata mismatch: {key}")
        if int(header["N_select"][bucket][DOMAINS.index(domain)]) != len(rows):
            raise ValueError(f"N_select matrix mismatch: {key}")
        if int(header.get("N_select_by_cell", {}).get(key, -1)) != len(rows):
            raise ValueError(f"N_select_by_cell mismatch: {key}")
        cells[key] = arrays
    return cells


def expected_non_anime(
    catalog_root: Path, sampling: dict[str, Any], domain: str,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    columns = ["row_id", "bucket_idx"]
    if domain in ("monet_real", "gpic", "synthetic"):
        columns.append("base_weight")
    if domain in ("monet_real", "gpic"):
        columns.append("served_mask")
    table = pq.read_table(catalog_root / f"{domain}.parquet", columns=columns, use_threads=True)
    rows = arrow_numpy(table["row_id"], np.int64)
    buckets = arrow_numpy(table["bucket_idx"], np.int16)
    exclusions = EXPECTED_EXCLUDED.get(domain, set())
    if domain == "textatlas":
        keep = ~np.isin(rows, np.fromiter(exclusions, dtype=np.int64))
        base = None
        served = None
    else:
        base = arrow_numpy(table["base_weight"], np.float64)
        keep = np.isfinite(base) & (base > 0)
        if domain in ("monet_real", "gpic"):
            served = arrow_numpy(table["served_mask"], np.uint8)
            keep &= served > 0
        else:
            served = None
        if exclusions:
            keep &= ~np.isin(rows, np.fromiter(exclusions, dtype=np.int64))
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    band_mix = np.array([float(sampling["band_mix_real"][x]) for x in BANDS], dtype=np.float64)
    for bucket in np.unique(buckets):
        cell = keep & (buckets == bucket)
        selected_rows = rows[cell]
        if len(selected_rows) == 0:
            continue
        if domain == "textatlas":
            weight = np.full(len(selected_rows), 1.0 / len(selected_rows), dtype=np.float64)
        elif domain == "synthetic":
            weight = normalized(base[cell])
        else:
            marginal = np.zeros(int(cell.sum()), dtype=np.float64)
            selected_base = base[cell]
            selected_served = served[cell]
            for band in range(4):
                mask = ((selected_served >> band) & 1).astype(bool)
                denom = float(selected_base[mask].sum(dtype=np.float64))
                if denom > 0:
                    marginal[mask] += band_mix[band] * selected_base[mask] / denom
            weight = normalized(marginal)
        out[int(bucket)] = (selected_rows, weight)
    return out


def validate_non_anime(
    cells: dict[str, dict[str, np.ndarray]], catalog_root: Path, sampling: dict[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for domain in DOMAINS[1:]:
        expected = expected_non_anime(catalog_root, sampling, domain)
        actual_keys = {int(k.split(":", 1)[0]) for k in cells if k.endswith(f":{domain}")}
        if actual_keys != set(expected):
            raise ValueError(f"cell set mismatch for {domain}: {actual_keys ^ set(expected)}")
        total = 0
        for bucket, (rows, weights) in expected.items():
            actual = cells[f"{bucket}:{domain}"]
            if not np.array_equal(actual["row_id"], rows):
                raise ValueError(f"row selection mismatch: {bucket}:{domain}")
            if not np.allclose(actual["weight"], weights, rtol=2e-13, atol=2e-16):
                delta = float(np.max(np.abs(actual["weight"] - weights)))
                raise ValueError(f"weight mismatch: {bucket}:{domain}: max_abs={delta}")
            total += len(rows)
        report[domain] = {"rows": total, "cells": len(expected), "weights_exact": True}
    return report


def validate_bound_catalog_hashes(
    header: dict[str, Any], catalog_header: dict[str, Any], catalog_root: Path,
    workers: int,
) -> None:
    def one(domain: str) -> tuple[str, str]:
        return domain, file_sha256(catalog_root / f"{domain}.parquet")

    observed: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(DOMAINS))) as pool:
        futures = [pool.submit(one, domain) for domain in DOMAINS]
        for future in concurrent.futures.as_completed(futures):
            domain, digest = future.result()
            observed[domain] = digest
    for domain in DOMAINS:
        expected = catalog_header["files"][domain]["sha256"]
        bound = header["bindings"]["catalog_files"][domain]["sha256"]
        if observed[domain] != expected or observed[domain] != bound:
            raise ValueError(f"catalog file SHA drift: {domain}")


# Globals are intentional: Linux fork workers share these small immutable CLI
# values while each worker writes only its own mmap output.
def authoritative_anime_worker(task: tuple[str, str, str, str, int, str]) -> dict[str, Any]:
    source_s, work_s, repo_s, sampling_s, declared_bytes, declared_sha = task
    source_path = Path(source_s)
    before = source_path.stat()
    if before.st_size != declared_bytes:
        raise ValueError(f"anime source byte-size drift: {source_path}")
    observed_sha = file_sha256(source_path)
    after = source_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"anime source changed while hashing: {source_path}")
    if observed_sha != declared_sha:
        raise ValueError(f"anime source SHA drift: {source_path}")
    repo = Path(repo_s)
    sys.path.insert(0, str(repo))
    from training.dataline.config import load_sampling_config
    from training.dataline.manifest_index import load_anime_shard
    cfg = load_sampling_config(sampling_s)
    data = load_anime_shard(source_s, cfg, with_caption=False)
    priority = np.asarray(data["priority"], dtype=np.float64)
    output = Path(work_s) / f"{source_path.stem}.priority.npy"
    np.save(output, priority)
    parquet = pq.ParquetFile(source_s)
    offsets = [0]
    for rg in range(parquet.num_row_groups):
        offsets.append(offsets[-1] + parquet.metadata.row_group(rg).num_rows)
    if offsets[-1] != len(priority):
        raise RuntimeError(f"authoritative anime scan row mismatch: {source_s}")
    return {
        "source": source_s, "priority_path": str(output), "offsets": offsets,
        "rows": len(priority), "raw_sha256": observed_sha,
    }


def authoritative_anime_priority(
    args: argparse.Namespace, source_manifest: dict[str, Any], catalog_path: Path, work: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tasks = [
        (
            str(entry["source_path"]), str(work), str(args.repo_root), str(args.sampling_config),
            int(entry["bytes"]), str(entry["sha256"]),
        )
        for _, entry in sorted(source_manifest["files"].items())
    ]
    metadata: dict[str, Any] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = [pool.submit(authoritative_anime_worker, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            metadata[result["source"]] = result
            print(f"[anime-priority] {Path(result['source']).name}: {result['rows']:,}", flush=True)

    table = pq.read_table(
        catalog_path,
        columns=["row_id", "bucket_idx", "caption_parquet_path", "caption_row_group", "caption_row_in_group"],
        use_threads=True,
    )
    rows = arrow_numpy(table["row_id"], np.int64)
    buckets = arrow_numpy(table["bucket_idx"], np.int16)
    rg = arrow_numpy(table["caption_row_group"], np.int32)
    ri = arrow_numpy(table["caption_row_in_group"], np.int32)
    encoded = pc.dictionary_encode(table["caption_parquet_path"].combine_chunks())
    codes = np.asarray(encoded.indices.to_numpy(zero_copy_only=False), dtype=np.int32)
    paths = encoded.dictionary.to_pylist()
    priority = np.empty(len(rows), dtype=np.float64)
    for code, path in enumerate(paths):
        mask = codes == code
        meta = metadata[path]
        offsets = np.asarray(meta["offsets"], dtype=np.int64)
        pos = offsets[rg[mask]] + ri[mask]
        source_priority = np.load(meta["priority_path"], mmap_mode="r")
        priority[mask] = source_priority[pos]
    return rows, buckets, priority


def validate_anime_priority_and_repeat(
    args: argparse.Namespace, cells: dict[str, dict[str, np.ndarray]],
    source_manifest: dict[str, Any], catalog_root: Path, work: Path,
) -> dict[str, Any]:
    rows, buckets, priority = authoritative_anime_priority(
        args, source_manifest, catalog_root / "anime.parquet", work,
    )
    sys.path.insert(0, str(args.repo_root))
    sampler = importlib.import_module("training.dataline.sampler")
    total = 0
    for key, actual in sorted(cells.items()):
        bucket_s, domain = key.split(":", 1)
        if domain != "anime":
            continue
        bucket = int(bucket_s)
        selected_rows = actual["row_id"]
        if "priority" not in actual or actual["priority"].dtype != np.float64:
            raise ValueError(f"anime priority missing/bad dtype: {key}")
        # Catalog row ids are contiguous globally and anime starts at zero.
        expected_priority = priority[selected_rows]
        if not np.array_equal(actual["priority"], expected_priority):
            delta = float(np.max(np.abs(actual["priority"] - expected_priority)))
            raise ValueError(f"anime authoritative priority mismatch: {key}: {delta}")
        seed = cell_seed(args.overlay_id, int(args.root_seed), bucket)
        if seed != int(args.header["anime_contract"]["cell_seeds"][str(bucket)]):
            raise ValueError(f"anime cell seed mismatch: {key}")
        expected_repeat = sampler.compute_repeat_counts(
            expected_priority, 4, 0.25, np.random.default_rng(seed),
        ).astype(np.uint8)
        if not np.array_equal(actual["repeat"], expected_repeat):
            raise ValueError(f"anime repeat assignment mismatch: {key}")
        expected_weight = normalized(expected_repeat.astype(np.float64))
        if not np.array_equal(actual["weight"], expected_weight):
            raise ValueError(f"anime repeat-derived weight mismatch: {key}")
        if np.any(buckets[selected_rows] != bucket) or np.any(rows[selected_rows] != selected_rows):
            raise ValueError(f"anime row range/bucket mismatch: {key}")
        total += len(selected_rows)
    return {"rows": total, "priority_exact": True, "repeat_exact": True, "weights_exact": True}


def sidecar_part_worker(
    task: tuple[str, dict[str, Any], np.ndarray, bool],
) -> dict[str, Any]:
    root_s, entry, selected, verify_hash = task
    path = Path(root_s) / entry["name"]
    if verify_hash and file_sha256(path) != entry["sha256"]:
        raise ValueError(f"sidecar part SHA mismatch: {path}")
    table = pq.read_table(path, columns=["row_id", *NL_COLUMNS], use_threads=False)
    rows = arrow_numpy(table["row_id"], np.int64)
    if len(rows) != int(entry["rows"]):
        raise ValueError(f"sidecar part row mismatch: {path}")
    valid = np.zeros(len(rows), dtype=bool)
    for column in NL_COLUMNS:
        valid |= stripped_nonempty(table[column])
    chosen = selected[rows]
    if not np.array_equal(chosen, valid):
        false_selected = int((chosen & ~valid).sum())
        missed_valid = int((~chosen & valid).sum())
        raise ValueError(
            f"anime sidecar selection mismatch: {path}: selected_invalid={false_selected}, missed_valid={missed_valid}"
        )
    return {"rows": len(rows), "selected": int(chosen.sum()), "valid": int(valid.sum()), "file": entry["name"]}


def validate_full_anime_sidecar(
    args: argparse.Namespace, sidecar_header: dict[str, Any], cells: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    anime_rows = int(sidecar_header["domains"]["anime"]["row_count"])
    selected = np.zeros(anime_rows, dtype=bool)
    for key, arrays in cells.items():
        if key.endswith(":anime"):
            selected[arrays["row_id"]] = True
    files = sidecar_header["domains"]["anime"]["files"]
    tasks = [
        (str(args.caption_sidecar_root), entry, selected, not args.skip_sidecar_file_hashes)
        for entry in files
    ]
    totals = {"rows": 0, "selected": 0, "valid": 0, "files": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(sidecar_part_worker, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            for field in ("rows", "selected", "valid"):
                totals[field] += result[field]
            totals["files"] += 1
            if totals["files"] % 16 == 0:
                print(f"[anime-sidecar] {totals['files']}/{len(files)} parts", flush=True)
    if totals["rows"] != anime_rows or totals["selected"] != int(selected.sum()) or totals["selected"] != totals["valid"]:
        raise ValueError(f"anime sidecar aggregate mismatch: {totals}")
    totals.update({"selected_invalid": 0, "tag_only_selected": 0, "exact_selected_equals_all_strict_nl": True})
    return totals


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.overlay_root.resolve()
    header, header_file_sha = json_snapshot(root / "header.json")
    success, _ = json_snapshot(root / "_SUCCESS")
    args.header = header
    args.overlay_id = str(header.get("overlay_id"))
    args.root_seed = int(header.get("seed"))
    if header.get("format") != FORMAT or header.get("format_version") != 1:
        raise ValueError("unsupported overlay format")
    if success.get("header_sha256") != header_file_sha:
        raise ValueError("_SUCCESS/header SHA mismatch")
    success_copy = dict(success)
    declared_success_sha = success_copy.pop("success_sha256", None)
    if declared_success_sha != canonical_sha256(success_copy):
        raise ValueError("_SUCCESS self hash mismatch")
    header_copy = dict(header)
    declared_content_sha = header_copy.pop("content_sha256", None)
    if declared_content_sha != canonical_sha256(header_copy):
        raise ValueError("header content self hash mismatch")
    if header.get("domains") != list(DOMAINS) or len(header.get("buckets", [])) != 95:
        raise ValueError("domain/bucket table contract mismatch")
    assertions = header.get("semantic_assertions", {})
    if assertions != {"anime_all_rows_strip_nonempty_nl": True, "anime_tag_only_rows": 0}:
        raise ValueError("semantic_assertions contract mismatch")
    actual_excluded = {k: set(v) for k, v in header["excluded_row_ids_by_domain"].items()}
    if actual_excluded != EXPECTED_EXCLUDED or set(header["excluded_row_ids"]) != set().union(*EXPECTED_EXCLUDED.values()):
        raise ValueError("fixed exclusion set mismatch")

    catalog_header, catalog_header_sha = json_snapshot(args.catalog_header)
    sidecar_header, sidecar_header_sha = json_snapshot(args.caption_sidecar_header)
    source_manifest, source_manifest_sha = json_snapshot(args.anime_source_manifest)
    sampling, sampling_sha = json_snapshot(args.sampling_config)
    bindings = (
        ("catalog_header_sha256", catalog_header_sha),
        ("caption_sidecar_header_sha256", sidecar_header_sha),
        ("anime_source_manifest_sha256", source_manifest_sha),
        ("sampling_config_sha256", sampling_sha),
    )
    for field, digest in bindings:
        if header.get(field) != digest:
            raise ValueError(f"input binding drift: {field}")
    if catalog_header.get("catalog_hash") != header.get("catalog_hash"):
        raise ValueError("catalog hash drift")

    validate_bound_catalog_hashes(header, catalog_header, args.catalog_root, args.workers)

    cells = load_and_validate_cells(root, header)
    non_anime = validate_non_anime(cells, args.catalog_root, sampling)
    work = Path(tempfile.mkdtemp(prefix="semantic-overlay-validate-", dir=args.work_dir))
    try:
        if args.skip_authoritative_anime_priority:
            anime_priority = {"skipped": True}
        else:
            if args.repo_root is None:
                raise ValueError("--repo-root is required for authoritative anime validation")
            expected_manifest = header["bindings"].get("manifest_index_py")
            expected_sampler = header["bindings"].get("sampler_py")
            for path, expected in (
                (args.repo_root / "training/dataline/manifest_index.py", expected_manifest),
                (args.repo_root / "training/dataline/sampler.py", expected_sampler),
            ):
                if not expected or file_sha256(path) != expected["sha256"]:
                    raise ValueError(f"authoritative source SHA mismatch: {path}")
            anime_priority = validate_anime_priority_and_repeat(
                args, cells, source_manifest, args.catalog_root, work,
            )
        sidecar = validate_full_anime_sidecar(args, sidecar_header, cells)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    report: dict[str, Any] = {
        "status": "PASS", "format": "offline_arb_semantic_overlay_validation_v1",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overlay_id": header["overlay_id"], "overlay_header_sha256": header_file_sha,
        "catalog_hash": header["catalog_hash"], "cell_count": len(cells),
        "selected_rows": sum(len(x["row_id"]) for x in cells.values()),
        "tag_only_exposure": 0,
        "eligible_anime_rows": int(sidecar["selected"]),
        "checks": {
            "output_files_hash_schema_ranges_exclusions": "PASS",
            "non_anime": non_anime, "anime_priority_repeat": anime_priority,
            "anime_full_sidecar_strict_nl": sidecar,
        },
    }
    report["validation_sha256"] = canonical_sha256(report)
    atomic_json(args.receipt, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def self_test() -> None:
    values = pa.array([None, "", " \t\n", "valid", " \u3000valid\u3000"])
    assert stripped_nonempty(values).tolist() == [False, False, False, True, True]
    assert cell_seed("x", 1, 2) == cell_seed("x", 1, 2)
    assert cell_seed("x", 1, 2) != cell_seed("x", 1, 3)
    assert np.isclose(normalized(np.array([1.0, 3.0])).sum(), 1.0)
    print("SELF_TEST_PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--catalog-root", type=Path)
    parser.add_argument("--catalog-header", type=Path)
    parser.add_argument("--caption-sidecar-root", type=Path)
    parser.add_argument("--caption-sidecar-header", type=Path)
    parser.add_argument("--anime-source-manifest", type=Path)
    parser.add_argument("--sampling-config", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-sidecar-file-hashes", action="store_true")
    parser.add_argument("--skip-authoritative-anime-priority", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    required = (
        "overlay_root", "catalog_root", "catalog_header", "caption_sidecar_root",
        "caption_sidecar_header", "anime_source_manifest", "sampling_config", "receipt",
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
        print(f"FATAL semantic overlay validation: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
