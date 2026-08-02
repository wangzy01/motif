#!/usr/bin/env python3
"""Build a catalog-bound Stage512a Anime priority sidecar.

The existing Stage512a catalog intentionally stores image/caption locators but
omits the streaming sampler's Anime priority.  This writer reconstructs the
priority from the exact source rows addressed by the catalog, verifies the key
join for every row, and publishes a read-only sidecar.  It never reads images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


MIB = 1 << 20
SCHEMA = "stage512a_anime_priority_sidecar_v1"
PARQUET_NAME = "anime_priority.parquet"
HEADER_NAME = "header.json"
SUCCESS_NAME = "_SUCCESS"
SIDE_SCHEMA = pa.schema(
    [
        pa.field("row_id", pa.int64(), nullable=False),
        pa.field("bucket_idx", pa.int16(), nullable=False),
        pa.field("priority", pa.float64(), nullable=False),
        pa.field("repeat_score", pa.float64(), nullable=False),
    ]
)


class BuildError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def write_json_exclusive(path: Path, value: Any, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise BuildError(f"short write: {path}")
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, mode)
    finally:
        os.close(fd)


def require_regular(path: Path, *, mode: int | None = None) -> os.stat_result:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise BuildError(f"unsafe regular input: {path}")
    if mode is not None and stat.S_IMODE(st.st_mode) != mode:
        raise BuildError(f"mode drift for {path}: {stat.S_IMODE(st.st_mode):04o}")
    return st


def arrow_semantic_bytes(keys: pa.Array | pa.ChunkedArray, priority: np.ndarray) -> bytes:
    table = pa.table(
        {
            "key": keys.combine_chunks() if isinstance(keys, pa.ChunkedArray) else keys,
            "priority": pa.array(priority, type=pa.float64()),
        }
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def compute_priority(table: pa.Table, sampling_cfg: Any, manifest_module: Any) -> np.ndarray:
    required = {
        "ct_weight",
        "quality_z",
        "hps_v3_score_qwen3vl",
        "ct_lanes",
        "risk_bucket",
        "tags_general",
        manifest_module.YOLO_ADDED_TEXT_AREA_COL,
    }
    missing = sorted(required - set(table.column_names))
    if missing:
        raise BuildError(f"Anime source priority columns missing: {missing}")

    n = table.num_rows
    ct_weight = np.asarray(
        [1.0 if value is None else float(value) for value in table["ct_weight"].to_pylist()],
        dtype=np.float64,
    )
    qz = np.asarray(
        [0.0 if value is None else float(value) for value in table["quality_z"].to_pylist()],
        dtype=np.float64,
    )
    hps = np.asarray(
        [0.0 if value is None else float(value) for value in table["hps_v3_score_qwen3vl"].to_pylist()],
        dtype=np.float64,
    )
    lanes = [list(value) if value is not None else [] for value in table["ct_lanes"].to_pylist()]
    risk = np.asarray(table["risk_bucket"].to_pylist(), dtype=object)
    tags = table["tags_general"].to_pylist()
    yolo = np.asarray(
        [math.inf if value is None else float(value)
         for value in table[manifest_module.YOLO_ADDED_TEXT_AREA_COL].to_pylist()],
        dtype=np.float64,
    )
    if any(array.shape != (n,) for array in (ct_weight, qz, hps, risk, yolo)):
        raise BuildError("priority source column length drift")

    spec = sampling_cfg.anime_sampling
    priority = ct_weight.copy()
    priority += np.where(
        (qz >= spec.quality_bonus_qz_min) & (hps >= spec.quality_bonus_hps_min),
        spec.quality_bonus_value,
        0.0,
    )
    priority += np.asarray([spec.lane_bonus_for(value) for value in lanes], dtype=np.float64)
    clean = manifest_module.composition_clean_mask(tags, yolo, qz)
    priority += np.where(clean, spec.composition_clean_bonus, 0.0)
    priority = np.clip(priority, 1.0, spec.max_priority)
    capped = np.isin(risk, np.asarray(spec.lane_cap_buckets, dtype=object))
    priority = np.where(capped, priority * spec.lane_cap_multiplier, priority)

    if not np.all(np.isfinite(priority)) or np.any(priority <= 0):
        raise BuildError("computed Anime priority is non-finite or non-positive")

    # Exact scalar-production comparison on a deterministic spread of rows.
    if n:
        checks = np.unique(np.linspace(0, n - 1, min(17, n), dtype=np.int64))
        for index in checks.tolist():
            expected = manifest_module.anime_priority(
                float(ct_weight[index]),
                lanes[index],
                risk[index],
                float(qz[index]),
                float(hps[index]),
                sampling_cfg,
                tags[index],
                float(yolo[index]),
            )
            if float(priority[index]) != float(expected):
                raise BuildError(
                    f"vector/scalar Anime priority mismatch at local row {index}: "
                    f"{priority[index]} != {expected}"
                )
    return priority


def source_paths_from_catalog(catalog: pq.ParquetFile) -> list[str]:
    found: set[str] = set()
    for rg in range(catalog.num_row_groups):
        table = catalog.read_row_group(rg, columns=["caption_parquet_path"], use_threads=False)
        found.update(str(value) for value in table["caption_parquet_path"].to_pylist())
    if not found:
        raise BuildError("Anime catalog contains no source parquet path")
    return sorted(found)


def load_source_priorities(
    source_paths: list[str], sampling_cfg: Any, manifest_module: Any
) -> tuple[dict[tuple[str, int], tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    semantic = hashlib.sha256()
    source_report: dict[str, Any] = {}
    columns = [
        "key",
        "ct_weight",
        "quality_z",
        "hps_v3_score_qwen3vl",
        "ct_lanes",
        "risk_bucket",
        "tags_general",
        manifest_module.YOLO_ADDED_TEXT_AREA_COL,
    ]
    total_rows = 0
    for path_text in source_paths:
        path = Path(path_text)
        require_regular(path)
        pf = pq.ParquetFile(path)
        path_rows = 0
        row_group_rows: list[int] = []
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=columns, use_threads=False)
            priority = compute_priority(table, sampling_cfg, manifest_module)
            keys = np.asarray(table["key"].to_pylist(), dtype=object)
            if keys.shape != priority.shape or any(not isinstance(value, str) or not value for value in keys):
                raise BuildError(f"bad Anime key column at {path_text} row-group {rg}")
            cache[(path_text, rg)] = (keys, priority)
            semantic.update(canonical_bytes({"path": path_text, "row_group": rg, "rows": len(keys)}))
            semantic.update(arrow_semantic_bytes(table["key"], priority))
            path_rows += len(keys)
            total_rows += len(keys)
            row_group_rows.append(len(keys))
        source_report[path_text] = {
            "bytes": int(path.stat().st_size),
            "rows": path_rows,
            "row_groups": pf.num_row_groups,
            "row_group_rows": row_group_rows,
        }
    return cache, {
        "semantic_sha256": semantic.hexdigest(),
        "source_rows": total_rows,
        "sources": source_report,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve(strict=True)
    catalog_root = Path(args.catalog_root).resolve(strict=True)
    output = Path(args.output_root).resolve(strict=False)
    sampling_path = Path(args.sampling_config).resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise BuildError(f"output root already exists: {output}")
    if not catalog_root.is_dir() or catalog_root.is_symlink():
        raise BuildError(f"unsafe catalog root: {catalog_root}")
    for path in (sampling_path, catalog_root / "header.json", catalog_root / "anime.parquet"):
        require_regular(path)

    sys.path.insert(0, str(repo))
    from training.dataline import manifest_index as manifest_module
    from training.dataline.config import load_sampling_config

    sampling_cfg = load_sampling_config(str(sampling_path))
    catalog_header = json.loads((catalog_root / "header.json").read_text(encoding="utf-8"))
    if catalog_header.get("catalog_hash") != args.catalog_hash:
        raise BuildError(
            f"catalog hash drift: {catalog_header.get('catalog_hash')} != {args.catalog_hash}"
        )
    if catalog_header.get("stage") != "512a" or int(catalog_header.get("image_size", -1)) != 512:
        raise BuildError("catalog is not the Stage512a/512 authority")

    catalog = pq.ParquetFile(catalog_root / "anime.parquet")
    source_paths = source_paths_from_catalog(catalog)
    cache, source_report = load_source_priorities(source_paths, sampling_cfg, manifest_module)

    output.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output, 0o700)
    parquet_path = output / PARQUET_NAME
    writer = pq.ParquetWriter(
        parquet_path,
        SIDE_SCHEMA,
        compression="zstd",
        compression_level=6,
        use_dictionary=False,
        write_statistics=True,
    )
    mapped = 0
    row_hash = hashlib.sha256()
    priority_sum = 0.0
    score_sum = 0.0
    priority_min = math.inf
    priority_max = -math.inf
    try:
        expected_next = 0
        columns = [
            "row_id",
            "bucket_idx",
            "key",
            "caption_parquet_path",
            "caption_row_group",
            "caption_row_in_group",
        ]
        for rg in range(catalog.num_row_groups):
            table = catalog.read_row_group(rg, columns=columns, use_threads=False)
            row_id = np.asarray(table["row_id"].to_numpy(zero_copy_only=False), dtype=np.int64)
            bucket = np.asarray(table["bucket_idx"].to_numpy(zero_copy_only=False), dtype=np.int16)
            keys = np.asarray(table["key"].to_pylist(), dtype=object)
            paths = np.asarray(table["caption_parquet_path"].to_pylist(), dtype=object)
            src_rg = np.asarray(table["caption_row_group"].to_numpy(zero_copy_only=False), dtype=np.int64)
            within = np.asarray(table["caption_row_in_group"].to_numpy(zero_copy_only=False), dtype=np.int64)
            n = table.num_rows
            if not np.array_equal(row_id, np.arange(expected_next, expected_next + n, dtype=np.int64)):
                raise BuildError(f"Anime catalog row_id order drift in row-group {rg}")
            priority = np.empty(n, dtype=np.float64)
            pairs = sorted({(str(paths[i]), int(src_rg[i])) for i in range(n)})
            for source_path, source_rg in pairs:
                mask = (paths == source_path) & (src_rg == source_rg)
                positions = np.flatnonzero(mask)
                source = cache.get((source_path, source_rg))
                if source is None:
                    raise BuildError(f"catalog points to an unknown source group: {source_path}#{source_rg}")
                source_keys, source_priority = source
                indexes = within[positions]
                if np.any(indexes < 0) or np.any(indexes >= len(source_priority)):
                    raise BuildError(f"source locator out of range: {source_path}#{source_rg}")
                joined_keys = source_keys[indexes]
                if not np.array_equal(joined_keys, keys[positions]):
                    bad = positions[np.flatnonzero(joined_keys != keys[positions])[:1]][0]
                    raise BuildError(
                        f"catalog/source Anime key mismatch row_id={int(row_id[bad])}: "
                        f"{keys[bad]!r} != {joined_keys[np.flatnonzero(joined_keys != keys[positions])[0]]!r}"
                    )
                priority[positions] = source_priority[indexes]
            if not np.all(np.isfinite(priority)) or np.any(priority <= 0):
                raise BuildError(f"invalid joined priority in catalog row-group {rg}")
            score = np.maximum(priority - 1.0, 0.0)
            out = pa.Table.from_arrays(
                [
                    pa.array(row_id, type=pa.int64()),
                    pa.array(bucket, type=pa.int16()),
                    pa.array(priority, type=pa.float64()),
                    pa.array(score, type=pa.float64()),
                ],
                schema=SIDE_SCHEMA,
            )
            writer.write_table(out, row_group_size=n)
            row_hash.update(row_id.astype("<i8", copy=False).tobytes())
            row_hash.update(bucket.astype("<i2", copy=False).tobytes())
            row_hash.update(priority.astype("<f8", copy=False).tobytes())
            priority_sum += float(priority.sum(dtype=np.float64))
            score_sum += float(score.sum(dtype=np.float64))
            priority_min = min(priority_min, float(priority.min(initial=math.inf)))
            priority_max = max(priority_max, float(priority.max(initial=-math.inf)))
            expected_next += n
            mapped += n
    finally:
        writer.close()

    expected_rows = int(catalog_header["coverage"]["anime"]["rows_kept"])
    if mapped != catalog.metadata.num_rows or mapped != expected_rows:
        raise BuildError(
            f"Anime sidecar row count drift: mapped={mapped}, parquet={catalog.metadata.num_rows}, "
            f"header={expected_rows}"
        )
    os.chmod(parquet_path, 0o444)
    require_regular(parquet_path, mode=0o444)

    manifest_index_path = repo / "training/dataline/manifest_index.py"
    config_module_path = repo / "training/dataline/config.py"
    header: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS",
        "generated_at": utc_now(),
        "catalog": {
            "root": str(catalog_root),
            "catalog_hash": args.catalog_hash,
            "header_sha256": sha256_file(catalog_root / "header.json"),
            "anime_parquet_rows": mapped,
        },
        "sampling": {
            "path": str(sampling_path),
            "sha256": sha256_file(sampling_path),
            "max_repeat": int(sampling_cfg.anime_sampling.max_repeat),
            "extra_frac": float(sampling_cfg.anime_sampling.extra_frac),
        },
        "producer_sources": {
            "manifest_index.py": sha256_file(manifest_index_path),
            "config.py": sha256_file(config_module_path),
            "this_script": sha256_file(Path(__file__).resolve(strict=True)),
        },
        "source_semantics": source_report,
        "output": {
            "file": PARQUET_NAME,
            "bytes": int(parquet_path.stat().st_size),
            "sha256": sha256_file(parquet_path),
            "row_semantic_sha256": row_hash.hexdigest(),
            "rows": mapped,
            "priority_min": priority_min,
            "priority_max": priority_max,
            "priority_sum": priority_sum,
            "repeat_score_sum": score_sum,
        },
    }
    header["identity_sha256"] = hashlib.sha256(canonical_bytes(header)).hexdigest()
    write_json_exclusive(output / HEADER_NAME, header)
    success = {
        "schema": SCHEMA,
        "status": "PASS",
        "header_sha256": sha256_file(output / HEADER_NAME),
        "parquet_sha256": header["output"]["sha256"],
        "identity_sha256": header["identity_sha256"],
    }
    write_json_exclusive(output / SUCCESS_NAME, success)
    os.chmod(output, 0o555)
    root_st = output.lstat()
    if stat.S_IMODE(root_st.st_mode) != 0o555 or root_st.st_nlink < 2:
        raise BuildError("sidecar root did not seal read-only")
    for member in (parquet_path, output / HEADER_NAME, output / SUCCESS_NAME):
        require_regular(member, mode=0o444)
    return header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--catalog-root", required=True)
    parser.add_argument("--sampling-config", required=True)
    parser.add_argument("--catalog-hash", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        header = build(parse_args())
    except Exception as exc:
        print(f"ANIME_PRIORITY_SIDECAR_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        "ANIME_PRIORITY_SIDECAR_PASS "
        f"rows={header['output']['rows']} identity={header['identity_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
