#!/usr/bin/env python3
"""Precompute exact per-shard anime ``tags_general`` frequencies."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import os
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_CATALOG = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_arb_catalog_v1/anime.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_arb_animefreq_v1"
)
_CHUNK_BYTES = 8 << 20


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_name(shard_path: str) -> str:
    name = Path(shard_path).name
    if not name.endswith(".parquet"):
        raise ValueError(f"anime shard path does not end in .parquet: {shard_path}")
    stem = name.removesuffix(".parquet")
    if not stem:
        raise ValueError(f"anime shard has an empty file stem: {shard_path}")
    return f"{stem}.json.gz"


def _reset_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_SUCCESS").unlink(missing_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _read_catalog_hash(catalog_path: Path) -> str:
    header_path = catalog_path.parent / "header.json"
    try:
        header = json.loads(header_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read catalog header {header_path}: {exc}") from exc
    catalog_hash = header.get("catalog_hash") if isinstance(header, dict) else None
    if not isinstance(catalog_hash, str) or len(catalog_hash) != 64:
        raise RuntimeError(f"catalog header has invalid catalog_hash: {header_path}")
    try:
        int(catalog_hash, 16)
    except ValueError as exc:
        raise RuntimeError(f"catalog header has invalid catalog_hash: {header_path}") from exc
    return catalog_hash


def _distinct_shards(catalog_path: Path) -> list[str]:
    pf = pq.ParquetFile(catalog_path)
    column = "caption_parquet_path"
    if column not in pf.schema_arrow.names:
        raise RuntimeError(f"anime catalog lacks {column}: {catalog_path}")
    shards: set[str] = set()
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=[column], use_threads=False)
        for value in table.column(column).to_pylist():
            if not isinstance(value, str) or not value:
                raise RuntimeError(
                    f"anime catalog contains invalid {column} in row group {rg}: {value!r}"
                )
            shards.add(value)
    return sorted(shards)


def _anime_general_freq(path: str) -> Counter:
    """Match OfflineExactDataset._anime_general_freq's runtime scan exactly."""
    pf = pq.ParquetFile(path)
    if "tags_general" not in pf.schema_arrow.names:
        raise RuntimeError(f"anime caption source lacks tags_general: {path}")
    freq: Counter = Counter()
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["tags_general"], use_threads=False)
        for tags in table.column("tags_general").to_pylist():
            if tags:
                freq.update(tags)
    return freq


def _write_freq(path: Path, freq: Counter) -> None:
    if any(not isinstance(tag, str) for tag in freq):
        raise TypeError(f"frequency table contains a non-string tag: {path}")
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("wb") as raw:
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", mtime=0
            ) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                    json.dump(
                        dict(freq),
                        handle,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_freq(path: Path) -> dict[str, int]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    if any(
        not isinstance(tag, str) or type(count) is not int or count <= 0
        for tag, count in value.items()
    ):
        raise ValueError(f"invalid tag/count entry in {path}")
    return value


def _build_one(shard_path: str, output_dir: Path) -> tuple[str, dict[str, str]]:
    artifact_name = _artifact_name(shard_path)
    artifact_path = output_dir / artifact_name
    freq = _anime_general_freq(shard_path)
    _write_freq(artifact_path, freq)
    return artifact_name, {
        "source_shard": shard_path,
        "sha256": _sha256_file(artifact_path),
    }


def _write_header(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--workers", type=_positive_int, default=14)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog_path = Path(args.catalog).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    _reset_output_dir(output_dir)

    catalog_hash = _read_catalog_hash(catalog_path)
    shards = _distinct_shards(catalog_path)
    if len(shards) < 2:
        raise RuntimeError(f"self-check requires at least 2 anime shards, found {len(shards)}")

    artifacts: dict[str, str] = {}
    for shard_path in shards:
        artifact_name = _artifact_name(shard_path)
        previous = artifacts.setdefault(artifact_name, shard_path)
        if previous != shard_path:
            raise RuntimeError(
                f"anime shard basename collision for {artifact_name}: "
                f"{previous!r} and {shard_path!r}"
            )

    files: dict[str, dict[str, str]] = {}
    worker_count = min(args.workers, len(shards))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="anime-freq"
    ) as pool:
        futures = {
            pool.submit(_build_one, shard_path, output_dir): shard_path
            for shard_path in shards
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            artifact_name, metadata = future.result()
            files[artifact_name] = metadata
            completed += 1
            print(
                f"[{completed}/{len(shards)}] {metadata['source_shard']} "
                f"-> {artifact_name}",
                flush=True,
            )

    sampled_shards = random.SystemRandom().sample(shards, 2)
    for shard_path in sampled_shards:
        rebuilt = _anime_general_freq(shard_path)
        written = _read_freq(output_dir / _artifact_name(shard_path))
        if written != dict(rebuilt):
            raise AssertionError(
                f"self-check frequency mismatch for {shard_path}: "
                f"runtime_tags={len(rebuilt)}, written_tags={len(written)}"
            )
        print(f"[self-check] PASS {shard_path}", flush=True)

    header = {
        "schema_version": "offline_arb_animefreq_v1",
        "generated_at": _utc_now(),
        "catalog_hash": catalog_hash,
        "catalog_path": str(catalog_path),
        "shard_count": len(shards),
        "files": dict(sorted(files.items())),
        "self_check": {
            "pass": True,
            "sampled_shards": sampled_shards,
        },
    }
    _write_header(output_dir / "header.json", header)
    (output_dir / "_SUCCESS").touch()
    print(
        json.dumps(
            {
                "status": "PASS",
                "shard_count": len(shards),
                "catalog_hash": catalog_hash,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
