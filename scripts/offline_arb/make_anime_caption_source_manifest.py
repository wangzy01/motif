#!/usr/bin/env python3
"""Freeze the 17 immutable anime caption parquet sources by size and SHA-256."""

from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq


BASE = Path("/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop")
ANIME_FREQ_HEADER = BASE / "offline_arb_animefreq_v1/header.json"
CATALOG_HEADER = BASE / "offline_arb_catalog_v1/header.json"
ANIME_CATALOG = BASE / "offline_arb_catalog_v1/anime.parquet"
ANIME_FREQ_SUCCESS = BASE / "offline_arb_animefreq_v1/_SUCCESS"
OUT_ROOT = BASE / "offline_arb_anime_caption_source_manifest_v1"
SOURCE_PARENT = Path("/s3/evan/anime_core_ditv04_ct_soft/data")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_object_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str, *, encoding: str) -> None:
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    try:
        with tmp.open("w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def inspect_source(path: Path) -> tuple[str, dict[str, Any]]:
    before = path.stat()
    digest = hashlib.sha256()
    bytes_read = 0
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
            bytes_read += len(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source changed while hashing: {path}")
    if bytes_read != after.st_size:
        raise RuntimeError(
            f"source short read while hashing: {path}: read={bytes_read}, stat={after.st_size}"
        )
    return path.name, {
        "source_path": str(path),
        "bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


def catalog_caption_sources(path: Path, expected: set[str]) -> set[str]:
    before = path.stat()
    parquet = pq.ParquetFile(path)
    field = parquet.schema_arrow.field("caption_parquet_path")
    if str(field.type) != "string" or field.nullable:
        raise ValueError(f"catalog caption_parquet_path column missing: {path}")
    sources: set[str] = set()
    for row_group in range(parquet.num_row_groups):
        values = parquet.read_row_group(
            row_group, columns=["caption_parquet_path"], use_threads=False
        ).column("caption_parquet_path")
        if values.null_count:
            raise ValueError(f"null catalog caption source in row group {row_group}")
        for value in pc.unique(values).to_pylist():
            if not isinstance(value, str) or not value:
                raise ValueError(f"invalid catalog caption source value: {value!r}")
            if value not in expected:
                raise ValueError(f"unexpected catalog caption source: {value}")
            sources.add(value)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"anime catalog changed while scanning: {path}")
    return sources


def _main_locked() -> int:
    try:
        catalog, catalog_header_sha = read_object_snapshot(CATALOG_HEADER)
        freq, animefreq_header_sha = read_object_snapshot(ANIME_FREQ_HEADER)
        catalog_hash = catalog.get("catalog_hash")
        if not isinstance(catalog_hash, str) or not catalog_hash:
            raise ValueError("catalog_hash is empty")
        if freq.get("catalog_hash") != catalog_hash:
            raise ValueError("animefreq/catalog hash mismatch")
        if catalog.get("schema_version") != "offline_arb_catalog_v1":
            raise ValueError("unsupported catalog schema")
        if freq.get("schema_version") != "offline_arb_animefreq_v1":
            raise ValueError("unsupported animefreq schema")
        if not ANIME_FREQ_SUCCESS.is_file():
            raise FileNotFoundError(f"animefreq _SUCCESS missing: {ANIME_FREQ_SUCCESS}")
        if not isinstance(freq.get("self_check"), dict) or freq["self_check"].get("pass") is not True:
            raise ValueError("animefreq self_check is not PASS")
        declared = freq.get("files")
        if not isinstance(declared, dict) or len(declared) != 17 or freq.get("shard_count") != 17:
            raise ValueError("animefreq must declare exactly 17 shards")

        sources: list[Path] = []
        for artifact_name, entry in sorted(declared.items()):
            if not isinstance(entry, dict):
                raise ValueError(f"invalid animefreq entry: {artifact_name!r}")
            source = Path(str(entry.get("source_shard", "")))
            if artifact_name != f"{source.stem}.json.gz":
                raise ValueError(
                    f"animefreq artifact/source basename mismatch: {artifact_name!r}/{source.name!r}"
                )
            if source.parent != SOURCE_PARENT or source.suffix != ".parquet":
                raise ValueError(f"unexpected anime caption source: {source}")
            if not source.is_file():
                raise FileNotFoundError(source)
            sources.append(source)
        if len({path.name for path in sources}) != 17:
            raise ValueError("anime caption source basenames are not unique")
        declared_sources = {str(path) for path in sources}
        anime_catalog_info = catalog.get("files", {}).get("anime")
        if (
            not isinstance(anime_catalog_info, dict)
            or anime_catalog_info.get("name") != ANIME_CATALOG.name
            or not isinstance(anime_catalog_info.get("rows"), int)
            or not isinstance(anime_catalog_info.get("sha256"), str)
        ):
            raise ValueError("catalog files.anime identity is malformed")
        parquet_rows = pq.ParquetFile(ANIME_CATALOG).metadata.num_rows
        if parquet_rows != anime_catalog_info["rows"]:
            raise ValueError(
                f"anime catalog row count mismatch: {parquet_rows}/{anime_catalog_info['rows']}"
            )
        catalog_before_scan = ANIME_CATALOG.stat()
        catalog_sources = catalog_caption_sources(ANIME_CATALOG, declared_sources)
        catalog_after_scan = ANIME_CATALOG.stat()
        catalog_stat_identity = (
            catalog_after_scan.st_size,
            catalog_after_scan.st_mtime_ns,
        )
        if catalog_stat_identity != (
            catalog_before_scan.st_size,
            catalog_before_scan.st_mtime_ns,
        ):
            raise RuntimeError("anime catalog changed across source-set scan")
        if catalog_sources != declared_sources:
            raise ValueError(
                "anime catalog/animefreq caption source set mismatch: "
                f"missing={sorted(catalog_sources-declared_sources)}, "
                f"added={sorted(declared_sources-catalog_sources)}"
            )

        print("hashing 17 anime caption parquet sources with 2 readers", file=sys.stderr)
        files: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(inspect_source, path) for path in sources]
            for future in concurrent.futures.as_completed(futures):
                name, entry = future.result()
                files[name] = entry

        catalog_before_hash = ANIME_CATALOG.stat()
        if (
            catalog_before_hash.st_size,
            catalog_before_hash.st_mtime_ns,
        ) != catalog_stat_identity:
            raise RuntimeError("anime catalog changed after source-set scan")
        catalog_live_sha = file_sha256(ANIME_CATALOG)
        catalog_after_hash = ANIME_CATALOG.stat()
        if (
            catalog_after_hash.st_size,
            catalog_after_hash.st_mtime_ns,
        ) != catalog_stat_identity:
            raise RuntimeError("anime catalog changed while hashing")
        if catalog_live_sha != anime_catalog_info["sha256"]:
            raise ValueError(
                "anime catalog SHA mismatch: "
                f"live={catalog_live_sha}, header={anime_catalog_info['sha256']}"
            )

        # Bind the exact header snapshots that selected these 17 files.  Refuse a
        # mixed manifest if either upstream header changed during the long hash.
        if file_sha256(CATALOG_HEADER) != catalog_header_sha:
            raise RuntimeError("catalog header changed while hashing caption sources")
        if file_sha256(ANIME_FREQ_HEADER) != animefreq_header_sha:
            raise RuntimeError("animefreq header changed while hashing caption sources")

        content = {
            "schema_version": "offline_arb_anime_caption_source_v1",
            "catalog_hash": catalog_hash,
            "animefreq_header_sha256": animefreq_header_sha,
            "catalog_caption_source_set_sha256": canonical_sha256(sorted(catalog_sources)),
            "catalog_anime_sha256": anime_catalog_info["sha256"],
            "source_parent": str(SOURCE_PARENT),
            "file_count": 17,
            "total_bytes": sum(int(entry["bytes"]) for entry in files.values()),
            "files": {name: files[name] for name in sorted(files)},
        }
        header = dict(content)
        header["generated_at"] = datetime.now(timezone.utc).isoformat()
        header["manifest_sha256"] = canonical_sha256(content)

        header_path = OUT_ROOT / "header.json"
        atomic_write_text(
            header_path,
            json.dumps(header, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        atomic_write_text(
            OUT_ROOT / "_SUCCESS",
            header["manifest_sha256"] + "\n",
            encoding="ascii",
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "header": str(header_path),
                    "manifest_sha256": header["manifest_sha256"],
                    "files": 17,
                    "total_bytes": header["total_bytes"],
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:  # fail closed with a concise JCC log
        print(f"FATAL anime caption source manifest: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def main() -> int:
    try:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        lock_path = OUT_ROOT / ".manifest.lock"
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            return _main_locked()
    except Exception as exc:
        print(f"FATAL anime caption source manifest lock: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
