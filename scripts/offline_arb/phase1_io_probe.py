#!/usr/bin/env python3
"""OFFLINE_ARB_V1 Phase-1 I/O 探针：Task B / C / E / F。

全部数据路径只读打开；运行产物（JSON 与 Task C 原型索引）只写 ``--output-dir``。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import socket
import sys
import tarfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from training.dataline.image_fetch import gpic_path


SEED = 20260709
MIB = 1 << 20
MROOT = Path("/s3-mewtant-raw-datasets/jasperai/monet/v1.1.0")
ANIME_RICH = Path("/s3/evan/anime_core_ditv04_ct_soft/data")
GPIC_MANIFEST = Path("/s3-pixai-test-uw2/evan/mot_filtering_v1/t5_manifest/rich_v1/gpic")
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def latency_stats_ms(values: Sequence[float]) -> dict[str, Any]:
    """Input is seconds; output is milliseconds."""
    return {
        "n": len(values),
        "p50_ms": None if not values else percentile(values, 50) * 1000.0,
        "p90_ms": None if not values else percentile(values, 90) * 1000.0,
        "p95_ms": None if not values else percentile(values, 95) * 1000.0,
        "p99_ms": None if not values else percentile(values, 99) * 1000.0,
        "max_ms": None if not values else max(values) * 1000.0,
    }


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


def first_n_tars(base: Path, n: int) -> list[Path]:
    found: list[Path] = []
    if not base.is_dir():
        return found
    for group in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name):
        for tar in sorted(group.glob("*.tar")):
            found.append(tar)
            if len(found) >= n:
                return found
    return found


def representative_tar(subsource: str) -> Path | None:
    base = MROOT / subsource
    direct = base / "0-0" / "000000.tar"
    if direct.is_file():
        return direct
    found = first_n_tars(base, 1)
    return found[0] if found else None


def timed_read_pair(path_s: str, offset: int, size: int) -> dict[str, Any]:
    with open(path_s, "rb", buffering=0) as f:
        f.seek(offset)
        started = time.perf_counter()
        cold = f.read(size)
        cold_s = time.perf_counter() - started
        f.seek(offset)
        started = time.perf_counter()
        hot = f.read(size)
        hot_s = time.perf_counter() - started
    return {
        "offset": offset,
        "requested_bytes": size,
        "cold_bytes": len(cold),
        "hot_bytes": len(hot),
        "cold_seconds": cold_s,
        "hot_seconds": hot_s,
        "prefix_hex": cold[:8].hex(),
    }


def endpoint_probe(path_s: str) -> dict[str, Any]:
    size = os.stat(path_s).st_size
    if size < 2 * MIB:
        raise ValueError(f"file too small for endpoint probe: {size}")
    # Task wording intentionally asks for size-2MiB, then a 1MiB read.
    tail = timed_read_pair(path_s, size - 2 * MIB, MIB)
    head = timed_read_pair(path_s, 0, MIB)
    return {"file_size": size, "tail": tail, "head": head, "gzip_magic": head["prefix_hex"].startswith("1f8b")}


def aligned_random_offsets(size: int, read_size: int, n: int, seed: int) -> list[int]:
    max_offset = size - read_size
    if max_offset < 0:
        return []
    slots = max_offset // 4096 + 1
    n = min(n, slots)
    rng = np.random.default_rng(seed)
    if n == slots:
        chosen = np.arange(slots, dtype=np.int64)
    else:
        chosen = rng.choice(slots, size=n, replace=False)
    return (chosen.astype(np.int64) * 4096).tolist()


def random_read_probe(path_s: str, read_size: int, n: int, seed: int) -> dict[str, Any]:
    size = os.stat(path_s).st_size
    offsets = aligned_random_offsets(size, read_size, n, seed)
    cold: list[float] = []
    hot: list[float] = []
    short_reads = 0
    offset_hash = hashlib.sha256()
    with open(path_s, "rb", buffering=0) as f:
        for offset in offsets:
            offset_hash.update(int(offset).to_bytes(8, "little"))
            f.seek(offset)
            started = time.perf_counter()
            first = f.read(read_size)
            cold.append(time.perf_counter() - started)
            f.seek(offset)
            started = time.perf_counter()
            second = f.read(read_size)
            hot.append(time.perf_counter() - started)
            if len(first) != read_size or len(second) != read_size:
                short_reads += 1
    return {
        "read_size_bytes": read_size,
        "requested_reads": n,
        "completed_pairs": len(offsets),
        "short_read_pairs": short_reads,
        "offsets_sha256": offset_hash.hexdigest(),
        "cold": latency_stats_ms(cold),
        "hot": latency_stats_ms(hot),
    }


def sequential_read_probe(path_s: str, chunk_size: int = 8 * MIB) -> dict[str, Any]:
    expected = os.stat(path_s).st_size
    read_bytes = 0
    started = time.perf_counter()
    with open(path_s, "rb", buffering=0) as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            read_bytes += len(block)
    elapsed = time.perf_counter() - started
    return {
        "expected_bytes": expected,
        "read_bytes": read_bytes,
        "seconds": elapsed,
        "throughput_mib_s": (read_bytes / MIB) / elapsed if elapsed else None,
        "complete": read_bytes == expected,
    }


def scan_tar_sequential(path_s: str, index_path_s: str, domain: str, subsource: str) -> dict[str, Any]:
    """Stream through payload bytes with tarfile r|; index only training image members."""
    keys: list[str] = []
    member_names: list[str] = []
    header_offsets: list[int] = []
    data_offsets: list[int] = []
    sizes: list[int] = []
    total_members = 0
    regular_members = 0
    locator_digest = hashlib.sha256()
    scan_started = time.perf_counter()
    with tarfile.open(path_s, "r|") as tf:
        for member in tf:
            total_members += 1
            if not member.isfile():
                continue
            regular_members += 1
            if not member.name.endswith(".jpg"):
                continue
            key = member.name[:-4].split("/")[-1]
            keys.append(key)
            member_names.append(member.name)
            header_offsets.append(int(member.offset))
            data_offsets.append(int(member.offset_data))
            sizes.append(int(member.size))
            locator_digest.update(f"{key}\0{member.offset_data}\0{member.size}\n".encode())
    scan_seconds = time.perf_counter() - scan_started
    duplicate_keys = len(keys) - len(set(keys))
    table = pa.table({
        "key": pa.array(keys, type=pa.string()),
        "tar_path": pa.array([path_s] * len(keys), type=pa.string()),
        "member_name": pa.array(member_names, type=pa.string()),
        "header_offset": pa.array(header_offsets, type=pa.int64()),
        "data_offset": pa.array(data_offsets, type=pa.int64()),
        "size": pa.array(sizes, type=pa.int64()),
        "domain": pa.array([domain] * len(keys), type=pa.string()),
        "sub_source": pa.array([subsource] * len(keys), type=pa.string()),
    })
    index_path = Path(index_path_s)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_name(f"{index_path.name}.tmp.{os.getpid()}")
    write_started = time.perf_counter()
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, index_path)
    write_seconds = time.perf_counter() - write_started
    total_build_seconds = time.perf_counter() - scan_started
    return {
        "tar_path": path_s,
        "tar_bytes": os.stat(path_s).st_size,
        "total_members": total_members,
        "regular_members": regular_members,
        "image_members": len(keys),
        "other_regular_members": regular_members - len(keys),
        "duplicate_image_keys": duplicate_keys,
        "locator_sha256": locator_digest.hexdigest(),
        "scan_seconds": scan_seconds,
        "index_write_seconds": write_seconds,
        "index_materialize_and_write_seconds": total_build_seconds - scan_seconds,
        "total_build_seconds": total_build_seconds,
        "index_path": str(index_path),
        "index_bytes": index_path.stat().st_size,
        "index_bytes_per_row": index_path.stat().st_size / len(keys) if keys else None,
    }


def scan_tar_seek_jump(path_s: str) -> dict[str, Any]:
    """Random-mode tar iteration seeks from header to header and handles POSIX/PAX names."""
    images = 0
    total_members = 0
    digest = hashlib.sha256()
    started = time.perf_counter()
    with tarfile.open(path_s, "r:") as tf:
        for member in tf:
            total_members += 1
            if member.isfile() and member.name.endswith(".jpg"):
                key = member.name[:-4].split("/")[-1]
                images += 1
                digest.update(f"{key}\0{member.offset_data}\0{member.size}\n".encode())
    return {
        "tar_path": path_s,
        "total_members": total_members,
        "image_members": images,
        "locator_sha256": digest.hexdigest(),
        "scan_seconds": time.perf_counter() - started,
    }


STEP_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "endpoint": endpoint_probe,
    "random": random_read_probe,
    "sequential": sequential_read_probe,
    "tar_sequential": scan_tar_sequential,
    "tar_seek_jump": scan_tar_seek_jump,
}


def step_worker(function_name: str, args: tuple[Any, ...], result_path_s: str) -> None:
    try:
        result = STEP_FUNCTIONS[function_name](*args)
        payload = {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001 - serialize child failures
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
        }
    atomic_json(Path(result_path_s), payload)


def run_step(
    output_dir: Path,
    label: str,
    function_name: str,
    function_args: tuple[Any, ...],
    timeout_s: float,
) -> dict[str, Any]:
    """Run one potentially blocking FUSE step in a killable forked process."""
    step_dir = output_dir / ".phase1_steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    result_path = step_dir / f"{label}.{os.getpid()}.json"
    try:
        result_path.unlink()
    except FileNotFoundError:
        pass
    ctx = mp.get_context("fork")
    proc = ctx.Process(target=step_worker, args=(function_name, function_args, str(result_path)))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(10)
        return {"ok": False, "timeout": True, "timeout_seconds": timeout_s, "label": label}
    if not result_path.is_file():
        return {"ok": False, "error": f"step exited {proc.exitcode} without result", "label": label}
    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    try:
        result_path.unlink()
    except OSError:
        pass
    payload["label"] = label
    payload["exit_code"] = proc.exitcode
    return payload


def task_b(args: argparse.Namespace) -> dict[str, Any]:
    n = args.sample_n if args.sample_n is not None else 500
    if n <= 0:
        raise ValueError("--sample-n must be positive")
    output_dir = Path(args.output_dir)
    files: dict[str, Any] = {}
    incomplete: list[str] = []
    hard_errors: list[str] = []
    range_flags: list[bool] = []
    for file_i, subsource in enumerate(("cc12m", "coyo")):
        tar = representative_tar(subsource)
        if tar is None:
            incomplete.append(f"{subsource}:no_tar")
            continue
        steps = {
            "endpoint": run_step(output_dir, f"B_{subsource}_endpoint", "endpoint", (str(tar),), min(300, args.step_timeout_s)),
            "random_64k": run_step(
                output_dir,
                f"B_{subsource}_random64k",
                "random",
                (str(tar), 64 << 10, n, args.seed + file_i * 10 + 1),
                min(1_200, args.step_timeout_s),
            ),
            "random_1m": run_step(
                output_dir,
                f"B_{subsource}_random1m",
                "random",
                (str(tar), MIB, n, args.seed + file_i * 10 + 2),
                min(1_200, args.step_timeout_s),
            ),
            "sequential": run_step(
                output_dir,
                f"B_{subsource}_sequential",
                "sequential",
                (str(tar), 8 * MIB),
                args.step_timeout_s,
            ),
        }
        failed_steps = [name for name, payload in steps.items() if not payload.get("ok")]
        if failed_steps:
            incomplete.extend(f"{subsource}:{name}" for name in failed_steps)
            files[subsource] = {"tar_path": str(tar), "steps": steps}
            continue
        endpoint = steps["endpoint"]["result"]
        sequential = steps["sequential"]["result"]
        if endpoint["gzip_magic"]:
            hard_errors.append(f"{subsource}:gzip_compressed")
        for name in ("random_64k", "random_1m"):
            if steps[name]["result"]["short_read_pairs"]:
                hard_errors.append(f"{subsource}:{name}:short_read")
        if not sequential["complete"]:
            hard_errors.append(f"{subsource}:short_sequential_read")
        head_s = endpoint["head"]["cold_seconds"]
        tail_s = endpoint["tail"]["cold_seconds"]
        full_s = sequential["seconds"]
        tail_head_ratio = tail_s / max(head_s, 1e-9)
        tail_full_ratio = tail_s / max(full_s, 1e-9)
        range_flag = tail_head_ratio <= 5.0 and tail_full_ratio <= 0.25
        range_flags.append(range_flag)
        files[subsource] = {
            "tar_path": str(tar),
            "steps": {name: payload["result"] for name, payload in steps.items()},
            "range_ratios": {
                "tail_cold_over_head_cold": tail_head_ratio,
                "tail_cold_over_full_sequential": tail_full_ratio,
            },
            "range_semantics": range_flag,
        }
    range_semantics = len(range_flags) == 2 and all(range_flags)
    if hard_errors:
        verdict = "FAIL"
    elif incomplete:
        verdict = "INCOMPLETE"
    else:
        # range=false 是一次成功且决定性的探测(否决 range 后端 → fetch 三选一退化为二选一),
        # 不是故障;job 级 FAIL 只留给真错误。yaml 侧 `VERDICT != FAIL` 自然放行(review P1)。
        verdict = "PASS" if range_semantics else "DECISIVE_FALSE"
    return {
        "task": "B",
        "verdict": verdict,
        "scope": "cc12m_and_coyo_representative_tar",
        "sample_n_random_offsets_per_size_per_tar": n,
        "cold_definition": "first access to an offset in this process; kernel/FUSE/CDN caches were not privileged-dropped",
        "range_semantics": range_semantics if not incomplete else None,
        "thresholds": {
            "tail_cold_over_head_cold_max": 5.0,
            "tail_cold_over_full_sequential_max": 0.25,
            "short_reads": 0,
        },
        "files": files,
        "incomplete_steps": incomplete,
        "hard_errors": hard_errors,
        "key_numbers": {
            "range_semantics": range_semantics if not incomplete else "unknown",
            "files_complete": len(range_flags),
            "random_n": n,
        },
    }


def source_specs() -> list[tuple[str, str, Path]]:
    specs = [("monet_real", sub, MROOT / sub) for sub in MONET_SUBSOURCES]
    specs.extend(("synthetic", sub, MROOT / "synthetic" / sub) for sub in SYNTH_SUBSOURCES)
    return specs


def enumerate_tar_corpus(base: Path) -> dict[str, Any]:
    tar_count = 0
    total_bytes = 0
    group_dirs = 0
    stat_errors: list[str] = []
    if not base.is_dir():
        return {"base": str(base), "group_dirs": 0, "tar_count": 0, "total_bytes": 0, "errors": ["missing base"]}
    for group in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name):
        group_dirs += 1
        for tar in group.glob("*.tar"):
            try:
                total_bytes += tar.stat().st_size
                tar_count += 1
            except OSError as exc:
                stat_errors.append(f"{tar}: {type(exc).__name__}: {exc}"[:500])
    return {
        "base": str(base),
        "group_dirs": group_dirs,
        "tar_count": tar_count,
        "total_bytes": total_bytes,
        "errors": stat_errors[:100],
        "error_count": len(stat_errors),
    }


def safe_name(domain: str, subsource: str, sample_i: int) -> str:
    return f"{domain}__{subsource.replace('-', '_')}__{sample_i:02d}"


def task_c(args: argparse.Namespace) -> dict[str, Any]:
    samples_per_source = 1 if args.sample_n is None else min(8, args.sample_n)
    if samples_per_source <= 0:
        raise ValueError("--sample-n must be positive")
    output_dir = Path(args.output_dir)
    index_dir = output_dir / "taskC_indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    b_result_path = output_dir / "taskB_result.json"
    b_range = False
    b_touched_tars: set[str] = set()
    b_dependency = "taskB_result.json absent or not PASS/range=true"
    if b_result_path.is_file():
        try:
            with b_result_path.open("r", encoding="utf-8") as f:
                b_data = json.load(f)
            b_range = b_data.get("verdict") == "PASS" and b_data.get("range_semantics") is True
            b_dependency = f"verdict={b_data.get('verdict')} range_semantics={b_data.get('range_semantics')}"
            for file_result in (b_data.get("files") or {}).values():
                if isinstance(file_result, dict) and file_result.get("tar_path"):
                    b_touched_tars.add(str(file_result["tar_path"]))
        except Exception as exc:  # noqa: BLE001
            b_dependency = f"invalid taskB result: {type(exc).__name__}: {exc}"

    sources: dict[str, Any] = {}
    incomplete: list[str] = []
    hard_errors: list[str] = []
    total_serial_seconds = 0.0
    total_index_bytes_projection = 0.0
    longest_sample_seconds = 0.0
    seek_serial_seconds = 0.0
    seek_longest_seconds = 0.0

    for domain, subsource, base in source_specs():
        corpus = enumerate_tar_corpus(base)
        # B 已顺序全读 cc12m/coyo 代表 tar；C 排除同文件，避免把热 cache 成本外推到全 corpus。
        sample_tars = [
            p for p in first_n_tars(base, samples_per_source + len(b_touched_tars))
            if str(p) not in b_touched_tars
        ][:samples_per_source]
        sample_results: list[dict[str, Any]] = []
        sample_tar_bytes = 0
        sample_scan_seconds = 0.0
        sample_payload_scan_seconds = 0.0
        sample_index_work_seconds = 0.0
        sample_index_bytes = 0
        sample_seek_seconds = 0.0
        sample_seek_header_seconds = 0.0
        successful_samples = 0
        for sample_i, tar in enumerate(sample_tars):
            stem = safe_name(domain, subsource, sample_i)
            index_path = index_dir / f"{stem}.parquet"
            # B 证明 range 后先测 seek-jump first-touch，再顺序全读；反序会把 seek 指标系统性加热。
            jump: dict[str, Any] | None = None
            entry: dict[str, Any] = {"tar_path": str(tar)}
            if b_range:
                jump = run_step(
                    output_dir,
                    f"C_{stem}_seekjump",
                    "tar_seek_jump",
                    (str(tar),),
                    args.step_timeout_s,
                )
                entry["seek_jump_step"] = jump
                if not jump.get("ok"):
                    incomplete.append(f"{stem}:seek_jump")
                else:
                    entry["seek_jump"] = jump["result"]
                    entry.pop("seek_jump_step", None)
            sequential = run_step(
                output_dir,
                f"C_{stem}_sequential",
                "tar_sequential",
                (str(tar), str(index_path), domain, subsource),
                args.step_timeout_s,
            )
            entry["sequential_step"] = sequential
            if not sequential.get("ok"):
                incomplete.append(f"{stem}:sequential")
                sample_results.append(entry)
                continue
            seq = sequential["result"]
            entry["sequential"] = seq
            entry.pop("sequential_step", None)
            if seq["image_members"] <= 0 or seq["duplicate_image_keys"]:
                hard_errors.append(f"{stem}:empty_or_duplicate_index")
            sample_tar_bytes += seq["tar_bytes"]
            sample_scan_seconds += seq["total_build_seconds"]
            sample_payload_scan_seconds += seq["scan_seconds"]
            sample_index_work_seconds += seq["index_materialize_and_write_seconds"]
            sample_index_bytes += seq["index_bytes"]
            successful_samples += 1
            longest_sample_seconds = max(
                longest_sample_seconds, seq["total_build_seconds"]
            )

            if b_range and jump is not None and jump.get("ok"):
                j = jump["result"]
                entry["seek_matches_sequential"] = (
                    j["image_members"] == seq["image_members"]
                    and j["locator_sha256"] == seq["locator_sha256"]
                )
                if not entry["seek_matches_sequential"]:
                    hard_errors.append(f"{stem}:seek_locator_mismatch")
                seek_build_seconds = j["scan_seconds"] + seq["index_materialize_and_write_seconds"]
                sample_seek_seconds += seek_build_seconds
                sample_seek_header_seconds += j["scan_seconds"]
                seek_longest_seconds = max(seek_longest_seconds, seek_build_seconds)
                entry["seek_jump"]["projected_build_sample_seconds"] = seek_build_seconds
            sample_results.append(entry)

        if len(sample_tars) < samples_per_source:
            incomplete.append(f"{domain}/{subsource}:only_{len(sample_tars)}_sample_tars")
        if corpus.get("errors"):
            incomplete.append(f"{domain}/{subsource}:corpus_enumeration_errors")

        projected_seconds = None
        projected_seconds_all_byte_scaled = None
        projected_index_bytes = None
        projected_seek_seconds = None
        if sample_tar_bytes > 0 and corpus["total_bytes"] > 0 and successful_samples > 0:
            byte_scale = corpus["total_bytes"] / sample_tar_bytes
            count_scale = corpus["tar_count"] / successful_samples
            projected_seconds_all_byte_scaled = sample_scan_seconds * byte_scale
            # payload按字节；header之外的索引materialize/write与索引体积按tar/item数量代理。
            projected_seconds = (
                sample_payload_scan_seconds * byte_scale + sample_index_work_seconds * count_scale
            )
            projected_index_bytes = sample_index_bytes * count_scale
            total_serial_seconds += projected_seconds
            total_index_bytes_projection += projected_index_bytes
            if b_range and sample_seek_seconds > 0:
                projected_seek_seconds = (
                    sample_seek_header_seconds * count_scale + sample_index_work_seconds * count_scale
                )
                seek_serial_seconds += projected_seek_seconds
        sources[f"{domain}/{subsource}"] = {
            "domain": domain,
            "sub_source": subsource,
            "corpus": corpus,
            "samples": sample_results,
            "excluded_task_b_tars": sorted(str(p) for p in b_touched_tars if str(p).startswith(str(base))),
            "projection": {
                "method": "sequential payload scan is byte-scaled; seek headers, index materialization/write, and index bytes are tar-count-scaled",
                "sequential_build_seconds": projected_seconds,
                "sequential_all_build_seconds_if_byte_scaled": projected_seconds_all_byte_scaled,
                "index_bytes": projected_index_bytes,
                "seek_jump_seconds": projected_seek_seconds,
                "sampled_tar_count": successful_samples,
            },
        }

    ideal64_seconds = max(total_serial_seconds / 64.0, longest_sample_seconds)
    seek_ideal64_seconds = (
        max(seek_serial_seconds / 64.0, seek_longest_seconds) if b_range and seek_serial_seconds else None
    )
    if b_range and seek_ideal64_seconds is not None and seek_ideal64_seconds < ideal64_seconds:
        selected_projection = seek_ideal64_seconds
        selected_method = "seek_jump"
    else:
        selected_projection = ideal64_seconds
        selected_method = "sequential_stream"
    if hard_errors:
        verdict = "FAIL"
    elif incomplete:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if selected_projection <= 24 * 3600 else "FAIL"
    return {
        "task": "C",
        "verdict": verdict,
        "scope": "six_monet_plus_three_synthetic_subsources",
        "samples_per_subsource": samples_per_source,
        "task_b_dependency": b_dependency,
        "seek_jump_enabled": b_range,
        "sources": sources,
        "projection": {
            "sequential_serial_worker_seconds": total_serial_seconds,
            "sequential_ideal_64core_wall_seconds": ideal64_seconds,
            "seek_jump_serial_worker_seconds": seek_serial_seconds if b_range else None,
            "seek_jump_ideal_64core_wall_seconds": seek_ideal64_seconds,
            "selected_ideal_64core_wall_seconds": selected_projection,
            "selected_method": selected_method,
            "projected_index_bytes": total_index_bytes_projection,
            "assumption": "64 shard scans scale ideally until bounded by the longest sampled shard; mount bandwidth contention is not modeled.",
        },
        "thresholds": {"selected_ideal_64core_wall_seconds_max": 86_400},
        "incomplete_steps": incomplete,
        "hard_errors": hard_errors,
        "key_numbers": {
            "sources": len(sources),
            "ideal64_hours": selected_projection / 3600.0,
            "index_gib": total_index_bytes_projection / (1 << 30),
            "seek_jump": b_range,
            "selected_method": selected_method,
        },
    }


def anime_url_sample(n: int, seed: int) -> list[dict[str, str]]:
    """File-stratified sample; read a minimal set of randomly selected row groups."""
    files = sorted(ANIME_RICH.glob("*.parquet"))
    if not files:
        return []
    pfs = [(path, pq.ParquetFile(path)) for path in files]
    totals = [pf.metadata.num_rows for _path, pf in pfs]
    total = sum(totals)
    raw = [n * rows / total for rows in totals]
    quotas = [int(math.floor(x)) for x in raw]
    for i in sorted(range(len(raw)), key=lambda j: raw[j] - quotas[j], reverse=True)[: n - sum(quotas)]:
        quotas[i] += 1
    out: dict[str, dict[str, str]] = {}
    for file_i, ((path, pf), quota) in enumerate(zip(pfs, quotas)):
        if quota <= 0:
            continue
        rng = np.random.default_rng(seed + file_i * 1_000_003)
        rg_ids = list(range(pf.metadata.num_row_groups))
        rng.shuffle(rg_ids)
        tables: list[pa.Table] = []
        available = 0
        for rg in rg_ids:
            table = pf.read_row_group(rg, columns=["key", "s3key"], use_threads=False)
            tables.append(table)
            available += table.num_rows
            if available >= quota:
                break
        table = pa.concat_tables(tables)
        take = rng.choice(table.num_rows, size=min(quota, table.num_rows), replace=False)
        for rec in table.take(pa.array(take, type=pa.int64())).to_pylist():
            key = str(rec["key"])
            out.setdefault(key, {"key": key, "url": str(rec["s3key"]), "manifest": str(path)})
    return list(out.values())[:n]


def retry_after_seconds(headers: Any) -> float | None:
    try:
        value = headers.get("Retry-After")
        if value is None:
            return None
        return max(0.0, min(10.0, float(value)))
    except (TypeError, ValueError, AttributeError):
        return None


def cdn_get(url: str, max_attempts: int, timeout_s: float, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    statuses: list[int] = []
    errors: list[str] = []
    bytes_read = 0
    attempts_done = 0
    rng = random.Random(seed)
    for attempt in range(max_attempts):
        attempts_done = attempt + 1
        retry_after = None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mot-offline-arb-phase1/1"})
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                status = int(getattr(response, "status", 200))
                statuses.append(status)
                data = response.read()
            bytes_read = len(data)
            if 200 <= status < 300:
                return {
                    "success": True,
                    "final_status": status,
                    "attempt_statuses": statuses,
                    "errors": errors,
                    "attempts": attempt + 1,
                    "bytes": bytes_read,
                    "latency_seconds": time.perf_counter() - started,
                }
            retryable = status in (408, 429) or status >= 500
            if not retryable:
                break
        except urllib.error.HTTPError as exc:
            statuses.append(int(exc.code))
            errors.append(f"HTTP{exc.code}")
            retryable = exc.code in (408, 429) or exc.code >= 500
            retry_after = retry_after_seconds(exc.headers)
            if not retryable:
                break
        except Exception as exc:  # noqa: BLE001 - network exceptions are expected probe data
            errors.append(f"{type(exc).__name__}:{exc}"[:300])
            retryable = True
            retry_after = None
        if attempt + 1 >= max_attempts:
            break
        delay = retry_after if retry_after is not None else 0.3 * (2 ** attempt)
        # Small deterministic jitter prevents synchronized retries without increasing concurrency.
        time.sleep(delay * (0.9 + 0.2 * rng.random()))
    return {
        "success": False,
        "final_status": statuses[-1] if statuses else None,
        "attempt_statuses": statuses,
        "errors": errors,
        "attempts": attempts_done,
        "bytes": bytes_read,
        "latency_seconds": time.perf_counter() - started,
    }


def task_e(args: argparse.Namespace) -> dict[str, Any]:
    n = args.sample_n if args.sample_n is not None else 50_000
    if n <= 0:
        raise ValueError("--sample-n must be positive")
    concurrency = max(1, min(32, args.http_concurrency))
    samples = anime_url_sample(n, args.seed)
    active = 0
    peak_active = 0
    active_lock = threading.Lock()

    def fetch(rec: dict[str, str]) -> tuple[dict[str, str], dict[str, Any]]:
        nonlocal active, peak_active
        with active_lock:
            active += 1
            peak_active = max(peak_active, active)
        try:
            return rec, cdn_get(
                rec["url"], args.http_attempts, args.http_timeout_s, stable_seed(args.seed, rec["key"])
            )
        finally:
            with active_lock:
                active -= 1

    completed: list[tuple[dict[str, str], dict[str, Any]]] = []
    started = time.perf_counter()
    iterator = iter(samples)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        in_flight: set[concurrent.futures.Future] = set()
        for _ in range(min(len(samples), concurrency * 4)):
            try:
                in_flight.add(pool.submit(fetch, next(iterator)))
            except StopIteration:
                break
        while in_flight:
            done, in_flight = concurrent.futures.wait(in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                completed.append(future.result())
                try:
                    in_flight.add(pool.submit(fetch, next(iterator)))
                except StopIteration:
                    pass
    wall = time.perf_counter() - started

    attempt_statuses: Counter[int] = Counter()
    final_statuses: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    all_latencies: list[float] = []
    success_latencies: list[float] = []
    failure_latencies: list[float] = []
    successes = 0
    total_bytes = 0
    failure_examples: list[dict[str, Any]] = []
    attempts = 0
    for rec, result in completed:
        all_latencies.append(float(result["latency_seconds"]))
        attempts += int(result["attempts"])
        attempt_statuses.update(int(x) for x in result["attempt_statuses"])
        final_statuses[str(result["final_status"])] += 1
        error_types.update(x.split(":", 1)[0] for x in result["errors"])
        if result["success"]:
            successes += 1
            total_bytes += int(result["bytes"])
            success_latencies.append(float(result["latency_seconds"]))
        elif len(failure_examples) < 50:
            failure_examples.append({"key": rec["key"], "url": rec["url"], **result})
        if not result["success"]:
            failure_latencies.append(float(result["latency_seconds"]))
    http_responses = sum(attempt_statuses.values())
    rate_denominator = max(attempts, 1)
    rate429 = attempt_statuses[429] / rate_denominator
    rate403 = attempt_statuses[403] / rate_denominator
    rate5xx = sum(v for k, v in attempt_statuses.items() if 500 <= k < 600) / rate_denominator
    throughput = successes / wall if wall else 0.0
    scaled = n < 50_000 or concurrency != 32
    incomplete = len(samples) < n or len(completed) < len(samples)
    gate_fail = rate429 >= 0.001 or throughput < 100.0
    if incomplete:
        verdict = "INCOMPLETE"
    elif gate_fail:
        verdict = "FAIL"
    elif scaled or rate403 >= 0.01 or rate5xx >= 0.01:
        verdict = "WARN"
    else:
        verdict = "PASS"
    return {
        "task": "E",
        "verdict": verdict,
        "scope": "anime_rich_manifest_http_get",
        "requested_keys": n,
        "sampled_keys": len(samples),
        "completed_keys": len(completed),
        "successful_keys": successes,
        "concurrency": concurrency,
        "peak_active": peak_active,
        "retry": {"max_attempts": args.http_attempts, "base_seconds": 0.3, "formula": "base*2**attempt with jitter; Retry-After capped at 10s"},
        "wall_seconds": wall,
        "throughput_img_s": throughput,
        "throughput_mib_s": (total_bytes / MIB) / wall if wall else None,
        "all_key_latency": latency_stats_ms(all_latencies),
        "success_latency": latency_stats_ms(success_latencies),
        "failure_latency": latency_stats_ms(failure_latencies),
        "attempts": attempts,
        "http_responses": http_responses,
        "attempt_status_counts": {str(k): v for k, v in sorted(attempt_statuses.items())},
        "final_status_counts": dict(sorted(final_statuses.items())),
        "error_type_counts": dict(sorted(error_types.items())),
        "rates_over_all_attempts": {"http_429": rate429, "http_403": rate403, "http_5xx": rate5xx},
        "rates_over_final_keys": {
            "http_429": final_statuses["429"] / max(len(completed), 1),
            "http_403": final_statuses["403"] / max(len(completed), 1),
            "http_5xx": sum(v for k, v in final_statuses.items() if k.isdigit() and 500 <= int(k) < 600)
            / max(len(completed), 1),
        },
        "failure_examples": failure_examples,
        "thresholds": {"http_429_rate_strict_max": 0.001, "concurrency": 32, "throughput_img_s_min": 100.0},
        "key_numbers": {
            "ok": successes,
            "img_s": throughput,
            "rate429": rate429,
            "rate403": rate403,
            "rate5xx": rate5xx,
        },
    }


def stable_seed(seed: int, value: str) -> int:
    raw = f"{seed}\0{value}".encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def gpic_manifest_sample(n: int, seed: int) -> list[dict[str, Any]]:
    """Globally file-stratified sample across all 256 one-row-group manifests."""
    files = sorted(GPIC_MANIFEST.glob("*.parquet"))
    if not files:
        return []
    pfs = [(p, pq.ParquetFile(p)) for p in files]
    totals = [pf.metadata.num_rows for _p, pf in pfs]
    full = sum(totals)
    raw = [n * rows / full for rows in totals]
    quotas = [int(math.floor(x)) for x in raw]
    remainder = n - sum(quotas)
    for i in sorted(range(len(raw)), key=lambda j: raw[j] - quotas[j], reverse=True)[:remainder]:
        quotas[i] += 1
    out: dict[str, dict[str, Any]] = {}
    columns = ["uid", "src_root", "src_shard", "src_key"]
    for file_i, ((path, pf), quota) in enumerate(zip(pfs, quotas)):
        if quota <= 0:
            continue
        table = pf.read_row_group(0, columns=columns, use_threads=False)
        rng = np.random.default_rng(seed + file_i * 1_000_003)
        indices = rng.choice(table.num_rows, size=min(quota, table.num_rows), replace=False)
        for rec in table.take(pa.array(indices, type=pa.int64())).to_pylist():
            rec["manifest"] = str(path)
            out.setdefault(str(rec["uid"]), rec)
    return list(out.values())[:n]


def resolve_gpic_paths(records: Sequence[dict[str, Any]]) -> tuple[list[tuple[dict[str, Any], str]], list[dict[str, Any]]]:
    """从权威 caption parquet 的 s3key join 扩展名，不 stat/触碰图片文件。"""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[(str(rec["src_root"]), str(rec["src_shard"]))].append(rec)
    resolved: list[tuple[dict[str, Any], str]] = []
    errors: list[dict[str, Any]] = []
    for (src_root, src_shard), rows in grouped.items():
        source = Path(src_root) / "parquets" / f"gpic_train_{src_shard}.parquet"
        wanted = {str(rec["src_key"]): rec for rec in rows}
        found: set[str] = set()
        try:
            pf = pq.ParquetFile(source)
            for batch in pf.iter_batches(batch_size=65_536, columns=["key", "s3key"], use_threads=False):
                keys = batch.column(0).to_pylist()
                s3keys = batch.column(1).to_pylist()
                for key, s3key in zip(keys, s3keys):
                    key_s = str(key)
                    if key_s not in wanted:
                        continue
                    ext = os.path.splitext(str(s3key))[1] if s3key else ".jpg"
                    path = gpic_path(src_root, src_shard, key_s, ext=ext)
                    if path is not None:
                        resolved.append((wanted[key_s], path))
                        found.add(key_s)
                if len(found) == len(wanted):
                    break
        except Exception as exc:  # noqa: BLE001 - path join failures belong in the probe result
            for rec in rows:
                errors.append({
                    "uid": rec["uid"],
                    "source": str(source),
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                })
            continue
        for key_s, rec in wanted.items():
            if key_s not in found:
                errors.append({"uid": rec["uid"], "source": str(source), "error": "src_key not found"})
    return resolved, errors


def task_f(args: argparse.Namespace) -> dict[str, Any]:
    n = args.sample_n if args.sample_n is not None else 1_000
    if n <= 0:
        raise ValueError("--sample-n must be positive")
    candidates = gpic_manifest_sample(max(n, math.ceil(n * 1.25)), args.seed)
    resolve_started = time.perf_counter()
    resolved_all, resolve_errors = resolve_gpic_paths(candidates)
    resolve_seconds = time.perf_counter() - resolve_started
    # 正式 n=1000 时先从每个 rich manifest 至少取一条，再全局随机回填，避免前缀偏差。
    by_manifest: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for item in resolved_all:
        by_manifest[str(item[0]["manifest"])].append(item)
    rng = random.Random(args.seed + 99)
    manifest_names = list(by_manifest)
    rng.shuffle(manifest_names)
    resolved: list[tuple[dict[str, Any], str]] = []
    used_uids: set[str] = set()
    for manifest in manifest_names:
        rng.shuffle(by_manifest[manifest])
        item = by_manifest[manifest][0]
        resolved.append(item)
        used_uids.add(str(item[0]["uid"]))
        if len(resolved) >= n:
            break
    remaining = [item for item in resolved_all if str(item[0]["uid"]) not in used_uids]
    rng.shuffle(remaining)
    resolved.extend(remaining[: max(0, n - len(resolved))])
    rng.shuffle(resolved)

    latencies: list[float] = []
    total_bytes = 0
    read_errors: list[dict[str, Any]] = []
    read_started = time.perf_counter()
    for rec, path in resolved:
        started = time.perf_counter()
        try:
            with open(path, "rb") as f:
                data = f.read()
            elapsed = time.perf_counter() - started
            if not data:
                raise OSError("empty file")
            latencies.append(elapsed)
            total_bytes += len(data)
        except OSError as exc:
            if len(read_errors) < 100:
                read_errors.append({"uid": rec["uid"], "path": path, "error": f"{type(exc).__name__}: {exc}"})
    read_wall = time.perf_counter() - read_started
    stats = latency_stats_ms(latencies)
    throughput = len(latencies) / read_wall if read_wall else 0.0
    p95_s = (stats["p95_ms"] / 1000.0) if stats["p95_ms"] is not None else math.inf
    gate_fail = p95_s >= 2.0 or throughput < args.gpic_required_img_s
    incomplete = len(latencies) < n
    scaled = n < 1_000
    if gate_fail:
        verdict = "FAIL"
    elif incomplete:
        verdict = "INCOMPLETE"
    elif scaled:
        verdict = "WARN"
    else:
        verdict = "PASS"
    return {
        "task": "F",
        "verdict": verdict,
        "scope": "gpic_random_independent_files_serial_cold_proxy",
        "requested_paths": n,
        "candidate_rows": len(candidates),
        "resolved_candidate_paths": len(resolved_all),
        "resolved_paths": len(resolved),
        "successful_reads": len(latencies),
        "cold_definition": "first image-file access in this process; extension came from caption parquet s3key without statting the image",
        "path_resolution_seconds": resolve_seconds,
        "resolve_error_count": len(resolve_errors),
        "resolve_error_examples": resolve_errors[:100],
        "read_error_count": len(resolved) - len(latencies),
        "read_error_examples": read_errors,
        "latency": stats,
        "read_wall_seconds": read_wall,
        "throughput_img_s": throughput,
        "throughput_mib_s": (total_bytes / MIB) / read_wall if read_wall else None,
        "total_bytes": total_bytes,
        "projected_64_stream_img_s_ideal": throughput * 64.0,
        "max_supported_global_img_s_at_stage256_mix_0_259_ideal": throughput * 64.0 / 0.259,
        "thresholds": {
            "p95_seconds_strict_max": 2.0,
            "single_stream_img_s_min_provisional": args.gpic_required_img_s,
            "throughput_basis": "25.9% stage256 mix applied to a provisional 100 global img/s target",
        },
        "key_numbers": {
            "ok": len(latencies),
            "p95_ms": stats["p95_ms"],
            "img_s": throughput,
            "mib_s": (total_bytes / MIB) / read_wall if read_wall else 0.0,
        },
    }


TASKS = {"B": task_b, "C": task_c, "E": task_e, "F": task_f}


def run_one(task: str, args: argparse.Namespace) -> str:
    started_wall = utc_now()
    started = time.perf_counter()
    try:
        result = TASKS[task](args)
    except Exception as exc:  # noqa: BLE001 - persist a machine-readable failure
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
        raise argparse.ArgumentTypeError(f"Task must be a comma-separated subset of B,C,E,F; invalid={invalid}")
    # B/C 使用 fork 隔离阻塞 FUSE；组合运行时固定在任何 HTTP 线程任务之前执行。
    requested = set(tasks)
    return [task for task in ("B", "C", "E", "F") if task in requested]


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
    parser.add_argument("--task", required=True, type=parse_tasks, help="B,C,E,F or a comma-separated subset")
    parser.add_argument("--output-dir", required=True, help="Run output directory; the only write destination")
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="B offsets/size, C tar samples/subsource (capped 8), E keys, or F paths",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--step-timeout-s", type=float, default=1_800.0, help="Killable B/C sub-step timeout")
    parser.add_argument("--http-concurrency", type=int, default=32, help="Task E; hard-capped at 32")
    parser.add_argument("--http-attempts", type=int, default=3)
    parser.add_argument("--http-timeout-s", type=float, default=20.0)
    parser.add_argument("--gpic-required-img-s", type=float, default=25.9, help="Task F provisional single-stream gate")
    args = parser.parse_args()
    if args.step_timeout_s <= 0 or args.http_attempts <= 0 or args.http_timeout_s <= 0:
        parser.error("timeouts and attempts must be positive")
    if args.gpic_required_img_s < 0:
        parser.error("--gpic-required-img-s must be non-negative")
    try:
        args.output_dir = validate_output_dir(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    verdicts = [run_one(task, args) for task in args.task]
    return 2 if "FAIL" in verdicts else 0


if __name__ == "__main__":
    sys.exit(main())
