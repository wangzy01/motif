#!/usr/bin/env python3
"""OFFLINE_ARB_V1 Phase-2 IO shootout：S1..S5，见 docs/OFFLINE_ARB_V1_SPEC.md 第3节 Phase-2 gate。

全部数据源严格只读；程序只写 ``--output-dir``。机制沿用 Phase-1
(scripts/offline_arb/phase1_io_probe.py / phase1_meta_validate.py，本文件不 import 它们、
按仓库既有约定各 phase 脚本自包含，仅复用其*机制*)：
  - seek_jump tar 索引构建（tarfile 'r:' 随机模式，header-to-header seek，不读 payload）。
  - range/ranged read（open+seek+read）与并发扫描曲线。
  - CDN 并发 GET（沿用 image_fetch.fetch_url_bytes 的重试/退避语义，压测层自带指数退避）。
  - atomic_json / append_summary / run_step(fork 子进程可 kill) / validate_output_dir 约定。
真正的生产 IO 函数（decode_rgb/stream_tar_images/read_gpic_bytes/fetch_url_bytes）直接从
training.dataline.image_fetch import，不重写。

参考需求基线（Evan 拍板，写死为常量，不在运行时重新推导）：
  stage256 计算步长 ~5s，G=2304 → 全局 ~461 img/s；单节点(8 rank) ~231 img/s。
  域拆分(全局 img/s)：anime 124 / monet_real 187 / gpic 119 / synthetic 24 / textatlas 7。
  gate：单节点 data-only 混合供给 ≥ 1.5×231 ≈ 347 img/s（S5 本体）。
  子域 1.5x 安全裕量同样用于 S2/S3/S4 的单项判定，口径见各 task 函数 docstring。

五个子测试（--test s1..s5，默认全跑，按 s1→s5 顺序执行——S5 依赖 S1/S2/S3 的产物/结论）：
  S1 monet/synth tar 后端 shootout（range vs shard_window，DECISIVE_* 结论，非 PASS/FAIL 否决）。
  S2 gpic 并发扩展性（头号问题；64 并发 <60 img/s → FAIL，意味着必须 /local 预取）。
  S3 textatlas 并发 row-group 读（完备性检查，需求仅 ~7 img/s，预期轻松过）。
  S4 anime 64 并发上探（复用 Phase-1 Task E 机制，仅测 64 并发档）。
  S5 单节点 8-rank 混合负载终测（gate 本体，10 分钟）。

硬约束：只读数据源；输出仅 --output-dir；CDN 并发 ≤64 带退避；无周期 faulthandler；
每步 timeout（run_step fork 隔离）；FAIL 只留给真错误——S1 的后端否决类结论用 DECISIVE_* token
（沿用 Phase-1 Task B 对 range_semantics=false 记 DECISIVE_FALSE 而非 FAIL 的先例）。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import queue
import random
import socket
import statistics
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

from training.dataline.image_fetch import (
    decode_rgb,
    fetch_url_bytes,
    read_gpic_bytes,
    stream_tar_images,
)
from training.dataline.native_safety import PARQUET_NATIVE_LOCK, configure_pyarrow_single_thread

configure_pyarrow_single_thread()

SEED = 20260709
MIB = 1 << 20
MROOT = Path("/s3-mewtant-raw-datasets/jasperai/monet/v1.1.0")
ANIME_RICH = Path("/s3/evan/anime_core_ditv04_ct_soft/data")
GPIC_MANIFEST = Path("/s3-pixai-test-uw2/evan/mot_filtering_v1/t5_manifest/rich_v1/gpic")
TEXTATLAS_SOURCE = Path("/s3/evan/textatlas5m")
OUTPUT_PARENT = Path("/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop")

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

# S1 表 shootout 覆盖的 tar 组：(domain, sub_source, n_tars)。monet cc12m+coyo 各 2 个 tar，
# synth 任取一个子源(flux-klein)取 2 个 tar——6 个物理 shard 给 shard-level 并发(1/4/8)留出空间。
S1_GROUPS: tuple[tuple[str, str, int], ...] = (
    ("monet_real", "cc12m", 2),
    ("monet_real", "coyo", 2),
    ("synthetic", SYNTH_SUBSOURCES[0], 2),
)

# ---- 参考需求基线（Evan 拍板，2026-07-09；docs/OFFLINE_ARB_V1_SPEC.md 第3节）----
STAGE256_STEP_SECONDS = 5.0
STAGE256_GLOBAL_BATCH = 2304
GLOBAL_IMG_S = STAGE256_GLOBAL_BATCH / STAGE256_STEP_SECONDS  # ≈460.8，四舍五入引用 461
SINGLE_NODE_IMG_S = 231.0  # 单节点(8 rank)，给定值(≈461/2，2x8 world)
GATE_MULTIPLIER = 1.5
NODE_GATE_IMG_S = GATE_MULTIPLIER * SINGLE_NODE_IMG_S  # ≈346.5，Evan 口径写 347

DOMAIN_GLOBAL_IMG_S = {
    "anime": 124.0,
    "monet_real": 187.0,
    "gpic": 119.0,
    "synthetic": 24.0,
    "textatlas": 7.0,
}
DOMAIN_MIX_S5 = {
    "anime": 0.27,
    "monet_real": 0.405,
    "gpic": 0.259,
    "synthetic": 0.051,
    "textatlas": 0.015,
}
assert abs(sum(DOMAIN_MIX_S5.values()) - 1.0) < 1e-9

GPIC_GATE_PASS_IMG_S = GATE_MULTIPLIER * DOMAIN_GLOBAL_IMG_S["gpic"]  # 178.5 ≈179
GPIC_GATE_FAIL_IMG_S = 60.0  # 显式给定(非本脚本推导)：64 并发 <60 → FAIL，意味着必须 /local 预取
ANIME_GATE_PASS_IMG_S = GATE_MULTIPLIER * DOMAIN_GLOBAL_IMG_S["anime"]  # 186.0
TEXTATLAS_GATE_PASS_IMG_S = GATE_MULTIPLIER * DOMAIN_GLOBAL_IMG_S["textatlas"]  # 10.5，本脚本按同一
    # 1.5x 惯例推导(委托书未给 textatlas 显式数值)；textatlas 是完备性检查非否决项，见 task_s3。

BATCH_SIZE_PER_RANK = STAGE256_GLOBAL_BATCH // 8  # 288；S5 的"虚拟 batch"单位
assert BATCH_SIZE_PER_RANK == 288

S1_RANGE_CONCURRENCIES = (1, 4, 16, 64)
S1_SHARD_CONCURRENCIES = (1, 4, 8)
S2_CONCURRENCIES = (1, 8, 32, 64, 128)
S3_CONCURRENCIES = (1, 8, 32)
S4_CONCURRENCY = 64


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def latency_stats_ms(values: Sequence[float]) -> dict[str, Any]:
    """输入单位秒；输出统一毫秒（S5 的 batch_ready 也复用本函数，字段名自带 _ms 消歧）。"""
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
        tmp.unlink(missing_ok=True)


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


# --------------------------------------------------------------------------- #
# run_step：killable fork 子进程（Phase-1 phase1_io_probe.run_step 机制，泛化成直接传 callable，
# 不再用字符串 registry——Phase-2 的 step 种类比 Phase-1 多得多，fork 天然不需要 pickle）。
# --------------------------------------------------------------------------- #
def _step_worker(fn: Callable[..., Any], call_args: tuple[Any, ...], result_path_s: str) -> None:
    try:
        result = fn(*call_args)
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
    fn: Callable[..., Any],
    call_args: tuple[Any, ...],
    timeout_s: float,
) -> dict[str, Any]:
    """在可 kill 的 fork 子进程里跑一个可能阻塞 FUSE/网络的函数。"""
    step_dir = output_dir / ".phase2_steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    result_path = step_dir / f"{label}.{os.getpid()}.json"
    try:
        result_path.unlink()
    except FileNotFoundError:
        pass
    ctx = mp.get_context("fork")
    proc = ctx.Process(target=_step_worker, args=(fn, call_args, str(result_path)))
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


def run_parallel_processes(
    output_dir: Path,
    label: str,
    target: Callable[..., None],
    arg_tuples: Sequence[tuple[Any, ...]],
    timeout_s: float,
) -> list[dict[str, Any]]:
    """N-way 并行版 run_step：为 S5 的 8 个 worker 进程复用同一 fork+timeout+升级 kill 纪律，
    只是并行 join 而非单个阻塞 join。target 自己负责把 {"ok":...} payload 原子写到 result_path
    （最后一个位置参数），本函数只管调度/收尸/超时。"""
    step_dir = output_dir / ".phase2_steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    procs = []
    result_paths = []
    for i, a in enumerate(arg_tuples):
        rp = step_dir / f"{label}.{i}.{os.getpid()}.json"
        try:
            rp.unlink()
        except FileNotFoundError:
            pass
        result_paths.append(rp)
        p = ctx.Process(target=target, args=(*a, str(rp)))
        p.start()
        procs.append(p)
    deadline = time.perf_counter() + timeout_s
    for p in procs:
        p.join(max(0.0, deadline - time.perf_counter()))
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(10)
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join(10)
    results: list[dict[str, Any]] = []
    for i, rp in enumerate(result_paths):
        if rp.is_file():
            with rp.open("r", encoding="utf-8") as f:
                results.append(json.load(f))
            try:
                rp.unlink()
            except OSError:
                pass
        else:
            results.append({"ok": False, "timeout": True, "rank": i})
    return results


def cap_timeout(args: argparse.Namespace, default_cap: float) -> float:
    return min(default_cap, args.step_timeout_s)


def stable_seed(seed: int, value: str) -> int:
    raw = f"{seed}\0{value}".encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


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


def group_base(domain: str, sub_source: str) -> Path:
    return (MROOT / sub_source) if domain == "monet_real" else (MROOT / "synthetic" / sub_source)


# --------------------------------------------------------------------------- #
# S1 — monet/synth tar 后端 shootout：range read vs shard_window（stream_tar_images）。
# --------------------------------------------------------------------------- #
def scan_tar_index(tar_path: str, index_path: str, domain: str, subsource: str) -> dict[str, Any]:
    """(a) seek_jump 索引构建：tarfile 'r:' 随机模式 header-to-header seek（Phase-1 Task C 机制），
    物化完整 key->(data_offset,size) 到 zstd parquet，供 (b) range read / (c) shard_window 使用。"""
    keys: list[str] = []
    offsets: list[int] = []
    sizes: list[int] = []
    started = time.perf_counter()
    with tarfile.open(tar_path, "r:") as tf:
        for member in tf:
            if member.isfile() and member.name.endswith(".jpg"):
                key = member.name[:-4].split("/")[-1]
                keys.append(key)
                offsets.append(int(member.offset_data))
                sizes.append(int(member.size))
    scan_seconds = time.perf_counter() - started
    table = pa.table({
        "key": pa.array(keys, type=pa.string()),
        "tar_path": pa.array([tar_path] * len(keys), type=pa.string()),
        "data_offset": pa.array(offsets, type=pa.int64()),
        "size": pa.array(sizes, type=pa.int64()),
        "domain": pa.array([domain] * len(keys), type=pa.string()),
        "sub_source": pa.array([subsource] * len(keys), type=pa.string()),
    })
    index_path_p = Path(index_path)
    index_path_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path_p.with_name(f"{index_path_p.name}.tmp.{os.getpid()}")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, index_path_p)
    return {
        "tar_path": tar_path,
        "domain": domain,
        "sub_source": subsource,
        "image_members": len(keys),
        "scan_seconds": scan_seconds,
        "index_path": str(index_path_p),
        "index_bytes": index_path_p.stat().st_size,
        "tar_bytes": os.stat(tar_path).st_size,
    }


def range_read_item(tar_path: str, offset: int, size: int) -> tuple[bool, float, int]:
    started = time.perf_counter()
    try:
        with open(tar_path, "rb", buffering=0) as f:
            f.seek(offset)
            data = f.read(size)
        return len(data) == size, time.perf_counter() - started, len(data)
    except OSError:
        return False, time.perf_counter() - started, 0


def range_concurrency_level(items: list[tuple[str, int, int]], concurrency: int) -> dict[str, Any]:
    """(b) backend=range：按索引随机抽 item 做 ranged read，并发 sweep 的单一档位。"""
    latencies: list[float] = []
    ok = 0
    nbytes = 0
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(range_read_item, t, o, s) for (t, o, s) in items]
        for fut in concurrent.futures.as_completed(futures):
            good, dt, n = fut.result()
            latencies.append(dt)
            if good:
                ok += 1
                nbytes += n
    wall = time.perf_counter() - started
    return {
        "backend": "range",
        "concurrency": concurrency,
        "n_items": len(items),
        "ok": ok,
        "wall_seconds": wall,
        "img_s": ok / wall if wall else 0.0,
        "mib_s": (nbytes / MIB) / wall if wall else 0.0,
        "latency_ms": latency_stats_ms(latencies),
    }


def shard_window_scan_one(tar_path: str, wanted_keys: set[str]) -> tuple[int, int, float]:
    started = time.perf_counter()
    n = 0
    nbytes = 0
    for _key, raw in stream_tar_images(tar_path, wanted_keys):
        n += 1
        nbytes += len(raw)
    return n, nbytes, time.perf_counter() - started


def shard_window_concurrency_level(
    shard_tasks: list[tuple[str, set[str]]], concurrency: int
) -> dict[str, Any]:
    """(c) backend=shard_window：同一 item 集按 shard 分组，每 shard 用 stream_tar_images 顺序扫收齐
    （含扫过不需要字节的代价），并发=shard 级。"""
    started = time.perf_counter()
    results: list[tuple[int, int, float]] = []
    actual_conc = max(1, min(concurrency, len(shard_tasks)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_conc) as pool:
        futures = [pool.submit(shard_window_scan_one, t, ks) for (t, ks) in shard_tasks]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    wall = time.perf_counter() - started
    total_items = sum(r[0] for r in results)
    total_bytes = sum(r[1] for r in results)
    return {
        "backend": "shard_window",
        "requested_concurrency": concurrency,
        "actual_concurrency": actual_conc,
        "n_shards": len(shard_tasks),
        "items_collected": total_items,
        "wall_seconds": wall,
        "img_s": total_items / wall if wall else 0.0,
        "mib_s": (total_bytes / MIB) / wall if wall else 0.0,
        "per_shard": [{"items": r[0], "bytes": r[1], "seconds": r[2]} for r in results],
    }


def sequential_tar_scan(tar_path: str, chunk_size: int = 8 * MIB) -> dict[str, Any]:
    """(d) 顺序全扫吞吐基线：1 并发，8MiB chunk 顺序读完整 tar，MB/s 基线（Phase-1 Task B 机制）。"""
    expected = os.stat(tar_path).st_size
    read_bytes = 0
    started = time.perf_counter()
    with open(tar_path, "rb", buffering=0) as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            read_bytes += len(block)
    elapsed = time.perf_counter() - started
    return {
        "tar_path": tar_path,
        "expected_bytes": expected,
        "read_bytes": read_bytes,
        "seconds": elapsed,
        "throughput_mib_s": (read_bytes / MIB) / elapsed if elapsed else None,
        "complete": read_bytes == expected,
    }


def task_s1(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    index_dir = output_dir / "s1_indexes"
    incomplete: list[str] = []
    hard_errors: list[str] = []
    built: list[dict[str, Any]] = []

    for domain, sub, n_tars in S1_GROUPS:
        tars = first_n_tars(group_base(domain, sub), n_tars)
        if len(tars) < n_tars:
            incomplete.append(f"{domain}/{sub}:only_{len(tars)}_tars")
        for i, tar in enumerate(tars):
            index_path = index_dir / f"{domain}__{sub.replace('-', '_')}__{i:02d}.parquet"
            step = run_step(
                output_dir, f"S1_index_{domain}_{sub}_{i}", scan_tar_index,
                (str(tar), str(index_path), domain, sub), cap_timeout(args, 900.0),
            )
            if not step.get("ok"):
                incomplete.append(f"{domain}/{sub}/{i}:index_build_{step.get('error', 'timeout')}")
                continue
            built.append(step["result"])

    if not built:
        hard_errors.append("no_tar_index_built")
        return {
            "task": "S1", "verdict": "FAIL", "scope": "monet_synth_tar_backend_shootout",
            "incomplete_steps": incomplete, "hard_errors": hard_errors,
            "key_numbers": {"exception": "no_index"},
        }

    # 装 item pool：(tar_path,key,offset,size)，稳定 shuffle 后按并发档切成不相交的 chunk
    # （range 用),shard_window 用同一并集(全部 chunk 合并)按 tar 分组、扫过全部命中 key。
    pool: list[tuple[str, str, int, int]] = []
    for b in built:
        t = pq.read_table(b["index_path"], columns=["tar_path", "key", "data_offset", "size"])
        for tp, key, off, sz in zip(
            t.column("tar_path").to_pylist(), t.column("key").to_pylist(),
            t.column("data_offset").to_pylist(), t.column("size").to_pylist(),
        ):
            pool.append((tp, str(key), int(off), int(sz)))
    rng = random.Random(args.seed + 1001)
    rng.shuffle(pool)

    n_per_level = args.sample_n if args.sample_n else 300
    levels = list(S1_RANGE_CONCURRENCIES)
    chunks: dict[int, list[tuple[str, int, int]]] = {}
    cursor = 0
    for level in levels:
        chunk = pool[cursor:cursor + n_per_level]
        cursor += n_per_level
        if len(chunk) < n_per_level:
            incomplete.append(f"range_level_{level}:short_sample_{len(chunk)}_of_{n_per_level}")
        chunks[level] = [(tp, off, sz) for tp, _key, off, sz in chunk]

    range_results: dict[str, Any] = {}
    for level in levels:
        step = run_step(
            output_dir, f"S1_range_{level}", range_concurrency_level,
            (chunks[level], level), cap_timeout(args, 600.0),
        )
        if not step.get("ok"):
            incomplete.append(f"range_level_{level}:{step.get('error', 'timeout')}")
            continue
        range_results[str(level)] = step["result"]

    # shard_window：同一 item 集 = 上面各并发档抽样的并集，按 tar_path 分组成 wanted_keys
    # （chunks 里丢了 key 只留 offset/size，这里从 pool 原始行按 (tar_path,offset) 反查 key）。
    combined_items = [it for c in chunks.values() for it in c]
    offset_to_key: dict[tuple[str, int], str] = {(tp, off): key for tp, key, off, _sz in pool}
    combined_keys_by_tar: dict[str, set[str]] = defaultdict(set)
    for tp, off, _sz in combined_items:
        key = offset_to_key.get((tp, off))
        if key is not None:
            combined_keys_by_tar[tp].add(key)
    shard_tasks = [(tp, ks) for tp, ks in combined_keys_by_tar.items() if ks]

    shard_results: dict[str, Any] = {}
    for level in S1_SHARD_CONCURRENCIES:
        step = run_step(
            output_dir, f"S1_shard_{level}", shard_window_concurrency_level,
            (shard_tasks, level), cap_timeout(args, 1200.0),
        )
        if not step.get("ok"):
            incomplete.append(f"shard_level_{level}:{step.get('error', 'timeout')}")
            continue
        shard_results[str(level)] = step["result"]

    # 顺序全扫基线：每组第一个 tar 各测一次
    sequential_baselines: dict[str, Any] = {}
    for b in built:
        key = f"{b['domain']}/{b['sub_source']}"
        if key in sequential_baselines:
            continue
        step = run_step(
            output_dir, f"S1_seq_{b['domain']}_{b['sub_source']}", sequential_tar_scan,
            (b["tar_path"],), cap_timeout(args, 1200.0),
        )
        if not step.get("ok"):
            incomplete.append(f"sequential:{key}:{step.get('error', 'timeout')}")
            continue
        seq = step["result"]
        img_s_equiv = (
            (b["image_members"] / b["tar_bytes"]) * seq["throughput_mib_s"] * MIB
            if seq.get("throughput_mib_s") else None
        )
        sequential_baselines[key] = {**seq, "image_members_in_tar": b["image_members"],
                                       "tar_bytes": b["tar_bytes"], "img_s_equivalent": img_s_equiv}

    range_best = max((r["img_s"] for r in range_results.values()), default=0.0)
    shard_best = max((r["img_s"] for r in shard_results.values()), default=0.0)
    range_error_rate = max(
        ((r["n_items"] - r["ok"]) / r["n_items"] for r in range_results.values() if r["n_items"]),
        default=0.0,
    )
    if range_error_rate > 0.05:
        hard_errors.append("range_backend_error_rate_over_5pct")

    saturation_suspected = None
    latency_degradation_suspected = None
    if "1" in range_results and "64" in range_results and range_results["1"]["img_s"] > 0:
        efficiency_64 = range_results["64"]["img_s"] / (64.0 * range_results["1"]["img_s"])
        saturation_suspected = efficiency_64 < 0.5
        p99_1 = range_results["1"]["latency_ms"]["p99_ms"]
        p99_64 = range_results["64"]["latency_ms"]["p99_ms"]
        if p99_1 and p99_64:
            latency_degradation_suspected = (p99_64 / p99_1) > 4.0

    if hard_errors:
        verdict = "FAIL"
        decision_reason = "hard_errors_present"
    elif range_best <= 0.0 and shard_best <= 0.0:
        verdict = "INCOMPLETE"
        decision_reason = "no_backend_produced_a_measurement"
    elif range_best > shard_best * 1.2:
        verdict = "DECISIVE_RANGE"
        decision_reason = f"range_best={range_best:.1f} > 1.2x shard_best={shard_best:.1f}"
    elif shard_best > range_best * 1.2:
        verdict = "DECISIVE_SHARD_WINDOW"
        decision_reason = f"shard_best={shard_best:.1f} > 1.2x range_best={range_best:.1f}"
    else:
        verdict = "DECISIVE_INCONCLUSIVE"
        decision_reason = f"range_best={range_best:.1f} vs shard_best={shard_best:.1f} within 20% margin"

    return {
        "task": "S1",
        "verdict": verdict,
        "scope": "monet_synth_tar_backend_shootout_range_vs_shard_window",
        "groups": [{"domain": d, "sub_source": s, "n_tars": n} for d, s, n in S1_GROUPS],
        "index_builds": built,
        "item_pool_size": len(pool),
        "range": range_results,
        "shard_window": shard_results,
        "sequential_baseline": sequential_baselines,
        "decision": {
            "recommendation": verdict,
            "reason": decision_reason,
            "range_best_img_s": range_best,
            "shard_window_best_img_s": shard_best,
            "saturation_suspected_at_64_range_concurrency": saturation_suspected,
            "latency_degradation_suspected_at_64_range_concurrency": latency_degradation_suspected,
        },
        "incomplete_steps": incomplete,
        "hard_errors": hard_errors,
        "key_numbers": {
            "range_best_img_s": round(range_best, 1),
            "shard_best_img_s": round(shard_best, 1),
            "decision": verdict,
        },
    }


# --------------------------------------------------------------------------- #
# S2 — gpic 并发扩展性（头号问题）：随机文件 open+read 并发 sweep + batch 预取模拟。
# --------------------------------------------------------------------------- #
def gpic_manifest_sample(n: int, seed: int) -> list[dict[str, Any]]:
    """全局按行数分层抽样 256 个 rich_v1/gpic manifest（机制同 Phase-1 Task F）。"""
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


def resolve_gpic_ext(
    records: Sequence[dict[str, Any]]
) -> tuple[list[tuple[dict[str, Any], str]], list[dict[str, Any]]]:
    """从权威 caption parquet 的 s3key join 扩展名，不 stat/触碰图片文件（机制同 Phase-1 Task F）。
    只解析 ext，真正取字节交给生产函数 image_fetch.read_gpic_bytes(src_root,src_shard,src_key,ext)。"""
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
                    resolved.append((wanted[key_s], ext))
                    found.add(key_s)
                if len(found) == len(wanted):
                    break
        except Exception as exc:  # noqa: BLE001 - path join failures belong in the probe result
            for rec in rows:
                errors.append({"uid": rec["uid"], "source": str(source), "error": f"{type(exc).__name__}: {exc}"[:500]})
            continue
        for key_s, rec in wanted.items():
            if key_s not in found:
                errors.append({"uid": rec["uid"], "source": str(source), "error": "src_key not found"})
    return resolved, errors


def gpic_open_read(rec: dict[str, Any], ext: str) -> tuple[bool, float, int]:
    started = time.perf_counter()
    data = read_gpic_bytes(rec["src_root"], rec["src_shard"], rec["src_key"], ext=ext)
    ok = bool(data)
    return ok, time.perf_counter() - started, (len(data) if data else 0)


def gpic_concurrency_level(items: list[tuple[dict[str, Any], str]], concurrency: int) -> dict[str, Any]:
    latencies: list[float] = []
    ok = 0
    nbytes = 0
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(gpic_open_read, rec, ext) for rec, ext in items]
        for fut in concurrent.futures.as_completed(futures):
            good, dt, n = fut.result()
            latencies.append(dt)
            if good:
                ok += 1
                nbytes += n
    wall = time.perf_counter() - started
    return {
        "concurrency": concurrency,
        "n": len(items),
        "ok": ok,
        "wall_seconds": wall,
        "img_s": ok / wall if wall else 0.0,
        "mib_s": (nbytes / MIB) / wall if wall else 0.0,
        "latency_ms": latency_stats_ms(latencies),
    }


def task_s2(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    n_per_level = args.sample_n if args.sample_n else 500
    levels = list(S2_CONCURRENCIES)
    batch_n = 512
    batch_concurrency = 128
    incomplete: list[str] = []
    hard_errors: list[str] = []

    total_needed = n_per_level * len(levels) + batch_n
    candidates = gpic_manifest_sample(max(total_needed * 2, total_needed + 200), args.seed)
    resolved_all, resolve_errors = resolve_gpic_ext(candidates)
    if len(resolved_all) < total_needed:
        incomplete.append(f"resolved_paths_short:{len(resolved_all)}_of_{total_needed}")
    rng = random.Random(args.seed + 2002)
    rng.shuffle(resolved_all)
    items = resolved_all

    levels_result: dict[str, Any] = {}
    cursor = 0
    for level in levels:
        chunk = items[cursor:cursor + n_per_level]
        cursor += n_per_level
        if len(chunk) < n_per_level:
            incomplete.append(f"level_{level}:short_sample_{len(chunk)}_of_{n_per_level}")
        if not chunk:
            continue
        step = run_step(
            output_dir, f"S2_level_{level}", gpic_concurrency_level,
            (chunk, level), cap_timeout(args, 900.0),
        )
        if not step.get("ok"):
            incomplete.append(f"level_{level}:{step.get('error', 'timeout')}")
            continue
        levels_result[str(level)] = step["result"]

    batch_chunk = items[cursor:cursor + batch_n]
    cursor += batch_n
    batch_result = None
    if len(batch_chunk) < batch_n:
        incomplete.append(f"batch_prefetch:short_sample_{len(batch_chunk)}_of_{batch_n}")
    else:
        step = run_step(
            output_dir, "S2_batch_prefetch", gpic_concurrency_level,
            (batch_chunk, batch_concurrency), cap_timeout(args, 600.0),
        )
        if step.get("ok"):
            batch_result = step["result"]
        else:
            incomplete.append(f"batch_prefetch:{step.get('error', 'timeout')}")

    if not levels_result:
        hard_errors.append("no_concurrency_level_completed")
    for lvl, r in levels_result.items():
        if r["n"] and (r["n"] - r["ok"]) / r["n"] > 0.05:
            hard_errors.append(f"level_{lvl}:error_rate_over_5pct")

    best_img_s = max((r["img_s"] for r in levels_result.values()), default=0.0)
    best_level = max(levels_result, key=lambda k: levels_result[k]["img_s"], default=None)
    img_s_64 = levels_result.get("64", {}).get("img_s")

    if hard_errors:
        verdict = "FAIL"
    elif img_s_64 is not None and img_s_64 < GPIC_GATE_FAIL_IMG_S:
        verdict = "FAIL"
    elif not levels_result:
        verdict = "INCOMPLETE"
    elif best_img_s >= GPIC_GATE_PASS_IMG_S:
        verdict = "PASS"
    else:
        verdict = "WARN"

    return {
        "task": "S2",
        "verdict": verdict,
        "scope": "gpic_concurrency_scaling_random_independent_files",
        "concurrency_levels": levels_result,
        "batch_prefetch": {"n": batch_n, "concurrency": batch_concurrency, "result": batch_result},
        "best_concurrency": best_level,
        "best_img_s": best_img_s,
        "resolve_error_count": len(resolve_errors),
        "thresholds": {
            "pass_img_s_min_any_level": GPIC_GATE_PASS_IMG_S,
            "fail_img_s_max_at_64": GPIC_GATE_FAIL_IMG_S,
            "note": "64并发<60 img/s -> FAIL(须 /local 预取方案)，给定值非本脚本推导",
        },
        "incomplete_steps": incomplete,
        "hard_errors": hard_errors,
        "key_numbers": {
            "best_concurrency": best_level,
            "best_img_s": round(best_img_s, 1),
            "img_s_64": round(img_s_64, 1) if img_s_64 is not None else None,
        },
    }


# --------------------------------------------------------------------------- #
# S3 — textatlas 并发 row-group 读（完备性检查；需求仅 ~7 img/s，预期轻松 PASS）。
# --------------------------------------------------------------------------- #
def textatlas_source_files(subsets: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for sub in subsets:
        out.extend(sorted((TEXTATLAS_SOURCE / sub).glob("*.parquet")))
    return out


def sample_row_groups(files: list[Path], n: int, seed: int) -> list[tuple[str, int]]:
    metas: list[tuple[Path, int]] = []
    for p in files:
        try:
            pf = pq.ParquetFile(p)
        except Exception:  # noqa: BLE001 - unreadable footer belongs in the probe result, not a crash
            continue
        metas.append((p, pf.metadata.num_row_groups))
    total_rg = sum(m[1] for m in metas)
    if total_rg == 0:
        return []
    rng = random.Random(seed)
    raw_quota = [n * m[1] / total_rg for m in metas]
    quotas = [min(m[1], int(math.floor(q))) for m, q in zip(metas, raw_quota)]
    remain = n - sum(quotas)
    order = sorted(range(len(metas)), key=lambda i: raw_quota[i] - math.floor(raw_quota[i]), reverse=True)
    for i in order:
        if remain <= 0:
            break
        if quotas[i] < metas[i][1]:
            quotas[i] += 1
            remain -= 1
    picks: list[tuple[str, int]] = []
    for (p, n_rg), q in zip(metas, quotas):
        if q <= 0:
            continue
        rgs = rng.sample(range(n_rg), min(q, n_rg))
        picks.extend((str(p), rg) for rg in rgs)
    rng.shuffle(picks)
    return picks[:n]


def textatlas_concurrency_level(picks: list[tuple[str, int]], concurrency: int) -> dict[str, Any]:
    pf_cache: dict[str, pq.ParquetFile] = {}
    cache_lock = threading.Lock()

    def task(path: str, rg: int) -> tuple[bool, float, int]:
        started = time.perf_counter()
        try:
            with cache_lock:
                pf = pf_cache.get(path)
                if pf is None:
                    with PARQUET_NATIVE_LOCK:
                        pf = pq.ParquetFile(path)
                    pf_cache[path] = pf
            with PARQUET_NATIVE_LOCK:
                table = pf.read_row_group(rg, columns=["image"], use_threads=False)
            n = table.num_rows
            return n > 0, time.perf_counter() - started, n
        except Exception:  # noqa: BLE001 - a damaged row group is itself observable
            return False, time.perf_counter() - started, 0

    latencies: list[float] = []
    ok = 0
    items = 0
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(task, p, rg) for p, rg in picks]
        for fut in concurrent.futures.as_completed(futures):
            good, dt, n = fut.result()
            latencies.append(dt)
            if good:
                ok += 1
                items += n
    wall = time.perf_counter() - started
    return {
        "concurrency": concurrency,
        "n_row_groups": len(picks),
        "ok_row_groups": ok,
        "items_read": items,
        "wall_seconds": wall,
        "img_s": items / wall if wall else 0.0,
        "latency_ms_per_row_group": latency_stats_ms(latencies),
    }


def task_s3(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    n_picks = args.sample_n if args.sample_n else 500
    files = textatlas_source_files(TEXTATLAS_ACTIVE_SUBSETS)
    incomplete: list[str] = []
    hard_errors: list[str] = []
    if not files:
        hard_errors.append("no_textatlas_source_files_found")
        return {
            "task": "S3", "verdict": "FAIL", "scope": "textatlas_row_group_concurrency",
            "incomplete_steps": incomplete, "hard_errors": hard_errors,
            "key_numbers": {"exception": "no_files"},
        }
    picks = sample_row_groups(files, n_picks, args.seed)
    if len(picks) < n_picks:
        incomplete.append(f"row_group_picks_short:{len(picks)}_of_{n_picks}")

    levels_result: dict[str, Any] = {}
    for level in S3_CONCURRENCIES:
        step = run_step(
            output_dir, f"S3_level_{level}", textatlas_concurrency_level,
            (picks, level), cap_timeout(args, 300.0),
        )
        if not step.get("ok"):
            incomplete.append(f"level_{level}:{step.get('error', 'timeout')}")
            continue
        levels_result[str(level)] = step["result"]

    if not levels_result:
        hard_errors.append("no_concurrency_level_completed")
    for lvl, r in levels_result.items():
        if r["n_row_groups"] and (r["n_row_groups"] - r["ok_row_groups"]) / r["n_row_groups"] > 0.05:
            hard_errors.append(f"level_{lvl}:error_rate_over_5pct")

    best_img_s = max((r["img_s"] for r in levels_result.values()), default=0.0)
    best_level = max(levels_result, key=lambda k: levels_result[k]["img_s"], default=None)

    if hard_errors:
        verdict = "FAIL"
    elif not levels_result:
        verdict = "INCOMPLETE"
    elif best_img_s >= TEXTATLAS_GATE_PASS_IMG_S:
        verdict = "PASS"
    else:
        # 低优先级完备性检查（需求仅~7 img/s）；未达 1.5x 裕量只 WARN，不因单一吞吐数字否决整个 job
        # （FAIL 只留给真错误，见模块 docstring）。
        verdict = "WARN"

    return {
        "task": "S3",
        "verdict": verdict,
        "scope": "textatlas_row_group_concurrency_completeness_check",
        "n_source_files": len(files),
        "concurrency_levels": levels_result,
        "best_concurrency": best_level,
        "best_img_s": best_img_s,
        "thresholds": {
            "domain_requirement_img_s": DOMAIN_GLOBAL_IMG_S["textatlas"],
            "pass_img_s_min": TEXTATLAS_GATE_PASS_IMG_S,
            "note": "1.5x 裕量按 S2/S4 同一惯例本脚本推导，委托书未给 textatlas 显式数值",
        },
        "incomplete_steps": incomplete,
        "hard_errors": hard_errors,
        "key_numbers": {
            "best_concurrency": best_level,
            "best_img_s": round(best_img_s, 1),
        },
    }


# --------------------------------------------------------------------------- #
# S4 — anime 并发上探：复用 Phase-1 Task E 机制，仅测 64 并发档，10k GET。
# --------------------------------------------------------------------------- #
def anime_url_sample(n: int, seed: int) -> list[dict[str, str]]:
    """文件分层抽样；只读中选行组（机制同 Phase-1 Task E）。"""
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
            out.setdefault(key, {"key": key, "url": str(rec["s3key"])})
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
    """机制同 Phase-1 Task E：指数退避 + 小幅确定性 jitter + Retry-After 尊重(封顶 10s)。"""
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
            req = urllib.request.Request(url, headers={"User-Agent": "mot-offline-arb-phase2/1"})
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                status = int(getattr(response, "status", 200))
                statuses.append(status)
                data = response.read()
            bytes_read = len(data)
            if 200 <= status < 300:
                return {
                    "success": True, "final_status": status, "attempt_statuses": statuses,
                    "errors": errors, "attempts": attempt + 1, "bytes": bytes_read,
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
        time.sleep(delay * (0.9 + 0.2 * rng.random()))
    return {
        "success": False, "final_status": statuses[-1] if statuses else None,
        "attempt_statuses": statuses, "errors": errors, "attempts": attempts_done,
        "bytes": bytes_read, "latency_seconds": time.perf_counter() - started,
    }


def anime_cdn_sweep(
    samples: list[dict[str, str]], concurrency: int, attempts: int, timeout_s: float, seed: int
) -> dict[str, Any]:
    def fetch(rec: dict[str, str]) -> tuple[dict[str, str], dict[str, Any]]:
        return rec, cdn_get(rec["url"], attempts, timeout_s, stable_seed(seed, rec["key"]))

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
    successes = 0
    total_bytes = 0
    all_latencies: list[float] = []
    success_latencies: list[float] = []
    attempts_total = 0
    for rec, result in completed:
        all_latencies.append(float(result["latency_seconds"]))
        attempts_total += int(result["attempts"])
        attempt_statuses.update(int(x) for x in result["attempt_statuses"])
        if result["success"]:
            successes += 1
            total_bytes += int(result["bytes"])
            success_latencies.append(float(result["latency_seconds"]))
    denom = max(attempts_total, 1)
    rate429 = attempt_statuses[429] / denom
    rate403 = attempt_statuses[403] / denom
    rate5xx = sum(v for k, v in attempt_statuses.items() if 500 <= k < 600) / denom
    return {
        "concurrency": concurrency, "requested": len(samples), "completed": len(completed),
        "successful": successes, "wall_seconds": wall,
        "img_s": successes / wall if wall else 0.0,
        "mib_s": (total_bytes / MIB) / wall if wall else None,
        "all_latency_ms": latency_stats_ms(all_latencies),
        "success_latency_ms": latency_stats_ms(success_latencies),
        "rates_over_all_attempts": {"http_429": rate429, "http_403": rate403, "http_5xx": rate5xx},
    }


def task_s4(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    n = args.sample_n if args.sample_n else 10_000
    concurrency = min(64, args.anime_concurrency)
    incomplete: list[str] = []
    hard_errors: list[str] = []
    samples = anime_url_sample(n, args.seed)
    if len(samples) < n:
        incomplete.append(f"samples_short:{len(samples)}_of_{n}")

    step = run_step(
        output_dir, "S4_sweep_64", anime_cdn_sweep,
        (samples, concurrency, args.http_attempts, args.http_timeout_s, args.seed),
        cap_timeout(args, 1800.0),
    )
    if not step.get("ok"):
        incomplete.append(f"sweep:{step.get('error', 'timeout')}")
        return {
            "task": "S4", "verdict": "INCOMPLETE", "scope": "anime_cdn_64_concurrency_get",
            "requested_keys": n, "incomplete_steps": incomplete, "hard_errors": hard_errors,
            "key_numbers": {"incomplete": True},
        }
    result = step["result"]
    if result["completed"] < len(samples):
        incomplete.append("sweep:did_not_complete_all_samples")

    rate429 = result["rates_over_all_attempts"]["http_429"]
    throughput = result["img_s"]
    gate_fail = rate429 >= 0.001
    if gate_fail:
        hard_errors.append("http_429_rate_over_0.1pct")
        verdict = "FAIL"
    elif incomplete:
        verdict = "INCOMPLETE"
    elif throughput >= ANIME_GATE_PASS_IMG_S:
        verdict = "PASS"
    else:
        verdict = "WARN"

    return {
        "task": "S4",
        "verdict": verdict,
        "scope": "anime_cdn_64_concurrency_get",
        "requested_keys": n,
        "result": result,
        "thresholds": {
            "concurrency": 64,
            "pass_img_s_min": ANIME_GATE_PASS_IMG_S,
            "http_429_rate_strict_max": 0.001,
        },
        "incomplete_steps": incomplete,
        "hard_errors": hard_errors,
        "key_numbers": {
            "img_s": round(throughput, 1),
            "rate429": rate429,
            "completed": result["completed"],
        },
    }


# --------------------------------------------------------------------------- #
# S5 — 单节点 8-rank 混合负载终测（gate 本体）。
#
# 设计：每个 worker 进程模拟一个 rank；rank 内部对每个域各起一个小线程池
# （DomainSource，producer 持续 fetch+decode 塞入有界 queue），主循环按 domain_mix 权重
# 加权随机选域、从对应 queue 阻塞取一个已完成 item——这样任何一个域供给跟不上其配比份额时，
# 主循环会自然阻塞在该域的 queue.get() 上，如实反映"这个域拖慢了整机"，而不是让各域各算各的
# 独立吞吐再线性相加（那样测不出 mix 下的相互挤占）。
#
# 每域并发 = 该域在 S1-S4 中测得的最佳档位 / 8（四舍五入，至少 1），近似把"全节点最佳并发"
# 摊到 8 个模拟 rank 上；这是本脚本的建模简化，非委托书逐字给出的公式，见 resolve_* 函数注释。
# --------------------------------------------------------------------------- #
def per_worker_concurrency(total: int, n_workers: int = 8) -> int:
    return max(1, round(total / n_workers))


class DomainSource:
    """一个域的后台生产线程池：每线程持有独立 cycle 迭代器，持续 fetch(+decode)，
    (ok, elapsed_seconds) 结果塞进有界 queue；消费方阻塞 get()。"""

    def __init__(
        self, concurrency: int,
        iter_factory: Callable[[], Any],
        process_fn: Callable[[Any], bool],
    ) -> None:
        self.queue: "queue.Queue[tuple[bool, float]]" = queue.Queue(maxsize=max(8, concurrency * 8))
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        for _ in range(concurrency):
            it = iter_factory()
            th = threading.Thread(target=self._run, args=(it, process_fn), daemon=True)
            th.start()
            self.threads.append(th)

    def _run(self, it: Any, process_fn: Callable[[Any], bool]) -> None:
        while not self.stop_event.is_set():
            started = time.perf_counter()
            try:
                item = next(it)
                ok = bool(process_fn(item))
            except StopIteration:
                ok = False
                time.sleep(0.05)  # 防御性：pool 为空时 cycle([]) 会立即再次 StopIteration，避免忙等
            except Exception:  # noqa: BLE001 - a fetch/decode failure is data, not a crash
                ok = False
            dt = time.perf_counter() - started
            try:
                self.queue.put((ok, dt), timeout=1.0)
            except queue.Full:
                pass  # 消费方落后于该域配额；丢弃本次计时样本，producer 立刻重试下一 item

    def get(self, timeout: float = 2.0) -> tuple[bool, float] | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self.stop_event.set()


def _monet_synth_range_iter_factory(items: list[tuple[str, str, int, int]], seed_offset: int) -> Callable[[], Any]:
    def factory() -> Any:
        pool = list(items)
        random.Random(seed_offset).shuffle(pool)
        return itertools.cycle(pool)
    return factory


def _monet_synth_range_process(item: tuple[str, str, int, int]) -> bool:
    tar_path, _key, offset, size = item
    try:
        with open(tar_path, "rb", buffering=0) as f:
            f.seek(offset)
            raw = f.read(size)
        if len(raw) != size:
            return False
        return decode_rgb(raw) is not None
    except OSError:
        return False


def _monet_synth_shard_iter_factory(tar_specs: list[tuple[str, frozenset[str]]], seed_offset: int) -> Callable[[], Any]:
    def factory() -> Any:
        specs = list(tar_specs)
        random.Random(seed_offset).shuffle(specs)

        def gen() -> Any:
            i = 0
            n = len(specs)
            while True:
                tar_path, wanted = specs[i % n]
                yielded = False
                for _key, raw in stream_tar_images(tar_path, set(wanted)):
                    yielded = True
                    yield (raw,)
                if not yielded:
                    time.sleep(0.05)  # 避免空 tar/坏路径下的忙等
                i += 1
        return gen()
    return factory


def _monet_synth_shard_process(item: tuple[bytes]) -> bool:
    (raw,) = item
    return decode_rgb(raw) is not None


def _gpic_iter_factory(items: list[tuple[dict[str, Any], str]], seed_offset: int) -> Callable[[], Any]:
    def factory() -> Any:
        pool = list(items)
        random.Random(seed_offset).shuffle(pool)
        return itertools.cycle(pool)
    return factory


def _gpic_process(item: tuple[dict[str, Any], str]) -> bool:
    rec, ext = item
    data = read_gpic_bytes(rec["src_root"], rec["src_shard"], rec["src_key"], ext=ext)
    if not data:
        return False
    return decode_rgb(data) is not None


def _textatlas_iter_factory(picks: list[tuple[str, int]], seed_offset: int) -> Callable[[], Any]:
    def factory() -> Any:
        pool = list(picks)
        random.Random(seed_offset).shuffle(pool)
        return itertools.cycle(pool)
    return factory


def _make_textatlas_process() -> Callable[[tuple[str, int]], bool]:
    pf_cache: dict[str, pq.ParquetFile] = {}
    cache_lock = threading.Lock()

    def process(item: tuple[str, int]) -> bool:
        path, rg = item
        try:
            with cache_lock:
                pf = pf_cache.get(path)
                if pf is None:
                    with PARQUET_NATIVE_LOCK:
                        pf = pq.ParquetFile(path)
                    pf_cache[path] = pf
            with PARQUET_NATIVE_LOCK:
                table = pf.read_row_group(rg, columns=["image"], use_threads=False)
            if table.num_rows == 0:
                return False
            image = table.column("image")[0].as_py()
            raw = image.get("bytes") if isinstance(image, dict) else None
            if not raw:
                return False
            return decode_rgb(raw) is not None
        except Exception:  # noqa: BLE001 - a damaged row group is itself observable
            return False
    return process


def _anime_iter_factory(records: list[dict[str, str]], seed_offset: int) -> Callable[[], Any]:
    def factory() -> Any:
        pool = list(records)
        random.Random(seed_offset).shuffle(pool)
        return itertools.cycle(pool)
    return factory


def _anime_process(rec: dict[str, str]) -> bool:
    raw = fetch_url_bytes(rec["url"], retries=3, timeout=20, cache_dir=None)
    if raw is None:
        return False
    return decode_rgb(raw) is not None


def resolve_s5_backend(output_dir: Path, args: argparse.Namespace) -> tuple[str, str]:
    p = output_dir / "taskS1_result.json"
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            verdict = str(data.get("verdict", ""))
            if verdict == "DECISIVE_SHARD_WINDOW":
                return "shard_window", f"taskS1_result.json verdict={verdict}"
            if verdict in ("DECISIVE_RANGE", "DECISIVE_INCONCLUSIVE"):
                # INCONCLUSIVE 时默认 range：range_semantics 已由 Phase-1 Task B 证实为 true，
                # 且 range 后端实现更简单/不依赖整 shard 扫描，作为并列时的默认更保守选择。
                return "range", f"taskS1_result.json verdict={verdict}"
        except Exception:  # noqa: BLE001 - fall through to override/default below
            pass
    if args.s5_backend:
        return args.s5_backend, "--s5-backend override"
    return "range", "no taskS1_result.json and no --s5-backend override; defaulting to range"


def resolve_sibling_concurrency(
    output_dir: Path, task_letter: str, override: int | None, default: int
) -> tuple[int, str]:
    p = output_dir / f"task{task_letter}_result.json"
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            best = data.get("best_concurrency")
            if best:
                return int(best), f"task{task_letter}_result.json best_concurrency={best}"
        except Exception:  # noqa: BLE001 - fall through to override/default below
            pass
    if override:
        return int(override), f"--s5 override={override}"
    return default, f"no task{task_letter}_result.json and no override; defaulting to {default}"


def build_monet_synth_pools(
    output_dir: Path, args: argparse.Namespace, backend: str,
    incomplete: list[str], hard_errors: list[str],
) -> tuple[Any, Any, dict[str, Any]]:
    """优先复用同一 output_dir 下 S1 已经建好的 seek_jump 索引；独立跑 S5 时现建（同一机制/同一
    tar 组），避免强耦合到 S1 的内部产物格式之外的东西——只要求 index parquet 的列 schema。"""
    index_dir = output_dir / "s1_indexes"
    built: list[dict[str, Any]] = []
    existing = sorted(index_dir.glob("*.parquet")) if index_dir.is_dir() else []
    for p in existing:
        try:
            t = pq.read_table(p, columns=["domain", "sub_source"])
        except Exception:  # noqa: BLE001 - unreadable prior index, rebuild fresh below
            continue
        if t.num_rows == 0:
            continue
        built.append({
            "index_path": str(p),
            "domain": t.column("domain")[0].as_py(),
            "sub_source": t.column("sub_source")[0].as_py(),
        })
    reused = bool(built)
    if not built:
        for domain, sub, n_tars in S1_GROUPS:
            tars = first_n_tars(group_base(domain, sub), n_tars)
            if len(tars) < n_tars:
                incomplete.append(f"s5_index/{domain}/{sub}:only_{len(tars)}_tars")
            for i, tar in enumerate(tars):
                idx_path = index_dir / f"{domain}__{sub.replace('-', '_')}__{i:02d}.parquet"
                step = run_step(
                    output_dir, f"S5_index_{domain}_{sub}_{i}", scan_tar_index,
                    (str(tar), str(idx_path), domain, sub), cap_timeout(args, 900.0),
                )
                if not step.get("ok"):
                    incomplete.append(f"s5_index/{domain}/{sub}/{i}:{step.get('error', 'timeout')}")
                    continue
                built.append(step["result"])
    if not built:
        hard_errors.append("s5_no_monet_synth_index")
        return [], [], {"mode": backend, "reused_s1_indexes": reused, "n_indexes": 0}

    if backend == "range":
        monet_rows: list[tuple[str, str, int, int]] = []
        synth_rows: list[tuple[str, str, int, int]] = []
        for b in built:
            t = pq.read_table(b["index_path"], columns=["tar_path", "key", "data_offset", "size", "domain"])
            cols = [t.column(c).to_pylist() for c in ("tar_path", "key", "data_offset", "size", "domain")]
            for tp, key, off, sz, dom in zip(*cols):
                row = (tp, str(key), int(off), int(sz))
                (monet_rows if dom == "monet_real" else synth_rows).append(row)
        return monet_rows, synth_rows, {"mode": "range", "reused_s1_indexes": reused, "n_indexes": len(built)}

    monet_specs: list[tuple[str, frozenset[str]]] = []
    synth_specs: list[tuple[str, frozenset[str]]] = []
    for b in built:
        t = pq.read_table(b["index_path"], columns=["tar_path", "key", "domain"])
        by_tar: dict[str, set[str]] = defaultdict(set)
        dom = b["domain"]
        for tp, key in zip(t.column("tar_path").to_pylist(), t.column("key").to_pylist()):
            by_tar[tp].add(str(key))
        for tp, keys in by_tar.items():
            spec = (tp, frozenset(keys))
            (monet_specs if dom == "monet_real" else synth_specs).append(spec)
    return monet_specs, synth_specs, {"mode": "shard_window", "reused_s1_indexes": reused, "n_indexes": len(built)}


def build_gpic_pool(args: argparse.Namespace, n: int) -> list[tuple[dict[str, Any], str]]:
    candidates = gpic_manifest_sample(max(n * 2, n + 200), args.seed + 5005)
    resolved, _errors = resolve_gpic_ext(candidates)
    return resolved[:n]


def build_textatlas_pool(args: argparse.Namespace, n: int) -> list[tuple[str, int]]:
    files = textatlas_source_files(TEXTATLAS_ACTIVE_SUBSETS)
    return sample_row_groups(files, n, args.seed + 6006)


def build_anime_pool(args: argparse.Namespace, n: int) -> list[dict[str, str]]:
    return anime_url_sample(n, args.seed + 7007)


def s5_worker_main(
    rank: int,
    domain_plan: dict[str, dict[str, Any]],
    mix_weights: dict[str, float],
    duration_s: float,
    batch_size: int,
    seed: int,
    result_path_s: str,
) -> None:
    try:
        domains = list(mix_weights)
        weights = [mix_weights[d] for d in domains]
        sources = {
            d: DomainSource(domain_plan[d]["concurrency"], domain_plan[d]["iter_factory"], domain_plan[d]["process"])
            for d in domains
        }
        rng = random.Random(seed + rank * 7919 + 13)
        counts = {d: 0 for d in domains}
        errors = {d: 0 for d in domains}
        latencies: dict[str, list[float]] = {d: [] for d in domains}
        batch_ready: list[float] = []
        total_ok = 0
        start = time.perf_counter()
        last_batch_t = start
        deadline = start + duration_s
        try:
            while time.perf_counter() < deadline:
                d = rng.choices(domains, weights=weights, k=1)[0]
                got = sources[d].get(timeout=2.0)
                if got is None:
                    continue
                ok, dt = got
                if ok:
                    counts[d] += 1
                    total_ok += 1
                    if len(latencies[d]) < 5000:
                        latencies[d].append(dt)
                    if total_ok % batch_size == 0:
                        now = time.perf_counter()
                        batch_ready.append(now - last_batch_t)
                        last_batch_t = now
                else:
                    errors[d] += 1
        finally:
            for s in sources.values():
                s.close()
        wall = time.perf_counter() - start
        payload = {
            "ok": True,
            "result": {
                "rank": rank,
                "wall_seconds": wall,
                "total_ok": total_ok,
                "img_s": total_ok / wall if wall else 0.0,
                "counts": counts,
                "errors": errors,
                "latency_ms_by_domain": {d: latency_stats_ms(v) for d, v in latencies.items()},
                "batches_completed": len(batch_ready),
                "batch_ready_ms": latency_stats_ms(batch_ready),
            },
        }
    except Exception as exc:  # noqa: BLE001 - a worker crash must be observable, not lost
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=20)}
    atomic_json(Path(result_path_s), payload)


def task_s5(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    incomplete: list[str] = []
    hard_errors: list[str] = []

    backend, backend_reason = resolve_s5_backend(output_dir, args)
    gpic_conc, gpic_reason = resolve_sibling_concurrency(output_dir, "S2", args.s5_gpic_concurrency, 64)
    ta_conc, ta_reason = resolve_sibling_concurrency(output_dir, "S3", args.s5_textatlas_concurrency, 32)
    anime_conc = args.s5_anime_concurrency

    monet_pool, synth_pool, ms_notes = build_monet_synth_pools(output_dir, args, backend, incomplete, hard_errors)
    if hard_errors:
        return {
            "task": "S5", "verdict": "FAIL", "scope": "single_node_8rank_mixed_workload_gate",
            "backend_decision": {"monet_synth_backend": backend, "reason": backend_reason},
            "incomplete_steps": incomplete, "hard_errors": hard_errors,
            "key_numbers": {"exception": "monet_synth_pool_build_failed"},
        }

    gpic_n = max(2000, per_worker_concurrency(gpic_conc) * 8 * 50)
    ta_n = max(200, per_worker_concurrency(ta_conc) * 8 * 20)
    anime_n = max(2000, per_worker_concurrency(anime_conc) * 8 * 50)
    gpic_pool = build_gpic_pool(args, gpic_n)
    ta_pool = build_textatlas_pool(args, ta_n)
    anime_pool = build_anime_pool(args, anime_n)
    if not gpic_pool:
        hard_errors.append("s5_gpic_pool_empty")
    if not ta_pool:
        incomplete.append("s5_textatlas_pool_empty")
    if not anime_pool:
        hard_errors.append("s5_anime_pool_empty")
    if backend == "range" and not monet_pool:
        hard_errors.append("s5_monet_pool_empty")
    if backend == "range" and not synth_pool:
        hard_errors.append("s5_synth_pool_empty")
    if backend == "shard_window" and not monet_pool:
        hard_errors.append("s5_monet_shard_specs_empty")
    if backend == "shard_window" and not synth_pool:
        hard_errors.append("s5_synth_shard_specs_empty")
    if hard_errors:
        return {
            "task": "S5", "verdict": "FAIL", "scope": "single_node_8rank_mixed_workload_gate",
            "backend_decision": {"monet_synth_backend": backend, "reason": backend_reason},
            "incomplete_steps": incomplete, "hard_errors": hard_errors,
            "key_numbers": {"exception": "pool_build_failed"},
        }

    domain_plan: dict[str, dict[str, Any]] = {
        "gpic": {
            "concurrency": per_worker_concurrency(gpic_conc),
            "iter_factory": _gpic_iter_factory(gpic_pool, args.seed + 101),
            "process": _gpic_process,
        },
        "textatlas": {
            "concurrency": per_worker_concurrency(ta_conc),
            "iter_factory": _textatlas_iter_factory(ta_pool, args.seed + 102),
            "process": _make_textatlas_process(),
        },
        "anime": {
            "concurrency": per_worker_concurrency(anime_conc),
            "iter_factory": _anime_iter_factory(anime_pool, args.seed + 103),
            "process": _anime_process,
        },
    }
    if backend == "range":
        domain_plan["monet_real"] = {
            "concurrency": per_worker_concurrency(S1_RANGE_CONCURRENCIES[-1]),
            "iter_factory": _monet_synth_range_iter_factory(monet_pool, args.seed + 104),
            "process": _monet_synth_range_process,
        }
        domain_plan["synthetic"] = {
            "concurrency": per_worker_concurrency(S1_RANGE_CONCURRENCIES[-1]),
            "iter_factory": _monet_synth_range_iter_factory(synth_pool, args.seed + 105),
            "process": _monet_synth_range_process,
        }
    else:
        domain_plan["monet_real"] = {
            "concurrency": per_worker_concurrency(S1_SHARD_CONCURRENCIES[-1]),
            "iter_factory": _monet_synth_shard_iter_factory(monet_pool, args.seed + 104),
            "process": _monet_synth_shard_process,
        }
        domain_plan["synthetic"] = {
            "concurrency": per_worker_concurrency(S1_SHARD_CONCURRENCIES[-1]),
            "iter_factory": _monet_synth_shard_iter_factory(synth_pool, args.seed + 105),
            "process": _monet_synth_shard_process,
        }

    duration_s = args.s5_duration_s
    scaled = duration_s < 600.0
    arg_tuples = [
        (rank, domain_plan, DOMAIN_MIX_S5, duration_s, BATCH_SIZE_PER_RANK, args.seed)
        for rank in range(8)
    ]
    worker_payloads = run_parallel_processes(
        output_dir, "S5_worker", s5_worker_main, arg_tuples, timeout_s=duration_s + 300.0,
    )
    ok_workers = [w["result"] for w in worker_payloads if w.get("ok")]
    crashed = [w for w in worker_payloads if not w.get("ok") and not w.get("timeout")]
    timed_out = [w for w in worker_payloads if w.get("timeout")]
    if crashed:
        hard_errors.append(f"{len(crashed)}_worker(s)_crashed")
    if timed_out:
        incomplete.append(f"{len(timed_out)}_worker(s)_timed_out")
    if len(ok_workers) < 8:
        incomplete.append(f"workers_completed={len(ok_workers)}_of_8")

    per_worker_img_s = [w["img_s"] for w in ok_workers]
    total_img_s = sum(per_worker_img_s)
    avg_worker = statistics.mean(per_worker_img_s) if per_worker_img_s else 0.0
    min_worker = min(per_worker_img_s) if per_worker_img_s else 0.0
    straggler = (min_worker < 0.7 * avg_worker) if per_worker_img_s else True
    max_rank_batch_ready_p99_ms = max(
        (w["batch_ready_ms"]["p99_ms"] for w in ok_workers if w["batch_ready_ms"]["p99_ms"] is not None),
        default=None,
    )

    if hard_errors:
        verdict = "FAIL"
    elif len(ok_workers) < 8:
        verdict = "INCOMPLETE"
    elif total_img_s >= NODE_GATE_IMG_S and not straggler:
        verdict = "WARN" if scaled else "PASS"
    else:
        # 这是 Phase-2 的 gate 本体（docs/OFFLINE_ARB_V1_SPEC.md 第3节）：未达标是真实的
        # "不过 → 停,找 Evan" 结论,不是后端选型的模糊地带,故此处 FAIL 是合法的(非 DECISIVE_*)。
        verdict = "FAIL"

    return {
        "task": "S5",
        "verdict": verdict,
        "scope": "single_node_8rank_mixed_workload_gate",
        "backend_decision": {
            "monet_synth_backend": backend, "reason": backend_reason,
            "monet_synth_pool": ms_notes,
            "gpic_concurrency_total": gpic_conc, "gpic_concurrency_reason": gpic_reason,
            "textatlas_concurrency_total": ta_conc, "textatlas_concurrency_reason": ta_reason,
            "anime_concurrency_total": anime_conc,
        },
        "domain_mix": DOMAIN_MIX_S5,
        "duration_seconds": duration_s,
        "scaled": scaled,
        "batch_size_per_rank": BATCH_SIZE_PER_RANK,
        "workers": worker_payloads,
        "aggregate": {
            "total_img_s": total_img_s,
            "avg_worker_img_s": avg_worker,
            "min_worker_img_s": min_worker,
            "straggler_detected": straggler,
            "max_rank_batch_ready_p99_ms": max_rank_batch_ready_p99_ms,
        },
        "thresholds": {
            "node_gate_img_s": NODE_GATE_IMG_S,
            "straggler_min_over_avg": 0.7,
        },
        "incomplete_steps": incomplete,
        "hard_errors": hard_errors,
        "key_numbers": {
            "total_img_s": round(total_img_s, 1),
            "min_worker_img_s": round(min_worker, 1),
            "straggler": straggler,
            "backend": backend,
        },
    }


TASKS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "S1": task_s1, "S2": task_s2, "S3": task_s3, "S4": task_s4, "S5": task_s5,
}

# 非 FAIL/PASS/WARN/INCOMPLETE 的合法结论 token（S1 后端否决类结论，见模块 docstring）。
DECISIVE_VERDICTS = {"DECISIVE_RANGE", "DECISIVE_SHARD_WINDOW", "DECISIVE_INCONCLUSIVE"}


def run_one(task: str, args: argparse.Namespace) -> str:
    started_wall = utc_now()
    started = time.perf_counter()
    try:
        result = TASKS[task](args)
    except Exception as exc:  # noqa: BLE001 - persist a machine-readable failure
        result = {
            "task": task, "verdict": "FAIL", "scope": "exception",
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=30),
            "key_numbers": {"exception": type(exc).__name__},
        }
    valid_verdicts = {"PASS", "FAIL", "WARN", "INCOMPLETE"} | DECISIVE_VERDICTS
    if result.get("verdict") not in valid_verdicts:
        # 内部一致性自检：任何 task 函数吐出未知 verdict token 都是脚本 bug，不应静默放行。
        result = {
            "task": task, "verdict": "FAIL", "scope": "internal_error",
            "error": f"unknown verdict token: {result.get('verdict')!r}",
            "key_numbers": {"exception": "unknown_verdict"},
        }
    result.update({
        "schema_version": "offline_arb_phase2_v1",
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


def parse_tests(raw: str) -> list[str]:
    tests = [x.strip().upper() for x in raw.split(",") if x.strip()]
    invalid = [x for x in tests if x not in TASKS]
    if not tests or invalid:
        raise argparse.ArgumentTypeError(f"--test must be a comma-separated subset of S1..S5; invalid={invalid}")
    # S5 读同一 output_dir 里 S1/S2/S3 的产物自动选后端/并发；组合运行时固定按 S1->S5 顺序执行，
    # 不按命令行书写顺序。
    requested = set(tests)
    return [t for t in ("S1", "S2", "S3", "S4", "S5") if t in requested]


def validate_output_dir(raw: str) -> Path:
    path = Path(raw).expanduser().resolve(strict=False)
    parent = OUTPUT_PARENT.resolve(strict=False)
    if path.parent != parent or not path.name.startswith("offline_arb_phase2_") or len(path.name) <= len("offline_arb_phase2_"):
        raise ValueError(
            f"--output-dir must be {OUTPUT_PARENT}/offline_arb_phase2_${{RUN_TAG}}, got {path}"
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, type=parse_tests, help="s1..s5 or a comma-separated subset")
    parser.add_argument("--output-dir", required=True, help="Run output directory; the only write destination")
    parser.add_argument(
        "--sample-n", type=int, default=None,
        help="按 test 缩放：S1 每并发档 range item 数(默认300)；S2 每并发档文件数(默认500)；"
             "S3 (shard,row_group) 抽样数(默认500)；S4 GET key 数(默认10000)。S5 不用此参数，见 --s5-duration-s。",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--step-timeout-s", type=float, default=1_800.0, help="单个可 kill 子步骤的封顶 timeout")
    parser.add_argument("--http-attempts", type=int, default=3, help="S4 每 key 最多尝试次数")
    parser.add_argument("--http-timeout-s", type=float, default=20.0, help="S4 单次 HTTP 请求 timeout")
    parser.add_argument("--anime-concurrency", type=int, default=64, help="S4 CDN 并发；硬上限 64")
    parser.add_argument("--s5-duration-s", type=float, default=600.0, help="S5 混合负载持续时长（默认 10 分钟）")
    parser.add_argument(
        "--s5-backend", choices=("range", "shard_window"), default=None,
        help="S5 monet/synth 后端；独立跑 S5 且同 output-dir 无 taskS1_result.json 时必须显式指定，"
             "否则默认 range 并在结果里注明理由",
    )
    parser.add_argument("--s5-gpic-concurrency", type=int, default=None, help="S5 gpic 总并发覆盖；默认读 taskS2_result.json")
    parser.add_argument("--s5-textatlas-concurrency", type=int, default=None, help="S5 textatlas 总并发覆盖；默认读 taskS3_result.json")
    parser.add_argument("--s5-anime-concurrency", type=int, default=64, help="S5 anime 总并发；硬上限 64")
    args = parser.parse_args()
    if args.step_timeout_s <= 0 or args.http_attempts <= 0 or args.http_timeout_s <= 0:
        parser.error("timeouts and attempts must be positive")
    if args.s5_duration_s <= 0:
        parser.error("--s5-duration-s must be positive")
    if args.anime_concurrency <= 0 or args.s5_anime_concurrency <= 0:
        parser.error("concurrency must be positive")
    args.anime_concurrency = min(64, args.anime_concurrency)
    args.s5_anime_concurrency = min(64, args.s5_anime_concurrency)
    try:
        args.output_dir = validate_output_dir(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    verdicts = [run_one(task, args) for task in args.test]
    return 2 if "FAIL" in verdicts else 0


if __name__ == "__main__":
    sys.exit(main())
