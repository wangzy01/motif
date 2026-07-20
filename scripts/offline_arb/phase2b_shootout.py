#!/usr/bin/env python3
"""OFFLINE_ARB_V1 Phase-2b IO shootout：T1..T4，对 Phase-2 (job p2-io-rp25) 结论的两点修正复测。

修正背景（Evan/主线已定，2026-07-10）：
  1. Phase-2 S1 的 DECISIVE_RANGE 是「稀疏随机集」前提下的产物——range 79.6 img/s@64 输给
     sequential 基线等效 3228 img/s 两个数量级；shard_window 20.5 img/s 是 tarfile 'r|' 小块读
     被 FUSE 放大的 artifact，不是真实上限。→ T1 在密集访问模式下重比（大块顺序 vs 系数化
     coalesced range）。
  2. Phase-2 S5 的浅预取不代表 plan-driven loader 真实形态（plan 确定 → 可任意深度预取）。
     → T2 实测 gpic 深预取管线（S3→/local→消费），T4 用各 T 胜者复测混合 gate。

机制全部复用 phase2_shootout.py（同目录 import；atomic_json/append_summary/run_step fork 隔离/
run_parallel_processes/采样器/常量），本文件只新增 Phase-2b 特有的测量逻辑。

四个子测试（--test T1..T4，默认全跑，按 T1→T4 顺序执行——T4 依赖 T1/T2/T3 的结论/产物）：
  T1 密集 shard 聚簇读：monet cc12m+coyo、synth flux-klein 各 2 tar（复用 --reuse-index-dir 里
     Phase-2 现成的 s1_indexes parquet，没有就重建）。(a) buffered 顺序全扫=大块 read+按 index
     手工切 member（不走 tarfile）；(b) 系数化 coalesced range=offset 排序后 <4MB 间隙合并成大
     块 ranged read、内存切分。密度 sweep 10/30/100% × tar 级并发 1/2/4，输出 wanted-img/s。
     判定：任一模式在 30% 密度、并发≤4 下 wanted-img/s ≥ 400（单节点 monet+synth 需求
     ~106=(187+24)/2 的 ~3.8x，留深余量）→ DECISIVE_DENSE_SEQ / DECISIVE_DENSE_COALESCE；
     否则 DECISIVE_INCONCLUSIVE（曲线落盘待人工判）。
     密度语义：「wanted 集取 index 前缀比例」解释为【固定种子 shuffle 后的前缀】→
     10%⊂30%⊂100% 嵌套随机子集（跨密度可比、覆盖全 tar）；物理文件前缀测不出密度效应。
  T2 gpic 深预取管线：预取进程池（1×32 与 2×16 两种布局）把随机 gpic 项持续下载到
     /local/evan/gpic_prefetch_test，消费者以 59.5 img/s（=119/2 单节点需求）目标速率从 /local
     读+decode。测稳态预取 img/s、积压深度曲线、消费者 p99 batch-ready。
     判定：预取稳态 ≥90 img/s（1.5×59.5）→ PASS。
  T3 textatlas 补测：样本 100 row-group、并发 1/8（修正 Phase-2 S3 的测量缺陷：S3 把
     read_row_group 包进全局 PARQUET_NATIVE_LOCK，并发档被串行化；本次每 task 独立
     ParquetFile 实例、无全局锁）。若仍慢 → 附测「整 shard 拷贝到 /local + 本地读」分段耗时
     （用 shutil.copyfileobj 分块拷贝，与 prestage_textatlas.py 生产 staging 机制同构）。
     产出无论如何不判 FAIL（生产走 /local staging，S3 只是回退）。
  T4 混合 gate 复测（S5b）：monet/synth 用 T1 胜者密集聚簇（per-worker 分 shard），gpic 用
     T2 深预取（消费端从 /local），anime 64 并发 CDN，textatlas 用 T3 最优或 7 img/s 常数流
     模拟（结果里显式标注 simulated）。8 worker × 10 分钟，gate 同 Phase-2 S5：
     聚合 ≥347 img/s 且 min worker ≥ 0.7×avg。

硬约束：只读数据源；输出仅 --output-dir；唯一允许的非 output-dir 写入 =
/local/evan/gpic_prefetch_test（T2/T3 staging/T4 预取共用此根，job 结尾 rm -rf）；CDN 并发 ≤64
带退避；无周期 faulthandler；每步 timeout；FAIL 只留给真错误——T1 的模式选型结论用
DECISIVE_* token；T3 永不 FAIL。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing as mp
import os
import queue
import random
import shutil
import socket
import statistics
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # 直跑与 py_compile 均可靠地找到同目录的 phase2_shootout
    sys.path.insert(0, str(_HERE))
import phase2_shootout as p2  # noqa: E402  复用 Phase-2 机制（见模块 docstring）

from training.dataline.image_fetch import decode_rgb, read_gpic_bytes  # noqa: E402

SEED = p2.SEED
MIB = p2.MIB
OUTPUT_PARENT = p2.OUTPUT_PARENT

# 本次唯一允许的非 output-dir 写入点（Evan 委托固定路径；T3 staging 与 T4 预取也放它下面的
# 子目录——单一 sanctioned 根，job 结尾统一 rm -rf）。
LOCAL_SCRATCH_DEFAULT = "/local/evan/gpic_prefetch_test"

T1_DENSITIES = (0.10, 0.30, 1.00)
T1_TAR_CONCURRENCIES = (1, 2, 4)
T1_GATE_IMG_S = 400.0  # 单节点 monet+synth 需求 ~106 img/s（(187+24)/2≈105.5）的 ~3.8x，留深余量（已定）
T1_DECISION_DENSITY = 0.30
T1_COALESCE_GAP_BYTES = 4 * MIB
T1_MAX_RUN_BYTES = 256 * MIB  # coalesced 大块读的单块上限，防止 100% 密度把整个 5GB tar 读进内存

T2_TARGET_IMG_S = 119.0 / 2.0  # gpic 单节点需求 59.5
T2_GATE_IMG_S = 90.0  # 1.5×59.5≈89.25，委托口径写 90
T2_ITEMS_DEFAULT = 20_000
T2_CONFIGS = (("p1x32", 1, 32), ("p2x16", 2, 16))

T3_GATE_IMG_S = p2.TEXTATLAS_GATE_PASS_IMG_S  # 10.5 = 1.5×7

T4_GPIC_PREFETCH_THREADS_PER_WORKER = 4   # 节点总 32 并发（委托口径）摊到 8 worker
T4_ANIME_THREADS_PER_WORKER = 8           # 节点总 64 并发（S4 实测 364 img/s 健康）


def utc_now() -> str:
    return p2.utc_now()


# --------------------------------------------------------------------------- #
# T1 — 索引准备与密集扫描
# --------------------------------------------------------------------------- #
def _load_index_parquet(path: Path) -> dict[str, Any] | None:
    """读一个 Phase-2 scan_tar_index 产出的 index parquet → per-tar 条目；不可用返回 None。"""
    try:
        t = pq.read_table(path, columns=["tar_path", "data_offset", "size", "domain", "sub_source"])
    except Exception:  # noqa: BLE001 - unreadable index is skipped, caller may rebuild
        return None
    if t.num_rows == 0:
        return None
    tar_path = str(t.column("tar_path")[0].as_py())
    offsets = t.column("data_offset").to_pylist()
    sizes = t.column("size").to_pylist()
    members = sorted(zip((int(o) for o in offsets), (int(s) for s in sizes)))
    try:
        tar_bytes = os.stat(tar_path).st_size
    except OSError:
        return None
    end_needed = max(o + s for o, s in members)
    if tar_bytes < end_needed:  # 索引与 tar 不匹配（源被换过）→ 不可用
        return None
    return {
        "tar_path": tar_path,
        "domain": str(t.column("domain")[0].as_py()),
        "sub_source": str(t.column("sub_source")[0].as_py()),
        "members": members,
        "n_members": len(members),
        "payload_bytes": sum(s for _o, s in members),
        "tar_bytes": tar_bytes,
        "index_path": str(path),
    }


def prepare_indexes(
    args: argparse.Namespace, output_dir: Path, incomplete: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """定位/重建 6 个 tar 的 member 索引。优先 --reuse-index-dir（Phase-2 输出目录或其
    s1_indexes 子目录），其次本 run 已重建过的 t1_indexes，最后用 p2.scan_tar_index 现建。"""
    candidates: list[Path] = []
    if args.reuse_index_dir:
        base = Path(args.reuse_index_dir)
        for sub in (base, base / "s1_indexes"):
            if sub.is_dir() and list(sub.glob("*.parquet")):
                candidates.append(sub)
                break
    local_dir = output_dir / "t1_indexes"
    if local_dir.is_dir() and list(local_dir.glob("*.parquet")):
        candidates.append(local_dir)

    for src in candidates:
        entries = [e for e in (_load_index_parquet(p) for p in sorted(src.glob("*.parquet"))) if e]
        if entries:
            return entries, {"source": str(src), "rebuilt": False, "n_indexes": len(entries)}

    # 重建（Phase-1 已证 seek_jump 可行；复用 p2.scan_tar_index 同一机制/同一 tar 组）
    entries = []
    for domain, sub, n_tars in p2.S1_GROUPS:
        tars = p2.first_n_tars(p2.group_base(domain, sub), n_tars)
        if len(tars) < n_tars:
            incomplete.append(f"index/{domain}/{sub}:only_{len(tars)}_tars")
        for i, tar in enumerate(tars):
            idx_path = local_dir / f"{domain}__{sub.replace('-', '_')}__{i:02d}.parquet"
            step = p2.run_step(
                output_dir, f"T1_index_{domain}_{sub}_{i}", p2.scan_tar_index,
                (str(tar), str(idx_path), domain, sub), p2.cap_timeout(args, 900.0),
            )
            if not step.get("ok"):
                incomplete.append(f"index/{domain}/{sub}/{i}:{step.get('error', 'timeout')}")
                continue
            e = _load_index_parquet(idx_path)
            if e:
                entries.append(e)
    return entries, {"source": str(local_dir), "rebuilt": True, "n_indexes": len(entries)}


def density_subset(
    members: Sequence[tuple[int, int]], density: float, seed: int, cap: int | None = None
) -> list[tuple[int, int]]:
    """固定种子 shuffle 后取前缀比例 → 嵌套随机子集（10%⊂30%⊂100%），再按 offset 排序供扫描。"""
    order = list(range(len(members)))
    random.Random(seed).shuffle(order)
    k = max(1, math.ceil(len(members) * density))
    if cap is not None:
        k = min(k, cap)
    picked = sorted(order[:k])
    return [members[i] for i in picked]


def iter_dense_payloads(
    tar_path: str,
    members: Sequence[tuple[int, int]],
    mode: str,
    gap_bytes: int,
    max_run_bytes: int,
    stats: dict[str, int] | None = None,
    stop_event: threading.Event | None = None,
    chunk_bytes: int = 8 * MIB,
) -> Iterator[tuple[int, bytes]]:
    """密集扫描的统一实现，yield (expected_size, payload_bytes)。members 必须按 offset 升序。

    mode="seq"：从头顺序读（间隙用 read 丢弃、绝不 seek），大块 read + 按 index 切 member。
    mode="coalesce"：offset 排序后把间隙 < gap_bytes 的 member 合并为一个 ranged read
      （单块 ≤ max_run_bytes，超限在 member 边界切开），seek+大块读回后内存切分。
    stats（可选）就地累计 bytes_read / runs。T1 计数与 T4 取图共用本函数。
    """
    def bump(key: str, value: int) -> None:
        if stats is not None:
            stats[key] = stats.get(key, 0) + value

    if mode == "seq":
        with open(tar_path, "rb", buffering=0) as f:
            pos = 0
            bump("runs", 1)
            for off, size in members:
                if stop_event is not None and stop_event.is_set():
                    return
                gap = off - pos
                while gap > 0:
                    block = f.read(min(chunk_bytes, gap))
                    if not block:
                        raise OSError(f"eof while skipping gap at {pos} in {tar_path}")
                    bump("bytes_read", len(block))
                    gap -= len(block)
                need = size
                parts: list[bytes] = []
                while need > 0:
                    block = f.read(min(chunk_bytes, need))
                    if not block:
                        break
                    parts.append(block)
                    need -= len(block)
                payload = b"".join(parts)
                bump("bytes_read", len(payload))
                pos = off + size
                yield size, payload
        return

    if mode != "coalesce":
        raise ValueError(f"unknown dense scan mode: {mode}")

    runs: list[tuple[int, int, list[tuple[int, int]]]] = []  # (start, end, members)
    cur_start = cur_end = -1
    cur_members: list[tuple[int, int]] = []
    for off, size in members:
        end = off + size
        if cur_members and off - cur_end < gap_bytes and end - cur_start <= max_run_bytes:
            cur_end = max(cur_end, end)
            cur_members.append((off, size))
        else:
            if cur_members:
                runs.append((cur_start, cur_end, cur_members))
            cur_start, cur_end, cur_members = off, end, [(off, size)]
    if cur_members:
        runs.append((cur_start, cur_end, cur_members))

    with open(tar_path, "rb", buffering=0) as f:
        for start, end, run_members in runs:
            if stop_event is not None and stop_event.is_set():
                return
            bump("runs", 1)
            f.seek(start)
            need = end - start
            parts = []
            while need > 0:
                block = f.read(min(chunk_bytes, need))
                if not block:
                    break
                parts.append(block)
                need -= len(block)
            buf = b"".join(parts)
            bump("bytes_read", len(buf))
            for off, size in run_members:
                lo = off - start
                yield size, buf[lo:lo + size]


def dense_scan_tar(
    tar_path: str, members: list[tuple[int, int]], mode: str, gap_bytes: int, max_run_bytes: int
) -> dict[str, Any]:
    """单 tar 单遍密集扫描（不 decode——T1 与 Phase-2 S1 同口径，纯 IO 测量）。"""
    stats: dict[str, int] = {}
    ok = short = 0
    payload_bytes = 0
    started = time.perf_counter()
    for size, payload in iter_dense_payloads(tar_path, members, mode, gap_bytes, max_run_bytes, stats):
        if len(payload) == size:
            ok += 1
            payload_bytes += size
        else:
            short += 1
    seconds = time.perf_counter() - started
    return {
        "tar_path": tar_path,
        "mode": mode,
        "wanted": len(members),
        "ok": ok,
        "short_reads": short,
        "payload_bytes": payload_bytes,
        "bytes_read": stats.get("bytes_read", 0),
        "runs": stats.get("runs", 0),
        "seconds": seconds,
    }


def t1_level(
    mode: str,
    tar_jobs: list[tuple[str, list[tuple[int, int]]]],
    concurrency: int,
    gap_bytes: int,
    max_run_bytes: int,
) -> dict[str, Any]:
    """一个 (mode, density, concurrency) 档位：tar 级线程并发跑全部 tar_jobs。"""
    started = time.perf_counter()
    per_tar: list[dict[str, Any]] = []
    actual = max(1, min(concurrency, len(tar_jobs)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual) as pool:
        futures = [
            pool.submit(dense_scan_tar, tp, ms, mode, gap_bytes, max_run_bytes)
            for tp, ms in tar_jobs
        ]
        for fut in concurrent.futures.as_completed(futures):
            per_tar.append(fut.result())
    wall = time.perf_counter() - started
    ok = sum(r["ok"] for r in per_tar)
    wanted = sum(r["wanted"] for r in per_tar)
    bytes_read = sum(r["bytes_read"] for r in per_tar)
    payload = sum(r["payload_bytes"] for r in per_tar)
    return {
        "mode": mode,
        "requested_concurrency": concurrency,
        "actual_concurrency": actual,
        "n_tars": len(tar_jobs),
        "wanted": wanted,
        "ok": ok,
        "short_reads": sum(r["short_reads"] for r in per_tar),
        "wall_seconds": wall,
        "wanted_img_s": ok / wall if wall else 0.0,
        "read_mib_s": (bytes_read / MIB) / wall if wall else 0.0,
        "bytes_read": bytes_read,
        "payload_bytes": payload,
        "read_amplification": bytes_read / payload if payload else None,
        "runs": sum(r["runs"] for r in per_tar),
        "per_tar": sorted(per_tar, key=lambda r: r["tar_path"]),
    }


def task_t1(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    incomplete: list[str] = []
    hard_errors: list[str] = []
    entries, index_meta = prepare_indexes(args, output_dir, incomplete)
    if not entries:
        hard_errors.append("no_tar_index_available")
        return {
            "task": "T1", "verdict": "FAIL", "scope": "dense_clustered_tar_read_shootout",
            "index_meta": index_meta, "incomplete_steps": incomplete, "hard_errors": hard_errors,
            "key_numbers": {"exception": "no_index"},
        }

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in ("seq", "coalesce"):
        results[mode] = {}
        for density in T1_DENSITIES:
            dkey = f"{density:.2f}"
            results[mode][dkey] = {}
            tar_jobs = [
                (
                    e["tar_path"],
                    density_subset(
                        e["members"], density, p2.stable_seed(args.seed, e["tar_path"]), cap=args.sample_n
                    ),
                )
                for e in entries
            ]
            for conc in T1_TAR_CONCURRENCIES:
                label = f"T1_{mode}_d{int(density * 100)}_c{conc}"
                step = p2.run_step(
                    output_dir, label, t1_level,
                    (mode, tar_jobs, conc, T1_COALESCE_GAP_BYTES, T1_MAX_RUN_BYTES),
                    p2.cap_timeout(args, 900.0),
                )
                if not step.get("ok"):
                    incomplete.append(f"{label}:{step.get('error', 'timeout')}")
                    continue
                level = step["result"]
                results[mode][dkey][str(conc)] = level
                if level["short_reads"]:
                    hard_errors.append(f"{label}:short_reads={level['short_reads']}")

    dkey30 = f"{T1_DECISION_DENSITY:.2f}"
    best30 = {
        mode: max((lv["wanted_img_s"] for lv in results[mode].get(dkey30, {}).values()), default=0.0)
        for mode in ("seq", "coalesce")
    }
    if hard_errors:
        verdict = "FAIL"
        decision_reason = "hard_errors_present"
    elif best30["seq"] <= 0.0 and best30["coalesce"] <= 0.0:
        verdict = "INCOMPLETE"
        decision_reason = "no_measurement_at_decision_density"
    elif max(best30.values()) >= T1_GATE_IMG_S:
        winner = "seq" if best30["seq"] >= best30["coalesce"] else "coalesce"
        verdict = "DECISIVE_DENSE_SEQ" if winner == "seq" else "DECISIVE_DENSE_COALESCE"
        decision_reason = (
            f"density30 seq={best30['seq']:.1f} coalesce={best30['coalesce']:.1f} "
            f"winner={winner} >= gate {T1_GATE_IMG_S}"
        )
    else:
        verdict = "DECISIVE_INCONCLUSIVE"
        decision_reason = (
            f"density30 seq={best30['seq']:.1f} coalesce={best30['coalesce']:.1f} "
            f"both < gate {T1_GATE_IMG_S}; curves recorded for manual review"
        )

    return {
        "task": "T1",
        "verdict": verdict,
        "scope": "dense_clustered_tar_read_shootout_seq_vs_coalesce",
        "index_meta": index_meta,
        "tars": [
            {k: e[k] for k in ("tar_path", "domain", "sub_source", "n_members", "payload_bytes", "tar_bytes")}
            for e in entries
        ],
        "density_semantics": "seeded-shuffle prefix per tar (nested random subsets across densities)",
        "cache_caveat": "同 tar 在多个档位间重复读取，后测档位可能被 FUSE/kernel cache 加热（Phase-1 同口径的 first-touch 代理）",
        "densities": list(T1_DENSITIES),
        "concurrencies": list(T1_TAR_CONCURRENCIES),
        "coalesce_gap_bytes": T1_COALESCE_GAP_BYTES,
        "max_run_bytes": T1_MAX_RUN_BYTES,
        "results": results,
        "decision": {
            "recommendation": verdict,
            "reason": decision_reason,
            "density30_best_img_s": best30,
            "gate_img_s": T1_GATE_IMG_S,
            "gate_basis": "3.8x of single-node monet+synth demand ~106 img/s ((187+24)/2)",
        },
        "incomplete_steps": incomplete,
        "hard_errors": hard_errors,
        "key_numbers": {
            "seq30_img_s": round(best30["seq"], 1),
            "coalesce30_img_s": round(best30["coalesce"], 1),
            "decision": verdict,
        },
    }


# --------------------------------------------------------------------------- #
# T2 — gpic 深预取管线（S3 → /local → 消费）
# --------------------------------------------------------------------------- #
def ensure_local_scratch(args: argparse.Namespace) -> tuple[Path | None, str | None]:
    """/local scratch 可用性检查；不可用返回 (None, reason) → 调用方记 INCOMPLETE 而非 FAIL。"""
    root = Path(args.local_scratch)
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f".probe.{os.getpid()}"
        with probe.open("wb") as f:
            f.write(b"ok")
        probe.unlink()
        return root, None
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def parallel_resolve_gpic(
    records: Sequence[dict[str, Any]], workers: int = 16
) -> tuple[list[tuple[dict[str, Any], str]], list[dict[str, Any]]]:
    """p2.resolve_gpic_ext 的按 shard 分组线程并行版（每组独立 caption ParquetFile，线程安全；
    20k 项 ~数千 shard 串行解析太慢）。"""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[(str(rec["src_root"]), str(rec["src_shard"]))].append(rec)
    resolved: list[tuple[dict[str, Any], str]] = []
    errors: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(p2.resolve_gpic_ext, rows) for rows in grouped.values()]
        for fut in concurrent.futures.as_completed(futures):
            got, errs = fut.result()
            resolved.extend(got)
            errors.extend(errs)
    return resolved, errors


def gpic_item_pool(output_dir: Path, args: argparse.Namespace, n: int) -> tuple[list[dict[str, Any]], int]:
    """采样+解析 n 个 gpic 项（uid/src_root/src_shard/src_key/ext），结果缓存在 output_dir 下
    供 T2/T4 共用（解析成本 ~分钟级，不重复付）。"""
    cache = output_dir / "gpic_items_cache.json"
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            items = data.get("items") or []
            if len(items) >= min(n, 1000):
                return items[:n], int(data.get("resolve_errors", 0))
        except Exception:  # noqa: BLE001 - corrupt cache is rebuilt below
            pass
    candidates = p2.gpic_manifest_sample(int(n * 1.25) + 200, args.seed + 4004)
    resolved, errors = parallel_resolve_gpic(candidates)
    rng = random.Random(args.seed + 4104)
    rng.shuffle(resolved)
    items = [
        {
            "uid": str(rec["uid"]),
            "src_root": str(rec["src_root"]),
            "src_shard": str(rec["src_shard"]),
            "src_key": str(rec["src_key"]),
            "ext": ext,
        }
        for rec, ext in resolved[:n]
    ]
    p2.atomic_json(cache, {"items": items, "resolve_errors": len(errors)})
    return items, len(errors)


def _t2_prefetch_proc(
    items: list[dict[str, Any]],
    threads: int,
    local_dir_s: str,
    deadline_ts: float,
    q: Any,
    cycle: bool,
) -> None:
    """一个预取进程：threads 个下载线程共享游标，S3 读 → /local 原子写 → 完成事件入 mp.Queue。"""
    state = {"i": 0}
    lock = threading.Lock()

    def loop() -> None:
        while time.time() < deadline_ts:
            with lock:
                i = state["i"]
                state["i"] += 1
            if i >= len(items):
                if not cycle:
                    return
                i %= len(items)
            it = items[i]
            raw = read_gpic_bytes(it["src_root"], it["src_shard"], it["src_key"], ext=it["ext"])
            ts = time.time()
            if not raw:
                q.put((ts, "dl_fail", 0, ""))
                continue
            name = it["uid"].replace(":", "_").replace("/", "_") + it["ext"]
            path = os.path.join(local_dir_s, name)
            tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
            try:
                with open(tmp, "wb") as f:
                    f.write(raw)
                os.replace(tmp, path)
                q.put((ts, "ok", len(raw), path))
            except OSError:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                q.put((ts, "write_fail", 0, ""))

    pool = [threading.Thread(target=loop, daemon=True) for _ in range(threads)]
    for th in pool:
        th.start()
    for th in pool:
        th.join(max(0.0, deadline_ts - time.time()) + 5.0)


def t2_pipeline(
    items: list[dict[str, Any]],
    n_procs: int,
    threads_per_proc: int,
    target_img_s: float,
    duration_s: float,
    warmup_s: float,
    local_dir_s: str,
    batch_size: int,
    cycle: bool,
) -> dict[str, Any]:
    """一种预取布局的完整管线（在 run_step fork 子进程中运行）：
    P 个预取进程 × C 线程 → /local；消费者按 target 速率从 /local 读 + decode。"""
    local_dir = Path(local_dir_s)
    local_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    q: Any = ctx.Queue()
    start = time.time()
    deadline = start + duration_s
    slices = [items[i::n_procs] for i in range(n_procs)]
    procs = [
        ctx.Process(target=_t2_prefetch_proc, args=(sl, threads_per_proc, local_dir_s, deadline, q, cycle))
        for sl in slices if sl
    ]
    for p in procs:
        p.start()

    lock = threading.Lock()
    ready: deque[str] = deque()
    dl_ts: list[float] = []
    counters = {"ok": 0, "dl_fail": 0, "write_fail": 0, "bytes": 0}
    unique_files: dict[str, int] = {}
    drain_stop = threading.Event()

    def drainer() -> None:
        while not drain_stop.is_set() or not q.empty():
            try:
                ts, kind, nbytes, path = q.get(timeout=0.2)
            except queue.Empty:
                continue
            with lock:
                if kind == "ok":
                    counters["ok"] += 1
                    counters["bytes"] += nbytes
                    dl_ts.append(ts)
                    unique_files[path] = nbytes
                    ready.append(path)
                else:
                    counters[kind] += 1

    drain_thread = threading.Thread(target=drainer, daemon=True)
    drain_thread.start()

    interval = 1.0 / target_img_s
    consumed = 0
    decode_fail = 0
    read_fail = 0
    stall_seconds = 0.0
    consume_latencies: list[float] = []
    batch_ready: list[float] = []
    backlog_samples: list[dict[str, float]] = []
    last_batch_mark = time.time()
    last_backlog_sample = 0.0
    next_due = start
    while time.time() < deadline:
        now = time.time()
        if now - last_backlog_sample >= 2.0 and len(backlog_samples) < 900:
            with lock:
                backlog_samples.append({
                    "t": round(now - start, 1),
                    "backlog": counters["ok"] - consumed,
                    "ready_queue": len(ready),
                })
            last_backlog_sample = now
        if now < next_due:
            time.sleep(min(0.05, next_due - now))
            continue
        with lock:
            path = ready.popleft() if ready else None
        if path is None:
            stall_seconds += 0.02
            time.sleep(0.02)
            continue
        t0 = time.perf_counter()
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            read_fail += 1
            next_due += interval
            continue
        img = decode_rgb(raw)
        consume_latencies.append(time.perf_counter() - t0)
        if img is None:
            decode_fail += 1
        consumed += 1
        if consumed % batch_size == 0:
            now2 = time.time()
            batch_ready.append(now2 - last_batch_mark)
            last_batch_mark = now2
        next_due += interval

    for p in procs:
        p.join(15)
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(10)
    time.sleep(0.5)  # 让 drainer 收掉队尾事件
    drain_stop.set()
    drain_thread.join(10)

    wall = time.time() - start
    steady_window = max(1.0, duration_s - warmup_s)
    steady_downloads = sum(1 for ts in dl_ts if ts >= start + warmup_s)
    total_bytes_on_disk = sum(unique_files.values())
    shutil.rmtree(local_dir, ignore_errors=True)  # 每个布局测完即清，峰值 /local 用量 ~单布局体量
    return {
        "n_procs": n_procs,
        "threads_per_proc": threads_per_proc,
        "total_prefetch_concurrency": n_procs * threads_per_proc,
        "item_pool": len(items),
        "cycled": cycle,
        "duration_seconds": wall,
        "prefetch": {
            "ok": counters["ok"],
            "dl_fail": counters["dl_fail"],
            "write_fail": counters["write_fail"],
            "overall_img_s": counters["ok"] / wall if wall else 0.0,
            "steady_img_s": steady_downloads / steady_window,
            "steady_window_seconds": steady_window,
            "mib_s": (counters["bytes"] / MIB) / wall if wall else 0.0,
            "unique_files": len(unique_files),
            "unique_bytes_on_local": total_bytes_on_disk,
        },
        "consumer": {
            "target_img_s": target_img_s,
            "consumed": consumed,
            "achieved_img_s": consumed / wall if wall else 0.0,
            "decode_fail": decode_fail,
            "read_fail": read_fail,
            "stall_seconds_approx": stall_seconds,
            "read_decode_latency_ms": p2.latency_stats_ms(consume_latencies),
            "batch_size": batch_size,
            "batches_completed": len(batch_ready),
            "batch_ready_ms": p2.latency_stats_ms(batch_ready),
        },
        "backlog_curve": backlog_samples,
    }


def task_t2(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    incomplete: list[str] = []
    hard_errors: list[str] = []
    scratch, reason = ensure_local_scratch(args)
    if scratch is None:
        return {
            "task": "T2", "verdict": "INCOMPLETE", "scope": "gpic_deep_prefetch_pipeline",
            "reason": f"local_scratch_unavailable: {reason}",
            "incomplete_steps": [f"local_scratch:{reason}"], "hard_errors": [],
            "key_numbers": {"local_scratch": "unavailable"},
        }
    n_items = args.t2_items
    items, resolve_errors = gpic_item_pool(output_dir, args, n_items)
    if len(items) < n_items:
        incomplete.append(f"item_pool_short:{len(items)}_of_{n_items}")
    if not items:
        hard_errors.append("gpic_item_pool_empty")
        return {
            "task": "T2", "verdict": "FAIL", "scope": "gpic_deep_prefetch_pipeline",
            "incomplete_steps": incomplete, "hard_errors": hard_errors,
            "key_numbers": {"exception": "empty_pool"},
        }
    # 两种布局用不相交的半池，避免第二种布局吃到第一种的 warm cache 偏差；cycle 保证测满时长。
    half = max(1, len(items) // 2)
    config_items = {"p1x32": items[:half], "p2x16": items[half:] or items[:half]}

    configs_result: dict[str, Any] = {}
    for name, n_procs, threads in T2_CONFIGS:
        step = p2.run_step(
            output_dir, f"T2_{name}", t2_pipeline,
            (
                config_items[name], n_procs, threads, args.t2_target_img_s, args.t2_duration_s,
                min(30.0, args.t2_duration_s / 4), str(scratch / "t2" / name),
                p2.BATCH_SIZE_PER_RANK, True,
            ),
            min(args.step_timeout_s, args.t2_duration_s + 240.0),
        )
        if not step.get("ok"):
            incomplete.append(f"{name}:{step.get('error', 'timeout')}")
            continue
        configs_result[name] = step["result"]

    for name, r in configs_result.items():
        pf = r["prefetch"]
        attempts = pf["ok"] + pf["dl_fail"]
        if attempts and pf["dl_fail"] / attempts > 0.05:
            hard_errors.append(f"{name}:download_fail_rate_over_5pct")
        if pf["ok"] and pf["write_fail"] / max(pf["ok"] + pf["write_fail"], 1) > 0.01:
            hard_errors.append(f"{name}:local_write_fail_rate_over_1pct")

    best_steady = max((r["prefetch"]["steady_img_s"] for r in configs_result.values()), default=0.0)
    best_config = max(
        configs_result, key=lambda k: configs_result[k]["prefetch"]["steady_img_s"], default=None
    )
    if hard_errors:
        verdict = "FAIL"
    elif not configs_result:
        verdict = "INCOMPLETE"
    elif best_steady >= T2_GATE_IMG_S:
        verdict = "PASS"
    else:
        verdict = "WARN"

    return {
        "task": "T2",
        "verdict": verdict,
        "scope": "gpic_deep_prefetch_pipeline_s3_to_local",
        "local_scratch": str(scratch),
        "item_pool": len(items),
        "resolve_errors": resolve_errors,
        "configs": configs_result,
        "best_config": best_config,
        "best_steady_img_s": best_steady,
        "thresholds": {
            "prefetch_steady_img_s_min": T2_GATE_IMG_S,
            "consumer_target_img_s": args.t2_target_img_s,
            "basis": "1.5x of gpic single-node demand 59.5 img/s (=119/2)",
        },
        "incomplete_steps": incomplete,
        "hard_errors": hard_errors,
        "key_numbers": {
            "best_config": best_config,
            "steady_img_s": round(best_steady, 1),
            "gate": T2_GATE_IMG_S,
        },
    }


# --------------------------------------------------------------------------- #
# T3 — textatlas 补测（并发修正 + /local staging 分段耗时）
# --------------------------------------------------------------------------- #
def t3_rg_read(path: str, rg: int) -> tuple[bool, float, float, int]:
    """一次 (shard,row_group) 读：独立 ParquetFile 实例、无全局锁（修正 Phase-2 S3 的
    PARQUET_NATIVE_LOCK 串行化测量缺陷；跨线程不共享 native 对象即线程安全）。"""
    t0 = time.perf_counter()
    try:
        pf = pq.ParquetFile(path)
        footer_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        table = pf.read_row_group(rg, columns=["image"], use_threads=False)
        read_s = time.perf_counter() - t1
        return table.num_rows > 0, footer_s, read_s, table.num_rows
    except Exception:  # noqa: BLE001 - a damaged row group is itself observable
        return False, time.perf_counter() - t0, 0.0, 0


def t3_level(picks: list[tuple[str, int]], concurrency: int) -> dict[str, Any]:
    started = time.perf_counter()
    footer_lat: list[float] = []
    read_lat: list[float] = []
    ok = 0
    rows = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(t3_rg_read, p, rg) for p, rg in picks]
        for fut in concurrent.futures.as_completed(futures):
            good, footer_s, read_s, n = fut.result()
            footer_lat.append(footer_s)
            read_lat.append(read_s)
            if good:
                ok += 1
                rows += n
    wall = time.perf_counter() - started
    return {
        "concurrency": concurrency,
        "n_row_groups": len(picks),
        "ok_row_groups": ok,
        "rows_read": rows,
        "wall_seconds": wall,
        "img_s": rows / wall if wall else 0.0,
        "row_groups_per_s": ok / wall if wall else 0.0,
        "footer_open_ms": p2.latency_stats_ms(footer_lat),
        "row_group_read_ms": p2.latency_stats_ms(read_lat),
    }


def t3_staging_probe(src_path_s: str, local_dir_s: str, n_rg: int, seed: int) -> dict[str, Any]:
    """「整 shard 拷贝到 /local + 本地读」分段耗时：copy(顺序 S3 读+本地写) / 本地 footer /
    本地 row-group 读+decode。拷贝用 shutil.copyfileobj 分块（与 prestage_textatlas.py 同构，
    委托书写 rsync，此处用等价的 python 拷贝以免依赖外部二进制）。"""
    local_dir = Path(local_dir_s)
    local_dir.mkdir(parents=True, exist_ok=True)
    dst = local_dir / "staged_shard.parquet"
    src_bytes = os.stat(src_path_s).st_size
    tmp = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
    t0 = time.perf_counter()
    try:
        with open(src_path_s, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=8 * MIB)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    copy_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pf = pq.ParquetFile(dst)
    footer_s = time.perf_counter() - t1
    n_groups = pf.metadata.num_row_groups
    rng = random.Random(seed)
    rgs = rng.sample(range(n_groups), min(n_rg, n_groups))
    rows = 0
    decode_ok = 0
    read_lat: list[float] = []
    t2 = time.perf_counter()
    for rg in rgs:
        t3 = time.perf_counter()
        table = pf.read_row_group(rg, columns=["image"], use_threads=False)
        read_lat.append(time.perf_counter() - t3)
        rows += table.num_rows
        image = table.column("image")[0].as_py()
        raw = image.get("bytes") if isinstance(image, dict) else None
        if raw and decode_rgb(raw) is not None:
            decode_ok += 1
    local_read_s = time.perf_counter() - t2
    try:
        dst.unlink()
    except OSError:
        pass
    return {
        "src_path": src_path_s,
        "src_bytes": src_bytes,
        "copy_seconds": copy_s,
        "copy_mib_s": (src_bytes / MIB) / copy_s if copy_s else None,
        "local_footer_open_seconds": footer_s,
        "row_groups_total": n_groups,
        "row_groups_read": len(rgs),
        "rows_read": rows,
        "first_item_decode_ok": decode_ok,
        "local_read_seconds": local_read_s,
        "local_img_s": rows / local_read_s if local_read_s else 0.0,
        "local_row_group_read_ms": p2.latency_stats_ms(read_lat),
    }


def task_t3(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    incomplete: list[str] = []
    hard_errors: list[str] = []
    n_picks = args.sample_n if args.sample_n is not None else 100
    files = p2.textatlas_source_files(p2.TEXTATLAS_ACTIVE_SUBSETS)
    if not files:
        return {
            "task": "T3", "verdict": "INCOMPLETE", "scope": "textatlas_retest",
            "reason": "no_textatlas_source_files_found",
            "incomplete_steps": ["no_source_files"], "hard_errors": [],
            "key_numbers": {"files": 0},
        }
    picks = p2.sample_row_groups(files, n_picks, args.seed)
    if len(picks) < n_picks:
        incomplete.append(f"row_group_picks_short:{len(picks)}_of_{n_picks}")

    levels_result: dict[str, Any] = {}
    for level in (1, 8):
        step = p2.run_step(
            output_dir, f"T3_level_{level}", t3_level, (picks, level), p2.cap_timeout(args, 650.0),
        )
        if not step.get("ok"):
            incomplete.append(f"level_{level}:{step.get('error', 'timeout')}")
            continue
        levels_result[str(level)] = step["result"]

    for lvl, r in levels_result.items():
        if r["n_row_groups"] and (r["n_row_groups"] - r["ok_row_groups"]) / r["n_row_groups"] > 0.05:
            hard_errors.append(f"level_{lvl}:error_rate_over_5pct")

    best_img_s = max((r["img_s"] for r in levels_result.values()), default=0.0)
    best_level = max(levels_result, key=lambda k: levels_result[k]["img_s"], default=None)

    # 直读慢/没测出来 → 附测 /local staging 分段耗时（生产方案的数据）
    staging_result: dict[str, Any] | None = None
    staging_error: str | None = None
    if best_img_s < T3_GATE_IMG_S or not levels_result:
        scratch, reason = ensure_local_scratch(args)
        if scratch is None:
            incomplete.append(f"staging:local_scratch_unavailable:{reason}")
        else:
            unique_paths = sorted({p for p, _rg in picks}) or [str(files[0])]
            sized: list[tuple[int, str]] = []
            for p in unique_paths[:20]:
                try:
                    sized.append((os.stat(p).st_size, p))
                except OSError:
                    continue
            if sized:
                src = min(sized)[1]  # 选最小的 shard 拷贝，界定 staging 探测耗时
                step = p2.run_step(
                    output_dir, "T3_staging", t3_staging_probe,
                    (src, str(scratch / "t3"), 20, args.seed + 33), p2.cap_timeout(args, 600.0),
                )
                if step.get("ok"):
                    staging_result = step["result"]
                else:
                    staging_error = str(step.get("error", "timeout"))
                    incomplete.append(f"staging:{staging_error}")

    # 委托口径：T3 产出无论如何不判 FAIL（生产走 /local staging，S3 只是回退）。
    if levels_result and best_img_s >= T3_GATE_IMG_S:
        verdict = "PASS"
    elif levels_result or staging_result:
        verdict = "WARN"
    else:
        verdict = "INCOMPLETE"

    return {
        "task": "T3",
        "verdict": verdict,
        "scope": "textatlas_retest_concurrency_fix_plus_local_staging",
        "measurement_fix": "Phase-2 S3 把 read_row_group 包进全局 PARQUET_NATIVE_LOCK,并发档被串行化;本次每 task 独立 ParquetFile、无全局锁",
        "n_source_files": len(files),
        "concurrency_levels": levels_result,
        "best_concurrency": best_level,
        "best_img_s": best_img_s,
        "staging_probe": staging_result,
        "staging_error": staging_error,
        "thresholds": {
            "domain_requirement_img_s": p2.DOMAIN_GLOBAL_IMG_S["textatlas"],
            "pass_img_s_min": T3_GATE_IMG_S,
            "never_fail": "production plan is /local staging; S3 direct read is only the fallback",
        },
        "incomplete_steps": incomplete,
        "hard_errors": hard_errors,
        "key_numbers": {
            "best_concurrency": best_level,
            "best_img_s": round(best_img_s, 1),
            "staging_copy_mib_s": (
                round(staging_result["copy_mib_s"], 1)
                if staging_result and staging_result.get("copy_mib_s") else None
            ),
        },
    }


# --------------------------------------------------------------------------- #
# T4 — 混合 gate 复测（S5b）。
#
# 与 Phase-2 S5 的差异：monet/synth 用 T1 胜者密集聚簇扫描（per-worker 分 shard + 密度窗口，
# 模拟 plan 窗口）；gpic 用 T2 深预取（预取线程 S3→/local，读取线程 /local→decode，深度
# ready 队列）；textatlas 可用 7 img/s 常数流模拟（显式标注）。生产者对 domain 队列改为
# 【阻塞 put】（S5 的丢弃语义会浪费已 decode 的密集扫描产出），消费侧混合语义与 S5 相同：
# 按 domain_mix 加权选域、阻塞取 → 任一域断供即拖慢整机。
# --------------------------------------------------------------------------- #
class _QueueSource:
    """T4 域生产者基类：有界队列 + 阻塞 put（背压）+ stop 事件。接口同 p2.DomainSource。"""

    def __init__(self, maxsize: int = 256) -> None:
        self.q: "queue.Queue[tuple[bool, float]]" = queue.Queue(maxsize=maxsize)
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []

    def _spawn(self, fn: Callable[..., None], *args: Any) -> None:
        th = threading.Thread(target=fn, args=args, daemon=True)
        th.start()
        self.threads.append(th)

    def _put(self, item: tuple[bool, float]) -> bool:
        while not self.stop_event.is_set():
            try:
                self.q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def get(self, timeout: float = 2.0) -> tuple[bool, float] | None:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self.stop_event.set()

    def extra_stats(self) -> dict[str, Any]:
        return {}


class DenseTarSource(_QueueSource):
    """monet/synth 密集聚簇源：每个分到的 tar 一个扫描线程，循环整遍扫描（T1 同一实现），
    每个 wanted member decode 后入队。dt = 与上一 item 的间隔（批量扫描无单 item 延迟语义）。"""

    def __init__(
        self, tar_jobs: list[tuple[str, list[tuple[int, int]]]], mode: str,
        gap_bytes: int, max_run_bytes: int,
    ) -> None:
        super().__init__(maxsize=256)
        self.passes = 0
        self.scan_errors = 0
        self._lock = threading.Lock()
        for tar_path, members in tar_jobs:
            self._spawn(self._scan_loop, tar_path, members, mode, gap_bytes, max_run_bytes)

    def _scan_loop(
        self, tar_path: str, members: list[tuple[int, int]], mode: str,
        gap_bytes: int, max_run_bytes: int,
    ) -> None:
        while not self.stop_event.is_set():
            last = time.perf_counter()
            try:
                for size, payload in iter_dense_payloads(
                    tar_path, members, mode, gap_bytes, max_run_bytes, stop_event=self.stop_event
                ):
                    ok = len(payload) == size and decode_rgb(payload) is not None
                    now = time.perf_counter()
                    if not self._put((ok, now - last)):
                        return
                    last = now
                with self._lock:
                    self.passes += 1
            except Exception:  # noqa: BLE001 - a broken scan pass is data, not a crash
                with self._lock:
                    self.scan_errors += 1
                if not self._put((False, 0.0)):
                    return
                time.sleep(0.5)

    def extra_stats(self) -> dict[str, Any]:
        with self._lock:
            return {"scan_passes": self.passes, "scan_errors": self.scan_errors}


class GpicPrefetchSource(_QueueSource):
    """gpic 深预取源：prefetch 线程 S3→/local（原子写）→ ready 深队列；reader 线程从 /local
    读 + decode 入 domain 队列。ready 深度上限即「任意深度预取」的工程化边界。"""

    def __init__(
        self, items: list[dict[str, Any]], prefetch_threads: int, reader_threads: int,
        local_dir_s: str, ready_depth: int = 4096,
    ) -> None:
        super().__init__(maxsize=256)
        self.local_dir = Path(local_dir_s)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.ready: "queue.Queue[str]" = queue.Queue(maxsize=ready_depth)
        self._items = items
        self._idx = 0
        self._lock = threading.Lock()
        self.downloaded = 0
        self.dl_fail = 0
        self.write_fail = 0
        for _ in range(prefetch_threads):
            self._spawn(self._prefetch_loop)
        for _ in range(reader_threads):
            self._spawn(self._read_loop)

    def _next_item(self) -> dict[str, Any]:
        with self._lock:
            i = self._idx
            self._idx += 1
        return self._items[i % len(self._items)]

    def _prefetch_loop(self) -> None:
        while not self.stop_event.is_set():
            it = self._next_item()
            raw = read_gpic_bytes(it["src_root"], it["src_shard"], it["src_key"], ext=it["ext"])
            if not raw:
                with self._lock:
                    self.dl_fail += 1
                continue
            name = it["uid"].replace(":", "_").replace("/", "_") + it["ext"]
            path = str(self.local_dir / name)
            tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
            try:
                with open(tmp, "wb") as f:
                    f.write(raw)
                os.replace(tmp, path)
            except OSError:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                with self._lock:
                    self.write_fail += 1
                continue
            with self._lock:
                self.downloaded += 1
            while not self.stop_event.is_set():
                try:
                    self.ready.put(path, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def _read_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                path = self.ready.get(timeout=0.5)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            try:
                with open(path, "rb") as f:
                    raw = f.read()
                ok = decode_rgb(raw) is not None
            except OSError:
                ok = False
            if not self._put((ok, time.perf_counter() - t0)):
                return

    def extra_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "downloaded": self.downloaded,
                "dl_fail": self.dl_fail,
                "write_fail": self.write_fail,
                "ready_backlog_at_close": self.ready.qsize(),
            }


class FetchSource(_QueueSource):
    """逐 item fetch 源（anime CDN / textatlas real）：p2.DomainSource 的阻塞 put 版。"""

    def __init__(self, concurrency: int, iter_factory: Callable[[], Any], process_fn: Callable[[Any], bool]) -> None:
        super().__init__(maxsize=max(64, concurrency * 8))
        for _ in range(concurrency):
            self._spawn(self._loop, iter_factory(), process_fn)

    def _loop(self, it: Any, process_fn: Callable[[Any], bool]) -> None:
        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            try:
                item = next(it)
                ok = bool(process_fn(item))
            except StopIteration:
                ok = False
                time.sleep(0.05)
            except Exception:  # noqa: BLE001 - a fetch/decode failure is data, not a crash
                ok = False
            if not self._put((ok, time.perf_counter() - t0)):
                return


class PacedSimSource(_QueueSource):
    """常数速率模拟源（textatlas 7 img/s 兜底；结果里显式标注 simulated，不做任何 IO/decode）。"""

    def __init__(self, rate_img_s: float) -> None:
        super().__init__(maxsize=64)
        self.rate_img_s = rate_img_s
        self._spawn(self._loop)

    def _loop(self) -> None:
        interval = 1.0 / self.rate_img_s
        while not self.stop_event.is_set():
            time.sleep(interval)
            if not self._put((True, interval)):
                return


def _t4_textatlas_process(item: tuple[str, int]) -> bool:
    """textatlas real 模式的单 item：独立 ParquetFile、无全局锁（同 t3_rg_read 口径），
    读一个 row-group、decode 首行图片，计 1 item（保守口径，与 S5 相同）。"""
    path, rg = item
    try:
        pf = pq.ParquetFile(path)
        table = pf.read_row_group(rg, columns=["image"], use_threads=False)
        if table.num_rows == 0:
            return False
        image = table.column("image")[0].as_py()
        raw = image.get("bytes") if isinstance(image, dict) else None
        return bool(raw) and decode_rgb(raw) is not None
    except Exception:  # noqa: BLE001 - a damaged row group is itself observable
        return False


def _cycle_factory(pool: list[Any], seed: int) -> Callable[[], Any]:
    def factory() -> Any:
        local = list(pool)
        random.Random(seed).shuffle(local)
        i = 0
        n = len(local)

        def gen() -> Any:
            nonlocal i
            while True:
                yield local[i % n]
                i += 1
        return gen()
    return factory


def t4_worker_main(rank: int, cfg: dict[str, Any], result_path_s: str) -> None:
    try:
        sources: dict[str, _QueueSource] = {
            "monet_real": DenseTarSource(
                cfg["monet_jobs"][rank], cfg["mode"], cfg["gap_bytes"], cfg["max_run_bytes"]
            ),
            "synthetic": DenseTarSource(
                cfg["synth_jobs"][rank], cfg["mode"], cfg["gap_bytes"], cfg["max_run_bytes"]
            ),
            "gpic": GpicPrefetchSource(
                cfg["gpic_items"][rank], cfg["gpic_threads"], cfg["gpic_readers"],
                os.path.join(cfg["gpic_local_root"], f"rank{rank}"),
            ),
            "anime": FetchSource(
                cfg["anime_threads"],
                p2._anime_iter_factory(cfg["anime_records"][rank], cfg["seed"] + 900 + rank),
                p2._anime_process,
            ),
        }
        if cfg["textatlas_mode"] == "real":
            sources["textatlas"] = FetchSource(
                cfg["textatlas_threads"],
                _cycle_factory(cfg["textatlas_picks"][rank], cfg["seed"] + 950 + rank),
                _t4_textatlas_process,
            )
        else:
            sources["textatlas"] = PacedSimSource(cfg["sim_rate_img_s"])

        time.sleep(cfg["warmup_s"])  # 给预取/首遍扫描一点起步时间，测量段更接近稳态
        domains = list(p2.DOMAIN_MIX_S5)
        weights = [p2.DOMAIN_MIX_S5[d] for d in domains]
        rng = random.Random(cfg["seed"] + rank * 7919 + 17)
        counts = {d: 0 for d in domains}
        errors = {d: 0 for d in domains}
        latencies: dict[str, list[float]] = {d: [] for d in domains}
        batch_ready: list[float] = []
        total_ok = 0
        start = time.perf_counter()
        last_batch_t = start
        deadline = start + cfg["duration_s"]
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
                    if total_ok % cfg["batch_size"] == 0:
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
                "latency_ms_by_domain": {d: p2.latency_stats_ms(v) for d, v in latencies.items()},
                "batches_completed": len(batch_ready),
                "batch_ready_ms": p2.latency_stats_ms(batch_ready),
                "source_stats": {d: s.extra_stats() for d, s in sources.items()},
                "textatlas_simulated": cfg["textatlas_mode"] != "real",
            },
        }
    except Exception as exc:  # noqa: BLE001 - a worker crash must be observable, not lost
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=20)}
    p2.atomic_json(Path(result_path_s), payload)


def read_result_json(output_dir: Path, name: str) -> dict[str, Any] | None:
    p = output_dir / name
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - unreadable sibling result -> fall back to defaults
        return None


def resolve_t4_monet_mode(output_dir: Path, args: argparse.Namespace) -> tuple[str, str]:
    if args.t4_monet_mode != "auto":
        return args.t4_monet_mode, "--t4-monet-mode override"
    data = read_result_json(output_dir, "taskT1_result.json")
    if data:
        verdict = str(data.get("verdict", ""))
        if verdict == "DECISIVE_DENSE_SEQ":
            return "seq", f"taskT1_result.json verdict={verdict}"
        if verdict == "DECISIVE_DENSE_COALESCE":
            return "coalesce", f"taskT1_result.json verdict={verdict}"
        best = (data.get("decision") or {}).get("density30_best_img_s") or {}
        if best:
            mode = "seq" if best.get("seq", 0.0) >= best.get("coalesce", 0.0) else "coalesce"
            return mode, f"taskT1_result.json inconclusive; higher density30 side={mode}"
    return "coalesce", "no taskT1_result.json; defaulting to coalesce"


def resolve_t4_textatlas(output_dir: Path, args: argparse.Namespace) -> tuple[str, int, str]:
    """返回 (mode, threads_per_worker, reason)。auto：T3 PASS → real，否则 simulated。"""
    if args.t4_textatlas_mode != "auto":
        return args.t4_textatlas_mode, 1, "--t4-textatlas-mode override"
    data = read_result_json(output_dir, "taskT3_result.json")
    if data and str(data.get("verdict")) == "PASS":
        best = data.get("best_concurrency")
        threads = max(1, round(int(best) / 8)) if best else 1
        return "real", threads, f"taskT3_result.json verdict=PASS best_concurrency={best}"
    return "simulated", 1, "taskT3_result.json absent or not PASS; simulating 7 img/s constant stream"


def task_t4(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    incomplete: list[str] = []
    hard_errors: list[str] = []

    mode, mode_reason = resolve_t4_monet_mode(output_dir, args)
    ta_mode, ta_threads, ta_reason = resolve_t4_textatlas(output_dir, args)

    scratch, scratch_reason = ensure_local_scratch(args)
    if scratch is None:
        return {
            "task": "T4", "verdict": "INCOMPLETE", "scope": "mixed_gate_retest_s5b",
            "reason": f"local_scratch_unavailable: {scratch_reason} (gpic deep prefetch premise)",
            "incomplete_steps": [f"local_scratch:{scratch_reason}"], "hard_errors": [],
            "key_numbers": {"local_scratch": "unavailable"},
        }

    entries, index_meta = prepare_indexes(args, output_dir, incomplete)
    monet_tars = [e for e in entries if e["domain"] == "monet_real"]
    synth_tars = [e for e in entries if e["domain"] == "synthetic"]
    if not monet_tars or not synth_tars:
        hard_errors.append("monet_or_synth_index_missing")
    gpic_items, _resolve_errors = gpic_item_pool(output_dir, args, args.t2_items)
    if not gpic_items:
        hard_errors.append("gpic_item_pool_empty")
    anime_pool = p2.anime_url_sample(args.t4_anime_pool_n, args.seed + 7007)
    if not anime_pool:
        hard_errors.append("anime_pool_empty")
    textatlas_picks: list[tuple[str, int]] = []
    if ta_mode == "real":
        files = p2.textatlas_source_files(p2.TEXTATLAS_ACTIVE_SUBSETS)
        textatlas_picks = p2.sample_row_groups(files, 400, args.seed + 6006)
        if not textatlas_picks:
            ta_mode, ta_reason = "simulated", ta_reason + "; no picks -> simulated"
    if hard_errors:
        return {
            "task": "T4", "verdict": "FAIL", "scope": "mixed_gate_retest_s5b",
            "index_meta": index_meta, "incomplete_steps": incomplete, "hard_errors": hard_errors,
            "key_numbers": {"exception": "pool_build_failed"},
        }

    # per-worker 分 shard（plan 窗口模拟）：monet 4 tar / synth 2 tar 轮转到 8 个 worker，
    # 每 worker 的 wanted 窗口用 rank 相关种子抽 density 子集（共享同 tar 的 worker 窗口不同）。
    monet_jobs: dict[int, list[tuple[str, list[tuple[int, int]]]]] = {}
    synth_jobs: dict[int, list[tuple[str, list[tuple[int, int]]]]] = {}
    for r in range(8):
        me = monet_tars[r % len(monet_tars)]
        se = synth_tars[r % len(synth_tars)]
        monet_jobs[r] = [(
            me["tar_path"],
            density_subset(me["members"], args.t4_density, p2.stable_seed(args.seed + 31 * r, me["tar_path"])),
        )]
        synth_jobs[r] = [(
            se["tar_path"],
            density_subset(se["members"], args.t4_density, p2.stable_seed(args.seed + 31 * r, se["tar_path"])),
        )]

    t4_local_root = str(scratch / "t4")
    cfg: dict[str, Any] = {
        "mode": mode,
        "gap_bytes": T1_COALESCE_GAP_BYTES,
        "max_run_bytes": T1_MAX_RUN_BYTES,
        "monet_jobs": monet_jobs,
        "synth_jobs": synth_jobs,
        "gpic_items": {r: gpic_items[r::8] for r in range(8)},
        "gpic_threads": args.t4_gpic_threads_per_worker,
        "gpic_readers": 2,
        "gpic_local_root": t4_local_root,
        "anime_records": {r: anime_pool[r::8] for r in range(8)},
        "anime_threads": T4_ANIME_THREADS_PER_WORKER,
        "textatlas_mode": ta_mode,
        "textatlas_threads": ta_threads,
        "textatlas_picks": {r: textatlas_picks[r::8] for r in range(8)} if ta_mode == "real" else {},
        # 委托口径「以 7 img/s 常数流模拟」按节点 7 img/s 计 → 每 worker 7/8=0.875 img/s。
        "sim_rate_img_s": p2.DOMAIN_GLOBAL_IMG_S["textatlas"] / 8.0,
        "warmup_s": args.t4_warmup_s,
        "duration_s": args.t4_duration_s,
        "batch_size": p2.BATCH_SIZE_PER_RANK,
        "seed": args.seed,
    }

    duration_s = args.t4_duration_s
    scaled = duration_s < 600.0
    arg_tuples = [(r, cfg) for r in range(8)]
    worker_payloads = p2.run_parallel_processes(
        output_dir, "T4_worker", t4_worker_main, arg_tuples,
        timeout_s=args.t4_warmup_s + duration_s + 300.0,
    )
    shutil.rmtree(t4_local_root, ignore_errors=True)

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
    simulated_share_img_s = (
        sum(w["counts"].get("textatlas", 0) for w in ok_workers)
        / max(1e-9, sum(w["wall_seconds"] for w in ok_workers) / max(len(ok_workers), 1))
        if ta_mode != "real" and ok_workers else 0.0
    )

    if hard_errors:
        verdict = "FAIL"
    elif len(ok_workers) < 8:
        verdict = "INCOMPLETE"
    elif total_img_s >= p2.NODE_GATE_IMG_S and not straggler:
        verdict = "WARN" if scaled else "PASS"
    else:
        # gate 本体（同 Phase-2 S5）：未达标是真实的"不过 → 停,找 Evan"结论,FAIL 合法。
        verdict = "FAIL"

    return {
        "task": "T4",
        "verdict": verdict,
        "scope": "mixed_gate_retest_s5b_dense_monet_deep_prefetch_gpic",
        "backend_decision": {
            "monet_synth_mode": mode, "monet_synth_reason": mode_reason,
            "monet_synth_density": args.t4_density,
            "index_meta": index_meta,
            "gpic": {
                "prefetch_threads_per_worker": args.t4_gpic_threads_per_worker,
                "reader_threads_per_worker": 2,
                "node_total_prefetch_concurrency": args.t4_gpic_threads_per_worker * 8,
                "consumes_from": "/local deep prefetch (T2 pipeline shape)",
            },
            "anime_threads_per_worker": T4_ANIME_THREADS_PER_WORKER,
            "textatlas_mode": ta_mode, "textatlas_reason": ta_reason,
        },
        "domain_mix": p2.DOMAIN_MIX_S5,
        "duration_seconds": duration_s,
        "warmup_seconds": args.t4_warmup_s,
        "scaled": scaled,
        "batch_size_per_rank": p2.BATCH_SIZE_PER_RANK,
        "workers": worker_payloads,
        "aggregate": {
            "total_img_s": total_img_s,
            "avg_worker_img_s": avg_worker,
            "min_worker_img_s": min_worker,
            "straggler_detected": straggler,
            "max_rank_batch_ready_p99_ms": max_rank_batch_ready_p99_ms,
            "textatlas_simulated": ta_mode != "real",
            "textatlas_simulated_img_s_included_in_total": (
                round(simulated_share_img_s, 2) if ta_mode != "real" else None
            ),
        },
        "thresholds": {
            "node_gate_img_s": p2.NODE_GATE_IMG_S,
            "straggler_min_over_avg": 0.7,
        },
        "incomplete_steps": incomplete,
        "hard_errors": hard_errors,
        "key_numbers": {
            "total_img_s": round(total_img_s, 1),
            "min_worker_img_s": round(min_worker, 1),
            "straggler": straggler,
            "monet_mode": mode,
            "textatlas_mode": ta_mode,
        },
    }


# --------------------------------------------------------------------------- #
# 编排（同 phase2_shootout 约定）
# --------------------------------------------------------------------------- #
TASKS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "T1": task_t1, "T2": task_t2, "T3": task_t3, "T4": task_t4,
}

DECISIVE_VERDICTS = {"DECISIVE_DENSE_SEQ", "DECISIVE_DENSE_COALESCE", "DECISIVE_INCONCLUSIVE"}


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
        result = {
            "task": task, "verdict": "FAIL", "scope": "internal_error",
            "error": f"unknown verdict token: {result.get('verdict')!r}",
            "key_numbers": {"exception": "unknown_verdict"},
        }
    result.update({
        "schema_version": "offline_arb_phase2b_v1",
        "host": socket.gethostname(),
        "seed": args.seed,
        "started_at": started_wall,
        "completed_at": utc_now(),
        "duration_seconds": time.perf_counter() - started,
    })
    output_dir = Path(args.output_dir)
    p2.atomic_json(output_dir / f"task{task}_result.json", result)
    p2.append_summary(output_dir, task, result["verdict"], result.get("key_numbers", {}))
    print(f"task={task} verdict={result['verdict']} result={output_dir / f'task{task}_result.json'}", flush=True)
    return str(result["verdict"])


def parse_tests(raw: str) -> list[str]:
    tests = [x.strip().upper() for x in raw.split(",") if x.strip()]
    invalid = [x for x in tests if x not in TASKS]
    if not tests or invalid:
        raise argparse.ArgumentTypeError(f"--test must be a comma-separated subset of T1..T4; invalid={invalid}")
    # T4 读同一 output_dir 里 T1/T3 结论自动选模式；组合运行固定按 T1->T4 顺序执行。
    requested = set(tests)
    return [t for t in ("T1", "T2", "T3", "T4") if t in requested]


def validate_output_dir(raw: str) -> Path:
    path = Path(raw).expanduser().resolve(strict=False)
    parent = OUTPUT_PARENT.resolve(strict=False)
    prefix = "offline_arb_phase2b_"
    if path.parent != parent or not path.name.startswith(prefix) or len(path.name) <= len(prefix):
        raise ValueError(
            f"--output-dir must be {OUTPUT_PARENT}/offline_arb_phase2b_${{RUN_TAG}}, got {path}"
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, type=parse_tests, help="T1..T4 or a comma-separated subset")
    parser.add_argument("--output-dir", required=True, help="Run output directory; the only write destination besides --local-scratch")
    parser.add_argument(
        "--reuse-index-dir", default=None,
        help="Phase-2 输出目录(或其 s1_indexes 子目录),复用现成 tar member 索引;空/缺省则重建",
    )
    parser.add_argument(
        "--sample-n", type=int, default=None,
        help="T1: 每密度每 tar wanted 上限(默认无上限); T3: row-group 抽样数(默认100)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--step-timeout-s", type=float, default=1_800.0, help="单个可 kill 子步骤的封顶 timeout")
    parser.add_argument(
        "--local-scratch", default=LOCAL_SCRATCH_DEFAULT,
        help="唯一允许的非 output-dir 写入根(T2/T3 staging/T4 预取共用);必须位于 /local 下",
    )
    parser.add_argument("--t2-items", type=int, default=T2_ITEMS_DEFAULT, help="T2/T4 gpic 项池大小")
    parser.add_argument("--t2-duration-s", type=float, default=300.0, help="T2 每种预取布局的测量时长")
    parser.add_argument("--t2-target-img-s", type=float, default=T2_TARGET_IMG_S, help="T2 消费者目标速率")
    parser.add_argument("--t4-duration-s", type=float, default=600.0, help="T4 混合负载时长(默认10分钟)")
    parser.add_argument("--t4-warmup-s", type=float, default=30.0, help="T4 测量前的预取/扫描起步时间")
    parser.add_argument("--t4-density", type=float, default=T1_DECISION_DENSITY, help="T4 monet/synth 密度窗口")
    parser.add_argument("--t4-monet-mode", choices=("auto", "seq", "coalesce"), default="auto")
    parser.add_argument("--t4-textatlas-mode", choices=("auto", "real", "simulated"), default="auto")
    parser.add_argument("--t4-anime-pool-n", type=int, default=20_000)
    parser.add_argument(
        "--t4-gpic-threads-per-worker", type=int, default=T4_GPIC_PREFETCH_THREADS_PER_WORKER,
        help="T4 每 worker gpic 预取线程数(节点总并发=×8;委托口径 32→4/worker)",
    )
    args = parser.parse_args()
    if args.step_timeout_s <= 0 or args.t2_duration_s <= 0 or args.t4_duration_s <= 0:
        parser.error("timeouts/durations must be positive")
    if args.t2_target_img_s <= 0 or args.t2_items <= 0 or args.t4_anime_pool_n <= 0:
        parser.error("rates and pool sizes must be positive")
    if not (0.0 < args.t4_density <= 1.0):
        parser.error("--t4-density must be in (0, 1]")
    if args.t4_gpic_threads_per_worker <= 0 or args.t4_gpic_threads_per_worker * 8 > 64:
        parser.error("--t4-gpic-threads-per-worker must keep node total prefetch concurrency in (0, 64]")
    if args.reuse_index_dir is not None and not str(args.reuse_index_dir).strip():
        args.reuse_index_dir = None  # yaml 侧 REUSE_INDEX_DIR 缺省会传空串
    scratch = Path(args.local_scratch).resolve(strict=False)
    if not str(scratch).startswith("/local/"):
        parser.error("--local-scratch must live under /local/ (fixed sanctioned write root)")
    args.local_scratch = str(scratch)
    try:
        args.output_dir = validate_output_dir(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    verdicts = [run_one(task, args) for task in args.test]
    return 2 if "FAIL" in verdicts else 0


if __name__ == "__main__":
    sys.exit(main())
