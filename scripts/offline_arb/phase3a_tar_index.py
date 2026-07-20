#!/usr/bin/env python3
"""OFFLINE_ARB_V1 Phase-3a · Job1：全量 tar member 索引（monet_real 6 子源 + synthetic 3 子源）。

见 docs/OFFLINE_ARB_V1_SPEC.md 第3节 Phase-3（"全量 catalog + assignment，输出完整 N[B,d]/坏
样本/...报告"）；本脚本只做 L1 raw catalog 的 tar member 定位子任务（(src_shard, src_key,
offset, size) 索引)，不做 assignment/N[B,d]（见同目录 phase3a_bucket_stats.py）。

扫描模式（--scan-mode，默认 stream；2026-07-10 实测后改版）：
  stream（默认）—— 整 tar 顺序流式：tarfile 'r|' 流模式（避免手写 header 解析在 PAX/GNU 长名与
    512B padding 上的边角坑）+ 底层无缓冲 fd + bufsize=8MiB —— tarfile._Stream 内部即按 8MiB
    大块顺序 read 推进，跳过 payload 走缓冲区消耗（_Stream.seek 只支持前向、以 read 实现），
    全程绝不对 FUSE fd seek。依据：Phase-2b T1 实测顺序读 718MB/s → 13.6GB tar ≈ 19s，全 corpus
    ~20TB 可行。
  seek —— 旧 seek_jump（tarfile 'r:'，header 间 fileobj.seek() 跳 payload）。保留作备选：实测在
    大 member tar 上劣化 ~17×（cc12m ~250KB/member 14ms/member；commoncatalog ~1.36MB/member
    ~234ms/member——mountpoint-s3 每次 seek 重置流/预读），大 member 子源不要用。
  两模式 locator 语义等价（Phase-1 Task C 已实测 'r|' 与 'r:' locator sha256 一致；--self-test
  的合成 tar 单测再次断言两模式逐字段相等）。

数据源（路径取自 docs/offline_arb_recon/monet_real.json / synthetic.json，运行时用 glob 展开
真实目录，不假设固定 tar 数）：
  monet_real：MROOT/{cc12m,commoncatalog-cc-by,coyo,diffusion-aesthetic-4k,laion,megalith10m}/
              <N>-0/*.tar（monet_real.json 实测每子源 90 个 "<N>-0" 目录）。
  synthetic： MROOT/synthetic/{flux-klein,flux-schnell,z-image}/<N>-0/*.tar
              （synthetic.json 实测 360+450+630=1440 个 tar shard）。

分片运行（新集群规则 2026-07-10：**禁 64cpu/0gpu job**；CPU job 单个 16 CPU、可 2-3 个并行）：
  --subsources 是**正式**分片参数（非 diag）：逗号分隔 domain__subsource token（连字符可写成
  下划线，如 monet_real__commoncatalog_cc_by），只处理指定子源，必须配 --job-tag。本 job 的记录
  写独立分片 manifest（manifest.<tag>.json）；per-subsource parquet 文件名不变——多个分片 job
  并行写同一输出目录时文件级互不冲突。3 个 JCC yaml（jobs/offline_arb_phase3a_taridx_{a,b,c}.yaml，
  各 16 CPU/96GB）分别覆盖：a=commoncatalog-cc-by(~7.5TB 最大单子源)；b=coyo+laion(~6.8TB)；
  c=cc12m(已完成,resume 秒过)+megalith10m+diffusion-aesthetic-4k+synthetic 3 子源，末尾跑
  --verify-only。
  --tar-shard K:N 是子源内部的 tar-range 分片：对每个 --subsources 子源的完整排序 tar 列表仅取
  i%N==K；各片写独立 foo.tshardKofN.parquet/manifest checkpoint。缺省不分片，沿用上述原行为。

输出（唯一写入位置，固定路径——断点续跑要求路径稳定，不按 RUN_TAG 建子目录）：
  /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_tarindex_v1/
    {domain}__{sub_source}.parquet      —— 9 个按子源分文件的 (src_shard,src_key,offset,size,
                                           domain,sub_source) 索引，zstd 压缩。
    {domain}__{sub_source}.tshardKofN.parquet —— 使用 --tar-shard 时的子源内 tar 子集索引。
    manifest.json / manifest.<tag>.json —— 子源记录（行数/sha256/scan_seconds/bad_tars）。
                                           无 --job-tag 的全量 run 写 manifest.json（legacy；
                                           cc12m 的既有完成记录就在这里）；分片 job 写自己的
                                           manifest.<tag>.json。
    gate_summary.json                   —— --verify-only 的汇总判定（PENDING/PASS/FAIL）。
    _SUCCESS                            —— 仅 --verify-only 判 PASS 时 touch。
    run_*.log / selftest_*.log          —— 由 job yaml 的 tee 写出，本脚本不管。

断点续跑（子源文件粒度，不做子源内部按 tar 的细粒度续跑）：resume 检查合并读取 manifest.json 与
所有 manifest.<tag>.json（兼容 cc12m 既有记录在 legacy manifest.json 的事实）；某子源目标 parquet
已存在、且任一分片 manifest 中有 status=ok 且 sha256 与文件实测一致的记录 → 跳过重扫。任何不一致
（文件缺失/sha256 不符/从未成功过）→ 该子源整体重扫（覆盖旧文件），用子源整体重跑的简单性换取
正确性。

gate（--verify-only，轻量只读汇总模式；c job 末尾跑）：合并全部非 diag manifest 分片，对 9 个子源
逐一核对 whole "文件在盘 + sha256 命中记录"（优先），或同一 N 下 K=0..N-1 全部 tshard 均
命中；有缺 → verdict=PENDING（其它分片 job 未完，不判失败，exit 0）；全齐 → 按参考行数
（monet_real=40,000,000 / synthetic=13,808,747）判总覆盖率 ≥99.9%，PASS→exit 0（并 touch
_SUCCESS），FAIL→exit 2。允许极少坏 tar（按完成子源聚合 bad_tars 清单，不阻断）。

诊断缩放（--only / --max-tars-per-subsource）产物一律隔离：文件名加 .diag.parquet 后缀、manifest
写到 manifest.diag.json（从不读旧 .diag manifest，每次视为全新），绝不覆盖/读取正式产物——避免
一次冒烟跑把断点续跑依赖的完整状态静默替换成截断结果。

自检（--self-test，本机可跑：秒级、纯 tempfile 合成数据，不触任何数据源/输出目录）：造 3-member
合成 tar（一个 PAX 长名 >100 chars 触发扩展头、一个 >8MiB member 跨缓冲边界、一个非 .jpg member；
含非 512 对齐 size 验 padding 推进），stream/seek 两模式各扫一遍，校验 (key,offset,size) 与按
offset 直读文件的 payload 逐字节一致、两模式结果完全相等；再把同一组合成 tar 做 whole/2-way
tshard 实扫，校验行数与行多重集合、shards 完成 gate，以及不同 N 残片不拼凑。

硬约束（同 Phase-1/2）：数据源只读；输出仅 --output-dir（固定为上面的规范路径，不接受任意路径）；
无周期 faulthandler；venv python；不改任何既有文件。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import shutil
import socket
import sys
import tarfile
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

MIB = 1 << 20
STREAM_BUFSIZE = 8 * MIB          # tarfile 'r|' _Stream 大块读缓冲（512 的倍数；FUSE 侧 8MiB 顺序读）
MROOT = Path("/s3-mewtant-raw-datasets/jasperai/monet/v1.1.0")
MONET_SUBSOURCES = (
    "cc12m",
    "commoncatalog-cc-by",
    "coyo",
    "diffusion-aesthetic-4k",
    "laion",
    "megalith10m",
)
SYNTH_SUBSOURCES = ("flux-klein", "flux-schnell", "z-image")

CANONICAL_OUTPUT_DIR = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_tarindex_v1"
)
# 参考行数（gate 基线）：monet_real.json "num_rows" 证据（256 shard * 156250 = 40,000,000，
# 与 stats cache source_counts 六项求和吻合）；synthetic.json "num_rows(footer,pf.metadata.num_rows)"
# 证据（64 shard 合计 13,808,747）。两者都是"该域全量图像 item 数"的独立交叉验证基线，覆盖率相对它们算。
REFERENCE_ROWS = {"monet_real": 40_000_000, "synthetic": 13_808_747}
COVERAGE_MIN = 0.999

SCHEMA = pa.schema([
    ("src_shard", pa.string()),
    ("src_key", pa.string()),
    ("offset", pa.int64()),
    ("size", pa.int64()),
    ("domain", pa.string()),
    ("sub_source", pa.string()),
])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_specs() -> list[tuple[str, str, Path]]:
    """(domain, sub_source, tar 根目录) 三元组；synthetic 与 monet_real 共享同一个 MROOT 挂载点。"""
    specs = [("monet_real", sub, MROOT / sub) for sub in MONET_SUBSOURCES]
    specs += [("synthetic", sub, MROOT / "synthetic" / sub) for sub in SYNTH_SUBSOURCES]
    return specs


def subsource_token(domain: str, subsource: str) -> str:
    """out_name 词干 = --subsources token（连字符规整为下划线）。"""
    return f"{domain}__{subsource.replace('-', '_')}"


def enumerate_tars(base: Path) -> list[Path]:
    """按子源根目录枚举全部 tar：<N>-0 子目录(sorted) × 每目录内 *.tar(sorted)；运行时 glob 实数，
    不假设每目录恰 1 个 shard（任务已明确"每目录若干 shard"）。"""
    found: list[Path] = []
    if not base.is_dir():
        return found
    for group in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name):
        found.extend(sorted(group.glob("*.tar")))
    return found


def parse_tar_shard(raw: str) -> tuple[int, int]:
    """argparse type：严格解析 K:N，并保证 0 <= K < N。"""
    match = re.fullmatch(r"([0-9]+):([0-9]+)", raw)
    if match is None:
        raise argparse.ArgumentTypeError("--tar-shard 必须是 K:N（十进制非负整数）")
    k, n = (int(x) for x in match.groups())
    if n <= 0 or k >= n:
        raise argparse.ArgumentTypeError("--tar-shard 必须满足 0 <= K < N")
    return k, n


def select_tar_shard(tars: list[Path], tar_shard: tuple[int, int] | None) -> list[Path]:
    """从已经全局排序的子源 tar 列表中取 i % N == K；None 保持原列表/行为。"""
    if tar_shard is None:
        return tars
    k, n = tar_shard
    return [tar_path for i, tar_path in enumerate(tars) if i % n == k]


def tar_shard_out_name(base_subsource: str, tar_shard: tuple[int, int]) -> str:
    """canonical whole 名 foo.parquet → foo.tshardKofN.parquet。"""
    if not base_subsource.endswith(".parquet"):
        raise ValueError(f"base_subsource 必须以 .parquet 结尾: {base_subsource}")
    k, n = tar_shard
    return f"{base_subsource[:-len('.parquet')]}.tshard{k}of{n}.parquet"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * MIB), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
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


# --------------------------------------------------------------------------- #
# 单 tar 扫描（stream 默认 / seek 备选）
# --------------------------------------------------------------------------- #
def scan_one_tar(task: tuple[str, str, str, str, str]) -> dict[str, Any]:
    """单 tar 的 header 扫描（子进程执行）：只索引 .jpg member 的 (src_key, data_offset, size)；
    member.offset_data 是内容体起点，供未来 range-read 直接使用。

    stream：tarfile 'r|' + 底层无缓冲 fd + bufsize=8MiB。_Stream 自带内部缓冲（按 bufsize 大块
      read 累积），header 解析与 payload 跳过都在缓冲区上推进，绝不对 FUSE fd seek；PAX/GNU 长名
      与 512B padding 全部由 tarfile 处理。TarInfo.offset_data 在流模式取自 _Stream.tell()（已
      消耗字节数）——绝对文件偏移语义与 'r:' 相同。
    seek：tarfile 'r:'（旧 seek_jump 备选；大 member tar 在 mountpoint-s3 上劣化 ~17×）。
    """
    domain, subsource, rel_shard, tar_path_s, scan_mode = task
    keys: list[str] = []
    offsets: list[int] = []
    sizes: list[int] = []
    total_members = 0
    started = time.perf_counter()

    def consume(tf: tarfile.TarFile) -> None:
        nonlocal total_members
        for member in tf:
            total_members += 1
            if member.isfile() and member.name.endswith(".jpg"):
                keys.append(member.name[:-4].split("/")[-1])
                offsets.append(int(member.offset_data))
                sizes.append(int(member.size))

    try:
        if scan_mode == "stream":
            with open(tar_path_s, "rb", buffering=0) as raw:
                with tarfile.open(fileobj=raw, mode="r|", bufsize=STREAM_BUFSIZE) as tf:
                    consume(tf)
        elif scan_mode == "seek":
            with tarfile.open(tar_path_s, "r:") as tf:
                consume(tf)
        else:
            raise ValueError(f"unknown scan_mode: {scan_mode}")
        return {
            "ok": True, "domain": domain, "sub_source": subsource, "rel_shard": rel_shard,
            "tar_path": tar_path_s, "keys": keys, "offsets": offsets, "sizes": sizes,
            "total_members": total_members, "image_members": len(keys),
            "scan_seconds": time.perf_counter() - started, "scan_mode": scan_mode,
        }
    except Exception as exc:  # noqa: BLE001 - 坏 tar 记录为条目，不让整子源崩
        return {
            "ok": False, "domain": domain, "sub_source": subsource, "rel_shard": rel_shard,
            "tar_path": tar_path_s, "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=10),
        }


def _empty_table() -> pa.Table:
    return pa.table({f.name: pa.array([], type=f.type) for f in SCHEMA}, schema=SCHEMA)


def scan_subsource(
    pool: concurrent.futures.ProcessPoolExecutor,
    domain: str, subsource: str, base: Path, out_path: Path,
    max_tars: int | None, scan_mode: str, job_tag: str | None,
    tar_shard: tuple[int, int] | None = None, source_root: Path = MROOT,
) -> dict[str, Any]:
    """扫一个子源全部 tar，流式写单个子源 parquet（每 tar 结果即写即弃，内存只顶一个 tar 的量）。"""
    all_tars = enumerate_tars(base)
    tars = select_tar_shard(all_tars, tar_shard)
    if max_tars is not None:
        tars = tars[:max_tars]
    tasks = [
        (domain, subsource, t.relative_to(source_root).with_suffix(".parquet").as_posix(), str(t), scan_mode)
        for t in tars
    ]
    tmp_path = out_path.with_suffix(".parquet.tmp")
    tmp_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    image_members_total = 0
    total_members_total = 0
    bad: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for res in pool.map(scan_one_tar, tasks, chunksize=1) if tasks else ():
            if not res["ok"]:
                bad.append({"tar_path": res["tar_path"], "error": res["error"]})
                continue
            total_members_total += res["total_members"]
            n = len(res["keys"])
            image_members_total += n
            if n == 0:
                continue
            table = pa.table({
                "src_shard": pa.array([res["rel_shard"]] * n, type=pa.string()),
                "src_key": pa.array(res["keys"], type=pa.string()),
                "offset": pa.array(res["offsets"], type=pa.int64()),
                "size": pa.array(res["sizes"], type=pa.int64()),
                "domain": pa.array([domain] * n, type=pa.string()),
                "sub_source": pa.array([subsource] * n, type=pa.string()),
            }, schema=SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, SCHEMA, compression="zstd")
            writer.write_table(table)
            rows += n
        if writer is not None:
            writer.close()
        else:
            pq.write_table(_empty_table(), tmp_path, compression="zstd")
    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:  # noqa: BLE001 - 已在异常路径，尽力关闭不再抛出新异常
                pass
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, out_path)
    digest = sha256_file(out_path)
    elapsed = time.perf_counter() - started
    result = {
        "domain": domain, "sub_source": subsource, "tar_count": len(tars),
        "bad_tar_count": len(bad), "bad_tars": bad[:50],
        "rows": rows, "image_members": image_members_total, "total_members": total_members_total,
        "sha256": digest, "scan_seconds": elapsed, "status": "ok",
        "scan_mode": scan_mode, "job_tag": job_tag,
        "output_path": str(out_path), "output_bytes": out_path.stat().st_size,
    }
    if tar_shard is not None:
        k, n = tar_shard
        result.update({
            "tar_shard_k": k,
            "tar_shard_n": n,
            "base_subsource": f"{subsource_token(domain, subsource)}.parquet",
        })
    return result


# --------------------------------------------------------------------------- #
# manifest 分片：装载/合并（resume 与 --verify-only 共用）
# --------------------------------------------------------------------------- #
def load_shard_manifest(path: Path, job_tag: str | None) -> dict[str, Any]:
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("files", {})
        return data
    return {"schema_version": "offline_arb_tarindex_v1", "files": {}, "job_tag": job_tag}


def load_all_manifest_records(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """合并读取 manifest.json 与所有 manifest.<tag>.json（排除 .diag），返回 out_name → 记录列表。
    resume 与 --verify-only 共用；cc12m 的既有完成记录在 legacy manifest.json 里，靠这里兼容。"""
    records: dict[str, list[dict[str, Any]]] = {}
    for p in sorted(output_dir.glob("manifest*.json")):
        if "diag" in p.name:                       # manifest.diag.json 隔离，永不参与 resume/gate
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            print(f"[tarindex] WARNING 跳过不可读 manifest 分片 {p}: {exc}", flush=True)
            continue
        for name, rec in (data.get("files") or {}).items():
            if name.endswith(".diag.parquet"):
                continue
            records.setdefault(name, []).append(rec)
    return records


def find_complete_record(
    records: dict[str, list[dict[str, Any]]], out_name: str, out_path: Path,
) -> dict[str, Any] | None:
    """"输出文件在盘 + 任一分片 manifest 有 status=ok 且 sha256 与盘上实测一致" → 返回该记录。
    resume 跳过与 verify 完成性判定共用同一严格口径。"""
    candidates = [r for r in records.get(out_name, []) if r.get("status") == "ok" and r.get("sha256")]
    if not candidates or not out_path.is_file():
        return None
    digest = sha256_file(out_path)
    for rec in candidates:
        if rec.get("sha256") == digest:
            return rec
    return None


def _tar_shard_metadata(out_name: str, rec: Any) -> tuple[str, int, int] | None:
    """返回结构自洽的 (base_subsource, K, N)；畸形/伪装记录不参与 shard gate。"""
    if not isinstance(rec, dict):
        return None
    base_subsource = rec.get("base_subsource")
    k, n = rec.get("tar_shard_k"), rec.get("tar_shard_n")
    if not isinstance(base_subsource, str) or type(k) is not int or type(n) is not int:
        return None
    if n <= 0 or k < 0 or k >= n:
        return None
    try:
        expected_name = tar_shard_out_name(base_subsource, (k, n))
    except ValueError:
        return None
    if out_name != expected_name:
        return None
    return base_subsource, k, n


def _has_tar_shard_records(records: dict[str, list[dict[str, Any]]]) -> bool:
    return any(
        _tar_shard_metadata(out_name, rec) is not None
        for out_name, recs in records.items()
        for rec in recs
    )


def _find_complete_shard_record(
    records: dict[str, list[dict[str, Any]]], out_name: str, out_path: Path,
    base_subsource: str, k: int, n: int,
) -> dict[str, Any] | None:
    """find_complete_record 的 shard 严格版：sha 命中记录还必须属于指定 base/K/N。"""
    candidates = [
        rec for rec in records.get(out_name, [])
        if isinstance(rec, dict)
        and rec.get("status") == "ok" and rec.get("sha256")
        and _tar_shard_metadata(out_name, rec) == (base_subsource, k, n)
    ]
    if not candidates or not out_path.is_file():
        return None
    digest = sha256_file(out_path)
    for rec in candidates:
        if rec.get("sha256") == digest:
            return rec
    return None


def resolve_subsource_completion(
    records: dict[str, list[dict[str, Any]]], base_subsource: str, output_dir: Path,
) -> dict[str, Any] | None:
    """whole 优先；否则取最小的、同一 N 下 K=0..N-1 全部严格命中的 tshard 组。"""
    whole = find_complete_record(records, base_subsource, output_dir / base_subsource)
    if whole is not None:
        return {"complete_via": "whole", "components": [(base_subsource, whole)]}

    groups: dict[int, set[int]] = {}
    for out_name, recs in records.items():
        for rec in recs:
            metadata = _tar_shard_metadata(out_name, rec)
            if metadata is None or metadata[0] != base_subsource:
                continue
            _base, k, n = metadata
            groups.setdefault(n, set()).add(k)

    for n in sorted(groups):
        # 所有 K 已经过 0 <= K < N 校验；因此数量为 N 即恰好覆盖 0..N-1。
        if len(groups[n]) != n:
            continue
        components: list[tuple[str, dict[str, Any]]] = []
        for k in range(n):
            out_name = tar_shard_out_name(base_subsource, (k, n))
            rec = _find_complete_shard_record(
                records, out_name, output_dir / out_name, base_subsource, k, n,
            )
            if rec is None:
                break
            components.append((out_name, rec))
        if len(components) == n:
            return {
                "complete_via": f"shards {n}",
                "tar_shard_n": n,
                "components": components,
            }
    return None


def _aggregate_completion(completion: dict[str, Any]) -> dict[str, Any]:
    components: list[tuple[str, dict[str, Any]]] = completion["components"]
    if completion["complete_via"] == "whole":
        return components[0][1]

    recs = [rec for _name, rec in components]

    def summed(field: str) -> int | float:
        return sum((rec.get(field) or 0) for rec in recs)

    def common_or_list(field: str) -> Any:
        values = [rec.get(field) for rec in recs]
        return values[0] if all(value == values[0] for value in values) else values

    bad_tars: list[dict[str, Any]] = []
    for out_name, rec in components:
        for bad in rec.get("bad_tars", []) or []:
            bad_tars.append({"shard_file": out_name, **bad})
    return {
        "domain": common_or_list("domain"),
        "sub_source": common_or_list("sub_source"),
        "tar_count": summed("tar_count"),
        "bad_tar_count": summed("bad_tar_count"),
        "bad_tars": bad_tars,
        "rows": summed("rows"),
        "image_members": summed("image_members"),
        "total_members": summed("total_members"),
        "sha256": {name: rec.get("sha256") for name, rec in components},
        "scan_seconds": summed("scan_seconds"),
        "scan_mode": common_or_list("scan_mode"),
        "job_tag": common_or_list("job_tag"),
        "output_bytes": summed("output_bytes"),
    }


def _verify_shard_aware(
    output_dir: Path, records: dict[str, list[dict[str, Any]]],
    specs: list[tuple[str, str, Path]], reference_rows: dict[str, int],
    coverage_min: float, *, emit: bool = True,
) -> int:
    """存在 tshard 记录时使用的 gate；参数化 specs/reference 供纯 tempfile self-test。"""
    per_sub: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    completion_by_subsource: dict[str, dict[str, Any]] = {}
    for domain, subsource, _base in specs:
        base_subsource = f"{subsource_token(domain, subsource)}.parquet"
        completion = resolve_subsource_completion(records, base_subsource, output_dir)
        if completion is None:
            pending.append(base_subsource)
            completion_by_subsource[base_subsource] = {"complete": False, "complete_via": None}
            continue
        completion["aggregate"] = _aggregate_completion(completion)
        per_sub[base_subsource] = completion
        completion_by_subsource[base_subsource] = {
            "complete": True,
            "complete_via": completion["complete_via"],
        }
        if "tar_shard_n" in completion:
            completion_by_subsource[base_subsource]["tar_shard_n"] = completion["tar_shard_n"]

    if pending:
        gate = {
            "verdict": "PENDING",
            "note": "部分子源尚无 whole 或同一 N 下全部 tshard 的"
                    "<文件在盘+sha256 命中 manifest 记录>完成证据"
                    "（其它分片 job 未完或未落盘），不判失败",
            "pending_subsources": pending,
            "complete_subsources": sorted(per_sub),
            "completion_by_subsource": completion_by_subsource,
            "generated_at": utc_now(), "host": socket.gethostname(),
        }
        atomic_write_json(output_dir / "gate_summary.json", gate)
        if emit:
            print("==== OFFLINE_ARB PHASE3a TARINDEX verify ====")
            print(json.dumps(gate, ensure_ascii=False, indent=2))
        return 0

    domain_totals: dict[str, Any] = {}
    for domain, ref_rows in reference_rows.items():
        total = sum(
            completion["aggregate"]["image_members"]
            for completion in per_sub.values()
            if completion["aggregate"].get("domain") == domain
        )
        coverage = total / ref_rows if ref_rows else 0.0
        domain_totals[domain] = {
            "image_members": total, "reference_rows": ref_rows,
            "coverage": coverage, "pass": coverage >= coverage_min,
        }
    overall_pass = all(value["pass"] for value in domain_totals.values())
    bad_tars_overall = [
        {"file": out_name, **bad}
        for _base_subsource, completion in sorted(per_sub.items())
        for out_name, rec in completion["components"]
        for bad in rec.get("bad_tars", []) or []
    ]

    per_subsource: dict[str, dict[str, Any]] = {}
    for base_subsource, completion in sorted(per_sub.items()):
        aggregate = completion["aggregate"]
        summary = {
            "domain": aggregate.get("domain"),
            "sub_source": aggregate.get("sub_source"),
            "tar_count": aggregate.get("tar_count"),
            "rows": aggregate.get("rows"),
            "image_members": aggregate.get("image_members"),
            "bad_tar_count": aggregate.get("bad_tar_count"),
            "sha256": aggregate.get("sha256"),
            "scan_seconds": aggregate.get("scan_seconds"),
            "scan_mode": aggregate.get("scan_mode"),
            "job_tag": aggregate.get("job_tag"),
            "complete_via": completion["complete_via"],
        }
        if "tar_shard_n" in completion:
            summary["tar_shard_n"] = completion["tar_shard_n"]
            summary["shard_files"] = [name for name, _rec in completion["components"]]
        per_subsource[base_subsource] = summary

    gate = {
        "verdict": "PASS" if overall_pass else "FAIL",
        "coverage_min": coverage_min,
        "domain_totals": domain_totals,
        "bad_tar_count_overall": sum(
            completion["aggregate"].get("bad_tar_count", 0) or 0
            for completion in per_sub.values()
        ),
        "bad_tars_overall": bad_tars_overall[:200],
        "per_subsource": per_subsource,
        "generated_at": utc_now(), "host": socket.gethostname(),
    }
    atomic_write_json(output_dir / "gate_summary.json", gate)
    if overall_pass:
        (output_dir / "_SUCCESS").touch()
    if emit:
        print("==== OFFLINE_ARB PHASE3a TARINDEX verify ====")
        print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0 if overall_pass else 2


# --------------------------------------------------------------------------- #
# --verify-only：只读汇总 gate（c job 末尾跑；其它分片未完 → PENDING 不判失败）
# --------------------------------------------------------------------------- #
def verify_only(
    output_dir: Path, *, specs: list[tuple[str, str, Path]] | None = None,
    reference_rows: dict[str, int] | None = None, coverage_min: float = COVERAGE_MIN,
    emit: bool = True,
) -> int:
    records = load_all_manifest_records(output_dir)
    # 纯 legacy whole 环境走下方原实现，保持旧 gate JSON/stdout 字段与计数语义不变。
    if _has_tar_shard_records(records):
        return _verify_shard_aware(
            output_dir, records,
            source_specs() if specs is None else specs,
            REFERENCE_ROWS if reference_rows is None else reference_rows,
            coverage_min, emit=emit,
        )
    per_sub: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for domain, subsource, _base in source_specs():
        out_name = f"{subsource_token(domain, subsource)}.parquet"
        rec = find_complete_record(records, out_name, output_dir / out_name)
        if rec is None:
            pending.append(out_name)
        else:
            per_sub[out_name] = rec

    if pending:
        gate = {
            "verdict": "PENDING",
            "note": "部分子源尚无 <文件在盘+sha256 命中 manifest 记录> 的完成证据"
                    "（其它分片 job 未完或未落盘），不判失败",
            "pending_subsources": pending,
            "complete_subsources": sorted(per_sub),
            "generated_at": utc_now(), "host": socket.gethostname(),
        }
        atomic_write_json(output_dir / "gate_summary.json", gate)
        print("==== OFFLINE_ARB PHASE3a TARINDEX verify ====")
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return 0

    domain_totals: dict[str, Any] = {}
    for domain, ref_rows in REFERENCE_ROWS.items():
        total = sum(r["image_members"] for r in per_sub.values() if r.get("domain") == domain)
        coverage = total / ref_rows if ref_rows else 0.0
        domain_totals[domain] = {
            "image_members": total, "reference_rows": ref_rows,
            "coverage": coverage, "pass": coverage >= COVERAGE_MIN,
        }
    overall_pass = all(v["pass"] for v in domain_totals.values())
    bad_tars_overall = [
        {"file": name, **b} for name, r in sorted(per_sub.items()) for b in r.get("bad_tars", [])
    ]
    gate = {
        "verdict": "PASS" if overall_pass else "FAIL",
        "coverage_min": COVERAGE_MIN,
        "domain_totals": domain_totals,
        "bad_tar_count_overall": len(bad_tars_overall),
        "bad_tars_overall": bad_tars_overall[:200],
        "per_subsource": {
            name: {k: r.get(k) for k in ("domain", "sub_source", "tar_count", "rows",
                                          "image_members", "bad_tar_count", "sha256",
                                          "scan_seconds", "scan_mode", "job_tag")}
            for name, r in sorted(per_sub.items())
        },
        "generated_at": utc_now(), "host": socket.gethostname(),
    }
    atomic_write_json(output_dir / "gate_summary.json", gate)
    if overall_pass:
        (output_dir / "_SUCCESS").touch()
    print("==== OFFLINE_ARB PHASE3a TARINDEX verify ====")
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0 if overall_pass else 2


# --------------------------------------------------------------------------- #
# --self-test：纯合成 tar 的顺序解析器单测（本机可跑：tempfile，秒级，不触数据源/输出目录）
# --------------------------------------------------------------------------- #
def self_test_stream_parser() -> dict[str, Any]:
    """3-member 合成 tar：PAX 长名(>100 chars 触发扩展头) + >8MiB member(跨 STREAM_BUFSIZE 缓冲
    边界) + 非 .jpg member(须被过滤但计入 total_members)；两个 jpg 的 size 都非 512 对齐，验
    padding 推进。校验：stream 扫描的 (key,offset,size) 使得按 offset 直读文件的 payload 逐字节
    一致；stream 与 seek 两模式结果完全相等。另用同组 3 个 tar 校验 2-way tshard 行守恒、
    shard-aware verify、bad_tars 聚合、跨 N 残片拒绝与 whole 优先。"""
    long_key = "p" * 130                                   # >100 chars → PAX 'path' 扩展头
    payload_long = b"long-name-jpg-payload" * 33           # 693B，非 512 对齐
    payload_big = os.urandom(8 * MIB + 123 * 512 + 77)     # >8MiB 且非 512 对齐，跨缓冲边界
    payload_json = b'{"meta": true}'
    members = [
        (f"{long_key}.jpg", payload_long),
        ("bigmember.jpg", payload_big),
        ("sidecar.json", payload_json),
    ]
    expected = {long_key: payload_long, "bigmember": payload_big}
    checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="p3a_taridx_selftest_") as td:
        root = Path(td)
        source_root = root / "mroot"
        source_base = source_root / "test-source"
        tar_group = source_base / "000-0"
        tar_group.mkdir(parents=True)

        def write_tar(path: Path, tar_members: list[tuple[str, bytes]]) -> None:
            with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tw:
                for name, data in tar_members:
                    ti = tarfile.TarInfo(name=name)
                    ti.size = len(data)
                    tw.addfile(ti, io.BytesIO(data))

        tar_path = tar_group / "000.tar"
        write_tar(tar_path, members)
        write_tar(tar_group / "001.tar", [
            ("nested/second.jpg", b"second-jpg" * 71),
            ("nested/second.txt", b"not-an-image"),
        ])
        write_tar(tar_group / "002.tar", [
            ("third.jpg", b"third-jpg" * 93),
        ])
        results = {
            mode: scan_one_tar(("t", "t", "t.parquet", str(tar_path), mode))
            for mode in ("stream", "seek")
        }
        for mode, res in results.items():
            checks.append({"check": f"{mode}_ok", "pass": bool(res.get("ok")),
                           "detail": res.get("error")})
            if not res.get("ok"):
                continue
            checks.append({"check": f"{mode}_member_counts",
                           "pass": res["total_members"] == 3 and res["image_members"] == 2,
                           "detail": {"total": res["total_members"], "image": res["image_members"]}})
            got = dict(zip(res["keys"], zip(res["offsets"], res["sizes"])))
            checks.append({"check": f"{mode}_key_set", "pass": set(got) == set(expected),
                           "detail": sorted(k[:12] for k in got)})
            with tar_path.open("rb") as f:
                for key, want in expected.items():
                    off, size = got.get(key, (None, None))
                    ok = off is not None and size == len(want)
                    if ok:
                        f.seek(off)
                        ok = f.read(size) == want
                    checks.append({"check": f"{mode}_payload_bytes_{key[:8]}", "pass": bool(ok),
                                   "detail": {"offset": off, "size": size}})
        if all(r.get("ok") for r in results.values()):
            same = all(results["stream"][k] == results["seek"][k]
                       for k in ("keys", "offsets", "sizes", "total_members", "image_members"))
            checks.append({"check": "stream_equals_seek", "pass": same, "detail": None})

        # 同一组排序后的 tar：整扫 + K=0/1,N=2 各实扫一遍并写 parquet，验证行多重集合守恒。
        domain, subsource = "test_domain", "test-source"
        base_subsource = f"{subsource_token(domain, subsource)}.parquet"
        whole_dir = root / "whole"
        shard_dir = root / "shards_positive"
        whole_dir.mkdir()
        shard_dir.mkdir()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            whole_path = whole_dir / base_subsource
            whole_result = scan_subsource(
                pool, domain, subsource, source_base, whole_path,
                None, "stream", "selftest-whole", None, source_root,
            )
            shard_results: dict[int, tuple[str, dict[str, Any]]] = {}
            for k in range(2):
                out_name = tar_shard_out_name(base_subsource, (k, 2))
                result = scan_subsource(
                    pool, domain, subsource, source_base, shard_dir / out_name,
                    None, "stream", f"selftest-shard-{k}", (k, 2), source_root,
                )
                shard_results[k] = (out_name, result)

        def parquet_rows(path: Path) -> Counter[tuple[Any, ...]]:
            table = pq.read_table(path, columns=SCHEMA.names)
            columns = [table.column(name).to_pylist() for name in SCHEMA.names]
            return Counter(zip(*columns))

        whole_rows = parquet_rows(whole_path)
        shard_rows = Counter()
        for out_name, _result in shard_results.values():
            shard_rows += parquet_rows(shard_dir / out_name)
        shard_row_count = sum(result["rows"] for _name, result in shard_results.values())
        checks.append({
            "check": "tar_shard_row_count_sum",
            "pass": shard_row_count == whole_result["rows"],
            "detail": {"whole": whole_result["rows"], "shards": shard_row_count},
        })
        checks.append({
            "check": "tar_shard_row_content_union",
            "pass": shard_rows == whole_rows,
            "detail": {"whole_unique_rows": len(whole_rows), "shard_unique_rows": len(shard_rows)},
        })

        # 正例 manifest 只放 N=2 的两片（不放 whole），走 verify 的 (b) 完成路径。
        positive_files: dict[str, dict[str, Any]] = {}
        for k, (out_name, result) in shard_results.items():
            gate_rec = dict(result)
            # 模拟每片 bad_tars 详情被截断：overall count 应求真实 count 之和，详情仍全部聚合。
            gate_rec["bad_tar_count"] = 51 if k == 0 else 2
            gate_rec["bad_tars"] = [{
                "tar_path": f"selftest-bad-{k}.tar", "error": "synthetic bad tar",
            }]
            positive_files[out_name] = gate_rec
        positive_manifest = {
            "schema_version": "offline_arb_tarindex_v1",
            "files": positive_files,
            "job_tag": "selftest-shards",
        }
        atomic_write_json(shard_dir / "manifest.selftest.json", positive_manifest)
        positive_rc = verify_only(
            shard_dir, specs=[(domain, subsource, source_base)],
            reference_rows={domain: whole_result["image_members"]}, emit=False,
        )
        with (shard_dir / "gate_summary.json").open("r", encoding="utf-8") as f:
            positive_gate = json.load(f)
        positive_summary = positive_gate.get("per_subsource", {}).get(base_subsource, {})
        checks.append({
            "check": "tar_shard_verify_complete",
            "pass": (
                positive_rc == 0
                and positive_gate.get("verdict") == "PASS"
                and positive_summary.get("complete_via") == "shards 2"
                and positive_summary.get("tar_shard_n") == 2
                and (shard_dir / "_SUCCESS").is_file()
            ),
            "detail": {
                "rc": positive_rc, "verdict": positive_gate.get("verdict"),
                "complete_via": positive_summary.get("complete_via"),
            },
        })
        checks.append({
            "check": "tar_shard_verify_coverage_aggregate",
            "pass": (
                positive_gate.get("domain_totals", {}).get(domain, {}).get("image_members")
                == whole_result["image_members"]
                and positive_summary.get("rows") == whole_result["rows"]
            ),
            "detail": {
                "whole_image_members": whole_result["image_members"],
                "gate_image_members": positive_gate.get("domain_totals", {})
                .get(domain, {}).get("image_members"),
            },
        })
        checks.append({
            "check": "tar_shard_verify_bad_tars_aggregate",
            "pass": (
                positive_gate.get("bad_tar_count_overall") == 53
                and positive_summary.get("bad_tar_count") == 53
                and len(positive_gate.get("bad_tars_overall", [])) == 2
                and {
                    bad.get("file") for bad in positive_gate.get("bad_tars_overall", [])
                } == {out_name for out_name, _result in shard_results.values()}
            ),
            "detail": {
                "count": positive_gate.get("bad_tar_count_overall"),
                "listed": len(positive_gate.get("bad_tars_overall", [])),
            },
        })

        # 负例：0of2 + 1of3 + 2of3 的 K 并集看似齐全，但不同 N 的残片绝不能拼成完整子源。
        negative_dir = root / "shards_mixed_n_negative"
        negative_dir.mkdir()
        negative_files: dict[str, dict[str, Any]] = {}
        for source_k, target_k, target_n in ((0, 0, 2), (1, 1, 3), (0, 2, 3)):
            source_name, source_rec = shard_results[source_k]
            target_name = tar_shard_out_name(base_subsource, (target_k, target_n))
            target_path = negative_dir / target_name
            shutil.copyfile(shard_dir / source_name, target_path)
            target_rec = dict(source_rec)
            target_rec.update({
                "tar_shard_k": target_k,
                "tar_shard_n": target_n,
                "base_subsource": base_subsource,
                "sha256": sha256_file(target_path),
                "output_path": str(target_path),
                "output_bytes": target_path.stat().st_size,
            })
            negative_files[target_name] = target_rec
        atomic_write_json(negative_dir / "manifest.selftest.json", {
            "schema_version": "offline_arb_tarindex_v1",
            "files": negative_files,
            "job_tag": "selftest-mixed-n",
        })
        negative_rc = verify_only(
            negative_dir, specs=[(domain, subsource, source_base)],
            reference_rows={domain: whole_result["image_members"]}, emit=False,
        )
        with (negative_dir / "gate_summary.json").open("r", encoding="utf-8") as f:
            negative_gate = json.load(f)
        checks.append({
            "check": "tar_shard_mixed_n_not_combined",
            "pass": (
                negative_rc == 0
                and negative_gate.get("verdict") == "PENDING"
                and base_subsource in negative_gate.get("pending_subsources", [])
                and not (negative_dir / "_SUCCESS").exists()
            ),
            "detail": {
                "rc": negative_rc, "verdict": negative_gate.get("verdict"),
                "pending": negative_gate.get("pending_subsources"),
            },
        })

        # whole 与完整 shards 同时存在时，完成证据必须优先选择 whole。
        whole_gate_path = shard_dir / base_subsource
        shutil.copyfile(whole_path, whole_gate_path)
        whole_gate_rec = dict(whole_result)
        whole_gate_rec.update({
            "sha256": sha256_file(whole_gate_path),
            "output_path": str(whole_gate_path),
            "output_bytes": whole_gate_path.stat().st_size,
        })
        positive_manifest["files"][base_subsource] = whole_gate_rec
        atomic_write_json(shard_dir / "manifest.selftest.json", positive_manifest)
        preferred = resolve_subsource_completion(
            load_all_manifest_records(shard_dir), base_subsource, shard_dir,
        )
        checks.append({
            "check": "tar_shard_whole_preferred",
            "pass": preferred is not None and preferred.get("complete_via") == "whole",
            "detail": None if preferred is None else preferred.get("complete_via"),
        })
    all_pass = all(c["pass"] for c in checks)
    return {"task": "self_test_stream_parser", "pass": all_pass, "checks": checks,
            "stream_bufsize": STREAM_BUFSIZE, "generated_at": utc_now()}


# --------------------------------------------------------------------------- #
# CLI + 主流程
# --------------------------------------------------------------------------- #
def validate_output_dir(raw: str) -> Path:
    path = Path(raw).expanduser().resolve(strict=False)
    if str(path) != str(CANONICAL_OUTPUT_DIR):
        raise ValueError(
            f"--output-dir 必须是固定路径 {CANONICAL_OUTPUT_DIR}（断点续跑依赖稳定路径，不按 "
            f"RUN_TAG 建子目录），got {path}"
        )
    return path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=str(CANONICAL_OUTPUT_DIR))
    ap.add_argument("--workers", type=int, default=14,
                     help="扫描进程数；16 CPU job 默认 14（留 2 核给 IO/主进程）")
    ap.add_argument("--scan-mode", choices=("stream", "seek"), default="stream",
                     help="stream=顺序流式(默认；大 member tar 必用)；seek=旧 seek_jump 备选")
    ap.add_argument("--subsources", default=None,
                     help="正式分片参数(非 diag)：逗号分隔 domain__subsource token"
                          "（如 monet_real__coyo,monet_real__laion；连字符可写下划线），"
                          "只处理指定子源。必须配 --job-tag")
    ap.add_argument("--tar-shard", type=parse_tar_shard, metavar="K:N", default=None,
                    help="对子源完整排序 tar 列表只扫描 i%%N==K；输出名加 .tshardKofN")
    ap.add_argument("--job-tag", default=None,
                     help="分片 manifest 后缀：记录写 manifest.<tag>.json（与其它并行分片 job 隔离）")
    ap.add_argument("--verify-only", action="store_true",
                     help="只读汇总 gate：合并全部 manifest 分片判 9 子源总覆盖率；"
                          "缺分片 → PENDING(exit 0)，全齐 → PASS/FAIL")
    ap.add_argument("--self-test", action="store_true",
                     help="纯合成 tar 的顺序解析器单测（tempfile，本机可跑，不触数据源/输出目录）")
    ap.add_argument("--only", default=None,
                     help="诊断缩放：逗号分隔 domain:sub_source 白名单（如 monet_real:cc12m），"
                          "给定时产物走 .diag 隔离(diagnostic_run=true)")
    ap.add_argument("--max-tars-per-subsource", type=int, default=None,
                     help="诊断缩放：每子源最多扫描的 tar 数；给定时产物走 .diag 隔离")
    args = ap.parse_args()
    if args.workers <= 0:
        ap.error("--workers 必须为正")
    if args.max_tars_per_subsource is not None and args.max_tars_per_subsource <= 0:
        ap.error("--max-tars-per-subsource 必须为正")
    if args.subsources and args.only:
        ap.error("--subsources(正式分片) 与 --only(诊断) 互斥")
    if args.subsources and not args.job_tag:
        ap.error("--subsources 必须配 --job-tag（分片 manifest 隔离）")
    if args.tar_shard is not None and not args.subsources:
        ap.error("--tar-shard 必须配 --subsources（以及其要求的 --job-tag）")
    if args.job_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.job_tag):
        ap.error("--job-tag 只允许 [A-Za-z0-9_-]+")
    return args


def main() -> int:
    args = parse_args()

    if args.self_test:
        result = self_test_stream_parser()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"SELF-TEST {'PASS' if result['pass'] else 'FAIL'}")
        return 0 if result["pass"] else 2

    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        return verify_only(output_dir)

    specs = source_specs()
    if args.subsources:
        wanted = {x.strip().replace("-", "_") for x in args.subsources.split(",") if x.strip()}
        matched = [s for s in specs if subsource_token(s[0], s[1]) in wanted]
        matched_tokens = {subsource_token(s[0], s[1]) for s in matched}
        unknown = wanted - matched_tokens
        if unknown:
            raise SystemExit(f"--subsources 含未知 token: {sorted(unknown)}；"
                             f"合法集合={sorted(subsource_token(d, s) for d, s, _ in specs)}")
        specs = matched
    if args.only:
        only = {x.strip() for x in args.only.split(",") if x.strip()}
        specs = [s for s in specs if f"{s[0]}:{s[1]}" in only]
        if not specs:
            raise SystemExit(f"--only 未匹配任何子源: {args.only}")
    diagnostic_run = bool(args.only) or (args.max_tars_per_subsource is not None)

    # manifest 分片选择。诊断缩放绝不能碰正式 parquet/manifest——否则一次冒烟跑会把断点续跑依赖的
    # "完整"文件/记录静默替换成截断结果。诊断产物一律 .diag 后缀+独立 manifest 且从不参与 resume。
    if diagnostic_run:
        manifest_path = output_dir / "manifest.diag.json"
        manifest: dict[str, Any] = {"schema_version": "offline_arb_tarindex_v1", "files": {},
                                    "diagnostic_run": True}
    elif args.job_tag:
        manifest_path = output_dir / f"manifest.{args.job_tag}.json"
        manifest = load_shard_manifest(manifest_path, args.job_tag)
    else:
        manifest_path = output_dir / "manifest.json"       # legacy 全量模式（cc12m 既有记录在此）
        manifest = load_shard_manifest(manifest_path, None)
    prior_records = {} if diagnostic_run else load_all_manifest_records(output_dir)

    overall_started = time.perf_counter()
    empty_subsources: list[str] = []

    ctx = mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        for domain, subsource, base in specs:
            base_subsource = f"{subsource_token(domain, subsource)}.parquet"
            formal_out_name = (
                tar_shard_out_name(base_subsource, args.tar_shard)
                if args.tar_shard is not None else base_subsource
            )
            out_name = (
                f"{formal_out_name[:-len('.parquet')]}.diag.parquet"
                if diagnostic_run else formal_out_name
            )
            out_path = output_dir / out_name
            if not diagnostic_run:
                rec = find_complete_record(prior_records, out_name, out_path)
                if rec is not None:
                    print(f"[tarindex] skip(resume) {out_name}: rows={rec.get('rows')} "
                          f"sha256 已核对匹配(来源分片 job_tag={rec.get('job_tag')})", flush=True)
                    continue
            print(f"[tarindex] scanning {out_name} (domain={domain} sub_source={subsource}, "
                  f"base={base}, mode={args.scan_mode}) ...", flush=True)
            result = scan_subsource(pool, domain, subsource, base, out_path,
                                    args.max_tars_per_subsource, args.scan_mode, args.job_tag,
                                    args.tar_shard)
            # 合法空片（例如 N 大于 tar 数）不等于源目录枚举为 0 的挂载异常；仅空片时再枚举一次。
            if result["tar_count"] == 0 and (
                args.tar_shard is None or not enumerate_tars(base)
            ):
                empty_subsources.append(out_name)
            manifest["files"][out_name] = result
            atomic_write_json(manifest_path, manifest)   # 每子源完成即落盘 checkpoint（断点续跑用）
            print(f"[tarindex] done {out_name}: tar_count={result['tar_count']} rows={result['rows']} "
                  f"bad_tars={result['bad_tar_count']} scan_seconds={result['scan_seconds']:.1f}", flush=True)

    manifest["total_duration_seconds"] = time.perf_counter() - overall_started
    manifest["generated_at"] = utc_now()
    manifest["host"] = socket.gethostname()
    manifest["scan_mode"] = args.scan_mode
    atomic_write_json(manifest_path, manifest)

    print("==== OFFLINE_ARB PHASE3a TARINDEX scan done ====")
    print(json.dumps({
        "manifest": str(manifest_path),
        "subsources_in_scope": [subsource_token(d, s) for d, s, _ in specs],
        "empty_subsources": empty_subsources,
        "diagnostic_run": diagnostic_run,
        "duration_seconds": manifest["total_duration_seconds"],
        "note": "总覆盖率 gate 由 --verify-only 汇总判定（c job 末尾跑）",
    }, ensure_ascii=False, indent=2))
    if empty_subsources and not diagnostic_run:
        print(f"[tarindex] ERROR 下列子源枚举到 0 个 tar（挂载/路径异常，fail-fast）: "
              f"{empty_subsources}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
