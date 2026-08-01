#!/usr/bin/env python3
"""OFFLINE_ARB_V1 Phase-3a · Job2：全量 N[B,d] + 裁剪保留率（桶表决策材料）。

见 docs/OFFLINE_ARB_V1_SPEC.md 第3节 Phase-3："全量 catalog + assignment，输出完整 N[B,d]/坏
样本/重复因子/crop-retention 报告"（本脚本产出 N[B,d] 与 crop-retention 部分；"桶表由 Evan 据全量
数据拍板"——本脚本只产出决策材料，不做桶表选择，也不做真正的 L2 assignment/plan 物化）。

方法（fetch-free：只读 manifest 的 width/height[,holdout,elig_256]，不取图/不解码）：
  对 5 域全量 manifest 逐行做桶指派，用**现行**（非归档）training/dataline/multi_ar.py 的
  build_adaptive_buckets + assign_adaptive_bucket；参数与 domain 分类（square_only_domains）
  全部从 configs/stage256_redesign_naifu_arb_v1.yaml 的 dataset.multi_ar 读出，不硬编码
  （硬约束）。domain_mix 从 dataset.sampling_config 指向的 t5_sampling_redesign_v1.json 读出
  （training.dataline.config.load_sampling_config），只做与任务给定的 stage256 参考值的一致性
  校验，不作为硬编码来源。manifest shard 列表用 training.dataline.manifest_index.list_shards()
  枚举（同一份生产枚举逻辑，含 textatlas 9 子集过滤/REDESIGN per-domain root），不重新拼路径。

domain 特判（均已核对 training/dataline/manifest_index.py 现行代码，不是本脚本自定义简化）：
  - anime：elig_256 生产语义恒为 True（不看宽高，见 load_anime_shard 的
    `"elig_256": np.ones(n, dtype=bool)`），本脚本按此不对 anime 做短边阈值过滤；rich anime
    源无 holdout 列。
  - textatlas：无 holdout 列；elig_256 = min(w,h)>=256 现算（sidecar 只落盘 elig_512/1024）。
  - monet_real/gpic/synthetic（REDESIGN rich_v1/synthetic_v1 root）：有 holdout 列（rule #1，
    holdout==0 才在池内）；无 elig_256 列 → 现算 min(w,h)>=256（与 load_real_synth_shard 的
    REDESIGN 分支完全一致）。

已知简化（不等价于生产 iter_real_synth_shard 的最终采样概率，只影响"哪些行进入 N[B,d] 分母"，
不影响"给定行分到哪个桶"）：不读 sub_source/cluster_id，不算 base_w（source_alpha/cluster_beta
下拉平权重），不做 anime 的 has_nl/composition_clean 类采样期过滤。也就是说本脚本的 N[d] 是
"holdout==0 且 elig_256 的域内全量池"，不是"最终会被采样器抽到的行的精确边际分布"——桶*形状*的
结论（哪些桶有原生供给、AR 分布长什么样）不受影响，但真用于最终 planner 配额求解前，行级权重
仍需在 L1 catalog 阶段补上。

square_only 处理：OFFLINE_ARB_V1_SPEC.md 硬口径#2"非方桶计数构造性为零"——square_only_domains
（从 config 读，本次值为 [synthetic, textatlas]）的行一律强制指派到方桶（square_index()），不跑
AR-based assign_adaptive_bucket；这是新离线设计的绝对约束，不是现有 streaming 代码里那套
MODE-A/MODE-B 软补偿（archive/streaming_arb_v1_20260709 的旧机制仅供参考，不照搬)。

裁剪保留率：retained = min(ar_img,ar_bucket)/max(ar_img,ar_bucket)，ar=W/H。对 resize-to-cover
再居中/锚点裁剪的几何精确推导（cover 到较小边，裁多出的那一维）：当 ar_img>ar_bucket 时
retained=ar_bucket/ar_img，反之retained=ar_img/ar_bucket——恰为 min/max，与
multi_ar.resize_crop_to_bucket 的几何完全一致。"全 square center-crop" baseline 同公式取
ar_bucket=1.0，代表旧方案（无论原生 AR 一律方图裁剪)。

候选剪枝 floor 变体（{0,0.005,0.01,0.02}）：P(B)=Σ_d mix_d·N[B,d]/N[d]（未剪枝的调度概率）；
floor 剪枝=移除 P(B)<floor 的桶，用现行 multi_ar.nearest_allowed_bucket 折进最近保留桶（同现网
bucket_prob_floor 语义）；折后 folded N'[B,d] 与折后 P(B) 重新算；per-cell C6 precheck 是一个粗
筛（非 L3 planner 的整数规划，只是 bs=144 下 P(d|B)=mix_d·N'[B,d]/N[d]/P(B) 的 ceil 是否超过
N'[B,d] 的标量检查，供桶表选择前置排雷用；真正的 C[B,d] 由 build_plan.py 的整数解算给出）。

多进程按 shard 拆分（各域独立 shard 列表拼一份任务队列，共享 64-worker 池；chunksize=1 便于跨
异构 shard 大小的负载均衡）；每 shard worker 用 pyarrow ParquetFile.iter_batches 窄列 batch 读
(width/height[,holdout,elig_256] 几列)，向量化（numpy）算桶指派与 retained 直方图，只把 95 桶
计数向量 + 直方图数组 + 标量计数器回传主进程——单 shard 内存峰值 O(batch_size)，主进程内存峰值
O(95*5 + 2000*5)，与 95.7M 行总量无关。

输出（固定路径，唯一写入位置）：
  /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_bucketstats_v1/
    N_matrix.parquet            —— 95 行(桶) × domain 计数列 + bucket_h/w/ar。
    bucket_decision_report.json —— coverage/retained 分布/floor 变体/C6 precheck 全量结果。
    bucket_decision_summary.md  —— 人读摘要（给 Evan 晨读)。
    run.log                     —— 由 job yaml tee 写出。

诊断缩放（--max-shards-per-domain）产物一律加 .diag 后缀（N_matrix.diag.parquet /
bucket_decision_report.diag.json / bucket_decision_summary.diag.md），绝不覆盖全量 run 的正式
三件套。

本 job 是后续 immutable catalog 的全量输入门禁：任何一个 shard 失败都必须非零退出。
报告仍会保留 failed_shards 诊断，但不允许用部分 N[B,d] 继续构建 catalog。

硬约束：数据源只读；输出仅 --output-dir（固定为上面的规范路径）；无周期 faulthandler；venv
python；不改任何既有文件；PYTHONPATH 指 crop 仓库。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing as mp
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from training.dataline.config import load_sampling_config
from training.dataline.manifest_index import list_shards
from training.dataline.multi_ar import (
    assign_adaptive_bucket,
    build_adaptive_buckets,
    nearest_allowed_bucket,
    square_index,
)
from training.dataline import anchors_v1

BUCKET_MODES = ("adaptive", anchors_v1.BUCKET_MODE)
#: The only stage bucket_mode=anchors_v1 may run under; see phase5 ANCHORS_STAGE.
ANCHORS_STAGE = "512b"

OUTPUT_BASE = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop"
)
CANONICAL_OUTPUT_DIRS = {
    "256": OUTPUT_BASE / "offline_arb_bucketstats_v1",
    "512a": OUTPUT_BASE / "offline_arb_bucketstats_stage512a_v1",
    # Stage512b-real: its own root so a 13-anchor run can never overwrite the frozen
    # 361-bucket stage512a artifacts.
    "512b": OUTPUT_BASE / "offline_arb_bucketstats_stage512b_v1",
}
CANONICAL_OUTPUT_DIR = CANONICAL_OUTPUT_DIRS["256"]
DEFAULT_PIPELINE_CONFIG = "configs/stage256_redesign_naifu_arb_v1.yaml"
DOMAINS = ("anime", "monet_real", "gpic", "synthetic", "textatlas")   # N[B,d] 列序，和 planner_input.json 的 domains 顺序约定一致
NBINS = 2000                                                          # retained∈(0,1] 直方图分辨率 5e-4
FLOORS = (0.0, 0.005, 0.01, 0.02)
LEGACY_STAGE256_BS_PER_RANK = 144
LEGACY_STAGE256_WORLD_SIZE = 16
# 仅作一致性校验用的参考值（Evan 指定，来自 t5_sampling_redesign_v1.json 的 domain_mix["256"]）；
# 真正参与计算的 domain_mix 来自运行时 load_sampling_config() 读取，不由此常量决定。
STAGE256_DOMAIN_MIX_REF = {
    "anime": 0.27, "monet_real": 0.405, "gpic": 0.259, "synthetic": 0.051, "textatlas": 0.015,
}
ELIGIBILITY_FALLBACK_DOMAINS = {
    # Anime uses the historical all-true compatibility path instead of fallback.
    "256": frozenset(("monet_real", "gpic", "synthetic", "textatlas")),
    # Schema audit: the four source manifests derive elig_512 from width/height;
    # TextAtlas is the only enumerated manifest that physically stores elig_512.
    "512a": frozenset(("anime", "monet_real", "gpic", "synthetic")),
    # 512b runs bucket_mode=anchors_v1, which reads no elig_* column at all: the gate is
    # the source area gate plus the feasible-anchor test. The whitelist is unused there,
    # but the key must exist because ELIGIBILITY_FALLBACK_DOMAINS[stage] is indexed
    # unconditionally.
    "512b": frozenset(("anime", "monet_real", "gpic", "synthetic", "textatlas")),
}
ANIME_NL_COLUMNS = (
    "refined_description",
    "medium_description",
    "short_description",
)


class EligibilityContractError(RuntimeError):
    """A manifest omitted the stage eligibility column outside the audited whitelist."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_output_dir(raw: str, stage: str) -> Path:
    path = Path(raw).expanduser().resolve(strict=False)
    expected = CANONICAL_OUTPUT_DIRS.get(stage)
    if expected is None:
        raise ValueError(f"unsupported offline bucket-stats stage {stage!r}")
    if path != expected:
        raise ValueError(
            f"--output-dir 必须是 stage={stage} 的固定路径 {expected}，got {path}"
        )
    return path


# --------------------------------------------------------------------------- #
# pipeline/sampling config 装载（不实例化 CompleteV1Dataset：只要 multi_ar 参数 + shard 枚举)
# --------------------------------------------------------------------------- #
def _strict_config_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def load_pipeline_config(config_path: str) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    stage_raw = OmegaConf.select(cfg, "dataset.stage")
    if not isinstance(stage_raw, str) or not stage_raw:
        raise ValueError(f"dataset.stage must be a non-empty string, got {stage_raw!r}")
    stage = stage_raw
    if stage not in CANONICAL_OUTPUT_DIRS:
        raise ValueError(f"unsupported offline bucket-stats stage {stage!r}")
    image_size = _strict_config_int(
        OmegaConf.select(cfg, "dataset.image_size"), "dataset.image_size"
    )
    expected_image_size = {"256": 256, "512a": 512, "512b": 512}[stage]
    if image_size != expected_image_size:
        raise ValueError(
            f"stage={stage} requires dataset.image_size={expected_image_size}, got {image_size}"
        )
    mar = cfg.dataset.multi_ar
    if OmegaConf.select(cfg, "dataset.multi_ar.enabled") is not True:
        raise ValueError("offline bucket stats require dataset.multi_ar.enabled=true")
    bucket_mode = OmegaConf.select(cfg, "dataset.multi_ar.bucket_mode")
    if bucket_mode not in BUCKET_MODES:
        raise ValueError(
            f"offline bucket stats require multi_ar.bucket_mode in {BUCKET_MODES}, got "
            f"{bucket_mode!r}; the legacy fixed ar_list/bucket_probs table must not be reused"
        )
    required_adaptive = (
        "target_area",
        "divisor",
        "adaptive_no_upscale",
        "adaptive_min_area_ratio",
        "adaptive_max_aspect",
        "square_only_domains",
    )
    missing = [
        key
        for key in required_adaptive
        if OmegaConf.select(cfg, f"dataset.multi_ar.{key}") is None
    ]
    if missing:
        raise ValueError(
            "offline adaptive bucket config is incomplete; refusing legacy fixed-bucket "
            f"statistics: missing {missing}"
        )
    multi_ar_params = {
        "target_area": int(mar.target_area),
        "divisor": int(mar.divisor),
        "no_upscale": bool(mar.adaptive_no_upscale),
        "min_area_ratio": float(mar.adaptive_min_area_ratio),
        "max_aspect": float(mar.adaptive_max_aspect),
    }
    if bucket_mode == anchors_v1.BUCKET_MODE:
        if stage != ANCHORS_STAGE:
            raise ValueError(
                f"bucket_mode={anchors_v1.BUCKET_MODE} requires dataset.stage="
                f"{ANCHORS_STAGE!r}, got {stage!r}"
            )
        # phase5 gates on `report_config["multi_ar_params"] != runtime["mar_params"]`, so
        # the anchors keys must be produced by the SAME expression on both sides or that
        # equality can never hold and anchors mode is unreachable end to end.
        multi_ar_params.update(anchors_v1.params_from_values(
            OmegaConf.select(cfg, "dataset.multi_ar.anchors_area_gate"),
            OmegaConf.select(cfg, "dataset.multi_ar.anchors_crop_loss_max"),
        ))
    if multi_ar_params["target_area"] != image_size * image_size:
        raise ValueError(
            f"stage={stage} image_size={image_size} requires multi_ar.target_area="
            f"{image_size * image_size}, got {multi_ar_params['target_area']}"
        )
    square_only_domains = tuple(str(d) for d in mar.square_only_domains)   # 从 config 读，不硬编码
    if set(square_only_domains) - set(DOMAINS):
        raise ValueError(f"unknown square_only_domains: {square_only_domains}")
    sampling_config_path = str(cfg.dataset.sampling_config)
    return {
        "stage": stage,
        "bucket_mode": bucket_mode,
        "image_size": image_size,
        "eligibility": f"elig_{image_size}",
        "anime_elig_always_true": stage == "256",
        "anime_nl_filter_applied": False,
        "eligibility_fallback_domains": ELIGIBILITY_FALLBACK_DOMAINS[stage],
        "multi_ar_params": multi_ar_params,
        "square_only_domains": square_only_domains,
        "sampling_config_path": sampling_config_path,
    }


def bind_sampling_contract(runtime: dict[str, Any], sampling_cfg: Any) -> None:
    stage = runtime["stage"]
    if stage not in sampling_cfg.domain_mix:
        raise ValueError(f"sampling config has no domain_mix for stage={stage}")
    mix = sampling_cfg.domain_mix[stage]
    if set(mix) != set(DOMAINS):
        raise ValueError(
            f"stage={stage} domain_mix keys must be exactly {list(DOMAINS)}, got {sorted(mix)}"
        )
    total = sum(float(value) for value in mix.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"stage={stage} domain_mix must sum to 1, got {total}")
    requires_nl = bool(sampling_cfg.anime_caption.requires_nl(stage))
    if stage in ("512a", "512b") and requires_nl:
        raise ValueError(
            "stage512a offline catalog contract requires requires_nl=false; "
            "refusing to silently drop tag-only anime rows"
        )
    runtime["requires_nl"] = requires_nl


def resolve_topology(
    stage: str, world_size: int | None, per_rank_bs: int | None
) -> dict[str, Any] | None:
    """Resolve the optional C6 topology without inventing a stage512a batch shape."""
    defaulted = world_size is None and per_rank_bs is None
    if (world_size is None) != (per_rank_bs is None):
        raise ValueError("--world-size and --per-rank-bs must be supplied together")
    if world_size is None:
        if stage == "256":
            world_size = LEGACY_STAGE256_WORLD_SIZE
            per_rank_bs = LEGACY_STAGE256_BS_PER_RANK
        else:
            return None
    if per_rank_bs is None:
        raise ValueError(
            "internal topology contract error: per_rank_bs is missing after paired validation"
        )
    if world_size <= 0 or per_rank_bs <= 0:
        raise ValueError("--world-size and --per-rank-bs must be positive")
    return {
        "world_size": world_size,
        "bs_per_rank": per_rank_bs,
        "global_batch": world_size * per_rank_bs,
        "source": "legacy_stage256_default" if defaulted else "explicit_cli",
    }


def extreme_aspect_sides(target_area: int, divisor: int, max_aspect: float) -> tuple[int, int]:
    """assign_adaptive_bucket 的极端长宽比钳制目标值——只依赖 (target_area,divisor,max_aspect)
    三个标量配置，与具体某行 (w,h) 无关，因此可预计算成两个常量，供整批向量化钳制复用。"""
    long_side = max(divisor, int(math.floor(math.sqrt(target_area * max_aspect) / divisor)) * divisor)
    short_side = max(divisor, int(math.floor((long_side / max_aspect) / divisor)) * divisor)
    return long_side, short_side


def build_bucket_table(buckets: list[tuple[int, int]], divisor: int) -> np.ndarray:
    """(H//divisor, W//divisor) -> bucket_idx 的稠密查表（-1=不在桶表内，需慢路径兜底)。"""
    max_h = max(h for h, _w in buckets)
    max_w = max(w for _h, w in buckets)
    table = np.full((max_h // divisor + 2, max_w // divisor + 2), -1, dtype=np.int64)
    for i, (h, w) in enumerate(buckets):
        table[h // divisor, w // divisor] = i
    return table


def vectorized_adaptive_assign(
    w: np.ndarray, h: np.ndarray, table: np.ndarray, divisor: int, target_area: int,
    no_upscale: bool, max_aspect: float, extreme_long: int, extreme_short: int,
    buckets: list[tuple[int, int]],
) -> np.ndarray:
    """assign_adaptive_bucket 的向量化版本，语义逐行对齐 multi_ar.assign_adaptive_bucket：
    1) 按 target_area 只降采样（raw_area>target_area 时）到 divisor 对齐网格；
    2) 极端长宽比钳制到 (extreme_long,extreme_short) 常量对；
    3) 精确匹配桶表；极少数落在 min_area 边界外的行（桶表未命中）直接回退调用**生产** scalar
       assign_adaptive_bucket(w,h,...)（同一份函数、同一组参数，对原始 w/h 重算一遍），保证与
       现行 multi_ar.py 逻辑逐行位级一致，不在本脚本里重复实现/可能漂移的最近桶搜索。这条慢路径
       只覆盖极少数边界行，逐行调用的开销可忽略。
    """
    raw_area = w * h
    scale = np.where(raw_area > target_area, np.sqrt(target_area / raw_area), 1.0)
    fw = w * scale
    fh = h * scale
    if no_upscale:
        W = np.maximum(divisor, np.floor(fw / divisor).astype(np.int64) * divisor)
        H = np.maximum(divisor, np.floor(fh / divisor).astype(np.int64) * divisor)
    else:
        W = np.maximum(divisor, np.round(fw / divisor).astype(np.int64) * divisor)
        H = np.maximum(divisor, np.round(fh / divisor).astype(np.int64) * divisor)
    aspect = W / H
    over = aspect > max_aspect
    under = aspect < (1.0 / max_aspect)
    H = np.where(over, extreme_short, np.where(under, extreme_long, H)).astype(np.int64)
    W = np.where(over, extreme_long, np.where(under, extreme_short, W)).astype(np.int64)

    hi = H // divisor
    wi = W // divisor
    n = w.shape[0]
    bucket_idx = np.full(n, -1, dtype=np.int64)
    valid = (hi >= 0) & (hi < table.shape[0]) & (wi >= 0) & (wi < table.shape[1])
    bucket_idx[valid] = table[hi[valid], wi[valid]]

    miss = np.where(bucket_idx < 0)[0]
    for i in miss:   # 极少数(min_area 边界之外)才落这条慢路径：直接复用生产 scalar 函数，零逻辑重复
        bucket_idx[i] = assign_adaptive_bucket(
            int(round(float(w[i]))), int(round(float(h[i]))), buckets, target_area, divisor,
            no_upscale, max_aspect,
        )
    return bucket_idx


# --------------------------------------------------------------------------- #
# 每 shard worker（子进程执行；只回传紧凑聚合数组，不回传逐行数据)
# --------------------------------------------------------------------------- #
def _to_float_array(batch: pa.RecordBatch, name: str) -> np.ndarray:
    arr = batch.column(name)
    try:
        return arr.to_numpy(zero_copy_only=False).astype(np.float64)
    except Exception:  # noqa: BLE001 - 含 null 等罕见 schema 情况兜底走慢路径
        return np.array([float(x) if x is not None else np.nan for x in arr.to_pylist()], dtype=np.float64)


def _to_bool_array(batch: pa.RecordBatch, name: str) -> np.ndarray:
    arr = batch.column(name)
    try:
        return arr.to_numpy(zero_copy_only=False).astype(bool)
    except Exception:  # noqa: BLE001
        return np.array([bool(x) if x is not None else False for x in arr.to_pylist()], dtype=bool)


def _to_int_array(batch: pa.RecordBatch, name: str) -> np.ndarray:
    arr = batch.column(name)
    try:
        return arr.to_numpy(zero_copy_only=False).astype(np.int64)
    except Exception:  # noqa: BLE001
        return np.array([int(x) if x is not None else -1 for x in arr.to_pylist()], dtype=np.int64)


def _to_nonempty_string_array(batch: pa.RecordBatch, name: str) -> np.ndarray:
    """Match production ``bool(value)`` semantics for nullable caption strings."""
    values = pc.fill_null(pc.not_equal(batch.column(name), ""), False)
    return values.to_numpy(zero_copy_only=False).astype(bool)


def scan_domain_shard(task: dict[str, Any]) -> dict[str, Any]:
    domain = task["domain"]
    shard_path = task["shard_path"]
    is_square_only = task["is_square_only"]
    elig_always_true = task["elig_always_true"]
    eligibility_column = task["eligibility_column"]
    image_size = task["image_size"]
    fallback_allowed = task["fallback_allowed"]
    buckets_h = task["buckets_h"]
    buckets_w = task["buckets_w"]
    table = task["table"]
    buckets_list = task["buckets_list"]
    sq_idx = task["sq_idx"]
    target_area = task["target_area"]
    divisor = task["divisor"]
    no_upscale = task["no_upscale"]
    max_aspect = task["max_aspect"]
    extreme_long = task["extreme_long"]
    extreme_short = task["extreme_short"]
    batch_size = task["batch_size"]
    anchors_mode = task["bucket_mode"] == anchors_v1.BUCKET_MODE
    area_gate = task.get("area_gate")
    crop_loss_max = task.get("crop_loss_max") or anchors_v1.CROP_LOSS_MAX
    n_buckets = len(buckets_h)

    n_matrix = np.zeros(n_buckets, dtype=np.int64)
    hist_native = np.zeros(NBINS, dtype=np.int64)
    hist_square = np.zeros(NBINS, dtype=np.int64)
    rows_raw = 0
    rows_holdout_dropped = 0
    rows_elig_dropped = 0
    rows_invalid_wh_dropped = 0
    rows_unservable_dropped = 0
    rows_kept = 0
    anime_has_nl_by_bucket = np.zeros(n_buckets, dtype=np.int64)
    anime_tag_only_by_bucket = np.zeros(n_buckets, dtype=np.int64)

    try:
        pf = pq.ParquetFile(shard_path)
        names = set(pf.schema_arrow.names)
        cols = ["width", "height"]
        has_holdout = "holdout" in names
        # anchors mode reads no elig_* column: the gate is the source area gate plus the
        # feasible-anchor test. Consulting the schema here would raise
        # EligibilityContractError over a column this path never touches.
        has_elig = (not anchors_mode) and (not elig_always_true) and (eligibility_column in names)
        if anchors_mode:
            eligibility_mode = "anchors_v1_area_gate_and_feasible_anchor"
        elif elig_always_true:
            eligibility_mode = "stage256_anime_all_true_compatibility"
        elif has_elig:
            eligibility_mode = "column"
        elif fallback_allowed:
            eligibility_mode = "short_dim_fallback"
        else:
            raise EligibilityContractError(
                f"domain={domain} shard={shard_path} is missing required {eligibility_column}"
            )
        if has_holdout:
            cols.append("holdout")
        if has_elig:
            cols.append(eligibility_column)
        if domain == "anime":
            missing_nl = set(ANIME_NL_COLUMNS) - names
            if missing_nl:
                raise EligibilityContractError(
                    f"anime shard={shard_path} missing NL availability columns {sorted(missing_nl)}"
                )
            cols.extend(ANIME_NL_COLUMNS)
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols, use_threads=False):
            n = batch.num_rows
            rows_raw += n
            w = _to_float_array(batch, "width")
            h = _to_float_array(batch, "height")
            if has_holdout:
                holdout = _to_int_array(batch, "holdout")
                holdout_keep = holdout == 0
            else:
                holdout_keep = np.ones(n, dtype=bool)
            rows_holdout_dropped += int((~holdout_keep).sum())

            if anchors_mode:
                elig = (
                    np.ones(n, dtype=bool) if area_gate is None
                    else (w * h) >= float(area_gate)
                )
            elif elig_always_true:
                # Historical stage256 production semantics: anime elig_256 is all true.
                elig = np.ones(n, dtype=bool)
            elif has_elig:
                elig = _to_bool_array(batch, eligibility_column)
            else:
                short_dim = np.minimum(w, h)
                elig = short_dim >= float(image_size)

            wh_valid = np.isfinite(w) & np.isfinite(h) & (w > 0) & (h > 0)
            if anchors_mode:
                # anchors_v1 raises above MAX_DIM so its exact integer arithmetic cannot
                # overflow; screen those rows out rather than let one absurd manifest row
                # abort a full-pool scan.
                wh_valid &= (w < anchors_v1.MAX_DIM) & (h < anchors_v1.MAX_DIM)
            keep = holdout_keep & elig & wh_valid
            rows_elig_dropped += int((holdout_keep & ~elig).sum())
            rows_invalid_wh_dropped += int((holdout_keep & elig & ~wh_valid).sum())
            kept_n = int(keep.sum())
            if kept_n == 0:
                continue
            wk = w[keep]
            hk = h[keep]

            if anchors_mode:
                # Run the real assignment for every domain, square-only included:
                # force-pinning square_idx without a feasibility test would upsample any
                # row that cannot cover the square anchor.
                bidx = anchors_v1.assign_anchor_batch(
                    np.rint(wk).astype(np.int64), np.rint(hk).astype(np.int64),
                    crop_loss_max=crop_loss_max, area_gate=area_gate,
                )[0]
                if is_square_only:
                    bidx = np.where(bidx == sq_idx, bidx, np.int64(-1))
                servable = bidx >= 0
                n_unserv = int((~servable).sum())
                if n_unserv:
                    rows_unservable_dropped += n_unserv
                    # `keep` must be narrowed too, not just wk/hk/bidx: `has_nl_all[keep]`
                    # below reads it, so leaving it wide would pair anime NL flags with
                    # the wrong rows.
                    kept_pos = np.flatnonzero(keep)
                    keep = keep.copy()
                    keep[kept_pos[~servable]] = False
                    wk = wk[servable]
                    hk = hk[servable]
                    bidx = bidx[servable]
                    kept_n = int(keep.sum())
                    if kept_n == 0:
                        continue
            elif is_square_only:
                bidx = np.full(kept_n, sq_idx, dtype=np.int64)
            else:
                bidx = vectorized_adaptive_assign(
                    wk, hk, table, divisor, target_area, no_upscale, max_aspect,
                    extreme_long, extreme_short, buckets_list,
                )
            rows_kept += kept_n
            n_matrix += np.bincount(bidx, minlength=n_buckets)[:n_buckets]

            if domain == "anime":
                has_nl_all = np.zeros(n, dtype=bool)
                for column in ANIME_NL_COLUMNS:
                    has_nl_all |= _to_nonempty_string_array(batch, column)
                has_nl = has_nl_all[keep]
                anime_has_nl_by_bucket += np.bincount(
                    bidx[has_nl], minlength=n_buckets
                )[:n_buckets]
                anime_tag_only_by_bucket += np.bincount(
                    bidx[~has_nl], minlength=n_buckets
                )[:n_buckets]

            Hb = buckets_h[bidx]
            Wb = buckets_w[bidx]
            ar_img = wk / hk
            ar_bucket = Wb / Hb
            lo = np.minimum(ar_img, ar_bucket)
            hi_ = np.maximum(ar_img, ar_bucket)
            retained = np.where(hi_ > 0, lo / hi_, 1.0)
            bins = np.clip((retained * NBINS).astype(np.int64), 0, NBINS - 1)
            hist_native += np.bincount(bins, minlength=NBINS)[:NBINS]

            lo2 = np.minimum(ar_img, 1.0)
            hi2 = np.maximum(ar_img, 1.0)
            retained_sq = np.where(hi2 > 0, lo2 / hi2, 1.0)
            bins2 = np.clip((retained_sq * NBINS).astype(np.int64), 0, NBINS - 1)
            hist_square += np.bincount(bins2, minlength=NBINS)[:NBINS]

        return {
            "ok": True, "domain": domain, "shard_path": shard_path,
            "n_matrix": n_matrix, "hist_native": hist_native, "hist_square": hist_square,
            "rows_raw": rows_raw, "rows_holdout_dropped": rows_holdout_dropped,
            "rows_elig_dropped": rows_elig_dropped, "rows_invalid_wh_dropped": rows_invalid_wh_dropped,
            "rows_unservable_dropped": rows_unservable_dropped,
            "rows_kept": rows_kept,
            "eligibility_column_shards": int(eligibility_mode == "column"),
            "eligibility_fallback_shards": int(eligibility_mode == "short_dim_fallback"),
            "eligibility_fallback_rows": (
                rows_raw if eligibility_mode == "short_dim_fallback" else 0
            ),
            "eligibility_compatibility_shards": int(
                eligibility_mode == "stage256_anime_all_true_compatibility"
            ),
            "eligibility_mode": eligibility_mode,
            "anime_has_nl_by_bucket": anime_has_nl_by_bucket,
            "anime_tag_only_by_bucket": anime_tag_only_by_bucket,
        }
    except Exception as exc:  # noqa: BLE001 - 坏 shard 记录不崩全 job
        return {
            "ok": False, "domain": domain, "shard_path": shard_path,
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=20),
            "fatal_contract_violation": isinstance(exc, EligibilityContractError),
        }


# --------------------------------------------------------------------------- #
# 直方图 → 百分位 / 粗粒度展示直方图
# --------------------------------------------------------------------------- #
def hist_percentiles(hist: np.ndarray, qs: tuple[int, ...] = (5, 25, 50)) -> dict[str, Any]:
    total = int(hist.sum())
    out: dict[str, Any] = {"n": total}
    if total == 0:
        for q in qs:
            out[f"p{q}"] = None
        return out
    cdf = np.cumsum(hist)
    nb = hist.shape[0]
    for q in qs:
        target = q / 100.0 * total
        b = int(np.searchsorted(cdf, target, side="left"))
        b = min(b, nb - 1)
        out[f"p{q}"] = (b + 0.5) / nb
    return out


def coarse_histogram(hist: np.ndarray, n_out_bins: int = 20) -> list[dict[str, float]]:
    total = int(hist.sum())
    nb = hist.shape[0]
    group = nb // n_out_bins
    out = []
    for i in range(n_out_bins):
        lo = i * group
        hi = nb if i == n_out_bins - 1 else (i + 1) * group
        c = int(hist[lo:hi].sum())
        out.append({
            "retained_lo": lo / nb, "retained_hi": hi / nb,
            "count": c, "frac": (c / total) if total else 0.0,
        })
    return out


# --------------------------------------------------------------------------- #
# floor 剪枝变体：折算调度分布 + per-cell C6 可行性预检
# --------------------------------------------------------------------------- #
def compute_bucket_probs(
    n_matrix: np.ndarray, domain_totals: dict[str, int], domain_mix: dict[str, float],
    domains: tuple[str, ...],
) -> np.ndarray:
    probs = np.zeros(n_matrix.shape[0], dtype=np.float64)
    for j, d in enumerate(domains):
        nd = domain_totals.get(d, 0)
        if nd <= 0:
            continue
        probs += domain_mix.get(d, 0.0) * (n_matrix[:, j] / nd)
    return probs


def fold_matrix(
    n_matrix: np.ndarray, buckets: list[tuple[int, int]], allowed: set[int],
) -> np.ndarray:
    folded = np.zeros_like(n_matrix)
    for b in range(n_matrix.shape[0]):
        target = b if b in allowed else nearest_allowed_bucket(b, buckets, allowed)
        folded[target, :] += n_matrix[b, :]
    return folded


def floor_variant_report(
    floor: float, n_matrix: np.ndarray, buckets: list[tuple[int, int]],
    domain_totals: dict[str, int], domain_mix: dict[str, float], domains: tuple[str, ...],
    topology: dict[str, Any] | None,
) -> dict[str, Any]:
    probs_raw = compute_bucket_probs(n_matrix, domain_totals, domain_mix, domains)
    if floor > 0:
        kept = {b for b in range(len(buckets)) if probs_raw[b] >= floor}
    else:
        kept = {b for b in range(len(buckets)) if probs_raw[b] > 0}
    if not kept:   # 理论不会发生（floor 不应大到连方桶都剪掉）；兜底防空集
        kept = {int(np.argmax(probs_raw))}
    folded = fold_matrix(n_matrix, buckets, kept)
    folded_probs = compute_bucket_probs(folded, domain_totals, domain_mix, domains)
    folded_mass = float(sum(probs_raw[b] for b in range(len(buckets)) if b not in kept))

    ar_values = [buckets[b][1] / buckets[b][0] for b in kept]
    violations: list[dict[str, Any]] = []
    if topology is not None:
        for b in sorted(kept):
            pb = float(folded_probs[b])
            if pb <= 0:
                continue
            for j, d in enumerate(domains):
                nd = domain_totals.get(d, 0)
                if nd <= 0:
                    continue
                p_d_given_b = (domain_mix.get(d, 0.0) * folded[b, j] / nd) / pb
                # C6 uses the full global step demand.  A stage512a topology is
                # deliberately never inferred here; it must be passed explicitly.
                c_t = math.ceil(topology["global_batch"] * p_d_given_b)
                n_bd = int(folded[b, j])
                if c_t > n_bd:
                    violations.append({
                        "bucket_idx": b, "bucket_hw": list(buckets[b]), "domain": d,
                        "c_t": c_t, "N_folded": n_bd, "deficit": c_t - n_bd,
                    })
        c6: dict[str, Any] = {
            "evaluated": True,
            **topology,
            "cells_checked": len(kept) * len(domains),
            "violation_count": len(violations),
            "feasible": len(violations) == 0,
            "violations": violations[:200],
            "violations_truncated": len(violations) > 200,
        }
    else:
        c6 = {
            "evaluated": False,
            "reason": (
                "stage512a topology is intentionally unresolved; supply both "
                "--world-size and --per-rank-bs after Evan approves the topology"
            ),
            "world_size": None,
            "bs_per_rank": None,
            "global_batch": None,
            "cells_checked": 0,
            "violation_count": None,
            "feasible": None,
            "violations": [],
            "violations_truncated": False,
        }
    return {
        "floor": floor,
        "kept_bucket_count": len(kept),
        "kept_bucket_indices": sorted(kept),
        "effective_ar_range": [min(ar_values), max(ar_values)],
        "folded_mass_fraction": folded_mass,
        "folded_bucket_probs": {str(b): float(folded_probs[b]) for b in sorted(kept)},
        "c6_precheck": c6,
    }


def anime_caption_availability_report(
    runtime: dict[str, Any],
    sampling_cfg: Any,
    domain_mix: dict[str, float],
    buckets: list[tuple[int, int]],
    floor_reports: list[dict[str, Any]],
    has_nl_raw: np.ndarray,
    tag_only_raw: np.ndarray,
) -> dict[str, Any]:
    total_has_nl = int(has_nl_raw.sum())
    total_tag_only = int(tag_only_raw.sum())
    total = total_has_nl + total_tag_only
    folded: dict[str, Any] = {}
    combined = np.column_stack((has_nl_raw, tag_only_raw))
    for candidate in floor_reports:
        kept = frozenset(int(value) for value in candidate["kept_bucket_indices"])
        folded_counts = fold_matrix(combined, buckets, set(kept))
        folded[str(candidate["floor"])] = {
            "kept_bucket_indices": sorted(kept),
            "has_nl_by_bucket": [int(value) for value in folded_counts[:, 0]],
            "tag_only_by_bucket": [int(value) for value in folded_counts[:, 1]],
        }

    estimate: dict[str, Any]
    if runtime["stage"] in ("512a", "512b") and total > 0:
        has_weights = sampling_cfg.anime_caption.route_weights(runtime["stage"], True)
        no_nl_weights = sampling_cfg.anime_caption.route_weights(runtime["stage"], False)
        has_weight_sum = sum(float(value) for value in has_weights.values())
        no_nl_weight_sum = sum(float(value) for value in no_nl_weights.values())
        if has_weight_sum <= 0 or no_nl_weight_sum <= 0:
            raise ValueError("anime caption route weights must have positive mass")
        has_nl_tag_route_fraction = float(has_weights.get("tag_medium", 0.0)) / has_weight_sum
        no_nl_tag_route_fraction = float(no_nl_weights.get("tag_medium", 0.0)) / no_nl_weight_sum
        anime_tag_fraction = (
            total_has_nl * has_nl_tag_route_fraction
            + total_tag_only * no_nl_tag_route_fraction
        ) / total
        estimate = {
            "basis": "eligible_pool_count_route_selection_pre_dropout",
            "has_nl_tag_route_fraction": has_nl_tag_route_fraction,
            "tag_only_tag_route_fraction": no_nl_tag_route_fraction,
            "anime_tag_fraction": anime_tag_fraction,
            "all_training_tag_fraction": anime_tag_fraction * domain_mix["anime"],
            "anime_domain_mix": domain_mix["anime"],
            "caption_dropout_config": float(sampling_cfg.anime_caption.dropout),
            "planner_integer_exact_pending": True,
            "limitations": [
                "pool-count estimate only; Phase-4 integer quotas/repeats and row priority are not applied",
                "uses the default has-NL stage route (10% tag) and no-NL route (100% tag); policy-specific route overrides are not applied",
                "reported before quality-gated CFG caption dropout, so it is a route-selection fraction rather than non-empty prompt fraction",
            ],
        }
    else:
        estimate = {
            "basis": "stage256_compatibility_scan_only",
            "anime_tag_fraction": None,
            "all_training_tag_fraction": None,
            "planner_integer_exact_pending": True,
        }
    return {
        "definition": (
            "has_nl = bool(refined_description) or bool(medium_description) or "
            "bool(short_description), exactly matching complete_dataset.iter_anime_shard"
        ),
        "eligibility": runtime["eligibility"],
        "requires_nl": runtime["requires_nl"],
        "anime_nl_filter_applied": runtime["anime_nl_filter_applied"],
        "total": total,
        "has_nl": total_has_nl,
        "tag_only": total_tag_only,
        "has_nl_fraction": (total_has_nl / total) if total else None,
        "tag_only_fraction": (total_tag_only / total) if total else None,
        "raw_bucket_distribution": {
            "has_nl_by_bucket": [int(value) for value in has_nl_raw],
            "tag_only_by_bucket": [int(value) for value in tag_only_raw],
        },
        "folded_bucket_distribution_by_floor": folded,
        "expected_tag_exposure": estimate,
    }


# --------------------------------------------------------------------------- #
# 任务队列构建 + 主流程
# --------------------------------------------------------------------------- #
def build_tasks(
    shards_by_domain: dict[str, list[str]], square_only_domains: tuple[str, ...],
    buckets: list[tuple[int, int]], table: np.ndarray, mar_params: dict[str, Any],
    sq_idx: int, extreme_long: int, extreme_short: int, batch_size: int,
    max_shards_per_domain: int | None, runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    buckets_h = np.array([h for h, _w in buckets], dtype=np.int64)
    buckets_w = np.array([w for _h, w in buckets], dtype=np.int64)
    tasks: list[dict[str, Any]] = []
    for domain in DOMAINS:
        shards = shards_by_domain.get(domain, [])
        if max_shards_per_domain is not None:
            shards = shards[:max_shards_per_domain]
        is_sq = domain in square_only_domains
        for sp in shards:
            tasks.append({
                "domain": domain, "shard_path": sp, "is_square_only": is_sq,
                "elig_always_true": domain == "anime" and runtime["anime_elig_always_true"],
                "eligibility_column": runtime["eligibility"],
                "image_size": runtime["image_size"],
                "fallback_allowed": domain in runtime["eligibility_fallback_domains"],
                "buckets_h": buckets_h, "buckets_w": buckets_w, "table": table,
                "buckets_list": buckets, "sq_idx": sq_idx,
                "target_area": mar_params["target_area"], "divisor": mar_params["divisor"],
                "no_upscale": mar_params["no_upscale"], "max_aspect": mar_params["max_aspect"],
                "extreme_long": extreme_long, "extreme_short": extreme_short,
                "batch_size": batch_size,
                # The worker runs in a separate process and receives this dict, not
                # `runtime`, so the anchors parameters must be threaded through here.
                "bucket_mode": runtime["bucket_mode"],
                "area_gate": mar_params.get("area_gate"),
                "crop_loss_max": mar_params.get("crop_loss_max"),
            })
    return tasks


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_PIPELINE_CONFIG,
                    help="训练 yaml；stage/multi_ar/sampling contract 均从此文件读取")
    ap.add_argument(
        "--stage",
        choices=tuple(CANONICAL_OUTPUT_DIRS),
        default=None,
        help="显式 stage guard；stage512a 必填，且必须与 config dataset.stage 一致",
    )
    ap.add_argument("--output-dir", default=str(CANONICAL_OUTPUT_DIR))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=65536, help="pyarrow iter_batches 窄列 batch 行数")
    ap.add_argument("--max-shards-per-domain", type=int, default=None, help="诊断缩放：每域最多扫描 shard 数")
    ap.add_argument("--floors", default=",".join(str(x) for x in FLOORS),
                    help="逗号分隔候选剪枝 floor 列表")
    ap.add_argument(
        "--world-size", type=int, default=None,
        help="仅用于 C6 决策材料；必须与 --per-rank-bs 成对提供",
    )
    ap.add_argument(
        "--per-rank-bs", type=int, default=None,
        help="仅用于 C6 决策材料；stage512a 不提供时明确记为未评估",
    )
    args = ap.parse_args()
    if args.workers <= 0:
        ap.error("--workers 必须为正")
    if args.batch_size <= 0:
        ap.error("--batch-size 必须为正")
    if args.max_shards_per_domain is not None and args.max_shards_per_domain <= 0:
        ap.error("--max-shards-per-domain 必须为正")
    return args


def main() -> int:
    args = parse_args()
    runtime = load_pipeline_config(args.config)
    if args.stage is not None and args.stage != runtime["stage"]:
        raise ValueError(
            f"--stage={args.stage} does not match config dataset.stage={runtime['stage']}"
        )
    if runtime["stage"] in ("512a", "512b") and args.stage is None:
        raise ValueError(
            f"stage{runtime['stage']} bucket stats require explicit "
            f"--stage {runtime['stage']}"
        )
    output_dir = validate_output_dir(args.output_dir, runtime["stage"])
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_run = args.max_shards_per_domain is not None
    topology = resolve_topology(runtime["stage"], args.world_size, args.per_rank_bs)

    mar_params = runtime["multi_ar_params"]
    square_only_domains = runtime["square_only_domains"]
    sampling_config_path = runtime["sampling_config_path"]
    if runtime["bucket_mode"] == anchors_v1.BUCKET_MODE:
        # anchor_id order IS the bucket index and is the identity key every Stage512b
        # audit is written against, so it must not be re-sorted by aspect ratio the way
        # build_adaptive_buckets() sorts.
        buckets = anchors_v1.anchors_hw()
    else:
        buckets = build_adaptive_buckets(
            target_area=mar_params["target_area"], divisor=mar_params["divisor"],
            min_area_ratio=mar_params["min_area_ratio"], max_aspect=mar_params["max_aspect"],
        )
    sq_idx = square_index(buckets)
    extreme_long, extreme_short = extreme_aspect_sides(
        mar_params["target_area"], mar_params["divisor"], mar_params["max_aspect"]
    )
    table = build_bucket_table(buckets, mar_params["divisor"])

    sampling_cfg = load_sampling_config(sampling_config_path)
    bind_sampling_contract(runtime, sampling_cfg)
    domain_mix = {
        d: float(w) for d, w in sampling_cfg.domain_mix[runtime["stage"]].items()
    }
    domain_mix_mismatch = (
        {
            d: {"config": domain_mix.get(d), "reference": STAGE256_DOMAIN_MIX_REF.get(d)}
            for d in DOMAINS
            if abs(domain_mix.get(d, -1.0) - STAGE256_DOMAIN_MIX_REF.get(d, -2.0)) > 1e-9
        }
        if runtime["stage"] == "256"
        else {}
    )
    if domain_mix_mismatch:
        print(f"[bucket-stats] WARNING domain_mix 与任务给定参考值不一致: {domain_mix_mismatch}", flush=True)

    shards_by_domain = list_shards(sampling_cfg)   # 生产同款枚举；某域为空会在 list_shards 内部 assert

    floors = tuple(float(x) for x in args.floors.split(",") if x.strip() != "")
    if not floors or len(set(floors)) != len(floors):
        raise ValueError("--floors must contain one or more unique values")
    if any(not math.isfinite(value) or value < 0.0 for value in floors):
        raise ValueError("--floors values must be finite and non-negative")

    tasks = build_tasks(
        shards_by_domain, square_only_domains, buckets, table, mar_params, sq_idx,
        extreme_long, extreme_short, args.batch_size, args.max_shards_per_domain, runtime,
    )
    print(f"[bucket-stats] {len(tasks)} shard 任务（{len(DOMAINS)} 域），workers={args.workers}", flush=True)

    n_matrix = np.zeros((len(buckets), len(DOMAINS)), dtype=np.int64)
    hist_native = {d: np.zeros(NBINS, dtype=np.int64) for d in DOMAINS}
    hist_square = {d: np.zeros(NBINS, dtype=np.int64) for d in DOMAINS}
    anime_has_nl_by_bucket = np.zeros(len(buckets), dtype=np.int64)
    anime_tag_only_by_bucket = np.zeros(len(buckets), dtype=np.int64)
    coverage = {
        d: {
            "shards_total": len(shards_by_domain.get(d, [])), "shards_scanned": 0, "shards_failed": 0,
            "rows_raw": 0, "rows_holdout_dropped": 0, "rows_elig_dropped": 0,
            "rows_invalid_wh_dropped": 0, "rows_unservable_dropped": 0, "rows_kept": 0,
            "eligibility_column_shards": 0, "eligibility_fallback_shards": 0,
            "eligibility_fallback_rows": 0, "eligibility_compatibility_shards": 0,
        }
        for d in DOMAINS
    }
    failed_shards: list[dict[str, Any]] = []
    eligibility_contract_failures: list[dict[str, Any]] = []
    domain_index = {d: i for i, d in enumerate(DOMAINS)}

    started = time.perf_counter()
    ctx = mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        for i, res in enumerate(pool.map(scan_domain_shard, tasks, chunksize=1)):
            d = res["domain"]
            if not res["ok"]:
                coverage[d]["shards_failed"] += 1
                failed_shards.append({"domain": d, "shard_path": res["shard_path"], "error": res["error"]})
                if res.get("fatal_contract_violation"):
                    eligibility_contract_failures.append(
                        {"domain": d, "shard_path": res["shard_path"], "error": res["error"]}
                    )
                continue
            coverage[d]["shards_scanned"] += 1
            coverage[d]["rows_raw"] += res["rows_raw"]
            coverage[d]["rows_holdout_dropped"] += res["rows_holdout_dropped"]
            coverage[d]["rows_elig_dropped"] += res["rows_elig_dropped"]
            coverage[d]["rows_invalid_wh_dropped"] += res["rows_invalid_wh_dropped"]
            coverage[d]["rows_unservable_dropped"] += res["rows_unservable_dropped"]
            coverage[d]["rows_kept"] += res["rows_kept"]
            coverage[d]["eligibility_column_shards"] += res["eligibility_column_shards"]
            coverage[d]["eligibility_fallback_shards"] += res["eligibility_fallback_shards"]
            coverage[d]["eligibility_fallback_rows"] += res["eligibility_fallback_rows"]
            coverage[d]["eligibility_compatibility_shards"] += res[
                "eligibility_compatibility_shards"
            ]
            n_matrix[:, domain_index[d]] += res["n_matrix"]
            hist_native[d] += res["hist_native"]
            hist_square[d] += res["hist_square"]
            if d == "anime":
                anime_has_nl_by_bucket += res["anime_has_nl_by_bucket"]
                anime_tag_only_by_bucket += res["anime_tag_only_by_bucket"]
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                print(f"[bucket-stats] progress {i + 1}/{len(tasks)} shard 任务完成", flush=True)
    scan_seconds = time.perf_counter() - started

    domain_totals = {d: int(coverage[d]["rows_kept"]) for d in DOMAINS}
    domain_total_failure = [d for d in DOMAINS if coverage[d]["shards_scanned"] == 0]

    # 诊断缩放（--max-shards-per-domain）产物一律加 .diag 后缀，绝不覆盖全量 run 的正式三件套
    # （N_matrix.parquet / bucket_decision_report.json / bucket_decision_summary.md）。
    suffix = ".diag" if diagnostic_run else ""
    n_matrix_path = output_dir / f"N_matrix{suffix}.parquet"
    report_path = output_dir / f"bucket_decision_report{suffix}.json"
    summary_path = output_dir / f"bucket_decision_summary{suffix}.md"

    # ---- N_matrix.parquet ----
    n_cols: dict[str, Any] = {
        "bucket_idx": pa.array(list(range(len(buckets))), type=pa.int32()),
        "bucket_h": pa.array([h for h, _w in buckets], type=pa.int32()),
        "bucket_w": pa.array([w for _h, w in buckets], type=pa.int32()),
        "bucket_ar": pa.array([w / h for h, w in buckets], type=pa.float64()),
    }
    for d in DOMAINS:
        n_cols[f"n_{d}"] = pa.array(n_matrix[:, domain_index[d]].tolist(), type=pa.int64())
    n_cols["n_total"] = pa.array(n_matrix.sum(axis=1).tolist(), type=pa.int64())
    pq.write_table(pa.table(n_cols), n_matrix_path, compression="zstd")

    # ---- retained 分布 ----
    retained_percentiles = {
        d: {
            "assigned_bucket": hist_percentiles(hist_native[d]),
            "square_baseline": hist_percentiles(hist_square[d]),
        }
        for d in DOMAINS
    }
    retained_hist20 = {
        d: {
            "assigned_bucket": coarse_histogram(hist_native[d]),
            "square_baseline": coarse_histogram(hist_square[d]),
        }
        for d in DOMAINS
    }

    # ---- floor 变体 ----
    floor_reports = [
        floor_variant_report(
            f, n_matrix, buckets, domain_totals, domain_mix, DOMAINS, topology
        )
        for f in floors
    ]
    anime_caption_availability = anime_caption_availability_report(
        runtime,
        sampling_cfg,
        domain_mix,
        buckets,
        floor_reports,
        anime_has_nl_by_bucket,
        anime_tag_only_by_bucket,
    )
    if anime_caption_availability["total"] != domain_totals["anime"]:
        raise RuntimeError(
            "anime has-NL/tag-only accounting does not equal eligible anime rows: "
            f"{anime_caption_availability['total']} != {domain_totals['anime']}"
        )

    pipeline_config_path = Path(args.config).expanduser().resolve(strict=True)
    sampling_config_resolved = Path(sampling_config_path).expanduser().resolve(strict=True)
    n_matrix_sha256 = sha256_file(n_matrix_path)

    report: dict[str, Any] = {
        "schema_version": "offline_arb_bucketstats_v1",
        "generated_at": utc_now(),
        "host": socket.gethostname(),
        "duration_seconds": scan_seconds,
        "diagnostic_run": diagnostic_run,
        "stage": runtime["stage"],
        "image_size": runtime["image_size"],
        "eligibility": runtime["eligibility"],
        "requires_nl": runtime["requires_nl"],
        "anime_nl_filter_applied": runtime["anime_nl_filter_applied"],
        "config": {
            "pipeline_config": str(pipeline_config_path),
            "pipeline_config_sha256": sha256_file(pipeline_config_path),
            "sampling_config": str(sampling_config_resolved),
            "sampling_config_sha256": sha256_file(sampling_config_resolved),
            "stage": runtime["stage"],
            "image_size": runtime["image_size"],
            "eligibility": runtime["eligibility"],
            "requires_nl": runtime["requires_nl"],
            "anime_nl_filter_applied": runtime["anime_nl_filter_applied"],
            "multi_ar_params": mar_params,
            "square_only_domains": list(square_only_domains),
            "domain_mix": domain_mix,
            "domain_mix_reference_mismatch": domain_mix_mismatch or None,
            "c6_topology": topology,
        },
        "N_matrix": {
            "path": str(n_matrix_path.resolve(strict=True)),
            "sha256": n_matrix_sha256,
        },
        "eligibility_audit": {
            "column": runtime["eligibility"],
            "fallback_rule": f"min(width,height)>={runtime['image_size']}",
            "fallback_domain_whitelist": sorted(runtime["eligibility_fallback_domains"]),
            "unexpected_missing_column_failures": eligibility_contract_failures,
            "pass": not eligibility_contract_failures,
        },
        "buckets": [{"idx": i, "h": h, "w": w, "ar": w / h} for i, (h, w) in enumerate(buckets)],
        "square_bucket_idx": sq_idx,
        "coverage": coverage,
        "domain_totals": domain_totals,
        "domain_totals_sum": int(sum(domain_totals.values())),
        "domain_total_scan_failure": domain_total_failure,
        "failed_shards": failed_shards[:200],
        "failed_shards_truncated": len(failed_shards) > 200,
        "failed_shard_count": len(failed_shards),
        "retained_percentiles": retained_percentiles,
        "retained_histograms_20bin": retained_hist20,
        "anime_caption_availability": anime_caption_availability,
        "floors": floor_reports,
        "known_simplifications": [
            f"N[d] 分母 = holdout==0(若该域有此列) 且 {runtime['eligibility']} 的域内全量池，未按 source_alpha/"
            "cluster_beta 做 base_w 降权，不代表最终采样器的精确边际分布(只影响分母权重，不影响"
            "单行的桶指派)。",
            (
                "stage256 compatibility: 未复刻 anime 的 has_nl/composition_clean 等采样期过滤；"
                "anime elig_256 保持生产历史语义恒为 True，且即使 sampling requires_nl=true，"
                "本统计仍未丢弃 tag-only 行。"
                if runtime["stage"] == "256"
                else (
                    "stage512a 使用 elig_512（列存在时直接读取，否则按短边>=512 回退），"
                    "sampling contract 已硬校验 requires_nl=false；不做额外 anime NL 过滤。"
                )
            ),
            "C6 precheck 是标量粗筛(ceil(global_batch*P(d|B))<=N'[B,d])，不是 build_plan.py 的整数"
            "规划解；真正的 C[B,d] 由 Phase-4 planner 给出。",
            "square_only_domains(synthetic/textatlas) 的行强制指派到方桶，其 assigned_bucket 与"
            "square_baseline 的 retained 分布因此逐行恒等（两栏重复不是 bug，是这两域的 AR-preserving"
            "裁剪本就等价于方图裁剪）；两栏的对比意义只对非 square_only 域(anime/monet_real/gpic)成立。",
        ],
    }
    atomic_write_json(report_path, report)
    write_markdown_summary(summary_path, report)

    print("==== OFFLINE_ARB PHASE3a BUCKETSTATS summary ====")
    print(json.dumps({
        "domain_totals": domain_totals, "domain_totals_sum": report["domain_totals_sum"],
        "failed_shard_count": len(failed_shards), "domain_total_scan_failure": domain_total_failure,
        "duration_seconds": scan_seconds,
    }, ensure_ascii=False, indent=2))

    return 0 if not failed_shards else 2


def write_markdown_summary(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# OFFLINE_ARB_V1 Phase-3a 桶表决策材料摘要\n")
    lines.append(f"生成时间(UTC): {report['generated_at']}  \n耗时: {report['duration_seconds']:.1f}s  \n")
    lines.append(
        f"stage/image/eligibility/requires_nl: `"
        f"{report['stage']}/{report['image_size']}/{report['eligibility']}/{report['requires_nl']}`  \n"
    )
    lines.append(f"multi_ar 参数: `{report['config']['multi_ar_params']}`  \n")
    lines.append(f"square_only_domains: `{report['config']['square_only_domains']}`（从 config 读)  \n")
    if report["config"]["domain_mix_reference_mismatch"]:
        lines.append(f"\n**WARNING** domain_mix 与任务参考值不一致: "
                      f"`{report['config']['domain_mix_reference_mismatch']}`\n")

    lines.append("\n## 覆盖率\n")
    lines.append("| domain | shards_total | scanned | failed | elig-column shards | fallback shards | fallback rows | rows_raw | rows_kept |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d, c in report["coverage"].items():
        lines.append(
            f"| {d} | {c['shards_total']} | {c['shards_scanned']} | {c['shards_failed']} | "
            f"{c['eligibility_column_shards']} | {c['eligibility_fallback_shards']} | "
            f"{c['eligibility_fallback_rows']:,} | {c['rows_raw']:,} | {c['rows_kept']:,} |"
        )
    lines.append(f"\n合计 kept 行数: **{report['domain_totals_sum']:,}**"
                  f"（对照 Phase-1 D 项 95.7M 量级 uid 全量口径)\n")

    anime = report["anime_caption_availability"]
    estimate = anime["expected_tag_exposure"]
    lines.append("\n## Anime NL/tag-only 可用性（eligible pool 精确计数）\n")
    has_fraction = anime["has_nl_fraction"]
    tag_fraction = anime["tag_only_fraction"]
    lines.append(
        f"- has-NL: **{anime['has_nl']:,}** ("
        f"{'n/a' if has_fraction is None else format(has_fraction, '.4%')})"
    )
    lines.append(
        f"- tag-only: **{anime['tag_only']:,}** ("
        f"{'n/a' if tag_fraction is None else format(tag_fraction, '.4%')})"
    )
    if estimate.get("anime_tag_fraction") is not None:
        lines.append(
            f"- 默认 route、dropout 前的 pool-count 估算: anime 内 tag "
            f"**{estimate['anime_tag_fraction']:.4%}**；乘 anime mix="
            f"{estimate['anime_domain_mix']:.4f} 后全训练 **{estimate['all_training_tag_fraction']:.4%}**"
        )
        lines.append("- 这不是最终曝光精确值；需等 planner 整数解/repeat/priority 后闭合。")

    lines.append("\n## 裁剪保留率分布（p5/p25/p50，assigned_bucket vs square_baseline）\n")
    lines.append("| domain | assigned p5 | assigned p25 | assigned p50 | square p5 | square p25 | square p50 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for d in report["retained_percentiles"]:
        a = report["retained_percentiles"][d]["assigned_bucket"]
        s = report["retained_percentiles"][d]["square_baseline"]
        def fmt(x):
            return "n/a" if x is None else f"{x:.3f}"
        lines.append(f"| {d} | {fmt(a['p5'])} | {fmt(a['p25'])} | {fmt(a['p50'])} | "
                      f"{fmt(s['p5'])} | {fmt(s['p25'])} | {fmt(s['p50'])} |")

    lines.append("\n## 候选剪枝 floor 变体\n")
    lines.append("| floor | kept_buckets | effective_ar_range | folded_mass_fraction | "
                  "C6 cells | C6 violations | C6 feasible |")
    lines.append("|---:|---:|---|---:|---:|---:|---|")
    for fr in report["floors"]:
        ar_lo, ar_hi = fr["effective_ar_range"]
        c6 = fr["c6_precheck"]
        cells = c6["cells_checked"] if c6["evaluated"] else "n/e"
        violations = c6["violation_count"] if c6["evaluated"] else "n/e"
        feasible = c6["feasible"] if c6["evaluated"] else "not evaluated"
        lines.append(f"| {fr['floor']} | {fr['kept_bucket_count']} | "
                      f"[{ar_lo:.3f}, {ar_hi:.3f}] | {fr['folded_mass_fraction']:.4f} | "
                      f"{cells} | {violations} | {feasible} |")

    lines.append("\n## 已知简化（决策前请复核)\n")
    for note in report["known_simplifications"]:
        lines.append(f"- {note}")

    if report["failed_shard_count"]:
        lines.append(f"\n## 坏 shard（共 {report['failed_shard_count']}，列前 20)\n")
        for f in report["failed_shards"][:20]:
            lines.append(f"- {f['domain']} {f['shard_path']}: {f['error']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
