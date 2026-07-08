"""multi-AR（多宽高比）分桶核心（REDESIGN Phase2b，§7）。

设计约束（已由 model-side 审计确认）：
  - 模型零改动：MRoPE/patchify/VAE 都按 grid_h/grid_w 独立处理任意网格（arch_mot_utils.py）。
  - batch 必须【单桶】：模型每 batch 取一个 grid（model_mot.py），attention_mask[B,1,S,S]/position_ids[3,B,S]
    单 S → batch 内所有样本须同 (H,W)。不能 packing，只能分桶。
  - rank 一致（FSDP2）：每 step 所有 rank 必须用【同一桶 shape】否则 collective desync。
    → 桶调度必须确定性 + 跨 rank 同种子（BucketSchedule.pick 只吃 base_seed+step，不掺 rank）。
    → loader 不做任何 per-batch collective（会与 continue-on-empty 路径错位）；rank 一致靠确定性调度，
      供给缺口靠 re-resize 兜底（保证每 rank 都出 bs 个 B_t shape）。

本模块只做纯逻辑（桶定义/分配/缩放/调度），可零模型加载单测。
"""
from __future__ import annotations
import math
import numpy as np
from PIL import Image

from .image_fetch import crop_offset, resize_crop_uint8
from .pil_safety import PIL_NATIVE_LOCK


def build_buckets(target_area: int = 262144, divisor: int = 16,
                  ar_list: list[float] | None = None) -> list[tuple[int, int]]:
    """生成等面积 AR 桶 [(H,W),...]：1:1 + 每个 ar(w/h>1) 的正反（横+竖）。每边整除 divisor、面积≈target_area。
    默认 ar_list=[4/3,3/2,16/9] → 7 桶（Evan 2026-06-30 拍板：1:1 + 4:3± + 3:2± + 16:9±）。
    返回按 W/H 升序（竖→方→横）。"""
    if ar_list is None:
        ar_list = [4.0 / 3.0, 3.0 / 2.0, 16.0 / 9.0]
    seen: dict[tuple[int, int], None] = {}

    def add(r: float):
        w = math.sqrt(target_area * r)
        h = math.sqrt(target_area / r)
        W = max(divisor, int(round(w / divisor)) * divisor)
        H = max(divisor, int(round(h / divisor)) * divisor)
        seen[(H, W)] = None

    add(1.0)
    for r in ar_list:
        add(r)
        add(1.0 / r)
    return sorted(seen.keys(), key=lambda hw: hw[1] / hw[0])   # 按 W/H 升序


def build_adaptive_buckets(target_area: int = 262144, divisor: int = 16,
                           min_area_ratio: float = 0.75,
                           max_aspect: float = 4.0) -> list[tuple[int, int]]:
    """NAI-like dense bucket table: enumerate all divisor-aligned shapes near target area.

    This mirrors the flat bucket-table idea used by naifu: item assignment is based on the
    image's own aspect ratio, then batches are grouped by exact target shape.  We keep the
    table bounded for streaming training by requiring area within [min_area_ratio, 1.0] of
    target_area and aspect within [1/max_aspect, max_aspect].
    """
    target_area = int(target_area)
    divisor = int(divisor)
    if target_area <= 0 or divisor <= 0:
        raise ValueError("target_area and divisor must be positive")
    min_area = max(divisor * divisor, int(round(target_area * float(min_area_ratio))))
    max_aspect = float(max_aspect)
    if max_aspect < 1.0:
        raise ValueError("max_aspect must be >= 1")
    max_side = max(divisor, int(math.floor(math.sqrt(target_area * max_aspect) / divisor)) * divisor)
    seen: dict[tuple[int, int], None] = {}
    for h in range(divisor, max_side + divisor, divisor):
        for w in range(divisor, max_side + divisor, divisor):
            area = h * w
            if area < min_area or area > target_area:
                continue
            aspect = w / h
            if aspect < 1.0 / max_aspect or aspect > max_aspect:
                continue
            seen[(h, w)] = None
    side = max(divisor, int(round(math.sqrt(target_area) / divisor)) * divisor)
    seen[(side, side)] = None
    return sorted(seen.keys(), key=lambda hw: (hw[1] / hw[0], hw[0] * hw[1], hw[0], hw[1]))


def square_index(buckets: list[tuple[int, int]]) -> int:
    """方图桶（H==W）下标；adaptive 表可能有多个方桶，取面积最大的。"""
    square = [(i, h * w) for i, (h, w) in enumerate(buckets) if h == w]
    if square:
        return max(square, key=lambda x: x[1])[0]
    return min(range(len(buckets)), key=lambda i: abs(math.log(buckets[i][1] / buckets[i][0])))


def assign_bucket(w: int, h: int, buckets: list[tuple[int, int]]) -> int:
    """native (w,h) → 最近邻桶 index（log-AR 距离最小）。非法尺寸回退方图。"""
    if not w or not h or w <= 0 or h <= 0:
        return square_index(buckets)
    lar = math.log(w / h)
    return min(range(len(buckets)),
               key=lambda i: abs(math.log(buckets[i][1] / buckets[i][0]) - lar))


def assign_adaptive_bucket(w: int, h: int, buckets: list[tuple[int, int]],
                           target_area: int, divisor: int = 16,
                           no_upscale: bool = True,
                           max_aspect: float = 4.0) -> int:
    """NAI-like item assignment: compute the item's divisor-aligned target shape, then map
    to the nearest bucket in the flat table.

    naifu's current path scales down only when raw_area > target_area, rounds to the
    divisibility grid, and groups equal target_h/target_w together.  We do the same, with
    aspect clamping for extreme images so they cannot create pathological buckets.
    """
    if not w or not h or w <= 0 or h <= 0:
        return square_index(buckets)
    target_area = int(target_area)
    divisor = int(divisor)
    max_aspect = float(max_aspect)
    raw_area = float(w * h)
    scale = math.sqrt(target_area / raw_area) if raw_area > target_area else 1.0
    fw = w * scale
    fh = h * scale
    if no_upscale:
        W = max(divisor, int(math.floor(fw / divisor)) * divisor)
        H = max(divisor, int(math.floor(fh / divisor)) * divisor)
    else:
        W = max(divisor, int(round(fw / divisor)) * divisor)
        H = max(divisor, int(round(fh / divisor)) * divisor)
    aspect = W / H
    if aspect > max_aspect:
        W = max(divisor, int(math.floor(math.sqrt(target_area * max_aspect) / divisor)) * divisor)
        H = max(divisor, int(math.floor((W / max_aspect) / divisor)) * divisor)
    elif aspect < 1.0 / max_aspect:
        H = max(divisor, int(math.floor(math.sqrt(target_area * max_aspect) / divisor)) * divisor)
        W = max(divisor, int(math.floor((H / max_aspect) / divisor)) * divisor)
    target = (H, W)
    try:
        return buckets.index(target)
    except ValueError:
        lar = math.log(W / H)
        return min(range(len(buckets)),
                   key=lambda i: (abs(math.log(buckets[i][1] / buckets[i][0]) - lar),
                                  abs(buckets[i][0] * buckets[i][1] - H * W)))


def resize_crop_to_bucket(img: Image.Image, H: int, W: int, v_anchor: float = 0.5) -> np.ndarray:
    """AR-preserving：scale 使【覆盖】(W,H)（短边匹配、不上采放大超过覆盖），crop 到 (W,H)。
    返回 uint8 HWC [H,W,3]。对齐 resize_crop_uint8 的"仅必要缩放 + 竖直锚点裁 / 水平居中"语义，但目标非方。
    v_anchor：竖直裁剪锚点∈[0,1]（0=贴顶保头 / 0.5=居中=旧行为 / 1=贴底）；水平方向仍居中。默认 0.5 逐字节不变。
    （C+B 关键组合：portrait 图被兜到方/横桶时竖直会裁掉 ~22%，anchor<0.5 保头，修残余缺头漏洞。）"""
    with PIL_NATIVE_LOCK:
        w0, h0 = img.size
        scale = max(W / w0, H / h0)                   # 覆盖目标框
        nw = max(W, int(round(w0 * scale)))
        nh = max(H, int(round(h0 * scale)))
        if (nw, nh) != (w0, h0):
            img = img.resize((nw, nh), Image.LANCZOS)
        left = max(0, (nw - W) // 2)
        top = crop_offset(nh - H, v_anchor)           # 竖直锚点（默认 0.5=旧 center-crop，字节不变）
        img = img.crop((left, top, left + W, top + H))
        if img.size != (W, H):                        # 兜底极端
            img = img.resize((W, H), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8)


def resize_u8_to_bucket(u8: np.ndarray, H: int, W: int, v_anchor: float = 0.5) -> np.ndarray:
    """已存 u8（HWC，某桶 shape）→ 重缩到 (H,W)（re-resize 兜底用：调度桶在本 rank buffer 无 native 供给时）。
    v_anchor 透传给竖直裁剪锚点（默认 0.5 逐字节不变；缺头修复走该域 anchor 保头）。"""
    if u8.shape[0] == H and u8.shape[1] == W:
        return u8
    with PIL_NATIVE_LOCK:
        return resize_crop_to_bucket(Image.fromarray(u8), H, W, v_anchor)


class BucketSchedule:
    """rank 一致的确定性桶调度。B_t = f(base_seed, step)，按 probs 加权。collective-free。
    ★base_seed 必须跨 rank 相同（不掺 rank）→ 所有 rank 每 step 选同一桶 → FSDP shape 对齐。"""

    def __init__(self, n_buckets: int, square_idx: int, probs: list[float] | None = None,
                 base_seed: int = 0):
        self.n = int(n_buckets)
        self.square_idx = int(square_idx)
        self.base_seed = int(base_seed)
        if probs is None:
            probs = self._default_probs(n_buckets, square_idx)
        p = np.asarray(probs, dtype=np.float64)
        assert p.shape == (self.n,) and (p >= 0).all() and p.sum() > 0, f"bad bucket_probs {probs}"
        self.probs = p / p.sum()
        self.cum = np.cumsum(self.probs)

    @staticmethod
    def _default_probs(n: int, square_idx: int) -> list[float]:
        """默认 square-heavy 先验（无数据 AR 分布时的兜底；真实分布建议由 config.bucket_probs 给）。
        square 0.40，其余均分 0.60。"""
        if n <= 1:
            return [1.0]
        rest = 0.60 / (n - 1)
        return [0.40 if i == square_idx else rest for i in range(n)]

    def pick(self, step: int) -> int:
        """确定性选桶：同 (base_seed, step) → 同结果（跨 rank 一致）。"""
        rng = np.random.default_rng(self.base_seed * 1_000_003 + int(step))
        return int(np.searchsorted(self.cum, rng.random(), side="right"))


class Bucketer:
    """producer 侧取图缩放器：multi_ar 激活(过 warmup)→按 native AR 分桶 + AR-preserving 缩放；
    warmup 期 / 未激活 → 方图 image_size（与旧行为一致）。读 ds._multi_ar_active 实时判断（warmup 软切换）。"""

    def __init__(self, buckets: list[tuple[int, int]], square_idx: int, ds,
                 adaptive: bool = False, target_area: int = 262144,
                 divisor: int = 16, no_upscale: bool = True,
                 max_aspect: float = 4.0):
        self.buckets = buckets
        self.square_idx = int(square_idx)
        self.ds = ds
        self.adaptive = bool(adaptive)
        self.target_area = int(target_area)
        self.divisor = int(divisor)
        self.no_upscale = bool(no_upscale)
        self.max_aspect = float(max_aspect)

    def resize(self, img: Image.Image, image_size: int, v_anchor: float = 0.5):
        """→ (u8 HWC, bucket_idx)。v_anchor 透传给竖直裁剪锚点（默认 0.5 逐字节不变；缺头修复走该域 anchor 保头）。
        active(过 warmup) → native AR 分桶裁；warmup 期 → 方图 image_size。两路都应用 v_anchor 竖直锚点。"""
        if getattr(self.ds, "_multi_ar_active", False):
            if self.adaptive:
                aspect = img.width / max(img.height, 1)
                if aspect > self.max_aspect or aspect < 1.0 / self.max_aspect:
                    try:
                        self.ds._mar_extreme_clamped += 1
                    except AttributeError:
                        pass
                bidx = assign_adaptive_bucket(img.width, img.height, self.buckets,
                                              self.target_area, self.divisor,
                                              self.no_upscale, self.max_aspect)
            else:
                bidx = assign_bucket(img.width, img.height, self.buckets)
            H, W = self.buckets[bidx]
            return resize_crop_to_bucket(img, H, W, v_anchor), bidx
        return resize_crop_uint8(img, image_size, v_anchor), self.square_idx
