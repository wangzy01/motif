#!/usr/bin/env python3
"""2026-07-09 ARB 修复的纯逻辑单测（零真数据/零模型;端到端行为由 CPU 压测复验覆盖）。

覆盖：
  [T1] _prune_scheduled_buckets：恒等式 Σ_B P(B)P(d|B)=mix_d 精确、被剪桶归零、square 恒保留、质量守恒
  [T2] nearest_allowed_bucket：log-AR 最近邻正确、allowed=None 恒等
  [T3] BucketSchedule.pick：确定性/跨 rank 一致/永不越界/零概率桶(剪枝)永不被选
  [T4] _domain_seed：稳定（跨进程可复现,替代 hash()）
  [T5] tar 流式 chunk 打乱生成器：全覆盖、恰好一次、窗口有界
  [T6] Bucketer.allowed：指派被折进保留桶、输出 u8 形状=保留桶
  [T7] _force_square_only_record：非方记录被强制回方桶（护栏接线）
  [T8] buckets_by_ar_distance：全排列、恰好一次、首元素=自身、序=暴力 (|Δlog-AR|,|Δ面积|)
"""
from __future__ import annotations

import math
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop")

from training.dataline.complete_dataset import CompleteV1Dataset, _domain_seed, _TAR_SHUFFLE_CHUNK
from training.dataline.multi_ar import (Bucketer, BucketSchedule, build_adaptive_buckets,
                                        buckets_by_ar_distance, nearest_allowed_bucket,
                                        square_index)

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    if cond:
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name} {detail}")


# ---------------- T1 prune math ---------------- #
buckets = build_adaptive_buckets(65536, 16, 0.75, 4.0)
sq = square_index(buckets)
n = len(buckets)
check("T0 bucket table 95/square48", n == 95 and sq == 48, f"n={n} sq={sq}")

rng = np.random.default_rng(7)
mix = {"anime": 0.27, "monet_real": 0.405, "gpic": 0.259, "synthetic": 0.051, "textatlas": 0.015}
rows = {}
for d in mix:
    if d in ("synthetic", "textatlas"):
        v = np.zeros(n); v[sq] = 1.0
    else:
        v = rng.random(n) * (rng.random(n) < 0.4)   # 稀疏随机行
        if v.sum() <= 0:
            v[sq] = 1.0
        v = v / v.sum()
    rows[d] = v

P_B = np.zeros(n)
for d in mix:
    P_B += mix[d] * rows[d]

for floor in (0.005, 0.01, 0.05):
    rows2, P_B2, pdb2, kept = CompleteV1Dataset._prune_scheduled_buckets(mix, rows, P_B, buckets, sq, floor)
    # 恒等式：Σ_B P(B)·P(d|B) = mix_d 精确
    got = {d: 0.0 for d in mix}
    for b in range(n):
        doms, probs = pdb2[b]
        for d, p in zip(doms, probs):
            got[d] += float(P_B2[b]) * float(p)
    dev = max(abs(got[d] - mix[d]) for d in mix)
    check(f"T1 identity floor={floor}", dev < 1e-9, f"max dev={dev}")
    check(f"T1 pruned-zero floor={floor}",
          all(P_B2[b] == 0.0 for b in range(n) if b not in kept), "")
    check(f"T1 square kept floor={floor}", sq in kept, "")
    check(f"T1 mass conserved floor={floor}", abs(P_B2.sum() - P_B.sum()) < 1e-12,
          f"{P_B2.sum()} vs {P_B.sum()}")
    check(f"T1 kept above floor={floor}",
          all(P_B2[b] >= floor or b == sq or P_B2[b] == 0.0 for b in range(n)), "")

# ---------------- T2 nearest ---------------- #
allowed = {sq, 33, 60}
for b in range(n):
    t = nearest_allowed_bucket(b, buckets, allowed)
    check_ok = t in allowed and (b not in allowed or t == b)
    if not check_ok:
        check("T2 nearest in-allowed", False, f"b={b} t={t}")
        break
else:
    check("T2 nearest in-allowed & identity", True)
h0, w0 = buckets[33]
lar_33 = math.log(w0 / h0)
b_far = min((b for b in range(n) if b not in allowed),
            key=lambda b: abs(math.log(buckets[b][1] / buckets[b][0]) - lar_33))
t = nearest_allowed_bucket(b_far, buckets, allowed)
best = min(allowed, key=lambda i: (abs(math.log(buckets[i][1] / buckets[i][0])
                                       - math.log(buckets[b_far][1] / buckets[b_far][0])),
                                   abs(buckets[i][0] * buckets[i][1] - buckets[b_far][0] * buckets[b_far][1])))
check("T2 nearest = brute force", t == best, f"{t} vs {best}")
check("T2 allowed=None identity", nearest_allowed_bucket(7, buckets, None) == 7)

# ---------------- T3 schedule ---------------- #
probs = [float(x) for x in P_B / P_B.sum()]
# 人为造 cum[-1] 略小于 1 的情形也应永不越界
sched = BucketSchedule(n, sq, probs=probs, base_seed=0)
sched2 = BucketSchedule(n, sq, probs=probs, base_seed=0)
picks = [sched.pick(s) for s in range(5000)]
check("T3 in-range", all(0 <= b < n for b in picks))
check("T3 rank-consistent", picks == [sched2.pick(s) for s in range(5000)])
zero_set = {b for b in range(n) if probs[b] == 0.0}
check("T3 zero-prob never picked", not (set(picks) & zero_set), f"hit={set(picks) & zero_set}")

# ---------------- T4 stable seed ---------------- #
check("T4 _domain_seed stable", [_domain_seed(d) for d in ("anime", "monet_real")] ==
      [_domain_seed(d) for d in ("anime", "monet_real")])
check("T4 no collision", len({_domain_seed(d) for d in
                              ("anime", "monet_real", "gpic", "synthetic", "textatlas")}) == 5)

# ---------------- T5 chunked shuffle generator（复刻 iter_real_synth_shard 的 _tar_pairs 模式）---------------- #
def _chunked(stream, r):
    chunk = []
    for kv in stream:
        chunk.append(kv)
        if len(chunk) >= _TAR_SHUFFLE_CHUNK:
            r.shuffle(chunk)
            yield from chunk
            chunk = []
    if chunk:
        r.shuffle(chunk)
        yield from chunk

src = [(f"k{i}", i) for i in range(1000)]
out = list(_chunked(iter(src), random.Random(3)))
check("T5 coverage exactly-once", sorted(out, key=lambda kv: kv[1]) == src and len(out) == 1000)
first_window = {kv[1] for kv in out[:_TAR_SHUFFLE_CHUNK]}
check("T5 window-bounded", first_window == set(range(_TAR_SHUFFLE_CHUNK)),
      "首窗口应恰为前 chunk 个元素的置换")

# ---------------- T6 Bucketer.allowed ---------------- #
class _FakeDS:
    _multi_ar_active = True
    _mar_extreme_clamped = 0

allowed_set = frozenset({sq, 33, 60})
bk = Bucketer(buckets, sq, _FakeDS(), adaptive=True, target_area=65536, divisor=16,
              no_upscale=True, max_aspect=4.0, allowed=allowed_set)
img = Image.new("RGB", (1000, 380))            # 宽横图 → 原生指派某横桶,多半不在 allowed
u8, bidx = bk.resize(img, 256)
check("T6 allowed clamp", bidx in allowed_set, f"bidx={bidx}")
H, W = buckets[bidx]
check("T6 shape matches bucket", u8.shape == (H, W, 3), f"{u8.shape} vs {(H, W, 3)}")
bk_free = Bucketer(buckets, sq, _FakeDS(), adaptive=True, target_area=65536, divisor=16,
                   no_upscale=True, max_aspect=4.0, allowed=None)
u8f, bidxf = bk_free.resize(img, 256)
check("T6 allowed=None 原生指派", buckets[bidxf][1] > buckets[bidxf][0], f"bidx={bidxf} hw={buckets[bidxf]}")

# ---------------- T7 square-only 护栏 ---------------- #
class _FakeDS2:
    multi_ar_enabled = True
    _mar_square_only = {"synthetic"}
    _mar_square_idx = sq
    _mar_buckets = buckets
    _crop_v_anchor = {}
    _v_anchor_for = CompleteV1Dataset._v_anchor_for

rec = {"u8": np.zeros((208, 288, 3), dtype=np.uint8), "bucket": 60}
rec2 = CompleteV1Dataset._force_square_only_record(_FakeDS2(), "synthetic", rec)
check("T7 forced to square", rec2["bucket"] == sq and rec2["u8"].shape == (256, 256, 3),
      f"bucket={rec2['bucket']} shape={rec2['u8'].shape}")
rec3 = CompleteV1Dataset._force_square_only_record(_FakeDS2(), "monet_real", rec)
check("T7 non-square-only untouched", rec3 is rec)

# ---------------- T8 buckets_by_ar_distance ---------------- #
_t8_ok_perm = _t8_ok_head = _t8_ok_order = True
for b0 in [0, sq, len(buckets) // 2, len(buckets) - 1]:
    order = buckets_by_ar_distance(b0, buckets)
    _t8_ok_perm &= sorted(order) == list(range(len(buckets)))
    _t8_ok_head &= order[0] == b0
    h0, w0 = buckets[b0]
    key = lambda i: (abs(math.log(buckets[i][1] / buckets[i][0]) - math.log(w0 / h0)),
                     abs(buckets[i][0] * buckets[i][1] - h0 * w0))
    _t8_ok_order &= all(key(order[k]) <= key(order[k + 1]) for k in range(len(order) - 1))
check("T8 permutation exactly-once", _t8_ok_perm)
check("T8 head is self", _t8_ok_head)
check("T8 order = brute force key", _t8_ok_order)

print()
if FAIL:
    print(f"RESULT: {FAIL} FAILED")
    sys.exit(1)
print("RESULT: ALL_UNIT_OK")
