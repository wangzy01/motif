#!/usr/bin/env python
"""聚合 16-rank realized-mix 探针结果(读各 rank 日志里的 `[probe-json]` 行)。

输出：(A) 聚合真实 mix(集群平均，复刻生产争抢) vs 目标 ±Δ；(B) per-rank 离散度(min/max/std)；
     (C) 稳态 buffer 占比 per-rank；(D) 与 Phase-1 修复前审计基线对比(写死在下方)。
用法：python scripts/aggregate_probe_mix.py <results_dir 或 rank*.log 文件列表>
"""
import glob
import json
import os
import sys
import statistics as st

_DOMAINS = ["anime", "monet_real", "gpic", "synthetic", "textatlas"]
# 修复前基线(审计 §1.1，2026-06-25，16-rank 307200 样本)：聚合实测 + per-rank 范围
_BASELINE = {
    "agg":  {"anime": 0.329, "monet_real": 0.417, "gpic": 0.181, "synthetic": 0.073},
    "range": {"anime": (0.264, 0.425), "monet_real": (0.299, 0.581),
              "gpic": (0.069, 0.289), "synthetic": (0.050, 0.112)},
}


def load(paths):
    recs = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    k = line.find("[probe-json]")
                    if k >= 0:
                        recs.append(json.loads(line[k + len("[probe-json]"):].strip()))
        except OSError:
            continue
    # 去重(同 rank 多行取最后)
    by_rank = {}
    for r in recs:
        by_rank[r["rank"]] = r
    return [by_rank[k] for k in sorted(by_rank)]


def main():
    args = sys.argv[1:]
    if not args:
        print("用法：aggregate_probe_mix.py <results_dir | rank*.log ...>"); sys.exit(2)
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths += sorted(glob.glob(os.path.join(a, "rank*.log")))
        else:
            paths += glob.glob(a)
    recs = load(paths)
    if not recs:
        print(f"未找到任何 [probe-json] 行(检查 {paths[:3]}... 是否完成)"); sys.exit(1)

    target = recs[0]["target"]
    world = recs[0].get("world", len(recs))
    total = sum(sum(r["counts"].values()) for r in recs)
    agg = {d: sum(r["counts"].get(d, 0) for r in recs) for d in _DOMAINS}
    agg_total = sum(agg.values()) or 1

    print(f"===== 聚合 {len(recs)}/{world} rank, 总样本 {total} =====")
    print(f"{'域':<12}{'实测':>8}{'目标':>8}{'Δ':>8}{'(修前Δ)':>10}   per-rank[min–max] std   (修前范围)")
    for d in _DOMAINS:
        share = agg[d] / agg_total
        tgt = float(target.get(d, 0.0))
        per = [r["counts"].get(d, 0) / (sum(r["counts"].values()) or 1) for r in recs]
        lo, hi = min(per), max(per)
        sd = st.pstdev(per) if len(per) > 1 else 0.0
        within = "✓" if abs(share - tgt) <= 0.05 else "✗(>±5%)"
        if d in _BASELINE["agg"]:                          # 4 老域有"修前"基线对照
            base_d = (_BASELINE["agg"][d] - tgt)
            brange = _BASELINE["range"][d]
            print(f"{d:<12}{share:>8.1%}{tgt:>8.1%}{share-tgt:>+8.1%}{base_d:>+10.1%}   "
                  f"[{lo:.1%}–{hi:.1%}] σ{sd:.1%}   (修前[{brange[0]:.1%}–{brange[1]:.1%}]) {within}")
        else:                                              # textatlas 等新域无基线
            print(f"{d:<12}{share:>8.1%}{tgt:>8.1%}{share-tgt:>+8.1%}{'—':>10}   "
                  f"[{lo:.1%}–{hi:.1%}] σ{sd:.1%}   (无基线) {within}")

    print("\n----- per-rank 稳态 buffer 占比(低=该域仍供不上) -----")
    for d in _DOMAINS:
        bf = [r.get("buffer_fill", {}).get(d, 0.0) for r in recs]
        if bf:
            print(f"  {d:<12} min {min(bf):.2f}  median {st.median(bf):.2f}  max {max(bf):.2f}")
    print("\n----- anime 分到的 shard 数(应每 rank=全量，证 force_row_level 行级) -----")
    asc = [r.get("shard_counts", {}).get("anime", 0) for r in recs]
    print(f"  anime shard/rank: min {min(asc)} max {max(asc)}  (修前文件级=1~2/rank → 现应全量一致)")
    gsc = [r.get("shard_counts", {}).get("gpic", 0) for r in recs]
    print(f"  gpic  shard/rank: min {min(gsc)} max {max(gsc)}  (文件级，world={world} 应≈{64//max(world,1)})")

    # 判据
    agg_ok = all(abs(agg[d] / agg_total - float(target.get(d, 0.0))) <= 0.05 for d in _DOMAINS)
    spreads = {d: (max(r["counts"].get(d, 0)/(sum(r["counts"].values()) or 1) for r in recs)
                   - min(r["counts"].get(d, 0)/(sum(r["counts"].values()) or 1) for r in recs))
               for d in _DOMAINS}
    print(f"\n判据：聚合 mix 全域在 ±5% 内 = {'✓ PASS' if agg_ok else '✗ FAIL'}")
    print(f"      per-rank 极差(max-min): " + "  ".join(f"{d}={spreads[d]:.1%}" for d in _DOMAINS)
          + "  (越小越好；修前 gpic 22pp/monet 28pp)")


if __name__ == "__main__":
    main()
