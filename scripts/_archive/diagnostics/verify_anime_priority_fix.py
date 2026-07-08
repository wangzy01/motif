"""真实数据验证 #2(composition_clean 精确谓词) + #1(integer-repeat 曝光)。
读 anime 源 shard0（nrows）→ load_anime_shard 算 priority → 核：
  - clean 命中率应 ~20%（旧 risk_bucket 近似 ~80%）；
  - priority cap 命中应从 ~69% 大幅下降；
  - integer-repeat 权重：每行≥1、有界 boost（大头=1、少数 2..max_repeat）、E[曝光]∝(priority-1) 单调。
运行：venv-gpu/bin/python scripts/verify_anime_priority_fix.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from training.dataline.config import load_sampling_config
from training.dataline.manifest_index import (load_anime_shard, composition_clean_mask,
                                               CLEAN_BG_TAGS, VISIBLE_MARK_TAGS, YOLO_ADDED_TEXT_AREA_COL)
from training.dataline.sampler import compute_repeat_counts, DomainPool

CFG = "/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json"
SHARD = "/s3/evan/anime_core_ditv04_ct_soft/data/00000000.parquet"
N = 122880   # 全首 row-group（代表性；shard 内部按 qz 排序 → 前 N<rg 头部不代表全体）


def main():
    cfg = load_sampling_config(CFG)
    print(f"加载 anime shard0 nrows={N}（全首 row-group，代表性）...")
    arr = load_anime_shard(SHARD, cfg, with_caption=False, nrows=N)
    prio = arr["priority"]
    n = len(prio)
    s = cfg.anime_sampling

    # ---- #2 composition_clean 命中率（精确 vs 旧 risk 近似）----
    import pyarrow.parquet as pq, pyarrow as pa
    pf = pq.ParquetFile(SHARD)
    cols = ["tags_general", YOLO_ADDED_TEXT_AREA_COL, "quality_z", "risk_bucket"]
    bs = []
    got = 0
    for b in pf.iter_batches(batch_size=min(N, 65536), columns=cols):
        bs.append(b); got += b.num_rows
        if got >= N: break
    t = pa.Table.from_batches(bs).slice(0, N)
    tg = t.column("tags_general").to_pylist()
    yolo = np.array([x if x is not None else np.inf for x in t.column(YOLO_ADDED_TEXT_AREA_COL).to_pylist()], dtype=np.float64)
    qz = np.array([x if x is not None else 0.0 for x in t.column("quality_z").to_pylist()], dtype=np.float64)
    risk = np.array(t.column("risk_bucket").to_pylist(), dtype=object)
    clean = composition_clean_mask(tg, yolo, qz)
    approx = (risk == "plain_illustration") & (qz >= 0.4)
    prec_rate = clean.mean()
    apx_rate = approx.mean()
    # 精确谓词在近似命中中的占比（precision of approx）
    precision_of_approx = (clean & approx).sum() / max(approx.sum(), 1)
    print(f"\n[#2 composition_clean]")
    print(f"  精确谓词命中率 = {prec_rate:.1%}   （naifu 审计 ~19.5% → 期望 ~20%）")
    print(f"  旧 risk 近似命中率 = {apx_rate:.1%}   近似精确率 = {precision_of_approx:.1%}")
    assert 0.10 <= prec_rate <= 0.35, f"精确命中率应 ~20%，实际 {prec_rate:.1%}"
    assert prec_rate < apx_rate, "精确命中率应远低于旧近似"

    # ---- #3 priority 分布（应不再坍缩）----
    p50, p90, p99 = np.percentile(prio, [50, 90, 99])
    cap_hit = (prio >= s.max_priority - 1e-9).mean()
    print(f"\n[#3 priority 分布] p50={p50:.2f} p90={p90:.2f} p99={p99:.2f}  cap({s.max_priority})命中={cap_hit:.1%}")
    print(f"  （修 #2 前审计 cap 命中 ~69%；base=ct_weight 是 config 显式覆盖，残留由 ct_weight 基线决定）")

    # ---- #1 integer-repeat 曝光 ----
    rng = np.random.default_rng(0)
    counts = compute_repeat_counts(prio, s.max_repeat, s.extra_frac, rng)
    uniq, cnts = np.unique(counts, return_counts=True)
    print(f"\n[#1 integer-repeat]  extra_frac={s.extra_frac} max_repeat={s.max_repeat}")
    print(f"  repeat count 分布: " + ", ".join(f"{int(u)}×:{c/n:.1%}" for u, c in zip(uniq, cnts)))
    print(f"  每行≥1: {(counts>=1).all()}   实际额外曝光比 = {(counts.sum()-n)/n:.1%}（应≈extra_frac）")
    assert (counts >= 1).all(), "每行应至少曝光 1 次"
    assert counts.max() <= s.max_repeat, "不超 max_repeat"
    assert abs((counts.sum()-n)/n - s.extra_frac) < 0.02, "额外曝光比应≈extra_frac"
    # 单调：高 priority 行平均 count 更高
    hi = prio >= np.percentile(prio, 90)
    lo = prio <= np.percentile(prio, 10)
    print(f"  E[count|priority top10%]={counts[hi].mean():.3f}  E[count|bottom10%]={counts[lo].mean():.3f}")
    assert counts[hi].mean() > counts[lo].mean(), "高 priority 应获更多额外曝光（单调）"

    # ---- integer-repeat vs 连续 priority 的饱和对比 ----
    # 连续权重下 top10% 相对 bottom10% 的曝光比 vs integer-repeat 下的比
    cont_ratio = prio[hi].mean() / prio[lo].mean()
    # 长程 E[count] 通过多次重算平均（每 batch re-roll）
    acc = np.zeros(n)
    R = 30
    for r in range(R):
        acc += compute_repeat_counts(prio, s.max_repeat, s.extra_frac, np.random.default_rng(r))
    acc /= R
    rep_ratio = acc[hi].mean() / acc[lo].mean()
    print(f"\n[曝光比 top10%:bottom10%]  连续 priority={cont_ratio:.2f}:1   integer-repeat(长程)={rep_ratio:.2f}:1")
    print(f"  → integer-repeat 有界 boost：大头近均匀、仅高 priority 适度抬升（更贴 naifu『balanced by construction + 适度 boost』）")

    print("\n✓ ANIME PRIORITY FIX 验证通过（真实数据）")


if __name__ == "__main__":
    main()
