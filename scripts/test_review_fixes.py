"""验证 review 修复（CPU）：
  F-A1  freq dropout 分母改 median(ref)：中位频率 tag ~base_p、高频顶 max_p、长尾压 min_p，
        且整体 freq-weighted 有效 dropout 远高于旧 maxf 版（不再名存实亡）。
  KeyError guard  sample_one 在仅 monet/gpic 且全 served_mask==0 时走安全兜底，不抛 KeyError。
运行：venv-gpu/bin/python scripts/test_review_fixes.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np

from training.dataline.caption_build import _freq_dropout
from training.dataline.sampler import DomainPool, CompleteSampler
from training.dataline.config import load_sampling_config

CFG_PATH = "/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json"


def _zipf_freq(n=300, seed=0):
    """造 Zipfian general tag 频率（真实 booru tag 频率近 Zipf）。"""
    rng = np.random.default_rng(seed)
    tags = [f"t{i}" for i in range(n)]
    # rank-1 frequency ~ C / rank
    freq = {t: int(max(1, round(20000 / (i + 1)))) for i, t in enumerate(tags)}
    return tags, freq


def test_freq_dropout_median():
    tags, freq = _zipf_freq()
    p = {"base_p": 0.65, "alpha": 0.75, "min_p": 0.1, "max_p": 0.9,
         "min_keep": 6, "min_keep_ratio": 0.2}
    vals = list(freq.values())
    ref = max(float(np.median(vals)), 1.0)
    # 直接核 drop_p 公式（不走 rng）：中位/高频/长尾的 dp
    def dp(f):
        return min(max(p["base_p"] * (max(int(f), 1) / ref) ** p["alpha"], p["min_p"]), p["max_p"])
    med_tag_f = int(np.median(vals))
    hi_f = max(vals)
    lo_f = min(vals)
    rare_f = 1                                            # 极稀有 tag（freq=1 << median）
    dp_med, dp_hi, dp_lo, dp_rare = dp(med_tag_f), dp(hi_f), dp(lo_f), dp(rare_f)
    print(f"  ref(median freq)={ref:.1f}  dp(median)={dp_med:.3f}  dp(hi={hi_f})={dp_hi:.3f}  "
          f"dp(lo={lo_f})={dp_lo:.3f}  dp(rare=1)={dp_rare:.3f}")
    # 期望：中位 tag dp≈base_p（非贴 min_p）；高频顶 max_p；freq=1 极稀有压到 min_p 地板；单调
    assert 0.4 <= dp_med <= 0.85, f"中位 dp 应≈base_p，实际 {dp_med}"
    assert dp_hi >= 0.85, f"高频应顶 max_p，实际 {dp_hi}"
    assert abs(dp_rare - p["min_p"]) < 1e-9, f"freq=1 极稀有应压到 min_p，实际 {dp_rare}"
    assert dp_hi > dp_med > dp_lo >= dp_rare, "单调性：高频>中位>长尾≥极稀有"

    # 对比旧 maxf 分母：几乎全部钉 min_p（证明旧实现名存实亡）
    maxf = max(vals)
    def dp_old(f):
        return min(max(p["base_p"] * (max(int(f), 1) / maxf) ** p["alpha"], p["min_p"]), p["max_p"])
    old_at_minp = sum(1 for f in vals if dp_old(f) <= p["min_p"] + 1e-9) / len(vals)
    new_at_minp = sum(1 for f in vals if dp(f) <= p["min_p"] + 1e-9) / len(vals)
    print(f"  旧 maxf 版 dp==min_p 占比={old_at_minp:.1%}  新 median 版={new_at_minp:.1%}")
    assert old_at_minp > 0.85, "旧版应几乎全钉 min_p（复现 bug）"
    assert new_at_minp < old_at_minp, "新版钉 min_p 比例应显著下降"

    # 蒙特卡洛：实际 dropout 后保留率（新版应明显低于旧版的近乎全保留）
    rng = np.random.default_rng(1)
    def mc_keep_ratio(fn_dp_mode):
        kept = 0
        trials = 200
        for _ in range(trials):
            # 用 _freq_dropout（median）或手算旧版
            if fn_dp_mode == "new":
                k = _freq_dropout(list(tags), freq, p, rng)
            else:
                k = [t for t in tags if rng.random() >= dp_old(freq[t])]
                floor = max(p["min_keep"], int(len(tags) * p["min_keep_ratio"]))
                if len(k) < floor:
                    k = tags[:floor]
            kept += len(k)
        return kept / (trials * len(tags))
    new_keep = mc_keep_ratio("new")
    old_keep = mc_keep_ratio("old")
    print(f"  保留率：旧 maxf={old_keep:.1%}  新 median={new_keep:.1%}（新版应更低 = 更多高频被丢）")
    assert new_keep < old_keep, "新版总保留率应低于旧版（dropout 真正生效）"
    print("  ✓ freq dropout median 分母：单调、中位≈base_p、有效 dropout 恢复")


def test_sample_one_keyerror_guard():
    cfg = load_sampling_config(CFG_PATH)
    samp = CompleteSampler(cfg, "512", seed=0)
    # 构造仅 monet/gpic 且 served_mask 全 0 的退化池（无 anime/synthetic key）
    n = 5
    bad = {
        "uid": np.arange(n),
        "served_mask": np.zeros(n, dtype=np.int16),     # 不服务任何档
        "base_w": np.ones(n, dtype=np.float64),
    }
    pools = {"monet_real": DomainPool("monet_real", bad),
             "gpic": DomainPool("gpic", dict(bad))}
    # 多抽几次，确保不 KeyError，且返回合法 (domain, idx, band)
    for _ in range(50):
        d, idx, b = samp.sample_one(pools)
        assert d in ("monet_real", "gpic"), d
        assert 0 <= idx < n, idx
    print("  ✓ sample_one 退化池（仅 monet/gpic + served_mask 全 0）走安全兜底，无 KeyError")

    # 真正空池 → 明确异常（present 为空在 domain_mix_renorm 即断言失败；非静默 KeyError）
    empty = {"uid": np.array([]), "served_mask": np.array([], dtype=np.int16),
             "base_w": np.array([], dtype=np.float64)}
    pools2 = {"monet_real": DomainPool("monet_real", empty)}
    raised = None
    try:
        samp.sample_one(pools2)
    except Exception as e:                                # noqa: BLE001
        raised = e
    assert raised is not None and not isinstance(raised, KeyError), \
        f"空池应抛明确(非 KeyError)异常，实际 {type(raised).__name__}: {raised}"
    print(f"  ✓ 全空池 → 明确异常 {type(raised).__name__}（renorm 断言，不静默 KeyError）")


if __name__ == "__main__":
    print("[1] freq dropout median 分母"); test_freq_dropout_median()
    print("[2] sample_one KeyError guard"); test_sample_one_keyerror_guard()
    print("\nALL REVIEW-FIX TESTS PASSED")
