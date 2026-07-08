"""核查外部审计 F3(caption 真实硬截断率) + F2b(integer-repeat 是否超 max_repeat) —— 用真 Qwen
tokenizer + 真 anime 数据，量化当前(已修)代码的实际行为。只读，不改训练代码。
运行：HF_HOME=/local/evan/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      venv-gpu/bin/python scripts/audit_probe_trunc_repeat.py
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from training.dataline.config import load_sampling_config
from training.dataline.manifest_index import load_anime_shard
from training.dataline.caption_build import build_anime_caption
from training.dataline.sampler import compute_repeat_counts

CFG = "/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json"
SHARD = "/s3/evan/anime_core_ditv04_ct_soft/data/00000000.parquet"
TPL = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
       "texture, quantity, text, spatial relationships of the objects and "
       "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
MAXLEN = 296


def main():
    cfg = load_sampling_config(CFG)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/fsx/models/Qwen3-VL-2B-Instruct")
    tpl_overhead = len(tok(TPL.format(""))["input_ids"])
    print(f"模板 token 开销 = {tpl_overhead}（caption 预算 = {MAXLEN}-{tpl_overhead} = {MAXLEN-tpl_overhead}）\n")

    # 用代表性 row-group（随机），跑 build_anime_caption 生成各 stage caption
    rng_load = random.Random(0)
    arr = load_anime_shard(SHARD, cfg, with_caption=True, nrows=20000, rng=rng_load)
    n = len(arr["priority"])
    cap_keys = ["refined_description", "medium_description", "short_description", "rating_bucket",
                "year", "tags_general", "tags_artist", "tags_character", "tags_copyright", "tags_meta"]
    freq = {}
    from collections import Counter
    fc = Counter()
    for tg in arr["tags_general"]:
        if tg: fc.update(tg)
    freq = dict(fc)

    print("=== F3：caption 真实硬截断率（模型端 max_length=296, truncation=True）===")
    for stage in ("256", "512", "1024"):
        rng = random.Random(42)
        n_samp = 3000
        hard = 0
        nonempty = 0
        over_tokens = []
        idx = list(range(n)); random.Random(7).shuffle(idx); idx = idx[:n_samp]
        for i in idx:
            f = {k: arr[k][i] for k in cap_keys}
            cap = build_anime_caption(f, stage, cfg.anime_caption, rng, freq)
            if not cap:
                continue
            nonempty += 1
            full = TPL.format(cap)
            ids = tok(full)["input_ids"]
            if len(ids) > MAXLEN:
                hard += 1
                over_tokens.append(len(ids))
        rate = hard / max(nonempty, 1)
        avg_over = (np.mean(over_tokens) if over_tokens else 0)
        print(f"  stage {stage}: 硬截断 {hard}/{nonempty} = {rate:.1%}  (超长样本 avg tok={avg_over:.0f})")

    print("\n=== F4：loader 自身 anime caption 空串率（应=has_nl·dropout）===")
    rng = random.Random(42)
    empt = 0; nl = 0
    for i in idx:
        f = {k: arr[k][i] for k in cap_keys}
        has_nl = bool(f.get("refined_description") or f.get("medium_description") or f.get("short_description"))
        nl += has_nl
        cap = build_anime_caption(f, "256", cfg.anime_caption, rng, freq)
        if not cap.strip():
            empt += 1
    print(f"  loader 空串 {empt}/{len(idx)} = {empt/len(idx):.1%}（has_nl 占 {nl/len(idx):.0%}，×dropout {cfg.anime_caption.dropout}）")
    print(f"  ⚠️ 模型 forward 再 uniform 10% dropout → anime NL 行实际≈{1-0.9*0.9:.0%}、tag-only 被模型 drop 10%（config 说永不）")

    print("\n=== F2b：integer-repeat 后，with-replacement 抽样的单行 draw 次数是否超 max_repeat ===")
    prio = arr["priority"][:256]                      # 模拟一个 buffer
    mr, ef = cfg.anime_sampling.max_repeat, cfg.anime_sampling.extra_frac
    counts = compute_repeat_counts(prio, mr, ef, np.random.default_rng(0))
    print(f"  repeat_counts（权重）max={counts.max()} ≤ max_repeat={mr} ✓（权重本身封顶）")
    # 但 sampler 是 with-replacement：模拟从该 buffer 抽 N 次，看单行命中次数
    cum = np.cumsum(counts.astype(float))
    rng2 = np.random.default_rng(1)
    draws = Counter()
    Ndraw = 256 * 4                                   # 约 buffer 在驻留期内被抽的次数量级
    for _ in range(Ndraw):
        r = rng2.random() * cum[-1]
        draws[int(np.searchsorted(cum, r))] += 1
    mx = max(draws.values())
    over = sum(1 for v in draws.values() if v > mr)
    print(f"  with-replacement 抽 {Ndraw} 次：单行最大命中={mx}（>{mr} 的行数={over}）")
    print(f"  → 权重封顶但 with-replacement 不硬封单行 draw（审计 F2b 实质）；E[曝光]∝repeat 已忠实、rolling buffer 软界")


if __name__ == "__main__":
    main()
