#!/usr/bin/env python3
"""EXP3 配套 — 对比两套 sampling config（arm A baseline vs arm B tag-heavy）在 anime@256 的
caption 特性：真实 Qwen token 长度分布、截断率、tag-vs-NL 占比。纯 CPU、不取图、不改核心代码。

复用 dataset 同款 caption 路径（load_anime_shard + build_anime_caption），只是换 config。
回答 exp3 的"截断率"问题 + 验证 arm B 确实更偏 tag。

用法：
  HF_HUB_OFFLINE=1 venv-gpu/bin/python experiments/exp_scripts/trunc_rate_probe.py \
      --config-a <baseline.json> --config-b <tagheavy.json> --stage 256 --n 4000
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from training.dataline.caption_build import build_anime_caption
from training.dataline.config import load_sampling_config
from training.dataline.manifest_index import list_shards, load_anime_shard

BASE = "/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json"
TPL = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
       "texture, quantity, text, spatial relationships of the objects and "
       "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
CAP_KEYS = ["refined_description", "medium_description", "short_description", "rating_bucket",
            "year", "tags_general", "tags_artist", "tags_character", "tags_copyright", "tags_meta",
            "style", "image_type", "camera_and_lens_effects", "foreground", "background"]
_SENT = re.compile(r"[.!?]")


def is_tag_like(cap: str) -> bool:
    """启发式：无句末标点 且 (含下划线 或 逗号密度高) → tag-style。"""
    if not cap:
        return False
    if _SENT.search(cap):
        return False
    return ("_" in cap) or (cap.count(",") >= 3) or (len(cap.split()) >= 8 and "." not in cap)


def run_config(cfg_path, arr, freq, stage, idx, tok, budget, maxlen):
    cfg = load_sampling_config(cfg_path)
    rng = random.Random(42)
    toks, empty, tag_like, hard_trunc, over_budget = [], 0, 0, 0, 0
    for i in idx:
        f = {k: arr[k][i] for k in CAP_KEYS if k in arr}
        cap = build_anime_caption(f, stage, cfg.anime_caption, rng, freq)
        if not cap.strip():
            empty += 1
            continue
        # pre-truncation 真实 token（caption 本体）
        ncap = len(tok(cap, add_special_tokens=False)["input_ids"])
        toks.append(ncap)
        if ncap > budget:
            over_budget += 1          # 超 caption 预算（会被 truncate_to_token_budget 截）
        if len(tok(TPL.format(cap))["input_ids"]) > maxlen:
            hard_trunc += 1           # 含模板超 max_length（无 tokenizer 截断时模型端硬截）
        if is_tag_like(cap):
            tag_like += 1
    ne = len(toks)
    pct = lambda x: round(100 * x / max(ne, 1), 1)
    return {
        "config": os.path.basename(cfg_path),
        "route_weights_256": cfg.anime_caption.stage_route_weights_has_nl.get(stage),
        "n_nonempty": ne, "empty": empty,
        "cap_tok_mean": round(float(np.mean(toks)), 1) if toks else 0,
        "cap_tok_median": int(np.median(toks)) if toks else 0,
        "cap_tok_p95": int(np.percentile(toks, 95)) if toks else 0,
        "cap_tok_max": int(np.max(toks)) if toks else 0,
        "over_budget_pct": pct(over_budget), "over_budget_n": over_budget,
        "hard_trunc_pct_no_tok": pct(hard_trunc),
        "tag_like_pct": pct(tag_like),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-a", default=BASE)
    ap.add_argument("--config-b", required=True)
    ap.add_argument("--stage", default="256")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--nrows", type=int, default=20000)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/fsx/models/Qwen3-VL-2B-Instruct")
    cfg0 = load_sampling_config(args.config_a)
    maxlen = cfg0.anime_caption.model_text_max_length
    tpl_overhead = len(tok(TPL.format(""))["input_ids"])
    budget = maxlen - tpl_overhead
    print(f"模板开销={tpl_overhead}, max_length={maxlen}, caption 预算={budget} tok\n", flush=True)

    shard = list_shards(cfg0)["anime"][0]
    arr = load_anime_shard(shard, cfg0, with_caption=True, nrows=args.nrows, rng=random.Random(0))
    n = len(arr["priority"])
    fc = Counter()
    for tg in arr["tags_general"]:
        if tg:
            fc.update(tg)
    freq = dict(fc)
    idx = list(range(n))
    random.Random(7).shuffle(idx)
    idx = idx[:args.n]

    ra = run_config(args.config_a, arr, freq, args.stage, idx, tok, budget, maxlen)
    rb = run_config(args.config_b, arr, freq, args.stage, idx, tok, budget, maxlen)

    print(f"=== EXP3 caption 对比 @ stage {args.stage} (n={len(idx)}) ===")
    print(f"{'metric':28} {'A (baseline)':>16} {'B (tag-heavy)':>16}")
    keys = [("route_weights_256", "256 route weights"), ("tag_like_pct", "tag-like %"),
            ("cap_tok_mean", "caption tok 均值"), ("cap_tok_median", "caption tok 中位"),
            ("cap_tok_p95", "caption tok p95"), ("cap_tok_max", "caption tok 最大"),
            ("over_budget_pct", f"超预算({budget}tok) %"), ("hard_trunc_pct_no_tok", "无tok时硬截 %"),
            ("empty", "空串数"), ("n_nonempty", "非空数")]
    for k, label in keys:
        print(f"{label:28} {str(ra[k]):>16} {str(rb[k]):>16}")
    import json
    print(f"\nTRUNC_PROBE_JSON {json.dumps({'A': ra, 'B': rb}, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
