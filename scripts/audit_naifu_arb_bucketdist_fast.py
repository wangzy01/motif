#!/usr/bin/env python3
"""bucketdist 快速审计（免取图版,2026-07-09）。

旧版 audit_naifu_arb_bucketdist_domain.py 走真实取图+decode,默认只测 256 样本/域、
且 rng=None 只取每 shard 头部行 → yaml native_bucket_dist 带 1/256 量化噪声 + 头部偏置,
稀有噪声桶被钉进调度表（bug hunt MAJOR 项）。

桶指派只依赖 (w,h)（assign_adaptive_bucket）,manifest 自带宽高 → 本脚本直接用 manifest
宽高 + 与 iter_* 完全一致的 elig/base_w/has_nl 过滤,复用生产 assign 函数,数万行/域秒级。
square_only 域(synthetic/textatlas)在 config 里是 one-hot(square),无需实测。

输出 JSON 与旧版兼容（native_bucket_dist/nonzero_buckets/...),另存 --out 文件。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

import numpy as np
from omegaconf import OmegaConf

from training.dataline.complete_dataset import CompleteV1Dataset
from training.dataline.manifest_index import load_anime_shard, load_real_synth_shard
from training.dataline.multi_ar import assign_adaptive_bucket, assign_bucket


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage256_redesign_naifu_arb_v1.yaml")
    ap.add_argument("--domain", required=True, help="anime | monet_real | gpic")
    ap.add_argument("--max-samples", type=int, default=50000)
    ap.add_argument("--max-shards", type=int, default=16, help="随机抽这么多 manifest shard")
    ap.add_argument("--anime-nrows", type=int, default=50000, help="anime 每 shard 随机 row-group 读行数")
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径（另打印 stdout）")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    kwargs = OmegaConf.to_container(cfg.dataset, resolve=True)
    kwargs.update({"rank": 0, "world_size": 1})
    ds = CompleteV1Dataset(**kwargs)
    d = args.domain
    if d not in ds.domains:
        raise SystemExit(f"domain {d!r} not active; active={ds.domains}")
    if d in ds._mar_square_only:
        raise SystemExit(f"domain {d!r} 是 square_only,config 已 one-hot(square),无需实测")

    rng = random.Random(args.seed)
    buckets = ds._mar_buckets
    adaptive = ds._mar_bucket_mode == "adaptive"
    counts: Counter = Counter()
    seen = 0
    clamped = 0
    shards = list(ds.shards.get(d, []))
    rng.shuffle(shards)
    elig_key = f"elig_{ds.image_size}"

    for shard in shards[: args.max_shards]:
        if seen >= args.max_samples:
            break
        if d == "anime":
            arr = load_anime_shard(shard, ds.cfg, with_caption=True,
                                   nrows=args.anime_nrows, rng=rng)
            elig = arr[elig_key]
            # 与 iter_anime_shard 一致：requires_nl(stage) 时丢无-NL 行（NL 可得性可能与宽高相关,须复刻）
            if ds.cfg.anime_caption.requires_nl(ds.stage):
                has_nl = np.fromiter((bool(r) or bool(m) or bool(s) for r, m, s in
                                      zip(arr["refined_description"], arr["medium_description"],
                                          arr["short_description"])),
                                     dtype=bool, count=len(elig))
                elig = elig & has_nl
        else:
            arr = load_real_synth_shard(shard, d, ds.cfg, ds.stats)
            # 与 iter_real_synth_shard 一致：base_w<=0 行不入池
            elig = arr[elig_key] & (arr["base_w"] > 0.0)
        w = arr["width"]
        h = arr["height"]
        idxs = np.where(elig & (w > 0) & (h > 0))[0]
        rng.shuffle(idxs)
        for i in idxs:
            wi, hi = int(w[i]), int(h[i])
            if adaptive:
                aspect = wi / max(hi, 1)
                if aspect > ds._mar_adaptive_max_aspect or aspect < 1.0 / ds._mar_adaptive_max_aspect:
                    clamped += 1
                b = assign_adaptive_bucket(wi, hi, buckets, ds._mar_target_area,
                                           ds._mar_divisor, ds._mar_adaptive_no_upscale,
                                           ds._mar_adaptive_max_aspect)
            else:
                b = assign_bucket(wi, hi, buckets)
            counts[int(b)] += 1
            seen += 1
            if seen >= args.max_samples:
                break
        print(f"[bucketdist-fast] domain={d} shard={shard} cum_seen={seen}", flush=True)

    total = sum(counts.values())
    row = [0.0] * len(buckets)
    if total:
        for b, c in counts.items():
            row[b] = c / total
    out = {
        "status": "PASS" if total else "EMPTY",
        "domain": d,
        "sampled": total,
        "bucket_count": len(buckets),
        "square_index": ds._mar_square_idx,
        "square_bucket": list(buckets[ds._mar_square_idx]),
        "extreme_clamped": clamped,
        "native_bucket_dist": [round(x, 8) for x in row],
        "nonzero_buckets": [
            {"idx": i, "hw": list(buckets[i]), "prob": round(row[i], 8), "n": counts.get(i, 0)}
            for i, p in enumerate(row) if p > 0
        ],
    }
    js = json.dumps(out, ensure_ascii=False, indent=2)
    print(js, flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(js + "\n")


if __name__ == "__main__":
    main()
