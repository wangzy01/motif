#!/usr/bin/env python3
"""Benchmark complete_v1 dataloader throughput.

Run under plain python or torchrun. The dataset reads RANK/WORLD_SIZE from the
environment even when torch.distributed is not initialized, so this script can
simulate per-rank loader pressure without touching GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
from omegaconf import OmegaConf

from training.dataline.complete_dataset import CompleteV1Dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--stats-cache", default=None)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-rows-per-shard", type=int, default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    ds_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    if args.stage is not None:
        ds_cfg["stage"] = str(args.stage)
    if args.image_size is not None:
        ds_cfg["image_size"] = int(args.image_size)
    if args.stats_cache is not None:
        ds_cfg["stats_cache"] = args.stats_cache
    if args.max_shards is not None:
        ds_cfg["max_shards"] = args.max_shards
    if args.max_rows_per_shard is not None:
        ds_cfg["max_rows_per_shard"] = args.max_rows_per_shard

    batch_size = int(args.batch_size or cfg.trainer.batch_size)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    t0 = time.perf_counter()
    ds = CompleteV1Dataset(**ds_cfg)
    init_s = time.perf_counter() - t0
    loader = ds.loader(batch_size)

    times = []
    batch_summaries = []
    total = args.warmup + args.batches
    for i in range(total):
        b0 = time.perf_counter()
        batch = next(loader)
        dt = time.perf_counter() - b0
        if i >= args.warmup:
            times.append(dt)
        if i < 3 or i == total - 1:
            pixels = batch["pixels"]
            batch_summaries.append(
                {
                    "iter": i,
                    "sec": round(dt, 4),
                    "pixels_shape": list(pixels.shape),
                    "empty_prompts": int(sum(1 for p in batch["prompts"] if not p)),
                    "domain_counter": dict(ds.domain_counter),
                }
            )
    n_img = batch_size * max(len(times), 1)
    out = {
        "rank": rank,
        "local_rank": local_rank,
        "world": world,
        "config": args.config,
        "stage": str(ds_cfg["stage"]),
        "image_size": int(ds_cfg["image_size"]),
        "batch_size_per_rank": batch_size,
        "init_s": round(init_s, 3),
        "warmup": args.warmup,
        "measured_batches": len(times),
        "first_measured_s": round(times[0], 4) if times else None,
        "mean_s": round(statistics.mean(times), 4) if times else None,
        "median_s": round(statistics.median(times), 4) if times else None,
        "p95_s": round(sorted(times)[max(0, int(len(times) * 0.95) - 1)], 4) if times else None,
        "img_per_s_rank": round(n_img / sum(times), 3) if times and sum(times) > 0 else None,
        "domain_counter": dict(ds.domain_counter),
        "samples": batch_summaries,
    }
    print("DATA_BENCH_JSON " + json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
