from __future__ import annotations

import argparse
import json
from collections import Counter

from omegaconf import OmegaConf

from training.dataline.complete_dataset import (
    CompleteV1Dataset,
    iter_anime_shard,
    iter_real_synth_shard,
    iter_textatlas_shard,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage256_redesign_naifu_arb_v1.yaml")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--per-shard", type=int, default=64)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    kwargs = OmegaConf.to_container(cfg.dataset, resolve=True)
    kwargs.update({
        "rank": 0,
        "world_size": 1,
        "max_shards": 4,
        "max_rows_per_shard": args.per_shard,
        "anime_nrows": 4000,
        "buffer_size": 512,
        "queue_size": 128,
    })
    ds = CompleteV1Dataset(**kwargs)
    ds._multi_ar_active = True
    d = args.domain
    if d not in ds.domains:
        raise SystemExit(f"domain {d!r} not active; active={ds.domains}")

    counts = Counter()
    seen = 0
    force_square = d in ds._mar_square_only
    for si, shard in enumerate(ds.shards.get(d, [])):
        if seen >= args.max_samples:
            break
        v_anchor = ds._v_anchor_for(d)
        print(f"[bucketdist] domain={d} shard_index={si} seen={seen}", flush=True)
        if d == "anime":
            iterator = iter_anime_shard(
                shard, ds.cfg, ds.stage, ds.image_size, ds.anime_cache_dir,
                max_rows=args.per_shard, rng=None, nrows=ds.anime_nrows, skip_image=False,
                fetch_workers=1, bucketer=ds._mar_bucketer, v_anchor=v_anchor)
        elif d == "textatlas":
            iterator = iter_textatlas_shard(
                shard, ds.cfg, ds.stage, ds.image_size, ds.cfg.textatlas_source_root,
                local_root=ds.textatlas_local_root, max_rows=args.per_shard,
                skip_image=False, rng=None, bucketer=ds._mar_bucketer, v_anchor=v_anchor,
                force_square=force_square, square_idx=ds._mar_square_idx)
        else:
            iterator = iter_real_synth_shard(
                d, shard, ds.cfg, ds.stats, ds.stage, ds.image_size,
                max_rows=args.per_shard, skip_image=False, fetch_workers=1, rng=None,
                bucketer=ds._mar_bucketer, v_anchor=v_anchor,
                force_square=force_square, square_idx=ds._mar_square_idx)
        for rec in iterator:
            counts[int(rec["bucket"])] += 1
            seen += 1
            if seen >= args.max_samples:
                break

    total = sum(counts.values())
    row = [0.0] * len(ds._mar_buckets)
    if total:
        for b, c in counts.items():
            row[b] = c / total
    out = {
        "status": "PASS" if total else "EMPTY",
        "domain": d,
        "sampled": total,
        "bucket_count": len(ds._mar_buckets),
        "square_index": ds._mar_square_idx,
        "square_bucket": ds._mar_buckets[ds._mar_square_idx],
        "extreme_clamped": int(ds._mar_extreme_clamped),
        "native_bucket_dist": [round(x, 8) for x in row],
        "nonzero_buckets": [
            {"idx": i, "hw": ds._mar_buckets[i], "prob": round(row[i], 8)}
            for i, p in enumerate(row) if p > 0
        ],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
