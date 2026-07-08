#!/usr/bin/env python3
"""把本【节点】训练要读的 TextAtlas 源图 parquet 预存到 /local NVMe（绕开 S3 挂载带宽争抢）。

为什么在训练 job 内 per-node 跑（而非单独 job）：JCC 动态调度——单独预存 job 与训练 job 可能落在
不同物理节点，而 /local 是节点本地盘。故须在【训练 job 启动时、train.py 之前】每节点各自预存其
8 个 rank 要读的 shard，保证同节点命中。读侧（iter_textatlas_shard）local 优先、缺失回退 S3，故
本步是纯优化、失败/部分完成也不影响正确性。

shard 归属与读侧【完全一致】：import list_shards + shard_subset_for_rank（同 force_row_level=False，
textatlas 595 shard≫world 走文件级 idx%world==rank）。幂等：已存在且非空则跳过（同节点重复 job 秒级）。

用法（训练 job 内，每节点跑一次；--config = sampling JSON，与 stage yaml 的 dataset.sampling_config 同）:
  PY scripts/prestage_textatlas.py \
     --config /fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1_textatlas.json \
     --world $((NNODES*8)) --node-rank $NODE_RANK --local-root /local/evan/textatlas5m [--procs 16]
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataline.config import load_sampling_config
from training.dataline.manifest_index import (list_shards, shard_subset_for_rank,
                                              textatlas_subset_shard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--world", type=int, required=True, help="NNODES*8（与训练一致）")
    ap.add_argument("--node-rank", type=int, required=True)
    ap.add_argument("--local-root", default="/local/evan/textatlas5m")
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()

    cfg = load_sampling_config(args.config)
    if not cfg.manifest_textatlas:
        print("[prestage] cfg 无 textatlas → 跳过", flush=True)
        return
    sidecars = list_shards(cfg)["textatlas"]
    src_root = cfg.textatlas_source_root

    # 本节点 8 个 rank 要读 shard 的并集（与读侧 shard_subset_for_rank 同口径）
    wanted = set()
    for L in range(8):
        gr = args.node_rank * 8 + L
        for sc in shard_subset_for_rank(sidecars, gr, args.world, force_row_level=False):
            wanted.add(sc)
    tasks = []
    for sc in sorted(wanted):
        subset, shardname = textatlas_subset_shard(sc)
        tasks.append((os.path.join(src_root, subset, shardname),
                      os.path.join(args.local_root, subset, shardname)))
    print(f"[prestage] node={args.node_rank} world={args.world}: {len(tasks)} shards → {args.local_root}",
          flush=True)

    def copy_one(t):
        srcp, dstp = t
        try:
            if os.path.exists(dstp) and os.path.getsize(dstp) > 0:
                return ("skip", 0)
            if not os.path.exists(srcp):
                return ("missing-src", 0)
            os.makedirs(os.path.dirname(dstp), exist_ok=True)
            tmp = f"{dstp}.tmp.{os.getpid()}"
            shutil.copyfile(srcp, tmp)          # mountpoint-s3 → local NVMe
            sz = os.path.getsize(tmp)
            os.replace(tmp, dstp)               # 原子
            return ("copied", sz)
        except Exception as e:                  # noqa: BLE001 单 shard 失败不致命（读侧回退 S3）
            print(f"[prestage][warn] {srcp}: {type(e).__name__} {e}", flush=True)
            return ("error", 0)

    import concurrent.futures as cf
    t0 = time.perf_counter()
    n_copied = n_skip = n_err = n_miss = 0
    bytes_copied = 0
    with cf.ThreadPoolExecutor(max_workers=args.procs) as ex:
        for i, (status, sz) in enumerate(ex.map(copy_one, tasks)):
            if status == "copied":
                n_copied += 1; bytes_copied += sz
            elif status == "skip":
                n_skip += 1
            elif status == "missing-src":
                n_miss += 1
            else:
                n_err += 1
            if (i + 1) % 50 == 0:
                print(f"[prestage] {i+1}/{len(tasks)} (copied={n_copied} skip={n_skip} "
                      f"miss={n_miss} err={n_err}) {bytes_copied/1e9:.1f}GB {time.perf_counter()-t0:.0f}s",
                      flush=True)
    print(f"[prestage] DONE node={args.node_rank}: copied={n_copied} skip={n_skip} miss={n_miss} "
          f"err={n_err} {bytes_copied/1e9:.1f}GB in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
