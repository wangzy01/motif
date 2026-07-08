"""构建 latent cache 的全局 uid->relpath 索引（训练读缓存用）。

读 stage={S}/index/*.parquet 的 (uid, path) 两列，合并写 stage={S}/uid2path.parquet。
训练侧 LatentCacheIndex 加载它建 uid->path 映射，按 uid 取回缓存 moments。

为什么要全局索引：anime 训练读源 parquet（cfg.anime_caption_source，17 shard），缓存按薄
manifest（cfg.manifest_anime，64 shard）切片写 part —— **shard 名不对齐**，无法按 shard 局部
glob，故需全局 uid->path。real/synth 虽对齐，统一走全局索引更简单。

写盘：safetensors/parquet 不能直接写 mountpoint-s3（缺随机写）→ 写 /local 再顺序 copy 到 /s3。
读 stage=*/ 下 parquet 用 read_parquet_file（绕开 pyarrow hive 分区推断，见 vae_cache 脚本）。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.vae_cache_flux2_moments import _copy, read_parquet_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out-root", default="/s3/evan/mot_latents/complete_v1_flux2_moments")
    ap.add_argument("--scratch", default="/local/evan/vae_cache_scratch")
    ap.add_argument("--out-name", default="uid2path.parquet")
    args = ap.parse_args()

    stage_dir = os.path.join(args.out_root, f"stage={args.stage}")
    index_dir = os.path.join(stage_dir, "index")
    parts = sorted(glob.glob(os.path.join(index_dir, "*.parquet")))
    assert parts, f"没找到 index part: {index_dir}"
    print(f"index parts: {len(parts)}", flush=True)

    t0 = time.perf_counter()
    tables = []
    rows = 0
    for i, p in enumerate(parts):
        t = read_parquet_file(p, columns=["uid", "path"])
        tables.append(t)
        rows += t.num_rows
        if (i + 1) % 200 == 0:
            print(f"  read {i+1}/{len(parts)} parts, {rows} rows, {time.perf_counter()-t0:.0f}s", flush=True)
    merged = pa.concat_tables(tables)
    print(f"merged rows: {merged.num_rows} ({time.perf_counter()-t0:.0f}s)", flush=True)

    # 统计：唯一 uid、域分布、distinct path
    uids = merged.column("uid").to_pylist()
    paths = merged.column("path")
    n_unique = len(set(uids))
    dom = Counter(u.split(":", 1)[0] for u in uids)
    n_paths = len(set(paths.to_pylist()))
    print(f"unique uids: {n_unique} (dup={merged.num_rows - n_unique})", flush=True)
    print(f"per-domain(uid prefix): {dict(dom)}", flush=True)
    print(f"distinct safetensors paths: {n_paths}", flush=True)

    os.makedirs(args.scratch, exist_ok=True)
    local_out = os.path.join(args.scratch, f"uid2path_{args.stage}.parquet")
    pq.write_table(merged, local_out, compression="zstd")
    dst = os.path.join(stage_dir, args.out_name)
    _copy(local_out, dst)
    sz = os.path.getsize(local_out)
    try:
        os.remove(local_out)
    except OSError:
        pass
    print(f"wrote {dst}  ({sz/(1<<20):.1f} MiB, {merged.num_rows} rows)  total {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
