"""验证 #5/#6 多卡全局 stats 修复（CPU，造临时 parquet）：
  #5  全池统计：stats 必须 over 全量 monet 列表（不被 rank 切片）→ 各 rank 一致；
      复现旧 bug：over 子集会得到不同 source_counts/cluster_sizes。
  #6  原子写：temp+os.replace，写后无 .tmp 残留、缓存可读、内容正确；compute=False 无缓存即报错。
运行：venv-gpu/bin/python scripts/test_global_stats_fix.py
"""
import sys, os, glob, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from training.dataline.manifest_index import compute_real_global_stats

TMP = tempfile.mkdtemp(prefix="stats_test_", dir="/local/evan" if os.path.isdir("/local/evan") else None)


def _make_shard(path, sub_sources, cluster_ids):
    t = pa.table({"sub_source": pa.array(sub_sources), "cluster_id": pa.array(cluster_ids, type=pa.int64())})
    pq.write_table(t, path)


def main():
    # 4 个 shard，故意让源/簇分布跨 shard 不同（这样子集统计 != 全池统计）
    shards = []
    specs = [
        (["A", "A", "B"], [1, 1, 2]),
        (["B", "C", "C"], [2, 3, 3]),
        (["A", "C"],      [1, 3]),
        (["D", "D", "D"], [4, 4, 4]),
    ]
    for i, (ss, ci) in enumerate(specs):
        p = os.path.join(TMP, f"shard{i}.parquet")
        _make_shard(p, ss, ci)
        shards.append(p)

    # ---- #5 全池 vs 子集 ----
    full = compute_real_global_stats(shards, cache_path=None)
    # world=2 rank0 旧 bug：只 over shard[0::2] = shard0,shard2
    subset_r0 = compute_real_global_stats(shards[0::2], cache_path=None)
    subset_r1 = compute_real_global_stats(shards[1::2], cache_path=None)
    print("[#5 全池 vs rank 子集]")
    print(f"  全池 source_counts = {full['source_counts']}")
    print(f"  rank0 子集 = {subset_r0['source_counts']}  rank1 子集 = {subset_r1['source_counts']}")
    assert full["source_counts"] == {"A": 3, "B": 2, "C": 3, "D": 3}, full["source_counts"]
    assert full["cluster_sizes"] == {1: 3, 2: 2, 3: 3, 4: 3}, full["cluster_sizes"]
    assert subset_r0["source_counts"] != subset_r1["source_counts"], "子集应不一致（复现旧 bug 的根因）"
    # 关键：两 rank 若都用【全池】列表 → 完全一致
    full_r0 = compute_real_global_stats(shards, cache_path=None)
    full_r1 = compute_real_global_stats(shards, cache_path=None)
    assert full_r0 == full_r1, "全池列表下各 rank 必须一致"
    print("  ✓ 全池列表 → 各 rank base_w 一致；子集列表会不一致（修复 = 传全池）")

    # ---- #6 原子写 ----
    cache = os.path.join(TMP, "stats_cache.json")
    s = compute_real_global_stats(shards, cache_path=cache, compute=True)
    assert os.path.exists(cache), "缓存应已写"
    assert not glob.glob(cache + ".tmp.*"), "不应残留 .tmp 临时文件"
    reread = compute_real_global_stats(shards, cache_path=cache)  # 走读缓存分支
    assert reread["source_counts"] == s["source_counts"] == full["source_counts"]
    assert reread["cluster_sizes"] == s["cluster_sizes"] == full["cluster_sizes"]
    print("\n[#6 原子写]")
    print(f"  ✓ 缓存写入 {os.path.basename(cache)}、无 .tmp 残留、读回内容一致（int key 还原）")

    # ---- compute=False 语义（非 rank0 协议）----
    cache2 = os.path.join(TMP, "nonexist.json")
    try:
        compute_real_global_stats(shards, cache_path=cache2, compute=False)
        raise AssertionError("无缓存 + compute=False 应抛 FileNotFoundError")
    except FileNotFoundError:
        pass
    # 缓存存在时 compute=False 正常读
    r = compute_real_global_stats(shards, cache_path=cache, compute=False)
    assert r["source_counts"] == full["source_counts"]
    print("  ✓ compute=False：无缓存→FileNotFoundError；有缓存→正常读（非 rank0 barrier 后路径）")

    print("\nALL GLOBAL-STATS FIX TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    finally:
        import shutil; shutil.rmtree(TMP, ignore_errors=True)
