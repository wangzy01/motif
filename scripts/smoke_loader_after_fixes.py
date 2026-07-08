"""修复后数据 loader 集成回归（CPU，真数据，不需模型/VAE）：
拉若干 batch，确认 {prompts, pixels} 契约、prompt 多为非空、4 域都能出现、integer-repeat/精确谓词
路径不报错。max_shards 小、buffer 小、anime_nrows 小以快收敛。
运行：venv-gpu/bin/python scripts/smoke_loader_after_fixes.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataline.complete_dataset import CompleteV1Dataset

CFG = "/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json"


def main():
    ds = CompleteV1Dataset(
        sampling_config=CFG, stage="256", image_size=256,
        seed=0, buffer_size=64, queue_size=32, min_fill=16,
        max_shards=1, anime_nrows=4000,
        anime_cache_dir="/local/evan/anime_img_cache",
        stats_cache="/local/evan/mot_stats/monet_global_stats_smoke.json",
    )
    print(f"dataset built; rank={ds.rank}/{ds.world_size}; stats sources={len(ds.stats['source_counts'])} clusters={len(ds.stats['cluster_sizes'])}")
    bs = 16
    n_batches = 4
    it = iter(ds.loader(bs))
    t0 = time.time()
    empty_prompts = 0
    total_prompts = 0
    keep_total = 0
    for bi in range(n_batches):
        batch = next(it)
        prompts, pixels = batch["prompts"], batch["pixels"]
        assert len(prompts) == bs, (len(prompts), bs)
        assert tuple(pixels.shape) == (bs, 3, 256, 256), pixels.shape
        assert pixels.dtype.is_floating_point
        assert float(pixels.min()) >= -1.001 and float(pixels.max()) <= 1.001, (float(pixels.min()), float(pixels.max()))
        # F4：batch 须带 cfg_keep（per-sample，tag-only anime=True 永不 drop）
        assert "cfg_keep" in batch and len(batch["cfg_keep"]) == bs, "batch 缺 cfg_keep"
        assert all(isinstance(x, bool) for x in batch["cfg_keep"])
        keep_total += sum(batch["cfg_keep"])
        empty_prompts += sum(1 for p in prompts if not p.strip())
        total_prompts += len(prompts)
        print(f"  batch {bi}: pixels{tuple(pixels.shape)} range[{float(pixels.min()):.2f},{float(pixels.max()):.2f}] "
              f"prompt[0]={prompts[0][:70]!r}")
    dt = time.time() - t0
    print(f"\n域曝光累计: {dict(ds.domain_counter)}")
    print(f"cfg_keep=True（tag-only anime 豁免）= {keep_total}/{total_prompts}")
    print(f"loader 空 prompt 占比 = {empty_prompts}/{total_prompts}（F4 后 loader 不再 dropout → 应≈0）")
    assert sum(ds.domain_counter.values()) >= bs * n_batches * 0.9
    assert empty_prompts == 0, f"F4 后 loader 不应产生空 prompt（dropout 已移到模型），实际 {empty_prompts}"
    print(f"{n_batches} batch ok in {dt:.1f}s")
    print("\nLOADER SMOKE PASSED（修复后端到端管线正常）")


if __name__ == "__main__":
    main()
