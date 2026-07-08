#!/usr/bin/env python3
"""TextAtlas 第5域接入单测（CPU，分钟级；读 1 个 sidecar + 少量真图 decode）。

覆盖：
  T1 load_textatlas_shard：schema/elig（256/512/1024）/uid 格式/base_w=1/served_mask=0。
  T2 iter_textatlas_shard：源 parquet 内联图 join → u8 形状/类型；caption join 正确性
     （rec.captions 与 sidecar 按 uid 对齐）；随机 row-group 去偏不丢 elig 行。
  T3 build_textatlas_caption：5 选 1（非空）、全空→""、确定性（同 rng 种子）。
  T4 sampler 注入：domain_mix 含 textatlas → CompleteSampler.sample_one 能抽中 textatlas，
     且边际占比 ≈ 目标（注入域块）。
  T5 路径回退：local_root 缺失 → 自动回退 cfg.textatlas_source_root（S3），仍能读图。

用法: venv-gpu/bin/python scripts/test_textatlas_dataline.py
坑点#28：纯 CPU 但读 S3 + decode → 正式跑上 JCC（jobs/textatlas_dataline_test.yaml）；本地快验少量行可。
"""
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from training.dataline.caption_build import build_textatlas_caption
from training.dataline.complete_dataset import iter_textatlas_shard
from training.dataline.config import ALL_DOMAINS, load_sampling_config
from training.dataline.manifest_index import (list_shards, load_textatlas_shard,
                                              textatlas_subset_shard)
from training.dataline.sampler import CompleteSampler, DomainPool

CFG_PATH = "/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1_textatlas.json"


def _ok(name):
    print(f"  ✓ {name}", flush=True)


def test_load(cfg):
    """T1：load_textatlas_shard 字段/elig/uid 格式。"""
    sc = sorted(glob.glob(f"{cfg.manifest_textatlas}/CleanTextSynth/*.parquet"))[0]
    subset, _ = textatlas_subset_shard(sc)
    arr = load_textatlas_shard(sc, "512")
    n = len(arr["uid"])
    assert n > 0
    assert arr["base_w"].shape == (n,) and (arr["base_w"] == 1.0).all()
    assert (arr["served_mask"] == 0).all()
    # elig 包含关系：1024 ⊆ 512 ⊆ 256
    assert (arr["elig_1024"] <= arr["elig_512"]).all()
    assert (arr["elig_512"] <= arr["elig_256"]).all()
    # uid == "<Subset>:<image_path>"
    i = int(np.where(arr["elig_512"])[0][0])
    assert arr["uid"][i] == f"{subset}:{arr['image_path'][i]}"
    for c in ("caption1", "caption2", "caption3", "caption4", "caption5"):
        assert len(arr[c]) == n
    _ok(f"T1 load_textatlas_shard (rows={n}, elig512={int(arr['elig_512'].sum())})")
    return sc, arr


def test_iter_and_caption_join(cfg, sc, arr):
    """T2：iter 内联图 join + caption 按 uid 对齐。"""
    # 建 uid->caption1 真值表（sidecar）
    truth = {arr["uid"][i]: arr["caption1"][i] for i in range(len(arr["uid"]))}
    rng = random.Random(0)
    recs = list(iter_textatlas_shard(sc, cfg, "512", 512, cfg.textatlas_source_root,
                                     local_root=None, max_rows=8, skip_image=False, rng=rng))
    assert len(recs) == 8, len(recs)
    for r in recs:
        assert r["domain"] == "textatlas" and r["served_mask"] == 0 and r["base_w"] == 1.0
        assert r["u8"].shape == (512, 512, 3) and r["u8"].dtype == np.uint8
        assert len(r["captions"]) == 5
        # caption join 正确性：rec.captions[0] 必须 == 该 uid 的 sidecar caption1
        assert r["uid"] in truth, f"uid {r['uid']} 不在 sidecar"
        assert r["captions"][0] == truth[r["uid"]], "caption join 错位！"
    _ok(f"T2 iter_textatlas_shard + caption join ({len(recs)} recs, u8 512²×3)")


def test_build_caption():
    """T3：5 选 1（非空）/全空→''/确定性。"""
    caps = ("alpha cat", "beta dog", "", "gamma bird", "")
    seen = {build_textatlas_caption(caps, random.Random(s)) for s in range(50)}
    assert seen <= {"alpha cat", "beta dog", "gamma bird"}, seen
    assert len(seen) >= 2, "应能选到多个变体"   # 随机性体现
    assert "" not in seen, "不应选到空串"
    assert build_textatlas_caption(("", "", "", "", ""), random.Random(0)) == ""
    # 确定性：同 rng 序列同结果
    assert build_textatlas_caption(caps, random.Random(7)) == build_textatlas_caption(caps, random.Random(7))
    _ok(f"T3 build_textatlas_caption (variants seen={sorted(seen)})")


def test_sampler_injection(cfg):
    """T4：sampler 注入域块能抽中 textatlas，边际占比 ≈ 目标。"""
    assert "textatlas" in ALL_DOMAINS
    stage = "512"
    target = cfg.domain_mix[stage]["textatlas"]
    # 造 5 域各一个非空 pool（textatlas 注入域：served_mask=0 / base_w=1）
    pools = {}
    for d in ("anime", "monet_real", "gpic", "synthetic", "textatlas"):
        n = 50
        if d == "anime":
            arrays = {"key": np.arange(n), "priority": np.ones(n)}
        else:
            # monet/gpic 需服务各 band 才能被 band-first 抽到；给全 band served
            sm = 0 if d in ("synthetic", "textatlas") else 0b1111
            arrays = {"uid": np.arange(n), "served_mask": np.full(n, sm, np.uint8),
                      "base_w": np.ones(n)}
        pools[d] = DomainPool(d, arrays)
    smp = CompleteSampler(cfg, stage, seed=0)
    from collections import Counter
    c = Counter()
    N = 40000
    for _ in range(N):
        dom, idx, band = smp.sample_one(pools)
        c[dom] += 1
    frac = c["textatlas"] / N
    assert c["textatlas"] > 0, "textatlas 从未被抽中！"
    assert abs(frac - target) < 0.02, f"textatlas 边际 {frac:.3f} 偏离目标 {target} 过多"
    _ok(f"T4 sampler 注入 textatlas: realized={frac:.3f} target={target} | all={dict(c)}")


def test_path_fallback(cfg, sc, arr):
    """T5：local_root 指向不存在目录 → 自动回退 S3 源，仍能读图。"""
    recs = list(iter_textatlas_shard(sc, cfg, "512", 512, cfg.textatlas_source_root,
                                     local_root="/nonexistent/textatlas_local", max_rows=3,
                                     skip_image=False, rng=random.Random(1)))
    assert len(recs) == 3 and all(r["u8"].shape == (512, 512, 3) for r in recs)
    _ok("T5 路径回退 local→S3（local 缺失仍读到图）")


def test_list_shards(cfg):
    """list_shards 含 textatlas 且只纳入配置的子集（不含 CoverBook）。"""
    sh = list_shards(cfg)
    assert "textatlas" in sh and len(sh["textatlas"]) > 0
    subsets = {textatlas_subset_shard(p)[0] for p in sh["textatlas"]}
    assert "CoverBook" not in subsets, "CoverBook 不该被纳入"
    assert subsets == set(cfg.textatlas_subsets), f"子集不符: {subsets}"
    _ok(f"list_shards textatlas: {len(sh['textatlas'])} shards, {len(subsets)} subsets (no CoverBook)")


def main():
    print("== TextAtlas 第5域接入单测 ==", flush=True)
    cfg = load_sampling_config(CFG_PATH)
    test_list_shards(cfg)
    sc, arr = test_load(cfg)
    test_iter_and_caption_join(cfg, sc, arr)
    test_build_caption()
    test_sampler_injection(cfg)
    test_path_fallback(cfg, sc, arr)
    print("== ALL PASS ==", flush=True)


if __name__ == "__main__":
    main()
