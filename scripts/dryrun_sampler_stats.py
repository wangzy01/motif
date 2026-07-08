"""complete_v1 采样 dry-run 统计（正式版数据线，含 anime）。

复刻并扩展旧 t5_e2e_dryrun（那是 60M-real-only 的独立模拟器）。本脚本**驱动正式版
training/dataline 的真实采样器**（CompleteSampler + DomainPool），对 256/512/1024 三阶段
各抽 N 次（默认 1M），只跑采样层、**不取图**，统计训练实际会收敛到的【边际抽样分布】，
另对小子集做 caption 构建/截断 + 回源验证。产出一份比原图丰富得多的 JSON（配套
scripts/dryrun_report_html.py 渲染 HTML）。

为什么"全池抽样"= 训练实际分布：训练 loader 从 256 大小的滚动 buffer 建池抽样，其边际
分布收敛到 domain_mix × 域内权重(real base_w / anime priority-repeat)，与全池蒙特卡洛等价
（原图也是全池）。anime integer-repeat 在 buffer 内每轮重算，对 1M 样本边际影响 O(1/√N)。

统计内容（每 stage）：
  [1] 域占比（measured vs config domain_mix）
  [2] 源占比 top-N（sub_source；anime 作为独立域单列）
  [3] caption 档占比（real 内 T1-T4 vs 目标 18/27/15/40）
  [4] 各档 captioner 服务占比（尤其 T4：gemini/sharegpt4v/internvl/...）
  [5] 重复统计（distinct/max/mean + 微源 da4k 类的过曝）
  [6] anime 深度：priority 分布 / 整数 repeat 次数分布 / composition_clean 率 / rating /
      NL-vs-tag / caption 路由分布 / cluster 覆盖与 top 簇占比
  [7] caption 实际 token 长度 + 截断率（真 Qwen tokenizer，子集；按 stage/域/档）
  [8] 实际回源验证（子集真读图，成功率 + 实际分辨率/AR + 是否需上采样）
  [9] 数据完整性断言（holdout 排除 / tok<=0 不可用 / gpic@1024=0 / 域占比收敛 / 簇平衡）

只读不写训练数据；纯 CPU。各统计块 try/except 隔离 → 无人值守 jcc 下一块失败不拖垮整体。

⚠️ 域覆盖：本工具只建 monet_real/gpic/synthetic/anime 4 域池，**不含 TextAtlas**（工具早于第 5 域）；
且对 present 域 renorm（domain_target = domain_mix_renorm(present)）会使 TextAtlas 缺失【不表现为 domain
deviation】。故本工具的 domain PASS ≠ 完整 5 域 mix 验证 —— 5 域 mix 须走训练 loader / Part C probe 验证。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# 让 scripts/ 下脚本能 import training.dataline.*
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from training.dataline.caption_build import (build_anime_caption, finalize_real_caption,  # noqa: E402
                                             proxy_tok)
from training.dataline.complete_dataset import (_ANIME_CAP_KEYS, _MONET_SRC_COL,  # noqa: E402
                                                _load_gpic_caption_src, _load_monet_caption_src)
from training.dataline.config import BANDS, STAGES, load_sampling_config  # noqa: E402
from training.dataline.image_fetch import (decode_rgb, fetch_url_bytes, monet_tar_path,  # noqa: E402
                                           read_gpic_bytes, stream_tar_images)
from training.dataline.manifest_index import (CODE_TO_CAPTIONER, YOLO_ADDED_TEXT_AREA_COL,  # noqa: E402
                                              composition_clean_mask, compute_real_global_stats,
                                              list_shards, load_anime_shard, load_real_synth_shard)
from training.dataline.sampler import CompleteSampler, build_pool_from_shards  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

REAL_SYNTH = ("monet_real", "gpic", "synthetic")
DEFAULT_CFG = "/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json"


def stage_px(stage) -> int:
    """stage 标签 → 分辨率像素（REDESIGN 512a/512b 都是 512）。旧纯数字 stage 兜底走 int()。"""
    s = str(stage)
    if s.startswith("1024"):
        return 1024
    if s.startswith("512"):
        return 512
    if s.startswith("256"):
        return 256
    return int(s)


def stage_seed(stage) -> int:
    """stage 标签 → 稳定 int 种子偏移（不用 int(stage) 防 512a/512b ValueError）。
    纯数字 stage 仍返回其数值（旧 256/512/1024 dryrun 种子逐位复现）；512a/512b 走确定性字节哈希且互不相同。"""
    s = str(stage)
    return int(s) if s.isdigit() else int.from_bytes(s.encode(), "big") % 100_000


def _log(msg):
    print(f"[dryrun {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _pct_counter(c: Counter, tot: int, topn: int | None = None) -> dict:
    """Counter → {key: 占比}，按占比降序，可截 topn。"""
    items = sorted(c.items(), key=lambda kv: -kv[1])
    if topn:
        items = items[:topn]
    return {str(k): round(v / tot, 5) for k, v in items if tot}


def _quantiles(vals, qs=(0.5, 0.9, 0.95, 0.99)) -> dict:
    if not len(vals):
        return {}
    a = np.asarray(vals, dtype=np.float64)
    out = {f"p{int(q * 100)}": round(float(np.quantile(a, q)), 2) for q in qs}
    out["mean"] = round(float(a.mean()), 2)
    out["max"] = round(float(a.max()), 2)
    out["n"] = int(a.size)
    return out


# --------------------------------------------------------------------------- #
# 加载（每域一次，stage 无关；elig 过滤在 build_pool 时按 stage 做）
# --------------------------------------------------------------------------- #
def _load_one_rs(args):
    p, d, cfg, stats = args
    a = load_real_synth_shard(p, d, cfg, stats)
    removed = 0
    try:                                            # holdout 已被 load 过滤；单读 holdout 列计被剔数
        ht = pq.read_table(p, columns=["holdout"]).column("holdout").to_pylist()
        removed = int(sum(1 for x in ht if x and int(x) != 0))
    except Exception:
        pass
    return d, a, removed


def load_real_synth(cfg, stats, shards, max_shards, workers):
    """返回 {domain: [shard_arrays...]} + holdout_removed（多线程 overlap S3 I/O）。"""
    out = {d: [] for d in REAL_SYNTH}
    holdout_removed = {d: 0 for d in REAL_SYNTH}
    tasks = []
    for d in REAL_SYNTH:
        sl = shards[d][: max_shards] if max_shards else shards[d]
        tasks += [(p, d, cfg, stats) for p in sl]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d, a, removed in ex.map(_load_one_rs, tasks):
            out[d].append(a)
            holdout_removed[d] += removed
    for d in REAL_SYNTH:
        n = sum(len(a["uid"]) for a in out[d])
        _log(f"加载 {d}: {len(out[d])} shard, {n:,} 行(holdout 剔 {holdout_removed[d]:,})")
    return out, holdout_removed


def _load_one_anime(args):
    p, cfg, nrows, rng_seed = args
    rng = random.Random(rng_seed) if nrows else None
    a = load_anime_shard(p, cfg, with_caption=True, nrows=nrows, rng=rng)
    # has_nl：任一 NL 描述非空
    a["has_nl"] = np.array(
        [bool(r or m or s) for r, m, s in zip(
            a["refined_description"], a["medium_description"], a["short_description"])], dtype=bool)
    # composition_clean：需 quality_z + YOLO（load_anime_shard 不返回）。仅全读(nrows=None)时同序补读
    # 对齐；nrows 采样模式行序无法对齐 → 跳过该项。
    if nrows is None:
        try:
            t2 = pq.read_table(p, columns=["quality_z", YOLO_ADDED_TEXT_AREA_COL])
            qz = np.array([x if x is not None else 0.0
                           for x in t2.column("quality_z").to_pylist()], dtype=np.float64)
            yolo = np.array([x if x is not None else np.inf
                             for x in t2.column(YOLO_ADDED_TEXT_AREA_COL).to_pylist()], dtype=np.float64)
            a["composition_clean"] = composition_clean_mask(a["tags_general"], yolo, qz)
            a["quality_z"] = qz
        except Exception as e:
            print(f"  ⚠️ anime composition_clean 补读失败({p}): {e}", flush=True)
    return a


def load_anime(cfg, shards, max_shards, nrows_per_shard, seed, workers):
    """加载 anime 源 shard（with_caption=True）。多线程 overlap S3 I/O。返回 [shard_arrays...]。"""
    sl = shards["anime"][: max_shards] if max_shards else shards["anime"]
    tasks = [(p, cfg, nrows_per_shard, seed + i) for i, p in enumerate(sl)]
    arrs = []
    with ThreadPoolExecutor(max_workers=min(workers, max(len(tasks), 1))) as ex:
        for a in ex.map(_load_one_anime, tasks):
            arrs.append(a)
            _log(f"加载 anime shard: {len(a['key']):,} 行 (累计 {len(arrs)}/{len(tasks)})")
    return arrs


# --------------------------------------------------------------------------- #
# 主抽样 + 统计（单 stage）
# --------------------------------------------------------------------------- #
def run_stage(cfg, stage, rs_arrays, anime_arrays, n_draws, seed):
    """全池抽样 N 次，累计 [1]-[6] 统计。返回 (stage_stats, pools, sampler, draw_counts)。
    pools/sampler 复用给后续 caption / 回源子集抽样。"""
    dm = cfg.domain_mix[stage]
    pools = {}
    pool_rng = np.random.default_rng(seed + 777)
    for d in REAL_SYNTH:
        if dm.get(d, 0.0) <= 0 or not rs_arrays.get(d):
            continue
        p = build_pool_from_shards(d, rs_arrays[d], stage)
        if p.n:
            pools[d] = p
    if dm.get("anime", 0.0) > 0 and anime_arrays:
        ap = build_pool_from_shards("anime", anime_arrays, stage,
                                    max_repeat=cfg.anime_sampling.max_repeat,
                                    extra_frac=cfg.anime_sampling.extra_frac, rng=pool_rng)
        if ap.n:
            pools["anime"] = ap
    assert pools, f"stage {stage} 无可用 pool"
    pool_sizes = {d: int(p.n) for d, p in pools.items()}
    _log(f"[{stage}] pool 行数: {pool_sizes}")

    sampler = CompleteSampler(cfg, stage, seed=seed)
    # 计数器
    dom_ct = Counter()
    src_ct = Counter()                          # sub_source（非 anime）
    band_ct = Counter()                         # real 内 band（monet/gpic）
    cap_by_band = {b: Counter() for b in BANDS}
    draw_counts = {d: np.zeros(p.n, dtype=np.int32) for d, p in pools.items()}
    # anime 深度
    an_prio, an_clean, an_rating, an_nl, an_cluster = [], 0, Counter(), 0, Counter()
    an_total = 0
    an_clean_avail = "composition_clean" in pools["anime"].arrays if "anime" in pools else False

    t0 = time.time()
    for i in range(n_draws):
        d, idx, band = sampler.sample_one(pools)
        dom_ct[d] += 1
        draw_counts[d][idx] += 1
        arr = pools[d].arrays
        if d == "anime":
            an_total += 1
            an_prio.append(float(arr["priority"][idx]))
            an_cluster[arr["cluster"][idx]] += 1
            if arr.get("has_nl") is not None:
                an_nl += int(bool(arr["has_nl"][idx]))
            if an_clean_avail:
                an_clean += int(bool(arr["composition_clean"][idx]))
            r = arr.get("rating_bucket")
            if r is not None:
                an_rating[r[idx]] += 1
        else:
            src_ct[arr["sub_source"][idx]] += 1
            if band >= 0:                       # monet/gpic band-first（synthetic band=-1 注入）
                bname = BANDS[band]
                band_ct[bname] += 1
                code = int(arr["band_captioner"][idx][band])
                cap_by_band[bname][CODE_TO_CAPTIONER.get(code, f"code{code}")] += 1
        if (i + 1) % 250000 == 0:
            _log(f"[{stage}] 抽样 {i + 1:,}/{n_draws:,} ({(i + 1) / (time.time() - t0):.0f}/s)")

    tot = sum(dom_ct.values())
    band_tot = sum(band_ct.values())
    # 域占比 vs 目标（renorm 到 present）
    target = cfg.domain_mix_renorm(stage, set(pools.keys()))
    st = {
        "draws": tot,
        "pool_sizes": pool_sizes,
        "domain_share": {k: round(v / tot, 5) for k, v in sorted(dom_ct.items(), key=lambda kv: -kv[1])},
        "domain_target": {k: round(v, 5) for k, v in target.items()},
        "source_share": _pct_counter(src_ct, sum(src_ct.values()), topn=20),
        "band_share_real": {b: round(band_ct.get(b, 0) / band_tot, 5) for b in BANDS} if band_tot else {},
        "band_target": dict(cfg.band_mix_real),
        "captioner_by_band": {b: _pct_counter(cap_by_band[b], sum(cap_by_band[b].values()))
                              for b in BANDS},
    }
    # [5] 重复统计
    st["repeat"] = _repeat_stats(pools, draw_counts)
    # [6] anime 深度
    if an_total:
        st["anime"] = _anime_deep(cfg, stage, pools["anime"], an_prio, an_clean, an_clean_avail,
                                  an_rating, an_nl, an_total, an_cluster)
    return st, pools, sampler, draw_counts


def _repeat_stats(pools, draw_counts):
    out = {"by_domain": {}, "micro_sources": []}
    all_distinct = all_max = 0
    all_draws = total_pool = 0
    for d, cnt in draw_counts.items():
        drawn = cnt[cnt > 0]
        nd = int(drawn.size)
        out["by_domain"][d] = {
            "draws": int(cnt.sum()), "distinct_drawn": nd, "pool_n": int(cnt.size),
            "coverage": round(nd / max(int(cnt.size), 1), 4),
            "max_repeat": int(drawn.max()) if nd else 0,
            "mean_repeat": round(float(drawn.mean()), 3) if nd else 0.0,
        }
        all_draws += int(cnt.sum())
        all_distinct += nd
        all_max = max(all_max, int(drawn.max()) if nd else 0)
        total_pool += int(cnt.size)
    out["overall"] = {"draws": all_draws, "distinct_drawn": all_distinct, "pool_n": total_pool,
                      "max_repeat": all_max,
                      "mean_repeat": round(all_draws / max(all_distinct, 1), 3)}
    # 微源（da4k 类）：按 sub_source 聚合 real/synth 各源的曝光/distinct/max（暴露过曝小源）
    for d in ("monet_real", "gpic", "synthetic"):
        if d not in pools:
            continue
        sub = pools[d].arrays.get("sub_source")
        cnt = draw_counts[d]
        if sub is None:
            continue
        by_src_draw = defaultdict(int)
        by_src_pool = defaultdict(int)
        by_src_distinct = defaultdict(int)
        by_src_max = defaultdict(int)
        for i in range(cnt.size):
            s = sub[i]
            by_src_pool[s] += 1
            c = int(cnt[i])
            if c:
                by_src_draw[s] += c
                by_src_distinct[s] += 1
                if c > by_src_max[s]:
                    by_src_max[s] = c
        for s, pooln in by_src_pool.items():
            dr = by_src_draw.get(s, 0)
            out["micro_sources"].append({
                "source": str(s), "domain": d, "pool_n": pooln, "draws": dr,
                "distinct_drawn": by_src_distinct.get(s, 0), "max_repeat": by_src_max.get(s, 0),
                "exposure_x": round(dr / pooln, 4) if pooln else 0.0,  # 平均每张被抽次数
            })
    # 按过曝(exposure_x)降序，方便一眼看哪些微源被重复抽
    out["micro_sources"].sort(key=lambda r: -r["exposure_x"])
    return out


def _anime_deep(cfg, stage, pool, an_prio, an_clean, an_clean_avail, an_rating, an_nl, an_total,
                an_cluster):
    arr = pool.arrays
    deep = {
        "drawn": an_total,
        "priority": _quantiles(an_prio),
        "rating_dist": _pct_counter(an_rating, sum(an_rating.values())) if an_rating else {},
        "has_nl_rate": round(an_nl / an_total, 4),
        "composition_clean_rate": (round(an_clean / an_total, 4) if an_clean_avail else None),
    }
    # 整数 repeat 次数分布（pool 内每行的 repeat count = _repeat_w，仅当启用 integer-repeat）
    rw = getattr(pool, "_repeat_w", None)
    if rw is not None:
        rw_int = np.rint(rw).astype(int)
        deep["repeat_count_dist"] = {str(int(k)): int(v) for k, v in
                                     sorted(Counter(rw_int.tolist()).items())}
        deep["repeat_count_mean"] = round(float(rw.mean()), 4)
    # caption 路由分布（解析式：按 drawn 频次 × 该行 has_nl 对应的路由权重表加权期望）
    try:
        ac = cfg.anime_caption
        route_acc = Counter()
        has_nl = arr.get("has_nl")
        if has_nl is not None:
            # 用整池 has_nl 比例加权（drawn 已按 priority，但路由只依赖 has_nl/stage）
            for nl_flag, w in (("nl", float(has_nl.mean())), ("no", float(1 - has_nl.mean()))):
                tbl = ac.route_weights(stage, nl_flag == "nl")
                z = sum(tbl.values())
                for rt, rv in tbl.items():
                    route_acc[rt] += w * (rv / z)
            tot = sum(route_acc.values())
            deep["caption_route_expected"] = {k: round(v / tot, 4) for k, v in route_acc.items()}
    except Exception as e:
        deep["caption_route_expected_error"] = str(e)
    # cluster 覆盖与平衡
    total_clusters = len(set(arr["cluster"].tolist()))
    drawn_clusters = sorted(an_cluster.items(), key=lambda kv: -kv[1])
    cl_tot = sum(an_cluster.values())
    top10 = sum(v for _, v in drawn_clusters[:10])
    deep["cluster"] = {
        "total_clusters_in_pool": total_clusters,
        "distinct_clusters_drawn": len(an_cluster),
        "top10_share": round(top10 / cl_tot, 4) if cl_tot else 0.0,
        "top10": [{"cluster": str(c), "share": round(v / cl_tot, 5)} for c, v in drawn_clusters[:10]],
    }
    return deep


# --------------------------------------------------------------------------- #
# [7] caption 长度 + 截断率（子集）
# --------------------------------------------------------------------------- #
def caption_stats(cfg, stage, pools, sampler, n_sample, tokenizer, budget, seed, max_src):
    """caption 长度/截断子集。anime 从内存构建（快、全处理）；real/synth 需回源源 parquet，
    全量读一个源 parquet 才能取几行 → 对【按样本数最多的 src_shard】优先加载、总加载数封顶
    max_src（其余样本计 skipped_src_cap），把 I/O 控在 O(max_src) 次大读内。"""
    from training.dataline.manifest_index import CAPTIONER_CODE
    rng = random.Random(seed + 13)
    samples = [sampler.sample_one(pools) for _ in range(n_sample)]
    # anime 全局 general tag 频率（freq dropout 用；训练是 shard-local，这里全局近似）
    anime_freq = Counter()
    if "anime" in pools:
        for tg in pools["anime"].arrays.get("tags_general", []):
            if tg:
                anime_freq.update(tg)

    by_dom = defaultdict(lambda: {"proxy": [], "real_tok": [], "trunc": 0, "n": 0, "empty": 0})
    by_band = defaultdict(lambda: {"real_tok": [], "trunc": 0, "n": 0})

    def _accum(d, cap, band):
        slot = by_dom[d]
        slot["n"] += 1
        if not cap:
            slot["empty"] += 1
            return
        slot["proxy"].append(proxy_tok(cap))
        if tokenizer is not None:
            rt = len(tokenizer(cap, add_special_tokens=False)["input_ids"])
            slot["real_tok"].append(rt)
            tr = int(rt > budget)
            slot["trunc"] += tr
            if d == "monet_real" and band >= 0:       # band 维度仅 monet（gpic 单 i1、synth 注入）
                bb = by_band[BANDS[band]]
                bb["real_tok"].append(rt)
                bb["trunc"] += tr
                bb["n"] += 1

    # anime：内存构建，全处理
    if "anime" in pools:
        arr = pools["anime"].arrays
        for d, idx, band in samples:
            if d != "anime":
                continue
            try:
                f = {k: arr[k][idx] for k in _ANIME_CAP_KEYS if k in arr}
                cap = build_anime_caption(f, stage, cfg.anime_caption, rng, anime_freq)
            except Exception:
                cap = None
            _accum("anime", cap, band)

    # real/synth：按 (domain, src_root, src_shard) 分组，大组优先，加载数封顶
    groups = defaultdict(list)
    for d, idx, band in samples:
        if d == "anime":
            continue
        arr = pools[d].arrays
        groups[(d, arr["src_root"][idx], arr["src_shard"][idx])].append((idx, band))
    loaded = skipped = 0
    for (d, root, shard), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if loaded >= max_src:
            skipped += len(items)
            for _ in items:
                by_dom[d]["n"] += 1
                by_dom[d]["empty"] += 1   # 记为未取到（src 加载封顶跳过）
            continue
        try:
            cs = (_load_gpic_caption_src(root, shard) if d == "gpic"
                  else _load_monet_caption_src(root, shard))
        except Exception:
            cs = None
        loaded += 1
        arr = pools[d].arrays
        for idx, band in items:
            cap = None
            if cs is not None:
                sk = arr["src_key"][idx]
                if d == "gpic":
                    cap = finalize_real_caption(cs.get(sk, ("", None))[0], "i1", cfg.caption_truncate_tok)
                elif d == "synthetic":
                    cap = finalize_real_caption((cs.get(sk, {}) or {}).get(CAPTIONER_CODE["original"]),
                                                "original", cfg.caption_truncate_tok)
                else:
                    bc = arr["band_captioner"][idx]
                    code = int(bc[band]) if band >= 0 else 0
                    if code == 0:
                        nz = [int(c) for c in bc if int(c) > 0]
                        code = nz[0] if nz else CAPTIONER_CODE["original"]
                    cap = finalize_real_caption((cs.get(sk, {}) or {}).get(code),
                                                CODE_TO_CAPTIONER.get(code, "original"),
                                                cfg.caption_truncate_tok)
            _accum(d, cap, band)

    out = {"sample_n": len(samples), "budget_tok": budget, "src_parquets_loaded": loaded,
           "src_cap_skipped": skipped, "by_domain": {}, "by_band": {}}
    for d, s in by_dom.items():
        rec = {"n": s["n"], "empty": s["empty"], "proxy_tok": _quantiles(s["proxy"])}
        if tokenizer is not None and s["real_tok"]:
            rec["real_tok"] = _quantiles(s["real_tok"])
            rec["trunc_rate"] = round(s["trunc"] / max(len(s["real_tok"]), 1), 4)
        out["by_domain"][d] = rec
    for b, s in by_band.items():
        if s["n"]:
            out["by_band"][b] = {"n": s["n"], "real_tok": _quantiles(s["real_tok"]),
                                 "trunc_rate": round(s["trunc"] / s["n"], 4)}
    return out


# --------------------------------------------------------------------------- #
# [8] 回源验证（真读图）
# --------------------------------------------------------------------------- #
def backfetch_stats(cfg, stage, pools, sampler, per_domain, anime_cache, seed):
    target = int(per_domain)
    if target <= 0:
        return {"skipped": True}
    spx = stage_px(stage)               # 512a/512b → 512（不用 int(stage) 防 ValueError）
    want = {d: [] for d in pools}
    rng = random.Random(seed + 29)
    tries = 0
    # 抽到每域各 target 个（带上限防卡死）
    while any(len(v) < target for v in want.values()) and tries < target * 40 * max(len(pools), 1):
        d, idx, band = sampler.sample_one(pools)
        if len(want[d]) < target:
            want[d].append(idx)
        tries += 1
    out = {}
    for d, idxs in want.items():
        arr = pools[d].arrays
        res = {"tried": 0, "ok": 0, "fail": 0, "min_dim": [], "ar": [], "below_stage": 0}
        if d in ("monet_real", "synthetic"):
            # 按 tar 分组流式读
            groups = defaultdict(list)
            for idx in idxs:
                groups[(arr["src_root"][idx], arr["src_shard"][idx])].append(idx)
            for (root, shard), gi in groups.items():
                wanted = {arr["src_key"][idx] for idx in gi}
                got = {}
                try:
                    got = dict(stream_tar_images(monet_tar_path(root, shard), wanted))
                except Exception:
                    pass
                for idx in gi:
                    res["tried"] += 1
                    raw = got.get(arr["src_key"][idx])
                    _tally_img(res, raw, spx)
        elif d == "gpic":
            for idx in idxs:
                res["tried"] += 1
                try:
                    raw = read_gpic_bytes(arr["src_root"][idx], arr["src_shard"][idx],
                                          arr["src_key"][idx])
                except Exception:
                    raw = None
                _tally_img(res, raw, spx)
        elif d == "anime":
            for idx in idxs:
                res["tried"] += 1
                try:
                    raw = fetch_url_bytes(arr["s3key"][idx], cache_dir=anime_cache)
                except Exception:
                    raw = None
                _tally_img(res, raw, spx)
        res["success_rate"] = round(res["ok"] / max(res["tried"], 1), 4)
        res["min_dim_q"] = _quantiles(res.pop("min_dim"))
        res["ar_q"] = _quantiles(res.pop("ar"))
        out[d] = res
    return out


def _tally_img(res, raw, min_px):
    if not raw:
        res["fail"] += 1
        return
    img = decode_rgb(raw)
    if img is None:
        res["fail"] += 1
        return
    res["ok"] += 1
    w, h = img.size
    res["min_dim"].append(min(w, h))
    res["ar"].append(round(max(w, h) / max(min(w, h), 1), 3))
    if min(w, h) < min_px:
        res["below_stage"] += 1


# --------------------------------------------------------------------------- #
# [9] 完整性断言
# --------------------------------------------------------------------------- #
def integrity_stats(cfg, stage, st, pools, holdout_removed):
    out = {"holdout_removed": holdout_removed, "assertions": []}

    def assrt(name, ok, detail):
        out["assertions"].append({"name": name, "ok": bool(ok), "detail": detail})

    # tok<=0 / 无可用 captioner：池内 served_mask==0（real/synth）
    unusable = {}
    for d in ("monet_real", "gpic"):
        if d in pools:
            sm = pools[d].arrays.get("served_mask")
            if sm is not None:
                n0 = int((sm == 0).sum())
                unusable[d] = {"count": n0, "pool_n": int(sm.size),
                               "frac": round(n0 / max(int(sm.size), 1), 5)}
    out["unusable_no_captioner"] = unusable
    # gpic@512/1024 不应被抽（domain_mix=0）
    gpic_drawn = st["domain_share"].get("gpic", 0.0)
    if stage_px(stage) in (512, 1024):
        expect0 = cfg.domain_mix[stage].get("gpic", 0.0) <= 0
        if stage == "1024":
            assrt("gpic@1024 不参与", expect0 and gpic_drawn == 0.0,
                  f"domain_mix.gpic={cfg.domain_mix[stage].get('gpic')}, drawn_share={gpic_drawn}")
    # 域占比收敛（measured vs target，最大偏差）
    devs = {d: round(st["domain_share"].get(d, 0.0) - st["domain_target"].get(d, 0.0), 5)
            for d in st["domain_target"]}
    out["domain_mix_dev"] = devs
    max_dev = max((abs(v) for v in devs.values()), default=0.0)
    assrt("域占比收敛(|dev|<0.02)", max_dev < 0.02, f"max|dev|={max_dev}")
    # band 收敛
    if st.get("band_share_real"):
        bdev = {b: round(st["band_share_real"].get(b, 0) - st["band_target"].get(b, 0), 5)
                for b in BANDS}
        out["band_dev"] = bdev
        assrt("band 占比收敛(|dev|<0.03)", max(abs(v) for v in bdev.values()) < 0.03,
              f"dev={bdev}")
    # anime 簇平衡（top10 占比不应过高）
    if st.get("anime", {}).get("cluster"):
        t10 = st["anime"]["cluster"]["top10_share"]
        assrt("anime 簇平衡(top10<0.15)", t10 < 0.15, f"top10_share={t10}")
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--stages", default="256,512,1024")
    ap.add_argument("--n-draws", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="experiments/results/dryrun/sampler_stats.json")
    ap.add_argument("--max-shards", type=int, default=0, help="每域 shard 上限(0=全部; smoke 用)")
    ap.add_argument("--anime-nrows-per-shard", type=int, default=0,
                    help="anime 每 shard 读行上限(0=全读; >0 启用随机 row-group 采样, 但跳过 composition_clean)")
    ap.add_argument("--caption-sample", type=int, default=30000, help="caption 长度/截断子集大小(每 stage)")
    ap.add_argument("--caption-max-src", type=int, default=80,
                    help="real/synth caption 回源时最多加载多少个源 parquet(控 I/O; anime 不受限)")
    ap.add_argument("--workers", type=int, default=16, help="shard 加载并发线程数")
    ap.add_argument("--tokenizer", default="/fsx/models/Qwen3-VL-2B-Instruct",
                    help="真 token 截断率用(空串=只算 proxy)")
    ap.add_argument("--template-reserve", type=int, default=41)
    ap.add_argument("--backfetch-per-domain", type=int, default=300, help="回源验证每域样本数(0=跳过)")
    ap.add_argument("--anime-cache", default="/local/evan/anime_img_cache")
    ap.add_argument("--stats-cache", default="/local/evan/mot_stats/monet_global_stats.json")
    args = ap.parse_args()

    max_shards = args.max_shards or None
    anime_nrows = args.anime_nrows_per_shard or None
    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    _log(f"config={args.config}")
    cfg = load_sampling_config(args.config)
    # stage 过滤改为按 config 实际 domain_mix 键（支持 REDESIGN 512a/512b；旧 STAGES 硬编码 256/512/1024 已不适用）
    valid = list(cfg.domain_mix.keys())
    stages = [s for s in requested if s in valid]
    dropped = [s for s in requested if s not in valid]
    if dropped:
        _log(f"⚠️ 跳过 config 中不存在的 stage: {dropped}（config domain_mix 有: {valid}）")
    if not stages:
        _log(f"❌ 无有效 stage（requested={requested}, config 有={valid}）")
        return
    shards = list_shards(cfg)
    _log(f"shard 数: " + ", ".join(f"{d}={len(v)}" for d, v in shards.items()))

    # tokenizer（caption 截断率）；budget 改为 per-stage（mtml_for(stage)，REDESIGN 256→343/512→471）
    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            budgets = {s: int(cfg.anime_caption.mtml_for(s)) - int(args.template_reserve) for s in stages}
            _log(f"tokenizer 就绪, per-stage caption budget={budgets} tok")
        except Exception as e:
            _log(f"⚠️ tokenizer 加载失败({e})，caption 仅算 proxy")

    # 全局 monet 源/簇统计（rule #7；缓存）
    _log("计算 monet 全局源/簇统计…")
    full_monet = shards["monet_real"][: max_shards] if max_shards else shards["monet_real"]
    stats = compute_real_global_stats(full_monet, cache_path=args.stats_cache)
    _log(f"全局统计: {len(stats['source_counts'])} 源, {len(stats['cluster_sizes'])} 簇")

    # 加载（一次，多线程）
    t_load = time.time()
    rs_arrays, holdout_removed = load_real_synth(cfg, stats, shards, max_shards, args.workers)
    anime_arrays = load_anime(cfg, shards, max_shards, anime_nrows, args.seed, args.workers)
    _log(f"全部加载完成，用时 {time.time() - t_load:.0f}s")

    report = {
        "meta": {
            "config": args.config, "n_draws": args.n_draws, "seed": args.seed, "stages": stages,
            "tokenizer": args.tokenizer or None, "caption_sample": args.caption_sample,
            "backfetch_per_domain": args.backfetch_per_domain,
            "max_shards": max_shards, "anime_nrows_per_shard": anime_nrows,
            "shard_counts": {d: len(v) for d, v in shards.items()},
            "domain_mix": {s: cfg.domain_mix[s] for s in stages},
            "source_alpha": cfg.source_alpha, "cluster_beta": cfg.cluster_beta,
            "holdout_removed": holdout_removed,
            "note": "全池蒙特卡洛 = 训练 loader buffer 抽样收敛到的边际分布；纯采样层不取图(回源验证除外)。",
        },
        "stages": {},
    }

    for stage in stages:
        _log(f"==== STAGE {stage} ====")
        try:
            st, pools, sampler, draw_counts = run_stage(
                cfg, stage, rs_arrays, anime_arrays, args.n_draws, args.seed + stage_seed(stage))
        except Exception as e:
            _log(f"[{stage}] 主抽样失败: {e}\n{traceback.format_exc()}")
            report["stages"][stage] = {"error": str(e)}
            continue
        for name, fn in [
            ("caption_len", lambda: caption_stats(cfg, stage, pools, sampler, args.caption_sample,
                                                  tokenizer,
                                                  (int(cfg.anime_caption.mtml_for(stage)) - int(args.template_reserve)
                                                   if tokenizer is not None else None),
                                                  args.seed + stage_seed(stage),
                                                  args.caption_max_src)),
            ("backfetch", lambda: backfetch_stats(cfg, stage, pools, sampler,
                                                  args.backfetch_per_domain, args.anime_cache,
                                                  args.seed + stage_seed(stage))),
        ]:
            try:
                _log(f"[{stage}] {name}…")
                st[name] = fn()
            except Exception as e:
                _log(f"[{stage}] {name} 失败: {e}\n{traceback.format_exc()}")
                st[name] = {"error": str(e)}
        try:
            st["integrity"] = integrity_stats(cfg, stage, st, pools, holdout_removed)
        except Exception as e:
            st["integrity"] = {"error": str(e)}
        report["stages"][stage] = st
        # 每 stage 落盘一次（无人值守容错）
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1, default=str)
        _log(f"[{stage}] 已写 {args.out}")

    _log(f"完成。结果: {args.out}")


if __name__ == "__main__":
    main()
