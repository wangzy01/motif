#!/usr/bin/env python3
"""OFFLINE_ARB_V1 · 把 Phase-3a 的真 N_matrix.parquet 转成 planner_input.json(§1.5 契约)。

输入:
  --n-matrix   offline_arb_bucketstats_v1/N_matrix.parquet(bucket_h/bucket_w/ar + 每域计数列)
  --config     stage 配置 yaml(读 multi_ar 参数重建同一张桶表校验一致性 + square_only + sampling_config→domain_mix)
  --total-steps/--world-size/--per-rank-bs/--seed/--plan-id/--block-steps
  --r-max      JSON 字符串,缺省用设计文档 §10 建议值。

输出:--out planner_input.json(§1.5 全字段)。桶表以 config 重建为准,并逐行核对
N_matrix 的 bucket_h/w 与其一致(不一致即报错——防 N 矩阵与桶表版本漂移)。
只读输入,唯一写出 --out。
"""
from __future__ import annotations

import argparse
import json
import sys

import pyarrow.parquet as pq
from omegaconf import OmegaConf

sys.path.insert(0, "/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop")
from training.dataline.config import load_sampling_config          # noqa: E402
from training.dataline.multi_ar import build_adaptive_buckets, square_index  # noqa: E402

DOMAINS = ["anime", "monet_real", "gpic", "synthetic", "textatlas"]   # 与 N_matrix/planner 列序契约一致
R_MAX_DEFAULT = {"anime": 20, "monet_real": 20, "gpic": 20, "synthetic": 200, "textatlas": 200}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-matrix", required=True)
    ap.add_argument("--config", default="configs/stage256_redesign_naifu_arb_v1.yaml")
    ap.add_argument("--stage", default="256")
    ap.add_argument("--total-steps", type=int, required=True)
    ap.add_argument("--world-size", type=int, default=16)
    ap.add_argument("--per-rank-bs", type=int, default=144)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--plan-id", required=True)
    ap.add_argument("--block-steps", type=int, default=1024)
    ap.add_argument("--r-max", default=None, help="JSON dict,缺省用 §10 建议值")
    ap.add_argument("--floor", type=float, default=0.0,
                    help="bucket_prob_floor 折叠(0=不折):P(B)=Σ_d mix_d·N[B,d]/N[d] < floor 的桶"
                         "经 nearest_allowed_bucket 折进保留桶——与生产 L2 assignment 同口径")
    ap.add_argument("--catalog-header", default=None,
                    help="生产模式:指向 offline_arb_catalog_v1/header.json。N 取"
                         " N.materialized_after_exclusion(tar-join 排除后的落盘精确值,"
                         "已含 floor 折叠——与 --floor 互斥),catalog_hash/assignment_hash"
                         " 一并写入 planner_input,使 plan 与该 catalog 真 hash 绑定"
                         "(loader 拒 proxy hash)。--n-matrix 仍须给出,用作交叉校验")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.catalog_header and args.floor > 0:
        raise SystemExit("--catalog-header 与 --floor 互斥:catalog N 已是折叠+排除后的最终值")

    cfg = OmegaConf.load(args.config)
    mar = cfg.dataset.multi_ar
    buckets = build_adaptive_buckets(
        int(mar.target_area), int(mar.divisor),
        min_area_ratio=float(mar.adaptive_min_area_ratio),
        max_aspect=float(mar.adaptive_max_aspect))   # no_upscale 是指派期参数,桶表构建不收
    sq = square_index(buckets)
    square_only = [str(d) for d in mar.square_only_domains]
    samp = load_sampling_config(str(cfg.dataset.sampling_config))
    mix = {d: float(samp.domain_mix[args.stage][d]) for d in DOMAINS}

    tbl = pq.read_table(args.n_matrix)
    cols = {c: tbl.column(c).to_pylist() for c in tbl.column_names}
    nb = len(cols["bucket_h"])
    if nb != len(buckets):
        raise SystemExit(f"N_matrix 桶数 {nb} != config 桶表 {len(buckets)}")
    for i in range(nb):
        if (cols["bucket_h"][i], cols["bucket_w"][i]) != buckets[i]:
            raise SystemExit(f"桶 {i} 尺寸不一致:N_matrix=({cols['bucket_h'][i]},{cols['bucket_w'][i]}) config={buckets[i]}")
    # N_matrix 域计数列名 = "n_"+域名(phase3a_bucket_stats 实际 schema,footer 已核)
    N = [[int(cols[f"n_{d}"][b]) for d in DOMAINS] for b in range(nb)]

    if args.floor > 0:
        from training.dataline.multi_ar import nearest_allowed_bucket
        nd = [sum(N[b][j] for b in range(nb)) for j in range(len(DOMAINS))]     # N[d]
        pb = [sum(mix[d] * N[b][j] / max(nd[j], 1) for j, d in enumerate(DOMAINS))
              for b in range(nb)]
        kept = {b for b in range(nb) if pb[b] >= args.floor}
        kept.add(sq)                                            # square 恒保留(生产语义)
        folded = [[0] * len(DOMAINS) for _ in range(nb)]
        for b in range(nb):
            tgt = b if b in kept else nearest_allowed_bucket(b, buckets, frozenset(kept))
            for j in range(len(DOMAINS)):
                folded[tgt][j] += N[b][j]
        N = folded
        print(f"floor={args.floor}: kept={len(kept)}/{nb} 桶,其余折叠进最近保留桶")

    catalog_hash = assignment_hash = None
    if args.catalog_header:
        with open(args.catalog_header, encoding="utf-8") as f:
            hdr = json.load(f)
        if hdr.get("diagnostic_run") is not False:
            raise SystemExit("catalog header 是 diag 产物(diagnostic_run != False),不得用于生产 plan")
        if list(hdr["domain_order"]) != DOMAINS:
            raise SystemExit(f"catalog domain_order {hdr['domain_order']} != {DOMAINS}")
        for pos, b in enumerate(hdr["buckets"]):
            if int(b["idx"]) != pos:
                raise SystemExit(f"catalog buckets[{pos}].idx={b['idx']} 与位置不符")
        hb = [(int(b["h"]), int(b["w"])) for b in hdr["buckets"]]
        if hb != [tuple(b) for b in buckets]:
            raise SystemExit("catalog header 桶表与 config 重建桶表不一致")
        if int(hdr["square_bucket_idx"]) != sq:
            raise SystemExit(f"square idx 不一致:header={hdr['square_bucket_idx']} config={sq}")
        n_cat = [[int(x) for x in row] for row in hdr["N"]["materialized_after_exclusion"]]
        if len(n_cat) != nb:
            raise SystemExit(f"catalog N 桶数 {len(n_cat)} != {nb}")
        kept_idx = set(int(k) for k in hdr["kept_bucket_indices"])
        for b in range(nb):
            if b not in kept_idx and any(n_cat[b]):
                raise SystemExit(f"catalog N 在非保留桶 {b} 上非零,header 自相矛盾")
        # 逐 cell 交叉校验(审查升级):header 的 raw_before_floor 必须 == N_matrix 本地现算——
        # 抓"喂错版本 N_matrix"的任何漂移(含总和不变的对换);再用权威 fold_map 折叠本地 N,
        # 逐 cell 核 materialized+excluded == folded。
        raw_hdr = [[int(x) for x in row] for row in hdr["N"]["raw_before_floor"]]
        if raw_hdr != N:
            bad = [(b, j) for b in range(nb) for j in range(len(DOMAINS)) if raw_hdr[b][j] != N[b][j]]
            raise SystemExit(f"catalog raw_before_floor 与 N_matrix 现算 N 逐 cell 不一致,首异 {bad[:5]}")
        fold_map = [int(x) for x in hdr["fold_map"]]
        folded_local = [[0] * len(DOMAINS) for _ in range(nb)]
        for b in range(nb):
            for j in range(len(DOMAINS)):
                folded_local[fold_map[b]][j] += N[b][j]
        excl = [[int(x) for x in row] for row in hdr["N"]["tar_index_excluded"]]
        for b in range(nb):
            for j in range(len(DOMAINS)):
                if n_cat[b][j] + excl[b][j] != folded_local[b][j]:
                    raise SystemExit(
                        f"逐 cell 守恒失败 (bucket={b},domain={DOMAINS[j]}):"
                        f"materialized({n_cat[b][j]})+excluded({excl[b][j]})"
                        f" != folded({folded_local[b][j]})")
        excl_total = sum(sum(r) for r in excl)
        N = n_cat
        catalog_hash = str(hdr["catalog_hash"])
        assignment_hash = str(hdr["assignment_hash"])
        print(f"catalog-header 模式:N=materialized_after_exclusion(排除 {excl_total} 行),"
              f"逐 cell 守恒全过,绑定 catalog_hash={catalog_hash[:12]}…")

    out = {
        "format_version": 1,
        "plan_id": args.plan_id,
        "seed": args.seed,
        "world_size": args.world_size,
        "per_rank_bs": args.per_rank_bs,
        "total_steps": args.total_steps,
        "domains": DOMAINS,
        "domain_mix": mix,
        "square_only_domains": square_only,
        "buckets": [list(b) for b in buckets],
        "square_bucket_idx": sq,
        "N": N,
        "R_max": json.loads(args.r_max) if args.r_max else R_MAX_DEFAULT,
        "epsilon_jitter": False,
        "block_steps": args.block_steps,
    }
    if catalog_hash is not None:
        out["catalog_hash"] = catalog_hash          # planner→plan header upstream_catalog_hash
        out["assignment_hash"] = assignment_hash    # loader 据此拒 proxy plan
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    tot = sum(sum(r) for r in N)
    print(f"planner_input 写出: {args.out} buckets={nb} sq={sq} N_total={tot} mix={mix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
