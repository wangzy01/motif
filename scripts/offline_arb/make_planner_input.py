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
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from training.dataline.config import load_sampling_config          # noqa: E402
from training.dataline.multi_ar import build_adaptive_buckets, square_index
from training.dataline import anchors_v1  # noqa: E402

DOMAINS = ["anime", "monet_real", "gpic", "synthetic", "textatlas"]   # 与 N_matrix/planner 列序契约一致
R_MAX_DEFAULT = {"anime": 20, "monet_real": 20, "gpic": 20, "synthetic": 200, "textatlas": 200}


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _stable_sampling_sha256(
    path: Path,
    *,
    expected_fingerprint: tuple[int, int, int, int, int, int] | None = None,
) -> tuple[str, tuple[int, int, int, int, int, int]]:
    """Hash one non-symlink sampling config through a closed file identity."""

    try:
        path_before = path.lstat()
    except OSError as exc:
        raise SystemExit(f"cannot stat sampling config {path}: {exc}") from exc
    before = _fingerprint(path_before)
    if not stat.S_ISREG(path_before.st_mode) or path_before.st_size <= 0:
        raise SystemExit(f"sampling config is not a nonempty regular file: {path}")
    if expected_fingerprint is not None and before != expected_fingerprint:
        raise SystemExit("sampling config fingerprint changed across planner load")

    digest = hashlib.sha256()
    try:
        with path.open("rb", buffering=0) as handle:
            fd_before = _fingerprint(os.fstat(handle.fileno()))
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
            fd_after = _fingerprint(os.fstat(handle.fileno()))
        path_after = _fingerprint(path.lstat())
    except OSError as exc:
        raise SystemExit(f"cannot hash sampling config {path}: {exc}") from exc
    if not (before == fd_before == fd_after == path_after):
        raise SystemExit("sampling config changed while being hashed")
    return digest.hexdigest(), before


def _validate_stage512a_catalog_sampling(
    header: dict, mix: dict[str, float], sampling_sha256: str
) -> None:
    """Close planner-v2 sampling semantics against the catalog authority."""

    inputs = header.get("inputs")
    if not isinstance(inputs, dict):
        raise SystemExit("catalog header inputs must be an object")
    if inputs.get("sampling_config_sha256") != sampling_sha256:
        raise SystemExit("sampling config SHA-256 differs from catalog header")
    catalog_mix = inputs.get("domain_mix")
    if (
        not isinstance(catalog_mix, dict)
        or set(catalog_mix) != set(DOMAINS)
        or any(type(catalog_mix[domain]) is not float for domain in DOMAINS)
        or catalog_mix != mix
    ):
        raise SystemExit(
            "sampling domain_mix differs from catalog header: "
            f"catalog={catalog_mix!r}, live={mix!r}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-matrix", required=True)
    ap.add_argument("--config", default="configs/stage256_redesign_naifu_arb_v1.yaml")
    ap.add_argument("--stage", default="256")
    ap.add_argument(
        "--format-version",
        type=int,
        choices=(1, 2),
        default=1,
        help="planner contract version; stage512a production must explicitly use 2",
    )
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
    config_stage = str(cfg.dataset.stage)
    if config_stage != args.stage:
        raise SystemExit(
            f"--stage {args.stage!r} != config dataset.stage {config_stage!r}"
        )
    mar = cfg.dataset.multi_ar
    bucket_mode = str(OmegaConf.select(cfg, "dataset.multi_ar.bucket_mode") or "adaptive")
    if bucket_mode == anchors_v1.BUCKET_MODE:
        # anchor_id order IS the bucket index and is the identity key the Stage512b
        # audits are written against; it must not be re-sorted by aspect ratio.
        buckets = anchors_v1.anchors_hw()
    else:
        buckets = build_adaptive_buckets(
            int(mar.target_area), int(mar.divisor),
            min_area_ratio=float(mar.adaptive_min_area_ratio),
            max_aspect=float(mar.adaptive_max_aspect))   # no_upscale 是指派期参数,桶表构建不收
    sq = square_index(buckets)
    square_only = [str(d) for d in mar.square_only_domains]
    sampling_path = Path(str(cfg.dataset.sampling_config))
    close_catalog_sampling = (
        args.format_version == 2
        and args.stage in ("512a", "512b")
        and args.catalog_header is not None
    )
    sampling_sha256 = None
    sampling_fingerprint = None
    if close_catalog_sampling:
        if not sampling_path.is_absolute():
            raise SystemExit("stage512a planner-v2 sampling config must be absolute")
        sampling_sha256, sampling_fingerprint = _stable_sampling_sha256(
            sampling_path
        )
    samp = load_sampling_config(str(sampling_path))
    if close_catalog_sampling:
        closing_sha256, closing_fingerprint = _stable_sampling_sha256(
            sampling_path, expected_fingerprint=sampling_fingerprint
        )
        if (
            closing_sha256 != sampling_sha256
            or closing_fingerprint != sampling_fingerprint
        ):
            raise SystemExit("sampling config changed across planner load")
    mix = {d: float(samp.domain_mix[args.stage][d]) for d in DOMAINS}
    image_size = int(cfg.dataset.image_size)
    eligibility = f"elig_{image_size}"
    requires_nl = bool(samp.anime_caption.requires_nl(args.stage))
    # Phase3/5 materializes the historical pool without applying the sampling
    # config's require-NL filter.  This is explicit in the artifact identity.
    anime_nl_filter_applied = False
    if args.format_version == 2 and not args.catalog_header:
        raise SystemExit("format_version=2 requires --catalog-header; virtual/proxy hashes are forbidden")

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
        if bucket_mode == anchors_v1.BUCKET_MODE:
            # Same refusal phase5 has. nearest_allowed_bucket is an aspect-ratio nearest
            # neighbour with NO feasibility test, so a pruned extent folds onto a TALLER
            # anchor: 704x352 -> 672x384 enlarges 1.091x, 352x704 -> 384x672 likewise,
            # 544x480 -> 560x448 1.029x. Without this a positive --floor silently hands the
            # planner an N matrix whose rows the source images cannot cover, and nothing
            # downstream re-checks it -- verify_binding() would reject them at train time.
            raise SystemExit(
                f"bucket_mode={anchors_v1.BUCKET_MODE} requires an identity fold; "
                f"--floor={args.floor} would prune anchors and the fold has no "
                "feasibility test, which would re-introduce upsampling. Use --floor 0."
            )
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
        if args.format_version == 2:
            expected_identity = {
                "stage": args.stage,
                "image_size": image_size,
                "eligibility": eligibility,
                "requires_nl": requires_nl,
                "anime_nl_filter_applied": anime_nl_filter_applied,
            }
            actual_identity = {key: hdr.get(key) for key in expected_identity}
            if actual_identity != expected_identity:
                raise SystemExit(
                    "catalog stage identity differs from config/sampling: "
                    f"actual={actual_identity!r}, expected={expected_identity!r}"
                )
            if close_catalog_sampling:
                if sampling_sha256 is None:
                    raise SystemExit("stage512a sampling evidence was not collected")
                _validate_stage512a_catalog_sampling(hdr, mix, sampling_sha256)
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
        "format_version": args.format_version,
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
    if args.format_version == 2:
        out.update(
            {
                "stage": args.stage,
                "image_size": image_size,
                "eligibility": eligibility,
                "requires_nl": requires_nl,
                "anime_nl_filter_applied": anime_nl_filter_applied,
            }
        )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    tot = sum(sum(r) for r in N)
    print(f"planner_input 写出: {args.out} buckets={nb} sq={sq} N_total={tot} mix={mix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
