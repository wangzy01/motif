"""QC for one stage of the FLUX.2 VAE moments cache.

汇总 stage={S} 下的 index/failed/run_stats，产出 summary.json + qc.json + 合并
failed_keys.parquet，并做随机读回校验。可独立于缓存 job 运行（单进程，读 /s3）。

报：总 eligible / ok / fail / 失败率 / 每域 ok·fail / latent_shape 分布 /
safetensors 文件数·总大小 / index 行数==tensor 数 / 随机 100 key 读回 /
吞吐(img/s/GPU·总·总耗时) / 路径·空间。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.torch import safe_open

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.dataline.config import load_sampling_config  # noqa: E402
from scripts.vae_cache_flux2_moments import (  # noqa: E402
    list_manifest_shards, participating_domains, _ANIME_COLS, _copy, read_parquet_file,
)


def count_eligible(cfg, stage) -> dict:
    """扫 manifest 数 eligible（elig_stage==1, holdout==0），每域。"""
    shards = list_manifest_shards(cfg)
    out = {}
    for d in participating_domains(cfg, stage):
        n = 0
        for sp in shards[d]:
            if d == "anime":
                t = pq.read_table(sp, columns=[f"elig_{stage}"])
                n += int(np.sum(np.array(t.column(f"elig_{stage}").to_pylist(), dtype=np.int8) == 1))
            else:
                t = pq.read_table(sp, columns=[f"elig_{stage}", "holdout"])
                e = np.array(t.column(f"elig_{stage}").to_pylist(), dtype=np.int8) == 1
                h = np.array(t.column("holdout").to_pylist(), dtype=np.int8) == 0
                n += int(np.sum(e & h))
        out[d] = n
    out["_total"] = sum(out.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out-root", default="/s3/evan/mot_latents/complete_v1_flux2_moments")
    ap.add_argument("--readback", type=int, default=100)
    ap.add_argument("--scratch", default="/local/evan/vae_cache_scratch")
    ap.add_argument("--skip-eligible", action="store_true", help="跳过全量 eligible 扫描（快速 QC）")
    ap.add_argument("--out-suffix", default="", help="输出文件名加后缀（如 _postpatch），防覆盖已存在的 summary/qc/failed_keys（mountpoint-s3 不能覆盖）")
    args = ap.parse_args()

    stage = args.stage
    stage_dir = os.path.join(args.out_root, f"stage={stage}")
    latents_dir = os.path.join(stage_dir, "latents")
    index_dir = os.path.join(stage_dir, "index")
    failed_dir = os.path.join(stage_dir, "failed")
    rs_dir = os.path.join(stage_dir, "run_stats")
    cfg = load_sampling_config(args.config)
    S = int(stage)
    t0 = time.perf_counter()

    # ---- index 汇总 ----
    idx_files = sorted(glob.glob(os.path.join(index_dir, "*.parquet")))
    print(f"index parts: {len(idx_files)}", flush=True)
    dom_ok = Counter(); shape_dist = Counter(); idx_rows = 0
    uid_to_path = {}; uid_list = []
    for p in idx_files:
        t = read_parquet_file(p, columns=["uid", "domain", "latent_shape", "path", "status"])
        idx_rows += t.num_rows
        for d in t.column("domain").to_pylist():
            dom_ok[d] += 1
        for s in t.column("latent_shape").to_pylist():
            shape_dist[s] += 1
        for u, pa_ in zip(t.column("uid").to_pylist(), t.column("path").to_pylist()):
            uid_to_path[u] = pa_
            uid_list.append(u)

    # ---- safetensors 文件数 + 大小 + tensor 数 ----
    st_files = sorted(glob.glob(os.path.join(latents_dir, "*.safetensors")))
    total_bytes = 0; tensor_count = 0
    for f in st_files:
        total_bytes += os.path.getsize(f)
    # tensor 数：抽样 or 全量？全量开销大；统计每文件 key 数（safe_open 只读 header，快）
    for f in st_files:
        try:
            with safe_open(f, framework="pt", device="cpu") as h:
                tensor_count += len(h.keys())
        except Exception as e:  # noqa: BLE001
            print(f"  WARN bad safetensors {f}: {e}", flush=True)

    # ---- failed 汇总 + 合并 failed_keys.parquet ----
    fail_files = sorted(glob.glob(os.path.join(failed_dir, "*.parquet")))
    # 排除可能已存在的合并产物 failed_keys.parquet（在 stage 根，不在 failed/ 下，glob 不会撞，这里防御）
    fail_files = [p for p in fail_files if os.path.basename(p) != "failed_keys.parquet"]
    dom_fail = Counter(); fail_total = 0; fail_tables = []
    for p in fail_files:
        t = read_parquet_file(p)
        fail_total += t.num_rows
        for d in t.column("domain").to_pylist():
            dom_fail[d] += 1
        fail_tables.append(t)
    if fail_tables:
        merged = pa.concat_tables(fail_tables)
        local_merged = os.path.join(args.scratch, f"failed_keys_{stage}.parquet")
        os.makedirs(args.scratch, exist_ok=True)
        pq.write_table(merged, local_merged, compression="zstd")
        _copy(local_merged, os.path.join(stage_dir, f"failed_keys{args.out_suffix}.parquet"))

    # ---- 随机读回 ----
    rb = {"requested": args.readback, "checked": 0, "ok": 0, "bad": []}
    if uid_list:
        rng = random.Random(0)
        sample = rng.sample(uid_list, min(args.readback, len(uid_list)))
        expect = (64, S // 8, S // 8)
        # group by file 减少 open 次数
        by_file = defaultdict(list)
        for u in sample:
            by_file[uid_to_path[u]].append(u)
        for rel, uids in by_file.items():
            fpath = os.path.join(stage_dir, rel)
            try:
                with safe_open(fpath, framework="pt", device="cpu") as h:
                    for u in uids:
                        rb["checked"] += 1
                        t = h.get_tensor(u)
                        if tuple(t.shape) == expect and str(t.dtype).endswith("bfloat16"):
                            rb["ok"] += 1
                        else:
                            rb["bad"].append({"uid": u, "shape": list(t.shape), "dtype": str(t.dtype)})
            except Exception as e:  # noqa: BLE001
                rb["bad"].append({"file": rel, "error": str(e)})

    # ---- 吞吐（run_stats）----
    rs_files = sorted(glob.glob(os.path.join(rs_dir, "*.json")))
    per_rank = []
    for p in rs_files:
        with open(p) as f:
            per_rank.append(json.load(f))
    total_ok_rank = sum(r["ok"] for r in per_rank) if per_rank else 0
    max_sec = max((r["sec"] for r in per_rank), default=0)
    sum_img_s = sum(r["img_per_s"] for r in per_rank) if per_rank else 0
    n_ranks = len(per_rank)

    # ---- eligible ----
    if args.skip_eligible:
        elig = {"_skipped": True}
    else:
        print("counting eligible (scan manifests)...", flush=True)
        elig = count_eligible(cfg, stage)

    elig_total = elig.get("_total") if not args.skip_eligible else None
    fail_rate = fail_total / (idx_rows + fail_total) if (idx_rows + fail_total) else 0.0

    summary = {
        "stage": stage, "out_root": args.out_root, "stage_dir": stage_dir,
        "eligible": elig,
        "ok_tensors": idx_rows,
        "failed": fail_total,
        "fail_rate": round(fail_rate, 5),
        "coverage_vs_eligible": (round(idx_rows / elig_total, 5) if elig_total else None),
        "per_domain_ok": dict(dom_ok),
        "per_domain_fail": dict(dom_fail),
        "latent_shape_dist": dict(shape_dist),
        "safetensors_files": len(st_files),
        "safetensors_total_bytes": total_bytes,
        "safetensors_total_GiB": round(total_bytes / (1 << 30), 2),
        "safetensors_total_TiB": round(total_bytes / (1 << 40), 3),
        "index_rows": idx_rows,
        "safetensors_tensor_count": tensor_count,
        "index_eq_tensors": (idx_rows == tensor_count),
        "readback": rb,
        "throughput": {
            "n_ranks": n_ranks,
            "wall_sec_max_rank": max_sec,
            "wall_min": round(max_sec / 60, 1) if max_sec else None,
            "img_per_s_per_rank_sum": round(sum_img_s, 1),
            "img_per_s_total_est": (round(total_ok_rank / max_sec, 1) if max_sec else None),
            "per_rank": per_rank,
        },
        "qc_sec": round(time.perf_counter() - t0, 1),
    }

    os.makedirs(args.scratch, exist_ok=True)
    sfx = args.out_suffix
    for name, obj in ((f"summary{sfx}.json", summary), (f"qc{sfx}.json", summary)):
        lp = os.path.join(args.scratch, f"{stage}_{name}")
        with open(lp, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        _copy(lp, os.path.join(stage_dir, name))

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
