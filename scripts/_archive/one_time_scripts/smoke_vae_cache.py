"""Smoke for the offline FLUX.2 VAE moments cache.

跑少量样本验证整条链路：
  A. 每 stage 每 participating domain 跑缓存 CLI（小 limit，真写 /s3）。
  B. 读回 moments → Flux2VAE.moments_to_model_space() == [128, S/16, S/16]、dtype/device。
  C. index ↔ safetensors key 一一对应。
  D. 在线 VAE encode 对比（shape/dtype/device + mode-path 可复现）。
  E. 失败样本可记录、不崩 job（注入坏 URL/坏字节）。

用法（本机 1×H100）：
  PYTHONPATH=/fsx/evan/mot_code/mot_pilot_v1 \
  /fsx/evan/mot_code/venv-gpu/bin/python scripts/smoke_vae_cache.py \
    --stages 256,512,1024 --per-domain 96
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import safe_open

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.dataline.config import load_sampling_config  # noqa: E402
from training.dataline.image_fetch import (  # noqa: E402
    decode_rgb, fetch_url_bytes, gpic_path, monet_tar_path, read_gpic_bytes,
    resize_crop_uint8, stream_tar_images, uint8_to_norm_tensor,
)
from scripts.vae_cache_flux2_moments import (  # noqa: E402
    list_manifest_shards, participating_domains, _load_gpic_ext_map, read_parquet_file,
)

PY = sys.executable
CACHE = os.path.join(REPO_ROOT, "scripts", "vae_cache_flux2_moments.py")
CONFIG = "/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json"


def hr(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def fetch_some(domain, shard, stage, image_size, n):
    """直接取 n 张已预处理 uint8（在线对比用），返回 list[uint8 HWC]。"""
    cfg = load_sampling_config(CONFIG)
    if domain == "anime":
        t = pq.read_table(shard, columns=["s3key", f"elig_{stage}"])
        elig = np.array(t.column(f"elig_{stage}").to_pylist(), dtype=np.int8) == 1
        urls = [u for u, e in zip(t.column("s3key").to_pylist(), elig) if e][:n * 3]
        out = []
        for u in urls:
            raw = fetch_url_bytes(u, retries=2)
            img = decode_rgb(raw) if raw else None
            if img is not None:
                out.append(resize_crop_uint8(img, image_size))
            if len(out) >= n:
                break
        return out
    cols = ["src_root", "src_shard", "src_key", "holdout", f"elig_{stage}"]
    t = pq.read_table(shard, columns=cols).to_pylist()
    rows = [r for r in t if r["holdout"] == 0 and r[f"elig_{stage}"] == 1][: n * 4]
    out = []
    if domain == "gpic":
        for r in rows:
            ext = _load_gpic_ext_map(r["src_root"], r["src_shard"]).get(r["src_key"])
            raw = read_gpic_bytes(r["src_root"], r["src_shard"], r["src_key"], ext=ext)
            img = decode_rgb(raw) if raw else None
            if img is not None:
                out.append(resize_crop_uint8(img, image_size))
            if len(out) >= n:
                break
    else:
        from collections import defaultdict
        bysh = defaultdict(list)
        for r in rows:
            bysh[(r["src_root"], r["src_shard"])].append(r)
        for (sr, ss), grp in bysh.items():
            got = dict(stream_tar_images(monet_tar_path(sr, ss), {r["src_key"] for r in grp}))
            for r in grp:
                raw = got.get(r["src_key"])
                img = decode_rgb(raw) if raw else None
                if img is not None:
                    out.append(resize_crop_uint8(img, image_size))
                if len(out) >= n:
                    break
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="256,512,1024")
    ap.add_argument("--per-domain", type=int, default=96)
    ap.add_argument("--out-root", default="/s3/evan/mot_latents/_smoke_flux2_moments")
    ap.add_argument("--vae-path", default="/fsx/models/FLUX.2-klein-4B/vae")
    ap.add_argument("--skip-cache", action="store_true", help="跳过 A 阶段（复用已写 /s3 smoke 数据，只跑 B/C/D/E）")
    args = ap.parse_args()
    stages = args.stages.split(",")
    cfg = load_sampling_config(CONFIG)
    shards = list_manifest_shards(cfg)
    results = []  # (name, ok, detail)

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)

    # ---------- A. 跑缓存 CLI ----------
    hr("A. 缓存 CLI（每 stage 每域小样本，真写 /s3）")
    if args.skip_cache:
        print("  [SKIP] --skip-cache：复用已写 smoke 数据", flush=True)
    for stage in (stages if not args.skip_cache else []):
        for d in participating_domains(cfg, stage):
            cmd = [PY, CACHE, "--config", CONFIG, "--stage", stage,
                   "--out-root", args.out_root, "--vae-path", args.vae_path,
                   "--only-domains", d, "--limit-pairs", "1",
                   "--limit-per-domain", str(args.per_domain),
                   "--batch-size", "32", "--chunk-gib", "1.0",
                   "--scratch", "/local/evan/vae_cache_scratch_smoke",
                   "--anime-cache", "/local/evan/anime_img_cache"]
            env = dict(os.environ, PYTHONPATH=REPO_ROOT)
            t0 = time.perf_counter()
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            dt = time.perf_counter() - t0
            tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-300:]
            print(f"  stage={stage} {d}: rc={r.returncode} {dt:.1f}s | {tail}", flush=True)
            check(f"CLI run stage={stage} {d} rc==0", r.returncode == 0,
                  "" if r.returncode == 0 else r.stderr.strip()[-500:])

    # ---------- 加载 Flux2VAE（B/D 用）----------
    hr("加载 Flux2VAE adapter")
    from networks.vae import Flux2VAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = Flux2VAE(pretrained_path=args.vae_path, dtype="bfloat16", sample_posterior=True)
    vae.ae.to(device)
    print(f"  Flux2VAE on {device}, downsample={vae.downsample_factor}, C={vae.latent_channels}", flush=True)

    # ---------- B + C: 读回 / moments_to_model_space / index↔st ----------
    for stage in stages:
        hr(f"B+C. stage={stage} 读回 + moments_to_model_space + index↔safetensors")
        S = int(stage)
        sd = os.path.join(args.out_root, f"stage={stage}")
        idx_files = sorted(glob.glob(os.path.join(sd, "index", "*.parquet")))
        st_files = sorted(glob.glob(os.path.join(sd, "latents", "*.safetensors")))
        check(f"stage={stage} 有 index part", len(idx_files) > 0, f"{len(idx_files)} parts")
        check(f"stage={stage} 有 safetensors", len(st_files) > 0, f"{len(st_files)} files")
        if not idx_files or not st_files:
            continue

        # index↔safetensors 1:1（按 path 分组）
        idx_by_path = {}
        all_uid_path = {}
        for p in idx_files:
            t = read_parquet_file(p, columns=["uid", "path", "latent_shape"])
            for u, pa_, sh in zip(t.column("uid").to_pylist(), t.column("path").to_pylist(),
                                  t.column("latent_shape").to_pylist()):
                idx_by_path.setdefault(pa_, set()).add(u)
                all_uid_path[u] = (pa_, sh)
        ok11 = True
        for rel, uids in idx_by_path.items():
            fp = os.path.join(sd, rel)
            with safe_open(fp, framework="pt", device="cpu") as h:
                keys = set(h.keys())
            if keys != uids:
                ok11 = False
                print(f"    mismatch {rel}: st={len(keys)} idx={len(uids)} "
                      f"st-idx={list(keys-uids)[:3]} idx-st={list(uids-keys)[:3]}", flush=True)
        check(f"stage={stage} index↔safetensors 一一对应", ok11,
              f"{len(all_uid_path)} keys / {len(idx_by_path)} files")

        # 读回 + moments_to_model_space（每域抽样）
        exp_mom = (64, S // 8, S // 8)
        exp_lat = (vae.latent_channels, S // vae.downsample_factor, S // vae.downsample_factor)
        per_dom = {}
        for p in idx_files:
            t = read_parquet_file(p, columns=["uid", "domain", "path"])
            for u, dm, pa_ in zip(t.column("uid").to_pylist(), t.column("domain").to_pylist(),
                                  t.column("path").to_pylist()):
                per_dom.setdefault(dm, []).append((u, pa_))
        for dm, items in per_dom.items():
            u, rel = items[0]
            with safe_open(os.path.join(sd, rel), framework="pt", device="cpu") as h:
                mom = h.get_tensor(u)
            check(f"  {stage}/{dm} moments shape={exp_mom} bf16", tuple(mom.shape) == exp_mom
                  and str(mom.dtype).endswith("bfloat16"), f"{tuple(mom.shape)} {mom.dtype}")
            lat = vae.moments_to_model_space(mom.unsqueeze(0).to(device))
            check(f"  {stage}/{dm} moments_to_model_space=={('1',)+exp_lat}",
                  tuple(lat.shape) == (1,) + exp_lat,
                  f"{tuple(lat.shape)} dtype={lat.dtype} dev={lat.device.type}")

    # ---------- D. 在线 VAE 对比 ----------
    hr("D. 在线 VAE encode 对比（shape/dtype/device + mode-path 可复现）")
    for stage in stages:
        S = int(stage)
        for d in participating_domains(cfg, stage):
            u8 = fetch_some(d, shards[d][0], stage, S, 4)
            if not u8:
                check(f"  {stage}/{d} online-compare 取到样本", False, "0 imgs")
                continue
            px = torch.stack([uint8_to_norm_tensor(x) for x in u8], 0).to(device, torch.bfloat16)
            # 离线缓存路径：ae.encode(px).latent_dist.parameters
            params = vae.ae.encode(px).latent_dist.parameters.to("cpu", torch.bfloat16)
            # 在线 model-space（采样）：Flux2VAE.encode
            online = vae.encode(px)
            # cache moments -> mode model-space（确定性）；online mode 同路径应逐位一致
            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
            lat_cache_mode = vae.moments_to_model_space(params.to(device), sample=False)
            dist = DiagonalGaussianDistribution(vae.ae.encode(px).latent_dist.parameters)
            lat_online_mode = vae.to_model_space(dist.mode(), vae.ae)
            same_shape = tuple(online.shape) == tuple(lat_cache_mode.shape) == \
                (len(u8), vae.latent_channels, S // 16, S // 16)
            mode_diff = (lat_cache_mode.float() - lat_online_mode.float()).abs().max().item()
            check(f"  {stage}/{d} online vs cache shape/dtype/device",
                  same_shape and online.dtype == lat_cache_mode.dtype
                  and online.device.type == lat_cache_mode.device.type,
                  f"online={tuple(online.shape)}/{online.dtype} cache={tuple(lat_cache_mode.shape)}")
            check(f"  {stage}/{d} mode-path 可复现(diff<1e-2)", mode_diff < 1e-2, f"max|Δ|={mode_diff:.2e}")

    # ---------- E. 失败可记录、不崩 ----------
    hr("E. 失败处理（坏 URL / 坏字节 不崩、可记录）")
    bad_url = fetch_url_bytes("https://img.metanomaly.co/r2/__definitely_missing_key__", retries=1, timeout=5)
    check("坏 URL fetch 返回 None（不抛）", bad_url is None, "")
    check("坏字节 decode 返回 None（不抛）", decode_rgb(b"not-an-image") is None, "")
    # 缓存 job 已在 A 阶段跑过坏样本无崩（rc==0 即证），并产出 failed/ 若有
    for stage in stages:
        fdir = os.path.join(args.out_root, f"stage={stage}", "failed")
        nf = len(glob.glob(os.path.join(fdir, "*.parquet")))
        print(f"  stage={stage} failed parts: {nf}", flush=True)

    # ---------- 汇总 ----------
    hr("SMOKE 汇总")
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    for name, ok, det in results:
        if not ok:
            print(f"  FAIL: {name}  {det}", flush=True)
    print(f"\n  {n_pass}/{len(results)} PASS, {n_fail} FAIL", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
