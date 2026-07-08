#!/usr/bin/env python3
"""EXP6c — FLUX.2 VAE 对 anime 目标域的重建保真度复核（纯推理，不训练、不改任何核心代码）。

背景（交接文档 G7 开放问题）："FLUX.2 VAE 对 anime 域是否需 domain norm 复核
（LPIPS/rFID/线条/网点）"。FLUX.2 VAE 在自然照片上训练，anime 线稿/网点(screentone)/
平涂/文字 是分布外、高频结构密集，VAE round-trip 可能糊线、抹网点、毁文字 —— 若如此，
latent 空间训练再好也被 VAE 上限卡死。本脚本量化：

  对 N 张 anime 图（+ 若干 MONET 真实照片作对照），在 256/512/1024 分辨率：
    pixels --VAE.encode--> model-space latent --VAE.decode--> pixels'
  度量：
    - PSNR / MSE（整图）
    - 高频(Laplacian)误差比：line/screentone/文字 主要是高频 → 专门看高频重建
    - 各分辨率趋势（分辨率越高、相对下采样越细、anime 高频越该保住）
  并存 side-by-side 对比图（原图 | 重建 | 高频残差放大），供肉眼盯线条/网点/文字 crisp。

判读：anime PSNR 若与 real-photo 接近且 >~30dB、高频误差比不显著高于 real → 无需 domain norm；
      若 anime 明显更差（PSNR 低几 dB、高频误差比 real 高很多、肉眼糊线/抹网点）→ 需复核
      （domain-specific latent norm / 换 VAE / anime 专项微调 VAE）。

用法：
  venv-gpu/bin/python experiments/exp_scripts/vae_anime_fidelity.py \
      --n-anime 24 --n-real 12 --sizes 256,512,1024 --out /local/evan/mot_exp/exp6_vae
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from networks.vae import build_vae
from training.dataline.config import load_sampling_config
from training.dataline.image_fetch import (decode_rgb, fetch_url_bytes, monet_tar_path,
                                           read_gpic_bytes, resize_crop_uint8, stream_tar_images,
                                           uint8_to_norm_tensor)
from training.dataline.manifest_index import (list_shards, load_anime_shard,
                                              load_real_synth_shard, compute_real_global_stats)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG_PATH = "/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json"
ANIME_CACHE = "/local/evan/anime_img_cache"

# 高频算子（line/screentone/文字 检测）：3x3 Laplacian，逐通道
_LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def high_freq(x: torch.Tensor) -> torch.Tensor:
    """x[B,3,H,W] -> Laplacian 响应[B,3,H,W]（高频能量，line/网点/文字 主要落这）。"""
    w = _LAP.to(x.device, x.dtype).repeat(3, 1, 1, 1)
    return torch.nn.functional.conv2d(x, w, padding=1, groups=3)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """a,b in [0,1]; 返回 dB。"""
    mse = (a - b).pow(2).mean().item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def to01(px: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1]，clamp。"""
    return ((px + 1.0) * 0.5).clamp(0, 1)


def grab_anime_u8(n: int, size: int) -> list:
    """抓 n 张 anime 图（随机 row-group，elig_size 过滤），返回 uint8 [H,W,3] 列表。"""
    cfg = load_sampling_config(CFG_PATH)
    shards = list_shards(cfg)["anime"]
    rng = np.random.default_rng(1234)
    imgs, tried = [], 0
    import random as _r
    pyrng = _r.Random(1234)
    shard_order = list(shards)
    pyrng.shuffle(shard_order)
    stage = "256" if size <= 256 else ("512" if size <= 512 else "1024")
    for shard in shard_order:
        if len(imgs) >= n:
            break
        arr = load_anime_shard(shard, cfg, with_caption=False, nrows=4000, rng=pyrng)
        elig = arr[f"elig_{stage}"]
        idxs = np.where(elig)[0]
        rng.shuffle(idxs)
        for i in idxs:
            if len(imgs) >= n:
                break
            tried += 1
            raw = fetch_url_bytes(arr["s3key"][int(i)], cache_dir=ANIME_CACHE)
            if not raw:
                continue
            img = decode_rgb(raw)
            if img is None:
                continue
            imgs.append(resize_crop_uint8(img, size))
    print(f"  [anime] size={size}: got {len(imgs)}/{n} (tried {tried})", flush=True)
    return imgs


def grab_real_u8(n: int, size: int) -> list:
    """抓 n 张 MONET 真实照片作对照（tar 流式；best-effort，失败返回空）。"""
    try:
        cfg = load_sampling_config(CFG_PATH)
        all_shards = list_shards(cfg)
        stats = compute_real_global_stats(all_shards["monet_real"],
                                          cache_path="/local/evan/mot_stats/monet_global_stats.json")
        stage = "256" if size <= 256 else ("512" if size <= 512 else "1024")
        imgs = []
        for manifest_shard in all_shards["monet_real"]:
            if len(imgs) >= n:
                break
            arr = load_real_synth_shard(manifest_shard, "monet_real", cfg, stats)
            elig = arr[f"elig_{stage}"]
            idxs = np.where(elig)[0]
            if len(idxs) == 0:
                continue
            from collections import defaultdict
            by_shard = defaultdict(list)
            for i in idxs[:2000]:
                by_shard[arr["src_shard"][int(i)]].append(int(i))
            for src_shard, rows in by_shard.items():
                if len(imgs) >= n:
                    break
                src_root = arr["src_root"][rows[0]]
                wanted = {arr["src_key"][i] for i in rows[:n * 3]}
                got = dict(stream_tar_images(monet_tar_path(src_root, src_shard), wanted))
                for i in rows:
                    if len(imgs) >= n:
                        break
                    raw = got.get(arr["src_key"][i])
                    if not raw:
                        continue
                    img = decode_rgb(raw)
                    if img is None:
                        continue
                    imgs.append(resize_crop_uint8(img, size))
        print(f"  [real ] size={size}: got {len(imgs)}/{n}", flush=True)
        return imgs
    except Exception as e:  # noqa: BLE001
        print(f"  [real ] size={size}: skipped ({type(e).__name__}: {e})", flush=True)
        return []


def save_grid(orig01, recon01, path, max_rows=8):
    """存 side-by-side：每行 [原图 | 重建 | 高频残差(放大)]。"""
    try:
        from PIL import Image
    except ImportError:
        print("  (PIL 缺失，跳过存图)", flush=True)
        return
    n = min(orig01.shape[0], max_rows)
    resid = (recon01 - orig01).abs().clamp(0, 1)
    resid = (resid / (resid.amax() + 1e-6)).clamp(0, 1)  # 放大残差便于肉眼看
    tiles = []
    for i in range(n):
        row = torch.cat([orig01[i], recon01[i], resid[i]], dim=2)  # 横向拼 3 张
        tiles.append(row)
    canvas = torch.cat(tiles, dim=1)  # 纵向堆 n 行
    arr = (canvas.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    print(f"  saved grid -> {path}", flush=True)


def run_domain(name, u8_list, vae, device, size, out_dir):
    if not u8_list:
        return None
    px = torch.stack([uint8_to_norm_tensor(u) for u in u8_list], 0).to(device)  # [B,3,H,W] [-1,1]
    with torch.no_grad():
        lat = vae.encode(px)
        recon = vae.decode(lat).float()
    o01, r01 = to01(px.float()), to01(recon)
    # 整图
    full_psnr = psnr(o01, r01)
    full_mse = (o01 - r01).pow(2).mean().item()
    # 高频（line/网点/文字）
    hf_o, hf_r = high_freq(o01), high_freq(r01)
    hf_mse = (hf_o - hf_r).pow(2).mean().item()
    hf_energy = hf_o.pow(2).mean().item()
    hf_rel = hf_mse / (hf_energy + 1e-9)  # 高频误差 / 高频能量（越小越保真）
    # per-image PSNR 分布
    per = [psnr(o01[i:i + 1], r01[i:i + 1]) for i in range(o01.shape[0])]
    save_grid(o01, r01, os.path.join(out_dir, f"vae_{name}_{size}.png"))
    return {"domain": name, "size": size, "n": int(o01.shape[0]),
            "psnr_db": round(full_psnr, 3), "mse": round(full_mse, 6),
            "hf_mse": round(hf_mse, 6), "hf_energy": round(hf_energy, 6),
            "hf_rel_err": round(hf_rel, 4),
            "psnr_min": round(min(per), 2), "psnr_p25": round(float(np.percentile(per, 25)), 2),
            "psnr_median": round(float(np.median(per)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-anime", type=int, default=24)
    ap.add_argument("--n-real", type=int, default=12)
    ap.add_argument("--sizes", default="256,512,1024")
    ap.add_argument("--out", default="/local/evan/mot_exp/exp6_vae")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizes = [int(s) for s in args.sizes.split(",")]

    print(f"[exp6c] building FLUX.2 VAE on {device} (dtype={args.dtype})", flush=True)
    t0 = time.perf_counter()
    vae = build_vae({"name": "flux2", "pretrained_path": "/fsx/models/FLUX.2-klein-4B/vae",
                     "dtype": args.dtype, "sample_posterior": False}).to(device)  # mode() 确定性
    print(f"  VAE ready in {time.perf_counter()-t0:.1f}s", flush=True)

    results = []
    for size in sizes:
        print(f"\n=== size {size} ===", flush=True)
        anime = grab_anime_u8(args.n_anime, size)
        real = grab_real_u8(args.n_real, size)
        for name, u8 in [("anime", anime), ("real", real)]:
            r = run_domain(name, u8, vae, device, size, args.out)
            if r:
                results.append(r)
                print(f"  [{name} {size}] PSNR={r['psnr_db']}dB (median {r['psnr_median']}, "
                      f"min {r['psnr_min']})  hf_rel_err={r['hf_rel_err']}", flush=True)

    # 汇总对照
    print("\n========== EXP6c VAE FIDELITY SUMMARY ==========", flush=True)
    print(f"{'domain':8} {'size':6} {'n':4} {'PSNR(dB)':9} {'median':7} {'min':6} {'hf_rel_err':10}", flush=True)
    for r in results:
        print(f"{r['domain']:8} {r['size']:<6} {r['n']:<4} {r['psnr_db']:<9} {r['psnr_median']:<7} "
              f"{r['psnr_min']:<6} {r['hf_rel_err']:<10}", flush=True)
    # anime vs real delta per size
    by = {(r["domain"], r["size"]): r for r in results}
    print("\n-- anime vs real (同尺寸) --", flush=True)
    for size in sizes:
        a, rr = by.get(("anime", size)), by.get(("real", size))
        if a and rr:
            print(f"  size {size}: ΔPSNR(real-anime)={rr['psnr_db']-a['psnr_db']:.2f}dB  "
                  f"hf_rel_err anime/real={a['hf_rel_err']/(rr['hf_rel_err']+1e-9):.2f}×", flush=True)
    out_json = os.path.join(args.out, "vae_fidelity_results.json")
    json.dump(results, open(out_json, "w"), ensure_ascii=False, indent=1)
    print(f"\nVAE_FIDELITY_JSON {json.dumps(results, ensure_ascii=False)}", flush=True)
    print(f"results -> {out_json}; grids -> {args.out}/vae_*.png", flush=True)


if __name__ == "__main__":
    main()
