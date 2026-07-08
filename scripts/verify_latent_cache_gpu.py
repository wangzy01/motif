"""GPU 数值对拍：缓存 moments vs 在线 VAE 重编码（同设备 GPU + 同 dtype bf16）。

缓存存的是 ae.encode(px).latent_dist.parameters（GPU/bf16）。在 GPU/bf16 上对同一张图同款预处理
重编码，mode-path model-space 应与缓存**逐位/近逐位一致**（确证缓存忠实复现在线 VAE 几何）。
这是"缓存能用且 100% 对"的核心数值证明（CPU 因 fp32/设备差异只能近似，故放 GPU）。

跑(JCC 1 GPU): PYTHONPATH=. python scripts/verify_latent_cache_gpu.py --n 16
"""
import argparse
import glob
import os
import sys

import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.dataline.latent_cache import LatentCacheIndex
from training.dataline.config import load_sampling_config
from training.dataline.image_fetch import fetch_url_bytes, decode_rgb, resize_crop_uint8, uint8_to_norm_tensor
from networks.vae import Flux2VAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/s3/evan/mot_latents/complete_v1_flux2_moments")
    ap.add_argument("--stage", default="256")
    ap.add_argument("--vae-path", default="/fsx/models/FLUX.2-klein-4B/vae")
    ap.add_argument("--config", default="/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = int(args.stage)
    stage_dir = f"{args.root}/stage={args.stage}"

    idx = LatentCacheIndex(stage_dir)
    print(f"index size: {len(idx)}", flush=True)
    cfg = load_sampling_config(args.config)
    man = sorted(glob.glob(os.path.join(cfg.manifest_anime, "*.parquet")))[0]
    t = pq.ParquetFile(man).read(columns=["uid", "s3key"])
    u2s = dict(zip(t.column("uid").to_pylist(), t.column("s3key").to_pylist()))
    test_uids = [u for u in idx.uid2path if u.startswith("anime:") and u in u2s][: args.n]
    print(f"test anime uids: {len(test_uids)}", flush=True)

    vae = Flux2VAE(pretrained_path=args.vae_path, dtype="bfloat16")
    vae.ae.to(dev)

    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    cached = idx.get_many(test_uids)
    max_abs, rel_list, n = 0.0, [], 0
    for u in test_uids:
        if u not in cached:
            continue
        raw = fetch_url_bytes(u2s[u], retries=5, timeout=30)
        img = decode_rgb(raw)
        if img is None:
            continue
        u8 = resize_crop_uint8(img, S)
        px = uint8_to_norm_tensor(u8).unsqueeze(0).to(dev, dtype=torch.bfloat16)
        with torch.no_grad():
            online_params = vae.ae.encode(px).latent_dist.parameters            # [1,64,S/8,S/8] bf16 GPU
            lat_online = vae.to_model_space(DiagonalGaussianDistribution(online_params).mode(), vae.ae)
            cm = cached[u].unsqueeze(0).to(dev, dtype=torch.bfloat16)
            lat_cached = vae.to_model_space(DiagonalGaussianDistribution(cm).mode(), vae.ae)
        d = (lat_online.float() - lat_cached.float()).abs()
        max_abs = max(max_abs, float(d.max()))
        rel_list.append(float(d.mean() / (lat_cached.float().abs().mean() + 1e-6)))
        n += 1
    rel = sum(rel_list) / max(1, len(rel_list))
    print(f"\n对拍 {n} 张 anime（GPU bf16，mode-path model-space）:", flush=True)
    print(f"  max|Δ| = {max_abs:.3e}", flush=True)
    print(f"  mean rel err = {rel:.3e}", flush=True)
    # 判据：缓存忠实复现 VAE 几何 → 相对误差应是 bf16+cuDNN 跨运行噪声量级（~亚%），
    # 远小于 1（错图/错 VAE 会是 O(1)=几十%）。注意缓存是另一次 GPU run 编的、bf16，且 cuDNN
    # conv 非确定性 → 不可能逐位相同（原 smoke 的 max|Δ|=0 是同 session 同 kernel）；用 bf16 容差判。
    ok = (n >= max(1, args.n // 2)) and (rel < 2e-2) and (max_abs < 0.5)
    print(f"\nGPU NUMERIC VERIFY {'PASS ✅' if ok else 'FAIL ❌'}  "
          f"(n={n}, max|Δ|={max_abs:.2e}, rel={rel:.2e}; bf16+cuDNN 跨运行噪声量级=正常)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
