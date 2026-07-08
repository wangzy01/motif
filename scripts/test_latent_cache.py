"""CPU 单测：latent cache 训练读路径的正确性（组件级）。

[A] LatentCacheIndex：加载 uid2path、get_many 形状/dtype、缺失 uid 不返回、anime uid 命中。
[B] Flux2VAE.moments_to_model_space：形状/dtype/finite、mode 确定性、sample 重采样有变化。
[C] 数值对拍（信息性）：取一个 anime uid，R2 取图→同款预处理→CPU 重 encode 出 posterior 参数，
    与缓存 moments 比（缓存是 bf16/GPU 编的，CPU/fp32 重编只能"接近"，报相对误差，粗界拦截大错）。

数据集级集成 + 真训练步 + GPU 精确 cache==online 对拍 由 JCC 端到端 smoke 覆盖（同设备同 dtype）。
跑: PYTHONPATH=. CUDA_VISIBLE_DEVICES= python scripts/test_latent_cache.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.dataline.latent_cache import LatentCacheIndex
from training.dataline.image_fetch import fetch_url_bytes, decode_rgb, resize_crop_uint8, uint8_to_norm_tensor

ROOT = "/s3/evan/mot_latents/complete_v1_flux2_moments"
STAGE = "256"
STAGE_DIR = f"{ROOT}/stage={STAGE}"
VAE_PATH = "/fsx/models/FLUX.2-klein-4B/vae"

results = []
def check(name, ok, extra=""):
    results.append((name, ok))
    print(("  PASS " if ok else "  FAIL ") + name + (f"  [{extra}]" if extra else ""), flush=True)


print("=== [A] LatentCacheIndex ===", flush=True)
idx = LatentCacheIndex(STAGE_DIR)
print(f"  index size: {len(idx)} uid", flush=True)
check("索引非空", len(idx) > 30_000_000, f"{len(idx)}")
# 各域 uid 抽样
sample_uids = list(idx.uid2path.keys())[:2000]
n_anime = sum(1 for u in sample_uids if u.startswith("anime:"))
got = idx.get_many(sample_uids)
check("get_many 全部命中(已知 uid)", len(got) == len(sample_uids), f"{len(got)}/{len(sample_uids)}")
shapes_ok = all(tuple(t.shape) == (64, 32, 32) for t in got.values())
dtype_ok = all(str(t.dtype).endswith("bfloat16") for t in got.values())
finite_ok = all(torch.isfinite(t.float()).all() for t in list(got.values())[:200])
check("moments 形状全 [64,32,32]", shapes_ok)
check("moments dtype 全 bf16", dtype_ok)
check("moments 全 finite(抽200)", finite_ok)
check("缺失 uid 不返回", "anime:__NOT_A_REAL_UID__" not in idx.get_many(["anime:__NOT_A_REAL_UID__"]))
# anime uid 命中（对齐验证：训练 key->anime:key 在索引）
anime_uids = [u for u in sample_uids if u.startswith("anime:")][:50]
check("anime uid 命中索引", len(anime_uids) > 0 and all(u in idx for u in anime_uids), f"anime采样{n_anime}")

print("=== [B] Flux2VAE.moments_to_model_space ===", flush=True)
from networks.vae import Flux2VAE
vae = Flux2VAE(pretrained_path=VAE_PATH, dtype="float32")
m = list(got.values())[0].unsqueeze(0)  # [1,64,32,32] bf16
lat_mode1 = vae.moments_to_model_space(m, sample=False)
lat_mode2 = vae.moments_to_model_space(m, sample=False)
check("model-space 形状 [1,128,16,16]", tuple(lat_mode1.shape) == (1, 128, 16, 16), str(tuple(lat_mode1.shape)))
check("model-space finite", bool(torch.isfinite(lat_mode1.float()).all()))
check("mode 路径确定性(两次相同)", torch.equal(lat_mode1, lat_mode2))
s1 = vae.moments_to_model_space(m, sample=True)
s2 = vae.moments_to_model_space(m, sample=True)
check("sample 路径重采样(两次不同)", not torch.equal(s1, s2))
# sample 的均值应接近 mode（posterior 均值=mode）
many = torch.stack([vae.moments_to_model_space(m, sample=True) for _ in range(64)], 0).mean(0)
rel = (many - lat_mode1).abs().mean() / (lat_mode1.abs().mean() + 1e-6)
check("sample 多次均值≈mode", float(rel) < 0.15, f"rel={float(rel):.3f}")

print("=== [C] 缓存 vs 在线重编码 数值对拍(信息性) ===", flush=True)
# 取一个 anime uid，R2 取图，同款预处理，CPU 重编码 posterior 参数，与缓存 moments 比
try:
    import glob, pyarrow.parquet as pq
    from training.dataline.config import load_sampling_config
    cfg = load_sampling_config("/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json")
    # 找一个 anime uid 的 s3key：从薄 manifest 取
    man = sorted(glob.glob(os.path.join(cfg.manifest_anime, "*.parquet")))[0]
    t = pq.ParquetFile(man).read(columns=["uid", "s3key"])
    u2s = dict(zip(t.column("uid").to_pylist(), t.column("s3key").to_pylist()))
    test_uid = next(u for u in idx.uid2path if u.startswith("anime:") and u in u2s)
    raw = fetch_url_bytes(u2s[test_uid], retries=4, timeout=30)
    img = decode_rgb(raw)
    u8 = resize_crop_uint8(img, 256)
    px = uint8_to_norm_tensor(u8).unsqueeze(0)  # [1,3,256,256]
    # CPU 重编码 -> posterior 参数（mean‖logvar）
    posterior = vae.ae.encode(px.to(torch.float32)).latent_dist
    re_params = posterior.parameters  # [1,64,32,32] fp32
    cached = idx.get_many([test_uid])[test_uid].unsqueeze(0).float()  # [1,64,32,32]
    # mode 路径 model-space 对比
    lat_re = vae.to_model_space(posterior.mode(), vae.ae)
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    lat_cached = vae.to_model_space(DiagonalGaussianDistribution(cached).mode(), vae.ae)
    rel_params = (re_params - cached).abs().mean() / (cached.abs().mean() + 1e-6)
    rel_ms = (lat_re - lat_cached).abs().mean() / (lat_cached.abs().mean() + 1e-6)
    print(f"  uid={test_uid}", flush=True)
    print(f"  参数相对误差(CPU-fp32重编 vs 缓存-GPU-bf16): {float(rel_params):.4f}", flush=True)
    print(f"  model-space(mode) 相对误差: {float(rel_ms):.4f}", flush=True)
    # 缓存是 GPU/bf16 编的，CPU/fp32 重编只能接近；粗界拦大错（同图同VAE应 ~几个百分点内）
    check("缓存 moments ≈ 在线重编码(粗界<8%)", float(rel_ms) < 0.08, f"rel_ms={float(rel_ms):.4f}")
except Exception as e:  # noqa: BLE001
    print(f"  [C] 跳过(取图/网络/依赖问题): {type(e).__name__}: {e}", flush=True)

print("\n=== RESULT ===", flush=True)
allok = all(ok for _, ok in results)
for name, ok in results:
    print(("  PASS " if ok else "  FAIL ") + name, flush=True)
print("\nLATENT CACHE TEST", "PASS ✅" if allok else "FAIL ❌", flush=True)
sys.exit(0 if allok else 1)
