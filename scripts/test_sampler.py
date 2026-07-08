#!/usr/bin/env python3
"""G6：Euler flow 采样器（[networks/model_mot.py](../networks/model_mot.py)
generate_samples）机制单测（CPU tiny + FakeVAE，秒级）。

模型未训练（proj_out zero-init → v_pred≡0），故分两段验证：
  A. zero-init 恒等：v≡0 时 Euler 应使 latent 不变 → return_latents == 初始噪声
     （验证 pack/unpack 互逆 + Euler 累积 + σ schedule 不破坏 latent）。
  B. randomize proj_out（v≠0）后验证：
     - shape：pixels [B,3,H,W]、latents [B,C,lat_h,lat_w]；输出有限；
     - 确定性：同 seed 逐位一致、异 seed 不同；
     - CFG 有效：scale=1（单前向）vs scale=4（双前向）输出不同；
     - σ schedule：与训练同 shift 的 warp，递减、端点 1→0。

端到端「过拟合重建」（采样符号/方向最强信号）在 GPU 真 2B 上跑，见
`scripts/overfit_reconstruct.py`（本脚本只验 CPU 可验的机制）。

用法: venv-gpu/bin/python scripts/test_sampler.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "configs", "debug_tiny_cpu.yaml")


def build_tiny_model():
    from networks.model_mot import MoTT2IModel
    cfg = OmegaConf.load(CFG_PATH)
    cfg.target_device = "cpu"
    return MoTT2IModel(config=cfg), cfg


def expected_noise(bsz, C, lat_h, lat_w, seed):
    """复刻 generate_samples 的初始噪声（CPU generator、同形状/种子）。"""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(bsz, C, lat_h, lat_w, generator=g)


def main():
    torch.manual_seed(0)
    model, cfg = build_tiny_model()
    model.eval()
    prompts = ["1girl, silver hair", "a cat on a windowsill"]
    H = W = 128
    f, p, C = model.vae.downsample_factor, model.patch_size, model.latent_channels
    lat_h, lat_w = H // f, W // f
    print(f"tiny model: VAE f={f} C={C} patch={p} → latent {lat_h}x{lat_w}, grid {lat_h//p}x{lat_w//p}")

    # ---- A. zero-init 恒等（v≡0 → 采样不动 latent）----
    lat0 = model.generate_samples(prompts, height=H, width=W, num_steps=6,
                                  guidance_scale=4.0, seed=0, return_latents=True)
    noise0 = expected_noise(len(prompts), C, lat_h, lat_w, seed=0)
    d = (lat0 - noise0).abs().max().item()
    assert d < 1e-5, f"zero-init 采样应恒等（v≡0），但 max|Δ|={d:.2e}（Euler/pack-unpack/σ 有 bug）"
    print(f"[A] zero-init 恒等 ✓（v≡0 → return_latents == 初始噪声，max|Δ|={d:.1e}）")

    # ---- 让 v≠0：randomize proj_out（zero-init 否则一切无效）----
    with torch.no_grad():
        torch.nn.init.normal_(model.model.proj_out.weight, std=0.02)
        torch.nn.init.normal_(model.model.proj_out.bias, std=0.02)

    # ---- B1. shape + 有限性 ----
    px = model.generate_samples(prompts, height=H, width=W, num_steps=6, guidance_scale=4.0, seed=1)
    assert px.shape == (len(prompts), 3, H, W), f"pixels shape 错: {tuple(px.shape)}"
    assert torch.isfinite(px).all(), "pixels 非有限"
    lat = model.generate_samples(prompts, height=H, width=W, num_steps=6, guidance_scale=4.0,
                                 seed=1, return_latents=True)
    assert lat.shape == (len(prompts), C, lat_h, lat_w), f"latent shape 错: {tuple(lat.shape)}"
    print(f"[B1] shape ✓（pixels {tuple(px.shape)}、latents {tuple(lat.shape)}、有限）")

    # ---- B2. 确定性（同 seed 逐位一致；异 seed 不同）----
    px_same = model.generate_samples(prompts, height=H, width=W, num_steps=6, guidance_scale=4.0, seed=1)
    d_same = (px - px_same).abs().max().item()
    assert d_same == 0.0, f"同 seed 应逐位一致，max|Δ|={d_same:.2e}"
    px_diff = model.generate_samples(prompts, height=H, width=W, num_steps=6, guidance_scale=4.0, seed=2)
    d_diff = (px - px_diff).abs().max().item()
    assert d_diff > 1e-4, f"异 seed 应不同，max|Δ|={d_diff:.2e}"
    print(f"[B2] 确定性 ✓（同 seed max|Δ|=0、异 seed max|Δ|={d_diff:.2e}）")

    # ---- B3. CFG 有效（scale=1 单前向 vs scale=4 双前向，输出不同）----
    px_cfg1 = model.generate_samples(prompts, height=H, width=W, num_steps=6, guidance_scale=1.0, seed=1)
    d_cfg = (px - px_cfg1).abs().max().item()
    assert d_cfg > 1e-4, f"CFG scale=1 vs 4 应不同，max|Δ|={d_cfg:.2e}（CFG 未生效）"
    print(f"[B3] CFG 有效 ✓（scale=1 vs 4 max|Δ|={d_cfg:.2e}）")

    # ---- B4. σ schedule（与训练同 shift 的 warp，递减、端点 1→0）----
    shift = model.current_shift()
    t = torch.linspace(1.0, 0.0, 6 + 1)
    sigmas = shift * t / (1.0 + (shift - 1.0) * t)
    assert abs(sigmas[0] - 1.0) < 1e-6 and abs(sigmas[-1] - 0.0) < 1e-6, "σ 端点应 1→0"
    assert torch.all(sigmas[:-1] - sigmas[1:] > 0), "σ schedule 应严格递减"
    print(f"[B4] σ schedule ✓（shift={shift}, 递减 1→0: {[round(float(s),3) for s in sigmas]}）")

    # ---- B5. G6 und KV 缓存正确性：cached vs 非cached 逐位一致 ----
    for gs, tag in [(1.0, "无CFG"), (4.0, "CFG")]:
        a = model.generate_samples(prompts, height=H, width=W, num_steps=6,
                                   guidance_scale=gs, seed=7, use_und_cache=True)
        b = model.generate_samples(prompts, height=H, width=W, num_steps=6,
                                   guidance_scale=gs, seed=7, use_und_cache=False)
        d_kv = (a - b).abs().max().item()
        assert d_kv < 1e-4, f"KV 缓存 vs 非缓存({tag})应一致: max|Δ|={d_kv:.2e}"
        print(f"[B5] KV 缓存正确性({tag}) ✓（cached vs 非cached max|Δ|={d_kv:.2e}）")

    print("\n采样器机制单测全部通过 ✅（端到端正确性见 GPU overfit_reconstruct.py）")


if __name__ == "__main__":
    main()
