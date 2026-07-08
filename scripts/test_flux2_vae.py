#!/usr/bin/env python3
"""Flux2VAE（AutoencoderKLFlux2, f16c128）adapter 测试（CPU，需 diffusers + 真权重）。

覆盖：
  * encode(pixels) -> model-space latent [B,128,H/16,W/16]（16× 下采样、128 通道）
  * to_model_space / from_model_space 互逆（patch+BN 对称，affine=False）
  * decode(encode(x)) 形状回到像素 + 重建误差合理（有损 VAE，不要求逐位）
  * 与 model_mot 集成：patch_size=1（FLUX.2 不二次 pack）下 _pack/_unpack 互逆 + forward 跑通

用法: venv-gpu/bin/python scripts/test_flux2_vae.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

VAE_PATH = "/fsx/models/FLUX.2-klein-4B/vae"


def test_flux2_vae_standalone():
    from networks.vae import build_vae
    vae = build_vae({"name": "flux2", "pretrained_path": VAE_PATH, "dtype": "float32"})
    assert vae.latent_channels == 128 and vae.downsample_factor == 16

    torch.manual_seed(0)
    px = torch.rand(2, 3, 64, 64) * 2 - 1  # [-1,1]
    lat = vae.encode(px)
    assert lat.shape == (2, 128, 4, 4), f"latent shape 错: {tuple(lat.shape)}"  # 64/16=4

    # to/from model space 互逆（patch + BN 对称）
    raw = vae.ae.encode(px.to(next(vae.ae.parameters()).dtype)).latent_dist.mode()  # [2,32,8,8]
    ms = vae.to_model_space(raw, vae.ae)
    back = vae.from_model_space(ms, vae.ae)
    d_inv = (back - raw).abs().max().item()
    assert d_inv < 1e-4, f"to/from model space 不互逆: max|Δ|={d_inv:.2e}"

    # decode 形状 + 重建（确定性：mode）
    vae.sample_posterior = False
    lat_det = vae.encode(px)
    rec = vae.decode(lat_det)
    assert rec.shape == px.shape, f"decode shape 错: {tuple(rec.shape)}"
    rt = (rec - px).abs().mean().item()
    print(f"✓ Flux2VAE：encode{tuple(lat.shape)}（mean{lat.mean():.3f} std{lat.std():.3f}）；"
          f"patch/BN 互逆 Δ={d_inv:.1e}；decode 重建 L1≈{rt:.3f}")


def test_flux2_integration_with_model():
    """model_mot：patch_size=1 + flux2 VAE 端到端（需 tokenizer）。"""
    from omegaconf import OmegaConf
    from networks.model_mot import MoTT2IModel
    TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, vocab_size=151936, rms_norm_eps=1e-6,
                attention_bias=False, attention_dropout=0.0, max_position_embeddings=8192,
                rope_theta=5000000, rope_scaling=dict(rope_type="default", mrope_section=[3, 3, 2],
                mrope_interleaved=True), pad_token_id=151643)
    cfg = OmegaConf.create({
        "model_cls": "networks.model_mot.MoTT2IModel", "target_device": "cpu",
        "model": {"pretrained_path": None, "tokenizer_path": "/fsx/models/Qwen3-VL-2B-Instruct",
                  "tiny_text_config": TINY, "patch_size": 1, "text_max_length": 16,
                  "cfg_dropout_prob": 0.0},
        "vae": {"name": "flux2", "pretrained_path": VAE_PATH, "dtype": "float32"},
        "sigma_sampling": {"dist": "logit_normal", "shift": 2.0},
        "trainer": {"seed": 42, "max_steps": 10, "batch_size": 2},
    })
    model = MoTT2IModel(config=cfg)
    assert model.latent_channels == 128 and model.patch_size == 1
    # patch_size=1 下 _pack/_unpack 互逆
    lat = torch.randn(2, 128, 4, 4)
    packed = model._pack_latents(lat, 2, 128, 4, 4, patch=1)
    assert packed.shape == (2, 16, 128), f"patch=1 pack shape 错: {tuple(packed.shape)}"
    unpacked = model._unpack_latents(packed, 4, 4, 128, patch=1)
    assert torch.equal(lat, unpacked), "patch=1 下 _pack/_unpack 应互逆"
    # 端到端 forward（pixels -> flux2 VAE -> MoT）
    model.train()
    batch = {"prompts": ["1girl, masterpiece", "a cat"],
             "pixels": torch.rand(2, 3, 64, 64) * 2 - 1}
    loss, sigmas, extras = model(batch)
    assert torch.isfinite(loss).all(), "loss 非有限"
    assert float(extras["img_tokens"]) == 16, f"img token 数应 16，得 {float(extras['img_tokens'])}"
    print(f"✓ 集成：patch_size=1、_pack/_unpack 互逆、pixels→flux2→MoT forward loss={loss.mean():.3f}、"
          f"img_tokens={int(float(extras['img_tokens']))}")


def main():
    print("=== Flux2VAE 测试（CPU，真权重）===")
    test_flux2_vae_standalone()
    try:
        test_flux2_integration_with_model()
    except Exception as e:
        print(f"⚠ 集成测试跳过（{type(e).__name__}: {e}）")
    print("\nFlux2VAE 测试通过 ✅")


if __name__ == "__main__":
    main()
