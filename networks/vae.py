"""VAE 抽象层：训练代码只依赖统一接口，具体 VAE 可配置切换。

VAE gate（FLUX.2 f16c128 / Wan2.2 f16c48 / Qwen-Image f8c16 选型）还没跑完，
所以这里：
  * FakeVAE —— CPU 调试用，形状/接口与真 VAE 一致、确定性、免权重；
  * Flux2VAE —— FLUX.2 f16c128 真实 adapter，已集成并 GPU 验证（scripts/test_flux2_vae.py 通过）；
  * QwenImageVAE —— 真实 adapter 草稿占位（权重未下载/逻辑未验证，VAE gate 选型后补全）。

统一约定：
  encode(pixels[B,3,H,W] in [-1,1]) -> 归一化后 latent [B, C, H/f, W/f]
  decode(latent) -> pixels [B,3,H,W]
latent 归一化（channel-wise mean/std 或官方 scaling/shift）在 adapter 内部做，
模型侧拿到的永远是"训练就绪"的 latent（方案0608：不要裸 latent 训练）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class FakeVAE(nn.Module):
    """确定性假 VAE：种子化随机卷积 encode / 转置卷积 decode，全程冻结。

    只为在 CPU 上验证训练逻辑（形状、patchify、loss 流向），无重建意义。
    """

    def __init__(self, downsample_factor: int = 16, latent_channels: int = 128, seed: int = 0, **_):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.latent_channels = latent_channels
        g = torch.Generator().manual_seed(seed)
        self.enc = nn.Conv2d(3, latent_channels, kernel_size=downsample_factor, stride=downsample_factor)
        self.dec = nn.ConvTranspose2d(latent_channels, 3, kernel_size=downsample_factor, stride=downsample_factor)
        with torch.no_grad():
            for m in (self.enc, self.dec):
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.02)
                m.bias.zero_()
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def encode(
        self,
        pixels: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        # FakeVAE is deterministic; accept the argument so callers cannot
        # accidentally omit the real VAE's explicit posterior generator.
        del generator
        return self.enc(pixels)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.dec(latents)


class Flux2VAE(nn.Module):
    """FLUX.2 Autoencoder (`AutoencoderKLFlux2`, f16c128) adapter。

    VAE gate 已拍板用 FLUX.2（2026-06）。权重在 `/fsx/models/FLUX.2-klein-4B/vae`
    （config `_name_or_path` 指向 black-forest-labs/FLUX.2-dev，即 FLUX.2 共享 autoencoder）。

    model-space latent 约定（与 naifu networks/model_flux2.py 对齐、已 CPU round-trip 验证）：
      raw     = ae.encode(px).latent_dist.sample()                     # [B, 32, H/8, W/8]
      patched = patchify_2x2(raw)  ("b c (h ph) (w pw)->b (c ph pw) h w")  # [B, 128, H/16, W/16]
      latent  = (patched - bn.running_mean) / sqrt(bn.running_var + eps)   # BN 标准化
    其中 `ae.bn` 是 128 通道、**affine=False** 的 BatchNorm（只有 running stats），故 decode
    逆 BN + unpatchify 完全对称。模型侧拿到的就是「训练就绪」的 128ch / 16× latent
    （方案：不裸 latent 训练）。**注意 MoT 侧 patch_size 应 = 1**（VAE 已内含 2×2 patch）。

    依赖：`diffusers>=0.37`（含 AutoencoderKLFlux2）。VAE 全程冻结、不进 checkpoint。
    """

    downsample_factor = 16   # conv 8× × patch 2× = 16×
    latent_channels = 128    # 32ch × (2×2 patch) = 128

    def __init__(
        self,
        pretrained_path: str = "/fsx/models/FLUX.2-klein-4B/vae",
        dtype: str = "float32",
        sample_posterior: bool = True,
        **_,
    ):
        super().__init__()
        try:
            from diffusers import AutoencoderKLFlux2
        except ImportError as e:
            raise ImportError(
                "Flux2VAE 需要 diffusers>=0.37（含 AutoencoderKLFlux2）："
                "pip install --no-deps diffusers requests。CPU 纯逻辑调试可改用 FakeVAE。"
            ) from e
        self._dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.sample_posterior = bool(sample_posterior)
        self.ae = AutoencoderKLFlux2.from_pretrained(pretrained_path).to(self._dtype).eval()
        self.ae.requires_grad_(False)
        assert self.ae.config.latent_channels * 4 == self.latent_channels, "通道约定不符"

    # ----- raw VAE latent <-> model-space latent（patchify + BN 标准化）----- #
    @staticmethod
    def to_model_space(raw: torch.Tensor, ae) -> torch.Tensor:
        """[B,32,H/8,W/8] -> [B,128,H/16,W/16]：2×2 patchify + BN 标准化。"""
        patched = rearrange(raw, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=2, pw=2)
        bn = ae.bn
        mean = bn.running_mean.view(1, -1, 1, 1).to(patched)
        std = torch.sqrt(bn.running_var + bn.eps).view(1, -1, 1, 1).to(patched)
        out = (patched - mean) / std
        if bn.affine:  # 已确认 FLUX.2 为 False；保留以防换权重
            out = out * bn.weight.view(1, -1, 1, 1).to(patched) + bn.bias.view(1, -1, 1, 1).to(patched)
        return out

    @staticmethod
    def from_model_space(latent: torch.Tensor, ae) -> torch.Tensor:
        """[B,128,H/16,W/16] -> [B,32,H/8,W/8]：逆 BN + unpatchify（to_model_space 的逆）。"""
        bn = ae.bn
        x = latent
        if bn.affine:
            x = (x - bn.bias.view(1, -1, 1, 1).to(x)) / bn.weight.view(1, -1, 1, 1).to(x)
        std = torch.sqrt(bn.running_var + bn.eps).view(1, -1, 1, 1).to(x)
        mean = bn.running_mean.view(1, -1, 1, 1).to(x)
        x = x * std + mean
        return rearrange(x, "b (c ph pw) h w -> b c (h ph) (w pw)", ph=2, pw=2)

    @staticmethod
    def _sample_posterior(
        posterior,
        *,
        do_sample: bool,
        generator: torch.Generator | None,
        context: str,
    ) -> torch.Tensor:
        if not do_sample:
            return posterior.mode()
        if generator is None:
            raise RuntimeError(
                f"{context}: posterior sampling requires an explicit per-rank VAE generator"
            )
        posterior_device = torch.device(posterior.parameters.device)
        generator_device = torch.device(generator.device)
        if generator_device != posterior_device:
            raise RuntimeError(
                f"{context}: VAE generator must exactly match posterior device; "
                f"generator={generator_device}, posterior={posterior_device}"
            )
        return posterior.sample(generator=generator)

    def moments_to_model_space(
        self,
        parameters: torch.Tensor,
        sample: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """缓存数据线用：raw moments [B, 2*32, H/8, W/8]（mean‖logvar，batch_vae_flux.py 存的）
        -> model-space latent。每次可重采样 posterior。"""
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        dist = DiagonalGaussianDistribution(parameters)
        do_sample = self.sample_posterior if sample is None else sample
        raw = self._sample_posterior(
            dist,
            do_sample=do_sample,
            generator=generator,
            context="moments_to_model_space",
        )
        return self.to_model_space(raw, self.ae)

    @torch.no_grad()
    def encode(
        self,
        pixels: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """pixels[B,3,H,W] in [-1,1] -> model-space latent [B,128,H/16,W/16]。"""
        ref = next(self.ae.parameters())
        px = pixels.to(device=ref.device, dtype=ref.dtype)
        posterior = self.ae.encode(px).latent_dist
        raw = self._sample_posterior(
            posterior,
            do_sample=self.sample_posterior,
            generator=generator,
            context="encode",
        )
        return self.to_model_space(raw, self.ae)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """model-space latent [B,128,H/16,W/16] -> pixels[B,3,H,W]。"""
        ref = next(self.ae.parameters())
        raw = self.from_model_space(latents.to(device=ref.device, dtype=ref.dtype), self.ae)
        return self.ae.decode(raw).sample


class QwenImageVAE(nn.Module):
    """Qwen-Image v1 VAE (f8c16) adapter —— 草稿占位，未验证。

    TODO: 本地已有 /fsx/models/Qwen-Image（含 vae 子目录），可参考
    naifu networks/model_qwen_image.py 的 patch_vae（Conv3d->Conv2d, T=1）与
    latents_mean/std 归一化写法，VAE gate 后如选中再补全。
    """

    downsample_factor = 8
    latent_channels = 16

    def __init__(self, pretrained_path: str = "/fsx/models/Qwen-Image/vae", **_):
        super().__init__()
        raise NotImplementedError(
            "QwenImageVAE 是草稿占位。CPU 调试请用 FakeVAE。"
        )


_VAE_REGISTRY = {
    "fake": FakeVAE,
    "flux2": Flux2VAE,
    "qwen_image": QwenImageVAE,
}


def build_vae(cfg) -> nn.Module:
    """cfg: {name: fake|flux2|qwen_image, downsample_factor?, latent_channels?, ...}"""
    cfg = dict(cfg or {})
    name = cfg.pop("name", "fake")
    if name not in _VAE_REGISTRY:
        raise ValueError(f"unknown vae name {name!r}, choices: {sorted(_VAE_REGISTRY)}")
    return _VAE_REGISTRY[name](**cfg)
