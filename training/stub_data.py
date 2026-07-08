"""数据占位 stub —— 本轮不做真实数据集读取（数据处理 T1-T5 还在跑）。

正式版会换成 naifu 的 dataset(store_cls) + sampler + Rust dataio 管线
（training/pretrain_data.py 契约：build_record / adapt_batch）。这里只提供
一个能让训练循环转起来的确定性假数据源，batch dict 的 key 与 naifu
adapt_batch 输出对齐（prompts / pixels / latents），方便后续无缝替换。

两种模式：
  * mode=pixels：随机图（[-1,1]），走 VAE encode 路径；
  * mode=latents：直接给随机 latent，跳过 VAE（最快的纯逻辑调试）。
固定一组 anime 风格 prompt 轮换；可 overfit_one=true 让每个 batch 完全相同
（检查模型能否过拟合单 batch，loss 应明显下降）。
"""

import torch

_STUB_PROMPTS = [
    "1girl, silver hair, school uniform, cherry blossoms, masterpiece",
    "a boy standing on a rooftop at sunset, anime style, dramatic lighting",
    "2girls, maid outfit, victorian mansion library, detailed background",
    "chibi dragon eating a giant strawberry, watercolor style",
    "cyberpunk city street at night, neon signs, rain, wide shot",
    "portrait of a knight with golden armor, intricate details",
    "a cat sleeping on a windowsill, soft morning light, slice of life",
    "magical girl transformation scene, sparkles, dynamic pose",
]


class StubT2IDataset:
    """配置见 configs/*.yaml 的 dataset 段。"""

    def __init__(
        self,
        image_size: int = 512,
        latent_channels: int = 128,
        downsample_factor: int = 16,
        mode: str = "latents",  # latents | pixels
        num_batches_per_epoch: int = 1000,
        seed: int = 0,
        overfit_one: bool = False,
        **_,
    ):
        assert mode in ("latents", "pixels")
        self.image_size = int(image_size)
        self.latent_channels = int(latent_channels)
        self.downsample_factor = int(downsample_factor)
        self.mode = mode
        self.num_batches_per_epoch = int(num_batches_per_epoch)
        self.seed = int(seed)
        self.overfit_one = bool(overfit_one)

    def __len__(self):
        return self.num_batches_per_epoch

    def build_batch(self, batch_size: int, batch_idx: int) -> dict:
        idx = 0 if self.overfit_one else batch_idx
        g = torch.Generator().manual_seed(self.seed + idx)
        prompts = [_STUB_PROMPTS[(idx * batch_size + i) % len(_STUB_PROMPTS)] for i in range(batch_size)]
        if self.overfit_one:
            prompts = [_STUB_PROMPTS[i % len(_STUB_PROMPTS)] for i in range(batch_size)]
        if self.mode == "pixels":
            pixels = torch.rand(batch_size, 3, self.image_size, self.image_size, generator=g) * 2 - 1
            return {"prompts": prompts, "pixels": pixels, "latents": []}
        s = self.image_size // self.downsample_factor
        latents = torch.randn(batch_size, self.latent_channels, s, s, generator=g)
        return {"prompts": prompts, "latents": latents}

    def loader(self, batch_size: int):
        """无限迭代器（naifu 的 continuous sampler 语义的极简替身）。"""
        batch_idx = 0
        while True:
            yield self.build_batch(batch_size, batch_idx)
            batch_idx += 1
