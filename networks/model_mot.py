"""naifu 风格的 model 包装类：MoT T2I（stage1a，仅 t2i）。

接口与 naifu `train_fsdp2.py` 的 model_cls 契约对齐（forward(batch) ->
(loss, timesteps, extras)、get_optimizer_param_groups、fully_shard、
generate_samples），后续可把 networks/ 两个文件直接搬进 naifu。

训练语义（MoT方案0608「初版架构（求稳版）」）：
  * frozen Qwen3-VL-2B 文本塔做理解路；生成塔复制初始化、可训练；
  * v-prediction + rectified flow-matching + 纯 MSE；
  * sigma：logit-normal 采样 + per-分辨率固定 noise-shift（v3 决策，不再按训练进度
    ramp）：256→1.0、512→2.0、1024→4.0(sweep 3.3-5)，由 sigma_sampling.shift 给定；
  * 10% caption dropout（CFG）；frozen LM 下 weight decay=0；
  * 文本走 Qwen-Image 式 chat template（可配置关闭）。
"""

import math
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from networks.arch_mot import MoTT2ITransformer
from networks.vae import build_vae

# Qwen-Image 官方 text-encode 模板（diffusers QwenImagePipeline 同款）。
# 注意：MoT 是逐层联合注意力，模板 token 始终作为上下文参与（没有
# Qwen-Image 那种 encode 后 drop prefix 的步骤）。
DEFAULT_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def _log(msg: str):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"[model_mot] {msg}", flush=True)


class MoTT2IModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        mcfg = config.model
        self.target_device = config.get("target_device", "cpu")

        self.patch_size = int(mcfg.get("patch_size", 2))
        self.text_max_length = int(mcfg.get("text_max_length", 384))  # caption ~343 + 模板 ~41（对标 Anima-2B/Qwen-Image 长 caption）
        self.cfg_dropout_prob = float(mcfg.get("cfg_dropout_prob", 0.1))
        # 文字样本特例（v3）：含文字/海报域样本用更低（甚至 0）dropout，避免学成无条件出字。
        self.text_rich_cfg_dropout_prob = float(mcfg.get("text_rich_cfg_dropout_prob", 0.0))
        tpl_cfg = mcfg.get("prompt_template", {}) or {}
        self.prompt_template = (
            str(tpl_cfg.get("template", DEFAULT_PROMPT_TEMPLATE))
            if bool(tpl_cfg.get("enabled", True))
            else "{}"
        )

        # 训练进度（train.py 每 step 调 set_train_progress，驱动 shift schedule）
        self._global_step = 0
        self._max_steps = max(1, int(config.trainer.get("max_steps", 1)))

        seed = int(config.trainer.get("seed", 42))
        # rank_independent_seed=true：强制各 rank 同 seed（同噪声/同 dropout）。仅用于
        # FSDP2 多卡 vs 单卡的严格数值一致性验证（2卡平均梯度=1卡 → loss 轨迹应吻合）；
        # 真实 DP 训练保持 false（各 rank 不同噪声 = 梯度更多样）。
        self._rng_base_seed = seed
        self._rng_global_rank = dist.get_rank() if dist.is_initialized() else 0
        self._rng_rank_independent = bool(mcfg.get("rank_independent_seed", False))
        self._rng_rank_stride = 1_000_003
        self._rng_stream_offsets = {"dropout": 11, "diffusion": 23, "vae": 37}

        # ----- tokenizer ----- #
        tok_path = mcfg.get("tokenizer_path") or mcfg.get("pretrained_path")
        assert tok_path, "model.tokenizer_path 或 model.pretrained_path 至少要给一个"
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ----- VAE ----- #
        self.vae = build_vae(config.get("vae"))
        self.latent_channels = int(self.vae.latent_channels)
        # Constructor-time default is CPU.  train.py calls this again *after*
        # model.to(device), which is required because torch.Generator is not a
        # Module buffer and does not migrate with .to().
        self.configure_training_rng(global_rank=self._rng_global_rank, device=torch.device("cpu"))
        patch_dim = self.latent_channels * self.patch_size**2

        # ----- backbone ----- #
        text_config = self._build_text_config(mcfg)

        def _ablation_cfg(key):
            v = mcfg.get(key, None)
            if v is None:
                return None
            from omegaconf import OmegaConf as _OC
            return _OC.to_container(v, resolve=True) if not isinstance(v, dict) else dict(v)

        self.model = MoTT2ITransformer(
            text_config,
            patch_dim=patch_dim,
            time_gap=int(mcfg.get("time_gap", 2048)),
            timestep_multiplier=float(mcfg.get("timestep_multiplier", 1.0)),  # v3：Cosmos3 对齐默认 1.0
            sandwich_norm=bool(mcfg.get("sandwich_norm", True)),              # Day-1 baseline，默认开
            noise_condition=bool(mcfg.get("noise_condition", True)),         # B3 消融，默认开
            long_skip=_ablation_cfg("long_skip"),                            # B2 消融，默认关
            text_adapter=_ablation_cfg("text_adapter"),                      # B1 消融，默认关
            und_lora=_ablation_cfg("und_lora"),                              # Ab3 消融，默认关
            abs_sincos=_ablation_cfg("abs_sincos"),                          # B4 消融，默认关
        )

        pretrained_path = mcfg.get("pretrained_path")
        if pretrained_path and not bool(config.get("skip_init_checkpoint_load", False)):
            self._load_und_weights(pretrained_path)
        self.model.init_generation_tower_from_und()
        self.model.freeze_understanding()

        if bool(mcfg.get("gradient_checkpointing", False)):
            self.model.enable_gradient_checkpointing()

        stats = self.model.param_count()
        _log(
            f"params: total={stats['total'] / 1e6:.1f}M, "
            f"trainable={stats['trainable'] / 1e6:.1f}M, frozen={stats['frozen'] / 1e6:.1f}M"
        )

    # ------------------------------------------------------------------ #
    # 构建 / 加载
    # ------------------------------------------------------------------ #
    def configure_training_rng(self, *, global_rank: int, device: torch.device | str) -> None:
        """Create explicit streams without touching PyTorch's global RNG.

        Dropout/diffusion deliberately retain their historical CPU streams so
        this control run changes only the broken VAE posterior stream.  The VAE
        stream is created on the exact live posterior device.
        """
        global_rank = int(global_rank)
        if global_rank < 0:
            raise ValueError(f"global_rank must be non-negative, got {global_rank}")
        if dist.is_initialized() and global_rank != dist.get_rank():
            raise RuntimeError(
                f"training RNG global_rank={global_rank} differs from process-group rank={dist.get_rank()}"
            )
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type not in ("cpu", "cuda"):
            raise ValueError(f"unsupported VAE RNG device: {device}")
        vae_ref = next(self.vae.parameters(), None)
        if vae_ref is not None and vae_ref.device != device:
            raise RuntimeError(
                f"configure_training_rng must run after model.to(device): "
                f"VAE={vae_ref.device}, requested={device}"
            )

        effective_rank = 0 if self._rng_rank_independent else global_rank
        seed_base = self._rng_base_seed + effective_rank * self._rng_rank_stride
        self.dropout_generator = torch.Generator(device="cpu").manual_seed(
            seed_base + self._rng_stream_offsets["dropout"]
        )
        self.diffusion_generator = torch.Generator(device="cpu").manual_seed(
            seed_base + self._rng_stream_offsets["diffusion"]
        )
        self.vae_generator = torch.Generator(device=device).manual_seed(
            seed_base + self._rng_stream_offsets["vae"]
        )
        self._rng_global_rank = global_rank
        self._training_rng_device = device

    def training_rng_generators(self) -> dict[str, torch.Generator]:
        return {
            "dropout": self.dropout_generator,
            "diffusion": self.diffusion_generator,
            "vae": self.vae_generator,
        }

    def training_rng_contract(self) -> dict:
        return {
            "base_seed": self._rng_base_seed,
            "rank_independent_seed": self._rng_rank_independent,
            "rank_stride": self._rng_rank_stride,
            "stream_offsets": dict(self._rng_stream_offsets),
        }

    def training_rng_device(self) -> torch.device:
        return self._training_rng_device

    @staticmethod
    def _build_text_config(mcfg) -> Qwen3VLTextConfig:
        pretrained_path = mcfg.get("pretrained_path")
        if pretrained_path:
            cfg_path = os.path.join(str(pretrained_path), "config.json")
            full = Qwen3VLTextConfig.from_pretrained(pretrained_path) \
                if not os.path.exists(cfg_path) else None
            if full is None:
                import json
                with open(cfg_path, encoding="utf-8") as f:
                    d = json.load(f)
                full = Qwen3VLTextConfig(**d["text_config"])
            return full
        tiny = mcfg.get("tiny_text_config")
        assert tiny, "没有 pretrained_path 时必须提供 model.tiny_text_config"
        from omegaconf import OmegaConf
        tiny = OmegaConf.to_container(tiny, resolve=True) if not isinstance(tiny, dict) else dict(tiny)
        return Qwen3VLTextConfig(**tiny)

    def _load_und_weights(self, pretrained_path: str):
        """从 HF Qwen3-VL checkpoint 加载理解路权重（忽略 ViT / lm_head）。"""
        from safetensors.torch import load_file

        files = sorted(
            f for f in os.listdir(pretrained_path)
            if f.endswith(".safetensors") and not f.startswith("._")
        )
        assert files, f"{pretrained_path} 下没有 safetensors 权重"

        prefix = "model.language_model."
        own_state = self.model.state_dict()
        loaded, skipped = set(), set()
        for fname in files:
            sd = load_file(os.path.join(pretrained_path, fname))
            mapped = {}
            for k, v in sd.items():
                if not k.startswith(prefix):
                    skipped.add(k.split(".")[0] + ".*")
                    continue
                name = k[len(prefix):]
                if name in own_state:
                    mapped[name] = v
                    loaded.add(name)
                else:
                    skipped.add(name)
            self.model.load_state_dict(mapped, strict=False)
            del sd

        # und 路应当全部命中（embed/und attn/mlp/norms/final norm）
        expected_und = {
            n for n in own_state
            if "_moe_gen" not in n and not n.startswith(("proj_in", "proj_out", "time_embed"))
            and "rotary_emb" not in n and "lora_und" not in n
        }
        missing = expected_und - loaded
        assert not missing, f"und 权重缺失 {len(missing)} 个，例如 {sorted(missing)[:5]}"
        _log(f"loaded und weights: {len(loaded)} tensors from {len(files)} shard(s); "
             f"skipped top-level: {sorted(skipped)[:8]}")

    # ------------------------------------------------------------------ #
    # 文本 / latent 准备
    # ------------------------------------------------------------------ #
    def encode_text(self, prompts: list[str]):
        """固定 pad 到 text_max_length（序列形状稳定，便于编译/分布式）。"""
        texts = [self.prompt_template.format(p) for p in prompts]
        tok = self.tokenizer(
            texts,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        return tok.input_ids.to(device), tok.attention_mask.bool().to(device)

    def _apply_cfg_dropout(self, prompts: list[str], text_rich=None, cfg_keep=None) -> list[str]:
        """全 prompt dropout（CFG，**唯一** dropout 点；数据线不再自行 dropout，见 review F4）。
        - `text_rich`（per-sample bool，可缺省）为真 → 用 text_rich_cfg_dropout_prob（默认 0），
          避免文字样本被学成无条件出字（v3）。
        - `cfg_keep`（per-sample bool，可缺省）为真 → prob 0 **永不 dropout**（数据线给
          tag-only anime 行带此标志，落实 config『tag-only 永不 drop』）。"""
        if not self.training or (self.cfg_dropout_prob <= 0 and self.text_rich_cfg_dropout_prob <= 0):
            return prompts
        n = len(prompts)
        probs = torch.full((n,), self.cfg_dropout_prob)
        if text_rich is not None:
            tr = torch.as_tensor(list(text_rich), dtype=torch.bool)
            probs = torch.where(tr, torch.full((n,), self.text_rich_cfg_dropout_prob), probs)
        if cfg_keep is not None:                                  # tag-only 等永不 drop 的样本
            kp = torch.as_tensor(list(cfg_keep), dtype=torch.bool)
            probs = torch.where(kp, torch.zeros(n), probs)
        keep = torch.rand(n, generator=self.dropout_generator) >= probs
        return [p if k else "" for p, k in zip(prompts, keep.tolist())]

    @torch.no_grad()
    def prepare_latents(self, batch) -> torch.Tensor:
        """batch -> 归一化 latent [B, C, h, w]。优先用现成 latents（缓存/stub）。"""
        latents = batch.get("latents", None)
        if latents is not None and not (isinstance(latents, list) and len(latents) == 0):
            if isinstance(latents, list):
                latents = torch.stack(latents, dim=0)
            return latents.to(next(self.model.parameters()).device)
        # 缓存数据线给原始 VAE moments（[B,2*32,S/8,S/8]），模型侧转 model-space 并重采样 posterior
        # （复用 Flux2VAE.moments_to_model_space，与在线 encode 同分布）。additive 分支：
        # 默认（pixels/latents）路径不变，不影响在跑训练。
        moments = batch.get("moments", None)
        if moments is not None and not (isinstance(moments, list) and len(moments) == 0):
            if isinstance(moments, list):
                moments = torch.stack(moments, dim=0)
            moments = moments.to(next(self.model.parameters()).device)
            return self.vae.moments_to_model_space(
                moments,
                generator=self.vae_generator,
            )
        pixels = batch["pixels"]
        if isinstance(pixels, list):
            pixels = torch.stack(pixels, dim=0)
        pixels = pixels.to(next(self.model.parameters()).device)
        return self.vae.encode(pixels.float(), generator=self.vae_generator)

    # ----- patchify（Qwen-Image/FLUX 同款 2x2 pack，复制自 naifu）----- #
    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width, patch=2):
        latents = latents.view(
            batch_size, num_channels_latents, height // patch, patch, width // patch, patch
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(
            batch_size, (height // patch) * (width // patch), num_channels_latents * patch * patch
        )

    @staticmethod
    def _unpack_latents(latents, grid_h, grid_w, num_channels, patch=2):
        bsz = latents.shape[0]
        latents = latents.view(bsz, grid_h, grid_w, num_channels, patch, patch)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(bsz, num_channels, grid_h * patch, grid_w * patch)

    # ------------------------------------------------------------------ #
    # sigma 采样：logit-normal + 按训练进度的 shift schedule
    # ------------------------------------------------------------------ #
    def set_train_progress(self, global_step: int, max_steps: int | None = None):
        self._global_step = int(global_step)
        if max_steps:
            self._max_steps = max(1, int(max_steps))

    def current_shift(self) -> float:
        """noise-shift = per-分辨率固定常数（v3 决策，不再按训练进度 ramp）。
        约定：256→1.0、512→2.0、1024→4.0(sweep 3.3-5)。config 用 sigma_sampling.shift
        给定；兼容旧字段 shift_start 作 fallback。注意它与 timestep_multiplier 是两回事：
        shift 改 sigma 分布，multiplier 只改 timestep embedding 的输入尺度。"""
        scfg = self.config.get("sigma_sampling", {}) or {}
        return float(scfg.get("shift", scfg.get("shift_start", 1.0)))

    def get_sigmas(self, batch_size: int) -> torch.Tensor:
        scfg = self.config.get("sigma_sampling", {}) or {}
        dist_name = str(scfg.get("dist", "logit_normal"))
        if dist_name == "logit_normal":
            u = torch.randn(batch_size, generator=self.diffusion_generator)
            sigmas = torch.sigmoid(
                float(scfg.get("logit_mean", 0.0)) + float(scfg.get("logit_std", 1.0)) * u
            )
        elif dist_name == "uniform":
            sigmas = torch.rand(batch_size, generator=self.diffusion_generator)
        else:
            raise ValueError(f"unknown sigma dist {dist_name!r}")
        sigmas = sigmas.clamp(1e-4, 1.0 - 1e-4)
        shift = self.current_shift()
        # SD3/FLUX 式 shift 变换：sigma' = s*sigma / (1 + (s-1)*sigma)
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        return sigmas

    # ------------------------------------------------------------------ #
    # 训练 forward：rectified flow, v = noise - x0
    # ------------------------------------------------------------------ #
    def forward(self, batch):
        prompts = self._apply_cfg_dropout(list(batch["prompts"]), batch.get("text_rich"), batch.get("cfg_keep"))
        input_ids, text_mask = self.encode_text(prompts)

        latents = self.prepare_latents(batch)  # [B, C, h, w]
        bsz, c, h, w = latents.shape
        p = self.patch_size
        assert h % p == 0 and w % p == 0, f"latent {h}x{w} 不能被 patch {p} 整除"
        grid_hw = (h // p, w // p)
        x0 = self._pack_latents(latents, bsz, c, h, w, patch=p)  # [B, N, C*p*p]

        device = x0.device
        sigmas = self.get_sigmas(bsz).to(device)  # [B]
        noise = torch.randn(x0.shape, generator=self.diffusion_generator).to(device, x0.dtype)
        sig = sigmas.view(bsz, 1, 1).to(x0.dtype)
        x_t = (1.0 - sig) * x0 + sig * noise
        target = noise - x0

        v_pred = self.model(input_ids, text_mask, x_t, grid_hw, sigmas)
        loss = (v_pred.float() - target.float()).pow(2).mean(dim=(1, 2))  # [B]

        extras = {
            "flow_mse": loss.detach().mean(),
            "shift": torch.tensor(self.current_shift()),
            "sigma_mean": sigmas.detach().mean(),
            "img_tokens": torch.tensor(float(x0.shape[1])),
        }
        return loss, sigmas, extras

    # ------------------------------------------------------------------ #
    # naifu 契约的其余接口
    # ------------------------------------------------------------------ #
    def get_optimizer_param_groups(self, weight_decay: float = 0.0, base_lr: float | None = None):
        # 方案0608：frozen LM 下 weight decay = 0（参考 Lance）
        main_params, lora_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_und" in name:
                lora_params.append(p)
            else:
                main_params.append(p)
        groups = []
        if main_params:
            groups.append({"params": main_params, "weight_decay": weight_decay})
        if lora_params:
            ul = self.config.model.get("und_lora", {}) or {}
            lr_scale = float(ul.get("lr_scale", 1.0))
            group = {"params": lora_params, "weight_decay": weight_decay}
            if base_lr is not None:
                group["lr"] = float(base_lr) * lr_scale
            groups.append(group)
        return groups

    def fully_shard(self, **fsdp_kwargs):
        """FSDP2 包装（按 naifu 惯例 per-layer + 整体）。2026-06-17 单卡 nproc=1 验过码路。"""
        from torch.distributed.fsdp import fully_shard
        for layer in self.model.layers:
            fully_shard(layer, **fsdp_kwargs, reshard_after_forward=True)
        fully_shard(self.model, **fsdp_kwargs, reshard_after_forward=False)
        _log(f"Applied FSDP2: {len(self.model.layers)} per-layer units")

    # ------------------------------------------------------------------ #
    # EMA（G5；仅可训练参数。FSDP2 下 EMA 镜像 sharded DTensor，shard-wise lerp，
    # ckpt 用 full_tensor gather / distribute_tensor scatter。单进程下退化为普通张量。
    # 须在 fully_shard 之后 init（这样 EMA 与参数同 DTensor 布局）。参考 naifu FSDP2EMA。
    # ------------------------------------------------------------------ #
    def init_ema(self, decay: float = 0.9999):
        self.ema_decay = float(decay)
        self._ema = {
            n: p.detach().clone().float()
            for n, p in self.model.named_parameters() if p.requires_grad
        }
        _log(f"EMA initialized: decay={self.ema_decay}, {len(self._ema)} trainable tensors")

    @torch.no_grad()
    def update_ema(self):
        """ema = decay*ema + (1-decay)*param（lerp_(end, 1-decay)）。每 optimizer.step 后调。"""
        if not getattr(self, "_ema", None):
            return
        w = 1.0 - self.ema_decay
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._ema[n].lerp_(p.detach().float(), w)

    @torch.no_grad()
    def ema_state_dict(self):
        """gather 成完整张量（FSDP2 下 full_tensor 是 collective，所有 rank 须调）→ cpu fp32。"""
        if not getattr(self, "_ema", None):
            return None
        out = {}
        for n, t in self._ema.items():
            full = t.full_tensor() if hasattr(t, "full_tensor") else t
            out[n] = full.detach().cpu()
        return out

    @torch.no_grad()
    def load_ema_state_dict(self, sd: dict):
        if not getattr(self, "_ema", None) or sd is None:
            return
        from torch.distributed.tensor import DTensor, distribute_tensor
        for n, t in self._ema.items():
            full = sd[n]
            if isinstance(t, DTensor):
                t.copy_(distribute_tensor(full.to(t.device, t.dtype), t.device_mesh, t.placements))
            else:
                t.copy_(full.to(t.device, t.dtype))

    @torch.no_grad()
    def generate_samples(
        self,
        prompts: list[str],
        height: int = 512,
        width: int = 512,
        num_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int | None = None,
        return_latents: bool = False,
        use_und_cache: bool = True,
    ):
        """Euler flow 采样 + 经典双前向 CFG。返回 pixels [B,3,H,W]（或 model-space latent）。

        约定（与训练严格一致，见 forward）：x_t=(1-σ)x0+σ·noise、v=noise-x0=dx_t/dσ。
        从 σ=1（纯噪声）沿递减 σ schedule 积分到 σ=0（数据）：Euler `x += (σ_next-σ_cur)·v`。
        σ schedule 用**与训练同一个 shift**（SD3/FLUX 乘式 `σ'=s·σ/(1+(s-1)σ)`，per-分辨率
        固定常数 current_shift），把 linspace(1→0) warp 到训练所在的 σ 空间。

        CFG（双前向，我们用 caption dropout 训练、无 guidance-embed）：
          v = v_uncond + scale·(v_cond - v_uncond)；scale==1 退化为单前向（纯 cond）。
        frozen und 逐层 KV 缓存（采样 loop 外只算一次）是后续优化，本版每步重算 und。
        ⚠️ 模型未训练时画质无意义——本函数定位是机制正确（σ 方向/Euler 符号/CFG/确定性）。
        """
        was_training = self.training
        self.eval()
        device = next(self.model.parameters()).device
        bsz = len(prompts)
        f, p, C = self.vae.downsample_factor, self.patch_size, self.latent_channels
        assert height % f == 0 and width % f == 0, f"H/W 必须能被 VAE 下采样因子 {f} 整除"
        lat_h, lat_w = height // f, width // f
        assert lat_h % p == 0 and lat_w % p == 0, f"latent {lat_h}x{lat_w} 不能被 patch {p} 整除"
        grid_h, grid_w = lat_h // p, lat_w // p

        # 文本：cond + （CFG 时）uncond=空 prompt（过同一模板）
        ids_c, mask_c = self.encode_text(list(prompts))
        do_cfg = guidance_scale is not None and float(guidance_scale) != 1.0
        if do_cfg:
            ids_u, mask_u = self.encode_text([""] * bsz)

        # 初始噪声（model-space latent → pack）。CPU generator → 跨设备可复现。
        gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        lat_noise = torch.randn(bsz, C, lat_h, lat_w, generator=gen).to(device)
        x = self._pack_latents(lat_noise, bsz, C, lat_h, lat_w, patch=p)  # [B,N,C*p*p]

        # σ schedule：与训练同 shift；linspace(1→0) 再 warp（递减、首=σ_max、末=0）
        shift = self.current_shift()
        t = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        sigmas = shift * t / (1.0 + (shift - 1.0) * t)  # [num_steps+1]

        grid_hw = (grid_h, grid_w)
        # G6 KV 缓存：frozen und 跨步不变 → loop 外预计算每层 und K/V，loop 内只跑 gen 塔（~2×）。
        # 仅 baseline 路径（adapter 臂把 und 整形成 T_cond，逐层缓存语义不同 → 回退 per-step forward）。
        use_cache = bool(use_und_cache) and not self.model.text_adapter_enabled
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with amp:
            if use_cache:
                cache_c = self.model.precompute_und_cache(ids_c, mask_c, grid_hw)
                cache_u = self.model.precompute_und_cache(ids_u, mask_u, grid_hw) if do_cfg else None
            for i in range(num_steps):
                s = torch.full((bsz,), float(sigmas[i]), device=device)  # 当前噪声级（训练所见的 σ）
                if use_cache:
                    v = self.model.forward_gen_cached(x, s, cache_c[0], cache_c[1], cache_c[2], grid_hw)
                    if do_cfg:
                        v_u = self.model.forward_gen_cached(x, s, cache_u[0], cache_u[1], cache_u[2], grid_hw)
                        v = v_u + guidance_scale * (v - v_u)
                else:
                    v = self.model(ids_c, mask_c, x, grid_hw, s)
                    if do_cfg:
                        v_u = self.model(ids_u, mask_u, x, grid_hw, s)
                        v = v_u + guidance_scale * (v - v_u)
                # Euler：σ 递减，dx=v·dσ。fp32 累积更稳（v 在 bf16 下算出）
                x = x + (sigmas[i + 1] - sigmas[i]) * v.float()

        lat0 = self._unpack_latents(x, grid_h, grid_w, C, patch=p)  # [B,C,lat_h,lat_w]
        if was_training:
            self.train()
        if return_latents:
            return lat0
        return self.vae.decode(lat0)
