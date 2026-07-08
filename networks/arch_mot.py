"""MoT (Mixture-of-Transformers) dual-pathway backbone for the 2-3B T2I pilot.

架构对齐 `MoT方案0608.md`「初版架构（求稳版）」：

* 理解路 (und) = frozen Qwen3-VL-2B 文本塔（embed_tokens + 28 层 + final norm）。
* 生成路 (gen) = 逐层复制理解路，权重独立（`*_moe_gen` 后缀）。
* MoT 权重划分与 Cosmos3 `unified_mot.py` 完全一致：
    分流 = LayerNorm + MLP + QKV/输出投影 (+ QK-norm)；
    共享 = 注意力运算本身 + RoPE + 同一条序列。
* 命名沿用 Cosmos3（`q_proj` / `q_proj_moe_gen`、`mlp` / `mlp_moe_gen`、
  `input_layernorm(_moe_gen)`、`post_attention_layernorm(_moe_gen)`、
  `norm` / `norm_moe_gen`），便于后续与 Cosmos / naifu 代码互通。

与 Cosmos3 的实现差异（工程选择，不改变数学）：
* Cosmos3 用 packed sequence（FactoredSequencePack，无 batch 维），我们 v1 用
  batched padded [B, S] + SDPA 显式 bool mask —— CPU 可验证、bucket 训练下每
  batch 图 token 数一致，损耗只有 text padding。GPU 阶段如吞吐不够再迁 packed。
* und 路全程 frozen 且看不到 gen token → und 中间量 requires_grad=False，
  autograd 不会为其建图，naive 联合前向没有额外反传开销。
  （后续可做：采样时缓存 und 各层 KV。）

frozen 侧子模块（MLP/RMSNorm/RoPE）直接复用 HF transformers 的 Qwen3-VL 实现，
保证与官方权重数值一致。
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

# 诊断脚手架（默认关、env MOT_DEBUG_TRACE_FILE 门控）已抽到 training.diagnostics；
# 调用点（_debug_trace/_debug_shape）与 env 语义逐字保持，关闭时对前向逐位无影响。
from training.diagnostics import _debug_shape, _debug_trace


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, n_kv, S, D] -> [B, n_kv*n_rep, S, D]（GQA 展开，HF 同款）。"""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class LoRAResidual(nn.Module):
    """Small LoRA residual branch for a frozen Linear: delta = scale * B(Ax).

    We keep the original Linear module untouched so HF checkpoint loading and
    und->gen tower copying still see the canonical `q_proj.weight` keys.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float | None = None):
        super().__init__()
        rank = int(rank)
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)  # step0: exact no-op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x)) * self.scaling


# --------------------------------------------------------------------------- #
# Mask & position helpers（无状态纯函数已抽到 networks.arch_mot_utils；
# 此处 re-export 保持 `from networks.arch_mot import build_t2i_*` 旧调用路径不变，
# 行为逐位一致。）
# --------------------------------------------------------------------------- #
from networks.arch_mot_utils import (  # noqa: E402
    build_2d_sincos_pos_embed,
    build_t2i_attention_mask,
    build_t2i_position_ids,
)


# --------------------------------------------------------------------------- #
# Timestep 注入（非 AdaLN-Zero：sinusoidal -> MLP -> 加到噪声 token）
# --------------------------------------------------------------------------- #
class TimestepEmbed(nn.Module):
    """sigmas∈[0,1] -> 256 维 sinusoidal -> MLP -> hidden，加性注入噪声 token。

    `timestep_multiplier`：**只缩放 sinusoidal 的输入尺度**（sinusoidal 输入 =
    sigma * multiplier），不影响真实加噪 x_t、loss target、也不影响 sigma 的
    noise-shift（那是另一回事，见 model_mot.get_sigmas）。
    默认 **1.0**（Cosmos3 对齐：等价于 Cosmos3 timestep_scale=0.001，直接喂 sigma）。
    diffusers 习惯刻度是 1000；方案文档把它列为 sweep 项 {1, 10, 1000}，GPU 阶段再扫。
    """

    def __init__(self, hidden_size: int, freq_dim: int = 256, timestep_multiplier: float = 1.0):
        super().__init__()
        assert freq_dim % 2 == 0
        self.freq_dim = freq_dim
        self.timestep_multiplier = timestep_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, sigmas: torch.Tensor) -> torch.Tensor:
        t = sigmas.float() * self.timestep_multiplier  # [B]
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=sigmas.device) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))  # [B, hidden]


# --------------------------------------------------------------------------- #
# MoT attention / decoder layer（Cosmos3 权重划分，batched 实现）
# --------------------------------------------------------------------------- #
class MoTAttention(nn.Module):
    """双路 QKV/输出投影 + 共享联合注意力。und 用 q/k/v/o_proj，gen 用 *_moe_gen。

    `text_kv=True`（B1 adapter 臂）：额外构造 text_k/v_proj_moe_gen + text_k_norm_moe_gen，
    供生成塔 cross-attend 整形后的 T_cond（见 forward_gen_cross）。baseline 联合注意力
    （forward）与之无关、默认不构造。
    """

    def __init__(self, config: Qwen3VLTextConfig, text_kv: bool = False):
        super().__init__()
        self.text_kv = bool(text_kv)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        hidden = config.hidden_size
        bias = config.attention_bias
        eps = config.rms_norm_eps

        # und（frozen Qwen3-VL 原权重落位）
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=bias)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=eps)

        # gen（训练，初始化 = 复制 und）
        self.q_proj_moe_gen = nn.Linear(hidden, self.num_heads * self.head_dim, bias=bias)
        self.k_proj_moe_gen = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj_moe_gen = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, hidden, bias=bias)
        self.q_norm_moe_gen = Qwen3VLTextRMSNorm(self.head_dim, eps=eps)
        self.k_norm_moe_gen = Qwen3VLTextRMSNorm(self.head_dim, eps=eps)

        # B1 adapter 臂：gen 侧 cross-attend T_cond 的新增 text K/V 投影（带 k-norm）。
        # 仅在 text_adapter 启用时构造，对 baseline 零影响。
        if self.text_kv:
            self.text_k_proj_moe_gen = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=bias)
            self.text_v_proj_moe_gen = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=bias)
            self.text_k_norm_moe_gen = Qwen3VLTextRMSNorm(self.head_dim, eps=eps)

    def add_und_lora(self, rank: int, target_modules: tuple[str, ...], alpha: float | None = None):
        """Attach LoRA residuals to frozen und q/k/v/o projections only."""
        targets = set(target_modules)
        if "q_proj" in targets:
            self.q_proj_lora_und = LoRAResidual(self.q_proj.in_features, self.q_proj.out_features, rank, alpha)
        if "k_proj" in targets:
            self.k_proj_lora_und = LoRAResidual(self.k_proj.in_features, self.k_proj.out_features, rank, alpha)
        if "v_proj" in targets:
            self.v_proj_lora_und = LoRAResidual(self.v_proj.in_features, self.v_proj.out_features, rank, alpha)
        if "o_proj" in targets:
            self.o_proj_lora_und = LoRAResidual(self.o_proj.in_features, self.o_proj.out_features, rank, alpha)

    @staticmethod
    def _linear_with_optional_lora(x, lin, lora):
        y = lin(x)
        return y if lora is None else y + lora(x).to(y.dtype)

    def _proj(self, x, lin, n_heads, norm, lora=None):
        """proj -> [B,S,H,D] -> head-dim RMSNorm -> [B,H,S,D]（HF Qwen3 顺序）。"""
        b, s, _ = x.shape
        y = self._linear_with_optional_lora(x, lin, lora)
        return norm(y.view(b, s, n_heads, self.head_dim)).transpose(1, 2)

    def _out_proj_und(self, x):
        return self._linear_with_optional_lora(x, self.o_proj, getattr(self, "o_proj_lora_und", None))

    def forward(
        self,
        h_und: torch.Tensor,  # [B, S_t, hidden]（已过 input_layernorm）
        h_gen: torch.Tensor,  # [B, S_i, hidden]（已过 input_layernorm_moe_gen）
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # cos/sin [B, S, head_dim]，S=S_t+S_i
        attn_mask: torch.Tensor,  # [B, 1, S, S] bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, s_t, _ = h_und.shape
        s_i = h_gen.shape[1]

        def proj(x, lin, n_heads, norm, lora=None):
            # HF Qwen3 顺序：proj -> view [B,S,H,D] -> RMSNorm(head_dim) -> [B,H,S,D]
            y = self._linear_with_optional_lora(x, lin, lora)
            return norm(y.view(bsz, -1, n_heads, self.head_dim)).transpose(1, 2)

        q_und = proj(h_und, self.q_proj, self.num_heads, self.q_norm, getattr(self, "q_proj_lora_und", None))
        k_und = proj(h_und, self.k_proj, self.num_kv_heads, self.k_norm, getattr(self, "k_proj_lora_und", None))
        v_und = self._linear_with_optional_lora(
            h_und, self.v_proj, getattr(self, "v_proj_lora_und", None)
        ).view(bsz, s_t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q_gen = proj(h_gen, self.q_proj_moe_gen, self.num_heads, self.q_norm_moe_gen)
        k_gen = proj(h_gen, self.k_proj_moe_gen, self.num_kv_heads, self.k_norm_moe_gen)
        v_gen = self.v_proj_moe_gen(h_gen).view(bsz, s_i, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 共享：拼成一条序列做联合注意力（und 在前，gen 在后，与 mask/pos 约定一致）
        q = torch.cat([q_und, q_gen], dim=2)  # [B, H, S, D]
        k = torch.cat([k_und, k_gen], dim=2)
        v = torch.cat([v_und, v_gen], dim=2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # unsqueeze_dim=1 广播到 heads

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(bsz, s_t + s_i, self.num_heads * self.head_dim)

        out_und = self._out_proj_und(attn[:, :s_t])
        out_gen = self.o_proj_moe_gen(attn[:, s_t:])
        return out_und, out_gen

    # ----- B1 adapter 臂：und 独立 self-attn + gen cross-attend T_cond ----- #
    def forward_und_self(self, h_und, position_embeddings, attn_mask):
        """und 独立 causal self-attention（adapter 臂下产末层 H_t 用，gen 不参与）。"""
        cos, sin = position_embeddings
        b, s, _ = h_und.shape
        _debug_trace("und_attn_begin", hidden=_debug_shape(h_und), mask=_debug_shape(attn_mask))
        _debug_trace("und_attn_before_q_proj", hidden=_debug_shape(h_und))
        q = self._proj(h_und, self.q_proj, self.num_heads, self.q_norm, getattr(self, "q_proj_lora_und", None))
        _debug_trace("und_attn_after_q_proj", q=_debug_shape(q))
        _debug_trace("und_attn_before_k_proj", hidden=_debug_shape(h_und))
        k = self._proj(h_und, self.k_proj, self.num_kv_heads, self.k_norm, getattr(self, "k_proj_lora_und", None))
        _debug_trace("und_attn_after_k_proj", k=_debug_shape(k))
        _debug_trace("und_attn_before_v_proj", hidden=_debug_shape(h_und))
        v = self._linear_with_optional_lora(
            h_und, self.v_proj, getattr(self, "v_proj_lora_und", None)
        ).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        _debug_trace("und_attn_after_v_proj", v=_debug_shape(v))
        _debug_trace("und_attn_before_rotary", q=_debug_shape(q), k=_debug_shape(k))
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        _debug_trace("und_attn_after_rotary", q=_debug_shape(q), k=_debug_shape(k))
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)
        _debug_trace("und_attn_after_repeat_kv", k=_debug_shape(k), v=_debug_shape(v))
        _debug_trace("und_attn_before_sdpa", q=_debug_shape(q), k=_debug_shape(k), v=_debug_shape(v))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        _debug_trace("und_attn_after_sdpa", attn=_debug_shape(attn))
        attn = attn.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        _debug_trace("und_attn_before_o_proj", attn=_debug_shape(attn))
        out = self._out_proj_und(attn)
        _debug_trace("und_attn_after_o_proj", out=_debug_shape(out))
        return out

    def forward_gen_cross(self, h_gen, t_cond, cos_text, sin_text, cos_gen, sin_gen, attn_mask):
        """gen 合并注意力：query=gen，key/value=[text_k/v_proj_moe_gen(T_cond) ‖ gen]。

        T_cond 的 K 加文本 1D MRoPE，gen 的 Q/K 加生成位置 —— 与 baseline「文本在前、
        生成在 Δ 之后」的相对几何一致。此臂下生成塔不再看逐层 und。
        """
        b, s_i, _ = h_gen.shape
        s_t = t_cond.shape[1]
        q = self._proj(h_gen, self.q_proj_moe_gen, self.num_heads, self.q_norm_moe_gen)
        k_g = self._proj(h_gen, self.k_proj_moe_gen, self.num_kv_heads, self.k_norm_moe_gen)
        v_g = self.v_proj_moe_gen(h_gen).view(b, s_i, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k_t = self._proj(t_cond, self.text_k_proj_moe_gen, self.num_kv_heads, self.text_k_norm_moe_gen)
        v_t = self.text_v_proj_moe_gen(t_cond).view(b, s_t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k_g = apply_rotary_pos_emb(q, k_g, cos_gen, sin_gen)       # gen 用生成位置
        k_t, _ = apply_rotary_pos_emb(k_t, k_t, cos_text, sin_text)  # text K 用文本位置（无对应 q）
        k = torch.cat([k_t, k_g], dim=2)
        v = torch.cat([v_t, v_g], dim=2)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # q:s_i, kv:s_t+s_i
        attn = attn.transpose(1, 2).reshape(b, s_i, self.num_heads * self.head_dim)
        return self.o_proj_moe_gen(attn)

    # ----- G6 采样 KV 缓存（baseline 路径）：und 预计算捕获 K/V + gen-only attend ----- #
    def forward_und_self_capture(self, h_und, position_embeddings, attn_mask):
        """同 forward_und_self（und 自注意力），但额外返回 **post-rotary + post-repeat_kv** 的
        k_und/v_und [B, num_heads, s_t, D]——即 baseline 联合 forward 里 gen 所 attend 的同一份
        und K/V，供采样 loop 外缓存一次。"""
        cos, sin = position_embeddings
        b, s, _ = h_und.shape
        q = self._proj(h_und, self.q_proj, self.num_heads, self.q_norm, getattr(self, "q_proj_lora_und", None))
        k = self._proj(h_und, self.k_proj, self.num_kv_heads, self.k_norm, getattr(self, "k_proj_lora_und", None))
        v = self._linear_with_optional_lora(
            h_und, self.v_proj, getattr(self, "v_proj_lora_und", None)
        ).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        return self._out_proj_und(attn), k, v

    def forward_gen_cached(self, h_gen, k_und, v_und, cos_gen, sin_gen, attn_mask):
        """gen-only 注意力：query=gen，key/value=[缓存 k/v_und ‖ gen]。与 baseline 联合 forward
        的 gen 部分数学等价（k_und/v_und 已按文本位置 rotary + repeat_kv，与联合一致）。"""
        b, s_i, _ = h_gen.shape
        q = self._proj(h_gen, self.q_proj_moe_gen, self.num_heads, self.q_norm_moe_gen)
        k_g = self._proj(h_gen, self.k_proj_moe_gen, self.num_kv_heads, self.k_norm_moe_gen)
        v_g = self.v_proj_moe_gen(h_gen).view(b, s_i, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k_g = apply_rotary_pos_emb(q, k_g, cos_gen, sin_gen)
        k_g = repeat_kv(k_g, self.num_kv_groups)
        v_g = repeat_kv(v_g, self.num_kv_groups)
        k = torch.cat([k_und, k_g], dim=2)
        v = torch.cat([v_und, v_g], dim=2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(b, s_i, self.num_heads * self.head_dim)
        return self.o_proj_moe_gen(attn)


class MoTDecoderLayer(nn.Module):
    """双路 LN/MLP + 共享注意力的 decoder 层（对齐 Cosmos3 MoTDecoderLayer）。

    `sandwich_norm`（方案 v3，Day-1 baseline，默认开）：仅生成支在 attn / MLP
    子层**输出处**各加一个 RMSNorm 再残差相加（`x_gen += RMSNorm(sublayer(...))`）；
    冻结理解侧保持纯 pre-norm 不碰。两个新 RMSNorm 命名带 `_moe_gen` 后缀 → 自动
    判为可训练、不被 und 权重加载、不被生成塔复制，默认 weight=1（非恒等）。定位是
    稳定性保险（Z-Image/CogView 产线在用）；关掉即逐位退回原 pre-norm 实现。
    Cosmos3 / i1 都没有这个无-gate 版本，是按 v3 spec 新写。
    """

    def __init__(self, config: Qwen3VLTextConfig, sandwich_norm: bool = True, text_kv: bool = False):
        super().__init__()
        eps = config.rms_norm_eps
        hidden = config.hidden_size
        self.sandwich_norm = bool(sandwich_norm)
        self.self_attn = MoTAttention(config, text_kv=text_kv)
        self.mlp = Qwen3VLTextMLP(config)
        self.mlp_moe_gen = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(hidden, eps=eps)
        self.input_layernorm_moe_gen = Qwen3VLTextRMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(hidden, eps=eps)
        self.post_attention_layernorm_moe_gen = Qwen3VLTextRMSNorm(hidden, eps=eps)
        # 生成侧 sandwich norm（子层输出处，仅生成支）。默认 weight=1、非恒等。
        if self.sandwich_norm:
            self.attn_out_norm_moe_gen = Qwen3VLTextRMSNorm(hidden, eps=eps)
            self.mlp_out_norm_moe_gen = Qwen3VLTextRMSNorm(hidden, eps=eps)

    def forward(self, *args, mode="joint", **kwargs):
        """FSDP2 分发器：让 B1 adapter 臂的 und/gen 自定义前向也经 layer.__call__ 入口，
        从而触发 FSDP2 per-layer 的 all-gather hook（前向 + 注册反向 all-gather）。
        mode='joint'（默认，baseline 走这条）= 调原 _forward_joint，逐字不变、数值零改动。
        （fix: 此前 _forward_adapter_arm 直接调 forward_und_self/forward_gen_adapter，绕过 hook →
          参数仍是 sharded DTensor → RoPE q*cos 混 Tensor/DTensor 崩；手动 unshard 修不了可训 gen 的反向。）"""
        if mode == "und_self":
            return self.forward_und_self(*args, **kwargs)
        if mode == "gen_adapter":
            return self.forward_gen_adapter(*args, **kwargs)
        return self._forward_joint(*args, **kwargs)

    def _forward_joint(self, h_und, h_gen, position_embeddings, attn_mask):
        a_und, a_gen = self.self_attn(
            self.input_layernorm(h_und),
            self.input_layernorm_moe_gen(h_gen),
            position_embeddings,
            attn_mask,
        )
        h_und = h_und + a_und                                # und 支：纯 pre-norm，不碰
        if self.sandwich_norm:
            a_gen = self.attn_out_norm_moe_gen(a_gen)        # gen 支：输出处 sandwich norm
        h_gen = h_gen + a_gen
        h_und = h_und + self.mlp(self.post_attention_layernorm(h_und))
        m_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(h_gen))
        if self.sandwich_norm:
            m_gen = self.mlp_out_norm_moe_gen(m_gen)
        h_gen = h_gen + m_gen
        return h_und, h_gen

    # ----- B1 adapter 臂的逐层前向（und / gen 解耦）----- #
    def forward_und_self(self, h_und, position_embeddings, attn_mask):
        """und 独立 self-attn + MLP（纯 pre-norm，与 baseline und 支数学一致）。"""
        _debug_trace("und_decoder_before_input_norm", hidden=_debug_shape(h_und))
        h_norm = self.input_layernorm(h_und)
        _debug_trace("und_decoder_after_input_norm", hidden=_debug_shape(h_norm))
        _debug_trace("und_decoder_before_attn", hidden=_debug_shape(h_norm))
        attn_out = self.self_attn.forward_und_self(h_norm, position_embeddings, attn_mask)
        _debug_trace("und_decoder_after_attn", hidden=_debug_shape(attn_out))
        h_und = h_und + attn_out
        _debug_trace("und_decoder_before_post_attn_norm", hidden=_debug_shape(h_und))
        h_norm = self.post_attention_layernorm(h_und)
        _debug_trace("und_decoder_after_post_attn_norm", hidden=_debug_shape(h_norm))
        _debug_trace("und_decoder_before_mlp", hidden=_debug_shape(h_norm))
        mlp_out = self.mlp(h_norm)
        _debug_trace("und_decoder_after_mlp", hidden=_debug_shape(mlp_out))
        h_und = h_und + mlp_out
        return h_und

    def forward_gen_adapter(self, h_gen, t_cond, cos_text, sin_text, cos_gen, sin_gen, attn_mask):
        """gen cross-attend T_cond + MLP（含生成侧 sandwich norm）。t_cond 不过 gen input-norm。"""
        a = self.self_attn.forward_gen_cross(
            self.input_layernorm_moe_gen(h_gen), t_cond, cos_text, sin_text, cos_gen, sin_gen, attn_mask
        )
        if self.sandwich_norm:
            a = self.attn_out_norm_moe_gen(a)
        h_gen = h_gen + a
        m = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(h_gen))
        if self.sandwich_norm:
            m = self.mlp_out_norm_moe_gen(m)
        h_gen = h_gen + m
        return h_gen

    # ----- G6 采样 KV 缓存的逐层前向（baseline；与 forward 数学等价）----- #
    def forward_und_capture(self, h_und, position_embeddings, attn_mask):
        """und 支自注意力 + MLP（同 baseline und 支），额外返回该层 k_und/v_und 供 gen 缓存。"""
        a_und, k_und, v_und = self.self_attn.forward_und_self_capture(
            self.input_layernorm(h_und), position_embeddings, attn_mask)
        h_und = h_und + a_und
        h_und = h_und + self.mlp(self.post_attention_layernorm(h_und))
        return h_und, k_und, v_und

    def forward_gen_cached(self, h_gen, k_und, v_und, cos_gen, sin_gen, attn_mask):
        """gen 支 attend 缓存的 und K/V（含生成侧 sandwich norm；与 baseline gen 支逐位等价）。"""
        a = self.self_attn.forward_gen_cached(
            self.input_layernorm_moe_gen(h_gen), k_und, v_und, cos_gen, sin_gen, attn_mask)
        if self.sandwich_norm:
            a = self.attn_out_norm_moe_gen(a)
        h_gen = h_gen + a
        m = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(h_gen))
        if self.sandwich_norm:
            m = self.mlp_out_norm_moe_gen(m)
        h_gen = h_gen + m
        return h_gen


# --------------------------------------------------------------------------- #
# i1 消融增量的模块（消融项，默认关）
# 共同原则：仅生成侧、zero-init/gate-zero、关时不构造任何参数 → 对 baseline 零影响。
# 排期见「模型架构修改规划」Part B：① text adapter → ② long skip → ③ 去噪声条件
#                                     → ④ 绝对 sinusoidal。
# --------------------------------------------------------------------------- #
class LongSkipFusion(nn.Module):
    """B2 生成侧 long skip 的单条对称连接（gate zero-init → step0 严格等价无 skip）。

    `x_gen += gate · Linear(concat(RMSNorm(x_gen), RMSNorm(skip)))`，仅作用生成 token。
    concat 前各自 RMSNorm 压浅/深层尺度差（比 i1 原版 replace 形更稳）；gate 是
    zero-init 标量参数，初始严格 = 无 skip。i1 原版无 gate/无 zero-init/concat 前不
    分别 norm —— 三处都加固以保 baseline。
    """

    def __init__(self, hidden: int, eps: float):
        super().__init__()
        self.norm_x = Qwen3VLTextRMSNorm(hidden, eps=eps)
        self.norm_skip = Qwen3VLTextRMSNorm(hidden, eps=eps)
        self.proj = nn.Linear(2 * hidden, hidden)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x_gen: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        fused = self.proj(torch.cat([self.norm_x(x_gen), self.norm_skip(skip)], dim=-1))
        return x_gen + self.gate.to(fused.dtype) * fused  # .to：bf16 训练下不被 fp32 gate 上浮


class MultiLayerFusion(nn.Module):
    """Ablation 2：冻结 und 塔末 N 层隐状态的 ELMo 式可学习加权融合（gate zero-init → step0 严格等价末层）。

    输入是 _forward_und 在 no_grad 下收集的末 N 层 hidden（`h_list[i]`，常量，无梯度）与末层
    `h_last`。融合在 `no_grad` 外做（本模块持可训练参数，梯度须流入）：
        elmo  = Σ_i softmax(logits)_i · h_list[i]
        fused = h_last + gate · (elmo − h_last)
    `logits` zero-init（softmax → 均匀权重，**非** one-hot：one-hot 会饱和 softmax 杀死 logit 可学性）；
    `gate` zero-init 标量 → step0 时 `fused == h_last` 逐位等价当前 B1（再过单个 self.norm，与现状一致）。
    梯度：`∂fused/∂gate = (elmo − h_last) ≠ 0` 故 gate 先学；gate≠0 后 logits 也拿到梯度。
    命名带 `_moe_gen`（见 is_trainable_param_name）→ 自动判可训练、不被 und 加载/gen 复制触及。
    与 LongSkipFusion.gate / abs_sincos_alpha_moe_gen 的 zero-init-gate 风格一致。
    """

    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = int(num_layers)
        self.logits = nn.Parameter(torch.zeros(self.num_layers))  # zero-init → softmax 均匀
        self.gate = nn.Parameter(torch.zeros(1))                  # zero-init → step0 == h_last

    def forward(self, h_list: list[torch.Tensor], h_last: torch.Tensor) -> torch.Tensor:
        assert len(h_list) == self.num_layers, \
            f"MultiLayerFusion 期望 {self.num_layers} 层，收到 {len(h_list)}"
        w = torch.softmax(self.logits.float(), dim=0).to(h_last.dtype)  # [num_layers]
        elmo = sum(w[i] * h_list[i] for i in range(self.num_layers))    # Σ_i softmax_i · h_i
        return h_last + self.gate.to(h_last.dtype) * (elmo - h_last)    # gate zero-init → == h_last


class _AdapterBlock(nn.Module):
    """transformer block：RMSNorm + QK-norm + SwiGLU，无 RoPE（H_t 已带 LM 位置信息，v3）。

    `causal=False` 双向（默认，消融 A·双向）；`causal=True` 因果（消融 A·因果）。pre-norm 残差。
    """

    def __init__(self, config: Qwen3VLTextConfig, causal: bool = False):
        super().__init__()
        eps = config.rms_norm_eps
        hidden = config.hidden_size
        self.causal = bool(causal)
        self.head_dim = getattr(config, "head_dim", hidden // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.ln1 = Qwen3VLTextRMSNorm(hidden, eps=eps)
        self.ln2 = Qwen3VLTextRMSNorm(hidden, eps=eps)
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=eps)
        self.mlp = Qwen3VLTextMLP(config)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq, _ = x.shape
        h = self.ln1(x)
        q = self.q_norm(self.q_proj(h).view(bsz, seq, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(h).view(bsz, seq, self.num_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(h).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None or self.causal:
            m = torch.ones(bsz, 1, seq, seq, dtype=torch.bool, device=x.device)
            if self.causal:
                m = m & torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
            if key_padding_mask is not None:
                m = m & key_padding_mask[:, None, None, :].bool()
            m = m | torch.eye(seq, dtype=torch.bool, device=x.device)[None, None]  # 保对角线防全 -inf
            attn_mask = m
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(bsz, seq, self.num_heads * self.head_dim)
        x = x + self.o_proj(out)
        x = x + self.mlp(self.ln2(x))
        return x


class TextAdapter(nn.Module):
    """B1 文本 adapter（上游双向重编码，末层投影 zero-init → 起点 T_cond == H_t）。

    输入 = 冻结 und 末层 hidden H_t [B, L_t, hidden]，输出 T_cond 同形。末层 out_proj
    zero-init，使初始增量为 0（纯增量、起点等价原始末层文本）。约 2 层、宽 hidden。
    与 i1 差异：i1 adapter 默认 1 层、无 zero-init —— zero-init 是 v3 自加的 baseline 保险。
    """

    def __init__(self, config: Qwen3VLTextConfig, num_layers: int = 2, bidirectional: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList(
            _AdapterBlock(config, causal=not bidirectional) for _ in range(num_layers)
        )
        self.out_norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # zero-init：初始 T_cond == H_t（num_layers=0 即中间对照臂）

    def forward(self, h_t: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = h_t
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return h_t + self.out_proj(self.out_norm(x))  # 残差 + zero-init 增量


# --------------------------------------------------------------------------- #
# 完整 T2I transformer
# --------------------------------------------------------------------------- #
class MoTT2ITransformer(nn.Module):
    """frozen Qwen3-VL 文本塔 + 复制生成塔 + 外部接口层（proj_in/out、timestep）。

    输入是 patchify 后的 latent token（patchify 在 model 包装层做，见
    model_mot.py），这里只管 [B, N, patch_dim] <-> v-pred。

    Args:
        config: Qwen3VLTextConfig（真实 2B 或 tiny debug 配置）。
        patch_dim: 每个图 token 的通道数 = vae_latent_channels * patch_size**2。
        time_gap: 文本->生成图的时间索引间隔 Δ（默认 2048，Cosmos3）。
        timestep_multiplier: 见 TimestepEmbed。
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        patch_dim: int,
        time_gap: int = 2048,
        timestep_multiplier: float = 1.0,
        sandwich_norm: bool = True,
        noise_condition: bool = True,
        long_skip: dict | None = None,
        text_adapter: dict | None = None,
        und_lora: dict | None = None,
        abs_sincos: dict | None = None,
    ):
        super().__init__()
        self.config = config
        self.patch_dim = patch_dim
        self.time_gap = time_gap
        self.noise_condition = bool(noise_condition)  # B3：关=彻底删噪声条件
        hidden = config.hidden_size

        self._ta_cfg = dict(text_adapter or {})        # B1：提前解析，决定每层是否建 text K/V
        self.text_adapter_enabled = bool(self._ta_cfg.get("enabled", False))
        self._und_lora_cfg = dict(und_lora or {})      # Ab3：baseline joint path 上的 und attention LoRA
        self.und_lora_enabled = bool(self._und_lora_cfg.get("enabled", False))
        if self.und_lora_enabled and self.text_adapter_enabled:
            raise ValueError(
                "und_lora.enabled=true 只支持 baseline joint path；不能与 text_adapter.enabled=true 同开。"
            )

        self.embed_tokens = nn.Embedding(config.vocab_size, hidden, config.pad_token_id)
        self.layers = nn.ModuleList(
            MoTDecoderLayer(config, sandwich_norm=sandwich_norm, text_kv=self.text_adapter_enabled)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = Qwen3VLTextRMSNorm(hidden, eps=config.rms_norm_eps)  # und final（t2i 不用，留给后续 CE）
        self.norm_moe_gen = Qwen3VLTextRMSNorm(hidden, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)

        self.proj_in = nn.Linear(patch_dim, hidden)
        if self.noise_condition:  # B3 关=彻底删：不构造 time_embed（省参数、避免孤立可训练参数）
            self.time_embed = TimestepEmbed(hidden, timestep_multiplier=timestep_multiplier)
        self.proj_out = nn.Linear(hidden, patch_dim)
        # v-head 零初始化：训练起点输出 0，flow 训练更稳（常规做法）
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        # i1 消融增量（默认关；关时不构造任何参数，对 baseline 逐位零影响）
        self._build_ablation_modules(config, hidden, long_skip, text_adapter, abs_sincos)
        self._build_und_lora_modules(config)

        self.gradient_checkpointing = False

    @staticmethod
    def _normalize_lora_targets(target_modules) -> tuple[str, ...]:
        aliases = {
            "q": "q_proj", "q_proj": "q_proj",
            "k": "k_proj", "k_proj": "k_proj",
            "v": "v_proj", "v_proj": "v_proj",
            "o": "o_proj", "o_proj": "o_proj",
        }
        if target_modules is None:
            raw = ["q", "k", "v", "o"]
        elif isinstance(target_modules, str):
            raw = [t.strip() for t in target_modules.split(",") if t.strip()]
        else:
            raw = list(target_modules)
        out = []
        for t in raw:
            key = str(t)
            if key not in aliases:
                raise ValueError(f"und_lora.target_modules 含未知项 {key!r}，允许 q/k/v/o 或 *_proj")
            out.append(aliases[key])
        return tuple(dict.fromkeys(out))

    def _build_und_lora_modules(self, config):
        """Ab3：只给最后 N 层 frozen und attention q/k/v/o 加 LoRA 旁路。"""
        if not self.und_lora_enabled:
            self.und_lora_layer_indices: set[int] = set()
            self.und_lora_target_modules: tuple[str, ...] = ()
            return
        n_layers = config.num_hidden_layers
        num_layers = int(self._und_lora_cfg.get("num_layers", 8))
        rank = int(self._und_lora_cfg.get("rank", 32))
        alpha = self._und_lora_cfg.get("alpha", rank)
        if not (1 <= num_layers <= n_layers):
            raise ValueError(f"und_lora.num_layers={num_layers} 必须在 [1, {n_layers}]")
        targets = self._normalize_lora_targets(self._und_lora_cfg.get("target_modules", ["q", "k", "v", "o"]))
        if not targets:
            raise ValueError("und_lora.target_modules 不能为空")
        start = n_layers - num_layers
        self.und_lora_layer_indices = set(range(start, n_layers))
        self.und_lora_target_modules = targets
        for idx in range(start, n_layers):
            self.layers[idx].self_attn.add_und_lora(rank=rank, target_modules=targets, alpha=float(alpha))

    def _build_ablation_modules(self, config, hidden, long_skip, text_adapter, abs_sincos):
        eps = config.rms_norm_eps
        n_layers = config.num_hidden_layers

        # B2 long skip（生成侧、对称连接、gate zero-init）
        ls = dict(long_skip or {})
        self.long_skip_enabled = bool(ls.get("enabled", False))
        self.long_skip_layers: dict[int, int] = {}  # 第二半层下标 m -> ModuleList 序号
        self.long_skip_sources: set[int] = set()     # 需缓存激活的第一半层下标
        if self.long_skip_enabled:
            mode = str(ls.get("mode", "A_full"))
            n_pairs = n_layers // 2
            n_outer = max(1, n_pairs // 3) if mode == "B_outer" else n_pairs
            fusions = []
            for m in range(n_layers - n_pairs, n_layers):  # 第二半层
                pair_idx = n_layers - 1 - m                # 配对的第一半层下标（越小越外侧）
                if pair_idx < n_outer:
                    self.long_skip_layers[m] = len(fusions)
                    self.long_skip_sources.add(pair_idx)
                    fusions.append(LongSkipFusion(hidden, eps))
            self.long_skip_moe_gen = nn.ModuleList(fusions)

        # B4 绝对 sinusoidal（仅生成 token、alpha zero-init）
        ab = dict(abs_sincos or {})
        self.abs_sincos_enabled = bool(ab.get("enabled", False))
        if self.abs_sincos_enabled:
            assert hidden % 4 == 0, "abs-sincos 需要 hidden 能被 4 整除"
            self.abs_sincos_alpha_moe_gen = nn.Parameter(torch.zeros(1))

        # B1 文本 adapter（attention 臂已接入，见 _forward_adapter_arm）。
        # num_layers=0 → adapter 恒等（中间对照臂「固定末层文本、无 adapter」）；
        # bidirectional 控双向/因果消融臂。enabled 已在 __init__ 早处解析（决定每层 text K/V）。
        if self.text_adapter_enabled:
            self.text_adapter_moe_gen = TextAdapter(
                config,
                num_layers=int(self._ta_cfg.get("num_layers", 2)),
                bidirectional=bool(self._ta_cfg.get("bidirectional", True)),
            )

        # Ablation 2 多层特征融合（ELMo 式，融合 und 末 N 层 → 单 self.norm → 喂 adapter）。
        # 仅在 text_adapter 也开时构造（它喂 adapter）；enabled 但 adapter 关 → 报错。
        mlf = dict((self._ta_cfg.get("multi_layer_fusion") or {}))
        self.multi_layer_fusion_enabled = bool(mlf.get("enabled", False))
        self.multi_layer_fusion_num = int(mlf.get("num_layers", 12))
        if self.multi_layer_fusion_enabled:
            if not self.text_adapter_enabled:
                raise ValueError(
                    "text_adapter.multi_layer_fusion.enabled 需要 text_adapter.enabled=true"
                    "（融合输出喂 B1 文本 adapter）。"
                )
            if not (1 <= self.multi_layer_fusion_num <= n_layers):
                raise ValueError(
                    f"multi_layer_fusion.num_layers={self.multi_layer_fusion_num} 必须在 [1, {n_layers}]"
                )
            self.multi_layer_fusion_moe_gen = MultiLayerFusion(self.multi_layer_fusion_num)

    # ----- 初始化 / 冻结 ----------------------------------------------------- #
    @torch.no_grad()
    def init_generation_tower_from_und(self):
        """逐层把 und 权重复制到 gen 路（BAGEL/Cosmos3 的常规初始化）。"""
        for layer in self.layers:
            attn = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                gen_mod = getattr(attn, f"{name}_moe_gen")
                und_mod = getattr(attn, name)
                gen_mod.load_state_dict(und_mod.state_dict())
            layer.mlp_moe_gen.load_state_dict(layer.mlp.state_dict())
            layer.input_layernorm_moe_gen.load_state_dict(layer.input_layernorm.state_dict())
            layer.post_attention_layernorm_moe_gen.load_state_dict(layer.post_attention_layernorm.state_dict())
        self.norm_moe_gen.load_state_dict(self.norm.state_dict())

    def freeze_understanding(self):
        """冻结理解路 + embedding；只留 gen 路与外部接口层可训。"""
        for name, p in self.named_parameters():
            p.requires_grad = self.is_trainable_param_name(name)

    @staticmethod
    def is_trainable_param_name(name: str) -> bool:
        if "lora_und" in name:
            return True
        if "_moe_gen" in name:
            return True
        return name.startswith(("proj_in", "proj_out", "time_embed"))

    # ----- forward ----------------------------------------------------------- #
    def forward(
        self,
        input_ids: torch.Tensor,  # [B, S_t]
        text_mask: torch.Tensor,  # [B, S_t] bool
        img_tokens: torch.Tensor,  # [B, N, patch_dim] 噪声 latent（已 patchify）
        grid_hw: tuple[int, int],  # patchify 后网格 (H', W')，N = H'*W'
        sigmas: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """返回 v-pred，[B, N, patch_dim]。"""
        grid_h, grid_w = grid_hw
        assert img_tokens.shape[1] == grid_h * grid_w, \
            f"img tokens {img_tokens.shape[1]} != grid {grid_h}x{grid_w}"

        if self.text_adapter_enabled:
            return self._forward_adapter_arm(input_ids, text_mask, img_tokens, grid_hw, sigmas)

        h_und = self.embed_tokens(input_ids)
        if self.noise_condition:
            t_emb = self.time_embed(sigmas)  # [B, hidden]
            h_gen = self.proj_in(img_tokens) + t_emb[:, None, :]  # 加性注入噪声 token（非 AdaLN）
        else:
            h_gen = self.proj_in(img_tokens)  # B3 消融：彻底删噪声条件（time_embed 此时不参与）
        if self.abs_sincos_enabled:  # B4：仅生成 token、alpha zero-init（初始增量为 0）
            sincos = build_2d_sincos_pos_embed(
                grid_h, grid_w, self.config.hidden_size, device=h_gen.device, dtype=h_gen.dtype
            )
            h_gen = h_gen + self.abs_sincos_alpha_moe_gen.to(h_gen.dtype) * sincos.unsqueeze(0)

        attn_mask = build_t2i_attention_mask(text_mask, img_tokens.shape[1])
        position_ids = build_t2i_position_ids(text_mask, grid_h, grid_w, self.time_gap)
        # HF rotary: position_ids [3, B, S] -> cos/sin [B, S, head_dim]（interleaved mrope 已内置）
        cos, sin = self.rotary_emb(h_gen, position_ids)
        position_embeddings = (cos, sin)

        skip_cache: dict[int, torch.Tensor] = {}
        for idx, layer in enumerate(self.layers):
            # B2 long skip：在生成支进入本层前融合对称浅层激活（gate zero-init→step0 等价）
            if self.long_skip_enabled and idx in self.long_skip_layers:
                fusion = self.long_skip_moe_gen[self.long_skip_layers[idx]]
                h_gen = fusion(h_gen, skip_cache[len(self.layers) - 1 - idx])
            if self.gradient_checkpointing and self.training:
                h_und, h_gen = torch.utils.checkpoint.checkpoint(
                    layer, h_und, h_gen, position_embeddings, attn_mask, use_reentrant=False
                )
            else:
                h_und, h_gen = layer(h_und, h_gen, position_embeddings, attn_mask)
            if self.long_skip_enabled and idx in self.long_skip_sources:
                skip_cache[idx] = h_gen

        return self.proj_out(self.norm_moe_gen(h_gen))

    # ----- B1 adapter 臂前向（und 独立跑出 H_t → adapter → gen cross-attend T_cond）----- #
    @torch.no_grad()
    def _forward_und(self, input_ids, text_mask):
        """frozen und 塔独立 causal 前向，默认返回末层 H_t = final-norm(hidden)（对齐 HF last_hidden_state）。

        Ablation 2 多层融合开启时返回 `(h_last, h_list)`（均为 final norm 前的 no-grad 常量），
        由 `_forward_adapter_arm` 在梯度区做 fusion + final norm，这样 fusion 的 gate/logits 可训练。

        und 全程冻结（embed/q/k/v/o/norm/mlp 均无 `_moe_gen` 后缀、requires_grad=False），故整段
        `no_grad`：不保留任何中间激活、梯度在此截断（H_t 作 adapter 的常量输入，梯度只进可训练
        adapter / text K/V）。比逐层 checkpoint 更彻底（零激活、零重算），消除 B1 臂 28 层 und 激活峰值。
        """
        _debug_trace(
            "und_begin",
            batch=input_ids.shape[0],
            seq=input_ids.shape[1],
            fusion=int(getattr(self, "multi_layer_fusion_enabled", False)),
        )
        h = self.embed_tokens(input_ids)
        _debug_trace("und_after_embed", hidden=_debug_shape(h), dtype=h.dtype, requires_grad=h.requires_grad)
        b, s_t = text_mask.shape
        device = text_mask.device
        valid = text_mask.bool()
        causal = torch.tril(torch.ones(s_t, s_t, dtype=torch.bool, device=device))
        mask = (causal.unsqueeze(0) & valid.unsqueeze(1)) | torch.eye(s_t, dtype=torch.bool, device=device).unsqueeze(0)
        mask = mask.unsqueeze(1)  # [B,1,s_t,s_t]
        pos = torch.arange(s_t, device=device)[None].expand(b, s_t)[None].expand(3, b, s_t).contiguous()  # t=h=w
        cos, sin = self.rotary_emb(h, pos)
        # 仅在 trace 开启（MOT_DEBUG_TRACE_FILE 已设）时才写 per-layer env：os.environ 改写走进程级
        # environ 锁（putenv/unsetenv），在多线程 dataline 下与底层库 getenv 争锁、扰动每层 CPU 时序
        # → 16 卡 FSDP all_gather 发起时序错位 → 间歇 hang。关 trace 时零 env 访问，与 main 逐位等价。
        _trace_on = os.environ.get("MOT_DEBUG_TRACE_FILE") is not None
        # Ablation 2：开启融合时收集末 N 层（含本层输出）的 hidden（no_grad 常量），交调用方融合+norm。
        if getattr(self, "multi_layer_fusion_enabled", False):
            n_total = len(self.layers)
            collect_from = n_total - self.multi_layer_fusion_num  # 末 N 层的首个层下标
            h_list: list[torch.Tensor] = []
            _debug_trace(
                "und_collect_config",
                layers=n_total,
                collect_from=collect_from,
                collect_num=self.multi_layer_fusion_num,
            )
            for idx, layer in enumerate(self.layers):
                if _trace_on:
                    os.environ["MOT_DEBUG_TRACE_LAYER"] = str(idx)
                try:
                    _debug_trace("und_layer_begin", layer=idx, collect=int(idx >= collect_from), hidden=_debug_shape(h))
                    h = layer(h, (cos, sin), mask, mode="und_self")   # 经 __call__ → 触发 FSDP2 all-gather
                    _debug_trace("und_layer_end", layer=idx, collect=int(idx >= collect_from), hidden=_debug_shape(h))
                finally:
                    if _trace_on:
                        os.environ.pop("MOT_DEBUG_TRACE_LAYER", None)
                if idx >= collect_from:
                    h_list.append(h)
            _debug_trace("und_return_h_list", hidden=_debug_shape(h), collected=len(h_list))
            return h, h_list  # h == 末层 hidden（norm 前）；融合+norm 在 _forward_adapter_arm（梯度区）
        for idx, layer in enumerate(self.layers):
            if _trace_on:
                os.environ["MOT_DEBUG_TRACE_LAYER"] = str(idx)
            try:
                _debug_trace("und_layer_begin", layer=idx, collect=0, hidden=_debug_shape(h))
                h = layer(h, (cos, sin), mask, mode="und_self")   # 经 __call__ → 触发 FSDP2 all-gather
                _debug_trace("und_layer_end", layer=idx, collect=0, hidden=_debug_shape(h))
            finally:
                if _trace_on:
                    os.environ.pop("MOT_DEBUG_TRACE_LAYER", None)
        _debug_trace("und_before_final_norm", hidden=_debug_shape(h))
        h = self.norm(h)
        _debug_trace("und_after_final_norm", hidden=_debug_shape(h))
        return h

    def _forward_adapter_arm(self, input_ids, text_mask, img_tokens, grid_hw, sigmas):
        grid_h, grid_w = grid_hw
        b, s_t = text_mask.shape
        s_i = img_tokens.shape[1]
        device = img_tokens.device

        _debug_trace(
            "adapter_arm_begin",
            batch=b,
            text=s_t,
            image=s_i,
            fusion=int(getattr(self, "multi_layer_fusion_enabled", False)),
        )
        if getattr(self, "multi_layer_fusion_enabled", False):
            # Ablation 2：und 在 no_grad 下吐末 N 层 hidden（常量），融合+final-norm 在此（梯度区）
            # 做，使 logits/gate 拿到梯度；gate zero-init → fused == h_last → self.norm 后逐位等价 B1。
            _debug_trace("adapter_before_forward_und")
            h_last, h_list = self._forward_und(input_ids, text_mask)
            _debug_trace("adapter_after_forward_und", hidden=_debug_shape(h_last), collected=len(h_list))
            _debug_trace("adapter_before_multi_layer_fusion", hidden=_debug_shape(h_last), collected=len(h_list))
            fused = self.multi_layer_fusion_moe_gen(h_list, h_last)
            _debug_trace("adapter_after_multi_layer_fusion", hidden=_debug_shape(fused))
            h_t = self.norm(fused)                                          # 单个 und final norm（与现状一致）
            _debug_trace("adapter_after_final_norm", hidden=_debug_shape(h_t))
        else:
            _debug_trace("adapter_before_forward_und")
            h_t = self._forward_und(input_ids, text_mask)                   # [B, s_t, hidden]
            _debug_trace("adapter_after_forward_und", hidden=_debug_shape(h_t), collected=0)
        _debug_trace("adapter_before_text_adapter", hidden=_debug_shape(h_t))
        t_cond = self.text_adapter_moe_gen(h_t, key_padding_mask=text_mask)  # 整形（num_layers=0 即恒等）
        _debug_trace("adapter_after_text_adapter", hidden=_debug_shape(t_cond))

        if self.noise_condition:
            h_gen = self.proj_in(img_tokens) + self.time_embed(sigmas)[:, None, :]
        else:
            h_gen = self.proj_in(img_tokens)
        _debug_trace("adapter_after_gen_embed", hidden=_debug_shape(h_gen), noise_condition=int(self.noise_condition))
        if self.abs_sincos_enabled:
            sincos = build_2d_sincos_pos_embed(
                grid_h, grid_w, self.config.hidden_size, device=h_gen.device, dtype=h_gen.dtype
            )
            h_gen = h_gen + self.abs_sincos_alpha_moe_gen.to(h_gen.dtype) * sincos.unsqueeze(0)

        # 位置复用 build_t2i_position_ids（text 1D + gen t=text_len+Δ），拆 text/gen 段
        pos = build_t2i_position_ids(text_mask, grid_h, grid_w, self.time_gap)
        cos, sin = self.rotary_emb(h_gen, pos)
        cos_text, sin_text = cos[:, :s_t], sin[:, :s_t]
        cos_gen, sin_gen = cos[:, s_t:], sin[:, s_t:]

        # gen cross mask：gen query 看 [有效 text ; 全 gen]
        valid = text_mask.bool()
        gmask = torch.zeros(b, 1, s_i, s_t + s_i, dtype=torch.bool, device=device)
        gmask[:, :, :, :s_t] = valid[:, None, None, :].expand(b, 1, s_i, s_t)
        gmask[:, :, :, s_t:] = True

        skip_cache: dict[int, torch.Tensor] = {}
        for idx, layer in enumerate(self.layers):
            if self.long_skip_enabled and idx in self.long_skip_layers:
                _debug_trace("gen_adapter_long_skip_begin", layer=idx, hidden=_debug_shape(h_gen))
                h_gen = self.long_skip_moe_gen[self.long_skip_layers[idx]](
                    h_gen, skip_cache[len(self.layers) - 1 - idx]
                )
                _debug_trace("gen_adapter_long_skip_end", layer=idx, hidden=_debug_shape(h_gen))
            # 可训练生成臂：与 baseline forward 一致，开 gradient_checkpointing 时逐层 checkpoint
            # （输入 h_gen/t_cond 均 require grad，不触发 "no input requires grad" 警告）
            _debug_trace(
                "gen_adapter_layer_begin",
                layer=idx,
                checkpoint=int(self.gradient_checkpointing and self.training),
                hidden=_debug_shape(h_gen),
            )
            if self.gradient_checkpointing and self.training:
                h_gen = torch.utils.checkpoint.checkpoint(
                    layer, h_gen, t_cond, cos_text, sin_text, cos_gen, sin_gen, gmask,
                    mode="gen_adapter", use_reentrant=False,   # 经 __call__ → FSDP2 前向+反向 all-gather
                )
            else:
                h_gen = layer(h_gen, t_cond, cos_text, sin_text, cos_gen, sin_gen, gmask, mode="gen_adapter")
            _debug_trace("gen_adapter_layer_end", layer=idx, hidden=_debug_shape(h_gen))
            if self.long_skip_enabled and idx in self.long_skip_sources:
                skip_cache[idx] = h_gen
        _debug_trace("adapter_before_proj_out", hidden=_debug_shape(h_gen))
        out = self.proj_out(self.norm_moe_gen(h_gen))
        _debug_trace("adapter_after_proj_out", hidden=_debug_shape(out))
        return out

    # ----- G6 采样 KV 缓存（baseline 路径；frozen und 跨步不变，loop 外算一次）----- #
    @torch.no_grad()
    def precompute_und_cache(self, input_ids, text_mask, grid_hw):
        """跑一次 frozen und 塔，缓存每层 (k_und, v_und)（post-rotary/post-repeat_kv）。
        返回 (kv_cache[L], (cos_gen, sin_gen), gen_mask)；位置/mask 与 baseline 联合 forward 一致。"""
        grid_h, grid_w = grid_hw
        b, s_t = text_mask.shape
        device = input_ids.device
        s_i = grid_h * grid_w
        pos = build_t2i_position_ids(text_mask, grid_h, grid_w, self.time_gap)  # [3,B,S]
        h_und = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(h_und, pos)
        cos_und, sin_und = cos[:, :s_t], sin[:, :s_t]
        cos_gen, sin_gen = cos[:, s_t:], sin[:, s_t:]
        valid = text_mask.bool()
        causal = torch.tril(torch.ones(s_t, s_t, dtype=torch.bool, device=device))
        und_mask = ((causal.unsqueeze(0) & valid.unsqueeze(1))
                    | torch.eye(s_t, dtype=torch.bool, device=device).unsqueeze(0)).unsqueeze(1)
        gen_mask = torch.zeros(b, 1, s_i, s_t + s_i, dtype=torch.bool, device=device)
        gen_mask[:, :, :, :s_t] = valid[:, None, None, :].expand(b, 1, s_i, s_t)
        gen_mask[:, :, :, s_t:] = True
        kv_cache = []
        for layer in self.layers:
            h_und, k_und, v_und = layer.forward_und_capture(h_und, (cos_und, sin_und), und_mask)
            kv_cache.append((k_und, v_und))
        return kv_cache, (cos_gen, sin_gen), gen_mask

    def forward_gen_cached(self, img_tokens, sigmas, kv_cache, pos_gen, gen_mask, grid_hw):
        """用缓存的 und K/V 只跑 gen 塔，返回 v-pred。与 baseline forward 的 gen 路逐位等价
        （含 noise_condition / abs_sincos / long_skip 等 gen 侧逻辑）。"""
        grid_h, grid_w = grid_hw
        cos_gen, sin_gen = pos_gen
        if self.noise_condition:
            h_gen = self.proj_in(img_tokens) + self.time_embed(sigmas)[:, None, :]
        else:
            h_gen = self.proj_in(img_tokens)
        if self.abs_sincos_enabled:
            sincos = build_2d_sincos_pos_embed(
                grid_h, grid_w, self.config.hidden_size, device=h_gen.device, dtype=h_gen.dtype)
            h_gen = h_gen + self.abs_sincos_alpha_moe_gen.to(h_gen.dtype) * sincos.unsqueeze(0)
        skip_cache: dict[int, torch.Tensor] = {}
        for idx, layer in enumerate(self.layers):
            if self.long_skip_enabled and idx in self.long_skip_layers:
                h_gen = self.long_skip_moe_gen[self.long_skip_layers[idx]](
                    h_gen, skip_cache[len(self.layers) - 1 - idx])
            k_und, v_und = kv_cache[idx]
            h_gen = layer.forward_gen_cached(h_gen, k_und, v_und, cos_gen, sin_gen, gen_mask)
            if self.long_skip_enabled and idx in self.long_skip_sources:
                skip_cache[idx] = h_gen
        return self.proj_out(self.norm_moe_gen(h_gen))

    # ----- 工具 ---------------------------------------------------------------- #
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def param_count(self) -> dict:
        total = trainable = 0
        for p in self.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
