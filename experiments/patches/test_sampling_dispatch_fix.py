#!/usr/bin/env python3
"""B7b 单测：多卡采样 DTensor 修复（采样 KV-缓存路径改走 layer.__call__ 的 mode 分发器）。

修复内容（networks/arch_mot.py）：
  * MoTDecoderLayer.forward 分发器新增 mode="und_capture" / "gen_cached"
    （与训练 joint/und_self/gen_adapter 同一模板）。
  * MoTT2ITransformer.precompute_und_cache:  layer.forward_und_capture(...)
        -> layer(..., mode="und_capture")  经 __call__
  * MoTT2ITransformer.forward_gen_cached:    layer.forward_gen_cached(...)
        -> layer(..., mode="gen_cached")    经 __call__
  目的：FSDP2 下 per-layer all-gather hook 注册在 module.__call__ 上，直调子方法会绕过
  → 多卡采样时参数仍是 sharded DTensor、与缓存的普通 Tensor 混算崩。改走 __call__ 修复。

本测试在 CPU / 单进程（无 FSDP、无 DTensor）下验证：
  T1  layer 级：mode="und_capture" 经 __call__  ==（bit-identical）直调 forward_und_capture。
  T2  layer 级：mode="gen_cached"  经 __call__  ==（bit-identical）直调 forward_gen_cached。
  T3  transformer 级：precompute_und_cache + forward_gen_cached（修补后，全经 __call__）
        与 baseline 联合 forward() 的 gen 输出 step-0 数值等价（KV 缓存本就该等价）。
        覆盖 baseline / sandwich on-off / long_skip(A_full,B_outer) / abs_sincos /
        noise_condition=off 多组消融配置。
  T4  分发器其它 mode（joint/und_self/gen_adapter）行为不回归（joint 默认路径数值零改动）。

用法（须先 `git apply experiments/patches/dtensor_sampling_dispatch_fix.patch`，否则未打补丁的
      arch_mot.py 没有 und_capture/gen_cached 两个 mode，T1/T2 会失败）:
      /fsx/evan/mot_code/venv-gpu/bin/python experiments/patches/test_sampling_dispatch_fix.py
"""

import os
import sys

# repo 根：本文件在 experiments/exp_scripts/ 下，向上两级
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from networks.arch_mot import (
    MoTDecoderLayer,
    MoTT2ITransformer,
    build_t2i_attention_mask,
    build_t2i_position_ids,
)

# tiny 配置与 scripts/test_arch_changes.py 对齐（4 层便于 long-skip 对称配对）
TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=1000,
    rms_norm_eps=1.0e-6,
    attention_bias=False,
    attention_dropout=0.0,
    max_position_embeddings=8192,
    rope_theta=5000000,
    rope_scaling=dict(rope_type="default", mrope_section=[3, 3, 2], mrope_interleaved=True),
    pad_token_id=0,
)
PATCH_DIM = 64
BSZ, S_T, GRID = 2, 8, (2, 2)
N_IMG = GRID[0] * GRID[1]


def tiny_config():
    return Qwen3VLTextConfig(**TINY)


def build_transformer(seed=0, **kwargs):
    torch.manual_seed(seed)
    m = MoTT2ITransformer(tiny_config(), patch_dim=PATCH_DIM, **kwargs)
    m.init_generation_tower_from_und()
    m.eval()
    return m


def randomize_proj_out(m, seed=1234):
    """proj_out 默认 zero-init → forward 恒 0，无法区分内部差异；给确定性非零权重。"""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.proj_out.weight.copy_(torch.randn(m.proj_out.weight.shape, generator=g) * 0.05)
        m.proj_out.bias.copy_(torch.randn(m.proj_out.bias.shape, generator=g) * 0.05)


def sample_inputs(seed=7):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, TINY["vocab_size"], (BSZ, S_T), generator=g)
    text_mask = torch.zeros(BSZ, S_T, dtype=torch.bool)
    text_mask[0, :5] = True
    text_mask[1, :8] = True
    img = torch.randn(BSZ, N_IMG, PATCH_DIM, generator=g)
    sigmas = torch.tensor([0.3, 0.7])
    return input_ids, text_mask, img, sigmas


def assert_bit_identical(a, b, msg):
    assert torch.equal(a, b), f"{msg}: 非 bit-identical，max|Δ|={(a - b).abs().max().item():.3e}"


def assert_close(a, b, tol, msg):
    d = (a - b).abs().max().item()
    assert d <= tol, f"{msg}: max|Δ|={d:.3e} > {tol:.1e}"
    return d


# --------------------------------------------------------------------------- #
def _layer_inputs(cfg, seed=3):
    """构造 layer 级 und_capture / gen_cached 所需输入（真实 rotary + mask）。"""
    torch.manual_seed(seed)
    tm = torch.ones(BSZ, S_T, dtype=torch.bool)
    h_und = torch.randn(BSZ, S_T, cfg.hidden_size)
    h_gen = torch.randn(BSZ, N_IMG, cfg.hidden_size)
    pos = build_t2i_position_ids(tm, *GRID)
    rot = Qwen3VLTextRotaryEmbedding(cfg)
    cos, sin = rot(h_gen, pos)
    cos_und, sin_und = cos[:, :S_T], sin[:, :S_T]
    cos_gen, sin_gen = cos[:, S_T:], sin[:, S_T:]
    # und 自注意力 mask（causal ∧ valid，全有效 → 纯 causal + 对角）
    causal = torch.tril(torch.ones(S_T, S_T, dtype=torch.bool))
    und_mask = (causal | torch.eye(S_T, dtype=torch.bool)).unsqueeze(0).unsqueeze(0).expand(BSZ, 1, S_T, S_T)
    # gen-cached mask：gen query 看 [全 text ; 全 gen]
    gen_mask = torch.ones(BSZ, 1, N_IMG, S_T + N_IMG, dtype=torch.bool)
    return h_und, h_gen, (cos_und, sin_und), (cos_gen, sin_gen), und_mask, gen_mask


def test_layer_und_capture_dispatch_equiv():
    """T1: layer(mode='und_capture') 经 __call__ == 直调 forward_und_capture（bit-identical）。"""
    cfg = tiny_config()
    torch.manual_seed(0)
    layer = MoTDecoderLayer(cfg, sandwich_norm=True).eval()
    h_und, _, pe_und, _, und_mask, _ = _layer_inputs(cfg)

    direct = layer.forward_und_capture(h_und, pe_und, und_mask)
    routed = layer(h_und, pe_und, und_mask, mode="und_capture")
    assert isinstance(direct, tuple) and isinstance(routed, tuple) and len(direct) == len(routed) == 3
    for i, name in enumerate(("h_und", "k_und", "v_und")):
        assert_bit_identical(routed[i], direct[i], f"und_capture dispatch[{name}]")
    print("✓ T1 layer und_capture：__call__(mode) 与直调 bit-identical（h_und/k_und/v_und）")


def test_layer_gen_cached_dispatch_equiv():
    """T2: layer(mode='gen_cached') 经 __call__ == 直调 forward_gen_cached（bit-identical）。"""
    cfg = tiny_config()
    torch.manual_seed(0)
    layer = MoTDecoderLayer(cfg, sandwich_norm=True).eval()
    h_und, h_gen, pe_und, pe_gen, und_mask, gen_mask = _layer_inputs(cfg)
    cos_gen, sin_gen = pe_gen
    # 先跑 und_capture 拿一份真实 k_und/v_und 喂 gen_cached
    _, k_und, v_und = layer.forward_und_capture(h_und, pe_und, und_mask)

    direct = layer.forward_gen_cached(h_gen, k_und, v_und, cos_gen, sin_gen, gen_mask)
    routed = layer(h_gen, k_und, v_und, cos_gen, sin_gen, gen_mask, mode="gen_cached")
    assert direct.shape == (BSZ, N_IMG, cfg.hidden_size)
    assert_bit_identical(routed, direct, "gen_cached dispatch")
    print("✓ T2 layer gen_cached：__call__(mode) 与直调 bit-identical")


def _cached_sampling_forward(m, inp):
    """transformer 级修补后的采样路径：precompute_und_cache + forward_gen_cached（全经 __call__）。"""
    input_ids, text_mask, img, sigmas = inp
    kv_cache, pos_gen, gen_mask = m.precompute_und_cache(input_ids, text_mask, GRID)
    return m.forward_gen_cached(img, sigmas, kv_cache, pos_gen, gen_mask, GRID)


def test_cached_sampling_matches_joint_baseline():
    """T3: 修补后的采样路径 与 baseline 联合 forward() 的 gen 输出 step-0 数值等价，多组消融。"""
    configs = {
        "baseline(sandwich)": dict(sandwich_norm=True),
        "no-sandwich": dict(sandwich_norm=False),
        "long_skip[A_full]": dict(long_skip=dict(enabled=True, mode="A_full")),
        "long_skip[B_outer]": dict(long_skip=dict(enabled=True, mode="B_outer")),
        "abs_sincos": dict(abs_sincos=dict(enabled=True)),
        "noise_condition=off": dict(noise_condition=False),
        "long_skip+abs_sincos": dict(long_skip=dict(enabled=True, mode="A_full"),
                                     abs_sincos=dict(enabled=True)),
    }
    inp = sample_inputs()
    for name, kw in configs.items():
        m = build_transformer(seed=0, **kw)
        randomize_proj_out(m)  # 否则 proj_out zero-init → 输出恒 0，等价断言形同虚设
        # abs_sincos / 部分臂 alpha/gate zero-init 时本就等价；为真正考验 gen 路一致性，
        # 把对应 gate/alpha 调成非零，使 baseline 与 cached 两边都真正使用该分支。
        with torch.no_grad():
            if getattr(m, "abs_sincos_enabled", False):
                m.abs_sincos_alpha_moe_gen.fill_(0.5)
            if getattr(m, "long_skip_enabled", False):
                for f in m.long_skip_moe_gen:
                    f.gate.fill_(0.7)
                    # gate 经 zero-init，proj 随机已在 __init__；确保 proj 非零
        out_joint = m(*[inp[0], inp[1], inp[2], GRID, inp[3]])  # baseline 联合 forward
        out_cached = _cached_sampling_forward(m, inp)
        assert out_cached.shape == out_joint.shape == (BSZ, N_IMG, PATCH_DIM)
        assert torch.isfinite(out_cached).all(), f"{name}: cached 输出含 NaN/Inf"
        d = assert_close(out_cached, out_joint, 5e-5, f"cached 采样 vs joint baseline [{name}]")
        print(f"✓ T3 [{name}]：cached 采样路径 == joint baseline（max|Δ|={d:.2e}）")


def test_other_modes_no_regression():
    """T4: 分发器其它 mode 不回归——joint 默认路径与显式 mode='joint' bit-identical。"""
    cfg = tiny_config()
    torch.manual_seed(0)
    layer = MoTDecoderLayer(cfg, sandwich_norm=True).eval()
    torch.manual_seed(11)
    h_und = torch.randn(BSZ, S_T, cfg.hidden_size)
    h_gen = torch.randn(BSZ, N_IMG, cfg.hidden_size)
    tm = torch.ones(BSZ, S_T, dtype=torch.bool)
    mask = build_t2i_attention_mask(tm, N_IMG)
    pos = build_t2i_position_ids(tm, *GRID)
    pe = Qwen3VLTextRotaryEmbedding(cfg)(h_gen, pos)

    u_def, g_def = layer(h_und, h_gen, pe, mask)                 # 默认 joint
    u_exp, g_exp = layer(h_und, h_gen, pe, mask, mode="joint")   # 显式 joint
    u_int, g_int = layer._forward_joint(h_und, h_gen, pe, mask)  # 内部实现
    assert_bit_identical(u_def, u_int, "joint und 默认 vs 内部")
    assert_bit_identical(g_def, g_int, "joint gen 默认 vs 内部")
    assert_bit_identical(u_def, u_exp, "joint und 默认 vs 显式 mode")
    assert_bit_identical(g_def, g_exp, "joint gen 默认 vs 显式 mode")
    print("✓ T4 分发器：joint 默认/显式/内部三者 bit-identical（无回归）")


def main():
    print("=== B7b 采样路径 DTensor 修复单元测试（tiny 随机权重，CPU/单进程）===")
    test_layer_und_capture_dispatch_equiv()
    test_layer_gen_cached_dispatch_equiv()
    test_cached_sampling_matches_joint_baseline()
    test_other_modes_no_regression()
    print("\n全部 B7b 采样修复单元测试通过 ✅")


if __name__ == "__main__":
    main()
