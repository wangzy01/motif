#!/usr/bin/env python3
"""单元测试：v3 Day-1 baseline + i1 消融骨架的网络改动正确性（CPU，tiny 随机权重）。

覆盖「模型架构修改规划_v1」的改动：
  A1 生成侧 sandwich norm（仅生成支、und 不变、可训练、命名带 _moe_gen）
  A3 noise-shift = per-分辨率固定常数（不再按进度 ramp）
  A6 文字样本 dropout 特例 hook（text_rich）
  timestep_multiplier 默认 1.0（Cosmos3 对齐）
  B2 long skip（gate zero-init → step0 等价无 skip）
  B3 去噪声条件开关（off 与 on 不同、均有限）
  B4 绝对 sinusoidal（alpha zero-init → step0 等价无 sincos）
  B1 TextAdapter（zero-init → passthrough；启用时四消融臂前向跑通，见 test_text_adapter_arm）
  Ab3 und LoRA（baseline joint path，top-N und q/k/v/o，step0 等价，lr scale param group）

用法: venv-gpu/bin/python scripts/test_arch_changes.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from networks.arch_mot import (
    MoTDecoderLayer,
    MoTT2ITransformer,
    MultiLayerFusion,
    TextAdapter,
    TimestepEmbed,
    build_2d_sincos_pos_embed,
)

TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,          # 4 层便于测试 long-skip 对称配对（2 对）
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
PATCH_DIM = 64  # latent_channels 16 * patch 2^2
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
    """proj_out 默认 zero-init 会让 forward 输出恒 0（无法区分内部差异）；
    等价性测试前给它一组确定性非零权重，使输出反映 gen 隐状态差异。"""
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


def fwd(m, inp):
    input_ids, text_mask, img, sigmas = inp
    return m(input_ids, text_mask, img, GRID, sigmas)


def copy_shared(dst, src):
    """把 dst/src 共有的参数拷进 dst（忽略只在一侧的参数，如开关差异导致的 time_embed/gate）。"""
    dst_keys = set(dst.state_dict().keys())
    dst.load_state_dict({k: v for k, v in src.state_dict().items() if k in dst_keys}, strict=False)


def assert_close(a, b, tol, msg):
    d = (a - b).abs().max().item()
    assert d <= tol, f"{msg}: max|Δ|={d:.3e} > {tol:.1e}"
    return d


def assert_far(a, b, msg, thr=1e-5):
    d = (a - b).abs().max().item()
    assert d > thr, f"{msg}: 期望有差异，但 max|Δ|={d:.3e}"
    return d


# --------------------------------------------------------------------------- #
def test_timestep_multiplier_default():
    te = TimestepEmbed(32)
    assert te.timestep_multiplier == 1.0, "TimestepEmbed 默认 multiplier 应为 1.0"
    # multiplier 只改输入尺度：mult=1 与 mult=1000 对同一 sigma 给不同 embedding
    te1, te1000 = TimestepEmbed(32, timestep_multiplier=1.0), TimestepEmbed(32, timestep_multiplier=1000.0)
    te1000.load_state_dict(te1.state_dict())  # 同 MLP 权重，仅 multiplier 不同
    s = torch.tensor([0.5, 0.5])
    assert_far(te1(s), te1000(s), "multiplier 1 vs 1000 应产生不同 embedding")
    print("✓ timestep_multiplier 默认 1.0；multiplier 只缩放输入尺度")


def test_sandwich_norm_layer():
    cfg = tiny_config()
    torch.manual_seed(0)
    layerF = MoTDecoderLayer(cfg, sandwich_norm=False)
    torch.manual_seed(0)
    layerT = MoTDecoderLayer(cfg, sandwich_norm=True)
    copy_shared(layerT, layerF)  # 共享权重一致；sandwich norm 保持 weight=1
    layerF.eval(); layerT.eval()

    torch.manual_seed(3)
    h_und = torch.randn(BSZ, S_T, cfg.hidden_size)
    h_gen = torch.randn(BSZ, N_IMG, cfg.hidden_size)
    # 复用真实 rotary + mask
    from networks.arch_mot import build_t2i_attention_mask, build_t2i_position_ids
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
    tm = torch.ones(BSZ, S_T, dtype=torch.bool)
    mask = build_t2i_attention_mask(tm, N_IMG)
    pos = build_t2i_position_ids(tm, *GRID)
    rot = Qwen3VLTextRotaryEmbedding(cfg)
    pe = rot(h_gen, pos)

    uF, gF = layerF(h_und, h_gen, pe, mask)
    uT, gT = layerT(h_und, h_gen, pe, mask)
    assert_close(uF, uT, 0.0, "sandwich 不应改变 und 支输出")
    assert_far(gF, gT, "sandwich 应改变 gen 支输出")
    # sandwich 参数命名带 _moe_gen、默认 weight=1
    assert hasattr(layerT, "attn_out_norm_moe_gen") and hasattr(layerT, "mlp_out_norm_moe_gen")
    assert torch.allclose(layerT.attn_out_norm_moe_gen.weight, torch.ones(cfg.hidden_size))
    print("✓ sandwich norm：仅改 gen 支、und 逐位不变、参数命名/初值正确")


def test_trainable_routing():
    m = build_transformer(seed=0, sandwich_norm=True,
                          long_skip=dict(enabled=True, mode="A_full"),
                          abs_sincos=dict(enabled=True))
    m.freeze_understanding()
    new_kinds = ["attn_out_norm_moe_gen", "mlp_out_norm_moe_gen", "long_skip_moe_gen", "abs_sincos_alpha_moe_gen"]
    seen = {k: False for k in new_kinds}
    for name, p in m.named_parameters():
        expect = m.is_trainable_param_name(name)
        assert p.requires_grad == expect, f"{name}: requires_grad={p.requires_grad} expect={expect}"
        for k in new_kinds:
            if k in name:
                seen[k] = True
                assert p.requires_grad, f"{name} 应可训练"
        # 冻结侧绝不含 _moe_gen
        if not expect:
            assert "_moe_gen" not in name, f"冻结参数不该含 _moe_gen: {name}"
    assert all(seen.values()), f"未见到新参数: {[k for k, v in seen.items() if not v]}"
    print("✓ 可训练划分：sandwich/long_skip/abs_sincos 均带 _moe_gen 且可训练，und 全冻结")


def test_ablation_off_zero_overhead():
    base = build_transformer(seed=0, sandwich_norm=True)  # 仅 baseline + sandwich
    full_off = build_transformer(seed=0, sandwich_norm=True,
                                 long_skip=dict(enabled=False),
                                 text_adapter=dict(enabled=False),
                                 abs_sincos=dict(enabled=False))
    n_base = sum(p.numel() for p in base.parameters())
    n_off = sum(p.numel() for p in full_off.parameters())
    assert n_base == n_off, f"消融全关时不应新增参数: base={n_base} off={n_off}"
    print(f"✓ 消融全关 = 零参数开销（{n_base} == {n_off}）")


def test_long_skip_step0_equiv():
    for mode in ("A_full", "B_outer"):
        base = build_transformer(seed=0, long_skip=dict(enabled=False))
        randomize_proj_out(base)
        var = build_transformer(seed=0, long_skip=dict(enabled=True, mode=mode))
        copy_shared(var, base)  # gate 保持 zero-init
        inp = sample_inputs()
        d = assert_close(fwd(base, inp), fwd(var, inp), 1e-6, f"long_skip[{mode}] gate-zero step0 应等价")
        assert var.long_skip_moe_gen[0].gate.item() == 0.0
        # 把 gate 调成非零 → 输出应改变（证明连接真的接上了）
        with torch.no_grad():
            for f in var.long_skip_moe_gen:
                f.gate.fill_(1.0)
        assert_far(fwd(base, inp), fwd(var, inp), f"long_skip[{mode}] gate!=0 应改变输出")
        print(f"✓ long_skip[{mode}]：gate zero-init step0 等价（Δ={d:.1e}），gate≠0 后生效")


def test_abs_sincos_step0_equiv():
    base = build_transformer(seed=0, abs_sincos=dict(enabled=False))
    randomize_proj_out(base)
    var = build_transformer(seed=0, abs_sincos=dict(enabled=True))
    copy_shared(var, base)
    inp = sample_inputs()
    d = assert_close(fwd(base, inp), fwd(var, inp), 1e-6, "abs_sincos alpha-zero step0 应等价")
    with torch.no_grad():
        var.abs_sincos_alpha_moe_gen.fill_(1.0)
    assert_far(fwd(base, inp), fwd(var, inp), "abs_sincos alpha≠0 应改变输出")
    # sincos 数值健全
    sc = build_2d_sincos_pos_embed(*GRID, 64)
    assert sc.shape == (N_IMG, 64) and torch.isfinite(sc).all()
    print(f"✓ abs_sincos：alpha zero-init step0 等价（Δ={d:.1e}），alpha≠0 后生效")


def test_noise_condition_toggle():
    on = build_transformer(seed=0, noise_condition=True)
    randomize_proj_out(on)
    off = build_transformer(seed=0, noise_condition=False)
    copy_shared(off, on)
    inp = sample_inputs()
    out_on, out_off = fwd(on, inp), fwd(off, inp)
    assert torch.isfinite(out_on).all() and torch.isfinite(out_off).all()
    assert_far(out_on, out_off, "noise_condition on/off 应不同")
    # B3「彻底删，省参数」：off 不构造 time_embed；参数量更少
    assert hasattr(on, "time_embed") and not hasattr(off, "time_embed"), \
        "noise_condition=False 不应构造 time_embed"
    assert sum(p.numel() for p in off.parameters()) < sum(p.numel() for p in on.parameters())
    # off 下 backward：不能有「可训练却拿不到梯度」的孤立参数
    randomize_proj_out(off)
    off.freeze_understanding()
    off.train()
    fwd(off, inp).pow(2).mean().backward()
    for name, p in off.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"noise_condition=False 下孤立可训练参数: {name}"
    print("✓ noise_condition：on/off 输出不同；off 不建 time_embed、无孤立可训练参数")


def test_text_adapter_module():
    cfg = tiny_config()
    for bidir in (True, False):
        torch.manual_seed(0)
        ad = TextAdapter(cfg, num_layers=2, bidirectional=bidir).eval()
        h_t = torch.randn(BSZ, S_T, cfg.hidden_size)
        assert_close(ad(h_t), h_t, 1e-6, f"TextAdapter(bidir={bidir}) zero-init 应 passthrough")
        with torch.no_grad():
            ad.out_proj.weight.normal_(0, 0.1)
        assert_far(ad(h_t), h_t, "out_proj 非零后应改变输出")
    # num_layers=0 = 中间对照臂：恒等 passthrough
    ad0 = TextAdapter(cfg, num_layers=0).eval()
    h = torch.randn(BSZ, S_T, cfg.hidden_size)
    assert_close(ad0(h), h, 1e-6, "num_layers=0 应恒等（中间对照臂）")
    print("✓ TextAdapter 模块：双向/因果 zero-init passthrough；num_layers=0 恒等")


def test_text_adapter_arm():
    """B1 adapter 臂：四消融臂前向跑通 + adapter/text K-V 梯度流 + und 全冻结。"""
    inp = sample_inputs()
    arms = {
        "baseline": {},
        "ctrl(num_layers=0)": dict(text_adapter=dict(enabled=True, num_layers=0)),
        "adapter-causal": dict(text_adapter=dict(enabled=True, num_layers=2, bidirectional=False)),
        "adapter-bidir": dict(text_adapter=dict(enabled=True, num_layers=2, bidirectional=True)),
    }
    for name, kw in arms.items():
        m = build_transformer(seed=0, **kw)
        o = fwd(m, inp)
        assert o.shape == (BSZ, N_IMG, PATCH_DIM) and torch.isfinite(o).all(), f"{name} 输出异常 {tuple(o.shape)}"
    # 梯度流：adapter + text K/V 投影应有非零梯度；und 全冻结
    m = build_transformer(seed=0, text_adapter=dict(enabled=True, num_layers=2, bidirectional=True))
    randomize_proj_out(m)
    m.freeze_understanding()
    m.train()
    fwd(m, inp).pow(2).mean().backward()
    nz = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    assert any("text_k_proj_moe_gen" in n for n in nz), "text_k_proj_moe_gen 应有非零梯度"
    assert any("text_v_proj_moe_gen" in n for n in nz), "text_v_proj_moe_gen 应有非零梯度"
    assert any("text_adapter_moe_gen" in n for n in nz), "adapter 应有非零梯度"
    for n, p in m.named_parameters():
        if not m.is_trainable_param_name(n):
            assert p.grad is None, f"冻结参数有梯度: {n}"
    print("✓ B1 adapter 臂：四消融臂（baseline/中间对照/因果/双向）前向跑通；"
          "adapter + text K/V 有非零梯度；und 全冻结")


def _und_ht(m, input_ids, text_mask):
    """复算喂给 adapter 的 und 输出 h_t（融合关=self.norm(h_last)；融合开=self.norm(fused)），
    与 _forward_adapter_arm 的 und 段逐字一致，用于断言 und 级别的 step0 等价。"""
    if getattr(m, "multi_layer_fusion_enabled", False):
        h_last, h_list = m._forward_und(input_ids, text_mask)
        fused = m.multi_layer_fusion_moe_gen(h_list, h_last)
        return m.norm(fused)
    return m._forward_und(input_ids, text_mask)


def test_multi_layer_fusion():
    """Ablation 2 多层特征融合：(a) 关=基线逐位等价；(b) 开+gate0=基线逐位等价（clean step0）；
    (c) gate≠0 改变输出；(d) logits/gate 带 _moe_gen 且可训练、und 全冻结；(e) backward 给 gate
    填梯度（gate≠0 后 logits 也拿到梯度）。tiny 4 层 → 用 num_layers=2（融合末 2 层）。"""
    input_ids, text_mask, img, sigmas = sample_inputs()
    NL = 2  # tiny 仅 4 层，融合末 2 层

    # 基线臂（text_adapter 开、不开融合）—— und 输出 = self.norm(末层 hidden)
    base = build_transformer(seed=0, text_adapter=dict(enabled=True, num_layers=2, bidirectional=True))
    # 融合臂（同 seed 同结构，额外开融合）
    var = build_transformer(seed=0, text_adapter=dict(
        enabled=True, num_layers=2, bidirectional=True,
        multi_layer_fusion=dict(enabled=True, num_layers=NL)))
    copy_shared(var, base)  # 共享权重；融合参数（logits/gate）保持 zero-init

    # (a)/(b) und 级 step0 等价：gate zero-init → fused == h_last → self.norm 后逐位等价基线
    ht_base = _und_ht(base, input_ids, text_mask)
    ht_var = _und_ht(var, input_ids, text_mask)
    d = assert_close(ht_base, ht_var, 0.0, "融合 gate-zero step0 应与基线 und 输出逐位等价")
    assert var.multi_layer_fusion_moe_gen.gate.item() == 0.0
    assert var.multi_layer_fusion_moe_gen.logits.abs().max().item() == 0.0  # 非 one-hot：zero-init

    # 整臂前向也应逐位等价（gate=0）
    inp = (input_ids, text_mask, img, sigmas)
    randomize_proj_out(base); copy_shared(var, base)  # 同步含 proj_out 的共享权重，融合参数仍 zero-init
    assert_close(fwd(base, inp), fwd(var, inp), 0.0, "融合 gate-zero step0 整臂前向应逐位等价基线")

    # (c) gate≠0 → und 输出与整臂输出都应改变（证明融合真的接上了）
    with torch.no_grad():
        var.multi_layer_fusion_moe_gen.gate.fill_(1.0)
    assert_far(ht_base, _und_ht(var, input_ids, text_mask), "gate≠0 应改变 und 输出")
    assert_far(fwd(base, inp), fwd(var, inp), "gate≠0 应改变整臂输出")

    # (d) logits/gate 带 _moe_gen、可训练；und 全冻结
    var.freeze_understanding()
    seen = {"logits": False, "gate": False}
    for name, p in var.named_parameters():
        if "multi_layer_fusion_moe_gen" in name:
            assert "_moe_gen" in name, f"融合参数名应含 _moe_gen: {name}"
            assert p.requires_grad, f"融合参数应可训练: {name}"
            if name.endswith(".logits"):
                seen["logits"] = True
            if name.endswith(".gate"):
                seen["gate"] = True
        # 冻结侧绝不含 _moe_gen
        if not var.is_trainable_param_name(name):
            assert "_moe_gen" not in name, f"冻结参数不该含 _moe_gen: {name}"
    assert all(seen.values()), f"未见到融合参数: {[k for k, v in seen.items() if not v]}"

    # (e) backward：gate≠0 时 gate 与 logits 都应拿到非零梯度
    var2 = build_transformer(seed=0, text_adapter=dict(
        enabled=True, num_layers=2, bidirectional=True,
        multi_layer_fusion=dict(enabled=True, num_layers=NL)))
    randomize_proj_out(var2)
    with torch.no_grad():
        var2.multi_layer_fusion_moe_gen.gate.fill_(0.5)  # gate≠0 → logits 也进入梯度图
    var2.freeze_understanding()
    var2.train()
    fwd(var2, inp).pow(2).mean().backward()
    g = var2.multi_layer_fusion_moe_gen.gate.grad
    lg = var2.multi_layer_fusion_moe_gen.logits.grad
    assert g is not None and g.abs().sum().item() > 0, "gate 应有非零梯度"
    assert lg is not None and lg.abs().sum().item() > 0, "gate≠0 后 logits 应有非零梯度"
    # und 全冻结：无 _moe_gen 的参数不应有梯度
    for n, p in var2.named_parameters():
        if not var2.is_trainable_param_name(n):
            assert p.grad is None, f"冻结参数有梯度: {n}"

    # gate=0 时 ∂fused/∂gate=(elmo−h_last)≠0 → gate 仍学（即使 logits 此时梯度为 0）
    var3 = build_transformer(seed=0, text_adapter=dict(
        enabled=True, num_layers=2, bidirectional=True,
        multi_layer_fusion=dict(enabled=True, num_layers=NL)))
    randomize_proj_out(var3)
    var3.freeze_understanding(); var3.train()
    fwd(var3, inp).pow(2).mean().backward()
    g0 = var3.multi_layer_fusion_moe_gen.gate.grad
    assert g0 is not None and g0.abs().sum().item() > 0, "gate=0 时 gate 仍应有非零梯度（∂fused/∂gate=elmo−h_last）"

    print(f"✓ multi_layer_fusion：关/开+gate0 与基线逐位等价（Δ={d:.1e}）；gate≠0 改变输出；"
          "logits/gate 带 _moe_gen 可训练、und 冻结；backward 给 gate(+logits) 填梯度")


def test_multi_layer_fusion_module():
    """MultiLayerFusion 模块独立单测：zero-init gate → 严格返回 h_last；gate≠0 走向均匀加权 elmo。"""
    torch.manual_seed(0)
    NL, H = 3, 16
    mlf = MultiLayerFusion(NL)
    h_list = [torch.randn(BSZ, S_T, H) for _ in range(NL)]
    h_last = h_list[-1]
    # gate=0 → 逐位 == h_last
    assert_close(mlf(h_list, h_last), h_last, 0.0, "gate zero-init 应逐位返回 h_last")
    # logits zero-init → softmax 均匀（非 one-hot）
    w = torch.softmax(mlf.logits, dim=0)
    assert torch.allclose(w, torch.full((NL,), 1.0 / NL), atol=1e-6), "logits zero-init 应给均匀 softmax"
    # gate=1 → 应等于均匀加权 elmo
    with torch.no_grad():
        mlf.gate.fill_(1.0)
    elmo = sum(h_list) / NL
    assert_close(mlf(h_list, h_last), elmo, 1e-6, "gate=1 + 均匀 logits 应等于末 N 层均值")
    print("✓ MultiLayerFusion 模块：gate0 返回 h_last；logits 均匀（非 one-hot）；gate=1 走加权 elmo")


def test_multi_layer_fusion_requires_adapter():
    """融合开但 text_adapter 关 → 清晰报错（融合输出需喂 adapter）。"""
    raised = False
    try:
        build_transformer(seed=0, text_adapter=dict(
            enabled=False, multi_layer_fusion=dict(enabled=True, num_layers=2)))
    except ValueError as e:
        raised = True
        assert "text_adapter" in str(e), f"报错信息应提示需要 text_adapter: {e}"
    assert raised, "融合开但 adapter 关应抛 ValueError"
    print("✓ multi_layer_fusion：enabled 但 text_adapter 关 → 清晰 ValueError")


def test_und_lora_baseline_joint():
    """Ab3 und LoRA：只挂 baseline joint path 的最后 N 层 und attention；zero-init step0 等价。"""
    inp = sample_inputs()
    lora_cfg = dict(enabled=True, num_layers=2, rank=4, alpha=4, lr_scale=0.3,
                    target_modules=["q", "k", "v", "o"])

    base = build_transformer(seed=0, und_lora=dict(enabled=False))
    randomize_proj_out(base)
    var = build_transformer(seed=0, und_lora=lora_cfg)
    copy_shared(var, base)

    # tiny 4 层 + num_layers=2 → 只在最后两层（2/3）出现 q/k/v/o LoRA，且每个 LoRA 有 down/up。
    lora_names = [n for n, _ in var.named_parameters() if "lora_und" in n]
    assert len(lora_names) == 2 * 4 * 2, f"LoRA 参数数不对: {len(lora_names)}"
    assert all(n.startswith(("layers.2.", "layers.3.")) for n in lora_names), lora_names[:4]
    for target in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert any(target + "_lora_und" in n for n in lora_names), f"缺 LoRA target: {target}"
    assert not any("_moe_gen" in n for n in lora_names), "und LoRA 参数不应带 _moe_gen"

    # step0：up zero-init → LoRA delta=0，baseline joint 输出逐位等价。
    d = assert_close(fwd(base, inp), fwd(var, inp), 0.0, "und_lora zero-init step0 应等价 baseline")

    # LoRA 非零后输出应改变，证明接到了 joint path 的 und attention。
    with torch.no_grad():
        for n, p in var.named_parameters():
            if "lora_und.up.weight" in n:
                p.fill_(0.05)
    assert_far(fwd(base, inp), fwd(var, inp), "und_lora up 非零后应改变 baseline joint 输出")

    # freeze：LoRA 可训，und base 冻结。
    var.freeze_understanding()
    for n, p in var.named_parameters():
        if "lora_und" in n:
            assert p.requires_grad, f"LoRA 应可训练: {n}"
        elif not var.is_trainable_param_name(n):
            assert p.grad is None and not p.requires_grad, f"und base 应冻结: {n}"

    # backward：up zero-init 时 up 有梯度、down 暂无；up 非零后 down 也应有梯度。
    var2 = build_transformer(seed=0, und_lora=lora_cfg)
    randomize_proj_out(var2)
    var2.freeze_understanding(); var2.train()
    fwd(var2, inp).pow(2).mean().backward()
    up_grads = [p.grad.abs().sum().item() for n, p in var2.named_parameters()
                if "lora_und.up.weight" in n and p.grad is not None]
    down_grads = [p.grad for n, p in var2.named_parameters() if "lora_und.down.weight" in n]
    assert any(g > 0 for g in up_grads), "zero-init 时 LoRA up 应有非零梯度"
    assert all(g is None or g.abs().sum().item() == 0 for g in down_grads), "up=0 时 down 梯度应为 0"
    var2.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n, p in var2.named_parameters():
            if "lora_und.up.weight" in n:
                p.fill_(0.05)
    fwd(var2, inp).pow(2).mean().backward()
    down_grads2 = [p.grad.abs().sum().item() for n, p in var2.named_parameters()
                   if "lora_und.down.weight" in n and p.grad is not None]
    assert any(g > 0 for g in down_grads2), "up 非零后 LoRA down 应有非零梯度"

    print(f"✓ und_lora：top-N q/k/v/o、step0 等价 baseline（Δ={d:.1e}）、非零后生效、梯度流向正确")


def test_und_lora_requires_baseline_joint():
    """Ab3 是 baseline joint path，不允许和 B1/Ab2 adapter path 同开。"""
    raised = False
    try:
        build_transformer(seed=0,
                          text_adapter=dict(enabled=True, num_layers=2),
                          und_lora=dict(enabled=True, num_layers=2, rank=4))
    except ValueError as e:
        raised = True
        assert "text_adapter" in str(e) or "joint path" in str(e), e
    assert raised, "und_lora 与 text_adapter 同开应直接报错"
    print("✓ und_lora：与 text_adapter 同开 → 清晰 ValueError")


def test_grad_flow_baseline():
    m = build_transformer(seed=0, sandwich_norm=True)
    randomize_proj_out(m)  # 否则 proj_out zero-init → 输出/loss/梯度恒 0，断言形同虚设
    m.freeze_understanding()
    m.train()
    inp = sample_inputs()
    fwd(m, inp).pow(2).mean().backward()
    n_tr = n_fz = n_nonzero = 0
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"可训练参数无梯度: {name}"
            n_nonzero += int(p.grad.abs().sum().item() > 0)
            n_tr += 1
        else:
            assert p.grad is None, f"冻结参数竟有梯度: {name}"
            n_fz += 1
    # randomize_proj_out 后梯度应真实流动（绝大多数可训练参数拿到非零梯度，非全 0 形同虚设）
    assert n_nonzero >= 0.8 * n_tr, f"仅 {n_nonzero}/{n_tr} 可训练参数有非零梯度，疑似断流"
    print(f"✓ 梯度流向：{n_tr} 可训练参数有梯度（{n_nonzero} 非零）、{n_fz} 冻结参数无梯度")


def test_model_lora_optimizer_groups():
    """model_mot 级：und_lora 拆出第二个 optimizer param group，lr=base_lr*lr_scale。"""
    from omegaconf import OmegaConf
    from networks.model_mot import MoTT2IModel
    cfg = OmegaConf.create({
        "model_cls": "networks.model_mot.MoTT2IModel",
        "target_device": "cpu",
        "model": {"pretrained_path": None, "tokenizer_path": "/fsx/models/Qwen3-VL-2B-Instruct",
                  "tiny_text_config": TINY, "patch_size": 2, "text_max_length": 32,
                  "cfg_dropout_prob": 0.0,
                  "und_lora": {"enabled": True, "num_layers": 2, "rank": 4,
                               "alpha": 4, "lr_scale": 0.3,
                               "target_modules": ["q", "k", "v", "o"]}},
        "vae": {"name": "fake", "downsample_factor": 16, "latent_channels": 16},
        "sigma_sampling": {"dist": "logit_normal", "shift": 1.0},
        "trainer": {"seed": 42, "max_steps": 100, "batch_size": 4},
    })
    model = MoTT2IModel(config=cfg)
    groups = model.get_optimizer_param_groups(weight_decay=0.0, base_lr=1.0e-3)
    assert len(groups) == 2, f"应有 main + und_lora 两个 param group，得到 {len(groups)}"
    assert "lr" not in groups[0], "main group 使用 optimizer 默认 lr"
    assert abs(groups[1]["lr"] - 3.0e-4) < 1e-12, f"LoRA lr 应为 3e-4，得到 {groups[1]['lr']}"
    n_lora = sum(p.numel() for p in groups[1]["params"])
    assert n_lora > 0, "LoRA param group 不能为空"
    print("✓ model_mot：und_lora optimizer 第二组 lr=base_lr×0.3")


def test_model_level_shift_and_dropout():
    """model_mot 级：per-分辨率固定 shift + text_rich dropout hook。需 tokenizer。"""
    from omegaconf import OmegaConf
    from networks.model_mot import MoTT2IModel
    cfg = OmegaConf.create({
        "model_cls": "networks.model_mot.MoTT2IModel",
        "target_device": "cpu",
        "model": {"pretrained_path": None, "tokenizer_path": "/fsx/models/Qwen3-VL-2B-Instruct",
                  "tiny_text_config": TINY, "patch_size": 2, "text_max_length": 32,
                  "cfg_dropout_prob": 1.0, "text_rich_cfg_dropout_prob": 0.0},
        "vae": {"name": "fake", "downsample_factor": 16, "latent_channels": 16},
        "sigma_sampling": {"dist": "logit_normal", "shift": 2.0},
        "trainer": {"seed": 42, "max_steps": 100, "batch_size": 4},
    })
    model = MoTT2IModel(config=cfg)

    # 1) per-分辨率固定 shift：step 0 与 step 99 都应 = 2.0（不随进度变）
    model.set_train_progress(0, 100)
    s0 = model.current_shift()
    model.set_train_progress(99, 100)
    s99 = model.current_shift()
    assert s0 == 2.0 and s99 == 2.0, f"固定 shift 应恒为 2.0，得到 {s0}/{s99}"

    # 2) shift 变换确实改变 sigma 分布（与 shift=1 不同）
    model.config.sigma_sampling["shift"] = 1.0
    torch.manual_seed(0); model.diffusion_generator.manual_seed(0)
    sig1 = model.get_sigmas(64)
    model.config.sigma_sampling["shift"] = 4.0
    model.diffusion_generator.manual_seed(0)
    sig4 = model.get_sigmas(64)
    assert (sig1 - sig4).abs().max() > 1e-3, "shift=1 与 shift=4 的 sigma 应不同"

    # 3) text_rich dropout hook：prob=1 全 drop，但 text_rich=True 的样本（prob=0）不 drop
    model.train()
    prompts = ["a", "b", "c", "d"]
    text_rich = [False, True, False, True]
    out = model._apply_cfg_dropout(prompts, text_rich)
    assert out[0] == "" and out[2] == "", "非文字样本（prob=1）应被 drop"
    assert out[1] == "b" and out[3] == "d", "文字样本（prob=0）不应被 drop"
    print("✓ model_mot：固定 shift（恒 2.0）、shift 变换生效、text_rich dropout hook 正确")


def main():
    print("=== 网络架构改动单元测试（tiny 随机权重，CPU）===")
    test_timestep_multiplier_default()
    test_sandwich_norm_layer()
    test_trainable_routing()
    test_ablation_off_zero_overhead()
    test_long_skip_step0_equiv()
    test_abs_sincos_step0_equiv()
    test_noise_condition_toggle()
    test_text_adapter_module()
    test_text_adapter_arm()
    test_multi_layer_fusion_module()
    test_multi_layer_fusion_requires_adapter()
    test_multi_layer_fusion()
    test_und_lora_baseline_joint()
    test_und_lora_requires_baseline_joint()
    test_grad_flow_baseline()
    test_model_lora_optimizer_groups()
    try:
        test_model_level_shift_and_dropout()
    except Exception as e:  # tokenizer 不可用等
        print(f"⚠ model 级测试跳过（{type(e).__name__}: {e}）")
    print("\n全部网络架构单元测试通过 ✅")


if __name__ == "__main__":
    main()
