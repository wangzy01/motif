#!/usr/bin/env python3
"""真权重 smoke test：加载 Qwen3-VL-2B → 复制生成塔 → 冻结 → 一次前向。

CPU 可跑（约 12GB 内存，分钟级）。验证点：
  1. und 权重 100% 命中（model_mot._load_und_weights 内部有断言）；
  2. 复制后 gen 塔与 und 塔权重逐 tensor 相等；
  3. 冻结划分正确（可训练 = *_moe_gen + proj_in/out + time_embed）；
  4. frozen und 路输出与 HF 官方 Qwen3VLTextModel 数值一致（关键！）；
  5. 一次 forward/backward：loss 有限、frozen 参数无梯度、可训练参数有梯度。

用法: venv/bin/python scripts/smoke_real_weights.py [--skip-hf-ref]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

MODEL_PATH = "/fsx/models/Qwen3-VL-2B-Instruct"


def build_config():
    return OmegaConf.create(
        {
            "model_cls": "networks.model_mot.MoTT2IModel",
            "target_device": "cpu",
            "model": {
                "pretrained_path": MODEL_PATH,
                "tokenizer_path": MODEL_PATH,
                "patch_size": 2,
                "time_gap": 2048,
                "text_max_length": 64,
                "cfg_dropout_prob": 0.1,
                "prompt_template": {"enabled": True},
            },
            "vae": {"name": "fake", "downsample_factor": 16, "latent_channels": 128},
            "sigma_sampling": {"dist": "logit_normal"},
            "trainer": {"seed": 42, "max_steps": 100, "batch_size": 2},
        }
    )


def check_tower_copy(model):
    print("\n[2] 检查生成塔复制 ...")
    n = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        pairs = [(getattr(attn, k), getattr(attn, f"{k}_moe_gen"))
                 for k in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
        pairs += [(layer.mlp, layer.mlp_moe_gen),
                  (layer.input_layernorm, layer.input_layernorm_moe_gen),
                  (layer.post_attention_layernorm, layer.post_attention_layernorm_moe_gen)]
        for und_m, gen_m in pairs:
            for (kn, u), (_, g) in zip(und_m.state_dict().items(), gen_m.state_dict().items()):
                assert torch.equal(u, g), f"复制不一致: {kn}"
                n += 1
    assert torch.equal(model.model.norm.weight, model.model.norm_moe_gen.weight)
    print(f"    ✓ gen 塔与 und 塔 {n + 1} 个 tensor 完全一致")


def check_freeze(model):
    print("\n[3] 检查冻结划分 ...")
    bad_trainable, bad_frozen = [], []
    for name, p in model.model.named_parameters():
        expect = model.model.is_trainable_param_name(name)
        if p.requires_grad != expect:
            (bad_trainable if p.requires_grad else bad_frozen).append(name)
    assert not bad_trainable and not bad_frozen, f"冻结划分错误: {bad_trainable[:3]} {bad_frozen[:3]}"
    stats = model.model.param_count()
    print(f"    ✓ trainable={stats['trainable']/1e9:.3f}B frozen={stats['frozen']/1e9:.3f}B "
          f"(生成塔预期 ~1.4B + 接口层)")


@torch.no_grad()
def check_und_matches_hf(model, prompts):
    """frozen und 路（无图 token 时）应与 HF Qwen3VLTextModel 完全一致。"""
    print("\n[4] und 路 vs HF 官方实现数值对照 ...")
    # 注意：Qwen3VLTextModel.from_pretrained 直接吃全量 checkpoint 会因
    # "model.language_model.*" 前缀全部 MISSING（拿到随机权重）。必须从
    # ForConditionalGeneration 整体加载后取 language_model 子模块。
    from transformers import Qwen3VLForConditionalGeneration

    full = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.float32, attn_implementation="sdpa"
    )
    hf = full.model.language_model
    hf.eval()

    input_ids, text_mask = model.encode_text(prompts)
    # 我们的 und 路：手动跑一遍（无图 token：给 0 个图 token 不可行，
    # 这里直接逐层跑 und 分支等价计算——文本看不到图，结果与联合前向一致）
    m = model.model
    h = m.embed_tokens(input_ids)
    from networks.arch_mot import build_t2i_attention_mask, build_t2i_position_ids
    # 1x1 的最小图块（会参与联合注意力但不影响文本：文本行看不到图）
    img = torch.zeros(input_ids.shape[0], 1, m.patch_dim)
    mask = build_t2i_attention_mask(text_mask, 1)
    pos = build_t2i_position_ids(text_mask, 1, 1, m.time_gap)
    cos, sin = m.rotary_emb(h, pos)
    h_gen = m.proj_in(img) + m.time_embed(torch.tensor([0.5] * input_ids.shape[0]))[:, None, :]
    h_und = h
    for layer in m.layers:
        h_und, h_gen = layer(h_und, h_gen, (cos, sin), mask)
    ours = m.norm(h_und)

    # HF 参考：causal LM 前向（右 pad 不影响左侧有效位置的 causal 输出）
    hf_out = hf(input_ids=input_ids, attention_mask=text_mask.long()).last_hidden_state

    valid = text_mask.bool()
    diff = (ours[valid] - hf_out[valid]).abs()
    print(f"    max|Δ|={diff.max().item():.3e}  mean|Δ|={diff.mean().item():.3e}")
    assert diff.max().item() < 5e-4, "und 路与 HF 输出不一致——权重映射或 mask/RoPE 有 bug"
    print("    ✓ frozen und 路与 HF Qwen3VLTextModel 数值一致")

    # B1 adapter 臂的独立 und 前向（_forward_und）也应与 HF 一致（验证 adapter 臂 und 路径正确）
    h_t_arm = m._forward_und(input_ids, text_mask)
    diff2 = (h_t_arm[valid] - hf_out[valid]).abs()
    assert diff2.max().item() < 5e-4, "B1 _forward_und 与 HF 不一致——adapter 臂 und 路径有 bug"
    print(f"    ✓ B1 _forward_und（adapter 臂 und 路径）也与 HF 一致 (max|Δ|={diff2.max().item():.3e})")
    del hf, full


def check_forward_backward(model):
    print("\n[5] 真规模 forward/backward ...")
    batch = {
        "prompts": ["1girl, silver hair, masterpiece", "a cat on a windowsill"],
        "latents": torch.randn(2, 128, 16, 16),  # 256px 等效，省 CPU 时间
    }
    model.train()
    model.set_train_progress(0, 100)
    t0 = time.perf_counter()
    loss, sigmas, extras = model(batch)
    scalar = loss.mean()
    scalar.backward()
    print(f"    loss={scalar.item():.4f} sigmas={sigmas.tolist()} ({time.perf_counter()-t0:.1f}s)")
    assert torch.isfinite(scalar), "loss 非有限"
    n_grad = n_nograd = 0
    for name, p in model.model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"可训练参数无梯度: {name}"
            n_grad += 1
        else:
            assert p.grad is None, f"frozen 参数竟有梯度: {name}"
            n_nograd += 1
    print(f"    ✓ {n_grad} 个可训练参数均有梯度；{n_nograd} 个 frozen 参数无梯度")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hf-ref", action="store_true", help="跳过 HF 对照（省内存/时间）")
    args = parser.parse_args()

    from networks.model_mot import MoTT2IModel

    print("[1] 加载真权重并构建模型（含 und 命中断言）...")
    t0 = time.perf_counter()
    model = MoTT2IModel(config=build_config())
    model.float()
    print(f"    ✓ 构建完成 {time.perf_counter()-t0:.1f}s")

    check_tower_copy(model)
    check_freeze(model)
    if not args.skip_hf_ref:
        model.eval()
        check_und_matches_hf(model, ["1girl, silver hair", "a cat"])
    check_forward_backward(model)
    print("\n真权重 smoke test 全部通过 ✅")


if __name__ == "__main__":
    main()
