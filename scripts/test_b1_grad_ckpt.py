#!/usr/bin/env python3
"""F3 单测（CPU, tiny）：B1 adapter 臂的显存路径正确性。

审计发现：B1 adapter 臂（text_adapter.enabled=true）绕过了 baseline 的 gradient
checkpointing —— `_forward_und` / `_forward_adapter_arm` 两个层循环都没有 checkpoint，
真规模 B1 消融会先撞显存。修复：
  * frozen und 塔 `_forward_und` 整段 `@torch.no_grad()`（零激活、零重算，比 checkpoint 更彻底）；
  * 可训练生成臂 `_forward_adapter_arm` 逐层 checkpoint（与 baseline forward 一致）。

本测试验证修复**只省显存、不改数值/梯度**：
  [A] und 前向在 no_grad 下（H_t.requires_grad=False）；
  [B] gradient_checkpointing on/off 的输出 + 全部可训练梯度数值一致（CPU fp32）；
  [C] und 全冻结无梯度、adapter + text K/V 有非零梯度（回归保护）。

用法: venv-gpu/bin/python scripts/test_b1_grad_ckpt.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from scripts.test_arch_changes import build_transformer, fwd, randomize_proj_out, sample_inputs

B1 = dict(text_adapter=dict(enabled=True, num_layers=2, bidirectional=True))


def _grads(m):
    return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def test_und_no_grad():
    """[A] frozen und 独立前向在 no_grad 下：H_t 不 require grad（零激活保留）。"""
    m = build_transformer(seed=0, **B1)
    m.freeze_understanding()
    m.train()
    inp = sample_inputs()
    h_t = m._forward_und(inp[0], inp[1])
    assert not h_t.requires_grad, "frozen und 应在 no_grad 下前向，H_t 不应 require grad"
    print("✓ [A] und 独立前向在 no_grad 下（H_t.requires_grad=False，零激活保留）")


def test_ckpt_equivalence():
    """[B] gradient_checkpointing on vs off：输出 + 可训练梯度数值一致。"""
    inp = sample_inputs()

    m = build_transformer(seed=0, **B1)
    randomize_proj_out(m)
    m.freeze_understanding()
    m.train()

    m.gradient_checkpointing = False
    out_off = fwd(m, inp)
    out_off.pow(2).mean().backward()
    g_off = _grads(m)
    m.zero_grad(set_to_none=True)

    m.gradient_checkpointing = True
    out_on = fwd(m, inp)
    od = (out_off - out_on).abs().max().item()
    assert od < 1e-5, f"ckpt on/off 输出不一致 max|Δ|={od:.2e}"
    out_on.pow(2).mean().backward()
    g_on = _grads(m)

    assert set(g_off) == set(g_on), "ckpt on/off 的梯度键集不一致"
    assert g_off, "没有可训练梯度——测试无效"
    worst = max((g_off[n] - g_on[n]).abs().max().item() for n in g_off)
    assert worst < 1e-5, f"ckpt on/off 梯度不一致 max|Δ|={worst:.2e}"
    print(f"✓ [B] gradient_checkpointing on/off：输出 max|Δ|={od:.2e}、梯度 max|Δ|={worst:.2e}（数值一致）")


def test_grad_partition():
    """[C] 回归：und 全冻结无梯度；adapter + text K/V 有非零梯度（ckpt 开启下）。"""
    inp = sample_inputs()
    m = build_transformer(seed=0, **B1)
    randomize_proj_out(m)
    m.freeze_understanding()
    m.train()
    m.gradient_checkpointing = True
    fwd(m, inp).pow(2).mean().backward()
    nz = [n for n, p in m.named_parameters()
          if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    assert any("text_k_proj_moe_gen" in n for n in nz), "text_k_proj_moe_gen 应有非零梯度"
    assert any("text_v_proj_moe_gen" in n for n in nz), "text_v_proj_moe_gen 应有非零梯度"
    assert any("text_adapter_moe_gen" in n for n in nz), "adapter 应有非零梯度"
    for n, p in m.named_parameters():
        if not m.is_trainable_param_name(n):
            assert p.grad is None, f"冻结参数有梯度: {n}"
    print("✓ [C] ckpt 开启下：und 全冻结无梯度；adapter + text K/V 有非零梯度")


if __name__ == "__main__":
    test_und_no_grad()
    test_ckpt_equivalence()
    test_grad_partition()
    print("\n✅ F3 通过：B1 臂 frozen und→no_grad + 生成臂 gradient checkpointing，省显存且数值/梯度不变")
