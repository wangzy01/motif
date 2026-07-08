#!/usr/bin/env python3
"""G5：EMA 单测（CPU 单进程；DTensor 多卡路径由 JCC job + nproc=1 FSDP2 覆盖）。

验证 [model_mot](../networks/model_mot.py) 的 EMA：
  1. init_ema 只覆盖可训练参数（frozen 不在 _ema）；
  2. update_ema 数学：ema = decay·ema + (1-decay)·param（逐位精确）；
  3. ckpt round-trip：ema_state_dict → load_ema_state_dict 到新模型 → 逐位一致；
  4. 多次 update 后 EMA 滞后于 param（确认是平滑而非直接拷贝）。

用法: venv-gpu/bin/python scripts/test_ema.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "configs", "debug_tiny_cpu.yaml")


def build():
    from networks.model_mot import MoTT2IModel
    cfg = OmegaConf.load(CFG)
    cfg.target_device = "cpu"
    return MoTT2IModel(config=cfg)


def main():
    torch.manual_seed(0)
    model = build()
    decay = 0.9
    model.init_ema(decay)

    trainable = {n for n, p in model.model.named_parameters() if p.requires_grad}
    frozen = {n for n, p in model.model.named_parameters() if not p.requires_grad}

    # 1. 只覆盖可训练
    assert set(model._ema) == trainable, "EMA 应只含可训练参数"
    assert not (set(model._ema) & frozen), "EMA 不应含 frozen 参数"
    print(f"[1] init_ema ✓（{len(model._ema)} 可训练，0 frozen）")

    # 2. update_ema 数学：挑一个可训练参数改值，验 lerp 公式
    name = next(n for n, p in model.model.named_parameters() if p.requires_grad)
    p = dict(model.model.named_parameters())[name]
    ema_old = model._ema[name].clone()
    with torch.no_grad():
        p.add_(1.0)  # param += 1
    model.update_ema()
    expected = decay * ema_old + (1.0 - decay) * p.detach().float()
    d = (model._ema[name] - expected).abs().max().item()
    assert d < 1e-6, f"EMA 数学错: max|Δ|={d:.2e}"
    print(f"[2] update_ema 数学 ✓（ema = {decay}·ema + {1-decay:.1f}·param，max|Δ|={d:.1e}）")

    # 3. ckpt round-trip：save → 新模型 load → 逐位一致
    sd = model.ema_state_dict()
    assert set(sd) == trainable and all(torch.is_tensor(v) for v in sd.values())
    model2 = build()
    model2.init_ema(decay)
    model2.load_ema_state_dict(sd)
    mism = max((model._ema[n] - model2._ema[n]).abs().max().item() for n in sd)
    assert mism == 0.0, f"ckpt round-trip 不一致: max|Δ|={mism:.2e}"
    print(f"[3] ckpt round-trip ✓（ema_state_dict→load 逐位一致，{len(sd)} tensors）")

    # 4. 多次 update 后 EMA 滞后于 param（平滑、非直接拷贝）
    with torch.no_grad():
        p.add_(5.0)  # 大幅改 param
    for _ in range(3):
        model.update_ema()
    lag = (model._ema[name] - p.detach().float()).abs().max().item()
    assert lag > 1e-3, "EMA 应滞后于 param（平滑），但几乎等于 param"
    print(f"[4] EMA 滞后 ✓（param 跳变后 ema 平滑跟随，残差 max|Δ|={lag:.3f}）")

    print("\nEMA 单测全部通过 ✅（多卡 DTensor gather/scatter 由 JCC job + nproc=1 FSDP2 覆盖）")


if __name__ == "__main__":
    main()
