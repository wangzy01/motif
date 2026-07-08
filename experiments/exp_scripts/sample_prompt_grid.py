#!/usr/bin/env python3
"""共享采样脚本（exp3 caption / exp4 noise-shift 用）：固定 prompt grid + 固定 seed 出图存 PNG。

不改任何核心代码：import MoTT2IModel，按 --config 建模型，可选 load checkpoint 的可训练参数
（strict=False，frozen 塔由 pretrained_path 复原），调 model.generate_samples 出图，存网格 PNG。

⚠️ 模型若未训练或仅早期训练，画质有限——本脚本用于 **A/B 早期对比**（同 seed/同 prompt/同步数下
不同 caption 配方 或 不同 shift 的相对差异），不是看绝对质量（与 generate_samples docstring 一致）。

用法：
  venv-gpu/bin/python experiments/exp_scripts/sample_prompt_grid.py \
      --config configs/stage1a_t2i_256.yaml --ckpt /local/.../last.pt \
      --out /local/evan/mot_exp/samples/exp3A_step20k.png --size 256 --steps 40 --cfg 4.0 --seed 7
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 固定 prompt grid：anime 为主 + 少量通用，含可核对的构图/计数/颜色（prompt-following 判读用）
DEFAULT_PROMPTS = [
    "1girl, long silver hair, blue eyes, school uniform, cherry blossoms, upper body, anime style",
    "1boy, black hair, red scarf, standing on rooftop at sunset, city skyline, detailed background",
    "two anime girls sitting together in a cafe, one with twintails, one with short hair, warm lighting",
    "a cute cat girl wearing a maid outfit, holding a cup of coffee, indoor, soft shading",
    "fantasy knight in silver armor, holding a glowing sword, dark forest background, dramatic",
    "chibi style mascot character, round body, big eyes, pastel colors, white background",
    "a serene mountain landscape with a red torii gate, autumn leaves, clear sky, scenic",
    "1girl, witch hat, purple dress, casting a spell, magic circle, night sky with stars",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="last.pt（含 trainable_state）；缺省 = 未训练模型")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()
    assert torch.cuda.is_available(), "需要 GPU"

    from networks.model_mot import MoTT2IModel

    cfg = OmegaConf.load(args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    cfg.model.cfg_dropout_prob = 0.0  # 采样不 dropout
    device = torch.device("cuda")
    size = int(args.size or {"256": 256, "512": 512, "1024": 1024}[str(cfg.dataset.stage)])

    t0 = time.perf_counter()
    model = MoTT2IModel(config=cfg).to(device)
    shift = float(cfg.get("sigma_sampling", {}).get("shift", 1.0))
    print(f"[sample] model built {time.perf_counter()-t0:.1f}s; size={size} shift={shift} "
          f"steps={args.steps} cfg={args.cfg} seed={args.seed}", flush=True)

    if args.ckpt and os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = model.model.load_state_dict(ck["trainable_state"], strict=False)
        print(f"[sample] loaded ckpt step={ck.get('global_step')} "
              f"(missing {len(missing)} frozen-restored, unexpected {len(unexpected)})", flush=True)
    else:
        print("[sample] ⚠️ 无 ckpt → 未训练模型，画质无意义（仅验路径）", flush=True)

    prompts = DEFAULT_PROMPTS
    t0 = time.perf_counter()
    px = model.generate_samples(prompts, height=size, width=size, num_steps=args.steps,
                                guidance_scale=args.cfg, seed=args.seed)  # [B,3,H,W]
    print(f"[sample] generated {len(prompts)} imgs in {time.perf_counter()-t0:.1f}s", flush=True)

    img01 = ((px.float() + 1.0) * 0.5).clamp(0, 1)
    # 网格：每行 4 张
    try:
        from PIL import Image
    except ImportError:
        print("[sample] PIL 缺失，存 .pt 代替", flush=True)
        torch.save(img01.cpu(), args.out.replace(".png", ".pt"))
        return
    ncol = 4
    n = img01.shape[0]
    nrow = (n + ncol - 1) // ncol
    H, W = img01.shape[2], img01.shape[3]
    canvas = torch.zeros(3, nrow * H, ncol * W)
    for i in range(n):
        r, c = i // ncol, i % ncol
        canvas[:, r * H:(r + 1) * H, c * W:(c + 1) * W] = img01[i].cpu()
    arr = (canvas.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Image.fromarray(arr).save(args.out)
    print(f"[sample] saved grid -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
