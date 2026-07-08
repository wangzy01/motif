#!/usr/bin/env python3
"""G6 端到端：过拟合重建——采样符号/方向/σ schedule 的最强正确性信号（GPU 真 2B）。

思路：固定一个 batch（StubT2IDataset, mode=latents → latent 即 x0），in-process 过拟合
若干百步（学会该 x0 在各噪声级的 v=noise-x0），再从纯噪声用 generate_samples
(return_latents, guidance_scale=1.0) 采样，比较 sampled latent 与 target 的 MSE：
应**远低于**「随机噪声 vs target」基线（≈2.0）。若 Euler 符号 / σ 方向 / schedule 错，
重建会发散（MSE >> 基线）——这是机制单测（CPU）覆盖不到的端到端方向校验。

用法: HF_HUB_OFFLINE=1 venv-gpu/bin/python scripts/overfit_reconstruct.py [--steps 600 --bs 4]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--sample-steps", type=int, default=50)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "需要 GPU"

    from networks.model_mot import MoTT2IModel
    from training.stub_data import StubT2IDataset

    cfg = OmegaConf.load(os.path.join(ROOT, "configs", "stage1a_t2i_512.yaml"))
    cfg.model.cfg_dropout_prob = 0.0          # 干净条件映射（去掉 dropout 混淆重建信号）
    cfg.model.gradient_checkpointing = False  # bs 小、显存够，关 ckpt 跑更快
    cfg.trainer.max_steps = args.steps
    device = torch.device("cuda")

    t0 = time.perf_counter()
    model = MoTT2IModel(config=cfg).to(device)
    print(f"[load] model on GPU in {time.perf_counter()-t0:.1f}s")

    # 固定 batch（mode=latents, overfit_one → 每次同一 x0 + prompts）
    ds = StubT2IDataset(image_size=512, latent_channels=128, downsample_factor=16,
                        mode="latents", overfit_one=True, seed=0)
    batch = ds.build_batch(args.bs, 0)
    target = batch["latents"].to(device)       # [B,128,32,32] == x0（model-space）
    prompts = batch["prompts"]
    base = (torch.randn_like(target) - target).pow(2).mean().item()  # 噪声基线 ≈ 2.0

    # ---- 过拟合 ----
    opt = torch.optim.AdamW(model.get_optimizer_param_groups(0.0), lr=args.lr, betas=(0.9, 0.95))
    model.train()
    t0 = time.perf_counter()
    for step in range(args.steps):
        model.set_train_progress(step, args.steps)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, _ = model(batch)
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0 or step == args.steps - 1:
            print(f"  overfit step {step:4d}: loss={loss.mean().item():.4f}  "
                  f"({(time.perf_counter()-t0)/(step+1)*1e3:.0f} ms/step)")

    # ---- 采样重建（cond-only, guidance_scale=1.0, return_latents）：cached vs 非cached 端到端对比 ----
    # 既验采样符号/σ 方向（重建≪基线），又验 G6 KV 缓存端到端无害（cached ≈ 非cached）。
    def recon_mse(use_cache):
        s = model.generate_samples(prompts, height=512, width=512, num_steps=args.sample_steps,
                                   guidance_scale=1.0, seed=123, return_latents=True, use_und_cache=use_cache)
        return (s.float() - target.float()).pow(2).mean().item()

    recon_nc = recon_mse(False)
    recon_c = recon_mse(True)

    print(f"\n=== 过拟合重建（bs={args.bs}, overfit {args.steps} step, 采样 {args.sample_steps} step）===")
    print(f"  MSE(noise, target) 基线       = {base:.3f}")
    print(f"  MSE(sampled, target) 非cached = {recon_nc:.3f}（{recon_nc/base:.3f}× 基线）")
    print(f"  MSE(sampled, target) cached   = {recon_c:.3f}（{recon_c/base:.3f}× 基线，G6 KV 缓存）")
    print(f"  cached vs 非cached 重建差      = {abs(recon_c - recon_nc):.4f}")
    assert recon_nc < 0.6 * base, f"非cached 重建 {recon_nc:.3f} 未明显低于基线——采样符号/σ 方向可能错"
    assert recon_c < 0.6 * base, f"cached 重建 {recon_c:.3f} 未明显低于基线——KV 缓存可能有 bug"
    assert abs(recon_c - recon_nc) < 0.1 * base, f"cached 与非cached 重建差过大→KV 缓存 end-to-end 不等价"
    print(f"\n✅ 过拟合重建通过（采样符号/σ 方向/VAE 端到端正确）+ G6 KV 缓存端到端无害"
          f"（cached {recon_c/base:.2f}× ≈ 非cached {recon_nc/base:.2f}× 基线）")


if __name__ == "__main__":
    main()
