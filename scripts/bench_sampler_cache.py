#!/usr/bin/env python3
"""G6：真 2B GPU 上验 und KV 缓存的「正确性 + 加速」。

未训练模型 proj_out zero-init → v≡0、cached/非cached 都吐噪声（对比无意义），故先
randomize proj_out 让 v≠0，再：① cached vs 非cached 输出对比（bf16 近一致）；② 计时加速。
CFG（双前向）下 und 缓存收益最大（每步省 2× und 塔重算）。

用法: HF_HUB_OFFLINE=1 venv-gpu/bin/python scripts/bench_sampler_cache.py [--steps 30 --bs 2]
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
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=4.0)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "需要 GPU"

    from networks.model_mot import MoTT2IModel
    cfg = OmegaConf.load(os.path.join(ROOT, "configs", "stage1a_t2i_512.yaml"))
    cfg.model.gradient_checkpointing = False
    device = torch.device("cuda")
    t0 = time.perf_counter()
    model = MoTT2IModel(config=cfg).to(device)
    # 让 v≠0（否则未训练模型 proj_out=0、对比无意义）
    with torch.no_grad():
        torch.nn.init.normal_(model.model.proj_out.weight, std=0.02)
        torch.nn.init.normal_(model.model.proj_out.bias, std=0.02)
    print(f"[load] real 2B on GPU in {time.perf_counter()-t0:.1f}s")

    prompts = ["1girl, silver hair, masterpiece"] * args.bs

    # ---- 正确性：单步 v_pred 对比（cached forward_gen_cached vs joint model.forward）----
    # 分 fp32（验数学：应≈0，与 CPU test_sampler [B5] 逐位 0 一致）+ bf16（真训练精度）。
    # bf16 偏差来自 28 层 und 塔的 reduction-order 累积（cached 孤立 und SDPA vs joint [und;gen] SDPA），
    # 非 bug；端到端是否无害见 overfit_reconstruct.py 的 cached-vs-非cached 重建对比。
    from contextlib import nullcontext
    ids, mask = model.encode_text(prompts)
    f, p, C = model.vae.downsample_factor, model.patch_size, model.latent_channels
    gh, gw = (512 // f) // p, (512 // f) // p
    torch.manual_seed(0)
    x = torch.randn(args.bs, gh * gw, C * p * p, device=device)
    rels = {}
    for tag, ctx in [("fp32", nullcontext()), ("bf16", torch.autocast("cuda", dtype=torch.bfloat16))]:
        worst = 0.0
        with torch.no_grad(), ctx:
            for sig in (0.2, 0.5, 0.8):
                s = torch.full((args.bs,), sig, device=device)
                v_joint = model.model(ids, mask, x, (gh, gw), s)
                kv, pos_gen, gmask = model.model.precompute_und_cache(ids, mask, (gh, gw))
                v_cached = model.model.forward_gen_cached(x, s, kv, pos_gen, gmask, (gh, gw))
                worst = max(worst, ((v_cached.float() - v_joint.float()).abs().max()
                                    / v_joint.float().abs().max()).item())
        rels[tag] = worst
        print(f"  单步 {tag}: cached vs joint worst rel={worst:.2e}")

    # ---- 加速：完整 generate_samples cached vs 非cached ----
    def run(use_cache):
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.generate_samples(prompts, height=512, width=512, num_steps=args.steps,
                               guidance_scale=args.cfg, seed=0, return_latents=True, use_und_cache=use_cache)
        torch.cuda.synchronize()
        return time.perf_counter() - t

    run(True)  # warmup（cudnn autotune）
    t_nc, t_c = run(False), run(True)
    speedup = t_nc / t_c
    print(f"\n=== G6 und KV 缓存（真 2B, {args.steps} step, bs{args.bs}, CFG={args.cfg} 双前向）===")
    print(f"  单步正确性：fp32 rel={rels['fp32']:.2e}（验数学≈0）| bf16 rel={rels['bf16']:.2e}（28层und塔 reduction 累积，非bug）")
    print(f"  非cached {t_nc:.2f}s（{t_nc/args.steps*1e3:.0f} ms/step）→ cached {t_c:.2f}s（{t_c/args.steps*1e3:.0f} ms/step）→ 加速 {speedup:.2f}×")
    assert rels["fp32"] < 1e-3, f"fp32 下 cached 应≈joint（数学等价），但 rel={rels['fp32']:.2e} → 有 bug"
    assert speedup > 1.3, f"加速不足 {speedup:.2f}×"
    print(f"\n✅ 数学正确（fp32 rel={rels['fp32']:.2e}）+ 加速 {speedup:.2f}×；bf16 单步 {rels['bf16']:.2e} 端到端无害见 overfit_reconstruct")


if __name__ == "__main__":
    main()
