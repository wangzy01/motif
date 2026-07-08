#!/usr/bin/env python3
"""G2：探测我们 bool-mask SDPA 的后端选择与吞吐，确认不静默退化到 MATH。

我们的注意力（[networks/arch_mot.py](../networks/arch_mot.py) MoTAttention）用
`F.scaled_dot_product_attention(q, k, v, attn_mask=bool[B,1,S,S])`。任意 / bool mask 下
FlashAttention 后端不可用（只吃 causal flag 或无 mask），PyTorch 会落到
mem-efficient / cuDNN / math。本脚本：
  1. 复刻训练 attention 的真实张量形状（默认 512px：S = 296 text + 1024 img = 1320）
     与真实 bool mask（build_t2i_attention_mask）；
  2. 逐后端 force（torch.nn.attention.sdpa_kernel）跑，报哪些后端支持当前 (bool mask, dtype)；
  3. 给每个支持的后端计时（含 warmup + cuda sync）；
  4. 测「默认（不强制）」选谁、多快；并给「FLASH + 无 mask」作吞吐天花板参照
     （量化用 bool mask 相对理论峰值的损耗，供 G2「是否迁 flex/packed+varlen」决策）。

用法: venv-gpu/bin/python scripts/probe_sdpa_backend.py [--batch 8 --s-img 1024 ...]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from networks.arch_mot import build_t2i_attention_mask

BACKENDS = {
    "FLASH": SDPBackend.FLASH_ATTENTION,
    "EFFICIENT": SDPBackend.EFFICIENT_ATTENTION,
    "CUDNN": SDPBackend.CUDNN_ATTENTION,
    "MATH": SDPBackend.MATH,
}


def bench(fn, iters: int = 50, warmup: int = 15) -> float:
    """返回单次平均耗时 (ms)。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--s-text", type=int, default=296)
    ap.add_argument("--s-img", type=int, default=1024, help="512px @ patch1: 32*32=1024")
    ap.add_argument("--heads", type=int, default=16, help="post-GQA repeat_kv：q/k/v 都 16 头")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "需要 GPU"
    dev = torch.device("cuda")
    dt = getattr(torch, args.dtype)
    B = args.batch
    S = args.s_text + args.s_img
    H, D = args.heads, args.head_dim

    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device=dev, dtype=dt)
    k = torch.randn(B, H, S, D, device=dev, dtype=dt)
    v = torch.randn(B, H, S, D, device=dev, dtype=dt)

    # 真实 bool mask（[B,1,S,S]）：text causal + img 看有效 text + 全 img
    text_mask = torch.ones(B, args.s_text, dtype=torch.bool, device=dev)
    mask = build_t2i_attention_mask(text_mask, args.s_img)
    keep_frac = mask.float().mean().item()

    print(f"=== SDPA backend probe ===")
    print(f"q/k/v: [{B}, {H}, {S}, {D}] {args.dtype}  (S = {args.s_text} text + {args.s_img} img)")
    print(f"mask:  {tuple(mask.shape)} bool, keep_frac={keep_frac:.3f}  (GPU={torch.cuda.get_device_name(0)})\n")

    def run(attn_mask):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    # 默认（不强制后端）—— 这是训练实际走的路径
    t_default = bench(lambda: run(mask))
    print(f"[default]                 {t_default:7.3f} ms/call")

    # 逐后端 force：先验证能否跑（支持性），再计时
    print("\n逐后端 force（bool mask）：")
    per_backend = {}
    for name, be in BACKENDS.items():
        try:
            with sdpa_kernel(be):
                out = run(mask)
                torch.cuda.synchronize()
            t = bench(lambda be=be: _forced(run, mask, be))
            per_backend[name] = t
            print(f"  {name:10s} 支持   {t:7.3f} ms/call")
        except Exception as e:
            per_backend[name] = None
            msg = str(e).strip().splitlines()[0][:80] if str(e).strip() else type(e).__name__
            print(f"  {name:10s} 不支持 ({type(e).__name__}: {msg})")

    # FLASH + 无 mask = 吞吐天花板参照
    print("\n参照（量化 bool mask 的开销）：")
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            _ = run(None)
            torch.cuda.synchronize()
        t_flash_nomask = bench(lambda: _forced(run, None, SDPBackend.FLASH_ATTENTION))
        print(f"  FLASH + 无 mask       {t_flash_nomask:7.3f} ms/call  (理论峰值；我们因 bool mask 用不了)")
    except Exception as e:
        t_flash_nomask = None
        print(f"  FLASH + 无 mask 失败: {type(e).__name__}")

    # 结论
    supported = {n: t for n, t in per_backend.items() if isinstance(t, (int, float))}
    fastest = min(supported, key=supported.get) if supported else None
    print("\n=== 结论 ===")
    flash_ok = isinstance(per_backend.get("FLASH"), (int, float))
    print(f"  FlashAttention 对 bool mask：{'可用' if flash_ok else '不可用（预期；任意 mask 不吃）'}")
    if fastest:
        # 判定默认落在哪个后端：默认耗时最接近哪个 force 后端
        closest = min(supported, key=lambda n: abs(supported[n] - t_default))
        print(f"  默认实际后端 ≈ {closest}（默认 {t_default:.3f}ms 最接近 force-{closest} {supported[closest]:.3f}ms）")
        is_math = closest == "MATH"
        print(f"  是否静默退化到 MATH：{'⚠️ 是（需处理）' if is_math else '否 ✅'}")
        print(f"  最快可用后端：{fastest}（{supported[fastest]:.3f}ms）")
        if t_flash_nomask:
            print(f"  相对 FLASH 无 mask 天花板：{supported[fastest] / t_flash_nomask:.2f}× 慢")


def _forced(run, mask, be):
    with sdpa_kernel(be):
        return run(mask)


if __name__ == "__main__":
    main()
