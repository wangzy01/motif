#!/usr/bin/env python3
"""固定-sigma 评估：消除 exp4『跨 shift 训练 loss 不可比』的 confound。

exp4 用各自 shift 采样分布上的训练 loss 比 shift → 被 sigma 分布污染（高 shift 多采高噪声区、
loss 与 sigma_mean 反相关 → "shift 大 loss 低"含假象）。本脚本对一个【训练好的模型】在一组
**固定 sigma 网格** + **固定 eval batch** + **固定噪声**上算 rectified-flow MSE，得 loss(sigma) 曲线。
同分辨率不同 shift 的模型在【完全相同的 sigma/数据/噪声】上比较 → 干净可比（谁的曲线低谁好）。

复用 model 内部（不重写）：model.prepare_latents(VAE encode) + model._pack_latents + model.model(
  input_ids, text_mask, x_t, grid_hw, sigmas) —— 与 forward 第 300-310 行同一套数学。

确定性：固定 eval shard 头部 nrows 行（无 rng→legacy 顺序读，跨 worker 一致）+ 固定 caption seed +
固定噪声 generator。anime R2 取图 ~100% 可靠 → 跨 worker batch 基本逐图一致。
注：eval shard 非严格 held-out（训练用全 shard），但对【相对 shift 比较】两模型同 eval 数据 → 排序不受影响。

用法：python fixed_sigma_eval.py --config <cfg> --ckpt <last.pt> --resolution 512 --shift-label 512_s3.0 --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--resolution", type=int, required=True)
    ap.add_argument("--shift-label", required=True, help="仅用于标注/文件名，如 512_s3.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-shard",
                    default="/s3-pixai-test-uw2/evan/mot_filtering_v1/t5_manifest/daily_v1/anime/anime_0063.parquet")
    ap.add_argument("--n-eval", type=int, default=48)
    ap.add_argument("--read-rows", type=int, default=3000, help="从 eval shard 头部读多少行再筛 elig")
    ap.add_argument("--sigmas", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--anime-cache", default="/local/evan/anime_img_cache")
    ap.add_argument("--overrides", nargs="*", default=[],
                    help="dotlist config overrides for the ablation, e.g. model.text_adapter.enabled=true")
    ap.add_argument("--weights", choices=["ema", "trainable"], default="ema")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "需要 GPU"
    torch.manual_seed(args.seed)

    from networks.model_mot import MoTT2IModel
    from training.dataline.caption_build import build_anime_caption
    from training.dataline.config import load_sampling_config
    from training.dataline.image_fetch import (decode_rgb, fetch_url_bytes, resize_crop_uint8,
                                               uint8_to_norm_tensor)
    from training.dataline.manifest_index import load_anime_shard

    cfg = OmegaConf.load(args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    cfg.model.cfg_dropout_prob = 0.0
    device = torch.device("cuda")
    S = int(args.resolution)
    sigmas_grid = [float(x) for x in args.sigmas.split(",")]

    t0 = time.perf_counter()
    model = MoTT2IModel(config=cfg).to(device)
    model.configure_training_rng(global_rank=0, device=device)
    # Historical protocol used torch.manual_seed(args.seed), hence the default
    # CUDA generator sampled posterior epsilon from exactly args.seed.  Preserve
    # that identity while switching to an explicit device-matched generator.
    model.vae_generator.manual_seed(args.seed)
    model.eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["trainable_state"]
    wsrc = "trainable"
    if args.weights == "ema" and ck.get("ema"):
        state = ck["ema"]
        wsrc = "ema"
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    step = ck.get("global_step")
    print(f"[eval] model+ckpt step={step} weights={wsrc} built {time.perf_counter()-t0:.1f}s "
          f"(missing {len(missing)} frozen, unexpected {len(unexpected)})", flush=True)

    # ---- 固定 eval batch（确定性顺序读 + 取图 + caption + VAE encode）----
    scfg = load_sampling_config(
        cfg.dataset.sampling_config if "sampling_config" in cfg.dataset else
        "/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json")
    arr = load_anime_shard(args.eval_shard, scfg, with_caption=True, nrows=args.read_rows, rng=None)
    elig = arr[f"elig_{S}"] if f"elig_{S}" in arr else arr["elig_256"]
    idxs = list(np.where(elig)[0])
    from collections import Counter
    freq = Counter()
    for tg in arr["tags_general"]:
        if tg:
            freq.update(tg)
    cap_rng = random.Random(args.seed)
    ANIME_KEYS = ["rating_bucket", "refined_description", "medium_description", "short_description",
                  "tags_general", "tags_artist", "tags_character", "tags_copyright", "tags_meta",
                  "foreground", "background", "style", "image_type", "camera_and_lens_effects", "year"]
    pixels, prompts = [], []
    for i in idxs:
        if len(pixels) >= args.n_eval:
            break
        raw = fetch_url_bytes(arr["s3key"][i], cache_dir=args.anime_cache)
        if not raw:
            continue
        img = decode_rgb(raw)
        if img is None or min(img.size) < S:
            continue
        u8 = resize_crop_uint8(img, S)
        f = {k: arr[k][i] for k in ANIME_KEYS if k in arr}
        prompts.append(build_anime_caption(f, str(S), scfg.anime_caption, cap_rng, freq))
        pixels.append(uint8_to_norm_tensor(u8))
    n = len(pixels)
    assert n >= 8, f"eval batch 太小({n})；增大 --read-rows"
    px = torch.stack(pixels, 0).to(device)
    print(f"[eval] eval batch n={n} @ {S}px built {time.perf_counter()-t0:.1f}s", flush=True)

    with torch.no_grad():
        latents = model.prepare_latents({"pixels": px})           # [n,C,h,w]
        bsz, c, h, w = latents.shape
        p = model.patch_size
        grid_hw = (h // p, w // p)
        x0 = model._pack_latents(latents, bsz, c, h, w, patch=p)  # [n,N,C*p*p]
        input_ids, text_mask = model.encode_text(prompts)
        # 固定噪声（同一 generator → 所有 sigma/模型用同噪声样本）
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        noise = torch.randn(x0.shape, generator=gen).to(device, x0.dtype)
        target = (noise - x0).float()

        loss_by_sigma = {}
        # 与训练/采样同款 bf16 autocast（model_mot.py:436）；不裹则 bf16 输入×fp32 权重 dtype 不匹配。
        amp = torch.autocast("cuda", dtype=torch.bfloat16)
        for sg in sigmas_grid:
            sig = torch.full((bsz, 1, 1), sg, device=device, dtype=x0.dtype)
            x_t = (1.0 - sig) * x0 + sig * noise
            sigmas = torch.full((bsz,), sg, device=device)
            with amp:
                v_pred = model.model(input_ids, text_mask, x_t, grid_hw, sigmas)
            mse = (v_pred.float() - target).pow(2).mean().item()
            loss_by_sigma[f"{sg:.2f}"] = round(mse, 5)
            print(f"[eval] sigma={sg:.2f} mse={mse:.5f}", flush=True)

    mean_loss = round(float(np.mean(list(loss_by_sigma.values()))), 5)
    out = {
        "shift_label": args.shift_label, "resolution": S, "ckpt_step": step, "weights": wsrc, "n_eval": n,
        "eval_shard": os.path.basename(args.eval_shard), "seed": args.seed,
        "loss_by_sigma": loss_by_sigma, "mean_loss_over_grid": mean_loss,
        "note": "固定 sigma 网格 + 固定 batch + 固定噪声；同分辨率不同 shift 可直接比（曲线低=好）。",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    print(f"[eval] DONE mean_loss={mean_loss} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
