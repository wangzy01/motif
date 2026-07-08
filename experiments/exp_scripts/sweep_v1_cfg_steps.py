#!/usr/bin/env python3
"""v1 baseline CFG / sampling-steps sensitivity sweep.

Single-GPU, read-only, EMA weights. Loads v1 once, samples the SAME 16 prompts
(imported from sample_current_s256_runs_html so the set is identical across docs)
across (cfg, steps) combos, writes one grid per combo + a browsable HTML.
"""
from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_current_s256_runs_html import PROMPTS, tensor_to_pil, save_grid  # noqa: E402

V1_CKPT = "/fsx/evan/checkpoints/mot_pilot_v1/s256_1x8_v1_20260618/last.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1a_t2i_256.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--weights", choices=["ema", "trainable"], default="ema")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "sampling requires CUDA"

    # cfg sweep @ steps=64, plus step sweep @ cfg=4 (cfg=4/steps=64 shared, sampled once)
    cfg_sweep = [2.0, 4.0, 6.0, 8.0]
    step_sweep = [16, 32]  # 64 already covered by cfg=4 in the cfg sweep
    combos = [("cfg", c, 64) for c in cfg_sweep] + [("steps", 4.0, s) for s in step_sweep]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from networks.model_mot import MoTT2IModel

    cfg = OmegaConf.load(str(ROOT / args.config))
    cfg.model.cfg_dropout_prob = 0.0
    print("[sweep] build v1", flush=True)
    model = MoTT2IModel(config=cfg).to("cuda")
    ckpt = torch.load(V1_CKPT, map_location="cpu", weights_only=False)
    state = ckpt["trainable_state"]
    wsrc = "trainable"
    if args.weights == "ema" and ckpt.get("ema"):
        state = ckpt["ema"]
        wsrc = "ema"
    model.model.load_state_dict(state, strict=False)
    step_label = ckpt.get("global_step")
    print(f"[sweep] v1 weights={wsrc} ckpt_step={step_label}", flush=True)

    results = []  # (kind, cfg, steps, gridfile)
    for kind, cval, sval in combos:
        imgs = []
        for start in range(0, len(PROMPTS), args.batch_size):
            prompts = PROMPTS[start:start + args.batch_size]
            px = model.generate_samples(
                prompts, height=args.size, width=args.size,
                num_steps=sval, guidance_scale=cval, seed=args.seed + start,
            )
            for j in range(px.shape[0]):
                imgs.append(tensor_to_pil(px[j]))
            del px
            torch.cuda.empty_cache()
        gridname = f"grid_cfg{cval:g}_steps{sval}.png"
        save_grid(imgs, out_dir / gridname)
        print(f"[sweep] done cfg={cval} steps={sval} -> {gridname}", flush=True)
        results.append((kind, cval, sval, gridname))

    def fig(r):
        _, cval, sval, g = r
        return (f'<figure><img src="{g}" loading="lazy">'
                f'<figcaption>cfg={cval:g}, steps={sval}</figcaption></figure>')

    cfg_figs = "".join(fig(r) for r in results if r[0] == "cfg")
    step_results = [r for r in results if r[0] == "steps"] + [r for r in results if r[0] == "cfg" and r[1] == 4.0]
    step_figs = "".join(fig(r) for r in sorted(step_results, key=lambda r: r[2]))
    prompt_items = "\n".join(f"<li><b>{i + 1:02d}</b> {html.escape(p)}</li>" for i, p in enumerate(PROMPTS))

    body = f"""<!doctype html><meta charset="utf-8">
<title>v1 CFG/steps sweep</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;color:#222}}
.row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px}}
img{{width:100%;max-width:1040px;border:1px solid #ddd}}
figcaption{{font-size:13px;color:#444;margin-top:5px}}
h2{{border-top:1px solid #ddd;padding-top:14px;margin-top:22px}} code{{background:#f4f4f4;padding:1px 4px}}</style>
<h1>v1 baseline — CFG / sampling-steps sensitivity</h1>
<p>weights <code>{wsrc}</code> · ckpt step <code>{step_label}</code> · size <code>{args.size}</code> ·
seed <code>{args.seed}</code> · same 16 prompts. Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}.</p>
<h2>CFG sweep (steps=64)</h2><div class="row">{cfg_figs}</div>
<h2>Steps sweep (cfg=4)</h2><div class="row">{step_figs}</div>
<h2>Prompt index</h2><ol>{prompt_items}</ol>
"""
    (out_dir / "index.html").write_text(body, encoding="utf-8")
    print(f"[sweep] html -> {out_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
