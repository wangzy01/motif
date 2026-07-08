#!/usr/bin/env python3
"""Sample current stage256 comparison runs and write a browsable HTML report.

This is intentionally a single-GPU, read-only checkpoint sampler. It does not
touch training jobs or checkpoint files.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


RUNS = [
    {
        "key": "v1_baseline",
        "name": "v1 baseline",
        "ckpt": "/fsx/evan/checkpoints/mot_pilot_v1/s256_1x8_v1_20260618/last.pt",
        "metrics": "/fsx/evan/checkpoints/mot_pilot_v1/s256_1x8_v1_20260618/metrics.jsonl",
        "overrides": [],
    },
    {
        "key": "b1_adapter",
        "name": "B1 text-adapter (bidir)",
        "ckpt": "/fsx/evan/checkpoints/mot_pilot_v1/s256_1x8_b1adapter_20260618/last.pt",
        "metrics": "/fsx/evan/checkpoints/mot_pilot_v1/s256_1x8_b1adapter_20260618/metrics.jsonl",
        "overrides": ["model.text_adapter.enabled=true", "model.text_adapter.bidirectional=true"],
    },
    {
        "key": "b2_longskip",
        "name": "B2 longskip A_full",
        "ckpt": "/fsx/evan/checkpoints/mot_pilot_v1/s256_1x8_b2longskip_20260618/last.pt",
        "metrics": "/fsx/evan/checkpoints/mot_pilot_v1/s256_1x8_b2longskip_20260618/metrics.jsonl",
        "overrides": ["model.long_skip.enabled=true", "model.long_skip.mode=A_full"],
    },
    {
        "key": "b3_nonoise",
        "name": "B3 no-noise",
        "ckpt": "/fsx/evan/checkpoints/mot_pilot_v1/s256_1x8_b3nonoise_20260618/last.pt",
        "metrics": "/fsx/evan/checkpoints/mot_pilot_v1/s256_1x8_b3nonoise_20260618/metrics.jsonl",
        "overrides": ["model.noise_condition=false"],
    },
]


PROMPTS = [
    "1girl, long silver hair, blue eyes, school uniform, cherry blossoms, upper body, anime style",
    "1boy, black hair, red scarf, standing on rooftop at sunset, city skyline, detailed background",
    "two anime girls sitting together in a cafe, one with twintails, one with short hair, warm lighting",
    "a cute cat girl wearing a maid outfit, holding a cup of coffee, indoor, soft shading",
    "fantasy knight in silver armor, holding a glowing sword, dark forest background, dramatic lighting",
    "chibi style mascot character, round body, big eyes, pastel colors, white background",
    "a serene mountain landscape with a red torii gate, autumn leaves, clear sky, scenic anime background",
    "1girl, witch hat, purple dress, casting a spell, magic circle, night sky with stars",
    "close-up portrait of a young woman, soft rim light, detailed eyes, clean line art",
    "a futuristic city street at night, neon signs, rain reflections, cinematic composition",
    "small cozy bedroom, desk with books and a glowing monitor, warm lamp light, anime background",
    "a dragon flying above snowy mountains, tiny rider silhouette, epic fantasy illustration",
    "three colorful desserts on a plate, strawberries, cream, bright studio lighting",
    "a robot gardener watering flowers in a glass greenhouse, whimsical, detailed background",
    "comic panel, superhero landing pose, speed lines, bold colors, dynamic perspective",
    "a graceful dancer in a flowing red dress, spotlight, dark stage, elegant motion",
]


def load_recent_metrics(path: str) -> dict:
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    if not rows:
        return {}
    rows_by_step = {int(r["step"]): r for r in rows if "step" in r}
    rows = [rows_by_step[k] for k in sorted(rows_by_step)]
    tail = rows[-1000:] if len(rows) >= 1000 else rows
    return {
        "metric_step": rows[-1].get("step"),
        "last1000_loss": float(np.mean([r["loss"] for r in tail if "loss" in r])),
        "last1000_grad": float(np.mean([r["grad_norm"] for r in tail if "grad_norm" in r])),
        "last1000_iter_med": float(np.median([r["iter_s"] for r in tail if "iter_s" in r])),
        "peak_mem_gb": max(float(r.get("peak_mem_gb", 0.0)) for r in tail),
    }


def tensor_to_pil(px: torch.Tensor):
    from PIL import Image

    img01 = ((px.float() + 1.0) * 0.5).clamp(0, 1)
    arr = (img01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_grid(images, out_path: Path, cols: int = 4):
    from PIL import Image, ImageDraw

    if not images:
        return
    w, h = images[0].size
    label_h = 22
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * (h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        x, y = c * w, r * (h + label_h)
        canvas.paste(im, (x, y + label_h))
        draw.rectangle([x, y, x + w, y + label_h], fill=(245, 245, 245))
        draw.text((x + 6, y + 4), f"{i + 1:02d}", fill=(20, 20, 20))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def sample_run(run: dict, args, out_dir: Path) -> dict:
    from networks.model_mot import MoTT2IModel

    cfg = OmegaConf.load(str(ROOT / args.config))
    if run["overrides"]:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(run["overrides"]))
    cfg.model.cfg_dropout_prob = 0.0

    print(f"[sample] build {run['name']}", flush=True)
    model = MoTT2IModel(config=cfg).to("cuda")
    ckpt = torch.load(run["ckpt"], map_location="cpu", weights_only=False)
    state = ckpt["trainable_state"]
    wsrc = "trainable"
    if getattr(args, "weights", "ema") == "ema" and ckpt.get("ema"):
        state = ckpt["ema"]
        wsrc = "ema"
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    print(
        f"[sample] {run['name']} weights={wsrc} ckpt_step={ckpt.get('global_step')} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )

    run_dir = out_dir / run["key"]
    run_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for seed in args.seeds:
        seed_images = []
        for start in range(0, len(PROMPTS), args.batch_size):
            prompts = PROMPTS[start:start + args.batch_size]
            px = model.generate_samples(
                prompts,
                height=args.size,
                width=args.size,
                num_steps=args.steps,
                guidance_scale=args.cfg,
                seed=int(seed) + start,
            )
            for j in range(px.shape[0]):
                idx = start + j
                im = tensor_to_pil(px[j])
                fn = f"seed{seed}_p{idx + 1:02d}.png"
                im.save(run_dir / fn)
                seed_images.append(im)
                saved.append({"seed": seed, "prompt_index": idx + 1, "file": f"{run['key']}/{fn}"})
            del px
            torch.cuda.empty_cache()
        save_grid(seed_images, run_dir / f"grid_seed{seed}.png")

    info = {
        **run,
        "ckpt_step": ckpt.get("global_step"),
        "weights_src": wsrc,
        "metrics_summary": load_recent_metrics(run["metrics"]),
        "images": saved,
    }
    del model, ckpt, state
    gc.collect()
    torch.cuda.empty_cache()
    return info


def write_html(infos: list[dict], args, out_dir: Path):
    prompt_items = "\n".join(
        f"<li><b>{i + 1:02d}</b> {html.escape(p)}</li>" for i, p in enumerate(PROMPTS)
    )
    sections = []
    for info in infos:
        if info.get("error"):
            sections.append(
                f"<section><h2>{html.escape(info['name'])}</h2>"
                f"<p class=\"meta\" style=\"color:#b00\">FAILED: {html.escape(str(info['error']))}</p></section>"
            )
            continue
        m = info.get("metrics_summary", {})
        metrics = (
            f"weights={info.get('weights_src','?')} · "
            f"ckpt step {info.get('ckpt_step')} · metrics step {m.get('metric_step')} · "
            f"last1000 loss {m.get('last1000_loss', float('nan')):.6f} · "
            f"grad {m.get('last1000_grad', float('nan')):.5f} · "
            f"iter median {m.get('last1000_iter_med', float('nan')):.3f}s · "
            f"peak {m.get('peak_mem_gb', float('nan')):.1f}G"
        )
        grids = "\n".join(
            f'<figure><img src="{html.escape(info["key"] + "/grid_seed" + str(seed) + ".png")}" loading="lazy">'
            f"<figcaption>seed {seed}</figcaption></figure>"
            for seed in args.seeds
        )
        cards = []
        for item in info["images"]:
            prompt = PROMPTS[item["prompt_index"] - 1]
            cards.append(
                f'<figure class="card"><img src="{html.escape(item["file"])}" loading="lazy">'
                f'<figcaption><b>{item["prompt_index"]:02d}</b> seed {item["seed"]}<br>'
                f'{html.escape(prompt)}</figcaption></figure>'
            )
        sections.append(
            f"<section><h2>{html.escape(info['name'])}</h2>"
            f"<p class=\"meta\">{html.escape(metrics)}</p>"
            f"<div class=\"grids\">{grids}</div>"
            f"<details><summary>individual images</summary><div class=\"cards\">{''.join(cards)}</div></details>"
            f"</section>"
        )

    body = f"""<!doctype html>
<meta charset="utf-8">
<title>Current stage256 sample grids</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #222; }}
h1 {{ margin-bottom: 4px; }}
.meta {{ color: #555; }}
section {{ border-top: 1px solid #ddd; padding-top: 18px; margin-top: 24px; }}
.grids {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
figure {{ margin: 0; }}
.grids img {{ width: 100%; max-width: 1040px; image-rendering: auto; border: 1px solid #ddd; }}
figcaption {{ font-size: 13px; color: #444; margin-top: 5px; line-height: 1.35; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; margin-top: 12px; }}
.card img {{ width: 100%; border: 1px solid #ddd; }}
li {{ margin: 4px 0; }}
code {{ background: #f4f4f4; padding: 1px 4px; }}
</style>
<h1>Current stage256 sample grids</h1>
<p class="meta">Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}.
Settings: size <code>{args.size}</code>, steps <code>{args.steps}</code>, cfg <code>{args.cfg}</code>,
seeds <code>{', '.join(map(str, args.seeds))}</code>. These are early-stage sanity samples, not final quality.</p>
<h2>Prompt Index</h2>
<ol>{prompt_items}</ol>
{''.join(sections)}
"""
    with open(out_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1a_t2i_256.yaml")
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/results/s256_current_samples"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 123])
    ap.add_argument("--weights", choices=["ema", "trainable"], default="ema")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "sampling requires CUDA"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    infos = []
    for run in RUNS:
        if not os.path.exists(run["ckpt"]):
            print(f"[sample] skip {run['name']}: missing {run['ckpt']}", flush=True)
            continue
        try:
            infos.append(sample_run(run, args, out_dir))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[sample] FAILED {run['name']}: {e}", flush=True)
            infos.append({**run, "error": repr(e), "ckpt_step": None,
                          "weights_src": "-", "metrics_summary": {}, "images": []})
    write_html(infos, args, out_dir)
    print(f"[sample] html -> {out_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
