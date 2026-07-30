#!/usr/bin/env python3
"""Generate images for DPG-Bench prompts with one or more stage-256 MoT runs.

This is the DPG analogue of experiments/eval/sample_evalset.py: SAME proven,
read-only checkpoint path (build MoTT2IModel from configs/stage1a_t2i_256.yaml +
per-run overrides, load EMA weights, call model.generate_samples, save PNGs).
The only differences vs sample_evalset.py:
  * iterates over the 1065 DPG prompts (experiments/eval/dpg/dpg_prompts.jsonl),
    NOT the anime evalset; no anime targets.
  * image filename = ``<item_id>__<run>__seed<N>.png`` so score_dpg.py maps each
    image back to its DPG question graph by item_id.

Runs in venv-gpu (the proven training/generation stack). READ-ONLY w.r.t.
training: never writes checkpoints/configs; prefer --snapshot-dir /local to read
``eval_<run>.pt`` snapshots (avoids the training atomic-replace race).

512 path: pass ``--size 512 --config configs/stage1a_t2i_512.yaml`` (the model is
resolution-agnostic; the 512 config only changes resolution/shift/steps/batch).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# repo root = .../mot_pilot_v1  (this file is experiments/eval/dpg/generate_dpg.py)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

CKPT_ROOT = "/fsx/evan/checkpoints/mot_pilot_v1"

# run_key -> (checkpoint dir under CKPT_ROOT, name, base config, config overrides).
# IDENTICAL to sample_evalset.py::_RUN_SPEC: the 4 CURRENTLY-RUNNING runs, each with
# its own training base config (B1=v1 cfg, ab*=v2 cfg) so non-overridden architecture
# defaults match training exactly. Override dotlists copied from each run's job yaml.
_CFG_V1 = "configs/stage1a_t2i_256.yaml"
_CFG_V2 = "configs/stage1a_t2i_256_v2.yaml"
_RUN_SPEC = {
    "b1": ("s256_1x8_b1adapter_20260618", "B1 adapter bidir (v1 cfg)", _CFG_V1,
           ["model.text_adapter.enabled=true", "model.text_adapter.bidirectional=true"]),
    "ab1causal": ("s256v2_2x8_ab1causal_20260623", "ab1 adapter causal (v2 cfg)", _CFG_V2,
           ["model.text_adapter.enabled=true", "model.text_adapter.bidirectional=false"]),
    "ab2fusion": ("s256v2_2x8_ab2fusion_20260623", "ab2 adapter+multilayer-fusion (bidir, v2 cfg)", _CFG_V2,
           # bidirectional=true 显式钉住：ckpt 是 bidir 训练的；v2 config 默认已改 causal，不钉会 train/test 错配。
           ["model.text_adapter.enabled=true", "model.text_adapter.bidirectional=true", "model.text_adapter.multi_layer_fusion.enabled=true"]),
    "ab3lora": ("s256v2_2x8_ab3lora_20260623", "ab3 und_lora top8 qkvo r32 (v2 cfg)", _CFG_V2,
           ["model.und_lora.enabled=true"]),
}


def build_runs(ckpt_root: str, snapshot_dir: str | None = None) -> dict:
    runs = {}
    for key, (subdir, name, config, overrides) in _RUN_SPEC.items():
        if snapshot_dir:
            ckpt = os.path.join(snapshot_dir, f"eval_{key}.pt")
        else:
            ckpt = os.path.join(ckpt_root, subdir, "last.pt")
        runs[key] = {"key": key, "name": name, "config": config,
                     "ckpt": ckpt, "overrides": overrides}
    return runs


def load_prompts(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out.append({"id": rec["id"], "prompt": rec["prompt"]})
    return out


def tensor_to_pil(px: torch.Tensor):
    from PIL import Image
    img01 = ((px.float() + 1.0) * 0.5).clamp(0, 1)
    arr = (img01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def tile_2x2(images: list):
    """Tile four NxN PIL images into one 2N x 2N image, quadrant order matching
    the upstream DPG ``crop_tuples_list`` ([TL, TR, BL, BR]).

    Used ONLY for ``--grid 2`` (the pic_num=4 comparability path the ORIGINAL
    mPLUG scorer expects). For the default ``--grid 1`` this is never called and
    the single-image behaviour is byte-for-byte unchanged.
    """
    from PIL import Image
    assert len(images) == 4, "tile_2x2 needs exactly 4 images"
    w, h = images[0].size
    grid = Image.new("RGB", (2 * w, 2 * h))
    # (left, upper) offsets in the SAME order as upstream crop_tuples_list:
    #   (0,0,R,R), (R,0,2R,R), (0,R,R,2R), (R,R,2R,2R)  ->  TL, TR, BL, BR
    offsets = [(0, 0), (w, 0), (0, h), (w, h)]
    for img, (ox, oy) in zip(images, offsets):
        grid.paste(img, (ox, oy))
    return grid


def sample_run(run: dict, entries: list[dict], args, out_dir: Path) -> dict:
    from networks.model_mot import MoTT2IModel

    cfg_path = run.get("config") or args.config
    cfg = OmegaConf.load(str(ROOT / cfg_path))
    if run["overrides"]:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(run["overrides"]))
    cfg.model.cfg_dropout_prob = 0.0

    print(f"[gen] build {run['name']}", flush=True)
    model = MoTT2IModel(config=cfg).to("cuda")
    ckpt = torch.load(run["ckpt"], map_location="cpu", weights_only=False)
    state = ckpt["trainable_state"]
    wsrc = "trainable"
    if args.weights == "ema" and ckpt.get("ema"):
        state = ckpt["ema"]
        wsrc = "ema"
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    print(f"[gen] {run['name']} weights={wsrc} ckpt_step={ckpt.get('global_step')} "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    run_dir = out_dir / run["key"]
    run_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    seed = args.seed

    # --- ADDITIVE: pic_num=4 / 2x2-grid path (only when --grid 2) ----------------
    # The ORIGINAL mPLUG DPG scorer tiles 4 images into one 2x2 grid and crops the
    # quadrants (pic_num=4). For leaderboard comparability we reproduce that here:
    # per prompt we sample 4 images (4 distinct seeds), tile them TL/TR/BL/BR, and
    # save ONE 2N x 2N PNG. score_dpg_mplug.py with --pic-num 4 --resolution N then
    # crops & VQA-averages the 4 quadrants. The default --grid 1 path below is
    # untouched (single image per prompt).
    if getattr(args, "grid", 1) == 2:
        for start in range(0, len(entries), args.batch_size):
            chunk = entries[start:start + args.batch_size]
            prompts = [e["prompt"] for e in chunk]
            # 4 independent samples per prompt (vary the seed across the grid slot).
            quad_pils = [[] for _ in range(len(chunk))]
            for g in range(4):
                px = model.generate_samples(
                    prompts, height=args.size, width=args.size,
                    num_steps=args.steps, guidance_scale=args.cfg,
                    seed=seed + start + g * 100003,  # distinct seed per quadrant
                )
                for j in range(px.shape[0]):
                    quad_pils[j].append(tensor_to_pil(px[j]))
                del px
                torch.cuda.empty_cache()
            for j in range(len(chunk)):
                iid = chunk[j]["id"]
                grid_im = tile_2x2(quad_pils[j])
                grid_im.save(run_dir / f"{iid}__{run['key']}__seed{seed}.png")
                n_saved += 1
            if (start // args.batch_size) % 10 == 0:
                print(f"[gen] {run['key']} (grid2) {n_saved}/{len(entries)}", flush=True)
        info = {"key": run["key"], "name": run["name"], "weights_src": wsrc,
                "ckpt_step": ckpt.get("global_step"), "n_saved": n_saved,
                "image_dir": str(run_dir), "grid": 2}
        del model, ckpt, state
        gc.collect()
        torch.cuda.empty_cache()
        return info
    # --- end ADDITIVE grid path; default single-image path unchanged below -------

    for start in range(0, len(entries), args.batch_size):
        chunk = entries[start:start + args.batch_size]
        prompts = [e["prompt"] for e in chunk]
        px = model.generate_samples(
            prompts, height=args.size, width=args.size,
            num_steps=args.steps, guidance_scale=args.cfg, seed=seed + start,
        )
        for j in range(px.shape[0]):
            iid = chunk[j]["id"]
            im = tensor_to_pil(px[j])
            im.save(run_dir / f"{iid}__{run['key']}__seed{seed}.png")
            n_saved += 1
        del px
        torch.cuda.empty_cache()
        if (start // args.batch_size) % 10 == 0:
            print(f"[gen] {run['key']} {n_saved}/{len(entries)}", flush=True)

    info = {"key": run["key"], "name": run["name"], "weights_src": wsrc,
            "ckpt_step": ckpt.get("global_step"), "n_saved": n_saved,
            "image_dir": str(run_dir)}
    del model, ckpt, state
    gc.collect()
    torch.cuda.empty_cache()
    return info


def main():
    HERE = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1a_t2i_256.yaml",
                    help="256: configs/stage1a_t2i_256.yaml; 512: configs/stage1a_t2i_512.yaml")
    ap.add_argument("--prompts", default=str(HERE / "dpg_prompts.jsonl"))
    ap.add_argument("--out-dir",
                    default=str(ROOT / "experiments/results/dpg_bench/samples"))
    ap.add_argument("--runs", nargs="+", default=["b1", "ab1causal", "ab2fusion", "ab3lora"],
                    help="subset of: b1 ab1causal ab2fusion ab3lora")
    ap.add_argument("--ckpt-root", default=CKPT_ROOT)
    ap.add_argument("--snapshot-dir", default=None,
                    help="if set, load <dir>/eval_<run>.pt snapshots instead of <ckpt-root>/<dir>/last.pt")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weights", choices=["ema", "trainable"], default="ema")
    ap.add_argument("--limit", type=int, default=0, help="cap #prompts (0=all 1065)")
    ap.add_argument("--grid", type=int, default=1, choices=[1, 2],
                    help="1 (default) = one image per prompt (single-image setting). "
                         "2 = 2x2 grid of 4 images tiled into one 2N x 2N PNG "
                         "(pic_num=4); use with score_dpg_mplug.py --pic-num 4 for "
                         "FULL leaderboard comparability with the original mPLUG scorer.")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "generation requires CUDA"

    entries = load_prompts(args.prompts)
    if args.limit:
        entries = entries[:args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_tbl = build_runs(args.ckpt_root, args.snapshot_dir)
    infos = []
    for rk in args.runs:
        run = runs_tbl.get(rk)
        if run is None:
            print(f"[gen] unknown run {rk}, skipping", flush=True)
            continue
        if not os.path.exists(run["ckpt"]):
            print(f"[gen] skip {run['name']}: missing {run['ckpt']}", flush=True)
            continue
        try:
            infos.append(sample_run(run, entries, args, out_dir))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[gen] FAILED {run['name']}: {e}", flush=True)
    manifest = out_dir / "gen_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"n_prompts": len(entries), "runs": infos,
                   "size": args.size, "steps": args.steps, "cfg": args.cfg,
                   "seed": args.seed, "weights": args.weights,
                   "config": args.config, "grid": args.grid}, f, indent=2)
    print(f"[gen] manifest -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
