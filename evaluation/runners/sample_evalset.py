#!/usr/bin/env python3
"""B6: sample the anime eval set with one or more stage-256 runs.

Read-only checkpoint sampler (same proven path as
experiments/exp_scripts/sample_current_s256_runs_html.py): builds MoTT2IModel
from configs/stage1a_t2i_256.yaml + per-run overrides, loads EMA (or trainable)
weights, runs generate_samples on every prompt in the eval jsonl, and saves PNGs
named ``<eval_id>__<run_key>__seed<seed>.png`` so the scorers map images back to
targets.

Does NOT touch training jobs or checkpoint files. Output images per run go to
<out-dir>/<run_key>/.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CKPT_ROOT = "/fsx/evan/checkpoints/mot_pilot_v1"

# run_key -> (checkpoint dir under CKPT_ROOT, name, base config, config overrides).
# The 4 CURRENTLY-RUNNING stage256 runs (2026-06-28). Each carries its OWN training
# base config so non-overridden architecture defaults match exactly what it trained
# with: B1 = v1 config (g512/400k); ab1/ab2/ab3 = v2 config (g1536/133k). The override
# dotlists are copied verbatim from each run's training job yaml (the authority).
#   b1        jobs/s256_1x8_b1_adapter.yaml      -> text_adapter bidir
#   ab1causal jobs/prod_2x8_ab1causal_wt.yaml    -> text_adapter causal
#   ab2fusion jobs/prod_2x8_ab2fusion_wt.yaml    -> text_adapter (bidir) + multi_layer_fusion；ckpt=bidir 训练，eval 显式钉 bidirectional=true（v2 config 默认已改 causal，2026-06-30）
#   ab3lora   jobs/prod_2x8_ab3lora_wt.yaml      -> und_lora (no adapter; mutually exclusive)
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
    """Resolve the run table. If snapshot_dir is given, look for per-run
    snapshot files ``<snapshot_dir>/eval_<key>.pt`` (used by the job to avoid
    the training atomic-replace race); otherwise canonical
    ``<ckpt_root>/<dir>/last.pt``."""
    runs = {}
    for key, (subdir, name, config, overrides) in _RUN_SPEC.items():
        if snapshot_dir:
            ckpt = os.path.join(snapshot_dir, f"eval_{key}.pt")
        else:
            ckpt = os.path.join(ckpt_root, subdir, "last.pt")
        runs[key] = {"key": key, "name": name, "config": config,
                     "ckpt": ckpt, "overrides": overrides}
    return runs


def load_evalset(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def tensor_to_pil(px: torch.Tensor):
    from PIL import Image
    img01 = ((px.float() + 1.0) * 0.5).clamp(0, 1)
    arr = (img01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def sample_run(run: dict, entries: list[dict], args, out_dir: Path) -> dict:
    from networks.model_mot import MoTT2IModel

    cfg_path = run.get("config") or args.config
    cfg = OmegaConf.load(str(ROOT / cfg_path))
    if run["overrides"]:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(run["overrides"]))
    cfg.model.cfg_dropout_prob = 0.0

    print(f"[sample] build {run['name']}", flush=True)
    model = MoTT2IModel(config=cfg).to("cuda")
    ckpt = torch.load(run["ckpt"], map_location="cpu", weights_only=False)
    state = ckpt["trainable_state"]
    wsrc = "trainable"
    if args.weights == "ema" and ckpt.get("ema"):
        state = ckpt["ema"]
        wsrc = "ema"
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    print(f"[sample] {run['name']} weights={wsrc} ckpt_step={ckpt.get('global_step')} "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    run_dir = out_dir / run["key"]
    run_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    seed = args.seed
    for start in range(0, len(entries), args.batch_size):
        chunk = entries[start:start + args.batch_size]
        prompts = [e["prompt"] for e in chunk]
        px = model.generate_samples(
            prompts, height=args.size, width=args.size,
            num_steps=args.steps, guidance_scale=args.cfg, seed=seed + start,
        )
        for j in range(px.shape[0]):
            eid = chunk[j]["id"]
            im = tensor_to_pil(px[j])
            im.save(run_dir / f"{eid}__{run['key']}__seed{seed}.png")
            n_saved += 1
        del px
        torch.cuda.empty_cache()

    info = {"key": run["key"], "name": run["name"], "weights_src": wsrc,
            "ckpt_step": ckpt.get("global_step"), "n_saved": n_saved,
            "image_dir": str(run_dir)}
    del model, ckpt, state
    gc.collect()
    torch.cuda.empty_cache()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1a_t2i_256.yaml")
    ap.add_argument("--evalset", default=str(Path(__file__).with_name("prompts")
                                             / "anime_evalset.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/results/eval_b1_prompt_following/samples"))
    ap.add_argument("--runs", nargs="+", default=["b1", "ab1causal", "ab2fusion", "ab3lora"],
                    help="subset of: b1 ab1causal ab2fusion ab3lora")
    ap.add_argument("--ckpt-root", default=CKPT_ROOT,
                    help="root of canonical <run>/last.pt checkpoint dirs")
    ap.add_argument("--snapshot-dir", default=None,
                    help="if set, load <dir>/eval_<run>.pt snapshots instead")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weights", choices=["ema", "trainable"], default="ema")
    ap.add_argument("--limit", type=int, default=0, help="cap #prompts (0=all)")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "sampling requires CUDA"

    entries = load_evalset(args.evalset)
    if args.limit:
        entries = entries[:args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_tbl = build_runs(args.ckpt_root, args.snapshot_dir)
    infos = []
    for rk in args.runs:
        run = runs_tbl.get(rk)
        if run is None:
            print(f"[sample] unknown run {rk}, skipping", flush=True)
            continue
        if not os.path.exists(run["ckpt"]):
            print(f"[sample] skip {run['name']}: missing {run['ckpt']}", flush=True)
            continue
        try:
            infos.append(sample_run(run, entries, args, out_dir))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[sample] FAILED {run['name']}: {e}", flush=True)
    manifest = out_dir / "sample_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"n_prompts": len(entries), "runs": infos,
                   "size": args.size, "steps": args.steps, "cfg": args.cfg,
                   "seed": args.seed, "weights": args.weights}, f, indent=2)
    print(f"[sample] manifest -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
