#!/usr/bin/env python3
"""Sample the head/foot crop evaluation matrix.

This is intentionally independent from sample_evalset.py so the historical eval
entrypoints stay untouched. It supports explicit checkpoint paths and non-square
H/W for the stage512a multi-AR visual checks.
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

RUN_SPECS = {
    "rd256_final": {
        "name": "stage256 final causal+MLF",
        "config": "configs/stage256_redesign_v1.yaml",
        "ckpt": f"{CKPT_ROOT}/redesign_256_causalmlf_global2304_20260630/last.pt",
        "exposure_per_step": 2304,
        "min_step": 128000,
        "overrides": [
            "model.text_adapter.enabled=true",
            "model.text_adapter.bidirectional=false",
            "model.text_adapter.multi_layer_fusion.enabled=true",
        ],
    },
    "rd512a_g1": {
        "name": "stage512a G1 gate causal+MLF",
        "config": "configs/stage512a_redesign_v1.yaml",
        "ckpt": f"{CKPT_ROOT}/redesign_512a_gates/gate_g1.pt",
        "exposure_per_step": 1536,
        "min_step": 18000,
        "overrides": [
            "model.text_adapter.enabled=true",
            "model.text_adapter.bidirectional=false",
            "model.text_adapter.multi_layer_fusion.enabled=true",
        ],
    },
    "rd512a_latest": {
        "name": "stage512a latest frozen causal+MLF",
        "config": "configs/stage512a_redesign_v1.yaml",
        "ckpt": os.environ.get(
            "RD512A_LATEST_CKPT",
            f"{CKPT_ROOT}/redesign_512a_multiar_anchor_global1536_20260705/last.pt",
        ),
        "exposure_per_step": 1536,
        "min_step": int(os.environ.get("RD512A_LATEST_MIN_STEP", "18001")),
        "overrides": [
            "model.text_adapter.enabled=true",
            "model.text_adapter.bidirectional=false",
            "model.text_adapter.multi_layer_fusion.enabled=true",
        ],
    },
}


COMPOSITION_TARGET_BY_ID = {
    "anime_eval_000": "full_body_multi",
    "anime_eval_012": "non_fullbody_control",
    "anime_eval_018": "full_body_multi",
    "anime_eval_023": "non_fullbody_control",
    "anime_eval_035": "non_fullbody_control",
    "anime_eval_044": "full_body",
    "anime_eval_052": "full_body",
    "anime_eval_062": "full_body_multi",
    "anime_eval_064": "full_body",
    "anime_eval_077": "lower_body_risk",
    "anime_eval_127": "full_body",
    "anime_eval_133": "full_body",
    "anime_eval_136": "full_body",
}


def load_evalset(path: str, limit: int = 0) -> list[dict]:
    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if "id" not in obj or "prompt" not in obj:
                    raise ValueError(f"{path}: each line must contain id and prompt")
                entries.append(obj)
                if limit and len(entries) >= limit:
                    break
    return entries


def tensor_to_pil(px: torch.Tensor):
    from PIL import Image

    img01 = ((px.float() + 1.0) * 0.5).clamp(0, 1)
    arr = (img01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def parse_ckpt_overrides(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--ckpt-override must be run_key=/path, got {item!r}")
        key, path = item.split("=", 1)
        out[key.strip()] = path.strip()
    return out


def resolve_run(key: str, ckpt_overrides: dict[str, str]) -> dict:
    if key not in RUN_SPECS:
        raise KeyError(f"unknown run {key!r}; known={sorted(RUN_SPECS)}")
    run = dict(RUN_SPECS[key])
    run["key"] = key
    if key in ckpt_overrides:
        run["ckpt"] = ckpt_overrides[key]
    return run


def infer_prompt_style(label: str, entries: list[dict]) -> str:
    styles = {str(e.get("prompt_style", "")) for e in entries if e.get("prompt_style")}
    if len(styles) == 1:
        return next(iter(styles))
    label = label.lower()
    if "tag" in label:
        return "tag"
    if "nl_plain" in label:
        return "nl_plain"
    if "nl_emph" in label or label == "nl13":
        return "nl_emph"
    if "real" in label:
        return "real_nl"
    return "unknown"


def infer_domain(label: str, entries: list[dict]) -> str:
    groups = {str(e.get("group", "")) for e in entries if e.get("group")}
    if len(groups) == 1:
        return next(iter(groups))
    return "real" if label.lower().startswith("real") else "anime"


def infer_composition_target(entry: dict) -> str:
    if entry.get("composition_target"):
        return str(entry["composition_target"])
    if entry.get("id") in COMPOSITION_TARGET_BY_ID:
        return COMPOSITION_TARGET_BY_ID[str(entry["id"])]

    prompt = str(entry.get("prompt", "")).lower()
    targets = " ".join(str(v).lower() for v in entry.get("target_general", []))
    text = f"{prompt} {targets}".replace("-", "_")
    multi = any(t in text for t in (
        "multiple_girls", "multiple boys", "several boys", "2girls", "two girls",
        "1boy, 1girl", "one boy and one girl", "group", "duo",
    ))
    full = any(t in text for t in ("full_body", "full body", "standing", "tachi_e", "tachi-e"))
    lower = any(t in text for t in ("barefoot", "boots", "shoes", "skates", "thighhighs", "pantyhose"))
    upper_control = any(t in text for t in ("close portrait", "upper body", "profile", "bust"))
    if multi and full:
        return "full_body_multi"
    if multi:
        return "full_body_multi"
    if full:
        return "full_body"
    if lower:
        return "lower_body_risk"
    if upper_control:
        return "upper_body_control"
    return "non_fullbody_control"


def build_model(run: dict, weights: str):
    from networks.model_mot import MoTT2IModel

    cfg = OmegaConf.load(str(ROOT / run["config"]))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(run["overrides"]))
    cfg.model.cfg_dropout_prob = 0.0

    print(f"[matrix] build {run['key']} :: {run['name']}", flush=True)
    model = MoTT2IModel(config=cfg).to("cuda")
    ckpt = torch.load(run["ckpt"], map_location="cpu", weights_only=False)
    step = ckpt.get("global_step")
    if run.get("min_step") is not None and (step is None or int(step) < int(run["min_step"])):
        raise RuntimeError(f"{run['key']} checkpoint step {step} < required {run['min_step']}: {run['ckpt']}")
    state = ckpt["trainable_state"]
    weights_src = "trainable"
    if weights == "ema" and ckpt.get("ema"):
        state = ckpt["ema"]
        weights_src = "ema"
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    print(
        f"[matrix] {run['key']} ckpt={run['ckpt']} step={ckpt.get('global_step')} "
        f"weights={weights_src} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return model, ckpt, weights_src, list(missing), list(unexpected)


def sample_run(run: dict, entries: list[dict], args, matrix_dir: Path) -> dict:
    if not os.path.exists(run["ckpt"]):
        raise FileNotFoundError(run["ckpt"])

    model, ckpt, weights_src, missing, unexpected = build_model(run, args.weights)
    run_info = {
        "key": run["key"],
        "name": run["name"],
        "config": run["config"],
        "ckpt": run["ckpt"],
        "ckpt_step": ckpt.get("global_step"),
        "exposure_per_step": run.get("exposure_per_step"),
        "exposure_estimate": (
            int(ckpt.get("global_step")) * int(run["exposure_per_step"])
            if ckpt.get("global_step") is not None and run.get("exposure_per_step")
            else None
        ),
        "weights_src": weights_src,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_keys_head": missing[:20],
        "unexpected_keys_head": unexpected[:20],
        "images": [],
    }

    try:
        for seed in args.seeds:
            run_dir = matrix_dir / f"seed{seed}" / run["key"]
            run_dir.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(entries), args.batch_size):
                chunk = entries[start:start + args.batch_size]
                prompts = [e["prompt"] for e in chunk]
                batch_seed = int(seed) + start
                px = model.generate_samples(
                    prompts,
                    height=args.height,
                    width=args.width,
                    num_steps=args.steps,
                    guidance_scale=args.cfg,
                    seed=batch_seed,
                )
                for j, entry in enumerate(chunk):
                    eid = entry["id"]
                    fname = f"{eid}__{run['key']}__{args.width}x{args.height}__seed{seed}.png"
                    path = run_dir / fname
                    tensor_to_pil(px[j]).save(path)
                    run_info["images"].append(
                        {
                            "id": eid,
                            "seed": int(seed),
                            "path": str(path),
                            "prompt": entry["prompt"],
                            "prompt_style": args.prompt_style,
                            "domain": args.domain,
                            "composition_target": infer_composition_target(entry),
                            "batch_seed": batch_seed,
                            "batch_start": start,
                            "meta": {k: v for k, v in entry.items() if k not in {"prompt"}},
                        }
                    )
                del px
                torch.cuda.empty_cache()
    finally:
        del model, ckpt
        gc.collect()
        torch.cuda.empty_cache()

    return run_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--runs", nargs="+", default=["rd256_final", "rd512a_g1", "rd512a_latest"])
    ap.add_argument("--ckpt-override", nargs="*", default=[])
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--size", type=int, default=0, help="convenience: set both H/W when nonzero")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 1007])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--weights", choices=["ema", "trainable"], default="ema")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "sampling requires CUDA"
    if args.size:
        args.height = int(args.size)
        args.width = int(args.size)
    if not args.height or not args.width:
        raise SystemExit("set either --size or both --height/--width")
    if args.height % 16 or args.width % 16:
        raise SystemExit("height and width must be multiples of 16 for the Flux2 VAE")
    entries = load_evalset(args.evalset, args.limit)
    ckpt_overrides = parse_ckpt_overrides(args.ckpt_override)

    eval_name = Path(args.evalset).stem
    size_name = f"{args.width}x{args.height}"
    label = args.label or eval_name
    args.prompt_style = infer_prompt_style(label, entries)
    args.domain = infer_domain(label, entries)
    matrix_dir = Path(args.out_dir) / label / size_name
    matrix_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "evalset": args.evalset,
        "label": label,
        "n_prompts": len(entries),
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "cfg": args.cfg,
        "seeds": args.seeds,
        "weights": args.weights,
        "prompt_style": args.prompt_style,
        "domain": args.domain,
        "runs": [],
    }
    for key in args.runs:
        run = resolve_run(key, ckpt_overrides)
        info = sample_run(run, entries, args, matrix_dir)
        manifest["runs"].append(info)

    out = matrix_dir / "sample_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[matrix] manifest -> {out}", flush=True)


if __name__ == "__main__":
    main()
