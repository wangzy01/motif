#!/usr/bin/env python3
"""B6 CLIP-score (prompt-image alignment) scorer using open_clip.

CLIP-score per image = cosine(image_embed, text_embed) of the prompt, reported
both raw (in [-1,1], typically ~0.15-0.35 for aligned anime) and as the common
"CLIPScore" = 100 * max(cos, 0).

Model: open_clip ViT-L/14 (OpenAI weights). The OpenAI tag in open_clip resolves
to the HF repo ``timm/vit_large_patch14_clip_224.openai`` which is present in the
local cache at /fsx/hf_cache/open_clip — so this runs fully offline with
HF_HUB_OFFLINE=1. We use the ``ViT-L-14-quickgelu`` arch to match the OpenAI
weights' QuickGELU activation (avoids the open_clip mismatch warning). Override
via --model / --pretrained / --cache-dir.

Exposes ``load_clip()`` -> (model, preprocess, tokenizer, device) and ``score_dir``,
plus a CLI. open_clip 3.3.0 is already installed in venv-gpu.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

DEFAULT_CACHE = os.environ.get("OPEN_CLIP_CACHE", "/fsx/hf_cache/open_clip")
# quickgelu arch matches the OpenAI ViT-L/14 weights' activation.
DEFAULT_MODEL = "ViT-L-14-quickgelu"
DEFAULT_PRETRAINED = "openai"


def load_clip(model_name: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED,
              cache_dir: str = DEFAULT_CACHE):
    import open_clip
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, cache_dir=cache_dir)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, device


def load_evalset(path: str) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                out[e["id"]] = e
    return out


def score_dir(image_dir: str, evalset_path: str, batch_size: int = 32,
              model_name: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED,
              cache_dir: str = DEFAULT_CACHE, id_from_name=None) -> dict:
    import torch
    from PIL import Image
    import torch.nn.functional as F

    evalset = load_evalset(evalset_path)
    model, preprocess, tokenizer, device = load_clip(model_name, pretrained, cache_dir)

    def default_id(name: str) -> str:
        return Path(name).stem.split("__")[0]

    idfn = id_from_name or default_id
    files = sorted(glob.glob(os.path.join(image_dir, "*.png")) +
                   glob.glob(os.path.join(image_dir, "*.jpg")) +
                   glob.glob(os.path.join(image_dir, "*.webp")))

    items = []  # (file, id, prompt)
    for fp in files:
        eid = idfn(os.path.basename(fp))
        e = evalset.get(eid)
        if e is None:
            continue
        items.append((fp, eid, e["prompt"], e.get("band")))

    per_image = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            imgs = torch.stack([preprocess(Image.open(fp).convert("RGB"))
                                for fp, _, _, _ in chunk]).to(device)
            toks = tokenizer([p for _, _, p, _ in chunk]).to(device)
            img_f = F.normalize(model.encode_image(imgs).float(), dim=-1)
            txt_f = F.normalize(model.encode_text(toks).float(), dim=-1)
            cos = (img_f * txt_f).sum(dim=-1).cpu().tolist()
            for (fp, eid, prompt, band), c in zip(chunk, cos):
                per_image.append({
                    "file": os.path.basename(fp), "id": eid, "prompt": prompt,
                    "band": band, "cosine": c, "clipscore": 100.0 * max(c, 0.0),
                })

    def mean(key):
        vals = [r[key] for r in per_image]
        return (sum(vals) / len(vals)) if vals else None

    summary = {
        "model": f"{model_name}/{pretrained}",
        "device": device,
        "n_images": len(per_image),
        "mean_cosine": mean("cosine"),
        "mean_clipscore": mean("clipscore"),
    }
    return {"summary": summary, "per_image": per_image}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--evalset", default=str(Path(__file__).resolve().parents[1]
                                             / "prompts" / "anime_evalset.jsonl"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = score_dir(args.image_dir, args.evalset, batch_size=args.batch_size,
                    model_name=args.model, pretrained=args.pretrained,
                    cache_dir=args.cache_dir)
    print(json.dumps(res["summary"], indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[clip_score] wrote {args.out}")


if __name__ == "__main__":
    main()
