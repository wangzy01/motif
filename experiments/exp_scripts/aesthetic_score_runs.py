#!/usr/bin/env python3
"""Score existing per-run eval sample images with the local aes-pixai aesthetic model.

Read-only over images already produced under eval_doc_4runs/<run>/. Siglip regression
head -> a scalar 'prettiness' score. Aesthetics != prompt-following (cannot alone decide
the ablation), but gives a cheap quantitative 'which version looks nicer' curve.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

AES = "/fsx/models/aes-pixai-1.2-large-xgb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True, help="dir with <run>/seed*_p*.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="+",
                    default=["v1_baseline", "b1_adapter", "b2_longskip", "b3_nonoise"])
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(AES)
    model = AutoModelForImageClassification.from_pretrained(AES, torch_dtype=torch.float32).to(dev).eval()
    res = {}
    for run in args.runs:
        files = sorted(f for f in glob.glob(f"{args.images_root}/{run}/*.png")
                       if "grid" not in os.path.basename(f))
        if not files:
            print(f"[aes] {run}: NO images", flush=True); res[run] = {"n": 0}; continue
        scores = []
        for i in range(0, len(files), 16):
            batch = [Image.open(f).convert("RGB") for f in files[i:i + 16]]
            inp = proc(images=batch, return_tensors="pt").to(dev)
            with torch.no_grad():
                logits = model(**inp).logits.squeeze(-1).float().cpu().numpy()
            scores.extend(np.atleast_1d(logits).tolist())
        res[run] = {"n": len(scores), "mean": round(float(np.mean(scores)), 4),
                    "std": round(float(np.std(scores)), 4),
                    "min": round(float(np.min(scores)), 3), "max": round(float(np.max(scores)), 3)}
        print(f"[aes] {run}: n={len(scores)} mean={res[run]['mean']} std={res[run]['std']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"model": AES, "runs": res}, open(args.out, "w"), ensure_ascii=False, indent=2)
    print("[aes] ->", args.out, flush=True)


if __name__ == "__main__":
    main()
