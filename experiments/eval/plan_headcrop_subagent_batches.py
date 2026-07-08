#!/usr/bin/env python3
"""Plan subagent scoring batches from headcrop matrix manifests.

This script does not inspect or transform images. It only reads sampling
manifests and writes small JSON/Markdown batch instruction files so each
generated image can be assigned to exactly one visual-scoring subagent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_expected(manifest_paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        size = f"{manifest.get('width')}x{manifest.get('height')}"
        for run in manifest.get("runs", []):
            for image in run.get("images", []):
                path = str(image["path"])
                if path in seen:
                    raise SystemExit(f"duplicate image path in manifests: {path}")
                seen.add(path)
                rows.append({
                    "image": path,
                    "run": run.get("key", ""),
                    "prompt_id": image.get("id", ""),
                    "prompt_style": image.get("prompt_style", manifest.get("prompt_style", "")),
                    "composition_target": image.get("composition_target", ""),
                    "domain": image.get("domain", manifest.get("domain", "")),
                    "size": size,
                    "seed": image.get("seed"),
                    "batch_seed": image.get("batch_seed"),
                    "ckpt_step": run.get("ckpt_step"),
                    "exposure_estimate": run.get("exposure_estimate"),
                    "prompt": image.get("prompt", ""),
                    "manifest": manifest_path,
                })
    return rows


def write_batch(out_dir: Path, batch_index: int, rows: list[dict]) -> dict:
    batch_id = f"batch_{batch_index:03d}"
    json_path = out_dir / f"{batch_id}.json"
    md_path = out_dir / f"{batch_id}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"batch_id": batch_id, "images": rows}, f, indent=2, ensure_ascii=False)

    lines = [
        f"# Headcrop Subagent Scoring {batch_id}",
        "",
        "Inspect every image at native resolution. Return JSONL only, one object per image.",
        "Use these fields exactly: image, run, prompt_id, prompt_style, composition_target, size, seed, primary_label, head_cut, foot_cut, side_cut, subject_complete_0to5, composition_follow_0to5, notes.",
        "",
        "Labels: A_full_ok = requested full-body/group is complete; B_head_cut = relevant head/top is clipped; B_foot_cut = relevant feet/lower legs are clipped; B_side_cut = relevant body/person is clipped at left/right; C_reasonable_crop = non-full-body or portrait crop is reasonable; D_other_bad = severe non-crop failure.",
        "Only count cuts when composition_target expects full-body, multi-person, or lower-body readability. For upper-body/non-full-body controls, use C_reasonable_crop unless the head itself is accidentally cut.",
        "Scores: subject_complete_0to5 and composition_follow_0to5 are integers/floats from 0 to 5, where 5 means complete and well-followed, 0 means unusable.",
        "",
        "Images:",
    ]
    for row in rows:
        lines.append(
            "- {image} | run={run} prompt_id={prompt_id} prompt_style={prompt_style} "
            "target={composition_target} size={size} seed={seed} batch_seed={batch_seed}".format(**row)
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return {"batch_id": batch_id, "json": str(json_path), "md": str(md_path), "n": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=18)
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_expected(args.manifests)
    rows.sort(key=lambda r: (r["run"], r["domain"], r["prompt_style"], r["size"], int(r["seed"]), r["prompt_id"]))

    batches = []
    for idx, start in enumerate(range(0, len(rows), args.batch_size), 1):
        batches.append(write_batch(out_dir, idx, rows[start:start + args.batch_size]))

    summary = {
        "n_images": len(rows),
        "batch_size": args.batch_size,
        "n_batches": len(batches),
        "batches": batches,
    }
    with open(out_dir / "subagent_batch_plan.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(batches)} batches for {len(rows)} images -> {out_dir}")


if __name__ == "__main__":
    main()
