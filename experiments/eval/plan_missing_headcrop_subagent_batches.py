#!/usr/bin/env python3
"""Plan subagent batches only for images that do not have score rows yet."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from plan_headcrop_subagent_batches import load_expected, write_batch


def load_scored(score_paths: list[str]) -> set[str]:
    scored: set[str] = set()
    for pattern in score_paths:
        paths = [Path(p) for p in sorted(glob.glob(pattern))] if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        for path in paths:
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
                    image = obj.get("image")
                    if not image:
                        raise SystemExit(f"{path}:{lineno}: missing image")
                    scored.add(str(image))
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--scores", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=18)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_expected(args.manifests)
    scored = load_scored(args.scores)
    missing = [row for row in rows if row["image"] not in scored]
    missing.sort(key=lambda r: (r["run"], r["domain"], r["prompt_style"], r["size"], int(r["seed"]), r["prompt_id"]))

    batches = []
    for idx, start in enumerate(range(0, len(missing), args.batch_size), 1):
        batches.append(write_batch(out_dir, idx, missing[start:start + args.batch_size]))

    summary = {
        "n_expected": len(rows),
        "n_scored": len(scored),
        "n_missing": len(missing),
        "batch_size": args.batch_size,
        "n_batches": len(batches),
        "batches": batches,
    }
    with open(out_dir / "subagent_batch_plan.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "missing_images.txt", "w", encoding="utf-8") as f:
        for row in missing:
            f.write(row["image"] + "\n")
    print(
        f"expected={len(rows)} scored={len(scored)} missing={len(missing)} "
        f"batches={len(batches)} -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
