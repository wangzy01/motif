#!/usr/bin/env python3
"""B6 driver: for each sampled run dir, run tagger hit-rate + CLIP-score and
write a per-run JSON plus a cross-run comparison table (markdown + JSON).

Assumes images already sampled by sample_evalset.py under
<samples-dir>/<run_key>/. Read-only w.r.t. training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scorers"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True,
                    help="dir containing <run_key>/ subdirs of images")
    ap.add_argument("--evalset", default=str(HERE / "prompts" / "anime_evalset.jsonl"))
    ap.add_argument("--runs", nargs="+", default=["v1", "b1"])
    ap.add_argument("--tagger-backend", default="auto",
                    choices=["auto", "company", "mldanbooru"])
    ap.add_argument("--skip-tagger", action="store_true")
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out_dir) if args.out_dir else samples_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    tagger_score = clip_score = None
    if not args.skip_tagger:
        import tagger_score as tagger_score  # noqa: PLC0414
    if not args.skip_clip:
        import clip_score as clip_score  # noqa: PLC0414

    table = []
    for rk in args.runs:
        rdir = samples_dir / rk
        if not rdir.is_dir():
            print(f"[run] skip {rk}: no dir {rdir}", flush=True)
            continue
        row = {"run": rk}
        if tagger_score is not None:
            try:
                tr = tagger_score.score_dir(str(rdir), args.evalset,
                                            backend=args.tagger_backend)
                with open(out_dir / f"tagger_{rk}.json", "w", encoding="utf-8") as f:
                    json.dump(tr, f, ensure_ascii=False, indent=2)
                s = tr["summary"]
                row.update({"tagger_backend": s["backend"],
                            "general_hit": s["mean_general_hit"],
                            "character_hit": s["mean_character_hit"],
                            "copyright_hit": s["mean_copyright_hit"],
                            "overall_hit": s["mean_overall_hit"],
                            "overall_cov50": s.get("mean_overall_cov50"),
                            "overall_cov80": s.get("mean_overall_cov80"),
                            "character_cov50": s.get("mean_character_cov50")})
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                row["tagger_error"] = repr(e)
        if clip_score is not None:
            try:
                cr = clip_score.score_dir(str(rdir), args.evalset)
                with open(out_dir / f"clip_{rk}.json", "w", encoding="utf-8") as f:
                    json.dump(cr, f, ensure_ascii=False, indent=2)
                row.update({"clip_model": cr["summary"]["model"],
                            "mean_cosine": cr["summary"]["mean_cosine"],
                            "mean_clipscore": cr["summary"]["mean_clipscore"]})
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                row["clip_error"] = repr(e)
        table.append(row)

    with open(out_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "-"

    lines = ["# Anime prompt-following comparison (anime evalset)", "",
             "| run | general_hit | character_hit | copyright_hit | overall_hit | overall@.5 | overall@.8 | char@.5 | CLIP cos | CLIPScore |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['run']} | {fmt(r.get('general_hit'))} | "
                     f"{fmt(r.get('character_hit'))} | {fmt(r.get('copyright_hit'))} | "
                     f"{fmt(r.get('overall_hit'))} | {fmt(r.get('overall_cov50'))} | "
                     f"{fmt(r.get('overall_cov80'))} | {fmt(r.get('character_cov50'))} | "
                     f"{fmt(r.get('mean_cosine'))} | {fmt(r.get('mean_clipscore'))} |")
    if table:
        b = table[0]
        lines += ["", f"tagger backend: `{b.get('tagger_backend','-')}` · "
                  f"clip: `{b.get('clip_model','-')}`",
                  "", "recall@prod = frac target tags with prob>=production thr (g0.2/c0.35/p0.35); "
                  "cov@.5/.8 = frac target tags with tagger prob>0.5/0.8 (naifu concept_v2 aligned); "
                  "CLIPScore = 100·max(cos,0)."]
    md = out_dir / "comparison.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[run] wrote {md} and comparison.json", flush=True)


if __name__ == "__main__":
    main()
