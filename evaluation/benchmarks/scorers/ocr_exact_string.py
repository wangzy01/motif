#!/usr/bin/env python3
"""OCR exact-string scorer for Stage512b.

Stage512b puts authoritative rendered-text ground truth into the prompt for the first
time (S512B-13).  Nothing in ``evaluation/benchmarks/scorers/`` could measure whether
that helped: there was ``clip_score.py`` and ``tagger_score.py`` and nothing else.  A
CLIP score cannot tell "GRAND OPENING" from "GRAMD OPENlNG", and those are exactly the
two outcomes this stage is trying to separate.

Two numbers, always reported together:

  exact             the rendered string equals the target, byte for byte
  exact_normalized  equal after NFKC + whitespace collapse -- the same fold the GT
                    sidecar applies to build ``gt_text_normalized``

The GAP between them is the measurement, not a rounding detail.  It is precisely the
class of error that is typographic rather than semantic: full-width vs half-width,
circled digits, curly vs straight quotes, a line break rendered as a space.  Collapsing
to one number would either count those as successes (if you only report normalized) or
bury a real capability gain under them (if you only report exact).

Targets come from ``gt_text_raw``, never ``gt_text_normalized``.  Training reads raw, so
scoring against normalized would forgive on the metric exactly what the model was taught
not to do -- and would silently report a model that renders half-width where the picture
had full-width as perfect.

Scope matches the GT sidecar: CleanTextSynth and TextScenesHQ only.  The other seven
TextAtlas subsets have no extractable exact string, and using the i1 caption's quoted
span as an answer key would score 26.5% of wrong answers as right (the captions are
73.5% exact against authoritative GT).

The OCR engine is injected, not chosen here.  Whichever it is, it is part of the
protocol: its identity and version go into the receipt, and changing it invalidates
comparison with every earlier snapshot just as surely as changing the prompts would.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


def normalize(text: str) -> str:
    """NFKC + whitespace collapse.  Byte-identical to the GT sidecar's ``_normalize``."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def score_one(target_raw: str, rendered: str) -> dict:
    target_raw = str(target_raw)
    rendered = str(rendered)
    tn, rn = normalize(target_raw), normalize(rendered)
    return {
        "exact": rendered == target_raw,
        "exact_normalized": rn == tn,
        # Similarity is a triage aid only.  A near-miss on a shop sign is still a wrong
        # sign; do not aggregate this into a headline number.
        "similarity": SequenceMatcher(None, tn, rn).ratio(),
        "target_chars": len(target_raw),
        "rendered_chars": len(rendered),
        "target_raw": target_raw,
        "rendered": rendered,
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    exact = sum(1 for r in rows if r["exact"])
    exact_norm = sum(1 for r in rows if r["exact_normalized"])
    return {
        "n": n,
        "exact": exact,
        "exact_rate": exact / n,
        "exact_normalized": exact_norm,
        "exact_normalized_rate": exact_norm / n,
        # The typographic-error count: right characters, wrong forms.
        "normalization_only_wins": exact_norm - exact,
        "mean_similarity": sum(r["similarity"] for r in rows) / n,
        "by_length_bucket": _by_length(rows),
    }


def _by_length(rows: list[dict]) -> dict:
    """Short strings are a different problem from long ones and must not be averaged.

    A model can be perfect on "EXIT" and hopeless past twenty characters; one aggregate
    rate hides the crossover, which is the thing you actually want to watch move.
    """
    buckets = {"1-8": [], "9-20": [], "21-40": [], "41+": []}
    for row in rows:
        n = row["target_chars"]
        key = "1-8" if n <= 8 else "9-20" if n <= 20 else "21-40" if n <= 40 else "41+"
        buckets[key].append(row)
    return {
        k: {
            "n": len(v),
            "exact_rate": (sum(1 for r in v if r["exact"]) / len(v)) if v else None,
            "exact_normalized_rate": (
                sum(1 for r in v if r["exact_normalized"]) / len(v)
            ) if v else None,
        }
        for k, v in buckets.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, type=Path,
                    help="JSONL with {id, gt_text_raw, rendered_text} per sample; "
                         "rendered_text is whatever the OCR engine read off the image")
    ap.add_argument("--ocr-engine", required=True,
                    help="engine identity and version, e.g. paddleocr-2.7.3-en. Part of "
                         "the protocol: changing it invalidates comparison with earlier "
                         "snapshots exactly as changing the prompts would")
    ap.add_argument("--checkpoint", required=True, help="checkpoint path or tag scored")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    missing = 0
    for line in args.predictions.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "gt_text_raw" not in record:
            raise SystemExit(
                f"record {record.get('id')!r} has no gt_text_raw. Score against RAW, "
                "not gt_text_normalized -- training reads raw, and scoring against the "
                "normalized column forgives on the metric exactly what the model was "
                "taught not to do."
            )
        rendered = record.get("rendered_text")
        if rendered is None:
            # An OCR engine that returned nothing is a failed render, not a missing
            # datapoint.  Dropping it would quietly inflate the rate.
            missing += 1
            rendered = ""
        row = score_one(record["gt_text_raw"], rendered)
        row["id"] = record.get("id")
        rows.append(row)

    summary = aggregate(rows)
    summary["ocr_engine"] = args.ocr_engine
    summary["checkpoint"] = args.checkpoint
    summary["empty_ocr_results_counted_as_wrong"] = missing

    print(f"checkpoint       {args.checkpoint}")
    print(f"ocr engine       {args.ocr_engine}")
    print(f"samples          {summary['n']:,}  (empty OCR results counted wrong: {missing})")
    print(f"exact            {summary['exact']:,}  {summary['exact_rate']:.4f}")
    print(f"exact normalized {summary['exact_normalized']:,}  "
          f"{summary['exact_normalized_rate']:.4f}")
    print(f"typographic-only {summary['normalization_only_wins']:,}   "
          "(right characters, wrong forms -- the gap IS the finding)")
    print(f"mean similarity  {summary['mean_similarity']:.4f}   (triage only, do not headline)")
    print(f"\n{'target length':<16}{'n':>8}{'exact':>10}{'exact_norm':>12}")
    for bucket, stats in summary["by_length_bucket"].items():
        if stats["n"]:
            print(f"{bucket:<16}{stats['n']:>8,}{stats['exact_rate']:>10.4f}"
                  f"{stats['exact_normalized_rate']:>12.4f}")

    if args.json_out:
        if args.json_out.exists():
            raise SystemExit(f"refusing to overwrite {args.json_out}")
        summary["per_sample"] = rows
        args.json_out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.json_out}")
    print("OCR_EXACT_STRING_SCORED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
