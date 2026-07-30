#!/usr/bin/env python3
"""Build a clean, self-contained DPG-Bench data file from the upstream
``dpg_bench.csv`` (TencentQQGYLab/ELLA).

Output: ``dpg_prompts.jsonl`` — one JSON object per *item_id* (1065 total):

    {
      "id": "partiprompts97",          # == prompt filename stem upstream
      "prompt": "An empty space ...",  # full DPG text prompt
      "questions": [
        {"qid": 1, "dependency": [0], "tuple": "entity - whole (space)",
         "l1": "entity", "l2": "entity - whole", "question": "Is there an empty space?"},
        ...
      ]
    }

Faithful to compute_dpg_bench.py's ``prepare_dpg_data``:
  * questions grouped by item_id, keyed by integer ``proposition_id`` (qid),
  * ``dependency`` parsed to a list of ints (parent qids; 0 == no parent),
  * ``tuple`` retained; L1 = tuple.split('-')[0], L2 = tuple.split('(')[0],
  * NOTE the upstream scorer SKIPS the very first CSV data row (``if i==0: continue``).
    We replicate that exactly so the question set matches the published protocol.

Run (any python with stdlib csv; no pandas needed):
    python build_dpg_data.py --csv dpg_bench.csv --out dpg_prompts.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(csv_path: Path, skip_first_row: bool = True) -> list[dict]:
    # raise the field-size limit: DPG prompts are long.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    items: dict[str, dict] = {}
    order: list[str] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(reader):
            if skip_first_row and i == 0:
                # upstream compute_dpg_bench.py drops the first data row.
                continue
            iid = line["item_id"]
            qid = int(line["proposition_id"])
            deps = [int(d.strip()) for d in line["dependency"].split(",")]
            tup = line["tuple"]
            l1 = tup.split("-")[0].strip()
            l2 = tup.split("(")[0].strip()
            q = {
                "qid": qid,
                "dependency": deps,
                "tuple": tup,
                "l1": l1,
                "l2": l2,
                "question": line["question_natural_language"],
            }
            if iid not in items:
                items[iid] = {"id": iid, "prompt": line["text"], "questions": []}
                order.append(iid)
            items[iid]["questions"].append(q)
    return [items[iid] for iid in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "dpg_bench.csv"))
    ap.add_argument("--out", default=str(HERE / "dpg_prompts.jsonl"))
    ap.add_argument("--keep-first-row", action="store_true",
                    help="do NOT drop the first CSV data row (upstream drops it)")
    args = ap.parse_args()

    data = build(Path(args.csv), skip_first_row=not args.keep_first_row)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    nq = sum(len(r["questions"]) for r in data)
    print(f"[build] {len(data)} items, {nq} questions -> {args.out}")


if __name__ == "__main__":
    main()
