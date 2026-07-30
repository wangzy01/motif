#!/usr/bin/env python3
"""Build the B6 anime prompt-following eval set (≈150 prompts) from the anime
source parquet (/s3/evan/anime_core_ditv04_ct_soft/data).

Read-only. Does NOT touch training or any file training reads. Output is a
single jsonl at experiments/eval/prompts/anime_evalset.jsonl.

Design
------
The anime training caption is dominated by the **tag route** (tag-only rows are
forced to tag_medium, and the per-stage NL weights still leave a large tag share).
So the eval prompts are built in tag style from the parquet's *processed* clean
danbooru tag lists (columns ``tags_general`` / ``tags_character`` /
``tags_copyright``), which is exactly the vocabulary the company anime tagger
emits — making concept hit-rate well-defined (predicted∩target / target).

Each prompt records the **target concepts** used for hit-rate:
  - ``target_general``  : the general tags rendered into the prompt
  - ``target_character``: the character tag(s) (if any)
  - ``target_copyright``: the series/IP tag(s) (if any)

Tag-frequency balancing
-----------------------
We do NOT just sample random rows (that over-represents head tags like
``1girl, solo, long_hair``). Instead we bucket candidate rows by their *rarest*
general tag's corpus frequency into [common / mid / rare] bands and draw a
balanced mix, and within selection we cap how many prompts any single general
tag may anchor. This yields a spread across common and long-tail concepts
without needing the full caption_build config.

The prompt string itself is rendered in the same tag-join style the loader uses
(see training/dataline/caption_build._tag_space_variant): characters + series +
general, comma-joined, underscores->spaces. CFG/length-cap/dropout are NOT
applied — the eval prompt must be a *clean, fully-specified* target so hit-rate
is meaningful.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

# General tags that are structural/uninformative for *concept* hit-rate — we
# still allow them in the prompt text (they help the model) but they do NOT
# count toward target_general (a model trivially gets "1girl"/"solo" right).
_NONCONCEPT_GENERAL = {
    "1girl", "1boy", "2girls", "3girls", "4girls", "5girls", "6+girls",
    "multiple_girls", "multiple_boys", "solo", "solo_focus",
    "looking_at_viewer", "simple_background", "white_background",
    "signature", "artist_name", "monochrome", "greyscale", "tagme",
    "commentary", "commentary_request", "english_commentary", "absurdres",
    "highres", "lowres", "translation_request", "bad_id", "bad_pixiv_id",
}


def _clean(t: str) -> str:
    return t.replace("_", " ").strip()


def load_rows(data_glob: str, max_files: int, per_file: int, seed: int) -> list[dict]:
    files = sorted(glob.glob(data_glob))
    if not files:
        raise SystemExit(f"no parquet at {data_glob}")
    files = files[:max_files]
    rng = random.Random(seed)
    cols = ["tags_general", "tags_character", "tags_copyright", "rating_bucket",
            "compound_aes_score"]
    rows: list[dict] = []
    for f in files:
        pf = pq.ParquetFile(f)
        # read a head slice per file (cheap, deterministic); pipeline's random
        # row-group sampling is for training balance, not needed for an evalset.
        tbl = next(pf.iter_batches(batch_size=per_file, columns=cols))
        rows.extend(
            {
                "tags_general": list(r["tags_general"] or []),
                "tags_character": list(r["tags_character"] or []),
                "tags_copyright": list(r["tags_copyright"] or []),
                "rating_bucket": r["rating_bucket"],
                "aes": r["compound_aes_score"],
            }
            for r in tbl.to_pylist()
        )
    rng.shuffle(rows)
    return rows


def general_freq(rows: list[dict]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        for t in r["tags_general"]:
            c[t] += 1
    return c


def render_prompt(r: dict, gen_keep: list[str]) -> str:
    """Tag-style prompt: [character], [series], general (comma-joined, spaces)."""
    parts: list[str] = []
    parts += [_clean(t) for t in r["tags_character"][:2]]
    parts += [_clean(t) for t in r["tags_copyright"][:1] if t and t != "original"]
    parts += [_clean(t) for t in gen_keep]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return ", ".join(out)


def build(rows: list[dict], n_prompts: int, seed: int,
          min_general: int, max_general: int, per_tag_cap: int) -> list[dict]:
    rng = random.Random(seed + 1)
    freq = general_freq(rows)
    if not freq:
        raise SystemExit("no general tags found")
    counts = sorted(freq.values())
    # band thresholds on the corpus general-tag frequency distribution
    lo = counts[len(counts) // 3]
    hi = counts[2 * len(counts) // 3]

    def band_of(row) -> str:
        concept = [t for t in row["tags_general"] if t not in _NONCONCEPT_GENERAL]
        if not concept:
            return "common"
        rarest = min(freq.get(t, 1) for t in concept)
        if rarest <= lo:
            return "rare"
        if rarest <= hi:
            return "mid"
        return "common"

    # bucket eligible rows (need enough concept tags to make a real prompt)
    buckets: dict[str, list[dict]] = {"common": [], "mid": [], "rare": []}
    for r in rows:
        concept = [t for t in r["tags_general"] if t not in _NONCONCEPT_GENERAL]
        if len(concept) < min_general:
            continue
        buckets[band_of(r)].append(r)

    # target a balanced mix across bands
    quota = {"rare": n_prompts * 40 // 100,
             "mid": n_prompts * 35 // 100}
    quota["common"] = n_prompts - quota["rare"] - quota["mid"]

    used_tag: Counter = Counter()
    out: list[dict] = []
    seen_prompts: set[str] = set()

    def try_take(r) -> bool:
        # drop structural tags + any general tag that is really this row's
        # character/series leaking into tags_general (pixiv-origin noise).
        own_named = set(r["tags_character"]) | set(r["tags_copyright"])
        concept = [t for t in r["tags_general"]
                   if t not in _NONCONCEPT_GENERAL and t not in own_named]
        # respect per-tag cap: skip if every concept tag is already saturated
        avail = [t for t in concept if used_tag[t] < per_tag_cap]
        if not avail:
            return False
        rng.shuffle(concept)
        # keep structural anchors first (1girl/solo) for prompt realism, then concept
        struct = [t for t in r["tags_general"] if t in _NONCONCEPT_GENERAL][:3]
        keep_concept = concept[:max_general]
        gen_keep = struct + keep_concept
        prompt = render_prompt(r, gen_keep)
        if not prompt or prompt in seen_prompts:
            return False
        # target = concept general tags actually rendered + character + series
        tgt_general = [t for t in keep_concept]
        tgt_char = [t for t in r["tags_character"][:2]]
        tgt_copy = [t for t in r["tags_copyright"][:1] if t and t != "original"]
        if not tgt_general and not tgt_char:
            return False
        seen_prompts.add(prompt)
        for t in keep_concept:
            used_tag[t] += 1
        out.append({
            "prompt": prompt,
            "target_general": tgt_general,
            "target_character": tgt_char,
            "target_copyright": tgt_copy,
            "rating": r["rating_bucket"],
            "band": band_of(r),
        })
        return True

    for band in ("rare", "mid", "common"):
        pool = buckets[band]
        rng.shuffle(pool)
        want = quota[band]
        got = 0
        for r in pool:
            if got >= want:
                break
            if try_take(r):
                got += 1
    # top up to n_prompts from any remaining rows if a band fell short
    if len(out) < n_prompts:
        leftover = [r for b in buckets.values() for r in b]
        rng.shuffle(leftover)
        for r in leftover:
            if len(out) >= n_prompts:
                break
            try_take(r)

    for i, e in enumerate(out):
        e["id"] = f"anime_eval_{i:03d}"
    return out[:n_prompts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", default="/s3/evan/anime_core_ditv04_ct_soft/data/*.parquet")
    ap.add_argument("--out", default=str(Path(__file__).with_name("anime_evalset.jsonl")))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-files", type=int, default=40)
    ap.add_argument("--per-file", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--min-general", type=int, default=4)
    ap.add_argument("--max-general", type=int, default=10)
    ap.add_argument("--per-tag-cap", type=int, default=6)
    args = ap.parse_args()

    rows = load_rows(args.data_glob, args.max_files, args.per_file, args.seed)
    entries = build(rows, args.n, args.seed, args.min_general, args.max_general,
                    args.per_tag_cap)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # report
    bands = Counter(e["band"] for e in entries)
    n_char = sum(1 for e in entries if e["target_character"])
    n_copy = sum(1 for e in entries if e["target_copyright"])
    avg_g = sum(len(e["target_general"]) for e in entries) / max(len(entries), 1)
    print(f"[evalset] wrote {len(entries)} prompts -> {out}")
    print(f"[evalset] bands={dict(bands)} with_char={n_char} with_copyright={n_copy} "
          f"avg_target_general={avg_g:.1f}")


if __name__ == "__main__":
    main()
