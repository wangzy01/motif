#!/usr/bin/env python
"""Per-subset summary of the TextAtlas v1 sidecar manifests (for 512 mix design).

Streams each sidecar parquet (bounded memory), accumulates per-subset:
  rows, elig_512, elig_1024, missing_caption (caption1==""),
  caption1 char-length histogram (p50/p90/p99/max).
Prints a compact table. Read-only; no decisions.
"""
import glob
import os

import numpy as np
import pyarrow.parquet as pq

OUT_ROOT = "/fsx/evan/textatlas_manifest_v1"
SUBSETS = [
    "CleanTextSynth", "CoverBook", "LongWordsSubset-A", "LongWordsSubset-M",
    "PPT2Details", "PPT2Structured", "Paper2Text", "StyledTextSynth",
    "TextScenesHQ", "TextVisionBlend",
]


def summarize_subset(s):
    rows = e512 = e1024 = miss = 0
    caplens = []  # sampled caption1 char lengths
    shards = sorted(glob.glob(os.path.join(OUT_ROOT, s, "*.parquet")))
    for sh in shards:
        pf = pq.ParquetFile(sh)
        for b in pf.iter_batches(
                batch_size=100_000,
                columns=["elig_512", "elig_1024", "caption1"]):
            e5 = b.column("elig_512").to_numpy(zero_copy_only=False)
            e10 = b.column("elig_1024").to_numpy(zero_copy_only=False)
            c1 = b.column("caption1").to_pylist()
            rows += len(e5)
            e512 += int(e5.sum())
            e1024 += int(e10.sum())
            lens = np.fromiter((len(x) for x in c1), dtype=np.int32, count=len(c1))
            miss += int((lens == 0).sum())
            # sample to keep memory bounded
            caplens.append(lens[::20])
    cl = np.concatenate(caplens) if caplens else np.array([0])
    cl = cl[cl > 0]  # length percentiles over non-empty captions
    pct = (lambda p: int(np.percentile(cl, p)) if len(cl) else 0)
    return {
        "rows": rows, "e512": e512, "e1024": e1024, "miss": miss,
        "shards": len(shards),
        "cap_p50": pct(50), "cap_p90": pct(90), "cap_p99": pct(99),
        "cap_max": int(cl.max()) if len(cl) else 0,
    }


def main():
    print(f"{'subset':<20}{'shards':>7}{'rows':>12}{'elig512':>12}{'e512%':>7}"
          f"{'elig1024':>11}{'e1024%':>8}{'miss':>7}"
          f"{'cap_p50':>9}{'cap_p90':>9}{'cap_p99':>9}{'cap_max':>9}", flush=True)
    tot = {"rows": 0, "e512": 0, "e1024": 0, "miss": 0}
    for s in SUBSETS:
        r = summarize_subset(s)
        for k in tot:
            tot[k] += r[k]
        p512 = r["e512"] / max(r["rows"], 1) * 100
        p1024 = r["e1024"] / max(r["rows"], 1) * 100
        print(f"{s:<20}{r['shards']:>7}{r['rows']:>12,}{r['e512']:>12,}{p512:>6.1f}%"
              f"{r['e1024']:>11,}{p1024:>7.1f}%{r['miss']:>7,}"
              f"{r['cap_p50']:>9}{r['cap_p90']:>9}{r['cap_p99']:>9}{r['cap_max']:>9}",
              flush=True)
    p512 = tot["e512"] / max(tot["rows"], 1) * 100
    p1024 = tot["e1024"] / max(tot["rows"], 1) * 100
    print(f"{'TOTAL':<20}{'':>7}{tot['rows']:>12,}{tot['e512']:>12,}{p512:>6.1f}%"
          f"{tot['e1024']:>11,}{p1024:>7.1f}%{tot['miss']:>7,}", flush=True)


if __name__ == "__main__":
    main()
