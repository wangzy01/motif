#!/usr/bin/env python3
"""Measure what the built plan ACTUALLY exposes, per sub-source, cluster and band.

Everything reported about the Stage512b mix so far is a POOL share or a
domain-level weight.  Neither is what the model sees.  The planner normalises per
(bucket, band) cell, and band eligibility is itself source-correlated -- short
captions concentrate in laion/coyo/cc12m -- so the realized mix can drift further
from the target than the domain-level figure suggests.

This reads the materialized plan row by row and counts.  No modelling, no sampling:
513,802,240 row references across 245 blocks, resolved through the catalog to a
sub-source, a cluster and a caption band.

The number to check it against is already known.  The explicit MONET sub-source
table is a target applied BEFORE the cluster factor with no renormalisation after,
so at cluster_beta=0.2 the DOMAIN-level realized mass was measured at L1 0.0767
from target (laion -7.9%, megalith10m +18.6%); at beta=0 it reproduces the table
exactly.  Evan accepted that drift on 2026-08-02.  What this answers is whether
per-cell normalisation makes it materially worse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BAND_NAMES = ("t1", "t2", "t3", "t4")


def domain_bounds(catalog_root: Path) -> list[tuple[int, int, str]]:
    header = json.loads((catalog_root / "header.json").read_text())
    bounds, cursor = [], 0
    for domain in header["domain_order"]:
        n = int(header["files"][domain]["rows"])
        bounds.append((cursor, cursor + n, domain))
        cursor += n
    return bounds


def load_domain_keys(catalog_root: Path, domain: str):
    """Return (source_code, cluster_id, source_names) arrays indexed by LOCAL row."""
    table = pq.read_table(
        catalog_root / f"{domain}.parquet", columns=["sub_source", "cluster_id"]
    )
    sub = table["sub_source"].to_pylist()
    names = sorted(set(sub))
    code = {name: i for i, name in enumerate(names)}
    return (
        np.fromiter((code[s] for s in sub), dtype=np.int16, count=len(sub)),
        table["cluster_id"].to_numpy(zero_copy_only=False).astype(np.int32),
        names,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-root", required=True, type=Path)
    ap.add_argument("--catalog-root", required=True, type=Path)
    ap.add_argument("--domain", default="monet_real")
    ap.add_argument("--sampling-config", default=None)
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="diagnostic: scan only the first N blocks")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    bounds = domain_bounds(args.catalog_root)
    lo, hi, _ = next(b for b in bounds if b[2] == args.domain)
    print(f"{args.domain}: catalog rows [{lo:,}, {hi:,})", flush=True)

    source_code, cluster_id, source_names = load_domain_keys(args.catalog_root, args.domain)
    n_sources = len(source_names)
    print(f"sub_sources: {source_names}", flush=True)

    header = json.loads((args.plan_root / "header.json").read_text())
    first_block = int(header["materialization"]["first_materialized_block"])
    n_blocks = int(header["num_blocks"])
    if args.max_blocks:
        n_blocks = min(n_blocks, first_block + args.max_blocks)

    src_exposures = np.zeros(n_sources, dtype=np.int64)
    src_band = np.zeros((n_sources, len(BAND_NAMES)), dtype=np.int64)
    # Fixed-size accumulator rather than a Counter: np.unique per block over ~2M
    # elements, 245 times, is far slower than one bincount into a 20k array.
    n_clusters = int(cluster_id.max()) + 1 if cluster_id.size else 0
    cluster_exposures = np.zeros(max(n_clusters, 1), dtype=np.int64)
    total_refs = 0
    domain_refs = 0

    for index in range(first_block, n_blocks):
        with np.load(args.plan_root / "blocks" / f"block_{index:06d}.npz",
                     allow_pickle=False) as archive:
            rows = np.asarray(archive["item_row"]).reshape(-1)
            bands = np.asarray(archive["caption_band"]).reshape(-1)
        total_refs += rows.size
        mask = (rows >= lo) & (rows < hi)
        if not mask.any():
            continue
        local = (rows[mask] - lo).astype(np.int64)
        codes = source_code[local]
        domain_refs += local.size
        src_exposures += np.bincount(codes, minlength=n_sources)
        # caption_band is 0-BASED, with -1 for "no band" (the non-real domains).
        # An earlier draft assumed 1-based, which silently dropped T1 and shifted
        # every other band down one -- and still produced a plausible-looking table.
        b = bands[mask].astype(np.int64)
        valid = b >= 0
        if valid.any():
            flat = codes[valid].astype(np.int64) * len(BAND_NAMES) + b[valid]
            src_band += np.bincount(
                flat, minlength=n_sources * len(BAND_NAMES)
            ).reshape(n_sources, len(BAND_NAMES))
        cl = cluster_id[local]
        cl = cl[cl >= 0]
        if cl.size:
            cluster_exposures += np.bincount(cl, minlength=cluster_exposures.size)
        if (index - first_block) % 40 == 0:
            print(f"  block {index}/{n_blocks}  domain refs so far {domain_refs:,}", flush=True)

    print(f"\ntotal plan row references {total_refs:,}; {args.domain} {domain_refs:,} "
          f"({100 * domain_refs / max(total_refs, 1):.3f}%)")

    target = {}
    if args.sampling_config:
        cfg = json.loads(Path(args.sampling_config).read_text())
        target = {
            k: float(v)
            for k, v in (cfg.get("monet_subsource_explicit_probs") or {}).items()
            if not k.startswith("_")
        }

    realized = src_exposures / max(src_exposures.sum(), 1)
    print(f"\n{'sub_source':<26}{'exposures':>16}{'realized':>11}{'target':>9}{'delta':>9}")
    l1 = 0.0
    for i, name in enumerate(source_names):
        t = target.get(name)
        d = (realized[i] - t) if t is not None else None
        if d is not None:
            l1 += abs(d)
        print(f"{name:<26}{src_exposures[i]:>16,}{realized[i]:>11.4f}"
              f"{(f'{t:.4f}' if t is not None else '-'):>9}"
              f"{(f'{d:+.4f}' if d is not None else '-'):>9}")
    if target:
        print(f"{'L1 from target':<26}{'':>16}{'':>11}{'':>9}{l1:>9.4f}")

    # Whole-plan band mix, as a check that band_mix_real is realized at all.
    band_tot = src_band.sum(axis=0)
    if band_tot.sum():
        share = band_tot / band_tot.sum()
        print("\nband mix over all real rows: " + "  ".join(
            f"{n}={100 * v:.2f}%" for n, v in zip(BAND_NAMES, share)))
    print(f"\n{'sub_source':<26}" + "".join(f"{b:>9}" for b in BAND_NAMES))
    for i, name in enumerate(source_names):
        row = src_band[i]
        tot = max(row.sum(), 1)
        print(f"{name:<26}" + "".join(f"{100 * v / tot:>8.2f}%" for v in row))

    counts = np.sort(cluster_exposures[cluster_exposures > 0].astype(np.float64))[::-1]
    eff = None
    if counts.size:
        p = counts / counts.sum()
        eff = float(np.exp(-(p * np.log(p)).sum()))
        print(f"\nclusters exposed {counts.size:,}   effective (exp H) {eff:,.0f}   "
              f"top-1 {100 * p[0]:.4f}%   top-10 {100 * p[:10].sum():.4f}%")

    summary = {
        "plan_root": str(args.plan_root),
        "domain": args.domain,
        "blocks_scanned": n_blocks - first_block,
        "total_row_references": int(total_refs),
        "domain_row_references": int(domain_refs),
        "source_names": source_names,
        "source_exposures": src_exposures.tolist(),
        "source_realized": realized.tolist(),
        "source_target": target,
        "l1_from_target": l1 if target else None,
        "source_band_counts": src_band.tolist(),
        "clusters_exposed": int(counts.size),
        "effective_clusters": eff,
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"wrote {args.json_out}")
    print("PLAN_REALIZED_EXPOSURE_AUDIT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
