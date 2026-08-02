#!/usr/bin/env python
"""Compute GPIC's own within-domain cluster_sizes for Stage512b.

Why this exists
---------------
``phase5_build_catalog.real_derived_columns`` applies

    cluster_factor = cluster_sizes[c] ** (-cluster_beta)

to every domain in ``REAL_DOMAINS``, but ``cluster_sizes`` comes from
``compute_real_global_stats(monet_shards, ...)`` -- a MONET-only scan.  GPIC
shares MONET's cluster namespace (one joint CLIP-B k-means over MONET+GPIC,
k=20000), so the lookup *succeeds* and the mis-weighting is silent.

It is not merely arbitrary, it is inverted.  Measured over the Stage512b
eligible pool, ``corr(log MONET cluster size, log GPIC cluster size) = -0.24``:
GPIC's densest modes sit in MONET's sparsest clusters, so inverse-MONET-size
weighting systematically *up*-weights exactly what cluster flattening is meant
to suppress.  Effective clusters fall from 16,226 (uniform) to 13,559 and the
top-1 cluster share more than doubles.

Separately, 314 of the 20,000 cluster ids never appear in MONET at all.  For
those ``cluster_sizes.get(int(c), 1) ** -beta`` returns 1.0 -- the *maximum*
possible factor -- handing 1.31% of eligible GPIC rows 4.08% of GPIC's mass.

This script produces the within-domain table so phase5 can weight GPIC by its
own cluster sizes.  Convention follows the MONET cache exactly: counted over
the FULL domain pool (not the Stage512b-eligible subset), so the two tables
mean the same thing.  Both figures are recorded for reference.

Unlike the MONET cache -- three bare keys with no provenance -- the output
carries the identity of what it was computed over, so a later reader can tell
whether it still describes the pool being sampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.offline_arb.phase3a_bucket_stats import (  # noqa: E402
    load_sampling_config,
    list_shards,
)
from training.dataline import anchors_v1  # noqa: E402

K_CLUSTERS = 20_000
NO_CLUSTER = -1


def build(domain: str, shards: list[str], area_gate: int, crop_loss_max: float) -> dict:
    sizes_full = np.zeros(K_CLUSTERS, dtype=np.int64)
    sizes_eligible = np.zeros(K_CLUSTERS, dtype=np.int64)
    rows_total = 0
    rows_eligible = 0
    rows_no_cluster = 0
    rows_holdout = 0
    shard_fingerprints: list[str] = []
    out_of_range: dict[int, int] = {}

    for path in shards:
        table = pq.read_table(
            path, columns=["cluster_id", "width", "height", "holdout"]
        )
        cluster = table["cluster_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        width = table["width"].to_numpy(zero_copy_only=False).astype(np.int64)
        height = table["height"].to_numpy(zero_copy_only=False).astype(np.int64)
        holdout = table["holdout"].to_numpy(zero_copy_only=False).astype(np.int64)
        rows_total += cluster.size
        rows_holdout += int((holdout != 0).sum())

        # A cluster id outside [0, K) would silently corrupt the histogram (or
        # throw on bincount), so it is recorded rather than clipped.
        known = cluster != NO_CLUSTER
        bad = known & ((cluster < 0) | (cluster >= K_CLUSTERS))
        if bad.any():
            for value in np.unique(cluster[bad]):
                out_of_range[int(value)] = out_of_range.get(int(value), 0) + int(
                    (cluster == value).sum()
                )
            known &= ~bad
        rows_no_cluster += int((~known).sum())

        sizes_full += np.bincount(cluster[known], minlength=K_CLUSTERS)

        # The eligible subset is informational: it is what Stage512b actually
        # samples, so a large divergence from the full-pool table is worth
        # seeing even though the full-pool table is the one phase5 uses.
        anchor_id, _ = anchors_v1.assign_anchor_batch(
            width, height, crop_loss_max=crop_loss_max, area_gate=area_gate
        )
        elig = (anchor_id >= 0) & (holdout == 0)
        rows_eligible += int(elig.sum())
        sizes_eligible += np.bincount(cluster[known & elig], minlength=K_CLUSTERS)

        meta = pq.ParquetFile(path).metadata
        shard_fingerprints.append(f"{Path(path).name}:{meta.num_rows}")

    present = int((sizes_full > 0).sum())
    shard_sha = hashlib.sha256(
        "\n".join(sorted(shard_fingerprints)).encode()
    ).hexdigest()

    def effective_clusters(counts: np.ndarray) -> float:
        mass = counts[counts > 0].astype(np.float64)
        p = mass / mass.sum()
        return float(np.exp(-(p * np.log(p)).sum()))

    return {
        "cluster_sizes": {
            str(i): int(v) for i, v in enumerate(sizes_full) if v > 0
        },
        "_domain": domain,
        "_k_clusters": K_CLUSTERS,
        "_counted_over": "full domain pool, matching the MONET cache convention",
        "_rows_total": rows_total,
        "_rows_with_cluster": int(sizes_full.sum()),
        "_rows_no_cluster": rows_no_cluster,
        "_rows_holdout": rows_holdout,
        "_rows_stage512b_eligible": rows_eligible,
        "_clusters_present": present,
        "_clusters_absent": K_CLUSTERS - present,
        "_effective_clusters_full": round(effective_clusters(sizes_full), 1),
        "_effective_clusters_eligible": round(effective_clusters(sizes_eligible), 1),
        "_n_shards": len(shards),
        "_shard_names_sha256": shard_sha,
        "_manifest_root": str(Path(shards[0]).parent),
        "_area_gate": area_gate,
        "_crop_loss_max": crop_loss_max,
        "_out_of_range_cluster_ids": out_of_range,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sampling-config", required=True)
    ap.add_argument("--domain", default="gpic")
    ap.add_argument("--area-gate", type=int, default=262_144)
    ap.add_argument("--crop-loss-max", type=float, default=0.20)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if args.out.exists():
        raise SystemExit(
            f"refusing to overwrite an existing stats file: {args.out}\n"
            "these are immutable inputs; write a new version instead"
        )

    cfg = load_sampling_config(args.sampling_config)
    shards = list_shards(cfg)[args.domain]
    print(f"{args.domain}: {len(shards)} shards", flush=True)

    result = build(args.domain, shards, args.area_gate, args.crop_loss_max)

    for key in sorted(k for k in result if k.startswith("_")):
        value = result[key]
        if isinstance(value, dict) and len(value) > 8:
            value = f"<{len(value)} entries>"
        print(f"  {key:<34} {value}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=1, sort_keys=True)
    args.out.write_text(payload)
    args.out.chmod(0o444)
    print(f"\nwrote {args.out}")
    print(f"sha256 {hashlib.sha256(payload.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
