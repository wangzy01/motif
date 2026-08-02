#!/usr/bin/env python3
"""Per-ROW reuse over the built plan: how many times each catalog row is actually shown.

Every exposure audit so far has been aggregate -- per sub-source, per cluster, per
band.  Those all average over rows, and averaging is exactly what hides the failure
mode worth worrying about at 512M exposures: a small set of rows shown hundreds of
times while the domain-level epoch count looks calm.  "anime sees ~9 epochs" is
compatible with a thousand rows seeing 300 and most seeing 4.

So this counts references per row_id and reports the distribution, not the mean:
percentiles over the rows that are actually used, how many rows are never used at
all, how concentrated the exposure mass is in the hottest 1%, and the hottest rows
by name so a suspicious one can be looked at.

Method note.  A global int32 counter over all 89,786,866 catalog rows is 359 MB,
which is cheaper than any per-block dictionary.  Per block it uses
np.unique(return_counts=True) over ~2.1M references and scatters that back, rather
than bincount(minlength=total): bincount would allocate an 89.8M-element temporary
245 times over, for a block that only touches 2M of them.

Findings go NEXT TO THE PLAN.  The catalog binds inputs.sampling_config_sha256 and
the loader refuses a runtime config whose digest differs, so writing conclusions into
the sampling config unbinds the catalog -- that mistake has been made once already.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PERCENTILES = (50, 75, 90, 95, 99, 99.9)


def domain_bounds(catalog_root: Path) -> list[tuple[int, int, str]]:
    header = json.loads((catalog_root / "header.json").read_text())
    bounds, cursor = [], 0
    for domain in header["domain_order"]:
        n = int(header["files"][domain]["rows"])
        bounds.append((cursor, cursor + n, domain))
        cursor += n
    return bounds, header


def summarise(counts: np.ndarray) -> dict:
    """Distribution over a domain's rows.

    Percentiles are reported over USED rows, not all rows.  With a domain whose pool
    is much larger than its exposure budget the median over all rows is 0, which is
    true and useless -- it says nothing about how hard the used rows are being reused.
    ``never_used`` carries the other half of the picture.
    """
    total = int(counts.size)
    exposures = int(counts.sum())
    used_mask = counts > 0
    used = int(used_mask.sum())
    used_counts = counts[used_mask]
    out = {
        "rows_in_domain": total,
        "rows_used": used,
        "rows_never_used": total - used,
        "exposures": exposures,
        "mean_repeats_over_used": (exposures / used) if used else 0.0,
        "epochs_over_whole_domain": (exposures / total) if total else 0.0,
    }
    if used:
        for p in PERCENTILES:
            out[f"p{p}".replace(".", "_")] = float(np.percentile(used_counts, p))
        out["max"] = int(used_counts.max())
        # Concentration: the single number that says whether a long tail of hot rows
        # exists.  If the hottest 1% of used rows carry ~1% of exposures the plan is
        # flat; if they carry 20% it is not.
        order = np.sort(used_counts)[::-1]
        for frac in (0.001, 0.01, 0.1):
            k = max(1, int(round(used * frac)))
            out[f"share_of_exposures_in_hottest_{frac:g}"] = float(
                order[:k].sum() / exposures
            )
        hist = np.bincount(np.minimum(used_counts, 64))
        out["histogram_1_to_64_plus"] = {
            str(i): int(hist[i]) for i in range(1, min(len(hist), 65)) if hist[i]
        }
    return out


def hottest_rows(catalog_root: Path, domain: str, counts: np.ndarray, top: int) -> list[dict]:
    """Name the hottest rows so a suspicious count can be inspected, not just noted."""
    if not counts.size or top <= 0:
        return []
    k = min(top, int((counts > 0).sum()))
    if k == 0:
        return []
    idx = np.argpartition(counts, -k)[-k:]
    idx = idx[np.argsort(counts[idx])[::-1]]

    reader = pq.ParquetFile(catalog_root / f"{domain}.parquet")
    schema = reader.schema_arrow.names
    columns = ["uid"] + [c for c in ("sub_source", "cluster_id", "bucket_idx") if c in schema]
    # Read only the row groups that hold these few rows.  monet_real.parquet is 40M
    # rows and its uid column alone is gigabytes; pulling the whole thing to name
    # fifteen of them would dwarf the audit itself.
    starts = np.cumsum([0] + [reader.metadata.row_group(g).num_rows
                              for g in range(reader.metadata.num_row_groups)])
    rows = []
    cache: dict[int, dict] = {}
    for local in idx:
        local = int(local)
        group = int(np.searchsorted(starts, local, side="right") - 1)
        if group not in cache:
            cache[group] = reader.read_row_group(group, columns=columns).to_pydict()
        offset = local - int(starts[group])
        entry = {"local_row": local, "repeats": int(counts[local])}
        for column in columns:
            entry[column] = cache[group][column][offset]
        rows.append(entry)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-root", required=True, type=Path)
    ap.add_argument("--catalog-root", required=True, type=Path)
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="diagnostic: scan only the first N materialized blocks")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    bounds, catalog_header = domain_bounds(args.catalog_root)
    total_rows = bounds[-1][1]
    print(f"catalog rows {total_rows:,}; domains "
          f"{[(d, hi - lo) for lo, hi, d in bounds]}", flush=True)

    plan_header = json.loads((args.plan_root / "header.json").read_text())
    upstream = plan_header.get("upstream_catalog_hash")
    if upstream and upstream != catalog_header.get("catalog_hash"):
        raise SystemExit(
            f"plan was built against a different catalog: plan={upstream!r} "
            f"catalog={catalog_header.get('catalog_hash')!r}"
        )
    first_block = int(plan_header["materialization"]["first_materialized_block"])
    n_blocks = int(plan_header["num_blocks"])
    if args.max_blocks:
        n_blocks = min(n_blocks, first_block + args.max_blocks)

    counts = np.zeros(total_rows, dtype=np.int32)
    total_refs = 0
    blocks_read = 0
    for index in range(first_block, n_blocks):
        with np.load(args.plan_root / "blocks" / f"block_{index:06d}.npz",
                     allow_pickle=False) as archive:
            rows = np.asarray(archive["item_row"]).reshape(-1)
        if rows.size:
            out_of_range = int(((rows < 0) | (rows >= total_rows)).sum())
            if out_of_range:
                raise SystemExit(
                    f"block {index}: {out_of_range} row references outside the catalog"
                )
            uniq, cnt = np.unique(rows, return_counts=True)
            counts[uniq] += cnt.astype(np.int32)
        total_refs += int(rows.size)
        blocks_read += 1
        if (index - first_block) % 40 == 0:
            print(f"  block {index}/{n_blocks}  refs so far {total_refs:,}", flush=True)

    if int(counts.sum()) != total_refs:
        raise SystemExit(
            f"counter identity broken: sum(counts)={int(counts.sum()):,} != "
            f"references={total_refs:,}"
        )
    print(f"\nblocks {blocks_read}, row references {total_refs:,}, "
          f"distinct rows used {int((counts > 0).sum()):,}", flush=True)

    report = {
        "plan_root": str(args.plan_root),
        "plan_hash": plan_header.get("plan_hash"),
        "catalog_hash": catalog_header.get("catalog_hash"),
        "blocks_read": blocks_read,
        "total_row_references": total_refs,
        "domains": {},
    }
    header = (f"{'domain':<12}{'rows':>12}{'used':>12}{'never':>12}{'exposures':>14}"
              f"{'mean':>8}{'p50':>7}{'p95':>7}{'p99':>7}{'p99.9':>8}{'max':>8}")
    print("\n" + header)
    print("-" * len(header))
    for lo, hi, domain in bounds:
        stats = summarise(counts[lo:hi])
        stats["hottest"] = hottest_rows(args.catalog_root, domain, counts[lo:hi], args.top)
        report["domains"][domain] = stats
        print(f"{domain:<12}{stats['rows_in_domain']:>12,}{stats['rows_used']:>12,}"
              f"{stats['rows_never_used']:>12,}{stats['exposures']:>14,}"
              f"{stats['mean_repeats_over_used']:>8.2f}"
              f"{stats.get('p50', 0):>7.0f}{stats.get('p95', 0):>7.0f}"
              f"{stats.get('p99', 0):>7.0f}{stats.get('p99_9', 0):>8.0f}"
              f"{stats.get('max', 0):>8,}")

    print(f"\n{'domain':<12}{'hottest 0.1%':>14}{'hottest 1%':>14}{'hottest 10%':>14}"
          "   (share of the domain's exposures)")
    for lo, hi, domain in bounds:
        stats = report["domains"][domain]
        print(f"{domain:<12}"
              f"{stats.get('share_of_exposures_in_hottest_0.001', 0):>13.2%} "
              f"{stats.get('share_of_exposures_in_hottest_0.01', 0):>13.2%} "
              f"{stats.get('share_of_exposures_in_hottest_0.1', 0):>13.2%}")

    for lo, hi, domain in bounds:
        rows = report["domains"][domain]["hottest"][:5]
        if rows:
            print(f"\nhottest {domain}:")
            for entry in rows:
                extra = " ".join(
                    f"{k}={entry[k]}" for k in ("sub_source", "cluster_id", "bucket_idx")
                    if k in entry
                )
                print(f"  x{entry['repeats']:<5} {str(entry.get('uid'))[:60]:<60} {extra}")

    if args.json_out:
        if args.json_out.exists():
            raise SystemExit(f"refusing to overwrite {args.json_out}")
        args.json_out.write_text(json.dumps(report, indent=1, sort_keys=True))
        print(f"\nwrote {args.json_out}")
    print("PLAN_ROW_REUSE_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
