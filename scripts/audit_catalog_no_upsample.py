#!/usr/bin/env python
"""Prove that a built catalog can never upsample, by scanning every row.

The loader now refuses an upsampling pairing at two points
(``OfflineExactDataset._prepare_window`` before any payload is fetched, and
``_bucket_u8_for_item`` after decode).  Both are fail-closed: they abort a
training step.  This script answers the question those guards cannot answer
cheaply at runtime -- *does the catalog contain such a row at all* -- so a bad
catalog is caught at build time instead of at step N of a 250,000-step run.

Run it on every catalog before wiring it to a trainer::

    python scripts/audit_catalog_no_upsample.py \
        /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_catalog_stage512b_v1

Exit code is 0 only when zero rows would upsample.  Memory stays flat (~220 MB
observed on the 70M-row Stage512a catalog) because rows stream in batches and
only counters are retained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

BATCH_ROWS = 1_000_000
EXAMPLE_LIMIT = 10


def load_bucket_table(header: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (widths, heights) indexed by bucket_idx."""
    buckets = header["buckets"]
    size = max(int(b["idx"]) for b in buckets) + 1
    widths = np.zeros(size, dtype=np.int64)
    heights = np.zeros(size, dtype=np.int64)
    for b in buckets:
        widths[int(b["idx"])] = int(b["w"])
        heights[int(b["idx"])] = int(b["h"])
    return widths, heights


def audit_domain(path: Path, widths: np.ndarray, heights: np.ndarray) -> dict:
    """Scan one domain parquet; never materialises more than BATCH_ROWS rows."""
    reader = pq.ParquetFile(path)
    rows = 0
    offenders = 0
    max_scale = 1.0
    examples: list[dict] = []
    for batch in reader.iter_batches(
        batch_size=BATCH_ROWS, columns=["bucket_idx", "width", "height"]
    ):
        bucket_idx = batch["bucket_idx"].to_numpy(zero_copy_only=False).astype(np.int64)
        width = batch["width"].to_numpy(zero_copy_only=False).astype(np.int64)
        height = batch["height"].to_numpy(zero_copy_only=False).astype(np.int64)

        # An out-of-range bucket_idx would index the table silently (or wrap on
        # a negative), so it is an error in its own right rather than something
        # to clip into range.
        if bucket_idx.size and (bucket_idx.min() < 0 or bucket_idx.max() >= widths.size):
            raise SystemExit(
                f"{path.name}: bucket_idx outside the header bucket table "
                f"[{bucket_idx.min()}, {bucket_idx.max()}] vs table size {widths.size}"
            )

        target_w = widths[bucket_idx]
        target_h = heights[bucket_idx]
        bad = (width < target_w) | (height < target_h)
        rows += int(width.size)
        n_bad = int(bad.sum())
        offenders += n_bad
        if n_bad:
            scale = np.maximum(target_w[bad] / width[bad], target_h[bad] / height[bad])
            max_scale = max(max_scale, float(scale.max()))
            for i in np.flatnonzero(bad)[: EXAMPLE_LIMIT - len(examples)]:
                examples.append(
                    {
                        "native": f"{int(width[i])}x{int(height[i])}",
                        "bucket_idx": int(bucket_idx[i]),
                        "bucket": f"{int(target_w[i])}x{int(target_h[i])}",
                        "scale": round(
                            float(max(target_w[i] / width[i], target_h[i] / height[i])), 4
                        ),
                    }
                )
    return {
        "rows": rows,
        "upsampling_rows": offenders,
        "max_scale": max_scale,
        "examples": examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("catalog_root", type=Path)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    header_path = args.catalog_root / "header.json"
    if not header_path.exists():
        raise SystemExit(f"no header.json under {args.catalog_root}")
    header = json.loads(header_path.read_text())
    widths, heights = load_bucket_table(header)

    print(
        f"catalog  {args.catalog_root}\n"
        f"stage    {header.get('stage')}   image_size {header.get('image_size')}\n"
        f"buckets  {widths.size}   kept {len(header.get('kept_bucket_indices') or [])}"
        f"   floor {header.get('floor')}"
    )

    results: dict[str, dict] = {}
    total_rows = 0
    total_bad = 0
    # Trust the header's domain list rather than globbing, so a domain the
    # header claims but never wrote is a failure instead of a silent skip.
    for domain in header.get("domains") or header.get("domain_order") or []:
        path = args.catalog_root / f"{domain}.parquet"
        if not path.exists():
            raise SystemExit(f"header lists domain {domain!r} but {path.name} is missing")
        res = audit_domain(path, widths, heights)
        results[domain] = res
        total_rows += res["rows"]
        total_bad += res["upsampling_rows"]
        pct = 100.0 * res["upsampling_rows"] / max(res["rows"], 1)
        print(
            f"  {domain:<12} rows {res['rows']:>12,}"
            f"  upsampling {res['upsampling_rows']:>10,} ({pct:.4f}%)"
            f"  max scale {res['max_scale']:.4f}"
        )
        for ex in res["examples"]:
            print(f"      {ex}")

    print(f"\nTOTAL rows {total_rows:,}   upsampling rows {total_bad:,}")
    summary = {
        "catalog_root": str(args.catalog_root),
        "stage": header.get("stage"),
        "total_rows": total_rows,
        "upsampling_rows": total_bad,
        "per_domain": results,
        "pass": total_bad == 0,
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"wrote {args.json_out}")

    if total_bad:
        print("CATALOG_NO_UPSAMPLE_AUDIT_FAIL")
        return 1
    print("CATALOG_NO_UPSAMPLE_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
