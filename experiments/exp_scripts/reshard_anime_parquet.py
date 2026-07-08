#!/usr/bin/env python3
"""Reshard the anime source parquet (row-level round-robin) into M finer shards.

WHY: training/dataline/manifest_index.py:shard_subset_for_rank does an INTEGER-SHARD
split (`i % world == rank`). The anime source is only 17 parquet, so at world>17 some
ranks get ZERO anime shards -> domain_mix_renorm drops anime for them -> global anime
ratio falls below target (worst at 512/1024 where anime is weighted 0.65/0.80).

This writes M (e.g. 64) shards where each row is round-robin assigned, so every output
shard is a representative sample of the whole pool -> any world<=M is skew-free.

Streaming (row-group batches) so peak memory stays small. Writes to a NEW path; the
running 256 runs are unaffected (they read the original via the live sampling config).
Point a 512-specific sampling config's anime_caption_source at --out-dir to use it.

NOT run automatically. Trigger only if 512 will use >16 GPUs (else cap world<=16).
"""
from __future__ import annotations
import argparse, glob, os, time
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-glob", default="/s3/evan/anime_core_ditv04_ct_soft/data/*.parquet")
    ap.add_argument("--out-dir", required=True, help="NEW dir for resharded parquet (not the source)")
    ap.add_argument("--num-shards", type=int, default=64)
    ap.add_argument("--batch-rows", type=int, default=20000)
    args = ap.parse_args()

    srcs = sorted(glob.glob(args.src_glob))
    assert srcs, f"no source parquet at {args.src_glob}"
    os.makedirs(args.out_dir, exist_ok=True)
    schema = pq.ParquetFile(srcs[0]).schema_arrow
    M = args.num_shards
    writers = [pq.ParquetWriter(os.path.join(args.out_dir, f"{i:08d}.parquet"), schema)
               for i in range(M)]
    print(f"[reshard] {len(srcs)} src -> {M} shards @ {args.out_dir}", flush=True)

    cursor = 0  # global round-robin cursor across all rows
    total = 0
    t0 = time.perf_counter()
    for si, src in enumerate(srcs):
        pf = pq.ParquetFile(src)
        for batch in pf.iter_batches(batch_size=args.batch_rows):
            n = batch.num_rows
            tbl = pa.Table.from_batches([batch])
            # assign rows to shards by (cursor+row_idx) % M, then write each shard slice
            for m in range(M):
                # indices in this batch that go to shard m
                idx = [r for r in range(n) if (cursor + r) % M == m]
                if idx:
                    writers[m].write_table(tbl.take(pa.array(idx)))
            cursor += n
            total += n
        print(f"[reshard] src {si+1}/{len(srcs)} done, rows={total} ({time.perf_counter()-t0:.0f}s)", flush=True)

    for w in writers:
        w.close()
    print(f"[reshard] DONE {total} rows -> {M} shards in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
