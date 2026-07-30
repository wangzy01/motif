"""T2 CPU subtask: exact pHash GPIC -> MONET duplicate hits.

This is the deterministic pHash layer of cross-source dedup. It compares GPIC
T1 pHash values against MONET kept manifest pHash values and emits GPIC keys
that have an exact MONET pHash match. Hamming<=4 remains a separate optional
approximate step; this script intentionally does exact matches only.
"""
import argparse
import glob
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


S3 = '/s3-pixai-test-uw2/evan/mot_filtering_v1'
MONET_MANIFEST = f'{S3}/monet_manifest'
GPIC_PASS = f'{S3}/gpic_pass'
OUT = f'{S3}/t2_phash_exact'
LOCAL = '/fsx/evan/mot_code/data_analysis/outputs/t2_phash_exact'
LOCAL_TMP = os.environ.get('MOT_LOCAL_TMP',
                           '/fsx/evan/mot_code/data_analysis/outputs/_tmp_parquet')


HASH_BIN = f'{LOCAL}/monet_keep_hashes.uint64.bin'
HASH_META = f'{LOCAL}/monet_keep_hashes.meta.json'


HASH_MEMMAP = None


def is_s3_mount(path):
    return os.path.abspath(path).startswith('/s3-')


def write_parquet_checked(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_dir = LOCAL_TMP if is_s3_mount(path) else os.path.dirname(path)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, f'{os.path.basename(path)}.tmp.{os.getpid()}')
    try:
        df.to_parquet(tmp, index=False)
        if is_s3_mount(path):
            if os.path.exists(path):
                os.remove(path)
            shutil.copyfile(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json_checked(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_dir = LOCAL_TMP if is_s3_mount(path) else os.path.dirname(path)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, f'{os.path.basename(path)}.tmp.{os.getpid()}')
    try:
        with open(tmp, 'w') as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
            fh.write('\n')
        if is_s3_mount(path):
            if os.path.exists(path):
                os.remove(path)
            shutil.copyfile(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def parse_hex_u64_list(values):
    out = np.empty(len(values), dtype=np.uint64)
    n = 0
    bad = 0
    for v in values:
        if v is None:
            bad += 1
            continue
        s = str(v)
        if len(s) == 16:
            try:
                out[n] = int(s, 16)
                n += 1
                continue
            except ValueError:
                pass
        bad += 1
    return out[:n], bad


def build_monet_hashes(args):
    os.makedirs(LOCAL, exist_ok=True)
    if os.path.exists(HASH_BIN) and os.path.exists(HASH_META):
        with open(HASH_META) as fh:
            meta = json.load(fh)
        expected_bytes = int(meta['hashes']) * np.dtype(np.uint64).itemsize
        if os.path.getsize(HASH_BIN) == expected_bytes and meta.get('sorted'):
            print(f'reuse sorted MONET hash bin: {HASH_BIN} hashes={meta["hashes"]:,}', flush=True)
            return meta

    files = sorted(glob.glob(f'{args.monet_manifest}/*.parquet'))
    if not files:
        raise SystemExit(f'no MONET manifest parquet in {args.monet_manifest}')
    t0 = time.time()
    rows = kept = valid = bad = 0
    with open(HASH_BIN, 'wb') as out:
        for i, f in enumerate(files):
            cols = ['hash_perceptual', 'keep', 'synthetic']
            tab = pq.read_table(f, columns=cols)
            keep = tab['keep'].to_numpy(zero_copy_only=False).astype(bool)
            syn = tab['synthetic'].to_numpy(zero_copy_only=False).astype(bool)
            mask = keep if args.include_synthetic else (keep & ~syn)
            hashes = np.asarray(tab['hash_perceptual'].to_pylist(), dtype=object)[mask]
            arr, b = parse_hex_u64_list(hashes)
            arr.tofile(out)
            rows += tab.num_rows
            kept += int(mask.sum())
            valid += int(arr.size)
            bad += int(b)
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f'build {i+1}/{len(files)} rows={rows:,} kept={kept:,} '
                      f'valid_hash={valid:,} eta={(len(files)-i-1)*el/(i+1)/60:.0f}min',
                      flush=True)

    hashes = np.memmap(HASH_BIN, dtype=np.uint64, mode='r+')
    print(f'sorting {hashes.size:,} MONET hashes', flush=True)
    hashes.sort()
    unique = int(np.count_nonzero(np.r_[True, hashes[1:] != hashes[:-1]])) if hashes.size else 0
    del hashes

    meta = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'monet_manifest': args.monet_manifest,
        'include_synthetic': bool(args.include_synthetic),
        'rows_scanned': rows,
        'kept_rows_considered': kept,
        'hashes': valid,
        'bad_hashes': bad,
        'unique_hashes': unique,
        'sorted': True,
        'bin': HASH_BIN,
        'elapsed_sec': round(time.time() - t0, 3),
    }
    with open(HASH_META, 'w') as fh:
        json.dump(meta, fh, indent=2)
        fh.write('\n')
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def init_search(hash_bin):
    global HASH_MEMMAP
    HASH_MEMMAP = np.memmap(hash_bin, dtype=np.uint64, mode='r')


def search_one(args):
    path, out_dir = args
    shard = os.path.basename(path).replace('gpic_pass_', '').replace('.parquet', '')
    op = f'{out_dir}/phash_exact_hits_{shard}.parquet'
    if os.path.exists(op):
        try:
            return shard, pq.ParquetFile(op).metadata.num_rows, True
        except Exception:
            pass

    tab = pq.read_table(path, columns=['key', 'phash'])
    keys = tab['key'].to_pylist()
    hashes, bad = parse_hex_u64_list(tab['phash'].to_pylist())
    if len(hashes) != len(keys):
        # Preserve alignment for rare malformed hashes.
        vals = []
        keep_keys = []
        bad = 0
        for k, h in zip(keys, tab['phash'].to_pylist()):
            arr, b = parse_hex_u64_list([h])
            if b:
                bad += 1
                continue
            keep_keys.append(k)
            vals.append(arr[0])
        keys = keep_keys
        hashes = np.asarray(vals, dtype=np.uint64)

    idx = np.searchsorted(HASH_MEMMAP, hashes)
    in_range = idx < HASH_MEMMAP.size
    hit = np.zeros(hashes.shape[0], dtype=bool)
    if in_range.any():
        hit[in_range] = HASH_MEMMAP[idx[in_range]] == hashes[in_range]
    rows = pd.DataFrame({
        'key': np.asarray(keys, dtype=object)[hit],
        'shard': shard,
        'phash': [f'{int(x):016x}' for x in hashes[hit]],
        'drop_phash_exact': np.ones(int(hit.sum()), dtype=np.int8),
    })
    write_parquet_checked(rows, op)
    return shard, int(hit.sum()), False


def search_gpic(args):
    files = sorted(glob.glob(f'{args.gpic_pass}/gpic_pass_*.parquet'))
    if not files:
        raise SystemExit(f'no GPIC pass parquet in {args.gpic_pass}')
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    total_hits = 0
    reused = 0
    with ProcessPoolExecutor(args.workers, initializer=init_search, initargs=(HASH_BIN,)) as ex:
        for i, (shard, hits, was_reused) in enumerate(ex.map(search_one, [(f, args.out) for f in files], chunksize=8)):
            total_hits += hits
            reused += int(was_reused)
            if (i + 1) % 400 == 0:
                el = time.time() - t0
                print(f'search {i+1}/{len(files)} hits={total_hits:,} reused={reused} '
                      f'eta={(len(files)-i-1)*el/(i+1)/60:.0f}min',
                      flush=True)
    summary = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'gpic_pass': args.gpic_pass,
        'out': args.out,
        'shards': len(files),
        'hit_rows': total_hits,
        'reused_outputs': reused,
        'elapsed_sec': round(time.time() - t0, 3),
    }
    write_json_checked(summary, f'{args.out}/summary.json')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--monet-manifest', default=MONET_MANIFEST)
    ap.add_argument('--gpic-pass', default=GPIC_PASS)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--stage', choices=['all', 'build', 'search'], default='all')
    ap.add_argument('--include-synthetic', action='store_true')
    args = ap.parse_args()

    if args.stage in ('all', 'build'):
        build_monet_hashes(args)
    if args.stage in ('all', 'search'):
        if not os.path.exists(HASH_BIN):
            raise SystemExit(f'missing {HASH_BIN}; run --stage build first')
        search_gpic(args)


if __name__ == '__main__':
    main()
