"""Post-T1 GPIC validation and lightweight statistics.

This intentionally avoids reading the large embedding columns. It validates the
expected shard/row counts from parquet footers and scans only ok/aesthetic to
produce the full-run aesthetic cut points used by T5.
"""
import argparse
import glob
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq


S3 = '/s3-pixai-test-uw2/evan/mot_filtering_v1'
GPIC_PASS = f'{S3}/gpic_pass'
GPIC_BASE = f'{S3}/gpic_manifest_base'
REPORT = f'{S3}/reports/t1_gpic_pass_summary.json'
LOCAL_TMP = os.environ.get('MOT_LOCAL_TMP',
                           '/fsx/evan/mot_code/data_analysis/outputs/_tmp_parquet')


def is_s3_mount(path):
    return os.path.abspath(path).startswith('/s3-')


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


def quantiles_from_hist(hist, edges, qs):
    total = int(hist.sum())
    if total == 0:
        return {str(q): None for q in qs}
    cdf = np.cumsum(hist)
    out = {}
    for q in qs:
        target = q * total
        idx = int(np.searchsorted(cdf, target, side='left'))
        idx = min(max(idx, 0), len(edges) - 2)
        prev = cdf[idx - 1] if idx > 0 else 0
        in_bin = max(int(hist[idx]), 1)
        frac = min(max((target - prev) / in_bin, 0.0), 1.0)
        out[f'{q:.3f}'] = float(edges[idx] + frac * (edges[idx + 1] - edges[idx]))
    return out


def one_pass_file(args):
    path, edges = args
    pf = pq.ParquetFile(path)
    rows = pf.metadata.num_rows
    tab = pq.read_table(path, columns=['ok', 'aes_laion_improved'])
    ok = tab['ok'].to_numpy(zero_copy_only=False)
    aes = tab['aes_laion_improved'].to_numpy(zero_copy_only=False)
    finite = np.isfinite(aes)
    hist, _ = np.histogram(aes[finite], bins=edges)
    return {
        'file': os.path.basename(path),
        'rows': int(rows),
        'ok': int(np.count_nonzero(ok)),
        'not_ok': int(rows - np.count_nonzero(ok)),
        'aes_finite': int(np.count_nonzero(finite)),
        'aes_nan': int(rows - np.count_nonzero(finite)),
        'aes_min': float(np.nanmin(aes)) if finite.any() else None,
        'aes_max': float(np.nanmax(aes)) if finite.any() else None,
        'hist': hist.astype(np.int64),
    }


def file_rows(path):
    return int(pq.ParquetFile(path).metadata.num_rows)


def embedding_spotcheck(files, per_file_rows):
    checks = []
    for f in files:
        n = min(per_file_rows, pq.ParquetFile(f).metadata.num_rows)
        if n <= 0:
            continue
        tab = pq.read_table(f, columns=['sscd', 'clip_l', 'clip_b']).slice(0, n)
        item = {'file': os.path.basename(f), 'rows_checked': int(n)}
        for col, dim in [('sscd', 512), ('clip_l', 768), ('clip_b', 512)]:
            vals = tab[col].to_pylist()
            want = dim * 2
            bad = sum(1 for v in vals if v is None or len(v) != want)
            item[f'{col}_bad_len'] = int(bad)
        checks.append(item)
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpic-pass', default=GPIC_PASS)
    ap.add_argument('--gpic-base', default=GPIC_BASE)
    ap.add_argument('--out', default=REPORT)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--expected-shards', type=int, default=8000)
    ap.add_argument('--expected-rows', type=int, default=100_000_000)
    ap.add_argument('--spotcheck-files', type=int, default=16)
    ap.add_argument('--spotcheck-rows', type=int, default=64)
    a = ap.parse_args()

    t0 = time.time()
    pass_files = sorted(glob.glob(f'{a.gpic_pass}/gpic_pass_*.parquet'))
    base_files = sorted(glob.glob(f'{a.gpic_base}/gpic_base_*.parquet'))
    if len(pass_files) != a.expected_shards:
        raise SystemExit(f'gpic_pass shards={len(pass_files)}, expected {a.expected_shards}')
    if len(base_files) != a.expected_shards:
        raise SystemExit(f'gpic_base shards={len(base_files)}, expected {a.expected_shards}')

    # A wide, fine-grained histogram is enough for deciding 10/15% aesthetic cuts.
    edges = np.linspace(-5.0, 15.0, 20001, dtype=np.float64)
    hist = np.zeros(len(edges) - 1, dtype=np.int64)
    totals = {'rows': 0, 'ok': 0, 'not_ok': 0, 'aes_finite': 0, 'aes_nan': 0}
    aes_min, aes_max = None, None
    with ProcessPoolExecutor(a.workers) as ex:
        for i, st in enumerate(ex.map(one_pass_file, [(f, edges) for f in pass_files], chunksize=8)):
            for k in totals:
                totals[k] += st[k]
            hist += st['hist']
            if st['aes_min'] is not None:
                aes_min = st['aes_min'] if aes_min is None else min(aes_min, st['aes_min'])
                aes_max = st['aes_max'] if aes_max is None else max(aes_max, st['aes_max'])
            if (i + 1) % 400 == 0:
                el = time.time() - t0
                print(f'{i+1}/{len(pass_files)} rows={totals["rows"]:,} '
                      f'ok={totals["ok"]:,} eta={(len(pass_files)-i-1)*el/(i+1)/60:.0f}min',
                      flush=True)

    with ProcessPoolExecutor(a.workers) as ex:
        base_rows = sum(ex.map(file_rows, base_files, chunksize=32))

    if totals['rows'] != a.expected_rows:
        raise SystemExit(f'gpic_pass rows={totals["rows"]}, expected {a.expected_rows}')
    if base_rows != a.expected_rows:
        raise SystemExit(f'gpic_base rows={base_rows}, expected {a.expected_rows}')

    step = max(1, len(pass_files) // max(1, a.spotcheck_files))
    spot_files = pass_files[::step][:a.spotcheck_files]
    emb_checks = embedding_spotcheck(spot_files, a.spotcheck_rows)

    qs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 0.99]
    report = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'gpic_pass_dir': a.gpic_pass,
        'gpic_base_dir': a.gpic_base,
        'shards': len(pass_files),
        'base_shards': len(base_files),
        'rows': totals['rows'],
        'base_rows': int(base_rows),
        'ok': totals['ok'],
        'not_ok': totals['not_ok'],
        'ok_rate': totals['ok'] / totals['rows'],
        'aes_finite': totals['aes_finite'],
        'aes_nan': totals['aes_nan'],
        'aes_min': aes_min,
        'aes_max': aes_max,
        'aes_quantiles': quantiles_from_hist(hist, edges, qs),
        'embedding_spotcheck': emb_checks,
        'elapsed_sec': round(time.time() - t0, 3),
    }
    write_json_checked(report, a.out)
    print(json.dumps({
        'rows': report['rows'],
        'ok_rate': report['ok_rate'],
        'aes_q10': report['aes_quantiles']['0.100'],
        'aes_q15': report['aes_quantiles']['0.150'],
        'out': a.out,
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
