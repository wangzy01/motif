"""Ground-truth near-duplicate check for MONET internal dedup (cuVS-independent).

Loads the SAME candidate pool used by t2_monet_internal_cuvs_dedup.py
(keep & !synthetic & sscd_nn >= thr), draws a random sample of query rows, and
computes the EXACT cosine nearest neighbour (excluding self) by brute-force
matmul against the full candidate set. This does NOT use cuVS/IVF, so it is a
trustworthy oracle: a correct ANN dedup run's rows_with_neighbor / candidate_rows
must match the sampled rate here (within sampling noise). If brute force finds a
healthy >=thr near-dup rate but the ANN run reported ~0, the ANN run is broken.

Runs on GPU via cupy when available, else falls back to numpy (CPU, slow).
"""
import argparse
import glob
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


S3 = '/s3-pixai-test-uw2/evan/mot_filtering_v1'
MONET_EMB = f'{S3}/monet_emb'
MONET_MANIFEST = f'{S3}/monet_manifest'
OUT_DIR = f'{S3}/t2_monet_internal_bruteforce_check'
LOCAL_TMP = os.environ.get(
    'MOT_LOCAL_TMP',
    '/fsx/evan/mot_code/data_analysis/outputs/_tmp_parquet',
)


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


def selected_files(files, max_files):
    if max_files <= 0 or max_files >= len(files):
        return files
    step = len(files) / max_files
    return [files[int(i * step)] for i in range(max_files)]


def bytes_to_matrix(vals, dim, out_dtype):
    arr = np.empty((len(vals), dim), dtype=out_dtype)
    good = np.zeros(len(vals), dtype=bool)
    want = dim * 2
    nout = 0
    for i, v in enumerate(vals):
        if v is None or len(v) != want:
            continue
        x = np.frombuffer(v, dtype=np.float16)
        if x.size != dim:
            continue
        xf = x.astype(np.float32)
        norm = np.linalg.norm(xf)
        if not np.isfinite(norm) or norm <= 0:
            continue
        arr[nout] = (xf / norm).astype(out_dtype, copy=False)
        good[i] = True
        nout += 1
    return arr[:nout], good


def load_candidates(files, manifest_root, thr, out_dtype):
    ids, sources, embs = [], [], []
    total_rows = 0
    cand_rows = 0
    t0 = time.time()
    for i, f in enumerate(files):
        base = os.path.basename(f)
        mf = f'{manifest_root}/{base}'
        if not os.path.exists(mf):
            raise FileNotFoundError(f'missing manifest for {f}: {mf}')
        man = pq.read_table(mf, columns=['keep', 'synthetic', 'sscd_nn'])
        keep = man['keep'].to_numpy(zero_copy_only=False).astype(bool)
        syn = man['synthetic'].to_numpy(zero_copy_only=False).astype(bool)
        nn = man['sscd_nn'].to_numpy(zero_copy_only=False)
        mask = keep & ~syn & (nn >= thr)
        total_rows += man.num_rows
        if mask.any():
            tab = pq.read_table(f, columns=['id', 'source', 'sscd'])
            mask = mask[:tab.num_rows]
            idx = np.flatnonzero(mask)
            vals = np.asarray(tab['sscd'].to_pylist(), dtype=object)[idx]
            x, good = bytes_to_matrix(vals, 512, out_dtype)
            if len(x):
                ids.append(np.asarray(tab['id'].to_pylist(), dtype=object)[idx][good])
                sources.append(np.asarray(tab['source'].to_pylist(), dtype=object)[idx][good])
                embs.append(x)
                cand_rows += int(len(x))
        if (i + 1) % 500 == 0 or (i + 1) == len(files):
            el = time.time() - t0
            print(f'load {i+1}/{len(files)} cand={cand_rows:,} '
                  f'eta={(len(files)-i-1)*el/max(i+1,1)/60:.0f}min', flush=True)
    if not embs:
        raise SystemExit('no candidates loaded')
    xb = np.concatenate(embs, axis=0).astype(out_dtype, copy=False)
    return {
        'id': np.concatenate(ids),
        'source': np.concatenate(sources),
        'xb': xb,
        'total_rows_seen': int(total_rows),
    }


def get_array_module():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return cp, 'cupy'
    except Exception as e:  # noqa: BLE001
        print(f'cupy unavailable ({e}); falling back to numpy CPU (slow)', flush=True)
        return np, 'numpy'


def to_host(arr, name):
    if name == 'cupy':
        import cupy as cp
        return cp.asnumpy(arr)
    return np.asarray(arr)


def brute_force_nn(xb, qpos_all, query_batch, db_chunk, xp, name):
    n = xb.shape[0]
    xb_g = xp.asarray(xb)
    m = len(qpos_all)
    best = np.full(m, -2.0, dtype=np.float32)
    best_j = np.full(m, -1, dtype=np.int64)
    t0 = time.time()
    for qs in range(0, m, query_batch):
        qe = min(m, qs + query_batch)
        qpos = qpos_all[qs:qe]
        q_g = xb_g[xp.asarray(qpos)]
        cur_best = xp.full(qe - qs, -2.0, dtype=xp.float32)
        cur_arg = xp.full(qe - qs, -1, dtype=xp.int64)
        for ds in range(0, n, db_chunk):
            de = min(n, ds + db_chunk)
            sims = (q_g @ xb_g[ds:de].T).astype(xp.float32)
            in_rng = (qpos >= ds) & (qpos < de)
            r = np.flatnonzero(in_rng)
            if len(r):
                c = (qpos[r] - ds).astype(np.int64)
                sims[xp.asarray(r), xp.asarray(c)] = -2.0
            cmax = sims.max(axis=1)
            carg = sims.argmax(axis=1) + ds
            upd = cmax > cur_best
            cur_best = xp.where(upd, cmax, cur_best)
            cur_arg = xp.where(upd, carg, cur_arg)
        best[qs:qe] = to_host(cur_best, name)
        best_j[qs:qe] = to_host(cur_arg, name)
        if name == 'cupy':
            xp.cuda.Stream.null.synchronize()
        el = time.time() - t0
        print(f'bruteforce {qe}/{m} eta={(m-qe)*el/max(qe,1)/60:.1f}min', flush=True)
    return best, best_j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--monet-emb', default=MONET_EMB)
    ap.add_argument('--monet-manifest', default=MONET_MANIFEST)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--thr', type=float, default=0.70)
    ap.add_argument('--sample', type=int, default=20000)
    ap.add_argument('--query-batch', type=int, default=4096)
    ap.add_argument('--db-chunk', type=int, default=500000)
    ap.add_argument('--max-files', type=int, default=0)
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()

    t0 = time.time()
    out_dtype = np.float32
    files = sorted(glob.glob(f'{args.monet_emb}/*.parquet'))
    if not files:
        raise SystemExit(f'no MONET embedding parquet in {args.monet_emb}')
    files = selected_files(files, args.max_files)

    meta = load_candidates(files, args.monet_manifest, args.thr, out_dtype)
    xb = meta['xb']
    n = xb.shape[0]
    rng = np.random.default_rng(args.seed)
    m = min(args.sample, n)
    qpos = np.sort(rng.choice(n, size=m, replace=False)).astype(np.int64)

    xp, name = get_array_module()
    print(f'candidates={n:,} sample={m:,} backend={name} thr={args.thr}', flush=True)
    best, best_j = brute_force_nn(xb, qpos, args.query_batch, args.db_chunk, xp, name)

    ge = best >= args.thr
    rate = float(ge.mean())
    pct = {p: float(np.percentile(best, p)) for p in (50, 90, 95, 99)}
    ids = meta['id']
    ex = np.flatnonzero(ge)[:50]
    examples = pd.DataFrame({
        'query_id': ids[qpos[ex]],
        'neighbor_id': ids[best_j[ex]],
        'query_source': meta['source'][qpos[ex]],
        'neighbor_source': meta['source'][best_j[ex]],
        'cos': best[ex].astype(np.float32),
    })
    write_parquet_checked(examples, f'{args.out_dir}/example_pairs.parquet')

    summary = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'backend': name,
        'candidate_rows': int(n),
        'sample': int(m),
        'thr': args.thr,
        'sample_rows_with_neighbor_ge_thr': int(ge.sum()),
        'sample_rate_ge_thr': rate,
        'estimated_total_rows_with_neighbor': int(round(rate * n)),
        'best_cos_pctl': pct,
        'best_cos_max': float(best.max()),
        'total_rows_seen': meta['total_rows_seen'],
        'elapsed_sec': round(time.time() - t0, 3),
    }
    write_json_checked(summary, f'{args.out_dir}/summary.json')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
