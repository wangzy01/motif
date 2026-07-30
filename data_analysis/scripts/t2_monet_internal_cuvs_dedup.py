"""T2 MONET internal near-duplicate drop list with cuVS IVF-Flat.

This replaces the older torch all-pairs style script, which is too slow for
millions of MONET candidates. It filters to kept real MONET rows with
sscd_nn >= threshold, builds a cuVS IVF-Flat index, searches each candidate
against the candidate set, unions neighbors above threshold, and keeps the
highest aesthetic / resolution item per connected component.
"""
import argparse
import glob
import json
import os
import shutil
import time
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


S3 = '/s3-pixai-test-uw2/evan/mot_filtering_v1'
MONET_EMB = f'{S3}/monet_emb'
MONET_MANIFEST = f'{S3}/monet_manifest'
OUT_DIR = f'{S3}/t2_monet_internal_cuvs'
LOCAL_TMP = os.environ.get(
    'MOT_LOCAL_TMP',
    '/fsx/evan/mot_code/data_analysis/outputs/_tmp_parquet',
)


class UF:
    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int64)

    def find(self, x):
        r = int(x)
        while self.p[r] != r:
            r = int(self.p[r])
        x = int(x)
        while self.p[x] != r:
            parent = int(self.p[x])
            self.p[x] = r
            x = parent
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb


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


def selected_files(files, max_files):
    if max_files <= 0 or max_files >= len(files):
        return files
    step = len(files) / max_files
    return [files[int(i * step)] for i in range(max_files)]


def bytes_to_matrix(vals, dim=512, out_dtype=np.float16):
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


def load_candidates(files, manifest_root, thr, include_synthetic, out_dtype):
    ids = []
    sources = []
    shards = []
    least_dim = []
    aes = []
    embs = []
    total_rows = 0
    cand_rows = 0
    bad_emb = 0
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
        mask = keep & (nn >= thr)
        if not include_synthetic:
            mask &= ~syn
        total_rows += man.num_rows
        if not mask.any():
            if (i + 1) % 200 == 0 or (i + 1) == len(files):
                el = time.time() - t0
                print(f'load candidates {i+1}/{len(files)} cand={cand_rows:,} '
                      f'rate={cand_rows/max(el,1):.0f}/s', flush=True)
            continue

        tab = pq.read_table(
            f,
            columns=['id', 'source', 'least_dimension', 'aes_jasperai', 'sscd'],
        )
        mask = mask[:tab.num_rows]
        idx = np.flatnonzero(mask)
        vals = np.asarray(tab['sscd'].to_pylist(), dtype=object)[idx]
        x, good = bytes_to_matrix(vals, 512, out_dtype)
        bad_emb += int(len(vals) - len(x))
        if len(x):
            ids.append(np.asarray(tab['id'].to_pylist(), dtype=object)[idx][good])
            sources.append(np.asarray(tab['source'].to_pylist(), dtype=object)[idx][good])
            shards.append(np.full(len(x), base, dtype=object))
            least_dim.append(tab['least_dimension'].to_numpy(zero_copy_only=False)[idx][good])
            aes.append(tab['aes_jasperai'].to_numpy(zero_copy_only=False)[idx][good])
            embs.append(x)
            cand_rows += int(len(x))
        if (i + 1) % 200 == 0 or (i + 1) == len(files):
            el = time.time() - t0
            print(f'load candidates {i+1}/{len(files)} cand={cand_rows:,} '
                  f'rate={cand_rows/max(el,1):.0f}/s eta={(len(files)-i-1)*el/max(i+1,1)/60:.0f}min',
                  flush=True)

    if not embs:
        raise SystemExit('no MONET internal dedup candidates loaded')
    t1 = time.time()
    xb = np.concatenate(embs, axis=0).astype(out_dtype, copy=False)
    out = {
        'id': np.concatenate(ids),
        'source': np.concatenate(sources),
        'shard': np.concatenate(shards),
        'least_dimension': np.concatenate(least_dim).astype(np.float32, copy=False),
        'aes_jasperai': np.concatenate(aes).astype(np.float32, copy=False),
        'xb': xb,
        'total_rows_seen': int(total_rows),
        'bad_embedding_rows': int(bad_emb),
    }
    print(f'candidate matrix ready shape={xb.shape} dtype={xb.dtype} '
          f'concat_sec={time.time()-t1:.1f}', flush=True)
    return out


class CuvsIndex:
    def __init__(self, xb, nlist, nprobe, kmeans_iters, train_fraction, multi_gpu):
        self.multi_gpu = multi_gpu
        self.nprobe = nprobe
        t0 = time.time()
        if multi_gpu:
            from cuvs.neighbors.mg import ivf_flat
            self.ivf_flat = ivf_flat
            params = ivf_flat.IndexParams(
                n_lists=nlist,
                metric='inner_product',
                kmeans_n_iters=kmeans_iters,
                kmeans_trainset_fraction=train_fraction,
                distribution_mode='sharded',
            )
            self.index = ivf_flat.build(params, xb)
            self.backend = 'cuvs_mg_ivf_flat'
        else:
            import cupy as cp
            from cuvs.neighbors import ivf_flat
            self.cp = cp
            self.ivf_flat = ivf_flat
            xg = cp.asarray(xb)
            params = ivf_flat.IndexParams(
                n_lists=nlist,
                metric='inner_product',
                kmeans_n_iters=kmeans_iters,
                kmeans_trainset_fraction=train_fraction,
            )
            self.index = ivf_flat.build(params, xg)
            cp.cuda.Stream.null.synchronize()
            self.backend = 'cuvs_single_ivf_flat'
        print(f'index backend={self.backend} nlist={nlist} nprobe={nprobe} '
              f'train_fraction={train_fraction:.6f} build_sec={time.time()-t0:.1f}',
              flush=True)

    def search(self, xq, topk):
        params = self.ivf_flat.SearchParams(n_probes=self.nprobe)
        if self.multi_gpu:
            scores, idx = self.ivf_flat.search(params, self.index, xq, topk)
            return np.asarray(scores, dtype=np.float32), np.asarray(idx, dtype=np.int64)
        qg = self.cp.asarray(xq)
        scores, idx = self.ivf_flat.search(params, self.index, qg, topk)
        self.cp.cuda.Stream.null.synchronize()
        return (
            self.cp.asnumpy(self.cp.asarray(scores)).astype(np.float32),
            self.cp.asnumpy(self.cp.asarray(idx)).astype(np.int64),
        )


def union_neighbors(index, xb, thr, topk, query_batch):
    n = xb.shape[0]
    uf = UF(n)
    edge_count = 0
    rows_with_neighbor = 0
    t0 = time.time()
    for qs in range(0, n, query_batch):
        qe = min(n, qs + query_batch)
        scores, nbrs = index.search(xb[qs:qe], topk + 1)
        global_i = np.arange(qs, qe, dtype=np.int64)[:, None]
        mask = (nbrs >= 0) & (nbrs != global_i) & (scores >= thr)
        rows_with_neighbor += int(mask.any(axis=1).sum())
        src_rows, dst_cols = np.nonzero(mask)
        edge_count += int(len(src_rows))
        for r, c in zip(src_rows, dst_cols):
            uf.union(qs + int(r), int(nbrs[r, c]))
        done = qe
        if done % max(query_batch * 10, 1) == 0 or done == n:
            el = time.time() - t0
            print(f'search internal {done}/{n} edges={edge_count:,} '
                  f'rows_with_neighbor={rows_with_neighbor:,} '
                  f'eta={(n-done)*el/max(done,1)/60:.0f}min',
                  flush=True)
    return uf, edge_count, rows_with_neighbor


def build_drop_frame(meta, uf):
    n = len(meta['id'])
    t0 = time.time()
    roots = np.empty(n, dtype=np.int64)
    for i in range(n):
        roots[i] = uf.find(i)
    csize = Counter(roots.tolist())
    aes = np.nan_to_num(meta['aes_jasperai'], nan=-1.0)
    ld = np.nan_to_num(meta['least_dimension'], nan=-1.0)
    order = np.lexsort((ld, aes))[::-1]
    best = {}
    for i in order:
        r = int(roots[i])
        if r not in best:
            best[r] = int(i)
    keep = np.zeros(n, dtype=bool)
    for i in best.values():
        keep[i] = True
    drop_idx = np.array(
        [i for i in range(n) if (not keep[i]) and csize[int(roots[i])] > 1],
        dtype=np.int64,
    )
    kept_idx = np.array([best[int(roots[i])] for i in drop_idx], dtype=np.int64)
    df = pd.DataFrame({
        'drop_id': meta['id'][drop_idx],
        'kept_id': meta['id'][kept_idx],
        'source': meta['source'][drop_idx],
        'shard': meta['shard'][drop_idx],
        'kept_source': meta['source'][kept_idx],
        'kept_shard': meta['shard'][kept_idx],
        'cluster_size': np.array([csize[int(roots[i])] for i in drop_idx], dtype=np.int32),
        'drop_sscd_internal': np.ones(len(drop_idx), dtype=np.int8),
    })
    multi_clusters = sum(1 for v in csize.values() if v > 1)
    print(f'components ready multi_clusters={multi_clusters:,} drops={len(df):,} '
          f'sec={time.time()-t0:.1f}', flush=True)
    return df, int(multi_clusters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--monet-emb', default=MONET_EMB)
    ap.add_argument('--monet-manifest', default=MONET_MANIFEST)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--thr', type=float, default=0.70)
    ap.add_argument('--topk', type=int, default=32)
    ap.add_argument('--nlist', type=int, default=65536)
    ap.add_argument('--nprobe', type=int, default=64)
    ap.add_argument('--kmeans-iters', type=int, default=12)
    ap.add_argument('--train-sample', type=int, default=4000000)
    ap.add_argument('--query-batch', type=int, default=8192)
    ap.add_argument('--index-dtype', choices=['float16', 'float32'], default='float16')
    ap.add_argument('--max-files', type=int, default=0)
    ap.add_argument('--multi-gpu', action='store_true')
    ap.add_argument('--include-synthetic', action='store_true')
    ap.add_argument('--cache-dir', default='',
                    help='if set, cache/reuse the loaded candidate matrix here '
                         '(FSX path) so the ~hours-long S3 load is paid only once')
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    out_dtype = np.float16 if args.index_dtype == 'float16' else np.float32
    files = sorted(glob.glob(f'{args.monet_emb}/*.parquet'))
    if not files:
        raise SystemExit(f'no MONET embedding parquet in {args.monet_emb}')
    files = selected_files(files, args.max_files)

    tag = f'thr{args.thr}_syn{int(args.include_synthetic)}_{args.index_dtype}'
    xb_path = os.path.join(args.cache_dir, f'cand_xb_{tag}.npy') if args.cache_dir else ''
    meta_path = os.path.join(args.cache_dir, f'cand_meta_{tag}.parquet') if args.cache_dir else ''
    info_path = os.path.join(args.cache_dir, f'cand_info_{tag}.json') if args.cache_dir else ''
    if args.cache_dir and os.path.exists(xb_path) and os.path.exists(meta_path):
        print(f'[cache] loading candidate matrix from {args.cache_dir}', flush=True)
        xb = np.load(xb_path)
        mdf = pd.read_parquet(meta_path)
        info = json.load(open(info_path)) if os.path.exists(info_path) else {}
        meta = {
            'id': mdf['id'].to_numpy(dtype=object),
            'source': mdf['source'].to_numpy(dtype=object),
            'shard': mdf['shard'].to_numpy(dtype=object),
            'least_dimension': mdf['least_dimension'].to_numpy(dtype=np.float32),
            'aes_jasperai': mdf['aes_jasperai'].to_numpy(dtype=np.float32),
            'xb': xb,
            'total_rows_seen': int(info.get('total_rows_seen', 0)),
            'bad_embedding_rows': int(info.get('bad_embedding_rows', 0)),
        }
        print(f'[cache] loaded xb={xb.shape} rows={len(meta["id"]):,}', flush=True)
    else:
        meta = load_candidates(files, args.monet_manifest, args.thr,
                               args.include_synthetic, out_dtype)
        xb = meta['xb']
        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)
            print(f'[cache] writing candidate cache to {args.cache_dir}', flush=True)
            np.save(xb_path, xb)
            pd.DataFrame({
                'id': meta['id'], 'source': meta['source'], 'shard': meta['shard'],
                'least_dimension': meta['least_dimension'],
                'aes_jasperai': meta['aes_jasperai'],
            }).to_parquet(meta_path, index=False)
            with open(info_path, 'w') as fh:
                json.dump({'total_rows_seen': meta['total_rows_seen'],
                           'bad_embedding_rows': meta['bad_embedding_rows'],
                           'candidate_rows': int(xb.shape[0])}, fh)
            print('[cache] candidate cache written', flush=True)
    nlist = min(args.nlist, max(1, xb.shape[0] // 2))
    train_fraction = min(1.0, max(1.0 / max(len(xb), 1),
                                  args.train_sample / max(len(xb), 1)))
    index = CuvsIndex(xb, nlist, args.nprobe, args.kmeans_iters,
                      train_fraction, args.multi_gpu)
    uf, edge_count, rows_with_neighbor = union_neighbors(
        index, xb, args.thr, args.topk, args.query_batch)
    drops, multi_clusters = build_drop_frame(meta, uf)
    out_parquet = f'{args.out_dir}/monet_neardup_drop_ids.parquet'
    write_parquet_checked(drops, out_parquet)
    summary = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'monet_emb': args.monet_emb,
        'monet_manifest': args.monet_manifest,
        'out_dir': args.out_dir,
        'out_parquet': out_parquet,
        'backend': index.backend,
        'multi_gpu': bool(args.multi_gpu),
        'include_synthetic': bool(args.include_synthetic),
        'index_dtype': args.index_dtype,
        'files': len(files),
        'total_rows_seen': meta['total_rows_seen'],
        'candidate_rows': int(xb.shape[0]),
        'bad_embedding_rows': meta['bad_embedding_rows'],
        'edge_count': int(edge_count),
        'rows_with_neighbor': int(rows_with_neighbor),
        'multi_clusters': int(multi_clusters),
        'drop_rows': int(len(drops)),
        'thr': args.thr,
        'topk': args.topk,
        'nlist': int(nlist),
        'nprobe': args.nprobe,
        'kmeans_iters': args.kmeans_iters,
        'train_sample': args.train_sample,
        'train_fraction': train_fraction,
        'query_batch': args.query_batch,
        'elapsed_sec': round(time.time() - t0, 3),
    }
    write_json_checked(summary, f'{args.out_dir}/summary.json')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
