"""T5 GPIC stratified selection to a target size (default 20M).

Policy (per 自己写的数据集文档_v3.md):
  * survivors = gpic_pass gated rows joined with T4 cluster_id, MINUS the cross-source
    drop union (pHash-exact ∪ SSCD cos>=0.75), MINUS the global bottom-AES_CUT_PCT% cut.
  * resolution: ALWAYS keep every survivor with short side >=512 (scarce, high value);
    "短边>=512 尽量全保".
  * remaining budget (TARGET - n_kept_512) is filled from the short<512 (256-bucket)
    survivors by a per-cluster quota (water-filling cap so no single cluster monopolizes),
    and WITHIN each cluster the highest-aesthetic rows win ("同层内美学优先",
    "cap 主要削 GPIC 自身的冗余大簇").

Exact, deterministic, single S3 scan. Loads survivor (key, cluster_id int16, res int8,
aes float32) into RAM via pyarrow (~5-6GB for ~82M rows) -- run on a roomy pod.
All S3 writes go through a local FSX temp + copyfile (FUSE streaming writes raise EPERM).
CPU-only.
"""
import glob
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

S3 = os.environ.get('MOT_FILTER_OUT', '/s3-pixai-test-uw2/evan/mot_filtering_v1')
T4 = f'{S3}/t4_clipb_joint_kmeans'
GPIC = f'{S3}/gpic_pass'
OUTDIR = os.environ.get('GPIC_SELECT_OUT', f'{S3}/t5_selection')
FSX = '/fsx/evan/mot_code/data_analysis/outputs'
K = int(os.environ.get('K', '20000'))
TARGET = int(os.environ.get('GPIC_TARGET', '20000000'))
AES_CUT_PCT = float(os.environ.get('AES_CUT_PCT', '10'))
N_OUT_SHARDS = int(os.environ.get('N_OUT_SHARDS', '256'))
t0 = time.time()


def put(localfn, dst):
    # re-ensure the parent prefix every write: mountpoint-s3 does not persist empty
    # dirs, so a subdir made at startup can vanish before the first file lands.
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(localfn, dst)


def write_json_s3(obj, dst):
    tmp = f'{FSX}/_t5gsel.{os.path.basename(dst)}.tmp.json'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2)
    put(tmp, dst)
    os.remove(tmp)


def waterfill_cap(counts, budget):
    """Largest integer cap C with sum(min(counts, C)) <= budget; plus the remainder
    spread 1-each over the clusters still above C (deterministic: smallest cluster_id
    first). Returns per-cluster quota array (same shape as counts)."""
    counts = counts.astype(np.int64)
    total = int(counts.sum())
    if budget >= total:
        return counts.copy()
    if budget <= 0:
        return np.zeros_like(counts)
    lo, hi = 0, int(counts.max())
    while lo < hi:                                   # smallest C with g(C) >= budget
        mid = (lo + hi + 1) // 2
        if int(np.minimum(counts, mid).sum()) <= budget:
            lo = mid
        else:
            hi = mid - 1
    C = lo
    q = np.minimum(counts, C)
    rem = budget - int(q.sum())
    if rem > 0:
        above = np.flatnonzero(counts > C)           # these can take +1
        take = above[:rem]
        q[take] += 1
    return q


def main():
    os.makedirs(f'{OUTDIR}/gpic_selected', exist_ok=True)
    # ---- drop union as a pyarrow array (membership done ONCE, globally; per-shard
    #      pandas .isin(8M set) is pathologically slow ~44s/shard -> never do it). ----
    union_arr = pq.read_table(f'{S3}/t5_inputs/gpic_drop_union.parquet',
                              columns=['drop_key']).column('drop_key').combine_chunks()
    print(f'[union] {len(union_arr):,} drop keys ({time.time()-t0:.0f}s)', flush=True)

    pass_files = sorted(glob.glob(f'{GPIC}/gpic_pass_*.parquet'))
    max_files = int(os.environ.get('MAX_FILES', '0'))
    if max_files > 0:
        pass_files = pass_files[:max_files]
        print(f'[smoke] MAX_FILES={max_files} -> {len(pass_files)} shards', flush=True)

    def load_shard(pfile):
        shard = os.path.basename(pfile).replace('gpic_pass_', '').replace('.parquet', '')
        cfile = f'{T4}/gpic_clusters/gpic_cluster_{shard}.parquet'
        try:
            if not os.path.exists(cfile):
                return ('ERR', f'missing cluster {shard}')
            c = pd.read_parquet(cfile, columns=['key', 'cluster_id'])
            p = pd.read_parquet(pfile, columns=['key', 'width', 'height', 'aes_laion_improved'])
            m = c.merge(p, on='key', how='inner')
            if not len(m):
                return (pa.array([], pa.large_string()), np.empty(0, np.int16),
                        np.empty(0, np.int8), np.empty(0, np.float32))
            short = np.minimum(m['width'].to_numpy(), m['height'].to_numpy())
            return (pa.array(m['key'].to_numpy(), pa.large_string()),
                    m['cluster_id'].to_numpy().astype(np.int16),
                    (short >= 512).astype(np.int8),
                    m['aes_laion_improved'].to_numpy().astype(np.float32))
        except Exception as e:  # noqa
            return ('ERR', f'{shard}: {e}')

    KEYS, CL, RES, AES = [], [], [], []
    errors = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, r in enumerate(ex.map(load_shard, pass_files)):
            if isinstance(r, tuple) and len(r) == 2 and r[0] == 'ERR':
                errors.append(r[1])
                continue
            ka, cl, res, aes = r
            if len(cl):
                KEYS.append(ka); CL.append(cl); RES.append(res); AES.append(aes)
            if (i + 1) % 1000 == 0:
                print(f'[scan] {i+1}/{len(pass_files)} kept_arrays={len(CL)} '
                      f'errs={len(errors)} ({time.time()-t0:.0f}s)', flush=True)
    if errors:
        print(f'[scan] WARNING {len(errors)} shard errors; first: {errors[:3]}', flush=True)

    keys = pa.chunked_array(KEYS, type=pa.large_string()) if KEYS else pa.chunked_array([], pa.large_string())
    cl = np.concatenate(CL); res = np.concatenate(RES); aes = np.concatenate(AES)
    del KEYS, CL, RES, AES
    gated = len(cl)
    print(f'[load] gated+clustered={gated:,} ({time.time()-t0:.0f}s)', flush=True)

    # ---- exclude cross-source drop union (ONE global pyarrow membership) ----
    keep_pa = pc.invert(pc.is_in(keys, value_set=union_arr))
    keep = keep_pa.combine_chunks().to_numpy(zero_copy_only=False).astype(bool)
    keys = keys.filter(keep_pa)
    cl = cl[keep]; res = res[keep]; aes = aes[keep]
    n = len(cl)
    print(f'[dedup] {gated:,} -> survivors(post-dedup)={n:,} '
          f'(dropped {gated-n:,}) ({time.time()-t0:.0f}s)', flush=True)

    # ---- global bottom-AES cut ----
    finite = np.isfinite(aes)
    thr_aes = float(np.percentile(aes[finite], AES_CUT_PCT))
    postcut = finite & (aes >= thr_aes)
    print(f'[cut] aes_p{int(AES_CUT_PCT)}={thr_aes:.4f} postcut={int(postcut.sum()):,} '
          f'nan={int((~finite).sum()):,} ({time.time()-t0:.0f}s)', flush=True)

    valid = (cl >= 0) & (cl < K)
    is512 = postcut & valid & (res == 1)
    is256 = postcut & valid & (res == 0)
    n512 = int(is512.sum())

    sel = np.zeros(n, dtype=bool)
    if n512 >= TARGET:
        # extreme edge case: keep top-TARGET 512 by aes, none from 256
        idx512 = np.flatnonzero(is512)
        order = idx512[np.argsort(-aes[idx512], kind='stable')][:TARGET]
        sel[order] = True
        budget256 = 0
        print(f'[plan] n512={n512:,} >= TARGET; capped 512 only', flush=True)
    else:
        sel[is512] = True
        budget256 = TARGET - n512
        counts256 = np.bincount(cl[is256], minlength=K).astype(np.int64)
        q = waterfill_cap(counts256, budget256)            # per-cluster res256 quota
        # within each cluster keep top-q by aes: sort res256 rows by (cluster, -aes)
        idx256 = np.flatnonzero(is256)
        order = np.lexsort((-aes[idx256], cl[idx256]))      # cluster asc, aes desc
        sidx = idx256[order]
        sclust = cl[sidx]
        # rank within cluster group
        starts = np.searchsorted(sclust, np.arange(K), side='left')
        rank = np.arange(len(sidx)) - starts[sclust]
        take = rank < q[sclust]
        sel[sidx[take]] = True
        print(f'[plan] keep_all_512={n512:,} budget256={budget256:,} '
              f'cap_clusters_capped={int((q < counts256).sum())} '
              f'selected256={int(take.sum()):,} ({time.time()-t0:.0f}s)', flush=True)

    nsel = int(sel.sum())
    print(f'[select] total_selected={nsel:,} (target {TARGET:,}) ({time.time()-t0:.0f}s)', flush=True)

    # ---- write selected rows, sharded, via local FSX tmp ----
    sel_idx = np.flatnonzero(sel)
    sel_keys = keys.take(pa.array(sel_idx)).to_numpy(zero_copy_only=False)
    sel_cl = cl[sel_idx]; sel_res = res[sel_idx]; sel_aes = aes[sel_idx]
    # join dominant_cat label
    lab = pd.read_parquet(f'{T4}/cluster_content_labels.parquet',
                          columns=['cluster_id', 'dominant_cat'])
    lab_map = dict(zip(lab['cluster_id'].to_numpy(), lab['dominant_cat'].to_numpy()))
    dom = np.array([lab_map.get(int(c), 'unknown') for c in range(K)], dtype=object)

    bounds = np.linspace(0, nsel, N_OUT_SHARDS + 1).astype(np.int64)
    for s in range(N_OUT_SHARDS):
        a, b = bounds[s], bounds[s + 1]
        if a >= b:
            continue
        df = pd.DataFrame({
            'key': sel_keys[a:b],
            'cluster_id': sel_cl[a:b].astype(np.int32),
            'res512': sel_res[a:b].astype(np.int8),
            'aes_laion_improved': sel_aes[a:b],
            'dominant_cat': dom[sel_cl[a:b]],
        })
        tmp = f'{FSX}/_t5gsel.shard{s:04d}.tmp.parquet'
        df.to_parquet(tmp, index=False)
        put(tmp, f'{OUTDIR}/gpic_selected/gpic_selected_{s:04d}.parquet')
        os.remove(tmp)
    print(f'[write] {N_OUT_SHARDS} shards ({time.time()-t0:.0f}s)', flush=True)

    summary = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'policy': 'keep_all_short>=512 + per-cluster waterfill cap on short<512, aes-priority within cluster',
        'gpic_target': TARGET,
        'aes_cut_pct': AES_CUT_PCT,
        'aes_cut_threshold': round(thr_aes, 5),
        'survivors_postdedup': int(n),
        'survivors_postcut': int(postcut.sum()),
        'kept_res512': int(min(n512, TARGET)),
        'kept_res256': int(nsel - min(n512, TARGET)),
        'total_selected': nsel,
        'nonempty_clusters_selected': int(np.unique(sel_cl).size),
        'shard_errors': len(errors),
        'elapsed_sec': round(time.time() - t0, 1),
    }
    write_json_s3(summary, f'{OUTDIR}/gpic_selection_summary.json')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'TOTAL {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
