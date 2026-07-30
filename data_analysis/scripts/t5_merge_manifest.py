"""T5 merge: combine the GPIC (20M) and MONET (40M) selections into a single
DRAFT training pool manifest + a composition QC report.

This is a v1 DRAFT: it depends on the MONET sub-source cap (a default Evan still
owes confirmation on) and on the GPIC selection policy. Re-running after either
changes is cheap. Synthetic MONET is NOT part of the 60M real pool (separate 1024
anchor pool); it is only reported.

Outputs (to t5_manifest/):
  train_pool_v1_draft/*.parquet   lean unified rows: uid,dataset,sub_source,res_bucket,aes,aes_scorer
  train_pool_v1_draft_qc.json     full composition QC

Detailed per-dataset fields (cluster_id, dominant_cat, caption bands ...) stay in
t5_selection/{gpic,monet}_selected and join back via uid. CPU-only.
"""
import glob
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

S3 = os.environ.get('MOT_FILTER_OUT', '/s3-pixai-test-uw2/evan/mot_filtering_v1')
SEL = f'{S3}/t5_selection'
OUTDIR = f'{S3}/t5_manifest'
FSX = '/fsx/evan/mot_code/data_analysis/outputs'
N_OUT_SHARDS = int(os.environ.get('N_OUT_SHARDS', '256'))
t0 = time.time()


def put(localfn, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(localfn, dst)


def main():
    os.makedirs(f'{OUTDIR}/train_pool_v1_draft', exist_ok=True)

    g = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{SEL}/gpic_selected/*.parquet'))],
                  ignore_index=True)
    print(f'[gpic] {len(g):,} rows ({time.time()-t0:.0f}s)', flush=True)
    m = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{SEL}/monet_selected/*.parquet'))],
                  ignore_index=True)
    print(f'[monet] {len(m):,} rows ({time.time()-t0:.0f}s)', flush=True)

    gdf = pd.DataFrame({
        'uid': ('gpic:' + g['key'].astype(str)),
        'dataset': 'gpic',
        'sub_source': 'gpic',
        'res_bucket': np.where(g['res512'].to_numpy() == 1, 'ge512', '256'),
        'aes': g['aes_laion_improved'].to_numpy().astype(np.float32),
        'aes_scorer': 'laion_improved',
    })
    mdf = pd.DataFrame({
        'uid': ('monet:' + m['id'].astype(str)),
        'dataset': 'monet',
        'sub_source': m['source'].astype(str),
        'res_bucket': np.where(m['bucket512'].to_numpy() == 1, 'ge512', '256'),
        'aes': m['aes_jasperai'].to_numpy().astype(np.float32),
        'aes_scorer': 'jasperai',
    })
    pool = pd.concat([gdf, mdf], ignore_index=True)
    n = len(pool)
    dup = int(pool['uid'].duplicated().sum())
    print(f'[pool] total={n:,} dup_uid={dup} ({time.time()-t0:.0f}s)', flush=True)

    bounds = np.linspace(0, n, N_OUT_SHARDS + 1).astype(np.int64)
    for s in range(N_OUT_SHARDS):
        a, b = bounds[s], bounds[s + 1]
        if a >= b:
            continue
        tmp = f'{FSX}/_t5merge.shard{s:04d}.tmp.parquet'
        pool.iloc[a:b].to_parquet(tmp, index=False)
        put(tmp, f'{OUTDIR}/train_pool_v1_draft/train_pool_{s:04d}.parquet')
        os.remove(tmp)
    print(f'[write] {N_OUT_SHARDS} shards ({time.time()-t0:.0f}s)', flush=True)

    def res_share(df):
        vc = df['res_bucket'].value_counts()
        return {k: int(vc.get(k, 0)) for k in ['256', 'ge512']}

    qc = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'note': 'v1 real pool (synthetic+holdout excluded). MONET sub-source cap = default 40M, accepted by Evan 2026-06-16.',
        'total': n,
        'dup_uid': dup,
        'by_dataset': {k: int(v) for k, v in pool['dataset'].value_counts().items()},
        'by_res_bucket': res_share(pool),
        'gpic_res': res_share(gdf),
        'monet_res': res_share(mdf),
        'monet_by_source': {k: int(v) for k, v in mdf['sub_source'].value_counts().items()},
        'gpic_aes_mean': round(float(gdf['aes'].mean()), 4),
        'monet_aes_mean': round(float(mdf['aes'].mean()), 4),
        'elapsed_sec': round(time.time() - t0, 1),
    }
    tmp = f'{FSX}/_t5merge_qc.tmp.json'
    with open(tmp, 'w') as fh:
        json.dump(qc, fh, indent=2)
    put(tmp, f'{OUTDIR}/train_pool_v1_draft_qc.json')
    os.remove(tmp)
    print(json.dumps(qc, indent=2), flush=True)
    print(f'TOTAL {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
