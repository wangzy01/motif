"""T5 decision-independent prep (does NOT need T4b labels):

  1. GPIC cross-source drop-union manifest = pHash-exact ∪ SSCD(cos>=0.75) GPIC keys,
     with provenance flags -> t5_inputs/gpic_drop_union.parquet
  2. GPIC survivor pool stats: join T4 gpic_clusters (cluster_id) with gpic_pass
     (aes_laion_improved, w/h), drop the union, apply the global bottom-15% aesthetic
     cut, and tabulate per (cluster_id x resolution bucket) survivor counts pre/post cut.
     -> t5_inputs/gpic_survivor_per_cluster.parquet + gpic_survivor_pool_stats.json

All S3 writes go through a local FSX temp + copyfile (pyarrow streaming writes to the
S3 FUSE mount raise 'Operation not permitted'). The actual stratified sampling and the
MONET sub-source cap are NOT done here (they use Evan's T4b cluster content labels).
"""
import glob
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.dataset as pds

S3 = '/s3-pixai-test-uw2/evan/mot_filtering_v1'
OUTDIR = f'{S3}/t5_inputs'
FSX = '/fsx/evan/mot_code/data_analysis/outputs'
T4 = f'{S3}/t4_clipb_joint_kmeans'
GPIC = f'{S3}/gpic_pass'
K = 20000
t0 = time.time()
os.makedirs(OUTDIR, exist_ok=True)


def put(localfn, dst):
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(localfn, dst)


def write_parquet_s3(df, dst):
    tmp = f'{FSX}/_t5prep.{os.path.basename(dst)}.tmp.parquet'
    df.to_parquet(tmp, index=False)
    put(tmp, dst)
    os.remove(tmp)


def write_json_s3(obj, dst):
    tmp = f'{FSX}/_t5prep.{os.path.basename(dst)}.tmp.json'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2)
    put(tmp, dst)
    os.remove(tmp)


# ---- 1. drop union ----
pf = sorted(glob.glob(f'{S3}/t2_phash_exact/phash_exact_hits_*.parquet'))
phash_keys = set(pds.dataset(pf, format='parquet').to_table(
    columns=['key'], filter=pds.field('drop_phash_exact') == 1).column('key').to_pylist())
print(f'[phash] {len(phash_keys):,} ({time.time()-t0:.0f}s)', flush=True)
sdf = pd.read_parquet(f'{S3}/t2_sscd_bruteforce/gpic_sscd_drop_ids.parquet',
                      columns=['drop_key', 'sscd_cos']).drop_duplicates('drop_key')
sscd_keys = set(sdf['drop_key'].tolist())
print(f'[sscd] {len(sscd_keys):,} ({time.time()-t0:.0f}s)', flush=True)
union = phash_keys | sscd_keys
inter = phash_keys & sscd_keys
udf = pd.DataFrame({'drop_key': list(union)})
udf['in_phash'] = udf['drop_key'].isin(phash_keys)
udf['in_sscd'] = udf['drop_key'].isin(sscd_keys)
udf = udf.merge(sdf, on='drop_key', how='left')
write_parquet_s3(udf, f'{OUTDIR}/gpic_drop_union.parquet')
print(f'[manifest] union={len(union):,} overlap={len(inter):,} '
      f'phash_only={len(phash_keys)-len(inter):,} sscd_only={len(sscd_keys)-len(inter):,} '
      f'({time.time()-t0:.0f}s)', flush=True)

# ---- 2. survivor pool join (shard-parallel) ----
pass_files = sorted(glob.glob(f'{GPIC}/gpic_pass_*.parquet'))
errors = []


def load_shard(pfile):
    shard = os.path.basename(pfile).replace('gpic_pass_', '').replace('.parquet', '')
    cfile = f'{T4}/gpic_clusters/gpic_cluster_{shard}.parquet'
    try:
        if not os.path.exists(cfile):
            return ('ERR', f'missing cluster file {shard}')
        c = pd.read_parquet(cfile, columns=['key', 'cluster_id'])
        p = pd.read_parquet(pfile, columns=['key', 'width', 'height', 'aes_laion_improved'])
        return ('OK', c.merge(p, on='key', how='inner'))
    except Exception as e:  # noqa
        return ('ERR', f'{shard}: {e}')


parts = []
with ThreadPoolExecutor(max_workers=16) as ex:
    for i, (tag, val) in enumerate(ex.map(load_shard, pass_files)):
        if tag == 'OK':
            if len(val):
                parts.append(val)
        else:
            errors.append(val)
        if (i + 1) % 1000 == 0:
            print(f'[join] {i+1}/{len(pass_files)} parts={len(parts)} '
                  f'errs={len(errors)} ({time.time()-t0:.0f}s)', flush=True)
if errors:
    print(f'[join] WARNING {len(errors)} shard errors; first few: {errors[:3]}', flush=True)
big = pd.concat(parts, ignore_index=True)
del parts
gated = len(big)
print(f'[join] gated+clustered rows={gated:,} ({time.time()-t0:.0f}s)', flush=True)

big = big[~big['key'].isin(union)].reset_index(drop=True)
surv = len(big)
print(f'[exclude] {gated:,} -> {surv:,} survivors (post-dedup) ({time.time()-t0:.0f}s)', flush=True)

short = np.minimum(big['width'].to_numpy(), big['height'].to_numpy())
res_idx = (short >= 512).astype(np.int8)            # 0 = 256-only, 1 = 512-capable
cl = big['cluster_id'].to_numpy().astype(np.int64)
aes = big['aes_laion_improved'].to_numpy().astype(np.float32)
finite = np.isfinite(aes)
print(f'[aes] finite={int(finite.sum()):,}/{len(aes):,} nan={int((~finite).sum()):,}', flush=True)

thr15 = float(np.percentile(aes[finite], 15))
keep = finite & (aes >= thr15)                       # drop nan + bottom 15%
print(f'[cut] aes_p15={thr15:.4f} after_cut={int(keep.sum()):,} ({time.time()-t0:.0f}s)', flush=True)

valid = (cl >= 0) & (cl < K)
pre = np.zeros((K, 2), dtype=np.int64)
post = np.zeros((K, 2), dtype=np.int64)
asum = np.zeros((K, 2), dtype=np.float64)
np.add.at(pre, (cl[valid], res_idx[valid]), 1)
af = valid & finite
np.add.at(asum, (cl[af], res_idx[af]), aes[af])
kp = valid & keep
np.add.at(post, (cl[kp], res_idx[kp]), 1)

cdf = pd.DataFrame({
    'cluster_id': np.arange(K, dtype=np.int32),
    'surv_256': pre[:, 0], 'surv_512': pre[:, 1],
    'postcut_256': post[:, 0], 'postcut_512': post[:, 1],
    'postcut_total': post[:, 0] + post[:, 1],
    'aes_mean': np.where(pre.sum(1) > 0, asum.sum(1) / np.maximum(pre.sum(1), 1), np.nan),
})
write_parquet_s3(cdf, f'{OUTDIR}/gpic_survivor_per_cluster.parquet')

postcut_total = int(keep.sum())
stats = {
    'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'gpic_gated_clustered': gated,
    'gpic_drop_union': len(union),
    'phash_only': len(phash_keys) - len(inter),
    'sscd_only': len(sscd_keys) - len(inter),
    'phash_sscd_overlap': len(inter),
    'gpic_survivors_postdedup': surv,
    'nan_aes': int((~finite).sum()),
    'aes_p15_threshold': round(thr15, 5),
    'gpic_survivors_postcut': postcut_total,
    'postcut_res256': int(post[:, 0].sum()),
    'postcut_res512': int(post[:, 1].sum()),
    'nonempty_clusters_postcut': int((post.sum(1) > 0).sum()),
    'gpic_target': 20000000,
    'implied_keep_rate_to_target': round(20000000 / max(postcut_total, 1), 4),
    'monet_real_keep': 76987465,
    'monet_internal_drop': 3132196,
    'monet_survivors': 73855269,
    'shard_errors': len(errors),
    'elapsed_sec': round(time.time() - t0, 1),
}
write_json_s3(stats, f'{OUTDIR}/gpic_survivor_pool_stats.json')
print(json.dumps(stats, indent=2), flush=True)
print(f'TOTAL {time.time()-t0:.0f}s', flush=True)
