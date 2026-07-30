"""T5 rich training manifest (v1) — joins the selected 60M pool with locators,
resolution, aesthetic, cluster, T4b content label, and caption-band/captioner
availability. Caption TEXT is NOT materialized (pointer model): the manifest
records src_root/src_shard/src_key so the dataloader reads image + caption text
from the source at train time. Sampling weights are NOT baked (per Evan): the
loader applies stage×res×caption-band×cluster weights from these fields.

Source roots (from t1/t3 extract scripts):
  GPIC  /s3-mewtant-raw-datasets/stanford-vision-lab/gpic_umziped
        images: images/train/{shard}/ ; caption+meta: parquets/gpic_train_{shard}.parquet
  MONET /s3-mewtant-raw-datasets/jasperai/monet/v1.1.0
        caption fields: caption_{original,florence-2-large,internvl-3-8b,sharegpt4v-7b,gemini-2.5-flash-lite}

Outputs (to t5_manifest/rich_v1/): gpic/*.parquet, monet/*.parquet (common schema),
summary.json. All membership via global pyarrow pc.is_in (never per-shard .isin).
CPU-only; S3 writes via local FSX tmp + copyfile (makedirs before each write).
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
SEL = f'{S3}/t5_selection'
T4 = f'{S3}/t4_clipb_joint_kmeans'
GPIC_BASE = f'{S3}/gpic_manifest_base'
MONET_MAN = f'{S3}/monet_manifest'
OUTDIR = os.environ.get('RICH_OUT', f'{S3}/t5_manifest/rich_v1')
FSX = '/fsx/evan/mot_code/data_analysis/outputs'
GROOT = '/s3-mewtant-raw-datasets/stanford-vision-lab/gpic_umziped'
MROOT = '/s3-mewtant-raw-datasets/jasperai/monet/v1.1.0'
N_OUT = int(os.environ.get('N_OUT_SHARDS', '256'))
MAXF = int(os.environ.get('MAX_FILES', '0'))
t0 = time.time()
TOKMAP = {'tok_original': 'original', 'tok_florence': 'florence-2-large',
          'tok_internvl': 'internvl-3-8b', 'tok_sharegpt4v': 'sharegpt4v-7b',
          'tok_gemini': 'gemini-2.5-flash-lite'}


def put(localfn, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(localfn, dst)


def write_sharded(df, subdir, prefix):
    n = len(df)
    b = np.linspace(0, n, N_OUT + 1).astype(np.int64)
    for s in range(N_OUT):
        a, c = b[s], b[s + 1]
        if a >= c:
            continue
        tmp = f'{FSX}/_t5rich.{prefix}{s:04d}.tmp.parquet'
        df.iloc[a:c].to_parquet(tmp, index=False)
        put(tmp, f'{OUTDIR}/{subdir}/{prefix}{s:04d}.parquet')
        os.remove(tmp)


def collect(files, key_col, want_cols, key_set_arr, label):
    """Read ALL source rows (key_col+want_cols) across files, then do ONE global
    pyarrow pc.is_in against the selected key set. CRITICAL: never call is_in per
    shard against the big set -- that rebuilds the (20-40M) value_set hash per shard
    (~hours). Build value_set hash exactly once by filtering the concatenated table."""
    def load(f):
        try:
            return pq.read_table(f, columns=[key_col] + want_cols)
        except Exception as e:  # noqa
            return ('ERR', f'{os.path.basename(f)}: {e}')
    parts, errs, done = [], [], 0
    with ThreadPoolExecutor(max_workers=int(os.environ.get('READ_WORKERS', '32'))) as ex:
        for r in ex.map(load, files):
            done += 1
            if isinstance(r, tuple):
                errs.append(r[1]); continue
            parts.append(r)
            if done % 3000 == 0:
                print(f'[{label}] read {done}/{len(files)} ({time.time()-t0:.0f}s)', flush=True)
    if errs:
        print(f'[{label}] WARN {len(errs)} errors; first {errs[:2]}', flush=True)
    if not parts:
        return pd.DataFrame()
    # Unify schema across shards before concat: shards vary (string vs large_string;
    # empty/degenerate shards give a `null`-typed column). Per field pick a canonical
    # type -- large_string for any string/large_string, else the first non-null type
    # (null-typed cols cast losslessly to it) -- and cast every part to it.
    names = parts[0].schema.names

    def canon(n):
        ts = [p.schema.field(n).type for p in parts]
        nonnull = [t for t in ts if t != pa.null()]
        if not nonnull:
            return pa.large_string()
        t0 = nonnull[0]
        return pa.large_string() if t0 in (pa.string(), pa.large_string()) else t0
    target = pa.schema([(n, canon(n)) for n in names])
    full = pa.concat_tables([p.cast(target) for p in parts])
    del parts
    mask = pc.is_in(full.column(key_col), value_set=key_set_arr)   # ONE global build
    matched = full.filter(mask)
    print(f'[{label}] scanned={full.num_rows:,} matched={matched.num_rows:,} ({time.time()-t0:.0f}s)', flush=True)
    return matched.to_pandas()


def main():
    os.makedirs(f'{OUTDIR}/gpic', exist_ok=True)
    os.makedirs(f'{OUTDIR}/monet', exist_ok=True)
    lab = pd.read_parquet(f'{T4}/cluster_content_labels.parquet', columns=['cluster_id', 'dominant_cat'])
    dom = dict(zip(lab['cluster_id'].to_numpy(), lab['dominant_cat'].to_numpy()))

    MAX_SEL = int(os.environ.get('MAX_SEL', '0'))
    # ===================== GPIC =====================
    gsel = sorted(glob.glob(f'{SEL}/gpic_selected/*.parquet'))
    if MAX_SEL:
        gsel = gsel[:MAX_SEL]
    g = pd.concat([pd.read_parquet(f) for f in gsel], ignore_index=True)
    print(f'[gpic] selected={len(g):,} ({time.time()-t0:.0f}s)', flush=True)
    gkeys = pa.array(g['key'].to_numpy(), pa.large_string())
    bfiles = sorted(glob.glob(f'{GPIC_BASE}/*.parquet'))
    if MAXF:
        bfiles = bfiles[:MAXF]
    meta = collect(bfiles, 'key', ['shard', 'width', 'height', 'short', 'tier', 'caption_type', 'tok'],
                   gkeys, 'gpic-base')
    gm = g.merge(meta, on='key', how='left')
    miss = int(gm['shard'].isna().sum())
    print(f'[gpic] joined base, missing_meta={miss:,} ({time.time()-t0:.0f}s)', flush=True)
    tier = gm['tier'].fillna(0).to_numpy().astype(np.int8)
    grich = pd.DataFrame({
        'uid': 'gpic:' + gm['key'].astype(str),
        'dataset': 'gpic', 'sub_source': 'gpic',
        'src_root': GROOT, 'src_shard': gm['shard'].astype('string'),
        'src_key': gm['key'].astype(str),
        'width': gm['width'].fillna(0).astype(np.int32), 'height': gm['height'].fillna(0).astype(np.int32),
        'res_bucket': np.where(gm['res512'].to_numpy() == 1, 'ge512', '256'),
        'aes': gm['aes_laion_improved'].astype(np.float32), 'aes_scorer': 'laion_improved',
        'cluster_id': gm['cluster_id'].astype(np.int32),
        'dominant_cat': gm['dominant_cat'].astype('string'),
        'caption_type': gm['caption_type'].astype('string'),   # GPIC i1 type: tag/short/medium/long
        'caption_tier': tier,
        'has_t1': (tier == 1).astype(np.int8), 'has_t2': (tier == 2).astype(np.int8),
        'has_t3': (tier == 3).astype(np.int8), 'has_t4': (tier == 4).astype(np.int8),
        'cap_i1': np.int8(1), 'cap_original': np.int8(0), 'cap_florence': np.int8(0),
        'cap_internvl': np.int8(0), 'cap_sharegpt4v': np.int8(0), 'cap_gemini': np.int8(0),
        # per-caption token counts (band->captioner selection at train time, per v3 doc).
        # GPIC: single i1 caption -> tok_i1; MONET captioner cols are 0 for GPIC.
        'tok_i1': gm['tok'].fillna(0).clip(lower=0).astype(np.int32),   # int32: int16 overflowed on garbage captions
        'tok_original': np.int32(0), 'tok_florence': np.int32(0), 'tok_internvl': np.int32(0),
        'tok_sharegpt4v': np.int32(0), 'tok_gemini': np.int32(0),
        'synthetic': np.int8(0), 'holdout': np.int8(0),
    })
    write_sharded(grich, 'gpic', 'gpic_')
    print(f'[gpic] wrote {len(grich):,} ({time.time()-t0:.0f}s)', flush=True)
    del g, gm, meta

    # ===================== MONET =====================
    msel = sorted(glob.glob(f'{SEL}/monet_selected/*.parquet'))
    if MAX_SEL:
        msel = msel[:MAX_SEL]
    m = pd.concat([pd.read_parquet(f) for f in msel], ignore_index=True)
    print(f'[monet] selected={len(m):,} ({time.time()-t0:.0f}s)', flush=True)
    mids = pa.array(m['id'].to_numpy(), pa.large_string())
    mfiles = sorted(glob.glob(f'{MONET_MAN}/*.parquet'))
    cfiles = sorted(glob.glob(f'{T4}/monet_clusters/*.parquet'))
    if MAXF:
        mfiles, cfiles = mfiles[:MAXF], cfiles[:MAXF]
    man = collect(mfiles, 'id', ['key', 'shard', 'width', 'height'] + list(TOKMAP), mids, 'monet-man')
    clu = collect(cfiles, 'id', ['cluster_id'], mids, 'monet-clu')
    mm = m.merge(man, on='id', how='left').merge(clu, on='id', how='left')
    miss_meta = int(mm['key'].isna().sum()); miss_clu = int(mm['cluster_id'].isna().sum())
    print(f'[monet] joined, missing_meta={miss_meta:,} missing_cluster={miss_clu:,} ({time.time()-t0:.0f}s)', flush=True)
    # vectorized dominant_cat lookup (NO per-row python over 40M)
    K = 20000
    dom_arr = np.array([dom.get(i, 'unknown') for i in range(K)], dtype=object)
    cid = mm['cluster_id'].fillna(-1).to_numpy().astype(np.int64)
    domcat = np.where((cid >= 0) & (cid < K), dom_arr[np.clip(cid, 0, K - 1)], 'unknown')
    # captioner availability as flags from tok_* > 0 (vectorized)
    capflag = {c: (mm[c].fillna(0).to_numpy() > 0).astype(np.int8) for c in TOKMAP}
    mrich = pd.DataFrame({
        'uid': 'monet:' + mm['id'].astype(str),
        'dataset': 'monet', 'sub_source': mm['source'].astype('string'),
        'src_root': MROOT, 'src_shard': mm['shard'].astype('string'),
        'src_key': mm['id'].astype(str),
        'width': mm['width'].fillna(0).astype(np.int32), 'height': mm['height'].fillna(0).astype(np.int32),
        'res_bucket': np.where(mm['bucket512'].to_numpy() == 1, 'ge512', '256'),
        'aes': mm['aes_jasperai'].astype(np.float32), 'aes_scorer': 'jasperai',
        'cluster_id': cid.astype(np.int32),
        'dominant_cat': pd.Series(domcat, dtype='string'),
        'caption_type': 'monet',   # MONET: multi-captioner; per-captioner bands are in tok_* below
        'caption_tier': np.int8(0),
        'has_t1': mm['has_t1'].astype(np.int8), 'has_t2': mm['has_t2'].astype(np.int8),
        'has_t3': mm['has_t3'].astype(np.int8), 'has_t4': mm['has_t4'].astype(np.int8),
        'cap_i1': np.int8(0), 'cap_original': capflag['tok_original'], 'cap_florence': capflag['tok_florence'],
        'cap_internvl': capflag['tok_internvl'], 'cap_sharegpt4v': capflag['tok_sharegpt4v'],
        'cap_gemini': capflag['tok_gemini'],
        # per-captioner token counts: loader picks captioner per band (T4->gemini, short->florence,
        # megalith T2->florence first sentence, etc.) and truncates ~224-256 tok per v3 doc.
        'tok_i1': np.int32(0),
        'tok_original': mm['tok_original'].fillna(0).clip(lower=0).astype(np.int32),
        'tok_florence': mm['tok_florence'].fillna(0).clip(lower=0).astype(np.int32),
        'tok_internvl': mm['tok_internvl'].fillna(0).clip(lower=0).astype(np.int32),
        'tok_sharegpt4v': mm['tok_sharegpt4v'].fillna(0).clip(lower=0).astype(np.int32),
        'tok_gemini': mm['tok_gemini'].fillna(0).clip(lower=0).astype(np.int32),
        'synthetic': np.int8(0), 'holdout': np.int8(0),
    })
    write_sharded(mrich, 'monet', 'monet_')
    print(f'[monet] wrote {len(mrich):,} ({time.time()-t0:.0f}s)', flush=True)

    summary = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'note': 'v1 rich manifest. caption=pointer (read at train time). sampling weights NOT baked.',
        'total': int(len(grich) + len(mrich)),
        'gpic_rows': int(len(grich)), 'monet_rows': int(len(mrich)),
        'gpic_missing_meta': miss, 'monet_missing_meta': miss_meta, 'monet_missing_cluster': miss_clu,
        'schema': list(grich.columns),
        'src_roots': {'gpic': GROOT, 'monet': MROOT},
        'gpic_caption_tier_dist': {int(k): int(v) for k, v in pd.Series(tier).value_counts().items()},
        'elapsed_sec': round(time.time() - t0, 1),
    }
    tmp = f'{FSX}/_t5rich_summary.tmp.json'
    with open(tmp, 'w') as fh:
        json.dump(summary, fh, indent=2)
    put(tmp, f'{OUTDIR}/summary.json')
    os.remove(tmp)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'TOTAL {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
