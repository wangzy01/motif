"""T5 synthetic 1024 anchor-pool manifest (v1).

The MONET `synthetic/` pool (flux-klein + flux-schnell + z-image, ~13.8M, all
1024px) is text-to-image-MODEL-generated -- correctly EXCLUDED from the 60M real
pool. But per the v3 design docs it IS used in training: a little at 256 (low
weight), mid-low at 512, and as the PRIMARY real-ish 1024 high-res source for
Stage-2 anti-forgetting (GPIC has ~nothing at 1024). So it needs its own manifest.

Same rich schema as t5_manifest/rich_v1 (union-compatible), with synthetic=1,
cluster_id=-1 / dominant_cat='unknown' (synthetic was NOT in the joint MONET+GPIC
clustering), caption_type='synthetic-prompt' (the generation prompt -- medium
length, lands mid-band). Caption = pointer (read at train time); weights not baked.

Selection: ALL rows with keep==1 & synthetic==1 (no further subsetting -- the pool
is used as-is per the docs). Single global scan over monet_manifest, boolean filter
(NOT per-shard is_in). CPU-only; S3 writes via local FSX tmp + copyfile.
Output: t5_manifest/synthetic_v1/ : monet_synth/*.parquet + summary.json.
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
MONET_MAN = f'{S3}/monet_manifest'
OUTDIR = os.environ.get('SYN_OUT', f'{S3}/t5_manifest/synthetic_v1')
FSX = '/fsx/evan/mot_code/data_analysis/outputs'
MROOT = '/s3-mewtant-raw-datasets/jasperai/monet/v1.1.0'
N_OUT = int(os.environ.get('N_OUT_SHARDS', '64'))
MAXF = int(os.environ.get('MAX_FILES', '0'))
t0 = time.time()
TOKS = ['tok_original', 'tok_florence', 'tok_internvl', 'tok_sharegpt4v', 'tok_gemini']
COLS = (['id', 'key', 'shard', 'source', 'width', 'height', 'aes_jasperai',
         'synthetic', 'keep', 'holdout', 'bucket512'] + TOKS
        + ['has_t1', 'has_t2', 'has_t3', 'has_t4'])
# rich_v1 column order (must match exactly so the two manifests union cleanly)
SCHEMA_COLS = ['uid', 'dataset', 'sub_source', 'src_root', 'src_shard', 'src_key',
               'width', 'height', 'res_bucket', 'aes', 'aes_scorer', 'cluster_id',
               'dominant_cat', 'caption_type', 'caption_tier', 'has_t1', 'has_t2',
               'has_t3', 'has_t4', 'cap_i1', 'cap_original', 'cap_florence',
               'cap_internvl', 'cap_sharegpt4v', 'cap_gemini', 'tok_i1', 'tok_original',
               'tok_florence', 'tok_internvl', 'tok_sharegpt4v', 'tok_gemini',
               'synthetic', 'holdout']


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
        tmp = f'{FSX}/_t5syn.{prefix}{s:04d}.tmp.parquet'
        df.iloc[a:c].to_parquet(tmp, index=False)
        put(tmp, f'{OUTDIR}/{subdir}/{prefix}{s:04d}.parquet')
        os.remove(tmp)


def collect_synth(files):
    """Read needed cols across all monet_manifest shards, filter keep==1 &
    synthetic==1 per shard (cheap boolean, NOT is_in), accumulate, concat once."""
    def load(f):
        try:
            t = pq.read_table(f, columns=COLS)
            # keep & synthetic & NOT holdout (global holdout exclusion, Evan 2026-06-16:
            # holdout==1 never trains, in any pool). NOTE: the already-shipped synthetic_v1
            # predates this and still carries 6,565 holdout rows -> loader must filter
            # holdout==1 until a rebuild; this filter makes future rebuilds clean.
            m = pc.and_(pc.and_(pc.equal(t['keep'], 1), pc.equal(t['synthetic'], 1)),
                        pc.equal(t['holdout'], 0))
            sub = t.filter(m)
            return sub if sub.num_rows else None
        except Exception as e:  # noqa
            return ('ERR', f'{os.path.basename(f)}: {e}')
    parts, errs, done = [], [], 0
    with ThreadPoolExecutor(max_workers=int(os.environ.get('READ_WORKERS', '48'))) as ex:
        for r in ex.map(load, files):
            done += 1
            if isinstance(r, tuple):
                errs.append(r[1]); continue
            if r is not None:
                parts.append(r)
            if done % 2000 == 0:
                got = sum(p.num_rows for p in parts)
                print(f'[synth] scanned {done}/{len(files)} matched={got:,} ({time.time()-t0:.0f}s)', flush=True)
    if errs:
        print(f'[synth] WARN {len(errs)} read errors; first {errs[:2]}', flush=True)
    if not parts:
        raise SystemExit('[synth] no synthetic rows found')
    names = parts[0].schema.names

    def canon(n):
        ts = [p.schema.field(n).type for p in parts]
        nonnull = [t for t in ts if t != pa.null()]
        t = nonnull[0] if nonnull else pa.large_string()
        return pa.large_string() if t in (pa.string(), pa.large_string()) else t
    target = pa.schema([(n, canon(n)) for n in names])
    full = pa.concat_tables([p.cast(target) for p in parts])
    print(f'[synth] total synthetic rows = {full.num_rows:,} ({time.time()-t0:.0f}s)', flush=True)
    return full.to_pandas()


def main():
    os.makedirs(f'{OUTDIR}/monet_synth', exist_ok=True)
    files = sorted(glob.glob(f'{MONET_MAN}/*.parquet'))
    if MAXF:
        files = files[:MAXF]
    print(f'[synth] scanning {len(files)} monet_manifest shards ({time.time()-t0:.0f}s)', flush=True)
    m = collect_synth(files)

    capflag = {c: (m[c].fillna(0).to_numpy() > 0).astype(np.int8) for c in TOKS}
    rich = pd.DataFrame({
        'uid': 'monet:' + m['id'].astype(str),
        'dataset': 'monet_synth', 'sub_source': m['source'].astype('string'),
        'src_root': MROOT, 'src_shard': m['shard'].astype('string'),
        'src_key': m['id'].astype(str),
        'width': m['width'].fillna(0).astype(np.int32), 'height': m['height'].fillna(0).astype(np.int32),
        'res_bucket': np.where(m['bucket512'].to_numpy() == 1, 'ge512', '256'),
        'aes': m['aes_jasperai'].astype(np.float32), 'aes_scorer': 'jasperai',
        'cluster_id': np.int32(-1),                       # synthetic not in joint clustering
        'dominant_cat': pd.Series(['unknown'] * len(m), dtype='string'),
        'caption_type': 'synthetic-prompt',               # generation prompt, ~mid band
        'caption_tier': np.int8(0),
        'has_t1': m['has_t1'].fillna(0).astype(np.int8), 'has_t2': m['has_t2'].fillna(0).astype(np.int8),
        'has_t3': m['has_t3'].fillna(0).astype(np.int8), 'has_t4': m['has_t4'].fillna(0).astype(np.int8),
        'cap_i1': np.int8(0), 'cap_original': capflag['tok_original'], 'cap_florence': capflag['tok_florence'],
        'cap_internvl': capflag['tok_internvl'], 'cap_sharegpt4v': capflag['tok_sharegpt4v'],
        'cap_gemini': capflag['tok_gemini'],
        'tok_i1': np.int32(0),
        'tok_original': m['tok_original'].fillna(0).clip(lower=0).astype(np.int32),
        'tok_florence': m['tok_florence'].fillna(0).clip(lower=0).astype(np.int32),
        'tok_internvl': m['tok_internvl'].fillna(0).clip(lower=0).astype(np.int32),
        'tok_sharegpt4v': m['tok_sharegpt4v'].fillna(0).clip(lower=0).astype(np.int32),
        'tok_gemini': m['tok_gemini'].fillna(0).clip(lower=0).astype(np.int32),
        'synthetic': np.int8(1), 'holdout': m['holdout'].fillna(0).astype(np.int8),
    })
    assert list(rich.columns) == SCHEMA_COLS, f'schema drift vs rich_v1: {list(rich.columns)}'
    write_sharded(rich, 'monet_synth', 'synth_')
    print(f'[synth] wrote {len(rich):,} ({time.time()-t0:.0f}s)', flush=True)

    by_src = {str(k): int(v) for k, v in m['source'].value_counts().items()}
    ge1024 = int((np.minimum(rich['width'], rich['height']) >= 1024).sum())
    summary = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'note': ('synthetic 1024 anchor pool (keep & synthetic). text-to-image generated; '
                 'EXCLUDED from 60M real; used as 1024 high-res source per v3 docs. '
                 'cluster_id=-1 (not in joint clustering). caption=pointer; weights not baked.'),
        'total': int(len(rich)),
        'by_sub_source': by_src,
        'res_bucket': {str(k): int(v) for k, v in pd.Series(rich['res_bucket']).value_counts().items()},
        'short_edge_ge1024': ge1024,
        'aes_nan': int(rich['aes'].isna().sum()),
        'holdout_sum': int(rich['holdout'].sum()),
        'has_band': {f'has_t{i}': int(rich[f'has_t{i}'].sum()) for i in (1, 2, 3, 4)},
        'schema': list(rich.columns),
        'src_root': MROOT,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    tmp = f'{FSX}/_t5syn_summary.tmp.json'
    with open(tmp, 'w') as fh:
        json.dump(summary, fh, indent=2)
    put(tmp, f'{OUTDIR}/summary.json')
    os.remove(tmp)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'TOTAL {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
