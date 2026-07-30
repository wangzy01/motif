"""T5 MONET per-sub-source quality breakdown (for Evan's cap decision).

Computes, on the ACTUAL selectable real pool (keep==1 & synthetic==0 & holdout==0,
MINUS internal near-dup drop ids), for each sub-source:
  count, aes_jasperai mean + p10/p25/p50/p75/p90, >=512 (bucket512) count+share,
  caption-band coverage has_t1..t4.
Also reports per-source synthetic count and the holdout-real count being excluded.

This is decision-INPUT only (no selection, no cap, no big writes). Lets Evan compare
laion vs coyo vs the clean sources before fixing the 40M cap. Single manifest scan,
global pyarrow membership for the dedup. CPU-only.
Output: t5_inputs/monet_subsource_breakdown.json
"""
import glob
import json
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

S3 = os.environ.get('MOT_FILTER_OUT', '/s3-pixai-test-uw2/evan/mot_filtering_v1')
MANIFEST = f'{S3}/monet_manifest'
DROP = f'{S3}/t2_monet_internal_cuvs_single_full/monet_neardup_drop_ids.parquet'
OUTDIR = f'{S3}/t5_inputs'
FSX = '/fsx/evan/mot_code/data_analysis/outputs'
HAS = ['has_t1', 'has_t2', 'has_t3', 'has_t4']
t0 = time.time()


def main():
    drop_arr = pq.read_table(DROP, columns=['drop_id']).column('drop_id').combine_chunks()
    print(f'[drop] {len(drop_arr):,} internal dup ids ({time.time()-t0:.0f}s)', flush=True)

    files = sorted(glob.glob(f'{MANIFEST}/*.parquet'))
    max_files = int(os.environ.get('MAX_FILES', '0'))
    if max_files > 0:
        files = files[:max_files]
    syn_per_src = defaultdict(int)
    holdout_real_per_src = defaultdict(int)

    def load_shard(f):
        try:
            df = pd.read_parquet(f, columns=['id', 'source', 'synthetic', 'keep', 'holdout',
                                             'bucket512', 'aes_jasperai'] + HAS)
            sc = df[df['synthetic'] == 1]['source'].value_counts().to_dict()
            kr = df[(df['keep'] == 1) & (df['synthetic'] == 0)]
            hc = kr[kr['holdout'] == 1]['source'].value_counts().to_dict()
            real = kr[kr['holdout'] == 0]           # selectable pool (pre internal-dedup)
            if not len(real):
                return ('OK', None, sc, hc)
            out = {
                'id': pa.array(real['id'].to_numpy(), pa.large_string()),
                'source': real['source'].to_numpy(),
                'aes': real['aes_jasperai'].to_numpy().astype(np.float32),
                'b512': real['bucket512'].to_numpy().astype(np.int8),
                'has': real[HAS].to_numpy().astype(np.int8),
            }
            return ('OK', out, sc, hc)
        except Exception as e:  # noqa
            return ('ERR', f'{os.path.basename(f)}: {e}', {}, {})

    IDS, SRC, AES, B512, HASM = [], [], [], [], []
    errs = []
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for tag, out, sc, hc in ex.map(load_shard, files):
            done += 1
            for s, v in sc.items():
                syn_per_src[s] += int(v)
            for s, v in hc.items():
                holdout_real_per_src[s] += int(v)
            if tag == 'ERR':
                errs.append(out)
            elif out is not None:
                IDS.append(out['id']); SRC.append(out['source']); AES.append(out['aes'])
                B512.append(out['b512']); HASM.append(out['has'])
            if done % 3000 == 0:
                print(f'[scan] {done}/{len(files)} errs={len(errs)} ({time.time()-t0:.0f}s)', flush=True)

    ids = pa.chunked_array(IDS, type=pa.large_string())
    src = np.concatenate(SRC); aes = np.concatenate(AES)
    b512 = np.concatenate(B512); hasm = np.concatenate(HASM)
    del IDS, SRC, AES, B512, HASM
    gated = len(src)
    keep_pa = pc.invert(pc.is_in(ids, value_set=drop_arr))
    keep = keep_pa.combine_chunks().to_numpy(zero_copy_only=False).astype(bool)
    src = src[keep]; aes = aes[keep]; b512 = b512[keep]; hasm = hasm[keep]
    n = len(src)
    print(f'[pool] keep&!syn&!holdout={gated:,} -> survivors(post internal-dedup)={n:,} '
          f'({time.time()-t0:.0f}s)', flush=True)

    rows = []
    for s in sorted(set(src.tolist())):
        msk = src == s
        a = aes[msk]
        af = a[np.isfinite(a)]
        pcts = np.percentile(af, [10, 25, 50, 75, 90]) if len(af) else [float('nan')] * 5
        rows.append({
            'source': s,
            'survivors': int(msk.sum()),
            'aes_mean': round(float(np.nanmean(a)), 3) if len(a) else None,
            'aes_p10': round(float(pcts[0]), 3), 'aes_p25': round(float(pcts[1]), 3),
            'aes_p50': round(float(pcts[2]), 3), 'aes_p75': round(float(pcts[3]), 3),
            'aes_p90': round(float(pcts[4]), 3),
            'ge512': int((b512[msk] == 1).sum()),
            'ge512_share': round(float((b512[msk] == 1).mean()), 4) if msk.sum() else 0.0,
            'has_t1': round(float((hasm[msk, 0] == 1).mean()), 4) if msk.sum() else 0.0,
            'has_t2': round(float((hasm[msk, 1] == 1).mean()), 4) if msk.sum() else 0.0,
            'has_t3': round(float((hasm[msk, 2] == 1).mean()), 4) if msk.sum() else 0.0,
            'has_t4': round(float((hasm[msk, 3] == 1).mean()), 4) if msk.sum() else 0.0,
            'synthetic': int(syn_per_src.get(s, 0)),
            'holdout_real_excluded': int(holdout_real_per_src.get(s, 0)),
        })
    rows.sort(key=lambda r: -r['survivors'])
    summary = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'note': 'selectable real pool = keep & !synthetic & !holdout, minus internal near-dup. For cap decision.',
        'selectable_survivors_total': int(n),
        'clean_sources_survivors': int(sum(r['survivors'] for r in rows
                                           if r['source'] not in ('laion', 'coyo'))),
        'crawl_sources_survivors': int(sum(r['survivors'] for r in rows
                                           if r['source'] in ('laion', 'coyo'))),
        'holdout_real_excluded_total': int(sum(holdout_real_per_src.values())),
        'synthetic_total': int(sum(syn_per_src.values())),
        'per_source': rows,
        'shard_errors': len(errs),
        'elapsed_sec': round(time.time() - t0, 1),
    }
    os.makedirs(OUTDIR, exist_ok=True)
    tmp = f'{FSX}/_t5_monet_breakdown.tmp.json'
    with open(tmp, 'w') as fh:
        json.dump(summary, fh, indent=2)
    dst = f'{OUTDIR}/monet_subsource_breakdown.json'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(tmp, dst)
    os.remove(tmp)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'TOTAL {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
