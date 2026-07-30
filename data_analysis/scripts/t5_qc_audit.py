"""T5 rich-manifest QC + caption-band SUPPLY audit (read-only).

Verifies the rich_v1 manifest in aggregate and answers the question that drives
sampling-weight design: can the selected 60M pool actually supply the target
caption-band mix T1 18 / T2 27 / T3 15 / T4 40 -- especially the long bands
(T3/T4), and at >=512 resolution for late high-res stages?

Caption-band SUPPLY model (why GPIC vs MONET differ):
  GPIC image -> exactly ONE caption (i1), in band = caption_tier. A GPIC image
    can serve ONLY its own tier's band (a tier-2 GPIC image cannot be a T4 sample).
  MONET image -> up to 5 captioner outputs spanning bands; has_tX==1 means at
    least one captioner produced a caption in band X, so the image CAN serve
    band X (pick that captioner at train time).
  supply(band X) = #{GPIC: caption_tier==X} + #{MONET: has_tX==1}.

ADEQUACY: over one-epoch-equivalent exposure (= pool size N draws), target band
X wants TARGET[X]*N draws. oversample_factor = wanted_draws / supply. <=1 means
the band is over-supplied (satisfy target with a subset, no repeats); >1 means
each supplying image must be shown that many times per epoch (binding band);
large (>~5-8x) flags over-concentration / low diversity risk for that band.
Reported overall AND restricted to res_bucket==ge512 (the late-stage constraint).

Read-only. Output: {OUTDIR}/rich_v1_qc.json (also printed). No manifest writes.
"""
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

S3 = os.environ.get('MOT_FILTER_OUT', '/s3-pixai-test-uw2/evan/mot_filtering_v1')
OUTDIR = os.environ.get('RICH_OUT', f'{S3}/t5_manifest/rich_v1')
FSX = '/fsx/evan/mot_code/data_analysis/outputs'
t0 = time.time()
TARGET = {1: 0.18, 2: 0.27, 3: 0.15, 4: 0.40}  # v3 caption ratio (T1/T2/T3/T4)
BANDS = [1, 2, 3, 4]
COLS = ['dataset', 'sub_source', 'res_bucket', 'aes', 'aes_scorer', 'cluster_id',
        'dominant_cat', 'caption_tier', 'has_t1', 'has_t2', 'has_t3', 'has_t4',
        'cap_i1', 'cap_original', 'cap_florence', 'cap_internvl', 'cap_sharegpt4v',
        'cap_gemini', 'synthetic', 'holdout', 'width', 'height']
CAPS = ['cap_i1', 'cap_original', 'cap_florence', 'cap_internvl', 'cap_sharegpt4v', 'cap_gemini']


def read_dir(d, label):
    files = sorted(glob.glob(f'{d}/*.parquet'))
    if not files:
        raise SystemExit(f'[{label}] no parquet under {d}')

    def load(f):
        return pq.read_table(f, columns=COLS)
    parts = []
    with ThreadPoolExecutor(max_workers=int(os.environ.get('READ_WORKERS', '48'))) as ex:
        for r in ex.map(load, files):
            parts.append(r)
    # unify dtype across shards (string vs large_string; defensive) then concat
    names = parts[0].schema.names

    def canon(n):
        ts = [p.schema.field(n).type for p in parts]
        nonnull = [t for t in ts if t != pa.null()]
        t = nonnull[0] if nonnull else pa.large_string()
        return pa.large_string() if t in (pa.string(), pa.large_string()) else t
    target = pa.schema([(n, canon(n)) for n in names])
    df = pa.concat_tables([p.cast(target) for p in parts]).to_pandas()
    print(f'[{label}] read {len(files)} shards, {len(df):,} rows ({time.time()-t0:.0f}s)', flush=True)
    return df


def pctiles(s):
    """NaN-aware: a few NaN must not poison the whole stat. Report nan/valid counts
    alongside percentiles computed over the valid subset."""
    a = s.to_numpy(dtype=np.float64)
    nan = int(np.isnan(a).sum())
    v = a[~np.isnan(a)]
    out = {f'p{q}': round(float(np.percentile(v, q)), 3) for q in (1, 5, 25, 50, 75, 95, 99)} if v.size else {}
    out['mean'] = round(float(v.mean()), 3) if v.size else None
    out['n_nan'] = nan
    out['n_valid'] = int(v.size)
    return out


def vc(s, top=None):
    d = s.value_counts()
    if top:
        d = d.head(top)
    return {str(k): int(v) for k, v in d.items()}


def supply_block(g, m, tag):
    """Per-band supply = GPIC tier-band count + MONET has-band count, for the
    given (already-filtered) g/m frames. Plus adequacy vs TARGET over |g|+|m|."""
    gt = {x: int((g['caption_tier'] == x).sum()) for x in BANDS}
    mt = {x: int(m[f'has_t{x}'].sum()) for x in BANDS}
    supply = {x: gt[x] + mt[x] for x in BANDS}
    N = len(g) + len(m)
    adeq = {}
    for x in BANDS:
        want = TARGET[x] * N
        osf = (want / supply[x]) if supply[x] else None
        adeq[x] = {
            'target_share': TARGET[x],
            'wanted_draws': int(round(want)),
            'supply': supply[x],
            'gpic_tier': gt[x], 'monet_has': mt[x],
            'oversample_factor': round(osf, 3) if osf is not None else None,
            'status': ('over-supplied' if osf and osf <= 1 else
                       'ok' if osf and osf <= 5 else
                       'tight' if osf and osf <= 8 else 'OVER-CONCENTRATED'),
        }
    return {'tag': tag, 'pool_N': N, 'per_band': adeq}


def main():
    g = read_dir(f'{OUTDIR}/gpic', 'gpic')
    m = read_dir(f'{OUTDIR}/monet', 'monet')
    N = len(g) + len(m)

    # ---- integrity sanity (should all hold by construction) ----
    sanity = {
        'total': N, 'gpic_rows': len(g), 'monet_rows': len(m),
        'gpic_synth_sum': int(g['synthetic'].sum()), 'monet_synth_sum': int(m['synthetic'].sum()),
        'gpic_holdout_sum': int(g['holdout'].sum()), 'monet_holdout_sum': int(m['holdout'].sum()),
        'gpic_caption_tier_range': [int(g['caption_tier'].min()), int(g['caption_tier'].max())],
        'monet_caption_tier_nonzero': int((m['caption_tier'] != 0).sum()),
        'res_bucket_values': sorted(set(g['res_bucket'].unique()) | set(m['res_bucket'].unique())),
        'gpic_cap_i1_sum': int(g['cap_i1'].sum()), 'monet_cap_i1_sum': int(m['cap_i1'].sum()),
    }

    # ---- resolution ----
    res = {
        'gpic': vc(g['res_bucket']), 'monet': vc(m['res_bucket']),
        'total_ge512': int((g['res_bucket'] == 'ge512').sum() + (m['res_bucket'] == 'ge512').sum()),
    }
    res['total_ge512_share'] = round(res['total_ge512'] / N, 4)

    # ---- aesthetics ----
    aes = {'gpic_laion_improved': pctiles(g['aes']), 'monet_jasperai': pctiles(m['aes'])}

    # ---- clusters (joint balance) ----
    allc = pd.concat([g['cluster_id'], m['cluster_id']], ignore_index=True)
    cc = allc.value_counts()
    csz = cc.to_numpy()
    clusters = {
        'k_total': 20000, 'covered': int(cc.size),
        'size_min': int(csz.min()), 'size_p1': int(np.percentile(csz, 1)),
        'size_p50': int(np.percentile(csz, 50)), 'size_mean': round(float(csz.mean()), 1),
        'size_p99': int(np.percentile(csz, 99)), 'size_max': int(csz.max()),
        'top10_clusters': {str(k): int(v) for k, v in cc.head(10).items()},
        'empty_clusters': 20000 - int(cc.size),
    }

    # ---- content labels ----
    cats = vc(pd.concat([g['dominant_cat'], m['dominant_cat']], ignore_index=True))

    # ---- MONET sub-source + per-captioner coverage ----
    m_aes_nan = m[m['aes'].isna()]
    monet = {
        'sub_source': vc(m['sub_source']),
        'captioner_coverage': {c: int(m[c].sum()) for c in CAPS if c != 'cap_i1'},
        'aes_nan_total': int(len(m_aes_nan)),
        'aes_nan_by_source': vc(m_aes_nan['sub_source']) if len(m_aes_nan) else {},
        'has_band_pattern': vc(
            m['has_t1'].astype(str) + m['has_t2'].astype(str)
            + m['has_t3'].astype(str) + m['has_t4'].astype(str), top=16),
    }
    gpic = {'caption_tier': {str(x): int((g['caption_tier'] == x).sum()) for x in BANDS}}

    # ---- THE KEY TABLE: caption-band supply + adequacy, overall and >=512 ----
    g512, m512 = g[g['res_bucket'] == 'ge512'], m[m['res_bucket'] == 'ge512']
    supply = {
        'overall': supply_block(g, m, 'all-res'),
        'ge512_only': supply_block(g512, m512, 'res>=512 (late stage)'),
    }

    report = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'manifest': OUTDIR,
        'note': ('Read-only QC + caption-band supply audit. oversample_factor = '
                 'TARGET_share*pool / supply; <=1 over-supplied, >1 binding band, '
                 '>8 over-concentrated. ge512_only models late high-res stages.'),
        'target_caption_ratio': TARGET,
        'integrity': sanity,
        'resolution': res,
        'aesthetics': aes,
        'clusters': clusters,
        'content_categories': cats,
        'gpic': gpic,
        'monet': monet,
        'caption_band_supply': supply,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    os.makedirs(FSX, exist_ok=True)
    tmp = f'{FSX}/_t5qc.tmp.json'
    with open(tmp, 'w') as fh:
        json.dump(report, fh, indent=2)
    dst = f'{OUTDIR}/rich_v1_qc.json'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    import shutil
    shutil.copyfile(tmp, dst)
    os.remove(tmp)
    print(json.dumps(report, indent=2), flush=True)
    print(f'WROTE {dst}  TOTAL {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
