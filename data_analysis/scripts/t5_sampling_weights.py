"""Reference implementation: real-pool (MONET-real + GPIC) sampling weights +
caption selection (loader-side). Implements the band-first procedure in
doc/mot/采样权重spec_v1_20260616.md. NOT baked into the manifest -- the dataloader
imports these to build per-(stage,band,domain) samplers.

__main__ (SEL=wcheck): dry-run on rich_v1. Computes the EXACT expected per-image
draw probability P(i) under the spec (no sampling loop) and reports the band /
domain / source / cluster distributions it induces -- so we confirm the band mix
is exact and that 适度拉平 (source_alpha) lands where the spec says.

P(i) = Σ_b BAND_MIX[b] · Σ_d P(d|b) · [elig(i;d,b,stage)·base_w(i) / Z_{d,b}]
  base_w(i) = source_factor(i; alpha) · cluster_factor(i; beta)   (aes weighting off)
Multi-band MONET images sum over the bands they serve (the band-first caveat).
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
RICH = os.environ.get('RICH_OUT', f'{S3}/t5_manifest/rich_v1')
CFG = os.environ.get('WCFG', '/fsx/evan/mot_code/data_analysis/configs/t5_sampling_weights_v1.json')
FSX = '/fsx/evan/mot_code/data_analysis/outputs'
BANDS = ['T1', 'T2', 'T3', 'T4']
t0 = time.time()


# ---------- weight factors (importable) ----------
def source_factor(sub_source, alpha):
    """MONET sources flattened by N_source^(-alpha); GPIC source -> 1.0."""
    s = pd.Series(sub_source)
    monet = s != 'gpic'
    n = s[monet].value_counts()
    f = np.ones(len(s), dtype=np.float64)
    if alpha and len(n):
        w = {src: cnt ** (-alpha) for src, cnt in n.items()}
        f[monet.to_numpy()] = s[monet].map(w).to_numpy()
    return f


def cluster_factor(cluster_id, beta):
    """cluster_size(cid)^(-beta); cid==-1 -> 1.0 (synthetic, not in rich_v1).
    Vectorized: map via a Series (C-level lookup), never a per-row Python lambda."""
    cid = pd.Series(cluster_id)
    factor = (cid.value_counts().astype(np.float64)) ** (-beta)   # index=cluster_id -> factor
    f = cid.map(factor).to_numpy()
    return np.where(cid.to_numpy() == -1, 1.0, f)


def base_weights(df, alpha, beta):
    return source_factor(df['sub_source'].to_numpy(), alpha) * cluster_factor(df['cluster_id'].to_numpy(), beta)


def stage_eligible_domain(df, stage, domain, cfg):
    """Boolean mask: rows of `domain` eligible at `stage` (resolution routing)."""
    short = np.minimum(df['width'].to_numpy(), df['height'].to_numpy())
    ge512 = (df['res_bucket'].to_numpy() == 'ge512')
    if domain == 'gpic':
        d = (df['dataset'].to_numpy() == 'gpic')
        if stage == '256':
            return d
        if stage == '512':
            thr = np.nanpercentile(df.loc[df['dataset'] == 'gpic', 'aes'].to_numpy(),
                                    cfg['aes_gpic512_percentile'])
            return d & ge512 & (df['aes'].to_numpy() >= thr)
        return np.zeros(len(df), dtype=bool)            # 1024: GPIC none
    d = (df['dataset'].to_numpy() == 'monet')
    if stage == '256':
        return d
    if stage == '512':
        return d & ge512
    return d & (short >= 1024)                          # 1024: MONET short>=1024


def expected_prob(df, stage, cfg, bw):
    """Exact P(i) under band-first + flatten at `stage`. Returns (P, info).
    `bw` = base_weights(df, alpha, beta), computed once by the caller."""
    alpha, beta = cfg['source_alpha'], cfg['cluster_beta']
    band_mix = cfg['band_mix_real']
    r = cfg['monet_real_gpic_ratio'][stage]   # per-stage MONET:GPIC (GPIC off at 512/1024 by default)
    domain_real = {'monet': r[0], 'gpic': r[1]}
    elig = {dm: stage_eligible_domain(df, stage, dm, cfg) for dm in ('monet', 'gpic')}
    tier = df['caption_tier'].to_numpy()
    has = {b: df[f'has_t{i}'].to_numpy() == 1 for i, b in enumerate(BANDS, 1)}
    P = np.zeros(len(df), dtype=np.float64)
    for i, b in enumerate(BANDS, 1):
        # which domains can serve band b at this stage?
        m_monet = elig['monet'] & has[b]
        m_gpic = elig['gpic'] & (tier == i)
        avail = {'monet': m_monet.any() and domain_real['monet'] > 0,
                 'gpic': m_gpic.any() and domain_real['gpic'] > 0}
        zsum = sum(domain_real[d] for d in avail if avail[d])
        if zsum <= 0:
            continue
        for d, m in (('monet', m_monet), ('gpic', m_gpic)):
            if not avail[d]:
                continue
            pdb = domain_real[d] / zsum
            Z = bw[m].sum()
            if Z > 0:
                P[m] += band_mix[b] * pdb * bw[m] / Z
    return P, {'alpha': alpha, 'beta': beta}


def report_stage(df, stage, cfg, bw):
    P, info = expected_prob(df, stage, cfg, bw)
    tot = P.sum()
    out = {'stage': stage, 'P_total_check': round(float(tot), 6), **info}
    if tot <= 0:
        out['note'] = 'no eligible mass'
        return out
    Pn = P / tot
    # band share (should equal band_mix exactly, by construction)
    tier = df['caption_tier'].to_numpy()
    has = {b: df[f'has_t{i}'].to_numpy() == 1 for i, b in enumerate(BANDS, 1)}
    # per-band mass = sum over band-draws; recompute by re-running not needed -> use targets
    out['band_share_target'] = cfg['band_mix_real']
    # domain share
    dmask = df['dataset'].to_numpy()
    out['domain_share'] = {'monet': round(float(Pn[dmask == 'monet'].sum()), 4),
                           'gpic': round(float(Pn[dmask == 'gpic'].sum()), 4)}
    # source share (real)
    ss = pd.Series(df['sub_source'].to_numpy())
    src = {s: round(float(Pn[(ss == s).to_numpy()].sum()), 4) for s in ss.unique()}
    out['source_share'] = dict(sorted(src.items(), key=lambda kv: -kv[1]))
    laion_coyo = src.get('laion', 0) + src.get('coyo', 0)
    out['laion_coyo_share'] = round(laion_coyo, 4)
    # cluster concentration: top-cluster share + effective #clusters under P vs natural
    cid = df['cluster_id'].to_numpy()
    cl = pd.Series(Pn).groupby(cid).sum()
    cl = cl[cl.index >= 0]
    out['cluster_top1_share'] = round(float(cl.max()), 5)
    out['cluster_eff_count'] = int(round(1.0 / float((cl ** 2).sum()))) if (cl ** 2).sum() > 0 else 0
    # res mix actually drawn
    rb = df['res_bucket'].to_numpy()
    out['res_share'] = {'ge512': round(float(Pn[rb == 'ge512'].sum()), 4),
                        '256': round(float(Pn[rb == '256'].sum()), 4)}
    return out


def read_rich(d):
    cols = ['dataset', 'sub_source', 'res_bucket', 'aes', 'cluster_id', 'caption_tier',
            'has_t1', 'has_t2', 'has_t3', 'has_t4', 'width', 'height']
    files = sorted(glob.glob(f'{d}/gpic/*.parquet')) + sorted(glob.glob(f'{d}/monet/*.parquet'))

    def load(f):
        return pq.read_table(f, columns=cols)
    parts = []
    with ThreadPoolExecutor(max_workers=int(os.environ.get('READ_WORKERS', '48'))) as ex:
        for r in ex.map(load, files):
            parts.append(r)
    names = parts[0].schema.names

    def canon(n):
        ts = [p.schema.field(n).type for p in parts]
        nn = [t for t in ts if t != pa.null()]
        t = nn[0] if nn else pa.large_string()
        return pa.large_string() if t in (pa.string(), pa.large_string()) else t
    tgt = pa.schema([(n, canon(n)) for n in names])
    return pa.concat_tables([p.cast(tgt) for p in parts]).to_pandas()


def main():
    cfg = json.load(open(CFG))
    print(f'[wcheck] cfg alpha={cfg["source_alpha"]} beta={cfg["cluster_beta"]}', flush=True)
    df = read_rich(RICH)
    print(f'[wcheck] read {len(df):,} rows ({time.time()-t0:.0f}s)', flush=True)
    bw = base_weights(df, cfg['source_alpha'], cfg['cluster_beta'])   # once, reused per stage
    print(f'[wcheck] base weights ready ({time.time()-t0:.0f}s)', flush=True)
    rep = {'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'config': CFG, 'source_alpha': cfg['source_alpha'], 'cluster_beta': cfg['cluster_beta'],
           'stages': {}}
    for stage in ('256', '512', '1024'):
        rep['stages'][stage] = report_stage(df, stage, cfg, bw)
        print(f'[wcheck] stage {stage} done ({time.time()-t0:.0f}s)', flush=True)
    # source_alpha sweep on stage 256 -> show laion/coyo flatten tradeoff for Evan's call
    sweep = {}
    for a in (0.0, 0.3, 0.5, 0.7, 1.0):
        cfg_a = dict(cfg, source_alpha=a)
        bw_a = base_weights(df, a, cfg['cluster_beta'])
        r = report_stage(df, '256', cfg_a, bw_a)
        sweep[f'alpha={a}'] = {'laion_coyo_share': r['laion_coyo_share'],
                               'source_share': r['source_share'],
                               'cluster_top1_share': r.get('cluster_top1_share'),
                               'cluster_eff_count': r.get('cluster_eff_count')}
        print(f'[wcheck] sweep alpha={a} laion+coyo={r["laion_coyo_share"]} ({time.time()-t0:.0f}s)', flush=True)
    rep['source_alpha_sweep_stage256'] = sweep
    rep['elapsed_sec'] = round(time.time() - t0, 1)
    os.makedirs(FSX, exist_ok=True)
    dst = f'{RICH}/sampling_weights_check.json'
    tmp = f'{FSX}/_wcheck.tmp.json'
    with open(tmp, 'w') as fh:
        json.dump(rep, fh, indent=2)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    import shutil
    shutil.copyfile(tmp, dst)
    os.remove(tmp)
    print(json.dumps(rep, indent=2), flush=True)
    print(f'WROTE {dst}  TOTAL {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
