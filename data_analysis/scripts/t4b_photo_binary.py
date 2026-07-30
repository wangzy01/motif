"""Binary photo-vs-nonphoto re-split of the real (MONET+GPIC) candidate pool.

Reuses t4b_cluster_labels.py machinery (stored clip_b embeddings, ViT-B-32-quickgelu/openai,
member sampling, CPU-only cosine) but, per sampled member image, computes BOTH:
  - the original 11-class content argmax (cat11)  -- to reproduce/cross-check the dominant_cat buckets
  - a 2-class photo vs non-photo argmax (binary)   -- the missing true photo/illustration signal
and cross-tabs counts by (sub_source, cat11, binary). This answers:
  (a) true photo% per MONET sub_source + GPIC, and
  (b) the photo% INSIDE the noisy render/3d (and any) 11-class bucket (the over-fire re-split).

Output: /fsx/evan/mot_code/data_analysis/outputs/photo_binary_{summary.json,counts.parquet}.
Read-only on all source data; writes only small analysis artifacts to FSX. CPU-only, ~1-2 min.
Aggregation is over the CLUSTERED CANDIDATE pool (monet_emb 77M + gpic_pass 99M survivors), used as a
proxy for rich_v1 60M content distribution (prior audit reconciled the 11-class dist to rich_v1_qc <0.1pp).
"""
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import open_clip

S3 = '/s3-pixai-test-uw2/evan/mot_filtering_v1'
T4 = f'{S3}/t4_clipb_joint_kmeans'
MONET_EMB = f'{S3}/monet_emb'
GPIC = f'{S3}/gpic_pass'
FSX = '/fsx/evan/mot_code/data_analysis/outputs'
MODEL, PRETRAINED, CACHE = 'ViT-B-32-quickgelu', 'openai', '/fsx/hf_cache/open_clip'
N_GPIC_SHARDS, N_MONET_SHARDS = 280, 340

# ---- 11-class (identical to t4b_cluster_labels.py) ----
CATS = {
    'person': ['a person', 'people', 'a portrait of a person', 'a man', 'a woman'],
    'wedding/bride': ['a bride in a wedding dress', 'a wedding ceremony', 'a bride and groom'],
    'illustration/art': ['an illustration', 'a digital painting', 'a drawing', 'concept art'],
    'render/3d': ['a 3d rendered image', 'a cgi render', 'a video game screenshot'],
    'text-rich': ['a page full of text', 'a poster with large text', 'a document', 'an infographic with text'],
    'white-bg/product': ['a product on a plain white background', 'an object isolated on a white background'],
    'food': ['a plate of food', 'a delicious meal', 'a dish of food'],
    'animal': ['an animal', 'wildlife', 'a pet dog or cat', 'a wild animal'],
    'landscape/nature': ['a landscape', 'natural scenery', 'a mountain view', 'a beach', 'a forest', 'a field'],
    'building/city': ['a building', 'a city street', 'architecture', 'a cityscape', 'the exterior of a house'],
    'car/vehicle': ['a car', 'a motorcycle', 'a truck', 'an airplane', 'a boat'],
}
CAT_NAMES = list(CATS.keys())
TEMPLATES = ['a photo of {}.', 'a photo of the {}.', 'a close-up photo of {}.', '{}.',
             'a cropped photo of {}.', 'a bright photo of {}.', 'a picture of {}.', 'a good photo of {}.']

# ---- binary photo vs non-photo. NEUTRAL templates (NO "a photo of": that would bias everything to photo). ----
BIN = {
    'photo': ['a photograph', 'a real photograph', 'a photo taken with a camera',
              'a photograph of a real scene', 'a real-life photo'],
    'nonphoto': ['an illustration', 'a digital painting', 'a drawing', 'a 3d render',
                 'a cgi rendering', 'a cartoon', 'clip art', 'a poster', 'a graphic design',
                 'a screenshot', 'anime art', 'a sketch'],
}
BIN_NAMES = list(BIN.keys())  # index 0 = photo, 1 = nonphoto
BIN_TEMPLATES = ['{}.']

# sub_source -> int code (gpic=0); unknown monet source -> 'other'
SOURCES = ['gpic', 'laion', 'coyo', 'cc12m', 'commoncatalog-cc-by', 'megalith10m', 'da4k', 'other']
SRC_IDX = {s: i for i, s in enumerate(SOURCES)}
NSRC, NCAT, NBIN = len(SOURCES), len(CAT_NAMES), len(BIN_NAMES)
CELLS = NSRC * NCAT * NBIN


def decode_clipb(vals):
    good = np.fromiter((v is not None and len(v) == 1024 for v in vals), dtype=bool, count=len(vals))
    gi = np.flatnonzero(good)
    if len(gi) == 0:
        return np.empty((0, 512), np.float32), gi
    buf = b''.join(vals[i] for i in gi)
    x = np.frombuffer(buf, dtype=np.float16).reshape(len(gi), 512).astype(np.float32)
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    return x, gi


def encode_cats(model, tok, device, cats, templates):
    names = list(cats.keys())
    emb = np.zeros((len(names), 512), dtype=np.float32)
    with torch.no_grad():
        for i, name in enumerate(names):
            prompts = [tpl.format(term) for term in cats[name] for tpl in templates]
            te = torch.nn.functional.normalize(model.encode_text(tok(prompts).to(device)).float(), dim=1)
            emb[i] = torch.nn.functional.normalize(te.mean(0, keepdim=True), dim=1).cpu().numpy()[0]
    return emb


def main():
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[photobin] device={device} loading {MODEL}/{PRETRAINED}', flush=True)
    model = open_clip.create_model(MODEL, pretrained=PRETRAINED, cache_dir=CACHE).eval().to(device)
    tok = open_clip.get_tokenizer(MODEL)
    cat_t = encode_cats(model, tok, device, CATS, TEMPLATES).T          # (512, 11)
    bin_t = encode_cats(model, tok, device, BIN, BIN_TEMPLATES).T       # (512, 2)
    print(f'[photobin] encoded {NCAT} cats + {NBIN} bin classes ({time.time()-t0:.0f}s)', flush=True)

    gpic_all = sorted(glob.glob(f'{GPIC}/gpic_pass_*.parquet'))
    monet_all = sorted(glob.glob(f'{MONET_EMB}/*.parquet'))
    gpic_s = gpic_all[::max(1, len(gpic_all) // N_GPIC_SHARDS)]
    monet_s = monet_all[::max(1, len(monet_all) // N_MONET_SHARDS)]
    print(f'[photobin] voting on {len(gpic_s)} gpic + {len(monet_s)} monet shards', flush=True)

    def vote_shard(spec):
        kind, f = spec
        try:
            if kind == 'gpic':
                shard = os.path.basename(f).replace('gpic_pass_', '').replace('.parquet', '')
                cf = f'{T4}/gpic_clusters/gpic_cluster_{shard}.parquet'
                idcol, ccols = 'key', ['key', 'cluster_id']
            else:
                cf = f'{T4}/monet_clusters/monet_cluster_{os.path.basename(f)}'
                idcol, ccols = 'id', ['id', 'source', 'cluster_id']
            if not os.path.exists(cf):
                return None
            emb = pq.read_table(f, columns=[idcol, 'clip_b']).to_pandas()
            clu = pq.read_table(cf, columns=ccols).to_pandas()
            m = emb.merge(clu, on=idcol, how='inner')
            if not len(m):
                return None
            x, gi = decode_clipb(m['clip_b'].to_numpy())
            if not len(gi):
                return None
            c11 = np.argmax(x @ cat_t, axis=1).astype(np.int64)          # (n,)
            cbin = np.argmax(x @ bin_t, axis=1).astype(np.int64)         # (n,) 0=photo 1=nonphoto
            if kind == 'gpic':
                src = np.zeros(len(gi), dtype=np.int64)                  # gpic == 0
            else:
                ss = m['source'].to_numpy()[gi]
                src = np.array([SRC_IDX.get(s, SRC_IDX['other']) for s in ss], dtype=np.int64)
            flat = (src * NCAT + c11) * NBIN + cbin
            return np.bincount(flat, minlength=CELLS)
        except Exception as e:  # noqa
            return ('ERR', f'{os.path.basename(f)}: {e}')

    counts = np.zeros(CELLS, dtype=np.int64)
    specs = [('gpic', f) for f in gpic_s] + [('monet', f) for f in monet_s]
    errs, done = 0, 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(vote_shard, specs):
            done += 1
            if r is None:
                continue
            if isinstance(r, tuple):
                errs += 1
                print(f'[err] {r[1]}', flush=True)
                continue
            counts += r
            if done % 100 == 0:
                print(f'[vote] {done}/{len(specs)} total={int(counts.sum()):,} errs={errs} ({time.time()-t0:.0f}s)', flush=True)

    C = counts.reshape(NSRC, NCAT, NBIN)  # [source, cat11, bin]
    total = int(C.sum())
    print(f'[vote] done total={total:,} errs={errs} ({time.time()-t0:.0f}s)', flush=True)

    def photo_frac(arr2):  # arr2 shape (...,2) -> photo / (photo+nonphoto)
        ph, npho = arr2[..., 0].sum(), arr2[..., 1].sum()
        return round(float(ph) / max(1.0, float(ph + npho)), 4)

    # per-source photo fraction + counts
    per_source = {}
    for si, s in enumerate(SOURCES):
        n = int(C[si].sum())
        if n == 0:
            continue
        per_source[s] = {'n': n, 'photo_frac': photo_frac(C[si]),
                         'cat11_photo_frac': {CAT_NAMES[ci]: photo_frac(C[si, ci]) for ci in range(NCAT) if C[si, ci].sum() > 0}}

    # MONET-aggregate (all sources except gpic) vs GPIC
    monet_mask = np.array([s != 'gpic' for s in SOURCES])
    Cmonet = C[monet_mask].sum(0)   # (cat11, bin)
    Cgpic = C[SRC_IDX['gpic']]      # (cat11, bin)
    agg = {
        'MONET_all': {'n': int(Cmonet.sum()), 'photo_frac': photo_frac(Cmonet),
                      'cat11_photo_frac': {CAT_NAMES[ci]: photo_frac(Cmonet[ci]) for ci in range(NCAT)},
                      'cat11_share': {CAT_NAMES[ci]: round(float(Cmonet[ci].sum()) / max(1, Cmonet.sum()), 4) for ci in range(NCAT)}},
        'GPIC_all': {'n': int(Cgpic.sum()), 'photo_frac': photo_frac(Cgpic),
                     'cat11_photo_frac': {CAT_NAMES[ci]: photo_frac(Cgpic[ci]) for ci in range(NCAT)},
                     'cat11_share': {CAT_NAMES[ci]: round(float(Cgpic[ci].sum()) / max(1, Cgpic.sum()), 4) for ci in range(NCAT)}},
    }

    summary = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'model': f'{MODEL}/{PRETRAINED}', 'device': device,
        'bin_prompts': BIN, 'bin_templates': BIN_TEMPLATES,
        'note': 'photo_frac = photo/(photo+nonphoto), binary CLIP argmax over sampled candidate-pool members; '
                'proxy for rich_v1 content dist. cat11_photo_frac re-splits each 11-class bucket (esp render/3d).',
        'total_members': total, 'errs': errs, 'elapsed_sec': round(time.time() - t0, 1),
        'aggregate': agg, 'per_source': per_source,
    }
    os.makedirs(FSX, exist_ok=True)
    with open(f'{FSX}/photo_binary_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)

    # tidy long-form counts parquet
    rows = []
    for si, s in enumerate(SOURCES):
        for ci in range(NCAT):
            for bi in range(NBIN):
                if C[si, ci, bi]:
                    rows.append({'source': s, 'cat11': CAT_NAMES[ci], 'bin': BIN_NAMES[bi], 'count': int(C[si, ci, bi])})
    pd.DataFrame(rows).to_parquet(f'{FSX}/photo_binary_counts.parquet', index=False)

    # console report
    print('\n=== photo_frac (photo / photo+nonphoto) by source ===', flush=True)
    print(f'{"source":22s}{"n":>12s}{"photo%":>9s}', flush=True)
    for s in SOURCES:
        if s in per_source:
            print(f'{s:22s}{per_source[s]["n"]:12,d}{per_source[s]["photo_frac"]*100:8.1f}%', flush=True)
    print(f'\nMONET_all photo% = {agg["MONET_all"]["photo_frac"]*100:.1f}   GPIC_all photo% = {agg["GPIC_all"]["photo_frac"]*100:.1f}', flush=True)
    print('\n=== render/3d bucket re-split (photo% inside dominant_cat==render/3d-style argmax) ===', flush=True)
    for nm, d in agg.items():
        print(f'{nm}: render/3d photo% = {d["cat11_photo_frac"]["render/3d"]*100:.1f}  '
              f'(render/3d share {d["cat11_share"]["render/3d"]*100:.1f}%);  '
              f'illustration/art photo% = {d["cat11_photo_frac"]["illustration/art"]*100:.1f}', flush=True)
    print(f'\nWROTE {FSX}/photo_binary_summary.json (+counts.parquet)  TOTAL {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
