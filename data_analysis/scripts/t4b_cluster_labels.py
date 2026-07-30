"""T4b: coarse content labels for the 20000 T4 clusters via CLIP zero-shot.

Method = MEMBER VOTING (high quality): for a large random sample of member images
we run image-level zero-shot (cosine to each category prompt-ensemble, argmax) and
take the per-cluster plurality. Image-level argmax is far more reliable than argmax
on the smoothed centroid (which over-fires rare categories like render/wedding).

Same CLIP space as clip_b: open_clip ViT-B-32-quickgelu/openai text encoder.
Validated by rank-comparing the GPIC dominant-label distribution to the
independently-computed caption_category_shares.json. CPU-only.
"""
import glob
import json
import os
import shutil
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
N_GPIC_SHARDS, N_MONET_SHARDS = 280, 340     # ~5M sampled members -> ~250/cluster
K = 20000

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


def put(localfn, dst):
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(localfn, dst)


def decode_clipb(vals):
    good = np.fromiter((v is not None and len(v) == 1024 for v in vals), dtype=bool, count=len(vals))
    gi = np.flatnonzero(good)
    if len(gi) == 0:
        return np.empty((0, 512), np.float32), gi
    buf = b''.join(vals[i] for i in gi)
    x = np.frombuffer(buf, dtype=np.float16).reshape(len(gi), 512).astype(np.float32)
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    return x, gi


def main():
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[t4b] device={device} loading {MODEL}/{PRETRAINED}', flush=True)
    model = open_clip.create_model(MODEL, pretrained=PRETRAINED, cache_dir=CACHE).eval().to(device)
    tok = open_clip.get_tokenizer(MODEL)
    cat_emb = np.zeros((len(CAT_NAMES), 512), dtype=np.float32)
    with torch.no_grad():
        for i, name in enumerate(CAT_NAMES):
            prompts = [tpl.format(term) for term in CATS[name] for tpl in TEMPLATES]
            te = torch.nn.functional.normalize(model.encode_text(tok(prompts).to(device)).float(), dim=1)
            cat_emb[i] = torch.nn.functional.normalize(te.mean(0, keepdim=True), dim=1).cpu().numpy()[0]
    print(f'[t4b] encoded {len(CAT_NAMES)} cats x {len(TEMPLATES)} templates ({time.time()-t0:.0f}s)', flush=True)
    cat_emb_t = cat_emb.T  # (512, 11)

    # ---- sampled member voting ----
    gpic_all = sorted(glob.glob(f'{GPIC}/gpic_pass_*.parquet'))
    monet_all = sorted(glob.glob(f'{MONET_EMB}/*.parquet'))
    gpic_s = gpic_all[::max(1, len(gpic_all) // N_GPIC_SHARDS)]
    monet_s = monet_all[::max(1, len(monet_all) // N_MONET_SHARDS)]
    print(f'[t4b] voting on {len(gpic_s)} gpic + {len(monet_s)} monet shards', flush=True)

    def vote_shard(spec):
        kind, f = spec
        try:
            if kind == 'gpic':
                shard = os.path.basename(f).replace('gpic_pass_', '').replace('.parquet', '')
                cf = f'{T4}/gpic_clusters/gpic_cluster_{shard}.parquet'
                idcol = 'key'
            else:
                cf = f'{T4}/monet_clusters/monet_cluster_{os.path.basename(f)}'
                idcol = 'id'
            if not os.path.exists(cf):
                return None
            emb = pq.read_table(f, columns=[idcol, 'clip_b']).to_pandas()
            clu = pq.read_table(cf, columns=[idcol, 'cluster_id']).to_pandas()
            m = emb.merge(clu, on=idcol, how='inner')
            if not len(m):
                return None
            vals = m['clip_b'].to_numpy()
            x, gi = decode_clipb(vals)
            if not len(gi):
                return None
            cats = np.argmax(x @ cat_emb_t, axis=1).astype(np.int16)
            cl = m['cluster_id'].to_numpy()[gi].astype(np.int64)
            return cl, cats
        except Exception as e:  # noqa
            return ('ERR', f'{os.path.basename(f)}: {e}')

    votes = np.zeros((K, len(CAT_NAMES)), dtype=np.int64)
    specs = [('gpic', f) for f in gpic_s] + [('monet', f) for f in monet_s]
    errs = 0
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(vote_shard, specs):
            done += 1
            if r is None:
                continue
            if isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], str):
                errs += 1
                continue
            cl, cats = r
            valid = (cl >= 0) & (cl < K)
            np.add.at(votes, (cl[valid], cats[valid]), 1)
            if done % 100 == 0:
                print(f'[vote] {done}/{len(specs)} total_votes={int(votes.sum()):,} errs={errs} '
                      f'({time.time()-t0:.0f}s)', flush=True)
    nvotes = votes.sum(1)
    print(f'[vote] done total_votes={int(votes.sum()):,} clusters_with_votes={int((nvotes>0).sum())}/{K} '
          f'errs={errs} ({time.time()-t0:.0f}s)', flush=True)

    # centroid cosine (auxiliary)
    cent = np.load(f'{T4}/centroids.npy').astype(np.float32)
    cent /= np.clip(np.linalg.norm(cent, axis=1, keepdims=True), 1e-8, None)
    cent_cos = cent @ cat_emb.T

    top1 = np.where(nvotes > 0, np.argmax(votes, axis=1), -1)
    vote_frac = votes / np.clip(nvotes[:, None], 1, None)
    purity = np.where(nvotes > 0, votes.max(1) / np.clip(nvotes, 1, None), np.nan)
    # fallback to centroid label for the (rare) clusters with no sampled members
    cent_top1 = np.argmax(cent_cos, axis=1)
    dom = np.where(top1 >= 0, top1, cent_top1)

    df = pd.DataFrame({'cluster_id': np.arange(K, dtype=np.int32)})
    df['dominant_cat'] = [CAT_NAMES[i] for i in dom]
    df['n_votes'] = nvotes
    df['purity'] = purity.astype(np.float32)
    df['label_source'] = np.where(top1 >= 0, 'member_vote', 'centroid_fallback')
    for i, name in enumerate(CAT_NAMES):
        df[f'vfrac_{name.replace("/", "_")}'] = vote_frac[:, i].astype(np.float32)
    df['person_frac'] = vote_frac[:, CAT_NAMES.index('person')].astype(np.float32)
    df['text_frac'] = vote_frac[:, CAT_NAMES.index('text-rich')].astype(np.float32)
    df['person_flag'] = (df['person_frac'] >= 0.5).astype(np.int8)
    df['text_flag'] = (df['text_frac'] >= 0.5).astype(np.int8)
    df['low_conf'] = ((purity < 0.4) & (nvotes > 0)).astype(np.int8)
    tmp = f'{FSX}/_t4b_labels.tmp.parquet'
    df.to_parquet(tmp, index=False)
    put(tmp, f'{T4}/cluster_content_labels.parquet')
    os.remove(tmp)

    # ---- validation vs caption_category_shares.json (GPIC img-weighted dominant share) ----
    cs = pd.read_parquet(f'{T4}/cluster_stats.parquet', columns=['cluster_id', 'gpic_count', 'monet_count'])
    cs = cs.merge(df[['cluster_id', 'dominant_cat']], on='cluster_id')
    gpic_share = (cs.groupby('dominant_cat')['gpic_count'].sum() / cs['gpic_count'].sum()).reindex(CAT_NAMES).fillna(0)
    ext = json.load(open(f'{FSX}/caption_category_shares.json'))
    ext_gpic = {k: ext[k].get('GPIC/all', 0) for k in CAT_NAMES}
    clu_dist = df['dominant_cat'].value_counts().reindex(CAT_NAMES).fillna(0).astype(int)

    summary = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'method': 'member_vote', 'model': f'{MODEL}/{PRETRAINED}', 'k': int(K),
        'total_votes': int(votes.sum()), 'clusters_voted': int((nvotes > 0).sum()),
        'mean_votes_per_cluster': int(nvotes.mean()), 'mean_purity': round(float(np.nanmean(purity)), 4),
        'clusters_per_dominant_cat': {k: int(clu_dist[k]) for k in CAT_NAMES},
        'gpic_imgweighted_dominant_share': {k: round(float(gpic_share[k]), 4) for k in CAT_NAMES},
        'external_caption_share_gpic_all': {k: round(float(ext_gpic[k]), 4) for k in CAT_NAMES},
        'low_conf_clusters': int(df['low_conf'].sum()),
        'person_clusters': int(df['person_flag'].sum()), 'text_clusters': int(df['text_flag'].sum()),
        'elapsed_sec': round(time.time() - t0, 1),
    }
    tmp = f'{FSX}/_t4b_summary.tmp.json'
    with open(tmp, 'w') as fh:
        json.dump(summary, fh, indent=2)
    put(tmp, f'{T4}/cluster_content_labels_summary.json')
    os.remove(tmp)

    print('\n=== per-dominant-cat: #clusters | GPIC img-share | ext caption-share(GPIC) ===', flush=True)
    print(f'{"category":18s}{"#clu":>7s}{"gpic_dom":>10s}{"ext_cap":>9s}', flush=True)
    for k in CAT_NAMES:
        print(f'{k:18s}{int(clu_dist[k]):7d}{gpic_share[k]:10.4f}{ext_gpic[k]:9.4f}', flush=True)
    my_rank = list(gpic_share.sort_values(ascending=False).index)
    ext_rank = [k for k, _ in sorted(ext_gpic.items(), key=lambda x: -x[1])]
    print(f'\nmy top5:  {my_rank[:5]}', flush=True)
    print(f'ext top5: {ext_rank[:5]}', flush=True)
    print(f'mean_purity={summary["mean_purity"]} mean_votes/clu={summary["mean_votes_per_cluster"]} '
          f'low_conf={summary["low_conf_clusters"]} person_clu={summary["person_clusters"]} '
          f'text_clu={summary["text_clusters"]}', flush=True)
    print(f'WROTE {T4}/cluster_content_labels.parquet  TOTAL {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
