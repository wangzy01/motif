#!/usr/bin/env python3
"""Independent ground-truth validation of the SSCD cross-source drop list.

This is a SECOND, deliberately independent implementation of the max-cosine
search used to cross-check `t2_sscd_bruteforce_sharded.py`'s output. It:

  1. Loads the 4 FSX-cached MONET chunks (the exact vectors the full run used),
     one at a time, in fp16.
  2. Samples a deterministic subset of GPIC pass shards, applies the SAME gate
     and SAME sscd byte-parsing as the run (so the query *population* matches),
     then recomputes each sampled query's exact max cosine vs ALL ~77M MONET
     vectors -- but in **fp32 accumulation with an fp32 reduction** (stricter
     than the run's fp16-reduced path), so it is a genuine reference, not a copy.
  3. Compares the fp32 ground truth against the official drop parquet:
       FP = run dropped a query whose true max-cos < thr  (threshold/merge bug)
       FN = run kept   a query whose true max-cos >= thr  (coverage bug)
     and reports precision / recall / cosine agreement + the sample drop rate
     vs the global rate.

Only the parsing helpers (bytes_to_matrix, gpic_basic_mask) are copied verbatim
from the run -- those DEFINE the query set and must match by construction; the
search itself is reimplemented independently.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

S3 = '/s3-pixai-test-uw2/evan/mot_filtering_v1'


# ---- parsing helpers: copied verbatim from the run (define the query set) ----
def bytes_to_matrix(vals, dim=512):
    arr = np.empty((len(vals), dim), dtype=np.float16)
    good = np.zeros(len(vals), dtype=bool)
    want = dim * 2
    nout = 0
    for i, v in enumerate(vals):
        if v is None or len(v) != want:
            continue
        x = np.frombuffer(v, dtype=np.float16)
        if x.size != dim:
            continue
        xf = x.astype(np.float32)
        norm = np.linalg.norm(xf)
        if not np.isfinite(norm) or norm <= 0:
            continue
        arr[nout] = (xf / norm).astype(np.float16, copy=False)
        good[i] = True
        nout += 1
    return arr[:nout], good


def gpic_basic_mask(width, height, ok):
    short = np.minimum(width, height)
    long = np.maximum(width, height)
    ar = long / np.clip(short, 1, None)
    return ok & (short >= 256) & (ar <= 3.0)


def load_gpic_sample(gpic_files, sample_shards, seed):
    """Deterministically pick `sample_shards` GPIC shards and return (keys, xq)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(gpic_files), size=min(sample_shards, len(gpic_files)),
                             replace=False))
    picked = [gpic_files[i] for i in idx]
    keys_all, xq_all = [], []
    for j, f in enumerate(picked):
        tab = pq.read_table(f, columns=['key', 'ok', 'width', 'height', 'sscd'])
        ok = tab['ok'].to_numpy(zero_copy_only=False).astype(bool)
        width = tab['width'].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        height = tab['height'].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        gate = gpic_basic_mask(width, height, ok)
        vals = np.asarray(tab['sscd'].to_pylist(), dtype=object)
        xq, good = bytes_to_matrix(vals)
        good_idx = np.flatnonzero(good)
        if len(xq):
            gate_good = gate[good_idx]
            xq = xq[gate_good]
            keys = np.asarray(tab['key'].to_pylist(), dtype=object)[good_idx][gate_good]
            keys_all.append(keys)
            xq_all.append(xq)
        print(f'[sample] shard {j+1}/{len(picked)} {os.path.basename(f)} '
              f'queries+={len(xq) if len(xq) else 0}', flush=True)
    if not xq_all:
        raise SystemExit('no GPIC queries sampled')
    keys = np.concatenate(keys_all)
    xq = np.concatenate(xq_all, axis=0)
    # de-dup keys within the sample (content-hash keys should be unique, but be safe)
    _, uniq = np.unique(keys, return_index=True)
    uniq = np.sort(uniq)
    return keys[uniq], xq[uniq]


def gt_max_cos(q32, chunk_paths, qb, sub, device):
    """Exact max cosine of each query vs ALL chunks, fp32 accumulation+reduction.

    One chunk resident at a time (fp16, ~19GB); subtiles upcast to fp32 for the
    matmul so accumulation is full-precision -- a stricter reference than the
    run's fp16-reduced path.
    """
    Q = q32.shape[0]
    best = torch.full((Q,), -2.0, device=device, dtype=torch.float32)
    best_chunk = torch.full((Q,), -1, device=device, dtype=torch.int32)
    for cid, path in enumerate(chunk_paths):
        t0 = time.time()
        xb = np.load(path)                       # fp16 (Ni,512), already normalized
        mt = torch.from_numpy(xb).to(device)     # fp16 resident
        del xb
        n = mt.shape[0]
        for qs in range(0, Q, qb):
            qe = min(Q, qs + qb)
            qbt = q32[qs:qe]
            cur = best[qs:qe].clone()
            curc = best_chunk[qs:qe].clone()
            for ms in range(0, n, sub):
                me = min(n, ms + sub)
                sims = qbt @ mt[ms:me].float().t()   # fp32 matmul (qb, sub)
                tmax, _ = sims.max(dim=1)            # fp32 reduction
                upd = tmax > cur
                cur = torch.where(upd, tmax, cur)
                curc = torch.where(upd, torch.full_like(curc, cid), curc)
            best[qs:qe] = cur
            best_chunk[qs:qe] = curc
        del mt
        torch.cuda.empty_cache()
        print(f'[gt] chunk {cid} n={n:,} done in {time.time()-t0:.0f}s', flush=True)
    return best.cpu().numpy(), best_chunk.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--gpic-pass', default=f'{S3}/gpic_pass')
    ap.add_argument('--drop-parquet',
                    default=f'{S3}/t2_sscd_bruteforce/gpic_sscd_drop_ids.parquet')
    ap.add_argument('--out', default=f'{S3}/t2_sscd_bruteforce/validate_groundtruth.json')
    ap.add_argument('--n-tag', type=int, default=10881)
    ap.add_argument('--monet-chunks', type=int, default=4)
    ap.add_argument('--thr', type=float, default=0.75)
    ap.add_argument('--sample-shards', type=int, default=16)
    ap.add_argument('--query-batch', type=int, default=4096)
    ap.add_argument('--monet-subtile', type=int, default=131072)
    ap.add_argument('--seed', type=int, default=20260615)
    ap.add_argument('--global-unique-dropped', type=int, default=8002184)
    ap.add_argument('--global-unique-queries', type=int, default=99037430)
    args = ap.parse_args()

    # fp16 accumulation OFF for the reference matmul too (we use fp32 inputs,
    # but be explicit so a future fp16 path here would still be honest).
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    device = 'cuda'
    t0 = time.time()

    chunk_paths = [os.path.join(
        args.cache_dir,
        f'monet_chunk{c}of{args.monet_chunks}_n{args.n_tag}_syn0.npy')
        for c in range(args.monet_chunks)]
    for p in chunk_paths:
        if not os.path.exists(p):
            raise SystemExit(f'missing cached MONET chunk: {p}')

    gpic_files = sorted(glob.glob(f'{args.gpic_pass}/gpic_pass_*.parquet'))
    print(f'[validate] gpic shards total={len(gpic_files)} sampling {args.sample_shards} '
          f'seed={args.seed}', flush=True)
    keys, xq = load_gpic_sample(gpic_files, args.sample_shards, args.seed)
    print(f'[validate] sampled unique queries = {len(keys):,}', flush=True)

    q32 = torch.from_numpy(xq.astype(np.float32)).to(device)
    gt_cos, gt_chunk = gt_max_cos(q32, chunk_paths, args.query_batch,
                                  args.monet_subtile, device)

    gt_drop = gt_cos >= args.thr
    n = len(keys)
    n_gt_drop = int(gt_drop.sum())

    # official drop list, restricted to the sampled keys
    off = pd.read_parquet(args.drop_parquet, columns=['drop_key', 'sscd_cos'])
    sample_set = set(keys.tolist())
    off_sub = off[off['drop_key'].isin(sample_set)]
    off_cos = dict(zip(off_sub['drop_key'].to_numpy(dtype=object),
                       off_sub['sscd_cos'].to_numpy()))
    in_off = np.array([k in off_cos for k in keys], dtype=bool)
    n_in_off = int(in_off.sum())

    tp = int((gt_drop & in_off).sum())
    fp = int((~gt_drop & in_off).sum())     # run dropped, truth says keep -> threshold/merge bug
    fn = int((gt_drop & ~in_off).sum())     # run kept, truth says drop -> coverage bug
    tn = int((~gt_drop & ~in_off).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    # cosine agreement on keys flagged by both (fp32 gt vs run's fp16-reduced cos)
    both = [k for k, d in zip(keys, gt_drop) if d and (k in off_cos)]
    diffs = np.array([abs(float(off_cos[k]) - float(gt_cos[i]))
                      for i, k in enumerate(keys) if gt_drop[i] and (k in off_cos)])
    cos_diff_max = float(diffs.max()) if len(diffs) else 0.0
    cos_diff_mean = float(diffs.mean()) if len(diffs) else 0.0

    # borderline FP/FN cosine context (how far off, if any)
    fp_keys = [(k, float(gt_cos[i])) for i, k in enumerate(keys) if (~gt_drop[i]) and in_off[i]]
    fn_keys = [(k, float(gt_cos[i])) for i, k in enumerate(keys) if gt_drop[i] and (~in_off[i])]

    report = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'thr': args.thr,
        'sample_shards': args.sample_shards,
        'sample_unique_queries': n,
        'gt_dropped': n_gt_drop,
        'gt_sample_drop_rate': round(n_gt_drop / max(n, 1), 6),
        'global_drop_rate': round(args.global_unique_dropped /
                                  max(args.global_unique_queries, 1), 6),
        'official_dropped_in_sample': n_in_off,
        'confusion': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
        'precision_vs_gt': round(precision, 6),
        'recall_vs_gt': round(recall, 6),
        'cos_agreement_max_abs_diff': round(cos_diff_max, 6),
        'cos_agreement_mean_abs_diff': round(cos_diff_mean, 6),
        'n_cos_compared': len(diffs),
        'fp_examples': fp_keys[:10],
        'fn_examples': fn_keys[:10],
        'elapsed_sec': round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(report, fh, indent=2)
    print('=== VALIDATION REPORT ===', flush=True)
    print(json.dumps(report, indent=2), flush=True)
    verdict = (fp == 0 and fn == 0)
    print(f'VERDICT: {"PASS (perfect agreement on sample)" if verdict else "REVIEW: FP/FN present"}',
          flush=True)


if __name__ == '__main__':
    main()
