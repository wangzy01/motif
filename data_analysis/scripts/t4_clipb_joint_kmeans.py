"""T4 joint CLIP-B clustering for MONET + GPIC.

This script intentionally separates mechanics from policy:
  * MONET candidates are kept real rows from monet_manifest by default.
  * GPIC candidates are T1-ok rows that also pass the basic size/aspect gate.
  * Training uses spherical k-means on L2-normalized CLIP-B vectors.
  * Assignment writes per-shard cluster parquet files, suitable for joining in T5.

It can run a small pilot end-to-end, or a full train/assign pass after the T4
policy choices are confirmed.
"""
import argparse
import glob
import json
import os
import shutil
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


S3 = '/s3-pixai-test-uw2/evan/mot_filtering_v1'
MONET_EMB = f'{S3}/monet_emb'
MONET_MANIFEST = f'{S3}/monet_manifest'
GPIC_PASS = f'{S3}/gpic_pass'
OUT_DIR = f'{S3}/t4_clipb_joint_kmeans'
LOCAL_TMP = os.environ.get(
    'MOT_LOCAL_TMP',
    '/fsx/evan/mot_code/data_analysis/outputs/_tmp_parquet',
)


def is_s3_mount(path):
    return os.path.abspath(path).startswith('/s3-')


def write_parquet_checked(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_dir = LOCAL_TMP if is_s3_mount(path) else os.path.dirname(path)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, f'{os.path.basename(path)}.tmp.{os.getpid()}')
    try:
        df.to_parquet(tmp, index=False)
        if is_s3_mount(path):
            if os.path.exists(path):
                os.remove(path)
            shutil.copyfile(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json_checked(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_dir = LOCAL_TMP if is_s3_mount(path) else os.path.dirname(path)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, f'{os.path.basename(path)}.tmp.{os.getpid()}')
    try:
        with open(tmp, 'w') as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
            fh.write('\n')
        if is_s3_mount(path):
            if os.path.exists(path):
                os.remove(path)
            shutil.copyfile(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_npy_checked(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_dir = LOCAL_TMP if is_s3_mount(path) else os.path.dirname(path)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, f'{os.path.basename(path)}.tmp.{os.getpid()}.npy')
    try:
        np.save(tmp, arr)
        if is_s3_mount(path):
            if os.path.exists(path):
                os.remove(path)
            shutil.copyfile(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def selected_files(files, max_files):
    if max_files <= 0 or max_files >= len(files):
        return files
    step = len(files) / max_files
    return [files[int(i * step)] for i in range(max_files)]


def shard_from_gpic(path):
    return os.path.basename(path).replace('gpic_pass_', '').replace('.parquet', '')


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


def bytes_indices_to_matrix(vals, indices, dim=512):
    arr = np.empty((len(indices), dim), dtype=np.float16)
    want = dim * 2
    nout = 0
    for i in indices:
        v = vals[int(i)]
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
        nout += 1
    return arr[:nout]


def monet_keep_mask(emb_file, manifest_root, include_synthetic):
    mf = f'{manifest_root}/{os.path.basename(emb_file)}'
    if not os.path.exists(mf):
        raise FileNotFoundError(f'missing MONET manifest for {emb_file}: {mf}')
    tab = pq.read_table(mf, columns=['keep', 'synthetic'])
    keep = tab['keep'].to_numpy(zero_copy_only=False).astype(bool)
    if include_synthetic:
        return keep
    syn = tab['synthetic'].to_numpy(zero_copy_only=False).astype(bool)
    return keep & ~syn


def gpic_basic_mask(width, height, ok):
    short = np.minimum(width, height)
    long = np.maximum(width, height)
    ar = long / np.clip(short, 1, None)
    return ok & (short >= 256) & (ar <= 3.0)


def sample_rows(x, target, rng):
    if len(x) <= target:
        return x
    idx = rng.choice(len(x), size=target, replace=False)
    return x[np.sort(idx)]


def progress_every(total, cap=200):
    return max(1, min(cap, max(1, total // 10)))


def load_monet_clipb(file, manifest_root, include_synthetic, need_meta=False):
    cols = ['id', 'source', 'least_dimension', 'aes_jasperai', 'clip_b']
    tab = pq.read_table(file, columns=cols)
    keep_full = monet_keep_mask(file, manifest_root, include_synthetic)
    if len(keep_full) < tab.num_rows:
        raise SystemExit(f'manifest shorter than emb for {file}: '
                         f'{len(keep_full)} < {tab.num_rows} (positional join broken)')
    keep = keep_full[:tab.num_rows]
    vals = np.asarray(tab['clip_b'].to_pylist(), dtype=object)
    x, good = bytes_to_matrix(vals)
    good_idx = np.flatnonzero(good)
    if len(x):
        keep_good = keep[good_idx]
        x = x[keep_good]
        good_idx = good_idx[keep_good]
    if not need_meta:
        return x
    meta = {
        'id': np.asarray(tab['id'].to_pylist(), dtype=object)[good_idx],
        'source': np.asarray(tab['source'].to_pylist(), dtype=object)[good_idx],
        'least_dimension': tab['least_dimension'].to_numpy(zero_copy_only=False)[good_idx],
        'aes': tab['aes_jasperai'].to_numpy(zero_copy_only=False)[good_idx],
    }
    return x, meta


def load_monet_clipb_sample(file, manifest_root, include_synthetic, target, rng):
    tab = pq.read_table(file, columns=['clip_b'])
    keep_full = monet_keep_mask(file, manifest_root, include_synthetic)
    if len(keep_full) < tab.num_rows:
        raise SystemExit(f'manifest shorter than emb for {file}: '
                         f'{len(keep_full)} < {tab.num_rows} (positional join broken)')
    keep = keep_full[:tab.num_rows]
    cand = np.flatnonzero(keep)
    if len(cand) == 0:
        return np.empty((0, 512), dtype=np.float16)
    take = min(len(cand), max(target * 3, target + 256))
    idx = np.sort(rng.choice(cand, size=take, replace=False))
    vals = tab['clip_b'].to_pylist()
    x = bytes_indices_to_matrix(vals, idx)
    return sample_rows(x, target, rng)


def load_gpic_clipb(file, need_meta=False):
    tab = pq.read_table(
        file,
        columns=['key', 'ok', 'width', 'height', 'aes_laion_improved', 'clip_b'],
    )
    ok = tab['ok'].to_numpy(zero_copy_only=False).astype(bool)
    width = tab['width'].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    height = tab['height'].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    keep = gpic_basic_mask(width, height, ok)
    vals = np.asarray(tab['clip_b'].to_pylist(), dtype=object)
    x, good = bytes_to_matrix(vals)
    good_idx = np.flatnonzero(good)
    if len(x):
        keep_good = keep[good_idx]
        x = x[keep_good]
        good_idx = good_idx[keep_good]
    if not need_meta:
        return x
    meta = {
        'key': np.asarray(tab['key'].to_pylist(), dtype=object)[good_idx],
        'width': width[good_idx],
        'height': height[good_idx],
        'aes': tab['aes_laion_improved'].to_numpy(zero_copy_only=False)[good_idx],
    }
    return x, meta


def load_gpic_clipb_sample(file, target, rng):
    tab = pq.read_table(
        file,
        columns=['ok', 'width', 'height', 'clip_b'],
    )
    ok = tab['ok'].to_numpy(zero_copy_only=False).astype(bool)
    width = tab['width'].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    height = tab['height'].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    keep = gpic_basic_mask(width, height, ok)
    cand = np.flatnonzero(keep)
    if len(cand) == 0:
        return np.empty((0, 512), dtype=np.float16)
    take = min(len(cand), max(target * 3, target + 256))
    idx = np.sort(rng.choice(cand, size=take, replace=False))
    vals = tab['clip_b'].to_pylist()
    x = bytes_indices_to_matrix(vals, idx)
    return sample_rows(x, target, rng)


def collect_train_sample(monet_files, gpic_files, args):
    rng = np.random.default_rng(args.seed)
    monet_target = int(round(args.train_sample * args.monet_train_fraction))
    gpic_target = args.train_sample - monet_target
    per_monet = max(1, int(np.ceil(monet_target / max(len(monet_files), 1))))
    per_gpic = max(1, int(np.ceil(gpic_target / max(len(gpic_files), 1))))
    xs = []
    counts = {'monet_sample_rows': 0, 'gpic_sample_rows': 0}
    t0 = time.time()
    monet_every = progress_every(len(monet_files))
    gpic_every = progress_every(len(gpic_files))
    for i, f in enumerate(monet_files):
        x = load_monet_clipb_sample(
            f, args.monet_manifest, args.include_synthetic, per_monet, rng)
        if len(x):
            xs.append(x)
            counts['monet_sample_rows'] += int(len(x))
        if (i + 1) % monet_every == 0 or (i + 1) == len(monet_files):
            el = time.time() - t0
            print(f'sample MONET {i+1}/{len(monet_files)} rows={counts["monet_sample_rows"]:,} '
                  f'eta={(len(monet_files)-i-1)*el/max(i+1,1)/60:.0f}min', flush=True)
    for i, f in enumerate(gpic_files):
        x = load_gpic_clipb_sample(f, per_gpic, rng)
        if len(x):
            xs.append(x)
            counts['gpic_sample_rows'] += int(len(x))
        if (i + 1) % gpic_every == 0 or (i + 1) == len(gpic_files):
            el = time.time() - t0
            print(f'sample GPIC {i+1}/{len(gpic_files)} rows={counts["gpic_sample_rows"]:,} '
                  f'eta={(len(gpic_files)-i-1)*el/max(i+1,1)/60:.0f}min', flush=True)
    if not xs:
        raise SystemExit('no CLIP-B training sample rows loaded')
    train = np.concatenate(xs, axis=0)
    if len(train) > args.train_sample:
        train = sample_rows(train, args.train_sample, rng)
    print(f'train sample ready shape={train.shape} monet={counts["monet_sample_rows"]:,} '
          f'gpic={counts["gpic_sample_rows"]:,}', flush=True)
    return train, counts


def spherical_kmeans(train, k, iters, batch_size, seed, device):
    if len(train) < k:
        raise SystemExit(f'train_sample rows {len(train):,} < k {k:,}')
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(len(train), size=k, replace=False)
    cent = torch.from_numpy(train[init_idx].astype(np.float32)).to(device)
    cent = torch.nn.functional.normalize(cent, dim=1)
    x_cpu = train.astype(np.float16, copy=False)
    last_shift = None
    t0 = time.time()
    for it in range(iters):
        sums = torch.zeros((k, cent.shape[1]), dtype=torch.float32, device=device)
        counts = torch.zeros((k,), dtype=torch.float32, device=device)
        for s in range(0, len(x_cpu), batch_size):
            e = min(len(x_cpu), s + batch_size)
            x = torch.from_numpy(x_cpu[s:e]).to(device, non_blocking=True).float()
            x = torch.nn.functional.normalize(x, dim=1)
            labels = torch.argmax(x @ cent.T, dim=1)
            sums.index_add_(0, labels, x)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
        nonempty = counts > 0
        new_cent = cent.clone()
        new_cent[nonempty] = sums[nonempty] / counts[nonempty, None]
        new_cent = torch.nn.functional.normalize(new_cent, dim=1)
        last_shift = torch.mean(1.0 - torch.sum(cent * new_cent, dim=1)).item()
        cent = new_cent
        empty = int((~nonempty).sum().item())
        print(f'kmeans iter {it+1}/{iters} empty={empty} mean_cos_shift={last_shift:.6f} '
              f'elapsed={(time.time()-t0)/60:.1f}min', flush=True)
    return cent.detach().cpu().numpy().astype(np.float32), last_shift


def assign_matrix(x_np, cent_t, batch_size, device):
    labels_all = np.empty((len(x_np),), dtype=np.int32)
    scores_all = np.empty((len(x_np),), dtype=np.float32)
    for s in range(0, len(x_np), batch_size):
        e = min(len(x_np), s + batch_size)
        x = torch.from_numpy(x_np[s:e]).to(device, non_blocking=True).float()
        x = torch.nn.functional.normalize(x, dim=1)
        scores = x @ cent_t.T
        vals, labels = torch.max(scores, dim=1)
        labels_all[s:e] = labels.detach().cpu().numpy().astype(np.int32)
        scores_all[s:e] = vals.detach().cpu().numpy().astype(np.float32)
    return labels_all, scores_all


def update_stats(stats, labels, aes, source):
    c = Counter(labels.tolist())
    for k, v in c.items():
        stats['count'][int(k)] += int(v)
        stats[f'{source}_count'][int(k)] += int(v)
    finite = np.isfinite(aes)
    if finite.any():
        sums = np.bincount(labels[finite], weights=aes[finite], minlength=stats['k'])
        nums = np.bincount(labels[finite], minlength=stats['k'])
        stats[f'{source}_aes_sum'][:len(sums)] += sums
        stats[f'{source}_aes_count'][:len(nums)] += nums


def assign_monet(files, cent_t, args, device, stats):
    out_dir = f'{args.out_dir}/monet_clusters'
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    rows = 0
    reused = 0
    every = progress_every(len(files))
    for i, f in enumerate(files):
        base = os.path.basename(f)
        op = f'{out_dir}/monet_cluster_{base}'
        if os.path.exists(op):
            reused += 1
            rows += pq.ParquetFile(op).metadata.num_rows
            continue
        x, meta = load_monet_clipb(f, args.monet_manifest, args.include_synthetic, True)
        if len(x):
            labels, scores = assign_matrix(x, cent_t, args.assign_batch, device)
            df = pd.DataFrame({
                'id': meta['id'],
                'source': meta['source'],
                'shard': base,
                'cluster_id': labels,
                'cluster_cos': scores,
            })
            update_stats(stats, labels, meta['aes'].astype(np.float64, copy=False), 'monet')
        else:
            df = pd.DataFrame({
                'id': pd.Series([], dtype=object),
                'source': pd.Series([], dtype=object),
                'shard': pd.Series([], dtype=object),
                'cluster_id': pd.Series([], dtype=np.int32),
                'cluster_cos': pd.Series([], dtype=np.float32),
            })
        write_parquet_checked(df, op)
        rows += len(df)
        if (i + 1) % every == 0 or (i + 1) == len(files):
            el = time.time() - t0
            print(f'assign MONET {i+1}/{len(files)} rows={rows:,} reused={reused} '
                  f'eta={(len(files)-i-1)*el/max(i+1,1)/60:.0f}min', flush=True)
    return rows, reused


def assign_gpic(files, cent_t, args, device, stats):
    out_dir = f'{args.out_dir}/gpic_clusters'
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    rows = 0
    reused = 0
    every = progress_every(len(files))
    for i, f in enumerate(files):
        shard = shard_from_gpic(f)
        op = f'{out_dir}/gpic_cluster_{shard}.parquet'
        if os.path.exists(op):
            reused += 1
            rows += pq.ParquetFile(op).metadata.num_rows
            continue
        x, meta = load_gpic_clipb(f, True)
        if len(x):
            labels, scores = assign_matrix(x, cent_t, args.assign_batch, device)
            df = pd.DataFrame({
                'key': meta['key'],
                'shard': shard,
                'cluster_id': labels,
                'cluster_cos': scores,
            })
            update_stats(stats, labels, meta['aes'].astype(np.float64, copy=False), 'gpic')
        else:
            df = pd.DataFrame({
                'key': pd.Series([], dtype=object),
                'shard': pd.Series([], dtype=object),
                'cluster_id': pd.Series([], dtype=np.int32),
                'cluster_cos': pd.Series([], dtype=np.float32),
            })
        write_parquet_checked(df, op)
        rows += len(df)
        if (i + 1) % every == 0 or (i + 1) == len(files):
            el = time.time() - t0
            print(f'assign GPIC {i+1}/{len(files)} rows={rows:,} reused={reused} '
                  f'eta={(len(files)-i-1)*el/max(i+1,1)/60:.0f}min', flush=True)
    return rows, reused


def empty_stats(k):
    stats = defaultdict(Counter)
    stats['k'] = k
    for p in ('monet', 'gpic'):
        stats[f'{p}_aes_sum'] = np.zeros(k, dtype=np.float64)
        stats[f'{p}_aes_count'] = np.zeros(k, dtype=np.int64)
    return stats


def stats_frame(stats):
    k = stats['k']
    rows = []
    for i in range(k):
        monet_n = int(stats['monet_count'].get(i, 0))
        gpic_n = int(stats['gpic_count'].get(i, 0))
        mn = int(stats['monet_aes_count'][i])
        gn = int(stats['gpic_aes_count'][i])
        monet_sum = float(stats['monet_aes_sum'][i])
        gpic_sum = float(stats['gpic_aes_sum'][i])
        rows.append({
            'cluster_id': i,
            'total_count': monet_n + gpic_n,
            'monet_count': monet_n,
            'gpic_count': gpic_n,
            'monet_aes_sum': monet_sum,
            'monet_aes_count': mn,
            'monet_aes_mean': float(monet_sum / mn) if mn else np.nan,
            'gpic_aes_sum': gpic_sum,
            'gpic_aes_count': gn,
            'gpic_aes_mean': float(gpic_sum / gn) if gn else np.nan,
        })
    return pd.DataFrame(rows)


def finalize_rank_outputs(args, t0):
    rank_dir = f'{args.out_dir}/rank_stats'
    stat_files = sorted(glob.glob(f'{rank_dir}/cluster_stats_rank*.parquet'))
    if not stat_files:
        raise SystemExit(f'no rank stats found in {rank_dir}')
    if args.world > 1 and len(stat_files) != args.world:
        raise SystemExit(f'finalize expected {args.world} rank stats files, found '
                         f'{len(stat_files)} in {rank_dir}; refusing to merge incomplete '
                         f'cluster stats')
    frames = [pd.read_parquet(f) for f in stat_files]
    stats = pd.concat(frames, ignore_index=True)
    sum_cols = [
        'total_count', 'monet_count', 'gpic_count',
        'monet_aes_sum', 'monet_aes_count',
        'gpic_aes_sum', 'gpic_aes_count',
    ]
    agg = stats.groupby('cluster_id', as_index=False)[sum_cols].sum()
    all_ids = pd.DataFrame({'cluster_id': np.arange(args.k, dtype=np.int32)})
    agg = all_ids.merge(agg, on='cluster_id', how='left')
    for col in sum_cols:
        agg[col] = agg[col].fillna(0)
    agg['monet_aes_mean'] = np.where(
        agg['monet_aes_count'] > 0,
        agg['monet_aes_sum'] / agg['monet_aes_count'],
        np.nan,
    )
    agg['gpic_aes_mean'] = np.where(
        agg['gpic_aes_count'] > 0,
        agg['gpic_aes_sum'] / agg['gpic_aes_count'],
        np.nan,
    )
    out_cols = [
        'cluster_id', 'total_count', 'monet_count', 'gpic_count',
        'monet_aes_sum', 'monet_aes_count', 'monet_aes_mean',
        'gpic_aes_sum', 'gpic_aes_count', 'gpic_aes_mean',
    ]
    write_parquet_checked(agg[out_cols], f'{args.out_dir}/cluster_stats.parquet')

    rank_summaries = []
    for f in sorted(glob.glob(f'{rank_dir}/summary_rank*.json')):
        with open(f) as fh:
            rank_summaries.append(json.load(fh))
    train_summary = None
    train_summary_path = f'{args.out_dir}/train_summary.json'
    if os.path.exists(train_summary_path):
        with open(train_summary_path) as fh:
            train_summary = json.load(fh)

    summary = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'out_dir': args.out_dir,
        'stage': 'finalize',
        'k': args.k,
        'world': args.world,
        'rank_stats_files': len(stat_files),
        'rank_summary_files': len(rank_summaries),
        'train_summary': train_summary,
        'monet_rows_assigned': int(agg['monet_count'].sum()),
        'gpic_rows_assigned': int(agg['gpic_count'].sum()),
        'total_rows_assigned': int(agg['total_count'].sum()),
        'nonempty_clusters': int((agg['total_count'] > 0).sum()),
        'elapsed_sec': round(time.time() - t0, 3),
    }
    if rank_summaries:
        summary['rank_summaries'] = rank_summaries
    write_json_checked(summary, f'{args.out_dir}/summary.json')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--monet-emb', default=MONET_EMB)
    ap.add_argument('--monet-manifest', default=MONET_MANIFEST)
    ap.add_argument('--gpic-pass', default=GPIC_PASS)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--k', type=int, default=20000)
    ap.add_argument('--train-sample', type=int, default=4000000)
    ap.add_argument('--monet-train-fraction', type=float, default=0.5)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--train-batch', type=int, default=8192)
    ap.add_argument('--assign-batch', type=int, default=8192)
    ap.add_argument('--max-monet-files', type=int, default=0)
    ap.add_argument('--max-gpic-files', type=int, default=0)
    ap.add_argument('--include-synthetic', action='store_true')
    ap.add_argument('--seed', type=int, default=20260615)
    ap.add_argument('--stage', choices=['train_assign', 'train', 'assign', 'finalize'],
                    default='train_assign')
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--world', type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.world < 1:
        raise SystemExit(f'world must be >= 1, got {args.world}')
    if args.rank < 0 or args.rank >= args.world:
        raise SystemExit(f'rank {args.rank} out of range for world {args.world}')
    if args.stage == 'finalize':
        finalize_rank_outputs(args, t0)
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise SystemExit('T4 k-means requires CUDA')
    torch.set_float32_matmul_precision('high')

    monet_files = selected_files(sorted(glob.glob(f'{args.monet_emb}/*.parquet')),
                                 args.max_monet_files)
    gpic_files = selected_files(sorted(glob.glob(f'{args.gpic_pass}/gpic_pass_*.parquet')),
                                args.max_gpic_files)
    if not monet_files:
        raise SystemExit(f'no MONET embedding parquet in {args.monet_emb}')
    if not gpic_files:
        raise SystemExit(f'no GPIC pass parquet in {args.gpic_pass}')
    all_monet_files = monet_files
    all_gpic_files = gpic_files
    if args.stage == 'assign' and args.world > 1:
        monet_files = all_monet_files[args.rank::args.world]
        gpic_files = all_gpic_files[args.rank::args.world]
    print(f'T4 inputs monet_files={len(monet_files):,} gpic_files={len(gpic_files):,} '
          f'out={args.out_dir} stage={args.stage} k={args.k:,} '
          f'train_sample={args.train_sample:,} rank={args.rank}/{args.world}', flush=True)

    cent_path = f'{args.out_dir}/centroids.npy'
    train_counts = {}
    last_shift = None
    if args.stage in ('train_assign', 'train'):
        train, train_counts = collect_train_sample(monet_files, gpic_files, args)
        cent, last_shift = spherical_kmeans(
            train, args.k, args.iters, args.train_batch, args.seed, device)
        save_npy_checked(cent, cent_path)
        del train
    else:
        if not os.path.exists(cent_path):
            raise SystemExit(f'missing centroids for assign stage: {cent_path}')
        cent = np.load(cent_path)
        if cent.shape != (args.k, 512):
            raise SystemExit(f'centroids shape {cent.shape} != expected ({args.k}, 512); '
                             f'corrupt or stale centroids at {cent_path}')

    monet_rows = gpic_rows = monet_reused = gpic_reused = 0
    stats = empty_stats(args.k)
    if args.stage in ('train_assign', 'assign'):
        cent_t = torch.from_numpy(cent.astype(np.float32, copy=False)).to(device)
        cent_t = torch.nn.functional.normalize(cent_t, dim=1)
        monet_rows, monet_reused = assign_monet(monet_files, cent_t, args, device, stats)
        gpic_rows, gpic_reused = assign_gpic(gpic_files, cent_t, args, device, stats)
        if args.stage == 'assign' and args.world > 1:
            write_parquet_checked(
                stats_frame(stats),
                f'{args.out_dir}/rank_stats/cluster_stats_rank{args.rank:03d}.parquet',
            )
        else:
            write_parquet_checked(stats_frame(stats), f'{args.out_dir}/cluster_stats.parquet')

    summary = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'out_dir': args.out_dir,
        'stage': args.stage,
        'rank': args.rank,
        'world': args.world,
        'k': args.k,
        'train_sample': args.train_sample,
        'monet_train_fraction': args.monet_train_fraction,
        'iters': args.iters,
        'train_batch': args.train_batch,
        'assign_batch': args.assign_batch,
        'seed': args.seed,
        'include_synthetic': bool(args.include_synthetic),
        'monet_files': len(monet_files),
        'gpic_files': len(gpic_files),
        'train_counts': train_counts,
        'last_mean_cos_shift': last_shift,
        'monet_rows_assigned': int(monet_rows),
        'gpic_rows_assigned': int(gpic_rows),
        'monet_reused_files': int(monet_reused),
        'gpic_reused_files': int(gpic_reused),
        'elapsed_sec': round(time.time() - t0, 3),
    }
    if args.stage == 'train':
        summary_path = f'{args.out_dir}/train_summary.json'
    elif args.stage == 'assign' and args.world > 1:
        summary_path = f'{args.out_dir}/rank_stats/summary_rank{args.rank:03d}.json'
    else:
        summary_path = f'{args.out_dir}/summary.json'
    write_json_checked(summary, summary_path)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
