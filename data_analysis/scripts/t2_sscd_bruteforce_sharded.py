"""T2 cross-source SSCD dedup by exhaustive brute-force cosine on multiple GPUs.

Why brute force instead of cuVS IVF:
  * cuVS's multi-GPU sharded IVF-Flat returns wrong results on this cluster
    (confirmed: 8-shard self-join found 0 near-dup edges where a single shard
    found 191k; cross-source full found 1001 hits where a 256x256 pilot found
    6838). The single-GPU cuVS path also tripped an RMM memory limiter at scale.
  * The full cross-source compare is only ~8e18 FLOPs, i.e. tens of minutes on
    8 H100s. Brute-force matmul is EXHAUSTIVE (every GPIC vs every MONET, no ANN
    recall loss, no nlist/nprobe to tune). It is computed in fp16 with fp32
    accumulation, so it is accurate to the fp16-stored embedding precision
    (~1e-3 near the 0.75 boundary) -- not bit-exact, but free of the
    recall/correctness failure mode that bit cuVS. Reuses the same torch +
    per-rank-GPU pattern that T4 already runs correctly.

Sharding (the only correct multi-GPU here = we split the work ourselves):
  * MONET kept-real vectors (keep & !synthetic) are split into C chunks by
    ROUND-ROBIN file stride (not contiguous: synthetic files and per-source
    density cluster, so contiguous chunks are wildly uneven). Each chunk's raw
    fp16 vectors fit one GPU and are cached to FSX.
  * GPIC (T1 ok=1 + basic size/AR gate) shards are split into P = world // C
    round-robin partitions.
  * rank r handles (chunk_id = r % C, gpic_part = r // C). The C ranks sharing a
    gpic_part together cover all MONET for that partition, so every GPIC row is
    compared against every MONET vector. finalize merges per-GPIC-key max cosine.

Output: per-rank hits/hits_rank{r}.parquet (key, shard, sscd_cos, monet_chunk,
monet_id) for cosine >= thr, then finalize -> gpic_sscd_drop_ids.parquet.
"""
import argparse
import glob
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


S3 = '/s3-pixai-test-uw2/evan/mot_filtering_v1'
MONET_EMB = f'{S3}/monet_emb'
MONET_MANIFEST = f'{S3}/monet_manifest'
GPIC_PASS = f'{S3}/gpic_pass'
OUT = f'{S3}/t2_sscd_bruteforce'
LOCAL_TMP = os.environ.get(
    'MOT_LOCAL_TMP',
    '/fsx/evan/mot_code/data_analysis/outputs/_tmp_parquet',
)


def is_s3_mount(path):
    return os.path.abspath(path).startswith('/s3-')


def _atomic_put(write_fn, path, suffix=''):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_dir = LOCAL_TMP if is_s3_mount(path) else os.path.dirname(path)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, f'{os.path.basename(path)}.tmp.{os.getpid()}{suffix}')
    try:
        write_fn(tmp)
        if is_s3_mount(path):
            if os.path.exists(path):
                os.remove(path)
            shutil.copyfile(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_parquet_checked(df, path):
    _atomic_put(lambda tmp: df.to_parquet(tmp, index=False), path)


def write_json_checked(obj, path):
    def _w(tmp):
        with open(tmp, 'w') as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
            fh.write('\n')
    _atomic_put(_w, path)


def _save_npy(arr, path):
    # np.save adds .npy if missing; write to an exact tmp then move to `path`.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_dir = LOCAL_TMP if is_s3_mount(path) else os.path.dirname(path)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, f'{os.path.basename(path)}.tmp.{os.getpid()}')
    try:
        with open(tmp, 'wb') as fh:
            np.save(fh, arr)
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
    """Evenly-spaced subset (matches the other T2/T4 scripts) so a bounded smoke
    scans the same data as the cuVS medium for cross-validation."""
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


def chunk_files(files, chunk_id, n_chunks):
    # Round-robin (stride) assignment, NOT contiguous: MONET emb files are sorted
    # by source and all ~1440 synthetic files cluster at the end, so a contiguous
    # split makes chunks wildly uneven (laion-heavy ~27M vectors, the synthetic
    # chunk ~9M). Stride assignment mixes sources evenly so every chunk holds
    # ~equal kept-real mass (~19M) -> balanced GPU memory and wall-clock.
    # Still disjoint and exhaustive across chunk_id in [0, n_chunks).
    return files[chunk_id::n_chunks]


def load_monet_chunk(files, manifest_root, include_synthetic):
    """Load all kept-real (or +synthetic) MONET sscd vectors + ids for a chunk."""
    embs, ids = [], []
    rows = 0
    t0 = time.time()
    for i, f in enumerate(files):
        tab = pq.read_table(f, columns=['id', 'sscd'])
        keep_full = monet_keep_mask(f, manifest_root, include_synthetic)
        if len(keep_full) < tab.num_rows:
            raise SystemExit(f'manifest shorter than emb for {f}: '
                             f'{len(keep_full)} < {tab.num_rows} (positional join broken)')
        keep = keep_full[:tab.num_rows]
        vals = np.asarray(tab['sscd'].to_pylist(), dtype=object)
        x, good = bytes_to_matrix(vals)
        good_idx = np.flatnonzero(good)
        if len(x):
            keep_good = keep[good_idx]
            x = x[keep_good]
            kept_idx = good_idx[keep_good]
            embs.append(x)
            ids.append(np.asarray(tab['id'].to_pylist(), dtype=object)[kept_idx])
            rows += int(len(x))
        if (i + 1) % 200 == 0 or (i + 1) == len(files):
            el = time.time() - t0
            print(f'load MONET chunk {i+1}/{len(files)} vectors={rows:,} '
                  f'eta={(len(files)-i-1)*el/max(i+1,1)/60:.0f}min', flush=True)
    if not embs:
        raise SystemExit('no MONET vectors loaded for this chunk')
    xb = np.concatenate(embs, axis=0).astype(np.float16, copy=False)
    del embs
    idv = np.concatenate(ids)
    del ids
    return xb, idv


def get_or_build_monet_chunk(monet_files, args, chunk_id):
    files = chunk_files(monet_files, chunk_id, args.monet_chunks)
    # cache tag encodes the total selected file count so a different
    # --max-monet-files (different vector set) never reuses a stale cache.
    tag = (f'chunk{chunk_id}of{args.monet_chunks}_n{len(monet_files)}'
           f'_syn{int(args.include_synthetic)}')
    if args.cache_dir:
        xb_path = os.path.join(args.cache_dir, f'monet_{tag}.npy')
        id_path = os.path.join(args.cache_dir, f'monet_{tag}_ids.parquet')
        if os.path.exists(xb_path) and os.path.exists(id_path):
            print(f'[cache] loading MONET {tag} from {args.cache_dir}', flush=True)
            xb = np.load(xb_path)
            idv = pd.read_parquet(id_path)['id'].to_numpy(dtype=object)
            print(f'[cache] MONET {tag} xb={xb.shape}', flush=True)
            return xb, idv
        xb, idv = load_monet_chunk(files, args.monet_manifest, args.include_synthetic)
        print(f'[cache] writing MONET {tag} to {args.cache_dir}', flush=True)
        _save_npy(xb, xb_path)
        write_parquet_checked(pd.DataFrame({'id': idv}), id_path)
        return xb, idv
    return load_monet_chunk(files, args.monet_manifest, args.include_synthetic)


def empty_hits_df():
    return pd.DataFrame({
        'key': pd.Series([], dtype=object),
        'shard': pd.Series([], dtype=object),
        'sscd_cos': pd.Series([], dtype=np.float32),
        'monet_chunk': pd.Series([], dtype=np.int32),
        'monet_id': pd.Series([], dtype=object),
    })


def search_shard(gpic_t, monet_t, thr, query_batch, monet_subtile):
    """Return (hit_mask, best_cos, best_local_idx) for one shard's queries.

    gpic_t: (Q,512) fp16 cuda, L2-normalized. monet_t: (N,512) fp16 cuda, normalized.
    Exhaustive max-inner-product (== max cosine) via tiled fp16 matmul with fp32
    accumulation (no ANN approximation). The (qb, sub) sims tile is kept in fp16
    and reduced in fp16; only the per-row (qb,) max is upcast to fp32 -- this
    avoids a 2x-larger fp32 tile (the OOM risk) while keeping the threshold
    decision accurate to fp16 embedding precision.
    """
    q = gpic_t.shape[0]
    n = monet_t.shape[0]
    best = torch.full((q,), -2.0, device=gpic_t.device, dtype=torch.float32)
    best_idx = torch.full((q,), -1, device=gpic_t.device, dtype=torch.int64)
    for qs in range(0, q, query_batch):
        qe = min(q, qs + query_batch)
        gb = gpic_t[qs:qe]
        cur_best = best[qs:qe]
        cur_idx = best_idx[qs:qe]
        for ms in range(0, n, monet_subtile):
            me = min(n, ms + monet_subtile)
            sims = gb @ monet_t[ms:me].T                 # (qb, sub) fp16, fp32-accumulated
            tmax, targ = sims.max(dim=1)                 # reduce in fp16
            tmax = tmax.float()                          # upcast only the (qb,) result
            upd = tmax > cur_best
            cur_best = torch.where(upd, tmax, cur_best)
            cur_idx = torch.where(upd, targ.to(torch.int64) + ms, cur_idx)
        best[qs:qe] = cur_best
        best_idx[qs:qe] = cur_idx
    hit = best >= thr
    return hit, best, best_idx


def run_search(args, monet_files, gpic_files, device, t0):
    chunk_id = args.rank % args.monet_chunks
    gpic_part = args.rank // args.monet_chunks
    n_parts = args.world // args.monet_chunks
    my_gpic = gpic_files[gpic_part::n_parts]
    rank_out = f'{args.out}/hits/hits_rank{args.rank:03d}.parquet'
    rank_done = f'{args.out}/hits/summary_rank{args.rank:03d}.json'
    # The summary json is written LAST (after the hits parquet), so its presence
    # is the completion sentinel -- a rank killed mid-write leaves no summary and
    # is correctly recomputed, even though the hits parquet write to S3 is not
    # itself atomic (remove+copyfile window).
    if os.path.exists(rank_done):
        print(f'[rank {args.rank}] already complete (summary exists), skipping', flush=True)
        return

    xb, monet_ids = get_or_build_monet_chunk(monet_files, args, chunk_id)
    monet_t = torch.from_numpy(xb).to(device)
    resident = monet_t.element_size() * monet_t.nelement()
    sims_bytes = args.query_batch * args.monet_subtile * 2  # fp16 sims tile
    est_peak = resident + sims_bytes + 2 * 1024**3
    total = torch.cuda.get_device_properties(0).total_memory
    print(f'[rank {args.rank}] chunk={chunk_id}/{args.monet_chunks} part={gpic_part}/{n_parts} '
          f'monet={monet_t.shape} gpic_shards={len(my_gpic)} thr={args.thr} '
          f'resident={resident/1e9:.1f}GB sims_tile={sims_bytes/1e9:.1f}GB '
          f'est_peak={est_peak/1e9:.1f}GB total={total/1e9:.1f}GB', flush=True)
    if est_peak > 0.92 * total:
        raise SystemExit(f'[rank {args.rank}] estimated GPU peak {est_peak/1e9:.1f}GB > 92% of '
                         f'{total/1e9:.1f}GB; lower --monet-subtile or raise --monet-chunks')

    parts = []
    total_q = 0
    total_hits = 0
    every = max(1, min(200, len(my_gpic) // 10))
    ts = time.time()
    for i, f in enumerate(my_gpic):
        shard = shard_from_gpic(f)
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
        else:
            keys = np.asarray([], dtype=object)
        total_q += int(len(xq))
        if len(xq):
            gpic_t = torch.from_numpy(xq).to(device)
            hit, best, best_idx = search_shard(
                gpic_t, monet_t, args.thr, args.query_batch, args.monet_subtile)
            hit_np = hit.cpu().numpy()
            if hit_np.any():
                hi = np.flatnonzero(hit_np)
                local = best_idx.cpu().numpy()[hi]
                parts.append(pd.DataFrame({
                    'key': keys[hi],
                    'shard': shard,
                    'sscd_cos': best.cpu().numpy()[hi].astype(np.float32),
                    'monet_chunk': np.int32(chunk_id),
                    'monet_id': monet_ids[local],
                }))
                total_hits += int(len(hi))
            del gpic_t
        if (i + 1) % every == 0 or (i + 1) == len(my_gpic):
            el = time.time() - ts
            print(f'[rank {args.rank}] search {i+1}/{len(my_gpic)} q={total_q:,} '
                  f'hits={total_hits:,} eta={(len(my_gpic)-i-1)*el/max(i+1,1)/60:.0f}min',
                  flush=True)

    df = pd.concat(parts, ignore_index=True) if parts else empty_hits_df()
    write_parquet_checked(df, rank_out)
    summary = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'stage': 'search',
        'rank': args.rank,
        'world': args.world,
        'monet_chunks': args.monet_chunks,
        'chunk_id': chunk_id,
        'gpic_part': gpic_part,
        'monet_vectors': int(monet_t.shape[0]),
        'gpic_shards': len(my_gpic),
        'gpic_queries_scanned': total_q,
        'hit_rows': total_hits,
        'thr': args.thr,
        'elapsed_sec': round(time.time() - t0, 3),
    }
    write_json_checked(summary, f'{args.out}/hits/summary_rank{args.rank:03d}.json')
    print(json.dumps(summary, indent=2), flush=True)


def run_finalize(args, t0):
    hit_files = sorted(glob.glob(f'{args.out}/hits/hits_rank*.parquet'))
    if not hit_files:
        raise SystemExit(f'no rank hit files in {args.out}/hits')
    # Coverage guard: every GPIC row is only declared a dup via the chunks that
    # actually scanned it, so a missing rank => silently under-counted drops.
    # Require all `world` rank files (one per rank) before merging.
    if len(hit_files) != args.world:
        raise SystemExit(
            f'finalize expected {args.world} rank hit files, found {len(hit_files)}: '
            f'{[os.path.basename(f) for f in hit_files]}. Refusing to merge an '
            f'incomplete set (would under-count drops).')
    # merge by GPIC key keeping the max-cosine MONET partner (vectorized)
    frames = [pd.read_parquet(f) for f in hit_files]
    allh = pd.concat(frames, ignore_index=True)
    total_hit_rows = len(allh)
    if total_hit_rows:
        keep_idx = allh.groupby('key')['sscd_cos'].idxmax()
        m = allh.loc[keep_idx].reset_index(drop=True)
        drop = pd.DataFrame({
            'drop_key': m['key'].to_numpy(dtype=object),
            'sscd_cos': m['sscd_cos'].to_numpy(dtype=np.float32),
            'monet_chunk': m['monet_chunk'].to_numpy(dtype=np.int32),
            'monet_id': m['monet_id'].to_numpy(dtype=object),
            'drop_sscd': np.ones(len(m), dtype=np.int8),
        })
    else:
        drop = pd.DataFrame({
            'drop_key': pd.Series([], dtype=object),
            'sscd_cos': pd.Series([], dtype=np.float32),
            'monet_chunk': pd.Series([], dtype=np.int32),
            'monet_id': pd.Series([], dtype=object),
            'drop_sscd': pd.Series([], dtype=np.int8),
        })
    write_parquet_checked(drop, f'{args.out}/gpic_sscd_drop_ids.parquet')

    rank_summaries = []
    for f in sorted(glob.glob(f'{args.out}/hits/summary_rank*.json')):
        with open(f) as fh:
            rank_summaries.append(json.load(fh))
    gpic_scanned = sum(s.get('gpic_queries_scanned', 0) for s in rank_summaries)
    n_parts = args.world // args.monet_chunks
    # each gpic row is scanned C (=monet_chunks) times across ranks
    summary = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'stage': 'finalize',
        'out': args.out,
        'world': args.world,
        'monet_chunks': args.monet_chunks,
        'gpic_partitions': n_parts,
        'rank_hit_files': len(hit_files),
        'total_rank_hit_rows': int(total_hit_rows),
        'unique_gpic_dropped': int(len(drop)),
        'gpic_query_passes_total': int(gpic_scanned),
        'gpic_unique_queries_est': int(gpic_scanned // max(args.monet_chunks, 1)),
        'thr': args.thr,
        'rank_summaries': rank_summaries,
        'elapsed_sec': round(time.time() - t0, 3),
    }
    write_json_checked(summary, f'{args.out}/summary.json')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--monet-emb', default=MONET_EMB)
    ap.add_argument('--monet-manifest', default=MONET_MANIFEST)
    ap.add_argument('--gpic-pass', default=GPIC_PASS)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--thr', type=float, default=0.75)
    ap.add_argument('--monet-chunks', type=int, default=4)
    ap.add_argument('--query-batch', type=int, default=8192)
    ap.add_argument('--monet-subtile', type=int, default=262144)
    ap.add_argument('--max-monet-files', type=int, default=0)
    ap.add_argument('--max-gpic-files', type=int, default=0)
    ap.add_argument('--include-synthetic', action='store_true')
    ap.add_argument('--cache-dir', default='')
    ap.add_argument('--stage', choices=['search', 'finalize'], default='search')
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--world', type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    if args.world < 1 or args.monet_chunks < 1:
        raise SystemExit('world and monet_chunks must be >= 1')
    if args.world % args.monet_chunks != 0:
        raise SystemExit(f'world ({args.world}) must be divisible by monet_chunks '
                         f'({args.monet_chunks})')
    if args.rank < 0 or args.rank >= args.world:
        raise SystemExit(f'rank {args.rank} out of range for world {args.world}')
    if args.cache_dir and is_s3_mount(args.cache_dir):
        raise SystemExit(f'--cache-dir must be on FSX/local for atomic writes, not '
                         f'the S3 mount: {args.cache_dir}')

    if args.stage == 'finalize':
        run_finalize(args, t0)
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise SystemExit('brute-force SSCD requires CUDA')
    torch.set_float32_matmul_precision('high')
    # fp16 GEMM must accumulate in fp32 (default allows fp16 split-K reduction,
    # which adds ~1e-2 error over 512 terms and can flip the 0.75 threshold).
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    monet_files = sorted(glob.glob(f'{args.monet_emb}/*.parquet'))
    gpic_files = sorted(glob.glob(f'{args.gpic_pass}/gpic_pass_*.parquet'))
    if not monet_files:
        raise SystemExit(f'no MONET embedding parquet in {args.monet_emb}')
    if not gpic_files:
        raise SystemExit(f'no GPIC pass parquet in {args.gpic_pass}')
    monet_files = selected_files(monet_files, args.max_monet_files)
    gpic_files = selected_files(gpic_files, args.max_gpic_files)

    run_search(args, monet_files, gpic_files, device, t0)


if __name__ == '__main__':
    main()
