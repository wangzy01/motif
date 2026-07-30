"""T5 MONET real selection to a target size (default 40M) with a sub-source cap.

Policy:
  * real survivors = monet_manifest keep==1 & synthetic==0 & holdout==0, MINUS internal
    near-dup drop ids (t2_monet_internal_cuvs_single_full/monet_neardup_drop_ids.parquet).
    (holdout = T3's deterministic 50K md5(id)%2098==0 set; Evan: exclude from training.)
  * sub-source cap (per 自己写的数据集文档_v3.md "对大子源设上限，防单一爬虫源垄断"):
    water-fill a per-source cap so laion/coyo can't monopolize; small sources kept whole;
    WITHIN each source keep the highest-aesthetic rows (aes_jasperai, the recommended head).
  * synthetic rows are NOT selected here (they are the 1024 anchor pool, flagged elsewhere);
    we only tally them per source for reporting.

NOTE: the cap LEVEL is a soft decision Evan still owes. This script runs a principled
DEFAULT (pure water-fill to hit MONET_TARGET) and reports per-source survivor-vs-selected
so the cap can be re-pointed in one re-run. Exact, deterministic, single manifest scan.
CPU-only; S3 writes via local FSX temp + copyfile.
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
MANIFEST = f'{S3}/monet_manifest'
DROP = f'{S3}/t2_monet_internal_cuvs_single_full/monet_neardup_drop_ids.parquet'
OUTDIR = os.environ.get('MONET_SELECT_OUT', f'{S3}/t5_selection')
FSX = '/fsx/evan/mot_code/data_analysis/outputs'
TARGET = int(os.environ.get('MONET_TARGET', '40000000'))
N_OUT_SHARDS = int(os.environ.get('N_OUT_SHARDS', '256'))
HAS = ['has_t1', 'has_t2', 'has_t3', 'has_t4']
t0 = time.time()


def put(localfn, dst):
    # re-ensure the parent prefix every write: mountpoint-s3 does not persist empty
    # dirs, so a subdir made at startup can vanish before the first file lands.
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(localfn, dst)


def write_json_s3(obj, dst):
    tmp = f'{FSX}/_t5msel.{os.path.basename(dst)}.tmp.json'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2)
    put(tmp, dst)
    os.remove(tmp)


def waterfill_cap(counts, budget):
    """Returns (per-source quota array, cap level C)."""
    counts = counts.astype(np.int64)
    total = int(counts.sum())
    if budget >= total:
        return counts.copy(), int(counts.max())
    if budget <= 0:
        return np.zeros_like(counts), 0
    lo, hi = 0, int(counts.max())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if int(np.minimum(counts, mid).sum()) <= budget:
            lo = mid
        else:
            hi = mid - 1
    C = lo
    q = np.minimum(counts, C)
    rem = budget - int(q.sum())
    if rem > 0:
        above = np.flatnonzero(counts > C)
        q[above[:rem]] += 1
    return q, C


def main():
    os.makedirs(f'{OUTDIR}/monet_selected', exist_ok=True)
    # internal near-dup drop ids as a pyarrow array; membership done ONCE globally
    # (per-shard pandas .isin(big set) is pathologically slow).
    drop_arr = pq.read_table(DROP, columns=['drop_id']).column('drop_id').combine_chunks()
    print(f'[drop] {len(drop_arr):,} internal near-dup drop ids ({time.time()-t0:.0f}s)', flush=True)

    files = sorted(glob.glob(f'{MANIFEST}/*.parquet'))
    max_files = int(os.environ.get('MAX_FILES', '0'))
    if max_files > 0:
        files = files[:max_files]
        print(f'[smoke] MAX_FILES={max_files} -> {len(files)} files', flush=True)

    syn_per_src = {}

    def load_shard(f):
        try:
            df = pd.read_parquet(f, columns=['id', 'source', 'synthetic', 'keep', 'holdout',
                                             'bucket512', 'aes_jasperai'] + HAS)
            syn = df[df['synthetic'] == 1]
            sc = syn['source'].value_counts().to_dict()
            # real = kept, non-synthetic, AND NOT in the holdout eval set (Evan: exclude holdout)
            real = df[(df['keep'] == 1) & (df['synthetic'] == 0) & (df['holdout'] == 0)]
            if not len(real):
                return ('OK', None, sc)
            out = {
                'id': pa.array(real['id'].to_numpy(), pa.large_string()),
                'source': real['source'].to_numpy(),
                'aes': real['aes_jasperai'].to_numpy().astype(np.float32),
                'b512': real['bucket512'].to_numpy().astype(np.int8),
                'has': real[HAS].to_numpy().astype(np.int8),
            }
            return ('OK', out, sc)
        except Exception as e:  # noqa
            return ('ERR', f'{os.path.basename(f)}: {e}', {})

    IDS, SRC, AES, B512, HASM = [], [], [], [], []
    errors = []
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for tag, out, sc in ex.map(load_shard, files):
            done += 1
            for s, v in sc.items():
                syn_per_src[s] = syn_per_src.get(s, 0) + int(v)
            if tag == 'ERR':
                errors.append(out)
                continue
            if out is not None:
                IDS.append(out['id']); SRC.append(out['source']); AES.append(out['aes'])
                B512.append(out['b512']); HASM.append(out['has'])
            if done % 2000 == 0:
                print(f'[scan] {done}/{len(files)} errs={len(errors)} ({time.time()-t0:.0f}s)', flush=True)
    if errors:
        print(f'[scan] WARNING {len(errors)} errors; first: {errors[:3]}', flush=True)

    ids = pa.chunked_array(IDS, type=pa.large_string())
    src = np.concatenate(SRC)
    aes = np.concatenate(AES)
    b512 = np.concatenate(B512)
    hasm = np.concatenate(HASM)        # (N,4)
    del IDS, SRC, AES, B512, HASM
    gated = len(src)
    print(f'[load] real(keep&!synthetic)={gated:,} ({time.time()-t0:.0f}s)', flush=True)

    # ---- exclude internal near-dup ids (ONE global pyarrow membership) ----
    keep_pa = pc.invert(pc.is_in(ids, value_set=drop_arr))
    keep = keep_pa.combine_chunks().to_numpy(zero_copy_only=False).astype(bool)
    ids = ids.filter(keep_pa)
    src = src[keep]; aes = aes[keep]; b512 = b512[keep]; hasm = hasm[keep]
    n = len(src)
    print(f'[dedup] {gated:,} -> real survivors={n:,} (dropped {gated-n:,}) ({time.time()-t0:.0f}s)', flush=True)

    sources = sorted(set(src.tolist()))
    code = {s: i for i, s in enumerate(sources)}
    scode = np.array([code[s] for s in src], dtype=np.int16)
    counts = np.bincount(scode, minlength=len(sources)).astype(np.int64)

    q, capC = waterfill_cap(counts, TARGET)
    print(f'[cap] target={TARGET:,} per-source cap_level={capC:,} '
          f'sources_capped={int((q < counts).sum())}/{len(sources)} ({time.time()-t0:.0f}s)', flush=True)

    # within each source keep top-q by aes
    sel = np.zeros(n, dtype=bool)
    for si in range(len(sources)):
        rows = np.flatnonzero(scode == si)
        if q[si] >= len(rows):
            sel[rows] = True
        elif q[si] > 0:
            top = rows[np.argsort(-aes[rows], kind='stable')[:q[si]]]
            sel[top] = True
    nsel = int(sel.sum())
    print(f'[select] selected={nsel:,} (target {TARGET:,}) ({time.time()-t0:.0f}s)', flush=True)

    # ---- write ----
    sel_idx = np.flatnonzero(sel)
    sel_ids = ids.take(pa.array(sel_idx)).to_numpy(zero_copy_only=False)
    s_src = src[sel_idx]; s_aes = aes[sel_idx]; s_b512 = b512[sel_idx]; s_has = hasm[sel_idx]
    bounds = np.linspace(0, nsel, N_OUT_SHARDS + 1).astype(np.int64)
    for s in range(N_OUT_SHARDS):
        a, b = bounds[s], bounds[s + 1]
        if a >= b:
            continue
        df = pd.DataFrame({
            'id': sel_ids[a:b], 'source': s_src[a:b],
            'bucket512': s_b512[a:b], 'aes_jasperai': s_aes[a:b],
            'has_t1': s_has[a:b, 0], 'has_t2': s_has[a:b, 1],
            'has_t3': s_has[a:b, 2], 'has_t4': s_has[a:b, 3],
        })
        tmp = f'{FSX}/_t5msel.shard{s:04d}.tmp.parquet'
        df.to_parquet(tmp, index=False)
        put(tmp, f'{OUTDIR}/monet_selected/monet_selected_{s:04d}.parquet')
        os.remove(tmp)
    print(f'[write] {N_OUT_SHARDS} shards ({time.time()-t0:.0f}s)', flush=True)

    per_source = []
    for si, s in enumerate(sources):
        rows = scode == si
        srows = rows & sel
        per_source.append({
            'source': s,
            'real_survivors': int(counts[si]),
            'selected': int(srows.sum()),
            'selected_share': round(float(srows.sum()) / max(nsel, 1), 4),
            'selected_512': int(((b512 == 1) & srows).sum()),
            'synthetic': int(syn_per_src.get(s, 0)),
        })
    per_source.sort(key=lambda r: -r['selected'])
    summary = {
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'policy': 'waterfill sub-source cap to MONET_TARGET, aes_jasperai-priority within source (default cap ACCEPTED by Evan 2026-06-16); real = keep & !synthetic & !holdout',
        'monet_target': TARGET,
        'per_source_cap_level': int(capC),
        'real_survivors_total': int(n),
        'selected_total': nsel,
        'selected_512_total': int(((b512 == 1) & sel).sum()),
        'caption_band_coverage_selected': {h: round(float(((hasm[:, i] == 1) & sel).sum()) / max(nsel, 1), 4)
                                           for i, h in enumerate(HAS)},
        'synthetic_total': int(sum(syn_per_src.values())),
        'per_source': per_source,
        'shard_errors': len(errors),
        'elapsed_sec': round(time.time() - t0, 1),
    }
    write_json_s3(summary, f'{OUTDIR}/monet_selection_summary.json')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'TOTAL {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
