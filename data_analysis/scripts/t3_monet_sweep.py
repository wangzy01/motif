"""T3: MONET 全量元数据 sweep + manifest 生成(纯 CPU,不依赖 T1/T2)。

规则照《数据集分析与处理方案.md》§4(2026-06-10 已拍板):
  1. 卫生: least_dimension >= 256;max(AR, 1/AR) <= 3;white_pixels < 95(0-100 刻度)
  2. 美学: aesthetic_jasperai >= 5.0 仅对 cc12m / commoncatalog-cc-by / megalith10m
  3. 水印: wk_jasperai 按源分位数切 top 2%(阈值 = 各源全量精确 p98,stage=wk 先算)
  4. nsfw 保险丝: max(三分数) >= 0.9 丢
  5. 分桶: least_dimension >= 512 双桶,[256,512) 仅 256 桶
  6. synthetic: 仅卫生项,synthetic=1
  另: holdout 50K(md5(id) % 2098 == 0,确定性)
caption: 每字段 token 数(词数×1.32 代理)+ T1-T4 档位(≤24 / 25-64 / 65-128 / ≥129)

用法:
  python t3_monet_sweep.py --stage wk      # pass1: 全量 wk 分位数 -> outputs/monet_wk_thresholds.json
  python t3_monet_sweep.py --stage sweep   # pass2: manifest -> outputs/monet_manifest/ + 统计报告
"""
import argparse, glob, hashlib, json, os, shutil
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MROOT = '/s3-mewtant-raw-datasets/jasperai/monet/v1.1.0'
OUT = '/fsx/evan/mot_code/data_analysis/outputs'
MAN = f'{OUT}/monet_manifest'
LOCAL_TMP = os.environ.get('MOT_LOCAL_TMP', f'{OUT}/_tmp_parquet')
AES_FLOOR_SOURCES = {'cc12m', 'commoncatalog-cc-by', 'megalith10m'}
TIER_BOUNDS = [(1, 24), (25, 64), (65, 128), (129, 10**9)]  # T1-T4(token)
CAP_COLS = ['caption_original', 'caption_florence-2-large', 'caption_internvl-3-8b',
            'caption_sharegpt4v-7b', 'caption_gemini-2.5-flash-lite']


def list_parquets():
    return sorted(set(glob.glob(f'{MROOT}/*/*/*.parquet')) |
                  set(glob.glob(f'{MROOT}/synthetic/*/*/*.parquet')))


def relpath(f):
    return os.path.relpath(f, MROOT)


def outpath(f):
    return f'{MAN}/' + relpath(f).replace('/', '__')


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


# ---------------- pass 1: wk 全量分位数 ----------------

def wk_one(f):
    t = pq.read_table(f, columns=['source', 'wk_jasperai']).to_pandas()
    return t


def stage_wk(workers):
    files = list_parquets()
    print(f'{len(files)} parquet files')
    acc = {}
    with ProcessPoolExecutor(workers) as ex:
        for i, t in enumerate(ex.map(wk_one, files, chunksize=8)):
            for src, g in t.groupby('source'):
                acc.setdefault(src, []).append(g['wk_jasperai'].values)
            if (i + 1) % 500 == 0:
                print(f'{i+1}/{len(files)}', flush=True)
    th = {}
    for src, chunks in acc.items():
        v = np.concatenate(chunks)
        v = v[~np.isnan(v)]
        th[src] = {'p98': float(np.percentile(v, 98)), 'n': int(v.size),
                   'p50': float(np.percentile(v, 50)), 'p95': float(np.percentile(v, 95))}
        print(src, th[src])
    with open(f'{OUT}/monet_wk_thresholds.json', 'w') as fp:
        json.dump(th, fp, indent=2)
    print('saved ->', f'{OUT}/monet_wk_thresholds.json')


# ---------------- pass 2: sweep ----------------

SCALAR_COLS = ['__key__', 'id', 'source', 'width', 'height', 'least_dimension', 'aspect_ratio',
               'white_pixels', 'aesthetic_jasperai', 'aesthetic_laion-improved', 'wk_jasperai',
               'nsfw_bumble', 'nsfw_falcon', 'nsfw_jasperai', 'sscd_nn', 'hash_perceptual']

WK_TH = None  # 子进程全局


def init_worker(th):
    global WK_TH
    WK_TH = th


def tok_proxy(s):
    # 词数×1.32;空/None -> 0
    w = s.fillna('').str.count(r'\S+')
    return (w * 1.32).round().astype('int32')


def sweep_one(f):
    rp = relpath(f)
    op = outpath(f)
    expected_rows = pq.ParquetFile(f).metadata.num_rows
    if os.path.exists(op):
        try:
            if pq.ParquetFile(op).metadata.num_rows != expected_rows:
                raise ValueError('existing output row count mismatch')
            prev = pq.read_table(op, columns=['source', 'keep', 'drop_reason', 'bucket512',
                                              'holdout', 'has_t1', 'has_t2', 'has_t3', 'has_t4',
                                              'synthetic']).to_pandas()
            n = len(prev)
            syn = bool(prev['synthetic'].iloc[0]) if n else rp.startswith('synthetic/')
            src_lbl = f'synthetic/{rp.split("/")[1]}' if syn else (prev['source'].iloc[0] if n else '')
            dr = prev['drop_reason'].fillna('')
            stats = {
                'rows': n, 'source': src_lbl,
                'keep': int(prev['keep'].sum()),
                'drop': dr[dr != ''].value_counts().to_dict(),
                'bucket512_kept': int(((prev['keep'] == 1) & (prev['bucket512'] == 1)).sum()),
                'tiers_kept': {f't{i}': int(prev.loc[prev['keep'] == 1, f'has_t{i}'].sum()) for i in range(1, 5)},
                'holdout': int(prev['holdout'].sum()),
            }
            return n, rp, stats
        except Exception:
            pass
    is_syn = rp.startswith('synthetic/')
    t = pq.read_table(f, columns=SCALAR_COLS + CAP_COLS).to_pandas()
    n = len(t)
    src = t['source'].iloc[0] if n else ''

    ar = t['aspect_ratio'].astype(float)
    ar_max = np.maximum(ar, 1.0 / ar.replace(0, np.nan))
    nsfw_max = t[['nsfw_bumble', 'nsfw_falcon', 'nsfw_jasperai']].max(axis=1)

    drop = pd.Series('', index=t.index, dtype=object)
    m = t['least_dimension'].fillna(0) < 256
    drop[m & (drop == '')] = 'short_edge'
    m = ar_max.fillna(1) > 3
    drop[m & (drop == '')] = 'aspect_ratio'
    m = t['white_pixels'].fillna(0) >= 95
    drop[m & (drop == '')] = 'white_pixels'
    m = nsfw_max.fillna(0) >= 0.9
    drop[m & (drop == '')] = 'nsfw_fuse'
    if not is_syn:
        if src in AES_FLOOR_SOURCES:
            m = t['aesthetic_jasperai'].fillna(99) < 5.0
            drop[m & (drop == '')] = 'aes_floor'
        wk_th = WK_TH.get(src, {}).get('p98', 1.0)
        m = t['wk_jasperai'].fillna(0) >= wk_th
        drop[m & (drop == '')] = 'wk_top2'

    out = pd.DataFrame({
        'key': t['__key__'], 'id': t['id'], 'source': t['source'], 'shard': rp,
        'width': t['width'], 'height': t['height'], 'least_dimension': t['least_dimension'],
        'aspect_ratio': ar, 'white_pixels': t['white_pixels'],
        'aes_jasperai': t['aesthetic_jasperai'], 'aes_laion': t['aesthetic_laion-improved'],
        'wk': t['wk_jasperai'], 'nsfw_max': nsfw_max, 'sscd_nn': t['sscd_nn'],
        'hash_perceptual': t['hash_perceptual'],
        'synthetic': np.int8(1) if is_syn else np.int8(0),
        'keep': (drop == '').astype('int8'), 'drop_reason': drop,
        'bucket512': (t['least_dimension'].fillna(0) >= 512).astype('int8'),
    })
    out['holdout'] = t['id'].astype(str).map(
        lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 2098 == 0).astype('int8')
    for c in CAP_COLS:
        short = c.replace('caption_', '').replace('-2-large', '').replace('-3-8b', '') \
                 .replace('4v-7b', '4v').replace('-2.5-flash-lite', '')
        out[f'tok_{short}'] = tok_proxy(t[c])
    tok_cols = [c for c in out.columns if c.startswith('tok_')]
    for ti, (lo, hi) in enumerate(TIER_BOUNDS, 1):
        out[f'has_t{ti}'] = (out[tok_cols].apply(lambda col: (col >= lo) & (col <= hi))
                             .any(axis=1)).astype('int8')
    write_parquet_checked(out, op)
    # 统计摘要(回传主进程聚合)
    stats = {
        'rows': n, 'source': src if not is_syn else f'synthetic/{rp.split("/")[1]}',
        'keep': int(out['keep'].sum()),
        'drop': drop[drop != ''].value_counts().to_dict(),
        'bucket512_kept': int(((out['keep'] == 1) & (out['bucket512'] == 1)).sum()),
        'tiers_kept': {f't{i}': int(out.loc[out['keep'] == 1, f'has_t{i}'].sum()) for i in range(1, 5)},
        'holdout': int(out['holdout'].sum()),
    }
    return n, rp, stats


def stage_sweep(workers):
    os.makedirs(MAN, exist_ok=True)
    with open(f'{OUT}/monet_wk_thresholds.json') as fp:
        th = json.load(fp)
    files = list_parquets()
    print(f'{len(files)} parquet files', flush=True)
    agg, total = {}, 0
    import time
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(th,)) as ex:
        for i, (n, rp, stats) in enumerate(ex.map(sweep_one, files, chunksize=4)):
            total += n
            if stats:
                s = stats['source']
                a = agg.setdefault(s, {'rows': 0, 'keep': 0, 'bucket512_kept': 0, 'holdout': 0,
                                       'drop': {}, 'tiers_kept': {'t1': 0, 't2': 0, 't3': 0, 't4': 0}})
                a['rows'] += stats['rows']; a['keep'] += stats['keep']
                a['bucket512_kept'] += stats['bucket512_kept']; a['holdout'] += stats['holdout']
                for k, v in stats['drop'].items():
                    a['drop'][k] = a['drop'].get(k, 0) + v
                for k in a['tiers_kept']:
                    a['tiers_kept'][k] += stats['tiers_kept'][k]
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f'{i+1}/{len(files)} rows={total} {el:.0f}s eta {(len(files)-i-1)*el/(i+1)/60:.0f}min', flush=True)
    summary = {'total_rows': total, 'per_source': agg,
               'totals': {'keep': sum(a['keep'] for a in agg.values()),
                          'bucket512_kept': sum(a['bucket512_kept'] for a in agg.values())}}
    with open(f'{OUT}/monet_sweep_summary.json', 'w') as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(json.dumps(summary['totals'], indent=2))
    print('saved ->', f'{OUT}/monet_sweep_summary.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['wk', 'sweep'], required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.stage == 'wk':
        stage_wk(a.workers)
    else:
        stage_sweep(a.workers)
