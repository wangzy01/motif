"""T5 前置: GPIC train 全量元数据 manifest base(纯 CPU, 不依赖 T1)。

读 8000 个 train 小 parquet(顺序读, FUSE 快路径), 产出每 shard 一个 manifest:
  key / shard / width / height / short / caption_type / split_tier(nano|lite|full)
  / tok(词数×1.32 代理) / tier(T1-T4 档) / keep_basic(短边>=256 且 AR<=3) / bucket512
caption 文本不落盘(训练时回读原 parquet)。aes / 去重标记等 T1/T2 产出后 join。

用法: python t5_gpic_meta_prep.py --workers 16
"""
import argparse, glob, json, os, shutil, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

GROOT = '/s3-mewtant-raw-datasets/stanford-vision-lab/gpic_umziped'
OUT = os.environ.get('MOT_FILTER_OUT', '/s3-pixai-test-uw2/evan/mot_filtering_v1')
MAN = os.environ.get('MOT_GPIC_BASE_OUT', f'{OUT}/gpic_manifest_base')
LOCAL_TMP = os.environ.get('MOT_LOCAL_TMP',
                           '/fsx/evan/mot_code/data_analysis/outputs/_tmp_parquet')
TIER_BOUNDS = [(1, 24), (25, 64), (65, 128), (129, 10**9)]


def stats_from(df, shard):
    kept = df[df['keep_basic'] == 1]
    return {
        'rows': len(df), 'keep_basic': int(df['keep_basic'].sum()),
        'bucket512_kept': int(((df['keep_basic'] == 1) & (df['bucket512'] == 1)).sum()),
        'ctype_kept': kept['caption_type'].value_counts().to_dict(),
        'tier_kept': kept['tier'].value_counts().to_dict(),
        'short_lt256': int((df['short'] < 256).sum()),
    }


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


def one(shard):
    op = f'{MAN}/gpic_base_{shard}.parquet'
    src = f'{GROOT}/parquets/gpic_train_{shard}.parquet'
    expected_rows = pq.ParquetFile(src).metadata.num_rows
    if os.path.exists(op):
        try:
            if pq.ParquetFile(op).metadata.num_rows != expected_rows:
                raise ValueError('existing output row count mismatch')
            return shard, stats_from(pq.read_table(op).to_pandas(), shard)
        except Exception:
            pass
    t = pq.read_table(src, columns=['key', 'img_width', 'img_height', 'caption_type', 'caption', 'split']).to_pandas()
    short = t[['img_width', 'img_height']].min(axis=1).astype('int32')
    longs = t[['img_width', 'img_height']].max(axis=1).astype('int32')
    ar = longs / short.clip(lower=1)
    tok = (t['caption'].fillna('').str.count(r'\S+') * 1.32).round().astype('int32')
    tier = pd.Series(np.int8(0), index=t.index)
    for ti, (lo, hi) in enumerate(TIER_BOUNDS, 1):
        tier[(tok >= lo) & (tok <= hi)] = np.int8(ti)
    df = pd.DataFrame({
        'key': t['key'], 'shard': shard,
        'width': t['img_width'].astype('int32'), 'height': t['img_height'].astype('int32'),
        'short': short, 'caption_type': t['caption_type'],
        'split_tier': t['split'].map(lambda x: x[0] if x is not None and len(x) else ''),
        'tok': tok, 'tier': tier,
        'keep_basic': ((short >= 256) & (ar <= 3)).astype('int8'),
        'bucket512': (short >= 512).astype('int8'),
    })
    write_parquet_checked(df, op)
    return shard, stats_from(df, shard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(MAN, exist_ok=True)
    shards = [f'{i:05d}' for i in range(8000)]
    agg = {'rows': 0, 'keep_basic': 0, 'bucket512_kept': 0, 'short_lt256': 0,
           'ctype_kept': {}, 'tier_kept': {}}
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (sh, st) in enumerate(ex.map(one, shards, chunksize=4)):
            for k in ('rows', 'keep_basic', 'bucket512_kept', 'short_lt256'):
                agg[k] += st[k]
            for d in ('ctype_kept', 'tier_kept'):
                for k, v in st[d].items():
                    agg[d][str(k)] = agg[d].get(str(k), 0) + int(v)
            if (i + 1) % 400 == 0:
                el = time.time() - t0
                print(f'{i+1}/8000 rows={agg["rows"]:,} {el:.0f}s eta {(8000-i-1)*el/(i+1)/60:.0f}min', flush=True)
    with open(f'{OUT}/gpic_meta_prep_summary.json', 'w') as fp:
        json.dump(agg, fp, indent=2, ensure_ascii=False)
    print(json.dumps({k: agg[k] for k in ('rows', 'keep_basic', 'bucket512_kept')}, indent=2))
    print('saved ->', f'{OUT}/gpic_meta_prep_summary.json')


if __name__ == '__main__':
    main()
