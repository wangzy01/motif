"""T2/T4 前置: 抽取 MONET 全量自带 embedding 列(sscd 给 T2 去重, clip_b 给 T4 聚类)。

IO 模式 = 读 9441 个大 parquet 的列裁剪(顺序读大文件,不碰小文件) → 本机 FUSE 可跑,
实测 ~945 行/s/线程,多线程并行;104.9M 行预计本机半天-1天。不需要 GPU。

输出: 分块 parquet,每块含 id / source / least_dimension / aes_jasperai / sscd(fp16 bytes) / clip_b(fp16 bytes)。
embedding 官方已 L2 归一化(parquet 存 float32 list);少数行可能为空 → 记数跳过。

用法(本机后台,T3 sweep 跑完再起,避免抢 IO):
  python t2_monet_extract.py --workers 32 --which sscd,clip_b
"""
import argparse, glob, os, shutil, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pyarrow.parquet as pq
import pandas as pd

MROOT = '/s3-mewtant-raw-datasets/jasperai/monet/v1.1.0'
OUT = os.environ.get('MOT_MONET_EMB_OUT',
                     '/s3-pixai-test-uw2/evan/mot_filtering_v1/monet_emb')
LOCAL_TMP = os.environ.get('MOT_LOCAL_TMP',
                           '/fsx/evan/mot_code/data_analysis/outputs/_tmp_parquet')
WHICH = ['sscd', 'clip_b']


def list_parquets():
    return sorted(set(glob.glob(f'{MROOT}/*/*/*.parquet')) |
                  set(glob.glob(f'{MROOT}/synthetic/*/*/*.parquet')))


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


def emb_to_fp16_bytes(col, dim):
    out = np.zeros((len(col), dim), np.float16)
    empty = 0
    for i, v in enumerate(col):
        if v is None:
            empty += 1; continue
        a = np.asarray(v.as_py() if hasattr(v, 'as_py') else v, dtype=np.float32)
        if a.size != dim:
            empty += 1; continue
        out[i] = a.astype(np.float16)
    return out, empty


def one(args):
    f, which = args
    rel = os.path.relpath(f, MROOT)
    op = f'{OUT}/' + rel.replace('/', '__')
    expected_rows = pq.ParquetFile(f).metadata.num_rows
    if os.path.exists(op):
        try:
            if pq.ParquetFile(op).metadata.num_rows == expected_rows:
                return expected_rows, 0
        except Exception:
            pass
    cols = ['id', 'source', 'least_dimension', 'aesthetic_jasperai', 'sscd_nn']
    if 'sscd' in which: cols.append('embedding_sscd')
    if 'clip_b' in which: cols.append('embedding_clip-vit-base-patch32')
    t = pq.read_table(f, columns=cols).to_pandas()
    n = len(t)
    out = {'id': t['id'].values, 'source': t['source'].values,
           'least_dimension': t['least_dimension'].values,
           'aes_jasperai': t['aesthetic_jasperai'].values, 'sscd_nn': t['sscd_nn'].values}
    empty_tot = 0
    if 'sscd' in which:
        e, em = emb_to_fp16_bytes(t['embedding_sscd'].values, 512); empty_tot += em
        out['sscd'] = [r.tobytes() for r in e]
    if 'clip_b' in which:
        e, em = emb_to_fp16_bytes(t['embedding_clip-vit-base-patch32'].values, 512)
        out['clip_b'] = [r.tobytes() for r in e]
    write_parquet_checked(pd.DataFrame(out), op)
    return n, empty_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--which', default='sscd,clip_b')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    which = a.which.split(',')
    files = list_parquets()
    print(f'{len(files)} parquet, extracting {which}', flush=True)
    tot, empty, t0 = 0, 0, time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (n, em) in enumerate(ex.map(one, [(f, which) for f in files], chunksize=4)):
            tot += n; empty += em
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f'{i+1}/{len(files)} rows={tot:,} empty={empty} '
                      f'{tot/el:.0f}rows/s eta {(len(files)-i-1)*el/(i+1)/60:.0f}min', flush=True)
    print(f'DONE {tot:,} rows, {empty} empty embeddings skipped, {(time.time()-t0)/60:.0f}min', flush=True)


if __name__ == '__main__':
    main()
