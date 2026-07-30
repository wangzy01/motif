"""T1 v2: GPIC 全量 GPU pass — 单遍解码,同时产出 SSCD / CLIP-L/14 / CLIP-B/32 / aesthetic / pHash。

相比 v1(t1_gpic_pass.py,CPU 机上未实测稿)的修正:
  1. SSCD 预处理 = MONET 官方 faiss/sscd.py: T.Resize([320,320]) bilinear + ImageNet norm(v1 错用 288 bicubic)
  2. CLIP 用 *-quickgelu 变体加载 openai 权重(v1 直接 ViT-L-14 会触发 QuickGELU mismatch,embedding 偏差)
  3. CLIP 预处理 = open_clip 官方(短边 224 bicubic + CenterCrop),v1 错用方形拉伸
  4. 扁平单 DataLoader 跨全部 shard + persistent workers(v1 每 shard 重建 DataLoader,8000 次启停开销)
  5. 按 shard 边界增量落盘,断点续传按 shard 粒度

用法(每 GPU 一进程):
  venv-gpu/bin/python t1_gpic_pass_v2.py --rank 0 --world 2 --device cuda:0 --workers 40
  venv-gpu/bin/python t1_gpic_pass_v2.py --pilot 2 --device cuda:0   # 试点 2 shard
"""
import argparse, io, os, shutil, time, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import imagehash

GROOT = '/s3-mewtant-raw-datasets/stanford-vision-lab/gpic_umziped'
OUT = os.environ.get('MOT_GPIC_PASS_OUT',
                     '/s3-pixai-test-uw2/evan/mot_filtering_v1/gpic_pass')
LOCAL_TMP = os.environ.get('MOT_LOCAL_TMP',
                           '/fsx/evan/mot_code/data_analysis/outputs/_tmp_parquet')
MODELS = '/fsx/models'
SHORT_MIN = 256

IMAGENET_MEAN = (0.485, 0.456, 0.406); IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073); CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

Image.MAX_IMAGE_PIXELS = None  # GPIC 已过滤极端分辨率,防 DecompressionBomb 误杀


def flatten_alpha(im):
    if im.mode in ('RGBA', 'LA', 'PA'):
        bg = Image.new('RGB', im.size, (255, 255, 255))
        bg.paste(im, mask=im.convert('RGBA').split()[-1])
        return bg
    return im.convert('RGB')


def sscd_input(im):
    # MONET 官方: T.Resize([320,320]) 默认 bilinear+antialias, ImageNet norm
    im = im.resize((320, 320), Image.BILINEAR)
    a = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255)
    return (a - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]


def clip_input(im):
    # open_clip 官方: Resize 短边 224 bicubic + CenterCrop(224) + CLIP norm
    w, h = im.size
    s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    im = im.crop((l, t, l + 224, t + 224))
    a = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255)
    return (a - torch.tensor(CLIP_MEAN)[:, None, None]) / torch.tensor(CLIP_STD)[:, None, None]


class FlatGpicDataset(Dataset):
    """跨多个 shard 的扁平 dataset;items 按 (shard, key) 排序,主进程预先批量 LIST。"""
    def __init__(self, shards, threads=48):
        self.items = []  # (shard, key, ext)
        def load(shard):
            keys = pq.read_table(f'{GROOT}/parquets/gpic_train_{shard}.parquet',
                                 columns=['key']).column('key').to_pylist()
            ext = {}
            for f in os.listdir(f'{GROOT}/images/train/{shard}'):
                stem, _, e = f.rpartition('.')
                ext[stem] = e
            return [(shard, k, ext.get(k)) for k in keys]
        with ThreadPoolExecutor(threads) as ex:
            for chunk in ex.map(load, shards):
                self.items.extend(chunk)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        shard, k, e = self.items[i]
        rec = {'shard': shard, 'key': k, 'ok': False, 'w': 0, 'h': 0, 'phash': '',
               'sscd_in': None, 'clip_in': None}
        if e is None:
            return rec
        try:
            im = Image.open(f'{GROOT}/images/train/{shard}/{k}.{e}')
            im = flatten_alpha(im)
            rec['w'], rec['h'] = im.size
            if min(im.size) < SHORT_MIN:
                return rec
            rec['phash'] = str(imagehash.phash(im, hash_size=8))
            rec['sscd_in'] = sscd_input(im)
            rec['clip_in'] = clip_input(im)
            rec['ok'] = True
        except Exception:
            rec['ok'] = False
        return rec


def collate(batch):
    oks = [b for b in batch if b['ok']]
    return {
        'shard': [b['shard'] for b in batch], 'key': [b['key'] for b in batch],
        'ok': torch.tensor([b['ok'] for b in batch]),
        'w': [b['w'] for b in batch], 'h': [b['h'] for b in batch],
        'phash': [b['phash'] for b in batch],
        'sscd_in': torch.stack([b['sscd_in'] for b in oks]) if oks else torch.zeros(0, 3, 320, 320),
        'clip_in': torch.stack([b['clip_in'] for b in oks]) if oks else torch.zeros(0, 3, 224, 224),
    }


class AestheticMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128),
                                    nn.Dropout(0.2), nn.Linear(128, 64), nn.Dropout(0.1),
                                    nn.Linear(64, 16), nn.Linear(16, 1))
    def forward(self, x):
        return self.layers(x)


def load_models(device):
    import open_clip
    sscd = torch.jit.load(f'{MODELS}/sscd_disc_mixup.torchscript.pt').eval().to(device)
    clip_l, _, _ = open_clip.create_model_and_transforms(
        'ViT-L-14-quickgelu', pretrained='openai', cache_dir='/fsx/hf_cache/open_clip')
    clip_b, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32-quickgelu', pretrained='openai', cache_dir='/fsx/hf_cache/open_clip')
    clip_l = clip_l.visual.eval().to(device).half()
    clip_b = clip_b.visual.eval().to(device).half()
    aes = AestheticMLP()
    aes.load_state_dict(torch.load(f'{MODELS}/sac+logos+ava1-l14-linearMSE.pth', map_location='cpu'))
    aes = aes.eval().to(device)
    return sscd, clip_l, clip_b, aes


def shard_done(shard, expected):
    p = f'{OUT}/gpic_pass_{shard}.parquet'
    if not os.path.exists(p):
        return False
    try:
        return pq.ParquetFile(p).metadata.num_rows == expected
    except Exception:
        return False


def is_s3_mount(path):
    return os.path.abspath(path).startswith('/s3-')


def flush_shard(shard, rows):
    path = f'{OUT}/gpic_pass_{shard}.parquet'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_dir = LOCAL_TMP if is_s3_mount(path) else os.path.dirname(path)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, f'{os.path.basename(path)}.tmp.{os.getpid()}')
    try:
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        if is_s3_mount(path):
            if os.path.exists(path):
                os.remove(path)
            shutil.copyfile(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)



@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--world', type=int, default=1)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--pilot', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(4)
    dev = args.device

    all_shards = [f'{i:05d}' for i in range(8000) if i % args.world == args.rank]
    if args.pilot:
        all_shards = all_shards[:args.pilot]

    # 行数预期(断点续传判定用):读各 shard parquet footer
    print(f'[rank{args.rank}] listing {len(all_shards)} shards ...', flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(48) as ex:
        expected = dict(zip(all_shards, ex.map(
            lambda s: pq.ParquetFile(f'{GROOT}/parquets/gpic_train_{s}.parquet').metadata.num_rows,
            all_shards)))
    todo = [s for s in all_shards if not shard_done(s, expected[s])]
    print(f'[rank{args.rank}] resume: {len(all_shards)-len(todo)} done, {len(todo)} todo '
          f'({time.time()-t0:.0f}s)', flush=True)
    if not todo:
        print('all done'); return

    ds = FlatGpicDataset(todo)
    print(f'[rank{args.rank}] dataset built: {len(ds)} items ({time.time()-t0:.0f}s)', flush=True)
    models = load_models(dev)
    sscd, clip_l, clip_b, aes = models

    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, collate_fn=collate,
                    prefetch_factor=6, pin_memory=True, persistent_workers=True)

    cur_shard, rows = None, []
    n_img, n_fail, t0 = 0, 0, time.time()
    last_log = time.time()
    for b in dl:
        ok = b['ok']
        n = len(b['key'])
        e_sscd = np.zeros((n, 512), np.float16)
        e_l = np.zeros((n, 768), np.float16)
        e_b = np.zeros((n, 512), np.float16)
        a_score = np.zeros(n, np.float32)
        if ok.any():
            idx = ok.nonzero(as_tuple=True)[0].numpy()
            si = b['sscd_in'].to(dev, non_blocking=True)
            ci = b['clip_in'].to(dev, non_blocking=True).half()
            fs = sscd(si).float()
            fl = clip_l(ci).float()
            fb = clip_b(ci).float()
            fl_n = torch.nn.functional.normalize(fl, dim=-1)
            sc = aes(fl_n).squeeze(-1)
            e_sscd[idx] = torch.nn.functional.normalize(fs, dim=-1).cpu().numpy().astype(np.float16)
            e_l[idx] = fl_n.cpu().numpy().astype(np.float16)
            e_b[idx] = torch.nn.functional.normalize(fb, dim=-1).cpu().numpy().astype(np.float16)
            a_score[idx] = sc.float().cpu().numpy()
        for j in range(n):
            sh = b['shard'][j]
            if sh != cur_shard:
                if cur_shard is not None:
                    flush_shard(cur_shard, rows)
                cur_shard, rows = sh, []
            okj = bool(ok[j])
            if not okj:
                n_fail += 1
            rows.append({'key': b['key'][j], 'ok': okj, 'width': b['w'][j], 'height': b['h'][j],
                         'phash': b['phash'][j], 'aes_laion_improved': float(a_score[j]),
                         'sscd': e_sscd[j].tobytes(), 'clip_l': e_l[j].tobytes(),
                         'clip_b': e_b[j].tobytes()})
        n_img += n
        if time.time() - last_log > 60:
            el = time.time() - t0
            print(f'[rank{args.rank}] {n_img}/{len(ds)} ({100*n_img/len(ds):.1f}%) '
                  f'| {n_img/el:.0f} img/s | fail {n_fail} ({100*n_fail/max(n_img,1):.2f}%) '
                  f'| eta {(len(ds)-n_img)/(n_img/el)/3600:.1f}h', flush=True)
            last_log = time.time()
    if cur_shard is not None:
        flush_shard(cur_shard, rows)
    el = time.time() - t0
    print(f'[rank{args.rank}] DONE {n_img} imgs in {el/3600:.2f}h ({n_img/el:.0f} img/s), '
          f'fail {n_fail} ({100*n_fail/max(n_img,1):.3f}%)', flush=True)


if __name__ == '__main__':
    main()
