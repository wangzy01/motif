"""complete_v1 真实数据集（替代 StubT2IDataset）。

架构（见交接文档"数据线"决策）：每域**后台生产者线程**流式读 shard、materialize（取图→uint8）+
回源 caption 字段，推入 bounded queue；主循环把 queue 排入 per-domain **shuffle buffer**，按
sampler（4-way domain_mix + real band-first + anime 优先级）**带放回**抽 batch_size 个，
finalize caption（real 按抽中档选 captioner / anime 原生构建），uint8→[-1,1] tensor 堆成
batch {prompts, pixels}，yield。每 batch 后 buffer 滚动（淘旧补新）。rank shard 在 shard 列表层。

batch 契约对齐 model_mot.forward：{"prompts": list[str], "pixels": Tensor[B,3,S,S] in [-1,1]}。
VAE 在线 encode（forward 内），故只给 pixels。
"""

from __future__ import annotations

import hashlib
import os
import queue
import random
import threading
import time
from collections import Counter, defaultdict, deque

import numpy as np
import pyarrow.parquet as pq

from .caption_build import (build_anime_caption, build_textatlas_caption, finalize_real_caption,
                            truncate_to_token_budget)
from .config import load_sampling_config
from .image_fetch import (decode_rgb, fetch_url_bytes, monet_tar_path, read_gpic_bytes,
                          resize_crop_uint8, stream_tar_images, uint8_to_norm_tensor)
from .manifest_index import (CAPTIONER_CODE, CODE_TO_CAPTIONER, compute_real_global_stats,
                             list_shards, load_anime_shard, load_real_synth_shard,
                             load_textatlas_shard, shard_subset_for_rank, textatlas_subset_shard)
from .multi_ar import (Bucketer, BucketSchedule, build_adaptive_buckets, build_buckets,
                       buckets_by_ar_distance, nearest_allowed_bucket, resize_u8_to_bucket,
                       square_index)
from .native_safety import PARQUET_NATIVE_LOCK, configure_pyarrow_single_thread
from .sampler import CompleteSampler, DomainPool

configure_pyarrow_single_thread()

# MONET 源 caption 列 ↔ captioner code（band_captioner 用）
_MONET_SRC_COL = {
    CAPTIONER_CODE["gemini"]: "caption_gemini-2.5-flash-lite",
    CAPTIONER_CODE["sharegpt4v"]: "caption_sharegpt4v-7b",
    CAPTIONER_CODE["internvl"]: "caption_internvl-3-8b",
    CAPTIONER_CODE["florence"]: "caption_florence-2-large",
    CAPTIONER_CODE["original"]: "caption_original",
    CAPTIONER_CODE["florence_first_sentence"]: "caption_florence-2-large",
}
_ANIME_CAP_KEYS = ["rating_bucket", "refined_description", "medium_description", "short_description",
                   "tags_general", "tags_artist", "tags_character", "tags_copyright", "tags_meta",
                   "foreground", "background", "style", "image_type", "camera_and_lens_effects", "year"]

# tar 流式 yield 的局部打乱窗口（2026-07-09 流式化）：≤该窗口条 (row, raw) 缓冲后洗牌再吐,
# 保留去「连续切片」偏置的效果,同时把首条记录延迟从【整 tar 读完】(2-5GB,54-139s) 降到秒级、
# producer 峰值内存从整 shard 原始 JPEG (~2-5GB) 降到 ~窗口×单图 (~百 MB 以内)。
_TAR_SHUFFLE_CHUNK = 256


def _domain_seed(name: str) -> int:
    """稳定的域种子偏移。替代 hash(name)：PYTHONHASHSEED 未固定时 str hash 每进程随机 →
    shard 访问序/anime row-group 抽样不可复现（bug hunt MINOR 项）。"""
    return int.from_bytes(hashlib.md5(name.encode("utf-8")).digest()[:4], "little") % 100003


# --------------------------------------------------------------------------- #
# 回源 caption 源 parquet
# --------------------------------------------------------------------------- #
def _load_gpic_caption_src(src_root: str, src_shard: str) -> dict:
    """gpic_train_{shard}.parquet → key -> (caption, ext)。"""
    p = os.path.join(src_root, "parquets", f"gpic_train_{src_shard}.parquet")
    with PARQUET_NATIVE_LOCK:
        t = pq.read_table(p, columns=["key", "caption", "s3key"], use_threads=False)
    keys = t.column("key").to_pylist()
    caps = t.column("caption").to_pylist()
    s3 = t.column("s3key").to_pylist()
    out = {}
    for k, c, s in zip(keys, caps, s3):
        ext = os.path.splitext(s)[1] if s else ".jpg"
        out[k] = (c, ext)
    return out


def _load_monet_caption_src(src_root: str, src_shard: str) -> dict:
    """MONET/synth 源 parquet → id -> {code: text}（仅读出现的 captioner 列）。"""
    p = os.path.join(src_root, src_shard)
    cols = ["id"] + list(dict.fromkeys(_MONET_SRC_COL.values()))
    with PARQUET_NATIVE_LOCK:
        t = pq.read_table(p, columns=cols, use_threads=False)
    ids = t.column("id").to_pylist()
    coldata = {col: t.column(col).to_pylist() for col in cols if col != "id"}
    out = {}
    for i, _id in enumerate(ids):
        out[_id] = {code: coldata[col][i] for code, col in _MONET_SRC_COL.items()}
    return out


# --------------------------------------------------------------------------- #
# 域 shard 迭代器（materialize 记录）
# --------------------------------------------------------------------------- #
def _prefetch_raw(items, fetch_fn, workers=8, chunk=64):
    """并发预取原始图字节，按【输入顺序】yield (item, raw)；raw=None 表示取图失败（调用方跳过）。
    workers<=1 退回串行（逐项）。chunk = 在途并发批大小（内存↔并发折中）。
    用途：把 gpic 逐文件随机 GET / anime 逐 URL HTTP 这种 I/O-bound 的【单线程串行取图】改成线程池并发，
    单 producer 吞吐 ~5-6 图/s → 数十/s（瓶颈转为串行 decode）。详见数据集审计文档 M2/M3/C2。
    ⚠️ decode 不放进本池（2026-06-26 实测：8 procs/node 下 decode 进池 CPU 过订阅 + 只加速缓存域 anime
    导致其在 256 过供刷屏，慢挂载域 gpic/monet 仍受 IOPS/带宽天花板，聚合 mix 反而更差 → 回退为串行 decode）。"""
    items = list(items)
    if workers <= 1:
        for it in items:
            yield it, fetch_fn(it)
        return
    import concurrent.futures as _cf
    with _cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for s in range(0, len(items), chunk):
            blk = items[s:s + chunk]
            for it, raw in zip(blk, ex.map(fetch_fn, blk)):   # map 保序、池内并发
                yield it, raw


def iter_real_synth_shard(domain, manifest_shard, cfg, stats, stage, image_size, max_rows=None,
                          skip_image=False, fetch_workers=8, rng=None, bucketer=None, v_anchor=0.5,
                          force_square=False, square_idx=0):
    """yield real/synth 记录（按 src_shard 分组，回源 caption + tar/文件取图）。
    skip_image=True（latent cache 模式）：不取图/decode/resize（跳过 tar 流式与逐文件读），
    只产 caption+uid 字段；moments 由 batch 装配时按 uid 从缓存读。
    rng：打乱 src_shard 处理顺序 + 行序去偏。gpic/skip_image 路径 = 整 shard 行级 shuffle（旧语义不变）;
    monet/synth tar 路径（2026-07-09 流式化后）= tar 物理序 + _TAR_SHUFFLE_CHUNK 窗口内打乱——
    窗口外保留 tar 局部性（vs 旧版整 shard dict 物化后全打乱）,由下游 per-bucket buffer + 带权抽样
    继续混合;换来的是首条记录秒级可达（旧版须读完整个 2-5GB tar）+ 峰值内存降两个量级。"""
    arr = load_real_synth_shard(manifest_shard, domain, cfg, stats)
    elig = arr[f"elig_{image_size}"]            # REDESIGN: elig 按分辨率(image_size)非 stage 标签 → 512a/512b 都用 elig_512；旧 stage==image_size 零变化
    idxs = np.where(elig)[0]
    by_shard = defaultdict(list)
    for i in idxs:
        by_shard[arr["src_shard"][i]].append(int(i))
    shard_items = list(by_shard.items())
    if rng is not None:
        rng.shuffle(shard_items)
    count = 0
    for src_shard, rows in shard_items:
        if rng is not None:
            rng.shuffle(rows)
        src_root = arr["src_root"][rows[0]]
        try:
            capsrc = (_load_gpic_caption_src(src_root, src_shard) if domain == "gpic"
                      else _load_monet_caption_src(src_root, src_shard))
        except (FileNotFoundError, OSError):
            continue
        # 根修(2026-07-06, 零权重竞态): base_w≤0 行(未列 explicit 源, 如 da4k)任何加权路径都不可抽,
        #   入池纯占坑——极端时(零权重源 shard 连续段)灌满整域 buffer → MODE-A 单域 sel 抽中即
        #   _weighted_idx 全零 raise(512a 正式 step8300 实撞)。在取图【前】过滤: 不拉图不入池。
        rows = [i for i in rows if float(arr["base_w"][i]) > 0.0]
        if not rows:
            continue
        # 取图：gpic 逐文件（线程池并发预取，修单线程瓶颈）；monet/synth tar 流式（cache 模式跳过）。
        # 2026-07-09 流式化：旧版 dict(stream_tar_images(...)) 把整 shard tar（monet ~2.2GB/
        # synth ~5GB,wanted≈全部成员 → early-stop 无效）读完才 yield 首条 → 首批饿死主因 +
        # producer 峰值内存 2-5GB（bug hunt MAJOR）。改为边流边吐：按 tar 顺序遇到 wanted 成员
        # 即出记录,叠 _TAR_SHUFFLE_CHUNK 窗口内打乱（替代旧「整 shard 行级 shuffle」的去偏作用;
        # 窗口外顺序 = tar 序,下游 shuffle buffer/bucket queue 再混合）。
        if skip_image:
            pairs = ((i, None) for i in rows)
        elif domain == "gpic":
            def _gpic_fetch(i):                              # 闭包在本 src_shard 内即时消费，无跨轮晚绑定
                k = arr["src_key"][i]
                return read_gpic_bytes(src_root, src_shard, k, ext=capsrc.get(k, (None, None))[1])
            pairs = _prefetch_raw(rows, _gpic_fetch, workers=fetch_workers)
        else:
            key2row = {arr["src_key"][i]: i for i in rows}

            def _tar_pairs():
                chunk = []
                for key, raw in stream_tar_images(monet_tar_path(src_root, src_shard), set(key2row)):
                    chunk.append((key2row[key], raw))
                    if len(chunk) >= _TAR_SHUFFLE_CHUNK:
                        if rng is not None:
                            rng.shuffle(chunk)
                        yield from chunk
                        chunk = []
                if chunk:
                    if rng is not None:
                        rng.shuffle(chunk)
                    yield from chunk

            pairs = _tar_pairs()
        # try/finally：max_rows 提前 return / 上游 close 时显式关生成器链（GeneratorExit 传进
        # stream_tar_images 的 with tarfile.open → 立即释放 S3 流 fd,不等 GC;review A3）。
        try:
            for i, raw in pairs:
                key = arr["src_key"][i]
                if not skip_image:
                    if not raw:
                        continue
                    img = decode_rgb(raw)
                    if img is None:
                        continue
                    if force_square:
                        u8, _bk = resize_crop_uint8(img, image_size, v_anchor), square_idx
                    else:
                        u8, _bk = (bucketer.resize(img, image_size, v_anchor) if bucketer is not None
                                   else (resize_crop_uint8(img, image_size, v_anchor), 0))
                bc = arr["band_captioner"][i].tolist()
                if domain == "gpic":
                    cap_texts = {CAPTIONER_CODE["i1"]: capsrc.get(key, ("", None))[0]}
                else:
                    src = capsrc.get(key, {})
                    cap_texts = {c: src.get(c) for c in set(bc) if c > 0}
                rec = {"domain": domain, "served_mask": int(arr["served_mask"][i]),
                       "base_w": float(arr["base_w"][i]), "band_captioner": bc,
                       "cap_texts": cap_texts, "uid": arr["uid"][i]}
                if not skip_image:
                    rec["u8"] = u8
                    rec["bucket"] = _bk
                yield rec
                count += 1
                if max_rows and count >= max_rows:
                    return
        finally:
            if hasattr(pairs, "close"):
                pairs.close()


def iter_anime_shard(source_shard, cfg, stage, image_size, anime_cache=None, max_rows=None,
                     rng=None, nrows=None, skip_image=False, fetch_workers=8, bucketer=None, v_anchor=0.5):
    """yield anime 记录（源 shard 自带 caption；HTTP 取图）。nrows 限制每次读取行数；传 rng →
    随机 row-group 取样（去 shard 头部偏置，逐渐覆盖全 shard）。
    skip_image=True（latent cache 模式）：不取图/decode/resize，只产 caption+uid 字段，
    moments 由 batch 装配时按 uid 从缓存读。"""
    arr = load_anime_shard(source_shard, cfg, with_caption=True, nrows=nrows, rng=rng)
    elig = arr[f"elig_{image_size}"]            # REDESIGN: elig 按分辨率(image_size)非 stage 标签 → 512a/512b 都用 elig_512；旧 stage==image_size 零变化
    # REDESIGN gated「此阶段不用 tag」：require_nl_stages 含本 stage 时，丢弃无-NL(tag-only) 行
    # （has_nl 与 caption_build:285 同义=bool(refined|medium|short)；无-NL 行只能走 tag，过滤后该阶段 anime caption 100% NL、零 tag）。
    # 空表(旧 JSON/512a/512b)→ 不过滤 → 字节级零变化。被丢的行在未启用该 stage 的阶段(如 512a/512b)照常参与。
    if cfg.anime_caption.requires_nl(stage):
        has_nl = np.fromiter((bool(r) or bool(m) or bool(s) for r, m, s in
                              zip(arr["refined_description"], arr["medium_description"], arr["short_description"])),
                             dtype=bool, count=len(elig))
        elig = elig & has_nl
    idxs = np.where(elig)[0]
    if rng is not None:
        rng.shuffle(idxs)
    freq = Counter()                                       # shard-local general tag 频率（freq dropout 用）
    for tg in arr["tags_general"]:
        if tg:
            freq.update(tg)
    count = 0
    # 取图：逐 URL HTTP（线程池并发预取，修单线程瓶颈 + 配合 403 不重试）
    if skip_image:
        pairs = ((int(i), None) for i in idxs)
    else:
        pairs = _prefetch_raw([int(i) for i in idxs],
                              lambda i: fetch_url_bytes(arr["s3key"][i], cache_dir=anime_cache),
                              workers=fetch_workers)
    for i, raw in pairs:
        i = int(i)
        if not skip_image:
            if not raw:
                continue
            img = decode_rgb(raw)
            if img is None:
                continue
            u8, _bk = (bucketer.resize(img, image_size, v_anchor) if bucketer is not None
                       else (resize_crop_uint8(img, image_size, v_anchor), 0))
        cap_fields = {k: arr[k][i] for k in _ANIME_CAP_KEYS}
        rec = {"domain": "anime", "priority": float(arr["priority"][i]),
               "cap_fields": cap_fields, "general_freq": freq, "key": arr["key"][i],
               "cluster": arr["cluster"][i]}
        if not skip_image:
            rec["u8"] = u8
            rec["bucket"] = _bk
        yield rec
        count += 1
        if max_rows and count >= max_rows:
            return


def iter_textatlas_shard(sidecar_shard, cfg, stage, image_size, source_root,
                         local_root=None, max_rows=None, skip_image=False, rng=None, bucketer=None, v_anchor=0.5,
                         force_square=False, square_idx=0):
    """yield textatlas 记录（第 5 域，注入域）。sidecar 取 elig/caption；源 parquet **内联图字节**
    按 image_path join（不像其它域走 image_fetch）。
    源图路径：优先 local_root（/local NVMe 预存，快）；缺失回退 source_root（S3 挂载，保正确性，
    换机器/未预存时仍可跑）。随机 row-group 顺序 + 组内打乱去【头部/连续切片】偏置（600 shard≫world
    故文件级分片已无 per-rank 偏斜，无需 force_row_level）。每 row-group ~100 行、内存有界。"""
    arr = load_textatlas_shard(sidecar_shard, stage)
    elig = arr[f"elig_{image_size}"]            # REDESIGN: elig 按分辨率(image_size)非 stage 标签 → 512a/512b 都用 elig_512；旧 stage==image_size 零变化
    ip_arr = arr["image_path"]
    lut = {ip_arr[i]: int(i) for i in np.where(elig)[0]}    # image_path -> sidecar 行（仅 elig）
    if not lut:
        return
    subset, shardname = textatlas_subset_shard(sidecar_shard)
    src_path = None
    if local_root:                                         # /local 预存优先（绕开 S3 挂载带宽）
        cand = os.path.join(local_root, subset, shardname)
        if os.path.exists(cand):
            src_path = cand
    if src_path is None:                                   # 回退 S3 源（保正确性）
        src_path = os.path.join(source_root, subset, shardname)
    with PARQUET_NATIVE_LOCK:
        pf = pq.ParquetFile(src_path)
        rg_order = list(range(pf.num_row_groups))
    if rng is not None:
        rng.shuffle(rg_order)
    count = 0
    for rg in rg_order:
        with PARQUET_NATIVE_LOCK:
            batch = pf.read_row_group(rg, columns=["image", "image_path"], use_threads=False)
        imgs = batch.column("image").to_pylist()
        ipaths = batch.column("image_path").to_pylist()
        order = list(range(len(imgs)))
        if rng is not None:
            rng.shuffle(order)
        for j in order:
            i = lut.get(ipaths[j])
            if i is None:                                  # 非 elig / 不在 sidecar
                continue
            if not skip_image:
                st = imgs[j]
                raw = st.get("bytes") if st else None
                if not raw:
                    continue
                img = decode_rgb(raw)
                if img is None:
                    continue
                if force_square:
                    u8, _bk = resize_crop_uint8(img, image_size, v_anchor), square_idx
                else:
                    u8, _bk = (bucketer.resize(img, image_size, v_anchor) if bucketer is not None
                               else (resize_crop_uint8(img, image_size, v_anchor), 0))
            caps = (arr["caption1"][i], arr["caption2"][i], arr["caption3"][i],
                    arr["caption4"][i], arr["caption5"][i])
            rec = {"domain": "textatlas", "served_mask": 0, "base_w": 1.0,
                   "captions": caps, "uid": arr["uid"][i]}
            if not skip_image:
                rec["u8"] = u8
                rec["bucket"] = _bk
            yield rec
            count += 1
            if max_rows and count >= max_rows:
                return


# --------------------------------------------------------------------------- #
# 后台生产者
# --------------------------------------------------------------------------- #
class _Producer(threading.Thread):
    def __init__(self, domain, shards, ds, out_q, stop_evt, seed):
        super().__init__(daemon=True)
        self.domain, self.shards, self.ds = domain, shards, ds
        self.q, self.stop_evt = out_q, stop_evt
        self.rng = random.Random(seed)

    def run(self):
        order = list(self.shards)
        serial_native = bool(getattr(self.ds, "producer_native_io_serial", False))
        native_lock = getattr(self.ds, "_producer_native_io_lock", None)

        def _next_record(iterator):
            if serial_native and native_lock is not None:
                with native_lock:
                    return next(iterator)
            return next(iterator)

        while not self.stop_evt.is_set():
            self.rng.shuffle(order)
            for shard in order:
                if self.stop_evt.is_set():
                    return
                try:
                    v_anchor = self.ds._v_anchor_for(self.domain)   # 该域裁剪竖直锚点(缺省 0.5=旧行为)
                    if self.domain == "anime":
                        it = iter_anime_shard(shard, self.ds.cfg, self.ds.stage, self.ds.image_size,
                                              self.ds.anime_cache_dir, self.ds.max_rows_per_shard,
                                              rng=self.rng, nrows=self.ds.anime_nrows,
                                              skip_image=self.ds.use_latent_cache,
                                              fetch_workers=self.ds.fetch_workers_anime,  # anime 串行(默认)，不过订阅挂载
                                              bucketer=self.ds._mar_bucketer, v_anchor=v_anchor)
                    elif self.domain == "textatlas":
                        it = iter_textatlas_shard(shard, self.ds.cfg, self.ds.stage, self.ds.image_size,
                                                  self.ds.cfg.textatlas_source_root,
                                                  local_root=self.ds.textatlas_local_root,
                                                  max_rows=self.ds.max_rows_per_shard,
                                                  skip_image=self.ds.use_latent_cache, rng=self.rng,
                                                  bucketer=self.ds._mar_bucketer, v_anchor=v_anchor,
                                                  force_square=(self.ds._mar_assignment_mode == "item_queue"
                                                                and self.domain in self.ds._mar_square_only),
                                                  square_idx=self.ds._mar_square_idx)
                    else:
                        it = iter_real_synth_shard(self.domain, shard, self.ds.cfg, self.ds.stats,
                                                   self.ds.stage, self.ds.image_size,
                                                   self.ds.max_rows_per_shard,
                                                   skip_image=self.ds.use_latent_cache,
                                                   fetch_workers=self.ds.fetch_workers,  # gpic 逐文件用并发(monet/synth 走 tar 流不用此参)
                                                   rng=self.rng,                          # 行内打乱(去连续切片偏置，对齐 anime)
                                                   bucketer=self.ds._mar_bucketer, v_anchor=v_anchor,
                                                   force_square=(self.ds._mar_assignment_mode == "item_queue"
                                                                 and self.domain in self.ds._mar_square_only),
                                                   square_idx=self.ds._mar_square_idx)
                    it = iter(it)
                    while not self.stop_evt.is_set():
                        try:
                            rec = _next_record(it)
                        except StopIteration:
                            break
                        if self.stop_evt.is_set():
                            return
                        # canary(2026-07-06 零权重修复): iter 层已根修过滤, 此处理应恒 0; >0=有路径漏网(metrics buf_zerowdrop_*)
                        if self.domain != "anime" and rec.get("base_w", 1.0) <= 0.0:
                            self.ds.zerow_dropped[self.domain] += 1
                            continue
                        while not self.stop_evt.is_set():
                            try:
                                self.q.put(rec, timeout=1.0)
                                self.ds.produced[self.domain] += 1   # 监控：该域产出+1（GIL 下计数足够安全）
                                break
                            except queue.Full:
                                continue
                except Exception as e:                     # noqa: BLE001 单 shard 失败不杀线程
                    print(f"[producer {self.domain}] shard {shard} 跳过: {type(e).__name__} {e}", flush=True)


# --------------------------------------------------------------------------- #
# 数据集主类
# --------------------------------------------------------------------------- #
class CompleteV1Dataset:
    """complete_v1 真实 T2I 数据集。store_cls 入口；config.dataset 整段做 kwargs。"""

    def __init__(self, sampling_config, stage, image_size=None, seed=0,
                 buffer_size=256, queue_size=128, min_fill=64,
                 max_shards=None, max_rows_per_shard=None, anime_nrows=20000, anime_cache_dir=None,
                 stats_cache=None, rank=None, world_size=None,
                 tokenizer_path=None, caption_token_budget=None, template_reserve=41,
                 use_latent_cache=False,
                 latent_cache_root="/s3/evan/mot_latents/complete_v1_flux2_moments",
                 latent_cache_index_name="uid2path.parquet",
                 first_fill_policy="all_min", first_fill_timeout_s=120.0, fetch_workers=8,
                 fetch_workers_anime=1, producer_native_io_serial=False,
                 drain_mode="uniform", keep_present_timeout_s=2.0,
                 textatlas_local_root=None, multi_ar=None, crop_v_anchor=None, **_):
        # 裁剪竖直锚点（Path B, 缺头修复 2026-07-05）：per-domain {domain: v_anchor∈[0,1]}，
        # 0=贴顶保头 / 0.5=居中=旧行为 / 1=贴底；缺省域=0.5（逐字节不变）。anime 中心裁方图×portrait
        # 顶损 → 配 anime:0.15 保头。透传到 resize_crop_uint8 / resize_crop_to_bucket / resize_u8_to_bucket 的竖直裁剪。
        self._crop_v_anchor = ({str(k): float(v) for k, v in dict(crop_v_anchor).items()}
                               if crop_v_anchor else {})
        # textatlas（第 5 域）源图预存根：优先从此本地 NVMe 读，缺失/未配置回退 cfg.textatlas_source_root
        # （S3）。Evan: 预存为首选、保底走 S3 保正确性；换机器无预存时自动回退。
        self.textatlas_local_root = textatlas_local_root or None
        self.producer_native_io_serial = bool(producer_native_io_serial)
        self._producer_native_io_lock = threading.RLock()
        self.anime_nrows = int(anime_nrows) if anime_nrows else None
        self.use_latent_cache = bool(use_latent_cache)
        self._latent_cache_root = latent_cache_root
        self._latent_cache_index_name = latent_cache_index_name
        self.cache_index = None
        self.cfg = load_sampling_config(sampling_config)
        self.stage = str(stage)
        self.image_size = int(image_size) if image_size else \
            {"256": 256, "512": 512, "512a": 512, "512b": 512, "1024": 1024}[self.stage]
        self.seed = int(seed)
        # ---- multi-AR（REDESIGN Phase2b，§7；gated by multi_ar.enabled；enabled=false→旧方图路径，live 零影响）----
        self._multi_ar_active = False            # warmup 期 False（producer 出方图）；过 warmup 由 loader 翻 True
        self._mar_start_step = 0                 # resume：trainer.set_start_step 注入续训全局步（warmup 续算，防 resume 重 warmup）
        self._mar_bucketer = None
        self._mar_schedule = None
        self._mar_buckets = None
        self._mar_square_idx = 0
        self._mar_warmup_steps = 0
        self._mar_warmup_exposure = 0            # gated(Path C 前置)：>0 时 loader 按 曝光/global_batch 派生 warmup_steps（0=沿用 warmup_steps 旧行为）
        self._mar_reresize_count = 0             # 监控：域内无 B_t-native → re-resize 兜底总次数
        self._mar_reresize_by_domain = Counter()  # 监控(护栏2)：每域 re-resize 次数（看 synthetic 等方图域被裁进非方桶的频率）
        self._mar_sampled_by_domain = Counter()   # 监控：每域被抽中总数（配合算 re-resize 率）
        self._mar_extreme_clamped = 0             # adaptive ARB: raw aspect 超过 max_aspect 后被 clamp 的样本数
        self._mar_zerow_pool_skips = Counter()    # 监控(2026-07-06 零权重修复): 全零权重池被判缺席次数(护栏层触发计数)
        self.zerow_dropped = Counter()            # 监控(同上): producer canary——iter 层根修后应恒 0, >0=漏网
        self._mar_last_bucket = None
        self._mar_square_only = set()            # 护栏2 逃生口(默认空)：列出的域【不】re-resize 进非方桶(square-only 特例); 已接线(2026-07-05), 见 loader 补偿推导
        self._mar_square_only_alpha = 0.0        # 护栏2 接线(MODE-B): 方图 batch 抽 square-only 子池的概率 alpha=Σ_{d∈S}mix_d/P(square)（0=旧行为/未配置/被 MODE-A 吸收）
        self._mar_square_prob = 1.0              # P(square 桶)（从 schedule.probs 归一后读; 未开 multi_ar=1.0）
        self._mar_native_dist = None             # Tier2(MODE-A affinity): per-domain native 桶分布 {d: np.array[n]}(行归一; square_only 域已 one-hot(square) 覆写)
        self._mar_pdb = None                     # Tier2(MODE-A affinity): P(d|B) 表 {bucket_idx: (doms, probs)}(init 算好 7×5; None=未配置→MODE-B/off)
        self._mar_assignment_mode = "current"    # current=旧 512a path; item_queue=NAI-like per-bucket queues
        self._mar_bucket_buffer_size = 64
        self._mar_bucket_min_fill = 1
        self._mar_bucket_wait_timeout_s = 120.0
        self._mar_bucket_wait_native_s = 30.0    # 软计划 Phase1：等全 native 供给的窗口（2026-07-09 重构）
        self._mar_bucket_prob_floor = 0.0        # 调度剪枝：派生 P(B)<floor 的稀有桶折进最近保留桶（0=不剪）
        self._mar_allowed_buckets = None         # 剪枝后的保留桶集（None=全部;producer 指派与调度共用）
        self._mar_plan_degraded = Counter()      # 监控：软计划 Phase3（计划封顶/present 重归一）触发批数
        self._mar_bucket_mode = "fixed"
        self._mar_target_area = 262144
        self._mar_divisor = 16
        self._mar_adaptive_no_upscale = True
        self._mar_adaptive_min_area_ratio = 0.75
        self._mar_adaptive_max_aspect = 4.0
        _mar = dict(multi_ar) if multi_ar else None
        self.multi_ar_enabled = bool(_mar and _mar.get("enabled", False))
        if self.multi_ar_enabled:
            if self.use_latent_cache:
                raise ValueError("multi_ar 与 use_latent_cache 暂不兼容（latent cache 按方图 stage 预存）；512 在线 encode 即可。")
            self._mar_assignment_mode = str(_mar.get("assignment_mode", "current") or "current")
            if self._mar_assignment_mode not in {"current", "item_queue"}:
                raise ValueError("dataset.multi_ar.assignment_mode must be one of {'current','item_queue'}")
            self._mar_bucket_mode = str(_mar.get("bucket_mode", "fixed") or "fixed")
            if self._mar_bucket_mode == "anchors_v1":
                # Refuse loudly instead of silently falling through to build_buckets()'s
                # 7-entry fixed AR table with adaptive=False. anchors_v1 is an
                # offline-only geometry: this streaming producer assigns through
                # Bucketer -> assign_bucket/assign_adaptive_bucket, neither of which has
                # a feasible-anchor test, so it would upsample (a 400x300 image lands on
                # 576x432, a 1.44x enlargement). Zero upsampling only holds through the
                # offline catalog path plus anchors_v1.verify_binding() at runtime.
                raise ValueError(
                    "dataset.multi_ar.bucket_mode=anchors_v1 is offline-only; the "
                    "streaming CompleteV1Dataset producer cannot honour its "
                    "zero-upsampling contract. Use the offline dataline "
                    "(training.dataline_offline.offline_dataset.OfflineExactDataset)."
                )
            if self._mar_bucket_mode not in {"fixed", "adaptive"}:
                raise ValueError("dataset.multi_ar.bucket_mode must be one of {'fixed','adaptive'}")
            self._mar_target_area = int(_mar.get("target_area", 262144))
            self._mar_divisor = int(_mar.get("divisor", 16))
            self._mar_adaptive_no_upscale = bool(_mar.get("adaptive_no_upscale", True))
            self._mar_adaptive_min_area_ratio = float(_mar.get("adaptive_min_area_ratio", 0.75))
            self._mar_adaptive_max_aspect = float(_mar.get("adaptive_max_aspect", 4.0))
            if self._mar_bucket_mode == "adaptive":
                self._mar_buckets = build_adaptive_buckets(
                    self._mar_target_area, self._mar_divisor,
                    self._mar_adaptive_min_area_ratio, self._mar_adaptive_max_aspect)
            else:
                self._mar_buckets = build_buckets(self._mar_target_area, self._mar_divisor, _mar.get("ar_list"))
            self._mar_square_idx = square_index(self._mar_buckets)
            self._mar_warmup_steps = int(_mar.get("warmup_steps", 0) or 0)
            # 曝光化 warmup（gated；Path C 前置 (c)）：>0 → loader(batch_size) 时按 曝光/global_batch 派生
            # warmup_steps，使 warmup 曝光量固定不随卡数/global_batch 浮动；0/缺省 → 沿用上面的 warmup_steps 旧行为。
            self._mar_warmup_exposure = int(_mar.get("warmup_exposure_images", 0) or 0)
            self._mar_square_only = set(_mar.get("square_only_domains", []) or [])
            self._mar_bucket_buffer_size = int(_mar.get("bucket_buffer_size", 64) or 64)
            self._mar_bucket_min_fill = int(_mar.get("bucket_min_fill", 1) or 1)
            self._mar_bucket_wait_timeout_s = float(_mar.get("bucket_wait_timeout_s", 120.0) or 0.0)
            self._mar_bucket_wait_native_s = float(_mar.get("bucket_wait_native_s", 30.0) or 0.0)
            self._mar_bucket_prob_floor = float(_mar.get("bucket_prob_floor", 0.0) or 0.0)
            # ---- 域采样模式（三选一, 互斥；warmup 期/未配置=旧行为）----
            #   MODE-A affinity（Tier2, native_bucket_dist 配了）：square_only 硬约束被 one-hot(square) 覆写吸收进
            #     同一套数学（不与 alpha 叠加=无双重补偿）；P(B)=Σ_d mix_d·P_native(B|d) 派生并【替代】config
            #     bucket_probs 作调度先验（保证恒等式精确）；域采样用 P(d|B_t)=mix_d·P_native(B_t|d)/P(B_t)。
            #   MODE-B alpha（无 native 表、只有 square_only）：方图 batch 以 alpha=Σ_S mix/P(square) 抽 S 子池。
            #   MODE-off（都没配）：全池 mix（旧行为逐字节不变）。
            _nbd = {str(k): list(v) for k, v in dict(_mar.get("native_bucket_dist") or {}).items()} or None
            _sched_probs = _mar.get("bucket_probs")
            if self._mar_assignment_mode == "item_queue" and self._mar_bucket_mode == "adaptive":
                if _sched_probs is None or _nbd is None:
                    raise ValueError("item_queue+adaptive 必须提供实测 bucket_probs 和 native_bucket_dist，"
                                     "不能使用默认 schedule（会抽到稀有/无货桶导致 rank 等待或超时）。")
            if _sched_probs is not None and len(list(_sched_probs)) != len(self._mar_buckets):
                if rank in (None, 0):
                    print(f"[dataline] ⚠️ multi-AR: bucket_probs 长度 {len(list(_sched_probs))} != "
                          f"bucket 数 {len(self._mar_buckets)}，忽略并使用默认/派生分布。", flush=True)
                _sched_probs = None
            if self._mar_assignment_mode == "item_queue" and self._mar_bucket_mode == "adaptive" and _sched_probs is None:
                raise ValueError("item_queue+adaptive 的 bucket_probs 长度必须等于 bucket 数。")
            if _nbd:
                self._mar_native_dist, _pB, self._mar_pdb = self._build_native_affinity(
                    self.cfg.domain_mix[self.stage], _nbd, len(self._mar_buckets),
                    self._mar_square_idx, self._mar_square_only)
                if _sched_probs is not None:                # config bucket_probs 仅作参考: 与派生值偏差>0.02 警告, 恒以派生值为准
                    _cp = np.asarray(list(_sched_probs), dtype=np.float64); _cp = _cp / _cp.sum()
                    _dev = float(np.max(np.abs(_cp - _pB / max(_pB.sum(), 1e-12))))
                    if _dev > 0.02 and rank in (None, 0):   # 构造参 rank(此时 self.rank 未定); None=单进程也打
                        print(f"[dataline] ⚠️ multi-AR: config bucket_probs 与派生 P(B)=Σ mix·P_native 最大偏差 "
                              f"{_dev:.4f}>0.02 → 以派生值为准（config 值仅参考）。派生={np.round(_pB, 4).tolist()}", flush=True)
                # ---- 调度剪枝（2026-07-09,组批 blocker 修复之一,gated by bucket_prob_floor）----
                #   派生 P(B)<floor 的稀有桶（多为 256 样本审计的 1/256 量化噪声）折进最近保留桶：
                #   各域 native 行质量搬运 → 重derive P(B)/P(d|B)（恒等式 Σ_B P(B)P(d|B)=mix_d 保持精确）;
                #   producer 侧 Bucketer 同步只指派保留桶（一次 resize 到位,无调度不到的死库存）。
                if self._mar_bucket_prob_floor > 0:
                    (self._mar_native_dist, _pB, self._mar_pdb,
                     self._mar_allowed_buckets) = self._prune_scheduled_buckets(
                        self.cfg.domain_mix[self.stage], self._mar_native_dist, _pB,
                        self._mar_buckets, self._mar_square_idx, self._mar_bucket_prob_floor)
                    if rank in (None, 0):
                        _nz = int(np.count_nonzero(_pB))
                        print(f"[dataline] multi-AR: bucket_prob_floor={self._mar_bucket_prob_floor} → "
                              f"调度桶 {_nz}/{len(self._mar_buckets)}（保留集质量=1.0,稀有桶已折进最近邻）", flush=True)
                _sched_probs = [float(x) for x in _pB]
            self._mar_schedule = BucketSchedule(len(self._mar_buckets), self._mar_square_idx,
                                                probs=_sched_probs, base_seed=self.seed)
            self._mar_bucketer = Bucketer(
                self._mar_buckets, self._mar_square_idx, self,
                adaptive=(self._mar_bucket_mode == "adaptive"),
                target_area=self._mar_target_area, divisor=self._mar_divisor,
                no_upscale=self._mar_adaptive_no_upscale,
                max_aspect=self._mar_adaptive_max_aspect,
                allowed=self._mar_allowed_buckets)
            self._mar_square_prob = float(self._mar_schedule.probs[self._mar_square_idx])
            if self._mar_pdb is None:
                # MODE-B alpha 接线(2026-07-05 拍板): 从归一后的 schedule.probs 读 P(square) + 按 config domain_mix 算补偿系数；
                #   物理不可行(P(square)<Σ_{d∈S}mix_d, 方图份额装不下这些域)→ __init__ 硬报错(不静默)。空列表→alpha=0=旧行为。
                #   （MODE-A 下无此可行性问题: P(B=square)=Σ mix·P_native(square|d) ≥ Σ_{d∈S}mix_d·1 结构性成立。）
                self._mar_square_only_alpha = self._square_only_compensation(
                    self.cfg.domain_mix[self.stage], self._mar_square_prob, self._mar_square_only)
            # 互斥硬断言: MODE-A(affinity) 与 MODE-B(alpha) 绝不同时生效（square_only 已被 one-hot 吸收, alpha 恒 0）
            assert not (self._mar_pdb is not None and self._mar_square_only_alpha > 0), \
                "BUG: affinity(P(d|B)) 与 square_only alpha 补偿同时激活（会双重补偿）"
        self.buffer_size = int(buffer_size)
        self.queue_size = int(queue_size)
        self.min_fill = int(min_fill)
        self.max_rows_per_shard = max_rows_per_shard
        self.anime_cache_dir = anime_cache_dir
        self.first_fill_policy = str(first_fill_policy or "all_min")
        if self.first_fill_policy not in {"all_min", "any_pool"}:
            raise ValueError("dataset.first_fill_policy must be one of {'all_min', 'any_pool'}")
        # 韧性：all_min 首批/补料阻塞的总超时（秒）。某域数据源慢/断供（anime R2 失败、tar/本地慢）
        # 到点不再死等该域，带已就绪域先开跑、慢域随后回填（sampler 对 present 域重归一）。
        # 健康启动通常远 <timeout 即填满 → 不改变行为；仅数据源异常时生效。<=0 关闭（退回旧死等）。
        self.first_fill_timeout_s = float(first_fill_timeout_s or 0.0)
        # 取图并发：per-域。gpic 逐文件随机读用 fetch_workers（默认 8，修逐文件单线程瓶颈）；
        # anime 逐 URL HTTP 用 fetch_workers_anime（默认 1=串行）——anime 是缓存/R2 域，并发只会
        # 过订阅 mountpoint-s3 把走 tar 顺序流的 monet/synth producer 饿死(审计 M2/回归诊断 §2.3)，
        # 且 anime 即便串行也供得过量(fwcompare fw=1 实测 anime 仍 65%)，配比由 drain_mode 兜住、无需并发。
        self.fetch_workers = max(1, int(fetch_workers))
        self.fetch_workers_anime = max(1, int(fetch_workers_anime))
        # buffer 排空策略（修 domain_mix 回归，回归诊断 §4）：
        #   uniform     —— 旧行为：每批对【每个域】无差别 popleft bs//2 → 每域都被要求以相同高速率产出
        #                  才能保持 present，慢域(gpic 逐文件/被过订阅的 monet/synth)跟不上→buffer 空→
        #                  sampler renorm 把份额全摊给永远 present 的 anime（实测 anime 57-65% 过供）。
        #   proportional(F1) —— 每批按各域【实际被抽中量】popleft → 每域只需产出自己那份 domain_mix 的
        #                  速率即可保持 present（synth 5% 只需 ~5/批 而非 48/批）；over-producer(anime)
        #                  被封顶在 target（多余的 FIFO 淘汰=无偏丢弃）。
        #   keep_present(F2) —— F1 + 目标域(domain_mix>0)buffer 空时阻塞补料(带 keep_present_timeout_s
        #                  超时回退)，进一步防 renorm-away；代价是慢域可能拖慢 batch 节奏。
        self.drain_mode = str(drain_mode or "uniform")
        if self.drain_mode not in {"uniform", "proportional", "keep_present"}:
            raise ValueError("dataset.drain_mode must be one of {'uniform','proportional','keep_present'}")
        self.keep_present_timeout_s = float(keep_present_timeout_s or 0.0)
        # caption 真实 token 截断（F3）：给 tokenizer_path 即按真实 Qwen token 截到 budget，
        # 消除模型端 max_length 静默硬截。budget = caption_token_budget 或
        # (model_text_max_length - template_reserve)；不给 tokenizer_path 则退化为上游 proxy 截断。
        self.tokenizer = None
        self.caption_budget = None
        if tokenizer_path:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.caption_budget = int(caption_token_budget) if caption_token_budget else \
                int(self.cfg.anime_caption.mtml_for(self.stage)) - int(template_reserve)  # REDESIGN per-stage（512a/512b=512→budget 471；256=384→343 不变）
        else:                                              # F5：无 tokenizer → 退 proxy 截断，模型端可能 mid-token 硬截
            print("[dataline] ⚠️ 未提供 tokenizer_path：caption 走 proxy(words×1.32)截断，"
                  "真实 token 可能超 text_max_length → 模型端静默硬截（实测 anime@256 ~8%）。"
                  "正式训练请在 config.dataset 设 tokenizer_path（3 个 stage config 已设）。", flush=True)
        # rank：优先显式参数，否则探测 torch.distributed
        if rank is None or world_size is None:
            rank, world_size = self._detect_rank()
        self.rank, self.world_size = int(rank), int(world_size)

        all_shards = list_shards(self.cfg)
        # 分片：anime 仅 17 个按质量/NL 有序的 fat shard，文件级分片会让每 rank 只见 1 个 shard 的偏斜
        # 子分布（has_nl 52.8%–95.2%、NSFW/risk 跨 shard 强偏，审计 M5/M2b）→ 恒走行级（anime 按 nrows
        # 随机 row-group 抽样 + 取图按需，行级几乎零额外开销）。gpic/monet/synth 各 64 shard、整 shard 读，
        # 保持文件级（仅 shard 数<world 时回退行级防域消失），避免高 world 下重复读全 shard 的吞吐回退。
        _row_level_domains = {"anime"}
        self.shards = {d: shard_subset_for_rank(v, self.rank, self.world_size,
                                                force_row_level=(d in _row_level_domains))
                       for d, v in all_shards.items()}
        if max_shards:
            self.shards = {d: v[: int(max_shards)] for d, v in self.shards.items()}
        # 全局源/簇统计（rule #7「整池统计一次」）：必须用【未切片】的全量 monet 列表，否则各 rank
        # base_w 不一致。多 rank 协调：仅 rank0 计算并原子写缓存，barrier 后非 rank0 读缓存。
        # stats 恒用【全量】monet（rule#7 整池统计）：max_shards 只缩 self.shards（data 加载），不缩 stats。
        # 2026-06-30 修复：原先 max_shards 也缩 full_monet，与 stats cache 指纹叠加会用【部分 stats】覆盖正式
        #   cache（_n_shards 写成 2）；且部分 stats 使 base_w 源归一不准。stats 恒全量 → cache 不被污染、base_w 正确。
        full_monet = all_shards["monet_real"]
        self.stats = self._build_global_stats(full_monet, stats_cache)
        # 该 stage 哪些域参与（gpic 在 512/1024 由 domain_mix=0 自然不进；但仍建生产者，sampler 不抽）
        self.domains = ["anime", "monet_real", "gpic", "synthetic"]
        # textatlas（第 5 域）：cfg 配了 sidecar 根才启用。latent-cache 模式暂不支持 textatlas
        # （无预存 latent）→ 跳过本域，domain_mix 的 textatlas 权重由 sampler 按 present 域重归一。
        if self.cfg.manifest_textatlas:
            if self.use_latent_cache:
                print("[dataline] ⚠️ textatlas 暂不支持 latent-cache 模式（无预存 latent）→ 本 run 跳过 "
                      "textatlas 域；其 domain_mix 权重由 sampler 按 present 域重归一摊给其余域。"
                      "在线(非 cache)模式正常启用。", flush=True)
            else:
                self.domains.append("textatlas")
        self.domain_counter = Counter()                  # 监控：累计各域【曝光/采样】数 → realized domain_mix 真值
        self.produced = Counter()                        # 监控：各域 producer 累计【产出】数（喂入 queue）→ 供数吞吐真值
        self.buffer_fill = {}                            # 监控：各域 buffer 当前占比（loader 每批更新）→ 供数瓶颈直观
        self.cluster_counter = Counter()                 # 监控（F2a）：anime cluster 曝光分布（确认构造平衡）
        self.cache_miss_counter = 0                      # latent cache 模式：累计缓存缺失（跳过）数
        # latent cache 模式：加载全局 uid->path 索引；producer 跳取图、batch 按 uid 读 moments、缺失跳过。
        if self.use_latent_cache:
            from training.dataline.latent_cache import LatentCacheIndex
            stage_dir = os.path.join(self._latent_cache_root, f"stage={self.stage}")
            self.cache_index = LatentCacheIndex(stage_dir, index_name=self._latent_cache_index_name)
            print(f"[dataline] ✅ latent cache 开启：stage={self.stage} 索引 {len(self.cache_index)} uid"
                  f"（root={self._latent_cache_root}）；producer 跳取图、batch 按 uid 读 moments、缺失跳过。",
                  flush=True)

    def _build_global_stats(self, full_monet_shards, stats_cache):
        """rule #7 全局统计的多 rank 安全构建：
          - world=1：直接全池计算（+原子写缓存供后续复用）。
          - world>1：rank0 计算并原子写缓存 → dist.barrier → 非 rank0 读缓存（compute=False）。
        缓存已存在时所有 rank 直接读（无重算）。"""
        if stats_cache and os.path.exists(stats_cache):
            return compute_real_global_stats(full_monet_shards, cache_path=stats_cache)
        if self.world_size <= 1:
            return compute_real_global_stats(full_monet_shards, cache_path=stats_cache)
        # 多 rank：rank0 先算/写，其余等 barrier 再读
        dist = None
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                dist = _dist
        except Exception:
            dist = None
        if self.rank == 0:
            stats = compute_real_global_stats(full_monet_shards, cache_path=stats_cache, compute=True)
            if dist is not None:
                dist.barrier()
            return stats
        if dist is not None:
            dist.barrier()                                # 等 rank0 写完缓存
        if not (stats_cache and os.path.exists(stats_cache)):
            # 无 dist 协调又无缓存：回退为各 rank 全池自算（仍一致，只是重复算）
            return compute_real_global_stats(full_monet_shards, cache_path=None, compute=True)
        return compute_real_global_stats(full_monet_shards, cache_path=stats_cache, compute=False)

    @staticmethod
    def _detect_rank():
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
        except Exception:
            pass
        return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))

    def set_start_step(self, step: int):
        """trainer 在 resume 后注入续训【全局步】→ multi-AR warmup 按全局步判定（防 resume 重跑 square warmup）。
        非 multi_ar 时无副作用。"""
        self._mar_start_step = int(step)

    def _v_anchor_for(self, domain: str) -> float:
        """该域裁剪竖直锚点（缺省 0.5=居中=旧行为，逐字节不变；配了才偏移，如 anime:0.15 保头）。"""
        return float(self._crop_v_anchor.get(str(domain), 0.5))

    @staticmethod
    def _square_only_compensation(domain_mix_stage: dict, square_prob: float, square_only) -> float:
        """护栏2(square_only) 补偿系数 alpha —— 方图 batch 抽 square-only 子池的概率（推导见 loader 注释）。

        令 S=square_only, M_S=Σ_{d∈S}mix_d, p=P(square 桶)。约束「每域全局期望 Σ_B P(B)·P(d|B)=mix_d」
        且「d∈S 时 P(d|非方)=0」 → 方图 batch 需以 alpha=M_S/p 的概率仅从 S 域采样（池内重归一 mix_d/M_S），
        使 P(d∈S|方)=alpha·mix_d/M_S=mix_d/p → Σ_B 复原 p·(mix_d/p)=mix_d。
        物理可行要求 alpha≤1 ⟺ p≥M_S（方图 batch 份额装得下 S 的 mix），否则硬报错（不静默）。
        返回 alpha∈[0,1]；square_only 空 → 0（旧行为逐字节不变）。"""
        S = set(square_only or [])
        if not S:
            return 0.0
        m_s = sum(float(domain_mix_stage.get(d, 0.0)) for d in S)
        p_sq = float(square_prob)
        if m_s > p_sq + 1e-9:                              # 物理不可行: 方图份额装不下这些 square-only 域的 mix
            raise ValueError(
                f"square_only_domains 物理不可行: Σmix(square_only)={m_s:.4f} > P(方图桶)={p_sq:.4f}"
                f"（方图 batch 份额装不下这些域的 mix; square_only={sorted(S)}）。"
                "增大 bucket_probs 的方图权重, 或从 square_only_domains 移出部分域。")
        return (m_s / p_sq) if p_sq > 0 else 0.0

    @staticmethod
    def _build_native_affinity(domain_mix_stage: dict, native_raw: dict, n_buckets: int,
                               square_idx: int, square_only):
        """Tier2(MODE-A) native-affinity 域采样的纯数学构建（零 I/O, 单测直击）。

        输入: mix_d(该 stage domain_mix, 只取 >0), native_raw={d: [n_buckets 浮点]}(实测 P(bucket|domain)),
              square_only 域集合（其 native 分布被覆写成 one-hot(square)——square_only 硬约束吸收进同一套
              数学, 与 MODE-B alpha 互斥、绝不叠加）。
        推导: 联合分布按「先桶后域」分解 P(d,B)=P(B)·P(d|B)，要求边际 Σ_B P(d,B)=mix_d。
              取 P(d,B)=mix_d·P_native(B|d)（每域把自己的 mix 按自家 native 桶分布铺开）→
                ① P(B)   = Σ_d mix_d·P_native(B|d)               （调度先验, 替代 config bucket_probs）
                ② P(d|B) = mix_d·P_native(B|d) / P(B)            （批内域采样表）
              恒等式 Σ_B P(B)·P(d|B) = Σ_B mix_d·P_native(B|d) = mix_d·1 = mix_d 【精确】——
              前提是调度 probs 恰用①（若仍用 config bucket_probs 则只近似, 故派生值为准）。
              square_only 域 one-hot 后 P_native(非方|d)=0 → P(d∈S|非方)=0（不进非方桶）且
              P(B=square) ≥ Σ_{d∈S} mix_d（可行性结构性成立, 无需 MODE-B 的硬校验）。
              域内取哪条记录不变（B_t-native 子池优先, 缺才 re-resize）——affinity 使被抽中域大概率有
              native 供给, 结构性 re-resize（如 anime 30-40%）随之下降。
        校验: 每个 mix>0 的域必须在 native_raw 有行（缺→硬报错）; 行长=n_buckets、全≥0、和>0（行归一）。
        返回: (rows{d: np.array 行归一}, P_B np.array[n_buckets], pdb {b: (doms tuple, probs np.array)})。
              mix 有 0 域被剔除时 Σmix<1 → P_B 和=Σmix、P(d|B) 仍列归一, 恒等式给出 mix_d/Σmix
              = sampler 一贯的「参与域重归一」语义。"""
        mix = {str(d): float(w) for d, w in domain_mix_stage.items() if float(w) > 0}
        S = {str(d) for d in (square_only or [])}
        rows: dict[str, np.ndarray] = {}
        for d in mix:
            if d in S:                                     # square_only: one-hot(square) 覆写（不看 native_raw, 可缺行）
                oh = np.zeros(n_buckets, dtype=np.float64); oh[square_idx] = 1.0
                rows[d] = oh
                continue
            if d not in native_raw:
                raise ValueError(f"multi_ar.native_bucket_dist 缺少 mix>0 的域 '{d}'（512a 参与域都必须给实测行, "
                                 f"已给={sorted(native_raw)}）——不静默兜底, 请补 bucketdist 实测值。")
            v = np.asarray([float(x) for x in native_raw[d]], dtype=np.float64)
            if v.shape != (n_buckets,) or (v < 0).any() or v.sum() <= 0:
                raise ValueError(f"multi_ar.native_bucket_dist['{d}'] 非法: 需 {n_buckets} 个非负数且和>0, got {native_raw[d]}")
            rows[d] = v / v.sum()
        doms = tuple(mix)                                   # 保 config 键序（dict 保序）
        P_B = np.zeros(n_buckets, dtype=np.float64)
        for d in doms:
            P_B += mix[d] * rows[d]
        pdb = {}
        for b in range(n_buckets):
            if P_B[b] > 0:
                probs = np.array([mix[d] * rows[d][b] / P_B[b] for d in doms], dtype=np.float64)
            else:                                           # P(B)=0 桶: 调度永不选（防御除0; loader 兜底退全池）
                probs = np.zeros(len(doms), dtype=np.float64)
            pdb[b] = (doms, probs)
        return rows, P_B, pdb

    @staticmethod
    def _prune_scheduled_buckets(domain_mix_stage: dict, rows: dict, P_B, buckets,
                                 square_idx: int, floor: float):
        """调度剪枝（纯数学,零 I/O,单测直击）：P(B)<floor 的桶从调度表移除。

        做法：每个域的 native 行把被剪桶的质量搬到最近保留桶（nearest_allowed_bucket:
        主序 |Δlog-AR| 次序 |Δ面积|）→ 重derive ① P(B)=Σ mix·P_native ② P(d|B)。
        质量只在桶间搬运 → Σ_B P(B)·P(d|B)=mix_d 恒等式在剪枝后依然精确。
        square 桶恒保留。返回 (rows2, P_B2, pdb2, kept frozenset)。"""
        mix = {str(d): float(w) for d, w in domain_mix_stage.items() if float(w) > 0}
        n = len(buckets)
        kept = {b for b in range(n) if float(P_B[b]) >= floor}
        kept.add(int(square_idx))
        rows2 = {}
        for d, v in rows.items():
            v2 = np.asarray(v, dtype=np.float64).copy()
            for b in range(n):
                if v2[b] > 0 and b not in kept:
                    tgt = nearest_allowed_bucket(b, buckets, kept)
                    v2[tgt] += v2[b]
                    v2[b] = 0.0
            rows2[d] = v2
        doms = tuple(d for d in mix if d in rows2)
        P_B2 = np.zeros(n, dtype=np.float64)
        for d in doms:
            P_B2 += mix[d] * rows2[d]
        pdb2 = {}
        for b in range(n):
            if P_B2[b] > 0:
                probs = np.array([mix[d] * rows2[d][b] / P_B2[b] for d in doms], dtype=np.float64)
            else:
                probs = np.zeros(len(doms), dtype=np.float64)
            pdb2[b] = (doms, probs)
        return rows2, P_B2, pdb2, frozenset(kept)

    def _affinity_present_dist(self, b_idx: int, pools):
        """MODE-A: 当前桶 B_t 的域采样表, 限制到 present(pools 有货)域（缺席域按 sampler 一贯语义重归一——
        cum[-1] 缩放隐式完成）。全缺/全零 → (None, None)=退全池旧路径（会 re-resize 兜底）。"""
        doms, probs = self._mar_pdb[b_idx]
        keep = [(d, p) for d, p in zip(doms, probs) if p > 0 and d in pools]
        if not keep:
            return None, None
        return [d for d, _ in keep], np.cumsum([p for _, p in keep])

    def _square_only_pools(self, pools, pools_R, pools_S, b_is_square, rng):
        """护栏2 MODE-B(alpha) 接线（仅无 native_bucket_dist 时用; 有则被 MODE-A one-hot 吸收, 互斥）:
        依 B_t 方/非方返回【本次单样本域采样】要用的 pool 子集（补偿推导见 loader 注释）。
        方 B_t : 以 alpha 概率用 square-only 子池(pools_S)、否则用非-S 子池(pools_R);
        非方 B_t: 只用非-S 子池(pools_R)（square-only 域不参与该 batch → 永不 re-resize 进非方桶）。
        空池优雅降级(缺 S→全走 R; 缺 R→全走 S; 都缺→全池 pools), 对齐 sampler 对 present 域重归一的兜底语义。"""
        has_R = bool(pools_R); has_S = bool(pools_S)
        if b_is_square:
            if has_S and (not has_R or rng.random() < self._mar_square_only_alpha):
                return pools_S
            return pools_R if has_R else (pools_S or pools)
        return pools_R if has_R else (pools_S or pools)   # 非方: 只非-S; 退化(仅剩 S present)→全池(会 re-resize)

    def _mar_pick_bucket(self, batch_count: int):
        """multi-AR：rank 一致选 B_t（warmup 期强制 square）。返回 (bucket_idx, (H,W))。
        ★base_seed 不掺 rank → 所有 rank 同 gstep 选同桶 → FSDP shape 对齐（collective-free）。
        过 warmup 翻 _multi_ar_active → producer 改出 native 桶（软切换；过渡窗口靠 re-resize 兜底）。"""
        # -1：batch_count 在调用前已自增(本批=第 batch_count 个)；train.py 在 step 后才 global_step++，
        # 故消费于 global_step=G 的批 batch_count=G+1 → gstep 精确等于 global_step（warmup 覆盖恰好 warmup_steps 步）。
        gstep = self._mar_start_step + int(batch_count) - 1
        if not self._multi_ar_active and gstep >= self._mar_warmup_steps:
            self._multi_ar_active = True
        bidx = self._mar_schedule.pick(gstep) if self._multi_ar_active else self._mar_square_idx
        return bidx, self._mar_buckets[bidx]

    @staticmethod
    def _is_cfg_protected(domain, rec) -> bool:
        """该样本是否【永不 CFG dropout】：config 规定 anime tag-only 行永不 drop。
        tag-only = anime 且无任何 NL 描述（refined/medium/short）。其余（含 anime NL、real/synth）
        交给模型按 cfg_dropout_prob 正常 dropout。"""
        if domain != "anime":
            return False
        f = rec.get("cap_fields", {})
        return not (f.get("refined_description") or f.get("medium_description") or f.get("short_description"))

    def _finalize_caption(self, domain, rec, band, rng):
        cap = self._build_caption(domain, rec, band, rng)
        if self.tokenizer is not None:                    # F3：真实 token 截断（消除模型端硬截）
            cap = truncate_to_token_budget(cap, self.tokenizer, self.caption_budget)
        return cap

    def _build_caption(self, domain, rec, band, rng):
        if domain == "anime":
            return build_anime_caption(rec["cap_fields"], self.stage, self.cfg.anime_caption,
                                       rng, rec.get("general_freq"))
        if domain == "textatlas":                          # 第 5 域：i1 caption1..5 随机选 1（非空）
            return build_textatlas_caption(rec["captions"], rng)
        if domain == "synthetic":
            code = CAPTIONER_CODE["original"]
            return finalize_real_caption(rec["cap_texts"].get(code), "original", self.cfg.caption_truncate_tok)
        # monet_real / gpic：按抽中档选 captioner
        code = rec["band_captioner"][band] if band >= 0 else 0
        if code == 0:                                     # 兜底：取任一非零档
            nz = [c for c in rec["band_captioner"] if c > 0]
            code = nz[0] if nz else CAPTIONER_CODE["original"]
        name = CODE_TO_CAPTIONER.get(code, "original")
        return finalize_real_caption(rec["cap_texts"].get(code), name, self.cfg.caption_truncate_tok)

    def _force_square_only_record(self, domain, rec):
        """item_queue hard guard: square-only domains are physically stored only in square bucket."""
        if (not self.multi_ar_enabled or domain not in self._mar_square_only or
                rec.get("bucket") == self._mar_square_idx or "u8" not in rec):
            return rec
        H, W = self._mar_buckets[self._mar_square_idx]
        rec = dict(rec)
        rec["u8"] = resize_u8_to_bucket(rec["u8"], H, W, self._v_anchor_for(domain))
        rec["bucket"] = self._mar_square_idx
        return rec

    def _loader_item_queue(self, bs: int):
        """NAI-like item-level bucket assignment + per-bucket queues（2026-07-09 软计划重构）。

        调度不变：B_t 由确定性 BucketSchedule 决定（同 seed 同 step → 跨 rank 同 shape,零 collective）。
        组批从「硬 multinomial 计划 + 全 native 前置 + 超时杀 rank」改为【软计划】三相：
          Phase1 ≤bucket_wait_native_s   等 native 计划齐 → 全 native 组批（零二次裁剪）。
                                          ⚠️ 生产配 0：_topup 被 buffer_size 封顶,补满后等待不再增
                                          库存 = 纯烧墙钟（首个 GPU smoke 实测 dw≈30s/步,85%+ 迭代
                                          耗在此;native 纯度由 buffer_size 决定,不由等待决定）;
          Phase2 ≤bucket_wait_timeout_s  等每个计划域【总量】(任意桶)够 → 缺 native 的槽位用该域
                                          AR 最近桶记录 re-resize 到 B_t（域配比不变,裁剪≈相邻桶级,
                                          损失记 buf_reresize_*）;
          Phase3 到点                     计划封顶到实际可得,缺口按 P(d|B_t) 在 present 域重归一
                                          （与主 loader「sampler 对 present 域重归一」同一语义,
                                          批数记 item_queue_degraded）。
        只有【所有域全空】才走 empty_rounds fail-fast（真断供）。旧版把 multinomial 当硬前置：
        需求 ≈bs·P(d|B) 结构性大于 buffer 内 ~buffer_size·P_native(B|d) 的可得量 → 稀有桶必超时
        raise（实测 lxsr 16/16 rank 死于首个调度桶）= 2×8 smoke 的真 blocker,故重构。
        batch_count 只在成功 yield 后自增（修 continue 路径消耗 schedule 步号 → FSDP shape desync）。
        """
        import torch
        if self._mar_bucket_buffer_size < int(bs):
            raise ValueError(f"item_queue bucket_buffer_size={self._mar_bucket_buffer_size} < batch_size={bs}")
        if self._mar_bucket_wait_timeout_s <= 0:
            raise ValueError("item_queue requires multi_ar.bucket_wait_timeout_s > 0")
        if self._mar_start_step >= self._mar_warmup_steps:
            self._multi_ar_active = True
        stop_evt = threading.Event()
        queues = {d: queue.Queue(self.queue_size) for d in self.domains}
        producers = []
        active = set()
        dm = self.cfg.domain_mix[self.stage]
        for d in self.domains:
            if not self.shards[d] or dm.get(d, 0.0) <= 0:
                continue
            p = _Producer(d, self.shards[d], self, queues[d],
                          stop_evt, self.seed + _domain_seed(d) + self.rank * 9973)
            p.start()
            producers.append(p)
            active.add(d)

        bucket_buffers = {
            b: {d: deque(maxlen=self._mar_bucket_buffer_size) for d in self.domains}
            for b in range(len(self._mar_buckets))
        }
        sampler = CompleteSampler(self.cfg, self.stage, seed=self.seed + self.rank * 100003)
        cap_rng = random.Random(self.seed + self.rank * 31 + 7)
        pool_rng = np.random.default_rng(self.seed + self.rank * 100003 + 777)
        n_buckets = len(self._mar_buckets)

        def _domain_total(d):
            return sum(len(bucket_buffers[b][d]) for b in bucket_buffers)

        def _total_all():
            return sum(_domain_total(d) for d in self.domains)

        def _pool_len(pool):
            return int(getattr(pool, "n", 0))

        def _pools_total(pools):
            return sum(_pool_len(p) for p in pools.values())

        def _pool_domain_len(pools, domain):
            p = pools.get(domain)
            return _pool_len(p) if p is not None else 0

        def _sample_affinity_plan(bidx, n):
            if self._mar_pdb is None:
                return None
            doms, probs = self._mar_pdb[bidx]
            keep = [(d, float(p)) for d, p in zip(doms, probs) if p > 0]
            if not keep:
                return None
            kd, kp = zip(*keep)
            w = np.asarray(kp, dtype=np.float64)
            w = w / max(float(w.sum()), 1e-12)
            counts = pool_rng.multinomial(int(n), w)
            return {d: int(c) for d, c in zip(kd, counts) if int(c) > 0}

        def _plan_native_ready(pools, plan):
            if plan is None:
                return _pools_total(pools) >= self._mar_effective_bucket_min_fill
            return all(_pool_domain_len(pools, d) >= c for d, c in plan.items())

        def _plan_soft_ready(plan):
            if plan is None:
                return _total_all() >= int(bs)
            return all(_domain_total(d) >= c for d, c in plan.items())

        def _ready_by_domain(pools):
            return {d: _pool_domain_len(pools, d) for d in self.domains}

        def _put_record(d, rec):
            rec = self._force_square_only_record(d, rec)     # square-only 硬护栏(producer 已 force,此处双保险)
            b = int(rec.get("bucket", self._mar_square_idx))
            if b < 0 or b >= n_buckets:
                b = self._mar_square_idx
            bucket_buffers[b][d].append(rec)

        def _topup(deadline=None, domain_fill_limit=None):
            """把各域 queue 记录搬进 bucket_buffers,每域封顶 fill_limit（修旧版非计划域无界增长）。"""
            made = False
            fill_limit = self.buffer_size if domain_fill_limit is None else max(1, int(domain_fill_limit))
            for d in self.domains:
                if d not in active:
                    continue
                while _domain_total(d) < fill_limit and not stop_evt.is_set():
                    try:
                        if deadline is not None:
                            remain = deadline - time.monotonic()
                            if remain <= 0:
                                break
                            rec = queues[d].get(timeout=min(0.2, max(0.01, remain)))
                        else:
                            rec = queues[d].get_nowait()
                    except queue.Empty:
                        break
                    _put_record(d, rec)
                    made = True
            return made

        def _bucket_pools(bidx):
            return self._build_pools(bucket_buffers[bidx], pool_rng)

        _ar_order_cache: dict = {}

        def _buckets_by_ar(bidx):
            """到 bidx 的 AR 距离升序桶号（re-resize 兜底取样序;首元素=bidx 自身,惰性缓存）。"""
            order = _ar_order_cache.get(bidx)
            if order is None:
                order = buckets_by_ar_distance(bidx, self._mar_buckets)
                _ar_order_cache[bidx] = order
            return order

        def _remove_at(d, b, i):
            del bucket_buffers[b][d][i]                      # deque.__delitem__,免整队重建（review D2）

        try:
            batch_count = 0                                  # = 已成功 yield 的批数（只在 yield 后自增）
            empty_rounds = 0
            first = True
            self._mar_effective_bucket_min_fill = max(int(self._mar_bucket_min_fill), int(bs))
            while True:
                if first:
                    # 首批填充：语义对齐主 loader（first_fill_timeout_s;旧版误用 bucket_wait_timeout_s）
                    first_deadline = (time.monotonic() + self.first_fill_timeout_s
                                      if self.first_fill_timeout_s > 0 else None)
                    while (not stop_evt.is_set()
                           and any(_domain_total(d) < self.min_fill for d in active)):
                        progressed = _topup(deadline=first_deadline, domain_fill_limit=self.min_fill)
                        if first_deadline is not None and time.monotonic() >= first_deadline:
                            break
                        if not progressed:
                            time.sleep(0.05)
                    first = False
                # 桶调度：prospective 步号 = batch_count+1（成功 yield 后才消费;retry 同一步号 → 同桶）
                b_idx, (b_h, b_w) = self._mar_pick_bucket(batch_count + 1)
                affinity_plan = _sample_affinity_plan(b_idx, bs)
                t0 = time.monotonic()
                native_deadline = t0 + max(0.0, self._mar_bucket_wait_native_s)
                hard_deadline = t0 + self._mar_bucket_wait_timeout_s
                # Phase1：等全 native 计划（有界;满足即全 native 组批,零二次裁剪）
                while (not stop_evt.is_set() and time.monotonic() < native_deadline
                       and not _plan_native_ready(_bucket_pools(b_idx), affinity_plan)):
                    if not _topup(deadline=min(native_deadline, time.monotonic() + 0.2)):
                        time.sleep(0.05)
                # Phase2：等软就绪（计划域【总量】够 → re-resize 可组满,域配比不变）
                while (not stop_evt.is_set() and time.monotonic() < hard_deadline
                       and not _plan_native_ready(_bucket_pools(b_idx), affinity_plan)
                       and not _plan_soft_ready(affinity_plan)):
                    if not _topup(deadline=min(hard_deadline, time.monotonic() + 0.2)):
                        time.sleep(0.05)
                # Phase3：到点降级——计划封顶到实际可得;缺口由组批时 P(d|B_t) present 重归一吸收
                degraded = False
                if affinity_plan and not _plan_soft_ready(affinity_plan):
                    affinity_plan = {d: min(c, _domain_total(d)) for d, c in affinity_plan.items()}
                    affinity_plan = {d: c for d, c in affinity_plan.items() if c > 0}
                    degraded = True
                if _total_all() == 0:
                    empty_rounds += 1
                    if empty_rounds > 5:
                        raise RuntimeError(
                            f"[dataline] item_queue 连续 {empty_rounds} 轮所有域 buffer 全空"
                            f"（active={sorted(active)}）——producer 卡死或数据源不可达（S3/R2/tar）。")
                    _topup(deadline=time.monotonic() + 5.0)
                    continue                                  # 未 yield → 不消费 schedule 步号
                empty_rounds = 0
                if degraded:
                    self._mar_plan_degraded["batches"] += 1

                # ---- 监控（键对齐旧版 + 新增 wait_s/degraded/reresize_total）----
                self.buffer_fill = {d: _domain_total(d) / max(self.buffer_size, 1) for d in self.domains}
                _mon_pools = _bucket_pools(b_idx)             # 复用同一次构建（review D4）
                ready_by_domain = _ready_by_domain(_mon_pools)
                for d in self.domains:
                    sm = self._mar_sampled_by_domain.get(d, 0)
                    self.buffer_fill[f"reresize_{d}"] = (self._mar_reresize_by_domain.get(d, 0) / sm) if sm else 0.0
                    self.buffer_fill[f"zerowpool_{d}"] = int(self._mar_zerow_pool_skips.get(d, 0))
                    self.buffer_fill[f"zerowdrop_{d}"] = int(self.zerow_dropped.get(d, 0))
                    self.buffer_fill[f"item_queue_ready_{d}"] = int(ready_by_domain.get(d, 0))
                    self.buffer_fill[f"item_queue_plan_{d}"] = int((affinity_plan or {}).get(d, 0))
                self.buffer_fill["item_queue_bucket"] = int(b_idx)
                self.buffer_fill["item_queue_bucket_h"] = int(b_h)
                self.buffer_fill["item_queue_bucket_w"] = int(b_w)
                self.buffer_fill["item_queue_bucket_ready"] = _pools_total(_mon_pools)
                self.buffer_fill["item_queue_wait_s"] = round(time.monotonic() - t0, 3)
                self.buffer_fill["item_queue_degraded"] = int(self._mar_plan_degraded.get("batches", 0))
                self.buffer_fill["item_queue_reresize_total"] = int(self._mar_reresize_count)
                self.buffer_fill["extreme_clamped"] = int(self._mar_extreme_clamped)

                # ---- 组批（软计划;槽位级不 raise,唯组批中途全域断供且超时才 fail-fast）----
                prompts, imgs, cfg_keep = [], [], []
                remaining_plan = Counter(affinity_plan or {})
                # 组批限时 = 相位 hard_deadline + 30s 宽限（review B1/E3：不再另起一份完整
                # bucket_wait_timeout_s,最坏总等待有界 ≈ timeout+30s,保住 JCC 干净失败信号）。
                assembly_deadline = hard_deadline + 30.0
                banned = set()          # 本批出局域（零权重池/空合并池;review A2 修 busy-loop）
                b_is_square = (b_idx == self._mar_square_idx)
                while len(imgs) < bs:
                    if stop_evt.is_set():
                        return
                    # 1) 选域：计划权重（仅计 present）→ P(d|B_t) present 重归一 → present 均匀兜底
                    dpick = None
                    if remaining_plan:
                        elig = [d for d, c in remaining_plan.items()
                                if c > 0 and d not in banned and _domain_total(d) > 0]
                        if elig:
                            w = np.asarray([remaining_plan[d] for d in elig], dtype=np.float64)
                            cw = np.cumsum(w)
                            dpick = elig[min(int(np.searchsorted(cw, pool_rng.random() * cw[-1], side="left")),
                                             len(elig) - 1)]   # min()：ulp 越界防御（同 BucketSchedule.pick）
                    if dpick is None:
                        present = [d for d in self.domains if d not in banned and _domain_total(d) > 0]
                        # square-only 硬不变量（review A1）：非方 B_t 的均匀兜底也不得抽 square-only 域
                        # （其记录物理上全是方图,抽中必被 re-resize 破坏性裁剪;与主 loader 的
                        # aff_fallback「无条件剔 S 域,剔空才退全池保活性」同一语义）。
                        if not b_is_square and self._mar_square_only:
                            present_ns = [d for d in present if d not in self._mar_square_only]
                            present = present_ns or present
                        if not present:
                            if time.monotonic() >= assembly_deadline:
                                raise RuntimeError(
                                    f"[dataline] item_queue bucket={b_idx} 组批中途无可用域且等待超限"
                                    f"（已组 {len(imgs)}/{bs}, banned={sorted(banned)}）。")
                            _topup(deadline=time.monotonic() + 1.0)
                            banned.clear()                    # 补料后解禁重评（producer 可能已供新的非零权重记录）
                            continue
                        keep_dp = []
                        if self._mar_pdb is not None:
                            doms, probs = self._mar_pdb[b_idx]
                            keep_dp = [(d, float(p)) for d, p in zip(doms, probs) if p > 0 and d in present]
                        if keep_dp:
                            kd = [d for d, _ in keep_dp]
                            cum = np.cumsum([p for _, p in keep_dp])
                            dpick = kd[min(int(np.searchsorted(cum, pool_rng.random() * cum[-1], side="left")),
                                           len(kd) - 1)]       # min()：ulp 越界防御
                        else:
                            dpick = present[int(pool_rng.integers(len(present)))]
                    # 2) 取记录：B_t-native 子池优先（零裁剪）;缺 → 该域全桶合并池 + re-resize 兜底
                    native_pools = self._build_pools({dpick: bucket_buffers[b_idx][dpick]}, pool_rng) \
                        if bucket_buffers[b_idx][dpick] else {}   # 只建选中域（review D1,免 5 域全建）
                    if dpick in native_pools:
                        domain, idx, band = sampler.sample_one({dpick: native_pools[dpick]})
                        rec = bucket_buffers[b_idx][dpick][idx]
                        _remove_at(dpick, b_idx, idx)
                    else:
                        # AR 最近桶优先（替代旧「全桶合并池」：合并池 AR-blind,可能抽 9:16 硬裁成
                        # 16:9 = 本项目要消灭的破坏性裁剪;逐桶按 |Δlog-AR| 试,裁剪幅度≈相邻桶级）
                        rec = None
                        for b_src in _buckets_by_ar(b_idx):
                            q = bucket_buffers[b_src][dpick]
                            if not q:
                                continue
                            mpool = self._build_pools({dpick: q}, pool_rng)
                            if dpick not in mpool:           # 该桶全零权重 → 试更远桶
                                continue
                            domain, idx, band = sampler.sample_one(mpool)
                            rec = q[idx]
                            _remove_at(dpick, b_src, idx)
                            break
                        if rec is None:                      # 全桶空/全零权重 → 本批禁选该域（防 busy-loop）
                            banned.add(dpick)
                            remaining_plan.pop(dpick, None)
                            continue
                    u8 = rec["u8"]
                    if u8.shape[0] != b_h or u8.shape[1] != b_w:
                        self._mar_reresize_count += 1
                        self._mar_reresize_by_domain[domain] += 1
                        u8 = resize_u8_to_bucket(u8, b_h, b_w, self._v_anchor_for(domain))
                    self._mar_sampled_by_domain[domain] += 1
                    prompts.append(self._finalize_caption(domain, rec, band, cap_rng))
                    imgs.append(uint8_to_norm_tensor(u8))
                    cfg_keep.append(self._is_cfg_protected(domain, rec))
                    self.domain_counter[domain] += 1
                    if remaining_plan.get(domain, 0) > 0:
                        remaining_plan[domain] -= 1
                        if remaining_plan[domain] <= 0:
                            del remaining_plan[domain]
                    if domain == "anime":
                        self.cluster_counter[rec.get("cluster", -1)] += 1
                batch_count += 1                              # 成功组满一批才消费 schedule 步号
                yield {"prompts": prompts, "pixels": torch.stack(imgs, 0), "cfg_keep": cfg_keep}
                _topup()
        finally:
            stop_evt.set()
            for p in producers:
                p.join(timeout=5.0)

    def loader(self, batch_size: int):
        import torch
        bs = int(batch_size)
        # multi-AR warmup 曝光化（gated；Path C 前置 (c)）：warmup_exposure_images>0 →
        #   warmup_steps = ceil(曝光 / global_batch)，global_batch = bs×world_size（本 trainer 无梯度累积，
        #   一次 optimizer step = 一个 global batch，对齐 checkpoint.derive_max_steps）。使 warmup 覆盖固定曝光量
        #   而非固定步数（max_steps 随实际 global batch 浮动时 warmup 不再错位）。=0/缺省 → 沿用 config.warmup_steps（字节级零变化）。
        if self.multi_ar_enabled and self._mar_warmup_exposure > 0:
            gbs = max(1, bs * int(self.world_size))
            self._mar_warmup_steps = -(-self._mar_warmup_exposure // gbs)   # ceil
            if self.rank == 0:
                print(f"[dataline] multi-AR warmup_steps={self._mar_warmup_steps} ← 曝光 {self._mar_warmup_exposure:,} / "
                      f"global_batch {gbs}（bs {bs}×world {self.world_size}；config.warmup_steps 仅作参考）", flush=True)
        # multi-AR resume：warmup 全局步由 train.py 的 dataset.set_start_step(global_step) 注入（唯一路径）。
        if self.multi_ar_enabled and self._mar_assignment_mode == "item_queue":
            yield from self._loader_item_queue(bs)
            return
        stop_evt = threading.Event()
        queues = {d: queue.Queue(self.queue_size) for d in self.domains}
        producers = []
        active = set()                                    # 实际起了 producer 的域（无 producer 的域别去阻塞等空 queue）
        dm = self.cfg.domain_mix[self.stage]
        for d in self.domains:
            if not self.shards[d] or dm.get(d, 0.0) <= 0:   # 跳过本 stage 不参与的域（如 512/1024 的 gpic）
                continue
            p = _Producer(d, self.shards[d], self, queues[d],
                          stop_evt, self.seed + _domain_seed(d) + self.rank * 9973)
            p.start()
            producers.append(p)
            active.add(d)
        buffers = {d: deque(maxlen=self.buffer_size) for d in self.domains}
        sampler = CompleteSampler(self.cfg, self.stage, seed=self.seed + self.rank * 100003)
        cap_rng = random.Random(self.seed + self.rank * 31 + 7)
        pool_rng = np.random.default_rng(self.seed + self.rank * 100003 + 777)  # anime integer-repeat 分配用
        mar_rng = np.random.default_rng(self.seed + self.rank * 100003 + 555)   # multi-AR 域内 B_t-native 记录抽样(rank 无关性仅需 shape 一致, 记录可 per-rank)

        def _topup(block_min, deadline=None):
            for d in self.domains:
                q = queues[d]
                # 首轮按需阻塞填到 min_fill；之后非阻塞补满。
                # ⚠️ 只对【有 producer 的域】阻塞——本 stage 不参与的域（如 512/1024 的 GPIC，
                #    domain_mix=0 没起 producer）queue 永远空，去阻塞会白等满 30s 超时（审计 F5）。
                # deadline（看门狗）：到点即停止死等慢域，带已就绪样本先开跑；慢域随后非阻塞回填。
                if block_min and d in active and len(buffers[d]) < self.min_fill:
                    while len(buffers[d]) < self.min_fill and not stop_evt.is_set():
                        if deadline is not None and time.monotonic() >= deadline:
                            break
                        try:
                            buffers[d].append(q.get(timeout=30))
                        except queue.Empty:
                            break
                while len(buffers[d]) < self.buffer_size:
                    try:
                        buffers[d].append(q.get_nowait())
                    except queue.Empty:
                        break

        def _wait_for_any_record(timeout_s=30.0):
            """Smoke/诊断用：首批只等任一 active 域有样本，避免某个慢 shard 卡住所有 rank。

            正式默认仍是 all_min；any_pool 会让 sampler 对 present 域重归一，可能在数据源慢时
            临时改变早期 domain_mix，因此只应由 job 显式覆盖后用于诊断/短跑。
            """
            if not active:
                return False
            deadline = threading.Event()
            # 用 Event.wait 做轻量 sleep，避免额外 import time。
            elapsed = 0.0
            step = 0.05
            while elapsed < timeout_s and not stop_evt.is_set():
                got = False
                for d in self.domains:
                    if d not in active:
                        continue
                    try:
                        buffers[d].append(queues[d].get_nowait())
                        got = True
                    except queue.Empty:
                        continue
                if got:
                    return True
                deadline.wait(step)
                elapsed += step
            return False

        try:
            first = True
            empty_rounds = 0                               # F3：连续无 pool 轮数（防 busy-spin / 早发现死 producer）
            batch_count = 0
            underfill_warned = set()                       # F3：已告警的持续低水位域（恢复后清除）
            while True:
                fill_deadline = (time.monotonic() + self.first_fill_timeout_s
                                 if (first and self.first_fill_timeout_s > 0) else None)
                _topup(block_min=first and self.first_fill_policy == "all_min", deadline=fill_deadline)
                if first:
                    under = {d: len(buffers[d]) for d in active if len(buffers[d]) < self.min_fill}
                    if under:
                        print(f"[dataline] 首批 ≤{self.first_fill_timeout_s:.0f}s 内未填满 {under}"
                              f"（min_fill={self.min_fill}）→ 带已就绪域先开跑、慢域随后回填"
                              "（sampler 对 present 域重归一）。", flush=True)
                first = False
                if self.drain_mode == "keep_present":
                    # F2(可选 fallback；生产默认用 proportional)：目标域(domain_mix>0)buffer 空 →
                    # 阻塞补料填到 min_fill，防被 sampler renorm-away 把份额摊给快域。⚠️ 单批【共享】一个超时
                    # (kp_dl 在循环外)——避免按域累加致最坏 N×timeout 拖速(对抗 review 中项)。慢域到点放弃→退回 renorm。
                    kp_dl = time.monotonic() + self.keep_present_timeout_s
                    for d in active:
                        if dm.get(d, 0.0) > 0 and not buffers[d]:
                            while (len(buffers[d]) < self.min_fill and not stop_evt.is_set()
                                   and time.monotonic() < kp_dl):
                                try:
                                    buffers[d].append(queues[d].get(timeout=0.2))
                                except queue.Empty:
                                    break
                pools = self._build_pools(buffers, pool_rng)
                if not pools:                              # 所有 active 域 buffer 全空
                    empty_rounds += 1
                    if empty_rounds > 5:                   # 阻塞补料仍连续空 → producers 卡死/数据源不可达
                        raise RuntimeError(
                            f"[dataline] 连续 {empty_rounds} 轮所有域 buffer 全空（active={sorted(active)}）——"
                            "producer 卡死或数据源不可达（S3/R2/tar）。检查挂载/网络/凭证。")
                    if self.first_fill_policy == "any_pool":
                        _wait_for_any_record()
                        _topup(block_min=False)
                    else:
                        # 阻塞补料但带总超时（看门狗）：避免某域断供时在此死等/慢爬；到点回上层，
                        # 连续空轮由 empty_rounds 兜底（>5 → RuntimeError → 退出 → maxRetries resume）。
                        _topup(block_min=True,
                               deadline=(time.monotonic() + self.first_fill_timeout_s
                                         if self.first_fill_timeout_s > 0 else None))
                    continue
                empty_rounds = 0
                batch_count += 1
                # 监控：每批刷新各域 buffer 占比，供 train.py 写入 metrics.jsonl（直观看供数瓶颈）
                self.buffer_fill = {d: len(buffers[d]) / max(self.buffer_size, 1) for d in self.domains}
                # Tier1 可观测性(2026-07-05): per-domain re-resize【累计】率注入 buffer_fill dict → train.py
                #   _dataline_metrics 原样透传成 metrics.jsonl 的 buf_reresize_<domain>（零改 train.py; 口径=累计,
                #   对齐 mix_* 的 domain_counter 累计比）。阈值语义见 512a config 注释（anime>10%/任一域>20% ⇒ 查配置）。
                #   multi_ar 关闭时计数恒 0 → 不注入键, 旧 metrics 逐字节不变。
                if self.multi_ar_enabled:
                    for d in self.domains:
                        sm = self._mar_sampled_by_domain.get(d, 0)
                        if sm:
                            self.buffer_fill[f"reresize_{d}"] = self._mar_reresize_by_domain.get(d, 0) / sm
                        # 零权重修复(2026-07-06)观测: zerowpool=护栏触发累计次数(偶发瞬时=正常自愈),
                        #   zerowdrop=producer canary 累计(根修后应恒0)。metrics 键 buf_zerowpool_*/buf_zerowdrop_*。
                        self.buffer_fill[f"zerowpool_{d}"] = int(self._mar_zerow_pool_skips.get(d, 0))
                        self.buffer_fill[f"zerowdrop_{d}"] = int(self.zerow_dropped.get(d, 0))
                if batch_count % 500 == 0:                 # F3：周期检查 active 域是否持续低水位（供数告警）
                    for d in active:
                        fill = len(buffers[d]) / max(self.buffer_size, 1)
                        if fill < 0.25 and d not in underfill_warned:
                            # ⚠️ 低水位≠欠采：最大域被抽得最狠也会长期低水位却仍超供。是否真欠采以
                            # metrics.jsonl 的 mix_*（domain_counter 真值）为准，勿凭此行推断 mix。
                            print(f"[dataline] ⚠️ 域 '{d}' buffer 低水位 {fill:.0%}(<25%)：producer 慢/数据源紧（注意：低水位≠欠采，"
                                  f"真实占比看 metrics 的 mix_{d}；目标 {dm.get(d, 0):.3f}，sampler 按 present 域重归一）。", flush=True)
                            underfill_warned.add(d)
                        elif fill >= 0.5 and d in underfill_warned:
                            underfill_warned.discard(d)    # 恢复后清除，允许再次告警
                # ---- multi-AR（Phase2b，★domain-first，bucket-second；Evan 2026-06-30 拍）----
                #   pools(全池)负责【按全 domain_mix 选域】——绝不被桶可用性重归一(护栏1)；
                #   mar_bt_pools(B_t-native per 域)负责【域内取 B_t 原生记录】；域内无 B_t-native → 该域记录 re-resize 到 B_t(域内兜底,
                #   不把份额转移给他域)。square_only_domains(护栏2 逃生口,默认空)的域在非方桶不 re-resize(留作实测裁剪伤害后开)。
                mar_bt_pools = None; b_idx = self._mar_square_idx; b_h = b_w = self.image_size
                if self.multi_ar_enabled:
                    b_idx, (b_h, b_w) = self._mar_pick_bucket(batch_count)
                    self._mar_last_bucket = b_idx
                    mar_bt_pools = self._build_pools(buffers, pool_rng, bucket=b_idx)  # 每域 B_t-native(可能缺某些域)
                sampled = Counter()                          # 本批各域实际被抽中数（proportional/keep_present 排空用）
                if self.use_latent_cache:
                    # 缓存模式：不取图，按 uid 从缓存读 moments（按文件分组批量读）；缺失跳过、补采到 bs。
                    prompts, moms, cfg_keep = [], [], []
                    attempts = 0
                    while len(moms) < bs and attempts < bs * 8:
                        need = bs - len(moms)
                        cand = []
                        for _ in range(need):
                            domain, idx, band = sampler.sample_one(pools)
                            rec = buffers[domain][idx]
                            uid = rec["uid"] if domain != "anime" else f"anime:{rec['key']}"
                            cand.append((domain, band, rec, uid))
                        got = self.cache_index.get_many([c[3] for c in cand])
                        for domain, band, rec, uid in cand:
                            m = got.get(uid)
                            if m is None:
                                self.cache_miss_counter += 1
                                continue
                            prompts.append(self._finalize_caption(domain, rec, band, cap_rng))
                            moms.append(m)
                            cfg_keep.append(self._is_cfg_protected(domain, rec))
                            self.domain_counter[domain] += 1
                            sampled[domain] += 1
                            if domain == "anime":
                                self.cluster_counter[rec.get("cluster", -1)] += 1
                            if len(moms) >= bs:
                                break
                        attempts += need
                    yield {"prompts": prompts, "moments": torch.stack(moms, 0), "cfg_keep": cfg_keep}
                else:
                    prompts, imgs, cfg_keep = [], [], []
                    # ---- 护栏2 MODE-B(alpha) square_only 域补偿（仅「无 native 表」时生效; warmup/未配置=旧行为逐字节不变）----
                    #   B_t 非方 → square_only 域【不参与】本 batch 域采样(只从非-S 域抽, sampler 自然重归一到 mix_d/M_rest)→ 永不 re-resize 进非方桶;
                    #   B_t 方   → 以 alpha=Σ_{d∈S}mix_d/P(square) 的概率抽 square_only 子池(池内重归一 mix_d/M_S)、否则抽非-S 子池。
                    #   ★补偿推导(S=square_only, M_S=Σ_{d∈S}mix_d, M_rest=1−M_S, p=P(square)=bucket_probs[square]):
                    #     约束 每域全局期望 Σ_B P(B)·P(d|B)=mix_d 且 d∈S 时 P(d|非方)=0。
                    #     · d∈S : P(d|方)=alpha·mix_d/M_S=mix_d/p, P(d|非方)=0 → Σ_B= p·(mix_d/p)+(1−p)·0 = mix_d ✓
                    #     · d∉S : P(d|方)=(1−alpha)·mix_d/M_rest, P(d|非方)=mix_d/M_rest
                    #             → Σ_B= p(1−alpha)mix_d/M_rest+(1−p)mix_d/M_rest = mix_d/M_rest·[p−M_S+1−p] = mix_d ✓
                    #     ⇒ 全域全局期望复原到 mix_d（与 square_only 空等价）。可行性 alpha≤1 ⟺ p≥M_S, __init__ 已硬断言。
                    #   为何只在 active 生效: warmup 期强制全方图、无非方 batch 可作补偿, square_only 须按常规 mix 正常参与(否则被 boost 过采)。
                    # ---- Tier2(MODE-A) native-affinity: 域采样 P(d|B_t)=mix_d·P_native(B_t|d)/P(B_t)（表在
                    #   init 算好=_mar_pdb; 调度 probs 已换成派生 P(B) → Σ_B P(B)·P(d|B)=mix_d 恒等式【精确】,
                    #   推导见 _build_native_affinity）。square_only 域已被 one-hot(square) 吸收（P(d∈S|非方)=0）。
                    #   与 MODE-B alpha 互斥（init 硬断言）。域采样本就 rank-local, 桶调度不动 → rank 一致性不受影响。
                    aff_doms = aff_cum = None
                    aff_fallback = None
                    so_active = False
                    pools_R = pools_S = None
                    b_is_square = (b_idx == self._mar_square_idx)
                    if self._multi_ar_active and self._mar_pdb is not None:      # MODE-A
                        aff_doms, aff_cum = self._affinity_present_dist(b_idx, pools)
                        # 退化(支撑域全缺席, 冷启/供给饿)兜底: 非方 B_t 时【无条件】剔除 square_only 域再退化——
                        # 硬不变量「S 域永不被裁进非方桶」在退化路径也成立(2026-07-05 dryrun 抓到的漏洞:
                        # 退全池会把 synthetic/textatlas re-resize 进非方桶=破坏性裁剪)。剔完为空(只剩 S present,
                        # 生产不可能)才退全池保活性, 与 MODE-B 文档化的最终降级语义一致。
                        if aff_cum is None and not b_is_square and self._mar_square_only:
                            aff_fallback = {d: p for d, p in pools.items()
                                            if d not in self._mar_square_only} or None
                    elif self._multi_ar_active and self._mar_square_only:        # MODE-B（无 native 表、只有 square_only）
                        so_active = True
                        pools_R = {d: p for d, p in pools.items() if d not in self._mar_square_only}
                        pools_S = {d: p for d, p in pools.items() if d in self._mar_square_only}
                    for _ in range(bs):
                        # ★domain-first：先选域/档——domain_mix 不被桶可用性重归一(护栏1)。
                        #   MODE-A: 按 P(d|B_t) 抽域(全局期望=mix); MODE-B: 按 B_t 方/非方选子池(见上);
                        #   未配置/warmup → 全池 mix(=旧行为)。
                        if aff_cum is not None:
                            dpick = aff_doms[int(np.searchsorted(aff_cum, mar_rng.random() * aff_cum[-1], side="left"))]
                            sel = {dpick: pools[dpick]}                # 单域池→sampler 域权重=1, 域内 band-first/priority/base_w 逻辑原样
                        elif aff_fallback is not None:
                            sel = aff_fallback                         # MODE-A 退化(非方 B_t): 剔 S 域后的 present 池
                        elif so_active:
                            sel = self._square_only_pools(pools, pools_R, pools_S, b_is_square, mar_rng)
                        else:
                            sel = pools
                        domain, idx, band = sampler.sample_one(sel)   # idx=buffer 位(选中子池的域内 arange, 与 buffers 对齐)
                        reresize = None
                        if self.multi_ar_enabled:
                            bp = mar_bt_pools.get(domain) if mar_bt_pools else None
                            native = False
                            if bp is not None:                            # 域内取 B_t-native 记录 = 无裁剪
                                if domain == "anime":                     # anime: 按 priority
                                    idx = int(bp.arrays["_buf_idx"][bp.sample_anime(mar_rng)]); native = True
                                elif band is not None and band >= 0:      # real(monet/gpic): 同档 band-first
                                    if bp.serves_band(band):
                                        idx = int(bp.arrays["_buf_idx"][bp.sample_band(band, mar_rng)]); native = True
                                else:                                     # 注入域(synthetic/textatlas, band<0, served_mask=0 不分档): 全池 base_w 抽
                                    idx = int(bp.arrays["_buf_idx"][bp.sample_weighted(mar_rng)]); native = True
                            if not native:                                # 该域(本档)无 B_t-native → 域内 re-resize 兜底(份额不转移给他域)
                                reresize = (b_h, b_w)                     # idx 保持子池抽中的该域记录(buffer 位); 护栏2(square_only)已接线于上: active 期非方 B_t 不选 square_only 域→其 re-resize 计数≈0
                                self._mar_reresize_count += 1
                                self._mar_reresize_by_domain[domain] += 1
                            self._mar_sampled_by_domain[domain] += 1
                        rec = buffers[domain][idx]
                        prompts.append(self._finalize_caption(domain, rec, band, cap_rng))
                        u8 = rec["u8"]
                        if reresize is not None:                          # 域内无 B_t-native → 裁到 B_t（保单桶同形）
                            # 竖直裁剪走该域 anchor（缺省 0.5=旧居中）→ portrait 兜到方/横桶时保头（缺头修复 Path B）
                            u8 = resize_u8_to_bucket(u8, reresize[0], reresize[1], self._v_anchor_for(domain))
                        imgs.append(uint8_to_norm_tensor(u8))
                        cfg_keep.append(self._is_cfg_protected(domain, rec))   # tag-only anime → 模型永不 drop
                        self.domain_counter[domain] += 1     # 监控用（域曝光统计）
                        sampled[domain] += 1
                        if domain == "anime":                # F2a 监控：仅 anime cluster 分布（确认构造平衡）
                            self.cluster_counter[rec.get("cluster", -1)] += 1
                    pixels = torch.stack(imgs, 0)
                    yield {"prompts": prompts, "pixels": pixels, "cfg_keep": cfg_keep}
                # 滚动 buffer：淘汰旧记录、下轮补新。
                #   uniform(旧)：每域无差别淘 bs//2 → 每域都被要求以相同高速率产出才保持 present（慢域饿死→
                #                renorm 把份额摊给快域=domain_mix 回归根因，见回归诊断 §2.1）。
                #   proportional/keep_present(F1/F2)：每域只淘【本批实际被抽中的数量】→ 每域只需产出自己那份
                #                domain_mix 速率即可保持 present；over-producer(anime)被封顶在 target、多余 FIFO
                #                淘汰=无偏丢弃。这是把"是否 present"从 producer 吞吐解耦回 domain_mix 的关键。
                if self.drain_mode in ("proportional", "keep_present"):
                    for d in self.domains:
                        for _ in range(min(sampled.get(d, 0), len(buffers[d]))):
                            buffers[d].popleft()
                else:                                          # uniform（默认/旧行为）
                    for d in self.domains:
                        for _ in range(min(bs // 2, len(buffers[d]))):
                            if buffers[d]:
                                buffers[d].popleft()
        finally:
            stop_evt.set()
            for p in producers:
                p.join(timeout=5.0)

    def _build_pools(self, buffers, pool_rng=None, bucket=None) -> dict:
        asmp = self.cfg.anime_sampling
        pools = {}
        for d, buf in buffers.items():
            if not buf:
                continue
            recs = list(buf)
            buf_idx = None
            if bucket is not None:                         # multi-AR：仅留 native==B_t 的记录；位置经 _buf_idx 映射回 buffer
                keep = [i for i, r in enumerate(recs) if r.get("bucket") == bucket]
                if not keep:                               # 本域 buffer 无 B_t native 供给 → 缺席（sampler renorm 到 present 域）
                    continue
                recs = [recs[i] for i in keep]
                buf_idx = np.array(keep, dtype=np.int64)
            if d == "anime":
                arrays = {"key": np.arange(len(recs)),
                          "priority": np.array([r["priority"] for r in recs], dtype=np.float64)}
                # 护栏(2026-07-06 零权重竞态): 全零权重池=任何加权抽取不可用 → 视为【缺席】——与上面
                #   「无 B_t native 供给 → 缺席(sampler renorm 到 present 域)」同一套既有语义。防 MODE-A
                #   单域 sel={dpick:全零池} 抽中后 _weighted_idx 全零 raise(512a 正式 step8300 实撞)。
                #   iter 层根修后 real/synth 理论不再出现; anime priority 全零同理防御。计数进 metrics。
                if float(arrays["priority"].sum()) <= 0.0:
                    self._mar_zerow_pool_skips[d] += 1
                    continue
                # anime integer-repeat 曝光（faithful naifu）：每 buffer 重算 repeat count 作权重
                if buf_idx is not None:
                    arrays["_buf_idx"] = buf_idx
                pools[d] = DomainPool(d, arrays, max_repeat=asmp.max_repeat,
                                      extra_frac=asmp.extra_frac, rng=pool_rng)
            else:
                arrays = {"uid": np.arange(len(recs)),
                          "served_mask": np.array([r["served_mask"] for r in recs], dtype=np.uint8),
                          "base_w": np.array([r["base_w"] for r in recs], dtype=np.float64)}
                if float(arrays["base_w"].sum()) <= 0.0:      # 护栏: 同上, 全零池→缺席
                    self._mar_zerow_pool_skips[d] += 1
                    continue
                if buf_idx is not None:
                    arrays["_buf_idx"] = buf_idx
                pools[d] = DomainPool(d, arrays)
        return pools
