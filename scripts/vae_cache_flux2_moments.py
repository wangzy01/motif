"""Offline FLUX.2 VAE **moments** cache for MoT complete_v1 (256 / 512 / 1024).

为什么是 moments 不是 sampled latent：缓存 `ae.encode(px).latent_dist.parameters`
（`[mean ‖ logvar]`，shape `[2*32, S/8, S/8]` = `[64, S/8, S/8]`），训练侧再用
`networks.vae.Flux2VAE.moments_to_model_space()` 每 epoch 重采样 posterior，保留 VAE
随机性（决策：默认缓存 moments，见任务说明 + naifu scripts/batch_vae_flux.py）。

预处理与在线训练严格一致：复用 training.dataline.image_fetch 的
`resize_crop_uint8`（短边→S Lanczos、center-crop S×S、仅下采样）+ `uint8_to_norm_tensor`
（[-1,1] CHW），逐位等同 CompleteV1Dataset 喂给在线 VAE 的 pixels。

遍历基于 **manifest scanner**（daily_v1 薄 manifest 的 elig_256/512/1024 列完整遍历
eligible 样本），不走训练采样 loader（buffer/replacement/anime_nrows）。缓存范围 =
该 stage `domain_mix[stage][domain] > 0` 的域 ∩ `elig_{stage}==1` ∩ `holdout==0`
（自动：512 含 GPIC、1024 不含；GPIC 在 1024 物理上 0 张 ≥1024px）。

存储：safetensors 不能直接写 mountpoint-s3（缺随机写/mmap）→ 各 rank 先写 /local
scratch 再 copy 到 /s3。结构（每 stage）：
  stage={S}/latents/{domain}__{shard}__r{rank}__{seq}.safetensors   tensor_key=uid
  stage={S}/index/  同名 .parquet（commit marker；先写 latents 再写 index）
  stage={S}/failed/rank{r}.parquet
  summary.json / qc.json 由 stage QC 脚本（vae_cache_qc.py）汇总。

resume：按 (domain, shard) 的 shard stem 找已存 index part → 跳过其中 uid（per-shard
done set，规模有界、world_size 改变也安全）；孤儿 latents（无对应 index）启动时清理。

多 rank：torchrun env://，(domain, shard) pair 全局 stripe 到各 rank（含跨域均衡）。
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file, safe_open

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.dataline.config import load_sampling_config  # noqa: E402
from training.dataline.image_fetch import (  # noqa: E402
    decode_rgb, fetch_url_bytes, gpic_path, monet_tar_path, read_gpic_bytes,
    resize_crop_uint8, stream_tar_images, uint8_to_norm_tensor,
)

STAGES = ("256", "512", "1024")
# manifest dir 名 ↔ domain_mix 键
_MANIFEST_DIR = {"gpic": "gpic", "monet_real": "monet", "synthetic": "monet_synth", "anime": "anime"}
_REALSYNTH = ("gpic", "monet_real", "synthetic")
_REAL_COLS = ["uid", "dataset", "src_root", "src_shard", "src_key", "width", "height",
              "holdout", "elig_256", "elig_512", "elig_1024"]
_ANIME_COLS = ["uid", "key", "s3key", "dataset", "width", "height",
               "elig_256", "elig_512", "elig_1024"]


def log(rank, msg):
    print(f"[vae_cache r{rank}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# --------------------------------------------------------------------------- #
# manifest scanner（薄 manifest 全 4 域；anime 用薄 manifest 而非源 parquet）
# --------------------------------------------------------------------------- #
def list_manifest_shards(cfg) -> dict[str, list[str]]:
    base = cfg.manifest_real_synth_glob.split("{")[0].rstrip("/")   # .../daily_v1
    out = {
        "gpic": sorted(glob.glob(os.path.join(base, "gpic", "*.parquet"))),
        "monet_real": sorted(glob.glob(os.path.join(base, "monet", "*.parquet"))),
        "synthetic": sorted(glob.glob(os.path.join(base, "monet_synth", "*.parquet"))),
        "anime": sorted(glob.glob(os.path.join(cfg.manifest_anime, "*.parquet"))),
    }
    for d, s in out.items():
        assert s, f"域 {d} 没找到 manifest shard"
    return out


def participating_domains(cfg, stage: str) -> list[str]:
    dm = cfg.domain_mix[stage]
    return [d for d in ("anime", "monet_real", "gpic", "synthetic") if dm.get(d, 0.0) > 0]


def assigned_pairs(cfg, stage, rank, world, only_domains=None) -> list[tuple[str, str]]:
    """全局 stripe (domain, shard) pair 到本 rank（跨域均衡，且 world_size 变动时确定）。"""
    shards = list_manifest_shards(cfg)
    doms = participating_domains(cfg, stage)
    if only_domains:
        doms = [d for d in doms if d in only_domains]
    pairs = []
    for d in doms:
        for sp in shards[d]:
            pairs.append((d, sp))
    pairs.sort()
    return [p for i, p in enumerate(pairs) if i % world == rank]


# --------------------------------------------------------------------------- #
# resume / orphan helpers（按 shard stem keyed）
# --------------------------------------------------------------------------- #
def _part_stem(domain, shard_path, rank, seq, run_tag=""):
    """part 文件名 stem。run_tag（如补编 'patch'）插在 seq 前，**不破坏 shard 前缀**
    `{domain}__{sh}__`，故 done_uids/existing_seq_count（按前缀 glob）仍能跨主跑+补编正确
    工作；同时保证补编 part 名与主跑 part 名绝不相同（防 mountpoint-s3 覆盖）。"""
    sh = os.path.splitext(os.path.basename(shard_path))[0]
    return f"{domain}__{sh}__r{rank:03d}__{run_tag}{seq:04d}"


def _shard_prefix(domain, shard_path):
    sh = os.path.splitext(os.path.basename(shard_path))[0]
    return f"{domain}__{sh}__"


def read_parquet_file(path, columns=None):
    """单文件读，绕开 pyarrow dataset 的 hive 分区推断。

    本缓存把 index/failed parquet 放在 `stage=256/` 目录下，`pq.read_table`/`ds.dataset`
    会把目录名 `stage=256` 当 hive 分区列（int），与文件里真实的 `stage` 列（string）冲突
    （ArrowTypeError: Unable to merge Field stage）。ParquetFile.read 只读这一个文件、不做
    分区推断，规避冲突。所有读 `stage=*/` 下 parquet 的地方都走这里。"""
    return pq.ParquetFile(path).read(columns=columns)


def done_uids_for_shard(index_dir, domain, shard_path) -> set:
    """读该 (domain, shard) 已 commit 的 index part 里的 uid（per-shard done set）。"""
    done = set()
    pref = _shard_prefix(domain, shard_path)
    for p in glob.glob(os.path.join(index_dir, pref + "*.parquet")):
        try:
            t = read_parquet_file(p, columns=["uid"])
            done.update(t.column("uid").to_pylist())
        except Exception as e:  # noqa: BLE001
            print(f"  warn: bad index part {p}: {e}", flush=True)
    return done


def existing_seq_count(index_dir, domain, shard_path) -> int:
    pref = _shard_prefix(domain, shard_path)
    return len(glob.glob(os.path.join(index_dir, pref + "*.parquet")))


def clean_orphans(latents_dir, index_dir, domain, shard_path):
    """删本 shard 有 latents 无 index 的孤儿（上次 crash 在 copy latents 后、copy index 前）。"""
    pref = _shard_prefix(domain, shard_path)
    idx_stems = {os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(index_dir, pref + "*.parquet"))}
    for lp in glob.glob(os.path.join(latents_dir, pref + "*.safetensors")):
        stem = os.path.splitext(os.path.basename(lp))[0]
        if stem not in idx_stems:
            try:
                os.remove(lp)
                print(f"  cleaned orphan latents {os.path.basename(lp)}", flush=True)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# 取图 + decode + resize（per-domain），yield recs：{uid, src_key, width, height, u8}
# --------------------------------------------------------------------------- #
def _eligible_rows(domain, shard_path, stage, done: set, only_uids: set | None = None):
    """读 manifest shard，返回 eligible（elig_stage==1, holdout==0, 未 done）行 list[dict]。

    only_uids（补编用）：非 None 时只保留 uid ∈ only_uids 的行（定向重编失败 key）。
    """
    if domain == "anime":
        t = pq.read_table(shard_path, columns=_ANIME_COLS)
        elig = np.array(t.column(f"elig_{stage}").to_pylist(), dtype=np.int8) == 1
        rows = []
        uid = t.column("uid").to_pylist(); key = t.column("key").to_pylist()
        s3 = t.column("s3key").to_pylist(); w = t.column("width").to_pylist()
        h = t.column("height").to_pylist()
        for i in np.where(elig)[0]:
            i = int(i)
            if uid[i] in done:
                continue
            if only_uids is not None and uid[i] not in only_uids:
                continue
            rows.append({"uid": uid[i], "src_key": key[i], "s3key": s3[i],
                         "width": int(w[i]), "height": int(h[i])})
        return rows
    t = pq.read_table(shard_path, columns=_REAL_COLS)
    elig = np.array(t.column(f"elig_{stage}").to_pylist(), dtype=np.int8) == 1
    hold = np.array(t.column("holdout").to_pylist(), dtype=np.int8) == 0
    sel = elig & hold
    uid = t.column("uid").to_pylist(); sr = t.column("src_root").to_pylist()
    ss = t.column("src_shard").to_pylist(); sk = t.column("src_key").to_pylist()
    w = t.column("width").to_pylist(); h = t.column("height").to_pylist()
    rows = []
    for i in np.where(sel)[0]:
        i = int(i)
        if uid[i] in done:
            continue
        if only_uids is not None and uid[i] not in only_uids:
            continue
        rows.append({"uid": uid[i], "src_root": sr[i], "src_shard": ss[i], "src_key": sk[i],
                     "width": int(w[i]), "height": int(h[i])})
    return rows


def _load_gpic_ext_map(src_root, src_shard) -> dict:
    """gpic_train_{shard}.parquet -> key -> ext（权威 ext 在源 s3key；避免逐图 4×HEAD 探测）。"""
    p = os.path.join(src_root, "parquets", f"gpic_train_{src_shard}.parquet")
    try:
        t = pq.read_table(p, columns=["key", "s3key"])
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for k, s in zip(t.column("key").to_pylist(), t.column("s3key").to_pylist()):
        out[k] = (os.path.splitext(s)[1] if s else ".jpg")
    return out


def _bounded_imap(ex, fn, items, max_inflight):
    """有界并发 map：最多 max_inflight 个 future 在飞（防 250k 任务一次性提交 + 结果堆爆内存）。
    乱序 yield（缓存不关心顺序）。consumer break 时，外层 `with ThreadPoolExecutor` 收尾。"""
    it = iter(items)
    inflight = set()
    for _ in range(max_inflight):
        try:
            inflight.add(ex.submit(fn, next(it)))
        except StopIteration:
            break
    while inflight:
        done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
        for fut in done:
            yield fut.result()
            try:
                inflight.add(ex.submit(fn, next(it)))
            except StopIteration:
                pass


def decode_shard(domain, shard_path, stage, image_size, done, fail, anime_cache,
                 fetch_workers, tar_workers, only_uids=None, fetch_retries=3, fetch_timeout=20):
    """yield decoded recs（{uid, src_key, width, height, u8}）。失败 append 到 fail。"""
    rows = _eligible_rows(domain, shard_path, stage, done, only_uids=only_uids)
    if not rows:
        return

    def _decode(raw, rec):
        if not raw:
            fail.append({"uid": rec["uid"], "domain": domain, "src_key": rec["src_key"],
                         "error": "fetch_empty"})
            return None
        img = decode_rgb(raw)
        if img is None:
            fail.append({"uid": rec["uid"], "domain": domain, "src_key": rec["src_key"],
                         "error": "decode_fail"})
            return None
        try:
            u8 = resize_crop_uint8(img, image_size)
        except Exception as e:  # noqa: BLE001
            fail.append({"uid": rec["uid"], "domain": domain, "src_key": rec["src_key"],
                         "error": f"resize_fail:{type(e).__name__}"})
            return None
        return {"uid": rec["uid"], "src_key": rec["src_key"],
                "width": rec["width"], "height": rec["height"], "u8": u8}

    if domain == "anime":
        def job(rec):
            return _decode(fetch_url_bytes(rec["s3key"], retries=fetch_retries,
                                           timeout=fetch_timeout, cache_dir=anime_cache), rec)
        with ThreadPoolExecutor(max_workers=fetch_workers) as ex:
            for out in _bounded_imap(ex, job, rows, fetch_workers * 2):
                if out is not None:
                    yield out
        return

    if domain == "gpic":
        by_shard = defaultdict(list)
        for r in rows:
            by_shard[(r["src_root"], r["src_shard"])].append(r)
        with ThreadPoolExecutor(max_workers=fetch_workers) as ex:
            for (src_root, src_shard), grp in by_shard.items():
                ext_map = _load_gpic_ext_map(src_root, src_shard)

                def job(rec, _sr=src_root, _ss=src_shard, _em=ext_map):
                    return _decode(read_gpic_bytes(_sr, _ss, rec["src_key"],
                                                   ext=_em.get(rec["src_key"])), rec)
                for out in _bounded_imap(ex, job, grp, fetch_workers * 2):
                    if out is not None:
                        yield out
        return

    # monet_real / synthetic：按 src_shard 分组，每 tar 顺序流式取一次（webdataset）
    by_shard = defaultdict(list)
    for r in rows:
        by_shard[(r["src_root"], r["src_shard"])].append(r)

    def tar_job(item):
        (src_root, src_shard), grp = item
        wanted = {r["src_key"]: r for r in grp}
        got = dict(stream_tar_images(monet_tar_path(src_root, src_shard), set(wanted)))
        outs = []
        for key, rec in wanted.items():
            out = _decode(got.get(key), rec)
            if out is not None:
                outs.append(out)
        return outs

    with ThreadPoolExecutor(max_workers=tar_workers) as ex:
        # tar 结果是 list（一个 tar 几百张），in-flight 限到 tar_workers 防解码图堆内存
        for outs in _bounded_imap(ex, tar_job, list(by_shard.items()), tar_workers):
            for out in outs:
                yield out


# --------------------------------------------------------------------------- #
# VAE encode
# --------------------------------------------------------------------------- #
def build_vae(vae_path, device):
    from diffusers import AutoencoderKLFlux2
    ae = AutoencoderKLFlux2.from_pretrained(vae_path).to(device, dtype=torch.bfloat16).eval()
    ae.requires_grad_(False)
    return ae


@torch.inference_mode()
def encode_moments(ae, u8_list, device):
    """list[uint8 HWC] -> moments tensor [B, 64, S/8, S/8] bf16 on CPU。"""
    px = torch.stack([uint8_to_norm_tensor(u) for u in u8_list], dim=0)  # [B,3,S,S] f32 [-1,1]
    px = px.to(device, dtype=torch.bfloat16, non_blocking=True)
    params = ae.encode(px).latent_dist.parameters                        # [B, 64, S/8, S/8] bf16
    return params.to("cpu", dtype=torch.bfloat16).contiguous()


# --------------------------------------------------------------------------- #
# 写盘：local scratch -> copy 到 /s3
# --------------------------------------------------------------------------- #
def _copy(src, dst):
    """流式 copy（mountpoint-s3 顺序写 OK）。"""
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            b = fi.read(16 << 20)
            if not b:
                break
            fo.write(b)


def flush_chunk(moments, index_rows, *, stage, domain, shard_path, rank, seq,
                latents_dir, index_dir, scratch_dir, run_tag=""):
    """写一个 chunk：local safetensors+parquet -> copy latents -> copy index（commit marker）。"""
    if not moments:
        return 0
    stem = _part_stem(domain, shard_path, rank, seq, run_tag=run_tag)
    name_st = stem + ".safetensors"
    name_ix = stem + ".parquet"
    rel = f"latents/{name_st}"
    for r in index_rows:
        r["path"] = rel
    local_st = os.path.join(scratch_dir, name_st)
    local_ix = os.path.join(scratch_dir, name_ix)
    save_file(moments, local_st)
    cols = {k: [r[k] for r in index_rows] for k in
            ("uid", "domain", "stage", "src_key", "path", "tensor_key",
             "height", "width", "latent_shape", "status", "error")}
    tbl = pa.table({
        "uid": pa.array(cols["uid"], pa.string()),
        "domain": pa.array(cols["domain"], pa.string()),
        "stage": pa.array(cols["stage"], pa.string()),
        "src_key": pa.array(cols["src_key"], pa.string()),
        "path": pa.array(cols["path"], pa.string()),
        "tensor_key": pa.array(cols["tensor_key"], pa.string()),
        "height": pa.array(cols["height"], pa.int32()),
        "width": pa.array(cols["width"], pa.int32()),
        "latent_shape": pa.array(cols["latent_shape"], pa.string()),
        "status": pa.array(cols["status"], pa.string()),
        "error": pa.array(cols["error"], pa.string()),
    })
    pq.write_table(tbl, local_ix, compression="zstd")
    # copy latents 先于 index（index 存在 = chunk 已 commit；resume/QC 以 index 为准）
    _copy(local_st, os.path.join(latents_dir, name_st))
    _copy(local_ix, os.path.join(index_dir, name_ix))
    n = len(moments)
    try:
        os.remove(local_st); os.remove(local_ix)
    except OSError:
        pass
    return n


# --------------------------------------------------------------------------- #
# 主流程（单 rank）
# --------------------------------------------------------------------------- #
def run(args, rank, world):
    cfg = load_sampling_config(args.config)
    stage = args.stage
    S = int(stage)
    device = f"cuda:{rank % max(1, torch.cuda.device_count())}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_root = args.out_root
    stage_dir = os.path.join(out_root, f"stage={stage}")
    latents_dir = os.path.join(stage_dir, "latents")
    index_dir = os.path.join(stage_dir, "index")
    failed_dir = os.path.join(stage_dir, "failed")
    scratch_dir = os.path.join(args.scratch, f"stage{stage}_r{rank}")
    for d in (latents_dir, index_dir, failed_dir, scratch_dir):
        os.makedirs(d, exist_ok=True)
    if args.anime_cache:
        os.makedirs(args.anime_cache, exist_ok=True)

    only = set(args.only_domains.split(",")) if args.only_domains else None
    pairs = assigned_pairs(cfg, stage, rank, world, only_domains=only)
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    doms = participating_domains(cfg, stage)

    # 补编模式：只重编 only_uids（取自 failed_keys.parquet 的 uid 集）；run_tag 给 part/
    # 簿记文件换名，避免与主跑文件同名（mountpoint-s3 不能覆盖）。
    run_tag = args.run_tag
    only_uids = None
    if args.only_uids:
        ut = read_parquet_file(args.only_uids, columns=["uid"])
        only_uids = set(ut.column("uid").to_pylist())
    log(rank, f"stage={stage} S={S} domains={doms} assigned_pairs={len(pairs)} device={device}"
              + (f" run_tag={run_tag!r} only_uids={len(only_uids)}" if only_uids is not None else ""))

    ae = build_vae(args.vae_path, device)
    log(rank, "VAE loaded (AutoencoderKLFlux2, bf16)")

    chunk_bytes = int(args.chunk_gib * (1 << 30))
    per_key_bytes = 64 * (S // 8) * (S // 8) * 2  # bf16 moments
    n_ok = n_fail = 0
    fail_all = []
    t_start = time.perf_counter()

    for di, (domain, shard_path) in enumerate(pairs):
        done = done_uids_for_shard(index_dir, domain, shard_path)
        clean_orphans(latents_dir, index_dir, domain, shard_path)
        seq = existing_seq_count(index_dir, domain, shard_path)
        per_domain_limit = args.limit_per_domain
        moments, index_rows, buf_bytes = {}, [], 0
        u8_buf, meta_buf = [], []
        produced = 0
        t_shard = time.perf_counter()
        fail = []

        def encode_and_buffer():
            nonlocal buf_bytes
            if not u8_buf:
                return
            params = encode_moments(ae, u8_buf, device)
            for j, m in enumerate(meta_buf):
                uid = m["uid"]
                t = params[j].clone()
                moments[uid] = t
                index_rows.append({
                    "uid": uid, "domain": domain, "stage": stage, "src_key": m["src_key"],
                    "tensor_key": uid, "height": m["height"], "width": m["width"],
                    "latent_shape": ",".join(str(x) for x in tuple(t.shape)),
                    "status": "ok", "error": None,
                })
                buf_bytes += per_key_bytes
            u8_buf.clear(); meta_buf.clear()

        for rec in decode_shard(domain, shard_path, stage, S, done, fail,
                                args.anime_cache, args.fetch_workers, args.tar_workers,
                                only_uids=only_uids, fetch_retries=args.fetch_retries,
                                fetch_timeout=args.fetch_timeout):
            u8_buf.append(rec["u8"])
            meta_buf.append(rec)
            produced += 1
            if len(u8_buf) >= args.batch_size:
                encode_and_buffer()
            if buf_bytes >= chunk_bytes:
                n = flush_chunk(moments, index_rows, stage=stage, domain=domain,
                                shard_path=shard_path, rank=rank, seq=seq,
                                latents_dir=latents_dir, index_dir=index_dir, scratch_dir=scratch_dir,
                                run_tag=run_tag)
                n_ok += n; seq += 1; moments, index_rows, buf_bytes = {}, [], 0
            if per_domain_limit and produced >= per_domain_limit:
                break
        encode_and_buffer()
        if moments:
            n = flush_chunk(moments, index_rows, stage=stage, domain=domain,
                            shard_path=shard_path, rank=rank, seq=seq,
                            latents_dir=latents_dir, index_dir=index_dir, scratch_dir=scratch_dir,
                            run_tag=run_tag)
            n_ok += n; seq += 1
        fail_all.extend(fail); n_fail += len(fail)
        dt = time.perf_counter() - t_shard
        rate = produced / dt if dt > 0 else 0
        log(rank, f"[{di+1}/{len(pairs)}] {domain} {os.path.basename(shard_path)} "
                  f"ok={produced} fail={len(fail)} {dt:.1f}s ({rate:.1f} img/s) "
                  f"| cum ok={n_ok} fail={n_fail}")

    # 写本 rank 的 failed parquet
    if fail_all:
        ft = pa.table({
            "uid": pa.array([f["uid"] for f in fail_all], pa.string()),
            "domain": pa.array([f["domain"] for f in fail_all], pa.string()),
            "src_key": pa.array([f["src_key"] for f in fail_all], pa.string()),
            "error": pa.array([f["error"] for f in fail_all], pa.string()),
        })
        tag_sfx = f"__{run_tag}" if run_tag else ""
        lf = os.path.join(scratch_dir, f"failed_rank{rank:03d}{tag_sfx}.parquet")
        pq.write_table(ft, lf, compression="zstd")
        _copy(lf, os.path.join(failed_dir, f"rank{rank:03d}{tag_sfx}.parquet"))
        try:
            os.remove(lf)
        except OSError:
            pass

    total = time.perf_counter() - t_start
    rate = n_ok / total if total > 0 else 0
    log(rank, f"DONE stage={stage} ok={n_ok} fail={n_fail} "
              f"{total/60:.1f}min ({rate:.1f} img/s/rank)")
    return {"rank": rank, "stage": stage, "ok": int(n_ok), "fail": int(n_fail),
            "sec": round(total, 1), "img_per_s": round(rate, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--out-root", default="/s3/evan/mot_latents/complete_v1_flux2_moments")
    ap.add_argument("--scratch", default="/local/evan/vae_cache_scratch")
    ap.add_argument("--vae-path", default="/fsx/models/FLUX.2-klein-4B/vae")
    ap.add_argument("--anime-cache", default="/local/evan/anime_img_cache")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk-gib", type=float, default=3.0)
    ap.add_argument("--fetch-workers", type=int, default=64)
    ap.add_argument("--tar-workers", type=int, default=12)
    ap.add_argument("--limit-per-domain", type=int, default=0, help="smoke：每 (domain,shard) 最多产出 N（0=不限）")
    ap.add_argument("--limit-pairs", type=int, default=0, help="smoke：只处理前 N 个 (domain,shard) pair（0=不限）")
    ap.add_argument("--only-domains", default="", help="smoke：逗号分隔，只处理这些域（如 anime,gpic）")
    ap.add_argument("--only-uids", default="", help="补编：parquet 路径（需 uid 列），只重编这些 uid（取自 failed_keys.parquet）")
    ap.add_argument("--run-tag", default="", help="补编：part/簿记文件名加此标签（如 patch），防与主跑文件同名覆盖")
    ap.add_argument("--fetch-retries", type=int, default=3, help="anime R2 取图重试次数（补编可调高，如 6）")
    ap.add_argument("--fetch-timeout", type=int, default=20, help="anime R2 取图单次超时秒（补编可调高，如 30）")
    args = ap.parse_args()
    if args.limit_per_domain <= 0:
        args.limit_per_domain = None
    if args.limit_pairs <= 0:
        args.limit_pairs = None

    distributed = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", timeout=datetime.timedelta(hours=8))
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
    else:
        rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))

    stats = run(args, rank, world)

    # 各 rank 落自己的 run-stats（QC 脚本汇总）。run_tag 时换名，防覆盖主跑 rankNNN.json。
    rs_dir = os.path.join(args.out_root, f"stage={args.stage}", "run_stats")
    os.makedirs(rs_dir, exist_ok=True)
    tag_sfx = f"__{args.run_tag}" if args.run_tag else ""
    local_rs = os.path.join(args.scratch, f"runstats_r{rank}{tag_sfx}.json")
    with open(local_rs, "w") as f:
        json.dump(stats, f)
    _copy(local_rs, os.path.join(rs_dir, f"rank{rank:03d}{tag_sfx}.json"))

    if distributed:
        # 注意：这里不调 torch.distributed.barrier()。各 rank 之间在退出前无依赖
        # （run_stats / latents / index 都已各自写盘），而 QC 是 torchrun 返回后才跑的
        # 独立单进程命令，torchrun 本就会 join 所有 rank。曾因 rank 间完成时间差
        # (~11h) > NCCL 8h 超时 → 早完成的 rank 超时退出、最后一个 rank 在无对端的
        # barrier 上永久 hang，导致 torchrun 不返回、QC 从未启动（stage256 踩过）。
        # 慢 rank 一写完 run_stats 即可独立退出。
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
