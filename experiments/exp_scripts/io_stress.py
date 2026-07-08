#!/usr/bin/env python3
"""EXP2 — 真实数据 IO 压测：在真实 forward+backward 负载下，分离测「data wait vs compute」，
采样 GPU util，统计各 domain 实际抽样比例、失败样本。判断正式训练是否会被 IO 卡住。

不改核心代码：复用 MoTT2IModel + CompleteV1Dataset + train.py 同款精度/grad_ckpt。
每次进程只跑【一个 (stage, 数据线 knob) 设置】（producer 是 daemon 线程，新进程最干净）；
外层 job 脚本循环 stage × knob。可单卡或 torchrun（多卡时各 rank 测自身 loader 压力 + 全局 stats 协调）。

度量（measured 窗口，warmup 后）：
  - data_wait_s   : next(loader) 耗时（数据没准备好时 GPU 干等的时间）
  - compute_s     : forward+backward+opt.step 耗时
  - data_frac     : data_wait / (data_wait+compute)  —— >~10-15% 说明 IO 是瓶颈
  - iter_s, img/s
  - gpu_util_mean : 后台 nvidia-smi 采样（measured 窗口内 GPU 利用率均值/中位）
  - peak_mem_gb
  - domain_counter: 各域实际曝光数 + 占比（对照 config domain_mix）
  - 失败样本：producer 跳过的 shard（看 stderr [producer ...] 行）

用法（单设置）：
  venv-gpu/bin/python experiments/exp_scripts/io_stress.py --config configs/stage1a_t2i_256.yaml \
      --stage 256 --image-size 256 --batch-size 64 --buffer-size 256 --queue-size 128 \
      --anime-nrows 20000 --warmup 5 --measure 40 --stats-cache /local/evan/mot_stats/monet_global_stats.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GpuSampler(threading.Thread):
    """后台每 interval 秒采 GPU util / mem（measured 窗口内）。"""
    def __init__(self, index, interval=0.5):
        super().__init__(daemon=True)
        self.index, self.interval = index, interval
        self.utils, self.mems = [], []
        self._stop = threading.Event()
        self._active = threading.Event()

    def run(self):
        while not self._stop.is_set():
            if self._active.is_set():
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                         "--format=csv,noheader,nounits", "-i", str(self.index)],
                        timeout=5).decode().strip().splitlines()[0]
                    u, m = out.split(",")
                    self.utils.append(float(u))
                    self.mems.append(float(m))
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(self.interval)

    def start_window(self):
        self.utils.clear(); self.mems.clear(); self._active.set()

    def stop_window(self):
        self._active.clear()

    def stop(self):
        self._stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--buffer-size", type=int, default=None)
    ap.add_argument("--queue-size", type=int, default=None)
    ap.add_argument("--anime-nrows", type=int, default=None)
    ap.add_argument("--min-fill", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--measure", type=int, default=40)
    ap.add_argument("--stats-cache", default="/local/evan/mot_stats/monet_global_stats.json")
    ap.add_argument("--no-compute", action="store_true", help="只测 loader（不建模型/不算）")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_dist = "LOCAL_RANK" in os.environ and world > 1
    if is_dist:
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    cfg = OmegaConf.load(args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config))
    ds_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    if args.stage:
        ds_cfg["stage"] = str(args.stage)
    if args.image_size:
        ds_cfg["image_size"] = int(args.image_size)
    for k_arg, k_cfg in [("buffer_size", "buffer_size"), ("queue_size", "queue_size"),
                         ("anime_nrows", "anime_nrows"), ("min_fill", "min_fill")]:
        v = getattr(args, k_arg)
        if v is not None:
            ds_cfg[k_cfg] = int(v)
    ds_cfg["stats_cache"] = args.stats_cache
    bs = int(args.batch_size or cfg.trainer.batch_size)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    from training.dataline.complete_dataset import CompleteV1Dataset
    t0 = time.perf_counter()
    ds = CompleteV1Dataset(**ds_cfg)
    init_s = time.perf_counter() - t0
    loader = ds.loader(bs)

    model = opt = None
    if not args.no_compute and device.type == "cuda":
        from networks.model_mot import MoTT2IModel
        tb = time.perf_counter()
        model = MoTT2IModel(config=cfg).to(device)
        model.train()
        op = OmegaConf.to_container(cfg.optimizer.params, resolve=True)
        wd = float(op.pop("weight_decay", 0.0))
        opt = torch.optim.AdamW(model.get_optimizer_param_groups(weight_decay=wd), **op)
        if rank == 0:
            print(f"[io_stress] model+opt built {time.perf_counter()-tb:.1f}s", flush=True)

    sampler = GpuSampler(local_rank) if device.type == "cuda" else None
    if sampler:
        sampler.start()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data_t, comp_t, iter_t = [], [], []
    dom0 = dict(ds.domain_counter)
    total = args.warmup + args.measure
    for i in range(total):
        if i == args.warmup:
            dom0 = dict(ds.domain_counter)  # 测量窗口起点
            if sampler:
                sampler.start_window()
        t_a = time.perf_counter()
        batch = next(loader)
        t_b = time.perf_counter()
        if model is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = model(batch)
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
        t_c = time.perf_counter()
        if i >= args.warmup:
            data_t.append(t_b - t_a)
            comp_t.append(t_c - t_b)
            iter_t.append(t_c - t_a)
    if sampler:
        sampler.stop_window()

    dom1 = dict(ds.domain_counter)
    dom_delta = {d: dom1.get(d, 0) - dom0.get(d, 0) for d in set(dom1) | set(dom0)}
    dom_total = sum(dom_delta.values()) or 1
    dom_frac = {d: round(c / dom_total, 4) for d, c in sorted(dom_delta.items())}

    def stat(xs):
        return {"mean": round(statistics.mean(xs), 4), "median": round(statistics.median(xs), 4),
                "p95": round(sorted(xs)[max(0, int(len(xs) * 0.95) - 1)], 4), "max": round(max(xs), 4)} if xs else {}

    data_sum, comp_sum = sum(data_t), sum(comp_t)
    out = {
        "rank": rank, "world": world, "stage": str(ds_cfg["stage"]), "image_size": ds_cfg["image_size"],
        "batch_size": bs, "buffer_size": ds_cfg.get("buffer_size"), "queue_size": ds_cfg.get("queue_size"),
        "anime_nrows": ds_cfg.get("anime_nrows"), "init_s": round(init_s, 2),
        "measured": len(iter_t),
        "data_wait_s": stat(data_t), "compute_s": stat(comp_t), "iter_s": stat(iter_t),
        "data_frac": round(data_sum / (data_sum + comp_sum), 4) if (data_sum + comp_sum) > 0 else None,
        "img_per_s": round(bs * len(iter_t) / sum(iter_t), 2) if iter_t else None,
        "img_per_s_global": round(bs * world * len(iter_t) / sum(iter_t), 2) if iter_t else None,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if device.type == "cuda" else None,
        "gpu_util_mean": round(statistics.mean(sampler.utils), 1) if (sampler and sampler.utils) else None,
        "gpu_util_median": round(statistics.median(sampler.utils), 1) if (sampler and sampler.utils) else None,
        "domain_frac_measured": dom_frac,
        "domain_mix_config": ds.cfg.domain_mix[str(ds_cfg["stage"])],
    }
    if sampler:
        sampler.stop()
    print(f"IO_STRESS_JSON {json.dumps(out, ensure_ascii=False)}", flush=True)
    if rank == 0:
        d, c = out["data_wait_s"], out["compute_s"]
        print(f"\n[io_stress] stage={out['stage']} bs={bs} buf={out['buffer_size']} q={out['queue_size']} "
              f"anime_nrows={out['anime_nrows']}", flush=True)
        print(f"  data_wait median={d.get('median')}s  compute median={c.get('median')}s  "
              f"data_frac={out['data_frac']}  img/s(rank)={out['img_per_s']} "
              f"img/s(global)={out['img_per_s_global']}", flush=True)
        print(f"  gpu_util mean={out['gpu_util_mean']}%  peak_mem={out['peak_mem_gb']}GB", flush=True)
        print(f"  domain_frac(measured)={out['domain_frac_measured']}", flush=True)
        print(f"  domain_mix(config)   ={out['domain_mix_config']}", flush=True)
        verdict = ("IO-BOUND ⚠️" if (out["data_frac"] or 0) > 0.15 else "compute-bound ✅")
        print(f"  -> {verdict} (data_frac {out['data_frac']})", flush=True)
    if is_dist:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
