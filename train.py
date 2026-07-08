#!/usr/bin/env python3
"""训练入口（stage1a t2i）：单进程（autocast）+ 多卡 FSDP2（torchrun）双模。

保留 naifu 的核心约定（OmegaConf yaml、model_cls 动态加载、
forward(batch)->(loss, sigmas, extras)、grad clip、checkpoint/resume、按 step 记日志）。

两种启动：
  * 单进程（CPU/单卡调试，autocast bf16）：
      python train.py --config configs/debug_tiny_cpu.yaml
  * 多卡 FSDP2（G4，Route B = 本地 train.py + FSDP2；mp_policy 处理 bf16，不叠 autocast）：
      torchrun --nproc_per_node=2 train.py --config configs/stage1a_t2i_512.yaml ...
检测到 torchrun 环境（LOCAL_RANK）即走 FSDP2：per-layer fully_shard（见 model_mot.fully_shard）
+ MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)（fp32 master shard + bf16 compute
+ fp32 梯度规约，对齐 naifu setup_parallelism）。checkpoint 用 DCP full_state_dict gather 到 rank0。
"""

import argparse
import faulthandler
import gc
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from training.checkpoint import (  # noqa: F401  re-export 供 scripts/experiments 仍以 `from train import ...` 调用
    _ablation_fingerprint,
    _ablation_fingerprint_check,
    _ablation_mismatch_check,
    build_scheduler,
    derive_max_steps,
    init_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from training.config_schema import validate_config
from training.diagnostics import BatchReadyBarrier, StepTraceDiagnostics


def get_class(name: str):
    import importlib
    module_name, class_name = name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


# --------------------------------------------------------------------------- #
# 分布式（torchrun）helpers
# --------------------------------------------------------------------------- #
def setup_distributed():
    """torchrun env → (is_dist, rank, world_size, local_rank)。plain python → 单进程。"""
    if "LOCAL_RANK" not in os.environ or "RANK" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return True, rank, world_size, local_rank


# --------------------------------------------------------------------------- #
# wandb（可选，默认关闭）：rank0-only；在线优先、连不上 api.wandb.ai 自动退 offline；
# run_id 持久化到 checkpoint_dir 以便 resume 续同一 run；任何 wandb 故障降级为 no-op
# （绝不中断训练）。metrics.jsonl 始终照常写（双写，离线可查）。
# --------------------------------------------------------------------------- #
class WandbLogger:
    def __init__(self, config, out_dir, is_main, resumed_step):
        self.run = None
        if not is_main:
            return
        tcfg = config.trainer
        if not bool(tcfg.get("use_wandb", False)):
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] use_wandb=true 但未安装 wandb（venv 内 pip install wandb），本次跳过。", flush=True)
            return
        mode = str(tcfg.get("wandb_mode", "online"))
        if mode == "online" and not self._reachable():
            mode = "offline"
            print("[wandb] api.wandb.ai 不可达 → 自动退 offline（训练后用 `wandb sync` 补传）。", flush=True)
        run_id_file = os.path.join(out_dir, "wandb_run_id.txt")
        run_id, resume = None, None
        if os.path.exists(run_id_file):
            with open(run_id_file, encoding="utf-8") as f:
                run_id = f.read().strip() or None
            resume = "allow" if run_id else None
        try:
            self.run = wandb.init(
                project=str(tcfg.get("wandb_project", "mot_pilot_v1")),
                entity=tcfg.get("wandb_entity", None),          # None → 个人默认 entity
                name=tcfg.get("run_name", None) or os.path.basename(out_dir.rstrip("/")) or None,
                id=run_id, resume=resume, mode=mode, dir=out_dir,
                config=OmegaConf.to_container(config, resolve=True),
            )
            if run_id is None:                                  # 新 run：写回 id 供下次 resume 续接
                with open(run_id_file, "w", encoding="utf-8") as f:
                    f.write(self.run.id)
            tail = f"，resume 续 run（from step {resumed_step}）" if resume else ""
            print(f"[wandb] run={self.run.id} mode={mode} project={self.run.project}{tail}", flush=True)
        except Exception as e:                                  # init 失败不中断训练
            print(f"[wandb] init 失败，降级为不记 wandb（继续 metrics.jsonl）：{e}", flush=True)
            self.run = None

    @staticmethod
    def _reachable(host="api.wandb.ai", port=443, timeout=5):
        import socket
        try:
            socket.create_connection((host, port), timeout=timeout).close()
            return True
        except OSError:
            return False

    def log(self, record, step):
        if self.run is None:
            return
        try:
            self.run.log({f"train/{k}": v for k, v in record.items() if k != "step"}, step=step)
        except Exception:
            pass                                                # wandb 故障绝不中断训练

    def finish(self):
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖 trainer.max_steps（调试用）")
    parser.add_argument("overrides", nargs="*", help="dotlist 覆盖，如 trainer.resume=true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    if args.max_steps is not None:
        config.trainer.max_steps = args.max_steps

    is_dist, rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    def log(msg):
        if is_main:
            print(msg, flush=True)

    # 配置 schema 早期校验（rank0-only）：对未知 key / 类型不符**只 warn 不 raise**，
    # 暴露拼写错误（见 training/config_schema.py 顶部动机）。绝不修改 cfg、不改运行行为。
    if is_main:
        validate_config(config, log=log)

    seed = int(config.trainer.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if is_dist:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(config.get("target_device", "cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = str(config.trainer.get("checkpoint_dir", "outputs/run"))
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    # 诊断脚手架（默认关、config 门控）已抽到 training.diagnostics：
    #   tracer.trace(step, phase, **fields)      ← 原 main() 内嵌套 trace()
    #   batch_ready.barrier(step, batch_size)    ← 原 main() 内 batch_ready_barrier()
    # 状态/env 写/文件名/超时语义逐字保持；关闭时对训练行为逐位无影响。
    tracer = StepTraceDiagnostics(config, out_dir, rank, local_rank, world_size, log)
    tracer.setup()
    trace = tracer.trace
    batch_ready = BatchReadyBarrier(config, out_dir, tracer, rank, local_rank, world_size, is_dist, log)
    batch_ready.setup()
    batch_ready_barrier = batch_ready.barrier
    batch_ready_enabled = batch_ready.enabled

    # ----- 精度 ----- #
    # 单进程：bf16 autocast（参数 fp32 master，无需 GradScaler）。
    # FSDP2：用 MixedPrecisionPolicy(param_dtype=bf16) 处理 bf16（fp32 master shard + bf16
    # compute + fp32 梯度规约），**不再叠 autocast**（cast_forward_inputs 已转输入）。
    amp_name = config.trainer.get("amp_dtype", None)
    if amp_name is not None and str(amp_name) not in ("bfloat16", "bf16"):
        raise NotImplementedError(f"amp_dtype={amp_name!r}：本入口只实现 bf16。fp16 需 GradScaler，未实现。")
    want_bf16 = amp_name is not None
    use_amp = want_bf16 and (not is_dist) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None

    def amp_ctx():
        return torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    # ----- model ----- #
    t0 = time.perf_counter()
    model = get_class(config.model_cls)(config=config)
    model.to(device)
    if is_dist:
        from torch.distributed.fsdp import MixedPrecisionPolicy
        mp_dtype = torch.bfloat16 if want_bf16 else torch.float32
        mp_policy = MixedPrecisionPolicy(
            param_dtype=mp_dtype, reduce_dtype=torch.float32, cast_forward_inputs=True,
        )
        model.fully_shard(mp_policy=mp_policy)  # per-layer + 整体（见 model_mot.fully_shard）
    log(f"[train] model built in {time.perf_counter() - t0:.1f}s, device={device}, "
        f"dist={is_dist}(world={world_size}), precision={'fsdp-bf16' if is_dist and want_bf16 else ('autocast-bf16' if use_amp else 'fp32')}")

    # ----- data（占位 stub，正式版换 naifu dataset+sampler+dataio）----- #
    dataset = get_class(config.dataset.store_cls)(**config.dataset)
    batch_size = int(config.trainer.batch_size)
    loader = dataset.loader(batch_size)

    # ----- optimizer / scheduler（FSDP2 下 params 已是 DTensor，须在 fully_shard 之后建）----- #
    # max_steps：默认用显式 trainer.max_steps；config 给了 max_exposure_images 则按曝光量/实际 global_batch 派生（gated）；
    # 命令行 --max-steps（调试/smoke）优先于一切（见 checkpoint.derive_max_steps）。
    max_steps = derive_max_steps(config, batch_size, world_size, log, cli_max_steps=args.max_steps)
    opt_params = OmegaConf.to_container(config.optimizer.params, resolve=True)
    wd = float(opt_params.pop("weight_decay", 0.0))
    base_lr = float(opt_params.get("lr", 0.0))
    optimizer = torch.optim.AdamW(model.get_optimizer_param_groups(weight_decay=wd, base_lr=base_lr), **opt_params)
    scheduler = build_scheduler(optimizer, config, max_steps)

    grad_clip_val = float(config.trainer.get("gradient_clip_val", 0.5))
    log_every = int(config.trainer.get("log_every", 10))
    ckpt_every = int(config.trainer.get("checkpoint_steps", 0))
    # 监控：每多少步把 dataline 真值（realized mix / buffer 水位 / producer 吞吐 / anime 取图成功率）
    # 写入 metrics.jsonl。默认对齐 log_every；<=0 关闭。修这条是为了不再只靠 low-water 代理推断 mix。
    dl_every = int(config.trainer.get("dataline_metrics_every", 0) or log_every)
    ckpt_path = os.path.join(out_dir, "last.pt")

    # ---- GC 调优（gated，默认全关=旧行为逐位一致；iter_s creep 修复 2026-07-03 第三轮）---- #
    # 实测根因（gcprobe 探针，1×8 复现）：数据线 producer(进程内线程)的 shard/caption 回源加载是
    # 10^5~10^6 级 Python 对象分配风暴 → gen2 全堆收集每 rank 每 ~21 步触发一次，单次停顿随堆
    # 基数/碎片化线性增长（fresh ~0.4s → 56k 步时 ~4s）；48 rank 任一停顿即拖全体 FSDP collective
    # → iter_s 中位线性爬升、重启归零。哨兵证实进程停顿与 gen2 事件一一对应。
    #   gc_gen2_threshold>0：调大 gen2 触发阈值（默认 10 → 建议 200，降频 20×）。churn 几乎全是
    #     无环对象（引用计数即时回收），gen2 只兜罕见循环引用，降频安全。
    #   gc_freeze_after_steps>0：续训后第 N 步 gc.collect()+gc.freeze()，把彼时存活对象（模型/
    #     framework/百万级启动对象）永久移出 GC 追踪 → gen2 扫描集与历史存活量解耦。冻结对象仍
    #     按引用计数正常释放（buffer 滚动/张量释放不受影响），仅不再参与循环收集。
    #   gc_refreeze_every>0（2026-07-05 第四轮）：单次 freeze 只解冻【彼时】存活对象；此后新晋长存对象
    #     （post-freeze 累积）仍进 gen2 扫描集 → p50 已被上面压平，但 p90/p99 停顿随进程年龄线性爬。
    #     每 N 步（与 gc_freeze_after_steps 同一套 steps_since_launch 语义，resume/init_from 兼容）再
    #     gc.collect()+gc.freeze()，把新晋长存对象持续移出扫描集。默认 0=完全关闭（零行为变化）。
    gc_gen2_threshold = int(config.trainer.get("gc_gen2_threshold", 0) or 0)
    gc_freeze_after = int(config.trainer.get("gc_freeze_after_steps", 0) or 0)
    gc_refreeze_every = int(config.trainer.get("gc_refreeze_every", 0) or 0)
    if gc_gen2_threshold > 0:
        _g0, _g1, _g2 = gc.get_threshold()
        gc.set_threshold(_g0, _g1, gc_gen2_threshold)
        log(f"[train] gc gen2 threshold {_g2} -> {gc_gen2_threshold}（gen0/gen1 不变 {_g0}/{_g1}）")

    # EMA(G5)：须在 fully_shard 之后（EMA 镜像 sharded DTensor）、resume 之前 init（往 _ema 加载）。
    ema_cfg = config.trainer.get("ema", {}) or {}
    use_ema = bool(ema_cfg.get("enabled", False))
    if use_ema:
        model.init_ema(float(ema_cfg.get("decay", 0.9999)))

    global_step = 0
    if os.path.exists(ckpt_path) and bool(config.trainer.get("resume", True)):
        # 同分辨率续训/崩溃恢复：完整恢复 model+optimizer+scheduler+global_step（逐字不变）。
        global_step = load_checkpoint(ckpt_path, model, optimizer, scheduler, is_dist, config)
        log(f"[train] resumed from {ckpt_path} at step {global_step}")
    elif config.trainer.get("init_from"):
        # 跨分辨率【阶段切换】（gated）：从上一阶段 final ckpt 载权重(+optimizer 动量)，重置 step=0、
        # 用本阶段自己的 scheduler 从头 re-warmup（设计 §6）。触发条件 = NOT(本地 last.pt 存在 AND resume)：
        #   ① 本地无 last.pt（首次启动新阶段，常态）；或 ② 本地有 last.pt 但 resume=false（显式弃进度、从上阶段重来）。
        # resume=true 时本地 last.pt 优先走上面的 resume → 新阶段 retry 不会丢进度。
        init_path = str(config.trainer.get("init_from"))
        load_opt = bool(config.trainer.get("init_load_optimizer", True))
        if os.path.exists(ckpt_path):                      # 仅 resume=false 才会走到这（②）：本地有进度却仍 init_from → 大声告警，防误弃
            log(f"[train] ⚠️ 本地存在 {ckpt_path} 但 resume=false：将从 init_from 重置 step=0，丢弃本地 last.pt 进度（如非本意请设 resume=true）")
        init_from_checkpoint(init_path, model, optimizer, is_dist, config, load_optimizer=load_opt)
        global_step = 0
        log(f"[train] init_from {init_path} (load_optimizer={load_opt}); step 重置为 0，scheduler 全新 re-warmup")
    # multi-AR（Phase2b）：把续训全局步注入 dataset → square warmup 按全局步判定（防 resume 重跑 warmup）。
    # gated：仅 CompleteV1Dataset 有此方法且 multi_ar.enabled 时生效；其它 store/非 multi_ar 为 no-op。
    if hasattr(dataset, "set_start_step"):
        dataset.set_start_step(global_step)
    steps_at_launch = global_step               # gc_freeze_after_steps 按【本次进程启动后的步数】计（resume 兼容）

    # wandb（可选）：resume 之后 init，使 run_id 续接 / 步数对齐
    wandb_logger = WandbLogger(config, out_dir, is_main, global_step)

    # ----- loop ----- #
    model.train()
    mf = open(metrics_path, "a", encoding="utf-8") if is_main else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # 监控：dataline 真值指标（不再只靠 low-water 代理）。读 dataset 上挂的计数器/水位 + 取图统计快照，
    # 产出扁平 key（mix_* = realized domain_mix 累计；mixw_* = 自上次记录的窗口 mix；buf_* = buffer 占比；
    # prod_* = producer 累计产出；anime_fetch_* = 取图 ok/cache/fail/成功率）。失败安全（任何缺失返回已有部分）。
    _dl_prev_mix: dict = {}

    def _dataline_metrics(ds, prev_mix):
        out = {}
        try:
            dc = dict(getattr(ds, "domain_counter", {}) or {})
            tot = sum(dc.values())
            if tot:
                for d, n in dc.items():
                    out[f"mix_{d}"] = round(n / tot, 4)
                win = {d: dc.get(d, 0) - prev_mix.get(d, 0) for d in dc}
                wtot = sum(v for v in win.values() if v > 0)
                if wtot:
                    for d, v in win.items():
                        if v > 0:
                            out[f"mixw_{d}"] = round(v / wtot, 4)
                prev_mix.clear()
                prev_mix.update(dc)
            for d, f in (getattr(ds, "buffer_fill", {}) or {}).items():
                out[f"buf_{d}"] = round(float(f), 3)
            for d, n in dict(getattr(ds, "produced", {}) or {}).items():
                out[f"prod_{d}"] = int(n)
        except Exception:
            pass
        try:
            from training.dataline.image_fetch import fetch_stats_snapshot
            fs = fetch_stats_snapshot()
            sr = fs.get("success_rate")
            if sr is not None:
                out["anime_fetch_ok_rate"] = sr
            out["anime_fetch_ok"] = fs.get("ok", 0)
            out["anime_fetch_cache_hit"] = fs.get("cache_hit", 0)
            out["anime_fetch_fail"] = fs.get("fail", 0)
        except Exception:
            pass
        return out

    t_iter = time.perf_counter()
    while global_step < max_steps:
        trace(global_step, "iter_begin", max_steps=max_steps)
        trace(global_step, "before_next_batch")
        _t_data = time.perf_counter()
        batch = next(loader)
        data_wait_s = time.perf_counter() - _t_data       # 训练循环阻塞等下一个 batch 的时长（dataline 供数等待）
        bsz = len(batch.get("prompts", ())) if isinstance(batch, dict) else "?"
        trace(global_step, "after_next_batch", batch_size=bsz, keys=",".join(sorted(batch.keys())))
        trace(global_step, "before_batch_ready_barrier", batch_size=bsz)
        wait_s = batch_ready_barrier(global_step, bsz)
        if batch_ready_enabled:
            trace(global_step, "after_batch_ready_barrier", wait_s=round(wait_s, 3))
        model.set_train_progress(global_step, max_steps)

        trace(global_step, "before_forward")
        with amp_ctx():
            loss, sigmas, extras = model(batch)
        trace(global_step, "after_forward")
        scalar_loss = loss.mean()
        trace(global_step, "before_backward")
        scalar_loss.backward()
        trace(global_step, "after_backward")

        if grad_clip_val > 0:
            trace(global_step, "before_grad_clip")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.model.parameters() if p.requires_grad], grad_clip_val
            )
            grad_norm = float(grad_norm)  # FSDP2 下是 mesh-aware 的全局范数
            trace(global_step, "after_grad_clip", grad_norm=grad_norm)
        else:
            grad_norm = 0.0
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite grad_norm={grad_norm} at step {global_step}")

        trace(global_step, "before_optimizer_step")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        trace(global_step, "after_optimizer_step")
        if use_ema:
            trace(global_step, "before_ema")
            model.update_ema()
            trace(global_step, "after_ema")
        global_step += 1
        if gc_freeze_after > 0 and (global_step - steps_at_launch) == gc_freeze_after:
            gc.collect()                        # 先清一遍真垃圾，避免把可回收循环永久冻结
            gc.freeze()
            log(f"[train] gc.freeze() @step {global_step}（进程启动后第 {gc_freeze_after} 步）："
                f"frozen={gc.get_freeze_count()} 对象移出 GC 追踪，gen2 扫描集与历史堆解耦")
        elif (gc_refreeze_every > 0 and (global_step - steps_at_launch) > 0
              and (global_step - steps_at_launch) % gc_refreeze_every == 0):
            # 周期 re-freeze（同 steps_since_launch 语义）：collect 掉真垃圾后再 freeze 新晋长存对象。
            # elif 保证与上面 freeze 命中同一步时只执行一次（不 collect+freeze 两遍）。
            _t_gc = time.perf_counter()
            gc.collect()
            gc.freeze()
            log(f"[train] gc re-freeze() @step {global_step}（进程启动后第 {global_step - steps_at_launch} 步，每 {gc_refreeze_every} 步）："
                f"frozen={gc.get_freeze_count()} 对象移出 GC 追踪，用时 {time.perf_counter() - _t_gc:.3f}s")

        iter_time = time.perf_counter() - t_iter
        t_iter = time.perf_counter()
        # 多卡：all-reduce loss 取均值用于展示（各 rank 数据/噪声不同）
        loss_val = scalar_loss.detach()
        if is_dist:
            trace(global_step, "before_loss_all_reduce")
            dist.all_reduce(loss_val, op=dist.ReduceOp.AVG)
            trace(global_step, "after_loss_all_reduce")
        if is_main:
            record = {
                "step": global_step,
                "loss": float(loss_val),
                "grad_norm": grad_norm,
                "lr": scheduler.get_last_lr()[0],
                "iter_s": round(iter_time, 3),
                "data_wait_s": round(data_wait_s, 3),         # 供数等待：next(loader) 阻塞时长
                "data_wait_frac": round(data_wait_s / iter_time, 3) if iter_time > 0 else 0.0,  # 占整 iter 比例 → 是否 data-bound
                **{k: float(v) for k, v in extras.items()},
            }
            if device.type == "cuda":
                record["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
            lrs = scheduler.get_last_lr()
            if len(lrs) > 1:
                record["lr_und_lora"] = lrs[-1]
            if dl_every > 0 and (global_step % dl_every == 0 or global_step == 1):
                record.update(_dataline_metrics(dataset, _dl_prev_mix))
            mf.write(json.dumps(record) + "\n")
            wandb_logger.log(record, global_step)
            if global_step % log_every == 0 or global_step == 1:
                mf.flush()
                mem_str = f" peak {record['peak_mem_gb']:.1f}G" if "peak_mem_gb" in record else ""
                print(
                    f"[train] step {global_step}/{max_steps} loss {record['loss']:.4f} "
                    f"grad {grad_norm:.3f} lr {record['lr']:.2e} shift {record.get('shift', 0):.2f} "
                    f"sigma_mean {record.get('sigma_mean', 0):.3f} iter {iter_time:.2f}s "
                    f"dw {data_wait_s:.2f}s({record['data_wait_frac']:.0%}){mem_str}", flush=True)
        trace(global_step, "iter_logged")

        if ckpt_every > 0 and global_step % ckpt_every == 0:
            trace(global_step, "before_checkpoint")
            save_checkpoint(ckpt_path, model, optimizer, scheduler, global_step, is_dist, rank, config)
            log(f"[train] checkpoint saved at step {global_step} -> {ckpt_path}")
            trace(global_step, "after_checkpoint")

    # 收尾 checkpoint：默认存（正式训练需要）；调试/短跑用真 config 时可设
    # trainer.save_final=false，避免把 ~20GB（含 optimizer+EMA）误写到 /fsx。
    trace(global_step, "loop_exit")
    if bool(config.trainer.get("save_final", True)):
        trace(global_step, "before_final_checkpoint")
        save_checkpoint(ckpt_path, model, optimizer, scheduler, global_step, is_dist, rank, config)
        trace(global_step, "after_final_checkpoint")
    elif is_main:
        log("[train] save_final=false → 跳过收尾 checkpoint")
    if mf is not None:
        mf.close()
    trace(global_step, "metrics_closed")
    wandb_logger.finish()
    trace(global_step, "wandb_finished")
    trace(global_step, "before_loader_close")
    close_loader = getattr(loader, "close", None)
    if callable(close_loader):
        close_loader()
    trace(global_step, "after_loader_close")
    summary = f"[train] done at step {global_step}; metrics -> {metrics_path}"
    if device.type == "cuda":
        summary += (f"; precision={'fsdp-bf16' if is_dist and want_bf16 else ('autocast-bf16' if use_amp else 'fp32')}"
                    f"; peak_mem={torch.cuda.max_memory_allocated(device) / 1e9:.2f}GB")
    log(summary)
    if is_dist:
        if bool(config.trainer.get("skip_distributed_teardown", False)):
            log("[train] skip_distributed_teardown=true → 跳过 dist.barrier()/destroy_process_group()")
            trace(global_step, "skip_distributed_teardown")
        else:
            trace(global_step, "before_final_dist_barrier")
            dist.barrier()
            trace(global_step, "after_final_dist_barrier")
            dist.destroy_process_group()
            trace(global_step, "after_destroy_process_group")
    if bool(config.trainer.get("force_os_exit_after_train", False)):
        log("[train] force_os_exit_after_train=true → os._exit(0)")
        trace(global_step, "before_os_exit")
        sys.stdout.flush()
        sys.stderr.flush()
        tracer.flush()
        os._exit(0)
    if tracer.trace_fp is not None:
        tracer.cancel_faulthandler()
        tracer.close()


if __name__ == "__main__":
    main()
