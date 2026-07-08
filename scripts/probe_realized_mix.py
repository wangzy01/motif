#!/usr/bin/env python
"""离线探针：量化 CompleteV1Dataset 的【实际 domain_mix】(不碰正在跑的 prod job)。

动机：dataline 只在 buffer<25% 时告警"实际占比会低于目标"，但 domain_counter(真实曝光计数)
事后才落 metrics → 离线先量化偏移。本探针建同一条产线、按生产节奏消费 N 个 batch，直接读
domain_counter / buffer_fill / produced。

【2026-06-26 Phase-1 数据修复验证版】在 ablations 版基础上增补：
  - 末尾打印 dataset.buffer_fill(稳态各域 buffer 占比，低=仍被饿死) + produced(各域累计产出)
    + len(dataset.shards[d]) (各域本 rank 分到的 shard 数：anime 经 force_row_level 后应=全量 17)。
  - 输出一行机器可读 `[probe-json] {...}`，便于跨 rank 聚合(见 aggregate_probe_mix.py)。

保真度：
  - rank/world：用 --rank G --world 16 复刻【生产单 rank 的 shard 切片】。
  - 消费节奏：--sleep 模拟 GPU step 间隙(生产 ~2.5s/step，期间 producer 后台回填)；不加会过量消费→悲观。
  - ⚠️ 单进程独享 /s3 + R2 带宽，无 16 路争抢 → 单 rank 测得的 monet_real 占比是【上界(乐观)】。
    16-rank 同时跑(本 job)才复刻真实争抢；聚合 16 个 rank 才是生产真值。
用法：python scripts/probe_realized_mix.py --config configs/stage1a_t2i_256_v2.yaml \
        --rank 0 --world 16 --warmup 80 --count 200 --sleep 2.5 [dotlist overrides...]
"""
import argparse
import json
import os
import time
import importlib
from collections import Counter
from omegaconf import OmegaConf

_DOMAINS = ["anime", "monet_real", "gpic", "synthetic", "textatlas"]


def get_class(name: str):
    module_name, class_name = name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def fmt_mix(counter: Counter, target: dict) -> str:
    total = sum(counter.values()) or 1
    rows = []
    for d in _DOMAINS:
        got = counter.get(d, 0)
        share = got / total
        tgt = float(target.get(d, 0.0))
        delta = share - tgt
        rows.append(f"    {d:11s} 实测 {share:6.1%}  目标 {tgt:6.1%}  Δ {delta:+6.1%}  (n={got})")
    return f"  total={total}\n" + "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=80, help="预热 batch 数(buffer 达稳态后再计数)")
    ap.add_argument("--count", type=int, default=200, help="计数 batch 数")
    ap.add_argument("--sleep", type=float, default=2.5, help="每 batch 后 sleep 秒，模拟 GPU step 间隙")
    ap.add_argument("--batch-size", type=int, default=None, help="默认取 config.trainer.batch_size")
    ap.add_argument("overrides", nargs="*", help="dotlist 覆盖，如 dataset.stats_cache=...")
    args = ap.parse_args()

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    ds_kwargs = OmegaConf.to_container(config.dataset, resolve=True)
    store_cls = ds_kwargs.get("store_cls")
    ds_kwargs["rank"] = args.rank
    ds_kwargs["world_size"] = args.world
    bs = int(args.batch_size or config.trainer.batch_size)

    print(f"[probe] 建 dataset：rank={args.rank}/world={args.world} bs={bs} "
          f"warmup={args.warmup} count={args.count} sleep={args.sleep}s", flush=True)
    t0 = time.perf_counter()
    dataset = get_class(store_cls)(**ds_kwargs)
    target = dict(dataset.cfg.domain_mix[dataset.stage])  # SamplingConfig.domain_mix[stage] 已是 plain dict
    shard_counts = {d: len(dataset.shards.get(d, [])) for d in _DOMAINS}
    print(f"[probe] dataset 就绪 in {time.perf_counter()-t0:.1f}s；目标 mix={target}", flush=True)
    print(f"[probe] 本 rank 各域 shard 数={shard_counts}  "
          f"(anime 经 force_row_level 应=全量；gpic/monet/synth 文件级切片)", flush=True)

    loader = dataset.loader(bs)
    it = iter(loader)

    # ---- 预热：消费 warmup 个 batch，让慢域 buffer 进入稳态(不计入最终统计)----
    print(f"[probe] 预热 {args.warmup} batch ...", flush=True)
    twarm = time.perf_counter()
    for i in range(args.warmup):
        next(it)
        if args.sleep > 0:
            time.sleep(args.sleep)
        if (i + 1) % 20 == 0:
            print(f"  [warmup] {i+1}/{args.warmup}  累计 mix(含预热):\n{fmt_mix(dataset.domain_counter, target)}",
                  flush=True)
    print(f"[probe] 预热完成 in {time.perf_counter()-twarm:.0f}s。重置计数器，开始正式计数。", flush=True)

    # ---- 重置：清空预热期的曝光，只统计稳态 ----
    dataset.domain_counter.clear()
    dataset.cluster_counter.clear()
    produced_at_reset = dict(dataset.produced)

    tcount = time.perf_counter()
    for i in range(args.count):
        next(it)
        if args.sleep > 0:
            time.sleep(args.sleep)
        if (i + 1) % 50 == 0:
            print(f"  [count] {i+1}/{args.count}  稳态 mix:\n{fmt_mix(dataset.domain_counter, target)}",
                  flush=True)
    elapsed = time.perf_counter() - tcount

    # ---- 稳态诊断：buffer 占比(低=仍被饿死) + 计数窗内各域产出速率 ----
    buf = {d: round(float(dataset.buffer_fill.get(d, 0.0)), 3) for d in _DOMAINS}
    prod_delta = {d: int(dataset.produced.get(d, 0)) - int(produced_at_reset.get(d, 0)) for d in _DOMAINS}
    prod_rate = {d: round(prod_delta[d] / elapsed, 2) for d in _DOMAINS} if elapsed > 0 else {}

    print("\n========== 最终稳态实际 mix ==========", flush=True)
    print(fmt_mix(dataset.domain_counter, target), flush=True)
    print(f"[probe] 稳态 buffer 占比={buf}  (低=该域仍供不上→欠采风险)", flush=True)
    print(f"[probe] 计数窗内产出 {prod_delta}  ≈速率(图/s){prod_rate}  "
          f"(稳态受 queue 背压，≈消费速率；某域明显低=瓶颈)", flush=True)
    print(f"[probe] 计数耗时 {elapsed:.0f}s({args.count} batch)。", flush=True)

    summary = {
        "rank": args.rank, "world": args.world, "bs": bs,
        "counts": {d: int(dataset.domain_counter.get(d, 0)) for d in _DOMAINS},
        "target": {d: float(target.get(d, 0.0)) for d in _DOMAINS},
        "buffer_fill": buf, "produced_count_window": prod_delta, "produced_rate_imgs_s": prod_rate,
        "shard_counts": shard_counts, "count_elapsed_s": round(elapsed, 1),
    }
    print(f"[probe-json] {json.dumps(summary, ensure_ascii=False)}", flush=True)

    # 审计 F4：json 已 flush 落盘后，停 producer 线程并强制退出。否则 loader 生成器未显式 close →
    # stop_evt 不触发、producer/取图 ThreadPoolExecutor 线程残留 → 个别 rank 进程不退、JCC job 卡 Running。
    try:
        loader.close()                  # 触发 loader 的 finally: stop_evt.set() → producer 收尾
    except Exception:
        pass
    os._exit(0)                         # 诊断工具：保证进程退出（生产 train.py 自身已正确 close loader，不用此法）


if __name__ == "__main__":
    main()
