#!/usr/bin/env python3
"""Stress the real NAI-like ARB dataline without GPUs.

This intentionally uses real manifests/tars/URLs and multiple rank-like
processes so native decoder/IO crashes can be reproduced without burning H100s.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from pathlib import Path


def _parse_domains(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [d.strip() for d in raw.replace(",", " ").split() if d.strip()]


def _jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def _make_domain_filtered_config(config_path: str, only_domains: list[str], out: Path) -> tuple[str, dict]:
    from omegaconf import OmegaConf

    config_file = Path(config_path)
    cfg = OmegaConf.load(config_file)
    stage = str(cfg.dataset.stage)
    sampling_path = Path(str(cfg.dataset.sampling_config))
    if not sampling_path.is_absolute():
        sampling_path = (config_file.parent / sampling_path).resolve()
    with sampling_path.open("r", encoding="utf-8") as f:
        sampling = json.load(f)

    domain_mix = sampling.get("domain_mix")
    if not isinstance(domain_mix, dict) or stage not in domain_mix:
        raise ValueError(f"sampling_config has no domain_mix for stage={stage!r}: {sampling_path}")
    mix = domain_mix[stage]
    if not isinstance(mix, dict) or not mix:
        raise ValueError(f"domain_mix[{stage!r}] is empty or invalid in {sampling_path}")

    unknown = [d for d in only_domains if d not in mix]
    if unknown:
        raise ValueError(f"unknown --only-domains entries {unknown}; valid domains are {sorted(mix)}")

    selected = list(dict.fromkeys(only_domains))
    selected_weight_sum = sum(float(mix[d]) for d in selected if float(mix[d]) > 0.0)
    if selected_weight_sum <= 0.0:
        normalized = {d: (1.0 / len(selected)) for d in selected}
    else:
        normalized = {d: (float(mix[d]) / selected_weight_sum) for d in selected}

    new_mix = {d: 0.0 for d in mix}
    for d, w in normalized.items():
        new_mix[d] = float(w)
    sampling["domain_mix"][stage] = new_mix

    suffix = "_".join(selected)
    sampling_out = out / f"sampling_config_only_{suffix}.json"
    config_out = out / f"config_only_{suffix}.yaml"
    with sampling_out.open("w", encoding="utf-8") as f:
        json.dump(sampling, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    cfg.dataset.sampling_config = str(sampling_out)
    OmegaConf.save(cfg, config_out)

    report = {
        "stage": stage,
        "source_config": str(config_file),
        "source_sampling_config": str(sampling_path),
        "filtered_config": str(config_out),
        "filtered_sampling_config": str(sampling_out),
        "selected_domains": selected,
        "domain_mix": new_mix,
    }
    return str(config_out), report


def _worker(rank: int, args_dict: dict) -> None:
    args = argparse.Namespace(**args_dict)
    out = Path(args.out)
    rank_log = out / f"rank{rank:05d}.jsonl"
    fh_path = out / f"rank{rank:05d}.faulthandler.log"
    started = time.time()

    try:
        import faulthandler

        fh = fh_path.open("w", encoding="utf-8")
        faulthandler.enable(file=fh, all_threads=True)
        faulthandler.dump_traceback_later(args.dump_every_s, repeat=True, file=fh)
    except Exception as exc:  # noqa: BLE001 diagnostics must not block stress
        fh = None
        _jsonl(rank_log, {"event": "faulthandler_setup_failed", "error": repr(exc), "ts": time.time()})

    try:
        import torch
        from omegaconf import OmegaConf

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        from training.dataline.complete_dataset import CompleteV1Dataset
        from training.dataline.image_fetch import fetch_stats_snapshot

        cfg = OmegaConf.load(args.config)
        kwargs = OmegaConf.to_container(cfg.dataset, resolve=True)
        overrides = {
            "rank": rank,
            "world_size": args.world_size,
            "buffer_size": args.buffer_size,
            "queue_size": args.queue_size,
            "min_fill": args.min_fill,
            "fetch_workers": args.fetch_workers,
            "fetch_workers_anime": args.fetch_workers_anime,
            "producer_native_io_serial": args.producer_native_io_serial,
        }
        if args.max_shards > 0:
            overrides["max_shards"] = args.max_shards
        if args.max_rows_per_shard > 0:
            overrides["max_rows_per_shard"] = args.max_rows_per_shard
        if args.anime_nrows > 0:
            overrides["anime_nrows"] = args.anime_nrows
        if args.stats_cache:
            overrides["stats_cache"] = args.stats_cache
        if args.bucket_wait_timeout_s is not None:
            mar = dict(kwargs.get("multi_ar") or {})
            mar["bucket_wait_timeout_s"] = float(args.bucket_wait_timeout_s)
            kwargs["multi_ar"] = mar
        kwargs.update(overrides)

        _jsonl(rank_log, {
            "event": "before_dataset",
            "rank": rank,
            "world_size": args.world_size,
            "overrides": overrides,
            "pid": os.getpid(),
            "ts": time.time(),
        })
        ds = CompleteV1Dataset(**kwargs)
        _jsonl(rank_log, {
            "event": "after_dataset",
            "rank": rank,
            "domains": ds.domains,
            "shards": {d: len(v) for d, v in ds.shards.items()},
            "buckets": len(getattr(ds, "_mar_buckets", []) or []),
            "pid": os.getpid(),
            "ts": time.time(),
        })

        loader = ds.loader(args.batch_size)
        deadline = started + args.seconds
        batches = 0
        while time.time() < deadline and batches < args.max_batches:
            _jsonl(rank_log, {"event": "before_next_batch", "rank": rank, "batch": batches + 1, "ts": time.time()})
            batch = next(loader)
            batches += 1
            pix = batch["pixels"]
            _jsonl(rank_log, {
                "event": "after_next_batch",
                "rank": rank,
                "batch": batches,
                "shape": list(pix.shape),
                "produced": dict(ds.produced),
                "sampled": dict(ds.domain_counter),
                "buffer_fill": getattr(ds, "buffer_fill", {}),
                "reresize": dict(getattr(ds, "_mar_reresize_by_domain", {})),
                "fetch_stats": fetch_stats_snapshot(),
                "ts": time.time(),
            })
            del batch, pix

        _jsonl(rank_log, {
            "event": "PASS",
            "rank": rank,
            "batches": batches,
            "elapsed_s": round(time.time() - started, 3),
            "ts": time.time(),
        })
    except BaseException as exc:  # noqa: BLE001 child must report Python exceptions
        _jsonl(rank_log, {
            "event": "EXCEPTION",
            "rank": rank,
            "type": type(exc).__name__,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "ts": time.time(),
        })
        raise
    finally:
        try:
            import faulthandler

            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass
        if fh is not None:
            fh.flush()
            fh.close()


def _exit_name(code: int | None) -> str:
    if code is None:
        return "running"
    if code < 0:
        sig = -code
        try:
            return f"signal:{signal.Signals(sig).name}"
        except ValueError:
            return f"signal:{sig}"
    return f"exit:{code}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage256_redesign_naifu_arb_v1.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--world-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=144)
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--max-batches", type=int, default=8)
    ap.add_argument("--buffer-size", type=int, default=512)
    ap.add_argument("--queue-size", type=int, default=128)
    ap.add_argument("--min-fill", type=int, default=64)
    ap.add_argument("--fetch-workers", type=int, default=1)
    ap.add_argument("--fetch-workers-anime", type=int, default=1)
    ap.add_argument("--producer-native-io-serial", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--max-rows-per-shard", type=int, default=0)
    ap.add_argument("--anime-nrows", type=int, default=20000)
    ap.add_argument("--stats-cache", default=None)
    ap.add_argument("--bucket-wait-timeout-s", type=float, default=None)
    ap.add_argument("--only-domains", default=None, help="Comma/space-separated domain subset for bisection.")
    ap.add_argument("--dump-every-s", type=int, default=60)
    ap.add_argument("--join-grace-s", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.jsonl"
    only_domains = _parse_domains(args.only_domains)
    domain_filter_report = None
    if only_domains:
        args.config, domain_filter_report = _make_domain_filtered_config(args.config, only_domains, out)
        args.only_domains = ",".join(domain_filter_report["selected_domains"])
    _jsonl(summary, {
        "event": "START",
        "args": vars(args),
        "domain_filter": domain_filter_report,
        "pid": os.getpid(),
        "env": {k: os.environ.get(k) for k in [
            "PYTHONFAULTHANDLER",
            "MALLOC_CHECK_",
            "MALLOC_ARENA_MAX",
            "OMP_NUM_THREADS",
            "ARROW_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES",
        ]},
        "ts": time.time(),
    })

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    procs: list[mp.Process] = []
    args_dict = vars(args)
    for rank in range(args.world_size):
        p = mp.Process(target=_worker, args=(rank, args_dict), name=f"rank{rank:05d}")
        p.start()
        procs.append(p)
        _jsonl(summary, {"event": "child_started", "rank": rank, "pid": p.pid, "ts": time.time()})

    deadline = time.time() + args.seconds + args.join_grace_s
    failed = False
    while time.time() < deadline:
        alive = [p for p in procs if p.is_alive()]
        for p in procs:
            if p.exitcode is not None and not getattr(p, "_reported_exit", False):
                setattr(p, "_reported_exit", True)
                code_name = _exit_name(p.exitcode)
                _jsonl(summary, {
                    "event": "child_exit",
                    "rank_name": p.name,
                    "pid": p.pid,
                    "exitcode": p.exitcode,
                    "exit": code_name,
                    "ts": time.time(),
                })
                if p.exitcode != 0:
                    failed = True
        if failed or not alive:
            break
        time.sleep(2.0)

    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
            p.join(timeout=5)

    exits = {p.name: p.exitcode for p in procs}
    _jsonl(summary, {
        "event": "END",
        "exits": exits,
        "exit_names": {k: _exit_name(v) for k, v in exits.items()},
        "failed": any(v != 0 for v in exits.values()),
        "ts": time.time(),
    })
    return 1 if any(v != 0 for v in exits.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
