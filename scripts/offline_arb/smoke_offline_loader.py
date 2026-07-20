#!/usr/bin/env python3
"""CPU smoke for the OFFLINE_ARB_V1 exact loader.

Production mode spawns one process per simulated rank and checks every emitted
item row against the authoritative plan block.  ``--self-test`` creates a tiny
local synthetic tar/caption/catalog/plan fixture and performs the same checks
without touching any production data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import queue
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _latency_stats_ms(values_s: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values_s) * 1000.0, 3) if values_s else 0.0,
        "p50": round(_percentile(values_s, 0.50) * 1000.0, 3),
        "p95": round(_percentile(values_s, 0.95) * 1000.0, 3),
        "p99": round(_percentile(values_s, 0.99) * 1000.0, 3),
        "max": round(max(values_s) * 1000.0, 3) if values_s else 0.0,
    }


def _update_pixels_digest(digest: Any, pixels: Any) -> None:
    """Append one final pixel batch using an unambiguous deterministic encoding."""
    tensor = pixels.detach().cpu().contiguous()
    array = tensor.numpy()
    metadata = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(len(metadata).to_bytes(8, byteorder="big", signed=False))
    digest.update(metadata)
    digest.update(memoryview(array).cast("B"))


def _load_dataset_kwargs(
    catalog_root: str,
    plan_root: str,
    anime_freq_root: str | None,
    rank: int,
    ranks: int,
    stage: str,
) -> dict[str, Any]:
    catalog_header = json.loads(
        (Path(catalog_root) / "header.json").read_text(encoding="utf-8")
    )
    kwargs: dict[str, Any] = {}
    # Diagnostics may opt into the exact formal dataset stanza without mutating
    # the immutable catalog header.  The default remains the catalog-recorded
    # construction config for backwards-compatible smoke evidence.
    pipeline_config = os.environ.get("OFFLINE_ARB_SMOKE_DATASET_CONFIG")
    if not pipeline_config:
        pipeline_config = catalog_header.get("inputs", {}).get("pipeline_config")
    if pipeline_config and Path(str(pipeline_config)).is_file():
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(str(pipeline_config))
        kwargs = dict(OmegaConf.to_container(cfg.dataset, resolve=True))
    else:
        sampling_config = catalog_header.get("inputs", {}).get("sampling_config")
        if not sampling_config:
            raise RuntimeError(
                "catalog header has neither a readable inputs.pipeline_config nor inputs.sampling_config"
            )
        kwargs = {"sampling_config": str(sampling_config)}
    kwargs.pop("store_cls", None)
    multi_ar = dict(kwargs.get("multi_ar") or {})
    multi_ar["assignment_mode"] = "offline_exact_v1"
    kwargs.update(
        {
            "offline_plan_root": plan_root,
            "offline_catalog_root": catalog_root,
            "offline_anime_freq_root": anime_freq_root,
            "rank": rank,
            "world_size": ranks,
            "stage": stage,
            "multi_ar": multi_ar,
        }
    )
    return kwargs


def _worker(
    rank: int,
    ranks: int,
    steps: int,
    start_step: int,
    stage: str,
    plan_root: str,
    catalog_root: str,
    anime_freq_root: str | None,
    result_queue,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(ranks)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(ranks)
    dataset = None
    loader = None
    try:
        from training.dataline_offline.offline_dataset import OfflineExactDataset

        kwargs = _load_dataset_kwargs(
            catalog_root, plan_root, anime_freq_root, rank, ranks, stage
        )
        dataset = OfflineExactDataset(**kwargs)
        batch_size = int(dataset.plan_header["per_rank_bs"])
        dataset.set_start_step(start_step)
        loader = dataset.loader(batch_size)
        latencies: list[float] = []
        records: list[dict[str, Any]] = []
        prompt_stream: list[list[str]] = []   # 审查 P2-1:供 freq on/off 的 caption 逐字节等值 A/B
        cfg_keep_stream: list[list[bool]] = []
        pixels_digest = hashlib.sha256()
        started = time.perf_counter()
        for index in range(steps):
            before = time.perf_counter()
            batch = next(loader)
            latency = time.perf_counter() - before
            latencies.append(latency)
            trace = dict(dataset.last_item_trace or {})
            expected_step = start_step + index
            if trace.get("plan_step") != expected_step:
                raise AssertionError(
                    f"rank {rank}: trace step {trace.get('plan_step')} != {expected_step}"
                )
            shape = list(batch["pixels"].shape)
            bucket_hw = list(trace["bucket_hw"])
            expected_shape = [batch_size, 3, bucket_hw[0], bucket_hw[1]]
            if shape != expected_shape:
                raise AssertionError(
                    f"rank {rank} step {expected_step}: shape {shape} != {expected_shape}"
                )
            if len(batch["prompts"]) != batch_size or len(batch["cfg_keep"]) != batch_size:
                raise AssertionError(
                    f"rank {rank} step {expected_step}: prompt/cfg_keep batch length mismatch"
                )
            prompt_stream.append([str(p) for p in batch["prompts"]])
            cfg_keep_stream.append([bool(value) for value in batch["cfg_keep"]])
            _update_pixels_digest(pixels_digest, batch["pixels"])
            rows = [int(x) for x in trace["item_rows"]]
            if len(rows) != batch_size:
                raise AssertionError(
                    f"rank {rank} step {expected_step}: item trace length mismatch"
                )
            records.append(
                {
                    "step": expected_step,
                    "shape": shape,
                    "bucket_idx": int(trace["bucket_idx"]),
                    "bucket_hw": bucket_hw,
                    "item_rows": rows,
                    "latency_s": latency,
                }
            )
        wall = time.perf_counter() - started
        trace_digest = hashlib.sha256(
            json.dumps(
                [record["item_rows"] for record in records],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        prompts_digest = hashlib.sha256(
            json.dumps(prompt_stream, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cfg_keep_digest = hashlib.sha256(
            json.dumps(cfg_keep_stream, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        result_queue.put(
            {
                "ok": True,
                "rank": rank,
                "batch_size": batch_size,
                "records": records,
                "wall_s": wall,
                "img_s": (steps * batch_size / wall) if wall else 0.0,
                "latency_ms": _latency_stats_ms(latencies),
                "trace_sha256": trace_digest,
                "prompts_sha256": prompts_digest,
                "cfg_keep_sha256": cfg_keep_digest,
                "pixels_sha256": pixels_digest.hexdigest(),
                "caption_workers_anime": int(dataset._caption_workers["anime"]),
                "caption_workers_effective": {
                    domain: int(workers)
                    for domain, workers in sorted(dataset._caption_workers.items())
                },
                "buffer_fill": dict(dataset.buffer_fill),
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "rank": rank,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if loader is not None:
            try:
                loader.close()
            except Exception:
                pass
        if dataset is not None:
            try:
                dataset.close()
            except Exception:
                pass
        # 镜像 trainer 的 force_os_exit_after_train 语义(2026-07-11 首跑实证,Evan 批准方案 A):
        # ThreadPoolExecutor 线程(Py3.9+ 非 daemon)会把进程退出钉死在在途 FUSE/HTTP 读上
        # (rank1/2 结果全 ok 却因收尾滞留 >90s 被判 timed_out)。结果已 put 并 flush 队列
        # 后硬退;成败判定一律以 result dict 为准(_exit(0) 使 nonzero_workers 检查恒空)。
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass
        os._exit(0)


def run_smoke(
    plan_root: str,
    catalog_root: str,
    anime_freq_root: str | None,
    ranks: int,
    steps: int,
    start_step: int,
    stage: str,
    worker_timeout_s: float,
) -> dict[str, Any]:
    from training.dataline_offline.planner_v1 import OfflinePlan

    plan = OfflinePlan(plan_root, world_size=ranks, verify_checksums=True)
    header = plan.header
    total_steps = int(header["total_steps"])
    if start_step < 0 or start_step + steps > total_steps:
        raise ValueError(
            f"requested steps [{start_step},{start_step + steps}) outside [0,{total_steps})"
        )
    if steps <= 0 or ranks <= 0:
        raise ValueError("--ranks and --steps must be positive")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_worker,
            args=(
                rank,
                ranks,
                steps,
                start_step,
                stage,
                str(plan_root),
                str(catalog_root),
                anime_freq_root,
                result_queue,
            ),
            name=f"offline-smoke-rank-{rank}",
        )
        for rank in range(ranks)
    ]
    wall_started = time.perf_counter()
    for process in processes:
        process.start()

    results: list[dict[str, Any]] = []
    deadline = time.monotonic() + worker_timeout_s
    while len(results) < ranks:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            results.append(result_queue.get(timeout=min(5.0, remaining)))
        except queue.Empty:
            if all(not process.is_alive() for process in processes):
                break
    # Workers publish their result before closing prefetch pools.  Give in-flight
    # HTTP/file calls a bounded grace period so a successful exact trace is not
    # mislabeled merely because daemon supply cleanup took a few seconds.
    cleanup_deadline = min(deadline, time.monotonic() + 90.0)
    while any(process.is_alive() for process in processes) and time.monotonic() < cleanup_deadline:
        for process in processes:
            process.join(timeout=0.25)
    timed_out = [process.name for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
    missing_ranks = sorted(set(range(ranks)) - {int(r["rank"]) for r in results})
    failed = [r for r in results if not r.get("ok")]
    nonzero = {
        process.name: process.exitcode
        for process in processes
        if process.exitcode not in (0, None)
    }
    if timed_out or missing_ranks or failed or nonzero:
        raise RuntimeError(
            json.dumps(
                {
                    "timed_out": timed_out,
                    "missing_ranks": missing_ranks,
                    "nonzero_workers": nonzero,
                    "worker_failures": failed,
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )

    by_rank = {int(result["rank"]): result for result in results}
    assertion_counts = {
        "shape": 0,
        "item_trace_exact": 0,
        "cross_rank_bucket": 0,
        "global_batch_unique": 0,
    }
    all_latencies: list[float] = []
    step_max_latencies: list[float] = []
    for index in range(steps):
        step = start_step + index
        buckets = []
        global_rows: list[int] = []
        rank_latencies: list[float] = []
        for rank in range(ranks):
            record = by_rank[rank]["records"][index]
            expected_hw, expected_rows_raw = plan.batch(step, rank)
            expected_rows = [int(x) for x in expected_rows_raw.tolist()]
            if record["item_rows"] != expected_rows:
                raise AssertionError(
                    f"item trace mismatch step={step},rank={rank}: "
                    f"got={record['item_rows'][:8]}, expected={expected_rows[:8]}"
                )
            if record["bucket_hw"] != list(expected_hw):
                raise AssertionError(
                    f"bucket/shape mismatch step={step},rank={rank}: "
                    f"got={record['bucket_hw']}, expected={list(expected_hw)}"
                )
            assertion_counts["shape"] += 1
            assertion_counts["item_trace_exact"] += len(expected_rows)
            buckets.append(tuple(record["bucket_hw"]))
            global_rows.extend(expected_rows)
            all_latencies.append(float(record["latency_s"]))
            rank_latencies.append(float(record["latency_s"]))
        if len(set(buckets)) != 1:
            raise AssertionError(f"cross-rank bucket mismatch step={step}: {buckets}")
        if len(set(global_rows)) != len(global_rows):
            raise AssertionError(f"cross-rank duplicate item row at step={step}")
        assertion_counts["cross_rank_bucket"] += 1
        assertion_counts["global_batch_unique"] += 1
        step_max_latencies.append(max(rank_latencies))

    wall = time.perf_counter() - wall_started
    batch_size = int(header["per_rank_bs"])
    total_images = ranks * steps * batch_size
    max_worker_wall = max(float(result["wall_s"]) for result in results)
    catalog_header = json.loads(
        (Path(catalog_root) / "header.json").read_text(encoding="utf-8")
    )
    return {
        "status": "PASS",
        "plan_id": header["plan_id"],
        "plan_hash": header["plan_hash"],
        "catalog_hash": catalog_header["catalog_hash"],
        "assignment_hash": catalog_header["assignment_hash"],
        "semantic_hash_schema": "offline_loader_semantic_hash_v1",
        "ranks": ranks,
        "per_rank_bs": batch_size,
        "start_step": start_step,
        "steps": steps,
        "total_images": total_images,
        "end_to_end_wall_s": round(wall, 6),
        "end_to_end_img_s": round(total_images / wall, 3) if wall else 0.0,
        "data_only_img_s": round(total_images / max_worker_wall, 3) if max_worker_wall else 0.0,
        "step_max_rank_latency_ms": _latency_stats_ms(step_max_latencies),
        "pooled_rank_batch_latency_ms": _latency_stats_ms(all_latencies),
        "assertions": assertion_counts,
        "per_rank": {
            str(rank): {
                "wall_s": round(float(by_rank[rank]["wall_s"]), 6),
                "img_s": round(float(by_rank[rank]["img_s"]), 3),
                "latency_ms": by_rank[rank]["latency_ms"],
                "trace_sha256": by_rank[rank]["trace_sha256"],
                "prompts_sha256": by_rank[rank]["prompts_sha256"],
                "cfg_keep_sha256": by_rank[rank]["cfg_keep_sha256"],
                "pixels_sha256": by_rank[rank]["pixels_sha256"],
                "caption_workers_anime": by_rank[rank]["caption_workers_anime"],
                "caption_workers_effective": by_rank[rank]["caption_workers_effective"],
                "buffer_fill": by_rank[rank]["buffer_fill"],
            }
            for rank in range(ranks)
        },
    }


def _jpeg_bytes(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _build_micro_fixture(root: Path) -> tuple[Path, Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from training.dataline.manifest_index import CAPTIONER_CODE
    from training.dataline_offline.planner_v1 import build_plan

    catalog_root = root / "catalog"
    plan_root = root / "plan"
    catalog_root.mkdir(parents=True)
    caption_path = root / "captions.parquet"
    caption_table = pa.table(
        {"caption_original": ["a red local test image.", "a blue local test image."]},
        schema=pa.schema([pa.field("caption_original", pa.string(), nullable=False)]),
    )
    pq.write_table(caption_table, caption_path, row_group_size=1)

    raw_images = [
        _jpeg_bytes(24, 20, (220, 20, 20)),
        _jpeg_bytes(24, 20, (20, 20, 220)),
    ]
    blob_path = root / "micro.tar"
    offsets: list[int] = []
    with blob_path.open("wb") as handle:
        for index, raw in enumerate(raw_images):
            if index:
                handle.write(b"micro-gap" * 4)
            offsets.append(handle.tell())
            handle.write(raw)

    original_code = int(CAPTIONER_CODE["original"])
    schema = pa.schema(
        [
            pa.field("row_id", pa.int64(), nullable=False),
            pa.field("uid", pa.string(), nullable=False),
            pa.field("domain", pa.string(), nullable=False),
            pa.field("bucket_idx", pa.int16(), nullable=False),
            pa.field("width", pa.int32(), nullable=False),
            pa.field("height", pa.int32(), nullable=False),
            pa.field("tar_path", pa.string(), nullable=False),
            pa.field("tar_offset", pa.int64(), nullable=False),
            pa.field("tar_size", pa.int64(), nullable=False),
            pa.field("src_key", pa.string(), nullable=False),
            pa.field("sub_source", pa.string(), nullable=False),
            pa.field("cluster_id", pa.int64()),
            pa.field("caption_type", pa.string()),
            pa.field("caption_tier", pa.int8()),
            pa.field("served_mask", pa.uint8(), nullable=False),
            pa.field("band_captioner_t1", pa.int8(), nullable=False),
            pa.field("band_captioner_t2", pa.int8(), nullable=False),
            pa.field("band_captioner_t3", pa.int8(), nullable=False),
            pa.field("band_captioner_t4", pa.int8(), nullable=False),
            pa.field("base_weight", pa.float64(), nullable=False),
            pa.field("caption_kind", pa.string(), nullable=False),
            pa.field("caption_parquet_path", pa.string(), nullable=False),
            pa.field("caption_row_group", pa.int32(), nullable=False),
            pa.field("caption_row_in_group", pa.int32(), nullable=False),
        ]
    )
    catalog_table = pa.Table.from_pydict(
        {
            "row_id": [0, 1],
            "uid": ["synthetic:micro-0", "synthetic:micro-1"],
            "domain": ["synthetic", "synthetic"],
            "bucket_idx": [0, 0],
            "width": [24, 24],
            "height": [20, 20],
            "tar_path": [str(blob_path), str(blob_path)],
            "tar_offset": offsets,
            "tar_size": [len(x) for x in raw_images],
            "src_key": ["micro-0", "micro-1"],
            "sub_source": ["micro", "micro"],
            "cluster_id": [None, None],
            "caption_type": ["original", "original"],
            "caption_tier": [1, 1],
            "served_mask": [1, 1],
            "band_captioner_t1": [original_code, original_code],
            "band_captioner_t2": [original_code, original_code],
            "band_captioner_t3": [original_code, original_code],
            "band_captioner_t4": [original_code, original_code],
            "base_weight": [1.0, 1.0],
            "caption_kind": ["original", "original"],
            "caption_parquet_path": [str(caption_path), str(caption_path)],
            "caption_row_group": [0, 1],
            "caption_row_in_group": [0, 0],
        },
        schema=schema,
    )
    catalog_path = catalog_root / "synthetic.parquet"
    pq.write_table(catalog_table, catalog_path, compression="zstd", row_group_size=1)

    domains = ["synthetic"]
    buckets = [[16, 16]]
    kept = [0]
    bucket_hash = _canonical_sha256(buckets)
    assignment_payload = {
        "bucket_table_hash": bucket_hash,
        "floor": 0.005,
        "kept": kept,
        "fold_map": [0],
        "square_only_domains": domains,
        "square_bucket_idx": 0,
        "assignment": "phase3a_vectorized_with_production_scalar_fallback_v1",
        "eligibility": "phase3a_stage256_v1",
    }
    assignment_hash = _canonical_sha256(assignment_payload)
    file_info = {
        "name": catalog_path.name,
        "rows": 2,
        "bytes": catalog_path.stat().st_size,
        "sha256": _sha256_file(catalog_path),
        "schema": str(catalog_table.schema),
        "compression": "zstd",
        "row_group_target_rows": 1,
    }
    catalog_payload = {
        "format": "offline_arb_catalog_v1",
        "domains": domains,
        "files": {"synthetic": {"rows": 2, "sha256": file_info["sha256"]}},
        "row_order": ["domain", "bucket_idx", "uid"],
        "floor": 0.005,
        "bucket_table": buckets,
        "kept_bucket_indices": kept,
    }
    catalog_hash = _canonical_sha256(catalog_payload)
    sampling_config = REPO_ROOT / "docs/采样config/t5_sampling_complete_v1.json"
    catalog_header = {
        "schema_version": "offline_arb_catalog_v1",
        "diagnostic_run": False,
        "catalog_hash": catalog_hash,
        "bucket_table_hash": bucket_hash,
        "assignment_hash": assignment_hash,
        "floor": 0.005,
        "domains": domains,
        "domain_order": domains,
        "square_only_domains": domains,
        "buckets": [{"idx": 0, "h": 16, "w": 16, "kept": True}],
        "square_bucket_idx": 0,
        "kept_bucket_indices": kept,
        "fold_map": [0],
        "row_scheme": "virtual_catalog_v1_materialized",
        "row_order": ["domain_order", "bucket_idx_0_to_94", "uid_lexicographic"],
        "row_count": 2,
        "row_group_target_rows": 1,
        "files": {"synthetic": file_info},
        "N": {"materialized_after_exclusion": [[2]]},
        "self_checks": {
            "cell_counts": {"pass": True},
            "row_id_contiguous": {"pass": True},
            "physical_order": {"pass": True},
            "tar_locator": {"pass": True},
            "uid_unique": {"pass": True},
        },
        "inputs": {
            "sampling_config": str(sampling_config),
            "sampling_config_sha256": _sha256_file(sampling_config),
        },
    }
    (catalog_root / "header.json").write_text(
        json.dumps(catalog_header, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    planner_input = {
        "format_version": 1,
        "plan_id": "offline_loader_micro_v1",
        "seed": 20260710,
        "world_size": 1,
        "per_rank_bs": 2,
        "total_steps": 2,
        "domains": domains,
        "domain_mix": {"synthetic": 1.0},
        "square_only_domains": domains,
        "buckets": buckets,
        "square_bucket_idx": 0,
        "N": [[2]],
        "R_max": {"synthetic": 2},
        "epsilon_jitter": False,
        "block_steps": 2,
        "catalog_hash": catalog_hash,
        "assignment_hash": assignment_hash,
    }
    build_plan(planner_input, plan_root)
    return plan_root, catalog_root


def run_self_test() -> dict[str, Any]:
    import training.dataline_offline.offline_dataset as offline_module
    from training.dataline_offline.offline_dataset import (
        OfflineExactDataset,
        OfflineLoaderError,
        decode_virtual_row_id,
    )
    from training.dataline_offline.planner_v1 import OfflinePlan

    if decode_virtual_row_id(0, ["synthetic"], [[2]]) != ("synthetic", 0, 0):
        raise AssertionError("row_id 0 decoder mismatch")
    if decode_virtual_row_id(1, ["synthetic"], [[2]]) != ("synthetic", 0, 1):
        raise AssertionError("row_id 1 decoder mismatch")
    # Non-uniform 2-domain x 2-bucket boundaries exercise both base[d] and
    # cellbase[d][B], including a zero-sized cell.
    boundary_n = [[2, 0], [1, 3]]
    boundary_domains = ["left", "right"]
    decoder_cases = {
        0: ("left", 0, 0),
        2: ("left", 1, 0),
        3: ("right", 1, 0),
        5: ("right", 1, 2),
    }
    for row_id, expected in decoder_cases.items():
        got = decode_virtual_row_id(row_id, boundary_domains, boundary_n)
        if got != expected:
            raise AssertionError(f"row_id {row_id} decoder mismatch: {got} != {expected}")
    with tempfile.TemporaryDirectory(prefix="offline-loader-micro-") as tmp:
        os.environ["OFFLINE_ARB_LOCAL_ROOT"] = str(Path(tmp) / "local")
        plan_root, catalog_root = _build_micro_fixture(Path(tmp))
        plan = OfflinePlan(plan_root, world_size=1, per_rank_bs=2, verify_checksums=True)
        sampling_config = REPO_ROOT / "docs/采样config/t5_sampling_complete_v1.json"
        real_resize = offline_module.resize_u8_to_bucket
        resize_calls = 0

        def counted_resize(*args, **kwargs):
            nonlocal resize_calls
            resize_calls += 1
            return real_resize(*args, **kwargs)

        offline_module.resize_u8_to_bucket = counted_resize

        # Intentionally in-process: this is the requested KB-scale mini unit,
        # independent of the production multi-process harness above.
        dataset = OfflineExactDataset(
            sampling_config=str(sampling_config),
            stage="256",
            rank=0,
            world_size=1,
            multi_ar={"assignment_mode": "offline_exact_v1"},
            offline_plan_root=str(plan_root),
            offline_catalog_root=str(catalog_root),
        )
        if not np.isclose(dataset._v_anchor_for("anime"), 0.15, rtol=0.0, atol=1e-12):
            raise AssertionError("OFFLINE_ARB_V1 anime crop anchor default is not 0.15")
        dataset.set_start_step(0)
        loader = dataset.loader(2)
        observed: list[list[int]] = []
        observed_prompts: list[list[str]] = []
        observed_cfg_keep: list[list[bool]] = []
        try:
            for step in range(2):
                batch = next(loader)
                expected_hw, expected_rows_raw = plan.batch(step, 0)
                expected_rows = [int(x) for x in expected_rows_raw.tolist()]
                trace = dict(dataset.last_item_trace or {})
                if trace.get("item_rows") != expected_rows:
                    raise AssertionError(
                        f"micro trace mismatch step={step}: {trace.get('item_rows')} != {expected_rows}"
                    )
                if list(batch["pixels"].shape) != [2, 3, expected_hw[0], expected_hw[1]]:
                    raise AssertionError(
                        f"micro batch shape mismatch step={step}: {list(batch['pixels'].shape)}"
                    )
                if len(batch["prompts"]) != 2 or len(batch["cfg_keep"]) != 2:
                    raise AssertionError("micro prompt/cfg_keep length mismatch")
                observed.append(expected_rows)
                observed_prompts.append(list(batch["prompts"]))
                observed_cfg_keep.append(list(batch["cfg_keep"]))
        finally:
            loader.close()
            dataset.close()

        # A fresh loader resumed with train.py's global_step=1 must emit plan
        # step 1, not step 2.  This catches the completed-step +1 trap.
        resumed = OfflineExactDataset(
            sampling_config=str(sampling_config),
            stage="256",
            rank=0,
            world_size=1,
            multi_ar={"assignment_mode": "offline_exact_v1"},
            offline_plan_root=str(plan_root),
            offline_catalog_root=str(catalog_root),
        )
        resumed.set_start_step(1)
        resume_loader = resumed.loader(2)
        try:
            resume_batch = next(resume_loader)
            _resume_hw, resume_rows_raw = plan.batch(1, 0)
            resume_rows = [int(x) for x in resume_rows_raw.tolist()]
            if resumed.last_item_trace["plan_step"] != 1:
                raise AssertionError("resume emitted the wrong absolute plan step")
            if resumed.last_item_trace["item_rows"] != resume_rows:
                raise AssertionError("resume item trace differs from plan step 1")
            if list(resume_batch["pixels"].shape) != [2, 3, 16, 16]:
                raise AssertionError("resume batch shape mismatch")
            if list(resume_batch["prompts"]) != observed_prompts[1]:
                raise AssertionError("stateless caption augmentation changed after resume")
            if list(resume_batch["cfg_keep"]) != observed_cfg_keep[1]:
                raise AssertionError("cfg_keep changed after resume")
        finally:
            resume_loader.close()
            resumed.close()
            offline_module.resize_u8_to_bucket = real_resize
        if resize_calls != 6:
            raise AssertionError(
                f"expected exactly one resize_u8_to_bucket call per exposure (6), got {resize_calls}"
            )

        # Corrupt one planned image payload without changing the catalog/plan.
        # Exact consumption must raise and must not substitute another row.
        with (Path(tmp) / "micro.tar").open("r+b") as handle:
            handle.seek(0)
            handle.write(b"not-a-valid-jpeg")
        bad = OfflineExactDataset(
            sampling_config=str(sampling_config),
            stage="256",
            rank=0,
            world_size=1,
            multi_ar={"assignment_mode": "offline_exact_v1"},
            offline_plan_root=str(plan_root),
            offline_catalog_root=str(catalog_root),
        )
        bad.set_start_step(0)
        bad_loader = bad.loader(2)
        bad_failed = False
        try:
            next(bad_loader)
        except OfflineLoaderError as exc:
            if "decode failed" not in str(exc):
                raise AssertionError(f"bad item failed for the wrong reason: {exc}") from exc
            bad_failed = True
        finally:
            bad_loader.close()
            bad.close()
        if not bad_failed or bad.last_item_trace is not None:
            raise AssertionError("bad item was not fail-fast or advanced the item trace")

        wrong_anchor_rejected = False
        try:
            OfflineExactDataset(
                sampling_config=str(sampling_config),
                stage="256",
                rank=0,
                world_size=1,
                multi_ar={"assignment_mode": "offline_exact_v1"},
                crop_v_anchor={"anime": 0.5},
                offline_plan_root=str(plan_root),
                offline_catalog_root=str(catalog_root),
            )
        except OfflineLoaderError as exc:
            wrong_anchor_rejected = "crop_v_anchor.anime=0.15" in str(exc)
        if not wrong_anchor_rejected:
            raise AssertionError("non-0.15 anime crop anchor was not rejected")

        catalog_path = Path(catalog_root) / "synthetic.parquet"
        with catalog_path.open("r+b") as handle:
            handle.seek(64)
            original = handle.read(1)
            if not original:
                raise AssertionError("micro catalog unexpectedly too small for SHA corruption test")
            handle.seek(64)
            handle.write(bytes([original[0] ^ 0x01]))
        catalog_sha_rejected = False
        try:
            OfflineExactDataset(
                sampling_config=str(sampling_config),
                stage="256",
                rank=0,
                world_size=1,
                multi_ar={"assignment_mode": "offline_exact_v1"},
                offline_plan_root=str(plan_root),
                offline_catalog_root=str(catalog_root),
            )
        except OfflineLoaderError as exc:
            catalog_sha_rejected = "SHA-256 mismatch" in str(exc)
        if not catalog_sha_rejected:
            raise AssertionError("equal-size catalog byte corruption was not rejected by SHA-256")

        return {
            "status": "PASS",
            "test": "synthetic_micro_plan_catalog",
            "execution": "single_process",
            "row_id_decode": "PASS",
            "batch_shape": "PASS",
            "item_trace": {
                "status": "PASS",
                "items_checked": sum(map(len, observed)) + len(resume_rows),
            },
            "resume_index_semantics": "PASS:absolute_zero_based",
            "resume_stateless_augmentation": "PASS",
            "single_resize_calls": {"status": "PASS", "calls": resize_calls},
            "bad_item_fail_fast": "PASS:no_substitution",
            "anime_crop_v_anchor": "PASS:frozen_0.15",
            "catalog_entity_sha256": "PASS:equal_size_corruption_rejected",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root")
    parser.add_argument("--catalog-root")
    parser.add_argument("--anime-freq-root", default=None)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--stage", default="256")
    parser.add_argument("--worker-timeout-s", type=float, default=3300.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and (not args.plan_root or not args.catalog_root):
        parser.error("--plan-root and --catalog-root are required unless --self-test is used")
    return args


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    try:
        if args.self_test:
            result = run_self_test()
        else:
            result = run_smoke(
                args.plan_root,
                args.catalog_root,
                args.anime_freq_root,
                args.ranks,
                args.steps,
                args.start_step,
                args.stage,
                args.worker_timeout_s,
            )
        result["harness_wall_s"] = round(time.perf_counter() - started, 6)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as exc:
        failure = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "harness_wall_s": round(time.perf_counter() - started, 6),
        }
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
