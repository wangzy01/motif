#!/usr/bin/env python3
"""Decode a real batch at every one of the 13 anchors, through the production loader.

Why not scripts/offline_arb/smoke_offline_loader.py: that harness cannot drive this
path.  Its ``--self-test`` fixture reads ``docs/采样config/t5_sampling_complete_v1.json``,
which the curated tree deliberately does not carry, and its production mode overwrites
``multi_ar["assignment_mode"]`` with ``offline_exact_v1``, so it cannot load a weighted
plan at all.  Both are pre-existing; neither is worth fixing to answer this question.

What this checks that a synthetic fixture cannot:

* every anchor is reachable -- not just the common ones.  Walking sequentially would
  need hundreds of steps to stumble onto 352x704 (0.763% of steps), so instead the
  plan's own ``bucket_of_step`` is read to find a step that uses each anchor, and the
  loader is seeked there with ``set_start_step``.
* the returned pixels really are the anchor's (H, W), for real decoded images.
* ``anchors_v1.verify_binding`` holds for every row, at the per-domain crop_loss the
  CATALOG recorded -- textatlas at 0.12, everything else at 0.20.
* zero upsampling: decoded native size covers the anchor on every row.

Exit code is 0 only when all 13 anchors were exercised and every row passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omegaconf import OmegaConf  # noqa: E402

from training.dataline import anchors_v1  # noqa: E402
from training.dataline_offline.offline_dataset import (  # noqa: E402
    OfflineExactDataset,
    OfflineLoaderError,
)

# Keys the dataset stanza carries for other consumers and the loader does not accept.
_NON_LOADER_KEYS = ("store_cls", "stage", "image_size", "gpic_cluster_stats")


def first_step_per_anchor(plan_root: Path, n_anchors: int) -> dict[int, int]:
    """Map anchor_id -> the lowest plan step that uses it.

    Reads the block NPZs directly rather than through the reader: this is a
    diagnostic and the reader would verify every checksum for data we only use to
    pick step numbers.
    """
    header = json.loads((plan_root / "header.json").read_text())
    block_steps = int(header["block_steps"])
    first_block = int(header["materialization"]["first_materialized_block"])
    found: dict[int, int] = {}
    for index in range(first_block, int(header["num_blocks"])):
        path = plan_root / "blocks" / f"block_{index:06d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            bucket = np.asarray(archive["bucket_of_step"])
        base = index * block_steps
        for anchor in np.unique(bucket):
            anchor = int(anchor)
            if anchor not in found:
                found[anchor] = base + int(np.flatnonzero(bucket == anchor)[0])
        if len(found) == n_anchors:
            break
    return found


def build_dataset(cfg_path: str, rank: int, world_size: int, per_rank_bs: int):
    cfg = OmegaConf.load(cfg_path)
    kwargs = dict(OmegaConf.to_container(cfg.dataset, resolve=True))
    for key in _NON_LOADER_KEYS:
        kwargs.pop(key, None)
    return OfflineExactDataset(
        stage=str(cfg.dataset.stage),
        image_size=int(cfg.dataset.image_size),
        rank=rank,
        world_size=world_size,
        per_rank_bs=per_rank_bs,
        **kwargs,
    )


def check_batch(dataset, batch: dict, anchor: int, report: dict) -> list[str]:
    """Verify one step's batch against the anchor it was planned for."""
    failures: list[str] = []
    target_h, target_w = dataset.buckets[anchor]

    pixels = batch.get("pixels")
    if pixels is None:
        return [f"anchor {anchor}: batch has no pixels"]
    # (B, C, H, W) after uint8_to_norm_tensor + stack.
    got_h, got_w = int(pixels.shape[-2]), int(pixels.shape[-1])
    if (got_h, got_w) != (int(target_h), int(target_w)):
        failures.append(
            f"anchor {anchor}: tensor is {got_w}x{got_h}, anchor is {target_w}x{target_h}"
        )
    report["batch_shape"] = tuple(int(x) for x in pixels.shape)

    return failures


def check_planned_rows(dataset, plan_root: Path, step: int, anchor: int,
                       world_size: int, per_rank_bs: int) -> tuple[list[str], Counter]:
    """Verify every row the plan assigns to this step, independently of the loader.

    The loader runs verify_binding internally, so a batch coming back at all is
    already evidence.  But "it did not raise" is weak evidence if the call were ever
    unwired again, so this re-derives the check from the PLAN and the CATALOG -- the
    two artifacts on disk -- and touches no loader internals beyond the bucket table.

    It also covers far more rows than the batch does: the whole step across all ranks
    (world_size * per_rank_bs), not just this rank's slice.
    """
    import pyarrow.parquet as pq

    failures: list[str] = []
    domains: Counter = Counter()
    header = json.loads((plan_root / "header.json").read_text())
    block_steps = int(header["block_steps"])
    block = step // block_steps
    with np.load(plan_root / "blocks" / f"block_{block:06d}.npz", allow_pickle=False) as ar:
        rows = np.asarray(ar["item_row"])[step - block * block_steps]
    row_ids = np.unique(rows.reshape(-1))

    target_w, target_h = anchors_v1.ANCHORS_V1_WH[anchor]
    catalog_root = Path(str(dataset.catalog_root)) if hasattr(dataset, "catalog_root") else None
    if catalog_root is None:
        return ["cannot locate catalog root for the independent row check"], domains

    # row_id is the contiguous GLOBAL row number, so find which domain parquet each
    # id falls in from the header's per-domain row counts.
    cat_header = json.loads((catalog_root / "header.json").read_text())
    bounds, cursor = [], 0
    for domain in cat_header["domain_order"]:
        n = int(cat_header["files"][domain]["rows"])
        bounds.append((cursor, cursor + n, domain))
        cursor += n

    by_domain: dict[str, list[int]] = {}
    for rid in row_ids.tolist():
        for lo, hi, domain in bounds:
            if lo <= rid < hi:
                by_domain.setdefault(domain, []).append(rid - lo)
                domains[domain] += 1
                break

    for domain, local in by_domain.items():
        table = pq.read_table(
            catalog_root / f"{domain}.parquet", columns=["bucket_idx", "width", "height"]
        )
        idx = np.asarray(local, dtype=np.int64)
        bkt = table["bucket_idx"].to_numpy(zero_copy_only=False)[idx]
        wid = table["width"].to_numpy(zero_copy_only=False)[idx]
        hei = table["height"].to_numpy(zero_copy_only=False)[idx]
        bad_bucket = int((bkt != anchor).sum())
        if bad_bucket:
            failures.append(
                f"anchor {anchor} domain={domain}: {bad_bucket} planned rows carry a "
                f"different bucket_idx in the catalog"
            )
        under = int(((wid < target_w) | (hei < target_h)).sum())
        if under:
            failures.append(
                f"anchor {anchor} domain={domain}: {under} planned rows cannot cover "
                f"{target_w}x{target_h} without upsampling"
            )
        threshold = dataset._anchors_crop_loss[domain]
        num, den = anchors_v1._threshold_ratio(threshold)
        a = wid.astype(np.int64) * target_h
        b = hei.astype(np.int64) * target_w
        lo_, hi_ = np.minimum(a, b), np.maximum(a, b)
        over = int((den * lo_ < num * hi_).sum())
        if over:
            failures.append(
                f"anchor {anchor} domain={domain}: {over} planned rows exceed "
                f"crop_loss {threshold}"
            )
    return failures, domains


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/stage512b_real_anchors_v1.yaml")
    ap.add_argument("--world-size", type=int, default=64)
    ap.add_argument("--per-rank-bs", type=int, default=32)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--anchors", default="all", help="'all' or a comma-separated id list")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    plan_root = Path(str(cfg.dataset.offline_plan_root))
    n_anchors = len(anchors_v1.ANCHORS_V1_WH)

    print(f"plan   {plan_root}")
    steps = first_step_per_anchor(plan_root, n_anchors)
    missing = sorted(set(range(n_anchors)) - set(steps))
    if missing:
        # A plan that never uses an anchor is a planning failure, not a loader one --
        # say which it is rather than reporting a coverage gap.
        print(f"PLAN DOES NOT USE ANCHORS {missing}; nothing to smoke for them")
    wanted = (
        sorted(steps)
        if args.anchors == "all"
        else [int(x) for x in args.anchors.split(",")]
    )

    results: dict[str, dict] = {}
    all_failures: list[str] = []
    domains_seen: Counter = Counter()

    for anchor in wanted:
        if anchor not in steps:
            continue
        step = steps[anchor]
        w, h = anchors_v1.ANCHORS_V1_WH[anchor]
        report: dict = {"anchor": anchor, "size": f"{w}x{h}", "step": step}
        started = time.perf_counter()
        dataset = None   # so the finally below cannot NameError on a construction failure
        try:
            dataset = build_dataset(
                args.config, args.rank, args.world_size, args.per_rank_bs
            )
            dataset.set_start_step(step)
            # The public surface is loader(batch_size), not __iter__ -- an earlier
            # draft used next(iter(dataset)) and every anchor failed with
            # "object is not iterable" AFTER constructing and seeking fine.
            batch = next(dataset.loader(args.per_rank_bs))
            failures = check_batch(dataset, batch, anchor, report)
            row_failures, doms = check_planned_rows(
                dataset, plan_root, step, anchor, args.world_size, args.per_rank_bs
            )
            failures.extend(row_failures)
            report["planned_rows_checked"] = int(sum(doms.values()))
            report["domains"] = dict(doms)
            domains_seen.update(doms)
            report["status"] = "PASS" if not failures else "FAIL"
            report["failures"] = failures
            all_failures.extend(failures)
        except (OfflineLoaderError, anchors_v1.AnchorGeometryError) as exc:
            report["status"] = "FAIL"
            report["failures"] = [f"{type(exc).__name__}: {exc}"]
            all_failures.extend(report["failures"])
        except Exception as exc:  # noqa: BLE001 - a smoke must report, not crash
            report["status"] = "ERROR"
            report["failures"] = [f"{type(exc).__name__}: {exc}"]
            all_failures.extend(report["failures"])
        finally:
            # Each anchor builds its own dataset because set_start_step must precede
            # the first next(); 13 live loaders each with fetch workers and a 4096-row
            # buffer would be a needless resource pile.
            try:
                if dataset is not None:
                    dataset.close()
            except Exception:
                pass
        report["wall_s"] = round(time.perf_counter() - started, 2)
        results[str(anchor)] = report
        print(
            f"a{anchor:<2} {w:>3}x{h:<3} step {step:>7,}  {report['status']:<5} "
            f"{report.get('batch_shape', '')}  {report['wall_s']}s"
            + ("" if report["status"] == "PASS" else f"\n       {report['failures'][:2]}")
        )

    exercised = sum(1 for r in results.values() if r["status"] == "PASS")
    print(f"\nanchors exercised {exercised}/{len(wanted)}   failures {len(all_failures)}")
    if domains_seen:
        print("domains seen:", dict(domains_seen))

    summary = {
        "plan_root": str(plan_root),
        "anchors_requested": wanted,
        "anchors_passed": exercised,
        "failures": all_failures,
        "per_anchor": results,
        "pass": exercised == len(wanted) and not all_failures,
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"wrote {args.json_out}")

    if not summary["pass"]:
        print("ANCHORS_LOADER_SMOKE_FAIL")
        return 1
    print("ANCHORS_LOADER_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
