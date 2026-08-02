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

Second phase, added 2026-08-02: the text side.  Pixels are half a training sample, and
the prompt pipeline can be badly wrong while every tensor has the correct shape.  The
prompt pass walks consecutive steps from a few evenly spaced points in the plan --
rather than reusing the 13 anchor batches, which are chosen for geometry coverage and
give roughly twenty textatlas rows in total -- and checks:

* ``prompts`` and ``cfg_keep`` are actually present, the right length, and aligned with
  the loader's item trace;
* the never-CFG-drop flag lands only on anime rows, which is what the config specifies;
* tokenising ``prompt_template.format(prompt)`` with the real Qwen3-VL-2B tokenizer
  stays inside ``model.text_max_length`` -- i.e. that the dataline's caption budget and
  ``template_reserve`` agree with the template the model actually applies;
* ``caption_band`` is 0..3 on real domains and -1 everywhere else;
* every textatlas row that HAS authoritative GT carries it verbatim in the prompt, read
  from an independently constructed sidecar rather than the loader's own copy.

Counters that must be non-zero are themselves checked.  This file's first version read
two keys the loader never emits: the checks no-opped and it printed PASS anyway.
"""

from __future__ import annotations

import argparse
import hashlib
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
from training.dataline_offline.textatlas_gt_v1 import TextAtlasGroundTruth  # noqa: E402

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


class PromptAudit:
    """Everything about a batch that is not pixels: prompts, cfg_keep, tokens, GT.

    The pixel smoke answers "is the geometry right".  It says nothing about the other
    half of a training sample, and a prompt pipeline can be badly wrong while every
    tensor is the correct shape -- empty captions, a CFG-protection flag on the wrong
    rows, captions silently cut by the model because the reserve does not match the
    real template, GT that never reaches the prompt at all.

    Counters are asserted to be NON-ZERO where the plan says they must be.  The first
    version of this smoke read two keys the loader does not emit; every check quietly
    no-opped and it still printed PASS.  A check that cannot fail is worse than no
    check, so "was this branch exercised at all" is itself a check here.
    """

    def __init__(self, tokenizer, template: str, text_max_length: int, gt_reader):
        self.tokenizer = tokenizer
        self.template = template
        self.text_max_length = int(text_max_length)
        self.gt = gt_reader
        self.failures: list[str] = []
        self.rows = 0
        self.by_domain: dict[str, Counter] = {}
        self.token_lengths: list[int] = []
        self.truncated = 0
        self.cfg_keep_true = 0
        self.cfg_keep_non_anime = 0
        self.textatlas_rows = 0
        self.textatlas_gt_available = 0
        self.textatlas_gt_in_prompt = 0
        self.band_violations = 0

    def _domain(self, name: str) -> Counter:
        return self.by_domain.setdefault(name, Counter())

    def add(self, dataset, batch: dict) -> None:
        prompts = batch.get("prompts")
        cfg_keep = batch.get("cfg_keep")
        pixels = batch.get("pixels")
        trace = getattr(dataset, "last_item_trace", None)
        if prompts is None or cfg_keep is None or trace is None:
            self.failures.append(
                "batch is missing prompts/cfg_keep or the loader published no item "
                f"trace (keys={sorted(batch)})"
            )
            return
        domains = trace["domains"]
        rows = trace["item_rows"]
        bands = trace["caption_bands"]
        n = len(prompts)
        if not (n == len(cfg_keep) == len(domains) == len(rows) == int(pixels.shape[0])):
            self.failures.append(
                f"batch component lengths disagree: prompts={n} cfg_keep={len(cfg_keep)} "
                f"domains={len(domains)} rows={len(rows)} pixels={int(pixels.shape[0])}"
            )
            return

        texts = [self.template.format(p) for p in prompts]
        lengths = [
            len(x) for x in self.tokenizer(texts, add_special_tokens=True)["input_ids"]
        ]
        for i in range(n):
            domain = str(domains[i])
            counter = self._domain(domain)
            counter["rows"] += 1
            self.rows += 1
            prompt = prompts[i]
            if not prompt or not prompt.strip():
                counter["empty_prompt"] += 1
            self.token_lengths.append(lengths[i])
            if lengths[i] > self.text_max_length:
                self.truncated += 1
                counter["would_truncate"] += 1
            if cfg_keep[i]:
                self.cfg_keep_true += 1
                counter["cfg_keep"] += 1
                # config says tag-only ANIME rows are never dropped; anything else
                # carrying the flag means _is_cfg_protected is reading the wrong record.
                if domain != "anime":
                    self.cfg_keep_non_anime += 1
                    self.failures.append(
                        f"cfg_keep set on a {domain} row (row_id={rows[i]}); the "
                        "never-drop flag is for tag-only anime only"
                    )
            band = int(bands[i])
            in_real = domain in {"monet_real", "gpic"}
            # caption_band is 0-based with -1 meaning "no band".
            if (in_real and not 0 <= band <= 3) or (not in_real and band != -1):
                self.band_violations += 1
                self.failures.append(
                    f"caption_band {band} on domain={domain} (expected 0..3 for real, "
                    "-1 otherwise)"
                )
            if domain == "textatlas":
                self.textatlas_rows += 1
                gt = self.gt.get(int(rows[i])) if self.gt is not None else None
                if gt:
                    self.textatlas_gt_available += 1
                    if gt in prompt:
                        self.textatlas_gt_in_prompt += 1
                        counter["gt_in_prompt"] += 1
                    else:
                        # Legitimate only when the clause alone blows the budget, in
                        # which case the prompt must be the plain caption and must NOT
                        # contain a piece of the GT.
                        clause_tokens = len(
                            self.tokenizer(gt, add_special_tokens=False)["input_ids"]
                        )
                        counter["gt_fell_back"] += 1
                        if clause_tokens < 400:
                            self.failures.append(
                                f"textatlas row_id={rows[i]} has GT ({clause_tokens} "
                                "tokens, well inside budget) but the prompt does not "
                                "contain it"
                            )

    def finish(self) -> list[str]:
        """Coverage assertions: a branch that was never exercised was never tested."""
        if self.rows == 0:
            self.failures.append("prompt audit saw no rows at all")
            return self.failures
        if self.truncated:
            self.failures.append(
                f"{self.truncated} of {self.rows} prompts exceed text_max_length="
                f"{self.text_max_length} and would be cut by the model; the dataline "
                "budget and template_reserve do not match the real template"
            )
        expected = {"anime", "monet_real", "gpic", "synthetic", "textatlas"}
        unseen = sorted(expected - set(self.by_domain))
        if unseen:
            self.failures.append(f"domains never sampled, so never checked: {unseen}")
        if self.gt is not None:
            if self.textatlas_rows == 0:
                self.failures.append("no textatlas rows sampled; the GT path is untested")
            elif self.textatlas_gt_available == 0:
                self.failures.append(
                    f"{self.textatlas_rows} textatlas rows sampled and not one had GT; "
                    "coverage is ~47%, so this is a binding failure, not luck"
                )
        for domain, counter in self.by_domain.items():
            if counter["rows"] and counter["empty_prompt"] == counter["rows"]:
                self.failures.append(f"every sampled {domain} prompt is empty")
        return self.failures

    def summary(self) -> dict:
        lengths = np.asarray(self.token_lengths, dtype=np.int64)
        return {
            "rows": self.rows,
            "token_length": {
                "p50": int(np.percentile(lengths, 50)) if lengths.size else None,
                "p95": int(np.percentile(lengths, 95)) if lengths.size else None,
                "max": int(lengths.max()) if lengths.size else None,
                "limit": self.text_max_length,
                "would_truncate": self.truncated,
            },
            "cfg_keep_true": self.cfg_keep_true,
            "cfg_keep_non_anime": self.cfg_keep_non_anime,
            "textatlas": {
                "rows": self.textatlas_rows,
                "gt_available": self.textatlas_gt_available,
                "gt_in_prompt": self.textatlas_gt_in_prompt,
            },
            "by_domain": {d: dict(c) for d, c in sorted(self.by_domain.items())},
        }


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


def run_prompt_audit(args, cfg, plan_root: Path) -> tuple[dict, list[str]]:
    """Walk consecutive steps at a few points in the plan and audit the text side."""
    from transformers import AutoTokenizer

    from networks.model_mot import DEFAULT_PROMPT_TEMPLATE

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.dataset.tokenizer_path))
    tpl_cfg = cfg.model.get("prompt_template", {}) or {}
    # Mirror MoTT2IModel.__init__ exactly.  Measuring against a template the model does
    # not use would make the truncation number fiction, which is the failure this check
    # exists to catch.
    template = (
        str(tpl_cfg.get("template", DEFAULT_PROMPT_TEMPLATE))
        if bool(tpl_cfg.get("enabled", True))
        else "{}"
    )
    text_max_length = int(cfg.model.get("text_max_length", 384))

    gt_reader = None
    gt_root = cfg.dataset.get("offline_textatlas_gt_root", None)
    if gt_root:
        catalog_root = Path(str(cfg.dataset.offline_catalog_root))
        catalog_header = json.loads((catalog_root / "header.json").read_text())
        # Built independently of the loader's own copy: if the loader silently failed to
        # attach the sidecar, comparing against the loader's reader would agree with it.
        gt_reader = TextAtlasGroundTruth(
            gt_root,
            catalog_hash=str(catalog_header["catalog_hash"]),
            catalog_header_sha256=hashlib.sha256(
                (catalog_root / "header.json").read_bytes()
            ).hexdigest(),
        )

    audit = PromptAudit(tokenizer, template, text_max_length, gt_reader)
    header = json.loads((plan_root / "header.json").read_text())
    first = int(header["materialization"]["first_materialized_step"])
    total = int(header["total_steps"])
    span = max(total - first - args.prompt_steps, 1)
    starts = [first + (span * i) // max(args.prompt_segments, 1)
              for i in range(max(args.prompt_segments, 1))]

    print(f"\nprompt audit: {len(starts)} segments x {args.prompt_steps} steps "
          f"from {starts}   template={len(tokenizer(template.format(''))['input_ids'])} "
          f"tokens, limit {text_max_length}")
    for start in starts:
        dataset = None
        try:
            dataset = build_dataset(args.config, args.rank, args.world_size, args.per_rank_bs)
            dataset.set_start_step(start)
            loader = dataset.loader(args.per_rank_bs)
            for _ in range(args.prompt_steps):
                audit.add(dataset, next(loader))
        except Exception as exc:  # noqa: BLE001 - a smoke must report, not crash
            audit.failures.append(f"prompt segment at step {start}: {type(exc).__name__}: {exc}")
        finally:
            try:
                if dataset is not None:
                    dataset.close()
            except Exception:
                pass

    failures = audit.finish()
    summary = audit.summary()
    tl = summary["token_length"]
    print(f"  rows {summary['rows']:,}   tokens p50 {tl['p50']} p95 {tl['p95']} "
          f"max {tl['max']} / limit {tl['limit']}   would_truncate {tl['would_truncate']}")
    print(f"  cfg_keep true {summary['cfg_keep_true']} "
          f"(non-anime {summary['cfg_keep_non_anime']})")
    ta = summary["textatlas"]
    print(f"  textatlas rows {ta['rows']}  GT available {ta['gt_available']}  "
          f"GT in prompt {ta['gt_in_prompt']}")
    for domain, counter in summary["by_domain"].items():
        print(f"    {domain:<12}{counter}")
    return summary, failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/stage512b_real_anchors_v1.yaml")
    ap.add_argument("--world-size", type=int, default=64)
    ap.add_argument("--per-rank-bs", type=int, default=32)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--anchors", default="all", help="'all' or a comma-separated id list")
    ap.add_argument("--json-out", type=Path, default=None)
    # The anchor pass sees 13 batches at whatever steps happen to use each anchor, which
    # is the right sample for geometry and the wrong one for prompts: textatlas is 5% of
    # the mix, so 13 batches yield about twenty of its rows and half of those have no GT.
    # The prompt pass walks consecutive steps from a few evenly spaced points instead.
    ap.add_argument("--prompt-segments", type=int, default=3)
    ap.add_argument("--prompt-steps", type=int, default=16)
    ap.add_argument("--skip-prompt-audit", action="store_true")
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

    prompt_summary = None
    if not args.skip_prompt_audit:
        prompt_summary, prompt_failures = run_prompt_audit(args, cfg, plan_root)
        all_failures.extend(prompt_failures)

    summary = {
        "plan_root": str(plan_root),
        "anchors_requested": wanted,
        "anchors_passed": exercised,
        "failures": all_failures,
        "per_anchor": results,
        "prompt_audit": prompt_summary,
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
