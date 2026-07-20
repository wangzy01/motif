#!/usr/bin/env python3
"""Full independent hard-gate audit for a materialized semantic plan.

The audit scans every stage slot and separately closes the formal training
prefix.  It writes ``PLAN_AUDIT_SUCCESS.json`` only after every assertion
passes.  Production execution is a CPU JCC operation (large sequential I/O).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from stage256_20260718.semantic_plan import (
    AUDIT_FORMAT,
    FORMAL_W64_ACTIVE_BUCKET_INDICES,
    FORMAL_W64_PREFIX_DOMAIN_TOTALS,
    SUCCESS_NAME,
    SemanticOfflinePlan,
    SemanticPlanError,
    load_overlay_cell,
    load_overlay_header,
    resolve_bound_root,
    sha256_file,
    write_json_create_once,
)


EVAL_AR_TARGETS = (
    (1, 4),
    (1, 3),
    (1, 2),
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
)


def _load_quota(
    plan: SemanticOfflinePlan,
    filename: str,
    info: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    path = plan.plan_root / "quotas" / filename
    expected = plan.quota_checksums[filename]
    actual = sha256_file(path)
    if actual != expected:
        raise SemanticPlanError(
            f"quota checksum mismatch for {filename}: expected={expected}, actual={actual}"
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"row_id", "quota"}:
                raise SemanticPlanError(f"quota file {filename} has unexpected arrays")
            row_ids = np.array(archive["row_id"], copy=True)
            quota = np.array(archive["quota"], copy=True)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SemanticPlanError(f"cannot load quota file {path}: {exc}") from exc
    rows = int(info["rows"])
    if row_ids.dtype != np.dtype(np.int64) or row_ids.shape != (rows,):
        raise SemanticPlanError(f"quota row_id dtype/shape mismatch in {filename}")
    if quota.dtype != np.dtype(np.int32) or quota.shape != (rows,):
        raise SemanticPlanError(f"quota dtype/shape mismatch in {filename}")
    if np.any(quota < 0) or int(quota.sum(dtype=np.int64)) != int(info["exposures"]):
        raise SemanticPlanError(f"quota conservation failure in {filename}")
    return row_ids, quota


def _nearest_eval_ar_coverage(
    buckets: Sequence[Sequence[int]], observed_steps: np.ndarray
) -> list[dict[str, Any]]:
    active = [i for i, steps in enumerate(observed_steps.tolist()) if int(steps) > 0]
    if not active:
        raise SemanticPlanError("no active native bucket in audited schedule")
    coverage: list[dict[str, Any]] = []
    for numerator, denominator in EVAL_AR_TARGETS:
        target = float(numerator) / float(denominator)
        best = min(
            active,
            key=lambda b: (
                abs(math.log((float(buckets[b][1]) / float(buckets[b][0])) / target)),
                b,
            ),
        )
        # Buckets are stored as [H,W]; evaluation AR labels are W:H.
        h, w = (int(value) for value in buckets[best])
        nearest = float(w) / float(h)
        coverage.append(
            {
                "eval_width_to_height": f"{numerator}:{denominator}",
                "nearest_active_bucket_idx": best,
                "nearest_active_bucket_hw": [h, w],
                "nearest_width_to_height": nearest,
                "log_ratio_distance": abs(math.log(nearest / target)),
                "training_steps": int(observed_steps[best]),
                "note": "evaluation target only; not represented as an invented training bucket",
            }
        )
    return coverage


def audit(
    *,
    plan_root: Path,
    prefix_steps: int,
    expected_prefix_exposures: int,
    expected_active_buckets: int,
    expected_exclusions: int,
) -> dict[str, Any]:
    plan = SemanticOfflinePlan(plan_root, require_audit=False, verify_checksums=True)
    header = plan.header
    H = int(header["total_steps"])
    G = int(header["G"])
    W = int(header["world_size"])
    bs = int(header["per_rank_bs"])
    B = len(header["buckets"])
    K = len(header["domains"])
    if prefix_steps <= 0 or prefix_steps > H:
        raise SemanticPlanError(f"prefix_steps must be in [1,{H}]")
    contract = header["audit_contract"]
    contract_expectations = (
        (prefix_steps, contract["formal_prefix_steps"], "formal prefix steps"),
        (
            expected_prefix_exposures,
            contract["formal_prefix_exposures"],
            "formal prefix exposures",
        ),
        (
            expected_active_buckets,
            contract["active_native_buckets"],
            "active native buckets",
        ),
        (expected_exclusions, contract["excluded_rows"], "excluded rows"),
    )
    for requested, bound, label in contract_expectations:
        if requested != bound:
            raise SemanticPlanError(
                f"audit CLI {label}={requested} differs from plan-bound contract {bound}"
            )
    if prefix_steps * G != expected_prefix_exposures:
        raise SemanticPlanError(
            f"prefix contract mismatch: {prefix_steps}*{G}={prefix_steps*G}, "
            f"expected {expected_prefix_exposures}"
        )
    exclusions = np.asarray(header["excluded_row_ids"], dtype=np.int64)
    if exclusions.size != expected_exclusions:
        raise SemanticPlanError(
            f"expected {expected_exclusions} exclusions, semantic plan binds {exclusions.size}"
        )

    overlay_binding = header["bindings"]["overlay"]
    overlay_root = resolve_bound_root(
        plan.plan_root, overlay_binding["root"], "bindings.overlay.root"
    )
    overlay = load_overlay_header(overlay_root)
    if overlay["semantic_assertions"]["anime_all_rows_strip_nonempty_nl"] is not True:
        raise SemanticPlanError("anime stripped-NL overlay assertion is not true")
    if int(overlay["semantic_assertions"]["anime_tag_only_rows"]) != 0:
        raise SemanticPlanError("anime tag-only overlay assertion is not zero")

    n_select = np.asarray(header["N_select"], dtype=np.int64)
    expected_C = np.asarray(header["C"], dtype=np.int64)
    expected_S = np.asarray(header["S"], dtype=np.int64)
    expected_D = np.asarray(header["D"], dtype=np.int64)
    rmax = np.asarray(
        [int(header["R_max"][name]) for name in header["domains"]],
        dtype=np.int64,
    )

    cell_rows: dict[tuple[int, int], np.ndarray] = {}
    cell_quota: dict[tuple[int, int], np.ndarray] = {}
    cell_actual: dict[tuple[int, int], np.ndarray] = {}
    quota_name_by_cell: dict[tuple[int, int], str] = {}
    for filename, info in header["quota_files"].items():
        b, d = int(info["bucket"]), int(info["domain"])
        key = (b, d)
        if key in quota_name_by_cell:
            raise SemanticPlanError(f"duplicate quota file for cell {key}")
        quota_name_by_cell[key] = filename
    for b in range(B):
        for d in range(K):
            rows = int(n_select[b, d])
            exposures = int(expected_C[b, d])
            if rows == 0 and exposures == 0:
                continue
            filename = quota_name_by_cell.get((b, d))
            if filename is None:
                raise SemanticPlanError(f"missing quota file for cell ({b},{d})")
            info = header["quota_files"][filename]
            overlay_rows, _weights = load_overlay_cell(
                overlay_root, overlay, b, d, verify_checksum=True
            )
            quota_rows, quota = _load_quota(plan, filename, info)
            if not np.array_equal(overlay_rows, quota_rows):
                raise SemanticPlanError(f"quota/overlay row mismatch in cell ({b},{d})")
            if exclusions.size and np.any(np.isin(overlay_rows, exclusions)):
                raise SemanticPlanError(f"overlay includes an exclusion in cell ({b},{d})")
            if quota.size and int(quota.max()) > int(rmax[d]):
                raise SemanticPlanError(f"quota exceeds R_max in cell ({b},{d})")
            cell_rows[(b, d)] = overlay_rows
            cell_quota[(b, d)] = quota
            cell_actual[(b, d)] = np.zeros(quota.shape, dtype=np.int32)

    observed_C = np.zeros((B, K), dtype=np.int64)
    prefix_C = np.zeros((B, K), dtype=np.int64)
    observed_S = np.zeros(B, dtype=np.int64)
    prefix_S = np.zeros(B, dtype=np.int64)
    audited_slots = 0
    prefix_slots = 0
    anime_index = (
        list(header["domains"]).index("anime") if "anime" in header["domains"] else None
    )
    anime_slots = 0
    for block_index in range(plan.first_materialized_block, int(header["num_blocks"])):
        bucket, items, counts = plan._load_block(block_index)  # noqa: SLF001
        block_start = block_index * int(header["block_steps"])
        for local_step in range(bucket.shape[0]):
            step = block_start + local_step
            b = int(bucket[local_step])
            flat = items[local_step].T.reshape(-1)
            if flat.shape != (G,) or np.unique(flat).size != G:
                raise SemanticPlanError(f"global batch uniqueness failure at step {step}")
            if exclusions.size and np.any(np.isin(flat, exclusions)):
                raise SemanticPlanError(f"excluded row observed at step {step}")
            row_counts = counts[local_step].astype(np.int64, copy=False)
            if int(row_counts.sum(dtype=np.int64)) != G:
                raise SemanticPlanError(f"domain_counts do not sum to G at step {step}")
            observed_S[b] += 1
            observed_C[b] += row_counts
            audited_slots += G
            in_prefix = step < prefix_steps
            if in_prefix:
                prefix_S[b] += 1
                prefix_C[b] += row_counts
                prefix_slots += G
            boundaries = np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(row_counts, dtype=np.int64))
            )
            for d in range(K):
                lo, hi = int(boundaries[d]), int(boundaries[d + 1])
                if lo == hi:
                    continue
                values = flat[lo:hi]
                rows = cell_rows.get((b, d))
                if rows is None:
                    raise SemanticPlanError(
                        f"plan uses cell ({b},{d}) absent from semantic overlay"
                    )
                indices = np.searchsorted(rows, values)
                safe = np.minimum(indices, max(0, int(rows.size) - 1))
                if (
                    np.any(indices >= rows.size)
                    or rows.size == 0
                    or np.any(rows[safe] != values)
                ):
                    raise SemanticPlanError(
                        f"overlay membership failure at step={step},cell=({b},{d})"
                    )
                np.add.at(cell_actual[(b, d)], indices, 1)
                if anime_index is not None and d == anime_index:
                    anime_slots += hi - lo

    if audited_slots != H * G:
        raise SemanticPlanError(
            f"full-stage exposure mismatch: observed={audited_slots}, expected={H*G}"
        )
    if prefix_slots != expected_prefix_exposures:
        raise SemanticPlanError(
            f"formal prefix exposure mismatch: observed={prefix_slots}, expected={expected_prefix_exposures}"
        )
    observed_prefix_domain_totals = {
        name: int(prefix_C[:, d].sum(dtype=np.int64))
        for d, name in enumerate(header["domains"])
    }
    if observed_prefix_domain_totals != header["audit_contract"][
        "formal_prefix_domain_totals"
    ]:
        raise SemanticPlanError(
            "audited formal prefix domain totals differ from the plan-bound contract"
        )
    if W == 64 and observed_prefix_domain_totals != FORMAL_W64_PREFIX_DOMAIN_TOTALS:
        raise SemanticPlanError(
            "formal W64 prefix domain totals differ from the frozen values: "
            f"observed={observed_prefix_domain_totals}, "
            f"expected={FORMAL_W64_PREFIX_DOMAIN_TOTALS}"
        )
    if not np.array_equal(observed_C, expected_C):
        raise SemanticPlanError("audited full-stage domain/bucket matrix differs from C")
    if not np.array_equal(observed_S, expected_S):
        raise SemanticPlanError("audited bucket schedule differs from S")
    if not np.array_equal(observed_C.sum(axis=0), expected_D):
        raise SemanticPlanError("audited domain totals differ from D")
    if int(observed_S.sum(dtype=np.int64)) != H:
        raise SemanticPlanError("audited bucket steps do not sum to total_steps")

    max_repeat_by_domain = np.zeros(K, dtype=np.int64)
    rows_with_exposure = 0
    for (b, d), actual in cell_actual.items():
        quota = cell_quota[(b, d)]
        if not np.array_equal(actual, quota):
            mismatch = np.flatnonzero(actual != quota)
            first = int(mismatch[0])
            raise SemanticPlanError(
                f"actual exposure differs from exact quota cell=({b},{d}),index={first},"
                f"actual={actual[first]},quota={quota[first]}"
            )
        if actual.size:
            max_repeat_by_domain[d] = max(
                int(max_repeat_by_domain[d]), int(actual.max())
            )
            rows_with_exposure += int(np.count_nonzero(actual))
            if int(actual.max()) > int(rmax[d]):
                raise SemanticPlanError(f"actual row repeat exceeds R_max in cell ({b},{d})")

    active_indices = np.flatnonzero(observed_S > 0)
    if int(active_indices.size) != expected_active_buckets:
        raise SemanticPlanError(
            f"active native bucket count={active_indices.size}, expected={expected_active_buckets}"
        )
    expected_active_indices = (
        list(FORMAL_W64_ACTIVE_BUCKET_INDICES)
        if W == 64
        else list(header["audit_contract"]["active_native_bucket_indices"])
    )
    if active_indices.tolist() != expected_active_indices:
        raise SemanticPlanError(
            "active native bucket indices differ from the bound/formal contract: "
            f"observed={active_indices.tolist()}, expected={expected_active_indices}"
        )
    active_native_buckets = [
        {
            "bucket_idx": int(b),
            "hw": [int(v) for v in header["buckets"][int(b)]],
            "steps": int(observed_S[int(b)]),
            "domain_exposures": {
                name: int(observed_C[int(b), d])
                for d, name in enumerate(header["domains"])
            },
        }
        for b in active_indices.tolist()
    ]
    eval_coverage = _nearest_eval_ar_coverage(header["buckets"], observed_S)

    receipt = {
        "format": AUDIT_FORMAT,
        "status": "PASS",
        "bindings": {
            "plan_hash": header["plan_hash"],
            "plan_header_sha256": sha256_file(plan.plan_root / "header.json"),
            "plan_checksums_sha256": sha256_file(plan.plan_root / "checksums.json"),
            "overlay_header_sha256": sha256_file(overlay_root / "header.json"),
            "audit_script_sha256": sha256_file(Path(__file__)),
        },
        "assertions": {
            "all_block_checksums": True,
            "global_batch_unique": True,
            "physical_cell_ranges": True,
            "overlay_membership": True,
            "exclusions_zero": True,
            "repeat_caps": True,
            "domain_bucket_matrix": True,
            "native_aspect_ratios": True,
            "anime_tag_only_zero": True,
            "exact_full_stage_quota": True,
            "formal_prefix": True,
            "formal_prefix_domain_totals": True,
        },
        "full_stage": {
            "steps": H,
            "exposures": audited_slots,
            "domain_totals": {
                name: int(observed_C[:, d].sum(dtype=np.int64))
                for d, name in enumerate(header["domains"])
            },
            "max_repeat_by_domain": {
                name: int(max_repeat_by_domain[d])
                for d, name in enumerate(header["domains"])
            },
            "rows_with_nonzero_exposure": rows_with_exposure,
            "anime_exposures": anime_slots,
            "anime_tag_only_exposures": 0,
            "excluded_row_exposures": 0,
        },
        "formal_prefix": {
            "steps": prefix_steps,
            "exposures": prefix_slots,
            "expected_exposures": expected_prefix_exposures,
            "domain_totals": observed_prefix_domain_totals,
            "bucket_steps": prefix_S.tolist(),
            "domain_bucket_exposures": prefix_C.tolist(),
        },
        "multi_ar": {
            "active_native_bucket_count": int(active_indices.size),
            "active_native_bucket_indices": active_indices.tolist(),
            "active_native_buckets": active_native_buckets,
            "evaluation_targets_nearest_active_training_bucket": eval_coverage,
            "caveat": (
                "The seven evaluation ARs are reporting targets only. The semantic plan "
                "preserves the selection plan's 20 active native HxW buckets exactly."
            ),
        },
        "exclusions": {
            "expected_count": expected_exclusions,
            "row_ids": [int(v) for v in exclusions.tolist()],
            "observed_exposures": 0,
        },
    }
    success_path = plan.plan_root / SUCCESS_NAME
    if success_path.exists():
        existing = json.loads(success_path.read_text(encoding="utf-8"))
        if existing != receipt:
            raise SemanticPlanError(
                f"existing {SUCCESS_NAME} differs from freshly recomputed audit"
            )
    else:
        write_json_create_once(success_path, receipt)
    # Final production-mode reopen proves the receipt itself closes the reader.
    SemanticOfflinePlan(plan.plan_root, require_audit=True, verify_checksums=True)
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prefix-steps", type=int, default=166_667)
    parser.add_argument("--expected-prefix-exposures", type=int, default=384_000_768)
    parser.add_argument("--expected-active-buckets", type=int, default=20)
    parser.add_argument("--expected-exclusions", type=int, default=11)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        receipt = audit(
            plan_root=args.plan,
            prefix_steps=args.prefix_steps,
            expected_prefix_exposures=args.expected_prefix_exposures,
            expected_active_buckets=args.expected_active_buckets,
            expected_exclusions=args.expected_exclusions,
        )
    except BaseException as exc:
        print(f"SEMANTIC_PLAN_AUDIT_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "plan_hash": receipt["bindings"]["plan_hash"],
                "full_exposures": receipt["full_stage"]["exposures"],
                "prefix_exposures": receipt["formal_prefix"]["exposures"],
                "active_native_buckets": receipt["multi_ar"]["active_native_bucket_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
