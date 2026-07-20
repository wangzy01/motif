#!/usr/bin/env python3
"""Materialize a physical-row semantic plan from a legacy selection plan.

Heavy production use belongs in a CPU JCC job.  The output is create-once: a
complete matching product is accepted idempotently; a partial/non-matching
directory is preserved and rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from stage256_20260718.semantic_plan import (
    ALGORITHM_VERSION,
    BLOCK_DATASETS,
    BLOCK_FORMAT,
    FORMAL_W64_ACTIVE_BUCKET_INDICES,
    FORMAL_W64_PREFIX_DOMAIN_TOTALS,
    PLAN_FORMAT,
    PLAN_FORMAT_VERSION,
    QUOTA_ALGORITHM,
    TICKET_ALGORITHM,
    SemanticOfflinePlan,
    SemanticPlanError,
    binding_for_plan,
    canonical_json_bytes,
    cell_key,
    load_overlay_cell,
    load_overlay_header,
    physical_cell_offsets,
    read_json_object,
    sha256_bytes,
    sha256_file,
    write_deterministic_npz_create_once,
    write_json_create_once,
)
from training.dataline_offline.planner_v1 import OfflinePlan, PlannerInputError


HAVEL_FALLBACK_MAX_TICKETS = 2_000_000


def _seed_for_cell(
    plan_id: str,
    selection_plan_hash: str,
    overlay_header_sha256: str,
    bucket: int,
    domain: int,
    layer: int,
) -> int:
    payload = canonical_json_bytes(
        {
            "namespace": TICKET_ALGORITHM,
            "plan_id": plan_id,
            "selection_plan_hash": selection_plan_hash,
            "overlay_header_sha256": overlay_header_sha256,
            "bucket": int(bucket),
            "domain": int(domain),
            "layer": int(layer),
        }
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def capped_largest_remainder(
    total: int,
    row_ids: np.ndarray,
    weights: np.ndarray,
    cap: int,
) -> np.ndarray:
    """Exact bounded apportionment using float64 water-filling + LR ties.

    Real-valued targets are ``min(cap, lambda * weight)``.  Integer floors are
    completed by descending fractional remainder, with physical row id as the
    stable tie-break.  Every result is in ``[0, cap]`` and sums exactly to
    ``total``.
    """

    total_i = int(total)
    cap_i = int(cap)
    if total_i < 0 or cap_i < 0:
        raise SemanticPlanError("quota total/cap must be nonnegative")
    if row_ids.dtype != np.dtype(np.int64) or weights.dtype != np.dtype(np.float64):
        raise SemanticPlanError("quota inputs must be row_id:int64 and weight:float64")
    if row_ids.ndim != 1 or weights.shape != row_ids.shape:
        raise SemanticPlanError("quota row_ids/weights shape mismatch")
    n = int(row_ids.size)
    if total_i > n * cap_i:
        raise SemanticPlanError(
            f"quota infeasible: total={total_i} exceeds rows*cap={n}*{cap_i}"
        )
    if total_i == 0:
        return np.zeros(n, dtype=np.int64)
    if n == 0 or cap_i == 0:
        raise SemanticPlanError("positive quota total has no capacity")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise SemanticPlanError("quota weights must be finite and positive")

    target = np.zeros(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)
    remaining = total_i
    # At least one row is fixed on every saturating pass, so this terminates.
    while remaining:
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            raise SemanticPlanError("bounded water-fill exhausted active rows")
        active_weights = weights[active_indices]
        weight_sum = float(active_weights.sum(dtype=np.float64))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise SemanticPlanError("bounded water-fill weight sum is invalid")
        proposed = (float(remaining) / weight_sum) * active_weights
        saturated_local = proposed >= float(cap_i)
        if not np.any(saturated_local):
            target[active_indices] = proposed
            remaining = 0
            break
        saturated = active_indices[saturated_local]
        target[saturated] = float(cap_i)
        active[saturated] = False
        remaining -= int(saturated.size) * cap_i
        if remaining < 0:
            raise SemanticPlanError("bounded water-fill overallocated saturated rows")

    quota = np.floor(target).astype(np.int64)
    quota = np.minimum(quota, cap_i)
    left = total_i - int(quota.sum(dtype=np.int64))
    if left < 0:
        raise SemanticPlanError("largest-remainder floor sum exceeded total")
    if left:
        eligible = np.flatnonzero(quota < cap_i)
        if left > int(eligible.size):
            raise SemanticPlanError(
                f"largest-remainder residual {left} exceeds eligible rows {eligible.size}"
            )
        fractions = target[eligible] - quota[eligible]
        order = np.lexsort((row_ids[eligible], -fractions))
        quota[eligible[order[:left]]] += 1
    if int(quota.sum(dtype=np.int64)) != total_i:
        raise SemanticPlanError("largest-remainder conservation failure")
    if np.any(quota < 0) or np.any(quota > cap_i):
        raise SemanticPlanError("largest-remainder cap failure")
    return quota


def _segment_index(offsets: np.ndarray, position: int) -> int:
    return int(np.searchsorted(offsets, int(position), side="right") - 1)


def _repair_layer_boundaries(
    sequence: np.ndarray,
    segment_counts: np.ndarray,
    layer_boundaries: Sequence[int],
    *,
    repair_seed: int,
) -> int:
    """Repair only layer-crossing segments with potential-decreasing swaps.

    Every layer is a permutation of distinct rows, so only a segment crossed by
    a layer boundary can initially contain a duplicate.  A repair swaps one
    duplicate occurrence with a row from a currently unique donor segment,
    requiring that both post-swap segments are unique.  Therefore each swap
    strictly decreases duplicate excess and never creates a new bad segment.
    The finite exhaustive donor scan either finds such a swap or proves this
    particular quota-layer realization cannot be repaired and fails closed.
    """

    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(segment_counts, dtype=np.int64))
    )
    total = int(offsets[-1])
    if sequence.shape != (total,):
        raise SemanticPlanError("boundary repair sequence/segment size mismatch")
    boundary_segments: set[int] = set()
    for boundary in layer_boundaries:
        boundary_i = int(boundary)
        insertion = int(np.searchsorted(offsets, boundary_i, side="left"))
        if insertion < offsets.size and int(offsets[insertion]) == boundary_i:
            continue  # Exact segment boundary: no segment crosses this layer edge.
        boundary_segments.add(_segment_index(offsets, boundary_i))

    repairs = 0
    num_segments = int(segment_counts.size)
    for bad_segment in sorted(boundary_segments):
        bad_lo, bad_hi = int(offsets[bad_segment]), int(offsets[bad_segment + 1])
        while True:
            bad_values = sequence[bad_lo:bad_hi]
            unique, counts = np.unique(bad_values, return_counts=True)
            duplicated = unique[counts > 1]
            if duplicated.size == 0:
                break
            x = int(duplicated[0])
            x_positions = np.flatnonzero(bad_values == x)
            # Replace the later occurrence, retaining a stable first occurrence.
            replace_position = bad_lo + int(x_positions[-1])
            bad_without = set(int(v) for i, v in enumerate(bad_values.tolist()) if bad_lo + i != replace_position)
            start_segment = int(
                hashlib.sha256(
                    f"{repair_seed}\0{bad_segment}\0{repairs}\0{x}".encode("utf-8")
                ).digest()[0:8].hex(),
                16,
            ) % max(1, num_segments)
            found: tuple[int, int] | None = None
            for segment_delta in range(num_segments):
                donor_segment = (start_segment + segment_delta) % num_segments
                if donor_segment == bad_segment:
                    continue
                donor_lo = int(offsets[donor_segment])
                donor_hi = int(offsets[donor_segment + 1])
                donor_values = sequence[donor_lo:donor_hi]
                # Never export damage: donors must be unique before the swap.
                if np.unique(donor_values).size != donor_values.size:
                    continue
                donor_set = set(int(v) for v in donor_values.tolist())
                if x in donor_set:
                    continue
                for donor_position in range(donor_lo, donor_hi):
                    y = int(sequence[donor_position])
                    if y in bad_without:
                        continue
                    found = (donor_position, y)
                    break
                if found is not None:
                    break
            if found is None:
                raise SemanticPlanError(
                    "quota-layer boundary repair found no validity-preserving donor: "
                    f"segment={bad_segment}, duplicate_row={x}, repairs={repairs}"
                )
            donor_position, y = found
            sequence[replace_position] = y
            sequence[donor_position] = x
            repairs += 1
            if np.unique(sequence[bad_lo:bad_hi]).size != unique.size + 1:
                raise AssertionError("boundary swap failed to decrease duplicate excess")

    # Full verification is required even though only boundary segments can be
    # initially bad; this also proves swaps did not export damage.
    for segment in range(num_segments):
        lo, hi = int(offsets[segment]), int(offsets[segment + 1])
        if np.unique(sequence[lo:hi]).size != hi - lo:
            raise SemanticPlanError(
                f"duplicate remains after quota-layer boundary repair in segment {segment}"
            )
    return repairs


def _bipartite_havel_hakimi(
    row_ids: np.ndarray,
    quota: np.ndarray,
    segment_counts: np.ndarray,
    *,
    tie_seed: int,
) -> np.ndarray:
    """Exact deterministic fallback for a graphical row/segment degree pair.

    This is the bipartite Havel-Hakimi construction: process segment vertices
    by descending degree and connect each to rows with the largest residual
    degree.  Chosen rows are held out of the heap until the segment closes, so
    a segment is unique by construction.  The theorem guarantees completion
    for graphical degree sequences; otherwise this function fails closed.

    Production-scale cells are expected to close through the O(number of layer
    boundaries) swap repair.  This fallback exists for adversarial/high-density
    degree sequences and correctness, not as the normal materialization path.
    """

    total = int(quota.sum(dtype=np.int64))
    if int(segment_counts.sum(dtype=np.int64)) != total:
        raise SemanticPlanError("Havel-Hakimi degree sums disagree")
    heap: list[tuple[int, int, int, int]] = []
    for index, (row_raw, degree_raw) in enumerate(zip(row_ids.tolist(), quota.tolist())):
        degree = int(degree_raw)
        if degree <= 0:
            continue
        row = int(row_raw)
        tie = int.from_bytes(
            hashlib.sha256(f"{tie_seed}\0{row}".encode("utf-8")).digest()[:8],
            "big",
        )
        heap.append((-degree, tie, row, index))
    heapq.heapify(heap)
    assignments: list[np.ndarray | None] = [None] * int(segment_counts.size)
    segment_order = sorted(
        range(int(segment_counts.size)),
        key=lambda segment: (-int(segment_counts[segment]), segment),
    )
    for segment in segment_order:
        required = int(segment_counts[segment])
        selected: list[tuple[int, int, int, int]] = []
        values = np.empty(required, dtype=np.int64)
        for slot in range(required):
            if not heap:
                raise SemanticPlanError(
                    "weighted quota degrees are not bipartite-graphical: "
                    f"segment={segment},required={required},slot={slot}"
                )
            negative_degree, tie, row, index = heapq.heappop(heap)
            degree = -int(negative_degree)
            if degree <= 0:
                raise SemanticPlanError("Havel-Hakimi selected a zero residual row")
            values[slot] = row
            selected.append((-(degree - 1), tie, row, index))
        for entry in selected:
            if entry[0] < 0:
                heapq.heappush(heap, entry)
        assignments[segment] = values
    if heap:
        residual = sum(-int(entry[0]) for entry in heap)
        raise SemanticPlanError(
            f"Havel-Hakimi left {residual} row tickets after all segments"
        )
    sequence = np.concatenate(
        [values for values in assignments if values is not None], axis=0
    )
    if sequence.shape != (total,):
        raise SemanticPlanError("Havel-Hakimi output size mismatch")
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(segment_counts, dtype=np.int64))
    )
    for segment in range(segment_counts.size):
        lo, hi = int(offsets[segment]), int(offsets[segment + 1])
        if np.unique(sequence[lo:hi]).size != hi - lo:
            raise AssertionError("Havel-Hakimi emitted a duplicate segment")
    return sequence


def _havel_fallback_after_boundary_failure(
    *,
    row_ids: np.ndarray,
    quota: np.ndarray,
    segment_counts: np.ndarray,
    tie_seed: int,
    bucket: int,
    domain: int,
    boundary_error: SemanticPlanError,
) -> np.ndarray:
    total = int(quota.sum(dtype=np.int64))
    if total > HAVEL_FALLBACK_MAX_TICKETS:
        raise SemanticPlanError(
            "quota-layer boundary repair failed and exact Python-heap "
            "Havel-Hakimi fallback is disabled above the performance guard: "
            f"cell=({bucket},{domain}),tickets={total},"
            f"limit={HAVEL_FALLBACK_MAX_TICKETS},"
            f"boundary_error={boundary_error}"
        ) from boundary_error
    return _bipartite_havel_hakimi(
        row_ids,
        quota,
        segment_counts,
        tie_seed=tie_seed,
    )


def ticket_permutation(
    *,
    plan_id: str,
    selection_plan_hash: str,
    overlay_header_sha256: str,
    bucket: int,
    domain: int,
    row_ids: np.ndarray,
    quota: np.ndarray,
    segment_counts: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Build exact tickets as unique quota layers, then repair boundaries.

    Layer ``l`` contains exactly the rows whose quota is greater than ``l``;
    it is independently SHA-seeded and permuted.  Thus duplicates are possible
    only where a fixed step segment crosses layer edges.  The repair routine is
    deterministic, quota-preserving, and strictly potential-decreasing.
    """

    if quota.dtype != np.dtype(np.int64) or quota.shape != row_ids.shape:
        raise SemanticPlanError("ticket quota must be int64 and align with row_ids")
    segment_counts = np.asarray(segment_counts, dtype=np.int64)
    total = int(quota.sum(dtype=np.int64))
    if int(segment_counts.sum(dtype=np.int64)) != total:
        raise SemanticPlanError("ticket segments do not sum to the exact cell quota")
    if segment_counts.size and int(segment_counts.max()) > int(row_ids.size):
        raise SemanticPlanError("one step requests more unique cell rows than overlay provides")
    candidate = np.empty(total, dtype=np.int64)
    cursor = 0
    layer_boundaries: list[int] = []
    max_quota = int(quota.max()) if quota.size else 0
    for layer in range(max_quota):
        values = np.array(row_ids[quota > layer], copy=True)
        seed = _seed_for_cell(
            plan_id,
            selection_plan_hash,
            overlay_header_sha256,
            bucket,
            domain,
            layer,
        )
        np.random.Generator(np.random.PCG64(seed)).shuffle(values)
        end = cursor + int(values.size)
        candidate[cursor:end] = values
        cursor = end
        if cursor < total:
            layer_boundaries.append(cursor)
    if cursor != total:
        raise SemanticPlanError(
            f"quota-layer conservation failure: built={cursor}, expected={total}"
        )
    repair_seed = _seed_for_cell(
        plan_id,
        selection_plan_hash,
        overlay_header_sha256,
        bucket,
        domain,
        max_quota,
    )
    try:
        repairs = _repair_layer_boundaries(
            candidate,
            segment_counts,
            layer_boundaries,
            repair_seed=repair_seed,
        )
        return candidate, repairs
    except SemanticPlanError as boundary_error:
        fallback = _havel_fallback_after_boundary_failure(
            row_ids=row_ids,
            quota=quota,
            segment_counts=segment_counts,
            tie_seed=repair_seed,
            bucket=bucket,
            domain=domain,
            boundary_error=boundary_error,
        )
        # -1 is an explicit, immutable materialization signal that the exact
        # Havel-Hakimi fallback (rather than boundary swaps) closed this cell.
        return fallback, -1


def _load_schedule(selection: OfflinePlan) -> tuple[np.ndarray, np.ndarray]:
    selection.verify_all_checksums()
    buckets: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    for block_index in range(
        selection.first_materialized_block, int(selection.header["num_blocks"])
    ):
        bucket, _items, domain_counts = selection._load_block(block_index)  # noqa: SLF001
        buckets.append(np.array(bucket, copy=True))
        counts.append(np.array(domain_counts, copy=True))
    bucket_all = np.concatenate(buckets, axis=0)
    counts_all = np.concatenate(counts, axis=0)
    if bucket_all.shape != (int(selection.header["total_steps"]),):
        raise SemanticPlanError("selection schedule does not cover every stage step")
    return bucket_all, counts_all


def _identity_payload(header: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "plan_format": header["plan_format"],
        "algorithm_versions": header["algorithm_versions"],
        "plan_id": header["plan_id"],
        "world_size": header["world_size"],
        "per_rank_bs": header["per_rank_bs"],
        "total_steps": header["total_steps"],
        "domains": header["domains"],
        "buckets": header["buckets"],
        "N": header["N"],
        "N_select": header["N_select"],
        "R_max": header["R_max"],
        "D": header["D"],
        "S": header["S"],
        "C": header["C"],
        "excluded_row_ids": header["excluded_row_ids"],
        "audit_contract": header["audit_contract"],
        "catalog_hash": header["catalog_hash"],
        "assignment_hash": header["assignment_hash"],
        "upstream_catalog_hash": header["upstream_catalog_hash"],
        "upstream_assignment_hash": header["upstream_assignment_hash"],
        "bindings": header["bindings"],
        "blocks_manifest_sha256": header["hashes"]["blocks_manifest_sha256"],
        "quota_manifest_sha256": header["hashes"]["quota_manifest_sha256"],
    }


def materialize(
    *,
    base_plan_root: Path,
    selection_plan_root: Path,
    overlay_root: Path,
    catalog_root: Path,
    sampling_config: Path,
    output_root: Path,
    plan_id: str,
    verify_base_blocks: bool,
    formal_prefix_steps: int = 166_667,
) -> dict[str, Any]:
    if not plan_id:
        raise SemanticPlanError("plan_id must be non-empty")
    if output_root.exists():
        try:
            existing = SemanticOfflinePlan(output_root, require_audit=False)
        except BaseException as exc:
            raise SemanticPlanError(
                f"refusing to touch existing non-valid/partial output {output_root}: {exc}"
            ) from exc
        if existing.header["plan_id"] != plan_id:
            raise SemanticPlanError(
                f"existing semantic plan_id={existing.header['plan_id']!r} differs from requested {plan_id!r}"
            )
        success_path = output_root / "MATERIALIZE_SUCCESS.json"
        if not success_path.is_file():
            raise SemanticPlanError(
                f"existing plan lacks create-once materialization closure: {success_path}"
            )
        success = json.loads(success_path.read_text(encoding="utf-8"))
        if (
            success.get("status") != "PASS"
            or success.get("plan_hash") != existing.header["plan_hash"]
            or success.get("header_sha256") != sha256_file(output_root / "header.json")
            or success.get("checksums_sha256") != sha256_file(output_root / "checksums.json")
        ):
            raise SemanticPlanError("existing MATERIALIZE_SUCCESS.json binding mismatch")
        return dict(existing.header)

    try:
        base = OfflinePlan(base_plan_root, verify_checksums=True)
        selection = OfflinePlan(selection_plan_root, verify_checksums=True)
    except PlannerInputError as exc:
        raise SemanticPlanError(f"cannot bind legacy plan: {exc}") from exc
    if verify_base_blocks:
        base.verify_all_checksums()
    if int(selection.header["materialization"]["first_materialized_step"]) != 0:
        raise SemanticPlanError("selection plan must be a full stage plan starting at step 0")

    overlay = load_overlay_header(overlay_root)
    overlay_header_sha = sha256_file(overlay_root / "header.json")
    catalog_header_sha = sha256_file(catalog_root / "header.json")
    catalog_header = read_json_object(catalog_root / "header.json")
    sampling_sha = sha256_file(sampling_config)
    if overlay["catalog_header_sha256"] != catalog_header_sha:
        raise SemanticPlanError("overlay catalog_header_sha256 differs from catalog_root")
    if overlay["sampling_config_sha256"] != sampling_sha:
        raise SemanticPlanError("overlay sampling_config_sha256 differs from sampling_config")
    if overlay["catalog_hash"] != base.header["catalog_hash"]:
        raise SemanticPlanError("overlay catalog_hash differs from base physical plan")
    catalog_hash = catalog_header.get("catalog_hash")
    assignment_hash = catalog_header.get("assignment_hash")
    if catalog_hash != overlay["catalog_hash"]:
        raise SemanticPlanError("catalog header hash differs from overlay catalog_hash")
    if not isinstance(assignment_hash, str) or len(assignment_hash) != 64:
        raise SemanticPlanError("catalog header assignment_hash is missing/invalid")
    for label, legacy in (("base physical", base.header), ("selection", selection.header)):
        expected_identity = {
            "catalog_hash": catalog_hash,
            "upstream_catalog_hash": catalog_hash,
            "assignment_hash": assignment_hash,
            "upstream_assignment_hash": assignment_hash,
        }
        for key, expected in expected_identity.items():
            if legacy.get(key) != expected:
                raise SemanticPlanError(
                    f"{label} plan {key} differs from catalog header: "
                    f"stored={legacy.get(key)!r}, expected={expected!r}"
                )
        legacy_hashes = legacy.get("hashes")
        if not isinstance(legacy_hashes, Mapping):
            raise SemanticPlanError(f"{label} plan hashes are missing")
        if legacy_hashes.get("catalog_sha256") != catalog_hash:
            raise SemanticPlanError(f"{label} plan hashes.catalog_sha256 mismatch")
        if legacy_hashes.get("assignment_sha256") != assignment_hash:
            raise SemanticPlanError(
                f"{label} plan hashes.assignment_sha256 mismatch"
            )
    if overlay["domains"] != base.header["domains"] or overlay["buckets"] != base.header["buckets"]:
        raise SemanticPlanError("overlay domain/bucket table differs from base physical plan")
    if overlay["domains"] != selection.header["domains"] or overlay["buckets"] != selection.header["buckets"]:
        raise SemanticPlanError("overlay domain/bucket table differs from selection plan")
    if overlay["N_select"] != selection.header["N"]:
        raise SemanticPlanError("overlay N_select differs from selection plan N")
    if base.header["N"] == selection.header["N"]:
        # Allowed in synthetic tests but retain an explicit record; production
        # anime NL filtering is expected to make at least one cell smaller.
        selection_reduces_physical = False
    else:
        selection_reduces_physical = True
    n_physical = np.asarray(base.header["N"], dtype=np.int64)
    n_select = np.asarray(selection.header["N"], dtype=np.int64)
    if np.any(n_select > n_physical):
        raise SemanticPlanError("selection N exceeds physical catalog N")
    if base.header["catalog_size"] != int(n_physical.sum(dtype=np.int64)):
        raise SemanticPlanError("base physical catalog_size disagrees with N")

    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "blocks").mkdir()
    (output_root / "quotas").mkdir()
    write_json_create_once(
        output_root / "MATERIALIZATION_INTENT.json",
        {
            "format": "offline_arb_semantic_materialization_intent_v1",
            "plan_id": plan_id,
            "base_plan_root": str(base_plan_root.resolve(strict=False)),
            "selection_plan_root": str(selection_plan_root.resolve(strict=False)),
            "overlay_root": str(overlay_root.resolve(strict=False)),
            "catalog_root": str(catalog_root.resolve(strict=False)),
            "sampling_config": str(sampling_config.resolve(strict=False)),
            "algorithm_version": ALGORITHM_VERSION,
        },
    )

    bucket_of_step, domain_counts = _load_schedule(selection)
    B, K = n_select.shape
    H = int(selection.header["total_steps"])
    W = int(selection.header["world_size"])
    bs = int(selection.header["per_rank_bs"])
    G = W * bs
    if formal_prefix_steps <= 0 or formal_prefix_steps > H:
        raise SemanticPlanError(
            f"formal_prefix_steps must be in [1,{H}], got {formal_prefix_steps}"
        )
    prefix_domain_totals_values = domain_counts[:formal_prefix_steps].sum(
        axis=0, dtype=np.int64
    )
    prefix_domain_totals = {
        name: int(prefix_domain_totals_values[d])
        for d, name in enumerate(selection.header["domains"])
    }
    if W == 64 and prefix_domain_totals != FORMAL_W64_PREFIX_DOMAIN_TOTALS:
        raise SemanticPlanError(
            "formal W64 prefix domain totals drift: "
            f"observed={prefix_domain_totals}, expected={FORMAL_W64_PREFIX_DOMAIN_TOTALS}"
        )
    C = np.asarray(selection.header["C"], dtype=np.int64)
    S = np.asarray(selection.header["S"], dtype=np.int64)
    active_native_indices = np.flatnonzero(S > 0).astype(np.int64).tolist()
    if W == 64 and active_native_indices != list(FORMAL_W64_ACTIVE_BUCKET_INDICES):
        raise SemanticPlanError(
            "formal W64 active native bucket indices drift: "
            f"observed={active_native_indices}, expected={list(FORMAL_W64_ACTIVE_BUCKET_INDICES)}"
        )
    offsets = physical_cell_offsets(n_physical)
    if base.header["cell_row_offset"] != offsets.tolist():
        raise SemanticPlanError("base physical row offsets differ from N-derived mapping")
    rmax = np.asarray(
        [int(selection.header["R_max"][name]) for name in selection.header["domains"]],
        dtype=np.int64,
    )
    excluded = np.asarray(overlay["excluded_row_ids"], dtype=np.int64)

    streams: dict[tuple[int, int], np.ndarray] = {}
    stream_cursor = np.zeros((B, K), dtype=np.int64)
    quota_checksums: dict[str, str] = {}
    quota_files: dict[str, dict[str, int]] = {}
    boundary_repairs: dict[str, int] = {}
    havel_fallback_cells: list[str] = []
    overlay_rows_verified = 0
    for b in range(B):
        bucket_mask = bucket_of_step == b
        if int(bucket_mask.sum()) != int(S[b]):
            raise SemanticPlanError(f"selection S mismatch for bucket {b}")
        for d in range(K):
            total = int(C[b, d])
            rows_expected = int(n_select[b, d])
            if total == 0:
                if rows_expected != 0:
                    # A zero-exposure positive cell is legal but still verify
                    # its overlay and emit a zero quota file for audit closure.
                    pass
                else:
                    continue
            row_ids, weights = load_overlay_cell(overlay_root, overlay, b, d)
            if row_ids.size != rows_expected:
                raise SemanticPlanError(f"overlay rows drift in cell ({b},{d})")
            row_lo = int(offsets[b, d])
            row_hi = row_lo + int(n_physical[b, d])
            if np.any(row_ids < row_lo) or np.any(row_ids >= row_hi):
                raise SemanticPlanError(f"overlay row outside physical cell ({b},{d})")
            if excluded.size and np.any(np.isin(row_ids, excluded)):
                raise SemanticPlanError(f"overlay includes excluded row in cell ({b},{d})")
            overlay_rows_verified += int(row_ids.size)
            step_counts = domain_counts[bucket_mask, d].astype(np.int64, copy=False)
            positive_counts = step_counts[step_counts > 0]
            occurrence_cap = int(positive_counts.size)
            cap = min(int(rmax[d]), occurrence_cap)
            quota = capped_largest_remainder(total, row_ids, weights, cap)
            quota_name = f"quota_b{b:03d}_d{d:03d}.npz"
            quota_path = output_root / "quotas" / quota_name
            write_deterministic_npz_create_once(
                quota_path,
                (
                    ("row_id", row_ids),
                    ("quota", quota.astype(np.int32)),
                ),
            )
            quota_checksums[quota_name] = sha256_file(quota_path)
            quota_files[quota_name] = {
                "bucket": b,
                "domain": d,
                "rows": int(row_ids.size),
                "exposures": total,
            }
            stream, repair_count = ticket_permutation(
                plan_id=plan_id,
                selection_plan_hash=str(selection.header["plan_hash"]),
                overlay_header_sha256=overlay_header_sha,
                bucket=b,
                domain=d,
                row_ids=row_ids,
                quota=quota,
                segment_counts=positive_counts,
            )
            streams[(b, d)] = stream
            boundary_repairs[cell_key(b, d)] = int(repair_count)
            if repair_count < 0:
                havel_fallback_cells.append(cell_key(b, d))

    block_checksums: dict[str, str] = {}
    block_steps = int(selection.header["block_steps"])
    num_blocks = int(selection.header["num_blocks"])
    for block_index in range(num_blocks):
        start = block_index * block_steps
        stop = min(H, start + block_steps)
        step_count = stop - start
        block_bucket = bucket_of_step[start:stop].astype(np.int16, copy=True)
        block_counts = domain_counts[start:stop].astype(np.int32, copy=True)
        block_items = np.empty((step_count, W, bs), dtype=np.int64)
        for local_step in range(step_count):
            b = int(block_bucket[local_step])
            counts = block_counts[local_step].astype(np.int64, copy=False)
            if int(counts.sum(dtype=np.int64)) != G:
                raise SemanticPlanError(f"domain counts do not sum to G at step {start+local_step}")
            flat = np.empty(G, dtype=np.int64)
            boundaries = np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
            )
            for d in range(K):
                count = int(counts[d])
                if count == 0:
                    continue
                cursor = int(stream_cursor[b, d])
                stream = streams[(b, d)]
                end = cursor + count
                if end > int(stream.size):
                    raise SemanticPlanError(
                        f"cell stream overrun at step={start+local_step},cell=({b},{d})"
                    )
                lo, hi = int(boundaries[d]), int(boundaries[d + 1])
                flat[lo:hi] = stream[cursor:end]
                stream_cursor[b, d] = end
            if np.unique(flat).size != G:
                raise SemanticPlanError(f"global batch duplicate at step {start+local_step}")
            block_items[local_step] = flat.reshape(bs, W).T
        filename = f"block_{block_index:06d}.npz"
        block_path = output_root / "blocks" / filename
        write_deterministic_npz_create_once(
            block_path,
            (
                (BLOCK_DATASETS[0], block_bucket),
                (BLOCK_DATASETS[1], block_items),
                (BLOCK_DATASETS[2], block_counts),
            ),
        )
        block_checksums[filename] = sha256_file(block_path)

    for b in range(B):
        for d in range(K):
            if int(stream_cursor[b, d]) != int(C[b, d]):
                raise SemanticPlanError(
                    f"cell stream conservation failure ({b},{d}): "
                    f"used={stream_cursor[b,d]}, expected={C[b,d]}"
                )

    block_manifest_sha = sha256_bytes(canonical_json_bytes(block_checksums))
    quota_manifest_sha = sha256_bytes(canonical_json_bytes(quota_checksums))
    algorithms = {
        "materializer": ALGORITHM_VERSION,
        "quota": QUOTA_ALGORITHM,
        "ticket_permutation": TICKET_ALGORITHM,
        "block_format": BLOCK_FORMAT,
    }
    bindings = {
        "base_physical_plan": binding_for_plan(base_plan_root, base),
        "selection_plan": binding_for_plan(selection_plan_root, selection),
        "overlay": {
            "root": str(overlay_root.resolve(strict=False)),
            "header_sha256": overlay_header_sha,
            "overlay_id": overlay["overlay_id"],
        },
        "catalog_header_sha256": catalog_header_sha,
        "sampling_config_sha256": sampling_sha,
    }
    header: dict[str, Any] = {
        "format_version": PLAN_FORMAT_VERSION,
        "plan_format": PLAN_FORMAT,
        "algorithm_version": ALGORITHM_VERSION,
        "algorithm_versions": algorithms,
        "plan_id": plan_id,
        "plan_hash": "",
        "world_size": W,
        "per_rank_bs": bs,
        "total_steps": H,
        "G": G,
        "global_batch_size": G,
        "total_exposures": H * G,
        "block_steps": block_steps,
        "num_blocks": num_blocks,
        "block_datasets": list(BLOCK_DATASETS),
        "domains": list(selection.header["domains"]),
        "domain_mix": dict(selection.header["domain_mix"]),
        "square_only_domains": list(selection.header["square_only_domains"]),
        "buckets": [list(hw) for hw in selection.header["buckets"]],
        "square_bucket_idx": int(selection.header["square_bucket_idx"]),
        "N": n_physical.tolist(),
        "N_select": n_select.tolist(),
        "R_max": dict(selection.header["R_max"]),
        "D": list(selection.header["D"]),
        "S": list(selection.header["S"]),
        "C": [list(row) for row in selection.header["C"]],
        "catalog_hash": str(base.header["catalog_hash"]),
        "assignment_hash": str(assignment_hash),
        "upstream_catalog_hash": str(catalog_hash),
        "upstream_assignment_hash": str(assignment_hash),
        "catalog_size": int(base.header["catalog_size"]),
        "cell_row_offset": offsets.tolist(),
        "excluded_row_ids": [int(value) for value in overlay["excluded_row_ids"]],
        "audit_contract": {
            "formal_prefix_steps": int(formal_prefix_steps),
            "formal_prefix_exposures": int(formal_prefix_steps) * G,
            "formal_prefix_domain_totals": prefix_domain_totals,
            "active_native_buckets": int(np.count_nonzero(S)),
            "active_native_bucket_indices": active_native_indices,
            "excluded_rows": int(len(overlay["excluded_row_ids"])),
        },
        "materialization": {
            "first_materialized_block": 0,
            "first_materialized_step": 0,
            "num_blocks": num_blocks,
            "source_schedule": "selection_plan.bucket_of_step+domain_counts",
            "selection_reduces_physical": selection_reduces_physical,
            "overlay_rows_verified": overlay_rows_verified,
            "ticket_boundary_repairs_by_cell": boundary_repairs,
            "ticket_boundary_repairs_total": int(
                sum(value for value in boundary_repairs.values() if value >= 0)
            ),
            "ticket_havel_fallback_cells": sorted(havel_fallback_cells),
            "ticket_havel_fallback_max_tickets": HAVEL_FALLBACK_MAX_TICKETS,
        },
        "bindings": bindings,
        "quota_files": quota_files,
        "hashes": {
            "blocks_manifest_sha256": block_manifest_sha,
            "quota_manifest_sha256": quota_manifest_sha,
            "identity_payload_sha256": "",
            "catalog_sha256": str(catalog_hash),
            "assignment_sha256": str(assignment_hash),
        },
    }
    identity_sha = sha256_bytes(canonical_json_bytes(_identity_payload(header)))
    header["plan_hash"] = identity_sha
    header["hashes"]["identity_payload_sha256"] = identity_sha
    write_json_create_once(output_root / "header.json", header)
    checksums = {
        "header": sha256_file(output_root / "header.json"),
        "blocks": block_checksums,
        "quotas": quota_checksums,
    }
    write_json_create_once(output_root / "checksums.json", checksums)
    # Re-open through the production reader (audit receipt is intentionally not
    # present yet) so any header/checksum/binding drift fails this job.
    reopened = SemanticOfflinePlan(output_root, require_audit=False)
    reopened.verify_all_checksums(include_quotas=True)
    write_json_create_once(
        output_root / "MATERIALIZE_SUCCESS.json",
        {
            "format": "offline_arb_semantic_materialize_success_v1",
            "status": "PASS",
            "plan_id": plan_id,
            "plan_hash": identity_sha,
            "header_sha256": checksums["header"],
            "checksums_sha256": sha256_file(output_root / "checksums.json"),
            "total_steps": H,
            "total_exposures": H * G,
            "overlay_rows_verified": overlay_rows_verified,
        },
    )
    return dict(reopened.header)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", required=True, type=Path)
    parser.add_argument("--selection-plan", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--catalog-root", required=True, type=Path)
    parser.add_argument("--sampling-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--formal-prefix-steps", type=int, default=166_667)
    parser.add_argument(
        "--skip-base-block-checksums",
        action="store_true",
        help="synthetic-test escape hatch; forbidden for the formal production job",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        header = materialize(
            base_plan_root=args.base_plan,
            selection_plan_root=args.selection_plan,
            overlay_root=args.overlay,
            catalog_root=args.catalog_root,
            sampling_config=args.sampling_config,
            output_root=args.output,
            plan_id=args.plan_id,
            verify_base_blocks=not args.skip_base_block_checksums,
            formal_prefix_steps=args.formal_prefix_steps,
        )
    except BaseException as exc:
        print(f"SEMANTIC_MATERIALIZE_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "plan_id": header["plan_id"],
                "plan_hash": header["plan_hash"],
                "total_steps": header["total_steps"],
                "total_exposures": header["total_exposures"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
