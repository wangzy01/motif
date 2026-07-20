"""OFFLINE_ARB_V1 deterministic L3 planner.

The planner is a pure function of its input contract.  It uses integer quota
bookkeeping, a low-discrepancy bucket schedule, deterministic cell-local PRPs,
and a fully materialized block format.  It intentionally imports neither torch
nor any code from the streaming dataline.

The v1 input contract supplies ``N[B,d]`` rather than an assignment snapshot's
physical row array.  Item identities therefore use the frozen virtual-catalog
scheme: domains are contiguous, and cells within a domain are contiguous in
input bucket order.  Production catalogs use the same ordering contract.
"""

from __future__ import annotations

import hashlib
import heapq
import io
import json
import math
import os
import shutil
import zipfile
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PLAN_FORMAT = "offline_arb_plan_v1"
PLAN_FORMAT_VERSION = 1
PLANNER_ALGORITHM_VERSION = "offline_arb_planner_v1_20260710"
QUOTA_MILP_VERSION = "scipy_milp_l1_transport_scaled_exactgap_v1"
QUOTA_FALLBACK_VERSION = "two_level_lr_residual_transport_v1"
BUCKET_SCHEDULE_VERSION = "bresenham_deficit_v1"
PER_STEP_QUOTA_VERSION = "column_bresenham_remaining_quota_repair_v2"
RANK_DEAL_VERSION = "domain_major_interleaved_deal_v1"
FEISTEL_VERSION = "feistel_index_splitmix64_v1"
FEISTEL_ROUNDS = 8
NPZ_FORMAT_VERSION = "deterministic_stored_npz_npy2_three_datasets_v2"
CHECKSUMS_FORMAT_VERSION = "header_and_basename_block_sha256_v2"
ROW_ID_MAPPING_VERSION = "virtual_catalog_v1"
BLOCK_DATASETS = ("bucket_of_step", "item_row", "domain_counts")
DOMAIN_MIX_SUM_TOLERANCE = Decimal("1e-9")

_INT64_MAX = int(np.iinfo(np.int64).max)
_UINT64_MASK = (1 << 64) - 1
_REQUIRED_KEYS = {
    "format_version",
    "plan_id",
    "seed",
    "world_size",
    "per_rank_bs",
    "total_steps",
    "domains",
    "domain_mix",
    "square_only_domains",
    "buckets",
    "square_bucket_idx",
    "N",
    "R_max",
    "epsilon_jitter",
    "block_steps",
}
_OPTIONAL_KEYS = {"catalog_hash", "assignment_hash"}


class PlannerInputError(ValueError):
    """The planner input violates the frozen v1 contract."""


class PlanInfeasibleError(RuntimeError):
    """Integer quotas cannot satisfy the frozen plan constraints."""

    def __init__(self, diagnostics: Sequence[Mapping[str, Any]]):
        self.diagnostics = [dict(item) for item in diagnostics]
        if self.diagnostics:
            lead = self.diagnostics[0]
            message = f"plan infeasible: {lead.get('constraint', lead.get('code', 'unknown'))}: {lead.get('message', '')}"
        else:
            message = "plan infeasible"
        super().__init__(message)

    @property
    def diagnostic(self) -> list[dict[str, Any]]:
        return self.diagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": "plan_infeasible",
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class PlannerConfig:
    """Validated, normalized planner input."""

    format_version: int
    plan_id: str
    seed: int
    world_size: int
    per_rank_bs: int
    total_steps: int
    domains: tuple[str, ...]
    domain_mix: tuple[Decimal, ...]
    square_only_domains: tuple[str, ...]
    buckets: tuple[tuple[int, int], ...]
    square_bucket_idx: int
    N: np.ndarray
    R_max: tuple[int, ...]
    epsilon_jitter: bool
    block_steps: int
    catalog_hash: str | None
    assignment_hash: str | None
    input_sha256: str
    config_sha256: str

    @property
    def B(self) -> int:
        return len(self.buckets)

    @property
    def K(self) -> int:
        return len(self.domains)

    @property
    def G(self) -> int:
        return self.world_size * self.per_rank_bs

    @property
    def M(self) -> int:
        return self.total_steps * self.G

    @property
    def square_domain_indices(self) -> tuple[int, ...]:
        selected = set(self.square_only_domains)
        return tuple(i for i, name in enumerate(self.domains) if name in selected)

    def normalized_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "format_version": self.format_version,
            "plan_id": self.plan_id,
            "seed": self.seed,
            "world_size": self.world_size,
            "per_rank_bs": self.per_rank_bs,
            "total_steps": self.total_steps,
            "domains": list(self.domains),
            "domain_mix": {
                name: _decimal_text(value)
                for name, value in zip(self.domains, self.domain_mix)
            },
            "square_only_domains": list(self.square_only_domains),
            "buckets": [list(hw) for hw in self.buckets],
            "square_bucket_idx": self.square_bucket_idx,
            "N": self.N.tolist(),
            "R_max": {
                name: int(value)
                for name, value in zip(self.domains, self.R_max)
            },
            "epsilon_jitter": self.epsilon_jitter,
            "block_steps": self.block_steps,
        }
        if self.catalog_hash is not None:
            result["catalog_hash"] = self.catalog_hash
        if self.assignment_hash is not None:
            result["assignment_hash"] = self.assignment_hash
        return result


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _reject_constant(value: str) -> None:
    raise PlannerInputError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PlannerInputError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _decode_input_bytes(data: bytes, source: str) -> Mapping[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PlannerInputError(f"{source}: input must be UTF-8: {exc}") from exc
    try:
        value = json.loads(
            text,
            parse_float=Decimal,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except PlannerInputError:
        raise
    except (json.JSONDecodeError, InvalidOperation) as exc:
        raise PlannerInputError(f"{source}: invalid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PlannerInputError(f"{source}: top-level JSON value must be an object")
    return value


def _require_int(value: Any, path: str, *, positive: bool = False, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlannerInputError(f"{path} must be an integer, got {value!r}")
    result = int(value)
    if positive and result <= 0:
        raise PlannerInputError(f"{path} must be > 0, got {result}")
    if nonnegative and result < 0:
        raise PlannerInputError(f"{path} must be >= 0, got {result}")
    return result


def _require_decimal(value: Any, path: str) -> Decimal:
    if isinstance(value, bool):
        raise PlannerInputError(f"{path} must be a finite number")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PlannerInputError(f"{path} must be a finite number, got {value!r}") from exc
    if not result.is_finite():
        raise PlannerInputError(f"{path} must be finite, got {value!r}")
    return result


def _optional_hash(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PlannerInputError(f"{key} must be a non-empty string when present")
    return value


def _normalize_input(raw: Mapping[str, Any], *, input_sha256: str | None = None) -> PlannerConfig:
    keys = set(raw)
    missing = sorted(_REQUIRED_KEYS - keys)
    unknown = sorted(keys - _REQUIRED_KEYS - _OPTIONAL_KEYS)
    if missing:
        raise PlannerInputError(f"missing required planner_input keys: {missing}")
    if unknown:
        raise PlannerInputError(f"unknown planner_input keys for format_version=1: {unknown}")

    format_version = _require_int(raw["format_version"], "format_version")
    if format_version != 1:
        raise PlannerInputError(f"format_version must be 1, got {format_version}")
    plan_id = raw["plan_id"]
    if not isinstance(plan_id, str) or not plan_id:
        raise PlannerInputError("plan_id must be a non-empty string")
    seed = _require_int(raw["seed"], "seed")
    world_size = _require_int(raw["world_size"], "world_size", positive=True)
    per_rank_bs = _require_int(raw["per_rank_bs"], "per_rank_bs", positive=True)
    total_steps = _require_int(raw["total_steps"], "total_steps", positive=True)
    block_steps = _require_int(raw["block_steps"], "block_steps", positive=True)
    if not isinstance(raw["epsilon_jitter"], bool):
        raise PlannerInputError("epsilon_jitter must be a JSON boolean")
    epsilon_jitter = bool(raw["epsilon_jitter"])

    domains_raw = raw["domains"]
    if not isinstance(domains_raw, list) or not domains_raw:
        raise PlannerInputError("domains must be a non-empty JSON array")
    if any(not isinstance(name, str) or not name for name in domains_raw):
        raise PlannerInputError("every domains entry must be a non-empty string")
    domains = tuple(domains_raw)
    if len(set(domains)) != len(domains):
        raise PlannerInputError("domains entries must be unique")

    mix_raw = raw["domain_mix"]
    if not isinstance(mix_raw, Mapping) or set(mix_raw) != set(domains):
        raise PlannerInputError("domain_mix keys must exactly equal domains")
    domain_mix = tuple(_require_decimal(mix_raw[name], f"domain_mix.{name}") for name in domains)
    if any(value < 0 for value in domain_mix):
        raise PlannerInputError("domain_mix values must be nonnegative")
    mix_sum = sum(domain_mix, Decimal(0))
    if abs(mix_sum - Decimal(1)) > DOMAIN_MIX_SUM_TOLERANCE:
        raise PlannerInputError(
            "domain_mix must sum to 1 within absolute tolerance 1e-9 "
            f"(decimal input semantics), got {_decimal_text(mix_sum)}; "
            "adjust the mix explicitly (the planner never renormalizes)"
        )

    square_raw = raw["square_only_domains"]
    if not isinstance(square_raw, list) or any(not isinstance(name, str) for name in square_raw):
        raise PlannerInputError("square_only_domains must be an array of domain names")
    if len(set(square_raw)) != len(square_raw):
        raise PlannerInputError("square_only_domains entries must be unique")
    unknown_square = sorted(set(square_raw) - set(domains))
    if unknown_square:
        raise PlannerInputError(f"square_only_domains contains unknown domains: {unknown_square}")
    square_only_domains = tuple(square_raw)

    buckets_raw = raw["buckets"]
    if not isinstance(buckets_raw, list) or not buckets_raw:
        raise PlannerInputError("buckets must be a non-empty array of [height,width]")
    buckets_list: list[tuple[int, int]] = []
    for b, value in enumerate(buckets_raw):
        if not isinstance(value, list) or len(value) != 2:
            raise PlannerInputError(f"buckets[{b}] must be [height,width]")
        h = _require_int(value[0], f"buckets[{b}][0]", positive=True)
        w = _require_int(value[1], f"buckets[{b}][1]", positive=True)
        buckets_list.append((h, w))
    buckets = tuple(buckets_list)
    if len(set(buckets)) != len(buckets):
        raise PlannerInputError("buckets must be unique and retain a stable input order")
    if len(buckets) > int(np.iinfo(np.int16).max):
        raise PlannerInputError("len(buckets) exceeds the int16 materialization limit (32767)")
    square_bucket_idx = _require_int(raw["square_bucket_idx"], "square_bucket_idx", nonnegative=True)
    if square_bucket_idx >= len(buckets):
        raise PlannerInputError("square_bucket_idx is outside buckets")
    if buckets[square_bucket_idx][0] != buckets[square_bucket_idx][1]:
        raise PlannerInputError("square_bucket_idx must point to an h == w bucket")

    n_raw = raw["N"]
    if not isinstance(n_raw, list) or len(n_raw) != len(buckets):
        raise PlannerInputError(f"N must have {len(buckets)} rows (one per bucket)")
    n_rows: list[list[int]] = []
    for b, row in enumerate(n_raw):
        if not isinstance(row, list) or len(row) != len(domains):
            raise PlannerInputError(f"N[{b}] must have {len(domains)} columns")
        parsed_row: list[int] = []
        for d, value in enumerate(row):
            parsed = _require_int(value, f"N[{b}][{d}]", nonnegative=True)
            if parsed > _INT64_MAX:
                raise PlannerInputError(f"N[{b}][{d}] exceeds int64")
            parsed_row.append(parsed)
        n_rows.append(parsed_row)
    N = np.asarray(n_rows, dtype=np.int64)

    r_raw = raw["R_max"]
    if not isinstance(r_raw, Mapping) or set(r_raw) != set(domains):
        raise PlannerInputError("R_max keys must exactly equal domains")
    R_max = tuple(_require_int(r_raw[name], f"R_max.{name}", nonnegative=True) for name in domains)

    square_set = set(square_only_domains)
    upstream_violations: list[str] = []
    for d, name in enumerate(domains):
        if name not in square_set:
            continue
        for b in range(len(buckets)):
            if b != square_bucket_idx and int(N[b, d]) != 0:
                upstream_violations.append(f"N[{b},{d}]={int(N[b,d])} ({buckets[b]}, {name})")
    if upstream_violations:
        sample = ", ".join(upstream_violations[:8])
        raise PlannerInputError(
            "square-only assignment snapshot violates the frozen L2 contract; non-square cells must be zero: "
            + sample
        )

    G = world_size * per_rank_bs
    M = total_steps * G
    if G > _INT64_MAX or M > _INT64_MAX:
        raise PlannerInputError("G or H*G exceeds int64 materialization/accounting limits")
    if G > int(np.iinfo(np.int32).max):
        raise PlannerInputError("G exceeds the int32 domain_counts materialization limit")
    total_items = sum(int(value) for value in N.flat)
    if total_items > _INT64_MAX:
        raise PlannerInputError("sum(N) exceeds the virtual catalog int64 row-ID space")

    catalog_hash = _optional_hash(raw, "catalog_hash")
    assignment_hash = _optional_hash(raw, "assignment_hash")
    normalized_for_hash: dict[str, Any] = {
        "format_version": format_version,
        "plan_id": plan_id,
        "seed": seed,
        "world_size": world_size,
        "per_rank_bs": per_rank_bs,
        "total_steps": total_steps,
        "domains": list(domains),
        "domain_mix": {name: _decimal_text(value) for name, value in zip(domains, domain_mix)},
        "square_only_domains": list(square_only_domains),
        "buckets": [list(hw) for hw in buckets],
        "square_bucket_idx": square_bucket_idx,
        "N": N.tolist(),
        "R_max": {name: value for name, value in zip(domains, R_max)},
        "epsilon_jitter": epsilon_jitter,
        "block_steps": block_steps,
    }
    if catalog_hash is not None:
        normalized_for_hash["catalog_hash"] = catalog_hash
    if assignment_hash is not None:
        normalized_for_hash["assignment_hash"] = assignment_hash
    config_sha256 = _sha256_bytes(_canonical_json_bytes(normalized_for_hash))
    if input_sha256 is None:
        input_sha256 = config_sha256

    return PlannerConfig(
        format_version=format_version,
        plan_id=plan_id,
        seed=seed,
        world_size=world_size,
        per_rank_bs=per_rank_bs,
        total_steps=total_steps,
        domains=domains,
        domain_mix=domain_mix,
        square_only_domains=square_only_domains,
        buckets=buckets,
        square_bucket_idx=square_bucket_idx,
        N=N,
        R_max=R_max,
        epsilon_jitter=epsilon_jitter,
        block_steps=block_steps,
        catalog_hash=catalog_hash,
        assignment_hash=assignment_hash,
        input_sha256=input_sha256,
        config_sha256=config_sha256,
    )


def load_planner_input(path: str | os.PathLike[str]) -> PlannerConfig:
    """Load and fully validate a v1 ``planner_input.json``."""

    input_path = Path(path)
    try:
        data = input_path.read_bytes()
    except OSError as exc:
        raise PlannerInputError(f"cannot read planner input {input_path}: {exc}") from exc
    raw = _decode_input_bytes(data, str(input_path))
    return _normalize_input(raw, input_sha256=_sha256_bytes(data))


def _as_config(value: PlannerConfig | Mapping[str, Any] | str | os.PathLike[str]) -> PlannerConfig:
    if isinstance(value, PlannerConfig):
        return value
    if isinstance(value, (str, os.PathLike)):
        return load_planner_input(value)
    if isinstance(value, Mapping):
        return _normalize_input(value)
    raise PlannerInputError(f"unsupported planner input type: {type(value).__name__}")


def _largest_remainder_decimal(total: int, weights: Sequence[Decimal]) -> np.ndarray:
    quotas = [Decimal(total) * weight for weight in weights]
    floors = [int(value.to_integral_value(rounding=ROUND_FLOOR)) for value in quotas]
    left = total - sum(floors)
    if left < 0 or left > len(weights):
        raise PlannerInputError(
            f"largest-remainder precondition failed: total={total}, floor_sum={sum(floors)}, left={left}"
        )
    order = sorted(range(len(weights)), key=lambda i: (-(quotas[i] - floors[i]), i))
    for i in order[:left]:
        floors[i] += 1
    result = np.asarray(floors, dtype=np.int64)
    if int(result.sum()) != total:
        raise AssertionError("largest-remainder conservation failure")
    return result


def _derive_domain_totals(cfg: PlannerConfig) -> np.ndarray:
    return _largest_remainder_decimal(cfg.M, cfg.domain_mix)


def _allowed_mask(cfg: PlannerConfig) -> np.ndarray:
    allowed = cfg.N > 0
    for d in cfg.square_domain_indices:
        allowed[:, d] = False
        allowed[cfg.square_bucket_idx, d] = cfg.N[cfg.square_bucket_idx, d] > 0
    return allowed


def _static_caps(cfg: PlannerConfig, allowed: np.ndarray) -> np.ndarray:
    caps = np.zeros((cfg.B, cfg.K), dtype=np.int64)
    for b in range(cfg.B):
        for d in range(cfg.K):
            if allowed[b, d]:
                caps[b, d] = min(cfg.M, int(cfg.N[b, d]) * int(cfg.R_max[d]))
    return caps


def _ideal_joint(cfg: PlannerConfig, D: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    Q = np.zeros((cfg.B, cfg.K), dtype=np.float64)
    for d in range(cfg.K):
        target = int(D[d])
        if target == 0:
            continue
        total_unique = sum(int(cfg.N[b, d]) for b in range(cfg.B) if allowed[b, d])
        if total_unique == 0:
            continue
        for b in range(cfg.B):
            if allowed[b, d]:
                Q[b, d] = target * (int(cfg.N[b, d]) / total_unique)
    return Q


def _bucket_step_upper(cfg: PlannerConfig, b: int, static_caps: np.ndarray) -> int:
    """Largest s<=H for which bucket b can fill s unique global batches."""

    if sum(int(cfg.N[b, d]) for d in range(cfg.K)) < cfg.G:
        return 0

    def feasible(s: int) -> bool:
        available = sum(
            min(int(static_caps[b, d]), int(cfg.N[b, d]) * s)
            for d in range(cfg.K)
        )
        return available >= cfg.G * s

    lo, hi = 0, cfg.total_steps
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _max_single_domain_exposure(
    cfg: PlannerConfig,
    d: int,
    static_caps: np.ndarray,
    step_upper: np.ndarray,
) -> int:
    """Necessary C5/C6 upper bound with all H steps optimized for one domain."""

    segments: list[tuple[int, int]] = []
    for b in range(cfg.B):
        steps = int(step_upper[b])
        n = int(cfg.N[b, d])
        cap = int(static_caps[b, d])
        if steps <= 0 or n <= 0 or cap <= 0:
            continue
        per_step = min(n, cfg.G)
        potential = min(cap, per_step * steps)
        full, rem = divmod(potential, per_step)
        if full:
            segments.append((per_step, full))
        if rem and full < steps:
            segments.append((rem, 1))
    remaining = cfg.total_steps
    result = 0
    for value, count in sorted(segments, key=lambda pair: -pair[0]):
        take = min(remaining, count)
        result += value * take
        remaining -= take
        if remaining == 0:
            break
    return result


def _diagnostic(
    code: str,
    constraint: str,
    message: str,
    *,
    required: int | None,
    available: int | None,
    suggestions: Sequence[str],
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": code,
        "constraint": constraint,
        "message": message,
        "required": required,
        "available": available,
        "shortfall": None if required is None or available is None else max(0, required - available),
        "suggestions": list(suggestions),
    }
    if context:
        result["context"] = dict(context)
    return result


def _feasibility_precheck(
    cfg: PlannerConfig,
    D: np.ndarray,
    allowed: np.ndarray,
    static_caps: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    step_upper = np.asarray(
        [_bucket_step_upper(cfg, b, static_caps) for b in range(cfg.B)],
        dtype=np.int64,
    )

    total_step_capacity = sum(int(value) for value in step_upper)
    if total_step_capacity < cfg.total_steps:
        diagnostics.append(
            _diagnostic(
                "F0_BUCKET_STEP_CAPACITY",
                "C1+C2+C5+C6",
                "bucket cells cannot fill the requested number of unique global batches",
                required=cfg.total_steps,
                available=total_step_capacity,
                suggestions=(
                    "lower total_steps (when repeat caps are binding)",
                    "increase R_max for the capacity-limiting domains",
                    "add unique items to usable bucket cells or revise the bucket table",
                ),
                context={"bucket_step_upper": step_upper.tolist(), "global_batch_size": cfg.G},
            )
        )

    # The section 9 zero-cell necessary condition is reported separately from
    # repeat capacity so an empty assignment is not disguised as R_max=0.
    for d, name in enumerate(cfg.domains):
        if int(D[d]) <= 0:
            continue
        allowed_unique = sum(int(cfg.N[b, d]) for b in range(cfg.B) if allowed[b, d])
        if allowed_unique == 0:
            diagnostics.append(
                _diagnostic(
                    "F0_ZERO_CELL",
                    f"C3+C4 zero-cell domain={name}",
                    f"domain {name!r} has positive target exposure but no allowed catalog item",
                    required=int(D[d]),
                    available=0,
                    suggestions=(
                        f"set domain_mix[{name!r}] to zero",
                        f"add items to an allowed N[B,{name!r}] cell",
                        "revise square-only assignment/bucket constraints when applicable",
                    ),
                    context={"domain": name},
                )
            )

    # F2: repeat-limited supply, after applying C4's allowed mask.
    for d, name in enumerate(cfg.domains):
        required = int(D[d])
        available = sum(int(static_caps[b, d]) for b in range(cfg.B))
        if required > available:
            total_unique = sum(int(cfg.N[b, d]) for b in range(cfg.B) if allowed[b, d])
            needed_r = None if total_unique == 0 else math.ceil(required / total_unique)
            diagnostics.append(
                _diagnostic(
                    "F2_REPEAT_SUPPLY",
                    f"F2/C5 domain={name}",
                    f"domain {name!r} needs {required} exposures but repeat-limited allowed cells supply {available}",
                    required=required,
                    available=available,
                    suggestions=(
                        f"lower domain_mix[{name!r}] or lower total_steps",
                        f"raise R_max[{name!r}] to at least {needed_r}" if needed_r is not None else f"add an allowed non-empty cell for {name!r}",
                        "add unique items in allowed cells / revise assignment and bucket table",
                    ),
                    context={
                        "domain": name,
                        "target_exposures": required,
                        "allowed_unique_items": total_unique,
                        "R_max": int(cfg.R_max[d]),
                    },
                )
            )

    # Revised F1 (§9.2): each square-only domain has exactly one allowed cell,
    # so its tight repeat-supply necessary condition is D <= R_max*N there.
    # The retired S_max_possible formula is deliberately not reconstructed;
    # joint/per-step shortages remain the solver/F3 authority.
    square = cfg.square_bucket_idx
    square_domains = cfg.square_domain_indices
    for d in square_domains:
        required = int(D[d])
        available = int(cfg.R_max[d]) * int(cfg.N[square, d])
        if required > available:
            name = cfg.domains[d]
            diagnostics.append(
                _diagnostic(
                    "F1_SQUARE_REPEAT_SUPPLY",
                    f"F1/C4+C5 domain={name}",
                    f"square-only domain {name!r} needs {required} exposures but its only allowed cell supplies {available}",
                    required=required,
                    available=available,
                    suggestions=(
                        f"lower domain_mix[{name!r}] to <= {_decimal_text(Decimal(available) / Decimal(cfg.M))}",
                        f"increase N[square,{name!r}] or R_max[{name!r}]",
                        "revise the square-only domain assignment",
                    ),
                    context={
                        "domain": name,
                        "square_bucket_idx": square,
                        "N_square": int(cfg.N[square, d]),
                        "R_max": int(cfg.R_max[d]),
                    },
                )
            )

    # Generic per-domain C6 upper; catches sparse non-square layouts too.
    already_square_short = {
        item.get("context", {}).get("domain")
        for item in diagnostics
        if item["code"] == "F1_SQUARE_REPEAT_SUPPLY"
    }
    for d, name in enumerate(cfg.domains):
        available = _max_single_domain_exposure(cfg, d, static_caps, step_upper)
        required = int(D[d])
        if required > available and name not in already_square_short:
            diagnostics.append(
                _diagnostic(
                    "F3_PRE_C6_DOMAIN_CAPACITY",
                    f"C6 domain={name}",
                    f"even an optimal bucket-step allocation can expose only {available} unique-per-batch slots for {name!r}",
                    required=required,
                    available=available,
                    suggestions=(
                        f"lower domain_mix[{name!r}]",
                        f"increase unique N[B,{name!r}] in usable cells",
                        "revise the bucket table/assignment; R_max does not fix a pure per-step uniqueness shortage",
                    ),
                    context={"domain": name, "target_exposures": required},
                )
            )
    return step_upper, diagnostics


def _quota_objective(C: np.ndarray, Q: np.ndarray, D: np.ndarray) -> float:
    result = 0.0
    for d in range(C.shape[1]):
        if int(D[d]) > 0:
            result += float(np.abs(C[:, d].astype(np.float64) - Q[:, d]).sum()) / int(D[d])
    return result


def _quota_violations(
    cfg: PlannerConfig,
    D: np.ndarray,
    S: np.ndarray,
    C: np.ndarray,
    allowed: np.ndarray,
    static_caps: np.ndarray,
) -> list[dict[str, Any]]:
    """Exact common C1--C6/F3 verifier for every solver path."""

    violations: list[dict[str, Any]] = []
    if S.shape != (cfg.B,) or C.shape != (cfg.B, cfg.K):
        return [
            _diagnostic(
                "F3_SHAPE",
                "F3 solver output shape",
                f"unexpected S/C shapes: S={S.shape}, C={C.shape}",
                required=None,
                available=None,
                suggestions=("fix the quota solver implementation",),
            )
        ]
    if not np.issubdtype(S.dtype, np.integer) or not np.issubdtype(C.dtype, np.integer):
        violations.append(
            _diagnostic(
                "F3_INTEGRALITY",
                "F3 integer quotas",
                "S and C must be integer arrays",
                required=None,
                available=None,
                suggestions=("round only solver values within tolerance, then run exact F3",),
            )
        )
        return violations
    if np.any(S < 0) or np.any(C < 0):
        violations.append(
            _diagnostic(
                "F3_NONNEGATIVE",
                "F3 nonnegative quotas",
                "S or C contains a negative value",
                required=0,
                available=int(min(int(S.min(initial=0)), int(C.min(initial=0)))),
                suggestions=("fix solver bounds",),
            )
        )
    if int(S.sum()) != cfg.total_steps:
        violations.append(
            _diagnostic(
                "F3_C1",
                "C1 sum_B S[B] = H",
                f"bucket steps sum to {int(S.sum())}, expected {cfg.total_steps}",
                required=cfg.total_steps,
                available=int(S.sum()),
                suggestions=("adjust the quota solver; do not renormalize at training time",),
            )
        )
    row_totals = C.sum(axis=1, dtype=np.int64)
    for b in range(cfg.B):
        expected = cfg.G * int(S[b])
        actual = int(row_totals[b])
        if actual != expected:
            violations.append(
                _diagnostic(
                    "F3_C2",
                    f"C2 bucket={b}",
                    f"bucket {b} has {actual} cell exposures, expected G*S={expected}",
                    required=expected,
                    available=actual,
                    suggestions=("repair the bucket row in the integer transportation solution",),
                    context={"bucket_idx": b, "bucket": list(cfg.buckets[b])},
                )
            )
    col_totals = C.sum(axis=0, dtype=np.int64)
    for d, name in enumerate(cfg.domains):
        expected = int(D[d])
        actual = int(col_totals[d])
        if actual != expected:
            violations.append(
                _diagnostic(
                    "F3_C3",
                    f"C3 domain={name}",
                    f"domain {name!r} has {actual} exposures, expected {expected}",
                    required=expected,
                    available=actual,
                    suggestions=(f"repair the {name!r} transportation-column residual",),
                    context={"domain": name},
                )
            )
    for b in range(cfg.B):
        for d, name in enumerate(cfg.domains):
            value = int(C[b, d])
            if not allowed[b, d] and value != 0:
                violations.append(
                    _diagnostic(
                        "F3_C4",
                        f"C4 bucket={b},domain={name}",
                        f"forbidden/empty cell has C={value}, expected 0",
                        required=0,
                        available=value,
                        suggestions=("zero the forbidden cell and re-solve transportation",),
                        context={"bucket_idx": b, "domain": name},
                    )
                )
            c5 = int(static_caps[b, d])
            if value > c5:
                violations.append(
                    _diagnostic(
                        "F3_C5",
                        f"C5 bucket={b},domain={name}",
                        f"cell quota {value} exceeds repeat limit {c5} by {value - c5}",
                        required=value,
                        available=c5,
                        suggestions=(f"raise R_max[{name!r}], lower its mix, or add unique items",),
                        context={"bucket_idx": b, "domain": name, "N": int(cfg.N[b, d])},
                    )
                )
            c6 = int(cfg.N[b, d]) * int(S[b])
            if value > c6:
                violations.append(
                    _diagnostic(
                        "F3_C6",
                        f"C6 bucket={b},domain={name}",
                        f"cell quota {value} exceeds per-step unique capacity N*S={c6} by {value - c6}",
                        required=value,
                        available=c6,
                        suggestions=(
                            f"add at least {math.ceil(value / int(S[b])) if int(S[b]) else value} unique items to this cell",
                            f"move bucket steps, or lower domain_mix[{name!r}]",
                        ),
                        context={"bucket_idx": b, "domain": name, "N": int(cfg.N[b, d]), "S": int(S[b])},
                    )
                )
            if len(violations) >= 64:
                return violations
    return violations


def _solve_quota_milp(
    cfg: PlannerConfig,
    D: np.ndarray,
    Q: np.ndarray,
    allowed: np.ndarray,
    static_caps: np.ndarray,
    step_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve C1--C6 and weighted L1 deviation with scipy/HiGHS."""

    import scipy
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_array

    B, K = cfg.B, cfg.K
    n_s = B
    n_c = B * K
    n_a = B * K
    c_offset = n_s
    a_offset = n_s + n_c
    n_vars = n_s + n_c + n_a

    objective = np.zeros(n_vars, dtype=np.float64)
    for b in range(B):
        for d in range(K):
            if int(D[d]) > 0:
                # A common positive factor does not change the argmin.  Scaling
                # by M keeps coefficients O(1) even at production H, avoiding
                # HiGHS treating 1/D (~1e-9) as numerically negligible.
                objective[a_offset + b * K + d] = cfg.M / int(D[d])

    lower = np.zeros(n_vars, dtype=np.float64)
    upper = np.full(n_vars, np.inf, dtype=np.float64)
    upper[:B] = step_upper.astype(np.float64)
    for b in range(B):
        for d in range(K):
            upper[c_offset + b * K + d] = float(static_caps[b, d])
            upper[a_offset + b * K + d] = float(cfg.M)

    integrality = np.zeros(n_vars, dtype=np.uint8)
    integrality[: a_offset] = 1

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lbs: list[float] = []
    ubs: list[float] = []

    def add_constraint(coefficients: Iterable[tuple[int, float]], lb: float, ub: float) -> None:
        row = len(lbs)
        for col, value in coefficients:
            rows.append(row)
            cols.append(col)
            data.append(float(value))
        lbs.append(float(lb))
        ubs.append(float(ub))

    # C1.
    add_constraint(((b, 1.0) for b in range(B)), cfg.total_steps, cfg.total_steps)
    # C2.
    for b in range(B):
        coeffs = [(b, -float(cfg.G))]
        coeffs.extend((c_offset + b * K + d, 1.0) for d in range(K))
        add_constraint(coeffs, 0.0, 0.0)
    # C3.
    for d in range(K):
        add_constraint(
            ((c_offset + b * K + d, 1.0) for b in range(B)),
            int(D[d]),
            int(D[d]),
        )
    # C6: C[B,d] - N[B,d] * S[B] <= 0.  C4/C5 are bounds.
    for b in range(B):
        for d in range(K):
            add_constraint(
                (
                    (c_offset + b * K + d, 1.0),
                    (b, -float(int(cfg.N[b, d]))),
                ),
                -np.inf,
                0.0,
            )
    # Absolute-deviation epigraph: A >= C-Q and A >= Q-C.
    for b in range(B):
        for d in range(K):
            c_idx = c_offset + b * K + d
            a_idx = a_offset + b * K + d
            q = float(Q[b, d])
            add_constraint(((c_idx, 1.0), (a_idx, -1.0)), -np.inf, q)
            add_constraint(((c_idx, -1.0), (a_idx, -1.0)), -np.inf, -q)

    matrix = coo_array(
        (np.asarray(data), (np.asarray(rows), np.asarray(cols))),
        shape=(len(lbs), n_vars),
    ).tocsr()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, np.asarray(lbs), np.asarray(ubs)),
        options={"disp": False, "presolve": True, "mip_rel_gap": 0.0},
    )
    record: dict[str, Any] = {
        "path": "scipy.optimize.milp",
        "algorithm_version": QUOTA_MILP_VERSION,
        "scipy_version": scipy.__version__,
        "backend": "HiGHS",
        "status": int(result.status),
        "success": bool(result.success),
        "message": str(result.message),
        "objective": None if result.fun is None else float(result.fun) / cfg.M,
        "objective_scaled_solver": None if result.fun is None else float(result.fun),
        "objective_common_scale": cfg.M,
        "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        "mip_node_count": None if getattr(result, "mip_node_count", None) is None else int(result.mip_node_count),
        "repair_steps": 0,
    }
    if not result.success or result.x is None:
        routed: dict[str, Any] | None = None
        if int(result.status) == 2:
            # A second, always-feasible elastic MILP maximizes routed target
            # exposures with row/domain inequalities.  Its deficit is the
            # exact joint Hall/transport shortfall, including cases that all
            # scalar F1/F2/C6 bounds miss.
            routed = _milp_max_routable_diagnostic(cfg, D, static_caps, step_upper)
        available = None if routed is None else int(routed["routed_exposures"])
        missing_domains = {} if routed is None else routed["missing_by_domain"]
        diagnostics = [
            _diagnostic(
                "MILP_TRANSPORT_SHORTFALL" if int(result.status) == 2 else "MILP_SOLVER_FAILURE",
                "C1-C6 integer transportation",
                (
                    f"joint bucket/domain transport can route only {available}/{cfg.M} target exposures; "
                    f"missing_by_domain={missing_domains}"
                    if available is not None
                    else f"scipy.optimize.milp/HiGHS did not produce a feasible quota solution: {result.message}"
                ),
                required=cfg.M,
                available=available,
                suggestions=(
                    "lower the mix of domains listed in missing_by_domain or move their items to additional compatible buckets",
                    "lower total_steps if repeat supply is binding, or raise R_max/add unique items",
                    "revise the bucket table/assignment so each active bucket can fill G unique slots",
                ),
                context={
                    "solver_status": int(result.status),
                    "solver_message": str(result.message),
                    "elastic_transport": routed,
                },
            )
        ]
        raise PlanInfeasibleError(diagnostics)

    rounded = np.rint(result.x[:a_offset])
    max_error = float(np.max(np.abs(result.x[:a_offset] - rounded), initial=0.0))
    if max_error > 1e-5:
        raise RuntimeError(f"MILP returned a non-integral solution beyond tolerance: max_error={max_error}")
    S = rounded[:B].astype(np.int64)
    C = rounded[B:].reshape(B, K).astype(np.int64)
    record["objective_recomputed"] = _quota_objective(C, Q, D)
    record["max_integrality_error"] = max_error
    return S, C, record


def _milp_max_routable_diagnostic(
    cfg: PlannerConfig,
    D: np.ndarray,
    static_caps: np.ndarray,
    step_upper: np.ndarray,
) -> dict[str, Any]:
    """Return the exact maximum target exposure routable under relaxed rows."""

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_array

    B, K = cfg.B, cfg.K
    c_offset = B
    n_vars = B + B * K
    objective = np.zeros(n_vars, dtype=np.float64)
    objective[c_offset:] = -1.0
    lower = np.zeros(n_vars, dtype=np.float64)
    upper = np.empty(n_vars, dtype=np.float64)
    upper[:B] = step_upper.astype(np.float64)
    upper[c_offset:] = static_caps.reshape(-1).astype(np.float64)
    integrality = np.ones(n_vars, dtype=np.uint8)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lbs: list[float] = []
    ubs: list[float] = []

    def add(coefficients: Iterable[tuple[int, float]], lb: float, ub: float) -> None:
        row = len(lbs)
        for col, value in coefficients:
            rows.append(row)
            cols.append(col)
            data.append(float(value))
        lbs.append(float(lb))
        ubs.append(float(ub))

    add(((b, 1.0) for b in range(B)), cfg.total_steps, cfg.total_steps)
    for b in range(B):
        coefficients = [(b, -float(cfg.G))]
        coefficients.extend((c_offset + b * K + d, 1.0) for d in range(K))
        add(coefficients, -np.inf, 0.0)
    for d in range(K):
        add(((c_offset + b * K + d, 1.0) for b in range(B)), -np.inf, int(D[d]))
    for b in range(B):
        for d in range(K):
            add(
                (
                    (c_offset + b * K + d, 1.0),
                    (b, -float(int(cfg.N[b, d]))),
                ),
                -np.inf,
                0.0,
            )
    matrix = coo_array(
        (np.asarray(data), (np.asarray(rows), np.asarray(cols))),
        shape=(len(lbs), n_vars),
    ).tocsr()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, np.asarray(lbs), np.asarray(ubs)),
        options={"disp": False, "presolve": True, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        return {
            "routed_exposures": 0,
            "missing_by_domain": {name: int(D[d]) for d, name in enumerate(cfg.domains) if int(D[d])},
            "diagnostic_solver_status": int(result.status),
            "diagnostic_solver_message": str(result.message),
        }
    rounded = np.rint(result.x).astype(np.int64)
    S_diag = rounded[:B]
    C_diag = rounded[B:].reshape(B, K)
    routed_domains = C_diag.sum(axis=0, dtype=np.int64)
    routed_total = int(routed_domains.sum())
    missing = {
        name: int(D[d] - routed_domains[d])
        for d, name in enumerate(cfg.domains)
        if int(D[d] - routed_domains[d]) > 0
    }
    return {
        "routed_exposures": routed_total,
        "shortfall": cfg.M - routed_total,
        "routed_by_domain": {name: int(routed_domains[d]) for d, name in enumerate(cfg.domains)},
        "missing_by_domain": missing,
        "diagnostic_bucket_steps": S_diag.tolist(),
        "diagnostic_solver_status": int(result.status),
        "diagnostic_solver_message": str(result.message),
    }


class _Dinic:
    def __init__(self, n: int):
        self.graph: list[list[list[int]]] = [[] for _ in range(n)]

    def add_edge(self, source: int, target: int, capacity: int) -> None:
        forward = [target, len(self.graph[target]), int(capacity)]
        reverse = [source, len(self.graph[source]), 0]
        self.graph[source].append(forward)
        self.graph[target].append(reverse)

    def max_flow(self, source: int, sink: int) -> int:
        total = 0
        n = len(self.graph)
        while True:
            level = [-1] * n
            level[source] = 0
            queue: deque[int] = deque([source])
            while queue:
                node = queue.popleft()
                for target, _reverse, capacity in self.graph[node]:
                    if capacity > 0 and level[target] < 0:
                        level[target] = level[node] + 1
                        queue.append(target)
            if level[sink] < 0:
                return total
            cursor = [0] * n

            def send(node: int, amount: int) -> int:
                if node == sink:
                    return amount
                while cursor[node] < len(self.graph[node]):
                    edge = self.graph[node][cursor[node]]
                    target, reverse, capacity = edge
                    if capacity > 0 and level[target] == level[node] + 1:
                        pushed = send(target, min(amount, capacity))
                        if pushed:
                            edge[2] -= pushed
                            self.graph[target][reverse][2] += pushed
                            return pushed
                    cursor[node] += 1
                return 0

            while True:
                pushed = send(source, 1 << 62)
                if not pushed:
                    break
                total += pushed


def _fixed_cell_caps(
    cfg: PlannerConfig,
    S: np.ndarray,
    static_caps: np.ndarray,
) -> np.ndarray:
    caps = np.empty_like(static_caps)
    for b in range(cfg.B):
        for d in range(cfg.K):
            caps[b, d] = min(int(static_caps[b, d]), int(cfg.N[b, d]) * int(S[b]))
    return caps


def _transport_flow_value(cfg: PlannerConfig, D: np.ndarray, S: np.ndarray, caps: np.ndarray) -> int:
    source = 0
    bucket_offset = 1
    domain_offset = bucket_offset + cfg.B
    sink = domain_offset + cfg.K
    flow = _Dinic(sink + 1)
    for b in range(cfg.B):
        flow.add_edge(source, bucket_offset + b, cfg.G * int(S[b]))
        for d in range(cfg.K):
            if int(caps[b, d]) > 0:
                flow.add_edge(bucket_offset + b, domain_offset + d, int(caps[b, d]))
    for d in range(cfg.K):
        flow.add_edge(domain_offset + d, sink, int(D[d]))
    return flow.max_flow(source, sink)


def _bounded_step_largest_remainder(Q: np.ndarray, cfg: PlannerConfig, upper: np.ndarray) -> np.ndarray:
    ideal = Q.sum(axis=1) / float(cfg.G)
    S = np.minimum(np.floor(ideal).astype(np.int64), upper)
    left = cfg.total_steps - int(S.sum())
    heap: list[tuple[float, int]] = []
    for b in range(cfg.B):
        if int(S[b]) < int(upper[b]):
            current = int(S[b])
            marginal = abs((current + 1) - ideal[b]) - abs(current - ideal[b])
            heapq.heappush(heap, (float(marginal), b))
    while left:
        if not heap:
            raise PlanInfeasibleError(
                [
                    _diagnostic(
                        "FALLBACK_S_CAPACITY",
                        "C1 bounded largest-remainder",
                        "fallback cannot place all H bucket steps within C5/C6 bucket bounds",
                        required=cfg.total_steps,
                        available=int(S.sum()),
                        suggestions=("increase usable cell capacity or lower total_steps",),
                    )
                ]
            )
        _marginal, b = heapq.heappop(heap)
        S[b] += 1
        left -= 1
        if int(S[b]) < int(upper[b]):
            current = int(S[b])
            marginal = abs((current + 1) - ideal[b]) - abs(current - ideal[b])
            heapq.heappush(heap, (float(marginal), b))
    return S


def _bounded_row_largest_remainder(target: np.ndarray, total: int, upper: np.ndarray) -> np.ndarray:
    row = np.minimum(np.floor(target).astype(np.int64), upper)
    current_total = int(row.sum())
    if current_total < total:
        heap: list[tuple[float, int]] = []
        for d in range(len(row)):
            if int(row[d]) < int(upper[d]):
                value = int(row[d])
                delta = abs((value + 1) - target[d]) - abs(value - target[d])
                heapq.heappush(heap, (float(delta), d))
        while current_total < total:
            if not heap:
                raise ValueError("row upper bounds cannot supply its bucket total")
            _delta, d = heapq.heappop(heap)
            row[d] += 1
            current_total += 1
            if int(row[d]) < int(upper[d]):
                value = int(row[d])
                delta = abs((value + 1) - target[d]) - abs(value - target[d])
                heapq.heappush(heap, (float(delta), d))
    elif current_total > total:
        heap = []
        for d in range(len(row)):
            if int(row[d]) > 0:
                value = int(row[d])
                delta = abs((value - 1) - target[d]) - abs(value - target[d])
                heapq.heappush(heap, (float(delta), d))
        while current_total > total:
            if not heap:
                raise ValueError("row cannot be reduced to its bucket total")
            _delta, d = heapq.heappop(heap)
            row[d] -= 1
            current_total -= 1
            if int(row[d]) > 0:
                value = int(row[d])
                delta = abs((value - 1) - target[d]) - abs(value - target[d])
                heapq.heappush(heap, (float(delta), d))
    return row


def _edge_damage(C: np.ndarray, Q: np.ndarray, D: np.ndarray, b: int, donor: int, receiver: int) -> float:
    damage = 0.0
    donor_value = int(C[b, donor])
    receiver_value = int(C[b, receiver])
    if int(D[donor]) > 0:
        damage += (
            abs((donor_value - 1) - Q[b, donor]) - abs(donor_value - Q[b, donor])
        ) / int(D[donor])
    if int(D[receiver]) > 0:
        damage += (
            abs((receiver_value + 1) - Q[b, receiver]) - abs(receiver_value - Q[b, receiver])
        ) / int(D[receiver])
    return float(damage)


def _best_residual_path(
    C: np.ndarray,
    caps: np.ndarray,
    Q: np.ndarray,
    D: np.ndarray,
    residual: np.ndarray,
) -> list[tuple[int, int, int]] | None:
    """Find a deterministic residual path donor-domain -> receiver-domain."""

    K = C.shape[1]
    surplus = [d for d in range(K) if int(residual[d]) < 0]
    deficit = {d for d in range(K) if int(residual[d]) > 0}
    edges: dict[tuple[int, int], tuple[float, int]] = {}
    for donor in range(K):
        for receiver in range(K):
            if donor == receiver:
                continue
            candidates: list[tuple[float, int]] = []
            for b in range(C.shape[0]):
                if int(C[b, donor]) > 0 and int(C[b, receiver]) < int(caps[b, receiver]):
                    candidates.append((_edge_damage(C, Q, D, b, donor, receiver), b))
            if candidates:
                edges[(donor, receiver)] = min(candidates)

    # The design asks for the least damaging cell pair.  Prefer a direct pair
    # whenever one repairs both a surplus and a deficit column.
    direct: list[tuple[float, int, int, int]] = []
    for donor in surplus:
        for receiver in sorted(deficit):
            edge = edges.get((donor, receiver))
            if edge is not None:
                direct.append((edge[0], edge[1], donor, receiver))
    if direct:
        _cost, b, donor, receiver = min(direct)
        return [(b, donor, receiver)]

    # If no direct pair exists, an alternating domain/bucket residual path is
    # required.  BFS gives the shortest deterministic augmenting path; outgoing
    # arcs are ordered by objective damage, then indices.
    queue: deque[int] = deque(sorted(surplus))
    predecessor: dict[int, tuple[int, int]] = {}
    seen = set(surplus)
    reached: int | None = None
    while queue and reached is None:
        donor = queue.popleft()
        outgoing: list[tuple[float, int, int]] = []
        for receiver in range(K):
            edge = edges.get((donor, receiver))
            if edge is not None and receiver not in seen:
                outgoing.append((edge[0], edge[1], receiver))
        for _cost, b, receiver in sorted(outgoing):
            seen.add(receiver)
            predecessor[receiver] = (donor, b)
            if receiver in deficit:
                reached = receiver
                break
            queue.append(receiver)
    if reached is None:
        return None
    path: list[tuple[int, int, int]] = []
    current = reached
    while current not in surplus:
        previous, b = predecessor[current]
        path.append((b, previous, current))
        current = previous
    path.reverse()
    return path


def _repair_transport_columns(
    C: np.ndarray,
    caps: np.ndarray,
    Q: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    residual = D.astype(np.int64) - C.sum(axis=0, dtype=np.int64)
    augmentations = 0
    unit_shifts = 0
    while np.any(residual != 0):
        path = _best_residual_path(C, caps, Q, D, residual)
        if path is None:
            raise ValueError(f"transport residual graph is stuck: residual={residual.tolist()}")
        source_domain = path[0][1]
        target_domain = path[-1][2]
        for b, donor, receiver in path:
            C[b, donor] -= 1
            C[b, receiver] += 1
            unit_shifts += 1
        residual[source_domain] += 1
        residual[target_domain] -= 1
        augmentations += 1
    return C, augmentations, unit_shifts


def _fallback_fixed_s(
    cfg: PlannerConfig,
    D: np.ndarray,
    Q: np.ndarray,
    S: np.ndarray,
    static_caps: np.ndarray,
) -> tuple[np.ndarray, int, int] | None:
    caps = _fixed_cell_caps(cfg, S, static_caps)
    if _transport_flow_value(cfg, D, S, caps) != cfg.M:
        return None
    C = np.zeros((cfg.B, cfg.K), dtype=np.int64)
    try:
        for b in range(cfg.B):
            C[b] = _bounded_row_largest_remainder(Q[b], cfg.G * int(S[b]), caps[b])
        return _repair_transport_columns(C, caps, Q, D)
    except ValueError as exc:
        raise RuntimeError(f"fallback residual repair failed despite a feasible max-flow: {exc}") from exc


def _solve_quota_fallback(
    cfg: PlannerConfig,
    D: np.ndarray,
    Q: np.ndarray,
    allowed: np.ndarray,
    static_caps: np.ndarray,
    step_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Two-level largest-remainder plus deterministic transportation repair."""

    del allowed  # C4 is already encoded in static_caps.
    S = _bounded_step_largest_remainder(Q, cfg, step_upper)
    ideal_steps = Q.sum(axis=1) / float(cfg.G)
    visited: set[tuple[int, ...]] = set()
    relocations = 0
    max_iterations = max(cfg.total_steps, cfg.B * 4)
    for _iteration in range(max_iterations + 1):
        state = tuple(int(value) for value in S)
        if state in visited:
            break
        visited.add(state)
        solved = _fallback_fixed_s(cfg, D, Q, S, static_caps)
        if solved is not None:
            C, augmentations, unit_shifts = solved
            record = {
                "path": "deterministic_fallback",
                "algorithm_version": QUOTA_FALLBACK_VERSION,
                "objective": _quota_objective(C, Q, D),
                "repair_steps": relocations + unit_shifts,
                "bucket_step_relocations": relocations,
                "transport_augmentations": augmentations,
                "transport_unit_shifts": unit_shifts,
                "visited_step_allocations": len(visited),
            }
            return S, C, record

        current_caps = _fixed_cell_caps(cfg, S, static_caps)
        current_flow = _transport_flow_value(cfg, D, S, current_caps)
        candidates: list[tuple[int, float, int, int, np.ndarray]] = []
        for donor in range(cfg.B):
            if int(S[donor]) == 0:
                continue
            for receiver in range(cfg.B):
                if donor == receiver or int(S[receiver]) >= int(step_upper[receiver]):
                    continue
                candidate = S.copy()
                candidate[donor] -= 1
                candidate[receiver] += 1
                candidate_state = tuple(int(value) for value in candidate)
                if candidate_state in visited:
                    continue
                caps = _fixed_cell_caps(cfg, candidate, static_caps)
                flow_value = _transport_flow_value(cfg, D, candidate, caps)
                old_damage = abs(int(S[donor]) - ideal_steps[donor]) + abs(int(S[receiver]) - ideal_steps[receiver])
                new_damage = abs(int(candidate[donor]) - ideal_steps[donor]) + abs(int(candidate[receiver]) - ideal_steps[receiver])
                candidates.append((-flow_value, float(new_damage - old_damage), donor, receiver, candidate))
        if not candidates:
            break
        best = min(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))
        best_flow = -best[0]
        # A plateau move is allowed to escape a one-step local cut, but visited
        # states make the repair finite and deterministic.
        if best_flow < current_flow:
            break
        S = best[4]
        relocations += 1

    caps = _fixed_cell_caps(cfg, S, static_caps)
    available = _transport_flow_value(cfg, D, S, caps)
    raise PlanInfeasibleError(
        [
            _diagnostic(
                "FALLBACK_TRANSPORT_INFEASIBLE",
                "C1-C6 fallback transportation",
                f"two-level largest-remainder plus {relocations} bucket-step relocations could route {available}/{cfg.M} exposures",
                required=cfg.M,
                available=available,
                suggestions=(
                    "lower the mix of domains with positive residual demand",
                    "increase R_max or unique N in their allowed cells",
                    "revise the bucket table; scipy.optimize.milp is recommended for adversarial layouts",
                ),
                context={"S": S.tolist(), "bucket_step_relocations": relocations},
            )
        ]
    )


def _solve_quotas(
    cfg: PlannerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    D = _derive_domain_totals(cfg)
    allowed = _allowed_mask(cfg)
    static_caps = _static_caps(cfg, allowed)
    Q = _ideal_joint(cfg, D, allowed)
    step_upper, diagnostics = _feasibility_precheck(cfg, D, allowed, static_caps)
    if diagnostics:
        raise PlanInfeasibleError(diagnostics)

    try:
        from scipy.optimize import milp as _milp  # noqa: F401
    except (ImportError, AttributeError):
        S, C, solver_record = _solve_quota_fallback(
            cfg, D, Q, allowed, static_caps, step_upper
        )
    else:
        S, C, solver_record = _solve_quota_milp(
            cfg, D, Q, allowed, static_caps, step_upper
        )
    violations = _quota_violations(cfg, D, S, C, allowed, static_caps)
    if violations:
        raise PlanInfeasibleError(violations)
    return D, S, C, Q, solver_record, allowed, static_caps


def _seed_digest(label: str, cfg: PlannerConfig, *parts: int) -> bytes:
    payload = {
        "algorithm": PLANNER_ALGORITHM_VERSION,
        "label": label,
        "plan_id": cfg.plan_id,
        "seed": cfg.seed,
        "parts": list(parts),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).digest()


def _bucket_schedule(cfg: PlannerConfig, S: np.ndarray) -> np.ndarray:
    """Exact integer implementation of design section 2's argmax formula."""

    used = np.zeros(cfg.B, dtype=np.int64)
    schedule = np.empty(cfg.total_steps, dtype=np.int16)
    jitter = [int.from_bytes(_seed_digest("bucket_epsilon_jitter", cfg, b), "big") for b in range(cfg.B)]
    for t in range(cfg.total_steps):
        best_b = 0
        best_score = int(S[0]) * (t + 1) - int(used[0]) * cfg.total_steps
        for b in range(1, cfg.B):
            score = int(S[b]) * (t + 1) - int(used[b]) * cfg.total_steps
            if score > best_score:
                best_b, best_score = b, score
            elif score == best_score:
                if cfg.epsilon_jitter:
                    if jitter[b] > jitter[best_b] or (jitter[b] == jitter[best_b] and b < best_b):
                        best_b = b
                elif b < best_b:
                    best_b = b
        schedule[t] = best_b
        used[best_b] += 1
    if not np.array_equal(used, S):
        raise AssertionError(f"bucket schedule does not conserve S: used={used.tolist()}, S={S.tolist()}")
    return schedule


def _per_step_domain_quotas(
    cfg: PlannerConfig,
    C: np.ndarray,
    S: np.ndarray,
) -> tuple[list[np.ndarray], int]:
    """Column Bresenham plus the §9.4 remaining-quota repair.

    Independent Bresenham columns conserve C but do not generally sum to G on
    every step.  A correction therefore moves one ``+1`` event within the same
    domain between the earliest overfull/underfull rows.  This realizes the
    specified per-step +/-1 operation without breaking ``sum_k c_k[d]=C[B,d]``
    or C6.  At every unit move, eligible domains are ordered by their integer
    quota remaining after the earlier affected step, descending, then by input
    domain index ascending.
    """

    result: list[np.ndarray] = []
    repair_moves = 0
    for b in range(cfg.B):
        steps = int(S[b])
        if steps == 0:
            result.append(np.zeros((0, cfg.K), dtype=np.int64))
            continue
        base = C[b] // steps
        remainder = C[b] % steps
        quotas = np.tile(base, (steps, 1)).astype(np.int64, copy=False)
        for d in range(cfg.K):
            accumulator = 0
            rem = int(remainder[d])
            for k in range(steps):
                accumulator += rem
                if accumulator >= steps:
                    quotas[k, d] += 1
                    accumulator -= steps
            if accumulator != 0:
                raise AssertionError("per-domain Bresenham accumulator did not close")

        row_sums = quotas.sum(axis=1, dtype=np.int64)
        over = deque(k for k in range(steps) if int(row_sums[k]) > cfg.G)
        under = deque(k for k in range(steps) if int(row_sums[k]) < cfg.G)
        consumed_before = np.zeros(cfg.K, dtype=np.int64)
        prefix_cursor = 0
        while over and under:
            donor_step = over[0]
            receiver_step = under[0]
            repair_step = min(donor_step, receiver_step)
            while prefix_cursor < repair_step:
                consumed_before += quotas[prefix_cursor]
                prefix_cursor += 1
            remaining = C[b] - consumed_before - quotas[repair_step]
            if np.any(remaining < 0):
                raise AssertionError(
                    f"negative C_rem during per-step repair for bucket={b},step={repair_step}"
                )
            candidates: list[int] = []
            for d in range(cfg.K):
                if int(quotas[donor_step, d]) == int(base[d]) + 1 and int(quotas[receiver_step, d]) == int(base[d]):
                    candidates.append(d)
            if not candidates:
                raise AssertionError(
                    f"no column-preserving step repair for bucket={b}, donor={donor_step}, receiver={receiver_step}"
                )
            d = min(candidates, key=lambda candidate: (-int(remaining[candidate]), candidate))
            quotas[donor_step, d] -= 1
            quotas[receiver_step, d] += 1
            row_sums[donor_step] -= 1
            row_sums[receiver_step] += 1
            repair_moves += 1
            if int(row_sums[donor_step]) == cfg.G:
                over.popleft()
            if int(row_sums[receiver_step]) == cfg.G:
                under.popleft()
        if over or under or np.any(row_sums != cfg.G):
            raise AssertionError(f"per-step quota row repair did not converge for bucket {b}")
        if not np.array_equal(quotas.sum(axis=0, dtype=np.int64), C[b]):
            raise AssertionError(f"per-step domain quotas do not conserve C for bucket {b}")
        for d in range(cfg.K):
            if int(quotas[:, d].max(initial=0)) > int(cfg.N[b, d]):
                raise AssertionError(f"per-step C6 violation after repair at bucket={b},domain={d}")
        result.append(quotas)
    return result, repair_moves


def _splitmix64_array(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = values.astype(np.uint64, copy=False) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _feistel_round_keys(cfg: PlannerConfig, b: int, d: int, cycle: int) -> tuple[int, ...]:
    master = _seed_digest("feistel_cell_cycle", cfg, b, d, cycle)
    return tuple(
        int.from_bytes(hashlib.sha256(master + bytes((round_idx,))).digest()[:8], "little")
        for round_idx in range(FEISTEL_ROUNDS)
    )


def _feistel_power_of_four(values: np.ndarray, half_bits: int, round_keys: Sequence[int]) -> np.ndarray:
    mask_int = (1 << half_bits) - 1
    mask = np.uint64(mask_int)
    left = values >> np.uint64(half_bits)
    right = values & mask
    for key in round_keys:
        mixed = _splitmix64_array(right ^ np.uint64(key)) & mask
        left, right = right, (left ^ mixed) & mask
    return (left << np.uint64(half_bits)) | right


def _feistel_permute_many(
    positions: np.ndarray,
    n: int,
    round_keys: Sequence[int],
) -> np.ndarray:
    if n <= 0:
        raise ValueError("cannot permute an empty cell")
    if n == 1:
        return np.zeros(positions.shape, dtype=np.int64)
    bits = (n - 1).bit_length()
    if bits % 2:
        bits += 1
    half_bits = bits // 2
    current = positions.astype(np.uint64, copy=True)
    active = np.ones(current.shape, dtype=bool)
    while bool(np.any(active)):
        current[active] = _feistel_power_of_four(current[active], half_bits, round_keys)
        active = current >= np.uint64(n)
    return current.astype(np.int64)


class _CellStream:
    """Lazy deterministic cycle-by-cycle PRP consumer for one (B,d) cell."""

    def __init__(
        self,
        cfg: PlannerConfig,
        b: int,
        d: int,
        row_base: int,
        stats: dict[str, int],
    ):
        self.cfg = cfg
        self.b = b
        self.d = d
        self.n = int(cfg.N[b, d])
        self.row_base = int(row_base)
        self.stats = stats
        self.cycle = 0
        self.position = 0
        self.swaps: dict[int, int] = {}
        self.round_keys = _feistel_round_keys(cfg, b, d, self.cycle) if self.n else ()
        if self.n:
            self.stats["feistel_cycles_created"] += 1

    def _next_cycle(self) -> None:
        self.cycle += 1
        self.position = 0
        self.swaps = {}
        self.round_keys = _feistel_round_keys(self.cfg, self.b, self.d, self.cycle)
        self.stats["feistel_cycles_created"] += 1

    def _base_values(self, start: int, stop: int) -> np.ndarray:
        positions = np.arange(start, stop, dtype=np.uint64)
        return _feistel_permute_many(positions, self.n, self.round_keys)

    def _values(self, start: int, stop: int) -> np.ndarray:
        values = self._base_values(start, stop)
        if self.swaps:
            for offset, position in enumerate(range(start, stop)):
                swapped = self.swaps.get(position)
                if swapped is not None:
                    values[offset] = swapped
        return values

    def _prepare_boundary_prefix(self, prefix_len: int, forbidden: set[int]) -> None:
        """Swap collisions out of a new cycle's consumed prefix.

        Different cycle keys alone do not make a batch crossing a cycle boundary
        unique.  This deterministic prefix/tail swap retains a full permutation
        for the new cycle while ensuring its immediately consumed prefix avoids
        the previous cycle's tail in the current global batch.
        """

        if prefix_len <= 0 or not forbidden:
            return
        prefix = self._base_values(0, prefix_len)
        bad = [i for i, value in enumerate(prefix.tolist()) if int(value) in forbidden]
        if not bad:
            return
        replacements: list[tuple[int, int]] = []
        scan = prefix_len
        while len(replacements) < len(bad) and scan < self.n:
            stop = min(self.n, scan + 4096)
            values = self._base_values(scan, stop)
            for offset, value_raw in enumerate(values.tolist()):
                value = int(value_raw)
                if value not in forbidden:
                    replacements.append((scan + offset, value))
                    if len(replacements) == len(bad):
                        break
            scan = stop
        if len(replacements) != len(bad):
            raise AssertionError("cycle-boundary de-duplication could not find enough replacement items")
        for bad_position, (replacement_position, replacement_value) in zip(bad, replacements):
            collision_value = int(prefix[bad_position])
            self.swaps[bad_position] = replacement_value
            self.swaps[replacement_position] = collision_value
            self.stats["cycle_boundary_swaps"] += 1

    def take(self, count: int) -> np.ndarray:
        if count < 0 or count > self.n:
            raise AssertionError(f"cell ({self.b},{self.d}) take={count} violates C6 N={self.n}")
        if count == 0:
            return np.empty(0, dtype=np.int64)
        selected: list[int] = []
        selected_set: set[int] = set()
        remaining = count
        while remaining:
            if self.position == self.n:
                self._next_cycle()
            if self.position == 0 and selected_set:
                self._prepare_boundary_prefix(remaining, selected_set)
            take_now = min(remaining, self.n - self.position)
            values = self._values(self.position, self.position + take_now)
            for raw in values.tolist():
                value = int(raw)
                if value in selected_set:
                    raise AssertionError(
                        f"duplicate item inside one global batch at cell ({self.b},{self.d}), item={value}"
                    )
                selected.append(value)
                selected_set.add(value)
            self.position += take_now
            remaining -= take_now
        if len(selected) != count:
            raise AssertionError("cell stream returned the wrong number of items")
        return np.asarray(selected, dtype=np.int64) + self.row_base


def _cell_row_bases(cfg: PlannerConfig) -> np.ndarray:
    """Return base[d] + cellbase[d][B] in the frozen virtual catalog."""

    bases = np.zeros((cfg.B, cfg.K), dtype=np.int64)
    cursor = 0
    for d in range(cfg.K):
        for b in range(cfg.B):
            bases[b, d] = cursor
            cursor += int(cfg.N[b, d])
    if cursor > _INT64_MAX:
        raise PlannerInputError("virtual catalog row-ID prefix sum exceeds int64")
    return bases


def _domain_item_totals(cfg: PlannerConfig) -> list[int]:
    return [sum(int(cfg.N[b, d]) for b in range(cfg.B)) for d in range(cfg.K)]


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        np.ascontiguousarray(array),
        version=(2, 0),
        allow_pickle=False,
    )
    return buffer.getvalue()


def _write_deterministic_npz(path: Path, arrays: Sequence[tuple[str, np.ndarray]]) -> None:
    """Write an NPZ without wall-clock ZIP metadata."""

    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise PlannerInputError(f"refusing to overwrite stale temporary file: {temporary}")
    try:
        with zipfile.ZipFile(temporary, mode="x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, array in arrays:
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                info.flag_bits = 0
                archive.writestr(info, _npy_bytes(array), compress_type=zipfile.ZIP_STORED)
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise PlannerInputError(f"refusing to overwrite output file: {path}")
    data = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _materialize_blocks(
    output: Path,
    cfg: PlannerConfig,
    D: np.ndarray,
    S: np.ndarray,
    C: np.ndarray,
    schedule: np.ndarray,
    per_bucket_quotas: Sequence[np.ndarray],
    start_step: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    blocks_dir = output / "blocks"
    blocks_dir.mkdir()
    row_bases = _cell_row_bases(cfg)
    stats: dict[str, int] = {
        "feistel_cycles_created": 0,
        "cycle_boundary_swaps": 0,
        "global_batches_explicitly_checked_unique": 0,
        "rank_balance_checks": 0,
    }
    streams: dict[tuple[int, int], _CellStream] = {}
    for b in range(cfg.B):
        for d in range(cfg.K):
            if int(C[b, d]) > 0:
                streams[(b, d)] = _CellStream(cfg, b, d, int(row_bases[b, d]), stats)

    occurrence = np.zeros(cfg.B, dtype=np.int64)
    realized_domains = np.zeros(cfg.K, dtype=np.int64)
    checksums: dict[str, str] = {}
    num_blocks = (cfg.total_steps + cfg.block_steps - 1) // cfg.block_steps
    first_block_index = start_step // cfg.block_steps
    for block_idx in range(num_blocks):
        start = block_idx * cfg.block_steps
        stop = min(cfg.total_steps, start + cfg.block_steps)
        count = stop - start
        write_block = block_idx >= first_block_index
        if write_block:
            bucket_of_step = schedule[start:stop].astype(np.int16, copy=True)
            item_row = np.empty(
                (count, cfg.world_size, cfg.per_rank_bs),
                dtype=np.int64,
            )
            domain_counts = np.empty((count, cfg.K), dtype=np.int32)
        else:
            bucket_of_step = None
            item_row = None
            domain_counts = None
        for local_step, global_step in enumerate(range(start, stop)):
            b = int(schedule[global_step])
            bucket_occurrence = int(occurrence[b])
            quotas = per_bucket_quotas[b][bucket_occurrence]
            occurrence[b] += 1
            if int(quotas.sum()) != cfg.G:
                raise AssertionError(f"step {global_step} quotas sum to {int(quotas.sum())}, expected G={cfg.G}")

            parts: list[np.ndarray] = []
            domain_slots: list[np.ndarray] = []
            for d in range(cfg.K):
                cell_count = int(quotas[d])
                if cell_count:
                    parts.append(streams[(b, d)].take(cell_count))
                    domain_slots.append(np.full(cell_count, d, dtype=np.int32))
                realized_domains[d] += cell_count
            flat_items = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
            flat_domains = np.concatenate(domain_slots) if domain_slots else np.empty(0, dtype=np.int32)
            if flat_items.shape != (cfg.G,) or flat_domains.shape != (cfg.G,):
                raise AssertionError(f"step {global_step} materialized the wrong global batch shape")
            if int(np.unique(flat_items).size) != cfg.G:
                raise AssertionError(f"step {global_step} has a duplicate global row ID")
            stats["global_batches_explicitly_checked_unique"] += 1

            dealt_domains = flat_domains.reshape(cfg.per_rank_bs, cfg.world_size).T
            for d in range(cfg.K):
                rank_counts = np.count_nonzero(dealt_domains == d, axis=1)
                if int(rank_counts.max()) - int(rank_counts.min()) > 1:
                    raise AssertionError(f"rank domain imbalance at step={global_step},domain={d}")
            stats["rank_balance_checks"] += 1
            # v[r], v[r+W], ... is reshape(bs,W).T.
            if write_block:
                assert item_row is not None and domain_counts is not None
                item_row[local_step] = flat_items.reshape(cfg.per_rank_bs, cfg.world_size).T
                domain_counts[local_step] = quotas

        if write_block:
            assert bucket_of_step is not None and item_row is not None and domain_counts is not None
            if not np.all(domain_counts.sum(axis=1, dtype=np.int64) == cfg.G):
                raise AssertionError(f"domain_counts rows do not sum to G in block {block_idx}")
            filename = f"block_{block_idx:06d}.npz"
            block_path = blocks_dir / filename
            _write_deterministic_npz(
                block_path,
                (
                    ("bucket_of_step", bucket_of_step),
                    ("item_row", item_row),
                    ("domain_counts", domain_counts),
                ),
            )
            checksums[filename] = _file_sha256(block_path)

    if not np.array_equal(occurrence, S):
        raise AssertionError(f"materialized bucket counts {occurrence.tolist()} != S {S.tolist()}")
    if not np.array_equal(realized_domains, D):
        raise AssertionError(f"materialized domain totals {realized_domains.tolist()} != D {D.tolist()}")
    stats_out: dict[str, Any] = dict(stats)
    stats_out["realized_domain_exposures"] = realized_domains.tolist()
    stats_out["realized_bucket_steps"] = occurrence.tolist()
    stats_out["num_blocks"] = num_blocks
    stats_out["materialized_num_blocks"] = len(checksums)
    stats_out["requested_start_step"] = start_step
    stats_out["first_materialized_block"] = first_block_index
    stats_out["first_materialized_step"] = first_block_index * cfg.block_steps
    return checksums, stats_out


def _algorithm_versions(cfg: PlannerConfig) -> dict[str, Any]:
    return {
        "planner": PLANNER_ALGORITHM_VERSION,
        "quota_milp": QUOTA_MILP_VERSION,
        "quota_fallback": QUOTA_FALLBACK_VERSION,
        "bucket_schedule": BUCKET_SCHEDULE_VERSION,
        "bucket_epsilon_jitter": "sha256_fixed_bucket_tiebreak_v1" if cfg.epsilon_jitter else "disabled",
        "per_step_domain_quota": PER_STEP_QUOTA_VERSION,
        "rank_deal": RANK_DEAL_VERSION,
        "item_permutation": FEISTEL_VERSION,
        "feistel_rounds": FEISTEL_ROUNDS,
        "feistel_key_derivation": "sha256(canonical_json(plan_id,seed,B,d,cycle))+sha256(master||round)_le64_v1",
        "feistel_domain": "next_even_bit_power_of_four_cycle_walk_v1",
        "cycle_boundary_dedup": "new_cycle_prefix_tail_swap_v1",
        "npz": NPZ_FORMAT_VERSION,
        "checksums": CHECKSUMS_FORMAT_VERSION,
        "row_id_mapping": ROW_ID_MAPPING_VERSION,
    }


def _resume_contract(cfg: PlannerConfig) -> dict[str, Any]:
    return {
        "bound_world_size": cfg.world_size,
        "bound_per_rank_bs": cfg.per_rank_bs,
        "checkpoint_field": "completed_optimizer_step",
        "first_step": "completed_optimizer_step + 1",
        "prefetch": "discard_and_cold_rebuild",
        "world_size_change": "rebuild_with_new_plan_id",
    }


def _effective_catalog_assignment_hashes(cfg: PlannerConfig) -> tuple[str, str, dict[str, str]]:
    """Return real upstream hashes or explicitly-labelled virtual proxies.

    Section 1.5 omits upstream hashes.  A non-empty deterministic proxy keeps
    the plan identity closed for N-only Phase-4 inputs without pretending it
    is an L1/L2 snapshot hash; ``hash_sources`` in the header makes the origin
    unambiguous.
    """

    if cfg.catalog_hash is None:
        catalog_hash = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "algorithm": "virtual_catalog_proxy_from_N_v1",
                    "row_id_mapping": ROW_ID_MAPPING_VERSION,
                    "domains": list(cfg.domains),
                    "buckets": [list(hw) for hw in cfg.buckets],
                    "N": cfg.N.tolist(),
                }
            )
        )
        catalog_source = "virtual_catalog_proxy_from_N (not an upstream L1 catalog hash)"
    else:
        catalog_hash = cfg.catalog_hash
        catalog_source = "planner_input.catalog_hash"
    if cfg.assignment_hash is None:
        assignment_hash = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "algorithm": "virtual_assignment_proxy_from_N_v1",
                    "catalog_proxy_sha256": catalog_hash,
                    "domains": list(cfg.domains),
                    "buckets": [list(hw) for hw in cfg.buckets],
                    "square_only_domains": list(cfg.square_only_domains),
                    "square_bucket_idx": cfg.square_bucket_idx,
                    "N": cfg.N.tolist(),
                }
            )
        )
        assignment_source = "virtual_assignment_proxy_from_N (not an upstream L2 assignment hash)"
    else:
        assignment_hash = cfg.assignment_hash
        assignment_source = "planner_input.assignment_hash"
    return catalog_hash, assignment_hash, {
        "catalog_hash": catalog_source,
        "assignment_hash": assignment_source,
    }


def _plan_identity_sha256(
    cfg: PlannerConfig,
    algorithms: Mapping[str, Any],
    seed_sha256: str,
) -> str:
    catalog_hash, assignment_hash, _sources = _effective_catalog_assignment_hashes(cfg)
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "format": PLAN_FORMAT,
                "algorithms": dict(algorithms),
                "catalog_sha256": catalog_hash,
                "assignment_sha256": assignment_hash,
                "config_sha256": cfg.config_sha256,
                "seed_sha256": seed_sha256,
            }
        )
    )


def _make_header(
    cfg: PlannerConfig,
    D: np.ndarray,
    S: np.ndarray,
    C: np.ndarray,
    solver_record: Mapping[str, Any],
    per_step_repair_moves: int,
    checksums: Mapping[str, str],
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    algorithms = _algorithm_versions(cfg)
    catalog_hash, assignment_hash, hash_sources = _effective_catalog_assignment_hashes(cfg)
    seed_sha256 = _seed_digest("plan_seed_identity", cfg).hex()
    blocks_manifest_sha256 = _sha256_bytes(_canonical_json_bytes(dict(checksums)))
    hashes = {
        "input_sha256": cfg.input_sha256,
        "planner_algo_version": PLANNER_ALGORITHM_VERSION,
        "catalog_sha256": catalog_hash,
        "assignment_sha256": assignment_hash,
        "planner_input_sha256": cfg.input_sha256,
        "config_sha256": cfg.config_sha256,
        "seed_sha256": seed_sha256,
        "blocks_manifest_sha256": blocks_manifest_sha256,
    }
    plan_identity_sha256 = _plan_identity_sha256(cfg, algorithms, seed_sha256)
    hashes["plan_identity_sha256"] = plan_identity_sha256
    domain_totals = {name: int(D[d]) for d, name in enumerate(cfg.domains)}
    return {
        "format_version": PLAN_FORMAT_VERSION,
        "plan_format": PLAN_FORMAT,
        "planner_algo_version": PLANNER_ALGORITHM_VERSION,
        "algorithm_version": PLANNER_ALGORITHM_VERSION,
        "algorithm_versions": algorithms,
        "plan_id": cfg.plan_id,
        "seed": cfg.seed,
        "catalog_hash": catalog_hash,
        "assignment_hash": assignment_hash,
        "upstream_catalog_hash": cfg.catalog_hash,
        "upstream_assignment_hash": cfg.assignment_hash,
        "hash_sources": hash_sources,
        "input_sha256": cfg.input_sha256,
        "config_hash": cfg.config_sha256,
        "plan_hash": plan_identity_sha256,
        "hashes": hashes,
        "world_size": cfg.world_size,
        "per_rank_bs": cfg.per_rank_bs,
        "total_steps": cfg.total_steps,
        "G": cfg.G,
        "global_batch_size": cfg.G,
        "total_exposures": cfg.M,
        "block_steps": cfg.block_steps,
        "num_blocks": int(materialization["num_blocks"]),
        "block_datasets": list(BLOCK_DATASETS),
        "domains": list(cfg.domains),
        "domain_mix": {
            name: _decimal_text(value)
            for name, value in zip(cfg.domains, cfg.domain_mix)
        },
        "square_only_domains": list(cfg.square_only_domains),
        "buckets": [list(hw) for hw in cfg.buckets],
        "square_bucket_idx": cfg.square_bucket_idx,
        "N": cfg.N.tolist(),
        "N_total": _domain_item_totals(cfg),
        "R_max": {name: int(cfg.R_max[d]) for d, name in enumerate(cfg.domains)},
        "D": D.tolist(),
        "domain_totals": domain_totals,
        "S": S.tolist(),
        "bucket_steps": S.tolist(),
        "C": C.tolist(),
        "solver": dict(solver_record),
        "per_step_domain_repair_moves": per_step_repair_moves,
        "materialization": dict(materialization),
        "item_row_mapping": {
            "algorithm": ROW_ID_MAPPING_VERSION,
            "catalog_global_rows": True,
            "virtual_contiguous_range": [0, sum(int(value) for value in cfg.N.flat)],
            "formula": "base[d] + cellbase[d][B] + local_idx",
        },
        "row_scheme": ROW_ID_MAPPING_VERSION,
        # Legacy reader hint; row_scheme above is the normative §9.3 field.
        "row_id_scheme": "global",
        "catalog_size": sum(int(value) for value in cfg.N.flat),
        "cell_row_offset": _cell_row_bases(cfg).tolist(),
        "planner_input": cfg.normalized_dict(),
        "resume_contract": _resume_contract(cfg),
    }


def _existing_output_header(output: Path, cfg: PlannerConfig, start_step: int) -> dict[str, Any] | None:
    if not output.exists():
        return None
    if not output.is_dir():
        raise PlannerInputError(f"--out exists and is not a directory: {output}")
    entries = list(output.iterdir())
    if not entries:
        return None
    if (output / "header.json").is_file():
        try:
            plan = OfflinePlan(output, expected_input=cfg, verify_checksums=True)
            plan.verify_all_checksums()
            if plan.header["materialization"].get("requested_start_step") != start_step:
                raise PlannerInputError(
                    "existing plan start-step mismatch: "
                    f"stored={plan.header['materialization'].get('requested_start_step')}, requested={start_step}"
                )
            return dict(plan.header)
        except (PlannerInputError, OSError, ValueError) as exc:
            raise PlannerInputError(f"existing plan root failed resume/header validation: {exc}") from exc
    raise PlannerInputError(f"refusing to write into non-empty output directory without a valid header: {output}")


def build_plan(
    planner_input: PlannerConfig | Mapping[str, Any] | str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    start_step: int = 0,
) -> dict[str, Any]:
    """Build a complete deterministic plan and return its header."""

    cfg = _as_config(planner_input)
    start_step_i = _require_int(start_step, "start_step", nonnegative=True)
    if start_step_i >= cfg.total_steps:
        raise PlannerInputError(f"start_step must be in [0,{cfg.total_steps}), got {start_step_i}")
    output = Path(out_dir)
    existing = _existing_output_header(output, cfg, start_step_i)
    if existing is not None:
        return existing

    # All plan-time feasibility work happens before the output root is created,
    # so an infeasible contract leaves no partial plan.
    D, S, C, Q, solver_record, _allowed, _static_caps = _solve_quotas(cfg)
    schedule = _bucket_schedule(cfg, S)
    per_bucket_quotas, per_step_repair_moves = _per_step_domain_quotas(cfg, C, S)

    created_root = not output.exists()
    if created_root:
        output.mkdir(parents=True)
    elif any(output.iterdir()):
        raise PlannerInputError(f"refusing to write into non-empty output directory: {output}")
    try:
        checksums, materialization = _materialize_blocks(
            output,
            cfg,
            D,
            S,
            C,
            schedule,
            per_bucket_quotas,
            start_step_i,
        )
        header = _make_header(
            cfg,
            D,
            S,
            C,
            solver_record,
            per_step_repair_moves,
            checksums,
            materialization,
        )
        _write_json_atomic(output / "header.json", header)
        checksums_document = {
            "header": _file_sha256(output / "header.json"),
            "blocks": checksums,
        }
        _write_json_atomic(output / "checksums.json", checksums_document)
        return header
    except BaseException:
        # Only remove files created by this invocation.  An existing empty root
        # is retained; a newly-created root is removed as a unit.
        if created_root:
            shutil.rmtree(output, ignore_errors=True)
        else:
            for name in ("header.json", "checksums.json"):
                try:
                    (output / name).unlink()
                except FileNotFoundError:
                    pass
            shutil.rmtree(output / "blocks", ignore_errors=True)
        raise


def build_plan_from_file(
    input_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    start_step: int = 0,
) -> dict[str, Any]:
    """CLI-facing convenience wrapper for the section 1.5 contract."""

    return build_plan(load_planner_input(input_path), out_dir, start_step=start_step)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PlannerInputError(f"cannot read {path}: {exc}") from exc
    value = _decode_input_bytes(data, str(path))
    return dict(value)


def _header_int_vector(value: Any, length: int, path: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise PlannerInputError(f"{path} must be an integer array of length {length}")
    parsed = [_require_int(item, f"{path}[{i}]", nonnegative=True) for i, item in enumerate(value)]
    if any(item > _INT64_MAX for item in parsed):
        raise PlannerInputError(f"{path} contains a value exceeding int64")
    return np.asarray(parsed, dtype=np.int64)


def _header_int_matrix(value: Any, rows: int, cols: int, path: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != rows:
        raise PlannerInputError(f"{path} must have {rows} rows")
    parsed: list[list[int]] = []
    for r, row in enumerate(value):
        if not isinstance(row, list) or len(row) != cols:
            raise PlannerInputError(f"{path}[{r}] must have {cols} columns")
        parsed.append(
            [_require_int(item, f"{path}[{r}][{c}]", nonnegative=True) for c, item in enumerate(row)]
        )
    if any(item > _INT64_MAX for row in parsed for item in row):
        raise PlannerInputError(f"{path} contains a value exceeding int64")
    return np.asarray(parsed, dtype=np.int64)


def _validate_header_document(header: Mapping[str, Any]) -> tuple[PlannerConfig, np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "format_version",
        "plan_format",
        "planner_algo_version",
        "algorithm_version",
        "algorithm_versions",
        "plan_id",
        "seed",
        "catalog_hash",
        "assignment_hash",
        "upstream_catalog_hash",
        "upstream_assignment_hash",
        "hash_sources",
        "input_sha256",
        "config_hash",
        "plan_hash",
        "hashes",
        "world_size",
        "per_rank_bs",
        "total_steps",
        "G",
        "global_batch_size",
        "total_exposures",
        "block_steps",
        "num_blocks",
        "block_datasets",
        "domains",
        "domain_mix",
        "square_only_domains",
        "buckets",
        "square_bucket_idx",
        "N",
        "N_total",
        "R_max",
        "D",
        "domain_totals",
        "S",
        "bucket_steps",
        "C",
        "solver",
        "per_step_domain_repair_moves",
        "materialization",
        "planner_input",
        "item_row_mapping",
        "row_scheme",
        "row_id_scheme",
        "catalog_size",
        "cell_row_offset",
        "resume_contract",
    }
    missing = sorted(required - set(header))
    if missing:
        raise PlannerInputError(f"header.json is missing required keys: {missing}")
    if _require_int(header["format_version"], "header.format_version") != PLAN_FORMAT_VERSION:
        raise PlannerInputError("unsupported plan header format_version")
    if header["plan_format"] != PLAN_FORMAT:
        raise PlannerInputError(f"unsupported plan_format: {header['plan_format']!r}")
    if header["planner_algo_version"] != PLANNER_ALGORITHM_VERSION:
        raise PlannerInputError(f"planner_algo_version mismatch: {header['planner_algo_version']!r}")
    if header["algorithm_version"] != PLANNER_ALGORITHM_VERSION:
        raise PlannerInputError(f"planner algorithm version mismatch: {header['algorithm_version']!r}")
    planner_input = header["planner_input"]
    if not isinstance(planner_input, Mapping):
        raise PlannerInputError("header.planner_input must be an object")
    cfg = _normalize_input(planner_input)
    if header["config_hash"] != cfg.config_sha256:
        raise PlannerInputError(
            f"header config hash mismatch: stored={header['config_hash']!r}, recomputed={cfg.config_sha256}"
        )
    expected_algorithms = _algorithm_versions(cfg)
    if header["algorithm_versions"] != expected_algorithms:
        raise PlannerInputError("header algorithm_versions do not match this v1 implementation")

    hashes = header["hashes"]
    item_row_mapping = header["item_row_mapping"]
    resume_contract = header["resume_contract"]
    solver = header["solver"]
    materialization = header["materialization"]
    hash_sources = header["hash_sources"]
    for path, value in (
        ("header.hashes", hashes),
        ("header.item_row_mapping", item_row_mapping),
        ("header.resume_contract", resume_contract),
        ("header.solver", solver),
        ("header.materialization", materialization),
        ("header.hash_sources", hash_sources),
    ):
        if not isinstance(value, Mapping):
            raise PlannerInputError(f"{path} must be an object")

    effective_catalog_hash, effective_assignment_hash, expected_hash_sources = _effective_catalog_assignment_hashes(cfg)
    scalar_expectations = {
        "plan_id": cfg.plan_id,
        "seed": cfg.seed,
        "catalog_hash": effective_catalog_hash,
        "assignment_hash": effective_assignment_hash,
        "upstream_catalog_hash": cfg.catalog_hash,
        "upstream_assignment_hash": cfg.assignment_hash,
        "world_size": cfg.world_size,
        "per_rank_bs": cfg.per_rank_bs,
        "total_steps": cfg.total_steps,
        "G": cfg.G,
        "global_batch_size": cfg.G,
        "total_exposures": cfg.M,
        "block_steps": cfg.block_steps,
        "num_blocks": (cfg.total_steps + cfg.block_steps - 1) // cfg.block_steps,
    }
    for key, expected in scalar_expectations.items():
        if header[key] != expected:
            raise PlannerInputError(f"header {key} mismatch: stored={header[key]!r}, expected={expected!r}")
    if not isinstance(header["input_sha256"], str) or len(header["input_sha256"]) != 64 or any(
        ch not in "0123456789abcdef" for ch in header["input_sha256"]
    ):
        raise PlannerInputError("header input_sha256 must be a lowercase SHA-256 hex string")
    if header["input_sha256"] != hashes.get("planner_input_sha256"):
        raise PlannerInputError("header input_sha256 and hashes.planner_input_sha256 disagree")
    if header["input_sha256"] != hashes.get("input_sha256"):
        raise PlannerInputError("header input_sha256 and hashes.input_sha256 disagree")
    if hashes.get("planner_algo_version") != PLANNER_ALGORITHM_VERSION:
        raise PlannerInputError("header hashes.planner_algo_version mismatch")
    if hashes.get("config_sha256") != cfg.config_sha256:
        raise PlannerInputError("header hashes.config_sha256 mismatch")
    if dict(hash_sources) != expected_hash_sources:
        raise PlannerInputError("header hash_sources mismatch")
    if hashes.get("catalog_sha256") != effective_catalog_hash:
        raise PlannerInputError("header catalog hash mismatch")
    if hashes.get("assignment_sha256") != effective_assignment_hash:
        raise PlannerInputError("header assignment hash mismatch")
    seed_sha256 = _seed_digest("plan_seed_identity", cfg).hex()
    if hashes.get("seed_sha256") != seed_sha256:
        raise PlannerInputError("header seed hash mismatch")
    expected_plan_identity = _plan_identity_sha256(cfg, expected_algorithms, seed_sha256)
    if hashes.get("plan_identity_sha256") != expected_plan_identity:
        raise PlannerInputError("header plan identity hash mismatch")
    if header["plan_hash"] != expected_plan_identity:
        raise PlannerInputError("header plan_hash mismatch")
    if item_row_mapping.get("algorithm") != ROW_ID_MAPPING_VERSION:
        raise PlannerInputError("unsupported item row mapping")
    if item_row_mapping.get("catalog_global_rows") is not True:
        raise PlannerInputError("virtual catalog mapping must explicitly declare catalog_global_rows=true")
    expected_row_range = [0, sum(int(value) for value in cfg.N.flat)]
    if item_row_mapping.get("virtual_contiguous_range") != expected_row_range:
        raise PlannerInputError("header virtual item row range mismatch")
    if header["row_scheme"] != ROW_ID_MAPPING_VERSION:
        raise PlannerInputError("header row_scheme must be virtual_catalog_v1")
    if header["row_id_scheme"] != "global":
        raise PlannerInputError("header row_id_scheme compatibility hint must be global")
    if header["catalog_size"] != expected_row_range[1]:
        raise PlannerInputError("header catalog_size mismatch")
    if header["cell_row_offset"] != _cell_row_bases(cfg).tolist():
        raise PlannerInputError("header cell_row_offset mismatch")
    if dict(resume_contract) != _resume_contract(cfg):
        raise PlannerInputError("header resume_contract does not match the frozen W/bs resume semantics")

    redundant_expectations: dict[str, Any] = {
        "domains": list(cfg.domains),
        "domain_mix": {
            name: _decimal_text(value)
            for name, value in zip(cfg.domains, cfg.domain_mix)
        },
        "square_only_domains": list(cfg.square_only_domains),
        "buckets": [list(hw) for hw in cfg.buckets],
        "square_bucket_idx": cfg.square_bucket_idx,
        "N": cfg.N.tolist(),
        "N_total": _domain_item_totals(cfg),
        "R_max": {name: int(cfg.R_max[d]) for d, name in enumerate(cfg.domains)},
        "block_datasets": list(BLOCK_DATASETS),
    }
    for key, expected in redundant_expectations.items():
        if header[key] != expected:
            raise PlannerInputError(f"header {key} disagrees with embedded planner_input")

    D = _header_int_vector(header["D"], cfg.K, "header.D")
    expected_D = _derive_domain_totals(cfg)
    if not np.array_equal(D, expected_D):
        raise PlannerInputError(f"header D mismatch: stored={D.tolist()}, expected={expected_D.tolist()}")
    S = _header_int_vector(header["S"], cfg.B, "header.S")
    C = _header_int_matrix(header["C"], cfg.B, cfg.K, "header.C")
    if header["domain_totals"] != {name: int(D[d]) for d, name in enumerate(cfg.domains)}:
        raise PlannerInputError("header domain_totals disagrees with D")
    if header["bucket_steps"] != S.tolist():
        raise PlannerInputError("header bucket_steps disagrees with S")
    allowed = _allowed_mask(cfg)
    static_caps = _static_caps(cfg, allowed)
    violations = _quota_violations(cfg, D, S, C, allowed, static_caps)
    if violations:
        raise PlannerInputError(f"header quota invariants fail: {violations[0]['message']}")
    repair_moves = _require_int(
        header["per_step_domain_repair_moves"],
        "header.per_step_domain_repair_moves",
        nonnegative=True,
    )
    del repair_moves
    requested_start_step = _require_int(
        materialization.get("requested_start_step"),
        "header.materialization.requested_start_step",
        nonnegative=True,
    )
    if requested_start_step >= cfg.total_steps:
        raise PlannerInputError("header materialization.requested_start_step is outside the plan")
    first_materialized_block = requested_start_step // cfg.block_steps
    num_blocks = (cfg.total_steps + cfg.block_steps - 1) // cfg.block_steps
    materialization_expectations = {
        "num_blocks": num_blocks,
        "materialized_num_blocks": num_blocks - first_materialized_block,
        "first_materialized_block": first_materialized_block,
        "first_materialized_step": first_materialized_block * cfg.block_steps,
        "realized_domain_exposures": D.tolist(),
        "realized_bucket_steps": S.tolist(),
        "global_batches_explicitly_checked_unique": cfg.total_steps,
        "rank_balance_checks": cfg.total_steps,
    }
    for key, expected in materialization_expectations.items():
        if materialization.get(key) != expected:
            raise PlannerInputError(f"header materialization.{key} mismatch")
    for key in ("feistel_cycles_created", "cycle_boundary_swaps"):
        _require_int(materialization.get(key), f"header.materialization.{key}", nonnegative=True)
    solver_algorithm = solver.get("algorithm_version")
    if solver_algorithm not in {QUOTA_MILP_VERSION, QUOTA_FALLBACK_VERSION}:
        raise PlannerInputError("header solver algorithm version is unsupported")
    objective_raw = solver.get("objective")
    if isinstance(objective_raw, bool) or not isinstance(objective_raw, (int, float, Decimal)):
        raise PlannerInputError("header solver.objective must be finite")
    objective_value = float(objective_raw)
    if not math.isfinite(objective_value):
        raise PlannerInputError("header solver.objective must be finite")
    expected_objective = _quota_objective(C, _ideal_joint(cfg, D, allowed), D)
    if not math.isclose(objective_value, expected_objective, rel_tol=1e-10, abs_tol=1e-12):
        raise PlannerInputError(
            f"header solver objective mismatch: stored={objective_value}, recomputed={expected_objective}"
        )
    if solver_algorithm == QUOTA_MILP_VERSION:
        if solver.get("path") != "scipy.optimize.milp" or solver.get("success") is not True or solver.get("status") != 0:
            raise PlannerInputError("header MILP solver status/path is inconsistent")
        if solver.get("objective_common_scale") != cfg.M:
            raise PlannerInputError("header MILP objective scale mismatch")
        scaled_raw = solver.get("objective_scaled_solver")
        if isinstance(scaled_raw, bool) or not isinstance(scaled_raw, (int, float, Decimal)):
            raise PlannerInputError("header MILP scaled objective is missing")
        if not math.isclose(float(scaled_raw) / cfg.M, objective_value, rel_tol=1e-10, abs_tol=1e-12):
            raise PlannerInputError("header MILP scaled/canonical objectives disagree")
    elif solver.get("path") != "deterministic_fallback":
        raise PlannerInputError("header fallback solver path is inconsistent")
    return cfg, D, S, C


class OfflinePlan:
    """Checksum-validating block reader with the frozen resume semantics."""

    def __init__(
        self,
        plan_root: str | os.PathLike[str],
        *,
        expected_input: PlannerConfig | Mapping[str, Any] | str | os.PathLike[str] | None = None,
        world_size: int | None = None,
        per_rank_bs: int | None = None,
        verify_checksums: bool = True,
    ):
        self.plan_root = Path(plan_root)
        if not self.plan_root.is_dir():
            raise PlannerInputError(f"plan root is not a directory: {self.plan_root}")
        self.header = _read_json_object(self.plan_root / "header.json")
        self.config, self.D, self.S, self.C = _validate_header_document(self.header)
        if expected_input is not None:
            expected = _as_config(expected_input)
            if expected.config_sha256 != self.config.config_sha256:
                raise PlannerInputError(
                    "resume header/config mismatch: "
                    f"plan={self.config.config_sha256}, expected={expected.config_sha256}"
                )
        if world_size is not None:
            requested_w = _require_int(world_size, "world_size", positive=True)
            if requested_w != self.config.world_size:
                raise PlannerInputError(
                    f"resume world_size mismatch: plan={self.config.world_size}, requested={requested_w}; "
                    "rebuild with a new plan_id"
                )
        if per_rank_bs is not None:
            requested_bs = _require_int(per_rank_bs, "per_rank_bs", positive=True)
            if requested_bs != self.config.per_rank_bs:
                raise PlannerInputError(
                    f"resume per_rank_bs mismatch: plan={self.config.per_rank_bs}, requested={requested_bs}; "
                    "rebuild the plan"
                )

        checksums_document = _read_json_object(self.plan_root / "checksums.json")
        if set(checksums_document) != {"header", "blocks"}:
            raise PlannerInputError("checksums.json must contain exactly 'header' and 'blocks'")
        header_checksum = checksums_document.get("header")
        if not isinstance(header_checksum, str) or len(header_checksum) != 64 or any(
            ch not in "0123456789abcdef" for ch in header_checksum
        ):
            raise PlannerInputError("checksums.json header must be a lowercase SHA-256 hex string")
        actual_header_checksum = _file_sha256(self.plan_root / "header.json")
        if actual_header_checksum != header_checksum:
            raise PlannerInputError(
                f"header checksum mismatch: expected={header_checksum}, actual={actual_header_checksum}"
            )
        blocks = checksums_document.get("blocks")
        if not isinstance(blocks, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in blocks.items()):
            raise PlannerInputError("checksums.json blocks must map relative paths to SHA-256 strings")
        self.first_materialized_block = int(self.header["materialization"]["first_materialized_block"])
        expected_paths = {
            f"block_{i:06d}.npz"
            for i in range(self.first_materialized_block, int(self.header["num_blocks"]))
        }
        if set(blocks) != expected_paths:
            raise PlannerInputError(
                f"checksums block set mismatch: missing={sorted(expected_paths-set(blocks))}, extra={sorted(set(blocks)-expected_paths)}"
            )
        for relative, checksum in blocks.items():
            if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
                raise PlannerInputError(f"invalid SHA-256 for {relative}: {checksum!r}")
            if not (self.plan_root / "blocks" / relative).is_file():
                raise PlannerInputError(f"missing plan block: blocks/{relative}")
        manifest_hash = _sha256_bytes(_canonical_json_bytes(dict(blocks)))
        if manifest_hash != self.header["hashes"].get("blocks_manifest_sha256"):
            raise PlannerInputError("block checksum manifest hash does not match header")
        self.checksums = dict(blocks)
        self.verify_checksums = bool(verify_checksums)
        self._verified_blocks: set[int] = set()
        self._cached_block_index: int | None = None
        self._cached_bucket: np.ndarray | None = None
        self._cached_items: np.ndarray | None = None
        self._cached_domain_counts: np.ndarray | None = None

    def _block_path(self, block_index: int) -> tuple[str, Path]:
        filename = f"block_{block_index:06d}.npz"
        return filename, self.plan_root / "blocks" / filename

    def _verify_block_checksum(self, block_index: int) -> None:
        if block_index in self._verified_blocks:
            return
        relative, path = self._block_path(block_index)
        actual = _file_sha256(path)
        expected = self.checksums[relative]
        if actual != expected:
            raise PlannerInputError(f"checksum mismatch for {relative}: expected={expected}, actual={actual}")
        self._verified_blocks.add(block_index)

    def verify_all_checksums(self) -> None:
        for block_index in range(self.first_materialized_block, int(self.header["num_blocks"])):
            self._verify_block_checksum(block_index)

    def _load_block(self, block_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._cached_block_index == block_index:
            assert (
                self._cached_bucket is not None
                and self._cached_items is not None
                and self._cached_domain_counts is not None
            )
            return self._cached_bucket, self._cached_items, self._cached_domain_counts
        if block_index < 0 or block_index >= int(self.header["num_blocks"]):
            raise IndexError(f"block index outside plan: {block_index}")
        if block_index < self.first_materialized_block:
            raise IndexError(
                f"block {block_index} precedes this suffix plan's first materialized block "
                f"{self.first_materialized_block}"
            )
        if self.verify_checksums:
            self._verify_block_checksum(block_index)
        _relative, path = self._block_path(block_index)
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(BLOCK_DATASETS):
                    raise PlannerInputError(f"unexpected NPZ members in {path}: {archive.files}")
                bucket = np.array(archive["bucket_of_step"], copy=True)
                items = np.array(archive["item_row"], copy=True)
                domain_counts = np.array(archive["domain_counts"], copy=True)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise PlannerInputError(f"cannot load plan block {path}: {exc}") from exc
        start = block_index * self.config.block_steps
        expected_steps = min(self.config.block_steps, self.config.total_steps - start)
        if bucket.dtype != np.dtype(np.int16) or bucket.shape != (expected_steps,):
            raise PlannerInputError(
                f"bad bucket_of_step in block {block_index}: dtype={bucket.dtype}, shape={bucket.shape}"
            )
        expected_item_shape = (expected_steps, self.config.world_size, self.config.per_rank_bs)
        if items.dtype != np.dtype(np.int64) or items.shape != expected_item_shape:
            raise PlannerInputError(
                f"bad item_row in block {block_index}: dtype={items.dtype}, shape={items.shape}, expected={expected_item_shape}"
            )
        expected_domain_shape = (expected_steps, self.config.K)
        if domain_counts.dtype != np.dtype(np.int32) or domain_counts.shape != expected_domain_shape:
            raise PlannerInputError(
                "bad domain_counts in block "
                f"{block_index}: dtype={domain_counts.dtype}, shape={domain_counts.shape}, "
                f"expected={expected_domain_shape}"
            )
        if np.any(bucket < 0) or np.any(bucket >= self.config.B):
            raise PlannerInputError(f"bucket index outside header table in block {block_index}")
        if np.any(domain_counts < 0):
            raise PlannerInputError(f"negative domain_counts value in block {block_index}")
        if not np.all(domain_counts.sum(axis=1, dtype=np.int64) == self.config.G):
            raise PlannerInputError(f"domain_counts rows do not sum to G in block {block_index}")
        total_rows = sum(int(value) for value in self.config.N.flat)
        if np.any(items < 0) or np.any(items >= total_rows):
            raise PlannerInputError(f"virtual catalog row ID outside [0,{total_rows}) in block {block_index}")
        row_bases = _cell_row_bases(self.config)
        for local_step in range(expected_steps):
            b = int(bucket[local_step])
            counts = domain_counts[local_step].astype(np.int64, copy=False)
            if np.any(counts > self.config.N[b]):
                raise PlannerInputError(
                    f"domain_counts exceeds same-batch cell capacity in block={block_index},step={local_step}"
                )
            flat_items = items[local_step].T.reshape(-1)
            if int(np.unique(flat_items).size) != self.config.G:
                raise PlannerInputError(f"duplicate item_row in block={block_index},step={local_step}")
            boundaries = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
            for d in range(self.config.K):
                lo = int(boundaries[d])
                hi = int(boundaries[d + 1])
                if lo == hi:
                    continue
                row_lo = int(row_bases[b, d])
                row_hi = row_lo + int(self.config.N[b, d])
                values = flat_items[lo:hi]
                if np.any(values < row_lo) or np.any(values >= row_hi):
                    raise PlannerInputError(
                        f"item_row violates virtual cell range in block={block_index},step={local_step},domain={d}"
                    )
        bucket.setflags(write=False)
        items.setflags(write=False)
        domain_counts.setflags(write=False)
        self._cached_block_index = block_index
        self._cached_bucket = bucket
        self._cached_items = items
        self._cached_domain_counts = domain_counts
        return bucket, items, domain_counts

    def batch(self, step: int, rank: int) -> tuple[tuple[int, int], np.ndarray]:
        """Return ``(bucket_hw, row_ids[per_rank_bs])`` for one rank."""

        step_i = _require_int(step, "step", nonnegative=True)
        rank_i = _require_int(rank, "rank", nonnegative=True)
        if step_i >= self.config.total_steps:
            raise IndexError(f"step {step_i} outside [0,{self.config.total_steps})")
        if rank_i >= self.config.world_size:
            raise IndexError(f"rank {rank_i} outside [0,{self.config.world_size})")
        block_index, offset = divmod(step_i, self.config.block_steps)
        bucket, items, _domain_counts = self._load_block(block_index)
        b = int(bucket[offset])
        return self.config.buckets[b], items[offset, rank_i]

    def first_step_after(self, completed_optimizer_step: int) -> int | None:
        completed = _require_int(completed_optimizer_step, "completed_optimizer_step")
        if completed < -1 or completed >= self.config.total_steps:
            raise PlannerInputError(
                f"completed_optimizer_step must be in [-1,{self.config.total_steps - 1}], got {completed}"
            )
        next_step = completed + 1
        return None if next_step == self.config.total_steps else next_step

    def seek_after_completed(self, completed_optimizer_step: int) -> int | None:
        """Discard cached prefetch state and cold-load the next step's block."""

        next_step = self.first_step_after(completed_optimizer_step)
        self._cached_block_index = None
        self._cached_bucket = None
        self._cached_items = None
        self._cached_domain_counts = None
        if next_step is not None:
            self._load_block(next_step // self.config.block_steps)
        return next_step


def validate_resume_header(
    plan_root: str | os.PathLike[str],
    *,
    planner_input: PlannerConfig | Mapping[str, Any] | str | os.PathLike[str] | None = None,
    world_size: int | None = None,
    per_rank_bs: int | None = None,
) -> dict[str, Any]:
    """Validate the immutable plan identity and W/bs resume binding."""

    plan = OfflinePlan(
        plan_root,
        expected_input=planner_input,
        world_size=world_size,
        per_rank_bs=per_rank_bs,
        verify_checksums=True,
    )
    return dict(plan.header)
