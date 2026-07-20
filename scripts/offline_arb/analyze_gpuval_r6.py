#!/usr/bin/env python3
"""Conservative postmortem gate for the two-round OFFLINE_ARB GPU validation.

The gate reads rank-0 per-step metrics, proves the exact A/B resume shape, and
keeps Round A, Round B, and late-window performance separate.  Formal wall
clock is projected from the slowest of those steady-state windows plus the
measured checkpoint overhead amortized at the formal 3,000-step cadence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


MAX_STEPS = 166_667
TARGET_SECONDS_PER_STEP = 2.592
TARGET_DAYS = 5.0
LOG_NAME_RE = re.compile(
    r"^train\.(?P<jobtag>.+)\.r(?P<round>[AB])\.n0\."
    r"(?P<stamp>[0-9]{8}T[0-9]{6}Z)\.log$"
)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    iters = [row["iter_s"] for row in rows]
    waits = [row["data_wait_s"] for row in rows]
    wait_fracs = [row["data_wait_frac"] for row in rows]
    result = {
        "rows": len(rows),
        "step_min": min(row["step"] for row in rows),
        "step_max": max(row["step"] for row in rows),
        "iter_mean_s": statistics.fmean(iters),
        "iter_median_s": statistics.median(iters),
        "iter_p95_s": percentile(iters, 0.95),
        "iter_max_s": max(iters),
        "data_wait_mean_s": statistics.fmean(waits),
        "data_wait_p95_s": percentile(waits, 0.95),
        "data_wait_frac_mean": statistics.fmean(wait_fracs),
        "iter_gt_4s": sum(value > 4.0 for value in iters),
        "iter_gt_5s": sum(value > 5.0 for value in iters),
    }
    result["iter_gt_5s_fraction"] = result["iter_gt_5s"] / result["rows"]
    result["predicted_days_166667_steps_no_checkpoint_adjustment"] = (
        result["iter_mean_s"] * MAX_STEPS / 86_400.0
    )
    return result


def _strict_number(value: Any, key: str, line_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"invalid {key} type at line {line_number}: {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"invalid {key} at line {line_number}: {value!r}")
    return number


def load_metrics(path: Path) -> list[list[dict[str, Any]]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"metrics file missing/not regular: {path}")
    segments: list[list[dict[str, Any]]] = [[]]
    previous = -1
    required = {"step", "iter_s", "data_wait_s", "data_wait_frac"}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not required.issubset(value):
                raise ValueError(f"malformed metrics row {line_number}")
            step = value["step"]
            if type(step) is not int or step <= 0:
                raise ValueError(f"invalid metrics step at line {line_number}: {step!r}")
            iter_s = _strict_number(value["iter_s"], "iter_s", line_number)
            wait_s = _strict_number(value["data_wait_s"], "data_wait_s", line_number)
            wait_frac = _strict_number(
                value["data_wait_frac"], "data_wait_frac", line_number
            )
            if iter_s <= 0:
                raise ValueError(f"non-positive iter_s at line {line_number}")
            if wait_frac > 1.0 or wait_s > iter_s + 0.002:
                raise ValueError(f"impossible data-wait metrics at line {line_number}")
            if abs(wait_frac - wait_s / iter_s) > 0.01:
                raise ValueError(f"inconsistent data-wait fraction at line {line_number}")
            normalized = dict(value)
            normalized.update(
                step=step,
                iter_s=iter_s,
                data_wait_s=wait_s,
                data_wait_frac=wait_frac,
            )
            if step <= previous:
                segments.append([])
            segments[-1].append(normalized)
            previous = step
    segments = [segment for segment in segments if segment]
    if len(segments) != 2:
        raise ValueError(f"expected exactly two monotonic metrics segments, got {len(segments)}")
    return segments


def exact_steps(rows: list[dict[str, Any]], start: int, stop: int, label: str) -> None:
    observed = [row["step"] for row in rows]
    expected = list(range(start, stop + 1))
    if observed != expected:
        missing = sorted(set(expected) - set(observed))[:20]
        extra = sorted(set(observed) - set(expected))[:20]
        raise ValueError(
            f"{label} metrics steps are not exact [{start},{stop}]: "
            f"rows={len(observed)} missing={missing} extra={extra}"
        )


def select(
    rows: Iterable[dict[str, Any]],
    *,
    lo: int,
    hi: int,
    exclude_steps: frozenset[int] = frozenset(),
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if lo <= row["step"] <= hi and row["step"] not in exclude_steps
    ]
    if not selected:
        raise ValueError("steady-state selection is empty")
    return selected


def require_log(path: Path, patterns: tuple[str, ...], label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} log missing/not regular: {path}")
    raw = path.read_text(encoding="utf-8")
    for pattern in patterns:
        if pattern not in raw:
            raise ValueError(f"{label} evidence missing: {pattern!r}")
    return raw


def log_identity(path: Path, expected_round: str) -> tuple[str, str]:
    match = LOG_NAME_RE.fullmatch(path.name)
    if match is None or match.group("round") != expected_round:
        raise ValueError(f"malformed Round {expected_round} log name: {path.name}")
    return match.group("jobtag"), match.group("stamp")


def checkpoint_observation(
    rows: list[dict[str, Any]], checkpoint_step: int, label: str
) -> dict[str, Any]:
    by_step = {row["step"]: row for row in rows}
    baseline_steps = list(range(checkpoint_step - 20, checkpoint_step)) + list(
        range(checkpoint_step + 11, checkpoint_step + 31)
    )
    event_steps = list(range(checkpoint_step + 1, checkpoint_step + 11))
    if any(step not in by_step for step in baseline_steps + event_steps):
        raise ValueError(f"{label} checkpoint overhead neighbor window incomplete")
    baseline = statistics.median(by_step[step]["iter_s"] for step in baseline_steps)
    event_iters = [by_step[step]["iter_s"] for step in event_steps]
    event_excess = sum(max(0.0, value - baseline) for value in event_iters)
    return {
        "label": label,
        "checkpoint_saved_after_step": checkpoint_step,
        "baseline_steps": baseline_steps,
        "event_steps": event_steps,
        "baseline_median_iter_s": baseline,
        "event_iter_s": event_iters,
        "event_excess_s": event_excess,
    }


def classify(
    combined: dict[str, Any],
    round_a: dict[str, Any],
    round_b: dict[str, Any],
    early_a: dict[str, Any],
    early_b: dict[str, Any],
    late_a: dict[str, Any],
    late_b: dict[str, Any],
    phases: dict[str, dict[str, Any]],
    formal: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any]]:
    round_stats = (round_a, round_b)
    early_stats = (early_a, early_b)
    late_stats = (late_a, late_b)
    all_windows = round_stats + late_stats
    phase_means = [value["iter_mean_s"] for value in phases.values()]
    phase_spread = max(phase_means) - min(phase_means)
    round_mean_spread = abs(round_a["iter_mean_s"] - round_b["iter_mean_s"])
    late_drift_a = max(0.0, late_a["iter_mean_s"] - early_a["iter_mean_s"])
    late_drift_b = max(0.0, late_b["iter_mean_s"] - early_b["iter_mean_s"])
    max_late_drift = max(late_drift_a, late_drift_b)
    max_round_mean = max(item["iter_mean_s"] for item in round_stats)
    max_late_mean = max(item["iter_mean_s"] for item in late_stats)
    max_p95 = max(item["iter_p95_s"] for item in all_windows)
    max_median = max(item["iter_median_s"] for item in all_windows)
    max_wait_frac = max(item["data_wait_frac_mean"] for item in all_windows)
    max_spike_fraction = max(item["iter_gt_5s_fraction"] for item in all_windows)

    observed = {
        "formal_predicted_days": formal["predicted_wall_clock_days"],
        "formal_conservative_steady_s": formal["conservative_steady_iter_s"],
        "max_round_mean_s": max_round_mean,
        "max_late_mean_s": max_late_mean,
        "max_early_mean_s": max(item["iter_mean_s"] for item in early_stats),
        "max_window_median_s": max_median,
        "max_window_p95_s": max_p95,
        "round_mean_spread_s": round_mean_spread,
        "max_late_drift_s": max_late_drift,
        "step_mod_4_mean_spread_s": phase_spread,
        "max_data_wait_frac_mean": max_wait_frac,
        "max_gt5s_fraction": max_spike_fraction,
        "combined_mean_s": combined["iter_mean_s"],
        "checkpoint_overhead_fraction": formal["checkpoint_overhead_fraction"],
    }

    hard_fail = {
        "formal_predicted_gt_5.8d": observed["formal_predicted_days"] > 5.8,
        "round_mean_gt_3.0s": max_round_mean > 3.0,
        "late_mean_gt_3.0s": max_late_mean > 3.0,
        "window_p95_gt_5.0s": max_p95 > 5.0,
        "round_mean_spread_gt_0.40s": round_mean_spread > 0.40,
        "late_drift_gt_0.35s": max_late_drift > 0.35,
        "phase_spread_gt_1.0s": phase_spread > 1.0,
        "data_wait_frac_gt_0.30": max_wait_frac > 0.30,
        "gt5s_fraction_gt_0.05": max_spike_fraction > 0.05,
        "checkpoint_overhead_fraction_gt_0.05": formal["checkpoint_overhead_fraction"] > 0.05,
    }
    failures = [name for name, failed in hard_fail.items() if failed]
    if failures:
        return "NO_GO", failures, {"observed": observed, "hard_fail": hard_fail}

    strict = {
        "formal_predicted_le_5.31d": observed["formal_predicted_days"] <= 5.31,
        "each_round_mean_le_2.75s": max_round_mean <= 2.75,
        "each_late_mean_le_2.75s": max_late_mean <= 2.75,
        "each_window_median_le_2.70s": max_median <= 2.70,
        "each_window_p95_le_3.50s": max_p95 <= 3.50,
        "round_mean_spread_le_0.15s": round_mean_spread <= 0.15,
        "late_vs_early_drift_le_0.10s": max_late_drift <= 0.10,
        "phase_spread_le_0.45s": phase_spread <= 0.45,
        "data_wait_frac_le_0.18": max_wait_frac <= 0.18,
        "gt5s_fraction_le_0.01": max_spike_fraction <= 0.01,
        "checkpoint_overhead_fraction_le_0.02": formal["checkpoint_overhead_fraction"] <= 0.02,
    }
    if all(strict.values()):
        return (
            "PASS",
            ["all_round_specific_and_formal_wall_clock_gates_pass"],
            {"observed": observed, "hard_fail": hard_fail, "strict": strict},
        )

    borderline = {
        "formal_predicted_le_5.50d": observed["formal_predicted_days"] <= 5.50,
        "each_round_mean_le_2.85s": max_round_mean <= 2.85,
        "each_late_mean_le_2.85s": max_late_mean <= 2.85,
        "each_window_median_le_2.80s": max_median <= 2.80,
        "each_window_p95_le_4.00s": max_p95 <= 4.00,
        "round_mean_spread_le_0.25s": round_mean_spread <= 0.25,
        "late_vs_early_drift_le_0.20s": max_late_drift <= 0.20,
        "phase_spread_le_0.65s": phase_spread <= 0.65,
        "data_wait_frac_le_0.22": max_wait_frac <= 0.22,
        "gt5s_fraction_le_0.02": max_spike_fraction <= 0.02,
        "checkpoint_overhead_fraction_le_0.05": formal["checkpoint_overhead_fraction"] <= 0.05,
    }
    if all(borderline.values()):
        return (
            "BORDERLINE",
            ["strict_gate_failed; formal_launch_is_not_authorized"],
            {
                "observed": observed,
                "hard_fail": hard_fail,
                "strict": strict,
                "borderline": borderline,
            },
        )
    return (
        "REVIEW",
        ["outside_strict_and_borderline_gates_but_not_hard_no_go"],
        {
            "observed": observed,
            "hard_fail": hard_fail,
            "strict": strict,
            "borderline": borderline,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--round-a-log", type=Path, required=True)
    parser.add_argument("--round-b-log", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpuval-yaml", type=Path, required=True)
    parser.add_argument("--formal-yaml", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--formal-world-size", type=int, required=True)
    parser.add_argument("--formal-checkpoint-steps", type=int, required=True)
    parser.add_argument("--formal-final-save", choices=("true", "false"), required=True)
    parser.add_argument("--setup-budget-s", type=float, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.is_file() or args.config.is_symlink():
        raise ValueError(f"config missing/not regular: {args.config}")
    for label, path in (("gpuval yaml", args.gpuval_yaml), ("formal yaml", args.formal_yaml)):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} missing/not regular: {path}")
    if args.formal_checkpoint_steps <= 0 or args.setup_budget_s < 0:
        raise ValueError("formal checkpoint cadence/setup budget must be positive/non-negative")
    if args.formal_world_size <= 0:
        raise ValueError("formal world size must be positive")
    from omegaconf import OmegaConf

    config = OmegaConf.load(args.config)
    gpuval_yaml = OmegaConf.load(args.gpuval_yaml)
    formal_yaml = OmegaConf.load(args.formal_yaml)
    yaml_world_size = int(formal_yaml.numWorkers) * int(formal_yaml.gpuPerWorker)
    if yaml_world_size != args.formal_world_size:
        raise ValueError(
            f"formal yaml world size mismatch: {yaml_world_size} != {args.formal_world_size}"
        )
    if int(gpuval_yaml.numWorkers) != 2 or int(gpuval_yaml.gpuPerWorker) != 8:
        raise ValueError("gpuval yaml is not the expected 2x8 validation topology")
    configured_checkpoint_steps = int(config.trainer.get("checkpoint_steps", 0))
    configured_final_save = bool(config.trainer.get("save_final", True))
    configured_batch_size = int(config.trainer.get("batch_size", 0))
    configured_exposure = int(config.trainer.get("max_exposure_images", 0))
    derived_max_steps = math.ceil(
        configured_exposure / (configured_batch_size * args.formal_world_size)
    )
    if (
        configured_batch_size != 36
        or configured_exposure != 384_000_000
        or derived_max_steps != MAX_STEPS
    ):
        raise ValueError(
            "formal exposure geometry mismatch: "
            f"batch={configured_batch_size} exposure={configured_exposure} "
            f"world={args.formal_world_size} derived_steps={derived_max_steps}"
        )
    requested_final_save = args.formal_final_save == "true"
    if configured_checkpoint_steps != args.formal_checkpoint_steps:
        raise ValueError(
            "formal checkpoint cadence argument/config mismatch: "
            f"{args.formal_checkpoint_steps} != {configured_checkpoint_steps}"
        )
    if configured_final_save != requested_final_save:
        raise ValueError(
            "formal final-save argument/config mismatch: "
            f"{requested_final_save} != {configured_final_save}"
        )
    if args.metrics.parent.parent.name != args.run_id:
        raise ValueError(
            f"metrics run-id mismatch: {args.metrics.parent.parent.name} != {args.run_id}"
        )
    jobtag_a, stamp_a = log_identity(args.round_a_log, "A")
    jobtag_b, stamp_b = log_identity(args.round_b_log, "B")
    if jobtag_a != jobtag_b or stamp_a != stamp_b:
        raise ValueError("Round A/B log attempt identity mismatch")

    expected_checkpoint = args.metrics.parent / "last.pt"
    require_log(
        args.round_a_log,
        (
            f"[train] checkpoint saved at step 200 -> {expected_checkpoint}",
            "[train] step 350/350 ",
        ),
        "Round A",
    )
    expected_resume = f"[train] resumed from {expected_checkpoint} at step 200"
    require_log(
        args.round_b_log,
        (
            expected_resume,
            f"[train] checkpoint saved at step 400 -> {expected_checkpoint}",
            "[train] step 500/500 ",
        ),
        "Round B",
    )
    round_a_rows, round_b_rows = load_metrics(args.metrics)
    exact_steps(round_a_rows, 1, 350, "Round A")
    exact_steps(round_b_rows, 201, 500, "Round B")

    # A save after c is charged mainly to c+1 and can leave a short tail.  The
    # fixed c+1..c+10 windows are measured separately and excluded from steady
    # state; their excess is conservatively restored at FORMAL cadence below.
    selected_a = select(
        round_a_rows, lo=40, hi=350, exclude_steps=frozenset(range(201, 211))
    )
    selected_b = select(
        round_b_rows, lo=230, hi=500, exclude_steps=frozenset(range(401, 411))
    )
    early_a_rows = select(selected_a, lo=91, hi=190)
    early_b_rows = select(selected_b, lo=301, hi=390)
    late_a_rows = select(selected_a, lo=251, hi=350)
    late_b_rows = select(selected_b, lo=411, hi=500)
    combined_rows = selected_a + selected_b

    combined = summary(combined_rows)
    round_a = summary(selected_a)
    round_b = summary(selected_b)
    early_a = summary(early_a_rows)
    early_b = summary(early_b_rows)
    late_a = summary(late_a_rows)
    late_b = summary(late_b_rows)
    phases: dict[str, dict[str, Any]] = {}
    for phase in range(4):
        phases[str(phase)] = summary(
            [row for row in combined_rows if row["step"] % 4 == phase]
        )

    checkpoints = [
        checkpoint_observation(round_a_rows, 200, "Round A save@200"),
        checkpoint_observation(round_b_rows, 400, "Round B save@400"),
    ]
    checkpoint_cost_safety_factor = 1.25
    checkpoint_cost = checkpoint_cost_safety_factor * max(
        item["event_excess_s"] for item in checkpoints
    )
    conservative_steady = max(
        round_a["iter_mean_s"],
        round_b["iter_mean_s"],
        late_a["iter_mean_s"],
        late_b["iter_mean_s"],
    )
    periodic_checkpoint_events = (MAX_STEPS - 1) // args.formal_checkpoint_steps
    final_checkpoint_events = 1 if requested_final_save else 0
    total_checkpoint_events = periodic_checkpoint_events + final_checkpoint_events
    steady_seconds = conservative_steady * MAX_STEPS
    checkpoint_seconds = checkpoint_cost * total_checkpoint_events
    predicted_seconds = steady_seconds + checkpoint_seconds + args.setup_budget_s
    formal = {
        "max_steps": MAX_STEPS,
        "per_rank_batch": configured_batch_size,
        "formal_world_size": args.formal_world_size,
        "global_batch": configured_batch_size * args.formal_world_size,
        "target_samples": configured_exposure,
        "target_seconds_per_step": TARGET_SECONDS_PER_STEP,
        "target_days": TARGET_DAYS,
        "formal_checkpoint_steps": args.formal_checkpoint_steps,
        "periodic_checkpoint_events": periodic_checkpoint_events,
        "final_checkpoint_events": final_checkpoint_events,
        "checkpoint_event_cost_safety_factor": checkpoint_cost_safety_factor,
        "conservative_checkpoint_cost_per_event_s": checkpoint_cost,
        "checkpoint_overhead_seconds": checkpoint_seconds,
        "checkpoint_overhead_fraction": checkpoint_seconds / steady_seconds,
        "setup_budget_seconds": args.setup_budget_s,
        "projection_scope": "training_process_initialization_through_final_checkpoint; JCC_queue_and_freeze_preflight_excluded",
        "conservative_steady_iter_s": conservative_steady,
        "raw_steady_days": steady_seconds / 86_400.0,
        "predicted_wall_clock_seconds": predicted_seconds,
        "predicted_wall_clock_days": predicted_seconds / 86_400.0,
        "projection_basis": "max(round_A,round_B,late_A,late_B)_mean_plus_1.25x_worst_checkpoint_event_plus_setup_budget",
    }

    status, reasons, gate = classify(
        combined, round_a, round_b, early_a, early_b, late_a, late_b, phases, formal
    )
    result = {
        "schema_version": "offline_arb_gpuval_r6_postmortem_v2",
        "status": status,
        "reasons": reasons,
        "identity": {
            "run_id": args.run_id,
            "jobtag": jobtag_a,
            "stamp": stamp_a,
            "config": str(args.config),
            "config_sha256": sha256_file(args.config),
            "gpuval_yaml": str(args.gpuval_yaml),
            "gpuval_yaml_sha256": sha256_file(args.gpuval_yaml),
            "formal_yaml": str(args.formal_yaml),
            "formal_yaml_sha256": sha256_file(args.formal_yaml),
            "metrics": str(args.metrics),
            "metrics_sha256": sha256_file(args.metrics),
            "round_a_log": str(args.round_a_log),
            "round_a_log_sha256": sha256_file(args.round_a_log),
            "round_b_log": str(args.round_b_log),
            "round_b_log_sha256": sha256_file(args.round_b_log),
        },
        "selection": {
            "round_a": "steps 40..350 excluding checkpoint event tail 201..210",
            "round_b": "steps 230..500 excluding checkpoint event tail 401..410",
            "early_a": "steps 91..190",
            "early_b": "steps 301..390",
            "late_a": "steps 251..350",
            "late_b": "steps 411..500",
        },
        "round_a": round_a,
        "round_b": round_b,
        "early_round_a": early_a,
        "early_round_b": early_b,
        "late_round_a": late_a,
        "late_round_b": late_b,
        "combined": combined,
        "step_mod_4": phases,
        "checkpoint_observations": checkpoints,
        "formal_projection": formal,
        "gate": gate,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if status == "PASS":
        return 0
    return {"BORDERLINE": 4, "REVIEW": 2, "NO_GO": 3}[status]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL gpuval r6 postmortem: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
