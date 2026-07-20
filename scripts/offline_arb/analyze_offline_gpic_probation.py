#!/usr/bin/env python3
"""Strict postflight for the 8x8, two-round GPIC blob probation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "offline_gpic_blob_probation_analysis_v1"
SUPERVISOR_SCHEMA = "offline_gpic_blob_probation_supervisor_v1"
SHA_RE = re.compile(r"[0-9a-f]{64}")
MAX_FORMAL_STEPS = 166_667
TARGET_MAX_S = 2.55
HARD_MAX_S = 2.75


class AnalysisError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    raw = canonical(value)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise AnalysisError(f"refusing analysis overwrite: {path}")
    with temporary.open("xb", buffering=0) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise AnalysisError(f"not a nonempty regular file: {path}")
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        if identity != _identity(os.fstat(handle.fileno())):
            raise AnalysisError(f"file changed while opening: {path}")
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
        closed = os.fstat(handle.fileno())
    if _identity(closed) != identity or _identity(path.lstat()) != identity:
        raise AnalysisError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"JSON is not an object: {path}")
    return value


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise AnalysisError(f"{label} is invalid: {value!r}")
    return result


def metric_segments(path: Path) -> tuple[list[list[dict[str, Any]]], str]:
    digest = hashlib.sha256()
    segments: list[list[dict[str, Any]]] = [[]]
    previous = -1
    with path.open("rb", buffering=0) as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.endswith(b"\n"):
                raise AnalysisError(f"unterminated metrics record line {line_number}")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"malformed metrics line {line_number}: {exc}") from exc
            if not isinstance(value, dict) or type(value.get("step")) is not int:
                raise AnalysisError(f"malformed metrics record line {line_number}")
            step = int(value["step"])
            iter_s = number(value.get("iter_s"), f"iter_s line {line_number}")
            wait_s = number(value.get("data_wait_s"), f"data_wait_s line {line_number}")
            wait_frac = number(
                value.get("data_wait_frac"), f"data_wait_frac line {line_number}"
            )
            if step <= 0 or iter_s <= 0 or wait_s > iter_s + 0.01 or wait_frac > 1.0:
                raise AnalysisError(f"impossible metrics values line {line_number}")
            if abs(wait_frac - wait_s / iter_s) > 0.02:
                raise AnalysisError(f"inconsistent data wait fraction line {line_number}")
            value = dict(value)
            value.update(iter_s=iter_s, data_wait_s=wait_s, data_wait_frac=wait_frac)
            if step <= previous:
                segments.append([])
            segments[-1].append(value)
            previous = step
    complete = [segment for segment in segments if segment]
    if len(complete) != 2:
        raise AnalysisError(f"expected two monotonic metrics segments, got {len(complete)}")
    return complete, digest.hexdigest()


def exact_steps(rows: list[dict[str, Any]], start: int, stop: int, label: str) -> None:
    observed = [int(row["step"]) for row in rows]
    expected = list(range(start, stop + 1))
    if observed != expected:
        raise AnalysisError(
            f"{label} step sequence differs: rows={len(observed)} "
            f"missing={sorted(set(expected)-set(observed))[:16]} "
            f"extra={sorted(set(observed)-set(expected))[:16]}"
        )


def select(rows: Iterable[dict[str, Any]], lo: int, hi: int) -> list[dict[str, Any]]:
    result = [row for row in rows if lo <= int(row["step"]) <= hi]
    if len(result) != hi - lo + 1:
        raise AnalysisError(f"incomplete steady window [{lo},{hi}]")
    return result


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    iters = [float(row["iter_s"]) for row in rows]
    waits = [float(row["data_wait_s"]) for row in rows]
    wait_fracs = [float(row["data_wait_frac"]) for row in rows]
    return {
        "label": label,
        "rows": len(rows),
        "step_start": int(rows[0]["step"]),
        "step_stop": int(rows[-1]["step"]),
        "iter_mean_s": statistics.fmean(iters),
        "iter_median_s": statistics.median(iters),
        "iter_p95_s": percentile(iters, 0.95),
        "iter_max_s": max(iters),
        "data_wait_mean_s": statistics.fmean(waits),
        "data_wait_p95_s": percentile(waits, 0.95),
        "data_wait_frac_mean": statistics.fmean(wait_fracs),
        "iter_gt_5s_fraction": sum(value > 5.0 for value in iters) / len(iters),
    }


def validate_health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    health = [row for row in rows if "gpic_allrank_min_blob_fd_cache_size" in row]
    if len(health) < 10:
        raise AnalysisError("too few all-rank GPIC health snapshots")
    zero_keys = (
        "gpic_allrank_sum_short_read_failures",
        "gpic_allrank_sum_integrity_failures",
        "gpic_allrank_sum_pread_errors",
        "gpic_allrank_sum_source_fallback_items",
        "gpic_allrank_health_snapshot_errors",
    )
    for row in health:
        step = int(row["step"])
        for key in zero_keys:
            if number(row.get(key), f"{key} step {step}") != 0:
                raise AnalysisError(f"nonzero {key} at step {step}")
        if number(row.get("gpic_allrank_min_blob_fd_cache_size"), "min fd cache") != 1024:
            raise AnalysisError(f"min fd cache is not 1024 at step {step}")
        if number(row.get("gpic_allrank_max_blob_fd_cache_size"), "max fd cache") != 1024:
            raise AnalysisError(f"max fd cache is not 1024 at step {step}")
        if number(
            row.get("gpic_allrank_min_fd_cache_covers_blob_count"), "fd cache coverage"
        ) != 1:
            raise AnalysisError(f"fd cache does not cover blob set at step {step}")
        requested = number(row.get("gpic_allrank_sum_pread_requested_bytes"), "requested")
        returned = number(row.get("gpic_allrank_sum_pread_returned_bytes"), "returned")
        if returned > requested:
            raise AnalysisError(f"pread returned bytes exceed requested at step {step}")
    final = health[-1]
    if number(final.get("gpic_allrank_slowest_supply_img_s"), "slowest supply") <= 0:
        raise AnalysisError("final all-rank slowest GPIC supply is non-positive")
    if number(final.get("gpic_allrank_sum_read_items"), "read items") <= 0:
        raise AnalysisError("final all-rank GPIC read count is non-positive")
    return {
        "snapshots": len(health),
        "first_step": int(health[0]["step"]),
        "last_step": int(final["step"]),
        "final_slowest_supply_img_s": final["gpic_allrank_slowest_supply_img_s"],
        "max_rss_bytes": max(row["gpic_allrank_max_rss_bytes"] for row in health),
        "max_process_fd_count": max(
            row["gpic_allrank_max_process_fd_count"] for row in health
        ),
        "max_blob_fd_peak": max(row["gpic_allrank_max_blob_fd_peak"] for row in health),
        "min_rlimit_nofile_soft": min(
            row["gpic_allrank_min_rlimit_nofile_soft"] for row in health
        ),
        "manifest_blob_count": max(
            row["gpic_allrank_max_manifest_blob_count"] for row in health
        ),
        "final_read_items": final["gpic_allrank_sum_read_items"],
        "final_pread_bytes": final["gpic_allrank_sum_pread_returned_bytes"],
        "max_inflight_pread_bytes": max(
            row["gpic_allrank_sum_pread_requested_bytes"]
            - row["gpic_allrank_sum_pread_returned_bytes"]
            for row in health
        ),
        "final_inflight_pread_bytes": (
            final["gpic_allrank_sum_pread_requested_bytes"]
            - final["gpic_allrank_sum_pread_returned_bytes"]
        ),
        "zero_short_integrity_pread_fallback": True,
        "fd_cache_size": 1024,
        "fd_cache_covers_blob_count": True,
    }


def require_log(path: Path, patterns: list[str]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AnalysisError(f"missing/non-regular train log: {path}")
    text = path.read_text(encoding="utf-8")
    for pattern in patterns:
        if pattern not in text:
            raise AnalysisError(f"log {path.name} lacks exact evidence {pattern!r}")
    return {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}


def validate_artifact(args: argparse.Namespace) -> dict[str, Any]:
    root = args.artifact_root
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.name != args.buildtag
        or args.buildtag in {"", ".", "..", "latest"}
        or SHA_RE.fullmatch(args.manifest_sha256) is None
    ):
        raise AnalysisError("artifact/buildtag/manifest submission binding is malformed")
    manifest = root / "manifest.json"
    if sha256_file(manifest) != args.manifest_sha256:
        raise AnalysisError("artifact manifest SHA changed during probation")
    if (root / "_SUCCESS").read_bytes() != (args.manifest_sha256 + "\n").encode("ascii"):
        raise AnalysisError("artifact _SUCCESS does not bind manifest after probation")
    value = json_object(manifest)
    storage = value.get("release", {}).get("storage", {})
    if (
        value.get("schema_version") != "offline_gpic_blob_v1"
        or value.get("status") != "PASS"
        or Path(str(storage.get("s3_root", ""))).name != args.buildtag
    ):
        raise AnalysisError("artifact manifest release/buildtag drift")
    return value


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    if args.result_root.name != args.jobtag:
        raise AnalysisError("result namespace is not bound to the exact JCC jobtag")
    if (
        args.checkpoint_root.name != args.jobtag
        or args.checkpoint_root.parent.name != args.buildtag
    ):
        raise AnalysisError("checkpoint namespace is not independently bound to buildtag/jobtag")
    manifest = validate_artifact(args)
    stop = args.result_root / f"early_stop.{args.jobtag}.json"
    if stop.exists() or stop.is_symlink():
        raise AnalysisError(f"probation early-stop marker exists: {stop}")
    compact_preflights: list[dict[str, Any]] = []
    compact_bindings: dict[str, Any] | None = None
    sidecar_controls: list[dict[str, Any]] = []
    sidecar_identity: dict[str, Any] | None = None
    sidecar_keys = (
        "catalog_hash",
        "catalog_header_sha256",
        "file_count",
        "header_file_sha256",
        "header_sha256",
        "release_digest",
        "root",
        "snapshot_fingerprints_sha256",
        "success_file_sha256",
        "total_bytes",
    )
    for node in range(8):
        preflight_path = args.result_root / f"preflight.{args.jobtag}.n{node}.json"
        preflight = json_object(preflight_path)
        if (
            preflight.get("schema_version") != "offline_gpic_blob_compact_preflight_v1"
            or preflight.get("status") != "PASS"
            or preflight.get("jobtag") != args.jobtag
            or preflight.get("node_rank") != node
            or preflight.get("buildtag") != args.buildtag
            or preflight.get("manifest_sha256") != args.manifest_sha256
            or preflight.get("backend") != "mountpoint_s3"
            or preflight.get("fd_cache_size") != 1024
            or preflight.get("full_publication_verified_once") is not True
        ):
            raise AnalysisError(f"compact preflight identity mismatch node={node}")
        if compact_bindings is None:
            compact_bindings = preflight.get("bindings")
        elif preflight.get("bindings") != compact_bindings:
            raise AnalysisError(f"compact preflight binding drift node={node}")
        compact_preflights.append(
            {"node_rank": node, "sha256": sha256_file(preflight_path)}
        )

        sidecar_path = args.result_root / f"sidecarcontrol.{args.jobtag}.n{node}.json"
        sidecar = json_object(sidecar_path)
        if sidecar.get("status") != "PASS" or sidecar.get("mode") != "verify-sidecar-control":
            raise AnalysisError(f"sidecar compact control failed node={node}")
        identity = {key: sidecar.get(key) for key in sidecar_keys}
        if sidecar_identity is None:
            sidecar_identity = identity
        elif identity != sidecar_identity:
            raise AnalysisError(f"sidecar compact control drift node={node}")
        sidecar_controls.append(
            {"node_rank": node, "sha256": sha256_file(sidecar_path)}
        )

    supervisors: list[dict[str, Any]] = []
    for round_name in ("A", "B"):
        for node in range(8):
            path = args.result_root / f"supervisor.{args.jobtag}.r{round_name}.n{node}.json"
            value = json_object(path)
            if (
                value.get("schema_version") != SUPERVISOR_SCHEMA
                or value.get("status") != "PASS"
                or value.get("jobtag") != args.jobtag
                or value.get("buildtag") != args.buildtag
                or value.get("manifest_sha256") != args.manifest_sha256
                or value.get("round") != round_name
                or value.get("node_rank") != node
                or value.get("train_returncode") != 0
                or value.get("stopped") is not None
            ):
                raise AnalysisError(f"supervisor result identity/final status mismatch {path}")
            supervisors.append(
                {
                    "round": round_name,
                    "node_rank": node,
                    "sha256": sha256_file(path),
                    "wall_s": value["wall_s"],
                }
            )

    segments, metrics_sha = metric_segments(args.checkpoint_root / "metrics.jsonl")
    round_a, round_b = segments
    exact_steps(round_a, 1, 250, "Round A")
    exact_steps(round_b, 201, 500, "Round B exact resume")
    windows = {
        "round_a_precheckpoint": summarize("A:61-190", select(round_a, 61, 190)),
        "round_a_late": summarize("A:216-250", select(round_a, 216, 250)),
        "round_b_postresume": summarize("B:261-390", select(round_b, 261, 390)),
        "round_b_late": summarize("B:416-500", select(round_b, 416, 500)),
    }
    conservative = max(value["iter_mean_s"] for value in windows.values())
    max_p95 = max(value["iter_p95_s"] for value in windows.values())
    max_wait_frac = max(value["data_wait_frac_mean"] for value in windows.values())
    max_spike_fraction = max(value["iter_gt_5s_fraction"] for value in windows.values())
    health = validate_health(round_a + round_b)

    backend_marker = (
        "[train] dataset gpic_payload_backend=offline_gpic_blob_v1 "
        f"root={args.artifact_root} manifest_sha256={args.manifest_sha256}"
    )
    log_a = args.result_root / f"train.{args.jobtag}.rA.n0.log"
    log_b = args.result_root / f"train.{args.jobtag}.rB.n0.log"
    logs = {
        "round_a": require_log(
            log_a,
            [backend_marker, "fd_cache_size=1024", "[train] step 250/250 ", "[train] done at step 250"],
        ),
        "round_b": require_log(
            log_b,
            [
                backend_marker,
                "fd_cache_size=1024",
                f"[train] resumed from {args.checkpoint_root}/last.pt at step 200",
                "[train] step 500/500 ",
                "[train] done at step 500",
            ],
        ),
    }
    checkpoint = args.checkpoint_root / "last.pt"
    metadata = checkpoint.lstat()
    if checkpoint.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise AnalysisError("probation checkpoint is missing/non-regular/empty")

    hard_failures = {
        "conservative_steady_mean_gt_2.75s": conservative > HARD_MAX_S,
        "any_steady_p95_gt_5s": max_p95 > 5.0,
        "any_steady_data_wait_fraction_gt_0.25": max_wait_frac > 0.25,
        "any_steady_gt5s_fraction_gt_0.05": max_spike_fraction > 0.05,
    }
    failures = [key for key, failed in hard_failures.items() if failed]
    classification = (
        "TARGET_2.50_TO_2.55"
        if conservative <= TARGET_MAX_S
        else ("ACCEPTABLE_VICINITY_LE_2.75" if not failures else "FAIL")
    )
    value = {
        "schema_version": SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "classification": classification,
        "jobtag": args.jobtag,
        "buildtag": args.buildtag,
        "artifact_root": str(args.artifact_root.resolve(strict=True)),
        "manifest_sha256": args.manifest_sha256,
        "release_sha256": manifest.get("release_sha256"),
        "world_size": 64,
        "global_batch": 2304,
        "round_contract": {
            "round_a_steps": [1, 250],
            "checkpoint_step": 200,
            "round_b_resume_step": 200,
            "round_b_steps": [201, 500],
            "exact_resume": True,
            "independent_checkpoint_namespace": str(args.checkpoint_root.resolve(strict=True)),
        },
        "speed_gate": {
            "target_seconds_per_step": [2.50, 2.55],
            "hard_max_seconds_per_step": HARD_MAX_S,
            "conservative_steady_mean_s": conservative,
            "predicted_formal_days": conservative * MAX_FORMAL_STEPS / 86400.0,
            "max_steady_p95_s": max_p95,
            "max_steady_data_wait_fraction": max_wait_frac,
            "max_gt5s_fraction": max_spike_fraction,
            "windows": windows,
            "hard_failures": hard_failures,
        },
        "overlay_health": health,
        "compact_preflights": compact_preflights,
        "compact_preflight_bindings": compact_bindings,
        "sidecar_controls": sidecar_controls,
        "sidecar_identity": sidecar_identity,
        "supervisors": supervisors,
        "logs": logs,
        "checkpoint": {
            "path": str(checkpoint.resolve(strict=True)),
            "bytes": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "content_hash_skipped": "compact probation postflight; exact resume is log+metrics bound",
        },
        "bindings": {
            "metrics_sha256": metrics_sha,
            "base_config_sha256": sha256_file(args.config),
            "plan_header_sha256": sha256_file(args.plan_root / "header.json"),
            "catalog_header_sha256": sha256_file(args.catalog_root / "header.json"),
            "manifest_sha256": args.manifest_sha256,
            "job_yaml_sha256": sha256_file(args.job_yaml),
            "analyzer_sha256": sha256_file(Path(__file__).resolve()),
            "supervisor_sha256": sha256_file(
                Path(__file__).resolve().with_name("supervise_offline_gpic_probation.py")
            ),
            "train_sha256": sha256_file(Path(__file__).resolve().parents[2] / "train.py"),
            "dataset_sha256": sha256_file(
                Path(__file__).resolve().parents[2]
                / "training/dataline_offline/offline_dataset.py"
            ),
            "blob_runtime_sha256": sha256_file(
                Path(__file__).resolve().parents[2]
                / "training/dataline_offline/gpic_blob.py"
            ),
        },
        "failures": failures,
    }
    return value, not failures


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--artifact-root", type=Path, required=True)
    value.add_argument("--buildtag", required=True)
    value.add_argument("--manifest-sha256", required=True)
    value.add_argument("--result-root", type=Path, required=True)
    value.add_argument("--checkpoint-root", type=Path, required=True)
    value.add_argument("--jobtag", required=True)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--plan-root", type=Path, required=True)
    value.add_argument("--catalog-root", type=Path, required=True)
    value.add_argument("--job-yaml", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    result, passed = analyze(args)
    atomic_json(args.output, result)
    print(
        f"GPIC_BLOB_PROBATION_{result['status']} classification={result['classification']} "
        f"conservative_s={result['speed_gate']['conservative_steady_mean_s']:.3f} "
        f"predicted_days={result['speed_gate']['predicted_formal_days']:.3f} "
        f"output={args.output}"
    )
    return 0 if passed else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AnalysisError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
