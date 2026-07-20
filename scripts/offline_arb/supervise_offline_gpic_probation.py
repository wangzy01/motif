#!/usr/bin/env python3
"""Cross-node supervisor and automatic hard-stop for the 8x8 GPIC probation."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


SHA_RE = re.compile(r"[0-9a-f]{64}")
SCHEMA = "offline_gpic_blob_probation_supervisor_v1"
STOP_SCHEMA = "offline_gpic_blob_probation_early_stop_v1"
HEALTH_ZERO_KEYS = (
    "gpic_allrank_sum_short_read_failures",
    "gpic_allrank_sum_integrity_failures",
    "gpic_allrank_sum_pread_errors",
    "gpic_allrank_sum_source_fallback_items",
    "gpic_allrank_health_snapshot_errors",
)


class SupervisorError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def atomic_json(path: Path, value: dict[str, Any], *, replace: bool = False) -> None:
    raw = canonical(value)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise SupervisorError(f"stale supervisor temporary: {temporary}")
    if not replace and (path.exists() or path.is_symlink()):
        raise SupervisorError(f"refusing supervisor output overwrite: {path}")
    with temporary.open("xb", buffering=0) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def metric_segments(path: Path) -> list[list[dict[str, Any]]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise SupervisorError(f"metrics path is not a regular file: {path}")
    segments: list[list[dict[str, Any]]] = [[]]
    previous = -1
    raw = path.read_bytes()
    lines = raw.splitlines()
    if raw and not raw.endswith(b"\n"):
        lines = lines[:-1]
    for line_number, line in enumerate(lines, 1):
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SupervisorError(f"malformed metrics JSON line {line_number}: {exc}") from exc
        if not isinstance(value, dict) or type(value.get("step")) is not int:
            raise SupervisorError(f"malformed metrics record line {line_number}")
        step = int(value["step"])
        if step <= 0:
            raise SupervisorError(f"invalid metrics step line {line_number}: {step}")
        if step <= previous:
            segments.append([])
        segments[-1].append(value)
        previous = step
    return [segment for segment in segments if segment]


def current_round_rows(path: Path, expected_first_step: int) -> list[dict[str, Any]]:
    for segment in reversed(metric_segments(path)):
        if int(segment[0]["step"]) == expected_first_step:
            return segment
    return []


def numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SupervisorError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise SupervisorError(f"{label} is invalid: {value!r}")
    return result


def health_failure(rows: list[dict[str, Any]]) -> str | None:
    for row in reversed(rows):
        if "gpic_allrank_min_blob_fd_cache_size" not in row:
            continue
        for key in HEALTH_ZERO_KEYS:
            if numeric(row.get(key), key) != 0.0:
                return f"{key}={row.get(key)} at step={row['step']}"
        if numeric(row.get("gpic_allrank_min_blob_fd_cache_size"), "min fd cache") != 1024:
            return f"min fd cache !=1024 at step={row['step']}"
        if numeric(row.get("gpic_allrank_max_blob_fd_cache_size"), "max fd cache") != 1024:
            return f"max fd cache !=1024 at step={row['step']}"
        if numeric(
            row.get("gpic_allrank_min_fd_cache_covers_blob_count"),
            "fd cache coverage",
        ) != 1.0:
            return f"fd cache does not cover blob set at step={row['step']}"
        requested = numeric(
            row.get("gpic_allrank_sum_pread_requested_bytes"), "pread requested"
        )
        returned = numeric(
            row.get("gpic_allrank_sum_pread_returned_bytes"), "pread returned"
        )
        # Supply threads remain active while this low-frequency snapshot is
        # gathered.  `_pread_exact` accounts requested bytes before the
        # blocking pread and returned bytes after it, so a non-negative gap is
        # an in-flight read, not a short read.  Completed-read failures remain
        # fail-closed through the zero counters above; completed W64 replay
        # separately requires exact byte closure.
        if returned > requested:
            return f"pread returned bytes exceed requested at step={row['step']}"
        break
    return None


def speed_observation(
    rows: list[dict[str, Any]], *, minimum: int, window: int
) -> dict[str, Any] | None:
    if len(rows) < minimum or len(rows) < window:
        return None
    selected = rows[-window:]
    values = [numeric(row.get("iter_s"), "iter_s") for row in selected]
    return {
        "rows": window,
        "step_start": int(selected[0]["step"]),
        "step_stop": int(selected[-1]["step"]),
        "median_iter_s": statistics.median(values),
        "mean_iter_s": statistics.fmean(values),
        "max_iter_s": max(values),
    }


def publish_stop(args: argparse.Namespace, reason: str, detail: dict[str, Any]) -> None:
    value = {
        "schema_version": STOP_SCHEMA,
        "status": "STOP",
        "jobtag": args.jobtag,
        "buildtag": args.buildtag,
        "manifest_sha256": args.manifest_sha256,
        "round": args.round,
        "reason": reason,
        "detail": detail,
        "published_unix_s": time.time(),
    }
    if args.stop_marker.exists():
        return
    atomic_json(args.stop_marker, value)


def validate_stop(path: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise SupervisorError("shared stop marker is not a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != STOP_SCHEMA
        or value.get("status") != "STOP"
        or value.get("jobtag") != args.jobtag
        or value.get("buildtag") != args.buildtag
        or value.get("manifest_sha256") != args.manifest_sha256
        or value.get("round") != args.round
    ):
        raise SupervisorError("shared stop marker identity is malformed/stale")
    return value


def terminate_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def stream_output(process: subprocess.Popen[bytes], log_handle: Any) -> None:
    assert process.stdout is not None
    while True:
        block = process.stdout.readline()
        if not block:
            break
        log_handle.write(block)
        log_handle.flush()
        sys.stdout.buffer.write(block)
        sys.stdout.buffer.flush()


def run(args: argparse.Namespace) -> int:
    if not args.command:
        raise SupervisorError("training command is empty")
    if args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SupervisorError("training command is empty after separator")
    if SHA_RE.fullmatch(args.manifest_sha256) is None:
        raise SupervisorError("manifest SHA-256 is malformed")
    if args.node_rank < 0 or args.node_rank > 7:
        raise SupervisorError("node rank is outside 0..7")
    if args.round not in {"A", "B"}:
        raise SupervisorError("probation round must be A or B")
    if args.log.exists() or args.log.is_symlink() or args.result.exists() or args.result.is_symlink():
        raise SupervisorError("probation supervisor log/result already exists")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stopped: dict[str, Any] | None = None
    consecutive_slow = 0
    last_checked_rows = 0
    latest_speed: dict[str, Any] | None = None
    with args.log.open("xb", buffering=0) as log_handle:
        process = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        output_thread = threading.Thread(
            target=stream_output,
            args=(process, log_handle),
            name="probation-output-copy",
            daemon=True,
        )
        output_thread.start()
        try:
            while process.poll() is None:
                existing_stop = validate_stop(args.stop_marker, args)
                if existing_stop is not None:
                    stopped = existing_stop
                    terminate_group(process)
                    break
                if time.monotonic() - started > args.timeout_s:
                    if args.node_rank == 0:
                        publish_stop(args, "round_timeout", {"timeout_s": args.timeout_s})
                    time.sleep(1)
                    continue
                if args.node_rank == 0:
                    rows = current_round_rows(args.metrics, args.expected_first_step)
                    failure = health_failure(rows)
                    if failure is not None:
                        publish_stop(args, "overlay_health_failure", {"message": failure})
                        time.sleep(1)
                        continue
                    if len(rows) >= last_checked_rows + args.check_stride:
                        last_checked_rows = len(rows)
                        latest_speed = speed_observation(
                            rows, minimum=args.minimum_observations, window=args.window
                        )
                        if latest_speed is not None:
                            slow = (
                                latest_speed["median_iter_s"] > args.hard_max_iter_s
                                and latest_speed["mean_iter_s"] > args.hard_max_iter_s
                            )
                            consecutive_slow = consecutive_slow + 1 if slow else 0
                            if consecutive_slow >= args.slow_windows:
                                publish_stop(args, "sustained_speed_above_hard_gate", latest_speed)
                                time.sleep(1)
                                continue
                time.sleep(2)
            rc = process.wait()
        finally:
            if process.poll() is None:
                terminate_group(process)
            output_thread.join(timeout=30)
    existing_stop = validate_stop(args.stop_marker, args)
    if existing_stop is not None:
        stopped = existing_stop
    result = {
        "schema_version": SCHEMA,
        "status": "STOPPED" if stopped else ("PASS" if rc == 0 else "FAIL"),
        "jobtag": args.jobtag,
        "buildtag": args.buildtag,
        "manifest_sha256": args.manifest_sha256,
        "round": args.round,
        "node_rank": args.node_rank,
        "host": os.uname().nodename,
        "expected_first_step": args.expected_first_step,
        "train_returncode": rc,
        "stopped": stopped,
        "latest_speed_observation": latest_speed,
        "wall_s": time.monotonic() - started,
        "hard_max_iter_s": args.hard_max_iter_s,
    }
    atomic_json(args.result, result)
    if stopped is not None:
        print(f"PROBATION_AUTO_STOP round={args.round} node={args.node_rank} reason={stopped['reason']}")
        return 86
    if rc != 0:
        print(f"PROBATION_TRAIN_FAIL round={args.round} node={args.node_rank} rc={rc}")
        return rc if 0 < rc < 126 else 87
    print(f"PROBATION_SUPERVISOR_PASS round={args.round} node={args.node_rank}")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--node-rank", type=int, required=True)
    value.add_argument("--round", required=True)
    value.add_argument("--metrics", type=Path, required=True)
    value.add_argument("--log", type=Path, required=True)
    value.add_argument("--result", type=Path, required=True)
    value.add_argument("--stop-marker", type=Path, required=True)
    value.add_argument("--jobtag", required=True)
    value.add_argument("--buildtag", required=True)
    value.add_argument("--manifest-sha256", required=True)
    value.add_argument("--expected-first-step", type=int, required=True)
    value.add_argument("--minimum-observations", type=int, default=100)
    value.add_argument("--window", type=int, default=40)
    value.add_argument("--check-stride", type=int, default=10)
    value.add_argument("--slow-windows", type=int, default=2)
    value.add_argument("--hard-max-iter-s", type=float, default=2.75)
    value.add_argument("--timeout-s", type=float, default=10800.0)
    value.add_argument("command", nargs=argparse.REMAINDER)
    return value


if __name__ == "__main__":
    try:
        raise SystemExit(run(parser().parse_args()))
    except (SupervisorError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
