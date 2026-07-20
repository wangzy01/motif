#!/usr/bin/env python3
"""Static contract checks for submission-blocked post-build overlay gates."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
PYTHON_FILES = (
    "scripts/offline_arb/run_offline_gpic_blob_semantic_ab.py",
    "scripts/offline_arb/run_w64_gpic_blob_supply_replay.py",
    "scripts/offline_arb/verify_offline_gpic_compact_controls.py",
    "scripts/offline_arb/supervise_offline_gpic_probation.py",
    "scripts/offline_arb/analyze_offline_gpic_probation.py",
)
TEMPLATES = (
    "jobs/offline_arb_gpic_blob_semantic_ab_TEMPLATE.yaml",
    "jobs/offline_arb_w64_gpic_blob_supply_replay_TEMPLATE.yaml",
    "jobs/offline_exact_256_8x8_gpic_blob_probation_TEMPLATE.yaml",
)


def read(relative: str) -> str:
    return (REPO / relative).read_text(encoding="utf-8")


def digest(relative: str) -> str:
    return hashlib.sha256((REPO / relative).read_bytes()).hexdigest()


def assert_submission_blocked(relative: str, text: str) -> None:
    for exact in ('ARTIFACT_ROOT=""', 'ARTIFACT_BUILDTAG=""', 'MANIFEST_SHA256=""'):
        assert text.count(exact) == 1, (relative, exact)
    for key in ("ARTIFACT_ROOT", "ARTIFACT_BUILDTAG", "MANIFEST_SHA256"):
        assert f'test -n "${key}"' in text, (relative, key)
    assert '"$ARTIFACT_BUILDTAG" != latest' in text
    assert "jcc job submit" not in text
    assert "sed -i" not in text


def exercise_live_health_accounting(
    supervisor: dict[str, object], analyzer: dict[str, object]
) -> None:
    """A live read may be in flight; only returned>requested is impossible."""

    base = {
        "step": 1,
        "gpic_allrank_sum_short_read_failures": 0,
        "gpic_allrank_sum_integrity_failures": 0,
        "gpic_allrank_sum_pread_errors": 0,
        "gpic_allrank_sum_source_fallback_items": 0,
        "gpic_allrank_health_snapshot_errors": 0,
        "gpic_allrank_min_blob_fd_cache_size": 1024,
        "gpic_allrank_max_blob_fd_cache_size": 1024,
        "gpic_allrank_min_fd_cache_covers_blob_count": 1,
        "gpic_allrank_sum_pread_requested_bytes": 100,
        "gpic_allrank_sum_pread_returned_bytes": 90,
        "gpic_allrank_slowest_supply_img_s": 1.0,
        "gpic_allrank_sum_read_items": 1,
        "gpic_allrank_max_rss_bytes": 1,
        "gpic_allrank_max_process_fd_count": 1,
        "gpic_allrank_max_blob_fd_peak": 1,
        "gpic_allrank_min_rlimit_nofile_soft": 4096,
        "gpic_allrank_max_manifest_blob_count": 2,
    }
    health_failure = supervisor["health_failure"]
    assert health_failure([base]) is None  # type: ignore[operator]

    impossible = dict(base, gpic_allrank_sum_pread_returned_bytes=101)
    assert health_failure([impossible]) is not None  # type: ignore[operator]
    for key in supervisor["HEALTH_ZERO_KEYS"]:  # type: ignore[union-attr]
        failed = dict(base)
        failed[key] = 1
        assert health_failure([failed]) is not None  # type: ignore[operator]

    rows = [
        {
            **base,
            "step": step,
            "gpic_allrank_sum_pread_requested_bytes": 100 * step,
            "gpic_allrank_sum_pread_returned_bytes": 100 * step - 10,
            "gpic_allrank_sum_read_items": step,
        }
        for step in range(1, 11)
    ]
    validate_health = analyzer["validate_health"]
    summary = validate_health(rows)  # type: ignore[operator]
    assert summary["snapshots"] == 10 and summary["final_pread_bytes"] == 990

    impossible_rows = [dict(row) for row in rows]
    impossible_rows[4]["gpic_allrank_sum_pread_returned_bytes"] = (
        impossible_rows[4]["gpic_allrank_sum_pread_requested_bytes"] + 1
    )
    try:
        validate_health(impossible_rows)  # type: ignore[operator]
    except analyzer["AnalysisError"]:  # type: ignore[misc]
        pass
    else:
        raise AssertionError("analyzer accepted returned>requested")


def main() -> int:
    for relative in PYTHON_FILES:
        source = read(relative)
        ast.parse(source, filename=relative)
        assert "jcc job submit" not in source

    loaded: dict[str, dict] = {}
    for relative in TEMPLATES:
        text = read(relative)
        value = yaml.safe_load(text)
        assert isinstance(value, dict) and isinstance(value.get("command"), str)
        assert value.get("maxRetries") == 0
        assert_submission_blocked(relative, text)
        checked = subprocess.run(
            ["bash", "-n"],
            input=value["command"],
            text=True,
            capture_output=True,
            check=False,
        )
        assert checked.returncode == 0, (relative, checked.stderr)
        loaded[relative] = value

    semantic = read("scripts/offline_arb/run_offline_gpic_blob_semantic_ab.py")
    for token in (
        '"raw_sha256"',
        '"native_rgb"',
        '"final_pixels_sha256"',
        '"prompts"',
        '"cfg_keep"',
        '"fresh_resume_exact": True',
        '"source_fallback_items": 0',
        "fresh suffix and exact-resume semantics differ",
    ):
        assert token in semantic, token
    semantic_yaml = read("jobs/offline_arb_gpic_blob_semantic_ab_TEMPLATE.yaml")
    assert "cpuPerWorker: 16" in semantic_yaml and "gpuPerWorker: 0" in semantic_yaml
    assert "configs/stage256_offline_exact_v1.yaml" in semantic_yaml
    assert "run_arm source_files_v1 source" in semantic_yaml
    assert "run_arm offline_gpic_blob_v1 blob" in semantic_yaml

    w64 = read("scripts/offline_arb/run_w64_gpic_blob_supply_replay.py")
    for token in (
        "WORLD_SIZE = 64",
        "LOCAL_WORLD_SIZE = 8",
        "FD_CACHE_SIZE = 1024",
        "CAPACITY_MIN = 1.25",
        "TARGET_STEP_SECONDS = 2.50",
        '"payload_backend": "offline_gpic_blob_v1"',
        '"storage_backend": "mountpoint_s3"',
        '"short_read_failures"',
        '"integrity_failures"',
        '"source_fallback_items"',
        '"rlimit_nofile_soft"',
        '"process_rss_bytes"',
        "accept_collective_index_verification",
    ):
        assert token in w64, token
    w64_yaml = loaded["jobs/offline_arb_w64_gpic_blob_supply_replay_TEMPLATE.yaml"]
    assert w64_yaml["numWorkers"] == 8 and w64_yaml["cpuPerWorker"] == 16
    assert w64_yaml["gpuPerWorker"] == 0
    assert "--nproc-per-node 8" in w64_yaml["command"]

    compact = read("scripts/offline_arb/verify_offline_gpic_compact_controls.py")
    assert "content_hash_deferred" in compact
    assert "full_publication_verified_once" in compact
    assert "full_segment_sha256_verified" in compact
    assert "exceeds fd_cache_size=1024" in compact
    assert "manifest self-hash differs" in compact
    assert "sha(index)" not in compact

    supervisor = read("scripts/offline_arb/supervise_offline_gpic_probation.py")
    for token in (
        "sustained_speed_above_hard_gate",
        "overlay_health_failure",
        "gpic_allrank_sum_source_fallback_items",
        "gpic_allrank_min_blob_fd_cache_size",
        "os.killpg",
        "hard-max-iter-s",
    ):
        assert token in supervisor, token
    analyzer = read("scripts/offline_arb/analyze_offline_gpic_probation.py")
    for token in (
        "TARGET_MAX_S = 2.55",
        "HARD_MAX_S = 2.75",
        "exact_steps(round_a, 1, 250",
        "exact_steps(round_b, 201, 500",
        "resumed from",
        "source_fallback",
        "predicted_formal_days",
        "compact preflight binding drift",
        "sidecar compact control drift",
    ):
        assert token in analyzer, token
    probation_yaml = loaded[
        "jobs/offline_exact_256_8x8_gpic_blob_probation_TEMPLATE.yaml"
    ]
    assert probation_yaml["numWorkers"] == 8
    assert probation_yaml["gpuPerWorker"] == 8
    assert probation_yaml["cpuPerWorker"] == 48
    command = probation_yaml["command"]
    assert "configs/stage256_offline_exact_v1.yaml" in command
    assert "dataset.gpic_payload_backend=offline_gpic_blob_v1" in command
    assert "dataset.gpic_blob_fd_cache_size=1024" in command
    assert "run_round A 250 false 1" in command
    assert "run_round B 500 true  201" in command
    assert "trainer.checkpoint_steps=200" in command
    assert "early_stop.$JOBTAG.json" in command
    assert "verify_offline_gpic_compact_controls.py" in command

    # Exercise the monitor's monotonic-segment parser without importing or
    # starting training.  This is intentionally tiny and entirely local to the
    # static JCC worker.
    namespace: dict[str, object] = {"__name__": "static_import"}
    exec(compile(supervisor, "supervisor", "exec"), namespace)
    analyzer_namespace: dict[str, object] = {"__name__": "static_import"}
    exec(compile(analyzer, "analyzer", "exec"), analyzer_namespace)
    exercise_live_health_accounting(namespace, analyzer_namespace)
    with tempfile.TemporaryDirectory(prefix="gpic-overlay-static-") as temporary:
        metrics = Path(temporary) / "metrics.jsonl"
        rows = [
            {"step": 1, "iter_s": 2.5, "data_wait_s": 0.1, "data_wait_frac": 0.04},
            {"step": 2, "iter_s": 2.6, "data_wait_s": 0.1, "data_wait_frac": 0.038},
            {"step": 250, "iter_s": 2.5, "data_wait_s": 0.1, "data_wait_frac": 0.04},
            {"step": 201, "iter_s": 2.5, "data_wait_s": 0.1, "data_wait_frac": 0.04},
        ]
        metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        segments = namespace["metric_segments"](metrics)  # type: ignore[index,operator]
        assert [[row["step"] for row in segment] for segment in segments] == [
            [1, 2, 250],
            [201],
        ]

    all_files = PYTHON_FILES + TEMPLATES + (
        "scripts/offline_arb/test_offline_gpic_overlay_gates_static.py",
        "jobs/offline_arb_gpic_overlay_gates_static.yaml",
    )
    print(
        "OFFLINE_GPIC_OVERLAY_GATES_STATIC_PASS "
        "submission_blocked=1 semantic_raw_decode_trace_prompt_cfg_resume=1 "
        "w64=64 fd_cache=1024 capacity=1.25 health=1 "
        "live_pread_gap=accepted impossible_pread_gap=rejected error_metrics=rejected "
        "probation=8x8 exact_resume=200 hard_speed=2.75 target=2.55 auto_stop=1 "
        + " ".join(f"{Path(path).name}={digest(path)}" for path in all_files)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
