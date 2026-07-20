#!/usr/bin/env python3
"""Static/synthetic selftests for the W16 GPIC fd crossover harness."""

from __future__ import annotations

import json
import inspect
import os
import tempfile
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

from scripts.offline_arb import analyze_w16_gpic_fd_crossover_abba as analyzer
from scripts.offline_arb import run_w16_gpic_fd_crossover_abba as runner


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_static_contract() -> None:
    require(
        runner.ARMS
        == (
            ("strict_x", 16, 0, 192),
            ("fd_x", 16, 0, 192),
            ("fd_y", 16, 192, 384),
            ("strict_y", 16, 192, 384),
        ),
        "arm order/slices drifted",
    )
    require(runner.MODE_BY_ARM == analyzer.MODE_BY_ARM, "runner/analyzer mode map drift")
    require(runner.RUN_ID == analyzer.RUN_ID, "runner/analyzer RUN_ID drift")
    require(runner.SLICE_STEPS == analyzer.SLICE_STEPS == 192, "slice size drift")
    strict_nonce = runner.arm_nonce("a" * 64, "strict_x", 16, 0, 192)
    fd_nonce = runner.arm_nonce("a" * 64, "fd_x", 16, 0, 192)
    require(strict_nonce != fd_nonce, "mode/arm nonce collision")
    require(
        fd_nonce == analyzer.expected_arm_nonce("a" * 64, "fd_x", 16, 0, 192),
        "runner/analyzer nonce algorithm drift",
    )
    require(
        runner.arm_nonce("b" * 64, "fd_x", 16, 0, 192) != fd_nonce,
        "attempt nonce did not bind arm nonce",
    )
    require(analyzer._positive_drift(1.0, 0.951) <= 0.05, "0.049 drift should pass")
    require(analyzer._positive_drift(1.0, 0.949) > 0.05, "0.051 drift should fail")
    require(analyzer._positive_drift(1.0, 0.901) <= 0.10, "0.099 strict drift should pass")
    require(analyzer._positive_drift(1.0, 0.899) > 0.10, "0.101 strict drift should fail")


def test_schema_translation() -> None:
    with tempfile.TemporaryDirectory(prefix="fd_crossover_schema.") as raw_root:
        root = Path(raw_root)
        master_path = root / "master.json"
        master_path.write_text(
            json.dumps(
                {
                    "schema_version": "offline_arb_w16_gpic_fd_crossover_master_aggregate_v1",
                    "gpic_source_consistency_mode_by_arm": analyzer.MODE_BY_ARM,
                    "crossover_pairs": [list(pair) for pair in analyzer.CROSSOVER_PAIRS],
                    "semantic_external_gate": "PENDING",
                    "production_selection_qualified": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        translated = analyzer.compatibility_read_object(master_path, "synthetic master")
        require(translated == {"schema_version": analyzer.core.MASTER_SCHEMA}, "master translation drift")
        contract_path = root / "contract.json"
        contract_path.write_text(
            json.dumps(
                {
                    "schema_version": "offline_arb_w16_gpic_fd_crossover_arm_contract_v1",
                    "arm": "fd_x",
                    "gpic_source_consistency_mode": "fd_stable",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        translated = analyzer.compatibility_read_object(contract_path, "synthetic contract")
        require(
            translated
            == {"schema_version": analyzer.core.ARM_CONTRACT_SCHEMA, "arm": "fd_x"},
            "arm translation drift",
        )
        bad_path = root / "bad.json"
        bad_path.write_text(
            json.dumps(
                {
                    "schema_version": "offline_arb_w16_gpic_fd_crossover_arm_contract_v1",
                    "arm": "fd_x",
                    "gpic_source_consistency_mode": "strict_path",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            analyzer.compatibility_read_object(bad_path, "bad synthetic contract")
        except ValueError:
            pass
        else:
            raise AssertionError("wrong-mode contract was accepted")


def test_atomic_lines() -> None:
    read_fd, write_fd = os.pipe()
    children: list[int] = []
    processes = 16
    lines_per_process = 65
    for process_index in range(processes):
        pid = os.fork()
        if pid == 0:
            try:
                os.close(read_fd)
                os.dup2(write_fd, 1)
                if write_fd != 1:
                    os.close(write_fd)
                for line_index in range(lines_per_process):
                    runner._atomic_sentinel(
                        runner.STEP_PREFIX
                        +
                        f"arm=fd_x mode=fd_stable rank={process_index} step={line_index} "
                        f"relative_step={line_index} items=36 raw_bytes=1 wait_s=0.000000001 "
                        f"scheduled_until=8 item_ledger_sha256={'a' * 64}"
                    )
            except BaseException:
                os._exit(91)
            os._exit(0)
        children.append(pid)
    os.close(write_fd)
    chunks: list[bytes] = []
    while True:
        block = os.read(read_fd, 1 << 16)
        if not block:
            break
        chunks.append(block)
    os.close(read_fd)
    for pid in children:
        waited, status = os.waitpid(pid, 0)
        require(waited == pid and os.waitstatus_to_exitcode(status) == 0, "atomic child failed")
    raw = b"".join(chunks)
    lines = raw.decode("utf-8", errors="strict").splitlines()
    require(len(lines) == processes * lines_per_process, "atomic line count mismatch")
    require(len(set(lines)) == len(lines), "atomic sentinel lines duplicated/merged")
    require(all(analyzer.STEP_LINE_RE.fullmatch(line) for line in lines), "atomic line malformed")


def synthetic_rank_evidence(rank: int, start: int) -> dict:
    records = [
        {
            "step": start + relative,
            "relative": relative,
            "items": 36,
            "raw_bytes": 1000 + rank + relative,
            "item_ledger_sha256": analyzer.core.canonical_sha256(
                {"rank": rank, "step": start + relative, "items": 36}
            ),
        }
        for relative in range(192)
    ]
    overall = analyzer.core.canonical_sha256(
        {
            "schema_version": "offline_arb_w16_fd_crossover_step_ledger_v1",
            "rank": rank,
            "start_step": start,
            "stop_step": start + 192,
            "step_hashes": [
                {"step": record["step"], "sha256": record["item_ledger_sha256"]}
                for record in records
            ],
        }
    )
    return {
        "steps": records,
        "summary": {
            "raw_bytes": sum(record["raw_bytes"] for record in records),
            "step_ledger_sha256": overall,
        },
    }


def test_crossover_exactness() -> None:
    analyzer._RUNNER_EVIDENCE.clear()
    for rank in range(16):
        expected = synthetic_rank_evidence(rank, 0)
        analyzer._RUNNER_EVIDENCE[("strict_x", rank)] = expected
        analyzer._RUNNER_EVIDENCE[("fd_x", rank)] = json.loads(json.dumps(expected))
    exact = analyzer._pair_raw_exact("strict_x", "fd_x")
    require(exact["exact"] is True and exact["mismatches"] == [], "exact pair rejected")
    analyzer._RUNNER_EVIDENCE[("fd_x", 7)]["steps"][31]["raw_bytes"] += 1
    mismatch = analyzer._pair_raw_exact("strict_x", "fd_x")
    require(mismatch["exact"] is False and mismatch["mismatches"], "raw-byte drift accepted")


def test_analysis_attempt_namespaces() -> None:
    with tempfile.TemporaryDirectory(prefix="fd_crossover_attempts.") as raw_root:
        run_root = Path(raw_root) / analyzer.RUN_ID
        source_jobtag = "capture-job"
        capture_root = run_root / source_jobtag
        capture_root.mkdir(parents=True)
        master = capture_root / f"abba.{source_jobtag}.aggregate.json"
        master.write_text("{}\n", encoding="utf-8")
        roots = []
        outputs = []
        for analyzer_jobtag in ("analyzer-one", "analyzer-two"):
            nonce = analyzer.expected_analysis_attempt_nonce(
                analyzer.RUN_ID, source_jobtag, analyzer_jobtag
            )
            attempt_root = (
                run_root / "_analysis" / source_jobtag / analyzer_jobtag / nonce
            )
            attempt_root.mkdir(parents=True)
            capture_input = attempt_root / "capture_input_attestation.json"
            capture_input.write_text("{}\n", encoding="utf-8")
            output = attempt_root / analyzer.analysis_output_name(
                source_jobtag, analyzer_jobtag, nonce
            )
            args = SimpleNamespace(
                master=str(master),
                run_id=analyzer.RUN_ID,
                analyzer_jobtag=analyzer_jobtag,
                analysis_attempt_nonce=nonce,
                analysis_attempt_root=str(attempt_root),
                output=str(output),
            )
            require(
                analyzer._validate_analysis_attempt_identity(args) == attempt_root,
                "canonical analysis attempt was rejected",
            )
            wrong = SimpleNamespace(**vars(args))
            wrong.output = str(attempt_root / "wrong-source-binding.json")
            try:
                analyzer._validate_analysis_attempt_identity(wrong)
            except ValueError:
                pass
            else:
                raise AssertionError("wrong source/attempt output binding was accepted")
            roots.append(attempt_root)
            outputs.append(output)
        require(roots[0] != roots[1] and outputs[0] != outputs[1], "retry outputs collide")
        require(not any(path.exists() for path in outputs), "namespace validation wrote output")
        require(
            {path.name for path in capture_root.iterdir()} == {master.name},
            "analysis namespace setup modified capture",
        )
        linked_jobtag = "analyzer-linked-parent"
        linked_nonce = analyzer.expected_analysis_attempt_nonce(
            analyzer.RUN_ID, source_jobtag, linked_jobtag
        )
        outside_job_root = Path(raw_root) / "outside-analyzer-root"
        outside_attempt = outside_job_root / linked_nonce
        outside_attempt.mkdir(parents=True)
        (outside_attempt / "capture_input_attestation.json").write_text(
            "{}\n", encoding="utf-8"
        )
        linked_job_root = run_root / "_analysis" / source_jobtag / linked_jobtag
        linked_job_root.symlink_to(outside_job_root, target_is_directory=True)
        supplied_attempt = linked_job_root / linked_nonce
        linked_args = SimpleNamespace(
            master=str(master),
            run_id=analyzer.RUN_ID,
            analyzer_jobtag=linked_jobtag,
            analysis_attempt_nonce=linked_nonce,
            analysis_attempt_root=str(supplied_attempt),
            output=str(
                supplied_attempt
                / analyzer.analysis_output_name(
                    source_jobtag, linked_jobtag, linked_nonce
                )
            ),
        )
        try:
            analyzer._validate_analysis_attempt_identity(linked_args)
        except ValueError:
            pass
        else:
            raise AssertionError("symlinked analyzer parent hierarchy was accepted")


def test_performance_crossover() -> None:
    def arm(capacity: float, elapsed: float, wait: float, window: float) -> dict:
        return {
            "domain_min_rank_capacity_ratio": {"gpic": capacity},
            "rank_runner_elapsed_per_step_s": {
                str(rank): elapsed for rank in range(16)
            },
            "global_critical_wait": {
                "mean_s": wait,
                "p95_s": wait * 2,
                "max_s": wait * 3,
            },
            "combined_critical_windows": {
                "mean_s": window,
                "p95_s": window * 2,
                "max_s": window * 3,
            },
        }
    arms = {
        "strict_x": arm(0.5, 4.0, 3.0, 12.0),
        "fd_x": arm(1.0, 2.0, 1.5, 6.0),
    }
    value = analyzer._crossover_performance(arms, "strict_x", "fd_x")
    require(value["gpic_capacity_speedup_fd_over_strict"] == 2.0, "capacity speedup drift")
    require(value["runner_elapsed_speedup_strict_over_fd"]["min"] == 2.0, "elapsed speedup drift")
    require(value["global_critical_wait_speedup_strict_over_fd"]["mean_s"] == 2.0, "wait speedup drift")
    require(value["combined_critical_window_speedup_strict_over_fd"]["mean_s"] == 2.0, "window speedup drift")
    require(analyzer.GPUVAL_CANDIDATE_MIN_GPIC_SPEEDUP == 1.5, "diagnostic threshold drift")


def test_analysis_redirect_contract() -> None:
    source = textwrap.dedent(inspect.getsource(analyzer.core.analyze))
    needle = (
        'analysis_full_path = result_root / '
        'f"sidecarfull.analysis.{result_root.name}.json"'
    )
    require(source.count(needle) == 1, "core analysis-sidecar redirect anchor drift")


def test_first_publication_is_final() -> None:
    with tempfile.TemporaryDirectory(prefix="fd_crossover_publication.") as raw_root:
        root = Path(raw_root)
        original_base = runner.core.BASE
        original_local = runner.LOCAL_PUBLICATION_BASE
        runner.core.BASE = root / "base"
        runner.LOCAL_PUBLICATION_BASE = root / "local"
        try:
            jobtag = "synthetic-job"
            attempt = runner.capture_attempt_nonce(jobtag)
            node_spec = next(item for item in runner.ARMS if item[0] == "fd_x")
            aggregate_spec = next(item for item in runner.ARMS if item[0] == "strict_y")
            node_nonce = runner.arm_nonce(attempt, *node_spec)
            aggregate_nonce = runner.arm_nonce(attempt, *aggregate_spec)
            node_value = {
                "schema_version": "offline_arb_w16_gpic_abba_node_manifest_v1",
                "jobtag": jobtag,
                "attempt_nonce": attempt,
                "arm": node_spec[0],
                "arm_nonce": node_nonce,
                "threads": node_spec[1],
                "start_step": node_spec[2],
                "stop_step": node_spec[3],
                "node_rank": 0,
            }
            aggregate_value = {
                "schema_version": "offline_arb_w16_gpic_abba_arm_aggregate_v1",
                "jobtag": jobtag,
                "attempt_nonce": attempt,
                "arm": aggregate_spec[0],
                "arm_nonce": aggregate_nonce,
                "threads": aggregate_spec[1],
                "start_step": aggregate_spec[2],
                "stop_step": aggregate_spec[3],
            }
            master_value = {
                "schema_version": "offline_arb_w16_gpic_abba_master_aggregate_v1",
                "run_id": runner.RUN_ID,
                "jobtag": jobtag,
                "attempt_nonce": attempt,
                "arms": [
                    {
                        "arm": arm,
                        "path": str(
                            runner.core.BASE
                            / runner.RUN_ID
                            / jobtag
                            / f"evidence.{jobtag}"
                            / arm
                            / "arm_aggregate.json"
                        ),
                        "sha256": "b" * 64,
                        "status": "PASS",
                    }
                    for arm, _threads, _start, _stop in runner.ARMS
                ],
            }
            cases = (
                (
                    "node",
                    runner.LOCAL_PUBLICATION_BASE
                    / jobtag
                    / attempt
                    / node_spec[0]
                    / node_nonce
                    / "node0"
                    / "node_manifest.json",
                    node_value,
                    "offline_arb_w16_gpic_fd_crossover_node_manifest_v1",
                ),
                (
                    "aggregate",
                    runner.core.BASE
                    / runner.RUN_ID
                    / jobtag
                    / f"evidence.{jobtag}"
                    / aggregate_spec[0]
                    / "arm_aggregate.json",
                    aggregate_value,
                    "offline_arb_w16_gpic_fd_crossover_arm_aggregate_v1",
                ),
                (
                    "master",
                    runner.core.BASE
                    / runner.RUN_ID
                    / jobtag
                    / f"abba.{jobtag}.aggregate.json",
                    master_value,
                    "offline_arb_w16_gpic_fd_crossover_master_aggregate_v1",
                ),
            )
            for label, path, source_value, expected_schema in cases:
                errors = []

                def publish() -> None:
                    try:
                        runner._publication_atomic_write_json(path, source_value)
                    except BaseException as exc:
                        errors.append(exc)

                writer = threading.Thread(target=publish, name=f"publish-{label}")
                writer.start()
                while writer.is_alive() and not path.exists():
                    writer.join(timeout=0.001)
                require(path.is_file() and not path.is_symlink(), f"{label} was not published")
                observed = json.loads(path.read_text(encoding="utf-8"))
                writer.join()
                require(not errors, f"{label} first publication failed: {errors}")
                require(
                    observed.get("schema_version") == expected_schema,
                    f"observer saw non-final schema for {label}",
                )
                require(
                    source_value["schema_version"].startswith("offline_arb_w16_gpic_abba_"),
                    f"{label} publication mutated caller object",
                )
                if label == "master":
                    require(
                        observed.get("semantic_external_gate") == "PENDING"
                        and observed.get("production_selection_qualified") is False,
                        "master production/semantic root flags missing",
                    )
                try:
                    runner._publication_atomic_write_json(path, source_value)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(f"{label} second publication overwrote evidence")
                foreign = root / "wrong-parent" / path.name
                try:
                    runner._publication_atomic_write_json(foreign, source_value)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(
                        f"{label} correct basename was accepted under a foreign parent"
                    )
            wrong_schema = dict(node_value)
            wrong_schema["schema_version"] = "offline_arb_unexpected_schema_v1"
            try:
                runner._publication_atomic_write_json(cases[0][1], wrong_schema)
            except RuntimeError as exc:
                require(
                    "unexpected source schema" in str(exc),
                    "target schema rejection used the wrong failure path",
                )
            else:
                raise AssertionError("publication target accepted an unexpected source schema")
        finally:
            runner.core.BASE = original_base
            runner.LOCAL_PUBLICATION_BASE = original_local
    runner._install_extension()
    require(
        runner.core.atomic_write_json is runner._publication_atomic_write_json,
        "capture install did not route first publication through extension writer",
    )


def test_source_tokens() -> None:
    runner_source = Path(runner.__file__).read_text(encoding="utf-8")
    analyzer_source = Path(analyzer.__file__).read_text(encoding="utf-8")
    for token in (
        "gpic_source_consistency_mode",
        "path_stat_calls",
        "fstat_calls",
        "item_ledger_sha256",
        "timed_elapsed_s",
        "crossover_pairs",
        "GPUVAL_CANDIDATE",
        "analysis_output_attestation",
        "_CROSSOVER_ANALYSIS_ATTEMPT_ROOT",
        "_publication_atomic_write_json",
    ):
        require(token in runner_source or token in analyzer_source, f"missing source token: {token}")
    require("hashlib.sha256(raw" not in runner_source, "timed raw payload hashing was added")


def main() -> int:
    test_static_contract()
    test_schema_translation()
    test_atomic_lines()
    test_crossover_exactness()
    test_analysis_attempt_namespaces()
    test_performance_crossover()
    test_analysis_redirect_contract()
    test_first_publication_is_final()
    test_source_tokens()
    print("W16_GPIC_FD_CROSSOVER_STATIC_SELFTEST_PASS cases=9 atomic_lines=1040")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
