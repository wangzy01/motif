#!/usr/bin/env python3
"""Independent standard-library verifier for the runtime prompt smoke receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


POLICY_SCHEMA = "stage512a_control_runtime_prompt_policy_v1"
REPORT_SCHEMA = "stage512a_control_runtime_prompt_smoke_report_v1"
SUCCESS_SCHEMA = "stage512a_control_runtime_prompt_smoke_success_v1"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

POLICY_KEYS = {"schema", "policy_id", "config", "execution", "assets", "runtime_files", "cases"}
CONFIG_KEYS = {"path", "sha256"}
EXECUTION_KEYS = {
    "rank", "world_size", "gpic_payload_backend", "gpic_payload_in_memory",
    "image_payload_reads", "fresh_dataset_constructions",
}
ASSET_KEYS = {
    "plan_root", "plan_hash", "plan_header_sha256", "plan_checksums_sha256",
    "catalog_root", "catalog_hash", "catalog_header_sha256",
    "sampling_config_path", "sampling_config_sha256", "caption_sidecar_root",
    "caption_sidecar_header_sha256", "anime_freq_root", "anime_freq_header_sha256",
}
CASE_KEYS = {
    "case_id", "step", "rank", "slot", "row_id", "uid", "domain",
    "bucket_idx", "bucket_hw", "planned_caption_band", "realized_caption_band",
    "cfg_protected", "prompt_utf8_length", "prompt_sha256",
}
RECORD_KEYS = CASE_KEYS | {"prompt_utf8_hex"}
REPORT_KEYS = {
    "schema", "verdict", "pass", "policy", "release", "assets",
    "config_bindings", "execution", "runtime", "runs",
    "repeat_bitwise_equal", "aggregate_digest",
}
POLICY_RECEIPT_KEYS = {"path", "sha256", "policy_id"}
RELEASE_KEYS = {"repo", "git_commit", "config", "self_files", "runtime_files"}
CONFIG_BINDING_KEYS = {
    "stage", "per_rank_batch", "offline_plan_root", "offline_plan_hash",
    "offline_catalog_root", "sampling_config", "offline_caption_sidecar_root",
    "offline_anime_freq_root",
}
RUNTIME_KEYS = {
    "python", "torch", "pyarrow", "transformers", "tokenizers",
    "tokenizer_path", "cuda_visible_devices",
}
RUN_KEYS = {
    "construction_index", "records", "records_digest", "image_payload_read_calls",
    "pipeline_started", "gpic_blob_initialized",
}
SUCCESS_KEYS = {
    "schema", "verdict", "policy_sha256", "aggregate_digest", "report_sha256",
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_constant(token: str) -> None:
    raise ValidationError(f"non-standard JSON constant: {token}")


def strict_json(raw: bytes, label: str) -> dict[str, Any]:
    require(raw.endswith(b"\n"), f"{label} is not newline terminated")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=strict_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise ValidationError(f"invalid {label}: {exc}") from exc
    require(type(value) is dict, f"{label} root is not an object")
    return value


def keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    require(type(value) is dict, f"{label} is not an object")
    actual = set(value)
    require(
        actual == expected,
        f"{label} key closure mismatch: missing={sorted(expected - actual)} "
        f"extra={sorted(actual - expected)}",
    )
    return value


def sha(value: Any, label: str) -> str:
    require(type(value) is str and SHA_RE.fullmatch(value) is not None,
            f"{label} is not a lowercase SHA-256")
    return value


def relative(value: Any, label: str) -> str:
    require(type(value) is str and value, f"{label} is empty/non-string")
    parsed = PurePosixPath(value)
    require(
        not parsed.is_absolute() and "." not in parsed.parts and ".." not in parsed.parts
        and str(parsed) == value,
        f"{label} is not a canonical repo-relative path",
    )
    return value


def absolute(value: Any, label: str) -> str:
    require(type(value) is str and value.startswith("/") and not value.startswith("//"),
            f"{label} is not absolute")
    require(str(PurePosixPath(value)) == value, f"{label} is not canonical")
    return value


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256_bytes(raw)


def regular(path: Path, mode: int | None, label: str) -> None:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink(),
            f"{label} is not an ordinary non-symlink file")
    require(value.st_nlink == 1, f"{label} nlink is not 1")
    if mode is not None:
        require(stat.S_IMODE(value.st_mode) == mode,
                f"{label} mode is not {mode:04o}")


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ValidationError(f"git {' '.join(args)} failed: {exc.output.strip()}") from exc


def validate_policy(policy: dict[str, Any]) -> None:
    keys(policy, POLICY_KEYS, "policy")
    require(policy["schema"] == POLICY_SCHEMA, "policy schema mismatch")
    require(type(policy["policy_id"]) is str and policy["policy_id"], "invalid policy_id")
    config = keys(policy["config"], CONFIG_KEYS, "policy.config")
    relative(config["path"], "policy.config.path")
    sha(config["sha256"], "policy.config.sha256")
    execution = keys(policy["execution"], EXECUTION_KEYS, "policy.execution")
    require(execution == {
        "rank": 0, "world_size": 64, "gpic_payload_backend": "source_files_v1",
        "gpic_payload_in_memory": False, "image_payload_reads": 0,
        "fresh_dataset_constructions": 2,
    }, "policy execution contract drift")
    assets = keys(policy["assets"], ASSET_KEYS, "policy.assets")
    for name in ("plan_root", "catalog_root", "sampling_config_path",
                 "caption_sidecar_root", "anime_freq_root"):
        absolute(assets[name], f"policy.assets.{name}")
    for name in ASSET_KEYS - {
        "plan_root", "catalog_root", "sampling_config_path", "caption_sidecar_root",
        "anime_freq_root",
    }:
        sha(assets[name], f"policy.assets.{name}")
    runtime = policy["runtime_files"]
    require(type(runtime) is dict and runtime, "empty/non-object policy.runtime_files")
    for name, digest in runtime.items():
        relative(name, f"policy.runtime_files[{name!r}]")
        sha(digest, f"policy.runtime_files[{name!r}]")
    cases = policy["cases"]
    require(type(cases) is list and len(cases) == 3, "policy must contain three cases")
    ids: set[str] = set()
    coordinates: set[tuple[int, int, int]] = set()
    domains: set[str] = set()
    for index, raw_case in enumerate(cases):
        case = keys(raw_case, CASE_KEYS, f"policy.cases[{index}]")
        require(type(case["case_id"]) is str and case["case_id"] not in ids,
                "invalid/duplicate case_id")
        ids.add(case["case_id"])
        for name in ("step", "rank", "slot", "row_id", "bucket_idx",
                     "planned_caption_band", "realized_caption_band", "prompt_utf8_length"):
            require(type(case[name]) is int, f"case integer field malformed: {name}")
        coord = (case["step"], case["rank"], case["slot"])
        require(coord not in coordinates, "duplicate case coordinate")
        coordinates.add(coord)
        require(case["rank"] == 0, "case rank drift")
        require(type(case["uid"]) is str and case["uid"], "case uid malformed")
        require(case["domain"] in {"anime", "textatlas", "gpic"}, "case domain malformed")
        domains.add(case["domain"])
        require(type(case["bucket_hw"]) is list and len(case["bucket_hw"]) == 2
                and all(type(x) is int and x > 0 for x in case["bucket_hw"]),
                "case bucket_hw malformed")
        require(type(case["cfg_protected"]) is bool, "case cfg_protected malformed")
        require(case["prompt_utf8_length"] > 0, "case prompt is empty")
        sha(case["prompt_sha256"], "case prompt SHA")
    require(domains == {"anime", "textatlas", "gpic"}, "policy domain coverage drift")


def validate_record(record: Any, expected: dict[str, Any], label: str) -> None:
    value = keys(record, RECORD_KEYS, label)
    for name in CASE_KEYS:
        require(value[name] == expected[name],
                f"{label}.{name} differs from fixed policy")
    prompt_hex = value["prompt_utf8_hex"]
    require(type(prompt_hex) is str and len(prompt_hex) % 2 == 0,
            f"{label}.prompt_utf8_hex malformed")
    try:
        raw = bytes.fromhex(prompt_hex)
    except ValueError as exc:
        raise ValidationError(f"{label}.prompt_utf8_hex malformed: {exc}") from exc
    require(raw.hex() == prompt_hex, f"{label}.prompt_utf8_hex is not canonical lowercase hex")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{label} prompt is not valid UTF-8: {exc}") from exc
    require(len(raw) == value["prompt_utf8_length"], f"{label} prompt length mismatch")
    require(sha256_bytes(raw) == value["prompt_sha256"], f"{label} prompt SHA mismatch")


def validate_external_assets(assets: dict[str, Any]) -> None:
    paths = {
        "plan_header_sha256": Path(assets["plan_root"]) / "header.json",
        "plan_checksums_sha256": Path(assets["plan_root"]) / "checksums.json",
        "catalog_header_sha256": Path(assets["catalog_root"]) / "header.json",
        "sampling_config_sha256": Path(assets["sampling_config_path"]),
        "caption_sidecar_header_sha256": Path(assets["caption_sidecar_root"]) / "header.json",
        "anime_freq_header_sha256": Path(assets["anime_freq_root"]) / "header.json",
    }
    for name, path in paths.items():
        regular(path, None, f"external asset {name}")
        require(sha256_file(path) == assets[name], f"external asset SHA drift: {path}")
    plan_header = strict_json(paths["plan_header_sha256"].read_bytes(), "plan header")
    catalog_header = strict_json(paths["catalog_header_sha256"].read_bytes(), "catalog header")
    require(plan_header.get("plan_hash") == assets["plan_hash"], "plan semantic hash drift")
    require(catalog_header.get("catalog_hash") == assets["catalog_hash"],
            "catalog semantic hash drift")


def validate_artifact(
    *, artifact_dir: Path, repo: Path, config_path: Path, policy_path: Path,
    expected_commit: str, expected_config_sha: str, expected_policy_sha: str,
    expected_runner_sha: str, expected_validator_sha: str,
) -> dict[str, Any]:
    require(COMMIT_RE.fullmatch(expected_commit) is not None, "expected commit malformed")
    for name, value in (
        ("config", expected_config_sha), ("policy", expected_policy_sha),
        ("runner", expected_runner_sha), ("validator", expected_validator_sha),
    ):
        sha(value, f"expected {name} SHA")
    repo = repo.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    policy_path = policy_path.resolve(strict=True)
    artifact_dir = artifact_dir.resolve(strict=True)
    require(repo.is_absolute(), "repo is not absolute")
    require(git(repo, "rev-parse", "HEAD") == expected_commit, "Git HEAD mismatch")
    require(git(repo, "status", "--porcelain") == "", "release repo is dirty")

    directory_stat = artifact_dir.lstat()
    require(stat.S_ISDIR(directory_stat.st_mode) and not artifact_dir.is_symlink(),
            "artifact root is not an ordinary directory")
    require(stat.S_IMODE(directory_stat.st_mode) == 0o555, "artifact root mode is not 0555")
    require(sorted(path.name for path in artifact_dir.iterdir()) == ["REPORT.json", "_SUCCESS"],
            "artifact root member closure mismatch")
    report_path = artifact_dir / "REPORT.json"
    success_path = artifact_dir / "_SUCCESS"
    regular(report_path, 0o444, "REPORT.json")
    regular(success_path, 0o444, "_SUCCESS")
    report_raw = report_path.read_bytes()
    success_raw = success_path.read_bytes()
    report = strict_json(report_raw, "REPORT.json")
    success = strict_json(success_raw, "_SUCCESS")
    policy_raw = policy_path.read_bytes()
    policy = strict_json(policy_raw, "policy")
    validate_policy(policy)
    require(sha256_bytes(policy_raw) == expected_policy_sha, "policy SHA drift")
    require(sha256_file(config_path) == expected_config_sha == policy["config"]["sha256"],
            "config SHA drift")
    require(str(config_path.relative_to(repo)) == policy["config"]["path"],
            "config path differs from policy")

    keys(report, REPORT_KEYS, "report")
    require(report["schema"] == REPORT_SCHEMA, "report schema mismatch")
    require(report["verdict"] == "PASS" and report["pass"] is True,
            "report does not say PASS")
    policy_receipt = keys(report["policy"], POLICY_RECEIPT_KEYS, "report.policy")
    require(policy_receipt == {
        "path": str(policy_path.relative_to(repo)), "sha256": expected_policy_sha,
        "policy_id": policy["policy_id"],
    }, "report policy binding mismatch")

    release = keys(report["release"], RELEASE_KEYS, "report.release")
    require(release["repo"] == str(repo), "report repo mismatch")
    require(release["git_commit"] == expected_commit, "report commit mismatch")
    release_config = keys(release["config"], CONFIG_KEYS, "report.release.config")
    require(release_config == policy["config"], "report config binding mismatch")
    expected_self = {
        "scripts/stage512a_control_runtime_prompt_smoke_v1.py": expected_runner_sha,
        "scripts/validate_stage512a_control_runtime_prompt_smoke_v1.py": expected_validator_sha,
        str(policy_path.relative_to(repo)): expected_policy_sha,
    }
    require(release["self_files"] == expected_self, "self-file binding mismatch")
    for name, digest in expected_self.items():
        regular(repo / name, None, f"self file {name}")
        require(sha256_file(repo / name) == digest, f"self-file SHA drift: {name}")
    require(release["runtime_files"] == policy["runtime_files"],
            "report runtime-file map differs from policy")
    for name, digest in policy["runtime_files"].items():
        regular(repo / name, None, f"runtime file {name}")
        require(git(repo, "ls-files", "--error-unmatch", name) == name,
                f"runtime file is not committed: {name}")
        require(sha256_file(repo / name) == digest, f"runtime-file SHA drift: {name}")

    report_assets = keys(report["assets"], ASSET_KEYS, "report.assets")
    require(report_assets == policy["assets"], "report asset bindings differ from policy")
    validate_external_assets(report_assets)
    bindings = keys(report["config_bindings"], CONFIG_BINDING_KEYS, "report.config_bindings")
    expected_bindings = {
        "stage": "512a", "per_rank_batch": 32,
        "offline_plan_root": report_assets["plan_root"],
        "offline_plan_hash": report_assets["plan_hash"],
        "offline_catalog_root": report_assets["catalog_root"],
        "sampling_config": report_assets["sampling_config_path"],
        "offline_caption_sidecar_root": report_assets["caption_sidecar_root"],
        "offline_anime_freq_root": report_assets["anime_freq_root"],
    }
    require(bindings == expected_bindings, "effective config bindings mismatch")
    execution = keys(report["execution"], EXECUTION_KEYS, "report.execution")
    require(execution == policy["execution"], "execution contract mismatch")
    runtime = keys(report["runtime"], RUNTIME_KEYS, "report.runtime")
    for name in RUNTIME_KEYS - {"cuda_visible_devices"}:
        require(type(runtime[name]) is str and runtime[name], f"runtime {name} is empty")
    require(runtime["cuda_visible_devices"] == "", "CUDA must be hidden from zero-GPU smoke")

    runs = report["runs"]
    require(type(runs) is list and len(runs) == 2, "report must contain exactly two runs")
    for index, raw_run in enumerate(runs):
        run = keys(raw_run, RUN_KEYS, f"report.runs[{index}]")
        require(run["construction_index"] == index, "construction index mismatch")
        require(run["image_payload_read_calls"] == 0, "image payload read occurred")
        require(run["pipeline_started"] is False, "image pipeline started")
        require(run["gpic_blob_initialized"] is False, "GPIC blob initialized")
        records = run["records"]
        require(type(records) is list and len(records) == len(policy["cases"]),
                "run record count mismatch")
        for case_index, (record, expected) in enumerate(zip(records, policy["cases"])):
            validate_record(record, expected, f"report.runs[{index}].records[{case_index}]")
        require(run["records_digest"] == canonical_sha(records), "records digest mismatch")
    require(report["repeat_bitwise_equal"] is True, "repeat equality flag is not true")
    require(runs[0]["records"] == runs[1]["records"], "fresh realizations are not bitwise equal")
    aggregate = canonical_sha({
        "records": runs[0]["records"],
        "run_digests": [run["records_digest"] for run in runs],
        "repeat_bitwise_equal": True,
    })
    require(report["aggregate_digest"] == aggregate, "aggregate digest mismatch")

    keys(success, SUCCESS_KEYS, "success")
    require(success == {
        "schema": SUCCESS_SCHEMA, "verdict": "PASS",
        "policy_sha256": expected_policy_sha,
        "aggregate_digest": aggregate,
        "report_sha256": sha256_bytes(report_raw),
    }, "_SUCCESS binding mismatch")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-validator-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_artifact(
        artifact_dir=args.artifact_dir, repo=args.repo, config_path=args.config,
        policy_path=args.policy, expected_commit=args.expected_commit,
        expected_config_sha=args.expected_config_sha256,
        expected_policy_sha=args.expected_policy_sha256,
        expected_runner_sha=args.expected_runner_sha256,
        expected_validator_sha=args.expected_validator_sha256,
    )
    print(
        "STAGE512A_CONTROL_RUNTIME_PROMPT_SMOKE_VALID "
        f"{report['aggregate_digest']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"STAGE512A_CONTROL_RUNTIME_PROMPT_SMOKE_INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1)
