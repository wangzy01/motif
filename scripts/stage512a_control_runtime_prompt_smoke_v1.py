#!/usr/bin/env python3
"""Zero-GPU realization gate for the Stage512A continuation prompt path.

The gate intentionally stops before image loading.  It constructs the production
``OfflineExactDataset`` twice, resolves fixed plan coordinates through the real
catalog and caption sidecar, and executes the exact slot-seed/band/final-caption
path used by ``_consume_step``.  A payload entry-point guard makes any accidental
image read a hard failure.

The output is a create-once, two-file, read-only receipt.  It is independently
checked by ``validate_stage512a_control_runtime_prompt_smoke_v1.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


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


class SmokeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_constant(token: str) -> None:
    raise SmokeError(f"non-standard JSON constant: {token}")


def load_strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), f"JSON is not newline terminated: {path}")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=strict_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise SmokeError(f"invalid JSON {path}: {exc}") from exc
    require(type(value) is dict, f"JSON root is not an object: {path}")
    return value, raw


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    require(type(value) is dict, f"{label} is not an object")
    actual = set(value)
    require(
        actual == expected,
        f"{label} key closure: missing={sorted(expected - actual)} "
        f"extra={sorted(actual - expected)}",
    )
    return value


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def safe_relative(value: Any, label: str) -> str:
    require(type(value) is str and value != "", f"{label} must be a nonempty string")
    parsed = PurePosixPath(value)
    require(
        not parsed.is_absolute() and "." not in parsed.parts and ".." not in parsed.parts
        and str(parsed) == value,
        f"{label} must be a canonical repo-relative path",
    )
    return value


def absolute_path(value: Any, label: str) -> str:
    require(type(value) is str and value.startswith("/") and not value.startswith("//"),
            f"{label} must be an absolute path")
    require(str(PurePosixPath(value)) == value, f"{label} is not canonical")
    return value


def sha_value(value: Any, label: str) -> str:
    require(type(value) is str and SHA_RE.fullmatch(value) is not None,
            f"{label} is not a lowercase SHA-256")
    return value


def validate_policy(policy: dict[str, Any]) -> None:
    exact_keys(policy, POLICY_KEYS, "policy")
    require(policy["schema"] == POLICY_SCHEMA, "wrong policy schema")
    require(type(policy["policy_id"]) is str and policy["policy_id"], "empty policy_id")

    config = exact_keys(policy["config"], CONFIG_KEYS, "policy.config")
    safe_relative(config["path"], "policy.config.path")
    sha_value(config["sha256"], "policy.config.sha256")

    execution = exact_keys(policy["execution"], EXECUTION_KEYS, "policy.execution")
    for key in ("rank", "world_size", "image_payload_reads", "fresh_dataset_constructions"):
        require(type(execution[key]) is int, f"policy.execution.{key} is not an integer")
    require(execution["rank"] == 0 and execution["world_size"] == 64,
            "policy must exercise formal rank 0/world 64")
    require(execution["gpic_payload_backend"] == "source_files_v1",
            "smoke must disable the image-only GPIC overlay")
    require(execution["gpic_payload_in_memory"] is False,
            "smoke must disable GPIC image payload reads")
    require(execution["image_payload_reads"] == 0, "policy permits image payload reads")
    require(execution["fresh_dataset_constructions"] == 2,
            "policy must require two fresh constructions")

    assets = exact_keys(policy["assets"], ASSET_KEYS, "policy.assets")
    for key in ("plan_root", "catalog_root", "sampling_config_path",
                "caption_sidecar_root", "anime_freq_root"):
        absolute_path(assets[key], f"policy.assets.{key}")
    for key in ASSET_KEYS - {
        "plan_root", "catalog_root", "sampling_config_path", "caption_sidecar_root",
        "anime_freq_root",
    }:
        sha_value(assets[key], f"policy.assets.{key}")

    runtime = policy["runtime_files"]
    require(type(runtime) is dict and runtime, "policy.runtime_files must be a nonempty object")
    for name, digest in runtime.items():
        safe_relative(name, f"policy.runtime_files[{name!r}]")
        sha_value(digest, f"policy.runtime_files[{name!r}]")

    cases = policy["cases"]
    require(type(cases) is list and len(cases) == 3, "policy must have exactly three cases")
    seen_ids: set[str] = set()
    seen_coords: set[tuple[int, int, int]] = set()
    domains: set[str] = set()
    for index, raw_case in enumerate(cases):
        case = exact_keys(raw_case, CASE_KEYS, f"policy.cases[{index}]")
        require(type(case["case_id"]) is str and case["case_id"], "invalid case_id")
        require(case["case_id"] not in seen_ids, "duplicate case_id")
        seen_ids.add(case["case_id"])
        for key in ("step", "rank", "slot", "row_id", "bucket_idx",
                    "planned_caption_band", "realized_caption_band", "prompt_utf8_length"):
            require(type(case[key]) is int, f"case {case['case_id']} {key} is not an integer")
        coord = (case["step"], case["rank"], case["slot"])
        require(coord not in seen_coords, f"duplicate coordinate: {coord}")
        seen_coords.add(coord)
        require(case["rank"] == execution["rank"], "case rank differs from execution rank")
        require(type(case["uid"]) is str and case["uid"], "invalid uid")
        require(case["domain"] in {"anime", "textatlas", "gpic"}, "invalid case domain")
        domains.add(case["domain"])
        require(
            type(case["bucket_hw"]) is list and len(case["bucket_hw"]) == 2
            and all(type(x) is int and x > 0 for x in case["bucket_hw"]),
            "invalid bucket_hw",
        )
        require(type(case["cfg_protected"]) is bool, "cfg_protected is not boolean")
        require(case["prompt_utf8_length"] > 0, "empty expected prompt")
        sha_value(case["prompt_sha256"], "case prompt_sha256")
    require(domains == {"anime", "textatlas", "gpic"}, "domain coverage is incomplete")


def git_output(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise SmokeError(f"git {' '.join(args)} failed: {exc.output.strip()}") from exc


def verify_regular_file(path: Path, label: str) -> None:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink(),
            f"{label} is not an ordinary non-symlink file: {path}")


def verify_release(
    repo: Path,
    policy_path: Path,
    policy_raw: bytes,
    config_path: Path,
    expected_commit: str,
    expected_policy_sha: str,
    expected_config_sha: str,
    expected_runner_sha: str,
    expected_validator_sha: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    require(COMMIT_RE.fullmatch(expected_commit) is not None, "expected commit is malformed")
    for label, value in (
        ("policy", expected_policy_sha), ("config", expected_config_sha),
        ("runner", expected_runner_sha), ("validator", expected_validator_sha),
    ):
        sha_value(value, f"expected {label} SHA")

    require(repo.is_absolute() and repo.resolve(strict=True) == repo,
            "repo must be a canonical absolute directory")
    require(git_output(repo, "rev-parse", "--is-inside-work-tree") == "true",
            "repo is not a Git worktree")
    require(git_output(repo, "rev-parse", "HEAD") == expected_commit, "Git HEAD drift")
    require(git_output(repo, "status", "--porcelain") == "", "release worktree is dirty")

    runner_path = Path(__file__).resolve(strict=True)
    validator_path = runner_path.with_name(
        "validate_stage512a_control_runtime_prompt_smoke_v1.py"
    ).resolve(strict=True)
    for path, label in ((runner_path, "runner"), (validator_path, "validator"),
                        (policy_path, "policy"), (config_path, "config")):
        verify_regular_file(path, label)
        require(path == repo / path.relative_to(repo), f"{label} escapes release repo")
    actual_self = {
        "scripts/stage512a_control_runtime_prompt_smoke_v1.py": sha256_file(runner_path),
        "scripts/validate_stage512a_control_runtime_prompt_smoke_v1.py": sha256_file(validator_path),
        str(policy_path.relative_to(repo)): sha256_bytes(policy_raw),
    }
    require(actual_self["scripts/stage512a_control_runtime_prompt_smoke_v1.py"] == expected_runner_sha,
            "runner SHA drift")
    require(actual_self["scripts/validate_stage512a_control_runtime_prompt_smoke_v1.py"] == expected_validator_sha,
            "validator SHA drift")
    require(actual_self[str(policy_path.relative_to(repo))] == expected_policy_sha,
            "policy SHA drift")
    actual_config_sha = sha256_file(config_path)
    require(actual_config_sha == expected_config_sha == policy["config"]["sha256"],
            "config SHA drift")

    runtime_actual: dict[str, str] = {}
    for name, expected in policy["runtime_files"].items():
        path = repo / name
        verify_regular_file(path, f"runtime file {name}")
        require(git_output(repo, "ls-files", "--error-unmatch", name) == name,
                f"runtime file is not committed: {name}")
        actual = sha256_file(path)
        require(actual == expected, f"runtime file SHA drift: {name}")
        runtime_actual[name] = actual
    return {
        "repo": str(repo),
        "git_commit": expected_commit,
        "config": {"path": str(config_path.relative_to(repo)), "sha256": actual_config_sha},
        "self_files": actual_self,
        "runtime_files": runtime_actual,
    }


def verify_assets(policy: Mapping[str, Any]) -> dict[str, Any]:
    assets = dict(policy["assets"])
    files = {
        "plan_header_sha256": Path(assets["plan_root"]) / "header.json",
        "plan_checksums_sha256": Path(assets["plan_root"]) / "checksums.json",
        "catalog_header_sha256": Path(assets["catalog_root"]) / "header.json",
        "sampling_config_sha256": Path(assets["sampling_config_path"]),
        "caption_sidecar_header_sha256": Path(assets["caption_sidecar_root"]) / "header.json",
        "anime_freq_header_sha256": Path(assets["anime_freq_root"]) / "header.json",
    }
    for key, path in files.items():
        verify_regular_file(path, f"asset {key}")
        require(sha256_file(path) == assets[key], f"asset SHA drift: {path}")
    return assets


def verify_config_bindings(cfg: Any, policy: Mapping[str, Any]) -> dict[str, Any]:
    dataset = cfg.dataset
    assets = policy["assets"]
    expected = {
        "offline_plan_root": assets["plan_root"],
        "offline_plan_hash": assets["plan_hash"],
        "offline_catalog_root": assets["catalog_root"],
        "sampling_config": assets["sampling_config_path"],
        "offline_caption_sidecar_root": assets["caption_sidecar_root"],
        "offline_anime_freq_root": assets["anime_freq_root"],
    }
    actual = {key: str(dataset[key]) for key in expected}
    require(actual == expected, f"config asset bindings drift: {actual!r}")
    require(str(dataset.stage) == "512a", "config stage is not 512a")
    require(int(cfg.trainer.batch_size) == 32, "config per-rank batch is not 32")
    return {"stage": "512a", "per_rank_batch": 32, **actual}


def realize_once(cfg: Any, policy: Mapping[str, Any], construction_index: int) -> dict[str, Any]:
    from omegaconf import OmegaConf
    from training.dataline_offline.offline_dataset import OfflineExactDataset

    kwargs = dict(OmegaConf.to_container(cfg.dataset, resolve=True))
    require(kwargs.pop("store_cls", None)
            == "training.dataline_offline.offline_dataset.OfflineExactDataset",
            "formal config store_cls drift")
    execution = policy["execution"]
    kwargs["gpic_payload_backend"] = execution["gpic_payload_backend"]
    kwargs["gpic_payload_in_memory"] = execution["gpic_payload_in_memory"]
    kwargs.pop("offline_gpic_blob_root", None)
    kwargs.pop("gpic_blob_manifest_sha256", None)

    dataset = OfflineExactDataset(
        rank=int(execution["rank"]), world_size=int(execution["world_size"]), **kwargs
    )
    payload_calls: list[str] = []

    def payload_forbidden(*_args: Any, **_kwargs: Any) -> Any:
        payload_calls.append("_image_payloads")
        raise SmokeError("image payload access is forbidden in the prompt realization smoke")

    dataset._image_payloads = payload_forbidden
    try:
        require(dataset._gpic_blob is None, "image-only GPIC blob overlay was initialized")
        require(dataset._pipeline_active is False, "dataset pipeline started during construction")
        records: list[dict[str, Any]] = []
        planned_cache: dict[int, list[Any]] = {}
        for expected in policy["cases"]:
            step = int(expected["step"])
            if step not in planned_cache:
                _bucket, planned_cache[step] = dataset._plan_items_for_step(step)
            items = planned_cache[step]
            slot = int(expected["slot"])
            require(0 <= slot < len(items), f"case slot outside plan batch: {expected['case_id']}")
            item = items[slot]
            planned_identity = {
                "step": item.step,
                "rank": dataset.rank,
                "slot": item.slot,
                "row_id": item.row_id,
                "domain": item.domain,
                "bucket_idx": item.bucket_idx,
                "bucket_hw": list(dataset.buckets[item.bucket_idx]),
                "planned_caption_band": item.caption_band,
            }
            for key, actual in planned_identity.items():
                require(actual == expected[key],
                        f"planned identity drift {expected['case_id']} {key}: {actual!r}")

            catalog = dataset._catalog_readers[item.domain].get_many([item.row_id])[item.row_id]
            require(str(catalog["uid"]) == expected["uid"],
                    f"catalog uid drift: {expected['case_id']}")
            caption = dataset._caption_records(item.domain, [(item, catalog)])[
                (item.step, item.slot)
            ]
            rng = random.Random(
                dataset._slot_seed(dataset.plan_id, item.step, dataset.rank, item.slot)
            )
            band = dataset._validate_planned_band(
                item.domain, caption, item.caption_band
            )
            prompt = dataset._finalize_caption(item.domain, caption, band, rng)
            require(type(prompt) is str, f"prompt is not text: {expected['case_id']}")
            raw = prompt.encode("utf-8")
            record = {
                "case_id": expected["case_id"],
                **planned_identity,
                "uid": str(catalog["uid"]),
                "realized_caption_band": int(band),
                "cfg_protected": bool(dataset._is_cfg_protected(item.domain, caption)),
                "prompt_utf8_hex": raw.hex(),
                "prompt_utf8_length": len(raw),
                "prompt_sha256": sha256_bytes(raw),
            }
            for key in CASE_KEYS:
                require(record[key] == expected[key],
                        f"realized value drift {expected['case_id']} {key}: {record[key]!r}")
            records.append(record)
        require(dataset._pipeline_active is False, "dataset image pipeline started")
        require(payload_calls == [], "image payload guard fired")
        return {
            "construction_index": construction_index,
            "records": records,
            "records_digest": canonical_sha(records),
            "image_payload_read_calls": len(payload_calls),
            "pipeline_started": False,
            "gpic_blob_initialized": False,
        }
    finally:
        dataset.close()


def atomic_write(path: Path, raw: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def publish(output_dir: Path, report: dict[str, Any], success: dict[str, Any]) -> None:
    require(output_dir.is_absolute(), "output directory must be absolute")
    output_dir.mkdir(mode=0o755)
    report_raw = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    success["report_sha256"] = sha256_bytes(report_raw)
    success_raw = json.dumps(
        success, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    atomic_write(output_dir / "REPORT.json", report_raw)
    os.chmod(output_dir / "REPORT.json", 0o444)
    atomic_write(output_dir / "_SUCCESS", success_raw)
    os.chmod(output_dir / "_SUCCESS", 0o444)
    directory_fd = os.open(output_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.chmod(output_dir, 0o555)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-validator-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    policy_path = args.policy.resolve(strict=True)
    require(not args.output_dir.exists(), "output directory already exists")

    policy, policy_raw = load_strict_json(policy_path)
    validate_policy(policy)
    require(str(config_path.relative_to(repo)) == policy["config"]["path"],
            "CLI config differs from policy config")
    release = verify_release(
        repo, policy_path, policy_raw, config_path, args.expected_commit,
        args.expected_policy_sha256, args.expected_config_sha256,
        args.expected_runner_sha256, args.expected_validator_sha256, policy,
    )
    assets = verify_assets(policy)

    sys.path.insert(0, str(repo))
    from omegaconf import OmegaConf
    import pyarrow
    import torch
    import tokenizers
    import transformers

    cfg = OmegaConf.load(config_path)
    config_bindings = verify_config_bindings(cfg, policy)

    os.environ.pop("OFFLINE_ARB_IO_TELEMETRY_DIR", None)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    runs = [realize_once(cfg, policy, index) for index in range(2)]
    repeat_equal = runs[0]["records"] == runs[1]["records"]
    require(repeat_equal, "fresh dataset constructions realized different records")
    require(runs[0]["records_digest"] == runs[1]["records_digest"],
            "fresh realization digests differ")
    aggregate_digest = canonical_sha({
        "records": runs[0]["records"],
        "run_digests": [run["records_digest"] for run in runs],
        "repeat_bitwise_equal": repeat_equal,
    })
    report = {
        "schema": REPORT_SCHEMA,
        "verdict": "PASS",
        "pass": True,
        "policy": {
            "path": str(policy_path.relative_to(repo)),
            "sha256": args.expected_policy_sha256,
            "policy_id": policy["policy_id"],
        },
        "release": release,
        "assets": assets,
        "config_bindings": config_bindings,
        "execution": dict(policy["execution"]),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pyarrow": pyarrow.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "tokenizer_path": str(cfg.dataset.tokenizer_path),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "runs": runs,
        "repeat_bitwise_equal": repeat_equal,
        "aggregate_digest": aggregate_digest,
    }
    success = {
        "schema": SUCCESS_SCHEMA,
        "verdict": "PASS",
        "policy_sha256": args.expected_policy_sha256,
        "aggregate_digest": aggregate_digest,
        "report_sha256": "",
    }
    publish(args.output_dir, report, success)
    print(f"STAGE512A_CONTROL_RUNTIME_PROMPT_SMOKE_PASS {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeError as exc:
        print(f"STAGE512A_CONTROL_RUNTIME_PROMPT_SMOKE_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
