#!/usr/bin/env python3
"""Positive/negative contract tests for the runtime-prompt receipt validator."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_stage512a_control_runtime_prompt_smoke_v1 as validator  # noqa: E402


def encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


class Fixture:
    def __init__(self, base: Path):
        self.base = base
        self.repo = base / "repo"
        self.assets_root = base / "assets"
        self.artifact = base / "artifact"
        self.repo.mkdir()
        # macOS exposes /var through /private/var; the production contract records
        # the canonical realpath, so the fixture must do the same.
        self.repo = self.repo.resolve()
        self.assets_root.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "smoke@example.invalid")
        git(self.repo, "config", "user.name", "Smoke Fixture")

        self.config = self.repo / "configs/stage512a_continue_control_rngfix_v1.yaml"
        self.runner = self.repo / "scripts/stage512a_control_runtime_prompt_smoke_v1.py"
        self.validator = self.repo / "scripts/validate_stage512a_control_runtime_prompt_smoke_v1.py"
        self.runtime_file = self.repo / "training/runtime.py"
        for path in (self.config, self.runner, self.validator, self.runtime_file):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.config.write_bytes(b'dataset:\n  stage: "512a"\n')
        self.runner.write_bytes(b"# fixture runner\n")
        self.validator.write_bytes(b"# fixture validator\n")
        self.runtime_file.write_bytes(b"# fixture runtime\n")

        plan = self.assets_root / "plan"
        catalog = self.assets_root / "catalog"
        sidecar = self.assets_root / "sidecar"
        freq = self.assets_root / "freq"
        for path in (plan, catalog, sidecar, freq):
            path.mkdir()
        plan_hash = "1" * 64
        catalog_hash = "2" * 64
        (plan / "header.json").write_bytes(encoded({"plan_hash": plan_hash}))
        (plan / "checksums.json").write_bytes(encoded({"blocks": []}))
        (catalog / "header.json").write_bytes(encoded({"catalog_hash": catalog_hash}))
        sampling = self.assets_root / "sampling.json"
        sampling.write_bytes(encoded({"fixture": True}))
        (sidecar / "header.json").write_bytes(encoded({"fixture": "sidecar"}))
        (freq / "header.json").write_bytes(encoded({"fixture": "freq"}))

        assets = {
            "plan_root": str(plan),
            "plan_hash": plan_hash,
            "plan_header_sha256": sha_file(plan / "header.json"),
            "plan_checksums_sha256": sha_file(plan / "checksums.json"),
            "catalog_root": str(catalog),
            "catalog_hash": catalog_hash,
            "catalog_header_sha256": sha_file(catalog / "header.json"),
            "sampling_config_path": str(sampling),
            "sampling_config_sha256": sha_file(sampling),
            "caption_sidecar_root": str(sidecar),
            "caption_sidecar_header_sha256": sha_file(sidecar / "header.json"),
            "anime_freq_root": str(freq),
            "anime_freq_header_sha256": sha_file(freq / "header.json"),
        }
        prompts = [b"anime prompt", b"gpic prompt", "文字 prompt".encode()]
        domains = ["anime", "gpic", "textatlas"]
        slots = [0, 12, 25]
        rows = [11, 22, 33]
        bands = [-1, 3, -1]
        cfg = [True, False, False]
        self.cases: list[dict[str, Any]] = []
        for index, (prompt, domain, slot, row, band, keep) in enumerate(
            zip(prompts, domains, slots, rows, bands, cfg)
        ):
            self.cases.append({
                "case_id": f"{domain}-case-{index}",
                "step": 0,
                "rank": 0,
                "slot": slot,
                "row_id": row,
                "uid": f"{domain}:uid-{index}",
                "domain": domain,
                "bucket_idx": 182,
                "bucket_hw": [512, 512],
                "planned_caption_band": band,
                "realized_caption_band": band,
                "cfg_protected": keep,
                "prompt_utf8_length": len(prompt),
                "prompt_sha256": sha_bytes(prompt),
            })
        self.prompt_bytes = prompts
        policy = {
            "schema": validator.POLICY_SCHEMA,
            "policy_id": "fixture-policy",
            "config": {
                "path": str(self.config.relative_to(self.repo)),
                "sha256": sha_file(self.config),
            },
            "execution": {
                "rank": 0,
                "world_size": 64,
                "gpic_payload_backend": "source_files_v1",
                "gpic_payload_in_memory": False,
                "image_payload_reads": 0,
                "fresh_dataset_constructions": 2,
            },
            "assets": assets,
            "runtime_files": {
                str(self.runtime_file.relative_to(self.repo)): sha_file(self.runtime_file)
            },
            "cases": self.cases,
        }
        self.policy = self.repo / "configs/stage512a_control_runtime_prompt_policy_v1.json"
        self.policy.write_bytes(encoded(policy))
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-qm", "fixture")
        self.commit = git(self.repo, "rev-parse", "HEAD")
        self.config_sha = sha_file(self.config)
        self.policy_sha = sha_file(self.policy)
        self.runner_sha = sha_file(self.runner)
        self.validator_sha = sha_file(self.validator)

        records = []
        for case, prompt in zip(self.cases, prompts):
            records.append({**case, "prompt_utf8_hex": prompt.hex()})
        runs = [
            {
                "construction_index": index,
                "records": json.loads(json.dumps(records)),
                "records_digest": validator.canonical_sha(records),
                "image_payload_read_calls": 0,
                "pipeline_started": False,
                "gpic_blob_initialized": False,
            }
            for index in range(2)
        ]
        aggregate = validator.canonical_sha({
            "records": runs[0]["records"],
            "run_digests": [run["records_digest"] for run in runs],
            "repeat_bitwise_equal": True,
        })
        self.report = {
            "schema": validator.REPORT_SCHEMA,
            "verdict": "PASS",
            "pass": True,
            "policy": {
                "path": str(self.policy.relative_to(self.repo)),
                "sha256": self.policy_sha,
                "policy_id": "fixture-policy",
            },
            "release": {
                "repo": str(self.repo),
                "git_commit": self.commit,
                "config": policy["config"],
                "self_files": {
                    str(self.runner.relative_to(self.repo)): self.runner_sha,
                    str(self.validator.relative_to(self.repo)): self.validator_sha,
                    str(self.policy.relative_to(self.repo)): self.policy_sha,
                },
                "runtime_files": policy["runtime_files"],
            },
            "assets": assets,
            "config_bindings": {
                "stage": "512a",
                "per_rank_batch": 32,
                "offline_plan_root": assets["plan_root"],
                "offline_plan_hash": assets["plan_hash"],
                "offline_catalog_root": assets["catalog_root"],
                "sampling_config": assets["sampling_config_path"],
                "offline_caption_sidecar_root": assets["caption_sidecar_root"],
                "offline_anime_freq_root": assets["anime_freq_root"],
            },
            "execution": policy["execution"],
            "runtime": {
                "python": "3.fixture",
                "torch": "fixture",
                "pyarrow": "fixture",
                "transformers": "fixture",
                "tokenizers": "fixture",
                "tokenizer_path": "/fixture/tokenizer",
                "cuda_visible_devices": "",
            },
            "runs": runs,
            "repeat_bitwise_equal": True,
            "aggregate_digest": aggregate,
        }
        self.artifact.mkdir()
        self.reseal()

    def reseal(self) -> None:
        for run in self.report["runs"]:
            run["records_digest"] = validator.canonical_sha(run["records"])
        self.report["aggregate_digest"] = validator.canonical_sha({
            "records": self.report["runs"][0]["records"],
            "run_digests": [run["records_digest"] for run in self.report["runs"]],
            "repeat_bitwise_equal": self.report["repeat_bitwise_equal"],
        })
        report_raw = encoded(self.report)
        success = {
            "schema": validator.SUCCESS_SCHEMA,
            "verdict": "PASS",
            "policy_sha256": self.policy_sha,
            "aggregate_digest": self.report["aggregate_digest"],
            "report_sha256": sha_bytes(report_raw),
        }
        os.chmod(self.artifact, 0o755)
        for name in ("REPORT.json", "_SUCCESS"):
            path = self.artifact / name
            if path.exists():
                os.chmod(path, 0o644)
        (self.artifact / "REPORT.json").write_bytes(report_raw)
        (self.artifact / "_SUCCESS").write_bytes(encoded(success))
        os.chmod(self.artifact / "REPORT.json", 0o444)
        os.chmod(self.artifact / "_SUCCESS", 0o444)
        os.chmod(self.artifact, 0o555)

    def validate(self) -> None:
        validator.validate_artifact(
            artifact_dir=self.artifact,
            repo=self.repo,
            config_path=self.config,
            policy_path=self.policy,
            expected_commit=self.commit,
            expected_config_sha=self.config_sha,
            expected_policy_sha=self.policy_sha,
            expected_runner_sha=self.runner_sha,
            expected_validator_sha=self.validator_sha,
        )


Mutation = Callable[[Fixture], None]


def mutate_record(field: str, value: Any, run: int = 0, record: int = 0) -> Mutation:
    def change(fixture: Fixture) -> None:
        fixture.report["runs"][run]["records"][record][field] = value
        fixture.reseal()
    return change


def main() -> int:
    checks = 0
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="s512a_prompt_validator_positive.") as raw:
        fixture = Fixture(Path(raw))
        fixture.validate()
        checks += 1
        print("PASS positive sealed fixture")

    def duplicate_report_key(fixture: Fixture) -> None:
        path = fixture.artifact / "REPORT.json"
        raw = path.read_bytes().replace(
            b'{\n  "aggregate_digest"',
            b'{\n  "schema": "duplicate",\n  "aggregate_digest"',
            1,
        )
        os.chmod(path, 0o644)
        path.write_bytes(raw)
        os.chmod(path, 0o444)

    def code_sha_drift(fixture: Fixture) -> None:
        fixture.runtime_file.write_bytes(b"# drift\n")

    def config_sha_drift(fixture: Fixture) -> None:
        fixture.config.write_bytes(b"# drift\n")

    def plan_sha_drift(fixture: Fixture) -> None:
        (Path(fixture.report["assets"]["plan_root"]) / "header.json").write_bytes(
            encoded({"plan_hash": "9" * 64})
        )

    def repeat_mismatch(fixture: Fixture) -> None:
        fixture.report["runs"][1]["records"][0]["prompt_utf8_hex"] = b"different".hex()
        fixture.report["runs"][1]["records"][0]["prompt_utf8_length"] = len(b"different")
        fixture.report["runs"][1]["records"][0]["prompt_sha256"] = sha_bytes(b"different")
        fixture.reseal()

    def missing_field(fixture: Fixture) -> None:
        del fixture.report["runtime"]
        fixture.reseal()

    def extra_field(fixture: Fixture) -> None:
        fixture.report["unexpected"] = True
        fixture.reseal()

    def success_mismatch(fixture: Fixture) -> None:
        path = fixture.artifact / "_SUCCESS"
        value = json.loads(path.read_text())
        value["report_sha256"] = "0" * 64
        os.chmod(path, 0o644)
        path.write_bytes(encoded(value))
        os.chmod(path, 0o444)

    def extra_member(fixture: Fixture) -> None:
        os.chmod(fixture.artifact, 0o755)
        (fixture.artifact / "EXTRA").write_text("x")
        os.chmod(fixture.artifact, 0o555)

    negatives: list[tuple[str, Mutation]] = [
        ("wrong fixed coordinate", mutate_record("step", 1)),
        ("wrong row", mutate_record("row_id", 999)),
        ("wrong bucket", mutate_record("bucket_idx", 181)),
        ("wrong caption band", mutate_record("realized_caption_band", 2, record=1)),
        ("wrong prompt SHA", mutate_record("prompt_sha256", "0" * 64)),
        ("two realizations differ", repeat_mismatch),
        ("runtime code SHA drift", code_sha_drift),
        ("config SHA drift", config_sha_drift),
        ("plan SHA drift", plan_sha_drift),
        ("duplicate JSON key", duplicate_report_key),
        ("missing report field", missing_field),
        ("extra report field", extra_field),
        ("success marker mismatch", success_mismatch),
        ("extra artifact member", extra_member),
    ]
    for label, mutation in negatives:
        checks += 1
        with tempfile.TemporaryDirectory(prefix="s512a_prompt_validator_negative.") as raw:
            fixture = Fixture(Path(raw))
            mutation(fixture)
            try:
                fixture.validate()
            except validator.ValidationError as exc:
                print(f"CAUGHT {label}: {str(exc)[:100]}")
            else:
                failures.append(label)
                print(f"MISSED {label}")

    print(f"{checks} checks, {len(failures)} failures")
    if failures:
        for label in failures:
            print(f"  {label}")
        return 1
    print("STAGE512A_CONTROL_RUNTIME_PROMPT_SMOKE_TESTS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
