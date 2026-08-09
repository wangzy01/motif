#!/usr/bin/env python3
"""CPU-only positive and adversarial tests for the smoke-v3 source contract."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import re
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent if (HERE.parent / "training").is_dir() else HERE
JOB_TEMPLATE = REPO_ROOT / (
    "provenance/jcc_jobs/20260809_stage512a_control_a_final_load_smoke_v3.yaml"
)
if not JOB_TEMPLATE.is_file():
    JOB_TEMPLATE = HERE / "stage512a_control_a_final_load_smoke_v3.yaml"
RECEIPT_SCHEMA = REPO_ROOT / (
    "provenance/stage512a_control_a_final_load_smoke_v3.schema.json"
)
if not RECEIPT_SCHEMA.is_file():
    RECEIPT_SCHEMA = HERE / "stage512a_control_a_final_load_smoke_v3.schema.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_probe_source_contract(path: Path):
    """Load the stdlib-only source-map slice without importing PyTorch.

    These tests must remain runnable on a CPU control host that has no torch.
    Selecting the definitions from the real probe AST still tests the shipped
    implementation instead of maintaining a duplicate test implementation.
    """

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    assignment_names = {
        "FINAL_CORE_GIT_BLOB_SOURCE_ARGS",
        "EXPECTED_FINAL_GIT_BLOB_ENTRY_COUNT",
        "A_FINAL_WEIGHTS_ONLY_DIAGNOSTIC_RECEIPT",
        "A_FINAL_WEIGHTS_ONLY_DIAGNOSTIC_RECEIPT_SHA256",
    }
    function_names = {
        "sha256_file",
        "require_sha256",
        "git_blob_raw_sha256",
        "verify_final_git_blob_source_map",
    }
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if names & assignment_names:
                selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in function_names:
                selected.append(node)
    namespace: dict[str, Any] = {
        "hashlib": hashlib,
        "subprocess": subprocess,
        "Path": Path,
        "Mapping": Mapping,
    }
    compiled = compile(
        ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
        str(path),
        "exec",
    )
    exec(compiled, namespace)
    missing = (assignment_names | function_names) - set(namespace)
    if missing:
        raise RuntimeError(f"probe source-contract extraction drift: {sorted(missing)}")
    return types.SimpleNamespace(**namespace)


PROBE_PATH = HERE / "stage512a_control_a_final_load_smoke_v3.py"
PROBE = load_probe_source_contract(PROBE_PATH)
VALIDATOR = load_module(
    "validate_stage512a_control_a_final_load_smoke_v3",
    HERE / "validate_stage512a_control_a_final_load_smoke_v3.py",
)
RENDERER = load_module(
    "render_stage512a_control_a_final_load_smoke_v3",
    HERE / "render_stage512a_control_a_final_load_smoke_v3.py",
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def make_source_repo(root: Path) -> tuple[str, dict[str, str]]:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    git(root, "config", "user.name", "smoke-v3-test")
    git(root, "config", "user.email", "smoke-v3-test@example.invalid")
    expected: dict[str, str] = {}
    paths = list(PROBE.FINAL_CORE_GIT_BLOB_SOURCE_ARGS) + [
        "configs/stage512a_continue_control_rngfix_v1.yaml",
        "scripts/stage512a_control_a_final_load_smoke_v3.py",
        "scripts/validate_stage512a_control_a_final_load_smoke_v3.py",
        "provenance/jcc_jobs/20260809_stage512a_control_a_final_load_smoke_v3.yaml",
    ]
    for index, path_name in enumerate(paths):
        target = root / path_name
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = f"# source fixture {index}: {path_name}\n".encode("utf-8")
        target.write_bytes(payload)
        expected[path_name] = hashlib.sha256(payload).hexdigest()
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "fixture")
    return git(root, "rev-parse", "HEAD"), expected


def canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class MiniSchemaError(AssertionError):
    """Failure from the independent fixed-keyword JSON Schema evaluator."""


def json_exact(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return set(actual) == set(expected) and all(
            json_exact(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            json_exact(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def resolve_local_ref(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise MiniSchemaError(f"unsupported non-local schema ref: {reference}")
    current: Any = root
    for encoded in reference[2:].split("/"):
        key = encoded.replace("~1", "/").replace("~0", "~")
        if type(current) is not dict or key not in current:
            raise MiniSchemaError(f"unresolved schema ref: {reference}")
        current = current[key]
    if type(current) is not dict:
        raise MiniSchemaError(f"schema ref is not an object: {reference}")
    return current


def validate_fixed_schema(
    instance: Any,
    schema: dict[str, Any],
    *,
    root: dict[str, Any] | None = None,
    where: str = "$",
) -> None:
    """Evaluate every JSON Schema keyword used by the checked-in receipt schema.

    This deliberately does not call the hand-written validator and introduces no
    jsonschema runtime dependency.  Unknown schema keywords that carry metadata
    are ignored; all assertion-bearing keywords in this fixed schema are handled.
    """

    root = schema if root is None else root
    if "$ref" in schema:
        validate_fixed_schema(
            instance, resolve_local_ref(root, schema["$ref"]), root=root, where=where
        )
    expected_type = schema.get("type")
    if expected_type is not None:
        type_ok = {
            "object": type(instance) is dict,
            "string": type(instance) is str,
            "array": type(instance) is list,
            "integer": type(instance) is int,
            "number": type(instance) in (int, float),
            "boolean": type(instance) is bool,
            "null": instance is None,
        }.get(expected_type)
        if type_ok is None:
            raise MiniSchemaError(f"{where}: unsupported schema type {expected_type}")
        if not type_ok:
            raise MiniSchemaError(f"{where}: expected {expected_type}")
    if "const" in schema and not json_exact(instance, schema["const"]):
        raise MiniSchemaError(f"{where}: const mismatch")
    if "pattern" in schema:
        if type(instance) is not str or re.search(schema["pattern"], instance) is None:
            raise MiniSchemaError(f"{where}: pattern mismatch")
    if "not" in schema:
        try:
            validate_fixed_schema(instance, schema["not"], root=root, where=where)
        except MiniSchemaError:
            pass
        else:
            raise MiniSchemaError(f"{where}: forbidden by not-schema")
    if type(instance) is dict:
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            raise MiniSchemaError(f"{where}: too few properties")
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            raise MiniSchemaError(f"{where}: too many properties")
        required = schema.get("required", [])
        missing = [key for key in required if key not in instance]
        if missing:
            raise MiniSchemaError(f"{where}: missing required {missing}")
        properties = schema.get("properties", {})
        for key, child_schema in properties.items():
            if key in instance:
                validate_fixed_schema(
                    instance[key], child_schema, root=root, where=f"{where}.{key}"
                )
        extras = set(instance) - set(properties)
        additional = schema.get("additionalProperties", {})
        if additional is False and extras:
            raise MiniSchemaError(f"{where}: unknown properties {sorted(extras)}")
        if type(additional) is dict:
            for key in extras:
                validate_fixed_schema(
                    instance[key], additional, root=root, where=f"{where}.{key}"
                )


def schema_at(
    root: dict[str, Any], schema: dict[str, Any]
) -> dict[str, Any]:
    return resolve_local_ref(root, schema["$ref"]) if "$ref" in schema else schema


def required_field_paths(
    root: dict[str, Any],
    schema: dict[str, Any],
    instance: Any,
    prefix: tuple[str, ...] = (),
) -> list[tuple[str, ...]]:
    schema = schema_at(root, schema)
    result: list[tuple[str, ...]] = []
    if type(instance) is not dict:
        return result
    for key in schema.get("required", []):
        result.append(prefix + (key,))
    properties = schema.get("properties", {})
    for key, child_schema in properties.items():
        if key in instance:
            result.extend(
                required_field_paths(
                    root, child_schema, instance[key], prefix + (key,)
                )
            )
    return result


def sha_field_paths(
    root: dict[str, Any],
    schema: dict[str, Any],
    instance: Any,
    prefix: tuple[str, ...] = (),
) -> list[tuple[str, ...]]:
    schema = schema_at(root, schema)
    if (
        type(instance) is str
        and schema.get("pattern") == "^[0-9a-f]{64}$"
    ):
        return [prefix]
    if type(instance) is not dict:
        return []
    result: list[tuple[str, ...]] = []
    properties = schema.get("properties", {})
    additional = schema.get("additionalProperties")
    for key, value in instance.items():
        child_schema = properties.get(key)
        if child_schema is None and type(additional) is dict:
            child_schema = additional
        if child_schema is not None:
            result.extend(
                sha_field_paths(root, child_schema, value, prefix + (key,))
            )
    return result


def object_at(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        current = current[key]
    return current


def make_render_fixture(root: Path) -> dict[str, Any]:
    # macOS exposes TemporaryDirectory through /var while its canonical path is
    # /private/var.  The renderer intentionally rejects non-canonical release
    # roots, so canonicalize the fixture root before deriving every bound path.
    root = root.resolve()
    repo = root / "release"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    git(repo, "config", "user.name", "smoke-v3-test")
    git(repo, "config", "user.email", "smoke-v3-test@example.invalid")
    for index, relative in enumerate(RENDERER.DEDICATED_RELS):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == RENDERER.TEMPLATE_REL:
            raw = JOB_TEMPLATE.read_bytes()
        elif relative == RENDERER.RECEIPT_SCHEMA_REL:
            raw = RECEIPT_SCHEMA.read_bytes()
        else:
            raw = f"# dedicated fixture {index}: {relative}\n".encode("utf-8")
        target.write_bytes(raw)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "render fixture")
    commit = git(repo, "rev-parse", "HEAD")
    config_sha = hashlib.sha256((repo / RENDERER.CONFIG_REL).read_bytes()).hexdigest()

    authority = root / "authority"
    authority.mkdir()
    lineage_path = authority / "TRAINING_LINEAGE.json"
    lineage_raw = canonical_json(
        {
            "schema": "stage512a_control_training_lineage_v1",
            "status": "PASS",
            "run_id": "stage512a_continue_control_rngfix_v1",
            "release": {
                "repo": str(repo),
                "commit": commit,
                "config_path": str(repo / RENDERER.CONFIG_REL),
                "config_sha256": config_sha,
            },
        }
    )
    lineage_path.write_bytes(lineage_raw)
    lineage_sha = hashlib.sha256(lineage_raw).hexdigest()
    manifest_path = authority / "PHASE_A.json"
    manifest_raw = canonical_json(
        {
            "schema": "stage512a_control_phase_a_release_v1",
            "status": "PASS",
            "release_repo": str(repo),
            "commit": commit,
            "tree": "synthetic",
            "config": {
                "path": str(repo / RENDERER.CONFIG_REL),
                "sha256": config_sha,
            },
            "evaluation_policy": {},
            "template": {},
            "training_lineage": {
                "path": str(lineage_path),
                "sha256": lineage_sha,
            },
        }
    )
    manifest_path.write_bytes(manifest_raw)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    (authority / "_PHASE_A_SUCCESS").write_text(
        manifest_sha + "\n", encoding="ascii"
    )

    resolved = root / "resolved.yaml"
    smoke_output = root / "smoke-output"
    receipt_dir = root / "render-receipt"
    render_args = types.SimpleNamespace(
        release_repo=repo,
        expected_commit=commit,
        phase_a_manifest=manifest_path,
        expected_phase_a_manifest_sha256=manifest_sha,
        training_lineage=lineage_path,
        expected_training_lineage_sha256=lineage_sha,
        smoke_output=smoke_output,
        resolved_spec_output=resolved,
        render_receipt_output=receipt_dir,
    )
    returned = RENDERER.render(render_args)
    report_raw = (receipt_dir / "REPORT.json").read_bytes()
    report = json.loads(report_raw)
    validator_args = types.SimpleNamespace(
        expected_release_repo=str(repo),
        expected_commit=commit,
        expected_phase_a_manifest_path=str(manifest_path),
        expected_phase_a_manifest_sha256=manifest_sha,
        expected_training_lineage_path=str(lineage_path),
        expected_training_lineage_sha256=lineage_sha,
        expected_resolved_spec_path=str(resolved),
        expected_smoke_output_path=str(smoke_output),
    )
    return {
        "repo": repo,
        "receipt_dir": receipt_dir,
        "report_raw": report_raw,
        "report": report,
        "returned": returned,
        "schema": json.loads(RECEIPT_SCHEMA.read_text(encoding="utf-8")),
        "validator_args": validator_args,
        "resolved": resolved,
    }


class ProbeWeightsOnlyContractTests(unittest.TestCase):
    def test_real_probe_uses_weights_only_mmap_load(self) -> None:
        tree = ast.parse(PROBE_PATH.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "load"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertIsInstance(keywords.get("weights_only"), ast.Constant)
        self.assertIs(keywords["weights_only"].value, True)
        self.assertIsInstance(keywords.get("mmap"), ast.Constant)
        self.assertIs(keywords["mmap"].value, True)
        self.assertIsInstance(keywords.get("map_location"), ast.Constant)
        self.assertEqual(keywords["map_location"].value, "cpu")

    def test_diagnostic_receipt_is_explicitly_non_authoritative_evidence(self) -> None:
        self.assertEqual(
            PROBE.A_FINAL_WEIGHTS_ONLY_DIAGNOSTIC_RECEIPT,
            "/fsx/evan/audits/stage512a_continue_control_rngfix_v1/"
            "a_final_weights_only_probe_v1_20260809/RECEIPT.json",
        )
        self.assertEqual(
            PROBE.A_FINAL_WEIGHTS_ONLY_DIAGNOSTIC_RECEIPT_SHA256,
            "f71538dd18b2bfc27cc23854d3692cbb05cfdfab1bb417305ebbb18399d2616e",
        )
        source = PROBE_PATH.read_text(encoding="utf-8")
        self.assertIn("Diagnostic compatibility evidence only", source)
        self.assertIn("not formal release authority", source)


class ProbeGitBlobSourceMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="smoke-v3-source-")
        self.repo = Path(self.temporary.name) / "repo"
        self.commit, self.expected = make_source_repo(self.repo)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_final_blob_and_worktree_map_passes(self) -> None:
        actual = PROBE.verify_final_git_blob_source_map(
            self.repo, self.commit, self.expected
        )
        self.assertEqual(actual, self.expected)

    def test_each_expected_blob_digest_is_independently_bound(self) -> None:
        for relative_path in self.expected:
            with self.subTest(relative_path=relative_path):
                mutated = dict(self.expected)
                mutated[relative_path] = digest(f"wrong:{relative_path}")
                with self.assertRaisesRegex(
                    RuntimeError, "final Git blob raw SHA drift"
                ):
                    PROBE.verify_final_git_blob_source_map(
                        self.repo, self.commit, mutated
                    )

    def test_map_rejects_missing_and_unknown_paths(self) -> None:
        missing = dict(self.expected)
        missing.pop("networks/vae.py")
        with self.assertRaisesRegex(RuntimeError, "source-map closure mismatch"):
            PROBE.verify_final_git_blob_source_map(self.repo, self.commit, missing)
        unknown = dict(self.expected)
        unknown["training/not_runtime.py"] = digest("unknown")
        with self.assertRaisesRegex(RuntimeError, "source-map closure mismatch"):
            PROBE.verify_final_git_blob_source_map(self.repo, self.commit, unknown)

    def test_runtime_checkout_drift_is_rejected_even_when_blob_matches(self) -> None:
        path = self.repo / "training/checkpoint.py"
        path.write_text("# uncommitted runtime drift\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "checkout bytes differ"):
            PROBE.verify_final_git_blob_source_map(
                self.repo, self.commit, self.expected
            )

    def test_runtime_symlink_is_rejected(self) -> None:
        path = self.repo / "networks/vae.py"
        path.unlink()
        path.symlink_to(self.repo / "train.py")
        with self.assertRaisesRegex(RuntimeError, "ordinary non-symlink"):
            PROBE.verify_final_git_blob_source_map(
                self.repo, self.commit, self.expected
            )


class ValidatorSourceMapTests(unittest.TestCase):
    def setUp(self) -> None:
        core_values = {
            path: digest(f"expected:{path}")
            for path in VALIDATOR.EXPECTED_CORE_SOURCE_KEYS
        }
        fields = {
            attribute: core_values[path]
            for path, attribute in VALIDATOR.EXPECTED_CORE_SOURCE_ARGUMENTS.items()
        }
        fields.update(
            expected_release_repo="/release",
            expected_commit="1" * 40,
            expected_config_path="/release/configs/control.yaml",
            expected_config_sha256=digest("config"),
            expected_probe_path="/release/scripts/probe.py",
            expected_probe_sha256=digest("probe"),
            expected_validator_path="/release/scripts/validator.py",
            expected_validator_sha256=digest("validator"),
            expected_job_template_path="/release/jobs/smoke-v3.yaml",
            expected_job_template_sha256=digest("job-template"),
            expected_phase_a_manifest_path="/release/PHASE_A.json",
            expected_phase_a_manifest_sha256=digest("phase-a"),
            expected_canonical_training_lineage_path="/release/LINEAGE.json",
            expected_canonical_training_lineage_sha256=digest("lineage"),
            expected_checkpoint_path="/checkpoints/alias.pt",
            expected_original_checkpoint_path="/checkpoints/original.pt",
            expected_checkpoint_sha256=VALIDATOR.EXPECTED_CHECKPOINT_SHA256,
            expected_ablation_fingerprint_sha256=(
                VALIDATOR.EXPECTED_ABLATION_FINGERPRINT_SHA256
            ),
            expected_asset_reseal_report_sha256=digest("reseal"),
            expected_report_sha256=None,
        )
        self.args = types.SimpleNamespace(**fields)
        self.authority = {
            "release_repo": self.args.expected_release_repo,
            "final_commit": self.args.expected_commit,
            "git_status_porcelain": [],
            "config_path": self.args.expected_config_path,
            "config_sha256": self.args.expected_config_sha256,
            "probe_path": self.args.expected_probe_path,
            "probe_sha256": self.args.expected_probe_sha256,
            "validator_path": self.args.expected_validator_path,
            "validator_sha256": self.args.expected_validator_sha256,
            "job_template_path": self.args.expected_job_template_path,
            "job_template_sha256": self.args.expected_job_template_sha256,
            "phase_a_manifest_path": self.args.expected_phase_a_manifest_path,
            "phase_a_manifest_sha256": self.args.expected_phase_a_manifest_sha256,
            "canonical_training_lineage_path": (
                self.args.expected_canonical_training_lineage_path
            ),
            "canonical_training_lineage_sha256": (
                self.args.expected_canonical_training_lineage_sha256
            ),
            "canonical_training_lineage_claimed": True,
        }
        self.source_values = VALIDATOR.expected_final_git_blob_raw_sha256(self.args)

    def report(self, source_values: dict[str, str]) -> dict[str, object]:
        return {
            "authority": dict(self.authority),
            "tested_final_git_blob_raw_sha256": source_values,
        }

    def test_validator_accepts_only_the_exact_external_map(self) -> None:
        VALIDATOR.validate_authority(self.report(dict(self.source_values)), self.args)

    def test_validator_rejects_each_independent_value_substitution(self) -> None:
        for relative_path in self.source_values:
            with self.subTest(relative_path=relative_path):
                mutated = dict(self.source_values)
                mutated[relative_path] = digest(f"substitution:{relative_path}")
                with self.assertRaisesRegex(
                    VALIDATOR.ReceiptValidationError,
                    "tested_final_git_blob_raw_sha256",
                ):
                    VALIDATOR.validate_authority(self.report(mutated), self.args)

    def test_validator_rejects_map_keyset_drift(self) -> None:
        missing = dict(self.source_values)
        missing.pop("training/formal_boot.py")
        with self.assertRaisesRegex(
            VALIDATOR.ReceiptValidationError, "key closure mismatch"
        ):
            VALIDATOR.validate_authority(self.report(missing), self.args)
        extra = dict(self.source_values)
        extra["training/extra.py"] = digest("extra")
        with self.assertRaisesRegex(
            VALIDATOR.ReceiptValidationError, "key closure mismatch"
        ):
            VALIDATOR.validate_authority(self.report(extra), self.args)

    def test_cli_rejects_zero_source_digest(self) -> None:
        bad = types.SimpleNamespace(**vars(self.args))
        bad.expected_networks_vae_py_git_blob_raw_sha256 = "0" * 64
        with self.assertRaisesRegex(
            VALIDATOR.ReceiptValidationError, "all-zero SHA-256"
        ):
            VALIDATOR.validate_cli(bad)


class RendererSchemaAgreementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="smoke-v3-render-")
        self.fixture = make_render_fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_both_reject(self, mutated: dict[str, Any]) -> None:
        with self.assertRaises(MiniSchemaError):
            validate_fixed_schema(mutated, self.fixture["schema"])
        with self.assertRaises(VALIDATOR.ReceiptValidationError):
            VALIDATOR.validate_render_receipt(
                mutated, self.fixture["validator_args"]
            )

    def test_renderer_e2e_receipt_passes_schema_and_hand_validator(self) -> None:
        report = self.fixture["report"]
        validate_fixed_schema(report, self.fixture["schema"])
        VALIDATOR.validate_render_receipt(report, self.fixture["validator_args"])
        report_sha = hashlib.sha256(self.fixture["report_raw"]).hexdigest()
        success = (self.fixture["receipt_dir"] / "_SUCCESS").read_text(
            encoding="ascii"
        )
        self.assertEqual(success, report_sha + "\n")
        self.assertEqual(set(report), set(self.fixture["schema"]["required"]))
        self.assertNotIn("render_receipt_sha256", report)
        self.assertEqual(
            self.fixture["returned"]["render_receipt_sha256"], report_sha
        )
        self.assertNotIn(
            b"PLACEHOLDER_", self.fixture["resolved"].read_bytes()
        )

    def test_every_schema_required_field_deletion_is_rejected_by_both(self) -> None:
        report = self.fixture["report"]
        paths = required_field_paths(
            self.fixture["schema"], self.fixture["schema"], report
        )
        self.assertGreater(len(paths), 50)
        for path in paths:
            with self.subTest(path=".".join(path)):
                mutated = copy.deepcopy(report)
                del object_at(mutated, path[:-1])[path[-1]]
                self.assert_both_reject(mutated)

    def test_unknown_fields_at_every_closed_object_are_rejected_by_both(self) -> None:
        report = self.fixture["report"]
        closed_paths = (
            (),
            ("release",),
            ("phase_a",),
            ("artifacts",),
            ("runtime_git_blob_raw_sha256",),
            ("dedicated_asset_git_blob_raw_sha256",),
            ("semantic_contract",),
        )
        for path in closed_paths:
            with self.subTest(path=".".join(path) or "$"):
                mutated = copy.deepcopy(report)
                target = object_at(mutated, path)
                target["unknown_contract_field"] = digest("valid-extra")
                self.assert_both_reject(mutated)

    def test_every_sha_field_rejects_zero_and_malformed_values_in_both(self) -> None:
        report = self.fixture["report"]
        paths = sha_field_paths(
            self.fixture["schema"], self.fixture["schema"], report
        )
        self.assertGreaterEqual(len(paths), 30)
        for path in paths:
            for replacement in ("0" * 64, "not-a-lowercase-sha256"):
                with self.subTest(path=".".join(path), replacement=replacement):
                    mutated = copy.deepcopy(report)
                    object_at(mutated, path[:-1])[path[-1]] = replacement
                    self.assert_both_reject(mutated)

    def test_map_key_replacement_is_rejected_even_at_same_cardinality(self) -> None:
        report = self.fixture["report"]
        for map_name in (
            "runtime_git_blob_raw_sha256",
            "dedicated_asset_git_blob_raw_sha256",
        ):
            with self.subTest(map_name=map_name):
                mutated = copy.deepcopy(report)
                source_map = mutated[map_name]
                removed = sorted(source_map)[0]
                source_map["unknown/replacement.py"] = source_map.pop(removed)
                self.assert_both_reject(mutated)

    def test_every_semantic_constant_drift_is_rejected_by_both(self) -> None:
        report = self.fixture["report"]
        for key, value in report["semantic_contract"].items():
            with self.subTest(key=key):
                mutated = copy.deepcopy(report)
                if type(value) is int:
                    replacement: Any = value + 1
                elif type(value) is float:
                    replacement = value * 2.0
                else:
                    replacement = value + ".drift"
                mutated["semantic_contract"][key] = replacement
                self.assert_both_reject(mutated)

    def test_external_resolved_spec_tamper_is_rejected_by_hand_validator(self) -> None:
        report = self.fixture["report"]
        validate_fixed_schema(report, self.fixture["schema"])
        resolved = self.fixture["resolved"]
        resolved.chmod(0o644)
        resolved.write_bytes(resolved.read_bytes() + b"# tamper\n")
        resolved.chmod(0o444)
        with self.assertRaisesRegex(
            VALIDATOR.ReceiptValidationError, "resolved spec exact bytes"
        ):
            VALIDATOR.validate_render_receipt(
                report, self.fixture["validator_args"]
            )

    def test_template_placeholder_residual_and_duplicate_are_rejected(self) -> None:
        template = JOB_TEMPLATE.read_bytes()
        names = RENDERER.PLACEHOLDER_RE.findall(template.decode("utf-8"))
        replacements = {name: f"safe-{index}" for index, name in enumerate(names)}
        rendered = RENDERER.render_template(template, replacements)
        self.assertIsNone(RENDERER.PLACEHOLDER_RE.search(rendered.decode("utf-8")))
        with self.assertRaisesRegex(RENDERER.RenderError, "placeholder occurrence"):
            RENDERER.render_template(template + b"\nPLACEHOLDER_EXTRA\n", replacements)
        duplicated = template.replace(
            names[0].encode("ascii"), names[1].encode("ascii"), 1
        )
        with self.assertRaisesRegex(RENDERER.RenderError, "placeholder occurrence"):
            RENDERER.render_template(duplicated, replacements)


class LoadSemanticsValidatorTests(unittest.TestCase):
    @staticmethod
    def valid_boot() -> dict[str, object]:
        return {
            "world_size": 2,
            "boot": {
                "mode": "fresh",
                "stage_step": 0,
                "checkpoint_dir": "/checkpoints/control",
                "checkpoint_dir_absent": True,
                "local_last_absent": True,
                "resolved_train_branch": "fresh",
                "branch_resolver": (
                    "training.formal_boot.resolve_checkpoint_boot_branch"
                ),
                "config_schema_warning_count": 0,
            },
        }

    @staticmethod
    def valid_scheduler() -> dict[str, object]:
        base = VALIDATOR.EXPECTED_BASE_LR
        first = VALIDATOR.EXPECTED_FIRST_OPTIMIZER_UPDATE_LR
        step1 = VALIDATOR.EXPECTED_STEP1_LOGGED_LR
        return {
            "scheduler": {
                "source_ignored": True,
                "state_unchanged_by_init": True,
                "warmup_steps": 500,
                "first_optimizer_update_lr": [first],
                "step1_logged_lr": [step1],
                "fresh_state": {
                    "last_epoch": 0,
                    "_step_count": 1,
                    "_last_lr": [first],
                    "base_lrs": [base],
                },
                "optimizer_hparams_preserved": True,
                "optimizer_group_hparams": [
                    {"lr": first, "initial_lr": base}
                ],
            }
        }

    @staticmethod
    def tensor_detail(shape: list[int], *, scalar: bool = False) -> dict[str, object]:
        local = [] if scalar else list(shape)
        return {
            "global_shape": [] if scalar else list(shape),
            "local_shape": local,
            "dtype": "torch.float32",
            "exact": True,
            "max_abs": 0.0,
        }

    @classmethod
    def valid_closure(cls) -> dict[str, object]:
        details: dict[str, object] = {}
        for index, shape in enumerate(([1], [4, 8], [1024, 1024])):
            tensor = cls.tensor_detail(list(shape))
            details[f"model.layer{index}.weight"] = {
                "raw": dict(tensor),
                "ema": dict(tensor),
                "adam_step": cls.tensor_detail([], scalar=True),
                "adam_exp_avg": dict(tensor),
                "adam_exp_avg_sq": dict(tensor),
                "adam_step_value": 291667,
            }
        return {
            "closure": {
                "raw_count": 483,
                "ema_count": 483,
                "optimizer_count": 483,
                "optimizer_param_group_count": 1,
                "adam_step_values": [291667],
                "exact_per_rank": {
                    "raw": 483,
                    "ema": 483,
                    "adam_step": 483,
                    "adam_exp_avg": 483,
                    "adam_exp_avg_sq": 483,
                },
                "ranks_completed": 2,
                "size_stratified_details": details,
            }
        }

    def test_fresh_real_resolver_and_exact_lr_order_pass(self) -> None:
        VALIDATOR.validate_boot(self.valid_boot())
        VALIDATOR.validate_scheduler(self.valid_scheduler())

    def test_manual_or_continue_branch_claim_is_rejected(self) -> None:
        report = self.valid_boot()
        report["boot"]["resolved_train_branch"] = "continue"
        with self.assertRaisesRegex(
            VALIDATOR.ReceiptValidationError, "resolved_train_branch"
        ):
            VALIDATOR.validate_boot(report)

    def test_ratio_preserving_but_wrong_lr_is_rejected(self) -> None:
        report = self.valid_scheduler()
        wrong_first = VALIDATOR.EXPECTED_FIRST_OPTIMIZER_UPDATE_LR * 3.0
        report["scheduler"]["first_optimizer_update_lr"] = [wrong_first]
        report["scheduler"]["step1_logged_lr"] = [wrong_first * 2.0]
        report["scheduler"]["fresh_state"]["_last_lr"] = [wrong_first]
        report["scheduler"]["fresh_state"]["base_lrs"] = [wrong_first * 500.0]
        report["scheduler"]["optimizer_group_hparams"] = [
            {"lr": wrong_first, "initial_lr": wrong_first * 500.0}
        ]
        with self.assertRaisesRegex(
            VALIDATOR.ReceiptValidationError, "first_optimizer_update_lr"
        ):
            VALIDATOR.validate_scheduler(report)

    def test_exact_483_raw_ema_adam_and_step_pass(self) -> None:
        VALIDATOR.validate_closure(self.valid_closure())

    def test_fresh_optimizer_or_wrong_adam_step_is_rejected(self) -> None:
        report = self.valid_closure()
        report["closure"]["optimizer_count"] = 0
        with self.assertRaisesRegex(
            VALIDATOR.ReceiptValidationError, "optimizer_count"
        ):
            VALIDATOR.validate_closure(report)
        report = self.valid_closure()
        report["closure"]["adam_step_values"] = [0]
        with self.assertRaisesRegex(
            VALIDATOR.ReceiptValidationError, "adam_step_values"
        ):
            VALIDATOR.validate_closure(report)


class JobTemplateTests(unittest.TestCase):
    def test_job_binds_all_core_and_release_asset_blobs(self) -> None:
        job = JOB_TEMPLATE.read_text(encoding="utf-8")
        for path, attribute in VALIDATOR.EXPECTED_CORE_SOURCE_ARGUMENTS.items():
            option = "--" + attribute.replace("_", "-")
            with self.subTest(path=path, option=option):
                self.assertGreaterEqual(job.count(option), 2)
        self.assertIn('"$PY" "$VALIDATOR"', job)
        self.assertIn("verify_final_blob training/formal_boot.py", job)
        self.assertIn("verify_final_blob training/config_schema.py", job)
        self.assertIn("verify_final_blob networks/vae.py", job)
        self.assertIn('verify_final_blob "$CONFIG_RELATIVE"', job)
        self.assertIn('verify_final_blob "$PROBE_RELATIVE"', job)
        self.assertIn('verify_final_blob "$VALIDATOR_RELATIVE"', job)
        self.assertIn('verify_final_blob "$JOB_TEMPLATE_RELATIVE"', job)


if __name__ == "__main__":
    unittest.main(verbosity=2)
