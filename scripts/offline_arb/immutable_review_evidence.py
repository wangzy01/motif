#!/usr/bin/env python3
"""Fail-closed helpers for immutable review-evidence directory publication."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import platform
import re
import select
import stat
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class EvidencePublishError(RuntimeError):
    pass


STAGE512A_REVIEWED_GATE_BINDING_LABELS = frozenset(
    {
        "plan_header",
        "catalog_header",
        "sidecar_header",
        "caption_audit",
        "textatlas_audit",
        "textatlas_bindings",
        "textatlas_bindings_receipt",
        "textatlas_bindings_success",
        "textatlas_final_review",
        "textatlas_final_review_receipt",
        "textatlas_final_review_success",
        "offline_dataset_runtime",
        "caption_runtime",
        "image_runtime",
        "image_fetch_runtime",
        "targeted_test_runtime",
        "gate_finalizer",
        "gate_evidence_helper",
        "write_renderer",
    }
)

# Additive W64 gate bindings.  The W48 set above is a released renderer ABI and
# remains unchanged.  W64 consumes one catalog-level rebind closure instead of
# relabeling the W48 plan-bound stage1/4a/4c artifacts as new authority.
STAGE512A_W64_REVIEWED_GATE_BINDING_LABELS = frozenset(
    {
        "plan_header",
        "plan_checksums",
        "catalog_header",
        "sidecar_header",
        "caption_audit",
        "caption_horizon_audit",
        "plan_rows",
        "plan_bitmap",
        "textatlas_rebind",
        "textatlas_rebind_receipt",
        "textatlas_rebind_success",
        "w64_exception_contract",
        "offline_dataset_runtime",
        "caption_runtime",
        "image_runtime",
        "image_fetch_runtime",
        "targeted_test_runtime",
        "gate_finalizer",
        "gate_evidence_helper",
        "write_renderer",
    }
)

# Exact immutable-write render bindings consumed by the topology-only
# TextAtlas evidence rebind.  Keeping this closure next to the generic render
# provenance verifier prevents the writer and renderer from growing separate
# notions of which live bytes authorized a publication.
STAGE512A_TEXTATLAS_REBIND_BINDING_LABELS = frozenset(
    {
        "legacy_plan_rows",
        "legacy_plan_bitmap",
        "legacy_integrity",
        "legacy_integrity_receipt",
        "legacy_bindings",
        "legacy_bindings_receipt",
        "legacy_final_review",
        "legacy_final_receipt",
        "new_plan_rows",
        "new_plan_bitmap",
        "new_plan_header",
        "new_plan_checksums",
        "rebind_writer",
        "evidence_helper",
        "write_renderer",
    }
)


def absolute_lexical(path: Path) -> Path:
    """Return an absolute normalized path without following symlinks."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def require_real_directory(path: Path, label: str) -> Path:
    lexical = absolute_lexical(path)
    try:
        resolved = lexical.resolve(strict=True)
        metadata = os.lstat(lexical)
    except (FileNotFoundError, OSError) as exc:
        raise EvidencePublishError(
            f"{label} is not an existing real directory: {lexical}"
        ) from exc
    if resolved != lexical or stat.S_ISLNK(metadata.st_mode):
        raise EvidencePublishError(
            f"{label} must not contain or be a symlink: {lexical}"
        )
    if not stat.S_ISDIR(metadata.st_mode):
        raise EvidencePublishError(f"{label} is not a directory: {lexical}")
    return lexical


def require_real_file(path: Path, label: str) -> Path:
    lexical = absolute_lexical(path)
    try:
        resolved = lexical.resolve(strict=True)
        metadata = os.lstat(lexical)
    except (FileNotFoundError, OSError) as exc:
        raise EvidencePublishError(
            f"{label} is not an existing real file: {lexical}"
        ) from exc
    if resolved != lexical or stat.S_ISLNK(metadata.st_mode):
        raise EvidencePublishError(
            f"{label} must not contain or be a symlink: {lexical}"
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise EvidencePublishError(f"{label} is not a regular file: {lexical}")
    return lexical


def require_exact_directory_members(
    path: Path,
    expected_names: set[str],
    label: str,
) -> Path:
    """Require a committed evidence directory to contain exactly regular members."""

    directory = require_real_directory(path, label)
    if any(
        not isinstance(name, str) or not name or Path(name).name != name
        for name in expected_names
    ):
        raise EvidencePublishError(f"{label} expected member contract is malformed")
    observed_names = set(os.listdir(directory))
    if observed_names != expected_names:
        raise EvidencePublishError(
            f"{label} member set drift: observed={sorted(observed_names)},"
            f"expected={sorted(expected_names)}"
        )
    for name in sorted(expected_names):
        require_real_file(directory / name, f"{label} member {name}")
    return directory


def bind_recoverable_child_directory(path: Path, parent: Path, label: str) -> Path:
    real_parent = require_real_directory(parent, f"{label}.parent")
    lexical = absolute_lexical(path)
    if lexical.parent != real_parent:
        raise EvidencePublishError(
            f"{label} must be a direct child of {real_parent}: {lexical}"
        )
    try:
        metadata = os.lstat(lexical)
    except FileNotFoundError:
        return lexical
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise EvidencePublishError(
            f"{label} must be absent or a real recovery directory: {lexical}"
        )
    if lexical.resolve(strict=True) != lexical:
        raise EvidencePublishError(f"{label} contains a symlink: {lexical}")
    return lexical


def read_stable_bytes(path: Path, label: str) -> bytes:
    """Read one immutable regular-file identity, rejecting replacement/mutation."""

    real_path = require_real_file(path, label)
    before = os.lstat(real_path)

    def identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_mode),
            int(metadata.st_nlink),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            int(metadata.st_ctime_ns),
        )

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(real_path, flags)
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise EvidencePublishError(f"SHA input is not regular: {real_path}")
        if identity(before) != identity(opened):
            raise EvidencePublishError(f"SHA input changed before open: {real_path}")
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(real_path)
    except OSError as exc:
        raise EvidencePublishError(f"SHA input disappeared after read: {real_path}") from exc
    if identity(before) != identity(after_open) or identity(before) != identity(after_path):
        raise EvidencePublishError(f"{label} changed during read: {real_path}")
    return b"".join(chunks)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(read_stable_bytes(path, "sha256 input")).hexdigest()


def file_provenance(
    path: Path,
    label: str,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    """Read a file once and return the path/SHA identity of those exact bytes."""

    real_path = require_real_file(path, label)
    raw = read_stable_bytes(real_path, label)
    observed_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        expected = validate_sha256(expected_sha256, f"{label} expected SHA")
        if observed_sha != expected:
            raise EvidencePublishError(
                f"{label} SHA drift: observed={observed_sha},expected={expected}"
            )
    return {"path": str(real_path), "sha256": observed_sha}


def load_json_object(
    path: Path,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    raw = read_stable_bytes(path, label)
    observed_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed_sha != validate_sha256(
        expected_sha256, f"{label} expected SHA"
    ):
        raise EvidencePublishError(
            f"{label} SHA drift: observed={observed_sha},expected={expected_sha256}"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidencePublishError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvidencePublishError(f"{label} JSON root is not an object: {path}")
    return value, observed_sha


def load_committed_json_object(
    receipt_path: Path,
    marker_path: Path,
    label: str,
    *,
    marker_prefix: str,
    expected_receipt_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Load one receipt byte identity and bind its commit marker to that SHA."""

    receipt, receipt_sha = load_json_object(
        receipt_path,
        f"{label} receipt",
        expected_receipt_sha256,
    )
    marker = read_stable_bytes(marker_path, f"{label} commit marker")
    expected_marker = f"{marker_prefix}{receipt_sha}\n".encode("ascii")
    if marker != expected_marker:
        raise EvidencePublishError(f"{label} commit marker/receipt mismatch")
    return receipt, receipt_sha


def load_rendered_job_provenance(
    path: Path,
    *,
    expected_container_image: str,
    expected_launch_token: str,
) -> dict[str, str]:
    """Bind the exact rendered submission YAML, its image, and launch token."""

    real_path = require_real_file(path, "rendered job spec")
    raw = read_stable_bytes(real_path, "rendered job spec")
    observed_sha = hashlib.sha256(raw).hexdigest()
    try:
        import yaml
    except ImportError as exc:
        raise EvidencePublishError("PyYAML is required to bind rendered job specs") from exc
    try:
        value = yaml.safe_load(raw)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EvidencePublishError("rendered job spec is not valid UTF-8 YAML") from exc
    if not isinstance(value, Mapping):
        raise EvidencePublishError("rendered job spec root is not a mapping")
    image = value.get("image")
    command = value.get("command")
    if image != expected_container_image:
        raise EvidencePublishError(
            f"rendered job container image drift: {image!r}"
        )
    if not isinstance(command, str):
        raise EvidencePublishError("rendered job command is not a string")
    launch_assignments = re.findall(
        r"(?m)^\s*LAUNCH_TOKEN=([A-Za-z0-9_.:-]+)\s*$", command
    )
    if launch_assignments != [expected_launch_token]:
        raise EvidencePublishError("rendered job launch token binding is missing")
    return {
        "path": str(real_path),
        "sha256": observed_sha,
        "container_image": expected_container_image,
        "launch_token": expected_launch_token,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidencePublishError(
            f"cannot canonically encode rendered-job payload: {exc}"
        ) from exc


def _artifact_with_size(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    real_path = require_real_file(path, label)
    raw = read_stable_bytes(real_path, label)
    return {
        "path": str(real_path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }, raw


def load_stage512a_rendered_write_provenance(
    rendered_yaml: Path,
    *,
    expected_launch_token: str | None,
    expected_template: Path,
    expected_container_image: str,
    expected_bindings: Mapping[str, Path],
    expected_binding_labels: frozenset[str],
    expected_producers: Mapping[str, Path],
    provenance_schema: str = "stage512a_rendered_write_producer_provenance_v1",
) -> dict[str, Any]:
    """Close one stage512a write to its exact rendered payload and producers.

    The rendered YAML contains the canonical render payload and its token.  It
    cannot bind its own SHA without a circular hash, so this function binds the
    completed YAML from the produced artifact instead.  Every expected input
    and producer is re-read and compared to the payload, closing the verifier to
    publisher TOCTOU window before publication.
    """

    if not expected_binding_labels:
        raise EvidencePublishError("rendered-write binding label contract is empty")
    if set(expected_bindings) != expected_binding_labels:
        raise EvidencePublishError(
            "reviewed-gate expected binding contract is incomplete: "
            f"observed={sorted(expected_bindings)},"
            f"expected={sorted(expected_binding_labels)}"
        )
    if not expected_producers or not set(expected_producers).issubset(
        expected_binding_labels
    ):
        raise EvidencePublishError(
            "rendered-write producer binding contract is incomplete"
        )

    yaml_artifact, yaml_raw = _artifact_with_size(
        rendered_yaml, "reviewed-gate rendered YAML"
    )
    if stat.S_IMODE(os.lstat(Path(yaml_artifact["path"])).st_mode) & 0o222:
        raise EvidencePublishError("reviewed-gate rendered YAML is writable")
    try:
        import yaml
    except ImportError as exc:
        raise EvidencePublishError(
            "PyYAML is required to bind the reviewed-gate rendered YAML"
        ) from exc
    try:
        document = yaml.safe_load(yaml_raw)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EvidencePublishError(
            "reviewed-gate rendered YAML is not valid UTF-8 YAML"
        ) from exc
    if not isinstance(document, Mapping) or not isinstance(
        document.get("command"), str
    ):
        raise EvidencePublishError(
            "reviewed-gate rendered YAML must contain one command string"
        )
    if document.get("image") != expected_container_image:
        raise EvidencePublishError("reviewed-gate rendered YAML image drift")
    command = str(document["command"])

    def assignment(name: str, pattern: str) -> str:
        observed = re.findall(
            rf"(?m)^\s*{re.escape(name)}='({pattern})'\s*$", command
        )
        if len(observed) != 1:
            raise EvidencePublishError(
                f"reviewed-gate rendered YAML {name} assignment is not unique"
            )
        return observed[0]

    token = assignment("RENDERED_WRITE_TOKEN", r"STAGE512A_WRITE_V1_[0-9a-f]{64}")
    payload_b64 = assignment("RENDERED_WRITE_BINDINGS_B64", r"[A-Za-z0-9+/]+={0,2}")
    yaml_self = assignment("RENDERED_WRITE_YAML_SELF", r"[A-Za-z0-9/_.-]+")
    if expected_launch_token is not None and token != expected_launch_token:
        raise EvidencePublishError("reviewed-gate launch token differs from producer CLI")
    if yaml_self != yaml_artifact["path"]:
        raise EvidencePublishError("reviewed-gate YAML-self path mismatch")
    try:
        payload_raw = base64.b64decode(payload_b64.encode("ascii"), validate=True)
        payload = json.loads(payload_raw.decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidencePublishError(
            f"cannot decode reviewed-gate render payload: {exc}"
        ) from exc
    if not isinstance(payload, Mapping) or _canonical_json_bytes(payload) != payload_raw:
        raise EvidencePublishError("reviewed-gate render payload is not canonical JSON")
    if set(payload) != {
        "schema_version",
        "approval",
        "template",
        "output_path",
        "job_name",
        "bindings",
    }:
        raise EvidencePublishError("reviewed-gate render payload key drift")
    if (
        payload.get("schema_version") != "stage512a_immutable_write_render_v1"
        or payload.get("approval") != "STAGE512A_IMMUTABLE_WRITE"
        or payload.get("output_path") != yaml_artifact["path"]
        or payload.get("job_name") != document.get("name")
    ):
        raise EvidencePublishError("reviewed-gate render payload identity drift")
    derived_token = "STAGE512A_WRITE_V1_" + hashlib.sha256(payload_raw).hexdigest()
    if token != derived_token:
        raise EvidencePublishError("reviewed-gate launch token/render payload mismatch")

    template_info = payload.get("template")
    if not isinstance(template_info, Mapping):
        raise EvidencePublishError("reviewed-gate template binding is malformed")
    template_artifact, template_raw = _artifact_with_size(
        expected_template, "reviewed-gate template"
    )
    expected_template_info = {"label": "template", **template_artifact}
    if dict(template_info) != expected_template_info:
        raise EvidencePublishError("reviewed-gate live template binding drift")
    try:
        template_text = template_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidencePublishError("reviewed-gate template is not UTF-8") from exc
    replacements = {
        "__STAGE512A_WRITE_TOKEN_UNRENDERED__": token,
        "__STAGE512A_WRITE_BINDINGS_B64_UNRENDERED__": payload_b64,
        "__STAGE512A_WRITE_YAML_SELF_UNRENDERED__": yaml_artifact["path"],
        "__STAGE512A_WRITE_JOB_NAME_UNRENDERED__": str(payload["job_name"]),
    }
    for placeholder in replacements:
        if template_text.count(placeholder) != 1:
            raise EvidencePublishError(
                f"reviewed-gate template placeholder count drift: {placeholder}"
            )
    expected_rendered = template_text
    for placeholder, replacement in replacements.items():
        expected_rendered = expected_rendered.replace(placeholder, replacement)
    if expected_rendered.encode("utf-8") != yaml_raw:
        raise EvidencePublishError(
            "reviewed-gate YAML differs from deterministic template rendering"
        )

    bindings = payload.get("bindings")
    if not isinstance(bindings, list) or any(
        not isinstance(item, Mapping) for item in bindings
    ):
        raise EvidencePublishError("reviewed-gate render bindings are malformed")
    by_label: dict[str, Mapping[str, Any]] = {}
    for item in bindings:
        label = item.get("label")
        if not isinstance(label, str) or label in by_label:
            raise EvidencePublishError("reviewed-gate render binding labels are invalid")
        by_label[label] = item
    if set(by_label) != expected_binding_labels:
        raise EvidencePublishError(
            "reviewed-gate render binding set drift: "
            f"observed={sorted(by_label)},"
            f"expected={sorted(expected_binding_labels)}"
        )
    binding_artifacts: dict[str, dict[str, Any]] = {}
    for label, path in expected_bindings.items():
        artifact, _ = _artifact_with_size(
            path, f"reviewed-gate complete input binding {label}"
        )
        if dict(by_label[label]) != {"label": label, **artifact}:
            raise EvidencePublishError(
                f"reviewed-gate complete input binding drift: {label}"
            )
        binding_artifacts[label] = artifact
    producer_artifacts: dict[str, dict[str, Any]] = {}
    for label, path in expected_producers.items():
        artifact, _ = _artifact_with_size(path, f"reviewed-gate producer {label}")
        expected_binding = {"label": label, **artifact}
        if dict(by_label.get(label, {})) != expected_binding:
            raise EvidencePublishError(
                f"reviewed-gate producer binding drift: {label}"
            )
        producer_artifacts[label] = artifact
    return {
        "schema": provenance_schema,
        "job_name": payload["job_name"],
        "container_image": expected_container_image,
        "launch_token": token,
        "render_payload_sha256": hashlib.sha256(payload_raw).hexdigest(),
        "rendered_yaml": yaml_artifact,
        "template": template_artifact,
        "bindings": binding_artifacts,
        "producers": producer_artifacts,
    }


def load_stage512a_write_producer_provenance(
    rendered_yaml: Path,
    *,
    expected_launch_token: str,
    expected_template: Path,
    expected_finalizer: Path,
    expected_evidence_helper: Path,
    expected_renderer: Path,
    expected_container_image: str,
    expected_bindings: Mapping[str, Path],
    expected_binding_labels: frozenset[str] = STAGE512A_REVIEWED_GATE_BINDING_LABELS,
) -> dict[str, Any]:
    """Backward-compatible reviewed-gate producer-provenance wrapper."""

    generic = load_stage512a_rendered_write_provenance(
        rendered_yaml,
        expected_launch_token=expected_launch_token,
        expected_template=expected_template,
        expected_container_image=expected_container_image,
        expected_bindings=expected_bindings,
        expected_binding_labels=expected_binding_labels,
        expected_producers={
            "gate_finalizer": expected_finalizer,
            "gate_evidence_helper": expected_evidence_helper,
            "write_renderer": expected_renderer,
        },
        provenance_schema="stage512a_reviewed_gate_producer_provenance_v1",
    )
    return {
        "schema": generic["schema"],
        "job_name": generic["job_name"],
        "container_image": generic["container_image"],
        "launch_token": generic["launch_token"],
        "render_payload_sha256": generic["render_payload_sha256"],
        "rendered_yaml": generic["rendered_yaml"],
        "template": generic["template"],
        "finalizer": generic["producers"]["gate_finalizer"],
        "immutable_evidence_helper": generic["producers"][
            "gate_evidence_helper"
        ],
        "renderer": generic["producers"]["write_renderer"],
    }


def runtime_environment_provenance(container_image: str) -> dict[str, Any]:
    """Record deterministic decoder/runtime version strings available in-process."""

    try:
        import PIL
        from PIL import features

        pillow_version: str | None = str(PIL.__version__)
        libjpeg_version: str | None = features.version("jpg")
        libjpeg_turbo = bool(features.check_feature("libjpeg_turbo"))
        libjpeg_turbo_version: str | None = features.version_feature(
            "libjpeg_turbo"
        )
    except (ImportError, ValueError):
        pillow_version = None
        libjpeg_version = None
        libjpeg_turbo = False
        libjpeg_turbo_version = None
    return {
        "container_image": container_image,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_build": list(platform.python_build()),
        "python_executable": sys.executable,
        "pillow_version": pillow_version,
        "libjpeg_version": libjpeg_version,
        "libjpeg_turbo": libjpeg_turbo,
        "libjpeg_turbo_version": libjpeg_turbo_version,
    }


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def validate_plan_coordinates(
    value: Any,
    *,
    bucket_idx: int,
    label: str,
    plan_horizon: int,
    formal_steps: int,
    world_size: int,
    per_rank_bs: int,
    global_batch: int,
    within_formal_field: str = "within_formal",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate plan coordinates and return all/formal rows.

    ``within_formal_field`` defaults to the topology-independent
    ``within_formal`` schema.  A caller that must read a frozen legacy schema
    may select its historical field explicitly; the stage512a W48 wrapper
    below is the only production caller expected to do so.
    """

    contract = {
        "plan_horizon": plan_horizon,
        "formal_steps": formal_steps,
        "world_size": world_size,
        "per_rank_bs": per_rank_bs,
        "global_batch": global_batch,
    }
    for key, observed in contract.items():
        if (
            not isinstance(observed, int)
            or isinstance(observed, bool)
            or observed <= 0
        ):
            raise EvidencePublishError(
                f"{label} coordinate contract {key} is not a positive integer"
            )
    if formal_steps > plan_horizon:
        raise EvidencePublishError(
            f"{label} coordinate contract formal_steps exceeds plan_horizon"
        )
    if global_batch != world_size * per_rank_bs:
        raise EvidencePublishError(
            f"{label} coordinate contract global_batch/topology drift"
        )
    if not isinstance(within_formal_field, str) or not within_formal_field:
        raise EvidencePublishError(
            f"{label} coordinate contract within_formal_field is malformed"
        )

    if not isinstance(value, list) or not value or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise EvidencePublishError(f"{label} coordinates are malformed or empty")
    coordinates: list[dict[str, Any]] = []
    plan_indices: list[int] = []
    for offset, source in enumerate(value):
        coordinate = dict(source)
        for key in ("step", "rank", "slot", "bucket_idx"):
            observed = coordinate.get(key)
            if not isinstance(observed, int) or isinstance(observed, bool):
                raise EvidencePublishError(
                    f"{label} coordinate[{offset}].{key} is not an integer"
                )
        step = int(coordinate["step"])
        rank = int(coordinate["rank"])
        slot = int(coordinate["slot"])
        if (
            not 0 <= step < plan_horizon
            or not 0 <= rank < world_size
            or not 0 <= slot < per_rank_bs
        ):
            raise EvidencePublishError(
                f"{label} coordinate[{offset}] topology/range drift: "
                f"step={step},rank={rank},slot={slot}"
            )
        if int(coordinate["bucket_idx"]) != bucket_idx:
            raise EvidencePublishError(
                f"{label} coordinate[{offset}] bucket drift"
            )
        within_formal = coordinate.get(within_formal_field)
        if not isinstance(within_formal, bool) or within_formal is not (
            step < formal_steps
        ):
            raise EvidencePublishError(
                f"{label} coordinate[{offset}] formal-boundary drift"
            )
        coordinates.append(coordinate)
        plan_indices.append(step * global_batch + rank * per_rank_bs + slot)
    if any(left >= right for left, right in zip(plan_indices, plan_indices[1:])):
        raise EvidencePublishError(
            f"{label} coordinates are not strictly ordered by plan index"
        )
    formal = [
        item for item in coordinates if item[within_formal_field] is True
    ]
    return coordinates, formal


def validate_stage512a_coordinates(
    value: Any,
    *,
    bucket_idx: int,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate exact legacy W48/B32 plan coordinates and return all/formal rows."""

    return validate_plan_coordinates(
        value,
        bucket_idx=bucket_idx,
        label=label,
        plan_horizon=172032,
        formal_steps=166667,
        world_size=48,
        per_rank_bs=32,
        global_batch=1536,
        within_formal_field="within_formal_166667",
    )


def _inode_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (int(metadata.st_dev), int(metadata.st_ino), int(metadata.st_mode))


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        *_inode_identity(metadata),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _write_exclusive_at(directory_fd: int, name: str, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise EvidencePublishError(f"short evidence write: {name}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_member_at(directory_fd: int, name: str, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise EvidencePublishError(f"{label} is not a regular file")
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise EvidencePublishError(f"{label} changed before open")
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_name = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        _file_identity(before) != _file_identity(after_open)
        or _file_identity(before) != _file_identity(after_name)
    ):
        raise EvidencePublishError(f"{label} changed during read")
    return b"".join(chunks)


def _object_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    """Identity that survives an intentional chmod but not replacement."""

    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        stat.S_IFMT(metadata.st_mode),
    )


def _make_member_readonly_at(
    directory_fd: int,
    name: str,
    raw: bytes,
) -> None:
    """Set one anchored regular member to 0444 and revalidate exact bytes."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise EvidencePublishError(
            f"readonly evidence member is not one regular inode: {name}"
        )
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise EvidencePublishError(
                f"readonly evidence member changed before chmod: {name}"
            )
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        after_open = os.fstat(descriptor)
        after_name = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            _object_identity(after_open) != _object_identity(after_name)
            or after_open.st_nlink != 1
            or after_name.st_nlink != 1
            or stat.S_IMODE(after_open.st_mode) != 0o444
            or stat.S_IMODE(after_name.st_mode) != 0o444
        ):
            raise EvidencePublishError(
                f"readonly evidence member mode/inode drift: {name}"
            )
    finally:
        os.close(descriptor)
    if _read_member_at(directory_fd, name, f"readonly evidence {name}") != raw:
        raise EvidencePublishError(
            f"readonly evidence differs after chmod/readback: {name}"
        )


def _verify_readonly_directory_at(
    parent_fd: int,
    final_name: str,
    directory_fd: int,
    files: Mapping[str, bytes],
) -> None:
    """Require exact 0444 members under one stable 0555 directory."""

    expected_names = set(files)
    observed_names = set(os.listdir(directory_fd))
    if observed_names != expected_names:
        raise EvidencePublishError(
            "readonly evidence member set is not exact: "
            f"observed={sorted(observed_names)},expected={sorted(expected_names)}"
        )
    for name, raw in files.items():
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o444
        ):
            raise EvidencePublishError(
                f"readonly evidence member is not exact mode 0444: {name}"
            )
        if _read_member_at(directory_fd, name, f"readonly evidence {name}") != raw:
            raise EvidencePublishError(
                f"readonly evidence differs byte-for-byte: {name}"
            )
    opened = os.fstat(directory_fd)
    named = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or _object_identity(opened) != _object_identity(named)
        or stat.S_IMODE(opened.st_mode) != 0o555
        or stat.S_IMODE(named.st_mode) != 0o555
    ):
        raise EvidencePublishError(
            "readonly evidence directory is not one stable mode-0555 inode"
        )


def _publish_or_verify_member(
    directory_fd: int,
    name: str,
    raw: bytes,
) -> None:
    temporary = f".{name}.tmp.{hashlib.sha256(raw).hexdigest()}"
    try:
        _write_exclusive_at(directory_fd, temporary, raw)
    except FileExistsError:
        observed = _read_member_at(
            directory_fd, temporary, f"recoverable temporary {name}"
        )
        if observed != raw:
            # The content-addressed name is non-authoritative until linked to
            # the final member.  A killed writer may leave a stable short file;
            # unlinking that hidden inode is safe and lets one concurrent writer
            # recover while another may fail closed.
            os.unlink(temporary, dir_fd=directory_fd)
            os.fsync(directory_fd)
            _write_exclusive_at(directory_fd, temporary, raw)
    try:
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            observed = _read_member_at(directory_fd, name, f"existing evidence {name}")
            if observed != raw:
                raise EvidencePublishError(
                    f"existing partial-set member differs byte-for-byte: {name}"
                )
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.fsync(directory_fd)


def publish_recoverable_directory(
    destination: Path,
    files: Mapping[str, bytes],
    *,
    commit_member: str,
    final_readonly: bool = False,
) -> dict[str, str]:
    """Publish deterministic members with exact crash recovery and commit last.

    The target filesystem does not support ``RENAME_NOREPLACE`` for directories.
    Instead, the final directory is reserved with ``mkdirat(O_EXCL semantics)``;
    every immutable member is installed with a same-directory hard link.  A
    rerun may only accept byte-identical members and fills missing members.  The
    named commit marker is linked last, so consumers never accept a partial set.
    All mutation and verification is anchored to stable parent/child dirfds.
    When ``final_readonly`` is true, members are finalized to exact mode 0444
    and the directory to 0555 after the commit marker is installed.  A rerun
    then performs a mutation-free same-byte/mode verification.  The default is
    deliberately false to preserve the released recovery behavior.
    """

    if not isinstance(final_readonly, bool):
        raise EvidencePublishError("final_readonly must be a boolean")
    parent = require_real_directory(destination.parent, "evidence parent")
    final = bind_recoverable_child_directory(
        destination, parent, "evidence destination"
    )
    if not files:
        raise EvidencePublishError("evidence directory cannot be empty")
    normalized: dict[str, bytes] = {}
    for name, raw in files.items():
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise EvidencePublishError(f"invalid evidence member name: {name!r}")
        if not isinstance(raw, bytes):
            raise EvidencePublishError(f"evidence member is not bytes: {name}")
        if name in normalized:
            raise EvidencePublishError(f"duplicate evidence member: {name}")
        normalized[name] = raw
    if commit_member not in normalized:
        raise EvidencePublishError(f"commit member missing: {commit_member}")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    parent_before = os.lstat(parent)
    parent_fd = os.open(parent, directory_flags)
    try:
        if _inode_identity(parent_before) != _inode_identity(os.fstat(parent_fd)):
            raise EvidencePublishError("evidence parent changed before open")
        try:
            os.mkdir(final.name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            existing = os.stat(final.name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(existing.st_mode):
                raise EvidencePublishError(
                    f"evidence destination exists and is not a real directory: {final}"
                )
        directory_fd = os.open(final.name, directory_flags, dir_fd=parent_fd)
        try:
            child_before = os.fstat(directory_fd)
            child_named = os.stat(
                final.name, dir_fd=parent_fd, follow_symlinks=False
            )
            if _inode_identity(child_before) != _inode_identity(child_named):
                raise EvidencePublishError("evidence directory identity drift")
            expected_names = set(normalized)
            recoverable_temporaries = {
                f".{name}.tmp.{hashlib.sha256(raw).hexdigest()}": raw
                for name, raw in normalized.items()
            }
            observed_names = set(os.listdir(directory_fd))
            unknown = observed_names - expected_names - set(recoverable_temporaries)
            if unknown:
                raise EvidencePublishError(
                    f"evidence directory contains unknown members: {sorted(unknown)}"
                )
            child_mode = stat.S_IMODE(child_before.st_mode)
            already_readonly = (
                final_readonly
                and child_mode == 0o555
                and observed_names == expected_names
            )
            if already_readonly:
                _verify_readonly_directory_at(
                    parent_fd,
                    final.name,
                    directory_fd,
                    normalized,
                )
            else:
                if final_readonly and child_mode == 0o555:
                    raise EvidencePublishError(
                        "readonly evidence directory is incomplete or uncommitted"
                    )
                for temporary in sorted(
                    observed_names & set(recoverable_temporaries)
                ):
                    _read_member_at(
                        directory_fd,
                        temporary,
                        f"recoverable temporary {temporary}",
                    )
                    # A deterministic temporary may be complete or a stable
                    # short write from a killed process.  It is never
                    # authoritative; the final member/commit marker are.
                    os.unlink(temporary, dir_fd=directory_fd)
                if observed_names & set(recoverable_temporaries):
                    os.fsync(directory_fd)
                observed_names = set(os.listdir(directory_fd))
                try:
                    os.stat(
                        commit_member,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    if not observed_names.issubset(expected_names):
                        raise EvidencePublishError(
                            "partial evidence set contains a non-contract member"
                        )
                else:
                    if observed_names != expected_names:
                        raise EvidencePublishError(
                            "commit marker exists but evidence member set is not exact: "
                            f"observed={sorted(observed_names)},"
                            f"expected={sorted(expected_names)}"
                        )
                for name, raw in normalized.items():
                    if name != commit_member:
                        _publish_or_verify_member(directory_fd, name, raw)
                _publish_or_verify_member(
                    directory_fd, commit_member, normalized[commit_member]
                )
                for name, raw in normalized.items():
                    if (
                        _read_member_at(
                            directory_fd, name, f"published evidence {name}"
                        )
                        != raw
                    ):
                        raise EvidencePublishError(
                            f"published evidence differs byte-for-byte: {name}"
                        )
                final_names = set(os.listdir(directory_fd))
                if final_names != expected_names:
                    raise EvidencePublishError(
                        "published evidence member set is not exact: "
                        f"observed={sorted(final_names)},"
                        f"expected={sorted(expected_names)}"
                    )
                child_after = os.stat(
                    final.name, dir_fd=parent_fd, follow_symlinks=False
                )
                if _inode_identity(child_before) != _inode_identity(child_after):
                    raise EvidencePublishError(
                        "evidence directory replaced during publication"
                    )
                if final_readonly:
                    for name, raw in normalized.items():
                        if name != commit_member:
                            _make_member_readonly_at(directory_fd, name, raw)
                    _make_member_readonly_at(
                        directory_fd,
                        commit_member,
                        normalized[commit_member],
                    )
                    os.fchmod(directory_fd, 0o555)
                    os.fsync(directory_fd)
                    os.fsync(parent_fd)
                    _verify_readonly_directory_at(
                        parent_fd,
                        final.name,
                        directory_fd,
                        normalized,
                    )
        finally:
            os.close(directory_fd)
        parent_after = os.lstat(parent)
        if _inode_identity(parent_before) != _inode_identity(parent_after):
            raise EvidencePublishError("evidence parent replaced during publication")
    finally:
        os.close(parent_fd)
    return {
        name: hashlib.sha256(raw).hexdigest() for name, raw in normalized.items()
    }


def validate_sha256(value: str, label: str) -> str:
    normalized = value.lower()
    if (
        value != normalized
        or len(normalized) != 64
        or any(char not in "0123456789abcdef" for char in normalized)
    ):
        raise EvidencePublishError(f"{label} is not a lowercase SHA256: {value!r}")
    return normalized


def _open_real_directory_fd(path: Path, label: str) -> tuple[Path, int, tuple[int, int, int]]:
    directory = require_real_directory(path, label)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    before = os.lstat(directory)
    descriptor = os.open(directory, flags)
    opened = os.fstat(descriptor)
    identity = _inode_identity(before)
    if identity != _inode_identity(opened):
        os.close(descriptor)
        raise EvidencePublishError(f"{label} changed before dir-fd open")
    return directory, descriptor, identity


@dataclass
class ExclusiveWriterLock:
    """One locked inode plus the dir-fd/path identity that names it."""

    path: Path
    descriptor: int
    parent_descriptor: int
    parent_identity: tuple[int, int, int]
    lock_identity: tuple[int, ...]
    closed: bool = False

    def validate(self) -> None:
        if self.closed:
            raise EvidencePublishError("writer lock handle is already closed")
        try:
            parent_opened = os.fstat(self.parent_descriptor)
            parent_named = os.lstat(self.path.parent)
            opened = os.fstat(self.descriptor)
            named = os.stat(
                self.path.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise EvidencePublishError(
                f"writer lock pathname disappeared while held: {self.path}: {exc}"
            ) from exc
        if (
            self.parent_identity != _inode_identity(parent_opened)
            or self.parent_identity != _inode_identity(parent_named)
        ):
            raise EvidencePublishError("writer lock parent changed while held")
        if (
            self.lock_identity != _file_identity(opened)
            or self.lock_identity != _file_identity(named)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            raise EvidencePublishError("writer lock pathname/inode changed while held")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            os.close(self.descriptor)
        finally:
            os.close(self.parent_descriptor)


def acquire_exclusive_writer_lock(path: Path) -> ExclusiveWriterLock:
    """Safely create/open and non-blockingly lock one direct-child lock file."""

    lexical = absolute_lexical(path)
    parent, parent_fd, parent_identity = _open_real_directory_fd(
        lexical.parent, "writer lock parent"
    )
    if lexical.parent != parent or Path(lexical.name).name != lexical.name:
        os.close(parent_fd)
        raise EvidencePublishError("writer lock must be a direct normalized child")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(lexical.name, flags, 0o600, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        named = os.stat(lexical.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _file_identity(opened) != _file_identity(named)
        ):
            raise EvidencePublishError("writer lock is not one stable regular inode")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvidencePublishError("another immutable writer holds the lock") from exc
        named_after = os.stat(
            lexical.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if _file_identity(opened) != _file_identity(named_after):
            raise EvidencePublishError("writer lock path changed during acquisition")
        if parent_identity != _inode_identity(os.lstat(parent)):
            raise EvidencePublishError("writer lock parent changed during acquisition")
        result = ExclusiveWriterLock(
            path=lexical,
            descriptor=descriptor,
            parent_descriptor=parent_fd,
            parent_identity=parent_identity,
            lock_identity=_file_identity(opened),
        )
        descriptor = None
        parent_fd = -1
        try:
            result.validate()
        except BaseException:
            result.close()
            raise
        return result
    except OSError as exc:
        raise EvidencePublishError(f"cannot safely acquire writer lock: {lexical}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)


def _open_exclusive_output(
    path: Path, mode: int = 0o400
) -> tuple[int, int, Path, tuple[int, int, int]]:
    lexical = absolute_lexical(path)
    parent, parent_fd, _identity = _open_real_directory_fd(
        lexical.parent, "immutable output parent"
    )
    if lexical.parent != parent or Path(lexical.name).name != lexical.name:
        os.close(parent_fd)
        raise EvidencePublishError("immutable output must be a direct normalized child")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lexical.name, flags, mode, dir_fd=parent_fd)
    except OSError as exc:
        os.close(parent_fd)
        raise EvidencePublishError(
            f"cannot create immutable output exclusively: {lexical}: {exc}"
        ) from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
        os.close(descriptor)
        os.close(parent_fd)
        raise EvidencePublishError("exclusive output is not one regular inode")
    return descriptor, parent_fd, lexical, _identity


def _validate_open_output(
    descriptor: int,
    parent_fd: int,
    lexical: Path,
    parent_identity: tuple[int, int, int],
    label: str,
) -> None:
    """Revalidate both the anchored entry and its canonical parent pathname."""

    try:
        parent_opened = os.fstat(parent_fd)
        parent_named = os.lstat(lexical.parent)
        opened = os.fstat(descriptor)
        named = os.stat(lexical.name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise EvidencePublishError(f"{label} pathname changed: {lexical}: {exc}") from exc
    if (
        parent_identity != _inode_identity(parent_opened)
        or parent_identity != _inode_identity(parent_named)
    ):
        raise EvidencePublishError(f"{label} parent changed during publication")
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or _file_identity(opened) != _file_identity(named)
    ):
        raise EvidencePublishError(f"{label} pathname/inode changed during publication")


def _read_all_from_open_file(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise EvidencePublishError("short immutable evidence write")
        offset += written


def publish_exclusive_readonly_file(path: Path, raw: bytes) -> dict[str, Any]:
    """Create one final immutable file directly; never rename or overwrite."""

    descriptor, parent_fd, lexical, parent_identity = _open_exclusive_output(path)
    try:
        _write_all(descriptor, raw)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        _validate_open_output(
            descriptor,
            parent_fd,
            lexical,
            parent_identity,
            "immutable output",
        )
        if _read_all_from_open_file(descriptor) != raw:
            raise EvidencePublishError("published immutable evidence readback differs")
        _validate_open_output(
            descriptor,
            parent_fd,
            lexical,
            parent_identity,
            "immutable output",
        )
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o444:
            raise EvidencePublishError("published immutable evidence mode is not 0444")
        os.fsync(parent_fd)
    finally:
        os.close(descriptor)
        os.close(parent_fd)
    return {
        "path": str(lexical),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def capture_command_exclusive(path: Path, command: list[str]) -> int:
    """Tee one command to stdout and a directly-created immutable log file."""

    if not command:
        raise EvidencePublishError("capture command cannot be empty")
    descriptor, parent_fd, lexical, parent_identity = _open_exclusive_output(path)
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        if process.stdout is None:
            raise EvidencePublishError("capture command stdout pipe is missing")
        while True:
            block = process.stdout.read(1024 * 1024)
            if not block:
                break
            _write_all(descriptor, block)
            sys.stdout.buffer.write(block)
            sys.stdout.buffer.flush()
        return_code = process.wait()
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        _validate_open_output(
            descriptor,
            parent_fd,
            lexical,
            parent_identity,
            "captured evidence",
        )
        _read_all_from_open_file(descriptor)
        _validate_open_output(
            descriptor,
            parent_fd,
            lexical,
            parent_identity,
            "captured evidence",
        )
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o444:
            raise EvidencePublishError("captured evidence mode is not 0444")
        os.fsync(parent_fd)
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        os.close(descriptor)
        os.close(parent_fd)
    return int(return_code)


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    lock = sub.add_parser("hold-lock")
    lock.add_argument("--lock", required=True, type=Path)
    capture = sub.add_parser("capture-command")
    capture.add_argument("--output", required=True, type=Path)
    capture.add_argument("command", nargs=argparse.REMAINDER)
    line = sub.add_parser("publish-line")
    line.add_argument("--output", required=True, type=Path)
    line.add_argument("--line", required=True)
    return parser


def _cli_main(argv: list[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    if args.mode == "hold-lock":
        lock = acquire_exclusive_writer_lock(args.lock)
        try:
            sys.stdout.write("STAGE512A_SAFE_WRITER_LOCK_HELD\n")
            sys.stdout.flush()
            while True:
                readable, _, _ = select.select([sys.stdin.buffer], [], [], 0.25)
                lock.validate()
                if readable and not sys.stdin.buffer.read(1024 * 1024):
                    break
            lock.validate()
        finally:
            lock.close()
        return 0
    if args.mode == "capture-command":
        command = list(args.command)
        if command and command[0] == "--":
            command.pop(0)
        return capture_command_exclusive(args.output, command)
    publish_exclusive_readonly_file(
        args.output, (args.line + "\n").encode("utf-8")
    )
    return 0


__all__ = [
    "EvidencePublishError",
    "STAGE512A_REVIEWED_GATE_BINDING_LABELS",
    "STAGE512A_TEXTATLAS_REBIND_BINDING_LABELS",
    "STAGE512A_W64_REVIEWED_GATE_BINDING_LABELS",
    "absolute_lexical",
    "bind_recoverable_child_directory",
    "file_provenance",
    "json_bytes",
    "load_committed_json_object",
    "load_json_object",
    "load_rendered_job_provenance",
    "load_stage512a_rendered_write_provenance",
    "load_stage512a_write_producer_provenance",
    "ExclusiveWriterLock",
    "acquire_exclusive_writer_lock",
    "capture_command_exclusive",
    "publish_exclusive_readonly_file",
    "publish_recoverable_directory",
    "read_stable_bytes",
    "require_exact_directory_members",
    "require_real_directory",
    "require_real_file",
    "sha256_file",
    "runtime_environment_provenance",
    "validate_plan_coordinates",
    "validate_stage512a_coordinates",
    "validate_sha256",
]


if __name__ == "__main__":
    try:
        raise SystemExit(_cli_main())
    except EvidencePublishError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
