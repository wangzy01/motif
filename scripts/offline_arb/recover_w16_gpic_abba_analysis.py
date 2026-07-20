#!/usr/bin/env python3
"""Narrow recovery wrapper for the immutable r4 W16 ABBA capture analyzer.

The capture-bound analyzer is intentionally left byte-for-byte unchanged.  Its
first independent run exposed one schema bug: capture sidecar control/ledger
declarations are ``{path, sha256}``, while the analyzer's generic helper expects
``{path, bytes, sha256}``.  This wrapper patches only those 24 exact labels.

The failed first analysis also preserved a valid full-sidecar snapshot before
reaching the schema bug.  A retry may reuse that one file only when its decoded
JSON is exactly equal to the freshly recomputed value.  Every other write keeps
the original no-overwrite behavior.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


REPO = Path("/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop")
CAPTURE_ANALYZER = REPO / "scripts/offline_arb/analyze_w16_gpic_abba_raw_gate.py"
EXPECTED_CAPTURE_ANALYZER_SHA256 = (
    "9451471ddc77cc5c831b68a6fb5c03abe5cfd93173d332a5355008e643cb848f"
)
ARM_NAMES = ("l16_a", "l8_a", "l8_b", "l16_b")
DECLARATION_LABEL_RE = re.compile(
    r"^(l16_a|l8_a|l8_b|l16_b) node ([01]) sidecar "
    r"(control|ledger (before|after))$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_capture_analyzer() -> Any:
    if (
        CAPTURE_ANALYZER.is_symlink()
        or not CAPTURE_ANALYZER.is_file()
        or sha256_file(CAPTURE_ANALYZER) != EXPECTED_CAPTURE_ANALYZER_SHA256
    ):
        raise RuntimeError("capture-bound W16 analyzer bytes changed")
    specification = importlib.util.spec_from_file_location(
        "offline_arb_w16_capture_analyzer_immutable", CAPTURE_ANALYZER
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load capture-bound W16 analyzer")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _expected_recovery_labels() -> set[str]:
    return {
        f"{arm} node {node} sidecar {suffix}"
        for arm in ARM_NAMES
        for node in range(2)
        for suffix in ("control", "ledger before", "ledger after")
    }


def validate_unsized_reference(
    module: Any,
    path: Path,
    declaration: Any,
    label: str,
) -> str:
    """Validate the one capture schema that the immutable analyzer mishandled."""

    if DECLARATION_LABEL_RE.fullmatch(label) is None:
        raise ValueError(f"{label} is not an approved recovery declaration")
    if not isinstance(declaration, dict) or set(declaration) != {"path", "sha256"}:
        raise ValueError(f"{label} recovery declaration key set drift")
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} missing/empty/non-regular/symlink")
    if declaration["path"] != str(path.resolve(strict=True)):
        raise ValueError(f"{label} canonical path mismatch")
    expected_sha = declaration["sha256"]
    if not isinstance(expected_sha, str) or module.SHA_RE.fullmatch(expected_sha) is None:
        raise ValueError(f"{label} SHA malformed")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"{label} SHA mismatch")
    return actual_sha


def validate_reusable_snapshot(
    module: Any,
    path: Path,
    value: Mapping[str, Any],
) -> str:
    """Accept a prior analysis snapshot only when fresh JSON is type-exact."""

    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("preserved analysis-sidecar snapshot is not reusable")
    existing = json.loads(path.read_text(encoding="utf-8"))
    if not module.exact_json_equal(existing, value):
        raise ValueError("preserved analysis-sidecar snapshot differs from fresh verification")
    return sha256_file(path)


def main() -> int:
    module = _load_capture_analyzer()
    args = module.parse_args()
    master = Path(args.master).resolve(strict=True)
    if not args.output:
        raise RuntimeError("post-hoc recovery requires an explicit output path")
    output_path = Path(args.output).resolve(strict=False)
    if (
        output_path.parent != master.parent
        or not output_path.name.startswith("w16_gpic_abba_posthoc_diagnostic.")
    ):
        raise RuntimeError("post-hoc recovery output path/name is not canonical")
    expected_reused_snapshot = master.parent / f"sidecarfull.analysis.{master.parent.name}.json"
    original_validate = module.validate_sha_artifact
    original_atomic_write = module.atomic_write
    recovered_labels: list[str] = []
    reused_snapshots: list[Path] = []

    def recovery_validate_sha_artifact(
        path: Path,
        declaration: Any,
        label: str,
        *,
        named: bool,
    ) -> str:
        match = DECLARATION_LABEL_RE.fullmatch(label)
        if match is None:
            return original_validate(path, declaration, label, named=named)
        if named:
            raise ValueError(f"{label} recovery declaration unexpectedly named")
        actual_sha = validate_unsized_reference(module, path, declaration, label)
        recovered_labels.append(label)
        return actual_sha

    def recovery_atomic_write(path: Path, value: Mapping[str, Any]) -> None:
        if path != expected_reused_snapshot:
            raise ValueError(f"immutable analyzer attempted an unexpected recovery write: {path}")
        validate_reusable_snapshot(module, path, value)
        reused_snapshots.append(path)

    module.validate_sha_artifact = recovery_validate_sha_artifact
    module.atomic_write = recovery_atomic_write
    source_result = module.analyze(args)
    expected_labels = _expected_recovery_labels()
    if len(recovered_labels) != len(expected_labels) or set(recovered_labels) != expected_labels:
        raise RuntimeError(
            "recovery declaration coverage mismatch: "
            f"count={len(recovered_labels)} unique={len(set(recovered_labels))}"
        )
    if reused_snapshots != [expected_reused_snapshot]:
        raise RuntimeError("analysis-sidecar snapshot recovery coverage mismatch")
    if sha256_file(CAPTURE_ANALYZER) != EXPECTED_CAPTURE_ANALYZER_SHA256:
        raise RuntimeError("capture-bound W16 analyzer changed during recovery")
    if (
        source_result.get("schema_version") != module.ANALYSIS_SCHEMA
        or source_result.get("status") != "NO_GO"
        or source_result.get("decision") not in {"NO_GO", "INCONCLUSIVE"}
        or source_result.get("selection_candidates") != []
    ):
        raise RuntimeError("post-hoc recovery refuses a selectable or malformed result")
    diagnostic = {
        "schema_version": "offline_arb_w16_gpic_abba_posthoc_diagnostic_v1",
        "status": "POSTHOC_DIAGNOSTIC",
        "production_selector_eligible": False,
        "identity": {
            "run_id": args.run_id,
            "source_jobtag": master.parent.name,
            "master": str(master),
            "master_sha256": sha256_file(master),
        },
        "recovery": {
            "reason": (
                "capture sidecar control/ledger declarations are exact {path,sha256}; "
                "the immutable analyzer expected sized path declarations"
            ),
            "capture_analyzer": str(CAPTURE_ANALYZER),
            "capture_analyzer_sha256": EXPECTED_CAPTURE_ANALYZER_SHA256,
            "wrapper": str(Path(__file__).resolve(strict=True)),
            "wrapper_sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "recovered_declaration_labels": sorted(recovered_labels),
            "reused_analysis_sidecar_snapshot": str(expected_reused_snapshot),
            "reused_analysis_sidecar_snapshot_sha256": sha256_file(
                expected_reused_snapshot
            ),
        },
        "source_analysis": source_result,
    }
    original_atomic_write(output_path, diagnostic)
    if output_path.is_symlink() or not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError("post-hoc diagnostic output did not close atomically")
    print(
        "W16_GPIC_ABBA_POSTHOC_DIAGNOSTIC_PRESERVED "
        f"output={output_path} sha256={sha256_file(output_path)} "
        f"decision={source_result['decision']}"
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
