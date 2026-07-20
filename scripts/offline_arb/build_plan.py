#!/usr/bin/env python3
"""Build a deterministic OFFLINE_ARB_V1 plan from ``planner_input.json``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


# Direct execution sets sys.path[0] to scripts/offline_arb rather than the repo
# root.  Add the root explicitly so this CLI behaves the same with and without
# an editable package installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _put_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def _header_summary(header: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    """Return a compact summary while accepting descriptive or symbolic keys."""

    summary: dict[str, Any] = {
        "status": "ok",
        "plan_root": str(out_dir.resolve()),
    }
    aliases = {
        "format_version": ("format_version", "plan_format_version"),
        "algorithm_version": ("planner_algo_version", "algorithm_version", "planner_algorithm_version"),
        "plan_id": ("plan_id",),
        "world_size": ("world_size", "W"),
        "per_rank_bs": ("per_rank_bs", "bs"),
        "total_steps": ("total_steps", "H"),
        "global_batch_size": ("global_batch_size", "G"),
        "block_steps": ("block_steps",),
        "num_blocks": ("num_blocks", "block_count"),
        "input_sha256": ("input_sha256", "planner_input_sha256", "config_hash"),
    }
    for output_key, input_keys in aliases.items():
        _put_if_present(summary, output_key, _first(header, *input_keys))

    domains = _first(header, "domains")
    domain_totals = _first(header, "domain_totals", "D")
    if isinstance(domain_totals, Sequence) and not isinstance(domain_totals, (str, bytes)):
        if (
            isinstance(domains, Sequence)
            and not isinstance(domains, (str, bytes))
            and len(domains) == len(domain_totals)
        ):
            domain_totals = dict(zip(domains, domain_totals))
    _put_if_present(summary, "domain_totals", domain_totals)

    bucket_steps = _first(header, "bucket_steps", "S")
    if isinstance(bucket_steps, Sequence) and not isinstance(bucket_steps, (str, bytes)):
        summary["active_buckets"] = sum(int(value) > 0 for value in bucket_steps)
        summary["bucket_steps_sum"] = sum(int(value) for value in bucket_steps)

    solver = _first(header, "solver", "solver_record")
    _put_if_present(summary, "solver", solver)
    return summary


def _diagnostic(exc: BaseException) -> Any:
    to_dict = getattr(exc, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    for attribute in ("diagnostic", "details"):
        value = getattr(exc, attribute, None)
        if value is not None:
            return value
    return None


def _print_error(kind: str, exc: BaseException, *, exit_code: int) -> int:
    payload: dict[str, Any] = {
        "status": "error",
        "error_type": kind,
        "message": str(exc),
    }
    diagnostic = _diagnostic(exc)
    if diagnostic is not None:
        payload["diagnostic"] = diagnostic
    print(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
        file=sys.stderr,
    )
    return exit_code


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="PLANNER_INPUT_JSON",
        help="OFFLINE_ARB_V1 planner_input.json (contract version 1)",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="PLAN_ROOT",
        help="output directory for header.json, blocks/, and checksums.json",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        metavar="STEP",
        help="rebuild from STEP; rewinds to its block boundary so common suffix block hashes stay identical",
    )
    return parser.parse_args(argv)


def _planner_api() -> tuple[Any, type[BaseException], type[BaseException]]:
    """Import lazily so ``--help`` remains usable in a partial checkout."""

    from training.dataline_offline.planner_v1 import (
        PlanInfeasibleError,
        PlannerInputError,
        build_plan_from_file,
    )

    return build_plan_from_file, PlannerInputError, PlanInfeasibleError


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        build_plan_from_file, PlannerInputError, PlanInfeasibleError = _planner_api()
    except (ImportError, AttributeError) as exc:
        return _print_error("PlannerImportError", exc, exit_code=1)

    try:
        header = build_plan_from_file(args.input, args.out, start_step=args.start_step)
    except PlannerInputError as exc:
        return _print_error("PlannerInputError", exc, exit_code=2)
    except PlanInfeasibleError as exc:
        return _print_error("PlanInfeasibleError", exc, exit_code=2)
    except (FileNotFoundError, IsADirectoryError, PermissionError, json.JSONDecodeError) as exc:
        # Keep malformed/missing contract input failures friendly even if they
        # arise before planner_v1 can normalize them to PlannerInputError.
        return _print_error(type(exc).__name__, exc, exit_code=2)

    if not isinstance(header, Mapping):
        return _print_error(
            "PlannerInternalError",
            TypeError("build_plan_from_file() did not return a header mapping"),
            exit_code=1,
        )

    print(
        json.dumps(
            _header_summary(header, args.out),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
