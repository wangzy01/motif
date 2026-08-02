#!/usr/bin/env python3
"""CLI wrapper for the weighted Stage512a offline plan writer."""

from __future__ import annotations

import argparse
import json
import sys

from training.dataline_offline.weighted_plan_v1 import build_weighted_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-input", required=True)
    parser.add_argument("--weighted-contract", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--allow-canary", action="store_true")
    args = parser.parse_args()
    try:
        header = build_weighted_plan(
            args.planner_input,
            args.weighted_contract,
            args.out,
            start_step=args.start_step,
            allow_canary=args.allow_canary,
        )
    except Exception as exc:
        print(f"WEIGHTED_PLAN_BUILD_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        "WEIGHTED_PLAN_BUILD_PASS "
        + json.dumps(
            {
                "plan_id": header["plan_id"],
                "plan_hash": header["plan_hash"],
                "total_exposures": header["total_exposures"],
                "blocks": header["num_blocks"],
                "weighted_collision_swaps": header["materialization"][
                    "weighted_collision_swaps"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
