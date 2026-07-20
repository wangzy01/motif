#!/usr/bin/env python3
"""Create an OFFLINE_ARB_V1 planner input from the verified semantic overlay.

The legacy physical catalog is *not* a valid source of ``N`` for this release:
Stage256 strict-NL filtering reduces anime cells.  This tool reconstructs the
bucket/domain/mix contract from the approved config and sampling JSON, then
uses ``overlay.header.N_select`` as the selection-plan capacity matrix.  Since
the v1 planner schema rejects extension keys, an adjacent create-once binding
receipt transitively binds the exact planner-input bytes to the overlay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage256_20260718.semantic_plan import load_overlay_header  # noqa: E402
from training.dataline.config import load_sampling_config  # noqa: E402
from training.dataline.multi_ar import build_adaptive_buckets, square_index  # noqa: E402
from training.dataline_offline.planner_v1 import load_planner_input  # noqa: E402


DOMAINS = ("anime", "monet_real", "gpic", "synthetic", "textatlas")
R_MAX = {
    "anime": 20,
    "monet_real": 20,
    "gpic": 20,
    "synthetic": 200,
    "textatlas": 200,
}
EXACT_FORMAL_STEPS = 166_667
EXACT_FORMAL_EXPOSURES = 384_000_768


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists():
        raise FileExistsError(f"refusing to replace create-once output: {path}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def verify_overlay_publication(overlay_root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    header_path = overlay_root / "header.json"
    success_path = overlay_root / "_SUCCESS"
    if not header_path.is_file() or header_path.is_symlink():
        raise FileNotFoundError(f"overlay header missing/non-regular: {header_path}")
    if not success_path.is_file() or success_path.is_symlink():
        raise FileNotFoundError(f"overlay _SUCCESS missing/non-regular: {success_path}")
    header = load_overlay_header(overlay_root)
    success = read_json(success_path)
    header_sha = sha256_file(header_path)
    if success.get("header_sha256") != header_sha:
        raise ValueError("overlay _SUCCESS/header SHA mismatch")
    content = dict(header)
    declared_content_sha = content.pop("content_sha256", None)
    computed_content_sha = hashlib.sha256(canonical_bytes(content)).hexdigest()
    if declared_content_sha != computed_content_sha:
        raise ValueError("overlay header content_sha256 mismatch")
    if success.get("content_sha256") != computed_content_sha:
        raise ValueError("overlay _SUCCESS/content SHA mismatch")
    success_without_digest = dict(success)
    declared_success_sha = success_without_digest.pop("success_sha256", None)
    computed_success_sha = hashlib.sha256(canonical_bytes(success_without_digest)).hexdigest()
    if declared_success_sha != computed_success_sha:
        raise ValueError("overlay _SUCCESS self hash mismatch")
    if header["semantic_assertions"] != {
        "anime_all_rows_strip_nonempty_nl": True,
        "anime_tag_only_rows": 0,
    }:
        raise ValueError("overlay strict-NL/tag-only contract drift")
    return header, success, header_sha


def build(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay, overlay_success, overlay_header_sha = verify_overlay_publication(args.overlay)
    catalog = read_json(args.catalog_header)
    catalog_header_sha = sha256_file(args.catalog_header)
    if overlay["catalog_header_sha256"] != catalog_header_sha:
        raise ValueError("overlay binds a different catalog header")
    if overlay["catalog_hash"] != catalog.get("catalog_hash"):
        raise ValueError("overlay/catalog hash mismatch")
    if catalog.get("diagnostic_run") is not False:
        raise ValueError("catalog is diagnostic, not production")
    if tuple(catalog.get("domain_order", ())) != DOMAINS:
        raise ValueError("catalog domain order drift")

    config = OmegaConf.load(args.config)
    mar = config.dataset.multi_ar
    buckets = build_adaptive_buckets(
        int(mar.target_area),
        int(mar.divisor),
        min_area_ratio=float(mar.adaptive_min_area_ratio),
        max_aspect=float(mar.adaptive_max_aspect),
    )
    if len(buckets) != 95:
        raise ValueError(f"expected 95 adaptive buckets, got {len(buckets)}")
    bucket_lists = [list(map(int, item)) for item in buckets]
    if overlay["buckets"] != bucket_lists:
        raise ValueError("overlay bucket table differs from config reconstruction")
    catalog_buckets = []
    for expected, item in enumerate(catalog.get("buckets", [])):
        if int(item.get("idx", -1)) != expected:
            raise ValueError(f"catalog bucket index drift at {expected}")
        catalog_buckets.append([int(item["h"]), int(item["w"])])
    if catalog_buckets != bucket_lists:
        raise ValueError("catalog bucket table differs from config reconstruction")
    square = int(square_index(buckets))
    square_domains = [str(item) for item in mar.square_only_domains]

    sampling_path = Path(str(config.dataset.sampling_config))
    if sampling_path.resolve(strict=False) != args.sampling_config.resolve(strict=False):
        raise ValueError("config sampling path differs from explicit sampling input")
    sampling = load_sampling_config(str(args.sampling_config))
    mix = {name: float(sampling.domain_mix["256"][name]) for name in DOMAINS}
    if mix != {
        "anime": 0.27,
        "monet_real": 0.405,
        "gpic": 0.259,
        "synthetic": 0.051,
        "textatlas": 0.015,
    }:
        raise ValueError(f"Stage256 domain mix drift: {mix}")
    if tuple(overlay["domains"]) != DOMAINS:
        raise ValueError("overlay domain order drift")

    n_select = np.asarray(overlay["N_select"], dtype=np.int64)
    if n_select.shape != (95, 5) or np.any(n_select < 0):
        raise ValueError(f"overlay N_select shape/value drift: {n_select.shape}")
    physical = np.asarray(catalog["N"]["materialized_after_exclusion"], dtype=np.int64)
    if physical.shape != n_select.shape or np.any(n_select > physical):
        raise ValueError("overlay N_select exceeds/mismatches physical catalog N")
    for domain_index, name in enumerate(DOMAINS):
        if name in square_domains:
            non_square = np.delete(n_select[:, domain_index], square)
            if np.any(non_square != 0):
                raise ValueError(f"square-only domain has non-square N_select: {name}")

    planner_input = {
        "format_version": 1,
        "plan_id": args.plan_id,
        "seed": int(args.seed),
        "world_size": int(args.world_size),
        "per_rank_bs": int(args.per_rank_bs),
        "total_steps": int(args.total_steps),
        "domains": list(DOMAINS),
        "domain_mix": mix,
        "square_only_domains": square_domains,
        "buckets": bucket_lists,
        "square_bucket_idx": square,
        "N": n_select.tolist(),
        "R_max": dict(R_MAX),
        "epsilon_jitter": False,
        "block_steps": int(args.block_steps),
        # These continue to bind the immutable physical row namespace.  The
        # overlay binding lives in the adjacent companion receipt.
        "catalog_hash": str(catalog["catalog_hash"]),
        "assignment_hash": str(catalog["assignment_hash"]),
    }
    if args.world_size == 64:
        if args.per_rank_bs != 36 or args.total_steps != EXACT_FORMAL_STEPS:
            raise ValueError("W64 formal must be bs36/H166667")
        if args.total_steps * args.world_size * args.per_rank_bs != EXACT_FORMAL_EXPOSURES:
            raise ValueError("W64 exact exposure contract drift")
    elif args.world_size == 8:
        if args.per_rank_bs != 36 or args.total_steps != 32:
            raise ValueError("W8 smoke selection plan must be bs36/H32")
    else:
        raise ValueError("this release permits only W8 smoke and W64 formal inputs")

    planner_payload = json.dumps(
        planner_input, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    planner_sha = hashlib.sha256(planner_payload).hexdigest()
    companion = {
        "format": "stage256_20260718_semantic_planner_input_binding_v1",
        "status": "PASS",
        "plan_id": args.plan_id,
        "planner_input_sha256": planner_sha,
        "overlay_id": overlay["overlay_id"],
        "overlay_header_sha256": overlay_header_sha,
        "overlay_content_sha256": overlay["content_sha256"],
        "overlay_success_sha256": sha256_file(args.overlay / "_SUCCESS"),
        "catalog_hash": catalog["catalog_hash"],
        "catalog_assignment_hash": catalog["assignment_hash"],
        "catalog_header_sha256": catalog_header_sha,
        "sampling_config_sha256": sha256_file(args.sampling_config),
        "config_sha256": sha256_file(args.config),
        "n_select_sha256": hashlib.sha256(canonical_bytes(n_select.tolist())).hexdigest(),
        "physical_n_sha256": hashlib.sha256(canonical_bytes(physical.tolist())).hexdigest(),
        "n_select_rows": int(n_select.sum(dtype=np.int64)),
        "physical_rows": int(physical.sum(dtype=np.int64)),
        "anime_tag_only_exposure": 0,
        "world_size": args.world_size,
        "per_rank_bs": args.per_rank_bs,
        "total_steps": args.total_steps,
        "total_exposures": args.total_steps * args.world_size * args.per_rank_bs,
    }
    return planner_input, {"payload": planner_payload, "receipt": companion}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--catalog-header", required=True, type=Path)
    parser.add_argument("--sampling-config", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--per-rank-bs", required=True, type=int)
    parser.add_argument("--total-steps", required=True, type=int)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--block-steps", type=int, default=1024)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binding-output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _input, products = build(args)
    atomic_create(args.output, products["payload"])
    # Reparse through the actual frozen planner before publishing the binding.
    parsed = load_planner_input(args.output)
    if parsed.input_sha256 != products["receipt"]["planner_input_sha256"]:
        raise RuntimeError("planner parser input SHA differs from companion binding")
    receipt_payload = json.dumps(
        products["receipt"], ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    atomic_create(args.binding_output, receipt_payload)
    print(json.dumps(products["receipt"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
