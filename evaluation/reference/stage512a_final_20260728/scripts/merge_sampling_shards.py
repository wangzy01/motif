#!/usr/bin/env python3
"""Fail-closed merge of eight immutable Stage512a paired-sampling shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path


SIZES = (
    (512, 512), (544, 448), (448, 544), (608, 416),
    (416, 608), (672, 384), (384, 672),
)
SEEDS = (7, 1007)
ANIME_SEEDS = (7, 1007, 2007)
RUN_STEPS = {
    "weighted_final_s125000": 125000,
    "streaming_final_s166667": 166667,
}


def fail(message: str) -> "NoReturn":
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing regular nonempty JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def write_json_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def same(values: list, label: str):
    if not values or any(value != values[0] for value in values[1:]):
        fail(f"{label} differs across shards")
    return values[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=sorted(RUN_STEPS), required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() or args.output.is_symlink():
        fail(f"canonical output already exists: {args.output}")
    manifests = []
    for rank in range(8):
        root = args.shard_root / f"rank{rank}"
        success = read_json(root / "SUCCESS.json")
        manifest_path = root / "sample_manifest.json"
        manifest = read_json(manifest_path)
        if success != {
            "status": "SUCCESS",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "n_images": manifest.get("n_images"),
            "run": args.run,
        }:
            fail(f"shard SUCCESS identity mismatch: rank{rank}")
        if (
            manifest.get("schema") != "stage256-two-final-shard-manifest-v1"
            or manifest.get("status") != "COMPLETE"
            or manifest.get("run") != args.run
            or manifest.get("checkpoint_step") != RUN_STEPS[args.run]
            or manifest.get("weights_src") != "ema"
        ):
            fail(f"shard identity/provenance mismatch: rank{rank}")
        expected_suite = "anime" if rank == 7 else "multiar"
        expected_count = 450 if rank == 7 else 102
        if manifest.get("suite") != expected_suite or manifest.get("n_images") != expected_count:
            fail(f"shard suite/coverage mismatch: rank{rank}")
        manifests.append((root, manifest))

    checkpoint_identity = same(
        [manifest["checkpoint_identity"] for _, manifest in manifests],
        "checkpoint identity",
    )
    if not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint_identity.get("sha256", ""))):
        fail("checkpoint SHA-256 identity missing")
    config_identity = same(
        [manifest["config_identity"] for _, manifest in manifests],
        "config identity",
    )
    load_audit = same(
        [manifest["load_audit"] for _, manifest in manifests],
        "strict EMA load audit",
    )
    if (
        load_audit.get("weights_src") != "ema"
        or load_audit.get("trainable_missing") != 0
        or load_audit.get("unexpected") != 0
    ):
        fail("non-strict EMA load audit")

    work = args.output.with_name(args.output.name + ".work")
    if work.exists() or work.is_symlink():
        fail(f"merge work namespace already exists: {work}")
    (work / "images").mkdir(parents=True, exist_ok=False)
    (work / "manifests").mkdir()

    images = []
    prompt_sets = {}
    seen = set()
    anime_source = None
    for root, manifest in manifests:
        prompt_sets.update(manifest.get("prompt_sets", {}))
        if manifest["suite"] == "anime":
            anime_source = (root, manifest)
        for source_record in manifest["images"]:
            record = dict(source_record)
            suite = record.get("suite")
            width, height = int(record.get("width", -1)), int(record.get("height", -1))
            seed = int(record.get("seed", -1))
            key = (suite, record.get("id"), seed, width, height)
            if key in seen:
                fail(f"duplicate image cell: {key}")
            seen.add(key)
            if suite == "multiar":
                if (width, height) not in SIZES or seed not in SEEDS:
                    fail(f"invalid multi-AR image protocol: {key}")
            elif suite == "anime":
                if (width, height) != (512, 512) or seed not in ANIME_SEEDS:
                    fail(f"invalid anime image protocol: {key}")
            else:
                fail(f"unknown image suite: {suite!r}")
            source_path = Path(record["path"])
            if source_path.is_symlink() or not source_path.is_file() or source_path.stat().st_size <= 0:
                fail(f"invalid shard image: {source_path}")
            target_dir = work / "images" / suite / f"{width}x{height}" / args.run
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / source_path.name
            if target_path.exists() or target_path.is_symlink():
                fail(f"duplicate target image: {target_path}")
            os.link(source_path, target_path)
            record["path"] = str(args.output / target_path.relative_to(work))
            images.append(record)

    if len(images) != 1164:
        fail(f"merged image coverage={len(images)} expected=1164")
    multiar = [record for record in images if record["suite"] == "multiar"]
    anime = [record for record in images if record["suite"] == "anime"]
    if len(multiar) != 714 or len(anime) != 450:
        fail(f"suite coverage mismatch: multiar={len(multiar)} anime={len(anime)}")
    if set(tuple(value) for value in SIZES) != {
        (record["width"], record["height"]) for record in multiar
    }:
        fail("merged multi-AR size matrix incomplete")
    if set(prompt_sets) != {"multiar", "anime"}:
        fail(f"prompt-set union mismatch: {sorted(prompt_sets)}")

    if anime_source is None:
        fail("anime shard missing")
    anime_legacy = read_json(anime_source[0] / "manifests" / "anime_512x512.json")
    if not isinstance(anime_legacy.get("runs"), list) or len(anime_legacy["runs"]) != 1:
        fail("anime legacy run cardinality mismatch")
    anime_legacy["runs"][0]["images"] = anime
    anime_legacy["runs"][0]["checkpoint_identity"] = checkpoint_identity
    anime_legacy["runs"][0]["config_identity"] = config_identity
    anime_legacy["runs"][0]["load_audit"] = load_audit
    write_json_new(work / "manifests" / "anime_512x512.json", anime_legacy)

    sampling = {
        "steps": 64,
        "cfg": 4.0,
        "multiar_batch_size": 4,
        "anime_batch_size": 8,
        "multiar_sizes": [list(size) for size in SIZES],
        "multiar_seeds": list(SEEDS),
        "anime_seeds": list(ANIME_SEEDS),
    }
    merged = {
        "schema": "stage256-two-final-shard-manifest-v1",
        "status": "COMPLETE",
        "run": args.run,
        "run_name": manifests[0][1]["run_name"],
        "suite": "both",
        "checkpoint_step": RUN_STEPS[args.run],
        "weights_src": "ema",
        "exposure_per_step": manifests[0][1]["exposure_per_step"],
        "exposure_estimate": manifests[0][1]["exposure_estimate"],
        "load_audit": load_audit,
        "checkpoint_identity": checkpoint_identity,
        "config_identity": config_identity,
        "prompt_sets": prompt_sets,
        "sampling": sampling,
        "n_images": len(images),
        "images": images,
        "started_unix": min(manifest["started_unix"] for _, manifest in manifests),
        "completed_unix": max(manifest["completed_unix"] for _, manifest in manifests),
        "runner_sha256": same(
            [manifest["runner_sha256"] for _, manifest in manifests],
            "runner SHA",
        ),
        "merge": {
            "schema": "stage512a-final-sampling-merge-v1",
            "completed_unix": time.time(),
            "shard_manifests": [
                {
                    "path": str(root / "sample_manifest.json"),
                    "sha256": sha256(root / "sample_manifest.json"),
                }
                for root, _ in manifests
            ],
            "hardlinked_images": True,
        },
    }
    manifest_path = work / "sample_manifest.json"
    write_json_new(manifest_path, merged)
    manifest_sha = sha256(manifest_path)
    write_json_new(
        work / "SUCCESS.json",
        {
            "status": "SUCCESS",
            "manifest": str(args.output / "sample_manifest.json"),
            "manifest_sha256": manifest_sha,
            "n_images": 1164,
            "run": args.run,
        },
    )
    os.replace(work, args.output)
    print(json.dumps({
        "status": "SUCCESS",
        "run": args.run,
        "manifest_sha256": manifest_sha,
        "n_images": 1164,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
