#!/usr/bin/env python3
"""Fail-closed merge of eight deterministic Stage512a GenEval shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


METADATA_SHA256 = "5c48e0813e812e3c373fa5c8ed07a8f0a483be30272b4427b0559c8048e67c13"
RUN_STEPS = {
    "weighted_final_s125000": 125000,
    "streaming_final_s166667": 166667,
}
MANIFEST_KEYS = {
    "schema_version", "kind", "run_key", "identity", "sampling", "metadata",
    "metadata_sha256", "n_prompts", "n_images_per_prompt", "n_images",
}


def fail(message: str) -> "NoReturn":
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


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
    parser.add_argument("--run-key", choices=sorted(RUN_STEPS), required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        fail(f"canonical output already exists: {args.output}")

    manifests = []
    coverage = set()
    for rank in range(8):
        root = args.shard_root / f"rank{rank}"
        success = root / "_SUCCESS"
        if success.is_symlink() or not success.is_file() or success.stat().st_size != 0:
            fail(f"invalid shard success marker: {success}")
        manifest = read_json(root / "manifest.json")
        assigned = [index for index in range(553) if index % 8 == rank]
        if (
            manifest.get("schema_version") != "stage512a_final_eval_20260728_v1"
            or manifest.get("kind") != "geneval_generation_shard"
            or manifest.get("run_key") != args.run_key
            or manifest.get("num_shards") != 8
            or manifest.get("shard_index") != rank
            or manifest.get("prompt_indices") != assigned
            or manifest.get("n_prompts") != len(assigned)
            or manifest.get("n_images_per_prompt") != 4
            or manifest.get("n_images") != len(assigned) * 4
            or manifest.get("metadata_sha256") != METADATA_SHA256
        ):
            fail(f"GenEval shard manifest mismatch: rank{rank}")
        if int(manifest["identity"]["global_step"]) != RUN_STEPS[args.run_key]:
            fail(f"GenEval shard step mismatch: rank{rank}")
        manifests.append((root, manifest))
        coverage.update(assigned)
    if coverage != set(range(553)):
        fail("GenEval shard union is not exactly prompts 0..552")

    identity = same([item[1]["identity"] for item in manifests], "model identity")
    sampling = same([item[1]["sampling"] for item in manifests], "sampling protocol")
    metadata_path = same([item[1]["metadata"] for item in manifests], "metadata path")
    if sampling != {
        "height": 512,
        "width": 512,
        "steps": 64,
        "cfg": 4.0,
        "seed": 7,
        "weights": "ema",
    }:
        fail(f"GenEval sampling protocol mismatch: {sampling}")

    work = args.output.with_name(args.output.name + ".work")
    if work.exists() or work.is_symlink():
        fail(f"merge work namespace already exists: {work}")
    samples = work / "samples"
    samples.mkdir(parents=True, exist_ok=False)
    for root, manifest in manifests:
        for index in manifest["prompt_indices"]:
            source_prompt = root / "samples" / f"{index:05d}"
            target_prompt = samples / f"{index:05d}"
            target_images = target_prompt / "samples"
            target_images.mkdir(parents=True)
            source_metadata = source_prompt / "metadata.jsonl"
            if source_metadata.is_symlink() or not source_metadata.is_file():
                fail(f"invalid shard metadata: {source_metadata}")
            os.link(source_metadata, target_prompt / "metadata.jsonl")
            for sample_index in range(4):
                source_image = source_prompt / "samples" / f"{sample_index:04d}.png"
                if (
                    source_image.is_symlink()
                    or not source_image.is_file()
                    or source_image.stat().st_size <= 0
                ):
                    fail(f"invalid shard image: {source_image}")
                os.link(source_image, target_images / source_image.name)

    if sorted(path.name for path in samples.iterdir()) != [
        f"{index:05d}" for index in range(553)
    ]:
        fail("merged GenEval prompt-directory coverage mismatch")
    manifest = {
        "schema_version": "stage512a_final_eval_20260728_v1",
        "kind": "geneval_generation",
        "run_key": args.run_key,
        "identity": identity,
        "sampling": sampling,
        "metadata": metadata_path,
        "metadata_sha256": METADATA_SHA256,
        "n_prompts": 553,
        "n_images_per_prompt": 4,
        "n_images": 2212,
    }
    if set(manifest) != MANIFEST_KEYS:
        fail("internal canonical manifest schema bug")
    write_json_new(work / "manifest.json", manifest)
    (work / "_SUCCESS").open("x").close()
    os.replace(work, args.output)
    print(json.dumps({
        "status": "SUCCESS",
        "run_key": args.run_key,
        "n_prompts": 553,
        "n_images": 2212,
        "manifest_sha256": sha256(args.output / "manifest.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
