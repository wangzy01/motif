#!/usr/bin/env python3
"""Merge and independently validate eight Stage512a Qwen rubric shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


SOURCE = Path(
    "/fsx/evan/mot_code/data_analysis/eval_releases/"
    "stage256_offline_exact_v1_vs_prev_final_20260718a/score_multiar_qwen.py"
)
EXPECTED_SOURCE_SHA256 = "e223e6861fe7e802cd50d4ccdd5704c72261211b360a8056fed0bbc771c86dec"
RUNS = {"weighted_final_s125000", "streaming_final_s166667"}
SIZES = {
    "512x512", "544x448", "448x544", "608x416",
    "416x608", "672x384", "384x672",
}
SEEDS = {7, 1007}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
    raise RuntimeError("frozen multi-AR scorer SHA drift")
source_text = SOURCE.read_text(encoding="utf-8")
namespace = {
    "__name__": "stage512a_final_multiar_merge_runtime",
    "__file__": str(Path(__file__).resolve()),
}
exec(compile(source_text, str(SOURCE), "exec"), namespace)


def fail(message: str) -> "NoReturn":
    raise RuntimeError(message)


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing regular nonempty JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing regular nonempty JSONL: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    fail(f"JSONL row is not an object: {path}:{line_no}")
                rows.append(value)
    return rows


def write_json_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def same(values: list, label: str):
    if not values or any(value != values[0] for value in values[1:]):
        fail(f"{label} differs across score shards")
    return values[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        fail(f"canonical score output already exists: {args.output}")

    manifests = []
    scores_by_global_index = {}
    for rank in range(8):
        root = args.shard_root / f"rank{rank}"
        marker = root / "_SUCCESS"
        if marker.is_symlink() or not marker.is_file() or marker.stat().st_size != 0:
            fail(f"invalid score-shard success marker: {marker}")
        manifest = read_json(root / "manifest.json")
        scores_path = root / "scores.jsonl"
        rows = read_jsonl(scores_path)
        expected_indices = list(range(rank, 1428, 8))
        if (
            manifest.get("schema") != "stage512a-final-qwen-rubric-shard-v1"
            or manifest.get("status") != "COMPLETE"
            or manifest.get("num_shards") != 8
            or manifest.get("shard_index") != rank
            or manifest.get("global_record_indices") != expected_indices
            or manifest.get("n_scores") != len(expected_indices)
            or manifest.get("scores_sha256") != sha256(scores_path)
            or len(rows) != len(expected_indices)
        ):
            fail(f"score-shard manifest/coverage mismatch: rank{rank}")
        for global_index, row in zip(expected_indices, rows):
            if global_index in scores_by_global_index:
                fail(f"duplicate global score index: {global_index}")
            scores_by_global_index[global_index] = row
        manifests.append(manifest)
    if set(scores_by_global_index) != set(range(1428)):
        fail("merged score indices are not exactly 0..1427")
    scores = [scores_by_global_index[index] for index in range(1428)]

    keys = []
    for row in scores:
        if (
            row.get("run") not in RUNS
            or row.get("size") not in SIZES
            or row.get("seed") not in SEEDS
            or not isinstance(row.get("prompt_id"), str)
            or not row["prompt_id"]
        ):
            fail(f"invalid merged score identity: {row}")
        namespace["validate_score"]({
            key: row[key]
            for key in (
                "primary_label", "head_cut", "foot_cut", "side_cut",
                "full_body", "multi_person", "subject_complete_0to5",
                "composition_follow_0to5", "notes",
            )
        })
        keys.append((row["run"], row["prompt_id"], row["seed"], row["size"]))
    if len(set(keys)) != 1428:
        fail("merged score grid contains duplicate keys")
    prompt_ids = {key[1] for key in keys}
    expected = {
        (run, prompt_id, seed, size)
        for run in RUNS
        for prompt_id in prompt_ids
        for seed in SEEDS
        for size in SIZES
    }
    if len(prompt_ids) != 51 or set(keys) != expected:
        fail("merged score grid is not exact 2x51x2x7 coverage")

    work = args.output.with_name(args.output.name + ".work")
    if work.exists() or work.is_symlink():
        fail(f"score merge work namespace already exists: {work}")
    work.mkdir(parents=True, exist_ok=False)
    scores_path = work / "multiar_scores.jsonl"
    with scores_path.open("x", encoding="utf-8") as handle:
        for row in scores:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    provenance = same([manifest["provenance"] for manifest in manifests], "provenance")
    model_path = same([manifest["model_path"] for manifest in manifests], "model path")
    model_files = same(
        [manifest["model_file_sha256"] for manifest in manifests],
        "model file identities",
    )
    result = {
        "schema": "stage512a-final-paired-qwen-rubric-v1",
        "status": "COMPLETE",
        "scorer": "Qwen3-VL-2B-Instruct",
        "comparability": "paired Stage512a final comparison; identical prompts/seeds/protocol",
        "coverage": {"expected": 1428, "scored": 1428, "missing": 0, "extra": 0},
        "provenance": provenance,
        "model_path": model_path,
        "model_file_sha256": model_files,
        "scorer_sha256": same(
            [manifest["runner_sha256"] for manifest in manifests],
            "shard runner SHA",
        ),
        "scores_path": str(args.output / "multiar_scores.jsonl"),
        "scores_sha256": sha256(scores_path),
        "summary": namespace["summarize"](scores),
        "completed_unix": time.time(),
        "merge": {
            "schema": "stage512a-final-qwen-rubric-merge-v1",
            "shard_manifest_sha256": [
                sha256(args.shard_root / f"rank{rank}" / "manifest.json")
                for rank in range(8)
            ],
        },
    }
    write_json_new(work / "summary.json", result)
    write_json_new(
        work / "SUCCESS.json",
        {
            "status": "SUCCESS",
            "scored": 1428,
            "summary_sha256": sha256(work / "summary.json"),
        },
    )
    os.replace(work, args.output)


if __name__ == "__main__":
    main()
