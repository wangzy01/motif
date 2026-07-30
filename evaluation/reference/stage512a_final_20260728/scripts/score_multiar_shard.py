#!/usr/bin/env python3
"""Score one deterministic modulo shard of the frozen 1,428-image Qwen rubric."""

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


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
    raise RuntimeError("frozen multi-AR scorer SHA drift")
text = SOURCE.read_text(encoding="utf-8")
patches = {
    'RUN_STEPS = {"current_final": 166667, "previous_final": 132000}':
        'RUN_STEPS = {"weighted_final_s125000": 125000, '
        '"streaming_final_s166667": 166667}',
    "SIZES = ((256, 256), (272, 224), (224, 272), (304, 208),\n"
    "         (208, 304), (336, 192), (192, 336))":
        "SIZES = ((512, 512), (544, 448), (448, 544), (608, 416),\n"
        "         (416, 608), (672, 384), (384, 672))",
}
for old, new in patches.items():
    if text.count(old) != 1:
        raise RuntimeError(f"multi-AR scorer patch anchor drift: {old!r}")
    text = text.replace(old, new)
namespace = {
    "__name__": "stage512a_final_multiar_shard_runtime",
    "__file__": str(Path(__file__).resolve()),
}
exec(compile(text, str(SOURCE), "exec"), namespace)


def write_json_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    if args.num_shards != 8 or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError("formal Qwen sharding requires exactly 8 modulo shards")
    if args.out_dir.exists() or args.out_dir.is_symlink():
        raise RuntimeError(f"output namespace already exists: {args.out_dir}")

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Qwen scorer requires exactly one visible GPU: {torch.cuda.device_count()}")
    all_records, provenance = namespace["load_and_validate_manifests"](args.manifests)
    records = all_records[args.shard_index::args.num_shards]
    expected_indices = list(range(args.shard_index, len(all_records), args.num_shards))
    if len(records) != len(expected_indices):
        raise RuntimeError("internal score-shard cardinality bug")
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        raise RuntimeError(f"invalid local Qwen model directory: {args.model}")
    model_files = {}
    for name in ("config.json", "preprocessor_config.json", "processor_config.json"):
        path = args.model / name
        if path.is_file():
            model_files[name] = namespace["sha256_path"](path)

    judge = namespace["Judge"](args.model)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    scores_path = args.out_dir / "scores.jsonl"
    from PIL import Image
    scores = []
    with scores_path.open("x", encoding="utf-8") as out:
        for local_index, row in enumerate(records, 1):
            full_req, multi_req = namespace["requested_flags"](
                str(row.get("composition_target", "")), row["prompt"]
            )
            with Image.open(row["path"]) as image:
                image = image.convert("RGB")
                last_error = ""
                for attempt in range(3):
                    correction = "" if attempt == 0 else (
                        f"Your prior reply was invalid ({last_error}). Return valid JSON only."
                    )
                    raw = judge.ask(
                        image,
                        namespace["rubric_prompt"](row, full_req, multi_req, correction),
                    )
                    try:
                        judged = namespace["validate_score"](
                            namespace["parse_json_object"](raw)
                        )
                        break
                    except Exception as error:
                        last_error = str(error)
                else:
                    raise RuntimeError(
                        f"unparseable judge result for {row['path']}: {last_error}"
                    )
            score = {
                "image": row["path"],
                "run": row["run"],
                "prompt_id": row["id"],
                "prompt_style": row.get("prompt_style", "mixed"),
                "domain": row.get("domain", "anime"),
                "composition_target": row.get("composition_target", "unknown"),
                "size": f"{row['width']}x{row['height']}",
                "seed": row["seed"],
                "full_body_requested": full_req,
                "multi_person_requested": multi_req,
                "judge_attempts": attempt + 1,
                "judge_raw": raw,
                **judged,
            }
            out.write(json.dumps(score, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            scores.append(score)
            if local_index % 25 == 0:
                print(
                    f"[rubric shard {args.shard_index}] {local_index}/{len(records)}",
                    flush=True,
                )
        os.fsync(out.fileno())
    manifest = {
        "schema": "stage512a-final-qwen-rubric-shard-v1",
        "status": "COMPLETE",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "global_record_indices": expected_indices,
        "n_scores": len(scores),
        "scores_path": str(scores_path),
        "scores_sha256": sha256(scores_path),
        "provenance": provenance,
        "model_path": str(args.model),
        "model_file_sha256": model_files,
        "runner_sha256": sha256(Path(__file__)),
        "completed_unix": time.time(),
    }
    write_json_new(args.out_dir / "manifest.json", manifest)
    (args.out_dir / "_SUCCESS").open("x").close()


if __name__ == "__main__":
    main()
