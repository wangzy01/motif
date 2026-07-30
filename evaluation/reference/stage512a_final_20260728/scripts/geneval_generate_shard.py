#!/usr/bin/env python3
"""Generate one deterministic modulo shard of the frozen 512px GenEval suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SOURCE = Path(
    "/fsx/evan/mot_code/data_analysis/eval_releases/"
    "stage256_offline_exact_v1_vs_prev_final_20260718a/scripts/strict_eval.py"
)
EXPECTED_SOURCE_SHA256 = "0afc7e6a8c0b9a9751ff9c7ee48edbdedc6c3b0847169a618d3ad3569235babc"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
    raise RuntimeError("strict evaluator SHA drift")
text = SOURCE.read_text(encoding="utf-8")
replacements = {
    "[x[\"prompt\"] for x in chunk], height=256, width=256,":
        "[x[\"prompt\"] for x in chunk], height=512, width=512,",
    "[record[\"prompt\"]] * 4, height=256, width=256,":
        "[record[\"prompt\"]] * 4, height=512, width=512,",
    '"height": 256,\n            "width": 256,':
        '"height": 512,\n            "width": 512,',
    '(("height", 256), ("width", 256), ("steps", 64), ("seed", 7))':
        '(("height", 512), ("width", 512), ("steps", 64), ("seed", 7))',
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise RuntimeError(f"strict evaluator patch anchor drift: {old!r}")
    text = text.replace(old, new)
text = text.replace("stage256_eval_20260718a_v1", "stage512a_final_eval_20260728_v1")
namespace = {
    "__name__": "stage512a_final_geneval_shard_runtime",
    "__file__": str(Path(__file__).resolve()),
}
exec(compile(text, str(SOURCE), "exec"), namespace)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--expect-adapter", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--expect-mlf", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.num_shards != 8 or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError("formal GenEval sharding requires exactly 8 modulo shards")

    metadata_path = Path(args.metadata).resolve()
    namespace["require_file"](metadata_path, args.metadata_sha256)
    metadata = namespace["read_jsonl"](metadata_path)
    if len(metadata) != 553:
        raise RuntimeError(f"GenEval metadata cardinality={len(metadata)} expected=553")
    model, identity, torch, np = namespace["load_model"](args)
    out = Path(args.output).resolve()
    namespace["new_output"](out)
    samples = out / "samples"
    samples.mkdir()
    assigned = [index for index in range(553) if index % args.num_shards == args.shard_index]
    for index in assigned:
        record = metadata[index]
        prompt_dir = samples / f"{index:05d}"
        image_dir = prompt_dir / "samples"
        image_dir.mkdir(parents=True)
        with (prompt_dir / "metadata.jsonl").open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        px = model.generate_samples(
            [record["prompt"]] * 4,
            height=512,
            width=512,
            num_steps=64,
            guidance_scale=4.0,
            seed=7 + index * 1000,
        )
        if int(px.shape[0]) != 4:
            raise RuntimeError(f"GenEval prompt {index}: generator returned {px.shape[0]} images")
        for sample_index in range(4):
            namespace["tensor_to_pil"](px[sample_index], np).save(
                image_dir / f"{sample_index:04d}.png"
            )
        del px
        torch.cuda.empty_cache()

    actual = sorted(int(path.name) for path in samples.iterdir())
    if actual != assigned:
        raise RuntimeError(f"GenEval shard coverage drift: {actual[:8]} != {assigned[:8]}")
    manifest = namespace["common_manifest"](args, identity, "geneval_generation_shard")
    manifest.update({
        "metadata": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "prompt_indices": assigned,
        "n_prompts": len(assigned),
        "n_images_per_prompt": 4,
        "n_images": len(assigned) * 4,
    })
    namespace["write_json"](out / "manifest.json", manifest)
    (out / "_SUCCESS").open("x").close()
    print(json.dumps({
        "status": "SUCCESS",
        "run_key": args.run_key,
        "shard_index": args.shard_index,
        "n_prompts": len(assigned),
        "n_images": len(assigned) * 4,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
