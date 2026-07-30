#!/usr/bin/env python3
"""Run the frozen Qwen crop/composition rubric over the paired 512px grid."""

from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE = Path(
    "/fsx/evan/mot_code/data_analysis/eval_releases/"
    "stage256_offline_exact_v1_vs_prev_final_20260718a/score_multiar_qwen.py"
)
EXPECTED_SOURCE_SHA256 = "e223e6861fe7e802cd50d4ccdd5704c72261211b360a8056fed0bbc771c86dec"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
    raise RuntimeError("frozen multi-AR scorer SHA drift")
text = SOURCE.read_text(encoding="utf-8")
replacements = {
    'RUN_STEPS = {"current_final": 166667, "previous_final": 132000}':
        'RUN_STEPS = {"weighted_final_s125000": 125000, '
        '"streaming_final_s166667": 166667}',
    "SIZES = ((256, 256), (272, 224), (224, 272), (304, 208),\n"
    "         (208, 304), (336, 192), (192, 336))":
        "SIZES = ((512, 512), (544, 448), (448, 544), (608, 416),\n"
        "         (416, 608), (672, 384), (384, 672))",
    '"schema": "stage256-two-final-qwen-rubric-v1"':
        '"schema": "stage512a-final-paired-qwen-rubric-v1"',
    '"comparability": "internal two-final comparison only"':
        '"comparability": "paired Stage512a comparison; identical prompts/seeds/protocol"',
}
for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"multi-AR scorer patch anchor count={count}: {old!r}")
    text = text.replace(old, new)
namespace = {
    "__name__": "stage512a_pair_multiar_runtime",
    "__file__": str(Path(__file__).resolve()),
}
exec(compile(text, str(SOURCE), "exec"), namespace)
namespace["main"]()
