#!/usr/bin/env python3
"""Strict 512px anime metrics wrapper for the paired Stage512a sampling shards."""

from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE = Path(
    "/fsx/evan/mot_code/data_analysis/eval_releases/"
    "stage256_0718_s75000_vs_previous_final_20260721b/scripts/strict_anime_metrics.py"
)
EXPECTED_SOURCE_SHA256 = "c12f5816f9474d348e0f2d99718956a4190a3d3c0454c202c5e9ea6d1099680a"
PROMPT_BUNDLE = Path(
    "/fsx/evan/mot_code/data_analysis/eval_releases/"
    "stage256_offline_exact_v1_vs_prev_final_20260718a"
)
EVAL_BASE = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/"
    "eval_stage512a_final_weighted_vs_streaming_20260728v1"
)
REPO = Path("/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop_stage512a_weighted_20260724v7")
CURRENT_CHECKPOINT = (
    "/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/"
    "stage512a_20260724_weighted_w64_recovery_v2/last.pt"
)
PREVIOUS_CHECKPOINT = (
    "/fsx/evan/checkpoints/mot_pilot_v1/"
    "redesign_512a_multiar_anchor_global1536_20260705/last.pt"
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
    raise RuntimeError("frozen anime scorer SHA drift")
text = SOURCE.read_text(encoding="utf-8")
replacements = {
    'sample_root = eval_base / "sampling_v2" / run':
        'sample_root = eval_base / "sampling" / run',
    '"anime_256x256.json"': '"anime_512x512.json"',
    '"256x256" / run': '"512x512" / run',
    "image.size != (256, 256)": "image.size != (512, 512)",
    'record.get("width") != 256 or record.get("height") != 256':
        'record.get("width") != 512 or record.get("height") != 512',
    'legacy.get("width") != 256, legacy.get("height") != 256,':
        'legacy.get("width") != 512, legacy.get("height") != 512,',
}
for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"anime scorer patch anchor count={count}: {old!r}")
    text = text.replace(old, new)
text = text.replace(
    '"schema_version": "stage256_eval_20260718a_anime_metrics_v1"',
    '"schema_version": "stage512a_final_eval_20260728_anime_metrics_v1"',
)
text = text.replace(
    '"schema_version": "stage256_eval_20260718a_anime_metrics_success_v1"',
    '"schema_version": "stage512a_final_eval_20260728_anime_metrics_success_v1"',
)
namespace = {
    "__name__": "stage512a_pair_anime_runtime",
    "__file__": str(Path(__file__).resolve()),
}
exec(compile(text, str(SOURCE), "exec"), namespace)
namespace["BUNDLE"] = PROMPT_BUNDLE
namespace["EVAL_BASE"] = EVAL_BASE
namespace["REPO"] = REPO
namespace["RUNS"] = {
    "weighted_final_s125000": {
        "step": 125000,
        "checkpoint": CURRENT_CHECKPOINT,
        "config": str(REPO / "configs/stage512a_20260722_weighted_w64_v1.yaml"),
        "config_sha256": "611635fc13b04c6d62c20875eff541736afbe6e35cc17c1f81caafb7f823a380",
        "overrides": [],
    },
    "streaming_final_s166667": {
        "step": 166667,
        "checkpoint": PREVIOUS_CHECKPOINT,
        "config": str(REPO / "configs/stage512a_redesign_v1.yaml"),
        "config_sha256": "a76756240d1d84a87679acd58d5f44d5925c5d6cb058bd8f53f3ccaee259a83c",
        "overrides": [
            "model.text_adapter.enabled=true",
            "model.text_adapter.bidirectional=false",
            "model.text_adapter.multi_layer_fusion.enabled=true",
        ],
    },
}
namespace["main"]()
