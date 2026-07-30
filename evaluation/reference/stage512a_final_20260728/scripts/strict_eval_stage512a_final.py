#!/usr/bin/env python3
"""512px wrapper around the frozen strict GenEval/DPG evaluator."""

from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE = Path(
    "/fsx/evan/mot_code/data_analysis/eval_releases/"
    "stage256_offline_exact_v1_vs_prev_final_20260718a/scripts/strict_eval.py"
)
EXPECTED_SOURCE_SHA256 = "0afc7e6a8c0b9a9751ff9c7ee48edbdedc6c3b0847169a618d3ad3569235babc"
REPO = Path("/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop_stage512a_weighted_20260724v7")
CURRENT_CHECKPOINT = Path(
    "/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/"
    "stage512a_20260724_weighted_w64_recovery_v2/last.pt"
)
PREVIOUS_CHECKPOINT = Path(
    "/fsx/evan/checkpoints/mot_pilot_v1/"
    "redesign_512a_multiar_anchor_global1536_20260705/last.pt"
)


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
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"strict evaluator patch anchor count={count}: {old!r}")
    text = text.replace(old, new)
text = text.replace("stage256_eval_20260718a_v1", "stage512a_final_eval_20260728_v1")

namespace = {
    "__name__": "stage512a_strict_eval_runtime",
    "__file__": str(Path(__file__).resolve()),
}
exec(compile(text, str(SOURCE), "exec"), namespace)
namespace["GENEVAL_REPO"] = REPO
namespace["GENEVAL_RUN_SPECS"] = {
    "weighted_final_s125000": {
        "config": REPO / "configs/stage512a_20260722_weighted_w64_v1.yaml",
        "config_sha256": "611635fc13b04c6d62c20875eff541736afbe6e35cc17c1f81caafb7f823a380",
        "checkpoint": CURRENT_CHECKPOINT,
        "global_step": 125000,
        "adapter": True,
        "mlf": True,
        "overrides": [],
    },
    "streaming_final_s166667": {
        "config": REPO / "configs/stage512a_redesign_v1.yaml",
        "config_sha256": "a76756240d1d84a87679acd58d5f44d5925c5d6cb058bd8f53f3ccaee259a83c",
        "checkpoint": PREVIOUS_CHECKPOINT,
        "global_step": 166667,
        "adapter": True,
        "mlf": True,
        "overrides": [
            "model.text_adapter.enabled=true",
            "model.text_adapter.bidirectional=false",
            "model.text_adapter.multi_layer_fusion.enabled=true",
        ],
    },
}
namespace["main"]()
