#!/usr/bin/env python3
"""rd256L (redesign 256 LATEST ckpt) 变体入口（DPG）：同 generate_dpg_rd256.py，run key = rd256L，
指向最新冻结权重 eval_rd256L.pt。生成逻辑 100% 复用原脚本。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.eval.dpg import generate_dpg as gd  # noqa: E402

gd._RUN_SPEC["rd256L"] = (
    "redesign_256_causalmlf_global2304_20260630",
    "redesign256 causal+MLF LATEST",
    "configs/stage256_redesign_v1.yaml",
    [
        "model.text_adapter.enabled=true",
        "model.text_adapter.bidirectional=false",
        "model.text_adapter.multi_layer_fusion.enabled=true",
    ],
)

if __name__ == "__main__":
    gd.main()
