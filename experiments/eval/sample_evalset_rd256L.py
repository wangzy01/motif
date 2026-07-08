#!/usr/bin/env python3
"""rd256L (redesign 256 LATEST ckpt) 变体入口：同 sample_evalset_rd256.py，仅 run key = rd256L，
指向最新冻结权重 eval_rd256L.pt（用于把"最新版 rd256 权重"结果加进对比报告）。
臂 override 与训练 job 逐字一致（causal + MLF）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.eval import sample_evalset as se  # noqa: E402

se._RUN_SPEC["rd256L"] = (
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
    se.main()
