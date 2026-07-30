#!/usr/bin/env python3
"""rd256 变体入口（DPG）：monkey-patch generate_dpg._RUN_SPEC 加 redesign run key，委托原 main()。
与 sample_evalset_rd256.py 同理：不改 eval-AI 的 generate_dpg.py，生成逻辑 100% 复用原脚本。
用法同原 generate_dpg.py，只是 --runs rd256 + --snapshot-dir <放 eval_rd256.pt 的目录>。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.eval.dpg import generate_dpg as gd  # noqa: E402

gd._RUN_SPEC["rd256"] = (
    "redesign_256_causalmlf_global2304_20260630",
    "redesign256 causal+MLF (global2304)",
    "configs/stage256_redesign_v1.yaml",
    [
        "model.text_adapter.enabled=true",
        "model.text_adapter.bidirectional=false",
        "model.text_adapter.multi_layer_fusion.enabled=true",
    ],
)

if __name__ == "__main__":
    gd.main()
