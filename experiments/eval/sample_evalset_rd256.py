#!/usr/bin/env python3
"""rd256 变体入口：monkey-patch sample_evalset._RUN_SPEC 加 redesign run key，再委托原 main()。

目的：给 redesign stage256 正式训练 run（causal adapter + multi_layer_fusion）跑与 4-模型
基准【逐字一致】的 anime prompt-following eval，而【不改动】eval-AI 的 experiments/eval/sample_evalset.py
（它当前有未提交改动 + 坑#11）。这里只追加一个 run_key，生成/打分逻辑 100% 复用原脚本，零偏离=公平。

用法（与原脚本 CLI 完全相同，只是 --runs rd256）：
  python experiments/eval/sample_evalset_rd256.py --runs rd256 --snapshot-dir <放了 eval_rd256.pt 的目录> \\
    --out-dir <out> --size 256 --steps 64 --cfg 4.0 --seed 7 --weights ema --batch-size 8
ckpt 解析：--snapshot-dir <dir> → <dir>/eval_rd256.pt（冻结副本，避开 last.pt 每3000步覆盖竞争）。
臂 override 与训练 job（redesign_256_6x8_causalmlf_FORMAL_job.yaml）逐字一致；bidirectional=false 必须显式
（config 默认 True；ckpt 是 causal 训练的，不钉会 train/test 语义错配）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.eval import sample_evalset as se  # noqa: E402

# 追加 redesign run key（不改原 _RUN_SPEC 文件；进程内注入）。
# 元组格式 = (ckpt 子目录, 名称, base config, override dotlist)，与 se._RUN_SPEC 一致。
se._RUN_SPEC["rd256"] = (
    "redesign_256_causalmlf_global2304_20260630",       # 仅 --snapshot-dir 未设时用；本 eval 恒用 --snapshot-dir
    "redesign256 causal+MLF (global2304)",
    "configs/stage256_redesign_v1.yaml",
    [
        "model.text_adapter.enabled=true",
        "model.text_adapter.bidirectional=false",       # causal（Evan 指定；非双向）
        "model.text_adapter.multi_layer_fusion.enabled=true",
    ],
)

if __name__ == "__main__":
    se.main()
