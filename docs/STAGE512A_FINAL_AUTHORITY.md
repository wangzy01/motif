# Stage512a 最终权威

冻结时间：2026-07-28（北京时间）

## 最终状态

- 正式训练：W64（8 nodes × 8 H100），global batch 2048。
- 完成量：125,000 steps / 256,000,000 exposures。
- 架构：causal text adapter 2 层、MLF 12 层。
- 正式训练 job
  `evan-s512a-weighted-w64-recovery-20260724v2-45hz` 已 `Succeeded`；
  最终 checkpoint 已经独立 JCC full-state 审计。

## 冻结身份

- 源码 release：
  `/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop_stage512a_weighted_20260724v7`
  - manifest SHA-256：
    `7256d093ac54f913fd50ab68c66a87667bcb8f0305c6ba4d5ae165afff6b2445`
  - release digest：
    `ff1d185229c8392f8b83bf612c288cbdbbdcffa28db67b46819e62f8d2a710ea`
- 最终配置：
  `/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop_stage512a_weighted_20260724v7/configs/stage512a_20260722_weighted_w64_v1.yaml`
  - SHA-256：
    `611635fc13b04c6d62c20875eff541736afbe6e35cc17c1f81caafb7f823a380`
- 最终 checkpoint：
  `/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage512a_20260724_weighted_w64_recovery_v2/last.pt`
  - bytes：`26,328,353,144`
  - step：`125000`
  - SHA-256：
    `957bfcf12b89c4042a32056566012a2451e72aa0b2706b0ca51808ab79155c31`
- full-state audit：
  JCC job `evan-s512a-final125k-audit-v1-jqzb`；
  `/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/stage512a_weighted_w64_final125000_fullstate_audit_20260728v1/REPORT.json`
  - report SHA-256：
    `9bbbbc3b8610c976a7a74b81458d4cf8b9a0116830cd9e23bfd40d9a24c4fe66`
  - 483 trainable/EMA/optimizer 闭包、optimizer、scheduler、adapter2/MLF12、
    tensor finite/nonzero 与 125,000 条 metrics 均通过。
- 加权离线计划：
  `/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_plan_stage512a_w64_256m_weighted_v1_20260722`
  - plan hash：
    `7295ca937d5002a0cf5e207c1d466c5992352e395adc9ae1c33f3cb616f675db`

以上 release、checkpoint、plan 和 audit 是冻结证据；不得原位编辑、覆盖、
移动或删除。`step_60000.pt` 只是保留里程碑，不能代替最终 `last.pt`。

## Stage512a 实际数据

域配比：

| 域 | 占比 | exposures | 计划中的 unique |
|---|---:|---:|---:|
| Anime | 37% | 94.72M | 16,294,879 |
| MONET real | 40% | 102.40M | 31,993,928 |
| GPIC real | 13% | 33.28M | 4,239,912 |
| Synthetic | 5% | 12.80M | 13,802,182 |
| TextAtlas | 5% | 12.80M | 4,036,998 |

## 最终评测

- 评测 release：
  `/fsx/evan/mot_code/data_analysis/eval_releases/stage512a_final_weighted_vs_streaming_20260728v1`
  - manifest SHA-256：
    `f96041fa142a44e49cb2649d0365a0cc27e0bde95e620459c80c819332f4a5b3`
- 正式输出：
  `/fsx/evan/mot_code/data_analysis/outputs/eval_stage512a_final_weighted_vs_streaming_20260728v1`
- 与旧 streaming W48 final 的同流程点估计：

| 指标 | W64 weighted final | 旧 W48 streaming final |
|---|---:|---:|
| Internal DPG-Qwen | 90.865 | 90.485 |
| Anime | 0.6160 | 0.6035 |
| GenEval | 0.6065 | 0.6032 |
| Fixed-sigma MSE | 0.80311 | 0.80275 |

这些值用于项目内同流程比较。Internal DPG-Qwen 不是官方 DPG/mPLUG 协议，
不能直接用于对外 SOTA 声明。GenEval 的主要弱项是 counting、two objects
以及 color/object binding；扩大随机 real 数据不能替代结构化 caption 或
synthetic 设计。

## 禁止误用的历史身份

旧 source v1/v2/v6、旧流式 W48、旧 6×8/row11163647 恢复链、历史
`stage512b_redesign_v1.yaml`、任何 `weights_ema` 和已经消费的 replay
身份都只作历史证据，不能启动 Stage512b-real。
