# jobs/_archive

一次性诊断 / 二分(bisect) / trace 类 job 的归档目录。这些 job 是为定位某次具体故障(code-vs-infra 二分、I/O stall 诊断等)临时编写的，已完成使命、不属于常规训练/评测/缓存流程，因此从 `jobs/` 根目录移出归档，避免与可复用的规范 job 混淆。

归档不删除，保留以备回溯当时的诊断思路与配置。

## 归档清单

| 文件 | 归档原因 |
| --- | --- |
| `bisect_2x8_A_clean_causal.yaml` | code-vs-infra 二分对照①：干净 main 树 causal text-adapter @2×8 跑 100 步,验证干净 adapter 16 卡稳定。一次性 bisect 实验。 |
| `bisect_2x8_B_wt_fusionoff.yaml` | code-vs-infra 二分对照③：worktree 树 + os.environ 补丁,融合关路径,隔离重构本身是否引入问题。一次性 bisect 实验。 |
| `bisect_2x8_C_wt_fusionon.yaml` | code-vs-infra 二分对照②(Ab2 真身):worktree 树 multi_layer_fusion 开,验证 Ab2 融合 16 卡能否稳定 100 步。一次性 bisect 实验。 |
| `diag_2x8_ab3_stall.yaml` | 一次性诊断:复现 dataline I/O stall 并用 faulthandler 每 45s dump 全线程栈,定位生产者线程硬阻塞点。诊断完成即归档。 |
| `归档_stage1a256_4x8_train.yaml` | 4×8(=32卡/4节点) stage1a t2i @256 正式训练 job 一次性快照(global batch 2048、LR 1e-4、max_steps 400k,Evan 06-18 拍板的无人值守配方)。已被多机 v2 产线配方取代,06-22 commit 45a5cbf 归档保留。 |
| `归档_train256_4x8_smoke.yaml` | 4×8(=32卡/4节点) stage256 正式训练前的多机 smoke 快照(验 torchrun+FSDP2+complete_v1@256 跑通,非正式 run)。一次性快照,06-22 commit 45a5cbf 归档保留。 |

## 未归档说明

任务原始列表还点名了 9 个 `diag_2x8_ab2_fusion_*_trace.yaml` / `smoke_2x8_ab2_fusion_*_{trace,force_exit,modeltrace}.yaml`,经核对在 `jobs/` 中**并不存在**,故未移动(无对应文件)。
