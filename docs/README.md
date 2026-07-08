# MoT docs 总索引（2026-06-27 文档瘦身后重建）

> **这是 `mot_pilot_v1/docs/` 的单一总索引**，按类别分组、每文档一行（路径 + 一句用途 + 状态标）。
> 实时进展/坑点见根目录 [`../当前进展_交接GPU.md`](../当前进展_交接GPU.md)；冲突时设计以「设计正本」、现状以「实现现状/交付权威」为准。
>
> **命名约定**：活跃文档 `主题_vN_YYYYMMDD.md`（日期=最后一次有意义更新）；被取代/历史快照移入 [`_archive/`](_archive/)。一切可经 git 恢复。
>
> **接手阅读顺序（最短路径）**：根 [CLAUDE.md](../CLAUDE.md)（≤1 屏 onboarding + 硬约束）→ 根 [当前进展_交接GPU](../当前进展_交接GPU.md) → 本索引 → §数据线（complete_v1 整合文档 + AAA 交付）→ §模型/架构（实现现状指南）→ §eval 产物。

## 入口 / 交接

| 文档 | 用途 | 状态 |
|---|---|---|
| [`../CLAUDE.md`](../CLAUDE.md) | **≤1 屏 onboarding + 硬约束**（接手 AI 先读，吸收原 GPU接手提示词） | ⭐入口 |
| [`../README.md`](../README.md) | 仓库根 README（项目总览 + 结构树 + 架构对照） | 入口 |
| [`../当前进展_交接GPU.md`](../当前进展_交接GPU.md) | **实时进展/坑点/决策流水**（接手必读） | 实时权威 |
| [交接文档_模型训练代码_GPU_20260611.md](交接文档_模型训练代码_GPU_20260611.md) | 06-11 训练代码交接（历史全背景） | 历史快照（Evan 确认保留） |

## 数据线

| 文档 | 用途 | 状态 |
|---|---|---|
| [数据+训练阶段_重设计方案_v1_20260629.md](数据+训练阶段_重设计方案_v1_20260629.md) | **⭐ 数据 + 训练阶段重设计方案**（新四阶段 256/512a/512b：domain_mix + MONET 显式源比例表 + per-stage caption/token + noise-shift + 训练配方 + multi-AR domain-first；1024+收尾段暂缓） | ⭐NEW 数据+训练阶段方案权威 |
| [数据+训练阶段_执行计划与进度_v1_20260629.md](数据+训练阶段_执行计划与进度_v1_20260629.md) | **重设计落地执行追踪**（全 file:line 改动 + 3 份审计发现+修复 + JCC 验证结果；Phase1/256 + Phase2/512a-512b-multiAR 全落地+2 commit `7e4a2cf`/`f346993`） | 执行追踪（配上行方案 doc 用；post-compact 续接读它） |
| [complete_v1_dataline_设计与变更_v1_20260627.md](complete_v1_dataline_设计与变更_v1_20260627.md) | **complete_v1 5 域数据线设计 + mix 回归修复 + TextAtlas 第 5 域**（合并 3 份已归档文档） | 数据线 loader/代码权威（⚠️ stage mix/源权重/token/shift 已被上行新方案取代） |
| [AAA数据交付文档_GPIC+MONET+anime_v2_20260617.md](AAA数据交付文档_GPIC+MONET+anime_v2_20260617.md) | 数据线交付权威（指标/路径/schema/loader 规则/anime+complete_v1） | 交付权威（5 域配比以上一行为准） |
| [complete_v1_anime集成_执行文档_20260617.md](complete_v1_anime集成_执行文档_20260617.md) | anime 集成 4 域执行记录 | 执行权威（anime 域内机制现行） |
| [采样权重spec_v1_20260616.md](采样权重spec_v1_20260616.md) | 采样权重 spec 人读版（band-first + 源/簇拉平 + loader 规则） | 活跃 |
| [日常探索版筛选spec_v1_20260617.md](日常探索版筛选spec_v1_20260617.md) | daily_v1 21M 子集筛选 spec（⚠️ 1024 band 供给 caveat） | 活跃 |
| [自己写的数据集文档_v3_20260616.md](自己写的数据集文档_v3_20260616.md) | 数据集**设计正本**（GPIC/MONET 处理策略、caption 配比、为什么这样设计） | 设计基线（冻结；⚠️ 配比/源权重/caption 路由部分已被新方案取代） |
| [采样config/](采样config/) | 采样 config 对照副本（`t5_sampling_{complete_v1,daily_v1,weights}_v1.json`；线上 5 域权威 config 在 repo 外 `/fsx data_analysis`，见整合文档 §5 坑点#11） | 对照 |

## 模型 / 架构

| 文档 | 用途 | 状态 |
|---|---|---|
| [模型架构实现现状与消融指南_v1_20260617.md](模型架构实现现状与消融指南_v1_20260617.md) | 当前架构「真相」：网络长什么样 + 默认超参 + v3 哪些已实现 + 消融怎么跑 | 实现现状权威 |
| [自己写的模型+训练设置文档_v3_20260617.md](自己写的模型+训练设置文档_v3_20260617.md) | 模型/架构/训练设置**设计正本**（设计意图与理由） | 设计正本（实现以上一行为准；⚠️ 训练阶段/token/shift/LR 部分见 [数据+训练阶段_重设计方案_v1](数据+训练阶段_重设计方案_v1_20260629.md)） |
| [i1_深度分析与对我们的启发_20260611.md](i1_深度分析与对我们的启发_20260611.md) | i1 论文分析底稿（消融增量依据；启发已落入 v3 设计与 TextAtlas） | 决策级参考 |
| [自己写的后续试验_v1_20260616.md](自己写的后续试验_v1_20260616.md) | RAEv2/Qwen3-VL ViT latent/RAC/RecA/RF 等后续实验路线图 | 研究 roadmap（不在 baseline，归属待 Evan 拍） |
| [naifu迁移评估与计划_v1_20260619.md](naifu迁移评估与计划_v1_20260619.md) | naifu 迁移评估与计划（2026-06-19 搁置；含 wandb/grad-accum 留路） | 评估结论（已搁置） |

## stage / VAE / eval / infra

| 文档 | 用途 | 状态 |
|---|---|---|
| [512_noise_shift_methodology_v1_20260622.md](512_noise_shift_methodology_v1_20260622.md) | 512 noise-shift 取值方法学协议（协议 A/B） | 方法学活跃（✅ 取值已定：512=1.0/1024=3.33，见新方案 §6） |
| [512_prep_notes_v1_20260622.md](512_prep_notes_v1_20260622.md) | 512 launch 就绪度备忘（anime 17-shard 偏斜 B7a、DTensor 补丁 B7b、显存/缓存） | 512 就绪度 |
| [vae_latent_cache_v1_20260622.md](vae_latent_cache_v1_20260622.md) | VAE latent/moments 离线缓存说明（256 不用、留脚手架） | 活跃 |
| [后续增量与eval计划_v1_20260620.md](后续增量与eval计划_v1_20260620.md) | 项目增量盘点 + eval 阶段1 实施计划（Phase 1 eval 未开始） | 规划（待 256 ckpt 稳定） |
| [公司参考代码索引_v1_20260619.md](公司参考代码索引_v1_20260619.md) | 公司可复用参考代码索引（打分器/loader/EMA-LoRA 合并等） | 活跃 |
| [T2_H100_GPU踩坑与集群使用建议_20260614.md](T2_H100_GPU踩坑与集群使用建议_20260614.md) | H100 / JCC 集群踩坑与使用建议 | 集群参考（非 MOT 训练） |

> **代码结构（重构后）模块说明**（行为不变的解耦重构；核心训练/前向逻辑数值零改动）：
> - `training/diagnostics.py` — 诊断脚手架（逐-phase trace / batch-ready barrier / faulthandler dump）。**默认全关、env 门控**；关闭时对训练逐位无影响。
> - `training/checkpoint.py` — checkpoint 存取/resume 链路（FSDP2/EMA round-trip、step 续训、跨-stage 加载与指纹校验）。
> - `training/config_schema.py` — 训练/架构配置 schema 与校验（消融开关 `text_adapter`/`long_skip`/`noise_condition` 合法性）。
> - `networks/arch_mot_utils.py` — `arch_mot.py` 的纯函数辅助件（RoPE / mask / phase 分发等）。
>
> ⚠️ 上列为本轮 cleanup 重构目标结构：`diagnostics.py` 已落地，其余随各自重构任务陆续合并（合并前以核心文件内联实现为准）。模块「现状真相」以 [模型架构实现现状与消融指南_v1_20260617.md](模型架构实现现状与消融指南_v1_20260617.md) 为准。

## eval 产物 → [`../experiments/results/`](../experiments/results/)

| 产物 | 是什么 | 状态 |
|---|---|---|
| `../experiments/results/EVAL_s256_4way_20260623/4way对比报告_20260623.md` | 256 四版消融最终 4-way 对比（最新权重 ~135–144k；B1 text-adapter 领先、B2/B3 已停） | ⭐最新权威（路径随主控合并落地，故以 code-span 列出） |
| [../experiments/results/EVAL_ablation_comparison.md](../experiments/results/EVAL_ablation_comparison.md) | 256 四版消融 head-to-head（中段、单 seed、小样本） | 🟡256-only · PROVISIONAL（被 4way 取代） |
| [../experiments/results/EVAL_base_v1_analysis.md](../experiments/results/EVAL_base_v1_analysis.md) | v1 baseline 评估分析（含配比无偏斜实测） | 🟡256-only · PROVISIONAL |
| [../experiments/results/STAGE256_3RUNS_STATUS.md](../experiments/results/STAGE256_3RUNS_STATUS.md) | 256 三/四版 run 实时状态流水 | 实时 |
| [../experiments/results/RESULTS.md](../experiments/results/RESULTS.md) | 数据线/实验结果汇总（多份 doc 引用；load-bearing） | load-bearing |

## RAE 线（另一负责人）

| 文档 | 用途 | 状态 |
|---|---|---|
| [E1_RAE解码器bakeoff实施spec_v1_20260624.md](E1_RAE解码器bakeoff实施spec_v1_20260624.md) | E1 RAE 解码器 bakeoff 实施 spec | **RAE 线，另一负责人**（本索引不维护其内容） |

## 归档 → [`_archive/`](_archive/)

> 逐项理由见 [`_archive/README.md`](_archive/README.md)。归档项均为「被整合 / 被取代 / 旧索引」，仅供溯源；当前决策以上列活跃文档为准。

| 文档 | 归档理由 |
|---|---|
| [_archive/数据集问题审计_complete_v1_dataline_20260625.md](_archive/数据集问题审计_complete_v1_dataline_20260625.md) | 已并入 [complete_v1_dataline_设计与变更_v1_20260627.md](complete_v1_dataline_设计与变更_v1_20260627.md)（根因部分） |
| [_archive/mix回归诊断_v1_20260626.md](_archive/mix回归诊断_v1_20260626.md) | 已并入整合文档（mix 回归诊断部分） |
| [_archive/TextAtlas5M接入设计_v1_20260626.md](_archive/TextAtlas5M接入设计_v1_20260626.md) | 已并入整合文档（第 5 域设计，当时 plan-only、现已实现） |
| [_archive/MoT方案0608_20260608.md](_archive/MoT方案0608_20260608.md) | 0608 早期主方案，整体设计已被 v3 设计正本取代 |
| [_archive/模型架构修改规划_v1_20260616.md](_archive/模型架构修改规划_v1_20260616.md) | 落地 spec 已实现，现状以 [模型架构实现现状与消融指南_v1_20260617.md](模型架构实现现状与消融指南_v1_20260617.md) 为准 |
| [_archive/mot文档总索引_20260617.md](_archive/mot文档总索引_20260617.md) | 旧 `doc/mot/` 文档入口索引；已并入本 README |
| [_archive/文档清理索引_20260615.md](_archive/文档清理索引_20260615.md) | 旧 `doc/mot/` 文档清理索引；已并入本 README |
| [_archive/GPU接手提示词.md](_archive/GPU接手提示词.md) | 一次性 onboarding 提示词，角色已被根 [`../CLAUDE.md`](../CLAUDE.md) 取代；2026-06-27 根文档 collapse 归档 |

## 上游正本与扩展资料（**仓库外**指针，勿当 repo-relative 链接）

> 以下路径在 `/fsx/evan/...`，是本目录快照的上游正本与未复制进仓库的扩展资料，需在文件系统上访问。

- **设计/数据线正本（上游）**：`/fsx/evan/doc/mot/`（`自己写的{数据集,模型+训练设置}文档_v3.md`、`自己写的后续试验.md`、`AAA数据交付文档...v2_20260617.md`、`采样权重spec_v1_20260616.md`、`日常探索版筛选spec_v1_20260617.md` 等）。
- **处理流水账**：`/fsx/evan/mot_code/data_analysis/cluster/README_T2_NEXT.md`（两轮审计 + 全部数据线处理决策；实时处理进度以它为准）。
- **5 域采样 config 正本**：`/fsx/evan/mot_code/data_analysis/configs/`（`t5_sampling_complete_v1_textatlas.json` = 5 域线上权威；`t5_sampling_complete_v1.json` = 4 域旧 ablation，勿改）。
- **环境 / AWS / JCC**：`/fsx/evan/doc/env/`。
- **论文与阅读报告**：`/fsx/evan/doc/mot/paper_filtered`、`/fsx/evan/doc/mot/mot_reading_report`。
- **上游历史归档**：`/fsx/evan/doc/mot/归档`（AI 中间产物、6.8 前调研、旧交接、旧环境副本）。
