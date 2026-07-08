# CLAUDE.md — mot_pilot_v1（cleanup worktree）

> **接手的 AI 先读这份（≤1 屏）**。然后:详细进展/坑点/决策 → [当前进展_交接GPU.md](当前进展_交接GPU.md);项目结构/架构对照 → [README.md](README.md);设计权威 + 文档总索引 → [docs/README.md](docs/README.md)。
> 读完先给 Evan 一句话「**我理解的现状 + 打算从哪步开始 + 需要你拍板的点**」,**等他确认范围再动手**。

## 这是什么
**2–3B MoT 二次元文生图（T2I）pilot**:frozen Qwen3-VL-2B 当理解路（~1.72B）+ 逐层复制的 ~1.42B 可训生成塔（Cosmos3 式 MoT，`*_moe_gen` 分流）,latent 空间 flow-matching（v-pred + rectified flow + 纯 MSE）,FLUX.2 VAE。目标先在 T2I 超过 Anima-2B。

**现状（已不是"刚写完代码"）**:GPU G0–G6 全验好（单卡 bf16 / SDPA=cuDNN / FSDP2 多卡近线性 / EMA / 最小采样器 Euler+CFG+und KV 缓存）;**complete_v1 真实数据线 = 5 域**（anime / MONET-real / GPIC / synthetic / **TextAtlas**）端到端跑通;stage256 四版消融 + 评测全部落地;**现 redesign 数据配方的 stage256 正式训练在 6×8 跑**（causal+MLF, global2304, ~166k步/~4天, job `...-zdfp`）。512a/512b 衔接代码已就绪（gated, 待 256 final ckpt）。

## worktree / 分支（先搞清在哪）
本仓库（`github.com/wangzy01/motif-t2i`）现存 2 个 worktree（消融已全跑完;`ablations` worktree + `ablations-v2`/`docs-reorg` 分支已于 2026-06-30 退役）:
- **`mot_pilot_v1_cleanup`（分支 `cleanup-merge`）= 事实上的主干/真源,所有开发在此改。** `main` 暂不跟进（Evan 定不动,现落后 16 commit）;`mot_pilot_v1`（main, 旧）不碰。
- ⚠️ **眼下 cleanup 自己正被直播训练实时读**:redesign stage256 正式训练（job `...-zdfp`, 6×8, ~4天）`cd` 进本 worktree 跑 → **运行期间别改它 import 的代码**（train.py/configs/networks/training/* = 坑点#11）;**改文档安全**。256 跑完前,256/eval 线在本 worktree 的未提交文件也先别动。

## 怎么跑
- **解释器**:`/fsx/evan/mot_code/venv-gpu/bin/python`（torch 2.11+cu128 / transformers 5.11 / diffusers 0.38 / omegaconf / einops）。
- **多机正式 job**:ECR 镜像 C `471112743779.dkr.ecr.us-west-2.amazonaws.com/autobuild:py312-torch211-cu129`（自带 EFA / aws-ofi-nccl）+ 同一 venv,零代码改动跨节点近线性。
- **JCC**:`/fsx/evan/bin/jcc job submit <yaml>` / `jcc job log <name> -p all -n 300`（不加 `-p` 会进交互菜单）/ `jcc job list` / `jcc node list`。多机需自组 rendezvous（`NODE_RANK=${HOST##*-}` + 共享文件传 `MASTER_ADDR`）。
- **完整 CPU + GPU 自检命令清单** → [当前进展_交接GPU.md](当前进展_交接GPU.md) §1。

## 硬约束（务必遵守）
- **坑点#28 — 所有运行（含纯 CPU 单测/smoke/loader 验证）都走 JCC job,别在本机 `python ...`**:本机内存有限,真 2B ~25GB RAM 易触发 OOM kill（杀进程/整个会话）。JCC 时 `trainer.save_final=false` + 输出/缓存指 `/local`,零 /fsx 写。只有秒级、零模型加载的纯文件/git 操作可本机直接做。
- **坑点#11 — JCC job 实时读 /fsx 文件**:提交 job 后别改它要跑的 train.py/config;两个直播 worktree 的代码文件不要碰。
- **两份 256 config 绝不互喂**:直播 v1/B1 单机绑 `configs/stage1a_t2i_256.yaml`（global512/400k）;多机/三消融绑 `configs/stage1a_t2i_256_v2.yaml`（global1536/133k）。
- **5 域采样 JSON 隔离（坑点#11）**:`cleanup` configs 指 `t5_sampling_complete_v1_textatlas.json`（repo 外,`/fsx/evan/mot_code/data_analysis/configs/`）;原 `t5_sampling_complete_v1.json` 保持 **4 域**给在跑的旧 ablation,**绝不能改**。
- **不碰 RAE E1 线**:`docs/E1_RAE*` 是另一 agent 的线,只读、不动。
- **commit 只在 Evan 明确要求时**;message 结尾加 `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`。
- **arxiv 被 block**:查论文先 grep `/fsx/evan/doc/mot/mot_reading_report/data/page_text/`;HF / github / pypi 可用。

## 工作流 & 风格（Evan 偏好）
- **动手前先用对话框（AskUserQuestion）和 Evan 对齐范围**;凡需决策 / 拍板 / 信息不明 / 提供偏好 → 先问,**多问**,别擅自定。重大设计点即使有推荐,也先说清再让他拍板。
- 每个改动**写测试验证、确保正确性**;**不擅自扩大范围**（聚焦当轮任务）。
- 风格:干货、表格对比、直接表态;中文为主、术语保留英文;不重复造轮子。
- 改完同步更新 [当前进展_交接GPU.md](当前进展_交接GPU.md)（Evan 实时维护的运维日志,每轮有实质改动就更新）。
