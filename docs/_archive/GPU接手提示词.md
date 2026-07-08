> ⚠️ **已归档（2026-06-27）**:本文件的 onboarding 角色已被根目录 [`../../CLAUDE.md`](../../CLAUDE.md) 取代（agent 读 CLAUDE.md 即被 onboard）。接手请直接读 `CLAUDE.md` → `当前进展_交接GPU.md` → `docs/README.md`。本文件仅留作溯源。

# GPU 接手提示词（喂给 GPU 机器上的 AI）

> 用法：把下面 `---` 之间的内容整段复制给 GPU 机器上的 AI 即可。
> （本文件是**一次性 onboarding 提示词**；实时进展/坑点以根目录 `当前进展_交接GPU.md` 为准。）

---

你接手一个 **2–3B MoT 二次元文生图（T2I）pilot** 项目：frozen Qwen3-VL-2B 当理解路（~1.72B）+ 逐层复制的 ~1.42B 可训生成塔（Cosmos3 式 MoT），latent flow-matching，FLUX.2 VAE，目标先在 T2I 超过 Anima-2B。

**第一步，按顺序读（不要跳）：**
1. `当前进展_交接GPU.md`（根目录）—— **实时入口**：进展 / 待办 / 坑点 / 用户偏好 / 完整路径导览（§6）。⭐先读 §0.0 worktree 工作流。
2. `docs/README.md` —— **docs 总索引 / 设计权威**（设计正本三件套、实现现状、数据线、eval 产物、上游正本指针）。
3. `README.md`（根目录）—— 项目结构 + 架构对照表 + 已验证清单。

**当前状态（已不是"刚写完代码"——大量已上 GPU 并验证）：**
- ✅ **GPU G0–G6 全部跑通验好**（单卡 bf16 / SDPA=cuDNN / FSDP2 多卡近线性 / EMA / 最小采样器 Euler+CFG+und KV 缓存）。
- ✅ **complete_v1 真实数据线已实现并跑通**（`training/dataline/`，**5 域**：anime / MONET-real / GPIC / synthetic / **TextAtlas（第5域，文字渲染）**；在线 VAE encode、loader 规则全落地）。
- ✅ **domain_mix 供给饿死回归已修**（`drain_mode=proportional`，512 四域 ±5% 实证，commit 546bdd0）；审计后修 stats_cache 多机脚枪（commit faab339）。
- ✅ **stage256 四版消融训练 + 4-way 评测落地**：B1 text-adapter 提示词跟随显著领先（保留并带进 512），B2/B3 已停。
- 🔄 **在飞**：新设计三消融 ab1causal / ab2fusion / ab3lora 在 **2×8 多机**跑（~13%）；512 阶段备料（noise-shift / TextAtlas 配比已就绪，Evan：512 暂不急跑）。

**你的首要任务**：跟 Evan 用对话框对齐这次要做哪一步（512 推进 / 新消融监控 / eval 复验 / 优化项），**对齐前别动手**。

**关键约束（务必遵守）：**
- **worktree**（§0.0）：开发只在 **本 worktree（`mot_pilot_v1_cleanup`，分支 `cleanup-merge`）**；另两个直播 worktree（`mot_pilot_v1` / `mot_pilot_v1_ablations`）在跑训练，**代码文件绝不要碰**（retry/续训会重读磁盘 → 坑点#11）。
- **坑点#28**：纯 CPU 任务（单测 / smoke / loader 验证）也走 **JCC job**、别在本机直接 `python ...`（真 2B ~25GB RAM 易触发 OOM kill）；输出/缓存指 `/local`、`trainer.save_final=false`，零 /fsx 写。
- 两份 256 config **绝不互喂**：v1/B1 单机绑 `stage1a_t2i_256.yaml`（global512/400k）；多机/三消融绑 `stage1a_t2i_256_v2.yaml`（global1536/133k）。
- 环境：解释器 `/fsx/evan/mot_code/venv-gpu/bin/python`；缓存放 `/local`；多机正式 job 用 ECR 镜像 C（自带 EFA）。arxiv 被 block（查论文先 grep `/fsx/evan/doc/mot/mot_reading_report/data/page_text/`）。
- 风格：干货、表格对比、直接表态；中文为主、术语保留英文；**凡需决策/拍板/信息不明 → 先用对话框（AskUserQuestion）问 Evan，多问**；每个改动写测试验证、不擅自扩大范围；改完同步更新 `当前进展_交接GPU.md`。

读完上面三份，给我一句话「**我理解的现状 + 打算从哪步开始 + 需要你拍板的点**」，等我确认范围再动手。

---
