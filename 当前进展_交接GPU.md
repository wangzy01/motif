# MoT Pilot — 当前进展 / 待办 / 坑点 / 路径导览（实时维护）

> **本文件 = LIVE OPS LOG**（Evan 实时维护的运维日志，每轮有实质改动就更新）。
> **首次接手** → 先读 `CLAUDE.md`（≤1 屏 onboarding + 硬约束）；**设计权威 / 文档总索引** → `docs/README.md`。

**非常重要！本机上进运行轻量程序，即便是CPU任务，也尽量上jcc。因为之前有好几次OOM kill了，导致相关记录都没了。**


> **更新：2026-06-27**　｜　维护约定：**每轮有实质改动就更新本文件**。最新实测/结论也散见 `experiments/results/*.md`（STAGE256 状态 / EVAL_* / multinode_scaling / microbatch）与 `docs/` 数据线文档（TextAtlas 接入 / mix 回归诊断 / dataline 审计）。
> **🆕 heads-up（2026-06-29，evan 数据+训练阶段重设计执行线）**：本人在 **cleanup worktree** 落地"数据+训练阶段重设计"（设计权威见 [docs/数据+训练阶段_重设计方案_v1_20260629.md](docs/数据+训练阶段_重设计方案_v1_20260629.md)，执行追踪见 [docs/数据+训练阶段_执行计划与进度_v1_20260629.md](docs/数据+训练阶段_执行计划与进度_v1_20260629.md)）。**✅ 已完成（2026-06-30，全 config 字段门控、live/256/旧 JSON 字节级零影响）**：Phase1(256, commit `7e4a2cf`) + Phase2(512a/512b 方图 + multi-AR domain-first sampler + caption 早减 tag, commit `f346993`)，均 JCC 验+独立审计。GPU 256 短训 smoke rc=0、GPU 非方图 forward rc=0。multi-AR 全程 **gated off**（`dataset.multi_ar.enabled:false`，re-resize 调优推迟到真启用 512 前）。新文件：`data_analysis/configs/t5_sampling_redesign_v1.json`(repo 外) + `configs/stage{256,512a,512b}_redesign_v1.yaml` + `training/dataline/multi_ar.py` + dataline gated 改。**承诺不碰（已守）**：两个直播 worktree 代码、共享 live 采样 JSON（`t5_sampling_complete_v1{,_textatlas}.json`）、RAE 线（`docs/E1_RAE*`）。JCC job 一律 `evan-` 前缀、产物指 `/local`。@raev2 / @模型测试 两位：碰撞面仅共享采样 JSON / data_analysis configs / 集群 GPU，已全程避让；如有冲突找 Evan。
>
> **最新主线（2026-06-27）**：
> ① **数据线 complete_v1 = 5 域**：新增 **TextAtlas（第5域，文字渲染，注入域如 synthetic）** + 修 **domain_mix 供给饿死回归**（`drain_mode=proportional`，512 四域全 ±5% 实证、不 data-bound）→ commit **546bdd0**（TextAtlas+mix-fix）+ **faab339**（审计后修 stats_cache 多机脚枪 + probe loader 残留）。详见 2026-06-27 块。
> ② stage256 四路评测已出 → **B1 text-adapter 提示词/概念跟随显著领先、保留并带进 512**，B2/B3 无可测收益 **已停**；v1/B1 继续训。
> ③ **新增 Ab2(末 N 层多层融合) + Ab3(冻结理解塔 und_lora) 消融机制已集成**，三消融 **ab1causal/ab2fusion/ab3lora 在 2×8 多机跑（~1.7 万/13.3 万 ≈ 13%）**；**2×8 多机产线配方就绪**（global 1536、133k 步 = 对齐 v1 样本预算）。
> eval 详见 [EVAL_ablation_comparison.md](experiments/results/EVAL_ablation_comparison.md)、stage256 状态见 [STAGE256_3RUNS_STATUS.md](experiments/results/STAGE256_3RUNS_STATUS.md)、多机配方见 [multinode_scaling_20260623.md](experiments/results/multinode_scaling_20260623.md)。**分支已收敛进单一 `main`；最新整合代码在 `cleanup-merge`（解耦重构已完成、经验证、待训练跑完 FF 进 main）——worktree 工作流见 §0.0。**
> 历史完整背景见 [doc/mot/交接文档_模型训练代码_GPU_20260611.md](/fsx/evan/doc/mot/交接文档_模型训练代码_GPU_20260611.md)；架构改动依据见 [doc/mot/模型架构修改规划_v1_20260616.md](/fsx/evan/doc/mot/模型架构修改规划_v1_20260616.md)。
>
> **🆕 2026-06-30（stage256 四模型最终评测 + ab2 adapter 方向默认改 causal — 监控/eval 执行线）**：① 4 个 stage256 run 全停、跑完最终 4-模型评测（报告 [experiments/results/EVAL_final_4model_stage256.md](experiments/results/EVAL_final_4model_stage256.md)，含图表+样图网格）——**anime 概念跟随(tagger 3-seed)：ab2fusion > ab1causal > b1 ≫ ab3lora**；**DPG 通用提示词 b1≈ab1≈ab2 打平**(mPLUG 极差 0.21)、ab3lora 垫底；结论 ab3lora **no-go**、ab1causal>b1 **数据支持「保留 causal、停 b1」**、ab2fusion 优势是 anime 概念专项(非通用)。② **adapter 方向开关 `model.text_adapter.bidirectional`（正交于 `multi_layer_fusion`）**：`stage1a_t2i_256_v2.yaml` 默认从双向**改为 causal（`false`，Evan 偏好）**；⚠️ 现有 `s256v2_2x8_ab2fusion` ckpt 是 **bidir 训练**的（领跑结论=bidir+fusion），三处 eval 入口（`experiments/eval/sample_evalset.py`/`dpg/generate_dpg.py`/`jobs/eval_final_fixedsigma.yaml`）已显式钉 `bidirectional=true` 复现、防 train/test 错配；**causal+fusion 是未训未测的新组合，启用需重新训练验证**。（仅改 256_v2 ablation config + eval 脚本，未碰直播 worktree / live 采样 JSON / 重设计线的新 configs。）

---

## 0. 一句话 + 现在在哪

### 0.0 仓库 / worktree 工作流（2026-06-24，⭐先读）

本项目是**单一 git 仓库**（`github.com/wangzy01/motif-t2i`），当前过渡期并存 **3 个 git worktree**：

| 目录 | 分支 | 是什么 | 状态 |
|---|---|---|---|
| `mot_code/mot_pilot_v1` | `main` | **切换后的唯一正式仓库** | 直播：v1/B1 单机训练在此 `cd` 运行；当前**暂落后**最新 7 笔 |
| `mot_code/mot_pilot_v1_ablations` | `ablations-v2` | Ab2/Ab3 + 2×8 产线化（临时） | 直播：ab1/ab2/ab3 三消融 2×8 在此运行；**切换后删除** |
| `mot_code/mot_pilot_v1_cleanup` | `cleanup-merge` | **当前最新整合代码** = ablations-v2 + 解耦重构 + 归档 + 文档 | 隔离、无直播；**当前开发都在这里改** |

**工作流（重要）：**
- **现在（直播训练未跑完的这几天）**：要看 / 基于最新代码、继续开发 → 一律在 **`cleanup-merge`（`mot_pilot_v1_cleanup`）** 上做。两个直播 worktree 的**代码文件不要碰**（v1/B1/ab* 在跑，retry/续训会重读磁盘，改动可能作废数天算力——坑点#11）。
- **切换（等 5 个直播训练全部跑完 + Evan 确认后）**：`cleanup-merge` **fast-forward 合并进 `main`** → **`mot_pilot_v1`(main) 成为唯一正式仓库**；`mot_pilot_v1_ablations`、`mot_pilot_v1_cleanup` 两个临时 worktree **删除**。切换步骤（前置校验 / FF / 产物归档 / 回滚）见 `scratchpad/CUTOVER_runbook_cleanup_merge.md`。
- ⚠️ 两份 256 config 隔离、**绝不互喂**：直播 v1/B1 绑 `stage1a_t2i_256.yaml`（global512/400k）；多机/三消融绑 `stage1a_t2i_256_v2.yaml`（global1536/133k）。
- 两个直播 worktree 根目录各放了一个 `_仓库状态_*.md` 过渡提示（未跟踪、切换时删）。

> **cleanup 整理已完成并经验证**（commit `affc89c` 重构 / `2b63c7e` 归档 / `14eef21` 文档 / `5954dfe` 审计修正）：稳健解耦重构**行为逐位不变**（诊断脚手架→`training/diagnostics.py`、checkpoint/指纹→`training/checkpoint.py`、配置校验→`training/config_schema.py`、mask/pos→`networks/arch_mot_utils.py`；`train.py` 646→383 行）。**验证**：4 审计全过（重构等价/导入完整/归档安全/文档准确）+ JCC 实跑（step-0 Δ=0 且**重构前==重构后**、全单测 ALL_UNIT_OK、ab3lora 端到端 rc=0、save→resume 往返 OK）→ 直播 ckpt 可在合并后代码上原样 resume。

---

**2–3B MoT 文生图 pilot**：frozen Qwen3-VL-2B 当理解路（~1.72B）+ 逐层复制的 ~1.42B 可训生成塔（Cosmos3 式 MoT，`*_moe_gen` 分流），latent 空间 flow-matching（v-pred + rectified flow + 纯 MSE），目标先在 T2I 超过 Anima-2B。

**现在**：v3 Day-1 baseline 架构 + i1 消融骨架 + **FLUX.2 VAE adapter** CPU 逻辑验证通过；**2026-06-17 GPU 段已推进 G0–G2 + G6**（单卡 H100 80GB）：环境 + 真权重 smoke 全过、**bf16 autocast 单卡真规模 step 跑通**（overfit loss 2.0→0.3）、**SDPA 后端实测 = cuDNN（不退化到 MATH，1.16× flash 天花板，无需迁 attention）**、batch/吞吐扫描完成（grad_ckpt 必开；512px 单卡 bs32/64 从容、~13 img/s）、pixels→真 VAE→MoT 路径 GPU 亦跑通；**G6 最小采样器（Euler flow + 双前向 CFG + VAE decode）实现并端到端验证**（过拟合重建 MSE 0.041 vs 噪声基线 2.0 = 0.02×）。
**2026-06-17 续**：**G4 多卡 FSDP2（Route B = 本地 train.py + FSDP2）+ G5 EMA 已跑通验好**（JCC 2×H100）：2卡 vs 1卡 loss 近逐位一致、吞吐 ~2.2× 近线性、分片显存 44.9→33.2GB/卡、EMA(0.9999) 多卡 ckpt gather 无错。
**2026-06-17 再续**：**G6 余项 und 逐层 KV 缓存已实现验好**——采样 loop 外预计算每层 frozen und K/V、loop 内只跑 gen 塔，**1.62× 采样加速**；正确性 CPU fp32 逐位 0、端到端过拟合重建 cached=非cached（MSE 0.026 一致、diff 0.0000）。
**2026-06-17 末（外部 AI 审计 review）**：逐条拿代码核实后**采纳 4 项**——B1 臂显存路径修复（frozen und→`no_grad` + 生成臂 `gradient_checkpointing`，此前绕过 baseline checkpoint）、B1 参数量更正（实测真 config **~231M≈16% 生成塔**，非文档 ~35M/<2.5%）、`trainer.save_final` 开关（防调试误写大 ckpt 到 /fsx）、多处滞后文案清理；**驳回 1 项**（B1"污染归因"——消融是设计好的 4 臂阶梯，启用臂≠baseline 是要测的拓扑差异本身，非 bug）。新增 `scripts/test_b1_grad_ckpt.py`（ckpt on/off 输出+梯度 max|Δ|=0）。
**2026-06-17 数据线（G3，complete_v1 全忠实接入）✅ 跑通**：数据交付后实现完整 `training/dataline/`（config/manifest_index/sampler/caption_build/image_fetch/complete_dataset），4 域真实数据（anime R2-HTTP / MONET+synth tar 流式 / GPIC 本地文件）→ 流式生产者+shuffle buffer+CompleteSampler→`{prompts,pixels∈[-1,1]}`，**VAE 在线 encode**。**端到端真数据+真 3.1B 模型+真 FLUX.2 VAE 训练 20 step loss 2.0→1.81、无 NaN、峰值 41GB → V1 已能完全开训**。configs 256/512/1024 已指向 `CompleteV1Dataset`。
**2026-06-17 数据线对抗 review + 修复**：15-agent 工作流（4 维 reviewer + 真数据对抗 verify）。**采纳并修 6 项真 bug**（均写测试/真数据验证）：① freq dropout 分母 maxf→naifu median（旧 96% tag 钉 min_p、dropout 名存实亡）② composition_clean 近似→naifu 真值谓词（命中 80%→**20.6%=naifu**）③ anime 曝光 连续 priority→naifu integer-repeat 有界预算（修 cap 饱和）④/⑤ 多卡全局 stats 用全池+rank0 原子写+barrier ⑥ sample_one KeyError 兜底。**自查再发现并修 1 项**：anime producer 只读每 shard 头 20k（2% 覆盖 + shard0 头部 qz≥0.4 仅 15%）→改**随机 row-group**（qz 回到 97%）。证伪 5 项（GPIC@512泄漏/上采样/synth空caption/cluster-balanced/sample_one崩 在生产路径不可达，verify 真数据确认）。
**2026-06-17 第二份外部审计（独立 AI，基于修复前代码）+ 复核修复**：严肃核对后——**已修 3 真新问题 + 确认 2 项早已修 + 2 项分歧按讨论定**（决策均经用户拍板）：
- **F3 caption 真实硬截断**（实测 anime 256/512/1024=8.0/7.2/3.6% mid-token 硬截）→ **tokenizer 精确截断@296**（dataset 传 Qwen tokenizer，句/tag 界截到真实 token≤255=296-模板41）。调研确认 256 预算对（=i1 256 / nano-T2I 224-256，最接近的两参考；512 是 Qwen-Image 文字渲染才用、不必）。验证 **8%→0.00%**。
- **F4 CFG dropout 重复**（loader anime NL 10% + 模型 uniform 10% → NL≈19%、tag-only 被 drop 违反 config）→ **模型单点 dropout + dataset 给 tag-only anime 带 `cfg_keep` 豁免**。验证 loader 空串 0、tag-only 永不 drop。
- **F5 512/1024 首 batch 空等 GPIC queue 30s** → `_topup` 只对【有 producer 的域】阻塞。验证首 batch 49.4s→**25.2s**。
- **F1 anime 2% 覆盖 / F6 多卡 stats** = 上轮已修（随机 row-group / 全池+rank0 原子写）。
- **F2a cluster_balance 未实现**（分歧）→ 保持（anime 池已按 cluster 构造平衡：QC top10 簇 4.27%；naifu 也不在 sampler 重做）+ 加 cluster 曝光监控（`dataset.cluster_counter`）。
- **F2b max_repeat 未硬封顶**（分歧）→ 接受：integer-repeat 权重已封顶、E[曝光]∝repeat 忠实、rolling buffer 软界；with-replacement 流式不硬封单行 draw（文档记此语义）。
**版本管理**：mot_pilot_v1 已建 git 仓库（main 分支，`.gitignore` 排除 outputs/ckpt/pycache）；`docs/` 归档设计/数据/采样 spec/交接文档快照 + 采样 config（自包含）。今后改动注意提交。
**2026-06-17 数据配比/步数（Evan 拍板）**：512 重纳 GPIC（domain_mix 512=anime0.65/monet0.23/**gpic0.08**/synth0.04，real 0.31 切 0.742:0.258；GPIC@512 池~1.15M，实测已出现在 512）；**1024 不加 GPIC**（elig_1024=0、无 ≥1024px 图，物理不可行）；256 步数 300k→**400k**。新增交接文档 **§8 步数/样本表（强制 fixed_steps+fixed_samples 双列）**。
**2026-06-17 第三份外部审计 + 修复（4 项，均经用户拍板）**：**评估后这份较弱**（F4=重炒已决策 F2a/F2b；F1 把"对照臂设计"当"基线污染"，基线其实干净；F5 by-design）。真有价值的是 **F2**（ckpt 静默错续）：
- **F1 文档澄清**：`text_adapter.num_layers=0` 是**线性投影对照臂**（仍有可训练 out_proj/out_norm），**不是无 adapter**；真正无 adapter 基线 = `enabled=false`。改消融指南措辞，未动代码（设计本就如此）。
- **F2 ckpt 消融指纹**：`bidirectional` 只翻 causal mask、不改参数名 → 旧 `_ablation_mismatch_check`（只比参数名）挡不住 bidirectional↔causal 静默错续。新增 `_ablation_fingerprint`（存架构/消融开关）+ resume 比对，不一致清晰报错（旧 ckpt 无指纹只 warn 兼容）。测试过。
- **F3 数据线可观测**：active 域持续低水位→warn（mix 偏移）；连续 5 轮全空→报错而非 busy-spin（producer 卡死/数据源不可达早发现）。
- **F5**：dataset 无 tokenizer_path → 构造期 warn（退 proxy 截断、模型端可能硬截；3 个 stage config 已设）。
**还没做**：naifu 全迁移（**2026-06-19 评估后搁置**：原始动机「借其 dataio」已被自研数据线替代、IO 已 compute-bound，改走方案①自研补齐 wandb+grad-accum；完整 gap 清单+决策+执行计划见 [docs/naifu迁移评估与计划_v1_20260619.md](docs/naifu迁移评估与计划_v1_20260619.md)）、~~i1 真规模消融（G7，数据已就绪可排）~~ **G7 已于 06-18 正式启动（见下）**、B1 兜底 Variant B；数据线优化（offline VAE/caption 缓存、AR-bucketing、domain-mix buffer 调优）= 数据已就绪的后续，开训前按需。
> **后续增量 / eval 规划（2026-06-20）**：**eval 是当前最紧迫缺口**（消融 loss 判不了 v1/B1/B2/B3 优劣，须靠 eval 出 go/no-go）。可加内容路线图 + eval 三层方案 + eval 阶段1（消融期）实施计划见 [docs/后续增量与eval计划_v1_20260620.md](docs/后续增量与eval计划_v1_20260620.md)。

**2026-06-18 夜跑 6 实验 + 2×8 30h burn-in（结论已回填 [experiments/results/RESULTS.md](experiments/results/RESULTS.md)，醒来先看那份）**：
- IO 全 **compute-bound**（data_frac<0.04）→ 勿为 IO 优化花精力；**训练 buffer 用 128–512**（768 会域漂移：anime 漂到 0.43–0.55）。
- caption A/B：训练 loss **无法区分** caption 策略（A=1.010/B=1.0085 噪声内）→ 靠**截断率**（NL-heavy@256 超预算 20.6% vs tag-heavy 6.2%）+ prompt-following 判。建议 256 anime 路由更偏中短 NL 减截断。
- noise-shift：训练 loss 有 **sigma confound 不可直接当判据**；512/1024 弱倾向 shift=3.33，但**须固定-sigma 网格 eval / 样图复核**才能拍板（exp4 ckpt 已随 /local 清，需重跑小训练）。
- **timestep_multiplier 保持 1.0**（对齐 Cosmos3，早期无收益，别再扫）。
- **FLUX.2 VAE 对 anime ≥512 PSNR 反更高**（anime 36.5/36.8 vs real 35.7/35.5，高频误差仅 real 的 ⅓–½）→ **关闭"是否需 domain norm / 换 VAE"open question，anime 不换 VAE**。
- **基础设施可放心正式开训**：FSDP2/EMA/ckpt/resume 长跑可靠（burn-in 14k 步 0 NaN、ckpt 每千步写 22GB、peak 25G 稳）。
- anime R2 取图失败查清 = VAE-cache 8 节点高聚合并发**打 Cloudflare 限流**，非死链/坏图/数据问题；**在线训练不受影响**（低 QPS+磁盘缓存），已加 fetch 失败状态码捕获 + `fetch_stats_snapshot()` 可监控。

**2026-06-18→19 正式消融训练启动（G7 起步；最新状态见下方 2026-06-20→24 块）**，实时状态/运维见 [experiments/results/STAGE256_3RUNS_STATUS.md](experiments/results/STAGE256_3RUNS_STATUS.md)：
- **4×8 → 1×8 拍板改方案（Evan）**：实测 4×8(Socket/无IB)仅 ~24% 扩展效率，32卡 8.2s/step@global2048 ≈ 8卡 2.0s/step@global512 **同吞吐** → 1×8 用 ¼ 卡跑同样有效产出。改为 **4× 单节点 1×8 并行**跑 stage256 消融（v1 baseline + B1 adapter双向 + B2 long_skip(A_full) + B3 去噪声条件，消融指南 §4.3 优先级 ①②③）。附带好处：**world=8 时各域 shard 均≥8 → 数据占比天然正确**（4×8 会把 anime 偏斜到 ~0.16，因 anime 仅 17 shard < 32，超出 rank 拿 0 anime 被 renorm 丢）。
- **B1 adapter FSDP2/DTensor 崩溃已修（06-19 与 Evan 一起修+验证后作第 4 个 run）**：adapter 臂 und/gen 循环**直调自定义方法** `forward_und_self`/`forward_gen_adapter` 绕过 FSDP2 per-layer `__call__` all-gather hook → 参数仍 sharded DTensor → RoPE `q*cos` 混用报 `aten.mul got mixed Tensor and DTensor`。修复=`MoTDecoderLayer.forward` 改 **`mode` 分发器**（`joint`/`und_self`/`gen_adapter`），自定义路径改走 `layer(..., mode=...)` 经 `__call__` → 前向+反向 all-gather 正常。`mode='joint'`(默认 baseline)=原 `_forward_joint` 逐字不变、**baseline 零数值改动**（v1/B2/B3 安全）。手动 `unshard` 修不了可训 gen 的反向所以不用。验证：CPU `test_arch_changes.py` 全过 + 多卡 1×8 smoke 15 步无 DTensor 错。**遗留**：采样 KV 缓存路径(`precompute_und_cache`→`forward_und_capture`/`forward_gen_cached`)同类直调，将来多卡 FSDP2 采样需同法（加 mode 分发）修。
- 4 run 各 `resume=true`+`maxRetries=8` 保活、各自 checkpoint_dir（消融切换指纹校验拒复用别版 ckpt）、global512/LR1e-4 const+warmup300/400k 步上限/EMA0.9999/全在线数据。**~2s/step → 400k 步 ~9 天，消融信号 100–200k 步(~2.5–5 天)可看 go/no-go**。
- keep-alive yaml bug 已修：原 `echo` 结尾(exit0)会把 Python 崩溃误报 Succeeded 且不触发 maxRetries；改 `exit ${PIPESTATUS[0]}`。**resume 路径已实测验证**（重提交从 step1000 续到 1110）。
- /fsx 占用：4 个 last.pt ≈ 93GB（B1 因 +231M 稍大），超 50GB 软预算但 lustre 有 ~7TB free，已告知 Evan。
- 多机(>8卡)rendezvous 备忘见 [jobs/集群节点与多机训练备忘.md](jobs/集群节点与多机训练备忘.md)（jcc 不注入 torch rendezvous env，需自组 `NODE_RANK=${HOST##*-}` / `MASTER_ADDR`）；4×8 相关 results 目录已移入 `experiments/results/_archive/`（去掉 `归档_` 前缀），jobs yaml 仍以 `归档_` 前缀留在 `jobs/`。

**2026-06-27（数据线 complete_v1 = 5 域：TextAtlas 第5域 + domain_mix 供给饿死回归修复）= 接手先看这段**：
- **domain_mix 供给饿死回归修复（512/1024，commit 546bdd0）**：旧 `drain_mode=uniform`（每域无差别淘 `bs//2`）会把供给慢的域过度排空 → buffer 长期缺该域 → sampler renorm 后实际配比偏离目标（512 实测 anime 被顶到 +6.5%）。改 **`drain_mode=proportional`**（每批只淘各域【实际被抽中量】→ 慢域不被过排空、保持 present → renorm 落回目标）+ anime 串行取图（fetch_workers_anime=1，并发只过订阅挂载饿死 tar 域）+ gpic 逐文件并发（fetch_workers=4）+ real/synth 行内打乱 + M1 监控埋点（mix_*/buf_*/prod_*/data_wait）。**验证**：16-rank probe + 2×8 GPU 双验，512 四域全 ±5%、不 data-bound（data_wait 4.6%）、anime/gpic ≥目标。设计/诊断/根因见合并文档 [docs/complete_v1_dataline_设计与变更_v1_20260627.md](docs/complete_v1_dataline_设计与变更_v1_20260627.md)。
- **TextAtlas5M 第5域接入（text，注入域如 synthetic，commit 546bdd0）**：i1 开源长 caption 文字渲染数据。builder/summary 出 600 shard sidecar（elig/caption1-5，elig_512=4.04M、caption 覆盖 99.96%）；接入 `config.py`（`ALL_DOMAINS` + 路径 + 9 子集，丢 CoverBook）/ `manifest_index.load_textatlas_shard`（elig_256 由 w/h 现算）/ `complete_dataset.iter_textatlas_shard`（源 parquet 内联图按 image_path join、随机 row-group 去偏、`/local` 预存优先 + S3 回退保正确性）/ sampler 注入块 / caption_build 随机选 caption1-5 / config_schema。**配比（Evan 拍）**：512 text=0.10（从 monet 让渡、守 anime0.65/gpic0.08 底线）、256 text=0.10（比例×0.9 让渡）、1024=0（待定），依据 i1 等权配方（TextAtlas 单源 ~9.1%）。**验证**：单测全过 + 16-rank probe 5 域 ±0.1%（textatlas 未饿、S3 buffer 1.00）+ 2×8 GPU textatlas 0.093/data_wait 4.4%。设计见合并文档 [docs/complete_v1_dataline_设计与变更_v1_20260627.md](docs/complete_v1_dataline_设计与变更_v1_20260627.md)。
- **审计后修复（commit faab339）**：外部 AI 审计 546bdd0 后两处明确修复（核心 mix/textatlas 结论经独立复验通过）——**F3** stage512/1024 config 的 `dataset.stats_cache` 从 `/local` 改 **`/fsx` 共享盘**（对齐 256_v2；`/local` 是节点本地盘、多机直跑非 node0 rank 看不到 rank0 写的 cache → 重算/不一致）；**F4** `scripts/probe_realized_mix.py` 写完 `[probe-json]` 后 `loader.close()+os._exit(0)`（原生成器未显式 close → producer/取图线程残留 → 个别 rank 卡 Running）。F1（冷启动配比瞬态）/F2（metrics rank0-only）经核为非阻塞、Evan 拍板接受现状。数据线审计/变更全貌见合并文档 [docs/complete_v1_dataline_设计与变更_v1_20260627.md](docs/complete_v1_dataline_设计与变更_v1_20260627.md)。
- ⚠️ **坑点#11**：5 域 sampling config 另存 `t5_sampling_complete_v1_textatlas.json`（在 `/fsx data_analysis`，repo 外、未随 commit）；**原版 `t5_sampling_complete_v1.json` 保持 4 域**给在跑的旧代码 ablation（共享 /fsx 实时读，绝不能改）；`cleanup-merge` 的 256_v2/512/1024 configs 已指向 5 域版。

**2026-06-20→24（eval + B2/B3 停 + Ab2/Ab3 新消融 + 2×8 多机 + 多机解药 + 缓存实测 + 分支收敛）**：
- **stage256 四路评测（~110–116k）+ B2/B3 停（06-22→24）**：标量 loss/MSE 判不出四版（≤0.8%、固定-σ 0.13%）→ **别用 loss 判消融**。**B1 text-adapter 提示词/概念跟随强正向 → 保留**（overall hit +47%、角色 7.3×、系列 3.4×、CLIP 最高；general/character/copyright/CLIP/precision 五项一致 + DPG(mPLUG+Qwen-VL) 交叉验证居首；对抗核验 precision +34% 同升、非 tag 数伪影）。⚠️ 中段·单 seed·小样本（tagger n=150、DPG n=23）→ 512 用对齐步数 ckpt + 多 seed + 全量 DPG-1065 复验才下最终结论。画质/美学/loss/重建各臂打平、256 data-mix 无偏斜。据此**停 B2(long-skip，+8.7G 最贵且不占优)/B3(去噪声，免费简化留 512 确认)**，**v1+B1 继续训 ~42%**（自然跑满 400k）；加步号快照保留（`checkpoint_keep_every=50000`、`keep_max=4`，供复验/回滚）。报告 [EVAL_ablation_comparison.md](experiments/results/EVAL_ablation_comparison.md) / [EVAL_base_v1_analysis.md](experiments/results/EVAL_base_v1_analysis.md)。
- **新增 Ab2/Ab3 消融机制 + 三消融 2×8 多机开跑（06-24）**：在 B1 基础上新设计三变体（默认关、step0 等价、关时零参数开销）——**ab1causal**(单向 adapter)、**ab2fusion**(末 N 层多层融合 `MultiLayerFusion`，ELMo 式 gate zero-init → step0 与 B1 逐位等价)、**ab3lora**(frozen und 末 8 层 q/k/v/o 低秩残差 `LoRAResidual`，rank=32 alpha=32、`lora_und` 单独 param-group lr_scale 0.3)。接入 `configs/stage1a_t2i_256_v2.yaml`，2×8 多机跑（~13%，job `jobs/prod_2x8_{ab1causal,ab2fusion,ab3lora}_wt.yaml`）。⚠️ Ab2 eval 不能走缓存采样路径（`precompute_und_cache`/`forward_gen_cached` 忽略 fusion）。
- **2×8 多机产线就绪 + 多机慢根因解药（06-23/24，[multinode_scaling_20260623.md](experiments/results/multinode_scaling_20260623.md)）**：根因 = harbor 镜像缺 aws-ofi-nccl/libfabric → NCCL 退 eth0 socket（1.8GB/s）。**解药 = 换 ECR 镜像 C（`autobuild:py312-torch211-cu129`，自带 EFA）+ 继续 venv-gpu + plain 1D FSDP，零代码改动** → 跨节点 396GB/s、近线性（2×8 256=88%/512=94%、4×8 512≈93%）；EFA 下 HSDP 无增益（不用）。配方 `configs/stage1a_t2i_256_v2.yaml`：**global batch 固定 1536**（强扩展、加卡只加速，每卡 micro-batch 由 job 按 world 覆盖 2×8→96/4×8→48/...）、**133k 步 = 对齐 v1 样本预算**（≈204M）、lr 1.73e-4(=1e-4×√3)、warmup 1000。jobs：`s256_{2,4,6,8}x8_train.yaml`（镜像 C 固化）+ 上述三消融产线。**512 暂不急跑（Evan）**。⚠️ v2 config（batch/lr/use_latent_cache）绑多机配方、**绝不互喂在跑的 v1/B1 单机 run**（它们绑 `stage1a_t2i_256.yaml`）。
- **辅助实测**：VAE latent 缓存「256 不用」（读预处理 moments 比在线 encode 慢 ~5×、mountpoint-s3 冷 open I/O 瓶颈 → `use_latent_cache=false`，消费代码留脚手架；[vae_latent_cache_v1_20260622.md](docs/vae_latent_cache_v1_20260622.md)）；micro-batch 加大不提速（compute-bound、mb≈160 饱和、最多 +12% → 维持 mb64；[microbatch_throughput_bench_20260622.md](experiments/results/microbatch_throughput_bench_20260622.md)）；wandb 接入（rank0-only、默认关、故障降级 no-op 不中断训练）。
- **分支收敛（06-23，Evan 拍板）+ cleanup 解耦重构（06-24，行为不变，commit affc89c）**：`docs-reorg`/`wandb-integration`/`mntest` 三分叉**收敛进单一 `main`**（mntest 的 HSDP/bf16 探索丢弃、仅留诊断 `scripts/_archive/diagnostics/nccl_diag.py`；冻结配方 tag `archive/s256-ablation-running-20260622` 保留）。隔离 worktree `mot_pilot_v1_cleanup`（`cleanup-merge`）做了一轮**行为不变**解耦：四模块 `training/{checkpoint,diagnostics,config_schema}.py` + `networks/arch_mot_utils.py` 全抽离落地（详见 §0.0 box）。⚠️ **在跑 job 仍 `cd mot_pilot_v1`（或 `_ablations`）实时读代码 + 读 v1 config `stage1a_t2i_256.yaml`；切勿在活工作区 checkout 旧分支 / 改在跑文件**（坑点#11）。

---

## 1. 怎么跑（CPU 自检，全部应通过）

```bash
cd /fsx/evan/mot_code/mot_pilot_v1
PY=/fsx/evan/mot_code/venv-gpu/bin/python
$PY scripts/_archive/diagnostics/debug_mask_positions.py          # mask/MRoPE 断言
$PY scripts/test_arch_changes.py             # v3 baseline + i1 消融骨架 单元测试（11 项）
$PY scripts/test_flux2_vae.py                # FLUX.2 VAE round-trip + 集成（需 diffusers + 真权重）
$PY train.py --config configs/debug_tiny_cpu.yaml --max-steps 30   # tiny 端到端训练
$PY scripts/smoke_real_weights.py            # 真 2B 权重 smoke（~25GB RAM，分钟级；und vs HF max|Δ|=0）
$PY scripts/test_sampler.py                  # G6 采样器机制单测（zero-init 恒等/shape/确定性/CFG/σ调度）
$PY scripts/test_ema.py                       # G5 EMA 单测（数学/ckpt round-trip/滞后；多卡 DTensor 路径由 JCC 覆盖）
# --- 数据线 complete_v1（dataline；真数据需 S3/R2 可达）---
$PY scripts/test_review_fixes.py             # freq dropout median 分母 + sample_one KeyError guard
$PY scripts/test_global_stats_fix.py         # 多卡全局 stats：全池一致 + 原子写 + compute flag
$PY scripts/_archive/diagnostics/verify_anime_priority_fix.py     # 真数据：composition_clean 20.6%=naifu + integer-repeat 曝光
$PY scripts/smoke_loader_after_fixes.py      # 真数据：loader 4 batch、4 域出现、契约 + cfg_keep
$PY scripts/test_f4_cfg_dropout.py           # F4：模型单点 CFG dropout + tag-only anime cfg_keep 豁免
$PY scripts/_archive/diagnostics/audit_probe_trunc_repeat.py      # 真数据：caption 硬截断率 + integer-repeat draw 分布（审计核查）
$PY scripts/test_audit3_fixes.py             # F2 ckpt 消融指纹（bidirectional↔causal 报错）+ F5 tokenizer warn
```

**GPU 自检（单卡，2026-06-17 已全过；缓存务必指 /local）**：
```bash
cd /fsx/evan/mot_code/mot_pilot_v1
PY=/fsx/evan/mot_code/venv-gpu/bin/python
ENV="HF_HOME=/local/evan/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
# G2：SDPA 后端探测（确认 bool-mask 走 cuDNN 不退化 MATH + 吞吐）
env $ENV $PY scripts/_archive/diagnostics/probe_sdpa_backend.py --batch 8     # 也可 --batch 32
# G1：单卡 bf16 真规模训练（amp_dtype=bfloat16 已在 stage1a config，overfit 看 loss 下降）
env $ENV $PY train.py --config configs/stage1a_t2i_512.yaml --max-steps 400 \
  dataset.overfit_one=true dataset.mode=latents trainer.batch_size=8 \
  scheduler.num_warmup_steps=10 trainer.checkpoint_dir=/local/evan/mot_runs/g1 trainer.resume=false
# G6：端到端过拟合重建（真 2B 过拟合一固定 latent → 采样重建，MSE 应≪噪声基线 2.0）
env $ENV $PY scripts/_archive/one_time_scripts/overfit_reconstruct.py --steps 600 --bs 4 --sample-steps 50
# G4：本地单卡 FSDP2 码路（torchrun）；多卡走 JCC
env $ENV $PY -m torch.distributed.run --nproc_per_node=1 train.py \
  --config configs/stage1a_t2i_512.yaml --max-steps 8 dataset.overfit_one=true dataset.mode=latents \
  trainer.batch_size=4 model.gradient_checkpointing=false trainer.checkpoint_dir=/local/g4 trainer.resume=false
# G6 KV 缓存：真 2B 上验「fp32 单步≈0(数学) + bf16 单步 + 加速」
env $ENV $PY scripts/bench_sampler_cache.py --steps 30 --bs 2 --cfg 4.0
```
**多卡（JCC，2026-06-17 已验）**：`jcc job submit jobs/g4_fsdp2_validate.yaml`；看日志 `jcc job log <name> -p all -n 300`（⚠️ 不加 `-p` 会进交互菜单）。
⚠️ checkpoint 含 optimizer state，每个 ~16GB（1.42B trainable × fp32 params+Adam 双态）；调试 ckpt 写 /local、别写 /fsx。

---

## 2. 已完成 / 已验证（CPU）

**架构（[networks/arch_mot.py](networks/arch_mot.py) / [networks/model_mot.py](networks/model_mot.py)）**
- MoT 双路 + Cosmos3 权重划分；t2i 混合 mask；Interleaved-3D-MRoPE（生成图 t=text_len+2048）。
- **生成侧 sandwich norm**（Day-1 baseline，默认开，仅 gen 支、und 逐位不变、`max|Δ|=0` vs HF）。
- `timestep_multiplier=1.0`（Cosmos3 对齐）；**per-分辨率固定 noise-shift**（256→1.0 / 512→2.0 / 1024→4.0）。
- **i1 消融**（默认全关、关时零参数开销）：B2 long_skip / B3 noise_condition / B4 abs_sincos（zero-init→step-0 等价）+ **B1 text_adapter attention 臂已接入**：und 独立前向→adapter→gen cross-attend T_cond（不再逐层看 und），四消融臂（baseline / 中间对照 num_layers=0 / 因果 / 双向），真 2B 下 `_forward_und` vs HF `max|Δ|=0`。
- **Ab2 末 N 层多层融合（`MultiLayerFusion`，默认关）**：融合 und 末 N 层（ELMo 式 `fused = h_last + gate·(elmo − h_last)`，gate zero-init 标量 → step0 与当前 B1 逐位等价、与 LongSkip/abs_sincos 的 zero-init-gate 风格一致）后喂 adapter；config `text_adapter.multi_layer_fusion`。⚠️ 采样缓存路径（`precompute_und_cache`/`forward_gen_cached`）忽略 fusion → Ab2 eval 不能走缓存采样。
- **Ab3 冻结理解塔 und_lora（`LoRAResidual`，默认关）**：在 baseline joint path 上给 frozen und 的 q/k/v/o 投影挂低秩可训残差（`delta = scale·B(Ax)`，`MoTAttention.add_und_lora`）；末 N 层（默认 8 层，可配 `num_layers`）q/k/v/o 低秩残差（rank=32 alpha=32）、`lora_und` 参数走单独 param-group（lr_scale 0.3）；config `model.und_lora`。
- **B1 adapter 臂 FSDP2/DTensor 修复**：`MoTDecoderLayer.forward` 改 `mode` 分发器（`joint`/`und_self`/`gen_adapter`），自定义路径走 `layer(..., mode=...)` 经 `__call__` 拿 all-gather hook；`mode='joint'`(默认 baseline) 逐字不变、baseline 数值零改动。
- 文本 dropout 的 `text_rich` 特例 hook。

**VAE（[networks/vae.py](networks/vae.py) `Flux2VAE`）**
- FLUX.2 `AutoencoderKLFlux2`（f16c128）adapter：`encode` = `ae.encode().latent_dist.sample()` → 2×2 patchify → BN 标准化 → model-space `[B,128,H/16,W/16]`；`decode` 逆之。
- `ae.bn` 是 128ch **affine=False** BatchNorm，round-trip 对称（CPU 验证 `max|Δ|=4.8e-7`）。
- 还提供 `moments_to_model_space()` 给将来缓存数据线（raw moments → model-space）。
- ⚠️ **FLUX.2 latent 已 16×/128ch，故 MoT 侧 `patch_size=1`**（不再二次 pack），与 naifu/FLUX.2 标准一致。

**数据线（[training/dataline/](training/dataline/)，complete_v1 全忠实）**
- 4 域：anime（源 parquet 17 shard 自驱，R2-HTTP 取 WebP）/ MONET-real+synth（daily_v1 manifest，tar 顺序流式取图、回源 caption）/ GPIC（manifest + 本地文件，512/1024 退场）。
- 采样：anime+synth 固定比率注入、MONET+GPIC band-first（band→domain∝domain_mix→域内 base_w=source^-0.2×cluster^-0.2）；anime **integer-repeat 曝光**（naifu weighted-budget，priority=clip(ct_weight+quality+Σlane+composition_clean,1,3)、extra_frac=0.25 按 priority-1 加权、封顶 max_repeat、每行≥1）。
- caption：MONET 按 band→captioner 优先级、anime 原生路由（rich_nl/nl/nl_short/tag_medium 按 stage、anchor keep、rating q/e、freq dropout median、length cap、224-256 截断、NL 行 10% CFG dropout）。
- 流式：per-域后台生产者→bounded queue→shuffle buffer→sampler→在线 VAE encode；rank shard-level 切分；anime **随机 row-group** 取样去头部偏置。
- 8 条 loader 规则全落地（holdout/band-by-tok/GPIC@512退场/全局 stats 一次/.tar 取图）。

**测试**：`test_arch_changes.py`(12) + `test_flux2_vae.py` + mask + tiny-train + real-2B smoke 全绿。数据线 `test_review_fixes` / `test_global_stats_fix` / `verify_anime_priority_fix` / `smoke_loader_after_fixes` 全过（真数据）。经多轮多维对抗 review（baseline 6 维→修 6 项；B1 4 维→clean；数据线 4 维→修 6+1 项）。

---

## 3. GPU 待办清单（交付：按依赖序，每项给验收标准）

> 现状（2026-06-17）：**G0–G2 已在单卡 H100 80GB 完成**（见下 ✅）；架构 + VAE 早先 CPU 逻辑验证通过。下一步从 **G3（数据线）/ G4（多卡）** 起，动手前先用对话框和用户对齐范围。
> 每项改完都要同步：本文件 + `README.md`「已验证/TODO」+ `env/requirements.txt`；新决策先问用户。

**G0 · GPU 环境 / 真权重 smoke ✅ 完成（2026-06-17）**
- 机器：单卡 **NVIDIA H100 80GB HBM3**（driver 580/CUDA13），`venv-gpu` torch 2.11+cu128 `cuda.is_available()=True`、transformers5.11/diffusers0.38 齐。
- `smoke_real_weights.py`：und & B1 `_forward_und` vs HF **`max|Δ|=0.000e+00`**；trainable=1.416B/frozen=1.721B；373 可训练有梯度/310 frozen 无梯度。`test_flux2_vae.py`：patch/BN 互逆 **Δ=4.8e-7**、集成 forward 跑通。两脚本全过（CPU 路径，属正确性回归）。

**G1 · 单卡真规模训练 step（bf16）✅ 完成（2026-06-17）**
- 实现：[train.py](train.py) 加 **bf16 autocast**（`trainer.amp_dtype=bfloat16` 门控；参数留 fp32 master + `torch.autocast(cuda,bf16)`，bf16 无需 GradScaler；loss 已 `.float()`；CPU/未设时退化 fp32）。选 autocast 而非 `model.to(bf16)`：先拿数值最稳的单卡基线，**纯 bf16 路线留 G4 迁 naifu/FSDP2 时切**（用户拍板）。
- 验收：真 2B + stub latents + 512px + overfit_one，**loss 2.0→~0.3**（400 step）单调下降、grad_norm 0.3–2.2 无 NaN。step/s 与显存见 G2 表。pixels→真 FLUX.2 VAE→MoT 路径 GPU 亦跑通（loss 1.63→1.35）。

**G2 · SDPA mask 吞吐 + 后端 ✅ 完成（2026-06-17）**
- 工具：新增 [scripts/_archive/diagnostics/probe_sdpa_backend.py](scripts/_archive/diagnostics/probe_sdpa_backend.py)（复刻真实形状 + bool mask，逐后端 force 探测 + 计时）。
- **结论：bool-mask SDPA 不退化到 MATH** —— PyTorch 2.11 在 H100 自动选 **cuDNN attention**（吃 bool mask）。bs8/512px：cuDNN **0.43ms** vs mem-efficient 1.0ms vs MATH 6.6ms vs FLASH+无mask 0.36ms（理论峰值）。cuDNN 仅 **1.16× flash 天花板**，**无需迁 flex/packed+FA varlen**（坑点#4 担心的退化未发生）。
- **端到端单卡 step / 显存（512px, bf16, overfit）**：

  | 配置 | iter/s | 峰值显存 | img/s |
  |---|---|---|---|
  | bs8 grad_ckpt **off** | 0.53 | **74.3GB** | 15.1 |
  | bs8 grad_ckpt on | 0.69 | 35.4GB | 11.6 |
  | bs32 grad_ckpt on | 2.40 | 41.4GB | 13.3 |
  | bs32 grad_ckpt **off** | — | **OOM** | — |
  | bs64 grad_ckpt on | 4.71 | 58.2GB | 13.6 |

  → **grad_ckpt 必开**（关掉 bs8 就 74GB、bs32 OOM）；单卡 512px bs32(41GB)/bs64(58GB) 从容，约 13 img/s。

**G3 · 数据线接入 ✅ 完成（2026-06-17，complete_v1 全忠实）**
- 实现 [training/dataline/](training/dataline/)（见 §2「数据线」）：4 域真实数据、流式生产者+buffer+CompleteSampler、在线 VAE encode、8 条 loader 规则全落地；configs 256/512/1024 已接 `CompleteV1Dataset`。
- **验收**：① 真数据+真 3.1B+真 FLUX.2 VAE 训练 20 step **loss 2.0→1.81、无 NaN、41GB**；② 对抗 review（15-agent）修 6 真 bug + 自查修 anime 头部偏置，全写测试/真数据验证（命中率、曝光分布、多卡一致性）。
- **后续优化（数据已就绪、开训前按需）**：offline 缓存 raw moments（参考 `naifu-main/scripts/batch_vae_flux.py` + `Flux2VAE.moments_to_model_space()`）省在线 VAE；AR-bucketing；domain-mix buffer 调优（流式有限 buffer 致 mix 近似，sampler 本身精确）；抽预处理 latent decode 回像素肉眼盯线条/网点/文字 crisp。

**G4 · FSDP2 多卡 ✅ 完成（2026-06-17，Route B = 本地 train.py + FSDP2；用户拍板）**
- 路线决策：数据 hold 期选 **Route B**（本地 train.py 补 FSDP2，用已验证 venv，快、无 naifu env 摩擦）；naifu 全迁移留数据/dataio 落地再做（那时其 dataio 才有价值）。
- 实现 [train.py](train.py)：torchrun/dist 检测（LOCAL_RANK）→ FSDP2 per-layer `fully_shard` + **MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32, cast_forward_inputs=True)**（= naifu 配方，fp32 master shard + bf16 compute + fp32 梯度规约；**FSDP2 路径不再叠 autocast**，单进程路径仍 autocast）。ckpt 用 DCP `get/set_state_dict(full_state_dict)` gather/scatter rank0。新增 `model.rank_independent_seed` 开关做严格一致性验证。
- 验收（JCC 2×H100，`jobs/g4_fsdp2_validate.yaml`）：

  | | RUN A 1卡 bs4 | RUN B 2卡 bs4/卡(rank_indep) | RUN C 2卡 bs32/卡+ckpt+EMA |
  |---|---|---|---|
  | step1/step30 loss | 2.0013 / 2.0258 | **2.0013 / 2.0295** | 1.9997 / (s25)1.9494 |
  | iter | 0.35s | 0.32s | 2.20s |
  | 吞吐 | 11.4 img/s | **25 img/s(2.2×)** | 29 img/s |
  | 峰值/卡 | 44.9GB | **33.2GB** | 28.1GB |

  → **2卡 vs 1卡近逐位一致**（前几步 bit-级、全程 ≤0.5% bf16 漂移）；**分片显存**降（44.9→33.2 同 per-GPU batch；RUN C bs32/卡 28.1GB < 单卡 bs32 41.4GB）；**吞吐 ~2.2× 近线性**。本地 nproc=1 另验过 resume + 真2B 28-layer 分片。

**G5 · EMA ✅ 完成（2026-06-17）**
- 实现 [model_mot](networks/model_mot.py) `init_ema/update_ema/ema_state_dict/load_ema_state_dict`：仅可训练参数、EMA 镜像 sharded DTensor、shard-wise `lerp_`，ckpt 用 `full_tensor()` gather / `distribute_tensor` scatter。train.py 接入（init 在 fully_shard 后/resume 前、每 step update、ckpt 存/读）。
- 验收：[scripts/test_ema.py](scripts/test_ema.py)（CPU 单测：数学 `ema=decay·ema+(1-decay)·param` Δ=1.5e-8、ckpt round-trip 逐位一致、滞后平滑）+ nproc=1 FSDP2 + JCC RUN C 多卡（373 trainable tensors，跨2卡 ckpt gather 无错）。

**G6 · 采样器 / 推理 ✅ 最小版完成（2026-06-17）**
- 已实现 [model_mot.generate_samples](networks/model_mot.py)：**Euler flow 采样 + 经典双前向 CFG + VAE decode**。
  - Euler：σ 从 1→0（含训练同款 shift warp），`x += (σ_next−σ_cur)·v`，v=模型预测=noise−x0=dx/dσ。
  - σ schedule **用训练同一个 shift（SD3/FLUX 乘式 current_shift）**，**非** naifu flux2 的 empirical-mu（那是 FLUX.2 专用、与我们训练 shift 不符）。
  - CFG：`v=v_u+scale·(v_c−v_u)`，scale==1 退化单前向。我们 caption-dropout 训练、**不用** naifu 的 guidance-embed。
- 验证：① [scripts/test_sampler.py](scripts/test_sampler.py)（CPU tiny）机制全过；② [scripts/_archive/one_time_scripts/overfit_reconstruct.py](scripts/_archive/one_time_scripts/overfit_reconstruct.py)（GPU 真 2B）**过拟合重建 MSE 0.041 vs 噪声基线 2.002 = 0.02×**，端到端确证 Euler 符号 / σ 方向 / VAE 路径正确。
- **✅ und 逐层 KV 缓存（2026-06-17）**：[arch_mot](networks/arch_mot.py) `precompute_und_cache`（loop 外跑一次 frozen und 塔，缓存每层 post-rotary/post-repeat K/V）+ `forward_gen_cached`（loop 内只跑 gen 塔 attend 缓存）；[model_mot.generate_samples](networks/model_mot.py) `use_und_cache`（默认开；baseline 用、adapter 臂回退 per-step）。**1.62× 加速**。验证：[test_sampler.py](scripts/test_sampler.py) [B5] CPU fp32 cached vs 非cached **逐位 0** + [bench_sampler_cache.py](scripts/bench_sampler_cache.py) 真2B fp32 单步 4e-6（数学）/ bf16 单步 13%（28层und塔 reduction 累积）+ overfit_reconstruct 端到端 **cached=非cached（MSE 0.026 一致、diff 0.0000）→ bf16 偏差完全 wash out**。
- **余项（未做）**：真实出图 + 与 Anima-2B 头对头（需先训练 + 数据/eval，方案 Eval 节 GenEval/DPG/MJHQ + 二次元专项）。

**G7 · i1 消融 + anime** ［进行中：① stage256 四路评测已出（~110–116k）= **B1 暂定保留**、v1/B2/B3 标量并列 → **据评测 B2/B3 已停、v1/B1 继续训 ~16.5–17 万/40 万≈42%**；② **新设计三消融 ab1causal/ab2fusion/ab3lora 在 2×8 多机跑 ~1.7 万/13.3 万≈13%**；详见 §0 末 2026-06-20→24 块 + STAGE256/EVAL_*/multinode 报告］
- 第一轮消融排期（256 便宜单元、100–200K 步、bs512、信号 DPG+tagger+自建二次元）：① B1 text_adapter（四臂含**中间对照臂** `num_layers=0`）→ ② B2 long_skip → ③ B3 去噪声条件 → ④ B4 abs_sincos。代码开关都已就绪（§2）。**已跑完中段评测**。
- 第二轮三消融（基于 B1，2×8 多机、global 1536、133k 步对齐 v1 样本预算）：**ab1causal**(单向 adapter) / **ab2fusion**(末 N 层多层融合) / **ab3lora**(und LoRA)，已实现集成（§2）+ 在跑。
- B1 兜底 Variant B（逐生成层 zero-init ΔK/ΔV）尚未写。
- ⚠️ Ab2 eval 不能走缓存采样路径（fusion 被 `precompute_und_cache`/`forward_gen_cached` 忽略）→ eval 时另行处理。
- anime manifest 未建（数据 gap，见 §6 / 数据交付文档）；FLUX.2 VAE 对 anime 域是否需 domain norm 复核（LPIPS/rFID/线条/网点，VAE gate 选了 FLUX.2 但 anime 偏移未单独测）。

---

## 4. 坑点（踩过的，直接抄）

1. **HF 加载陷阱**：`Qwen3VLTextModel.from_pretrained(全量ckpt)` 会因 `model.language_model.*` 前缀全 MISSING **静默拿随机权重**。必须 `Qwen3VLForConditionalGeneration.from_pretrained(...).model.language_model`（smoke 脚本已修正）。
2. **diffusers 装法**：venv-gpu 原没 diffusers。用 `pip install --no-deps diffusers requests`（`--no-deps` 避免动 torch/transformers；`requests` 是 diffusers 运行期硬依赖、venv 原缺）。已装 diffusers 0.38.0。`accelerate` 没装只是 warning，不影响。
3. **bf16**：模型默认 fp32。GPU 上 train.py 需小改（autocast 或 `model.to(bf16)`，loss 已 `.float()`）。VAE config 设了 `dtype: bfloat16`。
4. **SDPA mask（已实测，2026-06-17）**：我们用 bool `attn_mask` 的 SDPA。FlashAttention 确实不吃任意 mask（`No available kernel`），但 PyTorch 2.11 在 H100 **自动落到 cuDNN attention**（吃 bool mask、近最优、仅 1.16× flash 天花板），**没有退化到 MATH**——`scripts/_archive/diagnostics/probe_sdpa_backend.py` 可复现。结论：当前**不必**迁 flex_attention / packed+FA varlen。注意 venv 没装 flash-attn 也无所谓（bool mask 本就用不了 flash）。
5. **`patch_size`**：FLUX.2 用 **1**（VAE 已 16×/128ch）；只有 `debug_tiny_cpu.yaml`（FakeVAE c16）用 2。改 head_dim 时 tiny 的 `mrope_section` 之和必须 = head_dim//2。
6. **checkpoint**：只存可训练参数（frozen 塔由 pretrained_path 复原）。**切架构 / 消融开关后不能 resume 旧 ckpt**（model + optimizer 参数集都变）——[train.py](train.py) `load_checkpoint` 会给清晰 RuntimeError，请换新 `checkpoint_dir` 从头训。
7. **VAE 权重位置**：在 `/fsx/models/FLUX.2-klein-4B/vae`（config `_name_or_path` 指向 FLUX.2-dev，是共享 autoencoder）；`/fsx/models/FLUX.2-dev/vae/` 是**空的**。
8. **noise_condition=false**（B3 消融）：此时不构造 `time_embed`（省参数、无孤立可训练参数）；baseline 默认 true 不受影响。
9. **数据**：manifest 是 pointer model，存 locator+token 数；训练时按 locator 读源。**图像读 `.tar` 全分辨率，别读 parquet 的 thumbnail（≤384px）**。
10. arxiv 被 block（curl 403）；HF / github / pypi 可用。长任务 nohup + log 轮询（sleep/凭证探测类命令可能被安全策略拦）。
11. **JCC job 读的是 /fsx 上实时文件**：提交 job 后别再改它要跑的 `train.py`/configs——本轮就因边跑边改，RUN A/B 用了 pre-EMA、RUN C 用 post-EMA 代码（幸而 loss 对 EMA 无关、也没读到半文件）。教训：提交后冻结代码，或先 compile 校验再改、改完立即校验。
12. **FSDP2 mp_policy 与 autocast 不可叠**：`MixedPrecisionPolicy(param_dtype=bf16, cast_forward_inputs=True)` 已把输入/计算转 bf16，再套 `torch.autocast` 是双重转换。train.py 已分流：dist 走 mp_policy（无 autocast）、单进程走 autocast。
13. **`jcc job log` 默认交互**（弹 pod 选择菜单）：脚本里必须 `-p all`（或 `-p <pod>`）+ `-n <行数>` 非交互拉日志。`jcc node list` 看空闲 GPU、`jcc job list` 看自己的 job。
14. **FSDP2 EMA ckpt**：EMA 不是 FSDP module，gather 用 DTensor `full_tensor()`（collective，所有 rank 须调，放在 rank0 落盘前）、scatter 用 `distribute_tensor`。单进程退化为普通张量。
15. **KV 缓存 bf16 单步偏差≠bug**：und KV 缓存路径在 fp32 下与 joint forward **逐位一致**（CPU 0、真2B 4e-6），但 bf16 下单步 v_pred 偏 joint **~13%**——因 cached 孤立跑 und SDPA（s_t keys）vs joint 跑 [und;gen] SDPA（masked），形状不同→reduction 顺序不同→28 层 und 塔累积。**端到端无害**：过拟合重建 cached=非cached（MSE 0.026 一致、diff 0.0000，采样器自校正 wash out）。判 KV/分片类优化对错要看 **fp32 逐位 + 端到端**，别被 bf16 单步偏差吓到。
16. **frozen 子塔的独立前向用 `no_grad`、不只是 checkpoint**：B1 adapter 臂 `_forward_und` 跑的是全冻结 und 塔——正确做法是整段 `@torch.no_grad()`（零激活、零重算），比 gradient checkpointing（存输入+重算）更省；可训练支才用 checkpoint。另：**外部审计要逐条拿代码核实再采纳**——本轮 5 条里 4 条成立（B1 参数 16%/绕过 ckpt/save_final 脚枪/滞后文案），1 条（B1"归因污染"）是对 4 臂消融阶梯设计的误读。
17. **anime 源 shard 内部按 qz/cluster 有序，别读头部**：源 parquet 头部样本系统性偏低质量（shard0 头 20k 的 qz≥0.4 仅 15% vs 池 82%）。producer 必须**随机 row-group + 组内随机 offset** 取 nrows（`load_anime_shard(rng=...)`），否则只训到最差 2% 且永不覆盖全 shard。real/synth 读全 manifest shard 无此问题。**判数据偏置要在代表性切片（整 row-group）上核，别用头部 N 行**（本轮 composition_clean 命中率头 30k=9.6% vs 全 rg=20.6%，差一倍）。
18. **freq dropout 分母用 median 不是 max**：naifu `frequency_weighted_tag_dropout` 的 ref=`max(median(freq),1.0)`；误用 max(freq) 会让 ~全部 tag 的 f/maxf≪1、dp 钉 min_p，dropout 名存实亡（实测旧版 96% tag 钉 min_p、保留率 90%）。中位频率 tag 应 ~base_p、高频顶 max_p、长尾压 min_p。
19. **config 是本项目权威（覆盖 naifu 默认）**：anime priority `base=ct_weight`（非 naifu 1.0）、`max_priority=3.0`、含 composition_clean_bonus，都是 config 显式覆盖（exec doc §8）。但 composition_clean 的**谓词定义**仍取 naifu（simple-bg tag + 无 watermark/sig/logo/username + YOLO added_text_area<0.005 + qz≥0.4，常量见 `manifest_index.py` / naifu `tag_utils.py`）——源 parquet 139 列**有** tags_general/YOLO/quality_z，可精确实现，别用 risk_bucket 近似（命中 80% vs 真值 20%）。
20. **anime 曝光是 integer-repeat 不是连续加权**：config `priority._note` + naifu `sampler._build_sampling_repeat_counts` = 固定 extra 预算（`extra_frac=0.25·N`）按 `score=max(priority-1,0)` 加权分配、封顶 `max_repeat`、每行≥1（**有界 boost**）。直接把 priority 当连续 cumsum 权重会无界放大高 priority、cap 饱和时退化近均匀。流式实现：每 buffer 重算 repeat count 作权重（per-draw caption 重 roll 天然满足 naifu caption_index 变体）。
21. **多 rank 全局 stats 必须用全池**：rule #7 source/cluster 统计要 over【未按 rank 切片】的全量 monet 列表（否则各 rank base_w 不一致）；多卡协调 = rank0 算+**原子写**（temp+os.replace+fsync）+barrier+非rank0 读缓存（`complete_dataset._build_global_stats`）。`stats_cache` 指 /local。
22. **caption 截断要用真实 tokenizer、不能只靠 proxy**：manifest 的 tok 是 words×1.32 代理（给 band 分配用），但真实 Qwen token ≈words×1.41（更密）；只按 proxy 截到 256 → 真实 ~273 token + 模板 39 = 312 > 296 → 模型 `truncation=True` 静默 **mid-token 硬截 ~8%**（anime@256）。修：dataset 传 `tokenizer_path`，`truncate_to_token_budget` 按真实 token 在句/tag 界截到 ≤255。判 caption 长度问题要用真 tokenizer 在代表性数据上量，别信 proxy。**文本长度参考**：i1=256 / nano-T2I=224-256（最接近，Qwen-VL 系 T2I）；Qwen-Image=768（文字渲染）；Cosmos3=4096 / BAGEL=16384（交错多模态，不可比）。我们 296（=模板39+caption257）对齐 i1/nano。
23. **CFG dropout 只在模型一处做**：数据线**别**自行 dropout caption（早前 anime NL 在 build_anime_caption 里 10% 置空，叠模型 10% → NL 19% 双重、tag-only 被模型 drop 违反 config）。统一：模型 `_apply_cfg_dropout` 唯一点，数据线给【tag-only anime】带 `cfg_keep=True` → 模型对其用 prob 0（config『tag-only 永不 drop』）。real/synth + anime NL 走模型 10%。
24. **`_topup` 只对有 producer 的域阻塞**：本 stage 不参与的域（512/1024 的 GPIC，domain_mix=0 没起 producer）queue 永空；首轮去阻塞它会白等满 30s 超时（首 batch 多花 ~24s）。用 `active` 集合（实际起 producer 的域）gate 阻塞分支。
25. **anime integer-repeat 是 with-replacement 软界**：曝光权重=naifu 有界预算 repeat count（≤max_repeat、E[曝光]∝repeat 忠实），但流式 with-replacement **不硬封单行 draw 次数**（buffer 驻留期可被抽>max_repeat 次）；rolling buffer 限制驻留→实际过曝有限。这是设计选择（流式无 epoch、硬封顶是物化-epoch 概念），非 bug。
26. **`text_adapter.num_layers=0` ≠ 无 adapter**：它是**线性投影对照臂**（blocks 空，但仍构造**可训练** out_proj/out_norm，zero-init 起点等价、训练中学线性变换 + 逐层 text_k/v_proj_moe_gen）。**真正的无 adapter baseline = `text_adapter.enabled=false`**（根本不建 TextAdapter）。比 ablation 归因时别把 num_layers=0 当无-adapter 基线。
27. **ckpt resume 要比"消融指纹"、不只比参数名**：`text_adapter.bidirectional` 只翻 causal mask、**不改参数名/形状**，`_ablation_mismatch_check`（只比参数名）挡不住 → bidirectional↔causal 会静默错续。已加 `_ablation_fingerprint`（存 enabled/bidirectional/num_layers/long_skip/noise_condition/abs_sincos/sandwich_norm/patch_size）+ resume 比对报错。新 ckpt 才有指纹，旧的只 warn。换消融开关务必换 `checkpoint_dir`。
28. **纯 CPU 任务也走 JCC、别在本机/登录机直接跑**（Evan 2026-06-23 要求）：架构单测（`test_arch_changes.py`）、smoke、loader 验证、缓存 QC 等即便是 CPU 任务，也**提 JCC job**（CPU 或轻量 GPU 节点）跑，别在本机 `python ...`——本机内存有限，真 2B 加载 ~25GB RAM / 并发易触发 **OOM killer** 把进程或整个会话杀掉。提交 JCC 时 `trainer.save_final=false` + 输出/缓存指 `/local`，零 /fsx 写。只有秒级、零模型加载的纯文件/git 操作可本机直接做。

---

## 5. 用户（Evan，Mewtant 实习生，evan@mewtant.io）偏好

- **干货、表格对比、直接表态**；中文为主、术语保留英文；不重复造轮子。
- **凡需决策 / 拍板 / 信息不明 / 提供偏好 → 先用对话框（AskUserQuestion）问，多问**，不要擅自定。今后需要对话的一律用对话框选项形式。
- 改动要**确保正确性、写测试验证**；不擅自扩大范围（做事聚焦当轮任务）。
- 重大设计点即使有推荐，也先说明清楚再让他拍板。

---

## 6. 路径导览（其他 AI 按需查）

**设计正本（doc/mot）**
| 路径 | 是什么 |
|---|---|
| [doc/mot/README.md](/fsx/evan/doc/mot/README.md) | 文档总入口（哪些是正本、哪些归档） |
| [doc/mot/自己写的模型+训练设置文档_v3.md](/fsx/evan/doc/mot/自己写的模型+训练设置文档_v3.md) | **模型/网络架构 + 训练设置的设计正本**（本代码对齐它） |
| [doc/mot/模型架构修改规划_v1_20260616.md](/fsx/evan/doc/mot/模型架构修改规划_v1_20260616.md) | 把 v3 落成代码的逐项 spec + 决策记录（含 i1 消融排期） |
| [doc/mot/模型架构实现现状与消融指南_v1_20260617.md](/fsx/evan/doc/mot/模型架构实现现状与消融指南_v1_20260617.md) | **当前架构长什么样 + 默认超参 + v3 逐条实现对照 + 消融怎么跑 + 决策记录**（人/AI 理清现状用） |
| [doc/mot/自己写的数据集文档_v3.md](/fsx/evan/doc/mot/自己写的数据集文档_v3.md) | 数据集设计正本 |
| [doc/mot/自己写的后续试验.md](/fsx/evan/doc/mot/自己写的后续试验.md) | RAEv2/RAC/E0-E9 后续实验正本（**不在当前 baseline 范围**） |
| [doc/mot/AAA数据交付文档_GPIC+MONET+anime_v2_20260617.md](/fsx/evan/doc/mot/AAA数据交付文档_GPIC+MONET+anime_v2_20260617.md) | 数据线交付权威（指标/路径/schema/loader 规则） |
| [doc/mot/采样权重spec_v1_20260616.md](/fsx/evan/doc/mot/采样权重spec_v1_20260616.md) | 采样权重 spec（含 8 条 loader 规则） |
| [doc/mot/i1_深度分析与对我们的启发_20260611.md](/fsx/evan/doc/mot/i1_深度分析与对我们的启发_20260611.md) | i1 论文分析（消融增量的依据） |

**代码（mot_code）**
| 路径 | 是什么 |
|---|---|
| `mot_code/mot_pilot_v1/` | **本项目**（分支 `main`）：MoT 模型 + 训练代码（README.md 必读）。切换后的唯一正式仓库；当前暂落后最新——**开发先去 `mot_pilot_v1_cleanup`**（见 §0.0） |
| `mot_code/mot_pilot_v1_cleanup/` | **当前最新整合代码**（分支 `cleanup-merge`）：开发在此；训练跑完后 FF 进 main、本 worktree 删除（见 §0.0） |
| `mot_code/mot_pilot_v1_ablations/` | 临时 worktree（分支 `ablations-v2`）：三消融 2×8 直播在此运行；切换后删除 |
| `mot_pilot_v1/training/dataline/` | **complete_v1 数据线**：`config.py`(采样 config 解析)、`manifest_index.py`(枚举/全局 stats/load shard+priority+band)、`sampler.py`(4域 domain_mix+band-first+anime integer-repeat)、`caption_build.py`(MONET band-captioner + anime 原生路由/freq dropout)、`image_fetch.py`(GPIC文件/tar流式/R2-HTTP+缓存)、`complete_dataset.py`(`CompleteV1Dataset` 流式生产者+buffer+在线VAE encode) |
| `mot_code/naifu-main/` | **参考训练框架**（公司内部 FSDP2/dataio/wandb）。重点：`train_fsdp2.py`（FSDP2 主循环 + model_cls 契约）；`networks/ema.py`（EMA 实现，G5 抄）；`networks/{arch_flux2,model_flux2}.py` + `scripts/batch_vae_flux.py`（**FLUX.2 VAE 用法权威**，我们对齐它）；`inference/inference_flux2.py`（采样参考，G6）；`networks/{arch_qwen_image,arch_z_image,model_z_image,arch_hidream_o1}.py`（其它 T2I arch 参考） |
| `mot_code/troph-team_important_t2i/` | **公司 T2I 相关仓库浅克隆**（19 个：nano-t2i / 数据 / eval / caption / 打分 / 蒸馏等），按需参考触发器。索引 + 按需求反查见 [docs/公司参考代码索引_v1_20260619.md](docs/公司参考代码索引_v1_20260619.md)；最对标 = `nano-t2i-private` |
| `mot_code/data_analysis/` | 数据线代码（筛选/采样/聚类）；`t5_sampling_weights.py` 参考实现，`cluster/README_T2_NEXT.md` 处理流水账 |
| `mot_code/venv-gpu/` | Python 环境（torch 2.11+cu128 / transformers 5.11 / diffusers 0.38 / einops）。解释器 `venv-gpu/bin/python` |

**参考项目源码 repo（读源码用，根目录 `doc/mot/归档/old_训练框架选型调研/repos/`）**
| repo | 看什么 |
|---|---|
| `repos/cosmos-framework/` | **Cosmos3 框架**：`cosmos_framework/model/vfm/mot/{unified_mot,attention,modeling_utils}.py` —— 我们 MoT 权重划分 / `*_moe_gen` 命名 / timestep 加性注入的**直接来源** |
| `repos/BAGEL/` | BAGEL UMM：生成塔复制初始化、理解图 bidirectional mask 方案参考 |
| `repos/SenseNova-U1/`、`repos/Ovis-U1/` | U1 系：CE:MSE 配比(0.1:1)、理解图 mask 选择 |
| `repos/Lance/` | constant LR + wd=0(frozen LM) 配方来源 |
| `repos/{OmniGen2,Janus,Show-o,Emu3,BLIP3o,HiDream-O1-Image,HunyuanImage-3.0,InternVL-U,Lumina-DiMOO,TUNA-2}/` | 其它 UMM/T2I 实现，按需查具体设计 |
| `doc/mot/paper_filtered/i1/`（jax 源码） | **i1**：`jax/models/{dual_stream_backbone,single_stream_backbone,components}.py`（adapter/long-skip/sandwich/abs-pos）、`jax/configs/i1_training/`（shift/optimizer/EMA 数值）、`jax/diffusion/rectified_flow.py` |

**参考论文 PDF（`doc/mot/paper_filtered/`；⚠️ arxiv 被 block、别重下；细节先 grep `doc/mot/mot_reading_report/data/page_text/`）**
| PDF | 关键 |
|---|---|
| `2606_cosmos3.pdf` | Cosmos3（MoT 主参考） |
| `2606_RAEv2.pdf` / `RAC_2603_rac.pdf` / `2606_RF_UMM.pdf` | RAEv2 / RAC / RF-UMM（**后续试验、不在 baseline**，见 `docs/自己写的后续试验_v1_20260616.md`） |
| `2505_BAGEL.pdf` / `2510_LightBagel.pdf` / `2509_Hyper-Bagel.pdf` | BAGEL 系 |
| `2605_SenseNova-U1.pdf` / `2605_Lance.pdf` | U1 / Lance |
| `2605_Qwen-Image-VAE-2.0.pdf` / `2511_Z-Image.pdf`(sandwich norm 产线) / `2410_DC-AE.pdf` | VAE / Z-Image / DC-AE |
| `2605_Qwen-Image-2.0.pdf`(混合 CFG) / Show-o / Transfusion / Janus-Pro / Emu3 等 | 按需 |
| **阅读报告** `doc/mot/mot_reading_report/{mainstream_umm_data_training_tables.md, paradigm_shift_2025_to_2026.md, index.html}` | 默认只读这三 |

**模型权重（/fsx/models）**
| 路径 | 是什么 |
|---|---|
| `Qwen3-VL-2B-Instruct` | **理解路骨干**（hidden 2048/28层/16头/KV8/head_dim128/mrope[24,20,20] interleaved）。另有 `Qwen3-VL-8B`、`Qwen3-VL-30B-A3B`（更大 und 候选） |
| `FLUX.2-klein-4B/vae` | **在用的 FLUX.2 VAE**（AutoencoderKLFlux2 f16c128 = FLUX.2-dev 共享 autoencoder）；`FLUX.2-dev/vae/` 是空的 |
| `Qwen-Image/`、`Wan2.2-TI2V-5B/` | VAE gate 候选（f8c16 / f16c48），已落选 FLUX.2，留作对照 |
| `aes-pixai-*` / `hpsv3-qwen` / `sac+logos...` / `sscd_*` | 美学/偏好/去重打分模型（数据线 / eval 用） |

**数据 & 环境**
| 路径 | 是什么 |
|---|---|
| `s3-pixai-test-uw2/evan/mot_filtering_v1/t5_manifest/daily_v1/{gpic,monet,monet_synth}` | **complete_v1 real/synth manifest**（38 列；选样+band 权威，图/caption 回源 src_shard）。池：gpic 7M / monet 11M / synth 3M |
| `/s3/evan/anime_core_ditv04_ct_soft/data/`（17 shard, 139 列, 16.5M） | **anime 源 parquet**（自带全部采样+caption+取图字段，直接驱动；图走 R2 `https://img.metanomaly.co/r2/<s3key>`）。**目标域** |
| `/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json` | **采样 config 权威**（domain_mix/band_mix/anime priority+caption 路由全规格） |
| `HF_HOME=/local/evan/cache/huggingface`、pip 缓存 `/local/evan/cache/pip` | 缓存放 /local（FSx 紧张，别往 /fsx 写大产物/缓存） |
| `/fsx/evan/checkpoints/mot_pilot_v1/` | 调试 checkpoint；长存归档去 `/s3/evan/checkpoints/` |
| `doc/env/`、`doc/mot/T2_H100_GPU踩坑与集群使用建议_20260614.md` | AWS/JCC 集群上手、空间规则、H100 踩坑 |

---

## 7. 关键决策（2026-06，Evan 拍板；细节见规划文档）

- VAE gate 选 **FLUX.2**；MoT `patch_size=1`（VAE 已 16×/128ch）。
- `timestep_multiplier=1.0`（Cosmos3 对齐，仍可 sweep {1,10,1000}）。
- noise-shift = per-分辨率固定常数（不 ramp）；grad-clip 1.0 / warmup 300 / EMA 0.9999（实现交 naifu）。
- 生成侧 sandwich norm 进 Day-1 baseline（默认开）。
- i1 消融（B1-B4）**搭骨架保留在主线**、默认关；adapter 消融保留中间对照臂。
- caption `text_max_length=296`。

**2026-06-17 GPU 段（G0–G2）结论/决策**：
- **bf16 路线 = 先 autocast**（fp32 master 参数 + `torch.autocast`，`trainer.amp_dtype` 门控）拿单卡基线；**纯 bf16（`model.to(bf16)`/FSDP2 mixed precision）留 G4 迁 naifu 时切**（用户拍板）。
- **`gradient_checkpointing` 保持开**：实测关掉时 bs8 就 74GB、bs32 OOM；开了 ~30% 时间换约一半显存。
- **attention 不迁**：bool-mask SDPA 走 cuDNN、近最优（1.16× flash 天花板），暂不上 flex/packed+FA varlen。GPU 阶段 sweep `timestep_multiplier {1,10,1000}` 仍待做。
- 单卡 512px micro-batch 容量：bs32(41GB)/bs64(58GB) 从容，为 G4 FSDP2 的 per-GPU batch 规划提供依据。

**2026-06-17 GPU 段（G6 最小采样器）结论/决策/坑点**：
- 范围 = **最小正确采样器**（用户拍板）：Euler + 双前向 CFG + VAE decode；**und 逐层 KV 缓存优化延后**（本版每步重算 und）。
- σ schedule **用训练同 shift（乘式 current_shift）而非 naifu empirical-mu**；CFG **用经典双前向而非 naifu guidance-embed**（我们模型 caption-dropout 训练、无 guidance embedding）。
- **坑点**：`proj_out` zero-init → **未训练模型 v≡0 → 采样原样返回噪声、CFG 无差异**。所以机制测试必须先 randomize `proj_out` 才能验 CFG/确定性（zero-init 这点本身用作"v=0 恒等"健全性检查）。
- 验证策略 = 机制（CPU）+ 过拟合重建（GPU，用户拍板）；重建用 `cfg_dropout=0` + guidance_scale=1.0（cond-only）去掉混淆，MSE 0.02× 基线确证端到端方向正确。

**2026-06-17 GPU 段（G4 FSDP2 + G5 EMA）决策**：
- **数据 hold 期提前做多卡 infra（用户拍板）**：anime 数据未就绪 → G3 暂缓；先把 G4/G5 验好，数据一到零延迟开训。多卡用 JCC（hyperpod 8×H100 节点多数空闲）。
- **G4 路线 = Route B 本地 train.py + FSDP2（用户拍板，偏离 handoff 首选 naifu）**：理由 = 数据 hold 时 naifu 的 dataio 优势用不上，且 naifu 全 env 有摩擦；本地 FSDP2 在已验证 venv 上快、可控。naifu 全迁移等数据/dataio 落地再做。代价：FSDP2 glue（~一两百行）后续迁 naifu 时可能丢。
- **bf16 在 FSDP2 下 = MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)**（naifu 配方，fp32 master shard），即 G1 时说的"纯 bf16 留 FSDP2 切"已落地；单进程仍 autocast。
- EMA 仅可训练参数、shard-wise；多卡 ckpt 用 DTensor full_tensor/distribute_tensor 自己 gather（EMA 非 FSDP module）。

**2026-06-17 GPU 段（G6 und KV 缓存）决策**：
- 数据 hold 期继续无数据项（用户："继续做无数据可以做的内容"）；做 **G6 und KV 缓存**（采样 ~2× 加速，唯一剩的无数据优化项），**严格对拍验证**（用户拍板）。
- 实现 = baseline 路径预计算每层 und K/V、gen-only attend；**adapter 臂回退** per-step（其 und→T_cond 拓扑不同）。precompute 保持 bf16（与全 pipeline 一致，不特意 fp32）。
- 验证三层：fp32 逐位（数学）→ bf16 单步（数值现实）→ 端到端重建（cached=非cached，证无害）。**结论**：bf16 单步 13% 是 28 层 und 塔 reduction 累积、端到端 wash out；实测加速 1.62×（非满 2×，因 gen 塔本身也占一半算力）。

**2026-06-17 末（外部审计 review）决策**：
- 采纳 F3（B1 臂 `no_grad`+grad-ckpt）/ F4（`save_final` 开关）/ F2（参数量更正写进规划+实现指南）/ F5（清理 README/vae/configs/test 滞后文案）；**驳回 F1**（"归因污染"= 对 4 臂消融阶梯的误读，已在实现指南 §4 加 📌 澄清）。
- **B1 设计待重估（gated on 数据）**：16% 成本偏高，B1 排期时再定是否收窄 adapter / 改走 Variant B（逐生成层 ΔK/V，省独立 adapter）——需 anime 数据评估收益，现在不拍。

**2026-06-17 数据线配比/步数决策（Evan 拍板）**：
- **512 重新纳入 GPIC**（512 此前样本偏轻 51.2M vs i1 256M）：domain_mix 512 = anime 0.65 / monet 0.23 / **gpic 0.08** / synth 0.04（anime/synth 不动，real 预算 0.31 切 monet:gpic=0.742:0.258）。GPIC@512 池 ~1.15M（elig_512=16.5%）。config `monet_real_gpic_ratio[512]=[0.742,0.258]`、`resolution_routing[512].gpic=elig_512==1`。
- **1024 不加 GPIC（物理不可行）**：GPIC **elig_1024=0%**（无任何 ≥1024px 图）。强配 domain_mix>0 会起空 producer→空等。1024 维持 anime0.80/monet0.08/synth0.12。
- **256 步数 300k→400k**（256 是 ablation 稳定起步单元，原 300k 偏轻）。512=200k、1024=500k 不变（1024 的 32M 样本已接近 i1 的 38.4M）。

## 8. 训练步数 / 样本量计划（必标 fixed_steps + fixed_samples）

> ⚠️ **约定（强制）**：任何步数/时长表都**同时**标 `fixed_steps` 和 `fixed_samples`。原因：global batch = #GPU × per-GPU batch；**8→32→64 卡时 global batch ×4/×8**。同 `fixed_steps` 跑 32 卡 = 见 **4× 样本**（训了更多数据、≠ 更快）；要保持同 `fixed_samples` 则 `steps ÷4`（才是"更快训完同样数据"）。只看步数会把"更快"和"训了更多数据"混淆。

**8 卡 first full run（per-GPU batch 来自各 config；global = 8×per-GPU）**：

| stage | per-GPU bs | global bs(8卡) | **fixed_steps** | **fixed_samples** | 参考(i1/nano-T2I) |
|---|---|---|---|---|---|
| 256  | 64 | 512 | **400k** | **204.8M** | i1 ablation 稳定起步 ~500k 步 |
| 512  | 32 | 256 | **200k** | **51.2M** | i1 512 ~256M（本阶段仍偏轻，靠加 GPIC 补多样性） |
| 1024 | 8  | 64  | **500k** | **32M**  | i1 1024 ~38.4M（接近，不偏轻） |

**换卡数时（保 fixed_steps 不变的话）fixed_samples 随 global batch 等比变**：
| stage | 8卡 samples | 32卡 samples(同步数) | 64卡 samples(同步数) |
|---|---|---|---|
| 256(400k) | 204.8M | 819.2M | 1638.4M |
| 512(200k) | 51.2M | 204.8M | 409.6M |
| 1024(500k) | 32M | 128M | 256M |

→ 例：512 在 8 卡 200k 步只 51.2M（偏轻）；上 32 卡若**仍跑 200k 步**则变 204.8M（≈i1）——但那是"看了 4× 数据"，不是"更快"。若想"更快训完同样 51.2M"，32 卡只需 **50k 步**。开训定规模时按此换算、别只搬步数。

## 2026-07-03 深夜 —— redesign-256 iter_s 线性变慢【根因坐实 + 真修复 + 重启】（第三轮，前两轮的 allocator env 修法已证伪撤销）

**根因（探针实锤）**：数据线 producer 每次 shard/caption 回源加载 = 10^5~10^6 级 Python 对象分配风暴 → **gen2 全堆 GC 每 rank 每 ~21 步触发，单次停顿随进程年龄单调增长**（1×8 探针：333→571ms/71min 无一回落；6×8 30h 后 ~4s）；48 rank 任一停顿拖全体 FSDP collective → p50 线性爬、重启归零。停顿哨兵与 gen2 事件一一对应；对象数全程持平 = 无 Python 级泄漏（时长涨来自堆碎片化/局部性退化）。主训练 metrics 双峰分布（1/3 步在 1.4s floor + 2/3 步被移动模态停顿击中）吻合。

**修复（全 gated，E2 1×8 2600 步验证 PASS 后上主训练）**：
- `train.py`：`trainer.gc_gen2_threshold=200`（gen2 降频 20×）+ `trainer.gc_freeze_after_steps=50`（进程启动后第 50 步 collect+freeze，扫描集与历史堆解耦；resume 兼容）。默认 0 = 旧行为逐位一致。
- `training/dataline/image_fetch.py`：OSError 带 errno；anime 磁盘缓存读/写 best-effort（旧代码磁盘满时把已成功取到的图判失败丢弃 = fail rate 0.2%→9% 的 ERR:OSError 风暴根源；实测单节点缓存 673 万文件平铺目录）。
- job yaml v2（`redesign_256_{6x8,4x8}_causalmlf_FORMAL_v2creepfix_job.yaml`，旧 v1 保留=回滚）：撤掉 2026-07-03 上午加的三条 allocator env（1×8 对照实测它比 nofix 慢 30%）；GC 探针（`data_analysis/cluster/gcprobe/sitecustomize.py`，env 门控）+ nodemon + cache janitor（/local≥85% 轮换）；**rdzv 文件按 job 唯一命名 `rdzv.$JOBTAG.addr`（根治 delete+resubmit 竞态）**。

**结果**：新 job `evan-mot-redesign-256-6x8-causalmlf-v2-pfrl` resume@75000（旧 g7lg 到 75800 删除，丢 800 步=划算），首 500 稳态步 **p50=1.53s（首跑基线 2.09、坏 env 后 3.32；GPU floor 1.35）、dw=0.080** —— 比首跑还快 27%（原基线从 step1 就在缴 gen2 税）。剩余 91k 步 ETA ~40h。

**回滚**：`bash /fsx/evan/mot_code/mot_speed_fix_backup_20260703/RESTORE.sh`（文件级快照，勿 git checkout——工作区混多线未提交改动）。完整报告：`data_analysis/outputs/redesign_256_causalmlf_global2304_20260630/creepfix_v2_report_20260703.md`；实验产物 `/fsx/evan/tmp/mot_speed_experiments/mot_speed_{gcprobe,gcfix_e2}_20260703/`。

**⚠️未 commit**：train.py / image_fetch.py / config_schema.py 的修复在工作区（与 256零tag/eval 线的未提交改动共存）；commit 需 Evan 拍板且范围切净（只 add 这三个文件 + gcprobe + v2 yaml + 本 doc）。

## 2026-07-05 —— 512a 启动准备 + 【256 提前停训 @304M】+ 缺头问题定性与 512a 改道 multi-AR（Evan 拍板）

**512a 就绪审计（4-agent，全只读）**：训练逻辑/数据线/config 无 P0（init_from 链路+超参 snapshot 修复完好、consolidated ckpt 任意卡数可载、elig_512/budget471/mix/caption 路由全就绪）。缺口=①启动模板 `jobs/stage512a_redesign_launch.yaml` 缺 creepfix v2 全部 5 项硬化（GC 键/rdzv 唯一命名/janitor/unset env/守卫）+NNODES=2 默认是 50 万步陷阱；②`checkpoint_keep_every=25000` 非 3000 整数倍→周期快照几乎不触发（256 目录只有 last.pt 的真因）；③**VAE precompute 再次否决**（历史实测 I/O 慢 5×+stub 实验 VAE 只占步时 11-13%+multi_ar 硬 raise+缓存缺 textatlas/512 根索引）；④creepfix v2 后 p50 平但 **p90/p99/mean 尾部 creep 未根治**（p90 56k 步 2.4→5.9s）。

**512a init_from smoke = 17/17 PASS**（`evan-mot-512a-initfrom-smoke-vlfb`，1×2，init_from 真 26.3G ckpt）：指纹/step 重置/re-warmup 斜坡/48→2 卡 reshard/loss 下降/阶段 B resume 优先（retry 语义）/新键 `trainer.gc_refreeze_every`（周期 re-freeze，治尾部 creep，快照副本实现待移植工作树）全部正确。peak 25.8G@1×2 BS12。

**缺头问题（rd256@297M 全身 anime 头被截）——6-agent workflow 定性**：根因三因素缺一不可：①`image_fetch.py:210` 中心裁方图 × anime 62-66% portrait（37% 图顶损≥15%；monet/gpic 横向为主、synth/textatlas 方图=基本免疫）；②曝光收敛放大（193.5M 0/9 vs 297M 2/9，idx136 双 seed 复现；eval 配置逐字节同、loss 平滑=非伪影）；③NL-first 配方 vs 100% tag eval 的 OOD 共因（baseline 同裁剪同数据 0/9 → 裁剪必要不充分；对真实 NL 用户影响比 eval 显示轻）。主流对照：naifu=自适应 AR bucket 仅 64px 微裁；SDXL 方图 pretrain 有同款 artifact+crop-conditioning 解药。**256 权重可用、不重训**（framing 先验非表征损坏）。

**Evan 拍板（07-05）+ 已执行**：
- ✅ **删 5 个旧消融 ckpt 释放 119G**（s256_1x8_v1/ab3lora/b1adapter/**ab1causal/ab2fusion**——⚠️eval 线注意：后两个是原 512 carry-forward 候选，Evan 明确放弃该选项）。
- ✅ **256 正式训练提前停止**：`...v2-pfrl` 删于 step 133210；**final ckpt = last.pt@132000 = 304.1M 曝光**（26.3G），硬链接 `checkpoints/mot_pilot_v1/redesign_512a_prep/init_from_512a_step132000.pt` 为 512a init_from 源；rdzv 已清；6 节点已释放。理由：尾段 80M 曝光 foundation 收益递减 vs 缺头偏差确定性加深（对抗复核认定「照常跑满」被支配）。⚠️ **eval 线 heads-up**：本 RUN_ID 不再有新 ckpt；`redesign_frozen/eval_rd256.pt`(84000) 与 `eval_rd256L.pt`(129000，现为独立 26.3G) 的去留归 eval 线/Evan。
- **512a 改道 = multi-AR + 保头 anchor 组合**（推翻同日上午「方图先跑」）：`multi_ar.enabled=true` + per-domain `crop_v_anchor`（anime≈0.15 保头；兜底 re-resize 同样走 anchor——off-native 竖图兜到方桶顶裁 ~22% 是残余漏洞必须堵）+ warmup 曝光化。诊断三项：512a ~20-30M 曝光视觉 gate（idx136 双 seed 等，重训与否的判据）、NL 措辞重测（frozen 129000 ckpt）、require_nl 子集 AR 复测。
- 512b config 本轮不动（等 512a 验证后拍）。

**进行中**：v_anchor/multi_ar/warmup/gc_refreeze 代码落工作树 + multiar_anchor smoke yaml + re-resize dryrun yaml + NL13 evalset/probe（agent 实现中，待 review→jcc 验证）。之后：正式 512a job yaml（v2 硬化版）→2×8 多机验→Evan 拍 NNODES/global batch→launch。

**⚠️未 commit 积压（等 Evan 拍板，范围切净）**：creepfix v2 三文件 + gc_refreeze + 零tag两文件（256 线）+ 本轮 v_anchor/multi_ar/512a yaml + gcprobe + 各 job yaml + 本 doc。

### 2026-07-05 深夜追加 —— multi-AR 采样升级（affinity）+ 512a 验收 gate 文档 + launch 前验证
- **naifu 对照结论**（Evan 拍板 Tier1+Tier2）：不整体换 bucket-first（破坏 mix 保证/流式约束）；升级为 **native-affinity 域采样** `P(d|B_t)∝mix_d·P_native(B_t|d)`（`multi_ar.native_bucket_dist`，bucketdist 实测表；P(B) 派生保恒等式；square_only 域 one-hot 吸收与 alpha 机制互斥）+ **Tier1 可观测** `buf_reresize_<domain>` 进 metrics.jsonl。单测 41 项全过。
- **dryrun 抓到并修复一个退化路径漏洞**：MODE-A 支撑域全缺席时退全池 → square-only 域被裁进非方桶（首轮 500 batch 漏 27 synth+16 text）。修=退化时无条件剔 S 域（硬不变量），修复版 dryrun **6/6**：synth/text re-resize=0、mix 偏差 ±0.02、anime re-resize 0.89→0.64（CPU 饿 buffer 口径）。
- **2×8 测试（warmup 相）PASS**：rdzv v2 多机拉起 ✓、**BS32@512px peak 19.5G**（→ 正式 6×8 global 1536 锁定）、iter p50 2.53s（全程 83,334 步外推 ~2.5 天）、`buf_reresize_*` 指标管道 ✓、loss 1.07→0.88。开桶相（warmup 覆写 20 步、280 步多机桶轮转）测试进行中。
- **★新文档：`docs/512a_验收gate与256重训判据_v1_20260705.md`**（Evan 要求）——G0/G1(~25M)/G2(~64M)/G3 视觉 gate 硬判据（B 率口径固定 13 prompt×tag/NL 双口径×双 seed）、「必须重训 256」四条同时满足才触发、G3 eval 对照口径、512b/1024 语义同步待办、监控阈值、crop-conditioning 备忘。**512b 启动前必须同步缺头修复全套配置（anime 0.78 风险最高）。**
- 待办：开桶相 2×8 → commit 四组（分组补丁已构建并验证逐字节等于工作区：A 速度线/B 缺头修复/C 零tag 代提交/D docs，等 Evan 过目）→ launch 6×8 正式（`redesign_512a_FORMAL_v3_job.yaml`）→ 武装 G1 gate monitor（step≥16000 冻结+提交生成 job）。

### 2026-07-05 launch 前终态（compact 存档点）
- **2×8 开桶相 PASS**：4 桶形×280 步跨节点桶轮转零 desync、peak 19.6G；**真机 re-resize 率：anime 0.5%（affinity 设计实战生效，结构预估 30-40%→0.5%）**/monet 3.2%/gpic 5.9%/synthetic 0/textatlas 9.7%，全部低于告警阈值；mix 300 步内归位。
- **Evan 追加拍板**：①512a 默认曝光 **128M→256M**（config `max_exposure_images=256000000`、@global1536=166,667 步≈5 天、keep_every→60000=@60k/120k 两快照——config/gate doc/FORMAL_v3 yaml 三处已同步改）②正式 job 提交后 **Pending 最多等 10h**（超时再议 4×8）③利用率要求 ≈stage256 水位（dw<10%、iter 平稳，G0 核）。
- 待办=commit 四组（补丁已构建验证）→ 提交 `redesign_512a_FORMAL_v3_job.yaml` → 武装 G1/G2 gate monitor（判据见 `docs/512a_验收gate与256重训判据_v1_20260705.md`）。
### 2026-07-06 —— 512a 正式两次死亡 + 零权重竞态修复（Evan 拍板双管修法）
- **攻1 `...-6x8-v3-swqg`**：排队 10h 后 06:51 起跑，健康 1020 步（loss 0.87↓/iter 2.55s/dw 5%）后 07:41 死于**一节点 7×SIGBUS+1×SIGKILL**（node5；GPU UUID 指纹已存证 scratchpad/swqg_all.log）——节点级故障形态，死前无 shm/OOM/dataloader 告警。无 ckpt（<3000 步）。
- **攻2 `-retry-xdqd`**（jcc maxRetries 自动）：从 0 重跑至 **step 8300=开桶步**，16:16 死于 rank3 `ValueError: _weighted_idx 退化累积权重(len=512,total=0.0)`。**根因（我+GPT5.5 双独立取证一致）**：MONET 未列 explicit 源(da4k 等)行 base_w=0，producer 灌 buffer 不过滤 → 零权重 shard 连续段可灌满 512 槽 → **MODE-A affinity 单域 `sel={dpick:pool}` 无他域吸收** → `_fallback` 全池加权抽全零池 → raise。warmup 期全域池有吸收所以不炸=恰在开桶后爆。**概率竞态**（2×8/16rank 没撞上、48rank 撞上），157k 剩余步内随时复发。留下首个 ckpt last.pt@step6000（14:08）。
- **攻3 `-retry-2dcs`** 16:16 自动拉起，**resume@6000 语义正确**，健康巡航（iter 2.54s/dw 4.5%）。⚠️ GPT5.5 GPU-UUID 比对：**其 node_rank3 = 攻1 SIGBUS 同一物理节点**（坏节点仍在池，复发须报 Evan 找 ops 隔离；jcc CLI 无节点排除项）。
- **修复（Evan 拍板：双管+1×2 smoke 验证+不动 2dcs 择机换血+坑#11 例外批准）**，4 层全落地（`training/dataline/{complete_dataset,sampler}.py`，未 commit）：①iter_real_synth_shard **取图前过滤 base_w≤0 行**（根修，零权重行任何路径不可抽=纯占坑，不拉图不入池）②producer canary 计数（应恒 0，metrics `buf_zerowdrop_*`）③`_build_pools` **全零权重池→缺席语义**（=既有「无 native 供给→缺席」同构；`buf_zerowpool_*` 计数）④`sampler._fallback` 跳零权池+textatlas 纳入兜底（修原漏）+全不可抽 RuntimeError 响亮失败。
- **验证**：新单测 `data_analysis/scripts/redesign_512a_zerow_unittest.py` 全 PASS；**1×2 GPU smoke（补丁后全套断言含 warmup→开桶翻转/桶形/loss/GC）全 PASS**；GPT5.5 对抗 review 7 攻击面**无 blocker**（附注：`max_rows_per_shard` 诊断口径微变，正式 config 未用）。dryrun 复跑 5/6——挂的 textatlas reresize 93% 判定为**该 CPU 节点饿供假象**（buffer fill 0-1%，脚本自身 REPORT 亦标饿供噪声；补丁未触 textatlas 路径），A/B 归因 job（同节点 pre/post-patch 对照，`zerowfix_ab`）在跑以严格豁免。
- **运维改进**：FORMAL_v3 yaml 加 **per-attempt 日志后缀 $STAMP**（攻2 tee truncate 攻1 日志致 SIGBUS 取证丢失的教训）；wake watcher 加「家族新 retry 行=事件」检测 + `head -1` 取最新 attempt（retry 名以原名为前缀会 grep 双匹配）。
- **换血计划**：验证全绿后在 ckpt 刚落盘窗口 `jcc job stop` + 重提 FORMAL_v3（新 job 从 last.pt resume，损失 <10min）→ 下一攻自动 import 修复码。若 2dcs 先自然 crash 则 retry 自动带新码=零操作换血。
- **✅换血完成（19:05 UTC 终态）**：A/B 归因反转——pre-patch 同节点复现 dryrun 失败签名（textatlas reresize 100%）、post-patch 6/6 全过 ⇒ 死行一直在污染 native 子池推高全域 re-resize，补丁兼修此项。ckpt9000 落盘后 stop 2dcs（实停 step9050，jcc stop ~1min 滞后）→ 首提 qc6r 死于编排瞬态（节点 15min 未齐 rendezvous=旧 pod 释放 GPU 拖慢，非代码）→ jcc 自动 retry **`...-v3-qc6r-retry-tg2p` = 现役正式 job**：新码 resume@9000 实锤（`[train] resumed ... at step 9000` + metrics 新键 `buf_zerowpool_*/zerowdrop_*` 全 0=护栏未触发+canary 干净；reresize anime 2.4%/monet 0/synth+text 0；loss 0.862 续降/dw 5%）。总代价 ~35min 墙钟+20 步回滚。⚠️ metrics.jsonl 现为 4 段结构（step 回退点=攻界，同 step 多行），后续分析必须切段（切法见 data_analysis 侧脚本或按行序取最后一段）。**补丁未 commit，待 Evan 拍板**。G1 gate@18060 照旧（monitor 在岗）。
- **（compact 后追加）launch 完成态 + textatlas 口径修正**：四组 commit 落 HEAD（`b8a7749`/`b37cab0`/`82d056f`/`5ddcdb2`）后正式 job `evan-mot-redesign-512a-multiar-6x8-v3-swqg`（6×8, 256M 曝光=166,667 步）已于 20:48 UTC 提交，Pending 排队（Evan 授权最多 10h）。外部 AI 复审指出 gate doc 与本 doc 的 textatlas re-resize 数字矛盾——核验结论（Evan 拍板修 doc）：**textatlas 稳态 ≈9% 是守恒律决定的设计预期**（native 非方 9.03% 只能经兜底裁进方桶消费=旧管线同款处理；2×8 实测 9.7% 吻合，step50-250 的 0 是 buffer 死重累积期假象），gate doc G0 行与 §5-5 原「synth/text reresize=0 否则代码回归」判据已修正为：synthetic=0 硬不变量不变、textatlas ≈9% 正常/>20% 或持续爬升才查。复审其余三点（dirty worktree/monitor 缺失/keep_every 分层）均为过期信息或已覆盖，不采取行动。

### 2026-07-07 —— head/foot 截断三权重评估线准备（只新增 eval 文件，未碰训练 import 路径）
- 目标口径（Evan 本轮拍板）：对比 **stage256 final**、**stage512a G1≈25M**、**stage512a latest/G2 后较新权重**；latest 不抢当前 `last.pt`，等 G2/更近快照，冻结副本后评估；评测规模可以做宽。现状复核：`gate_g1.pt` 已冻结且已有 tag13/NL13 双 seed 52 张；G2 尚未到（monitor 阈值 step≥42060）；512a metrics 约 step31960，`last.pt` mtime 11:44，直接追读不作为权威 latest。
- 新增独立 sampler `experiments/eval/sample_headcrop_matrix.py`：不 import/monkey-patch 既有 `sample_evalset.py`，支持三 run spec（`rd256_final` / `rd512a_g1` / `rd512a_latest`）、EMA、显式 `--height/--width` 非方图、完整 manifest（含 ckpt path/step、missing/unexpected key 片段、每图路径）。新增 prompt/rubric/评分聚合：`experiments/eval/prompts/anime_evalset_nl13_plain_headcrop.jsonl`、`experiments/eval/prompts/real_headcrop_12.jsonl`、`experiments/eval/headcrop_matrix_rubric.md`、`experiments/eval/headcrop_subagent_scoring_prompt.md`、`experiments/eval/aggregate_headcrop_scores.py`。
- 新增 JCC 模板：`jobs/eval_headcrop_matrix_g2.yaml`（G2/latest 存在后生成三权重矩阵，预计 696 张：anime tag/NL plain/NL emphasis + real，square/portrait/landscape/tall/wide，seeds 7/1007）和 `jobs/eval_headcrop_matrix_g2_submitter.yaml`（0-GPU watcher，等 `gate_g2.pt` 后提交矩阵 job，state 防重复）。已提交 watcher `evan-mot-eval-headcrop-matrix-g2-submitter-d8x5`，当前 `Pending`；语法检查 job `evan-mot-eval-headcrop-matrix-pycompile-v2cw` 也 `Pending`，矩阵 job 自身开头含 `py_compile` 兜底。
- 后续：G2/latest 样图生成后，按 `headcrop_matrix_rubric.md` 用 gpt-5.5 子代理逐图原图判读，汇总 `trunc_any/head_cut/foot_cut/complete_rate` 分 checkpoint/domain/prompt_style/AR/seed 的定量表，再写最终报告判断 512a 是否已基本修复、是否仍需重训 256。
