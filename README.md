# mot_pilot_v1 — 2-3B MoT anime T2I pilot（stage1a 网络+训练代码）

**2–3B MoT 二次元文生图（T2I）pilot**:frozen Qwen3-VL-2B 当理解路（~1.72B）+ 逐层复制的 ~1.42B 可训生成塔（Cosmos3 式 MoT）,latent 空间 flow-matching（v-pred + rectified flow + 纯 MSE）,FLUX.2 VAE。目标先在 T2I 超过 Anima-2B。

本仓库 = 网络架构 + stage1a（仅 t2i）训练代码 + 推理采样（`generate_samples`,已 GPU 验证）
+ **complete_v1 真实数据线（[training/dataline/](training/dataline/),5 域全忠实,已端到端跑通 → V1 已开训）**
+ **stage256 四版消融正式训练 + 4-way 评测（v1/B1/B2/B3）**
+ **Ab2 末 N 层多层融合 / Ab3 冻结理解塔 und_lora 两个消融机制 + 2×8 多机产线配方**。

方案依据:[docs/自己写的模型+训练设置文档_v3_20260617.md](docs/自己写的模型+训练设置文档_v3_20260617.md)（模型/训练设计正本）+
[docs/AAA数据交付文档_GPIC+MONET+anime_v2_20260617.md](docs/AAA数据交付文档_GPIC+MONET+anime_v2_20260617.md)（数据线权威）。

> **接手先读**:[CLAUDE.md](CLAUDE.md)（≤1 屏 onboarding + 硬约束）→ [当前进展_交接GPU.md](当前进展_交接GPU.md)（实时进展/坑点/决策/步数）→ [docs/README.md](docs/README.md)（设计权威/文档总索引）。
>
> **本 README 的范围 = 项目结构 + 架构对照**（不常变的部分）。**现状 / 已验证 / 决策记录 / 遗留 TODO / 路径导览 / 训练步数表 全部以 [当前进展_交接GPU.md](当前进展_交接GPU.md) 为准**（§2/§3 已验证、§4 坑点、§5 偏好、§6 路径导览、§7 决策、§8 步数表）—— 本 README 不再复述,以免两处漂移。

> **⭐ worktree 范围**:本仓库过渡期并存 3 个 git worktree——本 worktree `mot_pilot_v1_cleanup`（分支 `cleanup-merge`）= **当前最新整合代码、开发都在此**;`mot_pilot_v1`(main) / `mot_pilot_v1_ablations`(ablations-v2) = **两个直播 worktree**（v1/B1 单机 + 三消融 2×8 多机 job 实时读它们的代码）,**代码文件绝不要碰**（retry/续训会重读磁盘 → 坑点#11）。训练全部跑完 + Evan 确认后,`cleanup-merge` FF 进 main、两个临时 worktree 删除。详见 [CLAUDE.md](CLAUDE.md) / [当前进展_交接GPU.md](当前进展_交接GPU.md) §0.0。

## 结构

```
networks/arch_mot.py    MoT 双路 transformer（核心架构，可整体搬进 naifu）
networks/model_mot.py   naifu model_cls 契约包装（加载/复制塔/冻结/文本/loss）
networks/vae.py         VAE 抽象：FakeVAE(调试用) + FLUX.2/Qwen-Image 占位
training/stub_data.py   数据占位 stub（latents/pixels/overfit；调试与 G1-G6 用）
training/dataline/      complete_v1 真实数据线（5 域：anime/monet_real/gpic/synthetic/textatlas）：
                        config / manifest_index / sampler / caption_build / image_fetch /
                        complete_dataset(CompleteV1Dataset) / latent_cache(脚手架)
train.py                精简训练入口（naifu train_fsdp2.py 主循环的单进程简化版）
configs/debug_tiny_cpu.yaml   CPU 调试（tiny 随机权重 + 假 VAE + stub 数据）
configs/stage1a_t2i_512.yaml  512 正式配置（训练/FSDP2/EMA/采样路径已验；数据读取经 2×8 probe 验、未单独跑满训练）
configs/stage2_1024.yaml      1024 配置（训练代码路径同 512 已验；本分辨率未单独跑）
scripts/_archive/diagnostics/debug_mask_positions.py  mask/MRoPE 可视化 + 断言
scripts/smoke_real_weights.py    真权重加载/复制/冻结/HF 数值对照/前反向
scripts/_archive/diagnostics/probe_sdpa_backend.py    G2：bool-mask SDPA 后端探测 + 吞吐（确认走 cuDNN 不退化 MATH）
scripts/test_sampler.py          G6：采样器机制单测（CPU tiny；zero-init 恒等/shape/确定性/CFG/σ调度）
scripts/_archive/one_time_scripts/overfit_reconstruct.py   G6：GPU 端到端过拟合重建（采样符号/σ 方向最强信号）
scripts/test_ema.py              G5：EMA 单测（CPU；数学/ckpt round-trip/滞后）
scripts/bench_sampler_cache.py   G6：und KV 缓存正确性(fp32≈0/bf16 单步)+加速 bench（真 2B GPU）
scripts/test_textatlas_dataline.py   TextAtlas 第5域接入单测（6 项，sampler 注入实测=目标）
scripts/probe_realized_mix.py    离线探针：量化实际 domain_mix（不碰在跑 job；配 aggregate_probe_mix.py 跨 rank 聚合）
configs/stage1a_t2i_256.yaml      stage256 v1 配方（global 512、400k 步；4 版 1×8 消融在跑的那份）
configs/stage1a_t2i_256_v2.yaml   stage256 多机 v2 配方（global 1536、133k 步 = 对齐 v1 样本预算；含 Ab2/Ab3 开关）
jobs/                   jcc job yaml：单机消融 s256_1x8_{v1_baseline,b1_adapter,b2_longskip,b3_nonoise}.yaml；
                        多机产线 prod_2x8_{ab1causal,ab2fusion,ab3lora}_wt.yaml + s256_{2,4,6,8}x8_train.yaml；
                        历史 G4 验证 g4_fsdp2_validate.yaml
env/requirements.txt    在 venv-gpu 上增量安装的依赖
```

环境:`/fsx/evan/mot_code/venv-gpu`（torch 2.11+cu128,已加装 transformers 5.11.0 / diffusers 0.38 / omegaconf / einops,CPU 可跑）。多机正式 job 用 ECR 镜像 C（`autobuild:py312-torch211-cu129`,自带 EFA / aws-ofi-nccl）+ 同一 venv-gpu,零代码改动跨节点近线性。
**cleanup 解耦重构（已完成,行为不变,commit affc89c）**:把 [train.py](train.py) 内联的 checkpoint save/load + 消融指纹、诊断脚手架、配置校验,以及 [networks/arch_mot.py](networks/arch_mot.py) 的 mask/位置编码 builder 抽成独立模块（`training/{checkpoint,diagnostics,config_schema}.py` + `networks/arch_mot_utils.py`）。

## 跑法

```bash
PY=/fsx/evan/mot_code/venv-gpu/bin/python
$PY scripts/_archive/diagnostics/debug_mask_positions.py            # mask/位置编码检查
$PY train.py --config configs/debug_tiny_cpu.yaml   # tiny CPU 训练（~3min/100step）
$PY scripts/smoke_real_weights.py              # 真权重 2B smoke（CPU ~5min, ~25GB RAM）
# dotlist 覆盖：
$PY train.py --config configs/debug_tiny_cpu.yaml --max-steps 10 dataset.mode=pixels
```

⚠️ 上面是**命令示例**;实际运行（含纯 CPU 单测）一律走 **JCC job**,别在本机直跑（坑点#28 OOM）。**完整 CPU + GPU 自检清单** → [当前进展_交接GPU.md](当前进展_交接GPU.md) §1。

## 架构要点（与 MoT方案0608 对照）

| 方案条目 | 实现 | 位置 |
|---|---|---|
| frozen Qwen3-VL-2B 理解路（LLM≈1.72B） | embed+28层+norm 全冻结，权重从 HF checkpoint 映射 | model_mot._load_und_weights |
| 生成塔逐层复制、~1.5B 可训 | init_generation_tower_from_und；实测 1.416B（不含 vocab embedding，与方案口径一致） | arch_mot |
| MoT 权重划分 = Cosmos3 | 分流 LN/MLP/QKV+O(+QK-norm)，共享注意力+RoPE+同一序列；命名同 Cosmos3 `*_moe_gen` | arch_mot.MoTAttention/MoTDecoderLayer |
| 文本 causal / 噪声图 bidirectional；噪声看[文本+全噪声]，文本不看噪声 | bool mask，pad 不作为 key | arch_mot.build_t2i_attention_mask |
| Interleaved-3D-MRoPE，文本1D、生成图 t=T_gen=max(context_t)+Δ(2048)，h/w 段内从0起 | 复用 HF Qwen3VLTextRotaryEmbedding（interleaved 内置） | arch_mot.build_t2i_position_ids |
| **生成侧 sandwich norm（v3 Day-1 baseline）** | 仅 gen 支在 attn/MLP 输出处各加 RMSNorm 再残差（`x_gen += RMSNorm(sublayer)`），und 不碰；命名带 `_moe_gen`、weight=1；`model.sandwich_norm` 开关默认开 | arch_mot.MoTDecoderLayer |
| timestep：sinusoidal256→MLP→加到噪声 token（非 AdaLN-Zero） | TimestepEmbed；multiplier **默认 1.0**（Cosmos3 对齐，=timestep_scale 0.001）；仍是 sweep 项 {1,10,1000} | arch_mot.TimestepEmbed |
| v-pred + rectified flow + 纯 MSE | x_t=(1-σ)x0+σε，target=ε−x0 | model_mot.forward |
| σ：logit-normal + **per-分辨率固定 shift** | 不再按进度 ramp；`sigma_sampling.shift` 常数：256→1.0、512→2.0、1024→4.0(sweep 3.3-5) | model_mot.current_shift |
| 10% caption dropout；frozen LM 下 wd=0 | cfg_dropout_prob / get_optimizer_param_groups | model_mot |
| VAE = FLUX.2 f16c128（VAE gate 已拍板） | `Flux2VAE`(AutoencoderKLFlux2)：encode=sample→2×2 patchify→BN 标准化→[B,128,H/16,W/16]，decode 逆之；CPU round-trip 验证 max\|Δ\|=4.8e-7。**MoT 侧 patch_size=1** | vae.py |
| 文本格式 | Qwen-Image 式 chat template（已拍板），39 token 模板+caption，pad 到 256 | model_mot.encode_text |
| patchify | latent 2×2 pack（Qwen-Image/FLUX 同款），patch_dim=C×4 | model_mot._pack_latents |
| text_max_length=296（caption ~256 + 模板 ~39） | v3 长 caption 不腰斩 | model_mot |
| 文字样本 dropout 特例 hook | `_apply_cfg_dropout(prompts, text_rich)`：text_rich 样本用 `text_rich_cfg_dropout_prob`(默认0)；数据线带标志后生效 | model_mot |
| **i1 消融（默认关、关时零参数开销）** | B2 long_skip(gate-zero)/B3 noise_condition/B4 abs_sincos(α-zero) step-0 等价；**B1 text_adapter attention 臂已接入**（und 独立前向→adapter→gen cross-attend T_cond、四消融臂） | arch_mot |
| **Ab2 末 N 层多层融合（ELMo 式，默认关）** | `MultiLayerFusion`：融合 und 末 N 层（`fused = h_last + gate·(elmo − h_last)`，gate zero-init → step0 与当前 B1 逐位等价）后喂 adapter；config `text_adapter.multi_layer_fusion` | arch_mot.MultiLayerFusion |
| **Ab3 冻结理解塔 und_lora（baseline joint path，默认关）** | `LoRAResidual` 给 frozen und 的 q/k/v/o 投影挂低秩可训残差（`delta=scale·B(Ax)`）；末 N 层（默认 8 层）q/k/v/o 低秩残差（rank=32 alpha=32），`lora_und` 参数单独 param-group（lr_scale 0.3）；config `model.und_lora` | arch_mot.LoRAResidual / MoTAttention.add_und_lora |

### 实现与 Cosmos3 的差异（工程，不改数学）
- batched padded `[B,S]` + SDPA bool mask，**非** packed sequence —— CPU 可验证、
  bucket 下 batch 内图 token 数一致；GPU 实测 bool mask 走 **cuDNN**（近最优、1.16× flash
  天花板，非 MATH 退化），暂不迁 packed/FA varlen（见 [当前进展_交接GPU.md](当前进展_交接GPU.md) §3 G2）。
- und 路 frozen 且看不到 gen → autograd 不为其建图，naive 联合前向无反传额外开销。

---

> **完整进展时间线、28 条坑点、决策记录、遗留 TODO、路径导览、训练步数/样本量表 → [当前进展_交接GPU.md](当前进展_交接GPU.md)**;数据线设计/变更全貌 → [docs/complete_v1_dataline_设计与变更_v1_20260627.md](docs/complete_v1_dataline_设计与变更_v1_20260627.md);eval 产物 → [experiments/results/EVAL_ablation_comparison.md](experiments/results/EVAL_ablation_comparison.md)。
