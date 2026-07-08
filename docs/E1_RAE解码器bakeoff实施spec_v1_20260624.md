# E1 · RAE 解码器 bake-off 实施 spec (v1, 2026-06-24)

> 本轮 **plan-only**（不动代码）产物，给 Evan 评审。E1 = RAEv2/Scale-RAE「后续试验」E 阶梯的第一块（**纯重建闸门**）。
> 已读底料：RAEv2 repo 全码（`rae_refs/RAEv2`）+ Scale-RAE repo（`rae_refs/Scale-RAE`）+ 3 篇论文
> （RAEv2 2605.18324 / **Scale-RAE 2601.16208**=最贴我们的 T2I-with-RAE 正本 / 原始 RAE 2510.11690）。
> 关联：[docs/自己写的后续试验_v1_20260616.md](自己写的后续试验_v1_20260616.md)（E0–E9 正本）、[docs/后续增量与eval计划_v1_20260620.md](后续增量与eval计划_v1_20260620.md)。
>
> **v1.1 修订（2026-06-24，据外部 review + 我读码核实）**：① §2 Qwen wrapper 大改——输入是**展平 patch+grid_thw+temporal=2**（非 `[B,C,H,W]`）、pre-merger **无 final norm**、token 是 **2×2-块序需 un-permute**、MLS 取 block 输出非 `deepstack_features`（均已读 `modeling_qwen3_vl.py` 核实）；② §4 归一化改为**显式 token-norm 消融**（删掉错误的"仿 DINO 删末 norm"）；③ §2 加 K1/K23 诊断边界；④ §1 加 256 文字假阴性警示 + 早 512 sanity；⑤ §7 加 probe 公平性；⑥ §6 对照 encoder 优先级；⑦ §12 加 run matrix。集成路线 = 独立 `mot_rae_e1/`（不改 `rae_refs/RAEv2`）。

## 0. 已固化决策（Evan 2026-06-24 拍板）

| 维度 | 决策 |
|---|---|
| 集成路线 | **独立新目录 `mot_code/mot_rae_e1/`**（RAEv2 clone 的工作副本/fork、自带 git、与 mot_pilot_v1 / cleanup-merge **完全独立、不受其 worktree 纪律约束**——E1 是独立 repo 的研究线，不碰 mot_pilot 在跑代码）；加 Qwen3-VL encoder + 喂我们数据、直接用它 stage1/eval 全套；`rae_refs/RAEv2` 留**只读参照**。E2 接 UMM 时再议与 mot_pilot_v1 的关系 |
| 分辨率 | **256px 起步**（复用它全套 config/encoder-stats/eval 参照，便宜快迭代），出信号后再上 512 |
| 数据 | **anime 为主 + 真实子集 + 文字集**（不用 200M；证据见 §5） |
| 参考下载 | Scale-RAE(2601.16208)+原始RAE(2510.11690) 已在 `rae_refs/paper/`；github/HF 服务器可直连 |

**开放点已拍板（2026-06-24 续）**：Gram = 先无基线·A/B 再加 · 取层 = **K7+Deep** · probe = **E1 就做** · 对照 encoder = 先 **dinov3+siglip2**（算力够再加 webssl/mae）· 文字数据 = bucket 漫画(mangadex/manga-detext/toongod)+TextAtlas（见 **§5b**）。余见 §11。

---

## 1. 目标 + 闸门（含现实预期校准）

**目标**：回答"冻结 **Qwen3-VL-2B ViT** 的 pre-merger patch 特征能不能解码回清晰图（含细节/文字），且在该 latent 上小 DiT 的**可扩散性**不弱于 VAE"。

**最关键科学问题**：Qwen3-VL 是为 **OCR/理解**训练的 encoder，这种训练是否**伤了低层重建**？
→ 靠与 **DINOv3 / SigLIP2 / WebSSL RAE**（repo 内置）+ **FLUX.2 VAE** 对照来回答。
（Scale-RAE 实测：理解型 SigLIP2 重建 *弱于* SSL 型 WebSSL/DINO——text rFID 1.621 vs 1.372；linear-probe 也是 DINO>SigLIP2。所以"理解训练略伤重建"是真实风险，Qwen3-VL 偏向更甚——这正是 bake-off 要量的。）

**闸门（两段，已按论文实测校准阈值——这是对原 doc"进 VAE 2× 内"的修正）**：

1. **重建 sanity**：anime/general rFID 进 **FLUX.2 VAE ~1.5–2×**；**文字 rFID 大概率 2–3×**（已知难——Scale-RAE 最好的 RAE 文字也 2.15–2.5× FLUX；FLUX.2 更强 → 差距更大）。**文字不作硬否决**，看绝对可读性。**⚠ 256 对小字/气泡/网点会因 resize/crop 失真 → 256 文字差不能直接 no-go（review pt#6）；早放一个 manga-detext/TextAtlas 子集的小 512 重建 sanity 复核**。
2. **可扩散性 probe（决定性信号，公平性见 §7）**：同一个小 DiT 在 **Qwen-RAE latent vs FLUX.2-VAE latent** 上训，**matched steps/images/GPU-hours 下 gFID/收敛 ≥ VAE**。这是 RAE 的核心赌注：Scale-RAE Fig1/Table4 = RAE **4–4.6× 更快收敛**、GenEval **49.5 vs VAE 39.6**、且 VAE finetune 64 epoch 后灾难过拟合而 RAE 稳到 256–512 epoch。

**判据**：① anime 重建可读（256 文字差先经 512 sanity 复核）+ ② probe ≥ VAE-probe → **过，进 E2**；重建确弱 / probe 明显弱 → 退 **E5 混合 latent** 或回 **VAE**。

---

## 2. 唯一新核心代码：Qwen3-VL encoder wrapper ⭐

读码确认：**`stage1/rae.py` + `decoders/decoder.py` 对 encoder 完全泛型**——只吃 `encoder.hidden_size`、`encoder.patch_size`、`encoder(x)→[B,N,C]`；`decoder_embed = Linear(hidden_size, decoder_hidden_size=1152)` 自动把 **1024→ViT-XL**。所以 **E1 的全部新核心代码 = 在 `src/encoders/vision_encoder.py` 加 Qwen3-VL encoder 类 + 注册进 `ENCODER_REGISTRY`**（仿现成 `DINOv3Encoder` / `DINOv3MultiLayerSimpleAddEncoder`）。

要实现两个类（仿 `DINOv3Encoder` / `DINOv3MultiLayerSimpleAddEncoder`）：
- **`Qwen3VLEncoder`**：`load_model()` 取 `Qwen3VLForConditionalGeneration...visual`（`Qwen3VLVisionModel`）、冻结、`eval()`；`_embed_dim=1024`(pre-merger)、`patch_size=16`。`preprocess` / `forward_features` 见下风险（**不是**简单 normalize+resize）。
- **`Qwen3VLMultiLayerSimpleAddEncoder`**：`output_hidden_states=True` 拿每个 `Qwen3VLVisionBlock` 的输出（**不是** output 里的 `deepstack_features`——那是已过 merger 的 2048 维），按 `[layers=...]` 选 → `stack.mean(0)`。
- 注册 `'qwen3vl'/'qwen3vlmls'`，`encoder_name` 形如 `qwen3vlmls-vit-2b[layers=17.18.19.20.21.22.23]`。

**⚠️ 最大实现风险 + 必测（已读 `modeling_qwen3_vl.py` 核实——review pt#2/#3 成立，milestone#1 核心验收，远不止形状 smoke）**：
1. **输入不是 `[B,C,H,W]`**：`Qwen3VLVisionModel.forward(hidden_states, grid_thw)` 吃**展平 patch 序列** `(seq_len, 3·2·16·16=1536)`（`PatchEmbed` 内 reshape 成 `[-1,3,temporal_patch_size=2,16,16]` 走 Conv3d）；**`temporal_patch_size=2` → 静图要打包成 2 帧（复制堆叠）**。pos-emb/旋转/注意力窗口全由 **`grid_thw` 驱动**。→ wrapper **必须复刻 Qwen processor 的 patch flatten + temporal pack + grid_thw**（最稳 = 直接调 `Qwen3VLImageProcessor`/`AutoImageProcessor` 造 `pixel_values`+`grid_thw`，别手搓）。
2. **pre-merger 无 final norm**：`forward` 返回的 `last_hidden_state` = **末 block 原始输出（无后置 LayerNorm）**；唯一 norm 在 merger 里 → feature 归一化是**显式选择**（§4），不能写成"仿 DINO 删末 norm"。
3. **token 不是行主序**：merger 用 `.view(-1, 1024·4)` 把**连续 4 token 当一个 2×2 块**合并 → processor 把 patch 排成 **2×2-块分组序**。→ 还原成 `[B,C,H,W]` 喂 decoder 前**必须 un-permute 回行主序**（`spatial_merge_size=2`），否则 decoder 学到打乱的空间。
4. **deepstack ≠ MLS 源**：output 的 `deepstack_features`(层[5,11,17]) 是**过了 deepstack merger 的 2048 维**，MLS 要取**原始 block 输出**。

**正确性单测（milestone#1 验收）**：① preprocess 的 `pixel_values`/`grid_thw` 与官方 processor 逐位一致；② 抽的 pre-merger 特征 vs 官方模型 `last_hidden_state`/选层 block 输出 **`max|Δ|=0`**；③ **token→2D 映射保真**（喂已知空间梯度图，reshape 后空间布局对）；④ 冻结无梯度；⑤ MLS 选层/聚合正确；⑥ 形状/token 数断言。CPU/小 GPU 可验、上 JCC（坑点#28）。

**取层（对应 doc Q-K1/K7/Deep；Qwen depth 24、deepstack=[5,11,17]）**：

| 名 | layers | 角色 | 规律 |
|---|---|---|---|
| **K7**（主） | [17..23] | 末 7 层 MLS | 生成 Pareto 最优（RAEv2 推荐） |
| **Deep**（主） | [5,11,17,23] | deepstack+末层 | Qwen 原生多深度 |
| K1（诊断·short） | [23] | 仅末层 | 重建最弱/unguided 生成最好；判"末层太抽象" |
| K23（诊断·short） | all | 全层 | 重建上界；连它都崩 → Qwen 整体不适合 |

主矩阵 = K7+Deep；**K1/K23 各 short-budget 做诊断边界**（review pt#4：区分"末层太抽象" vs "Qwen 整体不适合"，不爆卡）。重建随 K 单调升、生成 K7 最优。

---

## 3. 复用 RAEv2 stage1 管线（decoder / loss / disc / 优化）

**Decoder = `GeneralDecoder`（ViT-MAE 式纯自注意力，无 cross-attn/无上采样）**：`decoder_embed`(1024→1152) → 加 trainable CLS + 2D-sincos pos → 28× ViTMAELayer(1152d/16头) → `decoder_norm` → `decoder_pred`(1152→patch²·3) → 去 CLS → `unpatchify`。256px/256token/patch16 → 256px 输出。

**配方（三源对照，⚠ = 对原 doc 的更正）**：

| 项 | RAEv2 repo（我们直接跑的） | Scale-RAE（web 规模+文字） | 原始 RAE |
|---|---|---|---|
| 像素 loss | L1 | L1 | L1 |
| 感知 | LPIPS ×1.0 (VGG) | LPIPS ×1 | LPIPS ×1.0 |
| 对抗 | hinge-GAN ×0.75 + VQGAN 自适应权重 | GAN ×10 | GAN ×0.75 + 自适应权重 |
| **Gram/style** | **无** | **有，ω_G=100** ⚠ | 无 |
| 判别器 | 冻结 **DINO-S/8** dinodisc | DINO-**S/16**（S/8 web 规模不收敛） | DINO-S/8 |
| disc 时序 | LPIPS@0 / disc更新@ep6 / disc进G@ep8 | — | 同左 |
| 优化 | AdamW 2e-4 β(.9/.95) wd0 cosine→2e-5 | LLM/DiT 分开 optim | Adam β(.5/.9) |
| epoch/batch | 16ep / global 512 | 80ep / 2048 | 16ep / 512 |
| EMA(decoder) | 0.9978 | — | 未提（假设无） |
| noise_tau | 0.8（训练期给 latent 加噪→去噪解码器） | — | 0.8（`σ~|N(0,τ²)|`，gFID↓rFID↑微升） |
| 精度 | bf16 | — | — |

**⚠ Gram 更正**：doc 写的"ℓ1+LPIPS+**Gram**+对抗"= **Scale-RAE 配方**（不是 RAEv2 repo）。我前一轮"源码无 Gram"只看了 repo，**更正**：Scale-RAE 在 web+文字规模加了重 Gram(100)，很可能为**纹理/文字 sharpness**。anime 线条/网点/文字可能受益 → **§11 待拍：是否加 Gram**。
**512px**：decoder 有 `gradient_checkpointing` flag 但没接 config → 上 512 时加个开关（坑点#28/显存）。

---

## 4. 归一化（解码器用 token 级 norm，不是 per-channel dataset stats）

- **解码器（stage1）**：原始 RAE + RAEv2 repo 都**直接吃 encoder 输出**（`rae.py` 的 `do_normalization` 默认关）。原始 RAE 是取消 DINO encoder 末 LayerNorm 的 affine → per-token LN。
- **per-channel dataset stats**（`compute_encoder_stats.py` 存 `{mean,var}`）给 **E2 diffusion latent 归一化**用（≈单位方差喂 DiT），**不是 decoder 必需**。
- **⚠ 对 Qwen3-VL 不能照搬"删末 norm affine"（review pt#3，已读码核实）**：Qwen pre-merger 的 `last_hidden_state` = **末 block 原始输出、本就没后置 norm**（唯一 norm 在 merger 里）。→ Qwen feature 归一化做成**显式小消融**，三选一：
  - `raw`（直接吃 block 输出）；
  - **`LN(no-affine)`**（自己加个无 affine LayerNorm，最接近 RAEv2 精神 → **默认**）；
  - 复用 `merger.norm`（merger 里那个带 affine 的 `LayerNorm(1024)` token norm）。
- **⚠ 更正我前一轮"decoder 必须开 per-channel 归一化"**：那是 E2 的事；E1 decoder 用上面 token 级 norm 即可。

---

## 5. 数据接入 + "更多数据"证据

**E1 解码器是无条件重建**（trainer 直接 `for images,_ in loader`，丢 caption/label）→ **只需图**。
- 写一个**薄 image-only dataset 适配器**喂我们的图（复用 mot_pilot_v1 `dataline/image_fetch` 的 R2-HTTP/tar/本地取图，去掉 caption/sampler 复杂度；重建用均匀采样即可，但 **anime 仍要随机 row-group 去头部偏置**——坑点#17）。
- 数据组成 = **anime 主力**（目标域）+ **real 子集**（当前 ~20M 筛选取子集、保留细节多的图）+ **文字集（必需）**。

**数据规模证据（答 Evan"越多越好？"——decoder ≠ generator）**：

| 证据 | 结论 |
|---|---|
| RAEv2 Table14：1.28M→65M（50×）rFID 只降 **7.7%(K7)/14.6%(K23)** | decoder 是有界求逆，**强烈边际递减** |
| Scale-RAE Table1：Web+Synthetic(64M) 文字 rFID=2.406 ≈ ImageNet-only 2.640；**+文字数据 73M → 1.621(−40%)** | **coverage>count**：堆量没用、补**弱项内容类型(文字)**才有用 |
| Scale-RAE Table3(生成侧)：Synthetic×2=48.0 < Synthetic+Web=49.5 | 混**互补源** > 翻倍单源 |
| 原始 RAE：ImageNet 1.28M×16ep≈20M view 就到 VAE 竞争力 | **~20–40M view 预算**就收敛 |

→ **预算 ~20–40M view**（anime 16M 跑 1–2 epoch ≈ 16–32M view 就够）；**200M 真实照片对 decoder 边际最小、且多为 off-domain**，留给 E0/E2 生成器。

**per-channel stats（E2 用）**：目标域采样几十万张算一次即可（非规模问题）。
**eval 集**：anime（从源 parquet 按 tag 采有标注子集，参考 `Concept-Eval/tags_eval` 思路）+ **文字/细节集（闸门必需）**。

---

## 5b. 数据盘点：s3-mewtant-raw-datasets（已查，2026-06-24）

`/s3-mewtant-raw-datasets`（只读 mount，119 个数据集）里**已有同事备好的 manga/文字/细节数据**，多为 `images/ + jsons|parquet/` 布局，**无需外部下载即可起步**：

| 数据集 | 内容 / 对 E1 价值 | 格式 |
|---|---|---|
| **mangadex** | MangaDex 漫画页（多语含中日文、网点、对话框）= **目标域文字主力** | images/(每话 UUID 目录) + jsons/ |
| **manga-detext** | 漫画 + **去字/文字区标注**（parquet）→ 可做文字区加权 eval | images/(PNG) + parquet/ |
| **toongod** | webtoon（CJK 竖条，已 `resplit/` 切板）= 文字+细节 | images/(按标题) + jsons/ + resplit/ |
| **rawkuma / twitter-manga** | 漫画 raw / twitter 漫画图 | images/+jsons/ ; raw/+failed/ |
| **4khd** | 4K 高清 + **已跑 OCR worker**（文字区标注）+ cut 高细节裁块 | raw/cut/prefilter/cut_parquet_outputs |
| **DOCCI** | 15k 高质细节图 + dense caption（通用细节 eval 好用） | tar.gz + jsonlines |
| **FaceCaptionHQ-4M** | 人脸（RAE 另一"特定内容"弱项，可选） | — |

**外部高质文字集（补 CJK 字形广度，按需 HF 直拉/Evan 备）**：**AnyWord-3M / AnyText**（多语含**中文**+OCR 标注，CJK 字形最广）· **TextAtlas/TextAtlas5M**（Evan 在备，大规模密集英文文字、版式多样）· MARIO-10M(英文) / RenderedText(英文合成，RAEv2 用) / **Manga109**（JP 漫画基准，研究授权、小，适合 **eval**）。

**E1 文字数据（Evan 2026-06-24 拍板）**：训练 = **bucket 漫画 `mangadex`+`manga-detext`+`toongod`**（目标域、自带 CJK 文字）+ **TextAtlas**（英文密集，Evan 在备）。**AnyWord-3M / 4khd 暂不用**（备选，若 CJK 字形覆盖不足再补）。eval = `manga-detext` 文字区标注挑"含字"图 + Manga109(JP 基准，研究授权)。
⚠ 这些是 raw 图源（PNG/tar/webtoon 长条）：接入前要分辨率/方图裁切、webtoon 切板（toongod 已 resplit）、质量筛（去过小/水印）；复用 anime 那套 `image_fetch` + 加简单清洗。

---

## 6. 对照组（repo 几乎白嫖）

| 类 | 具体 | 回答什么 |
|---|---|---|
| RAE-SSL 标杆 | `dinov3mls-vit-l16` / `webssl` | "Qwen 的 OCR/理解训练是否伤重建"的上界对照 |
| RAE-理解型 | `siglip2mls-vit-l` | 与 Qwen 同属"理解型"的同类对照 |
| RAE-MAE | `mae-vit-l` | 重建强/表征弱的极端点 |
| VAE | **FLUX.2**（我们在用）+ Qwen-Image（落选留对照） | RAE 要打的基线（repo `stage1/vae.py` 有脚手架） |

全部**同分辨率/同数据/同 eval 集**，公平比。

**对照 encoder 优先级（review pt#5）**：SigLIP2 必跑（理解型、最像 Qwen）；SSL 槽 **DINOv3**(RAEv2 默认/最强/repo-native `dinov3_loader`) 或 **WebSSL**(Scale-RAE 正本里和 SigLIP2 头对头的 SSL 对照) 二选一——默认 DINOv3，若想更贴 Scale-RAE 叙事则换 WebSSL（接入成本相近，都 repo-native）。MAE 是极端点(重建强/表征弱)备选。

---

## 7. 可扩散性 probe（E1 决定性闸门）

- **最小 LightningDiT**：`patch_size=1`（→ 1×1 conv 把 C=1024 投到模型宽度，无需额外 head）、depth~12、width 768–1152；transport `prediction='x'`、`logit-normal`、`t_eps=0.05`、**REPA/IG 关**（隔离 base 可扩散性）。
- **noise-shift**：probe 阶段就按 latent 真实形状设 `shift=√(N·C/4096)`。256×1024 → `√(262144/4096)=8.0`。Scale-RAE 实测这是 **"non-negotiable"（GenEval 49.6 vs 23.6）**。
- **关键确认（验证 doc"省 wide-DDT 头"）**：RAE Theorem 1 = DiT 宽度须 ≥ token 通道维。我们目标模型宽度 **2048 > latent 1024** → 满足，**不需 wide-DDT 头**；probe 用 width≥1024 即可。
- **probe 闸门**：Qwen-RAE probe 的 gFID/收敛 vs **同尺寸 VAE-probe**（apples-to-apples）。
- **公平性（review pt#7，必写进报告）**：RAE latent=256tok×**1024**ch、FLUX.2-VAE latent=256tok×**128**ch——**token 数同(256)→ DiT FLOPs 近似**，但要同时报 **matched steps / matched images / matched GPU-hours / 参数量 / 吞吐**，否则 "probe ≥/< VAE" 可能被训练预算解释掉。⚠ 两者 noise-shift 按公式天然不同（VAE `√(256·128/4096)=√8≈2.83` vs RAE `√64=8.0`）——这是 per-latent **正确**设置、要 document，不是 unfair。

---

## 8. 评测 / 闸门产出

- 重建：rFID（torch-fidelity Inception，配对 recon vs 参照 npz）+ PSNR/SSIM/LPIPS —— repo `offline_eval_stage1.py` 现成。
- 生成：gFID（同参照）。
- **eval 集必含细节/文字**。产出**多路对照表**：Qwen-RAE{K1/K7/Deep} × {DINOv3 / SigLIP2 / WebSSL / FLUX.2-VAE}，列 rFID/PSNR/SSIM/LPIPS/probe-gFID。

---

## 9. 依赖 + 环境

- RAEv2 repo 用 `uv`；我们用 **venv-gpu**。增量装：`timm`、`open_clip`/CLIP、`torch_fidelity`、`torchmetrics`、DINOv3/SigLIP2（torch.hub / HF）。仿 diffusers 先例用 `pip install --no-deps`（避免动 torch/transformers，坑点#2）。**Qwen3-VL 走已装 transformers 5.11**。
- 缓存/产物 **/local**，不写 /fsx 大件。

---

## 10. 算力 / JCC 计划（守坑点#28）

- **全上 JCC**（含 CPU 单测/smoke）；本机只做秒级文件/git。`trainer.save_final=false`、输出/cache 指 /local。
- decoder bake-off：256px、8×H100、global batch 512、~20–40M view（anime 16M 1–2 epoch）→ 估 **~40–80k step/run**；每个 encoder/K 一个 run。
- probe：小 DiT，更便宜。
- 多 run 各独立 `checkpoint_dir`、`resume=true`+`maxRetries`。

---

## 11. 开放点（2026-06-24 已拍板）

1. ✅ **Gram 项**：先按 repo（无 Gram）跑基线，再 **A/B 加 Gram**（干净归因）。
2. ✅ **取层 K**：测 **K7 + Deep[5,11,17,23]** 两组。
3. ✅ **文字数据**：bucket 漫画 `mangadex`+`manga-detext`+`toongod`（目标域、自带 CJK）+ **TextAtlas**（英文，Evan 备）；AnyWord-3M/4khd 留备选。详见 §5b。
4. ✅ **probe 时机**：**E1 就做**（决定性信号）。
5. ✅ **对照 encoder**：先 **dinov3(SSL标杆) + siglip2(理解型对照)** 两个，算力够再加 webssl/mae。

---

## 12. 里程碑 / 交付顺序（每步带验收）

| # | 步骤 | 验收 |
|---|---|---|
| 0 | 新目录 `mot_rae_e1/`（fork RAEv2 clone、自带 git）+ venv-gpu 装 deps + 拉 DINOv3/SigLIP2/WebSSL 权重（JCC 小节点验 import） | encoder + decoder 能 import/构造 |
| 1 ⭐ | **Qwen3VL encoder wrapper + 正确性单测**（最关键、最先做） | ① preprocess 的 `pixel_values`/`grid_thw` == 官方 processor；② pre-merger 特征 vs 官方 `last_hidden_state`/选层 block `max\|Δ\|=0`；③ **token→2D 映射保真**；④ 冻结无梯度；⑤ MLS 选层/聚合正确 |
| 2 | RAE(Qwen)+decoder 小数据 overfit 几百步 | 重建 loss 降、出图像样 |
| 3 | image-only 数据适配器（anime+real子集+bucket漫画+TextAtlas）跑通 | 流式喂入、无 caption 依赖、token 顺序对 |
| 4 | 正式 decoder 训练（256px）→ extract EMA decoder → compute stats | 各 run 收敛、EMA decoder 落盘 |
| 5 | offline_eval：多路重建对照表（含文字集） | rFID/PSNR/SSIM/LPIPS 表出 |
| 5b | **早放小 512 文字/manga 重建 sanity**（不等 256 全跑完，review pt#6） | 判 256 文字差是否 resolution 假阴性 |
| 6 | probe：小 DiT 各 latent 上训 → gFID（含 §7 公平性列） | Qwen-RAE vs VAE probe 对照表 |
| 7 | 汇总闸门结论 | go/no-go：进 E2 / E5 / 回 VAE |

**run matrix（review 建议）**：Smoke(Qwen wrapper 正确性 + overfit) → **Main**(Qwen K7 / Qwen Deep / SigLIP2 / 一个 SSL) → **Diagnostic**(Qwen K1/K23 short，仅主结果不清时扩) → **A/B**(Gram 只在最优 Qwen 配置上做、不与所有 encoder/K 交叉) → **Early-512**(少量 anime/text 512 重建 sanity)。

> ⚠ 本 E1 是**独立 repo 管线**（`mot_rae_e1/`），不触 mot_pilot_v1 在跑代码；probe/采样若复用缓存路径，确认无遗留 bug（本项目坑点）。
