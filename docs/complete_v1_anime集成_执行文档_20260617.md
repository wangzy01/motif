# MoT complete_v1 —— anime 集成执行文档（交接给执行 AI）

**日期**：2026-06-17 ｜ **目标**：把团队交付的 anime 池并入已建好的 `daily_v1`(real+synth 21M)，形成"完整版" `complete_v1`（探索规模 ~30M：daily 21M + anime ~10–12M 有效曝光）。｜ **状态**：方案已与 Evan 全部拍板定稿；**本文交给另一个 AI 执行**。Evan 出品，本文是执行权威。

> 阅读顺序：先读本文 §1 上下文 + §3 决策汇总，再按 §4–§12 执行。配套既有件：`daily_v1` 交付见 [日常探索版筛选spec_v1_20260617.md](日常探索版筛选spec_v1_20260617.md)；母池见 [AAA数据交付文档_GPIC+MONET+anime_v2_20260617.md](AAA数据交付文档_GPIC+MONET+anime_v2_20260617.md) + [采样权重spec_v1_20260616.md](采样权重spec_v1_20260616.md)；设计正本两份 v3 doc。可复用代码 `mot_code/data_analysis/scripts/t5_build_daily_manifest.py`、`t5_daily_qc.py`、`configs/t5_sampling_daily_v1.json`；naifu 参考 `mot_code/naifu-main/training/{caption_plan,sampling_plan}.md`、`training/r2_utils.py`。
>
> **本文是 anime 集成的 4 域执行权威**；其交付物/指标摘要见 [AAA数据交付文档](AAA数据交付文档_GPIC+MONET+anime_v2_20260617.md) §11（AAA §11 是本文的汇总，执行细则以本文为准）。
>
> **【2026-06-27 更新：complete_v1 已扩到 5 域】** 在本文 4 域（anime/MONET-real/GPIC/synthetic）基础上加入第 5 个注入域 **TextAtlas**（i1 开源长 caption 文字渲染数据，像 synthetic 一样注入），并修复了 mix-regression。**第 5 域设计 + 5 域 domain_mix + mix-fix 见新整合文档 [complete_v1_dataline_设计与变更_v1_20260627.md](complete_v1_dataline_设计与变更_v1_20260627.md)；权威 config 为 `t5_sampling_complete_v1_textatlas.json`。** 下文 §6 的 4 域 domain_mix 表为历史沿革（real/synth 域内机制、anime caption/采样仍现行有效）。

---

## 0. TL;DR

- **complete_v1 = daily_v1（real+synth 21M，不动）+ anime**。三阶段课程 256→512→1024，anime 全程主力。
- **产出**：① 薄 anime manifest（`t5_manifest/daily_v1/anime/`，pointer，caption 文本训练时按 key 回团队 parquet 取）② 统一采样 config `configs/t5_sampling_complete_v1.json`（完整 domain_mix + anime caption/权重/路由 + real/synth 沿用）③ QC json。daily_v1 三域 manifest **不改**。
- NSFW全保留
- **纯 CPU**，无 GPU。anime 图走 R2 HTTP（训练侧，naifu `r2_utils` 已有）。

---

## 1. 背景与上下文（执行者必读）

- **项目**：Evan 的 MoT/UMM anime-T2I（~1.6B 可训练参数的生成塔）。anime 是**目标域**，real(GPIC+MONET 摄影/通用) + synthetic(flux/z-image) 是**多样性/防遗忘支撑**。
- **`daily_v1` 已建好**（job `evan-mot-t5-daily-select-tklr` Succeeded）：21M = GPIC 7M + MONET-real 11M + synthetic 3M，38 列 schema（33 母池列 + `wk/sscd_nn/elig_256/elig_512/elig_1024`），holdout=0，QC 全绿。本轮**不动它**，只在其旁并入 anime。
- **规模决策（团队群 + Evan）**：本轮做**探索版 ~30M**（anime ~10M + real ~20M；i1 经验：多样性 > 总量，30M 对 1.6B 够）。**完整全量 ~90M（full real 60M + synth 13.8M + anime 16.5M）是后续 scaling，本轮不做。**
- **anime 已是团队"独立过滤管线"产物**，不需我们再筛（沿用团队 `ct_*` 软校准）。
- **naifu = 公司训练框架**（`/fsx/evan/mot_code/naifu-main`，troph-team）。已查证：rating 作 caption 条件标签（非过滤）、policy 加权 caption route、R2 取图、`cluster_clipl14_level2` 平衡 + `ct_weight` 有界重采样。我们的 anime 用法对齐 naifu。

---

## 2. 输入数据

### 2.1 daily_v1（real+synth，已建，**不改**）
`/s3-pixai-test-uw2/evan/mot_filtering_v1/t5_manifest/daily_v1/{gpic,monet,monet_synth}/*.parquet`（各 64 shard）。38 列含 `elig_256/512/1024`。已含采样 config `configs/t5_sampling_daily_v1.json`（present_renorm 域配比 —— 本轮被 complete config 取代）。

### 2.2 anime（团队交付，待并入）
`/s3/evan/anime_core_ditv04_ct_soft/`（注意挂载是 `/s3/` 非 `/s3-pixai-test-uw2/`）。`README.md` + `data_schema.md` + `data/00000000.parquet ~ 00000016.parquet`（17 个，zstd）。
- **16,499,287 行，139 列，uuid 去重**。
- 图像 pointer-only：`s3key` = `https://ncotar-worker.mewtant.workers.dev/r2/<key>`（R2 worker URL）。`key` 主键、`uuid` 全局唯一。
- 分辨率：短边 ≥512 **98.8%(16.29M)**、≥1024 **66.6%(10.99M)**、<512 仅 1.2%。
- caption：`refined_description`(长，p50~318tok)/`medium_description`(~214)/`short_description`(~29) **三者同 null，captioned 子集 = 11,699,792(70.9%)**；其余 27% tag-only。`tags_{general,character,copyright,artist,meta}`(List[String]) 100% 有，general 中位 ~44 tag；character 68%/copyright 88%。
- rating_bucket：general 46.1% / sensitive 38.6% / questionable 8.1% / explicit 7.2%。is_non_ai 77%。
- 质量/校准（团队软校准，cell 内相对，勿跨源硬比）：`ct_weight`(1.0–2.5,p50 1.75 推荐训练权重)、`ct_filter_score`(0.009–1.0 软通过)、`ct_quality_score`、`quality_z`、`hps_v3_score_qwen3vl`、`compound_aes_score`、`raw_aes_score`、clarity 11 分量、anatomy、`ct_lanes`(List)、`ct_is_base/ct_is_supplement`、`ct_real_prob`。
- 簇：`cluster_clipl14_level2`（**naifu 平衡主轴**，执行时确认基数）、`cluster_clipl14_level1/3`、`cluster_style_v15_level1/2/3`、`style_l1`(1135)/`style_l2`(49992)。
- 路由桶：`risk_bucket`(plain_illustration 13.57M / text_layout 0.89M / comic 0.47M / comic_text 0.36M / photoish 0.50M / ...)、`cls_bucket`、`ar_bucket`、`person_bucket`、`source_group`、`dataset`(18 源含 ehentai 0.21M)。
- 完整列见 `/s3/evan/anime_core_ditv04_ct_soft/data_schema.md`。

---

## 3. 决策汇总（全部已拍板，含依据）

| # | 项 | 决定 | 依据 |
|---|---|---|---|
| 1 | 规模 | 探索版 ~30M：daily 21M + anime ~10–12M 有效曝光 | 团队群共识 + i1 多样性>总量 |
| 2 | anime 口径 | **manifest 放全 16.5M**；曝光由 domain_mix+ct_weight 收到 ~10–12M。**静态可用 ≠ 每 epoch 曝光** | Evan：全部放进去；规模靠采样收 |
| 3 | NSFW | **全保留** | 内部非开源模型。Evan + naifu + 我的判断 |
| 4 | domain_mix | 激进 anime：见 §6 表（256=30/512=65/1024=80） | Evan 选激进 |
| 5 | anime caption | naifu policy 加权 route（rich_nl/nl/nl_short/tag_medium）；Stage1 偏 NL → Stage2 提 tag + tag-freq dropout；rating(q/e) + 类目排序 booru tags | v3 + naifu caption_plan.md |
| 6 | anime 采样/多样性 | `cluster_clipl14_level2` 平衡 + `ct_weight`/quality/`ct_lanes` 有界重采样 + lane caps（text_layout/comic_text/photoish 低权）；**不做静态选样** | naifu sampling_plan.md；团队软校准意图 |
| 7 | 产出形态 | 薄 anime manifest + 统一 mixer config + spec；daily_v1 不动 | Evan 选 |
| 8 | 分辨率路由 | 256 全 / 512 ≥512 / 1024 ≥1024（anime ≥1024 **10.99M**；旧记 8.1M 实为其带 NL 子集 8.17M，QC 已更正，见 §2.2 / AAA §11.7）| v3 课程 |
| 9 | 图像 | R2 HTTP 取（训练侧，naifu r2_utils） | 数据本身 pointer |


## 4. 薄 anime manifest（产出①）

**目的**：与 daily_v1 并列、可统一查询的轻量 pointer 索引；caption 文本**不落库**，训练时按 `key` 回 `/s3/evan/anime_core_ditv04_ct_soft/data/` 取（naifu 同款）。

**输出**：`/s3-pixai-test-uw2/evan/mot_filtering_v1/t5_manifest/daily_v1/anime/anime_*.parquet`（建议 64 shard）。

**构建逻辑**（新脚本 `scripts/t5_build_anime_manifest.py`，复用 t5_build_daily_manifest 的 put/write_sharded/threaded-read 模式）：
1. 读 17 个 anime parquet 的**轻列**。
2. 派生 `elig_256=1`、`elig_512=(min(w,h)>=512)`、`elig_1024=(min(w,h)>=1024)`；caption 可用性 flag。
3. 写薄 manifest。CPU jcc，~10–20min。

**薄 manifest 列**：
| 列 | 来源/算法 | 用途 |
|---|---|---|
| `uid` | `anime:{key}` | 全局唯一（与 daily_v1 `uid` 风格一致，可 union/区分）|
| `key`,`uuid` | 源 | 回源 caption / 去重 |
| `s3key` | 源 | R2 取图 URL |
| `dataset` | 源 | booru 源（采样分层/审计）|
| `width`,`height` | 源 | 分桶 |
| `elig_256/512/1024` | 派生 | stage 路由 |
| `rating_bucket` | 源 | 条件标签(q/e) + 审计 |
| `is_non_ai` | 源 | 审计 |
| `aes` | `compound_aes_score`（另留 `ct_quality_score`）| 质量参考（不硬筛）|
| `ct_weight` | 源 | **采样权重** |
| `ct_filter_score` | 源 | 软通过参考 |
| `ct_lanes` | 源 | lane 路由/caps |
| `cluster_balance` | `cluster_clipl14_level2` | **多样性平衡主轴** |
| `style_l1` | 源 | 备用风格簇 |
| `risk_bucket` | 源 | lane caps（text_layout/comic_text/photoish）|
| `cls_bucket`,`person_bucket` | 源 | 分层/审计 |
| `has_nl` | `refined_description` 非空（或任一结构化字段，对齐 naifu `_has_visual_nl`）| caption route 选择 |
| `has_short` | `short_description` 非空 | route |
| `n_tags_general` | `len(tags_general)` | tag route 可用性 |

> caption 文本字段（refined/medium/short/tags_*/overlay_texts）**不进薄 manifest**，训练时按 `key` 从源 parquet 读（naifu caption builder 直接吃源 parquet）。

---

## 6. 统一采样 config（产出②）`configs/t5_sampling_complete_v1.json`

以 `t5_sampling_daily_v1.json` 为模板扩展，加入 anime 域。**domain_mix（激进 anime，已和不为 1 的隐患归一，三阶段各 sum=1.0）**：

| stage | anime | MONET-real | GPIC | synthetic | 该阶段可用池(静态) |
|---|---:|---:|---:|---:|---|
| 256 | **0.30** | **0.397** | **0.253** | 0.05 | ~37.5M（daily 21M + anime 16.5M）|
| 512 | **0.65** | 0.31 | 0 | 0.04 | ~28.1M（anime≥512 16.1M + MONET ge512 8.98M + synth 3M）|
| 1024 | **0.80** | 0.08 | 0 | 0.12 | **16.04M**（anime≥1024 **10.99M** + MONET ge1024 2.05M + synth 3M）｜旧记 13.1M/8.1M 为 QC 前估值，已更正 |

- real/synth 域内：沿用 daily（**256 MONET:GPIC=11:7**【2026-06-17 修订：原 80:20 → 改用 daily 静态池 11M:7M 自然比，即 monet_real 0.397 / gpic 0.253，让 GPIC 按真实池占比进 256】；512/1024 GPIC=0；band-first 18/27/15/40 + band→captioner 优先级 + 10% dropout；运行时 α/β 调轻防双重拉平）。
- anime 域内：见 §7 caption + §8 采样。
- **资格用 manifest 的 `elig_*` 列**；1024 仅 `elig_1024==1`。
- **探索规模实现**：不静态裁 anime；用 epoch/步数预算 + 上表 domain_mix + `ct_weight` 把 anime 每-epoch 曝光收到 ~10–12M-equiv（静态可用全 16.3M）。如要更小静态池可另出子集，默认不裁。
- **1024 档供给注意**：daily 自身 1024 的 T1/T4 曾绑定（见 daily spec §8）；**anime 并入后大幅缓解**（anime≥1024 **10.99M**，其中带 NL 8.17M → 海量 `refined`→长档）。但 real/synth 仍按各自 band-first，执行后用 QC 确认 1024 综合供给（见 §10）。

---

## 7. anime caption 路由（产出②内，对齐 naifu）

**字段映射**：NL-long=`refined_description`、NL-short=`short_description`（`medium` 备用）、tag=`tags_{artist,character,copyright,general,meta}` 类目排序、rating=仅 `questionable/explicit` 渲染 `rating:<x>`（general/sensitive 不渲染，naifu caption_plan.md:128）。

**route 策略**（建议**直接复用 naifu caption builder 逻辑**，见 caption_plan.md §Route weights/§Common-trait dropout/§Length cap）：
- policy 加权 route：`quality_default` = rich_nl 28 / nl 18 / nl_short 14 / tag_medium 40（has_nl 时）；no_nl 行 → tag_medium 100；另有 `nl_heavy`/`artist_anchor`/`concept_locked` 等表。
- **common-trait dropout**（频率加权丢高频 general tag，min_keep 6 / 20%）；**length cap**（按概率截 general tag）。
- **Stage1 偏 NL/hybrid**（保 prompt-following）；**Stage2 提 raw tag 比例 + tag-frequency dropout**（保角色/画风/长尾实体、防热门 tag 过拟合）。
- 长档截 ~224–256 tok（与 real 一致）。
- tag-only 行（27%）走纯 tag route，**不丢**。

> 执行者抉择（训练侧）：直接调 naifu 的 caption builder（它就吃这些源字段），或在 MoT loader 里实现等价简化版。推荐复用 naifu 逻辑以减偏差。

---

## 8. anime 采样 / 多样性（产出②内，对齐 naifu sampling_plan.md）

- **多样性平衡主轴 = `cluster_clipl14_level2`**（簇平衡构建为主，类似 real 的簇拉平）。
- **优先级/曝光 = 有界重采样**：由 `ct_weight` + quality(`quality_z`/`hps_v3`/`raw_aes`) + `ct_lanes` 成员 + 稀有概念覆盖加性成一个**封顶**优先分，转成整数 repeat（曝光塑形，非把小集合复制成新池）。
- **lane caps**：对 `risk_bucket ∈ {text_layout, comic_text, photoish, photoish_3d, meme, game_ui}` 设上限/低权，避免无提示出字、漫画格/写实漂移。
- **不做 real 式静态 Gumbel 选样**（团队软校准就是为全量带权设计）。

---

## 9. 产出物清单

| 产物 | 路径 | 说明 |
|---|---|---|
| 薄 anime manifest | `t5_manifest/daily_v1/anime/anime_*.parquet` | §5|
| 统一采样 config | `mot_code/data_analysis/configs/t5_sampling_complete_v1.json` | §6/§7/§8 |
| QC 报告 | `t5_manifest/daily_v1/complete_v1_qc.json` | §10 |
| 构建脚本 | `scripts/t5_build_anime_manifest.py` + dispatcher `daily`-style 分支 + jcc yaml | §11 |
| 集成 spec（本文）| `doc/mot/complete_v1_anime集成_执行文档_20260617.md` | 权威 |

daily_v1 的 `gpic/monet/monet_synth` 三域 **原样不动**。complete_v1 = daily_v1 四子目录(含新 anime) + complete config。

---

## 10. QC 验证清单（产出③）

复用 `t5_daily_qc.py` 模式，新增 anime + 跨域。必须报：
1. **完整性**：anime 行数（排除前后）、uuid 唯一、各 elig 计数。
2. **anime 分布**：rating（排除后）、source、is_non_ai、res（≥512/≥1024）、`ct_weight`/`ct_filter_score` 分位、`cluster_clipl14_level2` 覆盖与最大簇占比、has_nl vs tag-only 比例、risk_bucket（text/comic/photoish 占比，验 lane cap 必要性）。
3. **stage 池**：256/512/1024 各域可用量（对照 §6 表）；**1024 综合 caption 供给**（anime route 可服务量 + real/synth band-first 18/27/15/40 的 oversample_factor，确认不再绑定）。
4. **跨域曝光 dry-run**（可选但建议）：按 complete config 抽样百万级，报每 stage 实际 anime/MONET/GPIC/synth 占比命中 domain_mix、anime caption route 分布、单图 repeat top-k（ct_weight 重采样别爆）。

---

## 11. 构建步骤（jcc，CPU-only）

1. 写 `scripts/t5_build_anime_manifest.py`（§5）。复用 `t5_build_daily_manifest.py` 的 `put()`/`write_sharded()`/线程读/canonical-cast。读 `/s3/evan/anime_core_ditv04_ct_soft/data/*.parquet` 轻列。
2. `run_t5_select_in_pod.sh` 加 `anime)` 分支；写 `cluster/t5_anime_manifest_job.yaml`（`SEL=anime`、`gpuPerWorker:0`、cpu32/mem256，参考 daily yaml）。
3. 先 **smoke**（`MAX_FILES=2` 小子集）跑通逻辑，再全量。写 `daily_v1/anime/`。
4. 手写/模板生成 `configs/t5_sampling_complete_v1.json`（§6/§7/§8；以 `t5_sampling_daily_v1.json` 为底，加 anime 域 + 改 domain_mix）。
5. 跑 QC（§10），写 `complete_v1_qc.json`，核对全绿后交付。
6. 环境：`/fsx/evan/mot_code/venv-gpu/bin/python`（pandas/pyarrow），镜像 `harbor.jh1.mewtant.io/public/autobuild:pytorch-cu128`，job 名 `evan-` 前缀。

---

## 12. 训练侧依赖 & 已知 caveat

1. **anime 图 = R2 HTTP 取**（`s3key` worker URL），不在本地挂载；loader 需 R2 fetch（naifu `training/r2_utils.py` 已有，可复用/参考）。real/synth 仍读本地 .tar/parquet（daily 既有）。
2. **caption builder**：建议复用 naifu 的（吃源 parquet 字段）；若 MoT loader 自实现，对齐 caption_plan.md 的 route/dropout/length-cap + rating(q/e) 渲染。
3. **探索 vs 全量**：本轮探索 ~30M。后续 scaling 到 ~90M = 换 full rich_v1(60M) + full synth(13.8M) + anime full，同框架、改 domain_mix/规模即可。
4. **ct_* 是 cell 内相对值**，勿跨源/跨 level 硬比（团队说明）。
5. daily_v1 的 1024 档 caveat 由 anime 缓解但 real/synth 侧仍存，最终 1024 band 配置以 §10 QC 为准。
6. **caption 截断（text_max_length=384）**【2026-06-17 提醒，2026-06-18 上调】：模型 `text_max_length=384` ≈ 模板 ~41 tok + caption ~343 tok（真 token budget 343）；**超长由 tokenizer 硬截断、整条不丢**。loader 须先按**句子/tag 边界**把 caption 截到 ~300-340 proxy(真 token budget 343)（别 mid-token 硬切），并在 e2e dry-run 统计**真实截断率**，避免大量 caption 被静默硬截。anime/real/synth 同样适用。
