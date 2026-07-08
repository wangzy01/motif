# MoT 数据交付文档 —— GPIC + MONET real + synthetic 1024 + anime（v2）

**日期**：2026-06-17（**v2 加 anime**；v1 GPIC+MONET+synth 部分 2026-06-16 已交付，下文 §1–§10 不变）｜ **范围**：GPIC + MONET + synthetic（60M 母池 + 13.8M 1024 锚池）的过滤/去重/选样/manifest/采样权重；**v2 新增团队 anime 池 16.5M 薄 manifest + complete_v1 集成（见 §11）** ｜ **状态**：✅ 数据侧已交付、已验证（多轮独立 AI 审计 + QC 全绿）；✅ **训练侧 loader 已实现并端到端跑通（2026-06-17，V1 可开训，见 §11.6）**

> 本文是这条数据线的**单一权威交付文档**：记录交付物、路径、关键指标、所有决策、如何使用、溯源、未尽事项。配套细则见 [采样权重spec_v1_20260616.md](采样权重spec_v1_20260616.md)、处理过程流水账见 [README_T2_NEXT.md](../../data_analysis/cluster/README_T2_NEXT.md)。设计依据见两份 v3 doc：[模型+训练设置v3](自己写的模型+训练设置文档_v3_20260617.md)、[数据集v3](自己写的数据集文档_v3_20260616.md)。
>
> **v2（2026-06-17）新增 anime 交付 + complete_v1 集成 —— 见 §11**；统一采样 config 为 `configs/t5_sampling_complete_v1.json`；anime 集成执行文档见 [complete_v1_anime集成_执行文档_20260617.md](complete_v1_anime集成_执行文档_20260617.md)。daily_v1 三域与 60M 母池均**未改动**。
>
> **【2026-06-27 更新：complete_v1 现为 5 域】** complete_v1 已加入第 5 个注入域 **TextAtlas**（i1 开源长 caption 文字渲染数据，像 synthetic 一样注入），并修复了 mix-regression（drain_mode=proportional + anime 串行取图 + gpic 并发 + 行级 shuffle + 监控）。5 域 domain_mix（权威 config = `t5_sampling_complete_v1_textatlas.json`）：256 = anime 0.27 / monet_real 0.357 / gpic 0.228 / synth 0.045 / **text 0.10**；512 = 0.65 / 0.13 / 0.08 / 0.04 / **0.10**；1024 = 0.80 / 0.08 / 0 / 0.12 / **0**（1024 textatlas plumbing 就位、权重 0 暂不启用）。**第 5 域设计 + mix-fix 细节见新整合文档 [complete_v1_dataline_设计与变更_v1_20260627.md](complete_v1_dataline_设计与变更_v1_20260627.md)，本文不再复述**；下文 §5/§11 的 4 域骨架/数字为历史沿革，最新配比以该文 + 权威 config 为准。

---

## 0. TL;DR

- **real 训练池**：60,000,000 张 = **GPIC 20M + MONET 40M**（MONET 主力）。manifest：`t5_manifest/rich_v1/`（512 parquet shard，33 列）。
- **synthetic 1024 锚池**：13,808,747 张（全 1024px，flux/z-image 生成），**不计入 60M real**，供 1024 阶段高清防遗忘。manifest：`t5_manifest/synthetic_v1/`（64 shard，同 schema 可 union）。
- **caption = 指针模型**：manifest 只存定位符 + token 数/档位，**训练时回源读图与 caption 文本**；权重**不烘焙**进 manifest。
- **采样**：band-first（精确命中 caption 配比 T1/T2/T3/T4 = 18/27/15/40）+ 源/簇温和拉平，分辨率课程 256→512→1024。spec/config/参考实现齐备、dry-run 验证。
- **⚠️ 交付训练侧必带 8 条 loader 规则**（见 §6.2，两轮审计后定、数字已在 parquet 上验证）：holdout 全局排除、band 合格性以 tok>0 为准（非 has_tX）、caption fallback、tok≤0 不可用、GPIC@512/1024=0、aes 缺失兜底、全局统计、domain_mix 归一。
- **【v2】anime（目标域）**：团队交付池 `anime_core_ditv04_ct_soft` 16,499,287 行 → 薄 pointer manifest `$S3/t5_manifest/daily_v1/anime/`（64 shard，25 列）；全量放入、**不过滤、NSFW 全保留**，曝光靠采样收。complete_v1 = daily_v1(real+synth 21M) + anime，统一 config `configs/t5_sampling_complete_v1.json`（4-way domain_mix anime 0.30/0.65/0.80 + 三档 caption + ct_weight 优先级 + lane caps）。QC 全绿。**图像走 R2 HTTP、caption/tags 训练时回源**。详见 §11。✅ **训练侧 loader 已实现并端到端跑通（§11.6，V1 可开训）**。
- **【2026-06-27】complete_v1 现为 5 域**：anime/monet_real/gpic/synthetic + **TextAtlas（第 5 注入域）**；权威 config `t5_sampling_complete_v1_textatlas.json`，5 域配比与 mix-fix 详见 [complete_v1_dataline_设计与变更_v1_20260627.md](complete_v1_dataline_设计与变更_v1_20260627.md)。本文以下 §5/§11 的 4 域内容为历史沿革。

---

## 1. 交付物清单与路径

S3 根：`/s3-pixai-test-uw2/evan/mot_filtering_v1`（下记为 `$S3`）。源数据只读挂载见 §1.3。

### 1.1 Manifest（核心交付物）
| 产物 | 路径 | 行数 | shard | 说明 |
|---|---|---|---|---|
| real 富 manifest | `$S3/t5_manifest/rich_v1/gpic/*.parquet` | 20,000,000 | 256 | GPIC 部分 |
| | `$S3/t5_manifest/rich_v1/monet/*.parquet` | 40,000,000 | 256 | MONET-real 部分 |
| synthetic 锚池 | `$S3/t5_manifest/synthetic_v1/monet_synth/*.parquet` | 13,808,747 | 64 | 1024px 生成图 |
| real 汇总 | `$S3/t5_manifest/rich_v1/summary.json` | — | — | 行数/缺失/schema/tier 分布 |
| real QC | `$S3/t5_manifest/rich_v1/rich_v1_qc.json` | — | — | 分布 + caption 供给审计 |
| 权重 dry-run | `$S3/t5_manifest/rich_v1/sampling_weights_check.json` | — | — | 各 stage 期望分布 |
| synthetic 汇总 | `$S3/t5_manifest/synthetic_v1/summary.json` | — | — | |

### 1.2 选样中间产物 / 画像
| 产物 | 路径 |
|---|---|
| GPIC 选中明细 | `$S3/t5_selection/gpic_selected/*.parquet` + `gpic_selection_summary.json` |
| MONET 选中明细 | `$S3/t5_selection/monet_selected/*.parquet` + `monet_selection_summary.json` |
| MONET 各源质量画像 | `$S3/t5_inputs/monet_subsource_breakdown.json` |

### 1.3 源数据（指针模型回源读图/caption 用）

> ⚠️ **图像必须从 `.tar`（webdataset 全分辨率）读，不是 parquet 的 `thumbnail`**（已查证）：`jasperai/monet` 是 LAION 式数据集，parquet 里的 `thumbnail` 只是 **≤384px 预览图**，**全分辨率 JPEG 在同目录的 `.tar` 分片里**（每个 `.parquet` 都有一个同名 `.tar`，如 `cc12m/0-0/000000.tar`，已在挂载上）。已实测：`.tar` 成员名 `{id}.jpg/.json/.embedding_*.pth`，按 manifest `id` 取出的全分辨率图**尺寸恰好等于 manifest 的 width/height**（即 `width/height` 是真实可用像素，非仅元数据）。`.tar` 还带每图的 SANA-VAE latent。**用 parquet 的 thumbnail 训练 = 只有 384px，会毁掉 512/1024 阶段。**

| 数据集 | 根路径 | 图像（全分辨率）| caption |
|---|---|---|---|
| GPIC | `/s3-mewtant-raw-datasets/stanford-vision-lab/gpic_umziped` | `images/train/{src_shard}/{src_key}.{ext}`（真实文件、全分辨率）。**⚠️ 扩展名混合(~88% .jpg + ~12% .png)，不要硬编码 .jpg**——权威路径是 `parquets/gpic_train_{src_shard}.parquet` 的 `s3key` 列（含正确扩展名；loader 读 caption 时同一行就能拿到）| `parquets/gpic_train_{src_shard}.parquet` 的 `caption` 列（按 `key` 匹配；`caption_type`∈tag/short/medium/long）|
| MONET real + synthetic | `/s3-mewtant-raw-datasets/jasperai/monet/v1.1.0` | **`{src_root}/{src_shard 把 .parquet 换成 .tar}` 里的成员 `{src_key}.jpg`**（webdataset 全分辨率）| `{src_root}/{src_shard}`(.parquet) 的 `caption_{original,florence-2-large,internvl-3-8b,sharegpt4v-7b,gemini-2.5-flash-lite}` 列，按 `id` 匹配 |

说明：MONET 顶层源 `cc12m/ commoncatalog-cc-by/ coyo/ diffusion-aesthetic-4k/ laion/ megalith10m/ synthetic/`；`src_key == id == __key__`（各源已抽验一致）；`src_shard` 含源名前缀（如 `cc12m/0-0/000000.parquet`、`synthetic/flux-klein/...`）。da4k/synthetic 无 `url` 但 `.tar` 里有全分辨率图，故不影响。

### 1.4 代码（`/fsx/evan/mot_code/data_analysis/`）
| 文件 | 作用 |
|---|---|
| `scripts/t5_build_rich_manifest.py` | 建 real 富 manifest（SEL=rich） |
| `scripts/t5_build_synthetic_manifest.py` | 建 synthetic 锚池 manifest（SEL=syn） |
| `scripts/t5_gpic_select.py` / `t5_monet_select.py` / `t5_merge_manifest.py` | 选样到 20M / 40M / 合并 |
| `scripts/t5_qc_audit.py` | 富 manifest QC + caption 供给审计（SEL=qc） |
| `scripts/t5_sampling_weights.py` | **采样权重参考实现 + dry-run 校验**（SEL=wcheck） |
| `configs/t5_sampling_weights_v1.json` | **采样权重机器可读配置（含 loader_rules）** |
| `configs/t5_sampling_daily_v1.json` | **daily_v1 21M 日常探索版采样配置**（含 1024 阶段 band 供给提醒） |
| `cluster/run_t5_select_in_pod.sh` + `cluster/t5_*_job.yaml` | jcc 任务入口与定义 |
| `cluster/README_T2_NEXT.md` | 处理流水账（两轮审计 + 全部决策） |
| `../../doc/mot/采样权重spec_v1_20260616.md` | **采样权重 spec（人读，§8.5 含 4 条 loader 规则）** |
| `../../doc/mot/日常探索版筛选spec_v1_20260617.md` | **daily_v1 21M 日常探索版筛选 spec**（高质量多样子集） |

---

## 2. 关键指标与数据

### 2.1 real 池构成（60M）
| 数据集 | 选中 | ≥512(短边) | 角色 |
|---|---|---|---|
| GPIC | 20,000,000 | 4,239,912（21%）| 256 真实摄影多样性 + 干净短/tag caption |
| MONET-real | 40,000,000 | 31,995,551（80%）| 主力底子；512/1024 高清主要来源 |
| **合计** | **60,000,000** | **36,235,463（60.4%）** | |

**MONET 各子源（选中量）**：coyo 13,562,099 ｜ laion 13,562,098 ｜ commoncatalog-cc-by 5,497,680 ｜ cc12m 3,945,126 ｜ megalith10m 3,421,309 ｜ diffusion-aesthetic-4k 11,688。
> laion/coyo 触顶（每源 cap 13,562,098）；commoncatalog/cc12m/megalith/da4k 全保（在 cap 下）。爬虫源(laion+coyo)占 MONET 68% 是池子构成决定，靠采样权重降权（§5 源 α）。

### 2.2 美学（aes）
| 维度 | scorer | 均值 | p1 | p50 | p99 | 备注 |
|---|---|---|---|---|---|---|
| GPIC | laion_improved | 5.32 | 4.44 | 5.34 | 6.26 | 已切底 10%（阈值 4.370）|
| MONET | jasperai | 6.05 | 5.04 | 6.08 | 7.35 | jasperai 已策展；5,468 行 NaN(0.0137%，集中 da4k) |

### 2.3 分辨率
≥512 共 36,235,463（60.4%）。MONET 80% ≥512（高清主力），GPIC 仅 21% ≥512。synthetic 100% ≥1024px。

### 2.4 聚类（联合 CLIP-B/32 球面 k-means，k=20000）
covered 19,978/20,000（22 空簇）；簇 size：min 1 / p50 2,215 / mean 3,003 / p99 11,537 / max 27,050（最大簇仅占池 0.045%，无病态）。

### 2.5 内容类目（T4b 簇级 dominant_cat，real 60M）
render/3d **15.52M（25.9%，最大）** ｜ landscape/nature 7.46M ｜ person 7.45M ｜ building/city 7.17M ｜ food 5.62M ｜ illustration/art 3.83M ｜ text-rich 3.26M ｜ white-bg/product 3.23M ｜ animal 2.89M ｜ wedding/bride 1.97M ｜ car/vehicle 1.60M …
> render/3d 偏高，靠簇温和拉平间接控制（不单设类目目标，Evan 定）。

### 2.6 caption 档位供给 vs 目标 18/27/15/40（核心可行性）
档位边界（Qwen3 tok）：T1 ≤24 / T2 25-64 / T3 65-128 / T4 ≥129。`供给(档X)=GPIC tier==X + MONET has_tX`。`过采样系数 = 目标占比×池 / 供给`，**全 <1 = 供给过剩、取子集即可、无需重采样**：

| 档 | 目标 | 全 res 供给 / 系数 | ≥512 供给 / 系数 |
|---|---|---|---|
| T1 | 18% | 33.28M / 0.32 | 21.5M / 0.30 |
| T2 | 27% | 39.69M / 0.41 | 24.7M / 0.40 |
| T3 | 15% | 38.40M / 0.23 | 30.3M / 0.18 |
| T4 | 40% | 33.51M / **0.72** | 25.3M / **0.57** |

长档（T3/T4）几乎全靠 MONET 多 captioner：≥512 下 T4 = GPIC 0.38M + **MONET 24.9M**。
**GPIC 自带 caption 档分布**（i1 单条）：T1 6,102,960 / T2 11,473,931 / T3 631,543 / T4 1,791,566（严重偏短，长档靠 MONET）。
**MONET captioner 覆盖**：florence 99.4% / sharegpt4v 97.4% / internvl 89% / gemini 78% / original 78%。

### 2.7 synthetic 1024 锚池（13,808,747）
z-image 5,889,682 ｜ flux-schnell 4,413,058 ｜ flux-klein 3,506,007。**100% ≥1024px**；caption 为生成 prompt（多落中档：has_t2 53% / t3 28% / t1 12% / t4 6%）；aes_jasperai 全有（NaN=0）；含 6,565 holdout 标记（loader 排除，见 §6）；未进联合聚类故 `cluster_id=-1`。

### 2.8 去重（已验证）
| 路 | 删除量 | 方法 / 验证 |
|---|---|---|
| pHash 精确（GPIC↔MONET）| 计入 GPIC 8.02M 联合 | 感知哈希精确撞图 |
| SSCD 跨源（GPIC 侧删）| **8,002,184**（GPIC pass 的 8.08%）| 8 卡暴力精确，独立 fp32 抽检 **recall=1.0**、drop cos 中位 0.9966 |
| MONET 内部近重 | **3,132,196**（候选的 44%）| cuVS 单卡，连通分量留一，暴力抽检召回 ≈99.5% |

---

## 3. Manifest schema（33 列）

`rich_v1` 与 `synthetic_v1` **同 schema、可直接 union**（用 `dataset`/`synthetic` 区分）。列与语义：

| 列 | 类型 | 含义 / 取值 |
|---|---|---|
| `uid` | str | 全局唯一：`gpic:{key}` / `monet:{id}`（synthetic 也是 `monet:{id}`）|
| `dataset` | str | `gpic` / `monet` / `monet_synth` |
| `sub_source` | str | gpic；MONET：laion/coyo/commoncatalog-cc-by/cc12m/megalith10m/diffusion-aesthetic-4k；synthetic：synthetic-{flux-klein,flux-schnell,z-image} |
| `src_root` | str | 源根（回源用，见 §1.3）|
| `src_shard` | str | 源内相对路径 |
| `src_key` | str | 源内键（GPIC=__key__；MONET=id）|
| `width`/`height` | int32 | 原图宽高（loader 据此分桶，含 1024 细分）|
| `res_bucket` | str | `ge512`（短边≥512）/ `256`（短边 [256,512)）|
| `aes` | float32 | 美学分（GPIC=laion_improved；MONET/synthetic=jasperai）。**可能 NaN**（da4k 等，0.0137%）|
| `aes_scorer` | str | `laion_improved` / `jasperai` |
| `cluster_id` | int32 | 0..19999；synthetic = **-1**（未聚类）|
| `dominant_cat` | str | 簇内容标签（T4b）；synthetic/未知 = `unknown` |
| `caption_type` | str | GPIC：tag/short/medium/long；MONET：`monet`；synthetic：`synthetic-prompt` |
| `caption_tier` | int8 | GPIC：1-4（其 i1 caption 所属档）；MONET/synthetic：**0**（用 has_tX）|
| `has_t1`..`has_t4` | int8 | 该图**能否服务**对应档（=任一 captioner 落该档）|
| `cap_i1` | int8 | 是否有 GPIC i1 caption（GPIC=1）|
| `cap_original`/`cap_florence`/`cap_internvl`/`cap_sharegpt4v`/`cap_gemini` | int8 | MONET 各 captioner 是否存在（tok>0）|
| `tok_i1` | int32 | GPIC i1 caption token 数（MONET/synthetic=0）|
| `tok_original`/`tok_florence`/`tok_internvl`/`tok_sharegpt4v`/`tok_gemini` | int32 | MONET 各 captioner token 数（GPIC=0）。**band→captioner 选取靠它**（见 §6）|
| `synthetic` | int8 | 0=real；1=生成图（仅 synthetic_v1）|
| `holdout` | int8 | 1=评测 holdout。real 已排净=0；**synthetic_v1 仍有 6,565 个 1**（loader 排除，见 §6）|

> token 计数语义：`tok = 词数×1.32` 代理（误差 <2%）。**已发布 rich_v1 的 tok_* 是早期 int16，有 18 行因超长垃圾 caption 溢出成负**——loader 见 `tok<=0` 当不可用（脚本已改 int32，未来重建即无）。

---

## 4. 决策日志（谁定 / 依据）

| # | 决策 | 取值 | 依据 |
|---|---|---|---|
| 1 | MONET:GPIC 配比 | **20M:40M = 60M real**（MONET 主）| Evan：GPIC 分布略广、交集小；落 v3 doc 60-80M 区间 |
| 2 | caption 档配比（real）| **T1 18 / T2 27 / T3 15 / T4 40** | v3 doc 正本（取代旧 10/30/40/20）|
| 3 | synthetic | **排除出 real**，单列 1024 锚池 | Evan：40M 不含文生图生成图；synthetic=`synthetic/` 路径(flux/z-image) |
| 4 | holdout | **全局排除**（real+synthetic 都不进训）| Evan（审计后）；md5(id)%2098==0 |
| 5 | MONET 子源 cap | water-fill 40M（每源 ≤13.56M）| Evan 接受默认；干净源全保 + laion/coyo 触顶 |
| 6 | GPIC 美学切底 | 删最低 **10%**（laion_improved，阈值 4.370）| v3 doc |
| 7 | 源拉平 `source_alpha` | **0.5**（laion+coyo 自然 57%→46.6%）| Evan；α=1 病态(da4k 微源被抬到 14%) |
| 8 | 簇拉平 `cluster_beta` | **0.5**（~1/√size 温和）| Evan |
| 9 | aes 加权 | **关**（选样已切底）| Evan |
| 10 | MONET:GPIC @256 采样 | 母池/daily v1 线 **80:20**（v3）；**【2026-06-17 修订】complete_v1 stage-256 改用 11:7**（= daily 静态池 11M:7M 自然比，monet_real 0.397 / gpic 0.253），取代 80:20，见 §11.5 | v3 doc → Evan 改 |
| 11 | GPIC @512 | ~~0（退场）~~ → **0.08**（Evan 2026-06-17 改：512 样本偏轻 51.2M，加 GPIC 补真实图数据量；real 0.31→monet 0.23/gpic 0.08）。**@1024 仍 0**（GPIC elig_1024=0、无 ≥1024px 图，物理不可行）| Evan（先审计后退场、后 06-17 加回 512） |
| 12 | caption band→captioner | T4→gemini / 短→florence / megalith T2→florence 首句；**取不到 fallback 任意落档** | v3 doc + Evan（审计后）|
| 13 | caption dropout / 截断 | 全程 10% / 长档 ~300-340 proxy(真 token budget 343) 句界截 | v3 doc（2026-06-18 上调，见实现现状指南） |
| 14 | da4k(11,688) | **保留为 real**（非合成）| 查证：HF `zhang0jhon/Aesthetic-4K`，人工精选真实超高清照片，GPT-4o 仅写 caption |
| 15 | TextAtlas / anime | TextAtlas 本轮不做；**anime 已集成**（团队交付池 16.5M → 薄 manifest + complete_v1，见 §11；决策 A1–A9 在 §11.4）| Evan |

---

## 5. 采样权重（real 池内 + 分辨率路由）

详见 [采样权重spec_v1](采样权重spec_v1_20260616.md) 与 `configs/t5_sampling_weights_v1.json`。要点：

- **范围**：本权重只管 **real 池内（MONET-real + GPIC）** 的 intra 权重 + 分辨率路由；跨域大盘（real:anime:synthetic:TextAtlas）是**上层 mixer**（real 槽已就绪，anime/TextAtlas 待接）。
- **band-first 抽样**（精确命中 18/27/15/40）：抽档 b ~ 18/27/15/40 → 抽域（256 阶段 MONET:GPIC=80:20）→ 域内按 `source_factor(N_src^-0.5)×cluster_factor(size^-0.5)` 抽图（**不按 aes**）→ 按 §6 取 caption → 10% dropout。
- **分辨率路由（256→512→1024）**：256 全可用（ge512 可降采）；512 = MONET ge512 + **GPIC ge512（~1.15M，Evan 2026-06-17 加回）**；1024 仅短边≥1024（主要 commoncatalog）+ synthetic 锚池（上层注入）；**1024 无 GPIC**（GPIC 全部 <1024px）。
- **dry-run 实证**（`sampling_weights_check.json`）：band 配比精确命中、256 域 80:20、1024 commoncatalog 主导 57%、簇拉平后最大簇 0.019%、源 α=0.5→laion+coyo 46.6%。

**跨域上层 mixer —— 权威配比见外部 config / 新文档，本节不再维护实数**：

> ⚠️ **跨域 domain_mix 的权威来源已外移，不要用本节的历史骨架开训。** 演进：
> - v1 占位骨架（anime/TextAtlas 当时未接，仅作设计参考）：256/512/1024 anime = 0.20/0.55/0.70。
> - complete_v1（v2，4 域）：anime = 0.30/0.65/0.80（无 TextAtlas），见 §11.5 + `configs/t5_sampling_complete_v1.json`。
> - **complete_v1（2026-06-27，5 域，当前权威）**：加入 TextAtlas 注入域，配比改为 256 = anime 0.27 / monet_real 0.357 / gpic 0.228 / synth 0.045 / text 0.10；512 = 0.65 / 0.13 / 0.08 / 0.04 / 0.10；1024 = 0.80 / 0.08 / 0 / 0.12 / 0。**权威 config = `t5_sampling_complete_v1_textatlas.json`，完整设计 + mix-fix 见 [complete_v1_dataline_设计与变更_v1_20260627.md](complete_v1_dataline_设计与变更_v1_20260627.md)。**
> - real 池内（MONET:GPIC）配比：stage-256 由早期 80:20 改为 11:7（静态池 11M:7M 自然比），见 §11.5 / 决策 #10。
>
> 本节其余要点（band-first 18/27/15/40、源/簇拉平、分辨率路由）仍是 real 池内权威，未变。

---

## 6. 如何使用（loader 集成）

### 6.1 指针模型读取
manifest **不含图与 caption 文本**。loader 按 `src_root`/`src_shard`/`src_key` 回源。**⚠️ 图像从 `.tar`（全分辨率）读、不是 parquet 的 384px `thumbnail`**（MONET/synth：`src_shard` .parquet→.tar，成员 `{id}.jpg`；GPIC：`images/train/{src_shard}/{src_key}.jpg`）——见 §1.3。caption 从 parquet 列读（§1.3）。降采样用 Lanczos/bicubic+抗锯齿；**不上采样低分图**；长宽比分桶。

### 6.2 ⚠️ **必须实现**的 loader 规则（两轮审计后定，数字均在已发布 parquet 上验证过）
1. **holdout 全局排除**：训练前 `filter(holdout==0)`，所有池。real 已=0；**synthetic_v1 有 6,565 个 holdout==1 必须过滤掉**。
2. **band 合格性以"tok>0 且落档"为准,不是 has_tX/cap_***：一行只有在"存在某 captioner 的 tok 落在该档且 >0"时才算该档合格。**已验证 13 行 `has_t4=1` 却无任何正 T4 token**（唯一 T4 captioner gemini 因 int16 溢出成负）→ 这些行**不进 T4**。
3. **caption fallback**：按档选 captioner，优先级（T4: gemini→sharegpt4v→internvl；T3: internvl→sharegpt4v→gemini；T2: florence→original，megalith 用 florence 首句；T1: original，干净 tag 优先用 GPIC）是**软偏好**；都取不到时 fallback 到任意"tok>0 且落该档"的 captioner。
4. **token ≤0 视为该 captioner 不可用**：已发布 rich_v1 有 18 个负 token（17 gemini + 1 original，int16 溢出遗留），`tok<=0` 跳过该 captioner（其中 13 行因此无可用 T4，见 §6.2.2）。
5. **GPIC@1024 = 0；GPIC@512 = 0.08**（Evan 2026-06-17 更新，原为 512/1024 都 0）：**@1024 不抽 GPIC**（`monet_real_gpic_ratio.1024=[1,0]`；GPIC elig_1024=0、无 ≥1024px 图）；**@512 抽 GPIC**（`monet_real_gpic_ratio.512=[0.742,0.258]`、`resolution_routing.512.gpic=elig_512==1`、~1.15M 池）。loader 以 `domain_mix`/`monet_real_gpic_ratio` 为准。
6. **aes 缺失（null 或 NaN）当缺失**：MONET real 有 5,468 个。本方案不按 aes 加权 → 无影响；他处用到给中位兜底或排除。
7. **source/cluster 因子用全局 60M 统计**：`N_source`、`cluster_size` 全池各算一次,不要 per-shard/band/stage 子集重算。
8. **`domain_mix` 未归一**（256 之和 0.98，anime/TextAtlas 待接）→ loader 显式归一后再用。

### 6.3 取 caption 文本
抽到档 b 后：GPIC 用其 i1 caption；MONET 用 §6.2 优先级选中的 captioner 文本（回源读对应 `caption_*` 字段）；synthetic 用生成 prompt。长档按句子边界截 ~300-340 proxy(真 token budget 343)（**别截 150**，gemini 中位 179）；internvl 删末尾残句。含文字/海报样本调低或关 dropout。

### 6.4 接入权重
loader 读 `configs/t5_sampling_weights_v1.json`，按 `t5_sampling_weights.py` 的 `base_weights`（源/簇因子）+ band-first 流程构建 per-(stage,band,domain) sampler。`t5_sampling_weights.py` 既是参考实现，也可 `SEL=wcheck` 跑期望分布自检。

---

## 7. Pipeline 溯源（T1–T5）

| 阶段 | 脚本 | 产物 | 说明 |
|---|---|---|---|
| T1 GPIC 特征 | `t1_gpic_pass*.py` | `$S3/gpic_pass`, `gpic_manifest_base` | 基础过滤(短≥256/AR≤3/排 test-val)、SSCD/CLIP-B/aes/pHash 补算 |
| T2 去重 | `t2_phash_exact_cross_source.py`, `t2_sscd_bruteforce_sharded.py`(+`_validate_groundtruth`), `t2_monet_internal_cuvs_dedup.py` | `t2_phash_exact`, `t2_sscd_bruteforce`, `t2_monet_internal_cuvs_single_full` | pHash + SSCD 跨源(8卡暴力,删 GPIC 侧 8.0M) + MONET 内部(3.13M) |
| T3 MONET manifest | `t3_monet_sweep.py` | `$S3/monet_manifest`（10,881 shard）| 元数据 sweep、各 captioner token/档、keep/synthetic/holdout 标记 |
| T4 联合聚类 | `t4_clipb_joint_kmeans.py` | `$S3/t4_clipb_joint_kmeans/{centroids,gpic_clusters,monet_clusters}` | CLIP-B/32 球面 k-means k=20000，MONET-real 77M + GPIC 99M 全分配 |
| T4b 内容标签 | `t4b_cluster_labels.py` | `t4_clipb_joint_kmeans/cluster_content_labels.parquet` | 簇成员投票零样本标签 |
| T5 选样 | `t5_gpic_select.py` / `t5_monet_select.py` | `t5_selection/{gpic,monet}_selected` | GPIC→20M（短≥512全保+簇 water-fill+aes优先）；MONET→40M（子源 cap+aes 优先，排 synthetic+holdout）|
| T5 富 manifest | `t5_build_rich_manifest.py` | `t5_manifest/rich_v1` | join 定位符/分辨率/aes/cluster/标签/caption 粒度 |
| T5 synthetic | `t5_build_synthetic_manifest.py` | `t5_manifest/synthetic_v1` | keep&synthetic（未来重建排 holdout）|
| T5 QC/权重 | `t5_qc_audit.py`, `t5_sampling_weights.py` | `rich_v1_qc.json`, `sampling_weights_check.json` | 分布审计 + 权重 dry-run |

> 全程 jcc CPU-only（`gpuPerWorker:0`，T1/T2 GPU 阶段除外）。关键经验：pandas `.isin(大set)` 致命慢 → 用 `pyarrow pc.is_in` 全局判一次；mountpoint-s3 不持久空目录 → 写前 makedirs；跨 shard parquet schema 异构 → 统一类型再 concat。详见 README_T2_NEXT.md。

---

## 8. 复现 / 重建

jcc 任务（job 名须 `evan-` 前缀）：`jcc job submit cluster/t5_<name>_job.yaml`。`run_t5_select_in_pod.sh` 按 `SEL` 分发：`gpic`/`monet`/`merge`/`breakdown`/`rich`/`syn`/`qc`/`wcheck`。Python：`/fsx/evan/mot_code/venv-gpu/bin/python`（pandas 3.0.3 / pyarrow 24.0.0 / numpy 2.4.4）。镜像 `harbor.jh1.mewtant.io/public/autobuild:pytorch-cu128`。

**若要 pristine 重建**（消除当前 rich_v1 的 18 负 token、synthetic_v1 的 6565 holdout）：脚本已就绪（int32+clip、synthetic 加 `& holdout==0`），重交 `t5_rich_manifest_job.yaml`（~16min）与 `t5_synthetic_manifest_job.yaml`（~3min）即可。建议写到 `*_tmp_$job/` 校验后再切换（原子发布，防半新半旧）。**当前不重建**（Evan 定），靠 §6 loader 规则兜。

---

## 9. 已知 caveat / 未尽事项

1. **当前 manifest 不重建** → rich_v1 留 18 负 token、synthetic_v1 留 6,565 holdout，**靠 §6 loader 规则兜**（必带）。
2. **daily_v1 21M 子集的 1024 阶段不能默认无重采样沿用 18/27/15/40**：2026-06-17 审计显示 daily_v1 1024 可用池 5,049,529，T1/T4 oversample_factor=**1.202 / 1.949**。这是日常探索版小池约束，不推翻母池 rich_v1 供给结论；若用 daily_v1 开 1024，训练配置必须显式处理。
3. **跨域 mixer**：anime 已填实数（complete_v1 4-way domain_mix，见 §11.5）；TextAtlas 仍缓；real 槽就绪。
4. **band-first 下多档图（MONET 多为 2-4 档）总曝光略高于单档图**；v1 不做逐图曝光约束。
5. **gemini 覆盖 78%**（v3 doc 写~90%，实测 78%）——仍足够撑 T4。
6. **e2e dry-run 待补**：loader 实现后跑真实百万抽样，统计 domain/stage/band/captioner 实际取数失败率（审计建议）。
7. **TextAtlas** 本轮不做；**anime** 团队已交付并集成（见 §11）。
8. holdout eval 池如需可另出（synthetic_v1 已带 holdout 标记；real 的 holdout 在 monet_manifest 按 `md5(id)%2098==0`）。

---

## 10. 验证与可信度

- 选样后核验：60M、0 重复 uid、real holdout 排净、各源量与 cap 一致。
- 富 manifest QC：`missing_meta=0`、`missing_cluster=0`、schema 统一、抽检行字段正确。
- 权重 dry-run：band 精确、域 80:20、1024 commoncatalog 主导、簇拉平生效。
- 去重独立验证：SSCD recall=1.0（fp32 抽检）、MONET 内部召回 ≈99.5%。
- **三轮独立 AI 审计**（只读）+ 逐项独立复核：无硬冲突、与 v3 doc 一致；发现的边界问题（caption 粒度、synthetic 池、holdout、int16、512 口径、fallback、band 合格性）均已闭合或定为 loader 规则。
- **第 3 轮审计声称已逐项在已发布 parquet 上验证**（job `evan-mot-t5-verify-audit2`，只读）：tok_* 仍 int16 ✓；MONET 负 token 18（17 gemini+1 original）✓；**has_t4=1 但无正 T4 captioner = 13** ✓；aes 缺失 5,468 ✓；GPIC tok_i1 负值 0 ✓；synthetic holdout 6,565 ✓。据此把"band 合格性以 tok>0 为准（非 has_tX）"补进 §6.2，并修正 spec/config 的 GPIC@512 口径冲突、aes 缺失、全局统计、domain_mix 归一等。

---

# 【v2】§11. anime 交付 + complete_v1 集成（2026-06-17 新增）

> 团队交付的 anime 池 `anime_core_ditv04_ct_soft` 已建**薄 pointer manifest** 并入 **complete_v1 = daily_v1(real+synth 21M) + anime 16.5M**。anime 是**目标域**。本节按上文同标准记录交付物/指标/schema/决策/采样集成/loader/caveat/验证。**daily_v1 三域(gpic/monet/monet_synth) 与 60M 母池均未改动。** 执行依据：[complete_v1_anime集成_执行文档_20260617.md](complete_v1_anime集成_执行文档_20260617.md) + 两份 v3 设计正本。

## 11.0 TL;DR（anime）
- **薄 manifest**：`$S3/t5_manifest/daily_v1/anime/anime_*.parquet`（64 shard，**16,499,287 行**，25 列 pointer）。**全量放入、不过滤、NSFW 全保留**；每-epoch 曝光由 domain_mix + ct_weight 收到 ~10–12M-equiv（**静态可用 ≠ 每 epoch 曝光**，不静态裁）。
- **caption = 指针**：caption 文本 / 原始 booru tags **不落 manifest**，训练时按 `key` 回源 `/s3/evan/anime_core_ditv04_ct_soft/data` 取（naifu caption builder 同款）。
- **图像 = R2 HTTP**：`s3key` = `https://ncotar-worker.mewtant.workers.dev/r2/<key>`，**不在本地挂载**；训练侧用 naifu `training/r2_utils.py`。real/synth 仍读本地 `.tar`/parquet。
- **统一采样 config**：`configs/t5_sampling_complete_v1.json`（4-way domain_mix + anime 三档 caption + ct_weight 优先级 + lane caps；real/synth 机制沿用 daily_v1）。
- **QC 全绿**：`$S3/t5_manifest/daily_v1/complete_v1_qc.json`；独立 AI 审计确认数据侧 sound。
- ✅ **训练侧已闭环、V1 可开训**（2026-06-17，见 §11.6）：真实 loader（`training/dataline/CompleteV1Dataset`）+ caption/tags 回源 + R2 取图 + naifu 扩展（ct_weight-base / integer-repeat / 分 stage route / composition_clean 真值谓词）全实现，真数据+真模型+真VAE e2e 跑通；configs 256/512/1024 已接（不再 stub）。剩：anime VAE gate 复核（#7）、全量 scaling（#8）。

## 11.1 交付物清单与路径（anime）
| 产物 | 路径 | 行数 | shard | 说明 |
|---|---|---|---|---|
| anime 薄 manifest | `$S3/t5_manifest/daily_v1/anime/anime_*.parquet` | 16,499,287 | 64 | 25 列 pointer（§11.3）|
| anime 构建汇总 | `$S3/t5_manifest/daily_v1/anime/anime_manifest_summary.json` | — | — | 行数/elig/分布 |
| complete_v1 QC | `$S3/t5_manifest/daily_v1/complete_v1_qc.json` | — | — | 完整性/分布/stage 池/1024 供给 |
| 统一采样 config | `data_analysis/configs/t5_sampling_complete_v1.json` | — | — | 4-way mixer + anime caption/采样 |
| 构建脚本 | `data_analysis/scripts/t5_build_anime_manifest.py` | — | — | dispatcher `SEL=anime` |
| QC 脚本 | `data_analysis/scripts/t5_complete_qc.py` | — | — | §11.7 |
| job | `cluster/t5_anime_manifest_job.yaml` / `t5_complete_qc_job.yaml` | — | — | gpu0 / cpu32 / mem256 |

**源数据（回源读图/caption）**：
| 数据集 | 根路径 | 图像（全分辨率）| caption / tags |
|---|---|---|---|
| anime | `/s3/evan/anime_core_ditv04_ct_soft/data`（⚠️ `/s3/` **非** `/s3-pixai-test-uw2/`，两个不同 mountpoint-s3 桶，pod 均已挂）| **R2 HTTP**：`s3key` worker URL（不在本地挂载；naifu r2_utils）| 源 parquet 按 `key`：`refined/medium/short_description` + `tags_{general,character,copyright,artist,meta}` + rating/quality_z/hps_v3/ct_* |

## 11.2 关键指标（16,499,287 行，QC 实测）
- **rating**：general 46.1% / sensitive 38.6% / questionable 8.06% / explicit 7.23%（q+e≈15%，**NSFW 全保留**）。`is_non_ai` 77.1%。
- **分辨率**：≥512 **16,294,879（98.76%）**、≥1024 **10,991,143（66.62%）**、<512 仅 1.24%。
- **caption 可用**：`has_nl`=`has_short`=**11,699,792（70.91%）**；**tag-only 4,799,495（29.09%，走纯 tag route 不丢）**。≥1024 中带 NL = 8,171,900（74.35%，足撑 1024 NL 路由 40%）。`n_tags_general` p50=44。
- **质量**：`ct_weight` p5/p50/p95 = 1.45/1.75/2.18（mean 1.76，推荐训练权重）；`ct_filter_score` p50 0.757（软通过参考）。
- **多样性轴 `cluster_clipl14_level2`**：**4,863 唯一簇**，最大簇 0.53% / top10 4.27%（无病态）。
- **源（`dataset`，18 源）**：danbooru 31.3% / pixiv 26.5% / twitter 10.5% / sankaku 8.0% / nozomi 4.2% / …
- **路由桶 `risk_bucket`**：plain_illustration 82.3%；**lane-cap 6 桶合计 11.6%**（text_layout 5.4% / photoish 3.0% / comic_text 2.2% / meme 0.7% / photoish_3d 0.3% / game_ui 0.05%）。`cls_bucket` illustration 86.6%；`person_bucket` solo 76.0%。

## 11.3 anime 薄 manifest schema（25 列，**与 rich_v1 的 33 列不同，不可直接 union**）
| 列 | 类型 | 含义 / 取值 |
|---|---|---|
| `uid` | str | 全局唯一：`anime:{key}`（与 daily_v1 uid 风格一致，可区分/并查）|
| `key` | str | 源主键 → 回源取 caption/tags |
| `uuid` | str | 全局去重 id（100% 唯一）|
| `s3key` | str | **R2 worker URL** → 取图 |
| `dataset` | str | booru 源（18 值；分层/审计）|
| `width`/`height` | int64 | 原图宽高（分桶）|
| `elig_256`/`elig_512`/`elig_1024` | int8 | stage 路由：256 全 1；512=min(w,h)≥512；1024=min(w,h)≥1024 |
| `rating_bucket` | str | general/sensitive/questionable/explicit（条件标签 + 审计）|
| `is_non_ai` | bool | 审计 |
| `aes` | float32 | = `compound_aes_score`（质量参考，**不硬筛**）|
| `ct_quality_score` | float32 | 备用质量参考 |
| `ct_weight` | float32 | **采样权重**（1.0–2.5）|
| `ct_filter_score` | float32 | 软通过参考 |
| `ct_lanes` | list[str] | lane 路由/caps 成员（artist_canonical/concept_*/quality_completion…）|
| `cluster_balance` | int32 | = `cluster_clipl14_level2`，**多样性平衡主轴**（~5000 簇）|
| `style_l1` | str | 备用风格簇 |
| `risk_bucket` | str | lane caps（text_layout/comic_text/photoish/…）|
| `cls_bucket`/`person_bucket` | str | 分层/审计 |
| `has_nl` | int8 | naifu `_has_visual_nl`（primary NL 或结构化字段）→ caption route 表选择。**0/1 干净无 null**（tag-only 行 =0）|
| `has_short` | int8 | short_description 非空 → route 选择 |
| `n_tags_general` | int32 | `len(tags_general)` → tag route 可用性 |

> ⚠️ caption 文本字段（refined/medium/short/tags_*）**不在薄 manifest**，训练时按 `key` 从源 parquet 读（naifu caption builder 直接吃源字段）。`has_nl`≡`has_short`：本交付里 refined/medium/short 三者同 null、且结构化 NL ⊆ primary NL，故 `_has_visual_nl` = short 非空（构建时无需读长 refined）。

## 11.4 决策日志（anime / complete_v1）
| # | 决策 | 取值 | 依据 |
|---|---|---|---|
| A1 | 规模 | 探索版 ~30M：daily 21M + anime ~10–12M 有效曝光 | 团队群 + i1 多样性>总量 |
| A2 | anime 口径 | **manifest 放全 16.5M**；曝光由 domain_mix+ct_weight 收 | Evan：静态可用≠每 epoch 曝光 |
| A3 | NSFW | **全保留**（含 q/e），rating 作 caption 条件标签（非过滤）| Evan + naifu |
| A4 | domain_mix | 激进 anime 256/512/1024 = **0.30/0.65/0.80** | Evan |
| A5 | caption 三档（**平滑过渡**）| 256=NL长50/NL短20/hybrid20/tag10；512=35/20/20/25；1024=15/10/15/60 + tag 词频 dropout | v3 canon + Evan 2026-06-17 拍板 |
| A6 | 采样优先级 | `ct_weight` 为 base + naifu lane bonuses（cap 3.0 / repeat 4）| 执行文档 §8 + naifu sampling_plan |
| A7 | lane caps | `risk_bucket`∈{text_layout,comic_text,photoish,photoish_3d,meme,game_ui} 优先级 **×0.3**（软降权、不删）| Evan 2026-06-17 |
| A8 | 不静态选样 | 不做 real 式 Gumbel 选样（团队软校准为全量带权设计）| 执行文档 §8 |
| A9 | balance 轴 | `cluster_clipl14_level2`（~5000）| naifu sampling_plan |

## 11.5 采样集成（domain_mix 演进；当前权威 = 5 域）

> **【2026-06-27】complete_v1 现为 5 域**：加入 **TextAtlas**（第 5 注入域，像 synthetic 一样注入）。**当前权威 domain_mix 与第 5 域设计/mix-fix 见 [complete_v1_dataline_设计与变更_v1_20260627.md](complete_v1_dataline_设计与变更_v1_20260627.md) + config `t5_sampling_complete_v1_textatlas.json`，本节不再重复推导。** 5 域配比速查：256 = anime 0.27 / monet_real 0.357 / gpic 0.228 / synth 0.045 / text 0.10；512 = 0.65 / 0.13 / 0.08 / 0.04 / 0.10；1024 = 0.80 / 0.08 / 0 / 0.12 / 0（1024 textatlas 权重 0、plumbing 已就位）。
>
> 下表为 **4 域版（2026-06-17）历史沿革**，静态池数字（anime≥512/≥1024 等）仍准确、保留作分辨率供给参考；4 域→5 域的差异仅是 TextAtlas 注入 + real/synth 权重让渡。

| stage | anime | MONET-real | GPIC | synthetic | 静态池（QC 实测）|
|---|---:|---:|---:|---:|---|
| 256 | **0.30** | **0.397** | **0.253** | 0.05 | 37.50M（daily 21M + anime 16.499M）|
| 512 | **0.65** | 0.31 | 0 | 0.04 | 28.52M（anime≥512 16.29M + MONET≥512 9.23M + synth 3M）|
| 1024 | **0.80** | 0.08 | 0 | 0.12 | 16.04M（anime≥1024 **10.99M** + MONET≥1024 2.05M + synth 3M）|

- 各 stage sum=1.0；**256 内 MONET-real:GPIC = 11:7**（= daily 静态池 11M:7M 自然比 → monet_real 0.397 / gpic 0.253；**Evan 2026-06-17 拍板，取代早期 80:20**，让 GPIC 按真实池占比进 256、补真实摄影多样性）；**512 GPIC=0.08（Evan 2026-06-17 加回，monet:gpic=0.742:0.258）**；1024 GPIC=0（无 ≥1024px 图）。real/synth 域内仍沿用 §5 band-first 18/27/15/40。
- anime 域内 caption 见 A5；采样优先级见 A6；lane caps 见 A7。
- **1024 供给**：anime≥1024 **10.99M**（其中带 NL 8.17M → 海量 refined 撑 T4），大幅缓解 daily_v1 1024 档绑定；但 real/synth 那 0.20 切片仍 **T1/T4 binding**（QC `realsynth_1024_band_supply`：T1 1.202 / T4 1.949），训练 1024 时仍需 1024-specific band 处理（见 §9.2、config `stage_band_supply_warning`）。
- ⚠️ 注：§5 旧 mixer 占位（anime 0.20/0.55/0.70）已被本表取代。

## 11.6 ✅ 如何使用 / 训练侧实现状态（anime）
> **2026-06-17 更新（训练侧）**：complete_v1 真实 loader **已实现并端到端跑通**（`mot_code/mot_pilot_v1/training/dataline/`，4 域全忠实）——真数据+真 3.1B MoT+真 FLUX.2 VAE 在线 encode 训练 20 step loss 2.0→1.81、无 NaN、41GB，**V1 已能完全开训**；经 15-agent 对抗 review（真数据 verify）修 6+1 真 bug、全写测试验证。下列前置 **#1–#5 已闭环、#6 部分（e2e smoke+review 已做）、#7–#8 仍待**。
> 下文原"训练侧必须实现"清单保留作对照，逐条标注实现状态：
1. ✅ **真实 loader**：`training.dataline.complete_dataset.CompleteV1Dataset` 消费采样 config（anime 三档 caption + 优先级 + lane caps），configs 256/512/1024 已接（不再是 stub）。**【2026-06-27】config 已升级为 5 域 `t5_sampling_complete_v1_textatlas.json`（加 TextAtlas 注入域）；4 域版 `t5_sampling_complete_v1.json` 保留给旧 ablation。第 5 域 loader 细节见 [complete_v1_dataline_设计与变更_v1_20260627.md](complete_v1_dataline_设计与变更_v1_20260627.md)。**
2. ✅ **caption/tags 回源**：real/synth 按 `src_root/src_shard/src_key` 回源 parquet caption；**anime 直接以源 parquet（17 shard, 139 列）为 manifest 驱动**（自带 tags_*/descriptions/quality_z/ct_* 全字段，无需 join 薄 manifest）。
3. ✅ **R2 取图**：anime 图走 `https://img.metanomaly.co/r2/<s3key>`（urllib + 磁盘缓存，无需 dataio/creds）；real/synth 读本地 `.tar`（顺序流式）/文件。
4. ✅ **naifu 扩展全实现**：ct_weight-base 优先级 + **integer-repeat 曝光**（naifu weighted-budget extra_frac=0.25、按 priority-1 加权、封顶 max_repeat、每行≥1）+ 分 stage caption route（256/512/1024 不同 route 权重）；composition_clean 用 naifu 真值谓词（simple-bg + 无 watermark/sig + YOLO added_text<0.005 + qz≥0.4，真数据命中 20.6%=naifu，非 risk_bucket 近似）；freq dropout 用 naifu median 分母。
5. ✅ **NSFW/tag-only**：rating 仅 q/e 渲染；tag-only 行走纯 tag route 不丢（freq dropout 仅作用 NL 行）。
6. ⚠️→部分：**e2e smoke 已做**（4 域 + R2 拉图 + 回源 caption + 在线 VAE + 真模型训练跑通，review 验 domain_mix/route/priority/lane/repeat 各路径）；正式"caption 真实截断率"统计 + 全量 dry-run 仍可在开训前补。
> ⚠️ 训练侧发现并修复的数据消费坑（两轮对抗 review + 第二份独立审计）：
> - anime 源 shard **内部按 qz/cluster 有序**，读头部 nrows 严重偏低质量（shard0 头 20k 的 qz≥0.4 仅 15% vs 池 82%）→ loader 改**随机 row-group**取样（qz 回 97%）。
> - **caption 真实硬截断**（#9 关切的量化）：proxy（words×1.32）低估真实 Qwen token（×1.41）→ 模型端 max_length=296 静默 mid-token 硬截 anime@256=8% → dataset 传 tokenizer **真实 token 截断到 ≤255**（8%→0%）。**文本长度调研**：256 预算对齐 i1(256)/nano-T2I(224-256)，512 是 Qwen-Image 文字渲染才需、不必。（2026-06-18 更新：上述 296/≤255 为当时数值；text_max_length 已上调至 384、真 token budget 343，文本预算口径对标 Anima-2B/Qwen-Image/naifu 长 caption，见实现现状指南。）
> - **CFG dropout 重复**（loader+模型双 10%）→ 改模型单点 + tag-only anime `cfg_keep` 豁免（落实 #5『tag-only 永不 drop』）。
> - 首 batch 空等无 producer 的域 queue 30s（如 1024 的 GPIC——elig_1024=0 没数据但若误配 domain_mix>0）→ 修（_topup 只阻塞有 producer 的域）。
> - cluster_balance（balance_axis）：anime 池**已按 cluster 构造平衡**（QC top10 簇 4.27%），sampler 不重做（同 naifu）+ 加 cluster 曝光监控；max_repeat 流式 with-replacement 为软界（E[曝光]∝repeat 已忠实）。
> **原始"训练侧必须实现 #1–#9"清单已不再逐条复述**（其设计要求与 caveat 是 [complete_v1_anime集成_执行文档_20260617.md](complete_v1_anime集成_执行文档_20260617.md) §6–§8 + §12 的镜像；执行细则以该文为权威）。#1–#6（loader / caption-tags 回源 / R2 取图 / naifu 扩展 / NSFW-tag-only / e2e）状态见上方"✅ 已实现"列。**仍未尽的项**：
> - **#7 VAE gate（anime）**：v3 架构文档要求先验 FLUX.2 VAE 对 anime 线条/网点/透明重建（LPIPS/rFID + 人工）；anime 域偏移**未单独测**，上生成塔前复核（训练侧）。
> - **#8 探索 vs 全量 scaling**：本轮探索 ~30M；全量 ~90M（full rich_v1 60M + full synth 13.8M + anime full）是后续 scaling，同框架改 domain_mix/规模。`ct_*` 为 cell 内相对值，勿跨源/跨 level 硬比。
> - **#9 caption 真实截断率统计**：`text_max_length=384`（≈模板 ~41 tok + caption budget ~343），超长 tokenizer 硬截、整条不丢；loader 已按句/tag 边界预截，开训前补全量真实截断率统计（real/synth/anime 同适用）。

## 11.7 验证与可信度（anime）
- **构建 + QC**（job `evan-mot-t5-anime-manifest-g5wj` / `evan-mot-t5-complete-qc-q8kg`，均 Succeeded）：16,499,287 行、`uuid` 全唯一、`elig_256` 全 1；rating/源/`is_non_ai`/分辨率与团队交付 README **完全一致**；`has_nl`=11,699,792 正好落在文档记的 captioned 子集（70.9%）。
- **独立 AI 审计（只读）**：`has_nl` 抽 ~685 万行与 naifu `_has_visual_nl` 逐行差异 **0**（含 metadata 不对齐的 row group 解码后亦 0）；manifest 不变量（uid/s3key/elig/has_*/n_tags/ct_*）**0 异常**；R2 取图抽样 **10/10 成功**。审计结论"数据侧基本 ready、训练侧不能判 ready"，与 §11.6 一致。
- **构建期修过 1 个 bug**：`has_nl`/`has_short` 对 tag-only 行须为 **0**（原误为 null，会让"按 `has_nl==0` 识别 tag-only"的 loader 漏掉这 29%）→ 已修 `nonempty()` fill_null、重建 + 重 QC 确认干净。
- **审计修正项（已并入）**：config 注释 `anime≥1024` 8.1M→**10.99M**（8.1M 实为带 NL 子集 8.17M）、1024 池 13.1M→**16.04M**；并加"本 config 是对 naifu 的扩展、训练侧须实现回源/ct_weight-base/分 stage route"说明。功能数字（domain_mix 配比）不受影响。
