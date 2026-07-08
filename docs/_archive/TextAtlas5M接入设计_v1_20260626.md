> 已归档 2026-06-27：内容已并入 docs/complete_v1_dataline_设计与变更_v1_20260627.md。

# TextAtlas5M 接入 complete_v1 数据线设计(第 5 个域 · caption 用 i1)

**日期**: 2026-06-26 · **代码基线**: `mot_pilot_v1_cleanup`(branch `cleanup-merge`) · **作者**: Claude(数据线任务)

> **plan-only 设计**(不改代码), 给 Evan 评审后再实现。
> 决策前提(Evan 2026-06-26 拍板): TextAtlas5M 作为 complete_v1 的 **第 5 个独立域(text)**, caption 用 **i1 的 caption1..5**, 服务 **stage512 + rae(raev2)**, 为 stage512 / stage1024 raev2 训练做准备。
> 关联: [数据集问题审计_complete_v1_dataline_20260625.md](数据集问题审计_complete_v1_dataline_20260625.md)、[mix回归诊断_v1_20260626.md](mix回归诊断_v1_20260626.md)(**先修 mix 机制再加这个域**)、[E1_RAE解码器bakeoff实施spec_v1_20260624.md](../E1_RAE解码器bakeoff实施spec_v1_20260624.md)(rae 线对文字数据的需求)。

---

## 0. 现状(已盘点验证)

| 资产 | 状态 |
|---|---|
| 源数据 `/s3/evan/textatlas5m/`(10 子集, ~5M 图) | ✅ 在(挂载只读), schema 已验 |
| i1 captions `/fsx/evan/textatlas_i1_captions/textatlas/`(6 parquet × ~1M 行) | ✅ 在, `key=<Subset>:<image_path>` + caption1..5 |
| sidecar manifest builder `scripts/build_textatlas_manifest.py` | ✅ 已写(未跑) |
| sidecar 输出 `/fsx/evan/textatlas_manifest_v1/` | ❌ **不存在 = manifest 还没跑** |
| dataline 代码 / 采样 config 的 textatlas 接入 | ❌ **完全没有** |

→ **实质工作几乎全在前方**: 跑 manifest builder + 写 5 处接入代码 + 复验。

### 0.1 数据事实(schema 已逐字段确认)
- **源 parquet**(每子集 N shard, 共 **600 shard**): 列 = `image: struct<bytes:binary, path:string>`(**图字节内联!**)、`image_path: string`、`annotation: string`(TextAtlas 自带 caption, **不用**)。`TextScenesHQ` 多一列 `raw_text`。
  - shard 数极不均: StyledTextSynth 290 / Paper2Text 116 / CleanTextSynth 67 / PPT2Structured 36 / LongWordsSubset-M 34 / TextVisionBlend 22 / PPT2Details 14 / LongWordsSubset-A 11 / CoverBook 5 / TextScenesHQ 5。
  - 子集量级差异大且图很大(StyledTextSynth 425k 图占 578GB、Paper2Text 356k 占 231GB = 高分辨率密集文字页)。
- **i1 captions**: `key = "<Subset>:<image_path>"`, caption1..5(5 个英文密集文字 caption 变体)。
- **sidecar(builder 产出, 源行序对齐)**: `image_path, uid(=key), width, height, elig_512(min(w,h)≥512), elig_1024(min(w,h)≥1024), caption1..5("" 若 i1 缺)`。**不含图字节**(图在源 parquet)。

> ⚠️ **关键差异**: TextAtlas 图是 **源 parquet 内联字节**, 不像现有 4 域(anime=R2 HTTP / monet,synth=tar 流式 / gpic=逐文件)。所以 TextAtlas **不走 image_fetch**, producer 直接从源 parquet 读 image bytes。但源在 `/s3`(mountpoint-s3) → 仍是挂载流式读, 会和现有域抢挂载带宽(见 §5 与回归诊断的关联)。

---

## 1. 现有 4 域的"插槽"(第 5 个域要碰哪些)

现有域分两类(见 [sampler.py:133](../training/dataline/sampler.py#L133) `sample_one`):
- **注入域**(固定比率、不分 band): `anime`(按 priority 抽)、`synthetic`(按 base_w 全池抽)。
- **band-first 域**: `monet_real` / `gpic`(先抽档 band ~ band_mix_real, 再按 domain_mix 在服务该档的域里选)。

**TextAtlas 最自然 = 注入域**(像 synthetic): 文字图没有 MONET 那套 band/captioner 体系, 不分档, 给个固定 domain_mix 权重、buffer 内均匀(或按某权重)抽即可。

接入要碰的 5 处:
| 文件 | 现状 | TextAtlas 需要 |
|---|---|---|
| `training/dataline/config.py` | `ALL_DOMAINS=(anime,monet_real,gpic,synthetic)`; `domain_mix[stage]` 校验和=1([:135](../training/dataline/config.py#L135)) | 加 `textatlas` 到 ALL_DOMAINS; JSON config 的 domain_mix 各 stage 加 textatlas 权重(重新归一) |
| `training/dataline/manifest_index.py` | `list_shards`([:84](../training/dataline/manifest_index.py#L84)) glob 4 域 | 加 textatlas: glob sidecar(`/fsx/.../textatlas_manifest_v1/*/*.parquet`); 新 `load_textatlas_shard`(读 sidecar → uid/elig/caption1..5) |
| `training/dataline/complete_dataset.py` | `_Producer.run` 分发 anime/real_synth([:193 区](../training/dataline/complete_dataset.py#L193)); `self.domains` 列表([:324](../training/dataline/complete_dataset.py#L324)); `_build_pools`([:576](../training/dataline/complete_dataset.py#L576)) | 加 `iter_textatlas_shard`(**源 parquet 取内联图 + sidecar 取 caption, 按行序 zip**); producer 分发加 textatlas 分支; domains 加 textatlas; _build_pools 加 textatlas(注入域, 均匀/base_w) |
| `training/dataline/sampler.py` | 注入域块 anime/synthetic([:142-149](../training/dataline/sampler.py#L142-L149)) | sample_one 注入块加 textatlas(固定比率, `sample_weighted` 或均匀) |
| `training/dataline/caption_build.py` | anime/real/synth caption 构建 | 新 `build_textatlas_caption`: 从 caption1..5 选(策略见 §3) + 真 token 截断 |
| `configs/stage1a_t2i_512.yaml`(+ raev2 config) | dataset 块 | 指向 textatlas sidecar root + domain_mix 含 textatlas |

> 好消息: **600 shard ≫ world** → 文件级分片正常, **TextAtlas 不会有 anime 那种 per-rank 偏斜(M5)**, 也不需要 force_row_level。

---

## 2. Producer: 内联图新模式(`iter_textatlas_shard`)

sidecar 与源 parquet **同子集、同 shard 名、同行序**(builder 保证)。读法:
1. 整读 sidecar shard 进内存(只字符串/int, 无图字节, 便宜) → 得 uid/elig/caption1..5 数组。
2. 按 stage elig 过滤(512 → elig_512; 1024 → elig_1024)。
3. `pq.iter_batches(源 shard, columns=["image"])` 流式取内联字节, 按全局行号对齐 sidecar 行 → 只对 elig 行 decode_rgb + resize_crop。
4. 产 record(同其它域契约): `{u8, prompts 字段, uid, cfg_keep, ...}`。

注意:
- **取样去头部偏置**(坑点#17 同理): 源 parquet 内部是否按某轴有序未知 → 建议 **随机 row-group 取样**(像 anime 那样), 别永远读 shard 头部。**待确认源是否有序**(可抽查)。
- **挂载带宽**: 这是又一个 mountpoint-s3 流式域 → 和 monet/synth tar 抢带宽。**强依赖 mix 回归(F1/F3)先修对**, 否则 TextAtlas 要么被挤没、要么挤没别人。

---

## 3. caption: i1 caption1..5 选取策略(待 Evan 拍)

i1 给 5 个变体。候选:
- **A(推荐) 每次曝光随机选 1 个非空**: 等价 caption augmentation(像 anime per-draw 重 roll), 同图多次曝光见不同 caption, 增多样性、防过拟合单 caption。
- **B 固定用 caption1**: 简单确定, 但丢了 4 个变体的多样性。
- **C 按长度/质量加权选**: 需要先量 5 个变体的长度/质量分布, 复杂度高、暂无依据。

**统一**: 选出后过真 Qwen tokenizer 截断到 budget(复用 `truncate_to_token_budget`, 同审计 F3/M8); CFG dropout 仍只在模型一处做(坑点#23), TextAtlas 无 tag-only 概念 → 走正常 10% dropout(`cfg_keep=False`)。

**待定**: 5 选 1 的策略(A/B/C)? 是否需要先抽样看 caption1..5 的长度分布(英文密集文字 caption 可能很长, 截断率要测)?

---

## 4. 配比 / stage / 子集(待 Evan 拍)

### 4.1 哪些 stage 用 TextAtlas
- sidecar 只有 elig_512 / elig_1024(builder 未算 elig_256)。结合"为 stage512 和 rae"的需求:
  - **stage256 重跑: 不含 TextAtlas**(纯数据修复, 维持原 4 域)—— 待确认。
  - **stage512: 含 TextAtlas**(elig_512)。
  - **stage1024(raev2 线): 含 TextAtlas**(elig_1024)—— 但 elig_1024 比例可能很低(多数文字图 <1024px), 需 builder 跑完看真实 elig 率。

### 4.2 domain_mix 权重(stage512)
现状 512 = anime 0.65 / monet 0.23 / gpic 0.08 / synth 0.04。加 textatlas 要重新归一。
**需 Evan 给目标 text 占比**(例如 0.05 / 0.10?), 其余按比例缩, 或单独从某域(synth?)让渡。**这是要拍的数值**, 我不擅自定。

### 4.3 用哪些子集 / 多少量
- 10 子集差异大: 有的是合成密集文字(CleanTextSynth/StyledTextSynth)、有的是真实场景文字(TextScenesHQ)、PPT/Paper(版式)。
- elig_512 过滤后各子集留存量未知(**builder 跑完才有**: builder 会报 per-subset elig_512/elig_1024/missing-caption 率)。
- **待定**: 全 10 子集都用, 还是挑? (例如 Paper2Text/StyledTextSynth 是高分辨率密集页, elig_512 留存高但可能偏版式; 合成子集可能偏小图、elig_512 低。)

### 4.4 rae(raev2)线怎么用 TextAtlas — **需跨线协调 + Evan 拍**
E1 spec(另一 agent 的 rae 线)里 TextAtlas 是"英文密集文字"训练源, 但:
- **E1 decoder 是无条件重建(image-only, 不要 caption)** → 那条线只需要 TextAtlas 的**图**(可复用同一套源 parquet 读取, 丢 caption)。
- **raev2 生成器(E2/probe)才需要 caption** → 用 i1 caption。
- **问题**: rae 线是独立 repo `mot_rae_e1/`(不碰 mot_pilot dataline)。TextAtlas 接入是做**一份 complete_v1 内的 text 域**(MoT pilot 用)、**还是**单独给 rae 的 image-only adapter? 二者图读取逻辑可共享但落点不同。**需 Evan 明确 rae 线这块归谁做、复用到什么程度。**

---

## 5. 与 mix 回归的强耦合(顺序约束)
TextAtlas 是**新增的第 5 个 mountpoint-s3 流式域**, 直接加剧 [mix回归诊断](mix回归诊断_v1_20260626.md) §2.3 的挂载/资源争抢:
- 若 mix 机制不先修(F1 排空∝消费 / F3 并发保守), 新域要么被快域挤没、要么把别的慢域挤得更狼狈。
- 反之, F1/F3 修好后, TextAtlas 作为注入域能拿到稳定份额。

→ **实现顺序: 先修 mix 回归(诊断文档 §4), 再接 TextAtlas。** 二者可在设计/manifest 准备上并行, 但代码合入有先后。

---

## 6. 实现里程碑(每步带验收, 守坑点#28 全上 JCC)
| # | 步骤 | 验收 |
|---|---|---|
| 0 | **跑 manifest builder**(JCC, CPU 多进程; 输出 `/fsx/.../textatlas_manifest_v1/`) | 600 shard 全产出; 报告 per-subset elig_512/elig_1024/missing-caption 率(定 §4.1/4.3) |
| 1 | 抽样核验: caption1..5 长度分布 + 源是否行序有序 + elig 率 | 定 §3 caption 策略 + §2 取样去偏 + §4.2 配比 |
| 2 | `load_textatlas_shard` + `iter_textatlas_shard`(内联图 + sidecar zip) + 单测 | 离线读一个 shard, uid/caption/u8 对齐, token 顺序对 |
| 3 | config/sampler/_build_pools/caption_build 接入(textatlas 注入域) | 0-GPU loader 短跑, 5 域都出现, mix 含 textatlas 接近目标(**前提: mix 回归已修**) |
| 4 | 2×8 GPU 短跑读 `mix_*` 复验真实配比 | 512 含 textatlas 的 realized mix 达标 |
| 5 | rae 线 image-only adapter(若归本线)| 见 §4.4 与 Evan 定 |

---

## 7. 待 Evan 拍板汇总
1. **caption 5 选 1 策略**: A(每曝光随机) / B(固定 caption1) / C(加权)? 是否先抽样看长度分布?
2. **stage512 的 text 域 domain_mix 权重**: 目标占比多少? 从哪个域让渡(还是整体重归一)?
3. **stage256 重跑是否含 TextAtlas**: 倾向不含(纯修复), 确认?
4. **用哪些子集**: 全 10 还是挑? (等 builder 的 elig 率出来再定也行)
5. **rae(raev2)线的 TextAtlas 落点**: complete_v1 text 域复用 / 还是 mot_rae_e1 独立 image-only adapter? 归本线做到哪一步?
6. **manifest builder 是否现在就提 JCC 跑**(它是后续一切的前置, 且只产 sidecar 不碰训练代码, 风险低)?
