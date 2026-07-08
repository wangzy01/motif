# complete_v1 数据线（dataline）设计与变更

**日期**: 2026-06-27 · **代码基线**: `mot_pilot_v1_cleanup`（branch `cleanup-merge`）· **作者**: Claude（数据线任务，受 Evan 委托）

> 🔁 **部分取代（2026-06-29）**：本文描述的【各 stage domain_mix / MONET 源权重(count^-0.2) / anime caption 路由 / token 预算 / noise-shift】已被 [数据+训练阶段_重设计方案_v1_20260629](数据+训练阶段_重设计方案_v1_20260629.md) 取代（新四阶段 + 显式源比例表 + per-stage token）；本文的 **loader 机制 / TextAtlas 第 5 域接入 / mix 回归供给修复 / 数据线代码基线** 仍为当前数据线权威。

> 本文合并并取代三份已实现的数据线文档（均已归档 `docs/_archive/`，仅供溯源）：
> [数据集问题审计_complete_v1_dataline_20260625](_archive/数据集问题审计_complete_v1_dataline_20260625.md)（根因，已归档）、[mix回归诊断_v1_20260626](_archive/mix回归诊断_v1_20260626.md)（回归诊断，已归档）、[TextAtlas5M接入设计_v1_20260626](_archive/TextAtlas5M接入设计_v1_20260626.md)（第 5 域设计——当时 plan-only，**现已实现**，已归档）。
> 落地 commit：`546bdd0`（TextAtlas 第 5 域 + domain_mix 供给饿死回归修复）+ `faab339`（stats_cache 多机脚枪/probe loader 残留修补）。

---

## 1. 执行摘要 / 现状

complete_v1 数据线 **当前为 5 域**（anime / monet_real / gpic / synthetic / **textatlas（第 5 个，注入域）**），mix 回归已修、TextAtlas 已接入并双验通过。关键变更全部落在 `cleanup-merge` 的 `546bdd0`/`faab339`，**已 commit**（注：5 域采样 JSON 在 repo 外的 `/fsx data_analysis`，见 §5 坑点#11）。

- **mix 回归修复（F1 为主）**：`drain_mode=proportional`（排空∝消费）+ anime 串行取图（`fetch_workers_anime=1`）+ gpic 并发取图 + real/synth 行内打乱 + M1 监控埋点（`mix_*`/`buf_*`/`prod_*`/`data_wait`）。
- **TextAtlas 第 5 域**：sidecar manifest（uid/elig/caption1-5）+ 源 parquet 内联图按 `image_path` join；注入域（同 synthetic）；caption 每曝光从 caption1-5 随机选 1；/local 预存优先 + S3 回退；纳入 9 子集（丢 CoverBook）。
- **验证**：单测全过；16-rank probe 5 域聚合 ±0.1%（textatlas buffer 1.00 未饿）；2×8 GPU 512 五域达标、不 data-bound。

### per-stage 5-way domain_mix（renorm 前名义值，权威源 = 5 域采样 JSON）

| stage | anime | monet_real | gpic | synthetic | textatlas |
|---|---|---|---|---|---|
| 256 | 0.27 | 0.357 | 0.228 | 0.045 | 0.10 |
| 512 | 0.65 | 0.13 | 0.08 | 0.04 | 0.10 |
| 1024 | 0.80 | 0.08 | 0 | 0.12 | 0（待定，§5）|

> sampler 运行时只对 buffer **present** 的域 renorm 到 1.0（`config.py:domain_mix_renorm`）；某域临时断供即把份额摊给其余域——这既是兜底，也是 §2 回归的放大器。

---

## 2. 根因：为何这样设计（必须保留的 WHY）

### 2.1 最重要的认知纠正：low-water ≠ 欠采

排查初期被 `low-water` 缓冲告警**反向误导**——喊得最响的 `monet_real` 其实**超供**，几乎不报警的 `gpic` 才是真欠采的域。

- **机制**：`sampler.sample_one` 的 realized share 只取决于域是否 **present**（buffer 非空），与 buffer 占比无关。**被抽得最狠的过供域消耗最快 → buffer 最薄 → 最常告警，恰恰超标。**
- **实测反相关**：旧 `ab1causal` 日志低水位告警计数 = monet 168 / synth 73 / gpic 23 / anime 16；喊得最响的 monet 实测 +2pp 超标，最该报的 gpic（真欠 −7pp）几乎不报——告警与真相约 7× 反相关。
- **盲区根因（M1）**：在线只有"缓冲占比"这个代理信号，**没有 realized domain_mix 真值落盘**（`domain_counter` 等只在内存自增、训练路径从不写）。这是整套问题的核心。**已修**（§3 监控埋点）。

### 2.2 供数饥饿 + buffer "无差别排空" 是 mix 崩塌的机制根因

数据线结构：每域 1 个 producer 线程 → bounded queue → per-domain shuffle buffer（deque）→ sampler 带放回抽样。回归的致命环节是 **buffer 每批"无差别"排空**：

旧 `drain_mode=uniform`（`complete_dataset.py:700-704`）：每个 batch 对**每一个域**无条件 `popleft bs//2`。这把"域是否 present"**完全绑定到 producer 吞吐、与 domain_mix 目标无关**——一个域要保持 present，它的 producer 必须每批补 ≥bs//2；慢域（gpic 逐文件 / 被过订阅的 monet/synth tar 流）产出 < bs//2/批就被抽干 → buffer 空 → 不 present → 被 renorm-away。

这就是为什么 06-26 一版"数据修复"（并发取图 + anime 强制行级）虽把 per-rank 漂移修好（anime σ 5.8%→0.8%），却**把聚合 mix 搞崩**（monet 39.7%→0.6%、synth 5%→0.1%）：并发取图（每进程 anime 8 + gpic 8 = 16 取图线程 × 8 进程/节点）**过订阅网络/挂载/CPU**，把走 tar 顺序流、未提速的 monet/synth producer 饿到 ~0 img/s → buffer 空 → renorm 把份额全摊给永远 present 的 anime/gpic（anime 30→57%、gpic 25→42%）。

### 2.3 关键坑点（保留）

- **坑点#11**：别边跑边改——改一项验一项；两份采样 JSON 并存（§5）。
- **per-rank 漂移（M2b/M5）vs 聚合 mix 是两回事**：`shard_subset_for_rank` 的文件级分片在 shard 数少（anime 仅 17）时让每 rank 只见 1 个 shard 的偏斜子分布（has_nl 跨 shard 52.8%–95.2%）→ per-rank caption/NSFW 偏斜。修法是对 anime **强制行级分片**（`force_row_level`，每 rank 读全 17 shard、靠 producer RNG 行级错开）。但行级会放大 anime 产出 → 必须配合 §3 的 drain/取图修复，否则就是 2.2 的崩塌。**full-read 域（gpic/monet/synth 各 64 shard）不开行级**（高 world 下重复读全 shard 会吞吐回退）。
- **0-GPU probe 高估严重度**：probe 全速空转，§2.2 的 `bs//2/批` 排空速率极高 → 慢域瞬间被抽干；真实 GPU 训练每 step ~2.5–3s 被 forward 节流（data_wait ~5%，不 data-bound）→ 慢 producer 有时间回填。故 **mix 合格以 2×8 GPU 的 `mix_*` 真值为权威判据**，0-GPU probe 仅作便宜的相对比较/隔离实验。
- **anime R2 取图 403 = Cloudflare 反爬挑战页**（非死链/签名过期），完全随机、按 IP 信誉+并发 gating；对稳态 mix 基本无害（/local 缓存 + 重试兜住），但**加并发会恶化 gating**——故 anime 取图保持串行（§3）。

---

## 3. 修复 F1 + 验证

### 3.1 落地的修复（`546bdd0`）

| 项 | 做法 | 位置 |
|---|---|---|
| **drain_mode=proportional（F1，根治）** | 每批只 `popleft` 各域**本批实际被抽中量**（`sampled[d]`）→ 每域只需产出自己那份 domain_mix 速率即可保持 present；over-producer（anime）被封顶在 target、多余 FIFO 淘汰=无偏丢弃。把"是否 present"从 producer 吞吐**解耦回 domain_mix** | `complete_dataset.py:696-699`（drain）；`drain_mode` 入参；另有可选 `keep_present`(F2) 加固 |
| **anime 串行取图** | `fetch_workers_anime=1`——anime 是缓存/R2 域，并发只过订阅挂载饿死 tar 域；实测 anime 即便串行仍供 65%，配比由 drain 兜住 | `complete_dataset.py:294,368`；512 yaml `fetch_workers_anime:1` |
| **gpic 并发取图** | gpic 逐文件随机读用线程池（`fetch_workers`，512 设 4：demand<supply，4 足够不过订阅）| `_prefetch_raw` / `iter_real_synth_shard`；512 yaml `fetch_workers:4` |
| **real/synth 行内打乱** | producer 传 rng 打乱 src_shard 顺序 + 行序，去"连续 elig 切片"低变量偏置（tar 已整 shard 读进内存，零额外 I/O）| `iter_real_synth_shard` rng |
| **anime 强制行级分片** | 修 per-rank 偏斜（M5/M2b），见 §2.3 | `shard_subset_for_rank(force_row_level=True for anime)` |
| **M1 监控埋点** | rank0 每 N 步（`dataline_metrics_every`）把 `mix_*`（realized 累计真值）/`mixw_*`（窗口 mix）/`buf_*`/`prod_*`/`anime_fetch_*` + `data_wait_s`/`data_wait_frac` 写 `metrics.jsonl`（失败安全）；并修正 low-water 告警文案（删"占比会低于目标"的错误因果） | `train.py:_dataline_metrics`；`complete_dataset.py:635-645` 告警 |

### 3.2 验证（实测，repo 内可复核产物）

- **单测**：`scripts/test_textatlas_dataline.py` 全过（含 sampler 注入边际 ≈ 目标）。
- **16-rank probe（`probe_mix16_512ta`，2×8/0 GPU，61440 样本）**：聚合 5 域全 ±0.1%——anime 64.9%(t65)、monet 13.1%(t13)、gpic 8.1%(t8)、synth 4.1%(t4)、**textatlas 9.9%(t10)**；per-rank σ ≤0.9%（修前 gpic 22pp/monet 28pp 极差）；anime shard/rank=17（行级生效）；monet/gpic/synth/textatlas buffer 稳态 **1.00**（未饿），仅 anime 0.05–0.07（被抽得最狠，符合 §2.1 机制）。
- **2×8 GPU 512 双验**：
  - `val_2x8_512mix`（4 域 mix-fix，step200）：anime 0.6448 / monet 0.2214 / gpic 0.0869 / synth 0.0469——全 ±5%；`data_wait_frac` **0.047**（不 data-bound）。
  - `val_2x8_512ta`（5 域含 textatlas，step200）：anime 0.6473 / monet 0.1297 / gpic 0.0814 / synth 0.0469 / **textatlas 0.0947**；`data_wait_frac` **0.049**；textatlas buffer 1.00。
- 复现：`scripts/aggregate_probe_mix.py experiments/results/probe_mix16_512ta`；`experiments/results/val_2x8_512{mix,ta}/metrics.jsonl`。

---

## 4. TextAtlas 第 5 域接入

### 4.1 架构

- **数据形态**：源 parquet **图字节内联**（`image: struct<bytes,path>`），不像其它 4 域（anime=R2 HTTP / monet,synth=tar 流式 / gpic=逐文件）。故 TextAtlas **不走 image_fetch**，producer 直接从源 parquet 读 image bytes。
- **sidecar manifest**（`build_textatlas_manifest.py` 产，`/fsx/evan/textatlas_manifest_v1/<Subset>/<shard>.parquet`）：`image_path / uid(=<Subset>:<image_path>) / w / h / elig_512 / elig_1024 / caption1..5`（i1 开源长 caption 的 5 个变体，`""` 若 i1 缺）。**不含图字节**。`elig_256` 由 sidecar w/h 现算（builder 只产 512/1024）。
- **读取**（`iter_textatlas_shard`）：sidecar 与源 parquet 同子集/同 shard 名/同行序（builder 保证）；整读 sidecar（便宜，仅字符串/int）→ 建 `image_path→行` LUT（仅 elig 行）→ 随机 row-group 顺序流式取源 parquet 内联图，按 `image_path` join → decode+resize_crop。随机 row-group + 组内打乱去头部/连续切片偏置。
- **源图路径**：优先 `textatlas_local_root`（/local NVMe 预存，绕开 S3 挂载带宽）；缺失自动回退 `textatlas_source_root`（/s3，保正确性、换机器/未预存仍可跑）——纯优化，部分完成不影响正确性。
- **注入域**（同 synthetic）：`served_mask=0`、`base_w=1`（池内均匀）、不分 band；sampler 注入块按固定 domain_mix 比率抽（`sampler.py:150-153`）；caption 走正常 10% CFG dropout、无 tag-only 概念。
- **分片**：9 子集共 **595 shard ≫ world** → 文件级分片即可，**无 anime 那种 per-rank 偏斜**，不需 force_row_level。
- **接入面**（6 处）：`config.py`（ALL_DOMAINS+路径）/ `manifest_index.py`（`load_textatlas_shard`）/ `complete_dataset.py`（`iter_textatlas_shard` + producer 分支 + domains）/ `sampler.py`（注入块）/ `caption_build.py`（`build_textatlas_caption`）/ `config_schema.py` + 512 yaml。
- **caption 策略（Evan 拍：A=每曝光随机选 1 非空）**：`build_textatlas_caption` 从 caption1-5 随机选 1 个非空（= caption augmentation，同图多曝光见不同变体防过拟合），全空→`""`；选出后过真 Qwen tokenizer 截到 budget（同 anime/real）。

### 4.2 Evan 锁定的决策（已决，非待拍）

- **caption**：A 每曝光随机选 1 非空（上）。
- **512 text=0.10**：从 monet 让渡（守 anime 0.65 / gpic 0.08 底线），其余不动；依据 **i1 等权配方**（11 数据集等权 → TextAtlas 单源 ~9.1%），故 0.10 量级合理。
- **256 text=0.10**：原 4 域权重各 ×0.9 按比例让渡（256 = anime0.27/monet0.357/gpic0.228/synth0.045/text0.10）。
- **1024 text=0**：暂不纳入（deferred，§5）；多数文字图 <1024px，elig_1024 比例低。
- **9 子集（丢 CoverBook）**：CoverBook 仅 15 张 ≥512px，弃。纳入 = CleanTextSynth / LongWordsSubset-A / LongWordsSubset-M / PPT2Details / PPT2Structured / Paper2Text / StyledTextSynth / TextScenesHQ / TextVisionBlend。
- **latent-cache 模式**：textatlas 暂无预存 latent → cache 模式下跳过本域，其权重由 sampler 重归一摊给其余域（在线/非 cache 模式正常启用）。

### 4.3 builder/验证摘要

- builder 产 600 shard sidecar；**elig_512 ≈ 4.04M、caption 覆盖 99.96%** ⚠️[待核实 2026-06-27：此两数仅见于 `546bdd0` commit message / summarize_textatlas_manifest 的 summary 产物，repo 内 experiments/results 未存该 summary，未独立复核]。
- 接入验证见 §3.2（单测 + 16-rank probe textatlas 9.9% 未饿 + 2×8 GPU textatlas 0.0947 / data_wait 4.9%）。

---

## 5. 待办 / 已决

**已决（见 §4.2）**：caption=A、512 text=0.10、256 text=0.10、9 子集、注入域、cache 模式跳过。

**deferred（明确推后，非未决）**：

- **256 per-node 预存**：重跑 256 时由训练 job 内 `scripts/prestage_textatlas.py` **每节点** 预存其 rank 要读的源 shard 到 /local（JCC 动态调度 → 须 train.py 之前在同节点跑；读侧 local 优先、缺失回退 S3，故纯优化）。
- **1024 text 权重**：当前 0（多数文字图 <1024px、elig_1024 率低）；待 builder elig 率落定后再定是否纳入。
- **rae 线**：TextAtlas 在 rae（raev2）线的落点（complete_v1 text 域复用 / 还是 mot_rae_e1 独立 image-only adapter）**暂定 pilot-only**，跨线协调留待 Evan/rae 线另定（本文不触 `docs/E1_RAE*`）。

**已知问题 / backlog（来自 [_archive/数据集问题审计](_archive/数据集问题审计_complete_v1_dataline_20260625.md)，仍在代码中、未处理）**：

- **M9 anime rich_nl caption 取源自相矛盾（P2，待 Evan 拍意图）**：`caption_build.py` `_build_rich_nl` 用 `medium_description or refined_description or short_description`（**medium 优先**），而 `_build_nl` 非 short 路由用 **refined 优先** → rich_nl 实际永不选最长的 refined caption，与「NL_long=refined」意图不一致。需 Evan 定 rich_nl 应取哪条再统一。
- **M6 tar sidecar ~25% 取图带宽浪费**：monet/synth 的 webdataset tar 每样本带 `.json + 4×.pth` sidecar，mountpoint-s3 无法 seek 跳过，jpg 有效载荷仅 ~74-75%。重打包去 sidecar 可省 ~25% 取图带宽（与 512/256 供给瓶颈相关）。

**坑点#11（两份采样 JSON 并存，务必区分）**：

- `t5_sampling_complete_v1.json`（**4 域原版**）：保留给在跑的旧代码 ablation（`mot_pilot_v1_ablations` 共享 /fsx 实时读），**勿改**。
- `t5_sampling_complete_v1_textatlas.json`（**5 域含 textatlas**）：`cleanup-merge` 的 configs 指向此版（512 yaml `dataset.sampling_config` 已切）。
- 两 JSON 均在 repo 外的 `/fsx/evan/mot_code/data_analysis/configs/`，**未随 `546bdd0` commit**（repo 只含指向它们的 yaml/代码）。改 5 域配比改这份、别动 4 域版。
