> 已归档 2026-06-27：内容已并入 docs/complete_v1_dataline_设计与变更_v1_20260627.md。

# complete_v1 数据线（dataline）问题深度审计报告

**日期**: 2026-06-25 · **代码基线**: `mot_pilot_v1_cleanup`（branch `cleanup-merge`）· **作者**: Claude（受 Evan 委托整夜调研）

> 范围：MoT complete_v1 四域数据线（anime / monet_real / gpic / synthetic）在**供数吞吐、domain_mix 完整性、可用性、latent-cache 覆盖、S3 挂载可靠性、数据质量、监控**七个维度暴露的问题。
> 方法：6 维并行实测取证 + 对抗式核验 + 单 rank/16-rank 离线探针 + 在跑 run 日志复核 + 代码逐行审计。证据与复现命令见 §10。
> **当前在跑的 3 个 256 ablation**（`ab1causal` / `ab2fusion` / `ab3lora`，~42–48k/133k 步）全部 `use_latent_cache=false`、走在线 VAE encode。`v1baseline_20260624` 已停在 step 990（**已死，不在跑**）。

---

## 0. 一句话结论 + 严重度总览

**最重要的认知纠正**：本次排查最初被 `low-water` 缓冲告警**反向误导**——它喊得最响的 `monet_real` 其实**超供**，几乎不报警的 `gpic` 才是真正**欠采**的域。根因是**在线训练只有"缓冲占比"这个代理信号、没有 realized domain_mix 真值落盘**。这个监控盲区（M1）是整份报告的核心，**已在本次修复**（埋点 + 告警文案）。

| # | 问题 | 类别 | 严重度 | 影响在跑 run | 状态 |
|---|---|---|---|---|---|
| **M1** | 无 realized-mix 真值落盘，唯一信号 low-water 代理且文案反向误导 | 监控 | **P1** | 是 | ✅ **本次已修** |
| **M2** | gpic 欠采（聚合 −7pp，最坏 rank 仅 6.9%）：单线程逐文件随机读 | 供数吞吐 | **P1** | 是 | 待修 |
| **M2b** | **per-rank domain_mix 剧烈漂移**（gpic 6.9–28.9%）：world=16 文件级分片 | mix 完整性 | **P1** | 是 | 待修（新发现）|
| **M3** | anime R2 取图 403 重试自伤（~55% wall 浪费）+ 单线程冷取吞吐被钉死 | 供数/可用性 | **P1** | 是 | 部分可立即缓解 |
| **M5** | anime per-rank caption/NSFW 分布偏斜：17<16 触发文件级分片 | 数据质量 | **P1** | 是 | 待修 |
| **M4** | latent-cache：512 缺 uid2path 索引 + anime 仅 90.2%；1024 未建 | 缓存覆盖 | **P2** | 否 | 切 512/1024 前必修 |
| **M6** | monet/synth tar sidecar 浪费 ~25% 取数带宽（每 jpg 附 4 .pth+1 json）| 供数吞吐 | **P2** | 是 | 待修 |
| **M7** | mountpoint-s3 元数据零缓存，放大 gpic 逐文件域 | 挂载 | **P2** | 是 | 并入 M2 |
| **M8** | anime caption ~19% 在 budget 边界截断（proxy_tok 低估真实 Qwen token）| 数据质量 | **P2** | 是 | 待定 |
| **M9** | rich_nl 取序 medium-first 而非 refined，最长 NL 几乎不进 caption | 数据质量 | **P2** | 是 | **待 Evan 拍意图** |
| **M10** | anime 缓存仅节点本地(/local)+403 永不缓存：预下载须按 p_fail^k 规划 | 缓存 | **P2** | 否 | 预下载时注意 |
| **M11** | 1024 MONET elig 行 shard 间极不均（10/64 shard 为 0）| 供数吞吐 | **P2** | 否 | 切 1024 前 |
| **M12** | 一组登记/健康/文档不一致项（R2 host 文案、EIO 自愈、无 holdout、去重、NSFW 平衡良好）| 杂 | **P3** | 多数否 | 登记备查 |

**对在跑 3 个 run 的总体判断**：稳态 mix 的**恐慌被 low-water 告警严重夸大**——聚合真实 mix 只有 gpic −7pp 一处实质偏移，monet/anime/synth 基本达标。**唯二真正影响"训练看到的数据分布"的是 M2b（per-rank 配比漂移）和 M5（per-rank caption 偏斜）**，二者同源（world=16 文件级分片），且不在任何缓冲/吞吐信号里暴露。其余多为吞吐浪费、未来 stage 的缓存就绪缺口、或可观测性。**不建议为此重启在跑 run**（详见 §7）。

---

## 1. 核心证据：domain_mix 的真相

### 1.1 16-rank 生产真实 mix（307,200 样本，复刻生产 2×8 争抢）

| 域 | 目标 | 聚合实测 | Δ | per-rank 范围 | per-rank std |
|---|---|---|---|---|---|
| anime | 30.0% | 32.9% | +2.9pp | 26.4% – 42.5% | 5.8% |
| monet_real | 39.7% | 41.7% | +2.0pp | **29.9% – 58.1%** | 7.3% |
| gpic | 25.3% | **18.1%** | **−7.2pp** | **6.9% – 28.9%** | 5.5% |
| synthetic | 5.0% | 7.3% | +2.3pp | 5.0% – 11.2% | 1.8% |

**两层结论**：
- **聚合层（集群平均）**：偏移温和，仅 gpic −7pp 是实质问题。争抢反而把单 rank 探针里 monet 的 +9pp 过供拉回到 +2pp。
- **per-rank 层（真正的训练后果，M2b）**：**每个 rank 看到的配比天差地别**——有的 rank 只有 6.9% gpic / 58% monet，有的接近达标。数据并行下各 rank 在**不同的域分布**上算梯度再平均，聚合虽近目标，但梯度方差被放大、且部分 rank 系统性偏。**这是 rank0-only 监控永远看不见的盲区。**

### 1.2 为什么 low-water 告警是错的

`complete_dataset.py:469-475` 每 500 batch 检查 buffer 占比 <25%，原文案断言"**实际 domain_mix 占比会低于目标**"——这是把"buffer 薄"错误因果推断成"采得少"。
- **机制**：`sampler.py` 的 realized share 只取决于域是否 present（buffer 非空），与占比无关。**被抽得最狠的过供域（monet）消耗最快 → buffer 最薄 → 最常告警，恰恰超标。**
- **实测反相关**：`ab1causal` 在跑日志低水位告警计数 = **monet_real 168 / synthetic 73 / gpic 23 / anime 16**。喊得最响的 monet 实测 +2pp 超标，最该报的 gpic（真欠 −7pp）几乎不报——告警与真相**~7× 反相关**。

> 这正是为什么本次排查一开始误判 "monet 被饿死"。**修复见 §8（已落地）。**

---

## 2. P1 问题详解

### M1 · 监控盲区：无 realized-mix 真值（已修）
- **根因**：`domain_counter`(`complete_dataset.py:282` 定义、`:499/:514` 自增)、`cluster_counter`、`cache_miss_counter`、以及 `image_fetch.py` 的 `FETCH_STATS` **全部只在内存自增，训练路径从不落盘**。`train.py` 的 `metrics.jsonl` record 只写模型/优化器遥测（实测首行 keys 仅 `loss/grad_norm/lr/iter_s/shift/sigma_mean/img_tokens/flow_mse/peak_mem_gb/step`，data-line grep = 0 命中）。`fetch_stats_snapshot()`(`image_fetch.py:56`) 接口存在但**从无调用方**。在线唯一 mix 信号就是上面那个会误导的 low-water 告警。
- **后果（连锁）**：(a) 真欠的 gpic 看不见、过供的 monet 反向误报；(b) 定位真凶必须事后跑离线 21120-样本探针；(c) anime 取图健康度、缓存覆盖、producer 吞吐瓶颈全部无在线时序。
- **置信度**：high（代码行号、metrics.jsonl 全文 grep、告警文案逐字均复核命中）。
- **✅ 本次修复**：见 §8。

### M2 · gpic 欠采（−7pp 聚合，最坏 rank 6.9%）
- **根因**：gpic 走逐文件分支（`complete_dataset.py:108-110`），`read_gpic_bytes`(`image_fetch.py:81-86`) 每图一次 `open()` = mountpoint-s3 一次随机 GET；`gpic_path`(`:69-78`) 即使已知 ext 仍先 `os.path.exists()`（多一次 stat）→ **每图 ≥2 次串行往返**。关键：**每域只起 1 个 `_Producer` 线程**(`:372-377`)。实测 mountpoint-s3 随机读**可并发线性扩展**（1 线程 4.7 → 8 线程 68 → 32 线程 244 img/s），所以瓶颈是**单线程串行、不是挂载限流**。
- **数字**：gpic 单线程 **~5.9 img/s**（中位 128ms/p95 413ms/图）vs monet tar 顺序流式 ~50 img/s（实测，慢 ~8.5×；纯 I/O 口径 monet 可达 140–235，含 decode 后约 ~50）。gpic 图在 `/s3-mewtant-raw-datasets/stanford-vision-lab/gpic_umziped`（**不是本地盘，是 S3 FUSE 挂载**——"gpic 本地最快"是错误假设）。
- **后果**：buffer 来不及补 → 周期被清空 → present 缺 gpic → sampler renorm 把份额摊给其它域 → realized 聚合 18% / **最坏 rank 仅 6.9%**（目标 25.3%）。
- **置信度**：high（逐文件分支 + `os.path.exists` + 一域一线程 + 吞吐实测均命中）。
- **修向**：gpic 取图改**并发线程池（8–16 路）**，单这一项 ~6→60–240 img/s 直接修掉欠采；去掉已知 ext 时多余的 `exists`。或预生成 gpic latent-cache。注意：**并发取图（I/O bound，可线程化）与并发 decode（CPU bound，需进程/释放 GIL）要分开扩**。

### M2b · per-rank domain_mix 剧烈漂移（新发现）
- **根因**：`shard_subset_for_rank`(`manifest_index.py:97-112`) 的行级回退条件是 `len(shards) < world`。gpic/monet 各 64 shard、world=16 → `64<16` 为假 → 走**文件级分片**（每 rank 拿不相交的 shard 子集）。叠加 M2 的 gpic 慢、各 shard 的域可用性/速度不同 → **各 rank 的实测配比大幅漂移**（gpic 6.9–28.9%、monet 29.9–58.1%）。
- **置信度**：high（16-rank 探针 307200 样本实测 per-rank 离散 + 分片条件命中）。
- **修向**：把回退条件从 `< world` 放宽到 `< k*world`（k≥2，每 rank 至少覆盖多个 shard），或对易偏域强制行级分片。**与 M5 同源、可一并修。**

### M3 · anime R2 取图：403 重试自伤 + 冷取吞吐被钉死
- **根因**：anime 走 R2 HTTP（`img.metanomaly.co`，前置 Cloudflare 反爬）。`fetch_url_bytes`(`image_fetch.py:116`) 每 URL 重试 3 次、每次失败 `time.sleep(0.3*(i+1))` = **1.8s/最终失败图**（`:145` 确认），且 `except Exception` 把 403 也当可重试。每域 1 个 producer 线程。
- **403 本质**：Cloudflare **反爬挑战页**（`Attention Required` + `server:cloudflare` + `CF-RAY`，**非死链/非签名过期/非死key**）。**完全随机**：同 key 隔 60s 重抓 **81–88% 在通过↔403 间漂移**；403 率在来源/尺寸/rating 上均匀 ~63–70%（无任何内容类被系统性挑出）。集群成功率稳定 **70.3%**（403 占失败 99.98%），本机 ~33%（按 IP 信誉 gating，且并发越高 gating 越严）。06-18 旧文档曾"降并发可恢复 100%"——**这周 CF 收紧了**。
- **对 mix 的影响 = 基本无害**：anime realized **聚合 32.9%、达标**（retries + shard 重洗后重遇 key 是新轮盘 + /local 缓存共同兜住）。被钉死的是**新 key 冷取吞吐**（生产路径 ~0.75–0.85 good-img/s，重试 sleep 占 wall ~55%），不是稳态配比。
- **"随机"的关键利好**：因为是 per-request 随机，**重抓指数恢复**：P(扫 k 遍仍缺)=p_fail^k（生产 p_fail≈0.30 → 3 遍漏 2.7%、4 遍 0.8%）。预下载/补缓存**不修 CF 也能扫到 ~100%**。
- **置信度**：high（重试常数 + 在跑日志成功率 + 挑战页响应体 + 随机性双侧实验均命中）。
- **修向**：(1) **立即低风险**：把 403/404 列为**不可重试**，retries 仅对 timeout/连接抖动生效 → 省 1.8s/失败图、~消大半 wall 浪费 + 日志刷屏。(2) 治本：anime 预下载到**共享存储**（非 /local）走缓存绕开 CF。(3) **不要纯加并发**（反向恶化 CF gating），要加须配 CF allowlist/降 per-IP QPS（服务端，img.metanomaly.co 是 mewtant 自有域，运维可改）。

### M5 · anime per-rank caption/NSFW 分布偏斜
- **根因**：anime 仅 **17 个 shard**，world=16 → `17<16` 为假 → 文件级分片，rank1..15 各只拿 1 个 shard、rank0 拿 shard0+16。anime 17 个 fat shard 在 `has_nl`/rating/risk 上**按 shard 强烈有序**（manifest 似按质量/NL 轴排序），shard 内 random-row-group 无法跨 shard 纠偏 → **每个 rank 的 anime 子分布 = 它那一个 shard 的分布**。
- **数字**：`has_nl` 跨 shard **52.8%–95.2%**（mean 72.9%）；risk_downweight 1.9%–34.4%；NSFW(q+e) 8.7%–19.7%。
- **置信度**：high（分片条件 + world=16 + per-shard 偏斜实测均命中）。**这是 P1 里唯一影响"真实训练数据分布"而非配比/速度的偏斜。**
- **修向**：同 M2b（放宽行级分片回退条件，或对 anime 强制行级分片）。

---

## 3. P2 问题（应修 / 次级数据质量与可观测性）

- **M4 · latent-cache 覆盖缺口**（影响在跑=否，**切 512/1024 前必修**）：生产 root 下只有 `stage=256`（`uid2path.parquet` ✅、4.5TB、99.998% 就绪）和 `stage=512`（**无 stage 根 uid2path.parquet**，只有 4659 个 per-shard safetensors 索引 + anime 覆盖仅 **90.22%**，缺 159 万 uid，failed 100% 是 `fetch_empty`=CF 限流残留）；**`stage=1024` 完全不存在**。`latent_cache.py:36-40` 缺 uid2path 即 `raise FileNotFoundError` → 任何 512 的 `use_latent_cache=true` 启动即崩。QC `summary.json` 的 `index_eq_tensors=true` 只校验 per-shard、**不查训练真正消费的 stage 根 uid2path** → **QC 绿 ≠ 训练可用**。修向：切 512 前先 `vae_cache --only-domains anime --run-tag patch`（集群 IP + 低并发）补 anime → 再 `build_latent_cache_index.py --stage 512` 生成 uid2path；切 1024 前提前提交缓存 job（~30TiB、数天）；QC 加 uid2path 存在性+行数校验。

- **M6 · tar sidecar 浪费 ~25%**（影响在跑=是）：webdataset tar 每样本 = `{key}.jpg + .json + 4 个 .pth`；`stream_tar_images`（`'r|'` 顺序流，`image_fetch.py:94-113`）必须把每个成员字节流过，mountpoint-s3 无法 seek 跳过非 jpg。jpg 有效率仅 **74–75%**。修向：重打包只含 jpg(+精简 json) 的训练 tar，省 ~25% 带宽；或迁 latent-cache。**置信度 med-high**（定量 25% 来自单次 warm/cold 实测，机制确凿但未独立复跑）。

- **M7 · mountpoint-s3 元数据零缓存**（影响在跑=是）：warm stat ≈ cold stat（~15ms，强一致只读设计）。对 tar 顺序流无影响，对 gpic 逐文件是**结构放大器**（每图重付 stat+GET）。**并入 M2 一起修**（gpic 并发化 + 已知 ext 跳过 exists）。**置信度 med-high**（同 M6）。

- **M8 · anime caption ~19% 边界截断**（影响在跑=是）：caption 构建用 `proxy_tok = words×1.32`(`caption_build.py:25`)，对 booru tag（下划线/逗号）**系统性低估**真实 Qwen token；构建上限 `truncate_tok=340`(proxy)，真实 token mean236/p95 368/max 399 vs budget 343 → **19% 在句/逗号边界被截短**（丢尾部 tag）。已有 `tokenizer_path` 消除了模型端 mid-token 硬截，但构建侧仍截。属信息损耗，非数据错误。修向：构建上限改用真 tokenizer 或调高 proxy 系数。

- **M9 · rich_nl 取最长描述失败**（影响在跑=是，**待 Evan 拍意图**）：`caption_build.py:245` `nl = medium_description or refined_description or ...` **medium-first**，而 refined/medium/short 93.4% 同时 present → rich_nl **永远取 medium(~163 词) 而非 refined(~248 词)**，与 config "NL_long=refined" 语义不符（对比 `_build_nl:261-262` 是 refined-first，自相矛盾）。**核心是产品意图问题**：到底想要 medium 还是 refined？若要最长，把 `:245` 改 refined→medium→short。

- **M10 · anime 缓存本地 + 403 不缓存**（影响在跑=否）：`fetch_url_bytes` 仅 ok 时写缓存、403 从不缓存；`anime_cache_dir=/local/...` 是**节点本地 overlay**（非共享、重调度即丢）。预下载达 99% 覆盖需 k≈10 次重抓（p_fail≈0.63 本机口径；集群 p_fail≈0.30 则 k≈4）。修向：一次性预下载到**共享 S3/FSx**，每 key 重抓直到成功或 k=8–10；缓存目录改共享路径。

- **M11 · 1024 MONET elig shard 间极不均**（影响在跑=否）：MONET manifest 按分辨率排序，`elig_1024` 集中后段，前 10/64 shard 全 0；1024 走文件级分片 → 零-elig shard 落单 rank，per-rank monet@1024 供给差 8×。因 anime@1024 占 0.80、monet 仅 0.08，影响有限但存在。修向：1024 对 monet 走行级分片或按 elig 重排打散。

---

## 4. P3 登记项（无需紧急动作，含校验性"健康"结论）

- **R2 host 文案不一致**：config 写 `ncotar-worker...workers.dev`，实际生效是 manifest `s3key` 的 `img.metanomaly.co`（代码不读 config 那行）。无功能影响，但**排查 403 易找错 host** → 建议修正/注明。
- **256 `use_latent_cache=false` 是有意取舍**（符合预期）：Evan 2026-06-23 拍板，256 下读 moments 比在线 encode 慢 ~5×（冷 open 散落数十文件）。4 个 run 一致。**无需动作**。
- **挂载 EIO 自愈**：6 个 train log/27826 行仅 **3 次** `OSError[Errno5]`（synth manifest stat），producer 下轮重试自愈（3/3 复测恢复）。可选硬化：stat 加退避重试 + EIO 计数落盘（突增=S3 限流早期信号）。
- **无 holdout 切分**：real/synth `holdout` 列全 0、anime 走 fat 源绕过 thin manifest → 无 train/eval 切分（已知设计状态，后续要留评测需另建）。
- **sscd_nn 近重复**：已在选样压制（MONET ≥0.9 仅 0.0036%）；残余可忽略。
- **NSFW/cluster/tag-only**（校验性**健康**结论）：rating/cluster（4863 簇 top1 0.53%）平衡良好；tag-only 29.1% 正确**永不 CFG-drop**；empty caption=0%。**符合 config 意图，无需动作。**

---

## 5. 三条跨维根因（修这三条覆盖绝大部分）

- **C1 · 只有 buffer 代理、没有 realized-mix 真值** —— **issue 集的核心**。6 维独立撞同一墙：监控全靠会误导的 low-water 告警，真值 counter 从不落盘。修 C1（= M1，**已修**）后 gpic 欠采/per-rank 漂移/anime 取图健康/缓存命中全部**在线第一眼可见**，无需事后离线探针。
- **C2 · 每域 1 个 producer 线程 + mountpoint-s3 随机读特性** —— 决定所有"逐文件域"的吞吐天花板。tar 顺序流高效、逐文件随机读被钉死（而随机读本身可并发到 244 img/s → **瓶颈是单线程不是挂载**）。是 M2/M3/M6/M7 的共同结构背景。**对逐文件域加并发取图线程池是统一解。**
- **C3 · Cloudflare 对 anime R2 按 IP 信誉+并发反爬 gating** —— 跨维同源。对**稳态 mix 基本无害**（兜得住），对**冷取吞吐**（重试自伤）和**预缓存覆盖**（512 缺口）是实打实代价。治本统一指向：**预下载到共享存储绕开 CF + 403 不重试**，而非加并发。

---

## 6. 对在跑 3 个 256 run 的影响评估

| 维度 | 结论 |
|---|---|
| 稳态聚合 mix | 基本达标（gpic −7pp 是唯一实质偏移）；low-water 的"灾难"是误报 |
| **per-rank 配比/分布**（M2b/M5）| **真实存在且量级大**，但三 run 共用同一产线 → 对**消融相对比较**仍是共模、可比；对**绝对配方还原度**有损 |
| anime 403 | 对 mix 无害（达标），仅日志噪音 + 冷取慢 |
| latent-cache 缺口（M4）| **不影响**（256 全 `use_latent_cache=false`）|
| 吞吐浪费（M6/M7）| 拖慢供数，但训练是否 data-bound 未隔离（见 §9 未决#2）|

**建议：不为以上任何一条重启在跑 run。** 它们继续向 133k 推进即可。理由：(a) 偏移温和且共模、消融可比；(b) 任何数据路径改动都需 resume + 在该步引入一次 mix 不连续（反而是新混淆）；(c) 真正的修复（gpic 并发、行级分片、CF）应在**下一批 run 或 512 阶段**一次性带入。

---

## 7. 关于"把真实 domain_counter 周期写入 metrics.jsonl"这条建议的评估

**强烈赞同，而且本该一开始就有。** 这条直击 C1/M1——本次"monet 先被误判饿死、其实超供，真正欠的是 gpic"的来回，根源就是只有 low-water 代理、没有真值。**已在本次落地实现**（cleanup，未 commit，待你审阅）：

### 7.1 已实现的埋点（cleanup）
- **`training/dataline/complete_dataset.py`**：
  - `__init__` 新增 `self.produced`（各域 producer 累计产出）、`self.buffer_fill`（各域 buffer 占比，loader 每批刷新）；`domain_counter` 已有（realized mix 真值）。
  - `_Producer.run` 产出一条 `+1`（GIL 下计数安全）。
  - 修正 low-water 告警文案：**删掉"占比会低于目标"的错误因果断言**，改为"低水位≠欠采，真实占比看 metrics 的 `mix_*`"。
- **`train.py`**：新增 `dataline_metrics_every`（默认对齐 `log_every`）+ `_dataline_metrics()`，rank0 每 N 步把以下扁平 key 并入 `metrics.jsonl`（失败安全，内部 try/except 不影响训练）：
  - `mix_*` = realized domain_mix **累计**真值；`mixw_*` = 自上次记录的**窗口** mix（看当前配比）；
  - `buf_*` = 各域 buffer 占比；`prod_*` = 各域 producer 累计产出（→ 吞吐）；
  - `anime_fetch_ok_rate / ok / cache_hit / fail`（调既有 `fetch_stats_snapshot()`）。
- **`configs/stage1a_t2i_256_v2.yaml`**：加 `dataline_metrics_every: 50`；**`config_schema.py`** 注册该字段（消除 unknown 告警）。

### 7.2 验证
- **complete_dataset 半**：✅ 离线实测通过——`produced={monet 912, synth 912, gpic 218, anime 75}`、`buffer_fill={monet 1.0, synth 1.0, gpic 0.03, anime 0.0}`、`fetch ok=75/fail=35`。**这组数本身就一眼暴露"gpic/anime producer 慢、monet/synth 满仓"**——正是过去缺的诊断。
- **train.py `_dataline_metrics` 半**：✅ 离线对真实 dataset 跑通——产出 record 形如
  `{"mix_monet_real":0.418,"mix_gpic":0.3125,"mix_anime":0.2188,"mix_synthetic":0.0508, "mixw_*":..., "buf_anime":0.0,"buf_monet_real":0.043,"buf_gpic":0.023,"buf_synthetic":1.0, "prod_gpic":185,"prod_anime":74,"prod_synthetic":720,"prod_monet_real":220, "anime_fetch_ok_rate":0.638,"anime_fetch_ok":74,"anime_fetch_fail":42}`；
  连续两次记录的窗口 mix（`mixw_`）正确反映增量（第二窗口 monet 0.599），全部 key 写出、无异常。GPU smoke（`smk_dl_metrics.yaml`，1 GPU/20 步）也已 done rc=0（metrics 写到节点 /local，未回收）。

### 7.3 一个重要后续（建议）
当前 metrics 是 **rank0-only**，**看不见 M2b 的 per-rank 漂移**（rank0 只是 16 个异质 rank 之一）。若要在线观测 per-rank 配比，需 (a) 各 rank 各写一份、或 (b) 对 `domain_counter` 做 all-gather 后由 rank0 汇总离散度。建议作为下一步小改。

---

## 8. 修复优先级建议（供排期）

| 优先 | 动作 | 影响 | 工作量 |
|---|---|---|---|
| 1（已做）| **M1 真值埋点 + 告警文案** | 永久消除监控盲区 | ✅ 完成 |
| 2 | **M3 立即缓解**：403/404 不重试 | 省 ~55% 取图 wall + 日志噪音 | ~5 行，低风险 |
| 3 | **M2/M2b/M5 统一修**：逐文件域并发取图线程池 + 放宽分片回退到 `<k*world` | 修 gpic 欠采 + per-rank 漂移 + anime caption 偏斜 | 中（下批 run 带入）|
| 4 | **CF 服务端**：img.metanomaly.co 加 IP allowlist / 降 per-IP QPS | anime 取图 ~100% + 去噪 | 运维侧 |
| 5 | **M4**：切 512 前补 anime 缓存 + 建 uid2path；切 1024 前提前建缓存 | 解锁 512/1024 latent-cache | 切 stage 前 |
| 6 | M6/M9/M8 按需 | 带宽/caption 质量 | 低-中 |

---

## 9. 未决 / 待你拍板（open questions）

1. **M9 rich_nl 取序**：到底要 medium 还是 refined？（产品意图，非事实问题）
2. **训练是否 data-bound 未隔离**：`ab1` iter median 2.47s 但 **mean 3.88s、max 87.8s**（重尾，疑 ckpt 写 /fsx 尖峰或间歇 data-wait）。需补 dataloader 等待计时器才能定性 M2/M3 是"只伤配比"还是"也伤速度"。**这是免费的在线信号，建议登记。**
3. **per-rank 偏斜的 batch 级实际量级**：M2b/M5 是机制+静态分布实测，未跑多 rank loader 抓真实 batch 统计。落 per-rank metrics 后跑一小段可直证。
4. **512 anime 缺失 159 万 uid 现在是否仍可取**：256 当时抽样 200 全可取，512 未抽样；CF 本周收紧，恢复率可能更低。补编前需集群 IP + 低并发抽样验证。
5. **本机 vs 集群 anime 成功率口径不同**（本机 per-attempt 33% vs 集群 per-call 70%），确证 IP 信誉差需在集群节点跑同一单试脚本。

---

## 10. 证据与可复现（本次新增的探针/脚本）

- **单 rank mix 探针**：`mot_pilot_v1_ablations/scripts/probe_realized_mix.py` + `jobs/probe_realized_mix.yaml`（0 GPU）。产物 `experiments/results/probe_realized_mix/probe.log`。
- **16-rank 生产 mix 探针**：`jobs/probe_mix_16rank.yaml`（2×8、0 GPU，每节点 8 进程）。产物 `experiments/results/probe_mix16/rank*.log`，聚合见 §1.1。
- **anime 403 随机性/相关性实验**：`scripts/anime_403_correlate.py` + `jobs/anime_403_correlate.yaml`（本机 + 集群双侧）。
- **gpic vs monet 取图吞吐**：inline 实测 gpic ~5.9 img/s（中位 128ms）vs monet tar ~50 img/s。
- **dataline 埋点验证 smoke**：`mot_pilot_v1_cleanup/jobs/smk_dl_metrics.yaml`（1 GPU、20 步）。
- **可行性 workflow**（预下载评估）与**审计 workflow**（本报告 6 维取证）：见 session workflow 记录。
- **复核基线**：所有"代码行号"基于 `mot_pilot_v1_cleanup`（已确认 caption_build/manifest_index 在 ablations↔cleanup 间一致）。在跑 run 日志成功率/告警文案与代码逻辑吻合（未逐字节核对 git commit）。

> **附注**：本报告所有"在跑 run"特指 `ab1causal/ab2fusion/ab3lora`（live，~42–48k/133k）；`v1baseline` 已死（step 990）。`ab3lora` 的 metrics.jsonl 多一列 `lr_und_lora`，其余两 run 无（不影响结论）。
