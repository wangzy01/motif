# Stage512b-real 决策记录

本文件是 Stage512b-real 的**决策权威**。凡本文件明确列出的条目,与
`MOT_HANDOFF_CURRENT.md`、`STAGE512B_REAL_RESEARCH_BRIEF.md` 冲突时以本文件为准;
未列出的条目,那两份任务书仍然有效(架构冻结、real 不上采样、TextAtlas 5% 等)。

每条都记录**是谁、什么时候、依据什么实测**拍的板。没有实测依据的条目不写进来。

---

## S512B-01 域配比 = 30 / 40 / 20 / 5 / 5 —— **覆盖任务书**

**任务书原文**(`MOT_HANDOFF_CURRENT.md:73`、`STAGE512B_REAL_RESEARCH_BRIEF.md:15`):

```
Anime / MONET / GPIC / Synthetic / TextAtlas = 32 / 44 / 14 / 5 / 5,real 合计 58%
```

**实际执行**(`t5_sampling_stage512b_real_v1.json` 的 `domain_mix.512b`,已建成 plan
`stage512b_offline_exact_v2_w64_512m_anchors_v1` 所realize):

```
Anime .30 / MONET .40 / GPIC .20 / Synthetic .05 / TextAtlas .05,real 合计 60%
```

**这个改动的来历要说清楚**:它出自 commit `879180d`(2026-08-01,Claude),
commit message 只写了"取代 2026-06-29 的 anime 对齐版(anime 0.78)",**没有记录用户批准**。
任务书写明"real 58% 内的重分配须由用户批准后才能实施",而这次是把 real 从 58% 抬到 60%,
既越过了 58% 的边界,也没有走批准流程。**这是流程缺陷,由 Claude 造成。**

**512M 曝光下的实际差异**:

| 域 | 32/44/14 | 30/40/20 | 差 |
|---|---:|---:|---:|
| anime | 163,840,000 | 153,600,000 | **−10,240,000** |
| monet_real | 225,280,000 | 204,800,000 | **−20,480,000** |
| gpic | 71,680,000 | 102,400,000 | **+30,720,000** |
| synthetic | 25,600,000 | 25,600,000 | 0 |
| textatlas | 25,600,000 | 25,600,000 | 0 |

plan 实测的 monet_real 恰好 204,800,000(40.000%)、gpic 恰好 102,400,000(20.000%),
即配置值被精确兑现,没有额外漂移。

**决策**(Evan,2026-08-02,在看到上表之后):**保留 30/40/20/5/5**。理由是本阶段定位
是真实照片广度,GPIC 扩容已经在数据侧实际兑现(自有 `cluster_sizes` 加权后有效簇 16,798,
见 S512B-08),real 58%→60% 与该定位一致。

**为什么现在改回去代价很大**(供未来接手者判断,不是决策理由):`domain_mix` 在
`t5_sampling_stage512b_real_v1.json` 里,而 catalog 的 `inputs.sampling_config_sha256`
绑定该文件且 **loader 强制校验**。动一个字节就会解绑 catalog,进而需要重建
catalog(89,786,866 行)+ anime_priority + plan + animefreq + 56 GB caption sidecar。
详见 S512B-14。

---

## S512B-02 数据池 = 精选 `rich_v1`,不是 full-survivor

full survivor manifest 缺 phase5 需要的 17 列里的 20 个(含 `src_*` payload locator),
复活 full-survivor 必须先建 `rich_v2`。Evan,2026-08-01。

## S512B-03 512b 身份 = 真实照片广度阶段,取代 anime 对齐设计

2026-06-29 锁定的 512b(anime 0.78 / 256M 曝光 / anime 精选池 ~20 epoch)**不再计划执行**。
Evan,2026-08-01。

## S512B-04 曝光 = 512M = 250,000 steps × global batch 2048(world 64 × bs 32)

## S512B-05 几何 = `bucket_mode=anchors_v1`,13 个固定锚,面积门 262,144

指派 `F(i) = {k : w≥W_k 且 h≥H_k}`,`argmax retention`,平局按 `(-area, anchor_id)`;
`F=∅` 或 `crop_loss > 上限` 则拒绝。判据是整数比较 `den·lo ≥ num·hi`,不用浮点。

## S512B-06 crop_loss 上限:全局 0.20,TextAtlas 专用 0.12

TextAtlas 抽样审计后收紧,防止裁掉渲染文字。Evan,2026-08-02。

## S512B-07 band_mix_real = [0.18, 0.27, 0.15, 0.40],real 合并口径,MONET 兜底(Case A)

`caption_band` 是 **0-based,`-1` 表示无 band**。plan 实测 t1 18.02 / t2 26.98 /
t3 14.99 / t4 40.00。

## S512B-08 GPIC 用**自己的** `cluster_sizes`,不借 MONET 的

借用是反向相关的(`corr(log,log) = −0.24`),会把有效簇从 16,798 压到 13,559。
Evan,2026-08-02。

## S512B-09 MONET `cluster_beta = 0.2`,接受由此产生的曝光漂移

## S512B-10 子源保留显式概率表;`source_alpha` 对 MONET 空操作;da4k 因未列而清零

实测曝光确为 0,与"排除 da4k"的意图一致。

## S512B-11 anime 池接受 +0.674% 漂移(全池精确值,不是估计)

## S512B-12 上采样护栏现在就加

`verify_binding()` 已接进 loader 的生产调用路径。此前它有**零个生产调用点**,
而 `multi_ar.resize_crop_to_bucket` 的 `scale = max(W/w0, H/h0)` **从不拒绝放大**。

## S512B-13 TextAtlas 权威 GT 只用两个源

CleanTextSynth 的 `annotation`(图像由该串渲染而来,构造上即 GT)和 TextScenesHQ 的
`raw_text`。**Qwen(i1)caption 不能当 GT**:引号覆盖率跨 subset 统一 93–98%,但对
权威 GT 只有 **73.5% 精确**,长串只引片段、有时引完全不相关的一段。它是转写不是记录。

**2026-08-02 追加决策**(Evan):GT **要真正进入训练 prompt**,方式是**独立的运行时
sidecar 按 `row_id` join**,不重建 56 GB caption sidecar。附带要求:
- 同时保留 `gt_text_raw` 与 `gt_text_normalized`(v1 只有 NFKC + 空白折叠后的版本,
  会改变全角/半角、圈号、数学字符和换行版式);
- 用**真实 Qwen3-VL-2B tokenizer** 统计长度,不再用"约 2,000 字符 ≈ 512 token"的估算;
- exact text 放在 prompt **前面**,防止尾部截断吃掉它;
- 超预算的样本**回退到原 caption**,绝不喂残缺 GT。

覆盖率 1,941,791 / ~4.12M TextAtlas 行 ≈ 47%,所以回退分支是常态路径,必须被 smoke 覆盖。

## S512B-14 曝光偏差:接受,但把**真实原因**写进记录

子源 L1 从目标 0.1240,且五个源里**四个方向反了**。真正的驱动是 **band 资格,不是
`cluster_beta`** —— megalith10m 是 0.00% T1、commoncatalog 是 1.00% T1,填不满
`band_mix_real` 要的 18%,planner 逐 cell 归一化只能把质量推给 laion/coyo/cc12m。
这是 caption 长度效应披着源权重的外衣。Evan 在**看到实测之后**接受。
详见 plan 旁的 `*.REALIZED_EXPOSURE.md`。

**结论必须写在产物旁边,不能写进采样配置** —— 见 S512B-01 末段。

## S512B-15 评测:启动前只钉协议,Stage512a 基线与训练并行跑

512a 权重已冻结,基线随时可测,不阻塞启动;但协议必须在启动前写死,
否则早期 eval snapshot 会用到后来会改的协议,失去可比性。Evan,2026-08-02。
协议见 **`STAGE512B_EVAL_PROTOCOL.md`**(`EVALUATION_PROTOCOLS.md` 只讲对外报告纪律)。
其中三项**阻塞启动**:本协议文件、固定人评图集入 git、OCR exact-string scorer
(当前 `evaluation/benchmarks/scorers/` 只有 clip 和 tagger,没有 exact-string ——
本阶段第一次把权威 GT 送进 prompt,却没有能证明它有效的度量)。

## S512B-17 行级重复:实测比 Stage512a 更平,不做任何调整

全 245 blocks 逐行计数,**512,000,000 次引用**(正好 250,000 × 2048)。
同一审计对**冻结的 Stage512a plan**(`7295ca93…`)跑了一遍作对照 —— 没有这个对照,
"GPIC 最热 10% 的行吃掉 47% 的 GPIC 曝光"这句话无法判断是缺陷还是常态。

结论:**512b 在每一项集中度指标上都比 512a 更平,而曝光预算翻倍。**
GPIC 最热 1% 从 10.16% 降到 5.67%,p99 从 61 降到 33;从未被抽到的行
monet_real 从 499,367 降到 128,411、synthetic 从 1,002,182 降到 63,752。
唯一没改善的是单行最大重复 123 → 127,按曝光预算折算是减半。
GPIC 的重复分布是**双峰**的(×3/×4 一大堆,再加 323,630 行正好 ×30),
512a 同样如此,是加权 planner 的性质,不是 13 锚改动带来的。
详见 plan 旁的 `*.ROW_REUSE.md`。

**顺带更正**:`*.REALIZED_EXPOSURE.md` 里写的 "513,802,240 row references" 是错的
(那是假设 245 个 block 全满;末块只有 144 步)。真值 512,000,000,
也只有对 512,000,000 该报告自己的 40.000% 才成立。该文件是只读的,更正写在
`*.ROW_REUSE.md` 里。

## S512B-16 第一次真实 GPU 跑 = 约 200 step 的独立目录 GPUVAL

不是 pilot,不用于判断模型质量。20 step 的 loss 噪声太大,看不出几何/归一化错配;
200 step 能对照 Stage512a `metrics.jsonl` 末段拿到 loss 连续性。Evan,2026-08-02。

**执行形态改为 100 + 续训到 200**(总步数不变):单次 200 步只证明"能存",
不证明"能续",而 `resume: true` 是这条 250,000 步跑线**全部**的重试恢复机制。
phase B 必须重载 model/optimizer/scheduler/EMA 并把 loader seek 到正确 plan step,
作业按 `train.py` 实际打印的那行断言 —— 否则"phase B 悄悄从 512a 重新初始化"
在任何下游指标上都与成功续训一模一样。

## S512B-18 TextAtlas 的 CFG dropout 维持 10%,不接通 `text_rich`

**先说事实**:`MoTT2IModel.forward` 读 `batch.get("text_rich")` 来应用
`text_rich_cfg_dropout_prob`,但**没有任何数据路径发这个键** —— offline loader 产出
`{prompts, pixels, cfg_keep}`,streaming loader 也一样。所以这个键**自写下之日起就是死的**,
512a 也一样,每一行 TextAtlas 一直走的都是普通 `cfg_dropout_prob`。

**决定不改行为**(Evan,2026-08-02)。理由:
- dropout 把**整条 prompt 换成空串**,所以被丢掉的那 10% 里 prompt 原本写了什么完全
  不影响结果 —— **加 GT 不改变这 10%**。GT 只加强剩下 90% 的条件监督,反而让
  cond/uncond 的差**变大**,CFG 外推更可靠而不是更差。
  (我最初以"加了 GT 所以更要紧"提出这个问题,那个因果是错的,已更正。)
- CFG 的无条件分支**按设计**就该学完整图像分布,其中包含带字的图。
  把整个 TextAtlas 从无条件分布里排除是另一个未经验证的改动。
  参考:CFG 原论文 arXiv:2207.12598 · Unconditional Priors Matter arXiv:2503.20240。
- 量级:TextAtlas 占 5%,10% dropout = 全局 0.5%;其中带 GT 的约 0.24%。
  收益不足以在文字通路上同时引入第二个变量(第一个是 GT 进 prompt),归因会变脏。

**但配置不能继续撒谎**:`text_rich_cfg_dropout_prob` 由 `0.0` 改为 **`0.1`**。
运行时逐位相同(`text_rich` 恒为 None,该值从不被读),改的只是"配置描述的是意图还是行为"。
`0.0` 会让下一个读者以为文字样本受保护 —— 它们不受保护。

**若 OCR exact-string 确实差**,实验是只对**有权威 GT 的那约 47% 行**比较 10%/5%/0%,
不是一上来保护整个域。

## S512B-19 GPUVAL 实测:GPIC 不需要建 blob;吞吐 2.57 s/step ⇒ 全程约 7.4 天

作业 `evan-s512b-gpuval-w64-20260802-v2-fzm7`,64 GPU、真 512a 权重、真 512M plan,
跑到 **step 420**(本该 100 就停,见下面的 bug)。

| 指标 | Stage512a 末 200 步 | Stage512b GPUVAL |
|---|---:|---:|
| loss | 0.8611 | 前 5 步 **0.8715**、稳态 **0.8701** |
| iter_s p50 | 2.618 | **2.570** |
| **data_wait_frac p50** | **0.048** | **0.0435** |
| peak_mem_gb | 17.76 | **17.76**(共 80) |

**最大的未知已闭合,而且是否定的**:我判定"GPIC 从打包 blob 改成逐文件 S3 FUSE、
读量 ×3.08,是正式跑变成 I/O bound 的最可能原因"。实测 data wait **4.35%**,
比 512a 带 blob 的 4.8% 还略好(日志确认 `gpic_payload_backend=source_files_v1`)。
**不需要建 512b GPIC blob。**

loss 比 512a 末段高 **1.2%** —— 与数据改动(13 锚、real 58%→60%、GT 进 prompt)相称,
远不到"2 倍 = 几何/归一化错配"的判据。forward/backward、FSDP2 64 rank、
optimizer/EMA 跨阶段、`init_from` 重置 step=0 + 全新 warmup、每 50 步存 checkpoint,
全部实测正常。

**唯一还没验的是 resume** ——因为下面这个 bug 导致两阶段计划没执行。

### 这次跑出来的两个 bug(都是我的)
1. **`trainer.max_steps` 是装饰性的**。`checkpoint.derive_max_steps` 在有
   `max_exposure_images` 时按"曝光量 / global_batch"派生 horizon,`trainer.max_steps`
   只作参考;**只有命令行 `--max-steps` 能盖过它**。train.py 自己打了
   `max_steps=250000 ... config.max_steps=100 仅作参考`。
   ⚠️ **正式跑要改 horizon 就得改 `max_exposure_images`**,单改 `max_steps` 看起来
   生效实际不生效。FORMAL 作业里已加断言:两者必须自洽。
2. **create-once 的 claim 作用域错了**(进程级而非作业级)。`mkdir || exit 45` 意味着
   **rank 0 的 pod 一重启就必死** —— 第二次尝试就是这么死的,撞上它自己 75 分钟前
   claim 的目录。更要紧的是:同样的逻辑放进正式作业会**摧毁它本要保护的恢复机制**
   —— `resume: true` 存在的意义就是 pod 重启后从 last.pt 续上。已改为记录 JOBTAG,
   同作业可重入,别的作业仍然 exit 45。

### 正式作业用 `RUN_MODE` 取代 claim
`fresh`(目录必须不存在)/ `continue`(目录必须已有 last.pt)。
由**操作者声明意图**,而不是让代码从磁盘现状去猜 —— 后者正是
"`resume: true` 撞上陈旧 last.pt 就静默续接"这个坑。且 `continue` 模式下作业会
**断言 train.py 打出的 resume 那行**,所以"重试时静默从 512a 重新初始化"会当场失败,
而不是白跑几天。这也是为什么"专门验 resume 的 GPUVAL"**不是启动阻塞项**。

## S512B-20 noise shift 保持 1.0 —— 这是**有意延续 512a 的既有偏离**,不是"注释写错了"

`networks/model_mot.py:10-11,272-278` 把 v3 决策记为逐分辨率固定
noise-shift:**256→1.0、512→2.0、1024→4.0**。实测三档 config:

| 阶段 | config `sigma_sampling.shift` | 与设计一致? |
|---|---:|---|
| 256 | 1.0 | ✓ |
| **512a** | **1.0** | **✗(设计要 2.0)** |
| 512b | 1.0 | ✗ |
| 1024 | 4.0 | ✓ |

256 与 1024 都对上了,**只有 512 这一档从 Stage512a 起就是 1.0**,而且已经这样跑完了
整个 125,000 步(GPUVAL 日志确认 512b 继承 `shift 1.00`)。所以注释**不是陈旧的** ——
是 512 的 config 从 512a 开始就偏离了设计。

**决定保持 1.0**(Evan,2026-08-03)。理由:本阶段只改数据/几何/日程,动 shift 会
引入第二个难归因变量,并让 GPUVAL 赖以判断的"与 512a 末段 loss 对比"失去意义。
"512 该不该是 2.0"是**留给后续阶段的真问题**,要配自己的 GPUVAL —— 现在只记录,不修。

## S512B-21 caption sidecar 的 production 与 work 是 hardlink,已双向锁只读

外部审计实测:1,372 个 caption parquet 全部 `nlink=2`,production 与
`offline_arb_caption_sidecar_stage512b_work` **共享 inode** —— 原地改写 work 那份会
同时污染训练数据。这**是发布协议的设计**(`build_caption_sidecar.py:1511` 用 `os.link`
发布),不是意外。

处理(Evan,2026-08-03):文件 `0444`、production 目录 `0555`。
**关键:文件 mode 在 inode 上,所以两份一起变只读;目录 mode 不共享,只锁 production。**
这正好是想要的 —— 两份都不该被原地改写,而目录门只拦 production 的增删。
实测改后 loader 仍能正常读(65,536 行)。
一如既往:**我们是 root,chmod 只拦误写不拦故意写**,真正的保护是代码里的 create-once。

## S512B-22 固定人评图集已批准冻结

`evaluation/benchmarks/prompts/stage512b_human_eval_v1.jsonl`,42 条,覆盖全部 13 锚,
seed 唯一。Evan 2026-08-03 批准。
**sha256 `6005219a077cdf63ca2ef047f98ef2e8b6af0be8dc752d63b9a3f41c8175b39a`。**
此后不得增删改 —— 第一次 snapshot 之后动它就毁了它存在的意义(可比性)。

## S512B-23 GPT 独立审计(2026-08-04)的逐条处置

审计对象 `3f17d46..9a105ed`,结论 NO-GO。**逐条实测后:7 条是真 bug 已修,3 条是我
陈述过头已纠正,2 条判定为过度设计不做。**

### 成立并已修(commit `f9b9f5f`)
| 编号 | 实测证据 | 修法 |
|---|---|---|
| **F-01** | torchrun 只有 `--config`,无 `trainer.checkpoint_dir` | 补上。**本轮最严重** —— 门禁守 `CKPT_DIR`、训练写 `PLACEHOLDER_…`,且它是 F-08 的根因:placeholder 目录永远没有 `last.pt`,所以**任何** continue 都会静默 init_from |
| **F-04** | 512a 目录既无 `.jobtag` 也无 `FORMAL_RUN.json` | continue 必须有本阶段 schema 的 `FORMAL_RUN.json` |
| **F-10** | `jcc_preflight.sh 9 .` → **rc=0** | 要求 40 位小写 hex + 全等 |
| **F-16** | reader 的 `_header` 行噎死 scorer;`--self-test` 被 required 参数挡 | OCR 两半此前**从未真正对接跑过**;已修并加 engine 交叉校验 |
| **F-15a** | `gt in prompt` 会碰巧通过(纯 caption 自带 raw GT 的期望命中 69.5%) | 改成必须匹配三个规范从句前缀 |
| **F-15b** | 反例(日文 caption + 142-token GT)复现 474/473/474 > 471 | 对成品重新 tokenize、只缩 caption 尾巴 → 467/466/467,2484 组合零越界 |
| **F-03 / F-09** | `test ! -e` + `mkdir -p` 是 TOCTOU;`AUTO_RESUME_RECONCILE` 未开 | 裸 `mkdir` 原子 claim;打开该环境变量 |

### 我自己陈述过头,已纠正
- 显存"17.76 of 80"是 **torch allocated**;`nvidia-smi` 峰值 **42,317 MiB**。
- "首 batch 前验证全部 artifact digest" —— 实际只验存在性、强制的 sampling config 摘要、
  plan/catalog 绑定。已改写成实际所做的事。
- 评测协议里"`checkpoint_steps × keep_max` 滚动窗口" **是错的**:`checkpoint_steps`
  只覆盖同一个 `last.pt`,step 30,000 前**没有任何**可回溯 full checkpoint。

### 判定为过度设计,不做(Evan 2026-08-03 同意)
- **F-02 attempt nonce + lease + 旧 writer fencing**:陈旧 RDZV 的后果是新 peer 连到
  已死的 master ⇒ **作业失败,不是错误训练**,`maxRetries: 3` 能吸收。为一个"失败而非
  错误"的模式加三层协调机制不值。
- **F-05 每次启动前重算 56 GB caption SHA**:hardlink 已双向锁只读(S512B-21);
  plan 245/245、catalog 5/5、animefreq 17/17、GT 六列现物均已实测通过;
  loader 边读边校验逐文件摘要。每次 continue 重跑数小时哈希不划算。
- 同理不做:独立只读运行快照 + 镜像 digest + 包 manifest(F-10/F-11 的扩展部分)。
  对内部跑记录即可。

### 审计确认正确、不必重验的(直接引用,省下一轮)
plan 245 blocks 逐块 SHA 全通过 · catalog 5 parquet SHA 全通过 · animefreq 17/17 ·
GT 正确六列 1,941,791 行且 header/payload/`_SUCCESS` 闭合 · 10,880 个 tar locator 全存在 ·
world=64 / per-rank 32 / global 2048 / 250,000 steps / 512M 曝光一致 ·
plan 发牌与 reader 反变换闭合、resume seek 无 off-by-one · 真用 FSDP2(28 units)·
512a init checkpoint 的 trainable/optimizer/EMA 各 483 key 完整 · 8 个快照确实全保留 ·
FSx 空间充分(约 5.4 TiB 可用,稳态需约 132 GiB)。

### 仍未闭合(只受容量限制)
**F-07 容量**:8 个全空 8-GPU 节点当前为 0。
**F-13 真两阶段 resume GPUVAL**:代码已修好并有 19 项穷举测试,但没在 8 节点上真跑过。
其余启动阻塞项已全部关闭。

## S512B-24 真两阶段 resume GPUVAL 通过(F-13 闭合);正式跑必须 80 核,48 核不行

作业 `evan-s512b-gpuval-resume-20260803-v6-xgmr`,commit `fcf44f7`,8 节点全到,
phase A 跑到 100 → 停 → phase B 续到 200。**Succeeded,`GPUVAL_POSTCHECK_PASS`。**

| 检查 | 结果 |
|---|---|
| `--max-steps` 覆盖 | ✅ train.py 打印"显式 --max-steps=100 优先于 max_exposure_images" |
| phase A 干净停在 100 并写出可续训 `last.pt` | ✅ |
| **phase B 真续训** | ✅ `resumed from …/last.pt at step 100` |
| **scheduler 跨边界连续** | ✅ LR step100 `3.498743e-05` → step101 `3.533384e-05`,正好一步 warmup 增量,不是重置。**这比日志断言更硬** |
| metrics 跨 resume | ✅ 200 行全唯一、单调、**零重复** |
| 周期快照 | ✅ `step_100.pt` + `step_200.pt` + `last.pt` |
| loss vs 512a 末 200 步 | ✅ 比值 **1.012** |
| 显存 | ✅ 17.76 GB(torch),两次加载都稳 |
| 8 节点到达屏障 | ✅ 真实错峰下工作 |
| preflight 40 位 commit 全等门 | ✅ |

⚠️ **但这次是干净停止,不是崩溃。** F-09 关心的"崩溃导致 metrics 跑在 checkpoint 前面"
那一支**没被触发**;它靠 FORMAL 里刚打开的 `OFFLINE_ARB_AUTO_RESUME_RECONCILE=1` 兜底,
而该路径仍未在真崩溃下验过。

### 资源降配的实测结论:**48 核不可用于正式跑**
为改善可调度性把 GPUVAL 降到 `cpu 48 / mem 600`,量出来:

| | 420 步那次(80/900) | 这次(48/600) |
|---|---:|---:|
| `iter_s` p50(稳态,排掉两次冷启动) | **2.570 s** | **4.440 s** |
| `data_wait_frac` p50 | 0.0435 | 0.029 |
| `data_wait_s` **max** | — | **43.53 s** |
| `iter_s` max | 6.17 s | **56.36 s** |
| 折算 250,000 步 | **7.4 天** | **12.8 天** |

**中位步时几乎翻倍(1.73×)。** 注意 `data_wait_frac` 的**中位数反而更低**,是**尾部**炸了
(max 0.85、单次停顿 43.5 秒)—— 48 核要喂 16 gpic 线程 + 16 caption worker + 8 fetch
worker,超订后间歇性饥饿。可能还叠加了失去节点独占(48/600 能被塞进已有作业的节点),
**两个原因我没有分离开**,但结论一样:**FORMAL 保持 `cpu 80 / mem 900`**。
(这也正是审计说"不能靠假设砍资源"要求的那个测量。)

## S512B-25 第一次正式提交失败:容量 + 我的第七个同类 bug(JOBTAG 跨重试会变)

`evan-s512b-real-formal-w64-v2-5dlx`,2026-08-04 提交。两次失败,原因**不同**:

```
原作业:  FATAL: only 5/8 nodes arrived within 5400s        ← 容量,到达屏障按设计工作
重试:    FATAL: RUN_MODE=fresh but … is owned by
         'evan-s512b-real-formal-w64-v2-5dlx'              ← 我的 bug
```

**JCC 的重试会重命名 pod**:`…-v2-5dlx-N` → `…-v2-5dlx-retry-7mrd-N`,所以
`JOBTAG=${HOST%-*}` **跨重试不稳定**。我在注释里写了"JOBTAG 跨重试不变",那是个
**未经验证的假设**,于是 `maxRetries: 3` 仍然形同虚设 —— 同一个机制,第二次在它上面栽。

### 修法:所有权用 `RUN_ID`,rendezvous/arrival 继续用 JOBTAG
- **`RUN_ID`**(操作者提供,写在 job spec 里 ⇒ 跨重试**天然稳定**)决定目录所有权,
  写进 `$CKPT/.jobtag`。
- **JOBTAG**(pod 派生,重试即变)继续给 rendezvous 文件名和 arrival 目录命名。

**这个拆分两个方向都对,而且免费解决了审计的 F-02**:重试**不该**沿用上一次的
master 地址或到达标记,而 JCC 自己的重命名恰好提供了逐 attempt 隔离 ——
不需要 nonce 协议。审计要的效果拿到了,复杂度没进来。

### 测试:22 项(原 19 + 3)
新增"重试时 JOBTAG 与 RUN_ID 不同"的两个用例,外加一条**反向守卫**:
若有人把所有权改回按 JOBTAG 判定,该用例会翻转、测试套件失败,
而不是那个七天的作业失败。
⚠️ 我第一版新增用例把 `JOBTAG` 和 `RUN_ID` 设成了同一个值 —— **场景根本没被触发**,
又是一个"不会失败的检查"。已修成用真实的 retry pod 名。

### 运维参数教训
`ARRIVAL_WAIT_S=5400` 意味着先到的 5 个节点空转 90 分钟才退出,`maxRetries: 3`
会把这个乘以 4。在 63/90 节点满载的集群上这既不友好也不划算。
**下次用 1800(30 分钟)**:失败得快、节点还给别人、靠重试去碰"8 个同时空闲"的窗口。

## S512B-26 正式跑在 step 660 被自己的门禁杀死;anime metadata 漂移的 crop_loss 降为可容忍

**2026-08-04 06:13 启动,07:10 死。** 死因不是集群、不是 OOM,是 loader:

```
step=660 rank=12 slot=3 row_id=3902543 domain=anime
catalog 1032x1457 → plan 冻结 anchor 2 (432x576);实际解码 668x594
AnchorGeometryError: crop_loss 0.333084 > 0.2
```

anime 域**按设计**允许 upstream 的 width/height metadata 与当前像素不一致(HTTP 回源,
代码里那条容忍分支明说"legacy streaming loader trusted decoded pixels"),Stage512a
就这样跑完了 125,000 步。512b 把 `anchors_v1.verify_binding()` 接上生产(此前**没有任何
生产调用者**),它按**解码尺寸**重算 crop_loss,于是一条容忍分支已经放行的行变成了
杀作业的异常。

**这不是边缘情况:** 660 步 × 2048 = 1,351,680 个已解码样本里 1 次致命 ⇒ 7.4e-7;
512M 曝光期望命中 **~380 次**。这个配置**不可能跑完**,与集群状况无关。

### 处置:只对「anime 且不需上采样」这一条路把 crop_loss 降为 warning + 计数 + 熔断
- 新增 `AnchorCropLossError(AnchorGeometryError)`,**只**表示 crop_loss 超阈。
  上采样 / anchor 越界 / 尺寸非法仍抛基类,仍 fail-closed。**必须分型**:基类捕获
  会把上采样一起放过去,而上采样是整个锚设计承诺永不发生的那件事。
- 理由:`crop_loss_max` 是**语料组成**约束,归属离线 plan 建库阶段。运行期 1e-6 比例的
  行切狠一点对组成没有可测影响;而它对 7 天作业的杀伤是确定的。
- 熔断上限 **2000 / rank**(`_ANIME_DRIFT_CROP_TOLERATED_MAX`):按实测率每 rank 期望
  ~6 次(单 rank 全程解码 8,000,000 行),留 ~340 倍余量;而"row→file 映射系统性错误"
  这种真故障会在几分钟内撞穿它,而不是让我们对着错配的图文训练七天。**每一次都打印**,
  不是静音开关。

### 刻意**没有**做的三件事(避免过度设计)
1. **没有加 crop_loss 二级上限。** 极端情况(切掉 90%)全程期望约 380 行,占
   512,000,000 曝光的 7e-7 —— 保护不了任何可测的东西,只增加一个旋钮和一条失败路径。
2. **没有做同 anchor 换行。** 会破坏 plan 的精确曝光(512,000,000 这个数字不再成立),
   且要新写"捐赠行"选择逻辑 —— 新代码、新失败模式,而它要在七天里被触发 ~380 次。
3. **没有改成用解码尺寸重算 anchor。** 做不到:一个 step 内所有 slot 必须同分辨率才能
   拼 batch,改单行的 anchor 会让 collate 直接碎。

### ⚠️ 残留风险(同一类,**未修**,已知且刻意保留)
anime 行若漂移成**小于**目标锚的尺寸,走 `_validate_anime_source_bound_native_mismatch`,
随后 `verify_binding` 的上采样检查抛**基类** ⇒ **仍然杀作业**。
- 实测 660 步内 2 次漂移**都是** "without upsampling",这一路 0 次命中(< 1e-6)。
- 不修的理由:容忍它就等于训练在凭空生成的像素上,破坏"零上采样"这个载重不变量;
  而 512a 早就遇到过这一类,机制是 plan 里冻结的 `anime_native_upsample_exception`
  白名单(离线处理)。**行为与 512a 一致,不是新增风险。**
- 若它在正式跑期间命中:把该 row 加进冻结白名单后 continue 续训。GPUVAL 与正式跑都会
  用 WARNING 计数报出这一路的真实频率。

---

## S512B-27 慢的问题:我把 p50 当平均值用了;GC 假设已是一个月前查实的病根

### 我报错的 ETA
| 依据 | 步时 | 250,000 步 |
|---|---|---|
| 我报的 p50 | 2.909 s | 8.4 天 |
| mean(step 100–600) | 4.430 s | 12.8 天 |
| **纯墙钟**(step 1→640) | **4.920 s** | **14.2 天** |

`sum(iter_s)=3146 s` vs 墙钟 `3144 s` ⇒ 时间全在 step 内部,没有隐藏开销。
**右尾重(p90 = 8.4 s)时墙钟跟均值,不跟中位数。** 报 ETA 必须报 mean。

### 我第二个错:拿 512a 整段 125,000 步比 512b 的头 650 步
同龄(step 100–650)对比:512b formal mean/p50 = **1.66**、512b gpuval v2 = **1.66**、
512a v1 = **1.31**。而 512a 自己的尾巴**会消退**:step 2100–3099 降到 1.12,
4100–5099 到 **1.04** —— 且全部发生在第一次 re-freeze(step 10,000)之前,
所以那是**启动暖机**。⇒ 14 天是从暖机窗口外推的,真实值大概率好得多。
512b 同龄确实比 512a 差(1.66 vs 1.31),这一条是真的。

### GC:用户的判断对,而且这个仓库一个月前就查实并修了
`train.py` 的注释(07-03 第三轮 / 07-05 第四轮)写着实测根因:数据线 producer 线程的
10^5~10^6 级对象分配风暴 → gen2 全堆收集每 ~21 步一次,单次停顿随堆基数线性增长
(fresh ~0.4s → 56k 步 ~4s),**任一 rank 停顿即拖全体 FSDP collective**。
三个开关 `gc_gen2_threshold: 200` / `gc_freeze_after_steps: 50` / `gc_refreeze_every: 10000`
**512a 与 512b 配置里都开着、值相同**。文档记的创变尺度是 10^4–10^5 步,我们只跑了 650 步,
所以**我测到的尾巴主体不可能是 GC 创变**。`gc_refreeze_every` 暂不动 —— 盲调会让下一跑
无法归因。

### 真正的修法是补上监测盲区,不是再猜一轮
`metrics.jsonl` **只由 rank 0 写**,而这个失效模式的传导方式恰恰是:别的 rank 停顿 ⇒
**我们自己的 collective 变长、我们自己的 data_wait 不变**。我因此把
"慢的不是数据线"这个结论下得比证据允许的硬(慢步 dw 0.115 vs 快步 0.120)。
- 新增 `trainer.gc_pause_probe`(gated,默认关,512b 开):`gc.callbacks` 逐 generation
  直测每次回收的停顿,跨 rank all_reduce MAX 后写进 metrics。
- 同时新增 **`data_wait_allrank_max_s`** —— 最饿的那个 rank,不是 rank 0。这是补盲区的那一条。
- `gc_allrank_freeze_count_min` 取负后求 MAX,得到跨 rank **最小**值:能发现某个 rank 根本没冻结。
- 判据留给下一跑:慢窗口 vs 快窗口的 gen2 停顿均值有显著差 ⇒ 是 GC;齐平 ⇒ 在别处。

**教训**:07-03/07-05 两轮把 GC 修好了,却**从没让跑起来的作业量过一次停顿**,
于是残余尾巴只能靠推断吵架 —— 三种解释在 rank0 指标上全都成立。
修了病,没装体温计,下次发烧还是只能猜。

---

## 流程约定(不是数据决策,但同样具约束力)

- **建完 catalog 后不得编辑 `t5_sampling_stage512b_real_v1.json`**,连注释都不行。
  对比:`stage512b_real_anchors_v1.yaml` 的 `pipeline_config_sha256` **只记录不强制**,
  其漂移是结构性的(plan root 必须等 plan 建好才写得进去)。
- **不得在工作树脏的时候提 JCC 作业** —— 作业读活工作树,已经因此白跑过一次。
- **`chmod a-w` 挡不住 builder**(Lustre 无 `chattr +i` 且我们以 root 运行,实测写入
  仍成功)。唯一有效的保护是代码里的 create-once 门。
- **`checkpoint_dir` 必须是全新、create-once 的目录**。`resume: true` 在目录已存在时
  会**静默续接**陈旧 checkpoint 而不是从 Stage512a 初始化。
