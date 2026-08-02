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
