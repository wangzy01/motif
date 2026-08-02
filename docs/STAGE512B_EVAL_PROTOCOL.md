# Stage512b-real 评测协议(启动前钉死)

**为什么必须在启动前写死**:早期 eval snapshot 一旦用了后来会改的协议,就永久失去与
后续 snapshot 和与 Stage512a 的可比性。协议本身不占 GPU,写下来是零成本;而 Stage512a
基线因为权重已冻结,**可以与 512b 训练并行跑,不阻塞启动**(Evan,2026-08-02,S512B-15)。

本文件只钉**协议**。跑不跑、什么时候跑,见末尾"执行安排"。

---

## 1. 继承自 Stage512a 的严格协议(逐字节相同,不得改动)

来源:`/fsx/evan/mot_code/data_analysis/eval_releases/stage256_offline_exact_v1_vs_prev_final_20260718a/scripts/strict_eval.py`
(sha256 `0afc7e6a8c0b9a9751ff9c7ee48edbdedc6c3b0847169a618d3ad3569235babc`),
由 `evaluation/reference/stage512a_final_20260728/scripts/strict_eval_stage512a_final.py`
做 512px 包装。**该 evaluator 在 git 之外**,与两个 plan builder 是同一类问题;引用它必须
连 sha 一起引。

实测采样参数(从上述源码读出,不是记忆):

| 参数 | 值 |
|---|---|
| 分辨率 | 512 × 512 |
| steps | 64 |
| CFG | 4.0 |
| seed | GenEval 7;DPG `7 + start`(逐 shard 递增) |
| `model.cfg_dropout_prob` | 评测时强制 **0.0** |
| 架构开关 | 有 fingerprint 校验,不匹配直接 fail |

**512b 用完全相同的这组参数。** 任何一项改动都会让"与 512a 比较"这句话失去意义。

## 2. 五条评测轴

### 2.1 Stage512a 同协议对照(**主判据**)
同一 evaluator、同一参数,对 `stage512a_20260724_weighted_w64_recovery_v2/last.pt`
(step 125,000,sha `957bfcf1…`)与 512b 各 snapshot 各跑一次。
这是唯一能回答"512b 有没有变好"的东西;其余四条都是分项诊断。

### 2.2 GenEval 多实体分项
不看总分。**按 category 分开报**(counting / position / colors / color_attr /
two_object / single_object)。总分会把"多实体计数变好、单体变差"抹平,而本阶段把 real
广度从 58% 抬到 60%(S512B-01),最可能动的正是多实体那几项。
资产:`evaluation/reference/stage512a_final_20260728/scripts/geneval_generate_shard.py`
+ `merge_geneval_shards.py`。

### 2.3 OCR exact-string —— **当前没有 scorer,需要新建**
`evaluation/benchmarks/scorers/` 只有 `clip_score.py` 和 `tagger_score.py`。
本阶段第一次把权威 GT 送进 prompt(S512B-13),没有 exact-string 指标就等于
**改了文字通路却没有能证明它有效的度量**。

协议要求:
- prompt 取自 **TextAtlas GT sidecar v2 的 `gt_text_raw`**,不是 normalized ——
  训练目标是 raw,评测必须同口径,否则全角/圈号那一类差异会被 NFKC 抹掉再算成正确;
- 报 **exact match** 与 **归一化后 exact match** 两个数(后者用与 builder 相同的
  NFKC + 空白折叠),差值本身就是"排版类错误"的量;
- 固定 prompt 集、固定 seed,集合一旦定下不得增删;
- 只用 CleanTextSynth / TextScenesHQ 两个权威 subset —— 其余七个没有可提取的 GT,
  拿 i1 caption 里的引号当答案会把 26.5% 的错答算成对答(它对权威 GT 只有 73.5% 精确)。

### 2.4 非方形 prompt fidelity
Stage512a 的评测协议**只跑 512×512**,所以这条轴按定义**没有现成基线**。
但 512a 权重能在任意尺寸采样,所以基线是可以补的:用同一 evaluator、同一参数,
在 **13 个锚**上各跑一遍 512a 与 512b。
资产:`evaluation/reference/stage512a_final_20260728/scripts/score_multiar_final.py`。
这条轴是本阶段几何改动(361 桶 → 13 锚)唯一的直接度量。

### 2.5 固定人评图集
一组**永不变动**的 prompt + seed,每个 snapshot 出同样的图。
不打分,只并排看。自动指标对"塑料感""构图崩""文字歪但字对"这些是盲的。
集合必须在第一次 snapshot 之前定下并写进 git。

## 3. 报告纪律

- 内部 DPG-Qwen 与官方 DPG/mPLUG **不可直接比较**(见 `EVALUATION_PROTOCOLS.md`)。
  任何对外比较必须同时报:分辨率、steps、CFG、prompt 原文、是否做过 prompt 改写、
  scorer 身份、聚合规则。
- 每次评测产出一份 receipt:checkpoint 路径 + sha256、evaluator sha256、
  参数表、prompt 集 sha256、原始逐样本分数。只有聚合分的报告不算证据。

## 4. 执行安排

| 何时 | 跑什么 | 阻塞启动? |
|---|---|---|
| 启动前 | 本文件 + 固定人评图集入 git + OCR scorer 写好 | **是**(只有这三项) |
| 与训练并行 | Stage512a 全五轴基线(权重已冻结,随时可跑) | 否 |
| 训练中 | 512b snapshot 五轴 | 否 |

### snapshot 取点(**待确认**)
trainer 配置是 `checkpoint_steps: 3000`、`checkpoint_keep_max: 4`、
`checkpoint_keep_every: 60000`,所以**永久保留**的只有 60,000 / 120,000 / 180,000 /
240,000,加最终 `last.pt`(250,000)。滚动 checkpoint 只留最近 4 个 = 12,000 步窗口。

建议取点:**60k / 120k / 180k / 240k / 250k**(= 122.9M / 245.8M / 368.6M / 491.5M /
512M 曝光),外加一次早期健全性评测,用 step 6,000 左右的滚动 checkpoint —— 但那个必须
在它滚掉之前(约 step 18,000 前)跑掉,否则就没有早期观测点了。

若希望早期观测点更从容,唯一改动是把 `checkpoint_keep_every` 调小(例如 30,000),
这要在启动前决定,启动后再改不会追溯保留已经滚掉的 checkpoint。
