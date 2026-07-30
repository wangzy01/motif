# Stage512b-real 研究简报

状态：方向草案，不是可执行训练规范。

## 目标

在 Stage512a 最终模型上继续训练一个更强的通用 real T2I 版本，扩大真实
世界题材、视觉概念和精确文本描述的覆盖，使其在公平协议下达到或超过相近
参数体量的主流模型。Anime 专项增强不再是本阶段的第一优先级，可留给后续
独立阶段。

## 已确认的起点

- 网络架构完全继承并冻结 Stage512a；本阶段优先研究数据、训练日程和评测。
- 初始域 mix：
  `Anime 32 / MONET 44 / GPIC 14 / Synthetic 5 / TextAtlas 5`。
- real 合计 58%。`MONET 44 + GPIC 14` 是当前兜底，不是永久限制；若发现
  新的高质量 native real 来源，可以提出 real 58% 内的重分配，须由用户
  批准后才能实施。
- TextAtlas 保持 5%。文字能力提升应同时考察真实文字图、精确 caption、
  结构化 synthetic、CFG 行为和 OCR/排版评测。
- real 严格不上采样。规模以经过过滤、去重且能由某个 native AR bucket
  服务的样本数为准，不设 150–200M 硬目标。
- exposures、steps、gates 和指标阈值暂不固定；需以 benchmark 调研和
  Stage512a 可比基线为依据另行提案。
- 未来只有在用户批准正式方案后，才允许以
  `/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage512a_20260724_weighted_w64_recovery_v2/last.pt`
  （SHA-256
  `957bfcf12b89c4042a32056566012a2451e72aa0b2706b0ca51808ab79155c31`）
  作为初始化候选。启动前必须重新走 JCC full rehash、mmap/load、
  architecture、optimizer、scheduler 和 full-state 独立审计；禁止
  `weights_ema`、旧 W48/6×8/row11163647 恢复链和已消费 replay。

## 已知数据事实与发现线索

以下 survivor 数量和路径是截至 2026-07-30 的规划估计/发现线索，不是
Stage512b-real 已冻结的输入身份。正式纳入前必须由 fresh JCC 作
metadata-only 全量审计，输出 exact unique、native AR bucket 可服务量、
输入 manifest SHA、过滤/去重版本和审计回执；不得直接把本节数字写进正式
plan。

- 当前 60M 精选 real 母池来自 MONET 40M + GPIC 20M。
- 既有过滤/去重 survivor 约：
  - MONET：73,820,347
  - GPIC：81,915,002
  - 合计：155,735,349
- 严格 `min(width,height) >= 512` 约：
  - MONET：57,078,519
  - GPIC：4,239,912
  - 合计：61,318,431
- 上述 61.32M 只是保守的 square-style 下界。256–511 边长样本中有多少
  可由 Stage512 的 adaptive AR bucket 原生服务，必须以后通过 JCC 做
  metadata-only 全量审计，不能用原始规模猜测。
- TextAtlas（截至 2026-07-30 的已观察路径，仅作发现线索）：
  - 图像源：`/s3/evan/textatlas5m`
  - caption sidecar：`/fsx/evan/textatlas_i1_captions/textatlas`
  - manifest：`/fsx/evan/textatlas_manifest_v1`
  - Stage512a 实际可用约 4.037M。

real survivor 的代码发现入口包括
`/fsx/evan/mot_code/data_analysis/scripts/t5_prep_survivor_pool.py`、
`t5_monet_select.py`、`t5_gpic_select.py` 和 `t5_merge_manifest.py`；
这些是定位候选源及历史规则的线索，不等同于已冻结 artifact。fresh JCC
必须从现场找到实际 manifest/sidecar，记录其绝对路径与 SHA 后再报告数字。

新增 real 来源在进入设计前至少应说明：来源/许可、native geometry、
caption 质量与语言、NSFW/水印/低质过滤、源内及跨源去重、benchmark
污染、预计有效 unique 和存储成本。不得只报原始 URL 或下载条目数。

## 下一轮调研

### 1. 可比模型与协议

建立来源可追溯的对比表，至少核验 Cosmos3-Edge、nanoT2I、MONET 论文模型
及其他相近体量、较新的通用 T2I。不要把“Cosmos3 2B”或“nanoT2I”当成
无歧义名称；区分：

- Cosmos3-Edge：
  <https://huggingface.co/nvidia/Cosmos3-Edge>
- nano-t2i：
  <https://github.com/gojasper/nano-t2i>
- MONET 论文：
  <https://arxiv.org/html/2605.21272>

- 生成器参数与包含冻结文本编码器的在线总参数；
- 训练分辨率、阶段、总 exposures/tokens、synthetic 比例与 caption 流程；
- raw prompt 与 prompt rewrite/upsampling；
- GenEval、DPG、OCR、FID/人评所用的官方实现、采样器、CFG 和 sample 数；
- 公开权重、论文数字和第三方复现之间的差异。

当前 Stage512a 的 internal DPG-Qwen 不能冒充官方 DPG/mPLUG。若协议不同，
先补同协议基线，或在报告中明确标记不可比。

### 2. Real 池

用 JCC 对现有 survivor 做 native AR 可服务审计，输出每个来源、bucket、
过滤阶段和 caption 质量层的 exact unique 数。另行只读调研新的高质量
real 来源；候选纳入 real 58% 前提交去重、质量、许可和污染证据。

### 3. 能力定向

针对 Stage512a 在 counting、双对象、属性绑定和文字渲染上的弱项，比较
以下方案的边际价值：更精确的真实 caption、real text-rich 子层、结构化
synthetic、难例重采样以及 CFG/text-rich 处理。TextAtlas 总占比仍为 5%，
任何新的子层定义和权重先提案。

### 4. 预算与评测

根据可比模型训练量、现有模型学习曲线和数据重复率，提出 exposure 上限和
中间 gates，而不是沿用旧 Stage512b 的 128M/256M 数字。评测至少应同时
保留 raw-prompt 主结果、明确标注的 rewrite 辅助结果、同设置定性图和
文字/OCR 专项；最终指标与人评门槛由用户审批。

## 进入执行前必须交付

1. benchmark 与数据证据表；
2. native 可服务 real inventory 及质量/去重摘要；
3. 候选 mix、预算、gates、评测协议和预估成本；
4. 与 Stage512a 相比仅发生哪些变化；
5. 风险、回退点及一页用户决策清单；
6. 新 config/plan/release 的候选名称，确保不覆盖旧身份。

这些内容经用户批准并经过独立静态/GPUVAL 护栏后，才进入构建或训练。
