# MoT 零上下文 AI 交接提示词

你正在接手一个 MoT 文生图训练项目。请先完整阅读：

- `/fsx/evan/doc/mot/current/README.md`
- `/fsx/evan/doc/mot/current/STAGE512A_FINAL.md`
- `/fsx/evan/doc/mot/current/STAGE512B_REAL_BRIEF.md`
- `/fsx/evan/doc/mot/current/WORKSPACE_ARCHIVE_INDEX.md`

只有以下两层门同时闭合，本目录才是导航权威：先确认
`/fsx/evan/doc/mot/current/PUBLICATION_MANIFEST.json`、同目录
`_SUCCESS`、逐文件 SHA，以及 manifest 绑定的 v5 target-specific
directory lock、intent、completion 和 prepared hardlink twin 全部通过；
再确认 fresh 独立只读验证回执
`/fsx/evan/mot_code/data_analysis/doc_publication_verify_20260731_v5/REPORT.json`
及同目录 `_SUCCESS` 均为 PASS。缺任一项时 current 仍只是候选发布，不得
据此启动状态变更。

满足上述条件后，远端 FSx current 文档是权威；Mac 本地文件只作缓存。
不要把历史文档或旧 `stage512b` YAML 当作可执行方案。

## 项目现状

Stage512a 已以 8 nodes × 8 H100、global batch 2048、125,000 steps /
256M exposures 完整训练结束。最终架构包含 2 层 causal text adapter 和
12 层 MLF；最终 full-state checkpoint、源码、配置、离线计划和审计身份见
`STAGE512A_FINAL.md`。

常用远端入口：

- SSH：`CPU-AWS-0623-v2`
- JCC CLI：`/fsx/evan/bin/jcc`
- MoT 代码：`/fsx/evan/mot_code`
- 数据分析：`/fsx/evan/mot_code/data_analysis`
- 当前文档：`/fsx/evan/doc/mot/current`

下一阶段暂称 **Stage512b-real**：从 Stage512a 的最终能力出发，优先把
模型推进为能对标相近体量、主流且较新的通用 T2I 模型的 real 版本。用户
提到 Cosmos3、nanoT2I/MONET 作为调研线索，但名称、参数口径、数据规模和
评测协议都必须查证，不能据印象下结论。

本阶段先调研和设计，不启动数据构建、评测或训练。首先完成：

1. 对相近体量的最新主流 T2I 做来源可追溯的横向调研，区分可训练生成器
   参数、在线总参数、文本编码器、分辨率、训练 exposures、prompt rewrite
   和正式评测协议；
2. 审计现有及潜在 real 来源在各 native AR bucket 下的可服务规模、质量、
   去重、caption、许可与污染风险；
3. 提出数据组成、训练预算/gates、公开评测和人评方案，并把仍需用户拍板
   的选项单独列出；
4. 更新 compact 文档和私有 `motif` Git 审查分支，使代码、数据构建和评测
   入口清晰可复现。

## 已锁定边界

- **架构冻结**：本阶段不得擅自改变 Stage512a 网络结构、生成头、adapter
  或 MLF。若调研发现架构问题，只提交证据和独立提案。
- 初始域配比为
  `Anime / MONET / GPIC / Synthetic / TextAtlas = 32 / 44 / 14 / 5 / 5`。
  其中 real 合计 58%。若发现新的高质量 native real 来源，可提出 real 58%
  内的重新分配，但执行前必须得到用户批准。
- TextAtlas 固定 5%。提升文字能力不能简单等同于反复采样 TextAtlas；
  应研究真实文字图、精确 caption、结构化 synthetic 和可比较 OCR benchmark。
- real 图像严格禁止上采样。原先“150–200M”不是硬目标；应在 native
  可服务、质量、去重和许可均成立时尽量扩大，不得为了数量降质。
- 训练 exposures、步数和指标阈值暂不写死；先完成可比模型和协议调研，
  再向用户提交有依据的预算与 success gates。
- 本阶段的设计应尽量只改变数据和训练日程，避免同时改动多个难以归因的
  变量。

## 操作边界

- 所有耗时、大内存、大 I/O、全量存储扫描、checkpoint hash/copy 必须走
  JCC；登录节点只做小型只读检查。`/fsx/evan` 保持低于十进制 1 TB。
- 提交动作前检查 `jcc job list`、目标路径和现有产物；已有同一动作时只
  监控，绝不重复提交或覆盖。未知漂移、半成品或确定性错误 fail-closed。
- FORMAL/GPUVAL 禁止 `jcc job get`。自动重试最多 `maxRetries:3`，同一
  checkpoint 身份最多一次 replay；确定性错误不得靠重试掩盖。
- 不修改或删除 final v7、checkpoint、离线 plan/catalog、审计报告、失败
  现场和事故证据；不使用 `weights_ema` 接续。未来若获准续训，只能从经
  独立验证、含 optimizer/scheduler 的完整 `last.pt` 出发。
- 不输出 token、encoded bindings、`.env` 或云凭证。禁止 `rm -rf`、
  glob 删除、`aws s3 sync --delete` 和覆盖既有 S3 key。
- 整理范围限 MoT 自有代码、文档和私有 `motif` 仓库；不碰数据集、
  checkpoint、环境、第三方/参考仓库或其他项目。
- Git 先推新的干净审查分支，不直接更新 `main`。当前工作区有历史脏改动，
  不得用 reset/checkout 等破坏性命令“清理”。
- 任何会改变数据来源、real 内部比例、预算、评测口径或启动状态的决策，
  先向用户说明证据、取舍和建议；不要盲目代替用户拍板。

## 什么意味着本阶段成功

成功不是“已经开跑”，而是形成一套可审查、可追溯、无隐藏假设的启动提案：

- benchmark 对比采用同协议或明确标出不可比项；
- real 候选池给出 native 可服务数量和质量/去重证据，而非原始 URL 数；
- 数据 mix、预算、gates、文字能力评测和回退条件都有理由；
- 所有用户决策点集中列出，未获批准前没有不可逆动作；
- provenance 明确串起 frozen v7、workspace 备份和由其派生的 Git 核心审查
  分支，并分别标注各自身份；不得把精简后的 Git tree 冒充 v7 原字节副本。

发现事实与交接冲突时，以现场只读核验为准；记录冲突并询问用户，不要自行
覆盖证据。
