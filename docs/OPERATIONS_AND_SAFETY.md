# MoT 工作区与历史归档索引

更新时间：2026-07-31（北京时间）

## 活动目录

- 当前文档：`/fsx/evan/doc/mot/current/`
- 本轮历史文档：
  `/fsx/evan/doc/mot/archive/pre_20260730/`
- MoT 代码根：`/fsx/evan/mot_code`
- 数据分析代码根：`/fsx/evan/mot_code/data_analysis`
- 私有 Git：`git@github.com:wangzy01/motif.git`

活动根只保留当前索引、交接、最终权威和下一阶段 brief。Mac 本地工作区只是
编辑缓存，不是 canonical Git 或远端事实来源。

## 本次集中归档规则

除既有受保护 legacy 树外，旧 MoT 文档全部保留在 FSx，并集中迁入：

`/fsx/evan/doc/mot/archive/pre_20260730/`

归档前后生成机器可读 manifest。普通文件记录原路径、目标路径、bytes、
SHA-256、时间和分类；整体迁移的目录记录顶层 inode/stat 与 same-filesystem
rename 事实。完成相应复核后才更新 current 索引。不删除历史文档，也不把
旧文档的命令恢复为“当前”。

本轮归档范围是 `/fsx/evan/doc/mot` 下除 `归档` 外、发布前经精确一层
closure 核验的全部 19 个既有顶层项，以及：

- `/fsx/evan/mot_code/MOT_PILOT_AI_HANDOFF_20260718.md`

实际归档必须以 JCC inventory 的精确 allowlist 为准；不得递归猜测或使用
glob 移动。普通文件必须核验完整 SHA；4 个显式顶层历史目录只允许在同一
FSx 上整体 rename，并记录顶层 inode/stat 不变。发布器首先尝试
`renameat2(RENAME_NOREPLACE)`；仅当该调用返回 `EINVAL` 或 `ENOSYS` 时，
才允许使用目标路径专属、持久且 create-once、mode `0700` 的空目录锁；
锁必须用 `O_DIRECTORY` 打开并持有 exclusive `flock`。锁内先在精确目标
名原子创建 publisher-owned 空 sentinel，持久记录其 lstat/inode；再次
复核源对象身份、sentinel inode/mode，且目录 sentinel 必须保持精确空闭包，
普通 rename 只能替换该已绑定 sentinel，绝不能面对缺失或外来目标执行。
每次迁移的实际方法、触发 errno、目录锁身份、sentinel 和
intent/completion 证据必须进入最终回执。所有 intent、completion、最终
manifest/receipt/_SUCCESS 均先写入并 fsync 有界 prepared attempt，再用
hardlink 原子 no-replace 发布；prepared twin 与坏 attempt 永久保留并参与
最终核验。不得把顶层目录迁移描述成对子树内容做了新一轮 hash。嵌在冻结
source release 中的历史文档属于 release manifest，保持原字节和原位置，
不参与迁移。

原有 `/fsx/evan/doc/mot/归档/` 包含嵌套历史工程和路径敏感项，继续原位
保留；发布器只允许对它做顶层 `lstat` 前后对比，禁止进入、枚举、移动或
删除。

## 永久保留且不移动

- Stage512a final v7 release；
- 最终 `last.pt`、最终配置、当前离线 plan/catalog；
- full-state audit、最终评测输出和冻结评测 release；
- 事故、失败、旧 gate/checkpoint 和恢复证据。

这些项目即使有 S3 备份，也不自动获得移动或删除授权。

## 整理范围

仅整理 MoT 自有代码、文档、控制文件、项目评测代码及私有 `motif` Git。
以下默认不动：

- 数据集、checkpoint、离线 plan/catalog 大文件、既有训练/eval 输出；
- `venv-gpu`、`venv-dpg`、JCC CLI、`.env`、token 和云凭证；
- `minit2i-torch`、`mot_rae_e1`、`naifu-main`、`other_repos`、`rae_refs`、
  `troph-team_*` 等第三方/参考项目；
- 与 MoT 无关的 FSx、S3 和本地文件。

单个源文件大于十进制 100,000,000 bytes 时不进入“小文件包”，仅在
identity/exclusion manifest 中登记；这不构成清理授权。

## 备份门

在整理代码或迁移普通旧文档前：

1. JCC 生成精确 allowlist，并分类 small/large/secret/Git topology；
2. 在 fresh create-once S3 prefix 写入代码/文档小文件包、Git bundle、
   dirty working-tree snapshot、manifest 和 SHA；
3. 用独立新 JCC pod 重新下载和全量 rehash；
4. 在 pod 本地目录做 safe restore drill；
5. 所有检查通过后最后写 success marker。

禁止覆盖既有 S3 key、`aws s3 sync --delete`、输出敏感内容，或把备份成功
解释成可删除 final 证据。

## Git 整理

- 远端 canonical Git 工作树：
  `/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop`。
- Mac 缓存工作树（若存在）：
  `/Users/a26012/Documents/公司服务器 2/work/github_publish/motif/`。它可
  不存在、不是权威，也不是任何远端发布或 Git 操作的前置条件；若存在才与
  FSx 工作树分别 fingerprint，不得假设二者相同。
- 当前基础：`origin/main` at
  `21dfd7a4c2bc9f9a94344d1a35d724def9b45738`；FSx 现有分支
  `codex/publish-latest-mot` 带有未提交历史改动。
- 不 reset、不丢弃 dirty tree。先将 Git 历史和 dirty snapshot 纳入已验证
  备份，再从已核验源码身份构建新的干净 `codex/` 审查分支。
- 审查分支固定为 `codex/stage512b-real-core-20260730`，只保留可维护的
  训练、数据构建、评测核心及 compact 文档；历史 jobs/docs 放 archive，
  不继续占 current tree。
- 历史锚点 tag 固定为 `archive/pre-core-curation-20260730`。流程必须
  状态驱动：branch、tag 或对应 evidence 已存在时只核验 commit/tag object、
  tag target 和 remote ref，不重复 commit、tag 或 push。
- 先 push 审查分支供人工检查，不直接更新或 force-push `main`。完成门还
  必须证明 `origin/main` 未变化，并对 FSx dirty tree 与可选 Mac dirty tree
  分别复做 fingerprint；Mac tree 不存在时明确记为 absent，而不是失败。

## 执行回执

### Workspace S3 备份（已完成）

- control release：
  `/fsx/evan/mot_code/data_analysis/backup_releases/mot_workspace_pre_stage512b_real_20260730_v6`
  - 文件：`CONTROL_MANIFEST.json`
  - 该文件 raw bytes SHA-256：
    `85966a0bfdec8ecb2ee26f588e5f96dbf855271f74ba63042244edd3e10826fe`
- sealed inventory：
  `/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/workspace_inventory_pre_stage512b_real_20260730_v6`
  - 文件：`inventory.jsonl`
  - 该 JSONL raw bytes SHA-256：
    `f269ed812fc9a69a82cd0e83a9456f0f315bb5a9dd0ef398bd492fb02f0979d7`
- create-once S3 prefix：
  `/s3/evan/mot_pilot_backups/mot_workspace_pre_stage512b_real_20260730_v6`
  - `controls/build_manifest.json` raw bytes SHA-256：
    `51755321318151493e0b11380d824bc0f7349249986e741ef63e2cc044dc72ce`
  - `controls/verified_manifest.json` raw bytes SHA-256：
    `3b4d2837f43ffecc6fa57ce8cc20bb29747d6630b95427e4cc7cca1d53542a83`
  - `_SUCCESS` raw bytes SHA-256：
    `79e87adb08fac1ec4f1100a1d161042f7623a08d75b87d007e8afdae3d832016`
- full restore evidence：
  `/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/workspace_archive_pre_stage512b_real_20260730_v6/full_restore_drill/REPORT.json`
  - 该文件 raw bytes SHA-256：
    `2dfdde14f1662d02cd417ee900e526b491e502b1c90729dd5054cfb2aeef2963`
- 成功 JCC：
  `evan-mot-ws-archive-build-20260730-v6-6tfb`、
  `evan-mot-ws-archive-fresh-verify-20260730-v6b-c9q9`、
  `evan-mot-ws-archive-restore-full-20260730-v6b-pcsb`。

闭包为 33,464 archive members / 5,327,233,644 bytes / 1 canonical Git
bundle；全量 compressed/uncompressed SHA、safe restore、二次逐文件 SHA、
exact tree、Git mirror clone/ref 均通过。所有 receipt 仍明确
`deletion_authorized=false`；备份成功不授权删除 final/事故证据。

FSx 文档发布和 Git 整理的最终动态身份不写回本文自身，以避免 manifest
自引用。文档发布必须按以下确定性路径核对机器回执：

- current manifest：
  `/fsx/evan/doc/mot/current/PUBLICATION_MANIFEST.json`
- current success：`/fsx/evan/doc/mot/current/_SUCCESS`
- archive manifest：
  `/fsx/evan/doc/mot/archive/pre_20260730/ARCHIVE_MANIFEST.json`
- archive receipt：
  `/fsx/evan/doc/mot/archive/pre_20260730/ARCHIVE_RECEIPT.json`
- publisher rendered plan：
  `/fsx/evan/mot_code/data_analysis/doc_publication_evidence/mot_docpub_20260731_v5/RENDERED_PLAN.json`
- fresh 独立只读 verifier report：
  `/fsx/evan/mot_code/data_analysis/doc_publication_verify_20260731_v5/REPORT.json`
- verifier success：同目录 `_SUCCESS`

### 保留的 v4 失败证据

2026-07-31 的 v4 candidate 已完整发布并验证，但唯一 publisher JCC 在任何
evidence 创建、archive/current scaffold 或 legacy move 之前，被一个依赖
目录时间观测精度的发布前单测 fail-closed。v4 control、candidate、Failed
publisher job 与 lock metadata 都是只读事故证据，不是发布 authority，也
不得 replay、覆盖或作为 v5 的恢复源；生产文档在该失败中零移动。

### 保留的 v3 失败证据

2026-07-31 的 v3 控制发布在移动任何旧文档前因 FSx 新目录的易变
`st_size` 观测触发 fail-closed。其 staging、lock、intent 与 prepared
metadata 都是只读事故证据，不是发布 authority，也不得作为 v5 的恢复源。

只有 verifier 的 `REPORT.json`/`_SUCCESS` 绑定并重新核验 control、
candidate、archive/current、publisher plan、全部 lock/intent/completion
与 prepared twin 后，current 才可作为导航权威。另行核对：

- S3 prefix、manifest/SHA、独立 rehash 与 restore-drill 结果；
- Git branch、commit object、tag object/target、remote refs、`main` 未变化
  与两份 dirty fingerprint（Mac worktree absent 可显式记录）；
- 明确保留、排除和仍待用户决策的项目。

Workspace S3 备份已完成。FSx 文档集中归档、current 发布与 Git 审查分支
是否完成，只能由各自 create-once manifest、`_SUCCESS`、Git ref 和独立
验证回执判断，不能由本文或 workspace 备份结果推定。
