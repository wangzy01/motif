# MoT 当前文档入口

更新时间：2026-07-31（北京时间）

本目录是远端 MoT 项目的当前文档入口。下一位 AI 默认只需按以下顺序阅读：

1. `AI_HANDOFF.md`：零上下文任务交接与操作边界；
2. `STAGE512A_FINAL.md`：已完成 Stage512a 的冻结事实；
3. `STAGE512B_REAL_BRIEF.md`：下一阶段的粗略研究方向与待决策项；
4. `WORKSPACE_ARCHIVE_INDEX.md`：文档、代码、备份与 Git 的整理规则。

## 权威性

- `/fsx/evan/doc/mot/current/` 只有在
  `PUBLICATION_MANIFEST.json`、逐文件 SHA、`_SUCCESS` 和当前版本的
  create-once publication receipt 全部通过后，才成为当前文档导航权威；
  缺任一项都只是候选稿。本地 Mac 副本只用于编辑和恢复，不作为远端事实
  来源。2026-07-31 的 v6 文档发布将 FSx 使用上限更新为十进制 2 TB
  （`2,000,000,000,000` bytes）；v5 及更早记录仅作历史证据。
- 冻结源码、checkpoint、配置、离线计划和审计报告的身份，以
  `STAGE512A_FINAL.md` 中列出的远端路径和 SHA 为准。
- 本轮历史文档统一放在
  `/fsx/evan/doc/mot/archive/pre_20260730/`，并由该目录的 archive
  manifest 绑定。历史文档可用于取证，但其中的状态、路径和启动命令默认
  已经过期，不能直接执行。
- 原有 `/fsx/evan/doc/mot/归档/` 是包含嵌套历史工程的受保护 legacy
  树，继续原位保留；它不属于本轮 `pre_20260730` 的移动范围，也不得被
  发布工具递归读取。
- 任何未列入 current 的旧 `stage512b` 配置、流式方案或 anime-heavy
  方案都不是 Stage512b-real 的候选配置。

## 当前状态

- Stage512a W64 8×8 已完整结束并通过独立 full-state 审计。
- 当前只做 Stage512b-real 的调研、方案和工作区整理；尚未授权构建数据、
  修改训练代码、提交 GPUVAL 或启动正式训练。
- 当前 `/fsx/evan` 使用上限是十进制 2 TB；耗时或递归用量审计仍必须走
  JCC，不能把整个挂载的 `df` 当作 `/fsx/evan` 目录用量。
- 若现场事实与本文档冲突，停止会改变状态的动作，保留证据，并先向用户
  汇报差异。
