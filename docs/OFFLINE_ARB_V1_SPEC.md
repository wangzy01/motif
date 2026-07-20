# OFFLINE_ARB_V1 — 离线 exact-bucket 数据线重构·规格与路线(Phase-0 冻结)

> 状态:**Phase-0 已冻结**(Evan 拍板 2026-07-09 晚)。本文是重构的权威口径;实现/验收与本文冲突时,以本文为准或先改本文再改代码。
> 背景:streaming item_queue 线(NAI-inspired)经两轮审计(gpt-5.6 Sol)确认存在结构性问题(P0:buffer/deque 容量冲突 treadmill、square-only 泄漏、re-resize 常态化、代码未冻结),Evan 决定整体转向离线 metadata bucket index 方案,streaming 相关代码弃用并归档。

## 0. 已拍板决策(2026-07-09,Evan)

| # | 决策 | 内容 |
|---|---|---|
| D1 | 路线 | 弃 streaming 组批,重构为**离线 catalog + 精确 assignment + 确定性 plan**;数据面(range-read / shard-aware / reshard)由 Phase 1-2 实测决定,不预先押注 |
| D2 | r2 smoke | 已取消(evan-mot-crop-naifu-arb-2x8-smk144-clean-trxt,Pending 中撤除,零 GPU 消耗) |
| D3 | 阶段范围 | **设计覆盖全阶段(256/512a/512b),stage256 先落地**;512 配置同构预留,256 验收后启用。原 cleanup 工作树在跑的 512a 经线绝不碰 |
| D4 | 子代理 | 执行/调研/审计/独立视角默认走 **gpt-5.6-sol** subagent(codex exec,已验通);fable 留主线;调用出问题找 Evan 讨论 |
| D5 | 运行纪律 | 一切耗 CPU/内存任务走 JCC(含纯 CPU);本机只做轻量文件/git 操作。任何决策找 Evan 拍板 |
| D7 | **桶表拍板(2026-07-10 晨,Evan)** | `bucket_prob_floor=0.005` → **20 桶**,AR 0.524-2.0,折叠质量 2.3%,C6 可行(依 Phase-3a 全量 N[B,d] 决策材料);floor0 全表留作 Phase-4 压力档 |
| D8 | gpic gate 处置(2026-07-10 晨,Evan) | 接受 1.48× 裕度(88.2 img/s vs 真实需求 59.5/节点)进 loader 设计(gpic 滚动 /local 预取);后台继续调优上探(48 线程/双路径),不阻塞主线 |
| D9 | 进展批复(2026-07-10 午,Evan) | 阅毕 PROGRESS_20260710 全部认可:textatlas 直读+staging 后备、R_max 默认值、gpic 维持 D8、视觉审计 loader 后执行;指示**全功率推进 + 多开 Sol subagent**。集群新规则:禁 64cpu/0gpu job,CPU job 单个 16 核、可 2-3 个并行;本机(32GB)绝不冒 OOM |
| D10 | catalog/审计口径三连拍板(2026-07-10 午后,Evan) | ① tar-join miss 容忍保留 0.1%(与 taridx 覆盖率 gate ≥99.9% 同口径;排除行全程留痕 bad_rows/N_diff;**配套硬约束:最终生产 plan 必须经 make_planner_input --catalog-header 用 catalog 排除后精确 N 构建并绑定真 hash**,排除行绝不进训练);② 视觉审计 AR 分档冻结为 竖<0.8 / 近方[0.8,1.25] / 横>1.25(log-AR 对称,与 multi_ar 度量自洽);③ face 保留率 gate 双口径并报:AR-balanced 主线 + 自然频率加权对照,gate 仍 report-only,formal 授权人工拍 |
| D11 | **formal 上线授权(2026-07-10 晚,Evan 睡前拍板)** | 今晚目标:全链绿后上 8×8 formal。① **全链绿才停 512a**(`evan-mot-redesign-512a-multiar-6x8-v3-qc6r-retry-2cxf`):catalog 全量验证过 + W=64 生产 plan 验证过 + loader CPU 冒烟绿 + 2×8 短 GPU 验证(含 resume smoke)绿 → 停 512a → 立即提 8×8;任一环节不绿则 512a 不动,晨报;② **global batch 保持 2304**(per-rank bs=36 @64 ranks),配方/步数/LR 全沿用;③ **全新从头训**(Qwen 逐层复制初始化,全新 checkpoint_dir,严格口径⑦);④ 8×8 可无限期排队;正式跑后持续监控;⑤ **夜间处置权**:确凿异常(NaN/反复崩溃/供给饥饿步时>2×/桶调度与 plan 不符)授权直接停 formal 留尸检报告,晨起复盘;⑥ 视觉审计跑完留报告晨看,不阻塞 formal;2×8 仅作调试/验证用 |
| D6 | 夜间自主授权(2026-07-09 晚,Evan 睡前 ~10h 窗口) | ① GPU 基本自主:≤2×8 GPU,单 job 原则 ≤5h(formal 训练仍绝对禁止);② Task A w/h 超阈 → 自动切「catalog 期解码校正」线继续,晨起可否决;③ CPU 原则 ≤8 节点、/fsx 新增产物 ≤100GB,必要时可适当超,S3//local 余量更大(路径放 evan 下);④ 冲突时规格严格性优先,受阻线停住转并行线,待拍板项集中进晨报;⑤ Sol 产出必须经 fable5/sonnet5 子代理交叉审查后才提交;⑥ 桶表选择、formal 授权仍留给 Evan |

## 1. Phase-0 七条硬口径(全部采纳,验收以此为准)

1. **re-resize 严格为零**:item→bucket 指派后,从 native image 到训练 target 只做**一次** resize/crop;训练期不存在邻桶 fallback、不存在二次裁剪路径(代码上应不可表达,而非仅配置关闭)。
2. **square-only 绝对硬约束**:synthetic/textatlas 的非方桶计数**构造性为零**(assignment 阶段即保证),运行时无任何放回路径。
3. **domain mix 按图片曝光计数**(与旧训练语义一致),精确记账到 plan。
4. **token/算力只做旁路监控**,不影响 mix。
5. **坏 item fail-fast**:decode 失败/尺寸与 index 不符 → 直接失败,不静默替换(替换策略若未来需要,须显式拍板)。
6. **anime `crop_v_anchor=0.15` 暂保持不变**,等视觉 A/B(Phase 6)再定。
7. **独立 run/checkpoint namespace**:offline 线不续任何 streaming checkpoint。

## 2. 架构(四层,索引与读取解耦)

| 层 | 产物 | 职责 |
|---|---|---|
| L1 Raw catalog | 版本化 Parquet + 紧凑 locator | 全量 item:稳定 item_uid、domain、shard URI、tar member offset/length、EXIF 后宽高、codec/bytes、caption 引用、fingerprint、eligibility 与排除原因。**不绑定桶表**(换桶只重建 L2) |
| L2 Assignment snapshot | item→唯一 bucket 映射 | 由 (catalog hash, 桶表, 指派算法, square-only 表, filter/EXIF/crop 语义) 共同 hash 确定;square-only 非方 cell 构造性为零 |
| L3 Training plan | [step, rank, slot] → item row_id | 整数配额(小型运输/最小费用流,而非独立取整):Σ_B S[B]=H、Σ_d C[B,d]=G·S[B]、Σ_B C[B,d]=D[d];不可行 → **plan-time 失败**,绝不训练中 renorm。low-discrepancy 桶序;(B,d) 内确定性无放回 permutation;同 global batch 无跨 rank 重复;v1 全量物化分块存(~3.1GB int64,mmap 读),含各层 hash 与 checksum |
| L4 Fetch adapters | tar range-read / shard-aware window / bucket reshard 三后端 | Phase-2 shootout 实测后选;anime 已有逐图 fetch+cache 直接复用 |

**Resume 契约**:plan 绑定 world_size 与 per-rank BS;checkpoint 记录已完成 optimizer step;resume 丢弃未训练预取;crop offset/caption 选择/augment seed 由 (plan_id, step, rank, slot) 无状态派生。数据顺序 exact ≠ 浮点 bit-exact,两概念分开表述。

## 3. 阶段计划与 gate

| Phase | 内容 | Gate(过 → 下一步;不过 → 停,找 Evan) |
|---|---|---|
| 0 | 本规格冻结 + r2 取消 + 旧代码归档 | ✅ 已完成 |
| 1 | 代表性 source/shard 小型可行性(稳定 ID/宽高可信/ tar 压缩与 offset/cache 真实行为/动态域确定性/冷热延迟) | ✅ **GO**(2026-07-09 深夜,job p1-meta-g9k5 全 PASS + p1-io-rmvj 6/7):A=5域 w/h 可信(免解码成立,解码校正预案不触发);D=95.7M uid 零重复零跨域碰撞;B=mountpoint-s3 真 range 语义;C=全量 tar 索引 ~0.4h/3.5GiB;E=CDN 214img/s 零限流;G=textatlas 索引就绪。唯一红灯 F:gpic 串行 4.1img/s(需 ~119 全局)→ Phase-2 头号问题=gpic 并发扩展性,备选=/local 预取 |
| 2 | IO shootout:顺序 reader / 朴素 range / shard-aware / 小规模 reshard 同条件对比 | 最慢 rank data-only 供给 ≥ 训练需求 1.5×;max_rank(batch_ready) p99 占 step 小比例;无整 shard 放大/限流 |
| 3 | 全量 catalog + assignment,输出完整 N[B,d]/坏样本/重复因子/crop-retention 报告 | **桶表由 Evan 据全量数据拍板**(不预设 13 或 95;prob_floor=0.01 是 streaming 工程阈值,offline 下重新评估) |
| 4 | planner-only 验证(不读图):同桶/144 整除/square-only 零违规/配额守恒/无重复/resume 后缀一致/双跑同 hash/不可行报错 | 全项通过 |
| 5 | feature-flag loader `offline_exact_v1`(独立路径,streaming 保留为回退直至验收);复用 caption/augment/单次 resize/collate | 单测 + CPU 端到端 |
| 6 | 整合验证:CPU index+planner+audit 一体 job;2×8 串联(强制桶覆盖 smoke→自然分布 canary→save→主动中断→resume→item trace 对比);**视觉对照**(旧 square crop vs streaming ARB vs offline exact:头脚同保率/face·person retention/retained-area 分层) | **Evan 明确授权后才进 formal** |

## 4. 明确暂缓(由 Phase 1-3 数据决定,不许提前拍)

保留桶数;range-read vs reshard;"offline 必然更快"(需实测);工程周期承诺。

## 5. 归档

streaming 线代码快照 → `archive/streaming_arb_v1_20260709/`(附 README:它验证过什么/为何弃用/两轮审计结论指针)。原文件在 feature-flag 期间保留运行能力,offline 验收后再移除。

## 6. 风险与纪律(沿承)

- 坑点#11/#28 继续有效;jobs/ 里 cd 回原 cleanup 工作树的旧 yaml 属危险清单(如 redesign_256_multiar_2x8_causalmlf_FORMAL.yaml),重构期间不碰不删,归档时标注。
- MONET stats cache 的 fingerprint 覆盖写是跨目录写风险点,offline 线读它必须只读化。
- formal 前必须冻结不可变快照并记录 code/config/index/assignment/plan 全套 hash。
