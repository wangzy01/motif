# OFFLINE_ARB_V1 · 阶段汇报(2026-07-10 晨,覆盖 07-09 夜 → 07-10 晨)

> 一页看懂:**离线 exact-bucket 重构从拍板到 Phase-4 全链路打通只用了一夜**。五域数据源全部实证可行,
> 真 N[B,d] 证明裁剪保留率 p5 从 0.545→0.945,planner×独立验证器对撞全绿,IO 三大瓶颈两个彻底解决、
> 一个(gpic)接受 1.48× 裕度。当前:全规模 planner 验证 + 全量 tar 索引 + gpic 调优在跑,catalog 构建器在写。

## 关键结论(证据均已落盘)

| 项 | 结论 | 证据 |
|---|---|---|
| w/h 元数据可信 | 5 域×2000 解码比对全 PASS → **免解码桶指派成立** | p1-meta job,taskA_result.json |
| uid 唯一性 | 95.7M 行零重复、monet∩synthetic 零碰撞 | taskD |
| mountpoint-s3 range | **真 Range 语义**(500 次随机读实证) | p1-io taskB |
| 裁剪保留率(项目立项理由) | anime/monet/gpic p5:**0.545→0.945+**(vs 旧 square crop) | bstats bucket_decision_summary.md |
| 桶表(D7,Evan 拍板) | **floor=0.005 → 20 桶,AR 0.524-2.0,折叠 2.3%,C6 可行** | 同上,4 档变体对比 |
| monet/synth 供给 | 稀疏 range 79.6 → **密集 coalesced 1093.7 img/s**(13.7×) | p2b taskT1,DECISIVE_DENSE_COALESCE |
| textatlas 供给 | 1635.9 img/s@8 并发(p2 超时是全局锁 artifact,已修口径) | p2b taskT3 |
| anime 供给 | 364 img/s@64,CDN 零限流 | p2 taskS4 |
| gpic 供给(D8) | 深预取稳态 88.2 img/s = 真实需求 1.48×;接受,后台调优上探 48 线程 | p2b taskT2,p2c 在跑 |
| planner×验证器 | **实现者≠验证者对撞全绿**:V1-V6(双构建同 hash、--start-step 后缀一致)、不可行诊断完整、domain_counts 物化后全 step 可判 | 本地 demo×3 + 补丁回归 |

## 工程事实

- 架构四层(catalog/assignment/plan/fetch)+ §9 规格补钉 + §11 catalog 契约 + §12 loader 契约全部成文
  (docs/OFFLINE_ARB_V1_SPEC.md + OFFLINE_ARB_PLANNER_DESIGN.md)。
- planner_v1(Sol 实现,milp+回退修复环+Feistel 置换+分块物化)~117KB;独立验证器(sonnet 仅凭文档实现)
  1560 行,16 个注入式坏 plan 场景回归全绿。
- 旧 streaming 线归档 `archive/streaming_arb_v1_20260709/`(SIGSEGV 根因等有效资产见其 README)。
- 实测修正两处早期外推:tar header 扫描 FUSE ~14ms/member(Task C 乐观 ~10×,已改 5h40m 预算续跑);
  p2 的 S3 textatlas 并发被 PARQUET_NATIVE_LOCK 串行化(测量 artifact,p2b 已修)。

## 在跑 / 在写

- `p4-fullscale-2pmd`:H=166,667 全规模 plan(floor0.005 主档 + floor0 压力档)双构建+后缀重建+验证。
- `p3a-taridx-6jnz`:全量 tar member 索引(断点续跑,cc12m 已完成 6.4M/714 tar)。
- `p2c-gpictune-gtnv`:gpic 48 线程混合复测。
- Sol:phase5_build_catalog(§11 契约)编写中。

## 待 Evan 拍板 / 知悉

1. **textatlas /local staging 必须显式化**:CPU 节点无预铺,生产 GPU job 需加 staging 步骤(旧 formal
   yaml 没有此步——直读 S3 实测 1635 img/s 也够,可二选一,建议直读+staging 后备)。
2. R_max 默认(anime/monet/gpic 20,synth/textatlas 200)已按 §10 建议用于全规模验证,formal 前请过目。
3. gpic 调优若 48 线程仍 <90 img/s:维持 D8(1.48×)或上 boto3 直连,届时汇报数据再定。
4. Phase-6 视觉审计(头脚保留率人工对照)与 checkpoint save→resume smoke 未开始——loader 落地后执行。
