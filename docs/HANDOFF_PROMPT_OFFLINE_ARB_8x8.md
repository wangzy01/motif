# 接手提示词:OFFLINE_ARB_V1 收尾 → 停 512a → 上 8×8 正式训练

> 把本文件全文作为提示词交给全新的 AI。它没有任何上下文,以下即全部。

---

你接手 Evan 的 MoT T2I pilot 项目中「离线精确分桶数据线(OFFLINE_ARB_V1)」的收尾工作。前一个 AI 已把整条离线数据管线建好并验证到 95%,只剩最后一个 loader 性能问题没关掉。你的任务是关掉它、走完上线序列、然后监控正式训练。

## 1. 最终目的(唯一验收标准)

在**全链验证绿**的前提下:
1. 停掉正在跑的旧 streaming 训练 512a(6×8 卡);
2. 立即提交新的 **8×8 卡 stage256 离线精确分桶正式训练**(从头训,~166,667 步,预计 ~5 天);
3. 正式跑起来后**持续监控一段时间**(首小时 5–10 分钟粒度,之后 15–30 分钟粒度)。

**任何一个验证 gate 不绿 → 512a 一动不动,先修问题。** 这是 Evan 拍板的铁律(见 §4 D11①)。

## 2. 第一步:按顺序读这些文件(动手前必读)

1. `/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop/CLAUDE.md` — 项目总约束(≤1 屏)
2. `/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop/当前进展_交接GPU.md` — 运维日志/坑点全集
3. `/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop/docs/OFFLINE_ARB_V1_SPEC.md` — 本数据线的设计权威 + 决策记录(D1–D11)
4. `/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop/docs/OFFLINE_ARB_PLANNER_DESIGN.md` — planner/plan 语义(§7 resume 语义、§11 tar-miss 容忍)
5. 本文件 §5–§9

读完先给 Evan 一句话「我理解的现状 + 打算从哪步开始 + 需要拍板的点」,等他确认再动手。

## 3. 硬性行为边界(违反 = 事故)

- **一切运行(含纯 CPU 冒烟/单测)都提交 JCC job,绝不在本机跑重活**。本机只有 32GB 内存,OOM kill 会丢数据。只有秒级、零模型加载的纯文件/git 操作可以本机做。
  - JCC 用法:`/fsx/evan/bin/jcc job submit <yaml>`、`jcc job list`、`jcc job log <name> -p all -n 300`(不加 `-p` 会卡在交互菜单)、`jcc job stop <name>`。
  - **jcc 会给 job 名自动加随机后缀**(如 `-m6hm`),grep 时用 `"<前缀>-"` 匹配。
- **禁止 64cpu/0gpu 的 job**。CPU job 一律 16cpu 一个,默认 2–3 个并行;Evan 2026-07-11 特批过最多 8 个 16cpu 并行(注意互相别冲突)。
- **`/fsx/evan/mot_code/mot_pilot_v1_cleanup`(不带 `_crop`)是另一条直播线,代码绝对不碰**;你只在 `/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop` 这个 worktree 里改。
- **坑点#11:JCC job 实时读 /fsx 上的代码**。有 gpuval/正式 job 在跑时,不许改它 import 的任何文件(train.py / configs / training/* / networks/*)。要打补丁 → 先停掉在跑的验证 job(正式 job 则绝不能为打补丁而停,除非 §7 的异常停机条款触发)。
- **commit 只在 Evan 明确要求时做**。
- `docs/E1_RAE*` 是另一个 agent 的线,只读不动。
- `/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json`(4 域旧版)**绝不能改**;本线用的是 `t5_sampling_redesign_v1.json`。
- **Py3.12 下禁止周期性 faulthandler dump**(历史 SIGSEGV 根因)。
- arxiv 网络被 block;查论文 grep `/fsx/evan/doc/mot/mot_reading_report/data/page_text/`。
- **有需要拍板的事就问 Evan,给选项;但他授权范围内的事直接做,别空等。**
- **监控要主动、紧凑**:活跃流水线期间 2–5 分钟粒度看日志,不要长间隔死等(Evan 明确批评过等太久)。
- 汇报风格:中文、干货、表格对比、直接表态;每轮有实质改动就更新 `当前进展_交接GPU.md`。

## 4. 已拍板的决策(不要重新讨论,直接执行)

D11(Evan 2026-07-10/11 拍板,原话级约束):
- ① **全链绿才停 512a**:catalog 验证 + w64 plan 校验 + loader CPU 冒烟绿 + 2×8 gpuval(含 resume)绿 → 才 `jcc job stop` 512a,**停完立即提 8×8**;任一 gate 红 → 512a 不动。
- ② global batch 保持 **2304**(8×8=64 ranks × per-rank bs=36)。
- ③ **全新从头训**,checkpoint 全新 namespace,不续任何 streaming ckpt。
- ④ 夜间/离开授权:正式跑出现**确凿异常**(NaN / crash-loop / 数据供给饥饿致步时 >2× / bucket-plan 校验失败)→ 有权停正式 job,留尸检报告。
- 其他已定:2×8 只用于调试验证,正式直接 8×8,8×8 排队等卡可以一直等;AR 三带 竖<0.8/近方[0.8,1.25]/横>1.25 冻结;tar-miss 0.1% 容忍 + 精确 N 闭环。

## 5. 当前状态快照(2026-07-11 ~21:25,交接时刻)

### 已绿(全部有 JCC job 佐证,`jcc job list` 可查)
| 环节 | 状态 | 关键产物 |
|---|---|---|
| tar 索引 r2(10 分片+verify) | ✅ 0 bad tar | `…/outputs/mot_pilot_v1_cleanup_crop/offline_arb_tarindex_v1/` |
| catalog(5 域 parquet) | ✅ 94,986,755 行,8 项自检过 | `…/offline_arb_catalog_v1/`,catalog_hash `b5f83014e4eab304`,assignment_hash `10bd55e0a37aacef` |
| 三份 plan | ✅ 全过 validate_plan | `…/offline_arb_plans_v1/{w4_smoke,w16_gpuval,w64_prod}/plan`;w64 H=172032,hash 与 catalog 绑定 |
| 视觉抽审 | ✅ job `…p6-vaudit-rsbl` | 报告在其输出目录,给 Evan 过目用 |
| anime 标签频率预计算 | ✅ 17 shard,字节级等价扫描 | `…/offline_arb_animefreq_v1/`(header 绑 catalog_hash) |
| loader CPU 冒烟 | ✅ ×3 次 PASS,trace hash 跨运行不变 | job `…p5-loader-smoke-{6tcv,qnpf,m2lf}` |
| gpuval resume 语义 | ✅ r3(job `…gpuval-h4jb`)两轮:冷启 350 步(ckpt@200)+ resume 续到 500,loss 2.03→1.22 | `…/outputs/mot_pilot_v1_cleanup_crop/stage256_offline_exact_2x8_gpuval_20260711r3/` |

### 唯一红灯(阻塞正式提交的问题)
**gpuval r4(job `evan-mot-crop-offline-exact-256-2x8-gpuval-m6hm`)仍有周期性尖刺** → 详见 §6。r4 可能还在跑或已结束,先 `jcc job list` 确认。

### 在跑的 job(交接时刻)
- `evan-mot-redesign-512a-multiar-6x8-v3-qc6r-retry-bhdg`(6×8,旧 streaming 512a,**待停对象**;注意它带 auto-retry,后缀会变,停之前重新 `jcc job list | grep 512a` 确认当前名)。
- `…gpuval-h4jb`(r3)已 **Succeeded**:两轮 rc=0,Round B 从 step200 resume 续到 500,loss 连续下降——resume 语义 gate 已绿,不用重验。
- `…gpuval-m6hm`(r4,见 §6;交接时刻仍 Running,尖刺已复现至 step 85)。

## 6. 唯一未决问题:每 20 步一次的供给尖刺(接手后第一件事)

**现象**(r4 日志 `…/stage256_offline_exact_2x8_gpuval_20260711r4/train.rA.n0.log`):
步 1/5/25/45/65/85 出现 iter 70–115s(正常 2.4–3.9s),其中 dw(data wait)≈50–66s;**严格每 20 步一次**。此前 r1/r2/r3 修掉了三层根因(见 §9),r4 证明还有第四层。

**前一个 AI 的诊断线索**(接手后可直接验证):
- loader `training/dataline_offline/offline_dataset.py`:plan 按 **prefetch 窗口** 供给,`_DEFAULT_PREFETCH_STEPS=20`(环境变量 `OFFLINE_ARB_PREFETCH_STEPS` 只能调大,代码里 `max()`),尖刺周期与窗口大小完全一致。
- 供给是**每域单线程** `_supply_loop`(`_schedule_window` → `_prepare_window`),初始只预排 2×20 步。若最慢域一个窗口的准备耗时(~110s)> 窗口消费耗时(20 步 × ~2.5s = 50s),流水线存在**持续性赤字**,每个窗口边界卡 ~60s——与观测精确吻合。
- 结论:这不是突发抖动,是**最慢域吞吐不足**,加深 prefetch 只能推迟不能消除;需要让该域窗口准备吞吐 ≥2×。
- 定位哪个域:看 dataline metrics(`prefetch_depth`、per-domain supply 计数,`_refresh_metrics`/`_maintain_prefetch` 附近),或给 `_prepare_window` 加每域计时日志跑一次 2×8。首要嫌疑:anime URL fetch(已 8 线程但可能仍不够/冷缓存)、gpic /local staging、tar FUSE 首字节。
- 修法候选(按侵入性排序):该域 fetch workers 调大(config 键 `fetch_workers_anime`/`fetch_workers_tar`,构造器新键才生效,streaming 旧键 `fetch_workers` 会被 `**_` 吞掉不起作用);窗口准备内部再并行;每域多 supply 线程重叠准备相邻窗口(注意 publish 与断言逻辑是否要求窗口内有序,改前必须读懂 `_publish`/`_ready` 语义)。

**修复流程(必须走全)**:停 r4(若还在跑)→ 打补丁 → 重跑 CPU 冒烟(`jobs/offline_arb_phase5_loader_smoke.yaml`,验收 = PASS 且 trace/prompts hash 与历史一致,证明语义零漂移)→ 提 gpuval r5(`jobs/offline_exact_256_2x8_gpuval.yaml`,**改 RUN_ID 为 …r5** 防旧 ckpt 污染 resume 轮)→ 绿判定 = 两轮 rc=0 + Round B 有 resume 证据 + **除冷启前 ~5 步外无 >30s 尖刺** + 稳态 iter ~2–3s。改 loader 代码前后,找一个独立视角(另一个模型/子代理)做聚焦审查,重点看「补丁前后数据语义逐字节一致」。

## 7. r5 绿之后的执行序列(严格按序,一步一验)

1. 本机跑(纯 hash,KB 级,允许本机):`/fsx/evan/mot_code/venv-gpu/bin/python scripts/offline_arb/make_freeze_manifest.py`(在 worktree 根下)→ 生成 `…/offline_arb_plans_v1/FREEZE_MANIFEST.json`。formal yaml 的 preflight 会校验它并比对 live config 的 sha256,**不生成/生成后再改 config 都会拒启**。
2. 记录 512a 现状留档:`jcc job list | grep 512a` 确认当前 job 名;从 `/fsx/evan/mot_code/mot_pilot_v1_cleanup/jobs/stage512a_redesign_launch.yaml` 里找到其 checkpoint_dir($CK),记下 last.pt 的 step,写进交接 doc。
3. `jcc job stop <512a 当前全名>`,确认进入 Stopped。
4. **立即** `jcc job submit jobs/offline_exact_256_8x8_FORMAL.yaml`(8 workers×8 GPU,maxRetries 2,零 config 覆盖;排队等卡可以一直等)。
5. 正式监控:首小时 5–10 分钟粒度,之后 15–30 分钟。盯:loss 是否下降、iter/dw(稳态应 ~2–3s)、无 NaN、无 crash-loop、/local 磁盘(yaml 里有 85% anime cache 轮转 janitor)、节点掉线。loader 自带 fail-fast(shape/bucket/native 尺寸与 plan/catalog 不符即崩),崩了 = 真问题,别盲目重启。
6. 异常处置按 §4 D11④:确凿异常 → `jcc job stop` → 写尸检报告(现象/日志摘录/初步根因/建议)留给 Evan。
7. 全部落定后:更新 `当前进展_交接GPU.md`,给 Evan 总结报告(含 512a 停机 step、formal job 名、预计完成时间 ~166,667 步 × 稳态步时)。

## 8. 关键路径速查

| 什么 | 路径 |
|---|---|
| 工作 worktree(唯一可改代码处) | `/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop` |
| Python 解释器 | `/fsx/evan/mot_code/venv-gpu/bin/python` |
| 正式 config | `configs/stage256_offline_exact_v1.yaml`(文件头注释 = 与 streaming 基线的全部差异声明) |
| 生产 loader | `training/dataline_offline/offline_dataset.py` |
| 数据产物根 | `/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/`(下含 offline_arb_catalog_v1 / offline_arb_plans_v1 / offline_arb_animefreq_v1 / offline_arb_tarindex_v1 / gpuval 各 RUN_ID 目录) |
| gpuval 模板 | `jobs/offline_exact_256_2x8_gpuval.yaml`(两轮串联;文件内注释解释了每个坑) |
| 正式 job | `jobs/offline_exact_256_8x8_FORMAL.yaml` |
| CPU 冒烟 | `jobs/offline_arb_phase5_loader_smoke.yaml` + `scripts/offline_arb/smoke_offline_loader.py` |
| freeze manifest | `scripts/offline_arb/make_freeze_manifest.py` |
| plan 校验器 | `scripts/offline_arb/validate_plan.py`(README_VALIDATOR.md 同目录) |
| 正式 ckpt namespace | `/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage256_offline_exact_v1` |
| gpuval 日志 | `<数据产物根>/<RUN_ID>/train.r{A,B}.n{0,1}.log`(step 行只在 global rank0 = n0) |

## 9. 本线专属坑点(前人踩过,别再踩)

1. **max_steps 必须走 CLI `--max-steps`**:dotlist 的 `trainer.max_steps` 会被 `max_exposure_images` 派生值静默覆盖(train.py:226 附近),plan 步数耗尽直接 StopIteration 崩。
2. **RUN_ID 每轮 gpuval 换新**(r5、r6…),否则 Round B resume 会吃到上一轮的旧 ckpt。
3. loader 构造器只认新键 `fetch_workers_anime`/`fetch_workers_tar`;streaming 旧键 `fetch_workers` 被 `**_` 吞掉,改了没用。
4. anime freq 预计算表(`offline_anime_freq_root`)与 catalog 通过 catalog_hash 强绑定,loader 校验不过直接拒启——重建 catalog 就必须重跑 `precompute_anime_freq.py`。
5. 多机 rendezvous 用共享文件传 MASTER_ADDR,带 30 分钟新鲜度戳;陈旧 IP = 上次 attempt 残留,读方会拒收——job 重试后卡 rdzv 先看这里。
6. 冒烟脚本结尾 `os._exit(0)` 是有意为之(非 daemon 线程钉死进程退出),别"修"它。
7. `OFFLINE_ARB_STEP_TIMEOUT_S`(gpuval yaml 里 1800)是验证期加大的步超时;正式版沿 config/代码默认,别照抄。
8. 已修的三层尖刺根因(改 loader 时别回退):anime 标签频率运行时全 shard 扫描(→预计算表)、fetch_workers_anime 默认 1(→config 显式 8)、`_coalesced_tar_payloads` 逐 tar 串行 FUSE(→8 线程逐 tar 并行)。
9. GPU 利用率观感:稳态 dw 仍占步时 30–50%(256 分辨率下计算每步仅 ~1.5–2s,供给端有富余度问题),尖刺期 GPU 整分钟空转——**尖刺修掉后平均利用率会大幅回升,但稳态 50–60% 上下是当前架构的正常水位**,不要按 90% 去"修"。
