# 独立审计任务书 —— MoT Stage512b-real 训练启动前审查

你被请来做**独立对抗性审查**。你没有任何关于本项目的上下文,本文件包含你需要的全部背景。
被审查的对象是**另一个 AI 助手在 2026-08-02 至 08-03 这一轮所做的全部改动**。
你的任务不是复述它做了什么,而是**找出它做错的、漏掉的、以及说过头的地方**。

---

## 0. 硬性安全约束(违反任何一条都比"没审出问题"严重得多)

这套系统里有几个产物是**不可再生的**或**重建代价数小时到一天**。请把下面当作物理定律:

1. **绝对不要编辑 `data_analysis/configs/t5_sampling_stage512b_real_v1.json`,连加注释都不行。**
   已建成的 catalog(8,978 万行)在 header 里绑定了这个文件的 sha256
   (`inputs.sampling_config_sha256 = a3ecd965…`),loader **强制**校验。
   改一个字节 = catalog 解绑 = 必须重建 catalog + plan + 56 GB caption sidecar。
   被审的 AI 在 08-02 犯过这个错(想把结论写进注释),当场解绑,只能 revert。
2. **不要写、删、移动 `/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/` 下的任何东西。**
   只读。不要 `rm`、不要 glob 删除、不要覆盖已存在的文件。
3. **不要动 git `main`(`21dfd7a4…`)、不要移动 tag
   (`archive/pre-core-curation-20260730` = `cb63cf11…`)、不要重建 curation 分支
   (`a8120989`)。** 不要 push、不要 force-push、不要 rebase。你可以本地 checkout 和读。
4. **不要提交任何 JCC 作业**(`jcc job submit`)。集群是共享的且当前非常拥挤
   (90 节点中 63 个满载)。若你认为需要跑什么,写在报告里由人来提。
5. **不要在这台 dev pod 上跑重任务。** cgroup 内存上限 **32 GiB**
   (`free` 显示宿主机的 2 TB,是假的)。读 parquet 请分块;不要把 40M 行的列整个读进内存。
   被审 AI 之前把 pod OOMKill 过。
6. **不要输出任何 token / 密钥 / 凭据。**
7. 你**可以**自由:读代码、读文档、跑单元测试(`scripts/test_*.py`)、跑静态检查、
   读产物的 header.json、抽样读 parquet、`git log`/`git show`/`git diff`。

---

## 1. 项目背景(最少必要)

- **MoT** 是一个约 2B 参数的 text-to-image DiT(rectified flow),文本编码器是
  `/fsx/models/Qwen3-VL-2B-Instruct`。
- **Stage512a 已冻结**:125,000 steps × global batch 2048 = 256M 曝光。权重在
  `/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage512a_20260724_weighted_w64_recovery_v2/last.pt`
  (sha256 `957bfcf1…`,`global_step=125000`)。
- **Stage512b-real** 是下一阶段:**架构完全冻结,只改数据 / 几何 / 训练日程**。
  目标 **512M 曝光 = 250,000 steps × 2048**,world size 64(8 节点 × 8×H100)。
- 几何上最大的改动:从 361 个自适应桶换成 **13 个固定锚**(`anchors_v1`),
  面积门 262,144,严禁对 real 图上采样。
- 数据是**离线预先计划**的:catalog(逐行几何+元数据)→ weighted plan(逐 step 逐 rank
  逐 slot 指定 row_id)→ 训练时 loader 按 plan 取。所以"数据"是一批磁盘上的产物,
  彼此用 sha256 互相绑定。

### 关键路径
| 东西 | 路径 |
|---|---|
| 代码(被审对象) | `/fsx/evan/mot_code_s512b_impl`,分支 `codex/stage512b-real-v04-20260801`,HEAD **`f3d2482`** |
| 产物根目录(只读) | `/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/` |
| Stage512a 旧代码树(参考) | `/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop_stage512a_weighted_20260724v7` |
| Python | `/fsx/evan/mot_code/venv-gpu/bin/python`(**用这个**,系统 python3 没有 pyarrow/torch) |
| 跑测试 | `cd /fsx/evan/mot_code_s512b_impl && PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /fsx/evan/mot_code/venv-gpu/bin/python scripts/test_xxx.py` |

---

## 2. 本轮审查范围 = 这 22 个 commit

```
git -C /fsx/evan/mot_code_s512b_impl log --oneline 3f17d46..f3d2482
```
约 29 个文件、+3200 行。**`3f17d46` 之前的东西已经被前几轮审查覆盖过,不在本次范围内**
(除非你发现前面的结论是错的)。

> ⚠️ **其中两个 commit 不是本任务书作者写的**,是在本文件起草期间由**另一个并发会话**
> 提交并 push 的。请对它们单独加倍审查,**不要因为它们和其余 commit 在同一分支上就假定
> 同源同质**:
>
> - **`d848062`「Close the OCR loop」** —— 新增 `evaluation/runners/ocr_read_images.py`
>   (image→text reader,用 `/fsx/models/Qwen3-VL-8B-Instruct`)。
>   **从未在 GPU 上真跑过**(它自己承认"模型加载那段还没在 GPU 上真跑过")。
>   它同时把 `docs/STAGE512B_EVAL_PROTOCOL.md` 第 2.3 节从"只有 comparator"
>   改写成"两段都已建成" —— **请核实这个改写是否言过其实**。
> - **`f3d2482`「Fix two races in the unsubmitted formal job」** —— 修了正式作业里两个
>   并发缺陷(见第 4 节第 4、5 条)。**这两个修复本身也没有被任何人复核过**,
>   而且它们改的正是这次审查里风险最高的那个文件。**请把它当成新代码来审,
>   而不是当成"已经修好了"。**

改动可以分成六块:

### (a) 决策记录 —— `docs/STAGE512B_DECISIONS.md`(新)
19 条决策。最要紧的是 **S512B-01**:任务书
(`docs/MOT_HANDOFF_CURRENT.md:73`、`docs/STAGE512B_REAL_RESEARCH_BRIEF.md:15`)锁定域配比
`Anime/MONET/GPIC/Synthetic/TextAtlas = 32/44/14/5/5`(real 合计 58%),
而实际执行的是 **30/40/20/5/5**(real 60%)。这个改动出自更早的 commit `879180d`,
**当时没有得到用户批准**,且越过了任务书写明的"real 58% 内重分配须批准"边界。
用户在 08-02 看到逐域差异后确认保留 30/40/20/5/5。

### (b) JCC 作业门禁 —— `scripts/jcc_preflight.sh`(新)
JCC 作业直接在 `/fsx` 的**活工作树**上执行(没有 checkout 步骤、没有镜像打包),
所以工作树脏 = 作业跑的不是被审过的代码。此前 14 个 job spec 只 `echo` 不 `exit`。
新脚本在 HEAD 不等于预期 commit、或 `git status --porcelain` 非空时 **exit 46**。

### (c) TextAtlas 权威 GT 接进训练 prompt
- 背景:TextAtlas 是第五个域(5% 配比),里面有渲染文字的图。其中两个 subset
  (CleanTextSynth 的 `annotation`、TextScenesHQ 的 `raw_text`)带**权威的精确文字**;
  另外七个 subset 提不出可靠 GT。覆盖 1,941,791 / 约 4.12M 行(约 47%)。
- 此前建了一个 GT sidecar 但**训练根本不读它**(loader 只读 `caption1..5`)。
- 本轮:重建为 **v2**(`gt_text_raw` + `gt_text_normalized` 双列 + 真实 tokenizer 计数),
  新增运行时读取器 `training/dataline_offline/textatlas_gt_v1.py`(按 `row_id` join,
  **不重建 56 GB caption sidecar**),新增 prompt 合成
  `training/dataline/caption_build.py::compose_textatlas_gt_prompt`。
- 合成规则:GT 从句**前置**,先给 GT 记 token 预算、剩下的才给 caption,
  所以 **GT 要么完整出现要么完全不出现**(绝不喂半截 GT)。三个措辞模板按曝光 rng 轮换。
- 实测:GT token p50 = 17,只有 63 行(0.0032%)超预算;
  **70,604 行(3.64%)raw ≠ normalized**(这是 v1 缺陷的实际规模)。

### (d) smoke 从只查像素扩展到查 prompt
`scripts/offline_arb/smoke_anchors_loader_v1.py` 新增 `PromptAudit`:检查 prompts/
`cfg_keep`/经真实模板 tokenize 后的长度/`caption_band`/GT 是否逐字进了 prompt。
配套 `scripts/test_smoke_prompt_audit_v1.py` 用**故意坏掉的 batch** 证明每一条检查都真的会失败
(该文件历史上出现过"检查读了 loader 根本不产出的键 ⇒ 空转仍报 PASS")。

### (e) 审计与评测
- `scripts/offline_arb/audit_plan_row_reuse_v1.py`(新):逐行统计 plan 里每个 row 被曝光多少次。
- `docs/STAGE512B_EVAL_PROTOCOL.md`(新):钉死评测协议。
- `evaluation/benchmarks/scorers/ocr_exact_string.py`(新,**只做字符串比对**)、
  `evaluation/runners/ocr_read_images.py`(新,`d848062`,image→text reader,
  用 `/fsx/models/Qwen3-VL-8B-Instruct`,**未在 GPU 上跑过**)、
  `evaluation/benchmarks/prompts/stage512b_human_eval_v1.jsonl`(新,42 条,待用户删改后冻结)。

### (f) GPUVAL 与正式训练作业
- `provenance/jcc_jobs/20260802_stage512b_gpuval_w64.yaml`
- `provenance/jcc_jobs/20260803_stage512b_FORMAL_w64.yaml`(**尚未提交,尚未授权**)

---

## 3. 已经实测过的结论(**不要重复验证,除非你怀疑数字本身**)

GPUVAL 真跑过一次(64 GPU、真 512a 权重、真 plan,跑到 step 420 被集群杀掉):

| 指标 | Stage512a 末 200 步 | Stage512b GPUVAL |
|---|---:|---:|
| loss | 0.8611 | 前 5 步 0.8715 / 稳态 0.8701(**+1.2%**) |
| iter_s p50 | 2.618 | 2.570(⇒ 250k 步 ≈ **7.4 天**) |
| data_wait_frac p50 | 0.048 | **0.0435** |
| peak_mem_gb | 17.76 | 17.76(共 80) |

其他已实测:
- plan 真实引用数 = **512,000,000**(正好 250,000 × 2048)。
- 13 个锚的 loader smoke **13/13 PASS**,真实图像解码。
- prompt 侧 smoke:1,536 行,token p50 119 / max 509(上限 512),0 条被截断;
  textatlas 60 行中 28 行有 GT,**28/28 逐字进了 prompt**;`cfg_keep` 只落在 anime 上,
  **占 anime 行的 26.7%**。
- 行级重复:512b 在每一项集中度指标上都比 512a **更平**(GPIC 最热 1%:10.16% → 5.67%)。
- 真实 prompt 模板 = **39 token**;caption 预算 471;471+39+1 = 511 ≤ 512。

---

## 4. 被审 AI 自己承认的错误(**请专门去找同类的第四第五个**)

这一轮已经发现**五个** bug,**根因完全相同:只想到了一个参与者/一次调用,
忘了还有第二次、还有另外七个**。前三个是它自己发现并修的,**后两个是并发会话
在它起草这份任务书的时候发现的 —— 也就是说这个模式它到现在也没能自己穷尽**。
请假定**还有第六个**,并按这个模式去搜。

1. **`trainer.max_steps=` 覆盖是装饰性的**。`training/checkpoint.py::derive_max_steps`
   在存在 `max_exposure_images` 时按"曝光量 / global_batch"派生 horizon,
   config 里的 `max_steps` 只作参考;**只有命令行 `--max-steps` 能盖过它**。
   GPUVAL 本该 100 步停,实际跑到 420。
2. **create-once 的 claim 写成了进程级**(`mkdir || exit 45`)⇒ rank 0 的 pod 一重启就必死。
   实际发生了。更糟的是同样的逻辑本来要进正式作业,会**摧毁 `resume: true` 的恢复机制**。
3. **`RUN_MODE=fresh` 被应用到了 JCC 重试上** ⇒ 重试撞上自己建的目录 exit 45,
   `maxRetries: 3` 形同虚设,7 天的跑第一次被抢占就永久死。
4. **(并发会话在 `f3d2482` 里发现)每个 rank 都独立跑 fresh/continue 判定**,
   而 rank 0 建目录和写 `.jobtag` 是**两步**。非 0 rank 在这个窗口里读到"目录存在但没有
   `.jobtag`",走进 `owned by <none>` 分支 **exit 45** —— 在一次配置完全正确的首launch 上。
   8 个节点同时起、共享文件系统,正是最容易撞上这个窗口的场景。
5. **(同上)到达屏障的标记文件按 `JOBTAG` 命名,而 `JOBTAG` 在同一次提交的重试之间是不变的**
   ⇒ 重试时会看到上一次留下的 8 个陈旧标记,**屏障瞬间满足**,然后在只有 2 个节点真正
   起来的情况下进入 torchrun —— 屏障本身变成了它本要防止的那种静默失败。
   这条尤其值得注意:**一个"安全措施"被写成了新的失败源**。

它还有两类可复现的毛病,请按这两个模式去搜:
- **"不会失败的检查"**:写了断言但条件永远不成立 / 读了不存在的键 / 只 echo 不 exit。
- **说过头**:例如它一度把 `ocr_exact_string.py` 算作"已清掉的启动阻塞项",
  但那个文件只做字符串比对,**不会从生成图里读字**(没有 image → text 那一段)。

以及一条它自己纠正过的:它曾断言"仓库里根本没有 32/44/14/5/5",**是错的**,
就在 `docs/MOT_HANDOFF_CURRENT.md:73`。请对它所有"我查过,没有 X"式的断言保持怀疑。

---

## 5. 请重点回答的问题

按你判断的重要性排序回答,**每条都要给出文件:行号和你实际跑过/读过的证据**。
说"可能有风险"而没有具体触发路径的,请标注为推测。

### A. 正式训练作业(`provenance/jcc_jobs/20260803_stage512b_FORMAL_w64.yaml`)—— 最高优先级
这是要跑 7.4 天、烧 64 张 H100 的东西,且**还没提交**,现在是唯一能拦住错误的时刻。
- `RUN_MODE` / `EFFECTIVE_MODE` / `$CKPT/.jobtag` / `.runmode.$JOBTAG` 的逻辑,
  在**所有**组合下都正确吗?第 4 节第 4 条那个 race 已经在 `f3d2482` 里改成
  "rank 0 独自判定、发布到 `.runmode.$JOBTAG`、其他 rank 经 `$RDZV` 排序后再读" ——
  **请验证这个修复本身对不对**:发布与 `$RDZV` 的先后真的被强制了吗?
  非 0 rank 读不到 `.runmode` 会怎样?rank 0 在发布后、写 `$RDZV` 前死掉呢?
- 到达屏障(`arrival.$JOBTAG/ready.n*`):`f3d2482` 让 rank 0 在写 `$RDZV` 之前
  `rm -rf` 该目录来清掉上次残留。**这个 `rm -rf` 在共享文件系统上安全吗?**
  时机真的保证"此时不可能有 peer 已经写过标记"吗?
  第一次尝试的 pod 还没死透、正在写标记,而重试的 rank 0 同时在删,会怎样?
- 顺带:该作业第 149 行有一处 `rm -rf "$CKPT/arrival.$JOBTAG"`,路径由环境变量拼出。
  **本任务书作者已核对过**:`CKPT_DIR` 的非空检查在第 79 行、占位符检查在第 80 行,
  都早于第 149 行,且 `set -u` 生效,`JOBTAG` 由 hostname 派生不会为空 ——
  最坏情况也只落在 `$CKPT/` 内部。**这条列在这里是让你复核这个判断,不是说它坏了。**
  但请顺便扫一遍整个分支还有没有**没被这样防护过**的 `rm -rf` / 通配删除。
- `maxRetries: 3` 与 resume 的配合真的成立吗?
- 那段内嵌的绑定校验(sampling config sha / plan_hash / world_size / keep_max)漏了什么?
- **有没有任何路径会让它静默地从错误的权重开始训练?**

### B. TextAtlas GT 进 prompt 这个设计本身
- GT 前置 + i1 caption 在后,会不会导致**同一条 prompt 里出现两份互相矛盾的引文**?
  (i1 caption 也会引用图中文字,而它对权威 GT 只有 73.5% 精确。)这个代价评估得对吗?
- 训练目标用 `gt_text_raw`(未归一化)而不是 `gt_text_normalized`,对吗?
- 三个措辞模板轮换,是帮助还是引入了不必要的变量?
- `compose_textatlas_gt_prompt` 的 token 记账有没有 off-by-one 或边界错误?

### C. checkpoint 保留策略
`checkpoint_keep_every: 30000` + `checkpoint_keep_max: 8`。
`training/checkpoint.py:193` 是 `snaps[:-keep_max]`,对**所有** `step_*.pt` 一起剪。
- 这组值在 250,000 步下真的能全留住 8 个吗?
- `checkpoint_keep_format: weights_ema` 的快照**不含 optimizer、不可续训**,
  只能用于 eval/回滚 —— 这对评测协议够用吗?
- 磁盘够吗?(每个约 13 GB,`last.pt` 约 26 GB。)

### D. 数字与文档的一致性
- 有没有哪份文档 / 注释里的数字与代码或产物实际不符?
  (已知它修过一处:`REALIZED_EXPOSURE.md` 里的 513,802,240 是错的,真值 512,000,000。
  **请找第二处。**)
- `docs/STAGE512B_DECISIONS.md` 里 19 条决策,有没有与代码实际行为矛盾的?

### E. 你认为在开跑之前还**必须**做、但它没做的事
以及反过来:它做了但其实**不必要**、拖慢进度的事。

---

## 6. 报告格式

- 每条发现给:**严重度**(阻塞启动 / 应在启动前修 / 可以启动后修 / 仅记录)、
  **文件:行号**、**具体触发路径**、**你的证据**(跑了什么命令、看到什么输出)。
- 明确区分"我实测确认"与"我读代码推断"。
- 如果你**验证了某条它的说法是对的**,也请列出来(避免下一轮重复验证)。
- 如果你认为某条它标为"已修/已闭合"的其实没闭合,请单独高亮。
- 最后给一句:**以你的判断,现在可以授权启动这个 7.4 天的正式训练吗?** 并说明理由。

---

## 7. 一些会误导你的地方(预先说明,免得你浪费时间)

- `scripts/offline_arb/smoke_offline_loader.py` **对这条路径不可用**(它的 self-test 读一个
  curation 时没导入的旧配置,生产模式又硬写 `assignment_mode=offline_exact_v1`)。
  能用的是 `smoke_anchors_loader_v1.py`。
- `tests/` 目录有 "exact-two closure" 检查,**新测试必须放 `scripts/`**,放 `tests/` 会让检查失败。
- 有 4 个测试文件本来就跑不起来,**不是本轮造成的**:
  `test_offline_gpic_s3_production_static.py`、`test_offline_gpic_tar_sequential_static.py`
  (依赖 curation 时没导入的模块)、`test_stage512a_prep_contract.py`(同)、
  `run_exact_policy_tests.py`(需要参数)。
- `configs/stage512b_real_anchors_v1.yaml` 里的 `pipeline_config_sha256` 漂移是**正常的**
  (只记录不强制,且 plan root 必须等 plan 建好才写得进去)。
  真正被强制的是 catalog 里的 `sampling_config_sha256`。
- `jcc job log <name> -p all` 才能非交互地看日志;`jcc job get` 对正式/GPUVAL 作业**不要用**。
- `git` 需要显式 SSH:仓库已配 `core.sshCommand`,如果 push 失败那是凭据问题,**不要绕过**
  (而且你本来也不该 push)。
