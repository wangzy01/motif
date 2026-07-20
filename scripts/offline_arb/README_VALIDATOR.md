# validate_plan.py — OFFLINE_ARB_V1 L3 Planner 独立验证器

独立于 planner 实现(未阅读 `training/dataline_offline/` 下任何代码),只依据:

- `docs/OFFLINE_ARB_PLANNER_DESIGN.md`(§1 配额求解、§1.5 输入契约、§2-5 构造算法、
  §6 物化格式(2026-07-10 增补 domain_counts)、§7 resume 契约、§8 验证清单 V1-V8、
  **§9 规格补钉 9.1-9.10**)
- `docs/OFFLINE_ARB_V1_SPEC.md`(Phase-0 七条硬口径)

实现。初版验证器提出的 14 条歧义清单已被 §9 大部分裁定钉死(对照表见下);§9 钉死项
一律按文档字面精确实现,不再做启发式别名/形状探测。仍未闭合的少数点列在「遗留歧义」。

## 用法

```bash
python validate_plan.py --input planner_input.json --plan plan_root/ \
    [--rebuild-plan plan_root2/] [--report out.json] \
    [--rebuild-mode {auto,suffix,full}] [--max-blocks N] [--skip-deep-item-checks]
```

- 纯 `numpy` + Python 标准库,不 `import torch`。
- `--rebuild-plan` 给了才跑 V5/V6;`--report` 给了才落 JSON 报告。
- `--skip-deep-item-checks` 跳过逐 item 扫描(V4/VALUE_RANGE/ITEM_BUCKET/SQ_LEAK/
  DOMAIN_COMP),但 domain_counts 的块级校验(DC_STRUCT/C6_STEP/DC_AGG)仍执行。
- 退出码:0 = 无 `FAIL`/`ERROR`(`SKIP`/`WARN`/`INFO` 不影响);非 0 = 存在 `FAIL`/`ERROR`。
- 本机只跑微型合成 fixture(H=4, W=2, bs=3)smoke;真实规模(H≈166k, G=2304)按
  项目纪律经 JCC 提交。

## 每项检查的精确判定定义

| 编号 | 名称 | 精确判定(PASS 条件) |
|---|---|---|
| **INPUT** | §1.5 + §9.9 输入校验 | 契约字段齐备;`Σ domain_mix = 1`(容差 1e-9,**不归一,违者 ERROR**);N 非负整数、形状 `[n_buckets][n_domains]`;`square_bucket_idx` 指向 `h==w` 的桶。 |
| **V1** | 每 step 全 rank 同桶 + 每 rank 恰 bs | 每块 `bucket_of_step` 形状 `(block_len,)`(每 step 单桶值,结构上对全 rank 生效)、dtype **恰为 int16**、值 ∈ `[0,n_buckets)`;`item_row` 形状**恰为** `(block_len, W, bs)`、dtype **恰为 int64**(§9.8 严格:dtype/键名违规 = FAIL,不再宽容 reshape/别名)。 |
| **V1_S_REALIZED** | 各桶实际步数 = header S[B] | 全量 `bucket_of_step` 逐桶计数与 header `S` 相等(仅全块扫描时评估)。 |
| **V2** | square-only 违规 = 0 | ∀ `d∈square_only`、∀ `B≠square_bucket_idx`:header `C[B,d]=0`。 |
| **V3** | 曝光/步数守恒 | `Σ_B S[B]=H`(C1);`Σ_d C[B,d]=G·S[B]` ∀B(C2);`Σ_B C[B,d]=header D[d]` ∀d(C3);`header D[d] == LR(M·m[d])` 独立重算(LR 平局按 §9.1:小数降序,平局索引升序)。 |
| **C4/C5/C6** | cell 级约束 | C4 应零 cell(N=0 或 square-only 非方桶)为零;C5 `C≤R_max·N`;C6 聚合形式 `ceil(C/S)≤N`(S=0 则 C=0)。 |
| **C6_STEP** | §1 C6 逐步形式(增补 (b)) | 每 step `c_t[d] ≤ N[B_t,d]`,c_t 取自 `domain_counts`(或 fallback 重放)。 |
| **DC_STRUCT** | domain_counts 结构(增补 (a)) | 每块 `domain_counts` dtype **恰为 int32**、形状 `(block_len, n_domains)`、非负、每步 `Σ_d = G`。 |
| **DC_AGG** | domain_counts 聚合(增补 (d)) | `Σ_k c_k[d]` 按 `bucket_of_step` 分桶聚合后逐 cell 等于 header `C[B,d]`(仅全块扫描时评估)。 |
| **DOMAIN_COMP** | item↔c_t 逐槽位一致(增补 (c)) | 对每 step:按域主序展开 c_t 得到每个槽位 p 的期望域,经 §9.5 发牌公式(`rank = p mod W, slot = p div W`)对应到 `item_row[step,rank,slot]`,该行号按 §9.3 虚拟 catalog 解码出的域必须等于期望域。c_t 来源:`domain_counts`(主路径,**全部 step 都判,无 ambiguous 跳过**);块内缺失时 fallback 到 §3 Bresenham + §9.4 修正规则重放(报告 DC_PATH 注明用了哪条路径)。 |
| **V4** | 同 step 无重复 item | 每 step 把 `(W,bs)` 展平后行号两两不同——虚拟 catalog 行号全局唯一,扁平去重对"同一 global batch 无重复"充分且必要,不依赖域边界。 |
| **VALUE_RANGE** | 行号值域 | 每个行号 ∈ `[0, Σ_d N_total[d])`(§9.3 虚拟 catalog 全域;布局由 input N 独立构造,无空洞)。 |
| **ITEM_BUCKET** | item 桶 = step 桶(§9.3) | 行号解码 `row_id → (d,B,local)`(`base[d]+cellbase[d][B]+local` 的逆映射,searchsorted 实现),解码出的 B 必须等于该 step 的 `bucket_of_step`——plan 侧零 re-resize 的直接证据。 |
| **SQ_LEAK** | square-only item 级防御 | 解码域 ∈ square_only 的 item 不得出现在 `bucket≠square_bucket_idx` 的 step。 |
| **HDR_REQUIRED** | §9.6 必含字段 | header 含全部钉死字段(format_version, plan_id, seed, world_size, per_rank_bs, total_steps, G, block_steps, domains, buckets, square_bucket_idx, square_only_domains, D, S, C, N_total, R_max, hashes, solver, row_scheme),且 `row_scheme=="virtual_catalog_v1"`、`format_version==1`。 |
| **HDR_HASH** | hashes 字段 | `hashes.input_sha256` 与 `hashes.planner_algo_version` 非空 + `seed` 存在。 |
| **HDR_INPUT_SHA** | input 绑定(字节级) | `hashes.input_sha256` 等于 `--input` 文件实测 sha256;不等只 **WARN**(见遗留歧义 R3)。 |
| **BIND** | input↔header 语义绑定 | header 的 format_version/plan_id/seed/W/bs/H/block_steps/domains/square_only_domains/square_bucket_idx/buckets/R_max/G 与 input 逐项相等;`N_total` 等于从 input N 独立求和的逐域总数。 |
| **CHECKSUMS** | §9.7 结构 + 核对 | checksums.json 必须是 `{"header": hex, "blocks": {filename: hex}}`;`header.json` 实测 sha256 与 `checksums["header"]` 一致;每个期望块存在、在 blocks 表中、且实测 sha256 一致;blocks 表无多余条目。 |
| **V5** | 后缀重建一致(§9.10) | 两 plan_root 同名(公共后缀)块独立重算 sha256 全部一致,不信任任一侧 checksums.json。 |
| **V6** | 双次全构建一致(§9.10) | 仅当两侧块文件名集合相同(auto 判定或 `--rebuild-mode full`)时评估:逐块 sha256 相同 + header 逐字段相同;块集合不同(后缀重建语境)记 SKIP。 |
| **V7** | 不可行输入检测 | 独立重算三条必要条件:F0 零 cell(有需求的域必须有非空合法供给)、F1 §9.2 修订版(square-only 域 `D[d]≤R_max[d]·N[square,d]`)、F2(`D[d]≤Σ_B R_max[d]·N[B,d]`)。`不可行 ∧ plan 完整存在` → **FAIL**;`不可行 ∧ 无 plan` → PASS;`可行 ∧ 有 plan` → PASS(空对照,标 exercised=False);`可行 ∧ 无 plan` → SKIP(§9.2:solver 判定才是权威,F 系列只是必要条件,不妄判)。 |
| **V8** | 耗时/内存 | 验证器自身墙钟 + 峰值 RSS(`ru_maxrss`);planner 构建资源由 JCC 侧另测。 |
| **DC_PATH** | (INFO)域配额来源 | 注明本次运行用 `domain_counts` 物化还是 §9.4 重放回退(`bresenham_replay_9_4`)。 |

## 独立重算的公式(不 import planner)

```text
M = H * G, G = W * bs
D[d] = LargestRemainder(M · m[d])           # §9.1:小数降序,平局索引升序
F0: ∀d, D[d]>0 → eligible(d) > 0            # eligible = N[sq,d](square-only)或 Σ_B N[B,d]
F1: ∀d ∈ square_only: D[d] ≤ R_max[d]·N[square,d]     (§9.2 修订)
F2: ∀d: D[d] ≤ Σ_B R_max[d]·N[B,d]
虚拟 catalog(§9.3): row_id = base[d] + cellbase[d][B] + local
  base[d] = Σ_{d'<d} N_total[d'];cellbase[d][B] = Σ_{B'<B} N[B',d]
发牌(§9.5): 槽位向量按域主序展开;p → rank = p mod W, slot = p div W
```

## 初版 14 条歧义 → §9 裁定对照

| 初版歧义 | §9 裁定 | 验证器现状 |
|---|---|---|
| #1 header 无 JSON schema | §9.6 钉死字段名 | 按精确名实现,别名探测已删除 |
| #2 LR 平局规则未写 | §9.1(与初版假设一致) | 无需改动 |
| #3 F1 的 S_max_possible 无公式 | §9.2 废弃 S_max,改 `D[d]≤R_max·N[sq,d]` | 已重写 F1 |
| #4 item_row 行号口径不明 | §9.3 虚拟 catalog(钉死) | local/global 启发式已删除;实现解码 + 新增 ITEM_BUCKET 检查 |
| #5 ±1 修正规则不可重构 | §9.4 精确规则 + §6 增补 domain_counts 物化 | 主路径直接用 dc(全部 step 可判,无 ambiguous 跳过);fallback 实现 §9.4 重放 |
| #6 V7 必要非充分 | §9.2 明示 solver 判定是权威 | 语义保留(可行∧无 plan → SKIP 不妄判) |
| #7 block_steps 非必含字段 | §9.6 列为必含 | 缺失 = HDR_REQUIRED FAIL |
| #8 checksums 结构未定义 | §9.7 钉死 | 严格实现 + 增加 header.json 自身 sha256 核对 |
| #9 dtype 只 WARN | §9.8 严格 | dtype/键名违规 = FAIL,宽容 reshape/轴匹配已删除 |
| #10 发牌公式是文字反推 | §9.5 钉死(与初版假设一致) | 无需改动 |
| #11 mix 求和未强制 | §9.9(容差 1e-9,不归一,报错) | 归一化逻辑已删除,违者 INPUT ERROR |
| #12 V2 计数粒度 | 未明示 | 维持"违规 cell 数"口径(遗留 R5) |
| #13 V5/V6 判法被任务描述合并 | §9.10 分别钉死 | auto 模式按块集合区分,语义与 §9.10 对齐 |
| #14 sq idx 与桶形状矛盾的处置 | §9.9(必须 h==w) | 违者 INPUT ERROR(原先只记歧义) |

## 遗留歧义(§9 之后仍未闭合,编号 R1-R5)

1. **R1 — D/S/C/N_total 的 JSON 值形状未钉死**:§9.6 只给了字段名与语义("逐域"/"逐桶"/
   "桶×域"),没给 list vs dict。本工具接受按 domains/buckets 顺序对齐的 list,同时容忍
   D/R_max 用 domain 名做 key、S 用桶索引字符串做 key 的 dict。
2. **R2 — §9.4 重放的两个微观点**:C_rem 的计时点(取"含本步暂定 c_k 的累计消费"还是
   不含)与 −1 方向是否守护 `c_k[d]>0`(不守护会产生负配额)未写明。本工具取"含暂定"+
   "守护非负"。**仅影响 fallback 路径**(块内无 domain_counts 时);主路径直接消费物化的
   domain_counts,完全不受影响——这正是 §6 增补的意义,重放不一致时报告只标 ambiguous
   跳过并提示换带 domain_counts 的 planner 版本重建。
3. **R3 — hashes.input_sha256 的字节级语义**:若验证器拿到的 planner_input.json 是重新
   序列化的副本(键序/空白不同),字节级 sha256 会不同但语义相同。本工具字节级不匹配只
   WARN(HDR_INPUT_SHA),语义绑定由 BIND 检查负责 FAIL。
4. **R4 — header.block_datasets 字段的地位**:协调方说明 header 将新增该字段,但 §9.6
   必含清单未收录。本工具按可选处理:声明了就以它为准(声明含 domain_counts 而块内缺失
   = FAIL,反之亦然);未声明则以块内实际内容为准,且全部块必须一致(混合存在 = FAIL)。
5. **R5 — V2 计数粒度**:仍未明示是"违规 cell 数"还是"违规曝光量"。本工具 V2 用 header
   层面违规 cell 数,item 级证据另见 SQ_LEAK(违规 item 出现次数),两者分开报告。

## 已知局限(范围边界,非歧义)

- 不重实现 §1 求解器与 §2 桶步排列的最优性判定,只验证聚合守恒(C1-C6)与物化一致性;
  §9.2 明确 solver 的不可行判定是权威,本工具的 F0/F1/F2 只是独立的必要条件复核。
- 不读图、不碰 L1/L2,只消费 planner_input.json 的 N[B,d](Phase-4 口径);§9.3 虚拟
  catalog 行号仅由 N 派生,生产期 catalog 行序按同规则物化后本工具无需改动。
- 大规模验证走 JCC;本机仅跑合成 fixture 回归(14 个场景:全绿 1 + fallback 全绿 1 +
  注入违规 9 + V5/V6 对照 2 + 不可行输入对照 1×2 形态)。
