# OFFLINE_ARB_V1 · L3 Planner 设计(v1,2026-07-09 夜)

> 作者:主线(fable5)。实现将委托 Sol,验证器由独立 sonnet 仅凭本文实现(实现者≠验证者)。
> 上游:docs/OFFLINE_ARB_V1_SPEC.md(七条口径);输入依赖 L1 catalog + L2 assignment 的 N[B,d]。
> 本文所有记号:B=桶,d=域,W=world_size(16),bs=per-rank batch(144),G=W·bs(2304),
> H=总 global step 数,M=H·G=总曝光,m[d]=目标域配比,N[B,d]=cell 内唯一 item 数。

## 0. 设计原则

1. **一切随机性 = 确定性伪随机**:整个 plan 是 (catalog_hash, assignment_hash, plan_config, seed) 的纯函数;同输入两次构建逐字节相同。
2. **配额靠整数记账,不靠抽样**:无 multinomial;所有取整用确定性 largest-remainder / Bresenham。
3. **不可行 → plan-time 失败**:任何约束(square-only、cell 容量、repeat 上限)不满足,构建期报错并输出診断,绝不留到训练期降级。
4. **零 re-resize 由构造保证**:batch 内 item 全部来自该 batch 的桶 B_t 的 native cell;planner 层根本不存在跨桶取数的表达能力。

## 1. 配额求解(S[B], C[B,d])

理想联合曝光(保各域 native AR 分布、精确 mix):

```
D[d] = LR(M·m[d])          # 域总曝光,largest-remainder 取整,Σ_d D[d] = M
Q[B,d] = D[d] · N[B,d]/N[d]  # cell 理想曝光(实数)
```

求整数 S[B](桶 B 的 global step 数)与 C[B,d](cell 总曝光),约束:

```
(C1) Σ_B S[B] = H
(C2) Σ_d C[B,d] = G·S[B]           ∀B
(C3) Σ_B C[B,d] = D[d]             ∀d
(C4) C[B,d] = 0  当 N[B,d]=0 或 (d∈square_only 且 B≠square)
(C5) C[B,d] ≤ R_max[d]·N[B,d]      (repeat 上限,配置项;默认 R_max 按域给)
(C6) per-step 可行:c_t[d] ≤ N[B,d](见 §3;等价于 C[B,d]/S[B] 不得超 cell 容量,
     防同一 global batch 内 item 重复)
目标: min Σ_{B,d} w_d·|C[B,d] − Q[B,d]|   (w_d=1/D[d],各域相对偏差等权)
```

**求解器**:cell 数 ≤ 95×5=475,规模极小。首选 `scipy.optimize.milp`(HiGHS,venv 内确认版本);
不可用则回退「两级 largest-remainder + 运输修复环」:
S[B]=LR(Σ_d Q[B,d]/G) → 每桶内 LR 分配 C[B,·] → 对 (C3) 的残差在同 d 列内沿
(B↑,B↓) 做 ±1 循环修复(每次挑对目标函数损害最小的 cell 对),至残差归零。
两条路径都要过同一套 feasibility 前置检查:

```
F1: D[sq_only 域合计] ≤ G·S_max_possible[square]   (square 桶步数上界够不够吃下方图域)
F2: ∀d: D[d] ≤ Σ_B R_max[d]·N[B,d]                 (repeat 上限下总供给)
F3: 解出后逐 cell 验 (C4)(C5)(C6)
```

失败输出:哪条约束、差多少、按什么调整可行(调 m[d] / H / R_max / 桶表)——给 Evan 的诊断报告。

## 1.5 输入契约(planner_input.json,实现者与验证者共同接口)

```json
{
  "format_version": 1,
  "plan_id": "str(由调用方给,进 seed)",
  "seed": 20260709,
  "world_size": 16, "per_rank_bs": 144, "total_steps": 166667,
  "domains": ["anime","monet_real","gpic","synthetic","textatlas"],
  "domain_mix": {"anime":0.27, "...":0},
  "square_only_domains": ["synthetic","textatlas"],
  "buckets": [[h,w],...],
  "square_bucket_idx": 48,
  "N": [[...],...],
  "R_max": {"anime":20,"monet_real":20,"gpic":20,"synthetic":200,"textatlas":200},
  "epsilon_jitter": false,
  "block_steps": 1024
}
```
`N` 形状 = [len(buckets)][len(domains)],行=桶列=域(顺序严格对齐 buckets/domains)。
CLI:`build_plan.py --input planner_input.json --out plan_root/`;
验证器:`validate_plan.py --input planner_input.json --plan plan_root/ [--rebuild-cmd ...]`,
退出码 0=全过,非 0=违规(打印 V1..V8 逐项)。

## 2. 桶步排列(bucket step schedule)

Bresenham/deficit 式低差异序列,禁 iid 长 burst:

```
deficit[B](t) = S[B]·(t+1)/H − used[B]
step t 选 argmax deficit(平局按 B 序号)
```

性质:任意前缀 t 内桶 B 出现次数与 S[B]·t/H 差 <1;确定性;与 seed 无关(顺序无随机需求——
如需打散规律性,可选配在 deficit 加 seeded ε<1/H 抖动,默认关)。

## 3. 桶内 per-step 域配额(c_t[d])

桶 B 的第 k 个 step(k=1..S[B]):对每个 d 做域内 Bresenham:

```
c_k[d] = floor(C[B,d]·k/S[B]) − floor(C[B,d]·(k−1)/S[B])
```

自动满足 Σ_k c_k[d]=C[B,d];再验 Σ_d c_k[d]=G(Bresenham 保证总和逐步均衡,若个别 step
±1 偏差,用同 step 内对 |c−Q/S| 最不敏感的域做 ±1 修正,确定性规则)。
(C6) 即 max_k c_k[d] ≤ N[B,d],已在 §1 前置检查覆盖。

## 4. 槽位→rank 切分

step 内把 Σc_t[d]=G 个槽按域主序展开成槽位向量 v[0..G),然后 **交错发牌**:
rank r 拿 v[r], v[r+W], v[r+2W], …(每 rank 恰 bs 个)。
性质:各 rank 的域构成差 ≤1;确定性;解码代价均衡(同域连续块被打散到全 rank)。
(Sol 提的按 encoded bytes 均衡属 P2 优化,v1 不做,监控里留每 rank decode 时间指标。)

## 5. item 指派(cell 内无放回)

对每个 cell (B,d):

```
perm_c = PRP(seed_key = SHA256(plan_id, B, d, cycle)) 下 [0..N[B,d]) 的确定性置换
消费指针沿 C[B,d] 递增;耗尽 → cycle+1,新置换
```

- 置换用 Feistel/PRP 或 numpy PCG64(seed_key) 的 permutation——**实现必须写死算法与版本**,
  跨 numpy 版本可复现(倾向自实现 Feistel-on-index,零依赖)。
- 同 global batch 无重复:由 (C6) 保证 cycle 边界不会在单 step 内回绕同 item;验证器仍显式查。
- crop offset / caption 选择 / dropout 等随机增强:训练期由 (plan_id, step, rank, slot) 无状态派生,
  不写进 plan(plan 只存 item 身份)。

## 6. 物化格式

分块存储,块=1024 global steps:

```
plan_root/
  header.json      # 全部 hash(catalog/assignment/config/seed)、W、bs、H、G、D[d]、S[B]、C[B,d]、
                   # solver 记录(目标值/修复步数)、格式版本
  blocks/block_{i:06d}.npz   # bucket_of_step[1024]:int16;item_row[1024,W,bs]:int64(catalog 全局行号);
                             # domain_counts[1024,n_domains]:int32(=c_t[d] 物化;2026-07-10 增补:
                             # 让 V4/泄漏/值域判定不依赖 §3±1 规则重构,验证由构造闭合,~3.3MB@166k)
  checksums.json   # 每块 sha256
```

~3.1GB@384M 曝光(int64),mmap 按需读当前±少量块。训练器只需:`plan.batch(step, rank) → (bucket_hw, row_ids[bs])`。

## 7. Resume 契约

- plan 与 (W, bs) 绑定,header 校验不符即拒载。
- checkpoint 只记 `global_step`(现契约不变,零迁移)。**语义澄清(2026-07-10 审查仲裁,消 off-by-one 歧义)**:train.py 在 optimizer step 完成后才 `global_step += 1`,checkpoint 存的是**已完成步数(计数 N)**,已消费 plan step 0..N-1;loader 冷启动定位 **plan step == global_step(0-based,不再 +1)**,预取窗口作废重建。旧文"读 step+1 的块"中 step 指最后完成步的 0-based 序号(N-1),数值等价;实现以 `offline_dataset.set_start_step(N) → plan step N` 为准(smoke self-test 断言 set_start_step(1) 吐 plan step 1)。注意 planner_v1 的 `first_step_after(completed)` 是序号+1 语义,入参须是序号而非计数,loader 有意不用它。
- 换 world size = 重建 plan(新 plan_id),同一 catalog/assignment 下数据集合不变、顺序改变——语义上是新 run(符合 D6/口径⑦)。

## 8. 验证器(Phase-4 gate,独立实现)

对小合成 N[B,d] + 真实 pilot N 采样各跑:
V1 每 step 全 rank 同桶、每 rank 恰 bs;V2 square-only 违规=0;V3 Σ 域曝光=D[d] 精确、
Σ_B S[B]=H、joint C 与解器输出一致;V4 同 batch 无重复 item;V5 任意 t 处重建 → 后缀与原 plan 逐字节一致;
V6 双次全构建 hash 相同;V7 构造不可行输入(方图域配比超 square 桶容量 / R_max 过小 / 空 cell 需求)
→ 必须 plan-time 报错且诊断信息完整;V8 大规模(H=166k,G=2304)构建耗时与内存实测。

## 9. 规格补钉(2026-07-09 夜;来源=独立验证器作者的 14 条歧义清单,以下为裁定)

1. **largest-remainder 平局规则(全文统一)**:小数部分降序,平局按索引升序。
2. **F1 修订**(原 S_max_possible 无公式,废弃):square-only 域的紧必要条件是
   `D[d] ≤ R_max[d]·N[square,d]`(该域只能吃 square 桶);F2 不变;**solver 不可行判定是权威**,
   F1/F2 只为更早更好的诊断。验证器 V7 用 F1(修订版)+F2+零 cell 必要条件独立判。
3. **虚拟 catalog 行号(生产与 Phase-4 统一)**:`row_id = base[d] + cellbase[d][B] + local_idx`,
   其中 base[d]=Σ_{d'<d} N_total[d'],cellbase[d][B]=Σ_{B'<B} N[B',d](桶主序 cumsum)。
   仅由 N 派生,无需真 catalog;任何 row_id 可唯一解码回 (d,B,local) → 验证器可精确核
   「item 的桶 = step 的桶」与全局去重。生产期 catalog 行序必须按同规则物化,两者语义合一。
4. **§3 的 ±1 修正精确规则**:若 step k 内 Σ_d c_k[d] ≠ G,逐单位调整:每次挑「剩余配额
   C_rem[B,d] 最大」的域 ±1(平局按域索引升序),重复至相等。确定性、可重放。
5. **§4 发牌公式**:槽位向量 v 按域主序(输入 domains 顺序,域内按消费顺序)展开;
   全局位置 p 的 item → rank = p mod W,槽 = p div W。
6. **header.json 必含字段**:format_version, plan_id, seed, world_size, per_rank_bs, total_steps,
   G, block_steps, domains, buckets, square_bucket_idx, square_only_domains, D(逐域), S(逐桶),
   C(桶×域), N_total(逐域), R_max, hashes{input_sha256, planner_algo_version}, solver{...},
   row_scheme="virtual_catalog_v1"。
7. **checksums.json 结构**:`{"header": sha256hex, "blocks": {"block_000000.npz": sha256hex, ...}}`。
8. **dtype 严格**:bucket_of_step=int16,item_row=int64;npz 内键名 `bucket_of_step`/`item_row`。
9. **输入校验**:Σ domain_mix = 1(容差 1e-9,不自动归一,违者报错);N 非负整数;
   square_bucket_idx 对应 buckets[h==w]。
10. **V5/V6 口径**:V6=同输入双次全构建,块文件名集合必须相同且逐块 sha256 相同+header 逐字段相同;
    V5=从任意 t 起重建(planner 支持 --start-step)后,公共后缀块逐 sha256 相同。

## 11. L1/L2 catalog 物化契约(2026-07-10,配 D7 桶表拍板)

- **行序 = §9.3 虚拟 catalog 序的物化**:全表按 (domain 序=§1.5 domains, bucket_idx 0..94, cell 内序) 排列,
  row_id = 全局行号 = base[d]+cellbase[d][B]+local。**cell 内序 = uid 字典序**(确定性,钉死)。
- bucket_idx = **折叠后**指派(D7:floor=0.005 kept 集,nearest_allowed_bucket 折叠;square_only 域恒 square)。
  与 make_planner_input --floor 0.005 的 N 完全同口径——catalog 逐 cell 计数必须精确等于该 N(构建器自检项)。
- 每行字段:row_id, uid, domain, bucket_idx, width, height + 域特定 locator
  (monet/synth: tar_path+offset+size(join 自 offline_arb_tarindex_v1)+caption 源 locator;
   anime: fetch key/url + cluster;gpic: 文件路径;textatlas: shard+row_group+row_in_group)
  + caption/条件字段按现行 iter loader 产出 rec 所需(文本过大时存 locator,由 fetch adapter 随图取回)。
- 产物:offline_arb_catalog_v1/(分域 parquet + header.json 含 catalog_hash/桶表 hash/floor/行数/N 校验)。
- 一致性自检(构建器内置):①逐 cell 计数 == make_planner_input(floor=0.005) 的 N(**join 排除前**口径;
  排除后偏差单独记 N_diff.json);②row_id 连续无洞;③monet/synth 每行 locator 在 tar 索引中命中,
  **容忍 ≤0.1% miss 进 bad_rows 排除(D10 拍板,与 taridx 覆盖率 gate 同口径),超过即整体报错**;
  ④与 Phase-1 Task D 的 uid 唯一性口径一致。
- **精确 N 闭环(D10 硬约束)**:最终生产 plan 必须用 make_planner_input **--catalog-header** 模式构建——
  N 取 header.json 的 N.materialized_after_exclusion,catalog_hash/assignment_hash 随 planner_input 落
  plan header(loader 拒 proxy hash);禁止再从 N_matrix.parquet 现算 N 直接建生产 plan(排除行会导致
  plan/catalog N 不一致,loader 启动期报错)。

## 12. Phase-5 loader 契约(offline_exact_v1,2026-07-10)

- **入口**:新类 `training/dataline_offline/offline_dataset.py::OfflineExactDataset`,配置开关
  `dataset.multi_ar.assignment_mode: offline_exact_v1` 时由现 store_cls 工厂分发(train.py 零改动优先;
  若必须改 train.py,改动限于工厂分发一处)。streaming 路径保留为回退(D3 feature-flag 语义)。
- **接口对齐**(reuse_map 侦察结论):`loader(batch_size)` 生成器产 `{prompts, pixels, cfg_keep}`;
  `set_start_step(global_step)` → plan step 定位(checkpoint 契约不变:只存 optimizer step);
  metrics 鸭子字段 `buffer_fill` 挂同名属性(新增 plan_step/prefetch_depth/per-domain supply 指标)。
- **取数后端**(Phase-2b 实证定型):monet/synth = **coalesced range**(按 plan 窗口把 wanted 按 tar 分组、
  offset 排序、<4MB 间隙合并,1093 img/s 实证;tar locator 来自 catalog);gpic = **深预取管线**
  (32 线程/节点 → /local 滚动窗口,88.2 img/s,D8 已接受;调优线结果可回填并发数);
  anime = 现行逐图 fetch+cache 复用(364 img/s@64);textatlas = row-group 直读(1635 img/s@8,
  无全局锁版本;/local staging 保留为后备)。
- **预取深度**:per-rank 预取窗口 ≥ 4 个 global step(可配),按 plan 顺序流水;plan 确定性 →
  预取器只读 plan[step+k],无投机浪费。
- **单次 resize/crop**:record 按 catalog bucket_idx 取原图 → `resize_u8_to_bucket`(v_anchor 沿用
  config)一次到位;shape 断言 `(b_h,b_w)`;与 plan 的 bucket_of_step 不符 = fail-fast(口径①⑤)。
- **caption/增强**:`_finalize_caption`/`_is_cfg_protected` 复用;随机性 seed=f(plan_id,step,rank,slot)
  无状态派生(§7)。坏 item(decode 失败/尺寸与 catalog 不符)fail-fast(口径⑤)。
- **验证阶梯**:CPU 单 rank 8 step 冒烟(JCC)→ CPU 16-rank 仿真(stress harness 改造)→
  2×8 GPU ≤120 step(D6 权限内)→ save→中断→resume→item trace 对比(Phase-6)。

## 10. 开放项(晨报待拍板)

- R_max[d] 默认值(建议:anime/monet/gpic ≤ 20,synthetic/textatlas ≤ 200,依 D[d]/N[d] 实际推荐);
- 桶表(Phase-3 全量 N[B,d] 出来后 Evan 定);
- ε 抖动开关默认(建议关);
- plan 块大小(1024 步 ≈ 2.4M 曝光/块,可调)。
