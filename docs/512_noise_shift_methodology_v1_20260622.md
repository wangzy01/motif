# 512 noise-shift 取值方法学协议 v1 (B7c)

日期: 2026-06-22  作者: Claude (B7c task)  状态: 方法学文档, 不含已提交作业, 512 具体取值留给 Evan 拍

> ✅ **取值已定（2026-06-29，Evan 拍）**：见 [数据+训练阶段_重设计方案_v1_20260629](数据+训练阶段_重设计方案_v1_20260629.md) §6——**512 = 1.0（不 shift，对齐 i1 实测）、1024 = 3.33**（落地前仍跑下方协议 A 的 fixed-sigma eval 验证）。本文方法学（协议 A/B）仍有效，仅取值由上式替代。
>
> ⚠️ ~~**shift 决策待定（截至 2026-06-27 未拍）**: `stage1a_t2i_512.yaml:68` 现 `shift=2.0`（"sweep 起点"占位，**非最终值**），文档弱偏 ~3.33。取值须走下方协议 A 实测后由 Evan 拍。在拍板前，512 的 shift=2.0 视为占位。~~（已由上方取代）

---

## 0. 一句话结论 (先看这个)

**不能用单一固定 ckpt + `--overrides sigma_sampling.shift=X` 来决定 512 的 shift。**
`fixed_sigma_eval.py` 对一个【已训好的模型】在【固定 sigma 网格】上算 MSE,它**根本不读 `sigma_sampling.shift`**(见 §3 核实)。
shift 只通过**训练时见到的 sigma 分布**间接改变权重 → 要比 shift, 必须**比在不同 shift 下短训出来的不同权重**,再用本脚本在同一固定 sigma/数据/噪声上对比 MSE 曲线(曲线低者胜)。这正是已有 `jobs/shift_eval.yaml` 的做法。

因此本文档给两条协议:
- **协议 A (推荐, 真正能决定 512 shift)**: 在 512 上对每个候选 shift 各短训一段 → fixed-sigma eval 对比。基本等同已有 `jobs/shift_eval.yaml`(idx0/idx1)。
- **协议 B (本 B7c 交付的 dry-run, 256 ckpt 代理)**: 用现成 v1 256 ckpt 跑 `fixed_sigma_eval.py` across shift∈{1.0,2.0,3.33}, **预期三条曲线完全相同**(脚本不用 shift)。价值: (i) 验证 eval 管线/确定性; (ii) 用实测钉死"override 改 shift 在本脚本里无效", 避免后续误用单 ckpt + override 去"选 shift"。

---

## 1. 背景: 为什么需要这套协议

- 原始 `exp4_noiseshift` 用各自 shift 采样分布上的**训练 loss** 比 shift, 被 sigma 分布污染: 高 shift 把更多采样质量推到高噪声区(`σ'=s·σ/(1+(s-1)σ)` 把 σ 整体往 1 推), 高噪声区 MSE 本就高/低不一, 训练 loss 与 sigma_mean 强相关 → "shift 大 loss 低/高"含假象, 不可比。
- `fixed_sigma_eval.py` 的设计就是消这个 confound: 固定 sigma 网格 + 固定 eval batch + 固定噪声, 对两个模型在**完全相同**的输入上算 rectified-flow v-pred MSE。同分辨率不同 shift 训出的两个模型 → 干净可比, 曲线低者好。
- 现状: `configs/stage1a_t2i_512.yaml` `sigma_sampling.shift: 2.0`(写本文时第 60 行, 现第 68 行; 注释"sweep 起点; 随 VAE 下采样因子再核")。社区/文档弱偏 ~3.33(SD3 在 1024 用 3.0; FLUX 高分辨率更高; nano/Tsubaki 系经验 512 在 2-3.33 区间)。本协议给出"如何用数据把 2.0 vs 3.33 之争落到 MSE 曲线上"的流程。

---

## 2. 协议 A (推荐): 512 上短训 per-shift → fixed-sigma 对比

### 2.1 用哪些 ckpt
- **不是现成 ckpt**, 而是**为本实验新训的短段权重**: 对每个候选 shift, 在 512 上从同一起点(从零, 或从 256 v1 ckpt 暖启, 二选一但所有 shift 必须一致)训相同步数, 各自存一个 `last.pt`。
- 候选 shift 集合: **{2.0, 3.33}** 为主对比(现配置值 vs 文档弱偏值); 可选加 **3.0**(SD3-1024 经典值)作中间点。

### 2.2 比哪些 shift 设置
| label | shift | 来源 |
|---|---|---|
| 512_s2.0 | 2.0 | `stage1a_t2i_512.yaml` 现值 (reconfirm 注释) |
| 512_s3.0 | 3.0 | SD3 在 1024 的经典 shift, 作中间锚 (可选) |
| 512_s3.33 | 3.33 | 文档弱偏值 |

注: 这些 shift 是**改训练**的, 通过 `train.py ... sigma_sampling.shift=$SHIFT` 命令行 override 注入(train.py 走 `model_mot.get_sigmas` → `current_shift` 读 `sigma_sampling.shift`, 见 §3)。

### 2.3 sigma 网格 / eval batch / 噪声 (评估端, 对所有 shift 完全相同)
- **sigma 网格**: `fixed_sigma_eval.py` 默认 `0.1,0.2,...,0.9` (9 点)。512 建议加密低噪声端到 `0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9`(shift 主要重塑 σ 在 0.3-0.8 区间的密度, 多点更能看出曲线形状差)。所有 shift 用同一网格。
- **eval batch**: `--n-eval 48`(默认), `--read-rows 3000`。512 取图需 `min(img.size) >= 512`, 命中率比 256 低, 若 batch < 8 触发 assert → 调大 `--read-rows`(如 6000)。所有 shift 用**同一 eval shard + 同一 seed** → 逐图一致。
- **固定噪声**: `--seed 1234`(默认), 脚本用 `torch.Generator(cpu).manual_seed(seed)` 生成噪声, 所有 shift/所有 sigma 共用同一噪声样本。
- **weights**: 用 `--weights ema`(EMA 权重, 与单卡离线采样一致), 所有 shift 一致。

### 2.4 判定规则
- 对每个 shift 得 `loss_by_sigma` 曲线 + `mean_loss_over_grid`。
- **主判据**: `mean_loss_over_grid` 低者胜(同 batch/噪声/sigma, 纯比权重质量)。
- **辅判据**: 看曲线形状 —— 若某 shift 在中-高噪声端(σ∈0.5-0.8, 即 512 真正关心的去噪起步区)系统性更低, 即使均值接近也偏好它(512 生成轨迹起点在高 σ, 高 σ 预测准 → 大结构更稳)。
- **置信**: 两曲线差应 > 噪声地板。建议每 shift 跑 2 个 seed(`--seed 1234` 与 `--seed 5678`, 各自换 eval batch/噪声)看排序是否稳定; 排序翻转则判"无显著差", 保留现值 2.0 或按生成质量(实际采样图)再裁。

### 2.5 局限与建议: "256 ckpt 代理 vs 512 小训练"
- **256 ckpt 代理 (协议 B) 不能决定 512 shift**, 原因有二:
  1. **脚本层面**(致命): `fixed_sigma_eval.py` 不读 shift → 单 ckpt 改 shift override 输出不变(§3)。代理只能验管线, 不能选值。
  2. **即便换"256 不同 shift 训出的不同 ckpt"来比**: 256 当前训练用 shift=1.0(`stage1a_t2i_256.yaml`), 且 256 的最优 shift 与 512 不同(latent token 数 256 vs 1024, σ 重标定需求随分辨率上升)。256 上选出的 shift 不能直接搬到 512。
- **建议**:
  - 决定 512 shift **必须**用协议 A(512 上短训 per-shift)。已有 `jobs/shift_eval.yaml` 的 idx0(512_s3.0)/idx1(512_s3.33) 就是这个; 只需补一个 512_s2.0 worker 即可直接对比现配置值, 这是最低成本路径。
  - 短训步数偏少(`shift_eval.yaml` 用 350 步)→ 早期信号弱, 只看**相对**高低不看绝对; 若 350 步排序不稳, 提到 1-2k 步重跑。
  - 本 B7c 只交付协议 B 的 dry-run(下方 §4 的 yaml), 用来固化管线 + 钉死方法学结论。**512 取哪个值(2.0/3.0/3.33)留给 Evan 看协议 A 结果后拍。**

---

## 3. 核实: `fixed_sigma_eval.py` 能否经 `--overrides` 改 shift?

**核实结论: 语法上能 merge, 但对 eval 计算无效(脚本不读 shift)。**

证据(逐行):
- `fixed_sigma_eval.py:49-50` 定义 `--overrides`(dotlist); `:64-65` 执行 `cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))`。所以 `--overrides sigma_sampling.shift=3.33` **能**把 cfg 里的值改掉(用于架构消融开关如 `model.text_adapter.enabled=true` 是正确用法)。
- shift 字段位置: `stage1a_t2i_512.yaml` / `stage1a_t2i_256.yaml` 的 `sigma_sampling.shift`(512 现第 68 行=2.0; 256 第 55 行=1.0)。
- **但脚本的 eval 计算路径不碰它**:
  - `fixed_sigma_eval.py:69` 直接从 `--sigmas` 取固定网格; `:133-138` 对每个网格 σ 算 `x_t=(1-σ)x0+σ·noise`、`v_pred = model.model(...)`、`mse`。
  - 它调用的是**裸 backbone `model.model(...)`**, 不是 `model.forward`, 也不是 `model.get_sigmas`。
  - shift 在代码里只出现在两处, **本脚本都没走**: (a) `model_mot.py:270-286 get_sigmas` 训练采样, 由 `forward`(`:303`)调用; (b) 采样器 σ schedule(`:398-427`)。`current_shift()`(`:262-268`)读 `sigma_sampling.shift`, 但只被这两处调用。
- ⟹ 同一 ckpt 下, `--overrides sigma_sampling.shift=1.0` 与 `=3.33` 产出**逐位相同**的 `loss_by_sigma`。

**推论**: shift 的影响只能体现在**训练出的权重**里。要让 shift 在 fixed-sigma eval 中显形, 必须比"在不同 shift 下训出的不同 ckpt", 不能比"同一 ckpt 套不同 shift override"。`jobs/shift_eval.yaml` 把 shift 注入的是 `train.py`(训练侧), 不是 eval 侧 —— 印证了这一点。

---

## 4. 协议 B dry-run 作业 (本次交付, 不提交)

文件: `jobs/eval_512shift_methodology.yaml`(见该文件)。单卡, 只读, 用现成 v1 256 ckpt
(`/fsx/evan/checkpoints/mot_pilot_v1/s256_1x8_v1_20260618/last.pt`, ~step 114620, shift=1.0 训练)。
在 **resolution 256** 上对该 ckpt 跑 `fixed_sigma_eval.py` 三次, 分别 `--overrides sigma_sampling.shift=1.0 / 2.0 / 3.33`, 同 sigma 网格/同 seed/EMA 权重。

**预期产物与解读**:
- 三个 json (`eval_s256proxy_shift{1.0,2.0,3.33}.json`) 的 `loss_by_sigma` 与 `mean_loss_over_grid` 应**完全相等**。
- 这就是协议 B 的**预期阳性结果**: 实测证明"override 改 shift 对本脚本 eval 无效", 把 §0/§3 的方法学结论用一次运行钉死。若三者**不**相等 → 说明脚本/缓存有 bug, 需排查(不应发生)。
- 同时验证: eval 管线在单卡可跑通、256 取图缓存命中、EMA 权重加载正常、确定性 batch 成立 —— 为协议 A 的正式 512 对比铺路。

提交方式(留给 Evan): `jcc job submit jobs/eval_512shift_methodology.yaml`(本任务**不提交**)。

---

## 5. 给 Evan 的下一步建议 (摘要)

1. (可选) 跑协议 B dry-run, 确认管线 + 看到三 shift MSE 相同 → 固化"不能用单 ckpt override 选 shift"。
2. 决定 512 shift **走协议 A**: 在已有 `jobs/shift_eval.yaml` 基础上补一个 `512_s2.0` worker(把 numWorkers 调整或复用空 idx), 让 512_s2.0 / 512_s3.0 / 512_s3.33 三段同步短训 → fixed-sigma 对比 `mean_loss_over_grid` + 曲线形状。
3. 取值由 Evan 拍: 若 3.33 显著优于 2.0 → 把 `stage1a_t2i_512.yaml` 的 `sigma_sampling.shift` 改 3.33(注意: 改这个文件会被在跑训练 maxRetries 重启读到, **属硬约束禁改区, 须 Evan 亲自/确认后改**); 若不显著 → 保留 2.0。
4. 步数不足导致排序不稳时提到 1-2k 步重跑, 或用实际采样图(`sample_current_s256_runs_html.py` 的 512 版)做最终裁决。
