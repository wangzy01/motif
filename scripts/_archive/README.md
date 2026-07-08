# scripts/_archive/ — 归档脚本

一次性 / 诊断类脚本归档（行为不变重构 · 任务 A2）。归档 = 仅移动文件位置，**未修改任何脚本内容**，便于回溯历史；不再属于活跃工具集。

归档日期：2026-06-24

## 目录结构

```
_archive/
├── diagnostics/        诊断 / 探测类（排查问题用，非训练/数据管道一部分）
└── one_time_scripts/   一次性脚本（特定时点跑完即弃）
```

## diagnostics/

| 脚本 | 作用 | 归档原因 |
|---|---|---|
| `nccl_diag.py` | NCCL/分布式连通性诊断 | 一次性排障探测，无任何 job/代码引用 |
| `probe_sdpa_backend.py` | G2：bool-mask SDPA 后端探测（确认走 cuDNN 不退化 MATH）+ 吞吐 | 架构决策已定（暂不迁 attention），探测结论已写入 README #12 |
| `debug_mask_positions.py` | mask / MRoPE 可视化 + 断言 | 一次性可视化诊断；逻辑回归已由 `test_arch_changes.py` 等单测覆盖 |
| `audit_probe_trunc_repeat.py` | 截断 / 重复 token 审计探测 | 一次性数据审计探测 |
| `verify_anime_priority_fix.py` | 验 anime/naifu 优先级修复（真值谓词命中率核对） | 一次性修复验证，结论已写入 README #16 区段 |

## one_time_scripts/

| 脚本 | 作用 | 归档原因 |
|---|---|---|
| `overfit_reconstruct.py` | G6：GPU 端到端过拟合重建（采样符号/σ 方向最强信号） | 一次性端到端正确性验证，结论（MSE 0.041 vs 噪声基线 2.002）已写入 README #13/#16 |
| `smoke_vae_cache.py` | VAE latent cache smoke（每域每 stage 小样本，验 encode/写盘/读回/在线对比） | 一次性 smoke；全量缓存管道脚本（`vae_cache_flux2_moments.py`、`vae_cache_qc.py`）保留在 `scripts/` |

## 引用核查（移动前已确认）

逐个候选 `grep` 了 `jobs/`、`configs/`、全仓 `*.py/*.md/*.sh/*.yaml`，并核对是否被作为模块 `import`。

- **以上 7 个归档脚本：均无任何 job 执行引用、无代码 import 依赖。** 仅存在于 README/docs/注释中的**文字性引用**（描述性，非调用）。
- README.md、docs 中对这些脚本的路径提及（如 `scripts/probe_sdpa_backend.py`、`scripts/debug_mask_positions.py`、`scripts/overfit_reconstruct.py`、`scripts/verify_anime_priority_fix.py`，以及 `docs/vae_latent_cache_v1_20260622.md` 对 `scripts/smoke_vae_cache.py` 的 markdown 链接）现已指向 `scripts/_archive/...`。
  - **待文档更新**（不在 A2 归档范围内，留给「文档更新」一遍统一处理）：上述路径需改为 `scripts/_archive/diagnostics/...` 或 `scripts/_archive/one_time_scripts/...`。
  - `test_sampler.py` / `bench_sampler_cache.py` 注释中对 `overfit_reconstruct.py` 的交叉引用为说明性文字，不影响运行。
  - `docs/交接文档_..._20260611.md` 与 `docs/_archive/模型架构修改规划_v1_20260616.md` 为历史交接/规划文档（后者已在 `_archive`），其路径提及保持历史原样。

## 因被保留 job 引用而**未归档**的候选

以下两个原候选脚本**保留在 `scripts/`**，因为被保留(规范)的 JCC job 实际**执行调用**，移动会断引用：

| 脚本 | 被哪个 job 调用 | 调用形式 |
|---|---|---|
| `replace_sample_previews_fullres.py` | `jobs/replace_training_sample_previews_fullres.yaml` | `$PY scripts/replace_sample_previews_fullres.py --out-root ...` |
| `bench_complete_v1_data.py` | `jobs/complete_v1_8h100_bench.yaml` | `torch.distributed.run ... scripts/bench_complete_v1_data.py --config ...`（3 次，256/512/1024） |

## 保留的活跃脚本(未动)

单测 `test_*.py`、数据管道脚本（`vae_cache_flux2_moments.py`、`dryrun_sampler_stats.py`、`vae_cache_qc.py`、`build_latent_cache_index.py`、`dryrun_report_html.py`）、`smoke_real_weights.py`、`smoke_loader_after_fixes.py`、`verify_latent_cache_gpu.py`、`bench_sampler_cache.py`，以及上表两个被 job 引用的脚本，全部保留在 `scripts/`。
