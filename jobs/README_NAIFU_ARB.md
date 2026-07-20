# cleanup_crop job safety notes

This directory was copied from `mot_pilot_v1_cleanup`, so many legacy job YAMLs
still intentionally reference the original tree and original outputs.

For the isolated stage256 NAI-like ARB work, submit only jobs whose names start
with `naifu_arb_` unless you have manually audited the paths in the job body.

Known safe ARB job family:
- `naifu_arb_config_audit.yaml`
- `naifu_arb_dataline_audit.yaml`
- `naifu_arb_dataline_debug.yaml`
- `naifu_arb_pil_lock_audit.yaml`
- `naifu_arb_bucketdist_*.yaml`
- `naifu_arb_2x8_smoke_*.yaml`（r2：timeout 3600s + item_queue waitfix 后版本，见 `naifu_arb_2x8_smoke_clean_bs144.yaml` r1/r2 对照，2026-07-09）
- `naifu_arb_bucketdist_fast_remeasure.yaml`
- `naifu_arb_real_dataline_stress_cpu_ab_dump{0,5}.yaml`
- `naifu_arb_fixes_unittest.yaml`
- `naifu_arb_revalidate_waitfix.yaml`（2026-07-09 新增：CPU 复验 item_queue waitfix，fullmix16/monet16/anime16 三 case；判定 = 零 SIGSEGV/非零退出 + `item_queue_wait_s` 中位数<5s）

Before submitting any other job from this copied directory, check for hard-coded
paths to `/fsx/evan/mot_code/mot_pilot_v1_cleanup` or original checkpoint/result
directories.  Those jobs are not part of the isolated crop experiment by default.

⚠️ **FORBIDDEN**：periodic faulthandler dump probes（`trainer.debug_faulthandler_dump_interval`
/ `debug_trace_dump_interval` > 0，或 stress harness 的 `--dump-every-s` > 0）——已在 Py3.12 上
定案证实为 SIGSEGV 根因（watchdog 线程无锁遍历运行中线程帧栈竞态，CPython gh-140815，2026-07-09
定案）。任何 smoke / stress / 正式训练 yaml 一律不设，hang 诊断改用进程外 `py-spy`。

offline_arb 线的 job 见 `scripts/offline_arb/README.md` 与 `docs/OFFLINE_ARB_V1_SPEC.md`（streaming 线已归档弃用）。
