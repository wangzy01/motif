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
- `naifu_arb_2x8_smoke_*.yaml`

Before submitting any other job from this copied directory, check for hard-coded
paths to `/fsx/evan/mot_code/mot_pilot_v1_cleanup` or original checkpoint/result
directories.  Those jobs are not part of the isolated crop experiment by default.
