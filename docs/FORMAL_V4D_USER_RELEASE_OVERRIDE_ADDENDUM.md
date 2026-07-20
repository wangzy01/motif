# 7.18 Stage256 formal-v4d immediate-release authority

At 2026-07-19 14:36 CST the user explicitly lifted the temporary five-hour
pause and authorized immediate JCC submission and GPU use.  This is the newest
human timing authority and supersedes the 16:10/16:34 not-before gates in the
published v4 material.

The published v4 files, failed CPU v4b job, and all older evidence remain
read-only.  Main-v4 was never submitted.  To avoid mutating or bypassing its
embedded obsolete gate, this addendum authorizes one create-once successor:

```text
CPU preflight base   evan-offarb-s256-20260718-pause-preflight-v4c
GPU successor base   evan-offarb-s256-20260718-causalmlf-8x8-v4d
GPU runner           scripts/offline_arb/run_stage256_20260718_formal_v4d.sh
GPU YAML             jobs/offline_exact_stage256_20260718_FORMAL_RESUME_8x8_v4d.yaml
attempt root         formal_v4d_resume_attempts
release receipt      formal_v4/PRE_RESUME384_SUCCESS.json
effective not-before 2026-07-19T14:36:22+08:00 (unix 1784442982)
```

The v4c CPU32 gate must bind this exact addendum, runner, YAML and timing.  It
must full-hash/load the unchanged validated step-384 checkpoint, audit exact
metrics 1--3000 and all main-v3 evidence, and prove the step-384 replay claim
absent.  Only its exclusive PASS receipt authorizes main-v4d.

Main-v4d changes identity and the superseded wall-clock gate only.  It retains
the frozen config/source/plan/checkpoint and exact causal adapter2 + MLF12,
optimizer/scheduler/LR, world64/global2304, 20 native aspect ratios,
tag_only=0, and step166667 / 384000768-exposure target.  At startup it must
preserve metrics 1--3000 and discarded rows 385--3000, atomically restore the
active 1--384 prefix, consume the one replay claim for the exact checkpoint,
and resume from step384.  Only one 64-GPU writer may be Pending or Running.
