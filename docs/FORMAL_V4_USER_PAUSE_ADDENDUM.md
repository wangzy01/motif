# 7.18 Stage256 formal-v4 user-pause recovery authority

## Human decision and timing

At approximately 2026-07-19 11:10 CST the user requested that the active
formal-v3 8x8 allocation be stopped temporarily to release GPUs, then resumed
through a fresh successor submitted no earlier than 2026-07-19 16:10 CST.
This addendum authorizes that one delayed successor and its CPU32 preflight.
It does not authorize restarting an old JCC identity or running two 64-GPU
writers concurrently.

The stopped writer is exactly
`evan-offarb-s256-20260718-causalmlf-8x8-v3-qpvb`.  Its JCC state must remain
terminal `Stopped` immediately before the successor is submitted.  Formal-v1,
formal-v2, RECOVER384, main-v3, all attempts, logs, metrics backups, and every
historical failure remain read-only evidence.

## Preserved training semantics

No training, model, data, sampler, config, plan, optimizer, scheduler, LR,
world-size, batch, aspect-ratio, or exposure semantic is changed.  The model
remains the causal two-layer text adapter plus twelve-layer MLF.  The successor
uses the same offline-exact W64 plan, 20 native aspect-ratio buckets, world 64,
global batch 2304, LR 2.121e-4 with checkpoint optimizer/scheduler state, and
Stage256 tag-only exposure zero.  The terminal target remains step 166667 and
384000768 exposures.

## Pause evidence and safe recovery point

Main-v3 resumed the CPU32-validated step-384 checkpoint and produced exactly
contiguous, finite metrics through step 3000.  The last row is finite and the
runtime semantic/GPIC snapshot counters remain zero.  All eight train logs are
free of traceback, OOM, NCCL, non-finite, fatal, and marker-collision evidence;
all eight node-monitor streams show no memory pressure.

The stop occurred after the step-3000 metric was committed but before a new
full checkpoint was written.  `last.pt` is therefore still the validated
step-384 checkpoint with SHA-256
`6705da4257e2e306737457c5a8c255cdacf6b40b2494a59f6a4dab6392cc7dd6`.
Metrics 385--3000 are healthy but uncheckpointed and must never be promoted as
model state.  The successor must preserve the complete 1--3000 stream and the
discarded 385--3000 tail, atomically restore active metrics to the exact
1--384 prefix, then replay from checkpoint step 384 through the normal trainer.

The existing fail-closed `auto_resume_reconcile_v1` implementation provides
that operation.  Its checkpoint-identity digest for this exact step-384 file is
`f439d53f7350b3096049186ea45bd347c0c0b899e162c4513e88e48972322e21`.
The corresponding create-once claim must be absent before submission.  The
older step-256 claim is unrelated historical evidence and remains untouched.

## Versioned delayed-resume chain

1. `V4-PRE` is a new CPU32, zero-GPU, maxRetries=0 JCC.  It full-hashes and
   torch-loads `last.pt`, validates all 483 model/EMA/optimizer states and the
   step-384 scheduler, validates metrics 1--3000 and its exact 1--384 prefix,
   audits all eight main-v3 attempts/logs/GPIC receipts, checks the step-384
   replay claim is absent, and exclusively creates
   `formal_v4/PRE_RESUME384_SUCCESS.json`.
2. At or after 2026-07-19 16:10 CST, the controller rechecks that main-v3 is
   `Stopped`, that no 64-GPU writer is Pending/Running, that the fixed v4 JCC
   base/attempt does not exist, and that the V4-PRE receipt and published file
   hashes are unchanged.
3. `MAIN-V4` is then submitted once.  It is an 8x8, maxRetries=0 successor.  At
   startup it requires the V4-PRE receipt, consumes the one replay claim for
   the exact step-384 checkpoint, preserves and trims metrics, resumes at step
   384, and continues to the unchanged final horizon.

## Fixed identities

```text
preflight base       evan-offarb-s256-20260718-pause-preflight-v4
successor base       evan-offarb-s256-20260718-causalmlf-8x8-v4
checkpoint root      /fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v2
result root          /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v2
release receipt      /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/formal_v4/PRE_RESUME384_SUCCESS.json
attempt root         /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/formal_v4_resume_attempts
not-before           2026-07-19T16:10:00+08:00 (unix 1784448600)
```

Every new file, receipt, job identity, and attempt leaf is create-once.  Full
checkpoint hashing/loading is CPU-JCC-only.  Any checkpoint identity drift,
non-finite state, semantic drift, nonzero tag-only exposure, existing step-384
replay claim, partial metrics tail, or active competing writer is a hard stop.
