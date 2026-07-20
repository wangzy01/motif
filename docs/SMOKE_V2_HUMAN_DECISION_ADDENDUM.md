# 7.18 Stage256 full smoke v2 human-decision addendum

This addendum records the human-authorized successor to the spent v1 smoke.
It is a narrow exception to `docs/ORCHESTRATION.md` (SHA-256
`f7949b6fc8c347753e7fcf4f8fae7e441add42a1799031d6756216c1bdcd992f`).
All frozen model, data, GPIC, architecture, LR, exact-step, tag-only,
failure-preservation, GPU-cap and no-retry rules remain unchanged.

## Preserved v1 failure evidence

The immutable v1 JCC
`evan-offarb-s256-20260718-causalmlf-smoke-1x8-v1-hznz` is terminal
`Failed`.  Its training subprocess reached three steps and returned zero, but
the wrapper then required a progress-print line matching
`^[train] step 3/3 `.  The short run printed a progress row only at step 1,
although its durable metrics contained exact steps 1, 2 and 3.  It therefore
failed before launching the checkpoint verifier.

```text
v1 pod             evan-offarb-s256-20260718-causalmlf-smoke-1x8-v1-hznz-0-9jzz5
v1 result leaf     /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/smoke_1x8
v1 checkpoint      /fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v1_smoke_1x8
v1 YAML SHA-256    0630cfea8fd9a7f55b3c4e477520b81ec04952a01d36f6a90b7eb3b75f5ab8fa
train log SHA-256  c1c6dc844314ea29bf5c3d2758af2d454fa7d484f70fc787e002cba9fc83ba21
metrics SHA-256    74b147fe3e0dfb595f1a52edb42f4918d86802f06e3ce205d08c7dfcbd6661ec
GPIC SHA-256       7bd3d283fcb7f2458eb08f13c9e71e81f593a6bc1261b95db20cb84c76f62719
v1 checkpoint size 26,328,353,144 bytes
```

The failed job, its result/checkpoint leaves and logs remain read-only
forensic evidence.  The v2 job must not modify, rename, delete, reuse, copy,
hard-link or symlink any v1 artifact.  In particular, it must not promote the
orphan v1 checkpoint into a success receipt.

After this v1 failure was reported, the user explicitly instructed on
2026-07-19: “目前什么情况，继续执行任务”.  This is the human decision
authorizing the full, fresh v2 successor below; it does not authorize recovery
or promotion of any artifact left by the failed v1 job.

## Authorized full v2

The only authorized successor identity is:

```text
JCC name      evan-offarb-s256-20260718-causalmlf-smoke-1x8-v2
YAML          jobs/offline_exact_stage256_20260718_smoke_1x8_v2.yaml
checkpoint    /fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v1_smoke_1x8_v2
result        /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/smoke_1x8_v2
topology      1 worker x 8 GPUs
steps         exactly 3
maxRetries    0
```

Both v2 leaves must be absent at submission.  The still-absent canonical
receipts remain:

```text
/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/SMOKE_CHECKPOINT_SUCCESS.json
/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/SMOKE_SUCCESS.json
```

V2 may create them only at the final commit point, after every validation
below passes.  Earlier phases write versioned evidence only below the new v2
result leaf.

## Required natural-success proof

1. Re-run the unchanged three-step distributed training from pretrained
   components with `trainer.resume=false`, the W8 semantic plan and the
   immutable offline GPIC release; no architecture, LR or init override is
   permitted.
2. Require both training/tee pipeline return codes to be zero, a regular
   `last.pt`, and a regular `metrics.jsonl`.
3. Parse metrics as exact ordered steps `[1,2,3]`; require finite loss/gradient/
   LR/timing values.  This durable sequence replaces the brittle step-3
   progress-print assertion.
4. Require the terminal log to bind the v2 checkpoint and metrics paths through
   `checkpoint saved at step 3` and `done at step 3`; do not require a
   `step 3/3` progress row.
5. Bind strict-NL/tag-only evidence from the W8 audit, overlay and runtime
   counters.  `buf_semantic_anime_tag_only_records` and
   `buf_semantic_tag_fallback_records` must both be observed and zero.
6. Run the unchanged GPU checkpoint verifier (SHA-256
   `6716f55cb53a07a28b8e4d307cdae2d0682d3116052b5dc991606fc2e136e6c0`)
   with the unchanged eval helper (SHA-256
   `8d106c39914083a87b0a372b56c2bf41477ee0279ea0f51e5daa844a38cd736a`)
   and `--expected-step 3`.  It must pass architecture, ablation, parameter,
   EMA and adapter/MLF activation gates.
7. In the same JCC, and under a 7,200-second TERM/kill timeout, reload the v2
   checkpoint on CPU and check all 483 trainable/EMA tensors for exact keyset,
   identical shapes and finite values.  Separately require nonzero
   trainable/EMA deltas for exactly one adapter `out_norm` tensor and exactly
   28 `text_k_norm_moe_gen` tensors.
8. Bind the exact v2 YAML SHA, this addendum SHA, config SHA, GPIC receipt and
   its actual v2 `jobtag` derived from the runtime hostname, metrics/log
   evidence, stable checkpoint identity and deep verifier receipt into
   `SMOKE_SUCCESS.json`.

## Submission discipline

Before submission, recheck the exact v1 job is `Failed`, the exact v2 name is
absent, both new v2 leaves and both canonical receipts are absent, the v1
evidence is unchanged (including a regular, non-symlink v1 `last.pt` of the
recorded size), and no Pending/Running evaluation would make the new 8-GPU
smoke exceed the user's eight-GPU test cap.  Deploy the v2 YAML and this
addendum create-only/read-only with matching local/remote hashes, then submit
the exact YAML without overrides.

Any v2 failure or partial output is preserved and stops the chain.  Do not
retry, clean, switch namespace or start final-health.  H1 may continue only
after the v2 JCC naturally succeeds and both canonical receipts independently
validate.
