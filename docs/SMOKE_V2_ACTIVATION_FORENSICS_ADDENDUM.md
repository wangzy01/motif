# 7.18 Stage256 smoke-v2 activation-forensics addendum

This create-once addendum records the human decision made after the full v2
smoke failed its frozen postflight.  The user was told that training reached
step 3, but the checkpoint verifier observed only 14/22 changed
`text_adapter_moe_gen.blocks` tensors against a 20/22 threshold, and then
explicitly instructed on 2026-07-19: “继续”.

That decision authorizes a controlled continuation beginning with the
read-only forensics job below.  It does not convert either failed smoke into a
success, authorize mutation or deletion of v1/v2 evidence, or waive any model,
data, LR, topology, tag-only, GPU-cap, fresh-initialization, or maxRetries=0
contract in `docs/ORCHESTRATION.md` and
`docs/SMOKE_V2_HUMAN_DECISION_ADDENDUM.md`.

## Preserved v2 failure

```text
JCC            evan-offarb-s256-20260718-causalmlf-smoke-1x8-v2-nlxr
status         Failed
checkpoint     /fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v1_smoke_1x8_v2/last.pt
result         /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/smoke_1x8_v2
failure        adapter_blocks changed=14/22, required=20
```

Training, GPIC, ordered metrics `[1,2,3]`, terminal checkpoint creation,
runtime strict-NL observation, and planned/runtime tag-only zero had already
passed.  The deep verifier failed before writing a versioned checkpoint
receipt, so canonical `SMOKE_CHECKPOINT_SUCCESS.json` and
`SMOKE_SUCCESS.json` remain absent.  H1, T1 and formal G1 remain unsubmitted.

## Authorized read-only forensics

The only diagnostic attempt authorized by this addendum is:

```text
JCC YAML       jobs/offline_exact_stage256_20260718_smoke_v2_activation_forensics_v1.yaml
JCC name       evan-offarb-s256-20260718-smoke-v2-forensics-v1
topology       1 worker, 0 GPU, 32 CPU, maxRetries=0
output leaf    /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/smoke_v2_activation_forensics_v1
receipt        ACTIVATION_FORENSICS_EVIDENCE_SUCCESS.json
```

The job is intentionally a heavy-I/O CPU JCC: it hashes the 26.3-GB checkpoint
and scans the selected parameter, EMA, and optimizer tensors.  It may only read
the regular v2 checkpoint and frozen release sources.
It must first bind the exact v2 training evidence, metrics, log, YAML,
addendum, verifier and config identities.  It must not write below either v2
checkpoint/result leaf or any canonical smoke/final-health/formal leaf.

The receipt must report, without suppressing zero values, the two outer
zero-init `proj_out` tensors, every one of the 22
adapter-block tensor names, shapes and dtypes; trainable-versus-EMA deltas;
and optimizer `step`, `exp_avg`, and `exp_avg_sq` finite/nonzero evidence.  It
also reports the other adapter/MLF/text-KV activation groups and binds a stable
SHA-256 identity of the v2 checkpoint.  Evidence collection PASS is not smoke
PASS and cannot be promoted by itself.

## Decision after forensics

- If any adapter block lacks optimizer state, has non-finite state, or its
  first/second moment is wholly zero after the three completed steps, stop and
  report a real learning-path failure.  Do not submit another GPU smoke.
- The zero-init sequence is explicitly accounted for: the outer image
  `proj_out` opens first, then the adapter `out_proj`/MLF gate and text K/V,
  and only afterward the adapter blocks/out-norm and MLF logits.  Thus the
  blocks have only one very-small-warmup update opportunity in a three-step
  checkpoint.
- If optimizer moments prove that all intended adapter blocks received finite,
  nonzero gradients, while the 14/22 discrepancy is limited to parameter
  values that have not yet moved relative to EMA under the zero-init plus
  short-warmup schedule, treat the old 90-percent parameter-delta threshold as
  an invalid three-step proxy.  A separately versioned and reviewed v3 smoke
  may then retain the exact unchanged training invocation while replacing only
  that proxy with a moment-backed activation gate.  V3 still requires fresh
  checkpoint/result leaves, 1x8 GPUs, maxRetries=0, and all original semantic
  gates before it may create the canonical receipts.
- Any ambiguity, diagnostic failure, partial output, or identity mismatch
  preserves the scene and stops.  No automatic retry, cleanup, namespace
  switch, or downstream promotion is allowed.
