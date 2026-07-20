# 7.18 Stage256 formal-v2 bootstrap recovery authority

## Human decision and scope

After the original formal-v1 job trained healthily through step 490 but was
interrupted before its first step-3000 checkpoint, the user explicitly
authorized autonomous recovery and allowed non-essential process constraints
to be relaxed when doing so does not alter model, data, optimizer, or exposure
semantics.

This addendum authorizes one versioned formal-v2 chain.  It preserves every
formal-v1 artifact and uses the same frozen config, source tree, semantic W64
plan, causal two-layer text adapter, twelve-layer MLF, optimizer, scheduler,
world size 64, global batch 2304, native multi-AR buckets, and strict
`tag_only=0` contract.

## Reliability design

The code and config remain byte-identical to formal v1.  Formal v2 changes
only run/checkpoint/result/JCC identities and introduces a planned checkpoint
boundary:

1. `BOOT256` starts fresh from the same pretrained components in a new v2
   namespace and runs the first 256 optimizer steps on 8x8 GPUs.  It uses the
   existing `--max-steps 256` escape hatch and the normal final-checkpoint
   writer, so the checkpoint format is unchanged.
2. A CPU32 JCC performs a stable full-file hash/load/finite audit of the
   resulting checkpoint, all 483 model/EMA/optimizer states, scheduler step,
   exact metrics prefix, source/config/release identities, and the published
   resume YAML.  Only that JCC may create
   `formal_v2/BOOT256_CHECKPOINT_SUCCESS.json`.
3. The main v2 8x8 job may start only after the CPU receipt passes.  It resumes
   the same v2 checkpoint namespace at exact step 256, reconciles metrics with
   the existing bounded create-once replay guard, and continues to the
   original exact target step 166667 / exposure 384000768.

The planned process boundary can reset process-local RNG streams, as any
ordinary supported checkpoint resume does.  It does not reset model,
optimizer, EMA, scheduler, global step, semantic-plan position, or exposure
count.  This is accepted as a small reliability tradeoff and is not a change
to the sampling/data distribution contract.

## Fixed identities

```text
run/checkpoint id  stage256_20260718_causalmlf_offline_exact_v2
boot JCC base      evan-offarb-s256-20260718-causalmlf-boot256-8x8-v2
validator base     evan-offarb-s256-20260718-boot256-validate-v2
main JCC base      evan-offarb-s256-20260718-causalmlf-8x8-v2
checkpoint root    /fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v2
result root        /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v2
receipt            /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/formal_v2/BOOT256_CHECKPOINT_SUCCESS.json
```

Every identity and output is create-once.  Formal v1 remains read-only.  Only
one 64-GPU writer may be Pending/Running at a time.  All JCC jobs use
`maxRetries=0`; lower-level same-job re-entry remains fail-closed, while the
thread automation is authorized to preserve evidence and create a reviewed
versioned successor from the newest CPU-validated checkpoint.

## Permitted autonomous recovery

Infrastructure/node/EFA/NCCL/local-cache/JCC-wrapper failures may be recovered
without waking the user after evidence capture and checkpoint validation.  A
failure before the v2 step-256 checkpoint uses a new fresh versioned namespace.
A later failure uses a new versioned JCC/attempt identity and the latest stable
full checkpoint.  No checkpoint may be promoted from metrics alone.

Non-finite model/EMA/optimizer state, data or semantic-plan drift, nonzero
tag-only exposure, architecture/LR/global-batch changes, or irreducible
checkpoint identity ambiguity remain hard stops.  Old artifacts are never
deleted or overwritten.
