# 7.18 Stage256 full smoke v3 human-decision addendum

This create-once addendum authorizes one fresh v3 successor after the spent v1
and v2 smoke attempts.  It is a narrow exception to
`docs/ORCHESTRATION.md` (SHA-256
`f7949b6fc8c347753e7fcf4f8fae7e441add42a1799031d6756216c1bdcd992f`),
and incorporates the preserved-history contracts in:

- `docs/SMOKE_V2_HUMAN_DECISION_ADDENDUM.md`, SHA-256
  `6000007d0438caea8af1714a5ad07aadf732fe6a9c5ad6eebd4d7bbacab6bb18`;
- `docs/SMOKE_V2_ACTIVATION_FORENSICS_ADDENDUM.md`, SHA-256
  `e0a1091c8155e80b727f46bf5570db55053527f927faf5c8968c38c0714f4d72`.

All model, data, LR, exact-step, GPIC, tag-only-zero, fresh-initialization,
GPU-cap, evidence-preservation and `maxRetries=0` rules remain unchanged.
Nothing in this addendum authorizes promotion, resume, mutation, copying,
hard-linking, symlinking, cleanup or deletion of either failed attempt.

## Human decision and preserved failures

After the user was told that v2 completed three training steps but failed the
frozen `adapter_blocks changed=14/22, required=20` postflight, the user
explicitly instructed on 2026-07-19: “继续”.  That decision authorized the
read-only CPU forensics and, only after its evidence closed without ambiguity,
one complete fresh v3 smoke.  It does not authorize a v4 or an automatic retry
if v3 fails.

The immutable failed attempts remain:

```text
v1 JCC          evan-offarb-s256-20260718-causalmlf-smoke-1x8-v1-hznz
v1 status       Failed
v1 checkpoint   /fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v1_smoke_1x8
v1 result       /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/smoke_1x8

v2 JCC          evan-offarb-s256-20260718-causalmlf-smoke-1x8-v2-nlxr
v2 status       Failed
v2 checkpoint   /fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v1_smoke_1x8_v2
v2 result       /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/smoke_1x8_v2
```

The v1 evidence remains bound by the v2 addendum: YAML
`0630cfea8fd9a7f55b3c4e477520b81ec04952a01d36f6a90b7eb3b75f5ab8fa`,
train log
`c1c6dc844314ea29bf5c3d2758af2d454fa7d484f70fc787e002cba9fc83ba21`,
metrics
`74b147fe3e0dfb595f1a52edb42f4918d86802f06e3ce205d08c7dfcbd6661ec`,
GPIC
`7bd3d283fcb7f2458eb08f13c9e71e81f593a6bc1261b95db20cb84c76f62719`,
and regular checkpoint size `26,328,353,144` bytes.

The v2 evidence is independently bound as follows:

```text
v2 YAML                   56f18a4e375c4af3ec3a9fdb022f14e06b93ec5956164ed8301201e8d718b9b0
v2 TRAIN_EVIDENCE         b228786d1dfc9e7ba3469e58a8cc49467e9fe34a20c147e5b2c3c940067b53de
v2 train log              c2c701d4be9088c1e074cf0d09806f5a5bf2f2e74fe4618f800589a1bd82eabe
v2 metrics                acd9c39a845cec400b688f9eb6e605021cf6b56b4d67454ceb11957c951e00ac
v2 GPIC                    715dbabca764d434635383f29bd23c7381ecb3e47e168338f95d57bc1639ae18
v2 verifier failure log   e2293dfe0a2bbdd32286ad889edc59e10050569f217f1b655049957f36f31dcd
v2 checkpoint             010f754fabc59734e459b6224af692675c9562798d7d69f45097301a24a23480
v2 checkpoint bytes       26,328,353,144
```

Canonical `SMOKE_CHECKPOINT_SUCCESS.json`, `SMOKE_SUCCESS.json`, H1, T1 and
formal G1 all remain absent.  Neither v1 nor v2 may be promoted into them.

## Completed activation forensics

The authorized CPU32, zero-GPU, `maxRetries=0` job
`evan-offarb-s256-20260718-smoke-v2-forensics-v1-cl9w` naturally
`Succeeded`.  Its create-once receipt is:

```text
/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/smoke_v2_activation_forensics_v1/ACTIVATION_FORENSICS_EVIDENCE_SUCCESS.json
SHA-256 391d9fa31a99a9176367c65226f3966ba031c12b8e9262b6c9e8d1603707a994
```

The receipt is evidence only, not smoke success.  It proves all of the
following simultaneously:

- structural identity and the approved ablation fingerprint passed;
- all 14 dense adapter-block parameters changed relative to EMA;
- the only eight unchanged adapter-block parameters are exactly
  `ln1`, `ln2`, `q_norm` and `k_norm` in each of the two blocks;
- all 22/22 adapter-block parameters have valid step-3 optimizer state and
  finite, nonzero first and second moments;
- the adapter out-norm has nonzero optimizer moments even though its parameter
  has not yet moved relative to EMA;
- all 28 text-K norms have nonzero optimizer moments, including the seven
  whose parameter values have not yet moved relative to EMA;
- adapter out-proj, both final proj-out tensors, MLF gate, MLF logits and all
  56 text K/V projections changed, and every reported group has nonzero
  optimizer moments;
- all zero-init openers changed and the exact norm-only gap explains why the
  old three-step 90-percent parameter-delta proxy was a false negative.

Because optimizer weight decay is zero, those nonzero Adam moments are direct
evidence of nonzero training gradients.  V3 must report every unchanged value;
it may replace the invalid aggregate 20/22 proxy only with the stricter
structural, moment-backed and functional gates below.  It must not merely lower
the old threshold to 14.

## Only authorized v3 identity

```text
JCC YAML       jobs/offline_exact_stage256_20260718_smoke_1x8_v3.yaml
JCC name       evan-offarb-s256-20260718-causalmlf-smoke-1x8-v3
checkpoint     /fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v1_smoke_1x8_v3
result         /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/smoke_1x8_v3
topology       1 worker x 8 GPUs
steps          exactly 3 / 864 image exposures
maxRetries     0
```

Both v3 leaves and both canonical smoke receipts must be absent at submission.
After normalizing only attempt-identity values (`RUN_ID`, checkpoint/result
leaf, jobtag and postflight names), the training invocation must be identical
to v2: fresh pretrained initialization, `trainer.resume=false`, the same
config, W8 semantic plan and immutable GPIC release, with no init,
architecture, LR, data or step override.  No other normalization is allowed.
`V3_SEMANTIC_EQUIVALENCE.json` must retain both raw invocation hashes, require
the identity-normalized hashes to be equal, and independently bind the v2/v3
YAML hashes, config/training-source hashes, resource contract and forensics
receipt before canonical commit.

## Required v3 activation and checkpoint proof

The separately versioned v3 verifier must retain every valid v2 deep gate:
architecture, ablation, 483-trainable/310-frozen key identity, exact parameter
counts, step 3, stable checkpoint identity, EMA loading, all-state shape and
finite checks, and nonzero updates in zero-init opener groups.

It additionally must:

1. require exactly two adapter blocks and the exact 22-key structural set;
2. require all 14 dense adapter-block tensors to change, separately in each
   block, while reporting the eight norm deltas without suppressing zeros;
3. require valid step-3 optimizer state and finite nonzero `exp_avg` and
   `exp_avg_sq` for all 22/22 block tensors, the one adapter out-norm and all
   28 text-K norms;
4. require changed zero-init adapter out-proj, MLF gate/logits, both final
   proj-out tensors and all 56 text K/V projection tensors;
5. require all 483 trainable/EMA keys, shapes and values to match and be finite;
6. run a deterministic forward probe that records both adapter blocks and
   proves the causal prefix does not depend on a changed suffix.  The hard
   nonzero-effect gate must inspect the trainable-state FP32 branch tensors
   before their residual additions, and the changed suffix must produce a
   finite nonzero change in the trainable-state suffix output so the causal
   prefix check cannot pass vacuously.  EMA execution, shapes and values must
   be finite and are fully reported, but a step-3 EMA post-residual difference
   is not required to be nonzero because decay `0.9999` plus nested zero-init
   can make the real branch smaller than the residual-addition rounding unit;
7. bind the v2-forensics receipt and checkpoint identity into its versioned
   output, and write failure diagnostics before returning nonzero on a gate
   failure.

Any missing/zero optimizer moment, missing block/family, non-finite value,
zero trainable FP32 pre-residual branch effect, causal-prefix violation,
identity drift or unexpected unchanged dense tensor is a genuine failure.  It
must not be waived.

## Evidence and canonical commit

Earlier v3 phases may write only below the fresh v3 result leaf.  The required
versioned evidence is:

```text
TRAIN_EVIDENCE_V3.json
V3_SEMANTIC_EQUIVALENCE.json
ACTIVATION_DIAGNOSTIC_V3.json
FULL_STATE_AUDIT_V3.json
FORWARD_PROBE_V3.json
SMOKE_CHECKPOINT_SUCCESS_V3.json
SMOKE_SUCCESS_V3.json
```

The canonical receipts remain:

```text
/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/SMOKE_CHECKPOINT_SUCCESS.json
/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/SMOKE_SUCCESS.json
```

They may be created only at the final commit point after training/tee RC zero,
ordered finite metrics `[1,2,3]`, terminal step-3 save/done evidence,
planned/runtime tag-only and fallback zero, GPIC PASS, semantic equivalence,
the full v3 verifier and all versioned evidence pass.  The canonical checkpoint
receipt must be byte-identical to its v3 versioned receipt; canonical success
must be byte-identical to versioned v3 success.  Each is create-once, regular,
non-symlink, mode `0444`, atomically published and directory-fsynced.

The success receipt must bind the exact actual v3 jobtag, v3 YAML, this
addendum, v3 verifier, config, eval helper, forensics, checkpoint, GPIC,
metrics/log and every versioned-evidence SHA.  A partial canonical commit is a
failure and is preserved.

## Submission and failure discipline

Before submission, recheck both exact failed jobs and all bound evidence are
unchanged; the forensics JCC is uniquely `Succeeded`; the v3 JCC prefix, v3
leaves, canonical receipts, H1/T1/formal jobs and leaves are absent; all pinned
upstream receipts still match; and adding v3 keeps total concurrent evaluation
GPU usage at or below eight.  Deploy the addendum, verifier and YAML
create-only/read-only with matching local/remote hashes, then submit the exact
YAML without overrides.

Any v3 JCC failure, stopped replica, partial output, ambiguous identity or
receipt mismatch preserves all evidence and stops the chain.  Do not rerun its
verifier, retry, clean, submit a v4, switch namespace, start H1, modify the
frozen formal recipe or promote any earlier checkpoint without another
explicit human decision.
