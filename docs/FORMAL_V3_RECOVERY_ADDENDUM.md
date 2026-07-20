# 7.18 Stage256 formal-v3 recovery authority

## Human authority and preserved scope

The user authorized unattended recovery of ordinary JCC, node, EFA, NCCL,
local-cache, and wrapper failures, while preserving model/data/optimizer/
exposure semantics and every failed artifact.  This addendum is the versioned
successor authority after formal-v2 main job
`evan-offarb-s256-20260718-causalmlf-8x8-v2-kbc7` became `Stopped`.

Formal-v1 and formal-v2 artifacts remain read-only.  No training source,
network source, config, semantic plan, optimizer, scheduler, LR, world size,
global batch, native AR set, or tag-only contract is changed.  The fixed model
remains the causal two-layer text adapter plus twelve-layer MLF.  The exact
target remains step 166667 / 384000768 exposures with world 64, global batch
2304, and Stage256 tag-only exposure zero.

## Formal-v2 failure evidence

Formal-v2 loaded the CPU32-validated step-256 checkpoint and reconciled an
exact 256-row metrics prefix as a no-op.  All eight fresh markers, compact GPIC
receipts, and rendezvous passed.  Rank zero logged the exact resume at step
256 and produced contiguous finite metrics through step 430.  The last row was
finite (loss 1.1944658756256104, grad norm 0.7890557050704956, LR
9.14151e-05).  All eight node monitors ended synchronously without traceback,
OOM, NCCL, non-finite, data, tag-only, or GPIC errors.  JCC retained only
`Stopped` and cannot recover a more specific termination reason.

The full checkpoint was not rewritten: it remains the original validated
step-256 file with SHA-256
`61a9cccf75a5d9e6aef72d8404964bf705c7c727773459140fec7486dfb15e7a`.
Metrics rows 257--430 are therefore uncheckpointed and may not be promoted.
Their complete stream must be preserved by the existing bounded atomic
auto-resume reconciler before replay.

## Versioned recovery chain

1. `V3-PRE` is a CPU32/maxRetries=0 JCC.  It full-hashes and torch-loads the
   step-256 checkpoint, validates all 483 model/EMA/optimizer states and the
   scheduler, audits metrics 1--430 and the exact 1--256 prefix, validates all
   formal-v2 attempt evidence, and creates the exclusive receipt
   `formal_v3/PRE_RECOVER256_SUCCESS.json`.
2. `RECOVER384` is a new 8x8/maxRetries=0 JCC identity.  It requires that
   receipt, uses the frozen config and code, invokes the existing bounded
   reconciler once to preserve and atomically trim metrics 257--430, resumes
   the exact step-256 checkpoint, trains only through step 384, and uses the
   normal full-checkpoint writer at its planned terminal boundary.
3. `V3-POST` is a new CPU32/maxRetries=0 JCC.  It full-hashes/loads the new
   step-384 checkpoint, checks all 483 model/EMA/optimizer states, scheduler,
   exact metrics 1--384, the preserved discarded tail and reconciler guard,
   and creates the exclusive receipt
   `formal_v3/RECOVER384_CHECKPOINT_SUCCESS.json`.
4. `MAIN-V3` is the only subsequent 8x8 writer.  It requires the post receipt,
   resumes exact step 384, and continues to the unchanged final horizon.

The extra process boundary has the same accepted process-local RNG reset
semantics as any supported checkpoint resume.  Offline semantic-plan position,
model, optimizer, EMA, scheduler, global step, and exposure accounting remain
continuous.  The short 128-step roll-forward is chosen to create a new full
checkpoint identity before another long allocation, reducing infrastructure
loss without permanent high-frequency FSx checkpoint writes.

## Fixed versioned identities

```text
pre validator base   evan-offarb-s256-20260718-recover-preflight-v3
recover base         evan-offarb-s256-20260718-causalmlf-recover384-8x8-v3
post validator base  evan-offarb-s256-20260718-recover384-validate-v3
main base            evan-offarb-s256-20260718-causalmlf-8x8-v3
checkpoint root      /fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v2
result root          /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v2
release receipts     /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1/formal_v3
```

Each JCC identity, attempt directory, and receipt is create-once.  Only one
64-GPU writer may be Pending or Running.  A failed identity is never restarted,
deleted, overwritten, or reused.  Full checkpoint hashing/loading remains CPU
JCC-only.  Semantic drift, non-finite state, nonzero tag-only exposure, or an
irreducible checkpoint identity mismatch remains a hard stop.
