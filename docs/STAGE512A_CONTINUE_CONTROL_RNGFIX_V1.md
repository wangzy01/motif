# Stage512a Continue Control RNG-fix v1 — formal runbook

Status: **PREPARATION ONLY / LAUNCH BLOCKED** until every gate below passes.  The
target job is `stage512a_continue_control_rngfix_v1`, 8 nodes × 8 H100, and must
not be submitted before the Stage512B-REAL step-120,000 checkpoint is stably saved
and the B job is stopped as a complete eight-node job.

## What this run can claim

This is the one affordable practical counterfactual: start from terminal
Stage512a and continue for another 256M exposures with Stage512a data, weighted
mix, caption semantics and adaptive geometry.  It answers “what if we had kept
training the A recipe?” better than any other single arm.

The fixed result-table and manuscript label is **“A-recipe continuation
(RNG-corrected; new plan realization)”**.  It must not be shortened to a strict
control, and the historical bug must not be described as an effective batch size
of 32: it correlated posterior epsilon across ranks, not examples or gradients.

It is **not** a strict single-variable or pure A/B control.  The new arm fixes a
real stochastic correctness defect shared by historical A and B: VAE posterior
epsilon was slot-aligned across ranks instead of rank-specific.  Historical A
also demonstrated RNG replay during its step-6000 recovery: step-6001
`sigma_mean` was bit-identical to step 1 because the old checkpoint did not save
RNG state.  Rank-specific VAE sampling plus exact RNG checkpoint/resume is
therefore necessary, but it is also an explicit scientific variable relative to
the historical runs.  Results must be described as a **practical
counterfactual/control**, never as a clean single-variable ablation.

The repair is material rather than cosmetic.  The real FLUX.2 dose probe reports
posterior-std RMS 0.738–0.824 and model-space sample-minus-mode/mode RMS 0.52–0.63
(`vae_rank_rng_probe.json`, receipt SHA-256
`56326610f0305d87b0b9181fcc5479f4ac60a01bba1af613f4fedc4ffeef5efa`).

The fresh plan is another explicit stochastic delta: its seed changes the
`(step,rank,slot)->row` permutation, and `OfflineExactDataset._slot_seed` hashes
`plan_id`, so prompt/caption-slot random choices change as well.  Exact RNG resume
also changes CFG/dropout/diffusion trajectories after an interruption relative to
the historical replay bug.  These changes are required to avoid replaying the
fully consumed plan and to make recovery correct; they must not be hidden in the
paper's causal wording.

## Frozen scientific contract

The executable preflight compares the new config against
`configs/stage512a_20260722_weighted_w64_v1.yaml` with a closed allowlist.  Every
non-allowlisted leaf must be canonically equal.

Preserved:

- Stage512a catalog and sampling policy; domain mix Anime/MONET/GPIC/Synthetic/
  TextAtlas = 37/40/13/5/5.
- Stage512a caption sidecar and anime-frequency/priority assets; no Stage512b
  TextAtlas-GT sidecar.
- adaptive geometry, floor 0.005, target area 262144, no upscale, and
  `square_only_domains=[synthetic,textatlas]`.
- model architecture, Qwen/FLUX VAE identities, cfg dropout, optimizer LR/betas/
  eps/weight decay, shift 1, constant schedule, 500-step stage-local warmup,
  gradient clipping, BF16, EMA 0.9999, seed 42.
- world 64 × local batch 32 = global batch 2048; 125,000 stage-local steps =
  256,000,000 exposures.

Allowed changes only:

1. fresh-seed plan path/hash (the consumed 2026-07-22 row stream may not replay);
2. A-catalog GPIC blob view, 64 threads/node, and 680-FD cache — identical bytes,
   rows and order; transport/concurrency only;
3. create-once checkpoint directory, A-final `init_from`, checkpoint cadence and
   permanent evaluation snapshots;
4. `init_from_rng_policy=legacy_missing_rng` plus new exact RNG state;
5. measurement-only cross-rank GC/data-wait probe.

Optimizer semantics are intentional: `init_from` loads Adam moments and its
per-parameter step (observed 291,667) from A-final, while stage `global_step` and
scheduler reset to 0.  Do not reset Adam, and do not load A-final's scheduler.

## Bound identities

| Object | Identity |
|---|---|
| A catalog | `370b1679e8c5670338a03ba4c6972cdbf69fa5068fc35108bf646b2d64642331` |
| catalog header | `51a5f0904987a751b54842633d2968ea7f6485307850858d9472319c15748d1d` |
| A sampling JSON | `07a0b46fa9ebb840ecf7f97cc7fccc0ec68e44c47368ae13c450bcfe81a4fd48` |
| weighted contract | `c228f8b8400c4f96033431816b650d6e7093b3bf8e164018d22e694c60b96c38` |
| anime priority header/parquet | `93e23f11…` / `0a86f719…` |
| animefreq header | `4c2cc28f7b161b9f2a037d66c8f54405521f542ea81dfa1b0198e697cf7c15d4` |
| caption sidecar header | `72fe0c4d5b5258a1a6320a6ab57f422cd815dd26a8a4a29594fefdd7abfa7224` |
| A GPIC manifest | `c9365753356f942caca1ea622c788ae5d05db320ecfa5804595b25ac8af3a648` |
| A GPIC dense index | `37f1a2bca7a7236c13f0d757547175a654647ea2f81de526a553f113bac7bded` |
| A-final full state | `957bfcf12b89c4042a32056566012a2451e72aa0b2706b0ca51808ab79155c31` |

The new plan is create-once at
`offline_arb_plan_stage512a_continue_control_rngfix_v1_w64_256m_seed20260710_20260809`
with plan id
`stage512a_offline_exact_v2_continue_control_rngfix_v1_w64_256m_seed20260710`.
Its plan hash/header SHA remain launch blockers until the double-build/full-scan
receipt exists.  The plan must also bind a new exception closure: three reviewed
TextAtlas JPEG repairs and the actual fresh-plan coordinates of anime row
11,163,647.  Never copy the old five coordinates without scanning the new plan.
The tracked contract authority used by preflight is
`provenance/imported_control_scripts/weighted_contract_stage512a_v1.json`; do not
mint a second semantic contract merely because the new plan has a new name.

## Fresh run identity and resources

- code: `/fsx/evan/mot_code_s512a_continue_control_rngfix_v1`
- config: `configs/stage512a_continue_control_rngfix_v1.yaml`
- formal spec:
  `provenance/jcc_jobs/20260809_stage512a_continue_control_rngfix_FORMAL_w64.yaml`
- stable `RUN_ID`: `stage512a_continue_control_rngfix_v1`
- fresh `CKPT_DIR`:
  `/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage512a_continue_control_rngfix_v1`
- JCC: `numWorkers=8`, `gpuPerWorker=8`, `cpuPerWorker=80`,
  `memGBPerWorker=900`, `maxRetries=3`.

The first submission requires `RUN_MODE=fresh` and the checkpoint directory must
not exist.  Rank 0 claims it with one bare `mkdir`; losing that atomic race is a
hard failure.  `.jobtag` contains the stable RUN_ID.  A different owner is rejected
in both fresh and continue modes.  JCC retries retain RUN_ID; an owned directory
with `last.pt` promotes effective mode to continue, while an owned directory with
no checkpoint remains fresh only when declared fresh.

Rendezvous, arrival directory and reconciler nonce are attempt/generation scoped.
Rank 0 removes the old same-JOBTAG rendezvous before hashing A-final, preventing a
pod restart from adopting a dead master address.  Training does not begin until
all 8 node arrival markers exist; torchrun then launches 8 processes/node and the
plan/runtime enforce world size 64 and local batch 32.

## Launch gates, in order

1. B step-120,000 full-state checkpoint exists, size is stable across observations,
   its evaluation snapshot is preserved, and the entire B job has stopped.  No A
   submission while any B worker remains alive.
2. A-final is exposed through a create-once frozen input path (not the mutable old
   run's `last.pt` name) at
   `/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage512a_continue_control_rngfix_v1_inputs/a_final_step125000.last.pt`,
   with ordinary-file identity, size and full SHA above.
3. New plan double-build equality, all-block checksum/full-row scan, exposure
   report, source-producer binding and plan-specific exception closure are PASS.
4. RNG unit tests and <=4-GPU tests prove: different ranks receive different VAE
   epsilon; same checkpoint resume restores exact next draws; world-size/schema/
   missing-RNG resumes fail closed; A-final's one legacy init waiver is accepted.
5. `scripts/stage512a_control_formal_preflight.py` prints PASS against the final
   non-placeholder config.  All repository tests pass; the tree is committed,
   clean and `EXPECTED_COMMIT` is the full 40-character HEAD.
6. Read-only check proves the exact CKPT_DIR does not exist.  Submit once with the
   exact CKPT_DIR/RUN_ID and `RUN_MODE=fresh`.
7. Require `all 8 nodes arrived`, 64 torch ranks, and immediate stable marker
   `[formal-boot] verified mode=fresh ... step=0`.  Fresh must show exact A-final
   `init_from (load_optimizer=True)`, the legacy-to-new RNG lineage, and no
   `resumed from` line.
8. Require new training steps and sensible LR warmup.  At step 1,500 require a
   stable full `last.pt` with `checkpoint_schema=mot_full_training_v2` and
   `rng_state.schema=mot_rng_state_v1`, world size 64 and per-rank entries.

For a retry/explicit continuation, use the same CKPT_DIR and RUN_ID with
`RUN_MODE=continue`.  It must show the opposite boot proof: local `resumed from ...
at step N>0`, exact RNG restore for rank 0/world64, reconciler receipt, and no
A-final init branch.

## Checkpoints and evaluation

`last.pt` is full-state every 1,500 steps.  `step_30000.pt`, `step_60000.pt`,
`step_90000.pt` and `step_120000.pt` are permanent read-only hard links to the
corresponding full `mot_full_training_v2` payloads, retaining raw weights, EMA,
AdamW, scheduler, stage-local step and all-rank RNG; each is independently
resumable.  `keep_every=30000`, `keep_max=4`, and 30,000 is divisible by 1,500.
Their GenEval
checkpoint identities must be minted from these new exact paths and content
hashes.  Historical A/B GenEval identities must never be reused.  CFG4 and CFG8
evaluation is required downstream but does not block first launch.

## Failure and rollback

- Wrong boot mode, wrong init, owner mismatch, partial 8/8 arrival, plan/asset
  drift, missing RNG state or RNG world mismatch: stop/fail the whole JCC job;
  never let the surviving nodes continue.
- Before the first full checkpoint, preserve the directory/logs.  A JCC retry may
  remain fresh only under the same owner and with no nonempty metrics stream.  Any
  ambiguous directory is quarantined and a new release identity is required.
- After a full checkpoint, resubmit only as continue with the same RUN_ID.  The
  metrics reconciler preserves and trims an ahead-of-checkpoint tail once.  If the
  same checkpoint's replay guard is already consumed, automatic retry must stop for
  human audit; do not delete the guard to force a second replay.
- If semantic corruption is suspected, preserve the entire CKPT_DIR as evidence,
  stop all workers, and do not resume it.  Roll back operationally to the untouched
  frozen A-final input and a new create-once run/release after the cause is fixed.
- Never point `checkpoint_dir` at A-final, the abandoned old A directory, B's
  directory, or any weights+EMA-only snapshot.
