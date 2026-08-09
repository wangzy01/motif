# RNG/resume two-GPU hard gate v1

Status: prepared only.  Do not treat this document, an unpinned run, or a
pre-freeze receipt as launch evidence.

## Scope

The gate is a single-node, two-GPU JCC integration job.  It never invokes
`train.py` and is not a pilot or formal training run.  Its four GPU phases are
separate `torchrun` invocations:

1. `flux2-vae`: loads `/fsx/models/FLUX.2-klein-4B/vae`, exercises both online
   pixel encode and cached-moments sampling on both CUDA ranks, and checks exact
   generator-device binding, rank-specific posterior epsilon/latent,
   consumption, and local replay.  It records the old
   shared-seed correlation, the fixed correlation, posterior sigma scale, raw
   latent scale, model-space mode scale, and posterior-delta RMS as explanatory
   diagnostics.  For both pixels and cached moments it also reports the actual
   posterior contribution (`sigma * epsilon`) RMS relative to the sampled and
   mean/mode latent RMS; these dose ratios are diagnostic rather than tuned
   thresholds.  Only device binding, replay, uniqueness, and the explicit
   rank-derived seed contract are launch gates.
   It also asserts that `fixed_sigma_eval.py`'s explicit CUDA generator produces
   exactly the historical default-generator draw when both use the same seed.
2. `save-reference`: runs one warm-up update through the real FLUX.2 VAE and a
   tiny FSDP2 core, writes the production v2 checkpoint with DCP full model and
   optimizer state, then records the uninterrupted next update.  It also proves
   rank-0-only serialization through an unpredictable `O_EXCL|O_NOFOLLOW`
   temporary, file/directory fsync ordering, atomic replace, read-only hard-link
   publication, and resistance to a pre-created predictable `.tmp` symlink.
3. `resume-compare`: starts a new process group, consumes startup RNG on every
   stream, loads the v2 checkpoint, commits the late RNG restore, and requires
   exact equality for batch, latent, CFG, sigma, diffusion noise, loss, full
   gradient, raw state, EMA, optimizer, scheduler, and LR.
   After that exact comparison it publishes permanent full anchors for steps
   2-4, requires steps 1-4 to remain byte-exact and mode `0444`, requires a fifth
   full milestone to fail coherently at capacity without deleting an anchor,
   and verifies the terminal duplicate-checkpoint guard.
4. `corruption`: requires all-rank identical failure, with no live RNG/model
   mutation, for missing/wrong formal lineage keys, all-rank missing/wrong
   lineage environments, rank-1-only checkpoint lineage, rank-1-only
   environment lineage, missing RNG schema, wrong checkpoint
   schema, wrong world size, missing stream, duplicate per-rank stream, and an
   opaque RNG state corrupted only in rank 1's local view.
   The phase separately checks positive fresh/continue/none boot selection,
   rank-local checkpoint visibility split, expected-mode mismatch before load,
   and failure on a missing historical full anchor.

The existing CPU suites run first.  They cover the explicit A-final
`legacy_missing_rng` init-only waiver and ensure it starts a fresh stage-local
lineage rather than inheriting legacy RNG.

## Required freeze and submission inputs

Fill these values in the JCC spec only after the production owner announces a
freeze and the reviewed files are committed:

- `RELEASE_REPO`: the clean immutable release worktree path used by JCC;
- `EXPECTED_COMMIT`: full 40-character reviewed commit, pushed and clean in the
  live JCC worktree;
- `RESULT_DIR`: a fresh, non-existing `/fsx/evan/...` directory;
- `TRAINING_LINEAGE_PATH`: the immutable Phase-A canonical semantic lineage
  JSON.  It excludes GPU/A-smoke/B-handoff evidence, avoiding self-reference;
- `PHASE_A_MANIFEST_PATH` and `EXPECTED_PHASE_A_MANIFEST_SHA256`: the immutable
  Phase-A authority manifest and its exact SHA-256;
- `MOT_EXPECTED_RUN_LINEAGE_SHA256`: the SHA-256 of that exact lineage file and
  the attempt-invariant lineage stored in the test checkpoint;
- `TESTED_CONFIG` and `EXPECTED_CONFIG_SHA256`: the committed release-relative
  config and its exact SHA-256;
- SHA-256 values for `networks/model_mot.py`, `networks/vae.py`,
  `training/rng_state.py`, `training/checkpoint.py`, `training/formal_boot.py`, `train.py`,
  `evaluation/runners/fixed_sigma_eval.py`, both CPU test scripts, and the GPU
  gate script.

The job checks the clean commit and every SHA before allocating an artifact
directory.  Every JSON receipt repeats the production and gate hashes, and the
postcheck requires all phases to retain exactly those hashes (catching drift
during a run).  The final reviewed commit must reproduce those hashes;
otherwise the run is stale.

## Resource envelope and acceptance

- JCC only: 1 worker x 2 GPUs, 16 CPU, 120 GB host RAM, `maxRetries: 0`.
- Expected execution after scheduling: 8-15 minutes; timeout ceiling is 20
  minutes per phase.
- Expected GPU memory: below 8 GB/GPU (one bf16 FLUX.2 VAE plus a tiny FSDP2
  core); no dataset or 2B transformer is loaded.
- Expected shared storage: below 1 GB for the tiny checkpoint, reference state,
  logs, and receipts.

Acceptance requires both CPU suites, all four phase logs, all four JSON
receipts with top-level `status=verdict=PASS` and `pass=true`, the strict
aggregate `RESULT_RECEIPT.json`, and `_SUCCESS`.  Any missing
receipt, hash drift, world-size drift, non-exact field, or non-coherent failure
matrix blocks launch.  The job closes the evidence member list, writes a
top-level-PASS `MANIFEST.json` and deterministic `SHA256SUMS`, then writes JSON
`_SUCCESS` binding the manifest, aggregate receipt, and checksum-list SHAs.
Files are sealed `0444` and the result directory `0555` before exit.  Phase B's
outer release manifest may bind this evidence; the Phase-A training lineage
must never incorporate it.
