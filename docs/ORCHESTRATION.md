# 7.18 Stage256 execution authority

This document is the create-once execution contract for
`stage256_20260718_causalmlf_offline_exact_v1`.  It is a draft until the whole
local implementation passes static/synthetic review and is copied to the
server under new file names.  Never modify the frozen legacy training tree to
make this release work.

## Frozen intent

- Formal topology: 8 workers × 8 GPUs, per-rank batch 36, global batch 2304.
- Exact training length: 166,667 steps and 384,000,768 image exposures
  ("384M nominal").
- Fresh initialization from the configured pretrained components.  There is no
  `trainer.init_from`; the formal checkpoint leaf must not exist at launch.
- Model fingerprint: causal two-layer text adapter, `bidirectional=false`,
  `rope=false`, 12-layer MLF, other ablations off.  Ablation fingerprint SHA:
  `70546507a768a9d01705b8118b6e3cfea3f055f22db5e9bcdfb3e25e7d022642`.
- LR `2.121e-4`, Qwen3-VL-2B text tower, Flux2 VAE, EMA 0.9999, CFG dropout
  0.1, 384 text tokens, sigma shift 1.0.
- Offline exact and native multi-AR remain enabled.  W64 must retain 20 active
  native buckets; the seven evaluation ARs are reporting targets, not invented
  training buckets.
- Stage256 anime exposure must be strict-NL only.  Planned and runtime tag-only
  exposure are both exactly zero.
- Heavy CPU, large-memory and large-I/O work runs only on CPU JCC.  The smoke
  uses exactly one 8-GPU worker.  Formal 8×8 is explicitly authorized.
- Every job uses `maxRetries: 0`.  A failed or partial leaf is preserved for
  forensics; do not retry, rename around it, delete it, or use another namespace
  without a new human decision.

## Namespaces

```text
repo                  /fsx/evan/mot_code/mot_pilot_v1_cleanup_crop
release root          /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_stage256_20260718_semantic_v1
formal results        /fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v1
formal checkpoints    /fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v1
smoke checkpoints     /fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v1_smoke_1x8
formal JCC name        evan-offarb-s256-20260718-causalmlf-8x8-v1
```

The formal checkpoint and result leaves must be absent before the formal
attempt.  The release root is progressively populated only by the jobs below;
each canonical output is create-once.

## Dependency chain

Run one row at a time.  Before each submission, check both the exact JCC name
and every canonical output leaf.  An existing job or leaf means inspect, never
submit a duplicate.

| Gate | JCC YAML / exact name | Requires | Creates |
|---|---|---|---|
| O1 | `offline_arb_stage256_20260718_overlay_build.yaml` / `evan-offarb-s256-20260718-overlay-build-v1` | old immutable catalog, caption sidecar, 17-file anime source manifest, approved sampling JSON | `overlay/{header.json,_SUCCESS,cells/*}` |
| O2 | `offline_arb_stage256_20260718_overlay_verify.yaml` / `evan-offarb-s256-20260718-overlay-verify-v1` | O1 `_SUCCESS` | `OVERLAY_VERIFY_SUCCESS.json` |
| P1 | `offline_arb_stage256_20260718_selection_plans.yaml` / `evan-offarb-s256-20260718-selection-plans-v1` | O2 | W8/W64 `planner_input.json`, companion binding, `selection_plan/`, validation report; `SELECTION_PLANS_SUCCESS.json` |
| P2 | `offline_arb_stage256_20260718_semantic_materialize.yaml` / `evan-offarb-s256-20260718-semantic-materialize-v1` | P1 + O2 | W8/W64 `plan/` and `SEMANTIC_MATERIALIZE_SUCCESS.json` |
| P3 | `offline_arb_stage256_20260718_semantic_validate.yaml` / `evan-offarb-s256-20260718-semantic-validate-v1` | P2 | each `plan/PLAN_AUDIT_SUCCESS.json`; root `SEMANTIC_VALIDATE_SUCCESS.json` |
| F1 | `offline_arb_stage256_20260718_freeze.yaml` / `evan-offarb-s256-20260718-freeze-v1` | P3 | `FREEZE_MANIFEST.json`, create/verify receipts, upstream full-freeze verification |
| S1 | `offline_exact_stage256_20260718_smoke_1x8.yaml` / `evan-offarb-s256-20260718-causalmlf-smoke-1x8-v1` | F1 | exact three-step checkpoint plus `SMOKE_CHECKPOINT_SUCCESS.json` and `SMOKE_SUCCESS.json` |
| H1 | `offline_arb_stage256_20260718_final_health.yaml` / `evan-offarb-s256-20260718-final-health-v1` | S1 | fresh full release/upstream verification and `FINAL_HEALTH_SUCCESS.json` |
| T1 | `offline_arb_stage256_20260718_final_static.yaml` / `evan-offarb-s256-20260718-final-static-v1` | H1; formal YAML health hash pinned | `STATIC_SUCCESS.json`, `STATIC_CONTRACT_SUCCESS.json` |
| G1 | `offline_exact_stage256_20260718_FORMAL_8x8.yaml` / `evan-offarb-s256-20260718-causalmlf-8x8-v1` | T1 + H1, fresh checkpoint/result namespaces | running 8×8 formal training |

Do not run T1 before H1.  After H1 succeeds:

1. Compute `sha256sum FINAL_HEALTH_SUCCESS.json`.
2. Replace the single `__PIN_FINAL_HEALTH_SHA256__` placeholder in the formal
   YAML with that lowercase 64-character value.
3. Verify there is no remaining placeholder and make no other formal-YAML edit.
4. Submit T1.  Its receipt binds the final formal-YAML SHA.
5. Only after T1 PASS may G1 be submitted.

## Semantic data contract

O1 does not rebuild or renumber the 94.99M-row physical catalog.  It publishes
per `(bucket, domain)` physical row IDs and weights.  Anime eligibility is the
OR of stripped, non-empty refined/medium/short NL fields; whitespace-only NL is
not eligible.  Anime priority/repeat, real band-first weights, synthetic
weights, TextAtlas uniform sampling and the 11 known exclusions are reproduced
offline.

P1 must use `make_stage256_semantic_planner_input.py`; using the legacy
`make_planner_input.py` with the physical catalog N is a hard error.  P1 takes
`overlay.header.N_select` as N, reconstructs and compares the 95-bucket table,
domains and mix, and creates a companion binding because the legacy v1 planner
schema cannot carry extension keys.  W8 is H=32/bs36; W64 is
H=166667/bs36.  P2 binds the exact overlay header and translates the selection
schedule to physical catalog row IDs.  P3 audits every slot, exact quotas,
batch uniqueness, physical ranges, overlay membership, exclusions, repeat
caps, native ARs and formal prefix.

The runtime class is only
`stage256_20260718.offline_semantic_dataset.OfflineSemanticExactDataset`.
It uses the new top-level package and adds a final NL fail-closed check; it does
not edit `training/`, `networks/`, or the live/preparation training code.

## GPIC and freeze chain

The immutable GPIC authority remains:

```text
buildtag  evan-mot-crop-offlinearb-gpic-s3-production-build-zg8g
manifest  a18ba1f1250e3b00abdc1051dc71e08a254b8855d5b61b9975e88d2e6ae2ee05
```

F1 and H1 perform the expensive complete legacy freeze verification on CPU JCC
and hash the new overlay/plans/runtime/config.  GPU jobs never create or fully
rehash the release.  S1 and every G1 node still run the existing compact GPIC
control preflight; runtime retains the collective index and per-segment checks.

The package directory must contain exactly the three frozen runtime sources
`__init__.py`, `semantic_plan.py`, and `offline_semantic_dataset.py`; any
`__pycache__`, `.pyc`, or other object is a hard freeze failure.  Every
downstream job that can import the new package redirects Python bytecode to a
job/node-local `/local` path.  All data artifact trees are closed without an
ignore rule.

## Submission discipline

Use only `/fsx/evan/bin/jcc` through `SSH CPU-AWS-0623`.  Before submitting any
row:

1. `jcc job list`: exact name absent, or inspect the existing exact job.
2. Check canonical output leaf/receipt.  If present, validate it; never submit.
3. Check all predecessor receipts are regular, non-symlink PASS files and their
   bound hashes still match.
4. For S1/G1, inspect current GPU jobs.  S1 adds 8 GPUs and must not overlap
   another evaluation that would exceed the eight-GPU test cap.  G1 adds the
   explicitly approved 64 training GPUs.
5. Submit exactly the published YAML.  Do not pass architecture, LR, plan or
   init checkpoints as dotlist overrides.
6. On any failure or partial output, stop the dependent chain and report the
   evidence paths.  No automatic retry, cleanup, namespace switch or code edit.

## Final pre-submit checklist

- [ ] New config differs from old offline-exact only by the approved ten keys,
      including exact exposure 384,000,768.
- [ ] No `trainer.init_from`; formal checkpoint/result leaves absent.
- [ ] O2 proves all 252 anime sidecar parts, all cell hashes/weights and
      `tag_only_exposure=0`.
- [ ] P1 header N equals overlay `N_select`, not physical catalog N.
- [ ] W64 audit: 166,667 steps, 384,000,768 exposures, exact native bucket
      indices `[25,27,29,31,33,35,38,40,45,48,49,54,56,59,61,63,65,67,69,71]`,
      exact domain totals `anime=103,680,207`, `monet_real=155,520,311`,
      `gpic=99,456,199`, `synthetic=19,584,039`, `textatlas=5,760,012`,
      11 exclusions at zero exposure, anime tag-only exposure zero.
- [ ] W8 audit prefix: exactly 3 steps / 864 exposures; both square and
      non-square buckets are represented in its H=32 schedule.
- [ ] Freeze create and verify receipts bind identical manifest SHA.
- [ ] Semantic headers bind real legacy-loader catalog/assignment identities:
      direct, upstream, and `hashes.*` values are identical to the catalog.
- [ ] Three-step smoke checkpoint proves architecture/keyset/parameter count,
      finite state, EMA, and nonzero updates in zero-init adapter/MLF groups.
- [ ] FINAL_HEALTH SHA pinned into formal YAML; final static PASS generated
      afterward and binds that exact YAML SHA.
- [ ] Formal JCC exact name/output/checkpoint namespaces absent; 8×8 capacity
      available; then and only then submit G1.
