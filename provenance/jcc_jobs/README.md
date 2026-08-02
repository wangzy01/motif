# JCC job specs for the Stage512b-real offline chain

Every job that produced a Stage512b artifact, exactly as submitted. They lived only in a
session scratchpad until 2026-08-02, which meant the artifacts were reproducible in
principle and unreviewable in practice — the same gap that made Stage512a's plan
untraceable (its builders sat in `stage512a_*_control_*` directories outside git; two of
them are now imported under `scripts/offline_arb/`).

Job names carry a `-xxxx` suffix assigned at submission, so the name inside a file is the
name minus that suffix. Find a run with `/fsx/evan/bin/jcc job list` — note `jcc` is not
on PATH, use the full path, and the subcommand is `jcc job submit <file>`, not
`job create -f`.

## The chain, in order

| file | produced |
|---|---|
| `20260801_phase3a_anchors_diag.yaml` | 12-shard diagnostic; exposed 2 reporting defects |
| `20260801_phase3a_anchors_formal_v1_SUPERSEDED.yaml` | first formal N_matrix; retired by the TextAtlas cap |
| `20260802_phase3a_anchors_formal_v2.yaml` | **formal N_matrix in use**, textatlas capped at 0.12 |
| `20260802_phase5_catalog_diag_stride.yaml` | stratified diagnostic catalog (24 shards/domain, stride) |
| `20260802_phase5_catalog_formal.yaml` | **formal catalog**, 89,786,866 rows, `b91e4906…` |
| `20260802_anime_priority_sidecar.yaml` | anime priority, 16,404,684 rows |
| `20260802_weighted_plan_v3.yaml` | **weighted plan**, 245 blocks, `50b5ae2b…` |
| `20260802_animefreq.yaml` | anime tag frequencies, bound by catalog_hash |
| `20260802_caption_sidecar.yaml` | five-domain caption sidecar, 56 GB |
| `20260802_anchors_loader_smoke.yaml` | 13/13 anchors decoded through the production loader |
| `20260802_textatlas_gt_sidecar.yaml` | authoritative TextAtlas GT, 1,941,791 rows |
| `20260802_realized_exposure_audit.yaml` | per-source/cluster/band exposure over all 245 blocks |

## Failures, kept deliberately

`20260802_weighted_plan_v1_FAILED_bad_plan_id.yaml` — refused by `planner_v1.py:363`:
`format_version=2` requires the plan_id to start with `stage512b_offline_exact_v2_`.
Nothing was published. Its planner_input is archived under `_failed_attempts/`.

A v2 of that job also failed and is *not* kept, because its spec is identical to v3: it
died in its own preflight because the working tree was mid-edit and an in-flight
signature change broke a test the job runs. The lesson is about submission hygiene, not
about the spec — **do not submit a job from a dirty working tree**, since the job reads
the live tree.

`20260801_va_strata_audit_FAILED.yaml` — from the VA (full-survivor) line of work, before
the switch to the curated pool. Kept because its gate omitted `synthetic==0` and
`holdout==0` and it wrote `_SUCCESS` even on failure; it is a worked example of a gate
that does not gate.

## Conventions these encode

Pinned venv `/fsx/evan/mot_code/venv-gpu/bin/python`; rank guard exiting 42; missing
interpreter/repo exiting 43; sha-drift exiting 44; create-once violation exiting 45;
`timeout` wrapper on the long call; `maxRetries: 0`; the local test suites run *before*
anything touches data; and a postcheck that re-reads the published artifact rather than
trusting the builder's own exit code.
