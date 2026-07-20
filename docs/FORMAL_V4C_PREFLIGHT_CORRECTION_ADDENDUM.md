# Formal-v4c CPU preflight correction

The create-once v4 release files remain unchanged.  The v4 CPU YAML was never
submitted because a login-node import check found its missing repository
`PYTHONPATH`.  The create-once v4b YAML added that environment and was submitted
as actual CPU-only job
`evan-offarb-s256-20260718-pause-preflight-v4-84gb`.

V4b failed safely before publishing `PRE_RESUME384_SUCCESS.json`.  The full
checkpoint validator had already passed far enough to begin attempt/log audit;
the concrete failure was that the generic v3 `stat_identity` helper rejects an
empty file.  In main-v3, rank zero owns stdout while ranks 1--7 legitimately
produced zero-byte train stdout files.  All eight files are ordinary,
non-symlink files; all eight corresponding node-monitor files are nonempty and
healthy.  An empty nonzero-rank stdout is absence of output, not missing rank
evidence or a training failure.

This correction authorizes one fresh CPU-only/maxRetries=0 preflight base,
`evan-offarb-s256-20260718-pause-preflight-v4c`.  Its wrapper changes only the
main-v3 log audit:

- exactly one ordinary train-log file for every rank 0--7 remains mandatory;
- rank-zero remains nonempty and must prove exact step-384 resume,
  `already_at_checkpoint`, and observed step 3000;
- all eight logs remain scanned for fatal, traceback, OOM, NCCL, non-finite,
  and marker-collision tokens;
- an empty train stdout is allowed only for nonzero ranks;
- all eight node-monitor files remain mandatory, nonempty, ordinary, and must
  contain zero recent memory pressure.

Checkpoint, model/EMA/optimizer/scheduler, metrics, semantic, GPIC, replay
guard, source receipt, target, timing, and GPU successor checks are unchanged.
The wrapper and this correction are SHA-pinned inside the v4c CPU YAML, whose
SHA is in turn recorded by the exclusive v4 receipt.  The failed v4b job and
all v4/v4b files remain read-only evidence.  No GPU work is authorized by this
correction beyond the original delayed main-v4 successor.

## Follow-up delay authority

At 2026-07-19 11:33 CST, after the v4b failure was reported and before v4c was
deployed or submitted, the user explicitly directed that no further JCC work
be submitted for five hours.  Therefore v4c remains local-only until
2026-07-19 16:34 CST (unix 1784450040).  Neither this CPU preflight nor any GPU
successor may be submitted before that time.  At the first wake at or after the
deadline, the controller must re-read the local execution memory, verify that
main-v3 and v4b remain terminal and that no competing writer exists, then
deploy and submit v4c create-once.  Main-v4 remains downstream of a newly
validated exclusive success receipt.

## Restriction lifted and v4d binding

At 2026-07-19 14:36 CST the user explicitly lifted the temporary restriction
and authorized immediate JCC submission and GPU use.  This newest instruction
supersedes the 16:34 CST delay above.  V4c may therefore be deployed and
submitted immediately.  Because the already-published main-v4 runner embeds
the superseded time gate, it remains unused and read-only; successful v4c
preflight instead binds the fresh main-v4d addendum, runner, YAML, attempt root,
and JCC base.  No training semantic changes.
