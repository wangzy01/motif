# Head Crop Evaluation Runbook

Purpose: evaluate whether stage512a multi-AR training resolves the stage256
anime head/foot truncation bias.

## Current Plan

- Matrix size: 1530 images.
- Runs: `rd256_final`, `rd512a_g1`, `rd512a_latest`.
- Seeds: nominal `7` and `1007`.
- Square slices:
  - anime tag13 at 512x512
  - anime NL plain13 at 512x512
  - anime NL+full-body-emphasis13 at 512x512
  - real12 at 512x512
- Multi-AR slices:
  - anime tag13 at 416x624, 624x416, 384x688, 688x384
  - anime NL plain13 at 416x624, 624x416, 384x688, 688x384
  - anime NL+full-body-emphasis13 at 416x624, 624x416, 384x688, 688x384
  - real12 at 416x624, 624x416, 384x688, 688x384

The latest stage512a checkpoint is snapshotted at job start so every slice uses
the same weights even if training continues to update `last.pt`.

## JCC Jobs

- Current comprehensive matrix job:
  `evan-mot-eval-headcrop-matrix-now-comprehensive-6dng`

Use:

```bash
/fsx/evan/bin/jcc job list
/fsx/evan/bin/jcc job log <job-name> -p all -n 300
```

Do not run Python locally for verification or aggregation.

## Gate Checks

Stage512a G1:

- `/fsx/evan/checkpoints/mot_pilot_v1/redesign_512a_gates/gate_g1.pt`
- Expected step >= 18000, exposure estimate `step * 1536`.

Stage512a latest/current:

- Source: `/fsx/evan/checkpoints/mot_pilot_v1/redesign_512a_multiar_anchor_global1536_20260705/last.pt`
- Job snapshot: `/fsx/evan/mot_code/data_analysis/outputs/headcrop_matrix_now_comprehensive/ckpts/rd512a_latest_eval_snapshot.pt`
- Expected step > 18000, exposure estimate `step * 1536`.

Stage256 final:

- `/fsx/evan/checkpoints/mot_pilot_v1/redesign_256_causalmlf_global2304_20260630/last.pt`
- Expected step >= 128000, exposure estimate `step * 2304`.

## After Matrix Generation

Expected matrix outputs:

- Images: `/fsx/evan/mot_code/data_analysis/outputs/headcrop_matrix_now_comprehensive/all_images.txt`
- Manifests: `/fsx/evan/mot_code/data_analysis/outputs/headcrop_matrix_now_comprehensive/all_manifests.txt`
- Subagent batch plan:
  `/fsx/evan/mot_code/data_analysis/outputs/headcrop_matrix_now_comprehensive/subagent_batches/subagent_batch_plan.json`

Each generated image must be inspected by a gpt-5.5 subagent at original
resolution. Use the per-batch `.md` files under `subagent_batches/` as the
instructions and attach the listed PNGs as local images.

Save returned JSONL score rows, then aggregate through JCC:

```bash
/fsx/evan/mot_code/venv-gpu/bin/python experiments/eval/aggregate_headcrop_scores.py \
  --scores <score-jsonl...> \
  --manifests $(cat /fsx/evan/mot_code/data_analysis/outputs/headcrop_matrix_now_comprehensive/all_manifests.txt) \
  --out-dir /fsx/evan/mot_code/data_analysis/outputs/headcrop_matrix_now_comprehensive/score_summary
```

The aggregation command must be run inside a JCC job, not locally.

## Report Requirements

The final report must include:

- checkpoint paths, steps, and exposure estimates for all three runs
- expected-vs-scored coverage audit
- headline decision: resolved / mostly resolved / partially resolved / not resolved
- quantitative slices by run, domain, prompt style, AR, seed, and composition target
- qualitative notes on full-body, multi-person, dynamic pose, anime vs real, and controls
- caveat that 512/non-square rd256 samples are stress/generalization probes, not
  native stage256 quality comparisons
- explicit recommendation on whether 256 needs retraining or whether stage512a
  has sufficiently corrected the composition bias
