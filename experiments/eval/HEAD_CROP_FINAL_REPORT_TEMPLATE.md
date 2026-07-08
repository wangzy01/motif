# Stage256 Head/Foot Truncation Evaluation Report

Status: template. Fill after `headcrop_matrix_now_comprehensive` images are
generated and all images are scored by subagents.

## Executive Answer

- Decision: `resolved | mostly_resolved | partially_resolved | not_resolved`
- Recommendation: `continue stage512a | wait for later 512a gate | retrain/fix 256 | other`
- One-sentence basis:

## Compared Checkpoints

| run | checkpoint | step | exposure estimate | train setup | notes |
| --- | --- | ---: | ---: | --- | --- |
| rd256_final | | | | stage256 center-crop square | |
| rd512a_g1 | `/fsx/evan/checkpoints/mot_pilot_v1/redesign_512a_gates/gate_g1.pt` | | ~25M | stage512a multi-AR | |
| rd512a_latest | | | | stage512a multi-AR | |

## Evaluation Matrix

| domain | prompt style | prompt count | seeds | sizes / ARs | runs |
| --- | --- | ---: | --- | --- | --- |
| anime | tag | 13 | 7, 1007 | 512x512, 416x624, 624x416, 384x688, 688x384 | rd256_final, rd512a_g1, rd512a_latest |
| anime | NL plain | 13 | 7, 1007 | 512x512, 416x624, 624x416, 384x688, 688x384 | rd256_final, rd512a_g1, rd512a_latest |
| anime | NL + full-body emphasis | 13 | 7, 1007 | 512x512, 416x624, 624x416, 384x688, 688x384 | rd256_final, rd512a_g1, rd512a_latest |
| real | NL/full-body/control | 12 | 7, 1007 | 512x512, 416x624, 624x416, 384x688, 688x384 | rd256_final, rd512a_g1, rd512a_latest |

Expected images from current comprehensive matrix job: 1530.

## Quantitative Summary

Use `experiments/eval/aggregate_headcrop_scores.py` output.

Coverage audit:

- Expected images:
- Scored images:
- Missing scores:
- Extra / duplicate scores:

| run | n | any truncation | head cut | foot cut | side cut | full-ok | subject mean | composition mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rd256_final | | | | | | | | |
| rd512a_g1 | | | | | | | | |
| rd512a_latest | | | | | | | | |

## Required Slices

### Anime vs Real

| run | domain | n | any truncation | head cut | foot cut | subject mean | composition mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

### Prompt Style

| run | prompt style | n | any truncation | head cut | foot cut | subject mean | composition mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

### Aspect Ratio

| run | AR bucket | n | any truncation | head cut | foot cut | side cut | subject mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

### Worst Prompt IDs

| prompt_id | domain/style | rd256_final label | rd512a_g1 label | rd512a_latest label | note |
| --- | --- | --- | --- | --- | --- |

## Qualitative Findings

- Full-body / multi-person:
- Dynamic poses:
- Half-body / portrait controls:
- Anime vs real:
- Prompt emphasis effect:
- Multi-AR effect:

## Caveats

- Square 512 samples from the stage256 checkpoint test composition bias, but are
  not a native stage256 quality benchmark.
- Non-square stage256 samples are stress/generalization probes outside the
  native stage256 training regime; separate this from in-distribution 256
  diagnostic notes.
- All matrix H/W values are multiples of 16, matching the Flux2 VAE downsample
  constraint. The current stage256 and stage512a configs use `patch_size: 1`.
- Any pre-matrix existing-image comparisons are directional only; final judgment
  must use same prompt/seed/size matrix outputs.
- Subagent labels are visual judgments. Important borderline cases should be
  called out with image paths.

## Artifacts

- Matrix images:
- Sampling manifests:
- Subagent score JSONL:
- Aggregated summary JSON/Markdown:
- Interim notes:
