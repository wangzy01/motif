# Headcrop Matrix Visual Rubric

Scope: manual / subagent scoring for the stage256-final vs stage512a-G1 vs
stage512a-latest head/foot truncation evaluation. Score original generated PNGs
at native resolution; do not rely on thumbnails.

## Labels

- `A_full_ok`: requested full-body or multi-person composition is complete enough;
  head/face and feet/lower body are not cut by the image border.
- `B_head_cut`: prompt asks for a full-body / standing / multi-person view, but
  the head, face, hair top, or upper body is cut by the top image border.
- `B_foot_cut`: prompt asks for a full-body / standing / lower-body-readable view,
  but feet, shoes, boots, or lower legs are cut by the bottom border.
- `B_side_cut`: relevant body/person is cut by left/right border in a way that
  makes the figure visibly incomplete.
- `C_reasonable_crop`: close-up / half-body / portrait crop is reasonable for the
  prompt, with no accidental head truncation.
- `D_other_bad`: severe anatomy collapse, missing main subject, blank/near-blank,
  or unrelated output; note the failure mode.

For quantitative reporting:

- `trunc_any = B_head_cut or B_foot_cut or B_side_cut`
- `head_cut = B_head_cut`
- `foot_cut = B_foot_cut`
- `complete_rate = A_full_ok / full_body_target_count`
- Controls (`composition_target=non_fullbody_control` or `upper_body_control`) are
  reported separately and should not be counted in full-body B-rate.

## Score Fields

Each image should receive one JSON object:

```json
{
  "image": "/abs/path.png",
  "run": "rd256_final|rd512a_g1|rd512a_latest",
  "prompt_id": "anime_eval_000",
  "prompt_style": "tag|nl_plain|nl_emph|real_nl",
  "composition_target": "full_body|full_body_multi|lower_body_risk|upper_body_control|non_fullbody_control",
  "size": "512x512",
  "seed": 7,
  "primary_label": "A_full_ok",
  "head_cut": false,
  "foot_cut": false,
  "side_cut": false,
  "subject_complete_0to5": 5,
  "composition_follow_0to5": 4,
  "notes": "short concrete observation"
}
```

Use `subject_complete_0to5` as the continuous score:

- `5`: full requested body/group visible, no meaningful border crop.
- `4`: tiny crop or uncertain shoes/hair edge, still basically complete.
- `3`: one minor requested extremity missing, but not the dominant failure.
- `2`: obvious head/feet/side truncation on one main subject.
- `1`: multiple main subjects or major body regions truncated.
- `0`: unusable, blank, or not a coherent image.

## Reporting Slices

Aggregate at least these slices:

- checkpoint: `rd256_final`, `rd512a_g1`, `rd512a_latest`
- domain: anime vs real
- prompt style: tag, NL plain, NL with full-body emphasis, real NL
- aspect ratio: square, portrait 2:3, landscape 3:2, tall 9:16, wide 16:9
- seed: 7 vs 1007
- target type: full-body / multi-person vs control

Decision focus: whether `rd512a_latest` has materially lower `trunc_any` and
`head_cut` than `rd256_final`, and whether the improvement holds for anime full-
body prompts across both tag and NL-emphasis, without introducing the same issue
on real prompts.
