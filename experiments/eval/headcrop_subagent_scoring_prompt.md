# Subagent Scoring Prompt: Headcrop Matrix

You will receive original generated PNGs as local images. Inspect each image at
native resolution. Do not infer from thumbnails.

For every image, return exactly one JSON object in JSONL form. Use the filename
to fill `run`, `prompt_id`, `size`, and `seed` when possible:

`<prompt_id>__<run>__<width>x<height>__seed<seed>.png`

Required fields:

```json
{
  "image": "/abs/path.png",
  "run": "rd256_final|rd512a_g1|rd512a_latest",
  "prompt_id": "anime_eval_000",
  "prompt_style": "tag|nl_plain|nl_emph|real_nl",
  "composition_target": "full_body|full_body_multi|lower_body_risk|upper_body_control|non_fullbody_control",
  "size": "512x512",
  "seed": 7,
  "primary_label": "A_full_ok|B_head_cut|B_foot_cut|B_side_cut|C_reasonable_crop|D_other_bad",
  "head_cut": false,
  "foot_cut": false,
  "side_cut": false,
  "subject_complete_0to5": 5,
  "composition_follow_0to5": 4,
  "notes": "short concrete observation"
}
```

Rubric:

- `A_full_ok`: requested full-body or multi-person composition is complete enough;
  head/face and feet/lower body are not cut by the image border.
- `B_head_cut`: prompt asks for a full-body / standing / multi-person view, but
  the head, face, hair top, or upper body is cut by the top image border.
- `B_foot_cut`: prompt asks for full body / standing / lower-body readability, but
  feet, shoes, boots, or lower legs are cut by the bottom border.
- `B_side_cut`: relevant body/person is cut by left/right border in a way that
  makes the figure visibly incomplete.
- `C_reasonable_crop`: close-up / half-body / portrait crop is reasonable for the
  prompt, with no accidental head truncation.
- `D_other_bad`: severe anatomy collapse, missing main subject, blank/near-blank,
  or unrelated output; explain briefly.

Continuous completeness score:

- `5`: requested body/group visible, no meaningful border crop.
- `4`: tiny crop or uncertain shoes/hair edge, still basically complete.
- `3`: one minor requested extremity missing, not the dominant failure.
- `2`: obvious head/feet/side truncation on one main subject.
- `1`: multiple main subjects or major body regions truncated.
- `0`: unusable, blank, or not coherent.

Important:

- Count head/foot/side cuts only when the prompt target expects full-body,
  standing, multi-person, lower-body readability, or an object/person fit in frame.
- For explicit close-up / upper-body controls, use `C_reasonable_crop` unless the
  head itself is accidentally cut.
- Fill `composition_target` from the batch instruction when provided. If absent,
  infer it conservatively from the prompt: full-body/multi-person/lower-body risk
  vs upper-body or non-full-body control.
- Be conservative but concrete. If uncertain, mark the boolean false and mention
  uncertainty in notes.
