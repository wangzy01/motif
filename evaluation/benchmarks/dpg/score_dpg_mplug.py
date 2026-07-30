#!/usr/bin/env python3
"""DPG-Bench scorer using the ORIGINAL mPLUG VQA model (leaderboard-comparable).

WHY this file exists (read README.md for the full rationale):
  The default DPG scorer in this kit (``score_dpg.py``) uses a local Qwen3-VL
  judge. That judge is INTERNALLY-CONSISTENT ONLY -- great for our B1-vs-v1
  ablation, but NOT directly comparable to published DPG-Bench leaderboard
  numbers, which are produced by the ORIGINAL mPLUG VQA model
  (``damo/mplug_visual-question-answering_coco_large_en`` via ``modelscope``).

  This module reproduces the UPSTREAM protocol faithfully so a run scored here
  IS comparable to the leaderboard -- PROVIDED the images were generated as the
  original expects (the 2x2-grid ``pic_num=4`` tiling; see generate_dpg.py
  ``--grid 2``). FULL comparability needs BOTH: this mPLUG scorer AND pic_num=4.

  >>> Runs in the SEPARATE conda env  /fsx/evan/miniconda3/envs/dpg-mplug      <<<
  >>> (python 3.10 + torch 2.0.1+cu118 + fairseq 0.12.2 + modelscope 1.9.5).   <<<
  >>> It does NOT run in venv-gpu or venv-dpg (those have py3.12, no fairseq).  <<<

PROTOCOL -- byte-for-byte faithful to upstream ``compute_dpg_bench.py``
(``compute_dpg_one_sample``), the ELLA/TencentQQGYLab reference:
  * For each generated image, build the crop list from a 2x2 grid:
      crop_tuples_list = [(0,0,R,R),(R,0,2R,R),(0,R,R,2R),(R,R,2R,2R)]
      crops = crop_tuples_list[:pic_num]   # pic_num=1 -> whole top-left; 4 -> all
    where R = the per-tile resolution (``--resolution``).
  * For each crop, ask the mPLUG VQA model EVERY DPG question for that item_id;
    map answer -> 1.0 if the string == 'yes' else 0.0 (upstream uses ``==`` on
    the raw mPLUG text, which is a clean 'yes'/'no'; we keep that exact rule).
  * DEPENDENCY ZEROING (per crop): a question's score is forced to 0 if ANY of
    its parents (its ``dependency`` list, ignoring parent id 0) was answered
    'no'. (Mirrors upstream's parent-no -> child-zero loop, evaluated per crop.)
  * Per-crop score = mean over all questions AFTER zeroing.
  * Per-image score = mean over crops.
  * DPG overall = mean over images x100.
  * Category breakdown uses ``qid2scores_orig`` -- the *pre-zeroing* yes-rates.
    FAITHFUL DETAIL: upstream reassigns ``qid2scores_orig`` inside the crop loop,
    so the category aggregation uses the LAST crop's pre-zeroing scores only. We
    replicate that exactly (``_LAST_CROP_FOR_CATEGORY``) so numbers match.
      L1 = tuple.split('-')[0]   (entity/attribute/relation/global/other)
      L2 = tuple.split('(')[0]   (e.g. "entity - whole")

Entry point ``score_dir(...)`` mirrors score_dpg.py::score_dir so run_dpg.py can
call EITHER backend (additive ``--backend mplug`` there). The output ``summary``
dict has the same shape, with ``scorer='mplug'`` and a comparability note.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = str(HERE / "dpg_prompts.jsonl")
# damo/mplug_visual-question-answering_coco_large_en (downloaded into MODELSCOPE_CACHE).
DEFAULT_MODEL = "damo/mplug_visual-question-answering_coco_large_en"
# Where the model was cached (the cluster job exports MODELSCOPE_CACHE to this).
DEFAULT_MODELSCOPE_CACHE = "/fsx/evan/modelscope_cache"


def load_data(path: str) -> dict[str, dict]:
    """item_id -> {prompt, questions:[...]} keyed for O(1) lookup (same as score_dpg)."""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


class MPLUG:
    """Loads the original mPLUG VQA pipeline once; answers (PIL image, q) -> text.

    Faithful to upstream ``MPLUG`` in compute_dpg_bench.py: it constructs the
    modelscope ``visual_question_answering`` pipeline on the given device and
    returns ``result['text']`` (a clean 'yes'/'no' from this COCO-VQA model).
    """

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "gpu",
                 modelscope_cache: str | None = DEFAULT_MODELSCOPE_CACHE):
        # Point modelscope at the /fsx cache BEFORE importing the pipeline so the
        # model is read from there (and never re-downloaded on an offline node).
        if modelscope_cache:
            os.environ.setdefault("MODELSCOPE_CACHE", modelscope_cache)
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        # IMPORTANT: pass the LOCAL model DIRECTORY (not the bare model id) so
        # modelscope reads the cached files straight off disk and does NOT re-run
        # its download/integrity machinery (which, in 1.9.5, can re-fetch the
        # 2.8 GB weight and stall). If the model is already a local dir, use it;
        # otherwise resolve <cache>/<model_id> and use that if it exists.
        resolved = model
        if not os.path.isdir(model) and modelscope_cache:
            cand = os.path.join(modelscope_cache, model)
            if os.path.isdir(cand):
                resolved = cand
        # device: modelscope accepts 'gpu' / 'cpu' / 'cuda'. Upstream passes the
        # accelerate device string (e.g. 'cuda:0'); 'gpu' works on a GPU node.
        self.pipeline_vqa = pipeline(
            Tasks.visual_question_answering, model=resolved, device=device)

    def vqa(self, image, question: str) -> str:
        result = self.pipeline_vqa({"image": image, "question": question})
        return result["text"]


def crop_image(input_image, crop_tuple=None):
    """Crop helper, identical to upstream (None -> return as-is)."""
    if crop_tuple is None:
        return input_image
    return input_image.crop((crop_tuple[0], crop_tuple[1], crop_tuple[2], crop_tuple[3]))


def _crop_tuples(resolution: int, pic_num: int):
    """The upstream 2x2-grid crop list, truncated to ``pic_num`` quadrants."""
    R = resolution
    crop_tuples_list = [
        (0, 0, R, R),
        (R, 0, 2 * R, R),
        (0, R, R, 2 * R),
        (R, R, 2 * R, 2 * R),
    ]
    return crop_tuples_list[:pic_num]


def _item_id_from_filename(fname: str) -> str:
    """Map an image filename back to its DPG item_id (same rule as score_dpg).

      * generate_dpg.py:  ``<item_id>__<run>__seed<N>.png``  -> split on '__'
      * upstream/plain:   ``<item_id>.png``                  -> strip extension
    """
    stem = os.path.basename(fname)
    if "__" in stem:
        return stem.split("__")[0]
    return stem.rsplit(".", 1)[0]


def score_one_image(vqa_model: MPLUG, image, questions: list[dict],
                    resolution: int, pic_num: int, detail_writer=None,
                    image_path: str = "") -> dict:
    """Score one image with the upstream per-crop / per-question / zeroing logic.

    Returns per_image_score plus the LAST-crop pre-zeroing yes-rates
    (``qid2scores_orig``) used for category aggregation, exactly as upstream.
    """
    crop_tuples = _crop_tuples(resolution, pic_num)
    qid2question = {q["qid"]: q["question"] for q in questions}
    qid2dependency = {q["qid"]: q["dependency"] for q in questions}

    # absent-parent -> upstream indexes qid2scores[parent] directly, KeyErrors and
    # SKIPS the whole image (e.g. partiprompts97, qid=1 dropped). Match it: skip.
    _present = set(qid2question)
    for _deps in qid2dependency.values():
        if any(p != 0 and p not in _present for p in _deps):
            return None

    qid2scores_orig: dict[int, float] = {}  # reassigned each crop (upstream behaviour)
    scores: list[float] = []
    for crop_tuple in crop_tuples:
        cropped = crop_image(image, crop_tuple)
        qid2scores: dict[int, float] = {}
        qid2answer: dict[int, str] = {}
        for qid, question in qid2question.items():
            answer = vqa_model.vqa(cropped, question)
            qid2answer[qid] = answer
            # upstream: float(answer == 'yes') on the raw mPLUG text.
            qid2scores[qid] = float(answer == "yes")
            if detail_writer is not None:
                detail_writer.write(json.dumps({
                    "file": os.path.basename(image_path), "crop": list(crop_tuple),
                    "qid": qid, "question": question, "answer": answer,
                }, ensure_ascii=False) + "\n")
        # snapshot pre-zeroing scores (category aggregation uses this; upstream
        # keeps only the LAST crop's snapshot, which we faithfully reproduce).
        qid2scores_orig = dict(qid2scores)

        # dependency zeroing: parent answered 'no' -> child forced to 0.
        for qid, parent_ids in qid2dependency.items():
            any_parent_no = False
            for parent_id in parent_ids:
                if parent_id == 0:
                    continue
                if qid2scores.get(parent_id, 0.0) == 0.0:
                    any_parent_no = True
                    break
            if any_parent_no:
                qid2scores[qid] = 0.0

        scores.append(sum(qid2scores.values()) / len(qid2scores))

    per_image = sum(scores) / len(scores)
    return {
        "per_image_score": per_image,
        "crop_scores": scores,
        "qid2scores_orig": qid2scores_orig,
    }


def score_dir(image_dir: str, data_path: str = DEFAULT_DATA,
              model: str = DEFAULT_MODEL, device: str = "gpu",
              resolution: int | None = None, pic_num: int = 4,
              modelscope_cache: str | None = DEFAULT_MODELSCOPE_CACHE,
              limit: int = 0, detail_path: str | None = None) -> dict:
    """Score every image in ``image_dir`` and aggregate DPG metrics (mPLUG backend).

    ``resolution`` is the per-TILE size. For a 2x2 grid (``pic_num=4``) of NxN
    tiles, the saved image is 2N x 2N, so resolution = N. If ``resolution`` is
    None it is inferred as (image_width // grid_side) where grid_side is 2 when
    pic_num==4 else 1. Same return shape as score_dpg.py::score_dir.
    """
    data = load_data(data_path)
    files = sorted(f for f in os.listdir(image_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    if limit:
        files = files[:limit]

    vqa_model = MPLUG(model=model, device=device, modelscope_cache=modelscope_cache)
    from PIL import Image

    per_image_scores: list[float] = []
    cat_l1: dict[str, list[float]] = defaultdict(list)
    cat_l2: dict[str, list[float]] = defaultdict(list)
    records = []
    detail_f = open(detail_path, "w", encoding="utf-8") if detail_path else None
    grid_side = 2 if pic_num == 4 else 1

    for idx, fn in enumerate(files):
        iid = _item_id_from_filename(fn)
        rec = data.get(iid)
        if rec is None:
            print(f"[mplug-score] WARN no DPG entry for item_id '{iid}' ({fn}); skip", flush=True)
            continue
        img = Image.open(os.path.join(image_dir, fn)).convert("RGB")
        res = resolution if resolution is not None else (img.width // grid_side)
        out = score_one_image(vqa_model, img, rec["questions"], resolution=res,
                              pic_num=pic_num, detail_writer=detail_f,
                              image_path=fn)
        if out is None:  # absent-parent item (e.g. partiprompts97) — skip like upstream
            print(f"[mplug-score] skip '{iid}' ({fn}): question depends on a dropped parent qid", flush=True)
            continue
        per_image_scores.append(out["per_image_score"])
        # category aggregation uses the *pre-zeroing* yes-rates (qid2scores_orig).
        qmeta = {q["qid"]: q for q in rec["questions"]}
        for qid, raw in out["qid2scores_orig"].items():
            q = qmeta.get(qid)
            if q is None:
                continue
            cat_l1[q["l1"]].append(raw)
            cat_l2[q["l2"]].append(raw)
        records.append({"file": fn, "item_id": iid,
                        "per_image_score": out["per_image_score"]})
        if (idx + 1) % 25 == 0:
            print(f"[mplug-score] {idx+1}/{len(files)} images "
                  f"running_mean={100*sum(per_image_scores)/len(per_image_scores):.2f}",
                  flush=True)

    if detail_f:
        detail_f.close()

    overall = 100.0 * (sum(per_image_scores) / len(per_image_scores)) if per_image_scores else 0.0
    l1 = {k: 100.0 * (sum(v) / len(v)) for k, v in sorted(cat_l1.items())}
    l2 = {k: 100.0 * (sum(v) / len(v)) for k, v in sorted(cat_l2.items())}
    summary = {
        "scorer": "mplug",
        "model": model,
        "pic_num": pic_num,
        "comparability": ("LEADERBOARD-COMPARABLE iff images were generated with "
                          "pic_num=4 (2x2 grid); original mPLUG VQA + upstream protocol"),
        "n_images": len(per_image_scores),
        "dpg_overall": overall,
        "l1_category": l1,
        "l2_category": l2,
    }
    return {"summary": summary, "per_image": records}


def main():
    ap = argparse.ArgumentParser(description="DPG-Bench ORIGINAL mPLUG scorer.")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="gpu",
                    help="modelscope device: 'gpu'/'cpu'/'cuda:0' (mPLUG inference needs a GPU)")
    ap.add_argument("--resolution", type=int, default=None,
                    help="per-tile size N (2x2 grid image is 2N x 2N). Inferred if omitted.")
    ap.add_argument("--pic-num", type=int, default=4,
                    help="crops of the 2x2 grid: 4 = full upstream protocol (leaderboard); 1 = single image")
    ap.add_argument("--modelscope-cache", default=DEFAULT_MODELSCOPE_CACHE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the result JSON here")
    ap.add_argument("--detail", default=None, help="write per-question JSONL here")
    args = ap.parse_args()

    res = score_dir(args.image_dir, args.data, args.model, args.device,
                    args.resolution, args.pic_num, args.modelscope_cache,
                    args.limit, args.detail)
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[mplug-score] wrote {args.out}")


if __name__ == "__main__":
    main()
