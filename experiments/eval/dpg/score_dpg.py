#!/usr/bin/env python3
"""DPG-Bench scorer using a local Qwen3-VL judge (VLM-as-VQA).

WHY this scorer (read README.md for the full rationale):
  The ORIGINAL DPG-Bench scorer is mPLUG VQA via `modelscope`, which pulls in
  `fairseq` + `taming-transformers` + `librosa` + `decord`. fairseq has no
  Python 3.12 wheel and this box only has python3.12 (no conda, no 3.10) — the
  legacy stack is INFEASIBLE here. We therefore PIVOT to a modern VLM judge that
  answers the SAME DPG yes/no questions: the local Qwen3-VL-2B-Instruct.

  >>> Scores from this scorer are INTERNALLY-CONSISTENT ONLY (Qwen3-VL judge). <<<
  >>> They are NOT directly comparable to published mPLUG DPG-Bench numbers.   <<<
  For our use (B1-vs-v1 ablation, relative positioning) internal consistency is
  what matters; leaderboard comparability is out of scope on this box.

PROTOCOL (faithful to compute_dpg_bench.py::compute_dpg_one_sample):
  * For each image, ask the judge every DPG question for that prompt's item_id.
  * Map the answer to 1.0 if 'yes' else 0.0.
  * DEPENDENCY ZEROING: a question's score is forced to 0 if ANY of its parent
    questions (dependency list, ignoring parent id 0) was answered 'no'.
  * Per-image score = mean over all questions (after zeroing).
  * Overall DPG score = mean over images (×100). Category breakdowns:
      L1 = tuple.split('-')[0]  (entity/attribute/relation/global/other)
      L2 = tuple.split('(')[0]  (e.g. "entity - whole")
    Each category's score is the mean of that category's *original* (pre-zeroing)
    per-question yes-rates, exactly as the upstream `qid2scores_orig` aggregation.

DIFFERENCES from upstream that are intentional and documented:
  * pic_num: upstream tiles a 2x2 grid (pic_num=4) and crops quadrants; we
    generate ONE image per prompt (the whole frame, pic_num=1). This is the
    standard single-image setting and matches our generate_dpg.py output.
  * VQA backend is Qwen3-VL, not mPLUG (see above).

This module is import-safe: `score_dir(...)` is the entry point used by run_dpg.py.
Runs in the ISOLATED scorer venv /fsx/evan/mot_code/venv-dpg (NOT venv-gpu).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "/fsx/models/Qwen3-VL-2B-Instruct"
DEFAULT_DATA = str(HERE / "dpg_prompts.jsonl")

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


def load_data(path: str) -> dict[str, dict]:
    """item_id -> {prompt, questions:[...]} keyed for O(1) lookup."""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def parse_yes_no(text: str) -> float:
    """Map a free-form judge answer to 1.0 (yes) / 0.0 (no/uncertain).

    The judge is prompted to answer with a single word, but we are robust to
    extra text: a leading 'yes' wins; otherwise look for any yes/no token.
    """
    t = text.strip().lower()
    if t.startswith("yes"):
        return 1.0
    if t.startswith("no"):
        return 0.0
    has_yes = bool(_YES_RE.search(t))
    has_no = bool(_NO_RE.search(t))
    if has_yes and not has_no:
        return 1.0
    return 0.0  # 'no', ambiguous, or unparseable -> 0 (conservative, like upstream)


class QwenVLJudge:
    """Loads Qwen3-VL once; answers a (PIL image, question) -> 'yes'/'no'."""

    PROMPT_SUFFIX = (
        " Answer with only one word: 'yes' or 'no'."
    )

    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "cuda",
                 dtype: str = "auto", max_new_tokens: int = 8):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        want = torch.bfloat16 if (device == "cuda" and dtype == "auto") else (
            None if dtype == "auto" else getattr(torch, dtype))
        kwargs = {"dtype": want} if want is not None else {"dtype": "auto"}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, device_map=device if device == "cuda" else None, **kwargs)
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device

    def vqa(self, image, question: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question + self.PROMPT_SUFFIX},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.no_grad():
            gen = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        return text.strip()


def _item_id_from_filename(fname: str) -> str:
    """Map an image filename back to its DPG item_id.

    Supports two layouts:
      * generate_dpg.py:  ``<item_id>__<run>__seed<N>.png``  -> split on '__'
      * upstream/plain:   ``<item_id>.png``                  -> strip extension
    """
    stem = os.path.basename(fname)
    if "__" in stem:
        return stem.split("__")[0]
    return stem.rsplit(".", 1)[0]


def score_one_image(judge: QwenVLJudge, image, questions: list[dict]) -> dict:
    """Return per-question raw answers + the dependency-zeroed scores."""
    from PIL import Image as _PILImage  # noqa: F401  (ensures PIL available)

    qid2raw: dict[int, float] = {}   # pre-zeroing yes-rate (1/0)
    qid2answer: dict[int, str] = {}
    qid2meta: dict[int, dict] = {}
    for q in questions:
        ans = judge.vqa(image, q["question"])
        sc = parse_yes_no(ans)
        qid2raw[q["qid"]] = sc
        qid2answer[q["qid"]] = ans
        qid2meta[q["qid"]] = q

    # absent-parent -> match upstream, which indexes qid2scores[parent] directly,
    # raises KeyError and SKIPS the whole image (e.g. partiprompts97, whose qid=1
    # was dropped). Returning None lets score_dir drop it (denominator excludes it).
    present = set(qid2raw)
    for q in questions:
        if any(p != 0 and p not in present for p in q["dependency"]):
            return None

    # dependency zeroing — CASCADE like upstream compute_dpg_one_sample: iterate
    # qids ascending and test parents against the MUTATED dict, so a child of a
    # transitively-zeroed parent (grandparent said 'no') is also zeroed.
    qid2final = dict(qid2raw)
    for q in sorted(questions, key=lambda x: x["qid"]):
        qid = q["qid"]
        parents = [p for p in q["dependency"] if p != 0]
        if any(qid2final.get(p, 0.0) == 0.0 for p in parents):
            qid2final[qid] = 0.0
    per_image = (sum(qid2final.values()) / len(qid2final)) if qid2final else 0.0
    return {
        "per_image_score": per_image,
        "qid2raw": qid2raw,         # used for category aggregation (orig)
        "qid2final": qid2final,
        "qid2answer": qid2answer,
        "qid2meta": qid2meta,
    }


def score_dir(image_dir: str, data_path: str = DEFAULT_DATA,
              model_path: str = DEFAULT_MODEL, device: str = "cuda",
              limit: int = 0, detail_path: str | None = None) -> dict:
    """Score every image in ``image_dir`` and aggregate DPG metrics.

    Returns a dict with ``summary`` (overall + L1/L2 category scores ×100) and
    ``per_image`` records. ``limit`` caps the number of images (0 = all).
    """
    data = load_data(data_path)
    files = sorted(f for f in os.listdir(image_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    if limit:
        files = files[:limit]

    judge = QwenVLJudge(model_path=model_path, device=device)
    from PIL import Image

    per_image_scores: list[float] = []
    cat_l1: dict[str, list[float]] = defaultdict(list)
    cat_l2: dict[str, list[float]] = defaultdict(list)
    records = []
    detail_f = open(detail_path, "w", encoding="utf-8") if detail_path else None

    for idx, fn in enumerate(files):
        iid = _item_id_from_filename(fn)
        rec = data.get(iid)
        if rec is None:
            print(f"[score] WARN no DPG entry for item_id '{iid}' ({fn}); skip", flush=True)
            continue
        img = Image.open(os.path.join(image_dir, fn)).convert("RGB")
        res = score_one_image(judge, img, rec["questions"])
        if res is None:  # absent-parent item (e.g. partiprompts97) — skip like upstream
            print(f"[score] skip '{iid}' ({fn}): question depends on a dropped parent qid", flush=True)
            continue
        per_image_scores.append(res["per_image_score"])
        # category aggregation uses the *original* (pre-zeroing) yes-rates,
        # exactly like upstream's qid2scores_orig path.
        for q in rec["questions"]:
            raw = res["qid2raw"][q["qid"]]
            cat_l1[q["l1"]].append(raw)
            cat_l2[q["l2"]].append(raw)
        records.append({"file": fn, "item_id": iid,
                        "per_image_score": res["per_image_score"]})
        if detail_f:
            for q in rec["questions"]:
                detail_f.write(json.dumps({
                    "file": fn, "item_id": iid, "qid": q["qid"],
                    "tuple": q["tuple"], "question": q["question"],
                    "answer": res["qid2answer"][q["qid"]],
                    "raw": res["qid2raw"][q["qid"]],
                    "final": res["qid2final"][q["qid"]],
                }, ensure_ascii=False) + "\n")
        if (idx + 1) % 25 == 0:
            print(f"[score] {idx+1}/{len(files)} images "
                  f"running_mean={100*sum(per_image_scores)/len(per_image_scores):.2f}",
                  flush=True)

    if detail_f:
        detail_f.close()

    overall = 100.0 * (sum(per_image_scores) / len(per_image_scores)) if per_image_scores else 0.0
    l1 = {k: 100.0 * (sum(v) / len(v)) for k, v in sorted(cat_l1.items())}
    l2 = {k: 100.0 * (sum(v) / len(v)) for k, v in sorted(cat_l2.items())}
    summary = {
        "scorer": "qwen3vl",
        "model": os.path.basename(model_path),
        "comparability": "internally-consistent only (Qwen3-VL judge; NOT comparable to published mPLUG DPG numbers)",
        "n_images": len(per_image_scores),
        "dpg_overall": overall,
        "l1_category": l1,
        "l2_category": l2,
    }
    return {"summary": summary, "per_image": records}


def main():
    ap = argparse.ArgumentParser(description="DPG-Bench Qwen3-VL scorer.")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the result JSON here")
    ap.add_argument("--detail", default=None, help="write per-question JSONL here")
    args = ap.parse_args()

    res = score_dir(args.image_dir, args.data, args.model, args.device,
                    args.limit, args.detail)
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[score] wrote {args.out}")


if __name__ == "__main__":
    main()
