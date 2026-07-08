#!/usr/bin/env python3
"""B6 tagger-based concept hit-rate / coverage scorer.

Tags generated images with an anime tagger and, per prompt, scores how well the
generated image realises the target general / character / copyright concepts.

Two metric families (both reported), aligned with the pretrain team's naifu
``evaluation/concept_v2/eval_tags.py`` so our numbers are directly interpretable
against theirs (same tagger family, same coverage-at-threshold definition):

  * recall@prod  = |{target tag with prob >= production threshold}| / |target|
                   (production thresholds general=0.2 / character=0.35 / copyright=0.35,
                   matching the tagger-inference service). This is the original
                   ``*_hit`` field, kept for continuity.
  * coverage tiers (naifu concept_v2): fraction of target tags whose predicted
    prob exceeds {0, 0.5, 0.8}  ->  ``*_cov0`` / ``*_cov50`` / ``*_cov80``.
    cov0 ~= "tagger assigned any non-trivial prob"; cov80 = high-confidence hit.

Tagger backends (auto-selected; ``--backend`` to force):

  company  : the company anime tagger via ``transformers.pipeline(trust_remote_code
             =True)`` — the SAME model naifu concept_v2 uses. Default is the newer
             ``incantor/tagger_3_1_24`` with automatic fallback to the proven
             ``incantor/tagger_3_1_13`` if 3_1_24 fails to load. GATED on HF:
             needs HF_TOKEN (or a pre-populated cache). Output categories:
             general / character / copyright, each as {tag: prob}.
  mldanbooru : public fallback ``kiriyamaX/mld-caformer`` (flat tag list, folded
             into general). Used when the company model is unavailable.

The retrieval threshold for the pipeline is set LOW (so we get probs for target
tags even below the production threshold), enabling the cov0/cov50/cov80 tiers.

Exposes ``load_tagger()`` -> object with ``.tag(pil) -> {cat: {tag: prob}}`` and a
``score_dir`` helper, plus a CLI. Import-only; never starts a service.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

# Tagger model. We tried incantor/tagger_3_1_24 (2026-06-28): it LOADS but its HF
# custom pipeline returns an incompatible ["results"] shape under our
# pipeline(min_threshold=...) call -> all-zero hits. Per Evan, fall back to the
# proven incantor/tagger_3_1_13 (drop-in with this call). To use 3_1_24 later it
# needs naifu's build_pipeline loader (different interface), not this path.
# Override with TAGGER_MODEL_ID env if a compatible model is available.
COMPANY_MODEL_ID = os.environ.get("TAGGER_MODEL_ID", "incantor/tagger_3_1_13")
COMPANY_FALLBACK_ID = "incantor/tagger_3_1_13"
MLD_REPO = "kiriyamaX/mld-caformer"

# production thresholds (tagger-inference service defaults) used for recall@prod
PROD_THR = {"general": 0.2, "character": 0.35, "copyright": 0.35}
# low retrieval floor so target-tag probs below the production threshold are still
# returned (needed for the cov0/cov50/cov80 tiers). prob below this reads as 0.
RETRIEVAL_THR = 0.01
COV_TIERS = (0.0, 0.5, 0.8)


def _norm(tag: str) -> str:
    """Normalize a tag to underscore-canonical danbooru form for set ops."""
    return tag.strip().lower().replace(" ", "_")


# --------------------------------------------------------------------------- #
# Company tagger (transformers pipeline; gated)
# --------------------------------------------------------------------------- #
class CompanyTagger:
    """Loads the company anime tagger via transformers.pipeline directly.

    Mirrors tagger_inference.tag.TagHandler's load (task=None, trust_remote_code,
    bf16 on cuda) but without the service wrapper. Returns per-category
    {tag: prob} dicts (probs retrieved at a low floor for the coverage tiers).
    """

    def __init__(self, model_id: str = COMPANY_MODEL_ID,
                 retrieval_threshold: float = RETRIEVAL_THR):
        import torch
        from transformers import pipeline
        self.model_id = model_id
        self.name = f"company:{model_id}"
        self.retr = float(retrieval_threshold)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.pipe = pipeline(
            task=None, model=model_id, image_processor=model_id,
            trust_remote_code=True, dtype=dtype, device=device, hf_token=token,
        )

    def tag(self, image) -> dict:
        import torch
        # low retrieval floor on every category -> we get probs for low-conf tags too
        threshold = {"general": self.retr, "character": self.retr, "copyright": self.retr}
        with torch.inference_mode():
            results = self.pipe(image, min_threshold=threshold)["results"]

        def cat(name):
            d = results.get(name, {}) or {}
            return {_norm(t): float(p) for t, p in d.items()
                    if not (name == "copyright" and t == "original")}
        return {"general": cat("general"), "character": cat("character"),
                "copyright": cat("copyright")}


# --------------------------------------------------------------------------- #
# ml-danbooru fallback (public)
# --------------------------------------------------------------------------- #
class MLDanbooruTagger:
    """Public kiriyamaX/mld-caformer via the troph-team ml-danbooru-tagger pkg.

    Emits a flat danbooru tag list (general + character mixed); no copyright
    split. Exposed as {'general': {tag: prob}, 'character': {}, 'copyright': {}}
    and hit-rate folds character/copyright targets into general at scoring time.
    """

    name = "mldanbooru:kiriyamaX/mld-caformer"
    PKG = "/fsx/evan/mot_code/troph-team_recent_t2i_related/ml-danbooru-tagger"

    def __init__(self, thr: float = 0.6, image_size: int = 448):
        import argparse as _argparse
        import sys
        if self.PKG not in sys.path:
            sys.path.insert(0, self.PKG)
        from ml_danbooru_tagger.demo_ca_in_package import Demo  # type: ignore
        args = _argparse.Namespace(
            data="", ckpt="", class_map="", model_name="caformer_m36",
            num_classes=12547, image_size=image_size, thr=thr, keep_ratio=False,
            bs=1, str_thr=0.7, use_ml_decoder=0, fp16=False, ema=False,
            frelu=True, xformers=False, decoder_embedding=384,
            num_layers_decoder=4, num_head_decoder=8, num_queries=80,
            scale_skip=1, out_type="json",
        )
        self.demo = Demo(args)

    def tag(self, image) -> dict:
        device = next(self.demo.model.parameters()).device
        t = self.demo.trans(image).to(device)
        cls_list = self.demo.infer_one(t)  # [(tag, prob), ...]
        general = {_norm(name): float(prob) for name, prob in cls_list}
        return {"general": general, "character": {}, "copyright": {}}


def load_tagger(backend: str = "auto", **kw):
    """Return a tagger with ``.tag(pil) -> {cat: {tag: prob}}``.
    backend in {auto, company, mldanbooru}. ``auto``/``company`` try the newer
    company model, then the proven 3_1_13, then (auto only) mldanbooru."""
    errs = []
    if backend in ("auto", "company"):
        for mid in (COMPANY_MODEL_ID, COMPANY_FALLBACK_ID):
            try:
                t = CompanyTagger(model_id=mid)
                if mid != COMPANY_MODEL_ID:
                    print(f"[tagger] {COMPANY_MODEL_ID} unavailable; using fallback {mid}", flush=True)
                return t
            except Exception as e:  # noqa: BLE001
                errs.append(f"company:{mid}: {type(e).__name__}: {e}")
        if backend == "company":
            raise RuntimeError("company tagger failed:\n  " + "\n  ".join(errs))
    if backend in ("auto", "mldanbooru"):
        try:
            return MLDanbooruTagger(**{k: v for k, v in kw.items()
                                       if k in ("thr", "image_size")})
        except Exception as e:  # noqa: BLE001
            errs.append(f"mldanbooru: {type(e).__name__}: {e}")
            if backend == "mldanbooru":
                raise
    raise RuntimeError("no tagger backend available:\n  " + "\n  ".join(errs))


def _cov(target: set, probs: dict, thr: float) -> float | None:
    """fraction of target tags whose predicted prob exceeds thr (>, strict)."""
    if not target:
        return None
    return sum(1 for t in target if probs.get(t, 0.0) > thr) / len(target)


def _recall_prod(target: set, probs: dict, prod_thr: float) -> float | None:
    """recall at the production threshold (>=)."""
    if not target:
        return None
    return sum(1 for t in target if probs.get(t, 0.0) >= prod_thr) / len(target)


def hit_rate(pred: dict, target: dict, fold_char_into_general: bool) -> dict:
    """Concept recall@prod + coverage tiers (cov0/cov50/cov80) per category + overall.

    ``pred`` is {cat: {tag: prob}}; ``target`` carries target_{general,character,
    copyright} lists. ``fold_char_into_general`` handles flat-vocab backends
    (mldanbooru) that don't separate categories -> look every target tag up in the
    merged prob map.
    """
    tg = {_norm(t) for t in target.get("target_general", [])}
    tc = {_norm(t) for t in target.get("target_character", [])}
    tp = {_norm(t) for t in target.get("target_copyright", [])}

    if fold_char_into_general:
        merged = {}
        for c in ("general", "character", "copyright"):
            merged.update(pred.get(c, {}))
        probs = {"general": merged, "character": merged, "copyright": merged}
    else:
        probs = {c: pred.get(c, {}) for c in ("general", "character", "copyright")}

    cats = {"general": (tg, PROD_THR["general"]), "character": (tc, PROD_THR["character"]),
            "copyright": (tp, PROD_THR["copyright"])}
    out = {}
    for c, (tset, pthr) in cats.items():
        out[f"{c}_hit"] = _recall_prod(tset, probs[c], pthr)
        for x in COV_TIERS:
            out[f"{c}_cov{int(x*100)}"] = _cov(tset, probs[c], x)

    # overall = union of all target tags, each looked up in its own category map
    all_t = [(t, "general") for t in tg] + [(t, "character") for t in tc] + \
            [(t, "copyright") for t in tp]
    n = len(all_t)
    if n:
        out["overall_hit"] = sum(1 for t, c in all_t
                                 if probs[c].get(t, 0.0) >= PROD_THR[c]) / n
        for x in COV_TIERS:
            out[f"overall_cov{int(x*100)}"] = sum(1 for t, c in all_t
                                                  if probs[c].get(t, 0.0) > x) / n
    else:
        out["overall_hit"] = None
        for x in COV_TIERS:
            out[f"overall_cov{int(x*100)}"] = None
    out["n_target"] = n
    out["n_pred"] = sum(len(probs[c]) for c in ("general", "character", "copyright"))
    return out


def load_evalset(path: str) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                out[e["id"]] = e
    return out


# fields aggregated into the summary (means over images)
_MEAN_KEYS = (
    ["general_hit", "character_hit", "copyright_hit", "overall_hit"] +
    [f"{c}_cov{int(x*100)}" for c in ("general", "character", "copyright", "overall")
     for x in COV_TIERS]
)


def score_dir(image_dir: str, evalset_path: str, backend: str = "auto",
              id_from_name=None) -> dict:
    """Score every image in image_dir against the evalset.

    Filenames must encode the eval id; pass ``id_from_name`` (str->id) or rely on
    the default ``<id>__*.png`` / ``<id>.png`` convention.
    """
    from PIL import Image
    evalset = load_evalset(evalset_path)
    tagger = load_tagger(backend=backend)
    fold = isinstance(tagger, MLDanbooruTagger)

    def default_id(name: str) -> str:
        return Path(name).stem.split("__")[0]

    idfn = id_from_name or default_id
    files = sorted(glob.glob(os.path.join(image_dir, "*.png")) +
                   glob.glob(os.path.join(image_dir, "*.jpg")) +
                   glob.glob(os.path.join(image_dir, "*.webp")))
    per_image = []
    for fp in files:
        eid = idfn(os.path.basename(fp))
        target = evalset.get(eid)
        if target is None:
            continue
        with Image.open(fp) as im:
            pred = tagger.tag(im.convert("RGB"))
        hr = hit_rate(pred, target, fold)
        per_image.append({"file": os.path.basename(fp), "id": eid,
                          "prompt": target["prompt"], "band": target.get("band"),
                          **hr})

    def mean(key):
        vals = [r[key] for r in per_image if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    summary = {"backend": tagger.name, "n_images": len(per_image),
               "metric_note": "recall@prod thr g0.2/c0.35/p0.35; cov{0,50,80}=frac target "
                              "tags with prob>{0,0.5,0.8} (naifu concept_v2 aligned)"}
    for k in _MEAN_KEYS:
        summary[f"mean_{k}"] = mean(k)
    return {"summary": summary, "per_image": per_image}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--evalset", default=str(Path(__file__).resolve().parents[1]
                                             / "prompts" / "anime_evalset.jsonl"))
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "company", "mldanbooru"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = score_dir(args.image_dir, args.evalset, backend=args.backend)
    print(json.dumps(res["summary"], indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[tagger_score] wrote {args.out}")


if __name__ == "__main__":
    main()
