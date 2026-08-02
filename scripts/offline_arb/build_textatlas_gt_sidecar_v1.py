#!/usr/bin/env python3
"""Extract TextAtlas's AUTHORITATIVE rendered-text ground truth, where one exists.

Scope is deliberately narrow: only the two subsets whose exact string is a RECORD
rather than a transcription.

  CleanTextSynth   ``annotation`` is the generation parameter -- the image was
                   rendered FROM this string, so it is ground truth by construction.
                   Format: "... displaying the text: <GT>".
  TextScenesHQ     carries a dedicated ``raw_text`` column.

Everything else is excluded, and the reason is measured, not assumed.

Why not the Qwen (i1) captions, which look like the better source because they quote
the rendered text uniformly across all nine subsets (93-98% of caption1, 97-100%
across all five)?  Because a quote is a transcription.  Scored against
CleanTextSynth's authoritative annotation on 200 joined rows: exactly equal 73.5%,
>=0.90 similar 9.0%, wrong or unquoted 17.5%.  The failures are not noise -- for a
long string Qwen quotes a FRAGMENT, and sometimes an unrelated span entirely (GT
"The Cyclones represent Iowa State University in the NCAA's Big 12 Conference..."
against a quoted "modern era", similarity 0.028).  Feeding that to a text renderer
would teach it to draw words the image does not contain, on roughly a quarter of
rows.  The Qwen captions remain the caption source; they are not ground truth.

Why not ``annotation`` for the rest: PPT2Details is free-form prose describing a
slide, PPT2Structured and TextVisionBlend are bounding-box layout specs, and no
exact string is recoverable from any of them (0% marker hit over 300 sampled rows
each).  LongWordsSubset-A/M interleave a phrase with a comma-separated word list and
resisted a per-subset pattern (17%/16%).  Paper2Text extracts cleanly but is
whole-page academic text: median 2,376 characters, 65% over 2,000, and those pages
are illegible at 512 anyway.

Output is a standalone sidecar keyed by global catalog ``row_id``.  It does not
touch the caption sidecar: the Qwen rewrite must not be able to overwrite a
recorded ground truth, and keeping them in separate artifacts is what makes that
structural rather than a convention.

--------------------------------------------------------------------------------
v2 (2026-08-02).  v1 shipped a single ``gt_text`` that had already been through
NFKC and whitespace folding, and its length statistics were in characters.  Both
were wrong for a sidecar whose whole purpose is "the exact string".

* NFKC rewrites full-width to half-width, folds circled and squared characters,
  and normalises many mathematical alphanumerics.  For a text RENDERER those are
  not equivalent forms -- they are different glyphs on the page.  Whitespace
  folding additionally destroys line layout.  v2 emits ``gt_text_raw`` (only
  stripped at the ends) alongside ``gt_text_normalized``; the loader trains on
  raw, and normalized exists for matching and for OCR scoring.
* Character counts do not answer "will this truncate": the model tokenises with
  Qwen3-VL-2B and hard-truncates at ``text_max_length``.  v2 measures with that
  tokenizer and reports over-budget counts against the real caption budget.

The v1 artifact (``textatlas_gt.parquet``, sha256 2f6bb2bb..., 1,941,791 rows)
stays where it is; this script's v1 form is in git history at commit 3f17d46.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.dataline import caption_build  # noqa: E402

SCHEMA = "stage512b_textatlas_gt_sidecar_v2"
OUT_SCHEMA = pa.schema(
    [
        pa.field("row_id", pa.int64(), nullable=False),
        # What the image actually renders.  This is the training target.
        pa.field("gt_text_raw", pa.string(), nullable=False),
        # NFKC + whitespace collapse.  For matching and OCR scoring, not for training.
        pa.field("gt_text_normalized", pa.string(), nullable=False),
        pa.field("gt_source", pa.string(), nullable=False),
        # Real Qwen3-VL-2B tokens for gt_text_raw, and for the longest of the three
        # prompt clauses that wrap it -- the latter is what decides at runtime whether
        # the row gets its GT or falls back to the plain caption.
        pa.field("gt_tokens_raw", pa.int32(), nullable=False),
        pa.field("gt_clause_tokens_max", pa.int32(), nullable=False),
    ]
)

# "... displaying the text: <GT>" -- the trailing text runs to the end of the field.
_CLEAN_TEXT_SYNTH = re.compile(r"displaying the text:\s*(.+)\Z", re.S)

SUPPORTED = {
    "CleanTextSynth": "annotation_displaying_the_text",
    "TextScenesHQ": "raw_text_column",
}


def _normalize(text: str) -> str:
    """NFKC and whitespace only.  Case and punctuation are PART of the target."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def _dist(values: np.ndarray) -> dict:
    """min/p50/p90/p99/p99.9/p99.99/max plus over-threshold COUNTS.

    Percentiles alone hide the tail's size: the first build read p99 477 next to
    max 72,019, which says nothing about whether that tail is one row or a hundred
    thousand.  Counts answer the question percentiles only gesture at.
    """
    if not values.size:
        return {k: None for k in ("min", "p50", "p90", "p99", "p99_9", "p99_99", "max")}
    return {
        "min": int(values.min()),
        "p50": int(np.percentile(values, 50)),
        "p90": int(np.percentile(values, 90)),
        "p99": int(np.percentile(values, 99)),
        "p99_9": int(np.percentile(values, 99.9)),
        "p99_99": int(np.percentile(values, 99.99)),
        "max": int(values.max()),
        "over": {str(t): int((values > t).sum()) for t in (256, 471, 512, 1024)},
    }


def _token_counts(tokenizer, texts: list[str], chunk: int = 4096) -> np.ndarray:
    """Token count per string, ``add_special_tokens=False`` to match the dataline.

    ``truncate_to_token_budget`` calls the tokenizer that way, so anything measured
    here has to as well or the budget arithmetic is off by the special tokens.
    """
    if not texts:
        return np.zeros(0, dtype=np.int64)
    out = np.empty(len(texts), dtype=np.int64)
    for start in range(0, len(texts), chunk):
        window = texts[start : start + chunk]
        ids = tokenizer(window, add_special_tokens=False)["input_ids"]
        out[start : start + len(window)] = [len(x) for x in ids]
    return out


def extract(subset: str, annotation: str, raw_text: str | None) -> tuple[str, str] | None:
    """Return (raw, normalized), or None when this row carries no authoritative string.

    ``raw`` is stripped at the ends only.  Everything inside -- full-width forms,
    circled characters, internal newlines, runs of spaces -- is kept, because for a
    renderer those are the picture.
    """
    if subset == "TextScenesHQ":
        raw = str(raw_text or "").strip()
    elif subset == "CleanTextSynth":
        match = _CLEAN_TEXT_SYNTH.search(str(annotation or ""))
        if not match:
            return None
        raw = match.group(1).strip()
    else:
        return None
    if not raw:
        return None
    normalized = _normalize(raw)
    return (raw, normalized) if normalized else None


def build(
    catalog_root: Path,
    output_root: Path,
    batch_rows: int,
    tokenizer_path: Path,
    caption_budget: int,
) -> dict:
    catalog = catalog_root / "textatlas.parquet"
    reader = pq.ParquetFile(catalog)

    # Group catalog rows by their recorded source position so every source row group
    # is decoded exactly once.  The catalog stores exact locators, so there is no uid
    # join and therefore no chance of a silent mis-pairing.
    wanted: dict[tuple[str, int], list[tuple[int, int, str]]] = collections.defaultdict(list)
    seen_subsets: collections.Counter = collections.Counter()
    total_rows = 0
    for batch in reader.iter_batches(
        batch_size=batch_rows,
        columns=["row_id", "uid", "image_parquet_path", "image_row_group", "image_row_in_group"],
    ):
        rows = batch.to_pydict()
        for row_id, uid, path, group, offset in zip(
            rows["row_id"], rows["uid"], rows["image_parquet_path"],
            rows["image_row_group"], rows["image_row_in_group"],
        ):
            total_rows += 1
            subset = str(uid).split(":", 1)[0]
            seen_subsets[subset] += 1
            if subset in SUPPORTED:
                wanted[(str(path), int(group))].append((int(row_id), int(offset), subset))

    eligible = sum(len(v) for v in wanted.values())
    print(f"catalog textatlas rows {total_rows:,}; in-scope {eligible:,}", flush=True)
    print(f"subset spread: {dict(seen_subsets.most_common())}", flush=True)

    out_rows: list[tuple[int, str, str, str]] = []
    missing: collections.Counter = collections.Counter()
    for (path, group), entries in sorted(wanted.items()):
        columns = ["annotation"]
        names = set(pq.ParquetFile(path).schema_arrow.names)
        if "raw_text" in names:
            columns.append("raw_text")
        table = pq.ParquetFile(path).read_row_group(group, columns=columns)
        ann = table["annotation"].to_pylist()
        raw = table["raw_text"].to_pylist() if "raw_text" in columns else [None] * len(ann)
        for row_id, offset, subset in entries:
            if offset >= len(ann):
                # A locator past the end means the catalog and the source disagree;
                # that is a binding failure, not a missing string.
                raise SystemExit(
                    f"catalog locator out of range: {path} rg={group} offset={offset} "
                    f"has {len(ann)} rows"
                )
            value = extract(subset, ann[offset], raw[offset])
            if value is None:
                missing[subset] += 1
                continue
            out_rows.append((row_id, value[0], value[1], SUPPORTED[subset]))

    out_rows.sort(key=lambda r: r[0])

    # Tokenise with the SAME tokenizer the model uses.  Batched, because 1.9M calls
    # one at a time is the difference between minutes and an hour.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    raw_texts = [r[1] for r in out_rows]
    tok_raw = _token_counts(tokenizer, raw_texts)
    # The clause is what actually reaches the prompt; the longest template is the
    # worst case, and worst case is what decides the fallback.
    clause_counts = [
        _token_counts(tokenizer, [tpl.format(gt=t) for t in raw_texts])
        for tpl in caption_build.TEXTATLAS_GT_TEMPLATES
    ]
    tok_clause_max = np.max(np.stack(clause_counts), axis=0) if raw_texts else np.zeros(0, np.int64)

    lengths = (
        np.array([len(t) for t in raw_texts], dtype=np.int64) if out_rows else np.zeros(0, np.int64)
    )

    output_root.mkdir(parents=True)
    table = pa.Table.from_arrays(
        [
            pa.array([r[0] for r in out_rows], type=pa.int64()),
            pa.array([r[1] for r in out_rows], type=pa.string()),
            pa.array([r[2] for r in out_rows], type=pa.string()),
            pa.array([r[3] for r in out_rows], type=pa.string()),
            pa.array(tok_raw.astype(np.int32), type=pa.int32()),
            pa.array(tok_clause_max.astype(np.int32), type=pa.int32()),
        ],
        schema=OUT_SCHEMA,
    )
    parquet_path = output_root / "textatlas_gt.parquet"
    pq.write_table(table, parquet_path, row_group_size=65_536, compression="zstd")

    digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    by_source = collections.Counter(r[3] for r in out_rows)
    differs = int(sum(1 for r in out_rows if r[1] != r[2]))
    header = {
        "schema": SCHEMA,
        "catalog_root": str(catalog_root),
        "catalog_hash": json.loads((catalog_root / "header.json").read_text())["catalog_hash"],
        "catalog_header_sha256": hashlib.sha256(
            (catalog_root / "header.json").read_bytes()
        ).hexdigest(),
        "row_order": "global_catalog_row_id",
        "supported_subsets": SUPPORTED,
        "catalog_textatlas_rows": total_rows,
        "in_scope_rows": eligible,
        "emitted_rows": len(out_rows),
        "missing_by_subset": dict(missing),
        "rows_by_source": dict(by_source),
        "subset_spread": dict(seen_subsets),
        # min/p50/p99/max alone hides the shape of the tail: the first build read
        # p99 477 next to max 72,019, which is unreadable without knowing whether the
        # tail is one row or a hundred thousand.  It is a handful -- some
        # CleanTextSynth images render arXiv-style table dumps -- but the header has
        # to say so.  Over-threshold COUNTS are the field that actually answers
        # "will this truncate at tokenization".
        "gt_length_chars": {
            "min": int(lengths.min()) if lengths.size else None,
            "p50": int(np.percentile(lengths, 50)) if lengths.size else None,
            "p90": int(np.percentile(lengths, 90)) if lengths.size else None,
            "p99": int(np.percentile(lengths, 99)) if lengths.size else None,
            "p99_9": int(np.percentile(lengths, 99.9)) if lengths.size else None,
            "p99_99": int(np.percentile(lengths, 99.99)) if lengths.size else None,
            "max": int(lengths.max()) if lengths.size else None,
            "over": {
                str(t): int((lengths > t).sum()) for t in (512, 1000, 2000, 8000)
            },
        },
        # The number that justifies carrying two columns.  If normalization changed
        # nothing, gt_text_raw would be dead weight; if it changed a lot, v1 was
        # training on a rewritten target.
        "rows_where_raw_differs_from_normalized": differs,
        # Measured with Qwen3-VL-2B, add_special_tokens=False, the same call
        # truncate_to_token_budget makes.  ``over_budget`` counts the rows that will
        # FALL BACK to the plain caption at runtime because even the wrapped clause
        # does not fit -- those rows get no GT, by design, rather than a cut one.
        "gt_length_tokens": {
            "tokenizer": str(tokenizer_path),
            "caption_budget": caption_budget,
            "raw": _dist(tok_raw),
            "clause_max": _dist(tok_clause_max),
            "clause_overhead_tokens_by_template": {
                tpl: int(_token_counts(tokenizer, [tpl.format(gt="")])[0])
                for tpl in caption_build.TEXTATLAS_GT_TEMPLATES
            },
            "over_budget": int((tok_clause_max > caption_budget).sum()),
            "over_budget_frac": (
                float((tok_clause_max > caption_budget).mean()) if tok_clause_max.size else 0.0
            ),
        },
        "output": {"file": parquet_path.name, "sha256": digest, "rows": len(out_rows)},
        "excluded_subsets_reason": {
            "PPT2Details": "free-form prose description; no exact string recoverable",
            "PPT2Structured": "bounding-box layout spec; no exact string recoverable",
            "TextVisionBlend": "bounding-box layout spec; no exact string recoverable",
            "LongWordsSubset-A": "phrase plus comma word-list; no reliable pattern (17%)",
            "LongWordsSubset-M": "phrase plus comma word-list; no reliable pattern (16%)",
            "StyledTextSynth": "marker present (98%) but nested/duplicated spans; unverified",
            "Paper2Text": "extracts cleanly but is whole-page text (p50 2,376 chars)",
        },
        "not_ground_truth_note": (
            "The Qwen (i1) captions quote the rendered text across all nine subsets, but "
            "scored against CleanTextSynth's authoritative annotation they are exactly "
            "equal on only 73.5% of rows (9.0% within 0.90 similarity, 17.5% wrong or "
            "unquoted). They are a transcription, not a record, and are deliberately not "
            "used here."
        ),
    }
    (output_root / "header.json").write_text(json.dumps(header, indent=1, sort_keys=True))
    (output_root / "_SUCCESS").write_text(digest + "\n")
    for member in output_root.iterdir():
        member.chmod(0o444)
    output_root.chmod(0o555)
    return header


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--batch-rows", type=int, default=262_144)
    ap.add_argument("--tokenizer-path", required=True, type=Path,
                    help="must be the tokenizer the model uses, or the token counts are fiction")
    # Not a default worth guessing at: 512b's budget is mtml_for('512b')=512 minus
    # template_reserve=41.  Passing it explicitly keeps this artifact honest if the
    # model's text_max_length ever changes.
    ap.add_argument("--caption-budget", required=True, type=int)
    args = ap.parse_args()

    if args.output_root.exists() or args.output_root.is_symlink():
        raise SystemExit(f"refusing to overwrite an existing output root: {args.output_root}")
    if not args.tokenizer_path.is_dir():
        raise SystemExit(f"tokenizer path is not a directory: {args.tokenizer_path}")

    header = build(args.catalog_root, args.output_root, args.batch_rows,
                   args.tokenizer_path, args.caption_budget)
    for key in ("emitted_rows", "in_scope_rows", "rows_by_source", "missing_by_subset",
                "rows_where_raw_differs_from_normalized", "gt_length_chars",
                "gt_length_tokens"):
        print(f"  {key:<20} {json.dumps(header[key])}")
    print(f"  sha256               {header['output']['sha256']}")
    print("TEXTATLAS_GT_SIDECAR_BUILD_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
