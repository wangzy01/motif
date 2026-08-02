#!/usr/bin/env python3
"""Tests for the TextAtlas GT sidecar: the runtime reader and the prompt composition.

The property that matters is narrow and worth stating plainly: **the ground truth is
either whole in the prompt or absent from it.**  A truncated exact string is not a
degraded label, it is a wrong one -- it teaches the renderer to draw words the picture
does not contain, which is the exact failure that disqualified the Qwen captions as GT
in the first place.

Real tokenizer throughout.  A stub tokenizer would let the budget arithmetic drift
away from what ``encode_text`` actually does, and the budget is the whole mechanism.

New tests live under scripts/ rather than tests/: tests/ carries an exact-two closure
check that a third file would break.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import shutil
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.dataline.caption_build import (  # noqa: E402
    TEXTATLAS_GT_TEMPLATES,
    compose_textatlas_gt_prompt,
    truncate_to_token_budget,
)
from training.dataline_offline.textatlas_gt_v1 import (  # noqa: E402
    SCHEMA,
    TextAtlasGroundTruth,
    TextAtlasGroundTruthError,
)

TOKENIZER_PATH = "/fsx/models/Qwen3-VL-2B-Instruct"
BUDGET = 471  # mtml_for("512b") 512 - template_reserve 41
CATALOG_HASH = "a" * 64
CATALOG_HEADER_SHA = "b" * 64

CHECKS = 0
FAILURES: list[str] = []


def check(condition, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        print(f"  FAIL  {label}")


def _starts_with_clause(prompt: str) -> bool:
    return any(prompt.startswith(t.split("{", 1)[0]) for t in TEXTATLAS_GT_TEMPLATES)


# --------------------------------------------------------------------------- #
# prompt composition
# --------------------------------------------------------------------------- #
def test_composition(tokenizer) -> None:
    print("prompt composition")
    ntok = lambda s: len(tokenizer(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    caption = ("A storefront window with a large promotional sign. " * 40).strip()
    long_caption = ("A very detailed description of the scene. " * 200).strip()

    gt = "GRAND OPENING - 50% OFF"
    prompt = compose_textatlas_gt_prompt(gt, caption, tokenizer, BUDGET, random.Random(0))
    check(_starts_with_clause(prompt), "GT clause leads the prompt")
    check(gt in prompt, "GT survives verbatim")
    check(ntok(prompt) <= BUDGET, "within budget")
    check(caption.split(".")[0] in prompt, "caption still present")

    # No GT: byte-identical to the pre-GT behaviour.  This is what makes turning the
    # sidecar on a change only to the covered rows.
    for text in ("", None):
        check(
            compose_textatlas_gt_prompt(text, caption, tokenizer, BUDGET, random.Random(0))
            == truncate_to_token_budget(caption, tokenizer, BUDGET),
            f"no GT ({text!r}) falls through to plain truncation",
        )

    # The rng must not be touched when there is no GT, or enabling the sidecar would
    # desynchronise the prompt stream for uncovered rows too.
    untouched, reference = random.Random(7), random.Random(7)
    compose_textatlas_gt_prompt("", caption, tokenizer, BUDGET, untouched)
    check(untouched.random() == reference.random(), "no-GT path consumes no rng draw")

    # Over-budget GT: fall back, never emit a cut GT.
    huge = "Lorem ipsum dolor sit amet consectetur adipiscing elit " * 200
    prompt = compose_textatlas_gt_prompt(huge, caption, tokenizer, BUDGET, random.Random(0))
    check(not _starts_with_clause(prompt), "over-budget GT is dropped, not truncated")
    check(
        prompt == truncate_to_token_budget(caption, tokenizer, BUDGET),
        "over-budget GT falls back to exactly the plain caption",
    )

    # Sweep the whole fit/fallback transition.  This is the test that would have
    # caught a budget off-by-one that only shows up on the few thousand longest rows.
    partial = whole = 0
    sizes = list(range(1, 620, 3))
    for k in sizes:
        words = " ".join(f"w{i}" for i in range(k))
        out = compose_textatlas_gt_prompt(words, long_caption, tokenizer, BUDGET, random.Random(k))
        check(ntok(out) <= BUDGET, f"budget respected at gt_words={k}")
        if _starts_with_clause(out):
            whole += 1
            if words not in out:
                partial += 1
    check(partial == 0, "no size produces a partially present GT")
    check(whole > 0, "the sweep actually exercised the GT-present branch")
    print(f"  sweep: {len(sizes)} sizes, {whole} kept the GT, {len(sizes) - whole} fell back")

    # Empty caption -> clause alone, still a valid prompt.
    only = compose_textatlas_gt_prompt(gt, "", tokenizer, BUDGET, random.Random(3))
    check(only.endswith(f'"{gt}".'), "empty caption yields the clause alone")

    # Every template must actually be reachable, or the variation is decorative.
    seen = {
        compose_textatlas_gt_prompt(gt, "", tokenizer, BUDGET, random.Random(s))
        for s in range(60)
    }
    check(len(seen) == len(TEXTATLAS_GT_TEMPLATES), "all templates are reachable")

    # No tokenizer: degrade to concatenation, do not crash and do not silently drop GT.
    plain = compose_textatlas_gt_prompt(gt, caption, None, BUDGET, random.Random(0))
    check(gt in plain, "tokenizer=None still carries the GT")


# --------------------------------------------------------------------------- #
# runtime reader
# --------------------------------------------------------------------------- #
def _write_sidecar(root: Path, rows, *, schema=SCHEMA, catalog_hash=CATALOG_HASH,
                   catalog_header_sha=CATALOG_HEADER_SHA, corrupt_digest=False):
    root.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "row_id": pa.array([r[0] for r in rows], type=pa.int64()),
            "gt_text_raw": pa.array([r[1] for r in rows], type=pa.string()),
            "gt_text_normalized": pa.array(
                [unicodedata.normalize("NFKC", r[1]) for r in rows], type=pa.string()
            ),
            "gt_source": pa.array(["raw_text_column"] * len(rows), type=pa.string()),
            "gt_tokens_raw": pa.array([0] * len(rows), type=pa.int32()),
            "gt_clause_tokens_max": pa.array([0] * len(rows), type=pa.int32()),
        }
    )
    path = root / "textatlas_gt.parquet"
    pq.write_table(table, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    recorded = ("0" * 64) if corrupt_digest else digest
    (root / "header.json").write_text(
        json.dumps(
            {
                "schema": schema,
                "catalog_hash": catalog_hash,
                "catalog_header_sha256": catalog_header_sha,
                "emitted_rows": len(rows),
                "output": {"file": path.name, "sha256": recorded, "rows": len(rows)},
            }
        )
    )
    (root / "_SUCCESS").write_text(recorded + "\n")
    return root


def test_reader(tmp: Path) -> None:
    print("runtime reader")
    rows = [(10, "ALPHA"), (20, "BETA"), (30, "ＧＲＡＮＤ")]
    good = _write_sidecar(tmp / "good", rows)
    gt = TextAtlasGroundTruth(good, CATALOG_HASH, CATALOG_HEADER_SHA)
    check(gt.rows == 3, "row count")
    check(gt.get(10) == "ALPHA", "hit at the first row")
    check(gt.get(30) == rows[2][1], "full-width characters survive the round trip")
    check(gt.get(30) != unicodedata.normalize("NFKC", rows[2][1]),
          "raw column is genuinely un-normalised")
    check(gt.get(15) is None, "miss between two present ids")
    check(gt.get(5) is None and gt.get(99) is None, "miss outside the range")
    check(gt.coverage()["hits"] == 3 and gt.coverage()["misses"] == 3, "coverage counters")

    cases = [
        ("v1 schema is refused by name",
         _write_sidecar(tmp / "v1", rows, schema="stage512b_textatlas_gt_sidecar_v1")),
        ("catalog_hash mismatch is refused",
         _write_sidecar(tmp / "othercat", rows, catalog_hash="c" * 64)),
        ("catalog_header_sha256 mismatch is refused",
         _write_sidecar(tmp / "otherhdr", rows, catalog_header_sha="d" * 64)),
        ("digest mismatch is refused",
         _write_sidecar(tmp / "corrupt", rows, corrupt_digest=True)),
    ]
    for label, root in cases:
        try:
            TextAtlasGroundTruth(root, CATALOG_HASH, CATALOG_HEADER_SHA)
            check(False, label)
        except TextAtlasGroundTruthError:
            check(True, label)

    # A truncated parquet must not read as a short-but-valid sidecar.
    truncated = _write_sidecar(tmp / "trunc", rows)
    target = truncated / "textatlas_gt.parquet"
    data = target.read_bytes()
    target.write_bytes(data[: len(data) // 2])
    try:
        TextAtlasGroundTruth(truncated, CATALOG_HASH, CATALOG_HEADER_SHA)
        check(False, "truncated parquet is refused")
    except TextAtlasGroundTruthError:
        check(True, "truncated parquet is refused")

    # Unsorted ids would make searchsorted return the wrong string silently -- the
    # worst possible failure for this artifact, so it has to be structural.
    unsorted = _write_sidecar(tmp / "unsorted", [(30, "C"), (10, "A"), (20, "B")])
    try:
        TextAtlasGroundTruth(unsorted, CATALOG_HASH, CATALOG_HEADER_SHA)
        check(False, "unsorted row_id is refused")
    except TextAtlasGroundTruthError:
        check(True, "unsorted row_id is refused")

    dupes = _write_sidecar(tmp / "dupes", [(10, "A"), (10, "B"), (20, "C")])
    try:
        TextAtlasGroundTruth(dupes, CATALOG_HASH, CATALOG_HEADER_SHA)
        check(False, "duplicate row_id is refused")
    except TextAtlasGroundTruthError:
        check(True, "duplicate row_id is refused")

    for missing in ("header.json", "textatlas_gt.parquet", "_SUCCESS"):
        root = _write_sidecar(tmp / f"no_{missing}", rows)
        (root / missing).unlink()
        try:
            TextAtlasGroundTruth(root, CATALOG_HASH, CATALOG_HEADER_SHA)
            check(False, f"missing {missing} is refused")
        except TextAtlasGroundTruthError:
            check(True, f"missing {missing} is refused")

    empty = _write_sidecar(tmp / "empty", [])
    reader = TextAtlasGroundTruth(empty, CATALOG_HASH, CATALOG_HEADER_SHA)
    check(reader.get(1) is None and reader.rows == 0, "empty sidecar reads as all-miss")


def main() -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tmp = Path(tempfile.mkdtemp(prefix="textatlas_gt_test_"))
    try:
        test_composition(tokenizer)
        test_reader(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{CHECKS} checks, {len(FAILURES)} failures")
    for failure in FAILURES:
        print(f"  {failure}")
    if FAILURES:
        return 1
    print("TEXTATLAS_GT_TESTS_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
