#!/usr/bin/env python3
"""Read the rendered text out of generated images, so ocr_exact_string.py can score it.

The scorer compares two strings. This is the half that turns an image into one of them.
Without it the "OCR exact-string" axis of STAGE512B_EVAL_PROTOCOL.md is a comparator with
nothing to compare, which is what it was until this file existed.

READER CHOICE IS PART OF THE PROTOCOL, not an implementation detail. No dedicated OCR
engine is installed (no paddleocr, easyocr or tesseract, and the cluster is offline), but
Qwen3-VL-8B-Instruct is on disk and transcribes reliably. Once a snapshot has been scored
with it, changing the reader invalidates comparison against that snapshot exactly as
surely as changing the prompts would -- so the model path, the prompt, the decoding
settings and a config digest all go into the output header, and the scorer requires an
engine identity string.

Note for the report, not a disqualification: the model under test uses Qwen3-VL-2B as its
text encoder and the reader is Qwen3-VL-8B. Different role, different size, no circularity
-- the reader never sees the prompt, only the picture.

Decoding is greedy. A sampled reader would make the metric noisy in a way that looks like
model movement.

Extraction is deliberately minimal and auditable: the raw model output is kept alongside
the extracted string in every record, so a disagreement can be inspected rather than
guessed at. The only transformation is removing one matched pair of surrounding quotes --
because the instruction says to output the text alone and the model still sometimes quotes
it -- and stripping outer whitespace. Nothing is normalised here; normalisation belongs to
the scorer, which reports normalised and un-normalised matching separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Frozen. Editing this string changes the metric.
READER_PROMPT = (
    "Transcribe the text that appears in this image, exactly as it is written. "
    "Preserve capitalisation, punctuation, spacing and character forms. "
    "Output only the transcribed text and nothing else. "
    "If the image contains no text, output NOTEXT."
)
NO_TEXT_SENTINEL = "NOTEXT"
MAX_NEW_TOKENS = 256

_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))


def extract(raw: str) -> str:
    """Model output -> the transcribed string. Deterministic and reversible-by-eye.

    Only two transformations, and neither touches the interior of the string:
    outer whitespace, and ONE matched pair of surrounding quotes. Interior quotes are
    left alone, because a shop sign may legitimately contain them -- stripping greedily
    would silently corrupt exactly the rows this metric exists to measure.
    """
    text = (raw or "").strip()
    if text == NO_TEXT_SENTINEL:
        return ""
    for open_q, close_q in _QUOTE_PAIRS:
        if len(text) >= 2 and text.startswith(open_q) and text.endswith(close_q):
            return text[1:-1].strip()
    return text


def _dir_digest(root: Path) -> str:
    """Digest the reader's identity from its config, not its 16 GB of weights."""
    parts = []
    for name in ("config.json", "generation_config.json", "preprocessor_config.json"):
        path = root / name
        if path.is_file():
            parts.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashlib.sha256("".join(parts).encode()).hexdigest()


def load_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        for field in ("id", "image_path", "gt_text_raw"):
            if field not in record:
                raise SystemExit(
                    f"manifest record {record.get('id')!r} is missing {field!r}. "
                    "gt_text_raw must come from the GT sidecar's RAW column, not the "
                    "normalized one -- training reads raw."
                )
        rows.append(record)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path,
                    help="JSONL of {id, image_path, gt_text_raw}")
    ap.add_argument("--out", required=True, type=Path,
                    help="JSONL for ocr_exact_string.py --predictions")
    ap.add_argument("--model", type=Path, default=Path("/fsx/models/Qwen3-VL-8B-Instruct"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="diagnostic: first N rows")
    ap.add_argument("--self-test", action="store_true",
                    help="exercise extract() only; no model, no GPU")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if not args.model.is_dir():
        raise SystemExit(f"reader model not found: {args.model}")

    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    missing = [r["id"] for r in rows if not Path(r["image_path"]).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing, e.g. {missing[:5]}")

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(args.model))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model), dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    engine = f"qwen3vl-8b-instruct:{_dir_digest(args.model)[:16]}"
    print(f"reader   {args.model}\nengine   {engine}\nrows     {len(rows):,}", flush=True)

    out_rows: list[dict] = []
    for start in range(0, len(rows), args.batch_size):
        window = rows[start : start + args.batch_size]
        messages = [
            [{"role": "user", "content": [
                {"type": "image", "image": Image.open(r["image_path"]).convert("RGB")},
                {"type": "text", "text": READER_PROMPT},
            ]}]
            for r in window
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True,
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, temperature=None, top_p=None, top_k=None,
            )
        trimmed = [g[len(i):] for i, g in zip(inputs["input_ids"], generated)]
        decoded = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for record, raw in zip(window, decoded):
            out_rows.append({
                "id": record["id"],
                "gt_text_raw": record["gt_text_raw"],
                # Both kept on purpose: the extraction is then auditable rather than
                # something the reader has to be trusted about.
                "reader_raw_output": raw,
                "rendered_text": extract(raw),
            })
        if (start // args.batch_size) % 20 == 0:
            print(f"  {len(out_rows)}/{len(rows)}", flush=True)

    header = {
        "_header": True,
        "engine": engine,
        "model": str(args.model),
        "prompt": READER_PROMPT,
        "prompt_sha256": hashlib.sha256(READER_PROMPT.encode()).hexdigest(),
        "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
        "rows": len(out_rows),
    }
    with args.out.open("w") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}  ({len(out_rows):,} rows)")
    print(f"score with:\n  ocr_exact_string.py --predictions {args.out} "
          f"--ocr-engine {engine} --checkpoint <tag>")
    print("OCR_READ_IMAGES_DONE")
    return 0


def self_test() -> int:
    cases = [
        ("GRAND OPENING", "GRAND OPENING"),
        ('  "GRAND OPENING"  ', "GRAND OPENING"),
        ("“GRAND OPENING”", "GRAND OPENING"),
        ("'EXIT'", "EXIT"),
        (NO_TEXT_SENTINEL, ""),
        (f"  {NO_TEXT_SENTINEL}  ", ""),
        # Interior quotes must survive: a sign really can read Joe's "Best" Diner.
        ('Joe\'s "Best" Diner', 'Joe\'s "Best" Diner'),
        ('"Joe\'s "Best" Diner"', 'Joe\'s "Best" Diner'),
        # Full-width and circled forms must pass through untouched -- normalisation is
        # the scorer's job and conflating them here would hide the typographic gap.
        ("ＧＲＡＮＤ", "ＧＲＡＮＤ"),
        ("①②③", "①②③"),
        ("", ""),
        (None, ""),
        # A single stray quote is not a pair and must not be stripped.
        ('"unbalanced', '"unbalanced'),
        ("multi\nline text", "multi\nline text"),
    ]
    failures = []
    for raw, want in cases:
        got = extract(raw)
        if got != want:
            failures.append(f"extract({raw!r}) -> {got!r}, want {want!r}")
    print(f"{len(cases)} extraction cases, {len(failures)} failures")
    for f in failures:
        print(f"  {f}")
    if failures:
        return 1
    print("OCR_READER_SELFTEST_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
