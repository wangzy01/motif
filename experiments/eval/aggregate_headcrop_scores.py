#!/usr/bin/env python3
"""Aggregate headcrop matrix subagent scores into tables and a markdown report."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


VALID_LABELS = {
    "A_full_ok",
    "B_head_cut",
    "B_foot_cut",
    "B_side_cut",
    "C_reasonable_crop",
    "D_other_bad",
}
VALID_TARGETS = {
    "full_body",
    "full_body_multi",
    "lower_body_risk",
    "upper_body_control",
    "non_fullbody_control",
    "unknown",
}


def load_jsonl(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path}:{line_no}: invalid JSON: {e}") from e
                rows.append(obj)
    return rows


def load_expected(manifest_paths: list[str]) -> dict[str, dict]:
    expected: dict[str, dict] = {}
    for path in manifest_paths:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        size = f"{manifest.get('width')}x{manifest.get('height')}"
        for run in manifest.get("runs", []):
            for image in run.get("images", []):
                img_path = str(image["path"])
                row = {
                    "image": img_path,
                    "run": run.get("key", ""),
                    "prompt_id": image.get("id", ""),
                    "prompt_style": image.get("prompt_style", manifest.get("prompt_style", "")),
                    "domain": image.get("domain", manifest.get("domain", "")),
                    "composition_target": image.get("composition_target", ""),
                    "size": size,
                    "seed": image.get("seed"),
                    "ckpt": run.get("ckpt", ""),
                    "ckpt_step": run.get("ckpt_step"),
                    "exposure_estimate": run.get("exposure_estimate"),
                    "weights_src": run.get("weights_src", ""),
                    "manifest": path,
                }
                if img_path in expected:
                    raise SystemExit(f"duplicate expected image in manifests: {img_path}")
                expected[img_path] = row
    return expected


def validate_and_enrich_scores(
    rows: list[dict],
    expected: dict[str, dict],
    allow_missing: bool,
    allow_extra: bool,
) -> tuple[list[dict], dict]:
    if not expected:
        return rows, {
            "expected_images": 0,
            "score_rows": len(rows),
            "missing_images": [],
            "extra_scores": [],
            "duplicate_scores": [],
        }

    empty_image_rows = [i for i, r in enumerate(rows, 1) if not str(r.get("image", "")).strip()]
    if empty_image_rows:
        raise SystemExit(f"score rows missing image path: first rows={empty_image_rows[:5]}")

    seen = Counter(str(r.get("image", "")) for r in rows)
    duplicate_scores = sorted(path for path, n in seen.items() if path and n > 1)
    missing = sorted(path for path in expected if seen[path] == 0)
    extra = sorted(path for path in seen if path and path not in expected)

    if duplicate_scores:
        raise SystemExit(f"duplicate score rows for {len(duplicate_scores)} images; first={duplicate_scores[:3]}")
    if missing and not allow_missing:
        raise SystemExit(f"missing scores for {len(missing)} expected images; first={missing[:3]}")
    if extra and not allow_extra:
        raise SystemExit(f"extra score rows for {len(extra)} images outside manifests; first={extra[:3]}")

    enriched: list[dict] = []
    for row in rows:
        img = str(row.get("image", ""))
        merged = dict(expected.get(img, {}))
        merged.update(row)
        for key, value in expected.get(img, {}).items():
            if merged.get(key) in ("", None):
                merged[key] = value
        enriched.append(merged)
    return enriched, {
        "expected_images": len(expected),
        "score_rows": len(rows),
        "missing_images": missing,
        "extra_scores": extra,
        "duplicate_scores": duplicate_scores,
    }


def require_type(row: dict, key: str, typ, idx: int):
    if key not in row:
        raise SystemExit(f"score row {idx}: missing {key}")
    if not isinstance(row[key], typ):
        raise SystemExit(f"score row {idx}: {key} must be {typ}, got {type(row[key]).__name__}")


def validate_scores(rows: list[dict]):
    for idx, row in enumerate(rows, 1):
        for key in ("image", "run", "prompt_id", "prompt_style", "size"):
            require_type(row, key, str, idx)
            if not row[key]:
                raise SystemExit(f"score row {idx}: empty {key}")
        if row.get("primary_label") not in VALID_LABELS:
            raise SystemExit(f"score row {idx}: invalid primary_label {row.get('primary_label')!r}")
        for key in ("head_cut", "foot_cut", "side_cut"):
            require_type(row, key, bool, idx)
        for key in ("subject_complete_0to5", "composition_follow_0to5"):
            if not isinstance(row.get(key), (int, float)):
                raise SystemExit(f"score row {idx}: {key} must be numeric")
            if not 0 <= float(row[key]) <= 5:
                raise SystemExit(f"score row {idx}: {key} out of range 0..5: {row[key]}")
        seed = row.get("seed")
        if not isinstance(seed, int):
            raise SystemExit(f"score row {idx}: seed must be int, got {type(seed).__name__}")
        target = composition_target(row)
        if target not in VALID_TARGETS:
            raise SystemExit(f"score row {idx}: invalid composition_target {target!r}")


def parse_size(row: dict) -> tuple[int, int]:
    s = str(row.get("size", ""))
    if "x" not in s:
        return (0, 0)
    w, h = s.split("x", 1)
    return int(w), int(h)


def ar_bucket(row: dict) -> str:
    w, h = parse_size(row)
    if not w or not h:
        return "unknown"
    if w == h:
        return "square"
    if w > h:
        return "wide_16_9" if w / h > 1.65 else "landscape_3_2"
    return "tall_9_16" if h / w > 1.65 else "portrait_2_3"


def domain(row: dict) -> str:
    if row.get("domain"):
        return str(row["domain"])
    pid = str(row.get("prompt_id", ""))
    return "real" if pid.startswith("real_") else "anime"


def composition_target(row: dict) -> str:
    return str(row.get("composition_target") or "unknown")


def is_full_body_target(row: dict) -> bool:
    target = composition_target(row)
    return target in {"full_body", "full_body_multi", "lower_body_risk"}


def trunc_any(row: dict) -> bool:
    return bool(row.get("head_cut")) or bool(row.get("foot_cut")) or bool(row.get("side_cut"))


def summarize(rows: list[dict], keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        vals = []
        for k in keys:
            if k == "domain":
                vals.append(domain(r))
            elif k == "ar":
                vals.append(ar_bucket(r))
            elif k == "composition_target":
                vals.append(composition_target(r))
            else:
                vals.append(r.get(k, ""))
        groups[tuple(vals)].append(r)

    out = []
    for vals, rs in sorted(groups.items()):
        n = len(rs)
        head = sum(1 for r in rs if r.get("head_cut"))
        foot = sum(1 for r in rs if r.get("foot_cut"))
        side = sum(1 for r in rs if r.get("side_cut"))
        any_cut = sum(1 for r in rs if trunc_any(r))
        full_ok = sum(1 for r in rs if r.get("primary_label") == "A_full_ok")
        primary_trunc = sum(1 for r in rs if str(r.get("primary_label", "")).startswith("B_"))
        other_bad = sum(1 for r in rs if r.get("primary_label") == "D_other_bad")
        full_targets = [r for r in rs if is_full_body_target(r)]
        full_target_n = len(full_targets)
        full_target_trunc = sum(1 for r in full_targets if trunc_any(r))
        full_target_ok = sum(1 for r in full_targets if r.get("primary_label") == "A_full_ok")
        complete = sum(float(r.get("subject_complete_0to5", 0)) for r in rs) / n
        follow = sum(float(r.get("composition_follow_0to5", 0)) for r in rs) / n
        row = {k: v for k, v in zip(keys, vals)}
        row.update({
            "n": n,
            "trunc_any": any_cut,
            "trunc_any_rate": any_cut / n,
            "full_ok": full_ok,
            "full_ok_rate": full_ok / n,
            "primary_trunc": primary_trunc,
            "primary_trunc_rate": primary_trunc / n,
            "other_bad": other_bad,
            "other_bad_rate": other_bad / n,
            "full_body_target_n": full_target_n,
            "full_body_target_trunc_rate": (full_target_trunc / full_target_n) if full_target_n else None,
            "full_body_target_ok_rate": (full_target_ok / full_target_n) if full_target_n else None,
            "head_cut": head,
            "head_cut_rate": head / n,
            "foot_cut": foot,
            "foot_cut_rate": foot / n,
            "side_cut": side,
            "side_cut_rate": side / n,
            "complete_mean": complete,
            "follow_mean": follow,
        })
        out.append(row)
    return out


def md_table(rows: list[dict], cols: list[str]) -> str:
    def fmt(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--manifests", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-coverage-check", action="store_true")
    ap.add_argument("--allow-missing-scores", action="store_true")
    ap.add_argument("--allow-extra-scores", action="store_true")
    args = ap.parse_args()

    if not args.manifests and not args.no_coverage_check:
        raise SystemExit("pass --manifests for coverage verification, or explicit --no-coverage-check")
    rows = load_jsonl(args.scores)
    expected = load_expected(args.manifests)
    rows, coverage = validate_and_enrich_scores(
        rows,
        expected,
        allow_missing=args.allow_missing_scores,
        allow_extra=args.allow_extra_scores,
    )
    validate_scores(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "by_run": summarize(rows, ["run"]),
        "by_run_domain": summarize(rows, ["run", "domain"]),
        "by_run_prompt_style": summarize(rows, ["run", "prompt_style"]),
        "by_run_ar": summarize(rows, ["run", "ar"]),
        "by_run_seed": summarize(rows, ["run", "seed"]),
        "by_run_target": summarize(rows, ["run", "composition_target"]),
        "by_run_domain_style_ar": summarize(rows, ["run", "domain", "prompt_style", "ar"]),
    }

    with open(out_dir / "headcrop_score_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_scores": len(rows),
                "coverage": coverage,
                "tables": tables,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    cols = [
        "run", "domain", "prompt_style", "ar", "n", "trunc_any_rate",
        "head_cut_rate", "foot_cut_rate", "full_body_target_trunc_rate",
        "complete_mean", "follow_mean",
    ]
    report = [
        "# Headcrop Matrix Score Summary",
        "",
        f"Score rows: {len(rows)}",
        f"Expected images: {coverage['expected_images']}",
        f"Missing scores: {len(coverage['missing_images'])}",
        f"Extra scores: {len(coverage['extra_scores'])}",
        "",
        "## By Run",
        md_table(tables["by_run"], ["run", "n", "trunc_any_rate", "head_cut_rate", "foot_cut_rate", "full_ok_rate", "full_body_target_trunc_rate", "complete_mean", "follow_mean"]),
        "",
        "## By Run / Domain",
        md_table(tables["by_run_domain"], ["run", "domain", "n", "trunc_any_rate", "head_cut_rate", "foot_cut_rate", "full_body_target_trunc_rate", "complete_mean", "follow_mean"]),
        "",
        "## By Run / Prompt Style",
        md_table(tables["by_run_prompt_style"], ["run", "prompt_style", "n", "trunc_any_rate", "head_cut_rate", "foot_cut_rate", "full_body_target_trunc_rate", "complete_mean", "follow_mean"]),
        "",
        "## By Run / Aspect Ratio",
        md_table(tables["by_run_ar"], ["run", "ar", "n", "trunc_any_rate", "head_cut_rate", "foot_cut_rate", "side_cut_rate", "complete_mean", "follow_mean"]),
        "",
        "## By Run / Seed",
        md_table(tables["by_run_seed"], ["run", "seed", "n", "trunc_any_rate", "head_cut_rate", "foot_cut_rate", "complete_mean", "follow_mean"]),
        "",
        "## By Run / Composition Target",
        md_table(tables["by_run_target"], ["run", "composition_target", "n", "trunc_any_rate", "head_cut_rate", "foot_cut_rate", "full_ok_rate", "complete_mean", "follow_mean"]),
        "",
        "## Detailed Slice",
        md_table(tables["by_run_domain_style_ar"], cols),
        "",
    ]
    with open(out_dir / "HEAD_CROP_SCORE_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"wrote {out_dir / 'HEAD_CROP_SCORE_SUMMARY.md'}")


if __name__ == "__main__":
    main()
