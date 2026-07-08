#!/usr/bin/env python3
"""Aggregate the fixed-sigma MSE per-run JSONs + aesthetic JSON into one comparison table (md)."""
from __future__ import annotations
import argparse, glob, json, os

LABELS = [("v1", "v1_baseline", "v1 baseline"),
          ("b1", "b1_adapter", "B1 text-adapter"),
          ("b2", "b2_longskip", "B2 long-skip"),
          ("b3", "b3_nonoise", "B3 no-noise")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fs = {}
    for f in glob.glob(os.path.join(args.dir, "fixedsigma_*.json")):
        d = json.load(open(f))
        fs[d.get("shift_label", os.path.basename(f))] = d
    aes_path = os.path.join(args.dir, "aesthetic.json")
    aes = json.load(open(aes_path))["runs"] if os.path.exists(aes_path) else {}

    # collect the union of sigma keys
    sig_keys = []
    for d in fs.values():
        for k in d.get("loss_by_sigma", {}):
            if k not in sig_keys:
                sig_keys.append(k)
    sig_keys = sorted(sig_keys, key=float)

    lines = []
    lines.append("# 量化对比表 — fixed-sigma MSE + 美学分（2026-06-22, ~110-115k, EMA 权重）\n")
    lines.append("> fixed-sigma MSE：同一 eval batch + 同噪声 + 固定 sigma 网格上的 v-pred MSE（去 sigma confound，越低越好；相对比较有效）。")
    lines.append("> 美学分：本地 aes-pixai siglip 回归，对已出样图打分（越高越'好看'；不等于提示词跟随）。\n")

    # main table: mean MSE over grid + aesthetic
    lines.append("## 汇总")
    lines.append("| 版本 | ckpt step | weights | mean fixed-σ MSE ↓ | 美学 mean ↑ (n) |")
    lines.append("|---|---|---|---|---|")
    for key, run, name in LABELS:
        d = fs.get(key) or fs.get(run) or {}
        mse = d.get("mean_loss_over_grid", "—")
        step = d.get("ckpt_step", "—")
        w = d.get("weights", "—")
        a = aes.get(run, {})
        am = f"{a.get('mean','—')} (n={a.get('n','—')})" if a else "—"
        lines.append(f"| {name} | {step} | {w} | {mse} | {am} |")

    # per-sigma breakdown
    if sig_keys:
        lines.append("\n## 各 sigma 的 MSE（越低越好）")
        lines.append("| 版本 | " + " | ".join(f"σ={s}" for s in sig_keys) + " |")
        lines.append("|---" * (len(sig_keys) + 1) + "|")
        for key, run, name in LABELS:
            d = fs.get(key) or {}
            lb = d.get("loss_by_sigma", {})
            row = " | ".join(str(lb.get(s, "—")) for s in sig_keys)
            lines.append(f"| {name} | {row} |")

    lines.append("\n## 读法")
    lines.append("- mean fixed-σ MSE 是当前最干净的标量 go/no-go：差异越小越说明四版接近；某版在多个 σ 上一致更低=该消融弱占优。")
    lines.append("- ⚠️ B3 关了 noise_condition，其 MSE 曲线形状会与其余三版不同（不随 σ 条件变化），跨 B3 比要谨慎。")
    lines.append("- 美学分只衡量'好看'，B1 的卖点(提示词跟随)要看 B1 专项评测，不在本表。")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("[aggregate] ->", args.out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
