#!/usr/bin/env python3
"""Loss-trend / marginal-return analysis for the 4 CURRENTLY-RUNNING stage256 runs.

Question this answers: past ~100k steps, are the runs still improving, and what is
the marginal effect (diminishing returns)? Per-run window slopes + plateau test +
projection to run end, on BOTH step and sample axes.

CPU-only, read-only on /fsx metrics.jsonl. Writes small JSON + PNGs to results dir.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CK = "/fsx/evan/checkpoints/mot_pilot_v1"
OUT = "/fsx/evan/mot_code/mot_pilot_v1_cleanup/experiments/results/loss_trend_100k"
os.makedirs(OUT, exist_ok=True)

# (key, ckpt_dir, label, global_batch, max_steps, color)
RUNS = [
    ("b1",       "s256_1x8_b1adapter_20260618",   "B1 adapter (v1: g512/400k)",      512,  400000, "#d62728"),
    ("ab1causal","s256v2_2x8_ab1causal_20260623", "ab1 causal (v2: g1536/133k)",     1536, 133000, "#1f77b4"),
    ("ab2fusion","s256v2_2x8_ab2fusion_20260623", "ab2 fusion (v2: g1536/133k)",     1536, 133000, "#2ca02c"),
    ("ab3lora",  "s256v2_2x8_ab3lora_20260623",   "ab3 lora (v2: g1536/133k)",       1536, 133000, "#9467bd"),
]
FIELDS = ("loss", "flow_mse", "grad_norm", "lr", "sigma_mean")


def load(rid):
    """Stream metrics.jsonl, keep only needed numeric fields, dedup step (last wins)."""
    by = {}
    p = f"{CK}/{rid}/metrics.jsonl"
    if not os.path.exists(p):
        return np.array([]), {f: np.array([]) for f in FIELDS}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "step" not in d:
                continue
            by[int(d["step"])] = tuple(float(d.get(k, np.nan)) for k in FIELDS)
    steps = np.array(sorted(by), dtype=float)
    cols = {FIELDS[i]: np.array([by[int(s)][i] for s in steps]) for i in range(len(FIELDS))}
    return steps, cols


def rolling(x, y, win):
    if len(x) < 2:
        return np.array(x), np.array(y)
    k = max(1, min(win, len(y)))
    c = np.cumsum(np.insert(y, 0, 0.0))
    rm = (c[k:] - c[:-k]) / k
    return np.array(x[k - 1:]), rm


def slope_se(xs, ys):
    """Linear slope + std-error of slope (per unit x). Returns (slope, se, n)."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    n = len(xs)
    if n < 10:
        return float("nan"), float("nan"), n
    A = np.vstack([xs, np.ones_like(xs)]).T
    coef, res, *_ = np.linalg.lstsq(A, ys, rcond=None)
    yhat = A @ coef
    dof = max(1, n - 2)
    s2 = float(np.sum((ys - yhat) ** 2) / dof)
    sxx = float(np.sum((xs - xs.mean()) ** 2))
    se = (s2 / sxx) ** 0.5 if sxx > 0 else float("nan")
    return float(coef[0]), float(se), n


data = {}
for k, rid, name, gb, ms, col in RUNS:
    s, cols = load(rid)
    data[k] = dict(rid=rid, name=name, gb=gb, max_steps=ms, col=col, steps=s, **cols)

summary = {}
WIN = 600

for k, rid, name, gb, ms, col in RUNS:
    d = data[k]
    s = d["steps"]; loss = d["loss"]; fm = d["flow_mse"]
    if len(s) == 0:
        summary[k] = {"name": name, "error": "no metrics"}
        continue
    last = int(s[-1])
    # 1) 20k-wide non-overlapping bins -> mean loss + marginal drop (diminishing returns)
    bins = []
    e = 0
    while e < last:
        a, b = e, min(e + 20000, last + 1)
        m = (s >= a) & (s < b)
        if m.sum() > 5:
            bins.append({"lo": a, "hi": b, "n": int(m.sum()),
                         "loss_mean": float(np.nanmean(loss[m])),
                         "flow_mse_mean": float(np.nanmean(fm[m]))})
        e += 20000
    # marginal drop between consecutive 20k bins
    for i in range(1, len(bins)):
        bins[i]["d_loss_vs_prev"] = round(bins[i]["loss_mean"] - bins[i - 1]["loss_mean"], 5)

    # 2) recent window (last 20k) slope + SE, step & sample axes
    rec = s >= max(0, last - 20000)
    sl_step, se_step, n_rec = slope_se(s[rec], loss[rec])             # per step
    smp = s * gb
    sl_smp, se_smp, _ = slope_se(smp[rec], loss[rec])                 # per sample
    # rolling-smoothed noise band over last 20k
    rx, ry = rolling(s[rec], loss[rec], min(WIN, max(2, int(rec.sum() // 4))))
    noise_band = float(np.nanstd(ry)) if len(ry) else float("nan")

    # 3) early-window slope for contrast (40k-60k if available, else first 20k after warmup)
    if last >= 60000:
         early = (s >= 40000) & (s < 60000)
    else:
        early = (s >= 5000) & (s < 25000)
    sl_early, se_early, _ = slope_se(s[early], loss[early])

    # 4) projection to run end at recent slope
    remaining = max(0, ms - last)
    proj_dloss = sl_step * remaining if np.isfinite(sl_step) else float("nan")
    cur_loss = float(np.nanmean(loss[-1000:]))
    summary[k] = {
        "name": name, "global_batch": gb, "max_steps": ms,
        "last_step": last, "n_points": len(s), "pct_done": round(100 * last / ms, 1),
        "cur_loss_last1k": round(cur_loss, 5),
        "cur_flow_mse_last1k": round(float(np.nanmean(fm[-1000:])), 5),
        "samples_seen_M": round(last * gb / 1e6, 1),
        "slope_recent20k_per10k_steps": round(sl_step * 1e4, 5) if np.isfinite(sl_step) else None,
        "slope_recent20k_se_per10k": round(se_step * 1e4, 5) if np.isfinite(se_step) else None,
        "slope_recent20k_per10M_samples": round(sl_smp * 1e7, 5) if np.isfinite(sl_smp) else None,
        "slope_early_per10k_steps": round(sl_early * 1e4, 5) if np.isfinite(sl_early) else None,
        "recent_noise_band_rollstd": round(noise_band, 5),
        "remaining_steps": remaining,
        "proj_dloss_to_end": round(proj_dloss, 5) if np.isfinite(proj_dloss) else None,
        "proj_pct_of_cur": round(100 * proj_dloss / cur_loss, 3) if np.isfinite(proj_dloss) and cur_loss else None,
        # verdict: is recent slope distinguishable from flat?
        "still_improving": bool(np.isfinite(sl_step) and (sl_step < 0) and (abs(sl_step) > 2 * se_step)),
        "sigma_mean_recent": round(float(np.nanmean(d["sigma_mean"][rec])), 4),
        "grad_norm_recent_median": round(float(np.nanmedian(d["grad_norm"][rec])), 4),
        "lr_last": float(d["lr"][-1]),
        "bins_20k": bins,
    }

print(json.dumps(summary, indent=2, ensure_ascii=False))

# =================== PLOTS ===================
def style(ax):
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# A) rolling loss vs SAMPLES (comparable x across all 4, given sample-budget alignment)
fig, ax = plt.subplots(figsize=(10, 5))
for k, rid, name, gb, ms, col in RUNS:
    d = data[k]
    if len(d["steps"]) == 0:
        continue
    smp = d["steps"] * gb / 1e6
    m = smp >= 5  # skip warmup
    rx, ry = rolling(smp[m], d["loss"][m], WIN)
    ax.plot(rx, ry, color=col, lw=1.6, label=name)
ax.set_title("Rolling loss vs SAMPLES seen (M) — sample-aligned view")
ax.set_xlabel("samples seen (millions)"); ax.set_ylabel("loss (rolling 600)")
ax.legend(); style(ax); fig.tight_layout(); fig.savefig(f"{OUT}/loss_vs_samples.png", dpi=120); plt.close(fig)

# B) rolling loss vs STEP (native)
fig, ax = plt.subplots(figsize=(10, 5))
for k, rid, name, gb, ms, col in RUNS:
    d = data[k]
    if len(d["steps"]) == 0:
        continue
    s = d["steps"]; m = s >= 2000
    rx, ry = rolling(s[m], d["loss"][m], WIN)
    ax.plot(rx, ry, color=col, lw=1.6, label=name)
ax.set_title("Rolling loss vs STEP (native step axis)")
ax.set_xlabel("step"); ax.set_ylabel("loss (rolling 600)")
ax.legend(); style(ax); fig.tight_layout(); fig.savefig(f"{OUT}/loss_vs_step.png", dpi=120); plt.close(fig)

# C) marginal drop per 20k-step bin (diminishing returns), per run
fig, axs = plt.subplots(2, 2, figsize=(13, 8))
for ax, (k, rid, name, gb, ms, col) in zip(axs.ravel(), RUNS):
    b = summary.get(k, {}).get("bins_20k", [])
    if not b:
        continue
    centers = [(x["lo"] + x["hi"]) / 2 / 1000 for x in b]
    dl = [x.get("d_loss_vs_prev", np.nan) for x in b]
    ax.bar(centers, [-(v if v is not None else np.nan) for v in dl], width=15, color=col, alpha=0.8)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_title(f"{name}\nloss DROP per 20k-step bin (higher=more gain)")
    ax.set_xlabel("step (k)"); ax.set_ylabel("−Δloss vs prev bin"); style(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/marginal_drop_bins.png", dpi=120); plt.close(fig)

# D) recent-window zoom loss vs step, all 4 (does the tail still slope down?)
fig, ax = plt.subplots(figsize=(10, 5))
for k, rid, name, gb, ms, col in RUNS:
    d = data[k]
    if len(d["steps"]) == 0:
        continue
    s = d["steps"]; last = int(s[-1])
    m = s >= max(0, last - 60000)
    rx, ry = rolling(s[m], d["loss"][m], WIN)
    ax.plot(rx, ry, color=col, lw=1.6, label=name)
ax.set_title("Recent 60k steps — rolling loss tail (is it still declining?)")
ax.set_xlabel("step"); ax.set_ylabel("loss (rolling 600)")
ax.legend(); style(ax); fig.tight_layout(); fig.savefig(f"{OUT}/loss_tail_zoom.png", dpi=120); plt.close(fig)

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n[plots] saved to {OUT}")
print("files:", sorted(os.listdir(OUT)))
