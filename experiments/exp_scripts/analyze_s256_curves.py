#!/usr/bin/env python3
"""Analyze stage256 training curves for the eval docs.

Reads the 4 runs' metrics.jsonl, computes binned/rolling/slope/cross-run stats,
renders matplotlib PNGs, and prints a JSON summary. CPU-only, read-only.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CK = "/fsx/evan/checkpoints/mot_pilot_v1"
OUT = "/fsx/evan/mot_code/mot_pilot_v1/experiments/results/eval_doc_curves"
os.makedirs(OUT, exist_ok=True)
RUNS = [
    ("v1", "s256_1x8_v1_20260618", "v1 baseline", "#1f77b4"),
    ("b1", "s256_1x8_b1adapter_20260618", "B1 text-adapter", "#d62728"),
    ("b2", "s256_1x8_b2longskip_20260618", "B2 long-skip", "#2ca02c"),
    ("b3", "s256_1x8_b3nonoise_20260618", "B3 no-noise", "#9467bd"),
]


def load(rid):
    by = {}
    p = f"{CK}/{rid}/metrics.jsonl"
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if "step" in d:
                    by[int(d["step"])] = d  # last wins (dedup resume overlaps)
    steps = sorted(by)
    return steps, [by[s] for s in steps]


def arr(rows, key):
    return np.array([r.get(key, np.nan) for r in rows], dtype=float)


def rolling(x, y, win):
    if len(x) < 2:
        return np.array(x), np.array(y)
    k = max(1, win)
    c = np.cumsum(np.insert(y, 0, 0.0))
    rm = (c[k:] - c[:-k]) / k
    return np.array(x[k - 1:]), rm


def slope(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 10:
        return float("nan")
    return float(np.polyfit(xs[m], ys[m], 1)[0])


data = {}
for k, rid, name, col in RUNS:
    s, rows = load(rid)
    data[k] = dict(rid=rid, name=name, col=col, steps=s, rows=rows)

summary = {}

# ---- per-run binned + slopes + component stats ----
edges = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 10000), (10000, 20000),
         (20000, 40000), (40000, 60000), (60000, 80000), (80000, 100000), (100000, 10**9)]
for k in data:
    s = np.array(data[k]["steps"], float)
    loss = arr(data[k]["rows"], "loss")
    fm = arr(data[k]["rows"], "flow_mse")
    binned = []
    for a, b in edges:
        m = (s >= a) & (s < b)
        if m.sum():
            binned.append([a, b, int(m.sum()), float(np.nanmean(loss[m]))])
    last = int(s[-1]) if len(s) else 0
    rec = (s >= max(0, last - 20000))
    summary[k] = {
        "name": data[k]["name"],
        "last_step": last,
        "n_points": len(s),
        "binned_loss": binned,
        "slope_recent20k_per10k": slope(s[rec], loss[rec]) * 1e4 if rec.sum() > 10 else None,
        "slope_from5k_per10k": slope(s[s >= 5000], loss[s >= 5000]) * 1e4 if (s >= 5000).sum() > 10 else None,
        "loss_last1000_mean": float(np.nanmean(loss[-1000:])) if len(loss) else None,
        "flow_mse_last1000_mean": float(np.nanmean(fm[-1000:])) if len(fm) else None,
        "grad_last1000_median": float(np.nanmedian(arr(data[k]["rows"], "grad_norm")[-1000:])) if len(s) else None,
        "iter_s_median_all": float(np.nanmedian(arr(data[k]["rows"], "iter_s"))) if len(s) else None,
        "peak_mem_gb_max": float(np.nanmax(arr(data[k]["rows"], "peak_mem_gb"))) if len(s) else None,
        "lr_last": float(arr(data[k]["rows"], "lr")[-1]) if len(s) else None,
    }

# ---- cross-run aligned loss at target steps ----
targets = [10000, 20000, 40000, 60000, 80000, 100000]
cross = {}
for T in targets:
    row = {}
    for k in data:
        s = np.array(data[k]["steps"], float)
        loss = arr(data[k]["rows"], "loss")
        m = (s >= T - 1000) & (s <= T + 1000)
        row[k] = float(np.nanmean(loss[m])) if m.sum() else None
    cross[T] = row
summary["cross_run_loss_at_step"] = cross

print(json.dumps(summary, indent=2, ensure_ascii=False))

# =================== PLOTS ===================
WIN = 600


def style(ax):
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# A) v1 loss full (raw faint + rolling)
fig, ax = plt.subplots(figsize=(9, 4.5))
s = np.array(data["v1"]["steps"], float); loss = arr(data["v1"]["rows"], "loss")
ax.plot(s, loss, color="#1f77b4", alpha=0.15, lw=0.5, label="raw")
rx, ry = rolling(s, loss, WIN)
ax.plot(rx, ry, color="#0b3d66", lw=1.8, label=f"rolling(mean,{WIN})")
ax.set_title("v1 baseline — training loss (full)"); ax.set_xlabel("step"); ax.set_ylabel("loss")
ax.legend(); style(ax); fig.tight_layout(); fig.savefig(f"{OUT}/v1_loss_full.png", dpi=110); plt.close(fig)

# B) v1 loss zoom (step>=2000)
fig, ax = plt.subplots(figsize=(9, 4.5))
mask = s >= 2000
rx, ry = rolling(s[mask], loss[mask], WIN)
ax.plot(rx, ry, color="#0b3d66", lw=1.8)
if len(ry):
    ax.set_ylim(np.nanmin(ry) - 0.004, np.nanpercentile(ry, 98) + 0.004)
ax.set_title("v1 baseline — loss zoomed (step≥2k): still declining"); ax.set_xlabel("step"); ax.set_ylabel("loss (rolling)")
style(ax); fig.tight_layout(); fig.savefig(f"{OUT}/v1_loss_zoom.png", dpi=110); plt.close(fig)

# C) all-4 loss overlay full
fig, ax = plt.subplots(figsize=(9, 4.5))
for k, rid, name, col in RUNS:
    s = np.array(data[k]["steps"], float); loss = arr(data[k]["rows"], "loss")
    rx, ry = rolling(s, loss, WIN)
    ax.plot(rx, ry, color=col, lw=1.5, label=name)
ax.set_title("All 4 runs — rolling loss (full)"); ax.set_xlabel("step"); ax.set_ylabel("loss")
ax.legend(); style(ax); fig.tight_layout(); fig.savefig(f"{OUT}/all4_loss_full.png", dpi=110); plt.close(fig)

# D) all-4 loss overlay zoom (step>=10000) — the ablation plot
fig, ax = plt.subplots(figsize=(9, 4.5))
ymins = []
for k, rid, name, col in RUNS:
    s = np.array(data[k]["steps"], float); loss = arr(data[k]["rows"], "loss")
    mask = s >= 10000
    rx, ry = rolling(s[mask], loss[mask], WIN)
    ax.plot(rx, ry, color=col, lw=1.6, label=name)
    if len(ry):
        ymins.append(np.nanmin(ry))
if ymins:
    ax.set_ylim(min(ymins) - 0.004, 0.985)
ax.set_title("All 4 runs — rolling loss zoomed (step≥10k)"); ax.set_xlabel("step"); ax.set_ylabel("loss (rolling)")
ax.legend(); style(ax); fig.tight_layout(); fig.savefig(f"{OUT}/all4_loss_zoom.png", dpi=110); plt.close(fig)

# E) v1 components 2x2
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
s = np.array(data["v1"]["steps"], float)
for ax, key, ttl in [(axs[0, 0], "grad_norm", "grad_norm"), (axs[0, 1], "lr", "lr"),
                     (axs[1, 0], "sigma_mean", "sigma_mean (per-step random)"), (axs[1, 1], "iter_s", "iter_s")]:
    y = arr(data["v1"]["rows"], key)
    if key in ("grad_norm", "iter_s", "sigma_mean"):
        ax.plot(s, y, color="#bbb", alpha=0.3, lw=0.4)
        rx, ry = rolling(s, y, WIN); ax.plot(rx, ry, color="#0b3d66", lw=1.5)
    else:
        ax.plot(s, y, color="#0b3d66", lw=1.2)
    ax.set_title(ttl); ax.set_xlabel("step"); style(ax)
fig.suptitle("v1 baseline — training diagnostics"); fig.tight_layout(); fig.savefig(f"{OUT}/v1_components.png", dpi=110); plt.close(fig)

# F) all-4 grad_norm overlay (zoom)
fig, ax = plt.subplots(figsize=(9, 4.5))
for k, rid, name, col in RUNS:
    s = np.array(data[k]["steps"], float); g = arr(data[k]["rows"], "grad_norm")
    mask = s >= 10000
    rx, ry = rolling(s[mask], g[mask], WIN)
    ax.plot(rx, ry, color=col, lw=1.4, label=name)
ax.set_title("All 4 runs — rolling grad_norm (step≥10k)"); ax.set_xlabel("step"); ax.set_ylabel("grad_norm")
ax.legend(); style(ax); fig.tight_layout(); fig.savefig(f"{OUT}/all4_grad_zoom.png", dpi=110); plt.close(fig)

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n[plots] saved to {OUT}")
print("files:", sorted(os.listdir(OUT)))
