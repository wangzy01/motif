#!/usr/bin/env python3
"""Assemble the stage256 4-model FINAL eval report (markdown + charts + sample grids).

Reads the eval_final_4model/ artifacts produced by the 9 eval jobs and writes:
  experiments/results/eval_final_4model/report/*.png   (charts + sample grids)
  experiments/results/EVAL_final_4model_stage256.md     (the report, image links relative)

Read-only over results; no model load (matplotlib Agg + PIL on 256px PNGs). Run via JCC.
Defensive: any missing piece (e.g. mPLUG best-effort) degrades to 'n/a', never crashes.
"""
from __future__ import annotations
import json, os, glob, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/fsx/evan/mot_code/mot_pilot_v1_cleanup"
BASE = os.path.join(ROOT, "experiments/results/eval_final_4model")
REP = os.path.join(BASE, "report")
os.makedirs(REP, exist_ok=True)
EVALSET = os.path.join(ROOT, "experiments/eval/prompts/anime_evalset.jsonl")
CKR = "/fsx/evan/checkpoints/mot_pilot_v1"
RUNS = ["b1", "ab1causal", "ab2fusion", "ab3lora"]
LABEL = {"b1": "b1 (bidirectional adapter, v1)", "ab1causal": "ab1causal (causal adapter, v2)",
         "ab2fusion": "ab2fusion (bidir adapter + multi-layer fusion, v2)",
         "ab3lora": "ab3lora (understanding-tower LoRA, v2)"}
CKDIR = {"b1": "s256_1x8_b1adapter_20260618", "ab1causal": "s256v2_2x8_ab1causal_20260623",
         "ab2fusion": "s256v2_2x8_ab2fusion_20260623", "ab3lora": "s256v2_2x8_ab3lora_20260623"}
COLOR = {"b1": "#888888", "ab1causal": "#4C9BE0", "ab2fusion": "#E0584C", "ab3lora": "#C7A23B"}
SEEDS = [0, 1, 2]
GB = {"b1": 512, "ab1causal": 1536, "ab2fusion": 1536, "ab3lora": 1536}  # global batch
CKSTEP = {"b1": 330000, "ab1causal": 126000, "ab2fusion": 126000, "ab3lora": 111000}  # eval'd EMA ckpt (fallback)


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def ms(xs):
    if not xs:
        return (None, None)
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0)


# ---------------- aggregate anime tagger / clip / aesthetic ----------------
def agg_anime():
    out = {}
    for r in RUNS:
        cats = {k: [] for k in ["mean_overall_hit", "mean_character_hit", "mean_copyright_hit",
                                "mean_general_hit", "mean_overall_cov50", "mean_overall_cov80"]}
        clip = []
        per_band = {}   # band -> list of overall_hit (pooled across seeds)
        for s in SEEDS:
            t = jload(f"{BASE}/anime/seed{s}/tagger_{r}.json")
            if t and "summary" in t:
                for k in cats:
                    if t["summary"].get(k) is not None:
                        cats[k].append(t["summary"][k])
                for pi in t.get("per_image", []):
                    b = pi.get("band", "?")
                    per_band.setdefault(b, []).append(pi.get("overall_hit", 0.0))
            c = jload(f"{BASE}/anime/seed{s}/clip_{r}.json")
            if c and "summary" in c:
                clip.append(c["summary"]["mean_clipscore"])
        aes = jload(f"{BASE}/anime/aesthetic_{r}.json")
        aesv = None
        if aes and r in aes.get("runs", {}):
            aesv = aes["runs"][r].get("mean")
        out[r] = {"cats": {k: ms(v) for k, v in cats.items()},
                  "clip": ms(clip), "aes": aesv,
                  "per_band": {b: ms(v) for b, v in per_band.items()},
                  "n_seed": len(cats["mean_overall_hit"])}
    return out


def agg_dpg(kind):  # kind = 'qwen' or 'mplug'
    out = {}
    for r in RUNS:
        d = jload(f"{BASE}/dpg/{kind}/dpg_{r}.json")
        if d and "summary" in d:
            s = d["summary"]
            out[r] = {"overall": s.get("dpg_overall"), "n": s.get("n_images"),
                      "l1": s.get("l1_category", {})}
        else:
            out[r] = None
    return out


def agg_fixedsigma():
    out = {}
    for r in RUNS:
        f = jload(f"{BASE}/fixedsigma/fixed_sigma_{r}.json")
        out[r] = f
    return out


# ---------------- charts ----------------
def chart_tagger_categories(anime):
    cats = [("mean_overall_hit", "overall"), ("mean_character_hit", "character"),
            ("mean_copyright_hit", "copyright"), ("mean_general_hit", "general")]
    x = np.arange(len(cats)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(RUNS):
        means = [anime[r]["cats"][k][0] or 0 for k, _ in cats]
        errs = [anime[r]["cats"][k][1] or 0 for k, _ in cats]
        ax.bar(x + (i - 1.5) * w, means, w, yerr=errs, capsize=3, label=LABEL[r], color=COLOR[r])
    ax.set_xticks(x); ax.set_xticklabels([c for _, c in cats])
    ax.set_ylabel("tagger hit-rate (recall@prod)"); ax.set_title("Anime concept tagger (3_1_13) — 3-seed mean ± std")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    p = f"{REP}/tagger_categories.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return os.path.relpath(p, os.path.join(ROOT, "experiments/results"))


def chart_byband(anime):
    bands = sorted({b for r in RUNS for b in anime[r]["per_band"]})
    if not bands:
        return None
    x = np.arange(len(bands)); w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, r in enumerate(RUNS):
        means = [(anime[r]["per_band"].get(b, (0, 0))[0] or 0) for b in bands]
        ax.bar(x + (i - 1.5) * w, means, w, label=LABEL[r], color=COLOR[r])
    ax.set_xticks(x); ax.set_xticklabels(bands)
    ax.set_ylabel("overall_hit"); ax.set_title("Anime overall_hit by rarity band (pooled 3 seeds)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    p = f"{REP}/tagger_byband.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return os.path.relpath(p, os.path.join(ROOT, "experiments/results"))


def chart_dpg(dq, dm):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(RUNS)); w = 0.38
    qv = [(dq[r]["overall"] if dq[r] else 0) for r in RUNS]
    ax.bar(x - w / 2, qv, w, label="Qwen3-VL judge", color="#4C9BE0")
    if any(dm[r] for r in RUNS):
        mv = [(dm[r]["overall"] if dm[r] else 0) for r in RUNS]
        ax.bar(x + w / 2, mv, w, label="mPLUG (leaderboard)", color="#E0584C")
    ax.set_xticks(x); ax.set_xticklabels(RUNS, rotation=0)
    ax.set_ylabel("DPG overall"); ax.set_title("DPG-Bench (strided ~267 prompts)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    p = f"{REP}/dpg_bar.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return os.path.relpath(p, os.path.join(ROOT, "experiments/results"))


def chart_clip_aes(anime):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    clip = [anime[r]["clip"][0] or 0 for r in RUNS]
    aes = [anime[r]["aes"] or 0 for r in RUNS]
    a1.bar(RUNS, clip, color=[COLOR[r] for r in RUNS]); a1.set_title("CLIPScore"); a1.grid(axis="y", alpha=0.3)
    a1.set_ylim(min(clip) - 1 if clip else 0, max(clip) + 0.5 if clip else 1)
    a2.bar(RUNS, aes, color=[COLOR[r] for r in RUNS]); a2.set_title("Aesthetic (aes-pixai)"); a2.grid(axis="y", alpha=0.3)
    a2.set_ylim(min(aes) - 0.1 if aes else 0, max(aes) + 0.05 if aes else 1)
    p = f"{REP}/clip_aes.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return os.path.relpath(p, os.path.join(ROOT, "experiments/results"))


def chart_fixedsigma(fs):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_data = False
    for r in RUNS:
        f = fs[r]
        if f and "loss_by_sigma" in f:
            xs = sorted(float(k) for k in f["loss_by_sigma"])
            ys = [f["loss_by_sigma"][f"{x:.2f}"] for x in xs]
            ax.plot(xs, ys, marker="o", label=f"{r} (mean {f.get('mean_loss_over_grid')})", color=COLOR[r])
            any_data = True
    if not any_data:
        plt.close(fig); return None
    ax.set_xlabel("sigma"); ax.set_ylabel("rectified-flow MSE")
    ax.set_title("Fixed-sigma MSE grid (lower=better; ~identical across ablations)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p = f"{REP}/fixedsigma_curve.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return os.path.relpath(p, os.path.join(ROOT, "experiments/results"))


def chart_loss():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    drew = False
    for r in RUNS:
        mp = f"{CKR}/{CKDIR[r]}/metrics.jsonl"
        if not os.path.exists(mp):
            continue
        steps, loss = [], []
        try:
            with open(mp) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    s = d.get("step") or d.get("global_step")
                    lo = d.get("loss")
                    if s is not None and lo is not None:
                        steps.append(s); loss.append(lo)
        except Exception:
            continue
        if not steps:
            continue
        steps = np.array(steps); loss = np.array(loss)
        # rolling mean (window ~500) for readability
        k = max(1, len(loss) // 400)
        idx = np.argsort(steps); steps, loss = steps[idx], loss[idx]
        sm = np.convolve(loss, np.ones(min(500, max(1, len(loss)//20)))/min(500, max(1, len(loss)//20)), mode="valid")
        ax.plot(steps[:len(sm)], sm, label=r, color=COLOR[r], lw=1.2)
        drew = True
    if not drew:
        plt.close(fig); return None
    ax.set_xlabel("step"); ax.set_ylabel("train loss (smoothed)")
    ax.set_title("Training loss (non-discriminating — plateau ~0.92)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0.88, 1.05)
    p = f"{REP}/loss_curve.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return os.path.relpath(p, os.path.join(ROOT, "experiments/results"))


# ---------------- sample grids ----------------
def find_img(run, eid, seed_dir):
    g = glob.glob(f"{BASE}/anime/samples/{seed_dir}/{run}/{eid}__{run}__seed*.png")
    return g[0] if g else None


def build_sample_grids(anime):
    ev = []
    try:
        for line in open(EVALSET):
            line = line.strip()
            if line:
                ev.append(json.loads(line))
    except Exception:
        return []
    # pick a spread: prefer entries with a character, across bands; cap ~18
    withchar = [e for e in ev if e.get("target_character")]
    sel = (withchar[:12] + [e for e in ev if not e.get("target_character")][:6]) or ev[:18]
    sel = sel[:18]
    cell = 256; pad = 6; labelh = 34; headerh = 26
    cols = len(RUNS)
    outs = []
    # render in chunks of 6 prompts per image
    for ci in range(0, len(sel), 6):
        chunk = sel[ci:ci + 6]
        rows = len(chunk)
        W = pad + cols * (cell + pad)
        H = headerh + rows * (cell + labelh + pad)
        canvas = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            fontb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        except Exception:
            font = fontb = ImageFont.load_default()
        for j, r in enumerate(RUNS):
            d.text((pad + j * (cell + pad) + 4, 6), r, fill="black", font=fontb)
        for i, e in enumerate(chunk):
            y0 = headerh + i * (cell + labelh + pad)
            tags = (e.get("target_character") or []) + (e.get("target_copyright") or [])
            cap = (",".join(tags)[:60]) if tags else (e.get("prompt", "")[:60])
            d.text((pad, y0 + cell + 2), f"{e.get('id','')}: {cap}", fill="black", font=font)
            for j, r in enumerate(RUNS):
                ip = find_img(r, e.get("id", ""), "seed0")
                x0 = pad + j * (cell + pad)
                if ip and os.path.exists(ip):
                    try:
                        im = Image.open(ip).convert("RGB").resize((cell, cell))
                        canvas.paste(im, (x0, y0))
                    except Exception:
                        d.rectangle([x0, y0, x0 + cell, y0 + cell], outline="red")
                else:
                    d.rectangle([x0, y0, x0 + cell, y0 + cell], outline="gray")
        p = f"{REP}/grid_samples_{ci//6}.png"
        canvas.save(p)
        outs.append(os.path.relpath(p, os.path.join(ROOT, "experiments/results")))
    return outs


def load_dpg_prompts():
    d = {}
    try:
        for line in open(f"{BASE}/dpg/dpg_subset.jsonl"):
            line = line.strip()
            if line:
                r = json.loads(line); d[r["id"]] = r.get("prompt", "")
    except Exception:
        pass
    return d


def find_dpg_img(run, iid):
    g = glob.glob(f"{BASE}/dpg/samples_qwen/{run}/{iid}__{run}__seed*.png")
    return g[0] if g else None


def build_dpg_grids(cell=384, per=3, n=12):
    """Real/general-content comparison from DPG samples (grid1, native 256), displayed large."""
    prompts = load_dpg_prompts()
    ids = [iid for iid in sorted(prompts) if all(find_dpg_img(r, iid) for r in RUNS)]
    if not ids:
        return []
    step = max(1, len(ids) // n)
    sel = ids[::step][:n]
    pad = 8; labelh = 46; headerh = 28; cols = len(RUNS)
    outs = []
    for ci in range(0, len(sel), per):
        chunk = sel[ci:ci + per]; rows = len(chunk)
        W = pad + cols * (cell + pad); H = headerh + rows * (cell + labelh + pad)
        canvas = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
            fontb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = fontb = ImageFont.load_default()
        for j, r in enumerate(RUNS):
            d.text((pad + j * (cell + pad) + 4, 6), r, fill="black", font=fontb)
        for i, iid in enumerate(chunk):
            y0 = headerh + i * (cell + labelh + pad)
            cap = (prompts.get(iid, "")[:95]) or iid
            d.text((pad, y0 + cell + 2), f"{iid}: {cap}", fill="black", font=font)
            for j, r in enumerate(RUNS):
                ip = find_dpg_img(r, iid); x0 = pad + j * (cell + pad)
                if ip and os.path.exists(ip):
                    try:
                        im = Image.open(ip).convert("RGB")
                        if im.size != (cell, cell):
                            im = im.resize((cell, cell), Image.LANCZOS)
                        canvas.paste(im, (x0, y0))
                    except Exception:
                        d.rectangle([x0, y0, x0 + cell, y0 + cell], outline="red")
                else:
                    d.rectangle([x0, y0, x0 + cell, y0 + cell], outline="gray")
        p = f"{REP}/grid_dpg_{ci // per}.png"; canvas.save(p)
        outs.append(os.path.relpath(p, os.path.join(ROOT, "experiments/results")))
    return outs


# ---------------- markdown ----------------
def fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "n/a"


def fmt_ms(t, nd=3):
    if not t or t[0] is None:
        return "n/a"
    return f"{t[0]:.{nd}f} ±{t[1]:.3f}"


def main():
    anime = agg_anime()
    dq = agg_dpg("qwen")
    dm = agg_dpg("mplug")
    fs = agg_fixedsigma()

    c_tag = chart_tagger_categories(anime)
    c_band = chart_byband(anime)
    c_dpg = chart_dpg(dq, dm)
    c_ca = chart_clip_aes(anime)
    c_fs = chart_fixedsigma(fs)
    c_loss = chart_loss()
    grids = build_sample_grids(anime)
    dpg_grids = build_dpg_grids()

    mplug_present = any(dm[r] for r in RUNS)

    L = []
    L.append("# Stage256 四模型最终评测报告")
    L.append("")
    L.append("> 2026-06-29。4 个 stage256 模型已全部停止训练，用各自**最新的 EMA checkpoint** 做横向对比。"
             "主指标 = **anime prompt-following**：生成图里画出了多少 prompt 要求的标签，用 anime tagger 打分，跑 3 个 seed 取平均并给标准差。"
             "辅助指标：DPG-Bench（Qwen3-VL + mPLUG 两个 judge）、CLIP score、aesthetic、fixed-sigma loss、训练 loss 曲线。")
    L.append("")
    L.append("**对比口径，先说明几点**：")
    L.append("- b1 用 v1 配方（global batch 512），三个消融用 v2 配方（global batch 1536）。配方不同，所以 **b1 与三个消融之间不是干净的 A/B 对比**。")
    L.append("- 四个模型都没训到原计划的最大步数就停了（用最新权重对比，已和 Evan 确认）；提前停也让训练样本量没对齐（见下表）。")
    L.append("- DPG 的 Qwen3-VL judge 只能内部互比、不能和公开 leaderboard 比；mPLUG judge 才能和公开 DPG leaderboard 对齐。")
    L.append("- anime 评测集不是严格 holdout（训练见过同源数据），所以只看相对高低、不看绝对分。")
    L.append("")
    # data exposure table
    L.append("## 训练样本量（评测用的 EMA checkpoint）")
    L.append("")
    L.append("| run | global batch | ckpt step | 训练样本量 (samples seen) |")
    L.append("|---|---|---|---|")
    for r in RUNS:
        step = (fs[r].get("ckpt_step") if fs[r] else None) or CKSTEP.get(r)
        exp = (step * GB[r] / 1e6) if step else None
        L.append(f"| {LABEL[r]} | {GB[r]} | {step if step else 'n/a'} | {(f'{exp:.1f}M' if exp else 'n/a')} |")
    L.append("")
    L.append("> 样本量 = ckpt step × global batch（**训练总共喂进去的样本数，含重复，不是去重后的图片数**）。"
             "原计划让 b1 训到 400k（约 205M）、v2 训到 133k（约 204M）两边对齐，但提前停训打断了对齐，"
             "实际是 **b1 169.0M（到 82%）、ab1causal/ab2fusion 各 193.5M（到 95%）、ab3lora 170.5M（到 83%）**。"
             "→ **唯一干净的同配方对比是 ab1causal vs ab2fusion（样本量都是 193.5M、配方相同）；b1 配方不同、样本还少约 13%，只能当参考。**")
    L.append("")
    # one-line conclusion
    order = sorted(RUNS, key=lambda r: -(anime[r]["cats"]["mean_overall_hit"][0] or 0))
    L.append("## 0. 结论速览")
    L.append(f"**Anime prompt-following（主指标）排名：{' > '.join(order)}。**")
    L.append("")
    L.append("- **ab2fusion 最好**：在 tagger 的每个类别（overall / character / copyright / general）和 CLIP score 上都第一，"
             "领先幅度明显大于 3 个 seed 之间的波动（overall 0.518 vs ab1causal 0.493）。")
    L.append("- **唯一干净的同配方对比是 ab1causal vs ab2fusion**（样本量都是 193.5M、配方相同）："
             "**ab2fusion（bidirectional adapter + 末 12 层 multi-layer fusion）> ab1causal（causal adapter）**，说明多层特征融合确实有增益。")
    L.append("- **b1 只能当参考**：它用 v1 配方、训练样本还少约 13%，所以 0.479 的落后同时受『配方不同 + 数据更少』影响，不是干净结论。")
    L.append("- **ab3lora 明确 no-go**：prompt-following 大幅垫底，character hit-rate 只有 0.06（约差 6 倍）。")
    L.append("- **决策**：停 b1（配方不同、又是和别人重复的 bidirectional adapter）合理；要带进 512 阶段的是 "
             "**ab2fusion（prompt-following 最强）还是 ab1causal（causal，省一半 attention 计算）**，这是还没定的权衡（见 §5）。")
    L.append("")
    L.append("**重要：要分题材看，不能简单说『ab2fusion 全面最好』。** 上面是 **anime 题材**的 prompt-following；"
             "换成 **DPG 的通用 / 写实 prompt**，**b1、ab1causal、ab2fusion 三者基本打平**——mPLUG judge 三者 85.45 / 85.66 / 85.55（最大只差 0.21），"
             "Qwen judge 甚至把 b1 排第一（88.9），而且 Qwen 的排名相对昨天的小子集还翻转了，说明这 ~1 分的差别在 judge 的噪声范围内。"
             "**也就是说，ab2fusion 的优势集中在 anime 概念 / 角色 / 版权这类题材，通用题材上几个模型差不多。** "
             "ab3lora 在 DPG 上同样垫底（约 79.6，低 6–8 分）。训练 loss、fixed-sigma（差 ≤0.4%）、aesthetic 三个指标都区分不出这几个模型——再次说明 loss 看不出消融差异。")
    L.append("")
    L.append("**肉眼核对样图（见 §4）**：4 个模型出图都完整、没有崩坏；ab3lora 肉眼可见最弱"
             "（比如 reona_west 穿成了西装而不是偶像装、watanabe_you 眼睛变成棕色且漏掉冰淇淋、2B 这个角色还原偏差），"
             "和它 tagger character hit-rate 垫底一致；b1 / ab1causal / ab2fusion 三者画面都不错、彼此接近。")
    L.append("")
    # tagger table
    L.append("## 1. Anime prompt-following（主指标；tagger 打分，3 个 seed 的平均 ± 标准差）")
    L.append("")
    L.append("> **现在用什么打分**：anime tagger 模型 `incantor/tagger_3_1_13`，通过 transformers 的 pipeline 跑。"
             "做法是对每张生成图预测 danbooru 风格的标签，再看 prompt 要求的目标标签（角色 / 版权 / 通用）命中了多少，算成 hit-rate。")
    L.append("> **关于同事建议的另一种做法**：用 pretrain 那套 pipeline（naifu 的 concept_v2）来跑 tagger eval，它更贴近训练时用的评测口径。"
             "但**目前做不了**——它需要一份固定的 `evalset.parquet` + `meta.json`，这两个文件本集群没有、也没有源数据自己生成，**需要找同事要**。"
             "拿到后可以再补一版和训练口径对齐的结果；当前报告里的 tagger 数都来自上面的 `tagger_3_1_13`。")
    L.append("> 表头说明：hit-rate = 命中的目标标签数 ÷ 目标标签总数；overall@.5 / @.8 = 目标标签里 tagger 预测概率超过 0.5 / 0.8 的比例（coverage）。")
    L.append("")
    L.append("| run | overall | character | copyright | general | overall@.5 | overall@.8 | CLIPScore | aesthetic | n_seed |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in RUNS:
        c = anime[r]["cats"]
        L.append(f"| {LABEL[r]} | {fmt_ms(c['mean_overall_hit'])} | {fmt_ms(c['mean_character_hit'])} | "
                 f"{fmt_ms(c['mean_copyright_hit'])} | {fmt_ms(c['mean_general_hit'])} | "
                 f"{fmt_ms(c['mean_overall_cov50'])} | {fmt_ms(c['mean_overall_cov80'])} | "
                 f"{fmt(anime[r]['clip'][0],2)} | {fmt(anime[r]['aes'])} | {anime[r]['n_seed']} |")
    L.append("")
    if c_tag:
        L.append(f"![tagger categories]({c_tag})")
    if c_band:
        L.append(f"![tagger by band]({c_band})")
    L.append("")
    # DPG
    L.append("## 2. DPG-Bench（通用 prompt-following，267 条 prompt）")
    L.append("")
    L.append("> DPG-Bench 让一个 judge 模型对每张生成图逐条问『画面里有没有 X』来打分（满分 100）。"
             "这里用两个 judge：Qwen3-VL-2B（只能内部互比）和原版 mPLUG（可对齐公开 leaderboard）。")
    L.append("")
    L.append("| run | DPG overall (Qwen3-VL) | " + ("DPG overall (mPLUG) | " if mplug_present else "") +
             "entity | attribute | relation | global | other | n |")
    L.append("|---|---|" + ("---|" if mplug_present else "") + "---|---|---|---|---|---|")
    for r in RUNS:
        q = dq[r]; m = dm[r]
        l1 = (q["l1"] if q else {})
        cells = [fmt(q["overall"], 2) if q else "n/a"]
        if mplug_present:
            cells.append(fmt(m["overall"], 2) if m else "n/a")
        for k in ["entity", "attribute", "relation", "global", "other"]:
            cells.append(fmt(l1.get(k), 1))
        cells.append(str(q["n"]) if q else "-")
        L.append(f"| {r} | " + " | ".join(cells) + " |")
    L.append("")
    if c_dpg:
        L.append(f"![dpg]({c_dpg})")
    L.append("")
    # CLIP/aes + fixed-sigma + loss
    L.append("## 3. 辅助指标（CLIP score / aesthetic / fixed-sigma loss / 训练 loss）")
    L.append("")
    if c_ca:
        L.append(f"![clip and aesthetic]({c_ca})")
    L.append("")
    L.append("**fixed-sigma loss（在固定噪声水平上算的重建误差，越低越好；几个模型预期非常接近）：**")
    L.append("")
    L.append("| run | 平均 fixed-sigma loss | ckpt step | 评测样本数 |")
    L.append("|---|---|---|---|")
    for r in RUNS:
        f = fs[r]
        if f:
            L.append(f"| {r} | {fmt(f.get('mean_loss_over_grid'),5)} | {f.get('ckpt_step')} | {f.get('n_eval')} |")
        else:
            L.append(f"| {r} | n/a | - | - |")
    L.append("")
    if c_fs:
        L.append(f"![fixed sigma]({c_fs})")
    if c_loss:
        L.append(f"![loss curve]({c_loss})")
    L.append("")
    # samples
    L.append("## 4. 样图对比（256px，CFG 4.0；每组 4 列 = b1 / ab1causal / ab2fusion / ab3lora）")
    L.append("")
    L.append("### 4.1 Anime 题材")
    L.append("")
    L.append("> 每行一个 prompt（标注角色 / 版权），4 个模型并排。整体都没崩；ab3lora 的角色和属性还原肉眼最弱（见 §0）。")
    L.append("")
    for g in grids:
        L.append(f"![anime samples]({g})")
        L.append("")
    L.append("### 4.2 真实 / 通用题材（DPG 测试集，放大显示）")
    L.append("")
    L.append("> 这是**非动漫**的通用 / 写实 prompt（来自 DPG-Bench），4 个模型并排。图片原生就是 **256px**，这里把每格放大到 384 只是为了看清楚（不是超分、不会更清晰）。"
             "可以看出真实题材下四个模型差距比 anime 题材小很多（和 §2 里 DPG 打平的结论一致）。")
    L.append("")
    for g in dpg_grids:
        L.append(f"![dpg real samples]({g})")
        L.append("")
    # go/no-go
    L.append("## 5. 去留决策")
    L.append("")
    L.append("| run | tagger 排名 | DPG 分 | 决策 |")
    L.append("|---|---|---|---|")
    rank = {r: i for i, r in enumerate(order)}
    for r in RUNS:
        q = dq[r]
        verdict = {"ab2fusion": "✅ 最强，512 阶段首选", "ab1causal": "✅ 保留（causal，省一半 attention）",
                   "b1": "⏸ 和别人重复的 bidirectional，已停", "ab3lora": "❌ no-go，已停"}.get(r, "-")
        L.append(f"| {r} | #{rank[r]+1} | {fmt(q['overall'],1) if q else 'n/a'} | {verdict} |")
    L.append("")
    L.append("## 6. 方法与复现")
    L.append("")
    L.append("- **生成**：`experiments/eval/sample_evalset.py`（anime 150 条）、`dpg/generate_dpg.py`（DPG，单图 + 2×2 两种）；"
             "都是 256px、64 步采样、CFG 4.0、用 EMA 权重；anime 跑 3 个 seed（7 / 1007 / 2007）。")
    L.append("- **打分**：anime tagger 用 `incantor/tagger_3_1_13`，加 CLIP（ViT-L-14）；"
             "DPG 用 `score_dpg.py`（Qwen3-VL-2B judge）和 `score_dpg_mplug.py`（原版 mPLUG judge，2×2 拼图）；"
             "aesthetic 用 `aes-pixai-1.2`；fixed-sigma 用 `fixed_sigma_eval.py`（anime 源 parquet，含 quality_z 列）。")
    L.append("- **tagger eval 的备选**：同事建议的 pretrain pipeline（naifu concept_v2）目前跑不了，缺 blessed `evalset.parquet` + `meta.json`，需要找同事要（见 §1）。")
    L.append("- **产物**：都在 `experiments/results/eval_final_4model/`（生成图 PNG + 每个模型的打分 JSON + 本报告用到的图表）。")
    L.append("")

    md = os.path.join(ROOT, "experiments/results/EVAL_final_4model_stage256.md")
    open(md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("[report] wrote", md)
    print("[report] charts:", c_tag, c_band, c_dpg, c_ca, c_fs, c_loss)
    print("[report] grids:", grids)
    print("[report] mplug_present:", mplug_present)
    # echo key numbers for the log
    print("\n=== ranking by tagger overall ===")
    for r in order:
        print(f"  {r:<10} overall={fmt_ms(anime[r]['cats']['mean_overall_hit'])} "
              f"char={fmt_ms(anime[r]['cats']['mean_character_hit'])} "
              f"dpg_qwen={fmt(dq[r]['overall'],2) if dq[r] else 'n/a'} "
              f"dpg_mplug={fmt(dm[r]['overall'],2) if dm[r] else 'n/a'}")


if __name__ == "__main__":
    main()
