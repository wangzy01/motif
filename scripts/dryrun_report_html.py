"""把 dryrun_sampler_stats.py 的 JSON 渲染成自包含 HTML 仪表盘（纯 HTML+CSS 柱状条，无 JS）。
复刻并扩展旧 t5_e2e_report_html 的风格：每 stage 一卡（域/源/档/captioner/重复），外加
anime 深度、caption 长度+截断率、回源验证、完整性断言。
用法: python dryrun_report_html.py [stats.json] [out.html]
"""
import html
import json
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "experiments/results/dryrun/sampler_stats.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "experiments/results/dryrun/sampler_stats_report.html"
d = json.load(open(SRC))
meta = d.get("meta", {})
ST = d.get("stages", {})

PAL = {"anime": "#8c5fd6", "monet_real": "#4e79a7", "monet": "#4e79a7", "gpic": "#59a14f",
       "synthetic": "#e15759", "synth": "#e15759", "coyo": "#4e79a7", "laion": "#76b7b2",
       "cc12m": "#f28e2b", "commoncatalog-cc-by": "#edc948", "megalith10m": "#b07aa1",
       "diffusion-aesthetic-4k": "#ff9da7"}
DEF = "#9c9c9c"


def bar(label, frac, extra="", color=None, target=None):
    pct = max(0.0, min(1.0, frac or 0.0)) * 100
    c = color or PAL.get(label, DEF)
    tl = (f'<span class="tgt" style="left:{min(100,(target or 0)*100)}%"></span>'
          if target is not None else "")
    return (f'<div class="row"><div class="lab" title="{html.escape(str(label))}">{html.escape(str(label))}</div>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%;background:{c}"></div>{tl}</div>'
            f'<div class="val">{(frac or 0)*100:.1f}%{extra}</div></div>')


def card(title, inner, cls=""):
    return f'<div class="card {cls}"><h3>{title}</h3>{inner}</div>'


def q_line(label, q):
    if not q:
        return f"<div class='stat'>{label}: <i>n/a</i></div>"
    return (f"<div class='stat'>{label}: p50 <b>{q.get('p50','-')}</b> · p90 {q.get('p90','-')} · "
            f"p99 {q.get('p99','-')} · max {q.get('max','-')} · mean {q.get('mean','-')} (n={q.get('n','-')})</div>")


# ---- per-stage cards ----
band_target = meta.get("domain_mix") and None  # placeholder
stage_cards = ""
for sk in ("256", "512", "1024"):
    s = ST.get(sk)
    if not s or "error" in s:
        stage_cards += card(f"Stage {sk}", f"<i>{(s or {}).get('error','缺')}</i>")
        continue
    dom = "".join(bar(k, v, target=s.get("domain_target", {}).get(k)) for k, v in s["domain_share"].items())
    src = "".join(bar(k, v) for k, v in list(s.get("source_share", {}).items())[:10])
    bt = s.get("band_target", {})
    band = "".join(bar(b, s.get("band_share_real", {}).get(b, 0), color="#4e79a7", target=bt.get(b))
                   for b in ("T1", "T2", "T3", "T4"))
    t4 = "".join(bar(k.replace("_first_sentence", "·1st"), v, color="#af7aa1")
                 for k, v in list(s.get("captioner_by_band", {}).get("T4", {}).items())[:6] if v > 0)
    rep = s.get("repeat", {}).get("overall", {})
    micro = s.get("repeat", {}).get("micro_sources", [])[:4]
    micro_h = "".join(
        f"<div class='stat'>{html.escape(m['source'])} ({m['domain']}): 池 {m['pool_n']:,} · 抽 {m['draws']:,} · "
        f"distinct {m['distinct_drawn']:,} · max <b>{m['max_repeat']}</b> · 曝光×<b>{m['exposure_x']}</b></div>"
        for m in micro)
    stat = (f"<div class='stat'>distinct <b>{rep.get('distinct_drawn',0):,}</b>/{s['draws']:,} · "
            f"max repeat <b>{rep.get('max_repeat','-')}</b> · mean {rep.get('mean_repeat','-')} · "
            f"覆盖 {rep.get('distinct_drawn',0)/max(rep.get('pool_n',1),1)*100:.1f}%</div>")
    inner = (f"<div class='sub'>域占比 (竖线=配置 domain_mix)</div>{dom}"
             f"<div class='sub'>源占比 top10 (real+synth)</div>{src or '<i>n/a</i>'}"
             f"<div class='sub'>caption 档 (real内, 竖线=目标 18/27/15/40)</div>{band}"
             f"<div class='sub'>T4 由哪个 captioner 服务</div>{t4 or '<i>n/a</i>'}"
             f"<div class='sub'>重复 (全域)</div>{stat}"
             f"<div class='sub'>过曝微源 top4 (按曝光×)</div>{micro_h or '<i>n/a</i>'}")
    stage_cards += card(f"Stage {sk}", inner)

# ---- anime deep cards ----
anime_cards = ""
for sk in ("256", "512", "1024"):
    a = (ST.get(sk) or {}).get("anime")
    if not a:
        continue
    rd = a.get("repeat_count_dist", {})
    rd_h = " · ".join(f"{k}×: {v:,}" for k, v in rd.items())
    route = a.get("caption_route_expected", {})
    route_h = "".join(bar(k, v, color="#6a9fb5") for k, v in route.items())
    rating_h = "".join(bar(k, v, color="#d88") for k, v in list(a.get("rating_dist", {}).items())[:6])
    cl = a.get("cluster", {})
    inner = (q_line("priority", a.get("priority", {}))
             + f"<div class='stat'>has_NL 比例: <b>{(a.get('has_nl_rate') or 0)*100:.1f}%</b> · "
             f"composition_clean: <b>{('%.1f%%'%((a.get('composition_clean_rate') or 0)*100)) if a.get('composition_clean_rate') is not None else 'n/a'}</b></div>"
             + f"<div class='stat'>整数 repeat 分布: {rd_h or 'n/a'} (mean {a.get('repeat_count_mean','-')})</div>"
             + f"<div class='sub'>caption 路由(期望)</div>{route_h or '<i>n/a</i>'}"
             + f"<div class='sub'>rating 分布</div>{rating_h or '<i>n/a</i>'}"
             + f"<div class='stat'>簇覆盖: 抽中 <b>{cl.get('distinct_clusters_drawn',0):,}</b>/{cl.get('total_clusters_in_pool',0):,} · "
             f"top10 占比 <b>{(cl.get('top10_share') or 0)*100:.2f}%</b></div>")
    anime_cards += card(f"anime 深度 · Stage {sk}", inner, cls="an")

# ---- caption length / truncation ----
cap_rows = ""
for sk in ("256", "512", "1024"):
    c = (ST.get(sk) or {}).get("caption_len")
    if not c or "error" in (c or {}):
        continue
    for dom, rec in c.get("by_domain", {}).items():
        rt = rec.get("real_tok", {})
        cap_rows += (f"<tr><td>{sk}</td><td>{dom}</td><td>{rec.get('n','-')}</td>"
                     f"<td>{rec.get('proxy_tok',{}).get('p50','-')}/{rec.get('proxy_tok',{}).get('p99','-')}</td>"
                     f"<td>{rt.get('p50','-')}/{rt.get('p99','-')}/{rt.get('max','-')}</td>"
                     f"<td class='{'fail' if (rec.get('trunc_rate') or 0)>0.05 else 'pass'}'>"
                     f"{(rec.get('trunc_rate') or 0)*100:.1f}%</td></tr>")
budget = ""
for sk in ST:
    b = (ST[sk].get("caption_len") or {}).get("budget_tok")
    if b:
        budget = b
        break
cap_tbl = (f"<table class='checks'><tr><th>stage</th><th>域</th><th>n</th><th>proxy p50/p99</th>"
           f"<th>real p50/p99/max</th><th>截断率(>{budget}tok)</th></tr>{cap_rows}</table>"
           if cap_rows else "<i>n/a (未启用 tokenizer 或无样本)</i>")

# ---- backfetch ----
bf_rows = ""
for sk in ("256", "512", "1024"):
    bf = (ST.get(sk) or {}).get("backfetch")
    if not bf or bf.get("skipped") or "error" in bf:
        continue
    for dom, r in bf.items():
        md = r.get("min_dim_q", {})
        bf_rows += (f"<tr><td>{sk}</td><td>{dom}</td><td>{r.get('tried',0)}</td>"
                    f"<td class='{'pass' if (r.get('success_rate') or 0)>=0.95 else 'fail'}'>"
                    f"{(r.get('success_rate') or 0)*100:.1f}%</td>"
                    f"<td>{md.get('p50','-')} (min {md.get('n','-')}样)</td>"
                    f"<td>{r.get('ar_q',{}).get('p50','-')}</td>"
                    f"<td class='{'fail' if r.get('below_stage',0) else 'pass'}'>{r.get('below_stage',0)}</td></tr>")
bf_tbl = (f"<table class='checks'><tr><th>stage</th><th>域</th><th>试</th><th>成功率</th>"
          f"<th>短边 p50</th><th>AR p50</th><th>需上采样数</th></tr>{bf_rows}</table>"
          if bf_rows else "<i>n/a (未启用回源验证)</i>")

# ---- integrity ----
int_rows = ""
for sk in ("256", "512", "1024"):
    ig = (ST.get(sk) or {}).get("integrity")
    if not ig:
        continue
    for a in ig.get("assertions", []):
        int_rows += (f"<tr><td>{sk}</td><td>{html.escape(a['name'])}</td>"
                     f"<td>{html.escape(str(a['detail']))}</td>"
                     f"<td class='{'pass' if a['ok'] else 'fail'}'>{'✅' if a['ok'] else '❌'}</td></tr>")
    un = ig.get("unusable_no_captioner", {})
    for dom, u in un.items():
        int_rows += (f"<tr><td>{sk}</td><td>tok≤0/无captioner: {dom}</td>"
                     f"<td>{u['count']:,}/{u['pool_n']:,} ({u['frac']*100:.2f}%)</td><td></td></tr>")
int_tbl = (f"<table class='checks'><tr><th>stage</th><th>检查</th><th>详情</th><th>状态</th></tr>{int_rows}</table>"
           if int_rows else "<i>n/a</i>")
hr = meta.get("holdout_removed", {})
hr_h = " · ".join(f"{k}: {v:,}" for k, v in hr.items())

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,'PingFang SC','Microsoft YaHei',sans-serif;
background:#f6f7f9;color:#1d1f23;margin:0;padding:24px;font-size:14px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:22px 0 10px} .meta{color:#666;margin-bottom:18px}
.card{background:#fff;border:1px solid #e4e7eb;border-radius:10px;padding:16px 18px;margin:0 0 16px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.card.an{background:#fbf9ff;border-color:#e6ddf5}
h3{margin:0 0 12px;font-size:16px} .sub{color:#7a8088;font-size:12px;margin:12px 0 5px;font-weight:600}
.row{display:flex;align-items:center;margin:3px 0} .lab{width:175px;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.track{flex:1;background:#eef0f2;border-radius:4px;height:15px;position:relative;overflow:hidden}
.fill{height:100%;border-radius:4px} .val{width:120px;text-align:right;font-variant-numeric:tabular-nums;font-size:12px}
.tgt{position:absolute;top:-2px;height:19px;width:2px;background:#111;opacity:.55}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
table.checks{border-collapse:collapse;width:100%} table.checks td,table.checks th{border-bottom:1px solid #eceef0;padding:7px 10px;text-align:left;font-size:13px}
table.checks th{font-size:12px;color:#7a8088} .pass{color:#1a7f37;font-weight:600} .fail{color:#c0392b;font-weight:700}
.stat{font-size:12px;color:#333;margin:3px 0}
@media(max-width:980px){.grid{grid-template-columns:1fr}}
"""
m = meta
HTML = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>MoT complete_v1 采样 dry-run 报告</title><style>{CSS}</style></head><body>
<h1>MoT complete_v1 · 采样 dry-run 统计报告</h1>
<div class="meta">每 stage 抽样 {m.get('n_draws',0):,} · seed {m.get('seed','')} ·
α={m.get('source_alpha')} β={m.get('cluster_beta')} ·
shard {m.get('shard_counts',{})} ·
caption 子集 {m.get('caption_sample',0):,} · 回源/域 {m.get('backfetch_per_domain',0)} ·
holdout 移除 [{hr_h}]<br>{html.escape(m.get('note',''))}</div>
<h2>各阶段采样分布</h2>
<div class="grid">{stage_cards}</div>
<h2>anime 域深度</h2>
<div class="grid">{anime_cards or '<i>n/a</i>'}</div>
<h2>caption 实际长度 + 截断率 (真 Qwen tokenizer)</h2>
{card('caption 长度/截断', cap_tbl)}
<h2>回源验证 (真读图)</h2>
{card('取图成功率 / 实际分辨率', bf_tbl)}
<h2>数据完整性断言</h2>
{card('断言', int_tbl)}
</body></html>"""
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(HTML)
print("WROTE", OUT, f"({len(HTML)} bytes)")
