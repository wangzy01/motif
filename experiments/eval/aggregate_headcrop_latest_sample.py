#!/usr/bin/env python3
import glob
import json
import os
from collections import Counter, defaultdict


ROOT = "/fsx/evan/mot_code/data_analysis/outputs/headcrop_latest_step60000_20260708"
OUTDIR = os.path.join(ROOT, "report_sample_zh")
SCORE_DIRS = [
    os.path.join(ROOT, "subagent_scores_tag_ready"),
    os.path.join(ROOT, "subagent_scores_remaining_sample"),
]


def domain(row):
    return "real" if str(row.get("prompt_id", "")).startswith("real_") else "anime"


def any_cut(row):
    return bool(row.get("head_cut") or row.get("foot_cut") or row.get("side_cut"))


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def load_rows():
    rows = []
    bad = []
    for score_dir in SCORE_DIRS:
        for path in sorted(glob.glob(os.path.join(score_dir, "*.jsonl"))):
            with open(path) as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception as exc:
                        bad.append((path, line_no, str(exc)))
                        continue
                    row["_score_file"] = os.path.basename(path)
                    row["_score_dir"] = os.path.basename(score_dir)
                    rows.append(row)
    by_image = {}
    dup = 0
    for row in rows:
        image = row["image"]
        if image in by_image:
            dup += 1
            continue
        by_image[image] = row
    return list(by_image.values()), dup, bad


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(fn(row) if callable(fn) else row.get(fn, "") for fn in keys)
        groups[key].append(row)
    out = []
    for key, group_rows in sorted(groups.items()):
        n = len(group_rows)
        head = sum(bool(row.get("head_cut")) for row in group_rows)
        foot = sum(bool(row.get("foot_cut")) for row in group_rows)
        side = sum(bool(row.get("side_cut")) for row in group_rows)
        badn = sum(any_cut(row) for row in group_rows)
        fullok = sum(row.get("primary_label") == "A_full_ok" for row in group_rows)
        otherbad = sum(row.get("primary_label") == "D_other_bad" for row in group_rows)
        subj = sum(float(row.get("subject_complete_0to5", 0) or 0) for row in group_rows) / n
        comp = sum(float(row.get("composition_follow_0to5", 0) or 0) for row in group_rows) / n
        out.append(
            key
            + (
                n,
                fullok,
                pct(fullok, n),
                head,
                pct(head, n),
                foot,
                pct(foot, n),
                side,
                pct(side, n),
                badn,
                pct(badn, n),
                otherbad,
                pct(otherbad, n),
                subj,
                comp,
            )
        )
    return out


def write_tables(rows, dup, bad):
    os.makedirs(OUTDIR, exist_ok=True)
    header_tail = [
        "n",
        "full_ok",
        "full_ok_pct",
        "head_cut",
        "head_cut_pct",
        "foot_cut",
        "foot_cut_pct",
        "side_cut",
        "side_cut_pct",
        "any_cut",
        "any_cut_pct",
        "other_bad",
        "other_bad_pct",
        "subject_avg",
        "compose_avg",
    ]
    tables = [
        ("by_domain.tsv", [domain]),
        ("by_domain_style.tsv", [domain, "prompt_style"]),
        ("by_domain_size.tsv", [domain, "size"]),
        ("by_domain_style_size.tsv", [domain, "prompt_style", "size"]),
        ("by_domain_target.tsv", [domain, "composition_target"]),
        ("by_style_target.tsv", ["prompt_style", "composition_target"]),
        ("by_size_target.tsv", ["size", "composition_target"]),
    ]
    for fname, keys in tables:
        with open(os.path.join(OUTDIR, fname), "w") as f:
            names = [key if isinstance(key, str) else "domain" for key in keys]
            f.write("\t".join(names + header_tail) + "\n")
            for row in summarize(rows, keys):
                vals = [f"{value:.2f}" if isinstance(value, float) else str(value) for value in row]
                f.write("\t".join(vals) + "\n")

    with open(os.path.join(OUTDIR, "coverage.txt"), "w") as f:
        f.write(f"raw_rows\t{len(rows) + dup}\n")
        f.write(f"unique_images\t{len(rows)}\n")
        f.write(f"duplicates_skipped\t{dup}\n")
        f.write(f"parse_errors\t{len(bad)}\n")
        f.write("by_score_dir\t" + json.dumps(Counter(row["_score_dir"] for row in rows), sort_keys=True) + "\n")
        f.write("by_domain\t" + json.dumps(Counter(domain(row) for row in rows), sort_keys=True) + "\n")
        f.write("by_style\t" + json.dumps(Counter(row.get("prompt_style") for row in rows), sort_keys=True) + "\n")
        f.write("by_size\t" + json.dumps(Counter(row.get("size") for row in rows), sort_keys=True) + "\n")
        f.write("by_target\t" + json.dumps(Counter(row.get("composition_target") for row in rows), sort_keys=True) + "\n")
        if bad:
            f.write("parse_error_examples\t" + repr(bad[:10]) + "\n")

    examples = [row for row in rows if any_cut(row)]
    examples = sorted(
        examples,
        key=lambda row: (
            domain(row),
            row.get("prompt_style", ""),
            row.get("size", ""),
            row.get("prompt_id", ""),
            row.get("seed", 0),
        ),
    )
    with open(os.path.join(OUTDIR, "bad_examples.jsonl"), "w") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def lookup(fname, key):
    path = os.path.join(OUTDIR, fname)
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if tuple(parts[: len(key)]) == tuple(key):
                return dict(zip(header, parts))
    raise KeyError((fname, key))


def row_line(name, row):
    return (
        f"| {name} | {row['n']} | {row['full_ok_pct']}% | {row['head_cut_pct']}% | "
        f"{row['foot_cut_pct']}% | {row['side_cut_pct']}% | {row['any_cut_pct']}% | "
        f"{row['subject_avg']} | {row['compose_avg']} |"
    )


def write_report():
    anime = lookup("by_domain.tsv", ("anime",))
    real = lookup("by_domain.tsv", ("real",))
    tag = lookup("by_domain_style.tsv", ("anime", "tag"))
    nl_plain = lookup("by_domain_style.tsv", ("anime", "nl_plain"))
    nl_emph = lookup("by_domain_style.tsv", ("anime", "nl_emph"))
    real_style = lookup("by_domain_style.tsv", ("real", "real_nl"))

    style_rows = [
        row_line("anime tag", tag),
        row_line("anime NL plain", nl_plain),
        row_line("anime NL emph", nl_emph),
        row_line("real NL", real_style),
    ]
    size_rows = []
    for size in ["384x688", "416x624", "512x512", "624x416", "688x384"]:
        try:
            size_rows.append(row_line(size, lookup("by_domain_size.tsv", ("anime", size))))
        except KeyError:
            pass

    report = f"""# Stage512a 最新权重 head/foot 截断快速复评

日期：2026-07-08

## 结论

最新 `step_60000.pt` 的结果继续支持前一版判断：stage256 由中心裁剪带来的“系统性头部截断偏置”在 stage512a multi-AR/ARB 训练后已经基本解决。当前真正残留的是脚部/下肢截断，主要集中在 anime 的 full-body multi、NL plain、方图/宽图场景；real 样本没有观察到系统性头截断或脚截断。

不建议因为这个问题回头重训 256。更合理的后续方向是继续 stage512a，并针对 full-body multi、宽图、动态姿势和 NL plain prompt 做 targeted eval/数据或 prompt ablation。

## 覆盖

- 本轮最新权重生成图片：510 张。
- 已逐图评分并纳入本报告：250 张。
- 其中 anime tag 全量：130 张。
- 追加抽样：120 张，覆盖 anime NL plain、anime NL emph、real NL；每个 manifest 抽 8 张，两个 seed 均覆盖，优先 full-body/multi，再补 non-fullbody control。
- 分辨率：384x688、416x624、512x512、624x416、688x384。
- 权重：`/fsx/evan/checkpoints/mot_pilot_v1/redesign_512a_multiar_anchor_global1536_20260705/step_60000.pt`，EMA，约 92.16M exposure。

## 主表

| domain | n | full_ok | head_cut | foot_cut | side_cut | any_cut | subject | compose |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{row_line('anime', anime)}
{row_line('real', real)}

## 按提示词

| slice | n | full_ok | head_cut | foot_cut | side_cut | any_cut | subject | compose |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(style_rows)}

## anime 按长宽比

| size | n | full_ok | head_cut | foot_cut | side_cut | any_cut | subject | compose |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(size_rows)}

## 与前一版对照

前一版 quick report 的 `rd512a_latest` 是较早 latest，anime n=240、real n=18：anime head_cut 5.83%、foot_cut 28.75%、any_cut 34.17%；real head_cut 5.56%、foot_cut 11.11%、any_cut 16.67%。

本轮 step60000 使用更靠后的权重，并把 real 扩到 40 张、NL plain/NL emph 做了均衡抽样。总体上，头部截断没有反弹；real 的 frame cut 更少；anime 的残留主要被更难的 NL plain/full-body multi 抽样放大，表现为脚部截断而不是头顶被切。

## 定性观察

1. 头部截断：不再是普遍模式。多数 full-body/multi 图里头发、头顶和脸部完整，只有少数 non-fullbody control 或极紧方图/宽图出现头顶轻微切边。
2. 脚部截断：仍是最大残留，尤其 anime NL plain。很多图能放下上半身和大部分腿，但脚或小腿落到画面外。
3. full-body 强调：anime NL emph 明显优于 NL plain，说明提示词仍能降低脚部截断，但不能完全根治复杂多人/动态姿势。
4. real：主要问题不是截断，而是人数、主体关系、遮挡等 prompt-following 问题；从 head/foot crop 角度看已经稳定。
5. 长宽比：竖图和 416x624 相对稳；688x384/624x416 与 512x512 中的 anime full-body multi 更容易出现脚部/侧边残留。

## 建议

继续按当前 stage512a 路线推进。截断问题已经从“stage256 center-crop 学坏构图”转成“复杂 full-body/multi 服从与脚部保留”的局部问题；无需因该问题重训 256。

后续若要压残留，优先构建一个小而硬的 regression set：anime NL plain/full-body multi、动态姿势、宽图/方图，各固定 seed，专看 foot_cut 和 side_cut；同时保留 real full-body multi 作为 sanity check。
"""
    with open(os.path.join(OUTDIR, "HEAD_CROP_LATEST_STEP60000_SAMPLE_REPORT_ZH.md"), "w") as f:
        f.write(report)


def main():
    rows, dup, bad = load_rows()
    write_tables(rows, dup, bad)
    write_report()
    print("WROTE", OUTDIR)


if __name__ == "__main__":
    main()
