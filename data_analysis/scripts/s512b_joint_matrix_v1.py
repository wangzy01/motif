#!/usr/bin/env python3
"""Stage512b-real 的 `domain x anchor` 联合整数分配 + QA 报告（离线设计工具，只读）。

为什么必须有这个：只冻结**边际**（domain_mix 与 anchor 概率）**不能确定联合分配**。
GPIC 约一半行落在 3:2 且 1.75 / 2:1 两个锚为零行、synthetic 100% 方形，
所以"各自独立满足两个边际"会隐式产生一个谁都没审过的 `P(domain|anchor)`，
并制造"域/来源—AR"混淆。活的 planner 里这一步是
`planner_v1.py:613-627` 的 `Q[b,d] = D[d]*N[b,d]/sum_b N[b,d]` —— 一个**乘积形式**目标，
它假设"每个域按自己的供给比例使用各个桶"，而实际域间 AR 分布差异很大。

本脚本对 `phase3a/phase5/planner` 之前的**设计期**做这件事：解出一个满足
  行和  sum_d C[k][d] = G * S[k]      （每锚步数 x global batch，单桶 batch 契约）
  列和  sum_k C[k][d] = D[d]          （域曝光总量）
  结构零 C[k][d] = 0  where rows[k][d] == 0
的整数矩阵，并报告 planner 现在**完全不报**的东西：每 cell 的唯一行数 / 有效轮次、
`P(domain|anchor)`、`P(source|anchor)`、实现边际与目标的偏差。

输入只有已算好的审计 REPORT.json（不扫数据、不读图）。
输出一个 JSON + 人读表格。**不写入任何 plan 目录**（plan root 是 0555 且成员集被校验）。

用法：
  python data_analysis/scripts/s512b_joint_matrix_v1.py --variant curated
  python data_analysis/scripts/s512b_joint_matrix_v1.py --variant survivor --out /tmp/qa.json
  python data_analysis/scripts/s512b_joint_matrix_v1.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from fractions import Fraction

AUDIT_BASE = "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop"
P1 = f"{AUDIT_BASE}/stage512b_anchor_audit_part1_20260731_v1/REPORT.json"
P15 = f"{AUDIT_BASE}/stage512b_curated_audit_part1p5_20260731_v1/REPORT.json"
P2 = f"{AUDIT_BASE}/stage512b_expansion_audit_part2_20260731_v2/REPORT.json"
# The authoritative supply table, written by the formal Phase-3a run.
FORMAL_N_MATRIX = f"{AUDIT_BASE}/offline_arb_bucketstats_stage512b_v1/N_matrix.parquet"

DEFAULT_MIX = {"anime": 0.30, "monet": 0.40, "gpic": 0.20,
               "synthetic": 0.05, "textatlas": 0.05}
DEFAULT_BUDGET = 512_000_000
GLOBAL_BATCH = 2048
FLOOR = 0.01
CAP = 0.22


class DesignError(RuntimeError):
    """设计期契约被破坏（fail-closed，不静默产出可疑矩阵）。"""


# --------------------------------------------------------------------------- #
# 整数分配原语
# --------------------------------------------------------------------------- #

def largest_remainder(total: int, weights: list[float]) -> list[int]:
    """把 `total` 按 `weights` 拆成整数，和**精确等于** total。
    平局取更小下标（与 `planner_v1.py:575-589 _largest_remainder_decimal` 同规则）。
    用 Fraction 做余数比较，避免浮点让平局判断不确定。
    """
    if total < 0:
        raise DesignError(f"total must be non-negative, got {total}")
    s = sum(weights)
    if s <= 0:
        raise DesignError("weights sum to zero")
    exact = [Fraction(int(round(w * 10**12)), 10**12) * total / Fraction(int(round(s * 10**12)), 10**12)
             for w in weights]
    floors = [int(x) for x in exact]
    left = total - sum(floors)
    if left < 0:
        raise DesignError("floor sum exceeded total")
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - floors[i]), i))
    for i in order[:left]:
        floors[i] += 1
    if sum(floors) != total:
        raise DesignError("largest-remainder conservation failure")
    return floors


def water_fill(p: list[float], floor: float, cap: float,
               exempt: set[int]) -> list[float]:
    """把概率向量夹到 `[floor, cap]`（`exempt` 里的下标免 floor），
    差额在未受约束的锚上**按原始比例**再分配。

    `capped` / `floored` 两个集合**只增不减**，所以最多 n 轮必然终止 ——
    朴素的"每轮重算夹取集合"写法会在"这轮被 cap、下轮又掉回 cap 以下"之间震荡。
    先做可行性预检并显式报错，不把不可行输入拖成"未收敛"。
    """
    n = len(p)
    if n == 0:
        raise DesignError("empty probability vector")
    if not (0.0 <= floor <= cap <= 1.0):
        raise DesignError(f"need 0 <= floor <= cap <= 1, got floor={floor} cap={cap}")
    if n * cap < 1.0 - 1e-12:
        raise DesignError(
            f"infeasible: {n} anchors capped at {cap} can carry at most {n * cap:.4f} < 1.0")
    if (n - len(exempt)) * floor > 1.0 + 1e-12:
        raise DesignError(
            f"infeasible: {n - len(exempt)} floored anchors at {floor} need "
            f"{(n - len(exempt)) * floor:.4f} > 1.0")

    capped: set[int] = set()
    floored: set[int] = set()
    for _ in range(n + 1):
        free = [k for k in range(n) if k not in capped and k not in floored]
        need = 1.0 - cap * len(capped) - floor * len(floored)
        base = sum(p[k] for k in free)
        if need < -1e-12:
            raise DesignError(f"water-filling infeasible: fixed mass exceeds 1 (need={need})")
        if not free:
            if abs(need) > 1e-9:
                raise DesignError(f"water-filling infeasible: no free anchors, residual {need}")
            break
        if base <= 0:
            raise DesignError("water-filling infeasible: free anchors carry zero natural mass")
        q = [0.0] * n
        for k in capped:
            q[k] = cap
        for k in floored:
            q[k] = floor
        for k in free:
            q[k] = p[k] * need / base
        new_cap = {k for k in free if q[k] > cap + 1e-15}
        new_flr = {k for k in free if k not in exempt and q[k] < floor - 1e-15}
        if not new_cap and not new_flr:
            if abs(sum(q) - 1.0) > 1e-9:
                raise DesignError(f"water-filling did not normalize: sum={sum(q)}")
            return q
        capped |= new_cap
        floored |= new_flr
    raise DesignError("water-filling did not converge (should be unreachable)")


def ipf(target: list[list[float]], row_sums: list[int], col_sums: list[int],
        allowed: list[list[bool]], iters: int = 2000) -> list[list[float]]:
    """把 `target` 拟合到给定行/列边际（保持结构零）。返回浮点矩阵。"""
    nr, nc = len(row_sums), len(col_sums)
    M = [[target[k][d] if allowed[k][d] else 0.0 for d in range(nc)] for k in range(nr)]
    for _ in range(iters):
        for k in range(nr):
            s = sum(M[k])
            if s > 0:
                f = row_sums[k] / s
                for d in range(nc):
                    M[k][d] *= f
        worst = 0.0
        for d in range(nc):
            s = sum(M[k][d] for k in range(nr))
            if s > 0:
                f = col_sums[d] / s
                for k in range(nr):
                    M[k][d] *= f
            worst = max(worst, abs(s - col_sums[d]))
        if worst < 1e-6:
            break
    return M


def integerize(M: list[list[float]], row_sums: list[int], col_sums: list[int],
               allowed: list[list[bool]]) -> list[list[int]]:
    """把浮点矩阵取整成**同时精确满足**行和与列和的整数矩阵。

    floor 之后按小数部分降序贪心补 1（只补行/列都还欠的 cell），再做一轮不看小数
    的兜底填充。仍有欠额则报错——宁可失败也不产出一个边际对不上的矩阵。
    """
    nr, nc = len(row_sums), len(col_sums)
    if sum(row_sums) != sum(col_sums):
        raise DesignError(f"margins disagree: rows {sum(row_sums)} vs cols {sum(col_sums)}")
    A = [[int(math.floor(M[k][d])) if allowed[k][d] else 0 for d in range(nc)] for k in range(nr)]
    rdef = [row_sums[k] - sum(A[k]) for k in range(nr)]
    cdef = [col_sums[d] - sum(A[k][d] for k in range(nr)) for d in range(nc)]
    if min(rdef) < 0 or min(cdef) < 0:
        raise DesignError("floor overshot a margin; IPF target inconsistent with margins")
    cells = sorted(
        ((k, d) for k in range(nr) for d in range(nc) if allowed[k][d]),
        key=lambda kd: -(M[kd[0]][kd[1]] - math.floor(M[kd[0]][kd[1]])),
    )
    for _ in range(4):
        for k, d in cells:
            take = min(rdef[k], cdef[d])
            if take > 0:
                A[k][d] += take
                rdef[k] -= take
                cdef[d] -= take
        if not any(rdef) and not any(cdef):
            break
    if any(rdef) or any(cdef):
        raise DesignError(
            f"integerization stalled; residual row deficits {[(k, v) for k, v in enumerate(rdef) if v]} "
            f"col deficits {[(d, v) for d, v in enumerate(cdef) if v]} -- structural zeros "
            f"make these margins jointly infeasible")
    for k in range(nr):
        if sum(A[k]) != row_sums[k]:
            raise DesignError(f"row {k} sum {sum(A[k])} != {row_sums[k]}")
    for d in range(nc):
        if sum(A[k][d] for k in range(nr)) != col_sums[d]:
            raise DesignError(f"col {d} sum != {col_sums[d]}")
    return A


# --------------------------------------------------------------------------- #
# 审计输入
# --------------------------------------------------------------------------- #

def _rows_by_anchor(per_anchor: list[dict], anchors: list[dict]) -> list[int]:
    out = [0] * len(anchors)
    for e in per_anchor:
        i = int(e["anchor_id"])
        if (e["width"], e["height"]) != (anchors[i]["width"], anchors[i]["height"]):
            raise DesignError(
                f"anchor_id {i} geometry mismatch: report says {e['width']}x{e['height']}, "
                f"table says {anchors[i]['width']}x{anchors[i]['height']}")
        out[i] = int(e["rows"])
    return out


def load_supply(variant: str) -> tuple[list[dict], dict[str, list[int]], dict]:
    """返回 (anchors, {domain: per-anchor rows}, provenance)。"""
    d1 = json.load(open(P1))
    d15 = json.load(open(P15))
    anchors = d15["anchors"]
    sup: dict[str, list[int]] = {}
    for dom, key in (("anime", "anime"), ("synthetic", "synthetic"), ("textatlas", "textatlas")):
        sup[dom] = _rows_by_anchor(d1["results"][key]["B13"]["per_anchor"], anchors)
    prov = {"variant": variant, "part1": P1, "part1p5": P15,
            "fixed_pool_domains": ["anime", "synthetic", "textatlas"]}
    if variant == "curated":
        sup["monet"] = _rows_by_anchor(d15["by_dataset"]["monet"]["per_anchor"], anchors)
        sup["gpic"] = _rows_by_anchor(d15["by_dataset"]["gpic"]["per_anchor"], anchors)
        prov["real_source"] = P15
        prov["p_source_given_anchor"] = d15.get("p_source_given_anchor")
    elif variant == "survivor":
        d2 = json.load(open(P2))
        sup["monet"] = _rows_by_anchor(d2["results"]["monet"]["per_anchor"], anchors)
        sup["gpic"] = _rows_by_anchor(d2["results"]["gpic"]["per_anchor"], anchors)
        prov["real_source"] = P2
        prov["real_source_caveat"] = (
            "Part 2 exited 44 (GPIC reproduced +117 rows of 81.9M, a p10-threshold "
            "rounding artifact); MONET reproduced exactly. Planning use only.")
    elif variant == "stage512a_today":
        sup["monet"] = _rows_by_anchor(d1["results"]["monet_real"]["B13"]["per_anchor"], anchors)
        sup["gpic"] = _rows_by_anchor(d1["results"]["gpic"]["B13"]["per_anchor"], anchors)
        prov["real_source"] = P1
    elif variant == "formal_n_matrix":
        # The audit REPORTs above are the 2026-07-31 design-time estimates.  The
        # formal Phase-3a run has since produced the authoritative N_matrix, and
        # they disagree where it matters: textatlas is 4,036,998 rows in the
        # audit (still square-pinned) versus 4,118,484 formally, and its
        # 704x352 cell is 7,774 versus 13,891.  Planning off the stale table
        # understates exactly the cells the TextAtlas crop audit cares about.
        import pyarrow.parquet as pq

        table = pq.read_table(FORMAL_N_MATRIX)
        cols = {c: table[c].to_pylist() for c in table.column_names}
        order = {(int(w), int(h)): i for i, (w, h) in
                 enumerate(zip(cols["bucket_w"], cols["bucket_h"]))}
        idx = [order[(int(a["width"]), int(a["height"]))] for a in anchors]
        if len(set(idx)) != len(anchors):
            raise DesignError("N_matrix anchors do not match the audit anchor set")
        for dom, col in (("anime", "n_anime"), ("synthetic", "n_synthetic"),
                         ("textatlas", "n_textatlas"), ("monet", "n_monet_real"),
                         ("gpic", "n_gpic")):
            sup[dom] = [int(cols[col][i]) for i in idx]
        prov["real_source"] = FORMAL_N_MATRIX
        prov["fixed_pool_domains"] = []
        prov["note"] = "all five domains from the formal Phase-3a N_matrix"
    else:
        raise DesignError(f"unknown variant {variant!r}")
    return anchors, sup, prov


# --------------------------------------------------------------------------- #
# 设计
# --------------------------------------------------------------------------- #

def design(variant: str, mix: dict[str, float], budget: int,
           global_batch: int = GLOBAL_BATCH, floor: float = FLOOR,
           cap: float = CAP) -> dict:
    anchors, sup, prov = load_supply(variant)
    na = len(anchors)
    doms = list(mix)
    for d in doms:
        if d not in sup:
            raise DesignError(f"no supply for domain {d!r}")
    if abs(sum(mix.values()) - 1.0) > 1e-9:
        raise DesignError(f"domain mix sums to {sum(mix.values())}, not 1")
    if budget % global_batch:
        raise DesignError(f"budget {budget} is not a multiple of global batch {global_batch}")
    total_steps = budget // global_batch

    dom_rows = {d: sum(sup[d]) for d in doms}
    for d in doms:
        if dom_rows[d] <= 0:
            raise DesignError(f"domain {d!r} has zero rows")

    # 1) 自然锚概率 = 乘积形式联合的锚边际
    p_nat = [sum(mix[d] * sup[d][k] / dom_rows[d] for d in doms) for k in range(na)]
    tot = sum(p_nat)
    p_nat = [x / tot for x in p_nat]

    # 2) water-filling（1:2 extent 的两个锚免 floor）
    exempt = {k for k, a in enumerate(anchors)
              if abs(abs(math.log(a["ar"])) - math.log(2.0)) < 1e-9}
    p_final = water_fill(p_nat, floor, cap, exempt)
    binding = {
        "floor_raised": [f'{anchors[k]["width"]}x{anchors[k]["height"]}'
                         for k in range(na) if p_final[k] > p_nat[k] + 1e-12 and p_nat[k] < floor],
        "cap_clipped": [f'{anchors[k]["width"]}x{anchors[k]["height"]}'
                        for k in range(na) if p_nat[k] > cap + 1e-12],
        "is_noop": all(abs(p_final[k] - p_nat[k]) < 1e-12 for k in range(na)),
        "floor_exempt_anchors": sorted(f'{anchors[k]["width"]}x{anchors[k]["height"]}' for k in exempt),
    }

    # 3) 整数边际
    S = largest_remainder(total_steps, p_final)             # 每锚步数
    row_sums = [s * global_batch for s in S]                # 每锚曝光（单桶 batch）
    D = largest_remainder(budget, [mix[d] for d in doms])   # 每域曝光

    # 4) 联合整数矩阵
    allowed = [[sup[d][k] > 0 for d in doms] for k in range(na)]
    target = [[mix[doms[j]] * sup[doms[j]][k] / dom_rows[doms[j]] * budget
               for j in range(len(doms))] for k in range(na)]
    fitted = ipf(target, row_sums, D, allowed)
    C = integerize(fitted, row_sums, D, allowed)

    # 5) QA
    per_anchor = []
    for k in range(na):
        a = anchors[k]
        mix_rows = sum(sup[d][k] for d in doms)
        cells = {}
        for j, d in enumerate(doms):
            rows = sup[d][k]
            exp = C[k][j]
            cells[d] = {
                "unique_rows": rows,
                "exposures": exp,
                "effective_epochs": (exp / rows) if rows else None,
                "p_domain_given_anchor": (exp / row_sums[k]) if row_sums[k] else None,
                "structurally_zero": rows == 0,
            }
        per_anchor.append({
            "anchor_id": k,
            "anchor": f'{a["width"]}x{a["height"]}',
            "ar": a["ar"],
            "anchor_area": a["width"] * a["height"],
            "p_natural": p_nat[k],
            "p_final": p_final[k],
            "steps": S[k],
            "exposures": row_sums[k],
            "mix_unique_rows": mix_rows,
            "effective_epochs": (row_sums[k] / mix_rows) if mix_rows else None,
            "below_global_batch": 0 < mix_rows < global_batch,
            "floor_exempt": k in exempt,
            "cells": cells,
        })

    worst = sorted(
        ({"domain": d, "anchor": per_anchor[k]["anchor"],
          "unique_rows": sup[d][k], "exposures": C[k][j],
          "effective_epochs": C[k][j] / sup[d][k]}
         for k in range(na) for j, d in enumerate(doms)
         if sup[d][k] > 0 and C[k][j] > 0),
        key=lambda x: -x["effective_epochs"])[:10]

    achieved_dom = {d: sum(C[k][j] for k in range(na)) for j, d in enumerate(doms)}
    result = {
        "schema": "mot_stage512b_joint_matrix_v1",
        "read_only": True,
        "provenance": prov,
        "inputs": {
            "domain_mix": mix, "budget_exposures": budget, "global_batch": global_batch,
            "total_steps": total_steps, "prob_floor": floor, "prob_cap": cap,
            "anchor_set_id": "B13", "n_anchors": na,
        },
        "water_filling": binding,
        "domain_totals": {d: {"target_exposures": D[j], "achieved_exposures": achieved_dom[d],
                              "unique_rows": dom_rows[d],
                              "marginal_epochs": D[j] / dom_rows[d]}
                          for j, d in enumerate(doms)},
        "per_anchor": per_anchor,
        "worst_cells_by_reuse": worst,
        "qa": {
            "domain_marginal_exact": all(achieved_dom[d] == D[j] for j, d in enumerate(doms)),
            "anchor_marginal_exact": all(
                sum(C[k][j] for j in range(len(doms))) == row_sums[k] for k in range(na)),
            "total_exposures": sum(row_sums),
            "budget_matches": sum(row_sums) == budget,
            "anchors_with_zero_rows": [per_anchor[k]["anchor"] for k in range(na)
                                       if per_anchor[k]["mix_unique_rows"] == 0],
            "anchors_below_global_batch": [per_anchor[k]["anchor"] for k in range(na)
                                           if per_anchor[k]["below_global_batch"]],
            "structural_zero_cells": [
                {"domain": d, "anchor": per_anchor[k]["anchor"]}
                for k in range(na) for j, d in enumerate(doms) if sup[d][k] == 0],
            "max_cell_epochs": worst[0]["effective_epochs"] if worst else None,
            "cells_over_10_epochs": sum(1 for w in worst if w["effective_epochs"] > 10),
        },
        "caveats": [
            "effective_epochs is a CELL AVERAGE. The live planner's R_max is an aggregate "
            "per-cell cap, not a per-row cap: for monet/gpic/synthetic nothing bounds how "
            "many times an individual row appears (weighted_streams_v1.py systematic "
            "sampling proportional to base_weight). Per-row max reuse is NOT modelled here "
            "and is not reported anywhere in the pipeline.",
            "The caption-band axis (T1..T4) is a THIRD dimension not solved here. Band "
            "quotas are computed per (anchor, domain) cell and renormalized over locally "
            "supported bands, so the global achieved band mix is a supply-weighted mixture, "
            "not the target vector.",
            "P(source|anchor) is carried through from the audit report for the curated "
            "variant only; it is a supply share, not a planned-exposure share.",
        ],
    }
    return result


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #

def render(res: dict) -> str:
    out = []
    inp = res["inputs"]
    prov = res["provenance"]
    out.append(f'=== Stage512b joint matrix: variant={prov["variant"]} ===')
    out.append(f'budget {inp["budget_exposures"]:,} exposures = {inp["total_steps"]:,} steps '
               f'@ global batch {inp["global_batch"]}')
    wf = res["water_filling"]
    out.append(f'water-filling: {"NO-OP (natural vector already inside the band)" if wf["is_noop"] else "ACTIVE"}'
               f'  floor-raised={wf["floor_raised"] or "none"}  cap-clipped={wf["cap_clipped"] or "none"}')
    out.append("")
    hdr = (f'{"anchor":>9} {"AR":>6} {"nat%":>8} {"final%":>8} {"steps":>8} '
           f'{"exposures":>13} {"uniq rows":>12} {"epochs":>7}')
    out.append(hdr)
    out.append("-" * len(hdr))
    for e in res["per_anchor"]:
        ep = e["effective_epochs"]
        out.append(f'{e["anchor"]:>9} {e["ar"]:>6.3f} {e["p_natural"]*100:>7.3f}% '
                   f'{e["p_final"]*100:>7.3f}% {e["steps"]:>8,} {e["exposures"]:>13,} '
                   f'{e["mix_unique_rows"]:>12,} {ep:>7.2f}'
                   + ("  *floor-exempt" if e["floor_exempt"] else ""))
    out.append("-" * len(hdr))
    out.append(f'{"TOTAL":>9} {"":>6} {"100.000%":>8} {"100.000%":>8} '
               f'{inp["total_steps"]:>8,} {res["qa"]["total_exposures"]:>13,}')
    out.append("")
    out.append("per-domain marginals:")
    for d, v in res["domain_totals"].items():
        out.append(f'  {d:>10}  target {v["target_exposures"]:>13,}  achieved {v["achieved_exposures"]:>13,}'
                   f'  unique {v["unique_rows"]:>12,}  epochs {v["marginal_epochs"]:>6.2f}')
    out.append("")
    out.append("P(domain | anchor):")
    doms = list(res["domain_totals"])
    out.append("  " + f'{"anchor":>9} ' + " ".join(f'{d:>10}' for d in doms))
    for e in res["per_anchor"]:
        row = " ".join(
            (f'{e["cells"][d]["p_domain_given_anchor"]*100:>9.2f}%'
             if e["cells"][d]["p_domain_given_anchor"] is not None else f'{"-":>10}')
            for d in doms)
        out.append("  " + f'{e["anchor"]:>9} ' + row)
    out.append("")
    out.append("worst cells by reuse (cell-average epochs):")
    for w in res["worst_cells_by_reuse"][:8]:
        out.append(f'  {w["domain"]:>10} @ {w["anchor"]:>9}  rows {w["unique_rows"]:>12,}'
                   f'  exposures {w["exposures"]:>13,}  epochs {w["effective_epochs"]:>8.2f}')
    out.append("")
    qa = res["qa"]
    out.append("QA:")
    out.append(f'  domain marginal exact : {qa["domain_marginal_exact"]}')
    out.append(f'  anchor marginal exact : {qa["anchor_marginal_exact"]}')
    out.append(f'  budget matches        : {qa["budget_matches"]} ({qa["total_exposures"]:,})')
    out.append(f'  anchors w/ zero rows  : {qa["anchors_with_zero_rows"] or "none"}')
    out.append(f'  anchors < global batch: {qa["anchors_below_global_batch"] or "none"}')
    out.append(f'  structural zero cells : {len(qa["structural_zero_cells"])}')
    out.append(f'  max cell epochs       : {qa["max_cell_epochs"]:.2f}')
    out.append(f'  cells over 10 epochs  : {qa["cells_over_10_epochs"]}')
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #

def self_test() -> int:
    fails = []

    def ck(name, ok, detail=""):
        print(("  PASS  " if ok else "  FAIL  ") + name + ("" if ok else f"  {detail}"))
        if not ok:
            fails.append(name)

    # largest_remainder 守恒 + 平局取低位
    ck("largest_remainder conserves", sum(largest_remainder(1000, [0.3, 0.4, 0.3])) == 1000)
    ck("largest_remainder tie -> lower index",
       largest_remainder(3, [0.5, 0.5]) == [2, 1], str(largest_remainder(3, [0.5, 0.5])))
    ck("largest_remainder exact thirds",
       sum(largest_remainder(100, [1, 1, 1])) == 100, str(largest_remainder(100, [1, 1, 1])))

    # water_fill：一个真实形状的 13 锚向量，两个锚超 cap、一个锚低于 floor
    v13 = [0.30, 0.25, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03, 0.02, 0.015, 0.010, 0.005]
    ck("test vector sums to 1", abs(sum(v13) - 1.0) < 1e-12, str(sum(v13)))
    q = water_fill(v13, 0.01, 0.22, set())
    ck("water_fill caps and floors a 13-anchor vector",
       abs(sum(q) - 1) < 1e-12 and max(q) <= 0.22 + 1e-12 and min(q) >= 0.01 - 1e-12, str(q))
    ck("water_fill caps exactly the two over-cap anchors",
       abs(q[0] - 0.22) < 1e-12 and abs(q[1] - 0.22) < 1e-12, str(q[:2]))
    ck("water_fill raises exactly the under-floor anchor",
       abs(q[12] - 0.01) < 1e-12, str(q[12]))
    q = water_fill(v13, 0.01, 0.22, {12})
    ck("water_fill honours floor exemption", q[12] < 0.01 and abs(sum(q) - 1) < 1e-12, str(q[12]))
    q0 = [0.2, 0.3, 0.5]
    ck("water_fill is identity inside the band",
       water_fill(q0, 0.01, 0.9, set()) == q0, str(water_fill(q0, 0.01, 0.9, set())))
    # 单调性回归：朴素写法会在这种输入上 cap/uncap 震荡
    q = water_fill([0.40, 0.30, 0.20, 0.06, 0.04], 0.01, 0.30, set())
    ck("water_fill terminates on multi-cap input",
       abs(sum(q) - 1) < 1e-12 and max(q) <= 0.30 + 1e-12, str(q))
    # 不可行输入必须显式报错，而不是"未收敛"
    try:
        water_fill([0.5, 0.3, 0.2], 0.01, 0.22, set())
        ck("infeasible cap raises explicitly", False, "did not raise")
    except DesignError as exc:
        ck("infeasible cap raises explicitly", "at most" in str(exc), str(exc))
    try:
        water_fill([0.5, 0.5], 0.6, 0.9, set())
        ck("infeasible floor raises explicitly", False, "did not raise")
    except DesignError as exc:
        ck("infeasible floor raises explicitly", "> 1.0" in str(exc), str(exc))

    # integerize：结构零 + 精确边际
    tgt = [[50.4, 49.6], [30.2, 0.0], [20.1, 0.0]]
    allowed = [[True, True], [True, False], [True, False]]
    rows = [100, 30, 20]
    cols = [100, 50]
    fit = ipf(tgt, rows, cols, allowed)
    A = integerize(fit, rows, cols, allowed)
    ck("integerize row sums exact", all(sum(A[k]) == rows[k] for k in range(3)), str(A))
    ck("integerize col sums exact",
       all(sum(A[k][d] for k in range(3)) == cols[d] for d in range(2)), str(A))
    ck("integerize respects structural zeros", A[1][1] == 0 and A[2][1] == 0, str(A))

    # 不可行边际必须报错，不能悄悄产出
    try:
        integerize([[1.0, 0.0]], [10], [5, 5], [[True, False]])
        ck("infeasible margins raise", False, "did not raise")
    except DesignError:
        ck("infeasible margins raise", True)

    # 真实审计输入端到端
    try:
        res = design("curated", DEFAULT_MIX, DEFAULT_BUDGET)
        qa = res["qa"]
        ck("curated: domain marginal exact", qa["domain_marginal_exact"])
        ck("curated: anchor marginal exact", qa["anchor_marginal_exact"])
        ck("curated: budget matches", qa["budget_matches"])
        ck("curated: 13 anchors", len(res["per_anchor"]) == 13)
        ck("curated: no anchor below global batch",
           not qa["anchors_below_global_batch"], str(qa["anchors_below_global_batch"]))
        ck("curated: P(domain|anchor) sums to 1 per anchor",
           all(abs(sum(e["cells"][d]["p_domain_given_anchor"] for d in DEFAULT_MIX) - 1.0) < 1e-9
               for e in res["per_anchor"]))
    except (OSError, DesignError) as exc:
        ck("curated end-to-end", False, str(exc))

    try:
        res2 = design("survivor", DEFAULT_MIX, DEFAULT_BUDGET)
        ck("survivor: margins exact",
           res2["qa"]["domain_marginal_exact"] and res2["qa"]["anchor_marginal_exact"])
    except (OSError, DesignError) as exc:
        ck("survivor end-to-end", False, str(exc))

    print()
    if fails:
        print(f"FAILED {len(fails)}: {fails}")
        return 1
    print("S512B_JOINT_MATRIX_SELF_TEST_PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["curated", "survivor", "stage512a_today", "formal_n_matrix"],
                    default="curated")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--global-batch", type=int, default=GLOBAL_BATCH)
    ap.add_argument("--floor", type=float, default=FLOOR)
    ap.add_argument("--cap", type=float, default=CAP)
    ap.add_argument("--mix", type=str, default="",
                    help='override, e.g. "anime=0.3,monet=0.4,gpic=0.2,synthetic=0.05,textatlas=0.05"')
    ap.add_argument("--out", type=str, default="", help="write the QA JSON here (never inside a plan root)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    mix = dict(DEFAULT_MIX)
    if args.mix:
        mix = {}
        for part in args.mix.split(","):
            k, v = part.split("=")
            mix[k.strip()] = float(v)

    res = design(args.variant, mix, args.budget, args.global_batch, args.floor, args.cap)
    print(render(res))
    if args.out:
        path = os.path.abspath(args.out)
        if "/offline_arb_plan" in path or "/blocks/" in path:
            raise DesignError(f"refusing to write inside a plan root: {path}")
        with open(path, "w") as fh:
            json.dump(res, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
