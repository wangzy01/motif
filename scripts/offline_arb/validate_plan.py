#!/usr/bin/env python3
"""
validate_plan.py — 独立验证器,OFFLINE_ARB_V1 L3 Training Plan(offline exact-bucket planner)。

设计前提(务必先读 README_VALIDATOR.md):
  本文件仅依据以下两份文档实现,未阅读、未 import 任何 planner(build_plan.py /
  training/dataline_offline/*)实现代码:
    - docs/OFFLINE_ARB_PLANNER_DESIGN.md  (§1 配额求解, §1.5 输入契约, §2-5 构造算法,
      §6 物化格式(2026-07-10 增补 domain_counts), §8 验证清单 V1-V8,
      §9 规格补钉 9.1-9.10——初版验证器的 14 条歧义清单大部分已被 §9 裁定钉死)
    - docs/OFFLINE_ARB_V1_SPEC.md          (Phase-0 七条硬口径)

  实现者(planner)与验证者(本文件)故意相互独立,目的是让验证结果不受实现细节假设污染。
  §9 已钉死的项(header 精确字段名 9.6、checksums 结构 9.7、npz 键名/dtype 9.8、
  虚拟 catalog 行号 9.3、LR 平局规则 9.1、F1 修订 9.2、发牌公式 9.5、±1 修正规则 9.4、
  输入校验 9.9、V5/V6 口径 9.10)一律按文档字面精确实现,不再做启发式别名探测;
  仍未闭合的少数点在 README_VALIDATOR.md「遗留歧义」一节逐条列出,并在运行报告的
  "ambiguities" 字段里按需提示。

  纯 numpy + Python 标准库,不 import torch,不做重计算编排(真实大 plan 验证走 JCC;本文件
  本身在合成小规模输入上是轻量的,可在登录机跑 smoke)。

用法:
  python validate_plan.py --input planner_input.json --plan plan_root/ \
      [--rebuild-plan plan_root2/] [--report out.json] \
      [--rebuild-mode {auto,suffix,full}] [--max-blocks N] [--skip-deep-item-checks]

退出码:0 = 全部已跑的检查项 PASS(或 SKIP/WARN/INFO);非 0 = 存在 FAIL 或 ERROR。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FORMAT_VERSION_EXPECTED = 1
ROW_SCHEME_EXPECTED = "virtual_catalog_v1"  # §9.3 / §9.6 钉死

# §1.5 输入契约字段(精确名)
REQUIRED_INPUT_KEYS = [
    "format_version",
    "plan_id",
    "seed",
    "world_size",
    "per_rank_bs",
    "total_steps",
    "domains",
    "domain_mix",
    "square_only_domains",
    "buckets",
    "square_bucket_idx",
    "N",
    "R_max",
    "block_steps",
]

# §9.6 header.json 必含字段(精确名)
REQUIRED_HEADER_FIELDS = [
    "format_version",
    "plan_id",
    "seed",
    "world_size",
    "per_rank_bs",
    "total_steps",
    "G",
    "block_steps",
    "domains",
    "buckets",
    "square_bucket_idx",
    "square_only_domains",
    "D",
    "S",
    "C",
    "N_total",
    "R_max",
    "hashes",
    "solver",
    "row_scheme",
]

STATUS_ORDER = {"ERROR": 0, "FAIL": 1, "WARN": 2, "SKIP": 3, "PASS": 4, "INFO": 5}


# ============================================================================
# 通用工具
# ============================================================================


def to_jsonable(obj: Any) -> Any:
    """把 numpy 标量/数组等转成可 json.dump 的原生类型。"""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def largest_remainder(total: int, weights: Sequence[float]) -> np.ndarray:
    """确定性 largest-remainder 取整:Σ result == total。

    平局规则按 §9.1 钉死:小数部分降序,平局按索引升序。
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return np.zeros((0,), dtype=np.int64)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("largest_remainder: weights sum must be > 0")
    ideal = total * w / s
    floors = np.floor(ideal).astype(np.int64)
    remainder = int(total - int(floors.sum()))
    if remainder < 0:
        # 数值误差极端情况下的防御(理论上不应发生)
        remainder = 0
    if remainder > 0:
        frac = ideal - floors
        idx = np.arange(w.size)
        order = np.lexsort((idx, -frac))  # primary: -frac(降序 frac), tie: idx 升序(§9.1)
        for i in range(remainder):
            floors[order[i]] += 1
    return floors


class Report:
    """检查项累加器。"""

    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []
        self.ambiguities: List[str] = []
        self.timings: Dict[str, float] = {}

    def add(self, code: str, name: str, status: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        assert status in STATUS_ORDER, f"unknown status {status}"
        self.items.append(
            {
                "code": code,
                "name": name,
                "status": status,
                "message": message,
                "details": to_jsonable(details or {}),
            }
        )

    def note_ambiguity(self, text: str) -> None:
        if text not in self.ambiguities:
            self.ambiguities.append(text)

    def worst_status(self) -> str:
        """整体判定:ERROR > FAIL > WARN > PASS。SKIP/INFO 是预期中的"未执行/仅提示",
        不应该压过一个全绿的 PASS——例如未提供 --rebuild-plan 时 V5/V6 合理地 SKIP,
        不代表整体验证"劣于 PASS"。"""
        if not self.items:
            return "ERROR"
        statuses = {i["status"] for i in self.items}
        for s in ("ERROR", "FAIL", "WARN"):
            if s in statuses:
                return s
        return "PASS"

    def exit_code(self) -> int:
        bad = any(i["status"] in ("FAIL", "ERROR") for i in self.items)
        return 1 if bad else 0


def guarded(report: Report, code: str, name: str):
    """装饰一个检查函数:任何未捕获异常都转成该检查项的 ERROR,不让整个工具崩溃。"""

    def wrap(fn):
        def inner(*a, **kw):
            try:
                fn(*a, **kw)
            except Exception as e:  # noqa: BLE001 - 验证器必须对任意 planner 输出形状健壮
                tb = traceback.format_exc(limit=6)
                report.add(code, name, "ERROR", f"{type(e).__name__}: {e}", {"traceback": tb})

        return inner

    return wrap


# ============================================================================
# 输入加载(§1.5 契约 + §9.9 严格校验)
# ============================================================================


def load_input(path: Path, report: Report) -> Dict[str, Any]:
    with open(path, "r") as f:
        raw = json.load(f)
    missing = [k for k in REQUIRED_INPUT_KEYS if k not in raw]
    if missing:
        raise ValueError(f"planner_input.json 缺少 §1.5 契约要求字段: {missing}")

    domains = list(raw["domains"])
    buckets = [tuple(b) for b in raw["buckets"]]
    n_d = len(domains)
    n_b = len(buckets)

    N = np.asarray(raw["N"], dtype=np.float64)
    if N.shape != (n_b, n_d):
        raise ValueError(f"N 形状 {N.shape} 与 [len(buckets)={n_b}][len(domains)={n_d}] 不符")
    if not np.all(np.isclose(N, np.round(N))):
        raise ValueError("N 含非整数值,应为逐 cell item 计数(§9.9)")
    N_int = np.round(N).astype(np.int64)
    if np.any(N_int < 0):
        raise ValueError("N 含负值(§9.9 要求非负整数)")

    domain_mix = raw["domain_mix"]
    missing_mix = [d for d in domains if d not in domain_mix]
    if missing_mix:
        raise ValueError(f"domain_mix 缺少域: {missing_mix}")
    m = np.asarray([float(domain_mix[d]) for d in domains], dtype=np.float64)
    mix_sum = float(m.sum())
    # §9.9:Σ domain_mix = 1(容差 1e-9),不自动归一,违者报错
    if abs(mix_sum - 1.0) > 1e-9:
        raise ValueError(f"Σ domain_mix = {mix_sum!r} 偏离 1.0 超过 1e-9(§9.9:不自动归一,直接报错)")

    square_only = list(raw["square_only_domains"])
    unknown_sq = [d for d in square_only if d not in domains]
    if unknown_sq:
        raise ValueError(f"square_only_domains 含未知域: {unknown_sq}")

    sq_idx = int(raw["square_bucket_idx"])
    if not (0 <= sq_idx < n_b):
        raise ValueError(f"square_bucket_idx={sq_idx} 越界 (n_buckets={n_b})")
    h, w = buckets[sq_idx]
    # §9.9:square_bucket_idx 必须对应 h==w 的桶,违者报错
    if h != w:
        raise ValueError(f"square_bucket_idx={sq_idx} 指向非方桶 ({h},{w})(§9.9)")

    R_max = raw["R_max"]
    missing_rmax = [d for d in domains if d not in R_max]
    if missing_rmax:
        raise ValueError(f"R_max 缺少域: {missing_rmax}")

    world_size = int(raw["world_size"])
    per_rank_bs = int(raw["per_rank_bs"])
    total_steps = int(raw["total_steps"])
    G = world_size * per_rank_bs
    M = total_steps * G

    data = dict(raw)
    data.update(
        {
            "domains": domains,
            "buckets": buckets,
            "N": N_int,
            "m": m,
            "square_only_domains": square_only,
            "square_bucket_idx": sq_idx,
            "R_max": {d: int(R_max[d]) for d in domains},
            "world_size": world_size,
            "per_rank_bs": per_rank_bs,
            "total_steps": total_steps,
            "G": G,
            "M": M,
            "block_steps": int(raw["block_steps"]),
        }
    )
    return data


def load_header(plan_root: Path) -> Dict[str, Any]:
    hp = plan_root / "header.json"
    if not hp.exists():
        raise FileNotFoundError(f"{hp} 不存在")
    with open(hp, "r") as f:
        return json.load(f)


def plan_looks_complete(plan_root: Path) -> Tuple[bool, str]:
    """判定 plan_root 是否是一个"planner 已产出的完整 plan"(用于 V7)。"""
    if not plan_root.exists():
        return False, "plan_root 目录不存在"
    hp = plan_root / "header.json"
    if not hp.exists():
        return False, "header.json 不存在"
    try:
        with open(hp) as f:
            header = json.load(f)
    except Exception as e:  # noqa: BLE001
        return False, f"header.json 无法解析: {e}"
    blocks_dir = plan_root / "blocks"
    if not blocks_dir.exists() or not any(blocks_dir.glob("block_*.npz")):
        return False, "blocks/ 目录缺失或为空"
    cs = plan_root / "checksums.json"
    if not cs.exists():
        return False, "checksums.json 不存在"
    for key in ("world_size", "total_steps"):
        if key not in header:
            return False, f"header.json 缺少 {key}(§9.6 必含字段)"
    return True, "header.json / blocks/ / checksums.json 均存在且 header 含基本字段"


def expected_block_filenames(H: int, block_steps: int) -> List[str]:
    nblocks = max(1, math.ceil(H / block_steps)) if block_steps > 0 else 1
    return [f"block_{i:06d}.npz" for i in range(nblocks)]


def block_lengths(H: int, block_steps: int) -> List[int]:
    names = expected_block_filenames(H, block_steps)
    nblocks = len(names)
    lengths = []
    for i in range(nblocks):
        lengths.append(min(block_steps, H - i * block_steps) if block_steps > 0 else H)
    return lengths


# ============================================================================
# §1 独立重算:D[d];可行性必要条件 F0(零 cell)/F1(§9.2 修订)/F2
# ============================================================================


def recompute_D(input_data: Dict[str, Any]) -> np.ndarray:
    # §9.9 已强制 Σm=1;largest_remainder 内部除以 sum,数值上等价
    return largest_remainder(input_data["M"], input_data["m"])


def feasibility_check(input_data: Dict[str, Any], report: Report) -> Dict[str, Any]:
    """独立重实现可行性必要条件(§9.2 口径,不依赖、不 import 任何 planner 代码)。

    §9.2:solver 不可行判定是权威;F1/F2 只为更早更好的诊断。验证器 V7 用
    F1(修订版)+ F2 + 零 cell 必要条件独立判。
    """
    domains = input_data["domains"]
    sq_only = set(input_data["square_only_domains"])
    N = input_data["N"]  # [n_bucket, n_domain]
    R_max = input_data["R_max"]
    sq_idx = input_data["square_bucket_idx"]

    D = recompute_D(input_data)
    D_by_domain = {d: int(D[i]) for i, d in enumerate(domains)}

    # F1(§9.2 修订版):square-only 域只能吃 square 桶 → D[d] ≤ R_max[d]·N[square,d]
    f1_detail = {}
    f1_pass = True
    for i, d in enumerate(domains):
        if d not in sq_only:
            continue
        supply = int(R_max[d]) * int(N[sq_idx, i])
        ok = D_by_domain[d] <= supply
        f1_detail[d] = {"D": D_by_domain[d], "supply_square_cell": supply, "pass": ok}
        f1_pass = f1_pass and ok

    # F2: ∀d: D[d] <= Σ_B R_max[d]·N[B,d]
    f2_detail = {}
    f2_pass = True
    for i, d in enumerate(domains):
        supply = int(R_max[d]) * int(N[:, i].sum())
        ok = D_by_domain[d] <= supply
        f2_detail[d] = {"D": D_by_domain[d], "supply": supply, "pass": ok}
        f2_pass = f2_pass and ok

    # F0(零 cell 必要条件,§9.2"零 cell 必要条件" / §8 V7"空 cell 需求"):
    # 任何有曝光需求(D[d]>0)的域必须有非空的合法供给 cell——square-only 域看
    # N[square,d],其余看 Σ_B N[B,d]。数值上是 F1/F2 在 supply=0 时的特例,单列出来
    # 是为了诊断信息更直接(命中时报"域 X 无任何可用 item",而非"repeat 上限不足")。
    f0_detail = {}
    f0_pass = True
    for i, d in enumerate(domains):
        if D_by_domain[d] <= 0:
            continue
        eligible = int(N[sq_idx, i]) if d in sq_only else int(N[:, i].sum())
        ok = eligible > 0
        f0_detail[d] = {"D": D_by_domain[d], "eligible_items": eligible, "pass": ok}
        f0_pass = f0_pass and ok

    return {
        "D_recomputed": D_by_domain,
        "F0_zero_cell": {"pass": f0_pass, "per_domain": f0_detail},
        "F1": {"pass": f1_pass, "per_square_only_domain": f1_detail, "rule": "D[d] <= R_max[d]*N[square,d] (§9.2)"},
        "F2": {"pass": f2_pass, "per_domain": f2_detail},
        "feasible": bool(f0_pass and f1_pass and f2_pass),
    }


# ============================================================================
# header 检查(§9.6 精确字段名)
# ============================================================================


def check_header_required(header: Dict[str, Any], report: Report) -> None:
    missing = [k for k in REQUIRED_HEADER_FIELDS if k not in header]
    problems = []
    if missing:
        problems.append(f"缺少 §9.6 必含字段: {missing}")
    if "format_version" in header and header["format_version"] != FORMAT_VERSION_EXPECTED:
        problems.append(f"format_version={header['format_version']!r} != 期望 {FORMAT_VERSION_EXPECTED}")
    if "row_scheme" in header and header["row_scheme"] != ROW_SCHEME_EXPECTED:
        problems.append(f"row_scheme={header['row_scheme']!r} != §9.3/§9.6 钉死的 {ROW_SCHEME_EXPECTED!r}")
    status = "PASS" if not problems else "FAIL"
    report.add(
        "HDR_REQUIRED",
        "header §9.6 必含字段 + row_scheme=virtual_catalog_v1 + format_version",
        status,
        "全部满足" if not problems else "; ".join(problems),
        {"missing": missing},
    )


def check_header_hashes(header: Dict[str, Any], input_path: Path, report: Report) -> None:
    """§9.6: hashes{input_sha256, planner_algo_version} 必含且非空;顺带把 input_sha256
    与 --input 文件的实际 sha256 对比(字节级;不匹配只 WARN——契约未保证验证器拿到的
    input 副本与 planner 消费的字节级相同,语义级绑定由 BIND 检查覆盖,见遗留歧义 R3)。"""
    hashes = header.get("hashes")
    problems = []
    if not isinstance(hashes, dict):
        problems.append("header.hashes 缺失或不是 dict")
        hashes = {}
    for k in ("input_sha256", "planner_algo_version"):
        v = hashes.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            problems.append(f"hashes.{k} 缺失或为空")
    if header.get("seed") is None:
        problems.append("header.seed 缺失")
    status = "PASS" if not problems else "FAIL"
    report.add(
        "HDR_HASH",
        "header hashes{input_sha256, planner_algo_version} + seed 存在性(§9.6)",
        status,
        "齐备" if not problems else "; ".join(problems),
        {"problems": problems},
    )

    declared = hashes.get("input_sha256")
    if isinstance(declared, str) and declared.strip():
        actual = sha256_file(input_path)
        if actual.lower() == declared.lower():
            report.add("HDR_INPUT_SHA", "hashes.input_sha256 与 --input 文件实测 sha256 一致", "PASS", "一致", {})
        else:
            report.add(
                "HDR_INPUT_SHA",
                "hashes.input_sha256 与 --input 文件实测 sha256 一致",
                "WARN",
                "不一致——若 --input 是重新序列化过的副本属预期(语义绑定看 BIND);"
                "若应是同一文件则说明 plan 不是由该 input 构建",
                {"declared": declared, "actual": actual},
            )


def check_input_header_binding(input_data: Dict[str, Any], header: Dict[str, Any], report: Report) -> None:
    problems = []

    def norm(v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        return v

    def cmp(key: str, input_val: Any):
        if key not in header:
            problems.append(f"header 缺少字段 {key}")
            return
        if norm(header[key]) != norm(input_val):
            problems.append(f"{key}: header={header[key]!r} != input={input_val!r}")

    cmp("format_version", input_data["format_version"])
    cmp("plan_id", input_data["plan_id"])
    cmp("seed", input_data["seed"])
    cmp("world_size", input_data["world_size"])
    cmp("per_rank_bs", input_data["per_rank_bs"])
    cmp("total_steps", input_data["total_steps"])
    cmp("block_steps", input_data["block_steps"])
    cmp("domains", input_data["domains"])
    cmp("square_only_domains", input_data["square_only_domains"])
    cmp("square_bucket_idx", input_data["square_bucket_idx"])
    cmp("buckets", [list(b) for b in input_data["buckets"]])
    cmp("R_max", input_data["R_max"])

    if "G" in header and int(header["G"]) != input_data["G"]:
        problems.append(f"G: header={header['G']} != W*bs={input_data['G']}")

    # N_total(§9.6)必须等于从 input N 独立求和的逐域总数(虚拟 catalog 的地基)
    if "N_total" in header:
        expect_nt = input_data["N"].sum(axis=0).tolist()
        got_nt = header["N_total"]
        if isinstance(got_nt, dict):
            got_nt = [got_nt.get(d) for d in input_data["domains"]]
        if [int(x) for x in got_nt] != [int(x) for x in expect_nt]:
            problems.append(f"N_total: header={header['N_total']!r} != Σ_B N[B,d](input)={expect_nt!r}")
    else:
        problems.append("header 缺少字段 N_total")

    if problems:
        report.add(
            "BIND",
            "input<->header 绑定一致性(§9.6 精确字段)",
            "FAIL",
            "plan 与给定 input 不匹配,后续检查结果可能无意义(仍会继续跑,以 header 自身值为准)",
            {"problems": problems},
        )
    else:
        report.add("BIND", "input<->header 绑定一致性(§9.6 精确字段)", "PASS", "header 关键字段与 input 完全一致", {})


def check_header_wsbshg(input_data: Dict[str, Any], header: Dict[str, Any], report: Report) -> None:
    W = input_data["world_size"]
    bs = input_data["per_rank_bs"]
    H = input_data["total_steps"]
    G = input_data["G"]
    problems = []
    if header.get("world_size") != W:
        problems.append(f"world_size header={header.get('world_size')} input={W}")
    if header.get("per_rank_bs") != bs:
        problems.append(f"per_rank_bs header={header.get('per_rank_bs')} input={bs}")
    if header.get("total_steps") != H:
        problems.append(f"total_steps header={header.get('total_steps')} input={H}")
    if "G" in header and int(header["G"]) != G:
        problems.append(f"G header={header['G']} != W*bs={G}")
    status = "PASS" if not problems else "FAIL"
    report.add(
        "HDR_WSBSHG",
        "world_size/bs/H/G 一致性",
        status,
        "一致" if not problems else "; ".join(problems),
        {"W": W, "bs": bs, "H": H, "G": G},
    )


def get_header_SDC(input_data: Dict[str, Any], header: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 header 的 S[B]/D[d]/C[B,d](§9.6 精确 key 名 S/D/C)。
    JSON 值形状(list vs dict)未被 §9 钉死(遗留歧义 R1):list 按 buckets/domains
    顺序对齐;dict 允许按 domain 名(D/R_max)或桶索引字符串(S)取值。"""
    n_b = len(input_data["buckets"])
    n_d = len(input_data["domains"])
    for key in ("S", "D", "C"):
        if key not in header:
            raise ValueError(f"header 缺少 §9.6 必含字段 {key}")
    S_raw, D_raw, C_raw = header["S"], header["D"], header["C"]
    if isinstance(S_raw, dict):
        S = np.asarray([S_raw[str(i)] if str(i) in S_raw else S_raw[i] for i in range(n_b)], dtype=np.int64)
    else:
        S = np.asarray(S_raw, dtype=np.int64)
    if isinstance(D_raw, dict):
        D = np.asarray([D_raw[d] for d in input_data["domains"]], dtype=np.int64)
    else:
        D = np.asarray(D_raw, dtype=np.int64)
    C = np.asarray(C_raw, dtype=np.int64)
    if S.shape != (n_b,):
        raise ValueError(f"header S[B] 形状 {S.shape} != ({n_b},)")
    if D.shape != (n_d,):
        raise ValueError(f"header D[d] 形状 {D.shape} != ({n_d},)")
    if C.shape != (n_b, n_d):
        raise ValueError(f"header C[B,d] 形状 {C.shape} != ({n_b},{n_d})")
    return S, D, C


def check_V3_and_constraints(input_data: Dict[str, Any], header: Dict[str, Any], feas: Dict[str, Any], report: Report) -> None:
    domains = input_data["domains"]
    buckets = input_data["buckets"]
    N = input_data["N"]
    R_max = input_data["R_max"]
    G = input_data["G"]
    H = input_data["total_steps"]
    sq_only = set(input_data["square_only_domains"])
    sq_idx = input_data["square_bucket_idx"]

    S, D, C = get_header_SDC(input_data, header)

    # V3-a: Σ_B S[B] = H
    sum_S = int(S.sum())
    ok_c1 = sum_S == H
    report.add(
        "V3_C1", "C1: Σ_B S[B] = H", "PASS" if ok_c1 else "FAIL",
        f"Σ S[B]={sum_S}, H={H}", {"sum_S": sum_S, "H": H},
    )

    # V3-b: Σ_d C[B,d] = G*S[B]  ∀B
    row_sum = C.sum(axis=1)
    expect = G * S
    bad_rows = np.where(row_sum != expect)[0]
    report.add(
        "V3_C2", "C2: Σ_d C[B,d] = G·S[B] ∀B", "PASS" if bad_rows.size == 0 else "FAIL",
        "所有桶满足" if bad_rows.size == 0 else f"{bad_rows.size} 个桶不满足(示例前 10 个: {bad_rows[:10].tolist()})",
        {"n_bad": int(bad_rows.size), "bad_bucket_idx_sample": bad_rows[:20].tolist()},
    )

    # V3-c: Σ_B C[B,d] = D[d] = header 自报 D[d]
    col_sum = C.sum(axis=0)
    bad_cols = np.where(col_sum != D)[0]
    report.add(
        "V3_C3", "C3: Σ_B C[B,d] = header D[d] ∀d", "PASS" if bad_cols.size == 0 else "FAIL",
        "一致" if bad_cols.size == 0 else f"域 {[domains[i] for i in bad_cols]} 不一致",
        {"col_sum": col_sum.tolist(), "header_D": D.tolist()},
    )

    # V3-d: header D[d] 与本文件独立按 §1 公式 LR(M·m[d]) 重算的 D 是否一致
    D_recomp = np.asarray([feas["D_recomputed"][d] for d in domains], dtype=np.int64)
    mism = np.where(D != D_recomp)[0]
    report.add(
        "V3_D_MATCH",
        "各域总曝光 = header D[d] 且 = 独立重算 LR(M·m[d])",
        "PASS" if mism.size == 0 else "FAIL",
        "一致" if mism.size == 0 else f"域 {[domains[i] for i in mism]} 不一致: header={D[mism].tolist()} recompute={D_recomp[mism].tolist()}",
        {"header_D": D.tolist(), "recomputed_D": D_recomp.tolist()},
    )

    # C4: cell 应为 0 的情形(N[B,d]=0,或 d square_only 且 B!=square)
    zero_mask = (N == 0)
    for i, d in enumerate(domains):
        if d in sq_only:
            zero_mask[:, i] = True
            zero_mask[sq_idx, i] = (N[sq_idx, i] == 0)
    violate_c4 = np.where(zero_mask & (C != 0))
    n_v4 = len(violate_c4[0])
    report.add(
        "C4",
        "C4: 应为零的 cell 曝光确为 0(含 square-only 硬约束)",
        "PASS" if n_v4 == 0 else "FAIL",
        "满足" if n_v4 == 0 else f"{n_v4} 个 cell 违反,示例: " + ", ".join(
            f"(B={b},d={domains[dd]},C={int(C[b,dd])})" for b, dd in list(zip(*violate_c4))[:10]
        ),
        {"n_violations": int(n_v4)},
    )

    # V2: square-only 违规数(即 C4 中专门限定 square_only 子集的那部分)——独立报告一次,
    # 与设计文档 §8 的编号对齐。
    sq_cols = [i for i, d in enumerate(domains) if d in sq_only]
    if sq_cols:
        non_sq_rows = [b for b in range(len(buckets)) if b != sq_idx]
        sub = C[np.ix_(non_sq_rows, sq_cols)]
        n_sq_violation = int(np.count_nonzero(sub))
        report.add(
            "V2", "square-only 违规数 = 0", "PASS" if n_sq_violation == 0 else "FAIL",
            f"非 square 桶中 square-only 域曝光合计非零 cell 数 = {n_sq_violation}",
            {"n_violations": n_sq_violation},
        )
    else:
        report.add("V2", "square-only 违规数 = 0", "SKIP", "input 未声明 square_only_domains", {})

    # C5: C[B,d] <= R_max[d]*N[B,d]
    Rvec = np.asarray([R_max[d] for d in domains], dtype=np.int64)
    cap = N * Rvec[None, :]
    violate_c5 = np.where(C > cap)
    n_v5 = len(violate_c5[0])
    report.add(
        "C5", "C5: C[B,d] <= R_max[d]·N[B,d]", "PASS" if n_v5 == 0 else "FAIL",
        "满足" if n_v5 == 0 else f"{n_v5} 个 cell 超出 repeat 上限",
        {"n_violations": int(n_v5)},
    )

    # C6(§1 给出的等价形式): ceil(C[B,d]/S[B]) <= N[B,d](S[B]=0 时要求 C[B,d]=0)。
    # 逐 step 的精确形式(c_t[d] ≤ N[B_t,d])在 deep_scan 的 C6_STEP 项里用
    # domain_counts(或 §9.4 重放)逐步核。
    n_v6 = 0
    v6_examples = []
    for b in range(len(buckets)):
        if S[b] == 0:
            bad = np.where(C[b] != 0)[0]
            if bad.size:
                n_v6 += int(bad.size)
                v6_examples.extend((b, domains[i]) for i in bad[:3])
            continue
        per_step_max = -(-C[b].astype(np.int64) // int(S[b]))  # ceil div
        bad = np.where(per_step_max > N[b])[0]
        if bad.size:
            n_v6 += int(bad.size)
            v6_examples.extend((b, domains[i]) for i in bad[:3])
    report.add(
        "C6",
        "C6: per-step 可行 ceil(C[B,d]/S[B]) <= N[B,d](聚合形式;逐步形式见 C6_STEP)",
        "PASS" if n_v6 == 0 else "FAIL",
        "满足" if n_v6 == 0 else f"{n_v6} 处违反,示例: {v6_examples[:10]}",
        {"n_violations": n_v6},
    )


# ============================================================================
# checksums.json 核对(§9.7 钉死结构:{"header": hex, "blocks": {filename: hex}})
# ============================================================================


def check_checksums(plan_root: Path, filenames: Sequence[str], report: Report) -> None:
    blocks_dir = plan_root / "blocks"
    path = plan_root / "checksums.json"
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception as e:  # noqa: BLE001
        report.add("CHECKSUMS", "checksums.json 核对(§9.7)", "ERROR", f"无法读取/解析 checksums.json: {e}", {})
        return

    structure_problems = []
    if not isinstance(raw, dict):
        structure_problems.append("顶层不是 dict")
    else:
        if "header" not in raw or not isinstance(raw.get("header"), str):
            structure_problems.append('缺少 "header": sha256hex 字段')
        if "blocks" not in raw or not isinstance(raw.get("blocks"), dict):
            structure_problems.append('缺少 "blocks": {filename: sha256hex} 字段')
    if structure_problems:
        report.add(
            "CHECKSUMS", "checksums.json 核对(§9.7)", "FAIL",
            "结构不符合 §9.7 钉死格式: " + "; ".join(structure_problems),
            {"structure_problems": structure_problems},
        )
        return

    extra_top_keys = sorted(set(raw.keys()) - {"header", "blocks"})
    blocks_map: Dict[str, Any] = raw["blocks"]

    header_hash_ok = None
    header_actual = None
    hp = plan_root / "header.json"
    if hp.exists():
        header_actual = sha256_file(hp)
        header_hash_ok = header_actual.lower() == str(raw["header"]).lower()

    mismatches = []
    missing_on_disk = []
    missing_in_manifest = []
    for fn in filenames:
        p = blocks_dir / fn
        if not p.exists():
            missing_on_disk.append(fn)
            continue
        if fn not in blocks_map:
            missing_in_manifest.append(fn)
            continue
        actual = sha256_file(p)
        if actual.lower() != str(blocks_map[fn]).lower():
            mismatches.append({"file": fn, "expected": blocks_map[fn], "actual": actual})

    extra_in_manifest = sorted(set(blocks_map.keys()) - set(filenames))
    extra_files = sorted(set(p.name for p in blocks_dir.glob("block_*.npz")) - set(filenames)) if blocks_dir.exists() else []

    ok = (
        not mismatches
        and not missing_on_disk
        and not missing_in_manifest
        and not extra_in_manifest
        and header_hash_ok is True
    )
    msg_parts = []
    if header_hash_ok is not True:
        msg_parts.append("header.json 的 sha256 与 checksums['header'] 不符" if header_hash_ok is False else "header.json 缺失,无法核对其 sha256")
    if mismatches:
        msg_parts.append(f"{len(mismatches)} 块 sha256 不匹配")
    if missing_on_disk:
        msg_parts.append(f"{len(missing_on_disk)} 块文件缺失")
    if missing_in_manifest:
        msg_parts.append(f"{len(missing_in_manifest)} 块未出现在 checksums['blocks']")
    if extra_in_manifest:
        msg_parts.append(f"checksums['blocks'] 含 {len(extra_in_manifest)} 个非预期条目")
    status = "PASS" if ok else "FAIL"
    report.add(
        "CHECKSUMS",
        "checksums.json 核对(§9.7:header sha256 + 逐块 sha256)",
        status,
        "header 与全部块 sha256 一致" if ok else "; ".join(msg_parts),
        {
            "n_expected_blocks": len(filenames),
            "header_hash_ok": header_hash_ok,
            "header_actual_sha256": header_actual,
            "mismatches": mismatches[:20],
            "missing_on_disk": missing_on_disk[:20],
            "missing_in_manifest": missing_in_manifest[:20],
            "extra_in_manifest": extra_in_manifest[:20],
            "extra_files_not_expected": extra_files[:20],
            "extra_top_level_keys": extra_top_keys,
        },
    )


# ============================================================================
# 虚拟 catalog 行号(§9.3):row_id = base[d] + cellbase[d][B] + local_idx
# ============================================================================


def build_virtual_catalog(N: np.ndarray) -> Dict[str, Any]:
    """由 input N[B,d] 独立构造虚拟 catalog 行号布局(§9.3,仅由 N 派生,无需真 catalog)。

    返回 d-major、B-minor 的扁平 cell 起点数组 starts[n_d*n_b](非降),cell k 的
    (d,B) = (k // n_b, k % n_b)。零宽 cell 与后继 cell 共享起点,searchsorted
    side='right' 天然落到该起点上"最后一个 start<=row 的 cell",即正确的非空 cell。
    """
    n_b, n_d = N.shape
    N_total_d = N.sum(axis=0).astype(np.int64)  # (n_d,)
    base_d = np.zeros(n_d, dtype=np.int64)
    if n_d > 1:
        base_d[1:] = np.cumsum(N_total_d)[:-1]
    starts = np.empty(n_d * n_b, dtype=np.int64)
    for d in range(n_d):
        cb = np.zeros(n_b, dtype=np.int64)
        if n_b > 1:
            cb[1:] = np.cumsum(N[:, d])[:-1]
        starts[d * n_b:(d + 1) * n_b] = base_d[d] + cb
    return {
        "starts": starts,
        "base_d": base_d,
        "N_total_d": N_total_d,
        "total": int(N_total_d.sum()),
        "n_b": n_b,
        "n_d": n_d,
    }


def replay_ck_9_4(C_b: np.ndarray, S_b: int, k: int, consumed_b: np.ndarray, G: int) -> Tuple[np.ndarray, bool]:
    """按 §3 Bresenham + §9.4 精确 ±1 修正规则重放桶 B 第 k 步的 c_k[d]。

    §9.4:若 Σ_d c_k[d] ≠ G,逐单位调整:每次挑「剩余配额 C_rem[B,d] 最大」的域 ±1
    (平局按域索引升序),重复至相等。本实现取 C_rem = C[B,d] − (含本步暂定 c_k 的累计
    消费);−1 方向额外守护 c_k[d]>0(文档未写明,见遗留歧义 R2)。
    返回 (c_k, ok);ok=False 表示重放无法收敛到合法向量(标 ambiguous,跳过该步逐 item 判定)。
    """
    Cf = C_b.astype(np.float64)
    ck = (np.floor(Cf * k / S_b) - np.floor(Cf * (k - 1) / S_b)).astype(np.int64)
    delta = int(G - ck.sum())
    if delta != 0:
        C_rem = C_b - (consumed_b + ck)
        guard = 0
        while delta > 0 and guard < 4 * G:
            d = int(np.argmax(C_rem))  # np.argmax 平局取最小索引 = §9.4 域索引升序
            ck[d] += 1
            C_rem[d] -= 1
            delta -= 1
            guard += 1
        while delta < 0 and guard < 4 * G:
            cand = np.where(ck > 0)[0]
            if cand.size == 0:
                break
            d = int(cand[np.argmax(C_rem[cand])])
            ck[d] -= 1
            C_rem[d] += 1
            delta += 1
            guard += 1
    ok = int(ck.sum()) == G and bool(np.all(ck >= 0))
    return ck, ok


# ============================================================================
# 结构 + 深度逐 item 扫描
# (V1, DC_STRUCT/C6_STEP/DC_AGG/DOMAIN_COMP, V4, VALUE_RANGE, ITEM_BUCKET, SQ_LEAK)
# ============================================================================


def load_block_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """严格按 §9.8/§6 键名读取:bucket_of_step / item_row 必含,domain_counts 可选。"""
    with np.load(path) as npz:
        files = set(npz.files)
        if "bucket_of_step" not in files:
            raise ValueError(f"{path.name} 缺少数据集 'bucket_of_step'(§9.8 钉死键名)")
        if "item_row" not in files:
            raise ValueError(f"{path.name} 缺少数据集 'item_row'(§9.8 钉死键名)")
        bos = npz["bucket_of_step"]
        row = npz["item_row"]
        dc = npz["domain_counts"] if "domain_counts" in files else None
        extra = sorted(files - {"bucket_of_step", "item_row", "domain_counts"})
    notes = [f"{path.name}: 含非契约数据集 {extra}"] if extra else []
    return bos, row, dc, notes


def deep_scan(
    input_data: Dict[str, Any],
    header: Dict[str, Any],
    plan_root: Path,
    filenames: Sequence[str],
    lengths: Sequence[int],
    max_blocks: Optional[int],
    skip_deep: bool,
    report: Report,
) -> Dict[str, Any]:
    domains = input_data["domains"]
    n_b = len(input_data["buckets"])
    n_d = len(domains)
    N = input_data["N"]
    W = input_data["world_size"]
    bs = input_data["per_rank_bs"]
    G = input_data["G"]
    sq_idx = input_data["square_bucket_idx"]
    sq_only_mask = np.asarray([d in set(input_data["square_only_domains"]) for d in domains], dtype=bool)

    S, D, C = get_header_SDC(input_data, header)
    vc = build_virtual_catalog(N)
    starts, vc_total = vc["starts"], vc["total"]

    # header 可选 block_datasets(2026-07-10 增补;不在 §9.6 必含清单里,遗留歧义 R4):
    # 若声明,则以它为准判断 domain_counts 是否必须存在;未声明则由首个块的实际内容决定,
    # 全部块必须一致。
    block_datasets = header.get("block_datasets")
    dc_expected: Optional[bool] = None
    if isinstance(block_datasets, list):
        dc_expected = "domain_counts" in block_datasets

    # §9.5 发牌公式:全局位置 p → rank = p mod W, slot = p div W
    idx_flat = np.arange(G)
    rank_of_flat = idx_flat % W
    pos_of_flat = idx_flat // W
    dom_idx = np.arange(n_d)

    running_k = np.zeros(n_b, dtype=np.int64)
    consumed = np.zeros((n_b, n_d), dtype=np.int64)  # 仅 fallback 重放路径使用
    S_realized = np.zeros(n_b, dtype=np.int64)
    C_realized = np.zeros((n_b, n_d), dtype=np.int64)  # domain_counts 聚合

    n_steps_seen = 0
    n_steps_ambiguous = 0  # 仅 fallback 重放失败时出现

    shape_problems: List[str] = []
    parse_notes: List[str] = []

    n_dup_total = 0
    n_range_total = 0
    n_bucket_mismatch = 0
    n_sq_leak = 0
    n_comp_mismatch = 0
    n_c6_step_viol = 0
    n_dc_struct_viol = 0
    dup_examples: List[Dict[str, Any]] = []
    range_examples: List[Dict[str, Any]] = []
    bucket_examples: List[Dict[str, Any]] = []
    leak_examples: List[Dict[str, Any]] = []
    comp_examples: List[Dict[str, Any]] = []
    c6_examples: List[Dict[str, Any]] = []
    dc_struct_examples: List[str] = []

    dc_presence: List[bool] = []
    dc_path: Optional[str] = None  # "domain_counts" | "bresenham_replay_9_4"

    limit = len(filenames) if max_blocks is None else min(max_blocks, len(filenames))
    scanned_files = filenames[:limit]
    full_scan = limit == len(filenames)

    for bi, fname in enumerate(scanned_files):
        path = plan_root / "blocks" / fname
        if not path.exists():
            shape_problems.append(f"{fname} 缺失,跳过该块的深度检查")
            continue
        expected_len = lengths[bi]
        try:
            bos, row, dc, notes = load_block_arrays(path)
        except Exception as e:  # noqa: BLE001
            shape_problems.append(f"{fname}: 加载失败 {e}")
            continue
        parse_notes.extend(notes)

        # --- 结构 + dtype(§9.8 严格:违规 = FAIL,不再降级为 WARN)---
        if bos.shape != (expected_len,):
            shape_problems.append(f"{fname}: bucket_of_step 形状 {bos.shape} != ({expected_len},)")
            continue
        if bos.dtype != np.int16:
            shape_problems.append(f"{fname}: bucket_of_step dtype={bos.dtype} != int16(§9.8)")
        if row.shape != (expected_len, W, bs):
            shape_problems.append(f"{fname}: item_row 形状 {row.shape} != ({expected_len},{W},{bs})")
            continue
        if row.dtype != np.int64:
            shape_problems.append(f"{fname}: item_row dtype={row.dtype} != int64(§9.8)")

        dc_presence.append(dc is not None)
        if dc_expected is True and dc is None:
            shape_problems.append(f"{fname}: header.block_datasets 声明 domain_counts 但块内缺失")
        if dc_expected is False and dc is not None:
            shape_problems.append(f"{fname}: 块含 domain_counts 但 header.block_datasets 未声明")
        if dc is not None:
            if dc_path is None:
                dc_path = "domain_counts"
            if dc.shape != (expected_len, n_d):
                n_dc_struct_viol += 1
                dc_struct_examples.append(f"{fname}: domain_counts 形状 {dc.shape} != ({expected_len},{n_d})")
                dc = None
            elif dc.dtype != np.int32:
                n_dc_struct_viol += 1
                dc_struct_examples.append(f"{fname}: domain_counts dtype={dc.dtype} != int32(§6)")

        bad_bucket_vals = np.where((bos < 0) | (bos >= n_b))[0]
        if bad_bucket_vals.size:
            shape_problems.append(f"{fname}: {bad_bucket_vals.size} 个 step 的 bucket_of_step 越界 [0,{n_b})")

        bos64 = bos.astype(np.int64)
        valid_step = (bos64 >= 0) & (bos64 < n_b)
        bos_safe = np.where(valid_step, bos64, 0)

        # --- domain_counts 块级校验(a)(b)(d):不依赖逐 item 扫描,skip_deep 时也执行 ---
        if dc is not None:
            neg = int(np.count_nonzero(dc < 0))
            if neg:
                n_dc_struct_viol += neg
                dc_struct_examples.append(f"{fname}: domain_counts 含 {neg} 个负值")
            row_sums = dc.sum(axis=1, dtype=np.int64)
            bad_sum = np.where(row_sums != G)[0]
            if bad_sum.size:
                n_dc_struct_viol += int(bad_sum.size)
                dc_struct_examples.append(
                    f"{fname}: {bad_sum.size} 个 step 的 Σ_d domain_counts != G(示例 step {bad_sum[:5].tolist()})"
                )
            # (b) per-step per-cell:c_t[d] ≤ N[B_t,d](§1 C6 逐步形式)
            viol = (dc.astype(np.int64) > N[bos_safe]) & valid_step[:, None]
            nv = int(np.count_nonzero(viol))
            if nv:
                n_c6_step_viol += nv
                js, ds = np.where(viol)
                for j, di in list(zip(js, ds))[:5]:
                    if len(c6_examples) < 20:
                        c6_examples.append(
                            {"file": fname, "step_in_block": int(j), "bucket": int(bos_safe[j]),
                             "domain": domains[int(di)], "c": int(dc[j, di]), "N": int(N[bos_safe[j], di])}
                        )
            # (d) 聚合:Σ_k c_k[d] 按桶分组
            np.add.at(C_realized, bos_safe[valid_step], dc[valid_step].astype(np.int64))

        # --- 逐 step 处理(桶步计数 + 逐 item 判定)---
        n_steps_seen += int(np.count_nonzero(valid_step))
        np.add.at(S_realized, bos_safe[valid_step], 1)

        if skip_deep:
            # 仍需推进 running_k 语义?fallback 重放只在非 skip_deep 用;跳过即可。
            continue

        # 整块 item 解码(§9.3 虚拟 catalog;§9.5 发牌序)
        flat_block = row[:, rank_of_flat, pos_of_flat]  # (L, G),flat 顺序 p=0..G-1
        if vc_total > 0:
            inrange = (flat_block >= 0) & (flat_block < vc_total)
            kcell = np.searchsorted(starts, np.clip(flat_block, 0, vc_total - 1), side="right") - 1
            d_dec = kcell // n_b
            B_dec = kcell % n_b
        else:
            inrange = np.zeros_like(flat_block, dtype=bool)
            d_dec = np.zeros_like(flat_block)
            B_dec = np.zeros_like(flat_block)

        # VALUE_RANGE:行号 ∈ [0, Σ N_total)
        bad_range = (~inrange) & valid_step[:, None]
        nr = int(np.count_nonzero(bad_range))
        if nr:
            n_range_total += nr
            js, ps = np.where(bad_range)
            for j, p in list(zip(js, ps))[:5]:
                if len(range_examples) < 20:
                    range_examples.append(
                        {"file": fname, "step_in_block": int(j), "flat_pos": int(p),
                         "value": int(flat_block[j, p]), "valid_range": [0, vc_total]}
                    )

        # ITEM_BUCKET(§9.3):item 解码出的桶必须 = step 的桶
        bmis = inrange & (B_dec != bos_safe[:, None]) & valid_step[:, None]
        nb = int(np.count_nonzero(bmis))
        if nb:
            n_bucket_mismatch += nb
            js, ps = np.where(bmis)
            for j, p in list(zip(js, ps))[:5]:
                if len(bucket_examples) < 20:
                    bucket_examples.append(
                        {"file": fname, "step_in_block": int(j), "step_bucket": int(bos_safe[j]),
                         "item_bucket": int(B_dec[j, p]), "item_domain": domains[int(d_dec[j, p])],
                         "value": int(flat_block[j, p])}
                    )

        # SQ_LEAK:square-only 域 item 出现在非 square 桶的 step
        leak = inrange & sq_only_mask[d_dec] & (bos_safe[:, None] != sq_idx) & valid_step[:, None]
        nl = int(np.count_nonzero(leak))
        if nl:
            n_sq_leak += nl
            js, ps = np.where(leak)
            for j, p in list(zip(js, ps))[:5]:
                if len(leak_examples) < 20:
                    leak_examples.append(
                        {"file": fname, "step_in_block": int(j), "step_bucket": int(bos_safe[j]),
                         "item_domain": domains[int(d_dec[j, p])], "value": int(flat_block[j, p])}
                    )

        # V4:同一 global step 内无重复(虚拟 catalog 行号全局唯一 → 扁平去重充分且必要)
        srt = np.sort(flat_block, axis=1)
        dup_counts = (srt[:, 1:] == srt[:, :-1]).sum(axis=1)
        dup_counts = np.where(valid_step, dup_counts, 0)
        nd_ = int(dup_counts.sum())
        if nd_:
            n_dup_total += nd_
            for j in np.where(dup_counts > 0)[0][:5]:
                if len(dup_examples) < 20:
                    dup_examples.append(
                        {"file": fname, "step_in_block": int(j), "n_values": G,
                         "n_unique": int(G - dup_counts[j])}
                    )

        # DOMAIN_COMP + (fallback 时的 C6_STEP):需要每 step 的 c_t[d]
        for j in range(expected_len):
            if not valid_step[j]:
                continue
            b = int(bos_safe[j])
            k = int(running_k[b]) + 1
            running_k[b] = k

            if dc is not None:
                c_vec = dc[j].astype(np.int64)
                if int(c_vec.sum()) != G or np.any(c_vec < 0):
                    continue  # 已计入 DC_STRUCT,不再做组成比对
            else:
                if dc_path is None:
                    dc_path = "bresenham_replay_9_4"
                Sb = int(S[b])
                if Sb <= 0:
                    n_steps_ambiguous += 1
                    continue
                c_vec, ok = replay_ck_9_4(C[b], Sb, k, consumed[b], G)
                if not ok:
                    n_steps_ambiguous += 1
                    continue
                consumed[b] += c_vec
                # fallback 路径的 per-step C6(dc 路径已在块级做过)
                viol = np.where(c_vec > N[b])[0]
                if viol.size:
                    n_c6_step_viol += int(viol.size)
                    for di in viol[:3]:
                        if len(c6_examples) < 20:
                            c6_examples.append(
                                {"file": fname, "step_in_block": int(j), "bucket": b,
                                 "domain": domains[int(di)], "c": int(c_vec[di]), "N": int(N[b, di]),
                                 "path": "replay"}
                            )

            expected_d = np.repeat(dom_idx, c_vec)
            mism = (d_dec[j] != expected_d) & inrange[j]
            nm = int(np.count_nonzero(mism))
            if nm:
                n_comp_mismatch += nm
                p = int(np.where(mism)[0][0])
                if len(comp_examples) < 20:
                    comp_examples.append(
                        {"file": fname, "step_in_block": int(j), "flat_pos": p,
                         "expected_domain": domains[int(expected_d[p])],
                         "decoded_domain": domains[int(d_dec[j, p])],
                         "value": int(flat_block[j, p])}
                    )

    # --- domain_counts 路径一致性 ---
    if dc_presence and (dc_expected is None) and (len(set(dc_presence)) > 1):
        shape_problems.append("部分块含 domain_counts、部分块缺失——同一 plan 内必须一致")
    dc_everywhere = bool(dc_presence) and all(dc_presence)
    no_blocks_scanned = not dc_presence
    dc_skip_reason = "未扫描任何块(--max-blocks=0 或块全部缺失)" if no_blocks_scanned else "块内无 domain_counts(fallback 路径)"

    # ===== 报告输出 =====

    v1_ok = not shape_problems
    report.add(
        "V1",
        "每 step 全 rank 同桶 + 每 rank 恰 bs(结构性:bucket_of_step 单值/step,"
        "item_row 形状 [len,W,bs];§9.8 dtype/键名严格)",
        "PASS" if v1_ok else "FAIL",
        "结构一致" if v1_ok else f"{len(shape_problems)} 处结构问题",
        {"problems": shape_problems[:30], "n_blocks_scanned": len(scanned_files), "parse_notes": parse_notes[:20]},
    )

    report.add(
        "DC_PATH",
        "per-step 域配额来源(domain_counts 物化 vs §9.4 Bresenham 重放回退)",
        "INFO",
        f"path={dc_path or 'n/a'} domain_counts_present_in_all_scanned_blocks={dc_everywhere}",
        {"path": dc_path, "dc_presence_per_block": dc_presence[:50], "block_datasets_declared": block_datasets},
    )
    if dc_path == "bresenham_replay_9_4":
        report.note_ambiguity(
            "本 plan 的块内没有 domain_counts 数据集(§6 2026-07-10 增补),per-step 域边界"
            "由 §3 Bresenham + §9.4 修正规则重放重构。§9.4 的 C_rem 计时点(含/不含本步暂定"
            "c_k)与 −1 方向的非负守护未在文档写明(遗留歧义 R2),重放结果与 planner 逐位"
            "一致性无法保证;建议用带 domain_counts 的 planner 版本重建。"
        )

    if dc_everywhere:
        report.add(
            "DC_STRUCT",
            "domain_counts dtype=int32 / 形状 [len,n_domains] / Σ_d=G 每步(增补 §6 (a))",
            "PASS" if n_dc_struct_viol == 0 else "FAIL",
            "全部满足" if n_dc_struct_viol == 0 else f"{n_dc_struct_viol} 处违规",
            {"n_violations": n_dc_struct_viol, "examples": dc_struct_examples[:20]},
        )
    else:
        report.add("DC_STRUCT", "domain_counts 结构校验", "SKIP", dc_skip_reason, {})

    if dc_everywhere or (not skip_deep and dc_path == "bresenham_replay_9_4"):
        report.add(
            "C6_STEP",
            "per-step per-cell c_t[d] <= N[B_t,d](§1 C6 逐步形式;增补 §6 (b))",
            "PASS" if n_c6_step_viol == 0 else "FAIL",
            "满足" if n_c6_step_viol == 0 else f"{n_c6_step_viol} 处违反",
            {"n_violations": n_c6_step_viol, "examples": c6_examples, "source": dc_path},
        )
    else:
        report.add("C6_STEP", "per-step per-cell c_t[d] <= N[B_t,d]", "SKIP", "无 domain_counts 且逐 item 扫描被跳过", {})

    if dc_everywhere and full_scan:
        bad_agg = np.where(C_realized != C)
        n_bad_agg = len(bad_agg[0])
        report.add(
            "DC_AGG",
            "Σ_k domain_counts(按桶分组)= header C[B,d](增补 §6 (d))",
            "PASS" if n_bad_agg == 0 else "FAIL",
            "一致" if n_bad_agg == 0 else f"{n_bad_agg} 个 cell 不一致,示例: " + ", ".join(
                f"(B={b},d={domains[dd]},Σc={int(C_realized[b,dd])},C={int(C[b,dd])})"
                for b, dd in list(zip(*bad_agg))[:5]
            ),
            {"n_violations": int(n_bad_agg)},
        )
    elif dc_everywhere:
        report.add("DC_AGG", "Σ_k domain_counts = header C[B,d]", "SKIP", "--max-blocks 只扫了部分块,聚合不完整", {})
    else:
        report.add("DC_AGG", "Σ_k domain_counts = header C[B,d]", "SKIP", dc_skip_reason, {})

    if not skip_deep and full_scan:
        bad_S = np.where(S_realized != S)[0]
        report.add(
            "V1_S_REALIZED",
            "各桶实际出现步数(bucket_of_step 统计) = header S[B]",
            "PASS" if bad_S.size == 0 else "FAIL",
            "一致" if bad_S.size == 0 else f"{bad_S.size} 个桶不一致,示例: {bad_S[:10].tolist()}",
            {"S_header": S.tolist(), "S_realized": S_realized.tolist()},
        )
    else:
        report.add("V1_S_REALIZED", "各桶实际出现步数 = header S[B]", "SKIP", "深度扫描被跳过或只扫了部分块,不足以核对全量", {})

    if skip_deep:
        for code, name in (
            ("V4", "同一 global step 内 item_row 无重复"),
            ("VALUE_RANGE", "item_row 值域合法性(虚拟 catalog)"),
            ("ITEM_BUCKET", "item 解码桶 = step 桶(§9.3)"),
            ("SQ_LEAK", "square-only 域 item 仅出现在 square 桶(item 级防御)"),
            ("DOMAIN_COMP", "item_row 域组成与 c_t[d] 一致(增补 §6 (c))"),
        ):
            report.add(code, name, "SKIP", "--skip-deep-item-checks 已启用", {})
    else:
        report.add(
            "V4",
            "同一 global step 内 item_row 无重复(虚拟 catalog 行号全局唯一 → 扁平去重充分必要)",
            "PASS" if n_dup_total == 0 else "FAIL",
            "无重复" if n_dup_total == 0 else f"发现 {n_dup_total} 处重复",
            {"n_duplicate_slots": n_dup_total, "examples": dup_examples, "n_steps_scanned": n_steps_seen},
        )
        report.add(
            "VALUE_RANGE",
            f"item_row 值域 ∈ [0, Σ_d N_total)={vc_total}(§9.3 虚拟 catalog)",
            "PASS" if n_range_total == 0 else "FAIL",
            "全部合法" if n_range_total == 0 else f"发现 {n_range_total} 处越界",
            {"n_out_of_range": n_range_total, "examples": range_examples},
        )
        report.add(
            "ITEM_BUCKET",
            "item 解码桶 = step 桶(§9.3;零 re-resize 的 plan 侧硬保证)",
            "PASS" if n_bucket_mismatch == 0 else "FAIL",
            "全部匹配" if n_bucket_mismatch == 0 else f"发现 {n_bucket_mismatch} 处 item 桶与 step 桶不符",
            {"n_violations": n_bucket_mismatch, "examples": bucket_examples},
        )
        report.add(
            "SQ_LEAK",
            "square-only 域 item 仅出现在 square 桶的 step(item 级防御,理论上被 C4 已拦下)",
            "PASS" if n_sq_leak == 0 else "FAIL",
            "无泄漏" if n_sq_leak == 0 else f"发现 {n_sq_leak} 处泄漏",
            {"n_violations": n_sq_leak, "examples": leak_examples},
        )
        comp_status = "PASS" if n_comp_mismatch == 0 else "FAIL"
        report.add(
            "DOMAIN_COMP",
            f"item_row 域组成与 c_t[d] 逐槽位一致(增补 §6 (c);来源={dc_path or 'n/a'})",
            comp_status,
            "逐槽位一致" if n_comp_mismatch == 0 else f"发现 {n_comp_mismatch} 处槽位域不符",
            {
                "n_violations": n_comp_mismatch,
                "examples": comp_examples,
                "n_steps_ambiguous_skipped": n_steps_ambiguous,
                "source": dc_path,
            },
        )

    if n_steps_ambiguous > 0:
        report.note_ambiguity(
            f"{n_steps_ambiguous}/{n_steps_seen} 个 step 在 fallback 重放(§9.4)下无法收敛到"
            "合法 c_k(Σ=G 且非负),其 DOMAIN_COMP/C6_STEP 判定被跳过;若该 plan 本应带"
            "domain_counts,请检查 planner 版本。"
        )

    return {
        "n_steps_seen": n_steps_seen,
        "n_steps_ambiguous": n_steps_ambiguous,
        "dc_path": dc_path,
        "S_realized": S_realized.tolist(),
    }


# ============================================================================
# V5 / V6: 两个 plan_root 的块级 sha256 比较(§9.10)
# ============================================================================


def compare_plan_roots(root_a: Path, root_b: Path, rebuild_mode: str, report: Report) -> None:
    blocks_a = sorted(p.name for p in (root_a / "blocks").glob("block_*.npz")) if (root_a / "blocks").exists() else []
    blocks_b = sorted(p.name for p in (root_b / "blocks").glob("block_*.npz")) if (root_b / "blocks").exists() else []
    common = sorted(set(blocks_a) & set(blocks_b))
    only_a = sorted(set(blocks_a) - set(blocks_b))
    only_b = sorted(set(blocks_b) - set(blocks_a))

    mismatches = []
    for fn in common:
        ha = sha256_file(root_a / "blocks" / fn)
        hb = sha256_file(root_b / "blocks" / fn)
        if ha != hb:
            mismatches.append({"file": fn, "sha256_a": ha, "sha256_b": hb})

    header_diff = None
    try:
        header_a = json.load(open(root_a / "header.json"))
        header_b = json.load(open(root_b / "header.json"))
        header_diff = header_a == header_b
    except Exception:  # noqa: BLE001
        header_diff = None

    # §9.10:V6 = 双次全构建(块文件名集合必须相同 + 逐块 sha256 相同 + header 逐字段相同);
    # V5 = 从任意 t 起重建后,公共后缀块逐 sha256 相同。auto 模式按块集合是否相同区分两种语境。
    mode = rebuild_mode
    if mode == "auto":
        mode = "full" if (not only_a and not only_b and blocks_a) else "suffix"

    blocks_match = not mismatches and common
    detail = {
        "n_common": len(common),
        "n_mismatch": len(mismatches),
        "mismatches": mismatches[:20],
        "only_in_plan": only_a[:20],
        "only_in_rebuild_plan": only_b[:20],
        "header_identical": header_diff,
        "detected_rebuild_mode": mode,
    }

    v5_status = "PASS" if blocks_match else ("FAIL" if common else "SKIP")
    report.add(
        "V5",
        "任意 t 处重建 → 公共后缀块级 sha256 逐字节一致(§9.10)",
        v5_status,
        "共同块 sha256 全部一致" if v5_status == "PASS" else ("无共同块可比较" if v5_status == "SKIP" else f"{len(mismatches)} 块不一致"),
        detail,
    )

    if mode == "full":
        v6_ok = blocks_match and (header_diff is True)
        v6_status = "PASS" if v6_ok else "FAIL"
        msg = "双构建块与 header 完全一致" if v6_ok else "块或 header 存在差异(见 detail)"
    else:
        v6_status = "SKIP"
        msg = "rebuild-plan 是部分/后缀重建(块文件集合与原 plan 不同),不满足 §9.10 '双次全构建'语境,V6 不评估"
    report.add("V6", "双次全构建 hash 相同(§9.10:块集合相同+逐块 sha256 相同+header 逐字段相同)", v6_status, msg, detail)


# ============================================================================
# V7: 不可行输入检测(独立 F0/F1(修订)/F2 复核 + plan_root 是否"存在")
# ============================================================================


def check_V7(input_data: Dict[str, Any], plan_root: Path, feas: Dict[str, Any], report: Report) -> None:
    complete, reason = plan_looks_complete(plan_root)
    my_feasible = feas["feasible"]

    if not my_feasible and complete:
        report.add(
            "V7",
            "不可行输入检测(独立 F0/F1修订/F2 复核 vs plan_root 是否被物化)",
            "FAIL",
            "本验证器独立复核判定 input 按 §9.2 必要条件不可行,但 plan_root 存在完整已物化的"
            " plan——应在 plan-time 报错而非产出 plan",
            {"my_feasible": my_feasible, "plan_complete": complete, "plan_complete_reason": reason,
             "F0": feas["F0_zero_cell"], "F1": feas["F1"], "F2": feas["F2"]},
        )
    elif not my_feasible and not complete:
        report.add(
            "V7",
            "不可行输入检测(独立 F0/F1修订/F2 复核 vs plan_root 是否被物化)",
            "PASS",
            f"独立复核也判定不可行,且 plan_root 未产出完整 plan(原因: {reason}),与预期的"
            "plan-time 失败一致",
            {"my_feasible": my_feasible, "plan_complete": complete, "plan_complete_reason": reason},
        )
    elif my_feasible and complete:
        report.add(
            "V7",
            "不可行输入检测(独立 F0/F1修订/F2 复核 vs plan_root 是否被物化)",
            "PASS",
            "独立复核判定该 input 可行(必要条件均满足),plan_root 存在完整 plan——本次运行未"
            "实际触发不可行检测路径,V7 在此输入上是空对照",
            {"my_feasible": my_feasible, "plan_complete": complete, "exercised": False},
        )
    else:  # my_feasible and not complete
        report.add(
            "V7",
            "不可行输入检测(独立 F0/F1修订/F2 复核 vs plan_root 是否被物化)",
            "SKIP",
            "独立复核判定必要条件均满足,但 plan_root 未产出完整 plan。§9.2 明确 solver 不可行"
            "判定才是权威(F1/F2 只是诊断),因此这既可能是 planner 的 bug,也可能是真正只在"
            "联合配额层面不可行——不判 FAIL,需要人工复核",
            {"my_feasible": my_feasible, "plan_complete": complete, "plan_complete_reason": reason},
        )


# ============================================================================
# main
# ============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="planner_input.json 路径")
    p.add_argument("--plan", required=True, type=Path, help="待验证的 plan_root/ 目录")
    p.add_argument("--rebuild-plan", type=Path, default=None, help="用于 V5/V6 对比的第二个 plan_root/(可选)")
    p.add_argument("--report", type=Path, default=None, help="把完整 JSON 报告写到此路径(可选)")
    p.add_argument("--rebuild-mode", choices=["auto", "suffix", "full"], default="auto")
    p.add_argument("--max-blocks", type=int, default=None, help="限制深度逐 item 检查扫描的块数(结构/header 检查不受影响)")
    p.add_argument("--skip-deep-item-checks", action="store_true", help="跳过逐 item 深度扫描(V4/VALUE_RANGE/ITEM_BUCKET/SQ_LEAK/DOMAIN_COMP);domain_counts 的块级校验仍执行")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    t0 = time.perf_counter()
    args = build_arg_parser().parse_args(argv)
    report = Report()

    try:
        input_data = load_input(args.input, report)
    except Exception as e:  # noqa: BLE001
        report.add("INPUT", "加载 planner_input.json(§1.5 + §9.9 严格校验)", "ERROR", f"{type(e).__name__}: {e}", {})
        _finalize(report, args, t0)
        return 1
    report.add("INPUT", "加载 planner_input.json(§1.5 + §9.9 严格校验)", "PASS", "输入契约字段齐备且通过 §9.9 校验", {})

    feas = feasibility_check(input_data, report)
    # 注意:这里的"infeasible"不是本验证器自身的缺陷/FAIL——它只是 §9.2 必要条件的独立复核
    # 结果,真正的判定(是否应该 FAIL)在 check_V7() 里结合 plan_root 是否被物化来做。
    report.add(
        "FEASIBILITY",
        "独立重实现可行性必要条件 F0/F1(§9.2 修订)/F2",
        "INFO",
        f"my_feasible={feas['feasible']} (F0.pass={feas['F0_zero_cell']['pass']} "
        f"F1.pass={feas['F1']['pass']} F2.pass={feas['F2']['pass']});"
        " 本身不是 PASS/FAIL 判定,供 V7 使用",
        feas,
    )

    header = None
    if not args.plan.exists():
        # plan_root 整个目录不存在,这是 V7 负面测试(构造不可行 input,期待 planner
        # 拒绝产出)的正常形态之一——是否"合理"由下面的 check_V7() 判定,这里不算
        # ERROR,只是如实说明没有制品可读。
        report.add(
            "HEADER_LOAD", "加载 plan_root/header.json", "SKIP",
            f"--plan {args.plan} 目录不存在(header.json 无从读取);若这是 V7 不可行输入"
            "负面测试,预期如此,见下面的 V7 判定;若本应存在已构建 plan,请检查 V7 结论",
            {},
        )
    else:
        try:
            header = load_header(args.plan)
            report.add("HEADER_LOAD", "加载 plan_root/header.json", "PASS", "成功解析", {})
        except Exception as e:  # noqa: BLE001
            report.add("HEADER_LOAD", "加载 plan_root/header.json", "ERROR", f"{type(e).__name__}: {e}", {})

    check_V7(input_data, args.plan, feas, report)

    if header is None:
        report.add("ABORT", "后续检查", "SKIP", "header.json 无法加载,跳过所有依赖 header 的检查(V1-V6/checksums/etc.)", {})
        _finalize(report, args, t0)
        return report.exit_code()

    guarded(report, "HDR_REQUIRED", "header 必含字段")(check_header_required)(header, report)
    guarded(report, "BIND", "input<->header 绑定")(check_input_header_binding)(input_data, header, report)
    guarded(report, "HDR_HASH", "header hashes 字段")(check_header_hashes)(header, args.input, report)
    guarded(report, "HDR_WSBSHG", "world_size/bs/H/G 一致性")(check_header_wsbshg)(input_data, header, report)
    guarded(report, "V3", "V3 及 C1-C6")(check_V3_and_constraints)(input_data, header, feas, report)

    # block_steps 是 §9.6 必含字段(缺失已在 HDR_REQUIRED 报 FAIL);块布局推导回退用 input 值
    block_steps = header.get("block_steps", input_data["block_steps"])
    block_steps = int(block_steps)

    H = input_data["total_steps"]
    filenames = expected_block_filenames(H, block_steps)
    lengths = block_lengths(H, block_steps)

    guarded(report, "CHECKSUMS", "checksums.json 核对")(check_checksums)(args.plan, filenames, report)

    t_scan0 = time.perf_counter()
    guarded(report, "DEEP_SCAN", "结构 + 深度逐 item 扫描")(deep_scan)(
        input_data, header, args.plan, filenames, lengths, args.max_blocks, args.skip_deep_item_checks, report
    )
    report.timings["deep_scan_seconds"] = time.perf_counter() - t_scan0

    if args.rebuild_plan is not None:
        guarded(report, "REBUILD_CMP", "V5/V6 双 plan_root 比较")(compare_plan_roots)(args.plan, args.rebuild_plan, args.rebuild_mode, report)
    else:
        report.add("V5", "任意 t 处重建 → 公共后缀块级 sha256 逐字节一致(§9.10)", "SKIP", "未提供 --rebuild-plan", {})
        report.add("V6", "双次全构建 hash 相同(§9.10)", "SKIP", "未提供 --rebuild-plan", {})

    _finalize(report, args, t0)
    return report.exit_code()


def _finalize(report: Report, args: argparse.Namespace, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KB
    report.timings["total_seconds"] = elapsed
    report.add(
        "V8",
        "校验耗时/内存实测(本验证器自身,不含 planner 构建)",
        "INFO",
        f"elapsed={elapsed:.3f}s peak_rss={peak_rss_kb/1024:.1f}MB",
        {"elapsed_seconds": elapsed, "peak_rss_kb": peak_rss_kb, "timings": report.timings},
    )

    overall = report.worst_status()
    print("=" * 88)
    print(f"validate_plan.py 结果总览: overall={overall}  input={args.input}  plan={args.plan}")
    print("=" * 88)
    for item in report.items:
        print(f"[{item['status']:>5}] {item['code']:<16} {item['name']}: {item['message']}")
    if report.ambiguities:
        print("-" * 88)
        print("本次运行触发的设计文档歧义/缺口(详见 README_VALIDATOR.md 完整清单):")
        for i, a in enumerate(report.ambiguities, 1):
            print(f"  {i}. {a}")
    print("=" * 88)

    if args.report is not None:
        out = {
            "overall": overall,
            "input": str(args.input),
            "plan": str(args.plan),
            "rebuild_plan": str(args.rebuild_plan) if args.rebuild_plan else None,
            "checks": report.items,
            "ambiguities_triggered_this_run": report.ambiguities,
            "timings": report.timings,
        }
        with open(args.report, "w") as f:
            json.dump(to_jsonable(out), f, indent=2, ensure_ascii=False, sort_keys=False)
        print(f"完整 JSON 报告已写入: {args.report}")


if __name__ == "__main__":
    sys.exit(main())
