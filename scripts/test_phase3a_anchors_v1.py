#!/usr/bin/env python3
"""phase3a_bucket_stats 的 anchors_v1 路径测试（写临时 parquet，不读真实 manifest）。

运行：`python scripts/test_phase3a_anchors_v1.py`（退出码 0 = 全通过）。

phase3a 的 worker 比 phase5 缠得紧：`keep` 是**布尔掩码**，而 `has_nl_all[keep]` 也读它。
锚模式要在指派**之后**丢掉不可服务行，如果只窄化 `wk/hk/bidx` 而不窄化 `keep`，
anime 的 NL 标志就会和错误的行配对 —— 且 `n_matrix` 与 `anime_*_by_bucket` 的和会不一致，
但两者都不会报错。这里就是钉这件事。
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from training.dataline import anchors_v1

_P3_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "offline_arb", "phase3a_bucket_stats.py")
_spec = importlib.util.spec_from_file_location("_phase3a_under_test", _P3_PATH)
P3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P3)

FAILURES: list[str] = []
TMP = tempfile.mkdtemp(prefix="p3a_anchors_")


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  PASS  " if ok else "  FAIL  ") + name + ("" if ok else f"  {detail}"))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def write_shard(rows, name, holdout=None, nl=None):
    """rows = [(w, h), ...]；nl = 每行 refined_description 是否非空。"""
    cols = {
        "width": pa.array([float(w) for w, _ in rows], type=pa.float64()),
        "height": pa.array([float(h) for _, h in rows], type=pa.float64()),
    }
    if holdout is not None:
        cols["holdout"] = pa.array(holdout, type=pa.int8())
    if nl is not None:
        cols["refined_description"] = pa.array(
            ["a caption" if x else "" for x in nl], type=pa.string())
        cols["medium_description"] = pa.array([""] * len(rows), type=pa.string())
        cols["short_description"] = pa.array([""] * len(rows), type=pa.string())
    path = os.path.join(TMP, name)
    pq.write_table(pa.table(cols), path)
    return path


def task(shard_path, domain="monet_real", is_square_only=False, anchors=True,
         area_gate=anchors_v1.AREA_GATE_STAGE512A, batch_size=1000):
    buckets = (anchors_v1.anchors_hw() if anchors
               else __import__("training.dataline.multi_ar", fromlist=["x"])
               .build_adaptive_buckets(262144, 16, 0.75, 4.0))
    from training.dataline.multi_ar import square_index
    return {
        "domain": domain, "shard_path": shard_path, "is_square_only": is_square_only,
        "elig_always_true": False, "eligibility_column": "elig_512", "image_size": 512,
        "fallback_allowed": True,
        "buckets_h": np.array([h for h, _ in buckets], dtype=np.int64),
        "buckets_w": np.array([w for _, w in buckets], dtype=np.int64),
        "table": P3.build_bucket_table(buckets, 16), "buckets_list": buckets,
        "sq_idx": square_index(buckets), "target_area": 262144, "divisor": 16,
        "no_upscale": True, "max_aspect": 4.0,
        "extreme_long": 1024, "extreme_short": 256, "batch_size": batch_size,
        "bucket_mode": anchors_v1.BUCKET_MODE if anchors else "adaptive",
        "area_gate": area_gate if anchors else None,
        "crop_loss_max": anchors_v1.CROP_LOSS_MAX if anchors else None,
    }


def test_counters_reconcile_and_nmatrix_matches_rows_kept() -> None:
    rows = [
        (767, 512),    # -> 624x416
        (300, 300),    # area 90,000 < gate -> elig drop
        (1008, 435),   # -> 704x352
        (2000, 400),   # AR 5.0, area ok -> unservable
        (640, 640),    # -> 512x512
        (1024, 256),   # area == gate, AR 4.0 -> unservable
    ]
    res = P3.scan_domain_shard(task(write_shard(rows, "a.parquet")))
    check("worker ok", res.get("ok") is True, str(res.get("error")))
    if not res.get("ok"):
        return
    check("rows_raw == 6", res["rows_raw"] == 6, str(res["rows_raw"]))
    check("elig drop == 1", res["rows_elig_dropped"] == 1, str(res["rows_elig_dropped"]))
    check("unservable drop == 2", res["rows_unservable_dropped"] == 2,
          str(res["rows_unservable_dropped"]))
    check("rows_kept == 3", res["rows_kept"] == 3, str(res["rows_kept"]))
    check("n_matrix sums to rows_kept",
          int(res["n_matrix"].sum()) == res["rows_kept"],
          f"{int(res['n_matrix'].sum())} vs {res['rows_kept']}")
    check("counters reconcile (5-term identity)",
          res["rows_raw"] - res["rows_holdout_dropped"] - res["rows_elig_dropped"]
          - res["rows_invalid_wh_dropped"] - res["rows_unservable_dropped"]
          == res["rows_kept"], str({k: v for k, v in res.items() if k.startswith("rows_")}))
    nz = {i: int(res["n_matrix"][i]) for i in range(13) if res["n_matrix"][i]}
    want = {3: 1, 7: 1, 0: 1}
    check("n_matrix lands on the hand-checked anchors", nz == want, f"{nz} vs {want}")
    check("eligibility_mode reports the anchors rule",
          res["eligibility_mode"] == "anchors_v1_area_gate_and_feasible_anchor",
          res["eligibility_mode"])


def test_anime_nl_stays_aligned_after_dropping() -> None:
    """核心用例：不可服务行被丢掉后，NL 标志必须仍然属于正确的行。
    只窄化 wk/hk/bidx 而不窄化 keep 就会在这里错位。"""
    rows = [
        (767, 512),    # servable, NL=True   -> 624x416
        (2000, 400),   # UNSERVABLE, NL=True  (若不窄化 keep，它的 True 会串给下一行)
        (640, 640),    # servable, NL=False  -> 512x512
        (3000, 500),   # UNSERVABLE, NL=False
        (1008, 435),   # servable, NL=True   -> 704x352
    ]
    nl = [True, True, False, False, True]
    res = P3.scan_domain_shard(
        task(write_shard(rows, "anime.parquet", nl=nl), domain="anime"))
    check("anime worker ok", res.get("ok") is True, str(res.get("error")))
    if not res.get("ok"):
        return
    hasnl = res["anime_has_nl_by_bucket"]
    tagonly = res["anime_tag_only_by_bucket"]
    check("unservable drop == 2", res["rows_unservable_dropped"] == 2,
          str(res["rows_unservable_dropped"]))
    check("rows_kept == 3", res["rows_kept"] == 3, str(res["rows_kept"]))
    check("has_nl + tag_only == rows_kept",
          int(hasnl.sum() + tagonly.sum()) == res["rows_kept"],
          f"{int(hasnl.sum())}+{int(tagonly.sum())} vs {res['rows_kept']}")
    # 767x512(NL) -> anchor 3 ; 1008x435(NL) -> anchor 7 ; 640x640(no NL) -> anchor 0
    got_nl = {i: int(hasnl[i]) for i in range(13) if hasnl[i]}
    got_to = {i: int(tagonly[i]) for i in range(13) if tagonly[i]}
    check("NL rows attributed to anchors 3 and 7", got_nl == {3: 1, 7: 1}, str(got_nl))
    check("the non-NL row attributed to anchor 0", got_to == {0: 1}, str(got_to))


def test_square_only_verified_not_forced() -> None:
    """square_only 域也跑真实指派。注意归因：低于面积门的行在指派**之前**就被 elig 丢掉，
    所以它们计入 rows_elig_dropped 而不是 rows_unservable_dropped。
    另有一个几何事实：方图只要过了面积门，边长必然 >= 512（s^2 >= 262144 => s >= 512），
    所以"方但覆盖不了方锚"只能由**近方形**的非方图产生，例如 520x505。"""
    rows = [
        (1024, 1024),   # 方，覆盖得了 512x512 -> 留下
        (800, 600),     # 非方，真实指派落到 4:3；square_only 下判为不可服务
        (400, 400),     # 面积 160,000 < 门 -> elig 丢弃（不是 unservable）
        (520, 505),     # 面积 262,600 >= 门，但 505 < 512 且不够宽 -> 可行集为空
    ]
    res = P3.scan_domain_shard(task(write_shard(rows, "syn.parquet"),
                                    domain="synthetic", is_square_only=True))
    check("synthetic worker ok", res.get("ok") is True, str(res.get("error")))
    if not res.get("ok"):
        return
    check("only the coverable square survives", res["rows_kept"] == 1, str(res["rows_kept"]))
    check("it is on the square anchor", int(res["n_matrix"][0]) == 1, str(res["n_matrix"][:3]))
    check("sub-area-gate row attributed to elig, not unservable",
          res["rows_elig_dropped"] == 1, str(res["rows_elig_dropped"]))
    check("non-square + near-square-uncoverable dropped as unservable",
          res["rows_unservable_dropped"] == 2, str(res["rows_unservable_dropped"]))
    check("no mass in any non-square anchor",
          int(res["n_matrix"][1:].sum()) == 0, str(res["n_matrix"]))
    check("counters reconcile",
          res["rows_raw"] - res["rows_holdout_dropped"] - res["rows_elig_dropped"]
          - res["rows_invalid_wh_dropped"] - res["rows_unservable_dropped"]
          == res["rows_kept"], str({k: v for k, v in res.items() if k.startswith("rows_")}))
    # 独立复核那条几何事实
    bad = [s for s in range(512, 4000, 7)
           if s * s >= anchors_v1.AREA_GATE_STAGE512A and anchors_v1.assign_anchor(s, s) is None]
    check("every square passing the area gate covers the square anchor", not bad, str(bad[:3]))


def test_multi_batch_narrowing() -> None:
    """batch_size 小于行数时，每个 batch 独立窄化，累加仍须自洽。"""
    rows = [(767, 512), (2000, 400)] * 25          # 50 行，一半不可服务
    res = P3.scan_domain_shard(
        task(write_shard(rows, "multi.parquet"), batch_size=7))
    check("multi-batch worker ok", res.get("ok") is True, str(res.get("error")))
    if not res.get("ok"):
        return
    check("25 servable / 25 unservable across batches",
          res["rows_kept"] == 25 and res["rows_unservable_dropped"] == 25,
          f"kept={res['rows_kept']} unserv={res['rows_unservable_dropped']}")
    check("n_matrix still sums to rows_kept",
          int(res["n_matrix"].sum()) == 25, str(int(res["n_matrix"].sum())))


def test_all_rows_unservable_is_not_a_crash() -> None:
    res = P3.scan_domain_shard(task(write_shard([(2000, 400), (1024, 256)], "none.parquet")))
    check("all-unservable shard returns ok", res.get("ok") is True, str(res.get("error")))
    if res.get("ok"):
        check("rows_kept == 0 and n_matrix empty",
              res["rows_kept"] == 0 and int(res["n_matrix"].sum()) == 0, str(res["rows_kept"]))


def test_holdout_still_honoured() -> None:
    rows = [(767, 512), (767, 512), (640, 640)]
    res = P3.scan_domain_shard(
        task(write_shard(rows, "ho.parquet", holdout=[0, 1, 0])))
    check("holdout worker ok", res.get("ok") is True, str(res.get("error")))
    if res.get("ok"):
        check("holdout row excluded", res["rows_holdout_dropped"] == 1, str(res))
        check("rows_kept == 2", res["rows_kept"] == 2, str(res["rows_kept"]))


def test_legacy_adaptive_path_untouched() -> None:
    """adaptive 模式：361 桶、min(w,h)>=512 资格门、不丢不可服务行。"""
    rows = [(767, 512), (696, 510), (640, 640)]
    res = P3.scan_domain_shard(task(write_shard(rows, "legacy.parquet"), anchors=False))
    check("legacy worker ok", res.get("ok") is True, str(res.get("error")))
    if not res.get("ok"):
        return
    check("legacy drops 696x510 via the short-side gate",
          res["rows_elig_dropped"] == 1, str(res["rows_elig_dropped"]))
    check("legacy reports zero unservable", res["rows_unservable_dropped"] == 0,
          str(res["rows_unservable_dropped"]))
    check("legacy rows_kept == 2", res["rows_kept"] == 2, str(res["rows_kept"]))
    check("legacy n_matrix is in the 361 namespace",
          res["n_matrix"].shape[0] == 361 and int(res["n_matrix"].sum()) == 2,
          f"{res['n_matrix'].shape} sum={int(res['n_matrix'].sum())}")
    check("legacy eligibility_mode unchanged",
          res["eligibility_mode"] == "short_dim_fallback", res["eligibility_mode"])


def test_params_from_values_is_the_single_source() -> None:
    """phase5 与 phase3a 的 mar_params 必须由同一个表达式产出，否则等值门永远过不了。"""
    a = anchors_v1.params_from_values(262144, None)
    b = anchors_v1.params_from_values(262144, 0.20)
    check("area_gate normalized", a["area_gate"] == 262144, str(a))
    check("crop_loss_max defaults to the module constant",
          a["crop_loss_max"] == anchors_v1.CROP_LOSS_MAX and a == b, f"{a} {b}")
    check("0 means off", anchors_v1.params_from_values(0, None)["area_gate"] is None)
    for bad in (None,):
        try:
            anchors_v1.params_from_values(bad, None)
            check(f"area_gate={bad} rejected", False, "did not raise")
        except anchors_v1.AnchorGeometryError:
            check(f"area_gate={bad} rejected", True)
    try:
        anchors_v1.params_from_values(-1, None)
        check("negative area_gate rejected", False, "did not raise")
    except anchors_v1.AnchorGeometryError:
        check("negative area_gate rejected", True)
    try:
        anchors_v1.params_from_values(262144, 1.5)
        check("crop_loss_max out of range rejected", False, "did not raise")
    except anchors_v1.AnchorGeometryError:
        check("crop_loss_max out of range rejected", True)


def _fake_report(unservable=61565, break_identity=False):
    """write_markdown_summary 需要的最小完整 report。数字取自 12-shard 诊断实跑。"""
    kept = 11938033 - (7 if break_identity else 0)
    return {
        "schema_version": "offline_arb_bucketstats_v1", "generated_at": "x", "host": "h",
        "duration_seconds": 1.0, "diagnostic_run": True, "stage": "512b",
        "image_size": 512, "eligibility": "elig_512", "requires_nl": False,
        "domain_totals_sum": kept,
        "config": {
            "multi_ar_params": {"area_gate": 262144, "crop_loss_max": 0.2},
            "square_only_domains": ["synthetic"],
            "domain_mix_reference_mismatch": {},
        },
        "coverage": {"anime": {
            "shards_total": 17, "shards_scanned": 12, "shards_failed": 0,
            "rows_raw": 11999598, "rows_holdout_dropped": 0, "rows_elig_dropped": 0,
            "rows_invalid_wh_dropped": 0, "rows_unservable_dropped": unservable,
            "rows_kept": kept, "eligibility_column_shards": 0,
            "eligibility_fallback_shards": 0, "eligibility_fallback_rows": 0,
            "eligibility_compatibility_shards": 0,
        }},
        "anime_caption_availability": {
            "has_nl": 1, "tag_only": 0, "has_nl_fraction": 1.0, "tag_only_fraction": 0.0,
            "expected_tag_exposure": {"anime_tag_fraction": None},
        },
        "retained_percentiles": {}, "floors": [], "failed_shard_count": 0,
        "failed_shards": [], "known_simplifications": [],
    }


def test_summary_reports_every_drop_reason() -> None:
    """REGRESSION（由 12-shard 诊断实跑发现）：摘要原先只列 rows_raw 和 rows_kept，
    anime 那 61,565 行的缺口在人读摘要里毫无解释 —— 正是静默截断的模式。"""
    out = os.path.join(TMP, "sum.md")
    try:
        P3.write_markdown_summary(__import__("pathlib").Path(out), _fake_report())
        text = open(out, encoding="utf-8").read()
    except Exception as exc:                                   # noqa: BLE001
        check("write_markdown_summary runs", False, f"{type(exc).__name__}: {exc}")
        return
    check("summary has an unservable column", "unservable" in text, text[:200])
    check("summary prints the unservable count", "61,565" in text,
          [l for l in text.splitlines() if "anime" in l][:1])
    for label in ("holdout", "invalid wh"):
        check(f"summary has a {label!r} column", label in text)
    # 恒等式被破坏时必须大声报出来，而不是静静少一点
    out2 = os.path.join(TMP, "sum_bad.md")
    P3.write_markdown_summary(__import__("pathlib").Path(out2),
                              _fake_report(break_identity=True))
    t2 = open(out2, encoding="utf-8").read()
    check("broken coverage identity is called out loudly",
          "COVERAGE IDENTITY BROKEN" in t2,
          [l for l in t2.splitlines() if "anime" in l][:2])
    check("a sound report carries no broken-identity marker",
          "COVERAGE IDENTITY BROKEN" not in text)


def test_512b_whitelist_matches_512a() -> None:
    """REGRESSION：512b 的 fallback 白名单曾被放宽到含 textatlas，理由是"锚模式下不生效"。
    错了 —— textatlas **物理上就带 elig_512 列**，会被判为 column 模式，而白名单要求它是
    fallback 模式；phase5 把这个白名单当**发布门**用，于是整个构建在最后一步炸掉。"""
    a, b = P3.ELIGIBILITY_FALLBACK_DOMAINS["512a"], P3.ELIGIBILITY_FALLBACK_DOMAINS["512b"]
    check("512b whitelist == 512a whitelist", a == b, f"512a={sorted(a)} 512b={sorted(b)}")
    check("textatlas is NOT whitelisted (it really stores elig_512)",
          "textatlas" not in b, str(sorted(b)))
    # 直接核实那条物理事实，别只依赖注释
    import pyarrow.parquet as _pq
    try:
        cols = [c.name for c in _pq.ParquetFile(
            "/fsx/evan/textatlas_manifest_v1/CleanTextSynth/"
            "train-00000-of-00067.parquet").schema_arrow]
        check("textatlas manifest physically has elig_512", "elig_512" in cols, str(cols))
    except OSError as exc:
        check("textatlas manifest readable", False, str(exc))


def test_crop_loss_max_zero_survives_to_the_worker() -> None:
    """REGRESSION：`task.get("crop_loss_max") or DEFAULT` 会把合法的 0.0 换成 0.20，
    于是 N_matrix 按 0.20 算、报告记 0.0，而 phase5 的 mar_params 等值门照样通过 ——
    分歧要等到"manifest raw N differs from Phase-3a N_matrix"才暴露。"""
    rows = [(1024, 1024), (1030, 1024), (767, 512)]
    t = task(write_shard(rows, "clm0.parquet"))
    t["crop_loss_max"] = 0.0
    res = P3.scan_domain_shard(t)
    check("worker ok at crop_loss_max=0.0", res.get("ok") is True, str(res.get("error")))
    if not res.get("ok"):
        return
    # 0.0 = 只接受 AR 精确匹配 => 只有正方 1024x1024 落在方锚
    check("crop_loss_max=0.0 keeps only the exact-AR row",
          res["rows_kept"] == 1 and int(res["n_matrix"][0]) == 1,
          f'kept={res["rows_kept"]} n0={int(res["n_matrix"][0])}')
    check("the other two counted as unservable",
          res["rows_unservable_dropped"] == 2, str(res["rows_unservable_dropped"]))
    # 对照：默认 0.20 下三行应该都留
    res2 = P3.scan_domain_shard(task(write_shard(rows, "clm2.parquet")))
    check("default 0.20 keeps all three", res2["rows_kept"] == 3, str(res2["rows_kept"]))


def test_identity_reports_the_effective_thresholds() -> None:
    """REGRESSION：identity() 曾写死模块常量，于是 assignment_hash 的 anchor_identity
    在配置用非默认阈值时谎报，而紧邻的 payload["crop_loss_max"] 报真值。"""
    i = anchors_v1.identity(crop_loss_max=0.05, area_gate=262144)
    check("identity reflects the passed crop_loss_max", i["crop_loss_max"] == 0.05, str(i["crop_loss_max"]))
    check("identity reflects the passed area_gate", i["source_area_gate"] == 262144, str(i["source_area_gate"]))
    d = anchors_v1.identity()
    check("identity defaults to the module constant with the gate off",
          d["crop_loss_max"] == anchors_v1.CROP_LOSS_MAX and d["source_area_gate"] is None, str(d))


def test_known_simplifications_are_true_under_anchors() -> None:
    """REGRESSION：正式报告的 known_simplifications 曾声明三件锚模式下为假的事
    （读 elig_512 / 短边回退 / square_only 域被强钉到方桶）。"""
    import re
    src = open(_P3_PATH, encoding="utf-8").read()
    i = src.index('"known_simplifications"')
    block = src[i:i + 4000]
    anchors_branch = block[:block.index('if runtime["bucket_mode"] == anchors_v1.BUCKET_MODE')]
    check("anchors branch does not claim the elig_* column is read",
          "runtime['eligibility']" not in anchors_branch
          and "{runtime['eligibility']}" not in anchors_branch, "still references eligibility")
    check("anchors branch says no elig_* column is read",
          "不读任何 elig_" in anchors_branch)
    check("anchors branch says square_only is verified, not forced",
          "不是**" in anchors_branch or "落不到方锚的行被丢弃" in anchors_branch)
    check("anchors branch mentions rows_unservable_dropped",
          "rows_unservable_dropped" in anchors_branch)


def test_stage_and_dir_wiring() -> None:
    check("phase3a has a 512b bucketstats root", "512b" in P3.CANONICAL_OUTPUT_DIRS)
    check("512b root is distinct from 512a",
          P3.CANONICAL_OUTPUT_DIRS["512b"] != P3.CANONICAL_OUTPUT_DIRS["512a"])
    check("512b has an eligibility whitelist entry",
          "512b" in P3.ELIGIBILITY_FALLBACK_DOMAINS)
    check("ANCHORS_STAGE is 512b", P3.ANCHORS_STAGE == "512b")


def main() -> int:
    try:
        for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
            print(f"\n{fn.__name__}")
            fn()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("PHASE3A_ANCHORS_V1_TESTS_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
