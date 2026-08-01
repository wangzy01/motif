#!/usr/bin/env python3
"""phase5_build_catalog 的 anchors_v1 路径测试（合成表，不读真实 manifest）。

运行：`python scripts/test_phase5_anchors_v1.py`（退出码 0 = 全通过）。

`filter_and_assign()` 在锚模式下多了一件 legacy 路径没有的事：**指派失败的行要被丢掉**。
今天 legacy 路径里每个 selected 行都必然拿到一个桶，所以 `selected / wk / hk / bucket`
四个数组天然等长；锚模式下丢行必须四个一起丢，错一个就会让 catalog 的行与几何**静默错位**。
这里就是钉这件事。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

import numpy as np
import pyarrow as pa

from training.dataline import anchors_v1

# Load phase5 by path rather than as a package: `scripts/` deliberately has no
# __init__.py, and adding one to satisfy a test import would change the repo layout.
_P5_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "offline_arb", "phase5_build_catalog.py")
_spec = importlib.util.spec_from_file_location("_phase5_under_test", _P5_PATH)
P5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P5)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  PASS  " if ok else "  FAIL  ") + name + ("" if ok else f"  {detail}"))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def make_ctx(square_only=(), area_gate=anchors_v1.AREA_GATE_STAGE512A):
    buckets = anchors_v1.anchors_hw()
    return {
        "bucket_mode": anchors_v1.BUCKET_MODE,
        "buckets": buckets,
        "square_idx": anchors_v1.square_anchor_id(),
        "square_only": tuple(square_only),
        "fold_map": np.arange(len(buckets), dtype=np.int64),
        "image_size": 512,
        "eligibility": "elig_512",
        "anime_elig_always_true": False,
        "eligibility_fallback_domains": frozenset(
            ("anime", "monet_real", "gpic", "synthetic")),
        "mar_params": {
            "target_area": 262144, "divisor": 16, "no_upscale": True,
            "min_area_ratio": 0.75, "max_aspect": 4.0,
            "area_gate": area_gate, "crop_loss_max": anchors_v1.CROP_LOSS_MAX,
        },
    }


def make_table(rows, holdout=None):
    w = pa.array([float(r[0]) for r in rows], type=pa.float64())
    h = pa.array([float(r[1]) for r in rows], type=pa.float64())
    cols = {"width": w, "height": h}
    if holdout is not None:
        cols["holdout"] = pa.array(holdout, type=pa.int8())
    return pa.table(cols)


def run(rows, domain="monet_real", holdout=None, **ctxkw):
    P5._CTX = make_ctx(**ctxkw)
    t = make_table(rows, holdout)
    return P5.filter_and_assign(t, domain, folded=False)


def test_alignment_after_dropping_unservable() -> None:
    """混入不可服务行后，四个返回数组必须仍然逐行对齐。"""
    rows = [
        (767, 512),     # servable -> 624x416
        (300, 300),     # area 90,000 < gate -> elig drop
        (1008, 435),    # servable -> 704x352
        (2000, 400),    # area OK (800,000) but AR 5.0 -> no feasible anchor
        (640, 640),     # servable -> 512x512
        (1024, 256),    # area exactly at gate, AR 4.0 -> unservable
    ]
    selected, wk, hk, bucket, c = run(rows)
    check("returns 3 servable rows", selected.size == 3, f"{selected.size}")
    check("all four arrays equal length",
          len({selected.size, wk.size, hk.size, bucket.size}) == 1,
          f"{selected.size} {wk.size} {hk.size} {bucket.size}")
    check("selected indexes the original rows", list(selected) == [0, 2, 4], str(selected))
    check("widths follow the surviving rows", list(wk) == [767.0, 1008.0, 640.0], str(wk))
    check("heights follow the surviving rows", list(hk) == [512.0, 435.0, 640.0], str(hk))
    got = [anchors_v1.ANCHORS_V1_WH[b] for b in bucket]
    check("anchors are the hand-checked ones",
          got == [(624, 416), (704, 352), (512, 512)], str(got))
    check("elig drop counted", c["rows_elig_dropped"] == 1, str(c))
    check("unservable drop counted", c["rows_unservable_dropped"] == 2, str(c))
    check("rows_kept == surviving rows", c["rows_kept"] == 3, str(c))
    check("counters reconcile",
          c["rows_raw"] - c["rows_holdout_dropped"] - c["rows_elig_dropped"]
          - c["rows_invalid_wh_dropped"] - c["rows_unservable_dropped"] == c["rows_kept"],
          str(c))


def test_area_gate_is_the_eligibility_gate() -> None:
    """锚模式下资格门是面积门，不是 min(w,h) >= 512。"""
    rows = [(696, 510), (1008, 435)]      # 两行短边都 < 512，legacy 会全丢
    selected, _wk, _hk, bucket, c = run(rows)
    check("short-side-under-512 rows are now kept", selected.size == 2, str(c))
    check("no elig drops", c["rows_elig_dropped"] == 0, str(c))
    got = [anchors_v1.ANCHORS_V1_WH[b] for b in bucket]
    check("696x510 -> 576x432, 1008x435 -> 704x352",
          got == [(576, 432), (704, 352)], str(got))
    # 关掉面积门后，低于 262144 但可零上采样服务的行应该进来
    sel2, _, _, b2, _ = run([(720, 360)], area_gate=1)
    check("area_gate=1 admits 720x360 (area 259,200)",
          sel2.size == 1 and anchors_v1.ANCHORS_V1_WH[b2[0]] == (704, 352), str(b2))
    sel3, _, _, _, _ = run([(720, 360)])
    check("default area gate rejects 720x360", sel3.size == 0)


def test_square_only_is_verified_not_forced() -> None:
    """square_only 域也要跑真实指派：强行钉到方锚会让覆盖不了 512x512 的行被上采样。"""
    rows = [
        (1024, 1024),   # 方图，覆盖得了 512x512
        (800, 600),     # 非方，锚模式下应落到 4:3 而不是被钉成方图 -> square_only 下丢弃
        (400, 400),     # 方但太小，覆盖不了 512x512 -> 必须丢
    ]
    selected, _wk, _hk, bucket, c = run(rows, domain="synthetic", square_only=("synthetic",))
    check("only the row that can cover the square anchor survives",
          selected.size == 1 and selected[0] == 0, f"{selected} {c}")
    check("it landed on the square anchor",
          bucket[0] == anchors_v1.square_anchor_id(), str(bucket))
    check("the sub-512 square row was dropped, not upsampled",
          c["rows_unservable_dropped"] >= 1, str(c))
    # 同样三行在非 square_only 域下，800x600 应当被正常服务
    selected2, _, _, bucket2, _ = run(rows, domain="monet_real")
    got = [anchors_v1.ANCHORS_V1_WH[b] for b in bucket2]
    check("without square_only, 800x600 is served by the 4:3 anchor",
          list(selected2) == [0, 1] and got == [(512, 512), (576, 432)], f"{selected2} {got}")


def test_holdout_and_invalid_still_honoured() -> None:
    """注意归因顺序：`elig` 先于 `wh_valid` 求值，所以 `(0,512)` / `(767,-1)` 会被记进
    `rows_elig_dropped` 而不是 `rows_invalid_wh_dropped`。legacy 路径同样如此
    （`min(0,512) >= 512` 也是 False），这里保持一致，只要求它们**确实被丢掉**。"""
    rows = [(767, 512), (767, 512), (0, 512), (767, -1)]
    selected, _wk, _hk, _bucket, c = run(rows, holdout=[0, 1, 0, 0])
    check("holdout row excluded", c["rows_holdout_dropped"] == 1, str(c))
    check("degenerate w/h excluded (attributed to the area gate, as in legacy)",
          c["rows_elig_dropped"] == 2 and c["rows_invalid_wh_dropped"] == 0, str(c))
    check("one row survives", selected.size == 1 and selected[0] == 0, str(selected))
    check("counters reconcile",
          c["rows_raw"] - c["rows_holdout_dropped"] - c["rows_elig_dropped"]
          - c["rows_invalid_wh_dropped"] - c["rows_unservable_dropped"] == c["rows_kept"],
          str(c))
    # nan / inf 也必须被丢掉，且不引发异常
    sel2, _, _, _, c2 = run([(float("nan"), 512), (float("inf"), 512), (767, 512)])
    check("nan/inf sizes are dropped without raising",
          sel2.size == 1 and sel2[0] == 2, f"{sel2} {c2}")


def test_no_upsampling_on_the_catalog_path() -> None:
    """随机尺寸过一遍完整 filter_and_assign，凡是留下的行都必须能覆盖它的锚。"""
    rng = np.random.default_rng(20260801)
    rows = [(int(x), int(y)) for x, y in
            zip(rng.integers(200, 2000, 4000), rng.integers(200, 2000, 4000))]
    selected, wk, hk, bucket, c = run(rows)
    bad = []
    for i in range(selected.size):
        W, H = anchors_v1.ANCHORS_V1_WH[bucket[i]]
        if wk[i] < W or hk[i] < H:
            bad.append((wk[i], hk[i], W, H))
    check(f"zero upsampling over {selected.size} kept rows of 4000", not bad, str(bad[:3]))
    check("kept rows all pass the area gate",
          bool(np.all(wk * hk >= anchors_v1.AREA_GATE_STAGE512A)))
    check("a meaningful fraction was kept", 0 < selected.size < 4000, str(selected.size))


def test_bucket_indices_are_anchor_ids() -> None:
    """桶下标必须就是 anchor_id —— 所有已审计的行数/概率都按它索引。"""
    rows = [(w * 2, h * 2) for (w, h) in anchors_v1.ANCHORS_V1_WH]
    selected, _wk, _hk, bucket, _c = run(rows)
    check("every anchor is reachable and maps to its own id",
          selected.size == 13 and list(bucket) == list(range(13)), str(bucket))


def test_legacy_path_untouched() -> None:
    """adaptive 模式必须仍然走老逻辑：361 桶、min(w,h)>=512 资格门、不丢不可服务行。"""
    from training.dataline.multi_ar import build_adaptive_buckets, square_index
    buckets = build_adaptive_buckets(262144, 16, 0.75, 4.0)
    check("legacy table is still 361 buckets", len(buckets) == 361, str(len(buckets)))
    P5._CTX = {
        "bucket_mode": "adaptive",
        "buckets": buckets,
        "square_idx": square_index(buckets),
        "square_only": (),
        "fold_map": np.arange(len(buckets), dtype=np.int64),
        "image_size": 512,
        "eligibility": "elig_512",
        "anime_elig_always_true": False,
        "eligibility_fallback_domains": frozenset(("monet_real",)),
        "mar_params": {"target_area": 262144, "divisor": 16, "no_upscale": True,
                       "min_area_ratio": 0.75, "max_aspect": 4.0},
        "bucket_table": P5.build_bucket_table(buckets, 16),
        "extreme_long": 1024, "extreme_short": 256,
    }
    rows = [(767, 512), (696, 510), (640, 640)]
    selected, _wk, _hk, bucket, c = P5.filter_and_assign(make_table(rows), "monet_real",
                                                         folded=False)
    check("legacy keeps only min(w,h)>=512 rows", selected.size == 2, f"{selected} {c}")
    check("legacy drops 696x510 via the short-side gate",
          c["rows_elig_dropped"] == 1, str(c))
    check("legacy reports zero unservable (it never drops post-assignment)",
          c["rows_unservable_dropped"] == 0, str(c))
    check("legacy bucket indices are in the 361 namespace",
          bool(np.all((bucket >= 0) & (bucket < 361))), str(bucket))


def test_merge_coverage_accepts_the_new_counter() -> None:
    """REGRESSION: `rows_unservable_dropped` is emitted in BOTH modes, and
    `_merge_coverage` builds a plain dict with a fixed key set and does
    `row[key] += int(value)`. A missing key there is a KeyError raised at header
    assembly -- i.e. AFTER a multi-hour build has already done all its work, in legacy
    mode too. The previous test suite proved the key existed in the counters and then
    never fed it to the aggregator, which is exactly how that shipped green."""
    for mode, unservable in (("adaptive", 0), (anchors_v1.BUCKET_MODE, 7)):
        cov = {
            "rows_raw": 100, "rows_holdout_dropped": 2, "rows_elig_dropped": 5,
            "rows_invalid_wh_dropped": 1, "rows_unservable_dropped": unservable,
            "rows_kept": 100 - 2 - 5 - 1 - unservable,
            "eligibility_column_shards": 0, "eligibility_fallback_shards": 1,
            "eligibility_fallback_rows": 100, "eligibility_compatibility_shards": 0,
        }
        try:
            merged = P5._merge_coverage(
                [{"domain": "monet_real", "coverage": cov}],
                {d: 1 for d in P5.DOMAINS},
            )
            got = merged["monet_real"]
            ok = (got["rows_unservable_dropped"] == unservable
                  and got["rows_kept"] == cov["rows_kept"])
            detail = str(got)
        except Exception as exc:                               # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        check(f"_merge_coverage handles {mode} counters", ok, detail)
        # 覆盖率恒等式必须是五项而不是四项
        if ok:
            g = merged["monet_real"]
            check(f"coverage identity reconciles ({mode})",
                  g["rows_raw"] - g["rows_holdout_dropped"] - g["rows_elig_dropped"]
                  - g["rows_invalid_wh_dropped"] - g["rows_unservable_dropped"]
                  == g["rows_kept"], str(g))


def test_identity_fold_is_required_in_anchors_mode() -> None:
    """REGRESSION: compute_floor_mapping -> nearest_allowed_bucket has no feasibility
    test, so folding a pruned extent lands on a TALLER anchor and upsamples:
    704x352 -> 672x384 is 1.091x. Must refuse, not silently ship."""
    ident = np.arange(13, dtype=np.int64)
    try:
        P5.require_identity_fold(anchors_v1.BUCKET_MODE, ident, 0.0)
        check("identity fold accepted in anchors mode", True)
    except ValueError as exc:
        check("identity fold accepted in anchors mode", False, str(exc))
    # 审查里实测出来的那张 fold_map：两个 extent 和 15:17 对被折走
    bad = np.array([0, 1, 2, 3, 4, 5, 6, 5, 6, 9, 10, 9, 10], dtype=np.int64)
    # 先证明这张 fold_map 真的会上采样，警告不是空话
    ups = []
    for i, j in enumerate(bad):
        if i == j:
            continue
        W0, H0 = anchors_v1.ANCHORS_V1_WH[i]
        W1, H1 = anchors_v1.ANCHORS_V1_WH[int(j)]
        if W0 < W1 or H0 < H1:
            ups.append((f"{W0}x{H0}", f"{W1}x{H1}", round(max(W1 / W0, H1 / H0), 4)))
    check("the sample fold map really does upsample", len(ups) == 4, str(ups))
    try:
        P5.require_identity_fold(anchors_v1.BUCKET_MODE, bad, 0.005)
        check("non-identity fold refused in anchors mode", False, "did not raise")
    except ValueError as exc:
        check("non-identity fold refused in anchors mode", "identity fold" in str(exc),
              str(exc))
    # legacy 模式必须不受影响
    try:
        P5.require_identity_fold("adaptive", bad, 0.005)
        check("legacy fold untouched by the guard", True)
    except ValueError as exc:
        check("legacy fold untouched by the guard", False, str(exc))


def test_selected_indexes_original_positions_for_consumers() -> None:
    """`selected` 被消费方当作**原始 row-group 内下标**用
    （`caption_row_in_group = selected`、`absolute = rg_starts[rg] + selected`、
    `real_derived_columns(table, domain, selected)`）。返回 `arange(rows_kept)` 也能
    通过"四数组等长"检查，所以这里直接模拟消费方的 re-index。"""
    rows = [(300, 300), (767, 512), (2000, 400), (640, 640), (100, 100), (1008, 435)]
    selected, wk, hk, bucket, _c = run(rows)
    # 消费方用 selected 去索引整表列
    tbl = make_table(rows)
    w_full = np.asarray(tbl["width"].to_pylist(), dtype=np.float64)
    h_full = np.asarray(tbl["height"].to_pylist(), dtype=np.float64)
    check("selected re-indexes the full table to the same widths",
          list(w_full[selected]) == list(wk), f"{w_full[selected]} vs {wk}")
    check("selected re-indexes the full table to the same heights",
          list(h_full[selected]) == list(hk), f"{h_full[selected]} vs {hk}")
    check("selected is NOT a dense arange (that would pass a length-only check)",
          list(selected) != list(range(selected.size)), str(selected))
    check("selected is strictly increasing and in range",
          bool(np.all(np.diff(selected) > 0)) and selected.max() < len(rows), str(selected))
    # 每个留下的行，几何必须对应它自己的原始尺寸
    bad = [(int(s), float(w_full[s]), float(h_full[s]), anchors_v1.ANCHORS_V1_WH[b])
           for s, b in zip(selected, bucket)
           if w_full[s] < anchors_v1.ANCHORS_V1_WH[b][0]
           or h_full[s] < anchors_v1.ANCHORS_V1_WH[b][1]]
    check("each surviving row covers the anchor it was paired with", not bad, str(bad))


def test_oversized_row_is_dropped_not_fatal() -> None:
    """REGRESSION: 一行 width >= 2**31 会让 anchors_v1 抛错，进而杀掉整个多小时构建，
    且报错里既没有 shard 也没有 uid。必须在 wh_valid 里筛掉。"""
    rows = [(767, 512), (2 ** 31, 512), (float(2 ** 32), 1024), (640, 640)]
    try:
        selected, _wk, _hk, _b, c = run(rows)
        ok = selected.size == 2 and list(selected) == [0, 3]
        detail = f"{selected} {c}"
    except Exception as exc:                                   # noqa: BLE001
        ok, detail = False, f"raised {type(exc).__name__}: {exc}"
    check("oversized rows are dropped without aborting", ok, detail)


def main() -> int:
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("PHASE5_ANCHORS_V1_TESTS_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
