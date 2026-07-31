#!/usr/bin/env python3
"""anchors_v1 几何契约测试（纯逻辑，零模型 / 零数据 / 零网络依赖）。

运行：`python scripts/test_anchors_v1_geometry.py`（退出码 0 = 全通过）。

不放在 `tests/` 下：`scripts/run_exact_policy_tests.py:55-58` 对 `tests/` 目录做
**exact-two closure** 校验（`actual_files == expected_files` 否则 "curated tests
directory exact-two closure drift"），往里加任何文件都会弄坏那个既有 runner。
本文件沿用冻结 release 里 `scripts/test_naifu_arb_fixes.py` 的独立脚本惯例。

覆盖的失效模式，全部是这条链上真实踩过或差点踩到的：
  - 方向搞反：既有桶表是 `(H, W)`，本模块对外是 `{width, height}`
  - 上采样：`resize_crop_to_bucket()` 的 `scale = max(W/w0, H/h0)` 从不拒绝放大
  - anchor_id 错位：审计产出的行数 / 概率 / 配比全部按 anchor_id 索引
  - crop_loss 阈值与"只服务 AR <= 2.5"的等价性被改坏
  - EXIF 旋转导致解码尺寸与 manifest 不一致
  - 向量化实现与标量实现悄悄分叉（13 锚下**每一行**都会走到向量化路径）
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from training.dataline import anchors_v1 as A

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS  {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


def raises(fn, *args, needle: str = "", **kwargs) -> tuple[bool, str]:
    try:
        fn(*args, **kwargs)
    except A.AnchorGeometryError as exc:
        if needle and needle not in str(exc):
            return False, f"wrong message: {exc}"
        return True, ""
    except Exception as exc:                                  # noqa: BLE001
        return False, f"wrong exception type {type(exc).__name__}: {exc}"
    return False, "did not raise"


# --- 表身份 ---------------------------------------------------------------------

def test_table_identity_and_orientation() -> None:
    wh = A.anchors_wh()
    expect = [(512, 512), (576, 432), (432, 576), (624, 416), (416, 624),
              (672, 384), (384, 672), (704, 352), (352, 704), (448, 560),
              (560, 448), (480, 544), (544, 480)]
    check("anchor_set_id == B13", A.ANCHOR_SET_ID == "B13", A.ANCHOR_SET_ID)
    check("bucket_mode == anchors_v1", A.BUCKET_MODE == "anchors_v1", A.BUCKET_MODE)
    # anchor_id 顺序必须与 Stage512b 审计 REPORT.json 的 per_anchor[i]["anchor_id"] 一致
    check("anchor_id order matches audit reports", wh == expect, f"{wh}")
    check("id3 is 624 wide x 416 high (landscape)", wh[3] == (624, 416) and wh[3][0] > wh[3][1])
    check("id4 is 416 wide x 624 high (portrait)", wh[4] == (416, 624) and wh[4][0] < wh[4][1])
    check("anchors_hw() transposes to (H, W)", A.anchors_hw() == [(h, w) for (w, h) in wh])
    check("anchors_hw()[3] == (416, 624)", A.anchors_hw()[3] == (416, 624))


def test_table_invariants() -> None:
    ok_align = all(w % 16 == 0 and h % 16 == 0 for w, h in A.anchors_wh())
    ok_extent = all(0.5 - 1e-12 <= w / h <= 2.0 + 1e-12 for w, h in A.anchors_wh())
    ok_sym = all((h, w) in A.ANCHORS_V1_WH for w, h in A.anchors_wh())
    check("all anchors divisor-16 aligned", ok_align)
    check("all anchors within 1:2..2:1 extent", ok_extent)
    check("every anchor has a transposed partner", ok_sym)
    check("MIN_ANCHOR_AREA == 704*352", A.MIN_ANCHOR_AREA == 247808, str(A.MIN_ANCHOR_AREA))
    check("MAX_ANCHOR_SIDE == 704 (> stage512a's 672)", A.MAX_ANCHOR_SIDE == 704)
    check("square anchor exists at id 0", A.square_anchor_id() == 0
          and A.ANCHORS_V1_WH[0][0] == A.ANCHORS_V1_WH[0][1])
    # 必须有且只有一个方锚：square_index() 取"面积最大的方桶"，零方锚会被
    # planner_v1.py:428 / validate_plan.py:281 / offline_dataset.py:1941 三处拒绝
    n_square = sum(1 for w, h in A.anchors_wh() if w == h)
    check("exactly one square anchor", n_square == 1, str(n_square))


def test_area_gate_is_not_implied_by_zero_upsampling() -> None:
    """面积门与零上采样**不等价**——这就是把 area_gate 做成可选参数的理由。"""
    check("min anchor area < stage512a area gate",
          A.MIN_ANCHOR_AREA < A.AREA_GATE_STAGE512A,
          f"{A.MIN_ANCHOR_AREA} vs {A.AREA_GATE_STAGE512A}")


# --- crop_loss 的解析性质 --------------------------------------------------------

def test_crop_loss_depends_only_on_aspect_ratio() -> None:
    """retention = min(AR_src/AR_anchor, AR_anchor/AR_src)，与绝对尺寸无关。"""
    worst = 0.0
    for k, (W, H) in enumerate(A.ANCHORS_V1_WH):
        ar_k = W / H
        for w, h in [(W, H), (2 * W, 2 * H), (W + 133, H + 71), (3 * W, 3 * H + 40)]:
            closed = 1.0 - min((w / h) / ar_k, ar_k / (w / h))
            worst = max(worst, abs(A.crop_loss_wh(w, h, k) - closed))
    check("crop_loss matches the closed form min(AR ratio)", worst <= 1e-12, f"max dev {worst:.3e}")


def test_argmin_log_ar_is_also_argmin_crop_loss() -> None:
    """primary key |dlog AR| 与 crop_loss 同序 => 把阈值作用在胜者上，等价于
    '没有任何可行锚能做到 crop_loss <= T'。"""
    worst = 0.0
    n = 0
    for w in range(360, 1400, 23):
        for h in range(360, 1400, 29):
            feasible = [k for k, (W, H) in enumerate(A.ANCHORS_V1_WH) if w >= W and h >= H]
            if not feasible:
                continue
            best_cl = min(A.crop_loss_wh(w, h, k) for k in feasible)
            got = A.assign_anchor(w, h, crop_loss_max=1.0)
            if got is None:
                check("argmin==argmin", False, f"{w}x{h} feasible but unassigned")
                return
            worst = max(worst, abs(got[1] - best_cl))
            n += 1
    check(f"argmin |dlogAR| == argmin crop_loss over {n} sizes", worst <= 1e-12,
          f"max dev {worst:.3e}")


def test_crop_loss_threshold_equals_ar_band() -> None:
    """crop_loss <= 0.20  <=>  AR 之比 in [0.8, 1.25]  <=>  终端 2:1 锚服务到 AR 2.5。"""
    check("CROP_LOSS_MAX == 0.20", A.CROP_LOSS_MAX == 0.20, str(A.CROP_LOSS_MAX))
    got = A.assign_anchor(880, 352)                 # AR 2.5 exactly
    ok = got is not None and A.ANCHORS_V1_WH[got[0]] == (704, 352) and close(got[1], 0.20, 1e-9)
    check("AR 2.5 -> 2:1 anchor at crop_loss 0.20 (boundary included)", ok, str(got))
    check("AR 2.477 accepted", A.assign_anchor(872, 352) is not None)
    check("AR 2.528 rejected", A.assign_anchor(890, 352) is None)
    check("AR 1/2.528 rejected (vertical symmetry)", A.assign_anchor(352, 890) is None)


# --- 零上采样（本模块存在的唯一理由） -----------------------------------------------

def test_zero_upsampling_exhaustive() -> None:
    """穷举网格：凡被接受的 (w,h)，源图必须完整覆盖锚框，且缩放因子 <= 1。"""
    n_ok = 0
    bad = []
    for w in range(200, 1600, 7):
        for h in range(200, 1600, 11):
            got = A.assign_anchor(w, h)
            if got is None:
                continue
            n_ok += 1
            W, H = A.ANCHORS_V1_WH[got[0]]
            if w < W or h < H or max(W / w, H / h) > 1.0 + 1e-12:
                bad.append((w, h, W, H))
    check(f"zero upsampling over {n_ok} accepted sizes", not bad, f"violations: {bad[:5]}")
    check("grid dense enough to be meaningful", n_ok > 10000, str(n_ok))


def test_rejects_sizes_below_every_anchor() -> None:
    bad = [(w, h) for w, h in [(1, 1), (511, 511), (350, 350), (703, 351), (351, 703),
                               (500, 510), (520, 500), (640, 400), (0, 512), (-8, 512)]
           if A.assign_anchor(w, h) is not None]
    check("sub-anchor sizes all rejected", not bad, f"wrongly accepted: {bad}")


def test_exact_anchor_size_has_zero_loss() -> None:
    bad = []
    for k, (W, H) in enumerate(A.ANCHORS_V1_WH):
        got = A.assign_anchor(W, H)
        if got is None or got[0] != k or not close(got[1], 0.0):
            bad.append((k, W, H, got))
    check("exact anchor size -> itself, crop_loss 0", not bad, f"{bad}")


# --- 真实池样本（手算核对）--------------------------------------------------------

def test_real_pool_rows() -> None:
    """取自 t5_manifest/rich_v1 首个 shard 的真实行（monet_0000 / gpic_0000）。"""
    cases = [
        (767, 512, (624, 416), 0.00130, "monet cc12m ge512"),
        (696, 510, (576, 432), 0.02303, "monet cc12m res256; 510<512 so square infeasible"),
        (1008, 435, (704, 352), 0.13690, "monet cc12m elongated -> 2:1 terminal"),
        (825, 510, (624, 416), 0.07268, "monet cc12m; 1.75 anchor has larger |dlogAR|"),
        (639, 428, (624, 416), 0.00466, "gpic res256"),
        (427, 640, (416, 624), 0.00078, "gpic res256 portrait"),
        (482, 640, (432, 576), 0.00412, "gpic res256 portrait"),
        (640, 640, (512, 512), 0.0, "gpic ge512 square"),
    ]
    for w, h, expect_wh, expect_cl, note in cases:
        got = A.assign_anchor(w, h)
        ok = got is not None and A.ANCHORS_V1_WH[got[0]] == expect_wh and close(got[1], expect_cl, 5e-5)
        check(f"real row {w}x{h} -> {expect_wh[0]}x{expect_wh[1]}  ({note})", ok, str(got))


# --- 标量 / 向量化一致性 ----------------------------------------------------------

def test_scalar_and_batch_agree_exactly() -> None:
    """13 锚下 phase5 的稠密查表 99.9% 落空，**每一行**都会走向量化路径，
    所以两条实现必须逐位一致，否则目录与运行期会静默分叉。"""
    ws, hs = [], []
    for w in range(180, 1500, 13):
        for h in range(180, 1500, 17):
            ws.append(w)
            hs.append(h)
    w_arr = np.asarray(ws, dtype=np.int64)
    h_arr = np.asarray(hs, dtype=np.int64)
    ids, cls = A.assign_anchor_batch(w_arr, h_arr)
    mism_id, mism_cl = 0, 0
    for i, (w, h) in enumerate(zip(ws, hs)):
        got = A.assign_anchor(w, h)
        if got is None:
            if ids[i] != -1:
                mism_id += 1
            elif not math.isnan(float(cls[i])):
                mism_cl += 1
        else:
            if ids[i] != got[0]:
                mism_id += 1
            elif float(cls[i]) != got[1]:            # 逐位相等，不是 approx
                mism_cl += 1
    check(f"batch vs scalar anchor_id identical over {len(ws)} sizes", mism_id == 0, str(mism_id))
    check("batch vs scalar crop_loss bit-identical", mism_cl == 0, str(mism_cl))
    n_acc = int((ids >= 0).sum())
    check("batch accepts a non-trivial fraction", 0 < n_acc < len(ws), f"{n_acc}/{len(ws)}")


def test_batch_honours_area_gate_and_shapes() -> None:
    # 注意 704x352 = 247,808 本身就低于面积门 —— 整个 2:1 锚在审计口径下只能靠
    # 面积 >= 262,144 的更大源图供给（如 768x384 = 294,912）。
    w = np.asarray([720, 720, 1024, 768, 704], dtype=np.int64)
    h = np.asarray([360, 360, 256, 384, 352], dtype=np.int64)
    ids_open, _ = A.assign_anchor_batch(w, h, area_gate=None)
    ids_gate, _ = A.assign_anchor_batch(w, h, area_gate=A.AREA_GATE_STAGE512A)
    check("720x360 servable without area gate", ids_open[0] == 7, str(ids_open))
    check("720x360 killed by the area gate", ids_gate[0] == -1, str(ids_gate))
    check("1024x256 (AR 4) rejected either way", ids_open[2] == -1 and ids_gate[2] == -1)
    check("768x384 -> 2:1 anchor under both gates",
          ids_open[3] == 7 and ids_gate[3] == 7, f"{ids_open} {ids_gate}")
    check("704x352 exact anchor size is itself below the area gate",
          ids_open[4] == 7 and ids_gate[4] == -1, f"{ids_open} {ids_gate}")
    ok, detail = raises(A.assign_anchor_batch, np.zeros(3), np.zeros(4))
    check("batch rejects mismatched shapes", ok, detail)


# --- 可选面积门（标量） -----------------------------------------------------------

def test_area_gate_semantics() -> None:
    """存在 (w,h)：可行集非空、零上采样，但面积 < 262144 —— 面积门会误杀它。"""
    got = A.assign_anchor(720, 360, area_gate=None)
    check("720x360 (area 259,200) servable at zero upsampling",
          got is not None and A.ANCHORS_V1_WH[got[0]] == (704, 352), str(got))
    check("720x360 rejected under the audited area gate",
          A.assign_anchor(720, 360, area_gate=A.AREA_GATE_STAGE512A) is None)
    # 反向：面积达标但可行集为空 => 面积门是 unservable 的子集，不是独立漏斗阶段
    check("1024x256 has area >= gate but empty feasible set",
          1024 * 256 >= A.AREA_GATE_STAGE512A
          and A.assign_anchor(1024, 256, area_gate=A.AREA_GATE_STAGE512A) is None
          and A.assign_anchor(1024, 256, area_gate=None) is None)


# --- 运行期绑定校验（EXIF） --------------------------------------------------------

def test_verify_binding() -> None:
    try:
        A.verify_binding(1008, 435, 7)
        A.verify_binding(704, 352, 7)
        check("verify_binding accepts consistent decode", True)
    except A.AnchorGeometryError as exc:
        check("verify_binding accepts consistent decode", False, str(exc))
    # manifest 记 704x352 -> 解码成 352x704（EXIF 旋转）必须 fail-closed
    ok, detail = raises(A.verify_binding, 352, 704, 7, needle="upsample would be required")
    check("EXIF-swapped decode fails closed", ok, detail)
    ok, detail = raises(A.verify_binding, 703, 352, 7, needle="upsample would be required")
    check("one-pixel-short decode fails closed", ok, detail)
    ok, detail = raises(A.verify_binding, 1024, 1024, 13)
    check("out-of-range anchor_id fails closed", ok, detail)
    ok, detail = raises(A.verify_binding, 0, 512, 0)
    check("degenerate decode size fails closed", ok, detail)
    ok, detail = raises(A.verify_binding, 890, 352, 7, needle="crop_loss")
    check("over-threshold crop_loss fails closed", ok, detail)


# --- 概率向量 --------------------------------------------------------------------

def test_probs_from_mapping() -> None:
    p = A.probs_from_mapping({"624x416": 2.0, "512x512": 1.0, 3: 0.0})
    ok = (len(p) == 13 and close(sum(p), 1.0, 1e-12)
          and close(p[3], 2.0 / 3.0, 1e-12) and close(p[0], 1.0 / 3.0, 1e-12)
          and all(p[i] == 0.0 for i in range(13) if i not in (0, 3)))
    check("probs_from_mapping normalizes by name and id", ok, str(p))
    for bad in ({"624x416": -1.0, "512x512": 2.0}, {"999x999": 1.0}, {"512x512": 0.0}):
        ok, detail = raises(A.probs_from_mapping, bad)
        check(f"probs_from_mapping rejects {bad}", ok, detail)


def test_identity_payload() -> None:
    import json
    ident = A.identity()
    try:
        json.dumps(ident)
        ser = True
    except TypeError as exc:
        ser = False
        print(f"    (json error: {exc})")
    ok = (ser and ident["bucket_mode"] == "anchors_v1" and ident["anchor_set_id"] == "B13"
          and ident["n_anchors"] == 13 and len(ident["anchors_wh"]) == 13
          and ident["crop_loss_max"] == 0.20)
    check("identity() is serializable and complete", ok, str(ident)[:200])


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ANCHORS_V1_GEOMETRY_TESTS_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
