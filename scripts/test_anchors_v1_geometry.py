#!/usr/bin/env python3
"""anchors_v1 几何契约测试（纯逻辑，零模型 / 零数据 / 零网络依赖）。

运行：`python scripts/test_anchors_v1_geometry.py`（退出码 0 = 全通过）。

不放在 `tests/` 下：`scripts/run_exact_policy_tests.py:55-58` 对 `tests/` 目录做
**exact-two closure** 校验，往里加任何文件都会弄坏那个既有 runner。
本文件沿用冻结 release 里 `scripts/test_naifu_arb_fixes.py` 的独立脚本惯例（无 pytest）。

期望值一律用 `fractions.Fraction` **现场精确算出**，不写手抄常数 ——
上一版有两个手抄的 crop_loss 只对到小数第 4 位（`696x510` 写 0.02303，实际 2/87；
`825x510` 写 0.07268，实际 4/55），靠 5e-5 的容差蒙过去，
而那个容差恰好宽到无法区分正确实现与差 4.7e-5 的错误实现。

带 `REGRESSION` 标记的用例对应一次对抗性审查确认的 8 个缺陷，逐条锁死。
"""
from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

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


def raises(fn, *args, needle: str = "", **kwargs) -> tuple[bool, str]:
    try:
        fn(*args, **kwargs)
    except A.AnchorGeometryError as exc:
        if needle and needle not in str(exc):
            return False, f"wrong message: {exc}"
        return True, ""
    except Exception as exc:                                   # noqa: BLE001
        return False, f"wrong exception type {type(exc).__name__}: {exc}"
    return False, "did not raise"


def exact_retention(w: int, h: int, k: int) -> Fraction:
    """retention = min(w*H, h*W) / max(w*H, h*W)，精确有理数。"""
    W, H = A.ANCHORS_V1_WH[k]
    a, b = w * H, h * W
    return Fraction(min(a, b), max(a, b))


def exact_crop_loss(w: int, h: int, k: int) -> Fraction:
    return 1 - exact_retention(w, h, k)


# --- 表身份 ---------------------------------------------------------------------

def test_table_identity_and_orientation() -> None:
    wh = A.anchors_wh()
    expect = [(512, 512), (576, 432), (432, 576), (624, 416), (416, 624),
              (672, 384), (384, 672), (704, 352), (352, 704), (448, 560),
              (560, 448), (480, 544), (544, 480)]
    check("anchor_set_id == B13", A.ANCHOR_SET_ID == "B13", A.ANCHOR_SET_ID)
    # anchor_id 顺序必须与审计 REPORT.json 的 per_anchor[i]["anchor_id"] 一致
    check("anchor_id order matches audit reports", wh == expect, f"{wh}")
    check("id3 is 624 wide x 416 high (landscape)", wh[3] == (624, 416) and wh[3][0] > wh[3][1])
    check("id4 is 416 wide x 624 high (portrait)", wh[4] == (416, 624) and wh[4][0] < wh[4][1])
    check("anchors_hw() transposes to (H, W)", A.anchors_hw() == [(h, w) for (w, h) in wh])
    check("exactly one square anchor, at id 0",
          sum(1 for w, h in wh if w == h) == 1 and A.square_anchor_id() == 0)
    check("MIN_ANCHOR_AREA == 704*352 == 247808", A.MIN_ANCHOR_AREA == 247808)
    check("MAX_ANCHOR_SIDE == 704 (> stage512a's 672)", A.MAX_ANCHOR_SIDE == 704)
    check("area gate is NOT implied by zero upsampling",
          A.MIN_ANCHOR_AREA < A.AREA_GATE_STAGE512A)


# --- retention 的解析性质 ---------------------------------------------------------

def test_retention_is_exact_rational_of_aspect_ratio() -> None:
    """retention = min(w*H, h*W)/max(w*H, h*W)，只依赖 AR，与绝对尺寸无关。"""
    worst = 0.0
    scale_bad = []
    for k in range(13):
        W, H = A.ANCHORS_V1_WH[k]
        for w, h in [(W, H), (W + 133, H + 71), (3 * W, 3 * H + 40)]:
            got = A.crop_loss_wh(w, h, k)
            worst = max(worst, abs(got - float(exact_crop_loss(w, h, k))))
        # 同一 AR 的不同绝对尺寸必须给出同一 retention
        base = exact_retention(5 * 100, 2 * 100, k)
        for t in (7, 33, 991, 40001):
            if exact_retention(5 * t, 2 * t, k) != base:
                scale_bad.append((k, t))
    check("crop_loss_wh matches the exact rational", worst <= 1e-15, f"max dev {worst:.3e}")
    check("exact retention is scale-invariant", not scale_bad, str(scale_bad[:4]))


def test_argmax_retention_equals_argmin_log_ar() -> None:
    """|log AR_src - log AR_k| = -log(retention_k)，所以两个 argmax/argmin 同解。
    一个键就够，不需要 -retention 这一级。"""
    mism = 0
    for w in range(360, 1400, 23):
        for h in range(360, 1400, 29):
            feas = [k for k, (W, H) in enumerate(A.ANCHORS_V1_WH) if w >= W and h >= H]
            if not feas:
                continue
            by_ret = max(feas, key=lambda k: (exact_retention(w, h, k),
                                              A.ANCHOR_AREAS[k], -k))
            lar = math.log(w / h)
            by_log = min(feas, key=lambda k: (abs(math.log(A.ANCHOR_ARS[k]) - lar),
                                              -A.ANCHOR_AREAS[k], k))
            if by_ret != by_log:
                mism += 1
    check("argmax retention == argmin |dlog AR| over the grid", mism == 0, str(mism))


# --- REGRESSION #1: 阈值判据必须是精确整数比，不能看绝对尺寸抽签 -----------------------

def test_regression_threshold_is_scale_invariant() -> None:
    """曾经的缺陷：retention 用 `s = max(W/w,H/h)` 的浮点式算，于是数学上同一个
    AR 2.5 在不同绝对尺寸下得到相反判决 —— 在 (5t,2t) 上扫 t 有 9.30% 被误拒。
    现在判据是 `den*lo >= num*hi` 的整数比较，必须逐 t 一致。"""
    verdicts = set()
    rejected = []
    for t in range(176, 40000, 7):
        w, h = 5 * t, 2 * t                       # 精确 AR 2.5
        got = A.assign_anchor(w, h)
        verdicts.add(got is not None)
        if got is None:
            rejected.append((w, h))
    check("all exact-AR-2.5 sizes get the SAME verdict", len(verdicts) == 1,
          f"verdicts={verdicts} rejected_sample={rejected[:4]}")
    check("exact AR 2.5 is accepted (crop_loss == 0.20 exactly, boundary included)",
          verdicts == {True}, str(verdicts))
    # 审查里点名的三个尺寸，旧实现拒绝、新实现必须接受
    for w, h in [(1440, 576), (1165, 466), (1350, 540), (466, 1165)]:
        got = A.assign_anchor(w, h)
        ok = got is not None and float(exact_crop_loss(w, h, got[0])) == 0.2
        check(f"REGRESSION {w}x{h} (exact AR 2.5) accepted", ok, str(got))
    # 竖向对称：t 扫一遍
    vv = {A.assign_anchor(2 * t, 5 * t) is not None for t in range(176, 20000, 11)}
    check("portrait exact-AR-2.5 verdict is also uniform", len(vv) == 1, str(vv))
    # 略超阈值必须一致拒绝
    over = {A.assign_anchor(int(2.53 * t), t) is not None for t in range(400, 20000, 13)}
    check("AR 2.53 uniformly rejected", over == {False}, str(over))


# --- REGRESSION #2: 两个入口的判据必须逐位同一 ---------------------------------------

def test_regression_verify_binding_agrees_with_assign() -> None:
    """曾经的缺陷：`verify_binding` 用 `> crop_loss_max + 1e-9`，`assign_anchor` 用 `>`，
    于是 1440x576 被 assign 拒绝却被 verify 放行。"""
    disagree = []
    for w in range(340, 1500, 7):
        for h in range(340, 1500, 11):
            got = A.assign_anchor(w, h)
            for k in range(13):
                W, H = A.ANCHORS_V1_WH[k]
                if w < W or h < H:
                    continue
                ok_v = True
                try:
                    A.verify_binding(w, h, k)
                except A.AnchorGeometryError:
                    ok_v = False
                ok_a = float(exact_crop_loss(w, h, k)) <= A.CROP_LOSS_MAX
                if ok_v != ok_a:
                    disagree.append((w, h, k, ok_v, ok_a))
    check("verify_binding threshold == assign_anchor threshold on every feasible pair",
          not disagree, f"{len(disagree)} disagreements, e.g. {disagree[:3]}")
    # 点名的尺寸
    for w, h, k in [(1440, 576, 7), (1165, 466, 7), (1350, 540, 7), (466, 1165, 8)]:
        try:
            A.verify_binding(w, h, k)
            v = True
        except A.AnchorGeometryError:
            v = False
        check(f"REGRESSION verify_binding({w},{h},{k}) agrees with assign",
              v and A.assign_anchor(w, h) is not None, f"verify={v}")


# --- REGRESSION #3: 标量与向量化不能是两份实现 ---------------------------------------

def test_regression_scalar_is_a_wrapper_over_batch() -> None:
    """曾经的缺陷：两份实现的平局键不同（标量四级含 -retention，向量化只有 (-area,id)），
    在精确平局上返回不同 anchor。现在标量就是 batch 的封装。"""
    # 审查构造出的精确平局输入
    for w, h in [(109468101, 91971601), (133121449, 103115431), (470568128, 607500841)]:
        s = A.assign_anchor(w, h, crop_loss_max=1.0)
        b_ids, b_cls = A.assign_anchor_batch(np.asarray([w]), np.asarray([h]), crop_loss_max=1.0)
        ok = s is not None and s[0] == int(b_ids[0]) and s[1] == float(b_cls[0])
        check(f"REGRESSION constructed tie {w}x{h}: scalar == batch", ok,
              f"scalar={s} batch=({int(b_ids[0])}, {float(b_cls[0])})")
    # 大网格逐位一致
    ws, hs = [], []
    for w in range(180, 1500, 13):
        for h in range(180, 1500, 17):
            ws.append(w)
            hs.append(h)
    ids, cls = A.assign_anchor_batch(np.asarray(ws, dtype=np.int64),
                                    np.asarray(hs, dtype=np.int64))
    mism = 0
    for i, (w, h) in enumerate(zip(ws, hs)):
        got = A.assign_anchor(w, h)
        if got is None:
            if ids[i] != -1 or not math.isnan(float(cls[i])):
                mism += 1
        elif ids[i] != got[0] or float(cls[i]) != got[1]:
            mism += 1
    check(f"batch == scalar bit-for-bit over {len(ws)} sizes", mism == 0, str(mism))
    # crop_loss 必须等于精确值
    worst = 0.0
    for i, (w, h) in enumerate(zip(ws, hs)):
        if ids[i] >= 0:
            worst = max(worst, abs(float(cls[i]) - float(exact_crop_loss(w, h, int(ids[i])))))
    check("batch crop_loss matches the exact rational", worst <= 1e-15, f"{worst:.3e}")


# --- REGRESSION #4: 退化输入返回 None，不抛原生异常 ------------------------------------

def test_regression_degenerate_inputs_return_none() -> None:
    """曾经的缺陷：`0 < x < 1` 通过了 `x > 0` 判断，再被 int() 截成 0，
    然后 log(0) / 除零；nan/inf 也直接抛原生异常。向量化路径却返回 -1。"""
    for w, h in [(0.5, 600), (700, 0.5), (0.9, 0.9),
                 (float("nan"), 512), (512, float("nan")),
                 (float("inf"), 512), (512, float("-inf")),
                 (0, 512), (-8, 512), (511, 511)]:
        try:
            got = A.assign_anchor(w, h)
            ok = got is None
            detail = str(got)
        except Exception as exc:                               # noqa: BLE001
            ok = False
            detail = f"raised {type(exc).__name__}: {exc}"
        check(f"REGRESSION assign_anchor({w}, {h}) -> None", ok, detail)
    check("is_servable does not propagate an exception",
          A.is_servable(0.5, 600) is False)


# --- REGRESSION #5: 形状契约 ------------------------------------------------------

def test_regression_batch_shape_contract() -> None:
    """曾经的缺陷：`(n,1)` 形状被广播成 `n x n`（每行取值还是对的，所以 `[:,0]` 的
    调用方看不出来），在 10^9 行量级会吃爆内存。"""
    w2 = np.array([[767], [696], [1008]])
    h2 = np.array([[512], [510], [435]])
    ok, detail = raises(A.assign_anchor_batch, w2, h2, needle="1-D")
    check("REGRESSION (n,1)-shaped input raises instead of broadcasting", ok, detail)
    ok, detail = raises(A.assign_anchor_batch, np.zeros((2, 2)), np.zeros((2, 2)))
    check("REGRESSION 2-D input raises AnchorGeometryError", ok, detail)
    ok, detail = raises(A.assign_anchor_batch, np.zeros(3), np.zeros(4), needle="shape mismatch")
    check("mismatched lengths raise", ok, detail)
    ids, cls = A.assign_anchor_batch(np.asarray([767, 696]), np.asarray([512, 510]))
    check("1-D input returns equal-length 1-D output",
          ids.shape == (2,) and cls.shape == (2,), f"{ids.shape} {cls.shape}")
    ids0, _ = A.assign_anchor_batch(np.asarray(767), np.asarray(512))
    check("0-D input is accepted and returns length-1", ids0.shape == (1,), str(ids0.shape))
    ide, cle = A.assign_anchor_batch(np.asarray([], dtype=np.int64),
                                    np.asarray([], dtype=np.int64))
    check("empty input returns empty output", ide.shape == (0,) and cle.shape == (0,))
    # dtype 不该影响结果
    ref, _ = A.assign_anchor_batch(np.asarray([767], dtype=np.int64), np.asarray([512]))
    for dt in (np.int32, np.uint16, np.float32, np.float64):
        got, _ = A.assign_anchor_batch(np.asarray([767], dtype=dt), np.asarray([512], dtype=dt))
        check(f"dtype {np.dtype(dt).name} agrees with int64", got[0] == ref[0], str(got))
    got, _ = A.assign_anchor_batch([767, 696], [512, 510])
    check("python lists are accepted", got.tolist() == [3, 1], str(got))


# --- REGRESSION #6: 超大尺寸显式报错，不静默回绕 --------------------------------------

def test_regression_oversized_dims_raise() -> None:
    """曾经的缺陷：`area_gate` 下 `w*h` 在 int64 静默回绕，
    4294967296x4294967296 标量接受、向量化拒绝。现在两条路径都显式报错。"""
    for w, h in [(2 ** 31, 512), (512, 2 ** 31), (4294967296, 4294967296), (10 ** 10, 10 ** 10)]:
        ok_s, d_s = raises(A.assign_anchor, w, h, needle="MAX_DIM")
        ok_b, d_b = raises(A.assign_anchor_batch,
                           np.asarray([w], dtype=object).astype(np.float64),
                           np.asarray([h], dtype=object).astype(np.float64),
                           needle="MAX_DIM")
        check(f"REGRESSION {w}x{h} raises in both paths", ok_s and ok_b, f"{d_s} | {d_b}")
    # 恰好在界内必须正常工作
    big = 2 ** 31 - 1
    got = A.assign_anchor(big, big)
    check("largest in-range square size still assigns to the square anchor",
          got is not None and got[0] == 0, str(got))


# --- 零上采样（本模块存在的唯一理由） -----------------------------------------------

def test_zero_upsampling_exhaustive() -> None:
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
    check(f"zero upsampling over {n_ok} accepted sizes", not bad, f"violations {bad[:5]}")
    check("grid dense enough to be meaningful", n_ok > 10000, str(n_ok))


def test_exact_anchor_size_has_zero_loss() -> None:
    bad = [(k, W, H, A.assign_anchor(W, H))
           for k, (W, H) in enumerate(A.ANCHORS_V1_WH)
           if A.assign_anchor(W, H) != (k, 0.0)]
    check("exact anchor size -> itself with crop_loss 0", not bad, str(bad))


# --- REGRESSION #8: anchors_hw() 交给 legacy 指派器会上采样 ---------------------------

def test_regression_legacy_assigners_would_upsample() -> None:
    """`anchors_hw()` 只能用于**记录**几何身份。legacy 指派器没有可行集判据，
    把 13 锚表喂进去会放大图像。这里把这个事实钉住，确保 docstring 的警告不会腐化。"""
    try:
        from training.dataline.multi_ar import (assign_adaptive_bucket, assign_bucket,
                                                square_index)
    except Exception as exc:                                   # noqa: BLE001
        check("multi_ar importable", False, str(exc))
        return
    tbl = A.anchors_hw()
    check("square_index(anchors_hw()) == square_anchor_id()",
          square_index(tbl) == A.square_anchor_id())
    upsamples = []
    for w, h in [(400, 300), (200, 200), (64, 64)]:
        k = assign_bucket(w, h, tbl)
        H, W = tbl[k]
        if w < W or h < H:
            upsamples.append((w, h, W, H))
    check("REGRESSION legacy assign_bucket DOES upsample on the anchor table "
          "(so the docstring warning must stay)", len(upsamples) == 3, str(upsamples))
    k = assign_adaptive_bucket(200, 200, tbl, 262144)
    H, W = tbl[k]
    check("REGRESSION legacy assign_adaptive_bucket also upsamples", 200 < W and 200 < H,
          f"200x200 -> {W}x{H}")
    check("identity() no longer claims upsampling is impossible unconditionally",
          "legacy" in A.identity()["upsampling"])


# --- 真实池样本（锚为手工核对，crop_loss 精确算出）------------------------------------

def test_real_pool_rows() -> None:
    """取自 t5_manifest/rich_v1 首个 shard 的真实行（monet_0000 / gpic_0000）。
    期望**锚**是手工核对的离散结论；期望 crop_loss 由 Fraction 精确算出。"""
    cases = [
        (767, 512, (624, 416), "monet cc12m ge512"),
        (696, 510, (576, 432), "monet cc12m res256; 510<512 so square infeasible"),
        (1008, 435, (704, 352), "monet cc12m elongated -> 2:1 terminal"),
        (825, 510, (624, 416), "monet cc12m; the 1.75 anchor is further in log-AR"),
        (639, 428, (624, 416), "gpic res256"),
        (427, 640, (416, 624), "gpic res256 portrait"),
        (482, 640, (432, 576), "gpic res256 portrait"),
        (640, 640, (512, 512), "gpic ge512 square"),
    ]
    for w, h, expect_wh, note in cases:
        got = A.assign_anchor(w, h)
        if got is None:
            check(f"real row {w}x{h}", False, "unexpectedly unservable")
            continue
        aid, cl = got
        exact = float(exact_crop_loss(w, h, aid))
        ok = A.ANCHORS_V1_WH[aid] == expect_wh and abs(cl - exact) <= 1e-15
        check(f"real row {w}x{h} -> {expect_wh[0]}x{expect_wh[1]} cl={exact:.9f}  ({note})",
              ok, f"got anchor {A.ANCHORS_V1_WH[aid]} cl={cl!r} exact={exact!r}")


# --- 可选面积门 ------------------------------------------------------------------

def test_area_gate_semantics() -> None:
    got = A.assign_anchor(720, 360, area_gate=None)
    check("720x360 (area 259,200) servable at zero upsampling",
          got is not None and A.ANCHORS_V1_WH[got[0]] == (704, 352), str(got))
    check("720x360 rejected under the audited area gate",
          A.assign_anchor(720, 360, area_gate=A.AREA_GATE_STAGE512A) is None)
    check("1024x256 has area >= gate but an empty feasible set",
          1024 * 256 >= A.AREA_GATE_STAGE512A
          and A.assign_anchor(1024, 256, area_gate=A.AREA_GATE_STAGE512A) is None
          and A.assign_anchor(1024, 256, area_gate=None) is None)
    check("704x352 (the exact 2:1 anchor) is itself below the area gate",
          A.assign_anchor(704, 352, area_gate=None) is not None
          and A.assign_anchor(704, 352, area_gate=A.AREA_GATE_STAGE512A) is None)
    check("768x384 survives the area gate",
          A.assign_anchor(768, 384, area_gate=A.AREA_GATE_STAGE512A) == (7, 0.0))


# --- 阈值参数 --------------------------------------------------------------------

def test_threshold_ratio_is_exact_for_the_default() -> None:
    num, den = A._threshold_ratio(0.20)
    check("1 - 0.20 == 4/5 exactly", (num, den) == (4, 5), f"{num}/{den}")
    check("crop_loss_max=0 accepts only exact-AR matches",
          A.assign_anchor(1024, 1024, crop_loss_max=0.0) == (0, 0.0)
          and A.assign_anchor(1030, 1024, crop_loss_max=0.0) is None)
    check("crop_loss_max=1.0 means no threshold (1-1 = 0/1, lo >= 0 always)",
          A._threshold_ratio(1.0) == (0, 1), str(A._threshold_ratio(1.0)))
    for bad in (-0.1, 1.5, float("nan")):
        ok, detail = raises(A._threshold_ratio, bad)
        check(f"crop_loss_max={bad} rejected", ok, detail)


# --- 运行期绑定校验（EXIF） --------------------------------------------------------

def test_verify_binding() -> None:
    try:
        A.verify_binding(1008, 435, 7)
        A.verify_binding(704, 352, 7)
        check("verify_binding accepts consistent decode", True)
    except A.AnchorGeometryError as exc:
        check("verify_binding accepts consistent decode", False, str(exc))
    ok, d = raises(A.verify_binding, 352, 704, 7, needle="upsample would be required")
    check("EXIF-swapped decode fails closed", ok, d)
    ok, d = raises(A.verify_binding, 703, 352, 7, needle="upsample would be required")
    check("one-pixel-short decode fails closed", ok, d)
    ok, d = raises(A.verify_binding, 1024, 1024, 13)
    check("out-of-range anchor_id fails closed", ok, d)
    ok, d = raises(A.verify_binding, 0, 512, 0)
    check("degenerate decode size fails closed", ok, d)
    ok, d = raises(A.verify_binding, 890, 352, 7, needle="crop_loss")
    check("over-threshold crop_loss fails closed", ok, d)
    ok, d = raises(A.verify_binding, float("nan"), 352, 7)
    check("nan decode size fails closed", ok, d)


# --- 概率向量 --------------------------------------------------------------------

def test_probs_from_mapping() -> None:
    p = A.probs_from_mapping({"624x416": 2.0, "512x512": 1.0, 3: 0.0})
    ok = (len(p) == 13 and abs(sum(p) - 1.0) < 1e-12
          and abs(p[3] - 2.0 / 3.0) < 1e-12 and abs(p[0] - 1.0 / 3.0) < 1e-12
          and all(p[i] == 0.0 for i in range(13) if i not in (0, 3)))
    check("probs_from_mapping normalizes by name and id", ok, str(p))
    for bad in ({"624x416": -1.0, "512x512": 2.0}, {"999x999": 1.0}, {"512x512": 0.0},
                {"13": 1.0}, {True: 1.0}):
        ok, detail = raises(A.probs_from_mapping, bad)
        check(f"probs_from_mapping rejects {bad}", ok, detail)


def test_identity_payload() -> None:
    import json
    ident = A.identity()
    try:
        json.dumps(ident)
        ser = True
    except TypeError:
        ser = False
    ok = (ser and ident["bucket_mode"] == "anchors_v1" and ident["anchor_set_id"] == "B13"
          and ident["n_anchors"] == 13 and len(ident["anchors_wh"]) == 13
          and ident["crop_loss_max"] == 0.20 and "exact integers" in ident["acceptance_test"])
    check("identity() is serializable and complete", ok, str(ident)[:200])


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
    print("ANCHORS_V1_GEOMETRY_TESTS_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
