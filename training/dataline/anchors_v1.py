"""Stage512b-real 固定锚集几何（`bucket_mode: anchors_v1`）。

与 Stage512a 的 `build_adaptive_buckets()`（361 桶枚举 + `bucket_prob_floor` 剪枝 +
`nearest_allowed_bucket()` 折叠）并列的**第二条几何路径**。legacy 路径一行不改。

为什么换：361 桶里只有 104 个落在高面积带（`min(w,h)>=512` 的资格门保证面积 >= 262144，
低面积桶结构上不可达），实际被填的 98 个再经 `bucket_prob_floor=0.005` 折叠后只剩
**30 个活桶**。折叠要把图重指派到"最近保留桶"，跨桶 re-resize 就是上采样的入口。
固定 13 锚把"枚举 + 剪枝 + 折叠"三步压成一步，且指派规则本身即可证明零上采样。

核心契约（唯一约束是**只下采样、绝不上采样**）：

    F(i)     = { k : w_i >= W_k  and  h_i >= H_k }      # 可行集：源图能完整覆盖锚框
    final(i) = argmax_{k in F(i)} retention_k           # 平局用 (-area, anchor_id)
    F(i) = ∅ 或 所有可行锚 crop_loss > T  -> 拒绝

`w >= W_k and h >= H_k` 使 `s = max(W_k/w, H_k/h) <= 1`，所以零上采样是**可行集的结构性质**，
不是下游补的检查。

## retention 是有理数，不能用浮点算判据

设 `s = max(W/w, H/h)`，则
    AR_src > AR_anchor  =>  s = H/h  =>  retention = W*h/(w*H)
    AR_src < AR_anchor  =>  s = W/w  =>  retention = H*w/(h*W)
即 **retention = min(w*H, h*W) / max(w*H, h*W)** —— 两个整数之比，只依赖 AR，与绝对尺寸无关。

推论 1：`|log AR_src - log AR_anchor| = -log(retention)`，所以
**`argmin |ΔlogAR|` 与 `argmax retention` 是同一个 argmax**，一个键就够，不需要两级。
推论 2：判据 `crop_loss <= T` ⟺ `retention >= 1-T` ⟺ **`den*lo >= num*hi`**（整数比较），
其中 `1-T = num/den`。默认 T=0.20 时就是 `5*lo >= 4*hi`。

**这条必须用整数算。**早先版本用 `s` 的浮点式算 retention，判据在 T 边界上就变成了
"看绝对像素大小抽签"：数学上完全相同的 AR 2.5，`1000x400` / `1280x512` 被接受而
`1165x466` / `1350x540` / `1440x576` 被拒；在 `(5t,2t)` 上扫 t 有 **9.30%** 的
精确 AR 2.5 尺寸被误拒。整数判据把这条彻底消掉，且让 catalog 可逐位复现。

方向约定：本模块对外一律 `{width, height}`。
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

# --- 冻结锚表 -------------------------------------------------------------------
# anchor_id 顺序必须与 Stage512b 审计产出一致（Part 1 / Part 1.5 / Part 2 的
# per_anchor[i]["anchor_id"] == i），否则所有已审计的行数、概率、配比全部错位。
ANCHORS_V1_WH: tuple[tuple[int, int], ...] = (
    (512, 512),   # 0   1:1
    (576, 432),   # 1   4:3
    (432, 576),   # 2   3:4
    (624, 416),   # 3   3:2
    (416, 624),   # 4   2:3
    (672, 384),   # 5   7:4
    (384, 672),   # 6   4:7
    (704, 352),   # 7   2:1      extent（横）
    (352, 704),   # 8   1:2      extent（竖）
    (448, 560),   # 9   4:5
    (560, 448),   # 10  5:4
    (480, 544),   # 11  15:17
    (544, 480),   # 12  17:15
)

ANCHOR_SET_ID = "B13"
BUCKET_MODE = "anchors_v1"

#: 指派后仍超过该 crop_loss 的行一律拒绝。等价于"源 AR 与锚 AR 之比落在 [0.8, 1.25]"，
#: 即终端 2:1 锚只服务到 AR 2.5。
CROP_LOSS_MAX = 0.20

#: Stage512a 的源准入面积门。**默认关闭**：`F(i) != ∅` 已保证零上采样，面积门是更严的
#: 独立约束（最小锚面积 704x352 = 247,808 < 262,144，两者不等价）。
#: 传 262144 可逐行复现 Stage512b 审计（Part 1 / 1.5 / 2）的口径。
AREA_GATE_STAGE512A = 262144

#: 边长上界。整数判据用 int64 做 `den * (w*H)` 的乘法，取 den <= 10**6、边长 < 2**31 时
#: 最大约 1.5e18 < 9.22e18，不会溢出。超界**显式报错**，不静默回绕。
#: （早先版本在 `w*h` 上静默回绕，导致 4294967296x4294967296 在标量路径被接受、
#: 向量化路径被拒。）
MAX_DIM = 2 ** 31

#: `1 - CROP_LOSS_MAX` 的有理数分母上界。
_MAX_DEN = 10 ** 6

_N = len(ANCHORS_V1_WH)


class AnchorGeometryError(RuntimeError):
    """锚几何契约被破坏（fail-closed，不静默降级）。"""


def _validate_table() -> None:
    assert _N == 13, f"B13 must have 13 anchors, got {_N}"
    assert len(set(ANCHORS_V1_WH)) == _N, "duplicate anchor"
    for w, h in ANCHORS_V1_WH:
        assert w % 16 == 0 and h % 16 == 0, f"anchor {w}x{h} not divisor-16 aligned"
        assert 0.5 - 1e-12 <= w / h <= 2.0 + 1e-12, f"anchor {w}x{h} outside 1:2..2:1"
        assert (h, w) in ANCHORS_V1_WH, f"anchor {w}x{h} has no transposed partner"
    assert ANCHORS_V1_WH[3] == (624, 416), "anchor_id 3 must be 624 wide x 416 high"
    assert ANCHORS_V1_WH[0] == (512, 512), "anchor_id 0 must be the square anchor"
    assert sum(1 for w, h in ANCHORS_V1_WH if w == h) == 1, "need exactly one square anchor"


_validate_table()

ANCHOR_AREAS: tuple[int, ...] = tuple(w * h for w, h in ANCHORS_V1_WH)
ANCHOR_ARS: tuple[float, ...] = tuple(w / h for w, h in ANCHORS_V1_WH)
MIN_ANCHOR_AREA = min(ANCHOR_AREAS)
MIN_ANCHOR_SIDE = min(min(w, h) for w, h in ANCHORS_V1_WH)
#: 最长锚边（704）。Stage512a 实际只用到 672，故 shape smoke 必须覆盖 704。
MAX_ANCHOR_SIDE = max(max(w, h) for w, h in ANCHORS_V1_WH)

#: 平局键 `(-area, anchor_id)` 压成一个整数：面积大者优先，同面积取小 id。
_TIEBREAK: tuple[int, ...] = tuple(-(ANCHOR_AREAS[k] * 16) + k for k in range(_N))


def anchors_wh() -> list[tuple[int, int]]:
    """`[(width, height), ...]`，按 anchor_id 升序。"""
    return list(ANCHORS_V1_WH)


def anchors_hw() -> list[tuple[int, int]]:
    """`[(H, W), ...]` —— **仅用于记录几何身份的产物**：plan / catalog header 的
    `buckets` 字段、`BucketSchedule` 的长度与 `square_index()` 校验。

    ⚠️ **绝对不要把它交给 `multi_ar.assign_bucket()` / `assign_adaptive_bucket()` /
    `Bucketer`。**那三个 legacy 指派器**没有可行集判据**，把 13 锚表喂进去会产生上采样：
    实测 `assign_bucket(400, 300, anchors_hw())` 落到 (432,576) → 放大 1.44×；
    `assign_adaptive_bucket(200, 200, ...)` 落到 512×512 → 放大 2.56×。
    零上采样保证**只**经由本模块的 `assign_anchor*()` 与 `verify_binding()` 成立。
    """
    return [(h, w) for (w, h) in ANCHORS_V1_WH]


def square_anchor_id() -> int:
    return 0


# --- 精确判据 --------------------------------------------------------------------

def _threshold_ratio(crop_loss_max: float) -> tuple[int, int]:
    """把 `1 - crop_loss_max` 转成分母受限的有理数 `(num, den)`。0.20 -> (4, 5)。"""
    # 1.0 = "不设阈值"，是合法值：1-1.0 = 0 => 判据变成 lo >= 0，恒真。
    if not (0.0 <= crop_loss_max <= 1.0):
        raise AnchorGeometryError(f"crop_loss_max must be in [0,1], got {crop_loss_max}")
    fr = Fraction(1.0 - float(crop_loss_max)).limit_denominator(_MAX_DEN)
    return fr.numerator, fr.denominator


def _coerce_dim(x, name: str) -> int:
    """把尺寸收成正整数；非有限 / 非正 / 超界都在这里判掉。返回 0 表示"不可服务"。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise AnchorGeometryError(f"{name} is not a number: {x!r}") from None
    if not math.isfinite(v):
        return 0
    if v >= MAX_DIM:
        raise AnchorGeometryError(
            f"{name}={x} exceeds MAX_DIM={MAX_DIM}; exact integer arithmetic would overflow")
    iv = int(v)
    return iv if iv > 0 else 0


def crop_loss_wh(w: int, h: int, anchor_id: int) -> float:
    """把 `w x h` 指派到 `anchor_id` 时丢弃的源像素比例，由整数比精确算出。"""
    if not (0 <= anchor_id < _N):
        raise AnchorGeometryError(f"anchor_id {anchor_id} out of range [0,{_N})")
    wi = _coerce_dim(w, "w")
    hi = _coerce_dim(h, "h")
    if wi == 0 or hi == 0:
        raise AnchorGeometryError(f"bad size {w}x{h}")
    W, H = ANCHORS_V1_WH[anchor_id]
    a, b = wi * H, hi * W
    lo, hi_ = (a, b) if a <= b else (b, a)
    return 1.0 - lo / hi_


def assign_anchor(w, h, *, crop_loss_max: float = CROP_LOSS_MAX,
                  area_gate: int | None = None) -> tuple[int, float] | None:
    """源尺寸 `(w, h)` -> `(anchor_id, crop_loss)`；不可服务返回 `None`。

    这是 `assign_anchor_batch` 的单点封装 —— **只有一份判定实现**，
    所以标量与向量化路径不可能分叉（早先两份实现在精确平局上会给出不同 anchor）。
    """
    ids, cls = assign_anchor_batch(np.asarray([w]), np.asarray([h]),
                                   crop_loss_max=crop_loss_max, area_gate=area_gate)
    k = int(ids[0])
    return None if k < 0 else (k, float(cls[0]))


def is_servable(w, h, *, crop_loss_max: float = CROP_LOSS_MAX,
                area_gate: int | None = None) -> bool:
    return assign_anchor(w, h, crop_loss_max=crop_loss_max, area_gate=area_gate) is not None


def assign_anchor_batch(w, h, *, crop_loss_max: float = CROP_LOSS_MAX,
                        area_gate: int | None = None):
    """向量化指派。返回 `(anchor_id, crop_loss)` 两个与输入**等长**的 1-D 数组；
    不可服务的行 `anchor_id = -1`、`crop_loss = nan`。

    **必须存在的理由**：既有 `vectorized_adaptive_assign()`（`phase5_build_catalog.py:316`,
    `phase3a_bucket_stats.py:329`）走 `(h//16, w//16)` 稠密查表 + "查不到退标量循环"。
    13 锚下那张表 99.9% 是 -1，于是**每一行**都走 Python 标量分支 —— 全量池约 10^9 行，
    目录构建会静默地永远跑不完（不报错，只是不结束）。所以锚路径必须按锚循环
    （13 次 numpy 全列运算）而不是按行循环。

    只接受 0-D 或 1-D 输入。早先版本只校验 `w.shape != h.shape`，`(n,1)` 形状会广播成
    `n x n`（每行取值还都是对的，所以 `[:,0]` 的调用方看不出来，`.ravel()` 的调用方
    拿到 n^2 个值），在 10^9 行的量级上会直接吃爆内存。
    """
    w_arr = np.asarray(w)
    h_arr = np.asarray(h)
    if w_arr.ndim > 1 or h_arr.ndim > 1:
        raise AnchorGeometryError(
            f"expected 0-D or 1-D input, got shapes {w_arr.shape} and {h_arr.shape}")
    w_arr = np.atleast_1d(w_arr)
    h_arr = np.atleast_1d(h_arr)
    if w_arr.shape != h_arr.shape:
        raise AnchorGeometryError(f"w/h shape mismatch: {w_arr.shape} vs {h_arr.shape}")

    num, den = _threshold_ratio(crop_loss_max)
    n = w_arr.size

    # 非有限 -> 0（不可服务）；超界 -> 报错。整数化后才做正负判断，
    # 否则 0 < x < 1 的小数会通过 `x > 0` 然后被 int() 截成 0 并引发除零 / log(0)。
    wf = np.asarray(w_arr, dtype=np.float64)
    hf = np.asarray(h_arr, dtype=np.float64)
    if np.any(np.isfinite(wf) & (np.abs(wf) >= MAX_DIM)) or \
       np.any(np.isfinite(hf) & (np.abs(hf) >= MAX_DIM)):
        raise AnchorGeometryError(
            f"a dimension exceeds MAX_DIM={MAX_DIM}; exact integer arithmetic would overflow")
    wi = np.where(np.isfinite(wf), wf, 0.0).astype(np.int64)
    hi = np.where(np.isfinite(hf), hf, 0.0).astype(np.int64)

    valid = (wi > 0) & (hi > 0)
    if area_gate is not None:
        # wi,hi < 2**31 => 乘积 < 2**62，int64 安全
        valid &= (wi * hi) >= int(area_gate)

    best_ret = np.full(n, -1.0, dtype=np.float64)
    best_tb = np.zeros(n, dtype=np.int64)
    best_id = np.full(n, -1, dtype=np.int64)

    for k in range(_N):
        W, H = ANCHORS_V1_WH[k]
        feasible = valid & (wi >= W) & (hi >= H)
        if not feasible.any():
            continue
        a = wi * H
        b = hi * W
        lo = np.minimum(a, b)
        hi_ = np.maximum(a, b)
        # 精确判据：retention >= num/den  <=>  den*lo >= num*hi
        feasible &= (den * lo) >= (num * hi_)
        if not feasible.any():
            continue
        ret = np.where(feasible, lo / np.maximum(hi_, 1), -1.0)
        tb = _TIEBREAK[k]
        take = feasible & ((ret > best_ret) | ((ret == best_ret) & (tb < best_tb)))
        best_ret = np.where(take, ret, best_ret)
        best_tb = np.where(take, tb, best_tb)
        best_id = np.where(take, k, best_id)

    crop_loss = np.where(best_id >= 0, 1.0 - best_ret, np.nan)
    return best_id, crop_loss


def verify_binding(w, h, anchor_id: int, *,
                   crop_loss_max: float = CROP_LOSS_MAX,
                   context: str = "") -> None:
    """训练期**校验**离线 plan 给定的 (解码尺寸, anchor_id) 绑定，不重算指派。

    离线 plan 已写死每行的 anchor_id；运行期唯一的风险是**解码尺寸与 manifest 不一致**
    （EXIF 旋转最现实：manifest 记 704x352，解码出来 352x704）。
    `resize_crop_to_bucket()` 的 `scale = max(W/w0, H/h0)` **从不拒绝上采样**，
    所以零上采样保证只能落在这个校验上。fail-closed。

    判据与 `assign_anchor` **逐位同一**（都走 `den*lo >= num*hi`）。早先版本这里多加了
    `+ 1e-9` 容差，于是 `1440x576` 这类恰在阈值上的尺寸被 `assign_anchor` 拒绝、
    却能通过 `verify_binding` —— 两个入口对同一行给出相反结论。
    """
    if not (0 <= anchor_id < _N):
        raise AnchorGeometryError(f"anchor_id {anchor_id} out of range [0,{_N}) {context}")
    W, H = ANCHORS_V1_WH[anchor_id]
    wi = _coerce_dim(w, "w")
    hi = _coerce_dim(h, "h")
    if wi == 0 or hi == 0:
        raise AnchorGeometryError(f"bad decoded size {w}x{h} {context}")
    if wi < W or hi < H:
        raise AnchorGeometryError(
            f"upsample would be required: decoded {wi}x{hi} cannot cover anchor "
            f"{anchor_id} ({W}x{H}) {context} -- likely EXIF rotation or a stale manifest")
    num, den = _threshold_ratio(crop_loss_max)
    a, b = wi * H, hi * W
    lo, hi_ = (a, b) if a <= b else (b, a)
    if den * lo < num * hi_:
        raise AnchorGeometryError(
            f"crop_loss {1.0 - lo / hi_:.6f} > {crop_loss_max} for decoded {wi}x{hi} "
            f"-> anchor {anchor_id} ({W}x{H}) {context}")


def params_from_values(area_gate, crop_loss_max, crop_loss_max_by_domain=None) -> dict:
    """把配置里的锚参数归一化成 `mar_params` 的增补字典。**唯一来源。**

    phase5 用 `report_config["multi_ar_params"] != runtime["mar_params"]` 校验它与
    phase3a 报告的一致性，所以两边**必须走同一个表达式**：任何一边自己写一遍归一化，
    这个等值门就永远过不了，锚模式端到端不可达。同理，per-domain 覆盖必须归一化成
    **确定性的**形状（排序后的普通 dict），否则两边键序不同就会让等值门假阴性。

    `area_gate`：`None` 或 `0` = 关闭（服务性纯由可行锚判定）；正整数 = 额外面积门。
    `crop_loss_max_by_domain`：可选的按域收紧。文字域需要它 —— TextAtlas 在全局
    0.20 下有约 2,388 行会被切掉标题字形，而 caption 仍然引用完整字符串，等于
    在教模型渲染画面里没有的字。
    """
    if area_gate is None:
        raise AnchorGeometryError(
            "anchors_area_gate must be stated explicitly (0 disables it; "
            f"{AREA_GATE_STAGE512A} reproduces the Stage512b audits)")
    gate = int(area_gate)
    if gate < 0:
        raise AnchorGeometryError(f"anchors_area_gate must be >= 0, got {gate}")
    clm = CROP_LOSS_MAX if crop_loss_max is None else float(crop_loss_max)
    if not (0.0 <= clm <= 1.0):
        raise AnchorGeometryError(f"anchors_crop_loss_max must be in [0,1], got {clm}")
    by_domain: dict[str, float] | None = None
    if crop_loss_max_by_domain:
        by_domain = {}
        for key in sorted(crop_loss_max_by_domain):
            value = float(crop_loss_max_by_domain[key])
            if not (0.0 <= value <= 1.0):
                raise AnchorGeometryError(
                    f"anchors_crop_loss_max_by_domain[{key!r}] must be in [0,1], got {value}")
            if value > clm:
                # A per-domain value ABOVE the global one would loosen the contract
                # for that domain while the report still advertises the global
                # figure.  Overrides are for tightening only.
                raise AnchorGeometryError(
                    f"anchors_crop_loss_max_by_domain[{key!r}]={value} exceeds the global "
                    f"anchors_crop_loss_max={clm}; per-domain overrides may only tighten")
            by_domain[str(key)] = value
    return {"area_gate": gate or None, "crop_loss_max": clm,
            "crop_loss_max_by_domain": by_domain}


def crop_loss_for_domain(mar_params: dict, domain: str) -> float:
    """The threshold that actually applies to `domain`.

    Both phase3a and phase5 must resolve this identically or their N matrices
    diverge in a way no equality gate would catch: `mar_params` would match while
    the per-row admission differed.
    """
    by_domain = mar_params.get("crop_loss_max_by_domain") or {}
    default = mar_params.get("crop_loss_max")
    return float(by_domain.get(domain, CROP_LOSS_MAX if default is None else default))


def probs_from_mapping(mapping: dict) -> list[float]:
    """把 config 里的 `{anchor "624x416" | anchor_id: prob}` 归一化成 13 长概率向量。

    概率向量是**数据侧设计产物**（依赖池选择与配比），不在本模块冻结；
    由 plan/config 提供，`BucketSchedule` 消费。这里只做形状与归一化校验。
    """
    probs = [0.0] * _N
    name_to_id = {f"{w}x{h}": i for i, (w, h) in enumerate(ANCHORS_V1_WH)}
    for key, val in mapping.items():
        if isinstance(key, bool):
            raise AnchorGeometryError(f"anchor key {key!r} must not be a bool")
        if isinstance(key, int):
            idx = key
        elif key in name_to_id:
            idx = name_to_id[key]
        else:
            try:
                idx = int(key)
            except (TypeError, ValueError):
                raise AnchorGeometryError(
                    f"anchor key {key!r} is neither a WxH anchor name nor an anchor_id") from None
        if not (0 <= idx < _N):
            raise AnchorGeometryError(f"anchor key {key!r} out of range")
        if val < 0:
            raise AnchorGeometryError(f"negative probability for {key!r}: {val}")
        probs[idx] += float(val)
    total = sum(probs)
    if total <= 0:
        raise AnchorGeometryError("anchor probabilities sum to zero")
    return [p / total for p in probs]


def identity(crop_loss_max: float = CROP_LOSS_MAX, area_gate: int | None = None) -> dict:
    """写入 catalog / plan header 的几何身份。

    `crop_loss_max` / `area_gate` **必须传入实际生效值**：写死模块常量会让
    `assignment_hash_payload` 里的 `anchor_identity` 在配置用了非默认阈值时谎报，
    而紧邻的 `payload["crop_loss_max"]` 报的是真值 —— 同一个 payload 自相矛盾。
    """
    return {
        "bucket_mode": BUCKET_MODE,
        "anchor_set_id": ANCHOR_SET_ID,
        "n_anchors": _N,
        "anchors_wh": [list(a) for a in ANCHORS_V1_WH],
        "assignment": "argmax over feasible set of retention, ties by (-area, anchor_id)",
        "feasible_set": "w >= W_k and h >= H_k",
        "retention": "min(w*H_k, h*W_k) / max(w*H_k, h*W_k)  (exact rational)",
        "crop_loss_max": float(crop_loss_max),
        "source_area_gate": None if area_gate is None else int(area_gate),
        "acceptance_test": "den*min >= num*max where (1-crop_loss_max) = num/den (exact integers)",
        "upsampling": ("structurally impossible via assign_anchor*/verify_binding "
                       "(feasible set requires w>=W and h>=H); NOT guaranteed if the anchor "
                       "table is fed to the legacy multi_ar assigners, which have no "
                       "feasibility gate"),
    }
