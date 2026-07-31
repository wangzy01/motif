"""Stage512b-real 固定锚集几何（`bucket_mode: anchors_v1`）。

与 Stage512a 的 `build_adaptive_buckets()`（361 桶枚举 + `bucket_prob_floor` 剪枝 +
`nearest_allowed_bucket()` 折叠）并列的**第二条几何路径**。legacy 路径一行不改。

为什么换：361 桶里只有 104 个落在高面积带（`min(w,h)>=512` 的资格门保证面积 >= 262144，
低面积桶结构上不可达），实际被填的 98 个再经 `bucket_prob_floor=0.005` 折叠后只剩
**30 个活桶**（AR 约 0.571–1.75）。折叠本身要把图重指派到"最近保留桶"，跨桶 re-resize
会引入上采样。固定 13 锚把"枚举 + 剪枝 + 折叠"三步压成一步查表，活锚数从 30 提到 13
**但全部真实可调度**，且指派规则本身即可证明零上采样。

核心契约（唯一约束是**只下采样、绝不上采样**）：

    F(i)     = { k : w_i >= W_k  and  h_i >= H_k }          # 可行集：源图能完整覆盖锚框
    final(i) = argmin_{k in F(i)} ( |log AR_i - log AR_k|, -retention, -area, anchor_id )
    F(i) = ∅              -> 拒绝
    crop_loss(final) > T  -> 拒绝                            # T = CROP_LOSS_MAX = 0.20

    s          = max(W_k / w, H_k / h)          # w>=W_k 且 h>=H_k  =>  s <= 1  => 只缩小
    retention  = (W_k * H_k) / ((w * s) * (h * s))
    crop_loss  = 1 - retention

`w >= W_k and h >= H_k` 是零上采样的**充分条件**，也是本模块唯一的准入依据。
面积门（Stage512a 的 `w*h >= 262144`）**不是**零上采样所必需的，见 `area_gate` 参数说明。

方向约定：本模块对外一律 `{width, height}`；`anchors_hw()` 才返回既有代码期望的 `(H, W)`。
"""
from __future__ import annotations

import math

# --- 冻结锚表 -------------------------------------------------------------------
# anchor_id 顺序必须与 Stage512b 审计产出一致（Part 1 / Part 1.5 / Part 2 的
# per_anchor[i]["anchor_id"] == i），否则所有已审计的行数、概率、配比全部错位。
# 审计报告路径见 docs/STAGE512B_REAL_RESEARCH_BRIEF.md。
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

#: 指派后仍超过该 crop_loss 的行一律拒绝。
#: 因为 retention 只依赖 AR（见 `crop_loss_wh` 的推导），0.20 等价于
#: "源图 AR 与锚 AR 之比落在 [0.8, 1.25] 内"，即终端 2:1 锚只服务到 AR 2.5。
CROP_LOSS_MAX = 0.20

#: Stage512a 的源准入面积门。**默认关闭**：`F(i) != ∅` 已经保证零上采样，
#: 面积门是更严的独立约束，会拒掉一批本可零上采样服务的细长图
#: （最小锚面积 704x352 = 247,808 < 262,144，故两者不等价）。
#: 传 262144 可逐行复现 Stage512b 审计（Part 1 / 1.5 / 2）的口径。
AREA_GATE_STAGE512A = 262144

_N = len(ANCHORS_V1_WH)


class AnchorGeometryError(RuntimeError):
    """锚几何契约被破坏（fail-closed，不静默降级）。"""


def _validate_table() -> None:
    assert _N == 13, f"B13 must have 13 anchors, got {_N}"
    assert len(set(ANCHORS_V1_WH)) == _N, "duplicate anchor"
    for w, h in ANCHORS_V1_WH:
        assert w % 16 == 0 and h % 16 == 0, f"anchor {w}x{h} not divisor-16 aligned"
        assert 0.5 - 1e-12 <= w / h <= 2.0 + 1e-12, f"anchor {w}x{h} outside 1:2..2:1"
    # 对称性：除 1:1 外每个锚都有转置搭档
    for w, h in ANCHORS_V1_WH:
        assert (h, w) in ANCHORS_V1_WH, f"anchor {w}x{h} has no transposed partner"
    # 方向自检：横向锚 width > height
    assert ANCHORS_V1_WH[3] == (624, 416), "anchor_id 3 must be 624 wide x 416 high"
    assert ANCHORS_V1_WH[0] == (512, 512), "anchor_id 0 must be the square anchor"


_validate_table()

ANCHOR_AREAS: tuple[int, ...] = tuple(w * h for w, h in ANCHORS_V1_WH)
ANCHOR_ARS: tuple[float, ...] = tuple(w / h for w, h in ANCHORS_V1_WH)
_ANCHOR_LOG_ARS: tuple[float, ...] = tuple(math.log(a) for a in ANCHOR_ARS)

#: 最小锚面积——低于此面积的图不可能覆盖任何锚，可用于快速预筛。
MIN_ANCHOR_AREA = min(ANCHOR_AREAS)
#: 最小锚边长——短边低于此值的图不可能覆盖任何锚。
MIN_ANCHOR_SIDE = min(min(w, h) for w, h in ANCHORS_V1_WH)
#: 最长锚边（704）。Stage512a 实际用到的最长边是 672，故 shape smoke 必须覆盖 704。
MAX_ANCHOR_SIDE = max(max(w, h) for w, h in ANCHORS_V1_WH)


def anchors_wh() -> list[tuple[int, int]]:
    """`[(width, height), ...]`，按 anchor_id 升序。"""
    return list(ANCHORS_V1_WH)


def anchors_hw() -> list[tuple[int, int]]:
    """`[(H, W), ...]` —— 既有 `multi_ar` / `Bucketer` / `BucketSchedule` 的桶表格式。

    ⚠️ 顺序仍是 anchor_id 顺序（不是 legacy `build_adaptive_buckets()` 的 AR 排序），
    因为 anchor_id 是审计与 plan 的身份键。
    """
    return [(h, w) for (w, h) in ANCHORS_V1_WH]


def square_anchor_id() -> int:
    return 0


def crop_loss_wh(w: int, h: int, anchor_id: int) -> float:
    """把 `w x h` 指派到 `anchor_id` 时丢弃的源像素比例。

    推导：设 s = max(W/w, H/h)。
      AR_src > AR_anchor  =>  s = H/h  =>  retention = W*h/(w*H) = AR_anchor/AR_src
      AR_src < AR_anchor  =>  s = W/w  =>  retention = H*w/(h*W) = AR_src/AR_anchor
    故 retention = min(AR_src/AR_anchor, AR_anchor/AR_src)，**只依赖 AR，与绝对尺寸无关**。
    推论：`|log AR| 最小` 与 `crop_loss 最小` 是同一个 argmin，所以把阈值作用在
    argmin 胜者上，等价于"没有任何可行锚能做到 crop_loss <= T"。
    """
    W, H = ANCHORS_V1_WH[anchor_id]
    s = max(W / w, H / h)
    retention = (W * H) / ((w * s) * (h * s))
    return 1.0 - retention


def assign_anchor(w: int, h: int, *,
                  crop_loss_max: float = CROP_LOSS_MAX,
                  area_gate: int | None = None) -> tuple[int, float] | None:
    """源尺寸 `(w, h)` -> `(anchor_id, crop_loss)`；不可服务返回 `None`。

    `area_gate`：None（默认）= 只用 `F(i) != ∅`；给整数 = 额外要求 `w*h >= area_gate`
    （传 `AREA_GATE_STAGE512A` 复现 Stage512b 审计口径）。
    """
    if not w or not h or w <= 0 or h <= 0:
        return None
    w = int(w)
    h = int(h)
    if area_gate is not None and w * h < int(area_gate):
        return None
    log_ar = math.log(w / h)
    best_key: tuple[float, float, int, int] | None = None
    best_id = -1
    for k in range(_N):
        W, H = ANCHORS_V1_WH[k]
        if w < W or h < H:          # 可行集：唯一的零上采样依据
            continue
        s = max(W / w, H / h)       # w>=W 且 h>=H  =>  s <= 1
        retention = (W * H) / ((w * s) * (h * s))
        key = (abs(_ANCHOR_LOG_ARS[k] - log_ar), -retention, -ANCHOR_AREAS[k], k)
        if best_key is None or key < best_key:
            best_key = key
            best_id = k
    if best_key is None:
        return None                 # F(i) = ∅
    crop_loss = 1.0 - (-best_key[1])
    if crop_loss > crop_loss_max:
        return None
    return best_id, crop_loss


def is_servable(w: int, h: int, *, crop_loss_max: float = CROP_LOSS_MAX,
                area_gate: int | None = None) -> bool:
    return assign_anchor(w, h, crop_loss_max=crop_loss_max, area_gate=area_gate) is not None


def assign_anchor_batch(w, h, *,
                        crop_loss_max: float = CROP_LOSS_MAX,
                        area_gate: int | None = None):
    """`assign_anchor` 的向量化版本。返回 `(anchor_id, crop_loss)` 两个等长数组，
    不可服务的行 `anchor_id = -1`、`crop_loss = nan`。

    **必须存在的理由**：既有 `vectorized_adaptive_assign()`（`phase5_build_catalog.py:316`,
    `phase3a_bucket_stats.py:329`）走的是 `(h//16, w//16)` 稠密查表 + "查不到就退标量循环"。
    13 锚下那张表 99.9% 是 -1，于是**每一行**都走 Python 标量分支 —— 全量池约 10^9 行，
    目录构建会静默地永远跑不完（不报错，只是不结束）。所以锚路径必须自带按锚循环
    （13 次 numpy 全列运算）而不是按行循环的实现。

    与标量 `assign_anchor` **逐运算同构**（同样先算 `s = max(W/w, H/h)` 再算 retention），
    由 `test_scalar_and_batch_agree_exactly` 逐位核对。
    """
    import numpy as np

    w = np.asarray(w)
    h = np.asarray(h)
    if w.shape != h.shape:
        raise AnchorGeometryError(f"w/h shape mismatch: {w.shape} vs {h.shape}")
    wf = w.astype(np.float64)
    hf = h.astype(np.float64)

    valid = (w > 0) & (h > 0)
    if area_gate is not None:
        valid &= (w.astype(np.int64) * h.astype(np.int64)) >= int(area_gate)

    n = wf.size
    best_d = np.full(n, np.inf, dtype=np.float64)
    best_tb = np.zeros(n, dtype=np.int64)
    best_id = np.full(n, -1, dtype=np.int64)
    best_ret = np.full(n, np.nan, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ar = np.log(wf / hf)

    for k in range(_N):
        W, H = ANCHORS_V1_WH[k]
        feasible = valid & (w >= W) & (h >= H)
        if not feasible.any():
            continue
        s = np.maximum(W / wf, H / hf)
        retention = (W * H) / ((wf * s) * (hf * s))
        d = np.abs(_ANCHOR_LOG_ARS[k] - log_ar)
        # 三级键 (-area, anchor_id) 压成一个整数：面积越大越靠前，同面积取小 id
        tb = -(ANCHOR_AREAS[k] * 16) + k
        take = feasible & ((d < best_d) | ((d == best_d) & (tb < best_tb)))
        best_d = np.where(take, d, best_d)
        best_tb = np.where(take, tb, best_tb)
        best_id = np.where(take, k, best_id)
        best_ret = np.where(take, retention, best_ret)

    crop_loss = 1.0 - best_ret
    reject = (best_id < 0) | (crop_loss > crop_loss_max)
    best_id = np.where(reject, -1, best_id)
    crop_loss = np.where(reject, np.nan, crop_loss)
    return best_id, crop_loss


def verify_binding(w: int, h: int, anchor_id: int, *,
                   crop_loss_max: float = CROP_LOSS_MAX,
                   context: str = "") -> None:
    """训练期**校验**离线 plan 给定的 (解码尺寸, anchor_id) 绑定，不重算指派。

    离线 plan 已经写死了每行的 anchor_id；运行期唯一的风险是**解码尺寸与 manifest 不一致**
    （EXIF 旋转是最现实的成因：manifest 记 704x352，解码出来是 352x704）。
    `resize_crop_to_bucket()` 的 `scale = max(W/w0, H/h0)` **从不拒绝上采样**，
    所以零上采样保证只能落在这个校验上。fail-closed，不静默兜底。
    """
    if not (0 <= anchor_id < _N):
        raise AnchorGeometryError(f"anchor_id {anchor_id} out of range [0,{_N}) {context}")
    W, H = ANCHORS_V1_WH[anchor_id]
    if not w or not h or w <= 0 or h <= 0:
        raise AnchorGeometryError(f"bad decoded size {w}x{h} {context}")
    if w < W or h < H:
        raise AnchorGeometryError(
            f"upsample would be required: decoded {w}x{h} cannot cover anchor "
            f"{anchor_id} ({W}x{H}) {context} -- likely EXIF rotation or a stale manifest")
    cl = crop_loss_wh(w, h, anchor_id)
    if cl > crop_loss_max + 1e-9:
        raise AnchorGeometryError(
            f"crop_loss {cl:.6f} > {crop_loss_max} for decoded {w}x{h} -> anchor "
            f"{anchor_id} ({W}x{H}) {context}")


def probs_from_mapping(mapping: dict) -> list[float]:
    """把 config 里的 `{anchor "624x416" | anchor_id: prob}` 归一化成 13 长概率向量。

    概率向量是**数据侧设计产物**（依赖池选择与配比），不在本模块冻结；
    由 plan/config 提供，`BucketSchedule` 消费。这里只做形状与归一化校验。
    """
    probs = [0.0] * _N
    name_to_id = {f"{w}x{h}": i for i, (w, h) in enumerate(ANCHORS_V1_WH)}
    for key, val in mapping.items():
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


def identity() -> dict:
    """写入 catalog / plan header 的几何身份。"""
    return {
        "bucket_mode": BUCKET_MODE,
        "anchor_set_id": ANCHOR_SET_ID,
        "n_anchors": _N,
        "anchors_wh": [list(a) for a in ANCHORS_V1_WH],
        "assignment": ("argmin over feasible set of "
                       "(|dlog_ar|, -retention, -area, anchor_id)"),
        "feasible_set": "w >= W_k and h >= H_k",
        "crop_loss_max": CROP_LOSS_MAX,
        "upsampling": "structurally impossible; verified per row at runtime",
    }
