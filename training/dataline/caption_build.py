"""caption 构建层（纯函数，无 I/O；row 字段已由上游回源/加载）。

real/synth：上游回源拿到选中 captioner 的原始文本 → finalize（florence 首句 / internvl 删残句 / 截断）。
anime：原生忠实实现 complete_v1 config 的 anime.caption 规格（按 v3 canon + naifu caption_builder 结构）：
  per-stage 路由（rich_nl/nl/nl_short/tag_medium）→ anchor 块（keep-prob 节流 + 仅 q/e 渲染 rating）
  + NL 路由（结构化-NL + refined/medium/short）或 tag 路由（meta→character→series→artist→general
  + 频率 dropout + length_cap 四桶 + 50% 分隔变体）→ 句/tag 界截断到 224-256 proxy-tok → 10% CFG dropout。

token 计数用 manifest 同款 proxy（words×1.32，误差 <2%），避免 loader 内塞 tokenizer；
模型 encode_text 仍会硬截到 296，本层预截到 ≤256 使硬截基本不触发（config _text_max_length_note）。
"""

from __future__ import annotations

import re
import statistics

from .config import AnimeCaptionCfg

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def proxy_tok(text: str | None) -> int:
    """manifest 同款 token 代理：词数 × 1.32。"""
    return int(len(text.split()) * 1.32) if text else 0


def truncate_to_token_budget(text: str, tokenizer, budget: int) -> str:
    """用【真实 tokenizer】把 caption 截到 ≤budget token（消除代理 words×1.32 低估导致的
    模型端静默 mid-token 硬截，见 review F3）。在句末/逗号/空格边界回退，不切半个词/tag。
    tokenizer=None 时跳过（退化为上游 proxy 截断）。"""
    if not text or tokenizer is None:
        return text
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return text
    head = tokenizer.decode(ids[:budget]).strip()
    best, keep_punct = -1, False
    for sep in (". ", "! ", "? "):                       # 句末边界（保留标点）
        k = head.rfind(sep)
        if k > best:
            best, keep_punct = k, True
    for sep in (", ", " "):                              # tag / 词边界
        k = head.rfind(sep)
        if k > best:
            best, keep_punct = k, False
    if best > 0:
        head = head[: best + (1 if keep_punct else 0)]
    return head.strip().rstrip(",").strip()


def truncate_sentences(text: str, hi_tok: int = 256, lo_tok: int = 224) -> str:
    """句界截断到 ≤hi_tok（proxy）。在句号边界累加，超 hi 即停；单句超长则按词硬截。"""
    if not text or proxy_tok(text) <= hi_tok:
        return text or ""
    out, ntok = [], 0
    for s in _SENT_SPLIT.split(text.strip()):
        st = proxy_tok(s)
        if out and ntok + st > hi_tok:
            break
        out.append(s)
        ntok += st
        if ntok >= hi_tok:
            break
    res = " ".join(out)
    if proxy_tok(res) > hi_tok:                       # 单句仍超长 → 按词硬截
        res = " ".join(res.split()[: int(hi_tok / 1.32)])
    return res


def truncate_tags(tag_str: str, hi_tok: int = 256) -> str:
    """tag 界截断（逗号或空格分隔）到 ≤hi_tok（proxy）。"""
    if not tag_str or proxy_tok(tag_str) <= hi_tok:
        return tag_str or ""
    sep = ", " if ", " in tag_str else " "
    parts = tag_str.split(sep)
    out, ntok = [], 0
    for p in parts:
        pt = proxy_tok(p)
        if out and ntok + pt > hi_tok:
            break
        out.append(p)
        ntok += pt
    return sep.join(out)


# --------------------------------------------------------------------------- #
# real/synth：finalize 选中 captioner 文本
# --------------------------------------------------------------------------- #
def _drop_trailing_incomplete(t: str) -> str:
    """internvl 删末尾不完整句（不以句末标点结尾则截到最后一个句末标点）。"""
    if not t or t[-1] in ".!?":
        return t
    idx = max(t.rfind("."), t.rfind("!"), t.rfind("?"))
    return t[: idx + 1] if idx > 0 else t


def build_textatlas_caption(captions, rng) -> str:
    """textatlas caption：每曝光从 i1 caption1..5 随机选 1 个【非空】（Evan 拍板 = caption augmentation，
    同图多次曝光见不同变体、增多样性防过拟合单 caption）。全空则返回 ""（caption dropout 等价）。
    选出后由 loader 的 truncate_to_token_budget 统一过真 Qwen token 截断（同 anime/real）。"""
    cands = [c.strip() for c in captions if c and c.strip()]
    if not cands:
        return ""
    return cands[rng.randrange(len(cands))]


def finalize_real_caption(raw_text: str | None, captioner: str, truncate=(224, 256)) -> str:
    """real/synth 选中 captioner 原始文本 → 最终 caption。captioner ∈ MONET 简名 /
    florence_first_sentence / i1 / original。"""
    if not raw_text:
        return ""
    t = raw_text.strip()
    if captioner == "florence_first_sentence":
        t = _SENT_SPLIT.split(t, maxsplit=1)[0]
    elif captioner == "internvl":
        t = _drop_trailing_incomplete(t)
    return truncate_sentences(t, truncate[1], truncate[0])


# --------------------------------------------------------------------------- #
# anime：原生 caption 构建
# --------------------------------------------------------------------------- #
def _weighted_choice(weights: dict[str, float], rng) -> str:
    keys = list(weights.keys())
    tot = sum(weights.values())
    r = rng.random() * tot
    acc = 0.0
    for k in keys:
        acc += weights[k]
        if r <= acc:
            return k
    return keys[-1]


def _tag_space_variant(items: list[str], rng) -> str:
    """50% 用 ", " join；50% 把 tag 内空格换 _ 后空格 join（naifu _join_tag_items）。"""
    if rng.random() < 0.5:
        return " ".join(it.replace(" ", "_") for it in items)
    return ", ".join(items)


def _freq_dropout(general: list[str], freq: dict | None, p: dict, rng) -> list[str]:
    """频率加权 tag dropout（naifu frequency_weighted_tag_dropout）：高频 general tag 更易被丢，
    保护长尾。无 freq 表则跳过。

    drop_p = clip(base_p·(f/ref)^alpha, min_p, max_p)，保 min_keep。
    ref = max(median(freq.values()), 1.0)（naifu compute_frequency_weighted_dropout_probs 的
    ref_frequency；以中位频率为基准 → 中位 tag ~base_p、高频顶 max_p、长尾压 min_p）。
    早前误用 max(freq) 作分母会把 ~全部 tag 钉到 min_p、dropout 名存实亡（见 review F-A1）。"""
    if not freq or not general:
        return general
    vals = list(freq.values())
    ref = max(float(statistics.median(vals)) if vals else 1.0, 1.0)
    base_p, alpha = p["base_p"], p["alpha"]
    min_p, max_p = p["min_p"], p["max_p"]
    min_keep = int(p["min_keep"])
    min_keep_ratio = float(p["min_keep_ratio"])
    keep = []
    for t in general:
        f = max(int(freq.get(t, 1)), 1)                # naifu：缺省/下限频率 = 1
        dp = base_p * (f / ref) ** alpha
        dp = min(max(dp, min_p), max_p)
        if rng.random() >= dp:
            keep.append(t)
    floor = max(min_keep, int(len(general) * min_keep_ratio))
    if len(keep) < floor:                              # 不足保底：补回最长尾（低频）的
        rest = [t for t in general if t not in set(keep)]
        rest.sort(key=lambda t: freq.get(t, 1))        # 低频优先补
        keep = keep + rest[: floor - len(keep)]
        keep = [t for t in general if t in set(keep)]  # 恢复原顺序
    return keep


def _apply_length_cap(general: list[str], cap_p: dict, rng) -> list[str]:
    """naifu _apply_length_cap：30% keep≤max(10,ceil(0.3N)); 10% keep≤min(6,N);
    15% keep rand[1,5]; 45% 不截。"""
    n = len(general)
    if n == 0:
        return general
    r = rng.random()
    if r < cap_p["p_cap_30"]:
        cap = max(10, -(-n * 3 // 10))
    elif r < cap_p["p_cap_30"] + cap_p["p_cap_10"]:
        cap = min(6, n)
    elif r < cap_p["p_cap_30"] + cap_p["p_cap_10"] + cap_p["p_cap_random_1_5"]:
        cap = min(1 + int(rng.random() * 5), n)
    else:
        return general
    return general[:cap]


def _clean(t: str) -> str:
    return t.replace("_", " ").strip()


def _anchor_parts(f: dict, ac: AnimeCaptionCfg, rng) -> list[str]:
    """anchor 块：artist / character / series(copyright) / rating(仅 q/e) / year，按 keep-prob 节流。"""
    kp = ac.anchor_keep_probs
    parts: list[str] = []
    rb = f.get("rating_bucket")
    if rb in ("questionable", "explicit") and rng.random() < kp.get("rating", 0.6):
        parts.append(f"rating: {rb}")
    art = f.get("tags_artist") or []
    if art and rng.random() < kp.get("artist", 1.0):
        parts += [_clean(t) for t in art[:3]]
    chars = f.get("tags_character") or []
    if chars and rng.random() < kp.get("character_label", 0.5):
        parts += [_clean(t) for t in chars[:5]]
    cop = f.get("tags_copyright") or []
    if cop and rng.random() < kp.get("series", 0.4):
        parts += [_clean(t) for t in cop[:2]]
    yr = f.get("year")
    if yr and rng.random() < kp.get("year_period", 0.3):
        parts.append(f"year {int(yr)}")
    return parts


def _build_tag(f: dict, ac: AnimeCaptionCfg, rng, general_freq: dict | None = None) -> str:
    """tag 路由：meta→character→series→artist→general（naifu 顺序）+ freq dropout + length_cap。"""
    p = ac.tag_frequency_dropout
    general = list(f.get("tags_general") or [])
    use_freq = general_freq is not None and rng.random() < 0.20      # naifu TAG_DROPOUT_COMMON_PROB
    if use_freq:
        general = _freq_dropout(general, general_freq, p, rng)
    general = _apply_length_cap(general, ac.length_cap, rng)
    rng.shuffle(general)
    sections: list[str] = []
    meta = f.get("tags_meta") or []
    if meta:
        sections += [_clean(t) for t in meta[:2]]
    sections += _anchor_parts(f, ac, rng)               # character/series/artist/rating/year（带节流）
    sections += [_clean(t) for t in general]
    # 去重保序
    seen, out = set(), []
    for s in sections:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return _tag_space_variant(out, rng)


def _build_rich_nl(f: dict, ac: AnimeCaptionCfg, rng) -> str:
    """rich_nl：anchor + 结构化-NL（Style/image_type/Camera/Foreground/Background）+ 主 NL。"""
    head = _anchor_parts(f, ac, rng)
    rng.shuffle(head)
    head_s = (", ".join(head) + ".") if head else ""
    visual = []
    if f.get("style"):
        visual.append(f"Style: {f['style']}")
    if f.get("image_type"):
        visual.append(f["image_type"])
    if f.get("camera_and_lens_effects"):
        visual.append(f"Camera and lens: {f['camera_and_lens_effects']}")
    nl = f.get("refined_description") or f.get("medium_description") or f.get("short_description")  # M9 修复(2026-06-29): refined 优先(取最长 NL); 原 medium 优先是 bug, rich_nl 本就是 dense 路由
    if nl:
        visual.append(nl)
    if f.get("foreground"):
        visual.append(f"Foreground: {f['foreground']}")
    if f.get("background"):
        visual.append(f"Background: {f['background']}")
    body = " ".join(x.strip() for x in visual if x)
    return (head_s + " " + body).strip() if head_s else body


def _build_nl(f: dict, ac: AnimeCaptionCfg, rng, short_first: bool) -> str:
    """nl / nl_short：anchor + 单条主 NL。short_first 优先 short。"""
    head = _anchor_parts(f, ac, rng)
    rng.shuffle(head)
    head_s = (", ".join(head) + ".") if head else ""
    order = ("short_description", "medium_description", "refined_description") if short_first \
        else ("refined_description", "medium_description", "short_description")
    nl = ""
    for k in order:
        if f.get(k):
            nl = f[k]
            break
    return (head_s + " " + nl).strip() if head_s else nl


def build_anime_caption(f: dict, stage: str, ac: AnimeCaptionCfg, rng,
                        general_freq: dict | None = None) -> str:
    """anime 行 → 最终 caption。f 含源 caption 字段；rng = 行级确定性 random.Random。"""
    tt = ac.truncate_tok_for(stage)                          # REDESIGN per-stage proxy 上限（512a/512b=[410,450]）；旧/256 回退标量
    has_nl = bool(f.get("refined_description") or f.get("medium_description") or f.get("short_description"))
    # ⚠️ CFG dropout 不在此处做：改由模型单点 _apply_cfg_dropout 统一负责（避免与模型的 10%
    #   双重 dropout）。dataset 给【tag-only anime 行】带 cfg_keep 标志 → 模型对其用 prob 0
    #   实现 config『tag-only 永不 drop、NL 行 10%』。见 review F4 / complete_dataset.loader。
    route = _weighted_choice(ac.route_weights(stage, has_nl, caption_policy=None), rng)
    if route == "tag_medium" or not has_nl:
        cap = _build_tag(f, ac, rng, general_freq)
        return truncate_tags(cap, tt[1])
    if route == "rich_nl":
        cap = _build_rich_nl(f, ac, rng)
    elif route == "nl_short":
        cap = _build_nl(f, ac, rng, short_first=True)
    else:
        cap = _build_nl(f, ac, rng, short_first=False)
    cap = truncate_sentences(cap, tt[1], tt[0])
    return cap or _build_tag(f, ac, rng, general_freq)  # NL 路由空兜底到 tag
