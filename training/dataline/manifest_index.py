"""manifest 索引层：枚举 shard、全局源/簇统计、加载并预计算每行采样字段。

四域：
  real/synth（gpic / monet / monet_synth）—— daily_v1 manifest（38 列）是选样权威 +
    band 合格性来源；图/caption 回源到 src_shard（见 image_fetch / caption_build）。
  anime —— 源 parquet（17 shard，139 列）自带全部采样+caption+取图字段，直接驱动。

loader rules（数据交付文档 §6.2，逐条对应）：
  #1 holdout==0 全局排除；#2/#3/#4 band 合格性以 tok>0 落档为准（非 has_tX）、tok<=0 不可用；
  #5 GPIC@512/1024=0（由 sampler 的 domain_mix 处理，此处只标 elig）；#6 aes 缺失（不用 aes）；
  #7 source/cluster 因子用全局池统计一次；#8 图从 .tar/文件读（image_fetch 负责）。
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

from .native_safety import PARQUET_NATIVE_LOCK, configure_pyarrow_single_thread

configure_pyarrow_single_thread()

from .config import BANDS, SamplingConfig

# MONET captioner 简名 → manifest tok_* 列后缀（band_to_captioner_priority 用简名）
MONET_CAPTIONERS = ("gemini", "sharegpt4v", "internvl", "florence", "original")
_TOK_COL = {c: f"tok_{c}" for c in MONET_CAPTIONERS}
# 整数编码：0=不服务该档；1..5=对应 captioner（含 GPIC i1 / synth original）
CAPTIONER_CODE = {None: 0, "gemini": 1, "sharegpt4v": 2, "internvl": 3,
                  "florence": 4, "original": 5, "i1": 6, "florence_first_sentence": 7}
CODE_TO_CAPTIONER = {v: k for k, v in CAPTIONER_CODE.items()}

# --------------------------------------------------------------------------- #
# anime composition_clean 真值谓词常量（忠实 naifu tag_utils.py + sampling_plan.md Axis3）
#   clean = 有 simple-bg tag AND 无 watermark/sig/logo/username tag AND
#           YOLO added_text_area_nms < 0.005 AND quality_z >= 0.4
# 早前用 risk_bucket=='plain_illustration' 近似（命中 ~80%、精确率 25%），源字段齐备 → 改精确。
# --------------------------------------------------------------------------- #
CLEAN_BG_TAGS = frozenset({
    "simple_background", "white_background", "solid_background",
    "gradient_background", "transparent_background",
})
_PLATFORM_USERNAME_TAGS = frozenset({
    "twitter_username", "patreon_username", "weibo_username", "instagram_username",
    "deviantart_username", "pixiv_username", "fanbox_username", "facebook_username",
    "tumblr_username", "gumroad_username", "bluesky_username", "subscribestar_username",
    "artstation_username", "lofter_username", "skeb_username", "mihoyo_username",
    "ko-fi_username", "newgrounds_username",
})
_WATERMARK_SIGNATURE_TAGS = frozenset({
    "watermark", "sample_watermark", "weibo_watermark", "watermark_grid",
    "miyoushe_watermark", "official_watermark", "character_watermark", "alpha_signature",
    "signature", "artist_name", "artist_signature", "character_signature", "dated",
    "copyright_notice", "copyright_name", "too_many_watermarks",
})
_LOGO_TAGS = frozenset({
    "logo", "artist_logo", "patreon_logo", "twitter_logo", "copyright_logo",
    "instagram_logo", "facebook_logo", "deviantart_logo", "pixiv_logo",
    "twitter_x_logo", "youtube_logo",
})
VISIBLE_MARK_TAGS = _PLATFORM_USERNAME_TAGS | _WATERMARK_SIGNATURE_TAGS | _LOGO_TAGS  # 45 tag
CLEAN_QUALITY_Z_MIN = 0.4
CLEAN_YOLO_AREA_MAX = 0.005
YOLO_ADDED_TEXT_AREA_COL = "skytnt_yolo_textblock_detection_10_added_text_area_nms"


def composition_clean_mask(tags_general_list, yolo_area: np.ndarray, qz: np.ndarray) -> np.ndarray:
    """向量化 naifu 真值谓词 → bool 掩码（per-row set 交是唯一逐行项；tags_general 变长）。
    yolo_area 缺失（NaN/None→已转 +inf）则该项判不干净（无法确认 near-zero added text）。"""
    tg = [set(x) if x else set() for x in tags_general_list]
    has_bg = np.array([bool(s & CLEAN_BG_TAGS) for s in tg], dtype=bool)
    no_mark = np.array([not (s & VISIBLE_MARK_TAGS) for s in tg], dtype=bool)
    low_text = yolo_area < CLEAN_YOLO_AREA_MAX                  # +inf < x → False（缺失=不干净）
    qz_ok = qz >= CLEAN_QUALITY_Z_MIN
    return has_bg & no_mark & low_text & qz_ok


def _shard_glob(domain_root: str, domain: str) -> list[str]:
    """real_synth_glob 形如 .../daily_v1/{gpic,monet,monet_synth}；展开到具体域目录。"""
    base = domain_root.split("{")[0].rstrip("/")          # .../daily_v1
    return sorted(glob.glob(os.path.join(base, domain, "*.parquet")))


def list_shards(cfg: SamplingConfig) -> dict[str, list[str]]:
    """每域的 manifest/源 shard 路径列表。textatlas 列出 sidecar manifest 路径（仅纳入子集）。"""
    roots = getattr(cfg, "manifest_real_synth_roots", None)
    if roots:                                              # REDESIGN: per-domain 显式根（real→rich_v1, synth→synthetic_v1）
        out = {
            "gpic": sorted(glob.glob(os.path.join(roots["gpic"], "*.parquet"))),
            "monet_real": sorted(glob.glob(os.path.join(roots["monet_real"], "*.parquet"))),
            "synthetic": sorted(glob.glob(os.path.join(roots["synthetic"], "*.parquet"))),
            "anime": sorted(glob.glob(os.path.join(cfg.anime_caption_source, "*.parquet"))),
        }
    else:                                                  # 旧行为：单 glob 同 base 展开三域子目录
        out = {
            "gpic": _shard_glob(cfg.manifest_real_synth_glob, "gpic"),
            "monet_real": _shard_glob(cfg.manifest_real_synth_glob, "monet"),
            "synthetic": _shard_glob(cfg.manifest_real_synth_glob, "monet_synth"),
            "anime": sorted(glob.glob(os.path.join(cfg.anime_caption_source, "*.parquet"))),
        }
    if cfg.manifest_textatlas:                              # 第 5 域（可选；未配置则不启用）
        subs = cfg.textatlas_subsets or ("*",)            # 默认全子集；Evan: 9 子集（丢 CoverBook）
        ta = []
        for sub in subs:
            ta += glob.glob(os.path.join(cfg.manifest_textatlas, sub, "*.parquet"))
        out["textatlas"] = sorted(ta)
    for d, s in out.items():
        assert s, f"域 {d} 没找到 shard（路径 {cfg.manifest_real_synth_glob}/{d} 或 {cfg.anime_caption_source} 或 {cfg.manifest_textatlas}）"
    return out


# --------------------------------------------------------------------------- #
# textatlas sidecar shard 加载（uid/elig/caption；图在源 parquet 内联，由 iter 按 image_path join）
# --------------------------------------------------------------------------- #
_TEXTATLAS_COLS = ["image_path", "uid", "width", "height",
                   "elig_512", "elig_1024",
                   "caption1", "caption2", "caption3", "caption4", "caption5"]


def textatlas_subset_shard(sidecar_path: str) -> tuple[str, str]:
    """sidecar 路径 → (subset, shardname)。sidecar 布局 = <root>/<Subset>/<shard>.parquet，
    源图 parquet 同 subset/同 shardname（builder 保证 1:1 + 行序对齐）。"""
    shardname = os.path.basename(sidecar_path)
    subset = os.path.basename(os.path.dirname(sidecar_path))
    return subset, shardname


def load_textatlas_shard(sidecar_path: str, stage: str) -> dict:
    """加载一个 textatlas sidecar shard，返回结构化记录（注入域：base_w=1 均匀、served_mask=0 不分档）。

    返回 dict（各 key 等长 array/list）：
      image_path, uid, elig_256/512/1024(bool), base_w(ones), served_mask(zeros uint8),
      caption1..5(list[str])，domain="textatlas"。
    elig_256 由 sidecar 的 w/h 现算（builder 只产 elig_512/1024）；elig_512/1024 直读列。
    """
    with PARQUET_NATIVE_LOCK:
        t = pq.read_table(sidecar_path, columns=_TEXTATLAS_COLS, use_threads=False)
    n = t.num_rows
    w = np.array(t.column("width").to_pylist(), dtype=np.int64)
    h = np.array(t.column("height").to_pylist(), dtype=np.int64)
    short_dim = np.minimum(w, h)
    out = {
        "image_path": np.array(t.column("image_path").to_pylist(), dtype=object),
        "uid": np.array(t.column("uid").to_pylist(), dtype=object),
        "elig_256": short_dim >= 256,                       # 现算（builder 未产 elig_256）
        "elig_512": np.array(t.column("elig_512").to_pylist(), dtype=bool),
        "elig_1024": np.array(t.column("elig_1024").to_pylist(), dtype=bool),
        "base_w": np.ones(n, dtype=np.float64),             # 注入域：池内均匀
        "served_mask": np.zeros(n, dtype=np.uint8),         # 不参与 band-first
        "domain": "textatlas",
    }
    for c in ("caption1", "caption2", "caption3", "caption4", "caption5"):
        out[c] = t.column(c).to_pylist()
    return out


def shard_subset_for_rank(shards: list[str], rank: int, world: int,
                          force_row_level: bool = False) -> list[str]:
    """按 rank 切 shard（每 rank 拿 idx%world==rank 的 shard）——数据分片 + 降每 rank 内存。

    细粒度回退（2026-06-23，多机 world>17 必需）：当某域 shard 数 < world 时，文件级分片会让
    idx≥n 的 rank 拿到 **0 个**该域 shard → 该域被 sampler renorm 丢弃、全局 domain_mix 掉
    （anime 仅 17 shard：world=32 时 rank17-31 无 anime，anime 占比 0.30→0.16）。此时改为
    **每个 rank 都拿全部 shard**，靠已按 rank 加偏的 producer RNG（seed + rank*9973：随机
    row-group + 随机 offset + shuffle）在【行级】错开 → 每卡都有该域、域配比正确。跨 rank 行
    重叠在数百万行流式 + 带放回抽样下可忽略；尤其 latent-cache 模式 skip_image 后该域只读
    parquet 行（无取图开销）。shard 数 ≥ world 的域（gpic/monet/synth 各 64）仍走原文件级分片。

    force_row_level（2026-06-26，修审计 M5/M2b）：对【shard 少且内部按质量/NL 强有序、又按 nrows
    随机 row-group 抽样取图的域】（anime：仅 17 个 fat shard，has_nl 跨 shard 52.8%–95.2%、NSFW/risk
    强偏）**恒走行级**——文件级会让每 rank 只见 1 个 shard 的偏斜子分布（per-rank caption/NSFW 偏斜）。
    anime 按 nrows 抽样 + 取图按需，行级回退几乎零额外开销。full-read 域（gpic/monet/synth 整 shard 读）
    **不要**开此项：高 world 下行级会让每 rank 重复读全部 shard → 吞吐回退。
    """
    if world <= 1 or force_row_level:
        return list(shards)
    if len(shards) < world:
        return list(shards)
    return [s for i, s in enumerate(shards) if i % world == rank]


# --------------------------------------------------------------------------- #
# 全局源/簇统计（loader rule #7：整池统计一次）
# --------------------------------------------------------------------------- #
def _load_stats_cache(cache_path: str) -> dict:
    with open(cache_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = {"source_counts": d["source_counts"],
           "cluster_sizes": {int(k): v for k, v in d["cluster_sizes"].items()}}
    if "_n_shards" in d:                                  # REDESIGN: 指纹（防 stale cache 喂错池）
        out["_n_shards"] = d["_n_shards"]
    return out


def compute_real_global_stats(monet_shards: list[str], cache_path: str | None = None,
                              compute: bool = True) -> dict:
    """over MONET-real【全池】统计 sub_source 计数 + cluster_id size（rule #7：整池统计一次；
    GPIC/synth 另处理）。`monet_shards` 必须是【未按 rank 切片】的全量列表，否则 base_w 跨 rank 不一致。

    多 rank 协调（见 complete_dataset）：仅 rank0 以 compute=True 算并【原子写】缓存（temp+os.replace，
    防并发截断），其余 rank 在 barrier 后以 compute=False 读缓存。单进程(world=1)直接 compute=True。"""
    if cache_path and os.path.exists(cache_path):
        cached = _load_stats_cache(cache_path)
        if cached.pop("_n_shards", None) in (None, len(monet_shards)):   # 指纹匹配(或旧cache无指纹)→用; 不符(如 daily 11M cache 喂 rich_v1 40M)→落下重算覆盖
            return cached
        print(f"[dataline] ⚠️ stats cache 指纹不符(shard 数变了)，重算并覆盖: {cache_path}", flush=True)
    if not compute:
        raise FileNotFoundError(
            f"stats cache 不存在且 compute=False：{cache_path}（多 rank 下应由 rank0 先算/写、barrier 后非 rank0 再读）")
    from collections import Counter
    src_ctr, clu_ctr = Counter(), Counter()
    for sp in monet_shards:
        with PARQUET_NATIVE_LOCK:
            t = pq.read_table(sp, columns=["sub_source", "cluster_id"], use_threads=False)
        for s in t.column("sub_source").to_pylist():
            src_ctr[s] += 1
        for c in t.column("cluster_id").to_pylist():
            clu_ctr[int(c)] += 1
    stats = {"source_counts": dict(src_ctr), "cluster_sizes": dict(clu_ctr)}
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        payload = {"source_counts": stats["source_counts"],
                   "cluster_sizes": {str(k): v for k, v in stats["cluster_sizes"].items()},
                   "_n_shards": len(monet_shards)}        # REDESIGN: 指纹
        tmp = f"{cache_path}.tmp.{os.getpid()}"                # 原子写：先 temp 再 os.replace（同目录）
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, cache_path)
    return stats


# --------------------------------------------------------------------------- #
# real/synth manifest shard 加载（+ band 合格性 + base 权重）
# --------------------------------------------------------------------------- #
_REAL_COLS = ["uid", "dataset", "sub_source", "src_root", "src_shard", "src_key",
              "width", "height", "cluster_id", "caption_type", "caption_tier",
              "elig_256", "elig_512", "elig_1024", "holdout"] + list(_TOK_COL.values()) + ["tok_i1"]


def _band_captioner_vectorized(tok_arrays: dict, tok_i1, sub_source, caption_tier,
                               is_gpic: bool, cfg: SamplingConfig) -> np.ndarray:
    """向量化逐行逐档 captioner 选择（CAPTIONER_CODE）。loader rule #2/#3/#4：tok>0 且落档。

    GPIC：单 i1，只服务 caption_tier 那一档（且 tok_i1 落该档）。
    MONET/synth：每档按 band_to_captioner_priority 顺序填（先优先级，剩余=fallback "任意落档"），
      在该档仍为 0 处填入；megalith T2 用 florence 首句（florence 落 T2 时覆盖）。
    返回 int8[N,4]（列 = T1..T4）。
    """
    n = len(tok_i1)
    band_cap = np.zeros((n, 4), dtype=np.int8)
    if is_gpic:
        for bi, band in enumerate(BANDS):
            lo, hi = cfg.tier_bounds_tok[band]
            m = (caption_tier == bi + 1) & (tok_i1 > 0) & (tok_i1 >= lo) & (tok_i1 <= hi)
            band_cap[m, bi] = CAPTIONER_CODE["i1"]
        return band_cap

    def serves(c, lo, hi):
        t = tok_arrays[c]
        return (t > 0) & (t >= lo) & (t <= hi)

    is_meg = (sub_source == "megalith10m")
    for bi, band in enumerate(BANDS):
        lo, hi = cfg.tier_bounds_tok[band]
        prio = list(cfg.band_to_captioner_priority.get(band, []))
        order = prio + [c for c in MONET_CAPTIONERS if c not in prio]   # 优先级 then fallback(剩余)
        for name in order:
            if name not in tok_arrays:                                 # 只处理真实 captioner 列
                continue
            m = serves(name, lo, hi) & (band_cap[:, bi] == 0)
            band_cap[m, bi] = CAPTIONER_CODE[name]
        if band == "T2":                                               # megalith 特例：florence 首句覆盖
            m = is_meg & serves("florence", lo, hi)
            band_cap[m, bi] = CAPTIONER_CODE["florence_first_sentence"]
    return band_cap


def load_real_synth_shard(path: str, domain: str, cfg: SamplingConfig, stats: dict) -> dict:
    """加载一个 daily_v1 real/synth manifest shard，返回结构化 numpy 记录 + 派生字段。

    返回 dict（各 key 是等长 array / list）：
      uid, src_root, src_shard, src_key, sub_source, caption_type, caption_tier(int),
      elig_256/512/1024(bool), base_w(float), band_captioner(int[N,4]),
      served_mask(uint8 bit b set if 服务档 b), tok(dict 已并入 band_captioner 不再单存)
    已过滤 holdout==0。
    """
    with PARQUET_NATIVE_LOCK:
        avail = set(pq.read_schema(path).names)
    read_cols = [c for c in _REAL_COLS if c in avail]                   # REDESIGN: rich_v1/synthetic_v1 无 elig_* 列 → 不读, 下面从 w/h 派生
    with PARQUET_NATIVE_LOCK:
        t = pq.read_table(path, columns=read_cols, use_threads=False)
    n = t.num_rows
    cols = {c: t.column(c) for c in read_cols}
    keep = np.array(cols["holdout"].to_pylist(), dtype=np.int8) == 0     # rule #1
    # elig: 优先读列(daily_v1, elig_256 恒 1); 缺则从 width/height 派生(REDESIGN rich_v1/synthetic_v1 用 res_bucket 不带 elig 列)
    if {"elig_256", "elig_512", "elig_1024"}.issubset(avail):
        e256 = np.array(cols["elig_256"].to_pylist(), dtype=np.int8) == 1
        e512 = np.array(cols["elig_512"].to_pylist(), dtype=np.int8) == 1
        e1024 = np.array(cols["elig_1024"].to_pylist(), dtype=np.int8) == 1
    else:
        _short = np.minimum(np.array(cols["width"].to_pylist(), dtype=np.float64),
                            np.array(cols["height"].to_pylist(), dtype=np.float64))
        e256, e512, e1024 = _short >= 256, _short >= 512, _short >= 1024

    sub_source = np.array(cols["sub_source"].to_pylist(), dtype=object)
    cluster_id = np.array(cols["cluster_id"].to_pylist(), dtype=np.int64)
    caption_tier = np.array(cols["caption_tier"].to_pylist(), dtype=np.int16)
    is_gpic = (domain == "gpic")

    # base_w = source_factor × cluster_factor（aes off, rule #7 全局统计）
    a, b = cfg.source_alpha, cfg.cluster_beta
    src_counts = stats["source_counts"]
    clu_sizes = stats["cluster_sizes"]
    explicit = getattr(cfg, "monet_subsource_explicit_probs", None)
    if is_gpic:
        src_f = np.ones(n, dtype=np.float64)              # GPIC source_factor=1（reference 实现）
    elif explicit and domain == "monet_real":             # REDESIGN: 显式源比例表【仅 MONET-real】(每源总质量∝probs[s], 行内仍由 cluster_beta 拉平; 不在表的源(如 da4k)→0 丢弃)
        src_f = np.array([explicit.get(s, 0.0) / max(int(src_counts.get(s, 1)), 1) for s in sub_source], dtype=np.float64)
    else:                                                  # synthetic（及无显式表时的 MONET）：count^-a 软拉平；synthetic sub_source 不在 src_counts→count=1→src_f=1 均匀
        src_f = np.array([src_counts.get(s, 1) ** (-a) for s in sub_source], dtype=np.float64)
    clu_f = np.array([1.0 if c == -1 else clu_sizes.get(int(c), 1) ** (-b) for c in cluster_id],
                     dtype=np.float64)
    base_w = src_f * clu_f

    # 逐行逐档 captioner 选择（向量化）
    tok_arrays = {c: np.array(cols[_TOK_COL[c]].to_pylist(), dtype=np.int32) for c in MONET_CAPTIONERS}
    tok_i1 = np.array(cols["tok_i1"].to_pylist(), dtype=np.int32)
    band_cap = _band_captioner_vectorized(tok_arrays, tok_i1, sub_source, caption_tier, is_gpic, cfg)
    served_mask = np.zeros(n, dtype=np.uint8)
    for bi in range(4):
        served_mask |= ((band_cap[:, bi] > 0).astype(np.uint8) << bi)

    sel = keep
    out = {
        "uid": np.array(cols["uid"].to_pylist(), dtype=object)[sel],
        "src_root": np.array(cols["src_root"].to_pylist(), dtype=object)[sel],
        "src_shard": np.array(cols["src_shard"].to_pylist(), dtype=object)[sel],
        "src_key": np.array(cols["src_key"].to_pylist(), dtype=object)[sel],
        "sub_source": sub_source[sel],
        "caption_type": np.array(cols["caption_type"].to_pylist(), dtype=object)[sel],
        "caption_tier": caption_tier[sel],
        "elig_256": e256[sel],
        "elig_512": e512[sel],
        "elig_1024": e1024[sel],
        "base_w": base_w[sel],
        "band_captioner": band_cap[sel],
        "served_mask": served_mask[sel],
        "domain": domain,
    }
    return out


# --------------------------------------------------------------------------- #
# anime 源 shard 加载（自带全部字段 + 计算 elig/priority/has_nl）
# --------------------------------------------------------------------------- #
_ANIME_COLS = ["key", "uuid", "s3key", "dataset", "width", "height",
               "rating_bucket", "risk_bucket", "cluster_clipl14_level2",
               "ct_weight", "ct_lanes", "quality_z", "hps_v3_score_qwen3vl",
               "compound_aes_score",
               "refined_description", "medium_description", "short_description",
               "tags_general", "tags_artist", "tags_character", "tags_copyright", "tags_meta",
               "foreground", "background", "style", "image_type", "camera_and_lens_effects",
               "year", "themes", YOLO_ADDED_TEXT_AREA_COL]


def anime_priority(ct_weight: float, lanes, risk_bucket: str, quality_z: float,
                   hps: float, cfg: SamplingConfig,
                   tags_general=None, yolo_added_text_area: float | None = None) -> float:
    """anime 行优先级标量版（参考/单测用；生产走 load_anime_shard 向量化路径）。

    priority = clip(ct_weight + quality_bonus + Σlane_bonus + composition_clean_bonus, 1, max_pri)
    再对 lane_caps risk_bucket ×0.3。composition_clean 用 naifu 真值谓词：simple-bg tag +
    无 watermark/sig/logo/username tag + YOLO added_text_area<0.005 + quality_z>=0.4
    （需传 tags_general + yolo_added_text_area；缺省则该 bonus 不加）。
    """
    s = cfg.anime_sampling
    p = float(ct_weight if ct_weight is not None else 1.0)
    if quality_z is not None and hps is not None and quality_z >= s.quality_bonus_qz_min \
            and hps >= s.quality_bonus_hps_min:
        p += s.quality_bonus_value
    p += s.lane_bonus_for(lanes)
    if tags_general is not None:
        tg = set(tags_general)
        area = yolo_added_text_area if yolo_added_text_area is not None else float("inf")
        if (tg & CLEAN_BG_TAGS) and not (tg & VISIBLE_MARK_TAGS) \
                and area < CLEAN_YOLO_AREA_MAX and quality_z is not None \
                and quality_z >= CLEAN_QUALITY_Z_MIN:
            p += s.composition_clean_bonus
    p = min(max(p, 1.0), s.max_priority)
    if risk_bucket in s.lane_cap_buckets:                 # lane caps：软降权 ×0.3
        p *= s.lane_cap_multiplier
    return p


def load_anime_shard(path: str, cfg: SamplingConfig, with_caption: bool = True,
                     nrows: int | None = None, rng=None) -> dict:
    """加载一个 anime 源 shard，返回结构化记录 + 计算 elig/priority/has_nl。
    with_caption=False 时只读采样字段（dry-run 更快）。

    nrows 限制读取行数（避免整 shard 1M 行 ~279s 阻塞）。【取样策略】：
      - rng 提供 + 多 row-group：随机选一个 row-group、再在组内随机 offset 取 nrows 连续行
        → 去除『只读 shard 头部』偏置（源 shard 内部按 qz/cluster 有序，头部不代表全体；
          多次访问随机 row-group 逐渐覆盖全 shard，且单次样本无系统偏置）。
      - 无 rng（确定性/单测）：读头部 nrows（legacy）。"""
    cols_needed = _ANIME_COLS if with_caption else \
        ["key", "uuid", "s3key", "width", "height", "rating_bucket", "risk_bucket",
         "cluster_clipl14_level2", "ct_weight", "ct_lanes", "quality_z", "hps_v3_score_qwen3vl",
         "tags_general", YOLO_ADDED_TEXT_AREA_COL]    # composition_clean 真值谓词需要
    if nrows:
        import pyarrow as pa
        with PARQUET_NATIVE_LOCK:
            pf = pq.ParquetFile(path)
            nrg = pf.num_row_groups
        if rng is not None and nrg > 1:
            rg = rng.randrange(nrg)                          # 随机 row-group（去头部偏置）
            with PARQUET_NATIVE_LOCK:
                t = pf.read_row_group(rg, columns=cols_needed, use_threads=False)
            if t.num_rows > nrows:                           # 组内随机 offset 取 nrows（组内也可能有序）
                start = rng.randrange(0, t.num_rows - nrows + 1)
                t = t.slice(start, nrows)
        else:                                                # legacy：头部 nrows（确定性/单 row-group）
            batches, got = [], 0
            with PARQUET_NATIVE_LOCK:
                for b in pf.iter_batches(batch_size=min(nrows, 65536), columns=cols_needed, use_threads=False):
                    batches.append(b)
                    got += b.num_rows
                    if got >= nrows:
                        break
            t = pa.Table.from_batches(batches).slice(0, nrows)
    else:
        with PARQUET_NATIVE_LOCK:
            t = pq.read_table(path, columns=cols_needed, use_threads=False)
    n = t.num_rows
    w = np.array(t.column("width").to_pylist(), dtype=np.int64)
    h = np.array(t.column("height").to_pylist(), dtype=np.int64)
    short_dim = np.minimum(w, h)
    ct_weight = np.array(t.column("ct_weight").to_pylist(), dtype=np.float64)
    qz = np.array([x if x is not None else 0.0 for x in t.column("quality_z").to_pylist()], dtype=np.float64)
    hps = np.array([x if x is not None else 0.0 for x in t.column("hps_v3_score_qwen3vl").to_pylist()], dtype=np.float64)
    lanes = [list(x) if x is not None else [] for x in t.column("ct_lanes").to_pylist()]
    risk = np.array(t.column("risk_bucket").to_pylist(), dtype=object)
    # composition_clean 真值谓词所需：tags_general（set 交）+ YOLO added_text_area（缺失→+inf=不干净）
    tags_general_raw = t.column("tags_general").to_pylist()
    yolo_area = np.array([x if x is not None else np.inf
                          for x in t.column(YOLO_ADDED_TEXT_AREA_COL).to_pylist()], dtype=np.float64)

    # 向量化 priority（lane_bonus / composition_clean 的 set 交是逐行项：ct_lanes / tags_general 变长）
    s = cfg.anime_sampling
    prio = ct_weight.copy()
    prio += np.where((qz >= s.quality_bonus_qz_min) & (hps >= s.quality_bonus_hps_min),
                     s.quality_bonus_value, 0.0)
    prio += np.array([s.lane_bonus_for(l) for l in lanes], dtype=np.float64)
    clean = composition_clean_mask(tags_general_raw, yolo_area, qz)   # naifu 真值谓词（替代 risk_bucket 近似）
    prio += np.where(clean, s.composition_clean_bonus, 0.0)
    prio = np.clip(prio, 1.0, s.max_priority)
    capmask = np.isin(risk, np.array(s.lane_cap_buckets, dtype=object))
    prio = np.where(capmask, prio * s.lane_cap_multiplier, prio)
    out = {
        "key": np.array(t.column("key").to_pylist(), dtype=object),
        "uuid": np.array(t.column("uuid").to_pylist(), dtype=object),
        "s3key": np.array(t.column("s3key").to_pylist(), dtype=object),
        "elig_256": np.ones(n, dtype=bool),               # anime 256 全 1
        "elig_512": short_dim >= 512,
        "elig_1024": short_dim >= 1024,
        "priority": prio,
        "risk_bucket": risk,
        "cluster": np.array(t.column("cluster_clipl14_level2").to_pylist(), dtype=object),  # F2a 监控
        "domain": "anime",
    }
    if with_caption:
        for c in ["rating_bucket", "refined_description", "medium_description", "short_description",
                  "tags_general", "tags_artist", "tags_character", "tags_copyright", "tags_meta",
                  "foreground", "background", "style", "image_type", "camera_and_lens_effects",
                  "year", "themes"]:
            out[c] = t.column(c).to_pylist()
    return out
