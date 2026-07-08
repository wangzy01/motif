"""complete_v1 采样 config 解析层（权威源：data_analysis/configs/t5_sampling_complete_v1.json）。

把 JSON 解析成 typed dataclass，供 manifest 索引 / sampler / caption builder 消费。
只做"读 + 校验 + 便捷访问"，不掺业务逻辑。所有数值均来自 config，不在此硬编码。

设计依据见 doc/mot/AAA数据交付文档…v2 §5/§11.5 + complete_v1 config 本体。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

STAGES = ("256", "512", "1024")
REAL_SYNTH_DOMAINS = ("monet_real", "gpic", "synthetic")
# textatlas（第 5 个域，2026-06-27）：i1 开源长 caption 文字渲染数据，注入域（像 synthetic）。
ALL_DOMAINS = ("anime", "monet_real", "gpic", "synthetic", "textatlas")
BANDS = ("T1", "T2", "T3", "T4")

# anime caption 路由键（naifu 路由名；语义见 config route_mapping_to_naifu）
ROUTES = ("rich_nl", "nl", "nl_short", "tag_medium")


@dataclass(frozen=True)
class AnimeCaptionCfg:
    """anime 域 caption 规格（per-stage 路由 + dropout/length_cap/anchor 等）。"""
    stage_route_weights_has_nl: dict[str, dict[str, float]]  # stage -> {route: weight}
    route_weights_no_nl: dict[str, float]                    # {route: weight}（tag-only 行）
    policy_tables: dict[str, dict[str, float]]               # caption_policy -> {route: weight}
    tag_frequency_dropout: dict                              # base_p/alpha/min_p/max_p/min_keep/min_keep_ratio
    length_cap: dict                                         # p_cap_30/p_cap_10/p_cap_random_1_5/p_no_cap
    anchor_keep_probs: dict                                  # quality/year_period/series/rating/count/raw_meta/artist/character_label
    truncate_tok: tuple[int, int]                            # [300, 340]（proxy-tok 构建上限；真 token 由 budget=343 backstop 精切）
    model_text_max_length: int                              # 384（budget = 此 − template_reserve41 = 343）
    dropout: float                                          # 0.10
    # ---- REDESIGN per-stage token（gated；None=回退标量, 旧 JSON 无此字段照常）----
    model_text_max_length_per_stage: dict | None = None      # {stage:int}（256=384, 512a/512b=512）→ 驱动 caption_budget
    caption_truncate_tok_per_stage: dict | None = None       # {stage:[lo,hi]}（256=[300,340], 512a/512b=[410,450]）→ proxy 构建上限
    require_nl_stages: tuple = ()                             # REDESIGN gated：列出的 stage 丢弃无-NL(tag-only) anime 行→该阶段零 tag；空=旧行为字节级零变化（256=["256"] 实现"256 不用 tag"）

    def mtml_for(self, stage: str) -> int:
        """per-stage model_text_max_length（REDESIGN gated）：有 per-stage 表且含 stage→用之；否则回退标量。256 路 per-stage[256]=384=标量 → 零变化。"""
        if self.model_text_max_length_per_stage and stage in self.model_text_max_length_per_stage:
            return int(self.model_text_max_length_per_stage[stage])
        return int(self.model_text_max_length)

    def truncate_tok_for(self, stage: str) -> tuple[int, int]:
        """per-stage caption_truncate_tok（REDESIGN gated）：有 per-stage 表且含 stage→用之；否则回退标量。"""
        if self.caption_truncate_tok_per_stage and stage in self.caption_truncate_tok_per_stage:
            return tuple(self.caption_truncate_tok_per_stage[stage])
        return tuple(self.truncate_tok)

    def requires_nl(self, stage: str) -> bool:
        """gated：该 stage 是否要求 anime 行有 NL（drop tag-only 行）→ 实现"此阶段不用 tag"。空表=False=零变化。"""
        return stage in (self.require_nl_stages or ())

    def route_weights(self, stage: str, has_nl: bool, caption_policy: str | None = None) -> dict[str, float]:
        """选 caption 路由权重表：no-NL→纯 tag；已知 policy→policy 表；否则 per-stage 默认表。"""
        if not has_nl:
            return dict(self.route_weights_no_nl)
        if caption_policy:
            for key, tbl in self.policy_tables.items():
                if caption_policy in key.split("|"):   # config key 形如 "concept_locked|concept_backfill"
                    return dict(tbl)
        return dict(self.stage_route_weights_has_nl[stage])


@dataclass(frozen=True)
class AnimeSamplingCfg:
    """anime 域采样优先级（ct_weight base + 加性 bonus + lane caps + repeat）。"""
    balance_axis: str                         # "cluster_balance"
    priority_base: str                        # "ct_weight"
    max_priority: float                       # 3.0
    max_repeat: int                           # 4（每行整数 repeat 上限）
    extra_frac: float                         # 0.25（naifu 默认：额外曝光预算 = extra_frac·N，按 priority-1 加权分配）
    quality_bonus_value: float                # 0.30
    quality_bonus_qz_min: float               # 1.5
    quality_bonus_hps_min: float              # 0.0
    lane_bonus: dict[str, float]              # ct_lanes 成员 -> bonus
    lane_prefix_bonus: dict[str, float]      # 前缀 -> bonus（如 "concept_": 0.10）
    composition_clean_bonus: float            # 0.50
    lane_cap_buckets: tuple[str, ...]        # risk_bucket 降权桶
    lane_cap_multiplier: float                # 0.3

    def lane_bonus_for(self, lanes) -> float:
        """ct_lanes（list[str]）→ 累加 lane_bonus + 前缀 bonus。"""
        if not lanes:
            return 0.0
        total = 0.0
        for ln in lanes:
            total += self.lane_bonus.get(ln, 0.0)
            for pfx, b in self.lane_prefix_bonus.items():
                if ln.startswith(pfx):
                    total += b
        return total


@dataclass(frozen=True)
class SamplingConfig:
    raw: dict
    # 跨域
    domain_mix: dict[str, dict[str, float]]          # stage -> {domain: weight}
    monet_real_gpic_ratio: dict[str, list[float]]    # stage -> [monet, gpic]
    # real/synth 域内
    band_mix_real: dict[str, float]                  # {T1..T4: weight}
    tier_bounds_tok: dict[str, list[int]]            # {T1..T4: [lo, hi]}
    band_to_captioner_priority: dict                 # band -> [captioner...] + fallback/flags
    source_alpha: float
    cluster_beta: float
    aes_weighting: bool
    caption_truncate_tok: tuple[int, int]
    caption_dropout: float
    # anime
    anime_caption: AnimeCaptionCfg
    anime_sampling: AnimeSamplingCfg
    # 路径
    manifest_real_synth_glob: str                    # daily_v1/{gpic,monet,monet_synth}
    manifest_anime: str
    anime_caption_source: str
    # 1024 真实 band 供给（real/synth 部分的警告/兜底）
    band_supply_warning: dict = field(default_factory=dict)
    # textatlas（第 5 域）：sidecar manifest 根 + 源图 parquet 根 + 纳入的子集（默认 None=未配置则不启用）
    manifest_textatlas: str | None = None            # sidecar 根 /fsx/.../textatlas_manifest_v1
    textatlas_source_root: str | None = None         # 源图内联 parquet 根 /s3/evan/textatlas5m
    textatlas_subsets: tuple[str, ...] = ()          # 纳入子集（Evan: 9 子集，丢 CoverBook）
    # ---- REDESIGN（2026-06-29，gated；None=回退旧行为，live JSON 无此字段照常）----
    manifest_real_synth_roots: dict | None = None    # per-domain manifest 根 {monet_real,gpic,synthetic}（替单 glob；real→rich_v1, synth→synthetic_v1）
    monet_subsource_explicit_probs: dict | None = None  # MONET 子源显式概率表 {laion,coyo,...}（替 count^-0.2 软拉平）

    # ---- 便捷方法 ----
    def domain_mix_renorm(self, stage: str, present: set[str]) -> dict[str, float]:
        """按 loader_rules.domain_mix_renorm：只保留 present 的域、重归一到 1.0。"""
        raw = {d: w for d, w in self.domain_mix[stage].items() if d in present and w > 0}
        z = sum(raw.values())
        assert z > 0, f"stage {stage} 无可用域（present={present}）"
        return {d: w / z for d, w in raw.items()}

    def band_of_tok(self, tok: int) -> str | None:
        """token 数 → 档位（tok<=0 视为不可用，返回 None；loader rule #3/#4）。"""
        if tok is None or tok <= 0:
            return None
        for b in BANDS:
            lo, hi = self.tier_bounds_tok[b]
            if lo <= tok <= hi:
                return b
        return None


def _captioner_key_map():
    """band_to_captioner_priority 里的简名 → MONET 源 parquet 列名 / manifest tok_* 后缀。"""
    return {
        "gemini": "gemini", "sharegpt4v": "sharegpt4v", "internvl": "internvl",
        "florence": "florence", "original": "original",
    }


def load_sampling_config(path: str) -> SamplingConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # stage 列表从 JSON domain_mix keys 派生（REDESIGN: 支持 256/512a/512b；旧版 256/512/1024 照常）。
    stages = [s for s in raw["domain_mix"] if not s.startswith("_")]
    dm = {s: {d: float(w) for d, w in raw["domain_mix"][s].items() if not d.startswith("_")}
          for s in stages}
    for s in stages:                                  # 校验每 stage 名义和≈1（renorm 前）
        tot = sum(dm[s].values())
        assert abs(tot - 1.0) < 1e-6, f"domain_mix[{s}] 和={tot}≠1"

    ac_raw = raw["anime"]["caption"]
    anime_caption = AnimeCaptionCfg(
        stage_route_weights_has_nl={s: {k: float(v) for k, v in ac_raw["stage_route_weights_has_nl"][s].items()
                                        if not k.startswith("_")} for s in stages},
        route_weights_no_nl={k: float(v) for k, v in ac_raw["route_weights_no_nl"].items()
                             if not k.startswith("_")},
        policy_tables={k: {kk: float(vv) for kk, vv in v.items() if not kk.startswith("_")}
                       for k, v in ac_raw["other_naifu_policy_tables"].items() if not k.startswith("_")},
        tag_frequency_dropout={k: v for k, v in ac_raw["tag_frequency_dropout"].items()
                               if not k.startswith("_")},
        length_cap={k: float(v) for k, v in ac_raw["length_cap"].items() if not k.startswith("_")},
        anchor_keep_probs={k: float(v) for k, v in ac_raw["anchor_keep_probs"].items()
                           if not k.startswith("_")},
        truncate_tok=tuple(ac_raw["caption_truncate_tok"]),
        model_text_max_length=int(ac_raw["model_text_max_length"]),
        dropout=float(ac_raw["caption_dropout"]),
        # REDESIGN per-stage token（gated；缺字段→None→回退标量, 旧 JSON 照常）
        model_text_max_length_per_stage=({s: int(v) for s, v in ac_raw["model_text_max_length_per_stage"].items()
                                          if not s.startswith("_")} if ac_raw.get("model_text_max_length_per_stage") else None),
        caption_truncate_tok_per_stage=({s: list(v) for s, v in ac_raw["caption_truncate_tok_per_stage"].items()
                                         if not s.startswith("_")} if ac_raw.get("caption_truncate_tok_per_stage") else None),
        require_nl_stages=tuple(s for s in (ac_raw.get("require_nl_stages") or []) if not str(s).startswith("_")),  # gated：缺字段→()→零变化
    )

    as_raw = raw["anime"]["sampling"]
    pr = as_raw["priority"]
    anime_sampling = AnimeSamplingCfg(
        balance_axis=as_raw["balance_axis"],
        priority_base=pr["base"],
        max_priority=float(pr["max_priority"]),
        max_repeat=int(pr["max_repeat"]),
        extra_frac=float(pr.get("extra_frac", 0.25)),     # config 未指定时取 naifu 默认 0.25
        quality_bonus_value=float(pr["quality_bonus"]["value"]),
        quality_bonus_qz_min=1.5, quality_bonus_hps_min=0.0,
        lane_bonus={k: float(v) for k, v in pr["lane_bonus"].items() if not k.startswith("_")},
        lane_prefix_bonus={k: float(v) for k, v in pr["lane_prefix_bonus"].items()},
        composition_clean_bonus=float(pr["composition_clean_bonus"]["value"]),
        lane_cap_buckets=tuple(as_raw["lane_caps"]["risk_bucket_downweight"]["buckets"]),
        lane_cap_multiplier=float(as_raw["lane_caps"]["risk_bucket_downweight"]["priority_multiplier"]),
    )

    return SamplingConfig(
        raw=raw,
        domain_mix=dm,
        monet_real_gpic_ratio={s: [float(x) for x in raw["monet_real_gpic_ratio"][s]] for s in stages},
        band_mix_real={b: float(raw["band_mix_real"][b]) for b in BANDS},
        tier_bounds_tok={b: list(raw["tier_bounds_tok"][b]) for b in BANDS},
        band_to_captioner_priority=raw["band_to_captioner_priority"],
        source_alpha=float(raw["source_alpha"]),
        cluster_beta=float(raw["cluster_beta"]),
        aes_weighting=bool(raw["aes_weighting"]),
        caption_truncate_tok=tuple(raw["caption_truncate_tok"]),
        caption_dropout=float(raw["caption_dropout"]),
        anime_caption=anime_caption,
        anime_sampling=anime_sampling,
        manifest_real_synth_glob=raw["manifest"]["real_synth"],
        manifest_anime=raw["manifest"]["anime"],
        anime_caption_source=raw["manifest"]["anime_caption_source"].split(" ")[0],
        band_supply_warning=raw.get("stage_band_supply_warning", {}),
        manifest_textatlas=raw["manifest"].get("textatlas"),
        textatlas_source_root=raw["manifest"].get("textatlas_source"),
        textatlas_subsets=tuple(raw["manifest"].get("textatlas_subsets", []) or []),
        # REDESIGN（gated）：缺字段→None→回退旧行为
        manifest_real_synth_roots=({k: v for k, v in raw["manifest"]["real_synth_roots"].items()
                                    if not k.startswith("_")} if raw["manifest"].get("real_synth_roots") else None),
        monet_subsource_explicit_probs=({k: float(v) for k, v in raw["monet_subsource_explicit_probs"].items()
                                         if not k.startswith("_")} if raw.get("monet_subsource_explicit_probs") else None),
    )
