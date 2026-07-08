"""采样器：4-way domain_mix + real/synth band-first + anime 优先级。

对齐参考实现 t5_sampling_weights.py 的 band-first（band 先于 domain），并扩展到 4 域：
  P(anime) = domain_mix[stage].anime（renorm 后）→ anime 域按 priority 加权抽行；
  否则（real+synth）band-first：抽档 b ~ band_mix_real → 在 {monet,gpic,synth} 中服务 b 的域
    按 domain_mix[d] 抽域 → 域内服务 b 的行按 base_w（source×cluster 因子）抽行。
  → anime 边际 = domain_mix.anime 精确；real 各域边际 ≈ domain_mix（band-first 按服务能力微调，
    与 reference dry-run 验过的"256 MONET:GPIC 精确命中"同机理）。

操作对象是已按 stage elig 过滤的 in-memory pool（dataset 用 shuffle buffer 喂；dry-run 用加载子集）。
"""

from __future__ import annotations

import numpy as np

from .config import BANDS, SamplingConfig


def _weighted_idx(cum: np.ndarray, rng: np.random.Generator) -> int:
    """按累积权重数组抽一个下标。"""
    if len(cum) == 0 or cum[-1] <= 0:                     # 硬化(2026-06-30): 空/全0 cumsum=权重退化(某域/档 base_w 全0)→显式报错, 别静默退化到 idx0
        raise ValueError(f"_weighted_idx: 退化累积权重(len={len(cum)}, total={cum[-1] if len(cum) else 'NA'})——该域/档权重全0, 检查 base_w/源表/stats")
    r = rng.random() * cum[-1]
    return int(np.searchsorted(cum, r, side="left"))


def compute_repeat_counts(priority: np.ndarray, max_repeat: int, extra_frac: float,
                          rng: np.random.Generator) -> np.ndarray:
    """anime integer-repeat 曝光（忠实 naifu sampler._build_sampling_repeat_counts 的 weighted 模式）。

    每行基础 count=1（保最小曝光）；额外预算 budget=round(extra_frac·N) 个曝光，按
    score=max(priority-1,0) 加权、不放回地分配（每轮 take=min(remaining, |eligible|)，count 封顶
    max_repeat）。这是 config priority._note『clipped at cap; then -> integer repeat up to max_repeat』
    + sampling_plan.md step3『priority_score-1 -> weighted extra samples up to max_repeat』的规格。

    返回整数 count 数组（用作 anime 域抽样权重的 cumsum 基；E[曝光]∝count）。有界 boost ⇒ 大头近均匀、
    仅高 priority 行获额外曝光 ⇒ 修复『连续 priority 加权导致 ~uniform-among-cap 饱和』(review F-#3)。"""
    n = len(priority)
    counts = np.ones(n, dtype=np.int64)
    if n == 0 or max_repeat <= 1:
        return counts
    score = np.maximum(priority.astype(np.float64) - 1.0, 0.0)
    if not np.any(score > 0):
        return counts
    budget = min(int(round(n * extra_frac)), n * (max_repeat - 1))
    remaining = budget
    while remaining > 0:
        eligible = np.flatnonzero((counts < max_repeat) & (score > 0))
        if eligible.size == 0:
            break
        probs = score[eligible].astype(np.float64, copy=True)
        s = probs.sum()
        if s <= 0:
            break
        probs /= s
        take = min(remaining, int(eligible.size))
        chosen = rng.choice(eligible, size=take, replace=False, p=probs)
        counts[chosen] += 1
        remaining -= take
    return counts


class DomainPool:
    """一个域在某 stage 的 in-memory 行池 + 采样结构（已过滤 elig_stage）。

    real/synth：按档预建 served 行下标 + base_w 累积；anime：按 priority 累积。
    arrays 是 manifest_index.load_* 的输出子集（已 elig 过滤、各 key 等长 array）。
    """

    def __init__(self, domain: str, arrays: dict, *, max_repeat: int | None = None,
                 extra_frac: float = 0.25, rng: np.random.Generator | None = None):
        self.domain = domain
        self.arrays = arrays
        self.n = len(arrays.get("uid", arrays.get("key", [])))
        if domain == "anime":
            prio = arrays["priority"].astype(np.float64)
            if self.n and max_repeat and max_repeat > 1 and rng is not None:
                # integer-repeat 曝光（faithful naifu）：权重=每行 repeat count（有界 boost）
                w = compute_repeat_counts(prio, max_repeat, extra_frac, rng).astype(np.float64)
            else:
                w = prio                                  # 兜底（无 rng/未配置）：连续 priority 加权
            self._repeat_w = w
            self._cum = np.cumsum(w) if self.n else np.array([0.0])
        else:
            sm = arrays["served_mask"]
            bw = arrays["base_w"].astype(np.float64)
            self._allcum = np.cumsum(bw) if self.n else np.array([0.0])   # synth 注入用（全池 base_w）
            self.band_idx: dict[int, np.ndarray] = {}
            self.band_cum: dict[int, np.ndarray] = {}
            for b in range(4):
                idx = np.where(((sm >> b) & 1).astype(bool))[0]
                self.band_idx[b] = idx
                self.band_cum[b] = np.cumsum(bw[idx]) if len(idx) else np.array([])

    def serves_band(self, b: int) -> bool:
        # 不仅要有行(len>0)，还要该档累积权重>0：否则全 base_w=0 的【退化档】会让 sample_band→_weighted_idx
        # 触发"全0报错"杀掉 loader(单 rank 死→FSDP hang)。multi-AR 分桶把候选子集缩小后更易命中(如某桶某档
        # 只剩 da4k 等 explicit_probs=0 源)。非分桶 live 路径档内恒混有正权重行→此条件恒真→零行为变化。
        return self.domain != "anime" and len(self.band_idx[b]) > 0 and self.band_cum[b][-1] > 0

    def sample_band(self, b: int, rng: np.random.Generator) -> int:
        idx = self.band_idx[b]
        return int(idx[_weighted_idx(self.band_cum[b], rng)])

    def sample_anime(self, rng: np.random.Generator) -> int:
        return _weighted_idx(self._cum, rng)

    def sample_weighted(self, rng: np.random.Generator) -> int:
        """全池按 base_w 抽（synth 固定比率注入用，不分档）。"""
        return _weighted_idx(self._allcum, rng)


class CompleteSampler:
    """给定各域 DomainPool，按策略产出 (domain, local_index) 流。"""

    def __init__(self, cfg: SamplingConfig, stage: str, seed: int = 0):
        self.cfg = cfg
        self.stage = stage
        self.rng = np.random.default_rng(seed)
        self._band_keys = list(BANDS)
        self._band_w = np.array([cfg.band_mix_real[b] for b in BANDS], dtype=np.float64)
        self._band_cum = np.cumsum(self._band_w)

    def _fallback(self, pools: dict[str, DomainPool], present: set[str]) -> tuple[str, int, int]:
        """无服务档可抽时的安全兜底：anime → synthetic/textatlas → 任一 band-first 域全池 base_w。
        生产路径下 elig==served_mask（buffer 只放服务行）故几乎不触发；此处防 KeyError 退化。
        加固(2026-07-06, 零权重竞态修复·护栏层): 跳过累积权重为 0 的域——全零池(如未列 explicit 源的
        零 base_w 行灌满 buffer)对任何加权抽取不可用, 原实现直抽会触发 _weighted_idx 全零 raise
        (512a 正式 6×8 step8300 开桶后 MODE-A 单域 sel 实撞, 见 ops 日志 2026-07-06)。
        textatlas 纳入注入域兜底(原漏: 仅剩 textatlas present 时会误 raise)。"""
        rng = self.rng
        if "anime" in present and pools["anime"]._cum[-1] > 0:
            return "anime", pools["anime"].sample_anime(rng), -1
        for d in ("synthetic", "textatlas"):
            if d in present and pools[d]._allcum[-1] > 0:
                return d, pools[d].sample_weighted(rng), -1
        for d in ("monet_real", "gpic"):                  # 全池 base_w（忽略 band）兜底，不再硬引用 synthetic
            if d in present and pools[d]._allcum[-1] > 0:
                return d, pools[d].sample_weighted(rng), -1
        raise RuntimeError(f"sample_one: 无任何可抽域（present={present}, 全零权重池已剔除）")

    def sample_one(self, pools: dict[str, DomainPool]) -> tuple[str, int, int]:
        """返回 (domain, local_index, band)。band = 抽中档(0..3，仅 monet/gpic) 或 -1(anime/synth)。
        注入域（anime priority / synthetic base_w）固定比率；MONET+GPIC 走 band-first。"""
        cfg, stage, rng = self.cfg, self.stage, self.rng
        present = {d for d, p in pools.items() if p.n > 0}
        dm = cfg.domain_mix_renorm(stage, present)
        r = rng.random()
        acc = 0.0
        # 1) 注入域（固定比率）
        if "anime" in present:
            acc += dm.get("anime", 0.0)
            if r < acc:
                return "anime", pools["anime"].sample_anime(rng), -1
        if "synthetic" in present:
            acc += dm.get("synthetic", 0.0)
            if r < acc:
                return "synthetic", pools["synthetic"].sample_weighted(rng), -1
        if "textatlas" in present:                         # 第 5 域（注入域，同 synth：全池均匀 base_w）
            acc += dm.get("textatlas", 0.0)
            if r < acc:
                return "textatlas", pools["textatlas"].sample_weighted(rng), -1
        # 2) band-first（MONET-real + GPIC）
        bf = [d for d in ("monet_real", "gpic") if d in present]
        if not bf:                                        # 只有注入域 present（r 溢出兜底）
            return self._fallback(pools, present)
        for _ in range(8):                                # 抽档；无服务则在有服务档重抽
            b = int(np.searchsorted(self._band_cum, rng.random() * self._band_cum[-1], side="left"))
            cand = [(d, cfg.domain_mix[stage][d]) for d in bf if pools[d].serves_band(b)]
            if cand:
                break
        else:                                             # 兜底：任取有服务行的 (域,档)
            cand = []
            for d in bf:
                for bb in range(4):
                    if pools[d].serves_band(bb):
                        cand, b = [(d, 1.0)], bb
                        break
                if cand:
                    break
            if not cand:
                return self._fallback(pools, present)
        dom_w = np.array([w for _, w in cand], dtype=np.float64)
        d = cand[int(np.searchsorted(np.cumsum(dom_w), rng.random() * dom_w.sum(), side="left"))][0]
        return d, pools[d].sample_band(b, rng), b


def build_pool_from_shards(domain: str, shard_arrays: list[dict], stage: str, *,
                           max_repeat: int | None = None, extra_frac: float = 0.25,
                           rng: np.random.Generator | None = None) -> DomainPool:
    """把若干已加载 shard 的 arrays 按 stage elig 过滤后拼成一个 DomainPool（dry-run / buffer 用）。
    anime 域可传 max_repeat/extra_frac/rng 启用 integer-repeat 曝光（否则退化为连续 priority 权重）。"""
    # REDESIGN: elig 列按【分辨率】键控(256/512/1024)，stage 标签 512a/512b 都映射到 elig_512；旧 stage(256/512/1024) 原样透传
    elig_key = f"elig_{ {'512a': 512, '512b': 512}.get(stage, stage) }"
    parts = []
    for a in shard_arrays:
        mask = a[elig_key]
        sub = {k: (v[mask] if isinstance(v, np.ndarray) else
                   [x for x, m in zip(v, mask) if m]) for k, v in a.items()
               if k != "domain"}
        parts.append(sub)
    # 拼接
    merged: dict = {}
    keys = parts[0].keys() if parts else []
    for k in keys:
        vals = [p[k] for p in parts]
        if isinstance(vals[0], np.ndarray):
            merged[k] = np.concatenate(vals) if vals else np.array([])
        else:
            merged[k] = [x for v in vals for x in v]
    return DomainPool(domain, merged, max_repeat=max_repeat, extra_frac=extra_frac, rng=rng)
