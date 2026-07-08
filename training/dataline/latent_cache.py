"""训练读 FLUX.2 VAE moments 缓存（offline latent cache 的训练侧消费）。

LatentCacheIndex：加载 stage={S}/uid2path.parquet（uid->相对 latents 路径），按 uid 取回
缓存 moments（`[2*32, S/8, S/8]` bf16，即 `latent_dist.parameters`）。模型侧
`model_mot.prepare_latents` 再用 `Flux2VAE.moments_to_model_space()` 转 model-space latent
并每次重采样 posterior（与在线 VAE 同分布）。

设计：anime 训练读源 parquet、缓存按薄 manifest 切片 → shard 名不对齐，故用**全局 uid->path**
索引（scripts/build_latent_cache_index.py 产出）。取 moments 时**按文件分组批量读**（一个
safetensors 含数千 uid），减少 mountpoint-s3 随机 open。缺失 uid 不返回（训练侧直接跳过）。
线程安全：索引为只读 dict（多 producer 线程共享）；get_many 每次独立 safe_open。

★流水线读取（2026-06-23）：一个 batch 的 uid 经 shuffle buffer 后散落在 ~数十个不同
safetensors（多为 1 uid/文件）→ 原 get_many 串行逐文件 `safe_open` 在 mountpoint-s3 上是
冷 open（~0.3-0.5s/文件）→ 4×8 实测 ~10s/step（GPU 空转、纯 I/O 瓶颈，比在线 encode 慢 5×）。
改为用 ThreadPoolExecutor 并行各文件的 `safe_open`（safetensors 读 I/O 释放 GIL，线程有效），
把 N 次串行冷 open 压成 ~N/workers 批 → 目标贴近在线 encode 的流水线吞吐。
"""
from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from safetensors.torch import safe_open


class LatentCacheIndex:
    def __init__(self, stage_dir: str, index_name: str = "uid2path.parquet",
                 read_workers: int = 16):
        self.stage_dir = stage_dir
        self.read_workers = max(1, int(read_workers))
        self._pool = ThreadPoolExecutor(max_workers=self.read_workers) if self.read_workers > 1 else None
        idx_path = os.path.join(stage_dir, index_name)
        if not os.path.exists(idx_path):
            raise FileNotFoundError(
                f"latent cache 索引不存在: {idx_path}\n"
                f"先跑: python scripts/build_latent_cache_index.py --stage <S>"
            )
        t = pq.ParquetFile(idx_path).read(columns=["uid", "path"])  # 单文件读，绕 hive 分区推断
        # path 去重（仅 ~1.6k 个不同文件）省内存：uid -> 同一 path 字符串对象
        path_pool: dict[str, str] = {}
        uid2path: dict[str, str] = {}
        for u, p in zip(t.column("uid").to_pylist(), t.column("path").to_pylist()):
            uid2path[u] = path_pool.setdefault(p, p)
        self.uid2path = uid2path

    def __contains__(self, uid: str) -> bool:
        return uid in self.uid2path

    def __len__(self) -> int:
        return len(self.uid2path)

    def _read_one_file(self, rel: str, us: list[str]) -> dict:
        """读单个 safetensors 内的若干 uid（一次 open 复用句柄）。返回 {uid: tensor}，缺失/坏文件 → 空。"""
        fpath = os.path.join(self.stage_dir, rel)
        local = {}
        try:
            with safe_open(fpath, framework="pt", device="cpu") as h:
                for u in us:
                    try:
                        local[u] = h.get_tensor(u)
                    except Exception:          # noqa: BLE001 单 key 缺失（索引/文件不一致）→ 跳过该 uid
                        continue
        except Exception:                      # noqa: BLE001 文件缺失/损坏 → 这批 uid 当缺失跳过
            pass
        return local

    def get_many(self, uids) -> dict:
        """按文件分组批量读，返回 {uid: moments_tensor[64,S/8,S/8] bf16}。缺失/坏文件的 uid 不在返回里。
        各文件 `safe_open` 用线程池并行（mountpoint-s3 冷 open I/O-bound，读时释放 GIL）→ 把串行冷 open 流水线化。"""
        by_file: dict[str, list[str]] = defaultdict(list)
        for u in uids:
            rel = self.uid2path.get(u)
            if rel is not None:
                by_file[rel].append(u)
        out = {}
        if not by_file:
            return out
        if self._pool is None or len(by_file) <= 1:
            for rel, us in by_file.items():
                out.update(self._read_one_file(rel, us))
            return out
        for local in self._pool.map(lambda kv: self._read_one_file(*kv), list(by_file.items())):
            out.update(local)
        return out
