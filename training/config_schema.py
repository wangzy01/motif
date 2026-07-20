"""轻量 config schema + `validate_config(cfg)`：启动早期的"拼写错误护栏"。

动机（commit de65106 踩坑）：train.py / model_mot.py / arch_mot.py /
training/dataline/complete_dataset.py 各读各的 config 字段（多为 `.get(key, default)`），
拼错 key 不会报错——静默退默认值，整轮训练配方与预期不符却毫无预警。

本模块给出一份**已知字段清单 + 类型**，并在 train.py 启动早期调用 `validate_config(cfg)`：
对**未知 key / 类型不符**只 **warn（打印 [config] 警告）**，**绝不 raise、绝不修改 cfg**。
这是一个纯加法式护栏：

  🔴 硬约束
  ----------
  * 对所有现有有效配置必须**零 ERROR**、**运行行为完全不变**。
  * validate_config 不写回、不归一化、不抛异常；它只读 cfg 并 print 警告。
  * 默认值/类型解析仍由各消费点的 `.get(...)` 负责——本模块不替它们做转换。

设计要点
--------
* 字段以"点路径"声明（如 ``model.text_adapter.enabled``）。
* 三类节点：
    - 标量字段：声明期望 Python 类型（用 OmegaConf 解析后的原生类型判定）。
    - 结构化子树（``dict`` 声明）：递归校验已知子键。
    - **open 子树**（``OPEN`` 标记）：该子树整体作为 ``**kwargs`` 传给某构造器
      （dataset/vae/optimizer.params/tiny_text_config 等都用 ``**_`` 兜底接收），
      未知子键属正常用法，**不告警**；但其中**显式列出的**已知子键仍校验类型。
* 类型判定宽松（避免误报）：
    - ``int`` 接受 bool 之外的整数；``float`` 接受 int/float（YAML 里 1 与 1.0 常混用）；
    - ``str`` 接受 str；``bool`` 接受 bool；
    - ``list`` 接受 list/tuple（OmegaConf ListConfig 也算）。
    - 声明为 ``None``（NO_TYPECHECK）的字段：只认 key、不查类型（值形态多变，如
      ``betas`` 列表、``rope_scaling`` 嵌套等）。
"""

from __future__ import annotations

# 标记对象 -------------------------------------------------------------------- #
OPEN = object()          # 该子树作为 **kwargs 透传给构造器：未知子键不告警
NO_TYPECHECK = None      # 已知 key，但值形态多变 → 只认 key 不查类型


# --------------------------------------------------------------------------- #
# Schema：点路径 → 期望类型 / 子 schema / OPEN / NO_TYPECHECK
# 仅列"已知字段"。未在此出现的 key → 告警为 unknown（除非落在 OPEN 子树内）。
# 类型用 Python 原生类型；OmegaConf 解析后的容器用 dict/list。
# --------------------------------------------------------------------------- #
SCHEMA = {
    # --- 顶层 --- #
    "model_cls": str,
    "target_device": str,
    "skip_init_checkpoint_load": bool,   # model_mot：给 pretrained_path 时跳过加载（调试）

    # --- model.* （networks/model_mot.py + arch_mot.py 消费）--- #
    "model": {
        "pretrained_path": NO_TYPECHECK,       # str 或 null（tiny 模式）
        "tokenizer_path": str,
        "patch_size": int,
        "time_gap": int,
        "timestep_multiplier": float,
        "text_max_length": int,
        "cfg_dropout_prob": float,
        "text_rich_cfg_dropout_prob": float,
        "gradient_checkpointing": bool,
        "rank_independent_seed": bool,         # 多卡 vs 单卡数值一致性验证用
        "sandwich_norm": bool,
        "noise_condition": bool,
        "prompt_template": {
            "enabled": bool,
            "template": str,
        },
        # ---- 消融开关（默认全关；关时不构造参数，对 baseline 零影响）---- #
        "long_skip": {                         # B2 生成侧 long skip
            "enabled": bool,
            "mode": str,                       # A_full | B_outer
        },
        "text_adapter": {                      # B1 文本 adapter
            "enabled": bool,
            "num_layers": int,
            "bidirectional": bool,
            "rope": bool,
            "multi_layer_fusion": {            # Ablation 2：融合 und 末 N 层 → 喂 adapter
                "enabled": bool,
                "num_layers": int,
            },
        },
        "und_lora": {                          # Ablation 3：und attention 顶层 LoRA
            "enabled": bool,
            "num_layers": int,
            "rank": int,
            "alpha": float,
            "lr_scale": float,
            "target_modules": NO_TYPECHECK,    # list[str] 或逗号串
        },
        "abs_sincos": {                        # B4 绝对 sinusoidal
            "enabled": bool,
        },
        # tiny 调试用迷你文本塔：整体透传给 HF text config，键多变 → open。
        "tiny_text_config": OPEN,
    },

    # --- vae.* （networks/vae.py build_vae，**cfg 透传给各 VAE 构造器）--- #
    # name 必填语义在 build_vae 里；这里只列已知键、整体 open（FakeVAE/Flux2VAE/QwenImageVAE 各取所需）。
    "vae": {
        "__open__": True,
        "name": str,
        "pretrained_path": str,
        "dtype": str,
        "downsample_factor": int,
        "latent_channels": int,
        "sample_posterior": bool,
    },

    # --- dataset.* （**config.dataset 透传给 store_cls，构造器带 **_ 兜底）--- #
    # 整体 open（不同 store_cls 取不同子集；CompleteV1Dataset 与 StubT2IDataset 字段不同）。
    "dataset": {
        "__open__": True,
        "store_cls": str,
        "offline_plan_root": str,
        "offline_catalog_root": str,
        "offline_anime_freq_root": str,
        "offline_caption_sidecar_root": str,
        "offline_gpic_blob_root": str,
        "caption_sidecar_workers": int,
        "anime_caption_root": str,
        "sampling_config": str,
        "stage": NO_TYPECHECK,                 # "256"/"512"/"1024"（YAML 里是 str）
        "image_size": int,
        "buffer_size": int,
        "queue_size": int,
        "min_fill": int,
        "max_shards": NO_TYPECHECK,            # int 或 null
        "max_rows_per_shard": NO_TYPECHECK,    # int 或 null
        "anime_nrows": int,
        "anime_cache_dir": str,
        "stats_cache": str,
        "tokenizer_path": str,
        "caption_token_budget": NO_TYPECHECK,  # int 或 null
        "template_reserve": int,
        "use_latent_cache": bool,
        "latent_cache_root": str,
        "latent_cache_index_name": str,
        "first_fill_policy": str,              # all_min | any_pool
        "first_fill_timeout_s": float,
        "fetch_workers": int,                  # gpic 逐文件并发取图线程数（默认 8）；1=串行
        "fetch_workers_anime": int,            # anime 逐 URL 并发取图线程数（默认 1=串行：anime 缓存/R2 域并发只过订阅挂载、不提配比，反饿死 tar 域）
        "fetch_workers_tar": int,              # offline exact:monet/synth 合并读的逐 tar 并发数（默认 8;2026-07-11 gpuval r3:随机 plan 下串行逐 tar=每窗口 60-100s 停顿）
        "caption_workers_anime": int,          # offline exact:anime caption parquet row-group 并发数（默认 1）
        "caption_workers_monet_real": int,     # offline exact:monet_real caption parquet row-group 并发数（默认 1）
        "caption_workers_gpic": int,           # offline exact:gpic caption parquet row-group 并发数（默认 1）
        "caption_workers_synthetic": int,      # offline exact:synthetic caption parquet row-group 并发数（默认 1）
        "gpic_payload_in_memory": bool,        # offline exact:直接把 GPIC 原图 bytes 留在预取窗口，跳过 /local 写后再读
        "gpic_source_consistency_mode": str,   # offline exact:strict_path(默认,fd+路径双重稳定) | fd_stable(仅已打开 inode 稳定)
        "gpic_threads_per_node": int,          # offline exact:GPIC 节点级总并发；正式显式16，环境变量只能同值确认、不可静默覆盖
        "gpic_payload_backend": str,           # offline exact:source_files_v1 | offline_gpic_blob_v1；blob 模式全量覆盖且零 source fallback
        "gpic_blob_manifest_sha256": str,      # offline exact:FSx canonical manifest 文件 SHA；_SUCCESS/index/verifier/S3 blob map 由其闭合
        "gpic_blob_fd_cache_size": int,        # offline exact:每 rank lease-aware S3 blob FD LRU 上限；正式W64/probation显式冻结，首选>=manifest blob_count（预估候选1024），并受RLIMIT_NOFILE/FD/RSS门约束
        "producer_native_io_serial": bool,     # 诊断/安全阀：每 rank 串行执行 producer 的 native I/O/decode/resize next()，q.put 不进锁；不改变采样语义
        "drain_mode": str,                     # buffer 排空策略：uniform(默认,旧)/proportional(F1,排空∝消费)/keep_present(F2,F1+目标域空时阻塞补料防 renorm-away)
        "keep_present_timeout_s": float,       # keep_present(F2) 下单域阻塞补料的超时(s)；到点放弃该域回退 renorm，防慢域死等拖速
        "textatlas_local_root": NO_TYPECHECK,  # 第5域(textatlas)源图 /local NVMe 预存根(str 或 null)；优先读、缺失回退 cfg.textatlas_source_root(S3)保正确性
        "crop_v_anchor": NO_TYPECHECK,         # 缺头修复(Path B, 2026-07-05)：per-domain 裁剪竖直锚点 dict {domain: v∈[0,1]}(0=贴顶保头/0.5=居中=旧行为/1=贴底); 缺省域=0.5 逐字节不变(如 {anime:0.15})
        # ---- multi-AR 分桶（REDESIGN Phase2b；整体 open：未知子键不告警, 已知子键松校验）---- #
        "multi_ar": {
            "__open__": True,
            "enabled": bool,
            "assignment_mode": str,             # current | item_queue；item_queue=NAI-like item-level bucket assignment + per-bucket queues
            "bucket_mode": str,                 # fixed | adaptive；adaptive=naifu-like dense flat bucket table near target_area
            "target_area": int,                # 等面积桶目标像素数（~512²=262144）
            "divisor": int,                    # 每边整除因子（FLUX.2 VAE f16→16）
            "ar_list": NO_TYPECHECK,           # list[float]：w/h>1 的 AR（自动补 1:1 + 各自倒数）
            "adaptive_no_upscale": bool,        # adaptive bucket assignment: raw_area<=target_area 时不放大
            "adaptive_min_area_ratio": float,   # adaptive bucket table: 保留 area >= target_area*ratio 的 bucket
            "adaptive_max_aspect": float,       # adaptive bucket table/extreme clamp: 允许最大 W/H 或 H/W
            "bucket_buffer_size": int,          # item_queue: 每 bucket×domain 队列上限
            "bucket_min_fill": int,             # item_queue: 目标 bucket 至少多少条才开抽
            "bucket_wait_timeout_s": float,     # item_queue: 目标 bucket 等待超时（仍不跨 rank 自选其它桶）
            "warmup_steps": int,               # square-only 预热步数（回退值）
            "warmup_exposure_images": int,     # >0：按 曝光/global_batch 派生 warmup_steps（gated）
            "bucket_probs": NO_TYPECHECK,      # list[float]：per-bucket 先验（对齐 build_buckets 桶序）；配了 native_bucket_dist 时仅作参考(派生 P(B) 为准)
            "square_only_domains": NO_TYPECHECK,  # list[str]：护栏2——这些域不 re-resize 进非方桶(无 native 表→MODE-B alpha 补偿; 有 native 表→one-hot 吸收进 MODE-A), 全局 mix 不变; 见 complete_dataset._square_only_pools/_build_native_affinity
            "native_bucket_dist": NO_TYPECHECK,   # Tier2(MODE-A affinity, 2026-07-05)：dict {domain: [n_buckets 浮点]} 实测 P(bucket|domain)(桶序=build_buckets)。配了→域采样 P(d|B)=mix·P_native/P(B)、调度 P(B)=Σmix·P_native(替代 bucket_probs)→全局 mix 精确不变+结构性 re-resize 大降; 缺省→MODE-B/旧行为
        },
        # ---- StubT2IDataset 专用 ---- #
        "mode": str,                           # latents | pixels
        "latent_channels": int,
        "downsample_factor": int,
        "num_batches_per_epoch": int,
        "overfit_one": bool,
        "seed": int,
    },

    # --- sigma_sampling.* （model_mot._sample_sigmas / _current_shift 消费）--- #
    "sigma_sampling": {
        "dist": str,
        "logit_mean": float,
        "logit_std": float,
        "shift": float,
        "shift_start": float,                  # _current_shift 的备选键
    },

    # --- optimizer.* （train.py：name 仅注释，params 透传 AdamW）--- #
    "optimizer": {
        "name": str,
        "params": {
            "__open__": True,                  # 透传 torch.optim.AdamW(**params)
            "lr": float,
            "weight_decay": float,
            "betas": NO_TYPECHECK,             # [beta1, beta2]
            "eps": float,
        },
    },

    # --- scheduler.* （training/checkpoint.py build_scheduler 消费）--- #
    "scheduler": {
        "num_warmup_steps": int,
        "decay": str,                          # constant | cosine
    },

    # --- trainer.* （train.py + diagnostics.py + checkpoint.py 消费）--- #
    "trainer": {
        "seed": int,
        "amp_dtype": NO_TYPECHECK,             # "bfloat16"/"bf16" 或 null
        "batch_size": int,
        "max_steps": int,
        "max_exposure_images": int,            # 选填(gated)：给了则 max_steps=ceil(曝光量/global_batch)，步数随卡数自适应；不给用显式 max_steps
        "global_batch_size": int,              # 选填：配 max_exposure_images 用的显式 global batch；不给则按 batch_size×world_size 现算
        "gradient_clip_val": float,
        "log_every": int,
        "dataline_metrics_every": int,         # 每多少步把 dataline 真值（realized mix/buffer/吞吐/取图）写 metrics.jsonl；<=0 关，默认对齐 log_every
        "checkpoint_steps": int,
        "checkpoint_keep_every": int,
        "checkpoint_keep_max": int,
        "checkpoint_keep_format": str,         # weights_ema | full
        "checkpoint_dir": str,
        "resume": bool,
        "init_from": NO_TYPECHECK,             # str 或 null：跨分辨率阶段切换的上一阶段 final ckpt 路径；仅本地无 last.pt 时触发（resume 优先），载权重(+optimizer 动量)、重置 step、scheduler 全新 re-warmup
        "init_load_optimizer": bool,           # init_from 时是否继承 optimizer 动量（默认 true，设计 §6 要继承 Adam moment；只继承动量, 超参用本阶段 config）
        "allow_unsafe_init_from_no_fp": bool,  # init_from 缺 ablation_fp 时是否放行（默认 false=硬报错；显式 true 跳过, 须自行确认 arch 一致）
        "save_final": bool,
        "skip_distributed_teardown": bool,
        "force_os_exit_after_train": bool,
        "gc_gen2_threshold": int,              # gated GC 修复(2026-07-03 creep)：>0 调大 gen2 触发阈值(默认0=不动, 建议200)
        "gc_freeze_after_steps": int,          # gated GC 修复(2026-07-03 creep)：>0 进程启动后第N步 gc.collect+freeze(默认0=关, 建议50)
        "gc_refreeze_every": int,              # gated GC 修复(2026-07-05 p90/p99 creep)：>0 进程启动后每N步 gc.collect+freeze，治 post-freeze 长存对象累积(默认0=关, 建议100+)
        # ---- 诊断开关（默认关，关时对训练逐位无影响）---- #
        "debug_step_trace": bool,
        "debug_step_trace_dir": NO_TYPECHECK,  # str 或 null
        "debug_trace_dump_interval": float,
        "debug_faulthandler_dump_interval": float,
        "batch_ready_barrier": bool,
        "batch_ready_barrier_dir": NO_TYPECHECK,      # str 或 null
        "batch_ready_barrier_run_id": NO_TYPECHECK,   # str 或 null
        "batch_ready_barrier_timeout_s": float,
        # ---- EMA（G5）---- #
        "ema": {
            "enabled": bool,
            "decay": float,
        },
        # ---- wandb（可选，默认关）---- #
        "use_wandb": bool,
        "wandb_project": str,
        "wandb_entity": NO_TYPECHECK,          # str 或 null
        "wandb_mode": str,                     # online | offline
        "run_name": NO_TYPECHECK,              # str 或 null
    },
}


# --------------------------------------------------------------------------- #
# 类型判定（宽松，避免误报）
# --------------------------------------------------------------------------- #
def _type_name(t) -> str:
    return getattr(t, "__name__", str(t))


def _matches(value, expected_type) -> bool:
    """宽松类型匹配：YAML/OmegaConf 解析后的原生值 vs 期望 Python 类型。"""
    if value is None:
        # null 对任何已知字段都放行（多数字段允许 null 退默认/关闭）。
        return True
    if expected_type is bool:
        return isinstance(value, bool)
    if expected_type is int:
        # bool 是 int 子类，但语义不同 → 排除；其余整数放行。
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type is float:
        # YAML 里 1 与 1.0 常混用 → int 也算 float（不含 bool）。
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type is str:
        return isinstance(value, str)
    if expected_type is list:
        try:
            from omegaconf import ListConfig
            if isinstance(value, ListConfig):
                return True
        except Exception:
            pass
        return isinstance(value, (list, tuple))
    if expected_type is dict:
        return _is_mapping(value)
    return True  # 未知期望类型：不挡


def _is_mapping(value) -> bool:
    if isinstance(value, dict):
        return True
    try:
        from omegaconf import DictConfig
        return isinstance(value, DictConfig)
    except Exception:
        return False


def _iter_keys(node):
    """统一遍历 dict / OmegaConf DictConfig 的键。"""
    if isinstance(node, dict):
        return list(node.keys())
    try:
        from omegaconf import OmegaConf
        if _is_mapping(node):
            return list(node.keys())
    except Exception:
        pass
    return []


def _get(node, key):
    try:
        return node[key]
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 递归校验
# --------------------------------------------------------------------------- #
def _validate_node(cfg_node, schema_node, path, warnings):
    """校验单层。schema_node 形态：
        * OPEN              → 整子树透传，子键全部不告警（不递归）。
        * dict（schema）    → 递归；含 '__open__': True 时未知子键不告警但已知键仍查类型。
        * 类型 / NO_TYPECHECK → 不应到这里（标量在父层处理）。
    """
    if schema_node is OPEN:
        return  # 整体透传，放行所有子键

    if not _is_mapping(cfg_node):
        # 配置里该路径不是 mapping，但 schema 期望子结构 → 类型不符告警（除非 None）。
        if cfg_node is not None:
            warnings.append(
                f"[config] WARN: {path or '<root>'} 期望为结构化子树(mapping)，实际为 "
                f"{type(cfg_node).__name__}={cfg_node!r}"
            )
        return

    is_open = bool(schema_node.get("__open__", False))
    for key in _iter_keys(cfg_node):
        if key == "__open__":
            continue
        child_path = f"{path}.{key}" if path else key
        value = _get(cfg_node, key)
        if key not in schema_node:
            if not is_open:
                warnings.append(
                    f"[config] WARN: 未知字段 {child_path}（不在 schema 中；可能是拼写错误，"
                    f"将被消费点静默退默认值）"
                )
            # open 子树内的未知 key：正常透传，不告警。
            continue
        spec = schema_node[key]
        if spec is OPEN or _is_mapping(spec):
            _validate_node(value, spec, child_path, warnings)
        elif spec is NO_TYPECHECK:
            continue  # 已知 key，不查类型
        else:
            if not _matches(value, spec):
                warnings.append(
                    f"[config] WARN: 字段 {child_path} 类型不符：期望 {_type_name(spec)}，"
                    f"实际 {type(value).__name__}={value!r}"
                )


def validate_config(cfg, log=None) -> list:
    """对 cfg 跑 schema 校验，**只 warn 不 raise、不修改 cfg**。

    参数
    ----
    cfg : OmegaConf DictConfig（或普通 dict）
    log : 可选 callable(str)；默认用 print(flush=True)。仅用于输出警告，
          调用方可传 rank0-only 的 log 以避免多卡刷屏。

    返回
    ----
    list[str]：所有告警消息（也已逐条打印）。无告警时为空列表。
    """
    emit = log if callable(log) else (lambda m: print(m, flush=True))
    warnings: list = []
    _validate_node(cfg, SCHEMA, "", warnings)
    if warnings:
        emit(f"[config] schema 校验发现 {len(warnings)} 条告警（仅提示，不影响运行）：")
        for w in warnings:
            emit("  " + w)
    return warnings


# 独立 CLI：python -m training.config_schema configs/xxx.yaml [...]
if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf

    paths = sys.argv[1:]
    if not paths:
        print("用法: python -m training.config_schema <config.yaml> [更多...]")
        raise SystemExit(2)
    total = 0
    for p in paths:
        print(f"\n===== {p} =====")
        c = OmegaConf.load(p)
        ws = validate_config(c)
        total += len(ws)
        if not ws:
            print("[config] OK：无告警。")
    print(f"\n[config] 全部 {len(paths)} 个配置共 {total} 条告警；ERROR 数=0（本校验从不 raise）。")
