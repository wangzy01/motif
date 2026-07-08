# MoT complete_v1 — FLUX.2 VAE moments 离线缓存（交接 / 使用文档）

**作者**：执行 AI（2026-06-17）｜**状态**：脚本+smoke 完成；**stage 256 全量已完成 + QC 通过（2026-06-22，见 §6.1）**；512/1024 代码就绪、未提交。
**一句话**：把 complete_v1 四域图像离线编码成 **FLUX.2 VAE posterior moments**（`[mean‖logvar]`），训练时按 key 取回、再 `moments_to_model_space()` 重采样，省掉在线 VAE。

> 配套：VAE adapter [networks/vae.py](../networks/vae.py)、训练侧消费点 [networks/model_mot.py](../networks/model_mot.py) `prepare_latents()`、数据交付 [docs/AAA数据交付文档_GPIC+MONET+anime_v2_20260617.md](AAA数据交付文档_GPIC+MONET+anime_v2_20260617.md)。

---

## 0. 为什么是 moments 而不是 sampled latent

缓存 `ae.encode(px).latent_dist.parameters`，即 channel 维上的 `[mean ‖ logvar]`，shape `[64, S/8, S/8]`（32×2=64 通道）。**不缓存单次 sampled latent**：训练侧每个 epoch 用
`Flux2VAE.moments_to_model_space(parameters)` 重建 `DiagonalGaussianDistribution` 再 `sample()`，
保留 VAE posterior 随机性（与在线 encode 行为一致）。对齐 naifu `scripts/batch_vae_flux.py` 的做法。

- **精度**：bf16（与训练 config `vae.dtype=bfloat16`、naifu 参考、容量预算一致）。
- 训练侧拿到 moments 后：`moments_to_model_space()` → model-space latent `[128, S/16, S/16]`（2×2 patchify + BN 标准化，VAE adapter 内部做），即 `model_mot` patch_size=1 直接吃的 latent。

---

## 1. 产出路径与结构

根：`/s3/evan/mot_latents/complete_v1_flux2_moments/`（mountpoint-s3）

```
stage=256/   stage=512/   stage=1024/
  latents/   {domain}__{manifestshard}__r{rank}__{seq}.safetensors   # tensor_key = uid
  index/     同名 .parquet                                            # commit marker
  failed/    rank{r}.parquet        +  failed_keys.parquet（QC 合并）
  run_stats/ rank{r}.json           # 每 rank 吞吐/计数
  summary.json  qc.json             # QC 脚本产出
```

- **tensor_key = uid**：`gpic:{key}` / `monet:{id}` / `anime:{key}`（全局唯一、跨域跨 stage 不撞）。
- safetensors 一个文件含一个 `(domain, manifest_shard)` 的一段 keys（按 ~3GiB 切块）。
- **index parquet 列**：`uid, domain, stage, src_key, path, tensor_key, height, width, latent_shape, status, error`。
  - `path` = 相对 `latents/xxx.safetensors`（用 `os.path.join(stage_dir, path)` 解析）。
  - `height/width` = **原图**宽高（非 crop 后；crop 后恒为 stage×stage）。
  - `latent_shape` = moments 形状字符串，如 `"64,32,32"`（256）/`"64,64,64"`（512）/`"64,128,128"`（1024）。

---

## 2. 怎么用（训练侧读回 latent）

```python
import glob, pyarrow.parquet as pq
from safetensors.torch import safe_open
from networks.vae import Flux2VAE

STAGE = "256"; ROOT = "/s3/evan/mot_latents/complete_v1_flux2_moments"
sd = f"{ROOT}/stage={STAGE}"

# 1) 建 uid -> 相对 safetensors path 索引（一次性，读所有 index parquet 的两列）
uid2path = {}
for p in glob.glob(f"{sd}/index/*.parquet"):
    t = pq.read_table(p, columns=["uid", "path"])
    uid2path.update(dict(zip(t.column("uid").to_pylist(), t.column("path").to_pylist())))

# 2) 取某 uid 的 moments，转 model-space latent（每次重采样 posterior）
vae = Flux2VAE(pretrained_path="/fsx/models/FLUX.2-klein-4B/vae", dtype="bfloat16")
def load_latent(uid, device="cuda"):
    with safe_open(f"{sd}/{uid2path[uid]}", framework="pt", device="cpu") as f:
        moments = f.get_tensor(uid)                       # [64, S/8, S/8] bf16
    return vae.moments_to_model_space(moments.unsqueeze(0).to(device))  # [1,128,S/16,S/16]
```

上面是**原理**；训练侧已落地为生产路径（2026-06-22），见下 §2.1。

### 2.1 训练接入（已实现，生产路径）

**一键开关**：`config.dataset.use_latent_cache=true`（默认 `false`，关时 online 路径**字节不变** →
对在跑训练零风险）。开启后：

1. **一次性建全局索引**（anime 训练读源 parquet、缓存按薄 manifest 切片 → shard 名不对齐，
   故用全局 uid→path）：`python scripts/build_latent_cache_index.py --stage 256`
   → 产 `stage=256/uid2path.parquet`（stage256 实测 37,498,574 uid / 639 MiB）。
2. **数据线**（[training/dataline/complete_dataset.py](../training/dataline/complete_dataset.py)
   `use_latent_cache`）：producer **跳过取图/decode/resize**（`skip_image`），captions/采样/domain-mix
   逻辑全保留；batch 装配时按 uid 经 [latent_cache.py](../training/dataline/latent_cache.py)
   `LatentCacheIndex.get_many`（**按文件分组批量读** moments，减 mountpoint-s3 随机 open）取 moments，
   产 `batch["moments"]`（`[B,64,S/8,S/8]` bf16）；**缓存缺失的 uid 直接跳过**（补采到 bs，
   计入 `dataset.cache_miss_counter`）。
3. **模型侧**（[model_mot.py](../networks/model_mot.py) `prepare_latents`）：additive `moments` 分支 →
   `self.vae.moments_to_model_space(moments)`（复用已验证 VAE 方法、GPU 上转 model-space 并每次
   **重采样 posterior**，与在线 encode 同分布）。`train.py`/forward 无需改。

**好处**：无 R2/无 tar/无在线 VAE → buffer 秒级填满 → **消除域低水位 mix 漂移告警 + 提速**；
覆盖率 99.998%，缺失（713 monet + 极少 gap）跳过、不进训练。

**验证**：
- CPU 组件单测 [scripts/test_latent_cache.py](../scripts/test_latent_cache.py) 全 PASS（索引/get_many 形状·dtype/缺失跳过/anime 命中/moments→model_space mode 确定·sample 重采样；缓存 vs 在线重编码 model-space 相对误差 0.0088=CPU-fp32 vs GPU-bf16 噪声）。
- GPU 端到端 [jobs/latent_cache_smoke.yaml](../jobs/latent_cache_smoke.yaml)：① 数值对拍 [scripts/verify_latent_cache_gpu.py](../scripts/verify_latent_cache_gpu.py)（缓存 vs 在线 VAE，同设备 bf16，相对误差 ~0.5% = bf16+cuDNN 跨运行噪声，错图会是 O(1)）；② `train.py use_latent_cache=true` 真训练步（dataline 读缓存→moments 分支→loss/反向）。〔结果见 §6.2〕

---

## 3. 缓存范围（每 stage 哪些样本进了缓存）

规则：`domain_mix[stage][domain] > 0`（取自 `t5_sampling_complete_v1.json`）∩ `elig_{stage}==1` ∩ `holdout==0`。
**完整遍历 eligible，不走训练采样 loader**（无 buffer/replacement/anime_nrows 偏置）。

| stage | 参与域 | 规模(估) | moments bf16(估) | 备注 |
|---|---|---|---|---|
| 256 | anime, monet_real, gpic, synthetic | ~37.5M | ~4.5 TiB | 全 4 域 |
| 512 | anime, monet_real, **gpic(~1.15M)**, synthetic | ~29.7M | ~14.2 TiB | GPIC@512 已按 config(0.08)纳入 |
| 1024 | anime, monet_real, synthetic | ~16.0M | ~30.6 TiB | GPIC 无 ≥1024px 图，不参与 |

> 范围跟随 config：若 `domain_mix` 改了，重跑会自动按新参与域缓存。GPIC@1024 物理上 0 张 elig_1024。

预处理与在线训练**逐位一致**：复用 [training/dataline/image_fetch.py](../training/dataline/image_fetch.py)
`resize_crop_uint8`（短边→S Lanczos、center-crop S×S、仅下采样）+ `uint8_to_norm_tensor`（[-1,1] CHW）。

---

## 4. 代码与运行

| 文件 | 作用 |
|---|---|
| [scripts/vae_cache_flux2_moments.py](../scripts/vae_cache_flux2_moments.py) | 缓存主脚本（manifest scanner 全遍历、多 rank、异步取图、batch encode、分块写、resume） |
| [scripts/vae_cache_qc.py](../scripts/vae_cache_qc.py) | 每 stage QC（eligible/ok/fail、每域、shape 分布、文件数/大小、index==tensor、100-key 读回、吞吐、空间） |
| [scripts/_archive/one_time_scripts/smoke_vae_cache.py](../scripts/_archive/one_time_scripts/smoke_vae_cache.py) | smoke：每域每 stage 小样本，验 encode/写盘/读回/`moments_to_model_space`/在线对比/失败记录 |
| [jobs/vae_cache_moments_256.yaml](../jobs/vae_cache_moments_256.yaml) | stage 256 全量 JCC（8×H100 单节点） |
| [jobs/vae_cache_moments_512.yaml](../jobs/vae_cache_moments_512.yaml) | stage 512（未提交） |
| [jobs/vae_cache_moments_1024.yaml](../jobs/vae_cache_moments_1024.yaml) | stage 1024（未提交） |

**提交**：`jcc job submit jobs/vae_cache_moments_256.yaml`；看日志 `jcc job log evan-mot-vae-cache-256 -p all -n 400`。
**手动单机**：`PYTHONPATH=.../mot_pilot_v1 python -m torch.distributed.run --nproc_per_node=8 scripts/vae_cache_flux2_moments.py --stage 256 ...`。

**关键参数**：`--stage`、`--out-root`、`--scratch`（/local）、`--vae-path`、`--anime-cache`、
`--batch-size`、`--chunk-gib`（safetensors 切块 GiB）、`--fetch-workers`（anime/gpic 并发）、`--tar-workers`（monet/synth tar 并发）、
`--only-domains`/`--limit-pairs`/`--limit-per-domain`（smoke 用）。

---

## 5. 重要实现约束 / 坑（接手必读）

> 🔒 **本节为 load-bearing 生产知识，勿删/勿简**：尤其 #4（瓶颈是 tar I/O 非 anime R2）、#8（收尾 `dist.barrier()` NCCL hang，已修）是踩过的真坑，512/1024 直接受益。

1. **safetensors 不能直接写 mountpoint-s3**（缺随机写/mmap，报 `Operation not permitted`）。
   脚本各 rank 先写 `/local` scratch，再**顺序 copy** 到 `/s3`。**先 copy latents、后 copy index**——
   index 存在 = 该 chunk 已 commit（resume / QC 以 index 为准）。别改成直接 `save_file` 到 `/s3`。
2. **resume 按 (domain, manifest_shard) 的 shard stem keyed**：重跑时读该 shard 已有 index part 的 uid 跳过（per-shard done set，规模有界、world_size 变了也安全）。启动清理孤儿 latents（有 .safetensors 无 .parquet）。
3. **多 rank = (domain, shard) pair 全局 stripe**（含跨域均衡）。64 shard/域 → 8 卡每域 8 shard；world≤64 都均分。
4. **吞吐瓶颈是 MONET/synth 的 tar 流式读（S3），不是 anime R2**。anime 走 R2 HTTP 逐图、可大并发（带 `/local` 磁盘缓存）；monet/synth 是 webdataset `.tar`，必须**顺序流式**（`r|`，避免随机 seek），每个 tar 流一次取完它名下全部 wanted key（amortize）。`--tar-workers` 控制并发 tar 流。
5. **GPIC ext 混合**（~88% jpg / 12% png）：脚本读 `gpic_train_{shard}.parquet` 的 `s3key` 取权威 ext，避免逐图 4× HEAD 探测。
6. **失败不杀 job**：取图/decode/resize 失败记入 `failed/`，job 继续。QC 报失败率。
7. **anime 用薄 manifest 而非源 parquet**：`daily_v1/anime/*.parquet`（25 列）含 `uid/key/s3key/width/height/elig_*`，
   其 `s3key` = `https://img.metanomaly.co/r2/<key>`（与源 parquet 一致、已验证可取）。VAE 缓存不需要 caption。
8. **encode 末尾别加 `dist.barrier()`（已踩坑、已删）**：原代码每 rank 写完 run_stats 后调一次 `torch.distributed.barrier()`。但各 rank 之间退出前**无依赖**（latents/index/run_stats 都各自落盘），QC 又是 `torchrun` 返回后才跑的独立单进程命令，而 `torchrun` 本就会 join 所有 rank。stage256 实测 rank 间完成时间差 ~11h（monet tar 慢 / synth 快）> NCCL `init_process_group` 的 8h 超时 → 早完成的 rank 在 barrier 上超时退出，最后一个 rank 到达时**无对端**→ NCCL collective 永久 hang → `torchrun` 不返回 → **QC 从未执行**，job 假 Running 48h 占着 8×H100。修复 = 删掉 barrier，只留 `destroy_process_group`，慢 rank 一写完即独立退出（`vae_cache_flux2_moments.py:542`）。**512/1024 直接受益**。补救已落盘但 QC 没跑的 stage：单独 `python scripts/vae_cache_qc.py --stage <S>` 即可（CPU、只读 /s3）。

---

## 6. smoke / 吞吐 / 实测结果

（运行后补：smoke PASS/FAIL、每域 amortized img/s/rank、stage 256 实际耗时/吞吐/失败率/空间占用/路径。见 §6.1）

### 6.1 实测（2026-06-17）

**smoke（1×H100，每域每 stage 96 张，写 `/s3/.../_smoke_flux2_moments`）：51/52 PASS**
- 缓存 CLI 11 路（256×4 / 512×4 / 1024×3 域）全 rc==0、0 失败。
- moments 形状：256→`[64,32,32]` / 512→`[64,64,64]` / 1024→`[64,128,128]`，全 bf16。
- `moments_to_model_space` → `[1,128,16,16]`/`[1,128,32,32]`/`[1,128,64,64]` = `[128,S/16,S/16]`，bf16，cuda ✓。
- index↔safetensors 一一对应（256/512 各 384 keys、1024 192 keys，0 mismatch）。
- **在线 VAE 对比：online vs cache 同 shape/dtype/device；mode-path max|Δ| = 0.00e+00**（缓存 moments 逐位复现在线 VAE 几何）。
- 失败处理：坏 URL/坏字节→None 不崩；smoke 样本 0 自然失败。
- 唯一 1 个 FAIL = smoke 的在线对比辅助函数 `fetch_some` 在 1024/monet_real 的 16 行窗口取到 0 张（elig_1024 稀疏 + 与 live 训练抢 tar I/O）；非生产路径问题（该域 1024 readback 已 PASS）。

**修过的 bug**：`stage=256/` 目录名触发 pyarrow hive 分区推断，伪造出 int 型 `stage` 分区列、与文件内
string `stage` 列冲突（`ArrowTypeError: Unable to merge Field stage`）。修法：读 `stage=*/` 下 parquet
一律用 `pq.ParquetFile(p).read()`（单文件、不做分区推断），见 `read_parquet_file`。影响 resume/QC/smoke 读路径，**不影响写/encode**（缓存本身正确）。

**吞吐探针（纯 CPU fetch+decode+resize，单进程，gentle FW=32/TW=6，且与 live 训练 dataloader 抢 I/O → 偏保守下界）**：

| 域 | img/s/proc | 备注 |
|---|---:|---|
| gpic | 51.7 | 本地文件，最快 |
| anime | 30.8 | R2 HTTP，健康，**非瓶颈** |
| monet_real | 11.2 | tar 流式（S3） |
| synthetic | 3.9 | 1024px tar 最大最慢 |

> **瓶颈 = MONET/synthetic webdataset tar I/O，不是 anime R2**（与任务预期的"anime R2 瓶颈"相反）。GPU encode
> 不是瓶颈，rank 会 I/O-bound。生产 JCC（8 rank、FW=64/TW=12、专用节点无训练抢占）每 rank 应高于探针。
> **粗估 stage 256 全量 ~1–2 天**（被 monet/synth tar 主导）；真实吞吐以 256 JCC run 的 `run_stats`/`qc.json` 为准。

**stage 256 全量 JCC**（🔒 load-bearing 生产实测数字，勿改）：`evan-mot-vae-cache-256-tx49`（8×H100 单节点，FW=64/TW=12，bs64，chunk 3GiB）。
- 路径 `/s3/evan/mot_latents/complete_v1_flux2_moments/stage=256/`；`summary.json`/`qc.json`/`failed_keys.parquet` 已落盘。
- **实测结果（QC 2026-06-22，全绿）**：

  | 指标 | 值 |
  |---|---|
  | eligible | 37,499,287（anime 16.50M / monet_real 11M / gpic 7M / synth 3M） |
  | ok（已缓存） | **35,765,484**　覆盖率 **95.38%** |
  | fail | 1,733,803（**4.62%**） |
  | 完整性 | `index_eq_tensors=true`（35,765,484==35,765,484）；随机读回 100/100 ok（`[64,32,32]` bf16）；latent_shape 全 `64,32,32` |
  | 空间 | 1533 safetensors，**4.27 TiB**（4370 GiB） |
  | 吞吐 | 每 rank ~20–25 img/s，总 ~165 img/s；max wall 60.4h |

- **失败几乎全在 anime**：anime fail 1,733,090（≈anime eligible 的 **10.5%**）、monet_real 仅 713、**gpic/synth 0 fail**。归因 = 8 rank×FW64 聚合并发打 Cloudflare R2 限流（**取图失败，非数据损坏**，与 burn-in 结论一致）；失败 key 在 `failed_keys.parquet`，将来想补可定向重编。35.77M 已够 stage256 训练。
- ⚠️ **收尾 barrier 卡死坑（已修，见 §5#8）**：encode 跑完后 job 在 `dist.barrier()` 永久 hang（rank 间完成时间差 ~11h > NCCL 8h 超时），导致 yaml 里 encode 之后的 QC 步从未执行（无 qc.json）、job 一直假 Running 48h 占着 8×H100。**QC 本就是独立单进程 CPU 脚本**，事后单独跑 `python scripts/vae_cache_qc.py --stage 256` 即补出全部产物并校验通过，随后 stop 挂死 job。

**stage 256 定向补编（patch，2026-06-22）= 覆盖率 95.38% → 99.998%**
- 动机：主跑 4.62% 失败全是 `fetch_empty`（anime R2 在 8×64=512 聚合并发下被 Cloudflare 限流；非死链/坏图——抽样 200 个失败 uid 全在 manifest 且 URL 现仍可取）。
- 做法：复用缓存脚本的**补编模式**（`scripts/vae_cache_flux2_moments.py` 加 `--only-uids failed_keys.parquet --only-domains anime,monet_real --run-tag patch`，低并发 `--fetch-workers 16`=8×16=128 聚合避限流、`--fetch-retries 6`）。`--run-tag patch` 给 part/簿记文件换名，**不覆盖主跑文件**（mountpoint-s3 不能覆盖）；done_uids 跨主跑+补编按 shard 前缀 glob 跳已缓存，幂等可 resume。`jobs/vae_cache_moments_256_patch.yaml`（8×H100，跑完自动 QC `--out-suffix _postpatch`）。
- **实测（QC `summary_postpatch.json`）**：

  | 指标 | 主跑 | 补编后 |
  |---|---|---|
  | ok_tensors | 35,765,484 | **37,498,574** |
  | coverage_vs_eligible | 95.38% | **99.998%** |
  | anime 恢复 | 89.5% | **100%（16,499,287/16,499,287，补编 anime fail=0）** |
  | 真实残留(eligible−ok) | 1,733,803 | **713（全 monet_real）** |
  | index_eq_tensors / readback | ✅ / 100/100 | ✅ / 100/100 |
  | safetensors 文件数 | 1533 | 1632（+99 patch part） |

- **128 并发实测 anime fail=0**（vs 主跑 512 的 10.5%）→ 证实失败是限流、低并发可全恢复。吞吐 ~450 img/s 聚合、每 rank ~56 img/s（GPU ~75% 空转）= **fetch-bound**：本任务零跨卡通信，吞吐 ∝ 总取图线程数 ÷ R2 延迟，**加 GPU 无益**（一个 8 卡节点 80 CPU 即可拉满并发）；提速靠加 `--fetch-workers`，但对补编不建议（高并发重引失败、损恢复率）。
- **残留 713（全 monet_real，0.0019%）补编后仍 fetch_empty**：monet 是 S3 `.tar` 顺序流式（非 R2、非限流），多半是 manifest 有 key 但 tar 内缺/坏 = 源数据问题，重试无效，占比可忽略未追。需要时按 `failed_keys_postpatch.parquet` 的 patch 残留定位具体 tar。
- ⚠️ **`summary_postpatch.json` 的 `failed`/`fail_rate`/`per_domain_fail` 字段会双计历史失败**（QC 汇总 `failed/` 下所有 part：主跑 1.73M anime+713 monet + 补编 713 monet，其中 anime 1.73M 已恢复）。**判残留以 `coverage_vs_eligible` 或 `eligible−ok_tensors`=713 为准，别看 `failed` 字段。**

> ⚠️ **本机当前有一个 live `train.py burnin256` 训练在跑（非本任务起的，占 1×H100 50GB）**；本任务所有验证/探针都做成 CPU-only 或轻量 GPU、未干扰它。生产缓存跑在 JCC 独立节点。

### 6.2 训练接入端到端验证（2026-06-22，JCC 1 GPU，全 PASS）

`jobs/latent_cache_smoke.yaml`（`evan-mot-latcache-smoke`，Succeeded）：

| 验证 | 结果 |
|---|---|
| **GPU 数值对拍**（缓存 moments vs 在线 VAE 重编码，16 张 anime，同设备 bf16，mode-path model-space） | max\|Δ\|=**0.0625**、mean rel err=**0.49%** → bf16+cuDNN 跨运行噪声量级（错图/错VAE 会是 O(1)），缓存忠实复现 VAE 几何 ✅ |
| **数据线** | latent cache 开启、索引 37,498,574 uid、producer 跳取图、按 uid 读 moments ✅ |
| **train.py 真训练步**（`use_latent_cache=true`，40 step，1 GPU） | loss **1.9966→1.6651** 单调下降、**无 NaN、无报错**、跑到 `done` ✅ |

→ **结论：stage256 缓存可直接用于正式训练，dataline→model moments 分支→loss/反向 完全跑通且数值正确。** 正式 run 只需在 config 设 `dataset.use_latent_cache=true`（先跑一次 `build_latent_cache_index.py`）。
CPU 组件单测见 §2.1 / `scripts/test_latent_cache.py`。

---

## 7. 未尽 / 后续

- stage 512 / 1024：yaml 已就绪，**按顺序逐个提交**（别同时三个，控 S3 写 + R2/tar 读压力）。barrier 坑已修（§5#8）→ 跑完 `torchrun` 会正常返回、yaml 里的 QC 自动接上、不再假 Running 卡死；万一仍缺 qc.json，单独跑 `vae_cache_qc.py --stage <S>` 补。
- **anime R2 限流致 ~10.5% fail（stage256 实测）**：若 512/1024 在意 anime 覆盖率，可考虑降 `--fetch-workers`（减聚合 QPS）或事后按 `failed_keys.parquet` 定向重编（低并发补漏）。256 的 35.77M 已够训，未补。
- anime VAE gate（线条/网点/透明重建）数据交付文档 §11.6 #7 仍待复核（与本缓存无关，但上生成塔前要看）。
- 若后续 `domain_mix` 再变（如 GPIC@1024 有了高清源），重跑对应 stage 自动按新范围缓存。
