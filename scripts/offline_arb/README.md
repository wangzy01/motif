# OFFLINE_ARB_V1 Phase-1 pilot

本目录只实现 `docs/offline_arb_recon/PHASE1_PILOT_BRIEF.md` 中的 Phase-1 stop/go
探针，不构建正式 catalog，也不启动训练。两个脚本对全部数据源严格只读；唯一写入位置是显式传入的
`--output-dir`。

## 运行入口与共同输出

正式运行由两个 CPU-only JCC job 完成：

- `jobs/offline_arb_phase1_meta.yaml`：Task A、D、G。
- `jobs/offline_arb_phase1_io.yaml`：Task B、C、E、F。

脚本也支持单独或逗号分隔选择 Task。以下缩放诊断仍会读取真实源，**只能在 CPU JCC 节点运行，
不能在登录机/本机执行**：

```bash
PY=/fsx/evan/mot_code/venv-gpu/bin/python
export PYTHONPATH=/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop
RUN_TAG=manual_diag_$(date -u +%Y%m%dT%H%M%SZ)
OUT=/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_phase1_${RUN_TAG}
$PY scripts/offline_arb/phase1_meta_validate.py \
  --task A,D,G --output-dir "$OUT" --sample-n 20
$PY scripts/offline_arb/phase1_io_probe.py \
  --task B,F --output-dir "$OUT" --sample-n 10
```

组合选择时脚本按安全依赖顺序执行（meta：D→G→A；IO：B→C→E→F），不按命令行书写顺序；正式
YAML 仍为每个 Task 启一个独立 Python 进程和独立外层 timeout。

`--sample-n` 的含义随 Task 不同：

| Task | 未传时的正式规模 | 传入后的缩放语义 |
|---|---:|---|
| A | 每域 2,000 | 每域样本数 |
| D | 全量 | 每域最多哈希的行数；仅诊断，不能证明全量唯一 |
| G | 600 个源 parquet | 最多扫描的源 parquet 数；仅诊断 |
| B | 每 tar、每 read size 500 个 offset | offset 数 |
| C | 每个子源 1 个 tar | 每子源 tar 数，安全上限 8 |
| E | 50,000 个 key | GET key 数；少于正式规模只给诊断结论 |
| F | 1,000 个路径 | 随机路径数；少于正式规模只给诊断结论 |

每个 Task 原子写 `taskX_result.json`，并在文件锁保护下向 `pilot_summary.txt` 追加一行：

```text
task=A verdict=PASS key_numbers=domains_pass=5,decoded=10000
```

JSON 公共字段包含 `schema_version/task/verdict/host/seed/started_at/completed_at/duration_seconds`。
延迟统一输出毫秒，吞吐中的 `MiB` 统一按 `2**20` 字节计算，分位数使用 NumPy linear
percentile。正常业务结论是 `PASS/FAIL/WARN`；内部 I/O 子步骤或 job 外层 timeout 无法完成时为
`INCOMPLETE`。`INCOMPLETE` 不让整个 job 失败，业务 `FAIL` 会使 job 最终非零退出。委托书的摘要示例
只列出前三种 verdict，但同一委托书又明确要求 timeout 记 `INCOMPLETE`，这里以后者为准。

所有抽样使用固定种子 `20260709`。所谓 cold 均只是“本进程第一次触碰该 offset/file”的代理口径；脚本
不会也不应通过特权操作清理 kernel、mountpoint-s3 或 CDN cache。

## Task A：五域宽高/EXIF 一致性

目的：验证离线 catalog 是否能信任 manifest `width/height`，以及是否必须在 catalog 构建阶段增加
EXIF 方向校正。

正式规模为五域各 2,000 条。选择方法为：

1. `monet_real`/`synthetic` 先从物理 tar 全集做固定种子 hash 抽样，再在这些 tar 对应的
   manifest 行中分层。MONET 选择
   6 个子源各一个 tar，并给体量最大的 coyo/laion 各加一个，总数硬限制为 8；synthetic 三个子源
   各最多两个，总数 6。这里的 shard 指物理 tar，不是 rich manifest parquet。
2. 其余域先跨多个 manifest/source shard 取有界候选池。
3. 每个域给短边最接近 256 和 512 的样本各预留 10%，其余样本按候选池
   `min(width,height)` 的十个分位层轮询填充。重复分位边（例如几乎全方图的 synthetic）自动回填。

真实取图路径严格按现有生产接口：

- MONET/synthetic：按 tar 分组后直接复用
  `training.dataline.image_fetch.stream_tar_images`，每个 tar 只顺序扫一次。
- anime：直接复用 `fetch_url_bytes(..., cache_dir=None)`，线程并发硬上限 16，不写 anime cache。
- GPIC：复用 `read_gpic_bytes`，按独立文件读取并探测混合扩展名。
- TextAtlas：sidecar 的全局行号映射到同名源 parquet 的真实 row group，只读取命中的
  `image/image_path` row group，并校验两侧 `image_path`。

每张图都用以下口径检查，结果逐条保存在 `samples`：

```python
with Image.open(BytesIO(raw)) as image:
    decoded_size = image.size
    orientation = image.getexif().get(274, 1)
```

不调用 `ImageOps.exif_transpose`，也不先 `convert`。先判断 decoded `(w,h)` 与 manifest 完全相等；
不等时再判断是否恰为 `(h,w)`，因此方图不会误计为 transposed。其余为 `other_mismatch`。

逐域门槛：

- exact rate `>= 99.5%`；分母为成功打开并读到尺寸的图片。
- orientation `!= 1` rate `< 0.5%`；缺 tag 按默认 1 处理。
- “转置型集中”暂定为：transposed 至少 3 张、至少占成功 decode 的 0.1%，且占全部 mismatch
  不少于 50%；命中即 FAIL。
- 成功 decode 只要不足请求量 100% 就至少 WARN，避免少量成功样本假绿；已有硬阈值失败仍优先 FAIL。
- 小于正式 2,000/域的缩放运行即使数值达标也只能 WARN。

五域必须逐域通过，不能用全域汇总率掩盖单域问题。

## Task D：稳定 UID 全量唯一性

目的：验证五域域内唯一，并重点检查共享 `monet:` 前缀的
`monet_real ∩ synthetic`。

正式运行只读 UID 窄列、按 Arrow batch 流式处理：

- monet rich_v1 256 shard，40,000,000 行。
- synthetic_v1 64 shard，13,808,747 行。
- GPIC rich_v1 256 shard，20,000,000 行。
- anime 使用含字面 `uid` 列的 thin manifest 64 shard，16,499,287 行。
- TextAtlas 使用全部 10 subset/600 sidecar（含 CoverBook），5,398,826 行；这比当前训练用
  595 shard 的范围更强。

每个 UID 取 MD5 的完整 16 字节并拆成 `2 x int64`。连续二维数组 view 为单个 `V16` 后再
`np.unique`，不会错误地把两个 int64 flatten。输出域内 unique 数、重复 group 数、重复额外行数、
最大 multiplicity、null/empty 数；MONET 和 synthetic 的 unique digest 再做 `np.intersect1d`。

正式 PASS 要求：全部五域重复额外行数为 0、null/empty 为 0、manifest 文件数/footer 行数与已知
口径完全一致、读取行数完整，并且 MONET 与 synthetic 交集为 0。`--sample-n` 只跑前 N 行/域，
即使未发现重复也为 WARN。MD5 理论碰撞概率极低，但命中 digest 时仍应在后续正式修复任务中回查
原 UID，不能直接把它当成业务 UID 相等。

## Task G：TextAtlas row-group footer 普查

目的：确认 TextAtlas 源图 parquet 的 row-group 布局及 item→row-group 索引成本。这里扫描的是
`/s3/evan/textatlas5m/*/*.parquet` 全部 10 subset/600 文件，不是 sidecar，也不是只含训练
9 subset 的 595 文件。

实现只构造 `ParquetFile` 并遍历 `.metadata`，绝不 `read_table/read_row_group`。对每个 row group
记录行数、row start，以及 `image.bytes` column chunk 的：

- 原始 `file_offset`。
- `total_compressed_size`。
- `dictionary_page_offset` / `data_page_offset`。
- 由两个 page offset 的最小有效值推导的 `range_start/range_end`。

当前 TextAtlas/PyArrow 实物中的 `file_offset` 呈 chunk 末端语义，不能直接当 range 起点，因此上述原始值与推导值
都会保留。`row_groups` 给出逐 RG 机器可读记录，`per_shard` 给出 footer open/metadata extract 成本，
同时输出完整 histogram 与分位数。

当前工程化 PASS 口径为：600/600 footer 完整、offset/size 提取率 100%、至少 98% 的 row group
恰为 100 行、所有非末尾 row group 恰为 100 行。98% 考虑了每个 shard 最后一个不足 100 行的
正常尾组。索引成本同时给出：catalog 已有 row id 时每 item 新增 `(rg_id:uint16,row_in_rg:uint16)`
的 4 字节模型、独立映射 16 字节模型，以及每 RG 32 字节 locator 模型；均为未压缩 raw 估算，不含
Parquet writer 时间。

## Task B：mountpoint-s3 range-read 语义

目的：判断 tar 的随机小读是否是真 range，而不是隐式整对象拉取。代表文件是 MONET
`cc12m/0-0/000000.tar` 和 `coyo/0-0/000000.tar`；缺失时回退为该子源排序首个 tar。

每个 tar 依次执行：

1. `seek(size-2MiB)` 后读 1MiB，并立即重复同 offset（first-touch/hot）。
2. 从头读 1MiB，并立即重复同 offset。
3. 对 64KiB 与 1MiB 两种 read size 各做 500 个 4KiB 对齐随机 offset，每个 offset 连读两次，
   分别输出 cold/hot P50/P90/P99。
4. 8MiB chunk 顺序读完整 tar，核对总字节并输出 MiB/s。

endpoint、每个随机 size 组、顺序全读分别在可 kill 的子进程内执行，除整个 Task 的 job timeout 外
还有 `--step-timeout-s`。内部 step timeout 为 INCOMPLETE，不会误判 `range_semantics=false`。

委托书没有量化“末端约等于开头”，当前暂定每个 tar 同时满足：

- `tail_cold / head_cold <= 5`。
- `tail_cold / full_sequential_time <= 0.25`。

两个 tar 都满足才 `range_semantics=true/PASS`；任一个明确不满足为 FAIL。这个 verdict 表示
“range 后端可行/不可行”的选型证据，不表示数据本身坏。

## Task C：tar offset 索引原型

目的：实测索引构建时间、体积，并外推 64 核单节点全 corpus 成本。

默认从 MONET 六个子源与 synthetic 三个子源各选择排序首个 tar，共九个。顺序实现使用
`tarfile.open(path, "r|")`，因此跳过 member 时仍实际消费 payload；遍历全部 member，只把训练需要
的 `.jpg` 写入 zstd Parquet：

```text
key, tar_path, member_name, header_offset, data_offset, size, domain, sub_source
```

range locator 使用 payload `data_offset`，不是 header offset。索引写到
`taskC_indexes/*.parquet`，并记录条数、重复 key、物理字节数与 bytes/row。

脚本还实际枚举各子源的 group 目录、tar 数和总字节。顺序 payload 扫描按总字节缩放；seek-jump
header 成本、Arrow materialize/Parquet 写盘以及索引体积按 tar 数缩放：

```text
payload_cost = corpus_total_tar_bytes / sampled_tar_bytes * sampled_payload_scan
header_or_index_cost = corpus_tar_count / sampled_tar_count * sampled_header_or_index_cost
```

这仍隐含“每 tar 的 member/image 数近似同分布”的假设；结果同时保留“全部成本都按字节缩放”的旁路
估计，便于发现两种模型分歧。64 核墙钟采用
`max(total_worker_seconds/64, longest_sample_seconds)`，门槛 `<=24h`。这是理想并行模型，未计
mount 带宽饱和，结果中会明确保留 serial worker time。

同一 IO job 中 B 先运行且会完整顺序读 cc12m/coyo 代表 tar；C 会显式排除 B 已触碰的 tar，再取
排序下一个样本，避免把明显热读的构建成本外推到全 corpus。索引总构建计时覆盖 tar header/payload
遍历、Python locator 收集、Arrow table materialize、Parquet 写盘和原子替换，不只计 tar 扫描。

若同一 `--output-dir` 中 Task B 已明确 PASS 且 `range_semantics=true`，C 还用随机模式
`tarfile.open(path, "r:")` 做 seek-jump header 扫，并要求 image count 与 locator digest 和顺序结果
完全一致；独跑 C 时不会搜索或误用其他 run 的“最新 B”。seek-jump 的全量时间外推包含 header
扫描和由顺序原型测得的索引写盘时间。24h gate 在顺序与可用 seek-jump 两者中选择实测外推更快的
方法，不会因为 seek-jump 反而较慢而否决原本可行的顺序构建。为减少同 tar cache 污染，B=true
时 C 对每个样本先跑 seek-jump first-touch，再跑顺序全读；两者的 cache 状态仍不是特权清空后的真冷态。

## Task E：anime CDN GET 压测

目的：验证 `img.metanomaly.co` 的 32 并发持续取图能力。权威源为 rich 17-shard manifest 的
`key/s3key`，`s3key` 已是完整 URL，不重拼域名。抽样按文件行数分配配额，并在每个文件随机选择
最少数量的 row group。

正式参数为 50,000 key、并发硬上限 32、每 key 最多 3 次尝试、单次 timeout 20 秒。探针不用现有
线性退避 fetch 函数，而使用 `0.3 * 2**attempt` 的指数退避、小幅确定性 jitter，并在 429 提供
`Retry-After` 时尊重它（上限 10 秒）。不传 cache，不写 `/local`。

输出全部 key、成功 key、失败 key 三套端到端延迟（包含重试/退避）的 P50/P95/P99、success img/s、MiB/s、attempt/final
状态码计数，以及 429/403/5xx 对全部 attempt 的比例。PASS 门槛：

- 32 并发。
- 429 attempt rate `<0.1%`。
- 成功吞吐 `>=100 img/s`。

403/5xx 必须报告，但委托书没有给 fail 门槛；当前任一达到 1% 时在主 gate 通过的前提下给 WARN。
少于 50k 或并发不为 32 的缩放运行也只能 WARN。

## Task F：GPIC 独立文件随机 open

目的：量化 GPIC 混合 jpg/png 独立文件的 first-touch open+全读延迟。样本在全部 256 个 rich
manifest 间按行数分配，默认 1,000 个。rich manifest 不含扩展名，所以正式计时前按
`{src_root}/parquets/gpic_train_{src_shard}.parquet` 窄列读取 `key/s3key`，用权威 s3key join
扩展名；这一过程不 stat 图片文件。路径解析耗时单独报告，不混入 payload 延迟。payload 测量为串行
`open + read all + close`，输出 P50/P95/P99、img/s、MiB/s、成功/错误数。

PASS 的明确门槛是 P95 `<2s`。委托书没有给“吞吐支撑 mix 份额”的总训练需求，当前临时采用：
stage256 最大 GPIC mix 25.9% × 100 global img/s，即单流 `>=25.9 img/s`。同时报告理想
64-stream img/s 与按 0.259 反推的最大 global img/s，方便 Evan 用真实训练需求替换临时门槛。
样本不足 1,000 不能 PASS。

## Runbook 注意项

- mountpoint-s3(FUSE)读卡死会进入不可中断 D 状态,SIGKILL/`timeout --kill-after` 都打不断:
  job 墙钟显著超过声明预算时,先查节点是否有 D 态进程,不能只等脚本自身超时兜底。
- IO job 的非零退出 ≠ infra 故障:先看各 `task*_result.json` 的 `verdict`/`reason` 区分
  真错误(FAIL)与决定性探测(`DECISIVE_FALSE`,见下第 4 条)。

## 尚待拍板的设计口径

这些疑点不会被代码隐藏，正式扩大前建议 Evan 明确：

1. A 的“边界附近”和“转置集中”原文无数值；当前分别采用 10%/边界与
   `>=3、>=0.1% decode、>=50% mismatch`。tar 域先选最多 8 个物理 shard 后再分层，不等同于
   全域严格分层；方图也天然无法凭 w/h 区分转置。
2. TextAtlas A 使用当前训练 9 subset/595 shard，D/G 使用更强的全部 10 subset/600 shard；
   委托书没有完全消歧。侦察 JSON 中 `assets_configs.json` 的 4,765,434 漏算了
   StyledTextSynth 425,826 行；footer 交叉证据支持 active=5,191,260、all=5,398,826。
3. G 的“绝大多数约 100 行”没有数值，当前定为 exact-100 `>=98%` 且非尾组零异常；正式
   item→RG schema 选 4/16 字节哪种也尚未冻结。
4. B 的“末端约等于开头”原文未量化(当前用 5x/25%)；first-touch 无法保证系统 cache 真冷。
   【已裁定 2026-07-09 review P1】range=false 记 `DECISIVE_FALSE`(成功且决定性的探测,
   仅否决 range 后端,fetch 三选一退化为二选一),不再计入 job 级 FAIL;FAIL 只留给真错误。
5. C 的 64 核外推是假设理想 shard 并行，未实测 mountpoint-s3 在 64 路下的带宽饱和；“key”
   当前只索引训练消费的 `.jpg`，不索引 `.json/.pth`。
6. E 的 429 比例分母可能选择“全部 attempt”或“仅 HTTP response”；当前主门槛用全部 attempt，
   两类原始计数均落盘。403/5xx 原文只有报告要求，没有失败阈值。
7. F 的总体训练 img/s 和并发口径未给，25.9 img/s 只是可配置临时门槛，需用真实训练供给需求替换。
8. 两个 job 独立提交时默认各自产生 RUN_TAG/输出目录。若提交方注入同一个 `RUN_TAG`，文件锁能
   保护 summary 追加且 Task 文件名不冲突，但共享 run 的生命周期与汇总责任仍需由提交方约定。
