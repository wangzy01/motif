# Phase-1 Pilot 实现委托书(for gpt-5.6 Sol,2026-07-09)

你的任务:为 OFFLINE_ARB_V1 重构的 Phase-1(小型 source/index 可行性验证,stop/go gate)**编写全部脚本与 JCC job yaml**。设计权威见 `docs/OFFLINE_ARB_V1_SPEC.md`;8 份侦察证据 + 缺口审查在 `docs/offline_arb_recon/*.json`(务必先读 critique.json 与 monet_real/synthetic/anime/gpic/textatlas 五份,里面有全部路径/schema/行数/魔数实测证据)。

## 交付物

1. `scripts/offline_arb/phase1_meta_validate.py` —— 元数据与身份验证(下述 Task A/D/G)
2. `scripts/offline_arb/phase1_io_probe.py` —— IO 可行性实测(下述 Task B/C/E/F)
3. `jobs/offline_arb_phase1_meta.yaml` —— 跑 1(CPU-only)
4. `jobs/offline_arb_phase1_io.yaml` —— 跑 2(CPU-only)
5. `scripts/offline_arb/README.md` —— 每个 Task 的目的/判定阈值/输出说明

两个 job 互相独立可并行。所有输出写到
`/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_phase1_${RUN_TAG}/`,
每个 Task 产出:机器可读 `taskX_result.json` + 追加到 `pilot_summary.txt` 的一行人读 verdict(`task=A verdict=PASS|FAIL|WARN key_numbers=...`)。

## 七个 Task 规格(源自 critique.json 的 needs_jcc,已合并去重)

**Task A — 五域 manifest w/h vs 真实解码一致性 + EXIF 普查(最高优先)**
每域分层抽样 2000 条(按 min(w,h) 分位数分层,确保覆盖 256/512 边界附近):
- monet_real/synthetic:按抽中 key 分组到所属 tar shard,每 shard 用 `training/dataline/image_fetch.py` 的 `stream_tar_images` 顺序扫出目标 keys(复用现成函数,勿重写);为控制成本,每域先抽 shard(≤8 个)再在 shard 内抽 key。
- anime:HTTP GET(复用 image_fetch 的 anime fetch 函数,并发 ≤16)。
- gpic:直接按路径 open 独立文件。
- textatlas:pyarrow 按 row-group 读取抽中行。
对每张图:`PIL.Image.open(BytesIO(raw))` 取 `.size` 与 `getexif().get(274, 1)`,**不做** exif_transpose;记录 (manifest_w,h) vs (decoded_w,h) 一致/转置一致(w,h 互换恰等=方向嫌疑)/不一致,EXIF orientation tag ≠1 的占比。
判定:一致率 ≥99.5% 且 orientation≠1 占比 <0.5% → PASS;转置型不一致集中出现 → FAIL(须在 catalog 建阶段插入解码校正,报 Evan 拍板)。

**Task D — uid 全量唯一性 + monet_real∩synthetic 跨域碰撞**
用 pyarrow 窄列裁剪(只读 uid 列,batch 流式)读 monet rich_v1 全部 256 shard(40M)+ synthetic manifest 全量(13.8M):对每个 uid 取 md5 的前 16 字节拆成 2×int64,numpy 拼接后 np.unique 查重(内存 ~1.7GB,可行);同法各自域内查 anime/gpic/textatlas。输出:域内重复数、monet∩synthetic 交集数。判定:全部为 0 → PASS。

**Task G — textatlas 全量 row-group 普查**
600 个 shard 全部 `ParquetFile(p).metadata` footer 遍历(不 read_table):row-group 行数分布、每 rg 的 file_offset/size 提取成本。输出分布表 + 构建 item→row_group 索引的外推成本。判定:分布均匀(绝大多数 ~100 行/组)且 footer 可完整提取 → PASS。

**Task B — mountpoint-s3 range-read 真实语义(fetch 后端选型的决定性数据)**
选 monet cc12m(~2.2GB)与 coyo(~5.5GB)各 1 个 tar:
1. 冷读文件末端 1MB(seek 到 size-2MB 处 read 1MB),计时;对比顺序读开头 1MB。若末端小读耗时 ≈ 开头小读(而非 ≈ 整文件拉取时间)→ 真 range 语义。
2. 同一文件内 500 次随机 offset 的 64KB/1MB 读,P50/P90/P99 延迟。
3. 顺序全量读该 tar 一遍(`cat >/dev/null` 或分块 read),得顺序吞吐 MB/s 基线。
判定输出:range_semantics=true/false + 随机小读延迟表 + 顺序吞吐。**注意**:每步 `timeout` 包裹,冷/热区分(同一 offset 读两次,第二次为热)。

**Task C — tar offset 索引原型与成本外推**
对 monet 6 子源 + synthetic 3 子源各选 1 个 tar shard:顺序流式解析 tar header(512B 块,跳过内容体,只记 name/offset/size;可用 tarfile 'r|' 的 members 或手写 header 解析),产出 (key→offset,size) 索引;记录:扫描耗时、索引条数、索引体积(parquet 存储)。外推:全 corpus(monet 6源×90目录×每目录 shard 数,ls 计数;synthetic 全部 shard)总扫描时间与索引大小。若 Task B 证实真 range:另测"seek 跳跃式 header 扫描"(只读每个 header 512B + seek)与顺序扫的耗时对比。
判定:全量索引构建外推 ≤24h(单 64 核节点,可并行分片)→ PASS。

**Task E — anime CDN 规模化 GET 压测(自家 CDN,克制)**
从 anime manifest 抽 50k key,并发 **≤32**、带指数退避,GET 计时;统计 P50/P95/P99、429/403/5xx 比例、有效吞吐 img/s。判定:无限流(429<0.1%)且 32 并发吞吐 ≥100 img/s → PASS。

**Task F — gpic 独立文件随机 open 延迟**
随机 1000 个 gpic 图路径,open+read 全文件,冷读计时(P50/P95/P99 + 吞吐)。判定:P95 < 2s 且平均吞吐支撑 mix 份额 → PASS(阈值写进 README,可再议)。

## 硬约束(必须全部遵守)

- **所有数据源只读**;绝不写 /s3-mewtant-raw-datasets、/s3-pixai-test-uw2 等任何源;绝不碰 `/fsx/evan/mot_code/mot_pilot_v1_cleanup`(原工作树,直播训练)。
- 只在 `/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop` 下新建文件,且仅限 `scripts/offline_arb/` 与 `jobs/`;不改动任何既有文件。
- job yaml 模板对齐 `jobs/naifu_arb_revalidate_waitfix.yaml` 的既有风格:name 前缀 `evan-mot-crop-offlinearb-p1-`,image `471112743779.dkr.ecr.us-west-2.amazonaws.com/autobuild:py312-torch211-cu129`,gpuPerWorker 0,cpuPerWorker 64,memGBPerWorker 384,maxRetries 0,`set -euxo pipefail`,PY=/fsx/evan/mot_code/venv-gpu/bin/python,PYTHONPATH 指向 crop 仓库,каждый task 外层 `timeout`,总墙钟 ≤4h/job(超时的 task 记 INCOMPLETE 不算 job 失败,除非是 FAIL 判定)。
- **禁止任何周期性 faulthandler dump**(Py3.12 SIGSEGV 已实锤根因);PYTHONFAULTHANDLER=1 可以。
- Python 风格对齐仓库现有代码(中文注释、干货);每个脚本可 `--task A,B` 选择性跑、可 `--sample-n` 缩放。
- 写完自检:`python -m py_compile` 全部脚本;yaml 用 `python -c "import yaml,sys; yaml.safe_load(open(...))"` 验证;yaml 内嵌 bash 用 `bash -n` 验证(注意 YAML 块缩进剥离对内嵌 python heredoc 缩进的影响——参考 waitfix yaml 里 WAITSTAT_PY 的写法)。

## 你不需要做的

- 不要提交 job(由主线提交);不要跑任何重任务;不要动 archive/ 与 docs/(只读)。
- 完成后输出:创建的文件清单 + 每个 Task 的实现要点/自检结果 + 你认为的设计疑点(如有)。
