# GPIC repack 提案(2026-07-12,前任 AI + gpt-5.6-sol ultra 对抗审计后输出)

> 交给当前接手 AI 的行动建议。背景:GPIC 域是 8×8 正式训练唯一剩余瓶颈;调线程/去 path-stat/加深 prefetch 均已被你自己的实验证死;直连 S3 被 IAM 挡 ≥2 天。

## 0. 新实测事实(本提案的地基)

- GPIC 全集 = 20M 张独立 jpg/png,**抽样实测 mean 131KB / median 102KB(~640×426),总量 ~2.68TB**(48 样本跨 4 个 row group,catalog gpic.parquet image_path 直接 stat)。
- GPU 节点 /local 总共 500GB(r4 probe 实测,空闲 ~250GB)→ 每节点 unique 工作集 ~1.3TB 装不下,**节点缓存路线死**(要守恒需 ≥21.3% hit,250GB≈1.9M 对象 vs ~12.4M views/node,Feistel 近均匀,达不到)。
- /fsx 空闲 ~22TB → 2.7TB 放得下。
- 正式期 GPIC 真实需求 = 25.9% mix × 2304/step ÷ 2.592s ≈ **230 img/s 全局 ≈ 30MB/s 聚合**。瓶颈是 20M 次 FUSE open/stat,不是带宽。

## 1. 方案:`offline_gpic_blob_v1` 独立 overlay

一次性把 20M 图按原始字节打包成 /fsx 大文件(无压缩 tar 或裸拼接 blob,~2–8GB/个,pilot 定),附:
- 定宽 runtime index:row_id → (blob_id:u32, offset:u64, len:u32, sha256:32B);20M 行 ≈ 960MB,可整体 stage 到 /local 并 mmap;Parquet 版只作审计产物。
- header 绑定:catalog_hash、gpic.parquet 的 name/rows/bytes/SHA/schema、row base/count、builder 代码+layout contract hash、全 blob/index 文件集逐文件 SHA。**不绑定 plan**(全 catalog 打包,plan 已经绑 catalog)。

### 红线
1. **不改 gpic.parquet**(catalog SHA 级联 → plan/animefreq/caption-sidecar/freeze 全重建)。overlay 用 row_id 对齐。
2. 输入必须是正式 catalog 的 20M 行已物化 image_path,**不扫 8000 个源目录**(母目录 101.2M 对象,覆盖关系未闭合,会打包大量非 catalog 图)。
3. loader gpic 适配器:overlay 缺失/校验失败 fail-fast,**禁止静默回退 FUSE**;热路径每次 range 读校验对象 SHA(30MB/s 下成本可忽略)。
4. monet reader 不能直接复用(它硬读 tar locator 三列,且每窗口 open→seek→read→close 无常驻 fd);GPIC 写独立适配器。
5. range 合并参数与 tar 路径分开:4MiB-gap/256MiB-run 是 S3 口径,Lustre 上对 131KB 读会 ~30× 读放大。首版**不合并**,并记录 actual_read/requested。
6. fd:首版 open-per-window;实测 open 占主导再上有界 LRU + os.pread(多线程严禁共享 seek+read 游标)。

### 构建
- 8×16cpu JCC,每 job 2.5M 图,按 source shard/path 排序读(顺序局部性),同时写 bucket-aligned blob。
- **纯读时间 = 694.4/r 小时**(r = 每 job img/s)。先跑 10–30 万行 pilot 扫并发 8/16/24/32/48/64(48 线程历史上反向缩放,别默认 64–128)+ blob 2/8/16GB + stripe 1/4。r≥116 才能谈 6h,现实排期按 10–32h。
- 完整性:有限重试退避、fstat-before/read-exact/fstat-after、失败回滚对象起始 offset、永久失败 fail closed、fsync→rename→fsync dir→原子 _SUCCESS;按确定性 partition/blob checkpoint 可恢复(最多丢一个未提交 blob,不要 caption-sidecar 的整域粒度)。
- verifier:顺序重读 2.7TB,逐 segment 重算 20M SHA,row_id 全覆盖/无重无洞/offset 不越界不重叠/blob 首尾闭合。
- 诚实口径:这是"产物自洽"证明;字节保真用分层原源 A/B(覆盖全部 bucket、早/晚 step、resume 边界、PNG/JPEG、每 blob 首尾、8000 shard 分层)+ 既有 4-hash 门,不再全量二读原源。
- **控制面复用 caption-sidecar 契约框架,不新造证据格式。**

## 2. 验证金字塔压缩(512a 已自然跑完 → 不可逆约束消失)

删掉长 2×8 性能代理轮。序列:
静态门 + 负例(截断/错 offset/错 SHA/重叠/半发布/catalog mismatch/fd eviction)
→ 小语义 A/B(20–50 步集成冒烟即可,不作性能授权)
→ 64 进程 plan-shaped data-only replay(0.5–1h)
→ **8×8 probation:独立 RUN_ID/checkpoint namespace,200–500 步,强制触发一次 checkpoint,预声明硬门自动早停**:
  - 对象 SHA/locator/loader 错误 = 0;GPIC 最慢 rank 容量 ≥1.111(目标 ≥1.25);4-step p99 供给 <~8.3s;读放大 ≤1.05;fd/RSS 无增长;checkpoint 前后无新 FSx 长尾;
  - 速度门 **2.50–2.55s/step**(2.592 是踩线值,要留启动+checkpoint 预算)。
→ 全绿:重建 freeze → formal preflight → 提正式 8×8。

## 3. 并行后手

- IAM 申请照提(直连只需 ~29 GET/s/node,权限回来大概率达标);若 repack 未发布而 IAM 先恢复,同形 8×8 probation 过了才允许切直连。
- repack 后滚动写 /local:默认不做(重新引入小文件 IO);只有 8×8 实测 FSx p99 不过门才加。

## 4. 时间线

现实关键路径 **~16–43h**(pilot 1–3h ∥ builder/reader/verifier 实现 4–8h → 全量构建 10–32h → 验证+replay+probation 2–4h);pilot 若证明 r≥116 且扩节点近线性,可压到 ~8–12h 进 8×8。

## 5. 已否决路线(勿再回头)

| 路线 | 死因 |
|---|---|
| 继续调 FUSE 线程/fd_stable/prefetch | 0.787×1.2=0.944<1;48 线程反向缩放;prefetch 只推迟持续赤字 |
| /local 子集缓存 | 250GB vs 每节点 1.3TB 工作集;hit 率达不到 21–29% |
| 逐图复制到 /fsx | 同样 20M 次源 open + 20M inode/MDS 压力 |
| 转存自有 S3 前缀 | 仍要 20M 次读+2.7TB 上传;小对象不解决元数据,大对象=repack 但留在远端(正式段 ~99.5M 次读、~13TB 重复流量) |
| 空等 IAM | ≥48h 且无确定性;只作并行后手 |
