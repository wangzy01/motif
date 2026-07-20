# taridx r2:子源内 tar-range 分片(2026-07-10 晚)

r1 三分片(a/b/c)死因:单流带宽墙实测 ~732MB/s/节点(megalith 3.205TB/4378s 反推),
commoncatalog 25.9TB / laion 16.8TB 单子源即需 6-10h,而续跑粒度只有子源级 → 超时全损。

r2 方案:`phase3a_tar_index.py --tar-shard K:N`(子源排序 tar 列表按 i%N==K 切,产物
`<base>.tshardKofN.parquet`,verify 按「同 N 全片齐 = complete,whole 优先,混 N 不拼凑」),
10 个 scan job + 1 个终局 vfy,由调度器(scratchpad/taridx_r2_scheduler.sh)恒 ≤3 并行滚动提交。
预算 = 实测字节 ÷ 732MB/s × ~1.4。

## ⚠️ 运维纪律(审查 P1-1,必须遵守)

**换 N 重新分片、或对已有 tshard 的子源补扫 whole 之前,必须先删除该子源旧 parquet 产物
及对应 manifest 记录。** 原因:verify 的 whole-first 只是"选择"完成证据,不清理落选记录;
而 phase5 build_tar_rowgroup_map 消费манifest 里**全部** ok 记录,同 (domain,src_shard) 重复
key 会直接 raise——verify PASS 但 catalog 必崩。中期修法:phase5 复用 resolve_subsource_completion。

其余注意:vfy job exit 0 ≠ PASS(PENDING 也是 0),终态以 `_SUCCESS` 文件为准;
调度器对每片失败自动重提一次(resume 粒度=单 tshard 文件),第二次失败停机待人工。
