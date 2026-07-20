#!/usr/bin/env python3
"""OFFLINE_ARB_V1 taridx r2:生成子源内 tar-range 分片 job yamls(2026-07-10 晚)。

背景:单流带宽墙 ~735MB/s/节点,commoncatalog 25.9TB / laion 16.8TB / coyo 7.4TB /
synthetic 7.2TB 拆成 10 个 ~1.5-3h 的 16cpu job,由主线保持 ≤3 并行滚动提交。
超时预算 = 实测字节 / 735MB/s × ~1.4 裕量。

用法:python scripts/offline_arb/gen_taridx_r2_yamls.py  → 写 jobs/offline_arb_phase3a_taridx_r2/*.yaml
"""
from __future__ import annotations

from pathlib import Path

REPO = Path("/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop")
OUTDIR = REPO / "jobs" / "offline_arb_phase3a_taridx_r2"

# (job 短名, subsources, tar_shard 或 None, scan 超时秒)
JOBS = [
    # synthetic 三子源 7.24TB ≈ 9850s → 14400s;沿用无分片路径(c 的续跑记录直接生效)
    ("c2", "monet_real__cc12m,monet_real__megalith10m,monet_real__diffusion_aesthetic_4k,"
           "synthetic__flux_klein,synthetic__flux_schnell,synthetic__z_image", None, 14400),
    # commoncatalog 25.91TB / 4 ≈ 6.48TB ≈ 8815s → 12600s
    ("a0", "monet_real__commoncatalog_cc_by", "0:4", 12600),
    ("a1", "monet_real__commoncatalog_cc_by", "1:4", 12600),
    ("a2", "monet_real__commoncatalog_cc_by", "2:4", 12600),
    ("a3", "monet_real__commoncatalog_cc_by", "3:4", 12600),
    # laion 16.76TB / 3 ≈ 5.59TB ≈ 7600s → 10800s
    ("l0", "monet_real__laion", "0:3", 10800),
    ("l1", "monet_real__laion", "1:3", 10800),
    ("l2", "monet_real__laion", "2:3", 10800),
    # coyo 7.37TB / 2 ≈ 3.69TB ≈ 5010s → 7800s
    ("y0", "monet_real__coyo", "0:2", 7800),
    ("y1", "monet_real__coyo", "1:2", 7800),
    # 终局 verify:全部 scan 完成后单独提交;PASS → touch _SUCCESS
    ("vfy", None, None, 2400),
]

TEMPLATE = """\
# OFFLINE_ARB_V1 Phase-3a taridx r2 分片 {tag}(2026-07-10 晚重排:tar-range 分片,3 并行滚动)。
# {desc}
name: evan-mot-crop-offlinearb-p3a-tidx2-{tag}
image: 471112743779.dkr.ecr.us-west-2.amazonaws.com/autobuild:py312-torch211-cu129
numWorkers: 1
gpuPerWorker: 0
cpuPerWorker: 16
memGBPerWorker: 96
maxRetries: 0
command: |
  set -euxo pipefail
  cd /fsx/evan/mot_code/mot_pilot_v1_cleanup_crop
  export PYTHONPATH=/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop
  export CUDA_VISIBLE_DEVICES=""
  export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ARROW_NUM_THREADS=1
  PY=/fsx/evan/mot_code/venv-gpu/bin/python
  ROOT=/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_tarindex_v1
  mkdir -p "$ROOT"
  echo "JOB=tidx2-{tag} host=$(hostname) started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$ROOT/run_info_r2_{tag}.txt"
  "$PY" -m py_compile scripts/offline_arb/phase3a_tar_index.py
  "$PY" scripts/offline_arb/phase3a_tar_index.py --self-test 2>&1 | tee "$ROOT/selftest_r2_{tag}.log"
{body}
"""

SCAN_BODY = """\
  RC=0
  set +e
  timeout --signal=TERM --kill-after=60s {scan_to}s \\
    "$PY" scripts/offline_arb/phase3a_tar_index.py \\
      --output-dir "$ROOT" --workers 14 --scan-mode stream \\
      --subsources {subs} \\
      {shard_arg}--job-tag r2{tag} \\
    2>&1 | tee "$ROOT/run_r2_{tag}.log"
  RC=${{PIPESTATUS[0]}}
  set -e
  echo "==== TARIDX-R2-{tag} scan exit_code=$RC ===="
  exit "$RC"
"""

VERIFY_BODY = """\
  VRC=0
  set +e
  timeout --signal=TERM --kill-after=60s {scan_to}s \\
    "$PY" scripts/offline_arb/phase3a_tar_index.py \\
      --output-dir "$ROOT" --verify-only \\
    2>&1 | tee "$ROOT/verify_r2_final.log"
  VRC=${{PIPESTATUS[0]}}
  set -e
  echo "==== TARIDX-R2 FINAL VERIFY exit_code=$VRC ===="
  exit "$VRC"
"""


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for tag, subs, shard, scan_to in JOBS:
        if subs is None:
            body = VERIFY_BODY.format(scan_to=scan_to)
            desc = "终局 verify-only:9 子源(整文件或齐分片)全绿 → PASS + touch _SUCCESS。"
        else:
            shard_arg = f"--tar-shard {shard} " if shard else ""
            body = SCAN_BODY.format(scan_to=scan_to, subs=subs, shard_arg=shard_arg, tag=tag)
            desc = f"subsources={subs} tar_shard={shard or '无'} scan_timeout={scan_to}s。"
        (OUTDIR / f"taridx_r2_{tag}.yaml").write_text(
            TEMPLATE.format(tag=tag, desc=desc, body=body), encoding="utf-8")
        print("wrote", OUTDIR / f"taridx_r2_{tag}.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
