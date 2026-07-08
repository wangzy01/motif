"""诊断脚手架（调试 hang / collective 时序用，默认全关、env 门控）。

本模块集中本轮 2×8 hang 调试期间塞进核心代码（networks/arch_mot.py、train.py）的诊断
代码，使核心前向/训练循环回到干净状态。**硬约束：诊断关闭（默认）时，调用这些函数/方法
对训练行为必须逐位无影响。**

两类内容：

1. 模型侧逐 phase trace（原 arch_mot.py 的 `_debug_trace` / `_debug_shape`）：
   由 env `MOT_DEBUG_TRACE_FILE` 门控。未设该 env → `_debug_trace` 直接 return，
   `_debug_shape` 只做纯字符串拼接（无副作用）。env 名（`MOT_DEBUG_TRACE_FILE`/`_RANK`/
   `_LOCAL_RANK`/`_STEP`/`_LAYER`）与原实现逐字一致。

2. 训练循环侧（原 train.py 的 `trace()` 嵌套函数、`batch_ready_barrier()`、faulthandler
   周期 dump 的 setup/teardown）：
   由 config `trainer.debug_step_trace` / `trainer.batch_ready_barrier` /
   `trainer.debug_faulthandler_dump_interval` 门控；这里封装为 `StepTraceDiagnostics`
   / `BatchReadyBarrier`，状态、env 写、文件名、超时语义均与原 main() 内闭包逐字保持。
"""

import faulthandler
import json
import os
import time

import torch


# --------------------------------------------------------------------------- #
# 模型侧逐 phase trace（迁自 networks/arch_mot.py，env 门控、默认关）
# --------------------------------------------------------------------------- #
_DEBUG_TRACE_FP = None
_DEBUG_TRACE_PATH = None


def _debug_shape(x: "torch.Tensor | None") -> str:
    if x is None:
        return "none"
    return "x".join(str(dim) for dim in x.shape)


def _debug_trace(phase: str, **fields) -> None:
    path = os.environ.get("MOT_DEBUG_TRACE_FILE")
    if not path:
        return
    try:
        global _DEBUG_TRACE_FP, _DEBUG_TRACE_PATH
        if _DEBUG_TRACE_FP is None or _DEBUG_TRACE_PATH != path:
            if _DEBUG_TRACE_FP is not None:
                _DEBUG_TRACE_FP.close()
            _DEBUG_TRACE_PATH = path
            _DEBUG_TRACE_FP = open(path, "a", encoding="utf-8", buffering=1)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rank = os.environ.get("MOT_DEBUG_TRACE_RANK", os.environ.get("RANK", "?"))
        local_rank = os.environ.get("MOT_DEBUG_TRACE_LOCAL_RANK", os.environ.get("LOCAL_RANK", "?"))
        step = os.environ.get("MOT_DEBUG_TRACE_STEP", "?")
        layer = os.environ.get("MOT_DEBUG_TRACE_LAYER")
        if layer is not None and "layer" not in fields:
            fields = {"layer": layer, **fields}
        extras = " ".join(f"{k}={v}" for k, v in fields.items())
        _DEBUG_TRACE_FP.write(
            f"{now} rank={rank} local_rank={local_rank} step={step} "
            f"model_phase={phase} pid={os.getpid()} {extras}\n"
        )
    except Exception:
        # Debug tracing must never perturb training behavior.
        return


# --------------------------------------------------------------------------- #
# 训练循环侧诊断（迁自 train.py main()：step trace + faulthandler dump）
# --------------------------------------------------------------------------- #
class StepTraceDiagnostics:
    """逐 step phase trace + faulthandler 周期 dump 的封装（迁自 train.py main() 内闭包）。

    `setup()` 完整复刻原 main() 里 trace_fp/trace_dir 初始化、MOT_DEBUG_TRACE_* env 写/清、
    `debug_trace_dump_interval` 与独立 `debug_faulthandler_dump_interval` 的 faulthandler
    周期 dump 装载。`trace(step, phase, **fields)` 等价原嵌套 `trace`。teardown 相关由
    `cancel_faulthandler()` / `close()` 提供，与原 main() 收尾逐字一致。

    诊断关闭（debug_step_trace=false 且两个 dump_interval<=0）时：trace_fp is None →
    `trace()` 直接 return，且 setup 会 pop 掉所有 MOT_DEBUG_TRACE_* env（与原实现一致），
    故对训练行为零影响。
    """

    def __init__(self, config, out_dir, rank, local_rank, world_size, log):
        self._config = config
        self._out_dir = out_dir
        self._rank = rank
        self._local_rank = local_rank
        self._world_size = world_size
        self._log = log
        self.trace_fp = None
        self.trace_dir = None
        self.trace_enabled = False

    def setup(self):
        config = self._config
        out_dir = self._out_dir
        rank = self._rank
        local_rank = self._local_rank
        log = self._log

        self.trace_enabled = bool(config.trainer.get("debug_step_trace", False))
        if self.trace_enabled:
            trace_dir_cfg = config.trainer.get("debug_step_trace_dir", None)
            self.trace_dir = str(trace_dir_cfg) if trace_dir_cfg else os.path.join(out_dir, "step_trace")
            os.makedirs(self.trace_dir, exist_ok=True)
            host = os.uname().nodename.split(".")[0]
            trace_path = os.path.join(self.trace_dir, f"rank{rank:05d}_local{local_rank}_{host}_pid{os.getpid()}.log")
            self.trace_fp = open(trace_path, "a", encoding="utf-8", buffering=1)
            os.environ["MOT_DEBUG_TRACE_FILE"] = trace_path
            os.environ["MOT_DEBUG_TRACE_RANK"] = str(rank)
            os.environ["MOT_DEBUG_TRACE_LOCAL_RANK"] = str(local_rank)
            dump_interval = float(config.trainer.get("debug_trace_dump_interval", 0) or 0)
            if dump_interval > 0:
                faulthandler.dump_traceback_later(dump_interval, repeat=True, file=self.trace_fp)
        else:
            for env_key in (
                "MOT_DEBUG_TRACE_FILE",
                "MOT_DEBUG_TRACE_RANK",
                "MOT_DEBUG_TRACE_LOCAL_RANK",
                "MOT_DEBUG_TRACE_STEP",
                "MOT_DEBUG_TRACE_LAYER",
            ):
                os.environ.pop(env_key, None)

        # 独立 faulthandler 周期 dump：不设 MOT_DEBUG_TRACE_*（→ 不触发 arch_mot 每层 os.environ 写、
        # 不扰动 FSDP collective 时序），专用于诊断 dataline/collective 静默卡死。每 interval 秒把所有
        # 线程栈 dump 到 per-rank 文件；连续两次相同栈即卡点。默认 0=关。
        fh_interval = float(config.trainer.get("debug_faulthandler_dump_interval", 0) or 0)
        if fh_interval > 0:
            fh_dir_cfg = config.trainer.get("debug_step_trace_dir", None)
            fh_dir = str(fh_dir_cfg) if fh_dir_cfg else os.path.join(out_dir, "faulthandler")
            os.makedirs(fh_dir, exist_ok=True)
            _fh_host = os.uname().nodename.split(".")[0]
            _fh_path = os.path.join(fh_dir, f"rank{rank:05d}_local{local_rank}_{_fh_host}_pid{os.getpid()}.fh")
            _fh_fp = open(_fh_path, "a", encoding="utf-8", buffering=1)
            faulthandler.dump_traceback_later(fh_interval, repeat=True, file=_fh_fp)
            log(f"[train] faulthandler periodic dump every {fh_interval}s -> {fh_dir}")

    def trace(self, step, phase, **fields):
        if self.trace_fp is None:
            return
        os.environ["MOT_DEBUG_TRACE_STEP"] = str(step)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        extras = " ".join(f"{k}={v}" for k, v in fields.items())
        self.trace_fp.write(
            f"{now} rank={self._rank} local_rank={self._local_rank} step={step} phase={phase} {extras}\n"
        )
        self.trace_fp.flush()

    def cancel_faulthandler(self):
        faulthandler.cancel_dump_traceback_later()

    def flush(self):
        if self.trace_fp is not None:
            self.trace_fp.flush()

    def close(self):
        if self.trace_fp is not None:
            self.trace_fp.close()


class BatchReadyBarrier:
    """跨 rank 文件系统 barrier（迁自 train.py main() 内 batch_ready_barrier 闭包）。

    每 step 各 rank 落一个 marker 文件并轮询直到全员就位（或超时），用于把 dataline 不齐
    的 hang 暴露成超时报错。enabled 语义、marker 文件名、payload、超时/轮询间隔与原实现
    逐字一致。超时时通过传入的 `tracer.trace` 记一条 phase，再抛 TimeoutError。

    关闭（batch_ready_barrier=false 或非多卡）时：`barrier()` 直接返回 0.0，与原实现一致。
    """

    def __init__(self, config, out_dir, tracer, rank, local_rank, world_size, is_dist, log):
        self._config = config
        self._out_dir = out_dir
        self._tracer = tracer
        self._rank = rank
        self._local_rank = local_rank
        self._world_size = world_size
        self._is_dist = is_dist
        self._log = log
        self.enabled = False
        self.root = None
        self.timeout_s = 300.0

    def setup(self):
        config = self._config
        out_dir = self._out_dir
        rank = self._rank
        world_size = self._world_size
        is_dist = self._is_dist
        log = self._log

        self.enabled = (
            bool(config.trainer.get("batch_ready_barrier", False))
            and is_dist
            and world_size > 1
        )
        self.timeout_s = float(config.trainer.get("batch_ready_barrier_timeout_s", 300) or 300)
        if self.enabled:
            barrier_dir_cfg = config.trainer.get("batch_ready_barrier_dir", None)
            if barrier_dir_cfg:
                barrier_base = str(barrier_dir_cfg)
            elif self._tracer.trace_dir:
                barrier_base = os.path.join(self._tracer.trace_dir, "batch_ready")
            else:
                barrier_base = os.path.join(out_dir, "batch_ready")
            run_id = str(config.trainer.get("batch_ready_barrier_run_id", "") or "").strip()
            if not run_id:
                run_id = os.path.basename(out_dir.rstrip(os.sep)) or "run"
            self.root = os.path.join(barrier_base, run_id)
            os.makedirs(self.root, exist_ok=True)
            log(
                f"[train] batch_ready_barrier=true root={self.root} "
                f"timeout={self.timeout_s:.1f}s"
            )

    def barrier(self, step, batch_size):
        if not self.enabled:
            return 0.0
        rank = self._rank
        local_rank = self._local_rank
        world_size = self._world_size
        step_dir = os.path.join(self.root, f"step_{int(step):08d}")
        os.makedirs(step_dir, exist_ok=True)
        marker = os.path.join(step_dir, f"rank_{rank:05d}.ready")
        tmp = marker + f".tmp.{os.getpid()}"
        payload = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "step": int(step),
            "batch_size": batch_size,
            "host": os.uname().nodename.split(".")[0],
            "pid": os.getpid(),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
            f.write("\n")
        os.replace(tmp, marker)

        expected = {f"rank_{r:05d}.ready" for r in range(world_size)}
        t0 = time.monotonic()
        deadline = t0 + self.timeout_s
        missing = expected
        while time.monotonic() < deadline:
            try:
                present = {name for name in os.listdir(step_dir) if name in expected}
            except FileNotFoundError:
                present = set()
            missing = expected - present
            if not missing:
                return time.monotonic() - t0
            time.sleep(0.2)

        missing_ranks = [int(name.split("_")[1].split(".")[0]) for name in sorted(missing)]
        try:
            present = sorted(name for name in os.listdir(step_dir) if name in expected)
        except FileNotFoundError:
            present = []
        self._tracer.trace(
            step,
            "batch_ready_barrier_timeout",
            timeout_s=self.timeout_s,
            missing=",".join(str(r) for r in missing_ranks),
            present_count=len(present),
            step_dir=step_dir,
        )
        raise TimeoutError(
            f"batch_ready_barrier timeout after {self.timeout_s:.1f}s at step {step}: "
            f"missing ranks {missing_ranks}; present_count={len(present)}/{world_size}; dir={step_dir}"
        )
