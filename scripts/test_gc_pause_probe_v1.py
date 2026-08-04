#!/usr/bin/env python3
"""Test the GC pause probe added on 2026-08-04.

Context: train.py has carried three GC mitigations since the 2026-07-03/07-05 iter_s
creep investigations (gen2 threshold, freeze-after-N, periodic re-freeze).  Stage512b
enables all three, and the formal run still showed p90 = 3.0x p50 with the tail growing.
Nothing in the run had ever timed an actual collection, so "cluster interference", "GC
creep" and "warm-up transient" all fit the rank-0 metrics equally well and the question
could only be argued, not settled.

The probe's job is to replace that argument with a number.  A probe that silently
mis-times, double-counts, or never fires would be worse than none, because it would
produce confident wrong numbers -- hence this test.

The tracker is taken from the shipped train.py via ast, not reimplemented: a
reimplementation would pass while the shipped text was broken.  ast.unparse drops
comments but preserves semantics exactly, which is what is under test here.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train.py"
CONFIG = ROOT / "configs/stage512b_real_anchors_v1.yaml"
SCHEMA = ROOT / "training/config_schema.py"

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((bool(ok), label, detail))


src = TRAIN.read_text()

# ------------------------------------------------------------------ extract the tracker
cls_node = None
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.ClassDef) and node.name == "_GcPauseTracker":
        cls_node = node
if cls_node is None:
    print("FAIL: _GcPauseTracker not found in train.py -- the probe is gone")
    sys.exit(1)


class _FakeTime:
    """Deterministic clock: the probe must report the deltas we hand it, exactly."""

    def __init__(self) -> None:
        self.t = 0.0

    def perf_counter(self) -> float:
        return self.t


clock = _FakeTime()
ns: dict = {"time": clock}
exec(ast.unparse(cls_node), ns)  # noqa: S102 -- the subject under test is this text
Tracker = ns["_GcPauseTracker"]

# ----------------------------------------------------------------------- timing per gen
t = Tracker()
for gen, dur in ((2, 1.5), (0, 0.01), (2, 4.0), (1, 0.2), (0, 0.02)):
    t("start", {"generation": gen})
    clock.t += dur
    t("stop", {"generation": gen, "collected": 1, "uncollectable": 0})

n, sum_s, max_s = t.drain()
check(n == [2, 1, 2], "counts land in the right generation", f"{n}")
check(abs(sum_s[2] - 5.5) < 1e-9, "gen2 total is the sum of its pauses", f"{sum_s[2]}")
check(abs(max_s[2] - 4.0) < 1e-9, "gen2 max is the longest single pause", f"{max_s[2]}")
check(abs(sum_s[0] - 0.03) < 1e-9, "gen0 total", f"{sum_s[0]}")
check(abs(max_s[1] - 0.2) < 1e-9, "gen1 max", f"{max_s[1]}")

# drain must reset, or every window reports the run's all-time max and the "is it
# growing?" question -- the whole point -- becomes unanswerable.
n2, sum2, max2 = t.drain()
check(n2 == [0, 0, 0] and max2 == [0.0, 0.0, 0.0] and sum2 == [0.0, 0.0, 0.0],
      "drain resets the window", f"{n2} {sum2} {max2}")

# ------------------------------------------------------- a stop with no start is dropped
# gc.callbacks can be appended midway through a collection, so the first event a probe
# ever sees may be a "stop".  Timing it against a stale or absent t0 would invent a pause
# equal to the process uptime -- a spectacular false positive on the very first window.
t = Tracker()
t("stop", {"generation": 2})
n, sum_s, max_s = t.drain()
check(n == [0, 0, 0] and sum_s == [0.0, 0.0, 0.0],
      "a stop with no preceding start is ignored, not timed", f"{n} {sum_s}")

# Two starts in a row (should not happen -- CPython's gc is not reentrant -- but the
# probe must not produce a negative or absurd duration if it ever did).
t = Tracker()
t("start", {"generation": 2})
clock.t += 1.0
t("start", {"generation": 2})
clock.t += 2.0
t("stop", {"generation": 2})
_, sum_s, _ = t.drain()
check(sum_s[2] >= 0.0, "no negative durations under an unexpected event order",
      f"{sum_s[2]}")

# An out-of-range generation must not crash the training loop.  A probe that raises
# inside gc.callbacks would take the run down -- strictly worse than not measuring.
t = Tracker()
try:
    t("start", {"generation": 7})
    clock.t += 0.1
    t("stop", {"generation": 7})
    t("start", {})
    clock.t += 0.1
    t("stop", {})
    check(True, "out-of-range / missing generation does not raise")
except Exception as exc:  # noqa: BLE001
    check(False, "out-of-range / missing generation does not raise", repr(exc))

# --------------------------------------------------------------------- wiring, statically
# The collective is the deadlock risk: it must be issued by every rank, so the call has
# to sit OUTSIDE `if is_main:` and at the same level as the gpic helper that already
# follows this rule.
m_gpic = re.search(r"^(\s*)gpic_all_rank = \($", src, re.M)
m_gc = re.search(r"^(\s*)gc_all_rank = \($", src, re.M)
check(m_gpic is not None and m_gc is not None, "both all-rank helpers are called")
if m_gpic and m_gc:
    check(m_gpic.group(1) == m_gc.group(1),
          "gc_all_rank is at the same indent as gpic_all_rank (i.e. not inside is_main)",
          f"{len(m_gpic.group(1))} vs {len(m_gc.group(1))}")
    is_main_at = src.find("\n        if is_main:", m_gc.end())
    check(is_main_at != -1 and m_gc.end() < is_main_at,
          "gc_all_rank is computed before the is_main block it feeds")

check("record.update(gc_all_rank)" in src, "the metrics record picks the fields up")
check(re.search(r"dist\.all_reduce\(local, op=dist\.ReduceOp\.MAX\)", src) is not None,
      "the aggregation is an all_reduce MAX")
check("float(-gc.get_freeze_count())" in src and '"gc_allrank_freeze_count_min": -v[7]' in src,
      "freeze_count is negated so MAX yields the minimum across ranks")
check("data_wait_allrank_max_s" in src,
      "the cross-rank data_wait max is emitted -- the field whose absence caused the "
      "2026-08-04 misreading")

# Gated, default off: consistent with the three mitigations next to it, and it keeps the
# probe from changing behaviour for any config that does not ask for it.
check('config.trainer.get("gc_pause_probe", False)' in src,
      "the probe is gated and defaults to off")
gate_at = src.find('gc_pause_probe", False')
append_at = src.find("gc.callbacks.append(gc_tracker)")
check(gate_at != -1 and append_at != -1 and gate_at < append_at,
      "the callback is only registered when the gate is on")
check("gc_tracker is None" in src, "the helper returns nothing when the probe is off")

check('"gc_pause_probe": bool' in SCHEMA.read_text(),
      "config_schema registers gc_pause_probe (else it warns as unknown)")
check(re.search(r"^  gc_pause_probe: true$", CONFIG.read_text(), re.M) is not None,
      "the Stage512b config actually enables it")

failures = [(label, detail) for ok, label, detail in results if not ok]
for ok, label, detail in results:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
print(f"\n{len(results)} checks, {len(failures)} failures")
for label, detail in failures:
    print(f"  {label}: {detail}")
if failures:
    sys.exit(1)
print("GC_PAUSE_PROBE_TESTS_PASS")
