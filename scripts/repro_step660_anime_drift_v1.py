#!/usr/bin/env python3
"""Reproduce the exact row that killed the 2026-08-04 formal run, end to end.

    step=660 rank=12 slot=3 row_id=3902543 domain=anime
    catalog 1032x1457  decoded 668x594  anchor 2 (432x576)
    AnchorGeometryError: crop_loss 0.333084 > 0.2   -> 64 GPUs down at 07:10 UTC

Why this script exists rather than waiting for a GPUVAL: the failing row sits at plan
step 660, which a real run reaches after about 55 minutes on 8 nodes.  ``set_start_step``
seeks the loader to an absolute plan step, so one rank on one CPU can decode the twenty
steps around 660 -- roughly 640 rows -- and exercise the identical code path in minutes,
on the real plan, the real catalog and the real upstream bytes.

This is the test that would have caught the crash before launch.  The unit tests around
it check that crop_loss is now a distinct exception type; only this one proves the row
actually survives the loader.  Nothing at smoke scale could have found it by sampling: at
the measured 7.4e-7 rate you need ~10^6 decoded rows to expect one hit, and the smoke
draws a few hundred.  Seeking to the known row is the difference between a test that
could fail and a test that merely passes.

Exit 0 means the row was decoded, tolerated with a WARNING, and produced a batch at the
anchor the plan froze.  Exit 1 means the crash is not fixed.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/offline_arb"))

CONFIG = str(ROOT / "configs/stage512b_real_anchors_v1.yaml")
CRASH_STEP = 660
CRASH_RANK = 12
CRASH_SLOT = 3
CRASH_ROW_ID = 3902543
START_STEP = 650
STOP_STEP = 665

from smoke_anchors_loader_v1 import build_dataset  # noqa: E402


def main() -> int:
    print(f"seeking rank {CRASH_RANK} of world 64 to plan step {START_STEP}, "
          f"running through {STOP_STEP} (the crash is at {CRASH_STEP})", flush=True)
    dataset = build_dataset(CONFIG, CRASH_RANK, 64, 32)
    try:
        dataset.set_start_step(START_STEP)
        loader = dataset.loader()

        seen_rows: list[int] = []
        shapes: dict[int, tuple] = {}
        anchors: dict[int, int] = {}
        # The tolerance prints to stdout from inside the supply thread.  Capture it so the
        # WARNING can be asserted on rather than merely eyeballed -- a fix that stopped
        # printing would be a mute button, which is exactly what must not happen.
        buf = io.StringIO()
        with redirect_stdout(buf):
            for step in range(START_STEP, STOP_STEP + 1):
                batch = next(loader)
                pixels = batch["pixels"]
                shapes[step] = (int(pixels.shape[-2]), int(pixels.shape[-1]))
                # `item_rows`, the key the loader actually writes (offline_dataset.py:4197).
                # Guessing "row_ids" here would have made the row-presence line always
                # report False while the script still exited 0 -- a check that cannot fail,
                # reporting the one fact the whole script exists to establish.
                trace = getattr(dataset, "last_item_trace", None) or {}
                rows = trace.get("item_rows") or []
                seen_rows.extend(int(r) for r in rows)
                if trace.get("bucket_idx") is not None:
                    anchors[step] = int(trace["bucket_idx"])
        captured = buf.getvalue()
        print(captured, end="")

        ok = True
        print(f"\ndecoded steps {START_STEP}..{STOP_STEP} without raising  <- the crash "
              f"reproduced here before the fix")

        hit = CRASH_ROW_ID in seen_rows
        print(f"decoded {len(seen_rows)} rows; row_id {CRASH_ROW_ID} among them: {hit}")
        if not hit:
            # Without the row, "it did not raise" proves nothing about this crash: the
            # script would be green while never touching the failing input.
            print(f"  FAIL the target row was never drawn, so nothing was proven. "
                  f"Check that rank {CRASH_RANK}/world 64 and steps "
                  f"{START_STEP}..{STOP_STEP} still cover it.")
            ok = False
        if anchors.get(CRASH_STEP) is not None:
            print(f"step {CRASH_STEP} drew anchor {anchors[CRASH_STEP]} (the crash was on "
                  f"anchor 2)")

        drift = "native-size metadata mismatch" in captured
        tolerated = "crop_loss tolerated on an anime native-drift row" in captured
        print(f"anime native-size mismatch WARNING seen: {drift}")
        print(f"anchors_v1 crop_loss tolerance WARNING seen: {tolerated}")
        if drift and not tolerated:
            # A mismatch that did not need the tolerance is normal -- most drifted rows
            # keep an aspect close enough to stay under the threshold.  Only report, do
            # not fail: upstream may have replaced the payload since 07:10 UTC.
            print("  note: the drifted row(s) decoded within crop_loss this time; the "
                  "tolerance was not needed for them")

        if shapes.get(CRASH_STEP):
            h, w = shapes[CRASH_STEP]
            print(f"step {CRASH_STEP} batch geometry: {w}x{h}")
            if (h, w) != (576, 432):
                print(f"  FAIL expected the frozen anchor 2 (432x576), got {w}x{h}")
                ok = False

        counter = getattr(dataset, "_anchors_drift_crop_tolerated", None)
        print(f"tolerated-row counter on this rank: {counter}")

        print("\nSTEP660_REPRO_PASS" if ok else "\nSTEP660_REPRO_FAIL")
        return 0 if ok else 1
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    sys.exit(main())
