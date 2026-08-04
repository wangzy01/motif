#!/usr/bin/env python3
"""Test the fix for the crash that killed the 2026-08-04 formal run at step 660.

The run died on this row:

    row_id=3902543 domain=anime  catalog 1032x1457  decoded 668x594  anchor 2 (432x576)
    AnchorGeometryError: crop_loss 0.333084 > 0.2

The anime domain deliberately tolerates upstream width/height metadata that disagrees
with the decoded pixels; Stage512a ran that way for 125,000 steps.  Stage512b wired up
anchors_v1.verify_binding(), which re-derives crop_loss from the DECODED size, so a row
the tolerance had already approved became a job-killing exception -- expected ~380 times
over 512M exposures, i.e. the configuration could never finish.

What is covered here: that crop_loss is now a distinct exception type, that the other
failure modes did NOT become tolerable, that the loader's two except clauses are ordered
so the subclass is reachable, and that the tolerated row is actually usable (its resize
does not upsample).

What is NOT covered here: the live loader path.  Only a real run decoding real anime
payloads exercises that, and the fatal condition occurs at ~7.4e-7 -- far below what any
smoke-sized sample can reach.  The end-to-end proof is a GPUVAL that gets past step 660;
a unit test claiming to prove it would be a check that cannot fail.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.dataline import anchors_v1  # noqa: E402

LOADER = ROOT / "training/dataline_offline/offline_dataset.py"

# The production numbers, not invented ones.
CRASH_W, CRASH_H = 668, 594
CRASH_ANCHOR = 2
ANIME_CROP_LOSS_MAX = 0.20

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((bool(ok), label, detail))


# ---------------------------------------------------------------- type discrimination
try:
    anchors_v1.verify_binding(CRASH_W, CRASH_H, CRASH_ANCHOR,
                              crop_loss_max=ANIME_CROP_LOSS_MAX)
    check(False, "the production crash geometry still raises", "no exception at all")
except anchors_v1.AnchorCropLossError as exc:
    check(True, "crash geometry raises AnchorCropLossError", str(exc))
    check(abs(exc.crop_loss - 0.333084) < 1e-5,
          "crop_loss attribute matches the logged 0.333084", f"{exc.crop_loss:.6f}")
    check(exc.crop_loss_max == ANIME_CROP_LOSS_MAX,
          "crop_loss_max attribute is carried", str(exc.crop_loss_max))
except anchors_v1.AnchorGeometryError as exc:
    check(False, "crash geometry raises AnchorCropLossError",
          f"got the base class instead: {exc}")

# A subclass, so every existing `except AnchorGeometryError` still catches it.  Two
# callers rely on this (smoke_anchors_loader_v1.py and the phase5/phase3a tests); if the
# new type were a sibling they would stop catching crop-loss failures and report a
# traceback instead of a finding.
check(issubclass(anchors_v1.AnchorCropLossError, anchors_v1.AnchorGeometryError),
      "AnchorCropLossError is a subclass of AnchorGeometryError")
try:
    anchors_v1.verify_binding(CRASH_W, CRASH_H, CRASH_ANCHOR,
                              crop_loss_max=ANIME_CROP_LOSS_MAX)
except anchors_v1.AnchorGeometryError:
    check(True, "base-class handlers still catch a crop-loss failure")
except Exception as exc:  # noqa: BLE001
    check(False, "base-class handlers still catch a crop-loss failure", repr(exc))

# ------------------------------------------------- the fail-closed modes stay fail-closed
# These are the cases the tolerance must NOT cover.  If any of them became an
# AnchorCropLossError, the loader's new except clause would swallow a genuinely broken
# binding -- which is the failure mode that matters, since the whole point of the gate is
# to catch EXIF rotation and stale manifests.
for label, args, kwargs in (
    ("upsample required", (100, 100, 0), {}),
    ("anchor_id out of range (negative)", (1024, 1024, -1), {}),
    ("anchor_id out of range (too large)", (1024, 1024, 13), {}),
    ("zero decoded size", (0, 576, 2), {}),
):
    try:
        anchors_v1.verify_binding(*args, crop_loss_max=ANIME_CROP_LOSS_MAX, **kwargs)
        check(False, f"{label} still raises", "no exception")
    except anchors_v1.AnchorCropLossError as exc:
        check(False, f"{label} must NOT be an AnchorCropLossError",
              f"tolerable now: {exc}")
    except anchors_v1.AnchorGeometryError:
        check(True, f"{label} stays fail-closed (base class)")

# A rotated portrait/landscape pair is the case verify_binding was added for.  It must
# stay fatal for non-anime domains, and for anime it is only reachable via the drift
# tolerance -- which is a policy decision recorded in the loader, not here.
try:
    anchors_v1.verify_binding(1457, 1032, CRASH_ANCHOR, crop_loss_max=ANIME_CROP_LOSS_MAX)
    check(False, "EXIF-style rotation raises", "no exception")
except anchors_v1.AnchorCropLossError:
    check(True, "EXIF-style rotation is a crop-loss failure (tolerable only on the "
                "anime drift path)")
except anchors_v1.AnchorGeometryError as exc:
    check(True, "EXIF-style rotation stays fail-closed", str(exc))

# ------------------------------------------------------------- the tolerated row is usable
# The justification for tolerating is that the image can still be produced without
# upsampling; if the resize had to enlarge, tolerating would mean training on invented
# pixels.  scale = max(W/w0, H/h0) < 1 proves it only shrinks.
W, H = anchors_v1.ANCHORS_V1_WH[CRASH_ANCHOR]
scale = max(W / CRASH_W, H / CRASH_H)
check(scale < 1.0, "the tolerated row shrinks rather than upsamples",
      f"scale={scale:.4f} -> {round(CRASH_W * scale)}x{round(CRASH_H * scale)} "
      f"then crop to {W}x{H}")

# ------------------------------------------------------------- the shipped loader source
# Read the loader rather than importing it: importing pulls pyarrow and the whole offline
# stack, and what is being checked is textual structure that no import would reveal.
src = LOADER.read_text()

crop_at = src.find("except anchors_v1.AnchorCropLossError")
base_at = src.find("except anchors_v1.AnchorGeometryError")
check(crop_at != -1, "loader catches AnchorCropLossError")
check(base_at != -1, "loader still catches AnchorGeometryError")
# Python tries except clauses in order.  Base-class-first would shadow the subclass
# entirely and restore the crash, while looking correct in review.
check(crop_at != -1 and base_at != -1 and crop_at < base_at,
      "the subclass clause precedes the base clause (otherwise it is unreachable)",
      f"crop@{crop_at} base@{base_at}")

# The flag has to be initialised outside the mismatch branch, or a row that never enters
# the branch raises NameError at the gate instead of passing.
init_at = src.find("anime_native_drift = False")
set_at = src.find("anime_native_drift = True")
read_at = src.find("if not anime_native_drift:")
check(init_at != -1, "drift flag is initialised to False")
check(set_at != -1, "drift flag is set by the tolerance branch")
check(init_at < set_at < read_at,
      "flag is initialised, then set, then read -- in that order",
      f"init@{init_at} set@{set_at} read@{read_at}")
check(src.count("anime_native_drift = True") == 1,
      "exactly one place sets the flag",
      f"count={src.count('anime_native_drift = True')}")

# The breaker must be a strict > against the cap, so exactly _MAX rows are tolerated and
# the next one fails.  A >= would tolerate one fewer and a < would never fire at all.
m = re.search(r"self\._anchors_drift_crop_tolerated\s*(>=?|<=?)\s*"
              r"_ANIME_DRIFT_CROP_TOLERATED_MAX", src)
check(m is not None and m.group(1) == ">",
      "the circuit breaker compares with > against the cap",
      m.group(0) if m else "no comparison found")
check(re.search(r"^_ANIME_DRIFT_CROP_TOLERATED_MAX = (\d+)$", src, re.M) is not None,
      "the cap is a module constant, greppable and reviewable")
# Every tolerated row must print.  A counter with no output is a mute button.
tol_block = src[crop_at:base_at] if crop_at != -1 and base_at != -1 else ""
check("print(" in tol_block, "every tolerated row prints a WARNING")
check("self._anchors_drift_crop_tolerated += 1" in tol_block,
      "every tolerated row increments the counter")

failures = [(label, detail) for ok, label, detail in results if not ok]
for ok, label, detail in results:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
print(f"\n{len(results)} checks, {len(failures)} failures")
for label, detail in failures:
    print(f"  {label}: {detail}")
if failures:
    sys.exit(1)
print("ANIME_DRIFT_CROP_TOLERANCE_TESTS_PASS")
