#!/usr/bin/env python3
"""Closure audit: every row_id-keyed frozen registry must be re-keyed to the catalog
the run actually binds.

Two formal runs died on the same omission in one day (2026-08-04/05). The Stage512b
catalog renumbered every row; three registries carry per-row exceptions keyed by row_id:

  image_exceptions_v1        textatlas JPEG EOI repairs   -> killed the run at step 5746
  anime_native_upsample_*    below-anchor upsample waiver -> killed the run at step 11270
  caption_exceptions_v1      captionless MONET rows       -> plan references them 0 times

Each registry was individually correct and fail-closed; what failed was the MIGRATION
obligation: building a new catalog silently invalidates every old row_id key. This test
makes that obligation executable for the Stage512b lineage:

  * the Stage512b profiles must exist and carry the current plan/catalog identities
    (a rebuilt catalog or plan changes those shas, so this test fails loudly instead of
    the run failing at whatever step first draws an excepted row);
  * the re-keyed rows must resolve, by uid, to exactly the claimed new row ids;
  * the anime exception's first coordinate is spot-checked against the actual plan block
    (full 245-block scans live in the builder, not in a per-commit test; set
    REKEY_CLOSURE_FULL_SCAN=1 to re-derive all coordinates).

If a FOURTH registry keyed by row_id is ever added, add it to REGISTRIES here -- the
final check fails if a module under training/dataline_offline matches the registry
naming pattern but is not listed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

B = Path("/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop")
CATALOG = B / "offline_arb_catalog_stage512b_v1"
PLAN = B / "offline_arb_plan_stage512b_w64_512m_anchors_v1_20260802"

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((bool(ok), label, detail))


def fsha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows_by_uid(parquet: Path, uids: set[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    f = pq.ParquetFile(parquet)
    for rg in range(f.num_row_groups):
        tbl = f.read_row_group(rg, columns=["row_id", "uid"])
        col = tbl.column("uid").to_pylist()
        for i, u in enumerate(col):
            if u in uids and u not in found:
                found[u] = tbl.column("row_id")[i].as_py()
        if len(found) == len(uids):
            break
    return found


def main() -> int:
    plan_header_sha = fsha(PLAN / "header.json")
    catalog_header_sha = fsha(CATALOG / "header.json")
    catalog_hash = json.loads((CATALOG / "header.json").read_text())["catalog_hash"]

    # ---- registry 1: textatlas JPEG repairs -------------------------------------------
    from training.dataline_offline import image_exceptions_v1 as img

    check(img.STAGE512B_ANCHORS_CATALOG_HASH == catalog_hash,
          "image repairs: profile pins the live catalog hash")
    check(img.STAGE512B_ANCHORS_PLAN_HEADER_SHA256 == plan_header_sha,
          "image repairs: profile pins the live plan header",
          "a rebuilt plan must re-derive this profile")
    check(img.STAGE512B_ANCHORS_CATALOG_HEADER_SHA256 == catalog_header_sha,
          "image repairs: profile pins the live catalog header")
    uid_map = rows_by_uid(
        CATALOG / "textatlas.parquet",
        {r.uid for r in img.STAGE512B_ANCHORS_TEXTATLAS_JPEG_EOI_REPAIRS.values()},
    )
    for row_id, r in img.STAGE512B_ANCHORS_TEXTATLAS_JPEG_EOI_REPAIRS.items():
        check(uid_map.get(r.uid) == row_id,
              f"image repairs: uid resolves to row {row_id}",
              f"catalog says {uid_map.get(r.uid)}")

    # ---- registry 2: anime native-upsample exception ----------------------------------
    from training.dataline_offline import (
        anime_native_upsample_exceptions_weighted_512b_v1 as ups,
    )

    check(ups.CATALOG_HASH == catalog_hash,
          "anime upsample: profile pins the live catalog hash")
    check(ups.PLAN_HEADER_SHA256 == plan_header_sha,
          "anime upsample: profile pins the live plan header")
    check(ups.CATALOG_HEADER_SHA256 == catalog_header_sha,
          "anime upsample: profile pins the live catalog header")
    uid_map = rows_by_uid(
        CATALOG / "anime.parquet",
        {e.uid for e in ups.STAGE512B_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS.values()},
    )
    for row_id, e in ups.STAGE512B_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS.items():
        check(uid_map.get(e.uid) == row_id,
              f"anime upsample: uid resolves to row {row_id}",
              f"catalog says {uid_map.get(e.uid)}")

    # Coordinates: spot-check the first against the actual plan bytes; the full scan is
    # opt-in because 245 blocks cost minutes, not seconds.
    header = json.loads((PLAN / "header.json").read_text())
    bs = int(header["block_steps"])
    coords = sorted(
        (c.step, c.rank, c.slot)
        for e in ups.STAGE512B_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS.values()
        for c in e.coordinates
    )
    step, rank, slot = coords[0]
    with np.load(PLAN / "blocks" / f"block_{step // bs:06d}.npz",
                 allow_pickle=False) as a:
        actual_row = int(np.asarray(a["item_row"])[step % bs, rank, slot])
    check(actual_row == 534_123,
          f"anime upsample: plan block confirms coordinate ({step},{rank},{slot})",
          f"plan holds row {actual_row}")
    if os.environ.get("REKEY_CLOSURE_FULL_SCAN") == "1":
        hits = []
        for b in range(int(header["num_blocks"])):
            with np.load(PLAN / "blocks" / f"block_{b:06d}.npz",
                         allow_pickle=False) as a:
                for s, r, sl in np.argwhere(np.asarray(a["item_row"]) == 534_123):
                    hits.append((b * bs + int(s), int(r), int(sl)))
        check(sorted(hits) == coords,
              "anime upsample: FULL plan scan matches the pinned coordinates",
              f"scan found {len(hits)}")

    # ---- registry 3: captionless MONET rows -------------------------------------------
    # All six uids exist in the Stage512b catalog but the plan references none of them
    # (verified by full scan on 2026-08-05), so no Stage512b profile is required. What
    # this test pins is the REASONING's premise: if a rebuilt plan ever references any of
    # them, this check cannot see it cheaply -- so instead pin the plan identity the
    # audit was performed against.
    audited_plan_header = "5cd978e8b8392ff379c0e2961f2d515279b052cf1ab766a097fc15922349c806"
    check(plan_header_sha == audited_plan_header,
          "captionless-monet audit is bound to this exact plan",
          "the plan changed: re-run the captionless-monet occurrence scan")

    # ---- no unlisted registry ----------------------------------------------------------
    listed = {
        "image_exceptions_v1.py",
        "caption_exceptions_v1.py",
        "anime_native_upsample_exceptions_weighted_v1.py",
        "anime_native_upsample_exceptions_control_rngfix_v1.py",
        "anime_native_upsample_exceptions_weighted_512b_v1.py",
    }
    pattern = re.compile(r"(exception|repair)", re.I)
    present = {
        p.name
        for p in (ROOT / "training/dataline_offline").glob("*.py")
        if pattern.search(p.name)
    }
    unlisted = present - listed
    check(not unlisted,
          "every row_id-keyed registry module is covered by this audit",
          f"unlisted: {sorted(unlisted)}")

    failures = [(l, d) for ok, l, d in results if not ok]
    for ok, label, detail in results:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
    print(f"\n{len(results)} checks, {len(failures)} failures")
    if failures:
        return 1
    print("REGISTRY_REKEY_CLOSURE_TESTS_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
