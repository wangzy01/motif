#!/usr/bin/env python
"""Build "TextAtlas v1" per-source-shard sidecar manifests.

GOAL: produce one sidecar parquet per source image shard so the training reader,
given a source image parquet, knows each image's size-eligibility and its i1
caption WITHOUT re-reading the i1 dataset at train time.

INPUTS:
  - Source images:  /s3/evan/textatlas5m/<Subset>/train-NNNNN-of-MMMMM.parquet
      row schema: image (struct<bytes:binary, path:string>), image_path (string),
      annotation (string)  [TextScenesHQ also has raw_text]. Image bytes inline.
  - i1 captions: /fsx/evan/textatlas_i1_captions/textatlas/*.parquet
      schema: key (= "<Subset>:<image_path>"), caption1..caption5

OUTPUT (hard schema contract, in source row order):
  /fsx/evan/textatlas_manifest_v1/<Subset>/<same source shardname>.parquet
    image_path (string), uid (string = "<Subset>:<image_path>"),
    width (int32), height (int32),
    elig_512 (bool = min(w,h)>=512), elig_1024 (bool = min(w,h)>=1024),
    caption1..caption5 (string; "" if uid absent from i1)

METHOD:
  - Load ALL i1 textatlas captions into a module-global dict uid->(c1..c5)
    in the PARENT before the Pool fork (COW-shared, read-only across workers).
  - Per shard: stream rows with pyarrow (read only image + image_path), read
    header-only size via PIL.Image.open(...).size (does NOT fully decode),
    compute elig, look up captions, write sidecar preserving source row order.
  - Robustness: per-row image-open wrapped in try/except (corrupt -> w=h=0,
    elig False). Shards that error are skipped + logged.
"""
import argparse
import glob
import io
import os
import sys
import time
import traceback
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # textatlas has some large pages; don't bomb

SRC_ROOT = "/s3/evan/textatlas5m"
I1_DIR = "/fsx/evan/textatlas_i1_captions/textatlas"
OUT_ROOT = "/fsx/evan/textatlas_manifest_v1"
SUBSETS = [
    "CleanTextSynth", "CoverBook", "LongWordsSubset-A", "LongWordsSubset-M",
    "PPT2Details", "PPT2Structured", "Paper2Text", "StyledTextSynth",
    "TextScenesHQ", "TextVisionBlend",
]

# Hard output schema contract.
OUT_SCHEMA = pa.schema([
    ("image_path", pa.string()),
    ("uid", pa.string()),
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("elig_512", pa.bool_()),
    ("elig_1024", pa.bool_()),
    ("caption1", pa.string()),
    ("caption2", pa.string()),
    ("caption3", pa.string()),
    ("caption4", pa.string()),
    ("caption5", pa.string()),
])

# Module-global caption dict: uid -> (c1, c2, c3, c4, c5). Set in parent before fork.
CAPS = {}
EMPTY5 = ("", "", "", "", "")


def load_captions(verbose=True):
    """Load all i1 textatlas captions into the module-global CAPS dict."""
    global CAPS
    files = sorted(glob.glob(os.path.join(I1_DIR, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no i1 caption parquets under {I1_DIR}")
    t0 = time.perf_counter()
    n = 0
    cols = ["key", "caption1", "caption2", "caption3", "caption4", "caption5"]
    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=200_000, columns=cols):
            keys = b.column("key").to_pylist()
            c1 = b.column("caption1").to_pylist()
            c2 = b.column("caption2").to_pylist()
            c3 = b.column("caption3").to_pylist()
            c4 = b.column("caption4").to_pylist()
            c5 = b.column("caption5").to_pylist()
            for i, k in enumerate(keys):
                CAPS[k] = (
                    c1[i] or "", c2[i] or "", c3[i] or "",
                    c4[i] or "", c5[i] or "",
                )
            n += len(keys)
        if verbose:
            print(f"[caps] loaded {f} -> {n} cumulative entries "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    if verbose:
        print(f"[caps] DONE: {len(CAPS)} uids in {time.perf_counter()-t0:.0f}s", flush=True)
    return len(CAPS)


def build_shard(subset, src_path, out_path):
    """Build a single sidecar manifest. Returns (rows, elig512, elig1024, missing_caps)."""
    image_paths, uids, widths, heights = [], [], [], []
    e512, e1024 = [], []
    cap1, cap2, cap3, cap4, cap5 = [], [], [], [], []

    n_e512 = n_e1024 = n_missing = 0
    pf = pq.ParquetFile(src_path)
    for batch in pf.iter_batches(batch_size=512, columns=["image", "image_path"]):
        imgs = batch.column("image").to_pylist()
        ipaths = batch.column("image_path").to_pylist()
        for img, ip in zip(imgs, ipaths):
            w = h = 0
            try:
                b = img["bytes"]
                with Image.open(io.BytesIO(b)) as im:
                    w, h = im.size  # header-only; .size does not fully decode
                w, h = int(w), int(h)
            except Exception:
                w = h = 0
            m = min(w, h)
            ok512 = m >= 512
            ok1024 = m >= 1024
            uid = f"{subset}:{ip}"
            c = CAPS.get(uid)
            if c is None:
                c = EMPTY5
                n_missing += 1

            image_paths.append(ip)
            uids.append(uid)
            widths.append(w)
            heights.append(h)
            e512.append(ok512)
            e1024.append(ok1024)
            cap1.append(c[0]); cap2.append(c[1]); cap3.append(c[2])
            cap4.append(c[3]); cap5.append(c[4])
            n_e512 += ok512
            n_e1024 += ok1024

    table = pa.table({
        "image_path": image_paths,
        "uid": uids,
        "width": pa.array(widths, type=pa.int32()),
        "height": pa.array(heights, type=pa.int32()),
        "elig_512": e512,
        "elig_1024": e1024,
        "caption1": cap1, "caption2": cap2, "caption3": cap3,
        "caption4": cap4, "caption5": cap5,
    }, schema=OUT_SCHEMA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # write to tmp then rename for atomicity
    tmp = out_path + ".tmp"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, out_path)
    return table.num_rows, n_e512, n_e1024, n_missing


def _worker(task):
    """Pool worker: build one shard. CAPS inherited via fork (COW)."""
    subset, src_path, out_path, overwrite = task
    try:
        if (not overwrite) and os.path.exists(out_path):
            return ("skip-exists", subset, os.path.basename(src_path), None)
        rows, e512, e1024, miss = build_shard(subset, src_path, out_path)
        return ("ok", subset, os.path.basename(src_path),
                {"rows": rows, "e512": e512, "e1024": e1024, "missing": miss})
    except Exception:
        return ("error", subset, os.path.basename(src_path), traceback.format_exc())


def enumerate_shards(subsets):
    tasks = []
    for s in subsets:
        for p in sorted(glob.glob(os.path.join(SRC_ROOT, s, "*.parquet"))):
            out = os.path.join(OUT_ROOT, s, os.path.basename(p))
            tasks.append((s, p, out))
    return tasks


def dry_run(src_rel):
    """Process ONE shard, print samples + counts. src_rel like 'CleanTextSynth/train-00000-of-00067.parquet'."""
    subset = src_rel.split("/", 1)[0]
    src_path = os.path.join(SRC_ROOT, src_rel)
    out_path = os.path.join(OUT_ROOT, src_rel)
    print(f"[dry-run] loading captions...", flush=True)
    load_captions()
    print(f"[dry-run] building shard {src_path}\n  -> {out_path}", flush=True)
    rows, e512, e1024, miss = build_shard(subset, src_path, out_path)

    # read back + show samples
    t = pq.read_table(out_path)
    print(f"\n[dry-run] OUTPUT schema:\n{t.schema}", flush=True)
    print(f"\n[dry-run] shard row count = {rows}", flush=True)
    print(f"[dry-run] elig_512 count   = {e512}  ({e512/max(rows,1):.1%})", flush=True)
    print(f"[dry-run] elig_1024 count  = {e1024}  ({e1024/max(rows,1):.1%})", flush=True)
    print(f"[dry-run] missing captions = {miss}  ({miss/max(rows,1):.1%})", flush=True)
    print(f"\n[dry-run] 3 sample manifest rows:", flush=True)
    d = t.slice(0, 3).to_pylist()
    for r in d:
        found = r["uid"] in CAPS
        print(f"  uid={r['uid']!r} w={r['width']} h={r['height']} "
              f"elig_512={r['elig_512']} elig_1024={r['elig_1024']} "
              f"caption1_chars={len(r['caption1'])} caption_found={found}", flush=True)
    # also show a non-first sample to vary
    return rows, e512, e1024, miss


def full_run(subsets, procs, overwrite):
    print(f"[full] loading captions in parent (COW-shared to workers)...", flush=True)
    load_captions()
    tasks = enumerate_shards(subsets)
    print(f"[full] {len(tasks)} shards across {len(subsets)} subsets; "
          f"procs={procs} overwrite={overwrite}", flush=True)
    wtasks = [(s, sp, op, overwrite) for (s, sp, op) in tasks]

    t0 = time.perf_counter()
    n_ok = n_skip = n_err = 0
    tot_rows = tot_e512 = tot_e1024 = tot_miss = 0
    errors = []
    with Pool(processes=procs) as pool:
        for i, (status, subset, name, info) in enumerate(
                pool.imap_unordered(_worker, wtasks, chunksize=1)):
            if status == "ok":
                n_ok += 1
                tot_rows += info["rows"]; tot_e512 += info["e512"]
                tot_e1024 += info["e1024"]; tot_miss += info["missing"]
            elif status == "skip-exists":
                n_skip += 1
            else:
                n_err += 1
                errors.append((subset, name, info))
                print(f"[full][ERROR] {subset}/{name}:\n{info}", flush=True)
            if (i + 1) % 25 == 0 or status == "error":
                el = time.perf_counter() - t0
                print(f"[full] {i+1}/{len(wtasks)} done "
                      f"(ok={n_ok} skip={n_skip} err={n_err}) "
                      f"rows={tot_rows} elig512={tot_e512} elig1024={tot_e1024} "
                      f"missing={tot_miss}  {el:.0f}s", flush=True)

    el = time.perf_counter() - t0
    print(f"\n[full] DONE in {el:.0f}s: ok={n_ok} skip={n_skip} err={n_err}", flush=True)
    print(f"[full] totals: rows={tot_rows} elig_512={tot_e512} "
          f"({tot_e512/max(tot_rows,1):.1%}) elig_1024={tot_e1024} "
          f"({tot_e1024/max(tot_rows,1):.1%}) missing_caps={tot_miss} "
          f"({tot_miss/max(tot_rows,1):.1%})", flush=True)
    if errors:
        print(f"[full] {len(errors)} shards FAILED:", flush=True)
        for s, n, _ in errors:
            print(f"   FAILED: {s}/{n}", flush=True)
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", metavar="SUBSET/SHARD.parquet", default=None,
                    help="process one shard and print samples; e.g. "
                         "CleanTextSynth/train-00000-of-00067.parquet")
    ap.add_argument("--procs", type=int, default=40, help="pool workers for full run")
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="restrict to these subsets (default: all 10)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild even if sidecar exists")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args.dry_run)
    else:
        subs = args.subsets or SUBSETS
        full_run(subs, args.procs, args.overwrite)


if __name__ == "__main__":
    main()
