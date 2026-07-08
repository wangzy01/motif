#!/usr/bin/env python3
"""Replace MONET/synthetic sample preview thumbnails with full-res tar images.

This is for generated dive/sample artifacts such as
complete_v1_training_samples_seed20260617.  The sampler wrote MONET/synthetic
preview jpgs from the source parquet thumbnail field; training reads the
full-resolution jpg from the sibling webdataset tar.  This script rewrites the
existing preview jpg path with that tar jpg and updates parquet/summary metadata.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image


DEFAULT_NEW_SOURCE = "monet_tar_fullres"


def s3_to_mount(s: str) -> Path:
    if s.startswith("s3://pixai-test-uw2/"):
        return Path("/s3-pixai-test-uw2") / s[len("s3://pixai-test-uw2/"):]
    return Path(s)


def tar_path(src_root: str, src_shard: str) -> str:
    return str(Path(src_root) / src_shard.replace(".parquet", ".tar"))


def load_success_state(state_dir: Path, legacy_jsonl: Path | None = None) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    if legacy_jsonl and legacy_jsonl.exists():
        with open(legacy_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stage, uid = rec.get("stage"), rec.get("uid")
                if stage and uid:
                    out.add((str(stage), str(uid)))
    if not state_dir.exists():
        return out
    for marker in state_dir.glob("stage=*/*.json"):
        try:
            with open(marker, "r") as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        stage, uid = rec.get("stage"), rec.get("uid")
        if stage and uid:
            out.add((str(stage), str(uid)))
    return out


def write_success_marker(state_dir: Path, rec: dict[str, Any]) -> None:
    stage = str(rec["stage"])
    src_key = str(rec["src_key"])
    path = state_dir / f"stage={stage}" / f"{src_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp") / f"fullres_success_{stage}_{src_key}.json"
    with open(tmp, "w") as f:
        json.dump(rec, f, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    with open(tmp, "rb") as src, open(path, "wb") as dst:
        dst.write(src.read())
    tmp.unlink(missing_ok=True)


def load_tasks(out_root: Path, stages: list[str], new_source: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for stage in stages:
        p = out_root / f"stage_{stage}" / f"stage_{stage}_sample.parquet"
        table = pq.read_table(p)
        for i, row in enumerate(table.to_pylist()):
            if row.get("image_materialized_from") not in ("source_parquet_thumbnail", new_source):
                continue
            if row.get("data_source") not in ("monet_real", "synthetic"):
                continue
            dest = s3_to_mount(row["s3key"])
            tasks.append({
                "stage": str(stage),
                "row": i,
                "uid": row["uid"],
                "src_key": row["src_key"],
                "src_root": row["src_root"],
                "src_shard": row["src_shard"],
                "tar": tar_path(row["src_root"], row["src_shard"]),
                "member": f"{row['src_key']}.jpg",
                "dest": str(dest),
                "meta_w": int(row["width"]) if row.get("width") is not None else None,
                "meta_h": int(row["height"]) if row.get("height") is not None else None,
            })
    return tasks


def lookup_urls_for_group(item: tuple[tuple[str, str], list[dict[str, Any]]]) -> tuple[int, list[str]]:
    (src_root, src_shard), group = item
    source_parquet = Path(src_root) / src_shard
    wanted = {t["src_key"] for t in group}
    warnings = []
    try:
        table = pq.read_table(source_parquet, columns=["id", "__key__", "url"])
    except Exception as exc:  # noqa: BLE001
        return 0, [f"WARN url_lookup_failed shard={source_parquet} err={type(exc).__name__}:{exc}"]
    mask = pc.or_(pc.is_in(table["id"], value_set=pa.array(list(wanted))),
                  pc.is_in(table["__key__"], value_set=pa.array(list(wanted))))
    url_by_key: dict[str, str] = {}
    for row in table.filter(mask).to_pylist():
        url = row.get("url")
        if not url:
            continue
        if row.get("id") in wanted:
            url_by_key[row["id"]] = url
        if row.get("__key__") in wanted:
            url_by_key[row["__key__"]] = url
    attached = 0
    for task in group:
        url = url_by_key.get(task["src_key"])
        if url:
            task["url"] = url
            attached += 1
    return attached, warnings


def attach_real_urls(tasks: list[dict[str, Any]], workers: int) -> int:
    """Attach source parquet URL for monet_real tasks.

    This is only used for display replacement.  We still validate downloaded
    dimensions against metadata and fall back to tar if the URL is absent/bad.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        if task.get("src_key") and "monet_real" in task.get("dest", ""):
            groups[(task["src_root"], task["src_shard"])].append(task)

    attached = 0
    items = list(groups.items())
    total = len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(lookup_urls_for_group, item) for item in items]
        for gi, fut in enumerate(as_completed(futs), 1):
            n, warnings = fut.result()
            attached += n
            for msg in warnings:
                print(msg, flush=True)
            if gi % 25 == 0 or gi == total:
                print(f"url_lookup {gi}/{total} attached={attached}", flush=True)
    return attached


def validate_image(data: bytes, task: dict[str, Any]) -> tuple[bool, str | None, tuple[int, int] | None]:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        size = img.size
    except Exception as exc:  # noqa: BLE001
        return False, f"PIL_decode:{type(exc).__name__}:{exc}", None
    expected = (task["meta_w"], task["meta_h"])
    if size != expected:
        return False, f"size_mismatch got={size} expected={expected}", size
    return True, None, size


def fetch_url_bytes(url: str, timeout_sec: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mot-sample-fullres/1"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return resp.read()


def extract_tar_bytes(task: dict[str, Any], timeout_sec: int) -> tuple[bytes | None, str | None]:
    try:
        proc = subprocess.run(
            ["tar", "-xOf", task["tar"], task["member"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return None, f"tar_timeout>{timeout_sec}s"
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "replace")[-500:]
        return None, f"tar_rc={proc.returncode} {err}"
    return proc.stdout, None


def replace_one(task: dict[str, Any], timeout_sec: int, prefer_url_real: bool, url_timeout_sec: int) -> dict[str, Any]:
    start = time.time()
    errors: list[str] = []
    data = None
    size = None
    method = "tar"

    if prefer_url_real and task.get("url"):
        try:
            candidate = fetch_url_bytes(task["url"], url_timeout_sec)
            ok, err, candidate_size = validate_image(candidate, task)
            if ok:
                data = candidate
                size = candidate_size
                method = "url"
            else:
                errors.append(f"url:{err}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"url:{type(exc).__name__}:{exc}")

    if data is None:
        candidate, err = extract_tar_bytes(task, timeout_sec)
        if candidate is None:
            errors.append(err or "tar_unknown")
            return {**task, "ok": False, "error": " | ".join(errors), "sec": round(time.time() - start, 2)}
        ok, err, candidate_size = validate_image(candidate, task)
        if not ok:
            errors.append(f"tar:{err}")
            return {**task, "ok": False, "error": " | ".join(errors), "sec": round(time.time() - start, 2)}
        data = candidate
        size = candidate_size
        method = "tar"

    try:
        with open(task["dest"], "wb") as f:
            f.write(data)
    except Exception as exc:  # noqa: BLE001
        return {**task, "ok": False, "error": f"write:{type(exc).__name__}:{exc}", "sec": round(time.time() - start, 2)}

    return {
        **task,
        "ok": True,
        "error": None,
        "sec": round(time.time() - start, 2),
        "bytes": len(data),
        "size": size,
        "method": method,
    }


def write_parquet_via_tmp(path: Path, table: pa.Table) -> None:
    tmp = Path("/tmp") / f"{path.name}.fullres_tmp"
    pq.write_table(table, tmp)
    with open(tmp, "rb") as src, open(path, "wb") as dst:
        dst.write(src.read())
    tmp.unlink(missing_ok=True)


def update_source_column(path: Path, successes: set[tuple[str, str]], new_source: str) -> int:
    table = pq.read_table(path)
    idx = table.schema.names.index("image_materialized_from")
    field = table.schema.field(idx)
    stages = table.column("stage").to_pylist()
    uids = table.column("uid").to_pylist()
    vals = table.column("image_materialized_from").to_pylist()
    changed = 0
    new_vals = []
    for stage, uid, old in zip(stages, uids, vals):
        if (str(stage), uid) in successes and old in ("source_parquet_thumbnail", new_source):
            new_vals.append(new_source)
            if old != new_source:
                changed += 1
        else:
            new_vals.append(old)
    table = table.set_column(idx, field, pa.array(new_vals, type=field.type))
    write_parquet_via_tmp(path, table)
    return changed


def update_summary(out_root: Path, ok_count: int, fail_count: int) -> None:
    path = out_root / "summary.json"
    with open(path, "r") as f:
        data = json.load(f)
    image_stats = data.setdefault("image_stats", {})
    image_stats["tar"] = {"rows": ok_count, "ok": ok_count, "fail": fail_count}
    image_stats["thumbnail"] = {"rows": fail_count, "ok": fail_count, "fail": 0}
    data["monet_image_mode"] = "tar_fullres" if fail_count == 0 else "tar_fullres_partial"

    tmp = Path("/tmp/complete_v1_training_samples_summary.fullres_tmp.json")
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(tmp, "rb") as src, open(path, "wb") as dst:
        dst.write(src.read())
    tmp.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--stages", default="1024,512,256")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout-sec", type=int, default=900)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=0,
                    help="Process tasks in chunks; URL lookup is done per chunk when enabled.")
    ap.add_argument("--new-source", default=DEFAULT_NEW_SOURCE)
    ap.add_argument("--prefer-url-real", action="store_true")
    ap.add_argument("--url-timeout-sec", type=int, default=45)
    ap.add_argument("--url-lookup-workers", type=int, default=8)
    ap.add_argument("--no-metadata-update", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    state_dir = out_root / "fullres_replacement_successes"
    legacy_state_path = out_root / "fullres_replacement_successes.jsonl"
    fail_path = out_root / "fullres_replacement_failures.json"

    tasks = load_tasks(out_root, stages, args.new_source)
    successes = load_success_state(state_dir, legacy_state_path)
    tasks = [t for t in tasks if (t["stage"], t["uid"]) not in successes]
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    if args.prefer_url_real and args.batch_size <= 0:
        attached = attach_real_urls(tasks, args.url_lookup_workers)
        print(f"url_lookup_done attached={attached}", flush=True)

    print(
        f"tasks_to_run={len(tasks)} prior_successes={len(successes)} "
        f"workers={args.workers} timeout_sec={args.timeout_sec}",
        flush=True,
    )

    ok: list[dict[str, Any]] = []
    fail: list[dict[str, Any]] = []
    by_stage_ok = {s: 0 for s in stages}
    by_stage_fail = {s: 0 for s in stages}
    t0 = time.time()

    done = 0
    batch_size = args.batch_size if args.batch_size and args.batch_size > 0 else len(tasks)
    total_batches = (len(tasks) + batch_size - 1) // batch_size if batch_size else 1
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        batch_id = batch_start // batch_size + 1
        if args.prefer_url_real and args.batch_size > 0:
            attached = attach_real_urls(batch, args.url_lookup_workers)
            print(f"batch {batch_id}/{total_batches} url_lookup_done attached={attached}", flush=True)

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut_to_task = {
                ex.submit(replace_one, t, args.timeout_sec, args.prefer_url_real, args.url_timeout_sec): t
                for t in batch
            }
            for fut in as_completed(fut_to_task):
                task = fut_to_task[fut]
                done += 1
                try:
                    rec = fut.result()
                except Exception as exc:  # noqa: BLE001
                    rec = {**task, "ok": False, "error": f"worker_exception:{type(exc).__name__}:{exc}"}
                if rec["ok"]:
                    ok.append(rec)
                    successes.add((rec["stage"], rec["uid"]))
                    by_stage_ok[rec["stage"]] = by_stage_ok.get(rec["stage"], 0) + 1
                    try:
                        write_success_marker(state_dir, {
                            "stage": rec["stage"],
                            "uid": rec["uid"],
                            "src_key": rec["src_key"],
                            "dest": rec["dest"],
                            "bytes": rec.get("bytes"),
                            "size": rec.get("size"),
                            "method": rec.get("method"),
                            "sec": rec.get("sec"),
                        })
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"WARN marker_write_failed stage={rec['stage']} key={rec['src_key']} "
                            f"err={type(exc).__name__}:{exc}",
                            flush=True,
                        )
                else:
                    fail.append(rec)
                    by_stage_fail[rec["stage"]] = by_stage_fail.get(rec["stage"], 0) + 1
                    print(f"FAIL stage={rec['stage']} key={rec['src_key']} err={rec['error']}", flush=True)

                if done % args.progress_every == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed else 0.0
                    print(
                        f"progress {done}/{len(tasks)} ok={len(ok)} fail={len(fail)} "
                        f"elapsed={elapsed/60:.1f}m rate={rate:.3f}/s",
                        flush=True,
                    )

    print("ok_by_stage=" + json.dumps(by_stage_ok, sort_keys=True), flush=True)
    print("fail_by_stage=" + json.dumps(by_stage_fail, sort_keys=True), flush=True)
    print(f"replace_done new_ok={len(ok)} fail={len(fail)} total_successes={len(successes)}", flush=True)

    if fail:
        tmp = Path("/tmp/fullres_replacement_failures.json")
        with open(tmp, "w") as f:
            json.dump(fail, f, ensure_ascii=False, indent=2)
            f.write("\n")
        with open(tmp, "rb") as src, open(fail_path, "wb") as dst:
            dst.write(src.read())
        tmp.unlink(missing_ok=True)
        print(f"failures_written={fail_path}", flush=True)

    if not args.no_metadata_update:
        changed_total = 0
        for stage in stages:
            p = out_root / f"stage_{stage}" / f"stage_{stage}_sample.parquet"
            changed = update_source_column(p, successes, args.new_source)
            changed_total += changed
            print(f"updated {p}: {changed}", flush=True)
        combined = out_root / "complete_v1_all_stages_sample.parquet"
        changed_combined = update_source_column(combined, successes, args.new_source)
        update_summary(out_root, len(successes), len(fail))
        print(f"metadata_updated stage_total={changed_total} combined={changed_combined}", flush=True)

    if fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
