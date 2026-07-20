#!/usr/bin/env python3
"""Fail-closed source-files versus GPIC-blob semantic A/B.

The gate deliberately runs the production ``OfflineExactDataset`` twice for
each simulated rank.  The first instance consumes a fresh plan interval; the
second starts directly at the suffix boundary, exactly as ``train.py`` does on
resume.  A small hook at the dataset's already-prefetched raw-byte boundary
records GPIC-only raw and native decoded-pixel identities without replacing
any production decoding, crop, caption, or CFG code.

This program is submission-blocked until the caller supplies an immutable
artifact root, its build jobtag, and the canonical manifest file SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import threading
from pathlib import Path
from typing import Any


SCHEMA = "offline_gpic_blob_semantic_capture_v1"
COMPARE_SCHEMA = "offline_gpic_blob_semantic_ab_v1"
SHA_RE = re.compile(r"[0-9a-f]{64}")
BACKENDS = {"source_files_v1", "offline_gpic_blob_v1"}


class GateError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_size <= 0:
        raise GateError(f"not a nonempty regular file: {path}")
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        opened = os.fstat(handle.fileno())
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != identity:
            raise GateError(f"file changed while opening: {path}")
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
        closed = os.fstat(handle.fileno())
    after = path.lstat()
    for current in (closed, after):
        if (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ) != identity:
            raise GateError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def strict_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    if not raw:
        raise GateError(f"empty JSON: {path}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot parse JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"JSON root is not an object: {path}")
    return raw, value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise GateError(f"refusing overwrite/stale temporary output: {path}")
    with temporary.open("xb", buffering=0) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_artifact(root: Path, buildtag: str, manifest_sha256: str) -> dict[str, Any]:
    if not root.is_absolute() or root.is_symlink() or root.name != buildtag:
        raise GateError("artifact root/buildtag must be an exact absolute non-symlink binding")
    if not root.is_dir() or buildtag in {"", ".", "..", "latest"}:
        raise GateError("artifact root/buildtag is missing or unsafe")
    if SHA_RE.fullmatch(manifest_sha256) is None:
        raise GateError("manifest SHA-256 must be a nonempty lowercase digest")
    raw, manifest = strict_json(root / "manifest.json")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != manifest_sha256:
        raise GateError(f"manifest SHA mismatch: expected={manifest_sha256}, actual={actual}")
    marker = (root / "_SUCCESS").read_bytes()
    if marker != (manifest_sha256 + "\n").encode("ascii"):
        raise GateError("FSx _SUCCESS does not bind the exact manifest")
    release = manifest.get("release")
    storage = release.get("storage") if isinstance(release, dict) else None
    s3_root = storage.get("s3_root") if isinstance(storage, dict) else None
    if not isinstance(s3_root, str) or Path(s3_root).name != buildtag:
        raise GateError("manifest S3 root does not bind the buildtag")
    if manifest.get("status") != "PASS" or manifest.get("schema_version") != "offline_gpic_blob_v1":
        raise GateError("manifest schema/status is not published PASS")
    return manifest


def gpic_range(catalog: dict[str, Any]) -> tuple[int, int]:
    order = catalog.get("domain_order")
    files = catalog.get("files")
    if not isinstance(order, list) or not isinstance(files, dict) or "gpic" not in order:
        raise GateError("catalog domain/file structure is malformed")
    start = 0
    for domain in order:
        info = files.get(domain)
        if not isinstance(info, dict) or type(info.get("rows")) is not int:
            raise GateError(f"catalog rows missing for domain={domain!r}")
        if domain == "gpic":
            rows = int(info["rows"])
            if rows <= 0:
                raise GateError("catalog GPIC row count is not positive")
            return start, start + rows
        start += int(info["rows"])
    raise GateError("catalog GPIC range was not found")


def choose_start(
    plan_root: Path,
    catalog: dict[str, Any],
    *,
    ranks: int,
    fresh_steps: int,
    resume_offset: int,
    resume_steps: int,
) -> int:
    from training.dataline_offline.planner_v1 import OfflinePlan

    plan = OfflinePlan(plan_root, world_size=ranks, verify_checksums=True)
    lo, hi = gpic_range(catalog)
    max_start = min(256, int(plan.header["total_steps"]) - fresh_steps)
    for start in range(max_start + 1):
        acceptable = True
        for rank in range(ranks):
            total = 0
            suffix = 0
            for step in range(start, start + fresh_steps):
                _hw, rows = plan.batch(step, rank)
                count = sum(lo <= int(row) < hi for row in rows.tolist())
                total += count
                if step >= start + resume_offset:
                    suffix += count
            if total < 4 or suffix < 2:
                acceptable = False
                break
        if acceptable and resume_offset + resume_steps <= fresh_steps:
            return start
    raise GateError("no bounded plan interval gives every rank enough GPIC fresh/resume rows")


def tensor_sha(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    metadata = canonical_bytes({"dtype": array.dtype.str, "shape": list(array.shape)})
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def native_rgb_identity(raw: bytes) -> dict[str, Any]:
    import numpy as np
    from training.dataline.image_fetch import decode_rgb

    image = decode_rgb(raw)
    if image is None:
        raise GateError("GPIC raw capture could not be decoded by production decode_rgb")
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise GateError(f"decoded GPIC RGB shape is invalid: {array.shape}")
    return {
        "shape": list(array.shape),
        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
    }


def dataset_kwargs(args: argparse.Namespace, backend: str, rank: int) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(args.config))
    kwargs = dict(OmegaConf.to_container(cfg.dataset, resolve=True))
    kwargs.pop("store_cls", None)
    kwargs.update(
        rank=rank,
        world_size=args.ranks,
        offline_plan_root=str(args.plan_root),
        offline_catalog_root=str(args.catalog_root),
        gpic_payload_in_memory=True,
        gpic_payload_backend=backend,
        gpic_blob_fd_cache_size=1024,
    )
    if backend == "offline_gpic_blob_v1":
        kwargs.update(
            offline_gpic_blob_root=str(args.artifact_root),
            gpic_blob_manifest_sha256=args.manifest_sha256,
        )
    else:
        kwargs.pop("offline_gpic_blob_root", None)
        kwargs.pop("gpic_blob_manifest_sha256", None)
    return kwargs


def capture_segment(dataset: Any, *, start: int, steps: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset.set_start_step(start)
    loader = dataset.loader(int(dataset.plan_header["per_rank_bs"]))
    raw_by_token: dict[tuple[int, int], dict[str, Any]] = {}
    raw_by_token_lock = threading.Lock()
    original = type(dataset)._raw_for_item

    def observed_raw(prepared: Any) -> bytes:
        raw = original(prepared)
        planned = prepared.planned
        if planned.domain == "gpic" and start <= int(planned.step) < start + steps:
            token = (int(planned.step), int(planned.slot))
            entry = {
                "row_id": int(planned.row_id),
                "bytes": len(raw),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "native_rgb": native_rgb_identity(raw),
            }
            with raw_by_token_lock:
                if token in raw_by_token:
                    raise GateError(f"duplicate GPIC raw token: {token}")
                raw_by_token[token] = entry
        return raw

    type(dataset)._raw_for_item = staticmethod(observed_raw)
    records: list[dict[str, Any]] = []
    try:
        for expected_step in range(start, start + steps):
            batch = next(loader)
            trace = dict(dataset.last_item_trace or {})
            if trace.get("plan_step") != expected_step:
                raise GateError(f"loader step drift: {trace.get('plan_step')} != {expected_step}")
            domains = [str(value) for value in trace.get("domains", [])]
            rows = [int(value) for value in trace.get("item_rows", [])]
            if len(domains) != len(rows) or len(rows) != len(batch["prompts"]):
                raise GateError("trace/batch cardinality drift")
            gpic: list[dict[str, Any]] = []
            for slot, (domain, row_id) in enumerate(zip(domains, rows)):
                if domain != "gpic":
                    continue
                token = (expected_step, slot)
                with raw_by_token_lock:
                    raw_entry = raw_by_token.pop(token, None)
                if raw_entry is None or raw_entry["row_id"] != row_id:
                    raise GateError(f"missing/misaligned GPIC raw capture token={token}")
                raw_entry["slot"] = slot
                raw_entry["final_pixels_sha256"] = tensor_sha(batch["pixels"][slot])
                gpic.append(raw_entry)
            records.append(
                {
                    "step": expected_step,
                    "bucket_idx": int(trace["bucket_idx"]),
                    "bucket_hw": [int(value) for value in trace["bucket_hw"]],
                    "item_rows": rows,
                    "domains": domains,
                    "prompts": [str(value) for value in batch["prompts"]],
                    "cfg_keep": [bool(value) for value in batch["cfg_keep"]],
                    "pixels_sha256": tensor_sha(batch["pixels"]),
                    "gpic": gpic,
                }
            )
        with raw_by_token_lock:
            remaining_tokens = sorted(raw_by_token)
        if remaining_tokens:
            raise GateError(f"unconsumed GPIC raw tokens remain: {remaining_tokens[:8]}")
        health = (
            dict(dataset.gpic_blob_health_metrics())
            if getattr(dataset, "_gpic_payload_backend", None) == "offline_gpic_blob_v1"
            else {"backend": "source_files_v1", "source_fallback_items": 0}
        )
        return records, health
    finally:
        type(dataset)._raw_for_item = staticmethod(original)
        try:
            loader.close()
        finally:
            dataset.close()


def semantic_projection(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_bytes(records)).hexdigest()


def command_capture(args: argparse.Namespace) -> None:
    if args.backend not in BACKENDS:
        raise GateError(f"unsupported backend: {args.backend}")
    manifest = validate_artifact(args.artifact_root, args.buildtag, args.manifest_sha256)
    _catalog_raw, catalog = strict_json(args.catalog_root / "header.json")
    start = choose_start(
        args.plan_root,
        catalog,
        ranks=args.ranks,
        fresh_steps=args.fresh_steps,
        resume_offset=args.resume_offset,
        resume_steps=args.resume_steps,
    )
    if not 0 <= args.rank < args.ranks:
        raise GateError("rank is outside the simulated world")
    os.environ.update(
        RANK=str(args.rank),
        WORLD_SIZE=str(args.ranks),
        LOCAL_RANK=str(args.rank),
        LOCAL_WORLD_SIZE=str(args.ranks),
    )
    from training.dataline_offline.offline_dataset import OfflineExactDataset

    fresh_ds = OfflineExactDataset(**dataset_kwargs(args, args.backend, args.rank))
    fresh, fresh_health = capture_segment(fresh_ds, start=start, steps=args.fresh_steps)
    resume_start = start + args.resume_offset
    resume_ds = OfflineExactDataset(**dataset_kwargs(args, args.backend, args.rank))
    resumed, resume_health = capture_segment(
        resume_ds, start=resume_start, steps=args.resume_steps
    )
    expected_suffix = fresh[args.resume_offset : args.resume_offset + args.resume_steps]
    if resumed != expected_suffix:
        raise GateError("fresh suffix and exact-resume semantics differ within one backend")
    gpic_items = sum(len(record["gpic"]) for record in fresh)
    if gpic_items < 4:
        raise GateError("semantic capture did not consume enough GPIC rows")
    for health in (fresh_health, resume_health):
        if args.backend == "offline_gpic_blob_v1":
            for key in (
                "gpic_blob_short_read_failures",
                "gpic_blob_integrity_failures",
                "gpic_blob_pread_errors",
                "gpic_blob_source_fallback_items",
            ):
                if float(health.get(key, 0.0)) != 0.0:
                    raise GateError(f"blob semantic health failure: {key}={health.get(key)}")
            if int(health.get("gpic_blob_fd_cache_size", -1)) != 1024:
                raise GateError("blob semantic arm did not bind fd_cache_size=1024")
    value = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "backend": args.backend,
        "buildtag": args.buildtag,
        "artifact_root": str(args.artifact_root.resolve(strict=True)),
        "manifest_sha256": args.manifest_sha256,
        "release_sha256": manifest.get("release_sha256"),
        "rank": args.rank,
        "ranks": args.ranks,
        "start_step": start,
        "fresh_steps": args.fresh_steps,
        "resume_start_step": resume_start,
        "resume_steps": args.resume_steps,
        "fresh": fresh,
        "resumed": resumed,
        "fresh_semantic_sha256": semantic_projection(fresh),
        "resume_semantic_sha256": semantic_projection(resumed),
        "fresh_health": fresh_health,
        "resume_health": resume_health,
        "gpic_items": gpic_items,
        "bindings": {
            "config_sha256": sha256_file(args.config),
            "plan_header_sha256": sha256_file(args.plan_root / "header.json"),
            "catalog_header_sha256": sha256_file(args.catalog_root / "header.json"),
            "manifest_sha256": args.manifest_sha256,
            "dataset_source_sha256": sha256_file(
                Path(__file__).resolve().parents[2]
                / "training/dataline_offline/offline_dataset.py"
            ),
            "blob_source_sha256": sha256_file(
                Path(__file__).resolve().parents[2]
                / "training/dataline_offline/gpic_blob.py"
            ),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "job_yaml_sha256": sha256_file(args.job_yaml),
        },
    }
    atomic_json(args.output, value)
    print(
        f"GPIC_BLOB_SEMANTIC_CAPTURE_PASS backend={args.backend} rank={args.rank} "
        f"gpic_items={gpic_items} semantic={value['fresh_semantic_sha256']}"
    )


def command_compare(args: argparse.Namespace) -> None:
    source_paths = [args.input_root / f"source.rank{rank}.json" for rank in range(args.ranks)]
    blob_paths = [args.input_root / f"blob.rank{rank}.json" for rank in range(args.ranks)]
    pairs: list[dict[str, Any]] = []
    common_bindings: dict[str, Any] | None = None
    total_gpic = 0
    for rank, (source_path, blob_path) in enumerate(zip(source_paths, blob_paths)):
        source_raw, source = strict_json(source_path)
        blob_raw, blob = strict_json(blob_path)
        for value, backend in ((source, "source_files_v1"), (blob, "offline_gpic_blob_v1")):
            if (
                value.get("schema_version") != SCHEMA
                or value.get("status") != "PASS"
                or value.get("backend") != backend
                or value.get("rank") != rank
                or value.get("ranks") != args.ranks
            ):
                raise GateError(f"capture identity mismatch rank={rank} backend={backend}")
        if source["bindings"] != blob["bindings"]:
            raise GateError(f"source/blob immutable bindings differ rank={rank}")
        if common_bindings is None:
            common_bindings = source["bindings"]
        elif source["bindings"] != common_bindings:
            raise GateError(f"cross-rank immutable bindings differ rank={rank}")
        for key in (
            "buildtag",
            "artifact_root",
            "manifest_sha256",
            "release_sha256",
            "start_step",
            "fresh_steps",
            "resume_start_step",
            "resume_steps",
            "fresh_semantic_sha256",
            "resume_semantic_sha256",
            "fresh",
            "resumed",
            "gpic_items",
        ):
            if source.get(key) != blob.get(key):
                raise GateError(f"semantic A/B mismatch rank={rank} key={key}")
        total_gpic += int(source["gpic_items"])
        pairs.append(
            {
                "rank": rank,
                "source_file": {
                    "path": str(source_path.resolve(strict=True)),
                    "bytes": len(source_raw),
                    "sha256": hashlib.sha256(source_raw).hexdigest(),
                },
                "blob_file": {
                    "path": str(blob_path.resolve(strict=True)),
                    "bytes": len(blob_raw),
                    "sha256": hashlib.sha256(blob_raw).hexdigest(),
                },
                "fresh_semantic_sha256": source["fresh_semantic_sha256"],
                "resume_semantic_sha256": source["resume_semantic_sha256"],
                "gpic_items": source["gpic_items"],
            }
        )
    if total_gpic < args.ranks * 4:
        raise GateError("aggregate GPIC coverage is below the fixed minimum")
    assert common_bindings is not None
    value = {
        "schema_version": COMPARE_SCHEMA,
        "status": "PASS",
        "verdict": "SOURCE_FILES_V1_EQUALS_OFFLINE_GPIC_BLOB_V1",
        "ranks": args.ranks,
        "total_gpic_items": total_gpic,
        "comparisons": pairs,
        "bindings": common_bindings,
        "assertions": {
            "same_plan_trace": True,
            "same_raw_bytes": True,
            "same_native_decode_pixels": True,
            "same_final_pixels": True,
            "same_prompts": True,
            "same_cfg_keep": True,
            "fresh_resume_exact": True,
            "source_fallback_items": 0,
        },
    }
    atomic_json(args.output, value)
    print(
        f"GPIC_BLOB_SEMANTIC_AB_PASS ranks={args.ranks} "
        f"gpic_items={total_gpic} output={args.output}"
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    capture.add_argument("--rank", type=int, required=True)
    capture.add_argument("--ranks", type=int, default=4)
    capture.add_argument("--fresh-steps", type=int, default=8)
    capture.add_argument("--resume-offset", type=int, default=4)
    capture.add_argument("--resume-steps", type=int, default=4)
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--plan-root", type=Path, required=True)
    capture.add_argument("--catalog-root", type=Path, required=True)
    capture.add_argument("--artifact-root", type=Path, required=True)
    capture.add_argument("--buildtag", required=True)
    capture.add_argument("--manifest-sha256", required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--job-yaml", type=Path, required=True)
    capture.set_defaults(func=command_capture)
    compare = sub.add_parser("compare")
    compare.add_argument("--input-root", type=Path, required=True)
    compare.add_argument("--ranks", type=int, default=4)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(func=command_compare)
    return value


def main() -> int:
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
