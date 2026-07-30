#!/usr/bin/env python3
"""Close the TextAtlas 32KiB/no-EOI incident audit with pixel evidence.

This tool merges the sharded full-source scan, independently re-reads every
candidate, compares strict+EOI decoding with Pillow's truncated-image mode, and
maps every candidate back to all frozen-plan coordinates.  Its allowlist file
is a machine-readable *suggestion* only and is never consumed automatically.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageFile


BLOCK_RE = re.compile(r"block_(\d{6})\.npz")


class AuditError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _absolute_lexical(path: Path) -> Path:
    """Return an absolute path without following symlinks."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _require_real_directory(path: Path, label: str) -> Path:
    lexical = _absolute_lexical(path)
    try:
        resolved = lexical.resolve(strict=True)
        metadata = os.lstat(lexical)
    except (FileNotFoundError, OSError) as exc:
        raise AuditError(f"{label} is not an existing real directory: {lexical}") from exc
    if resolved != lexical or stat.S_ISLNK(metadata.st_mode):
        raise AuditError(f"{label} must not contain or be a symlink: {lexical}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise AuditError(f"{label} is not a directory: {lexical}")
    return lexical


def _bind_output(path: Path, audit_root: Path, label: str) -> Path:
    lexical = _absolute_lexical(path)
    if lexical.parent != audit_root:
        raise AuditError(
            f"{label} must be a direct child of audit_root: {lexical} != {audit_root}"
        )
    try:
        metadata = os.lstat(lexical)
    except FileNotFoundError:
        return lexical
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise AuditError(f"{label} must be absent or a regular non-symlink file: {lexical}")
    return lexical


def _verify_regular_bytes(path: Path, expected: bytes) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AuditError(f"cannot safely reopen existing output: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AuditError(f"existing output is not a regular file: {path}")
        chunks: list[bytes] = []
        remaining = len(expected) + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        observed = b"".join(chunks)
    finally:
        os.close(descriptor)
    if observed != expected:
        raise AuditError(
            f"existing partial-set member differs byte-for-byte; refusing recovery: {path}"
        )


def _publish_or_verify_bytes(path: Path, raw: bytes) -> None:
    """Publish once, or verify an identical crash-recovery member byte-for-byte."""

    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{os.urandom(8).hex()}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            _verify_regular_bytes(path, raw)
        _fsync_dir(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        finally:
            _fsync_dir(path.parent)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise AuditError(f"required field missing: {label}.{key}")
    return mapping[key]


def _strict_int(mapping: Mapping[str, Any], key: str, label: str) -> int:
    value = _required(mapping, key, label)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuditError(f"{label}.{key} must be a non-negative integer")
    return value


def validate_payload_shards(
    shards: list[dict[str, Any]],
    *,
    shard_count: int,
    plan_json_sha256: str,
    plan_bitmap_sha256: str,
    catalog_header_sha256: str,
    catalog_textatlas_sha256: str,
) -> dict[str, Any]:
    """Fail-closed validation of every payload-shard schema and provenance."""

    if shard_count <= 0 or len(shards) != shard_count:
        raise AuditError(
            f"payload shard file count drift: {len(shards)} != {shard_count}"
        )
    indexes: list[int] = []
    modes: set[str] = set()
    checked_unique = 0
    checked_files = 0
    source_failures = 0
    candidates: list[dict[str, Any]] = []
    expected_provenance = {
        "plan_json_sha256": plan_json_sha256,
        "plan_bitmap_sha256": plan_bitmap_sha256,
        "catalog_header_sha256": catalog_header_sha256,
        "catalog_textatlas_sha256": catalog_textatlas_sha256,
    }
    for position, shard in enumerate(shards):
        label = f"payload_shards[{position}]"
        if not isinstance(shard, Mapping):
            raise AuditError(f"{label} must be an object")
        if _required(shard, "status", label) != "PASS":
            raise AuditError(f"{label} status drift")
        if _required(shard, "audit", label) != "offline_arb_textatlas_planned_payloads_v2":
            raise AuditError(f"{label} audit schema drift")
        if _required(shard, "read_only", label) is not True:
            raise AuditError(f"{label} must be read_only=true")
        index = _strict_int(shard, "shard_index", label)
        if _strict_int(shard, "shard_count", label) != shard_count:
            raise AuditError(f"{label} shard_count drift")
        indexes.append(index)
        mode = _required(shard, "signature_mode", label)
        if not isinstance(mode, str):
            raise AuditError(f"{label}.signature_mode must be a string")
        modes.add(mode)
        provenance = _required(shard, "provenance", label)
        if not isinstance(provenance, Mapping):
            raise AuditError(f"{label}.provenance must be an object")
        for key, expected in expected_provenance.items():
            if _required(provenance, key, f"{label}.provenance") != expected:
                raise AuditError(f"{label} {key} provenance drift")
        summary = _required(shard, "summary", label)
        if not isinstance(summary, Mapping):
            raise AuditError(f"{label}.summary must be an object")
        checked_unique += _strict_int(summary, "planned_unique_rows_checked", f"{label}.summary")
        checked_files += _strict_int(summary, "source_files_checked", f"{label}.summary")
        shard_source_failures = _strict_int(
            summary, "strict_decode_failures", f"{label}.summary"
        )
        shard_signature_count = _strict_int(
            summary, "signature_candidates", f"{label}.summary"
        )
        shard_candidate_failures = _strict_int(
            summary,
            "strict_decode_failures_among_candidates",
            f"{label}.summary",
        )
        raw_candidates = _required(shard, "signature_candidates", label)
        if not isinstance(raw_candidates, list) or any(
            not isinstance(candidate, dict) for candidate in raw_candidates
        ):
            raise AuditError(f"{label}.signature_candidates must be a list of objects")
        if shard_signature_count != len(raw_candidates):
            raise AuditError(f"{label} signature candidate count mismatch")
        if mode == "strict_decode_failure_all":
            if not (
                shard_source_failures
                == shard_candidate_failures
                == shard_signature_count
            ):
                raise AuditError(f"{label} strict/raw/signature count mismatch")
            if any(candidate.get("strict_decode_ok") is not False for candidate in raw_candidates):
                raise AuditError(f"{label} strict candidate unexpectedly reports success")
        source_failures += shard_source_failures
        candidates.extend(raw_candidates)
    if sorted(indexes) != list(range(shard_count)):
        raise AuditError(f"payload shard index coverage drift: {indexes!r}")
    if len(modes) != 1:
        raise AuditError(f"payload shard signature-mode drift: {modes!r}")
    mode = next(iter(modes))
    if mode not in {
        "exact_32768_missing_eoi",
        "jpeg_missing_eoi",
        "strict_decode_failure_all",
    }:
        raise AuditError(f"unsupported signature mode: {mode}")
    return {
        "signature_mode": mode,
        "checked_unique": checked_unique,
        "checked_files": checked_files,
        "source_reported_strict_decode_failures": source_failures,
        "candidates": candidates,
    }


def _decode(raw: bytes, *, allow_truncated: bool) -> dict[str, Any]:
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = allow_truncated
    try:
        with Image.open(io.BytesIO(raw)) as image:
            source_format = image.format
            source_mode = image.mode
            source_size = [int(image.size[0]), int(image.size[1])]
            rgb = image.convert("RGB")
            rgb.load()
            pixels = rgb.tobytes()
            return {
                "ok": True,
                "format": source_format,
                "source_mode": source_mode,
                "decoded_size": source_size,
                "rgb_bytes": len(pixels),
                "rgb_sha256": hashlib.sha256(pixels).hexdigest(),
                "error_type": None,
                "error": None,
            }
    except Exception as exc:  # noqa: BLE001 - exact decoder evidence
        return {
            "ok": False,
            "format": None,
            "source_mode": None,
            "decoded_size": None,
            "rgb_bytes": None,
            "rgb_sha256": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def _read_candidate(candidate: dict[str, Any]) -> tuple[bytes, str]:
    path = str(candidate["image_parquet_path"])
    source_group = int(candidate["image_row_group"])
    source_row = int(candidate["image_row_in_group"])
    table = pq.ParquetFile(path).read_row_group(
        source_group, columns=["image", "image_path"], use_threads=False
    ).combine_chunks()
    if source_row < 0 or source_row >= table.num_rows:
        raise AuditError(f"candidate source row out of bounds: {path}#{source_group}:{source_row}")
    image_struct = table.column("image").chunk(0)
    raw = image_struct.field("bytes")[source_row].as_py()
    image_path = table.column("image_path").chunk(0)[source_row].as_py()
    if not isinstance(raw, bytes) or not raw:
        raise AuditError(f"candidate payload empty/non-bytes: row_id={candidate['row_id']}")
    return raw, str(image_path)


def _catalog_candidate_rows(
    catalog_path: Path,
    row_ids: Iterable[int],
    *,
    textatlas_base: int,
) -> dict[int, dict[str, Any]]:
    parquet = pq.ParquetFile(catalog_path)
    wanted = sorted(set(int(value) for value in row_ids))
    by_group: dict[int, list[int]] = {}
    cursor = 0
    group_ranges: list[tuple[int, int]] = []
    for group in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(group).num_rows
        group_ranges.append((cursor, cursor + count))
        cursor += count
    for row_id in wanted:
        local = row_id - textatlas_base
        if local < 0 or local >= cursor:
            raise AuditError(f"candidate outside TextAtlas catalog: {row_id}")
        group = next(
            index for index, (start, stop) in enumerate(group_ranges) if start <= local < stop
        )
        by_group.setdefault(group, []).append(row_id)
    columns = [
        "row_id",
        "uid",
        "domain",
        "bucket_idx",
        "width",
        "height",
        "image_path",
        "image_parquet_path",
        "image_row_group",
        "image_row_in_group",
    ]
    result: dict[int, dict[str, Any]] = {}
    for group, group_ids in by_group.items():
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        wanted_group = set(group_ids)
        for row in table.to_pylist():
            row_id = int(row["row_id"])
            if row_id in wanted_group:
                result[row_id] = row
    if set(result) != set(wanted):
        raise AuditError(f"candidate catalog coverage drift: {sorted(set(wanted) - set(result))}")
    return result


def _scan_block(
    path: Path,
    *,
    block_steps: int,
    total_steps: int,
    formal_steps: int,
    world_size: int,
    per_rank_bs: int,
    row_ids: np.ndarray,
) -> dict[int, list[dict[str, Any]]]:
    match = BLOCK_RE.fullmatch(path.name)
    if match is None:
        raise AuditError(f"bad plan block name: {path}")
    index = int(match.group(1))
    start = index * block_steps
    take = min(block_steps, total_steps - start)
    with np.load(path, allow_pickle=False) as archive:
        rows = np.asarray(archive["item_row"][:take], dtype=np.int64)
        buckets = np.asarray(archive["bucket_of_step"][:take])
    if rows.shape != (take, world_size, per_rank_bs):
        raise AuditError(f"plan block shape drift: {path}: {rows.shape}")
    positions = np.searchsorted(row_ids, rows)
    in_range = positions < row_ids.size
    safe = np.minimum(positions, row_ids.size - 1)
    matches = in_range & (row_ids[safe] == rows)
    result: dict[int, list[dict[str, Any]]] = {}
    for local_step, rank, slot in np.argwhere(matches).tolist():
        row_id = int(rows[local_step, rank, slot])
        step = start + int(local_step)
        result.setdefault(row_id, []).append(
            {
                "step": step,
                "rank": int(rank),
                "slot": int(slot),
                "bucket_idx": int(buckets[local_step]),
                "within_formal_166667": step < formal_steps,
            }
        )
    return result


def _plan_coordinates(
    plan_root: Path,
    plan: dict[str, Any],
    row_ids: Iterable[int],
    *,
    formal_steps: int,
    workers: int,
) -> dict[int, list[dict[str, Any]]]:
    ids = np.asarray(sorted(set(int(value) for value in row_ids)), dtype=np.int64)
    if ids.size == 0:
        return {}
    total_steps = int(plan["total_steps"])
    block_steps = int(plan["block_steps"])
    blocks = (total_steps + block_steps - 1) // block_steps
    paths = [plan_root / "blocks" / f"block_{index:06d}.npz" for index in range(blocks)]
    result = {int(value): [] for value in ids.tolist()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _scan_block,
                path,
                block_steps=block_steps,
                total_steps=total_steps,
                formal_steps=formal_steps,
                world_size=int(plan["world_size"]),
                per_rank_bs=int(plan["per_rank_bs"]),
                row_ids=ids,
            )
            for path in paths
        ]
        for future in concurrent.futures.as_completed(futures):
            for row_id, coordinates in future.result().items():
                result[row_id].extend(coordinates)
    for coordinates in result.values():
        coordinates.sort(key=lambda item: (item["step"], item["rank"], item["slot"]))
    return result


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument(
        "--plan-audit-root",
        type=Path,
        default=None,
        help="root containing plan_rows.json; defaults to --audit-root",
    )
    parser.add_argument("--catalog-root", required=True, type=Path)
    parser.add_argument("--plan-root", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--formal-steps", type=int, default=166667)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allowlist-output", required=True, type=Path)
    parser.add_argument(
        "--neutral-output",
        type=Path,
        default=None,
        help=(
            "optional suggestion-only manifest for exact neutral replacement of "
            "strict-decode failures that are not eligible for JPEG EOI repair"
        ),
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=None,
        help="optional immutable SHA/provenance receipt for all published evidence",
    )
    parser.add_argument(
        "--success-output",
        type=Path,
        default=None,
        help="optional immutable completion marker; requires --receipt-output",
    )
    return parser.parse_args()


def main() -> int:
    args = _args()
    audit_root = _require_real_directory(args.audit_root, "audit_root")
    plan_audit_root = (
        args.plan_audit_root.expanduser().resolve(strict=True)
        if args.plan_audit_root is not None
        else audit_root
    )
    catalog_root = args.catalog_root.expanduser().resolve(strict=True)
    plan_root = args.plan_root.expanduser().resolve(strict=True)
    output = _bind_output(args.output, audit_root, "output")
    allowlist_output = _bind_output(
        args.allowlist_output, audit_root, "allowlist_output"
    )
    neutral_output = (
        _bind_output(args.neutral_output, audit_root, "neutral_output")
        if args.neutral_output is not None
        else None
    )
    receipt_output = (
        _bind_output(args.receipt_output, audit_root, "receipt_output")
        if args.receipt_output is not None
        else None
    )
    success_output = (
        _bind_output(args.success_output, audit_root, "success_output")
        if args.success_output is not None
        else None
    )
    if (receipt_output is None) != (success_output is None):
        raise AuditError("--receipt-output and --success-output are all-or-none")
    output_paths = [output, allowlist_output]
    if neutral_output is not None:
        output_paths.append(neutral_output)
    if receipt_output is not None and success_output is not None:
        output_paths.extend((receipt_output, success_output))
    if len(set(output_paths)) != len(output_paths):
        raise AuditError("all output paths must be distinct")

    plan_rows_path = plan_audit_root / "plan_rows.json"
    plan_rows = _json(plan_rows_path)
    if plan_rows.get("status") != "PASS":
        raise AuditError("plan-row audit status drift")
    if plan_rows.get("audit") != "offline_arb_textatlas_plan_rows_v1":
        raise AuditError("plan-row audit schema drift")
    if plan_rows.get("read_only") is not True:
        raise AuditError("plan-row audit must be read_only=true")
    plan_json_sha = _sha256(plan_rows_path)
    bitmap_section = plan_rows.get("bitmap")
    if not isinstance(bitmap_section, Mapping):
        raise AuditError("plan-row audit bitmap section missing")
    bitmap_path = Path(str(_required(bitmap_section, "path", "plan_rows.bitmap"))).resolve(
        strict=True
    )
    expected_bitmap_path = (plan_audit_root / "plan_rows_bitmap.npz").resolve(
        strict=True
    )
    if bitmap_path != expected_bitmap_path:
        raise AuditError("plan-row audit bitmap path drift")
    plan_bitmap_sha = _sha256(bitmap_path)
    if plan_bitmap_sha != _required(bitmap_section, "sha256", "plan_rows.bitmap"):
        raise AuditError("plan-row audit bitmap SHA drift")

    shard_paths = [audit_root / f"payload_shard_{index}.json" for index in range(args.shard_count)]
    shards = [_json(path) for path in shard_paths]
    catalog_header_path = catalog_root / "header.json"
    catalog_header = _json(catalog_header_path)
    plan_header_path = plan_root / "header.json"
    plan_header = _json(plan_header_path)
    catalog_header_sha = _sha256(catalog_header_path)
    if catalog_header_sha != plan_rows["catalog"]["header_sha256"]:
        raise AuditError("catalog header changed after plan scan")
    if _sha256(plan_header_path) != plan_rows["plan"]["header_sha256"]:
        raise AuditError("plan header changed after plan scan")
    catalog_files = catalog_header.get("files")
    if not isinstance(catalog_files, Mapping) or not isinstance(
        catalog_files.get("textatlas"), Mapping
    ):
        raise AuditError("catalog TextAtlas file entry missing")
    textatlas_entry = catalog_files["textatlas"]
    catalog_textatlas_path = catalog_root / str(
        _required(textatlas_entry, "name", "catalog.files.textatlas")
    )
    catalog_textatlas_sha = _sha256(catalog_textatlas_path)
    if catalog_textatlas_sha != _required(
        textatlas_entry, "sha256", "catalog.files.textatlas"
    ):
        raise AuditError("live TextAtlas catalog file SHA drift")

    shard_validation = validate_payload_shards(
        shards,
        shard_count=args.shard_count,
        plan_json_sha256=plan_json_sha,
        plan_bitmap_sha256=plan_bitmap_sha,
        catalog_header_sha256=catalog_header_sha,
        catalog_textatlas_sha256=catalog_textatlas_sha,
    )
    signature_mode = str(shard_validation["signature_mode"])
    if not isinstance(plan_rows.get("textatlas"), Mapping):
        raise AuditError("plan-row audit TextAtlas summary missing")
    expected_unique = _strict_int(
        plan_rows["textatlas"], "total_plan_unique_rows", "plan_rows.textatlas"
    )
    checked_unique = int(shard_validation["checked_unique"])
    checked_files = int(shard_validation["checked_files"])
    if checked_unique != expected_unique or checked_files != 595:
        raise AuditError(
            f"payload shard coverage drift rows={checked_unique}/{expected_unique}, "
            f"files={checked_files}/595"
        )
    raw_candidates = list(shard_validation["candidates"])
    raw_candidates.sort(key=lambda item: int(item["row_id"]))
    row_ids = [int(item["row_id"]) for item in raw_candidates]
    if len(row_ids) != len(set(row_ids)):
        raise AuditError("duplicate candidate row across shards")

    textatlas_base = int(plan_rows["catalog"]["textatlas_row_base"])
    catalog_rows = _catalog_candidate_rows(
        catalog_textatlas_path,
        row_ids,
        textatlas_base=textatlas_base,
    )
    coordinates = _plan_coordinates(
        plan_root,
        plan_header,
        row_ids,
        formal_steps=args.formal_steps,
        workers=args.workers,
    )

    candidates: list[dict[str, Any]] = []
    for prior in raw_candidates:
        row_id = int(prior["row_id"])
        catalog_row = catalog_rows[row_id]
        locator_fields = (
            "uid",
            "image_path",
            "image_parquet_path",
            "image_row_group",
            "image_row_in_group",
        )
        for field in locator_fields:
            if str(prior[field]) != str(catalog_row[field]):
                raise AuditError(f"candidate/catalog locator mismatch row_id={row_id}, field={field}")
        raw, observed_image_path = _read_candidate(prior)
        if observed_image_path != str(catalog_row["image_path"]):
            raise AuditError(f"candidate source image_path drift row_id={row_id}")
        raw_sha = hashlib.sha256(raw).hexdigest()
        if signature_mode == "exact_32768_missing_eoi":
            signature_ok = len(raw) == 32768 and not raw.endswith(b"\xff\xd9")
        elif signature_mode == "jpeg_missing_eoi":
            signature_ok = raw.startswith(b"\xff\xd8\xff") and not raw.endswith(b"\xff\xd9")
        else:
            signature_ok = True
        if raw_sha != prior["payload_sha256"] or not signature_ok:
            raise AuditError(f"candidate raw identity/signature drift row_id={row_id}")
        strict_raw = _decode(raw, allow_truncated=False)
        if signature_mode == "strict_decode_failure_all" and strict_raw["ok"]:
            raise AuditError(f"full-decode candidate no longer fails strict decode row_id={row_id}")
        strict_append_eoi = _decode(raw + b"\xff\xd9", allow_truncated=False)
        load_truncated = _decode(raw, allow_truncated=True)
        pixel_equal = bool(
            strict_append_eoi["ok"]
            and load_truncated["ok"]
            and strict_append_eoi["rgb_sha256"] == load_truncated["rgb_sha256"]
            and strict_append_eoi["decoded_size"] == load_truncated["decoded_size"]
        )
        decoded_matches_catalog = bool(
            strict_append_eoi["ok"]
            and strict_append_eoi["decoded_size"]
            == [int(catalog_row["width"]), int(catalog_row["height"])]
        )
        eligible = bool(
            not strict_raw["ok"]
            and raw.startswith(b"\xff\xd8\xff")
            and not raw.endswith(b"\xff\xd9")
            and strict_append_eoi["ok"]
            and load_truncated["ok"]
            and pixel_equal
            and decoded_matches_catalog
            and strict_append_eoi["format"] == "JPEG"
        )
        row_coordinates = coordinates[row_id]
        candidates.append(
            {
                "row_id": row_id,
                "uid": str(catalog_row["uid"]),
                "domain": str(catalog_row["domain"]),
                "bucket_idx": int(catalog_row["bucket_idx"]),
                "catalog_width": int(catalog_row["width"]),
                "catalog_height": int(catalog_row["height"]),
                "locator": {
                    "image_path": str(catalog_row["image_path"]),
                    "image_parquet_path": str(catalog_row["image_parquet_path"]),
                    "image_row_group": int(catalog_row["image_row_group"]),
                    "image_row_in_group": int(catalog_row["image_row_in_group"]),
                },
                "payload_bytes": len(raw),
                "payload_sha256": raw_sha,
                "head_hex": raw[:16].hex(),
                "tail_hex": raw[-16:].hex(),
                "has_jpeg_soi": raw.startswith(b"\xff\xd8\xff"),
                "has_jpeg_eoi": raw.endswith(b"\xff\xd9"),
                "strict_raw": strict_raw,
                "strict_append_eoi": strict_append_eoi,
                "load_truncated_images": load_truncated,
                "append_eoi_equals_load_truncated_pixels": pixel_equal,
                "decoded_size_matches_catalog": decoded_matches_catalog,
                "eligible_for_explicit_append_eoi_allowlist": eligible,
                "total_plan_exposures": len(row_coordinates),
                "formal_plan_exposures": sum(
                    int(item["within_formal_166667"]) for item in row_coordinates
                ),
                "first_formal_coordinate": next(
                    (item for item in row_coordinates if item["within_formal_166667"]), None
                ),
                "coordinates": row_coordinates,
            }
        )
    candidates.sort(
        key=lambda item: (
            item["first_formal_coordinate"]["step"]
            if item["first_formal_coordinate"] is not None
            else 10**18,
            item["row_id"],
        )
    )
    eligible_rows = [
        item for item in candidates if item["eligible_for_explicit_append_eoi_allowlist"]
    ]
    neutral_rows = [
        item
        for item in candidates
        if (not item["strict_raw"]["ok"])
        and (not item["eligible_for_explicit_append_eoi_allowlist"])
    ]
    summary = {
        "catalog_textatlas_rows_checked": checked_unique,
        "source_files_checked": checked_files,
        "source_row_groups_checked": sum(
            _strict_int(item["summary"], "source_row_groups_checked", "payload_shard.summary")
            for item in shards
        ),
        "exact_32768_rows": sum(
            _strict_int(item["summary"], "exact_32768_rows", "payload_shard.summary")
            for item in shards
        ),
        "exact_32768_with_eoi_rows": sum(
            _strict_int(
                item["summary"],
                "exact_32768_with_eoi_rows",
                "payload_shard.summary",
            )
            for item in shards
        ),
        "jpeg_rows": sum(
            _strict_int(item["summary"], "jpeg_rows", "payload_shard.summary")
            for item in shards
        ),
        "jpeg_missing_eoi_rows": sum(
            _strict_int(
                item["summary"], "jpeg_missing_eoi_rows", "payload_shard.summary"
            )
            for item in shards
        ),
        "strict_decode_rows": sum(
            _strict_int(item["summary"], "strict_decode_rows", "payload_shard.summary")
            for item in shards
        ),
        "source_reported_strict_decode_failures": int(
            shard_validation["source_reported_strict_decode_failures"]
        ),
        "signature_mode": signature_mode,
        "signature_candidates": len(candidates),
        "raw_strict_decode_failures": sum(not item["strict_raw"]["ok"] for item in candidates),
        "noneligible_raw_strict_decode_failures": sum(
            (not item["strict_raw"]["ok"])
            and (not item["eligible_for_explicit_append_eoi_allowlist"])
            for item in candidates
        ),
        "raw_strict_failure_error_types": dict(
            sorted(
                Counter(
                    str(item["strict_raw"]["error_type"])
                    for item in candidates
                    if not item["strict_raw"]["ok"]
                ).items()
            )
        ),
        "append_eoi_strict_decode_successes": sum(
            item["strict_append_eoi"]["ok"] for item in candidates
        ),
        "append_eoi_truncated_pixel_equivalences": sum(
            item["append_eoi_equals_load_truncated_pixels"] for item in candidates
        ),
        "eligible_explicit_allowlist_rows": len(eligible_rows),
        "formal_candidate_exposures": sum(item["formal_plan_exposures"] for item in candidates),
        "all_plan_candidate_exposures": sum(item["total_plan_exposures"] for item in candidates),
    }
    if signature_mode == "strict_decode_failure_all" and not (
        summary["source_reported_strict_decode_failures"]
        == summary["raw_strict_decode_failures"]
        == summary["signature_candidates"]
    ):
        raise AuditError(
            "strict/raw/signature aggregate mismatch: "
            f"source={summary['source_reported_strict_decode_failures']}, "
            f"raw={summary['raw_strict_decode_failures']}, "
            f"candidates={summary['signature_candidates']}"
        )
    result = {
        "status": "PASS",
        "audit": {
            "exact_32768_missing_eoi": "offline_arb_textatlas_integrity_close_v1",
            "jpeg_missing_eoi": "offline_arb_textatlas_jpeg_missing_eoi_close_v1",
            "strict_decode_failure_all": "offline_arb_textatlas_full_strict_decode_close_v1",
        }[signature_mode],
        "read_only": True,
        "plan_contract": {
            "plan_horizon": int(plan_header["total_steps"]),
            "formal_steps": int(args.formal_steps),
            "world_size": int(plan_header["world_size"]),
            "per_rank_bs": int(plan_header["per_rank_bs"]),
            "global_batch": int(plan_header["world_size"])
            * int(plan_header["per_rank_bs"]),
            "formal_exposures": int(args.formal_steps)
            * int(plan_header["world_size"])
            * int(plan_header["per_rank_bs"]),
        },
        "scope": {
            "exact_32768_missing_eoi": (
                f"all {expected_unique:,} planned TextAtlas rows; "
                "exact 32768-byte/no-JPEG-EOI signature"
            ),
            "jpeg_missing_eoi": (
                f"all {expected_unique:,} planned TextAtlas rows; JPEG SOI "
                "present and JPEG EOI absent, any payload length"
            ),
            "strict_decode_failure_all": (
                f"all {expected_unique:,} planned TextAtlas rows; "
                "runtime-equivalent strict Pillow RGB conversion"
            ),
        }[signature_mode],
        "provenance": {
            "plan_id": plan_header.get("plan_id"),
            "plan_hash": plan_header.get("plan_hash"),
            "plan_header_sha256": _sha256(plan_header_path),
            "catalog_hash": catalog_header.get("catalog_hash"),
            "catalog_header_sha256": catalog_header_sha,
            "catalog_textatlas_sha256": catalog_textatlas_sha,
            "plan_rows_json_sha256": plan_json_sha,
            "plan_bitmap_sha256": plan_bitmap_sha,
            "payload_shards": [
                {"path": str(path), "sha256": _sha256(path)} for path in shard_paths
            ],
            "stage_identity": {
                key: plan_header.get(key)
                for key in (
                    "stage",
                    "image_size",
                    "eligibility",
                    "requires_nl",
                    "anime_nl_filter_applied",
                )
            },
        },
        "summary": summary,
        "candidates": candidates,
    }
    allowlist = {
        "status": "SUGGESTION_ONLY_NOT_RUNTIME_ACTIVE",
        "signature_mode": signature_mode,
        "policy": "append exactly one JPEG EOI marker before existing strict decode",
        "generic_signature_fallback_forbidden": True,
        "required_identity": {
            "plan_id": plan_header.get("plan_id"),
            "plan_hash": plan_header.get("plan_hash"),
            "plan_header_sha256": _sha256(plan_header_path),
            "catalog_hash": catalog_header.get("catalog_hash"),
            "catalog_header_sha256": _sha256(catalog_header_path),
            "integrity_audit_output": str(output),
        },
        "rows": [
            {
                "row_id": item["row_id"],
                "uid": item["uid"],
                "domain": item["domain"],
                "bucket_idx": item["bucket_idx"],
                "catalog_width": item["catalog_width"],
                "catalog_height": item["catalog_height"],
                **item["locator"],
                "payload_bytes": item["payload_bytes"],
                "payload_sha256": item["payload_sha256"],
                "appended_bytes_hex": "ffd9",
                "expected_rgb_sha256": item["strict_append_eoi"]["rgb_sha256"],
                "expected_decoded_size": item["strict_append_eoi"]["decoded_size"],
                "formal_plan_exposures": item["formal_plan_exposures"],
                "coordinates": [
                    coordinate
                    for coordinate in item["coordinates"]
                    if coordinate["within_formal_166667"]
                ],
            }
            for item in eligible_rows
        ],
    }
    neutral_suggestion = None
    if neutral_output is not None:
        if signature_mode != "strict_decode_failure_all":
            raise AuditError(
                "--neutral-output is only valid for strict_decode_failure_all"
            )
        neutral_suggestion = {
            "status": "SUGGESTION_ONLY_NOT_RUNTIME_ACTIVE",
            "action": "neutral_skip_v1",
            "policy": (
                "after exact row/catalog/payload/target binding, bypass Pillow and "
                "emit a target-sized constant uint8 RGB image"
            ),
            "semantic_effect": {
                "image": "np.full((target_height,target_width,3),128,dtype=uint8)",
                "prompt": "",
                "cfg_keep": False,
                "plan_slot_and_batch_shape_preserved": True,
            },
            "generic_error_fallback_forbidden": True,
            "required_identity": {
                "plan_id": plan_header.get("plan_id"),
                "plan_hash": plan_header.get("plan_hash"),
                "plan_header_sha256": _sha256(plan_header_path),
                "catalog_hash": catalog_header.get("catalog_hash"),
                "catalog_header_sha256": _sha256(catalog_header_path),
                "integrity_audit_output": str(output),
                "plan_rows_json_sha256": _sha256(plan_rows_path),
                "plan_bitmap_sha256": plan_bitmap_sha,
                "payload_shards": [
                    {"path": str(path), "sha256": _sha256(path)}
                    for path in shard_paths
                ],
                "complete_shard_indices": list(range(args.shard_count)),
            },
            "partition": {
                "strict_failure_rows": len(candidates),
                "jpeg_eoi_repair_rows": len(eligible_rows),
                "neutral_skip_rows": len(neutral_rows),
                "jpeg_eoi_repair_row_ids": [
                    int(item["row_id"]) for item in eligible_rows
                ],
            },
            "rows": [
                {
                    "action": "neutral_skip_v1",
                    "row_id": item["row_id"],
                    "uid": item["uid"],
                    "domain": item["domain"],
                    "bucket_idx": item["bucket_idx"],
                    "width": item["catalog_width"],
                    "height": item["catalog_height"],
                    **item["locator"],
                    "raw_bytes": item["payload_bytes"],
                    "raw_sha256": item["payload_sha256"],
                    "target_height": int(plan_header["buckets"][item["bucket_idx"]][0]),
                    "target_width": int(plan_header["buckets"][item["bucket_idx"]][1]),
                    "neutral_u8": 128,
                    "observed_error_type": item["strict_raw"]["error_type"],
                    "observed_error": item["strict_raw"]["error"],
                    "decoded_pixel_count": (
                        int(item["catalog_width"]) * int(item["catalog_height"])
                    ),
                    "formal_plan_exposures": item["formal_plan_exposures"],
                    "total_plan_exposures": item["total_plan_exposures"],
                    "first_formal_coordinate": item["first_formal_coordinate"],
                    "coordinates": [
                        coordinate
                        for coordinate in item["coordinates"]
                        if coordinate["within_formal_166667"]
                    ],
                }
                for item in neutral_rows
            ],
        }
    values = [(output, result), (allowlist_output, allowlist)]
    if neutral_output is not None:
        if neutral_suggestion is None:
            raise AuditError(
                "internal contract error: strict_decode_failure_all requires neutral suggestion"
            )
        values.append((neutral_output, neutral_suggestion))
    serialized_values = [(path, value, _json_bytes(value)) for path, value in values]
    for path, _value, raw in serialized_values:
        _publish_or_verify_bytes(path, raw)
    if receipt_output is not None and success_output is not None:
        receipt = {
            "status": "PASS",
            "schema_version": "offline_arb_textatlas_integrity_evidence_receipt_v1",
            "read_only": True,
            "runtime_policy_activated": False,
            "artifacts": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "status": value.get("status"),
                }
                for path, value, raw in serialized_values
            ],
        }
        receipt_raw = _json_bytes(receipt)
        _publish_or_verify_bytes(receipt_output, receipt_raw)
        receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
        _publish_or_verify_bytes(
            success_output,
            f"PASS|SUGGESTION_ONLY|{receipt_sha}\n".encode("ascii"),
        )
    print(json.dumps({"status": "PASS", **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
