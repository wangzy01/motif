#!/usr/bin/env python3
"""Deterministic full-plan/runtime-registry closure for Stage512a control RNG-fix v1.

``scan`` verifies all 123 plan blocks and every one of the 256M slots, joins every
planned MONET/GPIC (row, caption-band) back to the frozen catalog, invokes the real
runtime binders and loader selector, and emits a deterministic receipt. ``publish``
requires two byte-identical independent scans and installs the receipt plus a commit
marker with the repository's Lustre-safe immutable-evidence publisher. ``verify-root``
is the lightweight consumer used by launch preflight; it never re-scans 256M slots.

No correctness gate in this file uses ``assert``: optimized Python is rejected and all
drift is converted into an explicit non-zero failure.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import io
import json
import math
import os
import stat
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


POLICY_SCHEMA = "stage512a_continue_control_registry_closure_policy_v2"
REPORT_SCHEMA = "stage512a_continue_control_registry_closure_v2"
SUCCESS_PREFIX = "PASS|REGISTRY_CLOSURE|"
BLOCK_MEMBERS = frozenset(
    {"bucket_of_step", "item_row", "caption_band", "domain_counts"}
)
DOMAINS = ("anime", "monet_real", "gpic", "synthetic", "textatlas")
REAL_DOMAINS = ("monet_real", "gpic")


class RegistryClosureError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise RegistryClosureError(message)


def require_raises(
    expected: type[BaseException], function: Any, kwargs: Mapping[str, Any], label: str
) -> None:
    try:
        function(**dict(kwargs))
    except expected:
        return
    except BaseException as exc:
        raise RegistryClosureError(
            f"{label} raised unexpected {type(exc).__name__}: {exc}"
        ) from exc
    raise RegistryClosureError(f"{label} did not fail closed")


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(type(key) is str, "JSON key is not an exact string")
        require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def decode_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RegistryClosureError(f"non-finite JSON token in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryClosureError(f"invalid UTF-8 JSON: {label}") from exc
    require(type(value) is dict, f"JSON root is not an object: {label}")
    return value


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def bytes_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_sha256(value: Any) -> str:
    return bytes_sha256(canonical_bytes(value))


def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_nlink),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def ordinary_file(path: Path, *, mode: int | None = None) -> os.stat_result:
    info = path.lstat()
    require(not path.is_symlink(), f"symlink forbidden: {path}")
    require(stat.S_ISREG(info.st_mode), f"not a regular file: {path}")
    require(info.st_nlink == 1, f"file link closure drift: {path}")
    require(info.st_size > 0, f"empty file forbidden: {path}")
    if mode is not None:
        require(stat.S_IMODE(info.st_mode) == mode, f"file mode drift: {path}")
    return info


def real_directory(path: Path, *, mode: int | None = None) -> os.stat_result:
    info = path.lstat()
    require(not path.is_symlink(), f"directory symlink forbidden: {path}")
    require(stat.S_ISDIR(info.st_mode), f"not a directory: {path}")
    if mode is not None:
        require(stat.S_IMODE(info.st_mode) == mode, f"directory mode drift: {path}")
    return info


def read_stable(path: Path, *, mode: int | None = None) -> bytes:
    before = ordinary_file(path, mode=mode)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        require(_fingerprint(opened) == _fingerprint(before), f"file changed before open: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 8 << 20)
            if not block:
                break
            chunks.append(block)
        raw = b"".join(chunks)
        after_fd = os.fstat(fd)
        after_name = path.lstat()
        require(
            _fingerprint(before) == _fingerprint(after_fd) == _fingerprint(after_name),
            f"file changed during stable read: {path}",
        )
        return raw
    finally:
        os.close(fd)


def stable_file_sha256(path: Path) -> tuple[str, tuple[int, ...]]:
    before = ordinary_file(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(fd)
        require(_fingerprint(opened) == _fingerprint(before), f"file changed before hash: {path}")
        while True:
            block = os.read(fd, 8 << 20)
            if not block:
                break
            digest.update(block)
        after_fd = os.fstat(fd)
        after_name = path.lstat()
        require(
            _fingerprint(before) == _fingerprint(after_fd) == _fingerprint(after_name),
            f"file changed during hash: {path}",
        )
        return digest.hexdigest(), _fingerprint(before)
    finally:
        os.close(fd)


def require_fingerprint(path: Path, expected: tuple[int, ...], label: str) -> None:
    require(_fingerprint(path.lstat()) == expected, f"{label} changed after verification: {path}")


def write_exclusive(path: Path, raw: bytes) -> None:
    require(path.is_absolute(), "scan output must be absolute")
    real_directory(path.parent)
    require(not path.exists() and not path.is_symlink(), f"refusing output overwrite: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o400)
    try:
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            require(written > 0, f"short write: {path}")
            offset += written
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    require(read_stable(path, mode=0o444) == raw, f"output readback drift: {path}")


def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} key-set drift: {sorted(set(value) ^ expected)}")


def load_policy(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = read_stable(path)
    policy = decode_json(raw, "closure policy")
    exact_keys(
        policy,
        {
            "schema", "repo_root", "source_files", "plan", "catalog",
            "weighted_contract", "control", "expected_hits", "binders",
            "legacy_bridge", "publication",
        },
        "policy",
    )
    require(policy["schema"] == POLICY_SCHEMA, "policy schema drift")
    require(type(policy["source_files"]) is list and policy["source_files"], "source_files drift")
    require(set(policy["catalog"]["files"]) == set(DOMAINS), "catalog domain set drift in policy")
    require(Path(policy["repo_root"]).is_absolute(), "repo_root must be absolute")
    return policy, raw


def import_symbol(module_name: str, symbol: str) -> tuple[Any, Any]:
    module = importlib.import_module(module_name)
    require(hasattr(module, symbol), f"missing runtime symbol: {module_name}:{symbol}")
    return module, getattr(module, symbol)


def git_capture(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace')}")
    return result.stdout


def source_authorities(policy: Mapping[str, Any], policy_path: Path) -> dict[str, Any]:
    repo = Path(policy["repo_root"])
    real_directory(repo)
    files: dict[str, str] = {}
    for relative in policy["source_files"]:
        require(type(relative) is str and relative and not Path(relative).is_absolute(), "bad source path")
        path = repo / relative
        digest, _ = stable_file_sha256(path)
        files[relative] = digest
    require(str(policy_path) == str(repo / policy["source_files"][0]), "policy path/source list mismatch")
    head = git_capture(repo, "rev-parse", "HEAD").decode("ascii").strip()
    # The release is assembled concurrently in one worktree.  Unrelated formal/RNG
    # edits must not make two otherwise identical full scans non-deterministic, while
    # every source that can affect this closure remains explicitly status-bound.
    status = git_capture(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *policy["source_files"],
    )
    return {
        "repo_root": str(repo),
        "repo_head_at_scan": head,
        "scoped_sources_clean_at_scan": not bool(status),
        "scoped_source_status_sha256": bytes_sha256(status),
        "files": files,
    }


def domain_bounds(catalog: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    cursor = 0
    result: dict[str, tuple[int, int]] = {}
    for domain in DOMAINS:
        rows = int(catalog["files"][domain]["rows"])
        result[domain] = (cursor, cursor + rows)
        cursor += rows
    require(cursor == int(catalog["row_count"]), "catalog row-count closure drift")
    return result


def decode_domain(bounds: Mapping[str, tuple[int, int]], row_id: int) -> str:
    require(type(row_id) is int, "runtime binder row id is not exact int")
    for domain, (lo, hi) in bounds.items():
        if lo <= row_id < hi:
            return domain
    raise RegistryClosureError(f"row outside catalog: {row_id}")


def catalog_record(
    catalog_root: Path,
    catalog_header: Mapping[str, Any],
    bounds: Mapping[str, tuple[int, int]],
    row_id: int,
    columns: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    domain = decode_domain(bounds, row_id)
    lo, _ = bounds[domain]
    local = row_id - lo
    path = catalog_root / str(catalog_header["files"][domain]["name"])
    parquet = pq.ParquetFile(path)
    cursor = 0
    for group in range(parquet.num_row_groups):
        count = int(parquet.metadata.row_group(group).num_rows)
        if local < cursor + count:
            table = parquet.read_row_group(group, columns=list(columns), use_threads=False)
            offset = local - cursor
            return domain, {name: table[name][offset].as_py() for name in columns}
        cursor += count
    raise RegistryClosureError(f"cannot locate catalog row: {row_id}")


@dataclasses.dataclass(frozen=True)
class RealJoin:
    domain: str
    row_base: int
    served: np.ndarray
    codes: np.ndarray
    weights: np.ndarray
    fingerprint: tuple[int, ...]
    path: Path


def arrow_numpy(table: Any, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    array = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    result = np.asarray(array)
    require(result.dtype == dtype, f"catalog dtype drift: {name}: {result.dtype} != {dtype}")
    return result


def load_real_join(
    domain: str,
    catalog_root: Path,
    catalog_header: Mapping[str, Any],
    bounds: Mapping[str, tuple[int, int]],
    verified_fingerprint: tuple[int, ...],
) -> RealJoin:
    info = catalog_header["files"][domain]
    path = catalog_root / str(info["name"])
    require_fingerprint(path, verified_fingerprint, f"{domain} catalog")
    rows = int(info["rows"])
    row_base = int(bounds[domain][0])
    served = np.empty(rows, dtype=np.uint8)
    codes = np.empty((rows, 4), dtype=np.int8)
    weights = np.empty(rows, dtype=np.float64)
    parquet = pq.ParquetFile(path)
    cursor = 0
    columns = [
        "row_id", "served_mask", "base_weight",
        "band_captioner_t1", "band_captioner_t2",
        "band_captioner_t3", "band_captioner_t4",
    ]
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        count = len(table)
        ids = arrow_numpy(table, "row_id", np.dtype(np.int64))
        expected_ids = np.arange(row_base + cursor, row_base + cursor + count, dtype=np.int64)
        require(np.array_equal(ids, expected_ids), f"{domain} row_id physical order drift")
        served[cursor:cursor + count] = arrow_numpy(table, "served_mask", np.dtype(np.uint8))
        weights[cursor:cursor + count] = arrow_numpy(table, "base_weight", np.dtype(np.float64))
        for band in range(4):
            codes[cursor:cursor + count, band] = arrow_numpy(
                table, f"band_captioner_t{band + 1}", np.dtype(np.int8)
            )
        cursor += count
    require(cursor == rows, f"{domain} catalog join row closure drift")
    require_fingerprint(path, verified_fingerprint, f"{domain} catalog")
    for array in (served, codes, weights):
        array.setflags(write=False)
    return RealJoin(domain, row_base, served, codes, weights, verified_fingerprint, path)


def coordinates_from_policy(entry: Mapping[str, Any]) -> list[dict[str, int]]:
    result = []
    for value in entry["coordinates"]:
        exact_keys(value, {"step", "rank", "slot"}, "coordinate")
        coordinate = {name: int(value[name]) for name in ("step", "rank", "slot")}
        require(all(type(value[name]) is int for name in coordinate), "coordinate type drift")
        result.append(coordinate)
    require(result == sorted(result, key=lambda x: (x["step"], x["rank"], x["slot"])), "coordinate order drift")
    require(len({(x["step"], x["rank"], x["slot"]) for x in result}) == len(result), "duplicate coordinate")
    return result


def binder_closure(
    policy: Mapping[str, Any],
    header: Mapping[str, Any],
    catalog_header: Mapping[str, Any],
    bounds: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    binders = policy["binders"]
    common = {
        "stage": str(header["stage"]),
        "eligibility": str(header["eligibility"]),
        "plan_id": str(header["plan_id"]),
        "catalog_hash": str(catalog_header["catalog_hash"]),
        "plan_header_sha256": str(policy["plan"]["header_sha256"]),
        "catalog_header_sha256": str(policy["catalog"]["header_sha256"]),
        "decode_domain": lambda row_id: decode_domain(bounds, row_id),
    }
    with_hash = {**common, "plan_hash": str(header["plan_hash"])}

    image_module, image_binder = import_symbol(
        binders["image"]["module"], binders["image"]["function"]
    )
    image = image_binder(**with_hash)
    expected_text = policy["expected_hits"]["textatlas"]
    require(set(image) == {int(key) for key in expected_text}, "fresh image binder row-set drift")
    image_rows: dict[str, Any] = {}
    for row_id, repair in image.items():
        expected = expected_text[str(row_id)]
        require(repair.row_id == row_id, f"image binder key/row drift: {row_id}")
        require(repair.uid == expected["uid"] and repair.bucket_idx == expected["bucket_idx"], f"image binder definition drift: {row_id}")
        image_rows[str(row_id)] = {
            "uid": repair.uid,
            "bucket_idx": int(repair.bucket_idx),
            "raw_sha256": repair.raw_sha256,
            "repaired_sha256": repair.repaired_sha256,
            "pixel_sha256": repair.pixel_sha256,
        }

    anime_cfg = binders["anime"]
    anime_module, anime_binder = import_symbol(anime_cfg["module"], anime_cfg["function"])
    anime = anime_binder(**with_hash)
    mapping = getattr(anime_module, anime_cfg["mapping_symbol"], None)
    closure = getattr(anime_module, anime_cfg["closure_symbol"], None)
    require(anime is mapping, "fresh Anime binder did not return its frozen mapping object")
    require(tuple(closure) == (6, 125000, 64, 32), "fresh Anime runtime closure drift")
    expected_anime = policy["expected_hits"]["anime"]
    require(set(anime) == {int(key) for key in expected_anime}, "fresh Anime binder row-set drift")
    anime_rows: dict[str, Any] = {}
    for row_id, exception in anime.items():
        expected = expected_anime[str(row_id)]
        coordinates = sorted(
            ({"step": int(c.step), "rank": int(c.rank), "slot": int(c.slot)} for c in exception.coordinates),
            key=lambda x: (x["step"], x["rank"], x["slot"]),
        )
        require(coordinates == coordinates_from_policy(expected), f"Anime binder coordinate drift: {row_id}")
        require(exception.uid == expected["uid"] and exception.bucket_idx == expected["bucket_idx"], f"Anime binder definition drift: {row_id}")
        anime_rows[str(row_id)] = {
            "uid": exception.uid,
            "bucket_idx": int(exception.bucket_idx),
            "raw_sha256": exception.raw_sha256,
            "decoded_rgb_sha256": exception.decoded_rgb_sha256,
            "output_rgb_sha256": exception.output_rgb_sha256,
            "coordinates": coordinates,
        }

    caption_cfg = binders["caption"]
    caption_module, caption_binder = import_symbol(caption_cfg["module"], caption_cfg["function"])
    caption = caption_binder(**common)
    require(not caption, "fresh plan unexpectedly received a caption exception profile")

    selector_cfg = binders["loader_selector"]
    _selector_module, selector = import_symbol(selector_cfg["module"], selector_cfg["function"])
    selected, selected_closure = selector(**with_hash)
    require(dict(selected) == dict(anime), "loader selector did not select fresh Anime mapping")
    require(tuple(selected_closure) == tuple(closure), "loader selector closure drift")

    mutations = {
        "stage": "512b",
        "eligibility": "wrong-eligibility",
        "plan_id": "unknown-plan",
        "plan_hash": "0" * 64,
        "catalog_hash": "1" * 64,
        "plan_header_sha256": "2" * 64,
        "catalog_header_sha256": "3" * 64,
    }
    negative_cases: list[str] = []
    for field, bad_value in mutations.items():
        bad = {**with_hash, field: bad_value}
        require_raises(
            image_module.ImageExceptionBindingError,
            image_binder,
            bad,
            f"image partial identity/{field}",
        )
        require_raises(
            anime_module.AnimeNativeUpsampleBindingError,
            anime_binder,
            bad,
            f"Anime partial identity/{field}",
        )
        require_raises(
            anime_module.AnimeNativeUpsampleBindingError,
            selector,
            bad,
            f"loader partial identity/{field}",
        )
        negative_cases.extend(
            (f"image:{field}", f"anime:{field}", f"loader_selector:{field}")
        )
    for label, mixed_plan_id in (
        ("old_a_fresh", image_module.STAGE512A_WEIGHTED_PLAN_ID),
        ("stage512b_fresh", image_module.STAGE512B_ANCHORS_PLAN_ID),
    ):
        bad = {**with_hash, "plan_id": mixed_plan_id}
        require_raises(
            image_module.ImageExceptionBindingError,
            image_binder,
            bad,
            f"image mixed identity/{label}",
        )
        require_raises(
            anime_module.AnimeNativeUpsampleBindingError,
            selector,
            bad,
            f"loader mixed identity/{label}",
        )
        negative_cases.extend((f"image:mixed:{label}", f"loader_selector:mixed:{label}"))

    for field, bad_value in (
        ("plan_id", caption_module.STAGE512A_W64_PLAN_ID),
        ("plan_header_sha256", caption_module.STAGE512A_W64_PLAN_HEADER_SHA256),
    ):
        bad = {**common, field: bad_value}
        require_raises(
            caption_module.CaptionExceptionBindingError,
            caption_binder,
            bad,
            f"caption partial old-profile identity/{field}",
        )
        negative_cases.append(f"caption:partial_old_profile:{field}")

    return {
        "image": {
            "module": binders["image"]["module"],
            "function": binders["image"]["function"],
            "bound_row_ids": sorted(image),
            "rows": image_rows,
        },
        "anime": {
            "module": anime_cfg["module"],
            "function": anime_cfg["function"],
            "mapping_symbol": anime_cfg["mapping_symbol"],
            "closure_symbol": anime_cfg["closure_symbol"],
            "closure": list(closure),
            "bound_row_ids": sorted(anime),
            "rows": anime_rows,
            "loader_selector": f"{selector_cfg['module']}:{selector_cfg['function']}",
        },
        "caption": {
            "module": caption_cfg["module"],
            "function": caption_cfg["function"],
            "bound_row_ids": [],
        },
        "negative_identity_checks": {
            "status": "PASS",
            "count": len(negative_cases),
            "cases": negative_cases,
        },
    }


def validate_legacy_bridge(policy: Mapping[str, Any]) -> dict[str, str]:
    bridge = policy["legacy_bridge"]
    root = Path(bridge["root"])
    receipt = root / "CANONICAL_PLAN_RECEIPT.json"
    report = root / "CLOSURE_REPORT.json"
    require(bytes_sha256(read_stable(receipt, mode=0o444)) == bridge["canonical_receipt_sha256"], "legacy canonical receipt SHA drift")
    require(bytes_sha256(read_stable(report, mode=0o444)) == bridge["closure_report_sha256"], "legacy closure report SHA drift")
    _module, validator = import_symbol(bridge["validator_module"], bridge["validator_function"])
    validator(root)
    return {
        "root": str(root),
        "canonical_receipt_sha256": bridge["canonical_receipt_sha256"],
        "closure_report_sha256": bridge["closure_report_sha256"],
        "validator": f"{bridge['validator_module']}:{bridge['validator_function']}",
    }


def verify_catalog(
    policy: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    expected = policy["catalog"]
    root = Path(expected["root"])
    real_directory(root)
    header_raw = read_stable(root / "header.json")
    require(bytes_sha256(header_raw) == expected["header_sha256"], "catalog header SHA drift")
    header = decode_json(header_raw, "catalog header")
    require(header["catalog_hash"] == expected["catalog_hash"], "catalog hash drift")
    require(int(header["row_count"]) == int(expected["row_count"]), "catalog row_count drift")
    require(tuple(header["domain_order"]) == DOMAINS, "catalog domain order drift")
    fingerprints: dict[str, tuple[int, ...]] = {}
    files: dict[str, Any] = {}
    for domain in DOMAINS:
        wanted = expected["files"][domain]
        actual = header["files"][domain]
        for field in ("name", "rows", "bytes", "sha256"):
            require(actual[field] == wanted[field], f"catalog {domain}/{field} drift")
        path = root / wanted["name"]
        digest, fingerprint = stable_file_sha256(path)
        require(digest == wanted["sha256"], f"catalog parquet SHA drift: {domain}")
        require(path.stat().st_size == int(wanted["bytes"]), f"catalog parquet size drift: {domain}")
        fingerprints[domain] = fingerprint
        files[domain] = {**wanted, "verified": True}
    return {
        "root": str(root),
        "catalog_hash": expected["catalog_hash"],
        "header_sha256": expected["header_sha256"],
        "row_count": int(expected["row_count"]),
        "files": files,
    }, fingerprints


def verify_plan_and_control(policy: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected = policy["plan"]
    root = Path(expected["root"])
    real_directory(root, mode=0o555)
    real_directory(root / "blocks", mode=0o555)
    require({p.name for p in root.iterdir()} == {"header.json", "checksums.json", "blocks"}, "plan root member-set drift")
    header_raw = read_stable(root / "header.json", mode=0o444)
    checks_raw = read_stable(root / "checksums.json", mode=0o444)
    require(bytes_sha256(header_raw) == expected["header_sha256"], "plan header SHA drift")
    require(bytes_sha256(checks_raw) == expected["checksums_sha256"], "plan checksums SHA drift")
    header = decode_json(header_raw, "plan header")
    checksums = decode_json(checks_raw, "plan checksums")
    for field in ("plan_id", "plan_hash", "block_steps", "num_blocks", "world_size", "per_rank_bs", "total_steps", "total_exposures"):
        require(header[field] == expected[field], f"plan {field} drift")
    require(header["hashes"]["blocks_manifest_sha256"] == expected["block_manifest_sha256"], "block manifest header drift")
    exact_keys(checksums, {"header", "blocks"}, "plan checksums")
    require(checksums["header"] == expected["header_sha256"], "checksums/header drift")
    block_map = checksums["blocks"]
    names = {f"block_{index:06d}.npz" for index in range(int(expected["num_blocks"]))}
    require(set(block_map) == names, "block checksum member-set drift")
    require({p.name for p in (root / "blocks").iterdir()} == names, "block directory member-set drift")
    manifest = bytes_sha256(canonical_bytes(dict(block_map)) + b"\n")
    require(manifest == expected["block_manifest_sha256"], "computed block manifest drift")

    control = policy["control"]
    old_root = Path(control["old_plan_root"])
    real_directory(old_root, mode=0o555)
    old_raw = read_stable(old_root / "header.json", mode=0o444)
    require(bytes_sha256(old_raw) == control["old_plan_header_sha256"], "old plan header SHA drift")
    old = decode_json(old_raw, "old plan header")
    require(old["plan_hash"] == control["old_plan_hash"], "old plan hash drift")
    fields = control["invariant_fields"]
    require(type(fields) is list and len(fields) == len(set(fields)), "control invariant field list drift")
    differences = [field for field in fields if old.get(field) != header.get(field)]
    require(not differences, f"old/new control invariants drift: {differences}")
    for name, key in (("D", "d_sha256"), ("S", "s_sha256"), ("C", "c_sha256")):
        digest = canonical_sha256(header[name])
        require(digest == control[key], f"plan {name} digest drift")
        require(header[name] == old[name], f"old/new {name} drift")
    invariant_payload = {field: header.get(field) for field in fields}
    control_report = {
        "old_plan_root": str(old_root),
        "old_plan_hash": old["plan_hash"],
        "old_plan_header_sha256": control["old_plan_header_sha256"],
        "invariant_fields": fields,
        "invariant_sha256": canonical_sha256(invariant_payload),
        "d_sha256": control["d_sha256"],
        "s_sha256": control["s_sha256"],
        "c_sha256": control["c_sha256"],
        "d_s_c_equal_old": True,
    }
    return header, checksums, control_report


def scan_plan(
    policy: Mapping[str, Any],
    header: Mapping[str, Any],
    checksums: Mapping[str, Any],
    catalog_header: Mapping[str, Any],
    catalog_fingerprints: Mapping[str, tuple[int, ...]],
) -> dict[str, Any]:
    plan = policy["plan"]
    plan_root = Path(plan["root"])
    catalog_root = Path(policy["catalog"]["root"])
    bounds = domain_bounds(policy["catalog"])
    joins = {
        domain: load_real_join(domain, catalog_root, catalog_header, bounds, catalog_fingerprints[domain])
        for domain in REAL_DOMAINS
    }
    expected_hits = policy["expected_hits"]
    target_entries: dict[int, tuple[str, Mapping[str, Any]]] = {}
    for group in ("textatlas", "anime"):
        for key, entry in expected_hits[group].items():
            target_entries[int(key)] = (group, entry)
    for key, uid in expected_hits["captionless_monet"].items():
        target_entries[int(key)] = ("captionless_monet", {"uid": uid})
    target_array = np.asarray(sorted(target_entries), dtype=np.int64)
    hits: dict[int, list[dict[str, int]]] = defaultdict(list)
    domain_totals = np.zeros(len(DOMAINS), dtype=np.int64)
    bucket_steps = np.zeros(len(header["buckets"]), dtype=np.int64)
    band_counts = {domain: np.zeros(4, dtype=np.int64) for domain in REAL_DOMAINS}
    join_stats = {
        domain: {
            "selected_exposures": 0,
            "joined_exposures": 0,
            "missing_join": 0,
            "duplicate_join": 0,
            "invalid_band": 0,
            "unserved": 0,
            "nonpositive_captioner_code": 0,
            "nonpositive_or_nonfinite_base_weight": 0,
        }
        for domain in REAL_DOMAINS
    }
    blocks_verified = 0
    steps_scanned = 0
    slots_scanned = 0
    block_steps = int(plan["block_steps"])
    world = int(plan["world_size"])
    batch = int(plan["per_rank_bs"])

    for index in range(int(plan["num_blocks"])):
        name = f"block_{index:06d}.npz"
        path = plan_root / "blocks" / name
        payload = read_stable(path, mode=0o444)
        require(bytes_sha256(payload) == checksums["blocks"][name], f"block SHA drift: {name}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            require(set(archive.files) == BLOCK_MEMBERS, f"block NPZ member drift: {name}")
            buckets = np.asarray(archive["bucket_of_step"])
            rows = np.asarray(archive["item_row"])
            bands = np.asarray(archive["caption_band"])
            counts = np.asarray(archive["domain_counts"])
        expected_steps = min(block_steps, int(plan["total_steps"]) - index * block_steps)
        require(buckets.dtype == np.int16 and buckets.shape == (expected_steps,), f"bucket array drift: {name}")
        require(rows.dtype == np.int64 and rows.shape == (expected_steps, world, batch), f"item_row array drift: {name}")
        require(bands.dtype == np.int8 and bands.shape == rows.shape, f"caption_band array drift: {name}")
        require(counts.dtype == np.int32 and counts.shape == (expected_steps, len(DOMAINS)), f"domain_counts drift: {name}")
        require(np.all(counts >= 0), f"negative domain count: {name}")
        require(np.all(counts.sum(axis=1, dtype=np.int64) == world * batch), f"domain count sum drift: {name}")

        actual_counts = np.empty_like(counts)
        real_mask = np.zeros(rows.shape, dtype=bool)
        for domain_index, domain in enumerate(DOMAINS):
            lo, hi = bounds[domain]
            mask = (rows >= lo) & (rows < hi)
            actual_counts[:, domain_index] = mask.sum(axis=(1, 2), dtype=np.int64)
            if domain in REAL_DOMAINS:
                real_mask |= mask
                planned = bands[mask]
                stats = join_stats[domain]
                stats["selected_exposures"] += int(planned.size)
                invalid = (planned < 0) | (planned >= 4)
                stats["invalid_band"] += int(np.count_nonzero(invalid))
                require(not np.any(invalid), f"invalid planned band in {domain}/{name}")
                join = joins[domain]
                local = rows[mask] - join.row_base
                require(np.all((local >= 0) & (local < len(join.served))), f"{domain} local row range drift")
                stats["joined_exposures"] += int(local.size)
                selected_served = join.served[local]
                selected_codes = join.codes[local, planned]
                selected_weights = join.weights[local]
                served_ok = ((selected_served.astype(np.uint16) >> planned.astype(np.uint16)) & 1) == 1
                code_ok = selected_codes > 0
                weight_ok = np.isfinite(selected_weights) & (selected_weights > 0)
                stats["unserved"] += int(np.count_nonzero(~served_ok))
                stats["nonpositive_captioner_code"] += int(np.count_nonzero(~code_ok))
                stats["nonpositive_or_nonfinite_base_weight"] += int(np.count_nonzero(~weight_ok))
                require(np.all(served_ok), f"unserved planned {domain} band in {name}")
                require(np.all(code_ok), f"nonpositive planned {domain} captioner code in {name}")
                require(np.all(weight_ok), f"nonpositive/nonfinite planned {domain} base_weight in {name}")
                band_counts[domain] += np.bincount(planned.astype(np.int64), minlength=4)
        require(np.array_equal(actual_counts, counts), f"domain_counts/row-domain join drift: {name}")
        require(np.all(bands[~real_mask] == -1), f"non-real planned band drift: {name}")

        target_mask = np.isin(rows, target_array)
        for local_step, rank, slot in np.argwhere(target_mask):
            row_id = int(rows[local_step, rank, slot])
            group, entry = target_entries[row_id]
            step = index * block_steps + int(local_step)
            if group != "captionless_monet":
                require(int(buckets[local_step]) == int(entry["bucket_idx"]), f"target bucket drift: {row_id}")
                require(int(bands[local_step, rank, slot]) == -1, f"target non-real band drift: {row_id}")
            hits[row_id].append({"step": step, "rank": int(rank), "slot": int(slot)})

        domain_totals += counts.sum(axis=0, dtype=np.int64)
        bucket_steps += np.bincount(buckets.astype(np.int64), minlength=len(bucket_steps))
        blocks_verified += 1
        steps_scanned += expected_steps
        slots_scanned += int(rows.size)

    require(blocks_verified == int(plan["num_blocks"]), "verified block count drift")
    require(steps_scanned == int(plan["total_steps"]), "scanned step count drift")
    require(slots_scanned == int(plan["total_exposures"]), "scanned slot count drift")
    require(domain_totals.tolist() == [int(header["domain_totals"][domain]) for domain in DOMAINS], "realized domain exposure drift")
    require(bucket_steps.tolist() == header["S"], "realized bucket S drift")
    combined_bands = band_counts["monet_real"] + band_counts["gpic"]
    require(combined_bands.tolist() == list(plan["real_band_exposures"]), "real band exposure drift")
    require(combined_bands.tolist() == header["materialization"]["realized_real_band_exposures"], "real band/header drift")
    for row_id, (group, entry) in target_entries.items():
        actual = sorted(hits[row_id], key=lambda x: (x["step"], x["rank"], x["slot"]))
        if group == "captionless_monet":
            require(not actual, f"captionless MONET row became reachable: {row_id}")
        else:
            require(actual == coordinates_from_policy(entry), f"target coordinate closure drift: {row_id}")

    for domain, join in joins.items():
        require_fingerprint(join.path, join.fingerprint, f"{domain} catalog")
        require(
            join_stats[domain]["joined_exposures"]
            == join_stats[domain]["selected_exposures"],
            f"{domain} join coverage drift",
        )
        require(
            all(
                value == 0
                for key, value in join_stats[domain].items()
                if key not in {"selected_exposures", "joined_exposures"}
            ),
            f"{domain} join invalid counter nonzero",
        )

    target_report = {
        "textatlas": {
            key: {
                "uid": entry["uid"],
                "bucket_idx": int(entry["bucket_idx"]),
                "exposure_count": len(hits[int(key)]),
                "coordinates": sorted(hits[int(key)], key=lambda x: (x["step"], x["rank"], x["slot"])),
            }
            for key, entry in expected_hits["textatlas"].items()
        },
        "anime": {
            key: {
                "uid": entry["uid"],
                "bucket_idx": int(entry["bucket_idx"]),
                "exposure_count": len(hits[int(key)]),
                "coordinates": sorted(hits[int(key)], key=lambda x: (x["step"], x["rank"], x["slot"])),
            }
            for key, entry in expected_hits["anime"].items()
        },
        "captionless_monet": {
            key: {"uid": uid, "exposure_count": len(hits[int(key)])}
            for key, uid in expected_hits["captionless_monet"].items()
        },
    }
    return {
        "verification_grain": "one_zero_based_(step,rank,slot)_per_exposure",
        "blocks_expected": int(plan["num_blocks"]),
        "blocks_verified": blocks_verified,
        "verified_blocks": blocks_verified,
        "steps_scanned": steps_scanned,
        "slots_scanned": slots_scanned,
        "scanned_exposures": slots_scanned,
        "all_block_checksums_verified": True,
        "realized_domain_exposures": dict(zip(DOMAINS, domain_totals.tolist(), strict=True)),
        "realized_bucket_steps_sha256": canonical_sha256(bucket_steps.tolist()),
        "real_band_exposures": {
            domain: band_counts[domain].tolist() for domain in REAL_DOMAINS
        } | {"combined": combined_bands.tolist()},
        "real_band_catalog_join": join_stats,
        "real_band_join_coverage": {
            "selected_exposures": sum(
                join_stats[domain]["selected_exposures"] for domain in REAL_DOMAINS
            ),
            "joined_exposures": sum(
                join_stats[domain]["joined_exposures"] for domain in REAL_DOMAINS
            ),
            "missing_join": 0,
            "duplicate_join": 0,
            "coverage": "100%",
        },
        "unserved_real_band_selected": sum(
            join_stats[domain]["unserved"] for domain in REAL_DOMAINS
        ),
        "targets": target_report,
    }


def scan(policy_path: Path) -> dict[str, Any]:
    require(sys.flags.optimize == 0, "optimized Python is forbidden")
    policy, policy_raw = load_policy(policy_path)
    sources_before = source_authorities(policy, policy_path)
    legacy = validate_legacy_bridge(policy)
    header, checksums, control = verify_plan_and_control(policy)
    contract_path = Path(policy["weighted_contract"]["path"])
    contract_sha, contract_fingerprint = stable_file_sha256(contract_path)
    require(contract_sha == policy["weighted_contract"]["sha256"], "weighted contract SHA drift")
    catalog_report, catalog_fingerprints = verify_catalog(policy)
    catalog_header = decode_json(
        read_stable(Path(policy["catalog"]["root"]) / "header.json"),
        "catalog header",
    )
    bounds = domain_bounds(policy["catalog"])
    bindings = binder_closure(policy, header, catalog_header, bounds)

    # The catalog rows named by runtime definitions must match the immutable catalog.
    for key, entry in policy["expected_hits"]["textatlas"].items():
        row_id = int(key)
        domain, record = catalog_record(
            Path(policy["catalog"]["root"]), catalog_header, bounds, row_id,
            ("row_id", "uid", "domain", "bucket_idx"),
        )
        require(domain == "textatlas" and record == {"row_id": row_id, "uid": entry["uid"], "domain": "textatlas", "bucket_idx": entry["bucket_idx"]}, f"TextAtlas catalog definition drift: {row_id}")
    for key, entry in policy["expected_hits"]["anime"].items():
        row_id = int(key)
        domain, record = catalog_record(
            Path(policy["catalog"]["root"]), catalog_header, bounds, row_id,
            ("row_id", "uid", "domain", "bucket_idx"),
        )
        require(domain == "anime" and record == {"row_id": row_id, "uid": entry["uid"], "domain": "anime", "bucket_idx": entry["bucket_idx"]}, f"Anime catalog definition drift: {row_id}")
    for key, uid in policy["expected_hits"]["captionless_monet"].items():
        row_id = int(key)
        domain, record = catalog_record(
            Path(policy["catalog"]["root"]), catalog_header, bounds, row_id,
            ("row_id", "uid", "domain", "served_mask", "band_captioner_t1", "band_captioner_t2", "band_captioner_t3", "band_captioner_t4"),
        )
        require(domain == "monet_real" and record["row_id"] == row_id and record["uid"] == uid and record["domain"] == "monet_real", f"MONET catalog identity drift: {row_id}")
        require(record["served_mask"] == 0 and all(record[f"band_captioner_t{i}"] == 0 for i in range(1, 5)), f"captionless MONET definition no longer empty: {row_id}")

    plan_scan = scan_plan(policy, header, checksums, catalog_header, catalog_fingerprints)
    require_fingerprint(contract_path, contract_fingerprint, "weighted contract")
    for domain, fingerprint in catalog_fingerprints.items():
        require_fingerprint(Path(policy["catalog"]["root"]) / policy["catalog"]["files"][domain]["name"], fingerprint, f"{domain} catalog")
    sources_after = source_authorities(policy, policy_path)
    require(sources_after == sources_before, "source authority changed during scan")

    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "policy": {
            "path": str(policy_path),
            "schema": POLICY_SCHEMA,
            "sha256": bytes_sha256(policy_raw),
        },
        "plan": {
            "root": policy["plan"]["root"],
            "plan_id": header["plan_id"],
            "plan_hash": header["plan_hash"],
            "header_sha256": policy["plan"]["header_sha256"],
            "checksums_sha256": policy["plan"]["checksums_sha256"],
            "block_manifest_sha256": policy["plan"]["block_manifest_sha256"],
            "num_blocks": int(header["num_blocks"]),
            "block_steps": int(header["block_steps"]),
            "world_size": int(header["world_size"]),
            "per_rank_bs": int(header["per_rank_bs"]),
            "total_steps": int(header["total_steps"]),
            "total_exposures": int(header["total_exposures"]),
            "d_sha256": policy["control"]["d_sha256"],
            "s_sha256": policy["control"]["s_sha256"],
            "c_sha256": policy["control"]["c_sha256"],
        },
        "catalog": catalog_report,
        "weighted_contract": {
            "path": str(contract_path),
            "sha256": contract_sha,
        },
        "control": control,
        "coordinate_schema": {
            "zero_based": True,
            "fields": ["step", "rank", "slot"],
            "global_index_formula": "step*2048+rank*32+slot",
            "world_size": 64,
            "per_rank_bs": 32,
        },
        "registries": bindings,
        "legacy_bridge": legacy,
        "scan": plan_scan,
        "source_authorities": sources_before,
    }
    validate_report(policy, policy_raw, report, policy_path, verify_live_sources=False)
    return report


def validate_report(
    policy: Mapping[str, Any],
    policy_raw: bytes,
    report: Mapping[str, Any],
    policy_path: Path,
    *,
    verify_live_sources: bool,
) -> None:
    exact_keys(
        report,
        {
            "schema", "status", "policy", "plan", "catalog", "weighted_contract",
            "control", "coordinate_schema", "registries", "legacy_bridge", "scan",
            "source_authorities",
        },
        "registry closure report",
    )
    require(report["schema"] == REPORT_SCHEMA and report["status"] == "PASS", "report schema/status drift")
    require(report["policy"] == {"path": str(policy_path), "schema": POLICY_SCHEMA, "sha256": bytes_sha256(policy_raw)}, "report policy binding drift")
    for key in ("root", "plan_id", "plan_hash", "header_sha256", "checksums_sha256", "block_manifest_sha256", "num_blocks", "block_steps", "world_size", "per_rank_bs", "total_steps", "total_exposures"):
        require(report["plan"][key] == policy["plan"][key], f"report plan/{key} drift")
    require(report["plan"]["d_sha256"] == policy["control"]["d_sha256"], "report D digest drift")
    require(report["plan"]["s_sha256"] == policy["control"]["s_sha256"], "report S digest drift")
    require(report["plan"]["c_sha256"] == policy["control"]["c_sha256"], "report C digest drift")
    scan_value = report["scan"]
    require(scan_value["blocks_expected"] == scan_value["blocks_verified"] == policy["plan"]["num_blocks"], "report block closure drift")
    require(scan_value["verified_blocks"] == policy["plan"]["num_blocks"], "report explicit verified_blocks drift")
    require(scan_value["steps_scanned"] == policy["plan"]["total_steps"], "report step closure drift")
    require(scan_value["slots_scanned"] == policy["plan"]["total_exposures"], "report slot closure drift")
    require(scan_value["scanned_exposures"] == policy["plan"]["total_exposures"], "report explicit scanned_exposures drift")
    require(scan_value["all_block_checksums_verified"] is True, "report block checksum flag drift")
    require(scan_value["real_band_exposures"]["combined"] == policy["plan"]["real_band_exposures"], "report band exposure drift")
    for domain in REAL_DOMAINS:
        stats = scan_value["real_band_catalog_join"][domain]
        require(stats["selected_exposures"] == scan_value["realized_domain_exposures"][domain], f"report {domain} joined exposure drift")
        require(stats["joined_exposures"] == stats["selected_exposures"], f"report {domain} join coverage drift")
        require(all(value == 0 for key, value in stats.items() if key not in {"selected_exposures", "joined_exposures"}), f"report {domain} invalid join counter")
    coverage = scan_value["real_band_join_coverage"]
    require(
        coverage["selected_exposures"] == coverage["joined_exposures"]
        and coverage["coverage"] == "100%"
        and coverage["missing_join"] == coverage["duplicate_join"] == 0
        and scan_value["unserved_real_band_selected"] == 0,
        "report global real-band semantic join closure drift",
    )
    for group in ("textatlas", "anime"):
        require(set(scan_value["targets"][group]) == set(policy["expected_hits"][group]), f"report target set drift: {group}")
        for key, expected in policy["expected_hits"][group].items():
            actual = scan_value["targets"][group][key]
            require(actual["uid"] == expected["uid"] and actual["bucket_idx"] == expected["bucket_idx"], f"report target identity drift: {key}")
            require(actual["coordinates"] == coordinates_from_policy(expected), f"report target coordinates drift: {key}")
            require(actual["exposure_count"] == len(expected["coordinates"]), f"report target count drift: {key}")
    expected_monet = policy["expected_hits"]["captionless_monet"]
    require(set(scan_value["targets"]["captionless_monet"]) == set(expected_monet), "report MONET target set drift")
    for key, uid in expected_monet.items():
        require(scan_value["targets"]["captionless_monet"][key] == {"uid": uid, "exposure_count": 0}, f"report MONET zero-hit drift: {key}")
    require(report["registries"]["image"]["bound_row_ids"] == sorted(int(key) for key in policy["expected_hits"]["textatlas"]), "report image binder row-set drift")
    require(report["registries"]["anime"]["bound_row_ids"] == sorted(int(key) for key in policy["expected_hits"]["anime"]), "report Anime binder row-set drift")
    require(report["registries"]["anime"]["closure"] == [6, 125000, 64, 32], "report Anime closure drift")
    require(report["registries"]["caption"]["bound_row_ids"] == [], "report caption binder must be empty")
    negatives = report["registries"]["negative_identity_checks"]
    require(
        negatives["status"] == "PASS"
        and negatives["count"] == len(negatives["cases"])
        and negatives["count"] == 27,
        "report partial/mixed identity negative closure drift",
    )
    require(report["legacy_bridge"]["canonical_receipt_sha256"] == policy["legacy_bridge"]["canonical_receipt_sha256"], "legacy receipt bridge drift")
    require(report["legacy_bridge"]["closure_report_sha256"] == policy["legacy_bridge"]["closure_report_sha256"], "legacy report bridge drift")
    if verify_live_sources:
        require(report["source_authorities"]["files"] == source_authorities(policy, policy_path)["files"], "live source SHA drift from closure receipt")


def verify_root(policy_path: Path) -> str:
    require(sys.flags.optimize == 0, "optimized Python is forbidden")
    policy, policy_raw = load_policy(policy_path)
    root = Path(policy["publication"]["root"])
    real_directory(root, mode=0o555)
    require({path.name for path in root.iterdir()} == {"REGISTRY_CLOSURE.json", "_SUCCESS"}, "published member-set drift")
    receipt_raw = read_stable(root / "REGISTRY_CLOSURE.json", mode=0o444)
    marker_raw = read_stable(root / "_SUCCESS", mode=0o444)
    receipt_sha = bytes_sha256(receipt_raw)
    require(marker_raw == f"{SUCCESS_PREFIX}{receipt_sha}\n".encode("ascii"), "published success marker drift")
    report = decode_json(receipt_raw, "published registry closure")
    validate_report(policy, policy_raw, report, policy_path, verify_live_sources=True)
    return receipt_sha


def publish(policy_path: Path, left: Path, right: Path) -> str:
    require(sys.flags.optimize == 0, "optimized Python is forbidden")
    policy, policy_raw = load_policy(policy_path)
    left_raw = read_stable(left, mode=0o444)
    right_raw = read_stable(right, mode=0o444)
    require(left_raw == right_raw, "independent full scans are not byte-identical")
    report = decode_json(left_raw, "candidate registry closure")
    validate_report(policy, policy_raw, report, policy_path, verify_live_sources=True)
    receipt_sha = bytes_sha256(left_raw)
    marker = f"{SUCCESS_PREFIX}{receipt_sha}\n".encode("ascii")
    from scripts.offline_arb.immutable_review_evidence import publish_recoverable_directory

    publish_recoverable_directory(
        Path(policy["publication"]["root"]),
        {"REGISTRY_CLOSURE.json": left_raw, "_SUCCESS": marker},
        commit_member="_SUCCESS",
        final_readonly=True,
    )
    require(verify_root(policy_path) == receipt_sha, "published receipt readback drift")
    return receipt_sha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan_parser = sub.add_parser("scan")
    scan_parser.add_argument("--policy", type=Path, required=True)
    scan_parser.add_argument("--out", type=Path, required=True)
    publish_parser = sub.add_parser("publish")
    publish_parser.add_argument("--policy", type=Path, required=True)
    publish_parser.add_argument("--left", type=Path, required=True)
    publish_parser.add_argument("--right", type=Path, required=True)
    verify_parser = sub.add_parser("verify-root")
    verify_parser.add_argument("--policy", type=Path, required=True)
    check_parser = sub.add_parser("check-policy")
    check_parser.add_argument("--policy", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(sys.flags.optimize == 0, "optimized Python is forbidden")
        if args.command == "check-policy":
            policy, _raw = load_policy(args.policy)
            source_authorities(policy, args.policy)
            print("STAGE512A_CONTROL_REGISTRY_CLOSURE_V2_POLICY_PASS", flush=True)
            return 0
        if args.command == "scan":
            report = scan(args.policy)
            raw = json_bytes(report)
            write_exclusive(args.out, raw)
            print(
                "STAGE512A_CONTROL_REGISTRY_CLOSURE_V2_SCAN_PASS",
                bytes_sha256(raw),
                flush=True,
            )
            return 0
        if args.command == "publish":
            digest = publish(args.policy, args.left, args.right)
            print("STAGE512A_CONTROL_REGISTRY_CLOSURE_V2_PUBLISHED", digest, flush=True)
            return 0
        if args.command == "verify-root":
            digest = verify_root(args.policy)
            print("STAGE512A_CONTROL_REGISTRY_CLOSURE_V2_VERIFY_PASS", digest, flush=True)
            return 0
        raise RegistryClosureError(f"unknown command: {args.command}")
    except BaseException as exc:
        print(
            f"STAGE512A_CONTROL_REGISTRY_CLOSURE_V2_FAIL {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
