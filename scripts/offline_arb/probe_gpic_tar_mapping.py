#!/usr/bin/env python3
"""Read-only proof that GPIC catalog image paths map to official train tar members.

The probe writes compact evidence to FSx only.  Source tar archives and the
existing unzipped image tree are always opened read-only.  For deterministic
catalog rows spanning front/middle/back tar shards it proves:

``images/train/<shard>/<key>.<ext>``
    == ``train/gpic_train_<shard>.tar :: <key>.<ext>``

by exact length and SHA-256.  It also inventories all 8,000 train tar names and
checks the README-declared JSON/image pair order for every member in each
sampled tar.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import platform
import re
import tarfile
import time
import uuid
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq

from scripts.offline_arb import run_offline_gpic_blob_pilot as pilot_core


SCHEMA_VERSION = "offline_gpic_tar_mapping_probe_v1"
REPORT_SCHEMA = "offline_gpic_tar_mapping_report_v1"
INVENTORY_SCHEMA = "offline_gpic_tar_inventory_v1"
SAMPLE_SCHEMA = "offline_gpic_tar_catalog_sample_v1"
READ_AUTHORITY = "offline-gpic-tar-mapping-read-only-v1"
TRAIN_TAR_COUNT = 8_000
DEFAULT_TAR_SHARDS = (
    0,
    1,
    7,
    31,
    127,
    511,
    1_023,
    2_047,
    3_071,
    4_095,
    5_119,
    6_143,
    7_001,
    7_500,
    7_900,
    7_999,
)
DEFAULT_SAMPLES_PER_TAR = 16
DEFAULT_TAR_WORKERS = 8
SOURCE_PREFIX = (
    "/s3-mewtant-raw-datasets/stanford-vision-lab/"
    "gpic_umziped/images/train/"
)
TAR_ROOT = Path("/s3-mewtant-raw-datasets/stanford-vision-lab/gpic/train")
README_PATH = Path("/s3-mewtant-raw-datasets/stanford-vision-lab/gpic/README.md")
TAR_NAME_RE = re.compile(r"gpic_train_([0-9]{5})\.tar")
IMAGE_SUFFIXES = {".jpg", ".png"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MIB = 1 << 20


class ProbeError(RuntimeError):
    """Fail-closed diagnostic contract error."""


@dataclasses.dataclass(frozen=True)
class CatalogItem:
    local_index: int
    row_id: int
    image_path: str
    tar_shard: int
    member_basename: str
    catalog_row_group: int
    catalog_row_group_offset: int
    catalog_row_group_stratum: int


def utc_now() -> str:
    return pilot_core.utc_now()


def canonical_json_bytes(value: Any) -> bytes:
    return pilot_core.canonical_json_bytes(value)


def sha256_file(path: Path, block_bytes: int = 8 * MIB) -> str:
    return pilot_core.sha256_file(path, block_bytes)


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def self_hash(value: dict[str, Any], field: str) -> str:
    core = dict(value)
    core.pop(field, None)
    return hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def require_regular_file(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProbeError(f"missing source file: {path}") from exc
    if not path.is_file() or path.is_symlink() or metadata.st_size <= 0:
        raise ProbeError(f"unsafe/empty source file: {path}")
    return metadata


def require_regular_dir(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProbeError(f"missing source directory: {path}") from exc
    if not path.is_dir() or path.is_symlink():
        raise ProbeError(f"unsafe source directory: {path}")
    return metadata


def write_once_json(path: Path, value: Any) -> bytes:
    raw = canonical_json_bytes(value)
    temporary = path.parent / f".{path.name}.work.{uuid.uuid4().hex}"
    pilot_core.write_once_bytes(temporary, raw, fsync_file=True)
    pilot_core.fsync_dir(path.parent)
    pilot_core.publish_file_noreplace(temporary, path)
    pilot_core.fsync_dir(path.parent)
    return raw


def parse_catalog_path(image_path: str, source_prefix: str = SOURCE_PREFIX) -> tuple[int, str]:
    if not isinstance(image_path, str) or not image_path.startswith(source_prefix):
        raise ProbeError("catalog GPIC path escaped the frozen unzipped prefix")
    relative = image_path[len(source_prefix) :]
    pieces = relative.split("/")
    if len(pieces) != 2:
        raise ProbeError(f"catalog GPIC path is not shard/member shaped: {image_path}")
    shard_raw, basename = pieces
    if len(shard_raw) != 5 or not shard_raw.isascii() or not shard_raw.isdigit():
        raise ProbeError(f"catalog GPIC shard component is malformed: {image_path}")
    shard = int(shard_raw)
    if not (0 <= shard < TRAIN_TAR_COUNT):
        raise ProbeError(f"catalog GPIC shard is out of range: {image_path}")
    pure = PurePosixPath(basename)
    if (
        pure.name != basename
        or basename in {"", ".", ".."}
        or pure.suffix.lower() not in IMAGE_SUFFIXES
    ):
        raise ProbeError(f"catalog GPIC member basename is malformed: {image_path}")
    return shard, basename


def expected_tar_name(shard: int) -> str:
    if type(shard) is not int or not (0 <= shard < TRAIN_TAR_COUNT):
        raise ProbeError("invalid train tar shard")
    return f"gpic_train_{shard:05d}.tar"


def parse_tar_shards(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(piece) for piece in raw.split(","))
    except ValueError as exc:
        raise ProbeError("--tar-shards must be comma-separated integers") from exc
    if not values or values != tuple(sorted(set(values))):
        raise ProbeError("--tar-shards must be nonempty, unique, and increasing")
    if any(value < 0 or value >= TRAIN_TAR_COUNT for value in values):
        raise ProbeError("--tar-shards contains an out-of-range shard")
    return values


def row_group_stratum(group_index: int, total_groups: int, strata: int) -> int:
    if (
        type(group_index) is not int
        or type(total_groups) is not int
        or type(strata) is not int
        or total_groups < 1
        or strata < 1
        or strata > total_groups
        or not (0 <= group_index < total_groups)
    ):
        raise ProbeError("invalid catalog row-group stratum request")
    return min(strata - 1, (group_index * strata) // total_groups)


def inventory_train_tars(tar_root: Path) -> dict[str, Any]:
    require_regular_dir(tar_root)
    started = time.perf_counter()
    tar_names: list[str] = []
    other_names: list[str] = []
    try:
        with os.scandir(tar_root) as entries:
            for entry in entries:
                match = TAR_NAME_RE.fullmatch(entry.name)
                if match is None:
                    other_names.append(entry.name)
                else:
                    tar_names.append(entry.name)
    except OSError as exc:
        raise ProbeError("unable to inventory GPIC train tar root") from exc
    tar_names.sort()
    other_names.sort()
    expected = [expected_tar_name(shard) for shard in range(TRAIN_TAR_COUNT)]
    missing = sorted(set(expected).difference(tar_names))
    unexpected_tar_names = sorted(set(tar_names).difference(expected))
    duplicate_count = len(tar_names) - len(set(tar_names))
    value = {
        "schema_version": INVENTORY_SCHEMA,
        "tar_root": str(tar_root),
        "expected_tar_count": TRAIN_TAR_COUNT,
        "observed_tar_count": len(tar_names),
        "first_tar_names": tar_names[:8],
        "last_tar_names": tar_names[-8:],
        "missing_tar_names": missing,
        "unexpected_tar_names": unexpected_tar_names,
        "duplicate_tar_name_count": duplicate_count,
        "non_tar_entries": other_names,
        "inventory_elapsed_seconds": time.perf_counter() - started,
        "exact_tar_name_set": (
            tar_names == expected
            and not missing
            and not unexpected_tar_names
            and duplicate_count == 0
        ),
    }
    value["inventory_sha256"] = self_hash(value, "inventory_sha256")
    return value


def select_catalog_samples(
    catalog: pilot_core.CatalogBinding,
    *,
    tar_shards: Sequence[int],
    samples_per_tar: int,
    source_prefix: str,
    max_row_groups: int,
) -> tuple[dict[int, list[CatalogItem]], dict[str, Any]]:
    targets = set(tar_shards)
    selected_by_stratum: dict[int, dict[int, CatalogItem]] = {
        shard: {} for shard in tar_shards
    }
    seen_paths: set[str] = set()
    parquet = pq.ParquetFile(catalog.gpic_path)
    if samples_per_tar > parquet.num_row_groups:
        raise ProbeError("samples-per-tar exceeds catalog row-group count")
    groups_scanned = 0
    rows_scanned = 0
    started = time.perf_counter()
    local_start = 0
    for group_index in range(parquet.num_row_groups):
        if group_index >= max_row_groups:
            break
        stratum = row_group_stratum(
            group_index, parquet.num_row_groups, samples_per_tar
        )
        group = parquet.read_row_group(
            group_index,
            columns=["row_id", "image_path"],
            use_threads=False,
        ).combine_chunks()
        row_ids = group.column("row_id")
        paths = group.column("image_path")
        if len(group) != parquet.metadata.row_group(group_index).num_rows:
            raise ProbeError("catalog row-group length drift")
        for offset in range(len(group)):
            local_index = local_start + offset
            row_id = row_ids[offset].as_py()
            image_path = paths[offset].as_py()
            if type(row_id) is not int or row_id != catalog.gpic_row_base + local_index:
                raise ProbeError(f"catalog row identity drift local={local_index}")
            shard, basename = parse_catalog_path(image_path, source_prefix)
            bucket = selected_by_stratum.get(shard) if shard in targets else None
            if bucket is not None and stratum not in bucket:
                if image_path in seen_paths:
                    raise ProbeError("duplicate image path entered deterministic sample")
                seen_paths.add(image_path)
                bucket[stratum] = CatalogItem(
                    local_index=local_index,
                    row_id=row_id,
                    image_path=image_path,
                    tar_shard=shard,
                    member_basename=basename,
                    catalog_row_group=group_index,
                    catalog_row_group_offset=offset,
                    catalog_row_group_stratum=stratum,
                )
        groups_scanned += 1
        rows_scanned += len(group)
        local_start += len(group)
    if groups_scanned != parquet.num_row_groups:
        raise ProbeError(
            "cross-catalog sample requires a full row-group scan; max cap was too small"
        )
    missing = {
        str(shard): sorted(set(range(samples_per_tar)).difference(values))
        for shard, values in selected_by_stratum.items()
        if set(values) != set(range(samples_per_tar))
    }
    if missing:
        raise ProbeError(
            f"catalog scan did not fill requested per-tar samples within row-group cap: {missing}"
        )
    selected = {
        shard: [selected_by_stratum[shard][index] for index in range(samples_per_tar)]
        for shard in tar_shards
    }
    digest = hashlib.sha256()
    for shard in tar_shards:
        for item in selected[shard]:
            encoded = item.image_path.encode("utf-8")
            digest.update(item.local_index.to_bytes(8, "little", signed=True))
            digest.update(item.row_id.to_bytes(8, "little", signed=True))
            digest.update(item.catalog_row_group.to_bytes(4, "little"))
            digest.update(item.catalog_row_group_offset.to_bytes(4, "little"))
            digest.update(item.catalog_row_group_stratum.to_bytes(4, "little"))
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
    evidence = {
        "schema_version": SAMPLE_SCHEMA,
        "selection": (
            "first_matching_catalog_row_per_requested_tar_shard_per_row_group_stratum"
        ),
        "tar_shards": list(tar_shards),
        "samples_per_tar": samples_per_tar,
        "sample_rows": sum(len(values) for values in selected.values()),
        "sample_image_paths_unique": len(seen_paths),
        "catalog_row_groups_scanned": groups_scanned,
        "catalog_rows_scanned": rows_scanned,
        "catalog_total_row_groups": parquet.num_row_groups,
        "catalog_total_rows": catalog.gpic_rows,
        "row_group_strata": samples_per_tar,
        "required_strata_per_tar": list(range(samples_per_tar)),
        "observed_strata_per_tar": {
            str(shard): [item.catalog_row_group_stratum for item in values]
            for shard, values in selected.items()
        },
        "sample_row_groups_per_tar": {
            str(shard): [item.catalog_row_group for item in values]
            for shard, values in selected.items()
        },
        "selection_sha256": digest.hexdigest(),
        "selection_elapsed_seconds": time.perf_counter() - started,
    }
    evidence["sample_sha256"] = self_hash(evidence, "sample_sha256")
    return selected, evidence


def hash_open_file(handle: Any, *, expected_bytes: int, label: str) -> str:
    digest = hashlib.sha256()
    total = 0
    while total < expected_bytes:
        block = handle.read(min(MIB, expected_bytes - total))
        if not block:
            raise ProbeError(f"short read while hashing {label}")
        total += len(block)
        digest.update(block)
    if total != expected_bytes or handle.read(1):
        raise ProbeError(f"length/tail drift while hashing {label}")
    return digest.hexdigest()


def hash_unzipped_image(path: Path) -> tuple[int, str]:
    before = require_regular_file(path)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        opened = os.fstat(descriptor)
        if opened.st_size != before.st_size:
            raise ProbeError(f"unzipped image size changed while opening: {path}")
        with os.fdopen(descriptor, "rb", buffering=0, closefd=False) as handle:
            digest = hash_open_file(
                handle,
                expected_bytes=opened.st_size,
                label=f"unzipped image {path.name}",
            )
        after = os.fstat(descriptor)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise ProbeError(f"unzipped image identity changed while reading: {path}")
        return opened.st_size, digest
    finally:
        if descriptor is not None:
            os.close(descriptor)


def member_sample(member: tarfile.TarInfo) -> dict[str, Any]:
    return {
        "name": member.name,
        "bytes": member.size,
        "type": member.type.decode("latin1") if isinstance(member.type, bytes) else str(member.type),
        "regular": member.isreg(),
    }


def inspect_one_tar(
    *,
    tar_root: Path,
    shard: int,
    items: Sequence[CatalogItem],
) -> dict[str, Any]:
    if not items or any(item.tar_shard != shard for item in items):
        raise ProbeError(f"sample/tar shard binding drift shard={shard}")
    tar_path = tar_root / expected_tar_name(shard)
    tar_metadata = require_regular_file(tar_path)
    wanted = {item.member_basename: item for item in items}
    if len(wanted) != len(items):
        raise ProbeError(f"duplicate requested tar member basename shard={shard}")
    first_members: list[dict[str, Any]] = []
    last_members: deque[dict[str, Any]] = deque(maxlen=8)
    found: dict[str, dict[str, Any]] = {}
    member_count = 0
    regular_count = 0
    json_count = 0
    image_count = 0
    pairing_valid = True
    pairing_issues: list[str] = []
    pending_json_stem: str | None = None
    member_prefixes: set[str] = set()
    scan_started = time.perf_counter()
    try:
        archive = tarfile.open(tar_path, mode="r:")
    except (OSError, tarfile.TarError) as exc:
        raise ProbeError(f"cannot open sampled GPIC tar: {tar_path}") from exc
    with archive:
        for member in archive:
            member_count += 1
            compact = member_sample(member)
            if len(first_members) < 8:
                first_members.append(compact)
            last_members.append(compact)
            if not member.isreg():
                pairing_valid = False
                if len(pairing_issues) < 16:
                    pairing_issues.append(f"nonregular:{member.name}")
                continue
            regular_count += 1
            pure = PurePosixPath(member.name)
            basename = pure.name
            prefix = str(pure.parent)
            member_prefixes.add("" if prefix == "." else prefix)
            suffix = pure.suffix.lower()
            stem = pure.stem
            if pending_json_stem is None:
                if suffix != ".json":
                    pairing_valid = False
                    if len(pairing_issues) < 16:
                        pairing_issues.append(f"expected_json:{member.name}")
                else:
                    json_count += 1
                    pending_json_stem = stem
            else:
                if suffix not in IMAGE_SUFFIXES or stem != pending_json_stem:
                    pairing_valid = False
                    if len(pairing_issues) < 16:
                        pairing_issues.append(
                            f"expected_paired_image:{pending_json_stem}:{member.name}"
                        )
                else:
                    image_count += 1
                pending_json_stem = None
            if basename not in wanted:
                continue
            if basename in found:
                raise ProbeError(f"duplicate image member in tar shard={shard}: {basename}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ProbeError(f"cannot extract selected regular member: {member.name}")
            with extracted:
                tar_sha = hash_open_file(
                    extracted,
                    expected_bytes=member.size,
                    label=f"tar member {member.name}",
                )
            item = wanted[basename]
            unzipped_bytes, unzipped_sha = hash_unzipped_image(Path(item.image_path))
            found[basename] = {
                "local_index": item.local_index,
                "row_id": item.row_id,
                "catalog_row_group": item.catalog_row_group,
                "catalog_row_group_offset": item.catalog_row_group_offset,
                "catalog_row_group_stratum": item.catalog_row_group_stratum,
                "catalog_image_path": item.image_path,
                "derived_tar_path": str(tar_path),
                "derived_member_basename": basename,
                "observed_member_name": member.name,
                "tar_member_offset_data": member.offset_data,
                "tar_member_bytes": member.size,
                "tar_member_sha256": tar_sha,
                "unzipped_bytes": unzipped_bytes,
                "unzipped_sha256": unzipped_sha,
                "bytes_equal": member.size == unzipped_bytes,
                "sha256_equal": tar_sha == unzipped_sha,
            }
    if pending_json_stem is not None:
        pairing_valid = False
        pairing_issues.append(f"trailing_json_without_image:{pending_json_stem}")
    missing = sorted(set(wanted).difference(found))
    comparisons = [found[name] for name in sorted(found)]
    exact_equivalence = (
        not missing
        and len(comparisons) == len(items)
        and all(item["bytes_equal"] and item["sha256_equal"] for item in comparisons)
    )
    value = {
        "tar_shard": shard,
        "tar_path": str(tar_path),
        "tar_bytes": tar_metadata.st_size,
        "member_count": member_count,
        "regular_member_count": regular_count,
        "json_member_count": json_count,
        "image_member_count": image_count,
        "json_image_pair_count": min(json_count, image_count),
        "pairing_valid": pairing_valid and json_count == image_count,
        "pairing_issues": pairing_issues,
        "member_name_prefixes": sorted(member_prefixes),
        "first_members": first_members,
        "last_members": list(last_members),
        "requested_members": len(items),
        "found_members": len(found),
        "missing_member_basenames": missing,
        "exact_payload_equivalence": exact_equivalence,
        "comparisons": comparisons,
        "scan_elapsed_seconds": time.perf_counter() - scan_started,
    }
    value["tar_evidence_sha256"] = self_hash(value, "tar_evidence_sha256")
    return value


def inspect_tars(
    *,
    tar_root: Path,
    selected: dict[int, list[CatalogItem]],
    tar_workers: int,
) -> list[dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=tar_workers, thread_name_prefix="gpic-tar-map"
    ) as executor:
        futures = {
            executor.submit(
                inspect_one_tar,
                tar_root=tar_root,
                shard=shard,
                items=items,
            ): shard
            for shard, items in selected.items()
        }
        for future in concurrent.futures.as_completed(futures):
            shard = futures[future]
            results[shard] = future.result()
            value = results[shard]
            print(
                "GPIC_TAR_MAPPING_SHARD "
                f"shard={shard:05d} members={value['member_count']} "
                f"sample={value['found_members']}/{value['requested_members']} "
                f"pairing={value['pairing_valid']} bytes_sha={value['exact_payload_equivalence']} "
                f"seconds={value['scan_elapsed_seconds']:.3f}",
                flush=True,
            )
    return [results[shard] for shard in sorted(results)]


def code_snapshot(job_spec: Path) -> dict[str, Any]:
    runner = Path(__file__).resolve(strict=True)
    core = Path(pilot_core.__file__).resolve(strict=True)
    spec = job_spec.expanduser().resolve(strict=True)
    return {
        "runner_path": str(runner),
        "runner_sha256": sha256_file(runner),
        "pilot_core_path": str(core),
        "pilot_core_sha256": sha256_file(core),
        "job_spec_path": str(spec),
        "job_spec_sha256": sha256_file(spec),
        "python": platform.python_version(),
        "pyarrow": pq.__version__ if hasattr(pq, "__version__") else None,
    }


def validate_code_snapshot(value: dict[str, Any]) -> None:
    for prefix in ("runner", "pilot_core", "job_spec"):
        if sha256_file(Path(value[f"{prefix}_path"])) != value[f"{prefix}_sha256"]:
            raise ProbeError(f"{prefix} changed during tar-mapping probe")


def run(args: argparse.Namespace) -> dict[str, Any]:
    code = code_snapshot(args.job_spec)
    require_regular_file(args.readme)
    readme_raw = args.readme.read_bytes()
    if not all(
        token in readme_raw
        for token in (
            b"There are 8000 tars for GPIC train",
            b"gpic_train_{00000",
            b"Each tar archive contains alternating image",
        )
    ):
        raise ProbeError("GPIC README no longer carries the frozen tar-layout claims")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise ProbeError(f"diagnostic output namespace is not fresh: {args.output_root}")
    args.output_root.mkdir(mode=0o700)
    pilot_core.fsync_dir(args.output_root.parent)
    inventory = inventory_train_tars(args.tar_root)
    catalog = pilot_core.load_catalog(args.catalog_root, verify_gpic_sha256=False)
    selected, sample = select_catalog_samples(
        catalog,
        tar_shards=args.tar_shards,
        samples_per_tar=args.samples_per_tar,
        source_prefix=args.source_prefix,
        max_row_groups=args.max_catalog_row_groups,
    )
    tar_results = inspect_tars(
        tar_root=args.tar_root,
        selected=selected,
        tar_workers=args.tar_workers,
    )
    catalog_value = pilot_core.catalog_manifest(catalog)
    comparisons = [
        comparison
        for tar_result in tar_results
        for comparison in tar_result["comparisons"]
    ]
    exact_mapping = (
        inventory["exact_tar_name_set"]
        and len(tar_results) == len(args.tar_shards)
        and len(comparisons) == len(args.tar_shards) * args.samples_per_tar
        and all(result["pairing_valid"] for result in tar_results)
        and all(result["exact_payload_equivalence"] for result in tar_results)
    )
    member_prefix_sets = sorted(
        {prefix for result in tar_results for prefix in result["member_name_prefixes"]}
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "generated_at": utc_now(),
        "status": "PASS" if exact_mapping else "FAIL",
        "jobtag": args.jobtag,
        "mapping_rule": (
            "catalog images/train/<shard:05d>/<basename> -> "
            "gpic/train/gpic_train_<shard:05d>.tar member basename"
        ),
        "tar_root": str(args.tar_root),
        "source_prefix": args.source_prefix,
        "read_only_probe": True,
        "writes": {"fsx_output_root": str(args.output_root), "s3_writes": 0},
        "readme": {
            "path": str(args.readme),
            "bytes": len(readme_raw),
            "sha256": hashlib.sha256(readme_raw).hexdigest(),
        },
        "inventory": inventory,
        "catalog_hash": catalog.catalog_hash,
        "gpic_rows": catalog.gpic_rows,
        "gpic_row_base": catalog.gpic_row_base,
        "gpic_sha256": catalog.gpic_sha256,
        "catalog": catalog_value,
        "sample": sample,
        "tar_shards": list(args.tar_shards),
        "tar_workers": args.tar_workers,
        "tar_member_name_prefixes": member_prefix_sets,
        "sample_comparisons": len(comparisons),
        "byte_equal_comparisons": sum(item["bytes_equal"] for item in comparisons),
        "sha256_equal_comparisons": sum(item["sha256_equal"] for item in comparisons),
        "all_sampled_tars_pairing_valid": all(
            result["pairing_valid"] for result in tar_results
        ),
        "all_catalog_samples_found_in_derived_tar": all(
            not result["missing_member_basenames"] for result in tar_results
        ),
        "exact_catalog_tar_payload_equivalence": exact_mapping,
        "tar_results": tar_results,
        "runtime": pilot_core.runtime_identity(),
        "code": code,
    }
    report["report_sha256"] = self_hash(report, "report_sha256")
    validate_code_snapshot(code)
    pilot_core.validate_catalog_closure(catalog)
    report_raw = write_once_json(args.output_root / "report.json", report)
    marker_payload = hashlib.sha256(report_raw).hexdigest().encode("ascii") + b"\n"
    pilot_core.write_once_bytes(
        args.output_root / "_COMPLETE", marker_payload, fsync_file=True
    )
    pilot_core.fsync_dir(args.output_root)
    pilot_core.fsync_dir(args.output_root.parent)
    print(
        "GPIC_TAR_MAPPING_COMPLETE "
        f"status={report['status']} comparisons={len(comparisons)} "
        f"byte_equal={report['byte_equal_comparisons']} "
        f"sha_equal={report['sha256_equal_comparisons']} "
        f"report={args.output_root / 'report.json'}",
        flush=True,
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--tar-root", type=Path, default=TAR_ROOT)
    parser.add_argument("--readme", type=Path, default=README_PATH)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--jobtag", required=True)
    parser.add_argument(
        "--tar-shards",
        default=",".join(str(value) for value in DEFAULT_TAR_SHARDS),
    )
    parser.add_argument("--samples-per-tar", type=int, default=DEFAULT_SAMPLES_PER_TAR)
    parser.add_argument("--tar-workers", type=int, default=DEFAULT_TAR_WORKERS)
    parser.add_argument("--max-catalog-row-groups", type=int, default=314)
    parser.add_argument("--source-prefix", default=SOURCE_PREFIX)
    parser.add_argument("--read-authority", required=True)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--production-shape", action="store_true")
    args = parser.parse_args(argv)
    args.tar_shards = parse_tar_shards(args.tar_shards)
    args.catalog_root = args.catalog_root.expanduser().resolve(strict=True)
    args.tar_root = args.tar_root.expanduser().resolve(strict=True)
    args.readme = args.readme.expanduser().resolve(strict=True)
    args.output_prefix = args.output_prefix.expanduser().resolve(strict=True)
    args.job_spec = args.job_spec.expanduser().resolve(strict=True)
    if args.read_authority != READ_AUTHORITY:
        raise ProbeError("explicit read-only diagnostic authority is absent")
    if not str(args.output_prefix).startswith("/fsx/"):
        raise ProbeError("tar-mapping evidence must be written under FSx")
    if not args.jobtag or Path(args.jobtag).name != args.jobtag:
        raise ProbeError("jobtag is malformed")
    if not (1 <= args.samples_per_tar <= 1_000):
        raise ProbeError("samples-per-tar must be in [1,1000]")
    if not (1 <= args.tar_workers <= 16):
        raise ProbeError("tar-workers must be in [1,16]")
    if not (1 <= args.max_catalog_row_groups <= 314):
        raise ProbeError("max-catalog-row-groups must be in [1,314]")
    if args.source_prefix != SOURCE_PREFIX:
        raise ProbeError("source-prefix drift")
    if args.production_shape and (
        args.tar_root != TAR_ROOT
        or args.readme != README_PATH
        or args.tar_shards != DEFAULT_TAR_SHARDS
        or args.samples_per_tar != DEFAULT_SAMPLES_PER_TAR
        or args.tar_workers != DEFAULT_TAR_WORKERS
        or args.max_catalog_row_groups != 314
    ):
        raise ProbeError("production diagnostic shape drift")
    args.output_root = args.output_prefix / args.jobtag
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
