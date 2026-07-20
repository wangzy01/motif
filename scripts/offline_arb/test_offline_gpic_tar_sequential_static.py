#!/usr/bin/env python3
"""Synthetic/static red-team gate for the production GPIC tar adapter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from scripts.offline_arb import offline_gpic_s3_production_common as common
from scripts.offline_arb import offline_gpic_tar_sequential as reader


SOURCE_PREFIX = "/fixture/gpic_umziped/images/train/"
ROW_BASE = 10_000
CATALOG_HASH = "1" * 64
GPIC_SHA256 = "2" * 64
PLAN_RAW_SHA256 = "3" * 64
PLAN_SHA256 = "4" * 64


@dataclass(frozen=True)
class Item:
    local_index: int
    row_id: int
    image_path: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def expect_failure(call: Callable[[], object], label: str) -> None:
    try:
        call()
    except common.ProductionError:
        return
    raise RuntimeError(f"negative case was accepted: {label}")


def make_item(local_index: int, shard: int, member: str) -> Item:
    return Item(
        local_index=local_index,
        row_id=ROW_BASE + local_index,
        image_path=f"{SOURCE_PREFIX}{shard:05d}/{member}",
    )


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o600
    info.uid = 0
    info.gid = 0
    info.mtime = 0
    archive.addfile(info, io.BytesIO(payload))


def write_tar(
    path: Path,
    pairs: Sequence[tuple[str, bytes]],
    *,
    broken_pair: bool = False,
    nonregular: bool = False,
) -> None:
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        if nonregular:
            info = tarfile.TarInfo("forbidden")
            info.type = tarfile.DIRTYPE
            info.mode = 0o700
            archive.addfile(info)
        for index, (image_name, payload) in enumerate(pairs):
            stem = Path(image_name).stem
            if broken_pair and index == 0:
                add_bytes(archive, image_name, payload)
                add_bytes(archive, f"{stem}.json", b"{}")
            else:
                add_bytes(archive, f"{stem}.json", b'{"caption":"fixture"}')
                add_bytes(archive, image_name, payload)


def mapping(items: Sequence[Item], owners: dict[int, int]) -> dict:
    return reader.scan_worker_mapping(
        iter(items),
        build_jobtag="fixture-build",
        worker_id=0,
        workers=1,
        source_prefix=SOURCE_PREFIX,
        row_base=ROW_BASE,
        gpic_rows=len(items),
        expected_rows=len(items),
        shard_owners=owners,
        catalog_hash=CATALOG_HASH,
        gpic_sha256=GPIC_SHA256,
        partition_plan_raw_sha256=PLAN_RAW_SHA256,
        partition_plan_sha256=PLAN_SHA256,
    )


def run_reader(
    items: Sequence[Item],
    authority: dict,
    owners: dict[int, int],
    tar_root: Path,
    *,
    tar_workers: int,
    attempts: int = 1,
    max_selected_bytes: int = 1 << 30,
    max_archive_bytes: int = 1 << 30,
    max_selected_bytes_per_tar: int = 1 << 28,
    telemetry: list[dict] | None = None,
) -> list[reader.TarSourceRead]:
    return list(
        reader.iter_tar_sequential_reads(
            iter(items),
            worker_mapping=authority,
            worker_id=0,
            workers=1,
            shard_owners=owners,
            source_prefix=SOURCE_PREFIX,
            tar_root=tar_root,
            row_base=ROW_BASE,
            gpic_rows=len(items),
            expected_rows=len(items),
            attempts=attempts,
            tar_workers=tar_workers,
            max_pending_tars=tar_workers,
            read_buffer_bytes=1 << 20,
            io_block_bytes=4096,
            max_selected_bytes=max_selected_bytes,
            max_archive_bytes=max_archive_bytes,
            max_selected_bytes_per_tar=max_selected_bytes_per_tar,
            telemetry_callback=(telemetry.append if telemetry is not None else None),
        )
    )


def positive_and_ordering(root: Path) -> tuple[list[Item], dict[int, int], dict]:
    tar_root = root / "valid"
    tar_root.mkdir()
    alpha = b"alpha-payload" * 31
    beta = b"beta-payload" * 47
    gamma = b"gamma-payload" * 53
    # Catalog/structural order is alpha,beta.  Archive order is beta,alpha.
    items = [
        make_item(0, 0, "alpha.jpg"),
        make_item(1, 0, "beta.png"),
        make_item(2, 1, "gamma.jpg"),
    ]
    owners = {0: 0, 1: 0}
    authority = mapping(items, owners)
    write_tar(
        tar_root / reader.expected_tar_name(0),
        [("beta.png", beta), ("alpha.jpg", alpha), ("unused.jpg", b"unused")],
    )
    write_tar(
        tar_root / reader.expected_tar_name(1),
        [("gamma.jpg", gamma)],
    )
    expected_names = ["beta.png", "alpha.jpg", "gamma.jpg"]
    expected_payloads = [beta, alpha, gamma]
    for tar_workers in (1, 2, 4):
        telemetry: list[dict] = []
        values = run_reader(
            items,
            authority,
            owners,
            tar_root,
            tar_workers=tar_workers,
            telemetry=telemetry,
        )
        names = [Path(value.item.image_path).name for value in values]
        require(names == expected_names, f"ordered output drift workers={tar_workers}")
        require(
            [value.payload for value in values] == expected_payloads,
            "selected payload drift",
        )
        require(
            all(
                value.sha256_raw == hashlib.sha256(value.payload).digest()
                and value.attempts == 1
                and value.errors == 0
                for value in values
            ),
            "SourceRead compatibility/integrity drift",
        )
        require(
            [value["tar_shard"] for value in telemetry] == [0, 1],
            "per-tar telemetry order drift",
        )
        require(
            all(
                value["archive_bytes"] == value["archive_bytes_read"]
                and value["header_count"]
                == value["checksum_validated_headers"]
                == value["regular_member_count"]
                and value["json_member_count"] == value["image_member_count"]
                and value["extra_local_chunk_bytes"] == 0
                and value["fsx_writes"] == 0
                and value["s3_writes"] == 0
                and value["tar_telemetry_sha256"]
                == common.self_hash(value, "tar_telemetry_sha256")
                for value in telemetry
            ),
            "per-tar telemetry closure drift",
        )
    return items, owners, authority


def structural_negatives(items: list[Item], owners: dict[int, int]) -> None:
    duplicate = [items[0], make_item(1, 0, "alpha.jpg"), items[2]]
    expect_failure(lambda: mapping(duplicate, owners), "duplicate mapping key")
    expect_failure(
        lambda: mapping([items[1], items[0], items[2]], owners),
        "mapping order regression",
    )
    duplicate_row = [items[0], make_item(0, 0, "beta.png"), items[2]]
    expect_failure(lambda: mapping(duplicate_row, owners), "duplicate explicit row")
    expect_failure(
        lambda: reader.scan_worker_mapping(
            iter(items),
            build_jobtag="fixture-build",
            worker_id=0,
            workers=1,
            source_prefix=SOURCE_PREFIX,
            row_base=ROW_BASE,
            gpic_rows=len(items),
            expected_rows=len(items),
            shard_owners={0: 0, 1: 1},
            catalog_hash=CATALOG_HASH,
            gpic_sha256=GPIC_SHA256,
            partition_plan_raw_sha256=PLAN_RAW_SHA256,
            partition_plan_sha256=PLAN_SHA256,
        ),
        "foreign shard ownership",
    )
    malformed = [make_item(0, 0, "nested/name.jpg")]
    expect_failure(lambda: mapping(malformed, {0: 0}), "nested catalog member")


def copy_tars(source: Path, target: Path) -> None:
    target.mkdir()
    for path in source.iterdir():
        shutil.copyfile(path, target / path.name)


def reader_negatives(
    root: Path, items: list[Item], owners: dict[int, int], authority: dict
) -> None:
    valid = root / "valid"
    expect_failure(
        lambda: run_reader(
            items, authority, owners, valid, tar_workers=1, attempts=3
        ),
        "attempts != 1",
    )
    expect_failure(
        lambda: run_reader(
            items,
            authority,
            owners,
            valid,
            tar_workers=2,
            max_archive_bytes=1,
        ),
        "archive hard cap",
    )
    expect_failure(
        lambda: run_reader(
            items,
            authority,
            owners,
            valid,
            tar_workers=2,
            max_selected_bytes=10,
            max_selected_bytes_per_tar=10,
        ),
        "selected payload cap",
    )

    tampered_authority = copy.deepcopy(authority)
    tampered_authority["shards"][0]["mapping_sha256"] = "f" * 64
    tampered_authority["worker_mapping_sha256"] = common.self_hash(
        tampered_authority, "worker_mapping_sha256"
    )
    expect_failure(
        lambda: run_reader(
            items, tampered_authority, owners, valid, tar_workers=2
        ),
        "structural shard digest",
    )

    missing_root = root / "missing"
    missing_root.mkdir()
    write_tar(
        missing_root / reader.expected_tar_name(0),
        [("alpha.jpg", b"alpha"), ("unused.jpg", b"unused")],
    )
    shutil.copyfile(
        valid / reader.expected_tar_name(1),
        missing_root / reader.expected_tar_name(1),
    )
    expect_failure(
        lambda: run_reader(items, authority, owners, missing_root, tar_workers=2),
        "missing selected target",
    )

    pair_root = root / "broken-pair"
    pair_root.mkdir()
    write_tar(
        pair_root / reader.expected_tar_name(0),
        [("alpha.jpg", b"alpha"), ("beta.png", b"beta")],
        broken_pair=True,
    )
    shutil.copyfile(
        valid / reader.expected_tar_name(1),
        pair_root / reader.expected_tar_name(1),
    )
    expect_failure(
        lambda: run_reader(items, authority, owners, pair_root, tar_workers=2),
        "JSON/image pair order",
    )

    type_root = root / "nonregular"
    type_root.mkdir()
    write_tar(
        type_root / reader.expected_tar_name(0),
        [("alpha.jpg", b"alpha"), ("beta.png", b"beta")],
        nonregular=True,
    )
    shutil.copyfile(
        valid / reader.expected_tar_name(1),
        type_root / reader.expected_tar_name(1),
    )
    expect_failure(
        lambda: run_reader(items, authority, owners, type_root, tar_workers=1),
        "nonregular tar member",
    )

    checksum_root = root / "checksum"
    copy_tars(valid, checksum_root)
    checksum_path = checksum_root / reader.expected_tar_name(0)
    with checksum_path.open("r+b", buffering=0) as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
        handle.flush()
        os.fsync(handle.fileno())
    expect_failure(
        lambda: run_reader(items, authority, owners, checksum_root, tar_workers=2),
        "tar header checksum",
    )

    truncated_root = root / "truncated"
    copy_tars(valid, truncated_root)
    truncated = truncated_root / reader.expected_tar_name(0)
    with truncated.open("r+b", buffering=0) as handle:
        handle.truncate(700)
        handle.flush()
        os.fsync(handle.fileno())
    expect_failure(
        lambda: run_reader(items, authority, owners, truncated_root, tar_workers=1),
        "short tar read",
    )

    padding_root = root / "padding"
    copy_tars(valid, padding_root)
    padding = padding_root / reader.expected_tar_name(0)
    with padding.open("r+b", buffering=0) as handle:
        handle.seek(-1, os.SEEK_END)
        handle.write(b"X")
        handle.flush()
        os.fsync(handle.fileno())
    expect_failure(
        lambda: run_reader(items, authority, owners, padding_root, tar_workers=1),
        "nonzero tar end padding",
    )


def full_authority_contract(root: Path, authority: dict) -> None:
    sample_binding = {
        "schema_version": reader.SAMPLE_BINDING_SCHEMA,
        "raw_sha256": "5" * 64,
    }
    sample_binding["sample_binding_sha256"] = common.self_hash(
        sample_binding, "sample_binding_sha256"
    )
    value = reader.make_full_mapping_authority(
        [authority],
        build_jobtag="fixture-build",
        workers=1,
        shard_owners={0: 0, 1: 0},
        source_prefix=SOURCE_PREFIX,
        tar_root=root / "valid",
        row_base=ROW_BASE,
        gpic_rows=3,
        catalog_hash=CATALOG_HASH,
        gpic_sha256=GPIC_SHA256,
        partition_plan_raw_sha256=PLAN_RAW_SHA256,
        partition_plan_sha256=PLAN_SHA256,
        sample_binding=sample_binding,
        production_gate=False,
    )
    publication = root / "authority"
    publication.mkdir()
    raw = common.canonical_json_bytes(value)
    report = publication / "mapping.json"
    report.write_bytes(raw)
    raw_sha = hashlib.sha256(raw).hexdigest()
    (publication / "_COMPLETE").write_bytes((raw_sha + "\n").encode("ascii"))
    loaded = reader.validate_published_full_mapping(
        report,
        expected_raw_sha256=raw_sha,
        expected_self_sha256=value["full_mapping_sha256"],
    )
    require(loaded == value, "published full authority changed")
    (publication / "_COMPLETE").write_text("0" * 64 + "\n", encoding="ascii")
    expect_failure(
        lambda: reader.validate_published_full_mapping(
            report,
            expected_raw_sha256=raw_sha,
            expected_self_sha256=value["full_mapping_sha256"],
        ),
        "full authority COMPLETE marker",
    )


def real_sample_authority(path: Path, raw_sha256: str) -> None:
    binding = reader.load_sample_mapping_authority(
        path,
        expected_raw_sha256=raw_sha256,
        catalog_hash="b5f83014e4eab304460336adff217c35e993a2d892a5d889f69f87013c65f212",
        gpic_rows=20_000_000,
        row_base=56_497_535,
        gpic_sha256="bf2883d3665d8730c157d32180a81c3ad888b219b8f58c0e049ae7317cf4b0d8",
        tar_root=reader.PRODUCTION_TAR_ROOT,
        source_prefix=reader.PRODUCTION_SOURCE_PREFIX,
        production_gate=True,
    )
    require(binding["raw_sha256"] == raw_sha256, "real sample raw SHA")
    require(binding["sample_comparisons"] == 256, "real sample comparisons")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-report", type=Path)
    parser.add_argument("--sample-report-raw-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.root.exists() or args.root.is_symlink():
        raise RuntimeError("static test root must be fresh")
    if (args.sample_report is None) != (args.sample_report_raw_sha256 is None):
        raise RuntimeError("sample report and SHA must be supplied together")
    args.root.mkdir(parents=True)
    try:
        items, owners, authority = positive_and_ordering(args.root)
        structural_negatives(items, owners)
        reader_negatives(args.root, items, owners, authority)
        full_authority_contract(args.root, authority)
        if args.sample_report is not None:
            real_sample_authority(
                args.sample_report.resolve(strict=True),
                args.sample_report_raw_sha256,
            )
    finally:
        shutil.rmtree(args.root)
    print(
        "OFFLINE_GPIC_TAR_SEQUENTIAL_STATIC_PASS "
        "ordered_workers=1,2,4 negatives=duplicate,owner,attempt,cap,digest,"
        "missing,pair,type,checksum,short,padding,complete"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
