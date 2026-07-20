#!/usr/bin/env python3
"""Synthetic/static gate for the read-only GPIC tar mapping probe."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import tarfile
from pathlib import Path
from typing import Sequence

import yaml

from scripts.offline_arb import probe_gpic_tar_mapping as probe


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o600
    archive.addfile(info, io.BytesIO(payload))


def write_fixture_tar(path: Path, *, broken_pair: bool = False) -> dict[str, bytes]:
    payloads = {
        "alpha.jpg": b"alpha-image-payload",
        "beta.png": b"beta-image-payload" * 3,
    }
    with tarfile.open(path, "w") as archive:
        if broken_pair:
            add_tar_bytes(archive, "alpha.jpg", payloads["alpha.jpg"])
            add_tar_bytes(archive, "alpha.json", b"{}")
        else:
            add_tar_bytes(archive, "alpha.json", b'{"key":"alpha"}')
            add_tar_bytes(archive, "alpha.jpg", payloads["alpha.jpg"])
        add_tar_bytes(archive, "beta.json", b'{"key":"beta"}')
        add_tar_bytes(archive, "beta.png", payloads["beta.png"])
    return payloads


def mapping_items(images: Path, payloads: dict[str, bytes]) -> list[probe.CatalogItem]:
    result = []
    for index, (name, payload) in enumerate(sorted(payloads.items())):
        path = images / name
        path.write_bytes(payload)
        result.append(
            probe.CatalogItem(
                local_index=index,
                row_id=100 + index,
                image_path=str(path),
                tar_shard=0,
                member_basename=name,
                catalog_row_group=index,
                catalog_row_group_offset=index,
                catalog_row_group_stratum=index,
            )
        )
    return result


def pure_contracts() -> None:
    path = probe.SOURCE_PREFIX + "07001/deadbeef.jpg"
    require(probe.parse_catalog_path(path) == (7001, "deadbeef.jpg"), "path mapping")
    require(probe.expected_tar_name(7001) == "gpic_train_07001.tar", "tar name")
    require(
        probe.parse_tar_shards("0,7,7999") == (0, 7, 7999),
        "tar shard parser",
    )
    require(
        [probe.row_group_stratum(index, 314, 16) for index in (0, 156, 313)]
        == [0, 7, 15],
        "catalog first/middle/last stratum mapping",
    )
    for bad in (
        probe.SOURCE_PREFIX + "7001/deadbeef.jpg",
        probe.SOURCE_PREFIX + "07001/nested/deadbeef.jpg",
        probe.SOURCE_PREFIX + "08000/deadbeef.jpg",
        probe.SOURCE_PREFIX + "07001/deadbeef.gif",
    ):
        try:
            probe.parse_catalog_path(bad)
        except probe.ProbeError:
            pass
        else:
            raise RuntimeError(f"malformed path accepted: {bad}")
    for raw in ("", "1,1", "2,1", "-1", "8000"):
        try:
            probe.parse_tar_shards(raw)
        except probe.ProbeError:
            pass
        else:
            raise RuntimeError(f"malformed tar shard list accepted: {raw}")


def synthetic_tar_contracts(root: Path) -> None:
    tar_root = root / "tar"
    images = root / "images"
    tar_root.mkdir(parents=True)
    images.mkdir()
    tar_path = tar_root / "gpic_train_00000.tar"
    payloads = write_fixture_tar(tar_path)
    items = mapping_items(images, payloads)
    value = probe.inspect_one_tar(tar_root=tar_root, shard=0, items=items)
    require(value["pairing_valid"], "valid JSON/image pair order rejected")
    require(value["exact_payload_equivalence"], "equal tar/unzipped bytes rejected")
    require(value["found_members"] == 2, "selected member count")
    require(
        all(item["bytes_equal"] and item["sha256_equal"] for item in value["comparisons"]),
        "comparison exactness",
    )

    (images / "alpha.jpg").write_bytes(b"corrupted")
    mismatch = probe.inspect_one_tar(tar_root=tar_root, shard=0, items=items)
    require(not mismatch["exact_payload_equivalence"], "payload corruption accepted")
    (images / "alpha.jpg").write_bytes(payloads["alpha.jpg"])

    tar_path.unlink()
    write_fixture_tar(tar_path, broken_pair=True)
    broken = probe.inspect_one_tar(tar_root=tar_root, shard=0, items=items)
    require(not broken["pairing_valid"], "broken tar pair ordering accepted")

    missing_item = probe.CatalogItem(
        local_index=9,
        row_id=109,
        image_path=str(images / "missing.jpg"),
        tar_shard=0,
        member_basename="missing.jpg",
        catalog_row_group=0,
        catalog_row_group_offset=9,
        catalog_row_group_stratum=0,
    )
    missing = probe.inspect_one_tar(tar_root=tar_root, shard=0, items=[missing_item])
    require(not missing["exact_payload_equivalence"], "missing tar member accepted")
    require(missing["missing_member_basenames"] == ["missing.jpg"], "missing binding")


def synthetic_inventory(root: Path) -> None:
    inventory_root = root / "inventory"
    inventory_root.mkdir()
    original = probe.TRAIN_TAR_COUNT
    try:
        probe.TRAIN_TAR_COUNT = 4
        for shard in range(4):
            (inventory_root / f"gpic_train_{shard:05d}.tar").write_bytes(b"x")
        (inventory_root / "test.txt").write_text("known extra", encoding="utf-8")
        value = probe.inventory_train_tars(inventory_root)
        require(value["exact_tar_name_set"], "synthetic exact tar inventory rejected")
        require(value["observed_tar_count"] == 4, "synthetic inventory count")
        require(value["non_tar_entries"] == ["test.txt"], "non-tar evidence")
        (inventory_root / "gpic_train_00003.tar").unlink()
        drift = probe.inventory_train_tars(inventory_root)
        require(not drift["exact_tar_name_set"], "missing tar inventory accepted")
    finally:
        probe.TRAIN_TAR_COUNT = original


def static_source_contracts(probe_job_spec: Path) -> None:
    runner_path = Path(probe.__file__).resolve(strict=True)
    runner = runner_path.read_text(encoding="utf-8")
    job_raw = probe_job_spec.read_text(encoding="utf-8")
    job = yaml.safe_load(job_raw)
    require(job.get("numWorkers") == 1, "probe must be one JCC worker")
    require(job.get("gpuPerWorker") == 0, "probe must not request GPU")
    require(job.get("cpuPerWorker") == 16, "probe must request CPU16")
    require(job.get("maxRetries") == 0, "probe retries must be disabled")
    command = job.get("command")
    require(isinstance(command, str), "probe YAML command")
    required = (
        "probe_gpic_tar_mapping.py",
        "--samples-per-tar 16",
        "--tar-workers 8",
        "--max-catalog-row-groups 314",
        "--production-shape",
        "offline-gpic-tar-mapping-read-only-v1",
        "/s3-mewtant-raw-datasets/stanford-vision-lab/gpic/train",
        "/s3-mewtant-raw-datasets/stanford-vision-lab/gpic_umziped/images/train/",
        "OUTPUT_PREFIX=/fsx/",
        "EXPECTED_RUNNER_SHA=",
    )
    for token in required:
        require(token in command, f"probe YAML token missing: {token}")
    forbidden = (
        "/s3/evan",
        "--s3-prefix",
        "write-authority",
        "StagedS3BlobWriter",
        "write_s3",
    )
    for token in forbidden:
        require(token not in command, f"probe YAML leaked write surface: {token}")
        require(token not in runner, f"probe runner leaked write surface: {token}")
    expected_sha_line = next(
        line.strip() for line in command.splitlines() if "EXPECTED_RUNNER_SHA=" in line
    )
    expected_sha = expected_sha_line.split("=", 1)[1]
    require(probe.is_sha256(expected_sha), "runner SHA is not frozen in YAML")
    require(
        hashlib.sha256(runner_path.read_bytes()).hexdigest() == expected_sha,
        "runner/YAML SHA mismatch",
    )
    require("os.O_RDONLY" in runner, "unzipped source is not explicitly read-only")
    require('tarfile.open(tar_path, mode="r:")' in runner, "tar source is not read-only")
    for token in (
        '"catalog_hash": catalog.catalog_hash',
        '"gpic_rows": catalog.gpic_rows',
        '"gpic_sha256": catalog.gpic_sha256',
        '"catalog_row_group_stratum"',
        '"required_strata_per_tar"',
        '"exact_catalog_tar_payload_equivalence"',
    ):
        require(token in runner, f"mapping output-contract token missing: {token}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--probe-job-spec", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.root.exists() or args.root.is_symlink():
        raise RuntimeError("static root is not fresh")
    args.root.mkdir(parents=True)
    try:
        pure_contracts()
        synthetic_tar_contracts(args.root / "tar-contracts")
        synthetic_inventory(args.root)
        static_source_contracts(args.probe_job_spec.resolve(strict=True))
    finally:
        shutil.rmtree(args.root)
    print("GPIC_TAR_MAPPING_STATIC_PASS comparisons=2 negatives=path,shard,bytes,pair,missing,inventory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
