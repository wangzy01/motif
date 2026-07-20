#!/usr/bin/env python3
"""Fully local/static fail-closed tests for the GPIC concurrency-shape probe."""

from __future__ import annotations

import json
import os
import tempfile
import types
from pathlib import Path
from unittest import mock

from scripts.offline_arb import run_gpic_mount_concurrency_shape_probe as probe


def expect_error(token: str, function, *args) -> None:
    try:
        function(*args)
    except probe.ProbeError as exc:
        if str(exc) != token:
            raise AssertionError(f"wrong safe error class: {exc!s}") from None
    else:
        raise AssertionError(f"missing expected error: {token}")


def make_item(root: Path, index: int, payload: bytes, ext: str = ".jpg") -> probe.SourceItem:
    shard = f"{index % 97:05d}"
    key = f"{index:064x}"
    directory = root / "images" / "train" / shard
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key + ext)
    path.write_bytes(payload)
    return probe.SourceItem(
        uid="gpic:" + key,
        src_root=str(root),
        src_shard=shard,
        src_key=key,
        ext=ext,
        width=512 + index % 5 * 64,
        height=512 + index % 7 * 64,
    )


def test_exact_read(root: Path) -> int:
    item = make_item(root, 1, b"abc" * 4096)
    result = probe.exact_source_read(item)
    assert result.payload == b"abc" * 4096
    assert result.elapsed_ns > 0
    assert len(result.identity_sha256) == 64

    empty = make_item(root, 2, b"")
    expect_error("source_empty", probe.exact_source_read, empty)

    short = make_item(root, 3, b"0123456789" * 1024)
    real_read = probe.os.read
    calls = 0

    def short_read(fd: int, count: int) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_read(fd, max(1, count // 2))
        return b""

    with mock.patch.object(probe.os, "read", short_read):
        expect_error("source_short_read", probe.exact_source_read, short)

    changed = make_item(root, 4, b"identity" * 1024)
    real_fstat = probe.os.fstat
    fstat_calls = 0

    def changed_fstat(fd: int):
        nonlocal fstat_calls
        value = real_fstat(fd)
        fstat_calls += 1
        if fstat_calls == 2:
            return types.SimpleNamespace(
                st_dev=value.st_dev,
                st_ino=value.st_ino,
                st_mode=value.st_mode,
                st_size=value.st_size,
                st_mtime_ns=value.st_mtime_ns + 1,
                st_ctime_ns=value.st_ctime_ns,
            )
        return value

    with mock.patch.object(probe.os, "fstat", changed_fstat):
        expect_error("source_identity_changed", probe.exact_source_read, changed)
    return 4


def test_selection() -> int:
    exts = (".jpg", ".png", ".jpeg", ".webp")
    items = [
        probe.SourceItem(
            uid="gpic:" + f"{index:064x}",
            src_root=probe.EXPECTED_SOURCE_ROOT,
            src_shard=f"{index % 1000:05d}",
            src_key=f"{index:064x}",
            ext=exts[index % len(exts)],
            width=384 + (index % 13) * 64,
            height=384 + (index % 11) * 64,
        )
        for index in range(probe.SOURCE_CACHE_ITEMS)
    ]
    first = probe.build_balanced_cells(items)
    second = probe.build_balanced_cells(list(reversed(items)))
    assert list(first) == [spec[0] for spec in probe.CELL_SPECS]
    assert all(len(value) == probe.PER_CELL_ITEMS for value in first.values())
    first_ids = {
        name: [item.anonymous_id() for item in value] for name, value in first.items()
    }
    second_ids = {
        name: [item.anonymous_id() for item in value] for name, value in second.items()
    }
    assert first_ids == second_ids
    flattened = [item for value in first.values() for item in value]
    assert len(flattened) == probe.CAPTURE_SELECTED_ITEMS
    assert len({item.anonymous_id() for item in flattened}) == len(flattened)
    for name, value in first.items():
        summary = probe.cell_selection_summary(value)
        assert summary["items"] == probe.PER_CELL_ITEMS
        assert len(summary["selection_sha256"]) == 64
        assert summary["unique_shards"] > 100
        assert sum(summary["format_counts"].values()) == probe.PER_CELL_ITEMS
    return len(flattened)


def test_execution_shapes(root: Path) -> int:
    shared_root = root / "shared-source"
    shared = [
        make_item(shared_root, 1000 + index, bytes([index % 251]) * (2048 + index * 11))
        for index in range(32)
    ]
    shared_result = probe.run_shared_cell(shared, 8, root / "shared-local")
    assert shared_result["mount_full_read"]["files"] == len(shared)
    assert shared_result["local_atomic_stage_verify"]["verified_files"] == len(shared)
    assert shared_result["mount_full_read"]["bytes"] == sum(
        len(bytes([index % 251]) * (2048 + index * 11)) for index in range(32)
    )

    fragment_root = root / "fragment-source"
    fragmented = [
        make_item(fragment_root, 2000 + index, bytes([index % 251]) * (3072 + index * 13))
        for index in range(80)
    ]
    fragment_result = probe.run_fragmented_cell(fragmented, 2, root / "fragment-local")
    assert fragment_result["mount_full_read"]["files"] == len(fragmented)
    assert fragment_result["local_atomic_stage_verify"]["verified_files"] == len(fragmented)
    assert fragment_result["mount_full_read"]["per_worker_files"] == [10] * 8
    return len(shared) + len(fragmented)


def test_artifact_safety(root: Path) -> int:
    item = make_item(root / "leak-source", 9999, b"payload")
    safe = probe.canonical_json_bytes(
        {
            "schema": "safe",
            "digest": probe.canonical_sha256({"not": "a source key"}),
            "count": 1,
        }
    )
    probe.assert_artifact_safe(safe, [item])
    expect_error(
        "artifact_source_path_leak",
        probe.assert_artifact_safe,
        probe.canonical_json_bytes({"bad": item.source_path()}),
        [item],
    )
    expect_error(
        "artifact_source_key_leak",
        probe.assert_artifact_safe,
        probe.canonical_json_bytes({"bad": item.src_key}),
        [item],
    )
    artifact = root / "artifact.json"
    probe.atomic_publish_json(artifact, {"ok": True})
    assert json.loads(artifact.read_text(encoding="ascii")) == {"ok": True}
    expect_error("artifact_clobber", probe.atomic_publish_json, artifact, {"ok": False})
    return 4


def decision_fixture(speed_by_cell: dict[str, float], byte_override: dict[str, int] | None = None):
    return {
        name: {
            "mount_full_read": {
                "files_per_second": speed_by_cell[name],
                "bytes": (byte_override or {}).get(name, 1000),
            }
        }
        for name, _shape, _processes, _threads in probe.CELL_SPECS
    }


def test_decision_split() -> int:
    shared_candidate = {
        "a16_x": 50.0,
        "b16_x": 30.0,
        "a08_mid": 40.0,
        "a24_mid": 55.0,
        "a32_mid": 60.0,
        "b16_y": 31.0,
        "a16_y": 51.0,
    }
    decision = probe.final_decision(decision_fixture(shared_candidate))
    assert decision["status"] == "COMPLETE_WITH_SHARED16_CANDIDATE"
    assert decision["current_fragmented_status"]["status"] == "FAIL"
    assert decision["shared16_candidate_status"]["status"] == "PASS"
    assert decision["local_stage_is_gate"] is False
    current = dict(shared_candidate, b16_x=45.0, b16_y=46.0)
    assert probe.final_decision(decision_fixture(current))["status"] == "COMPLETE_CURRENT_FRAGMENTED_PASS"
    curve = dict(shared_candidate, a16_x=30.0, a16_y=31.0)
    assert probe.final_decision(decision_fixture(curve))["status"] == "COMPLETE_WITH_SHARED_CURVE_CANDIDATE"
    no_candidate = {name: 20.0 for name in shared_candidate}
    assert probe.final_decision(decision_fixture(no_candidate))["status"] == "COMPLETE_NO_CAPACITY_CANDIDATE"
    unbalanced = probe.final_decision(
        decision_fixture(current, {"a16_x": 2000})
    )
    assert unbalanced["status"] == "INCONCLUSIVE_UNBALANCED_BYTES"
    return 5


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="gpic-shape-static-") as raw:
        root = Path(raw)
        exact_cases = test_exact_read(root / "exact")
        selected = test_selection()
        execution_files = test_execution_shapes(root / "execution")
        safety_cases = test_artifact_safety(root / "safety")
        decision_cases = test_decision_split()
    print(
        "GPIC_CONCURRENCY_SHAPE_STATIC_PASS "
        f"exact_cases={exact_cases} selected={selected} "
        f"execution_files={execution_files} safety_cases={safety_cases}",
        f"decision_cases={decision_cases}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
