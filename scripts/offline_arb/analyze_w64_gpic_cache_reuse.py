#!/usr/bin/env python3
"""Streaming W64-plan analysis for a hypothetical node-local GPIC cache.

This is an evidence-only analysis.  It reads the frozen plan (never an image),
uses exactly formal optimizer steps ``[0, 166667)``, and reports:

* exact GPIC exposures and cold/infinite-cache reuse for every rank and node;
* exact LRU stack-distance quantiles for a deterministic logical node order;
* equal-sized-object LRU sensitivity for 64..1024 GiB caches.

The real loader has eight independently scheduled ranks on each node.  Its
wall-clock interleaving is not represented by the plan, so capacity-limited
LRU results are explicitly a logical-order model, not a measured hit rate.
Cold infinite-cache results are order independent and exact.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import math
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from training.dataline_offline.planner_v1 import OfflinePlan, PlannerInputError


SCHEMA_VERSION = "offline_arb_w64_gpic_node_cache_reuse_v1"
FORMAL_START_STEP = 0
FORMAL_STOP_STEP = 166_667
EXPECTED_WORLD_SIZE = 64
EXPECTED_PER_RANK_BS = 36
EXPECTED_GLOBAL_BATCH = 2_304
EXPECTED_FORMAL_EXPOSURES = 384_000_768
EXPECTED_PLAN_BLOCKS = 168
EXPECTED_BLOCK_STEPS = 1_024
EXPECTED_FREEZE_SCHEMA = "offline_arb_freeze_v3_caption_sidecar"
RANKS_PER_NODE = 8
NODE_COUNT = 8
TARGET_DOMAIN = "gpic"
CACHE_GIB_VALUES = (64, 128, 256, 512, 1024)
DEFAULT_AVERAGE_OBJECT_BYTES = (
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
)
QUANTILE_BASIS_POINTS = (0, 100, 500, 1000, 2500, 5000, 7500, 9000, 9500, 9900, 9990, 10000)


# The helper computes exact Mattson stack distance with a Fenwick tree.  It is
# embedded so the committed Python source SHA also binds the native algorithm;
# compilation and execution happen only inside the CPU JCC job.
CPP_SOURCE = r'''// offline_arb_w64_gpic_cache_reuse_fenwick_v1
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the uint32 sequence contract requires a little-endian host"
#endif

static long long parse_positive(const char* text, const char* label) {
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (errno || end == text || *end != '\0' || value <= 0) {
        throw std::runtime_error(std::string("invalid ") + label);
    }
    return value;
}

static std::vector<uint64_t> parse_capacities(const char* text) {
    std::vector<uint64_t> out;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) throw std::runtime_error("empty capacity token");
        const long long value = parse_positive(token.c_str(), "capacity");
        out.push_back(static_cast<uint64_t>(value));
    }
    if (out.empty()) throw std::runtime_error("no capacities");
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static inline int32_t prefix_sum(const std::vector<int32_t>& bit, int32_t index) {
    int32_t total = 0;
    for (int32_t i = index; i > 0; i -= i & -i) total += bit[static_cast<size_t>(i)];
    return total;
}

static inline void add(std::vector<int32_t>& bit, int32_t index, int32_t delta) {
    const int32_t limit = static_cast<int32_t>(bit.size());
    for (int32_t i = index; i < limit; i += i & -i) bit[static_cast<size_t>(i)] += delta;
}

static void write_no_clobber(const std::string& path, const std::string& payload) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) throw std::runtime_error("cannot create result without clobber");
    size_t offset = 0;
    while (offset < payload.size()) {
        const ssize_t wrote = ::write(fd, payload.data() + offset, payload.size() - offset);
        if (wrote < 0) {
            const int saved = errno;
            ::close(fd);
            errno = saved;
            throw std::runtime_error("result write failed");
        }
        offset += static_cast<size_t>(wrote);
    }
    if (::fsync(fd) != 0) {
        const int saved = errno;
        ::close(fd);
        errno = saved;
        throw std::runtime_error("result fsync failed");
    }
    if (::close(fd) != 0) throw std::runtime_error("result close failed");
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: helper SEQUENCE DOMAIN_SIZE CAPACITIES RESULT");
        const std::string sequence_path(argv[1]);
        const long long domain_size_ll = parse_positive(argv[2], "domain size");
        if (domain_size_ll > std::numeric_limits<int32_t>::max()) throw std::runtime_error("domain too large");
        const int32_t domain_size = static_cast<int32_t>(domain_size_ll);
        const std::vector<uint64_t> capacities = parse_capacities(argv[3]);
        const std::string output_path(argv[4]);

        const int fd = ::open(sequence_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) throw std::runtime_error("cannot open sequence");
        struct stat st {};
        if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0 || st.st_size % 4 != 0) {
            ::close(fd);
            throw std::runtime_error("invalid sequence file");
        }
        const uint64_t event_count_u64 = static_cast<uint64_t>(st.st_size) / 4U;
        if (event_count_u64 > static_cast<uint64_t>(std::numeric_limits<int32_t>::max() - 1)) {
            ::close(fd);
            throw std::runtime_error("sequence too long for v1 helper");
        }
        const int32_t event_count = static_cast<int32_t>(event_count_u64);
        void* mapped = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("sequence mmap failed");
        }
        (void)::madvise(mapped, static_cast<size_t>(st.st_size), MADV_SEQUENTIAL);
        const auto* ids = static_cast<const uint32_t*>(mapped);

        std::vector<int32_t> last(static_cast<size_t>(domain_size), -1);
        std::vector<int32_t> bit(static_cast<size_t>(event_count) + 1U, 0);
        std::vector<uint64_t> histogram(static_cast<size_t>(domain_size), 0);
        uint64_t cold = 0;
        uint64_t reuses = 0;
        int32_t max_distance = -1;

        for (int32_t position = 0; position < event_count; ++position) {
            const uint32_t id = ids[static_cast<size_t>(position)];
            if (id >= static_cast<uint32_t>(domain_size)) {
                ::munmap(mapped, static_cast<size_t>(st.st_size));
                ::close(fd);
                throw std::runtime_error("sequence object id outside domain");
            }
            const int32_t previous = last[static_cast<size_t>(id)];
            if (previous < 0) {
                ++cold;
            } else {
                const int32_t active = prefix_sum(bit, position);
                const int32_t through_previous = prefix_sum(bit, previous + 1);
                const int32_t distance = active - through_previous;
                if (distance < 0 || distance >= domain_size) {
                    ::munmap(mapped, static_cast<size_t>(st.st_size));
                    ::close(fd);
                    throw std::runtime_error("internal stack distance outside domain");
                }
                ++histogram[static_cast<size_t>(distance)];
                ++reuses;
                if (distance > max_distance) max_distance = distance;
                add(bit, previous + 1, -1);
            }
            add(bit, position + 1, 1);
            last[static_cast<size_t>(id)] = position;
        }

        if (cold + reuses != event_count_u64) throw std::runtime_error("event accounting mismatch");

        std::vector<std::pair<uint64_t, int32_t>> quantile_targets;
        const int qbps[] = {0,100,500,1000,2500,5000,7500,9000,9500,9900,9990,10000};
        if (reuses > 0) {
            for (const int bp : qbps) {
                const uint64_t rank = bp == 0 ? 0 : ((static_cast<uint64_t>(bp) * reuses + 9999U) / 10000U - 1U);
                quantile_targets.emplace_back(rank, bp);
            }
            std::sort(quantile_targets.begin(), quantile_targets.end());
        }
        std::vector<std::pair<int32_t, int32_t>> quantile_values;
        size_t qcursor = 0;
        uint64_t cumulative = 0;
        for (int32_t distance = 0; distance < domain_size && qcursor < quantile_targets.size(); ++distance) {
            cumulative += histogram[static_cast<size_t>(distance)];
            while (qcursor < quantile_targets.size() && cumulative > quantile_targets[qcursor].first) {
                quantile_values.emplace_back(quantile_targets[qcursor].second, distance);
                ++qcursor;
            }
        }
        if (qcursor != quantile_targets.size()) throw std::runtime_error("quantile accounting mismatch");

        std::vector<std::pair<uint64_t, uint64_t>> capacity_hits;
        uint64_t hit_cumulative = 0;
        size_t distance_cursor = 0;
        for (const uint64_t capacity : capacities) {
            const size_t limit = static_cast<size_t>(std::min<uint64_t>(capacity, static_cast<uint64_t>(domain_size)));
            while (distance_cursor < limit) {
                hit_cumulative += histogram[distance_cursor];
                ++distance_cursor;
            }
            capacity_hits.emplace_back(capacity, hit_cumulative);
        }

        if (::munmap(mapped, static_cast<size_t>(st.st_size)) != 0) {
            ::close(fd);
            throw std::runtime_error("sequence munmap failed");
        }
        if (::close(fd) != 0) throw std::runtime_error("sequence close failed");

        std::ostringstream result;
        result << "schema\tfenwick_stack_distance_v1\n";
        result << "events\t" << event_count_u64 << "\n";
        result << "cold\t" << cold << "\n";
        result << "reuses\t" << reuses << "\n";
        result << "max_distance\t" << max_distance << "\n";
        for (const auto& item : quantile_values) result << "quantile\t" << item.first << "\t" << item.second << "\n";
        for (const auto& item : capacity_hits) result << "capacity\t" << item.first << "\t" << item.second << "\n";
        write_no_clobber(output_path, result.str());
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "FATAL: " << exc.what() << "\n";
        return 2;
    }
}
'''


class EvidenceError(RuntimeError):
    """The evidence contract was not met."""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=1024 * 1024) as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be a JSON object")
    return value


def _require_canonical_file(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise EvidenceError(f"{label} must be an absolute regular non-symlink: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise EvidenceError(f"{label} is not canonical: {path} -> {resolved}")
    return resolved


def _require_canonical_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise EvidenceError(f"{label} must be an absolute directory non-symlink: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise EvidenceError(f"{label} is not canonical: {path} -> {resolved}")
    return resolved


def _stat_fingerprint(path: Path) -> dict[str, int]:
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise EvidenceError(f"not a regular input: {path}")
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _directory_fingerprint(path: Path) -> dict[str, int]:
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(value.st_mode):
        raise EvidenceError(f"not a directory input: {path}")
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _require_frozen_fingerprint(record: Mapping[str, Any], live: Mapping[str, int], label: str) -> None:
    frozen = record.get("fingerprint")
    if not isinstance(frozen, Mapping) or dict(frozen) != dict(live):
        raise EvidenceError(f"freeze five-field fingerprint mismatch for {label}")


def _fraction(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise EvidenceError(f"invalid fraction {numerator}/{denominator}")
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "decimal": format(numerator / denominator, ".12f"),
    }


def _parse_average_bytes(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(token) for token in raw.split(",") if token}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("average object bytes must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("average object bytes must be positive")
    return values


def _same_path_record(record: Mapping[str, Any], path: Path, label: str) -> None:
    if not isinstance(record, Mapping):
        raise EvidenceError(f"freeze {label} must be an object")
    if record.get("path") != str(path):
        raise EvidenceError(f"freeze {label} path mismatch")


def _freeze_binding(
    freeze_path: Path,
    freeze: Mapping[str, Any],
    plan_root: Path,
    header_path: Path,
    checksums_path: Path,
    header_sha: str,
    checksums_sha: str,
    checksums: Mapping[str, Any],
    block_steps: int,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    if block_steps <= 0:
        raise EvidenceError("plan header has an invalid block_steps value")
    if freeze.get("schema_version") != EXPECTED_FREEZE_SCHEMA:
        raise EvidenceError(f"freeze schema must be {EXPECTED_FREEZE_SCHEMA}")
    inputs = freeze.get("inputs")
    plan_blocks = freeze.get("plan_blocks")
    identities = freeze.get("identities")
    if not isinstance(inputs, Mapping) or not isinstance(plan_blocks, Mapping) or not isinstance(identities, Mapping):
        raise EvidenceError("freeze is missing inputs/plan_blocks/identities objects")
    header_record = inputs.get("plan_header")
    checksums_record = inputs.get("plan_checksums")
    _same_path_record(header_record, header_path, "inputs.plan_header")
    _same_path_record(checksums_record, checksums_path, "inputs.plan_checksums")
    _require_frozen_fingerprint(header_record, _stat_fingerprint(header_path), "inputs.plan_header")
    _require_frozen_fingerprint(checksums_record, _stat_fingerprint(checksums_path), "inputs.plan_checksums")
    if header_record.get("sha256") != header_sha or checksums_record.get("sha256") != checksums_sha:
        raise EvidenceError("freeze plan header/checksums SHA mismatch")
    if int(header_record.get("bytes", -1)) != header_path.stat().st_size:
        raise EvidenceError("freeze plan header byte count mismatch")
    if int(checksums_record.get("bytes", -1)) != checksums_path.stat().st_size:
        raise EvidenceError("freeze plan checksums byte count mismatch")
    if identities.get("plan_id") != "stage256_offline_exact_v1_w64_prod":
        raise EvidenceError("freeze plan identity is not w64_prod")
    if plan_blocks.get("root") != str(plan_root):
        raise EvidenceError("freeze plan_blocks.root mismatch")
    directories = plan_blocks.get("directories")
    if not isinstance(directories, Mapping) or set(directories) != {"root", "blocks"}:
        raise EvidenceError("freeze plan directory records must be exactly root/blocks")
    for name, path in (("root", plan_root), ("blocks", plan_root / "blocks")):
        record = directories[name]
        _same_path_record(record, path, f"plan_blocks.directories.{name}")
        _require_frozen_fingerprint(
            record, _directory_fingerprint(path), f"plan_blocks.directories.{name}"
        )
    files = plan_blocks.get("files")
    expected_blocks = checksums.get("blocks")
    if not isinstance(files, Mapping) or not isinstance(expected_blocks, Mapping):
        raise EvidenceError("freeze/checksums block maps are missing")
    if set(files) != set(expected_blocks) or len(files) != EXPECTED_PLAN_BLOCKS:
        raise EvidenceError("freeze block set does not exactly match all 168 checksums")
    plan_entries = {entry.name for entry in plan_root.iterdir()}
    if plan_entries != {"header.json", "checksums.json", "blocks"}:
        raise EvidenceError(f"plan root entry set drift: {sorted(plan_entries)}")
    block_entries = {entry.name for entry in (plan_root / "blocks").iterdir()}
    if block_entries != set(expected_blocks):
        raise EvidenceError("plan blocks directory entry set is not the frozen 168-file set")
    snapshots: dict[str, dict[str, int]] = {}
    manifest: list[dict[str, Any]] = []
    for name in sorted(expected_blocks):
        if name != f"block_{len(manifest):06d}.npz":
            raise EvidenceError(f"non-contiguous plan block set at {name}")
        expected_sha = expected_blocks[name]
        record = files[name]
        block_path = plan_root / "blocks" / name
        _require_canonical_file(block_path, f"plan block {name}")
        _same_path_record(record, block_path, f"plan_blocks.files.{name}")
        fingerprint = _stat_fingerprint(block_path)
        if record.get("sha256") != expected_sha:
            raise EvidenceError(f"freeze/checksums SHA disagreement for {name}")
        if int(record.get("bytes", -1)) != fingerprint["size_bytes"]:
            raise EvidenceError(f"freeze byte count mismatch for {name}")
        _require_frozen_fingerprint(record, fingerprint, f"plan_blocks.files.{name}")
        snapshots[name] = fingerprint
        manifest.append(
            {
                "name": name,
                "bytes": fingerprint["size_bytes"],
                "sha256": expected_sha,
                "formal_arrays_consumed": int(name[6:12]) <= ((FORMAL_STOP_STEP - 1) // block_steps),
                "pre_stat": fingerprint,
            }
        )
    return {
        "freeze_path": str(freeze_path),
        "freeze_sha256": _sha256_file(freeze_path),
        "freeze_schema_version": freeze.get("schema_version"),
        "freeze_plan_id": identities.get("plan_id"),
        "blocks": manifest,
    }, snapshots


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, mode)
    try:
        view = memoryview(payload)
        while view:
            wrote = os.write(fd, view)
            if wrote <= 0:
                raise EvidenceError(f"short write: {path}")
            view = view[wrote:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_publish_json(output: Path, value: Mapping[str, Any]) -> str:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evidence: {output}")
    parent = _require_canonical_directory(output.parent, "output parent")
    nonce = os.urandom(16).hex()
    temporary = parent / f".{output.name}.tmp.{os.getpid()}.{nonce}"
    payload = _canonical_bytes(value)
    try:
        _write_exclusive(temporary, payload)
        os.link(temporary, output, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if output.is_symlink() or not output.is_file() or output.read_bytes() != payload:
        raise EvidenceError("published output failed byte-exact readback")
    return _sha256_bytes(payload)


def _compile_helper(work_dir: Path) -> tuple[Path, dict[str, Any]]:
    source = work_dir / "fenwick_stack_distance.cc"
    binary = work_dir / "fenwick_stack_distance"
    _write_exclusive(source, CPP_SOURCE.encode("utf-8"))
    compiler = shutil.which("g++")
    if not compiler:
        raise EvidenceError("g++ is unavailable in the JCC image")
    compiler_path = Path(compiler).resolve(strict=True)
    version = subprocess.run(
        [str(compiler_path), "--version"], check=True, capture_output=True, text=True, timeout=30
    ).stdout.splitlines()[0]
    command = [str(compiler_path), "-O3", "-std=c++17", "-DNDEBUG", "-Wall", "-Wextra", str(source), "-o", str(binary)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if completed.returncode != 0:
        raise EvidenceError(f"native helper compilation failed rc={completed.returncode}: {completed.stderr[-2000:]}")
    if binary.is_symlink() or not binary.is_file():
        raise EvidenceError("native helper was not created as a regular file")
    return binary, {
        "algorithm": "exact_mattson_stack_distance_fenwick_v1",
        "embedded_source_sha256": _sha256_bytes(CPP_SOURCE.encode("utf-8")),
        "compiler_path": str(compiler_path),
        "compiler_version_line": version,
        "compile_flags": ["-O3", "-std=c++17", "-DNDEBUG", "-Wall", "-Wextra"],
        "binary_sha256": _sha256_file(binary),
    }


def _parse_worker_result(path: Path, capacities: Iterable[int]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"native worker result missing: {path.name}")
    scalar: dict[str, Any] = {}
    quantiles: dict[int, int] = {}
    hits: dict[int, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split("\t")
        if len(fields) == 2 and fields[0] in {"schema", "events", "cold", "reuses", "max_distance"}:
            if fields[0] in scalar:
                raise EvidenceError(f"duplicate native scalar {fields[0]}")
            scalar[fields[0]] = fields[1] if fields[0] == "schema" else int(fields[1])
        elif len(fields) == 3 and fields[0] == "quantile":
            quantiles[int(fields[1])] = int(fields[2])
        elif len(fields) == 3 and fields[0] == "capacity":
            hits[int(fields[1])] = int(fields[2])
        else:
            raise EvidenceError(f"malformed native worker line: {line!r}")
    if set(scalar) != {"schema", "events", "cold", "reuses", "max_distance"}:
        raise EvidenceError("native worker scalar key set drift")
    if scalar["schema"] != "fenwick_stack_distance_v1":
        raise EvidenceError("native worker schema drift")
    expected_capacities = set(capacities)
    if set(hits) != expected_capacities:
        raise EvidenceError("native worker capacity key set drift")
    expected_quantiles = set(QUANTILE_BASIS_POINTS) if scalar["reuses"] else set()
    if set(quantiles) != expected_quantiles:
        raise EvidenceError("native worker quantile key set drift")
    if scalar["reuses"]:
        ordered_quantiles = [quantiles[bp] for bp in QUANTILE_BASIS_POINTS]
        if ordered_quantiles != sorted(ordered_quantiles) or ordered_quantiles[-1] != scalar["max_distance"]:
            raise EvidenceError("native worker quantiles are not monotone/bound to max distance")
    elif scalar["max_distance"] != -1:
        raise EvidenceError("native worker empty-reuse max distance must be -1")
    if scalar["cold"] + scalar["reuses"] != scalar["events"]:
        raise EvidenceError("native worker event accounting drift")
    if any(value < 0 or value > scalar["reuses"] for value in hits.values()):
        raise EvidenceError("native worker hit accounting drift")
    ordered_hits = [hits[capacity] for capacity in sorted(expected_capacities)]
    if ordered_hits != sorted(ordered_hits):
        raise EvidenceError("native worker capacity hits are not monotone")
    if any(hits[capacity] != scalar["reuses"] for capacity in expected_capacities if capacity >= scalar["cold"]):
        raise EvidenceError("native worker failed the capacity>=unique infinite-hit identity")
    return {"scalar": scalar, "quantiles": quantiles, "capacity_hits": hits}


def _run_worker(
    binary: Path,
    sequence: Path,
    domain_size: int,
    capacities: tuple[int, ...],
    output: Path,
    timeout_s: int,
) -> dict[str, Any]:
    command = [str(binary), str(sequence), str(domain_size), ",".join(str(value) for value in capacities), str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    if completed.returncode != 0:
        raise EvidenceError(
            f"native cache worker failed for {sequence.name} rc={completed.returncode}: {completed.stderr[-2000:]}"
        )
    if completed.stdout or completed.stderr:
        raise EvidenceError(f"native cache worker emitted unexpected output for {sequence.name}")
    return _parse_worker_result(output, capacities)


def _quantile_name(basis_points: int) -> str:
    if basis_points == 0:
        return "p0"
    if basis_points == 10000:
        return "p100"
    whole, fractional = divmod(basis_points, 100)
    return f"p{whole}" if fractional == 0 else f"p{whole}.{fractional:02d}".rstrip("0")


def _set_summary(seen: np.ndarray, exposures: int) -> dict[str, Any]:
    unique = int(np.count_nonzero(seen))
    if unique > exposures:
        raise EvidenceError("unique object count exceeds exposures")
    packed = np.packbits(seen, bitorder="little")
    repeats = int(exposures - unique)
    return {
        "exposures": int(exposures),
        "unique_row_ids": unique,
        "repeat_exposures": repeats,
        "duplicate_fraction": _fraction(repeats, int(exposures)),
        "cold_infinite_cache_hits": repeats,
        "cold_infinite_cache_hit_fraction": _fraction(repeats, int(exposures)),
        "unique_set_bit_order": "gpic_local_row_id_ascending_packbits_little",
        "unique_set_sha256": hashlib.sha256(memoryview(packed)).hexdigest(),
    }


def _self_test(work_root: Path) -> None:
    work_root.mkdir(parents=True, exist_ok=True)
    canonical_root = work_root.resolve(strict=True)
    if not str(canonical_root).startswith("/local/"):
        raise EvidenceError("self-test work root must be under /local")
    work_dir = Path(tempfile.mkdtemp(prefix="cache_reuse_selftest.", dir=canonical_root))
    try:
        binary, _compile = _compile_helper(work_dir)
        sequence = work_dir / "sequence.u32"
        values = np.asarray([0, 1, 0, 2, 1, 0], dtype="<u4")
        _write_exclusive(sequence, values.tobytes(order="C"))
        result_path = work_dir / "result.tsv"
        result = _run_worker(binary, sequence, 3, (1, 2, 3), result_path, 60)
        scalar = result["scalar"]
        if scalar != {"schema": "fenwick_stack_distance_v1", "events": 6, "cold": 3, "reuses": 3, "max_distance": 2}:
            raise EvidenceError(f"self-test scalar mismatch: {scalar}")
        if result["capacity_hits"] != {1: 0, 2: 1, 3: 3}:
            raise EvidenceError(f"self-test capacity mismatch: {result['capacity_hits']}")
        if result["quantiles"][0] != 1 or result["quantiles"][5000] != 2 or result["quantiles"][10000] != 2:
            raise EvidenceError(f"self-test quantile mismatch: {result['quantiles']}")
        publish_dir = work_dir / "publish"
        publish_dir.mkdir()
        target = publish_dir / "result.json"
        _atomic_publish_json(target, {"self_test": "PASS"})
        try:
            _atomic_publish_json(target, {"self_test": "MUST_NOT_OVERWRITE"})
        except FileExistsError:
            pass
        else:
            raise EvidenceError("atomic no-clobber self-test did not reject overwrite")
        if _read_json_object(target, "self-test publish") != {"self_test": "PASS"}:
            raise EvidenceError("atomic no-clobber self-test changed first payload")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    print("W64_GPIC_CACHE_REUSE_SELF_TEST_PASS", flush=True)


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    script_path = _require_canonical_file(Path(__file__).resolve(strict=True), "analysis script")
    repo = script_path.parents[2]
    planner_source = _require_canonical_file(repo / "training/dataline_offline/planner_v1.py", "planner source")
    job_yaml = _require_canonical_file(args.job_yaml, "job YAML")
    freeze_path = _require_canonical_file(args.freeze, "freeze manifest")
    plan_root = _require_canonical_directory(args.plan_root, "w64 plan root")
    blocks_root = _require_canonical_directory(plan_root / "blocks", "w64 plan blocks")
    header_path = _require_canonical_file(plan_root / "header.json", "w64 plan header")
    checksums_path = _require_canonical_file(plan_root / "checksums.json", "w64 plan checksums")
    output = args.output
    if not output.is_absolute():
        raise EvidenceError("output must be absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evidence: {output}")

    source_hashes_before = {
        "script": _sha256_file(script_path),
        "planner_source": _sha256_file(planner_source),
        "job_yaml": _sha256_file(job_yaml),
        "freeze": _sha256_file(freeze_path),
        "plan_header": _sha256_file(header_path),
        "plan_checksums": _sha256_file(checksums_path),
    }
    header = _read_json_object(header_path, "plan header")
    checksums = _read_json_object(checksums_path, "plan checksums")
    freeze = _read_json_object(freeze_path, "freeze manifest")
    freeze_binding, block_snapshots = _freeze_binding(
        freeze_path,
        freeze,
        plan_root,
        header_path,
        checksums_path,
        source_hashes_before["plan_header"],
        source_hashes_before["plan_checksums"],
        checksums,
        int(header.get("block_steps", -1)),
    )
    if freeze_binding["freeze_sha256"] != source_hashes_before["freeze"]:
        raise EvidenceError("freeze SHA changed while reading binding")

    try:
        plan = OfflinePlan(plan_root, world_size=EXPECTED_WORLD_SIZE, per_rank_bs=EXPECTED_PER_RANK_BS, verify_checksums=True)
    except (PlannerInputError, OSError, ValueError) as exc:
        raise EvidenceError(f"OfflinePlan validation failed: {exc}") from exc
    config = plan.config
    pinned = {
        "plan_id": "stage256_offline_exact_v1_w64_prod",
        "world_size": EXPECTED_WORLD_SIZE,
        "per_rank_bs": EXPECTED_PER_RANK_BS,
        "global_batch_size": EXPECTED_GLOBAL_BATCH,
        "num_blocks": EXPECTED_PLAN_BLOCKS,
        "block_steps": EXPECTED_BLOCK_STEPS,
        "item_row_algorithm": "virtual_catalog_v1",
        "row_scheme": "virtual_catalog_v1",
        "row_id_scheme": "global",
        "catalog_global_rows": True,
    }
    actual = {
        "plan_id": header.get("plan_id"),
        "world_size": header.get("world_size"),
        "per_rank_bs": header.get("per_rank_bs"),
        "global_batch_size": header.get("global_batch_size"),
        "num_blocks": header.get("num_blocks"),
        "block_steps": header.get("block_steps"),
        "item_row_algorithm": header.get("item_row_mapping", {}).get("algorithm"),
        "row_scheme": header.get("row_scheme"),
        "row_id_scheme": header.get("row_id_scheme"),
        "catalog_global_rows": header.get("item_row_mapping", {}).get("catalog_global_rows"),
    }
    if actual != pinned:
        raise EvidenceError(f"w64 plan pin mismatch: {actual}")
    if FORMAL_STOP_STEP > config.total_steps or FORMAL_START_STEP != 0:
        raise EvidenceError("formal prefix is outside the plan")
    if FORMAL_STOP_STEP * EXPECTED_GLOBAL_BATCH != EXPECTED_FORMAL_EXPOSURES:
        raise EvidenceError("formal exposure constant drift")
    if RANKS_PER_NODE * NODE_COUNT != EXPECTED_WORLD_SIZE:
        raise EvidenceError("node/rank topology constant drift")
    try:
        domain_index = list(config.domains).index(TARGET_DOMAIN)
    except ValueError as exc:
        raise EvidenceError("gpic is absent from the frozen domain list") from exc
    domain_totals = [int(value) for value in header["N_total"]]
    gpic_lo = sum(domain_totals[:domain_index])
    gpic_size = domain_totals[domain_index]
    gpic_hi = gpic_lo + gpic_size
    if header["item_row_mapping"].get("virtual_contiguous_range") != [0, sum(domain_totals)]:
        raise EvidenceError("header virtual range disagrees with N_total")
    if gpic_size <= 0 or gpic_size > np.iinfo(np.uint32).max:
        raise EvidenceError("gpic virtual domain cannot be encoded as uint32 local IDs")

    work_root = args.work_root
    work_root.mkdir(parents=True, exist_ok=True)
    work_root = _require_canonical_directory(work_root.resolve(strict=True), "work root")
    if not str(work_root).startswith("/local/"):
        raise EvidenceError("large temporary sequences must remain under /local")
    work_dir = Path(tempfile.mkdtemp(prefix=f"w64_gpic_cache.{socket.gethostname()}.{os.getpid()}.", dir=work_root))
    try:
        binary, helper_identity = _compile_helper(work_dir)
        sequence_paths = [work_dir / f"node_{node}.u32" for node in range(NODE_COUNT)]
        sequence_handles = [path.open("xb", buffering=0) for path in sequence_paths]
        sequence_hashers = [hashlib.sha256() for _ in range(NODE_COUNT)]
        rank_exposures = np.zeros(EXPECTED_WORLD_SIZE, dtype=np.int64)
        node_exposures = np.zeros(NODE_COUNT, dtype=np.int64)
        domain_prefix_exposures = np.zeros(len(config.domains), dtype=np.int64)
        seen_rank = np.zeros((EXPECTED_WORLD_SIZE, gpic_size), dtype=np.bool_)
        seen_node = np.zeros((NODE_COUNT, gpic_size), dtype=np.bool_)
        used_last_block = (FORMAL_STOP_STEP - 1) // config.block_steps
        used_block_count = used_last_block + 1
        loaded_block_step_counts: list[int] = []

        try:
            for block_index in range(EXPECTED_PLAN_BLOCKS):
                if block_index > used_last_block:
                    plan._verify_block_checksum(block_index)  # noqa: SLF001 - evidence reader intentionally binds all blocks.
                    continue
                _bucket, items, domain_counts = plan._load_block(block_index)  # noqa: SLF001
                global_start = block_index * config.block_steps
                take = min(items.shape[0], FORMAL_STOP_STEP - global_start)
                if take <= 0:
                    raise EvidenceError(f"non-positive formal slice for block {block_index}")
                loaded_block_step_counts.append(int(take))
                formal_items = items[:take]
                formal_counts = domain_counts[:take]
                domain_prefix_exposures += formal_counts.sum(axis=0, dtype=np.int64)
                mask = (formal_items >= gpic_lo) & (formal_items < gpic_hi)
                mask_total = int(mask.sum(dtype=np.int64))
                declared_total = int(formal_counts[:, domain_index].sum(dtype=np.int64))
                if mask_total != declared_total:
                    raise EvidenceError(f"gpic virtual classification disagrees with domain_counts in block {block_index}")
                rank_exposures += mask.sum(axis=(0, 2), dtype=np.int64)

                for rank in range(EXPECTED_WORLD_SIZE):
                    rank_mask = mask[:, rank, :]
                    if np.any(rank_mask):
                        ids = formal_items[:, rank, :][rank_mask] - gpic_lo
                        seen_rank[rank, ids] = True
                for node in range(NODE_COUNT):
                    rank_lo = node * RANKS_PER_NODE
                    rank_hi = rank_lo + RANKS_PER_NODE
                    node_items = formal_items[:, rank_lo:rank_hi, :]
                    node_mask = mask[:, rank_lo:rank_hi, :]
                    ids64 = node_items[node_mask] - gpic_lo
                    if ids64.size:
                        if int(ids64.min()) < 0 or int(ids64.max()) >= gpic_size:
                            raise EvidenceError("gpic local ID conversion escaped domain")
                        ids = np.ascontiguousarray(ids64, dtype="<u4")
                        seen_node[node, ids] = True
                        ids.tofile(sequence_handles[node])
                        sequence_hashers[node].update(memoryview(ids).cast("B"))
                        node_exposures[node] += int(ids.size)
                rank_mask = ids = node_items = node_mask = ids64 = None
                del _bucket, items, domain_counts, formal_items, formal_counts, mask
                # OfflinePlan normally retains the last block for random batch()
                # calls.  This analysis is a forward-only stream, so release it
                # before the next NPZ is opened and keep the resident bound at
                # one materialized block.
                plan._cached_block_index = None  # noqa: SLF001
                plan._cached_bucket = None  # noqa: SLF001
                plan._cached_items = None  # noqa: SLF001
                plan._cached_domain_counts = None  # noqa: SLF001
        finally:
            for handle in sequence_handles:
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    handle.close()

        if len(plan._verified_blocks) != EXPECTED_PLAN_BLOCKS:  # noqa: SLF001
            raise EvidenceError("not all 168 plan blocks were SHA-verified")
        expected_tail_steps = FORMAL_STOP_STEP - used_last_block * config.block_steps
        if loaded_block_step_counts != [config.block_steps] * (used_block_count - 1) + [expected_tail_steps]:
            raise EvidenceError(f"formal block streaming shape drift: tail={loaded_block_step_counts[-3:]}")
        if int(domain_prefix_exposures.sum()) != EXPECTED_FORMAL_EXPOSURES:
            raise EvidenceError("formal domain exposure total mismatch")
        if int(rank_exposures.sum()) != int(domain_prefix_exposures[domain_index]):
            raise EvidenceError("rank GPIC exposure sum mismatch")
        if int(node_exposures.sum()) != int(rank_exposures.sum()):
            raise EvidenceError("node/rank GPIC exposure sum mismatch")
        for node in range(NODE_COUNT):
            rank_lo = node * RANKS_PER_NODE
            rank_hi = rank_lo + RANKS_PER_NODE
            if int(node_exposures[node]) != int(rank_exposures[rank_lo:rank_hi].sum()):
                raise EvidenceError(f"node {node} exposure mismatch")

        sequence_records = []
        for node, path in enumerate(sequence_paths):
            if path.is_symlink() or not path.is_file():
                raise EvidenceError(f"node {node} sequence is missing")
            expected_bytes = int(node_exposures[node]) * 4
            if path.stat().st_size != expected_bytes:
                raise EvidenceError(f"node {node} sequence byte count mismatch")
            sequence_records.append(
                {
                    "node": node,
                    "events": int(node_exposures[node]),
                    "bytes": expected_bytes,
                    "sha256": sequence_hashers[node].hexdigest(),
                }
            )

        rank_records = []
        for rank in range(EXPECTED_WORLD_SIZE):
            summary = _set_summary(seen_rank[rank], int(rank_exposures[rank]))
            summary.update(
                {
                    "rank": rank,
                    "node": rank // RANKS_PER_NODE,
                    "node_local_rank": rank % RANKS_PER_NODE,
                    "cache_scope_for_cold_infinite_fields": "rank_standalone",
                }
            )
            rank_records.append(summary)
        node_records_base = []
        for node in range(NODE_COUNT):
            rank_lo = node * RANKS_PER_NODE
            rank_hi = rank_lo + RANKS_PER_NODE
            rank_union = np.logical_or.reduce(seen_rank[rank_lo:rank_hi], axis=0)
            if not np.array_equal(rank_union, seen_node[node]):
                raise EvidenceError(f"node {node} unique set disagrees with its eight-rank union")
            del rank_union
            summary = _set_summary(seen_node[node], int(node_exposures[node]))
            rank_unique_sum = sum(int(rank_records[rank]["unique_row_ids"]) for rank in range(rank_lo, rank_hi))
            cross_rank_extra_hits = rank_unique_sum - int(summary["unique_row_ids"])
            if cross_rank_extra_hits < 0:
                raise EvidenceError(f"node {node} cross-rank reuse accounting became negative")
            summary.update(
                {
                    "node": node,
                    "global_ranks": list(range(rank_lo, rank_lo + RANKS_PER_NODE)),
                    "cache_scope_for_cold_infinite_fields": "node_shared_across_8_ranks",
                    "sum_of_eight_rank_standalone_unique_row_ids": rank_unique_sum,
                    "additional_infinite_hits_from_node_sharing_vs_rank_standalone": cross_rank_extra_hits,
                }
            )
            node_records_base.append(summary)
        global_seen = np.logical_or.reduce(seen_node, axis=0)
        global_unique = int(np.count_nonzero(global_seen))
        global_set_sha = hashlib.sha256(memoryview(np.packbits(global_seen, bitorder="little"))).hexdigest()
        del seen_rank, seen_node, global_seen
        gc.collect()

        capacity_map: dict[tuple[int, int], int] = {}
        for average_bytes in args.average_object_bytes:
            for gib in CACHE_GIB_VALUES:
                capacity_map[(average_bytes, gib)] = (gib * (1 << 30)) // average_bytes
        capacities = tuple(sorted(set(capacity_map.values())))
        if not capacities or capacities[0] <= 0:
            raise EvidenceError("equal-size sensitivity produced an invalid object capacity")

        worker_outputs = [work_dir / f"node_{node}.result.tsv" for node in range(NODE_COUNT)]
        worker_results: list[dict[str, Any] | None] = [None] * NODE_COUNT
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_worker,
                    binary,
                    sequence_paths[node],
                    gpic_size,
                    capacities,
                    worker_outputs[node],
                    args.worker_timeout_s,
                ): node
                for node in range(NODE_COUNT)
            }
            for future in concurrent.futures.as_completed(futures):
                node = futures[future]
                worker_results[node] = future.result()
        if any(result is None for result in worker_results):
            raise EvidenceError("one or more node cache workers did not return")

        node_records = []
        aggregate_hits: dict[tuple[int, int], int] = {(average, gib): 0 for average, gib in capacity_map}
        for node in range(NODE_COUNT):
            result = worker_results[node]
            assert result is not None
            scalar = result["scalar"]
            base = node_records_base[node]
            if scalar["events"] != base["exposures"] or scalar["cold"] != base["unique_row_ids"]:
                raise EvidenceError(f"node {node} native sequence/unique accounting mismatch")
            if scalar["reuses"] != base["repeat_exposures"]:
                raise EvidenceError(f"node {node} native reuse accounting mismatch")
            base["equal_size_infinite_cache_resident_sensitivity"] = [
                {
                    "assumed_equal_resident_bytes_per_object": average_bytes,
                    "resident_bytes_for_all_node_unique_objects": int(base["unique_row_ids"]) * average_bytes,
                    "resident_gib_for_all_node_unique_objects_decimal": format(
                        int(base["unique_row_ids"]) * average_bytes / (1 << 30), ".6f"
                    ),
                }
                for average_bytes in args.average_object_bytes
            ]
            sensitivity = []
            for average_bytes in args.average_object_bytes:
                rows = []
                for gib in CACHE_GIB_VALUES:
                    capacity = capacity_map[(average_bytes, gib)]
                    hits = int(result["capacity_hits"][capacity])
                    misses = int(base["exposures"] - hits)
                    aggregate_hits[(average_bytes, gib)] += hits
                    rows.append(
                        {
                            "cache_gib": gib,
                            "object_capacity_floor": capacity,
                            "unused_capacity_bytes": gib * (1 << 30) - capacity * average_bytes,
                            "hits": hits,
                            "misses": misses,
                            "hit_fraction": _fraction(hits, int(base["exposures"])),
                        }
                    )
                sensitivity.append({"assumed_equal_resident_bytes_per_object": average_bytes, "caches": rows})
            base["reuse_distance"] = {
                "unit": "distinct_GPIC_objects_more_recent_than_previous_access",
                "population": "repeat_exposures_only",
                "quantile_method": "nearest_rank_ceil(q*n)-1",
                "reuse_events": int(scalar["reuses"]),
                "max": int(scalar["max_distance"]),
                "quantiles": (
                    {_quantile_name(bp): int(result["quantiles"][bp]) for bp in QUANTILE_BASIS_POINTS}
                    if scalar["reuses"]
                    else {}
                ),
            }
            base["equal_size_lru_sensitivity"] = sensitivity
            node_records.append(base)

        aggregate_sensitivity = []
        total_gpic_exposures = int(node_exposures.sum())
        for average_bytes in args.average_object_bytes:
            rows = []
            for gib in CACHE_GIB_VALUES:
                capacity = capacity_map[(average_bytes, gib)]
                hits = aggregate_hits[(average_bytes, gib)]
                rows.append(
                    {
                        "cache_gib_per_node": gib,
                        "object_capacity_floor_per_node": capacity,
                        "node_sum_hits": hits,
                        "node_sum_misses": total_gpic_exposures - hits,
                        "node_sum_hit_fraction": _fraction(hits, total_gpic_exposures),
                    }
                )
            aggregate_sensitivity.append({"assumed_equal_resident_bytes_per_object": average_bytes, "caches": rows})

        for record, path in zip(sequence_records, sequence_paths):
            after_sha = _sha256_file(path)
            if after_sha != record["sha256"]:
                raise EvidenceError(f"temporary node sequence changed during native analysis: node {record['node']}")
            record["post_analysis_sha256"] = after_sha

        for name, before in block_snapshots.items():
            after = _stat_fingerprint(blocks_root / name)
            if after != before:
                raise EvidenceError(f"plan block stat changed during analysis: {name}")
        freeze_binding_after, block_snapshots_after = _freeze_binding(
            freeze_path,
            freeze,
            plan_root,
            header_path,
            checksums_path,
            source_hashes_before["plan_header"],
            source_hashes_before["plan_checksums"],
            checksums,
            config.block_steps,
        )
        if freeze_binding_after != freeze_binding or block_snapshots_after != block_snapshots:
            raise EvidenceError("frozen plan tree binding changed during analysis")
        source_hashes_after = {
            "script": _sha256_file(script_path),
            "planner_source": _sha256_file(planner_source),
            "job_yaml": _sha256_file(job_yaml),
            "freeze": _sha256_file(freeze_path),
            "plan_header": _sha256_file(header_path),
            "plan_checksums": _sha256_file(checksums_path),
        }
        if source_hashes_after != source_hashes_before:
            raise EvidenceError("source/freeze/header/checksums changed during analysis")

        result_document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "decision_scope": "diagnostic_only_not_a_FORMAL_promotion_authority",
            "scope": {
                "formal_steps": {"start_inclusive": FORMAL_START_STEP, "stop_exclusive": FORMAL_STOP_STEP, "count": FORMAL_STOP_STEP},
                "formal_global_exposures": EXPECTED_FORMAL_EXPOSURES,
                "world_size": EXPECTED_WORLD_SIZE,
                "per_rank_batch_size": EXPECTED_PER_RANK_BS,
                "global_batch_size": EXPECTED_GLOBAL_BATCH,
                "node_count": NODE_COUNT,
                "ranks_per_node": RANKS_PER_NODE,
                "target_domain": TARGET_DOMAIN,
                "gpic_virtual_row_range": [gpic_lo, gpic_hi],
                "gpic_virtual_row_count": gpic_size,
                "classification_contract": "cumulative header.N_total under virtual_catalog_v1",
                "cache_key_model": "one hypothetical cache object per GPIC virtual row_id",
                "logical_node_access_order": "step_major_then_global_rank_ascending_within_node_then_slot_ascending_filtered_to_gpic",
                "cache_initial_state": "cold",
                "cache_sharing": "one hypothetical GPIC-only cache shared by 8 ranks on each node",
                "image_payload_reads": 0,
            },
            "limitations": {
                "infinite_cache_order_independence": "exact: exposures-unique_row_ids",
                "capacity_limited_order_model": "The plan has no wall-clock interleaving for eight concurrent ranks; LRU uses the declared deterministic logical order. Loader prefetch across steps, thread completion order, and real cache-admission order may differ.",
                "equal_size_assumption": "Every GPIC object is assigned the selected assumed resident byte size; real object sizes and cache overhead were not read and may change hits.",
                "physical_aliasing_assumption": "Catalog/path payloads are intentionally not read; any physical aliasing between distinct virtual row_ids is not deduplicated.",
                "not_measured": ["image_bytes", "decoded_tensor_bytes", "physical_path_aliasing", "filesystem_cache", "actual_rank_interleaving", "cache_metadata_overhead"],
            },
            "inputs": {
                "plan_root": str(plan_root),
                "header": {"path": str(header_path), "sha256": source_hashes_before["plan_header"], "bytes": header_path.stat().st_size},
                "checksums": {"path": str(checksums_path), "sha256": source_hashes_before["plan_checksums"], "bytes": checksums_path.stat().st_size},
                "blocks_manifest_sha256": header["hashes"]["blocks_manifest_sha256"],
                "all_plan_blocks_sha256_verified": True,
                "all_plan_block_count": EXPECTED_PLAN_BLOCKS,
                "formal_array_blocks_streamed": used_block_count,
                "formal_array_block_index_range_inclusive": [0, used_last_block],
                "formal_last_block_steps_consumed": loaded_block_step_counts[-1],
                "nonformal_blocks_sha_only_verified": EXPECTED_PLAN_BLOCKS - used_block_count,
                "maximum_materialized_plan_blocks_resident_at_once": 1,
                "freeze_binding": freeze_binding,
                "source_files": {
                    "script": {"path": str(script_path), "sha256": source_hashes_before["script"]},
                    "planner": {"path": str(planner_source), "sha256": source_hashes_before["planner_source"]},
                    "job_yaml": {"path": str(job_yaml), "sha256": source_hashes_before["job_yaml"]},
                },
            },
            "runtime": {
                "hostname": socket.gethostname(),
                "python_executable": str(Path(sys.executable).resolve(strict=True)),
                "python_version": sys.version,
                "numpy_version": np.__version__,
                "declared_cpu_workers": args.workers,
                "worker_timeout_s": args.worker_timeout_s,
                "os_cpu_count": os.cpu_count(),
            },
            "native_stack_distance_helper": helper_identity,
            "formal_prefix_domain_exposures": {
                domain: int(domain_prefix_exposures[index]) for index, domain in enumerate(config.domains)
            },
            "gpic_results": {
                "total_exposures": total_gpic_exposures,
                "global_unique_row_ids": global_unique,
                "global_unique_set_bit_order": "gpic_local_row_id_ascending_packbits_little",
                "global_unique_set_sha256": global_set_sha,
                "global_unique_equal_size_resident_sensitivity": [
                    {
                        "assumed_equal_resident_bytes_per_object": average_bytes,
                        "resident_bytes_for_all_global_unique_objects": global_unique * average_bytes,
                        "resident_gib_for_all_global_unique_objects_decimal": format(
                            global_unique * average_bytes / (1 << 30), ".6f"
                        ),
                    }
                    for average_bytes in args.average_object_bytes
                ],
                "per_rank": rank_records,
                "per_node": node_records,
                "aggregate_node_local_equal_size_lru_sensitivity": aggregate_sensitivity,
            },
            "temporary_sequence_attestation": {
                "encoding": "little_endian_uint32_gpic_local_row_id",
                "records": sequence_records,
                "retained_after_success": False,
            },
        }
        output_sha = _atomic_publish_json(output, result_document)
        return result_document, output_sha
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plan-root", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--job-yaml", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worker-timeout-s", type=int, default=7200)
    parser.add_argument(
        "--average-object-bytes",
        type=_parse_average_bytes,
        default=DEFAULT_AVERAGE_OBJECT_BYTES,
        help="comma-separated equal-resident-size assumptions",
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > NODE_COUNT:
        parser.error(f"--workers must be in [1,{NODE_COUNT}]")
    if args.worker_timeout_s < 60:
        parser.error("--worker-timeout-s must be at least 60")
    if not args.self_test:
        missing = [name for name in ("plan_root", "freeze", "job_yaml", "output") if getattr(args, name) is None]
        if missing:
            parser.error(f"missing required analysis arguments: {missing}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            _self_test(args.work_root)
            return 0
        result, output_sha = analyze(args)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": str(args.output),
                    "output_sha256": output_sha,
                    "gpic_exposures": result["gpic_results"]["total_exposures"],
                    "global_unique_gpic": result["gpic_results"]["global_unique_row_ids"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except (EvidenceError, FileExistsError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
