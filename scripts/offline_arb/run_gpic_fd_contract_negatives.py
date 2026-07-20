#!/usr/bin/env python3
"""Small deterministic GPIC reader contract matrix for the fd semantic gate.

This is intentionally a KB-scale /local fixture.  It does not scan the catalog,
caption sidecar, or production GPIC tree.  Both the plain and instrumented
reader implementations must satisfy the same normal, short-read, in-place
mutation, and namespace-replacement expectations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

import training.dataline_offline.offline_dataset as offline_module
from training.dataline_offline.offline_dataset import (
    OfflineExactDataset,
    OfflineLoaderError,
    _PlannedItem,
)


SCHEMA = "offline_arb_gpic_fd_negative_contract_v1"
RUN_SCHEMA = "offline_arb_gpic_fd_semantic_run_v1"
CLAIM_SCHEMA = "offline_arb_gpic_fd_semantic_negative_claim_v1"
MODES = ("strict_path", "fd_stable")
GPIC_FIXTURE_WORKERS = 4
HEX64 = __import__("re").compile(r"[0-9a-f]{64}").fullmatch
SAFE_TOKEN = __import__("re").compile(r"[A-Za-z0-9_.-]+").fullmatch


class ContractError(RuntimeError):
    pass


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def regular_bytes(path: Path) -> bytes:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or metadata.st_size <= 0
    ):
        raise ContractError(f"expected non-empty regular file: {path}")
    raw = path.read_bytes()
    after = path.lstat()
    before_identity = (
        metadata.st_mode,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    after_identity = (
        after.st_mode,
        after.st_dev,
        after.st_ino,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if len(raw) != metadata.st_size or before_identity != after_identity:
        raise ContractError(f"file size changed while reading: {path}")
    return raw


def strict_json(path: Path) -> dict[str, Any]:
    raw = regular_bytes(path)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"top-level JSON is not an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    parent_meta = path.parent.lstat()
    if not stat.S_ISDIR(parent_meta.st_mode) or path.parent.is_symlink():
        raise ContractError(f"output parent is not a real directory: {path.parent}")
    raw = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("xb", buffering=0) as handle:
        view = memoryview(raw)
        while view:
            written = handle.write(view)
            if not written:
                raise OSError("short write")
            view = view[written:]
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as exc:
        temporary.unlink(missing_ok=True)
        raise ContractError(f"refusing to replace output: {path}") from exc
    try:
        temporary.unlink()
    except OSError as exc:
        raise ContractError(f"cannot remove output temporary: {temporary}") from exc
    fsync_directory(path.parent)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_claim(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    parent_meta = path.parent.lstat()
    if not stat.S_ISDIR(parent_meta.st_mode) or path.parent.is_symlink():
        raise ContractError(f"claim parent is not a real directory: {path.parent}")
    raw = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ContractError(f"negative ownership claim already exists: {path}") from exc
    with os.fdopen(descriptor, "wb", buffering=0, closefd=True) as handle:
        view = memoryview(raw)
        while view:
            written = handle.write(view)
            if not written:
                raise OSError("short claim write")
            view = view[written:]
        os.fsync(handle.fileno())
    fsync_directory(path.parent)
    return {"path": str(path), "bytes": len(raw), "sha256": sha256(raw)}


def new_dataset(mode: str) -> OfflineExactDataset:
    dataset = object.__new__(OfflineExactDataset)
    dataset.rank = 0
    dataset._gpic_workers = GPIC_FIXTURE_WORKERS
    dataset._gpic_source_consistency_mode = mode
    dataset._pipeline_active = False
    dataset._caption_sidecar = None
    dataset._threads = []
    return dataset


def pair(path: Path, *, step: int = 0, slot: int = 0, row_id: int = 1) -> tuple[Any, dict[str, str]]:
    return (
        _PlannedItem(step, slot, row_id, "gpic", 2, 0),
        {"image_path": str(path)},
    )


def with_module_open(replacement: Callable[..., Any], callback: Callable[[], Any]) -> Any:
    sentinel = object()
    previous = getattr(offline_module, "open", sentinel)
    setattr(offline_module, "open", replacement)
    try:
        return callback()
    finally:
        if previous is sentinel:
            delattr(offline_module, "open")
        else:
            setattr(offline_module, "open", previous)


def with_stat_counts(callback: Callable[[], Any]) -> tuple[Any, dict[str, int]]:
    real_fstat = offline_module.os.fstat
    real_stat = offline_module.os.stat
    counts = {"fstat_calls": 0, "path_stat_calls": 0}

    def counted_fstat(*args: Any, **kwargs: Any) -> Any:
        counts["fstat_calls"] += 1
        return real_fstat(*args, **kwargs)

    def counted_stat(*args: Any, **kwargs: Any) -> Any:
        counts["path_stat_calls"] += 1
        return real_stat(*args, **kwargs)

    offline_module.os.fstat = counted_fstat
    offline_module.os.stat = counted_stat
    try:
        return callback(), counts
    finally:
        offline_module.os.fstat = real_fstat
        offline_module.os.stat = real_stat


def expect_loader_error(
    label: str,
    callback: Callable[[], Any],
    *,
    category: str,
    required_substring: str,
) -> dict[str, Any]:
    try:
        callback()
    except OfflineLoaderError as exc:
        message = str(exc)
        if required_substring not in message:
            raise ContractError(
                f"{label} rejection category drift: missing {required_substring!r} in {message!r}"
            ) from exc
        encoded = message.encode("utf-8")
        return {
            "case": label,
            "outcome": "EXPECTED_REJECTION",
            "error_type": type(exc).__name__,
            "error_category": category,
            "required_substring": required_substring,
            "error_sha256": sha256(encoded),
        }
    except BaseException as exc:
        raise ContractError(
            f"{label} raised wrong exception {type(exc).__name__}: {exc}"
        ) from exc
    raise ContractError(f"{label} unexpectedly succeeded")


class ReaderProxy:
    def __init__(self, handle: Any):
        self.handle = handle

    def __enter__(self) -> "ReaderProxy":
        self.handle.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self.handle.__exit__(*args)

    def fileno(self) -> int:
        return self.handle.fileno()


class EarlyEofReader(ReaderProxy):
    def __init__(self, handle: Any):
        super().__init__(handle)
        self.first = True

    def read(self, _size: int) -> bytes:
        if self.first:
            self.first = False
            return self.handle.read(1)
        return b""


class MutatingReader(ReaderProxy):
    def __init__(self, handle: Any, path: Path):
        super().__init__(handle)
        self.path = path
        self.first = True

    def read(self, size: int) -> bytes:
        block = self.handle.read(size)
        if self.first:
            self.first = False
            metadata = self.path.stat()
            os.utime(
                self.path,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000),
            )
        return block


class NamespaceSwapReader(ReaderProxy):
    def __init__(self, handle: Any, visible: Path, replacement: Path):
        super().__init__(handle)
        self.visible = visible
        self.replacement = replacement
        self.first = True

    def read(self, size: int) -> bytes:
        block = self.handle.read(size)
        if self.first:
            self.first = False
            os.replace(self.replacement, self.visible)
        return block


def normal_case(workspace: Path, mode: str, telemetry: bool) -> dict[str, Any]:
    name = f"normal.{mode}.{'instrumented' if telemetry else 'plain'}"
    source_a = workspace / f"{name}.a.img"
    source_b = workspace / f"{name}.b.img"
    raw_a = b"gpic-normal-a\x00\xff"
    raw_b = ("GPIC/fd/语义/🌙".encode("utf-8") + b"\x00") * 11
    source_a.write_bytes(raw_a)
    source_b.write_bytes(raw_b)
    dataset = new_dataset(mode)
    expected = {(0, 0): raw_a, (0, 1): raw_b}
    pairs = [
        pair(source_a, step=0, slot=0, row_id=10),
        pair(source_b, step=0, slot=1, row_id=11),
    ]
    expected_workers = min(dataset._gpic_workers, len(pairs))
    result, counts = with_stat_counts(
        lambda: dataset._gpic_payloads_in_memory(pairs, telemetry=telemetry)
    )
    if telemetry:
        payloads, detail = result
    else:
        payloads, detail = result, None
    expected_counts = {
        "fstat_calls": 4,
        "path_stat_calls": 4 if mode == "strict_path" else 0,
    }
    if payloads != expected or counts != expected_counts:
        raise ContractError(f"normal bytes/counters drift case={name}")
    if telemetry:
        if (
            detail.get("gpic_source_consistency_mode") != mode
            or detail.get("files") != 2
            or detail.get("bytes") != len(raw_a) + len(raw_b)
            or detail.get("fstat_calls") != expected_counts["fstat_calls"]
            or detail.get("path_stat_calls") != expected_counts["path_stat_calls"]
            or detail.get("workers") != expected_workers
        ):
            raise ContractError(f"normal telemetry detail drift case={name}")
        if mode == "fd_stable" and (
            detail.get("thread_sum_s", {}).get("path_stat") != 0.0
            or detail.get("operation_max_s", {}).get("path_stat") != 0.0
        ):
            raise ContractError("fd_stable normal case recorded path-stat time")
    return {
        "case": name,
        "outcome": "PASS",
        "mode": mode,
        "telemetry": telemetry,
        "files": 2,
        "workers": expected_workers if telemetry else None,
        "payload_sha256": {
            "0:0": sha256(payloads[(0, 0)]),
            "0:1": sha256(payloads[(0, 1)]),
        },
        **expected_counts,
        "path_stat_timing_zero": (
            mode == "fd_stable" and telemetry
        ),
    }


def short_case(workspace: Path, mode: str, telemetry: bool) -> dict[str, Any]:
    name = f"short_read.{mode}.{'instrumented' if telemetry else 'plain'}"
    source = workspace / f"{name}.img"
    source.write_bytes(b"short-read-source-must-not-truncate")
    real_open = open

    def short_open(path: str, *args: Any, **kwargs: Any) -> EarlyEofReader:
        return EarlyEofReader(real_open(path, *args, **kwargs))

    dataset = new_dataset(mode)
    result = expect_loader_error(
        name,
        lambda: with_module_open(
            short_open,
            lambda: dataset._gpic_payloads_in_memory(
                [pair(source, step=1, row_id=20)], telemetry=telemetry
            ),
        ),
        category="short_read",
        required_substring="short gpic source read",
    )
    result.update({"mode": mode, "telemetry": telemetry})
    return result


def in_place_case(workspace: Path, mode: str, telemetry: bool) -> dict[str, Any]:
    name = f"in_place.{mode}.{'instrumented' if telemetry else 'plain'}"
    source = workspace / f"{name}.img"
    source.write_bytes((b"in-place-source\x00" * 19) + b"tail")
    real_open = open

    def mutating_open(path: str, *args: Any, **kwargs: Any) -> MutatingReader:
        return MutatingReader(real_open(path, *args, **kwargs), Path(path))

    dataset = new_dataset(mode)
    result = expect_loader_error(
        name,
        lambda: with_module_open(
            mutating_open,
            lambda: dataset._gpic_payloads_in_memory(
                [pair(source, step=2, row_id=30)], telemetry=telemetry
            ),
        ),
        category="in_place_identity_change",
        required_substring="gpic source changed while reading",
    )
    result.update({"mode": mode, "telemetry": telemetry})
    return result


def namespace_case(workspace: Path, mode: str, telemetry: bool) -> dict[str, Any]:
    name = f"namespace.{mode}.{'instrumented' if telemetry else 'plain'}"
    old_raw = b"namespace-opened-old-inode\x00"
    new_raw = b"namespace-visible-new-inode\xff"
    old_target = workspace / f"{name}.old.img"
    new_target = workspace / f"{name}.new.img"
    visible = workspace / f"{name}.visible.img"
    replacement = workspace / f"{name}.replacement.img"
    old_target.write_bytes(old_raw)
    new_target.write_bytes(new_raw)
    visible.symlink_to(old_target)
    replacement.symlink_to(new_target)
    real_open = open

    def swapping_open(path: str, *args: Any, **kwargs: Any) -> NamespaceSwapReader:
        return NamespaceSwapReader(
            real_open(path, *args, **kwargs),
            visible,
            replacement,
        )

    dataset = new_dataset(mode)
    pairs = [pair(visible, step=3, row_id=40)]
    expected_workers = min(dataset._gpic_workers, len(pairs))

    def read() -> Any:
        return with_module_open(
            swapping_open,
            lambda: dataset._gpic_payloads_in_memory(pairs, telemetry=telemetry),
        )

    if mode == "strict_path":
        result = expect_loader_error(
            name,
            read,
            category="namespace_replacement",
            required_substring="gpic source changed while reading",
        )
        result.update({"mode": mode, "telemetry": telemetry})
    else:
        value, counts = with_stat_counts(read)
        if telemetry:
            payloads, detail = value
        else:
            payloads, detail = value, None
        expected_counts = {"fstat_calls": 2, "path_stat_calls": 0}
        if payloads.get((3, 0)) != old_raw or counts != expected_counts:
            raise ContractError("fd_stable namespace case did not retain opened old inode")
        if telemetry and (
            detail.get("gpic_source_consistency_mode") != "fd_stable"
            or detail.get("fstat_calls") != 2
            or detail.get("path_stat_calls") != 0
            or detail.get("workers") != expected_workers
            or detail.get("thread_sum_s", {}).get("path_stat") != 0.0
            or detail.get("operation_max_s", {}).get("path_stat") != 0.0
        ):
            raise ContractError("fd_stable namespace telemetry drift")
        result = {
            "case": name,
            "outcome": "PASS_OLD_OPENED_INODE",
            "mode": mode,
            "telemetry": telemetry,
            "payload_sha256": sha256(payloads[(3, 0)]),
            "expected_old_sha256": sha256(old_raw),
            "workers": expected_workers if telemetry else None,
            **expected_counts,
        }
    if visible.read_bytes() != new_raw:
        raise ContractError(f"namespace replacement was not visible after case={name}")
    result["visible_after_sha256"] = sha256(new_raw)
    return result


def invalid_mode_cases() -> list[dict[str, Any]]:
    result = []
    for index, value in enumerate(("", "STRICT_PATH", "fd", True, None)):
        evidence = expect_loader_error(
            f"invalid_mode.{index}",
            lambda value=value: offline_module._validated_gpic_source_consistency_mode(value),
            category="invalid_mode",
            required_substring="gpic_source_consistency_mode must be exactly one of",
        )
        evidence["input_repr"] = repr(value)
        result.append(evidence)
    return result


def run(args: argparse.Namespace) -> None:
    run = strict_json(args.run_json)
    if (
        run.get("schema_version") != RUN_SCHEMA
        or run.get("jobtag") != args.jobtag
        or SAFE_TOKEN(str(args.jobtag or "")) is None
        or SAFE_TOKEN(str(args.host or "")) is None
        or HEX64(str(run.get("run_nonce", ""))) is None
        or run.get("worker0_host") != args.host
        or args.worker_index != 0
        or args.output != args.run_json.parent / "negative_contract.json"
    ):
        raise ContractError("run identity mismatch")
    claim = exclusive_claim(
        args.run_json.parent / "claim.negative.json",
        {
            "schema_version": CLAIM_SCHEMA,
            "jobtag": run["jobtag"],
            "run_nonce": run["run_nonce"],
            "worker_index": args.worker_index,
            "host": args.host,
            "pid": os.getpid(),
            "created_time_ns": __import__("time").time_ns(),
        },
    )
    args.local_root.mkdir(parents=True, exist_ok=False)
    workspace = Path(tempfile.mkdtemp(prefix="fixture-", dir=args.local_root))
    cases: list[dict[str, Any]] = []
    try:
        for mode in MODES:
            for telemetry in (False, True):
                cases.append(normal_case(workspace, mode, telemetry))
                cases.append(short_case(workspace, mode, telemetry))
                cases.append(in_place_case(workspace, mode, telemetry))
                cases.append(namespace_case(workspace, mode, telemetry))
        cases.extend(invalid_mode_cases())
    finally:
        # The fixture is no longer needed after all byte/content assertions have
        # been materialized in the result.  It is /local-only and deterministic.
        __import__("shutil").rmtree(args.local_root, ignore_errors=True)
    summary = {
        "normal_success": sum(
            case["case"].startswith("normal.") and case["outcome"] == "PASS"
            for case in cases
        ),
        "short_read_rejected": sum(
            case["case"].startswith("short_read.")
            and case["outcome"] == "EXPECTED_REJECTION"
            for case in cases
        ),
        "in_place_rejected": sum(
            case["case"].startswith("in_place.")
            and case["outcome"] == "EXPECTED_REJECTION"
            for case in cases
        ),
        "namespace_strict_rejected": sum(
            case["case"].startswith("namespace.strict_path.")
            and case["outcome"] == "EXPECTED_REJECTION"
            for case in cases
        ),
        "namespace_fd_old_inode": sum(
            case["case"].startswith("namespace.fd_stable.")
            and case["outcome"] == "PASS_OLD_OPENED_INODE"
            for case in cases
        ),
        "invalid_mode_rejected": sum(
            case["case"].startswith("invalid_mode.")
            and case["outcome"] == "EXPECTED_REJECTION"
            for case in cases
        ),
    }
    expected = {
        "normal_success": 4,
        "short_read_rejected": 4,
        "in_place_rejected": 4,
        "namespace_strict_rejected": 2,
        "namespace_fd_old_inode": 2,
        "invalid_mode_rejected": 5,
    }
    if summary != expected or len(cases) != sum(expected.values()):
        raise ContractError(f"negative matrix summary drift: {summary}")
    result = {
        "status": "PASS",
        "schema_version": SCHEMA,
        "jobtag": run["jobtag"],
        "run_nonce": run["run_nonce"],
        "worker0_host": args.host,
        "claim": claim,
        "case_count": len(cases),
        "summary": summary,
        "cases": cases,
    }
    atomic_json(args.output, result)
    print(
        f"GPIC_FD_NEGATIVE_CONTRACT_PASS cases={len(cases)} output={args.output}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--jobtag", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except ContractError as exc:
        print(f"FATAL GPIC fd negative contract: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
