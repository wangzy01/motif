#!/usr/bin/env python3
"""Run the curated exact-two/exact-three zero-argument policy tests."""

from __future__ import annotations

import inspect
import os
import runpy
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = ROOT / "tests"
EXPECTED = (
    (
        "test_anime_source_bound_native_mismatch_v1.py",
        ("test_source_bound_binding_and_negative_cases",),
    ),
    (
        "test_weighted_anime_native_upsample_v1.py",
        (
            "test_exact_weighted_binding",
            "test_partial_binding_fails_closed",
        ),
    ),
    (
        "test_stage512a_control_registry_profiles_v1.py",
        (
            "test_control_identity_fail_closed",
            "test_exact_control_bindings_and_selector",
            "test_legacy_selector_closures_unchanged",
            "test_sealed_closure_and_mutation_rejection",
        ),
    ),
)
EXPECTED_EXECUTIONS = 7


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def stat_identity(path: Path) -> tuple[int, int, int, int, int, int, int]:
    value = path.lstat()
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def main() -> None:
    root_stat = TESTS_ROOT.lstat()
    require(
        stat.S_ISDIR(root_stat.st_mode) and not TESTS_ROOT.is_symlink(),
        "curated tests root is not a real directory",
    )
    expected_files = tuple(sorted(module for module, _names in EXPECTED))
    actual_files = tuple(sorted(path.name for path in TESTS_ROOT.iterdir()))
    require(
        actual_files == expected_files,
        "curated tests directory exact-three closure drift",
    )
    root_identity = stat_identity(TESTS_ROOT)
    module_identities: dict[str, tuple[int, int, int, int, int, int, int]] = {}

    selected: list[tuple[str, object]] = []
    for module_name, expected_names in EXPECTED:
        path = TESTS_ROOT / module_name
        path_stat = path.lstat()
        require(
            stat.S_ISREG(path_stat.st_mode)
            and not path.is_symlink()
            and path_stat.st_nlink == 1
            and stat.S_IMODE(path_stat.st_mode) & 0o111 == 0,
            "curated policy test object/mode/link drift",
        )
        module_identities[module_name] = stat_identity(path)
        namespace = runpy.run_path(
            str(path),
            run_name=f"_mot_exact_policy_{path.stem}",
        )
        discovered = {
            name: value
            for name, value in namespace.items()
            if name.startswith("test_")
        }
        require(
            tuple(sorted(discovered)) == tuple(sorted(expected_names)),
            "curated policy test callable closure drift",
        )
        for name in sorted(expected_names):
            value = discovered[name]
            require(
                inspect.isfunction(value)
                and not inspect.iscoroutinefunction(value)
                and not inspect.isgeneratorfunction(value)
                and Path(value.__code__.co_filename).resolve() == path
                and len(inspect.signature(value).parameters) == 0,
                "curated policy test must be a local synchronous zero-arg function",
            )
            selected.append((f"{module_name}:{name}", value))

    require(
        len(selected) == EXPECTED_EXECUTIONS,
        "curated policy test execution count drift",
    )
    result_path_raw = os.environ.get("MOT_EXACT_POLICY_RESULT", "")
    result_path = Path(result_path_raw)
    require(
        result_path_raw != ""
        and result_path.is_absolute()
        and not os.path.lexists(result_path),
        "exact policy result path is not fresh and absolute",
    )
    executed = 0
    for _identity, function in selected:
        function()
        executed += 1
    require(
        executed == EXPECTED_EXECUTIONS,
        "curated policy test execution count drift",
    )
    require(
        tuple(sorted(path.name for path in TESTS_ROOT.iterdir()))
        == expected_files
        and stat_identity(TESTS_ROOT) == root_identity
        and all(
            stat_identity(TESTS_ROOT / module_name) == expected_identity
            for module_name, expected_identity in module_identities.items()
        ),
        "curated policy tests changed their exact source closure",
    )
    result_raw = f"EXACT_POLICY_TESTS_PASS count={executed}\n".encode("ascii")
    require(
        not os.path.lexists(result_path),
        "exact policy result path appeared during test execution",
    )
    descriptor = os.open(
        result_path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(result_raw)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, "short exact policy result write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(result_raw.decode("ascii"), end="")


if __name__ == "__main__":
    main()
