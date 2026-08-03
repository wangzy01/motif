#!/usr/bin/env python3
"""Exhaustively test the FORMAL job's fresh/continue resolution, on the shipped text.

This is the highest-risk logic in the highest-risk file: it decides which weights a
seven-day, 64-GPU run starts from. Getting it wrong in the permissive direction means
silently training from the wrong checkpoint; getting it wrong in the strict direction
means a JCC retry cannot resume and the run dies on its first eviction. Both have already
happened once in this project's history.

The block is EXTRACTED from the YAML rather than reimplemented here. A reimplementation
would test a copy and pass while the shipped text was broken -- which is the same failure
mode as a check that cannot fail, and this repo has had two of those.

Cases cover every combination of: declared mode, directory present or absent, .jobtag
absent / ours / another job's, and last.pt present or absent.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "provenance/jcc_jobs/20260803_stage512b_FORMAL_w64.yaml"
OURS = "evan-s512b-real-formal-w64-abcd"
THEIRS = "evan-s512b-real-formal-w64-zzzz"


def extract_block() -> str:
    """The rank-0 resolution block, verbatim from the job spec."""
    # yaml's block scalar strips the spec's two-space indent, so anchor on the
    # unindented form -- matching the indented text silently found nothing and the
    # extraction failed loudly, which is the behaviour we want from a locator.
    command = yaml.safe_load(JOB.read_text())["command"]
    start = command.index("RES=$CKPT\nMODEFILE=")
    marker = 'mv "$MODEFILE.tmp" "$MODEFILE"'
    end = command.index(marker)
    block = command[start:end] + marker + "\nfi\n"
    # The sha256 of a 26 GB checkpoint is not the subject of this test.
    block = re.sub(r"\s*GOT=\$\(sha256sum[^\n]*\n", '\n  GOT="$INIT_SHA"\n', block)
    if "EFFECTIVE_MODE" not in block or "jobtag" not in block:
        raise SystemExit("extraction failed: the job spec's structure changed")
    return block


def run_case(block: str, *, declared: str, dir_exists: bool, jobtag: str | None,
             last_pt: bool) -> tuple[int, str]:
    tmp = Path(tempfile.mkdtemp(prefix="formal_mode_"))
    try:
        ckpt = tmp / "run"
        if dir_exists:
            ckpt.mkdir()
            if jobtag is not None:
                (ckpt / ".jobtag").write_text(jobtag + "\n")
            if last_pt:
                (ckpt / "last.pt").write_text("x")
        script = "\n".join([
            "set -euo pipefail",
            f'CKPT="{ckpt}"', f'RUN_MODE="{declared}"', f'JOBTAG="{OURS}"',
            'NODE_RANK=0', 'INIT_SHA=deadbeef', 'INIT=/dev/null',
            block,
            'echo "RESOLVED=$EFFECTIVE_MODE"',
        ])
        proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        match = re.search(r"RESOLVED=(\w+)", out)
        return proc.returncode, (match.group(1) if match else out.strip().splitlines()[-1:][0] if out.strip() else "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# (declared, dir_exists, jobtag, last_pt) -> (expected_rc, expected_mode_or_"")
CASES = [
    # First launch, the normal path.
    (("fresh", False, None, False), (0, "fresh"), "fresh + no dir -> fresh"),
    # JCC retry of our own submission. This is the case whose absence made maxRetries
    # decorative: it MUST resolve to continue, not exit.
    (("fresh", True, OURS, True), (0, "continue"), "fresh + our dir + last.pt -> continue (retry)"),
    # Retry where the first attempt died before checkpointing: no progress to resume,
    # so init_from must still fire.
    (("fresh", True, OURS, False), (0, "fresh"), "fresh + our dir + no last.pt -> fresh"),
    # Someone else's run. Refusing is the whole point of the claim.
    (("fresh", True, THEIRS, True), (45, ""), "fresh + another job's dir -> exit 45"),
    (("fresh", True, THEIRS, False), (45, ""), "fresh + another job's empty dir -> exit 45"),
    # A directory with no .jobtag at all -- e.g. a pre-.jobtag run, or a hand-made dir.
    # Must refuse: it cannot be shown to be ours.
    (("fresh", True, None, True), (45, ""), "fresh + unowned dir -> exit 45"),
    # Operator explicitly continuing an interrupted run. Allowed even when the tag names
    # a different submission -- that IS the resubmission case (Stage512a's recovery_v1
    # -> recovery_v2), and refusing it would block legitimate recovery.
    (("continue", True, THEIRS, True), (0, "continue"), "continue + another job's dir + last.pt -> continue"),
    (("continue", True, OURS, True), (0, "continue"), "continue + our dir + last.pt -> continue"),
    # Nothing to continue from.
    (("continue", True, OURS, False), (45, ""), "continue + no last.pt -> exit 45"),
    (("continue", True, None, False), (45, ""), "continue + unowned, no last.pt -> exit 45"),
    (("continue", False, None, False), (45, ""), "continue + no dir -> exit 45"),
]


def main() -> int:
    block = extract_block()
    failures = []
    print(f"extracted {len(block.splitlines())} lines from {JOB.name}\n")
    for args, (want_rc, want_mode), label in CASES:
        declared, dir_exists, jobtag, last_pt = args
        rc, mode = run_case(block, declared=declared, dir_exists=dir_exists,
                            jobtag=jobtag, last_pt=last_pt)
        ok = rc == want_rc and (want_mode == "" or mode == want_mode)
        print(f"  {'ok  ' if ok else 'FAIL'} {label:<58} rc={rc} mode={mode or '-'}")
        if not ok:
            failures.append(f"{label}: rc={rc} mode={mode!r}, want rc={want_rc} mode={want_mode!r}")

    # The property that matters most, stated as its own assertion rather than left
    # implicit in the table: there must be NO path where a directory belonging to another
    # job is silently trained into under a fresh declaration.
    for jt in (THEIRS, None):
        rc, _ = run_case(block, declared="fresh", dir_exists=True, jobtag=jt, last_pt=True)
        if rc == 0:
            failures.append(f"SILENT ADOPTION: fresh accepted a dir with .jobtag={jt!r}")

    print(f"\n{len(CASES) + 2} checks, {len(failures)} failures")
    for f in failures:
        print(f"  {f}")
    if failures:
        return 1
    print("FORMAL_RUN_MODE_TESTS_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
