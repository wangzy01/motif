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
# The RUN_ID the operator supplies, stable across JCC retries because it lives in the job
# spec. RETRY_JOBTAG is what the pod name gives a retry -- JCC renames retry pods, so the
# pod-derived JOBTAG is NOT the run identity. Getting that wrong made maxRetries
# decorative for the second time: the first attempt exited 47 on capacity and the retry
# exited 45 against the directory it had itself created.
OURS = "s512b-real-20260804-v1"
THEIRS = "s512b-real-20260804-v9"
RETRY_JOBTAG = "evan-s512b-real-formal-w64-v2-5dlx-retry-7mrd"


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
             last_pt: bool, receipt: str | None = "ours",
             jobtag_env: str | None = None) -> tuple[int, str]:
    """receipt: "ours" = a Stage512b FORMAL_RUN.json, "other" = wrong schema, None = absent.

    Any directory a FORMAL attempt has actually run in HAS this file -- rank 0 writes it
    every attempt -- so "ours" is the realistic default. None models the Stage512a
    directory and any other checkpoint dir that was never a Stage512b formal run.
    """
    tmp = Path(tempfile.mkdtemp(prefix="formal_mode_"))
    try:
        ckpt = tmp / "run"
        if dir_exists:
            ckpt.mkdir()
            if jobtag is not None:
                (ckpt / ".jobtag").write_text(jobtag + "\n")
            if last_pt:
                (ckpt / "last.pt").write_text("x")
            if receipt == "ours":
                (ckpt / "FORMAL_RUN.json").write_text(
                    '{"schema": "stage512b_real_formal_run_v1", "attempts": []}')
            elif receipt == "other":
                (ckpt / "FORMAL_RUN.json").write_text(
                    '{"schema": "stage512a_weighted_w64_recovery_formal_run_20260724v2"}')
        script = "\n".join([
            "set -euo pipefail",
            f'CKPT="{ckpt}"', f'RUN_MODE="{declared}"',
            # JOBTAG is pod-derived and CHANGES on a retry; RUN_ID comes from the job spec
            # and does not. Setting them to the same value would make the retry cases
            # vacuous -- the bug being fixed is precisely that they diverge.
            f'JOBTAG="{jobtag_env or OURS}"',
            'NODE_RANK=0', 'INIT_SHA=deadbeef', 'INIT=/dev/null',
            f'RUN_ID="{OURS}"',
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

# The Stage512a-adoption case, which needs the receipt dimension. Pointing CKPT_DIR at
# the frozen Stage512a directory offers a last.pt that is architecturally identical, so
# nothing in train.py would object: it would resume at step 125000 and train only the
# last 125,000 steps of the plan, while the shell wrote .jobtag and arrival markers into
# a frozen artifact directory.
RECEIPT_CASES = [
    (("continue", True, None, True, None), (45, ""),
     "continue + Stage512a-like dir (no receipt) -> exit 45"),
    (("continue", True, OURS, True, None), (45, ""),
     "continue + our dir but no receipt -> exit 45"),
    (("continue", True, None, True, "other"), (45, ""),
     "continue + another stage's receipt -> exit 45"),
    (("fresh", False, None, False, None), (0, "fresh"),
     "fresh + no dir, no receipt needed -> fresh"),
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

    for args, (want_rc, want_mode), label in RECEIPT_CASES:
        declared, dir_exists, jobtag, last_pt, receipt = args
        rc, mode = run_case(block, declared=declared, dir_exists=dir_exists,
                            jobtag=jobtag, last_pt=last_pt, receipt=receipt)
        ok = rc == want_rc and (want_mode == "" or mode == want_mode)
        print(f"  {'ok  ' if ok else 'FAIL'} {label:<58} rc={rc} mode={mode or '-'}")
        if not ok:
            failures.append(f"{label}: rc={rc} mode={mode!r}, want rc={want_rc} mode={want_mode!r}")

    # A JCC retry: the pod name (and therefore JOBTAG) differs from the first attempt's,
    # while RUN_ID is unchanged. Ownership must key on RUN_ID, so this must resolve rather
    # than exit -- it is the case that broke the real run on 2026-08-04.
    for label, last_pt, want in (
        ("retry (new JOBTAG) + no checkpoint -> fresh", False, (0, "fresh")),
        ("retry (new JOBTAG) + checkpoint -> continue", True, (0, "continue")),
    ):
        rc, mode = run_case(block, declared="fresh", dir_exists=True, jobtag=OURS,
                            last_pt=last_pt, receipt="ours",
                            jobtag_env=RETRY_JOBTAG)
        ok = rc == want[0] and mode == want[1]
        print(f"  {'ok  ' if ok else 'FAIL'} {label:<58} rc={rc} mode={mode or '-'}")
        if not ok:
            failures.append(f"{label}: rc={rc} mode={mode!r}, want {want}")

    # Guard against a regression to JOBTAG-keyed ownership: with the pod-derived tag
    # written into .jobtag, a retry whose JOBTAG differs must be REFUSED -- which is what
    # broke the real run. If someone reverts the keying, this case flips and the suite
    # fails instead of the seven-day job.
    rc, _ = run_case(block, declared="fresh", dir_exists=True, jobtag=RETRY_JOBTAG,
                     last_pt=False, receipt="ours", jobtag_env=RETRY_JOBTAG)
    ok = rc == 45
    print(f"  {'ok  ' if ok else 'FAIL'} {'a dir owned by a pod-tag is not ours -> exit 45':<58} rc={rc}")
    if not ok:
        failures.append(f"ownership appears keyed on JOBTAG again: rc={rc}")

    # The two properties that matter most, asserted directly rather than left implicit in
    # the tables: a fresh declaration must never adopt another job's directory, and a
    # continue must never adopt a directory that was not a Stage512b formal run.
    for jt in (THEIRS, None):
        rc, _ = run_case(block, declared="fresh", dir_exists=True, jobtag=jt, last_pt=True)
        if rc == 0:
            failures.append(f"SILENT ADOPTION: fresh accepted a dir with .jobtag={jt!r}")
    for receipt in (None, "other"):
        rc, _ = run_case(block, declared="continue", dir_exists=True, jobtag=None,
                         last_pt=True, receipt=receipt)
        if rc == 0:
            failures.append(f"SILENT ADOPTION: continue accepted a dir with receipt={receipt!r}")

    # ---- the pod-generation rendezvous race (2026-08-04 10:28 UTC, cost the run) ----
    # JOBTAG comes from the pod hostname and a RESTART reuses it; only a JCC retry gets a
    # new one. So rdzv.$JOBTAG.addr survives a restart holding the dead generation's rank-0
    # IP, and rank 0 republishes it only after sha256-ing 26 GB -- a ~40s window in which
    # every restarting peer reads a dead address and wedges in TCPStore retry for 900s.
    # Seven nodes did exactly that. The fix is to remove the file BEFORE the slow work.
    src = yaml.safe_load(JOB.read_text())["command"]
    i_def = src.find("RDZV=$RES/rdzv.$JOBTAG.addr")
    i_rm = src.find('rm -f "$RDZV"')
    i_sha = src.find('sha256sum "$INIT"')
    i_pub = src.find('> "$RDZV.tmp"')
    checks = [
        (src.count("RDZV=$RES/rdzv.$JOBTAG.addr") == 1,
         "rendezvous path is defined exactly once (two spellings drift apart)"),
        (i_rm != -1, "rank 0 removes the stale rendezvous"),
        (i_def != -1 and i_def < i_rm,
         "the path is defined before it is removed"),
        (i_rm != -1 and i_sha != -1 and i_rm < i_sha,
         "removal precedes the 26 GB hash -- otherwise the window it closes is still open"),
        (i_rm != -1 and i_pub != -1 and i_rm < i_pub,
         "removal precedes republication"),
    ]
    for ok, label in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
        if not ok:
            failures.append(label)

    # Behavioural, on the shipped text: a stale rendezvous must not survive rank 0's pass.
    tmp = Path(tempfile.mkdtemp(prefix="formal_rdzv_"))
    try:
        ckpt = tmp / "run"
        ckpt.mkdir()
        (ckpt / ".jobtag").write_text(OURS + "\n")
        (ckpt / "FORMAL_RUN.json").write_text(
            '{"schema": "stage512b_real_formal_run_v1", "attempts": []}')
        stale = ckpt / f"rdzv.{OURS}.addr"
        stale.write_text(f"10.64.180.10 {int(1785839000)} {OURS}\n")
        script = "\n".join([
            "set -euo pipefail",
            f'CKPT="{ckpt}"', 'RUN_MODE="fresh"', f'JOBTAG="{OURS}"',
            'NODE_RANK=0', 'INIT_SHA=deadbeef', 'INIT=/dev/null', f'RUN_ID="{OURS}"',
            block,
            'echo "RESOLVED=$EFFECTIVE_MODE"',
        ])
        proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        gone = not stale.exists()
        ok = proc.returncode == 0 and gone
        print(f"  {'ok  ' if ok else 'FAIL'} a stale rendezvous is gone after rank 0's pass"
              f"  [rc={proc.returncode} removed={gone}]")
        if not ok:
            failures.append(
                f"stale rendezvous survived (rc={proc.returncode}, exists={stale.exists()}); "
                "restarting peers would read a dead master address")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(CASES) + len(RECEIPT_CASES) + 7 + len(checks) + 1} checks, "
          f"{len(failures)} failures")
    for f in failures:
        print(f"  {f}")
    if failures:
        return 1
    print("FORMAL_RUN_MODE_TESTS_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
