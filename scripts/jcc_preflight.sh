#!/usr/bin/env bash
# Refuse to run unless the live working tree IS the reviewed commit, byte for byte.
#
# JCC jobs execute against /fsx/evan/mot_code_s512b_impl directly -- there is no
# checkout step and no image bake, so whatever the tree happens to contain at the
# moment the pod starts is what runs.  That has already cost one run: weighted plan
# v2 died in its own preflight because an in-flight signature change was sitting in
# the tree, and a job that had *not* run tests would simply have produced an
# artifact built from half-edited code.
#
# Every job spec before 2026-08-02 only printed `git rev-parse HEAD` and
# `git status --porcelain | head`.  Printing is not gating: the output scrolls past
# in a log nobody reads until the artifact is already published.
#
# Usage:  scripts/jcc_preflight.sh <expected-commit-sha> [repo-dir]
# Exit:   46 on identity failure (42 rank / 43 env / 44 input sha / 45 create-once
#         are the other job-level codes; this is a new one so it is distinguishable
#         in `jcc job list` output).

set -euo pipefail

expected="${1:?usage: jcc_preflight.sh <expected-commit-sha> [repo-dir]}"
repo="${2:-$PWD}"

cd "$repo"

echo "=== preflight: tree identity ==="
echo "repo            : $repo"

head="$(git rev-parse HEAD)"
echo "expected commit : $expected"
echo "HEAD            : $head"

# Full 40 hex digits, compared for equality. Prefix matching was a mistake: it accepted
# "9" against 9a105ed... and returned 0, so a one-character typo -- or a value that got
# truncated somewhere in the submission path -- silently disabled the gate while printing
# a reassuring line. Verified before the fix: `jcc_preflight.sh 9 .` exited 0.
case "$expected" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]\
[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]\
[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]\
[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;;
  *)
    echo "FATAL: expected commit must be a full 40-character lowercase hex sha, got"
    echo "       '$expected' (${#expected} chars). A prefix is not accepted: it would"
    echo "       let a truncated value pass the gate while looking correct."
    exit 46
    ;;
esac
if [ "$head" != "$expected" ]; then
  echo "FATAL: HEAD is not the expected commit -- the tree is at a different revision"
  echo "       than the one that was reviewed.  Re-check out, or fix the job spec."
  exit 46
fi

# --porcelain includes untracked files on purpose.  An untracked .py next to a
# module it shadows changes behaviour just as effectively as an edit does.
dirty="$(git status --porcelain)"
if [ -n "$dirty" ]; then
  echo "FATAL: working tree is dirty; JCC reads the live tree, so this job would not"
  echo "       be running the reviewed code.  Commit or stash first."
  printf '%s\n' "$dirty"
  exit 46
fi
echo "tree            : clean"

# Whether HEAD is pushed is NOT checkable here and the gate does not pretend otherwise.
# Compute nodes have no git credentials, so `git ls-remote` cannot run; and this clone
# is --single-branch, so `git branch -r --contains HEAD` is empty for every branch but
# the cloned one and would fire a warning on every single job. Report the branch and
# leave the push check to the submitting side.
echo "branch          : $(git rev-parse --abbrev-ref HEAD) (push state not checkable from a compute node)"

echo "=== preflight: OK ==="
