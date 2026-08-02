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

# Prefix match so a short sha is accepted; anchored so a short sha cannot match in
# the middle of a longer one.
case "$head" in
  "$expected"*) ;;
  *)
    echo "FATAL: HEAD is not the expected commit -- the tree is at a different revision"
    echo "       than the one that was reviewed.  Re-check out, or fix the job spec."
    exit 46
    ;;
esac

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

# Uses local remote-tracking refs only -- no network, which matters because compute
# nodes have no git credentials.  Advisory: an unpushed commit still runs, it just
# means the artifact's provenance points at something only this pod can resolve.
if [ -z "$(git branch -r --contains HEAD 2>/dev/null)" ]; then
  echo "tree            : WARNING -- HEAD is not contained in any remote-tracking"
  echo "                  branch.  The artifact will reference a commit that is not"
  echo "                  pushed; push before treating its provenance as reviewable."
else
  echo "pushed          : yes ($(git branch -r --contains HEAD | tr -d ' ' | paste -sd, -))"
fi

echo "=== preflight: OK ==="
