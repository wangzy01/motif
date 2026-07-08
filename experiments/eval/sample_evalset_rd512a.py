#!/usr/bin/env python3
"""rd512a (redesign stage512a G1/G2 visual gate) sampler entry.

Same proven path as sample_evalset_rd256L.py: monkey-patch the shared
experiments.eval.sample_evalset module (run spec + ckpt resolution) and delegate
to its main(). sample_evalset.py itself is NOT modified (a sibling training line
has uncommitted edits there — read-only reuse only).

Run key  : rd512a
Config   : configs/stage512a_redesign_v1.yaml
Arch      : causal + multi-layer-fusion (verbatim with the 512a training job)
Ckpt      : <--snapshot-dir>/<$RD512A_GATE_CKPT>   (e.g. .../redesign_512a_gates/gate_g1.pt)

RD512A_GATE_CKPT is REQUIRED and has no default: the gate files are named
gate_g1.pt / gate_g2.pt (not the eval_<key>.pt convention that build_runs assumes),
so the two gate jobs disambiguate purely by this env var. Missing -> fail fast.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.eval import sample_evalset as se  # noqa: E402

GATE_CKPT = os.environ.get("RD512A_GATE_CKPT", "").strip()
if not GATE_CKPT:
    sys.exit(
        "FATAL: RD512A_GATE_CKPT env var is unset/empty. Set it to the frozen gate "
        "filename (e.g. gate_g1.pt or gate_g2.pt); refusing to guess a default."
    )

# Arch override copied verbatim from the 512a training job (causal + MLF), same
# shape as the rd256L wrapper.
se._RUN_SPEC["rd512a"] = (
    "redesign_512a_gates",  # subdir under CKPT_ROOT; only used on the non-snapshot path
    "redesign512a causal+MLF gate",
    "configs/stage512a_redesign_v1.yaml",
    [
        "model.text_adapter.enabled=true",
        "model.text_adapter.bidirectional=false",
        "model.text_adapter.multi_layer_fusion.enabled=true",
    ],
)

# build_runs() (called by se.main via its module global) resolves the snapshot
# path as <snapshot_dir>/eval_<key>.pt. Our gate files are gate_g{1,2}.pt, so
# rewrite ONLY the rd512a ckpt to <snapshot_dir>/<RD512A_GATE_CKPT>. All other
# behaviour is inherited unchanged.
_orig_build_runs = se.build_runs


def _build_runs(ckpt_root, snapshot_dir=None):
    runs = _orig_build_runs(ckpt_root, snapshot_dir)
    if snapshot_dir and "rd512a" in runs:
        runs["rd512a"]["ckpt"] = os.path.join(snapshot_dir, GATE_CKPT)
    return runs


se.build_runs = _build_runs

if __name__ == "__main__":
    se.main()
