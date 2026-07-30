#!/usr/bin/env python3
"""Strict paired Stage512a sampler built on the proven two-final sampler."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


SOURCE = Path(
    "/fsx/evan/mot_code/data_analysis/eval_releases/"
    "stage256_0718_s75000_vs_previous_final_20260721b/sample_two_final_eval.py"
)
EXPECTED_SOURCE_SHA256 = "9ed25c2ca0897aaac79bb2359d6cfd853a2b39bb3c1e1fb19b924c4d8a81f594"
CURRENT_CHECKPOINT = (
    "/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/"
    "stage512a_20260724_weighted_w64_recovery_v2/last.pt"
)
PREVIOUS_CHECKPOINT = (
    "/fsx/evan/checkpoints/mot_pilot_v1/"
    "redesign_512a_multiar_anchor_global1536_20260705/last.pt"
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
    raise RuntimeError("proven sampler SHA drift")
spec = importlib.util.spec_from_file_location("stage512a_pair_sampler_base", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import proven sampler")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

architecture = {
    "model.patch_size": 1,
    "model.text_max_length": 512,
    "model.sandwich_norm": True,
    "model.noise_condition": True,
    "model.long_skip.enabled": False,
    "model.text_adapter.enabled": True,
    "model.text_adapter.bidirectional": False,
    "model.text_adapter.multi_layer_fusion.enabled": True,
    "model.text_adapter.multi_layer_fusion.num_layers": 12,
    "model.und_lora.enabled": False,
    "model.abs_sincos.enabled": False,
}
module.RUNS = {
    "weighted_final_s125000": {
        "name": "Stage512a weighted offline W64 final step125000",
        "checkpoint": CURRENT_CHECKPOINT,
        "checkpoint_step": 125000,
        "config": "configs/stage512a_20260722_weighted_w64_v1.yaml",
        "config_sha256": "611635fc13b04c6d62c20875eff541736afbe6e35cc17c1f81caafb7f823a380",
        "overrides": [],
        "architecture_expect": architecture,
        "exposure_per_step": 2048,
    },
    "streaming_final_s166667": {
        "name": "Stage512a previous streaming final step166667",
        "checkpoint": PREVIOUS_CHECKPOINT,
        "checkpoint_step": 166667,
        "config": "configs/stage512a_redesign_v1.yaml",
        "config_sha256": "a76756240d1d84a87679acd58d5f44d5925c5d6cb058bd8f53f3ccaee259a83c",
        "overrides": [
            "model.text_adapter.enabled=true",
            "model.text_adapter.bidirectional=false",
            "model.text_adapter.multi_layer_fusion.enabled=true",
        ],
        "architecture_expect": architecture,
        "exposure_per_step": 1536,
    },
}

# Seven 512px-scale buckets, exactly 2x the frozen Stage256 matrix.
module.MULTIAR_SIZES = (
    (512, 512),
    (544, 448),
    (448, 544),
    (608, 416),
    (416, 608),
    (672, 384),
    (384, 672),
)

# The proven sampler's anime suite is frozen at 256x256. Redirect only that
# suite to 512x512 while retaining all prompts, seeds, batches, steps and CFG.
original_render_suite = module.render_suite


def render_suite(
    model,
    rows,
    suite,
    run_key,
    width,
    height,
    seeds,
    batch_size,
    steps,
    cfg,
    out,
):
    if suite == "anime":
        if (width, height) != (256, 256):
            raise RuntimeError("unexpected anime redirect source dimensions")
        width = height = 512
    return original_render_suite(
        model, rows, suite, run_key, width, height, seeds, batch_size, steps, cfg, out
    )


module.render_suite = render_suite

# Correct the one legacy anime manifest filename and its top-level dimensions.
# Canonical sample_manifest image records already carry the redirected 512 size.
original_atomic_json = module.atomic_json


def atomic_json(path: Path, value: dict) -> None:
    if path.name == "anime_256x256.json":
        path = path.with_name("anime_512x512.json")
        value = dict(value)
        value["width"] = 512
        value["height"] = 512
        value["label"] = "anime_evalset_512"
    original_atomic_json(path, value)


module.atomic_json = atomic_json
module.__file__ = str(Path(__file__).resolve())
module.main()
