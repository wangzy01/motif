#!/usr/bin/env python3
"""Two-GPU hard gate for CUDA VAE RNG and exact FSDP2/DCP continuation.

This is an integration test, not a training entry point.  A JCC job invokes the
phases in separate ``torchrun`` processes so ``save-reference`` -> process exit
-> ``resume-compare`` covers a real restart boundary.

Hard-gated evidence:

* the real FLUX.2 posterior uses a generator on the exact local CUDA device;
* rank-derived posterior epsilon is unique and exactly replayable per rank;
* a v2 checkpoint contains one valid RNG record per global rank;
* the next update after restart is exact for batch, latent, CFG, sigma,
  diffusion noise, loss, full gradient, raw parameters, EMA, optimizer, and
  stage-local scheduler/LR state;
* schema/world/missing/duplicate-stream failures and rank-1-only opaque-state
  corruption fail coherently on every rank before state/RNG mutation.

The model under FSDP2 is intentionally tiny so this gate stays below four GPUs.
The latent path is not fake: update phases encode pixels through the real
FLUX.2 VAE and sample its CUDA posterior with the production stream contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import stat
import subprocess
import sys
import unittest.mock
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from networks.vae import Flux2VAE  # noqa: E402
from training.checkpoint import (  # noqa: E402
    build_scheduler,
    load_checkpoint,
    save_checkpoint,
    should_write_final_checkpoint,
)
from training.formal_boot import (  # noqa: E402
    EXPECTED_RUN_LINEAGE_ENV,
    FORMAL_RUN_LINEAGE_KEY,
    normalize_expected_run_lineage_sha256,
    resolve_checkpoint_boot_branch,
)
from training.rng_state import (  # noqa: E402
    FULL_CHECKPOINT_SCHEMA,
    RNG_STATE_SCHEMA,
    assert_live_rng_matches_synchronized,
    capture_local_rng_state,
    restore_local_rng_state_synchronized,
)


BASE_SEED = 12_345
GLOBAL_SEED = 91_337
RANK_STRIDE = 1_000_003
STREAM_OFFSETS = {"dropout": 11, "diffusion": 23, "vae": 37}
PRODUCTION_FILES = (
    "evaluation/runners/fixed_sigma_eval.py",
    "networks/model_mot.py",
    "networks/vae.py",
    "training/rng_state.py",
    "training/checkpoint.py",
    "training/formal_boot.py",
    "train.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def git_text(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT
        ).strip()
    except BaseException as exc:  # provenance should never hide the test result
        return f"<unavailable:{type(exc).__name__}:{exc}>"


def provenance() -> dict[str, Any]:
    return {
        "git_head": git_text("rev-parse", "HEAD"),
        "git_status_porcelain": git_text("status", "--porcelain"),
        "production_sha256": {
            name: sha256_file(REPO / name) for name in PRODUCTION_FILES
        },
        "test_sha256": sha256_file(Path(__file__).resolve()),
        "expected_run_lineage_sha256": required_lineage(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "world_size": dist.get_world_size(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }


def init_distributed() -> tuple[int, int, int, torch.device]:
    require(torch.cuda.is_available(), "CUDA is required")
    require("RANK" in os.environ and "LOCAL_RANK" in os.environ, "launch with torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    require(world == 2, f"formal gate requires exactly 2 ranks, got {world}")
    require(torch.cuda.device_count() >= world, "one node with two visible GPUs is required")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return rank, world, local_rank, device


def seed_globals() -> None:
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)


def required_lineage() -> str:
    value = normalize_expected_run_lineage_sha256(
        os.environ.get(EXPECTED_RUN_LINEAGE_ENV)
    )
    require(value is not None, f"{EXPECTED_RUN_LINEAGE_ENV} is required by the GPU gate")
    return value


def make_config() -> SimpleNamespace:
    return SimpleNamespace(
        model={
            "rank_independent_seed": False,
            "patch_size": 1,
            "noise_condition": True,
            "sandwich_norm": True,
        },
        trainer={
            "seed": BASE_SEED,
            "checkpoint_keep_every": 1,
            # Four slots let the gate prove that later full milestones never
            # roll earlier ones away, then exercise the fail-closed fifth save.
            # Unlike compact snapshots, full retention may not delete anchors.
            "checkpoint_keep_max": 4,
            "checkpoint_keep_format": "full",
        },
        scheduler={"num_warmup_steps": 4, "decay": "constant"},
    )


class TinyCore(nn.Module):
    """Small trainable core; all parameters participate in every update."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.in_proj = nn.Linear(16, width)
        self.mid = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, 16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.tanh(self.in_proj(value))
        value = torch.nn.functional.silu(self.mid(value))
        return self.out_proj(value)


class GateModel:
    """Production checkpoint/RNG interface around a tiny FSDP2 core."""

    def __init__(self, core: nn.Module, rank: int, device: torch.device):
        self.model = core
        self.rank = int(rank)
        self.device = torch.device(device)
        seed_base = BASE_SEED + self.rank * RANK_STRIDE
        self.dropout_generator = torch.Generator(device="cpu").manual_seed(
            seed_base + STREAM_OFFSETS["dropout"]
        )
        self.diffusion_generator = torch.Generator(device="cpu").manual_seed(
            seed_base + STREAM_OFFSETS["diffusion"]
        )
        self.vae_generator = torch.Generator(device=self.device).manual_seed(
            seed_base + STREAM_OFFSETS["vae"]
        )
        self._ema: dict[str, torch.Tensor] | None = None
        self.ema_decay = 0.9

    def training_rng_generators(self) -> dict[str, torch.Generator]:
        return {
            "dropout": self.dropout_generator,
            "diffusion": self.diffusion_generator,
            "vae": self.vae_generator,
        }

    def training_rng_contract(self) -> dict[str, Any]:
        return {
            "base_seed": BASE_SEED,
            "rank_independent_seed": False,
            "rank_stride": RANK_STRIDE,
            "stream_offsets": dict(STREAM_OFFSETS),
        }

    def training_rng_device(self) -> torch.device:
        return self.device

    def init_ema(self) -> None:
        self._ema = {
            name: param.detach().clone().float()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update_ema(self) -> None:
        require(self._ema is not None, "EMA was not initialized")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._ema[name].lerp_(param.detach().float(), 1.0 - self.ema_decay)

    @torch.no_grad()
    def ema_state_dict(self) -> dict[str, torch.Tensor]:
        require(self._ema is not None, "EMA was not initialized")
        out: dict[str, torch.Tensor] = {}
        for name, value in self._ema.items():
            full = value.full_tensor() if hasattr(value, "full_tensor") else value
            out[name] = full.detach().cpu().clone()
        return out

    @torch.no_grad()
    def load_ema_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        require(self._ema is not None, "EMA was not initialized")
        from torch.distributed.tensor import DTensor, distribute_tensor

        require(set(state) == set(self._ema), "EMA key mismatch")
        for name, current in self._ema.items():
            source = state[name]
            if isinstance(current, DTensor):
                current.copy_(
                    distribute_tensor(
                        source.to(device=current.device, dtype=current.dtype),
                        current.device_mesh,
                        current.placements,
                    )
                )
            else:
                current.copy_(source.to(device=current.device, dtype=current.dtype))


def build_training_state(
    rank: int, device: torch.device
) -> tuple[GateModel, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, Any]:
    seed_globals()
    core = TinyCore().to(device)
    from torch.distributed.fsdp import fully_shard

    fully_shard(core, reshard_after_forward=False)
    wrapper = GateModel(core, rank, device)
    optimizer = torch.optim.AdamW(core.parameters(), lr=2e-3, betas=(0.9, 0.95), eps=1e-8)
    scheduler = build_scheduler(optimizer, make_config(), max_steps=8)
    wrapper.init_ema()
    return wrapper, optimizer, scheduler, core


def load_real_vae(device: torch.device, path: str, dtype: str) -> Flux2VAE:
    require(Path(path).is_dir(), f"FLUX.2 VAE path missing: {path}")
    vae = Flux2VAE(pretrained_path=path, dtype=dtype, sample_posterior=True)
    vae.to(device)
    vae.eval()
    ref = next(vae.parameters())
    require(ref.device == device, f"VAE landed on {ref.device}, expected {device}")
    return vae


def fixed_pixels(rank: int, device: torch.device, image_size: int, batch_jitter: float) -> torch.Tensor:
    count = 3 * image_size * image_size
    pixels = torch.linspace(-0.95, 0.95, count, dtype=torch.float32).reshape(
        1, 3, image_size, image_size
    )
    pixels = (pixels + 0.01 * rank + batch_jitter).clamp(-1.0, 1.0)
    return pixels.to(device)


def full_tensor(value: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "full_tensor"):
        value = value.full_tensor()
    return value.detach().cpu().clone()


def full_grad_state(model: nn.Module) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        require(param.grad is not None, f"missing gradient: {name}")
        result[name] = full_tensor(param.grad)
    return result


def clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    return copy.deepcopy(value)


def full_training_state(
    wrapper: GateModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    raw, optim = get_state_dict(
        wrapper.model,
        optimizer,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    ema = wrapper.ema_state_dict()
    return {
        "raw": clone_tree(raw),
        "ema": clone_tree(ema),
        "optimizer": clone_tree(optim),
        "scheduler": clone_tree(scheduler.state_dict()),
        "lr": [float(group["lr"]) for group in optimizer.param_groups],
    }


def stochastic_update(
    *,
    wrapper: GateModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    vae: Flux2VAE,
    rank: int,
    update_index: int,
    image_size: int,
) -> dict[str, Any]:
    """One update exposing every stochastic value covered by the contract."""
    py_value = random.random()
    np_value = float(np.random.random())
    torch_cpu_value = float(torch.rand(()))
    batch_jitter = (py_value + np_value + torch_cpu_value) * 1e-3
    pixels = fixed_pixels(rank, wrapper.device, image_size, batch_jitter)

    latent = vae.encode(pixels, generator=wrapper.vae_generator)
    require(latent.device == wrapper.device, "latent left the local CUDA device")
    latent_flat = latent.float().flatten(1)
    require(latent_flat.shape[1] >= 16, "unexpectedly small FLUX.2 latent")
    latent_features = latent_flat[:, :16]

    cfg_draw = torch.rand(1, generator=wrapper.dropout_generator)
    cfg_keep = cfg_draw >= 0.35
    sigma_u = torch.randn(1, generator=wrapper.diffusion_generator)
    sigma = torch.sigmoid(sigma_u)
    noise_cpu = torch.randn((1, 16), generator=wrapper.diffusion_generator)
    diffusion_noise = noise_cpu.to(wrapper.device)
    cuda_global = torch.rand((1, 16), device=wrapper.device)

    cfg_scale = cfg_keep.to(wrapper.device, torch.float32).view(1, 1)
    sig = sigma.to(wrapper.device).view(1, 1)
    x0 = latent_features * (0.75 + 0.25 * cfg_scale)
    x_t = (1.0 - sig) * x0 + sig * diffusion_noise + 1e-3 * cuda_global
    target = diffusion_noise - x0 + float(update_index) * 1e-4

    optimizer.zero_grad(set_to_none=True)
    prediction = wrapper.model(x_t)
    loss = (prediction.float() - target.float()).square().mean()
    loss.backward()
    gradients = full_grad_state(wrapper.model)
    optimizer.step()
    wrapper.update_ema()
    scheduler.step()

    local_trace = {
        "rank": rank,
        "batch": torch.tensor(
            [py_value, np_value, torch_cpu_value, batch_jitter], dtype=torch.float64
        ),
        "pixels_digest": tensor_digest(pixels),
        "latent": latent.detach().cpu().clone(),
        "cfg": cfg_keep.detach().cpu().clone(),
        "cfg_draw": cfg_draw.detach().cpu().clone(),
        "sigma": sigma.detach().cpu().clone(),
        "diffusion_noise": noise_cpu.detach().cpu().clone(),
        "cuda_global": cuda_global.detach().cpu().clone(),
        "loss": loss.detach().cpu().clone(),
    }
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_trace)
    state = full_training_state(wrapper, optimizer, scheduler)
    return {
        "per_rank": gathered,
        "grad": gradients,
        **state,
    }


def compare_tree(expected: Any, actual: Any, path: str = "root") -> list[str]:
    failures: list[str] = []
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            return [f"{path}: expected tensor, got {type(actual).__name__}"]
        if expected.dtype != actual.dtype or tuple(expected.shape) != tuple(actual.shape):
            return [
                f"{path}: tensor metadata differs expected={expected.dtype}/{tuple(expected.shape)} "
                f"actual={actual.dtype}/{tuple(actual.shape)}"
            ]
        if not torch.equal(expected, actual):
            if expected.is_floating_point() and actual.is_floating_point():
                diff = (expected.float() - actual.float()).abs()
                failures.append(
                    f"{path}: tensor differs max_abs={float(diff.max())} "
                    f"neq={int(torch.count_nonzero(diff))}/{diff.numel()}"
                )
            else:
                failures.append(f"{path}: tensor differs")
        return failures
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [f"{path}: expected mapping, got {type(actual).__name__}"]
        if set(expected) != set(actual):
            failures.append(
                f"{path}: keys differ expected_only={sorted(set(expected) - set(actual))} "
                f"actual_only={sorted(set(actual) - set(expected))}"
            )
        for key in sorted(set(expected) & set(actual), key=str):
            failures.extend(compare_tree(expected[key], actual[key], f"{path}.{key}"))
        return failures
    if isinstance(expected, (list, tuple)):
        # DCP intentionally normalizes a few optimizer hyperparameter tuples
        # (notably AdamW betas) to lists on set_state_dict -> get_state_dict.
        # Sequence semantics remain exact: require a sequence, identical length,
        # order, element types/values, and recursively bitwise-identical tensors.
        if not isinstance(actual, (list, tuple)):
            return [
                f"{path}: expected sequence {type(expected).__name__}={expected!r}, "
                f"got {type(actual).__name__}={actual!r}"
            ]
        if len(expected) != len(actual):
            return [
                f"{path}: sequence length differs "
                f"expected {type(expected).__name__}={expected!r}, "
                f"actual {type(actual).__name__}={actual!r}"
            ]
        for index, (left, right) in enumerate(zip(expected, actual)):
            failures.extend(compare_tree(left, right, f"{path}[{index}]"))
        return failures
    if expected != actual:
        failures.append(f"{path}: {expected!r} != {actual!r}")
    return failures


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.detach().float().flatten()
    y = right.detach().float().flatten()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(x.square().sum() * y.square().sum())
    require(float(denom) > 0, "Pearson correlation denominator is zero")
    return float((x * y).sum() / denom)


def describe(value: torch.Tensor) -> dict[str, float]:
    x = value.detach().float().flatten()
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "rms": float(torch.sqrt(x.square().mean())),
        "abs_p50": float(x.abs().median()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def rms_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    return describe(numerator)["rms"] / max(describe(denominator)["rms"], 1e-30)


@torch.no_grad()
def phase_flux2_vae(args: argparse.Namespace, rank: int, device: torch.device) -> None:
    seed_globals()
    # fixed_sigma_eval historically drew from the default CUDA generator after
    # torch.manual_seed(seed).  Its explicit device generator must preserve that
    # exact sequence when seeded identically.
    compatibility_seed = 1_234
    torch.manual_seed(compatibility_seed)
    default_cuda_draw = torch.randn((257,), device=device)
    custom_cuda_generator = torch.Generator(device=device).manual_seed(compatibility_seed)
    custom_cuda_draw = torch.randn(
        (257,), device=device, generator=custom_cuda_generator
    )
    require(
        torch.equal(default_cuda_draw, custom_cuda_draw),
        "fixed_sigma default-vs-explicit CUDA generator compatibility changed",
    )
    vae = load_real_vae(device, args.vae_path, args.vae_dtype)
    pixels = fixed_pixels(0, device, args.image_size, 0.0)  # identical posterior on both ranks
    posterior = vae.ae.encode(pixels.to(dtype=next(vae.parameters()).dtype)).latent_dist
    require(torch.device(posterior.parameters.device) == device, "posterior device mismatch")

    rank_seed = BASE_SEED + rank * RANK_STRIDE + STREAM_OFFSETS["vae"]
    generator = torch.Generator(device=device).manual_seed(rank_seed)
    require(torch.device(generator.device) == device, "VAE generator is not exact-device matched")
    state_before = generator.get_state().clone()
    state_before_hash = tensor_digest(state_before)
    raw = Flux2VAE._sample_posterior(
        posterior, do_sample=True, generator=generator, context="gpu hard gate"
    )
    generator.set_state(state_before)
    replay = Flux2VAE._sample_posterior(
        posterior, do_sample=True, generator=generator, context="gpu hard gate replay"
    )
    require(torch.equal(raw, replay), "rank-local real posterior replay is not exact")
    state_after_hash = tensor_digest(generator.get_state())
    require(state_before_hash != state_after_hash, "online posterior did not consume VAE generator")

    sigma = posterior.std
    epsilon = (raw - posterior.mean) / sigma
    raw_posterior_contribution = raw - posterior.mean
    model_latent = Flux2VAE.to_model_space(raw, vae.ae)
    model_mode = Flux2VAE.to_model_space(posterior.mode(), vae.ae)
    model_posterior_delta = model_latent - model_mode

    # Cached training data enters through moments_to_model_space rather than
    # ae.encode.  Exercise that production path against the same real posterior
    # parameters and prove it consumes/replays the rank-local CUDA stream.
    moments_seed = rank_seed + 101
    moments_generator = torch.Generator(device=device).manual_seed(moments_seed)
    moments_before = moments_generator.get_state().clone()
    moments_model_latent = vae.moments_to_model_space(
        posterior.parameters,
        generator=moments_generator,
    )
    moments_after = moments_generator.get_state().clone()
    require(
        not torch.equal(moments_before, moments_after),
        "moments posterior did not consume VAE generator",
    )
    moments_generator.set_state(moments_before)
    moments_replay = vae.moments_to_model_space(
        posterior.parameters,
        generator=moments_generator,
    )
    require(
        torch.equal(moments_model_latent, moments_replay),
        "rank-local moments posterior replay is not exact",
    )
    moments_model_posterior_contribution = moments_model_latent - model_mode

    # Reproduce the old rank-shared failure mode against the same posterior.
    legacy_generator = torch.Generator(device=device).manual_seed(
        BASE_SEED + STREAM_OFFSETS["vae"]
    )
    legacy_raw = Flux2VAE._sample_posterior(
        posterior,
        do_sample=True,
        generator=legacy_generator,
        context="gpu hard gate legacy emulation",
    )
    legacy_epsilon = (legacy_raw - posterior.mean) / sigma

    record = {
        "rank": rank,
        "device": str(device),
        "generator_device": str(generator.device),
        "generator_seed": rank_seed,
        "generator_state_before_hash": state_before_hash,
        "generator_state_after_hash": state_after_hash,
        "posterior_mean_hash": tensor_digest(posterior.mean),
        "epsilon_hash": tensor_digest(epsilon),
        "legacy_epsilon_hash": tensor_digest(legacy_epsilon),
        "epsilon": epsilon.detach().cpu(),
        "legacy_epsilon": legacy_epsilon.detach().cpu(),
        "raw": raw.detach().cpu(),
        "raw_mean": posterior.mean.detach().cpu(),
        "raw_posterior_contribution": raw_posterior_contribution.detach().cpu(),
        "model_latent": model_latent.detach().cpu(),
        "model_mode": model_mode.detach().cpu(),
        "model_posterior_delta": model_posterior_delta.detach().cpu(),
        "moments_model_latent": moments_model_latent.detach().cpu(),
        "moments_model_posterior_contribution": moments_model_posterior_contribution.detach().cpu(),
        "moments_model_latent_hash": tensor_digest(moments_model_latent),
        "moments_generator_state_before_hash": tensor_digest(moments_before),
        "moments_generator_state_after_hash": tensor_digest(moments_after),
        "posterior_sigma": sigma.detach().cpu(),
    }
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, record)
    require(
        len({item["posterior_mean_hash"] for item in gathered}) == 1,
        "identical pixels did not produce an identical posterior mean across ranks",
    )
    require(
        len({item["legacy_epsilon_hash"] for item in gathered}) == 1,
        "legacy shared-seed failure mode was not reproduced",
    )
    require(
        len({item["epsilon_hash"] for item in gathered}) == dist.get_world_size(),
        "rank-derived real posterior epsilon is not unique",
    )
    require(
        len({item["moments_model_latent_hash"] for item in gathered})
        == dist.get_world_size(),
        "rank-derived real moments posterior latent is not unique",
    )
    require(
        len({item["generator_device"] for item in gathered}) == dist.get_world_size(),
        "generators are not bound to distinct exact CUDA devices",
    )

    if rank == 0:
        legacy_corr = pearson(gathered[0]["legacy_epsilon"], gathered[1]["legacy_epsilon"])
        fixed_corr = pearson(gathered[0]["epsilon"], gathered[1]["epsilon"])
        fixed_latent_corr = pearson(gathered[0]["model_latent"], gathered[1]["model_latent"])
        require(
            abs(legacy_corr - 1.0) < 1e-6,
            f"legacy epsilon correlation must be 1 within arithmetic precision, got {legacy_corr}",
        )
        require(abs(fixed_corr) < 0.15, f"rank-derived epsilon correlation too large: {fixed_corr}")
        receipt = {
            "phase": "flux2-vae",
            "schema": "mot_rng_gpu_gate_phase_receipt_v1",
            "status": "PASS",
            "verdict": "PASS",
            "pass": True,
            "vae_path": args.vae_path,
            "vae_dtype": args.vae_dtype,
            "image_size": args.image_size,
            "coverage": {
                "real_flux2_vae": True,
                "online_pixels_posterior": True,
                "cached_moments_posterior": True,
                "exact_local_cuda_generator_device": True,
                "rank_unique_epsilon": True,
                "rank_unique_latent": True,
                "exact_replay": True,
                "generator_consumption": True,
                "posterior_dose_diagnostics": True,
            },
            "per_rank": [
                {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "epsilon",
                        "legacy_epsilon",
                        "raw",
                        "raw_mean",
                        "raw_posterior_contribution",
                        "model_latent",
                        "model_mode",
                        "model_posterior_delta",
                        "moments_model_latent",
                        "moments_model_posterior_contribution",
                        "posterior_sigma",
                    }
                }
                for item in gathered
            ],
            "diagnostics": {
                "fixed_sigma_default_custom_draw_exact": True,
                "fixed_sigma_draw_sha256": tensor_digest(default_cuda_draw),
                "legacy_shared_seed_epsilon_corr": legacy_corr,
                "fixed_rank_seed_epsilon_corr": fixed_corr,
                "fixed_model_space_latent_corr": fixed_latent_corr,
                "posterior_sigma": describe(gathered[0]["posterior_sigma"]),
                "raw_latent_rank0": describe(gathered[0]["raw"]),
                "raw_latent_rank1": describe(gathered[1]["raw"]),
                "model_space_latent_rank0": describe(gathered[0]["model_latent"]),
                "model_space_latent_rank1": describe(gathered[1]["model_latent"]),
                "model_space_mode_rank0": describe(gathered[0]["model_mode"]),
                "model_space_posterior_delta_rank0": describe(
                    gathered[0]["model_posterior_delta"]
                ),
                "moments_model_space_latent_rank0": describe(
                    gathered[0]["moments_model_latent"]
                ),
                "moments_model_space_latent_rank1": describe(
                    gathered[1]["moments_model_latent"]
                ),
                "moments_model_space_latent_corr": pearson(
                    gathered[0]["moments_model_latent"],
                    gathered[1]["moments_model_latent"],
                ),
                "posterior_delta_rms_over_mode_rms_rank0": (
                    describe(gathered[0]["model_posterior_delta"])["rms"]
                    / max(describe(gathered[0]["model_mode"])["rms"], 1e-30)
                ),
                "posterior_dose": {
                    "diagnostic_only": True,
                    "pixels_rank0": {
                        "raw_sample": describe(gathered[0]["raw"]),
                        "raw_mean": describe(gathered[0]["raw_mean"]),
                        "raw_sigma_epsilon": describe(
                            gathered[0]["raw_posterior_contribution"]
                        ),
                        "raw_sigma_epsilon_over_sample_rms": rms_ratio(
                            gathered[0]["raw_posterior_contribution"], gathered[0]["raw"]
                        ),
                        "raw_sigma_epsilon_over_mean_rms": rms_ratio(
                            gathered[0]["raw_posterior_contribution"],
                            gathered[0]["raw_mean"],
                        ),
                        "model_sample": describe(gathered[0]["model_latent"]),
                        "model_mode": describe(gathered[0]["model_mode"]),
                        "model_sigma_epsilon": describe(
                            gathered[0]["model_posterior_delta"]
                        ),
                        "model_sigma_epsilon_over_sample_rms": rms_ratio(
                            gathered[0]["model_posterior_delta"],
                            gathered[0]["model_latent"],
                        ),
                        "model_sigma_epsilon_over_mode_rms": rms_ratio(
                            gathered[0]["model_posterior_delta"], gathered[0]["model_mode"]
                        ),
                    },
                    "cached_moments_rank0": {
                        "model_sample": describe(gathered[0]["moments_model_latent"]),
                        "model_mode": describe(gathered[0]["model_mode"]),
                        "model_sigma_epsilon": describe(
                            gathered[0]["moments_model_posterior_contribution"]
                        ),
                        "model_sigma_epsilon_over_sample_rms": rms_ratio(
                            gathered[0]["moments_model_posterior_contribution"],
                            gathered[0]["moments_model_latent"],
                        ),
                        "model_sigma_epsilon_over_mode_rms": rms_ratio(
                            gathered[0]["moments_model_posterior_contribution"],
                            gathered[0]["model_mode"],
                        ),
                    },
                    "cached_moments_rank1": {
                        "model_sample": describe(gathered[1]["moments_model_latent"]),
                        "model_mode": describe(gathered[1]["model_mode"]),
                        "model_sigma_epsilon": describe(
                            gathered[1]["moments_model_posterior_contribution"]
                        ),
                        "model_sigma_epsilon_over_sample_rms": rms_ratio(
                            gathered[1]["moments_model_posterior_contribution"],
                            gathered[1]["moments_model_latent"],
                        ),
                        "model_sigma_epsilon_over_mode_rms": rms_ratio(
                            gathered[1]["moments_model_posterior_contribution"],
                            gathered[1]["model_mode"],
                        ),
                    },
                },
            },
            "provenance": provenance(),
        }
        write_json(Path(args.artifact_dir) / "flux2_vae_receipt.json", receipt)
        print("PASS flux2-vae", json.dumps(receipt["diagnostics"], sort_keys=True), flush=True)


def _trace_destination(value: Any) -> str:
    try:
        fd = int(value.fileno())
        return os.readlink(f"/proc/self/fd/{fd}")
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    candidate = getattr(value, "name", value)
    try:
        return os.fspath(candidate)
    except TypeError:
        return f"<{type(value).__name__}>"


def _validate_checkpoint_io_trace(
    events: list[dict[str, Any]],
    *,
    checkpoint_path: Path,
    periodic_path: Path,
) -> dict[str, Any]:
    """Require atomic, durable, rank-0-only full checkpoint publication."""

    checkpoint = str(checkpoint_path)
    periodic = str(periodic_path)
    saves = [(index, item) for index, item in enumerate(events) if item["event"] == "torch.save"]
    replaces = [(index, item) for index, item in enumerate(events) if item["event"] == "replace"]
    links = [(index, item) for index, item in enumerate(events) if item["event"] == "link"]
    chmods = [(index, item) for index, item in enumerate(events) if item["event"] == "chmod"]
    fsyncs = [(index, item) for index, item in enumerate(events) if item["event"] == "fsync"]
    opens = [(index, item) for index, item in enumerate(events) if item["event"] == "open"]
    require(len(saves) == 1, f"checkpoint must have exactly one torch.save: {events}")
    require(len(replaces) == 1, f"checkpoint must have exactly one atomic replace: {events}")
    require(len(links) == 1, f"full milestone must have exactly one hard-link publish: {events}")
    save_index, save_event = saves[0]
    replace_index, replace_event = replaces[0]
    link_index, link_event = links[0]
    temp_path = save_event["destination"]
    require(temp_path != checkpoint, "torch.save wrote directly to last.pt")
    require(
        Path(temp_path).parent == checkpoint_path.parent
        and Path(temp_path).name.startswith(f".{checkpoint_path.name}.tmp."),
        f"checkpoint temporary path is not randomized under last.pt: {temp_path!r}",
    )
    temp_open = [
        item
        for _, item in opens
        if item["path"] == temp_path and item["flags"] & os.O_EXCL
    ]
    require(len(temp_open) == 1, f"exclusive temporary open was not observed: {events}")
    require(
        temp_open[0]["flags"] & os.O_EXCL,
        f"checkpoint temporary was not opened O_EXCL: {temp_open[0]}",
    )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    require(
        nofollow == 0 or temp_open[0]["flags"] & nofollow,
        f"checkpoint temporary was not opened O_NOFOLLOW: {temp_open[0]}",
    )
    require(
        replace_event["source"] == temp_path and replace_event["destination"] == checkpoint,
        f"atomic replace did not publish the serialized temporary: {replace_event}",
    )
    require(save_index < replace_index, "checkpoint replace occurred before torch.save")
    require(
        link_event["source"] == checkpoint and link_event["destination"] == periodic,
        f"full milestone link has wrong endpoints: {link_event}",
    )
    require(replace_index < link_index, "full milestone linked before last.pt publication")
    require(
        any(index > link_index and item["mode"] == 0o444 for index, item in chmods),
        f"full milestone was not made mode 0444 after linking: {events}",
    )
    require(
        any(
            save_index < index < replace_index and item["kind"] == "file"
            for index, item in fsyncs
        ),
        f"serialized temporary was not fsynced before replace: {events}",
    )
    require(
        any(index > replace_index and item["kind"] == "file" for index, item in fsyncs),
        f"published checkpoint inode was not fsynced after replace: {events}",
    )
    require(
        any(index > link_index and item["kind"] == "directory" for index, item in fsyncs),
        f"checkpoint directory was not fsynced after milestone link: {events}",
    )
    require(not Path(temp_path).exists(), f"checkpoint temporary remains after publish: {temp_path}")
    return {
        "rank0_only_serialization": True,
        "random_exclusive_temporary": True,
        "exclusive_no_follow_temporary_open": True,
        "atomic_replace": True,
        "temporary_file_fsync_before_replace": True,
        "published_file_fsync_after_replace": True,
        "full_hardlink_0444": True,
        "directory_fsync_after_hardlink": True,
        "temporary_removed": True,
        "torch_save_calls": 1,
    }


def save_checkpoint_with_io_trace(
    *,
    checkpoint_path: Path,
    periodic_path: Path,
    wrapper: GateModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    global_step: int,
    rank: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run production save while observing, but not replacing, filesystem calls."""

    events: list[dict[str, Any]] = []
    real_torch_save = torch.save
    real_open = os.open
    real_fsync = os.fsync
    real_replace = os.replace
    real_link = os.link
    real_chmod = os.chmod

    def recording_open(path: Any, flags: int, *call_args: Any, **call_kwargs: Any) -> int:
        events.append({"event": "open", "path": os.fspath(path), "flags": int(flags)})
        return real_open(path, flags, *call_args, **call_kwargs)

    def recording_torch_save(*call_args: Any, **call_kwargs: Any) -> Any:
        destination = call_args[1] if len(call_args) > 1 else call_kwargs.get("f")
        events.append({"event": "torch.save", "destination": _trace_destination(destination)})
        return real_torch_save(*call_args, **call_kwargs)

    def recording_fsync(fd: int) -> None:
        try:
            mode = os.fstat(fd).st_mode
            kind = "directory" if stat.S_ISDIR(mode) else "file"
            target = os.readlink(f"/proc/self/fd/{fd}")
        except BaseException as exc:
            kind = "unknown"
            target = f"<unavailable:{type(exc).__name__}>"
        events.append({"event": "fsync", "kind": kind, "target": target})
        real_fsync(fd)

    def recording_replace(source: Any, destination: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        events.append(
            {
                "event": "replace",
                "source": os.fspath(source),
                "destination": os.fspath(destination),
            }
        )
        return real_replace(source, destination, *call_args, **call_kwargs)

    def recording_link(source: Any, destination: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        events.append(
            {
                "event": "link",
                "source": os.fspath(source),
                "destination": os.fspath(destination),
            }
        )
        return real_link(source, destination, *call_args, **call_kwargs)

    def recording_chmod(path: Any, mode: int, *call_args: Any, **call_kwargs: Any) -> Any:
        events.append({"event": "chmod", "path": os.fspath(path), "mode": int(mode)})
        return real_chmod(path, mode, *call_args, **call_kwargs)

    with (
        unittest.mock.patch.object(torch, "save", side_effect=recording_torch_save),
        unittest.mock.patch.object(os, "open", side_effect=recording_open),
        unittest.mock.patch.object(os, "fsync", side_effect=recording_fsync),
        unittest.mock.patch.object(os, "replace", side_effect=recording_replace),
        unittest.mock.patch.object(os, "link", side_effect=recording_link),
        unittest.mock.patch.object(os, "chmod", side_effect=recording_chmod),
    ):
        save_checkpoint(
            str(checkpoint_path),
            wrapper,
            optimizer,
            scheduler,
            global_step,
            True,
            rank,
            make_config(),
        )
    dist.barrier()
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, events)
    require(gathered[1] == [], f"rank1 performed checkpoint filesystem writes: {gathered[1]}")
    summary = _validate_checkpoint_io_trace(
        gathered[0], checkpoint_path=checkpoint_path, periodic_path=periodic_path
    )
    return gathered, summary


def expect_full_capacity_failure(
    *,
    checkpoint_path: Path,
    wrapper: GateModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rank: int,
) -> dict[str, Any]:
    """The fifth full milestone must fail without deleting the four anchors."""

    anchors = [checkpoint_path.parent / f"step_{step}.pt" for step in range(1, 5)]
    before_hashes = {path.name: sha256_file(path) for path in anchors}
    before_rng = capture_local_rng_state(wrapper, rank, dist.get_world_size())
    before_model = local_model_digest(wrapper.model)
    torch_save_calls = 0
    real_torch_save = torch.save

    def recording_torch_save(*call_args: Any, **call_kwargs: Any) -> Any:
        nonlocal torch_save_calls
        torch_save_calls += 1
        return real_torch_save(*call_args, **call_kwargs)

    error: str | None = None
    with unittest.mock.patch.object(torch, "save", side_effect=recording_torch_save):
        try:
            save_checkpoint(
                str(checkpoint_path),
                wrapper,
                optimizer,
                scheduler,
                5,
                True,
                rank,
                make_config(),
            )
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
    after_rng = capture_local_rng_state(wrapper, rank, dist.get_world_size())
    after_model = local_model_digest(wrapper.model)
    require(rng_states_equal(before_rng, after_rng), "full-capacity failure mutated live RNG")
    require(before_model == after_model, "full-capacity failure mutated model state")
    errors: list[Any] = [None] * dist.get_world_size()
    counts: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    dist.all_gather_object(counts, torch_save_calls)
    require(all(item is not None for item in errors), f"full-capacity save did not fail: {errors}")
    require(len(set(errors)) == 1, f"full-capacity errors differed across ranks: {errors}")
    require(counts[1] == 0, f"rank1 serialized during full-capacity failure: {counts}")
    require(
        not (checkpoint_path.parent / "step_5.pt").exists(),
        "full-capacity failure published forbidden step_5.pt",
    )
    for anchor in anchors:
        require(anchor.is_file() and not anchor.is_symlink(), f"full anchor missing: {anchor}")
        require(sha256_file(anchor) == before_hashes[anchor.name], f"full anchor changed: {anchor}")
        require(stat.S_IMODE(anchor.stat().st_mode) == 0o444, f"full anchor is writable: {anchor}")

    # Current production detects capacity while publishing the periodic anchor,
    # after last.pt is durably replaced.  Restore last.pt to the valid fourth
    # anchor so the next independent phase exercises the intended checkpoint.
    if rank == 0:
        checkpoint_path.unlink()
        os.link(anchors[-1], checkpoint_path)
        directory_fd = os.open(checkpoint_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    dist.barrier()
    require(os.path.samefile(checkpoint_path, anchors[-1]), "last.pt was not restored to step_4")
    require(stat.S_IMODE(checkpoint_path.stat().st_mode) == 0o444, "restored last.pt is writable")
    return {
        "case": "fifth_full_milestone_capacity_failure",
        "all_rank_error": errors[0],
        "all_rank_identical_failure": True,
        "rank0_torch_save_calls": counts[0],
        "rank1_torch_save_calls": counts[1],
        "retained_anchor_sha256": before_hashes,
        "retained_anchor_modes": {path.name: "0o444" for path in anchors},
        "step_5_published": False,
        "last_restored_to_step_4": True,
        "no_rng_mutation": True,
        "no_model_mutation": True,
    }


def phase_save_reference(args: argparse.Namespace, rank: int, device: torch.device) -> None:
    wrapper, optimizer, scheduler, _ = build_training_state(rank, device)
    vae = load_real_vae(device, args.vae_path, args.vae_dtype)
    stochastic_update(
        wrapper=wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        vae=vae,
        rank=rank,
        update_index=0,
        image_size=args.image_size,
    )
    checkpoint_path = Path(args.artifact_dir) / "last.pt"
    periodic_path = Path(args.artifact_dir) / "step_1.pt"
    predictable_tmp_canary = Path(args.artifact_dir) / "last.pt.tmp"
    if rank == 0:
        require(
            not os.path.lexists(predictable_tmp_canary),
            f"temporary symlink canary path already exists: {predictable_tmp_canary}",
        )
        predictable_tmp_canary.symlink_to("/dev/null")
    dist.barrier()
    io_traces, io_summary = save_checkpoint_with_io_trace(
        checkpoint_path=checkpoint_path,
        periodic_path=periodic_path,
        wrapper=wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=1,
        rank=rank,
    )
    require(predictable_tmp_canary.is_symlink(), "predictable tmp symlink canary was followed/replaced")
    require(os.readlink(predictable_tmp_canary) == "/dev/null", "tmp symlink canary target changed")
    if rank == 0:
        predictable_tmp_canary.unlink()
    dist.barrier()
    require(not os.path.lexists(predictable_tmp_canary), "tmp symlink canary cleanup failed")
    last_stat = checkpoint_path.stat()
    periodic_stat = periodic_path.stat()
    retention_local = {
        "rank": rank,
        "samefile": os.path.samefile(checkpoint_path, periodic_path),
        "last_dev": last_stat.st_dev,
        "last_ino": last_stat.st_ino,
        "step_dev": periodic_stat.st_dev,
        "step_ino": periodic_stat.st_ino,
        "nlink": last_stat.st_nlink,
        "torch_save_calls": sum(item["event"] == "torch.save" for item in io_traces[rank]),
    }
    require(retention_local["samefile"], "step_1.pt is not the same file as last.pt")
    require(last_stat.st_dev == periodic_stat.st_dev, "full snapshot device differs")
    require(last_stat.st_ino == periodic_stat.st_ino, "full snapshot inode differs")
    require(last_stat.st_nlink >= 2, "full snapshot hard-link count is below 2")
    retention_before_crash: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(retention_before_crash, retention_local)
    if rank == 0:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(payload["checkpoint_schema"] == FULL_CHECKPOINT_SCHEMA, "wrong checkpoint schema")
        envelope = payload["rng_state"]
        require(envelope["schema"] == RNG_STATE_SCHEMA, "wrong RNG schema")
        require(envelope["world_size"] == 2, "checkpoint RNG world size is not 2")
        require([item["rank"] for item in envelope["per_rank"]] == [0, 1], "rank closure mismatch")
        require(
            payload.get(FORMAL_RUN_LINEAGE_KEY) == required_lineage(),
            "saved checkpoint did not bind the expected formal run lineage",
        )
        for stream in STREAM_OFFSETS:
            hashes = {tensor_digest(item["generators"][stream]) for item in envelope["per_rank"]}
            require(len(hashes) == 2, f"checkpoint stream {stream} is duplicated across ranks")
        periodic_path.unlink()
    dist.barrier()
    require(not periodic_path.exists(), "simulated crash did not remove step_1.pt")

    expected = stochastic_update(
        wrapper=wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        vae=vae,
        rank=rank,
        update_index=1,
        image_size=args.image_size,
    )
    if rank == 0:
        require(
            expected["scheduler"]["last_epoch"] == 2,
            "stage-local scheduler did not advance from checkpoint step 1 to next step 2",
        )
        torch.save(expected, Path(args.artifact_dir) / "expected_next_update.pt")
        receipt = {
            "phase": "save-reference",
            "schema": "mot_rng_gpu_gate_phase_receipt_v1",
            "status": "PASS",
            "verdict": "PASS",
            "pass": True,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "expected_fields": [
                "batch",
                "latent",
                "cfg",
                "sigma",
                "diffusion_noise",
                "loss",
                "grad",
                "raw",
                "ema",
                "optimizer",
                "scheduler",
                "lr",
            ],
            "checkpoint_global_step": 1,
            "formal_run_lineage_sha256": required_lineage(),
            "retention_before_crash": {
                "format": "full",
                "checkpoint_keep_every": 1,
                "checkpoint_keep_max": 4,
                "per_rank": retention_before_crash,
                "rank0_torch_save_calls": sum(
                    item["event"] == "torch.save" for item in io_traces[0]
                ),
                "rank1_torch_save_calls": sum(
                    item["event"] == "torch.save" for item in io_traces[1]
                ),
                "durable_atomic_publication": io_summary,
                "predictable_tmp_symlink_canary_untouched": True,
                "step_1_deleted_after_validation": True,
            },
            "expected_next_scheduler_last_epoch": expected["scheduler"]["last_epoch"],
            "expected_next_lr": expected["lr"],
            "provenance": provenance(),
        }
        write_json(Path(args.artifact_dir) / "save_reference_receipt.json", receipt)
        print("PASS save-reference", flush=True)


def consume_resume_startup_rng(wrapper: GateModel) -> None:
    random.random()
    np.random.random(7)
    torch.rand(9)
    torch.rand(9, device=wrapper.device)
    torch.rand(9, generator=wrapper.dropout_generator)
    torch.rand(9, generator=wrapper.diffusion_generator)
    torch.rand(9, device=wrapper.device, generator=wrapper.vae_generator)


def phase_resume_compare(args: argparse.Namespace, rank: int, device: torch.device) -> None:
    wrapper, optimizer, scheduler, _ = build_training_state(rank, device)
    vae = load_real_vae(device, args.vae_path, args.vae_dtype)
    checkpoint_path = Path(args.artifact_dir) / "last.pt"
    require(checkpoint_path.is_file(), f"checkpoint missing: {checkpoint_path}")
    global_step, pending = load_checkpoint(
        str(checkpoint_path), wrapper, optimizer, scheduler, True, make_config()
    )
    require(global_step == 1, f"resume global_step={global_step}, expected 1")
    periodic_path = Path(args.artifact_dir) / "step_1.pt"
    require(periodic_path.is_file(), "resume did not repair missing step_1.pt")
    last_stat = checkpoint_path.stat()
    periodic_stat = periodic_path.stat()
    retention_local = {
        "rank": rank,
        "samefile": os.path.samefile(checkpoint_path, periodic_path),
        "last_dev": last_stat.st_dev,
        "last_ino": last_stat.st_ino,
        "step_dev": periodic_stat.st_dev,
        "step_ino": periodic_stat.st_ino,
        "nlink": last_stat.st_nlink,
    }
    require(retention_local["samefile"], "repaired step_1.pt is not last.pt hard link")
    require(last_stat.st_dev == periodic_stat.st_dev, "repaired snapshot device differs")
    require(last_stat.st_ino == periodic_stat.st_ino, "repaired snapshot inode differs")
    require(last_stat.st_nlink >= 2, "repaired snapshot link count below 2")
    retention_after_resume: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(retention_after_resume, retention_local)
    consume_resume_startup_rng(wrapper)
    restore_local_rng_state_synchronized(pending, wrapper, is_dist=True)
    assert_live_rng_matches_synchronized(
        pending,
        wrapper,
        rank=rank,
        world_size=dist.get_world_size(),
        is_dist=True,
        context="GPU gate resume startup tail",
    )
    actual = stochastic_update(
        wrapper=wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        vae=vae,
        rank=rank,
        update_index=1,
        image_size=args.image_size,
    )
    failures: list[str] = []
    optimizer_sequence_evidence: dict[str, Any] | None = None
    if rank == 0:
        expected = torch.load(
            Path(args.artifact_dir) / "expected_next_update.pt",
            map_location="cpu",
            weights_only=False,
        )
        failures = compare_tree(expected, actual)
        expected_betas = expected["optimizer"]["param_groups"][0]["betas"]
        actual_betas = actual["optimizer"]["param_groups"][0]["betas"]
        optimizer_sequence_evidence = {
            "field": "optimizer.param_groups[0].betas",
            "expected_type": type(expected_betas).__name__,
            "expected_repr": repr(expected_betas),
            "actual_type": type(actual_betas).__name__,
            "actual_repr": repr(actual_betas),
            "semantic_sequence_exact": not compare_tree(
                expected_betas, actual_betas, "optimizer.param_groups[0].betas"
            ),
        }
    objects: list[Any] = [failures]
    dist.broadcast_object_list(objects, src=0)
    failures = objects[0]
    require(not failures, "next update differs after restart:\n" + "\n".join(failures[:40]))

    # Publish full milestones 2-4 after the exact step-2 comparison.  Every
    # earlier anchor must remain immutable; a fifth milestone then fails closed
    # at capacity without deleting any of them.
    anchor_paths = {
        step: Path(args.artifact_dir) / f"step_{step}.pt" for step in range(1, 5)
    }
    step_1_sha256 = sha256_file(anchor_paths[1])
    io_summaries: dict[str, Any] = {}
    for step in range(2, 5):
        if step > 2:
            stochastic_update(
                wrapper=wrapper,
                optimizer=optimizer,
                scheduler=scheduler,
                vae=vae,
                rank=rank,
                update_index=step - 1,
                image_size=args.image_size,
            )
        _, io_summary = save_checkpoint_with_io_trace(
            checkpoint_path=checkpoint_path,
            periodic_path=anchor_paths[step],
            wrapper=wrapper,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=step,
            rank=rank,
        )
        io_summaries[str(step)] = io_summary
        for retained_step in range(1, step + 1):
            anchor = anchor_paths[retained_step]
            require(anchor.is_file() and not anchor.is_symlink(), f"missing full anchor: {anchor}")
            require(
                stat.S_IMODE(anchor.stat().st_mode) == 0o444,
                f"full anchor is not mode 0444: {anchor}",
            )
        require(
            sha256_file(anchor_paths[1]) == step_1_sha256,
            f"step_1.pt changed while publishing step_{step}.pt",
        )
        require(
            os.path.samefile(anchor_paths[step], checkpoint_path),
            f"step_{step}.pt is not the current last.pt hard link",
        )

    retention_local = {
        "rank": rank,
        "retained_steps": [1, 2, 3, 4],
        "anchors": {
            str(step): {
                "sha256": sha256_file(path),
                "device": path.stat().st_dev,
                "inode": path.stat().st_ino,
                "nlink": path.stat().st_nlink,
                "mode": oct(stat.S_IMODE(path.stat().st_mode)),
                "samefile_last": os.path.samefile(path, checkpoint_path),
            }
            for step, path in anchor_paths.items()
        },
    }
    require(retention_local["anchors"]["4"]["samefile_last"], "step_4 is not last.pt")
    require(
        all(not retention_local["anchors"][str(step)]["samefile_last"] for step in (1, 2, 3)),
        "historical full anchor unexpectedly aliases current last.pt",
    )
    retention_all_ranks: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(retention_all_ranks, retention_local)
    if rank == 0:
        for step, path in anchor_paths.items():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            require(payload["global_step"] == step, f"full anchor payload step mismatch: {path}")
            require(
                payload.get(FORMAL_RUN_LINEAGE_KEY) == required_lineage(),
                f"full anchor lost canonical lineage binding: {path}",
            )

    stochastic_update(
        wrapper=wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        vae=vae,
        rank=rank,
        update_index=4,
        image_size=args.image_size,
    )
    capacity_failure = expect_full_capacity_failure(
        checkpoint_path=checkpoint_path,
        wrapper=wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        rank=rank,
    )
    terminal_duplicate_guard = {
        "none_previous_writes": should_write_final_checkpoint(4, None),
        "earlier_previous_writes": should_write_final_checkpoint(4, 3),
        "same_terminal_step_skips": not should_write_final_checkpoint(4, 4),
    }
    require(all(terminal_duplicate_guard.values()), "terminal duplicate checkpoint guard failed")
    if rank == 0:
        receipt = {
            "phase": "resume-compare",
            "schema": "mot_rng_gpu_gate_phase_receipt_v1",
            "status": "PASS",
            "verdict": "PASS",
            "pass": True,
            "checkpoint_global_step": global_step,
            "formal_run_lineage_sha256": required_lineage(),
            "retention_after_resume_repair": {
                "step_1_present": True,
                "all_rank_hardlink_evidence": retention_after_resume,
            },
            "full_retention": {
                "full_snapshots_never_roll": True,
                "checkpoint_keep_every": 1,
                "checkpoint_keep_max": 4,
                "retained_global_steps": [1, 2, 3, 4],
                "step_1_byte_exact_through_step_4": True,
                "all_rank_evidence": retention_all_ranks,
                "durable_atomic_publication_by_step": io_summaries,
                "capacity_failure": capacity_failure,
            },
            "terminal_duplicate_guard": terminal_duplicate_guard,
            "actual_next_scheduler_last_epoch": actual["scheduler"]["last_epoch"],
            "actual_next_lr": actual["lr"],
            "optimizer_sequence_normalization": optimizer_sequence_evidence,
            "exact_fields": [
                "per_rank.batch",
                "per_rank.latent",
                "per_rank.cfg",
                "per_rank.sigma",
                "per_rank.diffusion_noise",
                "per_rank.loss",
                "grad",
                "raw",
                "ema",
                "optimizer",
                "scheduler",
                "lr",
            ],
            "provenance": provenance(),
        }
        write_json(Path(args.artifact_dir) / "resume_compare_receipt.json", receipt)
        print("PASS resume-compare exact next update", flush=True)


def rng_states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["rank"] != right["rank"] or left["device_type"] != right["device_type"]:
        return False
    if left["python"] != right["python"]:
        return False
    lnp, rnp = left["numpy"], right["numpy"]
    if not (lnp[0] == rnp[0] and np.array_equal(lnp[1], rnp[1]) and lnp[2:] == rnp[2:]):
        return False
    for key in ("torch_cpu", "torch_cuda"):
        if left[key] is None or right[key] is None:
            if left[key] is not right[key]:
                return False
        elif not torch.equal(left[key], right[key]):
            return False
    return all(
        torch.equal(left["generators"][name], right["generators"][name])
        for name in STREAM_OFFSETS
    )


def local_model_digest(model: nn.Module) -> dict[str, str]:
    result = {}
    for name, param in model.named_parameters():
        local = param.to_local() if hasattr(param, "to_local") else param
        result[name] = tensor_digest(local)
    return result


def expect_synchronized_boot_branch_failure(
    *,
    name: str,
    checkpoint_path: Path,
    expected_mode: str | None,
    init_from: str | None,
    wrapper: GateModel,
    rank: int,
) -> dict[str, Any]:
    before_rng = capture_local_rng_state(wrapper, rank, dist.get_world_size())
    before_model = local_model_digest(wrapper.model)
    error: str | None = None
    try:
        resolve_checkpoint_boot_branch(
            str(checkpoint_path),
            resume=True,
            init_from=init_from,
            expected_mode=expected_mode,
            is_dist=True,
        )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    after_rng = capture_local_rng_state(wrapper, rank, dist.get_world_size())
    after_model = local_model_digest(wrapper.model)
    require(rng_states_equal(before_rng, after_rng), f"{name}: branch gate mutated live RNG")
    require(before_model == after_model, f"{name}: branch gate mutated model state")
    errors: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    require(all(item is not None for item in errors), f"{name}: not every rank failed: {errors}")
    require(len(set(errors)) == 1, f"{name}: rank errors differ: {errors}")
    return {
        "case": name,
        "all_rank_error": errors[0],
        "all_rank_identical_failure": True,
        "pre_load": True,
        "no_rng_mutation": True,
        "no_model_mutation": True,
    }


def write_corrupt_checkpoint(
    original: Path, target: Path, mutation: str, rank: int
) -> None:
    if rank != 0:
        return
    payload = torch.load(original, map_location="cpu", weights_only=False)
    if mutation == "missing_rng_schema":
        del payload["rng_state"]["schema"]
    elif mutation == "wrong_checkpoint_schema":
        payload["checkpoint_schema"] = "mot_full_training_BAD"
    elif mutation == "world_size":
        payload["rng_state"]["world_size"] = 3
    elif mutation == "missing_stream":
        del payload["rng_state"]["per_rank"][1]["generators"]["vae"]
    elif mutation == "duplicate_stream":
        payload["rng_state"]["per_rank"][1]["generators"]["vae"] = payload[
            "rng_state"
        ]["per_rank"][0]["generators"]["vae"].clone()
    elif mutation == "rank1_opaque":
        payload["rng_state"]["per_rank"][1]["generators"]["vae"] = torch.ones(
            3, dtype=torch.uint8
        )
    elif mutation == "missing_lineage":
        del payload[FORMAL_RUN_LINEAGE_KEY]
    elif mutation == "wrong_lineage":
        expected = required_lineage()
        payload[FORMAL_RUN_LINEAGE_KEY] = ("b" if expected != "b" * 64 else "c") * 64
    else:
        raise ValueError(mutation)
    torch.save(payload, target)


def expect_synchronized_load_failure(
    *,
    name: str,
    path_for_rank: Path,
    wrapper: GateModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rank: int,
) -> dict[str, Any]:
    before_rng = capture_local_rng_state(wrapper, rank, dist.get_world_size())
    before_model = local_model_digest(wrapper.model)
    error: str | None = None
    try:
        load_checkpoint(
            str(path_for_rank), wrapper, optimizer, scheduler, True, make_config()
        )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    after_rng = capture_local_rng_state(wrapper, rank, dist.get_world_size())
    after_model = local_model_digest(wrapper.model)
    require(rng_states_equal(before_rng, after_rng), f"{name}: failed load mutated live RNG")
    require(before_model == after_model, f"{name}: failed load mutated model state")
    errors: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    require(all(item is not None for item in errors), f"{name}: not every rank failed: {errors}")
    require(len(set(errors)) == 1, f"{name}: ranks disagreed on failure: {errors}")
    return {"case": name, "all_rank_error": errors[0]}


def phase_corruption(args: argparse.Namespace, rank: int, device: torch.device) -> None:
    wrapper, optimizer, scheduler, _ = build_training_state(rank, device)
    artifact_dir = Path(args.artifact_dir)
    original = artifact_dir / "last.pt"
    require(original.is_file(), f"checkpoint missing: {original}")
    missing_boot_path = artifact_dir / "boot_branch_missing.pt"
    require(not os.path.lexists(missing_boot_path), "boot-branch missing-path canary exists")
    positive_boot_branches = {
        "continue": resolve_checkpoint_boot_branch(
            str(original),
            resume=True,
            init_from="/unused/init.pt",
            expected_mode="continue",
            is_dist=True,
        ),
        "fresh": resolve_checkpoint_boot_branch(
            str(missing_boot_path),
            resume=True,
            init_from="/synthetic/init.pt",
            expected_mode="fresh",
            is_dist=True,
        ),
        "none": resolve_checkpoint_boot_branch(
            str(missing_boot_path),
            resume=True,
            init_from=None,
            expected_mode=None,
            is_dist=True,
        ),
    }
    require(
        positive_boot_branches == {"continue": "continue", "fresh": "fresh", "none": "none"},
        f"positive boot branch selection mismatch: {positive_boot_branches}",
    )
    split_path = missing_boot_path if rank == 0 else original
    boot_branch_failures = [
        expect_synchronized_boot_branch_failure(
            name="rank_visibility_split_pre_load",
            checkpoint_path=split_path,
            expected_mode=None,
            init_from="/synthetic/init.pt",
            wrapper=wrapper,
            rank=rank,
        ),
        expect_synchronized_boot_branch_failure(
            name="all_rank_expected_fresh_but_continue_pre_load",
            checkpoint_path=original,
            expected_mode="fresh",
            init_from="/synthetic/init.pt",
            wrapper=wrapper,
            rank=rank,
        ),
    ]
    receipts = []
    shared_cases = (
        "missing_lineage",
        "wrong_lineage",
        "missing_rng_schema",
        "wrong_checkpoint_schema",
        "world_size",
        "missing_stream",
        "duplicate_stream",
    )
    for case in shared_cases:
        target = artifact_dir / f"corrupt_{case}.pt"
        write_corrupt_checkpoint(original, target, case, rank)
        dist.barrier()
        receipts.append(
            expect_synchronized_load_failure(
                name=case,
                path_for_rank=target,
                wrapper=wrapper,
                optimizer=optimizer,
                scheduler=scheduler,
                rank=rank,
            )
        )

    rank1_bad = artifact_dir / "corrupt_rank1_opaque.pt"
    write_corrupt_checkpoint(original, rank1_bad, "rank1_opaque", rank)
    dist.barrier()
    receipts.append(
        expect_synchronized_load_failure(
            name="rank1_only_opaque_corruption",
            path_for_rank=rank1_bad if rank == 1 else original,
            wrapper=wrapper,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
        )
    )

    # Formal resume is symmetric: a checkpoint carrying a lineage may not be
    # resumed with the expected environment missing, and a wrong all-rank
    # environment is just as fatal as a wrong checkpoint key.
    expected_lineage = required_lineage()
    for env_case in ("missing", "wrong"):
        if env_case == "missing":
            os.environ.pop(EXPECTED_RUN_LINEAGE_ENV, None)
        else:
            os.environ[EXPECTED_RUN_LINEAGE_ENV] = (
                "f" if expected_lineage != "f" * 64 else "0"
            ) * 64
        try:
            receipts.append(
                expect_synchronized_load_failure(
                    name=f"all_rank_environment_lineage_{env_case}",
                    path_for_rank=original,
                    wrapper=wrapper,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    rank=rank,
                )
            )
        finally:
            os.environ[EXPECTED_RUN_LINEAGE_ENV] = expected_lineage

    # A single rank reading a checkpoint with the wrong lineage must stop every
    # rank before any DCP state load or RNG mutation.
    wrong_lineage = artifact_dir / "corrupt_wrong_lineage.pt"
    receipts.append(
        expect_synchronized_load_failure(
            name="rank1_only_checkpoint_lineage_mismatch",
            path_for_rank=wrong_lineage if rank == 1 else original,
            wrapper=wrapper,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
        )
    )

    # Rank-local environment drift is caught in the lineage collective itself,
    # before any rank even opens the checkpoint.
    expected_lineage = required_lineage()
    if rank == 1:
        os.environ[EXPECTED_RUN_LINEAGE_ENV] = (
            "d" if expected_lineage != "d" * 64 else "e"
        ) * 64
    try:
        receipts.append(
            expect_synchronized_load_failure(
                name="rank1_only_environment_lineage_mismatch",
                path_for_rank=original,
                wrapper=wrapper,
                optimizer=optimizer,
                scheduler=scheduler,
                rank=rank,
            )
        )
    finally:
        os.environ[EXPECTED_RUN_LINEAGE_ENV] = expected_lineage

    # A current full milestone may be repaired, but an older full anchor is
    # immutable evidence and must never be silently recreated or ignored.
    prior_anchor = artifact_dir / "step_1.pt"
    withheld_anchor = artifact_dir / "step_1.pt.withheld"
    require(prior_anchor.is_file(), "prior full anchor missing before negative retention test")
    if rank == 0:
        os.replace(prior_anchor, withheld_anchor)
    dist.barrier()
    try:
        retention_failure_case = expect_synchronized_load_failure(
            name="missing_prior_full_anchor",
            path_for_rank=original,
            wrapper=wrapper,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
        )
    finally:
        if rank == 0 and withheld_anchor.is_file():
            os.replace(withheld_anchor, prior_anchor)
        dist.barrier()
    require(prior_anchor.is_file(), "prior full anchor was not restored after negative test")
    require(not withheld_anchor.exists(), "withheld-anchor temporary remains")
    if rank == 0:
        receipt = {
            "phase": "corruption",
            "schema": "mot_rng_gpu_gate_phase_receipt_v1",
            "status": "PASS",
            "verdict": "PASS",
            "pass": True,
            "case_count": len(receipts),
            "all_rank_identical_failure": True,
            "no_rng_mutation": True,
            "no_model_mutation": True,
            "cases": receipts,
            "retention_failure_cases": [retention_failure_case],
            "boot_branch_visibility": {
                "positive": positive_boot_branches,
                "negative_case_count": len(boot_branch_failures),
                "negative_cases": boot_branch_failures,
                "all_rank_identical_pre_load_failure": True,
                "no_rng_mutation": True,
                "no_model_mutation": True,
            },
            "invariants": [
                "every rank raised the identical synchronized error",
                "failed validation did not mutate live RNG",
                "failed validation did not mutate model state",
            ],
            "provenance": provenance(),
        }
        write_json(artifact_dir / "corruption_receipt.json", receipt)
        print("PASS corruption matrix", [item["case"] for item in receipts], flush=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=("flux2-vae", "save-reference", "resume-compare", "corruption"),
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument(
        "--vae-path", default=os.environ.get("FLUX2_VAE_PATH", "/fsx/models/FLUX.2-klein-4B/vae")
    )
    parser.add_argument("--vae-dtype", default="bfloat16", choices=("float32", "bfloat16"))
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()
    require(args.image_size >= 32 and args.image_size % 16 == 0, "image size must be >=32 and /16")
    return args


def main() -> None:
    args = parse_args()
    rank, _, _, device = init_distributed()
    required_lineage()
    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    dist.barrier()
    try:
        if args.phase == "flux2-vae":
            phase_flux2_vae(args, rank, device)
        elif args.phase == "save-reference":
            phase_save_reference(args, rank, device)
        elif args.phase == "resume-compare":
            phase_resume_compare(args, rank, device)
        elif args.phase == "corruption":
            phase_corruption(args, rank, device)
        else:  # argparse makes this unreachable
            raise AssertionError(args.phase)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
