#!/usr/bin/env python3
"""Read-only FSDP2 v3 gate for Stage512A-final -> A-control init_from.

The probe deliberately does not construct the dataset and never writes a training
checkpoint.  It exercises the same model/FSDP2/optimizer/scheduler/EMA construction
and ``init_from_checkpoint`` path as train.py, then proves that the old full-state
checkpoint was mapped completely while its stage step and scheduler were ignored.

Unlike the v2 gate, v3 independently binds every boot-critical Python source to
the SHA-256 of the raw blob bytes at the externally supplied final Git commit.
The working-tree bytes must also be identical to that blob.  This prevents a
clean commit check plus a few ad-hoc file hashes from being mistaken for exact
source closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist


FINAL_CORE_GIT_BLOB_SOURCE_ARGS = {
    "train.py": "expected_train_py_git_blob_raw_sha256",
    "training/checkpoint.py": (
        "expected_training_checkpoint_py_git_blob_raw_sha256"
    ),
    "training/rng_state.py": "expected_training_rng_state_py_git_blob_raw_sha256",
    "training/formal_boot.py": (
        "expected_training_formal_boot_py_git_blob_raw_sha256"
    ),
    "training/config_schema.py": (
        "expected_training_config_schema_py_git_blob_raw_sha256"
    ),
    "networks/model_mot.py": (
        "expected_networks_model_mot_py_git_blob_raw_sha256"
    ),
    "networks/vae.py": "expected_networks_vae_py_git_blob_raw_sha256",
}
EXPECTED_FINAL_GIT_BLOB_ENTRY_COUNT = 11

# Diagnostic compatibility evidence only; this is not formal release authority.
# The sealed receipt records a successful real A-final load with
# ``weights_only=true`` and ``mmap=true`` under torch 2.11.0+cu128.  The formal
# smoke below still has to exercise and validate the final release itself.
A_FINAL_WEIGHTS_ONLY_DIAGNOSTIC_RECEIPT = (
    "/fsx/evan/audits/stage512a_continue_control_rngfix_v1/"
    "a_final_weights_only_probe_v1_20260809/RECEIPT.json"
)
A_FINAL_WEIGHTS_ONLY_DIAGNOSTIC_RECEIPT_SHA256 = (
    "f71538dd18b2bfc27cc23854d3692cbb05cfdfab1bb417305ebbb18399d2616e"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    raw = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require_sha256(value: str, *, label: str) -> str:
    if (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise RuntimeError(f"{label} must be one nonzero lowercase SHA-256")
    return value


def git_blob_raw_sha256(repo: Path, commit: str, relative_path: str) -> str:
    """Hash raw bytes of one blob addressed by ``commit:path``.

    ``git cat-file blob`` is used deliberately: hashing the checkout alone does
    not prove which committed blob supplied the tested runtime source.
    """

    object_spec = f"{commit}:{relative_path}"
    object_id = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--verify", object_spec],
        text=True,
    ).strip()
    object_type = subprocess.check_output(
        ["git", "-C", str(repo), "cat-file", "-t", object_id], text=True
    ).strip()
    if object_type != "blob":
        raise RuntimeError(
            f"final Git object is not a blob: {relative_path} -> "
            f"{object_id} ({object_type})"
        )
    raw_blob = subprocess.check_output(
        ["git", "-C", str(repo), "cat-file", "blob", object_id]
    )
    return hashlib.sha256(raw_blob).hexdigest()


def verify_final_git_blob_source_map(
    repo: Path,
    commit: str,
    expected: Mapping[str, str],
) -> dict[str, str]:
    core_paths = set(FINAL_CORE_GIT_BLOB_SOURCE_ARGS)
    if len(expected) != EXPECTED_FINAL_GIT_BLOB_ENTRY_COUNT or not core_paths.issubset(
        expected
    ):
        raise RuntimeError(
            "expected final Git-blob source-map closure mismatch: "
            f"expected_count={EXPECTED_FINAL_GIT_BLOB_ENTRY_COUNT}, "
            f"actual_count={len(expected)}, "
            f"missing_core={sorted(core_paths - set(expected))}"
        )

    verified: dict[str, str] = {}
    for relative_path in sorted(expected):
        expected_sha256 = require_sha256(
            expected[relative_path],
            label=f"expected final Git blob {relative_path}",
        )
        source_path = repo / relative_path
        if not source_path.is_file() or source_path.is_symlink():
            raise RuntimeError(
                f"runtime source must be an ordinary non-symlink file: {source_path}"
            )
        blob_sha256 = git_blob_raw_sha256(repo, commit, relative_path)
        if blob_sha256 != expected_sha256:
            raise RuntimeError(
                f"final Git blob raw SHA drift for {relative_path}: "
                f"{blob_sha256} != {expected_sha256}"
            )
        worktree_sha256 = sha256_file(source_path)
        if worktree_sha256 != blob_sha256:
            raise RuntimeError(
                f"runtime checkout bytes differ from final Git blob for {relative_path}: "
                f"worktree={worktree_sha256}, blob={blob_sha256}"
            )
        verified[relative_path] = blob_sha256
    return verified


def repo_relative_asset(repo: Path, path: Path, *, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} must be an ordinary non-symlink file: {path}")
    try:
        relative = path.relative_to(repo)
    except ValueError as exc:
        raise RuntimeError(f"{label} must be inside final release repo: {path}") from exc
    rendered = relative.as_posix()
    if not rendered or rendered == "." or rendered.startswith("../"):
        raise RuntimeError(f"{label} has invalid release-relative path: {rendered!r}")
    return rendered


def tensor_local(value: torch.Tensor) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def scalar_number(value: Any) -> int | float:
    if torch.is_tensor(value):
        local = tensor_local(value.detach())
        if local.numel() != 1:
            raise RuntimeError(
                f"optimizer step must have one element, got shape={tuple(local.shape)}"
            )
        result = local.reshape(-1)[0].cpu().item()
    else:
        result = value
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise RuntimeError(f"optimizer step must be numeric scalar, got {result!r}")
    if not math.isfinite(float(result)):
        raise RuntimeError(f"optimizer step must be finite, got {result!r}")
    return result


def byte_state_sha256(value: torch.Tensor) -> str:
    state = value.detach().cpu().contiguous()
    if state.dtype != torch.uint8 or state.ndim != 1:
        raise RuntimeError(
            f"generator state must be 1-D uint8, got {state.dtype} {tuple(state.shape)}"
        )
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()


def gather_digest(local_digest: str, world_size: int) -> list[str]:
    gathered: list[str | None] = [None] * world_size
    dist.all_gather_object(gathered, local_digest)
    if any(not isinstance(value, str) for value in gathered):
        raise RuntimeError(f"distributed digest gather is malformed: {gathered}")
    return [str(value) for value in gathered]


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return repr(value)


def group_hparams(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    return [
        {key: jsonable(value) for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]


def select_summary_keys(named: Mapping[str, torch.nn.Parameter]) -> list[str]:
    by_size = sorted(
        ((int(param.numel()), name) for name, param in named.items()),
        key=lambda item: (item[0], item[1]),
    )
    if len(by_size) < 3:
        raise RuntimeError(f"too few trainable tensors for summary: {len(by_size)}")
    picks = (by_size[0], by_size[len(by_size) // 2], by_size[-1])
    return [name for _, name in picks]


def expected_local(full: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor, distribute_tensor

    if isinstance(target, DTensor):
        distributed = distribute_tensor(
            full.to(target.device), target.device_mesh, target.placements
        )
        return distributed.to_local().detach()
    return full.to(target.device).detach()


def exact_tensor_report(
    loaded: torch.Tensor,
    expected_full: torch.Tensor,
    *,
    label: str,
) -> dict[str, Any]:
    if not torch.is_tensor(loaded) or not torch.is_tensor(expected_full):
        raise RuntimeError(f"{label} must compare two tensors")
    if tuple(loaded.shape) != tuple(expected_full.shape):
        raise RuntimeError(
            f"{label} global shape mismatch: loaded={tuple(loaded.shape)}, "
            f"checkpoint={tuple(expected_full.shape)}"
        )
    if loaded.dtype != expected_full.dtype:
        raise RuntimeError(
            f"{label} dtype mismatch: loaded={loaded.dtype}, checkpoint={expected_full.dtype}"
        )
    actual = tensor_local(loaded.detach())
    expected = expected_local(expected_full, loaded)
    equal = bool(torch.equal(actual, expected))
    if not equal:
        max_abs = float((actual.float() - expected.float()).abs().max().item())
        raise RuntimeError(f"{label} tensor mismatch: max_abs={max_abs}")
    return {
        "global_shape": list(loaded.shape),
        "local_shape": list(actual.shape),
        "dtype": str(loaded.dtype),
        "exact": True,
        "max_abs": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validator", required=True)
    parser.add_argument("--job-template", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--original-checkpoint", required=True)
    parser.add_argument("--phase-a-manifest", required=True)
    parser.add_argument("--canonical-training-lineage", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-train-py-git-blob-raw-sha256", required=True)
    parser.add_argument(
        "--expected-training-checkpoint-py-git-blob-raw-sha256", required=True
    )
    parser.add_argument(
        "--expected-training-rng-state-py-git-blob-raw-sha256", required=True
    )
    parser.add_argument(
        "--expected-training-formal-boot-py-git-blob-raw-sha256", required=True
    )
    parser.add_argument(
        "--expected-training-config-schema-py-git-blob-raw-sha256", required=True
    )
    parser.add_argument(
        "--expected-networks-model-mot-py-git-blob-raw-sha256", required=True
    )
    parser.add_argument(
        "--expected-networks-vae-py-git-blob-raw-sha256", required=True
    )
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-validator-sha256", required=True)
    parser.add_argument("--expected-job-template-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-ablation-fingerprint-sha256", required=True)
    parser.add_argument("--expected-phase-a-manifest-sha256", required=True)
    parser.add_argument("--expected-canonical-training-lineage-sha256", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-optimizer-step", type=int, required=True)
    parser.add_argument("--expected-trainable-count", type=int, required=True)
    parser.add_argument("--expected-keyset-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("must run under torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"this gate is frozen to world_size=2, got {world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    repo = Path(args.repo).resolve()
    config_path = Path(args.config).resolve()
    probe_path = Path(__file__).resolve()
    validator_path = Path(args.validator).resolve()
    job_template_path = Path(args.job_template).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    original_checkpoint_path = Path(args.original_checkpoint).resolve()
    phase_a_manifest_path = Path(args.phase_a_manifest).resolve()
    canonical_training_lineage_path = Path(
        args.canonical_training_lineage
    ).resolve()
    output = Path(args.output).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    git_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if git_head != args.expected_commit:
        raise RuntimeError(f"repo commit drift: {git_head} != {args.expected_commit}")
    git_status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).splitlines()
    if git_status:
        raise RuntimeError(f"final-code smoke requires a clean worktree: {git_status[:12]}")
    expected_git_blob_source_map = {
        relative_path: getattr(args, argument_attribute)
        for relative_path, argument_attribute in FINAL_CORE_GIT_BLOB_SOURCE_ARGS.items()
    }
    tracked_assets = {
        repo_relative_asset(repo, config_path, label="config"): (
            args.expected_config_sha256
        ),
        repo_relative_asset(repo, probe_path, label="probe"): (
            args.expected_probe_sha256
        ),
        repo_relative_asset(repo, validator_path, label="validator"): (
            args.expected_validator_sha256
        ),
        repo_relative_asset(repo, job_template_path, label="job template"): (
            args.expected_job_template_sha256
        ),
    }
    if len(tracked_assets) != 4 or set(tracked_assets).intersection(
        expected_git_blob_source_map
    ):
        raise RuntimeError(
            "tracked config/probe/validator/template paths must be four distinct "
            "non-core final Git blobs"
        )
    expected_git_blob_source_map.update(tracked_assets)
    tested_git_blob_source_map = verify_final_git_blob_source_map(
        repo, git_head, expected_git_blob_source_map
    )
    config_sha256 = sha256_file(config_path)
    if config_sha256 != args.expected_config_sha256:
        raise RuntimeError(
            f"config SHA drift: {config_sha256} != {args.expected_config_sha256}"
        )
    if (
        not phase_a_manifest_path.is_file()
        or phase_a_manifest_path.is_symlink()
    ):
        raise RuntimeError(
            f"Phase A manifest must be an ordinary file: {phase_a_manifest_path}"
        )
    phase_a_manifest_sha256 = sha256_file(phase_a_manifest_path)
    if phase_a_manifest_sha256 != args.expected_phase_a_manifest_sha256:
        raise RuntimeError(
            "Phase A manifest SHA drift: "
            f"{phase_a_manifest_sha256} != {args.expected_phase_a_manifest_sha256}"
        )
    if (
        not canonical_training_lineage_path.is_file()
        or canonical_training_lineage_path.is_symlink()
    ):
        raise RuntimeError(
            "canonical training lineage must be an ordinary file: "
            f"{canonical_training_lineage_path}"
        )
    canonical_training_lineage_sha256 = sha256_file(
        canonical_training_lineage_path
    )
    if (
        canonical_training_lineage_sha256
        != args.expected_canonical_training_lineage_sha256
    ):
        raise RuntimeError(
            "canonical training lineage SHA drift: "
            f"{canonical_training_lineage_sha256} != "
            f"{args.expected_canonical_training_lineage_sha256}"
        )
    verified_checkpoint_sha256 = os.environ.get("VERIFIED_CHECKPOINT_SHA256")
    if verified_checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError(
            "launcher did not pass the exact full-file checkpoint digest receipt: "
            f"{verified_checkpoint_sha256!r} != {args.expected_checkpoint_sha256}"
        )
    verified_asset_receipt_sha256 = os.environ.get("VERIFIED_ASSET_RECEIPT_SHA256")
    if not verified_asset_receipt_sha256:
        raise RuntimeError("launcher did not bind the A-final asset reseal receipt")
    verified_checkpoint_device_inode = os.environ.get(
        "VERIFIED_CHECKPOINT_DEVICE_INODE"
    )
    verified_checkpoint_bytes = os.environ.get("VERIFIED_CHECKPOINT_BYTES")
    verified_checkpoint_mode = os.environ.get("VERIFIED_CHECKPOINT_MODE")
    verified_samefile = os.environ.get("VERIFIED_CHECKPOINT_SAMEFILE_WITH_ORIGINAL")
    verified_phase_a_manifest_sha256 = os.environ.get(
        "VERIFIED_PHASE_A_MANIFEST_SHA256"
    )
    verified_lineage_sha256 = os.environ.get(
        "VERIFIED_CANONICAL_TRAINING_LINEAGE_SHA256"
    )
    if (
        not checkpoint_path.is_file()
        or checkpoint_path.is_symlink()
        or not original_checkpoint_path.is_file()
        or original_checkpoint_path.is_symlink()
        or not os.path.samefile(checkpoint_path, original_checkpoint_path)
    ):
        raise RuntimeError(
            "A-final alias/original ordinary-hardlink identity check failed: "
            f"alias={checkpoint_path}, original={original_checkpoint_path}"
        )
    if (
        not verified_checkpoint_device_inode
        or verified_checkpoint_bytes != "26328353144"
        or verified_checkpoint_mode != "444"
        or verified_samefile != "1"
        or verified_phase_a_manifest_sha256 != phase_a_manifest_sha256
        or verified_lineage_sha256 != canonical_training_lineage_sha256
    ):
        raise RuntimeError(
            "launcher did not pass the frozen hardlink identity receipt: "
            f"device_inode={verified_checkpoint_device_inode!r}, "
            f"bytes={verified_checkpoint_bytes!r}, mode={verified_checkpoint_mode!r}, "
            f"samefile={verified_samefile!r}, "
            f"phase_a={verified_phase_a_manifest_sha256!r}, "
            f"lineage={verified_lineage_sha256!r}"
        )

    from omegaconf import OmegaConf
    from networks.model_mot import MoTT2IModel
    from training.checkpoint import (
        _ablation_fingerprint_check,
        _ablation_mismatch_check,
        _ema_state_integrity_check,
        _optimizer_state_integrity_check,
        build_scheduler,
        derive_max_steps,
        init_from_checkpoint,
    )
    from training.config_schema import validate_config
    from training.formal_boot import resolve_checkpoint_boot_branch
    from training.rng_state import verify_initial_vae_rng_uniqueness

    started = time.time()
    cfg = OmegaConf.load(config_path)
    config_schema_warnings = validate_config(cfg, log=lambda _message: None)
    if config_schema_warnings:
        raise RuntimeError(
            "final A-control config has schema warnings: "
            f"{config_schema_warnings[:20]}"
        )
    if str(cfg.model_cls) != "networks.model_mot.MoTT2IModel":
        raise RuntimeError(f"model_cls drift: {cfg.model_cls}")
    if str(cfg.target_device) != "cuda":
        raise RuntimeError(f"target_device drift: {cfg.target_device}")
    if str(cfg.trainer.amp_dtype) != "bfloat16":
        raise RuntimeError(f"amp_dtype drift: {cfg.trainer.amp_dtype}")
    if not bool(cfg.trainer.ema.enabled):
        raise RuntimeError("EMA must be enabled for the A-control arm")
    if int(cfg.trainer.batch_size) != 32:
        raise RuntimeError(f"per-rank batch drift: {cfg.trainer.batch_size}")
    if int(cfg.trainer.max_steps) != 125000:
        raise RuntimeError(f"max_steps drift: {cfg.trainer.max_steps}")
    if int(cfg.trainer.max_exposure_images) != 256000000:
        raise RuntimeError(
            f"max_exposure_images drift: {cfg.trainer.max_exposure_images}"
        )
    derived_w64_steps = derive_max_steps(
        cfg, batch_size=int(cfg.trainer.batch_size), world_size=64
    )
    if derived_w64_steps != 125000:
        raise RuntimeError(f"W64 exposure-derived max_steps drift: {derived_w64_steps}")
    checkpoint_dir = Path(str(cfg.trainer.checkpoint_dir)).resolve()
    if os.path.lexists(checkpoint_dir):
        raise RuntimeError(
            "A-control first-launch checkpoint_dir must be create-once absent: "
            f"{checkpoint_dir}"
        )
    cfg.target_device = "cuda"
    cfg.vae = OmegaConf.create(
        {"name": "fake", "downsample_factor": 16, "latent_channels": 128, "seed": 0}
    )
    # These values are the frozen A-control scientific contract.
    if int(cfg.scheduler.num_warmup_steps) != 500:
        raise RuntimeError(f"warmup drift: {cfg.scheduler.num_warmup_steps}")
    if str(cfg.scheduler.decay) != "constant":
        raise RuntimeError(f"scheduler decay drift: {cfg.scheduler.decay}")
    if not bool(cfg.trainer.init_load_optimizer):
        raise RuntimeError("init_load_optimizer must be true")
    if not bool(cfg.trainer.resume):
        raise RuntimeError("resume must be true for retry safety")
    if str(cfg.trainer.get("init_from_rng_policy", "")) != "legacy_missing_rng":
        raise RuntimeError("A-final must use the explicit legacy_missing_rng init policy")
    if Path(str(cfg.trainer.init_from)).resolve() != checkpoint_path:
        raise RuntimeError(
            f"config init_from is not the frozen alias: {cfg.trainer.init_from} != {checkpoint_path}"
        )

    # mmap metadata view is independent of the loader exercised below.  It lets us
    # prove exact key/state closure and compare every loaded value without retaining
    # a second 26 GB in-memory checkpoint copy.
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("checkpoint root is not a mapping")
    expected_legacy_top = {
        "trainable_state", "optimizer", "ema", "scheduler", "global_step", "ablation_fp"
    }
    if set(checkpoint) != expected_legacy_top:
        raise RuntimeError(
            f"A-final legacy top-level schema drift: {sorted(checkpoint)}"
        )
    if int(checkpoint.get("global_step", -1)) != args.expected_step:
        raise RuntimeError(f"global_step drift: {checkpoint.get('global_step')}")
    raw = checkpoint.get("trainable_state")
    ema = checkpoint.get("ema")
    old_optimizer = checkpoint.get("optimizer")
    old_scheduler = checkpoint.get("scheduler")
    if not all(isinstance(item, Mapping) for item in (raw, ema, old_optimizer, old_scheduler)):
        raise RuntimeError("checkpoint lacks full trainable/EMA/optimizer/scheduler payload")
    checkpoint_keys = set(raw)
    keyset_hash = hashlib.sha256(
        ("\n".join(sorted(str(key) for key in checkpoint_keys)) + "\n").encode()
    ).hexdigest()
    if len(checkpoint_keys) != args.expected_trainable_count:
        raise RuntimeError(f"checkpoint trainable count drift: {len(checkpoint_keys)}")
    if keyset_hash != args.expected_keyset_sha256:
        raise RuntimeError(f"checkpoint keyset SHA drift: {keyset_hash}")
    if set(ema) != checkpoint_keys:
        raise RuntimeError("EMA keyset is not exactly the trainable keyset")
    old_state = old_optimizer.get("state", {})
    old_groups = old_optimizer.get("param_groups", [])
    if set(old_state) != checkpoint_keys:
        raise RuntimeError("checkpoint optimizer.state keyset is not exactly trainable keyset")
    if len(old_groups) != 1 or set(old_groups[0].get("params", ())) != checkpoint_keys:
        raise RuntimeError("checkpoint optimizer param_group closure failed")
    if int(old_scheduler.get("last_epoch", -1)) != args.expected_step:
        raise RuntimeError("authority checkpoint scheduler is not at its global step")
    _ablation_fingerprint_check(
        checkpoint.get("ablation_fp"), cfg, require_exact=True
    )
    ablation_fingerprint_sha256 = canonical_json_sha256(checkpoint["ablation_fp"])
    expected_ablation_fingerprint_sha256 = require_sha256(
        args.expected_ablation_fingerprint_sha256,
        label="expected A-final ablation fingerprint",
    )
    if ablation_fingerprint_sha256 != expected_ablation_fingerprint_sha256:
        raise RuntimeError(
            "A-final ablation fingerprint SHA drift: "
            f"{ablation_fingerprint_sha256} != "
            f"{expected_ablation_fingerprint_sha256}"
        )

    torch.manual_seed(int(cfg.trainer.seed))
    model = MoTT2IModel(config=cfg).to(device)
    model.configure_training_rng(global_rank=rank, device=device)
    vae_rng_unique = verify_initial_vae_rng_uniqueness(
        model, rank=rank, is_dist=True
    )
    # This arm exists specifically to remove the historical cross-rank VAE
    # posterior correlation.  The generic helper follows the model contract and
    # would legitimately accept one shared state when rank_independent_seed=true;
    # that configuration is forbidden here, so bind the scientific requirement
    # directly instead of merely trusting the configured expectation.
    if vae_rng_unique != world_size:
        raise RuntimeError(
            "A-control requires one unique VAE RNG state per global rank: "
            f"unique={vae_rng_unique}, world_size={world_size}"
        )
    from torch.distributed.fsdp import MixedPrecisionPolicy

    model.fully_shard(
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
    )
    named = {name: param for name, param in model.model.named_parameters() if param.requires_grad}
    current_keys = set(named)
    if current_keys != checkpoint_keys:
        raise RuntimeError(
            f"full trainable key closure failed: current_only={sorted(current_keys-checkpoint_keys)[:8]} "
            f"checkpoint_only={sorted(checkpoint_keys-current_keys)[:8]}"
        )

    opt_params = OmegaConf.to_container(cfg.optimizer.params, resolve=True)
    weight_decay = float(opt_params.pop("weight_decay", 0.0))
    base_lr = float(opt_params.get("lr", 0.0))
    if base_lr != 1.732051e-4:
        raise RuntimeError(f"A-control base LR drift: {base_lr}")
    optimizer = torch.optim.AdamW(
        model.get_optimizer_param_groups(weight_decay=weight_decay, base_lr=base_lr),
        **opt_params,
    )
    scheduler = build_scheduler(optimizer, cfg, max_steps=125000)
    model.init_ema(float(cfg.trainer.ema.decay))

    # Negative controls run on the real A-final FQN dictionaries but never mutate
    # the checkpoint mmap view.  These prove the exact gates that strict=False
    # previously bypassed, before the positive state load begins.
    negative_controls: dict[str, str] = {}

    def expect_failure(label: str, needle: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - the gate must fail closed on any error.
            rendered = f"{type(exc).__name__}: {exc}"
            if needle not in rendered:
                raise RuntimeError(
                    f"negative control {label} failed for the wrong reason: {rendered}"
                ) from exc
            negative_controls[label] = rendered
            return
        raise RuntimeError(f"negative control {label} unexpectedly passed")

    old_ablation_tokens = ("_moe_gen", "time_embed", "lora_und")
    ordinary_key = next(
        name for name in sorted(current_keys)
        if not any(token in name for token in old_ablation_tokens)
    )
    expect_failure(
        "ordinary_current_only_trainable",
        "trainable-key closure mismatch",
        lambda: _ablation_mismatch_check(checkpoint_keys - {ordinary_key}, model),
    )
    missing_state = dict(old_state)
    missing_state.pop(ordinary_key)
    bad_optimizer_missing = dict(old_optimizer)
    bad_optimizer_missing["state"] = missing_state
    expect_failure(
        "optimizer_missing_fqn_state",
        "state coverage mismatch",
        lambda: _optimizer_state_integrity_check(
            bad_optimizer_missing, raw, model, optimizer
        ),
    )
    bad_optimizer_groups = dict(old_optimizer)
    bad_optimizer_groups["param_groups"] = list(old_groups) + [dict(old_groups[0])]
    expect_failure(
        "optimizer_extra_param_group",
        "param-group count mismatch",
        lambda: _optimizer_state_integrity_check(
            bad_optimizer_groups, raw, model, optimizer
        ),
    )
    missing_ema = dict(ema)
    missing_ema.pop(ordinary_key)
    expect_failure(
        "ema_missing_fqn",
        "EMA key closure mismatch",
        lambda: _ema_state_integrity_check(missing_ema, raw, model),
    )

    fresh_scheduler_state = scheduler.state_dict()
    fresh_hparams = group_hparams(optimizer)
    fresh_lr = list(scheduler.get_last_lr())
    fresh_training_rng_states = {
        name: generator.get_state().detach().cpu().clone()
        for name, generator in model.training_rng_generators().items()
    }
    pre_init_vae_rng_state_sha256_by_rank = gather_digest(
        byte_state_sha256(fresh_training_rng_states["vae"]), world_size
    )
    if len(set(pre_init_vae_rng_state_sha256_by_rank)) != world_size:
        raise RuntimeError(
            "pre-init VAE RNG digest evidence is not rank-specific: "
            f"{pre_init_vae_rng_state_sha256_by_rank}"
        )
    expected_fresh_lr = base_lr / int(cfg.scheduler.num_warmup_steps)
    if len(fresh_lr) != 1 or abs(float(fresh_lr[0]) - expected_fresh_lr) > 1e-15:
        raise RuntimeError(f"fresh scheduler LR mismatch: {fresh_lr} vs {expected_fresh_lr}")
    if (
        int(fresh_scheduler_state.get("last_epoch", -1)) != 0
        or int(fresh_scheduler_state.get("_step_count", -1)) != 1
    ):
        raise RuntimeError(
            f"fresh scheduler origin drift: {fresh_scheduler_state}"
        )
    expected_step1_logged_lr = base_lr * 2 / int(cfg.scheduler.num_warmup_steps)
    actual_step1_logged_lr = base_lr * float(scheduler.lr_lambdas[0](1))
    if abs(actual_step1_logged_lr - expected_step1_logged_lr) > 1e-15:
        raise RuntimeError(
            "step-1 scheduler LR drift: "
            f"actual={actual_step1_logged_lr}, expected={expected_step1_logged_lr}"
        )
    if optimizer.state:
        raise RuntimeError("fresh AdamW unexpectedly has materialized state before init_from")

    init_rng_policy = init_from_checkpoint(
        str(checkpoint_path), model, optimizer, True, cfg, load_optimizer=True
    )
    if init_rng_policy != "legacy_missing_rng":
        raise RuntimeError(f"wrong init RNG policy receipt: {init_rng_policy!r}")
    if scheduler.state_dict() != fresh_scheduler_state:
        raise RuntimeError("init_from loaded or mutated the old scheduler")
    if group_hparams(optimizer) != fresh_hparams:
        raise RuntimeError("init_from leaked old optimizer group hyperparameters")
    if list(scheduler.get_last_lr()) != fresh_lr:
        raise RuntimeError("fresh scheduler LR changed during init_from")
    post_init_training_rng_states = {
        name: generator.get_state().detach().cpu().clone()
        for name, generator in model.training_rng_generators().items()
    }
    post_init_vae_rng_state_sha256_by_rank = gather_digest(
        byte_state_sha256(post_init_training_rng_states["vae"]), world_size
    )
    if set(post_init_training_rng_states) != set(fresh_training_rng_states):
        raise RuntimeError("training RNG stream set changed during init_from")
    changed_rng_streams = sorted(
        name
        for name in fresh_training_rng_states
        if not torch.equal(
            post_init_training_rng_states[name], fresh_training_rng_states[name]
        )
    )
    if changed_rng_streams:
        raise RuntimeError(
            "legacy init_from consumed or restored fresh training RNG streams: "
            f"{changed_rng_streams}"
        )
    if (
        post_init_vae_rng_state_sha256_by_rank
        != pre_init_vae_rng_state_sha256_by_rank
    ):
        raise RuntimeError(
            "per-rank VAE RNG digest map changed during init_from: "
            f"pre={pre_init_vae_rng_state_sha256_by_rank}, "
            f"post={post_init_vae_rng_state_sha256_by_rank}"
        )
    post_init_vae_rng_unique = verify_initial_vae_rng_uniqueness(
        model, rank=rank, is_dist=True
    )
    if post_init_vae_rng_unique != world_size:
        raise RuntimeError(
            "A-control lost per-rank VAE RNG uniqueness during init_from: "
            f"unique={post_init_vae_rng_unique}, world_size={world_size}"
        )

    group_params = [param for group in optimizer.param_groups for param in group["params"]]
    if len(group_params) != len(named) or set(group_params) != set(named.values()):
        raise RuntimeError("runtime optimizer param_groups do not exactly cover current trainables")
    if set(optimizer.state) != set(named.values()):
        raise RuntimeError(
            f"runtime optimizer.state closure failed: state={len(optimizer.state)} model={len(named)}"
        )

    shape_errors = []
    state_key_errors = []
    step_values = set()
    for name, param in named.items():
        state = optimizer.state[param]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            state_key_errors.append((name, sorted(state)))
            continue
        step_values.add(scalar_number(state["step"]))
        for state_name in ("exp_avg", "exp_avg_sq"):
            value = state[state_name]
            if tuple(value.shape) != tuple(param.shape):
                shape_errors.append((name, state_name, tuple(value.shape), tuple(param.shape)))
            if value.dtype != torch.float32:
                shape_errors.append((name, state_name, str(value.dtype), "torch.float32"))
    if state_key_errors:
        raise RuntimeError(f"Adam state schema errors: {state_key_errors[:4]}")
    if shape_errors:
        raise RuntimeError(f"Adam state shape/dtype errors: {shape_errors[:4]}")
    old_steps = {scalar_number(record["step"]) for record in old_state.values()}
    expected_optimizer_steps = {args.expected_optimizer_step}
    if step_values != old_steps or old_steps != expected_optimizer_steps:
        raise RuntimeError(
            "Adam step mismatch: "
            f"loaded={step_values}, checkpoint={old_steps}, "
            f"expected={expected_optimizer_steps}"
        )

    # The authority claim is numerical, not just structural: compare every one
    # of the 483 loaded raw tensors, EMA tensors, and both Adam moments on each
    # rank's actual FSDP2 shard.  Keep only three size-stratified details in the
    # report; the aggregate counters are incremented only after exact equality.
    summary_keys = set(select_summary_keys(named))
    numerical_compare_counts = {
        "raw": 0,
        "ema": 0,
        "adam_step": 0,
        "adam_exp_avg": 0,
        "adam_exp_avg_sq": 0,
    }
    numerical_summary: dict[str, Any] = {}
    for name in sorted(named):
        param = named[name]
        state = optimizer.state[param]
        checks = {
            "raw": exact_tensor_report(
                param, raw[name], label=f"{name}:parameter"
            ),
            "ema": exact_tensor_report(
                model._ema[name], ema[name], label=f"{name}:ema"
            ),
            "adam_step": exact_tensor_report(
                state["step"], old_state[name]["step"], label=f"{name}:step"
            ),
            "adam_exp_avg": exact_tensor_report(
                state["exp_avg"],
                old_state[name]["exp_avg"],
                label=f"{name}:exp_avg",
            ),
            "adam_exp_avg_sq": exact_tensor_report(
                state["exp_avg_sq"],
                old_state[name]["exp_avg_sq"],
                label=f"{name}:exp_avg_sq",
            ),
        }
        for state_name in numerical_compare_counts:
            numerical_compare_counts[state_name] += int(checks[state_name]["exact"])
        if name in summary_keys:
            numerical_summary[name] = {
                **checks,
                "adam_step_value": scalar_number(state["step"]),
            }
    expected_compare_counts = {
        state_name: args.expected_trainable_count
        for state_name in numerical_compare_counts
    }
    if numerical_compare_counts != expected_compare_counts:
        raise RuntimeError(
            "full numerical comparison count mismatch: "
            f"actual={numerical_compare_counts}, expected={expected_compare_counts}"
        )

    # Exercise the exact distributed branch resolver used by train.py against
    # the actual formal checkpoint path.  The W64 runtime gate remains
    # authoritative against a later race after this smoke completes.
    if os.path.lexists(checkpoint_dir):
        raise RuntimeError(
            f"A-control checkpoint_dir appeared during the gate: {checkpoint_dir}"
        )
    local_last = checkpoint_dir / "last.pt"
    local_last_exists = os.path.lexists(local_last)
    resolved_train_branch = resolve_checkpoint_boot_branch(
        str(local_last),
        resume=bool(cfg.trainer.resume),
        init_from=str(cfg.trainer.init_from) or None,
        expected_mode="fresh",
        is_dist=True,
    )
    if resolved_train_branch != "fresh":
        raise RuntimeError(
            f"real first-launch branch resolver drift: {resolved_train_branch}"
        )

    local_ok = torch.tensor([1], device=device, dtype=torch.int32)
    dist.all_reduce(local_ok, op=dist.ReduceOp.SUM)
    if int(local_ok.item()) != world_size:
        raise RuntimeError("not all ranks completed the gate")

    if rank == 0:
        final_git_status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        ).splitlines()
        if final_git_status:
            raise RuntimeError(
                "final-code worktree changed during smoke: "
                f"{final_git_status[:12]}"
            )
        final_git_blob_source_map = verify_final_git_blob_source_map(
            repo, git_head, expected_git_blob_source_map
        )
        if final_git_blob_source_map != tested_git_blob_source_map:
            raise RuntimeError(
                "final Git-blob source map changed during smoke: "
                f"before={tested_git_blob_source_map}, "
                f"after={final_git_blob_source_map}"
            )
        # Publish the receipt atomically.  A killed probe may leave an explicitly
        # named attempt directory, but can never make the authority OUTPUT look
        # complete without its matching _SUCCESS digest.
        attempt_output = output.with_name(f".{output.name}.attempt")
        if attempt_output.exists() or attempt_output.is_symlink():
            raise RuntimeError(f"stale create-once attempt exists: {attempt_output}")
        attempt_output.mkdir(parents=False, exist_ok=False)
        probe_sha256 = sha256_file(probe_path)
        validator_sha256 = sha256_file(validator_path)
        job_template_sha256 = sha256_file(job_template_path)
        report = {
            "schema": "stage512a_control_a_final_load_smoke_v3",
            "status": "PASS",
            "verdict": "PASS",
            "authority": {
                "release_repo": str(repo),
                "final_commit": git_head,
                "git_status_porcelain": git_status,
                "config_path": str(config_path),
                "config_sha256": config_sha256,
                "probe_path": str(probe_path),
                "probe_sha256": probe_sha256,
                "validator_path": str(validator_path),
                "validator_sha256": validator_sha256,
                "job_template_path": str(job_template_path),
                "job_template_sha256": job_template_sha256,
                "phase_a_manifest_path": str(phase_a_manifest_path),
                "phase_a_manifest_sha256": phase_a_manifest_sha256,
                "canonical_training_lineage_path": str(
                    canonical_training_lineage_path
                ),
                "canonical_training_lineage_sha256": (
                    canonical_training_lineage_sha256
                ),
                "canonical_training_lineage_claimed": True,
            },
            "tested_final_git_blob_raw_sha256": final_git_blob_source_map,
            "checkpoint": {
                "alias_path": str(checkpoint_path),
                "original_path": str(original_checkpoint_path),
                "full_sha256": verified_checkpoint_sha256,
                "asset_reseal_report_sha256": verified_asset_receipt_sha256,
                "device_inode": verified_checkpoint_device_inode,
                "bytes": int(verified_checkpoint_bytes),
                "mode": f"0{verified_checkpoint_mode}",
                "samefile_with_original": True,
                "source_global_step": int(checkpoint["global_step"]),
                "source_scheduler_last_epoch": int(old_scheduler["last_epoch"]),
                "trainable_keyset_sha256": keyset_hash,
                "ablation_fingerprint_sha256": ablation_fingerprint_sha256,
                "legacy_top_level_keys": sorted(expected_legacy_top),
            },
            "world_size": world_size,
            "boot": {
                "mode": "fresh",
                "stage_step": 0,
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_dir_absent": True,
                "local_last_absent": not local_last_exists,
                "resolved_train_branch": resolved_train_branch,
                "branch_resolver": (
                    "training.formal_boot.resolve_checkpoint_boot_branch"
                ),
                "config_schema_warning_count": len(config_schema_warnings),
            },
            "closure": {
                "raw_count": len(raw),
                "ema_count": len(model._ema),
                "optimizer_count": len(optimizer.state),
                "optimizer_param_group_count": len(optimizer.param_groups),
                "adam_step_values": [args.expected_optimizer_step],
                "exact_per_rank": numerical_compare_counts,
                "ranks_completed": int(local_ok.item()),
                "size_stratified_details": numerical_summary,
            },
            "scheduler": {
                "source_ignored": True,
                "state_unchanged_by_init": True,
                "warmup_steps": int(cfg.scheduler.num_warmup_steps),
                "first_optimizer_update_lr": fresh_lr,
                "step1_logged_lr": [actual_step1_logged_lr],
                "fresh_state": jsonable(fresh_scheduler_state),
                "optimizer_hparams_preserved": True,
                "optimizer_group_hparams": fresh_hparams,
            },
            "rng": {
                "init_from_policy": init_rng_policy,
                "source_rng_restored": False,
                "all_stream_states_unchanged_by_init": True,
                "pre_vae_unique": vae_rng_unique,
                "post_vae_unique": post_init_vae_rng_unique,
                "pre_vae_sha256_by_rank": pre_init_vae_rng_state_sha256_by_rank,
                "post_vae_sha256_by_rank": post_init_vae_rng_state_sha256_by_rank,
                "stream_state_sha256_rank0": {
                    name: byte_state_sha256(state)
                    for name, state in fresh_training_rng_states.items()
                },
            },
            "negative_controls": negative_controls,
            "elapsed_s": time.time() - started,
        }
        report_path = attempt_output / "REPORT.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_sha = sha256_file(report_path)
        success_path = attempt_output / "_SUCCESS"
        success_path.write_text(report_sha + "\n", encoding="ascii")
        os.chmod(report_path, 0o444)
        os.chmod(success_path, 0o444)
        os.chmod(attempt_output, 0o555)
        if output.exists() or output.is_symlink():
            raise RuntimeError(f"create-once output appeared during probe: {output}")
        os.rename(attempt_output, output)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
