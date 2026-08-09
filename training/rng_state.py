"""Strict per-rank training RNG capture/restore.

The training checkpoint is a continuation contract, not merely a weights file.
Every stochastic stream that can change the next optimizer update is therefore
stored for every global rank.  A checkpoint written by this module may only be
resumed with the same schema, world size, and RNG contract.

``init_from`` is deliberately different: it starts a new stage-local RNG
lineage.  A legacy source checkpoint may omit this schema only when the config
explicitly opts into ``trainer.init_from_rng_policy=legacy_missing_rng``.
That exception is never used by same-stage ``resume``.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


FULL_CHECKPOINT_SCHEMA = "mot_full_training_v2"
RNG_STATE_SCHEMA = "mot_rng_state_v1"
LEGACY_MISSING_RNG_POLICY = "legacy_missing_rng"
RNG_STREAM_NAMES = ("dropout", "diffusion", "vae")


class RNGCheckpointError(RuntimeError):
    """A checkpoint cannot provide an exact next stochastic update."""


def _raise_synchronized_error(error, *, is_dist: bool, context: str) -> None:
    if is_dist:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(
            gathered,
            None if error is None else f"{type(error).__name__}: {error}",
        )
        failures = [(rank, value) for rank, value in enumerate(gathered) if value is not None]
        if failures:
            message = f"{context}: {failures}"
            if error is not None:
                raise RNGCheckpointError(message) from error
            raise RNGCheckpointError(message)
    elif error is not None:
        raise error


def _as_device(device: torch.device | str) -> torch.device:
    out = torch.device(device)
    if out.type == "cuda" and out.index is None:
        out = torch.device("cuda", torch.cuda.current_device())
    if out.type not in ("cpu", "cuda"):
        raise RNGCheckpointError(f"unsupported training RNG device: {out}")
    return out


def _byte_state(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RNGCheckpointError(f"{label} must be a torch.Tensor")
    if value.dtype != torch.uint8 or value.ndim != 1 or value.numel() == 0:
        raise RNGCheckpointError(
            f"{label} must be a non-empty 1-D uint8 tensor; "
            f"got dtype={value.dtype}, shape={tuple(value.shape)}"
        )
    return value.detach().cpu().clone()


def _numpy_state_for_storage(state) -> tuple:
    return (
        state[0],
        torch.tensor(state[1].tolist(), dtype=torch.int64),
        int(state[2]),
        int(state[3]),
        float(state[4]),
    )


def _numpy_state_for_restore(state) -> tuple:
    return (
        state[0],
        np.asarray(state[1].cpu().tolist(), dtype=np.uint32),
        int(state[2]),
        int(state[3]),
        float(state[4]),
    )


def _model_generators(model) -> dict[str, torch.Generator]:
    getter = getattr(model, "training_rng_generators", None)
    if not callable(getter):
        raise RNGCheckpointError("model does not expose training_rng_generators()")
    generators = dict(getter())
    if set(generators) != set(RNG_STREAM_NAMES):
        raise RNGCheckpointError(
            "model RNG streams differ from checkpoint contract: "
            f"expected={list(RNG_STREAM_NAMES)}, got={sorted(generators)}"
        )
    for name, generator in generators.items():
        if not isinstance(generator, torch.Generator):
            raise RNGCheckpointError(f"model RNG stream {name!r} is not torch.Generator")
    return generators


def _model_contract(model) -> dict[str, Any]:
    getter = getattr(model, "training_rng_contract", None)
    if not callable(getter):
        raise RNGCheckpointError("model does not expose training_rng_contract()")
    contract = dict(getter())
    required = {
        "base_seed",
        "rank_independent_seed",
        "rank_stride",
        "stream_offsets",
    }
    if set(contract) != required:
        raise RNGCheckpointError(
            f"model RNG contract keys mismatch: expected={sorted(required)}, "
            f"got={sorted(contract)}"
        )
    offsets = contract.get("stream_offsets")
    if not isinstance(offsets, Mapping) or set(offsets) != set(RNG_STREAM_NAMES):
        raise RNGCheckpointError("model RNG stream_offsets do not cover the exact stream set")
    return {
        "base_seed": int(contract["base_seed"]),
        "rank_independent_seed": bool(contract["rank_independent_seed"]),
        "rank_stride": int(contract["rank_stride"]),
        "stream_offsets": {name: int(offsets[name]) for name in RNG_STREAM_NAMES},
    }


def _normalise_contract(contract: Any, label: str) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise RNGCheckpointError(f"{label} must be a mapping")
    required = {
        "base_seed",
        "rank_independent_seed",
        "rank_stride",
        "stream_offsets",
    }
    if set(contract) != required:
        raise RNGCheckpointError(
            f"{label} keys mismatch: expected={sorted(required)}, got={sorted(contract)}"
        )
    offsets = contract["stream_offsets"]
    if not isinstance(offsets, Mapping) or set(offsets) != set(RNG_STREAM_NAMES):
        raise RNGCheckpointError(f"{label}.stream_offsets has wrong stream set")
    if isinstance(contract["base_seed"], bool) or not isinstance(contract["base_seed"], int):
        raise RNGCheckpointError(f"{label}.base_seed must be int")
    if not isinstance(contract["rank_independent_seed"], bool):
        raise RNGCheckpointError(f"{label}.rank_independent_seed must be bool")
    if isinstance(contract["rank_stride"], bool) or not isinstance(contract["rank_stride"], int):
        raise RNGCheckpointError(f"{label}.rank_stride must be int")
    for name in RNG_STREAM_NAMES:
        if isinstance(offsets[name], bool) or not isinstance(offsets[name], int):
            raise RNGCheckpointError(f"{label}.stream_offsets[{name!r}] must be int")
    return {
        "base_seed": int(contract["base_seed"]),
        "rank_independent_seed": bool(contract["rank_independent_seed"]),
        "rank_stride": int(contract["rank_stride"]),
        "stream_offsets": {name: int(offsets[name]) for name in RNG_STREAM_NAMES},
    }


def _model_rng_device(model) -> torch.device:
    getter = getattr(model, "training_rng_device", None)
    if not callable(getter):
        raise RNGCheckpointError("model does not expose training_rng_device()")
    return _as_device(getter())


def capture_local_rng_state(model, rank: int, world_size: int) -> dict[str, Any]:
    """Capture one rank without advancing any RNG stream."""
    rank = int(rank)
    world_size = int(world_size)
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise RNGCheckpointError(f"invalid rank/world_size: rank={rank}, world_size={world_size}")

    device = _model_rng_device(model)
    generators = _model_generators(model)
    vae_device = _as_device(generators["vae"].device)
    if vae_device != device:
        raise RNGCheckpointError(
            f"VAE generator is not device-matched: generator={vae_device}, model_rng_device={device}"
        )
    for name in ("dropout", "diffusion"):
        if torch.device(generators[name].device).type != "cpu":
            raise RNGCheckpointError(
                f"{name} generator must remain CPU because its draws are created on CPU"
            )

    numpy_state = _numpy_state_for_storage(np.random.get_state())
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device).detach().cpu().clone()
    return {
        "rank": rank,
        "device_type": device.type,
        "python": random.getstate(),
        "numpy": numpy_state,
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda": cuda_state,
        "generators": {
            name: generators[name].get_state().detach().cpu().clone()
            for name in RNG_STREAM_NAMES
        },
    }


def _validate_local_state(
    record: Any,
    rank: int,
    device_type: str | None = None,
    validation_device: torch.device | None = None,
) -> None:
    if not isinstance(record, Mapping):
        raise RNGCheckpointError(f"rng_state.per_rank[{rank}] must be a mapping")
    expected = {
        "rank",
        "device_type",
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "generators",
    }
    if set(record) != expected:
        raise RNGCheckpointError(
            f"rng_state rank {rank} keys mismatch: expected={sorted(expected)}, "
            f"got={sorted(record)}"
        )
    if isinstance(record["rank"], bool) or not isinstance(record["rank"], int):
        raise RNGCheckpointError(f"rng_state rank {rank} embedded rank must be int")
    if record["rank"] != rank:
        raise RNGCheckpointError(
            f"rng_state rank slot mismatch: slot={rank}, embedded={record['rank']}"
        )
    saved_device_type = str(record["device_type"])
    if saved_device_type not in ("cpu", "cuda"):
        raise RNGCheckpointError(f"rng_state rank {rank} has invalid device_type={saved_device_type!r}")
    if device_type is not None and saved_device_type != str(device_type):
        raise RNGCheckpointError(
            f"rng_state device type mismatch at rank {rank}: "
            f"checkpoint={saved_device_type}, current={device_type}"
        )
    if not isinstance(record["python"], tuple):
        raise RNGCheckpointError(f"rng_state rank {rank} python state is malformed")
    numpy_state = record["numpy"]
    if not (
        isinstance(numpy_state, tuple)
        and len(numpy_state) == 5
        and isinstance(numpy_state[1], torch.Tensor)
        and numpy_state[1].dtype == torch.int64
        and numpy_state[1].ndim == 1
        and numpy_state[1].numel() > 0
    ):
        raise RNGCheckpointError(f"rng_state rank {rank} numpy state is malformed")
    _byte_state(record["torch_cpu"], f"rng_state rank {rank} torch_cpu")
    if saved_device_type == "cuda":
        _byte_state(record["torch_cuda"], f"rng_state rank {rank} torch_cuda")
    elif record["torch_cuda"] is not None:
        raise RNGCheckpointError(
            f"rng_state rank {rank} CPU record must use explicit torch_cuda=None"
        )
    generators = record["generators"]
    if not isinstance(generators, Mapping) or set(generators) != set(RNG_STREAM_NAMES):
        raise RNGCheckpointError(f"rng_state rank {rank} generator keys are malformed")
    for name in RNG_STREAM_NAMES:
        _byte_state(generators[name], f"rng_state rank {rank} generator {name}")

    # Validate opaque state payloads on isolated RNG objects.  This catches bad
    # lengths/internal versions before any live global state is mutated.
    try:
        random.Random().setstate(record["python"])
    except BaseException as exc:
        raise RNGCheckpointError(f"rng_state rank {rank} Python payload is invalid") from exc
    try:
        np.random.RandomState(0).set_state(_numpy_state_for_restore(record["numpy"]))
    except BaseException as exc:
        raise RNGCheckpointError(f"rng_state rank {rank} NumPy payload is invalid") from exc
    try:
        torch.Generator(device="cpu").set_state(record["torch_cpu"])
        torch.Generator(device="cpu").set_state(generators["dropout"])
        torch.Generator(device="cpu").set_state(generators["diffusion"])
        if saved_device_type == "cpu":
            torch.Generator(device="cpu").set_state(generators["vae"])
        elif validation_device is not None:
            exact_device = _as_device(validation_device)
            torch.Generator(device=exact_device).set_state(record["torch_cuda"])
            torch.Generator(device=exact_device).set_state(generators["vae"])
    except BaseException as exc:
        raise RNGCheckpointError(f"rng_state rank {rank} Torch payload is invalid") from exc


def validate_rng_envelope(
    envelope: Any,
    *,
    expected_world_size: int | None = None,
    expected_contract: Mapping[str, Any] | None = None,
    expected_device_type: str | None = None,
    validation_device: torch.device | None = None,
) -> None:
    if not isinstance(envelope, Mapping):
        raise RNGCheckpointError("checkpoint rng_state must be a mapping")
    expected_keys = {"schema", "world_size", "contract", "per_rank"}
    if set(envelope) != expected_keys:
        raise RNGCheckpointError(
            f"rng_state envelope keys mismatch: expected={sorted(expected_keys)}, "
            f"got={sorted(envelope)}"
        )
    if envelope["schema"] != RNG_STATE_SCHEMA:
        raise RNGCheckpointError(
            f"unsupported RNG schema: checkpoint={envelope['schema']!r}, "
            f"required={RNG_STATE_SCHEMA!r}"
        )
    if isinstance(envelope["world_size"], bool) or not isinstance(envelope["world_size"], int):
        raise RNGCheckpointError("checkpoint RNG world_size must be int")
    saved_world_size = envelope["world_size"]
    if saved_world_size <= 0:
        raise RNGCheckpointError(f"invalid checkpoint RNG world_size={saved_world_size}")
    if expected_world_size is not None and saved_world_size != int(expected_world_size):
        raise RNGCheckpointError(
            "checkpoint RNG world size mismatch: "
            f"checkpoint={saved_world_size}, current={int(expected_world_size)}"
        )
    contract = _normalise_contract(envelope["contract"], "checkpoint RNG contract")
    if expected_contract is not None and contract != _normalise_contract(
        expected_contract, "current RNG contract"
    ):
        raise RNGCheckpointError(
            f"checkpoint RNG contract mismatch: checkpoint={contract!r}, "
            f"current={_normalise_contract(expected_contract, 'current RNG contract')!r}"
        )
    per_rank = envelope["per_rank"]
    if not isinstance(per_rank, list) or len(per_rank) != saved_world_size:
        raise RNGCheckpointError(
            f"rng_state per_rank length mismatch: expected={saved_world_size}, "
            f"got={len(per_rank) if isinstance(per_rank, list) else type(per_rank).__name__}"
        )
    embedded_ranks = []
    for slot, record in enumerate(per_rank):
        _validate_local_state(
            record,
            slot,
            expected_device_type,
            validation_device=validation_device,
        )
        embedded_ranks.append(int(record["rank"]))
    if embedded_ranks != list(range(saved_world_size)):
        raise RNGCheckpointError(
            f"rng_state rank set/order mismatch: got={embedded_ranks}, "
            f"expected={list(range(saved_world_size))}"
        )
    expected_unique = 1 if contract["rank_independent_seed"] else saved_world_size
    for name in RNG_STREAM_NAMES:
        digests = {
            hashlib.sha256(bytes(record["generators"][name].cpu().tolist())).hexdigest()
            for record in per_rank
        }
        if len(digests) != expected_unique:
            raise RNGCheckpointError(
                f"rng_state generator uniqueness violation for {name}: "
                f"unique={len(digests)}, expected={expected_unique}, "
                f"world_size={saved_world_size}"
            )


def collect_rng_state(model, *, rank: int, is_dist: bool) -> dict[str, Any] | None:
    """Collect all rank states; every rank participates, only rank 0 returns it.

    Local capture errors are represented in the object collective first so one
    bad rank cannot strand its peers inside ``all_gather_object``.
    """
    if is_dist:
        if not dist.is_initialized():
            raise RNGCheckpointError("is_dist=True but process group is not initialized")
        actual_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        if dist.is_initialized():
            raise RNGCheckpointError("is_dist=False while a process group is initialized")
        actual_rank, world_size = 0, 1
        if int(rank) != 0:
            raise RNGCheckpointError(f"non-distributed checkpoint rank must be 0, got {rank}")

    try:
        if int(rank) != actual_rank:
            raise RNGCheckpointError(f"rank argument {rank} != distributed rank {actual_rank}")
        local: Any = capture_local_rng_state(model, actual_rank, world_size)
    except BaseException as exc:
        local = {
            "__capture_error__": f"{type(exc).__name__}: {exc}",
            "rank": actual_rank,
        }
    if is_dist:
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    errors = [x for x in gathered if isinstance(x, Mapping) and "__capture_error__" in x]
    if errors:
        raise RNGCheckpointError(f"per-rank RNG capture failed: {errors}")
    try:
        local_contract: Any = {"contract": _model_contract(model)}
    except BaseException as exc:
        local_contract = {
            "__contract_error__": f"{type(exc).__name__}: {exc}",
            "rank": actual_rank,
        }
    if is_dist:
        contracts: list[Any] = [None] * world_size
        dist.all_gather_object(contracts, local_contract)
    else:
        contracts = [local_contract]
    contract_errors = [
        value for value in contracts
        if isinstance(value, Mapping) and "__contract_error__" in value
    ]
    if contract_errors:
        raise RNGCheckpointError(f"per-rank RNG contract capture failed: {contract_errors}")
    common_contract = contracts[0]["contract"]
    if any(value["contract"] != common_contract for value in contracts[1:]):
        raise RNGCheckpointError(f"RNG contract differs across ranks: {contracts}")
    envelope = {
        "schema": RNG_STATE_SCHEMA,
        "world_size": world_size,
        "contract": common_contract,
        "per_rank": gathered,
    }
    validation_error = None
    try:
        rng_device = _model_rng_device(model)
        validate_rng_envelope(
            envelope,
            expected_world_size=world_size,
            expected_contract=common_contract,
            expected_device_type=rng_device.type,
            validation_device=rng_device,
        )
    except BaseException as exc:
        validation_error = exc
    _raise_synchronized_error(
        validation_error,
        is_dist=is_dist,
        context="per-rank RNG envelope validation failed",
    )
    return envelope if actual_rank == 0 else None


def validate_resume_checkpoint(ckpt: Any, model, *, rank: int, is_dist: bool) -> dict[str, Any]:
    if not isinstance(ckpt, Mapping):
        raise RNGCheckpointError("training checkpoint must be a mapping")
    if ckpt.get("checkpoint_schema") != FULL_CHECKPOINT_SCHEMA:
        raise RNGCheckpointError(
            "resume requires exact checkpoint schema "
            f"{FULL_CHECKPOINT_SCHEMA!r}; got {ckpt.get('checkpoint_schema')!r}"
        )
    if is_dist:
        if not dist.is_initialized():
            raise RNGCheckpointError("resume is_dist=True but process group is not initialized")
        current_rank, world_size = dist.get_rank(), dist.get_world_size()
        if int(rank) != current_rank:
            raise RNGCheckpointError(f"resume rank argument {rank} != distributed rank {current_rank}")
    else:
        current_rank, world_size = 0, 1
        if int(rank) != 0:
            raise RNGCheckpointError(f"non-distributed resume rank must be 0, got {rank}")
    validate_rng_envelope(
        ckpt.get("rng_state"),
        expected_world_size=world_size,
        expected_contract=_model_contract(model),
        expected_device_type=_model_rng_device(model).type,
        validation_device=_model_rng_device(model),
    )
    return dict(ckpt["rng_state"]["per_rank"][current_rank])


def restore_local_rng_state(record: Mapping[str, Any], model) -> None:
    """Restore only after model/optimizer/scheduler/EMA loading has completed."""
    device = _model_rng_device(model)
    try:
        rank = int(record.get("rank", -1)) if isinstance(record, Mapping) else -1
    except (TypeError, ValueError) as exc:
        raise RNGCheckpointError("local RNG record has invalid rank") from exc
    _validate_local_state(record, rank, device.type, validation_device=device)
    generators = _model_generators(model)
    if _as_device(generators["vae"].device) != device:
        raise RNGCheckpointError("current VAE generator device changed before RNG restore")

    random.setstate(record["python"])
    np.random.set_state(_numpy_state_for_restore(record["numpy"]))
    torch.set_rng_state(_byte_state(record["torch_cpu"], "torch_cpu"))
    if device.type == "cuda":
        torch.cuda.set_rng_state(_byte_state(record["torch_cuda"], "torch_cuda"), device)
    for name in RNG_STREAM_NAMES:
        generators[name].set_state(_byte_state(record["generators"][name], name))

    # Read-back is cheap (states are only a few KiB) and turns a partial or
    # device-misdirected restore into an immediate startup failure.
    if random.getstate() != record["python"]:
        raise RNGCheckpointError("Python RNG state read-back mismatch")
    current_numpy = np.random.get_state()
    saved_numpy = record["numpy"]
    if not (
        current_numpy[0] == saved_numpy[0]
        and np.array_equal(current_numpy[1], _numpy_state_for_restore(saved_numpy)[1])
        and current_numpy[2:] == saved_numpy[2:]
    ):
        raise RNGCheckpointError("NumPy RNG state read-back mismatch")
    if not torch.equal(torch.get_rng_state().cpu(), record["torch_cpu"].cpu()):
        raise RNGCheckpointError("Torch CPU RNG state read-back mismatch")
    if device.type == "cuda" and not torch.equal(
        torch.cuda.get_rng_state(device).cpu(), record["torch_cuda"].cpu()
    ):
        raise RNGCheckpointError("Torch CUDA RNG state read-back mismatch")
    for name in RNG_STREAM_NAMES:
        if not torch.equal(
            generators[name].get_state().cpu(), record["generators"][name].cpu()
        ):
            raise RNGCheckpointError(f"{name} generator state read-back mismatch")


def restore_local_rng_state_synchronized(
    record: Mapping[str, Any],
    model,
    *,
    is_dist: bool,
) -> None:
    """Restore on every rank and propagate any rank-local failure coherently."""
    error = None
    try:
        restore_local_rng_state(record, model)
    except BaseException as exc:
        error = exc
    if is_dist:
        if not dist.is_initialized():
            raise RNGCheckpointError("synchronized RNG restore requires a process group")
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(
            gathered,
            None if error is None else f"{type(error).__name__}: {error}",
        )
        failures = [(rank, value) for rank, value in enumerate(gathered) if value is not None]
        if failures:
            message = f"per-rank RNG restore failed: {failures}"
            if error is not None:
                raise RNGCheckpointError(message) from error
            raise RNGCheckpointError(message)
    elif error is not None:
        raise error


def _local_rng_states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["rank"] != right["rank"] or left["device_type"] != right["device_type"]:
        return False
    if left["python"] != right["python"]:
        return False
    left_np, right_np = left["numpy"], right["numpy"]
    if not (
        left_np[0] == right_np[0]
        and torch.equal(left_np[1].cpu(), right_np[1].cpu())
        and left_np[2:] == right_np[2:]
    ):
        return False
    for key in ("torch_cpu", "torch_cuda"):
        if left[key] is None or right[key] is None:
            if left[key] is not right[key]:
                return False
        elif not torch.equal(left[key].cpu(), right[key].cpu()):
            return False
    return all(
        torch.equal(left["generators"][name].cpu(), right["generators"][name].cpu())
        for name in RNG_STREAM_NAMES
    )


def assert_live_rng_matches_synchronized(
    reference: Mapping[str, Any],
    model,
    *,
    rank: int,
    world_size: int,
    is_dist: bool,
    context: str,
) -> None:
    """Guard the short post-restore path immediately before first batch."""
    error = None
    try:
        current = capture_local_rng_state(model, rank, world_size)
        if not _local_rng_states_equal(reference, current):
            raise RNGCheckpointError(f"live RNG changed after restore during {context}")
    except BaseException as exc:
        error = exc
    if is_dist:
        failures = [None] * world_size
        dist.all_gather_object(
            failures,
            None if error is None else f"{type(error).__name__}: {error}",
        )
        bad = [(r, value) for r, value in enumerate(failures) if value is not None]
        if bad:
            message = f"post-restore RNG guard failed: {bad}"
            if error is not None:
                raise RNGCheckpointError(message) from error
            raise RNGCheckpointError(message)
    elif error is not None:
        raise error


def validate_init_from_rng_policy(ckpt: Any, config) -> str:
    """Validate source metadata but always retain the new stage's fresh streams."""
    if not isinstance(ckpt, Mapping):
        raise RNGCheckpointError("init_from checkpoint must be a mapping")
    schema_present = "checkpoint_schema" in ckpt
    rng_present = "rng_state" in ckpt
    if schema_present != rng_present:
        raise RNGCheckpointError(
            "init_from checkpoint has a partial RNG/checkpoint schema; "
            "partial metadata is corruption, not legacy_missing_rng"
        )
    if schema_present:
        if ckpt["checkpoint_schema"] != FULL_CHECKPOINT_SCHEMA:
            raise RNGCheckpointError(
                f"init_from checkpoint has unknown schema={ckpt['checkpoint_schema']!r}"
            )
        validate_rng_envelope(ckpt["rng_state"])
        return "fresh_rng_reset_from_v2"

    policy = str(config.trainer.get("init_from_rng_policy", "require_rng_schema"))
    if policy != LEGACY_MISSING_RNG_POLICY:
        raise RNGCheckpointError(
            "init_from source has no exact RNG schema.  The audited legacy A-final "
            "exception requires trainer.init_from_rng_policy=legacy_missing_rng"
        )
    return LEGACY_MISSING_RNG_POLICY


def verify_initial_vae_rng_uniqueness(model, *, rank: int, is_dist: bool) -> int:
    """Runtime canary: global-rank VAE streams must be unique in real training."""
    if is_dist:
        if not dist.is_initialized():
            raise RNGCheckpointError("VAE RNG uniqueness check requires initialized process group")
        world_size = dist.get_world_size()
    else:
        if int(rank) != 0:
            raise RNGCheckpointError("non-distributed VAE RNG uniqueness rank must be 0")
        world_size = 1
    try:
        if is_dist and int(rank) != dist.get_rank():
            raise RNGCheckpointError("VAE RNG uniqueness check rank mismatch")
        generators = _model_generators(model)
        state = generators["vae"].get_state().detach().cpu()
        local: Any = {
            "digest": hashlib.sha256(bytes(state.tolist())).hexdigest(),
            "contract": _model_contract(model),
        }
    except BaseException as exc:
        local = {"error": f"{type(exc).__name__}: {exc}", "rank": int(rank)}
    if is_dist:
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    errors = [value for value in gathered if "error" in value]
    if errors:
        raise RNGCheckpointError(f"VAE RNG initial-state capture failed: {errors}")
    contracts = [value["contract"] for value in gathered]
    if any(value != contracts[0] for value in contracts[1:]):
        raise RNGCheckpointError(f"VAE RNG contract differs across ranks: {contracts}")
    digests = [value["digest"] for value in gathered]
    unique = len(set(digests))
    contract = contracts[0]
    expected = 1 if contract["rank_independent_seed"] else world_size
    if unique != expected:
        raise RNGCheckpointError(
            "VAE RNG initial-state uniqueness violation: "
            f"unique={unique}, expected={expected}, world_size={world_size}"
        )
    return unique
