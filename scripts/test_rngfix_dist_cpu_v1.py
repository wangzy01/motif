#!/usr/bin/env python3
"""Two-process Gloo test for per-rank RNG gather/select/restore."""

from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.rng_state import (  # noqa: E402
    FULL_CHECKPOINT_SCHEMA,
    RNGCheckpointError,
    collect_rng_state,
    restore_local_rng_state,
    validate_resume_checkpoint,
)
from training.formal_boot import (  # noqa: E402
    EXPECTED_RUN_LINEAGE_ENV,
    resolve_checkpoint_boot_branch,
    resolve_expected_run_lineage_sha256,
)
from training.checkpoint import sync_metrics_before_checkpoint  # noqa: E402


class DummyRngModel:
    def __init__(self, rank):
        self.rank = rank
        base = 123 + rank * 1_000_003
        self.generators = {
            "dropout": torch.Generator().manual_seed(base + 11),
            "diffusion": torch.Generator().manual_seed(base + 23),
            "vae": torch.Generator().manual_seed(base + 37),
        }

    def training_rng_generators(self):
        return self.generators

    def training_rng_contract(self):
        return {
            "base_seed": 123,
            "rank_independent_seed": False,
            "rank_stride": 1_000_003,
            "stream_offsets": {"dropout": 11, "diffusion": 23, "vae": 37},
        }

    def training_rng_device(self):
        return torch.device("cpu")


def worker(rank, world_size, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        random.seed(500 + rank)
        np.random.seed(600 + rank)
        torch.manual_seed(700 + rank)
        model = DummyRngModel(rank)
        envelope = collect_rng_state(model, rank=rank, is_dist=True)
        objects = [envelope]
        dist.broadcast_object_list(objects, src=0)
        envelope = objects[0]
        assert [record["rank"] for record in envelope["per_rank"]] == [0, 1]
        assert not torch.equal(
            envelope["per_rank"][0]["generators"]["vae"],
            envelope["per_rank"][1]["generators"]["vae"],
        )

        ckpt = {"checkpoint_schema": FULL_CHECKPOINT_SCHEMA, "rng_state": envelope}
        pending = validate_resume_checkpoint(ckpt, model, rank=rank, is_dist=True)
        expected = torch.rand(5, generator=model.generators["vae"])
        restore_local_rng_state(pending, model)
        actual = torch.rand(5, generator=model.generators["vae"])
        assert torch.equal(expected, actual)

        if rank == 0:
            # Only rank 1's opaque state is corrupted.  Every rank validates
            # the complete envelope and must fail coherently before restore.
            envelope["per_rank"][1]["generators"]["vae"] = torch.ones(
                3, dtype=torch.uint8
            )
        objects = [envelope]
        dist.broadcast_object_list(objects, src=0)
        failed = False
        try:
            validate_resume_checkpoint(
                {"checkpoint_schema": FULL_CHECKPOINT_SCHEMA, "rng_state": objects[0]},
                model,
                rank=rank,
                is_dist=True,
            )
        except RNGCheckpointError:
            failed = True
        flags = [None] * world_size
        dist.all_gather_object(flags, failed)
        assert flags == [True, True]

        # One rank receiving a different formal lineage must make every rank
        # leave the collective with the same fail-closed outcome.
        os.environ[EXPECTED_RUN_LINEAGE_ENV] = ("a" if rank == 0 else "b") * 64
        lineage_failed = False
        try:
            resolve_expected_run_lineage_sha256(is_dist=True)
        except RuntimeError:
            lineage_failed = True
        flags = [None] * world_size
        dist.all_gather_object(flags, lineage_failed)
        assert flags == [True, True]
        os.environ.pop(EXPECTED_RUN_LINEAGE_ENV, None)

        # Deliberately give the ranks different filesystem observations.  Both
        # must fail before either can enter resume/init state collectives.
        visible = Path(rendezvous).with_name("rank0_visible_last.pt")
        if rank == 0:
            visible.write_bytes(b"checkpoint")
        dist.barrier()
        local_path = visible if rank == 0 else visible.with_name("rank1_missing_last.pt")
        branch_failed = False
        try:
            resolve_checkpoint_boot_branch(
                str(local_path),
                resume=True,
                init_from="/a/final.pt",
                expected_mode="continue",
                is_dist=True,
            )
        except RuntimeError:
            branch_failed = True
        flags = [None] * world_size
        dist.all_gather_object(flags, branch_failed)
        assert flags == [True, True]

        class BrokenMetrics:
            def flush(self):
                raise OSError("injected metrics fsync boundary failure")

        metrics_failed = False
        try:
            sync_metrics_before_checkpoint(
                BrokenMetrics() if rank == 0 else None,
                rank=rank,
                is_dist=True,
            )
        except RuntimeError as exc:
            assert "metrics durable checkpoint boundary" in str(exc)
            metrics_failed = True
        flags = [None] * world_size
        dist.all_gather_object(flags, metrics_failed)
        assert flags == [True, True]
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    with tempfile.TemporaryDirectory() as td:
        rendezvous = os.path.join(td, "gloo_init")
        mp.spawn(worker, args=(2, rendezvous), nprocs=2, join=True)
    print("PASS test_rngfix_dist_cpu_v1", flush=True)


if __name__ == "__main__":
    main()
