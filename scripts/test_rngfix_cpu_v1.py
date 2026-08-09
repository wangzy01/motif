#!/usr/bin/env python3
"""CPU policy tests for Stage512A-control RNG/checkpoint v2.

This is intentionally outside ``tests/`` because the repository has an exact
allow-list for that directory.  It needs no model weights, tokenizer, diffusers
installation, or network access.
"""

from __future__ import annotations

import copy
import os
import random
import shutil
import stat
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# networks.vae imports einops at module import.  The tests below exercise only
# the posterior sampler contract, so a no-op stand-in keeps this CPU policy test
# independent of optional runtime packages.
if "einops" not in sys.modules:
    einops = types.ModuleType("einops")
    einops.rearrange = lambda value, *_args, **_kwargs: value
    sys.modules["einops"] = einops

from networks.vae import Flux2VAE  # noqa: E402
from training.checkpoint import (  # noqa: E402
    WEIGHTS_EMA_SNAPSHOT_SCHEMA,
    _ablation_mismatch_check,
    _ema_state_integrity_check,
    _optimizer_state_integrity_check,
    _publish_periodic_snapshot,
    init_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    should_write_final_checkpoint,
    sync_metrics_before_checkpoint,
)
from training.formal_boot import verify_expected_boot_mode  # noqa: E402
from training.formal_boot import (  # noqa: E402
    EXPECTED_RUN_LINEAGE_ENV,
    FORMAL_RUN_LINEAGE_KEY,
    resolve_checkpoint_boot_branch,
)
from training.rng_state import (  # noqa: E402
    FULL_CHECKPOINT_SCHEMA,
    RNGCheckpointError,
    RNG_STATE_SCHEMA,
    assert_live_rng_matches_synchronized,
    capture_local_rng_state,
    restore_local_rng_state,
    validate_rng_envelope,
    validate_resume_checkpoint,
)


RANK_STRIDE = 1_000_003
OFFSETS = {"dropout": 11, "diffusion": 23, "vae": 37}


class DummyPosterior:
    def __init__(self, shape=(2, 4, 3, 3)):
        self.parameters = torch.zeros(shape)

    def sample(self, generator=None):
        assert generator is not None
        return torch.randn(self.parameters.shape, generator=generator)

    def mode(self):
        return torch.zeros_like(self.parameters)


class DummyWrapper:
    def __init__(self, seed=123, rank=0):
        self.model = torch.nn.Linear(3, 2)
        self._ema = None
        self.seed = int(seed)
        self.rank = int(rank)
        base = self.seed + self.rank * RANK_STRIDE
        self.dropout_generator = torch.Generator().manual_seed(base + OFFSETS["dropout"])
        self.diffusion_generator = torch.Generator().manual_seed(base + OFFSETS["diffusion"])
        self.vae_generator = torch.Generator().manual_seed(base + OFFSETS["vae"])

    def training_rng_generators(self):
        return {
            "dropout": self.dropout_generator,
            "diffusion": self.diffusion_generator,
            "vae": self.vae_generator,
        }

    def training_rng_contract(self):
        return {
            "base_seed": self.seed,
            "rank_independent_seed": False,
            "rank_stride": RANK_STRIDE,
            "stream_offsets": dict(OFFSETS),
        }

    def training_rng_device(self):
        return torch.device("cpu")


def make_config(
    *,
    legacy=False,
    keep_full=False,
    keep_format="full",
    keep_every=None,
    keep_max=None,
):
    trainer = {
        "checkpoint_keep_every": (
            int(keep_every) if keep_every is not None else (1 if keep_full else 0)
        ),
        "checkpoint_keep_max": (
            int(keep_max) if keep_max is not None else (1 if keep_full else 0)
        ),
        "checkpoint_keep_format": keep_format,
    }
    if legacy:
        trainer["init_from_rng_policy"] = "legacy_missing_rng"
    return SimpleNamespace(model={}, trainer=trainer)


def make_training(seed=123):
    model = DummyWrapper(seed=seed)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    return model, optimizer, scheduler


def stochastic_step(model, optimizer, scheduler):
    generators = model.training_rng_generators()
    draws = torch.tensor(
        [
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
            float(torch.rand((), generator=generators["dropout"])),
            float(torch.rand((), generator=generators["diffusion"])),
            float(torch.rand((), generator=generators["vae"])),
        ]
    )
    x = torch.tensor([[draws[0], draws[1], draws[2]]])
    target = torch.stack((draws[3], draws[4])).view(1, 2) + draws[5]
    optimizer.zero_grad(set_to_none=True)
    loss = (model.model(x) - target).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return draws, loss.detach().clone()


def rng_digest(model):
    local = capture_local_rng_state(model, 0, 1)
    return {
        "python": copy.deepcopy(local["python"]),
        "numpy": (
            local["numpy"][0],
            local["numpy"][1].clone(),
            *local["numpy"][2:],
        ),
        "torch_cpu": local["torch_cpu"].clone(),
        "generators": {k: v.clone() for k, v in local["generators"].items()},
    }


def assert_rng_digest_equal(test, a, b):
    test.assertEqual(a["python"], b["python"])
    test.assertEqual(a["numpy"][0], b["numpy"][0])
    test.assertTrue(torch.equal(a["numpy"][1], b["numpy"][1]))
    test.assertEqual(a["numpy"][2:], b["numpy"][2:])
    test.assertTrue(torch.equal(a["torch_cpu"], b["torch_cpu"]))
    for name in OFFSETS:
        test.assertTrue(torch.equal(a["generators"][name], b["generators"][name]))


class RngFixCpuTests(unittest.TestCase):
    def setUp(self):
        self._inherited_lineage = os.environ.pop(EXPECTED_RUN_LINEAGE_ENV, None)
        random.seed(77)
        np.random.seed(77)
        torch.manual_seed(77)

    def tearDown(self):
        os.environ.pop(EXPECTED_RUN_LINEAGE_ENV, None)
        if self._inherited_lineage is not None:
            os.environ[EXPECTED_RUN_LINEAGE_ENV] = self._inherited_lineage

    def test_rank0_rank8_vae_epsilon_differs_and_replays(self):
        posterior = DummyPosterior()
        global_before = torch.get_rng_state().clone()
        g0 = torch.Generator().manual_seed(123 + 0 * RANK_STRIDE + OFFSETS["vae"])
        g8 = torch.Generator().manual_seed(123 + 8 * RANK_STRIDE + OFFSETS["vae"])
        epsilon0 = Flux2VAE._sample_posterior(
            posterior, do_sample=True, generator=g0, context="test"
        )
        epsilon8 = Flux2VAE._sample_posterior(
            posterior, do_sample=True, generator=g8, context="test"
        )
        replay0 = Flux2VAE._sample_posterior(
            posterior,
            do_sample=True,
            generator=torch.Generator().manual_seed(123 + OFFSETS["vae"]),
            context="test",
        )
        self.assertFalse(torch.equal(epsilon0, epsilon8))
        self.assertTrue(torch.equal(epsilon0, replay0))
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))

    def test_mode_does_not_advance_vae_generator(self):
        posterior = DummyPosterior()
        generator = torch.Generator().manual_seed(9)
        before = generator.get_state().clone()
        out = Flux2VAE._sample_posterior(
            posterior, do_sample=False, generator=generator, context="test"
        )
        self.assertTrue(torch.equal(out, posterior.mode()))
        self.assertTrue(torch.equal(before, generator.get_state()))

    def test_missing_generator_fails_instead_of_global_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "explicit per-rank VAE generator"):
            Flux2VAE._sample_posterior(
                DummyPosterior(), do_sample=True, generator=None, context="test"
            )

    def test_checkpoint_resume_is_exact_at_next_update(self):
        model, optimizer, scheduler = make_training()
        stochastic_step(model, optimizer, scheduler)  # initialize every Adam state
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "last.pt")
            rng_before_save = rng_digest(model)
            save_checkpoint(
                path, model, optimizer, scheduler, 1, False, 0, make_config(keep_full=True)
            )
            assert_rng_digest_equal(self, rng_before_save, rng_digest(model))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            safe_payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(safe_payload["checkpoint_schema"], FULL_CHECKPOINT_SCHEMA)
            self.assertEqual(payload["checkpoint_schema"], FULL_CHECKPOINT_SCHEMA)
            self.assertEqual(payload["rng_state"]["schema"], RNG_STATE_SCHEMA)
            self.assertEqual(len(payload["rng_state"]["per_rank"]), 1)
            snapshot = os.path.join(td, "step_1.pt")
            self.assertTrue(os.path.isfile(snapshot))
            self.assertTrue(os.path.samefile(path, snapshot))
            self.assertEqual(os.lstat(snapshot).st_mode & 0o222, 0)
            snapshot_payload = torch.load(snapshot, map_location="cpu", weights_only=False)
            self.assertEqual(snapshot_payload["checkpoint_schema"], FULL_CHECKPOINT_SCHEMA)
            self.assertEqual(snapshot_payload["rng_state"]["schema"], RNG_STATE_SCHEMA)
            # Inject the exact crash window: last.pt is durable but the periodic
            # anchor was never published.  Resume must idempotently repair it.
            os.unlink(snapshot)
            self.assertFalse(os.path.exists(snapshot))

            expected_draws, expected_loss = stochastic_step(model, optimizer, scheduler)
            expected_params = {
                name: value.detach().clone() for name, value in model.model.named_parameters()
            }
            expected_scheduler = copy.deepcopy(scheduler.state_dict())

            resumed, resumed_optimizer, resumed_scheduler = make_training()
            step, pending = load_checkpoint(
                path,
                resumed,
                resumed_optimizer,
                resumed_scheduler,
                False,
                make_config(keep_full=True),
            )
            self.assertEqual(step, 1)
            self.assertTrue(os.path.isfile(snapshot))
            # Simulate resume-only startup RNG consumption; the late commit must erase it.
            random.random()
            np.random.random()
            torch.rand(4)
            restore_local_rng_state(pending, resumed)
            assert_live_rng_matches_synchronized(
                pending,
                resumed,
                rank=0,
                world_size=1,
                is_dist=False,
                context="test no-op startup tail",
            )
            actual_draws, actual_loss = stochastic_step(
                resumed, resumed_optimizer, resumed_scheduler
            )
            self.assertTrue(torch.equal(expected_draws, actual_draws))
            self.assertTrue(torch.equal(expected_loss, actual_loss))
            for name, value in resumed.model.named_parameters():
                self.assertTrue(torch.equal(expected_params[name], value))
            self.assertEqual(expected_scheduler, resumed_scheduler.state_dict())

    def test_checkpoint_loads_use_safe_mmap_for_resume_and_legacy_init(self):
        launcher = (REPO / "scripts/stage512a_control_formal_launch.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'torch.load(path, map_location="cpu", weights_only=True, mmap=True)',
            launcher,
        )
        self.assertNotIn("weights_only=False", launcher)

        source, source_optimizer, source_scheduler = make_training()
        stochastic_step(source, source_optimizer, source_scheduler)
        with tempfile.TemporaryDirectory() as td:
            v2 = os.path.join(td, "v2.pt")
            legacy = os.path.join(td, "legacy.pt")
            save_checkpoint(v2, source, source_optimizer, source_scheduler, 1, False, 0, make_config())
            payload = torch.load(v2, map_location="cpu", weights_only=True, mmap=True)
            del payload["checkpoint_schema"]
            del payload["rng_state"]
            torch.save(payload, legacy)

            target, target_optimizer, target_scheduler = make_training()
            with unittest.mock.patch(
                "training.checkpoint.torch.load", wraps=torch.load
            ) as safe_load:
                step, _ = load_checkpoint(
                    v2, target, target_optimizer, target_scheduler, False, make_config()
                )
            self.assertEqual(step, 1)
            self.assertTrue(all(call.kwargs.get("weights_only") is True for call in safe_load.call_args_list))
            self.assertTrue(all(call.kwargs.get("mmap") is True for call in safe_load.call_args_list))

            target, target_optimizer, _ = make_training()
            with unittest.mock.patch(
                "training.checkpoint.torch.load", wraps=torch.load
            ) as safe_load:
                policy = init_from_checkpoint(
                    legacy, target, target_optimizer, False, make_config(legacy=True), True
                )
            self.assertEqual(policy, "legacy_missing_rng")
            self.assertIs(safe_load.call_args.kwargs["weights_only"], True)
            self.assertIs(safe_load.call_args.kwargs["mmap"], True)

    def test_metrics_flush_and_fsync_precede_checkpoint_or_fail_closed(self):
        events = []

        class Metrics:
            def flush(self):
                events.append("flush")

            def fileno(self):
                events.append("fileno")
                return 17

        with unittest.mock.patch(
            "training.checkpoint.os.fsync", side_effect=lambda fd: events.append(("fsync", fd))
        ):
            sync_metrics_before_checkpoint(Metrics(), rank=0, is_dist=False)
        self.assertEqual(events, ["flush", "fileno", ("fsync", 17)])

        class BrokenMetrics:
            def flush(self):
                raise OSError("disk full")

        with self.assertRaisesRegex(OSError, "disk full"):
            sync_metrics_before_checkpoint(BrokenMetrics(), rank=0, is_dist=False)

    def test_full_snapshot_repairs_only_same_inode_0600_crash_window(self):
        config = make_config(keep_full=True, keep_format="full")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "last.pt")
            snapshot = os.path.join(td, "step_1.pt")
            torch.save({"global_step": 1}, path)
            os.chmod(path, 0o600)
            os.link(path, snapshot)
            _publish_periodic_snapshot(path, {"global_step": 1}, config)
            self.assertEqual(stat.S_IMODE(os.lstat(snapshot).st_mode), 0o444)

            unknown_alias = os.path.join(td, "unknown_alias.pt")
            os.link(snapshot, unknown_alias)
            os.chmod(snapshot, 0o600)
            self.assertEqual(os.lstat(snapshot).st_nlink, 3)
            with self.assertRaisesRegex(RuntimeError, "unexpected writable mode/link count"):
                _publish_periodic_snapshot(path, {"global_step": 1}, config)
            os.unlink(unknown_alias)

            os.unlink(snapshot)
            stale = snapshot + ".stale"
            torch.save({"global_step": 1}, stale)
            os.chmod(stale, 0o600)
            os.replace(stale, snapshot)
            with self.assertRaisesRegex(RuntimeError, "not the same inode"):
                _publish_periodic_snapshot(path, {"global_step": 1}, config)

    def test_periodic_snapshot_existing_targets_fail_closed(self):
        model, optimizer, scheduler = make_training()
        stochastic_step(model, optimizer, scheduler)
        full_config = make_config(keep_full=True, keep_format="full")
        compact_config = make_config(keep_full=True, keep_format="weights_ema")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "last.pt")
            snapshot = os.path.join(td, "step_1.pt")
            save_checkpoint(path, model, optimizer, scheduler, 1, False, 0, full_config)
            payload = torch.load(path, map_location="cpu", weights_only=True)

            self.assertTrue(os.path.samefile(path, snapshot))
            os.unlink(snapshot)
            shutil.copyfile(path, snapshot)
            os.chmod(snapshot, 0o444)
            with self.assertRaisesRegex(RuntimeError, "not the same inode"):
                _publish_periodic_snapshot(path, payload, full_config)
            self.assertFalse(os.path.samefile(path, snapshot))

            os.unlink(snapshot)
            os.symlink(path, snapshot)
            with self.assertRaisesRegex(RuntimeError, "not a regular file"):
                _publish_periodic_snapshot(path, payload, full_config)

            os.unlink(snapshot)
            _publish_periodic_snapshot(path, payload, compact_config)
            compact = torch.load(
                snapshot, map_location="cpu", weights_only=True, mmap=True
            )
            self.assertEqual(compact["snapshot_schema"], WEIGHTS_EMA_SNAPSHOT_SCHEMA)
            self.assertEqual(compact["global_step"], 1)
            compact["global_step"] = 2
            corrupt_tmp = snapshot + ".corrupt"
            torch.save(compact, corrupt_tmp)
            os.replace(corrupt_tmp, snapshot)
            os.chmod(snapshot, 0o444)
            with self.assertRaisesRegex(RuntimeError, "global_step mismatch"):
                _publish_periodic_snapshot(path, payload, compact_config)

            os.unlink(snapshot)
            _publish_periodic_snapshot(path, payload, compact_config)
            compact = torch.load(snapshot, map_location="cpu", weights_only=True)
            name = next(iter(compact["trainable_state"]))
            compact["trainable_state"][name] = compact["trainable_state"][name] + 1
            torch.save(compact, corrupt_tmp)
            os.replace(corrupt_tmp, snapshot)
            os.chmod(snapshot, 0o444)
            with self.assertRaisesRegex(RuntimeError, "value mismatch"):
                _publish_periodic_snapshot(path, payload, compact_config)

    def test_full_snapshot_formal_milestones_are_permanent(self):
        config = SimpleNamespace(
            model={},
            trainer={
                "checkpoint_keep_every": 30_000,
                "checkpoint_keep_max": 4,
                "checkpoint_keep_format": "full",
            },
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "last.pt")
            for step in (30_000, 60_000, 90_000, 120_000, 125_000):
                tmp = path + ".tmp"
                torch.save({"global_step": step}, tmp)
                os.replace(tmp, path)
                published = _publish_periodic_snapshot(
                    path, {"global_step": step}, config
                )
                if step % 30_000 == 0:
                    self.assertEqual(published, os.path.join(td, f"step_{step}.pt"))
                    self.assertTrue(os.path.samefile(path, published))
                    self.assertEqual(os.lstat(published).st_mode & 0o222, 0)
                else:
                    self.assertIsNone(published)
            self.assertEqual(
                sorted(name for name in os.listdir(td) if name.startswith("step_")),
                ["step_120000.pt", "step_30000.pt", "step_60000.pt", "step_90000.pt"],
            )
            for step in (30_000, 60_000, 90_000, 120_000):
                saved = torch.load(
                    os.path.join(td, f"step_{step}.pt"),
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertEqual(saved["global_step"], step)

            tmp = path + ".tmp"
            torch.save({"global_step": 150_000}, tmp)
            os.replace(tmp, path)
            with self.assertRaisesRegex(RuntimeError, "retention is full"):
                _publish_periodic_snapshot(path, {"global_step": 150_000}, config)
            self.assertFalse(os.path.exists(os.path.join(td, "step_150000.pt")))
            self.assertEqual(
                sorted(
                    torch.load(
                        os.path.join(td, name),
                        map_location="cpu",
                        weights_only=True,
                    )["global_step"]
                    for name in os.listdir(td)
                    if name.startswith("step_")
                ),
                [30_000, 60_000, 90_000, 120_000],
            )

    def test_compact_snapshot_rolls_and_exclusive_tmp_never_follows_symlink(self):
        config = make_config(
            keep_full=True,
            keep_format="weights_ema",
            keep_every=1,
            keep_max=1,
        )
        model, optimizer, scheduler = make_training()
        stochastic_step(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "last.pt")
            save_checkpoint(path, model, optimizer, scheduler, 1, False, 0, config)
            self.assertTrue(os.path.isfile(os.path.join(td, "step_1.pt")))
            self.assertEqual(os.lstat(os.path.join(td, "step_1.pt")).st_mode & 0o222, 0)
            stochastic_step(model, optimizer, scheduler)
            save_checkpoint(path, model, optimizer, scheduler, 2, False, 0, config)
            self.assertFalse(os.path.exists(os.path.join(td, "step_1.pt")))
            self.assertTrue(os.path.isfile(os.path.join(td, "step_2.pt")))

            victim = os.path.join(td, "victim")
            Path(victim).write_bytes(b"do-not-touch")
            token = "0" * 32
            blocked_tmp = os.path.join(
                td, f".last.pt.tmp.{os.getpid()}.{token}"
            )
            os.symlink(victim, blocked_tmp)
            with unittest.mock.patch(
                "training.checkpoint.secrets.token_hex", return_value=token
            ):
                with self.assertRaisesRegex(RuntimeError, "could not allocate exclusive"):
                    save_checkpoint(
                        path, model, optimizer, scheduler, 2, False, 0, make_config()
                    )
            self.assertEqual(Path(victim).read_bytes(), b"do-not-touch")

    def test_resume_requires_all_historical_full_milestones(self):
        config = make_config(
            keep_full=True, keep_format="full", keep_every=1, keep_max=2
        )
        model, optimizer, scheduler = make_training()
        stochastic_step(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "last.pt")
            save_checkpoint(path, model, optimizer, scheduler, 1, False, 0, config)
            stochastic_step(model, optimizer, scheduler)
            save_checkpoint(path, model, optimizer, scheduler, 2, False, 0, config)
            fresh, fresh_optimizer, fresh_scheduler = make_training()
            step, _ = load_checkpoint(
                path, fresh, fresh_optimizer, fresh_scheduler, False, config
            )
            self.assertEqual(step, 2)

            os.unlink(os.path.join(td, "step_1.pt"))
            fresh, fresh_optimizer, fresh_scheduler = make_training()
            with self.assertRaisesRegex(RuntimeError, "permanent full snapshot is missing"):
                load_checkpoint(
                    path, fresh, fresh_optimizer, fresh_scheduler, False, config
                )

        self.assertTrue(should_write_final_checkpoint(125_000, 124_500))
        self.assertFalse(should_write_final_checkpoint(120_000, 120_000))

    def test_resume_schema_failures_do_not_mutate_live_rng(self):
        model, optimizer, scheduler = make_training()
        stochastic_step(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, "good.pt")
            bad = os.path.join(td, "bad.pt")
            save_checkpoint(good, model, optimizer, scheduler, 1, False, 0, make_config())
            payload = torch.load(good, map_location="cpu", weights_only=False)
            cases = []
            wrong_schema = copy.deepcopy(payload)
            wrong_schema["rng_state"]["schema"] = "unknown"
            cases.append(wrong_schema)
            wrong_world = copy.deepcopy(payload)
            wrong_world["rng_state"]["world_size"] = 2
            cases.append(wrong_world)
            missing_stream = copy.deepcopy(payload)
            del missing_stream["rng_state"]["per_rank"][0]["generators"]["vae"]
            cases.append(missing_stream)
            bad_dtype = copy.deepcopy(payload)
            bad_dtype["rng_state"]["per_rank"][0]["torch_cpu"] = torch.ones(4)
            cases.append(bad_dtype)
            wrong_scheduler = copy.deepcopy(payload)
            wrong_scheduler["scheduler"]["last_epoch"] += 1
            cases.append(wrong_scheduler)

            for case in cases:
                torch.save(case, bad)
                fresh, fresh_optimizer, fresh_scheduler = make_training()
                before = rng_digest(fresh)
                with self.assertRaises((RNGCheckpointError, RuntimeError)):
                    load_checkpoint(
                        bad,
                        fresh,
                        fresh_optimizer,
                        fresh_scheduler,
                        False,
                        make_config(),
                    )
                after = rng_digest(fresh)
                assert_rng_digest_equal(self, before, after)

    def test_formal_run_lineage_is_saved_and_resume_bound(self):
        lineage = "a" * 64
        model, optimizer, scheduler = make_training()
        stochastic_step(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, "good.pt")
            bad = os.path.join(td, "bad.pt")
            with unittest.mock.patch.dict(
                os.environ, {EXPECTED_RUN_LINEAGE_ENV: lineage}, clear=False
            ):
                save_checkpoint(
                    good, model, optimizer, scheduler, 1, False, 0, make_config()
                )
            payload = torch.load(good, map_location="cpu", weights_only=True)
            self.assertEqual(payload[FORMAL_RUN_LINEAGE_KEY], lineage)
            fresh, fresh_optimizer, fresh_scheduler = make_training()
            with unittest.mock.patch.dict(
                os.environ, {EXPECTED_RUN_LINEAGE_ENV: lineage}, clear=False
            ):
                step, _ = load_checkpoint(
                    good,
                    fresh,
                    fresh_optimizer,
                    fresh_scheduler,
                    False,
                    make_config(),
                )
            self.assertEqual(step, 1)

            # Omitting the environment must not downgrade a formal checkpoint
            # into an unbound non-formal resume.
            fresh, fresh_optimizer, fresh_scheduler = make_training()
            before = rng_digest(fresh)
            with self.assertRaisesRegex(RuntimeError, "formal run lineage mismatch"):
                load_checkpoint(
                    good,
                    fresh,
                    fresh_optimizer,
                    fresh_scheduler,
                    False,
                    make_config(),
                )
            assert_rng_digest_equal(self, before, rng_digest(fresh))

            for case in ("missing", "wrong"):
                corrupted = copy.deepcopy(payload)
                if case == "missing":
                    del corrupted[FORMAL_RUN_LINEAGE_KEY]
                else:
                    corrupted[FORMAL_RUN_LINEAGE_KEY] = "b" * 64
                torch.save(corrupted, bad)
                fresh, fresh_optimizer, fresh_scheduler = make_training()
                before = rng_digest(fresh)
                with unittest.mock.patch.dict(
                    os.environ, {EXPECTED_RUN_LINEAGE_ENV: lineage}, clear=False
                ):
                    with self.assertRaisesRegex(RuntimeError, "formal run lineage mismatch"):
                        load_checkpoint(
                            bad,
                            fresh,
                            fresh_optimizer,
                            fresh_scheduler,
                            False,
                            make_config(),
                        )
                assert_rng_digest_equal(self, before, rng_digest(fresh))

            # The environment gate is opt-in for non-formal users.  A pre-lineage
            # v2 checkpoint remains resumable when the environment is absent.
            legacy_v2 = copy.deepcopy(payload)
            del legacy_v2[FORMAL_RUN_LINEAGE_KEY]
            torch.save(legacy_v2, bad)
            fresh, fresh_optimizer, fresh_scheduler = make_training()
            step, _ = load_checkpoint(
                bad, fresh, fresh_optimizer, fresh_scheduler, False, make_config()
            )
            self.assertEqual(step, 1)

            with unittest.mock.patch.dict(
                os.environ, {EXPECTED_RUN_LINEAGE_ENV: lineage.upper()}, clear=False
            ):
                with self.assertRaisesRegex(RuntimeError, "64 lowercase hex"):
                    save_checkpoint(
                        os.path.join(td, "malformed.pt"),
                        model,
                        optimizer,
                        scheduler,
                        1,
                        False,
                        0,
                        make_config(),
                    )

    def test_duplicate_per_rank_generator_state_is_rejected(self):
        rank0 = DummyWrapper(seed=123, rank=0)
        rank1 = DummyWrapper(seed=123, rank=1)
        envelope = {
            "schema": RNG_STATE_SCHEMA,
            "world_size": 2,
            "contract": rank0.training_rng_contract(),
            "per_rank": [
                capture_local_rng_state(rank0, 0, 2),
                capture_local_rng_state(rank1, 1, 2),
            ],
        }
        validate_rng_envelope(envelope, expected_world_size=2)
        envelope["per_rank"][1]["generators"]["vae"] = envelope["per_rank"][0][
            "generators"
        ]["vae"].clone()
        with self.assertRaisesRegex(RNGCheckpointError, "uniqueness violation for vae"):
            validate_rng_envelope(envelope, expected_world_size=2)

    def test_legacy_missing_rng_is_init_only_and_keeps_fresh_lineage(self):
        source, source_optimizer, source_scheduler = make_training()
        stochastic_step(source, source_optimizer, source_scheduler)
        with tempfile.TemporaryDirectory() as td:
            v2 = os.path.join(td, "v2.pt")
            legacy = os.path.join(td, "legacy.pt")
            save_checkpoint(v2, source, source_optimizer, source_scheduler, 1, False, 0, make_config())
            payload = torch.load(v2, map_location="cpu", weights_only=False)
            del payload["checkpoint_schema"]
            del payload["rng_state"]
            torch.save(payload, legacy)

            target, target_optimizer, target_scheduler = make_training(seed=123)
            with self.assertRaises((RNGCheckpointError, RuntimeError)):
                load_checkpoint(
                    legacy,
                    target,
                    target_optimizer,
                    target_scheduler,
                    False,
                    make_config(),
                )
            with self.assertRaises((RNGCheckpointError, RuntimeError)):
                init_from_checkpoint(
                    legacy, target, target_optimizer, False, make_config(), True
                )

            target, target_optimizer, _ = make_training(seed=123)
            before = rng_digest(target)
            policy = init_from_checkpoint(
                legacy,
                target,
                target_optimizer,
                False,
                make_config(legacy=True),
                True,
            )
            self.assertEqual(policy, "legacy_missing_rng")
            after = rng_digest(target)
            assert_rng_digest_equal(self, before, after)

    def test_strict_trainable_and_optimizer_closure(self):
        model, optimizer, scheduler = make_training()
        stochastic_step(model, optimizer, scheduler)
        state = optimizer.state_dict()
        trainable = {
            name: value.detach().clone() for name, value in model.model.named_parameters()
        }
        with self.assertRaisesRegex(RuntimeError, "trainable-key closure mismatch"):
            _ablation_mismatch_check(list(trainable)[:-1], model)
        broken = copy.deepcopy(state)
        broken["state"].pop(next(iter(broken["state"])))
        with self.assertRaisesRegex(RuntimeError, "state coverage mismatch"):
            _optimizer_state_integrity_check(broken, trainable, model, optimizer)
        broken = copy.deepcopy(state)
        broken["param_groups"].append(copy.deepcopy(broken["param_groups"][0]))
        with self.assertRaisesRegex(RuntimeError, "param-group count mismatch"):
            _optimizer_state_integrity_check(broken, trainable, model, optimizer)

        model._ema = {name: value.float().clone() for name, value in trainable.items()}
        with self.assertRaisesRegex(RuntimeError, "requires checkpoint ema mapping"):
            _ema_state_integrity_check(None, trainable, model)
        broken_ema = {name: value.float().clone() for name, value in trainable.items()}
        broken_ema.pop(next(iter(broken_ema)))
        with self.assertRaisesRegex(RuntimeError, "EMA key closure mismatch"):
            _ema_state_integrity_check(broken_ema, trainable, model)

    def test_formal_boot_contract(self):
        template = (
            REPO
            / "provenance/jcc_jobs/20260809_stage512a_continue_control_rngfix_FORMAL_w64.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "image: 471112743779.dkr.ecr.us-west-2.amazonaws.com/"
            "autobuild@sha256:0fa488a392ba06e91c576466dbb39407b823fd656fff88c1ee7a9eccf89b1d7f",
            template,
        )
        self.assertIn('BASH_ENV: "/dev/null"', template)
        self.assertIn(
            'exec /bin/bash --noprofile --norc -p '
            '"$FORMAL_REPO/scripts/stage512a_control_formal_launch.sh"',
            template,
        )
        self.assertNotIn(
            "exec /bin/bash --noprofile --norc -p scripts/", template
        )
        self.assertEqual(
            verify_expected_boot_mode(
                "fresh",
                actual="fresh",
                global_step=0,
                init_from="/a/last.pt",
                checkpoint_path="/new/last.pt",
            ),
            "[formal-boot] verified mode=fresh init_from=/a/last.pt step=0",
        )
        self.assertEqual(
            verify_expected_boot_mode(
                "continue",
                actual="continue",
                global_step=500,
                init_from="/a/last.pt",
                checkpoint_path="/new/last.pt",
            ),
            "[formal-boot] verified mode=continue checkpoint=/new/last.pt step=500",
        )
        with self.assertRaisesRegex(RuntimeError, "mode mismatch"):
            verify_expected_boot_mode(
                "fresh",
                actual="continue",
                global_step=500,
                init_from="/a/last.pt",
                checkpoint_path="/new/last.pt",
            )
        with tempfile.TemporaryDirectory() as td:
            last = os.path.join(td, "last.pt")
            self.assertEqual(
                resolve_checkpoint_boot_branch(
                    last,
                    resume=True,
                    init_from="/a/last.pt",
                    expected_mode="fresh",
                    is_dist=False,
                ),
                "fresh",
            )
            Path(last).write_bytes(b"checkpoint")
            self.assertEqual(
                resolve_checkpoint_boot_branch(
                    last,
                    resume=True,
                    init_from="/a/last.pt",
                    expected_mode="continue",
                    is_dist=False,
                ),
                "continue",
            )
            with self.assertRaisesRegex(RuntimeError, "branch mismatch before checkpoint load"):
                resolve_checkpoint_boot_branch(
                    last,
                    resume=True,
                    init_from="/a/last.pt",
                    expected_mode="fresh",
                    is_dist=False,
                )

    def test_all_mot_model_cuda_callers_configure_rng_after_to(self):
        callers = [
            REPO / "train.py",
            REPO / "evaluation/runners/fixed_sigma_eval.py",
        ]
        for path in callers:
            source = path.read_text(encoding="utf-8")
            self.assertIn("configure_training_rng", source, str(path))
        fixed_sigma = callers[1].read_text(encoding="utf-8")
        self.assertIn("vae_generator.manual_seed(args.seed)", fixed_sigma)


if __name__ == "__main__":
    unittest.main(verbosity=2)
