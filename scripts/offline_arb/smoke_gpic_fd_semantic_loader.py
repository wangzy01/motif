#!/usr/bin/env python3
"""Thin semantic-smoke wrapper that attests resolved GPIC concurrency per rank.

The underlying smoke implementation and its hash calculation remain the single
existing ``smoke_offline_loader`` implementation.  This wrapper only replaces
the multiprocessing target with a shim that observes the production dataset
immediately after its real constructor returns, emits one atomic identity line,
and then lets the original worker continue unchanged.
"""

from __future__ import annotations

import json
import os

from scripts.offline_arb import smoke_offline_loader as base


ORIGINAL_WORKER = base._worker
MARKER = "GPIC_FD_RUNTIME_IDENTITY "


def identity_worker(*args, **kwargs) -> None:
    import training.dataline_offline.offline_dataset as module

    expected_mode = os.environ.get("GPIC_FD_EXPECTED_MODE")
    expected_threads = int(os.environ.get("GPIC_FD_EXPECTED_THREADS", "-1"))
    expected_workers = int(os.environ.get("GPIC_FD_EXPECTED_WORKERS_PER_RANK", "-1"))
    state_token = os.environ.get("GPIC_FD_STATE_TOKEN", "")
    if len(state_token) != 64 or any(ch not in "0123456789abcdef" for ch in state_token):
        raise AssertionError("GPIC_FD_STATE_TOKEN is malformed")
    real_init = module.OfflineExactDataset.__init__

    def observed_init(self, *init_args, **init_kwargs):
        real_init(self, *init_args, **init_kwargs)
        evidence = {
            "rank": int(self.rank),
            "world_size": int(self.world_size),
            "local_rank": int(self._local_rank),
            "local_world_size": int(self._local_world),
            "gpic_source_consistency_mode": self._gpic_source_consistency_mode,
            "gpic_threads_per_node": int(self._gpic_threads_per_node),
            "gpic_workers_per_rank": int(self._gpic_workers),
            "gpic_payload_in_memory": self._gpic_payload_in_memory,
            "attempt_state_token": state_token,
            "pid": os.getpid(),
        }
        if (
            evidence["gpic_source_consistency_mode"] != expected_mode
            or evidence["gpic_threads_per_node"] != expected_threads
            or evidence["gpic_workers_per_rank"] != expected_workers
            or evidence["gpic_payload_in_memory"] is not True
            or evidence["world_size"] != 4
            or evidence["local_world_size"] != 4
            or evidence["local_rank"] != evidence["rank"]
        ):
            raise AssertionError(f"GPIC runtime identity mismatch: {evidence}")
        raw = (MARKER + json.dumps(evidence, sort_keys=True) + "\n").encode("utf-8")
        if len(raw) > 4096:
            raise AssertionError("GPIC runtime identity record exceeds atomic line budget")
        written = os.write(1, raw)
        if written != len(raw):
            raise OSError(f"short GPIC runtime identity write: {written}/{len(raw)}")

    module.OfflineExactDataset.__init__ = observed_init
    ORIGINAL_WORKER(*args, **kwargs)


def main() -> int:
    if os.environ.get("GPIC_FD_EXPECTED_MODE") not in {"strict_path", "fd_stable"}:
        raise SystemExit("FATAL: GPIC_FD_EXPECTED_MODE is missing or invalid")
    if os.environ.get("GPIC_FD_EXPECTED_THREADS") != "16":
        raise SystemExit("FATAL: GPIC_FD_EXPECTED_THREADS must be 16")
    if os.environ.get("GPIC_FD_EXPECTED_WORKERS_PER_RANK") != "4":
        raise SystemExit("FATAL: GPIC_FD_EXPECTED_WORKERS_PER_RANK must be 4")
    base._worker = identity_worker
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
