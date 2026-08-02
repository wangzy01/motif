"""Runtime reader for the immutable weighted Stage512a offline plan."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from training.dataline_offline import planner_v1 as base
# Single source for the per-stage horizon.  Duplicating the numbers here is how the
# builder and the reader would drift apart -- the builder would happily produce a plan
# the reader then refuses.
from training.dataline_offline.weighted_plan_v1 import FORMAL_HORIZONS
from training.dataline_offline.weighted_plan_v1 import (
    BLOCK_DATASETS,
    PLAN_FORMAT,
    PLANNER_ALGORITHM_VERSION,
    REAL_DOMAINS,
    _canonical_bytes,
)


class WeightedPlanReadError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, *, mode: int | None = None) -> os.stat_result:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise WeightedPlanReadError(f"unsafe plan file: {path}")
    if mode is not None and stat.S_IMODE(st.st_mode) != mode:
        raise WeightedPlanReadError(
            f"plan file mode drift: {path}: {stat.S_IMODE(st.st_mode):04o}"
        )
    return st


def _json_object(path: Path) -> dict[str, Any]:
    _regular(path, mode=0o444)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise WeightedPlanReadError(f"cannot parse plan JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WeightedPlanReadError(f"plan JSON is not an object: {path}")
    return value


def _lower_sha(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise WeightedPlanReadError(f"{path} is not a lowercase SHA-256")
    return value


class WeightedOfflinePlan:
    def __init__(
        self,
        plan_root: str | os.PathLike[str],
        *,
        world_size: int,
        per_rank_bs: int | None,
        expected_plan_hash: str,
        expected_contract_sha256: str,
        verify_checksums: bool = True,
        allow_canary: bool = False,
    ):
        self.plan_root = Path(plan_root)
        root_st = self.plan_root.lstat()
        if (
            not stat.S_ISDIR(root_st.st_mode)
            or self.plan_root.is_symlink()
            or stat.S_IMODE(root_st.st_mode) != 0o555
        ):
            raise WeightedPlanReadError("weighted plan root is not sealed mode0555")
        if {path.name for path in self.plan_root.iterdir()} != {
            "header.json",
            "checksums.json",
            "blocks",
        }:
            raise WeightedPlanReadError("weighted plan root member-set drift")
        blocks_st = (self.plan_root / "blocks").lstat()
        if not stat.S_ISDIR(blocks_st.st_mode) or stat.S_IMODE(blocks_st.st_mode) != 0o555:
            raise WeightedPlanReadError("weighted plan blocks directory is not sealed")
        self.header = _json_object(self.plan_root / "header.json")
        checksums = _json_object(self.plan_root / "checksums.json")
        self.config = self._validate_header(
            self.header,
            world_size=world_size,
            per_rank_bs=per_rank_bs,
            expected_plan_hash=expected_plan_hash,
            expected_contract_sha256=expected_contract_sha256,
            allow_canary=allow_canary,
        )
        if set(checksums) != {"header", "blocks"}:
            raise WeightedPlanReadError("checksums.json key-set drift")
        expected_header_sha = _lower_sha(checksums["header"], "checksums.header")
        if _sha256_file(self.plan_root / "header.json") != expected_header_sha:
            raise WeightedPlanReadError("weighted plan header checksum mismatch")
        block_map = checksums["blocks"]
        if not isinstance(block_map, Mapping):
            raise WeightedPlanReadError("checksums.blocks is not an object")
        first = int(self.header["materialization"]["first_materialized_block"])
        expected_names = {
            f"block_{index:06d}.npz"
            for index in range(first, int(self.header["num_blocks"]))
        }
        if set(block_map) != expected_names:
            raise WeightedPlanReadError("weighted plan block member set drift")
        actual_names = {path.name for path in (self.plan_root / "blocks").iterdir()}
        if actual_names != expected_names:
            raise WeightedPlanReadError("weighted plan blocks directory member drift")
        self.checksums: dict[str, str] = {}
        for name, value in block_map.items():
            self.checksums[str(name)] = _lower_sha(value, f"checksums.blocks.{name}")
            _regular(self.plan_root / "blocks" / str(name), mode=0o444)
        manifest_sha = hashlib.sha256(_canonical_bytes(dict(block_map)) + b"\n").hexdigest()
        if manifest_sha != self.header["hashes"].get("blocks_manifest_sha256"):
            raise WeightedPlanReadError("weighted block checksum manifest hash drift")
        self.verify_checksums = bool(verify_checksums)
        self._verified: set[int] = set()
        self._cache_index: int | None = None
        self._cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    @staticmethod
    def _validate_header(
        header: Mapping[str, Any],
        *,
        world_size: int,
        per_rank_bs: int | None,
        expected_plan_hash: str,
        expected_contract_sha256: str,
        allow_canary: bool,
    ) -> base.PlannerConfig:
        if header.get("plan_format") != PLAN_FORMAT:
            raise WeightedPlanReadError("weighted plan format mismatch")
        if header.get("planner_algo_version") != PLANNER_ALGORITHM_VERSION:
            raise WeightedPlanReadError("weighted planner algorithm mismatch")
        if header.get("algorithm_version") != PLANNER_ALGORITHM_VERSION:
            raise WeightedPlanReadError("weighted algorithm version mismatch")
        if header.get("block_datasets") != list(BLOCK_DATASETS):
            raise WeightedPlanReadError("weighted block dataset contract mismatch")
        planner_input = header.get("planner_input")
        if not isinstance(planner_input, Mapping):
            raise WeightedPlanReadError("missing embedded planner input")
        try:
            cfg = base._normalize_input(planner_input)
        except Exception as exc:
            raise WeightedPlanReadError(f"embedded planner input is invalid: {exc}") from exc
        if cfg.stage not in FORMAL_HORIZONS or cfg.image_size != 512:
            raise WeightedPlanReadError(
                f"weighted plan stage drift: stage={cfg.stage!r} not in "
                f"{sorted(FORMAL_HORIZONS)}, image_size={cfg.image_size}"
            )
        formal = (cfg.M, cfg.total_steps, cfg.G) == FORMAL_HORIZONS[cfg.stage]
        canary = (
            allow_canary
            and "canary" in cfg.plan_id
            and cfg.M <= 1_000_000
            and cfg.total_steps <= 10_000
            and cfg.world_size <= 16
        )
        if not formal and not canary:
            raise WeightedPlanReadError("weighted plan formal horizon drift")
        if header.get("config_hash") != cfg.config_sha256:
            raise WeightedPlanReadError("weighted embedded config hash drift")
        if cfg.world_size != world_size or (
            per_rank_bs is not None and cfg.per_rank_bs != per_rank_bs
        ):
            raise WeightedPlanReadError("weighted plan W/per-rank batch mismatch")
        expected_plan_hash = _lower_sha(expected_plan_hash, "expected_plan_hash")
        expected_contract_sha256 = _lower_sha(
            expected_contract_sha256, "expected_contract_sha256"
        )
        if header.get("plan_hash") != expected_plan_hash:
            raise WeightedPlanReadError("weighted plan identity differs from launch binding")
        if header.get("weighted_contract_sha256") != expected_contract_sha256:
            raise WeightedPlanReadError("weighted contract differs from launch binding")
        hashes = header.get("hashes")
        binding = header.get("weighted_binding")
        producers = header.get("producer_sources")
        if not isinstance(hashes, Mapping) or not isinstance(binding, Mapping):
            raise WeightedPlanReadError("weighted header hashes/binding missing")
        if not isinstance(producers, Mapping) or binding.get("producer_sources") != producers:
            raise WeightedPlanReadError("weighted producer source binding drift")
        for name, value in producers.items():
            _lower_sha(value, f"producer_sources.{name}")
        current_sources = {
            "weighted_plan_v1.py": _sha256_file(
                Path(__file__).with_name("weighted_plan_v1.py").resolve(strict=True)
            ),
            "weighted_streams_v1.py": _sha256_file(
                Path(__file__).with_name("weighted_streams_v1.py").resolve(strict=True)
            ),
            "planner_v1.py": _sha256_file(Path(base.__file__).resolve(strict=True)),
        }
        if dict(producers) != current_sources:
            raise WeightedPlanReadError("live weighted producer source drift")
        if binding.get("weighted_contract_sha256") != expected_contract_sha256:
            raise WeightedPlanReadError("weighted binding contract SHA drift")
        if binding.get("blocks_manifest_sha256") != hashes.get("blocks_manifest_sha256"):
            raise WeightedPlanReadError("weighted binding block manifest drift")
        if binding.get("block_datasets") != list(BLOCK_DATASETS):
            raise WeightedPlanReadError("weighted binding dataset drift")
        if binding.get("format") != PLAN_FORMAT or binding.get("algorithm") != PLANNER_ALGORITHM_VERSION:
            raise WeightedPlanReadError("weighted binding algorithm drift")
        recomputed_identity = hashlib.sha256(_canonical_bytes(dict(binding))).hexdigest()
        if recomputed_identity != expected_plan_hash:
            raise WeightedPlanReadError("weighted plan identity does not recompute")
        if hashes.get("plan_identity_sha256") != expected_plan_hash:
            raise WeightedPlanReadError("weighted header hash identity drift")
        seed_sha = base._seed_digest("plan_seed_identity", cfg).hex()
        expected_base_identity = base._plan_identity_sha256(
            cfg, base._algorithm_versions(cfg), seed_sha
        )
        if header.get("base_plan_identity_sha256") != expected_base_identity:
            raise WeightedPlanReadError("base quota-plan identity drift")
        if binding.get("base_plan_identity_sha256") != expected_base_identity:
            raise WeightedPlanReadError("weighted/base plan binding drift")
        D = np.asarray(header.get("D"), dtype=np.int64)
        S = np.asarray(header.get("S"), dtype=np.int64)
        C = np.asarray(header.get("C"), dtype=np.int64)
        if D.shape != (cfg.K,) or S.shape != (cfg.B,) or C.shape != (cfg.B, cfg.K):
            raise WeightedPlanReadError("weighted quota shape drift")
        expected_D = base._derive_domain_totals(cfg)
        allowed = base._allowed_mask(cfg)
        caps = base._static_caps(cfg, allowed)
        if not np.array_equal(D, expected_D):
            raise WeightedPlanReadError("weighted domain total drift")
        violations = base._quota_violations(cfg, D, S, C, allowed, caps)
        if violations:
            raise WeightedPlanReadError(f"weighted quota invariant failure: {violations[0]}")
        if header.get("N") != cfg.N.tolist() or header.get("domains") != list(cfg.domains):
            raise WeightedPlanReadError("weighted catalog mapping header drift")
        if header.get("total_exposures") != cfg.M:
            raise WeightedPlanReadError("weighted exposure count drift")
        return cfg

    def _verify_block(self, index: int) -> None:
        if index in self._verified:
            return
        name = f"block_{index:06d}.npz"
        actual = _sha256_file(self.plan_root / "blocks" / name)
        if actual != self.checksums[name]:
            raise WeightedPlanReadError(f"weighted plan block checksum mismatch: {name}")
        self._verified.add(index)

    def _load_block(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._cache_index == index and self._cache is not None:
            return self._cache
        first = int(self.header["materialization"]["first_materialized_block"])
        if index < first or index >= int(self.header["num_blocks"]):
            raise IndexError(f"weighted plan block outside materialized range: {index}")
        if self.verify_checksums:
            self._verify_block(index)
        path = self.plan_root / "blocks" / f"block_{index:06d}.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(BLOCK_DATASETS):
                    raise WeightedPlanReadError("weighted block NPZ member drift")
                bucket = np.array(archive["bucket_of_step"], copy=True)
                rows = np.array(archive["item_row"], copy=True)
                bands = np.array(archive["caption_band"], copy=True)
                counts = np.array(archive["domain_counts"], copy=True)
        except WeightedPlanReadError:
            raise
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise WeightedPlanReadError(f"cannot load weighted block {path}: {exc}") from exc
        start = index * self.config.block_steps
        steps = min(self.config.block_steps, self.config.total_steps - start)
        shape = (steps, self.config.world_size, self.config.per_rank_bs)
        if bucket.dtype != np.dtype(np.int16) or bucket.shape != (steps,):
            raise WeightedPlanReadError("weighted block bucket shape/dtype drift")
        if rows.dtype != np.dtype(np.int64) or rows.shape != shape:
            raise WeightedPlanReadError("weighted block item row shape/dtype drift")
        if bands.dtype != np.dtype(np.int8) or bands.shape != shape:
            raise WeightedPlanReadError("weighted block caption band shape/dtype drift")
        if counts.dtype != np.dtype(np.int32) or counts.shape != (steps, self.config.K):
            raise WeightedPlanReadError("weighted block domain count shape/dtype drift")
        if np.any(counts < 0):
            raise WeightedPlanReadError("weighted block has negative domain counts")
        if np.any(bucket < 0) or np.any(bucket >= self.config.B):
            raise WeightedPlanReadError("weighted block bucket range drift")
        if not np.all(counts.sum(axis=1, dtype=np.int64) == self.config.G):
            raise WeightedPlanReadError("weighted block domain counts do not sum to G")
        row_bases = base._cell_row_bases(self.config)
        for local in range(steps):
            b = int(bucket[local])
            flat_rows = rows[local].T.reshape(-1)
            flat_bands = bands[local].T.reshape(-1)
            if int(np.unique(flat_rows).size) != self.config.G:
                raise WeightedPlanReadError("weighted block duplicate global row")
            boundaries = np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(counts[local], dtype=np.int64))
            )
            for d, domain in enumerate(self.config.domains):
                lo = int(boundaries[d])
                hi = int(boundaries[d + 1])
                values = flat_rows[lo:hi]
                planned_bands = flat_bands[lo:hi]
                row_lo = int(row_bases[b, d])
                row_hi = row_lo + int(self.config.N[b, d])
                if np.any(values < row_lo) or np.any(values >= row_hi):
                    raise WeightedPlanReadError("weighted row/cell mapping drift")
                if domain in REAL_DOMAINS:
                    if np.any(planned_bands < 0) or np.any(planned_bands > 3):
                        raise WeightedPlanReadError("real row lacks a valid planned band")
                elif np.any(planned_bands != -1):
                    raise WeightedPlanReadError("non-real row has a caption band")
        for array in (bucket, rows, bands, counts):
            array.setflags(write=False)
        self._cache_index = index
        self._cache = (bucket, rows, bands, counts)
        return self._cache

    def verify_all_checksums(self) -> None:
        first = int(self.header["materialization"]["first_materialized_block"])
        for index in range(first, int(self.header["num_blocks"])):
            self._verify_block(index)

    def batch(self, step: int, rank: int) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
        if step < 0 or step >= self.config.total_steps:
            raise IndexError("weighted plan step outside range")
        if rank < 0 or rank >= self.config.world_size:
            raise IndexError("weighted plan rank outside range")
        index, offset = divmod(step, self.config.block_steps)
        bucket, rows, bands, _counts = self._load_block(index)
        b = int(bucket[offset])
        return self.config.buckets[b], rows[offset, rank], bands[offset, rank]

    def seek_after_completed(self, completed_optimizer_step: int) -> int | None:
        if completed_optimizer_step < -1 or completed_optimizer_step >= self.config.total_steps:
            raise WeightedPlanReadError("completed optimizer step outside range")
        next_step = completed_optimizer_step + 1
        self._cache_index = None
        self._cache = None
        if next_step == self.config.total_steps:
            return None
        self._load_block(next_step // self.config.block_steps)
        return next_step


__all__ = ["WeightedOfflinePlan", "WeightedPlanReadError"]
