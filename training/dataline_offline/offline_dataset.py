"""Exact, plan-driven OFFLINE_ARB_V1 training loader.

The loader deliberately has no fallback sampler.  A plan slot names exactly one
catalog row, and any catalog, locator, decode, or bucket mismatch is fatal.
Native-size mismatch is also fatal except for anime HTTP-served metadata when
the decoded image can fill the frozen target bucket without upsampling, or
when an otherwise-fatal upsample is bound back to the exact source-Parquet
row and its decoded-RGB MD5.  Raw bytes are fetched in deterministic plan
windows; decode and the single native-RGB -> target-bucket resize are prepared
ahead in the existing supply threads so that identical image math overlaps
GPU work.

The public surface mirrors ``CompleteV1Dataset`` as consumed by ``train.py``:

* ``loader(batch_size)`` yields ``{prompts, pixels, cfg_keep}``;
* ``set_start_step(global_step)`` seeks to that absolute, zero-based plan step;
* ``buffer_fill`` is a flat numeric metrics mapping.

Caption finalization and CFG protection are invoked directly from
``CompleteV1Dataset``.  Fetch scheduling is new because the old producer/buffer
path is intentionally not expressible here.
"""

from __future__ import annotations

import bisect
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import queue
import random
import re
import shutil
import stat
import threading
import time
import traceback
import warnings
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch

from training.dataline.complete_dataset import (
    CompleteV1Dataset,
    _ANIME_CAP_KEYS,
    _MONET_SRC_COL,
)
from training.dataline import anchors_v1
from training.dataline.caption_build import (
    build_textatlas_caption,
    compose_textatlas_gt_prompt,
)
from training.dataline.config import BANDS, load_sampling_config
from training.dataline.image_fetch import (
    decode_rgb,
    fetch_stats_snapshot,
    fetch_url_bytes,
    uint8_to_norm_tensor,
)
from training.dataline.manifest_index import CAPTIONER_CODE
from training.dataline.multi_ar import resize_u8_to_bucket
from training.dataline.pil_safety import PIL_NATIVE_LOCK
from training.dataline_offline.caption_sidecar import (
    CaptionSidecarError,
    CaptionSidecarStore,
)
from training.dataline_offline.caption_exceptions_v1 import (
    MONET_CAPTION_CODES,
    CaptionExceptionBindingError,
    bind_caption_exceptions,
)
from training.dataline_offline.gpic_blob import (
    CatalogBinding as GpicBlobCatalogBinding,
    GpicBlobError,
    GpicBlobOverlay,
)
from training.dataline_offline.image_exceptions_v1 import (
    ImageExceptionBindingError,
    ImageExceptionMismatchError,
    apply_textatlas_jpeg_eoi_repair,
    bind_image_exceptions,
)
from training.dataline_offline.anime_native_upsample_exceptions_weighted_v1 import (
    AnimeNativeUpsampleBindingError,
    AnimeNativeUpsampleMismatchError,
    apply_anime_native_upsample_exception,
    bind_anime_native_upsample_exceptions,
)
from training.dataline_offline.textatlas_gt_v1 import (
    TextAtlasGroundTruth,
    TextAtlasGroundTruthError,
)
from training.dataline_offline.weighted_plan_reader_v1 import WeightedOfflinePlan


_MIB = 1 << 20
_COALESCE_GAP_BYTES = 4 * _MIB
_COALESCE_MAX_RUN_BYTES = 256 * _MIB
_COALESCE_CHUNK_BYTES = 8 * _MIB
_DEFAULT_PREFETCH_STEPS = 4
_SUPPLY_WINDOW_STEPS = 1
_GPIC_THREADS_PER_NODE = 32
_GPIC_SOURCE_CONSISTENCY_MODES = frozenset({"strict_path", "fd_stable"})
_GPIC_PAYLOAD_BACKENDS = frozenset({"source_files_v1", "offline_gpic_blob_v1"})
_TEXTATLAS_ROWGROUP_WORKERS = 8
_TELEMETRY_SCHEMA = "offline_arb_io_profile_v1"
# ``fetch_url_bytes`` retains the streaming loader's short retry policy, whose
# non-retryable 4xx path assumes a failed row can be encountered again after a
# shard reshuffle.  An exact plan cannot reshuffle or replace its row, so give
# the *same URL* four longer, bounded refetch opportunities.  The final attempt
# occurs after 85 seconds for immediately returned 4xx responses; retryable
# network errors keep their existing inner 3x/20s policy.  The combined
# worst-case remains below the loader's default 600-second step timeout.
_ANIME_EXACT_REFETCH_DELAYS_S = (1.0, 4.0, 16.0, 64.0)
# Per-rank ceiling on anime metadata-drift rows whose crop_loss against the DECODED
# pixels exceeds the threshold their anchor was assigned under.  Such a row is usable
# (crop_loss_max is a corpus-composition constraint enforced offline, at plan build
# time); what is NOT acceptable is tolerating them at an unbounded rate, because a
# systematically wrong row->file mapping would present as exactly this signal.  So it
# is a circuit breaker, not a mute button, and every occurrence prints.
#
# Derived, not guessed.  The 2026-08-04 formal run measured 1 such row in 660 steps x
# 64 ranks x 32 slots = 1,351,680 decoded samples (7.4e-7).  Over the full plan each
# rank decodes 512,000,000 / 64 = 8,000,000 rows, so the expectation is ~6 per rank.
# 2000 leaves ~340x headroom for a rate estimate drawn from a single event, while a
# real mapping fault -- which would hit a large fraction of rows -- trips it within
# minutes instead of training on mismatched image/caption pairs for seven days.
_ANIME_DRIFT_CROP_TOLERATED_MAX = 2000
_ANIME_CAPTION_SOURCE_PARENT = Path("/s3/evan/anime_core_ditv04_ct_soft/data")
_ANIME_SOURCE_IMAGE_BINDING_COLUMNS = (
    "md5",
    "width",
    "height",
    "key",
    "image_size",
    "s3key",
)
_CAPTION_SOURCE_COLUMNS = {
    # ``rating_bucket`` and ``year`` are frozen catalog fields and are injected
    # below after the caption lookup; copying them into the sidecar would create
    # a redundant second authority.
    "anime": tuple(
        key for key in dict.fromkeys(_ANIME_CAP_KEYS) if key not in {"rating_bucket", "year"}
    ),
    "monet_real": tuple(dict.fromkeys(_MONET_SRC_COL.values())),
    "gpic": ("caption",),
    "synthetic": ("caption_original",),
    "textatlas": tuple(f"caption{i}" for i in range(1, 6)),
}

_COMMON_CATALOG_TYPES = {
    "row_id": "int64",
    "uid": "string",
    "domain": "string",
    "bucket_idx": "int16",
    "width": "int32",
    "height": "int32",
}
_REAL_CONDITION_TYPES = {
    "src_key": "string",
    "sub_source": "string",
    "cluster_id": "int64",
    "caption_type": "string",
    "caption_tier": "int8",
    "served_mask": "uint8",
    "band_captioner_t1": "int8",
    "band_captioner_t2": "int8",
    "band_captioner_t3": "int8",
    "band_captioner_t4": "int8",
    "base_weight": "double",
}
_CAPTION_LOCATOR_TYPES = {
    "caption_parquet_path": "string",
    "caption_row_group": "int32",
    "caption_row_in_group": "int32",
}
_DOMAIN_CATALOG_TYPES = {
    "anime": {
        "key": "string",
        "fetch_url": "string",
        "cluster": "int64",
        "rating_bucket": "string",
        "year": "int32",
        **_CAPTION_LOCATOR_TYPES,
    },
    "monet_real": {
        "tar_path": "string",
        "tar_offset": "int64",
        "tar_size": "int64",
        **_REAL_CONDITION_TYPES,
        **_CAPTION_LOCATOR_TYPES,
    },
    "gpic": {
        "image_path": "string",
        **_REAL_CONDITION_TYPES,
        **_CAPTION_LOCATOR_TYPES,
    },
    "synthetic": {
        "tar_path": "string",
        "tar_offset": "int64",
        "tar_size": "int64",
        **_REAL_CONDITION_TYPES,
        "caption_kind": "string",
        **_CAPTION_LOCATOR_TYPES,
    },
    "textatlas": {
        "image_path": "string",
        "image_parquet_path": "string",
        "image_row_group": "int32",
        "image_row_in_group": "int32",
        **_CAPTION_LOCATOR_TYPES,
    },
}


class OfflineLoaderError(RuntimeError):
    """The exact plan cannot be consumed without changing its semantics."""


@dataclass(frozen=True)
class _OfflineStageIdentity:
    """Frozen runtime identity for one supported offline curriculum stage."""

    stage: str
    image_size: int
    eligibility: str
    assignment_eligibility: str
    legacy_artifacts_allowed: bool
    requires_nl: bool | None
    anime_nl_filter_applied: bool


# ``eligibility`` is the public artifact contract.  The stage256 assignment
# hash predates that contract and must retain its historical internal token
# byte-for-byte; do not expose that legacy token to sidecars/exceptions.
_OFFLINE_STAGE_IDENTITIES = {
    "256": _OfflineStageIdentity(
        stage="256",
        image_size=256,
        eligibility="elig_256",
        assignment_eligibility="phase3a_stage256_v1",
        legacy_artifacts_allowed=True,
        requires_nl=True,
        anime_nl_filter_applied=False,
    ),
    "512a": _OfflineStageIdentity(
        stage="512a",
        image_size=512,
        eligibility="elig_512",
        assignment_eligibility="elig_512",
        legacy_artifacts_allowed=False,
        requires_nl=False,
        anime_nl_filter_applied=False,
    ),
    # Stage512b-real: same resolution and eligibility strings as 512a, but
    # bucket_mode=anchors_v1 -- 13 fixed anchors instead of 361 adaptive buckets.
    # The geometry difference is carried by bucket_table_hash and by the extra
    # anchors keys in the assignment payload, not by these fields.
    "512b": _OfflineStageIdentity(
        stage="512b",
        image_size=512,
        eligibility="elig_512",
        assignment_eligibility="elig_512",
        legacy_artifacts_allowed=False,
        requires_nl=False,
        anime_nl_filter_applied=False,
    ),
}


def _resolve_offline_stage_identity(
    stage: Any,
    image_size: Any,
    multi_ar: Mapping[str, Any],
) -> _OfflineStageIdentity:
    normalized_stage = str(stage)
    identity = _OFFLINE_STAGE_IDENTITIES.get(normalized_stage)
    if identity is None:
        raise OfflineLoaderError(
            "OfflineExactDataset has no frozen artifact contract for "
            f"stage={normalized_stage!r}; supported={sorted(_OFFLINE_STAGE_IDENTITIES)}"
        )

    if image_size is None:
        if not identity.legacy_artifacts_allowed:
            raise OfflineLoaderError(
                f"stage {normalized_stage!r} requires explicit dataset.image_size="
                f"{identity.image_size}"
            )
    elif type(image_size) is not int or image_size != identity.image_size:
        raise OfflineLoaderError(
            f"stage {normalized_stage!r} requires dataset.image_size="
            f"{identity.image_size}, got {image_size!r}"
        )

    target_area = multi_ar.get("target_area")
    expected_area = identity.image_size * identity.image_size
    if target_area is None:
        if not identity.legacy_artifacts_allowed:
            raise OfflineLoaderError(
                f"stage {normalized_stage!r} requires "
                f"dataset.multi_ar.target_area={expected_area}"
            )
    elif type(target_area) is not int or target_area != expected_area:
        raise OfflineLoaderError(
            f"stage {normalized_stage!r} requires dataset.multi_ar.target_area="
            f"{expected_area}, got {target_area!r}"
        )
    return identity


def _validate_artifact_stage_identity(
    header: Mapping[str, Any],
    *,
    artifact: str,
    identity: _OfflineStageIdentity,
    requires_nl: bool,
) -> bool:
    """Validate a plan/catalog stage identity; return whether it was explicit."""

    if identity.requires_nl is not None and requires_nl is not identity.requires_nl:
        raise OfflineLoaderError(
            f"runtime stage {identity.stage!r} requires requires_nl="
            f"{identity.requires_nl!r}, got {requires_nl!r} from sampling_config"
        )

    identity_keys = {
        "stage",
        "image_size",
        "eligibility",
        "requires_nl",
        "anime_nl_filter_applied",
    }
    present = identity_keys.intersection(header)
    if not present:
        if identity.legacy_artifacts_allowed:
            return False
        raise OfflineLoaderError(
            f"{artifact} lacks required stage/eligibility identity for "
            f"runtime stage {identity.stage!r}"
        )

    required = {
        "stage",
        "image_size",
        "eligibility",
        "requires_nl",
        "anime_nl_filter_applied",
    }
    missing = required - set(header)
    if missing:
        raise OfflineLoaderError(
            f"{artifact} has a partial stage identity; missing={sorted(missing)}"
        )
    if type(header.get("stage")) is not str or header["stage"] != identity.stage:
        raise OfflineLoaderError(
            f"{artifact}/runtime stage mismatch: artifact={header.get('stage')!r}, "
            f"runtime={identity.stage!r}"
        )
    if (
        type(header.get("eligibility")) is not str
        or header["eligibility"] != identity.eligibility
    ):
        raise OfflineLoaderError(
            f"{artifact}/runtime eligibility mismatch: "
            f"artifact={header.get('eligibility')!r}, "
            f"runtime={identity.eligibility!r}"
        )
    if (
        type(header["image_size"]) is not int
        or header["image_size"] != identity.image_size
    ):
        raise OfflineLoaderError(
            f"{artifact}/runtime image_size mismatch: "
            f"artifact={header.get('image_size')!r}, runtime={identity.image_size}"
        )
    if (
        type(header["requires_nl"]) is not bool
        or header["requires_nl"] is not requires_nl
    ):
        raise OfflineLoaderError(
            f"{artifact}/runtime requires_nl mismatch: "
            f"artifact={header.get('requires_nl')!r}, runtime={requires_nl!r}"
        )
    if (
        type(header["anime_nl_filter_applied"]) is not bool
        or header["anime_nl_filter_applied"]
        is not identity.anime_nl_filter_applied
    ):
        raise OfflineLoaderError(
            f"{artifact}/runtime anime_nl_filter_applied mismatch: "
            f"artifact={header.get('anime_nl_filter_applied')!r}, "
            f"runtime={identity.anime_nl_filter_applied!r}"
        )
    return True


def _validate_plan_stage_identity(
    header: Mapping[str, Any],
    *,
    identity: _OfflineStageIdentity,
    requires_nl: bool,
) -> bool:
    explicit = _validate_artifact_stage_identity(
        header,
        artifact="plan header",
        identity=identity,
        requires_nl=requires_nl,
    )
    expected_format = 2 if explicit else 1
    if (
        type(header.get("format_version")) is not int
        or header["format_version"] != expected_format
    ):
        raise OfflineLoaderError(
            "plan stage identity/format mismatch: "
            f"explicit={explicit}, expected format_version={expected_format}, "
            f"got {header.get('format_version')!r}"
        )
    return explicit


def _assignment_identity_payload(
    header: Mapping[str, Any],
    *,
    identity: _OfflineStageIdentity,
    requires_nl: bool,
) -> dict[str, Any]:
    payload = {
        "bucket_table_hash": header["bucket_table_hash"],
        "floor": header.get("floor"),
        "kept": list(header.get("kept_bucket_indices", [])),
        "fold_map": list(header.get("fold_map", [])),
        "square_only_domains": list(header.get("square_only_domains", [])),
        "square_bucket_idx": header.get("square_bucket_idx"),
        # Mirrors phase5_build_catalog.assignment_hash_payload.  The two must agree
        # byte for byte after canonical JSON, so any change here needs the same
        # change there -- that equality is the only thing binding a loaded catalog
        # to the geometry that produced it.
        "assignment": (
            "anchors_v1_feasible_argmax_retention_v1"
            if _header_bucket_mode(header) == anchors_v1.BUCKET_MODE
            else "phase3a_vectorized_with_production_scalar_fallback_v1"
        ),
    }
    if identity.stage == "256":
        payload["eligibility"] = identity.assignment_eligibility
    else:
        payload.update(
            {
                "stage": identity.stage,
                "image_size": identity.image_size,
                "eligibility": identity.assignment_eligibility,
                "requires_nl": requires_nl,
            }
        )
    if _header_bucket_mode(header) == anchors_v1.BUCKET_MODE:
        # Every value here is derived from the HEADER, never from the running config.
        # That distinction is the whole point: recomputing the geometry from whatever
        # config this process was launched with would make a mismatched config
        # silently self-consistent, which is exactly what this hash exists to catch.
        #
        # phase5 does not store anchor_identity at the header top level -- it only
        # feeds it into the hash -- so it is rebuilt here from the multi_ar_params the
        # header does record.  anchors_v1.identity() is therefore part of the ABI: a
        # change to its text invalidates every catalog, which is correct, because its
        # text is a description of the assignment semantics.
        mar = header.get("inputs", {}).get("multi_ar_params", {})
        gate = mar.get("area_gate")
        payload["bucket_mode"] = anchors_v1.BUCKET_MODE
        payload["anchor_identity"] = anchors_v1.identity(
            crop_loss_max=mar["crop_loss_max"], area_gate=gate
        )
        payload["source_area_gate"] = None if gate is None else int(gate)
        payload["crop_loss_max"] = float(mar["crop_loss_max"])
        payload["crop_loss_max_by_domain"] = mar.get("crop_loss_max_by_domain")
    return payload


def _header_bucket_mode(header: Mapping[str, Any]) -> str | None:
    """The bucket mode a catalog was built with.

    phase5 records it under ``eligibility_audit`` rather than at the top level,
    because in anchors mode the eligibility rule *is* the geometry (source area gate
    plus a feasible-anchor test, no elig_* column at all).  Reading it through one
    helper keeps that quirk in a single place.
    """
    audit = header.get("eligibility_audit")
    if isinstance(audit, Mapping):
        mode = audit.get("bucket_mode")
        if isinstance(mode, str):
            return mode
    return None


def _validated_gpic_source_consistency_mode(value: Any) -> str:
    if not isinstance(value, str) or value not in _GPIC_SOURCE_CONSISTENCY_MODES:
        raise OfflineLoaderError(
            "gpic_source_consistency_mode must be exactly one of "
            f"{sorted(_GPIC_SOURCE_CONSISTENCY_MODES)}, got {value!r}"
        )
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise OfflineLoaderError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OfflineLoaderError(f"expected JSON object in {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * _MIB), b""):
                digest.update(block)
    except OSError as exc:
        raise OfflineLoaderError(f"cannot hash catalog file {path}: {exc}") from exc
    return digest.hexdigest()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise OfflineLoaderError(f"{label} must be an integer, got {value!r}")
    result = int(value)
    if result <= 0:
        raise OfflineLoaderError(f"{label} must be positive, got {result}")
    return result


def _strict_positive_env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    if re.fullmatch(r"[1-9][0-9]*", raw) is None:
        raise OfflineLoaderError(
            f"{name} must be a strict positive integer, got {raw!r}"
        )
    return int(raw)


def _resolved_gpic_threads_per_node(
    configured: Any,
    environment: int | None,
    local_world: int,
) -> int:
    """Resolve GPIC concurrency without allowing a silent config/env mismatch."""

    world = _positive_int(local_world, "LOCAL_WORLD_SIZE")
    configured_threads = (
        None
        if configured is None
        else _positive_int(configured, "gpic_threads_per_node")
    )
    if environment is not None:
        environment = _positive_int(
            environment, "OFFLINE_ARB_GPIC_THREADS_PER_NODE"
        )
    if (
        configured_threads is not None
        and environment is not None
        and configured_threads != environment
    ):
        raise OfflineLoaderError(
            "dataset.gpic_threads_per_node conflicts with "
            "OFFLINE_ARB_GPIC_THREADS_PER_NODE: "
            f"configured={configured_threads}, environment={environment}"
        )
    resolved = (
        configured_threads
        if configured_threads is not None
        else environment
        if environment is not None
        else _GPIC_THREADS_PER_NODE
    )
    if resolved < world:
        raise OfflineLoaderError(
            "gpic_threads_per_node must be at least LOCAL_WORLD_SIZE: "
            f"{resolved} < {world}"
        )
    if resolved % world != 0:
        raise OfflineLoaderError(
            "gpic_threads_per_node must be divisible by LOCAL_WORLD_SIZE: "
            f"{resolved} % {world} != 0"
        )
    return resolved


def _strict_positive_env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise OfflineLoaderError(
            f"{name} must be a positive finite number, got {raw!r}"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise OfflineLoaderError(
            f"{name} must be a positive finite number, got {raw!r}"
        )
    return parsed


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:96] or "plan"


class _VirtualRowDecoder:
    """Decode §9.3 ``base[d] + cellbase[d][B] + local_idx`` row IDs."""

    def __init__(self, domains: Sequence[str], n_matrix: Sequence[Sequence[int]]):
        self.domains = tuple(str(x) for x in domains)
        n = np.asarray(n_matrix)
        if n.ndim != 2 or n.shape[1] != len(self.domains):
            raise OfflineLoaderError(
                f"N must have shape [bucket,{len(self.domains)}], got {n.shape}"
            )
        if not np.issubdtype(n.dtype, np.integer) or np.any(n < 0):
            raise OfflineLoaderError("N must contain nonnegative integers")
        self.N = n.astype(np.int64, copy=True)
        self.domain_totals = self.N.sum(axis=0, dtype=np.int64)
        self.domain_ends = np.cumsum(self.domain_totals, dtype=np.int64)
        self.domain_bases = np.concatenate(
            (np.zeros(1, dtype=np.int64), self.domain_ends[:-1])
        )
        self.cell_ends = [np.cumsum(self.N[:, d], dtype=np.int64) for d in range(len(self.domains))]
        self.total_rows = int(self.domain_ends[-1]) if len(self.domain_ends) else 0

    def decode(self, row_id: int) -> tuple[int, int, int]:
        if isinstance(row_id, bool) or not isinstance(row_id, (int, np.integer)):
            raise OfflineLoaderError(f"row_id must be an integer, got {row_id!r}")
        row = int(row_id)
        if row < 0 or row >= self.total_rows:
            raise OfflineLoaderError(f"row_id {row} outside [0,{self.total_rows})")
        d = bisect.bisect_right(self.domain_ends, row)
        within_domain = row - int(self.domain_bases[d])
        b = bisect.bisect_right(self.cell_ends[d], within_domain)
        if b >= self.N.shape[0]:
            raise OfflineLoaderError(f"cannot decode row_id {row}: bucket overflow")
        cell_base = int(self.cell_ends[d][b - 1]) if b else 0
        local = within_domain - cell_base
        if local < 0 or local >= int(self.N[b, d]):
            raise OfflineLoaderError(f"cannot decode row_id {row}: empty/out-of-range cell")
        return d, b, int(local)


def decode_virtual_row_id(
    row_id: int,
    domains: Sequence[str],
    n_matrix: Sequence[Sequence[int]],
) -> tuple[str, int, int]:
    """Public, independently testable §9.3 decoder."""

    decoder = _VirtualRowDecoder(domains, n_matrix)
    d, bucket, local = decoder.decode(row_id)
    return decoder.domains[d], bucket, local


class _CatalogDomainReader:
    """One-domain catalog reader with footer-derived row-group binary search."""

    def __init__(
        self,
        domain: str,
        path: Path,
        row_base: int,
        row_count: int,
        declared: Mapping[str, Any],
    ):
        self.domain = domain
        self.path = path
        self.row_base = int(row_base)
        self.row_count = int(row_count)
        if not path.is_file():
            raise OfflineLoaderError(f"catalog file missing for {domain}: {path}")
        declared_bytes = declared.get("bytes")
        if isinstance(declared_bytes, int) and path.stat().st_size != declared_bytes:
            raise OfflineLoaderError(
                f"catalog byte-size mismatch for {domain}: "
                f"header={declared_bytes}, actual={path.stat().st_size}"
            )
        try:
            self._pf = pq.ParquetFile(path)
        except Exception as exc:
            raise OfflineLoaderError(f"cannot open catalog parquet {path}: {exc}") from exc
        if self._pf.metadata.num_rows != self.row_count:
            raise OfflineLoaderError(
                f"catalog row count mismatch for {domain}: "
                f"header={self.row_count}, parquet={self._pf.metadata.num_rows}"
            )
        self._validate_schema()
        sizes = [
            int(self._pf.metadata.row_group(i).num_rows)
            for i in range(self._pf.num_row_groups)
        ]
        if sum(sizes) != self.row_count:
            raise OfflineLoaderError(f"catalog footer row-group accounting drift for {domain}")
        self._rg_ends = [self.row_base + int(x) for x in np.cumsum(sizes, dtype=np.int64)]
        self._rg_starts = [self.row_base] + self._rg_ends[:-1]
        self._cache: "OrderedDict[int, Any]" = OrderedDict()

    def _validate_schema(self) -> None:
        required = {**_COMMON_CATALOG_TYPES, **_DOMAIN_CATALOG_TYPES[self.domain]}
        schema = self._pf.schema_arrow
        fields = {field.name: field for field in schema}
        missing = sorted(set(required) - set(fields))
        if missing:
            raise OfflineLoaderError(
                f"catalog schema for {self.domain} is missing fields {missing}: {self.path}"
            )
        wrong = {
            name: (expected, str(fields[name].type))
            for name, expected in required.items()
            if str(fields[name].type) != expected
        }
        if wrong:
            raise OfflineLoaderError(f"catalog schema type mismatch for {self.domain}: {wrong}")

    def _row_group(self, rg: int):
        table = self._cache.pop(rg, None)
        if table is None:
            try:
                table = self._pf.read_row_group(rg, use_threads=False)
            except Exception as exc:
                raise OfflineLoaderError(
                    f"cannot read catalog row group {self.path}:{rg}: {exc}"
                ) from exc
        self._cache[rg] = table
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return table

    def get_many(self, row_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        unique = sorted({int(x) for x in row_ids})
        grouped: dict[int, list[int]] = defaultdict(list)
        for row in unique:
            if row < self.row_base or row >= self.row_base + self.row_count:
                raise OfflineLoaderError(
                    f"row_id {row} is outside domain {self.domain} range "
                    f"[{self.row_base},{self.row_base + self.row_count})"
                )
            rg = bisect.bisect_right(self._rg_ends, row)
            if rg >= len(self._rg_ends):
                raise OfflineLoaderError(f"cannot locate catalog row_id {row} in {self.domain}")
            grouped[rg].append(row)

        result: dict[int, dict[str, Any]] = {}
        for rg, rows in grouped.items():
            table = self._row_group(rg)
            start = self._rg_starts[rg]
            for row in rows:
                offset = row - start
                if offset < 0 or offset >= table.num_rows:
                    raise OfflineLoaderError(
                        f"catalog row-group binary search produced invalid offset for row_id {row}"
                    )
                value = table.slice(offset, 1).to_pylist()[0]
                if int(value.get("row_id", -1)) != row:
                    raise OfflineLoaderError(
                        f"catalog physical row mismatch: requested={row}, got={value.get('row_id')}"
                    )
                if str(value.get("domain")) != self.domain:
                    raise OfflineLoaderError(
                        f"catalog domain mismatch at row_id {row}: {value.get('domain')!r}"
                    )
                result[row] = value
        return result


@dataclass(frozen=True)
class _PlannedItem:
    step: int
    slot: int
    row_id: int
    domain: str
    domain_idx: int
    bucket_idx: int
    caption_band: int


@dataclass(frozen=True)
class _SupplyWindow:
    generation: int
    start: int
    stop: int
    items: tuple[_PlannedItem, ...]


@dataclass
class _PreparedItem:
    planned: _PlannedItem
    catalog: dict[str, Any]
    caption_record: dict[str, Any]
    raw: bytes | None = None
    local_path: str | None = None
    bucket_u8: np.ndarray | None = None


class OfflineExactDataset:
    """OFFLINE_ARB_V1 exact dataset, compatible with the current store factory."""

    def __init__(
        self,
        sampling_config=None,
        stage="256",
        image_size=None,
        seed=0,
        rank=None,
        world_size=None,
        tokenizer_path=None,
        caption_token_budget=None,
        template_reserve=41,
        anime_cache_dir=None,
        fetch_workers_anime=1,
        fetch_workers_tar=8,
        crop_v_anchor=None,
        multi_ar=None,
        offline_plan_root=None,
        offline_plan_hash=None,
        offline_weighted_contract_sha256=None,
        offline_catalog_root=None,
        offline_anime_freq_root=None,
        offline_caption_sidecar_root=None,
        offline_textatlas_gt_root=None,
        caption_sidecar_workers=16,
        anime_caption_root=None,
        caption_workers_anime=1,
        caption_workers_monet_real=1,
        caption_workers_gpic=1,
        caption_workers_synthetic=1,
        gpic_payload_in_memory=False,
        gpic_source_consistency_mode="strict_path",
        gpic_threads_per_node=None,
        gpic_payload_backend="source_files_v1",
        offline_gpic_blob_root=None,
        gpic_blob_manifest_sha256=None,
        gpic_blob_fd_cache_size=64,
        **_,
    ):
        if not offline_plan_root or not offline_catalog_root:
            raise OfflineLoaderError(
                "dataset.offline_plan_root and dataset.offline_catalog_root are required"
            )
        if not sampling_config:
            raise OfflineLoaderError("dataset.sampling_config is required for caption semantics")
        multi_ar_cfg = dict(multi_ar or {})
        assignment_mode = str(multi_ar_cfg.get("assignment_mode", ""))
        if assignment_mode != "offline_weighted_band_v1":
            raise OfflineLoaderError(
                "OfflineExactDataset requires "
                "dataset.multi_ar.assignment_mode=offline_weighted_band_v1"
            )
        for name, value in (
            ("offline_plan_hash", offline_plan_hash),
            ("offline_weighted_contract_sha256", offline_weighted_contract_sha256),
        ):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise OfflineLoaderError(f"{name} must be a lowercase SHA-256")

        self.rank, self.world_size = self._resolve_rank(rank, world_size)
        self.seed = int(seed)
        self.stage = str(stage)
        self._stage_identity = _resolve_offline_stage_identity(
            self.stage, image_size, multi_ar_cfg
        )
        self.image_size = self._stage_identity.image_size
        self.eligibility = self._stage_identity.eligibility
        self.sampling_config_path = Path(str(sampling_config)).expanduser()
        self.cfg = load_sampling_config(str(self.sampling_config_path))
        if self.stage not in self.cfg.domain_mix:
            raise OfflineLoaderError(
                f"stage {self.stage!r} is absent from sampling config {sampling_config}"
            )
        self.requires_nl = bool(self.cfg.anime_caption.requires_nl(self.stage))
        self.anime_cache_dir = str(anime_cache_dir) if anime_cache_dir else None
        requested_anchors = {
            str(k): float(v) for k, v in dict(crop_v_anchor or {}).items()
        }
        if "anime" in requested_anchors and not np.isclose(
            requested_anchors["anime"], 0.15, rtol=0.0, atol=1e-12
        ):
            raise OfflineLoaderError(
                "OFFLINE_ARB_V1 freezes crop_v_anchor.anime=0.15 until the Phase-6 visual A/B"
            )
        self._crop_v_anchor = {"anime": 0.15, **requested_anchors}
        for domain, anchor in self._crop_v_anchor.items():
            if not 0.0 <= anchor <= 1.0:
                raise OfflineLoaderError(
                    f"crop_v_anchor[{domain!r}] must be in [0,1], got {anchor}"
                )

        self.tokenizer = None
        self.caption_budget = None
        if tokenizer_path:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.caption_budget = (
                int(caption_token_budget)
                if caption_token_budget
                else int(self.cfg.anime_caption.mtml_for(self.stage)) - int(template_reserve)
            )

        self.plan_root = Path(str(offline_plan_root)).expanduser()
        self.catalog_root = Path(str(offline_catalog_root)).expanduser()
        self._anime_freq_root = (
            Path(str(offline_anime_freq_root)).expanduser()
            if offline_anime_freq_root
            else None
        )
        self._textatlas_gt_root = (
            Path(str(offline_textatlas_gt_root)).expanduser()
            if offline_textatlas_gt_root
            else None
        )
        if offline_caption_sidecar_root is not None and not str(
            offline_caption_sidecar_root
        ).strip():
            raise OfflineLoaderError("offline_caption_sidecar_root must not be empty")
        self._caption_sidecar_root = (
            Path(str(offline_caption_sidecar_root)).expanduser()
            if offline_caption_sidecar_root is not None
            else None
        )
        self._caption_sidecar_workers = _positive_int(
            caption_sidecar_workers, "caption_sidecar_workers"
        )
        if not isinstance(gpic_payload_in_memory, bool):
            raise OfflineLoaderError(
                "gpic_payload_in_memory must be a boolean, got "
                f"{gpic_payload_in_memory!r}"
            )
        self._gpic_payload_in_memory = gpic_payload_in_memory
        if (
            not isinstance(gpic_payload_backend, str)
            or gpic_payload_backend not in _GPIC_PAYLOAD_BACKENDS
        ):
            raise OfflineLoaderError(
                "gpic_payload_backend must be exactly one of "
                f"{sorted(_GPIC_PAYLOAD_BACKENDS)}, got {gpic_payload_backend!r}"
            )
        self._gpic_payload_backend = gpic_payload_backend
        self._gpic_blob_root: Path | None = None
        self._gpic_blob_manifest_sha256: str | None = None
        self._gpic_blob_release_sha256: str | None = None
        self._gpic_blob: GpicBlobOverlay | None = None
        self._gpic_blob_fd_cache_size = _positive_int(
            gpic_blob_fd_cache_size, "gpic_blob_fd_cache_size"
        )
        if self._gpic_payload_backend == "offline_gpic_blob_v1":
            if not self._gpic_payload_in_memory:
                raise OfflineLoaderError(
                    "offline_gpic_blob_v1 requires gpic_payload_in_memory=true"
                )
            if offline_gpic_blob_root is None or not str(offline_gpic_blob_root).strip():
                raise OfflineLoaderError(
                    "offline_gpic_blob_v1 requires offline_gpic_blob_root"
                )
            if (
                not isinstance(gpic_blob_manifest_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", gpic_blob_manifest_sha256) is None
            ):
                raise OfflineLoaderError(
                    "offline_gpic_blob_v1 requires a lowercase "
                    "gpic_blob_manifest_sha256"
                )
            self._gpic_blob_root = Path(str(offline_gpic_blob_root)).expanduser()
            self._gpic_blob_manifest_sha256 = gpic_blob_manifest_sha256
        elif (
            offline_gpic_blob_root is not None
            or gpic_blob_manifest_sha256 is not None
        ):
            raise OfflineLoaderError(
                "GPIC blob root/manifest SHA were configured but "
                "gpic_payload_backend is source_files_v1"
            )
        self._gpic_source_consistency_mode = (
            _validated_gpic_source_consistency_mode(
                gpic_source_consistency_mode
            )
        )
        # A configured sidecar is the sole caption backend.  In that mode the
        # legacy anime mirror is deliberately irrelevant (and must never become
        # an accidental fallback dependency).
        if self._caption_sidecar_root is None:
            env_anime_caption_root = os.environ.get("OFFLINE_ARB_ANIME_CAPTION_ROOT")
            if env_anime_caption_root is not None:
                if not env_anime_caption_root.strip():
                    raise OfflineLoaderError(
                        "OFFLINE_ARB_ANIME_CAPTION_ROOT must not be empty"
                    )
                anime_caption_root = env_anime_caption_root
            if anime_caption_root is not None and not str(anime_caption_root).strip():
                raise OfflineLoaderError("anime_caption_root must not be empty")
            self._anime_caption_root = (
                Path(str(anime_caption_root)).expanduser()
                if anime_caption_root is not None
                else None
            )
            if self._anime_caption_root is not None and not self._anime_caption_root.is_dir():
                raise OfflineLoaderError(
                    f"anime caption root is not a directory: {self._anime_caption_root}"
                )
        else:
            self._anime_caption_root = None
        if self._anime_freq_root is not None and not (
            self._anime_freq_root / "_SUCCESS"
        ).is_file():
            raise OfflineLoaderError(
                f"anime frequency root lacks _SUCCESS: {self._anime_freq_root}"
            )
        self.plan = WeightedOfflinePlan(
            self.plan_root,
            world_size=self.world_size,
            per_rank_bs=None,
            expected_plan_hash=offline_plan_hash,
            expected_contract_sha256=offline_weighted_contract_sha256,
            verify_checksums=True,
        )
        self.plan_header = self.plan.header
        self.plan_id = str(self.plan_header["plan_id"])
        _validate_plan_stage_identity(
            self.plan_header,
            identity=self._stage_identity,
            requires_nl=self.requires_nl,
        )
        self.domains = tuple(str(x) for x in self.plan_header["domains"])
        unknown_domains = sorted(set(self.domains) - set(_DOMAIN_CATALOG_TYPES))
        if unknown_domains:
            raise OfflineLoaderError(f"unsupported catalog domains: {unknown_domains}")
        self.buckets = tuple(tuple(int(x) for x in hw) for hw in self.plan_header["buckets"])
        if len(set(self.buckets)) != len(self.buckets):
            raise OfflineLoaderError("plan bucket table contains duplicate (H,W) entries")
        self._bucket_to_idx = {hw: i for i, hw in enumerate(self.buckets)}
        self.N = np.asarray(self.plan_header["N"], dtype=np.int64)
        self._row_decoder = _VirtualRowDecoder(self.domains, self.N)

        self.catalog_header = self._validate_catalog_header()
        try:
            self._captionless_monet_rows = bind_caption_exceptions(
                stage=self.stage,
                eligibility=self.eligibility,
                plan_id=self.plan_id,
                catalog_hash=str(self.catalog_header["catalog_hash"]),
                plan_header_sha256=_sha256_file(self.plan_root / "header.json"),
                catalog_header_sha256=_sha256_file(self.catalog_root / "header.json"),
                decode_domain=lambda row_id: self.domains[
                    self._row_decoder.decode(row_id)[0]
                ],
            )
        except CaptionExceptionBindingError as exc:
            raise OfflineLoaderError(
                f"cannot bind frozen caption exceptions: {exc}"
            ) from exc
        try:
            self._image_repairs = bind_image_exceptions(
                stage=self.stage,
                eligibility=self.eligibility,
                plan_id=self.plan_id,
                plan_hash=str(self.plan_header.get("plan_hash", "")),
                catalog_hash=str(self.catalog_header["catalog_hash"]),
                plan_header_sha256=_sha256_file(self.plan_root / "header.json"),
                catalog_header_sha256=_sha256_file(self.catalog_root / "header.json"),
                decode_domain=lambda row_id: self.domains[
                    self._row_decoder.decode(row_id)[0]
                ],
            )
        except ImageExceptionBindingError as exc:
            raise OfflineLoaderError(
                f"cannot bind frozen image exceptions: {exc}"
            ) from exc
        try:
            self._anime_native_upsample_exceptions = (
                bind_anime_native_upsample_exceptions(
                    stage=self.stage,
                    eligibility=self.eligibility,
                    plan_id=self.plan_id,
                    plan_hash=str(self.plan_header.get("plan_hash", "")),
                    catalog_hash=str(self.catalog_header["catalog_hash"]),
                    plan_header_sha256=_sha256_file(self.plan_root / "header.json"),
                    catalog_header_sha256=_sha256_file(self.catalog_root / "header.json"),
                    decode_domain=lambda row_id: self.domains[
                        self._row_decoder.decode(row_id)[0]
                    ],
                )
            )
        except AnimeNativeUpsampleBindingError as exc:
            raise OfflineLoaderError(
                f"cannot bind frozen anime native-upsample exceptions: {exc}"
            ) from exc
        if self._anime_freq_root is not None:
            # 审查 P1(2026-07-11):freq 表必须与本 catalog 同源——catalog 重建而忘重跑
            # precompute 时在启动期即拒,而非训练数小时后才踩缺文件。
            freq_header = _read_json_object(self._anime_freq_root / "header.json")
            freq_catalog_hash = freq_header.get("catalog_hash")
            if freq_catalog_hash != self.catalog_header.get("catalog_hash"):
                raise OfflineLoaderError(
                    "anime frequency table was built from a different catalog: "
                    f"freq={freq_catalog_hash!r} vs catalog="
                    f"{self.catalog_header.get('catalog_hash')!r}; rerun precompute_anime_freq"
                )
        # TextAtlas authoritative GT, joined at runtime by row_id.  Bound to this
        # catalog for the same reason the anime frequency table is: a rebuilt catalog
        # renumbers row_id, and a stale GT sidecar would then attach the wrong string
        # to the wrong picture -- which is worse than no GT at all, because it reads
        # as a confident label.  Refused at startup, not hours in.
        self._textatlas_gt = None
        if self._textatlas_gt_root is not None:
            try:
                self._textatlas_gt = TextAtlasGroundTruth(
                    self._textatlas_gt_root,
                    catalog_hash=str(self.catalog_header["catalog_hash"]),
                    catalog_header_sha256=_sha256_file(self.catalog_root / "header.json"),
                )
            except TextAtlasGroundTruthError as exc:
                raise OfflineLoaderError(f"textatlas GT sidecar unusable: {exc}") from exc
        self._verify_catalog_file_hashes()
        self._catalog_readers = self._open_catalog_readers()
        self.square_only_domains = frozenset(
            str(x) for x in self.plan_header["square_only_domains"]
        )
        self.square_bucket_idx = int(self.plan_header["square_bucket_idx"])
        self._validate_square_only_construction()
        if self._gpic_payload_backend == "offline_gpic_blob_v1":
            assert self._gpic_blob_root is not None
            assert self._gpic_blob_manifest_sha256 is not None
            gpic_info = self.catalog_header["files"]["gpic"]
            gpic_reader = self._catalog_readers["gpic"]
            local_overlay: GpicBlobOverlay | None = None
            local_preflight_error: str | None = None
            try:
                local_overlay = GpicBlobOverlay(
                    self._gpic_blob_root,
                    expected_manifest_sha256=self._gpic_blob_manifest_sha256,
                    catalog=GpicBlobCatalogBinding(
                        catalog_hash=str(self.catalog_header["catalog_hash"]),
                        catalog_header_sha256=_sha256_file(
                            self.catalog_root / "header.json"
                        ),
                        gpic_name=str(gpic_info["name"]),
                        gpic_row_base=int(gpic_reader.row_base),
                        gpic_rows=int(gpic_info["rows"]),
                        gpic_bytes=int(gpic_info["bytes"]),
                        gpic_sha256=str(gpic_info["sha256"]),
                        gpic_schema=str(gpic_info["schema"]),
                    ),
                    fd_cache_size=self._gpic_blob_fd_cache_size,
                )
            except BaseException as exc:
                local_preflight_error = (
                    f"{type(exc).__name__}: {exc}"
                )

            preflight_errors = self._collective_gpic_preflight_errors(
                local_preflight_error
            )
            if preflight_errors:
                if local_overlay is not None:
                    try:
                        local_overlay.close()
                    except GpicBlobError:
                        pass
                raise OfflineLoaderError(
                    "GPIC blob overlay preflight failed collectively: "
                    + "; ".join(preflight_errors)
                )
            if local_overlay is None:
                raise OfflineLoaderError(
                    "GPIC blob collective preflight returned PASS without a local overlay"
                )
            self._gpic_blob = local_overlay
            try:
                self._verify_gpic_blob_index()
                self._gpic_blob_release_sha256 = self._gpic_blob.release_sha256
            except (GpicBlobError, OfflineLoaderError) as exc:
                if self._gpic_blob is not None:
                    try:
                        self._gpic_blob.close()
                    except GpicBlobError:
                        pass
                    self._gpic_blob = None
                raise OfflineLoaderError(
                    f"cannot initialize required GPIC blob overlay "
                    f"{self._gpic_blob_root}: {exc}"
                ) from exc

        try:
            prefetch_steps = int(
                os.environ.get("OFFLINE_ARB_PREFETCH_STEPS", str(_DEFAULT_PREFETCH_STEPS))
            )
        except ValueError as exc:
            raise OfflineLoaderError("OFFLINE_ARB_PREFETCH_STEPS must be an integer") from exc
        self.prefetch_steps = max(_DEFAULT_PREFETCH_STEPS, prefetch_steps)
        self._anime_workers = max(1, int(fetch_workers_anime))
        self._caption_workers = {
            "anime": _positive_int(caption_workers_anime, "caption_workers_anime"),
            "monet_real": _positive_int(
                caption_workers_monet_real, "caption_workers_monet_real"
            ),
            "gpic": _positive_int(caption_workers_gpic, "caption_workers_gpic"),
            "synthetic": _positive_int(
                caption_workers_synthetic, "caption_workers_synthetic"
            ),
            # Preserve the existing textatlas caption path (single-threaded).
            "textatlas": 1,
        }
        env_anime_caption_workers = _strict_positive_env_int(
            "OFFLINE_ARB_ANIME_CAPTION_WORKERS"
        )
        if env_anime_caption_workers is not None:
            self._caption_workers["anime"] = env_anime_caption_workers
        profile_windows = os.environ.get("OFFLINE_ARB_PROFILE_WINDOWS")
        if profile_windows is None:
            self._profile_windows = False
        else:
            normalized_profile_windows = profile_windows.strip().lower()
            if normalized_profile_windows in {"1", "true", "yes", "on"}:
                self._profile_windows = True
            elif normalized_profile_windows in {"", "0", "false", "no", "off"}:
                self._profile_windows = False
            else:
                raise OfflineLoaderError(
                    "OFFLINE_ARB_PROFILE_WINDOWS must be a boolean flag, got "
                    f"{profile_windows!r}"
                )
        # gpuval r3 实证(2026-07-11):随机 plan 下窗口内 monet/synth 项散布 ~200 个 tar,
        # 逐 tar 串行 FUSE range read(~0.3-0.5s 首字节)= 每窗口 60-100s 停顿 → 逐 tar 并行化。
        self._tar_workers = max(1, int(fetch_workers_tar))
        local_world = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
        local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank % local_world)))
        if local_rank < 0 or local_rank >= local_world:
            raise OfflineLoaderError(
                f"LOCAL_RANK={local_rank} outside [0,{local_world})"
            )
        node_gpic_threads = _strict_positive_env_int(
            "OFFLINE_ARB_GPIC_THREADS_PER_NODE"
        )
        self._gpic_threads_per_node = _resolved_gpic_threads_per_node(
            gpic_threads_per_node,
            node_gpic_threads,
            local_world,
        )
        self._gpic_workers = len(
            range(local_rank, self._gpic_threads_per_node, local_world)
        )
        self._gpic_workers = max(1, self._gpic_workers)
        self._local_world = local_world
        self._local_rank = local_rank
        local_root = os.environ.get("OFFLINE_ARB_LOCAL_ROOT", "/local/evan/offline_arb_v1")
        self._local_base = (
            Path(local_root)
            / _safe_component(self.plan_id)
            / f"rank{self.rank:04d}_pid{os.getpid()}"
        )
        self._caption_sidecar: CaptionSidecarStore | None = None
        if self._caption_sidecar_root is not None:
            expected_sidecar_domains = {
                domain: {
                    "row_base": self._catalog_readers[domain].row_base,
                    "row_count": self._catalog_readers[domain].row_count,
                    "columns": _CAPTION_SOURCE_COLUMNS[domain],
                    "catalog_file": self.catalog_header["files"][domain],
                }
                for domain in self.domains
            }
            try:
                self._caption_sidecar = CaptionSidecarStore(
                    self._caption_sidecar_root,
                    catalog_hash=str(self.catalog_header["catalog_hash"]),
                    catalog_header_sha256=_sha256_file(
                        self.catalog_root / "header.json"
                    ),
                    expected_domains=expected_sidecar_domains,
                    workers=self._caption_sidecar_workers,
                )
            except CaptionSidecarError as exc:
                raise OfflineLoaderError(
                    f"cannot initialize required caption sidecar "
                    f"{self._caption_sidecar_root}: {exc}"
                ) from exc

        self.domain_counter: Counter = Counter()
        self.produced: Counter = Counter()
        self.buffer_fill: dict[str, float] = {
            "plan_step": 0.0,
            "prefetch_depth": 0.0,
            **{f"ready_{d}": 0.0 for d in self.domains},
            **{f"supply_img_s_{d}": 0.0 for d in self.domains},
        }
        self.last_item_trace: dict[str, Any] | None = None

        self._next_step = int(
            self.plan_header["materialization"]["first_materialized_step"]
        )
        self._pipeline_active = False
        self._generation = 0
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._domain_queues: dict[str, "queue.Queue[_SupplyWindow | None]"] = {}
        self._threads: list[threading.Thread] = []
        self._ready: dict[int, dict[int, _PreparedItem]] = defaultdict(dict)
        self._step_bucket: dict[int, int] = {}
        self._expected: dict[int, int] = {}
        self._domain_ready: Counter = Counter()
        self._supply_started: dict[str, float] = {}
        self._supply_completed: Counter = Counter()
        self._pipeline_error: tuple[BaseException, str] | None = None
        self._pipeline_poisoned = False
        self._scheduled_until = self._next_step
        self._anime_freq_cache: dict[str, Counter] = {}
        telemetry_root_raw = os.environ.get("OFFLINE_ARB_IO_TELEMETRY_DIR")
        self._telemetry_enabled = bool(
            telemetry_root_raw is not None and telemetry_root_raw.strip()
        )
        self._telemetry_root: Path | None = None
        self._telemetry_path: Path | None = None
        self._telemetry_queue: "queue.Queue[dict[str, Any] | None] | None" = None
        self._telemetry_thread: threading.Thread | None = None
        self._telemetry_writer_ready: threading.Event | None = None
        self._telemetry_writer_error: BaseException | None = None
        self._telemetry_emit_lock = threading.Lock()
        self._telemetry_accepting = False
        self._telemetry_closed = False
        self._telemetry_emit_count = 0
        self._telemetry_rejected_count = 0
        self._telemetry_dropped_count = 0
        self._telemetry_runtime_error_count = 0
        self._telemetry_writer_error_reported = False
        self._telemetry_emit_seconds = 0.0
        self._telemetry_wait_threshold_s = 2.0
        self._telemetry_state_every_steps = 4
        self._telemetry_jobtag: str | None = None
        self._telemetry_nonce: str | None = None
        self._telemetry_host: str | None = None
        self._telemetry_sidecar_header_sha256: str | None = None
        self._telemetry_sidecar_release_digest: str | None = None
        if self._telemetry_enabled:
            assert telemetry_root_raw is not None
            try:
                telemetry_root = Path(telemetry_root_raw).expanduser().resolve(
                    strict=False
                )
            except OSError as exc:
                raise OfflineLoaderError(
                    f"cannot resolve OFFLINE_ARB_IO_TELEMETRY_DIR: {exc}"
                ) from exc
            if (
                not telemetry_root.is_absolute()
                or telemetry_root.parts[:2] != ("/", "local")
            ):
                raise OfflineLoaderError(
                    "OFFLINE_ARB_IO_TELEMETRY_DIR must be an absolute /local path"
                )
            wait_threshold = _strict_positive_env_float(
                "OFFLINE_ARB_WAIT_LOG_THRESHOLD_S"
            )
            state_every = _strict_positive_env_int(
                "OFFLINE_ARB_IO_STATE_EVERY_STEPS"
            )
            if wait_threshold is not None:
                self._telemetry_wait_threshold_s = wait_threshold
            if state_every is not None:
                self._telemetry_state_every_steps = state_every
            self._telemetry_root = telemetry_root
            attempt_jobtag = os.environ.get("OFFLINE_ARB_ATTEMPT_JOBTAG", "")
            attempt_nonce = os.environ.get("OFFLINE_ARB_ATTEMPT_NONCE", "")
            attempt_host = os.environ.get("OFFLINE_ARB_ATTEMPT_HOST", "")
            sidecar_header = os.environ.get(
                "OFFLINE_ARB_SIDECAR_HEADER_SHA256", ""
            )
            sidecar_release = os.environ.get(
                "OFFLINE_ARB_SIDECAR_RELEASE_DIGEST", ""
            )
            if re.fullmatch(r"[A-Za-z0-9_.-]+", attempt_jobtag) is None:
                raise OfflineLoaderError(
                    "OFFLINE_ARB_ATTEMPT_JOBTAG is required and malformed"
                )
            if re.fullmatch(r"[0-9a-f]{64}", attempt_nonce) is None:
                raise OfflineLoaderError(
                    "OFFLINE_ARB_ATTEMPT_NONCE must be a lowercase SHA-256 token"
                )
            if re.fullmatch(r"[A-Za-z0-9_.-]+", attempt_host) is None:
                raise OfflineLoaderError(
                    "OFFLINE_ARB_ATTEMPT_HOST is required and malformed"
                )
            if attempt_host != os.uname().nodename:
                raise OfflineLoaderError(
                    "OFFLINE_ARB_ATTEMPT_HOST differs from the runtime hostname"
                )
            if not attempt_host.startswith(f"{attempt_jobtag}-"):
                raise OfflineLoaderError(
                    "OFFLINE_ARB_ATTEMPT_JOBTAG is not the runtime hostname prefix"
                )
            if re.fullmatch(r"[0-9a-f]{64}", sidecar_header) is None:
                raise OfflineLoaderError(
                    "OFFLINE_ARB_SIDECAR_HEADER_SHA256 is required and malformed"
                )
            if re.fullmatch(r"[0-9a-f]{64}", sidecar_release) is None:
                raise OfflineLoaderError(
                    "OFFLINE_ARB_SIDECAR_RELEASE_DIGEST is required and malformed"
                )
            if self._caption_sidecar is None:
                raise OfflineLoaderError(
                    "I/O telemetry requires the validated caption sidecar backend"
                )
            if self._caption_sidecar.header_sha256 != sidecar_header:
                raise OfflineLoaderError(
                    "telemetry sidecar header identity differs from the runtime store"
                )
            self._telemetry_jobtag = attempt_jobtag
            self._telemetry_nonce = attempt_nonce
            self._telemetry_host = attempt_host
            self._telemetry_sidecar_header_sha256 = sidecar_header
            self._telemetry_sidecar_release_digest = sidecar_release
        try:
            self.step_timeout_s = float(os.environ.get("OFFLINE_ARB_STEP_TIMEOUT_S", "600"))
        except ValueError as exc:
            raise OfflineLoaderError("OFFLINE_ARB_STEP_TIMEOUT_S must be numeric") from exc
        if self.step_timeout_s <= 0:
            raise OfflineLoaderError("OFFLINE_ARB_STEP_TIMEOUT_S must be positive")
        if self._telemetry_enabled:
            self._start_telemetry_writer()
            self._enqueue_telemetry(
                {
                    "record_type": "identity",
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "local_rank": self._local_rank,
                    "local_world_size": self._local_world,
                    "plan_id": self.plan_id,
                    "plan_hash": str(self.plan_header["plan_hash"]),
                    "catalog_hash": str(self.catalog_header["catalog_hash"]),
                    "sidecar_header_sha256": self._telemetry_sidecar_header_sha256,
                    "sidecar_release_digest": self._telemetry_sidecar_release_digest,
                    "attempt_jobtag": self._telemetry_jobtag,
                    "attempt_nonce": self._telemetry_nonce,
                    "attempt_host": self._telemetry_host,
                    "telemetry_root": str(self._telemetry_root),
                    "telemetry_path": str(self._telemetry_path),
                    "prefetch_steps": self.prefetch_steps,
                    "gpic_threads_per_node": self._gpic_threads_per_node,
                    "gpic_workers_per_rank": self._gpic_workers,
                    "tar_workers_per_rank": self._tar_workers,
                    "gpic_payload_in_memory": self._gpic_payload_in_memory,
                    "gpic_payload_backend": self._gpic_payload_backend,
                    "gpic_blob_manifest_sha256": self._gpic_blob_manifest_sha256,
                    "gpic_blob_release_sha256": self._gpic_blob_release_sha256,
                    "gpic_blob_fd_cache_size": self._gpic_blob_fd_cache_size,
                    "gpic_source_consistency_mode": (
                        self._gpic_source_consistency_mode
                    ),
                    "pid": os.getpid(),
                }
            )

    def _start_telemetry_writer(self) -> None:
        """Start the sole owner of this rank's diagnostic JSONL file."""

        if not self._telemetry_enabled or self._telemetry_root is None:
            return
        try:
            self._telemetry_root.mkdir(parents=True, exist_ok=True)
            if self._telemetry_root.is_symlink() or not self._telemetry_root.is_dir():
                raise OfflineLoaderError(
                    f"telemetry root is not a real directory: {self._telemetry_root}"
                )
        except OfflineLoaderError:
            raise
        except OSError as exc:
            raise OfflineLoaderError(
                f"cannot create telemetry root {self._telemetry_root}: {exc}"
            ) from exc

        path = self._telemetry_root / f"io_profile.rank{self.rank:04d}.jsonl"
        work_queue: "queue.Queue[dict[str, Any] | None]" = queue.Queue(maxsize=1024)
        ready = threading.Event()
        self._telemetry_path = path
        self._telemetry_queue = work_queue
        self._telemetry_writer_ready = ready

        def writer() -> None:
            sequence = 0
            writer_seconds = 0.0
            try:
                # Exclusive creation makes a stale rank file a hard failure instead
                # of silently mixing two attempts into one evidence stream.
                with path.open("x", encoding="utf-8", buffering=1) as handle:
                    ready.set()
                    while True:
                        record = work_queue.get()
                        try:
                            if record is None:
                                handle.flush()
                                os.fsync(handle.fileno())
                                return
                            value = dict(record)
                            if value.get("record_type") == "closure":
                                value["records_before_closure"] = sequence
                                value["writer_seconds_before_closure"] = writer_seconds
                                value["rejected_emits_observed_by_writer"] = (
                                    self._telemetry_rejected_count
                                )
                            value["schema_version"] = _TELEMETRY_SCHEMA
                            value["sequence"] = sequence
                            started = time.perf_counter()
                            payload = json.dumps(
                                value,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            handle.write(payload + "\n")
                            handle.flush()
                            writer_seconds += time.perf_counter() - started
                            sequence += 1
                        finally:
                            work_queue.task_done()
            except BaseException as exc:
                self._telemetry_writer_error = exc
                ready.set()

        thread = threading.Thread(
            target=writer,
            name=f"offline-io-telemetry-r{self.rank}",
            daemon=True,
        )
        self._telemetry_thread = thread
        thread.start()
        if not ready.wait(timeout=10.0):
            raise OfflineLoaderError(
                f"telemetry writer did not initialize within 10s: {path}"
            )
        if self._telemetry_writer_error is not None:
            exc = self._telemetry_writer_error
            raise OfflineLoaderError(
                f"cannot initialize telemetry writer {path}: {type(exc).__name__}: {exc}"
            ) from exc
        with self._telemetry_emit_lock:
            self._telemetry_accepting = True

    def _enqueue_telemetry(self, record: Mapping[str, Any]) -> bool:
        if not self._telemetry_enabled:
            return False
        with self._telemetry_emit_lock:
            if not self._telemetry_accepting or self._telemetry_closed:
                self._telemetry_rejected_count += 1
                return False
            if self._telemetry_writer_error is not None:
                if not self._telemetry_writer_error_reported:
                    self._telemetry_runtime_error_count += 1
                    self._telemetry_writer_error_reported = True
                self._telemetry_accepting = False
                return False
            work_queue = self._telemetry_queue
            thread = self._telemetry_thread
            if work_queue is None or thread is None or not thread.is_alive():
                self._telemetry_runtime_error_count += 1
                self._telemetry_accepting = False
                return False
            started = time.perf_counter()
            if work_queue.qsize() >= work_queue.maxsize - 2:
                self._telemetry_dropped_count += 1
                return False
            try:
                work_queue.put_nowait(dict(record))
            except queue.Full:
                self._telemetry_dropped_count += 1
                return False
            self._telemetry_emit_seconds += time.perf_counter() - started
            self._telemetry_emit_count += 1
            return True

    def _ready_snapshot_locked(self) -> dict[str, Any]:
        contiguous = 0
        step = self._next_step
        while step < self._scheduled_until:
            expected = self._expected.get(step)
            if expected is None or len(self._ready.get(step, {})) != expected:
                break
            contiguous += 1
            step += 1
        return {
            "scheduled_depth_steps": max(0, self._scheduled_until - self._next_step),
            "contiguous_ready_steps": contiguous,
            "domain_ready_items": {
                domain: max(0, int(self._domain_ready[domain]))
                for domain in self.domains
            },
        }

    def _emit_state_record(self, consumed_step: int) -> None:
        if not self._telemetry_enabled:
            return
        if (consumed_step + 1) % self._telemetry_state_every_steps != 0:
            return
        with self._condition:
            snapshot = self._ready_snapshot_locked()
        self._enqueue_telemetry(
            {
                "record_type": "state",
                "rank": self.rank,
                "consumed_step": consumed_step,
                "next_step": self._next_step,
                **snapshot,
            }
        )

    def _close_telemetry(self) -> None:
        if not self._telemetry_enabled or self._telemetry_closed:
            return
        work_queue = self._telemetry_queue
        thread = self._telemetry_thread
        if work_queue is None or thread is None:
            raise OfflineLoaderError("telemetry writer disappeared during closure")
        with self._telemetry_emit_lock:
            if self._telemetry_closed:
                return
            if self._telemetry_writer_error is not None:
                if not self._telemetry_writer_error_reported:
                    self._telemetry_runtime_error_count += 1
                    self._telemetry_writer_error_reported = True
                self._telemetry_accepting = False
                self._telemetry_closed = True
                return
            if not thread.is_alive():
                self._telemetry_runtime_error_count += 1
                self._telemetry_accepting = False
                self._telemetry_closed = True
                return
            self._telemetry_accepting = False
            prior_count = self._telemetry_emit_count
            prior_seconds = self._telemetry_emit_seconds
            closure = {
                "record_type": "closure",
                "rank": self.rank,
                "next_step": self._next_step,
                "records_enqueued_before_closure": prior_count,
                "producer_enqueue_seconds_before_closure": prior_seconds,
                "rejected_emits_before_closure": self._telemetry_rejected_count,
                "dropped_emits": self._telemetry_dropped_count,
                "runtime_errors": self._telemetry_runtime_error_count,
                "telemetry_fatal": bool(
                    self._telemetry_dropped_count
                    or self._telemetry_runtime_error_count
                ),
            }
            try:
                # Holding the producer lock makes this pair a sentry: no normal
                # record can be ordered after closure, even if a supply thread
                # finishes a future window during shutdown.
                work_queue.put_nowait(closure)
                work_queue.put_nowait(None)
            except queue.Full:
                self._telemetry_dropped_count += 1
                self._telemetry_runtime_error_count += 1
                self._telemetry_closed = True
                return
            self._telemetry_closed = True
        thread.join(timeout=5.0)
        if thread.is_alive():
            self._telemetry_runtime_error_count += 1
            return
        if self._telemetry_writer_error is not None:
            if not self._telemetry_writer_error_reported:
                self._telemetry_runtime_error_count += 1
                self._telemetry_writer_error_reported = True
            return

    @staticmethod
    def _resolve_rank(rank, world_size) -> tuple[int, int]:
        if rank is None or world_size is None:
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    detected = (dist.get_rank(), dist.get_world_size())
                else:
                    detected = (
                        int(os.environ.get("RANK", "0")),
                        int(os.environ.get("WORLD_SIZE", "1")),
                    )
            except Exception:
                detected = (
                    int(os.environ.get("RANK", "0")),
                    int(os.environ.get("WORLD_SIZE", "1")),
                )
            rank = detected[0] if rank is None else rank
            world_size = detected[1] if world_size is None else world_size
        r = int(rank)
        w = _positive_int(world_size, "world_size")
        if r < 0 or r >= w:
            raise OfflineLoaderError(f"rank {r} outside [0,{w})")
        return r, w

    def _validate_catalog_header(self) -> dict[str, Any]:
        header = _read_json_object(self.catalog_root / "header.json")
        if header.get("schema_version") != "offline_arb_catalog_v1":
            raise OfflineLoaderError(
                f"unsupported catalog schema_version={header.get('schema_version')!r}"
            )
        if header.get("diagnostic_run") is not False:
            raise OfflineLoaderError("diagnostic catalog products are forbidden for training")
        _validate_artifact_stage_identity(
            header,
            artifact="catalog header",
            identity=self._stage_identity,
            requires_nl=self.requires_nl,
        )
        if header.get("row_scheme") != "virtual_catalog_v1_materialized":
            raise OfflineLoaderError(f"unsupported catalog row_scheme={header.get('row_scheme')!r}")
        domains = [str(x) for x in header.get("domains", [])]
        if domains != list(self.domains) or header.get("domain_order") != domains:
            raise OfflineLoaderError(
                f"catalog/plan domain order mismatch: catalog={domains}, plan={list(self.domains)}"
            )

        raw_buckets = header.get("buckets")
        if not isinstance(raw_buckets, list) or len(raw_buckets) != len(self.buckets):
            raise OfflineLoaderError("catalog bucket table has the wrong length")
        catalog_buckets: list[tuple[int, int]] = []
        for idx, entry in enumerate(raw_buckets):
            if not isinstance(entry, Mapping) or int(entry.get("idx", -1)) != idx:
                raise OfflineLoaderError(f"catalog bucket index drift at position {idx}")
            catalog_buckets.append((int(entry["h"]), int(entry["w"])))
        if tuple(catalog_buckets) != self.buckets:
            raise OfflineLoaderError("catalog bucket table differs from plan bucket table")

        n_raw = header.get("N", {}).get("materialized_after_exclusion")
        n_catalog = np.asarray(n_raw)
        if n_catalog.shape != self.N.shape or not np.issubdtype(n_catalog.dtype, np.integer):
            raise OfflineLoaderError(
                f"catalog materialized N has shape/type {n_catalog.shape}/{n_catalog.dtype}, "
                f"expected {self.N.shape}/integer"
            )
        if not np.array_equal(n_catalog.astype(np.int64), self.N):
            raise OfflineLoaderError(
                "plan N differs from catalog N.materialized_after_exclusion; "
                "rebuild the plan after tar-index exclusions"
            )
        row_count = int(header.get("row_count", -1))
        if row_count != self._row_decoder.total_rows or row_count != int(
            self.plan_header["catalog_size"]
        ):
            raise OfflineLoaderError(
                f"catalog size mismatch: catalog={row_count}, decoded={self._row_decoder.total_rows}, "
                f"plan={self.plan_header['catalog_size']}"
            )

        inputs = header.get("inputs")
        if not isinstance(inputs, Mapping):
            raise OfflineLoaderError("catalog header inputs must be an object")

        # anchors_v1 geometry, captured from the CATALOG so runtime verification uses
        # the thresholds the rows were admitted under -- not whatever this process was
        # configured with.  The assignment_hash check above already proved the header
        # describes the geometry it claims, so these values are trustworthy by the
        # time we read them.
        self._anchors_mode = _header_bucket_mode(header) == anchors_v1.BUCKET_MODE
        self._anchors_crop_loss: dict[str, float] = {}
        # Per-rank, per-process count of anime native-drift rows whose crop_loss against
        # the decoded pixels exceeded the catalog's threshold.  Bounded by
        # _ANIME_DRIFT_CROP_TOLERATED_MAX; resets on restart, which is the conservative
        # direction (a resumed run gets a fresh budget rather than inheriting a spent one).
        self._anchors_drift_crop_tolerated = 0
        if self._anchors_mode:
            mar = inputs.get("multi_ar_params")
            if not isinstance(mar, Mapping) or "crop_loss_max" not in mar:
                raise OfflineLoaderError(
                    "anchors_v1 catalog header is missing inputs.multi_ar_params.crop_loss_max"
                )
            by_domain = mar.get("crop_loss_max_by_domain") or {}
            for domain in self.domains:
                self._anchors_crop_loss[domain] = float(
                    by_domain.get(domain, mar["crop_loss_max"])
                )
            if tuple(self.buckets) != tuple(
                (h, w) for w, h in anchors_v1.ANCHORS_V1_WH
            ):
                raise OfflineLoaderError(
                    "anchors_v1 catalog bucket table does not match ANCHORS_V1_WH; "
                    "verify_binding indexes ANCHORS_V1_WH by bucket_idx, so a "
                    "reordered table would verify every row against the wrong anchor"
                )

        sampling_sha = inputs.get("sampling_config_sha256")
        if not isinstance(sampling_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sampling_sha):
            raise OfflineLoaderError("catalog inputs.sampling_config_sha256 is missing/invalid")
        actual_sampling_sha = _sha256_file(self.sampling_config_path)
        if actual_sampling_sha != sampling_sha:
            raise OfflineLoaderError(
                "runtime sampling_config differs from the catalog-bound sampling config: "
                f"expected={sampling_sha}, actual={actual_sampling_sha}, "
                f"path={self.sampling_config_path}"
            )

        files = header.get("files")
        if not isinstance(files, Mapping) or set(files) != set(self.domains):
            raise OfflineLoaderError("catalog header files must map every plan domain exactly once")
        for d, domain in enumerate(self.domains):
            info = files[domain]
            if not isinstance(info, Mapping):
                raise OfflineLoaderError(f"catalog files[{domain}] must be an object")
            name = info.get("name")
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise OfflineLoaderError(
                    f"unsafe/invalid catalog file name for {domain}: {name!r}"
                )
            expected_rows = int(self.N[:, d].sum(dtype=np.int64))
            if int(info.get("rows", -1)) != expected_rows:
                raise OfflineLoaderError(
                    f"catalog files[{domain}].rows={info.get('rows')} != N_total={expected_rows}"
                )
            checksum = info.get("sha256")
            if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise OfflineLoaderError(f"catalog files[{domain}].sha256 is invalid")

        bucket_payload = [list(hw) for hw in self.buckets]
        if header.get("bucket_table_hash") != _canonical_sha256(bucket_payload):
            raise OfflineLoaderError("catalog bucket_table_hash mismatch")
        assignment_payload = _assignment_identity_payload(
            header,
            identity=self._stage_identity,
            requires_nl=self.requires_nl,
        )
        if header.get("assignment_hash") != _canonical_sha256(assignment_payload):
            raise OfflineLoaderError("catalog assignment_hash mismatch")
        catalog_payload = {
            "format": "offline_arb_catalog_v1",
            "domains": domains,
            "files": {
                domain: {
                    "rows": int(files[domain]["rows"]),
                    "sha256": str(files[domain]["sha256"]),
                }
                for domain in domains
            },
            "row_order": ["domain", "bucket_idx", "uid"],
            "floor": header.get("floor"),
            "bucket_table": bucket_payload,
            "kept_bucket_indices": list(header.get("kept_bucket_indices", [])),
        }
        catalog_hash = _canonical_sha256(catalog_payload)
        if header.get("catalog_hash") != catalog_hash:
            raise OfflineLoaderError("catalog_hash does not match its canonical header payload")

        upstream_catalog = self.plan_header.get("upstream_catalog_hash")
        upstream_assignment = self.plan_header.get("upstream_assignment_hash")
        if not upstream_catalog or not upstream_assignment:
            raise OfflineLoaderError(
                "plan uses virtual/proxy upstream hashes; rebuild it with this catalog's "
                "catalog_hash and assignment_hash"
            )
        if upstream_catalog != catalog_hash or self.plan_header.get("catalog_hash") != catalog_hash:
            raise OfflineLoaderError("plan/catalog catalog_hash mismatch")
        if (
            upstream_assignment != header.get("assignment_hash")
            or self.plan_header.get("assignment_hash") != header.get("assignment_hash")
        ):
            raise OfflineLoaderError("plan/catalog assignment_hash mismatch")
        hashes = self.plan_header.get("hashes", {})
        if hashes.get("catalog_sha256") != catalog_hash:
            raise OfflineLoaderError("plan hashes.catalog_sha256 mismatch")
        if hashes.get("assignment_sha256") != header.get("assignment_hash"):
            raise OfflineLoaderError("plan hashes.assignment_sha256 mismatch")

        checks = header.get("self_checks", {})
        if isinstance(checks, Mapping):
            failed = [
                name
                for name, value in checks.items()
                if isinstance(value, Mapping) and "pass" in value and value.get("pass") is not True
            ]
            if failed:
                raise OfflineLoaderError(f"catalog header contains failed self-checks: {failed}")
        return header

    def _verify_catalog_file_hashes(self) -> None:
        """Close catalog_hash over the actual Parquet bytes, once per dist job.

        In a real torch.distributed run rank 0 performs the sequential reads and
        broadcasts the outcome.  Standalone/simulated-rank processes each verify
        locally because they have no trustworthy collective channel.
        """

        def verify_here() -> str | None:
            for domain in self.domains:
                info = self.catalog_header["files"][domain]
                path = self.catalog_root / str(info["name"])
                actual = _sha256_file(path)
                expected = str(info["sha256"])
                if actual != expected:
                    return (
                        f"catalog file SHA-256 mismatch for {domain}: "
                        f"expected={expected}, actual={actual}, path={path}"
                    )
            return None

        dist = None
        try:
            import torch.distributed as torch_dist

            if torch_dist.is_available() and torch_dist.is_initialized():
                dist = torch_dist
        except Exception:
            dist = None
        if dist is None:
            error = verify_here()
        else:
            status: list[str | None] = [None]
            if self.rank == 0:
                try:
                    status[0] = verify_here()
                except BaseException as exc:
                    status[0] = f"catalog SHA verification raised {type(exc).__name__}: {exc}"
            dist.broadcast_object_list(status, src=0)
            error = status[0]
        if error:
            raise OfflineLoaderError(error)

    def _collective_gpic_preflight_errors(
        self, local_error: str | None
    ) -> list[str]:
        """Make every rank cross the same readiness collective before raising.

        A rank-local S3/control/mmap failure must not let that process leave the
        constructor while healthy peers enter the rank-0 index broadcast.
        """

        dist = None
        try:
            import torch.distributed as torch_dist

            if torch_dist.is_available() and torch_dist.is_initialized():
                dist = torch_dist
        except Exception:
            dist = None
        if dist is None:
            return [f"rank{self.rank}={local_error}"] if local_error else []

        actual_rank = int(dist.get_rank())
        actual_world = int(dist.get_world_size())
        status = local_error
        if self.rank != actual_rank or self.world_size != actual_world:
            mismatch = (
                "dataset/distributed rank identity mismatch: "
                f"dataset={self.rank}/{self.world_size}, "
                f"dist={actual_rank}/{actual_world}"
            )
            status = f"{status}; {mismatch}" if status else mismatch
        gathered: list[str | None] = [None] * actual_world
        dist.all_gather_object(gathered, status)
        return [
            f"rank{rank}={error}"
            for rank, error in enumerate(gathered)
            if error is not None
        ]

    def _verify_gpic_blob_index(self) -> None:
        """Hash the 1.12-GB FSx index once per distributed job.

        Every rank independently validates the small immutable controls and mmap
        identity. Rank 0 alone streams the fixed-width index, broadcasts either
        its exact digest or the failure, and all ranks authorize reads against
        that same manifest-declared digest.
        """

        overlay = self._gpic_blob
        if overlay is None:
            raise OfflineLoaderError("GPIC blob index verification has no overlay")
        dist = None
        try:
            import torch.distributed as torch_dist

            if torch_dist.is_available() and torch_dist.is_initialized():
                dist = torch_dist
        except Exception:
            dist = None
        if dist is None:
            try:
                overlay.verify_index_sha256()
            except GpicBlobError as exc:
                raise OfflineLoaderError(f"GPIC blob index verification failed: {exc}") from exc
            return

        status: list[str | None] = [None, None]
        if int(dist.get_rank()) == 0:
            try:
                status[0] = overlay.compute_index_sha256()
            except BaseException as exc:
                status[1] = (
                    f"GPIC blob index SHA verification raised "
                    f"{type(exc).__name__}: {exc}"
                )
        dist.broadcast_object_list(status, src=0)
        actual, error = status
        if error is not None or actual is None:
            raise OfflineLoaderError(error or "GPIC blob index verification returned no digest")
        local_accept_error: str | None = None
        try:
            overlay.accept_collective_index_verification(actual)
        except BaseException as exc:
            local_accept_error = f"{type(exc).__name__}: {exc}"
        accept_errors: list[str | None] = [None] * int(dist.get_world_size())
        dist.all_gather_object(accept_errors, local_accept_error)
        failures = [
            f"rank{rank}={failure}"
            for rank, failure in enumerate(accept_errors)
            if failure is not None
        ]
        if failures:
            raise OfflineLoaderError(
                "GPIC blob index verification failed collectively: "
                + "; ".join(failures)
            )

    def _open_catalog_readers(self) -> dict[str, _CatalogDomainReader]:
        readers: dict[str, _CatalogDomainReader] = {}
        row_base = 0
        files = self.catalog_header["files"]
        for d, domain in enumerate(self.domains):
            info = files[domain]
            name = info.get("name")
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise OfflineLoaderError(f"unsafe/invalid catalog file name for {domain}: {name!r}")
            count = int(self.N[:, d].sum(dtype=np.int64))
            readers[domain] = _CatalogDomainReader(
                domain,
                self.catalog_root / name,
                row_base,
                count,
                info,
            )
            row_base += count
        if row_base != self._row_decoder.total_rows:
            raise OfflineLoaderError("catalog domain row ranges do not close at row_count")
        return readers

    def _validate_square_only_construction(self) -> None:
        catalog_square = int(self.catalog_header.get("square_bucket_idx", -1))
        catalog_only = frozenset(
            str(x) for x in self.catalog_header.get("square_only_domains", [])
        )
        if catalog_square != self.square_bucket_idx or catalog_only != self.square_only_domains:
            raise OfflineLoaderError("catalog/plan square-only contract mismatch")
        if not (0 <= self.square_bucket_idx < len(self.buckets)):
            raise OfflineLoaderError("square_bucket_idx is outside the bucket table")
        h, w = self.buckets[self.square_bucket_idx]
        if h != w:
            raise OfflineLoaderError("square_bucket_idx does not point to a square bucket")
        domain_index = {name: i for i, name in enumerate(self.domains)}
        for domain in self.square_only_domains:
            if domain not in domain_index:
                raise OfflineLoaderError(f"unknown square-only domain {domain!r}")
            d = domain_index[domain]
            leaking = [
                b for b in range(len(self.buckets)) if b != self.square_bucket_idx and int(self.N[b, d])
            ]
            if leaking:
                raise OfflineLoaderError(
                    f"square-only domain {domain} has non-square catalog cells: {leaking}"
                )

    def _v_anchor_for(self, domain: str) -> float:
        return float(self._crop_v_anchor.get(domain, 0.5))

    # Direct reuse of the existing caption/CFG contract.  These methods depend
    # only on self.cfg/stage/tokenizer/caption_budget and the local record.
    def _build_caption(self, domain, rec, band, rng):
        return CompleteV1Dataset._build_caption(self, domain, rec, band, rng)

    def _finalize_caption(self, domain, rec, band, rng):
        if domain == "textatlas" and self._textatlas_gt is not None:
            caption = build_textatlas_caption(rec["captions"], rng)
            gt = self._textatlas_gt.get(int(rec["row_id"]))
            # Rows without GT (about 53% of textatlas) take the same single rng draw
            # as before, because compose_* returns before touching rng when gt is
            # empty.  That keeps the no-GT prompt stream byte-identical to a run with
            # the sidecar switched off, so enabling it changes only the covered rows.
            return compose_textatlas_gt_prompt(
                gt, caption, self.tokenizer, self.caption_budget, rng
            )
        return CompleteV1Dataset._finalize_caption(self, domain, rec, band, rng)

    @staticmethod
    def _is_cfg_protected(domain, rec) -> bool:
        return CompleteV1Dataset._is_cfg_protected(domain, rec)

    def set_start_step(self, global_step: int) -> None:
        """Cold-seek to the next absolute plan step used by ``train.py``.

        ``train.py`` checkpoints the number of completed optimizer steps, which
        is also the zero-based index of the next batch.  No ``+1`` is applied.
        """

        if isinstance(global_step, bool) or not isinstance(global_step, (int, np.integer)):
            raise OfflineLoaderError(f"global_step must be an integer, got {global_step!r}")
        step = int(global_step)
        total = int(self.plan_header["total_steps"])
        first = int(self.plan_header["materialization"]["first_materialized_step"])
        if step < first or step > total:
            raise OfflineLoaderError(
                f"global_step {step} outside materialized plan range [{first},{total}]"
            )
        if self._pipeline_active:
            raise OfflineLoaderError(
                "set_start_step must be called before the loader's first next(); "
                "close and recreate the dataset to seek an active pipeline"
            )
        self._next_step = step
        self._scheduled_until = step
        self.last_item_trace = None
        self._refresh_metrics(plan_step=step)

    @staticmethod
    def _slot_seed(plan_id: str, step: int, rank: int, slot: int) -> int:
        payload = f"{plan_id}\0{step}\0{rank}\0{slot}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")

    def _select_band(self, domain: str, rec: Mapping[str, Any], rng: random.Random) -> int:
        if domain not in {"monet_real", "gpic"}:
            return -1
        codes = [int(x) for x in rec["band_captioner"]]
        served = int(rec["served_mask"])
        available = [i for i, code in enumerate(codes) if code > 0 and ((served >> i) & 1)]
        if not available:
            row_id = rec.get("row_id")
            expected_uid = self._captionless_monet_rows.get(row_id)
            if expected_uid is not None:
                uid = rec.get("uid")
                cap_texts = rec.get("cap_texts")
                exact_empty_record = (
                    domain == "monet_real"
                    and uid == expected_uid
                    and served == 0
                    and codes == [0, 0, 0, 0]
                    and isinstance(cap_texts, Mapping)
                    and set(cap_texts) == set(MONET_CAPTION_CODES)
                    and all(value is None for value in cap_texts.values())
                )
                if not exact_empty_record:
                    raise OfflineLoaderError(
                        "frozen caption exception record drift: "
                        f"domain={domain},row_id={row_id},uid={uid!r},"
                        f"served={served},codes={codes},"
                        f"cap_codes={sorted(cap_texts) if isinstance(cap_texts, Mapping) else None}"
                    )
                warnings.warn(
                    "applying frozen captionless MONET exception "
                    f"row_id={row_id},uid={uid}; prompt is intentionally empty",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return -1
            raise OfflineLoaderError(
                f"planned {domain} item has no served caption band: row={rec.get('uid')}"
            )
        weights = [max(0.0, float(self.cfg.band_mix_real[BANDS[i]])) for i in available]
        total = sum(weights)
        if total <= 0:
            return available[0]
        pick = rng.random() * total
        acc = 0.0
        for band, weight in zip(available, weights):
            acc += weight
            if pick <= acc:
                return band
        return available[-1]

    def _validate_planned_band(
        self, domain: str, rec: Mapping[str, Any], planned_band: int
    ) -> int:
        band = int(planned_band)
        if domain not in {"monet_real", "gpic"}:
            if band != -1:
                raise OfflineLoaderError(
                    f"non-real planned row has caption band {band}: "
                    f"domain={domain},row_id={rec.get('row_id')}"
                )
            if domain == "synthetic":
                base_w = float(rec.get("base_w", 0.0))
                if not math.isfinite(base_w) or base_w <= 0:
                    raise OfflineLoaderError(
                        f"weighted plan selected a non-positive Synthetic row: "
                        f"row_id={rec.get('row_id')},base_w={base_w}"
                    )
            return -1
        if band < 0 or band >= 4:
            raise OfflineLoaderError(
                f"real planned row has invalid caption band {band}: "
                f"domain={domain},row_id={rec.get('row_id')}"
            )
        base_w = float(rec.get("base_w", 0.0))
        served = int(rec["served_mask"])
        codes = [int(value) for value in rec["band_captioner"]]
        if not math.isfinite(base_w) or base_w <= 0:
            raise OfflineLoaderError(
                f"weighted plan selected a non-positive real row: "
                f"domain={domain},row_id={rec.get('row_id')},base_w={base_w}"
            )
        if ((served >> band) & 1) == 0 or codes[band] <= 0:
            raise OfflineLoaderError(
                f"weighted plan selected an unserved caption band: "
                f"domain={domain},row_id={rec.get('row_id')},band={band},"
                f"served_mask={served},codes={codes}"
            )
        return band

    def _plan_items_for_step(self, step: int) -> tuple[int, list[_PlannedItem]]:
        bucket_hw, row_ids, caption_bands = self.plan.batch(step, self.rank)
        bucket = self._bucket_to_idx.get(tuple(int(x) for x in bucket_hw))
        if bucket is None:
            raise OfflineLoaderError(f"plan step {step} returned unknown bucket {bucket_hw}")
        if row_ids.dtype != np.dtype(np.int64) or row_ids.shape != (
            int(self.plan_header["per_rank_bs"]),
        ):
            raise OfflineLoaderError(
                f"plan batch shape drift at step {step}: {row_ids.dtype}/{row_ids.shape}"
            )
        if caption_bands.dtype != np.dtype(np.int8) or caption_bands.shape != row_ids.shape:
            raise OfflineLoaderError(
                f"plan caption-band shape drift at step {step}: "
                f"{caption_bands.dtype}/{caption_bands.shape}"
            )
        items: list[_PlannedItem] = []
        for slot, (row_raw, band_raw) in enumerate(
            zip(row_ids.tolist(), caption_bands.tolist())
        ):
            row = int(row_raw)
            d, decoded_bucket, _local = self._row_decoder.decode(row)
            if decoded_bucket != bucket:
                raise OfflineLoaderError(
                    f"plan row_id {row} decodes to bucket {decoded_bucket}, step {step} uses {bucket}"
                )
            items.append(
                _PlannedItem(
                    step, slot, row, self.domains[d], d, bucket, int(band_raw)
                )
            )
        return bucket, items

    def _start_pipeline(self) -> None:
        if self._pipeline_active:
            return
        if self._pipeline_poisoned:
            raise OfflineLoaderError(
                "this dataset cannot restart because a previous supply thread did not stop; "
                "create a fresh process/dataset"
            )
        self._generation += 1
        self._stop_event = threading.Event()
        self._domain_queues = {domain: queue.Queue() for domain in self.domains}
        self._threads = []
        self._ready = defaultdict(dict)
        self._step_bucket = {}
        self._expected = {}
        self._domain_ready = Counter()
        self._supply_started = {domain: time.monotonic() for domain in self.domains}
        self._supply_completed = Counter()
        self._pipeline_error = None
        self._scheduled_until = self._next_step
        self._pipeline_active = True
        for domain in self.domains:
            thread = threading.Thread(
                target=self._supply_loop,
                args=(domain, self._generation, self._stop_event),
                name=f"offline-{domain}-supply-r{self.rank}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        try:
            target = min(
                int(self.plan_header["total_steps"]),
                self._next_step + 2 * self.prefetch_steps,
            )
            while self._scheduled_until < target:
                self._schedule_window(self._scheduled_until)
        except BaseException:
            self._stop_pipeline()
            raise

    def _schedule_window(self, start: int) -> None:
        total = int(self.plan_header["total_steps"])
        if start != self._scheduled_until:
            raise OfflineLoaderError(
                f"non-contiguous prefetch scheduling: start={start}, scheduled={self._scheduled_until}"
            )
        # Keep the queue far ahead, but make each producer burst one training
        # step.  Four-step bursts were semantically correct yet synchronized
        # CPU/native work across all ranks and produced periodic FSDP stalls.
        stop = min(total, start + _SUPPLY_WINDOW_STEPS)
        by_domain: dict[str, list[_PlannedItem]] = {domain: [] for domain in self.domains}
        for step in range(start, stop):
            bucket, items = self._plan_items_for_step(step)
            self._step_bucket[step] = bucket
            self._expected[step] = len(items)
            for item in items:
                by_domain[item.domain].append(item)
        for domain, items in by_domain.items():
            if items:
                self._domain_queues[domain].put(
                    _SupplyWindow(self._generation, start, stop, tuple(items))
                )
        self._scheduled_until = stop
        self._refresh_metrics()

    def _supply_loop(
        self,
        domain: str,
        generation: int,
        stop_event: threading.Event,
    ) -> None:
        work_queue = self._domain_queues[domain]
        while not stop_event.is_set():
            try:
                task = work_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if task is None:
                return
            if task.generation != generation:
                continue
            try:
                window_record = self._prepare_window(
                    domain,
                    task,
                    generation=generation,
                    stop_event=stop_event,
                )
                if stop_event.is_set() or generation != self._generation:
                    return
                # Publication is the evidence boundary: a recorded window must
                # already be visible to the consumer in its entirety.
                if (
                    window_record is not None
                    and not stop_event.is_set()
                    and generation == self._generation
                ):
                    self._enqueue_telemetry(window_record)
            except BaseException as exc:
                detail = traceback.format_exc()
                with self._condition:
                    if generation == self._generation:
                        if self._pipeline_error is None:
                            self._pipeline_error = (exc, detail)
                        stop_event.set()
                        self._condition.notify_all()
                return

    def _prepare_window(
        self,
        domain: str,
        task: _SupplyWindow,
        *,
        generation: int,
        stop_event: threading.Event,
    ) -> dict[str, Any] | None:
        measure = self._profile_windows or self._telemetry_enabled
        profile_started = time.perf_counter() if measure else 0.0
        catalog_rows = self._catalog_readers[domain].get_many(
            [item.row_id for item in task.items]
        )
        pairs: list[tuple[_PlannedItem, dict[str, Any]]] = []
        for item in task.items:
            row = dict(catalog_rows[item.row_id])
            if int(row["bucket_idx"]) != item.bucket_idx:
                raise OfflineLoaderError(
                    f"catalog bucket mismatch row_id={item.row_id}: "
                    f"catalog={row['bucket_idx']}, plan={item.bucket_idx}"
                )
            if str(row["domain"]) != domain:
                raise OfflineLoaderError(
                    f"catalog domain mismatch row_id={item.row_id}: {row['domain']!r} != {domain!r}"
                )
            if int(row["width"]) <= 0 or int(row["height"]) <= 0:
                raise OfflineLoaderError(
                    f"invalid catalog native size row_id={item.row_id}: "
                    f"{row['width']}x{row['height']}"
                )
            # Zero upsampling, checked against the catalog before any payload is
            # fetched.  Every existing check here is identity/self-consistency
            # (bucket_idx matches the plan, domain matches, sizes are positive);
            # none of them compares the native size against the bucket it was
            # assigned to.  A catalog that is hash-self-consistent but
            # geometrically wrong therefore reaches resize_u8_to_bucket, whose
            # scale = max(W/w0, H/h0) enlarges without ever refusing.
            bucket_h, bucket_w = self.buckets[item.bucket_idx]
            if int(row["width"]) < int(bucket_w) or int(row["height"]) < int(bucket_h):
                raise OfflineLoaderError(
                    f"catalog geometry would upsample row_id={item.row_id}, "
                    f"domain={domain}, bucket={item.bucket_idx}: "
                    f"catalog={row['width']}x{row['height']}, "
                    f"bucket={bucket_w}x{bucket_h}"
                )
            if domain in self.square_only_domains and item.bucket_idx != self.square_bucket_idx:
                raise OfflineLoaderError(
                    f"square-only violation row_id={item.row_id}, domain={domain}, "
                    f"bucket={item.bucket_idx}"
                )
            pairs.append((item, row))

        profile_catalog_done = time.perf_counter() if measure else 0.0
        caption_records = self._caption_records(domain, pairs)
        profile_caption_done = time.perf_counter() if measure else 0.0
        image_result = self._image_payloads(
            domain,
            pairs,
            task.generation,
            telemetry=self._telemetry_enabled,
        )
        if self._telemetry_enabled:
            image_payloads, image_detail = image_result
        else:
            image_payloads = image_result
            image_detail = None
        self._payload_token_closure(pairs, image_payloads, f"{domain} image payload")
        prepared_step: list[_PreparedItem] = []
        prepared_step_id: int | None = None

        def publish_prepared_step() -> bool:
            if stop_event.is_set() or generation != self._generation:
                return False
            for value in prepared_step:
                if stop_event.is_set() or generation != self._generation:
                    return False
                self._publish(value)
            prepared_step.clear()
            return True

        for item, row in pairs:
            if item.step < task.start or item.step >= task.stop:
                raise OfflineLoaderError(
                    f"prepared item escaped supply window domain={domain}: "
                    f"step={item.step}, window=[{task.start},{task.stop})"
                )
            if prepared_step_id is None:
                prepared_step_id = item.step
            elif item.step != prepared_step_id:
                if item.step <= prepared_step_id:
                    raise OfflineLoaderError(
                        f"non-monotonic prepared step order domain={domain}: "
                        f"{item.step} after {prepared_step_id}"
                    )
                if not publish_prepared_step():
                    return None
                prepared_step_id = item.step
            if stop_event.is_set() or generation != self._generation:
                return None
            token = (item.step, item.slot)
            payload = image_payloads.pop(token)
            prepared = _PreparedItem(
                planned=item,
                catalog=row,
                caption_record=caption_records[token],
            )
            if domain == "gpic" and not self._gpic_payload_in_memory:
                prepared.local_path = str(payload)
            else:
                prepared.raw = bytes(payload)
            prepared.bucket_u8 = self._bucket_u8_for_item(prepared)
            # Retain only the prepared representation in the multi-step
            # buffer; native bytes are no longer needed after exact decode.
            prepared.raw = None
            prepared.local_path = None
            prepared_step.append(prepared)
        if prepared_step and not publish_prepared_step():
            return None
        profile_image_done = time.perf_counter() if measure else 0.0
        profile_total_done = time.perf_counter() if measure else 0.0
        if self._profile_windows:
            caption_backend = "sidecar" if self._caption_sidecar is not None else "source"
            caption_workers = (
                self._caption_sidecar_workers
                if self._caption_sidecar is not None
                else self._caption_workers[domain]
            )
            print(
                "OFFLINE_ARB_WINDOW_PROFILE "
                f"rank={self.rank} domain={domain} start={task.start} stop={task.stop} "
                f"n={len(task.items)} caption_backend={caption_backend} "
                f"caption_workers={caption_workers} "
                f"catalog_s={profile_catalog_done - profile_started:.6f} "
                f"caption_s={profile_caption_done - profile_catalog_done:.6f} "
                f"image_s={profile_image_done - profile_caption_done:.6f} "
                f"total_s={profile_total_done - profile_started:.6f}",
                flush=True,
            )
        if self._telemetry_enabled:
            caption_backend = "sidecar" if self._caption_sidecar is not None else "source"
            caption_workers = (
                self._caption_sidecar_workers
                if self._caption_sidecar is not None
                else self._caption_workers[domain]
            )
            source_counts = Counter(
                str(row.get("sub_source"))
                for _item, row in pairs
                if row.get("sub_source") is not None
            )
            window_record: dict[str, Any] = {
                "record_type": "window",
                "rank": self.rank,
                "domain": domain,
                "start": task.start,
                "stop": task.stop,
                "n": len(task.items),
                "caption_backend": caption_backend,
                "caption_workers": caption_workers,
                "catalog_s": profile_catalog_done - profile_started,
                "caption_s": profile_caption_done - profile_catalog_done,
                "image_s": profile_image_done - profile_caption_done,
                "total_s": profile_total_done - profile_started,
                "sub_source_counts": dict(sorted(source_counts.items())),
                "image_detail": image_detail,
            }
            return window_record
        return None

    def _anime_caption_path(self, path: str) -> str:
        if self._anime_caption_root is None:
            return path
        source_path = Path(path)
        if source_path.parent != _ANIME_CAPTION_SOURCE_PARENT:
            raise OfflineLoaderError(
                "anime caption local mapping only accepts sources whose parent is "
                f"{_ANIME_CAPTION_SOURCE_PARENT}; got {source_path}"
            )
        local_path = self._anime_caption_root / source_path.name
        if not local_path.is_file():
            raise OfflineLoaderError(
                f"mapped anime caption parquet does not exist: {local_path} "
                f"(source={source_path})"
            )
        return str(local_path)

    @staticmethod
    def _read_source_rows(
        pairs: Sequence[tuple[_PlannedItem, dict[str, Any]]],
        *,
        path_field: str,
        row_group_field: str,
        row_in_group_field: str,
        columns: Sequence[str],
        workers: int = 1,
        path_mapper: Callable[[str], str] | None = None,
    ) -> dict[tuple[int, int], dict[str, Any]]:
        groups: dict[tuple[str, int], list[tuple[tuple[int, int], int]]] = defaultdict(list)
        mapped_paths: dict[str, str] = {}
        for item, row in pairs:
            source_path = str(row[path_field])
            if path_mapper is None:
                path = source_path
            else:
                path = mapped_paths.get(source_path)
                if path is None:
                    path = path_mapper(source_path)
                    mapped_paths[source_path] = path
            rg = int(row[row_group_field])
            within = int(row[row_in_group_field])
            if not path or rg < 0 or within < 0:
                raise OfflineLoaderError(
                    f"invalid parquet locator row_id={item.row_id}: {path!r}/{rg}/{within}"
                )
            groups[(path, rg)].append(((item.step, item.slot), within))

        def read_one(entry):
            (path, rg), requests = entry
            try:
                # One ParquetFile per task: no global lock and no native object is
                # shared across textatlas row-group worker threads.
                pf = pq.ParquetFile(path)
                names = set(pf.schema_arrow.names)
                missing = sorted(set(columns) - names)
                if missing:
                    raise OfflineLoaderError(
                        f"source parquet {path} missing columns {missing}"
                    )
                if rg >= pf.num_row_groups:
                    raise OfflineLoaderError(
                        f"source row group out of range: {path}:{rg}/{pf.num_row_groups}"
                    )
                table = pf.read_row_group(rg, columns=list(columns), use_threads=False)
            except OfflineLoaderError:
                raise
            except Exception as exc:
                raise OfflineLoaderError(
                    f"cannot read source parquet row group {path}:{rg}: {exc}"
                ) from exc
            out = {}
            for token, within in requests:
                if within >= table.num_rows:
                    raise OfflineLoaderError(
                        f"source row_in_group out of range: {path}:{rg}:{within}/{table.num_rows}"
                    )
                out[token] = table.slice(within, 1).to_pylist()[0]
            return out

        result: dict[tuple[int, int], dict[str, Any]] = {}
        entries = list(groups.items())
        if workers <= 1 or len(entries) <= 1:
            for entry in entries:
                result.update(read_one(entry))
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(workers, len(entries)),
                thread_name_prefix="offline-parquet-rg",
            ) as pool:
                futures = [pool.submit(read_one, entry) for entry in entries]
                for future in concurrent.futures.as_completed(futures):
                    result.update(future.result())
        return result

    def _anime_general_freq(self, path: str) -> Counter:
        cached = self._anime_freq_cache.get(path)
        if cached is not None:
            return cached
        if self._anime_freq_root is not None:
            freq_path = self._anime_freq_root / (
                Path(path).name.removesuffix(".parquet") + ".json.gz"
            )
            try:
                with gzip.open(freq_path, "rt", encoding="utf-8") as handle:
                    raw_freq = json.load(handle)
                if not isinstance(raw_freq, dict) or any(
                    not isinstance(tag, str)
                    or type(count) is not int
                    or count <= 0
                    for tag, count in raw_freq.items()
                ):
                    raise ValueError("expected an object of positive integer tag counts")
                freq = Counter(raw_freq)
            except Exception as exc:
                raise OfflineLoaderError(
                    f"cannot load precomputed anime shard tag frequency {freq_path}: {exc}"
                ) from exc
        else:
            try:
                pf = pq.ParquetFile(path)
                if "tags_general" not in pf.schema_arrow.names:
                    raise OfflineLoaderError(f"anime caption source lacks tags_general: {path}")
                freq = Counter()
                for rg in range(pf.num_row_groups):
                    table = pf.read_row_group(rg, columns=["tags_general"], use_threads=False)
                    for tags in table.column("tags_general").to_pylist():
                        if tags:
                            freq.update(tags)
            except OfflineLoaderError:
                raise
            except Exception as exc:
                raise OfflineLoaderError(
                    f"cannot build deterministic anime shard tag frequency {path}: {exc}"
                ) from exc
        self._anime_freq_cache[path] = freq
        return freq

    def _caption_records(
        self,
        domain: str,
        pairs: Sequence[tuple[_PlannedItem, dict[str, Any]]],
    ) -> dict[tuple[int, int], dict[str, Any]]:
        columns = _CAPTION_SOURCE_COLUMNS.get(domain)
        if columns is None:
            raise AssertionError(domain)
        if self._caption_sidecar is None:
            source = self._read_source_rows(
                pairs,
                path_field="caption_parquet_path",
                row_group_field="caption_row_group",
                row_in_group_field="caption_row_in_group",
                columns=columns,
                workers=self._caption_workers[domain],
                path_mapper=self._anime_caption_path if domain == "anime" else None,
            )
        else:
            try:
                sidecar_rows = self._caption_sidecar.get_many(
                    domain,
                    [item.row_id for item, _row in pairs],
                )
            except CaptionSidecarError as exc:
                raise OfflineLoaderError(
                    f"caption sidecar read failed for domain={domain}: {exc}"
                ) from exc
            source = {
                (item.step, item.slot): sidecar_rows[item.row_id]
                for item, _row in pairs
            }
        result: dict[tuple[int, int], dict[str, Any]] = {}
        for item, row in pairs:
            token = (item.step, item.slot)
            values = source[token]
            if domain == "anime":
                fields = {key: values.get(key) for key in _ANIME_CAP_KEYS}
                fields["rating_bucket"] = row.get("rating_bucket")
                fields["year"] = row.get("year")
                result[token] = {
                    "row_id": int(item.row_id),
                    "cap_fields": fields,
                    "general_freq": self._anime_general_freq(str(row["caption_parquet_path"])),
                    "key": row["key"],
                    "cluster": row.get("cluster"),
                }
            elif domain == "textatlas":
                result[token] = {
                    "row_id": int(item.row_id),
                    "uid": row["uid"],
                    "captions": tuple(values.get(f"caption{i}") for i in range(1, 6)),
                }
            else:
                band_captioner = [int(row[f"band_captioner_t{i}"]) for i in range(1, 5)]
                if domain == "gpic":
                    cap_texts = {CAPTIONER_CODE["i1"]: values.get("caption")}
                elif domain == "synthetic":
                    cap_texts = {CAPTIONER_CODE["original"]: values.get("caption_original")}
                else:
                    cap_texts = {
                        int(code): values.get(column)
                        for code, column in _MONET_SRC_COL.items()
                    }
                result[token] = {
                    "row_id": int(item.row_id),
                    "uid": row["uid"],
                    "served_mask": int(row["served_mask"]),
                    "base_w": float(row["base_weight"]),
                    "band_captioner": band_captioner,
                    "cap_texts": cap_texts,
                }
        return result

    def _image_payloads(
        self,
        domain: str,
        pairs: Sequence[tuple[_PlannedItem, dict[str, Any]]],
        generation: int,
        *,
        telemetry: bool = False,
    ) -> (
        dict[tuple[int, int], bytes | str]
        | tuple[dict[tuple[int, int], bytes | str], dict[str, Any]]
    ):
        if domain in {"monet_real", "synthetic"}:
            return self._coalesced_tar_payloads(pairs, telemetry=telemetry)
        if domain == "gpic":
            if self._gpic_payload_backend == "offline_gpic_blob_v1":
                if self._gpic_blob is None:
                    raise OfflineLoaderError(
                        "offline_gpic_blob_v1 backend lost its required overlay"
                    )
                return self._gpic_blob_payloads(pairs, telemetry=telemetry)
            if self._gpic_payload_in_memory:
                return self._gpic_payloads_in_memory(pairs, telemetry=telemetry)
            result = self._stage_gpic_payloads(pairs, generation)
            if telemetry:
                return result, {
                    "kind": "gpic_staged",
                    "files": len(pairs),
                    "workers": min(self._gpic_workers, max(1, len(pairs))),
                }
            return result
        if domain == "anime":
            result = self._anime_payloads(pairs)
            if telemetry:
                return result, {
                    "kind": "anime_fetch",
                    "files": len(pairs),
                    "workers": min(self._anime_workers, max(1, len(pairs))),
                }
            return result
        if domain == "textatlas":
            source = self._read_source_rows(
                pairs,
                path_field="image_parquet_path",
                row_group_field="image_row_group",
                row_in_group_field="image_row_in_group",
                columns=["image", "image_path"],
                workers=_TEXTATLAS_ROWGROUP_WORKERS,
            )
            result: dict[tuple[int, int], bytes] = {}
            for item, row in pairs:
                token = (item.step, item.slot)
                values = source[token]
                if str(values.get("image_path")) != str(row["image_path"]):
                    raise OfflineLoaderError(
                        f"textatlas image locator mismatch row_id={item.row_id}: "
                        f"catalog={row['image_path']!r}, source={values.get('image_path')!r}"
                    )
                image = values.get("image")
                raw = image.get("bytes") if isinstance(image, Mapping) else None
                if not raw:
                    raise OfflineLoaderError(f"empty textatlas image bytes row_id={item.row_id}")
                result[token] = bytes(raw)
            if telemetry:
                return result, {
                    "kind": "textatlas_parquet",
                    "files": len(pairs),
                    "workers": min(_TEXTATLAS_ROWGROUP_WORKERS, max(1, len(pairs))),
                }
            return result
        raise AssertionError(domain)

    @staticmethod
    def _read_exact(handle, need: int, context: str) -> bytes:
        parts: list[bytes] = []
        remaining = need
        while remaining:
            block = handle.read(min(_COALESCE_CHUNK_BYTES, remaining))
            if not block:
                raise OfflineLoaderError(
                    f"short ranged read for {context}: wanted={need}, got={need - remaining}"
                )
            parts.append(block)
            remaining -= len(block)
        return b"".join(parts)

    def _coalesced_tar_payloads(
        self,
        pairs: Sequence[tuple[_PlannedItem, dict[str, Any]]],
        *,
        telemetry: bool = False,
    ) -> (
        dict[tuple[int, int], bytes]
        | tuple[dict[tuple[int, int], bytes], dict[str, Any]]
    ):
        if telemetry:
            return self._coalesced_tar_payloads_instrumented(pairs)
        return self._coalesced_tar_payloads_plain(pairs)

    def _coalesced_tar_payloads_plain(
        self,
        pairs: Sequence[tuple[_PlannedItem, dict[str, Any]]],
    ) -> dict[tuple[int, int], bytes]:
        """Original formal path: no diagnostic clocks, counters, or objects."""

        by_tar: dict[str, list[tuple[_PlannedItem, dict[str, Any]]]] = {}
        for item, row in pairs:
            by_tar.setdefault(str(row["tar_path"]), []).append((item, row))

        def read_one_tar(
            tar_path: str,
            entries: list[tuple[_PlannedItem, dict[str, Any]]],
        ) -> dict[tuple[int, int], bytes]:
            locator_tokens: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
            offset_sizes: dict[int, int] = {}
            for item, row in entries:
                offset = int(row["tar_offset"])
                size = int(row["tar_size"])
                if offset < 0 or size <= 0:
                    raise OfflineLoaderError(
                        f"invalid tar locator row_id={item.row_id}: {tar_path}@{offset}+{size}"
                    )
                previous = offset_sizes.setdefault(offset, size)
                if previous != size:
                    raise OfflineLoaderError(
                        f"conflicting tar locator sizes at {tar_path}@{offset}: {previous}/{size}"
                    )
                locator_tokens[(offset, size)].append((item.step, item.slot))
            members = sorted(locator_tokens)
            for (off0, size0), (off1, _size1) in zip(members, members[1:]):
                if off1 < off0 + size0:
                    raise OfflineLoaderError(
                        f"overlapping tar locators in prefetch window: {tar_path}@{off0} and @{off1}"
                    )

            runs: list[tuple[int, int, list[tuple[int, int]]]] = []
            cur_start = cur_end = -1
            current: list[tuple[int, int]] = []
            for offset, size in members:
                end = offset + size
                if (
                    current
                    and offset - cur_end < _COALESCE_GAP_BYTES
                    and end - cur_start <= _COALESCE_MAX_RUN_BYTES
                ):
                    cur_end = end
                    current.append((offset, size))
                else:
                    if current:
                        runs.append((cur_start, cur_end, current))
                    cur_start, cur_end, current = offset, end, [(offset, size)]
            if current:
                runs.append((cur_start, cur_end, current))

            tar_result: dict[tuple[int, int], bytes] = {}
            try:
                with open(tar_path, "rb", buffering=0) as handle:
                    for start, end, run_members in runs:
                        handle.seek(start)
                        buf = self._read_exact(
                            handle,
                            end - start,
                            f"tar={tar_path},run=[{start},{end})",
                        )
                        for offset, size in run_members:
                            lo = offset - start
                            payload = buf[lo : lo + size]
                            if len(payload) != size:
                                raise OfflineLoaderError(
                                    f"short member slice {tar_path}@{offset}+{size}"
                                )
                            for token in locator_tokens[(offset, size)]:
                                tar_result[token] = payload
            except OfflineLoaderError:
                raise
            except OSError as exc:
                raise OfflineLoaderError(
                    f"coalesced tar read failed for {tar_path}: {exc}"
                ) from exc
            return tar_result

        result: dict[tuple[int, int], bytes] = {}
        if len(by_tar) <= 1 or self._tar_workers <= 1:
            for tar_path, entries in by_tar.items():
                result.update(read_one_tar(tar_path, entries))
            return result
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self._tar_workers, len(by_tar)),
            thread_name_prefix=f"offline-tar-r{self.rank}",
        ) as pool:
            futures = [
                pool.submit(read_one_tar, tar_path, entries)
                for tar_path, entries in by_tar.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                result.update(future.result())
        return result

    def _coalesced_tar_payloads_instrumented(
        self,
        pairs: Sequence[tuple[_PlannedItem, dict[str, Any]]],
    ) -> tuple[dict[tuple[int, int], bytes], dict[str, Any]]:
        telemetry = True
        wall_started = time.perf_counter()
        by_tar: dict[str, list[tuple[_PlannedItem, dict[str, Any]]]] = {}
        for item, row in pairs:
            by_tar.setdefault(str(row["tar_path"]), []).append((item, row))

        def read_one_tar(
            tar_path: str,
            entries: list[tuple[_PlannedItem, dict[str, Any]]],
        ) -> (
            dict[tuple[int, int], bytes]
            | tuple[dict[tuple[int, int], bytes], dict[str, Any]]
        ):
            tar_started = time.perf_counter() if telemetry else 0.0
            locator_tokens: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
            offset_sizes: dict[int, int] = {}
            for item, row in entries:
                offset = int(row["tar_offset"])
                size = int(row["tar_size"])
                if offset < 0 or size <= 0:
                    raise OfflineLoaderError(
                        f"invalid tar locator row_id={item.row_id}: {tar_path}@{offset}+{size}"
                    )
                previous = offset_sizes.setdefault(offset, size)
                if previous != size:
                    raise OfflineLoaderError(
                        f"conflicting tar locator sizes at {tar_path}@{offset}: {previous}/{size}"
                    )
                locator_tokens[(offset, size)].append((item.step, item.slot))
            members = sorted(locator_tokens)
            for (off0, size0), (off1, _size1) in zip(members, members[1:]):
                if off1 < off0 + size0:
                    raise OfflineLoaderError(
                        f"overlapping tar locators in prefetch window: {tar_path}@{off0} and @{off1}"
                    )

            runs: list[tuple[int, int, list[tuple[int, int]]]] = []
            cur_start = cur_end = -1
            current: list[tuple[int, int]] = []
            for offset, size in members:
                end = offset + size
                if (
                    current
                    and offset - cur_end < _COALESCE_GAP_BYTES
                    and end - cur_start <= _COALESCE_MAX_RUN_BYTES
                ):
                    cur_end = end
                    current.append((offset, size))
                else:
                    if current:
                        runs.append((cur_start, cur_end, current))
                    cur_start, cur_end, current = offset, end, [(offset, size)]
            if current:
                runs.append((cur_start, cur_end, current))

            tar_result: dict[tuple[int, int], bytes] = {}
            open_s = 0.0
            seek_s = 0.0
            read_s = 0.0
            try:
                if telemetry:
                    operation_started = time.perf_counter()
                    handle = open(tar_path, "rb", buffering=0)
                    open_s += time.perf_counter() - operation_started
                else:
                    handle = open(tar_path, "rb", buffering=0)
                with handle:
                    for start, end, run_members in runs:
                        if telemetry:
                            operation_started = time.perf_counter()
                            handle.seek(start)
                            seek_s += time.perf_counter() - operation_started
                            operation_started = time.perf_counter()
                            buf = self._read_exact(
                                handle,
                                end - start,
                                f"tar={tar_path},run=[{start},{end})",
                            )
                            read_s += time.perf_counter() - operation_started
                        else:
                            handle.seek(start)
                            buf = self._read_exact(
                                handle,
                                end - start,
                                f"tar={tar_path},run=[{start},{end})",
                            )
                        for offset, size in run_members:
                            lo = offset - start
                            payload = buf[lo : lo + size]
                            if len(payload) != size:
                                raise OfflineLoaderError(
                                    f"short member slice {tar_path}@{offset}+{size}"
                                )
                            for token in locator_tokens[(offset, size)]:
                                tar_result[token] = payload
            except OfflineLoaderError:
                raise
            except OSError as exc:
                raise OfflineLoaderError(f"coalesced tar read failed for {tar_path}: {exc}") from exc
            if telemetry:
                return tar_result, {
                    "locator_requests": len(entries),
                    "unique_locators": len(members),
                    "runs": len(runs),
                    "member_bytes": sum(
                        int(row["tar_size"]) for _item, row in entries
                    ),
                    "requested_bytes": sum(size for _offset, size in members),
                    "actual_read_bytes": sum(end - start for start, end, _members in runs),
                    "open_s": open_s,
                    "seek_s": seek_s,
                    "read_s": read_s,
                    "tar_total_s": time.perf_counter() - tar_started,
                }
            return tar_result

        # gpuval r3 实证:随机 plan 使窗口项散布数百 tar,串行逐 tar 读 = 每窗口 60-100s 停顿。
        # 逐 tar 并行(单 tar 内 run 顺序不变,结果主调线程合并,无跨线程共享写)。
        result: dict[tuple[int, int], bytes] = {}
        stats: list[dict[str, Any]] = []

        def merge(value) -> None:
            if telemetry:
                payloads, detail = value
                result.update(payloads)
                stats.append(detail)
            else:
                result.update(value)

        if len(by_tar) <= 1 or self._tar_workers <= 1:
            for tar_path, entries in by_tar.items():
                merge(read_one_tar(tar_path, entries))
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self._tar_workers, len(by_tar)),
                thread_name_prefix=f"offline-tar-r{self.rank}",
            ) as pool:
                futures = [
                    pool.submit(read_one_tar, tar_path, entries)
                    for tar_path, entries in by_tar.items()
                ]
                for future in concurrent.futures.as_completed(futures):
                    merge(future.result())
        if telemetry:
            wall_s = time.perf_counter() - wall_started
            thread_total_s = sum(float(value["tar_total_s"]) for value in stats)
            return result, {
                "kind": "coalesced_tar",
                "unique_tars": len(by_tar),
                "locator_requests": sum(int(value["locator_requests"]) for value in stats),
                "unique_locators": sum(int(value["unique_locators"]) for value in stats),
                "runs": sum(int(value["runs"]) for value in stats),
                "member_bytes": sum(int(value["member_bytes"]) for value in stats),
                "requested_bytes": sum(int(value["requested_bytes"]) for value in stats),
                "actual_read_bytes": sum(int(value["actual_read_bytes"]) for value in stats),
                "thread_sum_s": {
                    "open": sum(float(value["open_s"]) for value in stats),
                    "seek": sum(float(value["seek_s"]) for value in stats),
                    "read": sum(float(value["read_s"]) for value in stats),
                    "tar_total": thread_total_s,
                },
                "max_tar_s": max(
                    (float(value["tar_total_s"]) for value in stats),
                    default=0.0,
                ),
                "operation_max_s": {
                    "open": max(
                        (float(value["open_s"]) for value in stats), default=0.0
                    ),
                    "seek": max(
                        (float(value["seek_s"]) for value in stats), default=0.0
                    ),
                    "read": max(
                        (float(value["read_s"]) for value in stats), default=0.0
                    ),
                    "tar_total": max(
                        (float(value["tar_total_s"]) for value in stats), default=0.0
                    ),
                },
                "wall_s": wall_s,
                "effective_concurrency": thread_total_s / wall_s if wall_s > 0 else 0.0,
                "workers": min(self._tar_workers, max(1, len(by_tar))),
            }
        return result

    @staticmethod
    def _payload_token_closure(
        pairs: Sequence[tuple[_PlannedItem, dict[str, Any]]],
        payloads: Mapping[tuple[int, int], Any],
        context: str,
    ) -> None:
        expected = [(item.step, item.slot) for item, _row in pairs]
        if len(set(expected)) != len(expected):
            raise OfflineLoaderError(
                f"{context} request contains duplicate (step,slot) tokens"
            )
        if set(payloads) != set(expected) or len(payloads) != len(expected):
            raise OfflineLoaderError(
                f"{context} token closure mismatch: "
                f"missing={sorted(set(expected)-set(payloads))[:8]}, "
                f"unexpected={sorted(set(payloads)-set(expected))[:8]}"
            )

    def _gpic_blob_payloads(
        self,
        pairs: Sequence[tuple[_PlannedItem, dict[str, Any]]],
        *,
        telemetry: bool = False,
    ) -> (
        dict[tuple[int, int], bytes]
        | tuple[dict[tuple[int, int], bytes], dict[str, Any]]
    ):
        overlay = self._gpic_blob
        if overlay is None or self._gpic_payload_backend != "offline_gpic_blob_v1":
            raise OfflineLoaderError("GPIC blob read requested without the bound backend")
        before = overlay.snapshot() if telemetry else None

        def read_one(pair):
            item, _row = pair
            try:
                raw = overlay.read(int(item.row_id))
            except GpicBlobError as exc:
                raise OfflineLoaderError(
                    f"GPIC blob read failed row_id={item.row_id}: {exc}"
                ) from exc
            return (item.step, item.slot), raw

        result: dict[tuple[int, int], bytes] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self._gpic_workers, max(1, len(pairs))),
            thread_name_prefix=f"offline-gpic-blob-r{self.rank}",
        ) as pool:
            for token, raw in pool.map(read_one, pairs):
                if token in result:
                    raise OfflineLoaderError(
                        f"GPIC blob reader returned duplicate token {token}"
                    )
                if not isinstance(raw, bytes) or not raw:
                    raise OfflineLoaderError(
                        f"GPIC blob reader returned empty/non-bytes token={token}"
                    )
                result[token] = raw
        self._payload_token_closure(pairs, result, "GPIC blob payload")
        if not telemetry:
            return result

        assert before is not None
        after = overlay.snapshot()
        cumulative_keys = {
            "read_items",
            "read_bytes",
            "pread_requested_bytes",
            "pread_returned_bytes",
            "read_seconds_micro",
            "integrity_failures",
            "fd_hits",
            "fd_misses",
            "fd_opens",
            "fd_evictions",
            "fd_acquire_timeouts",
        }
        delta = {
            key: after.get(key, 0.0) - before.get(key, 0.0)
            for key in sorted(cumulative_keys)
        }
        requested = delta["pread_requested_bytes"]
        returned = delta["pread_returned_bytes"]
        return result, {
            "kind": "gpic_blob_v1",
            "files": len(pairs),
            "workers": min(self._gpic_workers, max(1, len(pairs))),
            "selection_mode": "full_catalog",
            "source_fallback_items": 0,
            "manifest_sha256": overlay.manifest_sha256,
            "release_sha256": overlay.release_sha256,
            "delta": delta,
            "read_amplification": returned / requested if requested > 0 else 0.0,
            "fd_current": after.get("fd_current", 0.0),
            "fd_peak": after.get("fd_peak", 0.0),
        }

    def _stage_gpic_payloads(
        self,
        pairs: Sequence[tuple[_PlannedItem, dict[str, Any]]],
        generation: int,
    ) -> dict[tuple[int, int], str]:
        local_dir = self._local_base / f"generation_{generation:04d}" / "gpic"
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OfflineLoaderError(f"cannot create gpic /local rolling window {local_dir}: {exc}") from exc

        def stage_one(pair):
            item, row = pair
            source = str(row["image_path"])
            destination = local_dir / (
                f"s{item.step:09d}_k{item.slot:04d}_r{item.row_id:012d}.img"
            )
            temporary = destination.with_name(
                f"{destination.name}.tmp.{os.getpid()}.{threading.get_ident()}"
            )
            total = 0
            try:
                with open(source, "rb", buffering=0) as src, open(temporary, "xb", buffering=0) as dst:
                    while True:
                        block = src.read(_COALESCE_CHUNK_BYTES)
                        if not block:
                            break
                        dst.write(block)
                        total += len(block)
                if total <= 0:
                    raise OfflineLoaderError(
                        f"empty gpic source row_id={item.row_id}: {source}"
                    )
                os.replace(temporary, destination)
            except OfflineLoaderError:
                try:
                    temporary.unlink()
                except OSError:
                    pass
                raise
            except OSError as exc:
                try:
                    temporary.unlink()
                except OSError:
                    pass
                raise OfflineLoaderError(
                    f"gpic stage failed row_id={item.row_id}, source={source}: {exc}"
                ) from exc
            return (item.step, item.slot), str(destination)

        result: dict[tuple[int, int], str] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self._gpic_workers, max(1, len(pairs))),
            thread_name_prefix=f"offline-gpic-r{self.rank}",
        ) as pool:
            for token, path in pool.map(stage_one, pairs):
                result[token] = path
        return result

    def _gpic_payloads_in_memory(
        self,
        pairs: Sequence[tuple[_PlannedItem, dict[str, Any]]],
        *,
        telemetry: bool = False,
    ) -> (
        dict[tuple[int, int], bytes]
        | tuple[dict[tuple[int, int], bytes], dict[str, Any]]
    ):
        if telemetry:
            return self._gpic_payloads_in_memory_instrumented(pairs)
        return self._gpic_payloads_in_memory_plain(pairs)

    def _gpic_payloads_in_memory_plain(
        self,
        pairs: Sequence[tuple[_PlannedItem, dict[str, Any]]],
    ) -> dict[tuple[int, int], bytes]:
        """Original formal path: exact reads without diagnostic instrumentation."""

        consistency_mode = _validated_gpic_source_consistency_mode(
            self._gpic_source_consistency_mode
        )

        def read_one(pair):
            item, row = pair
            source = str(row["image_path"])
            parts: list[bytes] = []
            total = 0
            try:
                with open(source, "rb", buffering=0) as handle:
                    before = os.fstat(handle.fileno())
                    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                        raise OfflineLoaderError(
                            f"invalid gpic source row_id={item.row_id}: "
                            f"path={source}, bytes={before.st_size}"
                        )
                    before_identity = (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    )
                    if consistency_mode == "strict_path":
                        path_before = os.stat(source, follow_symlinks=True)
                        if (
                            not stat.S_ISREG(path_before.st_mode)
                            or (
                                path_before.st_dev,
                                path_before.st_ino,
                                path_before.st_size,
                                path_before.st_mtime_ns,
                                path_before.st_ctime_ns,
                            )
                            != before_identity
                        ):
                            raise OfflineLoaderError(
                                "gpic source path changed while opening "
                                f"row_id={item.row_id}: {source}"
                            )
                    while True:
                        block = handle.read(_COALESCE_CHUNK_BYTES)
                        if not block:
                            break
                        parts.append(block)
                        total += len(block)
                    after = os.fstat(handle.fileno())
                    if consistency_mode == "strict_path":
                        path_after = os.stat(source, follow_symlinks=True)
            except OfflineLoaderError:
                raise
            except OSError as exc:
                raise OfflineLoaderError(
                    f"gpic memory read failed row_id={item.row_id}, "
                    f"source={source}: {exc}"
                ) from exc
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            fd_changed = (
                not stat.S_ISREG(after.st_mode)
                or after_identity != before_identity
            )
            path_changed = False
            if consistency_mode == "strict_path":
                path_after_identity = (
                    path_after.st_dev,
                    path_after.st_ino,
                    path_after.st_size,
                    path_after.st_mtime_ns,
                    path_after.st_ctime_ns,
                )
                path_changed = (
                    not stat.S_ISREG(path_after.st_mode)
                    or path_after_identity != before_identity
                )
            if fd_changed or path_changed:
                raise OfflineLoaderError(
                    f"gpic source changed while reading row_id={item.row_id}: {source}"
                )
            if total != before.st_size:
                raise OfflineLoaderError(
                    f"short gpic source read row_id={item.row_id}: "
                    f"source={source}, read={total}, expected={before.st_size}"
                )
            raw = b"".join(parts)
            if not raw:
                raise OfflineLoaderError(
                    f"empty gpic source row_id={item.row_id}: {source}"
                )
            return (item.step, item.slot), raw

        result: dict[tuple[int, int], bytes] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self._gpic_workers, max(1, len(pairs))),
            thread_name_prefix=f"offline-gpic-memory-r{self.rank}",
        ) as pool:
            for token, raw in pool.map(read_one, pairs):
                result[token] = raw
        return result

    def _gpic_payloads_in_memory_instrumented(
        self,
        pairs: Sequence[tuple[_PlannedItem, dict[str, Any]]],
    ) -> tuple[dict[tuple[int, int], bytes], dict[str, Any]]:
        """Read exact GPIC source bytes without the rolling /local write/read cycle."""

        telemetry = True
        consistency_mode = _validated_gpic_source_consistency_mode(
            self._gpic_source_consistency_mode
        )
        wall_started = time.perf_counter()

        def read_one(pair):
            item, row = pair
            source = str(row["image_path"])
            parts: list[bytes] = []
            total = 0
            file_started = time.perf_counter() if telemetry else 0.0
            open_s = 0.0
            fstat_s = 0.0
            path_stat_s = 0.0
            read_s = 0.0
            fstat_calls = 0
            path_stat_calls = 0
            try:
                if telemetry:
                    operation_started = time.perf_counter()
                    handle = open(source, "rb", buffering=0)
                    open_s += time.perf_counter() - operation_started
                else:
                    handle = open(source, "rb", buffering=0)
                with handle:
                    if telemetry:
                        operation_started = time.perf_counter()
                        before = os.fstat(handle.fileno())
                        fstat_calls += 1
                        fstat_s += time.perf_counter() - operation_started
                    else:
                        before = os.fstat(handle.fileno())
                    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                        raise OfflineLoaderError(
                            f"invalid gpic source row_id={item.row_id}: "
                            f"path={source}, bytes={before.st_size}"
                        )
                    before_identity = (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    )
                    if consistency_mode == "strict_path":
                        if telemetry:
                            operation_started = time.perf_counter()
                            path_before = os.stat(source, follow_symlinks=True)
                            path_stat_calls += 1
                            path_stat_s += time.perf_counter() - operation_started
                        else:
                            path_before = os.stat(source, follow_symlinks=True)
                        if (
                            not stat.S_ISREG(path_before.st_mode)
                            or (
                                path_before.st_dev,
                                path_before.st_ino,
                                path_before.st_size,
                                path_before.st_mtime_ns,
                                path_before.st_ctime_ns,
                            )
                            != before_identity
                        ):
                            raise OfflineLoaderError(
                                "gpic source path changed while opening "
                                f"row_id={item.row_id}: {source}"
                            )
                    while True:
                        if telemetry:
                            operation_started = time.perf_counter()
                            block = handle.read(_COALESCE_CHUNK_BYTES)
                            read_s += time.perf_counter() - operation_started
                        else:
                            block = handle.read(_COALESCE_CHUNK_BYTES)
                        if not block:
                            break
                        parts.append(block)
                        total += len(block)
                    if telemetry:
                        operation_started = time.perf_counter()
                        after = os.fstat(handle.fileno())
                        fstat_calls += 1
                        fstat_s += time.perf_counter() - operation_started
                        if consistency_mode == "strict_path":
                            operation_started = time.perf_counter()
                            path_after = os.stat(source, follow_symlinks=True)
                            path_stat_calls += 1
                            path_stat_s += time.perf_counter() - operation_started
                    else:
                        after = os.fstat(handle.fileno())
                        if consistency_mode == "strict_path":
                            path_after = os.stat(source, follow_symlinks=True)
            except OfflineLoaderError:
                raise
            except OSError as exc:
                raise OfflineLoaderError(
                    f"gpic memory read failed row_id={item.row_id}, "
                    f"source={source}: {exc}"
                ) from exc
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            fd_changed = (
                not stat.S_ISREG(after.st_mode)
                or after_identity != before_identity
            )
            path_changed = False
            if consistency_mode == "strict_path":
                path_after_identity = (
                    path_after.st_dev,
                    path_after.st_ino,
                    path_after.st_size,
                    path_after.st_mtime_ns,
                    path_after.st_ctime_ns,
                )
                path_changed = (
                    not stat.S_ISREG(path_after.st_mode)
                    or path_after_identity != before_identity
                )
            if fd_changed or path_changed:
                raise OfflineLoaderError(
                    f"gpic source changed while reading row_id={item.row_id}: {source}"
                )
            if total != before.st_size:
                raise OfflineLoaderError(
                    f"short gpic source read row_id={item.row_id}: "
                    f"source={source}, read={total}, expected={before.st_size}"
                )
            raw = b"".join(parts)
            if not raw:
                raise OfflineLoaderError(
                    f"empty gpic source row_id={item.row_id}: {source}"
                )
            if telemetry:
                return (item.step, item.slot), raw, {
                    "bytes": total,
                    "open_s": open_s,
                    "fstat_s": fstat_s,
                    "path_stat_s": path_stat_s,
                    "read_s": read_s,
                    "fstat_calls": fstat_calls,
                    "path_stat_calls": path_stat_calls,
                    "file_total_s": time.perf_counter() - file_started,
                }
            return (item.step, item.slot), raw

        result: dict[tuple[int, int], bytes] = {}
        stats: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self._gpic_workers, max(1, len(pairs))),
            thread_name_prefix=f"offline-gpic-memory-r{self.rank}",
        ) as pool:
            if telemetry:
                for token, raw, detail in pool.map(read_one, pairs):
                    result[token] = raw
                    stats.append(detail)
            else:
                for token, raw in pool.map(read_one, pairs):
                    result[token] = raw
        if telemetry:
            wall_s = time.perf_counter() - wall_started
            thread_total_s = sum(float(value["file_total_s"]) for value in stats)
            return result, {
                "kind": "gpic_in_memory",
                "files": len(stats),
                "unique_files": len({str(row["image_path"]) for _item, row in pairs}),
                "bytes": sum(int(value["bytes"]) for value in stats),
                "gpic_source_consistency_mode": consistency_mode,
                "fstat_calls": sum(int(value["fstat_calls"]) for value in stats),
                "path_stat_calls": sum(
                    int(value["path_stat_calls"]) for value in stats
                ),
                "thread_sum_s": {
                    "open": sum(float(value["open_s"]) for value in stats),
                    "fstat": sum(float(value["fstat_s"]) for value in stats),
                    "path_stat": sum(float(value["path_stat_s"]) for value in stats),
                    "read": sum(float(value["read_s"]) for value in stats),
                    "file_total": thread_total_s,
                },
                "max_file_s": max(
                    (float(value["file_total_s"]) for value in stats),
                    default=0.0,
                ),
                "operation_max_s": {
                    "open": max(
                        (float(value["open_s"]) for value in stats), default=0.0
                    ),
                    "fstat": max(
                        (float(value["fstat_s"]) for value in stats), default=0.0
                    ),
                    "path_stat": max(
                        (float(value["path_stat_s"]) for value in stats), default=0.0
                    ),
                    "read": max(
                        (float(value["read_s"]) for value in stats), default=0.0
                    ),
                    "file_total": max(
                        (float(value["file_total_s"]) for value in stats), default=0.0
                    ),
                },
                "wall_s": wall_s,
                "effective_concurrency": thread_total_s / wall_s if wall_s > 0 else 0.0,
                "workers": min(self._gpic_workers, max(1, len(pairs))),
            }
        return result

    def _anime_payloads(
        self,
        pairs: Sequence[tuple[_PlannedItem, dict[str, Any]]],
    ) -> dict[tuple[int, int], bytes]:
        def fetch(pair):
            item, row = pair
            url = str(row["fetch_url"])
            rounds = len(_ANIME_EXACT_REFETCH_DELAYS_S) + 1
            for attempt in range(1, rounds + 1):
                raw = fetch_url_bytes(url, cache_dir=self.anime_cache_dir)
                if raw:
                    if attempt > 1:
                        warnings.warn(
                            "anime exact-row fetch recovered "
                            f"row_id={item.row_id},attempt={attempt}/{rounds}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    return (item.step, item.slot), bytes(raw)
                if attempt < rounds:
                    delay_s = _ANIME_EXACT_REFETCH_DELAYS_S[attempt - 1]
                    warnings.warn(
                        "anime exact-row fetch retry "
                        f"row_id={item.row_id},attempt={attempt}/{rounds},"
                        f"delay_s={delay_s:g},fetch_stats={fetch_stats_snapshot()}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    time.sleep(delay_s)
            raise OfflineLoaderError(
                "anime fetch failed after bounded exact-row refetch "
                f"row_id={item.row_id},rounds={rounds},"
                f"delays_s={_ANIME_EXACT_REFETCH_DELAYS_S},url={url},"
                f"fetch_stats={fetch_stats_snapshot()}"
            )

        result: dict[tuple[int, int], bytes] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self._anime_workers, max(1, len(pairs))),
            thread_name_prefix=f"offline-anime-r{self.rank}",
        ) as pool:
            for token, raw in pool.map(fetch, pairs):
                result[token] = raw
        return result

    def _publish(self, prepared: _PreparedItem) -> None:
        item = prepared.planned
        with self._condition:
            step_ready = self._ready[item.step]
            if item.slot in step_ready:
                raise OfflineLoaderError(
                    f"duplicate prepared slot step={item.step},slot={item.slot}"
                )
            step_ready[item.slot] = prepared
            self._domain_ready[item.domain] += 1
            self._supply_completed[item.domain] += 1
            self.produced[item.domain] += 1
            self._refresh_metrics_locked()
            self._condition.notify_all()

    def _wait_for_step(self, step: int) -> list[_PreparedItem]:
        expected = self._expected.get(step)
        if expected is None:
            raise OfflineLoaderError(f"step {step} was not scheduled")
        condition_wait_s = 0.0
        condition_wakeups = 0
        deadline = time.monotonic() + self.step_timeout_s
        with self._condition:
            if self._telemetry_enabled:
                entry_snapshot = self._ready_snapshot_locked()
                entry_ready_items = len(self._ready.get(step, {}))
            while len(self._ready.get(step, {})) < expected:
                if self._pipeline_error is not None:
                    exc, detail = self._pipeline_error
                    raise OfflineLoaderError(
                        f"offline supply failed before step {step}: {type(exc).__name__}: {exc}\n{detail}"
                    ) from exc
                if self._stop_event.is_set():
                    raise OfflineLoaderError(f"offline prefetch stopped before step {step} was ready")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    alive = [thread.name for thread in self._threads if thread.is_alive()]
                    raise OfflineLoaderError(
                        f"offline prefetch timed out after {self.step_timeout_s:.1f}s at "
                        f"step={step},rank={self.rank}; metrics={self.buffer_fill}; "
                        f"alive_supply_threads={alive}"
                    )
                if self._telemetry_enabled:
                    condition_started = time.perf_counter()
                    self._condition.wait(timeout=min(1.0, remaining))
                    condition_wait_s += time.perf_counter() - condition_started
                    condition_wakeups += 1
                else:
                    self._condition.wait(timeout=min(1.0, remaining))
            if self._pipeline_error is not None:
                exc, detail = self._pipeline_error
                raise OfflineLoaderError(
                    f"offline supply failed while preparing the plan window for step {step}: "
                    f"{type(exc).__name__}: {exc}\n{detail}"
                ) from exc
            by_slot = self._ready[step]
            if set(by_slot) != set(range(expected)):
                raise OfflineLoaderError(
                    f"ready slot set mismatch at step {step}: {sorted(by_slot)}"
                )
            result = [by_slot[slot] for slot in range(expected)]
            if self._telemetry_enabled:
                exit_snapshot = self._ready_snapshot_locked()
                exit_ready_items = len(by_slot)
        if self._telemetry_enabled:
            if condition_wait_s > self._telemetry_wait_threshold_s:
                self._enqueue_telemetry(
                    {
                        "record_type": "wait",
                        "rank": self.rank,
                        "step": step,
                        "expected_items": expected,
                        "condition_wait_s": condition_wait_s,
                        "condition_wakeups": condition_wakeups,
                        "entry_ready_items": entry_ready_items,
                        "exit_ready_items": exit_ready_items,
                        "entry": entry_snapshot,
                        "exit": exit_snapshot,
                    }
                )
        return result

    @staticmethod
    def _raw_for_item(item: _PreparedItem) -> bytes:
        if item.local_path is None:
            if not item.raw:
                raise OfflineLoaderError(
                    f"empty prefetched payload row_id={item.planned.row_id}"
                )
            return item.raw
        path = Path(item.local_path)
        try:
            with path.open("rb") as handle:
                raw = handle.read()
        except OSError as exc:
            raise OfflineLoaderError(
                f"cannot consume staged gpic row_id={item.planned.row_id}, path={path}: {exc}"
            ) from exc
        finally:
            try:
                path.unlink()
            except OSError:
                pass
        if not raw:
            raise OfflineLoaderError(
                f"empty staged gpic payload row_id={item.planned.row_id}, path={path}"
            )
        return raw

    def _validate_anime_source_bound_native_mismatch(
        self,
        prepared: _PreparedItem,
        raw: bytes,
        decoded_rgb: np.ndarray,
    ) -> str:
        """Bind an Anime native-size mismatch to its exact source pixels.

        This gate is intentionally used only when the existing decoded-pixel
        compatibility path would otherwise fail because upsampling is needed.
        The catalog's source locator and metadata must agree with the exact
        source-Parquet row, and that row's MD5 must identify the currently
        decoded RGB pixels.
        """

        planned = prepared.planned
        row = prepared.catalog
        if planned.domain != "anime" or row.get("domain") != "anime":
            raise OfflineLoaderError(
                f"source-bound native mismatch requires Anime row_id={planned.row_id}"
            )
        source_value = row.get("caption_parquet_path")
        if type(source_value) is not str:
            raise OfflineLoaderError(
                f"Anime source parquet path type drift row_id={planned.row_id}"
            )
        source_path = Path(source_value)
        if (
            source_path.parent != _ANIME_CAPTION_SOURCE_PARENT
            or source_path.suffix != ".parquet"
        ):
            raise OfflineLoaderError(
                f"Anime source parquet path drift row_id={planned.row_id}: "
                f"{source_path}"
            )
        for field in ("caption_row_group", "caption_row_in_group"):
            if type(row.get(field)) is not int or int(row[field]) < 0:
                raise OfflineLoaderError(
                    f"Anime source locator drift row_id={planned.row_id}: {field}"
                )
        if type(raw) is not bytes:
            raise OfflineLoaderError(
                f"Anime source-bound payload is not exact bytes row_id={planned.row_id}"
            )
        if (
            type(decoded_rgb) is not np.ndarray
            or decoded_rgb.dtype != np.uint8
            or decoded_rgb.ndim != 3
            or decoded_rgb.shape[2] != 3
            or not decoded_rgb.flags.c_contiguous
        ):
            raise OfflineLoaderError(
                f"Anime source-bound decoded RGB type drift row_id={planned.row_id}"
            )
        source_rows = self._read_source_rows(
            [(planned, row)],
            path_field="caption_parquet_path",
            row_group_field="caption_row_group",
            row_in_group_field="caption_row_in_group",
            columns=_ANIME_SOURCE_IMAGE_BINDING_COLUMNS,
            workers=1,
        )
        token = (planned.step, planned.slot)
        if set(source_rows) != {token}:
            raise OfflineLoaderError(
                f"Anime source row lookup drift row_id={planned.row_id}: "
                f"{sorted(source_rows)}"
            )
        source = source_rows[token]
        expected_source = {
            "key": row.get("key"),
            "s3key": row.get("fetch_url"),
            "width": row.get("width"),
            "height": row.get("height"),
            "image_size": len(raw),
        }
        actual_source: dict[str, Any] = {}
        for field, expected in expected_source.items():
            actual = source.get(field)
            if type(actual) is not type(expected):
                raise OfflineLoaderError(
                    f"Anime source field type drift row_id={planned.row_id}: {field}"
                )
            actual_source[field] = actual
        if actual_source != expected_source:
            raise OfflineLoaderError(
                f"Anime source metadata binding mismatch row_id={planned.row_id}"
            )
        expected_md5 = source.get("md5")
        if (
            type(expected_md5) is not str
            or re.fullmatch(r"[0-9a-f]{32}", expected_md5) is None
        ):
            raise OfflineLoaderError(
                f"Anime source decoded-RGB MD5 drift row_id={planned.row_id}"
            )
        actual_md5 = hashlib.md5(decoded_rgb.tobytes(order="C")).hexdigest()
        if actual_md5 != expected_md5:
            raise OfflineLoaderError(
                f"Anime source decoded-RGB binding mismatch row_id={planned.row_id}: "
                f"decoded_md5={actual_md5},source_md5={expected_md5}"
            )
        return actual_md5

    def _bucket_u8_for_item(self, prepared: _PreparedItem) -> np.ndarray:
        planned = prepared.planned
        target_h, target_w = self.buckets[planned.bucket_idx]
        raw = self._raw_for_item(prepared)
        expected_repaired_pixel_sha: str | None = None
        repair = self._image_repairs.get(planned.row_id)
        if repair is not None:
            try:
                raw, expected_repaired_pixel_sha = apply_textatlas_jpeg_eoi_repair(
                    repair,
                    domain=planned.domain,
                    row=prepared.catalog,
                    raw=raw,
                )
            except ImageExceptionMismatchError as exc:
                raise OfflineLoaderError(
                    f"frozen image exception mismatch row_id={planned.row_id}: {exc}"
                ) from exc
        # Keep the PIL image and its NumPy view inside one process-local native
        # critical section.  decode/resize also take this RLock, so their
        # established safety guards remain intact without concurrent gaps.
        with PIL_NATIVE_LOCK:
            image = decode_rgb(raw)
            if image is None:
                raise OfflineLoaderError(
                    f"decode failed step={planned.step},rank={self.rank},"
                    f"slot={planned.slot},row_id={planned.row_id}"
                )
            if expected_repaired_pixel_sha is not None:
                pixel_sha = hashlib.sha256(
                    np.asarray(image, dtype=np.uint8).tobytes()
                ).hexdigest()
                if pixel_sha != expected_repaired_pixel_sha:
                    raise OfflineLoaderError(
                        "frozen image exception pixel identity mismatch "
                        f"row_id={planned.row_id}: decoded={pixel_sha},"
                        f"expected={expected_repaired_pixel_sha}"
                    )
                warnings.warn(
                    "applying frozen TextAtlas JPEG EOI repair "
                    f"row_id={planned.row_id},pixel_sha256={pixel_sha}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            actual_w, actual_h = image.size
            expected_w = int(prepared.catalog["width"])
            expected_h = int(prepared.catalog["height"])
            native_upsample_exception = self._anime_native_upsample_exceptions.get(
                planned.row_id
            )
            if native_upsample_exception is not None:
                try:
                    bucket_u8 = apply_anime_native_upsample_exception(
                        native_upsample_exception,
                        step=planned.step,
                        rank=self.rank,
                        slot=planned.slot,
                        domain=planned.domain,
                        row=prepared.catalog,
                        raw=raw,
                        decoded_rgb=np.asarray(image, dtype=np.uint8),
                        target_height=target_h,
                        target_width=target_w,
                        crop_v_anchor=self._v_anchor_for(planned.domain),
                        resize_u8_to_bucket=resize_u8_to_bucket,
                    )
                except AnimeNativeUpsampleMismatchError as exc:
                    raise OfflineLoaderError(
                        "frozen anime native-upsample exception mismatch "
                        f"row_id={planned.row_id}: {exc}"
                    ) from exc
                print(
                    "[offline_dataset] WARNING applying frozen weighted-plan "
                    f"Anime native-upsample exception step={planned.step},"
                    f"rank={self.rank},slot={planned.slot},row_id={planned.row_id}",
                    flush=True,
                )
                return bucket_u8
            native_u8 = np.asarray(image, dtype=np.uint8)
            if (
                native_u8.ndim != 3
                or native_u8.shape[2] != 3
                or not native_u8.flags.c_contiguous
            ):
                raise OfflineLoaderError(
                    f"decoded RGB array has invalid shape/layout "
                    f"row_id={planned.row_id}: {native_u8.shape}"
                )
            # Set only by the anime tolerance below, and read only by the anchors_v1
            # gate further down.  The two checks are separated by the elif branch, so
            # the fact "this row's catalog geometry was overridden by its decoded
            # pixels" has to be carried across; recomputing it at the gate would
            # duplicate the tolerance condition and let the two drift apart.
            anime_native_drift = False
            if (actual_w, actual_h) != (expected_w, expected_h):
                # Anime HTTP-served payloads can have upstream width/height
                # metadata that disagrees with valid current pixels.  The
                # legacy streaming loader trusted decoded pixels.  Preserve the
                # frozen offline bucket only when doing so cannot upsample; all
                # immutable/local data domains retain the strict mapping guard.
                anime_no_upsample = (
                    planned.domain == "anime"
                    and actual_w >= target_w
                    and actual_h >= target_h
                )
                source_bound_md5: str | None = None
                if not anime_no_upsample and planned.domain == "anime":
                    source_bound_md5 = (
                        self._validate_anime_source_bound_native_mismatch(
                            prepared,
                            raw,
                            native_u8,
                        )
                    )
                elif not anime_no_upsample:
                    raise OfflineLoaderError(
                        f"native size mismatch step={planned.step},rank={self.rank},"
                        f"slot={planned.slot},row_id={planned.row_id}: "
                        f"decoded={actual_w}x{actual_h}, "
                        f"catalog={expected_w}x{expected_h}, "
                        f"target={target_w}x{target_h}"
                    )
                if source_bound_md5 is None:
                    detail = "without upsampling"
                else:
                    detail = f"with source-bound decoded_rgb_md5={source_bound_md5}"
                print(
                    "[offline_dataset] WARNING anime native-size metadata "
                    f"mismatch tolerated {detail} step={planned.step},"
                    f"rank={self.rank},slot={planned.slot},row_id={planned.row_id}: "
                    f"decoded={actual_w}x{actual_h}, "
                    f"catalog={expected_w}x{expected_h}, "
                    f"target={target_w}x{target_h}",
                    flush=True,
                )
                anime_native_drift = True
            elif actual_w < target_w or actual_h < target_h:
                # Defence in depth, and deliberately redundant: decoded ==
                # catalog here, and _prepare_window already refused any row
                # whose catalog size is below its bucket, so this cannot fire
                # unless that check was bypassed or self.buckets was rebound
                # between the two points.  It is cheap and it makes the
                # invariant local to the one place that actually resizes --
                # resize_u8_to_bucket's scale = max(W/w0, H/h0) enlarges without
                # ever refusing, so nothing downstream would catch it.
                #
                # Verified safe against the frozen Stage512a catalog: 0 of
                # 70,367,899 rows would trip it, max required scale 1.0000.
                raise OfflineLoaderError(
                    f"catalog geometry would upsample step={planned.step},"
                    f"rank={self.rank},slot={planned.slot},"
                    f"row_id={planned.row_id}: "
                    f"native={actual_w}x{actual_h}, target={target_w}x{target_h}"
                )
            if self._anchors_mode:
                # THE runtime geometry gate under anchors_v1, and the one
                # anchors_v1.verify_binding() documents itself as providing
                # ("零上采样保证只能落在这个校验上") while having had no production
                # caller at all.  Deliberately placed AFTER the branch above so it
                # sees the DECODED size in every case, including the anime
                # tolerated-mismatch path -- that path is exactly where a stale
                # manifest or an EXIF rotation shows up, and it is the case the
                # cheap catalog-side check structurally cannot see.
                #
                # Stronger than the size comparison above in two ways: it also
                # re-derives crop_loss against the per-domain threshold the catalog
                # was built with, and it uses the identical integer test as
                # assign_anchor (den*lo >= num*hi), so a row can never be accepted
                # here that offline assignment would have rejected.
                try:
                    anchors_v1.verify_binding(
                        actual_w,
                        actual_h,
                        int(planned.bucket_idx),
                        crop_loss_max=self._anchors_crop_loss[planned.domain],
                        context=(
                            f"step={planned.step},rank={self.rank},"
                            f"slot={planned.slot},row_id={planned.row_id},"
                            f"domain={planned.domain}"
                        ),
                    )
                except anchors_v1.AnchorCropLossError as exc:
                    # crop_loss超阈 + 这一行走过 anime metadata 漂移容忍 = 唯一放过的组合。
                    # 其他域、上采样、anchor 越界一律 fail-closed（下面那个 except）。
                    #
                    # 为什么放过：`crop_loss_max` 是**语料组成**约束，在离线 plan 建库时执行；
                    # 对 anime 漂移行，plan 依据的 catalog 尺寸和实际像素本来就允许不一致
                    # （见上面那条容忍分支的注释），所以这里的超阈是那条已批准的容忍在几何上
                    # 的必然投影，不是新的缺陷。切狠一点的个别行对组成无影响。
                    #
                    # 为什么原来会死：512a 一直容忍这种行（那时 verify_binding 没有生产调用者）；
                    # 512b 把它接上后，一行 1e-6 概率的行就能杀掉 64 卡 7 天的跑 —— 2026-08-04
                    # step 660 真实发生（row_id=3902543，catalog 1032x1457 / 解码 668x594）。
                    # 按实测频率，512M 曝光期望命中 ~380 次，等于这条路必然跑不完。
                    if not anime_native_drift:
                        raise OfflineLoaderError(f"anchors_v1 binding rejected: {exc}") from exc
                    self._anchors_drift_crop_tolerated += 1
                    if self._anchors_drift_crop_tolerated > _ANIME_DRIFT_CROP_TOLERATED_MAX:
                        raise OfflineLoaderError(
                            "anime native-drift crop_loss tolerance exhausted after "
                            f"{self._anchors_drift_crop_tolerated} rows on rank {self.rank} "
                            f"(cap {_ANIME_DRIFT_CROP_TOLERATED_MAX}); at the measured 7.4e-7 "
                            "rate this many rows means the row->file mapping is wrong, not "
                            f"that upstream metadata drifted: {exc}"
                        ) from exc
                    print(
                        "[offline_dataset] WARNING anchors_v1 crop_loss tolerated on an "
                        f"anime native-drift row ({self._anchors_drift_crop_tolerated}/"
                        f"{_ANIME_DRIFT_CROP_TOLERATED_MAX} on rank {self.rank}): {exc}",
                        flush=True,
                    )
                except anchors_v1.AnchorGeometryError as exc:
                    raise OfflineLoaderError(f"anchors_v1 binding rejected: {exc}") from exc
            # This remains the loader's only resize/crop operation.  Prepare-
            # ahead merely overlaps the same serialized image math with GPU
            # compute; it never permits concurrent native work within a rank.
            bucket_u8 = resize_u8_to_bucket(
                native_u8,
                target_h,
                target_w,
                self._v_anchor_for(planned.domain),
            )
            if bucket_u8.dtype != np.uint8 or bucket_u8.shape != (
                target_h,
                target_w,
                3,
            ):
                raise OfflineLoaderError(
                    f"target shape mismatch row_id={planned.row_id}: "
                    f"got={bucket_u8.dtype}/{bucket_u8.shape}, "
                    f"expected=uint8/{(target_h,target_w,3)}"
                )
            return bucket_u8

    def _consume_step(self, step: int, items: Sequence[_PreparedItem]) -> dict[str, Any]:
        bucket_idx = self._step_bucket[step]
        target_h, target_w = self.buckets[bucket_idx]
        prompts: list[str] = []
        tensors: list[torch.Tensor] = []
        cfg_keep: list[bool] = []
        trace_rows: list[int] = []
        trace_uids: list[str] = []
        trace_domains: list[str] = []
        trace_bands: list[int] = []
        successful_domains: Counter = Counter()

        for slot, prepared in enumerate(items):
            planned = prepared.planned
            if planned.step != step or planned.slot != slot or planned.bucket_idx != bucket_idx:
                raise OfflineLoaderError(
                    f"prepared item identity drift at step={step},slot={slot}: {planned}"
                )
            bucket_u8 = prepared.bucket_u8
            if bucket_u8 is None:
                raise OfflineLoaderError(
                    f"prepared bucket image missing step={step},rank={self.rank},"
                    f"slot={slot},row_id={planned.row_id}"
                )
            if bucket_u8.dtype != np.uint8 or bucket_u8.shape != (target_h, target_w, 3):
                raise OfflineLoaderError(
                    f"target shape mismatch row_id={planned.row_id}: "
                    f"got={bucket_u8.dtype}/{bucket_u8.shape}, expected=uint8/{(target_h,target_w,3)}"
                )

            rng = random.Random(
                self._slot_seed(self.plan_id, step, self.rank, slot)
            )
            band = self._validate_planned_band(
                planned.domain,
                prepared.caption_record,
                planned.caption_band,
            )
            prompt = self._finalize_caption(
                planned.domain, prepared.caption_record, band, rng
            )
            prompts.append(prompt)
            tensors.append(uint8_to_norm_tensor(bucket_u8))
            cfg_keep.append(
                self._is_cfg_protected(planned.domain, prepared.caption_record)
            )
            trace_rows.append(planned.row_id)
            trace_uids.append(str(prepared.catalog["uid"]))
            trace_domains.append(planned.domain)
            trace_bands.append(band)
            successful_domains[planned.domain] += 1

        pixels = torch.stack(tensors, dim=0)
        expected_shape = (len(items), 3, target_h, target_w)
        if tuple(pixels.shape) != expected_shape:
            raise OfflineLoaderError(
                f"batch tensor shape mismatch at step {step}: {tuple(pixels.shape)} != {expected_shape}"
            )
        self.domain_counter.update(successful_domains)
        self.last_item_trace = {
            "plan_step": step,
            "rank": self.rank,
            "bucket_idx": bucket_idx,
            "bucket_hw": [target_h, target_w],
            "item_rows": trace_rows,
            "uids": trace_uids,
            "domains": trace_domains,
            "caption_bands": trace_bands,
        }
        return {"prompts": prompts, "pixels": pixels, "cfg_keep": cfg_keep}

    def _retire_step(self, step: int, items: Sequence[_PreparedItem]) -> None:
        with self._condition:
            self._ready.pop(step, None)
            self._expected.pop(step, None)
            self._step_bucket.pop(step, None)
            for prepared in items:
                self._domain_ready[prepared.planned.domain] -= 1
            self._refresh_metrics_locked(plan_step=step)

    def _maintain_prefetch(self) -> None:
        total = int(self.plan_header["total_steps"])
        # Keep at least two configured prefetch depths queued.  Supply tasks
        # themselves are intentionally one step, spreading fetch/decode work
        # across iterations instead of launching a four-step CPU burst.
        minimum_lead = 2 * self.prefetch_steps
        while (
            self._scheduled_until < total
            and self._scheduled_until - self._next_step <= minimum_lead
        ):
            self._schedule_window(self._scheduled_until)

    def _refresh_metrics(self, plan_step: int | None = None) -> None:
        with self._condition:
            self._refresh_metrics_locked(plan_step=plan_step)

    def _refresh_metrics_locked(self, plan_step: int | None = None) -> None:
        now = time.monotonic()
        step_value = self._next_step if plan_step is None else int(plan_step)
        metrics: dict[str, float] = {
            "plan_step": float(step_value),
            "prefetch_depth": float(max(0, self._scheduled_until - self._next_step)),
        }
        for domain in self.domains:
            metrics[f"ready_{domain}"] = float(max(0, self._domain_ready[domain]))
            started = self._supply_started.get(domain)
            elapsed = now - started if started is not None else 0.0
            metrics[f"supply_img_s_{domain}"] = (
                float(self._supply_completed[domain]) / elapsed if elapsed > 0 else 0.0
            )
        if self._gpic_blob is not None:
            for key, value in self._gpic_blob.snapshot().items():
                metrics[f"gpic_blob_{key}"] = float(value)
            # Full-catalog mode has no legitimate source route. Keeping this
            # explicit numeric canary makes any future dual-route regression
            # visible in the standard dataline metrics contract.
            metrics["gpic_blob_source_fallback_items"] = 0.0
        self.buffer_fill = metrics

    def gpic_blob_health_metrics(self) -> dict[str, float]:
        """Return one rank's low-frequency GPIC health/throughput snapshot."""

        overlay = self._gpic_blob
        if self._gpic_payload_backend != "offline_gpic_blob_v1" or overlay is None:
            return {}
        with self._condition:
            supply_img_s = float(
                self.buffer_fill.get("supply_img_s_gpic", 0.0)
            )
        result = {
            f"gpic_blob_{key}": float(value)
            for key, value in overlay.health_snapshot().items()
        }
        result["gpic_blob_supply_img_s"] = supply_img_s
        result["gpic_blob_source_fallback_items"] = 0.0
        return result

    def loader(self, batch_size):
        """Yield exact plan batches in strict ``(step, rank, slot)`` order."""

        bs = _positive_int(batch_size, "batch_size")
        planned_bs = int(self.plan_header["per_rank_bs"])
        if bs != planned_bs:
            raise OfflineLoaderError(
                f"batch_size mismatch: loader={bs}, plan per_rank_bs={planned_bs}; rebuild the plan"
            )
        total = int(self.plan_header["total_steps"])
        try:
            while self._next_step < total:
                self._start_pipeline()  # lazy: train.py calls set_start_step before first next().
                step = self._next_step
                prepared = self._wait_for_step(step)
                batch = self._consume_step(step, prepared)
                self._retire_step(step, prepared)
                self._next_step = step + 1
                if self._next_step >= total:
                    # train.py stops immediately after the last yielded batch and
                    # does not close the generator, so close the terminal plan
                    # pipeline before yielding that batch.
                    self._stop_pipeline()
                else:
                    self._maintain_prefetch()
                self._refresh_metrics(plan_step=step)
                if self._telemetry_enabled:
                    self._emit_state_record(step)
                yield batch
        finally:
            self._stop_pipeline()
            self._close_gpic_blob(require_stopped=False)
            if self._telemetry_enabled and not self._telemetry_closed:
                try:
                    self._close_telemetry()
                except BaseException:
                    self._telemetry_runtime_error_count += 1

    def _stop_pipeline(self) -> None:
        if not self._pipeline_active:
            return
        self._stop_event.set()
        for work_queue in self._domain_queues.values():
            try:
                work_queue.put_nowait(None)
            except queue.Full:
                pass
        with self._condition:
            self._condition.notify_all()
        if self._telemetry_enabled and not self._telemetry_closed:
            try:
                self._close_telemetry()
            except BaseException:
                # Diagnostics must never turn a successful training path into a
                # supply failure. Missing/fatal closure evidence is rejected by
                # the r7 analyzer instead.
                self._telemetry_runtime_error_count += 1
                self._telemetry_accepting = False
        # A prefetched window can already be inside image reads/decodes when
        # shutdown is requested.  Those operations are not safely cancellable;
        # give them a bounded drain window before declaring the generation
        # poisoned and refusing to close the GPIC mmap/FD cache.
        shutdown_deadline = time.monotonic() + 30.0
        for thread in self._threads:
            remaining = shutdown_deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)
        live = [thread for thread in self._threads if thread.is_alive()]
        if live:
            # Never allow another generation to share catalog ParquetFile/cache
            # objects with a stuck old generation.
            self._pipeline_poisoned = True
        if not live:
            generation_dir = self._local_base / f"generation_{self._generation:04d}"
            shutil.rmtree(generation_dir, ignore_errors=True)
            try:
                self._local_base.rmdir()
            except OSError:
                pass
        self._ready.clear()
        self._expected.clear()
        self._step_bucket.clear()
        self._domain_ready.clear()
        self._threads = live
        self._domain_queues = {}
        self._pipeline_active = False

    def _close_gpic_blob(self, *, require_stopped: bool) -> None:
        overlay = self._gpic_blob
        if overlay is None:
            return
        if self._threads:
            if require_stopped:
                raise OfflineLoaderError(
                    "cannot close GPIC blob mmap/FDs while supply threads are alive: "
                    f"{[thread.name for thread in self._threads]}"
                )
            return
        try:
            overlay.close()
        except GpicBlobError as exc:
            raise OfflineLoaderError(f"cannot close GPIC blob overlay: {exc}") from exc

    def close(self) -> None:
        try:
            self._stop_pipeline()
            self._close_gpic_blob(require_stopped=True)
            if self._telemetry_enabled and not self._telemetry_closed:
                try:
                    self._close_telemetry()
                except BaseException:
                    self._telemetry_runtime_error_count += 1
        finally:
            if self._caption_sidecar is not None:
                self._caption_sidecar.close(wait=not bool(self._threads))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "OfflineExactDataset",
    "OfflineLoaderError",
    "decode_virtual_row_id",
]
