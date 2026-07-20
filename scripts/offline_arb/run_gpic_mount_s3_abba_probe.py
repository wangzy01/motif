#!/usr/bin/env python3
"""Capture a no-clobber GPIC mountpoint/S3 GetObject ABBA probe.

The hot path never logs a bucket, object key, URI, credential, VersionId, or
ETag.  The capture is CPU-only and uses eight spawned processes with two
threads each, matching the current two-GPIC-threads-per-rank 8-rank node.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import re
import secrets
import stat
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import boto3
from botocore.config import Config


RUN_ID = "offline_arb_gpic_mount_s3_abba_probe_20260712r1"
MAPPING_ID = "gpic_prod_v1"
SOURCE_CACHE = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_arb_phase2b_20260710T052630Z_"
    "evan-mot-crop-offlinearb-p2c-gpictune-gtnv-0_26335/gpic_items_cache.json"
)
SOURCE_CACHE_BYTES = 6_220_043
SOURCE_CACHE_SHA256 = "3c45ba6e3898b546dbc191e6b708e0bffc7047fd059708b7af8cc0e190cfa298"
SOURCE_CACHE_ITEMS = 20_000
SOURCE_ROOT = "/s3-mewtant-raw-datasets/stanford-vision-lab/gpic_umziped"
MOUNT_PREFIX = "/s3-mewtant-raw-datasets/"
S3_BUCKET = "mewtant-raw-datasets"
S3_KEY_PREFIX = "stanford-vision-lab/gpic_umziped/images/train/"
S3_KEY_RE = re.compile(
    r"stanford-vision-lab/gpic_umziped/images/train/[0-9]{5}/[0-9a-f]{64}\.(?:jpg|png)"
)
SOURCE_FIELDS = ("uid", "src_root", "src_shard", "src_key", "ext")
SOURCE_KEY_RE = re.compile(r"[0-9a-f]{64}")
SAFE_TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

SELECTION_SCHEMA = "offline_arb_gpic_s3_mount_abba_selection_v1"
SELECTION_DOMAIN = b"offline_arb_gpic_s3_mount_abba_selection_v1\0"
EXPECTED_SELECTION_SHA256 = "e358061beee8e23f0b154a93ecebd5b4023bb4e1c57492a2c92a007a82cf7a0d"
COHORT_SIZES = {"warmup": 64, "x": 2_048, "y": 2_048}
EXTENSION_ORDER = ("png", "jpg")
EXTENSION_COUNTS = {"jpg": 16_737, "png": 3_263}
SHARD_BANDS = 16
SHARDS_PER_BAND = 500

PROCESS_COUNT = 8
THREADS_PER_PROCESS = 2
EXPECTED_CONCURRENCY = 16
MAX_OBJECT_ATTEMPTS = 3
ARM_SPECS = (
    ("mount_x", "mount", "x", "00_mount_x"),
    ("s3_x", "s3", "x", "01_s3_x"),
    ("s3_y", "s3", "y", "02_s3_y"),
    ("mount_y", "mount", "y", "03_mount_y"),
)

ATTEMPT_SCHEMA = "offline_arb_gpic_mount_s3_abba_attempt_contract_v1"
PREFLIGHT_SCHEMA = "offline_arb_gpic_mount_s3_abba_preflight_v1"
ITEM_SCHEMA = "offline_arb_gpic_mount_s3_abba_item_v1"
ARM_CONTRACT_SCHEMA = "offline_arb_gpic_mount_s3_abba_arm_contract_v1"
ARM_RESULT_SCHEMA = "offline_arb_gpic_mount_s3_abba_arm_result_v1"
CAPTURE_MANIFEST_SCHEMA = "offline_arb_gpic_mount_s3_abba_capture_manifest_v1"
CAPTURE_COMPLETE_SCHEMA = "offline_arb_gpic_mount_s3_abba_capture_complete_v1"


class ContractError(ValueError):
    """A local input or capture contract is invalid."""


class SafeProbeError(RuntimeError):
    """A classified probe failure whose text contains no source detail."""

    def __init__(self, error_class: str) -> None:
        if error_class not in {
            "auth",
            "missing",
            "throttle",
            "transient",
            "short_read",
            "content_length_mismatch",
            "source_changed",
            "unexpected",
        }:
            error_class = "unexpected"
        self.error_class = error_class
        super().__init__(f"GPIC object read failed: {error_class}")


@dataclass(frozen=True)
class SourceItem:
    source_index: int
    canonical: bytes
    object_id: str
    uid: str
    src_root: str
    src_shard: str
    src_key: str
    ext: str
    ext_class: str
    shard_band: int


@dataclass(frozen=True)
class S3ReadResult:
    payload: bytes
    byte_count: int
    io_ns: int
    headers_ns: int
    read_ns: int
    version_id_present: bool
    if_match_used: bool
    object_attempts: int
    sdk_retry_attempts: int
    http_status: int
    content_length_match: bool
    response_header_count: int
    body_close_count: int
    first_response_attempt: int
    first_response_identity_kind: str
    preheader_retry_attempts: int
    version_id_retry_attempts: int
    if_match_retry_attempts: int
    unguarded_postheader_retry_attempts: int


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection_hash(*parts: bytes) -> str:
    return hashlib.sha256(SELECTION_DOMAIN + b"\0".join(parts)).hexdigest()


def _hamilton(total: int, weights: Mapping[Any, int]) -> dict[Any, int]:
    if type(total) is not int or total < 0 or not weights:
        raise ContractError("invalid Hamilton inputs")
    if any(type(value) is not int or value < 0 for value in weights.values()):
        raise ContractError("invalid Hamilton weights")
    denominator = sum(weights.values())
    if denominator <= 0:
        raise ContractError("zero Hamilton weight")
    result = {key: total * value // denominator for key, value in weights.items()}
    remainder = total - sum(result.values())
    order = sorted(
        weights,
        key=lambda key: (-(total * weights[key] % denominator), str(key)),
    )
    for key in order[:remainder]:
        result[key] += 1
    if sum(result.values()) != total:
        raise ContractError("Hamilton allocation did not close")
    return result


def validate_cache_item(item: Mapping[str, Any], source_index: int = 0) -> dict[str, str]:
    if type(source_index) is not int or source_index < 0:
        raise ContractError("source_index is invalid")
    if not isinstance(item, Mapping) or set(item) != set(SOURCE_FIELDS):
        raise ContractError("cache item exact key set mismatch")
    if any(type(item[field]) is not str for field in SOURCE_FIELDS):
        raise ContractError("cache item field type mismatch")
    value = {field: item[field] for field in SOURCE_FIELDS}
    key = value["src_key"]
    shard = value["src_shard"]
    if SOURCE_KEY_RE.fullmatch(key) is None:
        raise ContractError("cache item key shape mismatch")
    if value["uid"] != f"gpic:{key}":
        raise ContractError("cache item uid/key mismatch")
    if value["src_root"] != SOURCE_ROOT:
        raise ContractError("cache item root mismatch")
    if re.fullmatch(r"[0-9]{5}", shard) is None or int(shard) >= 8_000:
        raise ContractError("cache item shard mismatch")
    if value["ext"] not in {".jpg", ".png"}:
        raise ContractError("cache item extension mismatch")
    if any(token in key or token in shard for token in ("/", "\\", "\x00")):
        raise ContractError("cache item path component mismatch")
    return value


def _source_item(item: Mapping[str, Any], source_index: int) -> SourceItem:
    value = validate_cache_item(item, source_index=source_index)
    canonical = canonical_json_bytes({field: value[field] for field in SOURCE_FIELDS})
    return SourceItem(
        source_index=source_index,
        canonical=canonical,
        object_id=_selection_hash(b"item", canonical),
        uid=value["uid"],
        src_root=value["src_root"],
        src_shard=value["src_shard"],
        src_key=value["src_key"],
        ext=value["ext"],
        ext_class=value["ext"][1:],
        shard_band=int(value["src_shard"]) // SHARDS_PER_BAND,
    )


def cache_item_location(item: Mapping[str, Any] | SourceItem) -> dict[str, str]:
    if isinstance(item, SourceItem):
        value = {field: getattr(item, field) for field in SOURCE_FIELDS}
    else:
        value = validate_cache_item(item)
    filename = f"{value['src_key']}{value['ext']}"
    key = f"{S3_KEY_PREFIX}{value['src_shard']}/{filename}"
    if not key.startswith(S3_KEY_PREFIX) or ".." in key or "//" in key:
        raise ContractError("derived S3 key escaped the frozen prefix")
    mount_path = f"{MOUNT_PREFIX}{key}"
    if not mount_path.startswith(MOUNT_PREFIX) or ".." in mount_path:
        raise ContractError("derived mount path escaped the frozen prefix")
    return {"mount_path": mount_path, "bucket": S3_BUCKET, "key": key}


def load_cache(path: Path, expected_sha256: str = SOURCE_CACHE_SHA256) -> list[SourceItem]:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != SOURCE_CACHE_BYTES:
        raise ContractError("source cache file identity mismatch")
    if expected_sha256 != SOURCE_CACHE_SHA256 or sha256_file(path) != expected_sha256:
        raise ContractError("source cache SHA256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("source cache is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {"items", "resolve_errors"}:
        raise ContractError("source cache schema mismatch")
    if type(value["resolve_errors"]) is not int or value["resolve_errors"] != 0:
        raise ContractError("source cache resolve_errors mismatch")
    if not isinstance(value["items"], list) or len(value["items"]) != SOURCE_CACHE_ITEMS:
        raise ContractError("source cache item count mismatch")
    items = [_source_item(raw, index) for index, raw in enumerate(value["items"])]
    if len({item.uid for item in items}) != SOURCE_CACHE_ITEMS:
        raise ContractError("source cache uid uniqueness mismatch")
    if len({item.src_key for item in items}) != SOURCE_CACHE_ITEMS:
        raise ContractError("source cache key uniqueness mismatch")
    if Counter(item.ext_class for item in items) != Counter(EXTENSION_COUNTS):
        raise ContractError("source cache extension distribution mismatch")
    return items


def build_selection(items: Sequence[SourceItem]) -> dict[str, Any]:
    if len(items) != SOURCE_CACHE_ITEMS:
        raise ContractError("selection input count mismatch")
    ext_weights = Counter(item.ext_class for item in items)
    band_weights = {
        ext: {
            band: len(
                {
                    item.src_shard
                    for item in items
                    if item.ext_class == ext and item.shard_band == band
                }
            )
            for band in range(SHARD_BANDS)
        }
        for ext in EXTENSION_ORDER
    }
    used_shards: set[str] = set()
    cohorts: dict[str, list[dict[str, Any]]] = {}
    for cohort, total in COHORT_SIZES.items():
        ext_quota = _hamilton(total, ext_weights)
        chosen: list[tuple[SourceItem, str]] = []
        for ext in EXTENSION_ORDER:
            band_quota = _hamilton(ext_quota[ext], band_weights[ext])
            for band in range(SHARD_BANDS):
                best_by_shard: dict[str, tuple[str, str, SourceItem]] = {}
                for item in items:
                    if (
                        item.ext_class != ext
                        or item.shard_band != band
                        or item.src_shard in used_shards
                    ):
                        continue
                    score = _selection_hash(
                        b"member", cohort.encode("ascii"), item.canonical
                    )
                    candidate = (score, item.object_id, item)
                    previous = best_by_shard.get(item.src_shard)
                    if previous is None or candidate[:2] < previous[:2]:
                        best_by_shard[item.src_shard] = candidate
                candidates = sorted(best_by_shard.values(), key=lambda value: value[:2])
                quota = band_quota[band]
                if len(candidates) < quota:
                    raise ContractError("selection stratum lacks distinct shards")
                for _score, _object_id, item in candidates[:quota]:
                    used_shards.add(item.src_shard)
                    order_score = _selection_hash(
                        b"order", cohort.encode("ascii"), item.object_id.encode("ascii")
                    )
                    chosen.append((item, order_score))
        chosen.sort(key=lambda value: (value[1], value[0].object_id))
        if len(chosen) != total:
            raise ContractError("selection cohort did not close")
        cohorts[cohort] = [
            {
                "source_index": item.source_index,
                "object_id": item.object_id,
                "ext_class": item.ext_class,
                "shard_band": item.shard_band,
            }
            for item, _order_score in chosen
        ]
    if len(used_shards) != sum(COHORT_SIZES.values()):
        raise ContractError("selection physical shard closure mismatch")
    core = {
        "schema_version": SELECTION_SCHEMA,
        "source_cache_sha256": SOURCE_CACHE_SHA256,
        "cohorts": cohorts,
    }
    digest = canonical_sha256(core)
    if digest != EXPECTED_SELECTION_SHA256:
        raise ContractError("selection digest mismatch")
    return {**core, "selection_sha256": digest}


def classify_s3_error(exc: BaseException) -> str:
    if type(exc).__name__ in {
        "NoCredentialsError",
        "PartialCredentialsError",
        "CredentialRetrievalError",
        "TokenRetrievalError",
    }:
        return "auth"
    if type(exc).__name__ in {
        "ConnectTimeoutError",
        "ReadTimeoutError",
        "EndpointConnectionError",
        "ConnectionClosedError",
        "HTTPClientError",
        "ProxyConnectionError",
    }:
        return "transient"
    response = getattr(exc, "response", None)
    error: Mapping[str, Any] = {}
    metadata: Mapping[str, Any] = {}
    if isinstance(response, Mapping):
        raw_error = response.get("Error", {})
        raw_metadata = response.get("ResponseMetadata", {})
        if isinstance(raw_error, Mapping):
            error = raw_error
        if isinstance(raw_metadata, Mapping):
            metadata = raw_metadata
    code = str(error.get("Code", ""))
    raw_status = metadata.get("HTTPStatusCode", 0)
    status = raw_status if type(raw_status) is int else 0
    if status == 412 or code in {"PreconditionFailed", "InvalidObjectState"}:
        return "source_changed"
    if status in {401, 403} or code in {
        "AccessDenied",
        "InvalidAccessKeyId",
        "ExpiredToken",
        "SignatureDoesNotMatch",
        "KMSAccessDeniedException",
    }:
        return "auth"
    if status == 404 or code in {"NoSuchKey", "NotFound", "NoSuchBucket"}:
        return "missing"
    if status == 429 or code in {"SlowDown", "Throttling", "TooManyRequests"}:
        return "throttle"
    if status >= 500 or code in {
        "InternalError",
        "ServiceUnavailable",
        "RequestTimeout",
    }:
        return "transient"
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return "transient"
    return "unexpected"


def sdk_retry_attempts(response: Mapping[str, Any]) -> int:
    if not isinstance(response, Mapping):
        raise ContractError("S3 response is not a mapping")
    metadata = response.get("ResponseMetadata", {})
    if not isinstance(metadata, Mapping):
        raise ContractError("S3 ResponseMetadata is malformed")
    value = metadata.get("RetryAttempts", 0)
    if type(value) is not int or value < 0:
        raise ContractError("S3 RetryAttempts is malformed")
    return value


def build_botocore_config(
    *, connect_timeout_s: float = 5.0, read_timeout_s: float = 30.0
) -> Config:
    if (
        isinstance(connect_timeout_s, bool)
        or not isinstance(connect_timeout_s, (int, float))
        or not math.isfinite(float(connect_timeout_s))
        or float(connect_timeout_s) <= 0
    ):
        raise ContractError("connect timeout is invalid")
    if (
        isinstance(read_timeout_s, bool)
        or not isinstance(read_timeout_s, (int, float))
        or not math.isfinite(float(read_timeout_s))
        or float(read_timeout_s) <= 0
    ):
        raise ContractError("read timeout is invalid")
    return Config(
        region_name="us-west-2",
        max_pool_connections=64,
        retries={"mode": "standard", "total_max_attempts": 3},
        tcp_keepalive=True,
        connect_timeout=float(connect_timeout_s),
        read_timeout=float(read_timeout_s),
    )


_TEMPORARY_METHODS = {
    "assume-role",
    "assume-role-with-web-identity",
    "container-role",
    "iam-role",
    "sso",
    "custom-process",
}
_STATIC_METHODS = {"env", "shared-credentials-file", "config-file"}


def credential_summary(session: Any) -> dict[str, Any]:
    """Return allowlisted provider shape without touching credential values."""
    try:
        credentials = session.get_credentials()
    except BaseException:
        credentials = None
    if credentials is None:
        return {
            "method": "unavailable",
            "credentials_class": "unavailable",
            "refreshable": False,
            "has_session_token": False,
        }
    raw_method = getattr(credentials, "method", "unknown")
    method = raw_method if type(raw_method) is str else "unavailable"
    if method == "credential_process":
        method = "custom-process"
    if method not in _TEMPORARY_METHODS | _STATIC_METHODS:
        method = "unavailable"
    if method == "unavailable":
        return {
            "method": "unavailable",
            "credentials_class": "unavailable",
            "refreshable": False,
            "has_session_token": False,
        }
    instance_values = vars(credentials) if hasattr(credentials, "__dict__") else {}
    class_name = type(credentials).__name__.lower()
    refresh_hook = instance_values.get("_refresh_using")
    refreshable = callable(refresh_hook) or "refreshablecredentials" in class_name
    if refreshable:
        credential_class = "refreshable_temporary"
    elif method in _TEMPORARY_METHODS | _STATIC_METHODS:
        credential_class = "nonrefreshable_temporary"
    else:
        credential_class = "unavailable"
    has_session_token = credential_class in {
        "refreshable_temporary",
        "nonrefreshable_temporary",
    }
    return {
        "method": method,
        "credentials_class": credential_class,
        "refreshable": refreshable,
        "has_session_token": has_session_token,
    }


def validate_concurrency_contract(
    *, processes: int, threads_per_process: int, observed_peak: int
) -> None:
    if (
        type(processes) is not int
        or type(threads_per_process) is not int
        or type(observed_peak) is not int
        or processes != PROCESS_COUNT
        or threads_per_process != THREADS_PER_PROCESS
        or observed_peak != EXPECTED_CONCURRENCY
    ):
        raise ContractError("8x2/peak16 concurrency contract mismatch")


def _safe_response_status(response: Mapping[str, Any]) -> int:
    metadata = response.get("ResponseMetadata", {})
    if not isinstance(metadata, Mapping):
        raise SafeProbeError("unexpected")
    status = metadata.get("HTTPStatusCode", 0)
    if type(status) is not int or status < 100 or status > 599:
        raise SafeProbeError("unexpected")
    return status


def s3_get_object_bytes(
    client: Any,
    *,
    bucket: str,
    key: str,
    chunk_bytes: int = 8_388_608,
    max_attempts: int = MAX_OBJECT_ATTEMPTS,
) -> S3ReadResult:
    """Read one whole object, pinning every post-header retry to its identity."""
    if bucket != S3_BUCKET or type(key) is not str or S3_KEY_RE.fullmatch(key) is None:
        raise ContractError("S3 location is outside the frozen mapping")
    if ".." in key or "//" in key or "\x00" in key:
        raise ContractError("S3 key is unsafe")
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ContractError("chunk_bytes is invalid")
    if type(max_attempts) is not int or max_attempts < 1 or max_attempts > MAX_OBJECT_ATTEMPTS:
        raise ContractError("max_attempts is invalid")

    started_ns = time.perf_counter_ns()
    total_headers_ns = 0
    total_read_ns = 0
    total_sdk_retries = 0
    response_header_count = 0
    body_close_count = 0
    first_response_attempt = 0
    first_response_identity_kind = "none"
    preheader_retries = 0
    version_retries = 0
    if_match_retries = 0
    pinned_version: str | None = None
    pinned_etag: str | None = None
    last_status = 0
    final_class = "unexpected"

    for attempt_index in range(max_attempts):
        request: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if pinned_version is not None:
            request["VersionId"] = pinned_version
            if attempt_index:
                version_retries += 1
        elif pinned_etag is not None:
            request["IfMatch"] = pinned_etag
            if attempt_index:
                if_match_retries += 1
        elif attempt_index:
            preheader_retries += 1

        headers_started_ns = time.perf_counter_ns()
        try:
            response = client.get_object(**request)
        except BaseException as exc:
            total_headers_ns += time.perf_counter_ns() - headers_started_ns
            error_response = getattr(exc, "response", None)
            if isinstance(error_response, Mapping):
                try:
                    total_sdk_retries += sdk_retry_attempts(error_response)
                except ContractError:
                    raise SafeProbeError("unexpected") from None
            error_class = classify_s3_error(exc)
            final_class = error_class
            if error_class in {"transient", "throttle"} and attempt_index + 1 < max_attempts:
                continue
            raise SafeProbeError(error_class) from None
        total_headers_ns += time.perf_counter_ns() - headers_started_ns
        if not isinstance(response, Mapping):
            raise SafeProbeError("unexpected")
        response_header_count += 1
        if first_response_attempt == 0:
            first_response_attempt = attempt_index + 1
        total_sdk_retries += sdk_retry_attempts(response)
        last_status = _safe_response_status(response)
        body = response.get("Body")
        if body is None or not callable(getattr(body, "read", None)):
            raise SafeProbeError("unexpected")
        if last_status != 200:
            try:
                body.close()
                body_close_count += 1
            except BaseException:
                pass
            raise SafeProbeError("unexpected")

        raw_version = response.get("VersionId")
        raw_etag = response.get("ETag")
        version = raw_version if type(raw_version) is str and raw_version else None
        etag = raw_etag if type(raw_etag) is str and raw_etag else None
        if response_header_count == 1:
            first_response_identity_kind = (
                "version_pin" if version is not None else "if_match_pin" if etag is not None else "none"
            )
        content_length = response.get("ContentLength")
        read_started_ns = time.perf_counter_ns()
        payload: bytes | None = None
        read_error: str | None = None
        try:
            if pinned_version is not None and version != pinned_version:
                raise SafeProbeError("source_changed")
            if pinned_etag is not None and etag != pinned_etag:
                raise SafeProbeError("source_changed")
            if pinned_version is None and pinned_etag is None:
                if version is not None:
                    pinned_version = version
                elif etag is not None:
                    pinned_etag = etag
            if type(content_length) is not int or content_length <= 0:
                raise SafeProbeError("content_length_mismatch")
            chunks: list[bytes] = []
            total = 0
            while True:
                block = body.read(chunk_bytes)
                if not isinstance(block, bytes):
                    raise SafeProbeError("unexpected")
                if not block:
                    break
                chunks.append(block)
                total += len(block)
                if total > content_length:
                    raise SafeProbeError("content_length_mismatch")
            payload = b"".join(chunks)
            if len(payload) != content_length:
                read_error = (
                    "short_read" if len(payload) < content_length else "content_length_mismatch"
                )
        except SafeProbeError:
            raise
        except BaseException:
            read_error = "transient"
        finally:
            total_read_ns += time.perf_counter_ns() - read_started_ns
            try:
                body.close()
                body_close_count += 1
            except BaseException:
                raise SafeProbeError("unexpected") from None

        if read_error is None and payload is not None:
            return S3ReadResult(
                payload=payload,
                byte_count=len(payload),
                io_ns=time.perf_counter_ns() - started_ns,
                headers_ns=total_headers_ns,
                read_ns=total_read_ns,
                version_id_present=pinned_version is not None,
                if_match_used=if_match_retries > 0,
                object_attempts=attempt_index + 1,
                sdk_retry_attempts=total_sdk_retries,
                http_status=last_status,
                content_length_match=True,
                response_header_count=response_header_count,
                body_close_count=body_close_count,
                first_response_attempt=first_response_attempt,
                first_response_identity_kind=first_response_identity_kind,
                preheader_retry_attempts=preheader_retries,
                version_id_retry_attempts=version_retries,
                if_match_retry_attempts=if_match_retries,
                unguarded_postheader_retry_attempts=0,
            )

        final_class = read_error or "unexpected"
        if pinned_version is None and pinned_etag is None:
            raise SafeProbeError(final_class)
        if attempt_index + 1 >= max_attempts:
            raise SafeProbeError(final_class)

    raise SafeProbeError(final_class)


def _require_real_hierarchy(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise ContractError(f"{label} hierarchy is missing") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ContractError(f"{label} hierarchy is not a real directory")


def _mkdir_real(path: Path, *, mode: int = 0o700) -> None:
    path = path.absolute()
    parent = path.parent
    _require_real_hierarchy(parent, "directory parent")
    try:
        path.mkdir(mode=mode, exist_ok=False)
    except FileExistsError as exc:
        raise ContractError("refusing an existing attempt directory") from exc
    _require_real_hierarchy(path, "created directory")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_publish_bytes(path: Path, raw: bytes) -> None:
    path = Path(path).absolute()
    if not isinstance(raw, bytes):
        raise ContractError("publication payload must be bytes")
    _require_real_hierarchy(path.parent, "publication parent")
    if path.exists() or path.is_symlink():
        raise ContractError("refusing to replace a published artifact")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(16)}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", buffering=0, closefd=True) as handle:
            view = memoryview(raw)
            while view:
                written = handle.write(view)
                if not written:
                    raise OSError("short publication write")
                view = view[written:]
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444, follow_symlinks=False)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ContractError("refusing to replace a published artifact") from exc
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def atomic_publish_json(path: Path, value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise ContractError("JSON publication must be a mapping")
    atomic_publish_bytes(Path(path), canonical_json_bytes(dict(value)) + b"\n")


def _file_attestation(path: Path) -> dict[str, Any]:
    supplied = Path(path).absolute()
    try:
        canonical = supplied.resolve(strict=True)
    except OSError as exc:
        raise ContractError("attested input cannot be resolved") from exc
    if supplied != canonical or supplied.is_symlink() or not supplied.is_file():
        raise ContractError("attested input is not a canonical regular file")
    metadata = supplied.stat()
    if metadata.st_size <= 0:
        raise ContractError("attested input is empty")
    return {
        "path": str(supplied),
        "bytes": metadata.st_size,
        "sha256": sha256_file(supplied),
    }


def build_input_attestation(cache_path: Path) -> dict[str, dict[str, Any]]:
    repo = Path(__file__).resolve(strict=True).parents[2]
    paths = {
        "runner": Path(__file__).resolve(strict=True),
        "analyzer": repo / "scripts/offline_arb/analyze_gpic_mount_s3_abba_probe.py",
        "capture_yaml": repo / "jobs/offline_arb_gpic_mount_s3_abba_probe.yaml",
        "source_cache": Path(cache_path).absolute(),
    }
    return {name: _file_attestation(path) for name, path in paths.items()}


def verify_input_attestation(attestation: Mapping[str, Mapping[str, Any]]) -> None:
    expected_keys = {"runner", "analyzer", "capture_yaml", "source_cache"}
    if not isinstance(attestation, Mapping) or set(attestation) != expected_keys:
        raise ContractError("input attestation key set mismatch")
    for name in sorted(expected_keys):
        expected = attestation[name]
        if not isinstance(expected, Mapping) or set(expected) != {"path", "bytes", "sha256"}:
            raise ContractError("input attestation entry schema mismatch")
        if type(expected["path"]) is not str:
            raise ContractError("input attestation path is malformed")
        observed = _file_attestation(Path(expected["path"]))
        if observed != dict(expected):
            raise ContractError("live input attestation changed")


def _artifact_attestation(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError("artifact is not a regular file")
    metadata = path.stat()
    return {"bytes": metadata.st_size, "sha256": sha256_file(path)}


@dataclass(frozen=True)
class MountReadResult:
    payload: bytes
    byte_count: int
    io_ns: int
    read_ns: int


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_mode,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def mount_read_bytes(path: Path, *, chunk_bytes: int = 8_388_608) -> MountReadResult:
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ContractError("mount chunk size is invalid")
    raw_path = str(path)
    if not raw_path.startswith(MOUNT_PREFIX) or ".." in raw_path or "\x00" in raw_path:
        raise ContractError("mount path escaped the frozen prefix")
    started_ns = time.perf_counter_ns()
    # Match the production strict_path reader: ordinary follow-symlink open,
    # two fstats, and two follow-symlink path stats around the full read.
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(raw_path, flags)
    except OSError:
        raise SafeProbeError("missing") from None
    try:
        before_fd = os.fstat(descriptor)
        before_path = os.stat(raw_path, follow_symlinks=True)
        if (
            not stat.S_ISREG(before_fd.st_mode)
            or before_fd.st_size <= 0
            or _stat_identity(before_fd) != _stat_identity(before_path)
        ):
            raise SafeProbeError("source_changed")
        chunks: list[bytes] = []
        total = 0
        read_started_ns = time.perf_counter_ns()
        while True:
            block = os.read(descriptor, chunk_bytes)
            if not block:
                break
            chunks.append(block)
            total += len(block)
        read_ns = time.perf_counter_ns() - read_started_ns
        after_fd = os.fstat(descriptor)
        after_path = os.stat(raw_path, follow_symlinks=True)
        if (
            _stat_identity(before_fd) != _stat_identity(after_fd)
            or _stat_identity(before_path) != _stat_identity(after_path)
            or _stat_identity(after_fd) != _stat_identity(after_path)
            or total != before_fd.st_size
        ):
            raise SafeProbeError("source_changed")
        payload = b"".join(chunks)
        return MountReadResult(
            payload=payload,
            byte_count=len(payload),
            io_ns=time.perf_counter_ns() - started_ns,
            read_ns=read_ns,
        )
    except SafeProbeError:
        raise
    except OSError:
        raise SafeProbeError("transient") from None
    finally:
        os.close(descriptor)


def _item_record(
    *,
    arm: str,
    backend: str,
    cohort: str,
    worker_id: int,
    position: int,
    item: SourceItem,
    client: Any,
    chunk_bytes: int,
    max_attempts: int,
) -> dict[str, Any]:
    location = cache_item_location(item)
    if backend == "mount":
        result: Any = mount_read_bytes(Path(location["mount_path"]), chunk_bytes=chunk_bytes)
        transport = {
            "headers_ns": 0,
            "version_id_present": False,
            "response_header_count": 0,
            "body_close_count": 0,
            "first_response_attempt": 0,
            "first_response_identity_kind": "none",
            "preheader_retry_attempts": 0,
            "version_id_retry_attempts": 0,
            "if_match_retry_attempts": 0,
            "unguarded_postheader_retry_attempts": 0,
            "object_attempts": 1,
            "sdk_retry_attempts": 0,
            "http_status": 0,
            "content_length_match": True,
        }
    elif backend == "s3":
        result = s3_get_object_bytes(
            client,
            bucket=location["bucket"],
            key=location["key"],
            chunk_bytes=chunk_bytes,
            max_attempts=max_attempts,
        )
        transport = {
            "headers_ns": result.headers_ns,
            "version_id_present": result.version_id_present,
            "response_header_count": result.response_header_count,
            "body_close_count": result.body_close_count,
            "first_response_attempt": result.first_response_attempt,
            "first_response_identity_kind": result.first_response_identity_kind,
            "preheader_retry_attempts": result.preheader_retry_attempts,
            "version_id_retry_attempts": result.version_id_retry_attempts,
            "if_match_retry_attempts": result.if_match_retry_attempts,
            "unguarded_postheader_retry_attempts": result.unguarded_postheader_retry_attempts,
            "object_attempts": result.object_attempts,
            "sdk_retry_attempts": result.sdk_retry_attempts,
            "http_status": result.http_status,
            "content_length_match": result.content_length_match,
        }
    else:
        raise ContractError("unknown transport backend")
    hash_started_ns = time.perf_counter_ns()
    payload_sha = sha256_bytes(result.payload)
    hash_ns = time.perf_counter_ns() - hash_started_ns
    return {
        "schema_version": ITEM_SCHEMA,
        "arm": arm,
        "backend": backend,
        "cohort": cohort,
        "worker_id": worker_id,
        "position": position,
        "source_index": item.source_index,
        "object_id": item.object_id,
        "outcome": "success",
        "byte_count": result.byte_count,
        "payload_sha256": payload_sha,
        "io_ns": result.io_ns,
        "headers_ns": transport["headers_ns"],
        "read_ns": result.read_ns,
        "hash_ns": hash_ns,
        **transport,
        "error_class": "none",
    }


def _worker_process(
    worker_id: int,
    backend: str,
    arm: str,
    cohort: str,
    assignments: list[tuple[int, SourceItem]],
    warmup_assignments: list[tuple[int, SourceItem]],
    chunk_bytes: int,
    max_attempts: int,
    start_barrier: Any,
    peak_barrier: Any,
    inflight: Any,
    peak: Any,
    panel_start_ns: Any,
    panel_end_ns: Any,
    counter_lock: Any,
    result_queue: Any,
) -> None:
    client = None
    try:
        if backend == "s3":
            session = boto3.Session(region_name="us-west-2")
            client = session.client("s3", config=build_botocore_config())

        def execute(pair: tuple[int, SourceItem], initial: bool) -> dict[str, Any]:
            if initial:
                start_barrier.wait(timeout=120)
            with counter_lock:
                inflight.value += 1
                if inflight.value > peak.value:
                    peak.value = inflight.value
            try:
                if initial:
                    peak_barrier.wait(timeout=120)
                now_ns = time.perf_counter_ns()
                with counter_lock:
                    if panel_start_ns.value == 0 or now_ns < panel_start_ns.value:
                        panel_start_ns.value = now_ns
                record = _item_record(
                    arm=arm,
                    backend=backend,
                    cohort=cohort,
                    worker_id=worker_id,
                    position=pair[0],
                    item=pair[1],
                    client=client,
                    chunk_bytes=chunk_bytes,
                    max_attempts=max_attempts,
                )
                finished_ns = time.perf_counter_ns()
                with counter_lock:
                    if finished_ns > panel_end_ns.value:
                        panel_end_ns.value = finished_ns
                return record
            finally:
                with counter_lock:
                    inflight.value -= 1

        records: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=THREADS_PER_PROCESS,
            thread_name_prefix=f"gpic-{backend}-{worker_id}",
        ) as executor:
            warmup_futures = [
                executor.submit(
                    _item_record,
                    arm=f"warmup_{backend}",
                    backend=backend,
                    cohort="warmup",
                    worker_id=worker_id,
                    position=pair[0],
                    item=pair[1],
                    client=client,
                    chunk_bytes=chunk_bytes,
                    max_attempts=max_attempts,
                )
                for pair in warmup_assignments
            ]
            for future in concurrent.futures.as_completed(warmup_futures):
                future.result()
            futures = [
                executor.submit(execute, pair, index < THREADS_PER_PROCESS)
                for index, pair in enumerate(assignments)
            ]
            for future in concurrent.futures.as_completed(futures):
                records.append(future.result())
        records.sort(key=lambda value: value["position"])
        result_queue.put({"ok": True, "worker_id": worker_id, "records": records})
    except BaseException as exc:
        error_class = getattr(exc, "error_class", "unexpected")
        if error_class not in {
            "auth",
            "missing",
            "throttle",
            "transient",
            "short_read",
            "content_length_mismatch",
            "source_changed",
            "unexpected",
        }:
            error_class = "unexpected"
        result_queue.put({"ok": False, "worker_id": worker_id, "error_class": error_class})
    finally:
        if client is not None and callable(getattr(client, "close", None)):
            try:
                client.close()
            except BaseException:
                pass


def _run_process_panel(
    *,
    items: Sequence[SourceItem],
    selected: Sequence[Mapping[str, Any]],
    arm: str,
    backend: str,
    cohort: str,
    warmup_selected: Sequence[Mapping[str, Any]] | None = None,
    chunk_bytes: int,
    max_attempts: int,
    timeout_s: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    if len(selected) < EXPECTED_CONCURRENCY:
        raise ContractError("panel is too small to prove concurrency")
    context = multiprocessing.get_context("spawn")
    start_barrier = context.Barrier(EXPECTED_CONCURRENCY)
    peak_barrier = context.Barrier(EXPECTED_CONCURRENCY)
    inflight = context.Value("i", 0)
    peak = context.Value("i", 0)
    panel_start_ns = context.Value("q", 0)
    panel_end_ns = context.Value("q", 0)
    counter_lock = context.Lock()
    result_queue = context.Queue()
    assignments: list[list[tuple[int, SourceItem]]] = [[] for _ in range(PROCESS_COUNT)]
    for position, row in enumerate(selected):
        source_index = row["source_index"]
        item = items[source_index]
        if item.object_id != row["object_id"]:
            raise ContractError("selection/source object identity mismatch")
        assignments[position % PROCESS_COUNT].append((position, item))
    warmup_assignments: list[list[tuple[int, SourceItem]]] = [
        [] for _ in range(PROCESS_COUNT)
    ]
    if warmup_selected is not None:
        for position, row in enumerate(warmup_selected):
            item = items[row["source_index"]]
            if item.object_id != row["object_id"]:
                raise ContractError("warmup/source object identity mismatch")
            warmup_assignments[position % PROCESS_COUNT].append((position, item))
    processes = [
        context.Process(
            target=_worker_process,
            args=(
                worker_id,
                backend,
                arm,
                cohort,
                assignments[worker_id],
                warmup_assignments[worker_id],
                chunk_bytes,
                max_attempts,
                start_barrier,
                peak_barrier,
                inflight,
                peak,
                panel_start_ns,
                panel_end_ns,
                counter_lock,
                result_queue,
            ),
            name=f"gpic-{arm}-{worker_id}",
        )
        for worker_id in range(PROCESS_COUNT)
    ]
    started_ns = time.perf_counter_ns()
    for process in processes:
        process.start()
    results: dict[int, list[dict[str, Any]]] = {}
    failure_class: str | None = None
    deadline = time.monotonic() + timeout_s
    for _ in processes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failure_class = "transient"
            break
        try:
            value = result_queue.get(timeout=remaining)
        except BaseException:
            failure_class = "transient"
            break
        if not isinstance(value, dict) or value.get("ok") is not True:
            failure_class = (
                value.get("error_class", "unexpected")
                if isinstance(value, dict)
                else "unexpected"
            )
            break
        results[value["worker_id"]] = value["records"]
    if failure_class is not None:
        for process in processes:
            if process.is_alive():
                process.terminate()
    for process in processes:
        process.join(timeout=30)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
    if failure_class is not None or any(process.exitcode != 0 for process in processes):
        raise SafeProbeError(failure_class or "unexpected")
    if set(results) != set(range(PROCESS_COUNT)):
        raise ContractError("worker result closure mismatch")
    records = [record for worker_id in range(PROCESS_COUNT) for record in results[worker_id]]
    records.sort(key=lambda value: value["position"])
    if [record["position"] for record in records] != list(range(len(selected))):
        raise ContractError("panel position closure mismatch")
    observed_peak = peak.value
    validate_concurrency_contract(
        processes=PROCESS_COUNT,
        threads_per_process=THREADS_PER_PROCESS,
        observed_peak=observed_peak,
    )
    lifecycle_elapsed_ns = time.perf_counter_ns() - started_ns
    if panel_start_ns.value <= 0 or panel_end_ns.value <= panel_start_ns.value:
        raise ContractError("shared timed panel window did not close")
    elapsed_ns = panel_end_ns.value - panel_start_ns.value
    if max(record["io_ns"] for record in records) > elapsed_ns:
        raise ContractError("item I/O interval exceeds timed panel window")
    if elapsed_ns > lifecycle_elapsed_ns:
        raise ContractError("timed panel window exceeds process lifecycle")
    return records, observed_peak, elapsed_ns, lifecycle_elapsed_ns


def _publish_worker_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    raw = b"".join(canonical_json_bytes(dict(record)) + b"\n" for record in records)
    if not raw:
        raise ContractError("worker JSONL cannot be empty")
    atomic_publish_bytes(path, raw)


def _assert_artifact_safe(raw: bytes, forbidden_source_keys: set[str]) -> None:
    lowered = raw.lower()
    forbidden = (
        b"s3://",
        MOUNT_PREFIX.encode("ascii"),
        S3_BUCKET.encode("ascii"),
        S3_KEY_PREFIX.encode("ascii"),
        b"aws_secret_access_key",
        b"aws_session_token",
        b"x-amz-credential",
        b"x-amz-security-token",
        b"authorization:",
        b'"versionid"',
        b'"etag"',
    )
    if any(token in lowered for token in forbidden):
        raise ContractError("sensitive material detected in capture artifact")
    for token in re.findall(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", raw):
        if token.decode("ascii") in forbidden_source_keys:
            raise ContractError("source object key detected in capture artifact")


def _seal_capture(
    *,
    attempt_root: Path,
    attempt_nonce: str,
    selection_sha256: str,
    status: str,
    input_attestation: Mapping[str, Mapping[str, Any]],
    forbidden_source_keys: set[str],
) -> None:
    if status not in {"COMPLETE", "AUTH_UNAVAILABLE", "SOURCE_UNAVAILABLE"}:
        raise ContractError("invalid completion status")
    verify_input_attestation(input_attestation)
    payload_paths: list[Path] = [
        attempt_root / "attempt_contract.json",
        attempt_root / "selection.json",
        attempt_root / "preflight.json",
    ]
    if status == "COMPLETE":
        for _arm, _backend, _cohort, directory in ARM_SPECS:
            arm_root = attempt_root / "arms" / directory
            payload_paths.extend(
                [
                    arm_root / "arm_contract.json",
                    *(arm_root / f"worker_{worker_id:02d}.jsonl" for worker_id in range(PROCESS_COUNT)),
                    arm_root / "arm_result.json",
                ]
            )
    files: dict[str, dict[str, Any]] = {}
    for path in payload_paths:
        raw = path.read_bytes()
        _assert_artifact_safe(raw, forbidden_source_keys)
        files[path.relative_to(attempt_root).as_posix()] = _artifact_attestation(path)
    manifest_path = attempt_root / "capture_manifest.json"
    atomic_publish_json(
        manifest_path,
        {
            "schema_version": CAPTURE_MANIFEST_SCHEMA,
            "status": "SEALED",
            "attempt_nonce": attempt_nonce,
            "selection_sha256": selection_sha256,
            "files": files,
        },
    )
    verify_input_attestation(input_attestation)
    manifest_attestation = _artifact_attestation(manifest_path)
    atomic_publish_json(
        attempt_root / "capture_complete.json",
        {
            "schema_version": CAPTURE_COMPLETE_SCHEMA,
            "status": status,
            "attempt_nonce": attempt_nonce,
            "selection_sha256": selection_sha256,
            "arm_order": [spec[0] for spec in ARM_SPECS],
            "capture_manifest": manifest_attestation,
            "completed_at_unix": int(time.time()),
        },
    )


def _ensure_real_directory(path: Path) -> None:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        _require_real_hierarchy(path, "output directory")
        return
    _ensure_real_directory(path.parent)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    _require_real_hierarchy(path, "output directory")


def _arm_aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    succeeded = sum(record["outcome"] == "success" for record in records)
    return {
        "items": len(records),
        "succeeded": succeeded,
        "failed": len(records) - succeeded,
        "total_bytes": sum(record["byte_count"] for record in records),
        "total_io_ns": sum(record["io_ns"] for record in records),
        "object_attempts": sum(record["object_attempts"] for record in records),
        "object_retry_attempts": sum(max(0, record["object_attempts"] - 1) for record in records),
        "sdk_retry_attempts": sum(record["sdk_retry_attempts"] for record in records),
        "version_id_present_items": sum(record["version_id_present"] for record in records),
        "response_header_count": sum(record["response_header_count"] for record in records),
        "body_close_count": sum(record["body_close_count"] for record in records),
        "preheader_retry_attempts": sum(record["preheader_retry_attempts"] for record in records),
        "version_id_retry_attempts": sum(record["version_id_retry_attempts"] for record in records),
        "if_match_retry_attempts": sum(record["if_match_retry_attempts"] for record in records),
        "unguarded_postheader_retry_attempts": sum(
            record["unguarded_postheader_retry_attempts"] for record in records
        ),
    }


def _single_s3_permission_probe(item: SourceItem, chunk_bytes: int) -> None:
    location = cache_item_location(item)
    client = None
    try:
        session = boto3.Session(region_name="us-west-2")
        client = session.client("s3", config=build_botocore_config())
        s3_get_object_bytes(
            client,
            bucket=location["bucket"],
            key=location["key"],
            chunk_bytes=chunk_bytes,
            max_attempts=MAX_OBJECT_ATTEMPTS,
        )
    finally:
        if client is not None and callable(getattr(client, "close", None)):
            client.close()


def capture(args: argparse.Namespace) -> Path:
    if args.run_id != RUN_ID:
        raise ContractError("run-id drift")
    if args.processes != PROCESS_COUNT or args.threads_per_process != THREADS_PER_PROCESS:
        raise ContractError("capture topology drift")
    if args.object_attempts != MAX_OBJECT_ATTEMPTS:
        raise ContractError("object attempt budget drift")
    cache_path = Path(args.cache).absolute()
    items = load_cache(cache_path, expected_sha256=args.cache_sha256)
    selection = build_selection(items)
    jobtag = args.jobtag
    if type(jobtag) is not str or SAFE_TOKEN_RE.fullmatch(jobtag) is None:
        raise ContractError("jobtag is malformed")
    output_parent = Path(args.output_parent).absolute()
    _ensure_real_directory(output_parent)
    run_root = output_parent / RUN_ID
    _ensure_real_directory(run_root)
    job_root = run_root / jobtag
    _mkdir_real(job_root)
    attempt_nonce = secrets.token_hex(32)
    attempt_root = job_root / attempt_nonce
    _mkdir_real(attempt_root)

    input_attestation = build_input_attestation(cache_path)
    verify_input_attestation(input_attestation)
    session = boto3.Session(region_name="us-west-2")
    credentials = credential_summary(session)
    attempt_contract = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "READY",
        "run_id": RUN_ID,
        "jobtag": jobtag,
        "attempt_nonce": attempt_nonce,
        "created_at_unix": int(time.time()),
        "mapping_id": MAPPING_ID,
        "source_cache": {
            "path": str(cache_path.resolve(strict=True)),
            "bytes": SOURCE_CACHE_BYTES,
            "sha256": SOURCE_CACHE_SHA256,
            "item_count": SOURCE_CACHE_ITEMS,
            "resolve_errors": 0,
        },
        "selection_sha256": selection["selection_sha256"],
        "arm_order": [spec[0] for spec in ARM_SPECS],
        "processes": PROCESS_COUNT,
        "threads_per_process": THREADS_PER_PROCESS,
        "expected_concurrency": EXPECTED_CONCURRENCY,
        "warmup_items": COHORT_SIZES["warmup"],
        "panel_items": COHORT_SIZES["x"],
        "max_object_attempts": MAX_OBJECT_ATTEMPTS,
        "credential_summary": credentials,
        "input_attestation": input_attestation,
    }
    atomic_publish_json(attempt_root / "attempt_contract.json", attempt_contract)
    atomic_publish_json(attempt_root / "selection.json", selection)

    preflight_status = "PASS"
    auth_probe_error_class = "none"
    source_probe_error_class = "none"
    warmup_mount: list[dict[str, Any]] = []
    warmup_s3: list[dict[str, Any]] = []
    peak_inflight = 0
    if credentials["credentials_class"] == "unavailable":
        preflight_status = "AUTH_UNAVAILABLE"
        auth_probe_error_class = "credentials_unavailable"
    else:
        first_warmup = items[selection["cohorts"]["warmup"][0]["source_index"]]
        try:
            _single_s3_permission_probe(first_warmup, args.chunk_bytes)
        except SafeProbeError as exc:
            if exc.error_class == "auth":
                preflight_status = "AUTH_UNAVAILABLE"
                auth_probe_error_class = "access_denied"
            elif exc.error_class == "missing":
                preflight_status = "SOURCE_UNAVAILABLE"
                source_probe_error_class = "missing"
            else:
                raise
    if preflight_status == "PASS":
        verify_input_attestation(input_attestation)
        try:
            warmup_s3, s3_peak, _elapsed, _lifecycle = _run_process_panel(
                items=items,
                selected=selection["cohorts"]["warmup"],
                arm="warmup_s3",
                backend="s3",
                cohort="warmup",
                chunk_bytes=args.chunk_bytes,
                max_attempts=args.object_attempts,
                timeout_s=args.preflight_timeout_s,
            )
            warmup_mount, mount_peak, _elapsed, _lifecycle = _run_process_panel(
                items=items,
                selected=selection["cohorts"]["warmup"],
                arm="warmup_mount",
                backend="mount",
                cohort="warmup",
                chunk_bytes=args.chunk_bytes,
                max_attempts=args.object_attempts,
                timeout_s=args.preflight_timeout_s,
            )
            if s3_peak != EXPECTED_CONCURRENCY or mount_peak != EXPECTED_CONCURRENCY:
                raise ContractError("preflight concurrency drift")
            peak_inflight = EXPECTED_CONCURRENCY
        except SafeProbeError as exc:
            warmup_mount = []
            warmup_s3 = []
            peak_inflight = 0
            if exc.error_class == "auth":
                preflight_status = "AUTH_UNAVAILABLE"
                auth_probe_error_class = "access_denied"
            elif exc.error_class == "missing":
                preflight_status = "SOURCE_UNAVAILABLE"
                source_probe_error_class = "missing"
            else:
                raise
    atomic_publish_json(
        attempt_root / "preflight.json",
        {
            "schema_version": PREFLIGHT_SCHEMA,
            "status": preflight_status,
            "attempt_nonce": attempt_nonce,
            "selection_sha256": selection["selection_sha256"],
            "mapping_id": MAPPING_ID,
            "credential_summary": credentials,
            "auth_probe_error_class": auth_probe_error_class,
            "source_probe_error_class": source_probe_error_class,
            "peak_inflight": peak_inflight,
            "warmup": {"mount": warmup_mount, "s3": warmup_s3},
        },
    )
    forbidden_source_keys = {item.src_key for item in items}
    if preflight_status != "PASS":
        _seal_capture(
            attempt_root=attempt_root,
            attempt_nonce=attempt_nonce,
            selection_sha256=selection["selection_sha256"],
            status=preflight_status,
            input_attestation=input_attestation,
            forbidden_source_keys=forbidden_source_keys,
        )
        return attempt_root

    arms_root = attempt_root / "arms"
    _mkdir_real(arms_root)
    previous_result_sha: str | None = None
    previous_completed_at = 0
    for sequence_index, (arm, backend, cohort, directory) in enumerate(ARM_SPECS):
        verify_input_attestation(input_attestation)
        arm_root = arms_root / directory
        _mkdir_real(arm_root)
        created_at = max(int(time.time()), previous_completed_at)
        common = {
            "attempt_nonce": attempt_nonce,
            "selection_sha256": selection["selection_sha256"],
            "arm": arm,
            "backend": backend,
            "cohort": cohort,
            "sequence_index": sequence_index,
            "previous_arm_result_sha256": previous_result_sha,
            "processes": PROCESS_COUNT,
            "threads_per_process": THREADS_PER_PROCESS,
            "expected_concurrency": EXPECTED_CONCURRENCY,
            "panel_items": COHORT_SIZES[cohort],
        }
        atomic_publish_json(
            arm_root / "arm_contract.json",
            {"schema_version": ARM_CONTRACT_SCHEMA, "status": "READY", **common, "created_at_unix": created_at},
        )
        started_at = max(int(time.time()), previous_completed_at)
        records, peak, elapsed_ns, lifecycle_ns = _run_process_panel(
            items=items,
            selected=selection["cohorts"][cohort],
            arm=arm,
            backend=backend,
            cohort=cohort,
            warmup_selected=selection["cohorts"]["warmup"],
            chunk_bytes=args.chunk_bytes,
            max_attempts=args.object_attempts,
            timeout_s=args.arm_timeout_s,
        )
        verify_input_attestation(input_attestation)
        completed_at = max(int(time.time()), started_at)
        worker_files: dict[str, dict[str, Any]] = {}
        for worker_id in range(PROCESS_COUNT):
            name = f"worker_{worker_id:02d}.jsonl"
            path = arm_root / name
            worker_records = [
                record for record in records if record["worker_id"] == worker_id
            ]
            _publish_worker_jsonl(path, worker_records)
            worker_files[name] = _artifact_attestation(path)
        aggregate = _arm_aggregate(records)
        result_path = arm_root / "arm_result.json"
        atomic_publish_json(
            result_path,
            {
                "schema_version": ARM_RESULT_SCHEMA,
                "status": "PASS" if aggregate["failed"] == 0 else "FAIL",
                **common,
                "started_at_unix": started_at,
                "completed_at_unix": completed_at,
                "elapsed_ns": elapsed_ns,
                "lifecycle_elapsed_ns": lifecycle_ns,
                "peak_inflight": peak,
                **aggregate,
                "worker_files": worker_files,
            },
        )
        previous_result_sha = sha256_file(result_path)
        previous_completed_at = completed_at

    verify_input_attestation(input_attestation)
    _seal_capture(
        attempt_root=attempt_root,
        attempt_nonce=attempt_nonce,
        selection_sha256=selection["selection_sha256"],
        status="COMPLETE",
        input_attestation=input_attestation,
        forbidden_source_keys=forbidden_source_keys,
    )
    return attempt_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--cache", default=str(SOURCE_CACHE))
    capture_parser.add_argument("--cache-sha256", default=SOURCE_CACHE_SHA256)
    capture_parser.add_argument("--output-parent", required=True)
    capture_parser.add_argument("--run-id", default=RUN_ID)
    capture_parser.add_argument("--jobtag", required=True)
    capture_parser.add_argument("--processes", type=int, default=PROCESS_COUNT)
    capture_parser.add_argument("--threads-per-process", type=int, default=THREADS_PER_PROCESS)
    capture_parser.add_argument("--chunk-bytes", type=int, default=8_388_608)
    capture_parser.add_argument("--object-attempts", type=int, default=MAX_OBJECT_ATTEMPTS)
    capture_parser.add_argument("--preflight-timeout-s", type=int, default=600)
    capture_parser.add_argument("--arm-timeout-s", type=int, default=1_800)
    capture_parser.add_argument("--progress-interval-s", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.command != "capture":
            raise ContractError("unsupported command")
        root = capture(args)
        complete = json.loads((root / "capture_complete.json").read_text(encoding="utf-8"))
        print(f"GPIC_MOUNT_S3_ABBA_CAPTURE status={complete['status']} attempt={root.name}")
        if complete["status"] == "AUTH_UNAVAILABLE":
            return 3
        if complete["status"] == "SOURCE_UNAVAILABLE":
            return 4
        return 0
    except ContractError:
        print("GPIC_MOUNT_S3_ABBA_CAPTURE_FAILED class=contract", file=sys.stderr)
        return 2
    except SafeProbeError as exc:
        print(f"GPIC_MOUNT_S3_ABBA_CAPTURE_FAILED class={exc.error_class}", file=sys.stderr)
        return 5
    except BaseException:
        print("GPIC_MOUNT_S3_ABBA_CAPTURE_FAILED class=unexpected", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
