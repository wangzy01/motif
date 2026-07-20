#!/usr/bin/env python3
"""Independently analyze the GPIC mountpoint/S3 GetObject ABBA probe.

This program deliberately does not import the capture runner.  It revalidates
the frozen 20k-item cache, reproduces the deterministic warmup/X/Y selection,
closes every captured byte against a no-clobber manifest, and then evaluates
semantic, retry-safety, concurrency, secrecy, and performance gates.

Exit codes are part of the interface: 0=GO, 3=NO_GO, 4=INCONCLUSIVE, and
2=invalid/tampered contract.  Diagnostics never include object keys, URIs,
credential material, or raw capture exceptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


ANALYSIS_SCHEMA = "offline_arb_gpic_mount_s3_abba_analysis_v1"
ANALYSIS_INPUT_SCHEMA = "offline_arb_gpic_mount_s3_abba_analysis_input_v1"
ATTEMPT_SCHEMA = "offline_arb_gpic_mount_s3_abba_attempt_contract_v1"
SELECTION_SCHEMA = "offline_arb_gpic_s3_mount_abba_selection_v1"
PREFLIGHT_SCHEMA = "offline_arb_gpic_mount_s3_abba_preflight_v1"
ARM_CONTRACT_SCHEMA = "offline_arb_gpic_mount_s3_abba_arm_contract_v1"
ARM_RESULT_SCHEMA = "offline_arb_gpic_mount_s3_abba_arm_result_v1"
CAPTURE_MANIFEST_SCHEMA = "offline_arb_gpic_mount_s3_abba_capture_manifest_v1"
CAPTURE_COMPLETE_SCHEMA = "offline_arb_gpic_mount_s3_abba_capture_complete_v1"

SOURCE_CACHE_BYTES = 6_220_043
SOURCE_CACHE_SHA256 = (
    "3c45ba6e3898b546dbc191e6b708e0bffc7047fd059708b7af8cc0e190cfa298"
)
SOURCE_CACHE_ITEMS = 20_000
SOURCE_ROOT = "/s3-mewtant-raw-datasets/stanford-vision-lab/gpic_umziped"
SOURCE_FIELDS = ("uid", "src_root", "src_shard", "src_key", "ext")
SOURCE_KEY_RE = re.compile(r"[0-9a-f]{64}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
JOBTAG_RE = re.compile(r"[A-Za-z0-9_.-]+")

SELECTION_DOMAIN = b"offline_arb_gpic_s3_mount_abba_selection_v1\0"
EXPECTED_SELECTION_SHA256 = (
    "e358061beee8e23f0b154a93ecebd5b4023bb4e1c57492a2c92a007a82cf7a0d"
)
COHORT_SIZES = {"warmup": 64, "x": 2_048, "y": 2_048}
EXTENSION_ORDER = ("png", "jpg")  # scarce extension first
EXTENSION_COUNTS = {"jpg": 16_737, "png": 3_263}
SHARD_BANDS = 16
SHARDS_PER_BAND = 500

PROCESS_COUNT = 8
THREADS_PER_PROCESS = 2
EXPECTED_CONCURRENCY = 16
ITEMS_PER_PANEL = 2_048
ITEMS_PER_WORKER = 256
WARMUP_ITEMS_PER_WORKER = 8
RUN_ID = "offline_arb_gpic_mount_s3_abba_probe_20260712r1"
MAPPING_ID = "gpic_prod_v1"
ARM_SPECS = (
    ("mount_x", "mount", "x", "00_mount_x"),
    ("s3_x", "s3", "x", "01_s3_x"),
    ("s3_y", "s3", "y", "02_s3_y"),
    ("mount_y", "mount", "y", "03_mount_y"),
)

MIN_S3_IMAGES_PER_SECOND = 90.0
MIN_S3_SPEEDUP = 1.50
MAX_S3_PANEL_DRIFT = 0.10
MAX_S3_P99_SECONDS = 2.0
MAX_S3_SDK_RETRY_RATE = 0.001
MAX_OBJECT_ATTEMPTS = 3

ITEM_KEYS = {
    "schema_version",
    "arm",
    "backend",
    "cohort",
    "worker_id",
    "position",
    "source_index",
    "object_id",
    "outcome",
    "byte_count",
    "payload_sha256",
    "io_ns",
    "headers_ns",
    "read_ns",
    "hash_ns",
    "object_attempts",
    "sdk_retry_attempts",
    "version_id_present",
    "first_response_attempt",
    "first_response_identity_kind",
    "response_header_count",
    "body_close_count",
    "preheader_retry_attempts",
    "version_id_retry_attempts",
    "if_match_retry_attempts",
    "unguarded_postheader_retry_attempts",
    "http_status",
    "content_length_match",
    "error_class",
}
ERROR_CLASSES = {
    "none",
    "auth",
    "missing",
    "throttle",
    "transient",
    "short_read",
    "content_length_mismatch",
    "source_changed",
    "unexpected",
}

LEAK_SUBSTRINGS = (
    b"s3://",
    b"/s3-mewtant-raw-datasets",
    b"mewtant-raw-datasets",
    b"stanford-vision-lab/gpic_umziped",
    b"aws_access_key_id",
    b"aws_secret_access_key",
    b"aws_session_token",
    b"x-amz-credential",
    b"x-amz-security-token",
    b"x-amz-signature",
    b"authorization:",
    b"credential=",
    b"signature=",
)
AWS_ACCESS_KEY_RE = re.compile(rb"(?:AKIA|ASIA)[0-9A-Z]{16}")


class ContractError(ValueError):
    """The evidence cannot be trusted as an instance of the frozen contract."""


class SemanticFailure(ValueError):
    """Well-formed evidence proves that the two transports are not equivalent."""


def governed_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Mark every analyzer result as directional evidence, never promotion."""
    result = dict(value)
    for key, expected in (
        ("direction_probe_only", True),
        ("production_selection_qualified", False),
    ):
        if key in result and result[key] is not expected:
            raise ContractError("analysis governance field mismatch")
        result[key] = expected
    return result


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
        raise ContractError("value is not canonical-JSON serializable") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{label} must be an integer >= {minimum}")
    return value


def finite_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ContractError(f"{label} must be finite and >= {minimum}")
    return result


def exact_keys(value: Any, wanted: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != wanted:
        raise ContractError(f"{label} exact key set mismatch")
    return value


def require_regular_file(path: Path, label: str, *, nonempty: bool = True) -> None:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} is missing, non-regular, or a symlink")
    if nonempty and path.stat().st_size <= 0:
        raise ContractError(f"{label} is empty")


def require_real_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ContractError(f"{label} is missing, non-directory, or a symlink")


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    require_regular_file(path, label)
    try:
        raw = path.read_bytes()
        if b"\x00" in raw:
            raise ContractError(f"{label} contains NUL")
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} is not a JSON object")
    return value


def _selection_hash(*parts: bytes) -> str:
    return hashlib.sha256(SELECTION_DOMAIN + b"\0".join(parts)).hexdigest()


def _hamilton(total: int, weights: Mapping[Any, int]) -> dict[Any, int]:
    if type(total) is not int or total < 0 or not weights:
        raise ContractError("invalid Hamilton allocation inputs")
    if any(type(value) is not int or value < 0 for value in weights.values()):
        raise ContractError("Hamilton weights must be nonnegative integers")
    denominator = sum(weights.values())
    if denominator <= 0:
        raise ContractError("Hamilton allocation has zero weight")
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


def _validate_source_item(index: int, value: Any) -> SourceItem:
    item = exact_keys(value, set(SOURCE_FIELDS), f"cache item {index}")
    if any(type(item[field]) is not str for field in SOURCE_FIELDS):
        raise ContractError(f"cache item {index} contains a non-string field")
    src_key = item["src_key"]
    src_shard = item["src_shard"]
    ext = item["ext"]
    if SOURCE_KEY_RE.fullmatch(src_key) is None:
        raise ContractError(f"cache item {index} key shape mismatch")
    if item["uid"] != f"gpic:{src_key}":
        raise ContractError(f"cache item {index} uid/key binding mismatch")
    if item["src_root"] != SOURCE_ROOT:
        raise ContractError(f"cache item {index} root mismatch")
    if re.fullmatch(r"[0-9]{5}", src_shard) is None or int(src_shard) >= 8_000:
        raise ContractError(f"cache item {index} shard mismatch")
    if ext not in {".jpg", ".png"}:
        raise ContractError(f"cache item {index} extension mismatch")
    if any(token in src_key or token in src_shard for token in ("/", "\\", "\x00")):
        raise ContractError(f"cache item {index} contains a path separator")
    canonical = canonical_json_bytes({field: item[field] for field in SOURCE_FIELDS})
    return SourceItem(
        source_index=index,
        canonical=canonical,
        object_id=_selection_hash(b"item", canonical),
        uid=item["uid"],
        src_root=item["src_root"],
        src_shard=src_shard,
        src_key=src_key,
        ext=ext,
        ext_class=ext[1:],
        shard_band=int(src_shard) // SHARDS_PER_BAND,
    )


def load_and_validate_cache(
    path: Path, *, expected_sha256: str = SOURCE_CACHE_SHA256
) -> list[SourceItem]:
    require_regular_file(path, "source cache")
    if path.stat().st_size != SOURCE_CACHE_BYTES:
        raise ContractError("source cache byte size mismatch")
    if sha256_file(path) != expected_sha256 or expected_sha256 != SOURCE_CACHE_SHA256:
        raise ContractError("source cache SHA256 mismatch")
    value = read_json_object(path, "source cache")
    exact_keys(value, {"items", "resolve_errors"}, "source cache")
    if value["resolve_errors"] != 0 or type(value["resolve_errors"]) is not int:
        raise ContractError("source cache has resolver errors")
    raw_items = value["items"]
    if not isinstance(raw_items, list) or len(raw_items) != SOURCE_CACHE_ITEMS:
        raise ContractError("source cache item count mismatch")
    items = [_validate_source_item(index, raw) for index, raw in enumerate(raw_items)]
    if len({item.uid for item in items}) != SOURCE_CACHE_ITEMS:
        raise ContractError("source cache uid uniqueness mismatch")
    if len({item.src_key for item in items}) != SOURCE_CACHE_ITEMS:
        raise ContractError("source cache key uniqueness mismatch")
    physical = {(item.src_root, item.src_shard, item.src_key, item.ext) for item in items}
    if len(physical) != SOURCE_CACHE_ITEMS:
        raise ContractError("source cache physical-object uniqueness mismatch")
    if Counter(item.ext_class for item in items) != Counter(EXTENSION_COUNTS):
        raise ContractError("source cache extension distribution mismatch")
    return items


def build_expected_selection(items: Sequence[SourceItem]) -> dict[str, Any]:
    if len(items) != SOURCE_CACHE_ITEMS:
        raise ContractError("selection source item count mismatch")
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
        extension_quota = _hamilton(total, ext_weights)
        chosen: list[tuple[SourceItem, str]] = []
        for ext in EXTENSION_ORDER:
            band_quota = _hamilton(extension_quota[ext], band_weights[ext])
            for band in range(SHARD_BANDS):
                best_by_shard: dict[str, tuple[str, str, SourceItem]] = {}
                for item in items:
                    if (
                        item.ext_class != ext
                        or item.shard_band != band
                        or item.src_shard in used_shards
                    ):
                        continue
                    member_score = _selection_hash(
                        b"member", cohort.encode("ascii"), item.canonical
                    )
                    candidate = (member_score, item.object_id, item)
                    previous = best_by_shard.get(item.src_shard)
                    if previous is None or candidate[:2] < previous[:2]:
                        best_by_shard[item.src_shard] = candidate
                candidates = sorted(best_by_shard.values(), key=lambda value: value[:2])
                quota = band_quota[band]
                if len(candidates) < quota:
                    raise ContractError("selection stratum lacks distinct physical shards")
                for _score, _object_id, item in candidates[:quota]:
                    used_shards.add(item.src_shard)
                    order_score = _selection_hash(
                        b"order", cohort.encode("ascii"), item.object_id.encode("ascii")
                    )
                    chosen.append((item, order_score))
        chosen.sort(key=lambda value: (value[1], value[0].object_id))
        if len(chosen) != total:
            raise ContractError("selection cohort size did not close")
        cohorts[cohort] = [
            {
                "source_index": item.source_index,
                "object_id": item.object_id,
                "ext_class": item.ext_class,
                "shard_band": item.shard_band,
            }
            for item, _order_score in chosen
        ]
    selection_core = {
        "schema_version": SELECTION_SCHEMA,
        "source_cache_sha256": SOURCE_CACHE_SHA256,
        "cohorts": cohorts,
    }
    if len(used_shards) != sum(COHORT_SIZES.values()):
        raise ContractError("selection physical-shard closure mismatch")
    selection_sha256 = canonical_sha256(selection_core)
    if selection_sha256 != EXPECTED_SELECTION_SHA256:
        raise ContractError("independent selection reference SHA256 mismatch")
    return {**selection_core, "selection_sha256": selection_sha256}


def validate_and_select_cache(
    path: Path, *, expected_sha256: str = SOURCE_CACHE_SHA256
) -> tuple[list[SourceItem], dict[str, Any]]:
    items = load_and_validate_cache(path, expected_sha256=expected_sha256)
    return items, build_expected_selection(items)


def verify_file_attestation(
    path: Path | str, expected_bytes: int, expected_sha256: str
) -> bytes:
    """Read a regular file once and close its size/SHA/stable-inode attestation."""
    path = Path(path)
    expected_bytes = strict_int(expected_bytes, "attested bytes")
    if type(expected_sha256) is not str or SHA256_RE.fullmatch(expected_sha256) is None:
        raise ContractError("attested SHA256 is malformed")
    if path.is_symlink():
        raise ContractError("attested file is a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractError("attested file cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise ContractError("attested file type/size mismatch")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        remaining = expected_bytes
        while remaining:
            block = os.read(descriptor, min(1 << 20, remaining))
            if not block:
                raise ContractError("attested file ended early")
            chunks.append(block)
            digest.update(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ContractError("attested file grew while read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fingerprint_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    fingerprint_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if fingerprint_before != fingerprint_after:
        raise ContractError("attested file changed while read")
    if digest.hexdigest() != expected_sha256:
        raise ContractError("attested file SHA256 mismatch")
    return b"".join(chunks)


def scan_for_leaks(
    payload_or_path: bytes | bytearray | memoryview | Path | str,
    forbidden_source_keys: Iterable[str] = (),
) -> None:
    """Reject credential, URI, object-key, raw VersionId, and raw ETag material."""
    if isinstance(payload_or_path, (Path, str)):
        path = Path(payload_or_path)
        require_regular_file(path, "leak-scan input", nonempty=False)
        payload = path.read_bytes()
    elif isinstance(payload_or_path, (bytes, bytearray, memoryview)):
        payload = bytes(payload_or_path)
    else:
        raise ContractError("leak-scan input type mismatch")
    lowered = payload.lower()
    if any(token in lowered for token in LEAK_SUBSTRINGS):
        raise ContractError("sensitive URI/credential material detected")
    if AWS_ACCESS_KEY_RE.search(payload) is not None:
        raise ContractError("access-key-shaped material detected")
    # Only boolean/counter evidence is permitted; raw response identity values are not.
    if any(
        token in lowered
        for token in (
            b'"versionid"',
            b'"version_id"',
            b'"etag"',
            b'"e_tag"',
        )
    ):
        raise ContractError("raw object-version identity material detected")
    forbidden = set(forbidden_source_keys)
    if forbidden:
        for token in re.findall(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", payload):
            if token.decode("ascii") in forbidden:
                raise ContractError("raw source-object key detected")


def _semantic_record(record: Any, label: str) -> tuple[int, str, int, str]:
    if not isinstance(record, dict):
        raise ContractError(f"{label} record must be an object")
    source_index = strict_int(record.get("source_index"), f"{label}.source_index")
    object_id = record.get("object_id")
    if type(object_id) is not str or SHA256_RE.fullmatch(object_id) is None:
        raise ContractError(f"{label}.object_id is malformed")
    outcome = record.get("outcome")
    if outcome != "success":
        raise SemanticFailure(f"{label} contains a failed object read")
    byte_count = strict_int(record.get("byte_count"), f"{label}.byte_count", minimum=1)
    payload_sha256 = record.get("payload_sha256")
    if type(payload_sha256) is not str or SHA256_RE.fullmatch(payload_sha256) is None:
        raise ContractError(f"{label}.payload_sha256 is malformed")
    return source_index, object_id, byte_count, payload_sha256


def validate_payload_pairs(
    records_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    """Close same-object bytes/SHA for X and Y without trusting aggregate files."""
    expected_arms = {spec[0] for spec in ARM_SPECS}
    if not isinstance(records_by_arm, Mapping) or set(records_by_arm) != expected_arms:
        raise ContractError("payload-pair arm set mismatch")
    indexed: dict[str, dict[int, tuple[str, int, str]]] = {}
    for arm, records in records_by_arm.items():
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ContractError("payload-pair record collection is malformed")
        arm_index: dict[int, tuple[str, int, str]] = {}
        for position, record in enumerate(records):
            source_index, object_id, byte_count, payload_sha256 = _semantic_record(
                record, f"{arm}[{position}]"
            )
            if source_index in arm_index:
                raise ContractError("payload-pair duplicate source index")
            arm_index[source_index] = (object_id, byte_count, payload_sha256)
        indexed[arm] = arm_index
    pairs = (("mount_x", "s3_x", "x"), ("s3_y", "mount_y", "y"))
    result: dict[str, Any] = {}
    for left, right, cohort in pairs:
        if set(indexed[left]) != set(indexed[right]):
            raise SemanticFailure("same-cohort object sets differ across transports")
        mismatch_count = sum(
            indexed[left][source_index] != indexed[right][source_index]
            for source_index in indexed[left]
        )
        if mismatch_count:
            raise SemanticFailure("same-object byte/SHA closure failed")
        result[cohort] = {
            "objects": len(indexed[left]),
            "bytes": sum(value[1] for value in indexed[left].values()),
            "mismatches": 0,
        }
    return result


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ContractError("invalid percentile input")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_item_record(
    value: Any,
    *,
    arm: str,
    backend: str,
    cohort: str,
    worker_id: int,
    expected_selection: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    record = exact_keys(value, ITEM_KEYS, "item evidence")
    if record["schema_version"] != "offline_arb_gpic_mount_s3_abba_item_v1":
        raise ContractError("item evidence schema mismatch")
    position = strict_int(record["position"], "item position")
    if position >= len(expected_selection) or position % PROCESS_COUNT != worker_id:
        raise ContractError("item position/worker assignment mismatch")
    selected = expected_selection[position]
    if (
        record["arm"] != arm
        or record["backend"] != backend
        or record["cohort"] != cohort
        or record["worker_id"] != worker_id
        or record["source_index"] != selected["source_index"]
        or record["object_id"] != selected["object_id"]
    ):
        raise ContractError("item identity/selection binding mismatch")
    if record["outcome"] not in {"success", "error"}:
        raise ContractError("item outcome mismatch")
    byte_count = strict_int(record["byte_count"], "item byte count")
    for name in (
        "io_ns",
        "headers_ns",
        "read_ns",
        "hash_ns",
        "object_attempts",
        "sdk_retry_attempts",
        "first_response_attempt",
        "response_header_count",
        "body_close_count",
        "preheader_retry_attempts",
        "version_id_retry_attempts",
        "if_match_retry_attempts",
        "unguarded_postheader_retry_attempts",
    ):
        strict_int(record[name], f"item {name}")
    if type(record["version_id_present"]) is not bool:
        raise ContractError("item version-presence evidence must be boolean")
    if record["first_response_identity_kind"] not in {
        "none",
        "version_pin",
        "if_match_pin",
    }:
        raise ContractError("item first-response identity kind is not allowlisted")
    if type(record["content_length_match"]) is not bool:
        raise ContractError("item content-length evidence must be boolean")
    if type(record["http_status"]) is not int:
        raise ContractError("item HTTP status must be an integer")
    if record["error_class"] not in ERROR_CLASSES:
        raise ContractError("item error class is not an allowed enum")
    attempts = record["object_attempts"]
    first_response_attempt = record["first_response_attempt"]
    if attempts > MAX_OBJECT_ATTEMPTS:
        raise ContractError("item object-attempt bound mismatch")
    if backend == "mount":
        if (
            record["http_status"] != 0
            or record["version_id_present"] is not False
            or first_response_attempt != 0
            or record["first_response_identity_kind"] != "none"
            or record["response_header_count"] != 0
            or record["preheader_retry_attempts"] != 0
            or record["version_id_retry_attempts"] != 0
            or record["if_match_retry_attempts"] != 0
            or record["unguarded_postheader_retry_attempts"] != 0
            or record["sdk_retry_attempts"] != 0
            or record["body_close_count"] != 0
            or attempts != 1
        ):
            raise ContractError("mount transport retry contract mismatch")
    elif backend == "s3":
        if (
            attempts < 1
            or record["response_header_count"] > attempts
            or record["body_close_count"] != record["response_header_count"]
        ):
            raise ContractError("S3 response/body-close contract mismatch")
        pin_retries = (
            record["version_id_retry_attempts"]
            + record["if_match_retry_attempts"]
            + record["unguarded_postheader_retry_attempts"]
        )
        if first_response_attempt == 0:
            if (
                record["response_header_count"] != 0
                or record["preheader_retry_attempts"] != attempts - 1
                or pin_retries != 0
                or record["first_response_identity_kind"] != "none"
            ):
                raise ContractError("S3 no-response retry closure mismatch")
        elif (
            first_response_attempt > attempts
            or record["preheader_retry_attempts"] != first_response_attempt - 1
            or pin_retries != attempts - first_response_attempt
            or not (
                1
                <= record["response_header_count"]
                <= attempts - record["preheader_retry_attempts"]
            )
        ):
            raise ContractError("S3 first-response/pinned-retry closure mismatch")
        postheader_attempts = attempts - first_response_attempt
        identity_kind = record["first_response_identity_kind"]
        if identity_kind == "none":
            if postheader_attempts != 0 or pin_retries != 0:
                raise ContractError("S3 missing first-response identity for retry")
        elif identity_kind == "version_pin":
            if (
                not record["version_id_present"]
                or record["if_match_retry_attempts"] != 0
                or record["version_id_retry_attempts"]
                + record["unguarded_postheader_retry_attempts"]
                != postheader_attempts
            ):
                raise ContractError("S3 VersionId pin identity/counter mismatch")
        elif (
            record["version_id_present"]
            or record["version_id_retry_attempts"] != 0
            or record["if_match_retry_attempts"]
            + record["unguarded_postheader_retry_attempts"]
            != postheader_attempts
        ):
            raise ContractError("S3 ETag pin identity/counter mismatch")
        if record["version_id_retry_attempts"] and not record["version_id_present"]:
            raise ContractError("S3 VersionId retry lacks VersionId presence evidence")
    else:
        raise ContractError("unknown item backend")
    payload_sha = record["payload_sha256"]
    if record["outcome"] == "success":
        if (
            byte_count <= 0
            or type(payload_sha) is not str
            or SHA256_RE.fullmatch(payload_sha) is None
            or record["error_class"] != "none"
            or record["content_length_match"] is not True
            or record["io_ns"] <= 0
            or record["read_ns"] <= 0
            or (backend == "s3" and record["http_status"] != 200)
            or (backend == "s3" and record["response_header_count"] < 1)
        ):
            raise ContractError("successful item evidence is internally inconsistent")
    else:
        if (
            byte_count != 0
            or payload_sha != ""
            or record["error_class"] == "none"
            or record["content_length_match"] is not False
        ):
            raise ContractError("failed item evidence is internally inconsistent")
    return record


def _read_worker_jsonl(
    path: Path,
    *,
    arm: str,
    backend: str,
    cohort: str,
    worker_id: int,
    expected_selection: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    require_regular_file(path, "worker evidence")
    return _parse_worker_jsonl(
        path.read_bytes(),
        arm=arm,
        backend=backend,
        cohort=cohort,
        worker_id=worker_id,
        expected_selection=expected_selection,
    )


def _parse_worker_jsonl(
    raw: bytes,
    *,
    arm: str,
    backend: str,
    cohort: str,
    worker_id: int,
    expected_selection: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\x00" in raw:
        raise ContractError("worker JSONL framing mismatch")
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line:
            raise ContractError("worker JSONL contains an empty line")
        try:
            value = json.loads(line.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("worker JSONL contains invalid JSON") from exc
        if canonical_json_bytes(value) != line:
            raise ContractError("worker JSONL line is not canonical JSON")
        records.append(
            _validate_item_record(
                value,
                arm=arm,
                backend=backend,
                cohort=cohort,
                worker_id=worker_id,
                expected_selection=expected_selection,
            )
        )
    wanted = [
        position
        for position in range(len(expected_selection))
        if position % PROCESS_COUNT == worker_id
    ]
    if [record["position"] for record in records] != wanted:
        raise ContractError("worker JSONL position/order closure mismatch")
    return records


CREDENTIAL_METHODS = {
    "assume-role-with-web-identity",
    "assume-role",
    "container-role",
    "iam-role",
    "sso",
    "custom-process",
    "env",
    "shared-credentials-file",
    "config-file",
    "unavailable",
}
CREDENTIAL_CLASSES = {
    "refreshable_temporary",
    "nonrefreshable_temporary",
    "long_lived_static",
    "unavailable",
}


def _validate_credential_summary(value: Any, label: str) -> dict[str, Any]:
    summary = exact_keys(
        value,
        {"method", "credentials_class", "refreshable", "has_session_token"},
        label,
    )
    if summary["method"] not in CREDENTIAL_METHODS:
        raise ContractError(f"{label} method is not sanitized/allowlisted")
    if summary["credentials_class"] not in CREDENTIAL_CLASSES:
        raise ContractError(f"{label} class is not sanitized/allowlisted")
    if type(summary["refreshable"]) is not bool:
        raise ContractError(f"{label} refreshability must be boolean")
    if type(summary["has_session_token"]) is not bool:
        raise ContractError(f"{label} session-token presence must be boolean")
    if summary["refreshable"] != (
        summary["credentials_class"] == "refreshable_temporary"
    ):
        raise ContractError(f"{label} lifecycle fields are inconsistent")
    if (summary["method"] == "unavailable") != (
        summary["credentials_class"] == "unavailable"
    ):
        raise ContractError(f"{label} availability fields are inconsistent")
    if summary["credentials_class"] in {"unavailable", "long_lived_static"} and summary[
        "has_session_token"
    ]:
        raise ContractError(f"{label} class/session-token fields are inconsistent")
    if (
        summary["credentials_class"] == "nonrefreshable_temporary"
        and not summary["has_session_token"]
    ):
        raise ContractError(f"{label} temporary class lacks session-token evidence")
    return summary


def _credential_go_eligible(summary: Mapping[str, Any]) -> bool:
    return bool(summary["refreshable"]) or summary["credentials_class"] == "long_lived_static"


def _json_from_attested_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        if b"\x00" in raw:
            raise ContractError(f"{label} contains NUL")
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except ContractError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _stable_unattested_read(path: Path, label: str) -> bytes:
    require_regular_file(path, label)
    first_sha = sha256_file(path)
    first_size = path.stat().st_size
    raw = verify_file_attestation(path, first_size, first_sha)
    if sha256_file(path) != first_sha:
        raise ContractError(f"{label} changed after stable read")
    return raw


def _expected_payload_paths(*, full_capture: bool) -> set[str]:
    result = {"attempt_contract.json", "selection.json", "preflight.json"}
    if full_capture:
        for _arm, _backend, _cohort, directory in ARM_SPECS:
            prefix = f"arms/{directory}"
            result.add(f"{prefix}/arm_contract.json")
            result.add(f"{prefix}/arm_result.json")
            result.update(
                f"{prefix}/worker_{worker_id:02d}.jsonl"
                for worker_id in range(PROCESS_COUNT)
            )
    return result


def _validate_relative_artifact_name(relative: Any) -> str:
    if type(relative) is not str or not relative or "\\" in relative or "\x00" in relative:
        raise ContractError("manifest artifact name is malformed")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ContractError("manifest artifact name is non-canonical")
    if pure.as_posix() != relative:
        raise ContractError("manifest artifact name is not normalized")
    return relative


@dataclass(frozen=True)
class CaptureClosure:
    root: Path
    complete: dict[str, Any]
    complete_sha256: str
    manifest: dict[str, Any]
    manifest_sha256: str
    payloads: dict[str, bytes]


def _validate_capture_closure(
    capture_root: Path, *, forbidden_source_keys: set[str]
) -> CaptureClosure:
    require_real_directory(capture_root, "capture root")
    require_real_directory(capture_root.parent, "capture job root")
    if (
        capture_root.resolve(strict=True).parent != capture_root.parent.resolve(strict=True)
        or capture_root.parent.parent.name != RUN_ID
    ):
        raise ContractError("capture root hierarchy mismatch")
    complete_path = capture_root / "capture_complete.json"
    complete_raw = _stable_unattested_read(complete_path, "capture completion marker")
    complete_sha = hashlib.sha256(complete_raw).hexdigest()
    complete = exact_keys(
        _json_from_attested_bytes(complete_raw, "capture completion marker"),
        {
            "schema_version",
            "status",
            "attempt_nonce",
            "selection_sha256",
            "arm_order",
            "capture_manifest",
            "completed_at_unix",
        },
        "capture completion marker",
    )
    if complete["schema_version"] != CAPTURE_COMPLETE_SCHEMA:
        raise ContractError("capture completion schema mismatch")
    if complete["status"] not in {"COMPLETE", "AUTH_UNAVAILABLE", "SOURCE_UNAVAILABLE"}:
        raise ContractError("capture completion status mismatch")
    attempt_nonce = complete["attempt_nonce"]
    if (
        type(attempt_nonce) is not str
        or SHA256_RE.fullmatch(attempt_nonce) is None
        or capture_root.name != attempt_nonce
    ):
        raise ContractError("capture attempt nonce/path binding mismatch")
    if complete["selection_sha256"] != EXPECTED_SELECTION_SHA256:
        raise ContractError("capture completion selection SHA mismatch")
    if complete["arm_order"] != [spec[0] for spec in ARM_SPECS]:
        raise ContractError("capture completion arm order mismatch")
    strict_int(complete["completed_at_unix"], "capture completion time", minimum=1)
    manifest_decl = exact_keys(
        complete["capture_manifest"], {"bytes", "sha256"}, "capture manifest declaration"
    )
    manifest_path = capture_root / "capture_manifest.json"
    manifest_raw = verify_file_attestation(
        manifest_path,
        strict_int(manifest_decl["bytes"], "capture manifest bytes", minimum=1),
        manifest_decl["sha256"],
    )
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    manifest = exact_keys(
        _json_from_attested_bytes(manifest_raw, "capture manifest"),
        {"schema_version", "status", "attempt_nonce", "selection_sha256", "files"},
        "capture manifest",
    )
    if (
        manifest["schema_version"] != CAPTURE_MANIFEST_SCHEMA
        or manifest["status"] != "SEALED"
        or manifest["attempt_nonce"] != attempt_nonce
        or manifest["selection_sha256"] != EXPECTED_SELECTION_SHA256
    ):
        raise ContractError("capture manifest identity mismatch")
    full_capture = complete["status"] == "COMPLETE"
    expected_paths = _expected_payload_paths(full_capture=full_capture)
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != expected_paths:
        raise ContractError("capture manifest exact artifact set mismatch")
    payloads: dict[str, bytes] = {}
    for relative in sorted(expected_paths):
        _validate_relative_artifact_name(relative)
        declaration = exact_keys(
            files[relative], {"bytes", "sha256"}, "manifest file declaration"
        )
        path = capture_root.joinpath(*PurePosixPath(relative).parts)
        if path.resolve(strict=True).parent != path.parent.resolve(strict=True):
            raise ContractError("capture artifact path escapes its real parent")
        raw = verify_file_attestation(
            path,
            strict_int(declaration["bytes"], "manifest file bytes"),
            declaration["sha256"],
        )
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ContractError("capture artifact is writable after publication")
        scan_for_leaks(raw, forbidden_source_keys)
        payloads[relative] = raw
    for path, label in (
        (manifest_path, "capture manifest"),
        (complete_path, "capture completion marker"),
    ):
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ContractError(f"{label} is writable after publication")
    scan_for_leaks(manifest_raw, forbidden_source_keys)
    scan_for_leaks(complete_raw, forbidden_source_keys)
    root_entries = {path.name for path in capture_root.iterdir()}
    expected_root_entries = {
        "attempt_contract.json",
        "selection.json",
        "preflight.json",
        "capture_manifest.json",
        "capture_complete.json",
    }
    if full_capture:
        expected_root_entries.add("arms")
    if root_entries != expected_root_entries:
        raise ContractError("capture root exact entry set mismatch")
    if full_capture:
        arms_root = capture_root / "arms"
        require_real_directory(arms_root, "capture arms root")
        if {path.name for path in arms_root.iterdir()} != {spec[3] for spec in ARM_SPECS}:
            raise ContractError("capture arm directory exact set mismatch")
        for _arm, _backend, _cohort, directory in ARM_SPECS:
            arm_root = arms_root / directory
            require_real_directory(arm_root, "capture arm root")
            expected_entries = {
                "arm_contract.json",
                "arm_result.json",
                *(f"worker_{worker_id:02d}.jsonl" for worker_id in range(PROCESS_COUNT)),
            }
            if {path.name for path in arm_root.iterdir()} != expected_entries:
                raise ContractError("capture arm exact entry set mismatch")
    # Reclose every attested artifact after parsing the directory namespace.
    for relative, raw in payloads.items():
        declaration = files[relative]
        path = capture_root.joinpath(*PurePosixPath(relative).parts)
        if verify_file_attestation(path, declaration["bytes"], declaration["sha256"]) != raw:
            raise ContractError("capture artifact changed during closure")
    return CaptureClosure(
        root=capture_root,
        complete=complete,
        complete_sha256=complete_sha,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        payloads=payloads,
    )


def _validate_attempt_contract(
    raw: bytes, *, capture_root: Path, cache_path: Path, selection: Mapping[str, Any]
) -> dict[str, Any]:
    contract = exact_keys(
        _json_from_attested_bytes(raw, "attempt contract"),
        {
            "schema_version",
            "status",
            "run_id",
            "jobtag",
            "attempt_nonce",
            "created_at_unix",
            "mapping_id",
            "source_cache",
            "selection_sha256",
            "arm_order",
            "processes",
            "threads_per_process",
            "expected_concurrency",
            "warmup_items",
            "panel_items",
            "max_object_attempts",
            "credential_summary",
            "input_attestation",
        },
        "attempt contract",
    )
    cache = exact_keys(
        contract["source_cache"],
        {"path", "bytes", "sha256", "item_count", "resolve_errors"},
        "attempt source-cache declaration",
    )
    expected_cache = {
        "path": str(cache_path.resolve(strict=True)),
        "bytes": SOURCE_CACHE_BYTES,
        "sha256": SOURCE_CACHE_SHA256,
        "item_count": SOURCE_CACHE_ITEMS,
        "resolve_errors": 0,
    }
    if cache != expected_cache:
        raise ContractError("attempt source-cache declaration mismatch")
    if (
        contract["schema_version"] != ATTEMPT_SCHEMA
        or contract["status"] != "READY"
        or contract["run_id"] != RUN_ID
        or contract["jobtag"] != capture_root.parent.name
        or contract["attempt_nonce"] != capture_root.name
        or contract["mapping_id"] != MAPPING_ID
        or contract["selection_sha256"] != selection["selection_sha256"]
        or contract["arm_order"] != [spec[0] for spec in ARM_SPECS]
        or contract["processes"] != PROCESS_COUNT
        or contract["threads_per_process"] != THREADS_PER_PROCESS
        or contract["expected_concurrency"] != EXPECTED_CONCURRENCY
        or contract["warmup_items"] != COHORT_SIZES["warmup"]
        or contract["panel_items"] != ITEMS_PER_PANEL
        or contract["max_object_attempts"] != MAX_OBJECT_ATTEMPTS
    ):
        raise ContractError("attempt identity/topology contract mismatch")
    if JOBTAG_RE.fullmatch(contract["jobtag"] or "") is None:
        raise ContractError("attempt jobtag is malformed")
    strict_int(contract["created_at_unix"], "attempt creation time", minimum=1)
    _validate_credential_summary(contract["credential_summary"], "attempt credentials")
    _validate_capture_input_attestation(contract["input_attestation"], cache_path)
    return contract


def _validate_capture_input_attestation(
    value: Any, cache_path: Path
) -> dict[str, dict[str, Any]]:
    repo = Path(__file__).resolve(strict=True).parents[2]
    expected_paths = {
        "runner": repo / "scripts/offline_arb/run_gpic_mount_s3_abba_probe.py",
        "analyzer": repo / "scripts/offline_arb/analyze_gpic_mount_s3_abba_probe.py",
        "capture_yaml": repo / "jobs/offline_arb_gpic_mount_s3_abba_probe.yaml",
        "source_cache": cache_path.resolve(strict=True),
    }
    attestation = exact_keys(value, set(expected_paths), "capture input attestation")
    for label, expected_path in expected_paths.items():
        declaration = exact_keys(
            attestation[label], {"path", "bytes", "sha256"}, "capture input declaration"
        )
        if declaration["path"] != str(expected_path):
            raise ContractError("capture input canonical path mismatch")
        require_regular_file(expected_path, "capture live input")
        if expected_path.resolve(strict=True) != expected_path or expected_path.is_symlink():
            raise ContractError("capture live input is not a canonical real file")
        verify_file_attestation(
            expected_path,
            strict_int(declaration["bytes"], "capture input bytes", minimum=1),
            declaration["sha256"],
        )
    return attestation


def _validate_preflight(
    raw: bytes,
    *,
    attempt: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[str]]:
    preflight = exact_keys(
        _json_from_attested_bytes(raw, "preflight"),
        {
            "schema_version",
            "status",
            "attempt_nonce",
            "selection_sha256",
            "mapping_id",
            "credential_summary",
            "auth_probe_error_class",
            "source_probe_error_class",
            "peak_inflight",
            "warmup",
        },
        "preflight",
    )
    if (
        preflight["schema_version"] != PREFLIGHT_SCHEMA
        or preflight["attempt_nonce"] != attempt["attempt_nonce"]
        or preflight["selection_sha256"] != EXPECTED_SELECTION_SHA256
        or preflight["mapping_id"] != MAPPING_ID
    ):
        raise ContractError("preflight identity mismatch")
    credentials = _validate_credential_summary(
        preflight["credential_summary"], "preflight credentials"
    )
    if credentials != attempt["credential_summary"]:
        raise ContractError("attempt/preflight credential summary mismatch")
    warmup = exact_keys(preflight["warmup"], {"mount", "s3"}, "preflight warmup")
    if not isinstance(warmup["mount"], list) or not isinstance(warmup["s3"], list):
        raise ContractError("preflight warmup collections are malformed")
    def validate_optional_terminal_mount() -> list[dict[str, Any]]:
        if not warmup["mount"]:
            return []
        if len(warmup["mount"]) != COHORT_SIZES["warmup"]:
            raise ContractError("terminal preflight mount evidence count mismatch")
        expected_mount = selection["cohorts"]["warmup"]
        parsed_mount = [
            _validate_item_record(
                record,
                arm="warmup_mount",
                backend="mount",
                cohort="warmup",
                worker_id=position % PROCESS_COUNT,
                expected_selection=expected_mount,
            )
            for position, record in enumerate(warmup["mount"])
        ]
        if [record["position"] for record in parsed_mount] != list(
            range(COHORT_SIZES["warmup"])
        ):
            raise ContractError("terminal preflight mount order mismatch")
        return parsed_mount
    auth_error = preflight["auth_probe_error_class"]
    source_error = preflight["source_probe_error_class"]
    if auth_error not in {"none", "credentials_unavailable", "access_denied"}:
        raise ContractError("preflight auth probe class is not allowlisted")
    if source_error not in {"none", "missing"}:
        raise ContractError("preflight source probe class is not allowlisted")
    if preflight["status"] == "AUTH_UNAVAILABLE":
        if auth_error not in {"credentials_unavailable", "access_denied"} or source_error != "none":
            raise ContractError("AUTH_UNAVAILABLE probe class mismatch")
        if auth_error == "credentials_unavailable" and credentials["credentials_class"] != "unavailable":
            raise ContractError("credential-unavailable probe class/summary mismatch")
        if auth_error == "access_denied" and credentials["credentials_class"] == "unavailable":
            raise ContractError("access-denied probe lacks available credentials")
        if warmup["s3"] or preflight["peak_inflight"] != 0:
            raise ContractError("AUTH_UNAVAILABLE contains S3 timing evidence")
        return preflight, {"mount": validate_optional_terminal_mount(), "s3": []}, []
    if preflight["status"] == "SOURCE_UNAVAILABLE":
        if auth_error != "none" or source_error != "missing":
            raise ContractError("SOURCE_UNAVAILABLE probe class mismatch")
        if warmup["s3"] or preflight["peak_inflight"] != 0:
            raise ContractError("SOURCE_UNAVAILABLE contains S3 timing evidence")
        return preflight, {"mount": validate_optional_terminal_mount(), "s3": []}, ["source_unavailable"]
    if (
        preflight["status"] != "PASS"
        or auth_error != "none"
        or source_error != "none"
        or preflight["peak_inflight"] != EXPECTED_CONCURRENCY
    ):
        raise ContractError("preflight status/concurrency mismatch")
    expected = selection["cohorts"]["warmup"]
    parsed: dict[str, list[dict[str, Any]]] = {}
    for backend in ("mount", "s3"):
        source = warmup[backend]
        if len(source) != COHORT_SIZES["warmup"]:
            raise ContractError("preflight warmup item count mismatch")
        records = [
            _validate_item_record(
                record,
                arm=f"warmup_{backend}",
                backend=backend,
                cohort="warmup",
                worker_id=position % PROCESS_COUNT,
                expected_selection=expected,
            )
            for position, record in enumerate(source)
        ]
        if [record["position"] for record in records] != list(range(len(expected))):
            raise ContractError("preflight warmup position closure mismatch")
        parsed[backend] = records
    left = {
        record["source_index"]: (
            record["object_id"], record["byte_count"], record["payload_sha256"]
        )
        for record in parsed["mount"]
    }
    right = {
        record["source_index"]: (
            record["object_id"], record["byte_count"], record["payload_sha256"]
        )
        for record in parsed["s3"]
    }
    semantic_issues = []
    if any(record["outcome"] != "success" for records in parsed.values() for record in records):
        semantic_issues.append("preflight_failed_read")
    if left != right:
        semantic_issues.append("preflight_payload_closure")
    if any(record["unguarded_postheader_retry_attempts"] for record in parsed["s3"]):
        semantic_issues.append("preflight_unguarded_postheader_retry")
    return preflight, parsed, semantic_issues


def _validate_arm(
    closure: CaptureClosure,
    *,
    attempt: Mapping[str, Any],
    selection: Mapping[str, Any],
    arm: str,
    backend: str,
    cohort: str,
    directory: str,
    sequence_index: int,
    previous_arm_result_sha256: str | None,
    previous_completed_at_unix: int | None,
) -> dict[str, Any]:
    prefix = f"arms/{directory}"
    contract = exact_keys(
        _json_from_attested_bytes(
            closure.payloads[f"{prefix}/arm_contract.json"], "arm contract"
        ),
        {
            "schema_version",
            "status",
            "attempt_nonce",
            "selection_sha256",
            "arm",
            "backend",
            "cohort",
            "processes",
            "threads_per_process",
            "expected_concurrency",
            "panel_items",
            "created_at_unix",
            "sequence_index",
            "previous_arm_result_sha256",
        },
        "arm contract",
    )
    expected_contract = {
        "schema_version": ARM_CONTRACT_SCHEMA,
        "status": "READY",
        "attempt_nonce": attempt["attempt_nonce"],
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "arm": arm,
        "backend": backend,
        "cohort": cohort,
        "processes": PROCESS_COUNT,
        "threads_per_process": THREADS_PER_PROCESS,
        "expected_concurrency": EXPECTED_CONCURRENCY,
        "panel_items": ITEMS_PER_PANEL,
        "sequence_index": sequence_index,
        "previous_arm_result_sha256": previous_arm_result_sha256,
    }
    for key, wanted in expected_contract.items():
        if contract[key] != wanted or type(contract[key]) is not type(wanted):
            raise ContractError("arm contract identity/topology mismatch")
    contract_created = strict_int(contract["created_at_unix"], "arm creation time", minimum=1)
    records: list[dict[str, Any]] = []
    selected = selection["cohorts"][cohort]
    worker_declarations: dict[str, dict[str, Any]] = {}
    for worker_id in range(PROCESS_COUNT):
        name = f"worker_{worker_id:02d}.jsonl"
        relative = f"{prefix}/{name}"
        records.extend(
            _parse_worker_jsonl(
                closure.payloads[relative],
                arm=arm,
                backend=backend,
                cohort=cohort,
                worker_id=worker_id,
                expected_selection=selected,
            )
        )
        declaration = closure.manifest["files"][relative]
        worker_declarations[name] = {
            "bytes": declaration["bytes"],
            "sha256": declaration["sha256"],
        }
    records.sort(key=lambda value: value["position"])
    if [record["position"] for record in records] != list(range(ITEMS_PER_PANEL)):
        raise ContractError("arm item position closure mismatch")
    result = exact_keys(
        _json_from_attested_bytes(
            closure.payloads[f"{prefix}/arm_result.json"], "arm result"
        ),
        {
            "schema_version",
            "status",
            "attempt_nonce",
            "selection_sha256",
            "arm",
            "backend",
            "cohort",
            "processes",
            "threads_per_process",
            "expected_concurrency",
            "panel_items",
            "started_at_unix",
            "completed_at_unix",
            "elapsed_ns",
            "lifecycle_elapsed_ns",
            "peak_inflight",
            "items",
            "succeeded",
            "failed",
            "total_bytes",
            "total_io_ns",
            "object_attempts",
            "object_retry_attempts",
            "sdk_retry_attempts",
            "version_id_present_items",
            "response_header_count",
            "body_close_count",
            "preheader_retry_attempts",
            "version_id_retry_attempts",
            "if_match_retry_attempts",
            "unguarded_postheader_retry_attempts",
            "worker_files",
            "sequence_index",
            "previous_arm_result_sha256",
        },
        "arm result",
    )
    succeeded = sum(record["outcome"] == "success" for record in records)
    aggregate = {
        "items": len(records),
        "succeeded": succeeded,
        "failed": len(records) - succeeded,
        "total_bytes": sum(record["byte_count"] for record in records),
        "total_io_ns": sum(record["io_ns"] for record in records),
        "object_attempts": sum(record["object_attempts"] for record in records),
        "object_retry_attempts": sum(
            max(0, record["object_attempts"] - 1) for record in records
        ),
        "sdk_retry_attempts": sum(record["sdk_retry_attempts"] for record in records),
        "version_id_present_items": sum(
            record["version_id_present"] for record in records
        ),
        "response_header_count": sum(record["response_header_count"] for record in records),
        "body_close_count": sum(record["body_close_count"] for record in records),
        "preheader_retry_attempts": sum(
            record["preheader_retry_attempts"] for record in records
        ),
        "version_id_retry_attempts": sum(
            record["version_id_retry_attempts"] for record in records
        ),
        "if_match_retry_attempts": sum(
            record["if_match_retry_attempts"] for record in records
        ),
        "unguarded_postheader_retry_attempts": sum(
            record["unguarded_postheader_retry_attempts"] for record in records
        ),
    }
    expected_result = {
        "schema_version": ARM_RESULT_SCHEMA,
        "status": "PASS" if aggregate["failed"] == 0 else "FAIL",
        "attempt_nonce": attempt["attempt_nonce"],
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "arm": arm,
        "backend": backend,
        "cohort": cohort,
        "processes": PROCESS_COUNT,
        "threads_per_process": THREADS_PER_PROCESS,
        "expected_concurrency": EXPECTED_CONCURRENCY,
        "panel_items": ITEMS_PER_PANEL,
        "sequence_index": sequence_index,
        "previous_arm_result_sha256": previous_arm_result_sha256,
        **aggregate,
        "worker_files": worker_declarations,
    }
    for key, wanted in expected_result.items():
        if result[key] != wanted or type(result[key]) is not type(wanted):
            raise ContractError("arm result independent aggregate mismatch")
    elapsed_ns = strict_int(result["elapsed_ns"], "arm elapsed ns", minimum=1)
    lifecycle_elapsed_ns = strict_int(
        result["lifecycle_elapsed_ns"], "arm lifecycle elapsed ns", minimum=1
    )
    started = strict_int(result["started_at_unix"], "arm start time", minimum=contract_created)
    completed = strict_int(result["completed_at_unix"], "arm completion time", minimum=started)
    if previous_completed_at_unix is not None and started < previous_completed_at_unix:
        raise ContractError("ABBA arm wall-clock order mismatch")
    if (
        max(record["io_ns"] for record in records) > elapsed_ns
        or elapsed_ns > lifecycle_elapsed_ns
        or lifecycle_elapsed_ns > (completed - started + 2) * 1_000_000_000
    ):
        raise ContractError("arm shared-I/O/lifecycle timing envelope mismatch")
    peak = strict_int(result["peak_inflight"], "arm peak inflight", minimum=1)
    if peak > EXPECTED_CONCURRENCY:
        raise ContractError("arm peak inflight exceeds frozen topology")
    io_seconds = [record["io_ns"] / 1_000_000_000.0 for record in records]
    return {
        "arm": arm,
        "backend": backend,
        "cohort": cohort,
        "records": records,
        "status": result["status"],
        "elapsed_s": elapsed_ns / 1_000_000_000.0,
        "lifecycle_elapsed_s": lifecycle_elapsed_ns / 1_000_000_000.0,
        "peak_inflight": peak,
        "completed_at_unix": completed,
        "artifact_sha256": closure.manifest["files"][f"{prefix}/arm_result.json"]["sha256"],
        "items_per_second": ITEMS_PER_PANEL * 1_000_000_000.0 / elapsed_ns,
        "mib_per_second": aggregate["total_bytes"] / (1 << 20) * 1_000_000_000.0 / elapsed_ns,
        "io_mean_s": statistics.fmean(io_seconds),
        "io_p50_s": percentile(io_seconds, 0.50),
        "io_p95_s": percentile(io_seconds, 0.95),
        "io_p99_s": percentile(io_seconds, 0.99),
        "io_max_s": max(io_seconds),
        **aggregate,
    }


def _performance_decision(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rates = {arm: float(value["items_per_second"]) for arm, value in arms.items()}
    speedups = {
        "x": arms["mount_x"]["elapsed_s"] / arms["s3_x"]["elapsed_s"],
        "y": arms["mount_y"]["elapsed_s"] / arms["s3_y"]["elapsed_s"],
    }
    drift = abs(rates["s3_x"] - rates["s3_y"]) / min(
        rates["s3_x"], rates["s3_y"]
    )
    s3_records = arms["s3_x"]["records"] + arms["s3_y"]["records"]
    retry_attempts = sum(
        max(0, record["object_attempts"] - 1) + record["sdk_retry_attempts"]
        for record in s3_records
    )
    retry_rate = retry_attempts / (2 * ITEMS_PER_PANEL)
    gates = {
        "all_arms_complete": all(value["status"] == "PASS" for value in arms.values()),
        "all_arms_peak_concurrency_16": all(
            value["peak_inflight"] == EXPECTED_CONCURRENCY for value in arms.values()
        ),
        "s3_x_absolute_rate": rates["s3_x"] >= MIN_S3_IMAGES_PER_SECOND,
        "s3_y_absolute_rate": rates["s3_y"] >= MIN_S3_IMAGES_PER_SECOND,
        "x_paired_speedup": speedups["x"] >= MIN_S3_SPEEDUP,
        "y_paired_speedup": speedups["y"] >= MIN_S3_SPEEDUP,
        "s3_x_p99": arms["s3_x"]["io_p99_s"] < MAX_S3_P99_SECONDS,
        "s3_y_p99": arms["s3_y"]["io_p99_s"] < MAX_S3_P99_SECONDS,
        "s3_retry_rate": retry_rate < MAX_S3_SDK_RETRY_RATE,
        "s3_panel_drift": drift <= MAX_S3_PANEL_DRIFT,
        "zero_unguarded_postheader_retries": all(
            arms[arm]["unguarded_postheader_retry_attempts"] == 0
            for arm in ("s3_x", "s3_y")
        ),
        "response_bodies_closed": all(
            arms[arm]["body_close_count"] == arms[arm]["response_header_count"]
            for arm in ("s3_x", "s3_y")
        ),
    }
    no_go_gate_names = {
        "all_arms_complete",
        "all_arms_peak_concurrency_16",
        "s3_x_absolute_rate",
        "s3_y_absolute_rate",
        "x_paired_speedup",
        "y_paired_speedup",
        "s3_x_p99",
        "s3_y_p99",
        "s3_retry_rate",
        "zero_unguarded_postheader_retries",
        "response_bodies_closed",
    }
    return {
        "gates": gates,
        "no_go_failures": sorted(name for name in no_go_gate_names if not gates[name]),
        "inconclusive_failures": [] if gates["s3_panel_drift"] else ["s3_panel_drift"],
        "metrics": {
            "items_per_second": rates,
            "paired_speedup": speedups,
            "s3_panel_drift_formula": "abs(rate_x-rate_y)/min(rate_x,rate_y)",
            "s3_panel_drift": drift,
            "s3_retry_attempts": retry_attempts,
            "s3_retry_rate_denominator": 2 * ITEMS_PER_PANEL,
            "s3_retry_rate": retry_rate,
            "s3_io_p99_s": {
                "x": arms["s3_x"]["io_p99_s"],
                "y": arms["s3_y"]["io_p99_s"],
            },
        },
    }


def analyze_capture(
    capture_root: Path | str,
    cache_path: Path | str,
) -> dict[str, Any]:
    cache_path = Path(cache_path)
    items, selection = validate_and_select_cache(cache_path)
    closure = _validate_capture_closure(
        Path(capture_root), forbidden_source_keys={item.src_key for item in items}
    )
    selection_value = _json_from_attested_bytes(
        closure.payloads["selection.json"], "capture selection"
    )
    if canonical_json_bytes(selection_value) != canonical_json_bytes(selection):
        raise ContractError("capture/independent selection mismatch")
    attempt = _validate_attempt_contract(
        closure.payloads["attempt_contract.json"],
        capture_root=closure.root,
        cache_path=cache_path,
        selection=selection,
    )
    if (
        closure.complete["attempt_nonce"] != attempt["attempt_nonce"]
        or closure.manifest["attempt_nonce"] != attempt["attempt_nonce"]
    ):
        raise ContractError("capture attempt identity does not close")
    preflight, warmup, semantic_issues = _validate_preflight(
        closure.payloads["preflight.json"], attempt=attempt, selection=selection
    )
    if closure.complete["status"] == "AUTH_UNAVAILABLE":
        if preflight["status"] != "AUTH_UNAVAILABLE":
            raise ContractError("completion/preflight authorization status mismatch")
        _validate_capture_input_attestation(attempt["input_attestation"], cache_path)
        return governed_result({
            "schema_version": ANALYSIS_SCHEMA,
            "status": "INCONCLUSIVE",
            "decision_reasons": ["auth_unavailable"],
            "selection_sha256": EXPECTED_SELECTION_SHA256,
            "credential_qualification": {
                "credentials_class": preflight["credential_summary"]["credentials_class"],
                "refreshable": preflight["credential_summary"]["refreshable"],
                "go_eligible": False,
            },
            "evidence": {
                "source_cache_sha256": SOURCE_CACHE_SHA256,
                "capture_manifest_sha256": closure.manifest_sha256,
                "capture_complete_sha256": closure.complete_sha256,
                "capture_artifacts": len(closure.payloads) + 2,
            },
        })
    if closure.complete["status"] == "SOURCE_UNAVAILABLE":
        if preflight["status"] != "SOURCE_UNAVAILABLE":
            raise ContractError("completion/preflight source status mismatch")
        _validate_capture_input_attestation(attempt["input_attestation"], cache_path)
        return governed_result({
            "schema_version": ANALYSIS_SCHEMA,
            "status": "NO_GO",
            "decision_reasons": ["source_unavailable"],
            "selection_sha256": EXPECTED_SELECTION_SHA256,
            "evidence": {
                "source_cache_sha256": SOURCE_CACHE_SHA256,
                "capture_manifest_sha256": closure.manifest_sha256,
                "capture_complete_sha256": closure.complete_sha256,
                "capture_artifacts": len(closure.payloads) + 2,
            },
        })
    if preflight["status"] != "PASS" or not warmup["mount"] or not warmup["s3"]:
        raise ContractError("complete capture lacks passing semantic preflight")
    arms: dict[str, dict[str, Any]] = {}
    previous_result_sha256: str | None = None
    previous_completed_at_unix: int | None = None
    for sequence_index, (arm, backend, cohort, directory) in enumerate(ARM_SPECS):
        arms[arm] = _validate_arm(
            closure,
            attempt=attempt,
            selection=selection,
            arm=arm,
            backend=backend,
            cohort=cohort,
            directory=directory,
            sequence_index=sequence_index,
            previous_arm_result_sha256=previous_result_sha256,
            previous_completed_at_unix=previous_completed_at_unix,
        )
        previous_result_sha256 = arms[arm]["artifact_sha256"]
        previous_completed_at_unix = arms[arm]["completed_at_unix"]
    try:
        semantic = validate_payload_pairs(
            {arm: value["records"] for arm, value in arms.items()}
        )
    except SemanticFailure:
        semantic = {"status": "FAIL"}
        semantic_issues.append("timed_payload_closure")
    performance = _performance_decision(arms)
    credentials = preflight["credential_summary"]
    credential_go = _credential_go_eligible(credentials)
    no_go = list(performance["no_go_failures"])
    no_go.extend(semantic_issues)
    inconclusive = list(performance["inconclusive_failures"])
    if not credential_go:
        inconclusive.append("credential_lifecycle")
    if no_go:
        status = "NO_GO"
        reasons = no_go
    elif inconclusive:
        status = "INCONCLUSIVE"
        reasons = sorted(inconclusive)
    else:
        status = "GO"
        reasons = []
    arm_summary = {
        arm: {
            key: value[key]
            for key in (
                "backend",
                "cohort",
                "status",
                "elapsed_s",
                "lifecycle_elapsed_s",
                "peak_inflight",
                "items_per_second",
                "mib_per_second",
                "io_mean_s",
                "io_p50_s",
                "io_p95_s",
                "io_p99_s",
                "io_max_s",
                "items",
                "succeeded",
                "failed",
                "total_bytes",
                "object_retry_attempts",
                "sdk_retry_attempts",
                "version_id_present_items",
                "preheader_retry_attempts",
                "version_id_retry_attempts",
                "if_match_retry_attempts",
                "unguarded_postheader_retry_attempts",
            )
        }
        for arm, value in arms.items()
    }
    _validate_capture_input_attestation(attempt["input_attestation"], cache_path)
    return governed_result({
        "schema_version": ANALYSIS_SCHEMA,
        "status": status,
        "decision_reasons": reasons,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "selection": {
            "source_items": SOURCE_CACHE_ITEMS,
            "warmup_items": COHORT_SIZES["warmup"],
            "x_items": COHORT_SIZES["x"],
            "y_items": COHORT_SIZES["y"],
            "unique_physical_shards": sum(COHORT_SIZES.values()),
        },
        "semantic_closure": semantic,
        "credential_qualification": {
            "credentials_class": credentials["credentials_class"],
            "refreshable": credentials["refreshable"],
            "go_eligible": credential_go,
        },
        "thresholds": {
            "s3_images_per_second_min": MIN_S3_IMAGES_PER_SECOND,
            "paired_speedup_min": MIN_S3_SPEEDUP,
            "s3_panel_drift_max": MAX_S3_PANEL_DRIFT,
            "s3_io_p99_seconds_max_exclusive": MAX_S3_P99_SECONDS,
            "s3_retry_rate_max_exclusive": MAX_S3_SDK_RETRY_RATE,
            "expected_concurrency": EXPECTED_CONCURRENCY,
        },
        "performance": performance,
        "arms": arm_summary,
        "evidence": {
            "source_cache_sha256": SOURCE_CACHE_SHA256,
            "capture_manifest_sha256": closure.manifest_sha256,
            "capture_complete_sha256": closure.complete_sha256,
            "capture_artifacts": len(closure.payloads) + 2,
        },
        "scope": (
            "CPU transport probe only; GO still requires W16 transport crossover and "
            "clean 2x8 GPU validation before production selection."
        ),
    })


def atomic_publish_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish immutable analysis JSON without any overwrite-capable operation."""
    parent = path.parent
    require_real_directory(parent, "analysis output parent")
    if path.exists() or path.is_symlink():
        raise ContractError("analysis output already exists")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    scan_for_leaks(payload)
    temporary = parent / f".{path.name}.tmp.{os.getpid()}.{os.urandom(16).hex()}"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o400,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ContractError("analysis output write did not progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.chmod(temporary, 0o444, follow_symlinks=False)
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _validate_analysis_input_attestation(path: Path) -> str:
    raw = _stable_unattested_read(path, "analysis input attestation")
    value = exact_keys(
        _json_from_attested_bytes(raw, "analysis input attestation"),
        {"schema_version", "files"},
        "analysis input attestation",
    )
    if value["schema_version"] != ANALYSIS_INPUT_SCHEMA:
        raise ContractError("analysis input attestation schema mismatch")
    repo = Path(__file__).resolve(strict=True).parents[2]
    expected = {
        "analyzer": repo / "scripts/offline_arb/analyze_gpic_mount_s3_abba_probe.py",
        "analyze_yaml": repo / "jobs/offline_arb_gpic_mount_s3_abba_analyze.yaml",
    }
    files = exact_keys(value["files"], set(expected), "analysis input files")
    for label, expected_path in expected.items():
        declaration = exact_keys(
            files[label], {"path", "bytes", "sha256"}, "analysis input declaration"
        )
        if declaration["path"] != str(expected_path):
            raise ContractError("analysis input path mismatch")
        require_regular_file(expected_path, "analysis live input")
        if expected_path.is_symlink() or expected_path.resolve(strict=True) != expected_path:
            raise ContractError("analysis live input is not canonical/real")
        verify_file_attestation(
            expected_path,
            strict_int(declaration["bytes"], "analysis input bytes", minimum=1),
            declaration["sha256"],
        )
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ContractError("analysis input attestation is writable")
    return hashlib.sha256(raw).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--analysis-input-attestation", required=True)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        analysis_input_sha = _validate_analysis_input_attestation(
            Path(args.analysis_input_attestation)
        )
        result = analyze_capture(args.capture_root, args.cache)
        if _validate_analysis_input_attestation(
            Path(args.analysis_input_attestation)
        ) != analysis_input_sha:
            raise ContractError("analysis input attestation changed during analysis")
        result["analysis_input_attestation_sha256"] = analysis_input_sha
    except SemanticFailure:
        result = {
            "schema_version": ANALYSIS_SCHEMA,
            "status": "NO_GO",
            "decision_reasons": ["semantic_gate"],
        }
    except ContractError:
        result = {
            "schema_version": ANALYSIS_SCHEMA,
            "status": "INVALID",
            "decision_reasons": ["capture_contract_invalid"],
        }
    except BaseException:
        result = {
            "schema_version": ANALYSIS_SCHEMA,
            "status": "INVALID",
            "decision_reasons": ["analyzer_internal_failure"],
        }
    result = governed_result(result)
    status_to_exit = {"GO": 0, "NO_GO": 3, "INCONCLUSIVE": 4, "INVALID": 2}
    status = result.get("status")
    exit_code = status_to_exit.get(status, 2)
    if args.output:
        try:
            atomic_publish_json(Path(args.output), result)
        except BaseException:
            print(
                json.dumps(
                    governed_result({
                        "schema_version": ANALYSIS_SCHEMA,
                        "status": "INVALID",
                        "decision_reasons": ["analysis_publication_failure"],
                    }),
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2
    print(
        json.dumps(
            governed_result({
                "schema_version": ANALYSIS_SCHEMA,
                "status": status if status in status_to_exit else "INVALID",
                "decision_reasons": result.get("decision_reasons", []),
            }),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        file=sys.stdout if exit_code == 0 else sys.stderr,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
