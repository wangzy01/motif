#!/usr/bin/env python3
"""Run a one-object, four-cell GPIC S3 authorization diagnostic.

The published result contains only fixed cell labels, allowlisted statuses,
and payload-equivalence booleans.  Source locations, response details,
credential material, and exception text remain in memory only.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import boto3
from botocore import UNSIGNED
from botocore.config import Config


SCHEMA_VERSION = "offline_arb_gpic_s3_auth_matrix_v1"
SOURCE_CACHE = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_arb_phase2b_20260710T052630Z_"
    "evan-mot-crop-offlinearb-p2c-gpictune-gtnv-0_26335/gpic_items_cache.json"
)
SOURCE_CACHE_BYTES = 6_220_043
SOURCE_CACHE_SHA256 = "3c45ba6e3898b546dbc191e6b708e0bffc7047fd059708b7af8cc0e190cfa298"
SOURCE_CACHE_ITEMS = 20_000
FROZEN_SOURCE_INDEX = 3_065
FROZEN_OBJECT_ID = "695f731cad379747cd1a4488a5ee6863ec277294c80124e84fe6f05e6ca75bab"

SOURCE_ROOT = "/s3-mewtant-raw-datasets/stanford-vision-lab/gpic_umziped"
MOUNT_PREFIX = "/s3-mewtant-raw-datasets/"
S3_KEY_PREFIX = "stanford-vision-lab/gpic_umziped/images/train/"
BUCKET_CANDIDATE_A = "mewtant-raw-datasets"
BUCKET_CANDIDATE_B = "s3-mewtant-raw-datasets"

SELECTION_DOMAIN = b"offline_arb_gpic_s3_mount_abba_selection_v1\0"
SOURCE_FIELDS = ("uid", "src_root", "src_shard", "src_key", "ext")
SOURCE_KEY_RE = re.compile(r"[0-9a-f]{64}")
SHARD_RE = re.compile(r"[0-9]{5}")
SAFE_CELL_RE = re.compile(r"[a-z0-9_]+")
MAX_OBJECT_BYTES = 256 * 1024 * 1024
READ_CHUNK_BYTES = 8 * 1024 * 1024

CELL_SPECS = (
    ("signed_candidate_a", "signed", BUCKET_CANDIDATE_A),
    ("unsigned_candidate_a", "unsigned", BUCKET_CANDIDATE_A),
    ("signed_candidate_b", "signed", BUCKET_CANDIDATE_B),
    ("unsigned_candidate_b", "unsigned", BUCKET_CANDIDATE_B),
)
ALLOWED_STATUSES = {
    "match",
    "payload_mismatch",
    "access_denied",
    "credentials_unavailable",
    "signature_error",
    "missing",
    "redirect",
    "throttled",
    "timeout",
    "network_error",
    "response_invalid",
    "unexpected",
}


class ContractError(RuntimeError):
    """A local invariant failed without carrying source detail."""


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
        raise ContractError("canonical_json_invalid") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb", buffering=0) as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise ContractError("source_cache_read_failed") from exc
    return digest.hexdigest()


def _validate_frozen_item(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != set(SOURCE_FIELDS):
        raise ContractError("frozen_item_schema_invalid")
    if any(type(raw[field]) is not str for field in SOURCE_FIELDS):
        raise ContractError("frozen_item_type_invalid")
    item = {field: raw[field] for field in SOURCE_FIELDS}
    if (
        item["src_root"] != SOURCE_ROOT
        or SOURCE_KEY_RE.fullmatch(item["src_key"]) is None
        or item["uid"] != f"gpic:{item['src_key']}"
        or SHARD_RE.fullmatch(item["src_shard"]) is None
        or int(item["src_shard"]) >= 8_000
        or item["ext"] not in {".jpg", ".png"}
    ):
        raise ContractError("frozen_item_contract_invalid")
    if any(
        token in item[field]
        for field in ("src_key", "src_shard")
        for token in ("/", "\\", "\x00", "..")
    ):
        raise ContractError("frozen_item_component_invalid")
    canonical = canonical_json_bytes({field: item[field] for field in SOURCE_FIELDS})
    object_id = hashlib.sha256(
        SELECTION_DOMAIN + b"item\0" + canonical
    ).hexdigest()
    if not hmac.compare_digest(object_id, FROZEN_OBJECT_ID):
        raise ContractError("frozen_object_identity_invalid")
    return item


def load_frozen_location(cache_path: Path) -> tuple[str, str, set[bytes]]:
    path = cache_path.absolute()
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ContractError("source_cache_identity_invalid") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size != SOURCE_CACHE_BYTES
        or sha256_file(path) != SOURCE_CACHE_SHA256
    ):
        raise ContractError("source_cache_identity_invalid")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("source_cache_json_invalid") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"items", "resolve_errors"}
        or document.get("resolve_errors") != 0
        or type(document.get("resolve_errors")) is not int
        or not isinstance(document.get("items"), list)
        or len(document["items"]) != SOURCE_CACHE_ITEMS
    ):
        raise ContractError("source_cache_schema_invalid")
    item = _validate_frozen_item(document["items"][FROZEN_SOURCE_INDEX])
    key = (
        f"{S3_KEY_PREFIX}{item['src_shard']}/"
        f"{item['src_key']}{item['ext']}"
    )
    if (
        not key.startswith(S3_KEY_PREFIX)
        or "//" in key
        or ".." in key
        or "\x00" in key
    ):
        raise ContractError("derived_location_invalid")
    mount_path = f"{MOUNT_PREFIX}{key}"
    if not mount_path.startswith(MOUNT_PREFIX):
        raise ContractError("derived_location_invalid")
    forbidden = {
        item["src_key"].encode("ascii"),
        item["uid"].encode("ascii"),
        key.encode("ascii"),
        mount_path.encode("ascii"),
        SOURCE_ROOT.encode("ascii"),
        MOUNT_PREFIX.encode("ascii"),
        S3_KEY_PREFIX.encode("ascii"),
        BUCKET_CANDIDATE_A.encode("ascii"),
        BUCKET_CANDIDATE_B.encode("ascii"),
    }
    return mount_path, key, forbidden


def _fd_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_mount_baseline(path: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractError("mount_open_failed") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > MAX_OBJECT_BYTES
        ):
            raise ContractError("mount_identity_invalid")
        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                block = os.read(descriptor, READ_CHUNK_BYTES)
            except OSError as exc:
                raise ContractError("mount_read_failed") from exc
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > MAX_OBJECT_BYTES:
                raise ContractError("mount_size_invalid")
        after = os.fstat(descriptor)
        if _fd_identity(before) != _fd_identity(after) or total != before.st_size:
            raise ContractError("mount_changed_during_read")
        payload = b"".join(chunks)
        if not payload:
            raise ContractError("mount_empty")
        return payload
    finally:
        os.close(descriptor)


def build_s3_config(auth_mode: str) -> Config:
    if auth_mode not in {"signed", "unsigned"}:
        raise ContractError("auth_mode_invalid")
    signature_version: Any = "s3v4" if auth_mode == "signed" else UNSIGNED
    return Config(
        region_name="us-west-2",
        signature_version=signature_version,
        retries={"mode": "standard", "total_max_attempts": 1},
        max_pool_connections=1,
        tcp_keepalive=True,
        connect_timeout=10.0,
        read_timeout=30.0,
        s3={"addressing_style": "virtual"},
    )


def make_s3_client(auth_mode: str) -> Any:
    session = boto3.Session(region_name="us-west-2")
    return session.client("s3", config=build_s3_config(auth_mode))


def classify_exception(exc: BaseException) -> str:
    name = type(exc).__name__
    if name in {
        "NoCredentialsError",
        "PartialCredentialsError",
        "CredentialRetrievalError",
        "TokenRetrievalError",
    }:
        return "credentials_unavailable"
    if name in {"ConnectTimeoutError", "ReadTimeoutError", "TimeoutError"}:
        return "timeout"
    if name in {
        "EndpointConnectionError",
        "ConnectionClosedError",
        "HTTPClientError",
        "ProxyConnectionError",
        "ConnectionError",
    }:
        return "network_error"
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
    code = error.get("Code") if type(error.get("Code")) is str else ""
    status = metadata.get("HTTPStatusCode")
    status = status if type(status) is int else 0
    if status in {301, 307} or code in {
        "PermanentRedirect",
        "TemporaryRedirect",
        "AuthorizationHeaderMalformed",
    }:
        return "redirect"
    if code in {
        "InvalidAccessKeyId",
        "ExpiredToken",
        "SignatureDoesNotMatch",
        "InvalidToken",
    }:
        return "signature_error"
    if status in {401, 403} or code in {
        "AccessDenied",
        "AllAccessDisabled",
        "KMSAccessDeniedException",
    }:
        return "access_denied"
    if status == 404 or code in {"NoSuchBucket", "NoSuchKey", "NotFound"}:
        return "missing"
    if status == 429 or code in {"SlowDown", "Throttling", "TooManyRequests"}:
        return "throttled"
    if status >= 500 or code in {
        "InternalError",
        "RequestTimeout",
        "ServiceUnavailable",
    }:
        return "network_error"
    return "unexpected"


def _blank_cell(cell_label: str, status_value: str) -> dict[str, Any]:
    if (
        SAFE_CELL_RE.fullmatch(cell_label) is None
        or status_value not in ALLOWED_STATUSES
    ):
        raise ContractError("cell_result_invalid")
    return {
        "cell": cell_label,
        "status": status_value,
        "bytes_match": None,
        "sha256_match": None,
        "content_length_match": None,
    }


def probe_cell(
    *,
    cell_label: str,
    client: Any,
    bucket: str,
    key: str,
    baseline: bytes,
) -> dict[str, Any]:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except BaseException as exc:
        return _blank_cell(cell_label, classify_exception(exc))
    if not isinstance(response, Mapping):
        return _blank_cell(cell_label, "response_invalid")
    metadata = response.get("ResponseMetadata", {})
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    content_length = response.get("ContentLength")
    body = response.get("Body")
    if (
        status != 200
        or type(content_length) is not int
        or content_length <= 0
        or content_length > MAX_OBJECT_BYTES
        or not callable(getattr(body, "read", None))
        or not callable(getattr(body, "close", None))
    ):
        if callable(getattr(body, "close", None)):
            try:
                body.close()
            except BaseException:
                pass
        return _blank_cell(cell_label, "response_invalid")
    chunks: list[bytes] = []
    total = 0
    read_failed = False
    close_failed = False
    try:
        while True:
            block = body.read(READ_CHUNK_BYTES)
            if type(block) is not bytes:
                read_failed = True
                break
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > MAX_OBJECT_BYTES:
                read_failed = True
                break
    except BaseException:
        read_failed = True
    finally:
        try:
            body.close()
        except BaseException:
            close_failed = True
    if read_failed or close_failed:
        return _blank_cell(cell_label, "network_error" if read_failed else "response_invalid")
    payload = b"".join(chunks)
    content_length_match = total == content_length
    bytes_match = total == len(baseline)
    sha256_match = hmac.compare_digest(
        hashlib.sha256(payload).digest(), hashlib.sha256(baseline).digest()
    )
    result = {
        "cell": cell_label,
        "status": "match"
        if content_length_match and bytes_match and sha256_match
        else "payload_mismatch",
        "bytes_match": bytes_match,
        "sha256_match": sha256_match,
        "content_length_match": content_length_match,
    }
    return result


def execute_matrix(
    *,
    key: str,
    baseline: bytes,
    client_factory: Callable[[str], Any] = make_s3_client,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for cell_label, auth_mode, bucket in CELL_SPECS:
        client = None
        try:
            client = client_factory(auth_mode)
            result = probe_cell(
                cell_label=cell_label,
                client=client,
                bucket=bucket,
                key=key,
                baseline=baseline,
            )
        except BaseException as exc:
            result = _blank_cell(cell_label, classify_exception(exc))
        finally:
            if client is not None and callable(getattr(client, "close", None)):
                try:
                    client.close()
                except BaseException:
                    pass
        results.append(result)
    return results


def validate_public_result(result: Any, forbidden: set[bytes]) -> bytes:
    if (
        not isinstance(result, dict)
        or set(result) != {"schema_version", "status", "cells"}
        or result.get("schema_version") != SCHEMA_VERSION
        or result.get("status") != "COMPLETE"
        or not isinstance(result.get("cells"), list)
        or len(result["cells"]) != len(CELL_SPECS)
    ):
        raise ContractError("public_result_schema_invalid")
    expected_labels = [spec[0] for spec in CELL_SPECS]
    for index, cell in enumerate(result["cells"]):
        if (
            not isinstance(cell, dict)
            or set(cell)
            != {
                "cell",
                "status",
                "bytes_match",
                "sha256_match",
                "content_length_match",
            }
            or cell.get("cell") != expected_labels[index]
            or cell.get("status") not in ALLOWED_STATUSES
            or any(
                cell.get(field) not in {True, False, None}
                for field in (
                    "bytes_match",
                    "sha256_match",
                    "content_length_match",
                )
            )
        ):
            raise ContractError("public_cell_schema_invalid")
        if cell["status"] == "match" and not all(
            cell[field] is True
            for field in (
                "bytes_match",
                "sha256_match",
                "content_length_match",
            )
        ):
            raise ContractError("public_match_contract_invalid")
    raw = canonical_json_bytes(result) + b"\n"
    generic_forbidden = {
        b"aws_access_key_id",
        b"aws_secret_access_key",
        b"aws_session_token",
        b"authorization",
        b"x-amz-credential",
        b"arn:aws:",
        b"s3://",
        b"https://",
    }
    lowered = raw.lower()
    if any(token and token.lower() in lowered for token in forbidden | generic_forbidden):
        raise ContractError("public_result_leak_detected")
    return raw


def atomic_publish(path: Path, raw: bytes) -> None:
    destination = path.absolute()
    parent = destination.parent
    try:
        parent_metadata = parent.lstat()
    except OSError as exc:
        raise ContractError("output_parent_invalid") from exc
    if not stat.S_ISDIR(parent_metadata.st_mode) or parent.is_symlink():
        raise ContractError("output_parent_invalid")
    if destination.exists() or destination.is_symlink():
        raise ContractError("output_exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(destination, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ContractError("output_write_failed")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def run(args: argparse.Namespace) -> None:
    if args.cache.absolute() != SOURCE_CACHE:
        raise ContractError("source_cache_path_invalid")
    mount_path, key, forbidden = load_frozen_location(args.cache)
    baseline = read_mount_baseline(mount_path)
    cells = execute_matrix(key=key, baseline=baseline)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "cells": cells,
    }
    raw = validate_public_result(result, forbidden)
    atomic_publish(args.output, raw)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitized one-object GPIC S3 auth matrix")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(sys.argv[1:] if argv is None else argv))
    except BaseException as exc:
        category = "contract_error" if isinstance(exc, ContractError) else "unexpected"
        print(f"GPIC_S3_AUTH_MATRIX_FATAL category={category}", flush=True)
        return 2
    print("GPIC_S3_AUTH_MATRIX_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
