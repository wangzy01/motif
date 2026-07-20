#!/usr/bin/env python3
"""Static and synthetic tests for the GPIC mount/S3 ABBA probe.

This test deliberately performs no network I/O.  The S3 reader is exercised
with a scripted in-memory client and StreamingBody replacement.  The only
production-sized input read is the already-frozen 20,000-item GPIC cache.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

from scripts.offline_arb import analyze_gpic_mount_s3_abba_probe as analyzer
from scripts.offline_arb import run_gpic_mount_s3_abba_probe as runner


SOURCE_CACHE = Path(
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_arb_phase2b_20260710T052630Z_"
    "evan-mot-crop-offlinearb-p2c-gpictune-gtnv-0_26335/"
    "gpic_items_cache.json"
)
SOURCE_CACHE_BYTES = 6_220_043
SOURCE_CACHE_SHA256 = "3c45ba6e3898b546dbc191e6b708e0bffc7047fd059708b7af8cc0e190cfa298"
EXPECTED_SELECTION_SHA256 = "e358061beee8e23f0b154a93ecebd5b4023bb4e1c57492a2c92a007a82cf7a0d"
EXPECTED_COHORT_SIZES = {"warmup": 64, "x": 2048, "y": 2048}
EXPECTED_ARMS = (
    ("mount_x", "mount", "x", "00_mount_x"),
    ("s3_x", "s3", "x", "01_s3_x"),
    ("s3_y", "s3", "y", "02_s3_y"),
    ("mount_y", "mount", "y", "03_mount_y"),
)
ITEM_FIELDS = {"uid", "src_root", "src_shard", "src_key", "ext"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_failure(callback: Callable[[], Any], message: str) -> BaseException:
    try:
        callback()
    except BaseException as exc:  # the negative contract intentionally spans exception classes
        return exc
    raise AssertionError(message)


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite token: {token}")
        ),
    )
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def source_item_mapping(item: Any) -> dict[str, str]:
    value = {field: getattr(item, field) for field in ITEM_FIELDS}
    require(all(type(field_value) is str for field_value in value.values()), "SourceItem field type drift")
    return value


def call_validate_cache_item(item: Mapping[str, Any], source_index: int = 0) -> Any:
    """Bind the intentionally small public runner validation interface."""

    function = getattr(runner, "validate_cache_item", None)
    require(callable(function), "runner.validate_cache_item is missing")
    return function(dict(item), source_index=source_index)


def call_cache_item_location(item: Mapping[str, Any]) -> Mapping[str, Any]:
    function = getattr(runner, "cache_item_location", None)
    require(callable(function), "runner.cache_item_location is missing")
    value = function(dict(item))
    require(isinstance(value, Mapping), "cache_item_location must return a mapping")
    return value


def call_s3_get_object_result(
    client: Any,
    *,
    bucket: str,
    key: str,
    chunk_bytes: int = 3,
    max_attempts: int = 3,
) -> Any:
    function = getattr(runner, "s3_get_object_bytes", None)
    require(callable(function), "runner.s3_get_object_bytes is missing")
    result = function(
        client,
        bucket=bucket,
        key=key,
        chunk_bytes=chunk_bytes,
        max_attempts=max_attempts,
    )
    require(isinstance(getattr(result, "payload", None), bytes), "S3ReadResult.payload must be bytes")
    require(getattr(result, "byte_count", None) == len(result.payload), "S3ReadResult byte_count drift")
    return result


def call_s3_get_object_bytes(
    client: Any,
    *,
    bucket: str,
    key: str,
    chunk_bytes: int = 3,
    max_attempts: int = 3,
) -> bytes:
    return call_s3_get_object_result(
        client,
        bucket=bucket,
        key=key,
        chunk_bytes=chunk_bytes,
        max_attempts=max_attempts,
    ).payload


def call_atomic_publish_json(path: Path, value: Mapping[str, Any]) -> None:
    function = getattr(runner, "atomic_publish_json", None)
    require(callable(function), "runner.atomic_publish_json is missing")
    function(path, dict(value))


def call_validate_concurrency(processes: int, threads: int, peak: int) -> Any:
    function = getattr(runner, "validate_concurrency_contract", None)
    require(callable(function), "runner.validate_concurrency_contract is missing")
    return function(
        processes=processes,
        threads_per_process=threads,
        observed_peak=peak,
    )


class MockBody:
    """Small StreamingBody stand-in supporting chunked reads and failures."""

    def __init__(
        self,
        payload: bytes,
        *,
        fail_after_reads: int | None = None,
        failure: BaseException | None = None,
    ) -> None:
        self._payload = payload
        self._offset = 0
        self._reads = 0
        self._fail_after_reads = fail_after_reads
        self._failure = failure or OSError("synthetic body read failure")
        self.close_count = 0

    @property
    def closed(self) -> bool:
        return self.close_count > 0

    def read(self, amount: int = -1) -> bytes:
        require(not self.closed, "read after close")
        if self._fail_after_reads is not None and self._reads >= self._fail_after_reads:
            raise self._failure
        self._reads += 1
        if self._offset >= len(self._payload):
            return b""
        if amount is None or amount < 0:
            stop = len(self._payload)
        else:
            stop = min(len(self._payload), self._offset + amount)
        block = self._payload[self._offset : stop]
        self._offset = stop
        return block

    def close(self) -> None:
        self.close_count += 1


class MockClientError(RuntimeError):
    """Botocore-shaped exception without importing or contacting botocore."""

    def __init__(
        self,
        code: str,
        message: str,
        status: int,
        *,
        sdk_retries: int = 0,
    ) -> None:
        super().__init__(message)
        self.response = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status, "RetryAttempts": sdk_retries},
        }


class NoCredentialsError(RuntimeError):
    pass


class EndpointConnectionError(RuntimeError):
    pass


class ScriptedS3Client:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def get_object(self, **kwargs: Any) -> Mapping[str, Any]:
        self.calls.append(dict(kwargs))
        if not self._script:
            raise AssertionError("unexpected GetObject call")
        action = self._script.pop(0)
        if isinstance(action, BaseException):
            raise action
        require(isinstance(action, Mapping), "script action is not a response")
        return action


def response(
    payload: bytes,
    *,
    content_length: int | None = None,
    version_id: str | None = None,
    etag: str | None = None,
    sdk_retries: int = 0,
    status: int = 200,
    fail_after_reads: int | None = None,
    failure: BaseException | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "Body": MockBody(
            payload,
            fail_after_reads=fail_after_reads,
            failure=failure,
        ),
        "ContentLength": len(payload) if content_length is None else content_length,
        "ResponseMetadata": {"HTTPStatusCode": status, "RetryAttempts": sdk_retries},
    }
    if version_id is not None:
        result["VersionId"] = version_id
    if etag is not None:
        result["ETag"] = etag
    return result


def sensitive_canaries() -> dict[str, str]:
    # Assemble these at runtime so source scanners do not mistake test literals
    # for a real credential or signed request.
    key = "deadbeef" * 8
    bucket = "private" + "-canary-bucket"
    return {
        "key": key,
        "uri": "s3" + "://" + bucket + "/" + key,
        "access": "A" + "KIA" + "0123456789ABCDEF",
        "secret": "synthetic" + "-secret-access-key-canary",
        "session": "synthetic" + "-session-token-canary",
        "version": "synthetic" + "-version-id-canary",
        "etag": '"synthetic' + '-etag-canary"',
    }


def assert_exception_is_safe(exc: BaseException, canaries: Mapping[str, str]) -> None:
    rendered = str(exc)
    for label, value in canaries.items():
        require(value not in rendered, f"unsafe {label} leaked through exception")
    require("Traceback" not in rendered, "traceback leaked through safe exception")


def test_frozen_cache_and_selection() -> tuple[list[Any], dict[str, Any]]:
    require(SOURCE_CACHE.is_file() and not SOURCE_CACHE.is_symlink(), "source cache missing/symlink")
    require(SOURCE_CACHE.stat().st_size == SOURCE_CACHE_BYTES, "source cache size drift")
    require(sha256_file(SOURCE_CACHE) == SOURCE_CACHE_SHA256, "source cache SHA drift")
    require(analyzer.SOURCE_CACHE_SHA256 == SOURCE_CACHE_SHA256, "analyzer cache SHA drift")
    require(analyzer.SOURCE_CACHE_BYTES == SOURCE_CACHE_BYTES, "analyzer cache size drift")
    require(
        analyzer.EXPECTED_SELECTION_SHA256 == EXPECTED_SELECTION_SHA256,
        "analyzer expected selection SHA drift",
    )
    require(dict(analyzer.COHORT_SIZES) == EXPECTED_COHORT_SIZES, "cohort sizes drift")

    items, selection = analyzer.validate_and_select_cache(
        SOURCE_CACHE,
        expected_sha256=SOURCE_CACHE_SHA256,
    )
    require(isinstance(items, list) and len(items) == 20_000, "cache item count drift")
    require(isinstance(selection, dict), "selection is not an object")
    require(
        set(selection) == {
            "schema_version",
            "source_cache_sha256",
            "cohorts",
            "selection_sha256",
        },
        "selection schema drift",
    )
    require(selection["schema_version"] == analyzer.SELECTION_SCHEMA, "selection schema drift")
    require(selection["source_cache_sha256"] == SOURCE_CACHE_SHA256, "selection cache binding drift")
    selection_core = {key: selection[key] for key in ("schema_version", "source_cache_sha256", "cohorts")}
    require(selection["selection_sha256"] == EXPECTED_SELECTION_SHA256, "selection digest field drift")
    require(analyzer.canonical_sha256(selection_core) == EXPECTED_SELECTION_SHA256, "deterministic selection digest drift")

    all_indices: list[int] = []
    all_object_ids: list[str] = []
    all_shards: list[str] = []
    cohorts = selection["cohorts"]
    require(isinstance(cohorts, dict) and set(cohorts) == set(EXPECTED_COHORT_SIZES), "cohort map drift")
    for cohort, expected_size in EXPECTED_COHORT_SIZES.items():
        rows = cohorts[cohort]
        require(len(rows) == expected_size, f"{cohort} size drift")
        for row in rows:
            require(
                set(row) == {"source_index", "object_id", "ext_class", "shard_band"},
                f"{cohort} selection row schema drift",
            )
            index = row["source_index"]
            require(type(index) is int and 0 <= index < len(items), "invalid source_index")
            item = items[index]
            require(set(source_item_mapping(item)) == ITEM_FIELDS, "validated cache item schema drift")
            all_indices.append(index)
            all_object_ids.append(row["object_id"])
            all_shards.append(item.src_shard)
    require(len(all_indices) == 4160, "selected object count drift")
    require(len(set(all_indices)) == 4160, "cohort object overlap")
    require(len(set(all_object_ids)) == 4160, "cohort object_id overlap")
    require(len(set(all_shards)) == 4160, "physical shard overlap")
    require(
        tuple(tuple(value) for value in analyzer.ARM_SPECS) == EXPECTED_ARMS,
        "ABBA arm contract drift",
    )
    return items, selection


def test_cache_schema_and_path_mapping(items: list[Any]) -> None:
    valid = source_item_mapping(items[0])
    normalized = call_validate_cache_item(valid, source_index=17)
    require(isinstance(normalized, Mapping), "validated cache item is not a mapping")
    location = call_cache_item_location(valid)
    require(location.get("bucket") == "mewtant-raw-datasets", "bucket mapping drift")
    expected_key = (
        "stanford-vision-lab/gpic_umziped/images/train/"
        f"{valid['src_shard']}/{valid['src_key']}{valid['ext']}"
    )
    require(location.get("key") == expected_key, "S3 key mapping drift")
    require(
        location.get("mount_path")
        == f"/s3-mewtant-raw-datasets/{expected_key}",
        "mount path mapping drift",
    )

    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("missing field", lambda value: value.pop("ext")),
        ("extra field", lambda value: value.__setitem__("unexpected", "x")),
        ("non-string field", lambda value: value.__setitem__("src_shard", 1)),
        ("uid/key mismatch", lambda value: value.__setitem__("uid", "gpic:" + "0" * 64)),
        ("uppercase key", lambda value: value.__setitem__("src_key", value["src_key"].upper())),
        ("wrong root", lambda value: value.__setitem__("src_root", "/s3-other-bucket/root")),
        ("shard overflow", lambda value: value.__setitem__("src_shard", "08000")),
        ("unknown extension", lambda value: value.__setitem__("ext", ".gif")),
        ("shard traversal", lambda value: value.__setitem__("src_shard", "../01")),
        ("key traversal", lambda value: value.__setitem__("src_key", "../" + "a" * 61)),
        ("root traversal", lambda value: value.__setitem__("src_root", value["src_root"] + "/../escape")),
        ("NUL", lambda value: value.__setitem__("src_shard", "0000\x00")),
    ]
    for label, mutate in mutations:
        candidate: dict[str, Any] = dict(valid)
        mutate(candidate)
        expect_failure(
            lambda candidate=candidate: call_validate_cache_item(candidate, source_index=1),
            f"invalid cache item accepted: {label}",
        )
        expect_failure(
            lambda candidate=candidate: call_cache_item_location(candidate),
            f"unsafe cache path mapped: {label}",
        )


def test_topology_contract() -> None:
    call_validate_concurrency(8, 2, 16)
    for processes, threads, peak in ((7, 2, 14), (8, 1, 8), (8, 2, 15), (8, 2, 17), (16, 1, 16)):
        expect_failure(
            lambda p=processes, t=threads, k=peak: call_validate_concurrency(p, t, k),
            f"invalid topology accepted: {processes}x{threads} peak={peak}",
        )


class FakeRefreshableCredentials:
    method = "assume-role-with-web-identity"
    _refresh_using = staticmethod(lambda: None)

    @property
    def access_key(self) -> str:
        raise AssertionError("sensitive credential property was accessed")

    @property
    def secret_key(self) -> str:
        raise AssertionError("sensitive credential property was accessed")

    @property
    def token(self) -> str:
        raise AssertionError("sensitive credential property was accessed")

    @property
    def expiry_time(self) -> str:
        raise AssertionError("credential expiry was accessed")

    def get_frozen_credentials(self) -> Any:
        raise AssertionError("refreshable credentials were frozen")


class FakeSession:
    def __init__(self) -> None:
        self.credentials = FakeRefreshableCredentials()
        self.calls = 0

    def get_credentials(self) -> FakeRefreshableCredentials:
        self.calls += 1
        return self.credentials


class FakeNonrefreshableTemporaryCredentials:
    method = "assume-role-with-web-identity"

    @property
    def token(self) -> str:
        raise AssertionError("sensitive credential property was accessed")


class FakeNonrefreshableSession:
    def get_credentials(self) -> FakeNonrefreshableTemporaryCredentials:
        return FakeNonrefreshableTemporaryCredentials()


class FakeUnknownRefreshableCredentials:
    method = "untrusted-provider-name"
    _refresh_using = staticmethod(lambda: None)


class FakeUnknownSession:
    def get_credentials(self) -> FakeUnknownRefreshableCredentials:
        return FakeUnknownRefreshableCredentials()


def test_client_config_credentials_and_error_classes() -> None:
    config_function = getattr(runner, "build_botocore_config", None)
    require(callable(config_function), "runner.build_botocore_config is missing")
    config = config_function()
    require(config.max_pool_connections == 64, "botocore connection pool drift")
    require(config.tcp_keepalive is True, "botocore TCP keepalive disabled")
    require(config.retries.get("mode") == "standard", "botocore retry mode drift")
    require(
        config.retries.get("total_max_attempts") == 3,
        "botocore retry budget must use total_max_attempts=3",
    )
    require("max_attempts" not in config.retries, "ambiguous botocore max_attempts was used")
    require(config.connect_timeout > 0 and config.read_timeout > 0, "botocore timeouts must be positive")

    session = FakeSession()
    summary_function = getattr(runner, "credential_summary", None)
    require(callable(summary_function), "runner.credential_summary is missing")
    summary = summary_function(session)
    require(
        set(summary) == {"method", "credentials_class", "refreshable", "has_session_token"},
        "credential summary schema drift",
    )
    require(summary["method"] == "assume-role-with-web-identity", "credential method drift")
    require(summary["credentials_class"] == "refreshable_temporary", "credential class drift")
    require(summary["refreshable"] is True, "refreshable credentials were not recognized")
    require(summary["has_session_token"] is True, "web identity session-token shape not recognized")
    require(session.calls == 1, "credential provider was resolved more than once")
    encoded_summary = analyzer.canonical_json_bytes(summary).lower()
    for forbidden in (b"token-canary", b"arn:", b"access_key", b"secret_key", b"expiry"):
        require(forbidden not in encoded_summary, "credential material leaked into summary")

    nonrefreshable = summary_function(FakeNonrefreshableSession())
    require(
        nonrefreshable
        == {
            "method": "assume-role-with-web-identity",
            "credentials_class": "nonrefreshable_temporary",
            "refreshable": False,
            "has_session_token": True,
        },
        "temporary method without a refresh capability was mislabeled refreshable",
    )
    require(
        summary_function(FakeUnknownSession())
        == {
            "method": "unavailable",
            "credentials_class": "unavailable",
            "refreshable": False,
            "has_session_token": False,
        },
        "unknown credential provider leaked capability/class detail",
    )

    classify = getattr(runner, "classify_s3_error", None)
    require(callable(classify), "runner.classify_s3_error is missing")
    cases = (
        ("AccessDenied", 403, "auth"),
        ("NoSuchKey", 404, "missing"),
        ("TooManyRequests", 429, "throttle"),
        ("SlowDown", 503, "throttle"),
        ("InternalError", 500, "transient"),
        ("ServiceUnavailable", 503, "transient"),
        ("PreconditionFailed", 412, "source_changed"),
    )
    for code, status, expected in cases:
        observed = classify(MockClientError(code, "synthetic unsafe original", status))
        require(observed == expected, f"S3 error classification drift: {code}/{status}")
    require(
        classify(NoCredentialsError("synthetic credential canary")) == "auth",
        "NoCredentialsError classification drift",
    )
    require(
        classify(EndpointConnectionError("synthetic endpoint canary")) == "transient",
        "EndpointConnectionError classification drift",
    )

    retry_function = getattr(runner, "sdk_retry_attempts", None)
    require(callable(retry_function), "runner.sdk_retry_attempts is missing")
    require(
        retry_function({"ResponseMetadata": {"RetryAttempts": 0}}) == 0,
        "zero SDK retry count drift",
    )
    require(
        retry_function({"ResponseMetadata": {"RetryAttempts": 2}}) == 2,
        "positive SDK retry count drift",
    )
    for invalid in (-1, True, "2", 1.5):
        expect_failure(
            lambda invalid=invalid: retry_function(
                {"ResponseMetadata": {"RetryAttempts": invalid}}
            ),
            f"invalid SDK retry count accepted: {invalid!r}",
        )
    require(analyzer.MAX_S3_SDK_RETRY_RATE == 0.001, "SDK retry-rate threshold drift")


def test_s3_complete_and_pinned_retries() -> None:
    payload = b"deterministic-payload-for-s3-reader"
    bucket = "mewtant-raw-datasets"
    key = "stanford-vision-lab/gpic_umziped/images/train/00001/" + "a" * 64 + ".jpg"

    clean_response = response(
        payload,
        version_id="v-clean",
        etag='"e-clean"',
        sdk_retries=2,
    )
    clean = ScriptedS3Client([clean_response])
    clean_result = call_s3_get_object_result(clean, bucket=bucket, key=key)
    require(clean_result.payload == payload, "complete S3 read failed")
    require(clean_result.sdk_retry_attempts == 2, "SDK RetryAttempts were not captured")
    require(clean_result.object_attempts == 1, "SDK retries were confused with object attempts")
    require(clean_result.content_length_match is True, "complete ContentLength did not close")
    require(clean_result.first_response_attempt == 1, "clean first-response attempt drift")
    require(
        clean_result.first_response_identity_kind == "version_pin",
        "clean first-response identity kind drift",
    )
    require(len(clean.calls) == 1, "complete object unexpectedly retried")
    require("VersionId" not in clean.calls[0] and "IfMatch" not in clean.calls[0], "first GET was pinned")
    require(clean_response["Body"].close_count == 1, "complete body was not closed exactly once")

    version_first = response(
        payload[:5], content_length=len(payload), version_id="v-pinned", etag='"e-one"'
    )
    version_second = response(payload, version_id="v-pinned", etag='"e-one"')
    version_retry = ScriptedS3Client([version_first, version_second])
    version_result = call_s3_get_object_result(version_retry, bucket=bucket, key=key)
    require(version_result.payload == payload, "VersionId-pinned retry did not recover")
    require(version_result.object_attempts == 2, "VersionId object-attempt count drift")
    require(version_result.version_id_present is True, "VersionId presence was not recorded")
    require(version_result.if_match_used is False, "VersionId retry incorrectly reported IfMatch")
    require(
        version_result.first_response_attempt == 1
        and version_result.first_response_identity_kind == "version_pin",
        "VersionId first-response identity accounting drift",
    )
    require("v-pinned" not in repr(version_result) and "e-one" not in repr(version_result), "raw identity leaked from S3ReadResult")
    require(len(version_retry.calls) == 2, "short read did not retry exactly once")
    require(version_retry.calls[1].get("VersionId") == "v-pinned", "retry did not pin VersionId")
    require("IfMatch" not in version_retry.calls[1], "VersionId retry also used IfMatch")
    require(
        version_first["Body"].close_count == version_second["Body"].close_count == 1,
        "VersionId attempt bodies were not closed exactly once",
    )

    etag_first = response(payload[:3], content_length=len(payload), etag='"etag-pinned"')
    etag_second = response(payload, etag='"etag-pinned"')
    etag_retry = ScriptedS3Client([etag_first, etag_second])
    etag_result = call_s3_get_object_result(etag_retry, bucket=bucket, key=key)
    require(etag_result.payload == payload, "IfMatch-pinned retry did not recover")
    require(etag_result.object_attempts == 2, "IfMatch object-attempt count drift")
    require(etag_result.version_id_present is False, "ETag-only response invented VersionId")
    require(etag_result.if_match_used is True, "IfMatch use was not recorded")
    require(
        etag_result.first_response_attempt == 1
        and etag_result.first_response_identity_kind == "if_match_pin",
        "ETag first-response identity accounting drift",
    )
    require("etag-pinned" not in repr(etag_result), "raw ETag leaked from S3ReadResult")
    require(etag_retry.calls[1].get("IfMatch") == '"etag-pinned"', "retry did not pin ETag")
    require("VersionId" not in etag_retry.calls[1], "ETag retry invented VersionId")
    require(
        etag_first["Body"].close_count == etag_second["Body"].close_count == 1,
        "IfMatch attempt bodies were not closed exactly once",
    )

    midstream_first = response(
        payload,
        version_id="v-read-failure",
        fail_after_reads=1,
        failure=OSError("synthetic midstream timeout"),
    )
    midstream_second = response(payload, version_id="v-read-failure")
    midstream = ScriptedS3Client([midstream_first, midstream_second])
    require(
        call_s3_get_object_bytes(midstream, bucket=bucket, key=key, chunk_bytes=4) == payload,
        "midstream failure did not restart the whole object",
    )
    require(midstream.calls[1].get("VersionId") == "v-read-failure", "midstream retry was not pinned")
    require(
        midstream_first["Body"].close_count == midstream_second["Body"].close_count == 1,
        "midstream attempt bodies were not closed exactly once",
    )

    preheader_response = response(payload, version_id="v-after-preheader")
    preheader_retry = ScriptedS3Client(
        [
            MockClientError(
                "InternalError",
                "synthetic pre-header failure",
                500,
                sdk_retries=2,
            ),
            preheader_response,
        ]
    )
    preheader_result = call_s3_get_object_result(
        preheader_retry,
        bucket=bucket,
        key=key,
    )
    require(preheader_result.payload == payload, "pre-header transient retry did not recover")
    require(preheader_result.object_attempts == 2, "pre-header object-attempt count drift")
    require(preheader_result.sdk_retry_attempts == 2, "exception SDK RetryAttempts were dropped")
    require(
        preheader_result.first_response_attempt == 2
        and preheader_result.first_response_identity_kind == "version_pin",
        "pre-header retry/first-response accounting drift",
    )
    require(len(preheader_retry.calls) == 2, "pre-header failure did not retry exactly once")
    require(
        "VersionId" not in preheader_retry.calls[1] and "IfMatch" not in preheader_retry.calls[1],
        "pre-header retry invented an unavailable identity pin",
    )
    require(preheader_response["Body"].close_count == 1, "pre-header retry body close drift")

    unpinned_response = response(payload[:2], content_length=len(payload))
    unpinned = ScriptedS3Client([unpinned_response, response(payload)])
    exc = expect_failure(
        lambda: call_s3_get_object_bytes(unpinned, bucket=bucket, key=key),
        "short read without VersionId/ETag was retried unguarded",
    )
    require(
        getattr(exc, "error_class", None) in {"short_read", "source_changed"},
        "unpinned retry refusal classification drift",
    )
    require(len(unpinned.calls) == 1, "reader issued an unguarded retry")
    require(unpinned_response["Body"].close_count == 1, "unpinned body was not closed exactly once")


def test_s3_identity_drift_and_safe_errors() -> None:
    canaries = sensitive_canaries()
    bucket = "mewtant-raw-datasets"
    key = (
        "stanford-vision-lab/gpic_umziped/images/train/00001/"
        + canaries["key"]
        + ".jpg"
    )
    payload = b"identity-sensitive-payload"

    precondition_first = response(
        payload[:2],
        content_length=len(payload),
        version_id=canaries["version"],
        etag=canaries["etag"],
    )
    precondition = ScriptedS3Client(
        [
            precondition_first,
            MockClientError(
                "PreconditionFailed",
                " ".join(canaries.values()),
                412,
            ),
        ]
    )
    exc = expect_failure(
        lambda: call_s3_get_object_bytes(precondition, bucket=bucket, key=key),
        "HTTP 412 identity change was accepted",
    )
    require(getattr(exc, "error_class", None) == "source_changed", "412 classification drift")
    assert_exception_is_safe(exc, canaries)
    require(precondition_first["Body"].close_count == 1, "412 first body was not closed exactly once")

    drift_first = response(
        payload[:2], content_length=len(payload), version_id="v-before", etag='"e-before"'
    )
    drift_second = response(payload, version_id="v-after", etag='"e-after"')
    drift = ScriptedS3Client(
        [
            drift_first,
            # A malicious/mock service ignores VersionId.  The client-side
            # response identity check must still reject the returned object.
            drift_second,
        ]
    )
    exc = expect_failure(
        lambda: call_s3_get_object_bytes(drift, bucket=bucket, key=key),
        "response VersionId drift was accepted",
    )
    require(getattr(exc, "error_class", None) == "source_changed", "identity drift classification drift")
    assert_exception_is_safe(exc, canaries)
    require(
        drift_first["Body"].close_count == drift_second["Body"].close_count == 1,
        "identity-drift bodies were not closed exactly once",
    )

    exhausted_responses = [
        response(payload[:1], content_length=len(payload), etag=canaries["etag"])
        for _ in range(3)
    ]
    exhausted = ScriptedS3Client(exhausted_responses)
    exc = expect_failure(
        lambda: call_s3_get_object_bytes(exhausted, bucket=bucket, key=key),
        "repeated short reads were accepted",
    )
    require(
        getattr(exc, "error_class", None) in {"short_read", "content_length_mismatch"},
        "short-read exhaustion classification drift",
    )
    assert_exception_is_safe(exc, canaries)
    require(
        all(value["Body"].close_count == 1 for value in exhausted_responses),
        "exhausted short-read bodies were not closed exactly once",
    )

    partial_response = response(payload, status=206, version_id="v-partial")
    partial = ScriptedS3Client([partial_response])
    exc = expect_failure(
        lambda: call_s3_get_object_bytes(partial, bucket=bucket, key=key),
        "non-200 GetObject response was accepted",
    )
    require(getattr(exc, "error_class", None) == "unexpected", "non-200 classification drift")
    require(partial_response["Body"].close_count == 1, "non-200 body was not closed")


def test_no_clobber() -> None:
    with tempfile.TemporaryDirectory(prefix="gpic_s3_abba_publish.") as raw_root:
        root = Path(raw_root)
        target = root / "winner.json"
        barrier = threading.Barrier(2)
        outcomes: list[tuple[str, str]] = []
        lock = threading.Lock()

        def writer(name: str) -> None:
            barrier.wait()
            try:
                call_atomic_publish_json(
                    target,
                    {"schema_version": "synthetic_no_clobber_v1", "writer": name},
                )
                outcome = ("success", name)
            except BaseException:
                outcome = ("rejected", name)
            with lock:
                outcomes.append(outcome)

        threads = [threading.Thread(target=writer, args=(name,)) for name in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            require(not thread.is_alive(), "no-clobber writer deadlocked")
        require(sum(status == "success" for status, _ in outcomes) == 1, "race did not have one winner")
        require(sum(status == "rejected" for status, _ in outcomes) == 1, "race did not reject one writer")
        before = target.read_bytes()
        winner = strict_json(target)["writer"]
        require(winner in {"a", "b"}, "invalid no-clobber winner")
        expect_failure(
            lambda: call_atomic_publish_json(target, {"writer": "late"}),
            "late overwrite was accepted",
        )
        require(target.read_bytes() == before, "rejected overwrite changed winner bytes")
        require(
            {path.name for path in root.iterdir()} == {target.name},
            "no-clobber left temporary artifacts",
        )

        outside = root / "outside.json"
        outside.write_bytes(b"outside-must-not-change\n")
        symlink_target = root / "symlink.json"
        symlink_target.symlink_to(outside)
        expect_failure(
            lambda: call_atomic_publish_json(symlink_target, {"writer": "symlink"}),
            "symlink final target was accepted",
        )
        require(outside.read_bytes() == b"outside-must-not-change\n", "symlink target changed")

        real_parent = root / "real-parent"
        real_parent.mkdir()
        linked_parent = root / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        expect_failure(
            lambda: call_atomic_publish_json(linked_parent / "bad.json", {"writer": "linked"}),
            "symlink parent hierarchy was accepted",
        )
        require(list(real_parent.iterdir()) == [], "symlink parent received an artifact")


def test_input_freeze_attestation() -> None:
    attest = getattr(runner, "_file_attestation", None)
    verify = getattr(runner, "verify_input_attestation", None)
    require(callable(attest) and callable(verify), "runner input-freeze helpers are missing")
    with tempfile.TemporaryDirectory(prefix="gpic_s3_abba_input_freeze.") as raw_root:
        root = Path(raw_root)
        paths = {
            name: root / f"{name}.txt"
            for name in ("runner", "analyzer", "capture_yaml", "source_cache")
        }
        for index, path in enumerate(paths.values()):
            path.write_bytes(f"frozen-input-{index:02d}\n".encode("ascii"))
        declaration = {name: attest(path) for name, path in paths.items()}
        verify(declaration)

        target = paths["runner"]
        original = target.read_bytes()
        target.write_bytes(b"X" * (len(original) - 1) + b"\n")
        expect_failure(
            lambda: verify(declaration),
            "same-size frozen input tamper was accepted",
        )
        target.write_bytes(original)

        extra = copy.deepcopy(declaration)
        extra["unexpected"] = copy.deepcopy(extra["runner"])
        expect_failure(lambda: verify(extra), "extra input-attestation key was accepted")

        source = paths["source_cache"]
        source_bytes = source.read_bytes()
        source.unlink()
        real_source = root / "real-source.txt"
        real_source.write_bytes(source_bytes)
        source.symlink_to(real_source)
        expect_failure(
            lambda: verify(declaration),
            "symlinked frozen input was accepted",
        )


def test_runner_records_match_analyzer_schema() -> None:
    raw_item = {
        "uid": "gpic:" + "a" * 64,
        "src_root": "/s3-mewtant-raw-datasets/stanford-vision-lab/gpic_umziped",
        "src_shard": "00001",
        "src_key": "a" * 64,
        "ext": ".jpg",
    }
    item = runner._source_item(raw_item, 0)
    expected_selection = [{"source_index": 0, "object_id": item.object_id}]
    payload = b"cross-module-schema-payload"
    client = ScriptedS3Client([response(payload, version_id="v-schema")])
    s3_record = runner._item_record(
        arm="s3_x",
        backend="s3",
        cohort="x",
        worker_id=0,
        position=0,
        item=item,
        client=client,
        chunk_bytes=4,
        max_attempts=3,
    )
    require(set(s3_record) == analyzer.ITEM_KEYS, "runner S3 item exact schema drift")
    analyzer._validate_item_record(
        s3_record,
        arm="s3_x",
        backend="s3",
        cohort="x",
        worker_id=0,
        expected_selection=expected_selection,
    )

    with tempfile.TemporaryDirectory(prefix="gpic_s3_abba_mount_record.") as raw_root:
        original_prefix = runner.MOUNT_PREFIX
        try:
            runner.MOUNT_PREFIX = str(Path(raw_root)) + "/"
            location = runner.cache_item_location(raw_item)
            mount_path = Path(location["mount_path"])
            mount_path.parent.mkdir(parents=True)
            mount_path.write_bytes(payload)
            mount_record = runner._item_record(
                arm="mount_x",
                backend="mount",
                cohort="x",
                worker_id=0,
                position=0,
                item=item,
                client=None,
                chunk_bytes=4,
                max_attempts=3,
            )
        finally:
            runner.MOUNT_PREFIX = original_prefix
    require(set(mount_record) == analyzer.ITEM_KEYS, "runner mount item exact schema drift")
    analyzer._validate_item_record(
        mount_record,
        arm="mount_x",
        backend="mount",
        cohort="x",
        worker_id=0,
        expected_selection=expected_selection,
    )


def test_performance_decision_boundaries() -> None:
    def arm(
        *, elapsed: float, rate: float, p99: float = 1.99, status: str = "PASS"
    ) -> dict[str, Any]:
        return {
            "elapsed_s": elapsed,
            "items_per_second": rate,
            "status": status,
            "peak_inflight": 16,
            "io_p99_s": p99,
            "records": [],
            "unguarded_postheader_retry_attempts": 0,
            "body_close_count": 0,
            "response_header_count": 0,
        }

    passing = {
        "mount_x": arm(elapsed=1.5, rate=60.0),
        "s3_x": arm(elapsed=1.0, rate=90.0),
        "s3_y": arm(elapsed=1.0, rate=90.0),
        "mount_y": arm(elapsed=1.5, rate=60.0),
    }
    decision = analyzer._performance_decision(passing)
    require(decision["no_go_failures"] == [], "exact performance boundary did not pass")
    require(decision["inconclusive_failures"] == [], "stable panels became inconclusive")

    slow = copy.deepcopy(passing)
    slow["mount_x"]["elapsed_s"] = 1.49
    decision = analyzer._performance_decision(slow)
    require("x_paired_speedup" in decision["no_go_failures"], "sub-1.5x speedup was accepted")

    drift = copy.deepcopy(passing)
    drift["s3_y"]["items_per_second"] = 100.0
    decision = analyzer._performance_decision(drift)
    require(decision["no_go_failures"] == [], "panel drift incorrectly became NO_GO")
    require(decision["inconclusive_failures"] == ["s3_panel_drift"], "panel drift was not INCONCLUSIVE")


def test_payload_mismatch_and_artifact_tamper() -> None:
    pair_function = getattr(analyzer, "validate_payload_pairs", None)
    require(callable(pair_function), "analyzer.validate_payload_pairs is missing")

    def semantic_row(source_index: int) -> dict[str, Any]:
        identity = hashlib.sha256(f"object-{source_index}".encode("ascii")).hexdigest()
        payload_hash = hashlib.sha256(f"payload-{source_index}".encode("ascii")).hexdigest()
        return {
            "source_index": source_index,
            "object_id": identity,
            "outcome": "success",
            "byte_count": 100 + source_index,
            "payload_sha256": payload_hash,
        }

    x_rows = [semantic_row(index) for index in range(2048)]
    y_rows = [semantic_row(index) for index in range(2048, 4096)]
    valid_records = {
        "mount_x": copy.deepcopy(x_rows),
        "s3_x": copy.deepcopy(x_rows),
        "s3_y": copy.deepcopy(y_rows),
        "mount_y": copy.deepcopy(y_rows),
    }
    summary = pair_function(valid_records)
    require(isinstance(summary, Mapping), "payload-pair summary is not a mapping")
    mismatch = copy.deepcopy(valid_records["s3_x"][37])
    mismatch["payload_sha256"] = hashlib.sha256(b"equal-length-evil").hexdigest()
    mismatch_records = copy.deepcopy(valid_records)
    mismatch_records["s3_x"][37] = mismatch
    expect_failure(
        lambda: pair_function(mismatch_records),
        "same-length payload SHA mismatch was accepted",
    )
    wrong_length = copy.deepcopy(valid_records["mount_y"][41])
    wrong_length["byte_count"] += 1
    wrong_length_records = copy.deepcopy(valid_records)
    wrong_length_records["mount_y"][41] = wrong_length
    expect_failure(
        lambda: pair_function(wrong_length_records),
        "payload length mismatch was accepted",
    )
    duplicate_records = copy.deepcopy(valid_records)
    duplicate_records["mount_x"].append(copy.deepcopy(duplicate_records["mount_x"][0]))
    expect_failure(
        lambda: pair_function(duplicate_records),
        "duplicate payload record was accepted",
    )
    missing_records = copy.deepcopy(valid_records)
    missing_records["s3_y"].pop()
    expect_failure(
        lambda: pair_function(missing_records),
        "one-sided missing payload record was accepted",
    )
    failed_records = copy.deepcopy(valid_records)
    failed_records["s3_y"][13]["outcome"] = "short_read"
    expect_failure(
        lambda: pair_function(failed_records),
        "non-success payload outcome was accepted",
    )

    attest_function = getattr(analyzer, "verify_file_attestation", None)
    require(callable(attest_function), "analyzer.verify_file_attestation is missing")
    with tempfile.TemporaryDirectory(prefix="gpic_s3_abba_tamper.") as raw_root:
        path = Path(raw_root) / "evidence.jsonl"
        original = b'{"schema_version":"synthetic_evidence_v1"}\n'
        path.write_bytes(original)
        expected_sha = hashlib.sha256(original).hexdigest()
        observed = attest_function(
            path,
            expected_bytes=len(original),
            expected_sha256=expected_sha,
        )
        require(observed == original, "attestation did not return verified bytes")
        path.write_bytes(original[:-2] + b"x\n")
        expect_failure(
            lambda: attest_function(
                path,
                expected_bytes=len(original),
                expected_sha256=expected_sha,
            ),
            "same-size artifact tamper was accepted",
        )


def test_leak_scanner_and_safe_artifact() -> None:
    scan_function = getattr(analyzer, "scan_for_leaks", None)
    require(callable(scan_function), "analyzer.scan_for_leaks is missing")
    safe = (
        b'{"object_id":"' + b"a" * 64 + b'","error_class":"source_changed",'
        b'"version_id_present":true}\n'
    )
    result = scan_function(safe)
    require(result is None or result is True or result == [], "safe artifact rejected by leak scanner")
    canaries = sensitive_canaries()
    leak_payloads = {
        "key": canaries["key"].encode("ascii"),
        "uri": canaries["uri"].encode("ascii"),
        "access": canaries["access"].encode("ascii"),
        "secret": analyzer.canonical_json_bytes(
            {"aws_secret_access_key": canaries["secret"]}
        ),
        "session": analyzer.canonical_json_bytes(
            {"aws_session_token": canaries["session"]}
        ),
        "version": analyzer.canonical_json_bytes({"VersionId": canaries["version"]}),
        "etag": analyzer.canonical_json_bytes({"ETag": canaries["etag"]}),
    }
    for label, raw in leak_payloads.items():
        exc = expect_failure(
            lambda raw=raw: scan_function(
                raw,
                forbidden_source_keys=(canaries["key"],),
            ),
            f"{label} leak canary was accepted",
        )
        require(canaries[label] not in str(exc), f"{label} canary repeated by scanner exception")


def test_source_contract_tokens() -> None:
    runner_source = Path(runner.__file__).read_text(encoding="utf-8")
    analyzer_source = Path(analyzer.__file__).read_text(encoding="utf-8")
    for token in (
        "VersionId",
        "IfMatch",
        "ContentLength",
        "source_changed",
        "short_read",
        "max_pool_connections",
        "total_max_attempts",
        "tcp_keepalive",
        "ThreadPoolExecutor",
        "multiprocessing",
        "observed_peak",
        "atomic_publish_json",
        "credential_summary",
        "classify_s3_error",
        "sdk_retry_attempts",
    ):
        require(token in runner_source, f"runner contract token missing: {token}")
    for token in (
        EXPECTED_SELECTION_SHA256,
        "validate_payload_pairs",
        "verify_file_attestation",
        "scan_for_leaks",
        "payload_sha256",
        "source_cache_sha256",
        "capture_complete.json",
    ):
        require(token in analyzer_source, f"analyzer contract token missing: {token}")
    forbidden = ("aws_access_key_id=", "aws_secret_access_key=", "set_debug_logger")
    combined = (runner_source + analyzer_source).lower()
    for token in forbidden:
        require(token not in combined, f"forbidden credential/debug source token: {token}")


def main() -> int:
    items, _selection = test_frozen_cache_and_selection()
    test_cache_schema_and_path_mapping(items)
    test_topology_contract()
    test_client_config_credentials_and_error_classes()
    test_s3_complete_and_pinned_retries()
    test_s3_identity_drift_and_safe_errors()
    test_no_clobber()
    test_input_freeze_attestation()
    test_runner_records_match_analyzer_schema()
    test_performance_decision_boundaries()
    test_payload_mismatch_and_artifact_tamper()
    test_leak_scanner_and_safe_artifact()
    test_source_contract_tokens()
    print(
        "GPIC_MOUNT_S3_ABBA_STATIC_PASS "
        "cases=13 selected=4160 unique_physical_shards=4160 network_calls=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
