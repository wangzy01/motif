#!/usr/bin/env python3
"""Network-free static and mocked tests for the GPIC S3 auth matrix."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import yaml

import diagnose_gpic_s3_auth_matrix as target


class FakeBody:
    def __init__(self, payload: bytes, *, fail_read: bool = False) -> None:
        self.payload = payload
        self.offset = 0
        self.fail_read = fail_read
        self.closed = 0

    def read(self, amount: int) -> bytes:
        if self.fail_read:
            raise OSError("raw canary must never escape")
        if self.offset >= len(self.payload):
            return b""
        end = min(len(self.payload), self.offset + max(1, amount))
        block = self.payload[self.offset:end]
        self.offset = end
        return block

    def close(self) -> None:
        self.closed += 1


class FakeServiceError(Exception):
    def __init__(self, code: str, status: int) -> None:
        self.response = {
            "Error": {"Code": code, "Message": "raw canary must never escape"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }
        super().__init__("raw canary must never escape")


class FakeClient:
    def __init__(
        self,
        payload: bytes,
        *,
        error: BaseException | None = None,
        content_length: int | None = None,
    ) -> None:
        self.payload = payload
        self.error = error
        self.content_length = len(payload) if content_length is None else content_length
        self.calls: list[dict[str, Any]] = []
        self.bodies: list[FakeBody] = []
        self.closed = 0

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        body = FakeBody(self.payload)
        self.bodies.append(body)
        return {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "ContentLength": self.content_length,
            "Body": body,
        }

    def close(self) -> None:
        self.closed += 1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_config_contract() -> None:
    signed = target.build_s3_config("signed")
    unsigned = target.build_s3_config("unsigned")
    require(signed.signature_version == "s3v4", "signed signature drift")
    require(unsigned.signature_version is target.UNSIGNED, "unsigned signature drift")
    require(signed.retries.get("total_max_attempts") == 1, "signed retry drift")
    require(unsigned.retries.get("total_max_attempts") == 1, "unsigned retry drift")
    require(signed.connect_timeout == 10.0, "connect timeout drift")
    require(signed.read_timeout == 30.0, "read timeout drift")


def test_exact_four_calls_and_result_sanitization() -> None:
    baseline = b"frozen-payload" * 257
    clients: list[FakeClient] = []
    modes: list[str] = []

    def factory(mode: str) -> FakeClient:
        modes.append(mode)
        client = FakeClient(baseline)
        clients.append(client)
        return client

    raw_key_canary = "private/raw/key/canary"
    cells = target.execute_matrix(
        key=raw_key_canary,
        baseline=baseline,
        client_factory=factory,
    )
    require(
        modes == ["signed", "unsigned", "signed", "unsigned"],
        "cell auth order drift",
    )
    require(len(clients) == 4, "client count drift")
    for client in clients:
        require(len(client.calls) == 1, "cell must issue exactly one GetObject")
        require(set(client.calls[0]) == {"Bucket", "Key"}, "request parameter drift")
        require(client.calls[0]["Key"] == raw_key_canary, "one-object key drift")
        require(client.closed == 1, "client close drift")
        require(len(client.bodies) == 1 and client.bodies[0].closed == 1, "body close drift")
    require(all(cell["status"] == "match" for cell in cells), "match result drift")
    result = {
        "schema_version": target.SCHEMA_VERSION,
        "status": "COMPLETE",
        "cells": cells,
    }
    forbidden = {
        raw_key_canary.encode("ascii"),
        target.BUCKET_CANDIDATE_A.encode("ascii"),
        target.BUCKET_CANDIDATE_B.encode("ascii"),
        b"raw canary must never escape",
    }
    raw = target.validate_public_result(result, forbidden)
    require(not any(token in raw for token in forbidden), "public result leaked a canary")


def test_failure_classification_and_payload_checks() -> None:
    baseline = b"abcdef"
    denied = FakeClient(
        baseline,
        error=FakeServiceError("AccessDenied", 403),
    )
    denied_result = target.probe_cell(
        cell_label="signed_candidate_a",
        client=denied,
        bucket="mock-a",
        key="mock-key",
        baseline=baseline,
    )
    require(denied_result["status"] == "access_denied", "access classification drift")
    require(len(denied.calls) == 1, "denied call count drift")
    require("raw" not in json.dumps(denied_result), "exception text leaked")

    wrong = FakeClient(b"abcdeg")
    wrong_result = target.probe_cell(
        cell_label="unsigned_candidate_a",
        client=wrong,
        bucket="mock-a",
        key="mock-key",
        baseline=baseline,
    )
    require(wrong_result["status"] == "payload_mismatch", "payload mismatch drift")
    require(wrong_result["bytes_match"] is True, "length equivalence drift")
    require(wrong_result["sha256_match"] is False, "SHA equivalence drift")
    require(wrong_result["content_length_match"] is True, "content length drift")

    short = FakeClient(baseline, content_length=len(baseline) + 1)
    short_result = target.probe_cell(
        cell_label="signed_candidate_b",
        client=short,
        bucket="mock-b",
        key="mock-key",
        baseline=baseline,
    )
    require(short_result["status"] == "payload_mismatch", "short response drift")
    require(short_result["content_length_match"] is False, "short length drift")


def test_source_ast_and_job_contract(script_path: Path, job_path: Path) -> None:
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    forbidden_attrs = {
        "head_bucket",
        "head_object",
        "list_buckets",
        "list_objects",
        "list_objects_v2",
    }
    get_object_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            require(node.attr not in forbidden_attrs, "forbidden API call present")
            if node.attr == "get_object":
                get_object_calls += 1
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "client"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            require(node.args[0].value == "s3", "non-S3 service client present")
    require(get_object_calls == 1, "GetObject call count drift")
    require("session.client(\"s3\"" in source, "S3 client construction drift")

    job = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    for key, wanted in {
        "numWorkers": 1,
        "gpuPerWorker": 0,
        "cpuPerWorker": 16,
        "maxRetries": 0,
    }.items():
        require(job.get(key) == wanted, f"job {key} drift")
    command = job.get("command")
    require(isinstance(command, str), "job command missing")
    require("test_gpic_s3_auth_matrix_static.py" in command, "mock gate missing")
    require("diagnose_gpic_s3_auth_matrix.py" in command, "live diagnostic missing")
    require("AWS_EC2_METADATA_DISABLED" not in command, "credential provider disabled")
    require("unset AWS_" not in command, "AWS identity environment unset")
    for token in ("head_bucket", "head_object", "list_objects", "list_buckets"):
        require(token not in command, "forbidden API in job command")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: test_gpic_s3_auth_matrix_static.py SCRIPT JOB_YAML")
    script_path, job_path = map(Path, argv)
    test_config_contract()
    test_exact_four_calls_and_result_sanitization()
    test_failure_classification_and_payload_checks()
    test_source_ast_and_job_contract(script_path, job_path)
    print("GPIC_S3_AUTH_MATRIX_STATIC_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
