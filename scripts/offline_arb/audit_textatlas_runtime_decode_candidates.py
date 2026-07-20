#!/usr/bin/env python3
"""Recheck full-audit TextAtlas candidates with the actual production decoder.

The broad strict audit intentionally imported Pillow directly and therefore
used Pillow's default decompression-bomb limit.  Production imports
``training.dataline.image_fetch``, which sets ``Image.MAX_IMAGE_PIXELS=None``
before ``decode_rgb`` is called.  This targeted audit executes that exact
decoder policy in a fresh child process per row so a giant RGB allocation is
fully reclaimed before the next candidate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import multiprocessing
import os
import resource
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


class RecheckError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_kib() -> int:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


def _read_raw(candidate: dict[str, Any]) -> bytes:
    locator = candidate["locator"]
    parquet = pq.ParquetFile(locator["image_parquet_path"])
    table = parquet.read_row_group(
        int(locator["image_row_group"]),
        columns=["image", "image_path"],
        use_threads=False,
    ).combine_chunks()
    offset = int(locator["image_row_in_group"])
    if offset < 0 or offset >= table.num_rows:
        raise RecheckError(f"source offset out of range row_id={candidate['row_id']}")
    observed_path = table.column("image_path").chunk(0)[offset].as_py()
    if str(observed_path) != str(locator["image_path"]):
        raise RecheckError(f"source image_path drift row_id={candidate['row_id']}")
    raw = table.column("image").chunk(0).field("bytes")[offset].as_py()
    if not isinstance(raw, bytes) or not raw:
        raise RecheckError(f"empty source payload row_id={candidate['row_id']}")
    return raw


def _direct_runtime_decode_error(raw: bytes) -> tuple[str | None, str | None]:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert("RGB")
            rgb.load()
        return None, None
    except Exception as exc:  # noqa: BLE001 - preserve decoder evidence
        return type(exc).__name__, str(exc)


def _worker(
    candidate: dict[str, Any],
    binding: dict[str, Any] | None,
    connection: Any,
) -> None:
    started = time.monotonic()
    result: dict[str, Any] = {
        "row_id": int(candidate["row_id"]),
        "status": "FAIL",
    }
    try:
        # Importing the actual production module is itself part of the policy
        # proof: it sets the shared Pillow process global before decode_rgb.
        from PIL import Image
        from training.dataline.image_fetch import decode_rgb

        result["image_max_pixels_after_production_import"] = Image.MAX_IMAGE_PIXELS
        result["rss_kib_before_read"] = _rss_kib()
        raw = _read_raw(candidate)
        raw_sha = hashlib.sha256(raw).hexdigest()
        if len(raw) != int(candidate["payload_bytes"]) or raw_sha != str(
            candidate["payload_sha256"]
        ):
            raise RecheckError(
                f"payload identity drift row_id={candidate['row_id']}: "
                f"bytes={len(raw)},sha256={raw_sha}"
            )
        result.update(
            {
                "raw_bytes": len(raw),
                "raw_sha256": raw_sha,
                "rss_kib_after_read": _rss_kib(),
            }
        )

        image = decode_rgb(raw)
        if image is not None:
            image.load()
            size = [int(image.size[0]), int(image.size[1])]
            result.update(
                {
                    "raw_runtime_decode_ok": True,
                    "raw_runtime_error_type": None,
                    "raw_runtime_error": None,
                    "raw_decoded_size": size,
                    "raw_decoded_mode": image.mode,
                    "pixel_load_succeeded": True,
                    "corner_pixels": [
                        list(image.getpixel((0, 0))),
                        list(image.getpixel((image.size[0] - 1, image.size[1] - 1))),
                    ],
                    "rss_kib_after_raw_decode": _rss_kib(),
                }
            )
            image.close()
        else:
            error_type, error = _direct_runtime_decode_error(raw)
            result.update(
                {
                    "raw_runtime_decode_ok": False,
                    "raw_runtime_error_type": error_type,
                    "raw_runtime_error": error,
                    "raw_decoded_size": None,
                    "raw_decoded_mode": None,
                    "pixel_load_succeeded": False,
                    "corner_pixels": None,
                    "rss_kib_after_raw_decode": _rss_kib(),
                }
            )

        if binding is None:
            result["repair"] = None
        else:
            repaired = raw + b"\xff\xd9"
            repaired_sha = hashlib.sha256(repaired).hexdigest()
            if (
                int(binding["raw_bytes"]) != len(raw)
                or str(binding["raw_sha256"]) != raw_sha
                or int(binding["repaired_bytes"]) != len(repaired)
                or str(binding["repaired_payload_sha256"]) != repaired_sha
            ):
                raise RecheckError(
                    f"repair byte binding drift row_id={candidate['row_id']}"
                )
            repaired_image = decode_rgb(repaired)
            if repaired_image is None:
                raise RecheckError(
                    f"production decoder rejected repaired row_id={candidate['row_id']}"
                )
            repaired_image.load()
            repaired_size = [
                int(repaired_image.size[0]),
                int(repaired_image.size[1]),
            ]
            repaired_pixel_sha = hashlib.sha256(repaired_image.tobytes()).hexdigest()
            repaired_image.close()
            if (
                repaired_size != list(binding["repaired_decoded_size"])
                or repaired_pixel_sha != str(binding["repaired_rgb_sha256"])
            ):
                raise RecheckError(
                    f"repair pixel binding drift row_id={candidate['row_id']}"
                )
            result["repair"] = {
                "ok": True,
                "repaired_bytes": len(repaired),
                "repaired_payload_sha256": repaired_sha,
                "repaired_decoded_size": repaired_size,
                "repaired_rgb_sha256": repaired_pixel_sha,
            }

        result["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001 - return exact child failure
        result["worker_error_type"] = type(exc).__name__
        result["worker_error"] = str(exc)
    finally:
        gc.collect()
        result["elapsed_seconds"] = round(time.monotonic() - started, 6)
        result["ru_maxrss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        try:
            connection.send(result)
        finally:
            connection.close()


def _run_one(
    candidate: dict[str, Any],
    binding: dict[str, Any] | None,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(candidate, binding, send))
    process.start()
    send.close()
    process.join(timeout_seconds)
    if process.is_alive():
        process.kill()
        process.join()
        receive.close()
        raise RecheckError(
            f"runtime decode timeout row_id={candidate['row_id']},"
            f"timeout_seconds={timeout_seconds}"
        )
    if not receive.poll():
        exitcode = process.exitcode
        receive.close()
        raise RecheckError(
            f"runtime decode child returned no result row_id={candidate['row_id']},"
            f"exitcode={exitcode}"
        )
    result = receive.recv()
    receive.close()
    if process.exitcode != 0 or result.get("status") != "PASS":
        raise RecheckError(
            f"runtime decode child failed row_id={candidate['row_id']},"
            f"exitcode={process.exitcode},result={result!r}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--expected-bindings-sha256", required=True)
    parser.add_argument("--image-fetch", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    audit_path = args.audit.expanduser().resolve(strict=True)
    bindings_path = args.bindings.expanduser().resolve(strict=True)
    image_fetch_path = args.image_fetch.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise RecheckError(f"refusing to overwrite output: {output}")
    if not 0 <= args.shard_index < args.shard_count:
        raise RecheckError("invalid shard index/count")
    if _sha256(audit_path) != args.expected_audit_sha256:
        raise RecheckError("full integrity audit SHA drift")
    if _sha256(bindings_path) != args.expected_bindings_sha256:
        raise RecheckError("repaired bindings SHA drift")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    candidates = sorted(audit["candidates"], key=lambda item: int(item["row_id"]))
    bound = {int(item["row_id"]): item for item in bindings["rows"]}
    assigned = [
        item
        for index, item in enumerate(candidates)
        if index % args.shard_count == args.shard_index
    ]
    rows = []
    for position, candidate in enumerate(assigned, 1):
        result = _run_one(
            candidate,
            bound.get(int(candidate["row_id"])),
            timeout_seconds=args.timeout_seconds,
        )
        rows.append(result)
        print(
            json.dumps(
                {
                    "phase": "candidate_complete",
                    "shard": args.shard_index,
                    "position": position,
                    "total": len(assigned),
                    **result,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    value = {
        "status": "PASS",
        "audit": "offline_arb_textatlas_runtime_decode_candidate_recheck_v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "provenance": {
            "integrity_audit_path": str(audit_path),
            "integrity_audit_sha256": _sha256(audit_path),
            "repaired_bindings_path": str(bindings_path),
            "repaired_bindings_sha256": _sha256(bindings_path),
            "production_image_fetch_path": str(image_fetch_path),
            "production_image_fetch_sha256": _sha256(image_fetch_path),
        },
        "summary": {
            "assigned_rows": len(rows),
            "raw_runtime_decode_successes": sum(
                bool(item["raw_runtime_decode_ok"]) for item in rows
            ),
            "raw_runtime_decode_failures": sum(
                not bool(item["raw_runtime_decode_ok"]) for item in rows
            ),
            "repair_successes": sum(
                item.get("repair") is not None and bool(item["repair"]["ok"])
                for item in rows
            ),
            "max_child_ru_maxrss_kib": max(
                (int(item["ru_maxrss_kib"]) for item in rows), default=0
            ),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(value["summary"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
