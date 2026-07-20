#!/usr/bin/env python3
"""Independently bind suggested TextAtlas EOI repairs to repaired-byte SHA256."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


class BindingError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(row: dict[str, Any]) -> bytes:
    table = pq.ParquetFile(row["image_parquet_path"]).read_row_group(
        int(row["image_row_group"]), columns=["image", "image_path"], use_threads=False
    ).combine_chunks()
    offset = int(row["image_row_in_group"])
    observed_path = table.column("image_path").chunk(0)[offset].as_py()
    if str(observed_path) != str(row["image_path"]):
        raise BindingError(f"source path drift row_id={row['row_id']}")
    raw = table.column("image").chunk(0).field("bytes")[offset].as_py()
    if not isinstance(raw, bytes) or not raw:
        raise BindingError(f"empty source payload row_id={row['row_id']}")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    allowlist_path = args.allowlist.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise BindingError(f"refusing to overwrite output: {output}")
    allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
    rows = allowlist.get("rows")
    if not isinstance(rows, list) or not rows:
        raise BindingError("allowlist has no rows")
    bindings = []
    for row in rows:
        raw = _read(row)
        raw_sha = hashlib.sha256(raw).hexdigest()
        if raw_sha != row["payload_sha256"] or raw.endswith(b"\xff\xd9"):
            raise BindingError(f"raw identity/signature drift row_id={row['row_id']}")
        repaired = raw + b"\xff\xd9"
        with Image.open(io.BytesIO(repaired)) as image:
            rgb = image.convert("RGB")
            rgb.load()
            pixels = rgb.tobytes()
            size = [int(rgb.size[0]), int(rgb.size[1])]
        pixel_sha = hashlib.sha256(pixels).hexdigest()
        if pixel_sha != row["expected_rgb_sha256"] or size != row["expected_decoded_size"]:
            raise BindingError(f"repaired decode binding drift row_id={row['row_id']}")
        bindings.append(
            {
                "row_id": int(row["row_id"]),
                "raw_bytes": len(raw),
                "raw_sha256": raw_sha,
                "appended_bytes_hex": "ffd9",
                "repaired_bytes": len(repaired),
                "repaired_payload_sha256": hashlib.sha256(repaired).hexdigest(),
                "repaired_rgb_sha256": pixel_sha,
                "repaired_decoded_size": size,
            }
        )
    result = {
        "status": "PASS",
        "audit": "offline_arb_textatlas_repaired_payload_bindings_v1",
        "read_only": True,
        "allowlist_path": str(allowlist_path),
        "allowlist_sha256": _sha(allowlist_path),
        "rows": bindings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
