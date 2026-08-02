"""Runtime reader for the TextAtlas authoritative ground-truth sidecar.

A separate artifact from the caption sidecar, and separate on purpose: the Qwen (i1)
captions are a transcription (73.5% exact against CleanTextSynth's authoritative
``annotation``), and nothing that rewrites captions should be structurally able to
overwrite a recorded ground truth.  Keeping them apart also means adding GT to the
prompt did not require rebuilding the 56 GB caption sidecar.

Coverage is partial by construction -- 1,941,791 rows against ~4.12M TextAtlas rows in
the catalog, because only two of the nine subsets carry a string that is a RECORD
rather than a description.  A miss is normal and returns None; the caller falls back
to the plain caption.  See ``scripts/offline_arb/build_textatlas_gt_sidecar_v1.py``
for why the other seven are excluded.

The whole file is read once at construction (111 MB for the formal build) and hashed
from the bytes actually parsed, rather than re-read for hashing -- the phase5 lesson,
where a separate read left a 26-minute window in which the file could change between
being parsed and being digested.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SCHEMA = "stage512b_textatlas_gt_sidecar_v2"
PARQUET_NAME = "textatlas_gt.parquet"


class TextAtlasGroundTruthError(RuntimeError):
    pass


class TextAtlasGroundTruth:
    """row_id -> the exact string the image renders, or None.

    Lookup is ``searchsorted`` over a sorted int64 array rather than a dict: at
    1.9M entries a Python dict of int->str costs roughly 300 MB per rank, and this
    runs in every one of 64 ranks.  The arrays cost about a third of that, and a
    batch only performs 32 lookups per step, so the log-n is free.
    """

    def __init__(self, root, catalog_hash: str, catalog_header_sha256: str):
        self.root = Path(str(root)).expanduser()
        header_path = self.root / "header.json"
        parquet_path = self.root / PARQUET_NAME
        for path in (header_path, parquet_path, self.root / "_SUCCESS"):
            if not path.is_file():
                raise TextAtlasGroundTruthError(f"missing {path}")

        try:
            header = json.loads(header_path.read_text())
        except ValueError as exc:
            raise TextAtlasGroundTruthError(f"unreadable header: {exc}") from exc

        # v1 shipped a single NFKC-folded ``gt_text``; training on it would mean
        # training on a rewritten target.  Refuse it by name rather than by a
        # column-missing traceback.
        schema = header.get("schema")
        if schema != SCHEMA:
            raise TextAtlasGroundTruthError(
                f"expected schema {SCHEMA}, found {schema!r}. v1 folded NFKC and "
                "whitespace into the only text column and is not usable for training."
            )
        if header.get("catalog_hash") != catalog_hash:
            raise TextAtlasGroundTruthError(
                "GT sidecar was built against a different catalog: "
                f"sidecar={header.get('catalog_hash')!r} runtime={catalog_hash!r}"
            )
        if header.get("catalog_header_sha256") != catalog_header_sha256:
            raise TextAtlasGroundTruthError(
                "GT sidecar catalog_header_sha256 mismatch: "
                f"sidecar={header.get('catalog_header_sha256')!r} "
                f"runtime={catalog_header_sha256!r}"
            )

        payload = parquet_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        expected = (header.get("output") or {}).get("sha256")
        if digest != expected:
            raise TextAtlasGroundTruthError(
                f"{PARQUET_NAME} digest {digest} != header {expected!r}"
            )
        success = (self.root / "_SUCCESS").read_text().strip()
        if success != expected:
            raise TextAtlasGroundTruthError(
                f"_SUCCESS says {success!r}, header says {expected!r}"
            )

        table = pq.read_table(io.BytesIO(payload), columns=["row_id", "gt_text_raw"])
        del payload
        self._ids = np.asarray(table["row_id"].to_numpy(zero_copy_only=False), dtype=np.int64)
        # Combine chunks so one flat index works; leaving it chunked would mean
        # translating a global position into (chunk, offset) on every lookup.
        self._texts = table["gt_text_raw"].combine_chunks()
        if self._ids.size != len(self._texts):
            raise TextAtlasGroundTruthError("row_id and gt_text_raw lengths disagree")
        if self._ids.size and not bool(np.all(self._ids[1:] > self._ids[:-1])):
            # The builder sorts and de-duplicates; if that ever stops being true,
            # searchsorted silently returns the wrong row rather than failing.
            raise TextAtlasGroundTruthError("row_id is not strictly increasing")

        self.header = header
        self.rows = int(self._ids.size)
        self.catalog_hash = catalog_hash
        self.hits = 0
        self.misses = 0

    def get(self, row_id: int) -> str | None:
        if not self._ids.size:
            self.misses += 1
            return None
        index = int(np.searchsorted(self._ids, np.int64(row_id)))
        if index >= self._ids.size or int(self._ids[index]) != int(row_id):
            self.misses += 1
            return None
        self.hits += 1
        return self._texts[index].as_py()

    def coverage(self) -> dict:
        looked_up = self.hits + self.misses
        return {
            "rows": self.rows,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / looked_up) if looked_up else None,
        }
