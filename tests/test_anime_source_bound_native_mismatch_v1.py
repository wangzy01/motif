import hashlib

import numpy as np

from training.dataline_offline.offline_dataset import (
    OfflineExactDataset,
    OfflineLoaderError,
    _PlannedItem,
    _PreparedItem,
)


def _fixture():
    pixels = np.ascontiguousarray(np.arange(60, dtype=np.uint8).reshape(4, 5, 3))
    raw = b"abc"
    planned = _PlannedItem(
        step=10470,
        slot=20,
        row_id=4254719,
        domain="anime",
        domain_idx=0,
        bucket_idx=130,
        caption_band=-1,
    )
    catalog = {
        "row_id": 4254719,
        "domain": "anime",
        "caption_parquet_path": (
            "/s3/evan/anime_core_ditv04_ct_soft/data/00000013.parquet"
        ),
        "caption_row_group": 6,
        "caption_row_in_group": 38260,
        "key": "key",
        "fetch_url": "https://example.invalid/payload",
        "width": 1032,
        "height": 1457,
    }
    source = {
        "key": catalog["key"],
        "s3key": catalog["fetch_url"],
        "width": catalog["width"],
        "height": catalog["height"],
        "image_size": len(raw),
        "md5": hashlib.md5(pixels.tobytes(order="C")).hexdigest(),
    }
    dataset = OfflineExactDataset.__new__(OfflineExactDataset)

    def read_source_rows(pairs, **kwargs):
        assert len(pairs) == 1
        assert kwargs["columns"] == (
            "md5", "width", "height", "key", "image_size", "s3key"
        )
        return {(planned.step, planned.slot): dict(source)}

    dataset._read_source_rows = read_source_rows
    prepared = _PreparedItem(planned, catalog, {})
    return dataset, prepared, raw, pixels, source


def _must_fail(dataset, prepared, raw, pixels):
    try:
        dataset._validate_anime_source_bound_native_mismatch(
            prepared, raw, pixels
        )
    except OfflineLoaderError:
        return
    raise AssertionError("source-bound mismatch gate did not fail closed")


def test_source_bound_binding_and_negative_cases():
    dataset, prepared, raw, pixels, source = _fixture()
    assert dataset._validate_anime_source_bound_native_mismatch(
        prepared, raw, pixels
    ) == source["md5"]

    dataset, prepared, raw, pixels, source = _fixture()
    prepared.catalog["fetch_url"] = "https://example.invalid/drift"
    _must_fail(dataset, prepared, raw, pixels)

    dataset, prepared, raw, pixels, source = _fixture()
    _must_fail(dataset, prepared, raw + b"x", pixels)

    dataset, prepared, raw, pixels, source = _fixture()
    drifted = pixels.copy()
    drifted[0, 0, 0] ^= 1
    _must_fail(dataset, prepared, raw, drifted)

    dataset, prepared, raw, pixels, source = _fixture()
    prepared.catalog["caption_parquet_path"] = "/tmp/not-authoritative.parquet"
    _must_fail(dataset, prepared, raw, pixels)
