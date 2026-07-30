#!/usr/bin/env python3
"""Known-answer and negative tests for frozen caption exceptions v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import warnings
from pathlib import Path
from types import SimpleNamespace

from training.dataline.manifest_index import CAPTIONER_CODE
from training.dataline_offline.caption_exceptions_v1 import (
    AUDIT_JSON_SHA256,
    CAPTIONLESS_MONET_ROWS,
    CATALOG_HASH,
    CATALOG_HEADER_SHA256,
    ELIGIBILITY,
    MONET_CAPTION_CODES,
    PLAN_HEADER_SHA256,
    PLAN_ID,
    STAGE,
    CaptionExceptionBindingError,
    bind_caption_exceptions,
)
from training.dataline_offline.offline_dataset import (
    OfflineExactDataset,
    OfflineLoaderError,
    _VirtualRowDecoder,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expect_error(fn, contains: str) -> None:
    try:
        fn()
    except (CaptionExceptionBindingError, OfflineLoaderError) as exc:
        assert contains in str(exc), (contains, str(exc))
    else:
        raise AssertionError(f"expected error containing {contains!r}")


def record(row_id: int, uid: str) -> dict:
    return {
        "row_id": row_id,
        "uid": uid,
        "served_mask": 0,
        "band_captioner": [0, 0, 0, 0],
        "cap_texts": {code: None for code in MONET_CAPTION_CODES},
    }


def run(args: argparse.Namespace) -> None:
    plan_header = json.loads(args.plan_header.read_text(encoding="utf-8"))
    catalog_header = json.loads(args.catalog_header.read_text(encoding="utf-8"))
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    assert sha256(args.plan_header) == PLAN_HEADER_SHA256
    assert sha256(args.catalog_header) == CATALOG_HEADER_SHA256
    assert sha256(args.audit_json) == AUDIT_JSON_SHA256
    assert plan_header["plan_id"] == PLAN_ID
    assert plan_header["catalog_hash"] == CATALOG_HASH
    assert catalog_header["catalog_hash"] == CATALOG_HASH

    decoder = _VirtualRowDecoder(plan_header["domains"], plan_header["N"])
    bound = bind_caption_exceptions(
        stage=STAGE,
        eligibility=ELIGIBILITY,
        plan_id=PLAN_ID,
        catalog_hash=CATALOG_HASH,
        plan_header_sha256=PLAN_HEADER_SHA256,
        catalog_header_sha256=CATALOG_HEADER_SHA256,
        decode_domain=lambda row_id: plan_header["domains"][decoder.decode(row_id)[0]],
    )
    assert dict(bound) == dict(CAPTIONLESS_MONET_ROWS)
    audit_rows = {
        int(row["row_id"]): row["uid"]
        for row in audit["unavailable_rows"]
        if row["domain"] == "monet_real"
    }
    assert audit_rows == dict(CAPTIONLESS_MONET_ROWS)
    assert audit["plan_summary"]["monet_real"]["planned_unavailable_exposures"] == 32
    assert audit["monet_sidecar"]["exposure_weighted_mask_distribution"] == {
        "0b00000": 32
    }

    dataset = object.__new__(OfflineExactDataset)
    dataset._captionless_monet_rows = bound
    dataset.cfg = SimpleNamespace(caption_truncate_tok=(300, 340))
    for row_id, uid in CAPTIONLESS_MONET_ROWS.items():
        rec = record(row_id, uid)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            band = dataset._select_band("monet_real", rec, random.Random(0))
        assert band == -1
        assert len(caught) == 1 and "frozen captionless MONET exception" in str(
            caught[0].message
        )
        assert dataset._build_caption("monet_real", rec, band, random.Random(0)) == ""

    unrelated = bind_caption_exceptions(
        stage="512a",
        eligibility="elig_512",
        plan_id="synthetic-test-plan",
        catalog_hash=CATALOG_HASH,
        plan_header_sha256="1" * 64,
        catalog_header_sha256=CATALOG_HEADER_SHA256,
        decode_domain=lambda _row_id: "gpic",
    )
    assert not unrelated
    expect_error(
        lambda: bind_caption_exceptions(
            stage=STAGE,
            eligibility=ELIGIBILITY,
            plan_id=PLAN_ID,
            catalog_hash="0" * 64,
            plan_header_sha256=PLAN_HEADER_SHA256,
            catalog_header_sha256=CATALOG_HEADER_SHA256,
            decode_domain=lambda _row_id: "monet_real",
        ),
        "binding mismatch",
    )
    expect_error(
        lambda: bind_caption_exceptions(
            stage=STAGE,
            eligibility=ELIGIBILITY,
            plan_id=PLAN_ID,
            catalog_hash=CATALOG_HASH,
            plan_header_sha256=PLAN_HEADER_SHA256,
            catalog_header_sha256=CATALOG_HEADER_SHA256,
            decode_domain=lambda _row_id: "gpic",
        ),
        "not monet_real",
    )
    expect_error(
        lambda: bind_caption_exceptions(
            stage="512a",
            eligibility="elig_512",
            plan_id=PLAN_ID,
            catalog_hash=CATALOG_HASH,
            plan_header_sha256=PLAN_HEADER_SHA256,
            catalog_header_sha256=CATALOG_HEADER_SHA256,
            decode_domain=lambda _row_id: "monet_real",
        ),
        "binding mismatch",
    )

    first_id, first_uid = next(iter(CAPTIONLESS_MONET_ROWS.items()))
    expect_error(
        lambda: dataset._select_band(
            "monet_real", record(first_id - 1, "monet:unlisted"), random.Random(0)
        ),
        "no served caption band",
    )
    drift = record(first_id, first_uid)
    drift["uid"] = "monet:wrong"
    expect_error(
        lambda: dataset._select_band("monet_real", drift, random.Random(0)),
        "record drift",
    )
    drift = record(first_id, first_uid)
    drift["cap_texts"][CAPTIONER_CODE["original"]] = "now nonempty"
    expect_error(
        lambda: dataset._select_band("monet_real", drift, random.Random(0)),
        "record drift",
    )
    drift = record(first_id, first_uid)
    drift["cap_texts"][CAPTIONER_CODE["original"]] = ""
    expect_error(
        lambda: dataset._select_band("monet_real", drift, random.Random(0)),
        "record drift",
    )
    drift = record(first_id, first_uid)
    drift["served_mask"] = 1
    expect_error(
        lambda: dataset._select_band("monet_real", drift, random.Random(0)),
        "record drift",
    )
    expect_error(
        lambda: dataset._select_band(
            "gpic", record(first_id, first_uid), random.Random(0)
        ),
        "record drift",
    )
    print("CAPTION_EXCEPTIONS_V1_PASS positives=8 negatives=8 exposures=32")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(
        "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop"
    )
    parser.add_argument(
        "--plan-header",
        type=Path,
        default=base / "offline_arb_plans_v1/w64_prod/plan/header.json",
    )
    parser.add_argument(
        "--catalog-header",
        type=Path,
        default=base / "offline_arb_catalog_v1/header.json",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=base / "offline_arb_planned_caption_availability_audit_v1/audit.json",
    )
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
