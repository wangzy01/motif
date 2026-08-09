"""Validate the sealed Stage512a-control plan/registry bridge evidence.

This v1 artifact is an immutable evidence link from the fresh 256M plan to the
reviewed physical exceptions.  It is intentionally not a substitute for the
post-wiring v2 full semantic scan used by the formal launch gate.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from training.dataline_offline.anime_native_upsample_exceptions_control_rngfix_v1 import (
    CATALOG_HASH,
    CATALOG_HEADER_SHA256,
    CLOSURE,
    PLAN_HASH,
    PLAN_HEADER_SHA256,
    PLAN_ID,
    STAGE512A_CONTROL_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS,
    bind_stage512a_control_anime_native_upsample_exceptions,
)
from training.dataline_offline.caption_exceptions_v1 import (
    STAGE512A_CAPTIONLESS_MONET_ROWS,
    bind_caption_exceptions,
)
from training.dataline_offline.image_exceptions_v1 import (
    STAGE512A_CONTROL_TEXTATLAS_FORMAL_EXPOSURES,
    STAGE512A_CONTROL_TEXTATLAS_JPEG_EOI_REPAIRS,
    bind_image_exceptions,
)


DEFAULT_ROOT = Path(
    "/fsx/evan/audits/stage512a_continue_control_rngfix_v1/"
    "plan_closure_v1_20260809"
)
CANONICAL_PLAN_RECEIPT_SHA256 = (
    "5178207100c8af20230489dac88b9118665c93abb33ab5c97b6ccc532b2ea6ca"
)
CLOSURE_REPORT_SHA256 = (
    "cb92872fc8f13cae10bfafaeb3238c9cc0159bf0c31a096e85ee9b188eb53104"
)
SUCCESS_SHA256 = "0f83e65b88dc2c138c36cc1a80ddb84c937c8fca899d7bbf1c90d60b6c9a285b"
JOB_LOG_SHA256 = "8395cc72d6db5b6739f8528c1c629ed31d7f28c5fb863a4964cbb308ab2f99d9"
CHECKSUMS_SHA256 = "aa8283436369f3ddc2f2c7648c28f56cb8b8619b2e87b660f7b2ca29847fc8f0"

_MEMBERS = {
    "CANONICAL_PLAN_RECEIPT.json": CANONICAL_PLAN_RECEIPT_SHA256,
    "CLOSURE_REPORT.json": CLOSURE_REPORT_SHA256,
    "_SUCCESS": SUCCESS_SHA256,
    "job.log": JOB_LOG_SHA256,
}
_TEXTATLAS_COORDINATES = {
    68_973_396: ((34_504, 20, 31), (51_961, 41, 25), (79_902, 27, 31)),
    69_010_633: ((36_075, 29, 29), (76_084, 12, 30), (105_118, 39, 28)),
    70_063_978: ((32_950, 9, 30), (77_400, 42, 27), (98_443, 17, 25)),
}
_ANIME_COORDINATES = (
    (11_486, 35, 1),
    (50_429, 7, 2),
    (60_035, 46, 0),
    (95_244, 16, 1),
    (119_964, 33, 0),
    (121_904, 58, 1),
)


class Stage512aControlRegistryClosureError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage512aControlRegistryClosureError(message)


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(type(key) is str and key not in result, "duplicate/non-string JSON key")
        result[key] = value
    return result


def _decode_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage512aControlRegistryClosureError(
            f"invalid UTF-8 JSON in {label}"
        ) from exc
    _require(type(value) is dict, f"{label} must be a JSON object")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _project_dataclass(value: Any) -> dict[str, Any]:
    return {
        field.name: getattr(value, field.name)
        for field in dataclasses.fields(value)
        if field.name != "coordinates"
    }


def validate_closure_payloads(
    receipt: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    """Pure validation used by both filesystem gates and mutation tests."""

    _require(
        receipt.get("schema")
        == "stage512a_continue_control_plan_canonical_receipt_v1"
        and receipt.get("status") == "PASS",
        "canonical receipt schema/status drift",
    )
    _require(
        (
            receipt.get("plan_id"),
            receipt.get("plan_hash"),
            receipt.get("header_sha256"),
            receipt.get("checksums_sha256"),
        )
        == (PLAN_ID, PLAN_HASH, PLAN_HEADER_SHA256, CHECKSUMS_SHA256),
        "canonical receipt plan identity drift",
    )
    _require(
        (
            receipt.get("total_steps"),
            receipt.get("world_size"),
            receipt.get("per_rank_bs"),
            receipt.get("total_exposures"),
        )
        == (125_000, 64, 32, 256_000_000),
        "canonical receipt geometry drift",
    )
    full = receipt.get("full_256m_closure")
    _require(type(full) is dict, "canonical receipt closure is not an object")
    _require(
        full.get("sha256") == CLOSURE_REPORT_SHA256,
        "canonical receipt does not bind the closure report",
    )
    _require(
        full.get("textatlas_exposure_counts")
        == {str(key): value for key, value in STAGE512A_CONTROL_TEXTATLAS_FORMAL_EXPOSURES.items()},
        "receipt TextAtlas exposure counts drift",
    )
    _require(
        full.get("anime_exposure_counts") == {"11163647": 6},
        "receipt Anime exposure counts drift",
    )
    _require(
        full.get("captionless_monet_exposure_counts")
        == {str(key): 0 for key in STAGE512A_CAPTIONLESS_MONET_ROWS},
        "receipt captionless-MONET counts drift",
    )

    _require(
        report.get("schema") == "stage512a_continue_control_plan_closure_v1"
        and report.get("status") == "PASS",
        "closure report schema/status drift",
    )
    _require(
        (
            report.get("plan_id"),
            report.get("plan_hash"),
            report.get("header_sha256"),
            report.get("checksums_sha256"),
        )
        == (PLAN_ID, PLAN_HASH, PLAN_HEADER_SHA256, CHECKSUMS_SHA256),
        "closure report plan identity drift",
    )

    textatlas = report.get("textatlas_jpeg_repairs")
    _require(type(textatlas) is dict, "TextAtlas closure is not an object")
    _require(
        set(textatlas) == {str(key) for key in STAGE512A_CONTROL_TEXTATLAS_JPEG_EOI_REPAIRS},
        "TextAtlas closure row set drift",
    )
    for row_id, repair in STAGE512A_CONTROL_TEXTATLAS_JPEG_EOI_REPAIRS.items():
        entry = textatlas[str(row_id)]
        _require(type(entry) is dict, f"TextAtlas row {row_id} is not an object")
        for name, expected in _project_dataclass(repair).items():
            _require(entry.get(name) == expected, f"TextAtlas row {row_id}/{name} drift")
        _require(
            entry.get("realized_exposure_count")
            == STAGE512A_CONTROL_TEXTATLAS_FORMAL_EXPOSURES[row_id]
            and entry.get("realized_coordinates")
            == [list(item) for item in _TEXTATLAS_COORDINATES[row_id]],
            f"TextAtlas row {row_id} coordinate closure drift",
        )

    anime = report.get("anime_native_upsample")
    _require(type(anime) is dict and set(anime) == {"11163647"}, "Anime row set drift")
    anime_entry = anime["11163647"]
    _require(type(anime_entry) is dict, "Anime closure row is not an object")
    anime_runtime = STAGE512A_CONTROL_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS[11_163_647]
    for name, expected in _project_dataclass(anime_runtime).items():
        _require(anime_entry.get(name) == expected, f"Anime row field {name} drift")
    _require(
        anime_entry.get("realized_exposure_count") == 6
        and anime_entry.get("realized_coordinates")
        == [list(item) for item in _ANIME_COORDINATES]
        and CLOSURE == (6, 125_000, 64, 32),
        "Anime coordinate/runtime closure drift",
    )

    monet = report.get("captionless_monet_zero_hit_rows")
    _require(
        type(monet) is dict
        and set(monet) == {str(key) for key in STAGE512A_CAPTIONLESS_MONET_ROWS},
        "captionless-MONET row set drift",
    )
    for row_id, uid in STAGE512A_CAPTIONLESS_MONET_ROWS.items():
        _require(
            monet[str(row_id)]
            == {"uid": uid, "realized_exposure_count": 0},
            f"captionless-MONET row {row_id} is not sealed zero-hit evidence",
        )


def validate_runtime_bindings() -> None:
    """Prove the sealed rows are what the production binders now expose."""

    common = {
        "stage": "512a",
        "eligibility": "elig_512",
        "plan_id": PLAN_ID,
        "catalog_hash": CATALOG_HASH,
        "plan_header_sha256": PLAN_HEADER_SHA256,
        "catalog_header_sha256": CATALOG_HEADER_SHA256,
    }
    image = bind_image_exceptions(
        **common,
        plan_hash=PLAN_HASH,
        decode_domain=lambda _row_id: "textatlas",
    )
    _require(
        image is STAGE512A_CONTROL_TEXTATLAS_JPEG_EOI_REPAIRS,
        "fresh TextAtlas runtime profile is not the sealed mapping",
    )
    anime = bind_stage512a_control_anime_native_upsample_exceptions(
        **common,
        plan_hash=PLAN_HASH,
        decode_domain=lambda _row_id: "anime",
    )
    _require(
        anime is STAGE512A_CONTROL_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS,
        "fresh Anime runtime profile is not the sealed mapping",
    )
    caption = bind_caption_exceptions(
        **common,
        decode_domain=lambda _row_id: "monet_real",
    )
    _require(not caption, "fresh plan must not receive a captionless-MONET waiver")


def validate_stage512a_control_registry_closure(
    root: Path = DEFAULT_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify sealed objects, payloads, and current production binders."""

    root = Path(root)
    root_stat = os.lstat(root)
    _require(
        stat.S_ISDIR(root_stat.st_mode)
        and not root.is_symlink()
        and stat.S_IMODE(root_stat.st_mode) == 0o555,
        "closure root must be a real 0555 directory",
    )
    _require(set(path.name for path in root.iterdir()) == set(_MEMBERS), "member-set drift")
    raw: dict[str, bytes] = {}
    identities: dict[str, tuple[int, int, int, int, int]] = {}
    for name, expected_sha in _MEMBERS.items():
        path = root / name
        before = os.lstat(path)
        _require(
            stat.S_ISREG(before.st_mode)
            and not path.is_symlink()
            and before.st_nlink == 1
            and stat.S_IMODE(before.st_mode) == 0o444,
            f"sealed member object/mode/link drift: {name}",
        )
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        payload = path.read_bytes()
        after = os.lstat(path)
        _require(
            identity
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            f"sealed member changed while read: {name}",
        )
        _require(_sha256(payload) == expected_sha, f"sealed member SHA drift: {name}")
        raw[name] = payload
        identities[name] = identity
    _require(
        raw["_SUCCESS"] == (CANONICAL_PLAN_RECEIPT_SHA256 + "\n").encode("ascii"),
        "success marker payload drift",
    )
    receipt = _decode_json(raw["CANONICAL_PLAN_RECEIPT.json"], "canonical receipt")
    report = _decode_json(raw["CLOSURE_REPORT.json"], "closure report")
    validate_closure_payloads(receipt, report)
    validate_runtime_bindings()
    return receipt, report


__all__ = [
    "CANONICAL_PLAN_RECEIPT_SHA256",
    "CHECKSUMS_SHA256",
    "CLOSURE_REPORT_SHA256",
    "DEFAULT_ROOT",
    "Stage512aControlRegistryClosureError",
    "validate_closure_payloads",
    "validate_runtime_bindings",
    "validate_stage512a_control_registry_closure",
]
