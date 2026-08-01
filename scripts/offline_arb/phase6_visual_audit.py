#!/usr/bin/env python3
"""OFFLINE_ARB_V1 Phase-6 visual audit (quantitative bbox audit + contact sheets).

This tool intentionally measures only the two runnable crop paths requested for the
final visual audit:

* legacy square center crop;
* OFFLINE_ARB_V1 exact assignment, floor=0.005 folding, and per-domain v_anchor.

The archived streaming ARB implementation is documentary context only.  It is not
executed and no synthetic streaming metric is emitted.

``quant`` reads narrow columns from source parquet row groups.  It never reads image
bytes.  ``sheets`` fetches a deliberately small, AR-stratified image sample and draws
the two crop windows on thumbnails.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import hashlib
import json
import math
import os
import random
import sys
import textwrap
import unittest
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DOMAINS = ("anime", "monet_real", "gpic", "synthetic", "textatlas")
AR_BANDS = ("portrait", "near_square", "landscape")
ALL_BAND = "all"
MONET_REAL_SUBSOURCES = (
    "cc12m",
    "commoncatalog-cc-by",
    "coyo",
    "diffusion-aesthetic-4k",
    "laion",
    "megalith10m",
)
DEFAULT_CONFIG = "configs/stage256_redesign_naifu_arb_v1.yaml"
DEFAULT_N_MATRIX = (
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_arb_bucketstats_v1/N_matrix.parquet"
)
DEFAULT_OUTPUT = (
    "/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/"
    "offline_arb_vaudit_v1"
)
DEFAULT_MONET_SOURCE_ROOT = "/s3-mewtant-raw-datasets/jasperai/monet/v1.1.0"
DEFAULT_FACE_GATE_DELTA = 0.15
# D10 冻结(2026-07-10): portrait < 0.8, near-square [0.8, 1.25], landscape > 1.25.
AR_PORTRAIT_MAX = 0.8
AR_LANDSCAPE_MIN = 1.25
VISIBILITY_WARNING_RATE = 0.01

AR_BALANCED_RETENTION_DEFINITION = (
    "Existing bbox-level aggregate over a deterministic audit sample with equal "
    "image-row quotas in the three source-AR bands."
)
NATURAL_FREQUENCY_RETENTION_DEFINITION = (
    "Σ_band w_band × per-band bbox full-retention rate, where w_band "
    "is the exact eligible MONET rich-pool image-row frequency."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    os.replace(tmp, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(value)
    os.replace(tmp, path)


def finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def ar_band(width: float, height: float, portrait_max: float, landscape_min: float) -> str:
    if not finite_positive(width) or not finite_positive(height):
        raise ValueError(f"invalid dimensions: {width}x{height}")
    aspect = float(width) / float(height)
    if aspect < portrait_max:
        return "portrait"
    if aspect > landscape_min:
        return "landscape"
    return "near_square"


def crop_offset(excess: int, anchor: float = 0.5) -> int:
    """Exact copy of training.dataline.image_fetch.crop_offset's integer semantics."""
    if excess <= 0:
        return 0
    if anchor == 0.5:
        return excess // 2
    return max(0, min(int(excess), int(round(excess * float(anchor)))))


@dataclass(frozen=True)
class Window:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_list(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass(frozen=True)
class CropGeometry:
    window: Window
    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    target_width: int
    target_height: int
    crop_left: int
    crop_top: int
    v_anchor: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["window"] = self.window.as_list()
        return result


@dataclass(frozen=True)
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height


def _geometry_from_resized(
    source_width: int,
    source_height: int,
    resized_width: int,
    resized_height: int,
    target_width: int,
    target_height: int,
    crop_left: int,
    crop_top: int,
    v_anchor: float,
) -> CropGeometry:
    # PIL resizes independently to the rounded integer dimensions.  Mapping the crop
    # back with per-axis scales therefore matches the actual pixels more closely than
    # reusing the pre-rounding nominal scale.
    sx = resized_width / source_width
    sy = resized_height / source_height
    right = min(resized_width, crop_left + target_width)
    bottom = min(resized_height, crop_top + target_height)
    window = Window(
        max(0.0, crop_left / sx),
        max(0.0, crop_top / sy),
        min(float(source_width), right / sx),
        min(float(source_height), bottom / sy),
    )
    return CropGeometry(
        window=window,
        source_width=source_width,
        source_height=source_height,
        resized_width=resized_width,
        resized_height=resized_height,
        target_width=target_width,
        target_height=target_height,
        crop_left=crop_left,
        crop_top=crop_top,
        v_anchor=float(v_anchor),
    )


def old_square_geometry(width: int, height: int, target: int) -> CropGeometry:
    """Exact legacy resize_crop_uint8 inclusion window (center crop, anchor=0.5)."""
    if width <= 0 or height <= 0 or target <= 0:
        raise ValueError("width, height, and target must be positive")
    resized_width, resized_height = int(width), int(height)
    if min(width, height) > target:
        scale = target / min(width, height)
        resized_width = max(target, round(width * scale))
        resized_height = max(target, round(height * scale))
    left = max(0, (resized_width - target) // 2)
    top = crop_offset(resized_height - target, 0.5)
    return _geometry_from_resized(
        width,
        height,
        resized_width,
        resized_height,
        target,
        target,
        left,
        top,
        0.5,
    )


def new_bucket_geometry(
    width: int,
    height: int,
    target_width: int,
    target_height: int,
    v_anchor: float,
) -> CropGeometry:
    """Exact resize-to-cover + integer offset geometry from resize_crop_to_bucket."""
    if width <= 0 or height <= 0 or target_width <= 0 or target_height <= 0:
        raise ValueError("source and target dimensions must be positive")
    scale = max(target_width / width, target_height / height)
    resized_width = max(target_width, round(width * scale))
    resized_height = max(target_height, round(height * scale))
    left = max(0, (resized_width - target_width) // 2)
    top = crop_offset(resized_height - target_height, v_anchor)
    return _geometry_from_resized(
        width,
        height,
        resized_width,
        resized_height,
        target_width,
        target_height,
        left,
        top,
        v_anchor,
    )


def clip_bbox(raw: BBox, width: float, height: float) -> tuple[BBox | None, bool]:
    values = (raw.xmin, raw.ymin, raw.xmax, raw.ymax, width, height)
    if not all(math.isfinite(float(value)) for value in values):
        return None, False
    out_of_bounds = (
        raw.xmin < 0.0
        or raw.ymin < 0.0
        or raw.xmax > width
        or raw.ymax > height
    )
    clipped = BBox(
        max(0.0, min(float(width), raw.xmin)),
        max(0.0, min(float(height), raw.ymin)),
        max(0.0, min(float(width), raw.xmax)),
        max(0.0, min(float(height), raw.ymax)),
    )
    if clipped.area <= 0:
        return None, out_of_bounds
    return clipped, out_of_bounds


def bbox_retention(bbox: BBox, window: Window, epsilon: float = 1e-9) -> dict[str, Any]:
    ix1 = max(bbox.xmin, window.xmin)
    iy1 = max(bbox.ymin, window.ymin)
    ix2 = min(bbox.xmax, window.xmax)
    iy2 = min(bbox.ymax, window.ymax)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    retained = min(1.0, max(0.0, intersection / bbox.area)) if bbox.area else 0.0
    top_cut = bbox.ymin < window.ymin - epsilon
    bottom_cut = bbox.ymax > window.ymax + epsilon
    left_cut = bbox.xmin < window.xmin - epsilon
    right_cut = bbox.xmax > window.xmax + epsilon
    return {
        "retained_area_ratio": retained,
        "full_retained": retained >= 1.0 - epsilon,
        "top_cut": top_cut,
        "bottom_cut": bottom_cut,
        "left_cut": left_cut,
        "right_cut": right_cut,
        "head_foot_joint_retained": not top_cut and not bottom_cut,
    }


class GeometryFixtureTests(unittest.TestCase):
    """Hand-computed pure-math fixtures; no image or dataset access."""

    def assertWindowAlmostEqual(self, got: Window, expected: Sequence[float]) -> None:  # noqa: N802
        for actual, wanted in zip(got.as_list(), expected):
            self.assertAlmostEqual(actual, wanted, places=7)

    def test_01_old_landscape_center_square(self) -> None:
        geom = old_square_geometry(400, 200, 100)
        self.assertEqual((geom.resized_width, geom.resized_height), (200, 100))
        self.assertWindowAlmostEqual(geom.window, (100, 0, 300, 200))

    def test_02_old_portrait_center_square(self) -> None:
        geom = old_square_geometry(200, 400, 100)
        self.assertEqual((geom.crop_left, geom.crop_top), (0, 50))
        self.assertWindowAlmostEqual(geom.window, (0, 100, 200, 300))

    def test_03_anime_top_anchor(self) -> None:
        geom = new_bucket_geometry(200, 400, 100, 100, 0.15)
        self.assertEqual((geom.resized_width, geom.resized_height), (100, 200))
        self.assertEqual(geom.crop_top, 15)
        self.assertWindowAlmostEqual(geom.window, (0, 30, 200, 230))

    def test_04_matching_bucket_keeps_full_image(self) -> None:
        geom = new_bucket_geometry(400, 200, 200, 100, 0.5)
        self.assertWindowAlmostEqual(geom.window, (0, 0, 400, 200))

    def test_05_bbox_top_half_cut(self) -> None:
        result = bbox_retention(BBox(20, 20, 80, 80), Window(0, 50, 100, 100))
        self.assertAlmostEqual(result["retained_area_ratio"], 0.5)
        self.assertTrue(result["top_cut"])
        self.assertFalse(result["bottom_cut"])
        self.assertFalse(result["head_foot_joint_retained"])

    def test_06_bbox_bottom_half_and_full(self) -> None:
        bbox = BBox(10, 10, 90, 90)
        cut = bbox_retention(bbox, Window(0, 0, 100, 50))
        full = bbox_retention(bbox, Window(0, 0, 100, 100))
        self.assertAlmostEqual(cut["retained_area_ratio"], 0.5)
        self.assertTrue(cut["bottom_cut"])
        self.assertTrue(full["full_retained"])
        self.assertTrue(full["head_foot_joint_retained"])

    def test_07_natural_frequency_weighted_retention(self) -> None:
        weighted = _natural_frequency_weighted(
            {"portrait": 0.5, "near_square": 0.3, "landscape": 0.2},
            {
                "old_full_retention_rate": {
                    "portrait": 0.2,
                    "near_square": 0.6,
                    "landscape": 0.9,
                },
                "new_full_retention_rate": {
                    "portrait": 0.4,
                    "near_square": 0.7,
                    "landscape": 1.0,
                },
                "delta_full_retention_rate": {
                    "portrait": 0.2,
                    "near_square": 0.1,
                    "landscape": 0.1,
                },
            },
        )
        self.assertAlmostEqual(weighted["old_full_retention_rate"], 0.46)
        self.assertAlmostEqual(weighted["new_full_retention_rate"], 0.61)
        self.assertAlmostEqual(weighted["delta_full_retention_rate"], 0.15)

    def test_08_visibility_warning_strictly_above_one_percent(self) -> None:
        result = {
            "domain": "synthetic_fixture",
            "joined_rows_total": 100,
            "counters": {
                "person_malformed_boxes": 1,
                "person_raw_boxes": 100,
                "yolo_json_parse_error_rows": 1,
                "face_raw_oob_boxes": 1,
                "face_raw_boxes": 100,
            },
        }
        self.assertEqual(_visibility_warning_metrics(result), [])
        result["counters"]["person_malformed_boxes"] = 2
        self.assertEqual(
            _visibility_warning_metrics(result),
            ["person_malformed_boxes/person_raw_boxes"],
        )


@dataclass
class BucketPolicy:
    config_path: str
    n_matrix_path: str
    stage: str
    floor: float
    image_size: int
    buckets: list[tuple[int, int]]
    square_idx: int
    kept: frozenset[int]
    fold_map: list[int]
    raw_bucket_probs: list[float]
    square_only_domains: frozenset[str]
    crop_v_anchor: dict[str, float]
    domain_mix: dict[str, float]
    sampling_config: dict[str, Any]
    mar_params: dict[str, Any]
    _assign_fn: Callable[..., int] = field(repr=False)

    def anchor_for(self, domain: str) -> float:
        return float(self.crop_v_anchor.get(domain, 0.5))

    def assigned_bucket(self, domain: str, width: int, height: int) -> tuple[int, int]:
        if domain in self.square_only_domains:
            raw = self.square_idx
        else:
            raw = int(
                self._assign_fn(
                    width,
                    height,
                    self.buckets,
                    int(self.mar_params["target_area"]),
                    int(self.mar_params["divisor"]),
                    bool(self.mar_params["adaptive_no_upscale"]),
                    float(self.mar_params["adaptive_max_aspect"]),
                )
            )
        return raw, int(self.fold_map[raw])

    def new_geometry(self, domain: str, width: int, height: int) -> tuple[int, CropGeometry]:
        _raw, assigned = self.assigned_bucket(domain, width, height)
        target_h, target_w = self.buckets[assigned]
        geometry = new_bucket_geometry(
            width, height, target_w, target_h, self.anchor_for(domain)
        )
        return assigned, geometry

    def metadata(self) -> dict[str, Any]:
        return {
            "config": self.config_path,
            "n_matrix": self.n_matrix_path,
            "stage": self.stage,
            "floor": self.floor,
            "image_size": self.image_size,
            "full_bucket_count": len(self.buckets),
            "kept_bucket_count": len(self.kept),
            "kept_bucket_indices": sorted(self.kept),
            "kept_buckets": [
                {
                    "index": idx,
                    "height": self.buckets[idx][0],
                    "width": self.buckets[idx][1],
                    "ar": self.buckets[idx][1] / self.buckets[idx][0],
                    "probability_before_fold": self.raw_bucket_probs[idx],
                }
                for idx in sorted(self.kept)
            ],
            "square_bucket_idx": self.square_idx,
            "fold_map": self.fold_map,
            "square_only_domains": sorted(self.square_only_domains),
            "crop_v_anchor": {domain: self.anchor_for(domain) for domain in DOMAINS},
            "domain_mix": self.domain_mix,
            "multi_ar": self.mar_params,
        }


def _resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def load_bucket_policy(config_path: str, n_matrix_path: str, stage: str, floor: float) -> BucketPolicy:
    from training.dataline import anchors_v1
    from training.dataline.multi_ar import (
        assign_adaptive_bucket,
        build_adaptive_buckets,
        nearest_allowed_bucket,
        square_index,
    )

    config_file = _resolve_repo_path(config_path)
    with config_file.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dataset_cfg = config["dataset"]
    mar = dataset_cfg["multi_ar"]
    mar_params = {
        "target_area": int(mar["target_area"]),
        "divisor": int(mar["divisor"]),
        "adaptive_no_upscale": bool(mar["adaptive_no_upscale"]),
        "adaptive_min_area_ratio": float(mar["adaptive_min_area_ratio"]),
        "adaptive_max_aspect": float(mar["adaptive_max_aspect"]),
    }
    if str(mar.get("bucket_mode") or "adaptive") == anchors_v1.BUCKET_MODE:
        # Refuse rather than half-port. This audit re-derives assignment itself with
        # assign_adaptive_bucket and force-pins square_idx for square-only domains with
        # no feasibility test, so under anchors_v1 it would disagree with the catalog
        # row-for-row AND would itself upsample. A wrong visual audit is worse than none;
        # it gates nothing in the build chain.
        raise RuntimeError(
            f"phase6_visual_audit does not support bucket_mode={anchors_v1.BUCKET_MODE}: "
            "its independent assignment path has no feasible-anchor test and would "
            "disagree with the catalog. Port it before auditing an anchors catalog."
        )
    buckets = build_adaptive_buckets(
        mar_params["target_area"],
        mar_params["divisor"],
        min_area_ratio=mar_params["adaptive_min_area_ratio"],
        max_aspect=mar_params["adaptive_max_aspect"],
    )
    square_idx = square_index(buckets)

    sampling_path = _resolve_repo_path(str(dataset_cfg["sampling_config"]))
    with sampling_path.open(encoding="utf-8") as handle:
        sampling_config = json.load(handle)
    domain_mix = {
        domain: float(sampling_config["domain_mix"][str(stage)][domain]) for domain in DOMAINS
    }

    matrix_file = _resolve_repo_path(n_matrix_path)
    table = pq.read_table(matrix_file)
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    h_col = "bucket_h" if "bucket_h" in columns else "h"
    w_col = "bucket_w" if "bucket_w" in columns else "w"
    if len(columns.get(h_col, [])) != len(buckets):
        raise RuntimeError(
            f"N_matrix has {len(columns.get(h_col, []))} buckets, config rebuild has {len(buckets)}"
        )
    for idx, expected in enumerate(buckets):
        observed = (int(columns[h_col][idx]), int(columns[w_col][idx]))
        if observed != expected:
            raise RuntimeError(f"N_matrix bucket {idx}={observed}, config rebuild={expected}")

    counts = [
        [int(columns[f"n_{domain}"][idx]) for domain in DOMAINS]
        for idx in range(len(buckets))
    ]
    totals = [sum(row[j] for row in counts) for j in range(len(DOMAINS))]
    probabilities: list[float] = []
    for idx in range(len(buckets)):
        probability = sum(
            domain_mix[domain] * counts[idx][j] / max(totals[j], 1)
            for j, domain in enumerate(DOMAINS)
        )
        probabilities.append(float(probability))
    kept = {idx for idx, probability in enumerate(probabilities) if probability >= float(floor)}
    kept.add(square_idx)
    frozen_kept = frozenset(kept)
    fold_map = [
        idx if idx in kept else nearest_allowed_bucket(idx, buckets, frozen_kept)
        for idx in range(len(buckets))
    ]
    if floor == 0.005 and len(buckets) == 95 and len(kept) != 20:
        raise RuntimeError(
            f"canonical floor=0.005 should retain 20/95 buckets, observed {len(kept)}; "
            "check N_matrix/config drift"
        )

    crop_v_anchor = {
        str(domain): float(value)
        for domain, value in (dataset_cfg.get("crop_v_anchor") or {}).items()
    }
    return BucketPolicy(
        config_path=str(config_file),
        n_matrix_path=str(matrix_file),
        stage=str(stage),
        floor=float(floor),
        image_size=int(dataset_cfg["image_size"]),
        buckets=buckets,
        square_idx=square_idx,
        kept=frozen_kept,
        fold_map=fold_map,
        raw_bucket_probs=probabilities,
        square_only_domains=frozenset(str(x) for x in mar["square_only_domains"]),
        crop_v_anchor=crop_v_anchor,
        domain_mix=domain_mix,
        sampling_config=sampling_config,
        mar_params=mar_params,
        _assign_fn=assign_adaptive_bucket,
    )


def list_monet_source_parquets(root: str, synthetic: bool = False) -> list[str]:
    if synthetic:
        patterns = [os.path.join(root, "synthetic", "*", "*", "*.parquet")]
    else:
        patterns = [os.path.join(root, source, "*", "*.parquet") for source in MONET_REAL_SUBSOURCES]
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return sorted(set(paths))


def _manifest_paths(policy: BucketPolicy) -> dict[str, Any]:
    manifest = policy.sampling_config["manifest"]
    roots = manifest["real_synth_roots"]
    anime_source = str(manifest["anime_caption_source"]).split()[0]
    return {
        "monet_real_manifest": str(roots["monet_real"]),
        "gpic_manifest": str(roots["gpic"]),
        "synthetic_manifest": str(roots["synthetic"]),
        "anime_source": anime_source,
        "textatlas_manifest": str(manifest["textatlas"]),
        "textatlas_source": str(manifest["textatlas_source"]),
        "textatlas_subsets": [str(x) for x in manifest["textatlas_subsets"]],
    }


def _parse_json_array(raw: Any) -> list[Any]:
    if raw is None or raw == "":
        return []
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, list):
        raise ValueError(f"expected JSON list, got {type(value).__name__}")
    return value


def probe_detection_file(path: str, domain: str) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    requested = (
        "detection-face_mediapipe",
        "detection_yolo",
        "classifier_yolo",
    )
    present = [name for name in requested if name in names]
    result: dict[str, Any] = {
        "domain": domain,
        "path": path,
        "num_rows": pf.metadata.num_rows,
        "num_row_groups": pf.metadata.num_row_groups,
        "fields": {
            name: str(pf.schema_arrow.field(name).type) for name in present
        },
        "missing_detection_fields": [name for name in requested if name not in names],
        "fetch_free_quant_available": bool(
            "detection-face_mediapipe" in names or "detection_yolo" in names
        ),
    }
    if not present:
        result["interpretation"] = "no detection/classifier columns in parquet footer"
        return result

    cols = [name for name in ("id", "width", "height", *present) if name in names]
    sample = pf.read_row_group(0, columns=cols, use_threads=False)
    result["probed_row_group"] = 0
    result["probed_rows"] = sample.num_rows
    interpretations: dict[str, Any] = {}
    for name in present:
        values = sample.column(name).to_pylist()
        first: Any = None
        for raw in values:
            try:
                parsed = _parse_json_array(raw)
            except Exception:  # recorded by the full quant parser later
                continue
            if parsed:
                first = parsed[0]
                break
        if name == "detection-face_mediapipe":
            interpretations[name] = {
                "encoding": "JSON list in Arrow string",
                "bbox": "named xyxy {xmin,ymin,xmax,ymax}",
                "coordinates": "normalized to source width/height",
                "class": "implicit face",
                "keypoints": "six normalized [x,y] pairs when present",
                "observed_first_item_keys": sorted(first) if isinstance(first, dict) else None,
            }
        elif name == "detection_yolo":
            interpretations[name] = {
                "encoding": "JSON list in Arrow string",
                "bbox": "[xmin,ymin,xmax,ymax] continuous xyxy",
                "coordinates": "absolute source-image pixels",
                "class_field": "name (person is exact string 'person')",
                "observed_first_item_keys": sorted(first) if isinstance(first, dict) else None,
            }
        else:
            interpretations[name] = {
                "encoding": "JSON list in Arrow string",
                "content": "whole-image top-k labels {name,prob}; prob is a string",
                "geometry_use": "none (no bbox)",
                "observed_first_item_keys": sorted(first) if isinstance(first, dict) else None,
            }
    result["interpretation"] = interpretations
    return result


def _linear_quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


@dataclass
class MetricAccumulator:
    retained: list[float] = field(default_factory=list)
    full_retained: int = 0
    top_cut: int = 0
    bottom_cut: int = 0
    left_cut: int = 0
    right_cut: int = 0
    vertical_both_cut: int = 0
    head_foot_joint_retained: int = 0

    def add(self, result: dict[str, Any]) -> None:
        self.retained.append(float(result["retained_area_ratio"]))
        self.full_retained += int(result["full_retained"])
        self.top_cut += int(result["top_cut"])
        self.bottom_cut += int(result["bottom_cut"])
        self.left_cut += int(result["left_cut"])
        self.right_cut += int(result["right_cut"])
        self.vertical_both_cut += int(result["top_cut"] and result["bottom_cut"])
        self.head_foot_joint_retained += int(result["head_foot_joint_retained"])

    def summary(self) -> dict[str, Any]:
        count = len(self.retained)
        if not count:
            return {
                "bbox_count": 0,
                "full_retention_rate": None,
                "truncation_rate": None,
                "mean_retained_area_ratio": None,
                "p05_retained_area_ratio": None,
                "head_foot_joint_retention_rate": None,
                "top_cut_rate": None,
                "bottom_cut_rate": None,
                "top_only_rate": None,
                "bottom_only_rate": None,
                "top_and_bottom_rate": None,
                "vertical_cut_rate": None,
                "top_only_share_of_vertical_cuts": None,
                "bottom_only_share_of_vertical_cuts": None,
                "top_and_bottom_share_of_vertical_cuts": None,
                "left_cut_rate": None,
                "right_cut_rate": None,
            }
        top_only = self.top_cut - self.vertical_both_cut
        bottom_only = self.bottom_cut - self.vertical_both_cut
        vertical_any = top_only + bottom_only + self.vertical_both_cut
        return {
            "bbox_count": count,
            "full_retention_rate": self.full_retained / count,
            "truncation_rate": 1.0 - self.full_retained / count,
            "mean_retained_area_ratio": sum(self.retained) / count,
            "p05_retained_area_ratio": _linear_quantile(self.retained, 0.05),
            "head_foot_joint_retention_rate": self.head_foot_joint_retained / count,
            "top_cut_rate": self.top_cut / count,
            "bottom_cut_rate": self.bottom_cut / count,
            "top_only_rate": top_only / count,
            "bottom_only_rate": bottom_only / count,
            "top_and_bottom_rate": self.vertical_both_cut / count,
            "vertical_cut_rate": vertical_any / count,
            "top_only_share_of_vertical_cuts": (
                top_only / vertical_any if vertical_any else None
            ),
            "bottom_only_share_of_vertical_cuts": (
                bottom_only / vertical_any if vertical_any else None
            ),
            "top_and_bottom_share_of_vertical_cuts": (
                self.vertical_both_cut / vertical_any if vertical_any else None
            ),
            "left_cut_rate": self.left_cut / count,
            "right_cut_rate": self.right_cut / count,
        }


def _balanced_quotas(total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(AR_BANDS))
    return {
        band: base + int(index < remainder) for index, band in enumerate(AR_BANDS)
    }


def _add_bbox_metrics(
    accumulators: dict[tuple[str, str, str], MetricAccumulator],
    kind: str,
    band: str,
    bbox: BBox,
    old_window: Window,
    new_window: Window,
) -> None:
    for scheme, window in (("old_square", old_window), ("offline_exact", new_window)):
        result = bbox_retention(bbox, window)
        accumulators[(kind, scheme, band)].add(result)
        accumulators[(kind, scheme, ALL_BAND)].add(result)


def _accumulate_detection_row(
    domain: str,
    width: int,
    height: int,
    band: str,
    face_raw: Any,
    yolo_raw: Any,
    policy: BucketPolicy,
    accumulators: dict[tuple[str, str, str], MetricAccumulator],
    counters: Counter[str],
    bucket_counts: dict[str, Counter[int]],
) -> None:
    assigned, new_geometry = policy.new_geometry(domain, width, height)
    old_geometry = old_square_geometry(width, height, policy.image_size)
    bucket_counts[band][assigned] += 1

    try:
        face_items = _parse_json_array(face_raw)
    except Exception:  # noqa: BLE001
        counters["face_json_parse_error_rows"] += 1
        face_items = []
    for item in face_items:
        counters["face_raw_boxes"] += 1
        try:
            box = item["bbox"]
            raw_box = BBox(
                float(box["xmin"]) * width,
                float(box["ymin"]) * height,
                float(box["xmax"]) * width,
                float(box["ymax"]) * height,
            )
        except (KeyError, TypeError, ValueError):
            counters["face_malformed_boxes"] += 1
            continue
        clipped, out_of_bounds = clip_bbox(raw_box, width, height)
        counters["face_raw_oob_boxes"] += int(out_of_bounds)
        if clipped is None:
            counters["face_degenerate_after_clip"] += 1
            continue
        _add_bbox_metrics(
            accumulators,
            "face",
            band,
            clipped,
            old_geometry.window,
            new_geometry.window,
        )

    try:
        yolo_items = _parse_json_array(yolo_raw)
    except Exception:  # noqa: BLE001
        counters["yolo_json_parse_error_rows"] += 1
        yolo_items = []
    for item in yolo_items:
        counters["yolo_raw_boxes_all_classes"] += 1
        if not isinstance(item, dict) or item.get("name") != "person":
            continue
        counters["person_raw_boxes"] += 1
        try:
            box = item["bbox"]
            if len(box) != 4:
                raise ValueError("bbox is not length four")
            raw_box = BBox(*(float(value) for value in box))
        except (KeyError, TypeError, ValueError):
            counters["person_malformed_boxes"] += 1
            continue
        clipped, out_of_bounds = clip_bbox(raw_box, width, height)
        counters["person_raw_oob_boxes"] += int(out_of_bounds)
        if clipped is None:
            counters["person_degenerate_after_clip"] += 1
            continue
        _add_bbox_metrics(
            accumulators,
            "person",
            band,
            clipped,
            old_geometry.window,
            new_geometry.window,
        )


def _summarize_metric_accumulators(
    domain: str,
    accumulators: dict[tuple[str, str, str], MetricAccumulator],
) -> list[dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    for kind in ("face", "person"):
        for band in (ALL_BAND, *AR_BANDS):
            for scheme in ("old_square", "offline_exact"):
                summary = accumulators[(kind, scheme, band)].summary()
                metric_rows.append(
                    {
                        "domain": domain,
                        "detection": kind,
                        "ar_band": band,
                        "scheme": scheme,
                        **summary,
                    }
                )
    return metric_rows


def quantify_detection_domain(
    domain: str,
    files: Sequence[str],
    policy: BucketPolicy,
    sample_rows: int,
    max_rows_per_source: int,
    seed: int,
    portrait_max: float,
    landscape_min: float,
    min_short_side: int,
    max_source_files: int,
) -> dict[str, Any]:
    quotas = _balanced_quotas(sample_rows)
    selected = Counter({band: 0 for band in AR_BANDS})
    accumulators: dict[tuple[str, str, str], MetricAccumulator] = defaultdict(
        MetricAccumulator
    )
    counters: Counter[str] = Counter()
    bucket_counts: dict[str, Counter[int]] = {band: Counter() for band in AR_BANDS}
    rng = random.Random(stable_seed(seed, f"quant:{domain}"))
    order = list(files)
    rng.shuffle(order)
    if max_source_files > 0:
        order = order[:max_source_files]
    scanned_files = 0
    scanned_row_groups = 0

    for path in order:
        if all(selected[band] >= quotas[band] for band in AR_BANDS):
            break
        try:
            pf = pq.ParquetFile(path)
        except Exception as exc:  # noqa: BLE001 - audit must preserve diagnostics
            counters[f"source_open_error:{type(exc).__name__}"] += 1
            continue
        names = set(pf.schema_arrow.names)
        face_col = "detection-face_mediapipe" if "detection-face_mediapipe" in names else None
        yolo_col = "detection_yolo" if "detection_yolo" in names else None
        if not face_col and not yolo_col:
            counters["source_without_detection_columns"] += 1
            continue
        needed = ["width", "height"] + [x for x in (face_col, yolo_col) if x]
        rg_order = list(range(pf.num_row_groups))
        rng.shuffle(rg_order)
        selected_this_file = 0
        scanned_files += 1
        for rg in rg_order:
            if selected_this_file >= max_rows_per_source:
                break
            if all(selected[band] >= quotas[band] for band in AR_BANDS):
                break
            try:
                table = pf.read_row_group(rg, columns=needed, use_threads=False)
            except Exception as exc:  # noqa: BLE001
                counters[f"row_group_read_error:{type(exc).__name__}"] += 1
                continue
            scanned_row_groups += 1
            rows = list(range(table.num_rows))
            rng.shuffle(rows)
            widths = table.column("width").to_pylist()
            heights = table.column("height").to_pylist()
            faces = table.column(face_col).to_pylist() if face_col else [None] * table.num_rows
            yolos = table.column(yolo_col).to_pylist() if yolo_col else [None] * table.num_rows
            for row in rows:
                if selected_this_file >= max_rows_per_source:
                    break
                width_raw, height_raw = widths[row], heights[row]
                counters["rows_considered"] += 1
                if not finite_positive(width_raw) or not finite_positive(height_raw):
                    counters["rows_invalid_dimensions"] += 1
                    continue
                width, height = int(round(float(width_raw))), int(round(float(height_raw)))
                if domain != "anime" and min(width, height) < min_short_side:
                    counters["rows_ineligible_short_side"] += 1
                    continue
                band = ar_band(width, height, portrait_max, landscape_min)
                if selected[band] >= quotas[band]:
                    continue
                selected[band] += 1
                selected_this_file += 1
                counters["rows_selected"] += 1
                _accumulate_detection_row(
                    domain,
                    width,
                    height,
                    band,
                    faces[row],
                    yolos[row],
                    policy,
                    accumulators,
                    counters,
                    bucket_counts,
                )
    complete = all(selected[b] >= quotas[b] for b in AR_BANDS)
    gate_blockers = []
    if not complete:
        gate_blockers.append("stratified_sample_incomplete")
    if counters.get("face_json_parse_error_rows", 0):
        gate_blockers.append("face_json_parse_error_rows")
    if counters.get("face_malformed_boxes", 0):
        gate_blockers.append("face_malformed_boxes")
    return {
        "domain": domain,
        "requested_rows": sample_rows,
        "quota_rows_by_ar_band": quotas,
        "selected_rows_by_ar_band": dict(selected),
        "selected_rows_total": sum(selected.values()),
        "joined_rows_by_ar_band": dict(selected),
        "joined_rows_total": sum(selected.values()),
        "complete_stratified_sample": complete,
        "gate_input_ready": not gate_blockers,
        "gate_input_blockers": gate_blockers,
        "source_files_available": len(files),
        "source_files_scanned": scanned_files,
        "row_groups_scanned": scanned_row_groups,
        "counters": dict(sorted(counters.items())),
        "assigned_bucket_counts_by_ar_band": {
            band: {str(key): value for key, value in sorted(counts.items())}
            for band, counts in bucket_counts.items()
        },
        "metrics": _summarize_metric_accumulators(domain, accumulators),
    }


@dataclass(frozen=True)
class QuantManifestCandidate:
    src_root: str
    src_shard: str
    src_key: str
    sub_source: str
    width: int
    height: int
    band: str


def _proportional_integer_quotas(
    total: int, weights: dict[str, int]
) -> dict[str, int]:
    denominator = sum(max(0, int(value)) for value in weights.values())
    if denominator <= 0:
        raise ValueError("proportional quota weights must have positive sum")
    raw = {
        key: total * max(0, int(value)) / denominator for key, value in weights.items()
    }
    quotas = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(
        weights,
        key=lambda key: (-(raw[key] - quotas[key]), MONET_REAL_SUBSOURCES.index(key)),
    )
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def _monet_source_counts(config_path: str) -> dict[str, int]:
    with _resolve_repo_path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    stats_path = _resolve_repo_path(str(config["dataset"]["stats_cache"]))
    with stats_path.open(encoding="utf-8") as handle:
        stats = json.load(handle)
    counts = {
        source: int(stats["source_counts"].get(source, 0))
        for source in MONET_REAL_SUBSOURCES
    }
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError(f"stats cache lacks positive monet source counts: {counts}")
    return counts


def _parquet_stat_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _interleaved_monet_manifests(
    manifest_files: Sequence[str], seed: int
) -> list[str]:
    """Use footer min/max to prevent a shuffled 156k-row shard from monopolizing 50k."""
    by_source: dict[str, list[str]] = {source: [] for source in MONET_REAL_SUBSOURCES}
    source_index = {source: index for index, source in enumerate(MONET_REAL_SUBSOURCES)}
    unclassified: list[str] = []
    for path in manifest_files:
        try:
            pf = pq.ParquetFile(path)
            physical_index = pf.schema.names.index("sub_source")
            stats = pf.metadata.row_group(0).column(physical_index).statistics
            if stats is None or not stats.has_min_max:
                raise ValueError("sub_source statistics have no min/max")
            low = _parquet_stat_text(stats.min)
            high = _parquet_stat_text(stats.max)
            lo, hi = source_index[low], source_index[high]
            for source in MONET_REAL_SUBSOURCES[min(lo, hi) : max(lo, hi) + 1]:
                by_source[source].append(path)
        except Exception:  # noqa: BLE001 - fall back to the tail instead of dropping data
            unclassified.append(path)
    rng = random.Random(stable_seed(seed, "quant:monet_real:manifest_order"))
    for source, values in by_source.items():
        rng.shuffle(values)
        by_source[source] = values
    result: list[str] = []
    emitted: set[str] = set()
    cursors = Counter()
    while True:
        advanced = False
        for source in MONET_REAL_SUBSOURCES:
            values = by_source[source]
            while cursors[source] < len(values) and values[cursors[source]] in emitted:
                cursors[source] += 1
            if cursors[source] < len(values):
                path = values[cursors[source]]
                cursors[source] += 1
                emitted.add(path)
                result.append(path)
                advanced = True
        if not advanced:
            break
    tail = [path for path in manifest_files if path not in emitted]
    tail.extend(path for path in unclassified if path not in emitted and path not in tail)
    rng.shuffle(tail)
    result.extend(tail)
    return result


def quantify_monet_manifest_join(
    manifest_files: Sequence[str],
    policy: BucketPolicy,
    sample_rows: int,
    max_rows_per_source: int,
    seed: int,
    portrait_max: float,
    landscape_min: float,
    min_short_side: int,
    max_manifest_files: int,
) -> dict[str, Any]:
    """Count natural AR frequency while sampling the 40M pool, then join detections."""
    domain = "monet_real"
    quotas = _balanced_quotas(sample_rows)
    source_counts = _monet_source_counts(policy.config_path)
    subsource_quotas = {
        band: _proportional_integer_quotas(quotas[band], source_counts)
        for band in AR_BANDS
    }
    selected = Counter({band: 0 for band in AR_BANDS})
    selected_cells: Counter[tuple[str, str]] = Counter()
    selected_subsources: Counter[str] = Counter()
    selected_per_source: Counter[tuple[str, str]] = Counter()
    candidates_by_source: dict[tuple[str, str], dict[str, QuantManifestCandidate]] = {}
    counters: Counter[str] = Counter()
    rng = random.Random(stable_seed(seed, "quant:monet_real:manifest"))
    files = _interleaved_monet_manifests(manifest_files, seed)
    candidate_file_limit = (
        len(files) if max_manifest_files <= 0 else min(max_manifest_files, len(files))
    )
    manifest_files_scanned = 0
    manifest_row_groups_scanned = 0
    natural_band_counts = Counter({band: 0 for band in AR_BANDS})
    natural_counters: Counter[str] = Counter()
    natural_manifest_files_scanned = 0
    natural_manifest_row_groups_scanned = 0

    def sample_complete() -> bool:
        return all(
            selected_cells[(band, source)] >= subsource_quotas[band][source]
            for band in AR_BANDS
            for source in MONET_REAL_SUBSOURCES
        )

    for file_index, path in enumerate(files):
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        frequency_cols = ["sub_source", "width", "height"]
        if "holdout" in names:
            frequency_cols.append("holdout")
        rg_order = list(range(pf.num_row_groups))
        rng.shuffle(rg_order)
        candidate_file_allowed = file_index < candidate_file_limit
        if candidate_file_allowed and not sample_complete():
            manifest_files_scanned += 1
        natural_manifest_files_scanned += 1
        for rg in rg_order:
            collect_candidates = candidate_file_allowed and not sample_complete()
            cols = list(frequency_cols)
            if collect_candidates:
                cols.extend(("src_root", "src_shard", "src_key"))
            table = pf.read_row_group(rg, columns=cols, use_threads=False)
            natural_manifest_row_groups_scanned += 1
            if collect_candidates:
                manifest_row_groups_scanned += 1
            values = {name: table.column(name).to_pylist() for name in cols}
            rows = list(range(table.num_rows))
            if collect_candidates:
                rng.shuffle(rows)
            for row in rows:
                sampling_active = collect_candidates and not sample_complete()
                natural_counters["rows_scanned"] += 1
                if sampling_active:
                    counters["manifest_rows_considered"] += 1
                if "holdout" in values and values["holdout"][row] not in (0, False, None):
                    natural_counters["rows_holdout"] += 1
                    if sampling_active:
                        counters["manifest_rows_holdout"] += 1
                    continue
                width_raw, height_raw = values["width"][row], values["height"][row]
                if not finite_positive(width_raw) or not finite_positive(height_raw):
                    natural_counters["rows_invalid_dimensions"] += 1
                    if sampling_active:
                        counters["manifest_rows_invalid_dimensions"] += 1
                    continue
                width, height = int(round(float(width_raw))), int(round(float(height_raw)))
                if min(width, height) < min_short_side:
                    natural_counters["rows_ineligible_short_side"] += 1
                    if sampling_active:
                        counters["manifest_rows_ineligible_short_side"] += 1
                    continue
                band = ar_band(width, height, portrait_max, landscape_min)
                sub_source = str(values["sub_source"][row])
                if sub_source not in source_counts:
                    natural_counters["rows_unknown_subsource"] += 1
                    if sampling_active:
                        counters["manifest_rows_unknown_subsource"] += 1
                    continue
                natural_band_counts[band] += 1
                if not sampling_active:
                    continue
                if selected_cells[(band, sub_source)] >= subsource_quotas[band][sub_source]:
                    continue
                source = (str(values["src_root"][row]), str(values["src_shard"][row]))
                if selected_per_source[source] >= max_rows_per_source:
                    counters["manifest_rows_source_cap"] += 1
                    continue
                key = str(values["src_key"][row])
                per_source = candidates_by_source.setdefault(source, {})
                if key in per_source:
                    counters["manifest_duplicate_locator"] += 1
                    continue
                per_source[key] = QuantManifestCandidate(
                    src_root=source[0],
                    src_shard=source[1],
                    src_key=key,
                    sub_source=sub_source,
                    width=width,
                    height=height,
                    band=band,
                )
                selected[band] += 1
                selected_cells[(band, sub_source)] += 1
                selected_subsources[sub_source] += 1
                selected_per_source[source] += 1
                counters["rows_selected"] += 1

    natural_eligible_rows = sum(natural_band_counts.values())
    natural_weights = {
        band: (
            natural_band_counts[band] / natural_eligible_rows
            if natural_eligible_rows
            else None
        )
        for band in AR_BANDS
    }

    accumulators: dict[tuple[str, str, str], MetricAccumulator] = defaultdict(
        MetricAccumulator
    )
    bucket_counts: dict[str, Counter[int]] = {band: Counter() for band in AR_BANDS}
    joined_bands: Counter[str] = Counter()
    joined_cells: Counter[tuple[str, str]] = Counter()
    joined_subsources: Counter[str] = Counter()
    source_files_scanned = source_row_groups_scanned = 0
    for (src_root, src_shard), wanted_map in candidates_by_source.items():
        source_path = os.path.join(src_root, src_shard)
        try:
            pf = pq.ParquetFile(source_path)
        except Exception as exc:  # noqa: BLE001
            counters[f"source_open_error:{type(exc).__name__}"] += 1
            counters["selected_rows_missing_source"] += len(wanted_map)
            continue
        names = set(pf.schema_arrow.names)
        key_col = "id" if "id" in names else "__key__"
        required = {
            key_col,
            "width",
            "height",
            "detection-face_mediapipe",
            "detection_yolo",
        }
        if not required.issubset(names):
            counters["source_missing_detection_schema"] += 1
            counters["selected_rows_missing_source"] += len(wanted_map)
            continue
        remaining = dict(wanted_map)
        source_files_scanned += 1
        for rg in range(pf.num_row_groups):
            if not remaining:
                break
            table = pf.read_row_group(
                rg,
                columns=[
                    key_col,
                    "width",
                    "height",
                    "detection-face_mediapipe",
                    "detection_yolo",
                ],
                use_threads=False,
            )
            source_row_groups_scanned += 1
            keys = table.column(key_col).to_pylist()
            widths = table.column("width").to_pylist()
            heights = table.column("height").to_pylist()
            faces = table.column("detection-face_mediapipe").to_pylist()
            yolos = table.column("detection_yolo").to_pylist()
            for row, raw_key in enumerate(keys):
                key = str(raw_key)
                candidate = remaining.pop(key, None)
                if candidate is None:
                    continue
                if not finite_positive(widths[row]) or not finite_positive(heights[row]):
                    counters["source_rows_invalid_dimensions"] += 1
                    counters["selected_rows_rejected_dimension_mismatch"] += 1
                    continue
                source_width = int(round(float(widths[row])))
                source_height = int(round(float(heights[row])))
                if (source_width, source_height) != (candidate.width, candidate.height):
                    counters["manifest_source_dimension_mismatch"] += 1
                    counters["selected_rows_rejected_dimension_mismatch"] += 1
                    continue
                _accumulate_detection_row(
                    domain,
                    candidate.width,
                    candidate.height,
                    candidate.band,
                    faces[row],
                    yolos[row],
                    policy,
                    accumulators,
                    counters,
                    bucket_counts,
                )
                counters["rows_joined_to_detection_source"] += 1
                joined_bands[candidate.band] += 1
                joined_cells[(candidate.band, candidate.sub_source)] += 1
                joined_subsources[candidate.sub_source] += 1
        if remaining:
            counters["selected_rows_missing_source"] += len(remaining)

    joined_complete = all(
        joined_cells[(band, source)] >= subsource_quotas[band][source]
        for band in AR_BANDS
        for source in MONET_REAL_SUBSOURCES
    )
    gate_blockers: list[str] = []
    if not sample_complete():
        gate_blockers.append("manifest_candidate_strata_incomplete")
    if not joined_complete:
        gate_blockers.append("source_join_strata_incomplete")
    if counters.get("manifest_source_dimension_mismatch", 0):
        gate_blockers.append("manifest_source_dimension_mismatch")
    if counters.get("selected_rows_missing_source", 0):
        gate_blockers.append("selected_rows_missing_source")
    if counters.get("face_json_parse_error_rows", 0):
        gate_blockers.append("face_json_parse_error_rows")
    if counters.get("face_malformed_boxes", 0):
        gate_blockers.append("face_malformed_boxes")
    return {
        "domain": domain,
        "requested_rows": sample_rows,
        "quota_rows_by_ar_band": quotas,
        "quota_rows_by_ar_band_and_subsource": subsource_quotas,
        "selected_rows_by_ar_band": dict(selected),
        "selected_rows_by_subsource": {
            source: selected_subsources[source] for source in MONET_REAL_SUBSOURCES
        },
        "selected_rows_by_ar_band_and_subsource": {
            band: {
                source: selected_cells[(band, source)]
                for source in MONET_REAL_SUBSOURCES
            }
            for band in AR_BANDS
        },
        "selected_rows_total": sum(selected.values()),
        "joined_rows_by_ar_band": {
            band: joined_bands[band] for band in AR_BANDS
        },
        "joined_rows_by_subsource": {
            source: joined_subsources[source] for source in MONET_REAL_SUBSOURCES
        },
        "joined_rows_by_ar_band_and_subsource": {
            band: {
                source: joined_cells[(band, source)]
                for source in MONET_REAL_SUBSOURCES
            }
            for band in AR_BANDS
        },
        "joined_rows_total": sum(joined_bands.values()),
        "candidate_stratified_sample_complete": sample_complete(),
        "complete_stratified_sample": joined_complete,
        "gate_input_ready": not gate_blockers,
        "gate_input_blockers": gate_blockers,
        "manifest_files_available": len(manifest_files),
        "manifest_files_scanned": manifest_files_scanned,
        "manifest_row_groups_scanned": manifest_row_groups_scanned,
        "natural_frequency_population": {
            "definition": (
                "All known-subsource, non-holdout MONET rich-manifest rows with finite "
                f"positive dimensions and min(width,height) >= {min_short_side}."
            ),
            "method": (
                "exact row-level width/height count over every supplied rich-manifest "
                "file; independent of AR-balanced candidate quotas and --max-source-files"
            ),
            "exact": True,
            "eligible_rows": natural_eligible_rows,
            "counts_by_ar_band": dict(natural_band_counts),
            "weights_by_ar_band": natural_weights,
            "manifest_files_scanned": natural_manifest_files_scanned,
            "manifest_row_groups_scanned": natural_manifest_row_groups_scanned,
            "counters": dict(sorted(natural_counters.items())),
        },
        "physical_detection_files_selected": len(candidates_by_source),
        "physical_detection_files_scanned": source_files_scanned,
        "physical_detection_row_groups_scanned": source_row_groups_scanned,
        "counters": dict(sorted(counters.items())),
        "assigned_bucket_counts_by_ar_band": {
            band: {str(key): value for key, value in sorted(counts.items())}
            for band, counts in bucket_counts.items()
        },
        "metrics": _summarize_metric_accumulators(domain, accumulators),
    }


COMPARISON_METRICS = (
    "full_retention_rate",
    "truncation_rate",
    "mean_retained_area_ratio",
    "p05_retained_area_ratio",
    "head_foot_joint_retention_rate",
    "top_cut_rate",
    "bottom_cut_rate",
    "top_only_rate",
    "bottom_only_rate",
    "top_and_bottom_rate",
    "vertical_cut_rate",
    "top_only_share_of_vertical_cuts",
    "bottom_only_share_of_vertical_cuts",
    "top_and_bottom_share_of_vertical_cuts",
    "left_cut_rate",
    "right_cut_rate",
)


def build_comparison_rows(domain_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for domain_result in domain_results:
        lookup = {
            (row["detection"], row["ar_band"], row["scheme"]): row
            for row in domain_result["metrics"]
        }
        for detection in ("face", "person"):
            for band in (ALL_BAND, *AR_BANDS):
                old = lookup[(detection, band, "old_square")]
                new = lookup[(detection, band, "offline_exact")]
                row: dict[str, Any] = {
                    "domain": domain_result["domain"],
                    "detection": detection,
                    "ar_band": band,
                    "bbox_count": old["bbox_count"],
                }
                for metric in COMPARISON_METRICS:
                    old_value, new_value = old[metric], new[metric]
                    row[f"old_{metric}"] = old_value
                    row[f"new_{metric}"] = new_value
                    row[f"delta_{metric}"] = (
                        None if old_value is None or new_value is None else new_value - old_value
                    )
                output.append(row)
    return output


def _natural_frequency_weighted(
    weights_by_ar_band: dict[str, float | None],
    values_by_metric: dict[str, dict[str, float | None]],
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for metric, values_by_ar_band in values_by_metric.items():
        weighted = 0.0
        has_positive_weight = False
        valid = True
        for band in AR_BANDS:
            weight_raw = weights_by_ar_band.get(band)
            if weight_raw is None:
                valid = False
                break
            weight = float(weight_raw)
            if not math.isfinite(weight) or weight < 0:
                valid = False
                break
            if weight == 0:
                continue
            value = values_by_ar_band.get(band)
            if value is None:
                valid = False
                break
            weighted += weight * float(value)
            has_positive_weight = True
        output[metric] = weighted if valid and has_positive_weight else None
    return output


def build_face_retention_headlines(
    domain_results: Sequence[dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (row["domain"], row["detection"], row["ar_band"]): row
        for row in comparisons
    }
    output: list[dict[str, Any]] = []
    metric_names = (
        "old_full_retention_rate",
        "new_full_retention_rate",
        "delta_full_retention_rate",
    )
    for domain_result in domain_results:
        domain = str(domain_result["domain"])
        overall = lookup[(domain, "face", ALL_BAND)]
        population = domain_result.get("natural_frequency_population")
        weights = population.get("weights_by_ar_band") if population else None
        weighted = (
            _natural_frequency_weighted(
                weights,
                {
                    metric: {
                        band: lookup[(domain, "face", band)][metric]
                        for band in AR_BANDS
                    }
                    for metric in metric_names
                },
            )
            if weights is not None
            else {metric: None for metric in metric_names}
        )
        output.append(
            {
                "domain": domain,
                "ar_balanced": {
                    "definition": AR_BALANCED_RETENTION_DEFINITION,
                    **{metric: overall[metric] for metric in metric_names},
                },
                "natural_frequency_weighted": {
                    "definition": NATURAL_FREQUENCY_RETENTION_DEFINITION,
                    "report_only": True,
                    "population_definition": (
                        population.get("definition") if population else None
                    ),
                    "weights_by_ar_band": weights,
                    **weighted,
                },
            }
        )
    return output


def _visibility_metrics(domain_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    counters = domain_result.get("counters", {})
    specs = (
        (
            "person_malformed_boxes/person_raw_boxes",
            counters.get("person_malformed_boxes", 0),
            counters.get("person_raw_boxes", 0),
        ),
        (
            "yolo_json_parse_error_rows/joined_rows_total",
            counters.get("yolo_json_parse_error_rows", 0),
            domain_result.get("joined_rows_total", 0),
        ),
        (
            "face_raw_oob_boxes/face_raw_boxes",
            counters.get("face_raw_oob_boxes", 0),
            counters.get("face_raw_boxes", 0),
        ),
    )
    return {
        name: {
            "numerator": int(numerator),
            "denominator": int(denominator),
            "rate": int(numerator) / int(denominator) if int(denominator) > 0 else None,
        }
        for name, numerator, denominator in specs
    }


def _visibility_warning_metrics(domain_result: dict[str, Any]) -> list[str]:
    return [
        name
        for name, metric in _visibility_metrics(domain_result).items()
        if metric["rate"] is not None and metric["rate"] > VISIBILITY_WARNING_RATE
    ]


def _format_rate(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _format_ratio(metric: dict[str, Any]) -> str:
    return (
        f"{metric['numerator']}/{metric['denominator']} "
        f"({_format_rate(metric['rate'])})"
    )


def build_quant_markdown(
    report: dict[str, Any], comparisons: Sequence[dict[str, Any]]
) -> str:
    visibility_rows: list[tuple[str, dict[str, dict[str, Any]]]] = []
    warning_items: list[str] = []
    for result in report["domain_results"]:
        domain = str(result["domain"])
        metrics = _visibility_metrics(result)
        visibility_rows.append((domain, metrics))
        warning_items.extend(
            f"{domain} {name}={_format_ratio(metrics[name])}"
            for name in _visibility_warning_metrics(result)
        )
    lines = [
        "# OFFLINE_ARB_V1 Phase-6 bbox quant audit",
        "",
    ]
    if warning_items:
        lines.extend(
            [
                "> **WARNING:** report-only detection input visibility ratio exceeds "
                f"{_format_rate(VISIBILITY_WARNING_RATE)}: {'; '.join(warning_items)}.",
                "",
            ]
        )
    lines.extend(
        [
            f"Generated: `{report['created_at_utc']}`",
            "",
            "Streaming ARB is an archive-only documentary reference in this run; it was not executed and has no measured column.",
            "",
            "## Face full-retention headline (dual reporting)",
            "",
            f"- `ar_balanced` (mainline and +15pt gate): {AR_BALANCED_RETENTION_DEFINITION}",
            f"- `natural_frequency_weighted` (report-only comparator): {NATURAL_FREQUENCY_RETENTION_DEFINITION}",
            "",
            "| domain | AR-balanced old | AR-balanced offline | AR-balanced delta | natural w portrait/near-square/landscape | natural old | natural offline | natural delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for headline in report["face_full_retention_headlines"]:
        balanced = headline["ar_balanced"]
        natural = headline["natural_frequency_weighted"]
        weights = natural.get("weights_by_ar_band") or {}
        weight_text = "/".join(_format_rate(weights.get(band)) for band in AR_BANDS)
        lines.append(
            f"| {headline['domain']} | {_format_rate(balanced['old_full_retention_rate'])} | "
            f"{_format_rate(balanced['new_full_retention_rate'])} | "
            f"{_format_rate(balanced['delta_full_retention_rate'])} | {weight_text} | "
            f"{_format_rate(natural['old_full_retention_rate'])} | "
            f"{_format_rate(natural['new_full_retention_rate'])} | "
            f"{_format_rate(natural['delta_full_retention_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Detection input visibility (report-only)",
            "",
            "Ratios use joined detection rows for YOLO JSON parse errors and raw boxes for malformed/OOB boxes; zero denominators are `n/a`. These diagnostics do not alter gate blockers.",
            "",
            "| domain | person_malformed_boxes/person_raw_boxes | yolo_json_parse_error_rows/joined_rows_total | face_raw_oob_boxes/face_raw_boxes |",
            "|---|---:|---:|---:|",
        ]
    )
    for domain, metrics in visibility_rows:
        lines.append(
            f"| {domain} | {_format_ratio(metrics['person_malformed_boxes/person_raw_boxes'])} | "
            f"{_format_ratio(metrics['yolo_json_parse_error_rows/joined_rows_total'])} | "
            f"{_format_ratio(metrics['face_raw_oob_boxes/face_raw_boxes'])} |"
        )
    lines.extend(
        [
            "",
            "## Overall comparison (AR-balanced audit sample)",
            "",
            "| domain | bbox | n | old full | offline full | delta | old trunc | offline trunc | old head+foot | offline head+foot |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        if row["ar_band"] != ALL_BAND:
            continue
        lines.append(
            "| {domain} | {detection} | {bbox_count} | {old} | {new} | {delta} | {ot} | {nt} | {ohf} | {nhf} |".format(
                domain=row["domain"],
                detection=row["detection"],
                bbox_count=row["bbox_count"],
                old=_format_rate(row["old_full_retention_rate"]),
                new=_format_rate(row["new_full_retention_rate"]),
                delta=_format_rate(row["delta_full_retention_rate"]),
                ot=_format_rate(row["old_truncation_rate"]),
                nt=_format_rate(row["new_truncation_rate"]),
                ohf=_format_rate(row["old_head_foot_joint_retention_rate"]),
                nhf=_format_rate(row["new_head_foot_joint_retention_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Face reference gate",
            "",
            "The +15pt decision below uses the `ar_balanced` headline only; the natural-frequency value is report-only.",
            "",
        ]
    )
    for gate in report["face_reference_gate"]:
        blockers = ",".join(gate.get("input_blockers", [])) or "none"
        lines.append(
            f"- `{gate['domain']}`: **{gate['status']}**, old={_format_rate(gate['old_full_retention_rate'])}, "
            f"offline={_format_rate(gate['new_full_retention_rate'])}, delta={_format_rate(gate['delta'])}, "
            f"reference={_format_rate(gate['required_delta'])}, input_blockers=`{blockers}`."
        )
    lines.extend(
        [
            "",
            "## Per-AR-band comparison",
            "",
            "| domain | band | bbox | n | old full | offline full | delta | offline p05 area | top cut old→new | bottom cut old→new |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        if row["ar_band"] == ALL_BAND:
            continue
        lines.append(
            "| {domain} | {band} | {detection} | {bbox_count} | {old} | {new} | {delta} | {p05} | {top_old}→{top_new} | {bottom_old}→{bottom_new} |".format(
                domain=row["domain"],
                band=row["ar_band"],
                detection=row["detection"],
                bbox_count=row["bbox_count"],
                old=_format_rate(row["old_full_retention_rate"]),
                new=_format_rate(row["new_full_retention_rate"]),
                delta=_format_rate(row["delta_full_retention_rate"]),
                p05=_format_rate(row["new_p05_retained_area_ratio"]),
                top_old=_format_rate(row["old_top_cut_rate"]),
                top_new=_format_rate(row["new_top_cut_rate"]),
                bottom_old=_format_rate(row["old_bottom_cut_rate"]),
                bottom_new=_format_rate(row["new_bottom_cut_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "`head+foot` is bbox-weighted joint vertical-edge retention: neither the bbox upper edge nor lower edge lies outside the crop window. Full retention additionally requires both horizontal edges.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        write_text(path, "")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def build_output_readme(
    policy: BucketPolicy,
    portrait_max: float,
    landscape_min: float,
    face_gate_delta: float,
) -> str:
    return textwrap.dedent(
        f"""\
        # OFFLINE_ARB_V1 Phase-6 visual audit outputs

        This directory is generated by `scripts/offline_arb/phase6_visual_audit.py`.

        ## Adjustable reference line

        The current review reference is **offline face full-retention rate >= legacy face
        full-retention rate + {face_gate_delta:.0%} ({face_gate_delta * 100:.0f} percentage points)**.
        Evan can adjust it with `quant --face-improvement-gate <fraction>`.  This is a
        review reference added for the visual audit, not an automatic authorization to
        enter formal training.

        ## Crop definitions

        - Legacy: `{policy.image_size}x{policy.image_size}` square, center crop (`v_anchor=0.5`).
        - Offline exact: config-built adaptive table, item assignment, canonical
          `bucket_prob_floor={policy.floor}`, then resize-to-cover and the config anchor
          (`anime={policy.anchor_for('anime')}`, other domains default to `0.5`).
        - Streaming ARB: documentary reference at
          `archive/streaming_arb_v1_20260709`; not run and not assigned a metric here.

        AR strata use source `width/height`: portrait `< {portrait_max}`, near-square
        `[{portrait_max}, {landscape_min}]`, landscape `> {landscape_min}`.
        Quant assigns one third of its configured row budget to each AR stratum; within
        each stratum, MONET sub-sources follow their 40M rich-pool proportions.  The `all`
        row and +15pt reference remain **AR-balanced audit rates**.  Alongside them, the
        report-only `natural_frequency_weighted` face headline is the sum of exact eligible
        rich-pool image-row band frequencies times the corresponding per-band retention
        rates.  Per-band rows are always retained for interpretation.

        ## Bbox metric definitions

        MediaPipe face boxes are normalized named xyxy.  YOLO person boxes are absolute
        pixel xyxy with `name == "person"`.  Persisted detections have no confidence.
        `classifier_yolo` is a whole-image top-k label list with no bbox, so it is probed
        for schema documentation but excluded from geometry metrics.
        Quant first samples the production `rich_v1/monet` manifest, then joins
        `src_root/src_shard/src_key` to the source parquet's narrow detection columns;
        it never samples the larger unselected physical MONET pool.
        Raw boxes are clipped to the source image first; raw out-of-bounds and malformed
        counts remain visible in `quant_report.json`.

        - retained area = intersection(bbox, crop window) / clipped source-visible bbox area;
        - full retention = retained area is 100%; truncation is its complement;
        - head+foot joint retention = neither bbox upper edge nor lower edge is outside
          the crop window (bbox-weighted); horizontal clipping does not fail this metric;
        - top/bottom/left/right rates count the corresponding bbox edge outside the crop.
          The JSON/CSV also reports top-only, bottom-only, and both as shares conditional
          on at least one vertical edge being cut.

        GPIC currently has no detection columns in either its rich manifest or caption
        parquet, so fetch-free quant marks it unavailable.  It remains in contact sheets.

        ## Files

        - `detection_schema_probe.json`: footer + one-row-group structure probe;
        - `quant_report.json`, `quant_comparison.csv`, `quant_summary.md`: bbox audit;
        - `contact_sheets/<domain>_<band>.png`: 5 domains x 3 bands; each sample is
          original context, red legacy window, green offline-exact window;
        - `sheets_index.json`: sample locators, exact windows, shortages, and fetch diagnostics.

        Square-only domains or genuinely absent native AR strata are not fabricated.
        Their sheet is still emitted with explicit empty placeholders and the shortage is
        recorded in `sheets_index.json`.  The only default structural exemptions are
        synthetic/portrait and synthetic/landscape (the production pool is native-square).
        Any other cell below `--per-band` makes `sheets` return 3 after writing the evidence;
        `--allow-sheet-shortages` is an explicit diagnostic-only override.
        """
    )


def quant_command(args: argparse.Namespace) -> int:
    policy = load_bucket_policy(args.config, args.n_matrix, args.stage, args.floor)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "detection_schema_probe.json",
        "quant_report.json",
        "quant_comparison.csv",
        "quant_summary.md",
        "README.md",
    ):
        (output_dir / name).unlink(missing_ok=True)
    paths = _manifest_paths(policy)

    representative_monet_source = os.path.join(
        args.monet_source_root, "cc12m", "0-0", "000000.parquet"
    )
    if not os.path.isfile(representative_monet_source):
        fallback_sources = list_monet_source_parquets(args.monet_source_root, synthetic=False)
        if not fallback_sources:
            raise FileNotFoundError(
                f"no monet_real source parquet under {args.monet_source_root}"
            )
        representative_monet_source = fallback_sources[0]
    monet_manifest_root = paths["monet_real_manifest"]
    monet_manifest_files = sorted(
        glob.glob(os.path.join(monet_manifest_root, "*.parquet"))
    )
    if not monet_manifest_files:
        raise FileNotFoundError(
            f"no monet_real rich manifest parquet under {monet_manifest_root}"
        )
    gpic_root = args.gpic_manifest_root or paths["gpic_manifest"]
    gpic_files = sorted(glob.glob(os.path.join(gpic_root, "*.parquet")))
    if not gpic_files:
        raise FileNotFoundError(f"no GPIC rich manifest parquet under {gpic_root}")

    probes = [
        probe_detection_file(representative_monet_source, "monet_real"),
        probe_detection_file(gpic_files[0], "gpic"),
    ]
    if not probes[0]["fetch_free_quant_available"]:
        raise RuntimeError(
            "monet_real source parquet lost its required detection columns; quant cannot run"
        )
    write_json(
        output_dir / "detection_schema_probe.json",
        {
            "created_at_utc": utc_now(),
            "probe_policy": "parquet footer plus row-group 0 only when relevant fields exist",
            "sources": probes,
        },
    )

    domain_results: list[dict[str, Any]] = []
    source_sets = {"monet_real": monet_manifest_files, "gpic": gpic_files}
    for probe in probes:
        domain = str(probe["domain"])
        if not probe["fetch_free_quant_available"]:
            print(f"[quant] {domain}: SKIP, no detection columns", flush=True)
            continue
        print(
            f"[quant] {domain}: stratified sample target={args.sample_rows} "
            f"files={len(source_sets[domain])}",
            flush=True,
        )
        if domain == "monet_real":
            result = quantify_monet_manifest_join(
                manifest_files=monet_manifest_files,
                policy=policy,
                sample_rows=args.sample_rows,
                max_rows_per_source=args.max_rows_per_source,
                seed=args.seed,
                portrait_max=args.ar_portrait_max,
                landscape_min=args.ar_landscape_min,
                min_short_side=policy.image_size,
                max_manifest_files=args.max_source_files,
            )
        else:
            result = quantify_detection_domain(
                domain=domain,
                files=source_sets[domain],
                policy=policy,
                sample_rows=args.sample_rows,
                max_rows_per_source=args.max_rows_per_source,
                seed=args.seed,
                portrait_max=args.ar_portrait_max,
                landscape_min=args.ar_landscape_min,
                min_short_side=policy.image_size,
                max_source_files=args.max_source_files,
            )
        domain_results.append(result)
        print(
            f"[quant] {domain}: selected={result['selected_rows_by_ar_band']} "
            f"joined={result['counters'].get('rows_joined_to_detection_source', result['selected_rows_total'])}",
            flush=True,
        )

    comparisons = build_comparison_rows(domain_results)
    face_retention_headlines = build_face_retention_headlines(
        domain_results, comparisons
    )
    result_by_domain = {result["domain"]: result for result in domain_results}
    gates: list[dict[str, Any]] = []
    for row in comparisons:
        if row["detection"] != "face" or row["ar_band"] != ALL_BAND:
            continue
        delta = row["delta_full_retention_rate"]
        domain_result = result_by_domain[row["domain"]]
        input_ready = bool(domain_result.get("gate_input_ready", False))
        blockers = list(domain_result.get("gate_input_blockers", []))
        if delta is None:
            input_ready = False
            blockers.append("no_face_bbox_metric")
        gates.append(
            {
                "domain": row["domain"],
                "bbox_count": row["bbox_count"],
                "old_full_retention_rate": row["old_full_retention_rate"],
                "new_full_retention_rate": row["new_full_retention_rate"],
                "delta": delta,
                "required_delta": args.face_improvement_gate,
                "input_ready": input_ready,
                "input_blockers": blockers,
                "status": (
                    "NOT_EVALUABLE"
                    if delta is None or not input_ready
                    else "PASS"
                    if delta >= args.face_improvement_gate
                    else "FAIL"
                ),
            }
        )

    report = {
        "schema_version": "offline_arb_phase6_quant_v1",
        "created_at_utc": utc_now(),
        "scope": {
            "measured": ["legacy square center crop", "offline exact ARB"],
            "streaming_arb": "archive-only documentary reference; not run or measured",
            "sample_rows_per_detection_domain": args.sample_rows,
            "sampling": (
                "deterministic equal-quota source-AR strata; monet sub-sources are "
                "proportional to their 40M rich-pool counts inside each AR stratum"
            ),
            "overall_rate_semantics": (
                "The existing all row is the bbox aggregate from equal image-row AR-band "
                "quotas; the face +15pt gate continues to use that AR-balanced mainline"
            ),
            "retention_headline_definitions": {
                "ar_balanced": AR_BALANCED_RETENTION_DEFINITION,
                "natural_frequency_weighted": NATURAL_FREQUENCY_RETENTION_DEFINITION,
            },
            "face_reference_gate_basis": "ar_balanced only; natural frequency is report-only",
            "ar_bands": {
                "portrait": f"w/h < {args.ar_portrait_max}",
                "near_square": f"{args.ar_portrait_max} <= w/h <= {args.ar_landscape_min}",
                "landscape": f"w/h > {args.ar_landscape_min}",
            },
        },
        "bucket_policy": policy.metadata(),
        "detection_availability": probes,
        "domain_results": domain_results,
        "comparisons": comparisons,
        "face_full_retention_headlines": face_retention_headlines,
        "face_reference_gate": gates,
    }
    write_json(output_dir / "quant_report.json", report)
    write_csv(output_dir / "quant_comparison.csv", comparisons)
    write_text(output_dir / "quant_summary.md", build_quant_markdown(report, comparisons))
    write_text(
        output_dir / "README.md",
        build_output_readme(
            policy,
            args.ar_portrait_max,
            args.ar_landscape_min,
            args.face_improvement_gate,
        ),
    )
    print(f"[quant] outputs: {output_dir}", flush=True)

    if args.enforce_face_gate and (
        not gates or any(gate["status"] != "PASS" for gate in gates)
    ):
        return 2
    return 0


@dataclass
class ImageCandidate:
    domain: str
    uid: str
    manifest_width: int
    manifest_height: int
    locator: dict[str, Any]


@dataclass
class AuditImage:
    candidate: ImageCandidate
    source_width: int
    source_height: int
    preview: Image.Image


def _decode_candidate(raw: bytes | None, candidate: ImageCandidate, preview_size: int) -> AuditImage | None:
    if not raw:
        return None
    from training.dataline.image_fetch import decode_rgb

    image = decode_rgb(raw)
    if image is None:
        return None
    width, height = image.size
    preview = image.copy()
    preview.thumbnail((preview_size, preview_size), Image.Resampling.LANCZOS)
    image.close()
    return AuditImage(candidate, width, height, preview)


def _all_full(samples: dict[str, list[AuditImage]], per_band: int) -> bool:
    return all(len(samples[band]) >= per_band for band in AR_BANDS)


def _eligible_for_sheets(domain: str, width: int, height: int, image_size: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    return domain == "anime" or min(width, height) >= image_size


def collect_tar_domain_images(
    domain: str,
    manifest_files: Sequence[str],
    policy: BucketPolicy,
    per_band: int,
    seed: int,
    portrait_max: float,
    landscape_min: float,
    preview_size: int,
    max_source_shards: int,
    row_groups_per_shard: int,
) -> tuple[dict[str, list[AuditImage]], list[dict[str, Any]]]:
    from training.dataline.image_fetch import monet_tar_path, stream_tar_images

    output = {band: [] for band in AR_BANDS}
    diagnostics: list[dict[str, Any]] = []
    rng = random.Random(stable_seed(seed, f"sheets:{domain}"))
    files = list(manifest_files)
    rng.shuffle(files)
    tar_scans = 0
    for manifest_path in files[:max_source_shards]:
        if _all_full(output, per_band):
            break
        pf = pq.ParquetFile(manifest_path)
        names = set(pf.schema_arrow.names)
        cols = ["uid", "src_root", "src_shard", "src_key", "width", "height"]
        if "holdout" in names:
            cols.append("holdout")
        for rg in range(min(pf.num_row_groups, row_groups_per_shard)):
            if _all_full(output, per_band) or tar_scans >= max_source_shards:
                break
            table = pf.read_row_group(
                rg, columns=cols, use_threads=False
            )
            values = {name: table.column(name).to_pylist() for name in cols}
            groups: dict[tuple[str, str], list[int]] = defaultdict(list)
            for row in range(table.num_rows):
                groups[(str(values["src_root"][row]), str(values["src_shard"][row]))].append(row)
            group_order = list(groups)
            rng.shuffle(group_order)
            for src_root, src_shard in group_order:
                if _all_full(output, per_band) or tar_scans >= max_source_shards:
                    break
                rows = groups[(src_root, src_shard)]
                # Rich-manifest rows are contiguous and source-ordered within src_shard.
                # Keeping that order lets stream_tar_images stop near the wanted prefix
                # instead of forcing a multi-GB scan to a random tail member.  The source
                # shard itself is still selected deterministically at random above.
                candidates: dict[str, ImageCandidate] = {}
                candidate_counts = Counter()
                for row in rows:
                    if "holdout" in values and values["holdout"][row] not in (0, False, None):
                        continue
                    width_raw, height_raw = values["width"][row], values["height"][row]
                    if not finite_positive(width_raw) or not finite_positive(height_raw):
                        continue
                    width = int(round(float(width_raw)))
                    height = int(round(float(height_raw)))
                    if not _eligible_for_sheets(domain, width, height, policy.image_size):
                        continue
                    band = ar_band(width, height, portrait_max, landscape_min)
                    remaining = per_band - len(output[band])
                    if remaining <= 0 or candidate_counts[band] >= remaining + 3:
                        continue
                    key = str(values["src_key"][row])
                    candidates[key] = ImageCandidate(
                        domain=domain,
                        uid=str(values["uid"][row]),
                        manifest_width=width,
                        manifest_height=height,
                        locator={
                            "manifest": manifest_path,
                            "manifest_row_group": rg,
                            "src_root": src_root,
                            "src_shard": src_shard,
                            "src_key": key,
                        },
                    )
                    candidate_counts[band] += 1
                if not candidates:
                    continue
                tar_path = monet_tar_path(src_root, src_shard)
                tar_scans += 1
                fetched = decoded = 0
                for key, raw in stream_tar_images(tar_path, set(candidates)):
                    fetched += 1
                    audit = _decode_candidate(raw, candidates[key], preview_size)
                    if audit is None:
                        continue
                    decoded += 1
                    actual_band = ar_band(
                        audit.source_width,
                        audit.source_height,
                        portrait_max,
                        landscape_min,
                    )
                    if len(output[actual_band]) < per_band:
                        output[actual_band].append(audit)
                    else:
                        audit.preview.close()
                diagnostics.append(
                    {
                        "manifest": manifest_path,
                        "manifest_row_group": rg,
                        "src_root": src_root,
                        "src_shard": src_shard,
                        "tar": tar_path,
                        "candidate_counts": dict(candidate_counts),
                        "fetched": fetched,
                        "decoded": decoded,
                    }
                )
    if not diagnostics:
        diagnostics.append(
            {
                "manifest_files_examined": min(len(files), max_source_shards),
                "tar_scans": tar_scans,
                "note": "no missing-band candidates found in examined production manifests",
            }
        )
    return output, diagnostics


def _fetch_parallel(
    candidates: Sequence[ImageCandidate],
    fetch: Callable[[ImageCandidate], bytes | None],
    workers: int,
) -> Iterator[tuple[ImageCandidate, bytes | None]]:
    if workers <= 1:
        for candidate in candidates:
            yield candidate, fetch(candidate)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for candidate, raw in zip(candidates, executor.map(fetch, candidates)):
            yield candidate, raw


def _select_manifest_candidates(
    domain: str,
    table: Any,
    columns: dict[str, str],
    output: dict[str, list[AuditImage]],
    per_band: int,
    rng: random.Random,
    portrait_max: float,
    landscape_min: float,
    image_size: int,
    locator_builder: Callable[[int], dict[str, Any]],
    uid_builder: Callable[[int], str],
) -> list[ImageCandidate]:
    widths = table.column(columns["width"]).to_pylist()
    heights = table.column(columns["height"]).to_pylist()
    holdout = (
        table.column(columns["holdout"]).to_pylist()
        if columns.get("holdout") and columns["holdout"] in table.column_names
        else [0] * table.num_rows
    )
    rows = list(range(table.num_rows))
    rng.shuffle(rows)
    counts = Counter()
    candidates: list[ImageCandidate] = []
    for row in rows:
        if holdout[row] not in (0, False, None):
            continue
        if not finite_positive(widths[row]) or not finite_positive(heights[row]):
            continue
        width, height = int(round(float(widths[row]))), int(round(float(heights[row])))
        if not _eligible_for_sheets(domain, width, height, image_size):
            continue
        band = ar_band(width, height, portrait_max, landscape_min)
        remaining = per_band - len(output[band])
        if remaining <= 0 or counts[band] >= remaining + 4:
            continue
        candidates.append(
            ImageCandidate(
                domain=domain,
                uid=uid_builder(row),
                manifest_width=width,
                manifest_height=height,
                locator=locator_builder(row),
            )
        )
        counts[band] += 1
        if all(
            len(output[b]) >= per_band or counts[b] >= per_band - len(output[b]) + 4
            for b in AR_BANDS
        ):
            break
    return candidates


def collect_gpic_images(
    manifest_files: Sequence[str],
    policy: BucketPolicy,
    per_band: int,
    seed: int,
    portrait_max: float,
    landscape_min: float,
    preview_size: int,
    max_manifest_files: int,
    fetch_workers: int,
) -> tuple[dict[str, list[AuditImage]], list[dict[str, Any]]]:
    from training.dataline.image_fetch import read_gpic_bytes

    output = {band: [] for band in AR_BANDS}
    diagnostics: list[dict[str, Any]] = []
    rng = random.Random(stable_seed(seed, "sheets:gpic"))
    files = list(manifest_files)
    rng.shuffle(files)
    for path in files[:max_manifest_files]:
        if _all_full(output, per_band):
            break
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        cols = ["uid", "src_root", "src_shard", "src_key", "width", "height"]
        if "holdout" in names:
            cols.append("holdout")
        rg = rng.randrange(pf.num_row_groups)
        table = pf.read_row_group(rg, columns=cols, use_threads=False)
        values = {name: table.column(name).to_pylist() for name in cols}
        candidates = _select_manifest_candidates(
            "gpic",
            table,
            {
                "width": "width",
                "height": "height",
                "holdout": "holdout" if "holdout" in cols else "",
            },
            output,
            per_band,
            rng,
            portrait_max,
            landscape_min,
            policy.image_size,
            lambda row: {
                "manifest": path,
                "row_group": rg,
                "src_root": str(values["src_root"][row]),
                "src_shard": str(values["src_shard"][row]),
                "src_key": str(values["src_key"][row]),
            },
            lambda row: str(values["uid"][row]),
        )

        def fetch(candidate: ImageCandidate) -> bytes | None:
            loc = candidate.locator
            return read_gpic_bytes(
                loc["src_root"], loc["src_shard"], loc["src_key"], ext=None
            )

        fetched = decoded = 0
        for candidate, raw in _fetch_parallel(candidates, fetch, fetch_workers):
            fetched += int(raw is not None)
            audit = _decode_candidate(raw, candidate, preview_size)
            if audit is None:
                continue
            decoded += 1
            band = ar_band(audit.source_width, audit.source_height, portrait_max, landscape_min)
            if len(output[band]) < per_band:
                output[band].append(audit)
            else:
                audit.preview.close()
        diagnostics.append(
            {
                "manifest": path,
                "row_group": rg,
                "candidates": len(candidates),
                "fetched": fetched,
                "decoded": decoded,
            }
        )
    return output, diagnostics


def collect_anime_images(
    source_files: Sequence[str],
    policy: BucketPolicy,
    per_band: int,
    seed: int,
    portrait_max: float,
    landscape_min: float,
    preview_size: int,
    max_manifest_files: int,
    fetch_workers: int,
    cache_dir: str | None,
) -> tuple[dict[str, list[AuditImage]], list[dict[str, Any]]]:
    from training.dataline.image_fetch import fetch_url_bytes

    output = {band: [] for band in AR_BANDS}
    diagnostics: list[dict[str, Any]] = []
    rng = random.Random(stable_seed(seed, "sheets:anime"))
    files = list(source_files)
    rng.shuffle(files)
    for path in files[:max_manifest_files]:
        if _all_full(output, per_band):
            break
        pf = pq.ParquetFile(path)
        rg = rng.randrange(pf.num_row_groups)
        table = pf.read_row_group(
            rg, columns=["key", "s3key", "width", "height"], use_threads=False
        )
        values = {
            name: table.column(name).to_pylist()
            for name in ("key", "s3key", "width", "height")
        }
        candidates = _select_manifest_candidates(
            "anime",
            table,
            {"width": "width", "height": "height", "holdout": ""},
            output,
            per_band,
            rng,
            portrait_max,
            landscape_min,
            policy.image_size,
            lambda row: {
                "source_parquet": path,
                "row_group": rg,
                "key": str(values["key"][row]),
                "url": str(values["s3key"][row]),
            },
            lambda row: f"anime:{values['key'][row]}",
        )

        def fetch(candidate: ImageCandidate) -> bytes | None:
            return fetch_url_bytes(candidate.locator["url"], cache_dir=cache_dir)

        fetched = decoded = 0
        for candidate, raw in _fetch_parallel(candidates, fetch, fetch_workers):
            fetched += int(raw is not None)
            audit = _decode_candidate(raw, candidate, preview_size)
            if audit is None:
                continue
            decoded += 1
            band = ar_band(audit.source_width, audit.source_height, portrait_max, landscape_min)
            if len(output[band]) < per_band:
                output[band].append(audit)
            else:
                audit.preview.close()
        diagnostics.append(
            {
                "source_parquet": path,
                "row_group": rg,
                "candidates": len(candidates),
                "fetched": fetched,
                "decoded": decoded,
                "cache_dir": cache_dir,
            }
        )
    return output, diagnostics


def collect_textatlas_images(
    sidecar_files: Sequence[str],
    source_root: str,
    local_root: str | None,
    policy: BucketPolicy,
    per_band: int,
    seed: int,
    portrait_max: float,
    landscape_min: float,
    preview_size: int,
    max_manifest_files: int,
) -> tuple[dict[str, list[AuditImage]], list[dict[str, Any]]]:
    output = {band: [] for band in AR_BANDS}
    diagnostics: list[dict[str, Any]] = []
    rng = random.Random(stable_seed(seed, "sheets:textatlas"))
    files = list(sidecar_files)
    rng.shuffle(files)
    source_files_used = 0
    metadata_files_examined = 0
    # Sidecar width/height is cheap.  Inspect more sidecars than image parquets so a
    # square-heavy subset cannot consume the entire physical-read budget before the
    # portrait/landscape strata are found.
    for sidecar_path in files[: max_manifest_files * 20]:
        if _all_full(output, per_band) or source_files_used >= max_manifest_files:
            break
        metadata_files_examined += 1
        subset = Path(sidecar_path).parent.name
        shard_name = Path(sidecar_path).name
        source_path = os.path.join(source_root, subset, shard_name)
        if local_root:
            local_path = os.path.join(local_root, subset, shard_name)
            if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
                source_path = local_path
        side = pq.read_table(
            sidecar_path,
            columns=["uid", "image_path", "width", "height"],
            use_threads=False,
        )
        side_values = {
            name: side.column(name).to_pylist()
            for name in ("uid", "image_path", "width", "height")
        }
        has_needed_band = False
        for width_raw, height_raw in zip(side_values["width"], side_values["height"]):
            if not finite_positive(width_raw) or not finite_positive(height_raw):
                continue
            width, height = int(width_raw), int(height_raw)
            if not _eligible_for_sheets("textatlas", width, height, policy.image_size):
                continue
            band = ar_band(width, height, portrait_max, landscape_min)
            if len(output[band]) < per_band:
                has_needed_band = True
                break
        if not has_needed_band:
            continue
        source_pf = pq.ParquetFile(source_path)
        starts: list[int] = []
        cursor = 0
        for rg in range(source_pf.num_row_groups):
            starts.append(cursor)
            cursor += source_pf.metadata.row_group(rg).num_rows
        rg_order = list(range(source_pf.num_row_groups))
        rng.shuffle(rg_order)
        read_groups = decoded = 0
        for rg in rg_order:
            if _all_full(output, per_band):
                break
            start = starts[rg]
            count = source_pf.metadata.row_group(rg).num_rows
            if start + count > side.num_rows:
                diagnostics.append(
                    {
                        "sidecar": sidecar_path,
                        "source": source_path,
                        "error": "source row-group range exceeds sidecar row count",
                        "row_group": rg,
                    }
                )
                continue
            rows = list(range(start, start + count))
            rng.shuffle(rows)
            chosen: list[int] = []
            chosen_counts = Counter()
            for row in rows:
                width_raw = side_values["width"][row]
                height_raw = side_values["height"][row]
                if not finite_positive(width_raw) or not finite_positive(height_raw):
                    continue
                width = int(width_raw)
                height = int(height_raw)
                if not _eligible_for_sheets("textatlas", width, height, policy.image_size):
                    continue
                band = ar_band(width, height, portrait_max, landscape_min)
                remaining = per_band - len(output[band])
                if remaining > 0 and chosen_counts[band] < remaining:
                    chosen.append(row)
                    chosen_counts[band] += 1
            if not chosen:
                continue
            batch = source_pf.read_row_group(
                rg, columns=["image", "image_path"], use_threads=False
            )
            read_groups += 1
            raw_images = batch.column("image").to_pylist()
            raw_paths = batch.column("image_path").to_pylist()
            by_path = {str(path): image for path, image in zip(raw_paths, raw_images)}
            for row in chosen:
                image_path = str(side_values["image_path"][row])
                value = by_path.get(image_path)
                raw = value.get("bytes") if isinstance(value, dict) else None
                candidate = ImageCandidate(
                    domain="textatlas",
                    uid=str(side_values["uid"][row]),
                    manifest_width=int(side_values["width"][row]),
                    manifest_height=int(side_values["height"][row]),
                    locator={
                        "sidecar": sidecar_path,
                        "source_parquet": source_path,
                        "row_group": rg,
                        "absolute_row": row,
                        "image_path": image_path,
                    },
                )
                audit = _decode_candidate(raw, candidate, preview_size)
                if audit is None:
                    continue
                decoded += 1
                band = ar_band(
                    audit.source_width, audit.source_height, portrait_max, landscape_min
                )
                if len(output[band]) < per_band:
                    output[band].append(audit)
                else:
                    audit.preview.close()
        diagnostics.append(
            {
                "sidecar": sidecar_path,
                "source": source_path,
                "source_rows": cursor,
                "sidecar_rows": side.num_rows,
                "row_groups_read": read_groups,
                "decoded": decoded,
            }
        )
        if read_groups:
            source_files_used += 1
    diagnostics.append(
        {
            "summary": True,
            "metadata_sidecars_examined": metadata_files_examined,
            "source_image_parquets_used": source_files_used,
        }
    )
    return output, diagnostics


def _short_label(value: str, limit: int = 58) -> str:
    value = value.encode("ascii", "replace").decode("ascii")
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _draw_panel(
    preview: Image.Image,
    source_width: int,
    source_height: int,
    panel_size: int,
    title: str,
    window: Window | None,
    color: tuple[int, int, int],
) -> Image.Image:
    label_height = 20
    panel = Image.new("RGB", (panel_size, panel_size + label_height), (28, 28, 31))
    draw = ImageDraw.Draw(panel)
    draw.text((5, 4), title, fill=color if window else (225, 225, 225))
    image = ImageOps.contain(preview, (panel_size - 8, panel_size - 8))
    left = (panel_size - image.width) // 2
    top = label_height + (panel_size - image.height) // 2
    panel.paste(image, (left, top))
    if window is not None:
        x1 = left + window.xmin / source_width * image.width
        y1 = top + window.ymin / source_height * image.height
        x2 = left + window.xmax / source_width * image.width
        y2 = top + window.ymax / source_height * image.height
        draw.rectangle((round(x1), round(y1), round(x2), round(y2)), outline=color, width=3)
    return panel


def _audit_new_geometry(
    image: AuditImage, policy: BucketPolicy
) -> tuple[int, CropGeometry]:
    # L2 assignment is metadata-driven; the crop window is applied to decoded pixels.
    # Phase-1 found those dimensions consistent, but keeping the two roles explicit
    # makes any future mismatch visible without silently changing the assigned bucket.
    _raw, assigned = policy.assigned_bucket(
        image.candidate.domain,
        image.candidate.manifest_width,
        image.candidate.manifest_height,
    )
    target_h, target_w = policy.buckets[assigned]
    geometry = new_bucket_geometry(
        image.source_width,
        image.source_height,
        target_w,
        target_h,
        policy.anchor_for(image.candidate.domain),
    )
    return assigned, geometry


def _sample_index_entry(
    image: AuditImage, policy: BucketPolicy, sheet_path: str
) -> dict[str, Any]:
    assigned, new_geom = _audit_new_geometry(image, policy)
    old_geom = old_square_geometry(image.source_width, image.source_height, policy.image_size)
    target_h, target_w = policy.buckets[assigned]
    return {
        "domain": image.candidate.domain,
        "uid": image.candidate.uid,
        "source_width": image.source_width,
        "source_height": image.source_height,
        "manifest_width": image.candidate.manifest_width,
        "manifest_height": image.candidate.manifest_height,
        "assigned_bucket_idx": assigned,
        "assigned_bucket_height": target_h,
        "assigned_bucket_width": target_w,
        "v_anchor": policy.anchor_for(image.candidate.domain),
        "old_geometry": old_geom.to_dict(),
        "new_geometry": new_geom.to_dict(),
        "locator": image.candidate.locator,
        "sheet": sheet_path,
    }


def render_contact_sheet(
    domain: str,
    band: str,
    samples: Sequence[AuditImage],
    policy: BucketPolicy,
    output_path: Path,
    slots: int,
    panel_size: int,
    portrait_max: float,
    landscape_min: float,
) -> None:
    grid_columns = 3
    grid_rows = math.ceil(slots / grid_columns)
    gap = 10
    metadata_height = 38
    cell_width = panel_size * 3
    cell_height = panel_size + 20 + metadata_height
    header_height = 50
    footer_height = 28
    width = gap + grid_columns * (cell_width + gap)
    height = header_height + grid_rows * (cell_height + gap) + footer_height
    sheet = Image.new("RGB", (width, height), (18, 18, 21))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (gap, 8),
        f"OFFLINE_ARB_V1  domain={domain}  AR band={band}  samples={len(samples)}/{slots}",
        fill=(245, 245, 245),
    )
    draw.text(
        (gap, 26),
        f"portrait < {portrait_max}; near [{portrait_max},{landscape_min}]; landscape > {landscape_min}  |  red=legacy, green=offline",
        fill=(180, 180, 185),
    )
    for slot in range(slots):
        row, column = divmod(slot, grid_columns)
        x = gap + column * (cell_width + gap)
        y = header_height + row * (cell_height + gap)
        if slot >= len(samples):
            draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline=(80, 80, 85))
            draw.text((x + 12, y + 15), "NO ELIGIBLE/FETCHED SAMPLE", fill=(210, 145, 75))
            continue
        sample = samples[slot]
        assigned, new_geom = _audit_new_geometry(sample, policy)
        old_geom = old_square_geometry(
            sample.source_width, sample.source_height, policy.image_size
        )
        target_h, target_w = policy.buckets[assigned]
        panels = (
            _draw_panel(
                sample.preview,
                sample.source_width,
                sample.source_height,
                panel_size,
                "original",
                None,
                (235, 235, 235),
            ),
            _draw_panel(
                sample.preview,
                sample.source_width,
                sample.source_height,
                panel_size,
                f"legacy {policy.image_size}x{policy.image_size}",
                old_geom.window,
                (245, 75, 75),
            ),
            _draw_panel(
                sample.preview,
                sample.source_width,
                sample.source_height,
                panel_size,
                f"offline b{assigned} {target_w}x{target_h}",
                new_geom.window,
                (65, 220, 115),
            ),
        )
        for index, panel in enumerate(panels):
            sheet.paste(panel, (x + index * panel_size, y))
            panel.close()
        aspect = sample.source_width / sample.source_height
        draw.text(
            (x + 4, y + panel_size + 23),
            _short_label(sample.candidate.uid),
            fill=(225, 225, 225),
        )
        draw.text(
            (x + 4, y + panel_size + 39),
            f"decoded={sample.source_width}x{sample.source_height} AR={aspect:.3f}  anchor={policy.anchor_for(domain):.2f}",
            fill=(165, 165, 170),
        )
    draw.text(
        (gap, height - footer_height + 7),
        "streaming ARB archive is documentary only; it was not run or measured in this sheet",
        fill=(150, 150, 155),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.stem + f".tmp.{os.getpid()}.png")
    sheet.save(tmp, format="PNG")
    sheet.close()
    os.replace(tmp, output_path)


def sheets_command(args: argparse.Namespace) -> int:
    policy = load_bucket_policy(args.config, args.n_matrix, args.stage, args.floor)
    paths = _manifest_paths(policy)
    output_dir = Path(args.output_dir)
    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sheets_index.json").unlink(missing_ok=True)
    # Fixed output namespace: remove only the 15 known PNGs so an interrupted rerun
    # cannot leave a stale prior sheet masquerading as a current-domain result.
    for known_domain in DOMAINS:
        for known_band in AR_BANDS:
            (sheets_dir / f"{known_domain}_{known_band}.png").unlink(missing_ok=True)

    monet_files = sorted(
        glob.glob(os.path.join(paths["monet_real_manifest"], "*.parquet"))
    )
    synthetic_files = sorted(
        glob.glob(os.path.join(paths["synthetic_manifest"], "*.parquet"))
    )
    gpic_root = args.gpic_manifest_root or paths["gpic_manifest"]
    gpic_files = sorted(glob.glob(os.path.join(gpic_root, "*.parquet")))
    anime_root = args.anime_source_root or paths["anime_source"]
    anime_files = sorted(glob.glob(os.path.join(anime_root, "*.parquet")))
    textatlas_manifest = args.textatlas_manifest_root or paths["textatlas_manifest"]
    textatlas_source = args.textatlas_source_root or paths["textatlas_source"]
    textatlas_files = sorted(
        path
        for subset in paths["textatlas_subsets"]
        for path in glob.glob(os.path.join(textatlas_manifest, subset, "*.parquet"))
    )
    required = {
        "monet_real manifest": monet_files,
        "synthetic manifest": synthetic_files,
        "gpic manifest": gpic_files,
        "anime source": anime_files,
        "textatlas sidecar": textatlas_files,
    }
    missing = [name for name, values in required.items() if not values]
    if missing:
        raise FileNotFoundError(f"missing sheet inputs: {', '.join(missing)}")

    collectors: dict[str, Callable[[], tuple[dict[str, list[AuditImage]], list[dict[str, Any]]]]] = {
        "anime": lambda: collect_anime_images(
            anime_files,
            policy,
            args.per_band,
            args.seed,
            args.ar_portrait_max,
            args.ar_landscape_min,
            args.preview_size,
            args.max_source_shards,
            args.fetch_workers,
            args.anime_cache_dir or None,
        ),
        "monet_real": lambda: collect_tar_domain_images(
            "monet_real",
            monet_files,
            policy,
            args.per_band,
            args.seed,
            args.ar_portrait_max,
            args.ar_landscape_min,
            args.preview_size,
            args.max_source_shards,
            args.tar_row_groups_per_shard,
        ),
        "gpic": lambda: collect_gpic_images(
            gpic_files,
            policy,
            args.per_band,
            args.seed,
            args.ar_portrait_max,
            args.ar_landscape_min,
            args.preview_size,
            args.max_source_shards,
            args.fetch_workers,
        ),
        "synthetic": lambda: collect_tar_domain_images(
            "synthetic",
            synthetic_files,
            policy,
            args.per_band,
            args.seed,
            args.ar_portrait_max,
            args.ar_landscape_min,
            args.preview_size,
            args.max_source_shards,
            args.tar_row_groups_per_shard,
        ),
        "textatlas": lambda: collect_textatlas_images(
            textatlas_files,
            textatlas_source,
            args.textatlas_local_root or None,
            policy,
            args.per_band,
            args.seed,
            args.ar_portrait_max,
            args.ar_landscape_min,
            args.preview_size,
            args.max_source_shards,
        ),
    }

    index: dict[str, Any] = {
        "schema_version": "offline_arb_phase6_sheets_v1",
        "created_at_utc": utc_now(),
        "bucket_policy": policy.metadata(),
        "streaming_arb": "archive-only documentary reference; not run or measured",
        "requested_per_domain_band": args.per_band,
        "structural_shortage_exemptions": [
            "synthetic/portrait",
            "synthetic/landscape",
        ],
        "domains": {},
    }
    for domain in DOMAINS:
        print(f"[sheets] collecting {domain}", flush=True)
        samples, diagnostics = collectors[domain]()
        domain_entry: dict[str, Any] = {
            "counts": {band: len(samples[band]) for band in AR_BANDS},
            "shortages": {
                band: max(0, args.per_band - len(samples[band])) for band in AR_BANDS
            },
            "fetch_diagnostics": diagnostics,
            "samples": {},
        }
        for band in AR_BANDS:
            sheet_path = sheets_dir / f"{domain}_{band}.png"
            render_contact_sheet(
                domain,
                band,
                samples[band],
                policy,
                sheet_path,
                args.per_band,
                args.preview_size,
                args.ar_portrait_max,
                args.ar_landscape_min,
            )
            relative_sheet = str(sheet_path.relative_to(output_dir))
            domain_entry["samples"][band] = [
                _sample_index_entry(sample, policy, relative_sheet)
                for sample in samples[band]
            ]
            print(
                f"[sheets] {domain}/{band}: {len(samples[band])}/{args.per_band} -> {sheet_path}",
                flush=True,
            )
        index["domains"][domain] = domain_entry
        write_json(output_dir / "sheets_index.json", index)
        for band_samples in samples.values():
            for sample in band_samples:
                sample.preview.close()

    exempt_cells = {("synthetic", "portrait"), ("synthetic", "landscape")}
    unexpected_shortages: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for band in AR_BANDS:
            count = int(index["domains"][domain]["counts"][band])
            if count < args.per_band and (domain, band) not in exempt_cells:
                unexpected_shortages.append(
                    {
                        "domain": domain,
                        "ar_band": band,
                        "observed": count,
                        "required": args.per_band,
                        "missing": args.per_band - count,
                    }
                )
    index["unexpected_shortages"] = unexpected_shortages
    index["completion_status"] = (
        "PASS"
        if not unexpected_shortages
        else "INCOMPLETE_ALLOWED"
        if args.allow_sheet_shortages
        else "NOT_EVALUABLE"
    )
    write_json(output_dir / "sheets_index.json", index)
    readme_path = output_dir / "README.md"
    if not readme_path.exists():
        write_text(
            readme_path,
            build_output_readme(
                policy,
                args.ar_portrait_max,
                args.ar_landscape_min,
                DEFAULT_FACE_GATE_DELTA,
            ),
        )
    print(f"[sheets] outputs: {sheets_dir}", flush=True)
    if unexpected_shortages and not args.allow_sheet_shortages:
        print(
            f"[sheets] NOT_EVALUABLE: unexpected shortages={unexpected_shortages}",
            flush=True,
        )
        return 3
    return 0


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--n-matrix", default=DEFAULT_N_MATRIX)
    parser.add_argument("--stage", default="256")
    parser.add_argument("--floor", type=float, default=0.005)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--ar-portrait-max", type=float, default=AR_PORTRAIT_MAX)
    parser.add_argument("--ar-landscape-min", type=float, default=AR_LANDSCAPE_MIN)
    parser.add_argument("--monet-source-root", default=DEFAULT_MONET_SOURCE_ROOT)
    parser.add_argument("--gpic-manifest-root", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OFFLINE_ARB_V1 Phase-6 visual composition audit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    quant = subparsers.add_parser(
        "quant", help="fetch-free face/person bbox retention audit"
    )
    add_common_arguments(quant)
    quant.add_argument("--sample-rows", type=int, default=50_000)
    quant.add_argument("--max-rows-per-source", type=int, default=1_500)
    quant.add_argument(
        "--max-source-files",
        type=int,
        default=0,
        help="0 means unlimited; intended only for diagnostics",
    )
    quant.add_argument(
        "--face-improvement-gate", type=float, default=DEFAULT_FACE_GATE_DELTA
    )
    quant.add_argument(
        "--enforce-face-gate",
        action="store_true",
        help="return code 2 on FAIL/NOT_EVALUABLE; default is report-only so sheets can still run",
    )
    quant.set_defaults(func=quant_command)

    sheets = subparsers.add_parser(
        "sheets", help="fetch a small five-domain sample and render crop-window contact sheets"
    )
    add_common_arguments(sheets)
    sheets.add_argument("--per-band", type=int, default=12)
    sheets.add_argument("--fetch-workers", type=int, default=8)
    sheets.add_argument("--max-source-shards", type=int, default=6)
    sheets.add_argument("--tar-row-groups-per-shard", type=int, default=3)
    sheets.add_argument("--preview-size", type=int, default=176)
    sheets.add_argument(
        "--allow-sheet-shortages",
        action="store_true",
        help=(
            "diagnostic only: return 0 despite non-structural cells below --per-band; "
            "default returns 3 after writing all sheets/index"
        ),
    )
    sheets.add_argument("--anime-source-root", default=None)
    sheets.add_argument(
        "--anime-cache-dir",
        default=None,
        help="default None keeps the audit read-only; set explicitly to reuse/write the normal cache",
    )
    sheets.add_argument("--textatlas-manifest-root", default=None)
    sheets.add_argument("--textatlas-source-root", default=None)
    sheets.add_argument(
        "--textatlas-local-root",
        default="/local/evan/textatlas5m",
        help="read existing staged shards when present; falls back to configured source root",
    )
    sheets.set_defaults(func=sheets_command)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not (0 < args.ar_portrait_max <= 1.0 <= args.ar_landscape_min):
        parser.error("AR thresholds must satisfy 0 < portrait_max <= 1 <= landscape_min")
    if args.floor < 0:
        parser.error("--floor must be non-negative")
    if args.command == "quant":
        if args.sample_rows <= 0 or args.max_rows_per_source <= 0:
            parser.error("quant row counts must be positive")
    if args.command == "sheets":
        if min(
            args.per_band,
            args.fetch_workers,
            args.max_source_shards,
            args.tar_row_groups_per_shard,
            args.preview_size,
        ) <= 0:
            parser.error("sheet counts/workers/sizes must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
