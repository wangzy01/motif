"""Stage256 dataset adapter for :class:`SemanticOfflinePlan`.

No frozen legacy source file is edited.  During the base constructor only, a
process-local lock protects a fail-safe substitution of the loader's plan
class.  The legacy exception binders intentionally see the new semantic plan
identity and therefore remain inactive; all 11 exceptional rows are required
to have zero exposure by the overlay, reader and full audit.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Sequence

from training.dataline_offline import offline_dataset as _base

from .semantic_plan import SemanticOfflinePlan, SemanticPlanError, sha256_file


_CONSTRUCTION_LOCK = threading.RLock()
_NL_FIELDS = (
    "refined_description",
    "medium_description",
    "short_description",
)


class OfflineSemanticExactDataset(_base.OfflineExactDataset):
    """Legacy exact loader with a physical-row semantic plan and NL hard gate."""

    def __init__(self, *args: Any, **kwargs: Any):
        self._semantic_metric_lock = threading.Lock()
        self._semantic_anime_nl_records = 0
        self._semantic_anime_tag_only_records = 0

        with _CONSTRUCTION_LOCK:
            original_plan = _base.OfflinePlan

            def semantic_plan_factory(*plan_args: Any, **plan_kwargs: Any) -> SemanticOfflinePlan:
                semantic = SemanticOfflinePlan(
                    *plan_args,
                    **plan_kwargs,
                    require_audit=True,
                )
                return semantic

            _base.OfflinePlan = semantic_plan_factory
            try:
                super().__init__(*args, **kwargs)
            finally:
                _base.OfflinePlan = original_plan

        if not isinstance(self.plan, SemanticOfflinePlan):
            raise SemanticPlanError("base dataset did not retain SemanticOfflinePlan")
        expected_catalog_header = self.plan.header["bindings"][
            "catalog_header_sha256"
        ]
        actual_catalog_header = sha256_file(self.catalog_root / "header.json")
        if actual_catalog_header != expected_catalog_header:
            raise SemanticPlanError(
                "runtime catalog header differs from semantic binding: "
                f"expected={expected_catalog_header}, actual={actual_catalog_header}"
            )
        expected_sampling = self.plan.header["bindings"]["sampling_config_sha256"]
        actual_sampling = sha256_file(self.sampling_config_path)
        if actual_sampling != expected_sampling:
            raise SemanticPlanError(
                "runtime sampling config differs from semantic binding: "
                f"expected={expected_sampling}, actual={actual_sampling}"
            )
        anime_cfg = self.cfg.anime_caption
        if not anime_cfg.requires_nl(self.stage):
            raise SemanticPlanError(
                f"sampling config must require NL for Stage {self.stage}"
            )
        route_weights = anime_cfg.route_weights(
            self.stage,
            True,
            caption_policy=None,
        )
        tag_weight = sum(
            float(weight)
            for route, weight in route_weights.items()
            if str(route).startswith("tag")
        )
        if tag_weight != 0.0:
            raise SemanticPlanError(
                f"Stage {self.stage} NL route has nonzero tag weight: {route_weights}"
            )

    def _caption_records(
        self,
        domain: str,
        pairs: Sequence[tuple[Any, dict[str, Any]]],
    ) -> dict[tuple[int, int], dict[str, Any]]:
        records = super()._caption_records(domain, pairs)
        if domain != "anime":
            return records
        checked = 0
        for token, record in records.items():
            fields = record.get("cap_fields")
            if not isinstance(fields, dict):
                raise _base.OfflineLoaderError(
                    f"semantic anime record lacks cap_fields at token={token}"
                )
            for key in _NL_FIELDS:
                value = fields.get(key)
                if value is not None and not isinstance(value, str):
                    raise _base.OfflineLoaderError(
                        f"semantic anime NL field {key} has non-string value "
                        f"at token={token}: {type(value).__name__}"
                    )
                # Preserve non-empty NL text byte-for-byte.  Normalize only a
                # whitespace-only string so the legacy truthiness-first router
                # cannot select it ahead of a later valid NL field and then
                # fall back to tags after trimming.
                if isinstance(value, str) and not value.strip():
                    fields[key] = ""
            if not any(
                isinstance(fields.get(key), str) and fields[key].strip()
                for key in _NL_FIELDS
            ):
                with self._semantic_metric_lock:
                    self._semantic_anime_tag_only_records += 1
                raise _base.OfflineLoaderError(
                    "semantic Stage256 tag-only exposure is forbidden: "
                    f"token={token},row_id={record.get('row_id')}"
                )
            checked += 1
        with self._semantic_metric_lock:
            self._semantic_anime_nl_records += checked
        return records

    @staticmethod
    def _is_cfg_protected(domain: str, record: Mapping[str, Any]) -> bool:
        protected = _base.OfflineExactDataset._is_cfg_protected(domain, record)
        if domain == "anime" and protected:
            raise _base.OfflineLoaderError(
                "semantic Stage256 anime record reached tag-only CFG protection"
            )
        return protected

    def _refresh_metrics_locked(self, plan_step: int | None = None) -> None:
        super()._refresh_metrics_locked(plan_step=plan_step)
        with self._semantic_metric_lock:
            self.buffer_fill["semantic_anime_nl_records"] = float(
                self._semantic_anime_nl_records
            )
            self.buffer_fill["semantic_anime_tag_only_records"] = float(
                self._semantic_anime_tag_only_records
            )
            self.buffer_fill["semantic_tag_fallback_records"] = 0.0


__all__ = ["OfflineSemanticExactDataset"]
