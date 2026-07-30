#!/usr/bin/env python3
"""Focused tests for topology-independent coordinate evidence."""

from __future__ import annotations

import inspect
import sys
import types
import unittest

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    # Coordinate-schema helpers are deliberately pure.  A tiny import stub
    # lets this focused test run on control hosts where the production NumPy
    # dependency is not installed; no array-processing path is exercised.
    sys.modules["numpy"] = types.ModuleType("numpy")

from scripts.offline_arb.audit_textatlas_plan_rows import (
    AuditError,
    COORDINATE_SCHEMA_AUDIT_NAMES,
    _coordinate_record,
    _resolve_within_formal_field,
)
from scripts.offline_arb.immutable_review_evidence import (
    EvidencePublishError,
    validate_plan_coordinates,
    validate_stage512a_coordinates,
)


W64_CONTRACT = {
    "plan_horizon": 131072,
    "formal_steps": 125000,
    "world_size": 64,
    "per_rank_bs": 32,
    "global_batch": 2048,
}


def coordinate(
    step: int,
    rank: int,
    slot: int,
    *,
    formal: bool,
    field: str = "within_formal",
) -> dict:
    return {
        "step": step,
        "rank": rank,
        "slot": slot,
        "bucket_idx": 9,
        field: formal,
    }


class GenericCoordinateValidationTest(unittest.TestCase):
    def validate(self, value: list[dict]):
        return validate_plan_coordinates(
            value,
            bucket_idx=9,
            label="w64-test",
            **W64_CONTRACT,
        )

    def test_w64_boundaries_and_formal_partition(self) -> None:
        value = [
            coordinate(0, 0, 0, formal=True),
            coordinate(124999, 63, 31, formal=True),
            coordinate(125000, 0, 0, formal=False),
            coordinate(131071, 63, 31, formal=False),
        ]
        observed, formal = self.validate(value)
        self.assertEqual(observed, value)
        self.assertEqual(formal, value[:2])
        self.assertIsNot(observed, value)

    def test_plan_index_accepts_exact_step_transition(self) -> None:
        value = [
            coordinate(0, 63, 31, formal=True),
            coordinate(1, 0, 0, formal=True),
        ]
        observed, _ = self.validate(value)
        self.assertEqual(observed, value)

    def test_plan_index_rejects_reverse_and_duplicate_coordinates(self) -> None:
        reverse = [
            coordinate(10, 1, 0, formal=True),
            coordinate(10, 0, 31, formal=True),
        ]
        duplicate = [
            coordinate(10, 1, 0, formal=True),
            coordinate(10, 1, 0, formal=True),
        ]
        for value in (reverse, duplicate):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    EvidencePublishError, "strictly ordered by plan index"
                ):
                    self.validate(value)

    def test_topology_boundaries_are_fail_closed(self) -> None:
        for key, value in (
            ("step", 131072),
            ("rank", 64),
            ("slot", 32),
        ):
            item = coordinate(125000, 0, 0, formal=False)
            item[key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    EvidencePublishError, "topology/range drift"
                ):
                    self.validate([item])

    def test_formal_flag_and_global_batch_are_validated(self) -> None:
        with self.assertRaisesRegex(
            EvidencePublishError, "formal-boundary drift"
        ):
            self.validate([coordinate(124999, 0, 0, formal=False)])
        bad_contract = dict(W64_CONTRACT)
        bad_contract["global_batch"] = 2047
        with self.assertRaisesRegex(
            EvidencePublishError, "global_batch/topology drift"
        ):
            validate_plan_coordinates(
                [coordinate(0, 0, 0, formal=True)],
                bucket_idx=9,
                label="w64-test",
                **bad_contract,
            )


class LegacyW48CompatibilityTest(unittest.TestCase):
    def test_wrapper_signature_and_legacy_field_are_unchanged(self) -> None:
        self.assertEqual(
            list(inspect.signature(validate_stage512a_coordinates).parameters),
            ["value", "bucket_idx", "label"],
        )
        value = [
            coordinate(
                166666,
                47,
                31,
                formal=True,
                field="within_formal_166667",
            ),
            coordinate(
                166667,
                0,
                0,
                formal=False,
                field="within_formal_166667",
            ),
            coordinate(
                172031,
                47,
                31,
                formal=False,
                field="within_formal_166667",
            ),
        ]
        observed, formal = validate_stage512a_coordinates(
            value, bucket_idx=9, label="legacy-test"
        )
        self.assertEqual(observed, value)
        self.assertEqual(formal, value[:1])
        self.assertTrue(
            all("within_formal" not in item for item in observed)
        )

    def test_wrapper_does_not_accept_generic_field_in_legacy_evidence(self) -> None:
        with self.assertRaisesRegex(
            EvidencePublishError, "formal-boundary drift"
        ):
            validate_stage512a_coordinates(
                [coordinate(0, 0, 0, formal=True)],
                bucket_idx=9,
                label="legacy-test",
            )


class TextAtlasAuditCoordinateSchemaTest(unittest.TestCase):
    def test_schema_selects_only_its_bound_field(self) -> None:
        self.assertEqual(
            _resolve_within_formal_field("stage512a_w48_v1", None),
            "within_formal_166667",
        )
        self.assertEqual(
            _resolve_within_formal_field("generic_v1", "within_formal"),
            "within_formal",
        )
        with self.assertRaisesRegex(AuditError, "does not match"):
            _resolve_within_formal_field(
                "generic_v1", "within_formal_166667"
            )
        self.assertEqual(
            COORDINATE_SCHEMA_AUDIT_NAMES["stage512a_w48_v1"],
            "offline_arb_textatlas_plan_rows_v1",
        )
        self.assertEqual(
            COORDINATE_SCHEMA_AUDIT_NAMES["generic_v1"],
            "offline_arb_textatlas_plan_rows_generic_v1",
        )

    def test_coordinate_record_emits_only_selected_field(self) -> None:
        generic = _coordinate_record(
            step=124999,
            rank=63,
            slot=31,
            bucket_idx=9,
            formal_steps=125000,
            within_formal_field="within_formal",
        )
        legacy = _coordinate_record(
            step=166667,
            rank=0,
            slot=0,
            bucket_idx=9,
            formal_steps=166667,
            within_formal_field="within_formal_166667",
        )
        self.assertIs(generic["within_formal"], True)
        self.assertNotIn("within_formal_166667", generic)
        self.assertIs(legacy["within_formal_166667"], False)
        self.assertNotIn("within_formal", legacy)


if __name__ == "__main__":
    unittest.main()
