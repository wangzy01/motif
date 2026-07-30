from __future__ import annotations

import unittest

import numpy as np

from training.dataline_offline.offline_dataset import OfflineExactDataset, OfflineLoaderError


class _Plan:
    def batch(self, step: int, rank: int):
        assert (step, rank) == (7, 3)
        return (512, 512), np.asarray([10, 20], dtype=np.int64), np.asarray([2, 3], dtype=np.int8)


class _Decoder:
    def decode(self, row: int):
        return (1 if row == 10 else 2), 0, row


class WeightedLoaderBindingTest(unittest.TestCase):
    def make_dataset(self) -> OfflineExactDataset:
        dataset = OfflineExactDataset.__new__(OfflineExactDataset)
        dataset.plan = _Plan()
        dataset.plan_header = {"per_rank_bs": 2}
        dataset.rank = 3
        dataset._bucket_to_idx = {(512, 512): 0}
        dataset._row_decoder = _Decoder()
        dataset.domains = ["anime", "monet_real", "gpic", "synthetic", "textatlas"]
        return dataset

    def test_plan_band_is_carried_into_planned_items(self) -> None:
        bucket, items = self.make_dataset()._plan_items_for_step(7)
        self.assertEqual(bucket, 0)
        self.assertEqual([item.domain for item in items], ["monet_real", "gpic"])
        self.assertEqual([item.caption_band for item in items], [2, 3])

    def test_valid_real_band_is_accepted_exactly(self) -> None:
        record = {
            "row_id": 10,
            "base_w": 1.0,
            "served_mask": 0b1111,
            "band_captioner": [1, 2, 3, 4],
        }
        self.assertEqual(self.make_dataset()._validate_planned_band("monet_real", record, 2), 2)

    def test_zero_weight_and_unserved_real_rows_fail_closed(self) -> None:
        zero = {
            "row_id": 10,
            "base_w": 0.0,
            "served_mask": 0b1111,
            "band_captioner": [1, 1, 1, 1],
        }
        with self.assertRaisesRegex(OfflineLoaderError, "non-positive real row"):
            self.make_dataset()._validate_planned_band("monet_real", zero, 0)
        unserved = dict(zero, base_w=1.0, served_mask=0b0011)
        with self.assertRaisesRegex(OfflineLoaderError, "unserved caption band"):
            self.make_dataset()._validate_planned_band("gpic", unserved, 3)

    def test_non_real_band_and_synthetic_zero_weight_fail_closed(self) -> None:
        with self.assertRaisesRegex(OfflineLoaderError, "non-real planned row"):
            self.make_dataset()._validate_planned_band("anime", {"row_id": 1}, 0)
        with self.assertRaisesRegex(OfflineLoaderError, "non-positive Synthetic row"):
            self.make_dataset()._validate_planned_band(
                "synthetic", {"row_id": 30, "base_w": 0.0}, -1
            )
        self.assertEqual(
            self.make_dataset()._validate_planned_band(
                "synthetic", {"row_id": 31, "base_w": 1.0}, -1
            ),
            -1,
        )


if __name__ == "__main__":
    unittest.main()
