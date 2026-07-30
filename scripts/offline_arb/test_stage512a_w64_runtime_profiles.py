#!/usr/bin/env python3
"""Small-object tests for additive W64 runtime-exception activation."""

from __future__ import annotations

import unittest

from training.dataline_offline import caption_exceptions_v1 as caption
from training.dataline_offline import image_exceptions_v1 as image


def _domain(row_id: int) -> str:
    if row_id in caption.STAGE512A_W64_CAPTIONLESS_MONET_ROWS:
        return "monet_real"
    if row_id in image.STAGE512A_W64_TEXTATLAS_JPEG_EOI_REPAIRS:
        return "textatlas"
    raise AssertionError(f"unexpected reviewed row: {row_id}")


class RuntimeProfileTest(unittest.TestCase):
    def test_w48_public_abi_is_unchanged(self) -> None:
        self.assertEqual(
            (
                caption.STAGE512A_STAGE,
                caption.STAGE512A_ELIGIBILITY,
                caption.STAGE512A_PLAN_ID,
                caption.STAGE512A_CATALOG_HASH,
                caption.STAGE512A_PLAN_HEADER_SHA256,
                caption.STAGE512A_CATALOG_HEADER_SHA256,
                caption.STAGE512A_SIDECAR_HEADER_SHA256,
                caption.STAGE512A_AUDIT_JSON_SHA256,
            ),
            (
                "512a",
                "elig_512",
                "stage512a_offline_exact_v2_w48_prod",
                "370b1679e8c5670338a03ba4c6972cdbf69fa5068fc35108bf646b2d64642331",
                "939df84dd066d93c36cc10606a57e1dfc4e9d4c174f0b658c45ab2cad35cbd0d",
                "51a5f0904987a751b54842633d2968ea7f6485307850858d9472319c15748d1d",
                "72fe0c4d5b5258a1a6320a6ab57f422cd815dd26a8a4a29594fefdd7abfa7224",
                "9c1d809365de3badefe011d192b197615ff13504dbe470a2c4387636f358b0ad",
            ),
        )
        self.assertEqual(
            dict(caption.STAGE512A_CAPTIONLESS_MONET_ROWS),
            {
                19_760_606: "monet:058_6f9a1ddf56d9e468",
                22_783_601: "monet:023_a11e3e8b8dc53b6a",
                29_778_338: "monet:023_6af995e4741f26cd",
                29_779_948: "monet:023_df29c87ed5939c37",
                30_281_293: "monet:058_9ca806c37710fd48",
                35_574_104: "monet:056_dbaeba5eada7a499",
            },
        )
        self.assertEqual(
            dict(caption.STAGE512A_CAPTIONLESS_MONET_EXPOSURES),
            {
                19_760_606: 3,
                22_783_601: 3,
                29_778_338: 4,
                29_779_948: 3,
                30_281_293: 3,
                35_574_104: 3,
            },
        )
        self.assertEqual(caption.STAGE512A_CAPTIONLESS_MONET_TOTAL_EXPOSURES, 19)
        self.assertEqual(
            (
                image.STAGE512A_STAGE,
                image.STAGE512A_ELIGIBILITY,
                image.STAGE512A_PLAN_ID,
                image.STAGE512A_PLAN_HASH,
                image.STAGE512A_CATALOG_HASH,
                image.STAGE512A_PLAN_HEADER_SHA256,
                image.STAGE512A_CATALOG_HEADER_SHA256,
                image.STAGE512A_INTEGRITY_AUDIT_SHA256,
                image.STAGE512A_ALLOWLIST_SUGGESTION_SHA256,
                image.STAGE512A_REPAIRED_PAYLOAD_BINDINGS_SHA256,
                image.STAGE512A_FINAL_REVIEW_SHA256,
            ),
            (
                "512a",
                "elig_512",
                "stage512a_offline_exact_v2_w48_prod",
                "72363201789fc72803219604887095d287b5d52bbfc415a64744e84f99899a8a",
                "370b1679e8c5670338a03ba4c6972cdbf69fa5068fc35108bf646b2d64642331",
                "939df84dd066d93c36cc10606a57e1dfc4e9d4c174f0b658c45ab2cad35cbd0d",
                "51a5f0904987a751b54842633d2968ea7f6485307850858d9472319c15748d1d",
                "9d2b42eed098a1808589f418402cbfa19ffa5278852fa40d166ed31d89450a6f",
                "29fdce7368823a943a688434be095996d26ad4309594762f3ae9408150d2b932",
                "0450ba03cfa6dfca1a2b1b3cacc27c6c6d50ae40a4eda31a7589d95913a923d5",
                "d1734949e27c4ef8cdc037e9c37dbece1ad463d2f75f01139093189f57f9c4c3",
            ),
        )
        self.assertEqual(
            dict(image.STAGE512A_TEXTATLAS_FORMAL_EXPOSURES),
            {68_973_396: 4, 69_010_633: 3, 70_063_978: 3},
        )
        self.assertEqual(image.STAGE512A_TEXTATLAS_TOTAL_FORMAL_EXPOSURES, 10)
        self.assertEqual(
            {
                row_id: (
                    repair.uid,
                    repair.raw_sha256,
                    repair.repaired_sha256,
                    repair.pixel_sha256,
                )
                for row_id, repair in image.STAGE512A_TEXTATLAS_JPEG_EOI_REPAIRS.items()
            },
            {
                68_973_396: (
                    "PPT2Structured:986e2d6790064c0d9db822e405cfd804.jpg",
                    "2a84efd18e5c66e72cfb26f34c3195db6ed97becd02633ab7c2bd8449dcc45d2",
                    "0522d5ea8705494f9c8ebfb8442bacc888d6916dbc5659e22e8cde8f2ed7fc4a",
                    "120051cba9c51f1ad620acaf49a3ca3105845d60a947a5ef9f9a6ff69d76f79f",
                ),
                69_010_633: (
                    "PPT2Structured:fbc3c5a38a12459b92ed77870494e609.jpg",
                    "bbd73e4673977154cb76665bdbb29ca43227d710b23e77a4c2a10348883fd460",
                    "e38ba338c5c12c453e1a17fd57bdac536f165f175f0d1c3c181925bb39b3a361",
                    "7b44b9a43b7f221fff237018c312d696dd03cc9c1854585a770c9ba9ab6643aa",
                ),
                70_063_978: (
                    "TextVisionBlend:716f0b1cc7134b84bd6a098a55a70425.jpg",
                    "1b9bec49cef9e0b04709c2447712ef9b4a3825d18d60a000facc4e3631d5363c",
                    "03e276bb716de0d10da58fef1b22dd6f8458972012cd053d2ba6c2c6cf8a9767",
                    "bbb0b6cf52ed0430b7d52ebf5bf578420098ba0a54fdd65be6822024ae52089a",
                ),
            },
        )

    def test_exact_w64_caption_profile_activates(self) -> None:
        observed = caption.bind_caption_exceptions(
            stage=caption.STAGE512A_W64_STAGE,
            eligibility=caption.STAGE512A_W64_ELIGIBILITY,
            plan_id=caption.STAGE512A_W64_PLAN_ID,
            catalog_hash=caption.STAGE512A_W64_CATALOG_HASH,
            plan_header_sha256=caption.STAGE512A_W64_PLAN_HEADER_SHA256,
            catalog_header_sha256=caption.STAGE512A_W64_CATALOG_HEADER_SHA256,
            decode_domain=_domain,
        )
        self.assertEqual(dict(observed), dict(caption.STAGE512A_W64_CAPTIONLESS_MONET_ROWS))

    def test_exact_w64_image_profile_activates(self) -> None:
        observed = image.bind_image_exceptions(
            stage=image.STAGE512A_W64_STAGE,
            eligibility=image.STAGE512A_W64_ELIGIBILITY,
            plan_id=image.STAGE512A_W64_PLAN_ID,
            plan_hash=image.STAGE512A_W64_PLAN_HASH,
            catalog_hash=image.STAGE512A_W64_CATALOG_HASH,
            plan_header_sha256=image.STAGE512A_W64_PLAN_HEADER_SHA256,
            catalog_header_sha256=image.STAGE512A_W64_CATALOG_HEADER_SHA256,
            decode_domain=_domain,
        )
        self.assertEqual(dict(observed), dict(image.STAGE512A_W64_TEXTATLAS_JPEG_EOI_REPAIRS))

    def test_w16_gpuval_and_unrelated_plans_do_not_activate(self) -> None:
        caption_result = caption.bind_caption_exceptions(
            stage="512a",
            eligibility="elig_512",
            plan_id="stage512a_offline_exact_v2_w16_gpuval",
            catalog_hash=caption.STAGE512A_W64_CATALOG_HASH,
            plan_header_sha256="1" * 64,
            catalog_header_sha256=caption.STAGE512A_W64_CATALOG_HEADER_SHA256,
            decode_domain=_domain,
        )
        image_result = image.bind_image_exceptions(
            stage="512a",
            eligibility="elig_512",
            plan_id="stage512a_offline_exact_v2_w16_gpuval",
            plan_hash="2" * 64,
            catalog_hash=image.STAGE512A_W64_CATALOG_HASH,
            plan_header_sha256="3" * 64,
            catalog_header_sha256=image.STAGE512A_W64_CATALOG_HEADER_SHA256,
            decode_domain=_domain,
        )
        self.assertEqual(dict(caption_result), {})
        self.assertEqual(dict(image_result), {})

    def test_caption_w48_w64_signal_mix_is_fatal(self) -> None:
        with self.assertRaisesRegex(
            caption.CaptionExceptionBindingError, "conflicting profiles"
        ):
            caption.bind_caption_exceptions(
                stage="512a",
                eligibility="elig_512",
                plan_id=caption.STAGE512A_PLAN_ID,
                catalog_hash=caption.STAGE512A_CATALOG_HASH,
                plan_header_sha256=caption.STAGE512A_W64_PLAN_HEADER_SHA256,
                catalog_header_sha256=caption.STAGE512A_CATALOG_HEADER_SHA256,
                decode_domain=_domain,
            )

    def test_image_w48_w64_signal_mix_is_fatal(self) -> None:
        with self.assertRaisesRegex(
            image.ImageExceptionBindingError, "conflicting profiles"
        ):
            image.bind_image_exceptions(
                stage="512a",
                eligibility="elig_512",
                plan_id=image.STAGE512A_PLAN_ID,
                plan_hash=image.STAGE512A_W64_PLAN_HASH,
                catalog_hash=image.STAGE512A_CATALOG_HASH,
                plan_header_sha256=image.STAGE512A_PLAN_HEADER_SHA256,
                catalog_header_sha256=image.STAGE512A_CATALOG_HEADER_SHA256,
                decode_domain=_domain,
            )

    def test_partial_w64_identity_is_fatal(self) -> None:
        with self.assertRaisesRegex(
            caption.CaptionExceptionBindingError, "production binding mismatch"
        ):
            caption.bind_caption_exceptions(
                stage="512a",
                eligibility="elig_512",
                plan_id=caption.STAGE512A_W64_PLAN_ID,
                catalog_hash=caption.STAGE512A_W64_CATALOG_HASH,
                plan_header_sha256="4" * 64,
                catalog_header_sha256=caption.STAGE512A_W64_CATALOG_HEADER_SHA256,
                decode_domain=_domain,
            )


if __name__ == "__main__":
    unittest.main()
