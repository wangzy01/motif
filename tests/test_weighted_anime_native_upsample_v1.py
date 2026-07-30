from training.dataline_offline.anime_native_upsample_exceptions_weighted_v1 import (
    ANIME_NATIVE_UPSAMPLE_EXCEPTIONS,
    AnimeNativeUpsampleBindingError,
    EXPOSURE_COUNT,
    REVALIDATION_REPORT_SHA256,
    bind_anime_native_upsample_exceptions,
)


def test_exact_weighted_binding():
    mapping = bind_anime_native_upsample_exceptions(
        stage="512a",
        eligibility="elig_512",
        plan_id="stage512a_offline_exact_v2_w64_256m_weighted_v1",
        plan_hash="7295ca937d5002a0cf5e207c1d466c5992352e395adc9ae1c33f3cb616f675db",
        catalog_hash="370b1679e8c5670338a03ba4c6972cdbf69fa5068fc35108bf646b2d64642331",
        plan_header_sha256="dc994b05147e33ee7515a986b3edb48e42c777b9257bc663dfa78c35a77a892d",
        catalog_header_sha256="51a5f0904987a751b54842633d2968ea7f6485307850858d9472319c15748d1d",
        decode_domain=lambda row_id: "anime",
    )
    assert mapping is ANIME_NATIVE_UPSAMPLE_EXCEPTIONS
    assert EXPOSURE_COUNT == 5
    assert REVALIDATION_REPORT_SHA256 == "dd1d5ffc9a519a3c5522fb4296fba2e9f9788efbcac6a79f0d73e484cd0ee26c"


def test_partial_binding_fails_closed():
    try:
        bind_anime_native_upsample_exceptions(
            stage="512a",
            eligibility="elig_512",
            plan_id="stage512a_offline_exact_v2_w64_256m_weighted_v1",
            plan_hash="0" * 64,
            catalog_hash="370b1679e8c5670338a03ba4c6972cdbf69fa5068fc35108bf646b2d64642331",
            plan_header_sha256="dc994b05147e33ee7515a986b3edb48e42c777b9257bc663dfa78c35a77a892d",
            catalog_header_sha256="51a5f0904987a751b54842633d2968ea7f6485307850858d9472319c15748d1d",
            decode_domain=lambda row_id: "anime",
        )
    except AnimeNativeUpsampleBindingError:
        return
    raise AssertionError("partial production binding did not fail closed")
