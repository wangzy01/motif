import copy

from training.dataline_offline import (
    anime_native_upsample_exceptions_control_rngfix_v1 as control_anime,
)
from training.dataline_offline import (
    anime_native_upsample_exceptions_weighted_512b_v1 as stage512b_anime,
)
from training.dataline_offline import (
    anime_native_upsample_exceptions_weighted_v1 as weighted_anime,
)
from training.dataline_offline.caption_exceptions_v1 import bind_caption_exceptions
from training.dataline_offline.image_exceptions_v1 import (
    STAGE512A_CONTROL_TEXTATLAS_FORMAL_EXPOSURES,
    STAGE512A_CONTROL_TEXTATLAS_JPEG_EOI_REPAIRS,
    ImageExceptionBindingError,
    bind_image_exceptions,
)
from training.dataline_offline.offline_dataset import (
    _bind_anime_native_upsample_registry,
)
from training.dataline_offline.stage512a_control_registry_closure_v1 import (
    Stage512aControlRegistryClosureError,
    validate_closure_payloads,
    validate_stage512a_control_registry_closure,
)


def _control_common():
    return {
        "stage": control_anime.STAGE,
        "eligibility": control_anime.ELIGIBILITY,
        "plan_id": control_anime.PLAN_ID,
        "plan_hash": control_anime.PLAN_HASH,
        "catalog_hash": control_anime.CATALOG_HASH,
        "plan_header_sha256": control_anime.PLAN_HEADER_SHA256,
        "catalog_header_sha256": control_anime.CATALOG_HEADER_SHA256,
    }


def _must_raise(error, function, **kwargs):
    try:
        function(**kwargs)
    except error:
        return
    raise AssertionError(f"{function.__name__} did not fail closed")


def test_exact_control_bindings_and_selector():
    common = _control_common()
    image = bind_image_exceptions(
        **common,
        decode_domain=lambda _row_id: "textatlas",
    )
    assert image is STAGE512A_CONTROL_TEXTATLAS_JPEG_EOI_REPAIRS
    assert set(image) == {68_973_396, 69_010_633, 70_063_978}
    assert dict(STAGE512A_CONTROL_TEXTATLAS_FORMAL_EXPOSURES) == {
        68_973_396: 3,
        69_010_633: 3,
        70_063_978: 3,
    }

    anime = control_anime.bind_stage512a_control_anime_native_upsample_exceptions(
        **common,
        decode_domain=lambda _row_id: "anime",
    )
    assert anime is control_anime.STAGE512A_CONTROL_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS
    assert control_anime.CLOSURE == (6, 125_000, 64, 32)
    assert sorted(
        (coordinate.step, coordinate.rank, coordinate.slot)
        for coordinate in anime[11_163_647].coordinates
    ) == [
        (11_486, 35, 1),
        (50_429, 7, 2),
        (60_035, 46, 0),
        (95_244, 16, 1),
        (119_964, 33, 0),
        (121_904, 58, 1),
    ]
    selected, closure = _bind_anime_native_upsample_registry(
        **common,
        decode_domain=lambda _row_id: "anime",
    )
    assert selected is anime and closure == control_anime.CLOSURE

    caption_common = dict(common)
    caption_common.pop("plan_hash")
    assert not bind_caption_exceptions(
        **caption_common,
        decode_domain=lambda _row_id: "monet_real",
    )


def test_control_identity_fail_closed():
    common = _control_common()
    mutations = {
        "stage": "512b",
        "eligibility": "wrong",
        "plan_id": "unknown-plan",
        "plan_hash": "0" * 64,
        "catalog_hash": "1" * 64,
        "plan_header_sha256": "2" * 64,
        "catalog_header_sha256": "3" * 64,
    }
    for name, value in mutations.items():
        actual = dict(common)
        actual[name] = value
        _must_raise(
            ImageExceptionBindingError,
            bind_image_exceptions,
            **actual,
            decode_domain=lambda _row_id: "textatlas",
        )
        _must_raise(
            weighted_anime.AnimeNativeUpsampleBindingError,
            control_anime.bind_stage512a_control_anime_native_upsample_exceptions,
            **actual,
            decode_domain=lambda _row_id: "anime",
        )

    unknown = dict(common)
    unknown.update(
        plan_id="unknown-plan",
        plan_hash="0" * 64,
        plan_header_sha256="1" * 64,
    )
    assert not bind_image_exceptions(
        **unknown,
        decode_domain=lambda _row_id: "textatlas",
    )
    assert not control_anime.bind_stage512a_control_anime_native_upsample_exceptions(
        **unknown,
        decode_domain=lambda _row_id: "anime",
    )

    mixed = dict(common)
    mixed["plan_id"] = weighted_anime.PLAN_ID
    _must_raise(
        ImageExceptionBindingError,
        bind_image_exceptions,
        **mixed,
        decode_domain=lambda _row_id: "textatlas",
    )
    _must_raise(
        weighted_anime.AnimeNativeUpsampleBindingError,
        _bind_anime_native_upsample_registry,
        **mixed,
        decode_domain=lambda _row_id: "anime",
    )
    mixed["plan_id"] = stage512b_anime.PLAN_ID
    _must_raise(
        weighted_anime.AnimeNativeUpsampleBindingError,
        _bind_anime_native_upsample_registry,
        **mixed,
        decode_domain=lambda _row_id: "anime",
    )


def test_legacy_selector_closures_unchanged():
    weighted = {
        "stage": weighted_anime.STAGE,
        "eligibility": weighted_anime.ELIGIBILITY,
        "plan_id": weighted_anime.PLAN_ID,
        "plan_hash": weighted_anime.PLAN_HASH,
        "catalog_hash": weighted_anime.CATALOG_HASH,
        "plan_header_sha256": weighted_anime.PLAN_HEADER_SHA256,
        "catalog_header_sha256": weighted_anime.CATALOG_HEADER_SHA256,
        "decode_domain": lambda _row_id: "anime",
    }
    mapping, closure = _bind_anime_native_upsample_registry(**weighted)
    assert mapping is weighted_anime.ANIME_NATIVE_UPSAMPLE_EXCEPTIONS
    assert closure is None

    stage512b = {
        "stage": stage512b_anime.STAGE,
        "eligibility": stage512b_anime.ELIGIBILITY,
        "plan_id": stage512b_anime.PLAN_ID,
        "plan_hash": stage512b_anime.PLAN_HASH,
        "catalog_hash": stage512b_anime.CATALOG_HASH,
        "plan_header_sha256": stage512b_anime.PLAN_HEADER_SHA256,
        "catalog_header_sha256": stage512b_anime.CATALOG_HEADER_SHA256,
        "decode_domain": lambda _row_id: "anime",
    }
    mapping, closure = _bind_anime_native_upsample_registry(**stage512b)
    assert mapping is stage512b_anime.STAGE512B_ANIME_NATIVE_UPSAMPLE_EXCEPTIONS
    assert closure == (8, 250_000, 64, 32)


def test_sealed_closure_and_mutation_rejection():
    receipt, report = validate_stage512a_control_registry_closure()

    bad = copy.deepcopy(report)
    bad["captionless_monet_zero_hit_rows"]["19760606"][
        "realized_exposure_count"
    ] = 1
    _must_raise(
        Stage512aControlRegistryClosureError,
        validate_closure_payloads,
        receipt=receipt,
        report=bad,
    )

    bad = copy.deepcopy(report)
    bad["anime_native_upsample"]["11163647"]["realized_coordinates"].pop()
    _must_raise(
        Stage512aControlRegistryClosureError,
        validate_closure_payloads,
        receipt=receipt,
        report=bad,
    )

    bad_receipt = copy.deepcopy(receipt)
    bad_receipt["full_256m_closure"]["textatlas_exposure_counts"]["68973396"] = 4
    _must_raise(
        Stage512aControlRegistryClosureError,
        validate_closure_payloads,
        receipt=bad_receipt,
        report=report,
    )
