#!/usr/bin/env python3
"""Known-answer gates for bounded Anime refetch in the exact-plan loader."""

from __future__ import annotations

import hashlib
import warnings
from types import SimpleNamespace
from unittest import mock

import training.dataline_offline.offline_dataset as offline


URL_A = (
    "https://img.metanomaly.co/r2/"
    "ZGFuYm9vcnVfY2h1bmtfMDA3MTYkMjY4MDM1MDcyJDM3NTk5OA"
)
URL_B = (
    "https://img.metanomaly.co/r2/"
    "ZGFuYm9vcnVfY2h1bmtfMDA5MzUkMzE0NzUxNDg4MCQ0NDQxNDA"
)
PAYLOAD_A = b"frozen-anime-payload-a"
PAYLOAD_B = b"frozen-anime-payload-b"


def dataset():
    value = object.__new__(offline.OfflineExactDataset)
    value.anime_cache_dir = "/local/evan/anime_img_cache"
    value._anime_workers = 1
    value.rank = 6
    return value


def item(*, step: int, slot: int, row_id: int):
    return offline._PlannedItem(
        step=step,
        slot=slot,
        row_id=row_id,
        domain="anime",
        domain_idx=0,
        bucket_idx=0,
    )


def run():
    delays = list(offline._ANIME_EXACT_REFETCH_DELAYS_S)
    rounds = len(delays) + 1
    stats = {"ok": 0, "fail": 2, "reason:HTTP403": 2}

    # The first successful payload is returned byte-for-byte after two misses;
    # every call is bound to the same frozen URL/cache and only the first two
    # delays are consumed.
    with (
        mock.patch.object(
            offline,
            "fetch_url_bytes",
            side_effect=[None, None, PAYLOAD_A, AssertionError("over-fetch")],
        ) as fetch,
        mock.patch.object(offline, "fetch_stats_snapshot", return_value=stats),
        mock.patch.object(offline.time, "sleep") as sleep,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        result = offline.OfflineExactDataset._anime_payloads(
            dataset(), [(item(step=6358, slot=7, row_id=2_901_694), {"fetch_url": URL_A})]
        )
    assert result == {(6358, 7): PAYLOAD_A}
    assert hashlib.sha256(result[(6358, 7)]).digest() == hashlib.sha256(PAYLOAD_A).digest()
    assert fetch.call_count == 3
    assert fetch.call_args_list == [
        mock.call(URL_A, cache_dir="/local/evan/anime_img_cache")
    ] * 3
    assert sleep.call_args_list == [mock.call(delays[0]), mock.call(delays[1])]
    messages = [str(value.message) for value in caught]
    assert sum("anime exact-row fetch retry" in value for value in messages) == 2
    assert sum("anime exact-row fetch recovered" in value for value in messages) == 1

    # Exhaustion remains fail-closed: no row, URL, or payload substitution and
    # exactly the frozen number of rounds/delays.
    with (
        mock.patch.object(offline, "fetch_url_bytes", return_value=None) as fetch,
        mock.patch.object(offline, "fetch_stats_snapshot", return_value=stats),
        mock.patch.object(offline.time, "sleep") as sleep,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        try:
            offline.OfflineExactDataset._anime_payloads(
                dataset(),
                [(item(step=6358, slot=9, row_id=3_225_847), {"fetch_url": URL_B})],
            )
        except offline.OfflineLoaderError as exc:
            message = str(exc)
        else:
            raise AssertionError("permanent Anime failure did not fail closed")
    assert fetch.call_count == rounds
    assert fetch.call_args_list == [
        mock.call(URL_B, cache_dir="/local/evan/anime_img_cache")
    ] * rounds
    assert sleep.call_args_list == [mock.call(value) for value in delays]
    assert "row_id=3225847" in message and f"rounds={rounds}" in message and URL_B in message

    # Two planned items retain their own exact tokens and payloads.  A successful
    # first call never sleeps or performs an unnecessary second fetch.
    with (
        mock.patch.object(offline, "fetch_url_bytes", side_effect=[PAYLOAD_A, PAYLOAD_B]) as fetch,
        mock.patch.object(offline.time, "sleep") as sleep,
    ):
        result = offline.OfflineExactDataset._anime_payloads(
            dataset(),
            [
                (item(step=10, slot=1, row_id=101), {"fetch_url": URL_A}),
                (item(step=10, slot=2, row_id=202), {"fetch_url": URL_B}),
            ],
        )
    assert result == {(10, 1): PAYLOAD_A, (10, 2): PAYLOAD_B}
    assert fetch.call_args_list == [
        mock.call(URL_A, cache_dir="/local/evan/anime_img_cache"),
        mock.call(URL_B, cache_dir="/local/evan/anime_img_cache"),
    ]
    sleep.assert_not_called()

    # Programming errors are not hidden by the transport retry wrapper.
    with mock.patch.object(offline, "fetch_url_bytes", side_effect=RuntimeError("sentinel")) as fetch:
        try:
            offline.OfflineExactDataset._anime_payloads(
                dataset(), [(item(step=11, slot=3, row_id=303), {"fetch_url": URL_A})]
            )
        except RuntimeError as exc:
            assert str(exc) == "sentinel"
        else:
            raise AssertionError("programming exception was incorrectly retried")
    assert fetch.call_count == 1

    print(
        "ANIME_EXACT_REFETCH_KNOWN_ANSWER_PASS "
        f"rounds={rounds} delays={delays} payload_a_sha256={hashlib.sha256(PAYLOAD_A).hexdigest()}"
    )


if __name__ == "__main__":
    run()
