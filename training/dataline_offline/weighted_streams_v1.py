"""Deterministic weighted row streams for the Stage512a offline plan.

This module contains no filesystem publication logic.  It turns catalog row IDs
and streaming weights into deterministic finite sequences, and provides a
consumer that removes within-global-batch collisions by deterministic swaps.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from typing import Any, Iterable, Sequence

import numpy as np


ALGORITHM = "stage512a_weighted_streams_v1"
FEISTEL_ROUNDS = 6
_U64_DENOM = float(1 << 64)


class WeightedStreamError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def seed_digest(plan_id: str, seed: int, label: str, *parts: Any) -> bytes:
    return hashlib.sha256(
        canonical_bytes(
            {
                "algorithm": ALGORITHM,
                "plan_id": plan_id,
                "seed": int(seed),
                "label": label,
                "parts": list(parts),
            }
        )
    ).digest()


def rng_for(plan_id: str, seed: int, label: str, *parts: Any) -> np.random.Generator:
    entropy = np.frombuffer(seed_digest(plan_id, seed, label, *parts), dtype="<u8")
    return np.random.default_rng(np.random.SeedSequence(entropy.tolist()))


def largest_remainder(total: int, weights: Sequence[Decimal | float | str]) -> np.ndarray:
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise WeightedStreamError(f"invalid largest-remainder total: {total!r}")
    parsed = [Decimal(str(value)) for value in weights]
    if any(not value.is_finite() or value < 0 for value in parsed):
        raise WeightedStreamError(f"invalid largest-remainder weights: {weights!r}")
    norm = sum(parsed, Decimal(0))
    if total and norm <= 0:
        raise WeightedStreamError("positive total has zero weight mass")
    if total == 0:
        return np.zeros(len(parsed), dtype=np.int64)
    ideals = [Decimal(total) * value / norm for value in parsed]
    floors = [int(value.to_integral_value(rounding=ROUND_FLOOR)) for value in ideals]
    missing = total - sum(floors)
    order = sorted(range(len(parsed)), key=lambda i: (-(ideals[i] - floors[i]), i))
    for index in order[:missing]:
        floors[index] += 1
    result = np.asarray(floors, dtype=np.int64)
    if int(result.sum(dtype=np.int64)) != total:
        raise AssertionError("largest remainder did not conserve total")
    return result


def deterministic_band_schedule(
    total: int,
    weights: Sequence[Decimal | float | str],
    available: Sequence[bool],
    *,
    plan_id: str,
    seed: int,
    label: str,
    parts: Sequence[Any] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Return an exact, deterministically shuffled caption-band schedule.

    Unsupported bands receive zero slots.  Configured weights are
    renormalized over supported bands, matching the streaming sampler's redraw
    behavior when a particular AR/domain cell cannot serve a band.
    """

    if len(weights) != len(available) or not len(weights):
        raise WeightedStreamError("band weight/availability shape mismatch")
    supported_weights = [
        value if bool(is_available) else Decimal(0)
        for value, is_available in zip(weights, available)
    ]
    quotas = largest_remainder(total, supported_weights)
    if np.any(quotas[np.logical_not(np.asarray(available, dtype=bool))] != 0):
        raise AssertionError("unsupported band received a quota")
    schedule = np.repeat(
        np.arange(len(weights), dtype=np.int8), quotas.astype(np.int64, copy=False)
    )
    if len(schedule):
        order = permutation(
            plan_id, seed, f"{label}:band_schedule", len(schedule), *parts
        )
        schedule = schedule[order]
    if schedule.shape != (total,):
        raise AssertionError("band schedule length drift")
    return schedule, quotas


def collision_reserve_size(
    total: int,
    max_batch_take: int,
    *,
    fraction: Decimal | float | str = "0.05",
    minimum: int = 65_536,
    batch_multiple: int = 32,
) -> int:
    """Conservative deterministic reserve for cross-band/batch de-dup swaps."""

    if total < 0 or max_batch_take < 0 or minimum < 0 or batch_multiple < 0:
        raise WeightedStreamError("negative collision-reserve argument")
    frac = Decimal(str(fraction))
    if not frac.is_finite() or frac < 0:
        raise WeightedStreamError("invalid collision-reserve fraction")
    fractional = int(
        (Decimal(total) * frac).to_integral_value(rounding=ROUND_FLOOR)
    )
    return max(minimum, batch_multiple * max_batch_take, fractional)


def segment_label_maxima(
    schedule: np.ndarray,
    segment_lengths: np.ndarray,
    num_labels: int,
) -> np.ndarray:
    """Exact maximum occurrences of each label in variable-length segments."""

    counts = segment_label_counts(schedule, segment_lengths, num_labels)
    return counts.max(axis=0, initial=0)


def segment_label_counts(
    schedule: np.ndarray,
    segment_lengths: np.ndarray,
    num_labels: int,
) -> np.ndarray:
    """Exact per-segment counts for a finite integer-label schedule."""

    labels = np.asarray(schedule)
    lengths = np.asarray(segment_lengths, dtype=np.int64)
    if labels.ndim != 1 or lengths.ndim != 1 or np.any(lengths < 0):
        raise WeightedStreamError("invalid schedule/segment shape")
    if int(lengths.sum(dtype=np.int64)) != len(labels):
        raise WeightedStreamError("segment lengths do not close over schedule")
    if num_labels <= 0 or np.any(labels < 0) or np.any(labels >= num_labels):
        raise WeightedStreamError("schedule label outside declared range")
    starts = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64))
    )
    ends = starts + lengths
    nonempty = lengths > 0
    result = np.zeros((len(lengths), num_labels), dtype=np.int64)
    for label in range(num_labels):
        prefix = np.empty(len(labels) + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(labels == label, dtype=np.int64, out=prefix[1:])
        result[nonempty, label] = prefix[ends[nonempty]] - prefix[starts[nonempty]]
    return result


def _splitmix64_array(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = values.astype(np.uint64, copy=False) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _feistel_power_of_four(
    values: np.ndarray, half_bits: int, round_keys: Sequence[int]
) -> np.ndarray:
    mask = np.uint64((1 << half_bits) - 1)
    left = values >> np.uint64(half_bits)
    right = values & mask
    for key in round_keys:
        mixed = _splitmix64_array(right ^ np.uint64(key)) & mask
        left, right = right, (left ^ mixed) & mask
    return (left << np.uint64(half_bits)) | right


def feistel_permute_many(
    positions: np.ndarray, n: int, round_keys: Sequence[int]
) -> np.ndarray:
    if n <= 0:
        raise WeightedStreamError("cannot permute an empty range")
    if n == 1:
        return np.zeros(positions.shape, dtype=np.int64)
    bits = (n - 1).bit_length()
    if bits % 2:
        bits += 1
    half_bits = bits // 2
    current = positions.astype(np.uint64, copy=True)
    active = np.ones(current.shape, dtype=bool)
    while bool(np.any(active)):
        current[active] = _feistel_power_of_four(current[active], half_bits, round_keys)
        active = current >= np.uint64(n)
    return current.astype(np.int64)


def permutation(plan_id: str, seed: int, label: str, n: int, *parts: Any) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    master = seed_digest(plan_id, seed, label, *parts)
    keys = tuple(
        int.from_bytes(hashlib.sha256(master + bytes((round_index,))).digest()[:8], "little")
        for round_index in range(FEISTEL_ROUNDS)
    )
    return feistel_permute_many(np.arange(n, dtype=np.uint64), n, keys)


def _validate_rows_weights(row_ids: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(row_ids, dtype=np.int64)
    mass = np.asarray(weights, dtype=np.float64)
    if rows.ndim != 1 or mass.ndim != 1 or rows.shape != mass.shape:
        raise WeightedStreamError(f"row/weight shape mismatch: {rows.shape}/{mass.shape}")
    if rows.size and np.any(rows[1:] <= rows[:-1]):
        raise WeightedStreamError("row IDs must be strictly increasing")
    if not np.all(np.isfinite(mass)) or np.any(mass < 0):
        raise WeightedStreamError("weights must be finite and non-negative")
    positive = mass > 0
    rows = rows[positive]
    mass = mass[positive]
    if not len(rows):
        raise WeightedStreamError("weighted stream has no positive-mass row")
    return rows, mass


def systematic_weighted_sequence(
    row_ids: np.ndarray,
    weights: np.ndarray,
    total: int,
    *,
    plan_id: str,
    seed: int,
    label: str,
    parts: Sequence[Any] = (),
    reserve: int = 0,
    chunk_size: int = 4_000_000,
) -> np.ndarray:
    """Low-variance deterministic sampling proportional to positive weights.

    `reserve` appends independently sampled tail positions.  They are not part
    of the requested exposure count; a caller may swap them into the consumed
    prefix when the final global batch would otherwise contain a duplicate.
    """

    if total < 0 or reserve < 0:
        raise WeightedStreamError("negative systematic sequence size")
    rows, mass = _validate_rows_weights(row_ids, weights)
    requested = total + reserve
    if requested == 0:
        return np.empty(0, dtype=np.int64)

    row_order = permutation(plan_id, seed, f"{label}:row_order", len(rows), *parts)
    ordered_rows = rows[row_order]
    ordered_mass = mass[row_order]
    cumulative = np.cumsum(ordered_mass, dtype=np.float64)
    mass_total = float(cumulative[-1])
    if not math.isfinite(mass_total) or mass_total <= 0:
        raise WeightedStreamError("systematic stream has invalid total mass")

    rng = rng_for(plan_id, seed, f"{label}:systematic_offset", *parts)
    offset = float(rng.random())
    selected = np.empty(requested, dtype=np.int64)
    for start in range(0, requested, chunk_size):
        stop = min(requested, start + chunk_size)
        points = (np.arange(start, stop, dtype=np.float64) + offset) * (
            mass_total / requested
        )
        indexes = np.searchsorted(cumulative, points, side="right")
        if np.any(indexes < 0) or np.any(indexes >= len(ordered_rows)):
            raise AssertionError("systematic search produced an out-of-range row")
        selected[start:stop] = ordered_rows[indexes]

    exposure_order = permutation(
        plan_id, seed, f"{label}:exposure_order", requested, *parts
    )
    shuffled = selected[exposure_order]
    if shuffled.shape != (requested,):
        raise AssertionError("systematic sequence shape drift")
    return shuffled


def compute_repeat_counts(
    priority: np.ndarray,
    max_repeat: int,
    extra_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact integer-repeat rule used by the streaming Anime sampler."""

    values = np.asarray(priority, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise WeightedStreamError("invalid Anime priority vector")
    n = len(values)
    counts = np.ones(n, dtype=np.int16)
    if n == 0 or max_repeat <= 1:
        return counts
    score = np.maximum(values - 1.0, 0.0)
    if not np.any(score > 0):
        return counts
    budget = min(int(round(n * float(extra_frac))), n * (int(max_repeat) - 1))
    remaining = budget
    while remaining > 0:
        eligible = np.flatnonzero((counts < max_repeat) & (score > 0))
        if eligible.size == 0:
            break
        probs = score[eligible].astype(np.float64, copy=True)
        probs /= probs.sum(dtype=np.float64)
        take = min(remaining, int(eligible.size))
        chosen = rng.choice(eligible, size=take, replace=False, p=probs)
        counts[chosen] += 1
        remaining -= take
    return counts


def anime_repeat_sequence(
    row_ids: np.ndarray,
    priority: np.ndarray,
    total: int,
    *,
    max_repeat: int,
    extra_frac: float,
    plan_id: str,
    seed: int,
    label: str,
    parts: Sequence[Any] = (),
    reserve: int = 0,
) -> np.ndarray:
    rows = np.asarray(row_ids, dtype=np.int64)
    values = np.asarray(priority, dtype=np.float64)
    if rows.ndim != 1 or values.shape != rows.shape or not len(rows):
        raise WeightedStreamError("invalid Anime rows/priority shape")
    if np.any(rows[1:] <= rows[:-1]):
        raise WeightedStreamError("Anime row IDs must be strictly increasing")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise WeightedStreamError("Anime priority must be finite and positive")
    requested = total + reserve
    if requested == 0:
        return np.empty(0, dtype=np.int64)

    chunks: list[np.ndarray] = []
    produced = 0
    cycle = 0
    while produced < requested:
        rng = rng_for(plan_id, seed, f"{label}:repeat", *parts, cycle)
        counts = compute_repeat_counts(values, max_repeat, extra_frac, rng)
        cycle_rows = np.repeat(rows, counts.astype(np.int64))
        order = permutation(
            plan_id,
            seed,
            f"{label}:cycle_order",
            len(cycle_rows),
            *parts,
            cycle,
        )
        cycle_rows = cycle_rows[order]
        take = min(requested - produced, len(cycle_rows))
        chunks.append(cycle_rows[:take])
        produced += take
        cycle += 1
    result = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    if result.shape != (requested,):
        raise AssertionError("Anime repeat sequence shape drift")
    return result


@dataclass
class MutableSequenceStream:
    sequence: np.ndarray
    limit: int
    cursor: int = 0
    collision_swaps: int = 0

    def __post_init__(self) -> None:
        self.sequence = np.asarray(self.sequence, dtype=np.int64)
        if self.sequence.ndim != 1 or self.limit < 0 or self.limit > len(self.sequence):
            raise WeightedStreamError("invalid mutable sequence/limit")

    @property
    def remaining(self) -> int:
        return self.limit - self.cursor

    def take(self, count: int, forbidden: Iterable[int] = ()) -> np.ndarray:
        if count < 0 or count > self.remaining:
            raise WeightedStreamError(
                f"stream take {count} exceeds remaining prefix {self.remaining}"
            )
        if count == 0:
            return np.empty(0, dtype=np.int64)
        if isinstance(forbidden, np.ndarray):
            forbidden_values = np.asarray(forbidden, dtype=np.int64).reshape(-1)
        else:
            forbidden_values = np.fromiter(
                (int(value) for value in forbidden), dtype=np.int64
            )
        start = self.cursor
        stop = start + count
        window = self.sequence[start:stop]
        if (
            len(np.unique(window)) == count
            and (
                not len(forbidden_values)
                or not bool(np.isin(window, forbidden_values).any())
            )
        ):
            self.cursor = stop
            return window.copy()
        used = {int(value) for value in forbidden_values.tolist()}
        scan = stop
        for position in range(start, stop):
            value = int(self.sequence[position])
            if value in used:
                while scan < len(self.sequence) and int(self.sequence[scan]) in used:
                    scan += 1
                if scan >= len(self.sequence):
                    raise WeightedStreamError(
                        "collision reserve exhausted while enforcing global-batch uniqueness"
                    )
                self.sequence[position], self.sequence[scan] = (
                    self.sequence[scan],
                    self.sequence[position],
                )
                value = int(self.sequence[position])
                scan += 1
                self.collision_swaps += 1
            used.add(value)
        self.cursor = stop
        result = self.sequence[start:stop].copy()
        if len(np.unique(result)) != count:
            raise AssertionError("mutable stream returned a duplicate row")
        return result


class BandBoundCellStream:
    """Consume an exact band schedule and per-band weighted row streams."""

    def __init__(
        self,
        band_schedule: np.ndarray,
        row_streams: dict[int, MutableSequenceStream],
        band_order: Sequence[int] | None = None,
    ):
        self.band_schedule = np.asarray(band_schedule, dtype=np.int8)
        self.row_streams = dict(row_streams)
        self.cursor = 0
        if self.band_schedule.ndim != 1:
            raise WeightedStreamError("band schedule must be one-dimensional")
        if any(int(value) not in self.row_streams for value in np.unique(self.band_schedule)):
            raise WeightedStreamError("band schedule references a missing row stream")
        expected_order = set(int(value) for value in np.unique(self.band_schedule))
        self.band_order = (
            tuple(sorted(expected_order))
            if band_order is None
            else tuple(int(value) for value in band_order)
        )
        if set(self.band_order) != expected_order or len(self.band_order) != len(expected_order):
            raise WeightedStreamError("band order does not exactly cover scheduled bands")

    @property
    def remaining(self) -> int:
        return len(self.band_schedule) - self.cursor

    def take(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        if count < 0 or count > self.remaining:
            raise WeightedStreamError(f"band cell take {count} exceeds {self.remaining}")
        bands = self.band_schedule[self.cursor : self.cursor + count].copy()
        rows = np.empty(count, dtype=np.int64)
        used_chunks: list[np.ndarray] = []
        present = set(int(value) for value in np.unique(bands))
        for band in self.band_order:
            if band not in present:
                continue
            positions = np.flatnonzero(bands == band)
            forbidden = (
                np.concatenate(used_chunks)
                if used_chunks
                else np.empty(0, dtype=np.int64)
            )
            values = self.row_streams[band].take(len(positions), forbidden)
            rows[positions] = values
            used_chunks.append(values)
        self.cursor += count
        if len(np.unique(rows)) != count:
            raise AssertionError("band-bound cell returned duplicate rows")
        return rows, bands
