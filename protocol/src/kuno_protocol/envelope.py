"""Serving envelopes: which requests a worker's hardware can fit, per profile (MINING.md §6, PROTOCOL.md).

A worker on a card that cannot hold a profile's largest request (the open tier's RTX 4090 and 5090 classes) advertises,
at registration, the longest clip it serves at each size and frame rate:

    {"ltx-2.5-fast": {"1080p": {"16:9": {"24": 16, "25": 16, "48": 7, "50": 7}, "21:9": {...}, ...}, "720p": {...}}}

    profile id -> resolution -> aspect ratio -> fps -> the longest duration_s served

A (resolution, aspect ratio, fps) left out is not served at any duration. A profile the envelope leaves out, and a
registration without one (older workers, and every worker whose hardware holds the whole profile), serves the profile's
full limits.

Why durations rather than latent tokens: the gateway checks a job from its public GenerationParams alone, with a dict
lookup and no model-family formula (MiniMax H3 and LTX-2.5 count latents differently). The aspect ratio is a key because
sizes of one resolution differ in pixels (LTX 1080p 21:9 is 2560x1088, 16:9 is 1920x1088). For a fixed size and frame rate
a request's latent tokens only grow with its duration, so "duration_s <= the longest served" is exactly the worker's own
memory admission (kuno_worker.backends.quantized.admit).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .profiles import ModelProfile
    from .schemas import GenerationParams

# resolution -> aspect ratio -> fps -> the longest duration_s served. Stored JSON has string fps keys.
EnvelopeTable = dict[str, dict[str, dict[int, float]]]
# profile id -> its table
ServingEnvelope = dict[str, EnvelopeTable]

# The failure code a worker reports for a job its hardware cannot fit. Not a miner fault when the job lies outside the
# enclave's advertised envelope; the gateway records it as internal_error when it lies inside.
CAPACITY_REFUSED = "capacity_refused"

_EPSILON = 1e-6


class EnvelopeError(ValueError):
    """A registered envelope does not describe the profiles it names."""


@dataclass(frozen=True)
class EnvelopeQuery:
    """What a router knows about a request. A field left as None matches any value."""

    resolution: str | None = None
    aspect_ratio: str | None = None
    fps: int | None = None
    duration_s: float | None = None

    @classmethod
    def of(cls, params: GenerationParams) -> EnvelopeQuery:
        return cls(params.resolution, params.aspect_ratio, params.fps, params.render_duration_s)

    @property
    def empty(self) -> bool:
        return self.resolution is None and self.aspect_ratio is None and self.fps is None and self.duration_s is None


def profile_max_duration(profile: ModelProfile, fps: int) -> float:
    """The profile's own longest duration at this frame rate."""
    lim = profile.limits
    return min(lim.max_duration_s, lim.max_duration_s_by_fps.get(fps, lim.max_duration_s))


def full_table(profile: ModelProfile) -> EnvelopeTable:
    """Everything the profile's limits allow: what a worker whose hardware holds the whole profile serves."""
    lim = profile.limits
    return {
        resolution: {aspect: {fps: profile_max_duration(profile, fps) for fps in lim.fps} for aspect in ratios}
        for resolution, ratios in lim.sizes.items()
    }


def max_duration(table: Mapping[str, Any] | None, resolution: str, aspect_ratio: str, fps: int) -> float | None:
    """The longest duration served at this size and frame rate, or None when it is not served. `table` None (no envelope)
    has no answer here; callers treat it as the profile's limits."""
    if table is None:
        return None
    by_fps = (table.get(resolution) or {}).get(aspect_ratio) or {}
    longest = by_fps.get(fps, by_fps.get(str(fps)))
    return None if longest is None else float(longest)


def fits(table: Mapping[str, Any] | None, params: GenerationParams) -> bool:
    """Whether a job fits a profile's table. No table (no envelope) fits every job the profile accepts."""
    if table is None:
        return True
    longest = max_duration(table, params.resolution, params.aspect_ratio, params.fps)
    # A storyboard renders one shot at a time, so its longest shot is what has to fit.
    return longest is not None and params.render_duration_s <= longest + _EPSILON


def serves(table: Mapping[str, Any] | None, query: EnvelopeQuery) -> bool:
    """Whether some request matching the query fits the table; with every field set, the same as `fits`."""
    if table is None or query.empty:
        return True
    for resolution, ratios in table.items():
        if query.resolution is not None and resolution != query.resolution:
            continue
        for aspect, by_fps in (ratios or {}).items():
            if query.aspect_ratio is not None and aspect != query.aspect_ratio:
                continue
            for fps, longest in (by_fps or {}).items():
                if query.fps is not None and int(fps) != query.fps:
                    continue
                if query.duration_s is None or query.duration_s <= float(longest) + _EPSILON:
                    return True
    return False


def restricts(table: Mapping[str, Any] | None, profile: ModelProfile) -> bool:
    """Whether the table serves less than the profile's limits."""
    if table is None:
        return False
    for resolution, ratios in full_table(profile).items():
        for aspect, by_fps in ratios.items():
            for fps, longest in by_fps.items():
                served = max_duration(table, resolution, aspect, fps)
                if served is None or served + _EPSILON < longest:
                    return True
    return False


def to_json(table: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    """A table with string fps keys, as JSON carries it."""
    return {
        resolution: {aspect: {str(fps): float(longest) for fps, longest in by_fps.items()} for aspect, by_fps in ratios.items()}
        for resolution, ratios in table.items()
    }


def advertised(tables: Mapping[str, Mapping[str, Any] | None], profiles: Mapping[str, ModelProfile]) -> dict | None:
    """What a worker registers: the tables of the profiles its hardware restricts, or None when it restricts none (so an
    unrestricted worker's registration is byte-for-byte what it was before envelopes)."""
    restricted = {
        profile_id: to_json(table)
        for profile_id, table in tables.items()
        if table is not None and restricts(table, profiles[profile_id])
    }
    return restricted or None


def normalize(envelope: Mapping[str, Any] | None, profiles: Mapping[str, ModelProfile], claimed: Iterable[str]) -> dict | None:
    """Checks a registered envelope against the profile catalog and returns it as stored: string fps keys, durations capped
    at the profile's own limits (a worker with an older catalog may list more). Raises EnvelopeError for a profile the
    enclave does not attest, a size or frame rate the profile does not have, or a duration that is not a number at least
    the profile's minimum (leave out what the hardware cannot serve). None or {} is no envelope."""
    if not envelope:
        return None
    claimed = set(claimed)
    stored: dict[str, dict] = {}
    for profile_id, table in envelope.items():
        profile = profiles.get(profile_id)
        if profile_id not in claimed or profile is None:
            raise EnvelopeError(f"the envelope names {profile_id}, which this enclave does not attest")
        lim = profile.limits
        rows: dict[str, dict[str, dict[str, float]]] = {}
        for resolution, ratios in (table or {}).items():
            if resolution not in lim.sizes:
                raise EnvelopeError(f"{profile_id} has no {resolution} size")
            for aspect, by_fps in (ratios or {}).items():
                if aspect not in lim.sizes[resolution]:
                    raise EnvelopeError(f"{profile_id} has no {resolution} {aspect} size")
                for fps, longest in (by_fps or {}).items():
                    try:
                        fps, longest = int(fps), float(longest)
                    except (TypeError, ValueError):
                        raise EnvelopeError(f"{profile_id} {resolution} {aspect}: fps and durations must be numbers") from None
                    if fps not in lim.fps:
                        raise EnvelopeError(f"{profile_id} has no {fps} fps")
                    if not math.isfinite(longest) or longest + _EPSILON < lim.min_duration_s:
                        raise EnvelopeError(
                            f"{profile_id} {resolution} {aspect} at {fps} fps: {longest!r} s is below the profile's "
                            f"{lim.min_duration_s:g} s minimum; leave out what the hardware cannot serve"
                        )
                    rows.setdefault(resolution, {}).setdefault(aspect, {})[str(fps)] = min(longest, profile_max_duration(profile, fps))
        stored[profile_id] = rows
    return stored


def describe(table: Mapping[str, Any] | None, params: GenerationParams) -> str:
    """How much of a request's size and frame rate a table serves, for error messages."""
    size = f"{params.resolution} {params.aspect_ratio} at {params.fps} fps"
    longest = max_duration(table, params.resolution, params.aspect_ratio, params.fps)
    return f"serves {size} up to {longest:g} s" if longest is not None else f"does not serve {size}"
