"""VCU weights follow GPU cost: a weight per resolution, a duration slope past 5 s and an fps multiplier. Profiles written
before them, with one `vcu_per_output_second`, still load and weigh every job as they did."""

from __future__ import annotations

import json
from importlib import resources

import pytest
from pydantic import ValidationError

from kuno_protocol.profiles import ModelProfile, ParamError, VcuWeights, load_profiles
from kuno_protocol.schemas import GenerationParams

PROFILES = load_profiles()

# research/research_pricing.md §3 (placeholders): VCU per output second at 24/25 fps up to 5 s, lowest resolution first,
# and the duration slope.
EXPECTED = {
    "ltx-2.5-fast": ({"720p": 3, "1080p": 5}, 0.03),
    "ltx-2.5-pro": ({"720p": 9, "1080p": 20}, 0.03),
    "ltx-2.5-4k": ({"1440p": 22, "2160p": 60}, 0.03),
    "h3-turbo": ({"768p": 17}, 0.05),
    "h3": ({"768p": 60}, 0.06),
    "h3-reference": ({"768p": 90}, 0.065),
}


def params(profile: ModelProfile, resolution: str, seconds: float, fps: int) -> GenerationParams:
    return GenerationParams(profile_id=profile.id, mode="text_to_video", duration_s=seconds, resolution=resolution,
                            aspect_ratio=next(iter(profile.limits.sizes[resolution])), fps=fps)


def cases():
    for profile_id, (weights, slope) in EXPECTED.items():
        profile = PROFILES[profile_id]
        durations = sorted({profile.limits.min_duration_s, 5.0, 10.0, profile.limits.max_duration_s})
        for resolution, weight in weights.items():
            for fps in profile.limits.fps:
                for seconds in durations:
                    yield pytest.param(profile_id, resolution, fps, seconds, weight, slope, id=f"{profile_id}-{resolution}-{fps}fps-{seconds:g}s")


@pytest.mark.parametrize(("profile_id", "resolution", "fps", "seconds", "weight", "slope"), list(cases()))
def test_vcu_follows_resolution_duration_and_fps(profile_id, resolution, fps, seconds, weight, slope):
    profile = PROFILES[profile_id]
    fps_factor = 2.0 if fps in (48, 50) else 1.0
    expected = weight * fps_factor * (1 + slope * max(0.0, seconds - 5)) * seconds
    assert profile.vcu_for(params(profile, resolution, seconds, fps)) == pytest.approx(expected)


def test_worked_examples():
    fast, h3 = PROFILES["ltx-2.5-fast"], PROFILES["h3"]
    assert fast.vcu_for(params(fast, "720p", 5, 24)) == pytest.approx(15)  # 3 × 5
    assert fast.vcu_for(params(fast, "1080p", 10, 50)) == pytest.approx(115)  # 5 × 2 × (1 + 0.03 × 5) × 10
    assert fast.vcu_for(params(fast, "720p", 2, 25)) == pytest.approx(6)  # clips under 5 s get no discount
    assert h3.vcu_for(params(h3, "768p", 14, 24)) == pytest.approx(1293.6)  # 60 × (1 + 0.06 × 9) × 14
    # `seconds` are the billable seconds, the requested duration unless given.
    assert h3.vcu_for(params(h3, "768p", 14, 24), seconds=5) == pytest.approx(300)


def test_a_duration_only_caller_gets_the_lowest_resolution_at_the_default_fps():
    for profile_id, (weights, slope) in EXPECTED.items():
        profile = PROFILES[profile_id]
        lowest = next(iter(weights))
        assert profile.base_vcu_resolution == lowest
        assert profile.vcu(10) == pytest.approx(weights[lowest] * (1 + slope * 5) * 10), profile_id
    assert PROFILES["ltx-2.5-4k"].vcu(4) == pytest.approx(22 * 4)


def test_every_resolution_a_profile_sells_has_a_weight_marked_as_a_placeholder():
    for profile in PROFILES.values():
        assert set(profile.vcu_weights.per_output_second) == set(profile.limits.sizes), profile.id
        assert "PLACEHOLDER" in profile.vcu_weights.note
        with pytest.raises(ParamError):
            profile.vcu_at("8k", 24, 5)


def test_a_profile_with_one_vcu_per_output_second_still_loads_and_weighs_every_job_alike():
    raw = json.loads(resources.files("kuno_protocol").joinpath("profiles.json").read_text())
    current = next(p for p in raw["profiles"] if p["id"] == "ltx-2.5-fast")
    old = {key: value for key, value in current.items() if key != "vcu_weights"} | {"vcu_per_output_second": 5}
    profile = ModelProfile.model_validate(old)
    assert profile.vcu_weights.per_output_second == {"720p": 5.0, "1080p": 5.0}
    assert (profile.vcu_weights.duration_slope, profile.vcu_weights.fps_multiplier) == (0.0, {})
    for resolution, fps, seconds in (("720p", 24, 5), ("1080p", 50, 20), ("720p", 48, 2)):
        assert profile.vcu_for(params(profile, resolution, seconds, fps)) == pytest.approx(5 * seconds)
    assert profile.vcu(8) == pytest.approx(40)
    # Also when the limits arrive as a model rather than JSON.
    assert ModelProfile.model_validate({**old, "limits": profile.limits}).vcu_weights == profile.vcu_weights


@pytest.mark.parametrize(
    "change",
    [{"per_output_second": {}}, {"per_output_second": {"720p": 0}}, {"per_output_second": {"720p": float("nan")}},
     {"fps_multiplier": {48: -2}}, {"duration_slope": -0.1}],
)
def test_bad_vcu_weights_are_refused(change):
    with pytest.raises(ValidationError):
        VcuWeights(**{"per_output_second": {"720p": 3}, **change})
