"""Prices per privacy mode: the Private and Standard rates, the fps and long-clip multipliers, the minimum charge,
Private-only profiles, the fps-dependent duration cap, and profiles written before any of these existed."""

from __future__ import annotations

import json
from importlib import resources

import pytest

from kuno_protocol.profiles import (
    ModelProfile,
    ParamError,
    PrivacyModeUnavailable,
    load_profiles,
    validate_params,
)
from kuno_protocol.schemas import GenerationParams

PROFILES = load_profiles()

# (Private, Standard) per second; None: the profile is Private-only.
RATES = {
    # LTX-2.5's Standard prices are fal's list prices (2026-09-16); Private is about 1.3x.
    ("ltx-2.5-fast", "720p"): (0.12, 0.09),
    ("ltx-2.5-fast", "1080p"): (0.17, 0.13),
    ("ltx-2.5-pro", "720p"): (0.16, 0.12),
    ("ltx-2.5-pro", "1080p"): (0.22, 0.17),
    ("ltx-2.5-4k", "1440p"): (0.25, 0.19),
    ("ltx-2.5-4k", "2160p"): (0.39, 0.30),
    # H3's Standard prices are fal's list prices (2026-09-16); Private covers the measured cost at every length.
    ("h3-turbo", "768p"): (0.065, 0.04),
    ("h3", "768p"): (0.30, 0.06),
    ("h3-reference", "768p"): (0.50, 0.06),
}


def params(profile_id: str, **change) -> GenerationParams:
    profile = PROFILES[profile_id]
    base = dict(
        profile_id=profile_id, mode=profile.modes[0], duration_s=5, resolution=next(iter(profile.limits.sizes)),
        aspect_ratio="16:9", fps=profile.limits.default_fps,
    )
    return GenerationParams(**{**base, **change})


def test_every_profile_has_a_private_rate_a_standard_rate_or_none_and_a_ten_cent_minimum():
    for (profile_id, resolution), (private, standard) in RATES.items():
        pricing = PROFILES[profile_id].pricing
        assert pricing.usd_per_second[resolution] == private
        assert (pricing.standard_usd_per_second or {}).get(resolution) == standard
        assert pricing.min_job_usd == 0.10
    assert {p for p, _ in RATES} == set(PROFILES)


def test_private_is_the_default_and_standard_is_priced_by_its_own_table():
    fast = PROFILES["ltx-2.5-fast"]
    job = params("ltx-2.5-fast", resolution="1080p")
    assert fast.price_usd(job) == fast.price_usd(job, "private") == 0.85  # 0.17 x 5
    assert fast.price_usd(job, "standard") == 0.65  # 0.13 x 5
    assert fast.privacy_modes == ["private", "standard"] and fast.offers("standard")


def test_high_frame_rates_multiply_the_whole_ltx_job():
    fast = PROFILES["ltx-2.5-fast"]
    assert fast.price_usd(params("ltx-2.5-fast", fps=24)) == fast.price_usd(params("ltx-2.5-fast", fps=25)) == 0.60  # 0.12 x 5
    assert fast.price_usd(params("ltx-2.5-fast", fps=48)) == fast.price_usd(params("ltx-2.5-fast", fps=50)) == 0.90  # x 1.5
    assert PROFILES["ltx-2.5-4k"].price_usd(params("ltx-2.5-4k", resolution="2160p", fps=50), "standard") == 2.25  # 0.30 x 5 x 1.5


def test_long_private_h3_clips_cost_more_for_the_whole_clip_and_standard_stays_flat():
    turbo = PROFILES["h3-turbo"]
    assert turbo.price_usd(params("h3-turbo", duration_s=8)) == 0.52
    assert turbo.price_usd(params("h3-turbo", duration_s=10)) == 0.91  # 0.065 x 10 x 1.4
    assert turbo.price_usd(params("h3-turbo", duration_s=10), "standard") == 0.40  # no long-clip rule in Standard
    assert PROFILES["h3"].price_usd(params("h3", duration_s=14)) == 7.14  # 0.30 x 14 x 1.7
    assert PROFILES["h3"].price_usd(params("h3", duration_s=14), "standard") == 0.84
    assert PROFILES["h3-reference"].price_usd(params("h3-reference", duration_s=6)) == 3.0
    # LTX has no long-clip rule.
    assert PROFILES["ltx-2.5-fast"].price_usd(params("ltx-2.5-fast", duration_s=20)) == 2.4  # 0.12 x 20


def test_no_job_costs_less_than_the_minimum_charge():
    shipped = PROFILES["ltx-2.5-fast"]
    short = params("ltx-2.5-fast", duration_s=2)
    # No shipped profile reaches the minimum today: the cheapest job, 2 s of LTX-2.5 Fast 720p in Standard, is 0.09 x 2.
    # The rule is tested on a copy priced low enough to fall under it.
    assert shipped.price_usd(short, "standard") == 0.18
    fast = shipped.model_copy(update={"pricing": shipped.pricing.model_copy(update={
        "usd_per_second": {"720p": 0.045, "1080p": 0.08}, "standard_usd_per_second": {"720p": 0.04, "1080p": 0.06}})})
    assert fast.price_usd(short, "standard") == 0.10  # 0.04 x 2 = 0.08
    assert fast.price_usd(short) == 0.10  # 0.045 x 2 = 0.09


def test_private_only_profiles_refuse_a_standard_price():
    # Every shipped profile has a Standard price now; a profile without one is Private-only.
    for profile_id in ("h3", "h3-reference"):
        shipped = PROFILES[profile_id]
        profile = shipped.model_copy(update={"pricing": shipped.pricing.model_copy(update={"standard_usd_per_second": None})})
        assert profile.privacy_modes == ["private"] and not profile.offers("standard")
        assert profile.price_usd(params(profile_id)) > 0
        with pytest.raises(PrivacyModeUnavailable, match="Private mode only"):
            profile.price_usd(params(profile_id), "standard")
    assert issubclass(PrivacyModeUnavailable, ParamError)
    with pytest.raises(ParamError, match="unknown privacy mode"):
        PROFILES["h3-turbo"].price_usd(params("h3-turbo"), "public")


def test_ltx_fast_goes_past_ten_seconds_only_at_24_or_25_fps():
    fast = PROFILES["ltx-2.5-fast"]
    for fps in (24, 25):
        validate_params(fast, params("ltx-2.5-fast", duration_s=20, fps=fps))
    for fps in (48, 50):
        validate_params(fast, params("ltx-2.5-fast", duration_s=10, fps=fps))
        with pytest.raises(ParamError, match=f"at {fps} fps, duration must be at most 10 seconds"):
            validate_params(fast, params("ltx-2.5-fast", duration_s=11, fps=fps))
    for profile_id in ("ltx-2.5-pro", "ltx-2.5-4k"):
        validate_params(PROFILES[profile_id], params(profile_id, duration_s=10))
        with pytest.raises(ParamError, match="between 2 and 10 seconds"):
            validate_params(PROFILES[profile_id], params(profile_id, duration_s=11))


def test_a_profile_written_before_privacy_prices_still_loads_as_private_only_list_pricing():
    raw = json.loads(resources.files("kuno_protocol").joinpath("profiles.json").read_text())
    old = next(p for p in raw["profiles"] if p["id"] == "ltx-2.5-fast")
    old["pricing"] = {"usd_per_second": {"720p": 0.024, "1080p": 0.04}}
    old["limits"].pop("max_duration_s_by_fps")
    profile = ModelProfile.model_validate(old)
    assert profile.privacy_modes == ["private"] and profile.pricing.min_job_usd == 0.0
    assert profile.price_usd(params("ltx-2.5-fast", fps=48, duration_s=2)) == 0.048
    validate_params(profile, params("ltx-2.5-fast", duration_s=20, fps=50))
