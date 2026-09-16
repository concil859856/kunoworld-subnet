"""The rate card's VCU form: one USD rate per tier × each job's VCU. It is left out of the signed bytes while empty, so a
card signed before it existed still verifies, and the placeholder card prices by VCU at research_pricing.md §3's rates."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kuno_protocol.canonical import b64d
from kuno_protocol.crypto import generate_signing_key, public_key_bytes
from kuno_protocol.profiles import FAMILY_H3, FAMILY_LTX, load_profiles
from kuno_protocol.rate_card import (
    PLACEHOLDER_OPEN_TIER_SHARE,
    PLACEHOLDER_USD_PER_GPU_HOUR,
    PLACEHOLDER_USD_PER_VCU_SECOND,
    RateCard,
    SignedRateCard,
    placeholder_rate_card,
    sign_rate_card,
)
from kuno_protocol.schemas import GenerationParams

PROFILES = load_profiles()
# The Ed25519 key from seed bytes 0..31. This card was signed by rate_card.py as it was before VCU rates, GPU-hour rates set.
OWNER = b64d("A6EHv_POEL4dcN0Y50vAmWfk1jCbpQ1fHdyGZBJVMbg")
CARD_BEFORE_VCU_RATES = (
    '{"card":{"version":1,"issued_at":1790000000,"currency":"USD","unit":"verified_video_second","placeholder":false,'
    '"usd_per_second":{"ltx-2.5-fast":{"confidential":0.05,"open":0.025},"h3":{"confidential":0.6}},'
    '"gpu_hour_usd":{"ltx-2.5":2.0,"minimax-h3":2.0},"note":""},'
    '"signature":"0GzVWMSFT8xk5KS-Mj1Ji3EKpgXIHn6F4EghL9g8lMyM0O-tvNq3ymRDGzLRaL-C-UCylB-ZcOwl8zZrPOJtBw"}'
)


def test_a_card_signed_before_vcu_rates_still_verifies_prices_by_profile_and_is_written_as_before():
    signed = SignedRateCard.model_validate_json(CARD_BEFORE_VCU_RATES)
    assert signed.verify(OWNER)
    card = signed.card
    assert card.usd_per_vcu_second == {} and card.vcu_rate("confidential") is None
    # Validators that predate the field refuse unknown fields, so a card without VCU rates doesn't mention them.
    assert "usd_per_vcu_second" not in signed.model_dump_json() and "usd_per_vcu_second" not in card.signed_fields()
    assert SignedRateCard.model_validate_json(signed.model_dump_json()).verify(OWNER)
    # Priced per profile, whatever the job's VCU.
    assert card.job_usd("h3", "confidential", 10.0, vcu=780.0) == pytest.approx(6.0)
    assert card.job_usd("h3", "open", 10.0, vcu=780.0) is None


def test_vcu_rates_price_their_tiers_by_vcu_and_are_signed_once_set():
    owner = generate_signing_key()
    card = RateCard(issued_at=7, usd_per_second={"h3": {"confidential": 0.6, "open": 0.3}}, usd_per_vcu_second={"confidential": 0.002},
                    placeholder=False)
    signed = sign_rate_card(owner, card)
    assert signed.verify(public_key_bytes(owner)) and card.signed_fields()["usd_per_vcu_second"] == {"confidential": 0.002}
    assert SignedRateCard.model_validate_json(signed.model_dump_json()).verify(public_key_bytes(owner))
    raised = signed.model_copy(update={"card": card.model_copy(update={"usd_per_vcu_second": {"confidential": 0.02}})})
    assert not raised.verify(public_key_bytes(owner))
    # A tier with a VCU rate is priced by VCU, even where the profile table has a rate; a tier without one uses the table.
    assert card.job_usd("h3", "confidential", 10.0, vcu=780.0) == pytest.approx(1.56)
    assert card.job_usd("h3", "open", 10.0, vcu=780.0) == pytest.approx(3.0)
    assert card.job_usd("ltx-2.5-fast", "open", 10.0, vcu=30.0) is None


@pytest.mark.parametrize("rates", [{"premium": 0.002}, {"confidential": -0.001}, {"open": float("inf")}])
def test_bad_vcu_rates_are_refused(rates):
    with pytest.raises(ValidationError):
        RateCard(usd_per_second={}, usd_per_vcu_second=rates)


# Recommended confidential miner rates, USD per verified second: measured cost x 1.25 for the profiles measured on
# 2026-09-15 (research/pricing/measured_2026-09-15.md) and h3-turbo on 2026-09-16 (measured_2026-09-16_h3-turbo.md), the
# estimates of costs.md §8.2 for the rest. h3 and h3-turbo at 10 and 14 s scale their 5 s rate by the duration ratios
# measured on 2026-09-16 (h3 1.453 and 1.841, h3-turbo 1.33 and 1.65). ltx-2.5-fast 720p ($0.005) is left out: one VCU
# rate pays it +14%.
RESEARCH_RATES = {
    ("ltx-2.5-fast", "1080p", 5): 0.010, ("ltx-2.5-pro", "720p", 5): 0.058, ("ltx-2.5-pro", "1080p", 5): 0.147,
    ("ltx-2.5-4k", "1440p", 5): 0.042, ("ltx-2.5-4k", "2160p", 5): 0.12,
    ("h3-turbo", "768p", 5): 0.032, ("h3-turbo", "768p", 10): 0.0426, ("h3-turbo", "768p", 14): 0.0528,
    ("h3", "768p", 5): 0.173, ("h3", "768p", 10): 0.251, ("h3", "768p", 14): 0.3185,
    ("h3-reference", "768p", 5): 0.281, ("h3-reference", "768p", 10): 0.373, ("h3-reference", "768p", 14): 0.445,
}


def per_second(card: RateCard, profile_id: str, resolution: str, seconds: float, tier: str = "confidential", fps: int = 24) -> float:
    profile = PROFILES[profile_id]
    job = GenerationParams(profile_id=profile_id, mode="text_to_video", duration_s=seconds, resolution=resolution,
                           aspect_ratio=next(iter(profile.limits.sizes[resolution])), fps=fps)
    return card.job_usd(profile_id, tier, seconds, profile.vcu_for(job)) / seconds


def test_the_placeholder_card_prices_by_vcu_at_the_research_rates():
    card = placeholder_rate_card(issued_at=3)
    assert card.placeholder and card.usd_per_second == {}
    assert (PLACEHOLDER_USD_PER_VCU_SECOND, PLACEHOLDER_OPEN_TIER_SHARE) == (0.0019, 0.75)
    assert card.usd_per_vcu_second == {"confidential": 0.0019, "open": pytest.approx(0.0019 * 0.75)}
    assert card.gpu_hour_usd == PLACEHOLDER_USD_PER_GPU_HOUR == {FAMILY_LTX: 0.80, FAMILY_H3: 1.50}
    for (profile_id, resolution, seconds), research in RESEARCH_RATES.items():
        # One VCU rate cannot match every row exactly; with the measured weights it pays 5-16% above the recommendation
        # (h3-turbo the most, like ltx-2.5-fast 720p: both are light profiles priced off the same $0.0019).
        assert per_second(card, profile_id, resolution, seconds) == pytest.approx(research, rel=0.16), (profile_id, resolution, seconds)
    assert per_second(card, "h3", "768p", 5) == pytest.approx(0.19)
    assert per_second(card, "ltx-2.5-fast", "1080p", 5) == pytest.approx(0.0095)
    assert per_second(card, "ltx-2.5-fast", "1080p", 5, fps=50) == pytest.approx(0.019)
    assert per_second(card, "ltx-2.5-fast", "1080p", 5, tier="open") == pytest.approx(0.0095 * 0.75)
