"""Capacity pay's owner-signed fields: the switch's share, targets and minimum uptime, and the rate card's GPU-hour
rates. Switches and cards signed before these fields existed must still verify."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kuno_protocol import devkit
from kuno_protocol.canonical import b64d
from kuno_protocol.crypto import generate_signing_key, public_key_bytes
from kuno_protocol.profiles import load_profiles
from kuno_protocol.rate_card import PLACEHOLDER_USD_PER_GPU_HOUR, RateCard, SignedRateCard, placeholder_rate_card, sign_rate_card
from kuno_protocol.switch import (
    PLACEHOLDER_CAPACITY_SHARE,
    PLACEHOLDER_CAPACITY_TARGETS,
    SignedSwitch,
    SwitchConfig,
    placeholder_switch,
    sign_switch,
)

# The Ed25519 key from seed bytes 0..31. Both documents below were signed by the code as it was before capacity pay.
OWNER = b64d("A6EHv_POEL4dcN0Y50vAmWfk1jCbpQ1fHdyGZBJVMbg")
OLD_SWITCH = '{"config":{"version":1,"issued_at":1780000000,"mode":"both","default_family":"minimax-h3","disabled_profiles":[],"h3_authorized_everywhere":false,"emission_split":{"minimax-h3":0.6,"ltx-2.5":0.4}},"signature":"v58kG6f76LSdpH6GZU95pJ5vus5tp8xeTmj7rcDVWKBmf7kd0LWYe8RSrOxJfCinTGM1xCra4-aBhxHM0cHeBQ"}'
OLD_RATE_CARD = '{"card":{"version":1,"issued_at":1780000000,"currency":"USD","unit":"verified_video_second","placeholder":false,"usd_per_second":{"ltx-2.5-fast":{"confidential":0.01,"open":0.005}},"note":""},"signature":"xGATSHs0sC1J9ww9ApFhAlSo8h6p91yAeuJZ5-_EcZQCsrqVgiHx49HWXpsdnzgwa4XysujL1tbQV6pGri1EAg"}'
FAMILIES = {profile.family for profile in load_profiles().values()}


# ---------------------------------------------------------------- switch


def test_a_switch_signed_before_capacity_pay_still_verifies_and_pays_nothing_for_capacity():
    signed = SignedSwitch.model_validate_json(OLD_SWITCH)
    assert signed.verify(OWNER)
    config = signed.config
    assert (config.capacity_share, config.capacity_targets, config.capacity_min_uptime_s) == (0.0, {}, 3600)
    assert not {"capacity_share", "capacity_targets", "capacity_min_uptime_s"} & set(config.signed_fields())
    # Served back by a gateway that knows the new fields (and so writes them out), it still verifies.
    assert SignedSwitch.model_validate_json(signed.model_dump_json()).verify(OWNER)


def test_the_capacity_fields_are_signed_once_set():
    owner = generate_signing_key()
    config = SwitchConfig(issued_at=5, capacity_share=0.25, capacity_targets={"ltx-2.5": 4}, capacity_min_uptime_s=1800)
    signed = sign_switch(owner, config)
    assert signed.verify(public_key_bytes(owner))
    assert SignedSwitch.model_validate_json(signed.model_dump_json()).verify(public_key_bytes(owner))
    fields = config.signed_fields()
    assert (fields["capacity_share"], fields["capacity_targets"], fields["capacity_min_uptime_s"]) == (0.25, {"ltx-2.5": 4}, 1800)
    for change in ({"capacity_share": 0.5}, {"capacity_share": 0.0}, {"capacity_targets": {"ltx-2.5": 40}}, {"capacity_min_uptime_s": 3600}):
        tampered = signed.model_copy(update={"config": config.model_copy(update=change)})
        assert not tampered.verify(public_key_bytes(owner)), change
    # An explicit default signs exactly as an absent one.
    assert SwitchConfig(issued_at=5, capacity_min_uptime_s=3600, capacity_share=0).signed_fields() == SwitchConfig(issued_at=5).signed_fields()


@pytest.mark.parametrize(
    "change",
    [{"capacity_share": 1.5}, {"capacity_share": -0.1}, {"capacity_share": float("nan")}, {"capacity_targets": {"ltx-2.5": -1}},
     {"capacity_min_uptime_s": -1}],
)
def test_bad_capacity_settings_are_refused(change):
    with pytest.raises(ValidationError):
        SwitchConfig(**change)


def test_dev_networks_sign_placeholder_capacity_pay(tmp_path):
    env = devkit.init(tmp_path)
    signed = SignedSwitch.model_validate_json((tmp_path / "switch.json").read_text())
    assert signed.verify(b64d(env["KUNO_OWNER_PUBLIC_KEY"]))
    assert signed.config.capacity_share == PLACEHOLDER_CAPACITY_SHARE == 0.25
    assert signed.config.capacity_targets == PLACEHOLDER_CAPACITY_TARGETS and set(PLACEHOLDER_CAPACITY_TARGETS) == FAMILIES
    assert placeholder_switch(mode="ltx").mode == "ltx" and SwitchConfig().capacity_share == 0.0


# ---------------------------------------------------------------- rate card


def test_a_rate_card_signed_before_gpu_hour_rates_still_verifies_and_is_written_as_before():
    signed = SignedRateCard.model_validate_json(OLD_RATE_CARD)
    assert signed.verify(OWNER)
    assert signed.card.gpu_hour_usd == {} and signed.card.gpu_hour_rate("ltx-2.5") is None
    # Validators that predate the field refuse unknown fields, so a card without GPU-hour rates doesn't mention them.
    assert "gpu_hour_usd" not in signed.model_dump_json() and "gpu_hour_usd" not in signed.card.signed_fields()
    assert SignedRateCard.model_validate_json(signed.model_dump_json()).verify(OWNER)


def test_gpu_hour_rates_are_signed_validated_and_placeholders_by_default():
    owner = generate_signing_key()
    card = RateCard(issued_at=5, usd_per_second={}, gpu_hour_usd={"ltx-2.5": 2.0}, placeholder=False)
    signed = sign_rate_card(owner, card)
    assert signed.verify(public_key_bytes(owner))
    assert SignedRateCard.model_validate_json(signed.model_dump_json()).verify(public_key_bytes(owner))
    assert card.gpu_hour_rate("ltx-2.5") == 2.0 and card.gpu_hour_rate("minimax-h3") is None
    raised = signed.model_copy(update={"card": card.model_copy(update={"gpu_hour_usd": {"ltx-2.5": 20.0}})})
    assert not raised.verify(public_key_bytes(owner))
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValidationError):
            RateCard(usd_per_second={}, gpu_hour_usd={"ltx-2.5": bad})
    template = placeholder_rate_card(issued_at=1)
    assert template.placeholder and template.gpu_hour_usd == {family: PLACEHOLDER_USD_PER_GPU_HOUR for family in FAMILIES}
