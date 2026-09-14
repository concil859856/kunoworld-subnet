"""The owner-signed rate card for USD-denominated miner pay: signatures, validation, and placeholders."""

from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from kuno_protocol import rate_card as rc
from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import generate_signing_key, public_key_bytes, signing_key_bytes
from kuno_protocol.profiles import load_profiles
from kuno_protocol.rate_card import RateCard, SignedRateCard, placeholder_rate_card, rate_card_message, sign_rate_card


def test_a_signed_card_verifies_only_under_the_owner_key_and_unchanged():
    owner = generate_signing_key()
    card = RateCard(issued_at=100, usd_per_second={"ltx-2.5-fast": {"confidential": 0.02, "open": 0.01}}, placeholder=False)
    signed = sign_rate_card(owner, card)
    assert signed.verify(public_key_bytes(owner))
    assert not signed.verify(public_key_bytes(generate_signing_key()))
    raised = signed.model_copy(update={"card": card.model_copy(update={"usd_per_second": {"ltx-2.5-fast": {"confidential": 0.2}}})})
    assert not raised.verify(public_key_bytes(owner))
    assert not SignedRateCard(card=card).verify(public_key_bytes(owner))
    assert rate_card_message(card).startswith(b"kuno/v1/rate-card\n")
    # It round-trips through JSON and still verifies.
    assert SignedRateCard.model_validate_json(signed.model_dump_json()).verify(public_key_bytes(owner))


def test_rates_are_looked_up_per_profile_and_tier():
    card = RateCard(usd_per_second={"h3": {"confidential": 0.05}})
    assert card.rate("h3", "confidential") == 0.05
    assert card.rate("h3", "open") is None
    assert card.rate("ltx-2.5-fast", "confidential") is None


@pytest.mark.parametrize(
    "rates",
    [{"h3": {"confidential": -0.01}}, {"h3": {"confidential": float("nan")}}, {"h3": {"premium": 0.1}}],
)
def test_bad_rates_are_refused(rates):
    with pytest.raises(ValidationError):
        RateCard(usd_per_second=rates)


def test_a_card_is_a_placeholder_unless_the_owner_says_otherwise():
    assert RateCard(usd_per_second={}).placeholder is True
    card = placeholder_rate_card(issued_at=5)
    assert card.placeholder is True and "PLACEHOLDER" in card.note and card.issued_at == 5
    profiles = load_profiles()
    assert set(card.usd_per_second) == set(profiles)
    for profile_id, tiers in card.usd_per_second.items():
        assert set(tiers) == {"confidential", "open"}
        assert 0 < tiers["open"] <= tiers["confidential"], profile_id


def test_the_cli_writes_a_placeholder_template_and_signs_it_without_printing_the_key(tmp_path, monkeypatch, capsys):
    owner = generate_signing_key()
    key_file = tmp_path / "owner.key"
    key_file.write_text(b64e(signing_key_bytes(owner)))
    template, signed_path = tmp_path / "card.json", tmp_path / "card.signed.json"
    monkeypatch.setattr(sys, "argv", ["rate_card", "template", "--out", str(template)])
    rc.main()
    monkeypatch.setattr(sys, "argv", ["rate_card", "sign", "--key", str(key_file), "--card", str(template), "--out", str(signed_path)])
    rc.main()
    signed = SignedRateCard.model_validate_json(signed_path.read_text())
    assert signed.verify(public_key_bytes(owner)) and signed.card.placeholder
    out = capsys.readouterr().out
    assert "PLACEHOLDER" in out and key_file.read_text() not in out
