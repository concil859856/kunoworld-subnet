"""Serving envelopes (kuno_protocol.envelope): the table format, fit checks, what a worker advertises, what a gateway
stores, and the registration field that older gateways ignore."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from kuno_protocol.attestation import AttestationEvidence
from kuno_protocol.envelope import (
    CAPACITY_REFUSED,
    EnvelopeError,
    EnvelopeQuery,
    advertised,
    describe,
    fits,
    full_table,
    max_duration,
    normalize,
    restricts,
    serves,
    to_json,
)
from kuno_protocol.hotkey import HotkeyProof
from kuno_protocol.profiles import Mode, load_profiles
from kuno_protocol.schemas import GenerationParams, MinerRegistration

PROFILES = load_profiles()
FAST = PROFILES["ltx-2.5-fast"]
PRO = PROFILES["ltx-2.5-pro"]


def params(resolution="1080p", aspect_ratio="16:9", fps=24, duration_s=5.0, profile_id=FAST.id) -> GenerationParams:
    return GenerationParams(
        profile_id=profile_id, mode=Mode.TEXT_TO_VIDEO, duration_s=duration_s, resolution=resolution, aspect_ratio=aspect_ratio, fps=fps
    )


def small_card() -> dict:
    """A card that serves 1080p 16:9 up to 8 s at 24 fps and 4 s at 50 fps, nothing at 1080p 21:9, and 720p in full."""
    table = full_table(FAST)
    table["1080p"]["16:9"] = {24: 8.0, 25: 8.0, 48: 4.0, 50: 4.0}
    del table["1080p"]["21:9"]
    return table


def test_the_full_table_is_the_profile_limits_including_the_per_fps_caps():
    table = full_table(FAST)
    assert set(table) == set(FAST.limits.sizes) and set(table["1080p"]) == set(FAST.limits.sizes["1080p"])
    assert table["1080p"]["21:9"] == {24: 20.0, 25: 20.0, 48: 10.0, 50: 10.0}
    assert not restricts(table, FAST) and not restricts(None, FAST)
    assert restricts(small_card(), FAST)


def test_fits_is_a_lookup_on_the_public_params():
    table = small_card()
    assert fits(None, params(duration_s=20))  # no envelope: the profile's limits
    assert fits(table, params(duration_s=8)) and not fits(table, params(duration_s=9))
    assert fits(table, params(fps=50, duration_s=4)) and not fits(table, params(fps=50, duration_s=5))
    assert not fits(table, params(aspect_ratio="21:9", duration_s=2))  # left out: not served at all
    assert fits(table, params(resolution="720p", duration_s=20))
    stored = json.loads(json.dumps(to_json(table)))  # string fps keys, as the gateway stores it
    assert fits(stored, params(duration_s=8)) and not fits(stored, params(duration_s=9))
    assert max_duration(stored, "1080p", "16:9", 50) == 4.0 and max_duration(stored, "1080p", "21:9", 24) is None
    assert describe(stored, params(duration_s=9)) == "serves 1080p 16:9 at 24 fps up to 8 s"
    assert describe(stored, params(aspect_ratio="21:9")) == "does not serve 1080p 21:9 at 24 fps"


def test_partial_queries_match_any_value_of_an_omitted_field():
    table = to_json(small_card())
    assert serves(table, EnvelopeQuery()) and serves(None, EnvelopeQuery(duration_s=999))
    assert serves(table, EnvelopeQuery(resolution="1080p", fps=24, duration_s=20))  # 1080p 1:1 still serves 20 s
    assert not serves(table, EnvelopeQuery(resolution="1080p", aspect_ratio="16:9", duration_s=9))
    assert not serves(table, EnvelopeQuery(resolution="1080p", aspect_ratio="21:9"))
    assert serves(table, EnvelopeQuery(resolution="1080p", aspect_ratio="16:9", fps=50, duration_s=4))
    assert serves(table, EnvelopeQuery.of(params(duration_s=8))) == fits(table, params(duration_s=8))


def test_a_worker_advertises_only_the_profiles_it_cannot_serve_in_full():
    assert advertised({FAST.id: full_table(FAST), PRO.id: full_table(PRO)}, PROFILES) is None
    sent = advertised({FAST.id: small_card(), PRO.id: full_table(PRO)}, PROFILES)
    assert list(sent) == [FAST.id]
    assert sent[FAST.id]["1080p"]["16:9"] == {"24": 8.0, "25": 8.0, "48": 4.0, "50": 4.0}
    json.dumps(sent)  # plain JSON


def test_the_gateway_normalizes_or_refuses_an_envelope():
    wire = json.loads(json.dumps(advertised({FAST.id: small_card()}, PROFILES)))
    stored = normalize(wire, PROFILES, [FAST.id, PRO.id])
    assert stored[FAST.id]["1080p"]["16:9"]["50"] == 4.0 and "21:9" not in stored[FAST.id]["1080p"]
    assert normalize(None, PROFILES, [FAST.id]) is None and normalize({}, PROFILES, [FAST.id]) is None
    # A worker with an older catalog may list longer clips than this gateway's profile allows: capped, not refused.
    capped = normalize({FAST.id: {"720p": {"16:9": {"50": 30}}}}, PROFILES, [FAST.id])
    assert capped[FAST.id] == {"720p": {"16:9": {"50": 10.0}}}
    refusals = {
        "does not attest": ({PRO.id: {}}, [FAST.id]),
        "no 4k size": ({FAST.id: {"4k": {}}}, [FAST.id]),
        "no 1080p 2:1 size": ({FAST.id: {"1080p": {"2:1": {}}}}, [FAST.id]),
        "no 30 fps": ({FAST.id: {"1080p": {"16:9": {"30": 5}}}}, [FAST.id]),
        "below the profile's 2 s minimum": ({FAST.id: {"1080p": {"16:9": {"24": 1}}}}, [FAST.id]),
        "must be numbers": ({FAST.id: {"1080p": {"16:9": {"24": "long"}}}}, [FAST.id]),
    }
    for message, (envelope, claimed) in refusals.items():
        with pytest.raises(EnvelopeError, match=message):
            normalize(envelope, PROFILES, claimed)
    with pytest.raises(EnvelopeError, match="below"):
        normalize({FAST.id: {"1080p": {"16:9": {"24": float("nan")}}}}, PROFILES, [FAST.id])


class RegistrationBeforeEnvelopes(BaseModel):
    """MinerRegistration as gateways from before envelopes define it."""

    evidence: AttestationEvidence
    miner_hotkey: str | None = None
    capacity: int = Field(default=1, ge=1, le=64)
    hotkey_proof: HotkeyProof | None = None


def test_registration_carries_the_envelope_and_older_gateways_ignore_it():
    from kuno_protocol import devkit
    from kuno_protocol.attestation import OpenTEE, build_evidence

    evidence = build_evidence(OpenTEE(), b"\x01" * 32, b"\x02" * 32, b"\x03" * 32, devkit.DEV_IMAGE_DIGEST, [FAST.id], {})
    base = {"evidence": evidence.model_dump(mode="json"), "miner_hotkey": "5Miner", "capacity": 1}
    old_worker = MinerRegistration.model_validate(base)
    assert old_worker.envelope is None
    body = json.loads(json.dumps({**base, "envelope": advertised({FAST.id: small_card()}, PROFILES)}))
    new_worker = MinerRegistration.model_validate(body)
    assert new_worker.envelope[FAST.id]["1080p"]["16:9"] == {24: 8.0, 25: 8.0, 48: 4.0, 50: 4.0}
    assert fits(new_worker.envelope[FAST.id], params(duration_s=8)) and not fits(new_worker.envelope[FAST.id], params(duration_s=9))
    older_gateway = RegistrationBeforeEnvelopes.model_validate(body)
    assert older_gateway.miner_hotkey == "5Miner" and "envelope" not in older_gateway.model_dump()
    assert CAPACITY_REFUSED == "capacity_refused"
