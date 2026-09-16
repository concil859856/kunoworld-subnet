"""Location proofs: signed landmark round trips bound miners by the speed of light. A fast enough signed answer rules
out the excluded territory; anything slow, forged, replayed or for another enclave rules out nothing."""

from __future__ import annotations

import os
import threading
import time

import httpx
import pytest

from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import generate_signing_key, public_key_bytes
from kuno_protocol.location import (
    SPEED_OF_LIGHT_KM_PER_MS,
    Landmark,
    LandmarkList,
    LocationProof,
    LocationSample,
    SignedLandmarks,
    answer_ping,
    make_server,
    ping_message,
    ping_nonce,
    radius_km,
    sign_landmarks,
    verify_location,
)

TOKYO_KEY, SINGAPORE_KEY = generate_signing_key(), generate_signing_key()
NONCE = os.urandom(32).hex()
ENCLAVE = "e" * 32


def landmark(id_: str, key, clearance: float, url: str = "https://landmark.test") -> Landmark:
    return Landmark(id=id_, url=url, public_key=b64e(public_key_bytes(key)), latitude=35.68, longitude=139.69,
                    clearance_km={"minimax-h3": clearance})


LANDMARKS = LandmarkList(issued_at=1, landmarks=[landmark("tokyo-1", TOKYO_KEY, 950.0), landmark("singapore-1", SINGAPORE_KEY, 4000.0)])


def sample(id_: str, key, rtt_ms: float, index: int = 0, nonce: str = NONCE, enclave: str = ENCLAVE) -> LocationSample:
    signature = key.sign(ping_message(id_, ping_nonce(nonce, enclave, id_, index)))
    return LocationSample(landmark_id=id_, index=index, rtt_ms=rtt_ms, signature=b64e(signature))


def check(*samples: LocationSample, nonce: str = NONCE, enclave: str = ENCLAVE):
    return verify_location(LocationProof(samples=list(samples)), LANDMARKS, registration_nonce=nonce, enclave_id=enclave,
                           region_policy="minimax-h3")


def test_a_fast_signed_round_trip_rules_out_the_excluded_territory():
    verdict = check(sample("tokyo-1", TOKYO_KEY, 2.0))
    assert verdict.ok and verdict.landmark_id == "tokyo-1"
    assert verdict.radius_km == pytest.approx(2.0 / 2 * SPEED_OF_LIGHT_KM_PER_MS, abs=0.1)
    assert "within 299.8 km of landmark tokyo-1" in verdict.detail


def test_a_slow_round_trip_proves_nothing():
    # 7 ms from Tokyo reaches 1049 km, past the 950 km to the nearest excluded territory.
    verdict = check(sample("tokyo-1", TOKYO_KEY, 7.0))
    assert not verdict.ok and "too slow to rule it out" in verdict.detail
    assert radius_km(6.3) < 950 < radius_km(6.4)


def test_the_best_sample_decides():
    assert check(sample("tokyo-1", TOKYO_KEY, 40.0), sample("singapore-1", SINGAPORE_KEY, 20.0)).landmark_id == "singapore-1"


@pytest.mark.parametrize(
    "bad, reason",
    [
        (lambda: sample("tokyo-1", SINGAPORE_KEY, 1.0), "not signed by the landmark"),                 # a key other than the landmark's
        (lambda: sample("tokyo-1", TOKYO_KEY, 1.0, nonce=os.urandom(32).hex()), "not signed by the landmark"),  # another registration
        (lambda: sample("tokyo-1", TOKYO_KEY, 1.0, enclave="f" * 32), "not signed by the landmark"),    # another enclave's proof
        (lambda: sample("osaka-1", TOKYO_KEY, 1.0), "not a landmark in the owner's list"),
    ],
)
def test_forged_replayed_or_foreign_samples_prove_nothing(bad, reason):
    verdict = check(bad())
    assert not verdict.ok and reason in verdict.detail


def test_no_proof_or_no_clearance_for_the_policy_is_refused():
    assert not verify_location(None, LANDMARKS, registration_nonce=NONCE, enclave_id=ENCLAVE, region_policy="minimax-h3").ok
    bare = LandmarkList(issued_at=1, landmarks=[Landmark(id="tokyo-1", url="https://x", public_key=b64e(public_key_bytes(TOKYO_KEY)),
                                                         latitude=0, longitude=0)])
    verdict = verify_location(LocationProof(samples=[sample("tokyo-1", TOKYO_KEY, 1.0)]), bare, registration_nonce=NONCE,
                              enclave_id=ENCLAVE, region_policy="minimax-h3")
    assert not verdict.ok and "no clearance for minimax-h3" in verdict.detail


def test_the_owner_signs_the_landmark_list():
    owner = generate_signing_key()
    signed = sign_landmarks(owner, LANDMARKS)
    assert SignedLandmarks.model_validate_json(signed.model_dump_json()).verify(public_key_bytes(owner))
    moved = signed.model_copy(deep=True)
    moved.landmarks.landmarks[0].clearance_km["minimax-h3"] = 20000.0  # widening a landmark's reach breaks the signature
    assert not moved.verify(public_key_bytes(owner))
    with pytest.raises(ValueError, match="unknown region policies"):
        Landmark(id="x", url="https://x", public_key="AA", latitude=0, longitude=0, clearance_km={"nowhere": 1.0})
    with pytest.raises(ValueError, match="landmark id"):
        Landmark(id="Tokyo 1", url="https://x", public_key="AA", latitude=0, longitude=0)


def test_a_landmark_server_signs_ping_nonces_quickly():
    server = make_server("tokyo-1", TOKYO_KEY, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        nonce = ping_nonce(NONCE, ENCLAVE, "tokyo-1", 3)
        with httpx.Client() as client:
            answer = client.get(f"{url}/v1/ping", params={"nonce": nonce.hex()}).json()
            # Loopback round trips on a kept-alive connection must not stall on delayed ACKs (about 40 ms each).
            timings = []
            for _ in range(5):
                started = time.perf_counter()
                client.get(f"{url}/v1/ping", params={"nonce": nonce.hex()})
                timings.append((time.perf_counter() - started) * 1000)
            assert min(timings) < 20, timings
            assert client.get(f"{url}/v1/ping", params={"nonce": "zz"}).status_code == 422
            assert client.get(f"{url}/elsewhere").status_code == 404
        assert answer["landmark_id"] == "tokyo-1"
        signed = LocationSample(landmark_id="tokyo-1", index=3, rtt_ms=0.5, signature=answer["signature"])
        assert check(signed).ok
    finally:
        server.shutdown()
    assert answer_ping("tokyo-1", TOKYO_KEY, "00") == (422, {"code": "bad_nonce", "message": "GET /v1/ping?nonce=<64 hex>"})
