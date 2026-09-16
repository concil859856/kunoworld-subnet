"""The worker's landmark measurement: a kept-alive connection, a signed answer per ping, the fastest one kept, and
landmarks that don't answer or don't sign correctly left out rather than trusted."""

from __future__ import annotations

import os
import threading

from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import generate_signing_key, public_key_bytes
from kuno_protocol.location import Landmark, LandmarkList, make_server, verify_location
from kuno_worker.location import measure

ENCLAVE = "a" * 32


def serve(landmark_id: str, key):
    server = make_server(landmark_id, key, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_measured_round_trips_verify_and_the_fastest_is_kept():
    key, rogue = generate_signing_key(), generate_signing_key()
    good, good_url = serve("near-1", key)
    liar, liar_url = serve("liar-1", rogue)  # answers, but not with the key the owner's list names
    try:
        landmarks = LandmarkList(issued_at=1, landmarks=[
            Landmark(id="near-1", url=good_url, public_key=b64e(public_key_bytes(key)), latitude=0, longitude=0,
                     clearance_km={"minimax-h3": 2000.0}),
            Landmark(id="liar-1", url=liar_url, public_key=b64e(public_key_bytes(key)), latitude=0, longitude=0,
                     clearance_km={"minimax-h3": 2000.0}),
            Landmark(id="gone-1", url="http://127.0.0.1:9", public_key=b64e(public_key_bytes(key)), latitude=0, longitude=0,
                     clearance_km={"minimax-h3": 2000.0}),
        ])
        nonce = os.urandom(32)
        ticks = iter(range(0, 10**12, 1_000_000))  # every clock read 1 ms later: each round trip measures 1 ms
        proof = measure(landmarks, nonce, ENCLAVE, samples=3, clock=lambda: next(ticks), timeout_s=1.0)
        assert [s.landmark_id for s in proof.samples] == ["near-1"]
        assert proof.samples[0].rtt_ms == 1.0
        verdict = verify_location(proof, landmarks, registration_nonce=nonce.hex(), enclave_id=ENCLAVE, region_policy="minimax-h3")
        assert verdict.ok, verdict.detail
    finally:
        good.shutdown()
        liar.shutdown()


def test_real_loopback_timings_are_small_and_positive():
    key = generate_signing_key()
    server, url = serve("loop-1", key)
    try:
        landmarks = LandmarkList(issued_at=1, landmarks=[Landmark(id="loop-1", url=url, public_key=b64e(public_key_bytes(key)),
                                                                  latitude=0, longitude=0, clearance_km={"minimax-h3": 950.0})])
        proof = measure(landmarks, os.urandom(32), ENCLAVE)
        (sample,) = proof.samples
        assert 0 < sample.rtt_ms < 20  # no ~40 ms delayed-ACK stall
    finally:
        server.shutdown()
