"""Measuring this worker's distance to KunoWorld's landmarks, for profiles whose licence is bound to territory.

kuno_protocol.location explains the proof. Here the worker, inside its confidential VM, pings each landmark from a
kept-alive connection (so connection setup isn't timed), times every signed answer, and keeps each landmark's fastest.
Timing overhead only makes a round trip look longer, which makes a proof weaker, never wrongly stronger.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx

from kuno_protocol.location import MAX_SAMPLES_PER_LANDMARK, LandmarkList, LocationProof, LocationSample, ping_message, ping_nonce
from kuno_protocol.canonical import b64d
from kuno_protocol.crypto import verify_signature

log = logging.getLogger("kuno.worker.location")

DEFAULT_SAMPLES = 5


def measure(
    landmarks: LandmarkList,
    registration_nonce: bytes,
    enclave_id: str,
    samples: int = DEFAULT_SAMPLES,
    *,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    timeout_s: float = 3.0,
) -> LocationProof:
    """Each reachable landmark's fastest round trip whose answer it signed for this registration."""
    samples = max(1, min(samples, MAX_SAMPLES_PER_LANDMARK))
    out: list[LocationSample] = []
    for landmark in landmarks.landmarks:
        url = landmark.url.rstrip("/") + "/v1/ping"
        public_key = b64d(landmark.public_key)
        best: LocationSample | None = None
        try:
            with httpx.Client(timeout=timeout_s, transport=transport) as client:
                client.get(url, params={"nonce": "00" * 32})  # opens the connection; not timed
                for index in range(samples):
                    nonce = ping_nonce(registration_nonce.hex(), enclave_id, landmark.id, index)
                    started = clock()
                    # A GET is one packet; a POST's separate body write can stall on delayed ACKs (kuno_protocol.location).
                    response = client.get(url, params={"nonce": nonce.hex()})
                    elapsed_ms = (clock() - started) / 1e6
                    if response.status_code != 200:
                        continue
                    signature = response.json().get("signature", "")
                    # A landmark that answers unsigned or wrongly signed pings proves nothing; skip it here, not at the gateway.
                    if not verify_signature(public_key, b64d(signature), ping_message(landmark.id, nonce)):
                        log.warning("landmark %s returned a signature that doesn't verify", landmark.id)
                        break
                    if best is None or elapsed_ms < best.rtt_ms:
                        best = LocationSample(landmark_id=landmark.id, index=index, rtt_ms=max(elapsed_ms, 0.001), signature=signature)
        except (httpx.HTTPError, ValueError) as exc:
            log.info("landmark %s unreachable: %s", landmark.id, exc)
            continue
        if best is not None:
            out.append(best)
            log.info("landmark %s: fastest signed round trip %.2f ms", landmark.id, best.rtt_ms)
    return LocationProof(samples=out)
