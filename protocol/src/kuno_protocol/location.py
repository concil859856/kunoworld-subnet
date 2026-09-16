"""Where a miner's GPUs are, bounded by the speed of light instead of taken from an IP address.

MiniMax H3's licence bars running it in its Excluded Territories (regions.py). A worker declares its country and the
gateway checks the country of the address it connects from, but both are easy to fake: a VPN exit in Tokyo says
nothing about GPUs in Virginia. Distance is harder to fake. No signal travels faster than light, so a round trip of t
milliseconds between two machines puts them at most t/2 × 299.79 km apart, however the traffic is routed. Delay can be
added; it can't be removed.

**Landmarks.** KunoWorld runs small landmark servers at known places. Each holds an Ed25519 key and answers a ping by
signing its nonce. The owner signs the list of landmarks (`SignedLandmarks`). For each landmark and region policy the
list gives its *clearance*: the great-circle distance to the nearest point of that policy's excluded territory, islands
and overseas territories included. The owner measures it; this module only compares against it.

**Proof.** At registration the worker pings landmarks from inside its confidential VM and times each round trip. The
code doing that is attested, so its timings are genuine measurements from the machine with the GPUs. Each ping's nonce
derives from the gateway's single-use registration nonce and the enclave id, so a proof can't be computed ahead of
time, reused by another enclave, or replayed later. The worker keeps each landmark's fastest signed answer and sends
them as a `LocationProof`.

**Verdict.** A sample proves the machine is outside a policy's excluded territory when its radius, rtt/2 × c, is
smaller than the landmark's clearance. One such sample is enough. Slow networks only make proofs fail, never pass:
a miner who can't prove its location is refused the region-restricted profiles, like a miner in an unknown country.

Assumptions, stated plainly:
- A landmark's private key never leaves it. With the key, anyone could answer pings from anywhere.
- The confidential VM's clock runs at its true rate (Intel TDX's TSC is protected from the host).
- The GPUs are the ones in this VM's attested evidence. Confidential-computing GPUs are passed through locally, so
  the timings are made from the GPUs' own machine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .canonical import b64d, b64e, canonical_json
from .crypto import verify_signature
from .regions import REGION_POLICIES

log = logging.getLogger("kuno.location")

SPEED_OF_LIGHT_KM_PER_MS = 299.792458
MAX_SAMPLES_PER_LANDMARK = 16
MAX_LANDMARKS = 32
_LANDMARK_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")


class Landmark(BaseModel):
    id: str
    url: str
    public_key: str  # base64 Ed25519
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    # Region policy -> km from this landmark to the nearest point of that policy's excluded territory.
    clearance_km: dict[str, float] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        if not _LANDMARK_ID.match(value):
            raise ValueError("a landmark id is 1-32 lowercase letters, digits and dashes")
        return value

    @field_validator("clearance_km")
    @classmethod
    def _clearance(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = [policy for policy in value if policy not in REGION_POLICIES]
        if unknown:
            raise ValueError(f"unknown region policies: {', '.join(unknown)}")
        if any(km < 0 or not math.isfinite(km) for km in value.values()):
            raise ValueError("clearances are finite, non-negative distances")
        return value


class LandmarkList(BaseModel):
    v: Literal[1] = 1
    issued_at: int
    landmarks: list[Landmark] = Field(default_factory=list, max_length=MAX_LANDMARKS)

    def by_id(self) -> dict[str, Landmark]:
        return {landmark.id: landmark for landmark in self.landmarks}


class SignedLandmarks(BaseModel):
    landmarks: LandmarkList
    signature: str | None = None

    def verify(self, owner_public_key: bytes) -> bool:
        return self.signature is not None and verify_signature(owner_public_key, b64d(self.signature), landmarks_message(self.landmarks))


def landmarks_message(landmarks: LandmarkList) -> bytes:
    return b"kuno/v1/landmarks\n" + canonical_json(landmarks.model_dump(mode="json"))


def sign_landmarks(owner_key, landmarks: LandmarkList) -> SignedLandmarks:
    return SignedLandmarks(landmarks=landmarks, signature=b64e(owner_key.sign(landmarks_message(landmarks))))


def ping_nonce(registration_nonce: str, enclave_id: str, landmark_id: str, index: int) -> bytes:
    """The nonce of ping `index` to a landmark: bound to one registration and one enclave, unknown before registration."""
    return hashlib.sha256(f"kuno/v1/location-nonce\n{registration_nonce.lower()}\n{enclave_id}\n{landmark_id}\n{index}".encode()).digest()


def ping_message(landmark_id: str, nonce: bytes) -> bytes:
    return f"kuno/v1/landmark-ping\n{landmark_id}\n{nonce.hex()}".encode()


class LocationSample(BaseModel):
    landmark_id: str
    index: int = Field(ge=0, lt=MAX_SAMPLES_PER_LANDMARK)
    rtt_ms: float = Field(gt=0, le=60_000)
    signature: str


class LocationProof(BaseModel):
    v: Literal[1] = 1
    samples: list[LocationSample] = Field(default_factory=list, max_length=MAX_LANDMARKS)


@dataclass
class LocationVerdict:
    ok: bool
    detail: str
    landmark_id: str | None = None
    radius_km: float | None = None
    clearance_km: float | None = None


def radius_km(rtt_ms: float) -> float:
    """The farthest a round trip of `rtt_ms` can reach: half the time, at the speed of light."""
    return rtt_ms / 2 * SPEED_OF_LIGHT_KM_PER_MS


def verify_location(
    proof: LocationProof | None, landmarks: LandmarkList, *, registration_nonce: str, enclave_id: str, region_policy: str
) -> LocationVerdict:
    """Whether `proof` places the enclave outside `region_policy`'s excluded territory. The best sample decides."""
    if proof is None or not proof.samples:
        return LocationVerdict(False, "no location proof: the worker measured no landmark")
    known = landmarks.by_id()
    best: LocationVerdict | None = None
    problems: list[str] = []
    for sample in proof.samples:
        landmark = known.get(sample.landmark_id)
        if landmark is None:
            problems.append(f"{sample.landmark_id}: not a landmark in the owner's list")
            continue
        nonce = ping_nonce(registration_nonce, enclave_id, landmark.id, sample.index)
        try:
            signed = verify_signature(b64d(landmark.public_key), b64d(sample.signature), ping_message(landmark.id, nonce))
        except ValueError:
            signed = False
        if not signed:
            problems.append(f"{landmark.id}: the ping answer is not signed by the landmark for this registration")
            continue
        clearance = landmark.clearance_km.get(region_policy)
        if clearance is None:
            problems.append(f"{landmark.id}: no clearance for {region_policy}")
            continue
        reach = radius_km(sample.rtt_ms)
        candidate = LocationVerdict(reach < clearance, "", landmark.id, round(reach, 1), clearance)
        if best is None or (candidate.clearance_km - candidate.radius_km) > (best.clearance_km - best.radius_km):
            best = candidate
    if best is None:
        return LocationVerdict(False, "no sample verified: " + "; ".join(problems[:4]))
    where = f"{best.radius_km:g} km of landmark {best.landmark_id}, whose nearest excluded territory is {best.clearance_km:g} km away"
    best.detail = (f"within {where}" if best.ok else f"only placed within {where}: the round trip is too slow to rule it out")
    return best


# ---------------------------------------------------------------- the landmark server


def answer_ping(landmark_id: str, key, nonce_hex: str | None) -> tuple[int, dict]:
    try:
        nonce = bytes.fromhex(nonce_hex or "")
    except ValueError:
        nonce = b""
    if len(nonce) != 32:
        return 422, {"code": "bad_nonce", "message": "GET /v1/ping?nonce=<64 hex>"}
    return 200, {"landmark_id": landmark_id, "signature": b64e(key.sign(ping_message(landmark_id, nonce)))}


def make_server(landmark_id: str, key, host: str = "0.0.0.0", port: int = 8480) -> ThreadingHTTPServer:
    """A landmark: signs ping nonces at GET /v1/ping?nonce= as fast as it can, and does nothing else.

    Timing is the whole point, so each exchange is one packet each way: the ping is a GET (no body to send separately),
    and the answer goes out in a single write with Nagle's algorithm off. Split writes on a kept-alive connection wait
    on delayed ACKs, which adds about 40 ms, a false 6,000 km, to every round trip.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive, so a worker's timed pings don't pay for new connections
        disable_nagle_algorithm = True

        def do_GET(self):  # noqa: N802 — http.server's naming
            path, _, query = self.path.partition("?")
            if path == "/v1/ping":
                status, payload = answer_ping(landmark_id, key, parse_qs(query).get("nonce", [None])[0])
            elif path == "/healthz":
                status, payload = 200, {"landmark_id": landmark_id, "at": time.time()}
            else:
                status, payload = 404, {"code": "not_found"}
            data = json.dumps(payload).encode()
            head = (f"HTTP/1.1 {status} {self.responses.get(status, ('',))[0]}\r\n"
                    f"Content-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n").encode()
            self.wfile.write(head + data)

        def log_message(self, *args) -> None:  # pings are too frequent to log
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    from .crypto import generate_signing_key, public_key_bytes, signing_key_from_bytes

    parser = argparse.ArgumentParser(prog="kuno-landmark", description="Run a KunoWorld landmark, or make its key.")
    sub = parser.add_subparsers(dest="command", required=True)
    keygen = sub.add_parser("keygen", help="write a new Ed25519 key and print its public key for the landmark list")
    keygen.add_argument("--key-file", type=Path, required=True)
    serve = sub.add_parser("serve", help="answer pings at GET /v1/ping?nonce= (put TLS in front of it)")
    serve.add_argument("--id", required=True)
    serve.add_argument("--key-file", type=Path, required=True)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8480)
    args = parser.parse_args(argv)
    if args.command == "keygen":
        if args.key_file.exists():
            parser.error(f"{args.key_file} exists; refusing to overwrite a landmark key")
        key = generate_signing_key()
        from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

        args.key_file.write_bytes(b64e(key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())).encode())
        args.key_file.chmod(0o600)
        print(b64e(public_key_bytes(key)))
        return 0
    key = signing_key_from_bytes(b64d(args.key_file.read_text().strip()))
    server = make_server(args.id, key, args.host, args.port)
    log.info("landmark %s listening on %s:%d", args.id, args.host, args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
