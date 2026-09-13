"""The validator judges challenge answers with the configured attestation policy, not bare checks."""

from __future__ import annotations

import json

import httpx

from kuno_protocol.attestation import GoldenManifest, MockTEE, Verdict, build_evidence
from kuno_protocol.crypto import generate_signing_key
from kuno_protocol.devkit import DEV_IMAGE_DIGEST
from kuno_validator.validator import Validator
from kuno_worker.identity import EnclaveIdentity

from test_receipt_ledger import FakeEnclave


class RecordingPolicy:
    production = False

    def __init__(self, enclave_id: str):
        self.enclave_id = enclave_id
        self.nonces: list[bytes | None] = []

    def verify(self, evidence, manifest, expected_nonce=None, now=None):
        self.nonces.append(expected_nonce)
        return Verdict(False, self.enclave_id, ["refused by the configured policy"])


def test_challenge_answers_are_verified_by_the_policy_with_the_validators_own_nonce():
    enclave = FakeEnclave("A").public()
    identity = EnclaveIdentity.generate()
    evidence = build_evidence(
        MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), b"\x00" * 32, identity.hpke_public, identity.signing_public,
        DEV_IMAGE_DIGEST, ["ltx-2.5-fast"], {},
    )
    sent_nonces: list[bytes] = []

    def gateway(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/validator/v1/enclaves":
            return httpx.Response(200, json=[enclave])
        if path == "/validator/v1/challenges":
            sent_nonces.append(bytes.fromhex(json.loads(request.content)["nonce"]))
            return httpx.Response(201, json={"challenge_id": "c1"})
        if path == "/validator/v1/challenges/c1":
            return httpx.Response(200, json={"status": "answered", "evidence": evidence.model_dump(mode="json")})
        return httpx.Response(404)

    policy = RecordingPolicy(enclave["enclave_id"])
    validator = Validator("http://gateway.test", "kuno_val_key", GoldenManifest(), transport=httpx.MockTransport(gateway), policy=policy)

    verdicts = validator.check_enclaves(timeout_s=5)

    assert policy.nonces == sent_nonces and len(sent_nonces) == 1
    assert verdicts[enclave["enclave_id"]].reasons == ["refused by the configured policy"]
    assert not verdicts[enclave["enclave_id"]].ok
