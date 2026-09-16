"""The committed cross-language endorsement vectors still describe what the Python implementation decides.

The JS SDK checks the same file (sdk/js/test/endorsements.test.mjs); make_endorsement_vectors.py regenerates it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kuno_protocol.attestation import AttestationEvidence, GoldenManifest, SignedManifest, verify_endorsed_evidence
from kuno_protocol.canonical import b64d
from kuno_protocol.endorsements import EndorsedQuoteVerifier
from kuno_protocol.tdx import TdxQuoteResult

HERE = Path(__file__).resolve().parent
VECTORS = json.loads((HERE / "endorsement_vectors.json").read_text())


@pytest.fixture(autouse=True)
def vector_quote_verifier(monkeypatch):
    def verify_quote(self, quote):
        if self.endorsements.tdx_collateral != VECTORS["collateral"]:
            return TdxQuoteResult(False, "no Intel collateral was relayed for this quote" if self.endorsements.tdx_collateral is None else "collateral does not verify")
        return TdxQuoteResult(True, "TCB status UpToDate", "UpToDate")

    monkeypatch.setattr(EndorsedQuoteVerifier, "verify_quote", verify_quote)


@pytest.mark.parametrize("case", VECTORS["cases"], ids=[c["name"] for c in VECTORS["cases"]])
def test_python_reaches_the_committed_verdict(case):
    verdict = verify_endorsed_evidence(
        AttestationEvidence.model_validate(VECTORS["evidence"]), GoldenManifest.model_validate(VECTORS["manifest"]), case["endorsements"],
        expected_nonce=bytes.fromhex(VECTORS["expected_nonce"]), now=case.get("now", VECTORS["now"]), trusted_spki=VECTORS["trusted_spki"],
    )
    assert (verdict.ok, verdict.reasons, verdict.gpu_count) == (case["ok"], case["reasons"], case["gpu_count"])


def test_signed_manifest_vectors_verify_as_committed():
    owner = b64d(VECTORS["signed_manifests"]["owner_public_key"])
    assert all(SignedManifest.model_validate(d).verify(owner) for d in VECTORS["signed_manifests"]["valid"])
    assert not any(SignedManifest.model_validate(d).verify(owner) for d in VECTORS["signed_manifests"]["invalid"])


def test_the_js_sdk_carries_the_same_vectors():
    js = HERE.parents[2] / "sdk" / "js" / "test" / "endorsement_vectors.json"
    if not js.exists():
        pytest.skip("the JS SDK is not checked out beside this repository")
    assert js.read_text() == (HERE / "endorsement_vectors.json").read_text()
