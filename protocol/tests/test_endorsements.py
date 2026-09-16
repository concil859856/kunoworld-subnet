"""Endorsements: a client checks an enclave's hardware evidence with what Intel and NVIDIA signed, relayed by a gateway
it doesn't trust.

NVIDIA: NRAS-shaped tokens under a local RSA intermediate that mirrors NVIDIA's real chain (RSA-2048 intermediate,
P-384 signing certificate, sha256WithRSA between them), pinned by SPKI hash; plus NVIDIA's real signing certificate and
intermediate from the live JWKS, checked against the production pin. No real NRAS token with a known nonce exists
without a confidential GPU, so the token logic is proven on local tokens. Intel: Phala's real TDX quotes and collateral.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

from kuno_protocol.attestation import build_evidence, gpu_nonce_for, verify_endorsed_evidence, verify_evidence
from kuno_protocol.canonical import b64d
from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key, public_key_bytes
from kuno_protocol.endorsements import (
    NRAS_INTERMEDIATE_SPKI_SHA256,
    EndorsedGpuVerifier,
    EndorsedQuoteVerifier,
    Endorsements,
    pinned_nras_key,
    verify_endorsed_token,
)
from kuno_protocol.nvidia import GpuTokenError, NrasGpuVerifier, NvidiaResult
from kuno_protocol.tdx import TdxQuoteResult

from test_nvidia import GOOD_GPU_CLAIMS, FakeCollector, FakeNras, SimulatedTdx, bundle, tdx_manifest

DATA = Path(__file__).parent / "data"
UTC = datetime.timezone.utc


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def spki_sha256(cert: x509.Certificate) -> str:
    return hashlib.sha256(cert.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)).hexdigest()


class PinnedNras(FakeNras):
    """FakeNras whose signing key sits under an RSA intermediate, the way NVIDIA's does."""

    def __init__(self, leaf_from: datetime.datetime | None = None, leaf_days: float = 2, intermediate_signer=None):
        super().__init__(x5c=False)
        now = datetime.datetime.now(UTC)
        root_key = ec.generate_private_key(ec.SECP384R1())
        self.intermediate_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.intermediate = (
            x509.CertificateBuilder().subject_name(_name("Test Attestation Intermediate")).issuer_name(_name("Test Attestation CA"))
            .public_key(self.intermediate_key.public_key()).serial_number(2)
            .not_valid_before(now - datetime.timedelta(days=30)).not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(root_key, hashes.SHA384())
        )
        start = leaf_from or now - datetime.timedelta(hours=1)
        leaf = (
            x509.CertificateBuilder().subject_name(_name("Test Attestation Service GPU")).issuer_name(_name("Test Attestation Intermediate"))
            .public_key(self.key.public_key()).serial_number(3)
            .not_valid_before(start).not_valid_after(start + datetime.timedelta(days=leaf_days))
            .sign(intermediate_signer or self.intermediate_key, hashes.SHA256())
        )
        der = lambda cert: base64.b64encode(cert.public_bytes(serialization.Encoding.DER)).decode()  # noqa: E731
        self.jwks["keys"][0]["x5c"] = [der(leaf), der(self.intermediate)]
        self.pin = spki_sha256(self.intermediate)
        self.issued_at = datetime.datetime.now(UTC).timestamp()

    def token(self, claims: dict, key=None, kid=None) -> str:
        return super().token({"iat": self.issued_at, **claims}, key, kid)


def tdx_worker():
    """Simulated TDX evidence (a synthetic quote binding real keys) and the parts a test needs to check it."""
    collector = FakeCollector()
    _, hpke_pk = generate_hpke_keypair()
    sign_pk, nonce = public_key_bytes(generate_signing_key()), os.urandom(32)
    evidence = build_evidence(SimulatedTdx(collector), nonce, hpke_pk, sign_pk, "sha256:img", ["ltx-2.5-fast"])
    return evidence, nonce, gpu_nonce_for(nonce, hpke_pk, sign_pk)


class CollateralQuoteVerifier:
    """A gateway's quote verifier, minus Intel: accepts the synthetic quote and reports the collateral it 'used'."""

    collateral = {"tcb_info": "{}", "qe_identity": "{}", "pck_crl": "00"}

    def verify(self, quote):
        return True, "TCB status UpToDate"

    def verify_quote(self, quote):
        return TdxQuoteResult(True, "TCB status UpToDate", "UpToDate", collateral=dict(self.collateral))


@pytest.fixture
def relay_accepts_synthetic_quotes(monkeypatch):
    """Client-side, the synthetic quote can't pass dcap-qvl; stand in for it only when collateral was relayed."""

    def verify_quote(self, quote):
        if self.endorsements.tdx_collateral != CollateralQuoteVerifier.collateral:
            return TdxQuoteResult(False, "no Intel collateral was relayed for this quote")
        return TdxQuoteResult(True, "TCB status UpToDate", "UpToDate")

    monkeypatch.setattr(EndorsedQuoteVerifier, "verify_quote", verify_quote)


def gateway_verdict(nras: PinnedNras):
    evidence, nonce, _ = tdx_worker()
    verdict = verify_evidence(
        evidence, tdx_manifest(), expected_nonce=nonce, quote_verifier=CollateralQuoteVerifier(), gpu_verifier=NrasGpuVerifier(http=nras)
    )
    assert verdict.ok, verdict.reasons
    return evidence, nonce, verdict


# ---------------------------------------------------------------- the relay path, end to end


def test_a_gateway_verdict_carries_what_intel_and_nvidia_signed():
    nras = PinnedNras()
    _, _, verdict = gateway_verdict(nras)
    material = verdict.endorsements
    assert isinstance(material, Endorsements)
    assert material.tdx_collateral == CollateralQuoteVerifier.collateral
    (gpu,) = material.nvidia
    assert gpu.device == "gpu" and gpu.answer[0][0] == "JWT" and set(gpu.answer[1]) == {"GPU-0"}
    assert [k["kid"] for k in gpu.keys] == [nras.kid] and len(gpu.keys[0]["x5c"]) == 2


def test_a_client_verifies_the_relayed_material_itself(relay_accepts_synthetic_quotes):
    nras = PinnedNras()
    evidence, nonce, verdict = gateway_verdict(nras)
    relayed = json.loads(verdict.endorsements.model_dump_json())  # what the route response carries
    client = verify_endorsed_evidence(evidence, tdx_manifest(), relayed, expected_nonce=nonce, trusted_spki=[nras.pin])
    assert client.ok, client.reasons
    assert client.gpu_count == 1


def test_tdx_evidence_without_endorsements_is_refused_not_half_checked(relay_accepts_synthetic_quotes):
    nras = PinnedNras()
    evidence, nonce, verdict = gateway_verdict(nras)
    for relayed, expected in ((None, "no endorsements were relayed"), ({"v": 9}, "malformed")):
        reasons = verify_endorsed_evidence(evidence, tdx_manifest(), relayed, expected_nonce=nonce, trusted_spki=[nras.pin]).reasons
        assert any(expected in r for r in reasons), reasons
        assert not any("verifier configured" in r for r in reasons)  # the client has none; that's not the point to report
    no_gpu = verdict.endorsements.model_copy(update={"nvidia": []})
    reasons = verify_endorsed_evidence(evidence, tdx_manifest(), no_gpu, expected_nonce=nonce, trusted_spki=[nras.pin]).reasons
    assert any("no NVIDIA attestation result was relayed" in r for r in reasons)
    no_intel = verdict.endorsements.model_copy(update={"tdx_collateral": None})
    reasons = verify_endorsed_evidence(evidence, tdx_manifest(), no_intel, expected_nonce=nonce, trusted_spki=[nras.pin]).reasons
    assert any("no Intel collateral" in r for r in reasons)


def test_a_relay_cannot_sign_gpu_claims_with_a_key_of_its_own(relay_accepts_synthetic_quotes):
    nras = PinnedNras()
    evidence, nonce, verdict = gateway_verdict(nras)
    # The production pin is NVIDIA's intermediate; the local one isn't it.
    reasons = verify_endorsed_evidence(evidence, tdx_manifest(), verdict.endorsements, expected_nonce=nonce).reasons
    assert any("pinned attestation intermediate" in r for r in reasons)
    # A relay that re-signs NRAS's answer under its own chain fails the same way, even with the kid unchanged.
    impostor = PinnedNras()
    impostor.kid = nras.kid
    impostor.jwks["keys"][0]["kid"] = nras.kid
    forged = [["JWT", impostor.token({"x-nvidia-overall-att-result": True, "eat_nonce": gpu_nonce_for_evidence(evidence)})],
              {"GPU-0": impostor.token(GOOD_GPU_CLAIMS)}]
    swapped = Endorsements(tdx_collateral=verdict.endorsements.tdx_collateral,
                           nvidia=[NvidiaResult(device="gpu", answer=forged, keys=impostor.jwks["keys"])])
    reasons = verify_endorsed_evidence(evidence, tdx_manifest(), swapped, expected_nonce=nonce, trusted_spki=[nras.pin]).reasons
    assert any("pinned attestation intermediate" in r for r in reasons)


def gpu_nonce_for_evidence(evidence) -> str:
    return gpu_nonce_for(bytes.fromhex(evidence.nonce), b64d(evidence.hpke_public_key), b64d(evidence.signing_public_key)).hex()


# ---------------------------------------------------------------- NVIDIA: the token checks one by one


def answer(nras: PinnedNras, gpu_nonce: bytes, per_gpu: list[dict] | None = None, overall: dict | None = None):
    detached = {f"GPU-{i}": nras.token(claims) for i, claims in enumerate(per_gpu or [GOOD_GPU_CLAIMS])}
    return [["JWT", nras.token({"x-nvidia-overall-att-result": True, "eat_nonce": gpu_nonce.hex(), **(overall or {})})], detached]


def endorsed(nras: PinnedNras, *answers: tuple[str, list]) -> Endorsements:
    return Endorsements(nvidia=[NvidiaResult(device=device, answer=a, keys=nras.jwks["keys"]) for device, a in answers])


def test_good_relayed_gpu_claims_verify_and_count_devices():
    nras, nonce = PinnedNras(), os.urandom(32)
    material = endorsed(nras, ("gpu", answer(nras, nonce, [GOOD_GPU_CLAIMS] * 4)))
    result = EndorsedGpuVerifier(material, trusted_spki=[nras.pin]).verify_devices(bundle(nonce, gpus=4), nonce)
    assert result.ok, result.detail
    assert result.gpu_count == 4


@pytest.mark.parametrize(
    "per_gpu, overall, expected",
    [
        (None, {"x-nvidia-overall-att-result": False}, "overall attestation result"),
        (None, {"eat_nonce": "00" * 32}, "different nonce"),
        ([{**GOOD_GPU_CLAIMS, "measres": "fail"}], None, "reference values"),
        ([{**GOOD_GPU_CLAIMS, "dbgstat": "enabled"}], None, "debug"),
        ([{**GOOD_GPU_CLAIMS, "eat_nonce": "11" * 32}], None, "GPU-0: token is for a different nonce"),
        ([GOOD_GPU_CLAIMS, GOOD_GPU_CLAIMS], None, "attested 2 GPU(s) but the evidence holds 1"),
    ],
)
def test_relayed_claims_are_judged_like_nras_answers(per_gpu, overall, expected):
    nras, nonce = PinnedNras(), os.urandom(32)
    material = endorsed(nras, ("gpu", answer(nras, nonce, per_gpu, overall)))
    result = EndorsedGpuVerifier(material, trusted_spki=[nras.pin]).verify_devices(bundle(nonce), nonce)
    assert not result.ok and expected in result.detail


def test_nvswitch_evidence_needs_its_own_relayed_answer():
    from kuno_protocol.nvidia import GpuCcSettings, GpuEvidenceBundle, GpuEvidenceItem

    nras, nonce = PinnedNras(), os.urandom(32)
    item = GpuEvidenceItem(arch="HOPPER", evidence="ZXZpZGVuY2U=", certificate="LS0t")
    switch = GpuEvidenceItem(arch="LS10", evidence="c3dpdGNo", certificate="LS0t")
    evidence = GpuEvidenceBundle(nonce=nonce.hex(), gpus=[item] * 8, cc=GpuCcSettings(mode="ppcie", devtools=False), switches=[switch] * 4).encode()
    gpus_only = endorsed(nras, ("gpu", answer(nras, nonce, [GOOD_GPU_CLAIMS] * 8)))
    result = EndorsedGpuVerifier(gpus_only, trusted_spki=[nras.pin]).verify_devices(evidence, nonce)
    assert not result.ok and "for the NVSwitches" in result.detail


def test_token_signature_age_and_issue_time_are_enforced():
    nras, nonce = PinnedNras(), os.urandom(32)
    keys, now = nras.jwks["keys"], nras.issued_at
    token = nras.token({"eat_nonce": nonce.hex()})
    assert verify_endorsed_token(token, keys, now, trusted_spki=[nras.pin])["eat_nonce"] == nonce.hex()
    header, payload, signature = token.split(".")
    tampered = f"{header}.{payload}.{signature[:-4]}AAAA"
    with pytest.raises(GpuTokenError, match="signature is invalid"):
        verify_endorsed_token(tampered, keys, now, trusted_spki=[nras.pin])
    with pytest.raises(GpuTokenError, match="older than a client accepts"):
        verify_endorsed_token(token, keys, now + 7200, trusted_spki=[nras.pin])
    with pytest.raises(GpuTokenError, match="issued in the future"):
        verify_endorsed_token(token, keys, now - 3600, trusted_spki=[nras.pin])
    with pytest.raises(GpuTokenError, match="no relayed signing key"):
        verify_endorsed_token(nras.token({}, kid="another"), keys, now, trusted_spki=[nras.pin])
    undated = FakeNras.token(nras, {"eat_nonce": nonce.hex()})  # no iat or nbf
    with pytest.raises(GpuTokenError, match="no issue time"):
        verify_endorsed_token(undated, keys, now, trusted_spki=[nras.pin])


def test_the_signing_certificate_must_be_issued_by_the_pinned_intermediate_and_valid_when_the_token_was():
    nras = PinnedNras()
    jwk, at = nras.jwks["keys"][0], nras.issued_at
    assert pinned_nras_key(jwk, at, [nras.pin])

    rogue = PinnedNras(intermediate_signer=rsa.generate_private_key(public_exponent=65537, key_size=2048))
    rogue.jwks["keys"][0]["x5c"][1] = nras.jwks["keys"][0]["x5c"][1]  # claims the pinned intermediate, signed by another key
    with pytest.raises(GpuTokenError, match="not signed by the pinned intermediate"):
        pinned_nras_key(rogue.jwks["keys"][0], at, [nras.pin])

    expired = PinnedNras(leaf_from=datetime.datetime.now(UTC) - datetime.timedelta(days=5), leaf_days=2)
    with pytest.raises(GpuTokenError, match="signing certificate was not valid"):
        pinned_nras_key(expired.jwks["keys"][0], at, [expired.pin])

    with pytest.raises(GpuTokenError, match="x5c of two"):
        pinned_nras_key({**jwk, "x5c": jwk["x5c"][:1]}, at, [nras.pin])

    other = ec.generate_private_key(ec.SECP384R1()).public_key().public_numbers()
    swapped = {**jwk, "x": base64.urlsafe_b64encode(other.x.to_bytes(48, "big")).rstrip(b"=").decode()}
    with pytest.raises(GpuTokenError, match="not the key its certificate holds"):
        pinned_nras_key(swapped, at, [nras.pin])


def test_nvidias_real_signing_certificate_chains_to_the_production_pin():
    """NVIDIA's live JWKS (fetched 2026-09-16): the chain logic works on its real RSA intermediate, and the pin is right."""
    jwk = json.loads((DATA / "nras" / "jwks_entry_2026-09-16.json").read_text())["key"]
    leaf, intermediate = (x509.load_der_x509_certificate(base64.b64decode(c)) for c in jwk["x5c"])
    assert spki_sha256(intermediate) in NRAS_INTERMEDIATE_SPKI_SHA256
    assert intermediate.subject.rfc4514_string() == "C=US,O=NVIDIA Corporation,CN=NVIDIA Attestation Service GPU Intermediate 004"
    during = leaf.not_valid_before_utc.timestamp() + 3600
    assert pinned_nras_key(jwk, during).curve.name == "secp384r1"
    with pytest.raises(GpuTokenError, match="not valid when the token was issued"):
        pinned_nras_key(jwk, leaf.not_valid_after_utc.timestamp() + 3600)
    der = bytearray(base64.b64decode(jwk["x5c"][0]))
    der[-10] ^= 0x01  # inside the intermediate's RSA signature over the certificate
    with pytest.raises(GpuTokenError):
        pinned_nras_key({**jwk, "x5c": [base64.b64encode(bytes(der)).decode(), jwk["x5c"][1]]}, during)


# ---------------------------------------------------------------- Intel: real quotes against relayed collateral

dcap_qvl = pytest.importorskip("dcap_qvl")
DCAP = DATA / "dcap"
VALID_AT = 1751000000  # inside the sample collateral's validity window (test_tdx.py)


def collateral_json(name: str) -> dict:
    return json.loads((DCAP / f"{name}_collateral.json").read_text())


def test_relayed_intel_collateral_verifies_a_real_quote_offline():
    quote = (DCAP / "tdx_quote").read_bytes()
    result = EndorsedQuoteVerifier(Endorsements(tdx_collateral=collateral_json("tdx_quote")), now=VALID_AT).verify_quote(quote)
    assert result.ok and result.status == "UpToDate", result.detail
    forged = bytearray(quote)
    forged[48 + 520 + 1] ^= 0x01  # REPORTDATA: a relay can't retarget a real quote at other keys
    assert not EndorsedQuoteVerifier(Endorsements(tdx_collateral=collateral_json("tdx_quote")), now=VALID_AT).verify_quote(bytes(forged)).ok
    other = EndorsedQuoteVerifier(Endorsements(tdx_collateral=collateral_json("tdx_quote_outdated")), now=VALID_AT).verify_quote(quote)
    assert not other.ok
    tampered = collateral_json("tdx_quote")
    tampered["tcb_info"] = tampered["tcb_info"].replace("UpToDate", "UpToDate ", 1)
    assert not EndorsedQuoteVerifier(Endorsements(tdx_collateral=tampered), now=VALID_AT).verify_quote(quote).ok


def test_the_gateway_keeps_the_collateral_a_quote_verified_against():
    from kuno_protocol.tdx import verify_tdx_quote

    quote = (DCAP / "tdx_quote").read_bytes()
    result = verify_tdx_quote(quote, collateral=dcap_qvl.QuoteCollateralV3.from_json((DCAP / "tdx_quote_collateral.json").read_text()), now=VALID_AT)
    assert result.ok and result.collateral == collateral_json("tdx_quote")
