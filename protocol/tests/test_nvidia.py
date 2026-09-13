"""NVIDIA GPU evidence verification, against a fake NRAS that signs real ES384 tokens.

No public NRAS response or SPDM evidence sample with a known nonce was available, so the
tokens here are produced by a local P-384 key in the exact shape NRAS returns; what the
tests prove is our signature, claim and binding logic, not NVIDIA's service.
"""

from __future__ import annotations

import base64
import datetime
import json
import os
import subprocess
import time

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.x509.oid import NameOID

from kuno_protocol.attestation import (
    AllowedMeasurement,
    GoldenManifest,
    GpuEvidenceUnavailable,
    TdxQuoteUnavailable,
    TdxTEE,
    build_evidence,
    gpu_nonce_for,
    parse_tdx_quote,
    verify_evidence,
)
from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key, public_key_bytes
from kuno_protocol.nvidia import (
    GpuEvidenceBundle,
    GpuEvidenceItem,
    NrasGpuVerifier,
    NvattestGpuVerifier,
    gpu_claim_problems,
    verify_es384_jwt,
)

GOOD_GPU_CLAIMS = {
    "measres": "success",
    "dbgstat": "disabled",
    "secboot": True,
    "hwmodel": "GH100 A01 GSP BROM",
    "x-nvidia-gpu-attestation-report-nonce-match": True,
    "x-nvidia-gpu-attestation-report-signature-verified": True,
}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


class FakeNras:
    """Signs detached EAT bundles like NRAS and serves its JWKS."""

    def __init__(self, x5c: bool = True):
        self.key = ec.generate_private_key(ec.SECP384R1())
        self.kid = "nv-eat-kid-test"
        numbers = self.key.public_key().public_numbers()
        jwk = {"kty": "EC", "crv": "P-384", "kid": self.kid, "x": _b64url(numbers.x.to_bytes(48, "big")), "y": _b64url(numbers.y.to_bytes(48, "big"))}
        if x5c:
            name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "NRAS test")])
            now = datetime.datetime.now(datetime.timezone.utc)
            cert = (
                x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(self.key.public_key())
                .serial_number(1).not_valid_before(now).not_valid_after(now + datetime.timedelta(days=1))
                .sign(self.key, hashes.SHA384())
            )
            jwk["x5c"] = [base64.b64encode(cert.public_bytes(serialization.Encoding.DER)).decode()]
        self.jwks = {"keys": [jwk]}
        self.requests: list[dict] = []
        self.jwks_fetches = 0
        self.overall: dict = {}
        self.per_gpu: list[dict] | None = None
        self.status = 200

    def token(self, claims: dict, key=None, kid=None) -> str:
        header = _b64url(json.dumps({"alg": "ES384", "typ": "JWT", "kid": kid or self.kid}).encode())
        payload = _b64url(json.dumps(claims).encode())
        r, s = decode_dss_signature((key or self.key).sign(f"{header}.{payload}".encode(), ec.ECDSA(hashes.SHA384())))
        return f"{header}.{payload}.{_b64url(r.to_bytes(48, 'big') + s.to_bytes(48, 'big'))}"

    def __call__(self, method, url, body, headers, timeout):
        if url.endswith("/.well-known/jwks.json"):
            self.jwks_fetches += 1
            return 200, json.dumps(self.jwks).encode()
        request = json.loads(body)
        self.requests.append({"url": url, "headers": headers, "body": request})
        if self.status != 200:
            return self.status, b'{"error":"boom"}'
        count = len(request["evidence_list"])
        per_gpu = self.per_gpu if self.per_gpu is not None else [dict(GOOD_GPU_CLAIMS) for _ in range(count)]
        overall = {"x-nvidia-overall-att-result": True, "eat_nonce": request["nonce"], "exp": time.time() + 300, **self.overall}
        detached = {f"GPU-{i}": self.token(claims) for i, claims in enumerate(per_gpu)}
        return 200, json.dumps([["JWT", self.token(overall)], detached]).encode()


def bundle(nonce: bytes, gpus: int = 1, arch: str = "HOPPER") -> bytes:
    items = [GpuEvidenceItem(arch=arch, evidence=base64.b64encode(os.urandom(64)).decode(), certificate="LS0t") for _ in range(gpus)]
    return GpuEvidenceBundle(nonce=nonce.hex(), gpus=items).encode()


@pytest.mark.parametrize("x5c", [True, False])
def test_nras_verifier_accepts_signed_good_claims_and_sends_nras_shaped_request(x5c):
    nras, nonce = FakeNras(x5c=x5c), os.urandom(32)
    verifier = NrasGpuVerifier(service_key="sk-test", http=nras)
    ok, detail = verifier.verify(bundle(nonce, gpus=4), nonce)
    assert ok, detail
    assert "4 GPU(s)" in detail
    sent = nras.requests[0]
    assert sent["url"] == "https://nras.attestation.nvidia.com/v4/attest/gpu"
    assert sent["headers"]["authorization"] == "Bearer sk-test"
    assert sent["body"]["nonce"] == nonce.hex() and sent["body"]["arch"] == "HOPPER"
    assert set(sent["body"]["evidence_list"][0]) == {"evidence", "certificate"}
    verifier.verify(bundle(nonce), nonce)
    assert nras.jwks_fetches == 1  # cached


@pytest.mark.parametrize(
    "overall, per_gpu, expected",
    [
        ({"x-nvidia-overall-att-result": False}, None, "overall attestation result"),
        ({"eat_nonce": "00" * 32}, None, "different nonce"),
        ({}, [{**GOOD_GPU_CLAIMS, "measres": "fail"}], "reference values"),
        ({}, [{**GOOD_GPU_CLAIMS, "dbgstat": "enabled"}], "debug"),
        ({}, [{**GOOD_GPU_CLAIMS, "secboot": False}], "secure boot"),
        ({}, [{**GOOD_GPU_CLAIMS, "x-nvidia-gpu-attestation-report-nonce-match": False}], "nonce-match"),
        ({}, [], "attested 0 GPU(s)"),
    ],
)
def test_nras_verifier_rejects_bad_claims(overall, per_gpu, expected):
    nras, nonce = FakeNras(), os.urandom(32)
    nras.overall, nras.per_gpu = overall, per_gpu
    ok, detail = NrasGpuVerifier(http=nras).verify(bundle(nonce), nonce)
    assert not ok and expected in detail


def test_nras_verifier_rejects_forged_tokens_wrong_nonce_and_http_errors():
    nras, nonce = FakeNras(), os.urandom(32)
    forged = FakeNras()
    forged.kid = nras.kid  # same key id, different key: the signature must not verify
    ok, detail = NrasGpuVerifier(http=lambda m, u, b, h, t: forged(m, u, b, h, t) if "jwks" not in u else nras(m, u, b, h, t)).verify(
        bundle(nonce), nonce
    )
    assert not ok and "signature is invalid" in detail

    ok, detail = NrasGpuVerifier(http=nras).verify(bundle(os.urandom(32)), nonce)
    assert not ok and "different nonce" in detail

    nras.status = 500
    ok, detail = NrasGpuVerifier(http=nras).verify(bundle(nonce), nonce)
    assert not ok and "HTTP 500" in detail

    ok, detail = NrasGpuVerifier(http=nras).verify(b"not evidence", nonce)
    assert not ok and "kuno/v1/nvidia-gpu" in detail

    def unreachable(*_args):
        raise OSError("connection refused")

    ok, detail = NrasGpuVerifier(http=unreachable).verify(bundle(nonce), nonce)
    assert not ok and "connection refused" in detail


def test_rotated_nras_key_triggers_one_jwks_refresh():
    nras, nonce = FakeNras(), os.urandom(32)
    verifier = NrasGpuVerifier(http=nras)
    assert verifier.verify(bundle(nonce), nonce)[0]
    old = nras.jwks
    rotated = FakeNras()
    rotated.kid = "nv-eat-kid-rotated"
    rotated.jwks["keys"][0]["kid"] = rotated.kid
    nras.key, nras.kid, nras.jwks = rotated.key, rotated.kid, rotated.jwks
    assert verifier.verify(bundle(nonce), nonce)[0]
    assert nras.jwks_fetches == 2 and old != nras.jwks


def test_jwt_rejects_other_algorithms_and_expiry():
    nras = FakeNras()
    token = nras.token({"exp": 100})
    with pytest.raises(ValueError, match="expired"):
        verify_es384_jwt(token, nras.jwks, now=1000)
    header = _b64url(json.dumps({"alg": "none", "kid": nras.kid}).encode())
    with pytest.raises(ValueError, match="algorithm"):
        verify_es384_jwt(f"{header}.{_b64url(b'{}')}.", nras.jwks, now=0)


def test_nvattest_spellings_of_claims_are_understood():
    assert gpu_claim_problems({**GOOD_GPU_CLAIMS, "measres": "Success", "dbgstat": False}) == []
    assert gpu_claim_problems({**GOOD_GPU_CLAIMS, "dbgstat": None})


def test_local_nvattest_verifier_runs_nvidia_cli_on_an_evidence_file():
    nonce, seen = os.urandom(32), {}

    def run(command, **kwargs):
        seen["command"] = command
        evidence_file = command[command.index("--gpu-evidence-file") + 1]
        seen["file"] = json.loads(open(evidence_file).read())
        out = {"claims": [{**GOOD_GPU_CLAIMS, "eat_nonce": nonce.hex()}] * 2, "result_code": 0, "result_message": "Ok"}
        return subprocess.CompletedProcess(command, 0, json.dumps(out), "")

    ok, detail = NvattestGpuVerifier(run=run).verify(bundle(nonce, gpus=2), nonce)
    assert ok, detail
    assert seen["command"][:3] == ["nvattest", "attest", "--device"] and "--verifier" in seen["command"]
    assert len(seen["file"]["evidences"]) == 2 and seen["file"]["evidences"][0]["nonce"] == nonce.hex()

    def failing(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, json.dumps({"result_code": 12, "result_message": "RIM mismatch"}), "")

    ok, detail = NvattestGpuVerifier(run=failing).verify(bundle(nonce), nonce)
    assert not ok and "RIM mismatch" in detail

    def missing(command, **kwargs):
        raise FileNotFoundError

    ok, detail = NvattestGpuVerifier(run=missing).verify(bundle(nonce), nonce)
    assert not ok and "not installed" in detail


# ---------------------------------------------------------------- the whole TDX path, simulated below the kernel


class FakeCollector:
    def __init__(self):
        self.nonces: list[bytes] = []

    def collect(self, gpu_nonce):
        self.nonces.append(gpu_nonce)
        return [GpuEvidenceItem(arch="HOPPER", evidence=base64.b64encode(b"spdm" + gpu_nonce).decode(), certificate="LS0t")]


MEASUREMENTS = {k: os.urandom(48).hex() for k in ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3")}


def synthetic_td_quote(report_data: bytes, version: int = 4, td_attributes: bytes = b"\x00" * 8) -> bytes:
    header = version.to_bytes(2, "little") + (2).to_bytes(2, "little") + (0x81).to_bytes(4, "little") + b"\x00" * 40
    body = (
        b"\x00" * 16 + b"\x00" * 48 * 2 + b"\x00" * 8 + td_attributes + b"\x00" * 8
        + bytes.fromhex(MEASUREMENTS["mrtd"]) + b"\x00" * 48 * 3
        + b"".join(bytes.fromhex(MEASUREMENTS[k]) for k in ("rtmr0", "rtmr1", "rtmr2", "rtmr3")) + report_data
    )
    descriptor = (3).to_bytes(2, "little") + len(body).to_bytes(4, "little") if version == 5 else b""
    return header + descriptor + body + b"\x00" * 64  # signature data follows the body in real quotes


class SimulatedTdx(TdxTEE):
    def __init__(self, collector, version=4, td_attributes=b"\x00" * 8):
        super().__init__(gpu_collector=collector)
        self.version, self.td_attributes = version, td_attributes

    def quote(self, report_data: bytes) -> bytes:
        return synthetic_td_quote(report_data, self.version, self.td_attributes)


class AcceptingQuoteVerifier:
    def verify(self, quote):
        return True, "TCB status UpToDate"


def tdx_manifest() -> GoldenManifest:
    return GoldenManifest(allowed=[AllowedMeasurement(platform="tdx", image_digest="sha256:img", profiles=["ltx-2.5-fast"], **MEASUREMENTS)])


@pytest.mark.parametrize("version", [4, 5])
def test_tdx_evidence_binds_gpu_evidence_keys_and_nonce_end_to_end(version):
    collector, nras = FakeCollector(), FakeNras()
    _, hpke_pk = generate_hpke_keypair()
    sign_pk, nonce = public_key_bytes(generate_signing_key()), os.urandom(32)
    evidence = build_evidence(SimulatedTdx(collector, version), nonce, hpke_pk, sign_pk, "sha256:img", ["ltx-2.5-fast"])
    assert collector.nonces == [gpu_nonce_for(nonce, hpke_pk, sign_pk)]

    kwargs = dict(expected_nonce=nonce, quote_verifier=AcceptingQuoteVerifier(), gpu_verifier=NrasGpuVerifier(http=nras))
    verdict = verify_evidence(evidence, tdx_manifest(), **kwargs)
    assert verdict.ok, verdict.reasons
    assert nras.requests[0]["body"]["nonce"] == gpu_nonce_for(nonce, hpke_pk, sign_pk).hex()

    # Swapping in GPU evidence from another attestation breaks REPORTDATA even if NRAS would accept it.
    other = build_evidence(SimulatedTdx(FakeCollector(), version), os.urandom(32), hpke_pk, sign_pk, "sha256:img", ["ltx-2.5-fast"])
    spliced = evidence.model_copy(update={"gpu_evidence": other.gpu_evidence})
    assert any("REPORTDATA" in r for r in verify_evidence(spliced, tdx_manifest(), **kwargs).reasons)

    assert "no TDX quote verifier configured" in verify_evidence(evidence, tdx_manifest(), expected_nonce=nonce).reasons


def test_debug_tds_and_unknown_quote_versions_are_rejected():
    collector = FakeCollector()
    _, hpke_pk = generate_hpke_keypair()
    sign_pk, nonce = public_key_bytes(generate_signing_key()), os.urandom(32)
    debug = build_evidence(SimulatedTdx(collector, td_attributes=b"\x01" + b"\x00" * 7), nonce, hpke_pk, sign_pk, "sha256:img", ["ltx-2.5-fast"])
    reasons = verify_evidence(debug, tdx_manifest(), quote_verifier=AcceptingQuoteVerifier(), gpu_verifier=NrasGpuVerifier(http=FakeNras())).reasons
    assert any("debug mode" in r for r in reasons)
    with pytest.raises(ValueError, match="v4/v5"):
        parse_tdx_quote(synthetic_td_quote(b"\x00" * 64, version=3))


def test_tdx_provider_explains_what_is_missing(tmp_path):
    with pytest.raises(TdxQuoteUnavailable, match="configfs"):
        TdxTEE(tsm_root=tmp_path / "missing").quote(b"\x00" * 64)
    with pytest.raises(GpuEvidenceUnavailable, match="nvattest"):
        TdxTEE().gpu_evidence(b"\x00" * 32)

    class Empty:
        def collect(self, _nonce):
            return []

    with pytest.raises(GpuEvidenceUnavailable, match="passthrough"):
        TdxTEE(gpu_collector=Empty()).gpu_evidence(b"\x00" * 32)
