"""Hardware identities come only from verified evidence: the PPID in a real Intel PCK certificate,
NVIDIA's signed per-GPU ueid claims, and the mock TEE's signed simulated machine."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from kuno_protocol import attestation
from kuno_protocol.attestation import (
    AllowedMeasurement,
    AttestationPolicy,
    GoldenManifest,
    MockTEE,
    Verdict,
    build_evidence,
    mock_measurements,
    verify_evidence,
)
from kuno_protocol.canonical import b64e, canonical_json
from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key, public_key_bytes
from kuno_protocol.hardware import (
    HardwareIdentity,
    capacity_limit,
    hardware_token,
    mock_gpu_ueids,
    pck_chain_pem,
    pck_ppid_from_quote,
    ppid_from_sgx_extensions,
)
from kuno_protocol.nvidia import GpuVerification, NrasGpuVerifier, NvattestGpuVerifier
from kuno_protocol.tdx import TdxQuoteResult

from test_nvidia import GOOD_GPU_CLAIMS, FakeNras, bundle

DCAP = Path(__file__).parent / "data" / "dcap"
QUOTE = (DCAP / "tdx_quote").read_bytes()
OUTDATED = (DCAP / "tdx_quote_outdated").read_bytes()
# What dcap-qvl 0.6.3 returns as VerifiedReport.ppid for the sample quote after full verification.
SAMPLE_PPID = "811dca2a26b952e85bb6448b097ba4fd"
OUTDATED_PPID = "66498c9263c04ed2f0657c530ac2b0cb"
DIGEST = "sha256:hardware-test"


# ---------------------------------------------------------------- TDX: PPID from the PCK certificate


def test_ppid_comes_from_the_pck_leaf_certificate_in_real_quotes():
    assert pck_ppid_from_quote(QUOTE).hex() == SAMPLE_PPID
    assert pck_ppid_from_quote(OUTDATED).hex() == OUTDATED_PPID
    assert pck_chain_pem(QUOTE).count(b"BEGIN CERTIFICATE") == 3


def test_dcap_qvl_reports_the_same_ppid_after_verifying_the_chain():
    dcap_qvl = pytest.importorskip("dcap_qvl")
    from kuno_protocol.tdx import verify_tdx_quote

    collateral = dcap_qvl.QuoteCollateralV3.from_json((DCAP / "tdx_quote_collateral.json").read_text())
    result = verify_tdx_quote(QUOTE, collateral=collateral, now=1751000000)
    assert result.ok and result.ppid.hex() == SAMPLE_PPID


def test_quotes_without_a_pck_chain_have_no_ppid():
    with pytest.raises(ValueError):
        pck_ppid_from_quote(QUOTE[:900])
    with pytest.raises(ValueError, match="no PPID"):
        ppid_from_sgx_extensions(bytes.fromhex("3000"))
    with pytest.raises(ValueError):
        ppid_from_sgx_extensions(bytes.fromhex("0400"))


class FakeQuoteVerifier:
    def __init__(self, ok: bool):
        self.ok = ok

    def verify(self, quote: bytes) -> tuple[bool, str]:
        return self.ok, "fake"


class PpidQuoteVerifier:
    def verify_quote(self, quote: bytes) -> TdxQuoteResult:
        return TdxQuoteResult(True, "fake", "UpToDate", [], bytes.fromhex("00" * 16))

    def verify(self, quote: bytes):
        raise AssertionError("verify_quote is preferred")


def test_the_platform_identity_exists_only_once_the_quote_verified():
    reasons: list[str] = []
    _, _, platform = attestation._quote_claims("tdx", QUOTE, GoldenManifest(), FakeQuoteVerifier(True), reasons)
    assert platform == HardwareIdentity("cpu_platform", hardware_token("cpu_platform", bytes.fromhex(SAMPLE_PPID)), "intel-pck-ppid")
    assert SAMPLE_PPID not in platform.token

    _, _, refused = attestation._quote_claims("tdx", QUOTE, GoldenManifest(), FakeQuoteVerifier(False), [])
    assert refused is None
    _, _, unverified = attestation._quote_claims("tdx", QUOTE, GoldenManifest(), None, [])
    assert unverified is None


def test_a_verifier_that_reports_the_ppid_is_believed_over_parsing():
    _, _, platform = attestation._quote_claims("tdx", QUOTE, GoldenManifest(), PpidQuoteVerifier(), [])
    assert platform.token == hardware_token("cpu_platform", bytes(16))


# ---------------------------------------------------------------- NVIDIA: ueid claims


def test_nras_reports_each_gpus_ueid_from_its_signed_token():
    nras, nonce = FakeNras(), os.urandom(32)
    nras.per_gpu = [{**GOOD_GPU_CLAIMS, "ueid": f"65533310790447807788282634442627054552420306731{i}"} for i in range(2)]
    result = NrasGpuVerifier(http=nras).verify_devices(bundle(nonce, gpus=2), nonce)
    assert isinstance(result, GpuVerification) and result.ok and result.gpu_count == 2
    assert result.ueids == ["655333107904478077882826344426270545524203067310", "655333107904478077882826344426270545524203067311"]
    assert NrasGpuVerifier(http=nras).verify(bundle(nonce, gpus=2), nonce)[0]  # the yes/no interface is unchanged


def test_refused_or_unidentified_gpu_claims_yield_no_ueids():
    nras, nonce = FakeNras(), os.urandom(32)
    nras.per_gpu = [{**GOOD_GPU_CLAIMS, "ueid": "1", "secboot": False}]
    refused = NrasGpuVerifier(http=nras).verify_devices(bundle(nonce), nonce)
    assert not refused.ok and refused.ueids == []
    nras.per_gpu = [dict(GOOD_GPU_CLAIMS)]
    assert NrasGpuVerifier(http=nras).verify_devices(bundle(nonce), nonce).ueids == [None]


def test_nvattest_reports_ueids_too():
    nonce = os.urandom(32)
    out = {"claims": [{**GOOD_GPU_CLAIMS, "eat_nonce": nonce.hex(), "ueid": "42"}], "result_code": 0, "result_message": "Ok"}

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(out), stderr="")

    result = NvattestGpuVerifier(run=run).verify_devices(bundle(nonce), nonce)
    assert result.ok and result.ueids == ["42"]


# ---------------------------------------------------------------- mock TEE


@pytest.fixture
def quote_key():
    return generate_signing_key()


def manifest_for(quote_key) -> GoldenManifest:
    return GoldenManifest(
        allowed=[AllowedMeasurement(platform="mock", image_digest=DIGEST, profiles=["ltx-2.5-fast"], **mock_measurements(DIGEST))],
        mock_quote_keys=[b64e(public_key_bytes(quote_key))],
    )


def attest(provider, nonce: bytes | None = None):
    nonce = nonce or os.urandom(32)
    _, hpke_public = generate_hpke_keypair()
    return build_evidence(provider, nonce, hpke_public, public_key_bytes(generate_signing_key()), DIGEST, ["ltx-2.5-fast"]), nonce


def test_mock_identities_follow_the_simulated_machine(quote_key):
    manifest = manifest_for(quote_key)
    first, nonce1 = attest(MockTEE(quote_key, DIGEST, machine_id="rig-1"))
    again, nonce2 = attest(MockTEE(quote_key, DIGEST, machine_id="rig-1", gpus=4))
    other, nonce3 = attest(MockTEE(quote_key, DIGEST, machine_id="rig-2"))
    v1, v2, v3 = (verify_evidence(e, manifest, n) for e, n in ((first, nonce1), (again, nonce2), (other, nonce3)))
    assert v1.ok and v2.ok and v3.ok
    assert v1.gpu_count == 4 and len(v1.hardware_tokens("gpu")) == 4 and len(v1.hardware_tokens("cpu_platform")) == 1
    assert v1.hardware_tokens() == v2.hardware_tokens()  # new keys, same machine
    assert not v1.hardware_tokens() & v3.hardware_tokens()
    assert all(h.source == "mock" for h in v1.hardware)
    # A simulated GPU never looks like a real one with the same id.
    assert hardware_token("gpu", mock_gpu_ueids("rig-1", 1)[0]) not in v1.hardware_tokens()


def test_each_mock_tee_is_its_own_machine_unless_told_otherwise(quote_key, monkeypatch):
    assert MockTEE(quote_key, DIGEST).machine_id != MockTEE(quote_key, DIGEST).machine_id
    monkeypatch.setenv("KUNO_MOCK_MACHINE_ID", "shared-rig")
    assert MockTEE(quote_key, DIGEST).machine_id == "shared-rig"


def test_a_refused_verdict_carries_no_identities(quote_key):
    evidence, _ = attest(MockTEE(quote_key, DIGEST, machine_id="rig-1"))
    verdict = verify_evidence(evidence, manifest_for(quote_key), expected_nonce=os.urandom(32))
    assert not verdict.ok and verdict.hardware == [] and verdict.gpu_count is None


def test_the_self_reported_hardware_dictionary_is_never_an_identity(quote_key):
    manifest = manifest_for(quote_key)
    evidence, nonce = attest(MockTEE(quote_key, DIGEST, machine_id="rig-1"))
    claimed = evidence.model_copy(update={"hardware": {"platform_id": "rig-2", "gpus": 64}})
    assert verify_evidence(claimed, manifest, nonce).hardware_tokens() == verify_evidence(evidence, manifest, nonce).hardware_tokens()


class RepeatedGpuTEE(MockTEE):
    """Repeats one GPU's evidence to look like more GPUs; REPORTDATA still binds it honestly."""

    def gpu_evidence(self, gpu_nonce: bytes) -> bytes | None:
        ueid = mock_gpu_ueids(self.machine_id, 1)[0]
        return canonical_json({"mock_gpu": "sim", "nonce": gpu_nonce.hex(), "gpus": [{"ueid": ueid}, {"ueid": ueid}]})


def test_one_gpu_repeated_in_the_evidence_is_refused(quote_key):
    evidence, nonce = attest(RepeatedGpuTEE(quote_key, DIGEST, machine_id="rig-1"))
    verdict = verify_evidence(evidence, manifest_for(quote_key), nonce)
    assert not verdict.ok and "the same GPU appears more than once in the GPU evidence" in verdict.reasons


def test_production_refuses_evidence_that_names_no_hardware(monkeypatch):
    policy = AttestationPolicy(
        production=True, quote_verifier=FakeQuoteVerifier(True), gpu_verifier=object(), owner_public_key=b"k" * 32
    )
    evidence, _ = attest(MockTEE(generate_signing_key(), DIGEST))
    tdx = evidence.model_copy(update={"tee": "tdx"})
    platform = HardwareIdentity("cpu_platform", hardware_token("cpu_platform", b"p"), "intel-pck-ppid")
    gpu = HardwareIdentity("gpu", hardware_token("gpu", "1"), "nvidia-ueid")

    def verified(hardware, gpu_count):
        return lambda *args, **kwargs: Verdict(True, "e", hardware=list(hardware), gpu_count=gpu_count)

    monkeypatch.setattr(attestation, "verify_evidence", verified([], None))
    refused = policy.verify(tdx, GoldenManifest())
    assert not refused.ok and refused.hardware == []
    assert any("no platform identity" in r for r in refused.reasons)
    assert any("did not report which GPUs" in r for r in refused.reasons)

    monkeypatch.setattr(attestation, "verify_evidence", verified([platform, gpu], 2))
    assert any("carries no device identity" in r for r in policy.verify(tdx, GoldenManifest()).reasons)

    monkeypatch.setattr(attestation, "verify_evidence", verified([platform, gpu], 1))
    accepted = policy.verify(tdx, GoldenManifest())
    assert accepted.ok and accepted.hardware == [platform, gpu]


# ---------------------------------------------------------------- tokens and capacity


def test_tokens_are_stable_salted_and_kind_separated():
    assert hardware_token("gpu", "ABC ") == hardware_token("gpu", "abc")
    assert hardware_token("gpu", "abc") != hardware_token("cpu_platform", "abc")
    assert hardware_token("gpu", "abc").startswith("hw1:") and len(hardware_token("gpu", "abc")) == 44
    with pytest.raises(ValueError):
        hardware_token("disk", "abc")


@pytest.mark.parametrize(
    "gpus, per_worker, limit",
    [(4, [1], 4), (4, [4], 1), (4, [1, 4], 1), (8, [4], 2), (2, [4], 0), (0, [1], 0), (3, [], 3)],
)
def test_capacity_is_bounded_by_the_largest_profile(gpus, per_worker, limit):
    assert capacity_limit(gpus, per_worker) == limit
