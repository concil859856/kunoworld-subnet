"""GPU confidential-computing modes and NVSwitch evidence: the bundle format, the verifiers, the manifest entry's
GPU fields and the production policy, against a fake NRAS that signs real ES384 tokens and a simulated TD."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time

import pytest

from kuno_protocol.attestation import (
    OPTIONAL_ENTRY_FIELDS,
    AllowedMeasurement,
    AttestationPolicy,
    GoldenManifest,
    GpuEvidenceUnavailable,
    MockTEE,
    TdxTEE,
    build_evidence,
    manifest_message,
    mock_measurements,
    parse_manifest,
    sign_manifest,
    verify_evidence,
)
from kuno_protocol.canonical import b64e, canonical_json
from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key, public_key_bytes
from kuno_protocol.hardware import hardware_token
from kuno_protocol.nvidia import (
    DEFAULT_NRAS_URL,
    GPU_EVIDENCE_FORMAT,
    GpuCcSettings,
    GpuEvidenceBundle,
    GpuEvidenceItem,
    NrasGpuVerifier,
    NvattestGpuVerifier,
    cc_problems,
)

from test_hardware_identity import PpidQuoteVerifier
from test_nvidia import GOOD_GPU_CLAIMS, MEASUREMENTS, FakeNras, synthetic_td_quote

GOOD_SWITCH_CLAIMS = {
    "measres": "success",
    "dbgstat": "disabled",
    "secboot": True,
    "hwmodel": "LS10",
    "x-nvidia-switch-attestation-report-nonce-match": True,
    "x-nvidia-switch-attestation-report-signature-verified": True,
}
PPCIE = GpuCcSettings(mode="ppcie", devtools=False)


def items(arch: str, count: int, nonce: bytes = b"") -> list[GpuEvidenceItem]:
    return [GpuEvidenceItem(arch=arch, evidence=base64.b64encode(f"{arch}-{i}".encode() + nonce).decode(), certificate="LS0t") for i in range(count)]


class SwitchingNras(FakeNras):
    """NRAS for GPUs and NVSwitches: detached tokens GPU-<n> from /attest/gpu and SWITCH<n> from /attest/switch."""

    def __init__(self):
        super().__init__()
        self.gpu_claims = lambda i: {**GOOD_GPU_CLAIMS, "ueid": f"gpu-{i}"}
        self.switch_claims = lambda i: {**GOOD_SWITCH_CLAIMS, "ueid": f"switch-{i}"}

    def __call__(self, method, url, body, headers, timeout):
        if url.endswith("/.well-known/jwks.json"):
            return super().__call__(method, url, body, headers, timeout)
        request = json.loads(body)
        self.requests.append({"url": url, "headers": headers, "body": request})
        switch = url.endswith("/attest/switch")
        claims, name = (self.switch_claims, "SWITCH{}") if switch else (self.gpu_claims, "GPU-{}")
        overall = {"x-nvidia-overall-att-result": True, "eat_nonce": request["nonce"], "exp": time.time() + 300}
        detached = {name.format(i): self.token(claims(i)) for i in range(len(request["evidence_list"]))}
        return 200, json.dumps([["JWT", self.token(overall)], detached]).encode()


# ---------------------------------------------------------------- the bundle


def test_evidence_without_a_mode_or_switches_keeps_the_bytes_it_always_had():
    nonce, gpus = os.urandom(32), items("HOPPER", 2)
    legacy = canonical_json({"format": GPU_EVIDENCE_FORMAT, "nonce": nonce.hex(), "gpus": [g.model_dump() for g in gpus]})
    assert GpuEvidenceBundle(nonce=nonce.hex(), gpus=gpus).encode() == legacy

    ppcie = GpuEvidenceBundle(nonce=nonce.hex(), gpus=gpus, cc=PPCIE, switches=items("LS10", 4))
    assert GpuEvidenceBundle.decode(ppcie.encode()) == ppcie
    assert json.loads(ppcie.encode())["cc"] == {"devtools": False, "mode": "ppcie"}
    with pytest.raises(ValueError):
        GpuEvidenceBundle.decode(canonical_json({**json.loads(legacy), "cc": {"mode": "hopper-cc", "devtools": False}}))


@pytest.mark.parametrize(
    ("mode", "architectures", "switches", "problem"),
    [
        ("ppcie", {"HOPPER"}, 4, None),
        ("ppcie", {"HOPPER"}, 0, "needs evidence from the VM's NVSwitches"),
        ("ppcie", {"BLACKWELL"}, 4, "only on Hopper"),
        ("mpt", {"BLACKWELL"}, 0, None),
        ("mpt", {"HOPPER"}, 0, "only on Blackwell"),
        ("mpt", {"BLACKWELL"}, 4, "keeps the NVSwitches out"),
        ("spt", {"HOPPER"}, 0, None),
        ("spt", {"BLACKWELL"}, 1, "keeps the NVSwitches out"),
        (None, {"HOPPER"}, 0, None),
        (None, {"HOPPER"}, 4, "declares no GPU confidential-computing mode"),
    ],
)
def test_the_devices_in_the_evidence_must_fit_its_declared_mode(mode, architectures, switches, problem):
    problems = cc_problems(GpuCcSettings(mode=mode, devtools=False) if mode else None, architectures, switches)
    assert problems == [] if problem is None else any(problem in p for p in problems)


# ---------------------------------------------------------------- verifiers


def test_nras_attests_the_nvswitches_of_a_protected_pcie_vm_after_its_gpus():
    nras, nonce = SwitchingNras(), os.urandom(32)
    evidence = GpuEvidenceBundle(nonce=nonce.hex(), gpus=items("HOPPER", 4), cc=PPCIE, switches=items("LS10", 4)).encode()
    result = NrasGpuVerifier(http=nras).verify_devices(evidence, nonce)
    assert result.ok, result.detail
    assert result.cc == PPCIE and result.gpu_count == 4 and result.switch_count == 4
    assert result.switch_ueids == [f"switch-{i}" for i in range(4)] and "4 NVSwitch(es)" in result.detail
    assert [r["url"] for r in nras.requests] == [DEFAULT_NRAS_URL, "https://nras.attestation.nvidia.com/v4/attest/switch"]
    assert nras.requests[1]["body"]["arch"] == "LS10" and nras.requests[1]["body"]["nonce"] == nonce.hex()

    nras.switch_claims = lambda i: {**GOOD_SWITCH_CLAIMS, "ueid": "s", "x-nvidia-switch-attestation-report-nonce-match": False}
    refused = NrasGpuVerifier(http=nras).verify_devices(evidence, nonce)
    assert not refused.ok and "NVSwitch evidence: SWITCH0: x-nvidia-switch-attestation-report-nonce-match" in refused.detail
    assert refused.switch_ueids == [] and refused.cc is None

    nras.requests.clear()
    no_switches = GpuEvidenceBundle(nonce=nonce.hex(), gpus=items("HOPPER", 4), cc=PPCIE).encode()
    refused = NrasGpuVerifier(http=nras).verify_devices(no_switches, nonce)
    assert not refused.ok and "NVSwitches" in refused.detail and nras.requests == []  # refused before any NRAS call


def test_nvattest_attests_nvswitches_with_its_nvswitch_device():
    nonce, commands = os.urandom(32), []

    def run(command, **kwargs):
        commands.append(command)
        switch = command[command.index("--device") + 1] == "nvswitch"
        claim = {**(GOOD_SWITCH_CLAIMS if switch else GOOD_GPU_CLAIMS), "eat_nonce": nonce.hex(), "ueid": "nvs" if switch else "g"}
        count = 4 if switch else 1
        claims = [{**claim, "ueid": f"{claim['ueid']}{i}"} for i in range(count)]
        return subprocess.CompletedProcess(command, 0, json.dumps({"claims": claims, "result_code": 0}), "")

    evidence = GpuEvidenceBundle(nonce=nonce.hex(), gpus=items("HOPPER", 1), cc=PPCIE, switches=items("LS10", 4)).encode()
    result = NvattestGpuVerifier(run=run).verify_devices(evidence, nonce)
    assert result.ok, result.detail
    assert result.ueids == ["g0"] and result.switch_ueids == ["nvs0", "nvs1", "nvs2", "nvs3"]
    switch_command = commands[1]
    assert switch_command[:4] == ["nvattest", "attest", "--device", "nvswitch"]
    assert "--nvswitch-evidence-source" in switch_command and "--nvswitch-evidence-file" in switch_command


# ---------------------------------------------------------------- TDX: collected, bound and matched to the manifest


class Devices:
    def __init__(self, arch: str, count: int):
        self.arch, self.count, self.calls = arch, count, 0

    def collect(self, gpu_nonce):
        self.calls += 1
        return items(self.arch, self.count, gpu_nonce)


class SimulatedTd(TdxTEE):
    def quote(self, report_data: bytes) -> bytes:
        return synthetic_td_quote(report_data)


def td(cc: GpuCcSettings | None, gpus: int = 4, switches: int = 4, arch: str = "HOPPER") -> SimulatedTd:
    return SimulatedTd(gpu_collector=Devices(arch, gpus), switch_collector=Devices("LS10", switches), cc_settings=(lambda: cc) if cc else None)


def entry(measurements=MEASUREMENTS, **gpu) -> AllowedMeasurement:
    return AllowedMeasurement(platform="tdx", image_digest="sha256:img", profiles=["h3"], **measurements, **gpu)


def attest(provider):
    _, hpke = generate_hpke_keypair()
    nonce = os.urandom(32)
    return build_evidence(provider, nonce, hpke, public_key_bytes(generate_signing_key()), "sha256:img", ["h3"]), nonce


def test_a_protected_pcie_td_attests_its_switches_and_must_fit_the_entry_its_measurements_matched():
    evidence, nonce = attest(td(PPCIE))
    bundle = GpuEvidenceBundle.decode(base64.urlsafe_b64decode(evidence.gpu_evidence + "=="))
    assert bundle.cc == PPCIE and len(bundle.switches) == 4
    kwargs = dict(expected_nonce=nonce, quote_verifier=PpidQuoteVerifier(), gpu_verifier=NrasGpuVerifier(http=SwitchingNras()))

    pinned = GoldenManifest(allowed=[entry(gpu_mode="ppcie", gpus_per_enclave=4, nvswitches_per_enclave=4)])
    verdict = verify_evidence(evidence, pinned, **kwargs)
    assert verdict.ok, verdict.reasons
    assert (verdict.gpu_mode, verdict.gpu_devtools, verdict.gpu_count, verdict.nvswitch_count) == ("ppcie", False, 4, 4)
    assert len(verdict.hardware_tokens("nvswitch")) == 4 and len(verdict.hardware_tokens("gpu")) == 4

    for gpu, reason in (
        (dict(gpu_mode="mpt"), "declares ppcie mode, but the manifest entry requires mpt"),
        (dict(gpus_per_enclave=8), "attests 4 GPU(s), but the manifest entry requires 8 per enclave"),
        (dict(nvswitches_per_enclave=0), "attests 4 NVSwitch(es), but the manifest entry requires 0 per enclave"),
    ):
        refused = verify_evidence(evidence, GoldenManifest(allowed=[entry(**gpu)]), **kwargs)
        assert not refused.ok and any(reason in r for r in refused.reasons), refused.reasons
        assert refused.hardware == [] and refused.gpu_mode is None and refused.nvswitch_count is None

    # Another entry's GPU fields don't apply: only the entry these measurements matched decides.
    other = {k: "00" * 48 for k in MEASUREMENTS}
    assert verify_evidence(evidence, GoldenManifest(allowed=[entry(other, gpu_mode="mpt"), entry()]), **kwargs).ok

    legacy, legacy_nonce = attest(td(None))
    refused = verify_evidence(legacy, pinned, **{**kwargs, "expected_nonce": legacy_nonce})
    assert "the GPU evidence declares no confidential-computing mode, but the manifest entry requires ppcie" in refused.reasons


def test_the_td_collects_switch_evidence_only_in_protected_pcie_mode():
    single = td(GpuCcSettings(mode="spt", devtools=False), gpus=1)
    bundle = GpuEvidenceBundle.decode(single.gpu_evidence(os.urandom(32)))
    assert bundle.switches is None and single.switch_collector.calls == 0
    with pytest.raises(GpuEvidenceUnavailable, match="no NVSwitch evidence collector"):
        SimulatedTd(gpu_collector=Devices("HOPPER", 4), cc_settings=lambda: PPCIE).gpu_evidence(os.urandom(32))
    with pytest.raises(GpuEvidenceUnavailable, match="no NVSwitch evidence was collected"):
        td(PPCIE, switches=0).gpu_evidence(os.urandom(32))


def test_production_needs_a_declared_mode_without_devtools_and_an_identity_for_every_switch():
    manifest = GoldenManifest(allowed=[entry()])
    nras = SwitchingNras()
    policy = AttestationPolicy(production=True, quote_verifier=PpidQuoteVerifier(), gpu_verifier=NrasGpuVerifier(http=nras), owner_public_key=b"k" * 32)

    evidence, nonce = attest(td(PPCIE))
    assert policy.verify(evidence, manifest, nonce).ok

    devtools, nonce = attest(td(GpuCcSettings(mode="ppcie", devtools=True)))
    assert "the GPUs run in devtools mode, which production refuses" in policy.verify(devtools, manifest, nonce).reasons
    assert verify_evidence(devtools, manifest, nonce, quote_verifier=PpidQuoteVerifier(), gpu_verifier=NrasGpuVerifier(http=nras)).gpu_devtools

    undeclared, nonce = attest(td(None))
    assert any("declares no confidential-computing mode" in r for r in policy.verify(undeclared, manifest, nonce).reasons)

    nras.switch_claims = lambda i: dict(GOOD_SWITCH_CLAIMS)
    unnamed, nonce = attest(td(PPCIE))
    assert "an attested NVSwitch carries no device identity (ueid)" in policy.verify(unnamed, manifest, nonce).reasons


# ---------------------------------------------------------------- the simulated TEE


@pytest.fixture
def quote_key():
    return generate_signing_key()


def mock_manifest(quote_key, **gpu) -> GoldenManifest:
    digest = "sha256:mock-hgx"
    return GoldenManifest(
        allowed=[AllowedMeasurement(platform="mock", image_digest=digest, profiles=["h3"], **mock_measurements(digest), **gpu)],
        mock_quote_keys=[b64e(public_key_bytes(quote_key))],
    )


def mock_verdict(quote_key, manifest, **tee):
    evidence, nonce = attest_mock(MockTEE(quote_key, "sha256:mock-hgx", machine_id="hgx-1", **tee))
    return verify_evidence(evidence, manifest, nonce)


def attest_mock(provider):
    _, hpke = generate_hpke_keypair()
    nonce = os.urandom(32)
    return build_evidence(provider, nonce, hpke, public_key_bytes(generate_signing_key()), "sha256:mock-hgx", ["h3"]), nonce


def test_two_simulated_workers_of_one_protected_pcie_vm_share_the_platform_and_switches_but_no_gpu(quote_key):
    manifest = mock_manifest(quote_key, gpu_mode="ppcie", gpus_per_enclave=4, nvswitches_per_enclave=4)
    first = mock_verdict(quote_key, manifest, gpu_indices=[0, 1, 2, 3], gpu_mode="ppcie", nvswitches=4)
    second = mock_verdict(quote_key, manifest, gpu_indices=[4, 5, 6, 7], gpu_mode="ppcie", nvswitches=4)
    assert first.ok and second.ok, first.reasons + second.reasons
    assert (first.gpu_mode, first.gpu_count, first.nvswitch_count) == ("ppcie", 4, 4)
    assert first.hardware_tokens("nvswitch") == second.hardware_tokens("nvswitch") and len(first.hardware_tokens("nvswitch")) == 4
    assert first.hardware_tokens("cpu_platform") == second.hardware_tokens("cpu_platform")
    assert not first.hardware_tokens("gpu") & second.hardware_tokens("gpu")

    whole = mock_verdict(quote_key, manifest, gpus=8, gpu_mode="ppcie", nvswitches=4)
    assert not whole.ok and "the evidence attests 8 GPU(s), but the manifest entry requires 4 per enclave" in whole.reasons
    single = mock_verdict(quote_key, mock_manifest(quote_key), gpus=1, gpu_mode="spt", nvswitches=4)
    assert not single.ok and any("keeps the NVSwitches out of the VM" in r for r in single.reasons)


def test_simulated_evidence_without_the_new_fields_is_unchanged(quote_key):
    nonce = os.urandom(32)
    legacy = {"mock_gpu": "NVIDIA H200 (simulated)", "nonce": nonce.hex(), "cc_mode": "on"}
    evidence = MockTEE(quote_key, "sha256:x", machine_id="rig").gpu_evidence(nonce)
    assert set(json.loads(evidence)) == {*legacy, "gpus"} and len(json.loads(evidence)["gpus"]) == 4
    assert hardware_token("nvswitch", "abc") not in (hardware_token("gpu", "abc"), hardware_token("cpu_platform", "abc"))


# ---------------------------------------------------------------- signed manifests


def test_manifests_signed_before_entries_had_gpu_fields_still_verify():
    owner = generate_signing_key()
    manifest = GoldenManifest(issued_at=1_700_000_000, allowed=[entry()])
    legacy_body = manifest.model_dump(mode="json")
    legacy_body.pop("open_tier")
    legacy_body.pop("model_digests")
    for item in legacy_body["allowed"]:
        for name in OPTIONAL_ENTRY_FIELDS:
            item.pop(name)
    legacy_message = b"kuno/v1/manifest\n" + canonical_json(legacy_body)
    assert manifest_message(manifest) == legacy_message
    signed = {"manifest": legacy_body, "signature": b64e(owner.sign(legacy_message))}
    assert parse_manifest(json.dumps(signed), public_key_bytes(owner), require_signature=True) == manifest

    tampered = json.loads(json.dumps(signed))
    tampered["manifest"]["allowed"][0]["gpu_mode"] = "mpt"
    with pytest.raises(ValueError, match="does not verify"):
        parse_manifest(json.dumps(tampered), public_key_bytes(owner))

    pinned = GoldenManifest(issued_at=1_700_000_000, allowed=[entry(gpu_mode="ppcie", gpus_per_enclave=4, nvswitches_per_enclave=4)])
    assert manifest_message(pinned) != legacy_message
    parsed = parse_manifest(sign_manifest(owner, pinned).model_dump_json(), public_key_bytes(owner), require_signature=True)
    assert parsed.allowed[0].gpu_mode == "ppcie" and parsed.allowed[0].nvswitches_per_enclave == 4
