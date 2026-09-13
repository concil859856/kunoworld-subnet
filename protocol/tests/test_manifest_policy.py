"""Owner-signed golden manifests and the production attestation policy."""

from __future__ import annotations

import os

import pytest

from kuno_protocol import devkit
from kuno_protocol.attestation import (
    AttestationPolicy,
    GoldenManifest,
    ManifestError,
    MockTEE,
    PolicyError,
    SignedManifest,
    build_evidence,
    load_manifest,
    parse_manifest,
    sign_manifest,
    verify_evidence,
)
from kuno_protocol.canonical import b64d
from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key, public_key_bytes, signing_key_from_bytes
from kuno_protocol.hotkey import Sr25519Signer
from kuno_protocol.policy import policy_from_env


class Accepting:
    def verify(self, *_args):
        return True, "ok"


@pytest.fixture
def dev(tmp_path):
    env = devkit.init(tmp_path)
    owner = signing_key_from_bytes(b64d((tmp_path / "owner.key").read_text()))
    quote_key = signing_key_from_bytes(b64d((tmp_path / "mock_quote.key").read_text()))
    _, hpke_pk = generate_hpke_keypair()
    nonce = os.urandom(32)
    evidence = build_evidence(
        MockTEE(quote_key, devkit.DEV_IMAGE_DIGEST), nonce, hpke_pk, public_key_bytes(generate_signing_key()), devkit.DEV_IMAGE_DIGEST, ["h3"]
    )
    return env, owner, evidence, nonce


def test_devkit_writes_a_signed_manifest_and_keeps_the_bare_one(dev, tmp_path):
    env, owner, _, _ = dev
    owner_pk = b64d(env["KUNO_OWNER_PUBLIC_KEY"])
    signed = load_manifest(env["KUNO_SIGNED_MANIFEST"], owner_pk, require_signature=True)
    bare = GoldenManifest.model_validate_json((tmp_path / "manifest.json").read_text())
    assert signed == bare == load_manifest(tmp_path / "manifest.json")
    seed = bytes.fromhex((tmp_path / "hotkey.seed").read_text()[2:])
    assert Sr25519Signer.from_seed(seed).ss58_address == env["KUNO_MINER_HOTKEY"]
    assert (tmp_path / "hotkey.seed").stat().st_mode & 0o077 == 0


def test_manifest_signature_is_checked_whenever_an_owner_key_is_configured(dev, tmp_path):
    env, owner, _, _ = dev
    owner_pk = b64d(env["KUNO_OWNER_PUBLIC_KEY"])
    manifest = GoldenManifest.model_validate_json((tmp_path / "manifest.json").read_text())

    tampered = sign_manifest(owner, manifest).model_copy(update={"manifest": manifest.model_copy(update={"max_evidence_age_s": 10**9})})
    with pytest.raises(ManifestError, match="does not verify"):
        parse_manifest(tampered.model_dump_json(), owner_pk)
    impostor = sign_manifest(generate_signing_key(), manifest).model_dump_json()
    with pytest.raises(ManifestError, match="does not verify"):
        parse_manifest(impostor, owner_pk)
    assert parse_manifest(impostor) == manifest  # no key configured: development stays permissive

    with pytest.raises(ManifestError, match="unsigned manifest refused"):
        parse_manifest(manifest.model_dump_json(), owner_pk, require_signature=True)
    with pytest.raises(ManifestError, match="no owner public key"):
        parse_manifest(sign_manifest(owner, manifest).model_dump_json(), None, require_signature=True)
    with pytest.raises(ManifestError, match="not JSON"):
        parse_manifest("nope")
    unsigned = SignedManifest(manifest=manifest).model_dump_json()
    with pytest.raises(ManifestError):
        parse_manifest(unsigned, owner_pk)


def test_production_policy_needs_both_verifiers_and_the_owner_key():
    with pytest.raises(PolicyError, match="TDX quote verifier.*GPU evidence verifier.*owner public key"):
        AttestationPolicy(production=True)
    with pytest.raises(PolicyError, match="GPU evidence verifier"):
        AttestationPolicy(production=True, quote_verifier=Accepting(), owner_public_key=b"k" * 32)


def test_production_rejects_mock_evidence_that_dev_accepts(dev, tmp_path):
    env, owner, evidence, nonce = dev
    owner_pk = b64d(env["KUNO_OWNER_PUBLIC_KEY"])
    manifest = load_manifest(env["KUNO_SIGNED_MANIFEST"], owner_pk)

    dev_policy = AttestationPolicy(owner_public_key=owner_pk)
    assert dev_policy.verify(evidence, manifest, nonce).ok
    assert dev_policy.verify(evidence, manifest, nonce).reasons == verify_evidence(evidence, manifest, nonce).reasons
    assert dev_policy.load_manifest(tmp_path / "manifest.json") == manifest

    production = AttestationPolicy(production=True, quote_verifier=Accepting(), gpu_verifier=Accepting(), owner_public_key=owner_pk)
    verdict = production.verify(evidence, manifest, nonce)
    assert not verdict.ok and verdict.reasons[0] == "mock evidence is not accepted in production"
    with pytest.raises(ManifestError, match="simulated TEE"):
        production.load_manifest(env["KUNO_SIGNED_MANIFEST"])
    with pytest.raises(ManifestError, match="unsigned"):
        production.load_manifest(tmp_path / "manifest.json")

    clean = GoldenManifest(allowed=[a.model_copy(update={"platform": "tdx"}) for a in manifest.allowed])
    path = tmp_path / "production.json"
    path.write_text(sign_manifest(owner, clean).model_dump_json())
    assert production.load_manifest(path) == clean


def test_policy_from_env():
    dev = policy_from_env({})
    assert not dev.production and dev.quote_verifier is None and dev.gpu_verifier is None
    with pytest.raises(PolicyError):
        policy_from_env({"KUNO_ATTESTATION": "prod"})
    with pytest.raises(PolicyError, match="KUNO_GPU_VERIFIER"):
        policy_from_env({"KUNO_TDX_VERIFY": "1", "KUNO_GPU_VERIFIER": "cloud"})


def test_production_without_dcap_or_owner_key_refuses_to_start():
    from kuno_protocol.tdx import TdxVerifierUnavailable

    with pytest.raises((PolicyError, TdxVerifierUnavailable)):
        policy_from_env({"KUNO_ATTESTATION": "production"})


def test_policy_from_env_builds_real_verifiers_in_production():
    pytest.importorskip("dcap_qvl")
    with pytest.raises(PolicyError, match="owner public key"):
        policy_from_env({"KUNO_ATTESTATION": "production"})
    owner = public_key_bytes(generate_signing_key())
    from kuno_protocol.canonical import b64e
    from kuno_protocol.nvidia import NrasGpuVerifier, NvattestGpuVerifier
    from kuno_protocol.tdx import DcapQuoteVerifier

    policy = policy_from_env(
        {"KUNO_ATTESTATION": "production", "KUNO_OWNER_PUBLIC_KEY": b64e(owner), "KUNO_TDX_TCB_ALLOWED": "UpToDate,SWHardeningNeeded"}
    )
    assert isinstance(policy.quote_verifier, DcapQuoteVerifier) and isinstance(policy.gpu_verifier, NrasGpuVerifier)
    assert policy.quote_verifier.allowed_statuses == ("UpToDate", "SWHardeningNeeded")
    local = policy_from_env({"KUNO_TDX_VERIFY": "1", "KUNO_GPU_VERIFIER": "local"})
    assert isinstance(local.gpu_verifier, NvattestGpuVerifier) and not local.production
    with pytest.raises(ValueError, match="unknown TCB statuses"):
        policy_from_env({"KUNO_TDX_VERIFY": "1", "KUNO_TDX_TCB_ALLOWED": "Fine"})
