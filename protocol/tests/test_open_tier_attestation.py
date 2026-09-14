"""Open-tier evidence: no quote, accepted only where the owner's manifest allows the image and the caller asks for it."""

from __future__ import annotations

import os
import time

import pytest

from kuno_protocol import devkit
from kuno_protocol.attestation import (
    AttestationPolicy,
    GoldenManifest,
    OpenTEE,
    OpenTierImage,
    OpenTierPolicy,
    build_evidence,
    manifest_message,
    parse_manifest,
    sign_manifest,
    verify_evidence,
)
from kuno_protocol.canonical import b64d, b64e, canonical_json
from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key, public_key_bytes, signing_key_from_bytes
from kuno_protocol.tiers import CONFIDENTIAL, OPEN, hotkey_proof_required

IMAGE = "sha256:kuno-worker-open-1"


class Accepting:
    def verify(self, *_args):
        return True, "ok"


def open_manifest(**policy) -> GoldenManifest:
    fields = {"enabled": True, "images": [OpenTierImage(image_digest=IMAGE, profiles=["ltx-2.5-fast", "ltx-2.5-pro"])], **policy}
    return GoldenManifest(open_tier=OpenTierPolicy(**fields))


def open_evidence(profiles=("ltx-2.5-fast",), image=IMAGE, nonce=None):
    nonce = nonce or os.urandom(32)
    _, hpke = generate_hpke_keypair()
    evidence = build_evidence(
        OpenTEE(), nonce, hpke, public_key_bytes(generate_signing_key()), image, list(profiles), {"gpu": "RTX 5090", "gpu_count": 1}
    )
    return evidence, nonce


def test_open_evidence_carries_no_quote_and_is_accepted_only_when_allowed():
    evidence, nonce = open_evidence()
    assert evidence.tee == "open" and evidence.quote == "" and evidence.gpu_evidence is None

    verdict = verify_evidence(evidence, open_manifest(), nonce, allow_open=True)
    assert verdict.ok, verdict.reasons
    assert verdict.tier == OPEN and verdict.hardware == [] and verdict.gpu_count is None  # nothing about hardware is verified

    # A client about to seal a private job never passes allow_open, so open enclaves never verify for it.
    refused = verify_evidence(evidence, open_manifest(), nonce)
    assert not refused.ok and "confidential-tier evidence only" in refused.reasons[0]


@pytest.mark.parametrize(
    ("manifest", "profiles", "image", "reason"),
    [
        (GoldenManifest(), ["ltx-2.5-fast"], IMAGE, "does not allow the open tier"),
        (open_manifest(enabled=False), ["ltx-2.5-fast"], IMAGE, "does not allow the open tier"),
        (open_manifest(), ["ltx-2.5-fast"], "sha256:someone-elses-build", "not an approved open-tier image"),
        (open_manifest(), ["ltx-2.5-fast", "h3"], IMAGE, "not approved for all claimed profiles"),
    ],
)
def test_the_manifest_decides_which_open_tier_images_and_profiles_register(manifest, profiles, image, reason):
    evidence, nonce = open_evidence(profiles, image)
    verdict = verify_evidence(evidence, manifest, nonce, allow_open=True)
    assert not verdict.ok and any(reason in r for r in verdict.reasons)


def test_open_evidence_cannot_smuggle_a_quote_or_gpu_evidence_and_still_checks_nonce_and_age():
    evidence, nonce = open_evidence()
    with_quote = evidence.model_copy(update={"quote": b64e(b"not a quote")})
    assert "must not carry a quote" in " ".join(verify_evidence(with_quote, open_manifest(), nonce, allow_open=True).reasons)
    with_gpu = evidence.model_copy(update={"gpu_evidence": b64e(b"{}")})
    assert "must not carry GPU evidence" in " ".join(verify_evidence(with_gpu, open_manifest(), nonce, allow_open=True).reasons)
    assert not verify_evidence(evidence, open_manifest(), os.urandom(32), allow_open=True).ok
    assert not verify_evidence(evidence, open_manifest(), nonce, now=time.time() + 7200, allow_open=True).ok


def test_production_refuses_open_evidence_by_default_and_accepts_it_only_from_an_owner_enabled_manifest(tmp_path):
    owner = generate_signing_key()
    production = AttestationPolicy(production=True, quote_verifier=Accepting(), gpu_verifier=Accepting(), owner_public_key=public_key_bytes(owner))
    evidence, nonce = open_evidence()

    default = production.parse_manifest(sign_manifest(owner, GoldenManifest()).model_dump_json())
    assert default.open_tier is None
    assert not production.verify(evidence, default, nonce, allow_open=True).ok

    enabled = production.parse_manifest(sign_manifest(owner, open_manifest()).model_dump_json())
    assert production.verify(evidence, enabled, nonce, allow_open=True).ok
    assert not production.verify(evidence, enabled, nonce).ok  # still needs the caller to ask for the open tier


def test_manifests_signed_before_open_tier_existed_still_verify():
    owner = generate_signing_key()
    manifest = GoldenManifest(issued_at=1_700_000_000)
    legacy_body = manifest.model_dump(mode="json")
    legacy_body.pop("open_tier")
    legacy_message = b"kuno/v1/manifest\n" + canonical_json(legacy_body)
    assert manifest_message(manifest) == legacy_message
    signed = {"manifest": legacy_body, "signature": b64e(owner.sign(legacy_message))}
    import json

    assert parse_manifest(json.dumps(signed), public_key_bytes(owner), require_signature=True) == manifest
    # Adding the block changes the signed bytes, so it can't be slipped into a signed manifest.
    tampered = {"manifest": {**legacy_body, "open_tier": open_manifest().open_tier.model_dump(mode="json")}, "signature": signed["signature"]}
    with pytest.raises(ValueError, match="does not verify"):
        parse_manifest(json.dumps(tampered), public_key_bytes(owner))


def test_confidential_evidence_reports_its_tier_and_dev_manifests_allow_the_dev_image_on_the_open_tier(tmp_path):
    devkit.init(tmp_path)
    manifest = parse_manifest((tmp_path / "manifest.json").read_text())
    assert manifest.open_tier is not None and manifest.open_tier.refusal(devkit.DEV_IMAGE_DIGEST, ["ltx-2.5-fast"]) is None
    from kuno_protocol.attestation import MockTEE

    quote_key = signing_key_from_bytes(b64d((tmp_path / "mock_quote.key").read_text()))
    nonce = os.urandom(32)
    _, hpke = generate_hpke_keypair()
    mock = build_evidence(MockTEE(quote_key, devkit.DEV_IMAGE_DIGEST), nonce, hpke, public_key_bytes(generate_signing_key()), devkit.DEV_IMAGE_DIGEST, ["h3"])
    assert verify_evidence(mock, manifest, nonce).tier == CONFIDENTIAL


def test_the_open_tier_always_needs_a_hotkey_proof():
    assert hotkey_proof_required(OPEN, production=False) and hotkey_proof_required(OPEN, production=True)
    assert hotkey_proof_required(CONFIDENTIAL, production=True) and not hotkey_proof_required(CONFIDENTIAL, production=False)
