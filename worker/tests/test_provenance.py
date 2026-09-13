"""C2PA provenance: claims, certificates and the embed-then-sign order. Tests that
need the C2PA SDK skip when the worker's `provenance` extra is not installed."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from kuno_protocol.blobs import decrypt_blob, encrypt_blob
from kuno_protocol.canonical import sha256_hex
from kuno_protocol.crypto import generate_signing_key, public_key_bytes
from kuno_protocol.mp4 import probe
from kuno_protocol.receipts import ReceiptBody, VideoInfo, sign_receipt
from kuno_worker.backends.media_tools import ffmpeg_exe
from kuno_worker.provenance import (
    DOCUMENT_SIGNING_EKU,
    ProvenanceError,
    ProvenanceSigner,
    certificate_signing_request,
    issue_dev_certificate,
    provenance_claims,
    receipt_body_for_delivery,
    verify_claims,
)

ENCLAVE_ID = "e" * 32


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> bytes:
    out = Path(tmp_path_factory.mktemp("c2pa")) / "render.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "1", "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(out)],
        check=True, capture_output=True, timeout=60,
    )
    return out.read_bytes()


@pytest.fixture(scope="module")
def enclave():
    key = generate_signing_key()
    chain, root = issue_dev_certificate(key, ENCLAVE_ID)
    return ProvenanceSigner(key, chain), root


def draft_for(video: bytes) -> ReceiptBody:
    return ReceiptBody(
        job_id="0b7c1a52-8a3e-4f4a-9a53-2b1f8a1e7c11", enclave_id=ENCLAVE_ID, profile_id="ltx-2.5-fast",
        image_digest="sha256:kuno-worker-dev", params_digest="a" * 64, input_digest="b" * 64,
        output_digest="", output_bytes=0, content_digest=sha256_hex(video), attestation_digest="c" * 64,
        started_at=1.0, finished_at=2.0, gpu_seconds=1.0,
        video=VideoInfo(duration_s=1.0, width=160, height=90, fps=24, frames=24, audio=True), miner_hotkey="5Miner",
    )


# ---------------------------------------------------------------- without the SDK


def test_claims_are_signed_by_the_enclave_key(rendered, enclave):
    signer, _ = enclave
    claims = provenance_claims(rendered, draft_for(rendered), signer)
    assert verify_claims(claims)
    assert claims["rendered_digest"] == sha256_hex(rendered) and claims["model"]["family"] == "ltx-2.5"
    assert not verify_claims(dict(claims, job_id="someone-else"))


def test_a_draft_that_does_not_describe_the_bytes_is_refused(rendered, enclave):
    with pytest.raises(ProvenanceError):
        provenance_claims(rendered + b"x", draft_for(rendered), enclave[0])


def test_the_dev_certificate_is_a_c2pa_shaped_leaf_for_the_enclave_key(enclave):
    signer, root_pem = enclave
    leaf, root = x509.load_pem_x509_certificates(signer.certificate_chain_pem.encode())
    assert leaf.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == ENCLAVE_ID
    assert leaf.public_key().public_bytes_raw() == public_key_bytes(signer.signing_key)
    eku = leaf.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert DOCUMENT_SIGNING_EKU in eku and ExtendedKeyUsageOID.EMAIL_PROTECTION in eku
    assert not leaf.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    assert x509.load_pem_x509_certificate(root_pem.encode()) == root
    leaf.verify_directly_issued_by(root)


def test_a_csr_names_the_enclave(enclave):
    csr = x509.load_pem_x509_csr(certificate_signing_request(enclave[0].signing_key, ENCLAVE_ID).encode())
    assert csr.is_signature_valid and csr.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == ENCLAVE_ID


def test_the_delivery_body_hashes_the_final_file_and_the_sealed_blob(rendered):
    draft = draft_for(rendered)
    body = receipt_body_for_delivery(draft, b"final file", b"sealed blob")
    assert body.content_digest == sha256_hex(b"final file") and body.output_digest == sha256_hex(b"sealed blob")
    assert body.output_bytes == len(b"sealed blob") and body.job_id == draft.job_id


def test_embedding_without_the_sdk_is_a_clear_error(rendered, enclave, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_c2pa(name, *args, **kwargs):
        if name == "c2pa":
            raise ImportError("no c2pa")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_c2pa)
    from kuno_worker.provenance import embed_provenance

    with pytest.raises(ProvenanceError, match="provenance"):
        embed_provenance(rendered, draft_for(rendered), enclave[0])


# ---------------------------------------------------------------- with the SDK


@pytest.fixture(scope="module")
def c2pa():
    return pytest.importorskip("c2pa")


@pytest.fixture(scope="module")
def delivered(c2pa, rendered, enclave):
    """The worker's order: render, draft, embed, seal, finalize, sign."""
    from kuno_worker.provenance import embed_provenance

    signer, _ = enclave
    draft = draft_for(rendered)
    final = embed_provenance(rendered, draft, signer)
    output_key = b"\x07" * 32
    sealed = encrypt_blob(output_key, "job/output/video", final)
    receipt = sign_receipt(signer.signing_key, receipt_body_for_delivery(draft, final, sealed))
    return final, sealed, receipt, output_key


def test_the_delivered_file_verifies_end_to_end(delivered, enclave):
    from kuno_worker.provenance import read_provenance, verify_provenance

    final, sealed, receipt, output_key = delivered
    signer, root = enclave
    assert decrypt_blob(output_key, "job/output/video", sealed) == final
    assert sha256_hex(final) == receipt.body.content_digest  # the receipt binds the file the customer holds
    assert verify_provenance(final, receipt, signer.public_key) == []
    assert verify_provenance(final, receipt, signer.public_key, trust_anchors_pem=root) == []

    provenance = read_provenance(final)
    assert provenance.ok and not provenance.trusted and provenance.state == "Valid"
    assert provenance.status_codes == ["signingCredential.untrusted"]  # a dev certificate: intact but unidentified
    assert provenance.claims["attestation_digest"] == receipt.body.attestation_digest
    # The manifest names the rendered bytes; only the receipt can name the final file it lives in.
    assert provenance.claims["rendered_digest"] != receipt.body.content_digest
    assert read_provenance(final, trust_anchors_pem=root).trusted


def test_embedding_keeps_the_video_intact(delivered, rendered):
    before, after = probe(rendered), probe(delivered[0])
    assert (after.duration_s, after.width, after.height, after.frames, after.audio) == (
        before.duration_s, before.width, before.height, before.frames, before.audio
    )


def test_reading_does_not_change_the_file(delivered):
    from kuno_worker.provenance import read_provenance

    final = delivered[0]
    digest = sha256_hex(final)
    read_provenance(final)
    assert sha256_hex(final) == digest


def test_tampered_media_fails_validation(delivered, enclave):
    from kuno_worker.provenance import read_provenance, verify_provenance

    final, _, receipt, _ = delivered
    tampered = bytearray(final)
    tampered[final.find(b"mdat") + 64] ^= 0xFF
    assert not read_provenance(bytes(tampered)).ok
    assert verify_provenance(bytes(tampered), receipt, enclave[0].public_key)


def test_a_different_attested_key_is_detected(delivered):
    from kuno_worker.provenance import verify_provenance

    final, _, receipt, _ = delivered
    problems = verify_provenance(final, receipt, public_key_bytes(generate_signing_key()))
    assert "manifest was not signed by the attested enclave key" in problems


def test_a_file_without_a_manifest_reads_as_none(c2pa, rendered):
    from kuno_worker.provenance import read_provenance

    assert read_provenance(rendered) is None
