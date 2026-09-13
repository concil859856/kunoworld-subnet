"""C2PA provenance embedded in the delivered MP4.

The receipt proves where a video came from, but it travels separately. A C2PA
manifest travels inside the file, so any C2PA-aware tool can show that the clip
is AI-generated, by which model, on which attested enclave. Uses the Content
Authenticity Initiative's `c2pa-python` (the worker's `provenance` extra; tested
with 0.37.10, which embeds into MP4 with a `c2pa.hash.bmff.v3` hard binding).

What the manifest carries (assertion `com.kunoworld.provenance`): job, enclave,
miner, profile and model, image digest, params and input digests, the attestation
digest, `rendered_digest`, and an Ed25519 signature over all of that by the enclave's
attested signing key. The C2PA claim itself is signed with the same enclave key
under an X.509 certificate issued for it.

Order of operations (the chicken-and-egg problem)
-------------------------------------------------
A receipt's `content_digest` is, by protocol, the SHA-256 of the exact file the
customer receives. Embedding a manifest changes the file, so the manifest cannot
contain the final digest of the file it is part of. We therefore embed first and
sign the receipt last:

  1. the backend renders MP4 bytes R;
  2. the worker drafts the receipt body with content_digest = SHA-256(R);
  3. F = embed_provenance(R, draft, signer); the manifest records SHA-256(R) as
     `rendered_digest` and refuses a draft whose content_digest is anything else;
  4. the worker seals F for the customer;
  5. body = receipt_body_for_delivery(draft, F, sealed): content_digest = SHA-256(F),
     output_digest and output_bytes from the sealed blob;
  6. the enclave signs that body.

So the receipt binds the final file (plain SHA-256, unchanged protocol, still found
at /v1/provenance/{sha256}), and the manifest binds the media through C2PA's own
BMFF hash, which excludes the manifest box. Both are signed by the same attested key.
We rejected publishing an exclusion-range digest as `content_digest`: it would change
the receipt format and break every existing lookup by plain SHA-256.

Certificates
------------
`issue_dev_certificate` creates a throwaway root CA and a leaf for the enclave key.
Readers report such a file as `Valid` (signature and hashes check out) with the status
`signingCredential.untrusted`: it proves integrity and key possession, not identity.
Production needs a certificate chain to a CA on the C2PA trust list. The intended
design: the enclave sends `certificate_signing_request()` to a KunoWorld issuing CA
that checks the enclave's attestation first; the CA's root is submitted to the C2PA
trust list. Neither the issuing service nor trust-list membership exists yet.
"""

from __future__ import annotations

import datetime
import io
import json
from dataclasses import dataclass, field
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID, ObjectIdentifier

from kuno_protocol.canonical import b64d, b64e, canonical_json, sha256_hex
from kuno_protocol.crypto import public_key_bytes, verify_signature
from kuno_protocol.profiles import load_profiles
from kuno_protocol.receipts import Receipt, ReceiptBody, verify_receipt

KUNO_ASSERTION = "com.kunoworld.provenance"
TRAINED_ALGORITHMIC_MEDIA = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
PROVENANCE_CONTEXT = b"kuno/v1/provenance\n"
DOCUMENT_SIGNING_EKU = ObjectIdentifier("1.3.6.1.5.5.7.3.36")
UNTRUSTED_CREDENTIAL = "signingCredential.untrusted"


class ProvenanceError(Exception):
    """Provenance could not be embedded or read."""


@dataclass
class ProvenanceSigner:
    """The enclave's attested Ed25519 key plus a certificate chain (leaf first) issued for it."""

    signing_key: Ed25519PrivateKey
    certificate_chain_pem: str
    tsa_url: str | None = None

    @property
    def public_key(self) -> bytes:
        return public_key_bytes(self.signing_key)


def _c2pa():
    try:
        import c2pa  # noqa: PLC0415 — optional extra
    except ImportError as exc:
        raise ProvenanceError("C2PA support needs the worker's `provenance` extra (c2pa-python)") from exc
    return c2pa


# ---------------------------------------------------------------- claims


def provenance_claims(mp4_bytes: bytes, receipt: ReceiptBody | Receipt, signer: ProvenanceSigner) -> dict[str, Any]:
    """The KunoWorld assertion, signed by the enclave key so it verifies without X.509."""
    body = receipt.body if isinstance(receipt, Receipt) else receipt
    rendered = sha256_hex(mp4_bytes)
    if body.content_digest != rendered:
        raise ProvenanceError("the draft receipt must describe these bytes (content_digest = SHA-256 of the rendered MP4)")
    profile = load_profiles().get(body.profile_id)
    claims: dict[str, Any] = {
        "v": 1,
        "job_id": body.job_id,
        "enclave_id": body.enclave_id,
        "miner_hotkey": body.miner_hotkey,
        "profile_id": body.profile_id,
        "model": (
            {"family": profile.family, "name": profile.name, "checkpoint": profile.checkpoint, "license": profile.license.name}
            if profile is not None
            else None
        ),
        "image_digest": body.image_digest,
        "params_digest": body.params_digest,
        "input_digest": body.input_digest,
        "attestation_digest": body.attestation_digest,
        "rendered_digest": rendered,
        "enclave_signing_public_key": b64e(signer.public_key),
    }
    claims["enclave_signature"] = b64e(signer.signing_key.sign(PROVENANCE_CONTEXT + canonical_json(claims)))
    return claims


def verify_claims(claims: dict[str, Any]) -> bool:
    """Checks the enclave signature inside a KunoWorld assertion."""
    try:
        unsigned = {k: v for k, v in claims.items() if k != "enclave_signature"}
        return verify_signature(
            b64d(claims["enclave_signing_public_key"]), b64d(claims["enclave_signature"]), PROVENANCE_CONTEXT + canonical_json(unsigned)
        )
    except (KeyError, TypeError, ValueError):
        return False


def manifest_definition(claims: dict[str, Any]) -> dict[str, Any]:
    from . import __version__  # noqa: PLC0415

    model = claims.get("model") or {}
    return {
        "claim_generator_info": [{"name": "kunoworld-worker", "version": __version__}],
        "title": f"{claims['job_id']}.mp4",
        "format": "video/mp4",
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": TRAINED_ALGORITHMIC_MEDIA,
                            "softwareAgent": {"name": model.get("name") or claims["profile_id"]},
                        }
                    ]
                },
            },
            {"label": KUNO_ASSERTION, "data": claims},
        ],
    }


# ---------------------------------------------------------------- embed and read


def embed_provenance(mp4_bytes: bytes, receipt: ReceiptBody | Receipt, signer: ProvenanceSigner) -> bytes:
    """Returns the MP4 with a signed C2PA manifest. `receipt` is the draft body (see module docs)."""
    c2pa = _c2pa()
    claims = provenance_claims(mp4_bytes, receipt, signer)
    destination = io.BytesIO()
    try:
        c2pa_signer = c2pa.Signer.from_callback(
            lambda data: signer.signing_key.sign(bytes(data)), c2pa.C2paSigningAlg.ED25519, signer.certificate_chain_pem, signer.tsa_url
        )
        with c2pa.Builder(manifest_definition(claims)) as builder:
            builder.sign(c2pa_signer, "video/mp4", io.BytesIO(mp4_bytes), destination)
    except ProvenanceError:
        raise
    except Exception as exc:
        raise ProvenanceError(f"C2PA signing failed ({type(exc).__name__})") from None
    return destination.getvalue()


@dataclass
class Provenance:
    state: str
    status_codes: list[str]
    claims: dict[str, Any] | None
    signer: dict[str, Any] = field(default_factory=dict)
    claims_signature_ok: bool = False

    @property
    def trusted(self) -> bool:
        return self.state == "Trusted"

    @property
    def ok(self) -> bool:
        """Intact and signed; `trusted` additionally requires a chain to a configured trust anchor."""
        failures = [code for code in self.status_codes if code != UNTRUSTED_CREDENTIAL]
        return self.state in ("Valid", "Trusted") and not failures and self.claims is not None and self.claims_signature_ok


def read_provenance(mp4_bytes: bytes, trust_anchors_pem: str | None = None) -> Provenance | None:
    """Reads and validates the embedded manifest; None when the file carries none."""
    c2pa = _c2pa()
    kwargs = {}
    if trust_anchors_pem:
        kwargs["context"] = c2pa.Context.from_dict({"trust": {"trust_anchors": trust_anchors_pem}, "verify": {"verify_trust": True}})
    try:
        reader = c2pa.Reader("video/mp4", io.BytesIO(mp4_bytes), **kwargs)
    except Exception as exc:
        if "ManifestNotFound" in type(exc).__name__:
            return None
        raise ProvenanceError(f"C2PA manifest could not be read ({type(exc).__name__})") from None
    try:
        report = json.loads(reader.json())
    finally:
        reader.close()
    manifest = report.get("manifests", {}).get(report.get("active_manifest"), {})
    claims = next((a.get("data") for a in manifest.get("assertions", []) if a.get("label") == KUNO_ASSERTION), None)
    return Provenance(
        state=report.get("validation_state", "Invalid"),
        status_codes=[s.get("code", "") for s in report.get("validation_status", [])],
        claims=claims,
        signer=manifest.get("signature_info", {}),
        claims_signature_ok=claims is not None and verify_claims(claims),
    )


def verify_provenance(
    mp4_bytes: bytes, receipt: Receipt, attested_signing_key: bytes, trust_anchors_pem: str | None = None
) -> list[str]:
    """Everything a customer or validator checks; returns the problems found (empty means verified)."""
    provenance = read_provenance(mp4_bytes, trust_anchors_pem)
    if provenance is None:
        return ["the file carries no C2PA manifest"]
    problems: list[str] = []
    if not provenance.ok:
        problems.append(f"C2PA validation failed: {provenance.state} {provenance.status_codes}")
    if trust_anchors_pem and not provenance.trusted:
        problems.append("signing certificate does not chain to the trust anchors")
    claims = provenance.claims or {}
    body = receipt.body
    if not verify_receipt(receipt, attested_signing_key):
        problems.append("receipt signature does not verify against the attested key")
    if claims.get("enclave_signing_public_key") != b64e(attested_signing_key):
        problems.append("manifest was not signed by the attested enclave key")
    if sha256_hex(mp4_bytes) != body.content_digest:
        problems.append("file does not match the receipt's content digest")
    for name in ("job_id", "enclave_id", "profile_id", "attestation_digest", "image_digest", "params_digest", "input_digest"):
        if claims.get(name) != getattr(body, name):
            problems.append(f"manifest {name} does not match the receipt")
    if provenance.signer.get("common_name") not in (None, body.enclave_id):
        problems.append("signing certificate was issued for a different enclave")
    return problems


def receipt_body_for_delivery(draft: ReceiptBody, final_mp4: bytes, sealed: bytes) -> ReceiptBody:
    """Step 5: the body to sign once the manifest is embedded and the file sealed."""
    return draft.model_copy(update={"content_digest": sha256_hex(final_mp4), "output_digest": sha256_hex(sealed), "output_bytes": len(sealed)})


# ---------------------------------------------------------------- certificates


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name), x509.NameAttribute(NameOID.ORGANIZATION_NAME, "KunoWorld")])


def issue_dev_certificate(signing_key: Ed25519PrivateKey, enclave_id: str, days: int = 30) -> tuple[str, str]:
    """(chain_pem, root_pem) from a throwaway root CA. Development only: readers report it untrusted.

    C2PA rejects a self-signed end-entity certificate and a leaf without an extended key
    usage, so this issues a CA and a leaf with digitalSignature and documentSigning.
    """
    from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: PLC0415

    now = datetime.datetime.now(datetime.timezone.utc)
    root_key = ed25519.Ed25519PrivateKey.generate()
    root = (
        x509.CertificateBuilder()
        .subject_name(_name("KunoWorld development C2PA root (untrusted)"))
        .issuer_name(_name("KunoWorld development C2PA root (untrusted)"))
        .public_key(root_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(root_key.public_key()), critical=False)
        .sign(root_key, None)
    )
    leaf = (
        x509.CertificateBuilder()
        .subject_name(_name(enclave_id))
        .issuer_name(root.subject)
        .public_key(signing_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([DOCUMENT_SIGNING_EKU, ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(signing_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), critical=False)
        .sign(root_key, None)
    )
    pem = lambda cert: cert.public_bytes(serialization.Encoding.PEM).decode()  # noqa: E731
    return pem(leaf) + pem(root), pem(root)


def certificate_signing_request(signing_key: Ed25519PrivateKey, enclave_id: str) -> str:
    """A CSR for the enclave key, for an attestation-gated issuing CA (production path)."""
    csr = x509.CertificateSigningRequestBuilder().subject_name(_name(enclave_id)).sign(signing_key, None)
    return csr.public_bytes(serialization.Encoding.PEM).decode()
