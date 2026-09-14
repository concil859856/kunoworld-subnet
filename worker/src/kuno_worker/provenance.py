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
After each attestation the worker sends `certificate_signing_request()` to the gateway's
issuing CA, which signs a short-lived leaf only for a freshly attested enclave's own key
(worker.py `_refresh_certificate`, gateway `ca.py`). Readers given the KunoWorld root from
`GET /v1/c2pa/trust` as a trust anchor report such a file as `Trusted`. See subnet/PROVENANCE.md.

`issue_dev_certificate` creates a throwaway root CA and a leaf for the enclave key; a
mock-TEE worker uses it only when the gateway has no CA. Readers report such a file as
`Valid` with the status `signingCredential.untrusted`: integrity and key possession, not
identity. The KunoWorld root is not on the C2PA Trust List yet.

Timestamps and failover
-----------------------
The manifest is timestamped by an RFC 3161 timestamp authority so it outlives the short
certificate. The gateway lists its TSAs in order of preference (`tsa_urls`, with `tsa_url` its
first for older workers); an operator's `KUNO_PROVENANCE_TSA_URL`, one URL or a comma-separated
list, replaces that list. The C2PA SDK sends the timestamp request itself, so failover wraps
the whole signing step (`sign_with_tsa_failover`):

* TSAs are tried in order, each within `TSA_TIMEOUT_S` (plus `TSA_TIMEOUT_S_PER_MIB` for hashing
  the video). The SDK has no timeout of its own to set, so an attempt runs on its own thread; one
  that never returns is left behind on a daemon thread.
* A TSA that failed is remembered for `TSA_COOLDOWN_S` (`TsaBreaker`) and tried after the others,
  so the next videos don't wait on it. It is moved to the back, never skipped.
* When every TSA fails, signing fails and the job fails: a video is never shipped with a manifest
  that would stop validating when its certificate expires.
"""

from __future__ import annotations

import datetime
import io
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Iterable
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

# Timestamp authority failover (module docstring, "Timestamps and failover").
TSA_TIMEOUT_S = 10.0
TSA_TIMEOUT_S_PER_MIB = 0.05
TSA_COOLDOWN_S = 120.0

log = logging.getLogger("kuno.provenance")


class ProvenanceError(Exception):
    """Provenance could not be embedded or read."""


def parse_tsa_urls(value: str | Iterable[str] | None) -> list[str]:
    """TSA URLs in order, without repeats, from None, one URL, a comma- or space-separated string, or a list. Entries
    that aren't http(s) URLs are dropped: the C2PA SDK refuses them."""
    if value is None:
        return []
    items = re.split(r"[\s,]+", value) if isinstance(value, str) else [v for v in value if isinstance(v, str)]
    urls = []
    for item in (item.strip() for item in items):
        if not item:
            continue
        if not item.lower().startswith(("http://", "https://")):
            log.warning("ignoring timestamp authority %r: not an http(s) URL", item)
            continue
        urls.append(item)
    return list(dict.fromkeys(urls))


@dataclass
class ProvenanceSigner:
    """The enclave's attested Ed25519 key plus a certificate chain (leaf first) issued for it."""

    signing_key: Ed25519PrivateKey
    certificate_chain_pem: str
    # One TSA URL, or several separated by commas, tried in order.
    tsa_url: str | None = None

    @property
    def public_key(self) -> bytes:
        return public_key_bytes(self.signing_key)

    @property
    def tsa_urls(self) -> list[str]:
        return parse_tsa_urls(self.tsa_url)


class TsaBreaker:
    """Remembers for `cooldown_s` which timestamp authorities just failed, so the next videos try the others first. A
    failed TSA moves to the back of the order and is never skipped: when all have failed lately, they are tried oldest
    failure first."""

    def __init__(self, cooldown_s: float = TSA_COOLDOWN_S, clock: Callable[[], float] = time.monotonic):
        self.cooldown_s, self.clock = cooldown_s, clock
        self._failed: dict[str, float] = {}
        self._lock = threading.Lock()

    def order(self, urls: list[str]) -> list[str]:
        now = self.clock()
        with self._lock:
            for url, failed_at in list(self._failed.items()):
                if now - failed_at >= self.cooldown_s:
                    del self._failed[url]
            failed = dict(self._failed)
        healthy = [url for url in urls if url not in failed]
        return healthy + sorted((url for url in urls if url in failed), key=failed.__getitem__)

    def record_failure(self, url: str) -> None:
        with self._lock:
            self._failed[url] = self.clock()

    def record_success(self, url: str) -> None:
        with self._lock:
            self._failed.pop(url, None)


TSA_BREAKER = TsaBreaker()


class _NoAnswer(Exception):
    pass


def _within(fn: Callable[[], bytes], timeout_s: float) -> bytes:
    """Runs `fn` on its own thread and waits at most `timeout_s` for it. The C2PA SDK makes the timestamp request itself
    and has no timeout to set, so an attempt still waiting on a TSA is left behind on a daemon thread."""
    outcome: dict[str, Any] = {}
    done = threading.Event()

    def run() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # handed to the waiting caller
            outcome["error"] = exc
        finally:
            done.set()

    threading.Thread(target=run, name="c2pa-sign", daemon=True).start()
    if not done.wait(timeout_s):
        raise _NoAnswer()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def sign_with_tsa_failover(
    attempt: Callable[[str | None], bytes], urls: list[str], *, timeout_s: float = TSA_TIMEOUT_S, breaker: TsaBreaker | None = None,
) -> bytes:
    """`attempt(tsa_url)` signs once with that TSA. Tries the TSAs in order, recently failed ones last, each within
    `timeout_s`, until one signs. With no TSA it signs once without a timestamp (development only: real-TEE workers
    refuse a certificate without one). Raises ProvenanceError when every TSA fails."""
    breaker = breaker or TSA_BREAKER
    if not urls:
        return attempt(None)
    failures: list[str] = []
    for url in breaker.order(urls):
        try:
            signed = _within(lambda url=url: attempt(url), timeout_s)
        except _NoAnswer:
            failures.append(f"{url}: no answer within {timeout_s:.0f}s")
        except Exception as exc:
            failures.append(f"{url}: {type(exc).__name__}")
        else:
            breaker.record_success(url)
            return signed
        breaker.record_failure(url)
        log.warning("C2PA signing with timestamp authority %s failed", failures[-1])
    raise ProvenanceError("C2PA signing failed with every timestamp authority (" + "; ".join(failures) + ")")


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


def embed_provenance(
    mp4_bytes: bytes, receipt: ReceiptBody | Receipt, signer: ProvenanceSigner, *, breaker: TsaBreaker | None = None,
) -> bytes:
    """Returns the MP4 with a signed C2PA manifest. `receipt` is the draft body (see module docs). Fails over between
    the signer's timestamp authorities (`sign_with_tsa_failover`)."""
    c2pa = _c2pa()
    claims = provenance_claims(mp4_bytes, receipt, signer)
    definition = manifest_definition(claims)
    # Read once, before any TSA is tried: a missing or expired certificate is not a TSA failure.
    chain = signer.certificate_chain_pem
    urls = list(getattr(signer, "tsa_urls", None) or [])

    def attempt(tsa_url: str | None) -> bytes:
        destination = io.BytesIO()
        c2pa_signer = c2pa.Signer.from_callback(
            lambda data: signer.signing_key.sign(bytes(data)), c2pa.C2paSigningAlg.ED25519, chain, tsa_url
        )
        with c2pa.Builder(definition) as builder:
            builder.sign(c2pa_signer, "video/mp4", io.BytesIO(mp4_bytes), destination)
        return destination.getvalue()

    timeout_s = TSA_TIMEOUT_S + TSA_TIMEOUT_S_PER_MIB * len(mp4_bytes) / (1 << 20)
    try:
        return sign_with_tsa_failover(attempt, urls, timeout_s=timeout_s, breaker=breaker)
    except ProvenanceError:
        raise
    except Exception as exc:
        raise ProvenanceError(f"C2PA signing failed ({type(exc).__name__})") from None


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
