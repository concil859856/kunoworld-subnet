"""The enclave's C2PA signing certificate, as issued by the gateway's attestation-gated CA.

`CertifiedSigner` stands in for a `ProvenanceSigner` whose certificate the worker swaps as
the gateway reissues it. Reading `certificate_chain_pem` without a usable certificate raises
`ProvenanceError`, which fails the job (`internal_error`) instead of shipping a video with
untrusted or missing provenance.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.x509.oid import NameOID

from kuno_protocol.crypto import public_key_bytes

from .provenance import ProvenanceError, parse_tsa_urls

GATEWAY, DEV, PROVISIONAL = "gateway", "dev", "provisional"


@dataclass(frozen=True)
class EnclaveCertificate:
    chain_pem: str
    not_before: float
    not_after: float
    # gateway: issued by the gateway CA; dev: throwaway, because a mock-TEE gateway has no CA;
    # provisional: throwaway, installed before the first registration has asked the gateway.
    source: str

    @classmethod
    def parse(cls, chain_pem: str, signing_public_key: bytes, enclave_id: str, source: str) -> EnclaveCertificate:
        """Checks the leaf really is for this enclave's key and id before it is used."""
        certs = x509.load_pem_x509_certificates(chain_pem.encode())
        leaf = certs[0]
        key = leaf.public_key()
        if not isinstance(key, ed25519.Ed25519PublicKey) or key.public_bytes_raw() != signing_public_key:
            raise ValueError("the certificate is not for this enclave's signing key")
        if [a.value for a in leaf.subject.get_attributes_for_oid(NameOID.COMMON_NAME)] != [enclave_id]:
            raise ValueError("the certificate names a different enclave")
        return cls(chain_pem, leaf.not_valid_before_utc.timestamp(), leaf.not_valid_after_utc.timestamp(), source)

    def usable(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.not_before <= now < self.not_after

    def refresh_at(self, margin_s: float) -> float:
        """Renew once less than a third of the lifetime, or `margin_s`, remains."""
        return self.not_after - max((self.not_after - self.not_before) / 3, margin_s)


class CertifiedSigner:
    def __init__(self, signing_key: ed25519.Ed25519PrivateKey, tsa_url: str | None = None):
        self.signing_key = signing_key
        # The operator's own TSAs (KUNO_PROVENANCE_TSA_URL: one URL, or several separated by commas, in order) replace
        # the gateway's suggestions.
        self.configured_tsa_urls = parse_tsa_urls(tsa_url)
        # The TSAs signing tries, in order (provenance.sign_with_tsa_failover).
        self.tsa_urls: list[str] = list(self.configured_tsa_urls)
        self._certificate: EnclaveCertificate | None = None
        self._lock = threading.Lock()

    @property
    def configured_tsa_url(self) -> str | None:
        return self.configured_tsa_urls[0] if self.configured_tsa_urls else None

    @property
    def tsa_url(self) -> str | None:
        """The first TSA signing tries."""
        return self.tsa_urls[0] if self.tsa_urls else None

    @tsa_url.setter
    def tsa_url(self, value: str | None) -> None:
        self.tsa_urls = parse_tsa_urls(value)

    def effective_tsa_urls(self, suggested=None) -> list[str]:
        """The TSAs this signer uses given the gateway's suggestion: its `tsa_urls` list, or an older gateway's single
        `tsa_url`. The operator's setting wins."""
        return list(self.configured_tsa_urls) or parse_tsa_urls(suggested)

    @property
    def public_key(self) -> bytes:
        return public_key_bytes(self.signing_key)

    @property
    def certificate(self) -> EnclaveCertificate | None:
        return self._certificate

    def install(self, certificate: EnclaveCertificate | None, tsa_urls=None) -> None:
        """`tsa_urls`: the gateway's suggestion, a list or one URL. An operator's own TSA setting wins over it."""
        with self._lock:
            self._certificate = certificate
            self.tsa_urls = self.effective_tsa_urls(tsa_urls)

    @property
    def certificate_chain_pem(self) -> str:
        certificate = self._certificate
        if certificate is None:
            raise ProvenanceError("this enclave has no C2PA signing certificate from the gateway")
        if not certificate.usable():
            raise ProvenanceError("this enclave's C2PA signing certificate is expired or not yet valid")
        return certificate.chain_pem
