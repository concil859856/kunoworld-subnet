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

from .provenance import ProvenanceError

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
        self.configured_tsa_url = tsa_url
        self.tsa_url = tsa_url
        self._certificate: EnclaveCertificate | None = None
        self._lock = threading.Lock()

    @property
    def public_key(self) -> bytes:
        return public_key_bytes(self.signing_key)

    @property
    def certificate(self) -> EnclaveCertificate | None:
        return self._certificate

    def install(self, certificate: EnclaveCertificate | None, tsa_url: str | None = None) -> None:
        with self._lock:
            self._certificate = certificate
            # An operator's own TSA setting wins over the one the gateway suggests.
            self.tsa_url = self.configured_tsa_url or tsa_url

    @property
    def certificate_chain_pem(self) -> str:
        certificate = self._certificate
        if certificate is None:
            raise ProvenanceError("this enclave has no C2PA signing certificate from the gateway")
        if not certificate.usable():
            raise ProvenanceError("this enclave's C2PA signing certificate is expired or not yet valid")
        return certificate.chain_pem
