"""End-to-end encryption between a client and an attested enclave.

A job is sealed with HPKE (RFC 9180, base mode) to the enclave's attested X25519
key: DHKEM(X25519, HKDF-SHA256) / HKDF-SHA256 / ChaCha20-Poly1305.

The same HPKE context exports two 32-byte keys that both sides derive without an
extra round trip:
  * the input key encrypts reference media the client uploads as blobs;
  * the output key encrypts the finished video, so only the client can open it.

Ed25519 helpers for enclave signatures (receipts, request auth) live here too.
"""

from __future__ import annotations

import os

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pyhpke import AEADId, CipherSuite, KDFId, KEMId

SUITE = CipherSuite.new(KEMId.DHKEM_X25519_HKDF_SHA256, KDFId.HKDF_SHA256, AEADId.CHACHA20_POLY1305)
HPKE_INFO = b"kuno/v1/job"
_EXPORT_INPUT = b"kuno/v1/input-key"
_EXPORT_OUTPUT = b"kuno/v1/output-key"


class DecryptionError(Exception):
    """Ciphertext failed authentication or was malformed."""


def generate_hpke_keypair() -> tuple[bytes, bytes]:
    """Returns (private_key, public_key), 32 raw bytes each."""
    pair = SUITE.kem.derive_key_pair(os.urandom(32))
    return pair.private_key.to_private_bytes(), pair.public_key.to_public_bytes()


class SenderSession:
    """Client side of one job. Export keys first, upload blobs, then seal once."""

    def __init__(self, enclave_public_key: bytes):
        public_key = SUITE.kem.deserialize_public_key(enclave_public_key)
        self.enc, self._ctx = SUITE.create_sender_context(public_key, info=HPKE_INFO)
        self.input_key = self._ctx.export(_EXPORT_INPUT, 32)
        self.output_key = self._ctx.export(_EXPORT_OUTPUT, 32)
        self._sealed = False

    def seal(self, plaintext: bytes, aad: bytes) -> bytes:
        if self._sealed:
            raise RuntimeError("a sender session seals exactly one request")
        self._sealed = True
        return self._ctx.seal(plaintext, aad=aad)


class RecipientSession:
    """Enclave side of one job."""

    def __init__(self, enclave_private_key: bytes, enc: bytes):
        private_key = SUITE.kem.deserialize_private_key(enclave_private_key)
        try:
            self._ctx = SUITE.create_recipient_context(enc, private_key, info=HPKE_INFO)
        except Exception as exc:  # pyhpke raises several types for a bad encapsulation
            raise DecryptionError("invalid encapsulated key") from exc
        self.input_key = self._ctx.export(_EXPORT_INPUT, 32)
        self.output_key = self._ctx.export(_EXPORT_OUTPUT, 32)

    def open(self, ciphertext: bytes, aad: bytes) -> bytes:
        try:
            return self._ctx.open(ciphertext, aad=aad)
        except Exception as exc:
            raise DecryptionError("request failed authentication") from exc


def generate_signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def signing_key_from_bytes(raw: bytes) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(raw)


def signing_key_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())


def public_key_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def verify_signature(public_key: bytes, signature: bytes, message: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
        return True
    except (InvalidSignature, ValueError):
        return False


def request_signature_message(method: str, path: str, timestamp: str, body: bytes) -> bytes:
    """What an enclave signs to authenticate an HTTP request to the gateway."""
    import hashlib

    return b"\n".join(
        [b"kuno/v1/request", method.upper().encode(), path.encode(), timestamp.encode(), hashlib.sha256(body).hexdigest().encode()]
    )
