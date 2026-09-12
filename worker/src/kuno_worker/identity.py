"""Keys that exist only in this VM's memory. They are generated at boot, bound
into every attestation quote, and die with the process."""

from __future__ import annotations

from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kuno_protocol.attestation import enclave_id_for
from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key, public_key_bytes


@dataclass(frozen=True)
class EnclaveIdentity:
    hpke_private: bytes
    hpke_public: bytes
    signing_key: Ed25519PrivateKey

    @classmethod
    def generate(cls) -> EnclaveIdentity:
        private, public = generate_hpke_keypair()
        return cls(private, public, generate_signing_key())

    @property
    def signing_public(self) -> bytes:
        return public_key_bytes(self.signing_key)

    @property
    def enclave_id(self) -> str:
        return enclave_id_for(self.hpke_public, self.signing_public)
