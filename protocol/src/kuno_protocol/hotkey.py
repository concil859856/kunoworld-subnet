"""Proof that a Bittensor hotkey vouches for an enclave.

The gateway pays and scores enclaves by miner hotkey, so a claimed hotkey must be
proven: the worker signs, with the hotkey's sr25519 key,

    "kuno/v1/hotkey-proof\\n" | canonical_json({"v":1, "hotkey", "nonce", "enclave_id", "signing_public_key"})

where `nonce` is the gateway-issued registration nonce (the same one bound into the
attestation quote). The signature therefore cannot be replayed for another enclave,
another registration, or another hotkey.

Signatures are plain Schnorrkel/sr25519 with the "substrate" signing context, which is
what `bittensor_wallet.Keypair.sign`, `btcli` and polkadot.js produce. A polkadot.js
`signRaw` signature over the `<Bytes>…</Bytes>`-wrapped message is also accepted, so an
operator can sign from a browser wallet.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Protocol

from pydantic import BaseModel

from .canonical import b64d, b64e, canonical_json

PROOF_CONTEXT = b"kuno/v1/hotkey-proof\n"
BITTENSOR_SS58_FORMAT = 42
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_POLKADOT_WRAP = (b"<Bytes>", b"</Bytes>")


class HotkeyError(ValueError):
    pass


def _b58decode(text: str) -> bytes:
    number = 0
    for char in text:
        index = _B58.find(char)
        if index < 0:
            raise HotkeyError("address is not base58")
        number = number * 58 + index
    body = number.to_bytes((number.bit_length() + 7) // 8, "big") if number else b""
    return b"\x00" * (len(text) - len(text.lstrip("1"))) + body


def _b58encode(data: bytes) -> str:
    number, out = int.from_bytes(data, "big"), ""
    while number:
        number, rem = divmod(number, 58)
        out = _B58[rem] + out
    return "1" * (len(data) - len(data.lstrip(b"\x00"))) + out


def _ss58_checksum(payload: bytes) -> bytes:
    return hashlib.blake2b(b"SS58PRE" + payload, digest_size=64).digest()[:2]


def ss58_decode(address: str, ss58_format: int | None = BITTENSOR_SS58_FORMAT) -> bytes:
    """Returns the 32-byte public key of an SS58 account address, checking its checksum and network prefix."""
    raw = _b58decode(address)
    if len(raw) == 35:
        prefix_len, prefix = 1, raw[0]
    elif len(raw) == 36 and raw[0] & 0x40:
        prefix_len = 2
        prefix = ((raw[0] & 0x3F) << 2) | (raw[1] >> 6) | ((raw[1] & 0x3F) << 8)
    else:
        raise HotkeyError("address is not a 32-byte SS58 account")
    if _ss58_checksum(raw[:-2]) != raw[-2:]:
        raise HotkeyError("address checksum mismatch")
    if ss58_format is not None and prefix != ss58_format:
        raise HotkeyError(f"address uses network prefix {prefix}, expected {ss58_format}")
    return raw[prefix_len:-2]


def ss58_encode(public_key: bytes, ss58_format: int = BITTENSOR_SS58_FORMAT) -> str:
    if len(public_key) != 32:
        raise HotkeyError("sr25519 public keys are 32 bytes")
    if ss58_format < 64:
        prefix = bytes([ss58_format])
    else:
        prefix = bytes([((ss58_format & 0xFC) >> 2) | 0x40, (ss58_format >> 8) | ((ss58_format & 0x03) << 6)])
    payload = prefix + public_key
    return _b58encode(payload + _ss58_checksum(payload))


class HotkeyProof(BaseModel):
    v: Literal[1] = 1
    crypto: Literal["sr25519"] = "sr25519"
    hotkey: str
    nonce: str
    enclave_id: str
    signing_public_key: str
    signature: str


def hotkey_proof_message(hotkey: str, nonce_hex: str, enclave_id: str, signing_public_key_b64: str) -> bytes:
    body = {
        "v": 1,
        "hotkey": hotkey,
        "nonce": nonce_hex,
        "enclave_id": enclave_id,
        "signing_public_key": signing_public_key_b64,
    }
    return PROOF_CONTEXT + canonical_json(body)


class HotkeySigner(Protocol):
    @property
    def ss58_address(self) -> str: ...

    def sign(self, message: bytes) -> bytes: ...


def _sr25519():
    try:
        import sr25519
    except ImportError as exc:  # pragma: no cover - dependency of kuno-protocol
        raise HotkeyError("sr25519 support needs the py-sr25519-bindings package") from exc
    return sr25519


class Sr25519Signer:
    """A hotkey held as a raw 32-byte mini-secret seed. Never printed, never logged."""

    def __init__(self, public_key: bytes, secret_key: bytes, ss58_format: int = BITTENSOR_SS58_FORMAT):
        self._public, self._secret, self._format = public_key, secret_key, ss58_format

    @classmethod
    def from_seed(cls, seed: bytes, ss58_format: int = BITTENSOR_SS58_FORMAT) -> Sr25519Signer:
        if len(seed) != 32:
            raise HotkeyError("an sr25519 seed is 32 bytes")
        public, secret = _sr25519().pair_from_seed(seed)
        return cls(public, secret, ss58_format)

    @property
    def public_key(self) -> bytes:
        return self._public

    @property
    def ss58_address(self) -> str:
        return ss58_encode(self._public, self._format)

    def sign(self, message: bytes) -> bytes:
        return _sr25519().sign((self._public, self._secret), message)

    def __repr__(self) -> str:
        return f"Sr25519Signer({self.ss58_address})"


def sign_hotkey_proof(signer: HotkeySigner, nonce: bytes, enclave_id: str, signing_public_key: bytes) -> HotkeyProof:
    hotkey, nonce_hex, sign_pk = signer.ss58_address, nonce.hex(), b64e(signing_public_key)
    signature = signer.sign(hotkey_proof_message(hotkey, nonce_hex, enclave_id, sign_pk))
    return HotkeyProof(
        hotkey=hotkey, nonce=nonce_hex, enclave_id=enclave_id, signing_public_key=sign_pk, signature=b64e(bytes(signature))
    )


def verify_hotkey_proof(
    proof: HotkeyProof,
    *,
    expected_nonce: bytes | str,
    enclave_id: str,
    signing_public_key: bytes | str,
    hotkey: str | None = None,
    ss58_format: int | None = BITTENSOR_SS58_FORMAT,
) -> tuple[bool, str]:
    """Checks a registration's hotkey proof against what the gateway itself knows. Returns (ok, detail).

    `expected_nonce`, `enclave_id` and `signing_public_key` must come from the verified
    attestation evidence, never from the proof, or the binding means nothing.
    """
    nonce_hex = expected_nonce.hex() if isinstance(expected_nonce, bytes) else expected_nonce.lower()
    sign_pk = b64e(signing_public_key) if isinstance(signing_public_key, bytes) else signing_public_key
    if hotkey is not None and proof.hotkey != hotkey:
        return False, "proof is for a different hotkey than the one claimed"
    if proof.nonce.lower() != nonce_hex:
        return False, "proof does not sign this registration nonce"
    if proof.enclave_id != enclave_id:
        return False, "proof is for a different enclave"
    if proof.signing_public_key != sign_pk:
        return False, "proof is for a different enclave signing key"
    try:
        public_key = ss58_decode(proof.hotkey, ss58_format)
        signature = b64d(proof.signature)
    except (HotkeyError, ValueError) as exc:
        return False, f"malformed proof: {exc}"
    if len(signature) != 64:
        return False, "sr25519 signatures are 64 bytes"
    message = hotkey_proof_message(proof.hotkey, proof.nonce, proof.enclave_id, proof.signing_public_key)
    sr = _sr25519()
    for candidate in (message, _POLKADOT_WRAP[0] + message + _POLKADOT_WRAP[1]):
        if sr.verify(signature, candidate, public_key):
            return True, "ok"
    return False, "signature does not verify for this hotkey"
