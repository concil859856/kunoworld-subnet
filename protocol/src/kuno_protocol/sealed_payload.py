"""Framing and size padding for the sealed job request: the HPKE plaintext that carries a `SealedPayload`.

    version 1 (unpadded):  plaintext = UTF-8 JSON of SealedPayload
    version 2 (padded):    plaintext = 0x02 | length:u32be | JSON | 0x00 …,
                           len(plaintext) = bucket(5 + length)

    bucket(n) = max(MIN_PADDED, the smallest power of two ≥ n), and never above MAX_PADDED

A version 1 plaintext is a JSON object, so its first byte is `{` or JSON whitespace; 0x02 can never start one, which
is what tells the two forms apart. Nothing else changes: `GenerationParams`, the AAD (`job_aad`) and the JSON inside
are byte for byte what they were, and the HPKE context, including its exported input and output keys, is the same.

Why powers of two and not PADMÉ (which pads blobs, see blobs.py). A request is a few kilobytes, and what its size
leaks is the prompt's length. PADMÉ's buckets are 32–128 bytes wide at that size, so a ciphertext would still pin a
prompt's length to within a sentence. Rounding up to a power of two from a 4 KiB floor leaves seven possible sizes
between 4 KiB and 256 KiB (under 3 bits), costs at most one extra request's worth of bytes, and makes almost every
real prompt, which fits in 4 KiB with its input manifest, the same size.

Senders always write version 2 (`seal_payload`). Receivers accept both (`open_payload`), and refuse a version 2
plaintext unless its declared length fits, its length is exactly the bucket for that length, and every padding byte
is zero, so a sender can't leak a length by padding differently.
"""

from __future__ import annotations

import struct

from .crypto import RecipientSession, SenderSession
from .schemas import SealedPayload

PAYLOAD_V1 = 1  # bare JSON: still opened, no longer written
PAYLOAD_V2 = 2  # padded to a power-of-two bucket
HEADER_LEN = 5  # version byte and u32 length
MIN_PADDED = 4 * 1024
MAX_PADDED = 256 * 1024
MAX_JSON_LEN = MAX_PADDED - HEADER_LEN
# The bytes a JSON object may start with: the brace or JSON whitespace.
_JSON_START = frozenset(b"{ \t\n\r")


class MalformedPayload(ValueError):
    """An authenticated request whose framing or padding is invalid."""


class PayloadTooLarge(ValueError):
    """The request's JSON does not fit in the largest padded size."""


def _bucket(framed_length: int) -> int:
    return max(MIN_PADDED, 1 << (framed_length - 1).bit_length())


def padded_payload_length(json_length: int) -> int:
    """Length of the version 2 plaintext for a JSON payload of this many bytes."""
    if json_length < 0:
        raise ValueError("length cannot be negative")
    if json_length > MAX_JSON_LEN:
        raise PayloadTooLarge(f"a sealed request is limited to {MAX_JSON_LEN} bytes of JSON")
    return _bucket(HEADER_LEN + json_length)


def pad_payload(payload_json: bytes) -> bytes:
    """The version 2 plaintext: version byte, JSON length, the JSON, then zeros up to its bucket."""
    total = padded_payload_length(len(payload_json))
    return bytes([PAYLOAD_V2]) + struct.pack(">I", len(payload_json)) + payload_json + bytes(total - HEADER_LEN - len(payload_json))


def payload_version(plaintext: bytes) -> int | None:
    """The form a decrypted request is in, or None if it is neither."""
    if not plaintext:
        return None
    if plaintext[0] == PAYLOAD_V2:
        return PAYLOAD_V2
    return PAYLOAD_V1 if plaintext[0] in _JSON_START else None


def unpad_payload(plaintext: bytes) -> bytes:
    """The JSON inside a decrypted request of either form. Refuses unknown framing, a length that does not fit, a
    plaintext that is not exactly its bucket, and non-zero padding."""
    version = payload_version(plaintext)
    if version == PAYLOAD_V1:
        return plaintext
    if version is None:
        raise MalformedPayload("the sealed request has an unknown framing")
    if len(plaintext) < HEADER_LEN:
        raise MalformedPayload("the padded request is too short to hold its length")
    if len(plaintext) > MAX_PADDED:
        raise MalformedPayload("the padded request is larger than the maximum padded size")
    (length,) = struct.unpack_from(">I", plaintext, 1)
    end = HEADER_LEN + length
    if end > len(plaintext):
        raise MalformedPayload("the padded request declares more JSON than it holds")
    if _bucket(end) != len(plaintext):
        raise MalformedPayload("the padded request is not padded to its size bucket")
    if plaintext.count(0, end) != len(plaintext) - end:
        raise MalformedPayload("the padded request has non-zero padding")
    return plaintext[HEADER_LEN:end]


def seal_payload(session: SenderSession, payload: SealedPayload, aad: bytes) -> bytes:
    """Seals a request in the padded form. Raises PayloadTooLarge before the session is used."""
    return session.seal(pad_payload(payload.model_dump_json().encode()), aad)


def open_payload(session: RecipientSession, ciphertext: bytes, aad: bytes) -> SealedPayload:
    """Opens a request in either form. Raises DecryptionError if it fails authentication, MalformedPayload if its
    framing is invalid, and pydantic's ValidationError if the JSON is not a SealedPayload."""
    return SealedPayload.model_validate_json(unpad_payload(session.open(ciphertext, aad)))
