"""Chunked authenticated encryption for media blobs (inputs and output video).

Format (all integers big-endian):

    header  = "KUNOB1" | version:u8 | chunk_size:u32 | nonce_prefix:7 bytes   (18 bytes)
    body    = chunk_0 || chunk_1 || ... ; each chunk = ChaCha20-Poly1305(plaintext) + 16-byte tag
    nonce_i = nonce_prefix | i:u32 | final:u8

The key is HKDF-SHA256(base_key, info="kuno/v1/blob/" + label). The label binds a
blob to its job and role (e.g. "<job>/input/0"), so blobs cannot be swapped
between jobs or roles. The header is the AAD of every chunk, and the final-chunk
flag makes truncation at a chunk boundary detectable.
"""

from __future__ import annotations

import os
import struct

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .crypto import DecryptionError

MAGIC = b"KUNOB1"
VERSION = 1
DEFAULT_CHUNK = 1 << 20
MAX_CHUNK = 64 << 20
TAG_LEN = 16
_HEADER = struct.Struct(">6sBI7s")


def _blob_key(base_key: bytes, label: str) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"kuno/v1/blob/" + label.encode()).derive(base_key)


def _nonce(prefix: bytes, index: int, final: bool) -> bytes:
    return prefix + struct.pack(">I", index) + (b"\x01" if final else b"\x00")


def encrypt_blob(base_key: bytes, label: str, plaintext: bytes, chunk_size: int = DEFAULT_CHUNK) -> bytes:
    if not 0 < chunk_size <= MAX_CHUNK:
        raise ValueError("chunk_size out of range")
    aead = ChaCha20Poly1305(_blob_key(base_key, label))
    prefix = os.urandom(7)
    header = _HEADER.pack(MAGIC, VERSION, chunk_size, prefix)
    chunks = [plaintext[i : i + chunk_size] for i in range(0, len(plaintext), chunk_size)] or [b""]
    last = len(chunks) - 1
    return header + b"".join(aead.encrypt(_nonce(prefix, i, i == last), chunk, header) for i, chunk in enumerate(chunks))


def decrypt_blob(base_key: bytes, label: str, blob: bytes) -> bytes:
    if len(blob) < _HEADER.size + TAG_LEN:
        raise DecryptionError("blob too short")
    magic, version, chunk_size, prefix = _HEADER.unpack_from(blob)
    if magic != MAGIC or version != VERSION or not 0 < chunk_size <= MAX_CHUNK:
        raise DecryptionError("not a KunoWorld blob")
    header, body = blob[: _HEADER.size], blob[_HEADER.size :]
    step = chunk_size + TAG_LEN
    pieces = [body[i : i + step] for i in range(0, len(body), step)]
    aead = ChaCha20Poly1305(_blob_key(base_key, label))
    last = len(pieces) - 1
    try:
        return b"".join(aead.decrypt(_nonce(prefix, i, i == last), piece, header) for i, piece in enumerate(pieces))
    except InvalidTag as exc:
        raise DecryptionError("blob failed authentication (wrong key, label, or tampered/truncated data)") from exc
