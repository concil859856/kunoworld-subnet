"""Chunked authenticated encryption for media blobs (inputs and output video).

Format (all integers big-endian):

    header  = "KUNOB1" | version:u8 | chunk_size:u32 | nonce_prefix:7 bytes   (18 bytes)
    body    = chunk_0 || chunk_1 || ... ; each chunk = ChaCha20-Poly1305(stream piece) + 16-byte tag
    nonce_i = nonce_prefix | i:u32 | final:u8

    version 1 (unpadded):  stream = plaintext
    version 2 (padded):    stream = length:u64 | plaintext | zero bytes,
                           len(stream) = padme(8 + len(plaintext))

The key is HKDF-SHA256(base_key, info="kuno/v1/blob/" + label). The label binds a
blob to its job and role (e.g. "<job>/input/0"), so blobs cannot be swapped
between jobs or roles. The header, version byte included, is the AAD of every
chunk, so a padded blob cannot be relabelled as version 1. The final-chunk flag
makes truncation at a chunk boundary detectable.

Padding. Version 2 hides a blob's exact size: its plaintext length travels inside the
encrypted stream, and zeros fill the stream up to the next PADMÉ size (Nikitin et al.,
"Reducing Metadata Leakage from Encrypted Files and Communication with PURBs", PoPETs 2019,
arXiv:1806.03160, Algorithm 1). PADMÉ leaks O(log log M) bits for sizes up to M, like
padding to a power of two, but costs under 12% (worst case +11.63%, at 129 bytes) and about
1.5–3% for video-sized files. Decoders accept a version 2 stream only when its length is exactly
that bucket and every padding byte is zero, so a sealer cannot leak a length by
padding differently. New sealing writes version 2; version 1 blobs still decrypt.
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
V1 = 1  # unpadded: still decrypted everywhere, no longer written by default
V2 = 2  # padded to a PADMÉ bucket
DEFAULT_VERSION = V2
VERSIONS = frozenset({V1, V2})
DEFAULT_CHUNK = 1 << 20
MAX_CHUNK = 64 << 20
TAG_LEN = 16
PREFIX_LEN = 7
LENGTH_LEN = 8
_HEADER = struct.Struct(">6sBI7s")
HEADER_LEN = _HEADER.size


def padme(length: int) -> int:
    """PADMÉ (Nikitin et al. 2019, Algorithm 1): rounds `length` up so its low E−S bits are zero,
    where E = ⌊log2 length⌋ and S = ⌊log2 E⌋ + 1."""
    if length < 0:
        raise ValueError("length cannot be negative")
    if length < 2:
        return length
    exponent = length.bit_length() - 1  # E
    bits = exponent.bit_length()  # S = ⌊log2 E⌋ + 1
    mask = (1 << (exponent - bits)) - 1
    return (length + mask) & ~mask


def padded_stream_length(plaintext_length: int) -> int:
    """Length of a version 2 stream (length prefix, plaintext, padding) for a plaintext of this size."""
    return padme(LENGTH_LEN + plaintext_length)


def sealed_size(plaintext_length: int, chunk_size: int = DEFAULT_CHUNK, version: int = DEFAULT_VERSION) -> int:
    """Size of the blob `encrypt_blob` produces: what the gateway, the miner and receipts see."""
    if version not in VERSIONS:
        raise ValueError(f"unknown blob version {version}")
    stream = plaintext_length if version == V1 else padded_stream_length(plaintext_length)
    chunks = max(1, -(-stream // chunk_size))
    return HEADER_LEN + stream + TAG_LEN * chunks


def blob_version(blob: bytes) -> int | None:
    """The format version a blob declares, or None if it is not a KunoWorld blob. Unauthenticated."""
    if len(blob) < HEADER_LEN or blob[: len(MAGIC)] != MAGIC:
        return None
    return blob[len(MAGIC)]


def _blob_key(base_key: bytes, label: str) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"kuno/v1/blob/" + label.encode()).derive(base_key)


def _nonce(prefix: bytes, index: int, final: bool) -> bytes:
    return prefix + struct.pack(">I", index) + (b"\x01" if final else b"\x00")


def pad_stream(plaintext: bytes) -> bytes:
    """The version 2 stream for a plaintext: its length, the bytes, then zeros up to the PADMÉ bucket."""
    total = padded_stream_length(len(plaintext))
    return struct.pack(">Q", len(plaintext)) + plaintext + bytes(total - LENGTH_LEN - len(plaintext))


def encrypt_blob(
    base_key: bytes, label: str, plaintext: bytes, chunk_size: int = DEFAULT_CHUNK, *, version: int = DEFAULT_VERSION
) -> bytes:
    """Seals `plaintext` under `label`. Writes the padded format unless `version=V1` is asked for."""
    if version not in VERSIONS:
        raise ValueError(f"unknown blob version {version}")
    stream = pad_stream(plaintext) if version == V2 else plaintext
    return _encrypt_stream(base_key, label, stream, chunk_size, version, os.urandom(PREFIX_LEN))


def _encrypt_stream(base_key: bytes, label: str, stream: bytes, chunk_size: int, version: int, prefix: bytes) -> bytes:
    """Encrypts an already framed stream with a given nonce prefix. The prefix must never repeat under one key;
    only `encrypt_blob` (random prefix) and the protocol vector generator (fixed test keys) call this."""
    if not 0 < chunk_size <= MAX_CHUNK:
        raise ValueError("chunk_size out of range")
    if len(prefix) != PREFIX_LEN:
        raise ValueError("nonce prefix must be 7 bytes")
    aead = ChaCha20Poly1305(_blob_key(base_key, label))
    header = _HEADER.pack(MAGIC, version, chunk_size, prefix)
    chunks = [stream[i : i + chunk_size] for i in range(0, len(stream), chunk_size)] or [b""]
    last = len(chunks) - 1
    return header + b"".join(aead.encrypt(_nonce(prefix, i, i == last), chunk, header) for i, chunk in enumerate(chunks))


def decrypt_blob(base_key: bytes, label: str, blob: bytes) -> bytes:
    """Opens a version 1 or version 2 blob and returns the original plaintext."""
    if len(blob) < HEADER_LEN + TAG_LEN:
        raise DecryptionError("blob too short")
    magic, version, chunk_size, prefix = _HEADER.unpack_from(blob)
    if magic != MAGIC or version not in VERSIONS or not 0 < chunk_size <= MAX_CHUNK:
        raise DecryptionError("not a KunoWorld blob")
    header, body = blob[:HEADER_LEN], blob[HEADER_LEN:]
    step = chunk_size + TAG_LEN
    pieces = [body[i : i + step] for i in range(0, len(body), step)]
    aead = ChaCha20Poly1305(_blob_key(base_key, label))
    last = len(pieces) - 1
    try:
        stream = b"".join(aead.decrypt(_nonce(prefix, i, i == last), piece, header) for i, piece in enumerate(pieces))
    except InvalidTag as exc:
        raise DecryptionError("blob failed authentication (wrong key, label, or tampered/truncated data)") from exc
    return stream if version == V1 else unpad_stream(stream)


def unpad_stream(stream: bytes) -> bytes:
    """The plaintext inside an authenticated version 2 stream. Refuses a length that does not fit, a stream that is
    not exactly its PADMÉ bucket, and non-zero padding."""
    if len(stream) < LENGTH_LEN:
        raise DecryptionError("padded blob is too short to hold its length")
    (length,) = struct.unpack_from(">Q", stream)
    end = LENGTH_LEN + length
    if end > len(stream):
        raise DecryptionError("padded blob declares more plaintext than it holds")
    if padme(end) != len(stream):
        raise DecryptionError("padded blob is not padded to its size bucket")
    if stream.count(0, end) != len(stream) - end:
        raise DecryptionError("padded blob has non-zero padding")
    return stream[LENGTH_LEN:end]
