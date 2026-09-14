"""Blob format version 2: PADMÉ size padding with an authenticated length, and version 1 compatibility.
sdk/js/test/padding.test.mjs runs the same checks in TypeScript."""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

import pytest

from kuno_protocol.blobs import (
    DEFAULT_CHUNK,
    DEFAULT_VERSION,
    HEADER_LEN,
    TAG_LEN,
    V1,
    V2,
    _encrypt_stream,
    blob_version,
    decrypt_blob,
    encrypt_blob,
    pad_stream,
    padded_stream_length,
    padme,
    sealed_size,
)
from kuno_protocol.canonical import b64d
from kuno_protocol.crypto import DecryptionError

VECTORS = json.loads((Path(__file__).parent / "vectors.json").read_text(encoding="utf-8"))
KEY = bytes(range(32))
LABEL = "job/output/video"


def algorithm_1(length: int) -> int:
    """PADMÉ exactly as printed in Nikitin et al. 2019, Algorithm 1 (floating-point logs, fine for these sizes)."""
    exponent = math.floor(math.log2(length))
    bits = math.floor(math.log2(exponent)) + 1
    mask = (1 << (exponent - bits)) - 1
    return (length + mask) & ~mask


# ---------------------------------------------------------------- PADMÉ buckets


def test_padme_matches_algorithm_1_of_the_paper():
    for length in [*range(2, 70_000), 1 << 20, (1 << 20) + 1, 5_000_000, 30_000_000, 123_456_789]:
        assert padme(length) == algorithm_1(length), length
    assert [padme(n) for n in (0, 1)] == [0, 1]
    with pytest.raises(ValueError):
        padme(-1)


def test_padme_overhead_is_bounded_and_never_shrinks_a_length():
    overheads = {n: (padme(n) - n) / n for n in range(2, 200_000)}
    worst = max(overheads, key=overheads.get)
    assert (worst, overheads[worst]) == (129, pytest.approx(15 / 129))  # +11.63%, under the paper's 12% bound
    previous = 0
    for n in range(0, 100_000):
        assert padme(n) >= n and padme(n) >= previous
        previous = padme(n)
    # Video-sized files pay at most about 3% (2^(E-S) - 1 bytes over 2^E).
    for n in (1 << 20, 3_000_000, 12_000_000, 30_000_000, 100_000_000):
        assert (padme(n) - n) / n < 0.032


def test_plaintexts_in_one_bucket_seal_to_one_size():
    # 8 + L in (1280, 1344] all pad to 1344 bytes of stream.
    sizes = {len(encrypt_blob(KEY, LABEL, os.urandom(n), chunk_size=100)) for n in range(1273, 1337)}
    assert sizes == {sealed_size(1336, 100)}
    assert len(encrypt_blob(KEY, LABEL, os.urandom(1337), chunk_size=100)) > sizes.pop()


@pytest.mark.parametrize("chunk", [1, 7, 8, 100, 1024, DEFAULT_CHUNK])
def test_sealed_size_predicts_the_blob_for_both_versions(chunk):
    for n in (0, 1, 7, 8, 9, 99, 100, 101, 1023, 1024, 3000):
        data = os.urandom(n)
        assert len(encrypt_blob(KEY, LABEL, data, chunk)) == sealed_size(n, chunk) == sealed_size(n, chunk, V2)
        assert len(encrypt_blob(KEY, LABEL, data, chunk, version=V1)) == sealed_size(n, chunk, V1)
    assert sealed_size(5_000_000) - sealed_size(5_000_000, version=V1) == padded_stream_length(5_000_000) - 5_000_000


# ---------------------------------------------------------------- round trips and versions


@pytest.mark.parametrize("chunk", [8, 100, 1024])
def test_padded_round_trips_across_chunk_boundaries(chunk):
    for n in (0, 1, 7, 8, 9, 255, 256, 1023, 1024, 1025, 3 * 1024 + 17, 70_000):
        data = os.urandom(n)
        sealed = encrypt_blob(KEY, LABEL, data, chunk)
        assert blob_version(sealed) == V2
        assert decrypt_blob(KEY, LABEL, sealed) == data


def test_new_sealing_is_padded_by_default_and_version_1_can_still_be_written():
    assert DEFAULT_VERSION == V2
    data = os.urandom(5000)
    assert blob_version(encrypt_blob(KEY, LABEL, data)) == V2
    old = encrypt_blob(KEY, LABEL, data, version=V1)
    assert blob_version(old) == V1 and len(old) == HEADER_LEN + len(data) + TAG_LEN
    assert decrypt_blob(KEY, LABEL, old) == data
    assert blob_version(b"not a blob at all, just bytes") is None
    with pytest.raises(ValueError):
        encrypt_blob(KEY, LABEL, data, version=3)


def test_the_published_version_1_vector_still_decrypts_and_reproduces():
    blob = VECTORS["blob"]
    key, plaintext, ciphertext = b64d(blob["base_key_b64"]), b64d(blob["plaintext_b64"]), b64d(blob["ciphertext_b64"])
    assert blob_version(ciphertext) == V1
    assert decrypt_blob(key, blob["label"], ciphertext) == plaintext
    assert _encrypt_stream(key, blob["label"], plaintext, blob["chunk_size"], V1, ciphertext[11:18]) == ciphertext


# ---------------------------------------------------------------- tampering


def _flip(blob: bytes, index: int) -> bytes:
    out = bytearray(blob)
    out[index] ^= 1
    return bytes(out)


def test_tampering_with_the_length_padding_or_version_fails_authentication():
    data = os.urandom(1280)
    sealed = encrypt_blob(KEY, LABEL, data, chunk_size=100)  # 1344-byte stream: the last chunk is all padding
    cases = {
        "length prefix": _flip(sealed, HEADER_LEN),
        "last padding byte": _flip(sealed, len(sealed) - TAG_LEN - 1),
        "downgrade to version 1": sealed[:6] + bytes([V1]) + sealed[7:],
        "chunk size": _flip(sealed, 10),
        "dropped final chunk": sealed[: HEADER_LEN + 13 * (100 + TAG_LEN)],
        "truncated final chunk": sealed[:-1],
        "trailing bytes": sealed + b"\x00" * 20,
    }
    for name, blob in cases.items():
        with pytest.raises(DecryptionError):
            decrypt_blob(KEY, LABEL, blob)
        assert name
    with pytest.raises(DecryptionError):
        decrypt_blob(KEY, "job/input/0", sealed)
    # Upgrading a version 1 blob's header to version 2 fails too: the header is every chunk's AAD.
    old = encrypt_blob(KEY, LABEL, data, 100, version=V1)
    with pytest.raises(DecryptionError):
        decrypt_blob(KEY, LABEL, old[:6] + bytes([V2]) + old[7:])
    with pytest.raises(DecryptionError, match="not a KunoWorld blob"):
        decrypt_blob(KEY, LABEL, sealed[:6] + b"\x03" + sealed[7:])


@pytest.mark.parametrize(
    "stream",
    [
        pytest.param(struct.pack(">Q", 1337) + bytes(1336), id="length-past-the-stream"),
        pytest.param(b"\xff" * 8 + bytes(8), id="huge-length"),
        pytest.param(pad_stream(bytes(1280))[:-1] + b"\x01", id="non-zero-padding"),
        pytest.param(struct.pack(">Q", 1280) + bytes(1280), id="unpadded"),
        pytest.param(struct.pack(">Q", 1280) + bytes(1280) + bytes(1408 - 1288), id="over-padded"),
        pytest.param(b"\x00" * 7, id="shorter-than-a-length"),
    ],
)
def test_an_authentic_stream_with_a_bad_length_or_padding_is_refused(stream):
    """The sealer holds the key, so these decrypt; the decoder must still refuse them."""
    blob = _encrypt_stream(KEY, LABEL, stream, 100, V2, os.urandom(7))
    with pytest.raises(DecryptionError, match="padded blob"):
        decrypt_blob(KEY, LABEL, blob)


# ---------------------------------------------------------------- shared vectors


def test_the_version_2_vectors():
    v2 = VECTORS["blob_v2"]
    key, label, chunk = b64d(v2["base_key_b64"]), v2["label"], v2["chunk_size"]
    assert [c["padded_stream_length"] for c in v2["cases"]] == [8, 10, 1344, 1344]
    for case in v2["cases"]:
        plaintext, ciphertext = b64d(case["plaintext_b64"]), b64d(case["ciphertext_b64"])
        assert decrypt_blob(key, label, ciphertext) == plaintext, case["name"]
        assert padded_stream_length(len(plaintext)) == case["padded_stream_length"]
        assert len(ciphertext) == sealed_size(len(plaintext), chunk) == case["sealed_size"]
        assert _encrypt_stream(key, label, pad_stream(plaintext), chunk, V2, bytes.fromhex(case["nonce_prefix_hex"])) == ciphertext
        with pytest.raises(DecryptionError):
            decrypt_blob(key, "other/label", ciphertext)
    for case in v2["invalid"]:
        with pytest.raises(DecryptionError):
            decrypt_blob(key, label, b64d(case["ciphertext_b64"]))
    for length, padded in v2["padme"]:
        assert padme(length) == padded, length
    for row in v2["sealed_sizes"]:
        assert sealed_size(row["plaintext_length"], row["chunk_size"], V1) == row["v1"]
        assert sealed_size(row["plaintext_length"], row["chunk_size"], V2) == row["v2"]
