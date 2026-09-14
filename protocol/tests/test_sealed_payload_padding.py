"""The sealed request (the HPKE plaintext carrying a SealedPayload) is padded to a power-of-two bucket: the shared
vectors, buckets and limits, malformed framing, and bare-JSON compatibility. sdk/js/test/payload-padding.test.mjs
checks the same vectors in TypeScript."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kuno_protocol.canonical import b64d
from kuno_protocol.crypto import (
    _EXPORT_INPUT,
    _EXPORT_OUTPUT,
    HPKE_INFO,
    SUITE,
    DecryptionError,
    RecipientSession,
    SenderSession,
    generate_hpke_keypair,
)
from kuno_protocol.schemas import SealedPayload
from kuno_protocol.sealed_payload import (
    HEADER_LEN,
    MAX_JSON_LEN,
    MAX_PADDED,
    MIN_PADDED,
    PAYLOAD_V1,
    PAYLOAD_V2,
    MalformedPayload,
    PayloadTooLarge,
    open_payload,
    pad_payload,
    padded_payload_length,
    payload_version,
    seal_payload,
    unpad_payload,
)

VECTORS = json.loads((Path(__file__).parent / "vectors.json").read_text(encoding="utf-8"))["sealed_payload"]
AAD = b'{"inputs":[],"job_id":"test","v":1}'
TAG_LEN = 16
EMPTY_JSON_LEN = len(SealedPayload(prompt="").model_dump_json())  # 79 bytes


def json_of(prompt: str, **fields) -> bytes:
    return SealedPayload(prompt=prompt, **fields).model_dump_json().encode()


def sparse(case: dict) -> bytes:
    head, tail = bytes.fromhex(case["head_hex"]), bytes.fromhex(case["tail_hex"])
    return head + bytes(case["length"] - len(head) - len(tail)) + tail


def seal_raw(plaintext: bytes) -> tuple[bytes, bytes, bytes]:
    """Seals any plaintext as a sender would, bypassing the padding, so the receiver's checks can be exercised."""
    private, public = generate_hpke_keypair()
    sender = SenderSession(public)
    return private, sender.enc, sender.seal(plaintext, AAD)


def opened(private: bytes, enc: bytes, ciphertext: bytes) -> SealedPayload:
    return open_payload(RecipientSession(private, enc), ciphertext, AAD)


# ---------------------------------------------------------------- shared vectors


def test_the_limits_are_the_published_ones():
    assert (VECTORS["header_len"], VECTORS["min_padded"], VECTORS["max_padded"]) == (HEADER_LEN, MIN_PADDED, MAX_PADDED)
    assert (HEADER_LEN, MIN_PADDED, MAX_PADDED, MAX_JSON_LEN) == (5, 4096, 262144, 262139)
    assert VECTORS["info"].encode() == HPKE_INFO


def test_the_sealed_vectors_reproduce_byte_for_byte_and_open():
    recipient = SUITE.kem.derive_key_pair(bytes.fromhex(VECTORS["recipient_ikm_hex"]))
    private = b64d(VECTORS["recipient_private_key_b64"])
    assert recipient.private_key.to_private_bytes() == private
    assert recipient.public_key.to_public_bytes() == b64d(VECTORS["recipient_public_key_b64"])
    aad = VECTORS["aad"].encode()
    assert [case["form"] for case in VECTORS["sealed"]] == [PAYLOAD_V2, PAYLOAD_V2, PAYLOAD_V1]
    for case in VECTORS["sealed"]:
        payload_json = case["payload_json"].encode()
        plaintext = pad_payload(payload_json) if case["form"] == PAYLOAD_V2 else payload_json
        assert len(plaintext) == case["plaintext_length"], case["name"]
        assert hashlib.sha256(plaintext).hexdigest() == case["plaintext_sha256"], case["name"]

        ephemeral = SUITE.kem.derive_key_pair(bytes.fromhex(case["ephemeral_ikm_hex"]))
        enc, ctx = SUITE.create_sender_context(recipient.public_key, info=HPKE_INFO, eks=ephemeral)
        ciphertext = b64d(case["ciphertext_b64"])
        assert enc == b64d(case["enc_b64"])
        assert ctx.export(_EXPORT_INPUT, 32).hex() == case["input_key_hex"]
        assert ctx.export(_EXPORT_OUTPUT, 32).hex() == case["output_key_hex"]
        assert ctx.seal(plaintext, aad=aad) == ciphertext, case["name"]

        session = RecipientSession(private, enc)
        assert (session.input_key.hex(), session.output_key.hex()) == (case["input_key_hex"], case["output_key_hex"])
        payload = open_payload(session, ciphertext, aad)
        assert payload.model_dump_json().encode() == payload_json, case["name"]
        with pytest.raises(DecryptionError):
            open_payload(RecipientSession(private, enc), ciphertext, aad + b" ")
    assert [len(b64d(case["ciphertext_b64"])) for case in VECTORS["sealed"][:2]] == [MIN_PADDED + TAG_LEN] * 2


def test_the_bucket_table_and_padded_plaintexts_match():
    for json_length, padded in VECTORS["buckets"]:
        if padded is None:
            with pytest.raises(PayloadTooLarge):
                padded_payload_length(json_length)
        else:
            assert padded_payload_length(json_length) == padded, json_length
    for case in VECTORS["padded"]:
        payload_json = json_of("a" * case["prompt_chars"])
        assert len(payload_json) == case["json_length"], case["name"]
        padded = pad_payload(payload_json)
        assert len(padded) == case["padded_length"] and hashlib.sha256(padded).hexdigest() == case["padded_sha256"], case["name"]
        assert unpad_payload(padded) == payload_json


def test_malformed_framing_is_refused_after_authentication():
    assert len(VECTORS["invalid"]) == 10
    for case in VECTORS["invalid"]:
        plaintext = sparse(case)
        assert len(plaintext) == case["length"]
        with pytest.raises(MalformedPayload):
            unpad_payload(plaintext)
        private, enc, ciphertext = seal_raw(plaintext)
        with pytest.raises(MalformedPayload):
            opened(private, enc, ciphertext)


# ---------------------------------------------------------------- sizes


def test_an_empty_prompt_is_padded_to_the_minimum_bucket():
    private, public = generate_hpke_keypair()
    sender = SenderSession(public)
    ciphertext = seal_payload(sender, SealedPayload(prompt=""), AAD)
    assert len(ciphertext) == MIN_PADDED + TAG_LEN
    plaintext = RecipientSession(private, sender.enc).open(ciphertext, AAD)
    assert payload_version(plaintext) == PAYLOAD_V2 and plaintext[1:5] == EMPTY_JSON_LEN.to_bytes(4, "big")
    assert opened(private, sender.enc, ciphertext) == SealedPayload(prompt="")


@pytest.mark.parametrize(("character", "bucket"), [("a", 8 * 1024), ("é", 16 * 1024), ("🌊", 32 * 1024)])
def test_a_7000_character_prompt_round_trips(character, bucket):
    payload = SealedPayload(prompt=character * 7000, negative_prompt="blurry", seed=2**31 - 1)
    private, public = generate_hpke_keypair()
    sender = SenderSession(public)
    ciphertext = seal_payload(sender, payload, AAD)
    assert len(ciphertext) == bucket + TAG_LEN
    assert opened(private, sender.enc, ciphertext) == payload


def test_every_prompt_that_fits_the_minimum_bucket_seals_to_one_size():
    fill = MIN_PADDED - HEADER_LEN - EMPTY_JSON_LEN  # 4012 ASCII characters
    assert {len(pad_payload(json_of("a" * n))) for n in range(fill + 1)} == {MIN_PADDED}
    assert len(pad_payload(json_of("é" * (fill // 2)))) == MIN_PADDED
    assert len(pad_payload(json_of("a" * (fill + 1)))) == 2 * MIN_PADDED


def test_bucket_boundaries_up_to_the_maximum():
    bucket = MIN_PADDED
    while bucket <= MAX_PADDED:
        exact = b"{" + b" " * (bucket - HEADER_LEN - 1)
        assert len(pad_payload(exact)) == bucket and unpad_payload(pad_payload(exact)) == exact
        if bucket < MAX_PADDED:
            assert len(pad_payload(exact + b" ")) == 2 * bucket
        bucket *= 2
    assert padded_payload_length(0) == MIN_PADDED
    with pytest.raises(PayloadTooLarge):
        pad_payload(bytes(MAX_JSON_LEN + 1))
    with pytest.raises(ValueError):
        padded_payload_length(-1)


def test_a_request_too_large_is_refused_before_the_session_is_used():
    private, public = generate_hpke_keypair()
    sender = SenderSession(public)
    with pytest.raises(PayloadTooLarge):
        seal_payload(sender, SealedPayload(prompt="a" * MAX_JSON_LEN), AAD)
    ciphertext = seal_payload(sender, SealedPayload(prompt="still usable"), AAD)
    assert opened(private, sender.enc, ciphertext).prompt == "still usable"


# ---------------------------------------------------------------- compatibility and strictness


def test_bare_json_requests_still_open():
    payload = SealedPayload(prompt="sealed before padding", seed=3)
    for plaintext in (payload.model_dump_json().encode(), b"\n " + payload.model_dump_json().encode()):
        assert payload_version(plaintext) == PAYLOAD_V1 and unpad_payload(plaintext) == plaintext
        private, enc, ciphertext = seal_raw(plaintext)
        assert opened(private, enc, ciphertext) == payload


def test_padding_must_be_zero_everywhere_and_the_length_must_be_exact():
    padded = pad_payload(json_of("a fox in the snow"))
    end = HEADER_LEN + int.from_bytes(padded[1:5], "big")
    for index in (end, end + 1, len(padded) // 2, len(padded) - 1):
        bad = bytearray(padded)
        bad[index] = 0x20
        with pytest.raises(MalformedPayload):
            unpad_payload(bytes(bad))
    # Declaring one byte less turns the closing brace into non-zero padding.
    short = bytearray(padded)
    short[1:5] = (end - HEADER_LEN - 1).to_bytes(4, "big")
    with pytest.raises(MalformedPayload):
        unpad_payload(bytes(short))


def test_tampering_fails_authentication_and_bad_json_is_a_validation_error():
    private, enc, ciphertext = seal_raw(pad_payload(json_of("a fox")))
    with pytest.raises(DecryptionError):
        opened(private, enc, ciphertext[:-1] + bytes([ciphertext[-1] ^ 1]))
    private, enc, ciphertext = seal_raw(pad_payload(b'{"v":2,"prompt":"x"}'))
    with pytest.raises(ValidationError):
        opened(private, enc, ciphertext)
