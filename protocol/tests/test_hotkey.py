"""Proof of hotkey ownership, checked against signatures made by Bittensor's own wallet library."""

from __future__ import annotations

import os

import pytest

from kuno_protocol.attestation import MockTEE, build_evidence
from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key, public_key_bytes
from kuno_protocol.hotkey import (
    HotkeyError,
    HotkeyProof,
    Sr25519Signer,
    sign_hotkey_proof,
    ss58_decode,
    ss58_encode,
    verify_hotkey_proof,
)
from kuno_protocol.schemas import MinerRegistration

ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
ALICE_PUBLIC_KEY = bytes.fromhex("d43593c715fdd31c61141abd04a99fd6822c8558854ccde39a5684e7a56da27d")
VECTOR = dict(hotkey=ALICE, nonce="11" * 32, enclave_id="ab" * 16, signing_public_key=b64e(b"\x07" * 32))
# bittensor-wallet 4.1.1: Keypair.create_from_uri("//Alice").sign(hotkey_proof_message(**VECTOR)),
# and the same over the polkadot.js "<Bytes>…</Bytes>" wrapping.
WALLET_SIGNATURE = "QpbwcHFS5trcRjSnCB8Rf6a48_AjKYgy3pKfeoyd5BtIrutFyrppKRpJoMcTqkT-O7vkUwpbvd15Mq7b-1_KhA"
WRAPPED_SIGNATURE = "oI6xekRwm4mhdEmXhD6ey_L77inlBnT2n6EJkgye4X_cfPM7fdGVIh_L4EZLL_jYUHw8zD6m2sRnXh-wrU6hhw"


def check(proof: HotkeyProof, **overrides):
    expected = dict(expected_nonce=VECTOR["nonce"], enclave_id=VECTOR["enclave_id"], signing_public_key=VECTOR["signing_public_key"])
    return verify_hotkey_proof(proof, **{**expected, **overrides})


def test_ss58_addresses_match_substrate():
    assert ss58_decode(ALICE) == ALICE_PUBLIC_KEY
    assert ss58_encode(ALICE_PUBLIC_KEY) == ALICE
    with pytest.raises(HotkeyError, match="checksum"):
        ss58_decode(ALICE[:-1] + ("A" if ALICE[-1] != "A" else "B"))
    polkadot = ss58_encode(ALICE_PUBLIC_KEY, ss58_format=0)
    assert polkadot.startswith("1")
    with pytest.raises(HotkeyError, match="prefix 0"):
        ss58_decode(polkadot)
    assert ss58_decode(polkadot, ss58_format=None) == ALICE_PUBLIC_KEY
    assert ss58_decode(ss58_encode(ALICE_PUBLIC_KEY, 1000), 1000) == ALICE_PUBLIC_KEY
    with pytest.raises(HotkeyError):
        ss58_decode("0OIl")


def test_signatures_from_bittensor_wallet_verify():
    assert check(HotkeyProof(**VECTOR, signature=WALLET_SIGNATURE)) == (True, "ok")
    assert check(HotkeyProof(**VECTOR, signature=WRAPPED_SIGNATURE)) == (True, "ok")
    assert not check(HotkeyProof(**{**VECTOR, "nonce": "22" * 32}, signature=WALLET_SIGNATURE), expected_nonce="22" * 32)[0]


def test_proof_binds_nonce_enclave_signing_key_and_hotkey():
    signer = Sr25519Signer.from_seed(os.urandom(32))
    nonce, sign_pk = os.urandom(32), public_key_bytes(generate_signing_key())
    proof = sign_hotkey_proof(signer, nonce, "cd" * 16, sign_pk)
    ok = dict(expected_nonce=nonce, enclave_id="cd" * 16, signing_public_key=sign_pk)
    assert verify_hotkey_proof(proof, **ok, hotkey=signer.ss58_address) == (True, "ok")

    assert "nonce" in verify_hotkey_proof(proof, **{**ok, "expected_nonce": os.urandom(32)})[1]
    assert "different enclave" in verify_hotkey_proof(proof, **{**ok, "enclave_id": "ef" * 16})[1]
    assert "signing key" in verify_hotkey_proof(proof, **{**ok, "signing_public_key": os.urandom(32)})[1]
    assert "different hotkey" in verify_hotkey_proof(proof, **ok, hotkey=ALICE)[1]

    impostor = Sr25519Signer.from_seed(os.urandom(32))
    stolen = proof.model_copy(update={"signature": sign_hotkey_proof(impostor, nonce, "cd" * 16, sign_pk).signature})
    assert verify_hotkey_proof(stolen, **ok) == (False, "signature does not verify for this hotkey")
    claimed = sign_hotkey_proof(impostor, nonce, "cd" * 16, sign_pk).model_copy(update={"hotkey": ALICE})
    assert not verify_hotkey_proof(claimed, **ok)[0]
    assert "malformed" in verify_hotkey_proof(proof.model_copy(update={"hotkey": "nope"}), **ok)[1]


def test_signer_does_not_reveal_its_secret():
    seed = os.urandom(32)
    signer = Sr25519Signer.from_seed(seed)
    assert signer.ss58_address in repr(signer) and seed.hex() not in repr(signer)
    with pytest.raises(HotkeyError):
        Sr25519Signer.from_seed(b"short")


def test_registration_schema_accepts_old_and_new_workers():
    _, hpke_pk = generate_hpke_keypair()
    sign_pk = public_key_bytes(generate_signing_key())
    evidence = build_evidence(MockTEE(generate_signing_key(), "sha256:x"), os.urandom(32), hpke_pk, sign_pk, "sha256:x", ["h3"])
    old = MinerRegistration.model_validate({"evidence": evidence.model_dump(mode="json"), "miner_hotkey": "5Old", "capacity": 2})
    assert old.hotkey_proof is None and old.capacity == 2

    signer = Sr25519Signer.from_seed(os.urandom(32))
    proof = sign_hotkey_proof(signer, bytes.fromhex(evidence.nonce), evidence.enclave_id, sign_pk)
    new = MinerRegistration.model_validate_json(
        MinerRegistration(evidence=evidence, miner_hotkey=signer.ss58_address, hotkey_proof=proof).model_dump_json()
    )
    assert verify_hotkey_proof(
        new.hotkey_proof,
        expected_nonce=new.evidence.nonce,
        enclave_id=new.evidence.enclave_id,
        signing_public_key=new.evidence.signing_public_key,
        hotkey=new.miner_hotkey,
    )[0]
