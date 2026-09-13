"""Verified-mode encodings: canonical latent hashes, the step Merkle tree and its proofs,
receipts with and without a step commitment, and sealed audit openings."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from kuno_protocol.canonical import b64d, b64e, canonical_json
from kuno_protocol.crypto import DecryptionError, generate_hpke_keypair, generate_signing_key, public_key_bytes
from kuno_protocol.profiles import load_profiles
from kuno_protocol.receipts import Receipt, ReceiptBody, VideoInfo, receipt_message, sign_receipt, verify_receipt
from kuno_protocol.toy_denoiser import run_toy_trajectory, toy_model_digest, toy_transcript
from kuno_protocol.verified import (
    LatentRecord,
    LeafProof,
    StepLeaf,
    StepOpening,
    TensorSpec,
    VerifiedModeError,
    build_commitment,
    f64_hex,
    f64_value,
    inclusion_proof,
    latent_digest,
    leaf_hash,
    merkle_root,
    new_salt,
    node_hash,
    open_sealed_opening,
    pack_tensors,
    required_leaves,
    seal_opening,
    tensor_from_array,
    unpack_tensors,
    verify_inclusion,
    verify_opening,
    verify_sealed_opening,
)

PROFILES = load_profiles()

# Merkle roots over leaf hashes SHA-256("leaf-i"); any change to the tree encoding breaks these.
ROOTS = {
    1: "d2dbf006f96dd05044a8f63d8f118f23925ba4cc5750f8b6c8e287fd506c8188",
    2: "bca491367c5592f0e1d9bdd49f4c8a59626ca1705048a459946ffd27d3ec1f6e",
    3: "241dc5eaaa367df9902c6a873eccaa4ac71df94f6e18e4ac9f621b12ad1637f1",
    5: "22401a3736026932f2bed17874d2569e8508223e62ac5ec388d0ef6486e93741",
    8: "8fc38d1841fe767b6d6ed1d2cece24c126c02e9afc48410c15585ff48dac8f72",
    13: "80c779ae587a2172b162385496e0019a5c2393d5e79cdacfcc802b66edb327d4",
}


def hashes(n: int) -> list[bytes]:
    return [hashlib.sha256(f"leaf-{i}".encode()).digest() for i in range(n)]


# ---------------------------------------------------------------- tree


@pytest.mark.parametrize(("n", "root"), sorted(ROOTS.items()))
def test_merkle_roots_are_pinned(n, root):
    assert merkle_root(hashes(n)).hex() == root


def test_tree_shape_follows_rfc9162():
    h = hashes(5)
    assert merkle_root(h[:1]) == h[0]
    # Split at the largest power of two below n: MTH(5) = node(MTH(0..4), MTH(4..5)).
    assert merkle_root(h) == node_hash(node_hash(node_hash(h[0], h[1]), node_hash(h[2], h[3])), h[4])
    with pytest.raises(VerifiedModeError):
        merkle_root([])


@pytest.mark.parametrize("n", range(1, 18))
def test_every_inclusion_proof_verifies(n):
    h = hashes(n)
    root = merkle_root(h)
    for index in range(n):
        assert verify_inclusion(root, h[index], index, n, inclusion_proof(h, index))


def test_tampered_proofs_are_rejected():
    n, index = 11, 6
    h = hashes(n)
    root, proof = merkle_root(h), inclusion_proof(h, index)
    assert verify_inclusion(root, h[index], index, n, proof)
    flipped = [bytes([proof[0][0] ^ 1]) + proof[0][1:], *proof[1:]]
    cases = [
        (root, h[index], index + 1, n, proof),  # wrong position
        # A tree size with a different audit-path shape. (Sizes 9–12 share leaf 6's path shape, which is
        # why verifiers always take the size from the signed commitment, never from the opening.)
        (root, h[index], index, 8, proof),
        (root, h[index + 1], index, n, proof),  # a different leaf
        (root, h[index], index, n, flipped),
        (root, h[index], index, n, proof[:-1]),  # truncated
        (root, h[index], index, n, proof + [h[0]]),  # extended
        (root, h[index], index, n, list(reversed(proof))),
        (root, h[index], n, n, proof),  # out of range
        (hashes(12)[0], h[index], index, n, proof),  # another root
    ]
    for case in cases:
        assert not verify_inclusion(*case)


def test_leaves_and_nodes_are_domain_separated_and_salted():
    a, b = hashes(2)
    assert node_hash(a, b) != hashlib.sha256(a + b).digest()
    leaf = StepLeaf(index=0, stage=0, kind="init", sigma=f64_hex(1.0), latent="0" * 64)
    assert leaf_hash(leaf, bytes(32)) != leaf_hash(leaf, b"\x01" * 32)
    assert leaf_hash(leaf, bytes(32)) != leaf_hash(leaf.model_copy(update={"index": 1}), bytes(32))
    with pytest.raises(VerifiedModeError):
        leaf_hash(leaf, bytes(16))


# ---------------------------------------------------------------- latents


def test_latent_digest_is_pinned_and_covers_dtype_shape_and_name():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    digest = latent_digest([tensor_from_array("video", a)])
    assert digest == "a6c244091ecdb25fe61e6d667b1cb05436713e3b18a6e16f7a8b212c26d1445d"
    assert latent_digest([tensor_from_array("video", a.reshape(3, 2))]) != digest
    assert latent_digest([tensor_from_array("video", a.astype(np.float64))]) != digest
    assert latent_digest([tensor_from_array("audio", a)]) != digest
    # Host byte order and memory layout do not matter: little-endian C order is what gets hashed.
    assert latent_digest([tensor_from_array("video", a.astype(">f4"))]) == digest
    assert latent_digest([tensor_from_array("video", np.asfortranarray(a))]) == digest
    # A single flipped bit changes it.
    b = a.copy()
    b.reshape(-1).view(np.uint8)[5] ^= 1
    assert latent_digest([tensor_from_array("video", b)]) != digest


def test_a_latent_state_is_order_independent_and_checked():
    video = tensor_from_array("video", np.zeros((2, 2), dtype=np.float16))
    audio = tensor_from_array("audio", np.ones((3,), dtype=np.float32))
    assert latent_digest([video, audio]) == latent_digest([audio, video])
    assert unpack_tensors(pack_tensors([video, audio])) == [audio, video]
    spec = TensorSpec(name="video", dtype="bfloat16", shape=(2, 3))
    with pytest.raises(VerifiedModeError):
        latent_digest([(spec, b"\x00" * 11)])
    with pytest.raises(VerifiedModeError):
        latent_digest([video, video])
    with pytest.raises(ValidationError):
        TensorSpec(name="video", dtype="int8", shape=(1,))
    assert f64_value(f64_hex(0.1)) == 0.1 and f64_hex(1.0) == "3ff0000000000000"


# ---------------------------------------------------------------- toy trajectories


def honest(seed: int = 7, prompt: str = "a boat", job_id: str = "job-1", steps: int = 11):
    profile = PROFILES["ltx-2.5-fast"]
    transcript = toy_transcript(
        job_id=job_id, params_digest="0" * 64, profile_id=profile.id, family=profile.family,
        model_digest=toy_model_digest(profile.id, profile.checkpoint), seed=seed, prompt=prompt, negative_prompt=None,
        frames=49, steps=steps,
    )
    leaves, states = [], {}
    for index, stage, kind, sigma, tensors in run_toy_trajectory(transcript, prompt, None):
        leaves.append(StepLeaf(index=index, stage=stage, kind=kind, sigma=f64_hex(sigma), latent=latent_digest(tensors)))
        states[index] = tensors
    return transcript, leaves, states


def test_the_toy_reference_trajectory_is_pinned():
    """Guards bitwise determinism of the dev reference model across machines and NumPy builds."""
    transcript, leaves, _ = honest(seed=1, prompt="hi", job_id="j")
    commitment, _ = build_commitment(transcript, leaves, bytes(32))
    assert commitment.root == "7b4ac23947a8bca099abb6837651f67201dbb23946808f5e6bdb2f141b97abf0"
    assert (commitment.leaves, commitment.steps, commitment.latent_shape) == (12, 11, [4, 4, 8, 8])


def test_a_commitment_needs_exactly_the_scheduled_leaves():
    transcript, leaves, _ = honest()
    with pytest.raises(VerifiedModeError):
        build_commitment(transcript, leaves[:-1], new_salt())
    skipped = leaves[:5] + [leaf.model_copy(update={"index": leaf.index - 1}) for leaf in leaves[6:]]
    with pytest.raises(VerifiedModeError):
        build_commitment(transcript, skipped, new_salt())


# ---------------------------------------------------------------- receipts


def receipt_body(**extra) -> ReceiptBody:
    return ReceiptBody(
        job_id="3f2504e0-4f89-41d3-9a0c-0305e82c3301", enclave_id="0" * 32, profile_id="ltx-2.5-fast", image_digest="sha256:img",
        params_digest="1" * 64, input_digest="2" * 64, output_digest="3" * 64, output_bytes=10, content_digest="4" * 64,
        attestation_digest="5" * 64, started_at=1.0, finished_at=2.5, gpu_seconds=1.5,
        video=VideoInfo(duration_s=2, width=1280, height=704, fps=24, frames=49, audio=True), miner_hotkey=None, **extra,
    )


def test_a_receipt_without_a_commitment_encodes_and_verifies_as_before():
    body = receipt_body()
    dumped = body.model_dump(mode="json")
    assert "step_commitment" not in dumped and "step_commitment" not in body.model_dump_json()
    assert receipt_message(body) == b"kuno/v1/receipt\n" + canonical_json(dumped)
    key = generate_signing_key()
    receipt = sign_receipt(key, body)
    # A relay that writes the absent field back as null changes nothing.
    relayed = Receipt.model_validate({"body": {**dumped, "step_commitment": None}, "signature": receipt.signature})
    assert verify_receipt(relayed, public_key_bytes(key))


def test_a_step_commitment_is_signed_with_the_receipt():
    transcript, leaves, _ = honest()
    commitment, _ = build_commitment(transcript, leaves, new_salt())
    key = generate_signing_key()
    public = public_key_bytes(key)
    receipt = sign_receipt(key, receipt_body(step_commitment=commitment))
    assert verify_receipt(receipt, public)
    again = Receipt.model_validate_json(receipt.model_dump_json())
    assert verify_receipt(again, public) and again.body.step_commitment == commitment

    forged = Receipt.model_validate_json(receipt.model_dump_json())
    forged.body.step_commitment.root = "f" * 64
    assert not verify_receipt(forged, public)
    stripped = Receipt(body=receipt.body.model_copy(update={"step_commitment": None}), signature=receipt.signature)
    assert not verify_receipt(stripped, public)
    with pytest.raises(ValidationError):
        Receipt.model_validate(
            {"body": {**receipt.body.model_dump(mode="json"), "step_commitment": {**commitment.model_dump(), "extra": 1}}, "signature": "x"}
        )


# ---------------------------------------------------------------- openings


def make_opening(step: int = 4, include_leaves: bool = False):
    transcript, leaves, states = honest()
    salt = new_salt()
    commitment, leaf_hashes = build_commitment(transcript, leaves, salt)
    indices = required_leaves(step, commitment.leaves, include_leaves)
    latents = {step - 1: states[step - 1], step: states[step]}
    opening = StepOpening(
        audit_id="a1", job_id=transcript.job_id, enclave_id="e" * 32, step=step, commitment=commitment, transcript=transcript,
        salt=salt.hex(), leaves=[leaves[i] for i in indices],
        proofs=[LeafProof(index=i, path=[p.hex() for p in inclusion_proof(leaf_hashes, i)]) for i in indices],
        latents=[LatentRecord(index=i, tensors=[spec for spec, _ in latents[i]]) for i in sorted(latents)],
    )
    enclave_key = generate_signing_key()
    private, public = generate_hpke_keypair()
    sealed = seal_opening(enclave_key, opening, latents, public)
    return SimpleNamespace(
        commitment=commitment, opening=opening, latents=latents, sealed=sealed, private=private, enclave_public=public_key_bytes(enclave_key),
    )


@pytest.mark.parametrize(("step", "include_leaves"), [(1, False), (4, False), (11, False), (6, True)])
def test_a_sealed_opening_round_trips_and_verifies(step, include_leaves):
    o = make_opening(step, include_leaves)
    assert verify_sealed_opening(o.sealed, o.enclave_public)
    opening, latents = open_sealed_opening(o.private, o.sealed)
    assert opening == o.opening
    assert verify_opening(o.commitment, opening, latents, job_id="job-1", step=step, include_leaves=include_leaves) is None


def test_an_opening_is_sealed_to_the_validator_and_signed_by_the_enclave():
    o = make_opening()
    other_private, _ = generate_hpke_keypair()
    with pytest.raises(DecryptionError):
        open_sealed_opening(other_private, o.sealed)
    assert not verify_sealed_opening(o.sealed.model_copy(update={"step": 5}), o.enclave_public)
    ciphertext = bytearray(b64d(o.sealed.ciphertext))
    ciphertext[-1] ^= 1
    swapped = o.sealed.model_copy(update={"ciphertext": b64e(bytes(ciphertext))})
    assert not verify_sealed_opening(swapped, o.enclave_public)
    with pytest.raises(DecryptionError):
        open_sealed_opening(o.private, swapped)
    assert not verify_sealed_opening(o.sealed, public_key_bytes(generate_signing_key()))


def _flip(tensors):
    (spec, data), = tensors
    raw = bytearray(data)
    raw[0] ^= 1
    return [(spec, bytes(raw))]


def test_tampered_openings_fail_verification():
    o = make_opening(step=4)
    c, op, lat = o.commitment, o.opening, o.latents

    def check(opening=op, latents=lat, **kw):
        args = {"job_id": "job-1", "step": 4, **kw}
        return verify_opening(c, opening, latents, **args)

    assert "does not hash" in check(latents={3: lat[3], 4: _flip(lat[4])})
    assert "does not hash" in check(latents={3: _flip(lat[3]), 4: lat[4]})
    moved = [leaf.model_copy(update={"latent": "a" * 64}) if leaf.index == 4 else leaf for leaf in op.leaves]
    assert "inclusion proof" in check(opening=op.model_copy(update={"leaves": moved}))
    assert "omits leaf 0" in check(opening=op.model_copy(update={"leaves": [l for l in op.leaves if l.index != 0]}))
    assert "omits leaf" in check(include_leaves=True)
    assert "not 5" in check(step=5)
    assert "different job" in check(job_id="job-2")
    assert "transcript" in check(opening=op.model_copy(update={"transcript": op.transcript.model_copy(update={"seed": 8})}))
    assert "different commitment" in check(opening=op.model_copy(update={"commitment": c.model_copy(update={"root": "b" * 64})}))
    assert "omits the latent" in check(latents={4: lat[4]})
    assert "salt" in check(opening=op.model_copy(update={"salt": "00"}))
