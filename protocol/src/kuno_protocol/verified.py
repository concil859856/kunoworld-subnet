"""Verified mode: per-step latent commitments and single-step audit openings.

TEE attestation is the primary check that a miner runs the pinned model. Verified mode
adds an audit that still holds if a TEE is broken: in a deterministic profile (pinned
image, fixed GPU SKU/count/parallel layout, deterministic kernels, CPU-seeded noise, no
data-dependent caching) the enclave hashes the latent state after every denoising step,
builds a Merkle tree over those per-step leaves and signs the root inside the receipt.
A validator later asks for one step of one of *its own* canary jobs, receives the latent
before and after that step with inclusion proofs, re-executes the step and compares the
result bit for bit. A substituted model or a skipped step changes the committed leaves,
so it survives an audit of step k only if the cheat happens to leave step k untouched;
with random k and repeated canaries, cheating is caught with high probability.

Encodings (all hashes SHA-256, all integers big-endian):

    latent digest  H("kuno/v1/latent\\n" | u32(len(hdr)) | hdr | tensor bytes…)
                   hdr = canonical_json({"v":1,"byte_order":"little","order":"C",
                                         "tensors":[{"name","dtype","shape"}… sorted by name]})
    leaf hash      H(0x00 | "kuno/v1/step-leaf\\n" | salt(32) | canonical_json(StepLeaf))
    node hash      H(0x01 | "kuno/v1/step-node\\n" | left | right)
    tree           RFC 9162 §2.1.1 (split at the largest power of two below n)
    transcript     H("kuno/v1/step-transcript\\n" | canonical_json(StepTranscript))

Floats that must round-trip exactly (sigmas, guidance) travel as 16 hex digits of the
big-endian IEEE-754 float64, never as JSON numbers. Leaf i is the state *after* step i;
leaf 0 of each stage is the stage's initial latent. "Step k" means the transition from
leaf k-1 to leaf k.

The per-job salt stays in the enclave and is revealed only inside an opening, so the
public root is not a confirmation oracle for guessed latents.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .blobs import decrypt_blob, encrypt_blob
from .canonical import b64d, b64e, canonical_json, sha256_hex
from .crypto import SUITE, DecryptionError, verify_signature

VERIFIED_MODE = "verified"
# Per-element size of every dtype a latent may use.
DTYPE_SIZES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
SALT_BYTES = 32

_LATENT_TAG = b"kuno/v1/latent\n"
_LEAF_TAG = b"\x00kuno/v1/step-leaf\n"
_NODE_TAG = b"\x01kuno/v1/step-node\n"
_TRANSCRIPT_TAG = b"kuno/v1/step-transcript\n"
_OPENING_SIG_TAG = b"kuno/v1/audit-opening\n"
_OPENING_HPKE_INFO = b"kuno/v1/audit-opening"
_OPENING_EXPORT = b"kuno/v1/audit-opening-key"
_OPENING_MAGIC = b"KUNOSTEP1\n"
_TENSORS_MAGIC = b"KUNOTNS1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")

# Where a job's sealed payload may name the audit key allowed to open it (see VERIFIED_MODE.md).
AUDIT_BINDING_OPTION = "kuno_audit_key"


class VerifiedModeError(ValueError):
    """A trajectory, commitment or opening is malformed or inconsistent."""


# ------------------------------------------------------------------ floats


def f64_hex(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def f64_value(text: str) -> float:
    return struct.unpack(">d", bytes.fromhex(text))[0]


# ------------------------------------------------------------------ latents


class TensorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=32)
    dtype: str
    shape: tuple[int, ...]

    @field_validator("dtype")
    @classmethod
    def _dtype(cls, value: str) -> str:
        if value not in DTYPE_SIZES:
            raise ValueError(f"unsupported latent dtype {value!r}")
        return value

    @field_validator("shape")
    @classmethod
    def _shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(d <= 0 for d in value):
            raise ValueError("latent shapes must be non-empty and positive")
        return value

    @property
    def nbytes(self) -> int:
        n = DTYPE_SIZES[self.dtype]
        for d in self.shape:
            n *= d
        return n


Tensor = tuple[TensorSpec, bytes]


def tensor_from_array(name: str, array: Any) -> Tensor:
    """A numpy array as (spec, little-endian C-order bytes). Duck-typed: numpy is not imported."""
    dtype = array.dtype
    if dtype.name not in DTYPE_SIZES:
        raise VerifiedModeError(f"unsupported latent dtype {dtype.name}")
    if dtype.byteorder == ">":
        array = array.astype(dtype.newbyteorder("<"))
    return TensorSpec(name=name, dtype=dtype.name, shape=tuple(int(d) for d in array.shape)), array.tobytes(order="C")


def array_from_tensor(tensor: Tensor):
    """The inverse of `tensor_from_array` (needs numpy; bfloat16 has no numpy dtype)."""
    import numpy as np

    spec, data = tensor
    if spec.dtype == "bfloat16":
        raise VerifiedModeError("bfloat16 latents have no numpy dtype; use a torch executor")
    return np.frombuffer(data, dtype=np.dtype(spec.dtype).newbyteorder("<")).reshape(spec.shape)


def _sorted(tensors: Iterable[Tensor]) -> list[Tensor]:
    items = sorted(tensors, key=lambda t: t[0].name)
    names = [spec.name for spec, _ in items]
    if len(set(names)) != len(names):
        raise VerifiedModeError("latent state has duplicate tensor names")
    if not items:
        raise VerifiedModeError("latent state has no tensors")
    for spec, data in items:
        if len(data) != spec.nbytes:
            raise VerifiedModeError(f"tensor {spec.name}: {len(data)} bytes for {spec.nbytes}-byte shape")
    return items


def _tensor_header(items: list[Tensor]) -> bytes:
    return canonical_json(
        {"v": 1, "byte_order": "little", "order": "C", "tensors": [spec.model_dump(mode="json") for spec, _ in items]}
    )


def latent_digest(tensors: Iterable[Tensor]) -> str:
    """Canonical hash of one latent state (e.g. video and audio latents after a step)."""
    items = _sorted(tensors)
    header = _tensor_header(items)
    h = hashlib.sha256()
    h.update(_LATENT_TAG)
    h.update(struct.pack(">I", len(header)))
    h.update(header)
    for _, data in items:
        h.update(data)
    return h.hexdigest()


def pack_tensors(tensors: Iterable[Tensor]) -> bytes:
    items = _sorted(tensors)
    header = _tensor_header(items)
    return _TENSORS_MAGIC + struct.pack(">I", len(header)) + header + b"".join(data for _, data in items)


def unpack_tensors(blob: bytes) -> list[Tensor]:
    import json

    if not blob.startswith(_TENSORS_MAGIC) or len(blob) < len(_TENSORS_MAGIC) + 4:
        raise VerifiedModeError("not a packed latent state")
    offset = len(_TENSORS_MAGIC)
    (length,) = struct.unpack_from(">I", blob, offset)
    offset += 4
    header = json.loads(blob[offset : offset + length])
    offset += length
    out: list[Tensor] = []
    for raw in header["tensors"]:
        spec = TensorSpec.model_validate(raw)
        out.append((spec, blob[offset : offset + spec.nbytes]))
        offset += spec.nbytes
    if offset != len(blob):
        raise VerifiedModeError("packed latent state has trailing or missing bytes")
    return _sorted(out)


# ------------------------------------------------------------------ transcript


class StageTranscript(BaseModel):
    """One denoising stage: an initial latent and `len(sigmas) - 1` steps."""

    model_config = ConfigDict(extra="forbid")

    name: str
    scheduler: str
    sigmas: list[str] = Field(min_length=2)  # f64_hex, one per leaf of the stage
    tensors: list[TensorSpec]  # the latent state's layout in this stage
    settings: dict[str, Any] = Field(default_factory=dict)  # guidance (f64_hex), shift, upsampler...

    @field_validator("sigmas")
    @classmethod
    def _sigmas(cls, value: list[str]) -> list[str]:
        if not all(_HEX16.match(s) for s in value):
            raise ValueError("sigmas are 16 hex digits of a big-endian float64")
        return value

    @field_validator("tensors")
    @classmethod
    def _canonical_order(cls, value: list[TensorSpec]) -> list[TensorSpec]:
        # Sorted by name like the latent digest, so any producer yields the same transcript digest.
        return sorted(value, key=lambda spec: spec.name)

    @property
    def steps(self) -> int:
        return len(self.sigmas) - 1


class StepTranscript(BaseModel):
    """Everything a re-executor needs besides the conditioning inputs themselves.

    Stays inside the enclave and is revealed only in an encrypted opening; the receipt
    carries its digest.
    """

    model_config = ConfigDict(extra="forbid")

    v: Literal[1] = 1
    mode: Literal["verified"] = "verified"
    job_id: str
    params_digest: str
    profile_id: str
    family: str
    runtime: str  # e.g. "kuno-toy-denoiser/1", "diffusers-ltx2/1", "diffusers-modular-h3/1"
    model_digest: str  # identity of the weights actually loaded
    hardware_class: str
    seed: int
    noise: str  # how the initial latent derives from the seed
    conditioning_digest: str  # latent_digest of the encoded conditioning (prompt embeddings…)
    stages: list[StageTranscript] = Field(min_length=1)
    determinism: dict[str, Any] = Field(default_factory=dict)


def transcript_digest(transcript: StepTranscript) -> str:
    return sha256_hex(_TRANSCRIPT_TAG + canonical_json(transcript.model_dump(mode="json")))


class LeafSlot(BaseModel):
    index: int
    stage: int
    kind: Literal["init", "denoise"]
    sigma: str


def expected_layout(transcript: StepTranscript) -> list[LeafSlot]:
    slots: list[LeafSlot] = []
    for s, stage in enumerate(transcript.stages):
        for i, sigma in enumerate(stage.sigmas):
            slots.append(LeafSlot(index=len(slots), stage=s, kind="init" if i == 0 else "denoise", sigma=sigma))
    return slots


# ------------------------------------------------------------------ leaves and tree


class StepLeaf(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    stage: int = Field(ge=0)
    kind: Literal["init", "denoise"]
    sigma: str
    latent: str  # latent_digest

    @field_validator("latent")
    @classmethod
    def _hex(cls, value: str) -> str:
        if not _HEX64.match(value):
            raise ValueError("latent digest must be 64 lowercase hex digits")
        return value


def leaf_hash(leaf: StepLeaf, salt: bytes) -> bytes:
    if len(salt) != SALT_BYTES:
        raise VerifiedModeError("salt must be 32 bytes")
    return hashlib.sha256(_LEAF_TAG + salt + canonical_json(leaf.model_dump(mode="json"))).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE_TAG + left + right).digest()


def _split(n: int) -> int:
    return 1 << ((n - 1).bit_length() - 1)


def merkle_root(hashes: Sequence[bytes]) -> bytes:
    n = len(hashes)
    if n == 0:
        raise VerifiedModeError("a step tree needs at least one leaf")
    if n == 1:
        return hashes[0]
    k = _split(n)
    return node_hash(merkle_root(hashes[:k]), merkle_root(hashes[k:]))


def inclusion_proof(hashes: Sequence[bytes], index: int) -> list[bytes]:
    """RFC 9162 §2.1.3.1 audit path for leaf `index`, leaf-most sibling first."""
    n = len(hashes)
    if not 0 <= index < n:
        raise VerifiedModeError("leaf index out of range")
    if n == 1:
        return []
    k = _split(n)
    if index < k:
        return inclusion_proof(hashes[:k], index) + [merkle_root(hashes[k:])]
    return inclusion_proof(hashes[k:], index - k) + [merkle_root(hashes[:k])]


def verify_inclusion(root: bytes, leaf: bytes, index: int, size: int, proof: Sequence[bytes]) -> bool:
    """RFC 9162 §2.1.3.2."""
    if not 0 <= index < size:
        return False
    fn, sn, r = index, size - 1, leaf
    for p in proof:
        if len(p) != 32 or sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = node_hash(p, r)
            if not fn & 1:
                while not fn & 1 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            r = node_hash(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


# ------------------------------------------------------------------ commitment


class StepCommitment(BaseModel):
    """Signed inside the receipt. Integers and strings only, so every canonical JSON agrees."""

    model_config = ConfigDict(extra="forbid")

    v: Literal[1] = 1
    mode: Literal["verified"] = "verified"
    root: str
    leaves: int = Field(ge=2)
    steps: int = Field(ge=1)
    latent_shape: list[int]
    dtype: str
    hardware_class: str
    transcript_digest: str

    @field_validator("root", "transcript_digest")
    @classmethod
    def _hex(cls, value: str) -> str:
        if not _HEX64.match(value):
            raise ValueError("must be 64 lowercase hex digits")
        return value


def check_leaves(transcript: StepTranscript, leaves: Sequence[StepLeaf]) -> None:
    layout = expected_layout(transcript)
    if len(leaves) != len(layout):
        raise VerifiedModeError(f"trajectory has {len(leaves)} leaves; the transcript schedules {len(layout)}")
    for leaf, slot in zip(leaves, layout):
        if (leaf.index, leaf.stage, leaf.kind, leaf.sigma) != (slot.index, slot.stage, slot.kind, slot.sigma):
            raise VerifiedModeError(f"leaf {slot.index} does not match the transcript's schedule")


def _primary(stage: StageTranscript) -> TensorSpec:
    return next((t for t in stage.tensors if t.name == "video"), stage.tensors[0])


def build_commitment(transcript: StepTranscript, leaves: Sequence[StepLeaf], salt: bytes) -> tuple[StepCommitment, list[bytes]]:
    check_leaves(transcript, leaves)
    hashes = [leaf_hash(leaf, salt) for leaf in leaves]
    final = _primary(transcript.stages[-1])
    commitment = StepCommitment(
        root=merkle_root(hashes).hex(),
        leaves=len(leaves),
        steps=sum(1 for leaf in leaves if leaf.kind == "denoise"),
        latent_shape=list(final.shape),
        dtype=final.dtype,
        hardware_class=transcript.hardware_class,
        transcript_digest=transcript_digest(transcript),
    )
    return commitment, hashes


def new_salt() -> bytes:
    return os.urandom(SALT_BYTES)


# ------------------------------------------------------------------ audits


def audit_binding(recipient_public_key: bytes) -> str:
    """What a sealed payload names under AUDIT_BINDING_OPTION to pre-authorize an audit key."""
    return sha256_hex(b"kuno/v1/audit-binding\n" + recipient_public_key)


class AuditRequest(BaseModel):
    """Validator → gateway: open step `step` of one of this validator's own jobs."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    step: int = Field(ge=1)
    # X25519 public key (32 bytes, base64url) the opening is sealed to; fresh per audit.
    recipient_public_key: str
    # Also reveal every leaf (digests only, no latents): full re-runs and golden-set checks.
    include_leaves: bool = False

    @field_validator("recipient_public_key")
    @classmethod
    def _key(cls, value: str) -> str:
        try:
            raw = b64d(value)
        except ValueError:
            raise ValueError("recipient_public_key must be base64url") from None
        if len(raw) != 32:
            raise ValueError("recipient_public_key must be a 32-byte X25519 key")
        return value


class MinerAudit(BaseModel):
    """Gateway → enclave work item, delivered like a challenge."""

    kind: Literal["audit"] = "audit"
    audit_id: str
    job_id: str
    step: int
    recipient_public_key: str
    include_leaves: bool = False
    expires_at: float


class LeafProof(BaseModel):
    index: int
    path: list[str]


class LatentRecord(BaseModel):
    index: int
    tensors: list[TensorSpec]


class StepOpening(BaseModel):
    """The decrypted header of an opening; latent bytes follow it in the plaintext."""

    model_config = ConfigDict(extra="forbid")

    v: Literal[1] = 1
    audit_id: str
    job_id: str
    enclave_id: str
    step: int
    commitment: StepCommitment
    transcript: StepTranscript
    salt: str
    leaves: list[StepLeaf]
    proofs: list[LeafProof]
    latents: list[LatentRecord]


class SealedOpening(BaseModel):
    """Enclave → gateway → validator. HPKE to the validator's key, signed by the enclave."""

    model_config = ConfigDict(extra="forbid")

    v: Literal[1] = 1
    audit_id: str
    job_id: str
    enclave_id: str
    step: int
    recipient_public_key: str
    enc: str
    ciphertext: str
    ciphertext_sha256: str
    signature: str = ""

    def signed_fields(self) -> dict:
        return self.model_dump(mode="json", exclude={"signature", "ciphertext"})


def opening_signature_message(sealed: SealedOpening) -> bytes:
    return _OPENING_SIG_TAG + canonical_json(sealed.signed_fields())


def _opening_label(job_id: str, audit_id: str, step: int) -> str:
    return f"{job_id}/audit/{audit_id}/{step}"


def seal_opening(
    signing_key, opening: StepOpening, latents: dict[int, list[Tensor]], recipient_public_key: bytes
) -> SealedOpening:
    """Frames the header and latent bytes, seals them to the validator, signs the envelope."""
    records = {r.index: r for r in opening.latents}
    if set(records) != set(latents):
        raise VerifiedModeError("latent records and latent bytes disagree")
    header = canonical_json(opening.model_dump(mode="json"))
    parts = [_OPENING_MAGIC, struct.pack(">I", len(header)), header]
    for record in opening.latents:
        items = _sorted(latents[record.index])
        if [spec for spec, _ in items] != list(record.tensors):
            raise VerifiedModeError(f"latent record {record.index} does not describe its bytes")
        parts.extend(data for _, data in items)
    plaintext = b"".join(parts)

    public_key = SUITE.kem.deserialize_public_key(recipient_public_key)
    enc, ctx = SUITE.create_sender_context(public_key, info=_OPENING_HPKE_INFO)
    ciphertext = encrypt_blob(ctx.export(_OPENING_EXPORT, 32), _opening_label(opening.job_id, opening.audit_id, opening.step), plaintext)
    sealed = SealedOpening(
        audit_id=opening.audit_id,
        job_id=opening.job_id,
        enclave_id=opening.enclave_id,
        step=opening.step,
        recipient_public_key=b64e(recipient_public_key),
        enc=b64e(enc),
        ciphertext=b64e(ciphertext),
        ciphertext_sha256=sha256_hex(ciphertext),
    )
    sealed.signature = b64e(signing_key.sign(opening_signature_message(sealed)))
    return sealed


def verify_sealed_opening(sealed: SealedOpening, signing_public_key: bytes) -> bool:
    try:
        ciphertext = b64d(sealed.ciphertext)
        signature = b64d(sealed.signature)
    except ValueError:
        return False
    if sha256_hex(ciphertext) != sealed.ciphertext_sha256:
        return False
    return verify_signature(signing_public_key, signature, opening_signature_message(sealed))


def open_sealed_opening(recipient_private_key: bytes, sealed: SealedOpening) -> tuple[StepOpening, dict[int, list[Tensor]]]:
    """Decrypts and parses an opening. Raises DecryptionError or VerifiedModeError."""
    try:
        private_key = SUITE.kem.deserialize_private_key(recipient_private_key)
        ctx = SUITE.create_recipient_context(b64d(sealed.enc), private_key, info=_OPENING_HPKE_INFO)
    except Exception as exc:  # pyhpke raises several types for a bad encapsulation
        raise DecryptionError("invalid encapsulated key") from exc
    plaintext = decrypt_blob(
        ctx.export(_OPENING_EXPORT, 32), _opening_label(sealed.job_id, sealed.audit_id, sealed.step), b64d(sealed.ciphertext)
    )
    if not plaintext.startswith(_OPENING_MAGIC):
        raise VerifiedModeError("not a step opening")
    offset = len(_OPENING_MAGIC)
    (length,) = struct.unpack_from(">I", plaintext, offset)
    offset += 4
    try:
        opening = StepOpening.model_validate_json(plaintext[offset : offset + length])
    except ValueError as exc:
        raise VerifiedModeError("malformed opening header") from exc
    offset += length
    latents: dict[int, list[Tensor]] = {}
    for record in opening.latents:
        tensors = []
        for spec in record.tensors:
            tensors.append((spec, plaintext[offset : offset + spec.nbytes]))
            offset += spec.nbytes
        latents[record.index] = tensors
    if offset != len(plaintext):
        raise VerifiedModeError("opening has trailing or missing latent bytes")
    if (opening.audit_id, opening.job_id, opening.enclave_id, opening.step) != (
        sealed.audit_id, sealed.job_id, sealed.enclave_id, sealed.step,
    ):
        raise VerifiedModeError("opening header does not match its envelope")
    return opening, latents


def required_leaves(step: int, leaves: int, include_leaves: bool) -> list[int]:
    if include_leaves:
        return list(range(leaves))
    return sorted({0, step - 1, step})


def verify_opening(
    commitment: StepCommitment,
    opening: StepOpening,
    latents: dict[int, list[Tensor]],
    *,
    job_id: str,
    step: int,
    include_leaves: bool = False,
) -> str | None:
    """Checks an opening against the commitment signed in the receipt. None means it holds.

    Establishes: the transcript is the committed one; the revealed leaves sit at their
    positions under the signed root; the latents before and after `step` hash to those
    leaves and have the transcript's layout. It does not re-execute anything.
    """
    if opening.commitment != commitment:
        return "opening carries a different commitment than the signed receipt"
    transcript = opening.transcript
    if transcript_digest(transcript) != commitment.transcript_digest:
        return "transcript does not match the committed transcript digest"
    if transcript.job_id != job_id or opening.job_id != job_id:
        return "opening belongs to a different job"
    if transcript.hardware_class != commitment.hardware_class:
        return "transcript hardware class differs from the commitment"
    layout = expected_layout(transcript)
    if len(layout) != commitment.leaves or sum(1 for s in layout if s.kind == "denoise") != commitment.steps:
        return "transcript schedule does not match the committed leaf count"
    final = _primary(transcript.stages[-1])
    if list(final.shape) != commitment.latent_shape or final.dtype != commitment.dtype:
        return "transcript latent layout does not match the commitment"
    if opening.step != step or not 1 <= step < commitment.leaves:
        return f"opening is for step {opening.step}, not {step}"
    try:
        salt = bytes.fromhex(opening.salt)
    except ValueError:
        return "malformed salt"
    if len(salt) != SALT_BYTES:
        return "malformed salt"

    leaves = {leaf.index: leaf for leaf in opening.leaves}
    proofs = {proof.index: proof for proof in opening.proofs}
    root = bytes.fromhex(commitment.root)
    for index in required_leaves(step, commitment.leaves, include_leaves):
        leaf, proof = leaves.get(index), proofs.get(index)
        if leaf is None or proof is None:
            return f"opening omits leaf {index}"
        slot = layout[index]
        if (leaf.index, leaf.stage, leaf.kind, leaf.sigma) != (slot.index, slot.stage, slot.kind, slot.sigma):
            return f"leaf {index} does not match the transcript's schedule"
        try:
            path = [bytes.fromhex(p) for p in proof.path]
        except ValueError:
            return f"malformed proof for leaf {index}"
        if not verify_inclusion(root, leaf_hash(leaf, salt), index, commitment.leaves, path):
            return f"inclusion proof for leaf {index} does not verify against the signed root"

    for index in (step - 1, step):
        tensors = latents.get(index)
        if tensors is None:
            return f"opening omits the latent for leaf {index}"
        expected = sorted(transcript.stages[layout[index].stage].tensors, key=lambda t: t.name)
        if [spec for spec, _ in sorted(tensors, key=lambda t: t[0].name)] != expected:
            return f"latent {index} does not have the transcript's layout"
        try:
            digest = latent_digest(tensors)
        except VerifiedModeError as exc:
            return f"latent {index}: {exc}"
        if digest != leaves[index].latent:
            return f"latent {index} does not hash to its committed leaf"
    return None
