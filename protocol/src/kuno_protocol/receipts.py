"""Enclave-signed receipts: the proof of work for validators and the provenance
certificate for customers.

A receipt contains no content, only digests. `content_digest` is the SHA-256 of
the decrypted MP4, so anyone holding the video can look up and verify where it
came from. A plan job's receipt describes its plan instead (`plan`, no `video`):
`content_digest` is then the SHA-256 of the unpadded plan JSON.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_serializer, model_validator

from .canonical import b64d, b64e, canonical_json, sha256_hex
from .crypto import verify_signature
from .verified import StepCommitment


class VideoInfo(BaseModel):
    duration_s: float
    width: int
    height: int
    fps: float
    frames: int
    audio: bool


class PlanInfo(BaseModel):
    """What a plan job delivered (kuno_protocol.plans): its shot count and stitched length, the loaded component that
    wrote it and the system prompt version, and the tokens generated for the job, retries included."""

    model_config = ConfigDict(extra="forbid")

    shots: int
    duration_s: float
    planner: str
    prompt_version: str
    output_tokens: int


class ReceiptBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    v: Literal[1] = 1
    job_id: str
    enclave_id: str
    profile_id: str
    image_digest: str
    params_digest: str
    input_digest: str
    output_digest: str
    output_bytes: int
    content_digest: str
    attestation_digest: str
    started_at: float
    finished_at: float
    gpu_seconds: float
    # A video job's output. A plan job has none and carries `plan` instead; exactly one of the two is set.
    video: VideoInfo | None = None
    miner_hotkey: str | None = None
    # Verified mode (see VERIFIED_MODE.md): the Merkle root over per-step latents, signed with
    # the rest of the body. When absent the key is left out of every encoding, so a receipt
    # without it serializes, and therefore signs and verifies, exactly as before it existed.
    step_commitment: StepCommitment | None = None
    # A plan job's output (PROTOCOL.md "Plans (Director)"). Left out when absent, as `step_commitment` is, and `video` is
    # left out when absent too, so every video receipt signs and verifies byte-for-byte as before plans.
    plan: PlanInfo | None = None

    @model_validator(mode="after")
    def _one_output(self) -> ReceiptBody:
        if (self.video is None) == (self.plan is None):
            raise ValueError("a receipt describes exactly one output: a video or a plan")
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_commitment(self, handler) -> dict[str, Any]:
        data = handler(self)
        if isinstance(data, dict):
            for key in ("step_commitment", "video", "plan"):
                if data.get(key) is None:
                    data.pop(key, None)
        return data


class Receipt(BaseModel):
    body: ReceiptBody
    signature: str

    def message(self) -> bytes:
        return receipt_message(self.body)


def receipt_message(body: ReceiptBody) -> bytes:
    return b"kuno/v1/receipt\n" + canonical_json(body.model_dump(mode="json"))


def sign_receipt(signing_key, body: ReceiptBody) -> Receipt:
    return Receipt(body=body, signature=b64e(signing_key.sign(receipt_message(body))))


def verify_receipt(receipt: Receipt, signing_public_key: bytes) -> bool:
    return verify_signature(signing_public_key, b64d(receipt.signature), receipt_message(receipt.body))


def input_digest(enc: bytes, ciphertext: bytes, input_blobs: list[bytes]) -> str:
    return sha256_hex(
        canonical_json(
            {"enc": sha256_hex(enc), "ciphertext": sha256_hex(ciphertext), "inputs": [sha256_hex(b) for b in input_blobs]}
        )
    )
