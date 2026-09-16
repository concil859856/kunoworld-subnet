"""Request, job and wire schemas.

Two layers:
  * GenerationParams is public: the gateway sees it to price, route and validate.
    It is bound into the HPKE AAD, so nobody between the client and the enclave
    can change it without decryption failing.
  * SealedPayload is private: prompt, seed, input manifest and model options.
    Only the attested enclave can read it.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer

from .attestation import AttestationEvidence
from .canonical import canonical_json
from .hotkey import HotkeyProof
from .location import LocationProof
from .profiles import InputRole, Mode
from .receipts import Receipt

JOB_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED)


# A private job is encrypted end to end and runs only on confidential miners; a standard job is readable by
# the platform and the GPU provider and may run on any miner. Kept out of GenerationParams on purpose: the
# params are the encryption's associated data, and existing clients must keep producing the same bytes.
PrivacyMode = Literal["private", "standard"]


# How a storyboard shot attaches to the one before it (PROTOCOL.md, "Storyboards"): `continue` carries the previous
# shot's last latent frames and its sound into this one (one unbroken take), `cut` carries only the sound (a new picture,
# the same voice and room tone), `fresh` carries nothing. The first shot is always `fresh`.
ShotJoin = Literal["fresh", "continue", "cut"]


class ShotSpec(BaseModel):
    """One storyboard shot as the gateway sees it: its length and its join. Its prompt is sealed (`SealedPayload.shots`)."""

    model_config = ConfigDict(extra="forbid")

    duration_s: float
    join: ShotJoin


class GenerationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    mode: Mode
    # For a storyboard: the stitched video's length, `profiles.storyboard_duration_s` of its shots, exactly.
    duration_s: float
    resolution: str
    aspect_ratio: str
    fps: int
    audio: bool = True
    input_roles: list[InputRole] = Field(default_factory=list)
    # Storyboard mode only. Serialized only when set, so every other job's params (the encryption's associated data, and
    # the receipt's params_digest) stay byte-identical to clients that predate storyboards. The bound is a parser limit;
    # a profile's `limits.storyboard.max_shots` is the real one (`validate_params`).
    shots: list[ShotSpec] | None = Field(default=None, max_length=64)

    @model_serializer(mode="wrap")
    def _omit_absent_shots(self, handler) -> dict[str, Any]:
        data = handler(self)
        if isinstance(data, dict) and data.get("shots") is None:
            data.pop("shots", None)
        return data

    @property
    def render_duration_s(self) -> float:
        """The longest single model call the job needs: its duration, or a storyboard's longest shot. Memory admission and
        serving envelopes use this; price and billing use `duration_s`."""
        return max(shot.duration_s for shot in self.shots) if self.shots else self.duration_s


class InputRef(BaseModel):
    """Describes one encrypted input blob, by position in JobCreate.input_blob_ids."""

    model_config = ConfigDict(extra="forbid")

    index: int
    role: InputRole
    mime: str
    sha256: str
    size: int
    # Keyframes: where the image lands (seconds from the start) and how strongly it conditions.
    time_s: float | None = None
    strength: float | None = None
    # References: what the input is for (subject, style, scene, motion, voice, music, source_edit, source_continue).
    hint: str | None = None
    # Source clips: the window to use, in seconds.
    start_s: float | None = None
    end_s: float | None = None


class ShotPrompt(BaseModel):
    """A storyboard shot's private half: what happens in it. Paired by position with `GenerationParams.shots`."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)

    @field_validator("prompt")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        # `shot_prompt` strips it, so a whitespace-only shot would reach the model as an empty prompt.
        if not value.strip():
            raise ValueError("a shot prompt can't be blank")
        return value


class SealedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    v: Literal[1] = 1
    # For a storyboard: the scene every shot shares (characters, place, style), put before each shot's own prompt. It may
    # be empty there; `profiles.shot_prompt` builds what the model sees.
    prompt: str
    negative_prompt: str | None = None
    seed: int | None = None
    inputs: list[InputRef] = Field(default_factory=list)
    # Model-specific knobs (camera motion, guidance, prompt enhancement...).
    options: dict[str, Any] = Field(default_factory=dict)
    # Storyboard mode only, one per `GenerationParams.shots`, in order. Serialized only when set.
    shots: list[ShotPrompt] | None = Field(default=None, max_length=64)

    @model_serializer(mode="wrap")
    def _omit_absent_shots(self, handler) -> dict[str, Any]:
        data = handler(self)
        if isinstance(data, dict) and data.get("shots") is None:
            data.pop("shots", None)
        return data


class JobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    params: GenerationParams
    enclave_id: str
    enc: str
    ciphertext: str
    input_blob_ids: list[str] = Field(default_factory=list)
    webhook_url: str | None = None

    @field_validator("job_id")
    @classmethod
    def _uuid4(cls, value: str) -> str:
        if not JOB_ID_RE.match(value):
            raise ValueError("job_id must be a lowercase UUIDv4")
        return value


class JobStatus(BaseModel):
    job_id: str
    status: JobState
    stage: str | None = None
    progress: float = 0.0
    params: GenerationParams
    enclave_id: str
    price_usd: float
    created_at: float
    updated_at: float
    output_blob_id: str | None = None
    receipt: Receipt | None = None
    # Machine-readable failure reason (e.g. safety_blocked, timeout); `error` is the human message.
    error_code: str | None = None
    error: str | None = None
    privacy: PrivacyMode = "private"


class MinerJob(BaseModel):
    kind: Literal["job"] = "job"
    job_id: str
    params: GenerationParams
    enc: str
    ciphertext: str
    input_blob_ids: list[str]


class MinerChallenge(BaseModel):
    kind: Literal["challenge"] = "challenge"
    challenge_id: str
    nonce: str


class MinerRegistration(BaseModel):
    """Body of `POST /miner/v1/enclaves`.

    `hotkey_proof` is optional so older workers still parse; a production gateway requires it
    and checks it with `verify_hotkey_proof` against the verified evidence.

    `envelope` is the serving envelope (kuno_protocol.envelope): profile id -> resolution -> aspect
    ratio -> fps -> the longest duration_s this hardware serves. Only profiles the hardware cannot
    serve in full are listed; None (older workers, and hardware that holds every profile) serves the
    profiles' full limits. Gateways from before it ignore the field.
    """

    evidence: AttestationEvidence
    miner_hotkey: str | None = None
    capacity: int = Field(default=1, ge=1, le=64)
    hotkey_proof: HotkeyProof | None = None
    envelope: dict[str, dict[str, dict[str, dict[int, float]]]] | None = None
    # Signed landmark round trips for profiles whose licence is bound to territory (kuno_protocol.location).
    location: LocationProof | None = None


class RouteResponse(BaseModel):
    profile_id: str
    requested_profile_id: str | None
    fallback_reason: str | None
    enclaves: list[dict[str, Any]]


def job_aad(job_id: str, enclave_id: str, params: GenerationParams, input_blob_ids: list[str]) -> bytes:
    """Associated data for the HPKE seal. Mirrors `jobAad` in the JS SDK."""
    return canonical_json(
        {
            "v": 1,
            "job_id": job_id,
            "enclave_id": enclave_id,
            "params": params.model_dump(mode="json"),
            "inputs": list(input_blob_ids),
        }
    )


def input_label(job_id: str, index: int) -> str:
    return f"{job_id}/input/{index}"


def output_label(job_id: str) -> str:
    return f"{job_id}/output/video"
