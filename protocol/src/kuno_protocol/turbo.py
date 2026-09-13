"""The Turbo track: an owner-signed competition for faster attested pipelines.

KunoWorld runs two Bittensor incentive mechanisms:

  mechanism 0  serving   pays attested organic work (validator `scoring.py`)
  mechanism 1  Turbo     pays miners whose pipeline for a target profile is faster than the
                         incumbent at a guaranteed quality floor; the winner is adopted as a
                         new, separately pinned profile the whole network then serves

The owner signs a `TurboSpec` (like the model switch). Miners sign a `TurboSubmission` with
their hotkey, host the document anywhere, and publish only its digest as their on-chain
commitment (`Commitments.set_commitment`), so there is no gateway database of entries and
the chain orders submissions. Validators read the spec and the commitments, fetch each
document, check it against the committed digest and the hotkey signature, then benchmark
enclaves that attest exactly the submitted image.

Signing:
  spec        Ed25519 (owner key) over "kuno/v1/turbo-spec\\n" | canonical_json(TurboSpec)
  submission  sr25519 (miner hotkey) over "kuno/v1/turbo-submission\\n" | canonical_json(TurboSubmission)
  digest      SHA-256 of canonical_json(SignedTurboSubmission), committed on chain as
              "kt1:" + b64url(digest) [+ "@" + location], at most 128 bytes (a Raw128 field)

Eval sets are hidden and rotate per window: the spec carries SHA-256 commitments of each
window's salted prompt set, validators receive the set privately, and it is revealed once
the window ends so anyone can recompute the scores.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .attestation import AllowedMeasurement, GoldenManifest, TeeKind
from .canonical import b64d, b64e, canonical_json, sha256_hex
from .crypto import verify_signature
from .hotkey import _POLKADOT_WRAP, BITTENSOR_SS58_FORMAT, HotkeyError, HotkeySigner, _sr25519, ss58_decode

SPEC_CONTEXT = b"kuno/v1/turbo-spec\n"
SUBMISSION_CONTEXT = b"kuno/v1/turbo-submission\n"
EVAL_SET_CONTEXT = b"kuno/v1/turbo-eval-set\n"
COMMITMENT_PREFIX = "kt1:"
# The Commitments pallet's largest plain field is Raw128.
MAX_COMMITMENT_BYTES = 128
MAX_SUBMISSION_BYTES = 64 * 1024
# Profiles an enclave registered as a Turbo candidate is stored under at the gateway. No real
# profile id contains a colon, so customer routing can never select a candidate enclave.
CANDIDATE_PROFILE_PREFIX = "turbo:"

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,23}$")
_VARIANT = re.compile(r"^[a-z0-9][a-z0-9.+-]{0,63}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX96 = re.compile(r"^[0-9a-f]{96}$")


class TurboError(ValueError):
    """A spec, submission, eval set or commitment is malformed or does not verify."""


# ---------------------------------------------------------------- spec


class BaseMeasurements(BaseModel):
    """The owner's measured CVM base (firmware, VM shape, kernel, initrd). Only RTMR3, the
    application layer, comes from a submission, so every candidate boots the same audited base
    with its egress policy and cannot, for example, send hidden prompts anywhere but the gateway."""

    model_config = ConfigDict(extra="forbid")

    platform: TeeKind
    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str


class QualityFloor(BaseModel):
    """The quality a pipeline must keep. Every configured bound must hold.

    `metric` names a pluggable prompt-alignment metric every validator runs (see
    `kuno_validator.turbo_quality`): "clip", "xclip", "vlm-judge", or "dev-caption" on dev networks.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    # Mean score over verified samples in the window.
    min_mean: float | None = None
    # Per-sample floor, with a tolerated fraction of samples below it.
    min_sample: float | None = None
    max_below_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    # Relative floor: the mean may be at most this far below the reference profile's mean on
    # the same hidden prompts in the same window.
    max_drop_vs_reference: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _some_bound(self) -> QualityFloor:
        if self.min_mean is None and self.min_sample is None and self.max_drop_vs_reference is None:
            raise ValueError("a quality floor needs min_mean, min_sample or max_drop_vs_reference")
        return self


class SpeedMetric(BaseModel):
    """What is measured, at which fixed request shape.

    `wall_s_per_output_s`  enclave wall-clock seconds per second of requested video
    `gpu_s_per_output_s`   the same times the number of GPUs the enclave attested
    Latency is max(receipt interval, gateway pull-to-complete interval): the candidate image signs
    its own receipts, so its claimed interval is only accepted inside what the gateway observed.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["wall_s_per_output_s", "gpu_s_per_output_s"] = "wall_s_per_output_s"
    resolution: str
    aspect_ratio: str = "16:9"
    durations_s: list[float] = Field(min_length=1)
    statistic: Literal["median", "mean", "p90"] = "p90"
    # The incumbent pipeline's measured value on the pinned hardware class (owner golden run).
    baseline: float = Field(gt=0)
    # A submission must be at least this many times faster than the baseline to earn.
    min_speedup: float = Field(default=1.1, ge=1.0)
    # Clock slack when a receipt's interval is checked against the gateway's.
    timing_slack_s: float = Field(default=2.0, ge=0.0)


class SamplingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobs_per_window: int = Field(default=16, ge=1)
    min_samples: int = Field(default=12, ge=1)
    # Miner-caused failures (timeouts, crashes, invalid output) over finished benchmark jobs.
    max_failure_rate: float = Field(default=0.1, ge=0.0, le=1.0)


class RewardCurve(BaseModel):
    """Winner-take-most over eligible submissions, ranked by speed.

    A later commitment only ranks above an earlier one if it is faster by more than
    `displace_margin`, so copying a pipeline and shaving noise off its timing earns nothing.
    Submissions committed in the same block and within the margin of each other split the
    shares of the places they occupy. Weights are averaged over the last `smoothing_windows`
    finalized windows with missing windows counted as zero, so a gap never helps.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["podium", "exponential"] = "podium"
    shares: list[float] = Field(default_factory=lambda: [0.7, 0.2, 0.1])
    decay: float = Field(default=0.5, gt=0.0, lt=1.0)
    top_k: int = Field(default=5, ge=1)
    displace_margin: float = Field(default=0.03, ge=0.0)
    smoothing_windows: int = Field(default=3, ge=1)
    # With no eligible submission: "serving" copies the serving weights into mechanism 1, so
    # emission still pays useful work (never burn); "hold" submits nothing.
    empty_policy: Literal["serving", "hold"] = "serving"

    @field_validator("shares")
    @classmethod
    def _shares(cls, value: list[float]) -> list[float]:
        if not value or any(s < 0 for s in value) or sum(value) <= 0:
            raise ValueError("podium shares must be non-negative with a positive sum")
        return value

    def place_shares(self, places: int) -> list[float]:
        """Unnormalized share for each of the first `places` ranks (zero past the podium)."""
        if self.kind == "podium":
            return [self.shares[i] if i < len(self.shares) else 0.0 for i in range(places)]
        return [self.decay**i if i < self.top_k else 0.0 for i in range(places)]


class AdoptionRule(BaseModel):
    """When the owner adopts a winner and under which new profile id it is pinned."""

    model_config = ConfigDict(extra="forbid")

    # The new profile id the adopted pipeline serves under; never the target profile's own id.
    profile_id: str
    min_windows_leading: int = Field(default=2, ge=1)
    min_speedup: float = Field(default=1.15, ge=1.0)
    # Serving-emission ramp for the new profile after adoption (research_models §4.2).
    overlap_days: int = Field(default=14, ge=0)


class EvalWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    starts_at: int
    ends_at: int
    # sha256 hex of the window's eval set (`eval_set_digest`).
    eval_set_commitment: str
    # Unix time after which the set is published for audit (default: the window's end).
    reveal_at: int | None = None

    @field_validator("eval_set_commitment")
    @classmethod
    def _hex(cls, value: str) -> str:
        if not _HEX64.match(value):
            raise ValueError("eval_set_commitment must be 64 lowercase hex characters")
        return value

    @property
    def reveals_at(self) -> int:
        return self.reveal_at if self.reveal_at is not None else self.ends_at


class TurboSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    competition_id: str
    issued_at: int = Field(default_factory=lambda: int(time.time()))
    netuid: int | None = None
    mechid: int = Field(default=1, ge=1)
    # The profile whose pipeline is being optimized; benchmark jobs use exactly its public params.
    target_profile: str
    # The profile the quality floor is measured against, and the manifest images that serve it.
    reference_profile: str
    reference_image_digests: list[str] = Field(default_factory=list)
    base_measurements: list[BaseMeasurements] = Field(min_length=1)
    hardware_class: str
    max_gpus: int = Field(default=1, ge=1)
    quality: QualityFloor
    speed: SpeedMetric
    sampling: SamplingRule = Field(default_factory=SamplingRule)
    curve: RewardCurve = Field(default_factory=RewardCurve)
    adoption: AdoptionRule
    windows: list[EvalWindow] = Field(min_length=1)
    # Commitments made after this block are ignored (None: open until the last window ends).
    submissions_close_block: int | None = None
    # Where validators look for a submission whose commitment names no location; "{digest}"
    # is replaced with the hex digest. E.g. "https://turbo.kunoworld.com/submissions/{digest}.json".
    submission_locations: list[str] = Field(default_factory=list)

    @field_validator("competition_id")
    @classmethod
    def _competition_id(cls, value: str) -> str:
        if not _ID.match(value):
            raise ValueError("competition_id is 1-24 characters of a-z, 0-9 and '-'")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> TurboSpec:
        if self.adoption.profile_id in (self.target_profile, self.reference_profile):
            raise ValueError("an adopted pipeline needs its own profile id, not the target's or the reference's")
        indices = [w.index for w in self.windows]
        if indices != sorted(set(indices)):
            raise ValueError("windows must have unique, increasing indices")
        for previous, window in zip(self.windows, self.windows[1:]):
            if window.starts_at < previous.ends_at:
                raise ValueError("windows must not overlap")
        for window in self.windows:
            if window.ends_at <= window.starts_at or window.reveals_at < window.ends_at:
                raise ValueError(f"window {window.index} must end after it starts and reveal after it ends")
        if self.sampling.min_samples > self.sampling.jobs_per_window:
            raise ValueError("min_samples cannot exceed jobs_per_window")
        return self

    def window_at(self, now: float) -> EvalWindow | None:
        return next((w for w in self.windows if w.starts_at <= now < w.ends_at), None)

    def window(self, index: int) -> EvalWindow:
        for window in self.windows:
            if window.index == index:
                return window
        raise TurboError(f"competition {self.competition_id} has no window {index}")

    def ended_windows(self, now: float) -> list[EvalWindow]:
        return [w for w in self.windows if w.ends_at <= now]

    def candidate_manifest(self, submission: TurboSubmission, production: GoldenManifest) -> GoldenManifest:
        """The only measurements a candidate enclave for `submission` may attest: the owner's
        base layers plus the submitted application layer, for the target profile only.

        Simulated-TEE keys and evidence age come from the production manifest, so a candidate is
        held to exactly the attestation rules serving enclaves are.
        """
        allowed = [
            AllowedMeasurement(
                platform=base.platform,
                image_digest=submission.image_digest,
                profiles=[self.target_profile],
                mrtd=base.mrtd,
                rtmr0=base.rtmr0,
                rtmr1=base.rtmr1,
                rtmr2=base.rtmr2,
                rtmr3=submission.rtmr3,
            )
            for base in self.base_measurements
            if base.platform == submission.platform
        ]
        return GoldenManifest(
            version=production.version,
            issued_at=production.issued_at,
            allowed=allowed,
            mock_quote_keys=list(production.mock_quote_keys) if any(a.platform == "mock" for a in allowed) else [],
            max_evidence_age_s=production.max_evidence_age_s,
        )


class SignedTurboSpec(BaseModel):
    spec: TurboSpec
    signature: str | None = None

    def verify(self, owner_public_key: bytes) -> bool:
        try:
            signature = b64d(self.signature) if self.signature else None
        except ValueError:
            return False
        return signature is not None and verify_signature(owner_public_key, signature, turbo_spec_message(self.spec))


def turbo_spec_message(spec: TurboSpec) -> bytes:
    return SPEC_CONTEXT + canonical_json(spec.model_dump(mode="json"))


def sign_turbo_spec(owner_key, spec: TurboSpec) -> SignedTurboSpec:
    return SignedTurboSpec(spec=spec, signature=b64e(owner_key.sign(turbo_spec_message(spec))))


def newer_spec(current: SignedTurboSpec | None, candidate: SignedTurboSpec) -> bool:
    """Monotonic acceptance, as for the switch: never an older spec, never a different one at the same time."""
    if current is None:
        return True
    if candidate.spec.issued_at != current.spec.issued_at:
        return candidate.spec.issued_at > current.spec.issued_at
    return candidate == current


# ---------------------------------------------------------------- eval sets


class EvalPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    prompt: str
    seed: int | None = None
    duration_s: float


class EvalSet(BaseModel):
    """One window's hidden prompts. The random salt keeps the commitment from being brute-forced
    against guessed prompts before the reveal."""

    model_config = ConfigDict(extra="forbid")

    competition_id: str
    window: int
    salt: str
    prompts: list[EvalPrompt] = Field(min_length=1)

    @field_validator("salt")
    @classmethod
    def _salt(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{32,}", value):
            raise ValueError("salt must be at least 16 random bytes, hex encoded")
        return value


def eval_set_digest(eval_set: EvalSet) -> str:
    return sha256_hex(EVAL_SET_CONTEXT + canonical_json(eval_set.model_dump(mode="json")))


def verify_eval_set(spec: TurboSpec, eval_set: EvalSet) -> None:
    """Raises TurboError unless the set is exactly the one the spec committed to for its window."""
    if eval_set.competition_id != spec.competition_id:
        raise TurboError("eval set belongs to a different competition")
    window = spec.window(eval_set.window)
    if eval_set_digest(eval_set) != window.eval_set_commitment:
        raise TurboError(f"eval set does not match the commitment for window {window.index}")
    bad = [p.id for p in eval_set.prompts if p.duration_s not in spec.speed.durations_s]
    if bad:
        raise TurboError(f"eval prompts {bad[:3]} use durations the spec does not measure")
    if len({p.id for p in eval_set.prompts}) != len(eval_set.prompts):
        raise TurboError("eval prompt ids must be unique")


# ---------------------------------------------------------------- submissions


class PipelineDescription(BaseModel):
    """What the image does differently. Published so the winner can be audited and adopted."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(max_length=2000)
    runtime: str = Field(max_length=64)
    steps: int = Field(ge=1)
    precision: str = Field(default="bf16", max_length=32)
    techniques: list[str] = Field(default_factory=list, max_length=32)
    # Weight file name -> sha256 hex, for every file the image loads beyond the reference profile's.
    weights: dict[str, str] = Field(default_factory=dict)
    # Reproducible-build source (repository at a commit), required before adoption.
    source_url: str = Field(max_length=500)


class TurboSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    v: Literal[1] = 1
    competition_id: str
    hotkey: str
    # A new variant id for the pipeline, e.g. "ltx-2.5-fast+sage-fp8.1"; becomes adoption metadata.
    profile_variant: str
    pipeline: PipelineDescription
    image_digest: str = Field(max_length=128)
    platform: TeeKind
    # The application-layer measurement the enclave will attest (RTMR3, 48 bytes hex).
    rtmr3: str
    created_at: int = Field(default_factory=lambda: int(time.time()))

    @field_validator("profile_variant")
    @classmethod
    def _variant(cls, value: str) -> str:
        if not _VARIANT.match(value):
            raise ValueError("profile_variant is 1-64 characters of a-z, 0-9, '.', '+' and '-'")
        return value

    @field_validator("rtmr3")
    @classmethod
    def _rtmr3(cls, value: str) -> str:
        if not _HEX96.match(value):
            raise ValueError("rtmr3 must be 96 lowercase hex characters")
        return value


class SignedTurboSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission: TurboSubmission
    signature: str

    def digest(self) -> str:
        return submission_digest(self)


def submission_message(submission: TurboSubmission) -> bytes:
    return SUBMISSION_CONTEXT + canonical_json(submission.model_dump(mode="json"))


def submission_digest(signed: SignedTurboSubmission) -> str:
    return sha256_hex(canonical_json(signed.model_dump(mode="json")))


def sign_submission(signer: HotkeySigner, submission: TurboSubmission) -> SignedTurboSubmission:
    if submission.hotkey != signer.ss58_address:
        raise TurboError("the submission names a different hotkey than the signer")
    return SignedTurboSubmission(submission=submission, signature=b64e(bytes(signer.sign(submission_message(submission)))))


def verify_submission(signed: SignedTurboSubmission, ss58_format: int | None = BITTENSOR_SS58_FORMAT) -> tuple[bool, str]:
    """Checks the hotkey's sr25519 signature (raw, or polkadot.js `<Bytes>`-wrapped)."""
    try:
        public_key = ss58_decode(signed.submission.hotkey, ss58_format)
        signature = b64d(signed.signature)
    except (HotkeyError, ValueError) as exc:
        return False, f"malformed submission: {exc}"
    if len(signature) != 64:
        return False, "sr25519 signatures are 64 bytes"
    message = submission_message(signed.submission)
    sr = _sr25519()
    for candidate in (message, _POLKADOT_WRAP[0] + message + _POLKADOT_WRAP[1]):
        if sr.verify(signature, candidate, public_key):
            return True, "ok"
    return False, "signature does not verify for the submission's hotkey"


# ---------------------------------------------------------------- commitments


@dataclass(frozen=True)
class OnChainCommitment:
    """One `Commitments.CommitmentOf(netuid, hotkey)` entry: the text and the block it was made in."""

    hotkey: str
    block: int
    data: str
    uid: int | None = None


@dataclass(frozen=True)
class ParsedCommitment:
    digest: str  # hex
    location: str | None


def commitment_string(digest_hex: str, location: str | None = None) -> str:
    if not _HEX64.match(digest_hex):
        raise TurboError("a submission digest is 64 lowercase hex characters")
    text = COMMITMENT_PREFIX + b64e(bytes.fromhex(digest_hex)) + (f"@{location}" if location else "")
    if len(text.encode()) > MAX_COMMITMENT_BYTES:
        raise TurboError(
            f"commitment is {len(text.encode())} bytes; the chain field holds {MAX_COMMITMENT_BYTES}. "
            "Use a shorter location or publish to the spec's submission_locations and commit the digest alone."
        )
    return text


def parse_commitment(data: str) -> ParsedCommitment | None:
    """None for commitments that are not Turbo submissions (miners may commit other things)."""
    if not isinstance(data, str) or not data.startswith(COMMITMENT_PREFIX) or len(data.encode()) > MAX_COMMITMENT_BYTES:
        return None
    body = data[len(COMMITMENT_PREFIX) :]
    encoded, _, location = body.partition("@")
    if len(encoded) != 43:
        return None
    try:
        digest = b64d(encoded)
    except ValueError:
        return None
    if len(digest) != 32 or b64e(digest) != encoded:
        return None
    return ParsedCommitment(digest.hex(), location or None)


Fetcher = Callable[[str], bytes]


@dataclass
class AcceptedSubmission:
    hotkey: str
    block: int
    digest: str
    signed: SignedTurboSubmission

    @property
    def submission(self) -> TurboSubmission:
        return self.signed.submission


def submission_urls(spec: TurboSpec, parsed: ParsedCommitment) -> list[str]:
    urls = [parsed.location] if parsed.location else []
    urls += [template.replace("{digest}", parsed.digest) for template in spec.submission_locations]
    return urls


def load_submission(raw: bytes, digest: str) -> SignedTurboSubmission:
    """Parses a fetched document and checks it is exactly what was committed."""
    if len(raw) > MAX_SUBMISSION_BYTES:
        raise TurboError("submission document is too large")
    try:
        signed = SignedTurboSubmission.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as exc:
        raise TurboError(f"submission document is malformed: {str(exc)[:200]}") from None
    if submission_digest(signed) != digest:
        raise TurboError("submission document does not match the committed digest")
    return signed


def collect_submissions(
    spec: TurboSpec,
    commitments: list[OnChainCommitment],
    fetch: Fetcher,
    ss58_format: int | None = BITTENSOR_SS58_FORMAT,
) -> tuple[list[AcceptedSubmission], dict[str, str]]:
    """Valid submissions in commitment order, and the reason every other Turbo commitment was refused.

    A submission whose image digest or RTMR3 an earlier commitment already claimed is a copy and is
    refused; ties within a block go to the lexicographically smaller hotkey, deterministically.
    """
    accepted: list[AcceptedSubmission] = []
    rejected: dict[str, str] = {}
    for commitment in sorted(commitments, key=lambda c: (c.block, c.hotkey)):
        parsed = parse_commitment(commitment.data)
        if parsed is None:
            continue
        if spec.submissions_close_block is not None and commitment.block > spec.submissions_close_block:
            rejected[commitment.hotkey] = f"committed at block {commitment.block}, after submissions closed"
            continue
        signed, errors = None, []
        for url in submission_urls(spec, parsed):
            try:
                signed = load_submission(fetch(url), parsed.digest)
                break
            except TurboError as exc:
                errors.append(str(exc))
            except Exception as exc:  # network errors from any fetcher implementation
                errors.append(f"{url}: {type(exc).__name__}")
        if signed is None:
            rejected[commitment.hotkey] = "; ".join(errors) or "no location to fetch the submission from"
            continue
        sub = signed.submission
        if sub.hotkey != commitment.hotkey:
            rejected[commitment.hotkey] = "submission names a different hotkey than the one that committed it"
            continue
        if sub.competition_id != spec.competition_id:
            rejected[commitment.hotkey] = f"submission is for competition {sub.competition_id}"
            continue
        ok, detail = verify_submission(signed, ss58_format)
        if not ok:
            rejected[commitment.hotkey] = detail
            continue
        if not any(base.platform == sub.platform for base in spec.base_measurements):
            rejected[commitment.hotkey] = f"the spec admits no {sub.platform} base"
            continue
        original = next(
            (a for a in accepted if a.submission.image_digest == sub.image_digest or a.submission.rtmr3 == sub.rtmr3), None
        )
        if original is not None:
            rejected[commitment.hotkey] = f"copies the image committed by {original.hotkey} at block {original.block}"
            continue
        accepted.append(AcceptedSubmission(commitment.hotkey, commitment.block, parsed.digest, signed))
    return accepted, rejected


def candidate_profiles(target_profile: str) -> list[str]:
    """What the gateway stores for a candidate enclave instead of the real profile id."""
    return [CANDIDATE_PROFILE_PREFIX + target_profile]


def is_candidate_profile_list(profiles: list[str]) -> bool:
    return bool(profiles) and all(p.startswith(CANDIDATE_PROFILE_PREFIX) for p in profiles)


# ---------------------------------------------------------------- adoption


class GoldenSample(BaseModel):
    prompt_id: str
    content_digest: str
    quality: float
    speed: float


class GoldenReference(BaseModel):
    """What an adopted profile is audited against afterwards: the revealed eval set's digest and
    the winning enclave's verified outputs, scores and speed on it."""

    profile_id: str
    competition_id: str
    window: int
    eval_set_digest: str
    image_digest: str
    rtmr3: str
    quality_metric: str
    quality_mean: float
    speed_kind: str
    speed: float
    samples: list[GoldenSample] = Field(default_factory=list)
