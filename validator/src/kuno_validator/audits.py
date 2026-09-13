"""Step-replay audits of a validator's own canaries (verified mode; see VERIFIED_MODE.md).

For a sample (1–5 %, configurable) of the canaries this validator created whose receipts
carry a step commitment, the auditor:

1. picks a random step k the executor for that runtime can replay;
2. asks the gateway to open step k, supplying a fresh X25519 key (the gateway refuses any
   job this validator's account did not create);
3. waits for the enclave's opening, checks the enclave signature, decrypts it and verifies
   the Merkle proofs against the root in the enclave-signed receipt;
4. checks the transcript describes the canary (profile, params digest, seed, conditioning,
   model identity, schedule) and that leaf 0 is the seed's noise;
5. re-executes step k from the latent at k-1 and compares the result bit for bit with leaf k;
6. occasionally (`full_rerun_rate`) also asks for every leaf and re-runs the whole
   trajectory on reference hardware.

Outcome policy mirrors canaries: a failure is *attributable* once the enclave's signature
establishes who produced the opening (or when the enclave declines or misses the deadline),
and any attributable failure in the scoring window zeroes that miner. A missing signature,
a gateway refusal or a validator-side gap (no executor for the runtime) costs nothing.

Bitwise comparison is the only mode: verified profiles pin everything that changes bits.
Tolerance-based comparison (TAO-style acceptance regions) is future work.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from kuno_protocol.canonical import b64e, canonical_json, sha256_hex
from kuno_protocol.crypto import DecryptionError, generate_hpke_keypair
from kuno_protocol.profiles import ModelProfile
from kuno_protocol.receipts import Receipt, verify_receipt
from kuno_protocol.schemas import GenerationParams
from kuno_protocol.toy_denoiser import (
    TOY_RUNTIME,
    run_toy_trajectory,
    toy_model_digest,
    toy_noise,
    toy_replay_step,
    toy_state,
    toy_transcript,
)
from kuno_protocol.verified import (
    SealedOpening,
    StepCommitment,
    StepTranscript,
    Tensor,
    VerifiedModeError,
    latent_digest,
    open_sealed_opening,
    verify_opening,
    verify_sealed_opening,
)

from .ledger import EnclaveKey

log = logging.getLogger("kuno.validator.audits")

# Outcomes older than this are pruned from the state file; it must exceed any scoring window.
AUDIT_RETENTION_S = 7 * 86400.0


class TranscriptMismatch(Exception):
    """Raised by an executor mid-replay when the miner's transcript cannot be an honest run
    (e.g. the conditioning computed from the canary prompt differs). Attributable."""


@dataclass
class CanaryRecord:
    """What the validator itself knows about one of its canaries: enough to re-execute a step."""

    job_id: str
    profile_id: str
    params: dict  # GenerationParams as JSON; must hash to the receipt's params_digest
    prompt: str
    seed: int
    receipt: dict
    negative_prompt: str | None = None
    created_at: float = 0.0


@dataclass
class AuditOutcome:
    job_id: str
    profile_id: str
    ok: bool
    detail: str
    enclave_id: str | None = None
    miner_hotkey: str | None = None
    attributable: bool = False
    step: int | None = None
    audit_id: str | None = None
    full_rerun: bool = False
    at: float = 0.0


class StepExecutor(Protocol):
    """Re-executes single steps of one runtime, on the hardware class it is certified for."""

    runtime: str

    def candidate_steps(self, commitment: StepCommitment, profile: ModelProfile) -> list[int]:
        """Steps (leaf indices ≥ 1) this executor can replay; the auditor picks one at random."""

    def check_transcript(self, transcript: StepTranscript, canary: CanaryRecord, profile: ModelProfile) -> str | None:
        """Why the transcript cannot be an honest run of this canary, or None."""

    def initial_state(self, transcript: StepTranscript, canary: CanaryRecord) -> list[Tensor] | None:
        """Leaf 0 recomputed from the seed, or None if this executor cannot."""

    def execute(self, transcript: StepTranscript, canary: CanaryRecord, step: int, state: list[Tensor]) -> list[Tensor]:
        """The latent state at leaf `step`, from the state at `step - 1`."""

    def trajectory(self, transcript: StepTranscript, canary: CanaryRecord) -> list[str] | None:
        """Every leaf's latent digest from a full honest run, or None if too costly here."""


class ToyStepExecutor:
    """Reference executor for the mock backend's toy denoiser, so audits really run on dev networks."""

    runtime = TOY_RUNTIME

    def candidate_steps(self, commitment: StepCommitment, profile: ModelProfile) -> list[int]:
        return list(range(1, commitment.leaves))

    def expected_transcript(self, transcript: StepTranscript, canary: CanaryRecord, profile: ModelProfile) -> StepTranscript:
        params = GenerationParams.model_validate(canary.params)
        return toy_transcript(
            job_id=canary.job_id,
            params_digest=sha256_hex(canonical_json(params.model_dump(mode="json"))),
            profile_id=profile.id,
            family=profile.family,
            model_digest=toy_model_digest(profile.id, profile.checkpoint),
            seed=canary.seed,
            prompt=canary.prompt,
            negative_prompt=canary.negative_prompt,
            frames=profile.num_frames(params.duration_s, params.fps),
            steps=profile.steps,
            hardware_class=transcript.hardware_class,
        )

    def check_transcript(self, transcript: StepTranscript, canary: CanaryRecord, profile: ModelProfile) -> str | None:
        expected = self.expected_transcript(transcript, canary, profile)
        for name in type(expected).model_fields:
            if getattr(transcript, name) != getattr(expected, name):
                return f"transcript {name} does not match an honest run of this canary"
        return None

    def initial_state(self, transcript: StepTranscript, canary: CanaryRecord) -> list[Tensor]:
        return toy_state(toy_noise(canary.seed, transcript.stages[0].tensors[0]))

    def execute(self, transcript: StepTranscript, canary: CanaryRecord, step: int, state: list[Tensor]) -> list[Tensor]:
        return toy_replay_step(transcript, canary.prompt, canary.negative_prompt, step, state)

    def trajectory(self, transcript: StepTranscript, canary: CanaryRecord) -> list[str]:
        return [latent_digest(tensors) for *_, tensors in run_toy_trajectory(transcript, canary.prompt, canary.negative_prompt)]


@dataclass
class AuditPolicy:
    # Share of auditable canaries to audit; None uses each profile's verified.audit_rate.
    rate: float | None = None
    # Share of audits that also open every leaf and re-run the full trajectory.
    full_rerun_rate: float = 0.1
    # How long the enclave has to answer.
    deadline_s: float = 600.0
    poll_s: float = 1.0
    # Production refuses simulated hardware classes (like it refuses simulated TEEs).
    production: bool = False
    # A missing or late opening is the miner's fault. See VERIFIED_MODE.md for the caveat.
    missing_is_attributable: bool = True
    # Once the network requires verified mode, a canary receipt without a commitment is a failure.
    require_commitment: bool = False


@dataclass
class PendingAudit:
    audit_id: str
    canary: CanaryRecord
    step: int
    include_leaves: bool
    private_key: bytes
    public_key: bytes
    enclave_id: str
    miner_hotkey: str | None
    requested_at: float
    deadline: float


class Auditor:
    def __init__(
        self,
        request: Callable[..., httpx.Response],
        profiles: dict[str, ModelProfile],
        keys: Callable[[str], EnclaveKey | None],
        executors: dict[str, StepExecutor] | None = None,
        policy: AuditPolicy | None = None,
        rng: random.Random | None = None,
        state_path: Path | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """`request(method, path, **httpx kwargs)` is the validator's authenticated gateway call
        (Validator._request); `keys(enclave_id)` returns a self-certified enclave key."""
        self._request = request
        self.profiles = profiles
        self.keys = keys
        self.executors: dict[str, StepExecutor] = executors if executors is not None else {TOY_RUNTIME: ToyStepExecutor()}
        self.policy = policy or AuditPolicy()
        self.rng = rng or random.SystemRandom()
        self.state_path = state_path
        self.clock = clock
        self.sleep = sleep
        self.pending: dict[str, PendingAudit] = {}
        self.outcomes: list[AuditOutcome] = []
        self._load_state()

    # ------------------------------------------------------------ selection

    def auditable(self, canary: CanaryRecord) -> bool:
        profile = self.profiles.get(canary.profile_id)
        try:
            receipt = Receipt.model_validate(canary.receipt)
        except ValidationError:
            return False
        return profile is not None and profile.verified is not None and receipt.body.step_commitment is not None

    def should_audit(self, canary: CanaryRecord) -> bool:
        if not self.auditable(canary):
            return self.policy.require_commitment and self.profiles.get(canary.profile_id) is not None
        rate = self.policy.rate if self.policy.rate is not None else self.profiles[canary.profile_id].verified.audit_rate
        return self.rng.random() < rate

    def select(self, canaries: list[CanaryRecord]) -> list[CanaryRecord]:
        return [c for c in canaries if self.should_audit(c)]

    def _runtime_for(self, profile: ModelProfile, hardware_class: str) -> str | None:
        hardware = profile.verified.hardware_class(hardware_class) if profile.verified else None
        if hardware is None:
            return None
        return TOY_RUNTIME if hardware.dev else profile.verified.runtime

    # ------------------------------------------------------------ requesting

    def _outcome(self, canary: CanaryRecord, ok: bool, detail: str, **extra: Any) -> AuditOutcome:
        outcome = AuditOutcome(canary.job_id, canary.profile_id, ok, detail, **extra)
        outcome.at = outcome.at or self.clock()
        self.outcomes.append(outcome)
        self._save_state()
        level = logging.INFO if ok else (logging.WARNING if outcome.attributable else logging.ERROR)
        log.log(level, "audit of job %s (%s): %s", canary.job_id, outcome.miner_hotkey or "unknown miner", detail)
        return outcome

    def request(self, canary: CanaryRecord, step: int | None = None, include_leaves: bool | None = None) -> PendingAudit | AuditOutcome | None:
        """Asks the gateway to open a step. Returns the pending audit, an immediate outcome, or None if not auditable."""
        try:
            receipt = Receipt.model_validate(canary.receipt)
        except ValidationError:
            return self._outcome(canary, False, "canary receipt is malformed")
        body = receipt.body
        key = self.keys(body.enclave_id)
        if key is None or body.job_id != canary.job_id or not verify_receipt(receipt, key.signing_public_key):
            return self._outcome(canary, False, "canary receipt does not verify against a known enclave key", enclave_id=body.enclave_id)
        facts = {"enclave_id": body.enclave_id, "miner_hotkey": key.miner_hotkey}
        if sha256_hex(canonical_json(GenerationParams.model_validate(canary.params).model_dump(mode="json"))) != body.params_digest:
            return self._outcome(canary, False, "validator's canary record does not match the signed params digest", **facts)
        profile = self.profiles.get(body.profile_id)
        commitment = body.step_commitment
        if profile is None or profile.verified is None:
            return None
        if commitment is None:
            if self.policy.require_commitment:
                return self._outcome(canary, False, "receipt carries no step commitment for a verified-mode profile", attributable=True, **facts)
            return None
        hardware = profile.verified.hardware_class(commitment.hardware_class)
        if hardware is None:
            return self._outcome(canary, False, f"committed on unknown hardware class {commitment.hardware_class}", attributable=True, **facts)
        if hardware.dev and self.policy.production:
            return self._outcome(canary, False, "committed on a simulated hardware class", attributable=True, **facts)

        executor = self.executors.get(self._runtime_for(profile, commitment.hardware_class) or "")
        # Without a replayable step the opening is still verified against the root; it just isn't re-executed.
        choices = (executor.candidate_steps(commitment, profile) if executor is not None else []) or list(range(1, commitment.leaves))
        if step is None:
            step = self.rng.choice(choices)
        if include_leaves is None:
            include_leaves = self.rng.random() < self.policy.full_rerun_rate
        private_key, public_key = generate_hpke_keypair()
        response = self._request(
            "POST",
            "/validator/v1/audits",
            json={"job_id": canary.job_id, "step": step, "recipient_public_key": b64e(public_key), "include_leaves": include_leaves},
        )
        if response.status_code != 201:
            code = _error_code(response)
            return self._outcome(canary, False, f"gateway refused the audit ({response.status_code} {code})", step=step, **facts)
        data = response.json()
        now = self.clock()
        pending = PendingAudit(
            audit_id=data["audit_id"], canary=canary, step=step, include_leaves=include_leaves,
            private_key=private_key, public_key=public_key, enclave_id=body.enclave_id, miner_hotkey=key.miner_hotkey,
            requested_at=now, deadline=now + self.policy.deadline_s,
        )
        self.pending[pending.audit_id] = pending
        return pending

    # ------------------------------------------------------------ collecting

    def poll(self) -> list[AuditOutcome]:
        """Checks every pending audit once; returns the ones that concluded."""
        done: list[AuditOutcome] = []
        for audit_id, pending in list(self.pending.items()):
            now = self.clock()
            response = self._request("GET", f"/validator/v1/audits/{audit_id}")
            facts = self._facts(pending)
            if response.status_code != 200:
                if now < pending.deadline:
                    continue
                outcome = self._outcome(pending.canary, False, f"gateway lost the audit ({response.status_code})", **facts)
            else:
                data = response.json()
                status = data.get("status")
                if status == "answered" and data.get("opening"):
                    outcome = self.verify(pending, data["opening"])
                elif status == "failed":
                    outcome = self._outcome(
                        pending.canary, False, f"enclave declined to open step {pending.step} ({data.get('error_code')})", attributable=True, **facts
                    )
                elif status in ("expired", "gone") or now >= pending.deadline:
                    outcome = self._outcome(
                        pending.canary, False, f"no opening for step {pending.step} within {self.policy.deadline_s:g}s",
                        attributable=self.policy.missing_is_attributable, **facts,
                    )
                else:
                    continue
            del self.pending[audit_id]
            done.append(outcome)
        return done

    def run(self, canary: CanaryRecord, step: int | None = None, include_leaves: bool | None = None) -> AuditOutcome | None:
        """Requests one audit and waits for its outcome (bounded by the policy deadline)."""
        pending = self.request(canary, step, include_leaves)
        if not isinstance(pending, PendingAudit):
            return pending
        while pending.audit_id in self.pending:
            for outcome in self.poll():
                if outcome.audit_id == pending.audit_id:
                    return outcome
            if pending.audit_id in self.pending:
                self.sleep(self.policy.poll_s)
        return None

    def _facts(self, pending: PendingAudit) -> dict:
        return {
            "enclave_id": pending.enclave_id, "miner_hotkey": pending.miner_hotkey, "step": pending.step,
            "audit_id": pending.audit_id, "full_rerun": pending.include_leaves,
        }

    # ------------------------------------------------------------ verifying

    def verify(self, pending: PendingAudit, opening_json: dict) -> AuditOutcome:
        canary, step, facts = pending.canary, pending.step, self._facts(pending)

        def fail(detail: str, attributable: bool = True) -> AuditOutcome:
            return self._outcome(canary, False, detail, attributable=attributable, **facts)

        receipt = Receipt.model_validate(canary.receipt)
        commitment = receipt.body.step_commitment
        key = self.keys(pending.enclave_id)
        assert commitment is not None
        try:
            sealed = SealedOpening.model_validate(opening_json)
        except ValidationError:
            return fail("gateway relayed a malformed opening", attributable=False)
        expected = (pending.audit_id, canary.job_id, step, b64e(pending.public_key), pending.enclave_id)
        if (sealed.audit_id, sealed.job_id, sealed.step, sealed.recipient_public_key, sealed.enclave_id) != expected:
            return fail("gateway relayed an opening for a different audit", attributable=False)
        if key is None or not verify_sealed_opening(sealed, key.signing_public_key):
            log.error("audit %s: opening is not signed by enclave %s; the relay may be tampering", pending.audit_id, pending.enclave_id)
            return fail("opening is not signed by the enclave", attributable=False)

        # From here on the enclave signed exactly what we hold: every failure is the miner's.
        try:
            opening, latents = open_sealed_opening(pending.private_key, sealed)
        except (DecryptionError, VerifiedModeError):
            return fail("signed opening does not decrypt to a valid opening for this validator's key")
        reason = verify_opening(commitment, opening, latents, job_id=canary.job_id, step=step, include_leaves=pending.include_leaves)
        if reason is not None:
            return fail(reason)

        profile = self.profiles[receipt.body.profile_id]
        transcript = opening.transcript
        if transcript.params_digest != receipt.body.params_digest or transcript.profile_id != profile.id or transcript.seed != canary.seed:
            return fail("transcript does not describe this canary (params, profile or seed)")
        runtime = self._runtime_for(profile, commitment.hardware_class)
        if transcript.runtime != runtime:
            return fail(f"transcript runtime {transcript.runtime} is not the verified runtime for {commitment.hardware_class}")
        executor = self.executors.get(runtime or "")
        if executor is None:
            return self._outcome(
                canary, True, f"opening verified against the signed root; no {runtime} executor here, step {step} not re-executed", **facts
            )
        reason = executor.check_transcript(transcript, canary, profile)
        if reason is not None:
            return fail(reason)

        leaves = {leaf.index: leaf for leaf in opening.leaves}
        try:
            initial = executor.initial_state(transcript, canary)
            if initial is not None and latent_digest(initial) != leaves[0].latent:
                return fail("leaf 0 is not the noise the transcript's seed produces")
            details = []
            if step in executor.candidate_steps(commitment, profile):
                produced = executor.execute(transcript, canary, step, latents[step - 1])
                if latent_digest(produced) != leaves[step].latent:
                    return fail(f"re-executing step {step} does not reproduce the committed latent (bitwise)")
                details.append(f"step {step} re-executed bitwise")
            else:
                details.append(f"opening for step {step} verified; this executor cannot replay that step")
            digests = executor.trajectory(transcript, canary) if pending.include_leaves else None
        except TranscriptMismatch as exc:
            return fail(str(exc))
        except Exception as exc:  # the validator's own executor broke: not the miner's fault
            log.exception("executor %s failed on audit %s", transcript.runtime, pending.audit_id)
            return fail(f"validator executor failed ({type(exc).__name__}); nothing concluded", attributable=False)
        if pending.include_leaves:
            if digests is not None:
                committed = [leaves[i].latent for i in range(commitment.leaves)]
                mismatch = next((i for i, (a, b) in enumerate(zip(digests, committed)) if a != b), None)
                if mismatch is not None or len(digests) != len(committed):
                    return fail(f"full re-run diverges from the commitment at leaf {mismatch}")
                details.append("full re-run matches every leaf")
        return self._outcome(canary, True, "; ".join(details), **facts)

    # ------------------------------------------------------------ scoring

    def penalties(self, now: float, window_s: float) -> dict[str, list[str]]:
        """Same policy as canaries: an attributable failure inside the window zeroes the miner."""
        out: dict[str, list[str]] = {}
        for outcome in self.outcomes:
            if outcome.ok or not outcome.attributable or not outcome.miner_hotkey or outcome.at < now - window_s:
                continue
            out.setdefault(outcome.miner_hotkey, []).append(f"failed step audit of {outcome.profile_id} ({outcome.detail})")
        return out

    # ------------------------------------------------------------ state

    def _load_state(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text())
            self.outcomes = [AuditOutcome(**item) for item in state.get("audits", [])]
        except (ValueError, TypeError):
            log.error("audit state file %s is unreadable; starting without audit history", self.state_path)

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        cutoff = self.clock() - AUDIT_RETENTION_S
        self.outcomes = [o for o in self.outcomes if o.at >= cutoff]
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"audits": [asdict(o) for o in self.outcomes]}, indent=2))
        tmp.replace(self.state_path)


def _error_code(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", {})
        return detail.get("code", "error") if isinstance(detail, dict) else "error"
    except ValueError:
        return "error"
