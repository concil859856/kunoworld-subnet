"""The enclave job loop: attest, pull sealed work, decrypt, generate, seal, sign."""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace

import httpx
from pydantic import ValidationError

from kuno_protocol.attestation import AttestationEvidence, AttestationUnavailable, TEEProvider, build_evidence
from kuno_protocol.blobs import decrypt_blob, encrypt_blob
from kuno_protocol.canonical import b64d, canonical_json, sha256_hex
from kuno_protocol.content_policy import ContentPolicyViolation, check_prompt
from kuno_protocol.crypto import DecryptionError, RecipientSession
from kuno_protocol.envelope import CAPACITY_REFUSED, advertised, describe, fits, max_duration
from kuno_protocol.hotkey import HotkeySigner, sign_hotkey_proof
from kuno_protocol.media import ROLE_TYPES, sniff_mime
from kuno_protocol.plans import (
    PLAN_FAILED,
    PLAN_FEATURE,
    PLAN_OPTION,
    Plan,
    PlanContext,
    PlanError,
    PlanOptions,
    check_revision,
    choose,
    plan_context,
    plan_messages,
    repair,
    retry_messages,
    seal_plan,
    validate as validate_plan,
)
from kuno_protocol.profiles import Mode, ModelProfile, ParamError, load_profiles, shot_prompt, validate_params
from kuno_protocol.receipts import PlanInfo, Receipt, ReceiptBody, input_digest, sign_receipt
from kuno_protocol.schemas import GenerationParams, MinerChallenge, MinerJob, SealedPayload, input_label, job_aad, output_label
from kuno_protocol.sealed_payload import MalformedPayload, open_payload
from kuno_protocol.verified import MinerAudit

from .audits import AuditCalls, AuditResponder
from .backends.base import ENHANCE_PROMPT_OPTION, Backend, GenerationTask, InputFile
from .backends.media_tools import BackendError, CapacityRefused
from .config import WorkerConfig
from .gateway_client import GatewayClient, GatewayError
from .identity import EnclaveIdentity
from .safety import SafetyUnavailable, SafetyViolation, check_output, check_request, request_signals
from .safety_frames import RequestSignals

log = logging.getLogger("kuno.worker")

PROGRESS_INTERVAL_S = 0.5
# While a job runs the worker doesn't pull, and the gateway counts an enclave silent for a minute (enclave_heartbeat_s)
# as gone. A render or an upload can block the job thread for longer, so a side thread repeats the last progress report.
HEARTBEAT_S = 20.0
# One message for every blocked prompt, whoever wrote it. In Private mode the gateway sees failure messages but not the
# sealed options, so a message of its own for an enhanced prompt would tell it that enhancement was asked for.
PROMPT_BLOCKED = "The request was blocked by the content policy."
# A plan job whose planner never wrote a usable plan (kuno_protocol.plans.PLAN_FAILED): refunded, and not a miner fault.
PLAN_FAILED_MESSAGE = "The planner could not write a usable plan for this brief."


class JobRejected(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class PlanRefused(JobRejected):
    """The planner refused the brief: `safety_blocked`, and never regenerated."""

    def __init__(self) -> None:
        super().__init__("safety_blocked", PROMPT_BLOCKED)


class JobCanceled(Exception):
    pass


class CertificateUnavailable(Exception):
    """C2PA is on but the gateway issued no usable signing certificate; registration is retried."""


class MissingTimestampAuthority(ValueError):
    """A real enclave was given a certificate but no RFC 3161 timestamp authority to sign with."""


class Worker:
    def __init__(
        self,
        config: WorkerConfig,
        tee: TEEProvider,
        backends: dict[str, Backend],
        identity: EnclaveIdentity | None = None,
        transport=None,
        hotkey: HotkeySigner | None = None,
    ):
        catalog = load_profiles()
        unknown = [p for p in config.profiles if p not in catalog]
        if unknown:
            raise ValueError(f"unknown profiles: {', '.join(unknown)}")
        if hotkey is not None and config.miner_hotkey and config.miner_hotkey != hotkey.ss58_address:
            raise ValueError(f"KUNO_MINER_HOTKEY is {config.miner_hotkey} but the configured hotkey secret is {hotkey.ss58_address}")
        if config.tee == "open":
            # No quote binds an open-tier worker's keys to anyone; its hotkey proof must (PRIVACY_MODES.md).
            if hotkey is None:
                raise ValueError("an open-tier worker (KUNO_TEE=open) needs its miner hotkey secret to prove every registration")
            if config.provenance == "c2pa" and config.provenance_cert_chain is None:
                raise ValueError("the gateway issues C2PA certificates to attested enclaves only: run open-tier workers with KUNO_PROVENANCE=off")
        self.hotkey = hotkey
        self.miner_hotkey = hotkey.ss58_address if hotkey is not None else config.miner_hotkey
        self.failures = 0
        self.config = config
        self.turbo_submission = self._load_turbo_submission()
        self.tee = tee
        self.backends = backends
        self.profiles: dict[str, ModelProfile] = {p: catalog[p] for p in config.profiles}
        for profile in self.profiles.values():
            self.backend_for(profile)
        self.identity = identity or EnclaveIdentity.generate()
        self._provenance_signer = self._build_provenance_signer()
        self.client = GatewayClient(
            config.gateway_url, self.identity.signing_key, self.identity.enclave_id, transport=transport, country=config.miner_country
        )
        self.evidence: AttestationEvidence | None = None
        self.last_attested = 0.0
        self.ready = threading.Event()
        self.busy = False
        self._seen: set[str] = set()
        self._last_progress: dict[str, float] = {}
        self._reported: dict[str, tuple[float, str]] = {}
        self._canceled: set[str] = set()
        # Verified mode: retained step openings per job, discarded if the job fails after generation.
        self._openings: dict[str, object] = {}
        self.audits = AuditResponder(self.identity)
        # Landmark pings go straight to the landmarks, not through the gateway; tests replace the transport.
        self.location_transport = None

    # ------------------------------------------------------------ attestation

    def backend_for(self, profile: ModelProfile) -> Backend:
        backend = self.backends.get(profile.family) or self.backends.get("*")
        if backend is None:
            raise ValueError(f"no backend configured for {profile.family}")
        return backend

    def serving_envelope(self) -> dict[str, dict]:
        """Per profile, the longest duration this worker's hardware serves at each resolution, aspect ratio and fps
        (kuno_protocol.envelope): the memory plan on a quantized class, the profile's limits otherwise. Computed once,
        at start-up (`run`), from plain data: no GPU is touched."""
        if getattr(self, "_envelope", None) is None:
            from kuno_protocol.envelope import full_table

            tables = {}
            for profile_id, profile in self.profiles.items():
                compute = getattr(self.backend_for(profile), "serving_envelope", None)
                tables[profile_id] = compute(profile) if compute is not None else full_table(profile)
            self._envelope = tables
            self._advertised = advertised(tables, self.profiles)
        return self._envelope

    @property
    def features(self) -> list[str]:
        """The optional job kinds registration advertises (MinerRegistration.features): `plan/1` when some served profile
        offers plans and its backend writes them."""
        plans = any(Mode.PLAN in profile.modes and getattr(self.backend_for(profile), "plans", False) for profile in self.profiles.values())
        return [PLAN_FEATURE] if plans else []

    @property
    def advertised_envelope(self) -> dict | None:
        """What registration carries: the profiles this hardware can't serve in full, or None when it serves them all."""
        self.serving_envelope()
        return self._advertised

    def attest(self, nonce: bytes) -> AttestationEvidence:
        return build_evidence(
            self.tee,
            nonce,
            self.identity.hpke_public,
            self.identity.signing_public,
            self.config.image_digest,
            list(self.profiles),
            self.config.hardware,
        )

    def register(self) -> None:
        nonce = self.client.nonce()
        evidence = self.attest(nonce)
        proof = None
        if self.hotkey is not None:
            proof = sign_hotkey_proof(self.hotkey, nonce, self.identity.enclave_id, self.identity.signing_public)
        extra: dict = {}
        if self.turbo_submission is not None:
            extra["turbo_submission"] = self.turbo_submission
        if self.advertised_envelope is not None:
            extra["envelope"] = self.advertised_envelope
        if self.features:
            extra["features"] = self.features
        location = self._location_proof(nonce)
        if location is not None:
            extra["location"] = location
        self.client.register(evidence, self.miner_hotkey, self.config.capacity, proof, **extra)
        self.evidence = evidence
        self.last_attested = time.time()
        self._refresh_certificate()
        self.ready.set()
        log.info("attested enclave %s for %s", self.identity.enclave_id, ", ".join(self.profiles))

    def _location_proof(self, nonce: bytes) -> dict | None:
        """For profiles whose licence is bound to territory (MiniMax H3): signed round trips to the gateway's landmarks,
        measured from inside this VM, so the gateway and validators can bound where it runs (kuno_protocol.location)."""
        if not any(profile.license.region_policy for profile in self.profiles.values()):
            return None
        try:
            document = self.client.landmarks()
        except Exception as exc:  # a gateway without landmarks, or one that's down: register without a proof
            log.warning("could not read the gateway's landmarks (%s); registering without a location proof", exc)
            return None
        if not document:
            return None
        from kuno_protocol.location import SignedLandmarks

        from .location import measure

        landmarks = SignedLandmarks.model_validate(document).landmarks
        proof = measure(landmarks, nonce, self.identity.enclave_id, transport=self.location_transport)
        return proof.model_dump(mode="json")

    # ------------------------------------------------------------ C2PA certificate

    def _certificate_margin_s(self) -> float:
        # Checked only at (re-)attestation, which a long job can delay: leave room for two missed rounds.
        return 2 * self.config.reattest_s + self.config.pull_wait_s

    def _refresh_certificate(self) -> None:
        """Right after each successful (re-)attestation, while the gateway's verification is fresh:
        obtain a gateway-issued C2PA certificate if there is none, it is a stand-in, or it is due."""
        from .certificates import DEV, GATEWAY, CertifiedSigner, EnclaveCertificate
        from .provenance import certificate_signing_request

        signer = self._provenance_signer
        if not isinstance(signer, CertifiedSigner):
            return  # provenance off, or an operator-supplied certificate chain
        now = time.time()
        current = signer.certificate
        if current is not None and current.source == GATEWAY and now < current.refresh_at(self._certificate_margin_s()):
            return
        try:
            csr = certificate_signing_request(self.identity.signing_key, self.identity.enclave_id)
            response = self.client.request_certificate(csr)
            issued = EnclaveCertificate.parse(
                response["certificate_chain_pem"], self.identity.signing_public, self.identity.enclave_id, GATEWAY
            )
            if not issued.usable(now + 1):
                raise ValueError("the gateway issued a certificate that is not currently valid")
        except GatewayError as exc:
            no_ca = exc.status == 503 and exc.code == "ca_unavailable"
            if no_ca and self.config.tee == "mock":
                if current is not None and current.source == GATEWAY and current.usable(now):
                    return  # keep a real certificate until it runs out
                if current is None or current.source != DEV:
                    self._install_dev_certificate(signer, DEV)
                    log.warning("the gateway has no C2PA CA; this mock-TEE worker signs provenance with an untrusted dev certificate")
                return
            self._certificate_unavailable(signer, current, now, exc)
            return
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            self._certificate_unavailable(signer, current, now, exc)
            return
        # The gateway lists its timestamp authorities in order of preference (`tsa_urls`); an older one sends `tsa_url`.
        suggested_tsas = response.get("tsa_urls") or response.get("tsa_url")
        if not signer.effective_tsa_urls(suggested_tsas) and self.config.tee != "mock":
            # Without a timestamp, readers reject every manifest once the short-lived certificate expires,
            # so a customer's video would stop verifying a day after delivery. Refuse instead.
            self._certificate_unavailable(
                signer, current, now,
                MissingTimestampAuthority("no RFC 3161 timestamp authority: set KUNO_PROVENANCE_TSA_URL or the gateway's KUNO_C2PA_TSA_URLS"),
            )
            return
        signer.install(issued, suggested_tsas)
        log.info("C2PA certificate issued for enclave %s, valid until %s", self.identity.enclave_id,
                 time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(issued.not_after)))

    def _certificate_unavailable(self, signer, current, now: float, exc: Exception) -> None:
        """Keep a still-valid certificate through a failed renewal; without one, stop taking jobs."""
        from .certificates import PROVISIONAL

        if isinstance(exc, GatewayError):
            reason = exc.message
        elif isinstance(exc, MissingTimestampAuthority):
            reason = str(exc)
        else:
            reason = type(exc).__name__
        if current is not None and current.source != PROVISIONAL and current.usable(now):
            log.warning("could not renew the C2PA certificate (%s); the current one stays valid for %.0fs", reason, current.not_after - now)
            return
        signer.install(None)
        self.ready.clear()
        log.error("no C2PA signing certificate (%s): this worker takes no jobs until the gateway issues one", reason)
        raise CertificateUnavailable(f"no C2PA signing certificate: {reason}") from exc

    def _install_dev_certificate(self, signer, source: str) -> None:
        from .certificates import EnclaveCertificate
        from .provenance import issue_dev_certificate

        chain, _root = issue_dev_certificate(self.identity.signing_key, self.identity.enclave_id)
        signer.install(EnclaveCertificate.parse(chain, self.identity.signing_public, self.identity.enclave_id, source))

    # ------------------------------------------------------------ main loop

    def run(self, stop: threading.Event | None = None) -> None:
        """Serves until `stop` is set. Attestation, registration and gateway failures are
        logged and retried with capped exponential backoff; they never end the loop."""
        stop = stop or threading.Event()
        for profile in self.profiles.values():
            self.backend_for(profile).warm(profile)
        # After warm-up, which refuses a class that cannot serve a profile at all; every registration advertises it.
        if self.advertised_envelope is not None:
            log.info("serving envelope: %s serve less than their full limits on this hardware", ", ".join(self.advertised_envelope))
        while not stop.is_set():
            step = "register"
            try:
                if not self.ready.is_set() or time.time() - self.last_attested > self.config.reattest_s:
                    self.register()
                step = "pull"
                work = self.client.pull(wait=self.config.pull_wait_s)
                self.failures = 0
            except Exception as exc:
                if step == "pull" and isinstance(exc, GatewayError) and exc.status in (401, 403):
                    self.ready.clear()  # the gateway no longer knows this enclave: attest again
                delay = self._retry_delay(exc)
                self._log_failure(step, exc, delay)
                stop.wait(delay)
                continue
            kind = work.get("kind")
            try:
                if kind == "job":
                    self.handle_job(MinerJob.model_validate(work))
                elif kind == "challenge":
                    self.handle_challenge(MinerChallenge.model_validate(work))
                elif kind == "audit":
                    self.audits.handle(MinerAudit.model_validate(work), AuditCalls(self.client))
            except ValidationError:
                log.error("the gateway sent a malformed %s; ignoring it", kind)

    RETRY_BASE_S = 1.0
    # Network blips and gateway restarts clear quickly; a broken attestation setup does not.
    TRANSIENT_RETRY_CAP_S = 30.0

    @staticmethod
    def _transient(exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPError):
            return True
        return isinstance(exc, GatewayError) and (exc.status >= 500 or exc.status in (401, 429) or exc.code == "bad_nonce")

    def _retry_delay(self, exc: Exception) -> float:
        cap = self.TRANSIENT_RETRY_CAP_S if self._transient(exc) else max(self.config.retry_max_s, self.RETRY_BASE_S)
        delay = min(self.RETRY_BASE_S * 2 ** min(self.failures, 16), cap)
        self.failures += 1
        return delay

    def _log_failure(self, step: str, exc: Exception, delay: float) -> None:
        """Operator-facing and actionable. Registration never touches customer content, so its
        errors are safe to print; a failed pull is logged by type only, like a failed job."""
        if isinstance(exc, AttestationUnavailable):
            log.error("cannot attest on this machine: %s (retrying in %.0fs)", exc, delay)
        elif isinstance(exc, GatewayError) and exc.code == "attestation_failed":
            log.error(
                "the gateway rejected this enclave's attestation: %s — check KUNO_IMAGE_DIGEST, KUNO_PROFILES and "
                "the golden manifest (retrying in %.0fs)", exc.message, delay,
            )
        elif isinstance(exc, GatewayError) and not self._transient(exc):
            log.error("the gateway refused the %s (%s): %s (retrying in %.0fs)", step, exc.code, exc.message, delay)
        elif self._transient(exc):
            log.warning("gateway unavailable during %s (%s); retrying in %.0fs", step, type(exc).__name__, delay)
        elif step == "register":
            log.error("registration failed: %s: %s (retrying in %.0fs)", type(exc).__name__, exc, delay)
        else:
            log.error("pulling work failed with %s; retrying in %.0fs", type(exc).__name__, delay)

    def retire(self) -> None:
        """Best-effort goodbye on shutdown; the gateway releases anything still queued for us."""
        if not self.ready.is_set():
            return
        try:
            released = self.client.retire().get("released", 0)
            log.info("retired enclave %s (%d queued job(s) released)", self.identity.enclave_id, released)
        except (httpx.HTTPError, GatewayError) as exc:
            log.warning("could not retire cleanly (%s); the gateway will notice within a minute", type(exc).__name__)

    def _load_turbo_submission(self) -> dict | None:
        """The signed submission this worker competes with, checked against its own hotkey and image."""
        path = self.config.turbo_submission
        if path is None:
            return None
        from kuno_protocol.turbo import SignedTurboSubmission, verify_submission

        signed = SignedTurboSubmission.model_validate_json(path.read_text())
        ok, detail = verify_submission(signed)
        if not ok:
            raise ValueError(f"KUNO_TURBO_SUBMISSION does not verify: {detail}")
        submission = signed.submission
        if self.miner_hotkey and self.miner_hotkey != submission.hotkey:
            raise ValueError(f"the Turbo submission is for hotkey {submission.hotkey}, but this worker runs {self.miner_hotkey}")
        if submission.image_digest != self.config.image_digest:
            raise ValueError("the Turbo submission names a different image digest than KUNO_IMAGE_DIGEST")
        self.miner_hotkey = submission.hotkey
        return signed.model_dump(mode="json")

    def handle_challenge(self, challenge: MinerChallenge) -> None:
        try:
            evidence = self.attest(bytes.fromhex(challenge.nonce))
            if self.turbo_submission is None:
                self.client.answer_challenge(challenge.challenge_id, evidence)
            else:
                self.client.answer_challenge(challenge.challenge_id, evidence, candidate=True)
        except AttestationUnavailable as exc:
            log.error("challenge %s failed: %s", challenge.challenge_id, exc)
        except (httpx.HTTPError, GatewayError, ValueError) as exc:
            log.warning("challenge %s failed: %s", challenge.challenge_id, type(exc).__name__)

    def handle_job(self, job: MinerJob) -> Receipt | None:
        self.busy = True
        try:
            with self._heartbeat(job.job_id):
                return self.process(job)
        except JobRejected as exc:
            self._discard_openings(job.job_id)
            self._fail(job.job_id, exc.code, exc.message)
        except JobCanceled:
            self._discard_openings(job.job_id)
            log.info("job %s canceled by the customer", job.job_id)
        except CapacityRefused as exc:
            # The message names the class, sizes and memory only. Outside the advertised envelope this costs the miner
            # nothing; inside it the gateway records internal_error (MINING.md §6).
            self._discard_openings(job.job_id)
            log.warning("job %s refused: %s", job.job_id, exc)
            self._fail(job.job_id, CAPACITY_REFUSED, str(exc))
        except Exception as exc:  # never log the message: it may echo request content
            self._discard_openings(job.job_id)
            log.error("job %s failed with %s", job.job_id, type(exc).__name__)
            self._fail(job.job_id, "internal_error", "Generation failed inside the worker.")
        finally:
            # A delivered job keeps its openings for the retention window; the store expires them.
            self._openings.pop(job.job_id, None)
            self._last_progress.pop(job.job_id, None)
            self._reported.pop(job.job_id, None)
            self._canceled.discard(job.job_id)
            self.busy = False
        return None

    def _discard_openings(self, job_id: str) -> None:
        handle = self._openings.pop(job_id, None)
        if handle is None:
            return
        try:
            handle.discard()
        except Exception as exc:  # never let cleanup mask the job's own failure
            log.warning("could not discard retained openings for job %s (%s)", job_id, type(exc).__name__)

    def _fail(self, job_id: str, code: str, message: str) -> None:
        try:
            self.client.fail(job_id, code, message)
        except (httpx.HTTPError, GatewayError):
            log.warning("could not report failure for job %s", job_id)

    def _progress(self, job_id: str, value: float, stage: str, force: bool = False) -> None:
        if job_id in self._canceled:  # heard by the heartbeat
            raise JobCanceled()
        value = min(max(value, 0.0), 1.0)
        self._reported[job_id] = (value, stage)
        now = time.time()
        if not force and now - self._last_progress.get(job_id, 0.0) < PROGRESS_INTERVAL_S:
            return
        self._last_progress[job_id] = now
        if self.client.progress(job_id, value, stage):
            raise JobCanceled()

    @contextmanager
    def _heartbeat(self, job_id: str):
        """Repeats the job's last progress report every HEARTBEAT_S while the job thread is busy, so the gateway keeps
        this enclave fresh through a long render. A cancel it hears is raised at the job's next progress report."""
        stop = threading.Event()

        def beat() -> None:
            while not stop.wait(HEARTBEAT_S):
                value, stage = self._reported.get(job_id, (0.0, "starting"))
                try:
                    if self.client.progress(job_id, value, stage):
                        self._canceled.add(job_id)
                except (httpx.HTTPError, GatewayError) as exc:
                    log.warning("heartbeat for job %s failed with %s", job_id, type(exc).__name__)

        thread = threading.Thread(target=beat, name=f"kuno-heartbeat-{job_id[:8]}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=1.0)

    # ------------------------------------------------------------ one job

    def process(self, job: MinerJob) -> Receipt:
        started = time.time()
        profile = self.profiles.get(job.params.profile_id)
        if profile is None:
            raise JobRejected("profile_not_served", "This worker does not serve that model.")
        if job.job_id in self._seen:
            raise JobRejected("replay", "This enclave already processed that job id.")
        self._seen.add(job.job_id)
        try:
            validate_params(profile, job.params)
        except ParamError as exc:
            raise JobRejected("invalid_params", str(exc)) from None
        table = self.serving_envelope().get(profile.id)
        if not fits(table, job.params):
            # Before any download or decryption. The gateway doesn't route such a job here, so this costs nothing.
            raise JobRejected(CAPACITY_REFUSED, f"This worker's hardware {describe(table, job.params)}.")
        backend = self.backend_for(profile)
        if job.params.mode is Mode.STORYBOARD and not getattr(backend, "storyboards", False):
            # The cold backends would render one clip of the stitched length: refuse rather than deliver that.
            raise JobRejected("internal_error", "This worker's backend does not render storyboards.")
        if job.params.mode is Mode.PLAN and not getattr(backend, "plans", False):
            # Registration didn't advertise plan/1, so the gateway shouldn't have routed it here.
            raise JobRejected("internal_error", "This worker's backend does not write plans.")

        enc, ciphertext = b64d(job.enc), b64d(job.ciphertext)
        aad = job_aad(job.job_id, self.identity.enclave_id, job.params, job.input_blob_ids)
        try:
            session = RecipientSession(self.identity.hpke_private, enc)
            # Either form: padded (strictly checked) or bare JSON from clients that predate padding.
            payload = open_payload(session, ciphertext, aad)
        except DecryptionError:
            raise JobRejected("decrypt_failed", "The request did not decrypt for this enclave (tampered or wrong key).") from None
        except (MalformedPayload, ValidationError):
            raise JobRejected("bad_payload", "The decrypted request is malformed.") from None
        model_prompts = self._model_prompts(job.params, payload)
        self._progress(job.job_id, 0.02, "decrypted", force=True)

        blobs = [self.client.download_blob(blob_id) for blob_id in job.input_blob_ids]
        inputs = self._open_inputs(job, payload, session.input_key, blobs)
        if job.params.mode is Mode.PLAN:
            return self._plan(job, profile, backend, payload, session, started, input_digest(enc, ciphertext, blobs))
        if any(len(prompt) > profile.limits.max_prompt_chars for prompt in (payload.prompt, *model_prompts)):
            raise JobRejected("prompt_too_long", f"Prompts are limited to {profile.limits.max_prompt_chars} characters.")
        if payload.negative_prompt and not profile.limits.negative_prompt:
            raise JobRejected("unsupported_option", f"{profile.name} does not use negative prompts.")
        enhance = bool(profile.limits.prompt_enhancer and payload.options.get(ENHANCE_PROMPT_OPTION))
        if enhance and job.params.mode is Mode.STORYBOARD:
            # Refused rather than enhanced shot by shot: each shot would be its own enhancer run (tens of seconds on
            # the GPU, unmeasured, times up to max_shots) and its own check, and shots rewritten one at a time are free
            # to describe the same scene differently across the joins. Shots render the prompts the customer wrote.
            raise JobRejected("unsupported_option", "Prompt enhancement is not available for storyboards.")
        try:
            for prompt in model_prompts:
                check_request(prompt, payload.negative_prompt)
        except SafetyViolation:
            raise JobRejected("safety_blocked", PROMPT_BLOCKED) from None

        width, height = profile.size_for(job.params.resolution, job.params.aspect_ratio)
        task = GenerationTask(
            job_id=job.job_id,
            profile=profile,
            params=job.params,
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            seed=payload.seed if payload.seed is not None else secrets.randbelow(2**31),
            width=width,
            height=height,
            inputs=inputs,
            options=payload.options,
            shot_prompts=model_prompts if job.params.mode is Mode.STORYBOARD else None,
        )
        # Before any enhancement, which counts as generating: a stage of its own would tell the gateway, which sees
        # progress but not the sealed options, that enhancement was asked for.
        self._progress(job.job_id, 0.05, "generating", force=True)
        checked = list(model_prompts)
        if enhance and getattr(backend, "prompt_enhancement", False):
            task = self._enhance(backend, task)
            checked.append(task.prompt)
        # Booleans only (e.g. "the prompt names a minor"), over every prompt checked: the customer's, and the enhanced
        # one, which may name what the customer's didn't. The frame check uses them to err toward blocking.
        signals = RequestSignals.combine(request_signals(prompt, payload.negative_prompt) for prompt in checked)
        result = backend.generate(task, lambda value, stage: self._progress(job.job_id, 0.05 + 0.85 * value, stage))
        if result.openings is not None:
            self._openings[job.job_id] = result.openings

        # Judge the rendered frames before anything is signed, sealed or uploaded.
        self._progress(job.job_id, 0.9, "checking", force=True)
        try:
            check_output(result.data, signals, shot_frames=task.shot_frames)
        except SafetyViolation:
            raise JobRejected("safety_blocked", "The video was blocked by the content policy.") from None
        except SafetyUnavailable:
            raise JobRejected("internal_error", "The worker could not run its content safety check.") from None

        self._progress(job.job_id, 0.92, "sealing", force=True)
        assert self.evidence is not None
        rendered = result.data
        # A draft describing the rendered file; the sealed-output fields are filled in below.
        draft = ReceiptBody(
            job_id=job.job_id,
            enclave_id=self.identity.enclave_id,
            profile_id=profile.id,
            image_digest=self.config.image_digest,
            params_digest=sha256_hex(canonical_json(job.params.model_dump(mode="json"))),
            input_digest=input_digest(enc, ciphertext, blobs),
            output_digest=sha256_hex(rendered),
            output_bytes=len(rendered),
            content_digest=sha256_hex(rendered),
            attestation_digest=self.evidence.digest(),
            started_at=started,
            finished_at=started,
            gpu_seconds=0.0,
            video=result.info,
            miner_hotkey=self.miner_hotkey,
            step_commitment=result.step_commitment,
        )
        # Provenance goes in before sealing, so the receipt's content digest covers the delivered file.
        final = self._embed_provenance(rendered, draft)
        sealed = encrypt_blob(session.output_key, output_label(job.job_id), final)
        blob_id = self.client.upload_blob(job.job_id, sealed)
        finished = time.time()
        body = draft.model_copy(update={
            "content_digest": sha256_hex(final),
            "output_digest": sha256_hex(sealed),
            "output_bytes": len(sealed),
            "finished_at": finished,
            "gpu_seconds": round((finished - started) * profile.gpus_per_worker, 3),
        })
        receipt = sign_receipt(self.identity.signing_key, body)
        self.client.complete(job.job_id, blob_id, receipt)
        return receipt

    def _embed_provenance(self, rendered: bytes, draft: ReceiptBody) -> bytes:
        if self._provenance_signer is None:
            return rendered
        from .provenance import ProvenanceError, embed_provenance

        try:
            return embed_provenance(rendered, draft, self._provenance_signer)
        except ProvenanceError as exc:
            log.error("C2PA embedding failed for job %s: %s", draft.job_id, exc)
            raise JobRejected("internal_error", "The worker could not sign the video's provenance.") from None

    def _build_provenance_signer(self):
        mode = self.config.provenance
        if mode == "off":
            return None
        if mode != "c2pa":
            raise ValueError(f"KUNO_PROVENANCE must be 'off' or 'c2pa', not {mode!r}")
        from .certificates import PROVISIONAL, CertifiedSigner
        from .provenance import ProvenanceSigner, _c2pa

        _c2pa()  # fail at start-up, not on the first job, when the extra is missing
        if self.config.provenance_cert_chain is not None:
            return ProvenanceSigner(self.identity.signing_key, self.config.provenance_cert_chain.read_text(), self.config.provenance_tsa_url)
        # The gateway's CA issues the certificate after each attestation (see _refresh_certificate).
        signer = CertifiedSigner(self.identity.signing_key, self.config.provenance_tsa_url)
        if self.config.tee == "mock":
            # Only reachable before the first registration (jobs arrive after it): the first answer from the
            # gateway replaces this with an issued certificate, or with a dev one only if the gateway has no CA.
            self._install_dev_certificate(signer, PROVISIONAL)
        return signer

    # ------------------------------------------------------------ plans

    def _plan(
        self, job: MinerJob, profile: ModelProfile, backend: Backend, payload: SealedPayload, session: RecipientSession,
        started: float, inputs_digest: str,
    ) -> Receipt:
        """A plan job (PROTOCOL.md "Plans (Director)"): the brief and style checked like prompts, a plan written by the
        backend's planner and repaired (`_draft_plan`), every shot's model prompt checked like a prompt and its labels
        against the content policy, with one regeneration after a block; then the canonical plan JSON sealed, padded, as
        `<job_id>/output/plan`, and a receipt with `plan` and no `video`. Nothing renders, so there is no frame check
        and no C2PA manifest."""
        limits = profile.limits.plan
        if limits is None:  # validate_params refuses plan mode on such a profile; kept for a hand-edited catalog
            raise JobRejected("invalid_params", f"{profile.name} does not support plan")
        try:
            options = PlanOptions.model_validate(payload.options.get(PLAN_OPTION) or {})
        except ValidationError:
            raise JobRejected("bad_payload", "The plan options are malformed.") from None
        brief, style = payload.prompt, options.style or ""
        if len(brief) > limits.max_brief_chars:
            raise JobRejected("prompt_too_long", f"Briefs are limited to {limits.max_brief_chars} characters.")
        if len(style) > limits.max_style_chars:
            raise JobRejected("prompt_too_long", f"Styles are limited to {limits.max_style_chars} characters.")
        if payload.negative_prompt:
            raise JobRejected("unsupported_option", "Plans take no negative prompt.")
        if not brief.strip() and options.revise is None:
            raise JobRejected("bad_payload", "A plan needs a brief.")
        # With no max_shot_s from the client, shots are planned to what this worker's own hardware renders at this size.
        params = job.params
        table = self.serving_envelope().get(profile.id)
        served = max_duration(table, params.resolution, params.aspect_ratio, params.fps) if table is not None else None
        try:
            context = plan_context(profile, params, options, served_max_s=served)
            if options.revise is not None:
                check_revision(options.revise, context)
        except PlanError as exc:  # the message names limits and shot numbers, never text
            raise JobRejected("bad_payload", f"The plan options don't fit this job: {exc}.") from None
        written = [brief, style, options.revise.instruction if options.revise is not None else ""]
        try:
            for text in written:
                if text.strip():
                    check_request(text)
        except SafetyViolation:
            raise JobRejected("safety_blocked", PROMPT_BLOCKED) from None

        width, height = profile.size_for(params.resolution, params.aspect_ratio)
        task = GenerationTask(
            job_id=job.job_id, profile=profile, params=params, prompt=brief, negative_prompt=None,
            seed=payload.seed if payload.seed is not None else secrets.randbelow(2**31), width=width, height=height,
            options=payload.options,
        )
        self._progress(job.job_id, 0.05, "planning", force=True)
        plan, tokens = self._checked_plan(backend, task, context, options)
        try:
            validate_plan(plan, profile, context=context)
        except PlanError:
            # repair and fit only deliver plans that pass; one that doesn't is a bug in this worker.
            log.error("job %s: the repaired plan failed validation", job.job_id)
            raise JobRejected("internal_error", "Generation failed inside the worker.") from None

        self._progress(job.job_id, 0.92, "sealing", force=True)
        assert self.evidence is not None
        plan_json, sealed = seal_plan(session.output_key, job.job_id, plan)
        blob_id = self.client.upload_blob(job.job_id, sealed)
        finished = time.time()
        body = ReceiptBody(
            job_id=job.job_id,
            enclave_id=self.identity.enclave_id,
            profile_id=profile.id,
            image_digest=self.config.image_digest,
            params_digest=sha256_hex(canonical_json(params.model_dump(mode="json"))),
            input_digest=inputs_digest,
            output_digest=sha256_hex(sealed),
            output_bytes=len(sealed),
            content_digest=sha256_hex(plan_json),
            attestation_digest=self.evidence.digest(),
            started_at=started,
            finished_at=finished,
            gpu_seconds=round((finished - started) * profile.gpus_per_worker, 3),
            miner_hotkey=self.miner_hotkey,
            plan=PlanInfo(
                shots=len(plan.shots), duration_s=plan.duration_s, planner=plan.planner.model,
                prompt_version=plan.planner.prompt_version, output_tokens=tokens,
            ),
        )
        receipt = sign_receipt(self.identity.signing_key, body)
        self.client.complete(job.job_id, blob_id, receipt)
        return receipt

    # Plans a job may write before it fails: the first, and one more after an output the safety checks blocked.
    PLAN_WRITES = 2

    def _checked_plan(self, backend: Backend, task: GenerationTask, context: PlanContext, options: PlanOptions) -> tuple[Plan, int]:
        """A plan whose text passed the checks, and the tokens every reply for it took. Each shot's model prompt,
        `shot_prompt(scene, prompt)`, is text a language model wrote for the video model, so it goes through
        `_generate_checked` like an enhanced prompt; the title, notes and beats are only shown to people, so they get the
        shared content policy. A block is written again once, from the next seeds, and a second block is
        `safety_blocked`. The planner refusing the brief is `safety_blocked` at once."""
        drafts: list[tuple[Plan, int]] = []

        def write() -> list[str]:
            plan, tokens = self._draft_plan(backend, task, context, options, attempt=len(drafts))
            drafts.append((plan, tokens))
            if len(drafts) == 1:
                self._progress(task.job_id, 0.85, "checking", force=True)
            return plan.model_prompts()

        for attempt in range(self.PLAN_WRITES):
            try:
                self._generate_checked(write, None)
                plan, _ = drafts[-1]
                for label in (plan.title, plan.notes, *(shot.beat for shot in plan.shots)):
                    if label.strip():
                        check_prompt(label)
                return plan, sum(tokens for _, tokens in drafts)
            except ContentPolicyViolation:
                blocked = JobRejected("safety_blocked", PROMPT_BLOCKED)
            except PlanRefused:
                raise
            except JobRejected as exc:
                if exc.code != "safety_blocked":
                    raise
                blocked = exc
            if attempt == self.PLAN_WRITES - 1:
                raise blocked
        raise AssertionError("unreachable")

    def _draft_plan(self, backend: Backend, task: GenerationTask, context: PlanContext, options: PlanOptions, attempt: int) -> tuple[Plan, int]:
        """One plan from the planner (kuno_protocol.plans): a reply, repaired; when it has problems (unparseable, fewer
        than 2 shots, more than 25% short, a quoted phrase of the brief missing), a single retry with the problems as a
        user turn, and the better of the two. A reply no repair or retry makes into a plan is `plan_failed`; problems a
        plan can live with are named in its `repairs`. Replies are seeded (seed + 2 × attempt + k) mod 2^31, so a job's
        plans reproduce."""
        limits = task.profile.limits.plan
        assert limits is not None
        messages = plan_messages(task.prompt, context, options)

        def ask(chat: list[dict[str, str]], offset: int):
            reply = backend.write_plan(task, chat, seed=(task.seed + 2 * attempt + offset) % 2**31, max_new_tokens=limits.max_new_tokens)
            result = repair(reply.text, context, planner=reply.planner, brief=task.prompt, revise=options.revise)
            if result.refusal:
                raise PlanRefused()
            return reply, result

        reply, result = ask(messages, 0)
        tokens = reply.output_tokens
        if result.problems:
            second_reply, second = ask(retry_messages(messages, reply.text, result.problems), 1)
            tokens += second_reply.output_tokens
            result = choose(result, second)
        if result.plan is None:
            raise JobRejected(PLAN_FAILED, PLAN_FAILED_MESSAGE)
        return result.deliverable(), tokens

    def _enhance(self, backend: Backend, task: GenerationTask) -> GenerationTask:
        """The task to render when the customer asked for prompt enhancement: the backend's rewrite of the prompt,
        checked like the customer's prompt, with the option removed so nothing downstream enhances again."""
        [enhanced] = self._generate_checked(lambda: [backend.enhance_prompt(task)], task.negative_prompt)
        options = {key: value for key, value in task.options.items() if key != ENHANCE_PROMPT_OPTION}
        return replace(task, prompt=enhanced, options=options)

    @staticmethod
    def _generate_checked(generate: Callable[[], list[str]], negative_prompt: str | None) -> list[str]:
        """Text a language model writes inside the enclave, held to the customer's own prompt check before anything
        conditions on it: the shared content policy, then the prompt classifier (`check_request`), each text with the
        negative prompt it will render beside. Every step that generates text with a loaded model goes through here: an
        enhanced prompt, and every shot prompt of a plan (`_checked_plan`). It reports no progress stage of its own, since the gateway
        sees stages, and whether a prompt was enhanced is sealed.

        A block is `safety_blocked` with the fixed message any prompt gets. A classifier that cannot answer fails the
        job as the miner's internal_error, as it does for the customer's prompt. The customer's length limit is not
        applied: the enhancer's own token budget bounds what it writes, and the customer can't shorten it."""
        texts = generate()
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            # A model that wrote nothing: a failure of this worker, not a prompt to render or to blame on the customer.
            raise BackendError("the language model returned no text")
        try:
            for text in texts:
                check_request(text, negative_prompt)
        except SafetyViolation:
            raise JobRejected("safety_blocked", PROMPT_BLOCKED) from None
        return texts

    @staticmethod
    def _model_prompts(params: GenerationParams, payload: SealedPayload) -> list[str]:
        """What the model is prompted with: the prompt, or a storyboard's `shot_prompt(scene, shot)` for every shot, in
        order. A storyboard needs exactly one non-empty shot prompt per shot; no other job has any."""
        if params.mode is not Mode.STORYBOARD:
            if payload.shots is not None:
                raise JobRejected("bad_payload", "Shot prompts are only for storyboards.")
            return [payload.prompt]
        shots = payload.shots or []
        if len(shots) != len(params.shots or []):
            raise JobRejected("bad_payload", "A storyboard needs one shot prompt per shot.")
        if any(not shot.prompt.strip() for shot in shots):
            raise JobRejected("bad_payload", "Every storyboard shot needs a prompt.")
        return [shot_prompt(payload.prompt, shot.prompt) for shot in shots]

    def _open_inputs(self, job: MinerJob, payload: SealedPayload, input_key: bytes, blobs: list[bytes]) -> list[InputFile]:
        refs = sorted(payload.inputs, key=lambda r: r.index)
        if [r.index for r in refs] != list(range(len(blobs))):
            raise JobRejected("bad_inputs", "The input manifest does not match the uploaded blobs.")
        if [r.role for r in refs] != list(job.params.input_roles):
            raise JobRejected("bad_inputs", "Input roles do not match the public parameters.")
        files = []
        for ref, blob in zip(refs, blobs):
            try:
                data = decrypt_blob(input_key, input_label(job.job_id, ref.index), blob)
            except DecryptionError:
                raise JobRejected("bad_inputs", f"Input {ref.index} failed authentication.") from None
            if sha256_hex(data) != ref.sha256 or len(data) != ref.size:
                raise JobRejected("bad_inputs", f"Input {ref.index} does not match its manifest entry.")
            mime = sniff_mime(data)
            if mime is None or mime not in ROLE_TYPES[ref.role]:
                raise JobRejected("unsupported_media", f"Input {ref.index} is not a supported {ref.role.value} file.")
            files.append(InputFile(ref=ref, data=data, mime=mime))
        return files
