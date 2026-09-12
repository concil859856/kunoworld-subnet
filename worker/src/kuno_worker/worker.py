"""The enclave job loop: attest, pull sealed work, decrypt, generate, seal, sign."""

from __future__ import annotations

import logging
import secrets
import threading
import time

import httpx
from pydantic import ValidationError

from kuno_protocol.attestation import AttestationEvidence, TEEProvider, build_evidence
from kuno_protocol.blobs import decrypt_blob, encrypt_blob
from kuno_protocol.canonical import b64d, canonical_json, sha256_hex
from kuno_protocol.crypto import DecryptionError, RecipientSession
from kuno_protocol.media import ROLE_TYPES, sniff_mime
from kuno_protocol.profiles import ModelProfile, ParamError, load_profiles, validate_params
from kuno_protocol.receipts import Receipt, ReceiptBody, input_digest, sign_receipt
from kuno_protocol.schemas import MinerChallenge, MinerJob, SealedPayload, input_label, job_aad, output_label

from .backends.base import Backend, GenerationTask, InputFile
from .config import WorkerConfig
from .gateway_client import GatewayClient, GatewayError
from .identity import EnclaveIdentity
from .safety import SafetyViolation, check_request

log = logging.getLogger("kuno.worker")

PROGRESS_INTERVAL_S = 0.5


class JobRejected(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class JobCanceled(Exception):
    pass


class Worker:
    def __init__(
        self,
        config: WorkerConfig,
        tee: TEEProvider,
        backends: dict[str, Backend],
        identity: EnclaveIdentity | None = None,
        transport=None,
    ):
        catalog = load_profiles()
        unknown = [p for p in config.profiles if p not in catalog]
        if unknown:
            raise ValueError(f"unknown profiles: {', '.join(unknown)}")
        self.config = config
        self.tee = tee
        self.backends = backends
        self.profiles: dict[str, ModelProfile] = {p: catalog[p] for p in config.profiles}
        for profile in self.profiles.values():
            self.backend_for(profile)
        self.identity = identity or EnclaveIdentity.generate()
        self.client = GatewayClient(config.gateway_url, self.identity.signing_key, self.identity.enclave_id, transport=transport)
        self.evidence: AttestationEvidence | None = None
        self.last_attested = 0.0
        self.ready = threading.Event()
        self.busy = False
        self._seen: set[str] = set()
        self._last_progress: dict[str, float] = {}

    # ------------------------------------------------------------ attestation

    def backend_for(self, profile: ModelProfile) -> Backend:
        backend = self.backends.get(profile.family) or self.backends.get("*")
        if backend is None:
            raise ValueError(f"no backend configured for {profile.family}")
        return backend

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
        evidence = self.attest(self.client.nonce())
        self.client.register(evidence, self.config.miner_hotkey, self.config.capacity)
        self.evidence = evidence
        self.last_attested = time.time()
        self.ready.set()
        log.info("attested enclave %s for %s", self.identity.enclave_id, ", ".join(self.profiles))

    # ------------------------------------------------------------ main loop

    def run(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        for profile in self.profiles.values():
            self.backend_for(profile).warm(profile)
        backoff = 1.0
        while not stop.is_set():
            try:
                if not self.ready.is_set() or time.time() - self.last_attested > self.config.reattest_s:
                    self.register()
                work = self.client.pull(wait=self.config.pull_wait_s)
                backoff = 1.0
            except (httpx.HTTPError, GatewayError) as exc:
                log.warning("gateway unavailable (%s); retrying in %.0fs", type(exc).__name__, backoff)
                stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            kind = work.get("kind")
            if kind == "job":
                self.handle_job(MinerJob.model_validate(work))
            elif kind == "challenge":
                self.handle_challenge(MinerChallenge.model_validate(work))

    def handle_challenge(self, challenge: MinerChallenge) -> None:
        try:
            self.client.answer_challenge(challenge.challenge_id, self.attest(bytes.fromhex(challenge.nonce)))
        except (httpx.HTTPError, GatewayError, ValueError) as exc:
            log.warning("challenge %s failed: %s", challenge.challenge_id, type(exc).__name__)

    def handle_job(self, job: MinerJob) -> Receipt | None:
        self.busy = True
        try:
            return self.process(job)
        except JobRejected as exc:
            self._fail(job.job_id, exc.code, exc.message)
        except JobCanceled:
            log.info("job %s canceled by the customer", job.job_id)
        except Exception as exc:  # never log the message: it may echo request content
            log.error("job %s failed with %s", job.job_id, type(exc).__name__)
            self._fail(job.job_id, "internal_error", "Generation failed inside the worker.")
        finally:
            self._last_progress.pop(job.job_id, None)
            self.busy = False
        return None

    def _fail(self, job_id: str, code: str, message: str) -> None:
        try:
            self.client.fail(job_id, code, message)
        except (httpx.HTTPError, GatewayError):
            log.warning("could not report failure for job %s", job_id)

    def _progress(self, job_id: str, value: float, stage: str, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_progress.get(job_id, 0.0) < PROGRESS_INTERVAL_S:
            return
        self._last_progress[job_id] = now
        if self.client.progress(job_id, min(max(value, 0.0), 1.0), stage):
            raise JobCanceled()

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

        enc, ciphertext = b64d(job.enc), b64d(job.ciphertext)
        aad = job_aad(job.job_id, self.identity.enclave_id, job.params, job.input_blob_ids)
        try:
            session = RecipientSession(self.identity.hpke_private, enc)
            payload = SealedPayload.model_validate_json(session.open(ciphertext, aad))
        except DecryptionError:
            raise JobRejected("decrypt_failed", "The request did not decrypt for this enclave (tampered or wrong key).") from None
        except ValidationError:
            raise JobRejected("bad_payload", "The decrypted request is malformed.") from None
        self._progress(job.job_id, 0.02, "decrypted", force=True)

        blobs = [self.client.download_blob(blob_id) for blob_id in job.input_blob_ids]
        inputs = self._open_inputs(job, payload, session.input_key, blobs)
        if len(payload.prompt) > profile.limits.max_prompt_chars:
            raise JobRejected("prompt_too_long", f"Prompts are limited to {profile.limits.max_prompt_chars} characters.")
        if payload.negative_prompt and not profile.limits.negative_prompt:
            raise JobRejected("unsupported_option", f"{profile.name} does not use negative prompts.")
        try:
            check_request(payload.prompt, payload.negative_prompt)
        except SafetyViolation:
            raise JobRejected("safety_blocked", "The request was blocked by the content policy.") from None

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
        )
        self._progress(job.job_id, 0.05, "generating", force=True)
        result = self.backend_for(profile).generate(
            task, lambda value, stage: self._progress(job.job_id, 0.05 + 0.85 * value, stage)
        )

        self._progress(job.job_id, 0.92, "sealing", force=True)
        sealed = encrypt_blob(session.output_key, output_label(job.job_id), result.data)
        blob_id = self.client.upload_blob(job.job_id, sealed)
        finished = time.time()
        assert self.evidence is not None
        body = ReceiptBody(
            job_id=job.job_id,
            enclave_id=self.identity.enclave_id,
            profile_id=profile.id,
            image_digest=self.config.image_digest,
            params_digest=sha256_hex(canonical_json(job.params.model_dump(mode="json"))),
            input_digest=input_digest(enc, ciphertext, blobs),
            output_digest=sha256_hex(sealed),
            output_bytes=len(sealed),
            content_digest=sha256_hex(result.data),
            attestation_digest=self.evidence.digest(),
            started_at=started,
            finished_at=finished,
            gpu_seconds=round((finished - started) * profile.gpus_per_worker, 3),
            video=result.info,
            miner_hotkey=self.config.miner_hotkey,
        )
        receipt = sign_receipt(self.identity.signing_key, body)
        self.client.complete(job.job_id, blob_id, receipt)
        return receipt

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
