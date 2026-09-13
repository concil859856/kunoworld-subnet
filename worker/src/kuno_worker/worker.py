"""The enclave job loop: attest, pull sealed work, decrypt, generate, seal, sign."""

from __future__ import annotations

import logging
import secrets
import threading
import time

import httpx
from pydantic import ValidationError

from kuno_protocol.attestation import AttestationEvidence, AttestationUnavailable, TEEProvider, build_evidence
from kuno_protocol.blobs import decrypt_blob, encrypt_blob
from kuno_protocol.canonical import b64d, canonical_json, sha256_hex
from kuno_protocol.crypto import DecryptionError, RecipientSession
from kuno_protocol.hotkey import HotkeySigner, sign_hotkey_proof
from kuno_protocol.media import ROLE_TYPES, sniff_mime
from kuno_protocol.profiles import ModelProfile, ParamError, load_profiles, validate_params
from kuno_protocol.receipts import Receipt, ReceiptBody, input_digest, sign_receipt
from kuno_protocol.schemas import MinerChallenge, MinerJob, SealedPayload, input_label, job_aad, output_label
from kuno_protocol.verified import MinerAudit

from .audits import AuditCalls, AuditResponder
from .backends.base import Backend, GenerationTask, InputFile
from .config import WorkerConfig
from .gateway_client import GatewayClient, GatewayError
from .identity import EnclaveIdentity
from .safety import SafetyUnavailable, SafetyViolation, check_output, check_request, request_signals

log = logging.getLogger("kuno.worker")

PROGRESS_INTERVAL_S = 0.5


class JobRejected(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


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
        self.client = GatewayClient(config.gateway_url, self.identity.signing_key, self.identity.enclave_id, transport=transport)
        self.evidence: AttestationEvidence | None = None
        self.last_attested = 0.0
        self.ready = threading.Event()
        self.busy = False
        self._seen: set[str] = set()
        self._last_progress: dict[str, float] = {}
        # Verified mode: retained step openings per job, discarded if the job fails after generation.
        self._openings: dict[str, object] = {}
        self.audits = AuditResponder(self.identity)

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
        nonce = self.client.nonce()
        evidence = self.attest(nonce)
        proof = None
        if self.hotkey is not None:
            proof = sign_hotkey_proof(self.hotkey, nonce, self.identity.enclave_id, self.identity.signing_public)
        if self.turbo_submission is None:
            self.client.register(evidence, self.miner_hotkey, self.config.capacity, proof)
        else:
            self.client.register(evidence, self.miner_hotkey, self.config.capacity, proof, turbo_submission=self.turbo_submission)
        self.evidence = evidence
        self.last_attested = time.time()
        self._refresh_certificate()
        self.ready.set()
        log.info("attested enclave %s for %s", self.identity.enclave_id, ", ".join(self.profiles))

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
        tsa_url = signer.configured_tsa_url or response.get("tsa_url")
        if tsa_url is None and self.config.tee != "mock":
            # Without a timestamp, readers reject every manifest once the short-lived certificate expires,
            # so a customer's video would stop verifying a day after delivery. Refuse instead.
            self._certificate_unavailable(
                signer, current, now,
                MissingTimestampAuthority("no RFC 3161 timestamp authority: set KUNO_PROVENANCE_TSA_URL or the gateway's KUNO_C2PA_TSA_URL"),
            )
            return
        signer.install(issued, response.get("tsa_url"))
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
            return self.process(job)
        except JobRejected as exc:
            self._discard_openings(job.job_id)
            self._fail(job.job_id, exc.code, exc.message)
        except JobCanceled:
            self._discard_openings(job.job_id)
            log.info("job %s canceled by the customer", job.job_id)
        except Exception as exc:  # never log the message: it may echo request content
            self._discard_openings(job.job_id)
            log.error("job %s failed with %s", job.job_id, type(exc).__name__)
            self._fail(job.job_id, "internal_error", "Generation failed inside the worker.")
        finally:
            # A delivered job keeps its openings for the retention window; the store expires them.
            self._openings.pop(job.job_id, None)
            self._last_progress.pop(job.job_id, None)
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
        # Booleans only (e.g. "the prompt names a minor"); the frame check uses them to err toward blocking.
        signals = request_signals(payload.prompt, payload.negative_prompt)

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
        if result.openings is not None:
            self._openings[job.job_id] = result.openings

        # Judge the rendered frames before anything is signed, sealed or uploaded.
        self._progress(job.job_id, 0.9, "checking", force=True)
        try:
            check_output(result.data, signals)
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
