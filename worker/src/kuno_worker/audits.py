"""Answering verified-mode step audits from inside the enclave.

A validator that wants to check step k of one of its own canary jobs asks the gateway,
which refuses any job the validator did not create and queues an audit work item for the
enclave that ran the job. This module turns such an item into an opening: the transcript,
the per-job salt, leaves 0, k-1 and k (or all leaves on request) with their Merkle proofs,
and the latents before and after step k. The opening is sealed with HPKE to the key the
validator put in the request and signed with the enclave key, so the gateway can neither
read nor forge it.

The enclave adds a second privacy check the gateway cannot bypass: when a job's sealed
payload named an audit key (`options["kuno_audit_key"] = audit_binding(pubkey)`), only that
key can open it. With `require_binding=True` jobs that named no key are never opened at all;
that becomes the production setting once client SDKs attach a decoy binding to every job,
so canaries stay indistinguishable (see VERIFIED_MODE.md, "Privacy").
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx

from kuno_protocol.canonical import b64d
from kuno_protocol.verified import (
    LatentRecord,
    LeafProof,
    MinerAudit,
    SealedOpening,
    StepOpening,
    audit_binding,
    inclusion_proof,
    leaf_hash,
    required_leaves,
    seal_opening,
)

from .gateway_client import GatewayClient, GatewayError
from .identity import EnclaveIdentity
from .verified import RetentionError, RetentionStore, shared_retention

log = logging.getLogger("kuno.worker.audits")


class AuditRefused(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class AuditCalls:
    """The miner audit endpoints, over the enclave-signed gateway client.

    These belong on GatewayClient (see the integration notes in VERIFIED_MODE.md); they live
    here until that file's owner adds them.
    """

    def __init__(self, client: GatewayClient):
        self.client = client

    def pull(self) -> list[MinerAudit]:
        items = self.client._send("GET", "/miner/v1/audits").json().get("audits", [])
        return [MinerAudit.model_validate(item) for item in items]

    def post_opening(self, sealed: SealedOpening) -> None:
        body = self.client._json(sealed.model_dump(mode="json"))
        self.client._send("POST", f"/miner/v1/audits/{sealed.audit_id}/opening", body, timeout=600)

    def fail(self, audit_id: str, code: str, message: str) -> None:
        body = self.client._json({"code": code, "message": message[:500]})
        self.client._send("POST", f"/miner/v1/audits/{audit_id}/fail", body)


class AuditResponder:
    def __init__(
        self,
        identity: EnclaveIdentity,
        store: RetentionStore | None = None,
        *,
        require_binding: bool = False,
        clock: Callable[[], float] = time.time,
    ):
        self.identity = identity
        self.store = store or shared_retention()
        self.require_binding = require_binding
        self.clock = clock

    def open(self, item: MinerAudit) -> SealedOpening:
        """Builds the sealed opening, or raises AuditRefused with a machine-readable code."""
        if item.expires_at <= self.clock():
            raise AuditRefused("expired", "The audit deadline has passed.")
        record = self.store.record(item.job_id)
        if record is None:
            raise AuditRefused("not_retained", "No retained trajectory for this job (expired, or not run in verified mode).")
        commitment = record.commitment
        if not 1 <= item.step < commitment.leaves:
            raise AuditRefused("bad_step", "The requested step is outside the committed trajectory.")
        try:
            recipient = b64d(item.recipient_public_key)
        except ValueError:
            recipient = b""
        if len(recipient) != 32:
            raise AuditRefused("bad_key", "The recipient key is not a 32-byte X25519 key.")
        if record.audit_binding is not None:
            if audit_binding(recipient) != record.audit_binding:
                raise AuditRefused("binding_mismatch", "This job only opens to the audit key its sealed request named.")
        elif self.require_binding:
            raise AuditRefused("unbound_job", "This job's sealed request named no audit key.")

        try:
            latents = {i: self.store.latents(item.job_id, i) for i in (item.step - 1, item.step)}
        except RetentionError as exc:
            raise AuditRefused(exc.code, str(exc)) from None
        hashes = [leaf_hash(leaf, record.salt) for leaf in record.leaves]
        indices = required_leaves(item.step, commitment.leaves, item.include_leaves)
        opening = StepOpening(
            audit_id=item.audit_id,
            job_id=item.job_id,
            enclave_id=self.identity.enclave_id,
            step=item.step,
            commitment=commitment,
            transcript=record.transcript,
            salt=record.salt.hex(),
            leaves=[record.leaves[i] for i in indices],
            proofs=[LeafProof(index=i, path=[p.hex() for p in inclusion_proof(hashes, i)]) for i in indices],
            latents=[
                LatentRecord(index=i, tensors=[spec for spec, _ in sorted(latents[i], key=lambda t: t[0].name)])
                for i in sorted(latents)
            ],
        )
        return seal_opening(self.identity.signing_key, opening, latents, recipient)

    def handle(self, item: MinerAudit, calls: AuditCalls) -> bool:
        """Answers one audit work item. Never logs latents, prompts or keys; returns True if an opening was posted."""
        try:
            sealed = self.open(item)
        except AuditRefused as exc:
            log.warning("audit %s for job %s refused: %s", item.audit_id, item.job_id, exc.code)
            self._report_failure(calls, item.audit_id, exc.code, exc.message)
            return False
        except Exception as exc:  # never the message: it could echo retained content
            log.error("audit %s for job %s failed with %s", item.audit_id, item.job_id, type(exc).__name__)
            self._report_failure(calls, item.audit_id, "internal_error", "The enclave could not build the opening.")
            return False
        try:
            calls.post_opening(sealed)
        except (httpx.HTTPError, GatewayError) as exc:
            log.warning("could not post the opening for audit %s (%s)", item.audit_id, type(exc).__name__)
            return False
        return True

    @staticmethod
    def _report_failure(calls: AuditCalls, audit_id: str, code: str, message: str) -> None:
        try:
            calls.fail(audit_id, code, message)
        except (httpx.HTTPError, GatewayError):
            log.warning("could not report the failed audit %s", audit_id)
