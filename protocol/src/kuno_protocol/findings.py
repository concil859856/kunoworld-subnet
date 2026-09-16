"""Findings: what the main validator penalised, signed so auditor validators can apply it without trusting the relay.

KunoWorld runs one main validator. It assigns the work that tests miners: challenges with its own nonces, canary jobs
indistinguishable from customer traffic, and step audits of those canaries. Every other validator is an auditor. It
sends no jobs, so it can't see a canary fail. To score the same way, it needs to know which miners the main validator
caught, and it must not take the gateway's word for that. So the main validator signs a report of its attributable
failures each round with its hotkey, the gateway relays it, and auditors check the signature against the main
validator hotkey they were configured with (VALIDATING.md, "Validator roles").

A finding carries what an auditor needs to judge it independently where possible: the job id (whose receipt is in the
public ledger) and the enclave. An auditor that can re-check a finding and disagrees logs the disagreement; a finding
it can't check is applied on the main validator's signature, because that is the design: the main validator tests,
auditors audit it.

    message = "kuno/v1/findings\\n" + canonical_json(report)
    signature = sr25519(main validator hotkey, message)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .canonical import b64d, b64e, canonical_json
from .hotkey import HotkeyError, HotkeySigner, ss58_decode

FINDINGS_VERSION = 1
# Attestation failures aren't reported: auditors verify attestation themselves.
FindingKind = Literal["canary_failed", "audit_failed"]
# A report stays under the gateway's 1 MB JSON body limit.
MAX_FINDINGS = 1_000


class Finding(BaseModel):
    kind: FindingKind
    miner_hotkey: str
    detail: str = Field(max_length=500)
    at: float
    job_id: str | None = None
    enclave_id: str | None = None
    profile_id: str | None = None


class FindingsReport(BaseModel):
    v: Literal[1] = FINDINGS_VERSION
    validator_hotkey: str
    issued_at: float
    window_s: float = Field(gt=0)
    findings: list[Finding] = Field(default_factory=list, max_length=MAX_FINDINGS)
    # The main validator's normalized serving weights this round (hotkey -> weight), so an auditor can measure how far
    # its own are. Validators with gateway access therefore see them before commit-reveal would: auditors follow the
    # main validator by design, so hiding its weights from them protects nothing this subnet relies on.
    weights: dict[str, float] | None = None


class SignedFindings(BaseModel):
    report: FindingsReport
    signature: str


def findings_message(report: FindingsReport) -> bytes:
    return b"kuno/v1/findings\n" + canonical_json(report.model_dump(mode="json"))


def sign_findings(signer: HotkeySigner, report: FindingsReport) -> SignedFindings:
    if report.validator_hotkey != signer.ss58_address:
        raise ValueError("a findings report must name the hotkey that signs it")
    return SignedFindings(report=report, signature=b64e(bytes(signer.sign(findings_message(report)))))


def verify_findings(signed: SignedFindings, main_validator_hotkey: str) -> tuple[bool, str]:
    """(ok, detail): the report names the configured main validator and its sr25519 signature over it verifies."""
    if signed.report.validator_hotkey != main_validator_hotkey:
        return False, f"signed by {signed.report.validator_hotkey}, not the main validator {main_validator_hotkey}"
    try:
        public_key = ss58_decode(main_validator_hotkey)
        signature = b64d(signed.signature)
    except (HotkeyError, ValueError) as exc:
        return False, f"malformed findings signature: {exc}"
    if len(signature) != 64:
        return False, "sr25519 signatures are 64 bytes"
    from .hotkey import _POLKADOT_WRAP, _sr25519

    message = findings_message(signed.report)
    # Wallet tools that sign arbitrary bytes may wrap them the way polkadot.js does; hotkey proofs accept both too.
    if any(_sr25519().verify(signature, candidate, public_key) for candidate in (message, _POLKADOT_WRAP[0] + message + _POLKADOT_WRAP[1])):
        return True, "signed by the main validator"
    return False, "the findings signature does not verify against the main validator hotkey"
