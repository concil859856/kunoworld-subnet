"""Findings reports: the main validator signs them with its hotkey; only a report its hotkey signed, unchanged, verifies."""

from __future__ import annotations

import os

import pytest

from kuno_protocol.canonical import b64d, b64e
from kuno_protocol.findings import Finding, FindingsReport, SignedFindings, findings_message, sign_findings, verify_findings
from kuno_protocol.hotkey import _POLKADOT_WRAP, Sr25519Signer

MAIN = Sr25519Signer.from_seed(os.urandom(32))


def report(signer: Sr25519Signer = MAIN, **changes) -> FindingsReport:
    finding = Finding(kind="canary_failed", miner_hotkey="5Miner", detail="receipt signs different params", at=100.0,
                      job_id="job-1", enclave_id="e" * 32, profile_id="ltx-2.5-fast")
    return FindingsReport(validator_hotkey=signer.ss58_address, issued_at=200.0, window_s=86400.0, findings=[finding],
                          weights={"5Miner": 0.0, "5Other": 1.0}, **changes)


def test_a_report_signed_by_the_main_validator_verifies():
    assert verify_findings(sign_findings(MAIN, report()), MAIN.ss58_address) == (True, "signed by the main validator")


def test_another_validators_report_is_refused_even_if_validly_signed():
    other = Sr25519Signer.from_seed(os.urandom(32))
    ok, detail = verify_findings(sign_findings(other, report(other)), MAIN.ss58_address)
    assert not ok and "not the main validator" in detail


def test_a_changed_finding_or_weight_breaks_the_signature():
    signed = sign_findings(MAIN, report())
    for change in (
        lambda d: d["report"]["findings"][0].update(miner_hotkey="5Innocent"),
        lambda d: d["report"]["weights"].update({"5Miner": 1.0}),
        lambda d: d["report"].update(window_s=1.0),
    ):
        document = signed.model_dump(mode="json")
        change(document)
        ok, detail = verify_findings(SignedFindings.model_validate(document), MAIN.ss58_address)
        assert not ok and "does not verify" in detail


def test_a_report_must_name_the_hotkey_that_signs_it():
    other = Sr25519Signer.from_seed(os.urandom(32))
    with pytest.raises(ValueError, match="name the hotkey"):
        sign_findings(other, report(MAIN))


def test_wallet_signatures_over_wrapped_bytes_also_verify():
    body = report()
    signature = MAIN.sign(_POLKADOT_WRAP[0] + findings_message(body) + _POLKADOT_WRAP[1])
    assert verify_findings(SignedFindings(report=body, signature=b64e(bytes(signature))), MAIN.ss58_address)[0]


def test_malformed_signatures_are_refused_with_a_reason():
    signed = sign_findings(MAIN, report())
    short = signed.model_copy(update={"signature": b64e(b64d(signed.signature)[:32])})
    assert verify_findings(short, MAIN.ss58_address) == (False, "sr25519 signatures are 64 bytes")
