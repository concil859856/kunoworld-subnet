"""Validator roles: the main validator tests miners and publishes signed findings; an auditor sends no jobs, verifies the
published evidence itself (with spot challenges), applies only findings its main validator signed, and measures how far
its weights are from the main validator's."""

from __future__ import annotations

import json
import os
import time

import httpx
import pytest

from kuno_protocol.attestation import AllowedMeasurement, GoldenManifest, MockTEE, build_evidence, mock_measurements
from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import generate_signing_key, public_key_bytes
from kuno_protocol.devkit import DEV_IMAGE_DIGEST
from kuno_protocol.findings import Finding, FindingsReport, SignedFindings, sign_findings, verify_findings
from kuno_protocol.hotkey import Sr25519Signer
from kuno_validator.validator import CanaryResult, Validator, weight_divergence

from test_receipt_ledger import NOW, FakeEnclave
from test_validator import API_KEY, FakeGateway

MAIN = Sr25519Signer.from_seed(os.urandom(32))
QUOTE_KEY = generate_signing_key()
MANIFEST = GoldenManifest(
    mock_quote_keys=[b64e(public_key_bytes(QUOTE_KEY))],
    allowed=[AllowedMeasurement(platform="mock", image_digest=DEV_IMAGE_DIGEST, profiles=["ltx-2.5-fast"], **mock_measurements(DEV_IMAGE_DIGEST))],
)


class AttestedEnclave(FakeEnclave):
    """An enclave whose feed entry carries real (simulated-TEE) evidence bound to its keys, as the gateway publishes."""

    def __init__(self, hotkey: str, evidence: bool = True):
        super().__init__(hotkey)
        self.with_evidence = evidence

    def public(self) -> dict:
        row = {**super().public(), "tee": "mock", "tier": "confidential", "profiles": ["ltx-2.5-fast"], "capacity": 1}
        if self.with_evidence:
            evidence = build_evidence(MockTEE(QUOTE_KEY, DEV_IMAGE_DIGEST), os.urandom(32), self.hpke_public, self.signing_public,
                                      DEV_IMAGE_DIGEST, ["ltx-2.5-fast"])
            row["evidence"] = evidence.model_dump(mode="json")
        return row


def live_row(enclave: FakeEnclave) -> dict:
    return enclave.entry(age_s=60.0 + (NOW - time.time()))


class RolesGateway(FakeGateway):
    """FakeGateway plus the findings relay."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.findings: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/validator/v1/findings":
            self.requests.append(request)
            if request.method == "POST":
                self.findings.append(json.loads(request.content))
                return httpx.Response(201, json={"accepted": True})
            return httpx.Response(200, json=self.findings)
        return super().__call__(request)


def validator(gateway, **kwargs) -> Validator:
    return Validator("http://gateway.test", API_KEY, MANIFEST, None, transport=httpx.MockTransport(gateway), **kwargs)


def signed_report(signer: Sr25519Signer, findings: list[Finding], weights: dict[str, float] | None = None, issued_at: float | None = None) -> dict:
    report = FindingsReport(validator_hotkey=signer.ss58_address, issued_at=issued_at or time.time(), window_s=86400.0,
                            findings=findings, weights=weights)
    return sign_findings(signer, report).model_dump(mode="json")


def canary_failure(hotkey: str, job_id: str = "job-x", at: float | None = None) -> Finding:
    return Finding(kind="canary_failed", miner_hotkey=hotkey, detail="receipt signs different params", at=at or time.time(),
                   job_id=job_id, profile_id="ltx-2.5-fast")


def paths(gateway) -> list[tuple[str, str]]:
    return [(r.method, r.url.path) for r in gateway.requests]


# ---------------------------------------------------------------- auditor


def test_an_auditor_sends_no_jobs_and_attests_from_published_evidence():
    a, b = AttestedEnclave("5A"), AttestedEnclave("5B")
    gateway = RolesGateway(enclaves=[a, b], ledger=[live_row(a), live_row(b)])
    auditor = validator(gateway, role="auditor", main_validator_hotkey=MAIN.ss58_address, spot_check_rate=0.0)
    weights = auditor.step(canary_profiles=["ltx-2.5-fast"])  # ignored: auditors send no canaries
    assert set(weights) == {"5A", "5B"} and all(w > 0 for w in weights.values())
    sent = paths(gateway)
    assert not any(path.startswith("/v1/videos") or path.startswith("/v1/blobs") for _, path in sent)
    assert ("POST", "/validator/v1/challenges") not in sent
    assert ("POST", "/validator/v1/findings") not in sent


def test_an_enclave_without_valid_published_evidence_earns_nothing_from_an_auditor():
    good, bare = AttestedEnclave("5Good"), AttestedEnclave("5Bare", evidence=False)
    forged = AttestedEnclave("5Forged")
    gateway = RolesGateway(enclaves=[good, bare, forged], ledger=[live_row(good), live_row(bare), live_row(forged)])
    rogue_key = generate_signing_key()  # evidence "signed" by a quote key the manifest doesn't trust
    rows = gateway.enclaves
    forged_row = next(r for r in rows if r["miner_hotkey"] == "5Forged")
    forged_row["evidence"] = build_evidence(MockTEE(rogue_key, DEV_IMAGE_DIGEST), os.urandom(32), forged.hpke_public,
                                            forged.signing_public, DEV_IMAGE_DIGEST, ["ltx-2.5-fast"]).model_dump(mode="json")
    auditor = validator(gateway, role="auditor", main_validator_hotkey=MAIN.ss58_address, spot_check_rate=0.0)
    verdicts = auditor.published_verdicts()
    assert verdicts[good.enclave_id].ok
    assert verdicts[bare.enclave_id].reasons == ["the gateway publishes no evidence for this enclave"]
    assert not verdicts[forged.enclave_id].ok
    assert set(auditor.step()) == {"5Good"}


def test_published_evidence_under_another_enclaves_keys_is_refused():
    a, b = AttestedEnclave("5A"), AttestedEnclave("5B")
    gateway = RolesGateway(enclaves=[a, b])
    gateway.enclaves[1]["evidence"] = gateway.enclaves[0]["evidence"]  # B's feed row serving A's evidence
    auditor = validator(gateway, role="auditor", spot_check_rate=0.0)
    verdict = auditor.published_verdicts()[b.enclave_id]
    assert not verdict.ok and "different keys than the registered enclave" in verdict.reasons[-1]


def test_spot_challenges_override_published_evidence_an_enclave_no_longer_answers():
    a = AttestedEnclave("5A")
    gateway = RolesGateway(enclaves=[a], ledger=[live_row(a)])  # its challenges come back expired
    auditor = validator(gateway, role="auditor", main_validator_hotkey=MAIN.ss58_address, spot_check_rate=1.0)
    verdicts = auditor.published_verdicts(spot_timeout_s=0.5)
    assert ("POST", "/validator/v1/challenges") in paths(gateway)
    assert not verdicts[a.enclave_id].ok
    assert auditor.step() == {}


def test_an_auditor_applies_findings_its_main_validator_signed_and_nothing_else():
    caught, clean, framed = AttestedEnclave("5Caught"), AttestedEnclave("5Clean"), AttestedEnclave("5Framed")
    gateway = RolesGateway(enclaves=[caught, clean, framed], ledger=[live_row(caught), live_row(clean), live_row(framed)])
    impostor = Sr25519Signer.from_seed(os.urandom(32))
    gateway.findings = [
        signed_report(MAIN, [canary_failure("5Caught")]),
        signed_report(impostor, [canary_failure("5Framed")]),  # validly signed, by the wrong validator
    ]
    tampered = SignedFindings.model_validate(signed_report(MAIN, [canary_failure("5Clean")])).model_dump(mode="json")
    tampered["report"]["findings"][0]["miner_hotkey"] = "5Framed"  # a relay retargeting a real finding
    gateway.findings.append(tampered)
    auditor = validator(gateway, role="auditor", main_validator_hotkey=MAIN.ss58_address, spot_check_rate=0.0)
    weights = auditor.step()
    assert set(weights) == {"5Clean", "5Framed"}
    reasons = auditor.main_validator_findings(time.time(), 86400.0)
    assert list(reasons) == ["5Caught"] and "[main validator]" in reasons["5Caught"][0]


def test_findings_outside_the_window_are_not_applied_and_duplicates_count_once():
    miner = AttestedEnclave("5Miner")
    gateway = RolesGateway(enclaves=[miner], ledger=[live_row(miner)])
    old = time.time() - 3 * 86400
    gateway.findings = [signed_report(MAIN, [canary_failure("5Miner", at=old)], issued_at=old)]
    auditor = validator(gateway, role="auditor", main_validator_hotkey=MAIN.ss58_address, spot_check_rate=0.0)
    assert auditor.main_validator_findings(time.time(), 86400.0) == {}
    gateway.findings = [signed_report(MAIN, [canary_failure("5Miner")]), signed_report(MAIN, [canary_failure("5Miner")])]
    assert len(auditor.main_validator_findings(time.time(), 86400.0)["5Miner"]) == 1


def test_an_auditor_without_a_main_validator_hotkey_applies_no_findings(caplog):
    miner = AttestedEnclave("5Miner")
    gateway = RolesGateway(enclaves=[miner], ledger=[live_row(miner)])
    gateway.findings = [signed_report(MAIN, [canary_failure("5Miner")])]
    auditor = validator(gateway, role="auditor", spot_check_rate=0.0)
    assert set(auditor.step()) == {"5Miner"}
    assert "no main validator hotkey configured" in caplog.text


def test_an_auditor_measures_how_far_its_weights_are_from_the_main_validators(caplog):
    a, b = AttestedEnclave("5A"), AttestedEnclave("5B")
    gateway = RolesGateway(enclaves=[a, b], ledger=[live_row(a), live_row(b)])
    gateway.findings = [signed_report(MAIN, [], weights={"5A": 1.0})]
    auditor = validator(gateway, role="auditor", main_validator_hotkey=MAIN.ss58_address, spot_check_rate=0.0)
    weights = auditor.step()
    assert weights == pytest.approx({"5A": 0.5, "5B": 0.5})
    assert auditor.last_divergence == pytest.approx(0.5)
    assert "differ from the main validator's by 50.0%" in caplog.text


# ---------------------------------------------------------------- main validator


def test_the_main_validator_publishes_its_attributable_failures_and_weights_signed():
    caught, clean = AttestedEnclave("5Caught"), AttestedEnclave("5Clean")
    gateway = RolesGateway(enclaves=[caught, clean], ledger=[live_row(caught), live_row(clean)])
    main = validator(gateway, role="main", findings_signer=MAIN)
    now = time.time()
    main.canary_results = [
        CanaryResult("ltx-2.5-fast", False, "receipt signs different params", "job-1", caught.enclave_id, "5Caught", True, now),
        CanaryResult("ltx-2.5-fast", False, "gateway lost the job", "job-2", clean.enclave_id, "5Clean", False, now),  # not attributable
        CanaryResult("ltx-2.5-fast", True, "ok", "job-3", clean.enclave_id, "5Clean", True, now),
    ]
    weights = main.step()
    (published,) = gateway.findings
    signed = SignedFindings.model_validate(published)
    assert verify_findings(signed, MAIN.ss58_address)[0]
    assert [(f.kind, f.miner_hotkey, f.job_id) for f in signed.report.findings] == [("canary_failed", "5Caught", "job-1")]
    assert signed.report.weights == pytest.approx(weights) and "5Caught" not in weights


def test_a_main_validator_without_a_hotkey_publishes_nothing(caplog):
    a = AttestedEnclave("5A")
    gateway = RolesGateway(enclaves=[a], ledger=[live_row(a)])
    validator(gateway, role="main").step()
    assert gateway.findings == [] and "no hotkey to sign findings with" in caplog.text


def test_roles_and_rates_are_checked():
    with pytest.raises(ValueError, match="role"):
        validator(RolesGateway(), role="observer")
    with pytest.raises(ValueError, match="spot-check rate"):
        validator(RolesGateway(), role="auditor", spot_check_rate=1.5)


def test_weight_divergence_is_the_share_of_weight_that_would_move():
    assert weight_divergence({}, {}) == 0.0
    assert weight_divergence({"a": 2.0, "b": 2.0}, {"a": 0.5, "b": 0.5}) == pytest.approx(0.0)
    assert weight_divergence({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
    assert weight_divergence({"a": 0.75, "b": 0.25}, {"a": 0.25, "b": 0.75}) == pytest.approx(0.5)
