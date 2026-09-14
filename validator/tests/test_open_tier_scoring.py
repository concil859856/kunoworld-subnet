"""Validator scoring by tier: open-tier work earns at a lower rate, only after admission probes, needs more
collateral per GPU, and a private-mode receipt from an open-tier enclave is fraud."""

from __future__ import annotations

import time

import httpx
import pytest

from kuno_protocol.attestation import GoldenManifest, Verdict
from kuno_protocol.canonical import canonical_json, sha256_hex
from kuno_protocol.receipts import ReceiptBody, VideoInfo, sign_receipt
from kuno_protocol.switch import SwitchConfig
from kuno_validator.collateral import RAO_PER_ALPHA, CollateralGate
from kuno_validator.open_tier import AdmissionTracker, TierPolicy, apply_tiers, fraud_penalties, open_tier_gpus
from kuno_validator.scoring import compute_scores, normalize
from kuno_validator.validator import CanaryResult, Validator

from test_receipt_ledger import NOW, PROFILES, FakeEnclave
from test_tolerance_audits import NoisyToyMiner, ledger_row, gateway_job
from test_validator import API_KEY, FakeGateway


class OpenEnclave(FakeEnclave):
    def public(self) -> dict:
        return {**super().public(), "tee": "open", "tier": "open", "profiles": ["ltx-2.5-fast"], "capacity": 2, "hardware": {"gpu_count": 1}}


class ConfidentialEnclave(FakeEnclave):
    def public(self) -> dict:
        return {**super().public(), "tee": "tdx", "tier": "confidential", "profiles": ["ltx-2.5-fast"], "capacity": 1}


def live(enclave: FakeEnclave, privacy: str, **kwargs) -> dict:
    """A ledger row finished a minute ago, in real time, since Validator.score uses the clock."""
    return {**enclave.entry(age_s=kwargs.pop("age_s", 60.0) + (NOW - time.time()), **kwargs), "privacy": privacy}


class Chain:
    def __init__(self, locked: dict[str, float]):
        self.locked = {hotkey: int(alpha * RAO_PER_ALPHA) for hotkey, alpha in locked.items()}

    def locked_collateral(self, netuid, hotkeys):
        return {hotkey: self.locked.get(hotkey, 0) for hotkey in hotkeys}


def make_validator(gateway, **kwargs) -> Validator:
    return Validator("http://gateway.test", API_KEY, GoldenManifest(), None, transport=httpx.MockTransport(gateway), **kwargs)


def verdicts(*enclaves) -> dict[str, Verdict]:
    return {e.enclave_id: Verdict(True, e.enclave_id, tier="open" if isinstance(e, OpenEnclave) else "confidential", gpu_count=1) for e in enclaves}


# ---------------------------------------------------------------- units


def test_open_tier_work_earns_at_the_tier_rate():
    entries = [
        {"miner_hotkey": "C", "profile_id": "ltx-2.5-fast", "status": "succeeded", "receipt": {"body": "signed"}, "billable_s": 4.0, "finished_at": 100.0, "tier": "confidential"},
        {"miner_hotkey": "O", "profile_id": "ltx-2.5-fast", "status": "succeeded", "receipt": {"body": "signed"}, "billable_s": 4.0, "finished_at": 100.0, "tier": "open"},
    ]
    weights = normalize(compute_scores(entries, {"C", "O"}, PROFILES, SwitchConfig(), 150.0, tier_rates=TierPolicy().rates()))
    assert weights["C"] == pytest.approx(1 / 1.75) and weights["O"] == pytest.approx(0.75 / 1.75)  # the default rate, 0.75
    # Without tier rates (or without tiers on entries) nothing changes.
    assert normalize(compute_scores(entries, {"C", "O"}, PROFILES, SwitchConfig(), 150.0))["O"] == pytest.approx(0.5)


def test_tier_policy_from_env_and_its_limits():
    assert TierPolicy.from_env({}) == TierPolicy(0.75, 5)
    assert TierPolicy.from_env({"KUNO_OPEN_TIER_RATE": "0.25", "KUNO_OPEN_TIER_PROBES": "10"}).rates()["open"] == 0.25
    with pytest.raises(ValueError, match="between 0 and 1"):
        TierPolicy.from_env({"KUNO_OPEN_TIER_RATE": "1.5"})


def test_admission_needs_n_passed_probes_and_an_attributable_failure_restarts_it():
    tracker = AdmissionTracker(required=3)
    for at in (1, 2):
        tracker.record("O", True, True, at)
    tracker.record("O", False, False, 3)  # the gateway's fault: no effect
    assert tracker.progress("O") == (2, 3) and not tracker.admitted("O")
    tracker.record("O", False, True, 4)
    assert tracker.progress("O") == (0, 3)
    for at in (5, 6, 7):
        tracker.record("O", True, True, at)
    assert tracker.admitted("O")
    tracker.record("O", False, True, 8)
    assert tracker.admitted("O")  # later failures are ordinary penalties
    restored = AdmissionTracker(required=3)
    restored.load(tracker.dump())
    assert restored.admitted("O") and AdmissionTracker(required=0).admitted("anyone")


def test_unadmitted_open_tier_work_earns_nothing_and_private_receipts_from_verified_open_enclaves_are_fraud():
    entries = [
        {"enclave_id": "e-open", "miner_hotkey": "O", "status": "succeeded", "receipt": {"body": "signed"}, "privacy": "standard", "job_id": "j1"},
        {"enclave_id": "e-open", "miner_hotkey": "O", "status": "succeeded", "receipt": {"body": "signed"}, "privacy": "private", "job_id": "j2"},
        {"enclave_id": "e-conf", "miner_hotkey": "C", "status": "succeeded", "receipt": {"body": "signed"}, "privacy": "private", "job_id": "j3"},
        {"enclave_id": "e-unknown", "miner_hotkey": "U", "status": "succeeded", "receipt": {"body": "signed"}, "job_id": "j4"},
    ]
    flags = apply_tiers(entries, {"e-open": "open", "e-conf": "confidential"}, AdmissionTracker(required=2))
    assert [e["tier"] for e in entries] == ["open", "open", "confidential", "confidential"]
    assert [e.get("credit") for e in entries] == [False, False, None, None]
    assert flags == {"O": ["open-tier admission: 0/2 probes passed; work does not earn yet"]}

    assert list(fraud_penalties(entries, {"e-open": "open"})) == ["O"]
    assert "private job j2" in fraud_penalties(entries, {"e-open": "open"})["O"][0]
    assert fraud_penalties(entries, {}) == {}  # a tier only the gateway's feed claims never frames a miner
    assert fraud_penalties([{**entries[1], "privacy": None}], {"e-open": "open"}) == {}


def test_open_tier_gpus_are_the_larger_of_reported_and_capacity_derived():
    needs = {p: profile.gpus_per_worker for p, profile in PROFILES.items()}
    assert open_tier_gpus({"profiles": ["ltx-2.5-fast"], "capacity": 2, "hardware": {"gpu_count": 1}}, needs) == 2
    assert open_tier_gpus({"profiles": ["ltx-2.5-fast"], "capacity": 1, "hardware": {"gpu_count": "4"}}, needs) == 4
    assert open_tier_gpus({"profiles": ["h3"], "capacity": 1, "hardware": {}}, needs) == 4
    assert open_tier_gpus({}, needs) == 1


def test_open_tier_collateral_per_gpu_is_higher_and_cannot_be_set_lower():
    gate = CollateralGate(Chain({"O": 30, "C": 30}), 7, 10 * RAO_PER_ALPHA)
    assert gate.min_per_gpu_open == 20 * RAO_PER_ALPHA
    penalties = gate.penalties({"C": 2}, 100.0, {"O": 2})
    assert "C" not in penalties
    assert penalties["O"] == ["collateral 30 alpha is below the 40 alpha required for 2 open-tier GPU(s) (20 per GPU)"]
    with pytest.raises(ValueError, match="cannot be below"):
        CollateralGate.from_env({"KUNO_MIN_COLLATERAL_PER_GPU": "10", "KUNO_MIN_COLLATERAL_PER_GPU_OPEN": "5"}, 7)
    only_open = CollateralGate.from_env({"KUNO_MIN_COLLATERAL_PER_GPU_OPEN": "25"}, None)
    assert only_open.enabled and only_open.min_per_gpu == 0 and only_open.min_per_gpu_open == 25 * RAO_PER_ALPHA
    assert CollateralGate.from_env({}, 7) is None


# ---------------------------------------------------------------- the validator


def test_the_validator_scores_by_tier_admits_open_miners_and_zeroes_fraud(tmp_path):
    confidential, opened = ConfidentialEnclave("C"), OpenEnclave("O")
    gateway = FakeGateway(enclaves=[confidential, opened], ledger=[live(confidential, "private"), live(opened, "standard")])
    validator = make_validator(gateway, state_path=tmp_path / "state.json", tier_policy=TierPolicy(open_rate=0.5, admission_probes=2))

    before = validator.score(verdicts(confidential, opened))
    assert before["O"].score == 0 and not before["O"].reasons and "0/2 probes" in before["O"].flags[0]
    assert before["C"].score == pytest.approx(1.0)

    for _ in range(2):
        validator._record(CanaryResult("ltx-2.5-fast", True, "ok", "job", opened.enclave_id, "O", True))
    admitted = normalize(validator.score(verdicts(confidential, opened)))
    assert admitted["C"] == pytest.approx(2 / 3) and admitted["O"] == pytest.approx(1 / 3)

    reloaded = make_validator(gateway, state_path=tmp_path / "state.json", tier_policy=TierPolicy(open_rate=0.5, admission_probes=2))
    assert reloaded.admission.admitted("O") and reloaded.enclave_tiers[opened.enclave_id] == "open"

    gateway.ledger.append(live(opened, "private"))
    fraud = validator.score(verdicts(confidential, opened))
    assert fraud["O"].score == 0 and any(r.startswith("fraud:") for r in fraud["O"].reasons)


def test_tiers_the_validator_never_verified_come_from_the_feed_but_never_count_as_fraud():
    opened = OpenEnclave("O")
    gateway = FakeGateway(enclaves=[opened], ledger=[live(opened, "private")])
    validator = make_validator(gateway, tier_policy=TierPolicy(admission_probes=0))
    scores = validator.score({opened.enclave_id: Verdict(True, opened.enclave_id)})  # a verdict without a tier
    assert not scores["O"].reasons and scores["O"].score > 0
    assert validator.known_tiers()[opened.enclave_id] == "open"


def test_collateral_counts_open_tier_gpus_separately():
    confidential, opened = ConfidentialEnclave("C"), OpenEnclave("O")
    gateway = FakeGateway(enclaves=[confidential, opened], ledger=[])
    gate = CollateralGate(Chain({"C": 10, "O": 30}), 7, 10 * RAO_PER_ALPHA)
    validator = make_validator(gateway, collateral=gate, tier_policy=TierPolicy(admission_probes=0))
    validator.enclaves()
    both = verdicts(confidential, opened)
    assert validator.attested_gpus(both) == {"C": 1}
    assert validator.open_tier_gpus(both) == {"O": 2}  # capacity 2 × one GPU per ltx job
    scores = validator.score(both)
    assert not scores["C"].reasons and "40 alpha required for 2 open-tier GPU(s)" in scores["O"].reasons[0]


class StandardGateway(FakeGateway):
    def __init__(self, enclave: FakeEnclave, **kwargs):
        super().__init__(enclaves=[enclave], **kwargs)
        self.enclave, self.jobs = enclave, {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/standard/videos" and request.method == "POST":
            self.requests.append(request)
            body = __import__("json").loads(request.content)
            job_id, now = "11111111-1111-4111-8111-111111111111", time.time()
            receipt = sign_receipt(self.enclave.key, ReceiptBody(
                job_id=job_id, enclave_id=self.enclave.enclave_id, profile_id=body["params"]["profile_id"], image_digest="sha256:img",
                params_digest=sha256_hex(canonical_json(body["params"])), input_digest="0" * 64, output_digest="1" * 64, output_bytes=1,
                content_digest="2" * 64, attestation_digest="3" * 64, started_at=now - 5, finished_at=now, gpu_seconds=5.0,
                video=VideoInfo(duration_s=body["params"]["duration_s"], width=1280, height=704, fps=24, frames=49, audio=True),
                miner_hotkey=self.enclave.hotkey,
            ))
            status = {
                "job_id": job_id, "status": "succeeded", "params": body["params"], "enclave_id": self.enclave.enclave_id, "price_usd": 0.1,
                "created_at": now, "updated_at": now, "receipt": receipt.model_dump(mode="json"), "privacy": "standard",
            }
            self.jobs[job_id] = status
            return httpx.Response(201, json=status)
        if path.startswith("/v1/standard/videos/") and path.endswith("/video"):
            return httpx.Response(200, content=b"mp4 bytes")
        if path.startswith("/validator/v1/standard-jobs/"):
            job = self.jobs.get(path.rsplit("/", 1)[1])
            return httpx.Response(200, json=job) if job else httpx.Response(404)
        return super().__call__(request)


def test_standard_canaries_are_the_admission_probes_of_open_tier_miners(monkeypatch):
    opened = OpenEnclave("O")
    gateway = StandardGateway(opened)
    validator = make_validator(gateway)
    validator.enclaves()
    validator.enclave_tiers[opened.enclave_id] = "open"
    seen = {}

    def checked(profile, job_id, video, receipt, duration_s, resolution):
        seen["video"] = video
        return CanaryResult(profile.id, True, "ok", job_id, receipt.body.enclave_id, "O", True)

    monkeypatch.setattr(validator, "check_canary_output", checked)
    result = validator.run_canary("ltx-2.5-fast", privacy="standard")
    assert result.ok and seen["video"] == b"mp4 bytes"
    assert validator.admission.progress("O")[0] == 1
    (record,) = validator._unaudited
    assert record.tier == "open" and record.source == "canary"


def test_the_validator_samples_standard_jobs_from_the_ledger_and_fetches_their_records():
    miner = NoisyToyMiner()
    record = miner.run("22222222-2222-4222-8222-222222222222")
    row = ledger_row(record, finished_at=time.time() - 60)
    gateway = FakeGateway(ledger=[row, {**row, "job_id": "private-job", "privacy": "private"}])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/validator/v1/standard-jobs/{record.job_id}":
            return httpx.Response(200, json=gateway_job(record))
        return gateway(request)

    validator = make_validator(handler)
    validator.enclave_tiers[miner.enclave_id] = "open"
    validator.auditor.policy.open_tier_rate = 1.0
    (audit,) = validator.standard_audit_records(time.time())
    assert (audit.job_id, audit.source, audit.tier, audit.prompt, audit.seed) == (record.job_id, "standard", "open", record.prompt, record.seed)
