"""Only paid jobs earn job pay. A job whose `billable_usd` is 0 (a canary, a refund, promo credit) earns nothing in VCU or
USD mode but still counts for every gate, and a row without the field is billable. In USD mode job pay is capped at a
multiple of billable revenue and paid at face value, and an undersubscribed pool's residual goes to capacity, not jobs."""

from __future__ import annotations

import json
import time

import pytest

from kuno_protocol.crypto import generate_signing_key, public_key_bytes
from kuno_protocol.rate_card import RateCard, placeholder_rate_card, sign_rate_card
from kuno_protocol.schemas import GenerationParams
from kuno_protocol.switch import SwitchConfig
from kuno_validator.ledger import audit_ledger, enclave_keys
from kuno_validator.open_tier import fraud_penalties
from kuno_validator.scoring import CapacityCredit, MinerScore, compute_scores, earns_job_pay, job_vcu, normalize
from kuno_validator.usd_pay import OwedWork, PayPolicy, UsdPay, capacity_owed, list_price_usd, settle, usd_owed

from test_capacity_pay import DAY, H3, HOUR, LTX, GpuEnclave, capacity_validator, check, gpu, signed, switch, verdict
from test_receipt_ledger import FakeEnclave
from test_scoring import NOW, PROFILES, job
from test_usd_pay import FakeReader, emission, oracle
from test_validator import FakeGateway

PER_TEMPO = 4320 / DAY  # the fake chain's tempo over the scoring window


def paid(row: dict, usd: float | None) -> dict:
    return {**row, "billable_usd": usd}


def entry(hotkey: str, profile_id: str, seconds: float, resolution: str | None = None, fps: int = 24, **extra) -> dict:
    """An audited, credited confidential-tier ledger row with full params."""
    profile = PROFILES[profile_id]
    resolution = resolution or next(iter(profile.limits.sizes))
    params = GenerationParams(profile_id=profile_id, mode="text_to_video", duration_s=seconds, resolution=resolution,
                              aspect_ratio=next(iter(profile.limits.sizes[resolution])), fps=fps)
    return {"job_id": f"{hotkey}-{profile_id}-{resolution}-{fps}-{seconds}-{sorted(extra.items())}", "miner_hotkey": hotkey,
            "profile_id": profile_id, "status": "succeeded", "receipt": {"body": {}}, "finished_at": 1000.0, "billable_s": seconds,
            "credit": True, "tier": "confidential", "params": params.model_dump(mode="json"), **extra}


# ---------------------------------------------------------------- the rule


def test_only_a_positive_billable_usd_or_a_row_without_one_earns():
    assert earns_job_pay({}) and earns_job_pay({"billable_usd": None})  # gateways that predate the field
    assert earns_job_pay({"billable_usd": 0.25})
    for unpaid in (0, 0.0, -1.0, float("nan"), "0.25", True):
        assert not earns_job_pay({"billable_usd": unpaid}), unpaid


def test_scoring_weighs_each_job_by_its_resolution_fps_and_duration():
    fast = PROFILES["ltx-2.5-fast"]
    assert job_vcu(fast, entry("A", fast.id, 10, "1080p", fps=50), 10) == pytest.approx(115)
    # Rows without params: the row's resolution and fps, else the lowest resolution at the default fps.
    assert job_vcu(fast, {"resolution": "1080p", "fps": 48}, 5) == pytest.approx(50)
    assert job_vcu(fast, {"resolution": "4320p"}, 5) == job_vcu(fast, {}, 5) == pytest.approx(15)
    scores = compute_scores([entry("A", fast.id, 5, "1080p"), entry("B", fast.id, 5, "720p")], {"A", "B"}, PROFILES, SwitchConfig(), 1100.0)
    assert normalize(scores) == pytest.approx({"A": 5 / 8, "B": 3 / 8})


# ---------------------------------------------------------------- VCU mode and the gates


def test_unpaid_jobs_earn_no_vcu_but_count_for_reliability_and_are_flagged():
    ledger = [paid(job("A", "ltx-2.5-fast", 5), 0.0), paid(job("B", "ltx-2.5-fast", 5), 0.25), job("C", "ltx-2.5-fast", 5)]
    scores = compute_scores(ledger, {"A", "B", "C"}, PROFILES, SwitchConfig(), NOW)
    assert scores["A"].work == {} and scores["A"].succeeded == 1 and scores["A"].unpaid_jobs == 1 and not scores["A"].reasons
    assert "1 verified job(s) earn no job pay: no customer paid for them (billable_usd 0)" in scores["A"].flags
    assert normalize(scores) == {"B": pytest.approx(0.5), "C": pytest.approx(0.5)}  # C's row has no billable_usd: billable
    many = [paid(job("A", "ltx-2.5-fast"), 0.0) for _ in range(19)]
    many.append(paid(job("A", "ltx-2.5-fast", status="failed", error_code="timeout"), 0.0))
    assert any("success rate" in r for r in compute_scores(many, {"A"}, PROFILES, SwitchConfig(), NOW)["A"].reasons)
    penalized = compute_scores(ledger, {"A", "B", "C"}, PROFILES, SwitchConfig(), NOW, penalties={"A": ["failed canary ltx-2.5-fast (wrong size)"]})
    assert penalized["A"].reasons and penalized["A"].score == 0


def test_a_canary_still_satisfies_capacity_pays_served_job_requirement():
    ledger = [paid(job("A", "ltx-2.5-fast", 5), 0.0), paid(job("B", "ltx-2.5-fast", 5), 0.5)]
    credit = CapacityCredit({"A": {LTX: 2 * DAY}})
    scores = compute_scores(ledger, {"A", "B"}, PROFILES, switch(), NOW, capacity=credit)
    assert LTX in scores["A"].served and scores["A"].capacity == {LTX: 2 * DAY} and scores["A"].work == {}
    assert credit.families[LTX].blend == 0.125  # capacity_share 0.25 × 2 of 4 target GPUs
    assert scores["A"].score == pytest.approx(0.125) and scores["B"].score == pytest.approx(0.875)
    # With only a canary in the family nobody earned job pay there, so the family's whole split goes to capacity.
    alone = CapacityCredit({"A": {LTX: 2 * DAY}})
    assert normalize(compute_scores(ledger[:1], {"A"}, PROFILES, switch(), NOW, capacity=alone)) == {"A": pytest.approx(1.0)}
    assert alone.families[LTX].blend == 1.0


def test_unpaid_jobs_are_still_checked_for_replays_and_fraud():
    a, b = FakeEnclave("A"), FakeEnclave("B")
    rows = [paid(a.entry(content=b"same bytes", age_s=120), 0.0), paid(b.entry(content=b"same bytes"), 0.0)]
    audit = audit_ledger(rows, enclave_keys([a.public(), b.public()]), PROFILES)
    assert any("replayed output" in reason for reason in audit.penalties["B"])
    private = {"enclave_id": "e-open", "miner_hotkey": "O", "status": "succeeded", "receipt": {"body": "signed"}, "privacy": "private",
               "job_id": "j1", "billable_usd": 0.0}
    assert list(fraud_penalties([private], {"e-open": "open"})) == ["O"]


# ---------------------------------------------------------------- USD mode: what is owed and what customers paid


def test_usd_mode_owes_only_paid_jobs_by_vcu_and_counts_only_billable_revenue():
    card = placeholder_rate_card(issued_at=1)
    entries = [
        entry("A", "h3", 10, billable_usd=2.0),
        entry("A", "ltx-2.5-fast", 5, "1080p", fps=50, billable_usd=0.4),
        entry("B", "h3", 10, billable_usd=0.0),  # a validator canary: no pay, and not revenue
        entry("C", "ltx-2.5-fast", 5, privacy="standard"),  # a row from an older gateway: billable, revenue at list price
    ]
    work = usd_owed(entries, {"A", "B", "C"}, card, PROFILES, SwitchConfig(), now=1100.0, window_s=500.0)
    h3_vcu, fast_1080p_50fps_vcu, fast_720p_vcu = 100 * (1 + 0.093 * 5) * 10, 5 * 2 * 5, 3 * 5
    assert work.owed_usd == pytest.approx({"A": (h3_vcu + fast_1080p_50fps_vcu) * 0.0019, "C": fast_720p_vcu * 0.0019})
    assert work.unpaid_seconds == {"B": 10.0}
    standard = PROFILES["ltx-2.5-fast"].price_usd(GenerationParams.model_validate(entries[3]["params"]), privacy="standard")
    assert work.revenue_billable_usd == pytest.approx(2.4) and work.revenue_list_price_usd == pytest.approx(standard)
    assert work.revenue_usd == pytest.approx(2.4 + standard)
    assert (work.revenue_jobs, work.unbilled_jobs, work.revenue_unknown_jobs) == (3, 1, 0)


def test_a_list_price_in_a_mode_the_profile_isnt_offered_in_is_unknown_not_a_crash():
    private_only = next(profile for profile in PROFILES.values() if not profile.offers("standard"))
    row = entry("C", private_only.id, 5, privacy="standard")
    assert list_price_usd(row, private_only) is None
    assert list_price_usd({**row, "privacy": None}, private_only) == private_only.price_usd(GenerationParams.model_validate(row["params"]))
    work = usd_owed([row], {"C"}, placeholder_rate_card(), PROFILES, SwitchConfig(), now=1100.0, window_s=500.0)
    assert work.revenue_unknown_jobs == 1 and work.owed_usd["C"] > 0
    resolution = next(iter(private_only.limits.sizes))
    bare = {key: value for key, value in row.items() if key != "params"} | {"resolution": resolution, "duration_s": 5}
    assert list_price_usd(bare, private_only) is None
    assert list_price_usd({**bare, "privacy": "private"}, private_only) == pytest.approx(private_only.pricing.usd_per_second[resolution] * 5)


# ---------------------------------------------------------------- USD mode: settling


def test_job_pay_is_capped_at_the_revenue_multiple_pro_rata():
    card = RateCard(usd_per_second={}, placeholder=False)
    work = OwedWork(owed_usd={"A": 30.0, "B": 10.0}, revenue_usd=20.0, revenue_billable_usd=20.0)
    weights, report = settle(work, emission(), oracle().quote(), card, now=5.0, window_s=DAY)
    assert report.job_uncapped_usd_per_tempo == pytest.approx(40 * PER_TEMPO) and report.revenue_usd_per_tempo == pytest.approx(20 * PER_TEMPO)
    assert report.job_cap_usd_per_tempo == pytest.approx(20 * PER_TEMPO) and report.job_scale == pytest.approx(0.5)
    assert report.job_usd_per_tempo == pytest.approx(20 * PER_TEMPO) and report.owed_usd_window == pytest.approx(20.0)
    assert report.miners["A"]["job_usd_owed"] == 30.0 and report.miners["A"]["job_usd_per_tempo"] == pytest.approx(15 * PER_TEMPO)
    assert report.miners["A"]["usd_owed"] == pytest.approx(15.0)
    # Nobody is owed capacity, so the residual renormalizes job owed up as before: the shares stand, nothing is burned.
    assert report.residual_to == "jobs" and weights == pytest.approx({"A": 0.75, "B": 0.25})
    _, doubled = settle(work, emission(), oracle().quote(), card, now=5.0, window_s=DAY, job_revenue_multiple=2.0)
    assert doubled.job_scale == 1.0 and doubled.job_usd_per_tempo == pytest.approx(40 * PER_TEMPO)
    nothing, report = settle(work, emission(), oracle().quote(), card, now=5.0, window_s=DAY, job_revenue_multiple=0.0)
    assert nothing == {} and report.regime == "no_work" and report.miners["A"]["weight"] == 0
    # No revenue at all (every job a canary, say): no job pay.
    assert settle(OwedWork(owed_usd={"A": 30.0}), emission(), oracle().quote(), card, now=5.0, window_s=DAY)[0] == {}


def test_a_self_bought_job_is_paid_at_face_value_while_capacity_takes_the_residual():
    card = RateCard(usd_per_second={}, gpu_hour_usd={LTX: 0.80}, placeholder=False)
    # S bought a $10 job that landed on itself; R kept a GPU verified and ready all day and served only canaries.
    scores = {"S": MinerScore("S"), "R": MinerScore("R", capacity={LTX: DAY})}
    work = capacity_owed(OwedWork(owed_usd={"S": 10.0}, revenue_usd=10.0, revenue_billable_usd=10.0), scores, card)
    weights, report = settle(work, emission(), oracle().quote(), card, now=5.0, window_s=DAY, capacity_share=0.25)
    pool, job, capacity = report.pool_usd_per_tempo, 10.0 * PER_TEMPO, 24 * 0.80 * PER_TEMPO
    assert report.regime == "undersubscribed" and report.residual_to == "capacity" and report.job_scale == report.capacity_scale == 1.0
    assert report.residual_to_capacity_usd_per_tempo == pytest.approx(pool - job - capacity)
    assert weights["S"] == pytest.approx(job / pool)  # $0.50 of a $442.80 pool, not scaled up to a third of it
    assert weights["R"] == pytest.approx(1 - job / pool) and sum(weights.values()) == pytest.approx(1.0)
    assert report.miners["R"]["residual_usd_per_tempo"] == pytest.approx(pool - job - capacity)
    assert report.miners["S"]["residual_usd_per_tempo"] == 0 and job / (job + capacity) > 0.3


def test_the_residual_goes_to_capacity_pro_rata_after_the_capacity_share_cap():
    card = RateCard(usd_per_second={}, gpu_hour_usd={LTX: 0.80, H3: 1.50}, placeholder=False)
    scores = {"A": MinerScore("A", capacity={LTX: 3 * DAY}), "B": MinerScore("B", capacity={H3: DAY}), "J": MinerScore("J")}
    work = capacity_owed(OwedWork(owed_usd={"J": 20.0, "A": 4.0}, revenue_usd=24.0, revenue_billable_usd=24.0), scores, card)
    a_cap, b_cap, jobs = 72 * 0.80 * PER_TEMPO, 24 * 1.50 * PER_TEMPO, 24.0 * PER_TEMPO
    weights, report = settle(work, emission(), oracle().quote(), card, now=5.0, window_s=DAY, capacity_share=0.25)
    pool = report.pool_usd_per_tempo
    residual = pool - jobs - a_cap - b_cap
    assert report.residual_to_capacity_usd_per_tempo == pytest.approx(residual)
    assert weights["J"] == pytest.approx(20.0 * PER_TEMPO / pool)
    assert weights["A"] == pytest.approx((4.0 * PER_TEMPO + a_cap + residual * a_cap / (a_cap + b_cap)) / pool)
    assert weights["B"] == pytest.approx((b_cap + residual * b_cap / (a_cap + b_cap)) / pool)
    assert sum(weights.values()) == pytest.approx(1.0)

    # A pool worth $4.43 per tempo: capacity owed is first held to a quarter of it, then the residual still lifts it above.
    weights, report = settle(work, emission(tao_per_alpha=0.0001), oracle().quote(), card, now=5.0, window_s=DAY, capacity_share=0.25)
    pool = report.pool_usd_per_tempo
    limit = 0.25 * pool
    scale = limit / (a_cap + b_cap)
    residual = pool - jobs - limit
    assert report.capacity_scale == pytest.approx(scale) and report.capacity_usd_per_tempo == pytest.approx(limit)
    assert report.regime == "undersubscribed" and report.residual_to_capacity_usd_per_tempo == pytest.approx(residual)
    assert weights["J"] == pytest.approx(20.0 * PER_TEMPO / pool)
    assert weights["B"] == pytest.approx((b_cap * scale + residual * b_cap / (a_cap + b_cap)) / pool)
    assert weights["A"] + weights["B"] - 4.0 * PER_TEMPO / pool > 0.25  # capacity's part of the pool, above capacity_share


def test_without_capacity_owed_the_residual_renormalizes_job_owed_as_before():
    card = RateCard(usd_per_second={}, gpu_hour_usd={LTX: 0.80}, placeholder=False)
    work = capacity_owed(OwedWork(owed_usd={"A": 3.0, "B": 1.0}, revenue_usd=10.0), {"C": MinerScore("C", capacity={LTX: DAY})}, card)
    weights, report = settle(work, emission(), oracle().quote(), card, now=5.0, window_s=DAY, capacity_share=0.0)
    assert report.capacity_usd_per_tempo == 0 and report.residual_to == "jobs" and report.residual_to_capacity_usd_per_tempo == 0
    assert weights == pytest.approx({"A": 0.75, "B": 0.25}) and report.miners["C"]["weight"] == 0


def test_an_oversubscribed_pool_renormalizes_job_and_capacity_owed_down_alike():
    card = RateCard(usd_per_second={}, gpu_hour_usd={LTX: 0.80}, placeholder=False)
    work = capacity_owed(OwedWork(owed_usd={"J": 30.0}, revenue_usd=30.0), {"R": MinerScore("R", capacity={LTX: DAY})}, card)
    weights, report = settle(work, emission(tao_per_alpha=0.00001), oracle().quote(), card, now=5.0, window_s=DAY, capacity_share=1.0)
    job, capacity = 30.0 * PER_TEMPO, report.pool_usd_per_tempo  # capacity owed held to the whole (tiny) pool
    assert report.regime == "oversubscribed" and report.residual_to == "none" and report.residual_to_capacity_usd_per_tempo == 0
    assert report.job_scale == 1.0 and report.capacity_usd_per_tempo == pytest.approx(capacity)
    assert weights == pytest.approx({"J": job / (job + capacity), "R": capacity / (job + capacity)})


def test_the_job_revenue_multiple_comes_from_the_environment():
    assert PayPolicy.from_env({}).job_revenue_multiple == 1.0
    assert PayPolicy.from_env({"KUNO_JOB_PAY_REVENUE_MULTIPLE": "1.5"}).job_revenue_multiple == 1.5
    for bad in ("-1", "nan", "inf"):
        with pytest.raises(ValueError, match="KUNO_JOB_PAY_REVENUE_MULTIPLE"):
            PayPolicy.from_env({"KUNO_JOB_PAY_REVENUE_MULTIPLE": bad})


# ---------------------------------------------------------------- a whole round


def test_a_usd_round_pays_paid_jobs_and_sends_the_residual_to_a_miner_whose_capacity_only_a_canary_proved(tmp_path):
    owner = generate_signing_key()
    a, b = GpuEnclave("A"), GpuEnclave("B")
    (tmp_path / "card.json").write_text(sign_rate_card(owner, placeholder_rate_card(issued_at=1000)).model_dump_json())
    pay = UsdPay(PayPolicy(mode="usd", rate_card_path=tmp_path / "card.json"), 7, FakeReader(), oracle(), public_key_bytes(owner),
                 report_path=tmp_path / "pay.jsonl")
    ledger = [paid(a.entry(), 0.5), paid(a.entry(), 0.5), paid(b.entry(), 0.0)]  # B's only job is a validator canary
    validator = capacity_validator(FakeGateway(enclaves=[a, b], ledger=ledger, switch=signed()), state_path=tmp_path / "state.json", pay=pay)
    validator.capacity.record(check("B", gpu("b0").token), time.time() - 2 * HOUR, DAY)
    verdicts = {a.enclave_id: verdict(a), b.enclave_id: verdict(b, gpu("b0"))}
    validator.check_enclaves = lambda: verdicts
    weights = validator.step()
    [line] = (tmp_path / "pay.jsonl").read_text().splitlines()
    report = json.loads(line)
    job = 2 * 3 * 4 * 0.0019  # two 4 s ltx-2.5-fast 720p jobs at 3 VCU per second
    pool, job_tempo = report["pool_usd_per_tempo"], job * PER_TEMPO
    assert report["miners"]["A"]["job_usd_owed"] == pytest.approx(job) and report["miners"]["B"]["job_usd_owed"] == 0
    assert report["unpaid_seconds"] == {"B": 4.0} and report["unbilled_jobs"] == 1 and report["revenue_jobs"] == 2
    assert report["revenue_billable_usd_window"] == pytest.approx(1.0) and report["revenue_list_price_usd_window"] == 0
    assert report["job_cap_usd_per_tempo"] == pytest.approx(PER_TEMPO) and report["job_scale"] == 1.0
    assert report["miners"]["B"]["capacity_gpu_hours"] == pytest.approx(2.0, abs=0.05)
    assert report["residual_to"] == "capacity"
    assert report["residual_to_capacity_usd_per_tempo"] == pytest.approx(pool - job_tempo - report["capacity_usd_per_tempo"])
    assert weights == pytest.approx({"A": job_tempo / pool, "B": 1 - job_tempo / pool})
    assert any("earn no job pay" in flag for flag in validator.score(verdicts)["B"].flags)
