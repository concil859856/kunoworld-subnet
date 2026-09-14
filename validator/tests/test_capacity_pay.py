"""Capacity pay: verified GPU-time of ready confidential-tier GPUs, earned only after an hour of continuous verification,
gated like job work, capped by the owner-signed switch's targets, blended into VCU scores or priced in USD, and kept
across restarts."""

from __future__ import annotations

import json
import logging
import time

import httpx
import pytest

from kuno_protocol.attestation import GoldenManifest, Verdict
from kuno_protocol.crypto import generate_signing_key, public_key_bytes
from kuno_protocol.hardware import HardwareIdentity, hardware_token
from kuno_protocol.rate_card import RateCard, sign_rate_card
from kuno_protocol.switch import SwitchConfig, sign_switch
from kuno_validator.capacity import CapacityTracker
from kuno_validator.scoring import CapacityCredit, MinerScore, compute_scores, normalize
from kuno_validator.usd_pay import OwedWork, PayPolicy, UsdPay, capacity_owed, settle
from kuno_validator.validator import Validator

from test_receipt_ledger import FakeEnclave
from test_scoring import NOW, PROFILES, job
from test_usd_pay import FakeReader, emission, oracle
from test_validator import API_KEY, FakeGateway

LTX, H3 = "ltx-2.5", "minimax-h3"
HOUR, DAY = 3600.0, 86400.0


def families(profile_ids: list[str]) -> list[str]:
    return sorted({PROFILES[p].family for p in profile_ids if p in PROFILES})


def check(hotkey: str = "A", token: str = "g0", profiles=("ltx-2.5-fast",)) -> list[tuple[str, str, list[str]]]:
    return [(hotkey, token, list(profiles))]


def switch(**changes) -> SwitchConfig:
    return SwitchConfig(**{"capacity_share": 0.25, "capacity_targets": {LTX: 4, H3: 8}, **changes})


# ---------------------------------------------------------------- verified runs


def test_a_run_earns_nothing_before_an_hour_of_continuous_verification_and_all_of_it_after():
    tracker = CapacityTracker(max_gap_s=1500)
    for at in (0, 1200, 2400, 3500):
        tracker.record(check(), NOW + at, DAY)
    assert tracker.gpu_seconds(NOW + 3500, DAY, 3600, families) == {}  # 3500 s verified so far
    tracker.record(check(), NOW + 4700, DAY)
    assert tracker.gpu_seconds(NOW + 4700, DAY, 3600, families) == {"A": {LTX: 4700.0}}  # the first hour included


def test_a_gap_longer_than_the_max_breaks_the_run():
    tracker = CapacityTracker(max_gap_s=1500)
    for at in (0, 1200, 2400, 4000, 5200, 6400):  # 1600 s between 2400 and 4000: a missed round
        tracker.record(check(), NOW + at, DAY)
    assert tracker.gpu_seconds(NOW + 6400, DAY, 3600, families) == {}  # two runs of 2400 s, neither an hour
    tracker.record(check(), NOW + 7600, DAY)
    assert tracker.gpu_seconds(NOW + 7600, DAY, 3600, families) == {"A": {LTX: 3600.0}}  # only the second run
    assert len(tracker.runs["A"]["g0"]) == 2


def test_credit_is_clipped_to_the_window_and_split_over_the_families_the_enclave_serves():
    tracker = CapacityTracker(max_gap_s=1500)
    for at in range(0, 10801, 1200):  # three hours of checks on an enclave serving LTX and H3
        tracker.record(check(profiles=("ltx-2.5-fast", "h3")), NOW + at, window_s=1800)
    # Pruning drops stretches before the window but keeps when the run began, so a run older than the window still counts.
    assert tracker.runs["A"]["g0"][0][0] == NOW and all(span[1] > NOW + 9000 for span in tracker.runs["A"]["g0"][0][2])
    assert tracker.gpu_seconds(NOW + 10800, 1800, 3600, families) == {"A": {LTX: 900.0, H3: 900.0}}
    # The switch decides which families count: with H3 off the whole stretch is LTX's.
    assert tracker.gpu_seconds(NOW + 10800, 1800, 3600, lambda ids: [LTX]) == {"A": {LTX: 1800.0}}


def test_a_gpu_two_enclaves_of_one_hotkey_show_is_one_check():
    tracker = CapacityTracker(max_gap_s=1500)
    for at in (0, 1200, 2400, 3600):
        tracker.record(check(profiles=("ltx-2.5-fast",)) + check(profiles=("ltx-2.5-pro",)), NOW + at, DAY)
    assert tracker.gpu_seconds(NOW + 3600, DAY, 3600, families) == {"A": {LTX: 3600.0}}


def test_the_max_gap_follows_the_round_interval_unless_set():
    assert CapacityTracker.from_env({}, interval_s=4320).max_gap_s == 8640
    assert CapacityTracker.from_env({"KUNO_CAPACITY_MAX_GAP_S": "900"}, interval_s=4320).max_gap_s == 900
    with pytest.raises(ValueError, match="KUNO_CAPACITY_MAX_GAP_S"):
        CapacityTracker.from_env({"KUNO_CAPACITY_MAX_GAP_S": "0"})


# ---------------------------------------------------------------- scoring


def test_with_capacity_share_zero_scores_are_exactly_todays():
    ledger = [job("A", "ltx-2.5-fast", 5), job("B", "ltx-2.5-fast", 15), job("C", "h3", 5)]
    gpu_time = {"A": {LTX: 20 * HOUR}, "C": {H3: 30 * HOUR}}
    today = compute_scores(ledger, {"A", "B", "C"}, PROFILES, SwitchConfig(), NOW)
    off = SwitchConfig(capacity_targets={LTX: 4, H3: 8})  # targets set, but no share
    scores = compute_scores(ledger, {"A", "B", "C"}, PROFILES, off, NOW, capacity=CapacityCredit(gpu_time))
    assert {h: m.score for h, m in scores.items()} == {h: m.score for h, m in today.items()}
    assert scores == today  # reasons, flags and capacity untouched too
    assert normalize(scores) == normalize(today)
    # Paying for capacity but with nobody credited changes nothing either.
    nobody = compute_scores(ledger, {"A", "B", "C"}, PROFILES, switch(), NOW, capacity=CapacityCredit({}))
    assert {h: m.score for h, m in nobody.items()} == {h: m.score for h, m in today.items()}


def test_capacity_is_blended_in_at_the_share_scaled_by_utilization_of_the_target():
    ledger = [job("A", "ltx-2.5-fast", 5), job("B", "ltx-2.5-fast", 15)]
    credit = CapacityCredit({"A": {LTX: 2 * DAY}})  # two of A's GPUs verified all window: half the target of 4
    scores = compute_scores(ledger, {"A", "B"}, PROFILES, switch(), NOW, capacity=credit)
    ltx = credit.families[LTX]
    assert (ltx.average_gpus, ltx.utilization, ltx.scale, ltx.blend) == (2.0, 0.5, 1.0, 0.125)
    assert scores["A"].score == pytest.approx(0.875 * 0.25 + 0.125)
    assert scores["B"].score == pytest.approx(0.875 * 0.75)
    assert scores["A"].capacity == {LTX: 2 * DAY} and "capacity ltx-2.5: 48.00 GPU-hours credited" in scores["A"].flags


def test_capacity_beyond_the_target_is_scaled_down_and_dilutes_instead_of_adding_emission():
    ledger = [job(h, "ltx-2.5-fast", 5) for h in "ABC"]
    credit = CapacityCredit({"A": {LTX: 6 * DAY}, "B": {LTX: 2 * DAY}})  # 8 GPUs on average for a target of 4
    scores = compute_scores(ledger, set("ABC"), PROFILES, switch(), NOW, capacity=credit)
    assert credit.families[LTX].scale == 0.5 and credit.families[LTX].blend == 0.25
    assert scores["A"].capacity == {LTX: 3 * DAY} and scores["B"].capacity == {LTX: DAY}
    assert any("scaled by 0.500 (the window averaged 8.00 GPUs for a target of 4)" in f for f in scores["A"].flags)
    job_part = 0.75 / 3
    assert {h: scores[h].score - job_part for h in "ABC"} == pytest.approx({"A": 0.25 * 0.75, "B": 0.25 * 0.25, "C": 0.0})

    # C brings 8 more GPUs: everyone's credit shrinks, and capacity still gets a quarter of the family, no more.
    more = CapacityCredit({"A": {LTX: 6 * DAY}, "B": {LTX: 2 * DAY}, "C": {LTX: 8 * DAY}})
    diluted = compute_scores(ledger, set("ABC"), PROFILES, switch(), NOW, capacity=more)
    assert more.families[LTX].scale == 0.25
    assert {h: diluted[h].score - job_part for h in "ABC"} == pytest.approx({"A": 0.25 * 6 / 16, "B": 0.25 * 2 / 16, "C": 0.25 * 8 / 16})
    assert sum(m.score for m in diluted.values()) == pytest.approx(1.0)


def test_a_family_the_switch_disables_or_gives_no_target_earns_no_capacity():
    ledger = [job("A", "ltx-2.5-fast", 5), job("B", "h3", 5)]
    gpu_time = {"A": {LTX: DAY}, "B": {H3: 8 * DAY}}
    h3_only = compute_scores(ledger, {"A", "B"}, PROFILES, switch(mode="h3"), NOW, capacity=CapacityCredit(gpu_time))
    assert h3_only["A"].capacity == {} and "capacity ltx-2.5: 24.00 verified GPU-hours earn nothing (the family is switched off)" in h3_only["A"].flags
    assert normalize(h3_only) == {"B": pytest.approx(1.0)}

    no_target = CapacityCredit(gpu_time)
    scores = compute_scores(ledger, {"A", "B"}, PROFILES, switch(capacity_targets={H3: 8}), NOW, capacity=no_target)
    assert scores["A"].capacity == {} and any("sets no capacity target" in f for f in scores["A"].flags)
    assert set(no_target.families) == {H3}
    assert scores["A"].score == pytest.approx(0.4)  # LTX's split, all of it by VCU


def test_capacity_needs_every_scoring_gate_and_a_succeeded_confidential_job_of_the_family():
    ledger = [job("A", "h3", 5), dict(job("B", "ltx-2.5-fast", 5), tier="open"), job("C", "ltx-2.5-fast", 5), job("E", "ltx-2.5-fast", 5)]
    gpu_time = {hotkey: {LTX: DAY} for hotkey in "ABCDE"}
    penalties = {"C": ["failed canary ltx-2.5-fast (wrong size)"]}
    credit = CapacityCredit(gpu_time)
    scores = compute_scores(ledger, set("ABCDE"), PROFILES, switch(), NOW, penalties=penalties, capacity=credit)
    assert scores["A"].capacity == {} and any("no succeeded confidential-tier job" in f for f in scores["A"].flags)  # served H3 only
    assert scores["B"].capacity == {}  # an open-tier job doesn't show the confidential GPUs can serve
    assert scores["C"].capacity == {} and scores["C"].score == 0  # penalized: its reasons say why
    assert scores["D"].capacity == {} and scores["D"].score == 0  # up all day, never served a job
    assert scores["E"].capacity == {LTX: DAY}
    assert credit.families[LTX].gpu_seconds == DAY


def test_when_nobody_did_verified_work_in_a_family_its_whole_split_goes_to_capacity():
    ledger = [dict(job("A", "ltx-2.5-fast", 5), tier="confidential"), dict(job("B", "ltx-2.5-fast", 5), tier="confidential")]
    worthless = {"confidential": 0.0}  # verified jobs worth no VCU: the family has no job part
    credit = CapacityCredit({"A": {LTX: HOUR}})
    scores = compute_scores(ledger, {"A", "B"}, PROFILES, switch(), NOW, tier_rates=worthless, capacity=credit)
    assert credit.families[LTX].blend == 1.0
    assert normalize(scores) == {"A": pytest.approx(1.0)}


# ---------------------------------------------------------------- the validator's rounds


class GpuEnclave(FakeEnclave):
    def public(self) -> dict:
        return {**super().public(), "tee": "tdx", "tier": "confidential", "profiles": ["ltx-2.5-fast"], "capacity": 1}


def gpu(name: str) -> HardwareIdentity:
    return HardwareIdentity("gpu", hardware_token("gpu", f"test:{name}"), "mock")


def verdict(enclave: FakeEnclave, *gpus: HardwareIdentity, tier: str = "confidential") -> Verdict:
    return Verdict(True, enclave.enclave_id, hardware=list(gpus), gpu_count=len(gpus), tier=tier)


def signed(**changes):
    return sign_switch(generate_signing_key(), switch(**changes))


def capacity_validator(gateway: FakeGateway, state_path=None, pay=None) -> Validator:
    validator = Validator(
        "http://gateway.test", API_KEY, GoldenManifest(), None, transport=httpx.MockTransport(gateway), state_path=state_path, pay=pay,
    )
    validator.enclaves()
    return validator


def test_a_round_blends_ready_gpu_time_into_the_weights_and_logs_it(caplog):
    a, b = GpuEnclave("A"), GpuEnclave("B")
    validator = capacity_validator(FakeGateway(enclaves=[a, b], ledger=[a.entry(), b.entry()], switch=signed(capacity_targets={LTX: 1})))
    validator.capacity.record(check("A", gpu("a0").token), time.time() - 2 * HOUR, DAY)  # verified two hours ago too
    validator.check_enclaves = lambda: {a.enclave_id: verdict(a, gpu("a0")), b.enclave_id: verdict(b)}
    with caplog.at_level(logging.INFO, logger="kuno.validator"):
        weights = validator.step()
    ltx = validator.last_capacity.families[LTX]
    assert ltx.gpu_seconds == pytest.approx(2 * HOUR, abs=60) and ltx.blend == pytest.approx(0.25 * ltx.average_gpus)
    assert weights == pytest.approx({"A": (1 - ltx.blend) * 0.5 + ltx.blend, "B": (1 - ltx.blend) * 0.5})
    assert "capacity ltx-2.5: 0.08 verified GPUs on average" in caplog.text and "GPU-hours credited" in caplog.text


def test_open_tier_gpus_and_gpus_without_an_identity_are_never_checked():
    a, o = GpuEnclave("A"), GpuEnclave("O")
    validator = capacity_validator(FakeGateway(enclaves=[a, o], ledger=[a.entry(), o.entry()], switch=signed()))
    for _ in range(2):
        scores = validator.score({
            a.enclave_id: Verdict(True, a.enclave_id, gpu_count=2, tier="confidential"),  # counted, never identified
            o.enclave_id: verdict(o, gpu("o0"), tier="open"),  # open-tier evidence proves no GPU, even one it names
        })
    assert validator.capacity.runs == {}
    assert scores["A"].capacity == {} and scores["O"].capacity == {}


def test_a_gpu_shown_under_two_hotkeys_is_paid_once_to_the_hotkey_dedupe_keeps():
    a, b = GpuEnclave("A"), GpuEnclave("B")
    validator = capacity_validator(FakeGateway(enclaves=[a, b], ledger=[a.entry(), b.entry()], switch=signed()))
    token, now = gpu("g0").token, time.time()
    for hours_ago in (3, 2, 1):  # A showed the GPU first; B too, for long enough to qualify on its own
        hotkeys = "AB" if hours_ago < 3 else "A"
        validator.capacity.record([c for h in hotkeys for c in check(h, token)], now - hours_ago * HOUR, DAY)
        for hotkey in hotkeys:
            seen = validator.hardware_sightings.setdefault(token, {"kind": "gpu", "hotkeys": {}})["hotkeys"]
            seen[hotkey] = [seen.get(hotkey, [now - hours_ago * HOUR])[0], now - hours_ago * HOUR]
    scores = validator.score({a.enclave_id: verdict(a, gpu("g0")), b.enclave_id: verdict(b, gpu("g0"))})
    assert scores["A"].capacity == {LTX: pytest.approx(3 * HOUR, abs=60)} and not scores["A"].reasons
    assert scores["B"].capacity == {} and scores["B"].score == 0
    assert any("first attested by A" in reason for reason in scores["B"].reasons)
    assert validator.last_capacity.families[LTX].gpu_seconds == pytest.approx(3 * HOUR, abs=60)  # A's hours, not A's and B's


def test_verified_runs_survive_a_restart(tmp_path):
    a = GpuEnclave("A")
    gateway = FakeGateway(enclaves=[a], ledger=[a.entry()], switch=signed())
    state, token, started = tmp_path / "state.json", gpu("a0").token, time.time() - 2 * HOUR
    first = capacity_validator(gateway, state_path=state)
    first.capacity.record(check("A", token), started, DAY)
    assert first.score({a.enclave_id: verdict(a, gpu("a0"))})["A"].capacity == {LTX: pytest.approx(2 * HOUR, abs=60)}
    [run] = json.loads(state.read_text())["capacity"]["runs"]["A"][token]
    assert run[0] == started and run[2][0][2] == ["ltx-2.5-fast"]

    restarted = capacity_validator(gateway, state_path=state)
    assert restarted.capacity.runs == first.capacity.runs
    scores = restarted.score({a.enclave_id: verdict(a, gpu("a0"))})
    assert len(restarted.capacity.runs["A"][token]) == 1  # the restart didn't break the run
    assert scores["A"].capacity[LTX] >= 2 * HOUR


# ---------------------------------------------------------------- USD pay


def test_usd_capacity_pay_is_priced_per_gpu_hour_and_capped_at_the_share_of_the_pool():
    card = RateCard(usd_per_second={"ltx-2.5-fast": {"confidential": 0.05}}, gpu_hour_usd={LTX: 2.0}, placeholder=False)
    scores = {
        "A": MinerScore("A", capacity={LTX: DAY}),
        "B": MinerScore("B", capacity={LTX: DAY, H3: 10 * HOUR}),
        "Z": MinerScore("Z", capacity={LTX: DAY}, reasons=["failed canary"]),
    }
    work = capacity_owed(OwedWork(owed_usd={"A": 0.6}), scores, card)
    assert work.capacity_usd == pytest.approx({"A": 48.0, "B": 48.0}) and work.capacity_unpriced == {H3: 10.0}
    per_tempo = 4320 / DAY
    job = 0.6 * per_tempo

    # $48 a day is $2.40 per tempo each, far below a quarter of the $442.80 pool: nothing is capped.
    weights, report = settle(work, emission(), oracle().quote(), card, now=5.0, window_s=DAY, capacity_share=0.25)
    assert report.capacity_scale == 1.0 and report.capacity_usd_per_tempo == pytest.approx(4.8)
    assert weights == pytest.approx({"A": (job + 2.4) / (job + 4.8), "B": 2.4 / (job + 4.8)})

    # A pool worth $4.43 per tempo: capacity owed is held to a quarter of it, scaled down alike; job owed is not.
    weights, report = settle(work, emission(tao_per_alpha=0.0001), oracle().quote(), card, now=5.0, window_s=DAY, capacity_share=0.25)
    limit = 0.25 * report.pool_usd_per_tempo
    assert report.capacity_limit_usd_per_tempo == pytest.approx(limit) and report.capacity_scale == pytest.approx(limit / 4.8)
    assert report.capacity_usd_per_tempo == pytest.approx(limit) and report.capacity_usd_window == pytest.approx(limit / per_tempo)
    assert weights == pytest.approx({"A": (job + limit / 2) / (job + limit), "B": (limit / 2) / (job + limit)})
    line = json.loads(report.to_json())
    assert line["miners"]["B"]["capacity_gpu_hours"] == 24.0 and line["miners"]["A"]["job_usd_owed"] == 0.6
    assert line["capacity_gpu_hours"] == {LTX: 48.0} and line["capacity_unpriced"] == {H3: 10.0}

    # A switch that doesn't pay for capacity owes nothing for it, whatever was credited.
    weights, report = settle(work, emission(), oracle().quote(), card, now=5.0, window_s=DAY)
    assert weights == {"A": pytest.approx(1.0)} and report.capacity_usd_per_tempo == 0


def test_a_usd_round_adds_capacity_owed_and_reports_it(tmp_path):
    owner = generate_signing_key()
    a, b = GpuEnclave("A"), GpuEnclave("B")
    card = RateCard(issued_at=1000, usd_per_second={"ltx-2.5-fast": {"confidential": 0.05}}, gpu_hour_usd={LTX: 2.0}, placeholder=False)
    (tmp_path / "card.json").write_text(sign_rate_card(owner, card).model_dump_json())
    pay = UsdPay(PayPolicy(mode="usd", rate_card_path=tmp_path / "card.json"), 7, FakeReader(), oracle(), public_key_bytes(owner),
                 report_path=tmp_path / "pay.jsonl")
    gateway = FakeGateway(enclaves=[a, b], ledger=[a.entry(), b.entry()], switch=signed())
    validator = capacity_validator(gateway, state_path=tmp_path / "state.json", pay=pay)
    validator.capacity.record(check("A", gpu("a0").token), time.time() - 2 * HOUR, DAY)
    validator.check_enclaves = lambda: {a.enclave_id: verdict(a, gpu("a0")), b.enclave_id: verdict(b)}
    weights = validator.step()
    [line] = (tmp_path / "pay.jsonl").read_text().splitlines()
    report = json.loads(line)
    hours = report["capacity_gpu_hours"][LTX]
    assert hours == pytest.approx(2.0, abs=0.05) and report["capacity_scale"] == 1.0 and report["capacity_share"] == 0.25
    assert report["capacity_families"][LTX]["target"] == 4
    assert report["miners"]["A"]["capacity_usd_owed"] == pytest.approx(2.0 * hours)
    assert report["miners"]["A"]["job_usd_owed"] == pytest.approx(4 * 0.05) and report["miners"]["B"]["capacity_usd_owed"] == 0
    assert weights == pytest.approx({"A": (0.2 + 2.0 * hours) / (0.4 + 2.0 * hours), "B": 0.2 / (0.4 + 2.0 * hours)})
