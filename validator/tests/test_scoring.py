"""Scoring rules decide who gets paid, so they are pinned down here rather than
only exercised through the end-to-end tests."""

from __future__ import annotations

import pytest

from kuno_protocol.profiles import load_profiles
from kuno_protocol.switch import SwitchConfig
from kuno_validator.scoring import compute_scores, normalize

PROFILES = load_profiles()
NOW = 1_800_000_000.0


def job(hotkey: str, profile_id: str, duration_s: float = 5.0, *, status: str = "succeeded", error_code: str | None = None, age_s: float = 60.0, claimed_s: float | None = None) -> dict:
    """An already-audited ledger row. `duration_s` is the public (paid) duration; `claimed_s` is what the miner says."""
    receipt = {"body": {"video": {"duration_s": duration_s if claimed_s is None else claimed_s}}} if status == "succeeded" else None
    return {
        "miner_hotkey": hotkey,
        "profile_id": profile_id,
        "status": status,
        "error_code": error_code,
        "duration_s": duration_s,
        "finished_at": NOW - age_s,
        "receipt": receipt,
    }


def score(ledger, attested=("A", "B"), switch=None, **kwargs):
    return compute_scores(ledger, set(attested), PROFILES, switch or SwitchConfig(), NOW, **kwargs)


def test_pay_is_proportional_to_verified_video_compute_units():
    scores = score([job("A", "ltx-2.5-fast", 5)] + [job("B", "ltx-2.5-fast", 5) for _ in range(3)])
    weights = normalize(scores)
    assert weights["A"] == pytest.approx(0.25) and weights["B"] == pytest.approx(0.75)
    # A longer clip costs more GPU time per second, so one 15 s clip is worth more than three 5 s ones.
    fast = PROFILES["ltx-2.5-fast"]
    longer = normalize(score([job("A", "ltx-2.5-fast", 5), job("B", "ltx-2.5-fast", 15)]))
    assert longer["A"] == pytest.approx(fast.vcu(5) / (fast.vcu(5) + fast.vcu(15))) and fast.vcu(15) > 3 * fast.vcu(5)


def test_heavier_models_earn_more_per_second():
    """A second of H3 is worth more compute than a second of LTX, per the profile weights."""
    scores = score([job("A", "h3", 5), job("B", "ltx-2.5-fast", 5)], switch=SwitchConfig(emission_split={"minimax-h3": 0.5, "ltx-2.5": 0.5}))
    assert PROFILES["h3"].vcu(5) > PROFILES["ltx-2.5-fast"].vcu(5)
    # With an even family split, each family's pool is shared inside that family.
    assert normalize(scores)["A"] == pytest.approx(0.5)


def test_family_split_follows_the_owner_switch():
    ledger = [job("A", "h3", 5), job("B", "ltx-2.5-fast", 5)]
    weights = normalize(score(ledger, switch=SwitchConfig(emission_split={"minimax-h3": 0.7, "ltx-2.5": 0.3})))
    assert weights["A"] == pytest.approx(0.7) and weights["B"] == pytest.approx(0.3)


def test_work_on_a_switched_off_family_earns_nothing():
    weights = normalize(score([job("A", "h3", 5), job("B", "ltx-2.5-fast", 5)], switch=SwitchConfig(mode="ltx")))
    assert weights == {"B": pytest.approx(1.0)}


def test_a_miner_without_a_live_attestation_is_excluded():
    scores = score([job("A", "ltx-2.5-fast"), job("B", "ltx-2.5-fast")], attested=("B",))
    assert scores["A"].score == 0 and "no currently attested enclave" in scores["A"].reasons
    assert normalize(scores) == {"B": pytest.approx(1.0)}


def test_attested_miners_with_no_work_appear_but_earn_nothing():
    scores = score([job("A", "ltx-2.5-fast")], attested=("A", "B"))
    assert "B" in scores and scores["B"].score == 0 and not scores["B"].reasons
    assert normalize(scores) == {"A": pytest.approx(1.0)}


def test_only_failures_the_miner_caused_count_against_reliability():
    ledger = [job("A", "ltx-2.5-fast") for _ in range(30)]
    ledger += [job("A", "ltx-2.5-fast", status="failed", error_code="safety_blocked") for _ in range(10)]
    scores = score(ledger)
    assert scores["A"].failed == 0 and not scores["A"].reasons  # a blocked prompt is the customer's doing

    ledger += [job("A", "ltx-2.5-fast", status="failed", error_code="internal_error") for _ in range(10)]
    scores = score(ledger)
    assert scores["A"].failed == 10 and any("success rate" in r for r in scores["A"].reasons)


def test_the_reliability_gate_waits_for_enough_samples():
    few = [job("A", "ltx-2.5-fast"), job("A", "ltx-2.5-fast", status="failed", error_code="timeout")]
    assert not score(few)["A"].reasons  # 50% success, but only 2 jobs: too early to judge

    many = [job("A", "ltx-2.5-fast") for _ in range(19)] + [job("A", "ltx-2.5-fast", status="failed", error_code="timeout")]
    assert any("success rate" in r for r in score(many)["A"].reasons)  # 95% over 20 jobs


def test_work_outside_the_window_does_not_count():
    scores = score([job("A", "ltx-2.5-fast", age_s=200_000), job("B", "ltx-2.5-fast", age_s=60)])
    assert normalize(scores) == {"B": pytest.approx(1.0)}


def test_weights_sum_to_one_or_are_empty():
    assert normalize(score([job("A", "ltx-2.5-fast"), job("B", "h3")])) and sum(
        normalize(score([job("A", "ltx-2.5-fast"), job("B", "h3")])).values()
    ) == pytest.approx(1.0)
    # Nothing to pay: the validator keeps its previous weights rather than burning.
    assert normalize(score([], attested=())) == {}


def test_unknown_profiles_in_the_ledger_are_ignored():
    scores = score([job("A", "some-retired-profile"), job("B", "ltx-2.5-fast")])
    assert normalize(scores) == {"B": pytest.approx(1.0)}


def test_pay_follows_the_public_duration_not_the_miners_claim():
    scores = score([job("A", "ltx-2.5-fast", 5, claimed_s=500), job("B", "ltx-2.5-fast", 5)])
    assert normalize(scores) == {"A": pytest.approx(0.5), "B": pytest.approx(0.5)}


def test_uncredited_entries_count_for_reliability_but_earn_nothing():
    ledger = [dict(job("A", "ltx-2.5-fast"), credit=False), job("B", "ltx-2.5-fast")]
    scores = score(ledger)
    assert scores["A"].succeeded == 1 and scores["A"].score == 0 and not scores["A"].reasons
    assert normalize(scores) == {"B": pytest.approx(1.0)}


def test_a_penalty_zeroes_a_miner_for_the_window():
    ledger = [job("A", "ltx-2.5-fast", 15), job("B", "ltx-2.5-fast", 5)]
    scores = score(ledger, penalties={"A": ["failed canary ltx-2.5-fast (output does not match the receipt's content digest)"]})
    assert scores["A"].score == 0 and any("failed canary" in r for r in scores["A"].reasons)
    assert normalize(scores) == {"B": pytest.approx(1.0)}


def test_flags_are_reported_without_disqualifying():
    scores = score([job("A", "ltx-2.5-fast")], flags={"A": ["job x: receipt reports 1s for a 5s request"]})
    assert scores["A"].flags and not scores["A"].reasons and scores["A"].score > 0


def test_a_stale_now_does_not_resurrect_old_work():
    later = compute_scores([job("A", "ltx-2.5-fast", age_s=0)], {"A"}, PROFILES, SwitchConfig(), NOW + 100_000)
    assert normalize(later) == {}
