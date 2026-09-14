"""USD-denominated miner pay (KUNO_PAY_MODE=usd) against a fake chain and fake price sources: normal, undersubscribed,
oversubscribed, stale-price and missing-card rounds, with every scoring gate still applied."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from kuno_protocol.attestation import GoldenManifest, Verdict
from kuno_protocol.crypto import generate_signing_key, public_key_bytes
from kuno_protocol.profiles import load_profiles
from kuno_protocol.rate_card import RateCard, sign_rate_card
from kuno_protocol.schemas import GenerationParams
from kuno_protocol.switch import SwitchConfig
from kuno_validator import main as validator_main
from kuno_validator.emission import EmissionUnavailable, SubnetEmission, SubstrateEmissionReader, mechanism_share
from kuno_validator.price_feeds import SOURCES, PriceUnavailable, TaoUsdOracle
from kuno_validator.scoring import MinerScore
from kuno_validator.usd_pay import OwedWork, PayPolicy, PayUnavailable, UsdPay, settle, usd_owed
from kuno_validator.validator import CanaryResult, Validator

from test_receipt_ledger import FakeEnclave
from test_validator import API_KEY, FakeGateway

PROFILES = load_profiles()
LTX, H3 = "ltx-2.5-fast", "h3"
RATES = {LTX: {"confidential": 0.05, "open": 0.02}, H3: {"confidential": 0.30}}


def emission(**changes) -> SubnetEmission:
    """1 alpha per block, 18% owner cut, 360-block epochs (4320 s), all of it to mechanism 0: 147.6 miner alpha per tempo."""
    values = dict(netuid=7, block="0xabc", alpha_out_per_block=1.0, owner_cut=0.18, tempo_blocks=359, mechanism_share=1.0, tao_per_alpha=0.01)
    return SubnetEmission(**{**values, **changes})


class FakeReader:
    def __init__(self, value: SubnetEmission | None = None, error: Exception | None = None):
        self.value, self.error, self.calls = value or emission(), error, []

    def read(self, netuid, mechid=0):
        self.calls.append((netuid, mechid))
        if self.error is not None:
            raise self.error
        return self.value


def _iso(at: float) -> str:
    return datetime.fromtimestamp(at, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def oracle(kraken=300.0, coinbase=301.0, coingecko=299.5, age_s=30.0, stale=(), down=()) -> TaoUsdOracle:
    """The real parsers over canned answers; `stale` sources date their quote two hours back."""
    now = time.time()

    def dated(name):
        return now - 7200 if name in stale else now - age_s

    answers = {
        "kraken": {"error": [], "result": {"TAOUSD": {"c": [str(kraken), "1.0"]}}},
        "coinbase": {"price": str(coinbase), "time": _iso(dated("coinbase"))},
        "coingecko": {"bittensor": {"usd": coingecko, "last_updated_at": int(dated("coingecko"))}},
    }
    by_url = {source.url: name for name, source in SOURCES.items()}

    def fetch(url):
        name = by_url[url]
        if name in down:
            raise httpx.ConnectError("unreachable")
        return answers[name]

    return TaoUsdOracle(fetch=fetch)


def write_card(path: Path, owner, issued_at: int = 1000, rates=None, placeholder: bool = False) -> Path:
    card = RateCard(issued_at=issued_at, usd_per_second=rates or RATES, placeholder=placeholder)
    path.write_text(sign_rate_card(owner, card).model_dump_json())
    return path


def make_pay(tmp_path, owner, reader=None, price=None, card_path=None, report=True) -> UsdPay:
    policy = PayPolicy(mode="usd", rate_card_path=card_path if card_path is not None else write_card(tmp_path / "card.json", owner))
    return UsdPay(policy, 7, reader or FakeReader(), price or oracle(), public_key_bytes(owner),
                  report_path=tmp_path / "pay.jsonl" if report else None)


def attested(validator: Validator, *enclaves: FakeEnclave) -> None:
    validator.check_enclaves = lambda: {e.enclave_id: Verdict(True, e.enclave_id, []) for e in enclaves}


def usd_validator(tmp_path, owner, gateway, pay) -> Validator:
    validator = Validator("http://gateway.test", API_KEY, GoldenManifest(), public_key_bytes(owner),
                          transport=httpx.MockTransport(gateway), state_path=tmp_path / "state.json", pay=pay)
    return validator


@pytest.fixture
def owner():
    return generate_signing_key()


@pytest.fixture
def network():
    a, b = FakeEnclave("A"), FakeEnclave("B")
    ledger = [a.entry(), a.entry(), a.entry(), b.entry()]  # 4 s each: A 12 s, B 4 s of ltx-2.5-fast
    return a, b, FakeGateway(enclaves=[a, b], ledger=ledger)


# ---------------------------------------------------------------- configuration


def test_todays_scoring_stays_the_default():
    assert PayPolicy.from_env({}).mode == "vcu"
    assert UsdPay.from_env({}, 7, "finney", None, None) is None
    pay = UsdPay.from_env({"KUNO_PAY_MODE": "USD", "KUNO_RATE_CARD": "/tmp/card.json"}, 7, "finney", None, Path("/var/lib/kuno/state.json"))
    assert pay is not None and pay.policy.usd and pay.policy.residual == "renormalize"
    assert pay.report_path == Path("/var/lib/kuno/state-pay.jsonl")
    assert pay.oracle.min_sources == 2


def test_recycle_is_refused_with_the_emission_penalty_explained():
    with pytest.raises(ValueError, match="MinerBurned"):
        PayPolicy.from_env({"KUNO_PAY_MODE": "usd", "KUNO_PAY_RESIDUAL": "recycle"})
    with pytest.raises(ValueError, match="KUNO_PAY_MODE"):
        PayPolicy.from_env({"KUNO_PAY_MODE": "eur"})
    with pytest.raises(ValueError, match="KUNO_PAY_RESIDUAL"):
        PayPolicy.from_env({"KUNO_PAY_MODE": "usd", "KUNO_PAY_RESIDUAL": "burn"})


# ---------------------------------------------------------------- chain and prices


def test_the_serving_miner_pool_follows_owner_cut_tempo_and_mechanism_split():
    pool = emission(mechanism_share=mechanism_share([13107, 52428], 2, 0))
    assert pool.epoch_blocks == 360 and pool.tempo_seconds == 4320.0
    assert pool.miner_alpha_per_tempo == pytest.approx(360 * 0.82 * 0.5 * 0.2)
    assert pool.emission_alpha(86400) == pytest.approx(7200)
    assert mechanism_share(None, 2, 1) == 0.5
    assert mechanism_share([13107, 52428], 2, 1) == pytest.approx(0.8)
    assert mechanism_share([], 1, 0) == 1.0
    assert mechanism_share([65535], 1, 1) == 0.0


class FakeSubstrate:
    def __init__(self, fail: bool = False):
        self.fail, self.closed, self.calls = fail, False, []
        self.storage = {
            "SubnetAlphaOutEmission": 1_000_000_000, "SubnetOwnerCut": 11796, "OwnerCutEnabled": True, "Tempo": 360,
            "MechanismCountCurrent": 2, "MechanismEmissionSplit": [13107, 52428], "SubnetMovingPrice": {"bits": 3 << 30},
            "MinerBurned": {"bits": 1 << 62},
        }

    def get_chain_finalised_head(self):
        return "0xhead"

    def query(self, module, name, params, block_hash=None):
        assert (module, block_hash) == ("SubtensorModule", "0xhead")
        if self.fail:
            raise ConnectionError("socket closed")
        return SimpleNamespace(value=self.storage[name])

    def rpc_request(self, method, params):
        self.calls.append((method, params))
        return {"jsonrpc": "2.0", "result": "0x1e546b0300000000", "id": 1}  # finney's answer for netuid 4

    def close(self):
        self.closed = True


def test_the_substrate_reader_decodes_emission_and_the_pool_price():
    substrate = FakeSubstrate()
    reading = SubstrateEmissionReader("ws://chain", connect=lambda _url: substrate).read(4)
    assert substrate.calls == [("state_call", ["SwapRuntimeApi_current_alpha_price", "0x0400", "0xhead"])]
    assert reading.tao_per_alpha == pytest.approx(0.057365534)  # 0x036b541e rao
    assert reading.alpha_out_per_block == 1.0 and reading.tempo_blocks == 360
    assert reading.owner_cut == pytest.approx(11796 / 65535)
    assert reading.mechanism_share == pytest.approx(0.2)
    assert reading.moving_tao_per_alpha == pytest.approx(0.75) and reading.miner_burned == pytest.approx(0.25)
    broken = FakeSubstrate(fail=True)
    with pytest.raises(EmissionUnavailable, match="socket closed"):
        SubstrateEmissionReader("ws://chain", connect=lambda _url: broken).read(4)
    assert broken.closed


def test_tao_usd_is_a_median_of_fresh_sources_that_agree():
    quote = oracle().quote()
    assert quote.usd_per_tao == 300.0 and set(quote.quotes) == {"kraken", "coinbase", "coingecko"}
    assert oracle(stale=("coingecko",)).quote().usd_per_tao == 300.5  # two fresh sources are enough
    with pytest.raises(PriceUnavailable, match="stale"):
        oracle(stale=("coinbase", "coingecko")).quote()  # Kraken alone can't be checked against anything
    with pytest.raises(PriceUnavailable, match="disagree"):
        oracle(kraken=300.0, coinbase=330.0, coingecko=310.0).quote()
    with pytest.raises(PriceUnavailable, match="unavailable"):
        oracle(down=("kraken",), stale=("coinbase",)).quote()


# ---------------------------------------------------------------- pricing and weights


def _entry(hotkey, profile, seconds, tier="confidential", **extra):
    params = GenerationParams(profile_id=profile, mode="text_to_video", duration_s=seconds, resolution=next(iter(PROFILES[profile].limits.sizes)),
                              aspect_ratio=next(iter(next(iter(PROFILES[profile].limits.sizes.values())))), fps=24)
    return {"job_id": f"{hotkey}-{profile}-{seconds}-{tier}-{len(extra)}", "miner_hotkey": hotkey, "profile_id": profile, "status": "succeeded",
            "receipt": {"body": {}}, "finished_at": 1000.0, "billable_s": seconds, "credit": True, "tier": tier,
            "params": params.model_dump(mode="json"), **extra}


def test_owed_usd_prices_each_profile_and_tier_for_gated_miners_only():
    card = RateCard(usd_per_second=RATES, placeholder=False)
    entries = [
        _entry("A", LTX, 10), _entry("A", H3, 5),
        _entry("B", LTX, 10, tier="open"),
        _entry("B", H3, 5, tier="open"),                         # the card doesn't price H3 on the open tier
        _entry("C", LTX, 10),                                    # C failed a gate
        _entry("A", LTX, 10, credit=False),                      # e.g. a duration mismatch: no pay
        _entry("A", LTX, 10, credit=False, replay_of="x"),       # a replay: no pay, and not revenue either
        {**_entry("A", LTX, 10), "status": "failed", "receipt": None},
        {**_entry("A", LTX, 10), "finished_at": 1.0},            # outside the window
    ]
    work = usd_owed(entries, {"A", "B"}, card, PROFILES, SwitchConfig(), now=1100.0, window_s=500.0)
    assert work.owed_usd == pytest.approx({"A": 10 * 0.05 + 5 * 0.30, "B": 10 * 0.02})
    assert work.unpriced == {"h3@open": 5.0}
    listed = [e for e in entries[:6]]
    assert work.revenue_jobs == 6
    assert work.revenue_usd == pytest.approx(sum(PROFILES[e["profile_id"]].price_usd(GenerationParams.model_validate(e["params"])) for e in listed))
    # A family the switch turns off earns nothing in USD mode either.
    ltx_only = usd_owed(entries, {"A", "B"}, card, PROFILES, SwitchConfig(mode="ltx"), now=1100.0, window_s=500.0)
    assert ltx_only.owed_usd == pytest.approx({"A": 0.5, "B": 0.2})


def test_undersubscribed_rounds_renormalize_up_and_burn_nothing():
    card = RateCard(issued_at=9, usd_per_second=RATES, placeholder=True)
    work = OwedWork(owed_usd={"A": 0.60, "B": 0.20}, revenue_usd=2.0, revenue_jobs=4)
    quote = oracle().quote()
    weights, report = settle(work, emission(), quote, card, now=5.0, window_s=86400.0)
    assert weights == pytest.approx({"A": 0.75, "B": 0.25})
    assert set(weights) == {"A", "B"}  # no owner uid, no burn uid
    assert report.regime == "undersubscribed" and report.subscription < 1
    assert report.pool_usd_per_tempo == pytest.approx(147.6 * 0.01 * 300.0)
    assert report.owed_usd_per_tempo == pytest.approx(0.80 * 4320 / 86400)
    assert report.subsidy_ratio == pytest.approx(442.8 / 0.04)
    assert report.emission_usd_window == pytest.approx(7200 * 3.0)
    assert report.emission_to_revenue == pytest.approx(21600 / 2.0)
    assert report.rate_card_placeholder and report.miners["A"]["raw_weight"] == pytest.approx(0.03 / 442.8)
    json.loads(report.to_json())


def test_oversubscribed_rounds_renormalize_down():
    work = OwedWork(owed_usd={"A": 0.60, "B": 0.20})
    weights, report = settle(work, emission(tao_per_alpha=1e-7), oracle().quote(), RateCard(usd_per_second=RATES), now=5.0, window_s=86400.0)
    assert report.regime == "oversubscribed" and report.subscription > 1
    assert report.subsidy_ratio < 1
    assert weights == pytest.approx({"A": 0.75, "B": 0.25})
    assert sum(weights.values()) == pytest.approx(1.0)


def test_a_worthless_pool_or_no_work():
    card = RateCard(usd_per_second=RATES)
    with pytest.raises(PayUnavailable, match="worth"):
        settle(OwedWork(owed_usd={"A": 1.0}), emission(tao_per_alpha=0.0), oracle().quote(), card, 0.0, 86400.0)
    weights, report = settle(OwedWork(), emission(), oracle().quote(), card, 0.0, 86400.0)
    assert weights == {} and report.regime == "no_work" and report.subsidy_ratio is None and report.emission_to_revenue is None
    with pytest.raises(ValueError, match="MinerBurned"):
        settle(OwedWork(), emission(), oracle().quote(), card, 0.0, 86400.0, residual="recycle")


# ---------------------------------------------------------------- whole rounds through the validator


def test_a_normal_round_pays_by_usd_owed_logs_and_exports_the_kpis(tmp_path, owner, network, caplog):
    a, b, gateway = network
    pay = make_pay(tmp_path, owner)
    validator = usd_validator(tmp_path, owner, gateway, pay)
    attested(validator, a, b)
    with caplog.at_level(logging.INFO, logger="kuno.validator.usd_pay"):
        weights = validator.step()
    assert weights == pytest.approx({"A": 0.75, "B": 0.25})
    assert "subsidy ratio" in caplog.text and "emission/revenue" in caplog.text
    [line] = (tmp_path / "pay.jsonl").read_text().splitlines()
    report = json.loads(line)
    assert report["regime"] == "undersubscribed" and report["subsidy_ratio"] > 1 and report["emission_to_revenue"] > 0
    assert report["miners"]["A"]["usd_owed"] == pytest.approx(12 * 0.05)
    assert json.loads((tmp_path / "state.json").read_text())["rate_card"]["card"]["issued_at"] == 1000


def test_every_scoring_gate_still_applies(tmp_path, owner, network):
    a, b, gateway = network
    validator = usd_validator(tmp_path, owner, gateway, make_pay(tmp_path, owner))
    attested(validator, a, b)
    validator.canary_results.append(CanaryResult(LTX, False, "wrong size", "job-x", a.enclave_id, "A", True, at=time.time()))
    assert validator.step() == pytest.approx({"B": 1.0})
    # And without a live attestation nobody earns, as in VCU scoring.
    validator.canary_results.clear()
    attested(validator)
    assert validator.step() == {}


def test_an_oversubscribed_round_through_the_validator(tmp_path, owner, network):
    a, b, gateway = network
    pay = make_pay(tmp_path, owner, reader=FakeReader(emission(alpha_out_per_block=1e-9)))
    validator = usd_validator(tmp_path, owner, gateway, pay)
    attested(validator, a, b)
    assert validator.step() == pytest.approx({"A": 0.75, "B": 0.25})
    assert pay.last_report.regime == "oversubscribed"


def test_stale_prices_keep_the_previous_weights(tmp_path, owner, network, caplog):
    a, b, gateway = network
    pay = make_pay(tmp_path, owner, price=oracle(stale=("coinbase", "coingecko")))
    validator = usd_validator(tmp_path, owner, gateway, pay)
    attested(validator, a, b)
    with pytest.raises(PayUnavailable, match="stale"):
        validator.step()
    with caplog.at_level(logging.ERROR, logger="kuno.validator"):
        assert validator_main.serving_round(validator, [], []) is None  # main then submits nothing
    assert "leaving the previous serving weights" in caplog.text
    assert not (tmp_path / "pay.jsonl").exists()


def test_an_unreadable_chain_keeps_the_previous_weights(tmp_path, owner, network):
    a, b, gateway = network
    pay = make_pay(tmp_path, owner, reader=FakeReader(error=EmissionUnavailable("endpoint down")))
    validator = usd_validator(tmp_path, owner, gateway, pay)
    attested(validator, a, b)
    with pytest.raises(PayUnavailable, match="endpoint down"):
        validator.step()


def test_a_missing_or_forged_card_keeps_the_previous_weights(tmp_path, owner, network):
    a, b, gateway = network
    missing = make_pay(tmp_path, owner, card_path=tmp_path / "nowhere.json")
    validator = usd_validator(tmp_path, owner, gateway, missing)
    attested(validator, a, b)
    with pytest.raises(PayUnavailable, match="no rate card"):
        validator.step()
    forged = write_card(tmp_path / "forged.json", generate_signing_key())
    with pytest.raises(PayUnavailable, match="not signed by the owner"):
        make_pay(tmp_path, owner, card_path=forged).rate_card()
    unset = UsdPay(PayPolicy(mode="usd"), 7, FakeReader(), oracle(), public_key_bytes(owner))
    with pytest.raises(PayUnavailable, match="KUNO_RATE_CARD is not set"):
        unset.rate_card()


def test_an_accepted_card_survives_a_lost_file_rollbacks_and_restarts(tmp_path, owner, network, caplog):
    a, b, gateway = network
    card = write_card(tmp_path / "card.json", owner, issued_at=2000)
    pay = make_pay(tmp_path, owner, card_path=card)
    validator = usd_validator(tmp_path, owner, gateway, pay)
    attested(validator, a, b)
    assert validator.step()
    card.unlink()
    assert validator.step() == pytest.approx({"A": 0.75, "B": 0.25})  # the accepted card still prices the round
    write_card(card, owner, issued_at=1500, rates={LTX: {"confidential": 9.0}})
    assert pay.rate_card().issued_at == 2000  # an older card is a rollback

    restarted = make_pay(tmp_path, owner, card_path=card)
    usd_validator(tmp_path, owner, gateway, restarted)  # loads the state file
    assert restarted.accepted is not None and restarted.rate_card().issued_at == 2000

    placeholder = write_card(tmp_path / "placeholder.json", owner, issued_at=3000, placeholder=True)
    with caplog.at_level(logging.ERROR, logger="kuno.validator.usd_pay"):
        assert make_pay(tmp_path, owner, card_path=placeholder).rate_card().placeholder
    assert "PLACEHOLDER" in caplog.text


def test_without_pay_the_validator_scores_as_before(tmp_path, owner, network):
    a, b, gateway = network
    validator = Validator("http://gateway.test", API_KEY, GoldenManifest(), public_key_bytes(owner), transport=httpx.MockTransport(gateway))
    attested(validator, a, b)
    assert validator.step() == pytest.approx({"A": 0.75, "B": 0.25})  # VCU: 12 s against 4 s of one profile
    assert isinstance(next(iter(validator.score({}).values())), MinerScore)
