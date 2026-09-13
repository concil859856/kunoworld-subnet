"""One machine, one miner; stake behind every GPU. The validator's own dedupe of verified hardware
and its collateral gate, against a scripted gateway and a fake chain."""

from __future__ import annotations

import time

import httpx
import pytest

from kuno_protocol.attestation import GoldenManifest, Verdict
from kuno_protocol.hardware import HardwareIdentity, hardware_token
from kuno_validator.collateral import (
    RAO_PER_ALPHA,
    CollateralGate,
    SubstrateCollateralReader,
    format_alpha,
    parse_alpha,
)
from kuno_validator.scoring import hardware_conflicts
from kuno_validator.validator import Validator

from test_receipt_ledger import FakeEnclave
from test_validator import API_KEY, FakeGateway

WINDOW = 3600.0


def gpu(name: str) -> HardwareIdentity:
    return HardwareIdentity("gpu", hardware_token("gpu", f"test:{name}"), "mock")


def platform(name: str) -> HardwareIdentity:
    return HardwareIdentity("cpu_platform", hardware_token("cpu_platform", f"test:{name}"), "mock")


def verdict(enclave: FakeEnclave, *hardware: HardwareIdentity, gpu_count: int | None = None) -> Verdict:
    count = gpu_count if gpu_count is not None else sum(1 for h in hardware if h.kind == "gpu")
    return Verdict(True, enclave.enclave_id, hardware=list(hardware), gpu_count=count)


class FakeChain:
    """`MinerCollateral` locked amounts per hotkey, or a failure."""

    def __init__(self, locked: dict[str, float]):
        self.locked = {hotkey: int(alpha * RAO_PER_ALPHA) for hotkey, alpha in locked.items()}
        self.fail = False
        self.reads: list[list[str]] = []

    def locked_collateral(self, netuid, hotkeys):
        self.reads.append(list(hotkeys))
        if self.fail:
            raise ConnectionError("chain endpoint unreachable")
        return {hotkey: self.locked.get(hotkey, 0) for hotkey in hotkeys}


def make_validator(gateway: FakeGateway, state_path=None, collateral: CollateralGate | None = None) -> Validator:
    validator = Validator(
        "http://gateway.test", API_KEY, GoldenManifest(), None,
        transport=httpx.MockTransport(gateway), state_path=state_path, collateral=collateral,
    )
    validator.enclaves()
    return validator


def live_ledger(*enclaves: FakeEnclave) -> list[dict]:
    now = time.time()
    rows = [e.entry(age_s=30) for e in enclaves]
    for row in rows:
        row["finished_at"] = now - 30
    return rows


# ---------------------------------------------------------------- the dedupe rule itself


def test_the_hotkey_that_showed_hardware_first_keeps_it():
    sightings = {gpu("g0").token: {"kind": "gpu", "hotkeys": {"A": [100.0, 500.0], "B": [400.0, 500.0]}}}
    penalties = hardware_conflicts(sightings, now=500.0, window_s=WINDOW)
    assert "A" not in penalties
    assert penalties["B"] == ["shares 1 verified hardware identity (gpu) first attested by A"]


def test_hotkeys_showing_the_same_hardware_in_the_same_round_are_all_zeroed():
    sightings = {platform("host").token: {"kind": "cpu_platform", "hotkeys": {"A": [100.0, 100.0], "B": [100.0, 100.0]}}}
    penalties = hardware_conflicts(sightings, now=100.0, window_s=WINDOW)
    assert penalties["A"] == ["shares 1 verified hardware identity (cpu platform) also attested by B in the same round"]
    assert "also attested by A in the same round" in penalties["B"][0]


def test_a_previous_owner_outside_the_window_no_longer_conflicts():
    sightings = {gpu("g0").token: {"kind": "gpu", "hotkeys": {"A": [0.0, 10.0], "B": [5000.0, 5000.0]}}}
    assert hardware_conflicts(sightings, now=5000.0, window_s=WINDOW) == {}


def test_many_shared_devices_collapse_into_one_reason():
    hotkeys = {"A": [1.0, 9.0], "B": [5.0, 9.0]}
    sightings = {
        gpu(f"g{i}").token: {"kind": "gpu", "hotkeys": dict(hotkeys)} for i in range(4)
    } | {platform("host").token: {"kind": "cpu_platform", "hotkeys": dict(hotkeys)}}
    assert hardware_conflicts(sightings, now=10.0, window_s=WINDOW)["B"] == [
        "shares 5 verified hardware identities (cpu platform, gpu) first attested by A"
    ]


# ---------------------------------------------------------------- dedupe inside the validator


def test_a_second_hotkey_on_the_same_gpu_gets_zero_weight_and_the_first_keeps_it():
    a, b = FakeEnclave("A"), FakeEnclave("B")
    gateway = FakeGateway(enclaves=[a, b], ledger=live_ledger(a, b))
    validator = make_validator(gateway)

    first = validator.score({a.enclave_id: verdict(a, gpu("g0"), platform("rig"))}, window_s=WINDOW)
    assert not first["A"].reasons
    time.sleep(0.01)
    second = validator.score(
        {a.enclave_id: verdict(a, gpu("g0"), platform("rig")), b.enclave_id: verdict(b, gpu("g0"), platform("rig2"))},
        window_s=WINDOW,
    )
    assert not second["A"].reasons and second["A"].score > 0
    assert second["B"].score == 0
    assert any("first attested by A" in r for r in second["B"].reasons)


def test_dedupe_uses_only_the_validators_own_verdicts():
    """A gateway can publish whatever hardware it likes; it can't frame a miner with it."""
    a, b = FakeEnclave("A"), FakeEnclave("B")
    shared = [{"kind": "gpu", "token": gpu("g0").token}]
    gateway = FakeGateway(enclaves=[a, b], ledger=live_ledger(a, b))
    gateway.enclaves = [dict(e, hardware_ids=shared) for e in gateway.enclaves]
    validator = make_validator(gateway)
    scores = validator.score({a.enclave_id: verdict(a, gpu("g1")), b.enclave_id: verdict(b, gpu("g2"))}, window_s=WINDOW)
    assert not scores["A"].reasons and not scores["B"].reasons


def test_failed_verdicts_are_not_recorded_as_sightings():
    a, b = FakeEnclave("A"), FakeEnclave("B")
    validator = make_validator(FakeGateway(enclaves=[a, b]))
    refused = Verdict(False, b.enclave_id, ["nope"], hardware=[gpu("g0")], gpu_count=1)
    validator.score({a.enclave_id: verdict(a, gpu("g0")), b.enclave_id: refused}, window_s=WINDOW)
    assert set(validator.hardware_sightings[gpu("g0").token]["hotkeys"]) == {"A"}


def test_step_weights_exclude_the_duplicate_and_sightings_survive_a_restart(tmp_path, caplog):
    a, b = FakeEnclave("A"), FakeEnclave("B")
    gateway = FakeGateway(enclaves=[a, b], ledger=live_ledger(a, b))
    state = tmp_path / "state.json"
    validator = make_validator(gateway, state_path=state)
    validator.check_enclaves = lambda: {a.enclave_id: verdict(a, gpu("g0"))}
    assert set(validator.step()) == {"A"}

    restarted = make_validator(gateway, state_path=state)
    restarted.check_enclaves = lambda: {a.enclave_id: verdict(a, gpu("g0")), b.enclave_id: verdict(b, gpu("g0"))}
    with caplog.at_level("INFO", logger="kuno.validator"):
        weights = restarted.step()
    assert weights == {"A": pytest.approx(1.0)}
    assert "miner B: score=0.0000" in caplog.text and "first attested by A" in caplog.text


# ---------------------------------------------------------------- collateral


def test_alpha_amounts_parse_exactly():
    assert parse_alpha("12.5") == 12_500_000_000 and parse_alpha("0") == 0
    assert format_alpha(12_500_000_000) == "12.5" and format_alpha(0) == "0"
    for bad in ("-1", "lots", "nan"):
        with pytest.raises(ValueError):
            parse_alpha(bad)


def test_collateral_is_required_per_attested_gpu():
    a, b = FakeEnclave("A"), FakeEnclave("B")
    chain = FakeChain({"A": 40.0, "B": 39.999})
    gate = CollateralGate(chain, netuid=7, min_per_gpu=parse_alpha("10"))
    gateway = FakeGateway(enclaves=[a, b], ledger=live_ledger(a, b))
    validator = make_validator(gateway, collateral=gate)
    gpus = [gpu(f"a{i}") for i in range(4)]
    scores = validator.score(
        {a.enclave_id: verdict(a, *gpus), b.enclave_id: verdict(b, *[gpu(f"b{i}") for i in range(4)])}, window_s=WINDOW
    )
    assert not scores["A"].reasons and scores["A"].score > 0
    assert scores["B"].score == 0
    assert scores["B"].reasons == ["collateral 39.999 alpha is below the 40 alpha required for 4 attested GPU(s) (10 per GPU)"]
    assert chain.reads == [["A", "B"]]


def test_gpus_are_counted_once_across_a_hotkeys_enclaves_and_unnamed_gpus_still_count():
    a1, a2, c = FakeEnclave("A"), FakeEnclave("A"), FakeEnclave("C")
    validator = make_validator(FakeGateway(enclaves=[a1, a2, c]))
    counts = validator.attested_gpus(
        {
            a1.enclave_id: verdict(a1, gpu("g0"), gpu("g1")),
            a2.enclave_id: verdict(a2, gpu("g1"), gpu("g2")),  # a replacement enclave still answering
            c.enclave_id: Verdict(True, c.enclave_id, gpu_count=None),  # verifier counted nothing
        }
    )
    assert counts == {"A": 3, "C": 1}


def test_a_failed_chain_read_keeps_the_last_view_for_a_bounded_time_then_fails_closed():
    chain = FakeChain({"A": 10.0})
    gate = CollateralGate(chain, netuid=7, min_per_gpu=parse_alpha("10"), max_stale_s=100)
    assert gate.penalties({"A": 1}, now=1000.0) == {}
    chain.fail = True
    assert gate.penalties({"A": 1}, now=1090.0) == {}  # within the bound: last good reading
    penalties = gate.penalties({"A": 1, "N": 1}, now=1090.0)
    assert list(penalties) == ["N"] and "chain read failed: ConnectionError" in penalties["N"][0]
    late = gate.penalties({"A": 1}, now=1101.0)
    assert "collateral unknown" in late["A"][0]
    chain.fail = False
    assert gate.penalties({"A": 1}, now=1200.0) == {}


def test_without_a_chain_every_gpu_holder_fails_the_requirement():
    gate = CollateralGate(None, netuid=None, min_per_gpu=parse_alpha("1"))
    assert "no chain is configured" in gate.penalties({"A": 2}, now=1.0)["A"][0]
    assert CollateralGate(None, None, 0).penalties({"A": 2}, now=1.0) == {}


def test_collateral_view_survives_a_restart_only_for_the_same_subnet(tmp_path):
    a = FakeEnclave("A")
    chain = FakeChain({"A": 5.0})
    state = tmp_path / "state.json"
    gateway = FakeGateway(enclaves=[a], ledger=live_ledger(a))
    validator = make_validator(gateway, state_path=state, collateral=CollateralGate(chain, 7, parse_alpha("5")))
    assert not validator.score({a.enclave_id: verdict(a, gpu("g0"))}, window_s=WINDOW)["A"].reasons

    chain.fail = True
    same = make_validator(gateway, state_path=state, collateral=CollateralGate(chain, 7, parse_alpha("5")))
    assert not same.score({a.enclave_id: verdict(a, gpu("g0"))}, window_s=WINDOW)["A"].reasons
    other = make_validator(gateway, state_path=state, collateral=CollateralGate(chain, 8, parse_alpha("5")))
    assert other.score({a.enclave_id: verdict(a, gpu("g0"))}, window_s=WINDOW)["A"].score == 0


def test_gate_from_env():
    assert CollateralGate.from_env({}, netuid=7) is None
    assert CollateralGate.from_env({"KUNO_MIN_COLLATERAL_PER_GPU": "0"}, netuid=7) is None
    gate = CollateralGate.from_env({"KUNO_MIN_COLLATERAL_PER_GPU": "2.5", "KUNO_COLLATERAL_MAX_STALE_S": "60"}, netuid=7, network="test")
    assert gate.min_per_gpu == 2_500_000_000 and gate.max_stale_s == 60
    assert isinstance(gate.reader, SubstrateCollateralReader) and gate.reader.url.startswith("wss://test.")
    assert CollateralGate.from_env({"KUNO_MIN_COLLATERAL_PER_GPU": "1"}, netuid=None).reader is None


class FakeSubstrate:
    """Answers the two storage items the reader uses, like substrate-interface's ScaleType results."""

    class Result:
        def __init__(self, value):
            self.value = value

    def __init__(self):
        self.owners = {"5Hot": "5Cold"}
        self.collateral = {(7, "5Hot", "5Cold"): {"locked": 84416858261, "drain_ratio": {"bits": 2**64}, "min_locked": 0, "earned": 0}}
        self.queries: list[tuple] = []
        self.closed = False

    def get_chain_finalised_head(self):
        return "0xblock"

    def query(self, module, storage, params, block_hash=None):
        self.queries.append((module, storage, tuple(params), block_hash))
        if storage == "Owner":
            return self.Result(self.owners.get(params[0], "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"))
        if storage == "MinerCollateral":
            return self.Result(self.collateral.get(tuple(params)))
        raise KeyError(storage)

    def close(self):
        self.closed = True


def test_the_substrate_reader_reads_the_owner_position_at_one_finalized_block():
    substrate = FakeSubstrate()
    reader = SubstrateCollateralReader("ws://chain.test", connect=lambda url: substrate)
    assert reader.locked_collateral(7, ["5Hot", "5Unregistered"]) == {"5Hot": 84416858261, "5Unregistered": 0}
    assert ("SubtensorModule", "MinerCollateral", (7, "5Hot", "5Cold"), "0xblock") in substrate.queries
    assert {q[3] for q in substrate.queries} == {"0xblock"}


def test_the_substrate_reader_reconnects_after_a_failure():
    substrates = []

    def connect(url):
        substrates.append(FakeSubstrate())
        return substrates[-1]

    reader = SubstrateCollateralReader("ws://chain.test", connect=connect)
    substrates_query = FakeSubstrate.query

    def broken(self, *args, **kwargs):
        raise ConnectionError("socket closed")

    FakeSubstrate.query = broken
    try:
        with pytest.raises(ConnectionError):
            reader.locked_collateral(7, ["5Hot"])
    finally:
        FakeSubstrate.query = substrates_query
    assert substrates[0].closed
    assert reader.locked_collateral(7, ["5Hot"]) == {"5Hot": 84416858261}
    assert len(substrates) == 2
