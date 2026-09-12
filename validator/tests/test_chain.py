"""Weight submission, tested against a stub chain: the real bittensor API cannot run
here, but the mapping and the refusal rules can."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kuno_validator.chain import resolve_uids, set_weights


class StubBittensor:
    """Mimics the small slice of the bittensor API we use."""

    def __init__(self, hotkeys: list[str], result: bool = True):
        self.hotkeys = hotkeys
        self.result = result
        self.submitted: list[dict] = []
        outer = self

        class Subtensor:
            def __init__(self, network: str):
                outer.network = network

            def metagraph(self, netuid: int):
                outer.netuid = netuid
                return SimpleNamespace(hotkeys=outer.hotkeys)

            def set_weights(self, **kwargs):
                outer.submitted.append(kwargs)
                return outer.result

        class Wallet:
            def __init__(self, name: str, hotkey: str):
                outer.wallet = (name, hotkey)

        self.Subtensor = Subtensor
        self.Wallet = Wallet


def test_hotkeys_map_to_uids_and_renormalize_over_registered_miners():
    uids, values = resolve_uids(["a", "b", "c"], {"a": 0.25, "c": 0.25})
    assert uids == [0, 2] and values == [0.5, 0.5]  # unregistered "b" is gone; the rest add up to 1


def test_weights_are_submitted_for_registered_hotkeys():
    bt = StubBittensor(hotkeys=["miner-a", "owner", "miner-b"])
    out = set_weights({"miner-a": 0.75, "miner-b": 0.25}, 42, "wallet", "hotkey", "finney", bt=bt)
    assert out["submitted"] is True and out["uids"] == [0, 2]
    assert out["weights"] == pytest.approx([0.75, 0.25])
    assert bt.netuid == 42 and bt.network == "finney" and bt.wallet == ("wallet", "hotkey")
    call = bt.submitted[0]
    assert call["netuid"] == 42 and call["uids"] == [0, 2] and call["wait_for_inclusion"] is True
    assert 1 not in call["uids"], "the owner hotkey must never receive weight"


def test_nothing_is_submitted_when_no_miner_qualifies():
    bt = StubBittensor(hotkeys=["miner-a"])
    out = set_weights({}, 42, "w", "h", "finney", bt=bt)
    assert out == {"submitted": False, "reason": "no qualifying miners"}
    assert not bt.submitted  # previous weights stay; we never submit zeros or burn


def test_nothing_is_submitted_when_scored_miners_are_not_registered():
    bt = StubBittensor(hotkeys=["someone-else"])
    out = set_weights({"miner-a": 1.0}, 42, "w", "h", "finney", bt=bt)
    assert out["submitted"] is False and out["reason"] == "no scored hotkey is registered"
    assert not bt.submitted


def test_dry_run_resolves_everything_but_touches_nothing():
    bt = StubBittensor(hotkeys=["miner-a", "miner-b"])
    out = set_weights({"miner-a": 0.5, "miner-b": 0.5, "gone": 0.2}, 7, "w", "h", "test", dry_run=True, bt=bt)
    assert out["dry_run"] is True and out["uids"] == [0, 1] and out["dropped"] == ["gone"]
    assert out["weights"] == pytest.approx([0.5, 0.5])
    assert not bt.submitted


def test_a_rejected_submission_is_reported():
    bt = StubBittensor(hotkeys=["miner-a"], result=False)
    assert set_weights({"miner-a": 1.0}, 42, "w", "h", "finney", bt=bt)["submitted"] is False
