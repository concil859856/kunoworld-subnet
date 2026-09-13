"""Mechanism ids and commitments against fake bittensor 10 and 11 SDKs: weights reach the mechanism
they are meant for or are refused, and commitments come back with the block that orders them."""

from __future__ import annotations

import json
import secrets
from types import SimpleNamespace

import pytest

from kuno_protocol.hotkey import Sr25519Signer
from kuno_protocol.turbo import commitment_string
from kuno_validator.chain import (
    TURBO_MECHID,
    ChainCommitments,
    decode_commitment_info,
    read_commitments,
    set_weights,
)

from test_turbo_scoring import SPEC, signed_submission


class Bittensor10:
    """Subtensor.set_weights(..., mechid=0) plus get_all_commitments / get_commitment_metadata."""

    def __init__(self, hotkeys, *, mechid_param=True, result=True, commitments=None):
        self.hotkeys, self.result, self.calls = hotkeys, result, []
        self.commitments = commitments or {}
        outer = self

        if mechid_param:
            def submit(self, wallet, netuid, uids, weights, mechid=0, wait_for_inclusion=False):
                outer.calls.append(dict(wallet=wallet, netuid=netuid, uids=uids, weights=weights, mechid=mechid))
                return outer.result
        else:
            def submit(self, wallet, netuid, uids, weights, wait_for_inclusion=False):
                outer.calls.append(dict(wallet=wallet, netuid=netuid, uids=uids, weights=weights))
                return outer.result

        class Subtensor:
            def __init__(self, network):
                outer.network = network

            def metagraph(self, netuid):
                return SimpleNamespace(hotkeys=outer.hotkeys)

            set_weights = submit

            def get_all_commitments(self, netuid):
                return {hotkey: text for hotkey, (_block, text) in outer.commitments.items()}

            def get_commitment_metadata(self, netuid, hotkey):
                block, text = outer.commitments[hotkey]
                data = text.encode()
                return {"deposit": 0, "block": block, "info": {"fields": [{f"Raw{len(data)}": "0x" + data.hex()}]}}

        self.Subtensor = Subtensor
        self.Wallet = lambda name, hotkey: (name, hotkey)


class Bittensor11:
    """No Subtensor.set_weights: a module-level bt.set_weights intent and namespaced reads."""

    def __init__(self, hotkeys, *, rows=(), error: Exception | None = None):
        self.calls, self.error = [], error
        outer = self

        class Subtensor:
            def __init__(self, network="finney"):
                self.subnets = SimpleNamespace(
                    metagraph=lambda netuid: {"netuid": netuid, "hotkeys": list(hotkeys)},
                    commitments=lambda netuid: list(rows),
                )

        def set_weights(netuid, weights, *, uids=None, wallet=None, hotkey=None, mechid=0, version_key=0, network="finney", retries=2):
            if outer.error:
                raise outer.error
            outer.calls.append(dict(netuid=netuid, weights=weights, wallet=wallet, hotkey=hotkey, mechid=mechid, network=network))
            return SimpleNamespace(success=True, message="ok")

        self.Subtensor = Subtensor
        self.set_weights = set_weights


WEIGHTS = {"miner-a": 0.75, "miner-b": 0.25}


# ---------------------------------------------------------------- weights


def test_turbo_weights_go_to_mechanism_one_on_bittensor_10():
    bt = Bittensor10(["miner-a", "owner", "miner-b"])
    out = set_weights(WEIGHTS, 42, "w", "h", "finney", mechid=TURBO_MECHID, bt=bt)
    assert out["submitted"] is True and out["mechid"] == 1 and out["uids"] == [0, 2]
    assert bt.calls == [dict(wallet=("w", "h"), netuid=42, uids=[0, 2], weights=pytest.approx([0.75, 0.25]), mechid=1)]


def test_serving_weights_stay_on_mechanism_zero_by_default():
    bt = Bittensor10(["miner-a", "miner-b"])
    set_weights(WEIGHTS, 42, "w", "h", "finney", bt=bt)
    assert bt.calls[0]["mechid"] == 0


def test_an_sdk_without_mechanism_ids_is_refused_for_mechanism_one():
    bt = Bittensor10(["miner-a", "miner-b"], mechid_param=False)
    out = set_weights(WEIGHTS, 42, "w", "h", "finney", mechid=1, bt=bt)
    assert out["submitted"] is False and "cannot set weights for mechanism 1" in out["reason"]
    assert not bt.calls, "never let turbo weights overwrite the serving mechanism"
    assert set_weights(WEIGHTS, 42, "w", "h", "finney", mechid=0, bt=bt)["submitted"] is True
    assert "mechid" not in bt.calls[0]


@pytest.mark.parametrize("result, submitted", [(SimpleNamespace(success=False, message="rate limited"), False), ((True, ""), True), ((False, "x"), False)])
def test_sdk_response_shapes_are_read_correctly(result, submitted):
    bt = Bittensor10(["miner-a"], result=result)
    assert set_weights({"miner-a": 1.0}, 1, "w", "h", "finney", mechid=1, bt=bt)["submitted"] is submitted


def test_bittensor_11_intent_receives_uid_weights_and_mechanism():
    bt = Bittensor11(["owner", "miner-a", "miner-b"])
    out = set_weights(WEIGHTS, 42, "w", "h", "test", mechid=1, bt=bt)
    assert out["submitted"] is True and out["uids"] == [1, 2]
    [call] = bt.calls
    assert call["mechid"] == 1 and call["network"] == "test" and (call["wallet"], call["hotkey"]) == ("w", "h")
    assert call["weights"] == {1: pytest.approx(0.75), 2: pytest.approx(0.25)}


def test_a_chain_error_from_bittensor_11_is_reported_not_raised():
    bt = Bittensor11(["miner-a"], error=RuntimeError("NotRegistered"))
    out = set_weights({"miner-a": 1.0}, 42, "w", "h", "finney", mechid=1, bt=bt)
    assert out["submitted"] is False and "NotRegistered" in out["reason"]


def test_dry_run_reports_the_mechanism_and_touches_nothing():
    bt = Bittensor10(["miner-a", "miner-b"])
    out = set_weights(WEIGHTS, 7, "w", "h", "finney", mechid=1, dry_run=True, bt=bt)
    assert out["dry_run"] is True and out["mechid"] == 1 and out["uids"] == [0, 1] and not bt.calls


@pytest.mark.parametrize("mechid", [-1, 256, "1", 1.0])
def test_invalid_mechanism_ids_are_rejected(mechid):
    with pytest.raises(ValueError, match="mechanism id"):
        set_weights(WEIGHTS, 1, "w", "h", "finney", mechid=mechid, bt=Bittensor10(["miner-a"]))


# ---------------------------------------------------------------- commitments


def test_raw_commitment_fields_decode_as_finney_returns_them():
    # Shape copied from a live Commitments.CommitmentOf entry on finney (spec 455).
    info = {"fields": [{"Raw64": "0x4573656c53636877616e7a2f736e31392d67352d313a313a64376264303937653235613634663437303132386266313139326635653465383163333332616133"}]}
    assert decode_commitment_info(info) == "EselSchwanz/sn19-g5-1:1:d7bd097e25a64f470128bf1192f5e4e81c332aa3"
    assert decode_commitment_info({"fields": [[{"Raw3": "0x6b7431"}]], "x": 1}) == "kt1"
    assert decode_commitment_info({"fields": [{"Sha256": "0x" + "00" * 32}]}) == ""


def test_bittensor_10_commitments_carry_blocks_and_skip_unregistered_hotkeys():
    bt = Bittensor10(["miner-a", "miner-b"], commitments={"miner-a": (1200, "kt1:abc"), "gone": (1100, "kt1:def"), "miner-b": (900, "hello")})
    found = read_commitments(42, "finney", bt=bt)
    assert sorted((c.hotkey, c.block, c.data, c.uid) for c in found) == [("miner-a", 1200, "kt1:abc", 0), ("miner-b", 900, "hello", 1)]


def test_a_fake_chain_feeds_turbo_submissions_end_to_end():
    """Commitment parsing with a fake bittensor 11 chain: the row's text resolves to a signed document."""
    miner = Sr25519Signer.from_seed(secrets.token_bytes(32))
    signed = signed_submission(miner, "sha256:chain-entry")
    url = "https://entries.test/one.json"
    rows = [
        {"hotkey": miner.ss58_address, "uid": 3, "commitment": commitment_string(signed.digest(), url), "block": 777, "is_revealed": True},
        {"hotkey": "5Deregistered", "uid": None, "commitment": "kt1:whatever", "block": 1},
        {"hotkey": "5Sealed", "uid": 4, "commitment": None, "block": 2, "is_revealed": False},
    ]
    source = ChainCommitments(42, "finney", bt=Bittensor11([], rows=rows))
    from kuno_protocol.turbo import collect_submissions

    accepted, rejected = collect_submissions(SPEC, source(), {url: json.dumps(signed.model_dump(mode="json")).encode()}.__getitem__)
    assert [(a.hotkey, a.block, a.digest) for a in accepted] == [(miner.ss58_address, 777, signed.digest())]
    assert not rejected
