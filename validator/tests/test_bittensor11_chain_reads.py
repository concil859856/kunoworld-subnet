"""The chain paths `kuno-validator[chain]` takes with bittensor 11.1.0, against stubs shaped like what that SDK returned
from a spec-458 localnet (scripts/localnet/): a wallet kept outside ~/.bittensor, `{hotkey: NeuronCommitment}`
commitments, and collateral and emission reads over bittensor's own RPC client (11.x ships no substrate-interface)."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from kuno_validator.chain import read_commitments, registered_hotkeys, set_weights
from kuno_validator.collateral import CollateralUnavailable, SubstrateCollateralReader, _default_substrate
from kuno_validator.emission import SubstrateEmissionReader

WEIGHTS = {"miner-a": 0.75, "miner-b": 0.25}
ZERO_ACCOUNT = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"  # what Owner() answers for a hotkey nobody owns


@dataclass
class NeuronCommitment:
    """The fields and properties of bittensor 11.1.0's metagraph.NeuronCommitment that the validator reads."""

    hotkey: str
    uid: int | None
    block: int
    data: str
    encrypted: bool = False
    revealed: list[tuple[int, str]] = field(default_factory=list)

    @property
    def is_revealed(self) -> bool:
        return any(block >= self.block for block, _ in self.revealed) or not self.encrypted

    @property
    def value(self) -> str | None:
        for block, text in reversed(self.revealed):
            if block >= self.block:
                return text
        return self.data or None


class Bittensor111:
    """No Subtensor.metagraph or Subtensor.set_weights: namespaced reads and a module-level set_weights intent."""

    def __init__(self, hotkeys, commitments=None, *, subnet_exists=True):
        self.calls, self.wallets = [], []
        outer = self

        class Subtensor:
            def __init__(self, network="finney"):
                outer.network = network
                self.subnets = SimpleNamespace(
                    metagraph=lambda netuid: SimpleNamespace(hotkeys=list(hotkeys)) if subnet_exists else None,
                    commitments=lambda netuid: dict(commitments or {}),
                )

        class Wallet:
            def __init__(self, name="default", hotkey="default", path="~/.bittensor/wallets"):
                self.name, self.hotkey_name, self.path = name, hotkey, path
                outer.wallets.append(self)

        def set_weights(netuid, weights, *, uids=None, wallet=None, hotkey=None, mechid=0, version_key=0, network="finney", retries=2):
            if wallet is not None and not isinstance(wallet, str) and hotkey is not None:
                raise ValueError("hotkey= only combines with a wallet *name*")  # 11.1.0's _resolve_wallet refuses this too
            outer.calls.append(dict(netuid=netuid, weights=weights, wallet=wallet, hotkey=hotkey, mechid=mechid, network=network))
            return SimpleNamespace(success=True, message="Success")

        self.Subtensor, self.Wallet, self.set_weights = Subtensor, Wallet, set_weights


# ---------------------------------------------------------------- weights


def test_a_wallet_directory_reaches_bittensor_11_as_a_wallet_object():
    bt = Bittensor111(["owner", "miner-a", "miner-b"])
    out = set_weights(WEIGHTS, 2, "validator", "default", "local", wallet_path="/data/localnet/wallets", bt=bt)
    assert out["submitted"] is True and out["uids"] == [1, 2]
    [call] = bt.calls
    assert call["wallet"] is bt.wallets[0] and call["hotkey"] is None
    assert (bt.wallets[0].name, bt.wallets[0].hotkey_name, bt.wallets[0].path) == ("validator", "default", "/data/localnet/wallets")
    assert call["weights"] == pytest.approx({1: 0.75, 2: 0.25}) and call["network"] == "local"


def test_without_a_wallet_directory_the_wallet_name_goes_through_unchanged():
    bt = Bittensor111(["miner-a", "miner-b"])
    assert set_weights(WEIGHTS, 2, "validator", "hot", "local", bt=bt)["submitted"] is True
    assert (bt.calls[0]["wallet"], bt.calls[0]["hotkey"]) == ("validator", "hot") and not bt.wallets


def test_bittensor_10_wallets_get_the_directory_too():
    made = []

    class Legacy:
        class Subtensor:
            def __init__(self, network):
                pass

            def metagraph(self, netuid):
                return SimpleNamespace(hotkeys=["miner-a", "miner-b"])

            def set_weights(self, wallet, netuid, uids, weights, mechid=0, wait_for_inclusion=False):
                return True

        @staticmethod
        def Wallet(**kwargs):
            made.append(kwargs)
            return kwargs

    assert set_weights(WEIGHTS, 2, "v", "h", "local", wallet_path="/w", bt=Legacy)["submitted"] is True
    assert made == [{"name": "v", "hotkey": "h", "path": "/w"}]


def test_a_subnet_that_does_not_exist_is_named():
    bt = Bittensor111([], subnet_exists=False)
    with pytest.raises(RuntimeError, match="netuid 9 does not exist"):
        registered_hotkeys(bt.Subtensor(network="local"), 9)


# ---------------------------------------------------------------- commitments


def test_neuron_commitments_from_bittensor_11_1_are_read_by_their_visible_value():
    commitments = {
        "miner-c": NeuronCommitment("miner-c", 3, 1200, "", encrypted=True, revealed=[(1300, "kt1:def")]),
        "miner-a": NeuronCommitment("miner-a", 1, 900, "kt1:abc"),
        "gone": NeuronCommitment("gone", None, 800, "kt1:old"),
        "sealed": NeuronCommitment("sealed", 4, 1100, "", encrypted=True),
        "sealed-with-plaintext": NeuronCommitment("sealed-with-plaintext", 5, 1150, "kt1:early", encrypted=True),
    }
    found = read_commitments(2, "local", bt=Bittensor111([], commitments))
    assert sorted((c.hotkey, c.block, c.data, c.uid) for c in found) == [("miner-a", 900, "kt1:abc", 1), ("miner-c", 1200, "kt1:def", 3)]


# ---------------------------------------------------------------- collateral and emission over bittensor's RPC client


class FakeRpcSubstrate:
    """bittensor.RpcSubstrate: async reads that return plain decoded values, plus the raw connection."""

    instances: list[FakeRpcSubstrate] = []

    def __init__(self, url):
        self.url, self.connected, self.closed, self.queries, self.requests = url, False, False, [], []
        self.raw = SimpleNamespace(get_chain_finalised_head=self._finalized_head, rpc_request=self._rpc_request)
        FakeRpcSubstrate.instances.append(self)

    async def connect(self):
        self.connected = True

    async def _finalized_head(self):
        return "0xfinal"

    async def _rpc_request(self, method, params):
        self.requests.append((method, params))
        return "0x1e546b0300000000"  # finney's price answer for netuid 4

    async def query(self, module, name, params=None, block_hash=None):
        self.queries.append((module, name, tuple(params or []), block_hash))
        if name == "Owner":
            return {"5Hot": "5Cold"}.get(params[0], ZERO_ACCOUNT)
        if name == "MinerCollateral":
            # Decoded as the localnet returned it: a dict with the U64F64 drain ratio as {"bits": ...}.
            positions = {(7, "5Hot", "5Cold"): {"locked": 84416858261, "drain_ratio": {"bits": 2**64}, "min_locked": 0, "earned": 0}}
            return positions.get(tuple(params))
        return {
            "SubnetAlphaOutEmission": 1_000_000_000, "SubnetOwnerCut": 11796, "OwnerCutEnabled": True, "Tempo": 360,
            "MechanismCountCurrent": 1, "MechanismEmissionSplit": None, "SubnetMovingPrice": {"bits": 3 << 30}, "MinerBurned": {"bits": 0},
        }[name]

    async def close(self):
        self.closed = True


@pytest.fixture
def only_bittensor_11(monkeypatch):
    """An environment like kuno-validator[chain] with bittensor 11.1.0: neither substrate-interface package importable."""
    for name in ("substrateinterface", "async_substrate_interface", "async_substrate_interface.sync_substrate"):
        monkeypatch.setitem(sys.modules, name, None)
    module = types.ModuleType("bittensor")
    module.RpcSubstrate = FakeRpcSubstrate
    monkeypatch.setitem(sys.modules, "bittensor", module)
    FakeRpcSubstrate.instances.clear()
    return FakeRpcSubstrate.instances


def test_collateral_is_read_through_bittensors_own_client_at_one_finalized_block(only_bittensor_11):
    reader = SubstrateCollateralReader("ws://127.0.0.1:9944")
    assert reader.locked_collateral(7, ["5Hot", "5Unregistered"]) == {"5Hot": 84416858261, "5Unregistered": 0}
    [rpc] = only_bittensor_11
    assert rpc.url == "ws://127.0.0.1:9944" and rpc.connected
    assert ("SubtensorModule", "MinerCollateral", (7, "5Hot", "5Cold"), "0xfinal") in rpc.queries
    assert {query[3] for query in rpc.queries} == {"0xfinal"}
    reader.close()
    assert rpc.closed


def test_emission_and_the_pool_price_are_read_through_bittensors_own_client(only_bittensor_11):
    reading = SubstrateEmissionReader("ws://127.0.0.1:9944").read(4)
    [rpc] = only_bittensor_11
    assert rpc.requests == [("state_call", ["SwapRuntimeApi_current_alpha_price", "0x0400", "0xfinal"])]
    assert reading.tao_per_alpha == pytest.approx(0.057365534) and reading.tempo_blocks == 360
    assert reading.mechanism_share == 1.0 and reading.block == "0xfinal"


def test_without_any_chain_client_the_error_says_what_to_install(monkeypatch, only_bittensor_11):
    monkeypatch.setitem(sys.modules, "bittensor", None)
    with pytest.raises(CollateralUnavailable, match="kuno-validator\\[chain\\]"):
        _default_substrate("ws://127.0.0.1:9944")
