"""What the serving mechanism's miners are paid per tempo, and what alpha is worth, read from the chain.

Used only by USD-denominated pay (usd_pay.py). Requires substrate-interface or the `chain` extra.

Read at the finalized head (storage names and types checked against finney runtime metadata at
spec_version 455; the price call was answered by finney for netuid 4):

    SubtensorModule.SubnetAlphaOutEmission(netuid) -> u64      alpha minted per block for participants, in rao
    SubtensorModule.SubnetOwnerCut -> u16                      owner share of that, /65535 (11796 = 18%)
    SubtensorModule.OwnerCutEnabled(netuid) -> bool
    SubtensorModule.Tempo(netuid) -> u16                       an epoch runs every tempo + 1 blocks
    SubtensorModule.MechanismCountCurrent(netuid) -> u8
    SubtensorModule.MechanismEmissionSplit(netuid) -> Option<Vec<u16>>   unset: an even split
    SubtensorModule.SubnetMovingPrice(netuid) -> I96F32        reported only (starts near zero on a new subnet)
    SubtensorModule.MinerBurned(netuid) -> U64F64              reported only
    state_call SwapRuntimeApi_current_alpha_price(netuid: u16 LE) -> u64 LE   TAO per alpha, in rao

After the owner cut, miners and validators split what remains evenly (research_bittensor.md
§2.1: 18% owner, 41% miners, 41% validators), and each mechanism's miners get its share of the
split.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .chain import SERVING_MECHID
from .collateral import _default_substrate

RAO = 10**9
BLOCK_SECONDS = 12.0
U16_MAX = 65535


class EmissionUnavailable(RuntimeError):
    """The chain could not be read, or said something this validator can't use."""


@dataclass(frozen=True)
class SubnetEmission:
    netuid: int
    block: str | None
    alpha_out_per_block: float  # alpha, for all participants
    owner_cut: float  # 0..1 of alpha_out actually cut for the owner
    tempo_blocks: int
    mechanism_share: float  # this mechanism's share of the subnet's miner (and validator) emission
    tao_per_alpha: float
    moving_tao_per_alpha: float | None = None
    miner_burned: float | None = None
    block_seconds: float = BLOCK_SECONDS

    @property
    def epoch_blocks(self) -> int:
        return self.tempo_blocks + 1

    @property
    def tempo_seconds(self) -> float:
        return self.epoch_blocks * self.block_seconds

    @property
    def miner_alpha_per_tempo(self) -> float:
        """Alpha this mechanism's miners share per epoch."""
        return self.alpha_out_per_block * self.epoch_blocks * (1.0 - self.owner_cut) * 0.5 * self.mechanism_share

    def emission_alpha(self, seconds: float) -> float:
        """Alpha minted for all participants (owner, validators, miners; every mechanism) over `seconds`."""
        return self.alpha_out_per_block * seconds / self.block_seconds


class EmissionReader(Protocol):
    def read(self, netuid: int, mechid: int = SERVING_MECHID) -> SubnetEmission: ...


def mechanism_share(split: Sequence[int] | None, count: int, mechid: int) -> float:
    """A mechanism's share of emission: its entry of the owner's split, or an even split when none is set."""
    count = max(int(count or 1), 1)
    if not 0 <= mechid < count:
        return 0.0
    values = [int(v) for v in (split or [])][:count]
    if len(values) < count or sum(values) <= 0:
        return 1.0 / count
    return values[mechid] / sum(values)


def _number(value: Any) -> int:
    """A decoded integer, newtype wrapper ({"None": n} or a one-item dict) or fixed-point {"bits": n}."""
    value = getattr(value, "value", value)
    if isinstance(value, dict):
        if "bits" in value:
            return int(value["bits"])
        if len(value) == 1:
            return _number(next(iter(value.values())))
    if value is None:
        raise EmissionUnavailable("the chain returned no value")
    return int(value)


class SubstrateEmissionReader:
    def __init__(self, url: str, connect: Callable[[str], Any] = _default_substrate):
        self.url = url
        self._connect = connect
        self._substrate: Any = None

    def close(self) -> None:
        substrate, self._substrate = self._substrate, None
        if substrate is not None:
            try:
                substrate.close()
            except Exception:  # a broken socket has nothing left to close
                pass

    def read(self, netuid: int, mechid: int = SERVING_MECHID) -> SubnetEmission:
        if self._substrate is None:
            self._substrate = self._connect(self.url)
        substrate = self._substrate
        try:
            block = substrate.get_chain_finalised_head()

            def query(name: str, params: list | None = None) -> Any:
                result = substrate.query("SubtensorModule", name, params or [], block_hash=block)
                return getattr(result, "value", result)

            alpha_out = _number(query("SubnetAlphaOutEmission", [netuid]))
            cut = _number(query("SubnetOwnerCut")) / U16_MAX if query("OwnerCutEnabled", [netuid]) else 0.0
            tempo = _number(query("Tempo", [netuid]))
            count = _number(query("MechanismCountCurrent", [netuid]) or 1)
            split = query("MechanismEmissionSplit", [netuid])
            moving = query("SubnetMovingPrice", [netuid])
            burned = query("MinerBurned", [netuid])
            data = "0x" + netuid.to_bytes(2, "little").hex()
            answer = substrate.rpc_request("state_call", ["SwapRuntimeApi_current_alpha_price", data, block])
            price_hex = (answer or {}).get("result") if isinstance(answer, dict) else answer
            if not isinstance(price_hex, str) or not price_hex.startswith("0x"):
                raise EmissionUnavailable("SwapRuntimeApi_current_alpha_price gave no answer")
            price_rao = int.from_bytes(bytes.fromhex(price_hex[2:]), "little")
        except EmissionUnavailable:
            self.close()
            raise
        except Exception as exc:
            self.close()  # reconnect next round
            raise EmissionUnavailable(f"reading netuid {netuid} emission failed: {type(exc).__name__}: {exc}") from exc
        return SubnetEmission(
            netuid=netuid,
            block=str(block),
            alpha_out_per_block=alpha_out / RAO,
            owner_cut=cut,
            tempo_blocks=tempo,
            mechanism_share=mechanism_share(split if isinstance(split, (list, tuple)) else None, count, mechid),
            tao_per_alpha=price_rao / RAO,
            moving_tao_per_alpha=_number(moving) / 2**32 if moving is not None else None,
            miner_burned=_number(burned) / 2**64 if burned is not None else None,
        )
