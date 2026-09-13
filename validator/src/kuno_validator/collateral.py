"""Miner collateral: real stake at risk behind every attested GPU.

Bittensor's registration collateral (Subtensor runtime >= 437) locks part of each registration
price as alpha on the miner's hotkey. The lock is released only as the miner earns, survives
deregistration, and has no withdrawal path, so a miner caught cheating loses the
unreleased part of the lock. There is no on-chain slash: validators enforce it by giving
under-collateralized hotkeys zero weight.

What is read (verified against finney metadata at spec_version 455, and against bittensor
11.1.0's own `reads/collateral.py`):

    SubtensorModule.Owner(hotkey) -> coldkey
    SubtensorModule.MinerCollateral(netuid, hotkey, coldkey)
        -> MinerCollateralState { locked, drain_ratio: U64F64, min_locked, earned } | None

`locked` is in alpha base units (1 alpha = 1e9). Collateral is keyed per (hotkey, coldkey)
position; only the owning coldkey can call `add_collateral`, so the owner's position is the one
that counts.

The requirement is `KUNO_MIN_COLLATERAL_PER_GPU` alpha per attested GPU. It is set in alpha,
not TAO, because that is what the chain locks. A TAO-equivalent requirement would need the
pool's spot price, which anyone can move inside a block, so miners could be pushed under the
line on purpose. The owner revisits the number when alpha's price moves a lot.

Failing closed: if a chain read fails, the last successful reading of each hotkey is used
for up to `KUNO_COLLATERAL_MAX_STALE_S` seconds (default two tempos). After that, and for a
hotkey never read, the hotkey gets zero weight.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

log = logging.getLogger("kuno.validator.collateral")

RAO_PER_ALPHA = 10**9
DEFAULT_MAX_STALE_S = 2 * 360 * 12.0  # two default tempos of 360 blocks at 12 s
NETWORK_ENDPOINTS = {
    "finney": "wss://entrypoint-finney.opentensor.ai:443",
    "archive": "wss://archive.chain.opentensor.ai:443",
    "test": "wss://test.finney.opentensor.ai:443",
    "local": "ws://127.0.0.1:9944",
}


class CollateralUnavailable(RuntimeError):
    """No library to read the chain with."""


class CollateralReader(Protocol):
    def locked_collateral(self, netuid: int, hotkeys: Sequence[str]) -> dict[str, int]:
        """Locked registration collateral of each hotkey's owner position, in alpha base units.
        Hotkeys without a position map to 0. Raises if any part of the read fails."""


def parse_alpha(text: str | int | float) -> int:
    """'12.5' alpha -> base units. Negative or malformed amounts are refused."""
    try:
        amount = Decimal(str(text).strip())
    except InvalidOperation:
        raise ValueError(f"not an alpha amount: {text!r}") from None
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"not an alpha amount: {text!r}")
    return int(amount * RAO_PER_ALPHA)


def format_alpha(amount: int) -> str:
    text = f"{Decimal(amount) / RAO_PER_ALPHA:f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _value(result: Any) -> Any:
    return getattr(result, "value", result)


def _default_substrate(url: str) -> Any:
    try:
        from substrateinterface import SubstrateInterface  # substrate-interface
    except ImportError:
        try:
            from async_substrate_interface.sync_substrate import SubstrateInterface  # what bittensor >= 9 ships
        except ImportError:
            raise CollateralUnavailable(
                "reading miner collateral needs substrate-interface or kuno-validator[chain]"
            ) from None
    return SubstrateInterface(url=url, ss58_format=42)


class SubstrateCollateralReader:
    """Reads `MinerCollateral` at the finalized head, so every hotkey is judged at one block."""

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

    def locked_collateral(self, netuid: int, hotkeys: Sequence[str]) -> dict[str, int]:
        if self._substrate is None:
            self._substrate = self._connect(self.url)
        substrate = self._substrate
        try:
            block = substrate.get_chain_finalised_head()
            locked: dict[str, int] = {}
            for hotkey in hotkeys:
                owner = _value(substrate.query("SubtensorModule", "Owner", [hotkey], block_hash=block))
                state = (
                    _value(substrate.query("SubtensorModule", "MinerCollateral", [netuid, hotkey, owner], block_hash=block))
                    if owner
                    else None
                )
                locked[hotkey] = int(state["locked"]) if isinstance(state, Mapping) and state.get("locked") is not None else 0
            return locked
        except Exception:
            self.close()  # reconnect next round
            raise


class CollateralGate:
    """Zero weight for hotkeys whose locked collateral doesn't cover their attested GPUs."""

    def __init__(
        self,
        reader: CollateralReader | None,
        netuid: int | None,
        min_per_gpu: int,
        max_stale_s: float = DEFAULT_MAX_STALE_S,
    ):
        if min_per_gpu < 0:
            raise ValueError("the collateral requirement cannot be negative")
        self.reader, self.netuid = reader, netuid
        self.min_per_gpu = min_per_gpu
        self.max_stale_s = max_stale_s
        # hotkey -> (locked alpha base units, when it was read)
        self.view: dict[str, tuple[int, float]] = {}

    @property
    def enabled(self) -> bool:
        return self.min_per_gpu > 0

    @classmethod
    def from_env(cls, env: Mapping[str, str], netuid: int | None, network: str = "finney") -> CollateralGate | None:
        """KUNO_MIN_COLLATERAL_PER_GPU (alpha; unset or 0 disables), KUNO_COLLATERAL_MAX_STALE_S,
        KUNO_CHAIN_ENDPOINT (defaults to the --network's public endpoint)."""
        text = (env.get("KUNO_MIN_COLLATERAL_PER_GPU") or "").strip()
        minimum = parse_alpha(text) if text else 0
        if minimum == 0:
            return None
        reader = None
        if netuid is not None:
            reader = SubstrateCollateralReader(env.get("KUNO_CHAIN_ENDPOINT") or NETWORK_ENDPOINTS.get(network, network))
        return cls(reader, netuid, minimum, float(env.get("KUNO_COLLATERAL_MAX_STALE_S") or DEFAULT_MAX_STALE_S))

    def refresh(self, hotkeys: Sequence[str], now: float) -> str | None:
        """Reads every hotkey at once; returns why it couldn't, keeping the previous view."""
        if not hotkeys:
            return None
        if self.reader is None or self.netuid is None:
            return "no chain is configured to read collateral from (run with --netuid)"
        try:
            amounts = self.reader.locked_collateral(self.netuid, sorted(hotkeys))
        except Exception as exc:  # network, decoding, missing library: all mean "unknown this round"
            log.error("reading miner collateral on netuid %s failed: %s: %s", self.netuid, type(exc).__name__, exc)
            return f"{type(exc).__name__}: {exc}"
        for hotkey in hotkeys:
            self.view[hotkey] = (max(int(amounts.get(hotkey, 0)), 0), now)
        return None

    def penalties(self, gpus: Mapping[str, int], now: float) -> dict[str, list[str]]:
        """`gpus`: attested GPUs per hotkey this round."""
        if not self.enabled or not gpus:
            return {}
        error = self.refresh(list(gpus), now)
        penalties: dict[str, list[str]] = {}
        for hotkey, count in sorted(gpus.items()):
            required = self.min_per_gpu * max(count, 1)
            reading = self.view.get(hotkey)
            if reading is None or now - reading[1] > self.max_stale_s:
                why = f"chain read failed: {error}" if error else "never read"
                penalties[hotkey] = [f"collateral unknown ({why}); {format_alpha(required)} alpha required"]
                continue
            locked = reading[0]
            if locked < required:
                penalties[hotkey] = [
                    f"collateral {format_alpha(locked)} alpha is below the {format_alpha(required)} alpha required "
                    f"for {count} attested GPU(s) ({format_alpha(self.min_per_gpu)} per GPU)"
                ]
        for hotkey in [h for h, (_, at) in self.view.items() if now - at > self.max_stale_s]:
            del self.view[hotkey]
        return penalties

    def dump(self) -> dict:
        return {"netuid": self.netuid, "view": {hotkey: [amount, at] for hotkey, (amount, at) in self.view.items()}}

    def load(self, data: Mapping) -> None:
        """Restores readings so a restart doesn't zero everyone, but only for the same subnet."""
        if data.get("netuid") != self.netuid:
            return
        for hotkey, item in (data.get("view") or {}).items():
            try:
                self.view[hotkey] = (int(item[0]), float(item[1]))
            except (TypeError, ValueError, IndexError):
                continue
