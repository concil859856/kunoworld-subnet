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

The requirement is `KUNO_MIN_COLLATERAL_PER_GPU` alpha per attested GPU, and
`KUNO_MIN_COLLATERAL_PER_GPU_OPEN` alpha per open-tier GPU (default: twice the confidential amount;
it may not be lower). Open-tier GPUs are not attested, so they are counted as the larger of what the
enclave reports and what its capacity needs (open_tier.open_tier_gpus), and collateral carries the
weight attestation carries for the confidential tier. It is set in alpha,
not TAO, because that is what the chain locks. A TAO-equivalent requirement would need the
pool's spot price, which anyone can move inside a block, so miners could be pushed under the
line on purpose. The owner revisits the number when alpha's price moves a lot.

Failing closed: if a chain read fails, the last successful reading of each hotkey is used
for up to `KUNO_COLLATERAL_MAX_STALE_S` seconds (default two tempos). After that, and for a
hotkey never read, the hotkey gets zero weight.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

log = logging.getLogger("kuno.validator.collateral")

RAO_PER_ALPHA = 10**9
DEFAULT_MAX_STALE_S = 2 * 360 * 12.0  # two default tempos of 360 blocks at 12 s
# Open-tier collateral per GPU relative to the confidential requirement, when not set explicitly.
DEFAULT_OPEN_MULTIPLIER = 2
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


class BittensorRpcSubstrate:
    """The blocking substrate-interface calls this package makes, over bittensor >= 11's own RPC client.

    bittensor 11 dropped async-substrate-interface, so this is what `kuno-validator[chain]` can read the chain with.
    Every call blocks on a private event loop: use it from synchronous code, not inside a running loop.
    """

    def __init__(self, url: str, rpc_class: Any):
        self._loop = asyncio.new_event_loop()
        self._rpc = rpc_class(url)
        try:
            self._run(self._rpc.connect())
        except BaseException:
            self._loop.close()
            raise

    def _run(self, coroutine: Any) -> Any:
        return self._loop.run_until_complete(coroutine)

    def get_chain_finalised_head(self) -> str:
        return self._run(self._rpc.raw.get_chain_finalised_head())

    def query(self, module: str, storage_function: str, params: list | None = None, block_hash: str | None = None) -> Any:
        return self._run(self._rpc.query(module, storage_function, params or [], block_hash=block_hash))

    def rpc_request(self, method: str, params: list | None = None) -> dict:
        """Wrapped as {"result": ...}, the shape substrate-interface returns."""
        return {"result": self._run(self._rpc.raw.rpc_request(method, params or []))}

    def close(self) -> None:
        if self._loop.is_closed():
            return
        try:
            self._run(self._rpc.close())
        finally:
            self._loop.close()


def _default_substrate(url: str) -> Any:
    try:
        from substrateinterface import SubstrateInterface  # substrate-interface
    except ImportError:
        try:
            from async_substrate_interface.sync_substrate import SubstrateInterface  # what bittensor 9 and 10 ship
        except ImportError:
            try:
                from bittensor import RpcSubstrate  # bittensor >= 11 (kuno-validator[chain]) ships its own client
            except ImportError:
                raise CollateralUnavailable(
                    "reading miner collateral needs substrate-interface or kuno-validator[chain]"
                ) from None
            return BittensorRpcSubstrate(url, RpcSubstrate)
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
        min_per_gpu_open: int | None = None,
    ):
        if min_per_gpu < 0 or (min_per_gpu_open or 0) < 0:
            raise ValueError("the collateral requirement cannot be negative")
        open_minimum = DEFAULT_OPEN_MULTIPLIER * min_per_gpu if min_per_gpu_open is None else min_per_gpu_open
        if open_minimum < min_per_gpu:
            raise ValueError("KUNO_MIN_COLLATERAL_PER_GPU_OPEN cannot be below KUNO_MIN_COLLATERAL_PER_GPU: open-tier GPUs are not attested")
        self.reader, self.netuid = reader, netuid
        self.min_per_gpu = min_per_gpu
        self.min_per_gpu_open = open_minimum
        self.max_stale_s = max_stale_s
        # hotkey -> (locked alpha base units, when it was read)
        self.view: dict[str, tuple[int, float]] = {}

    @property
    def enabled(self) -> bool:
        return self.min_per_gpu > 0 or self.min_per_gpu_open > 0

    @classmethod
    def from_env(cls, env: Mapping[str, str], netuid: int | None, network: str = "finney") -> CollateralGate | None:
        """KUNO_MIN_COLLATERAL_PER_GPU (alpha; unset or 0 disables), KUNO_MIN_COLLATERAL_PER_GPU_OPEN (alpha per
        open-tier GPU; default twice the former), KUNO_COLLATERAL_MAX_STALE_S, KUNO_CHAIN_ENDPOINT (defaults to the
        --network's public endpoint). Both unset or 0 disables the gate."""
        text = (env.get("KUNO_MIN_COLLATERAL_PER_GPU") or "").strip()
        minimum = parse_alpha(text) if text else 0
        open_text = (env.get("KUNO_MIN_COLLATERAL_PER_GPU_OPEN") or "").strip()
        open_minimum = parse_alpha(open_text) if open_text else None
        if minimum == 0 and not open_minimum:
            return None
        reader = None
        if netuid is not None:
            reader = SubstrateCollateralReader(env.get("KUNO_CHAIN_ENDPOINT") or NETWORK_ENDPOINTS.get(network, network))
        return cls(reader, netuid, minimum, float(env.get("KUNO_COLLATERAL_MAX_STALE_S") or DEFAULT_MAX_STALE_S), open_minimum)

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

    def penalties(self, gpus: Mapping[str, int], now: float, open_gpus: Mapping[str, int] | None = None) -> dict[str, list[str]]:
        """`gpus`: attested (confidential-tier) GPUs per hotkey this round; `open_gpus`: open-tier GPUs per hotkey."""
        open_gpus = dict(open_gpus or {})
        if not self.enabled or not (gpus or open_gpus):
            return {}
        hotkeys = sorted(set(gpus) | set(open_gpus))
        error = self.refresh(hotkeys, now)
        penalties: dict[str, list[str]] = {}
        for hotkey in hotkeys:
            count, open_count = gpus.get(hotkey, 0), open_gpus.get(hotkey, 0)
            required = (self.min_per_gpu * max(count, 1) if hotkey in gpus else 0) + self.min_per_gpu_open * open_count
            reading = self.view.get(hotkey)
            if reading is None or now - reading[1] > self.max_stale_s:
                why = f"chain read failed: {error}" if error else "never read"
                penalties[hotkey] = [f"collateral unknown ({why}); {format_alpha(required)} alpha required"]
                continue
            locked = reading[0]
            if locked < required:
                if not open_count:
                    basis = f"for {count} attested GPU(s) ({format_alpha(self.min_per_gpu)} per GPU)"
                else:
                    basis = f"for {open_count} open-tier GPU(s) ({format_alpha(self.min_per_gpu_open)} per GPU)"
                    if hotkey in gpus:
                        basis += f" and {count} attested GPU(s) ({format_alpha(self.min_per_gpu)} per GPU)"
                penalties[hotkey] = [f"collateral {format_alpha(locked)} alpha is below the {format_alpha(required)} alpha required {basis}"]
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
