"""Setting weights per incentive mechanism, and reading miner commitments.

Requires the `chain` extra. KunoWorld runs two mechanisms on one netuid: 0 is serving
(`scoring.py`), 1 is the Turbo competition (`turbo.py`). Miners keep one UID across both,
so the same hotkey -> UID mapping serves both weight vectors.

Chain facts this module relies on (subtensor spec 455, read from finney metadata):
  SubtensorModule.set_mechanism_weights(netuid, mecid: u8, dests, weights, version_key)
  SubtensorModule.commit_timelocked_mechanism_weights(netuid, mecid, commit, reveal_round, commit_reveal_version)
  Commitments.set_commitment(netuid, info: CommitmentInfo{fields: [Data; <=3]}), Data::Raw0..Raw128
  Commitments.CommitmentOf(netuid, hotkey) -> Registration{deposit, block, info}
The SDK picks plain or timelocked commit-reveal submission itself, per mechanism.

Supported SDKs:
  bittensor 10.x  Subtensor.set_weights(wallet, netuid, uids, weights, mechid=0, ...) and
                  Subtensor.get_all_commitments / get_commitment_metadata
  bittensor 11.x  bt.set_weights(netuid, {uid: w}, wallet=, hotkey=, mechid=, network=),
                  Subtensor().subnets.metagraph(netuid).hotkeys and Subtensor().subnets.commitments(netuid)
                  ({hotkey: NeuronCommitment} in 11.1.0). The 11.1.0 paths were run against a spec-458 localnet
                  (scripts/localnet/).
The API has changed between major releases, so run `kuno-validator once --netuid <n> --dry-run`
against your wallet before trusting a live run: it resolves hotkeys to UIDs and prints the
vector it would submit without touching the chain. An SDK that cannot address a mechanism id
is refused for any mechanism but 0, rather than silently overwriting the serving weights.

Never route weight to the owner hotkey or a burn UID: since June 2026 burned miner
emission directly reduces the subnet's TAO emission share. When nothing qualifies we
leave the previous weights in place rather than submitting zeros.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from typing import Any

from kuno_protocol.turbo import OnChainCommitment

log = logging.getLogger("kuno.validator.chain")

SERVING_MECHID = 0
TURBO_MECHID = 1
# MechId is a u8 on chain; MaxMechanismCount is currently 2.
MAX_MECHID = 255


def resolve_uids(hotkeys: list[str], weights: dict[str, float]) -> tuple[list[int], list[float]]:
    """Maps scored hotkeys onto their UIDs, renormalized over those actually registered."""
    pairs = [(uid, weights[hotkey]) for uid, hotkey in enumerate(hotkeys) if hotkey in weights]
    total = sum(weight for _uid, weight in pairs)
    if total <= 0:
        return [], []
    return [uid for uid, _ in pairs], [weight / total for _, weight in pairs]


def registered_hotkeys(subtensor: Any, netuid: int) -> list[str]:
    """Hotkeys by UID, from a bittensor 10 metagraph object or a bittensor 11 metagraph dict."""
    metagraph = getattr(subtensor, "metagraph", None)
    if callable(metagraph):
        return [str(hotkey) for hotkey in metagraph(netuid).hotkeys]
    subnets = getattr(subtensor, "subnets", None)
    if subnets is not None and callable(getattr(subnets, "metagraph", None)):
        graph = subnets.metagraph(netuid)
        if graph is None:  # bittensor 11.1 answers None for a netuid that doesn't exist
            raise RuntimeError(f"netuid {netuid} does not exist on this chain")
        hotkeys = graph.get("hotkeys") if isinstance(graph, dict) else getattr(graph, "hotkeys", None)
        if hotkeys is not None:
            return [str(hotkey) for hotkey in hotkeys]
    raise RuntimeError("the installed bittensor exposes no metagraph this validator understands")


def _accepts(function: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def _succeeded(result: Any) -> bool:
    """bool (old SDKs), (bool, message) tuples (bittensor 8/9), or a response object with `.success`."""
    if isinstance(result, bool):
        return result
    if isinstance(result, tuple):
        return bool(result and result[0])
    success = getattr(result, "success", None)
    if success is not None:
        return bool(success)
    return bool(result)


def set_weights(
    weights: dict[str, float],
    netuid: int,
    wallet_name: str,
    hotkey_name: str,
    network: str,
    *,
    mechid: int = SERVING_MECHID,
    dry_run: bool = False,
    wallet_path: str | None = None,
    bt: Any = None,
) -> dict[str, Any]:
    """Submits one mechanism's weights. Returns what happened; `bt` is injectable for tests.

    `wallet_path` is the wallet directory; unset, the SDK's default (~/.bittensor/wallets) applies."""
    if not isinstance(mechid, int) or not 0 <= mechid <= MAX_MECHID:
        raise ValueError(f"mechanism id must be an integer in 0..{MAX_MECHID}, not {mechid!r}")
    if not weights:
        log.warning("no qualifying miners for mechanism %d this round; leaving previous weights in place", mechid)
        return {"submitted": False, "reason": "no qualifying miners"}
    if bt is None:
        import bittensor as bt  # noqa: PLC0415 — optional dependency

    subtensor = bt.Subtensor(network=network)
    hotkeys = registered_hotkeys(subtensor, netuid)
    uids, values = resolve_uids(hotkeys, weights)
    if not uids:
        log.warning("none of the %d scored hotkeys are registered on netuid %d", len(weights), netuid)
        return {"submitted": False, "reason": "no scored hotkey is registered"}

    registered = set(hotkeys)
    missing = [hotkey for hotkey in weights if hotkey not in registered]
    if missing:
        log.warning("%d scored hotkey(s) are not registered and were dropped: %s", len(missing), ", ".join(missing[:5]))
    outcome: dict[str, Any] = {"mechid": mechid, "uids": uids, "weights": values, "dropped": missing}
    if dry_run:
        log.info("dry run: would set %d weights on netuid %d mechanism %d", len(uids), netuid, mechid)
        return {"submitted": False, "dry_run": True, **outcome}

    legacy = getattr(subtensor, "set_weights", None)
    if callable(legacy):
        wallet_kwargs = {"name": wallet_name, "hotkey": hotkey_name, **({"path": wallet_path} if wallet_path else {})}
        kwargs: dict[str, Any] = {
            "wallet": bt.Wallet(**wallet_kwargs),
            "netuid": netuid,
            "uids": uids,
            "weights": values,
            "wait_for_inclusion": True,
        }
        if _accepts(legacy, "mechid"):
            kwargs["mechid"] = mechid
        elif mechid != SERVING_MECHID:
            log.error("the installed bittensor cannot target mechanism %d; nothing submitted", mechid)
            return {"submitted": False, "reason": f"the installed bittensor cannot set weights for mechanism {mechid}", **outcome}
        result = legacy(**kwargs)
    elif callable(getattr(bt, "set_weights", None)):
        # bittensor 11: one blocking call that raises ChainError on failure. A wallet *name* always resolves under
        # ~/.bittensor/wallets, so a wallet kept elsewhere goes in as a Wallet object (which carries its hotkey).
        wallet: Any = wallet_name
        hotkey: str | None = hotkey_name
        if wallet_path:
            wallet, hotkey = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path), None
        try:
            result = bt.set_weights(netuid, dict(zip(uids, values)), wallet=wallet, hotkey=hotkey, mechid=mechid, network=network)
        except Exception as exc:  # the SDK's ChainError, or a connection failure
            log.error("setting mechanism %d weights failed: %s", mechid, exc)
            return {"submitted": False, "reason": f"chain rejected the weights: {exc}", **outcome}
    else:
        return {"submitted": False, "reason": "the installed bittensor has no way to set weights", **outcome}
    log.info("set mechanism %d weights for %d miners: %s", mechid, len(uids), result)
    return {"submitted": _succeeded(result), **outcome, "result": result}


# ---------------------------------------------------------------- commitments


def decode_commitment_info(info: Any) -> str:
    """Text of a raw `CommitmentInfo` as substrate-interface decodes it, e.g.
    {"fields": [{"Raw64": "0x4573..."}]}. Hash and timelocked fields carry no plaintext and are skipped."""
    fields = info.get("fields", []) if isinstance(info, dict) else []
    parts: list[str] = []
    for item in fields:
        while isinstance(item, (list, tuple)) and len(item) == 1:
            item = item[0]
        if not isinstance(item, dict):
            continue
        for kind, value in item.items():
            if not str(kind).startswith("Raw"):
                continue
            if isinstance(value, str):
                raw = bytes.fromhex(value[2:]) if value.startswith("0x") else value.encode()
            else:
                raw = bytes(value)
            parts.append(raw.decode("utf-8", "replace"))
    return "".join(parts)


def _commitment_row(row: Any) -> tuple[Any, Any, Any, str | None]:
    """(hotkey, uid, block, visible text) of one bittensor 11 commitment.

    bittensor 11.1.0 returns `{hotkey: NeuronCommitment}`, whose `value` is the readable text and whose `is_revealed`
    is False while a timelocked payload is still sealed (verified against a spec-458 localnet). Rows that are plain
    dicts ({"hotkey", "uid", "block", "commitment"}) come from earlier 11.x builds."""
    if isinstance(row, Mapping):
        return row.get("hotkey"), row.get("uid"), row.get("block"), row.get("commitment")
    text = getattr(row, "value", None) if getattr(row, "is_revealed", True) else None
    return getattr(row, "hotkey", None), getattr(row, "uid", None), getattr(row, "block", None), text


def read_commitments(netuid: int, network: str, *, bt: Any = None) -> list[OnChainCommitment]:
    """Every registered hotkey's current plaintext commitment on the subnet, with its block."""
    if bt is None:
        import bittensor as bt  # noqa: PLC0415 — optional dependency

    subtensor = bt.Subtensor(network=network)
    subnets = getattr(subtensor, "subnets", None)
    if subnets is not None and callable(getattr(subnets, "commitments", None)):
        rows = subnets.commitments(netuid)
        found = []
        for row in rows.values() if isinstance(rows, Mapping) else rows:
            hotkey, uid, block, text = _commitment_row(row)
            # Deregistered hotkeys (uid None) and still-sealed timelocked payloads cannot compete.
            if text is None or uid is None:
                continue
            found.append(OnChainCommitment(str(hotkey), int(block), str(text), int(uid)))
        return found
    if callable(getattr(subtensor, "get_all_commitments", None)):
        uids = {hotkey: uid for uid, hotkey in enumerate(registered_hotkeys(subtensor, netuid))}
        found = []
        for hotkey, text in subtensor.get_all_commitments(netuid=netuid).items():
            if hotkey not in uids:
                continue
            metadata = subtensor.get_commitment_metadata(netuid, hotkey)
            block = metadata.get("block") if isinstance(metadata, dict) else None
            if block is None:
                log.warning("commitment of %s has no block number; ignored", hotkey)
                continue
            if not isinstance(text, str) or not text:
                text = decode_commitment_info(metadata.get("info"))
            found.append(OnChainCommitment(hotkey, int(block), text, uids[hotkey]))
        return found
    raise RuntimeError("the installed bittensor cannot read commitments")


class ChainCommitments:
    """A commitment source for `TurboTrack`, read fresh on every call."""

    def __init__(self, netuid: int, network: str, bt: Any = None):
        self.netuid, self.network, self.bt = netuid, network, bt

    def __call__(self) -> list[OnChainCommitment]:
        return read_commitments(self.netuid, self.network, bt=self.bt)
