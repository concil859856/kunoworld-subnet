"""Setting weights on the Bittensor subnet.

Requires the `chain` extra (bittensor >= 11). The Bittensor Python API has changed
between major releases, so run `kuno-validator once --netuid <n> --dry-run` against your
wallet before trusting a live run: it resolves hotkeys to UIDs and prints the vector it
would submit without touching the chain.

Never route weight to the owner hotkey or a burn UID: since June 2026 burned miner
emission directly reduces the subnet's TAO emission share. When nothing qualifies we
leave the previous weights in place rather than submitting zeros.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("kuno.validator.chain")


def resolve_uids(hotkeys: list[str], weights: dict[str, float]) -> tuple[list[int], list[float]]:
    """Maps scored hotkeys onto their UIDs, renormalized over those actually registered."""
    pairs = [(uid, weights[hotkey]) for uid, hotkey in enumerate(hotkeys) if hotkey in weights]
    total = sum(weight for _uid, weight in pairs)
    if total <= 0:
        return [], []
    return [uid for uid, _ in pairs], [weight / total for _, weight in pairs]


def set_weights(
    weights: dict[str, float],
    netuid: int,
    wallet_name: str,
    hotkey_name: str,
    network: str,
    *,
    dry_run: bool = False,
    bt: Any = None,
) -> dict[str, Any]:
    """Returns what happened; `bt` is injectable so the mapping can be tested without a chain."""
    if not weights:
        log.warning("no qualifying miners this round; leaving previous weights in place")
        return {"submitted": False, "reason": "no qualifying miners"}
    if bt is None:
        import bittensor as bt  # noqa: PLC0415 — optional dependency

    subtensor = bt.Subtensor(network=network)
    metagraph = subtensor.metagraph(netuid)
    uids, values = resolve_uids(list(metagraph.hotkeys), weights)
    if not uids:
        log.warning("none of the %d scored hotkeys are registered on netuid %d", len(weights), netuid)
        return {"submitted": False, "reason": "no scored hotkey is registered"}

    missing = [hotkey for hotkey in weights if hotkey not in set(metagraph.hotkeys)]
    if missing:
        log.warning("%d scored hotkey(s) are not registered and were dropped: %s", len(missing), ", ".join(missing[:5]))
    if dry_run:
        log.info("dry run: would set %d weights on netuid %d", len(uids), netuid)
        return {"submitted": False, "dry_run": True, "uids": uids, "weights": values, "dropped": missing}

    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    result = subtensor.set_weights(wallet=wallet, netuid=netuid, uids=uids, weights=values, wait_for_inclusion=True)
    log.info("set weights for %d miners: %s", len(uids), result)
    return {"submitted": bool(result), "uids": uids, "weights": values, "dropped": missing, "result": result}
