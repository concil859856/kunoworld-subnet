"""Setting weights on the Bittensor subnet.

Requires the `chain` extra (bittensor ≥ 11). Check these calls against the
installed SDK version before mainnet; the bittensor Python API has changed
between major releases.

Never route weight to the owner hotkey or a burn UID: since June 2026 burned
miner emission directly reduces the subnet's TAO emission share. When no miner
qualifies, keep the previous weights instead.
"""

from __future__ import annotations

import logging

log = logging.getLogger("kuno.validator.chain")


def set_weights(weights: dict[str, float], netuid: int, wallet_name: str, hotkey_name: str, network: str) -> bool:
    if not weights:
        log.warning("no qualifying miners this round; leaving previous weights in place")
        return False
    import bittensor as bt

    subtensor = bt.Subtensor(network=network)
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    metagraph = subtensor.metagraph(netuid)
    uids, values = [], []
    for uid, hotkey in enumerate(metagraph.hotkeys):
        if hotkey in weights:
            uids.append(uid)
            values.append(weights[hotkey])
    if not uids:
        log.warning("none of the scored hotkeys are registered on netuid %d", netuid)
        return False
    result = subtensor.set_weights(wallet=wallet, netuid=netuid, uids=uids, weights=values, wait_for_inclusion=True)
    log.info("set weights for %d miners: %s", len(uids), result)
    return bool(result)
