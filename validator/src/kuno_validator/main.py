from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from kuno_protocol.attestation import GoldenManifest
from kuno_protocol.canonical import b64d

from .validator import Validator


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.startswith("#"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="kuno-validator")
    parser.add_argument("command", choices=["once", "run"], help="one scoring round, or loop forever")
    parser.add_argument("--interval", type=float, default=4320.0, help="seconds between rounds (default: one tempo)")
    parser.add_argument("--canary", action="append", default=[], help="profile id to send a canary job to (repeatable)")
    parser.add_argument("--netuid", type=int, help="set weights on this subnet (requires the chain extra)")
    parser.add_argument("--wallet-name", default="default")
    parser.add_argument("--wallet-hotkey", default="default")
    parser.add_argument("--network", default="finney")
    parser.add_argument("--dry-run", action="store_true", help="resolve hotkeys to UIDs and print the vector without submitting")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    env = {**_read_env_file(Path(os.environ.get("KUNO_DATA_DIR", "data")) / "dev.env"), **os.environ}
    manifest = GoldenManifest.model_validate_json(Path(env["KUNO_MANIFEST"]).read_text())
    owner = env.get("KUNO_OWNER_PUBLIC_KEY")
    validator = Validator(env.get("KUNO_GATEWAY_URL", "http://127.0.0.1:8080"), env["KUNO_VALIDATOR_API_KEY"], manifest, b64d(owner) if owner else None)

    while True:
        weights = validator.step(args.canary)
        print(json.dumps(weights, indent=2))
        if args.netuid is not None:
            from .chain import set_weights

            outcome = set_weights(
                weights, args.netuid, args.wallet_name, args.wallet_hotkey, args.network, dry_run=args.dry_run
            )
            print(json.dumps(outcome, indent=2, default=str))
        if args.command == "once":
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
