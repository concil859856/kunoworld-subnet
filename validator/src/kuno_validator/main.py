from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path

from kuno_protocol.canonical import b64d
from kuno_protocol.policy import policy_from_env

from .capacity import CapacityTracker
from .collateral import CollateralGate
from .open_tier import TierPolicy
from .usd_pay import PayUnavailable, UsdPay
from .validator import Validator

log = logging.getLogger("kuno.validator")


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.startswith("#"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="kuno-validator")
    parser.add_argument("command", choices=["once", "run"], help="one scoring round, or loop forever")
    parser.add_argument("--interval", type=float, default=4320.0, help="seconds between rounds (default: one tempo)")
    parser.add_argument("--canary", action="append", default=[], help="profile id to send a canary job to (repeatable)")
    parser.add_argument(
        "--standard-canary", action="append", default=[],
        help="profile id to send a standard-mode canary to (repeatable); these reach open-tier miners and admit them",
    )
    parser.add_argument("--netuid", type=int, help="set weights on this subnet (requires the chain extra)")
    parser.add_argument("--wallet-name", default="default")
    parser.add_argument("--wallet-hotkey", default="default")
    parser.add_argument("--network", default="finney")
    parser.add_argument("--dry-run", action="store_true", help="resolve hotkeys to UIDs and print the vector without submitting")
    parser.add_argument(
        "--allow-unsigned-switch",
        action="store_true",
        help="submit weights even though KUNO_OWNER_PUBLIC_KEY is not set (unsafe: the gateway controls the family split)",
    )
    parser.add_argument("--no-turbo", action="store_true", help="don't run the Turbo track (mechanism 1)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    data_dir = Path(os.environ.get("KUNO_DATA_DIR", "data"))
    env = {**_read_env_file(data_dir / "dev.env"), **os.environ}
    if not env.get("KUNO_VALIDATOR_API_KEY"):
        parser.error("KUNO_VALIDATOR_API_KEY is required: the gateway authenticates every validator request")
    owner = env.get("KUNO_OWNER_PUBLIC_KEY")
    if not owner and args.netuid is not None and not args.dry_run and not args.allow_unsigned_switch:
        parser.error("refusing to set weights without KUNO_OWNER_PUBLIC_KEY (pass --allow-unsigned-switch to override)")
    # KUNO_ATTESTATION=production refuses to start without Intel DCAP and NVIDIA verifiers and an
    # owner-signed manifest, exactly as the gateway does.
    policy = policy_from_env(env)
    manifest = policy.load_manifest(env.get("KUNO_SIGNED_MANIFEST") or env["KUNO_MANIFEST"])
    state_path = Path(env.get("KUNO_VALIDATOR_STATE", data_dir / "validator-state.json"))
    # KUNO_MIN_COLLATERAL_PER_GPU (alpha) zeroes miners whose locked collateral doesn't cover their GPUs.
    collateral = CollateralGate.from_env(env, args.netuid, args.network)
    if collateral is not None and args.netuid is None:
        log.error("KUNO_MIN_COLLATERAL_PER_GPU is set but --netuid is not: collateral can't be read, so every miner fails it")
    # Open tier: KUNO_OPEN_TIER_RATE (default 0.5), KUNO_OPEN_TIER_PROBES (default 5); tolerance thresholds from
    # KUNO_TOLERANCE_CALIBRATION, else the file shipped with kuno-protocol (empty until the owner calibrates: unproven).
    from kuno_protocol.tolerance import load_calibration

    calibration = load_calibration(env.get("KUNO_TOLERANCE_CALIBRATION") or None)
    # KUNO_PAY_MODE=usd prices verified work with the owner-signed rate card (usd_pay.py); the default is VCU scoring.
    try:
        pay = UsdPay.from_env(env, args.netuid, args.network, b64d(owner) if owner else None, state_path)
    except ValueError as exc:
        parser.error(str(exc))
    if pay is not None and args.netuid is None:
        parser.error("KUNO_PAY_MODE=usd prices work against the subnet's emission, which is read from the chain: pass --netuid")
    # Capacity pay (capacity.py): a GPU's verified run breaks after KUNO_CAPACITY_MAX_GAP_S without a successful check
    # (default two round intervals).
    try:
        capacity = CapacityTracker.from_env(env, args.interval)
    except ValueError as exc:
        parser.error(str(exc))
    validator = Validator(
        env.get("KUNO_GATEWAY_URL", "http://127.0.0.1:8080"),
        env["KUNO_VALIDATOR_API_KEY"],
        manifest,
        b64d(owner) if owner else None,
        state_path=state_path,
        policy=policy,
        collateral=collateral,
        tier_policy=TierPolicy.from_env(env),
        calibration=calibration,
        pay=pay,
        capacity=capacity,
    )

    turbo = None
    if args.netuid is not None and not args.no_turbo:
        from .chain import ChainCommitments
        from .turbo import TurboTrack

        turbo = TurboTrack(
            env.get("KUNO_GATEWAY_URL", "http://127.0.0.1:8080"),
            env["KUNO_VALIDATOR_API_KEY"],
            manifest,
            b64d(owner) if owner else None,
            ChainCommitments(args.netuid, args.network),
            policy=policy,
            state_path=data_dir / "turbo-state.json",
        )
        validator.turbo = turbo
    stop = threading.Event()
    latest: dict[str, dict[str, float]] = {}

    if turbo is not None and args.command == "run":
        # Benchmarks block for up to a job timeout each, so the Turbo track keeps its own cadence.
        threading.Thread(target=_turbo_loop, args=(turbo, latest, args, stop), daemon=True).start()

    while True:
        weights = serving_round(validator, args.canary, args.standard_canary)
        if weights is not None:
            latest["serving"] = weights
            print(json.dumps(weights, indent=2))
        if weights is not None and args.netuid is not None:
            from .chain import set_weights

            outcome = set_weights(
                weights, args.netuid, args.wallet_name, args.wallet_hotkey, args.network, dry_run=args.dry_run, mechid=0
            )
            print(json.dumps(outcome, indent=2, default=str))
        if args.command == "once":
            if turbo is not None:
                _turbo_round(turbo, latest, args)
            break
        time.sleep(args.interval)
    stop.set()


def serving_round(validator: Validator, canaries: list[str], standard_canaries: list[str]) -> dict[str, float] | None:
    """One serving round's weights, or None when USD pay can't price the round: the previous weights then stay on chain."""
    try:
        return validator.step(canaries, standard_canaries)
    except PayUnavailable as exc:
        log.error("USD pay is unavailable this round (%s); leaving the previous serving weights in place", exc)
        return None


def _turbo_round(turbo, latest: dict, args) -> None:
    from .chain import set_weights

    weights = turbo.step(latest.get("serving"))
    print(json.dumps({"mechanism": turbo.mechid, "weights": weights}, indent=2))
    outcome = set_weights(
        weights, args.netuid, args.wallet_name, args.wallet_hotkey, args.network, dry_run=args.dry_run, mechid=turbo.mechid
    )
    print(json.dumps(outcome, indent=2, default=str))


def _turbo_loop(turbo, latest: dict, args, stop: threading.Event) -> None:
    while not stop.is_set():
        if turbo.due():
            try:
                _turbo_round(turbo, latest, args)
            except Exception:  # a failed round must not stop the serving track
                log.exception("Turbo round failed")
        stop.wait(30)


if __name__ == "__main__":
    main()
