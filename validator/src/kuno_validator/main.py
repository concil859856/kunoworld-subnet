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
from .plan_canaries import load_briefs
from .usd_pay import PayUnavailable, UsdPay
from .validator import DEFAULT_DIVERGENCE_WARNING, DEFAULT_SPOT_CHECK_RATE, ROLES, Validator

log = logging.getLogger("kuno.validator")


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.startswith("#"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="kuno-validator")
    parser.add_argument("command", choices=["once", "run"], help="one scoring round, or loop forever")
    parser.add_argument(
        "--role", choices=ROLES, default=os.environ.get("KUNO_VALIDATOR_ROLE", "auditor"),
        help="auditor (default, $KUNO_VALIDATOR_ROLE): verify published evidence and receipts, apply the main validator's "
        "signed findings, send no jobs. main: KunoWorld's own validator, which challenges, sends canaries and audits",
    )
    parser.add_argument(
        "--main-validator-hotkey", default=os.environ.get("KUNO_MAIN_VALIDATOR_HOTKEY"),
        help="auditors: the main validator's hotkey (ss58), whose signed findings they apply ($KUNO_MAIN_VALIDATOR_HOTKEY)",
    )
    parser.add_argument(
        "--spot-check-rate", type=float, default=float(os.environ.get("KUNO_SPOT_CHECK_RATE", DEFAULT_SPOT_CHECK_RATE)),
        help="auditors: share of active enclaves challenged with this validator's own nonce each round (default 0.1)",
    )
    parser.add_argument("--interval", type=float, default=4320.0, help="seconds between rounds (default: one tempo)")
    parser.add_argument("--canary", action="append", default=[], help="profile id to send a canary job to (repeatable)")
    parser.add_argument(
        "--standard-canary", action="append", default=[],
        help="profile id to send a standard-mode canary to (repeatable); these reach open-tier miners and admit them",
    )
    parser.add_argument(
        "--plan-canary", action="append", default=[],
        help="profile id to send a private plan canary to (repeatable); briefs from $KUNO_PLAN_CANARY_BRIEFS, else the fallback set",
    )
    parser.add_argument(
        "--standard-plan-canary", action="append", default=[], help="profile id to send a standard-mode plan canary to (repeatable)",
    )
    parser.add_argument("--netuid", type=int, help="set weights on this subnet (requires the chain extra)")
    parser.add_argument("--wallet-name", default="default")
    parser.add_argument("--wallet-hotkey", default="default")
    parser.add_argument(
        "--wallet-path",
        default=os.environ.get("BT_WALLET_PATH") or None,
        help="wallet directory (default: $BT_WALLET_PATH, else the SDK's ~/.bittensor/wallets)",
    )
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
    if args.role == "auditor" and (args.canary or args.standard_canary or args.plan_canary or args.standard_plan_canary):
        parser.error("auditor validators send no canaries: drop the canary options, or run with --role main")
    if args.role == "auditor" and not args.main_validator_hotkey and args.netuid is not None and not args.dry_run:
        parser.error("an auditor needs --main-validator-hotkey (or KUNO_MAIN_VALIDATOR_HOTKEY) to apply its findings")
    policy = policy_from_env(env)
    manifest = policy.load_manifest(env.get("KUNO_SIGNED_MANIFEST") or env["KUNO_MANIFEST"])
    state_path = Path(env.get("KUNO_VALIDATOR_STATE", data_dir / "validator-state.json"))
    # KUNO_MIN_COLLATERAL_PER_GPU (alpha) zeroes miners whose locked collateral doesn't cover their GPUs.
    collateral = CollateralGate.from_env(env, args.netuid, args.network)
    if collateral is not None and args.netuid is None:
        log.error("KUNO_MIN_COLLATERAL_PER_GPU is set but --netuid is not: collateral can't be read, so every miner fails it")
    # Open tier: KUNO_OPEN_TIER_RATE (default 0.75), KUNO_OPEN_TIER_PROBES (default 5); tolerance thresholds from
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
        role=args.role,
        findings_signer=_findings_signer(env, args) if args.role == "main" else None,
        main_validator_hotkey=args.main_validator_hotkey,
        spot_check_rate=args.spot_check_rate,
        divergence_warning=float(env.get("KUNO_DIVERGENCE_WARNING", DEFAULT_DIVERGENCE_WARNING)),
        require_location_proof=env.get("KUNO_REQUIRE_LOCATION_PROOF", "0") == "1",
        plan_briefs=load_briefs(env["KUNO_PLAN_CANARY_BRIEFS"]) if env.get("KUNO_PLAN_CANARY_BRIEFS") else None,
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
        weights = serving_round(validator, args.canary, args.standard_canary, args.plan_canary, args.standard_plan_canary)
        if weights is not None:
            latest["serving"] = weights
            print(json.dumps(weights, indent=2))
            if validator.role == "auditor":
                print(json.dumps({"divergence_from_main_validator": validator.last_divergence}))
        if weights is not None and args.netuid is not None:
            from .chain import set_weights

            outcome = set_weights(
                weights, args.netuid, args.wallet_name, args.wallet_hotkey, args.network,
                dry_run=args.dry_run, mechid=0, wallet_path=args.wallet_path,
            )
            print(json.dumps(outcome, indent=2, default=str))
        if args.command == "once":
            if turbo is not None:
                _turbo_round(turbo, latest, args)
            break
        time.sleep(args.interval)
    stop.set()


def _findings_signer(env: dict[str, str], args):
    """The main validator's hotkey, which signs its findings: a raw seed for development, else the wallet's hotkey."""
    seed = env.get("KUNO_VALIDATOR_HOTKEY_SEED")
    if seed:
        from kuno_protocol.hotkey import Sr25519Signer

        text = Path(seed).read_text().strip() if Path(seed).is_file() else seed.strip()
        return Sr25519Signer.from_seed(bytes.fromhex(text.removeprefix("0x")))
    try:
        from bittensor_wallet import Wallet  # noqa: PLC0415 — the chain extra
    except ImportError:
        log.error("no KUNO_VALIDATOR_HOTKEY_SEED and no bittensor wallet: this main validator can't sign its findings")
        return None
    kwargs = {"name": args.wallet_name, "hotkey": args.wallet_hotkey, **({"path": args.wallet_path} if args.wallet_path else {})}
    return Wallet(**kwargs).hotkey


def serving_round(
    validator: Validator, canaries: list[str], standard_canaries: list[str], plan_canaries: list[str] | None = None,
    standard_plan_canaries: list[str] | None = None,
) -> dict[str, float] | None:
    """One serving round's weights, or None when USD pay can't price the round: the previous weights then stay on chain."""
    try:
        return validator.step(
            canaries, standard_canaries, plan_canary_profiles=plan_canaries, standard_plan_canary_profiles=standard_plan_canaries,
        )
    except PayUnavailable as exc:
        log.error("USD pay is unavailable this round (%s); leaving the previous serving weights in place", exc)
        return None


def _turbo_round(turbo, latest: dict, args) -> None:
    from .chain import set_weights

    weights = turbo.step(latest.get("serving"))
    print(json.dumps({"mechanism": turbo.mechid, "weights": weights}, indent=2))
    outcome = set_weights(
        weights, args.netuid, args.wallet_name, args.wallet_hotkey, args.network,
        dry_run=args.dry_run, mechid=turbo.mechid, wallet_path=args.wallet_path,
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
