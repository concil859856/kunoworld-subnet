from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time

from kuno_protocol.attestation import MockTEE, OpenTEE, TdxTEE
from kuno_protocol.canonical import b64d
from kuno_protocol.crypto import signing_key_from_bytes

from .attestation import NvmlCcSettings, build_gpu_collector, build_switch_collector
from .backends import build_backends
from .config import WorkerConfig
from .hotkey import HotkeyConfigError, load_hotkey
from .worker import Worker

log = logging.getLogger("kuno.worker")

# On shutdown, an in-flight job gets this long to finish before the process exits.
SHUTDOWN_GRACE_S = 120.0


def build_tee(config: WorkerConfig):
    if config.tee == "tdx":
        try:
            # The GPU mode goes into every GPU evidence bundle; Protected PCIe adds the VM's NVSwitch evidence.
            return TdxTEE(
                gpu_collector=build_gpu_collector(config.gpu_evidence, config.nvattest_bin),
                switch_collector=build_switch_collector(config.nvattest_bin),
                cc_settings=NvmlCcSettings(),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    if config.tee == "mock":
        if config.mock_quote_key_file is None:
            raise SystemExit("KUNO_MOCK_QUOTE_KEY_FILE is required for the mock TEE (run `kuno-devkit init`).")
        key = signing_key_from_bytes(b64d(config.mock_quote_key_file.read_text().strip()))
        return MockTEE(key, config.image_digest)
    if config.tee == "open":
        return OpenTEE()
    raise SystemExit(f"unknown TEE {config.tee!r}; use 'tdx', 'open' or 'mock'")


def check_hotkey(config: WorkerConfig, hotkey) -> None:
    """An open-tier worker has no quote, so its hotkey proof is the only binding of its keys to a miner: no hotkey, no start."""
    if config.tee == "open" and hotkey is None:
        raise SystemExit(
            "KUNO_TEE=open needs the miner's hotkey secret (KUNO_HOTKEY_SEED_FILE or KUNO_WALLET_NAME): "
            "every open-tier registration must carry a hotkey proof"
        )
    if hotkey is None and config.tee == "tdx":
        log.warning(
            "no hotkey secret configured (KUNO_HOTKEY_SEED_FILE or KUNO_WALLET_NAME): "
            "production gateways refuse registrations without a hotkey proof"
        )


def check_safety(config: WorkerConfig, gate) -> None:
    """A production (TDX) worker refuses to start without a working prompt classifier and frame classifier.

    Private content is judged only inside the enclave, so these checks are the whole safety story there.
    Mock (dev) and open-tier workers may start without classifiers, but every worker still enforces the
    shared content policy (all sexual content banned); for standard jobs the gateway runs that policy too.
    """
    if config.tee != "tdx":
        return
    from .safety import SafetyConfigError

    errors = gate.startup_errors(required=True)
    if errors:
        raise SafetyConfigError("a TDX worker requires content safety classifiers: " + "; ".join(errors))
    gate.require_classifier = True  # and fail closed on every job if one stops answering


def main() -> None:
    parser = argparse.ArgumentParser(prog="kuno-worker", description="KunoWorld miner worker (runs inside the confidential VM)")
    parser.add_argument("--profiles", help="comma-separated profile ids (default: KUNO_PROFILES)")
    parser.add_argument("--backend", choices=["mock", "real", "cold"], help="default: KUNO_BACKEND or mock")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = WorkerConfig.from_env()
    if args.profiles:
        config.profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    if args.backend:
        config.backend = args.backend
    try:
        hotkey = load_hotkey(config)
    except HotkeyConfigError as exc:
        raise SystemExit(f"hotkey: {exc}") from None
    check_hotkey(config, hotkey)
    try:
        from .safety import default_gate

        check_safety(config, default_gate())  # load any configured classifier now, not on the first job
        worker = Worker(config, build_tee(config), build_backends(config.backend, config), hotkey=hotkey)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    stop = threading.Event()

    def request_stop(*_):
        if stop.is_set():
            os._exit(1)  # second signal: leave immediately
        log.info("stopping (send the signal again to exit immediately)")
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    # The loop runs in a daemon thread so a blocking long-poll cannot delay shutdown.
    loop = threading.Thread(target=worker.run, args=(stop,), daemon=True, name="kuno-worker-loop")
    loop.start()
    while not stop.wait(1.0):
        if not loop.is_alive():
            raise SystemExit("worker loop exited unexpectedly (a backend failed to warm up?)")
    deadline = time.time() + SHUTDOWN_GRACE_S
    while worker.busy and time.time() < deadline:
        time.sleep(0.5)
    worker.retire()


if __name__ == "__main__":
    main()
