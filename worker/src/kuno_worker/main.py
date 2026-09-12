from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time

from kuno_protocol.attestation import MockTEE, TdxTEE
from kuno_protocol.canonical import b64d
from kuno_protocol.crypto import signing_key_from_bytes

from .backends import build_backends
from .config import WorkerConfig
from .worker import Worker

log = logging.getLogger("kuno.worker")

# On shutdown, an in-flight job gets this long to finish before the process exits.
SHUTDOWN_GRACE_S = 120.0


def build_tee(config: WorkerConfig):
    if config.tee == "tdx":
        return TdxTEE()
    if config.tee == "mock":
        if config.mock_quote_key_file is None:
            raise SystemExit("KUNO_MOCK_QUOTE_KEY_FILE is required for the mock TEE (run `kuno-devkit init`).")
        key = signing_key_from_bytes(b64d(config.mock_quote_key_file.read_text().strip()))
        return MockTEE(key, config.image_digest)
    raise SystemExit(f"unknown TEE {config.tee!r}; use 'tdx' or 'mock'")


def main() -> None:
    parser = argparse.ArgumentParser(prog="kuno-worker", description="KunoWorld miner worker (runs inside the confidential VM)")
    parser.add_argument("--profiles", help="comma-separated profile ids (default: KUNO_PROFILES)")
    parser.add_argument("--backend", choices=["mock", "real"], help="default: KUNO_BACKEND or mock")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = WorkerConfig.from_env()
    if args.profiles:
        config.profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    if args.backend:
        config.backend = args.backend
    worker = Worker(config, build_tee(config), build_backends(config.backend, config))

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
            raise SystemExit("worker loop exited unexpectedly")
    deadline = time.time() + SHUTDOWN_GRACE_S
    while worker.busy and time.time() < deadline:
        time.sleep(0.5)


if __name__ == "__main__":
    main()
