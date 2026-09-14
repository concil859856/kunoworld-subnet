"""kuno-h3-worker: MiniMax H3's SGLang servers and the worker, supervised together in one container.

The H3 image's entry point. With `KUNO_BACKEND=real` (or `cold`) the `h3` and `h3-reference` profiles
go to SGLang's official server over loopback (backends/h3.py, at KUNO_H3_FL2VA_URL and
KUNO_H3_REF2VA_URL), and in the confidential VM each worker container has to bring its own servers:
kuno-app starts one container per GPU group and gives each its own ports (image/CVM.md, §6). This:
  1. starts one `sglang serve` per checkpoint variant the profiles route to (fl2va for text, image and
     first/last-frame modes, ref2va for reference modes), on 127.0.0.1 at the URL's port;
  2. waits for each server's /health, then starts `kuno-worker` with this command's arguments, so the
     worker never registers capacity it cannot serve yet;
  3. when any of them exits, stops the rest. SIGTERM reaches the worker first, so an in-flight job
     can finish, then the servers.
Profiles with no SGLang server (h3-turbo runs in the worker process; LTX-2.5) just run the worker.

SGLang's output is discarded unless KUNO_SGLANG_LOG=inherit: nothing guarantees it keeps prompts out of
its logs, and a console is not private to the enclave. kuno-app does not pass that setting into the VM.

Environment, besides the worker's own:
  KUNO_SGLANG_BIN              the sglang executable (default: `sglang` on PATH; the H3 image sets its venv's)
  KUNO_H3_MODEL_ID             --model-path (default MiniMaxAI/MiniMax-H3: with HF_HUB_OFFLINE=1 it resolves in HF_HUB_CACHE)
  KUNO_H3_NUM_GPUS             --num-gpus and --ulysses-degree (default: the profiles' gpus_per_worker)
  KUNO_SGLANG_ARGS             extra arguments for every server, shell-quoted
  KUNO_SGLANG_START_TIMEOUT_S  how long to wait for /health (default 3600: loading ~124 GB in a CVM is slow)
  KUNO_SGLANG_LOG              discard (default) | inherit

Not yet run against a real SGLang server or on GPUs: only against fake servers in the tests.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from kuno_protocol.profiles import ModelProfile, load_profiles

from .backends.h3 import FL2VA_MODES
from .config import WorkerConfig

log = logging.getLogger("kuno.worker.h3_servers")

VARIANTS = ("fl2va", "ref2va")
URL_KEYS = {"fl2va": "KUNO_H3_FL2VA_URL", "ref2va": "KUNO_H3_REF2VA_URL"}
# SGLang settles its torch.distributed master and scheduler ports itself, checking only that a port is
# free when it looks, so two servers starting together can pick the same one. Fixed offsets from each
# server's HTTP port keep every server on the host apart (30010 -> 31010 and 32010).
MASTER_PORT_OFFSET = 1000
SCHEDULER_PORT_OFFSET = 2000
# kuno-worker gives an in-flight job 120 s after SIGTERM.
WORKER_GRACE_S = 150.0
SERVER_GRACE_S = 30.0


class LaunchError(RuntimeError):
    """An SGLang server failed to start or to become ready."""


@dataclass(frozen=True)
class Server:
    variant: str
    port: int
    argv: list[str]


def _loopback_port(url: str, key: str) -> int:
    parts = urlsplit(url)
    if parts.hostname != "127.0.0.1" or parts.port is None:
        raise ValueError(f"{key} must be http://127.0.0.1:<port> for a server started here, got {url!r}")
    return parts.port


def plan_servers(config: WorkerConfig, env: Mapping[str, str], catalog: Mapping[str, ModelProfile] | None = None) -> list[Server]:
    """The `sglang serve` commands the configured profiles need, in fl2va, ref2va order."""
    if config.backend not in ("real", "cold"):
        return []
    catalog = load_profiles() if catalog is None else catalog
    served = [catalog[p] for p in config.profiles if p in catalog and catalog[p].family == "minimax-h3" and catalog[p].runtime == "sglang"]
    needed = {"fl2va" if mode in FL2VA_MODES else "ref2va" for profile in served for mode in profile.modes}
    if not needed:
        return []
    gpus = int(env.get("KUNO_H3_NUM_GPUS") or max(profile.gpus_per_worker for profile in served))
    binary = env.get("KUNO_SGLANG_BIN") or "sglang"
    extra = shlex.split(env.get("KUNO_SGLANG_ARGS", ""))
    urls = {"fl2va": config.h3_fl2va_url, "ref2va": config.h3_ref2va_url}
    servers = []
    for variant in VARIANTS:
        if variant not in needed:
            continue
        port = _loopback_port(urls[variant], URL_KEYS[variant])
        argv = [
            binary, "serve", "--model-path", config.h3_model_id, "--model-variant", variant,
            "--num-gpus", str(gpus), "--ulysses-degree", str(gpus), "--performance-mode", "speed",
            "--host", "127.0.0.1", "--port", str(port),
            "--master-port", str(port + MASTER_PORT_OFFSET), "--scheduler-port", str(port + SCHEDULER_PORT_OFFSET),
            *extra,
        ]
        servers.append(Server(variant, port, argv))
    if len({server.port for server in servers}) != len(servers):
        raise ValueError("KUNO_H3_FL2VA_URL and KUNO_H3_REF2VA_URL must use different ports")
    return servers


def server_env(env: Mapping[str, str], binary: str) -> dict[str, str]:
    """The servers' environment: without the worker's settings and secrets, and with the SGLang venv's
    bin directory first on PATH, so the python, ninja and other tools its kernels call resolve there.

    SGLang's JIT kernels find nvcc through CUDA_HOME (then PATH, then /usr/local/cuda). Unless one is set, a
    CUDA toolkit the venv installed from pip wheels (site-packages/nvidia/cu13, holding bin/nvcc) becomes it."""
    out = {key: value for key, value in env.items() if not key.startswith("KUNO_")}
    if os.sep in binary:
        venv_bin = Path(binary).parent
        out["PATH"] = os.pathsep.join(part for part in (str(venv_bin), env.get("PATH", "")) if part)
        if not (env.get("CUDA_HOME") or env.get("CUDA_PATH")):
            toolkits = sorted(venv_bin.parent.glob("lib/python*/site-packages/nvidia/cu*/bin/nvcc"))
            if toolkits:
                out["CUDA_HOME"] = str(toolkits[-1].parent.parent)
    return out


def wait_until_ready(servers: Sequence[Server], processes: Sequence[subprocess.Popen], timeout_s: float, stop: threading.Event, poll_s: float = 2.0) -> bool:
    """True once every server answers /health with 200; False if `stop` is set first. Raises LaunchError."""
    pending = {server.port: server for server in servers}
    deadline = time.monotonic() + timeout_s
    with httpx.Client(timeout=5.0) as client:
        while True:
            for server, process in zip(servers, processes):
                if process.poll() is not None:
                    raise LaunchError(f"the SGLang {server.variant} server exited with code {process.returncode} before it was ready")
            for port, server in list(pending.items()):
                with contextlib.suppress(httpx.HTTPError):
                    if client.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                        log.info("SGLang %s server ready on 127.0.0.1:%d", server.variant, port)
                        del pending[port]
            if not pending:
                return True
            if time.monotonic() > deadline:
                raise LaunchError(f"SGLang {', '.join(s.variant for s in pending.values())} not ready after {timeout_s:g} s")
            if stop.wait(poll_s):
                return False


def _exit_code(returncode: int | None) -> int:
    if returncode is None:
        return 1
    return 128 - returncode if returncode < 0 else returncode


def _stop(process: subprocess.Popen, grace_s: float, group: bool) -> None:
    """SIGTERM, then SIGKILL after `grace_s`. A server is stopped with its whole process group, since
    SGLang runs its schedulers in child processes."""

    def send(sig: int) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, sig) if group else process.send_signal(sig)

    if process.poll() is None:
        send(signal.SIGTERM)
        try:
            process.wait(grace_s)
        except subprocess.TimeoutExpired:
            send(signal.SIGKILL)
            process.wait()
    if group:
        send(signal.SIGKILL)  # anything the server left behind in its group


def run(argv: Sequence[str], env: Mapping[str, str] | None = None, *, worker_command: Sequence[str] | None = None,
        stop: threading.Event | None = None, poll_s: float = 1.0) -> int:
    """Runs the servers and the worker until one exits or `stop` is set; returns the exit code."""
    env = dict(os.environ if env is None else env)
    worker = [*(worker_command or [sys.executable, "-m", "kuno_worker.main"]), *argv]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profiles")
    parser.add_argument("--backend")
    parser.add_argument("-h", "--help", action="store_true")
    known, _ = parser.parse_known_args(list(argv))
    config = WorkerConfig.from_env(env)
    if known.profiles:
        config.profiles = [p.strip() for p in known.profiles.split(",") if p.strip()]
    if known.backend:
        config.backend = known.backend
    servers = [] if known.help else plan_servers(config, env)
    if stop is None:
        stop = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.set())

    output = None if env.get("KUNO_SGLANG_LOG", "discard") == "inherit" else subprocess.DEVNULL
    processes: list[subprocess.Popen] = []
    worker_process: subprocess.Popen | None = None
    try:
        for server in servers:
            log.info("starting the SGLang %s server on 127.0.0.1:%d", server.variant, server.port)
            processes.append(subprocess.Popen(server.argv, env=server_env(env, server.argv[0]), stdin=subprocess.DEVNULL,
                                              stdout=output, stderr=output, start_new_session=True))
        timeout_s = float(env.get("KUNO_SGLANG_START_TIMEOUT_S", "3600"))
        if servers and not wait_until_ready(servers, processes, timeout_s, stop, poll_s=min(poll_s * 2, 2.0)):
            return 143
        worker_process = subprocess.Popen(worker, env=env)
        while True:
            if stop.is_set():
                _stop(worker_process, WORKER_GRACE_S, group=False)
                return _exit_code(worker_process.returncode)
            if worker_process.poll() is not None:
                if servers:
                    log.info("kuno-worker exited with code %s; stopping the SGLang servers", worker_process.returncode)
                return _exit_code(worker_process.returncode)
            for server, process in zip(servers, processes):
                if process.poll() is not None:
                    log.error("the SGLang %s server exited with code %s; stopping the worker", server.variant, process.returncode)
                    return 1
            stop.wait(poll_s)
    except (LaunchError, OSError) as exc:
        log.error("%s (set KUNO_SGLANG_LOG=inherit on a development machine to see the server's output)", exc)
        return 1
    finally:
        if worker_process is not None:
            _stop(worker_process, WORKER_GRACE_S, group=False)
        for process in processes:
            _stop(process, SERVER_GRACE_S, group=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        code = run(sys.argv[1:])
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    raise SystemExit(code)


if __name__ == "__main__":
    main()
