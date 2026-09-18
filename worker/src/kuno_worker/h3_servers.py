"""kuno-h3-worker: MiniMax H3's SGLang servers and the worker, supervised together in one container.

The H3 image's entry point. With `KUNO_BACKEND=real` (or `cold`) the H3 profiles go to SGLang's official
server over loopback (backends/h3.py), and in the confidential VM each worker container has to bring its own
servers: kuno-app starts one container per GPU group and gives each its own ports (image/CVM.md, §6). This:
  1. starts the `sglang serve` each profile routes to, on 127.0.0.1 at the URL's port:
       fl2va   h3 (text, image and first/last-frame modes)                    KUNO_H3_FL2VA_URL
       ref2va  h3-reference (reference modes)                                 KUNO_H3_REF2VA_URL
       turbo   h3-turbo: fl2va with LightX2V's LoRA (KUNO_H3_TURBO_LORA)      KUNO_H3_TURBO_URL
     In verified mode on a class h3-turbo pins, h3-turbo runs in the worker process instead (backends/h3.py)
     and gets no server;
  2. waits for each server's /health, then starts `kuno-worker` with this command's arguments, so the
     worker never registers capacity it cannot serve yet;
  3. when any of them exits, stops the rest. SIGTERM reaches the worker first, so an in-flight job
     can finish, then the servers.
Profiles with no SGLang server (LTX-2.5) just run the worker.

One H3 load per container. A loaded 4-GPU server holds 87-97 GB per GPU and peaks at about 103 GB (measured
on H200s, 2026-09-16), so two cannot share 141 GB H200s and very likely not 180 GB B200s. A profile set that
needs more than one (two servers, or a server beside the in-process Turbo pipeline) is refused unless
KUNO_H3_SHARED_SERVERS=1, meant for GPUs that hold two, such as 288 GB B300s. Give each GPU group its own
profiles instead (kuno-app's per-group KUNO_PROFILES).

SGLang's output is discarded unless KUNO_SGLANG_LOG=inherit: nothing guarantees it keeps prompts out of
its logs, and a console is not private to the enclave. kuno-app does not pass that setting into the VM.

Environment, besides the worker's own:
  KUNO_SGLANG_BIN              the sglang executable (default: `sglang` on PATH; the H3 image sets its venv's)
  KUNO_H3_MODEL_ID             --model-path (default MiniMaxAI/MiniMax-H3: with HF_HUB_OFFLINE=1 it resolves in HF_HUB_CACHE)
  KUNO_H3_TURBO_LORA           --lora-path of the Turbo server, and the in-process pipeline's LoRA; h3-turbo needs it
  KUNO_H3_NUM_GPUS             --num-gpus and --ulysses-degree of every server (default: the gpus_per_worker of the
                               profiles that server holds — 4 for h3 and h3-reference, 1 for h3-turbo)
  KUNO_H3_ATTENTION            default (SGLang's own choice, FlashAttention on Hopper) | sage (SageAttention's 8-bit
                               attention, built into the H3 image; 6.5% faster and a different picture, §3 of
                               research/h3-image-check_2026-09-17.md). Refused when the image cannot serve it
  KUNO_H3_SHARED_SERVERS       1 lets one container load H3 more than once on its GPUs (default: refused)
  KUNO_SGLANG_ARGS             extra arguments for every server, shell-quoted
  KUNO_SGLANG_START_TIMEOUT_S  how long to wait for /health (default 3600: loading ~124 GB in a CVM is slow)
  KUNO_SGLANG_LOG              discard (default) | inherit

On GPUs: this launcher started the fl2va and ref2va servers on 4x H200 in the 2026-09-15 smoke test, one profile
per container, and on 2026-09-17 it started the four-GPU Turbo server and served a job through a real gateway, with
the one-load refusal firing as designed and two GPU groups running at once. The one-GPU Turbo server this now plans,
and the `sage` attention backend, have run only straight against SGLang, never from here.
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

from .backends.h3 import TURBO_NICKNAME, is_turbo, server_for, turbo_in_process
from .config import WorkerConfig

log = logging.getLogger("kuno.worker.h3_servers")

SERVERS = ("fl2va", "ref2va", "turbo")
# KUNO_H3_ATTENTION: which attention implementation the SGLang servers run. `default` is whatever SGLang picks
# (FlashAttention on Hopper), the one every measurement and every published clip so far used.
ATTENTION_KEY = "KUNO_H3_ATTENTION"
DEFAULT_ATTENTION = "default"
ATTENTION_BACKENDS = {DEFAULT_ATTENTION: (), "sage": ("--attention-backend", "sage_attn")}
ATTENTION_PACKAGES = {"sage": "sageattention"}
URL_KEYS = {"fl2va": "KUNO_H3_FL2VA_URL", "ref2va": "KUNO_H3_REF2VA_URL", "turbo": "KUNO_H3_TURBO_URL"}
IN_PROCESS = "in-process"
SHARED_KEY = "KUNO_H3_SHARED_SERVERS"
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
    name: str  # fl2va, ref2va or turbo
    port: int
    argv: list[str]

    @property
    def gpus(self) -> int:
        return int(self.argv[self.argv.index("--num-gpus") + 1])

    @property
    def attention(self) -> str:
        """The attention backend this server runs, as SGLang names it: its `--attention-backend`, or SGLang's own default."""
        if "--attention-backend" in self.argv:
            return self.argv[self.argv.index("--attention-backend") + 1]
        return DEFAULT_ATTENTION


def _loopback_port(url: str, key: str) -> int:
    parts = urlsplit(url)
    if parts.hostname != "127.0.0.1" or parts.port is None:
        raise ValueError(f"{key} must be http://127.0.0.1:<port> for a server started here, got {url!r}")
    return parts.port


def _describe(load: str) -> str:
    if load == IN_PROCESS:
        return "in the worker process (verified mode)"
    return f"on the SGLang {load} server"


def attention_args(env: Mapping[str, str], binary: str) -> list[str]:
    """The `--attention-backend` the servers run with, from KUNO_H3_ATTENTION (default | sage).

    `sage` is SageAttention's 8-bit attention, built into the H3 image but off by default: on one H200 it rendered a
    5 s Turbo clip in 47.95 s against FlashAttention's 51.26 s (6.5% faster) for about 2 GB more memory, and the two
    clips differ (30.2 dB PSNR, 0.925 SSIM) — a different picture, not a worse one (research/h3-image-check_2026-09-17.md
    §3). SGLang falls back to FlashAttention with only a log line when the package is missing, so a value this image
    cannot serve is refused here instead. Raises ValueError for an unknown value, or for `sage` in an image without
    sageattention in the SGLang venv."""
    choice = (env.get(ATTENTION_KEY) or DEFAULT_ATTENTION).strip()
    if choice not in ATTENTION_BACKENDS:
        raise ValueError(f"{ATTENTION_KEY}={choice!r} is not one of {', '.join(ATTENTION_BACKENDS)}")
    if choice != DEFAULT_ATTENTION and not _installed(binary, ATTENTION_PACKAGES[choice]):
        raise ValueError(
            f"{ATTENTION_KEY}={choice} needs {ATTENTION_PACKAGES[choice]} in the SGLang environment of {binary}, which "
            f"this image does not have; SGLang would fall back to FlashAttention without failing. Use "
            f"{ATTENTION_KEY}={DEFAULT_ATTENTION}, or an image built with it (image/worker.Dockerfile, target h3)"
        )
    return list(ATTENTION_BACKENDS[choice])


def _installed(binary: str, package: str) -> bool:
    """Whether `package` is in the site-packages of the venv `binary` lives in. True when the venv cannot be located
    (a bare `sglang` on PATH, as in tests and on a development machine): there is nothing to check it against."""
    if os.sep not in binary:
        return True
    site = sorted(Path(binary).parent.parent.glob(f"lib/python*/site-packages/{package}"))
    return bool(site) or not sorted(Path(binary).parent.parent.glob("lib/python*/site-packages"))


def plan_servers(config: WorkerConfig, env: Mapping[str, str], catalog: Mapping[str, ModelProfile] | None = None) -> list[Server]:
    """The `sglang serve` commands the configured profiles need, in fl2va, ref2va, turbo order. Raises ValueError for
    a profile set that loads H3 more than once on this container's GPUs (unless KUNO_H3_SHARED_SERVERS=1), and
    for h3-turbo without KUNO_H3_TURBO_LORA."""
    if config.backend not in ("real", "cold"):
        return []
    catalog = load_profiles() if catalog is None else catalog
    h3 = [catalog[p] for p in config.profiles if p in catalog and catalog[p].family == "minimax-h3"]
    # Each H3 load (a server, or the in-process Turbo pipeline) and the profiles that use it.
    loads: dict[str, list[str]] = {}
    served = []
    for profile in h3:
        if config.backend == "real" and turbo_in_process(profile, config.verified_hardware_class):
            loads.setdefault(IN_PROCESS, []).append(profile.id)
            continue
        served.append(profile)
        for server in sorted({server_for(profile, mode) for mode in profile.modes}, key=SERVERS.index):
            loads.setdefault(server, []).append(profile.id)
    if any(is_turbo(profile) for profile in h3) and not config.h3_turbo_lora:
        raise ValueError("h3-turbo needs KUNO_H3_TURBO_LORA, the path of LightX2V's 8-step 768p LoRA (lightx2v/Minimax-h3-Turbo)")
    if len(loads) > 1 and env.get(SHARED_KEY) != "1":
        uses = "; ".join(f"{', '.join(ids)} {_describe(load)}" for load, ids in loads.items())
        times = "twice" if len(loads) == 2 else f"{len(loads)} times"
        raise ValueError(
            f"profiles {', '.join(p.id for p in h3)} would load MiniMax H3 {times} on this worker's GPUs ({uses}). "
            "One loaded H3 holds 87-97 GB per H200, so two don't fit on 141 GB H200s and very likely not on 180 GB B200s. "
            "Give each GPU group one of them (KUNO_GPU_GROUPS with a KUNO_PROFILES list per group, image/CVM.md §6), "
            f"or set {SHARED_KEY}=1 on GPUs that hold both, such as 288 GB B300s"
        )
    if not served:
        return []
    binary = env.get("KUNO_SGLANG_BIN") or "sglang"
    extra = shlex.split(env.get("KUNO_SGLANG_ARGS", ""))
    attention = attention_args(env, binary)
    override = env.get("KUNO_H3_NUM_GPUS")
    by_id = {profile.id: profile for profile in served}
    urls = {"fl2va": config.h3_fl2va_url, "ref2va": config.h3_ref2va_url, "turbo": config.h3_turbo_url}
    servers = []
    for name in SERVERS:
        if name not in loads:
            continue
        port = _loopback_port(urls[name], URL_KEYS[name])
        # Each server runs on the GPUs its own profiles need: four for h3 and h3-reference, one for h3-turbo,
        # which a single H200 serves more cheaply (profiles.json, measured 2026-09-17).
        gpus = int(override or max(by_id[p].gpus_per_worker for p in loads[name] if p in by_id))
        # The Turbo server is the fl2va checkpoint with the LoRA loaded at start, as measured on 2026-09-16.
        lora = ["--lora-path", str(config.h3_turbo_lora), "--lora-nickname", TURBO_NICKNAME] if name == "turbo" else []
        argv = [
            binary, "serve", "--model-path", config.h3_model_id, "--model-variant", "fl2va" if name == "turbo" else name,
            "--num-gpus", str(gpus), "--ulysses-degree", str(gpus), "--performance-mode", "speed",
            "--host", "127.0.0.1", "--port", str(port),
            "--master-port", str(port + MASTER_PORT_OFFSET), "--scheduler-port", str(port + SCHEDULER_PORT_OFFSET),
            *lora, *attention, *extra,
        ]
        servers.append(Server(name, port, argv))
    if len({server.port for server in servers}) != len(servers):
        raise ValueError(f"{', '.join(URL_KEYS[s.name] for s in servers)} must use different ports")
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
                    raise LaunchError(f"the SGLang {server.name} server exited with code {process.returncode} before it was ready")
            for port, server in list(pending.items()):
                with contextlib.suppress(httpx.HTTPError):
                    if client.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                        log.info("SGLang %s server ready on 127.0.0.1:%d", server.name, port)
                        del pending[port]
            if not pending:
                return True
            if time.monotonic() > deadline:
                raise LaunchError(f"SGLang {', '.join(s.name for s in pending.values())} not ready after {timeout_s:g} s")
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
            # The attention backend changes the pictures a miner produces, so say in the log which one this worker ran.
            log.info("starting the SGLang %s server on 127.0.0.1:%d (%d GPU(s), %s attention)", server.name, server.port,
                     server.gpus, server.attention)
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
                    log.error("the SGLang %s server exited with code %s; stopping the worker", server.name, process.returncode)
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
