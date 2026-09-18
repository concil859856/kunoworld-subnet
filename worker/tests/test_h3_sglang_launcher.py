"""kuno-h3-worker: which SGLang servers the H3 image starts for which profiles, and how it supervises them
with the worker. Fake servers and a fake worker stand in; the Turbo server and the one-load rule have not run
against SGLang or a GPU."""

from __future__ import annotations

import logging
import os
import re
import socket
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from kuno_worker.backends import build_backends
from kuno_worker.config import WorkerConfig
from kuno_worker import h3_servers
from kuno_worker.h3_servers import ATTENTION_KEY, plan_servers, run, server_env


LORA = "/models/h3-turbo/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors"
# h3-turbo is a one-GPU profile, and the classes it pins are the single-GPU ones (profiles.json, 2026-09-17).
H3_CLASS = "C2.h200-141gb.x1"
# A verified class h3-turbo does not pin: h3 and h3-reference's four-GPU one.
OTHER_CLASS = "C4.h200-141gb.x4.ulysses4"


@pytest.fixture(autouse=True)
def gpus_nvml_cannot_see(monkeypatch):
    """As on a machine without a driver, whatever this one has: tests that need GPUs say which."""
    monkeypatch.setattr(h3_servers, "visible_compute_capabilities", lambda: None)


def on_gpus(monkeypatch, *capabilities: tuple[int, int]) -> None:
    monkeypatch.setattr(h3_servers, "visible_compute_capabilities", lambda: list(capabilities))


H200, B200 = (9, 0), (10, 0)


def configure(tmp_path: Path, **values: str) -> tuple[WorkerConfig, dict[str, str]]:
    env = {"KUNO_DATA_DIR": str(tmp_path / "no-dev-env"), "KUNO_BACKEND": "real", **values}
    return WorkerConfig.from_env(env), env


# ---------------------------------------------------------------- which servers


def test_h3_needs_only_the_fl2va_server_run_with_the_official_recipe(tmp_path):
    config, env = configure(tmp_path, KUNO_PROFILES="h3")
    [server] = plan_servers(config, env)
    assert (server.name, server.port) == ("fl2va", 30010)
    assert server.argv == [
        "sglang", "serve", "--model-path", "MiniMaxAI/MiniMax-H3", "--model-variant", "fl2va",
        "--num-gpus", "4", "--ulysses-degree", "4", "--performance-mode", "speed",
        "--host", "127.0.0.1", "--port", "30010", "--master-port", "31010", "--scheduler-port", "32010",
    ]


@pytest.mark.parametrize("backend", ["real", "cold"])
def test_h3_turbo_gets_its_own_one_gpu_fl2va_server_with_the_lora_loaded_as_measured(tmp_path, backend):
    """One GPU, not four: a 1-GPU H200 runs Turbo at 9.92 GPU-seconds per output second against 15.6 through a
    4-GPU worker, the only serving that fits our prices (research/h3-image-check_2026-09-17.md §4)."""
    config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo", KUNO_BACKEND=backend, KUNO_H3_TURBO_LORA=LORA)
    [server] = plan_servers(config, env)
    assert (server.name, server.port, server.gpus, server.attention) == ("turbo", 30012, 1, "default")
    assert server.argv == [
        "sglang", "serve", "--model-path", "MiniMaxAI/MiniMax-H3", "--model-variant", "fl2va",
        "--num-gpus", "1", "--ulysses-degree", "1", "--performance-mode", "speed",
        "--host", "127.0.0.1", "--port", "30012", "--master-port", "31012", "--scheduler-port", "32012",
        "--lora-path", LORA, "--lora-nickname", "turbo",
    ]


def test_each_server_runs_on_the_gpus_its_own_profiles_need(tmp_path):
    """A B300 container sharing both loads: h3 keeps its four GPUs and h3-turbo its one."""
    config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo,h3,h3-reference", KUNO_H3_TURBO_LORA=LORA, KUNO_H3_SHARED_SERVERS="1")
    assert [(s.name, s.gpus) for s in plan_servers(config, env)] == [("fl2va", 4), ("ref2va", 4), ("turbo", 1)]
    # KUNO_H3_NUM_GPUS still overrides every server.
    config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo,h3,h3-reference", KUNO_H3_TURBO_LORA=LORA,
                            KUNO_H3_SHARED_SERVERS="1", KUNO_H3_NUM_GPUS="2")
    assert [(s.name, s.gpus) for s in plan_servers(config, env)] == [("fl2va", 2), ("ref2va", 2), ("turbo", 2)]


# ---------------------------------------------------------------- the attention backend


def venv(tmp_path: Path, *packages: str) -> str:
    """An SGLang venv's `sglang` binary, with these packages in its site-packages."""
    site = tmp_path / "sglang" / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    for package in packages:
        (site / package).mkdir(exist_ok=True)
    binary = tmp_path / "sglang" / "bin" / "sglang"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.touch()
    return str(binary)


def test_turbo_runs_sageattention_by_default_on_h200s(tmp_path, monkeypatch):
    """The owner compared Turbo's clips by eye on 2026-09-18 and could not tell SageAttention's from FlashAttention's,
    and SageAttention is 6.5% faster, so `auto` (the default) gives it to the Turbo server wherever it can run."""
    on_gpus(monkeypatch, H200)
    sage_venv = venv(tmp_path / "with", "sageattention")
    for value in ("", "auto"):
        config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo", KUNO_H3_TURBO_LORA=LORA, KUNO_SGLANG_BIN=sage_venv,
                                **({ATTENTION_KEY: value} if value else {}))
        [server] = plan_servers(config, env)
        assert server.argv[-6:] == ["--lora-path", LORA, "--lora-nickname", "turbo", "--attention-backend", "sage_attn"]
        assert server.attention == "sage_attn"
    # Full H3 runs 50 passes, where nobody has compared the pictures: it keeps SGLang's default, even beside Turbo.
    config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo,h3,h3-reference", KUNO_H3_TURBO_LORA=LORA,
                            KUNO_H3_SHARED_SERVERS="1", KUNO_SGLANG_BIN=sage_venv)
    assert [(s.name, s.attention) for s in plan_servers(config, env)] == [
        ("fl2va", "default"), ("ref2va", "default"), ("turbo", "sage_attn"),
    ]
    config, env = configure(tmp_path, KUNO_PROFILES="h3", KUNO_SGLANG_BIN=sage_venv)
    [server] = plan_servers(config, env)
    assert "--attention-backend" not in server.argv and server.attention == "default"


@pytest.mark.parametrize("gpus, package, reason", [
    ([B200], True, "built for sm_90, not sm_100"),
    ([H200, B200], True, "built for sm_90, not sm_100"),
    ([H200], False, "this image has no SageAttention"),
    (None, True, "NVML did not report the GPUs' architecture"),
])
def test_auto_falls_back_to_sglangs_default_where_sageattention_cannot_run(tmp_path, monkeypatch, caplog, gpus, package, reason):
    """The image's kernels are built for Hopper only (SAGE_ARCH 9.0), so a B200 or B300 Turbo worker serves with
    SGLang's default instead of failing, and says why."""
    if gpus is not None:
        on_gpus(monkeypatch, *gpus)
    binary = venv(tmp_path / "v", *(["sageattention"] if package else []))
    config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo", KUNO_H3_TURBO_LORA=LORA, KUNO_SGLANG_BIN=binary)
    with caplog.at_level(logging.INFO, logger="kuno.worker.h3_servers"):
        [server] = plan_servers(config, env)
    assert "--attention-backend" not in server.argv and server.attention == "default"
    assert "KUNO_H3_ATTENTION=auto: SGLang's default attention on every server, because " in caplog.text
    assert reason in caplog.text


def test_default_and_sage_are_asked_for_by_name(tmp_path, monkeypatch):
    """`default` puts every server back on exactly the command the measurements before 2026-09-17 used; `sage` adds
    SGLang's `--attention-backend sage_attn` to every server, full H3 included."""
    on_gpus(monkeypatch, H200)
    sage_venv = venv(tmp_path / "with", "sageattention")
    config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo", KUNO_H3_TURBO_LORA=LORA, KUNO_SGLANG_BIN=sage_venv,
                            KUNO_H3_ATTENTION="default")
    [server] = plan_servers(config, env)
    assert "--attention-backend" not in server.argv and server.attention == "default"
    config, env = configure(tmp_path, KUNO_PROFILES="h3", KUNO_SGLANG_BIN=sage_venv, KUNO_H3_ATTENTION="sage")
    [server] = plan_servers(config, env)
    assert server.argv[-2:] == ["--attention-backend", "sage_attn"] and server.attention == "sage_attn"
    # The extra arguments still come last, so KUNO_SGLANG_ARGS can override the choice by hand.
    config, env = configure(tmp_path, KUNO_PROFILES="h3", KUNO_SGLANG_BIN=sage_venv, KUNO_H3_ATTENTION="sage",
                            KUNO_SGLANG_ARGS="--mem-fraction-static 0.8")
    [server] = plan_servers(config, env)
    assert server.argv[-4:] == ["--attention-backend", "sage_attn", "--mem-fraction-static", "0.8"]


def test_an_attention_backend_the_image_cannot_serve_is_refused(tmp_path, monkeypatch):
    """SGLang falls back to FlashAttention with only a log line, so the worker refuses `sage` asked for by name
    instead of serving something other than what the miner asked for."""
    config, env = configure(tmp_path, KUNO_PROFILES="h3", KUNO_H3_ATTENTION="flash")
    with pytest.raises(ValueError, match="KUNO_H3_ATTENTION='flash' is not one of auto, default, sage"):
        plan_servers(config, env)
    config, env = configure(tmp_path, KUNO_PROFILES="h3", KUNO_H3_ATTENTION="sage", KUNO_SGLANG_BIN=venv(tmp_path / "without"))
    with pytest.raises(ValueError, match="needs sageattention in the SGLang environment"):
        plan_servers(config, env)
    on_gpus(monkeypatch, B200)
    config, env = configure(tmp_path, KUNO_PROFILES="h3", KUNO_H3_ATTENTION="sage", KUNO_SGLANG_BIN=venv(tmp_path / "with", "sageattention"))
    with pytest.raises(ValueError, match="kernels are built for sm_90 only, and this worker's GPUs include sm_100"):
        plan_servers(config, env)
    # A bare `sglang` on PATH on a machine without a driver has nothing to check against, and is taken at its word.
    monkeypatch.setattr(h3_servers, "visible_compute_capabilities", lambda: None)
    config, env = configure(tmp_path, KUNO_PROFILES="h3", KUNO_H3_ATTENTION="sage")
    assert plan_servers(config, env)[0].attention == "sage_attn"


@pytest.mark.parametrize("backend", ["real", "cold"])
def test_h3_turbo_is_refused_without_its_lora(tmp_path, backend):
    config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo", KUNO_BACKEND=backend)
    with pytest.raises(ValueError, match="h3-turbo needs KUNO_H3_TURBO_LORA"):
        plan_servers(config, env)
    config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo", KUNO_VERIFIED_HARDWARE_CLASS=H3_CLASS, KUNO_BACKEND=backend)
    with pytest.raises(ValueError, match="h3-turbo needs KUNO_H3_TURBO_LORA"):  # the in-process pipeline needs it too
        plan_servers(config, env)


@pytest.mark.parametrize(
    ("profiles", "extra", "named"),
    [
        ("h3,h3-reference", {}, ["h3 on the SGLang fl2va server", "h3-reference on the SGLang ref2va server"]),
        ("h3-turbo,h3", {}, ["h3 on the SGLang fl2va server", "h3-turbo on the SGLang turbo server"]),
        ("h3-turbo,h3-reference,ltx-2.5-fast", {}, ["h3-reference on the SGLang ref2va server", "h3-turbo on the SGLang turbo server"]),
        ("h3-turbo,h3", {"KUNO_VERIFIED_HARDWARE_CLASS": H3_CLASS}, ["h3-turbo in the worker process", "h3 on the SGLang fl2va server"]),
        ("h3,h3-reference", {"KUNO_H3_SHARED_SERVERS": "true"}, ["h3 on the SGLang fl2va server"]),  # only "1" shares
    ],
)
def test_a_worker_that_would_load_h3_more_than_once_on_its_gpus_is_refused(tmp_path, profiles, extra, named):
    config, env = configure(tmp_path, KUNO_PROFILES=profiles, KUNO_H3_TURBO_LORA=LORA, **extra)
    with pytest.raises(ValueError) as refused:
        plan_servers(config, env)
    message = str(refused.value)
    assert "would load MiniMax H3 twice" in message and all(part in message for part in named)
    assert "KUNO_PROFILES list per group" in message and "KUNO_H3_SHARED_SERVERS=1" in message


def test_shared_servers_start_every_server_the_profiles_route_to(tmp_path):
    config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo,h3,h3-reference,ltx-2.5-fast", KUNO_H3_TURBO_LORA=LORA, KUNO_H3_SHARED_SERVERS="1")
    assert [(s.name, s.port) for s in plan_servers(config, env)] == [("fl2va", 30010), ("ref2va", 30011), ("turbo", 30012)]
    config, env = configure(tmp_path, KUNO_PROFILES="h3-reference")
    assert [s.name for s in plan_servers(config, env)] == ["ref2va"]


def test_verified_h3_turbo_runs_in_the_worker_process_and_gets_no_server(tmp_path):
    config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo", KUNO_H3_TURBO_LORA=LORA, KUNO_VERIFIED_HARDWARE_CLASS=H3_CLASS)
    assert plan_servers(config, env) == []
    # A class h3-turbo does not pin leaves it in performance mode, on its server; cold has no in-process pipeline.
    for extra in ({"KUNO_VERIFIED_HARDWARE_CLASS": OTHER_CLASS}, {"KUNO_VERIFIED_HARDWARE_CLASS": H3_CLASS, "KUNO_BACKEND": "cold"}):
        config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo", KUNO_H3_TURBO_LORA=LORA, **extra)
        assert [s.name for s in plan_servers(config, env)] == ["turbo"]


@pytest.mark.parametrize(("profiles", "backend"), [("ltx-2.5-fast,ltx-2.5-pro", "real"), ("h3,h3-reference", "mock")])
def test_no_server_starts_when_no_profile_is_served_by_sglang(tmp_path, profiles, backend):
    config, env = configure(tmp_path, KUNO_PROFILES=profiles, KUNO_BACKEND=backend)
    assert plan_servers(config, env) == []


def test_the_second_worker_of_a_whole_server_td_gets_its_own_ports(tmp_path):
    # kuno-app's settings for worker 1 (image/cvm/rootfs/usr/lib/kuno/kuno-app, worker_settings).
    config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo,h3,h3-reference", KUNO_H3_TURBO_LORA=LORA, KUNO_H3_SHARED_SERVERS="1",
                            KUNO_H3_FL2VA_URL="http://127.0.0.1:30020", KUNO_H3_REF2VA_URL="http://127.0.0.1:30021",
                            KUNO_H3_TURBO_URL="http://127.0.0.1:30022")
    ports = [(s.port, s.argv[s.argv.index("--master-port") + 1], s.argv[s.argv.index("--scheduler-port") + 1]) for s in plan_servers(config, env)]
    assert ports == [(30020, "31020", "32020"), (30021, "31021", "32021"), (30022, "31022", "32022")]


def test_servers_started_together_need_different_ports(tmp_path):
    config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo,h3", KUNO_H3_TURBO_LORA=LORA, KUNO_H3_SHARED_SERVERS="1",
                            KUNO_H3_TURBO_URL="http://127.0.0.1:30010")
    with pytest.raises(ValueError, match="KUNO_H3_FL2VA_URL, KUNO_H3_TURBO_URL must use different ports"):
        plan_servers(config, env)


def test_operator_settings_reach_the_server_command(tmp_path):
    config, env = configure(tmp_path, KUNO_PROFILES="h3", KUNO_SGLANG_BIN="/opt/sglang/bin/sglang", KUNO_H3_MODEL_ID="/models/MiniMax-H3",
                            KUNO_H3_NUM_GPUS="8", KUNO_SGLANG_ARGS="--tp-size 2 --dit-cpu-offload 'false'")
    argv = plan_servers(config, env)[0].argv
    assert argv[:4] == ["/opt/sglang/bin/sglang", "serve", "--model-path", "/models/MiniMax-H3"]
    assert argv[argv.index("--num-gpus") + 1] == "8" == argv[argv.index("--ulysses-degree") + 1]
    assert argv[-4:] == ["--tp-size", "2", "--dit-cpu-offload", "false"]


@pytest.mark.parametrize(("profiles", "key"), [("h3", "KUNO_H3_FL2VA_URL"), ("h3-turbo", "KUNO_H3_TURBO_URL")])
@pytest.mark.parametrize("url", ["http://10.0.0.5:30010", "http://127.0.0.1", "http://localhost:30010"])
def test_a_server_started_here_must_listen_on_loopback_at_an_explicit_port(tmp_path, url, profiles, key):
    config, env = configure(tmp_path, KUNO_PROFILES=profiles, KUNO_H3_TURBO_LORA=LORA, **{key: url})
    with pytest.raises(ValueError, match=key):
        plan_servers(config, env)


def test_the_h3_images_default_profiles_start_one_server(tmp_path):
    dockerfile = Path(__file__).resolve().parents[2] / "image" / "worker.Dockerfile"
    if not dockerfile.exists():
        pytest.skip("image/ is not part of this checkout")
    stage = dockerfile.read_text().split(" AS h3\n", 1)[1]
    profiles = re.search(r"KUNO_PROFILES=(\S+)", stage).group(1)
    config, env = configure(tmp_path, KUNO_PROFILES=profiles)
    assert len(plan_servers(config, env)) == 1


def test_the_servers_get_none_of_the_workers_settings_and_find_their_own_venv_first():
    env = server_env({"PATH": "/opt/kuno/bin:/usr/bin", "KUNO_HOTKEY_SEED_FILE": "/run/secrets/hotkey.seed", "HF_HUB_OFFLINE": "1"}, "/opt/sglang/bin/sglang")
    assert env == {"PATH": "/opt/sglang/bin:/opt/kuno/bin:/usr/bin", "HF_HUB_OFFLINE": "1"}


def test_the_servers_find_the_cuda_compiler_their_venv_ships(tmp_path):
    venv = tmp_path / "sglang"
    nvcc = venv / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    binary = str(venv / "bin" / "sglang")
    assert server_env({"PATH": "/usr/bin"}, binary)["CUDA_HOME"] == str(nvcc.parent.parent)
    assert server_env({"PATH": "/usr/bin", "CUDA_HOME": "/usr/local/cuda-13.0"}, binary)["CUDA_HOME"] == "/usr/local/cuda-13.0"
    assert "CUDA_HOME" not in server_env({"PATH": "/usr/bin", "CUDA_PATH": "/usr/local/cuda"}, binary)
    assert "CUDA_HOME" not in server_env({"PATH": "/usr/bin"}, "sglang")


# ---------------------------------------------------------------- which runtime the worker sends each profile to


@pytest.fixture
def loaded(monkeypatch):
    from kuno_worker.backends import runtimes

    seen = []

    def fake_loader(model_id, **kwargs):
        seen.append((model_id, kwargs.get("turbo_lora")))
        return lambda profile: None

    monkeypatch.setattr(runtimes, "h3_loader", fake_loader)
    return seen


def test_verified_h3_turbo_gets_the_in_process_pipeline_with_the_servers_weights(tmp_path, loaded):
    from kuno_protocol.profiles import load_profiles

    config, _ = configure(tmp_path, KUNO_H3_MODEL_ID="/models/MiniMax-H3", KUNO_LTX_MODELS_DIR=str(tmp_path),
                          KUNO_VERIFIED_HARDWARE_CLASS=H3_CLASS, KUNO_H3_TURBO_LORA=LORA)
    h3 = build_backends("real", config)["minimax-h3"]
    assert loaded == [("/models/MiniMax-H3", LORA)]
    profiles = load_profiles()
    assert h3.verified_enabled(profiles["h3-turbo"]) and h3.turbo.hardware_class == H3_CLASS
    # SGLang has no step hook: h3 and h3-reference stay on their servers, without step commitments.
    assert not h3.verified_enabled(profiles["h3"]) and not h3.verified_enabled(profiles["h3-reference"])


@pytest.mark.parametrize(("backend", "hardware_class"), [("real", None), ("real", OTHER_CLASS), ("cold", H3_CLASS)])
def test_performance_mode_sends_every_h3_profile_to_the_servers(tmp_path, loaded, backend, hardware_class):
    values = {"KUNO_LTX_MODELS_DIR": str(tmp_path), "KUNO_H3_TURBO_URL": "http://127.0.0.1:30022/"}
    if hardware_class:
        values["KUNO_VERIFIED_HARDWARE_CLASS"] = hardware_class
    config, _ = configure(tmp_path, **values)
    h3 = build_backends(backend, config)["minimax-h3"]
    assert h3.turbo is None and loaded == []
    assert h3.urls == {"fl2va": "http://127.0.0.1:30010", "ref2va": "http://127.0.0.1:30011", "turbo": "http://127.0.0.1:30022"}


def test_the_turbo_pipeline_refuses_to_load_without_its_lora():
    from kuno_protocol.profiles import load_profiles
    from kuno_worker.backends.runtimes import h3_loader

    with pytest.raises(ValueError, match="h3-turbo needs KUNO_H3_TURBO_LORA"):
        h3_loader("/models/MiniMax-H3")(load_profiles()["h3-turbo"])  # before any weights or torch load


# ---------------------------------------------------------------- supervision, with fake processes

FAKE_SGLANG = """\
import os, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
if os.environ.get("FAKE_EXIT"):
    sys.exit(int(os.environ["FAKE_EXIT"]))
port = int(sys.argv[sys.argv.index("--port") + 1])
with open(os.path.join(os.environ["FAKE_DIR"], f"server-{port}.pid"), "w") as handle:
    handle.write(str(os.getpid()))
time.sleep(float(os.environ.get("FAKE_READY_AFTER", "0")))
if os.environ.get("FAKE_DIE_AFTER"):
    threading.Timer(float(os.environ["FAKE_DIE_AFTER"]), lambda: os._exit(9)).start()
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
    def log_message(self, *args):
        pass
HTTPServer(("127.0.0.1", port), Health).serve_forever()
"""

FAKE_WORKER = """\
import os, signal, sys, time, urllib.request
marks = os.environ["FAKE_DIR"]
def mark(name):
    open(os.path.join(marks, name), "w").write(" ".join(sys.argv[1:]))
if not os.environ.get("FAKE_NO_HEALTH"):
    for key in ("KUNO_H3_FL2VA_URL", "KUNO_H3_REF2VA_URL"):
        assert urllib.request.urlopen(os.environ[key] + "/health").status == 200
def terminated(*_):
    mark("worker-terminated")
    sys.exit(0)
signal.signal(signal.SIGTERM, terminated)
mark("worker-started")
if os.environ.get("FAKE_WORKER_EXIT"):
    sys.exit(int(os.environ["FAKE_WORKER_EXIT"]))
while True:
    time.sleep(0.05)
"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def launch(tmp_path):
    binary = tmp_path / "sglang"
    binary.write_text(f"#!{sys.executable}\n{FAKE_SGLANG}")
    binary.chmod(0o755)
    env = {
        "PATH": os.environ.get("PATH", ""), "FAKE_DIR": str(tmp_path), "KUNO_DATA_DIR": str(tmp_path / "no-dev-env"),
        "KUNO_BACKEND": "real", "KUNO_PROFILES": "h3,h3-reference", "KUNO_SGLANG_BIN": str(binary),
        "KUNO_H3_FL2VA_URL": f"http://127.0.0.1:{free_port()}", "KUNO_H3_REF2VA_URL": f"http://127.0.0.1:{free_port()}",
        "KUNO_SGLANG_START_TIMEOUT_S": "30", "KUNO_H3_SHARED_SERVERS": "1",  # two servers, to supervise more than one
    }

    def start(argv=("--profiles", "h3,h3-reference"), stop=None, **overrides):
        return run(list(argv), {**env, **overrides}, worker_command=[sys.executable, "-c", FAKE_WORKER],
                   stop=stop or threading.Event(), poll_s=0.05)

    return start


def server_pids(tmp_path: Path) -> list[int]:
    return [int(path.read_text()) for path in tmp_path.glob("server-*.pid")]


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_the_worker_starts_once_every_server_is_ready_and_the_servers_stop_with_it(launch, tmp_path):
    assert launch(FAKE_READY_AFTER="0.3", FAKE_WORKER_EXIT="7") == 7
    assert (tmp_path / "worker-started").read_text() == "--profiles h3,h3-reference"
    pids = server_pids(tmp_path)
    assert len(pids) == 2 and all(gone(pid) for pid in pids)


def test_a_server_that_dies_before_it_is_ready_fails_without_starting_the_worker(launch, tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        assert launch(FAKE_EXIT="3") == 1
    assert "exited with code 3 before it was ready" in caplog.text
    assert not (tmp_path / "worker-started").exists()


def test_a_server_that_dies_while_serving_stops_the_worker(launch, tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        assert launch(FAKE_DIE_AFTER="0.5") == 1
    assert "server exited with code 9; stopping the worker" in caplog.text
    assert (tmp_path / "worker-terminated").exists()
    assert all(gone(pid) for pid in server_pids(tmp_path))


def test_a_stop_signal_reaches_the_worker_then_the_servers(launch, tmp_path):
    stop = threading.Event()

    def stop_once_started():
        deadline = time.monotonic() + 20
        while not (tmp_path / "worker-started").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        stop.set()

    threading.Thread(target=stop_once_started, daemon=True).start()
    assert launch(stop=stop) == 0
    assert (tmp_path / "worker-terminated").exists()
    assert all(gone(pid) for pid in server_pids(tmp_path))


def test_a_refused_profile_set_starts_nothing(launch, tmp_path):
    with pytest.raises(ValueError, match="would load MiniMax H3 twice"):
        launch(KUNO_H3_SHARED_SERVERS="0")
    assert server_pids(tmp_path) == [] and not (tmp_path / "worker-started").exists()


def test_help_goes_straight_to_the_worker(launch, tmp_path):
    assert launch(argv=["--help"], FAKE_EXIT="3", FAKE_WORKER_EXIT="0", FAKE_NO_HEALTH="1") == 0
    assert server_pids(tmp_path) == []


def test_a_missing_sglang_executable_fails_cleanly(launch, tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        assert launch(KUNO_SGLANG_BIN=str(tmp_path / "missing" / "sglang")) == 1
    assert not (tmp_path / "worker-started").exists()


def test_the_supervisor_module_imports_without_gpu_packages():
    code = textwrap.dedent("""
        import sys, kuno_worker.h3_servers
        assert "torch" not in sys.modules and "sglang" not in sys.modules
    """)
    assert os.system(f'"{sys.executable}" -c \'{code}\'') == 0
