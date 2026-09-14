"""kuno-h3-worker: which SGLang servers the H3 image starts for which profiles, and how it supervises them
with the worker. Fake servers and a fake worker stand in; none of this has run against SGLang or a GPU."""

from __future__ import annotations

import logging
import os
import socket
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from kuno_worker.backends import build_backends
from kuno_worker.config import WorkerConfig
from kuno_worker.h3_servers import plan_servers, run, server_env


def configure(tmp_path: Path, **values: str) -> tuple[WorkerConfig, dict[str, str]]:
    env = {"KUNO_DATA_DIR": str(tmp_path / "no-dev-env"), "KUNO_BACKEND": "real", **values}
    return WorkerConfig.from_env(env), env


# ---------------------------------------------------------------- which servers


def test_h3_needs_only_the_fl2va_server_run_with_the_official_recipe(tmp_path):
    config, env = configure(tmp_path, KUNO_PROFILES="h3")
    [server] = plan_servers(config, env)
    assert (server.variant, server.port) == ("fl2va", 30010)
    assert server.argv == [
        "sglang", "serve", "--model-path", "MiniMaxAI/MiniMax-H3", "--model-variant", "fl2va",
        "--num-gpus", "4", "--ulysses-degree", "4", "--performance-mode", "speed",
        "--host", "127.0.0.1", "--port", "30010", "--master-port", "31010", "--scheduler-port", "32010",
    ]


def test_each_checkpoint_variant_the_profiles_route_to_gets_one_server(tmp_path):
    config, env = configure(tmp_path, KUNO_PROFILES="h3-turbo,h3,h3-reference,ltx-2.5-fast")
    assert [(s.variant, s.port) for s in plan_servers(config, env)] == [("fl2va", 30010), ("ref2va", 30011)]
    config, env = configure(tmp_path, KUNO_PROFILES="h3-reference")
    assert [s.variant for s in plan_servers(config, env)] == ["ref2va"]


@pytest.mark.parametrize(("profiles", "backend"), [("h3-turbo", "real"), ("ltx-2.5-fast,ltx-2.5-pro", "real"), ("h3,h3-reference", "mock")])
def test_no_server_starts_when_no_profile_is_served_by_sglang(tmp_path, profiles, backend):
    config, env = configure(tmp_path, KUNO_PROFILES=profiles, KUNO_BACKEND=backend)
    assert plan_servers(config, env) == []


def test_the_second_worker_of_a_whole_server_td_gets_its_own_ports(tmp_path):
    config, env = configure(tmp_path, KUNO_PROFILES="h3,h3-reference",
                            KUNO_H3_FL2VA_URL="http://127.0.0.1:30020", KUNO_H3_REF2VA_URL="http://127.0.0.1:30021")
    ports = [(s.port, s.argv[s.argv.index("--master-port") + 1], s.argv[s.argv.index("--scheduler-port") + 1]) for s in plan_servers(config, env)]
    assert ports == [(30020, "31020", "32020"), (30021, "31021", "32021")]


def test_operator_settings_reach_the_server_command(tmp_path):
    config, env = configure(tmp_path, KUNO_PROFILES="h3", KUNO_SGLANG_BIN="/opt/sglang/bin/sglang", KUNO_H3_MODEL_ID="/models/MiniMax-H3",
                            KUNO_H3_NUM_GPUS="8", KUNO_SGLANG_ARGS="--tp-size 2 --dit-cpu-offload 'false'")
    argv = plan_servers(config, env)[0].argv
    assert argv[:4] == ["/opt/sglang/bin/sglang", "serve", "--model-path", "/models/MiniMax-H3"]
    assert argv[argv.index("--num-gpus") + 1] == "8" == argv[argv.index("--ulysses-degree") + 1]
    assert argv[-4:] == ["--tp-size", "2", "--dit-cpu-offload", "false"]


@pytest.mark.parametrize("url", ["http://10.0.0.5:30010", "http://127.0.0.1", "http://localhost:30010"])
def test_a_server_started_here_must_listen_on_loopback_at_an_explicit_port(tmp_path, url):
    config, env = configure(tmp_path, KUNO_PROFILES="h3", KUNO_H3_FL2VA_URL=url)
    with pytest.raises(ValueError, match="KUNO_H3_FL2VA_URL"):
        plan_servers(config, env)


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


def test_the_turbo_pipeline_loads_the_weights_the_servers_load(tmp_path, monkeypatch):
    from kuno_worker.backends import runtimes

    seen = []

    def fake_loader(model_id, **_):
        seen.append(model_id)
        return lambda profile: None

    monkeypatch.setattr(runtimes, "h3_loader", fake_loader)
    config, _ = configure(tmp_path, KUNO_H3_MODEL_ID="/models/MiniMax-H3", KUNO_LTX_MODELS_DIR=str(tmp_path))
    build_backends("real", config)
    assert seen == ["/models/MiniMax-H3"]


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
        "KUNO_SGLANG_START_TIMEOUT_S": "30",
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
