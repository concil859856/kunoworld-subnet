"""The H3 and LTX backends have never run on a GPU. These tests pin down exactly
what they would send to each official runtime, and run both against fake runtimes
so the plumbing (files in, video out, audio stripping, cleanup) is exercised."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from kuno_protocol.profiles import InputRole, Mode, load_profiles
from kuno_worker.backends.base import Backend
from kuno_worker.backends.h3 import H3SglangBackend, build_sglang_request
from kuno_worker.backends.ltx import LtxPaths, build_command, pick_pipeline
from kuno_worker.backends.media_tools import ffmpeg_exe
from kuno_worker.plan import build_task, example_task

PROFILES = load_profiles()
NOOP = lambda _value, _stage: None  # noqa: E731


def task_for(profile_id: str, mode: Mode, tmp_path: Path, **kwargs):
    profile = PROFILES[profile_id]
    params = example_task(profile, mode, **{k: v for k, v in kwargs.items() if k in {"duration_s", "resolution", "aspect_ratio", "fps", "audio", "roles"}})
    task = build_task(profile, params, tmp_path, **{k: v for k, v in kwargs.items() if k in {"seed", "prompt", "negative_prompt", "options", "time_s"}})
    for item in task.inputs:
        item.save(tmp_path)
    return task


# ---------------------------------------------------------------- MiniMax H3 requests


@pytest.mark.parametrize(
    ("mode", "expected_task", "frame_indexes"),
    [
        (Mode.TEXT_TO_VIDEO, "t2va", []),
        (Mode.IMAGE_TO_VIDEO, "fl2va", [0]),
        (Mode.LAST_FRAME, "fl2va", [-1]),
        (Mode.FIRST_LAST_FRAME, "fl2va", [0, -1]),
    ],
)
def test_h3_keyframe_requests(mode, expected_task, frame_indexes, tmp_path):
    task = task_for("h3", mode, tmp_path)
    variant, body = build_sglang_request(task, tmp_path)
    assert variant == "fl2va"  # both tasks are served by the FL2VA checkpoint
    assert body["task"] == expected_task
    assert [c["frame_index"] for c in body["conditions"]] == frame_indexes
    assert all(c["type"] == "image" and c["role"] == "keyframe" for c in body["conditions"])
    assert all(c["uri"].startswith("file://") for c in body["conditions"])
    assert body["target"] == {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0}
    assert body["num_inference_steps"] == 51 and body["seed"] == 42  # 50 passes: the grid includes the terminal 0


@pytest.mark.parametrize(("mode", "expected_task"), [(Mode.TEXT_TO_VIDEO, "t2va"), (Mode.FIRST_LAST_FRAME, "fl2va")])
def test_h3_turbo_requests_go_to_the_turbo_server_with_the_loras_shifts_and_pass_count(mode, expected_task, tmp_path):
    task = task_for("h3-turbo", mode, tmp_path, duration_s=14)
    server, body = build_sglang_request(task, tmp_path)
    assert server == "turbo" and body["task"] == expected_task
    # The 8-step LoRA is trained on the 9-point grid, and with video shift 6 and audio shift 3 (measured 2026-09-16).
    assert body["num_inference_steps"] == 9
    assert (body["flow_shift"], body["audio_flow_shift"]) == (6.0, 3.0)
    assert body["target"] == {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 14.0}


@pytest.mark.parametrize(("profile_id", "mode"), [("h3", Mode.TEXT_TO_VIDEO), ("h3-reference", Mode.REFERENCE_TO_VIDEO)])
def test_full_h3_requests_leave_the_shifts_to_the_server(profile_id, mode, tmp_path):
    _, body = build_sglang_request(task_for(profile_id, mode, tmp_path), tmp_path)
    assert "flow_shift" not in body and "audio_flow_shift" not in body and body["num_inference_steps"] == 51


def test_h3_reference_request_keeps_customer_order_and_maps_types(tmp_path):
    roles = [InputRole.REFERENCE_IMAGE, InputRole.REFERENCE_VIDEO, InputRole.REFERENCE_AUDIO]
    task = task_for("h3-reference", Mode.REFERENCE_TO_VIDEO, tmp_path, roles=roles)
    variant, body = build_sglang_request(task, tmp_path)
    assert variant == "ref2va" and body["task"] == "ref2va"
    # Order matters: it sets the <Picture n> / <Video n> / <Audio n> labels the prompt refers to.
    assert [c["type"] for c in body["conditions"]] == ["image", "video", "audio"]
    assert all(c["role"] == "reference" for c in body["conditions"])


def test_h3_source_video_carries_its_audio_unless_turned_off(tmp_path):
    task = task_for("h3-reference", Mode.VIDEO_EDIT, tmp_path)
    _, body = build_sglang_request(task, tmp_path)
    assert body["conditions"][0]["type"] == "video_audio"

    muted = task_for("h3-reference", Mode.VIDEO_EDIT, tmp_path, options={"keep_source_audio": False})
    _, body = build_sglang_request(muted, tmp_path)
    assert body["conditions"][0]["type"] == "video"


def test_h3_aspect_ratios_stay_inside_the_pixel_budget():
    profile = PROFILES["h3"]
    for aspect, (width, height) in profile.limits.sizes["768p"].items():
        assert width % 32 == 0 and height % 32 == 0, aspect
        assert width * height <= 1_032_192, f"{aspect} exceeds H3's pixel cap"


# ---------------------------------------------------------------- LTX-2.5 commands


def argv_value(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


@pytest.mark.parametrize(
    ("profile_id", "mode", "pipeline"),
    [
        ("ltx-2.5-fast", Mode.TEXT_TO_VIDEO, "distilled"),
        ("ltx-2.5-fast", Mode.KEYFRAMES, "distilled"),
        ("ltx-2.5-fast", Mode.RETAKE, "retake"),
        ("ltx-2.5-pro", Mode.TEXT_TO_VIDEO, "ti2vid_two_stages"),
        ("ltx-2.5-pro", Mode.FIRST_LAST_FRAME, "keyframe_interpolation"),
        ("ltx-2.5-pro", Mode.AUDIO_TO_VIDEO, "a2vid_two_stage"),
        ("ltx-2.5-4k", Mode.TEXT_TO_VIDEO, "dfr_pipeline"),
    ],
)
def test_ltx_pipeline_selection_and_weights(profile_id, mode, pipeline, tmp_path):
    task = task_for(profile_id, mode, tmp_path)
    argv, _, _ = build_command(task, LtxPaths(Path("/models")), tmp_path, tmp_path / "out.mp4")
    assert pick_pipeline(task) == pipeline
    assert argv[2] == f"ltx_pipelines.{pipeline}"
    transformer = argv_value(argv, "--transformer-path")
    expected = "dev" if pipeline in {"ti2vid_two_stages", "keyframe_interpolation", "a2vid_two_stage"} else "distilled"
    assert f"22b-{expected}-transformer" in transformer
    # The dev pipelines need the distilled LoRA for their second stage.
    assert ("--distilled-lora" in argv) is (expected == "dev")


def test_ltx_frame_counts_and_image_positions(tmp_path):
    task = task_for("ltx-2.5-fast", Mode.FIRST_LAST_FRAME, tmp_path, duration_s=5, fps=24)
    argv, frames, fps = build_command(task, LtxPaths(Path("/models")), tmp_path, tmp_path / "out.mp4")
    assert frames == 121 and (frames - 1) % 8 == 0 and fps == 24
    assert argv_value(argv, "--num-frames") == "121"
    positions = [argv[i + 2] for i, a in enumerate(argv) if a == "--image"]
    assert positions == ["0", "120"]  # first frame at 0, last frame at the final index


def test_ltx_keyframe_times_become_frame_indexes(tmp_path):
    task = task_for("ltx-2.5-fast", Mode.KEYFRAMES, tmp_path, duration_s=4, fps=24, time_s=2.0)
    argv, frames, _ = build_command(task, LtxPaths(Path("/models")), tmp_path, tmp_path / "out.mp4")
    assert argv[argv.index("--image") + 2] == "48"  # 2.0 s × 24 fps
    assert int(argv[argv.index("--image") + 2]) < frames


def test_ltx_retake_and_audio_to_video_flags(tmp_path):
    retake = task_for("ltx-2.5-fast", Mode.RETAKE, tmp_path, options={"retake": {"start_s": 2, "end_s": 4}})
    argv, _, _ = build_command(retake, LtxPaths(Path("/models")), tmp_path, tmp_path / "out.mp4")
    assert argv_value(argv, "--start-time") == "2" and argv_value(argv, "--end-time") == "4"
    assert argv_value(argv, "--video-path", ) is not None
    assert "--spatial-upsampler-path" not in argv  # retake works on the source resolution

    a2v = task_for("ltx-2.5-pro", Mode.AUDIO_TO_VIDEO, tmp_path, duration_s=6)
    argv, _, _ = build_command(a2v, LtxPaths(Path("/models")), tmp_path, tmp_path / "out.mp4")
    # The official CLI rejects --num-frames together with --audio-max-duration.
    assert "--num-frames" not in argv
    assert argv_value(argv, "--audio-max-duration") == "6" and argv_value(argv, "--audio-path")


def test_ltx_4k_doubles_playback_fps_with_temporal_upscaling(tmp_path):
    task = task_for("ltx-2.5-4k", Mode.TEXT_TO_VIDEO, tmp_path, duration_s=4, fps=48, resolution="2160p")
    argv, frames, fps = build_command(task, LtxPaths(Path("/models")), tmp_path, tmp_path / "out.mp4")
    assert argv_value(argv, "--temporal-upscalings") == "1"
    assert argv_value(argv, "--frame-rate") == "24" and fps == 48  # rendered at 24, played at 48
    assert frames == (int(argv_value(argv, "--num-frames")) - 1) * 2 + 1
    assert argv_value(argv, "--width") == "3840" and argv_value(argv, "--height") == "2176"


def test_ltx_negative_prompt_only_reaches_the_full_model(tmp_path):
    pro = task_for("ltx-2.5-pro", Mode.TEXT_TO_VIDEO, tmp_path, negative_prompt="blurry, watermark")
    argv, _, _ = build_command(pro, LtxPaths(Path("/models")), tmp_path, tmp_path / "out.mp4")
    assert argv_value(argv, "--negative-prompt") == "blurry, watermark"

    fast = task_for("ltx-2.5-fast", Mode.TEXT_TO_VIDEO, tmp_path, negative_prompt="blurry")
    argv, _, _ = build_command(fast, LtxPaths(Path("/models")), tmp_path, tmp_path / "out.mp4")
    assert "--negative-prompt" not in argv  # distilled sigmas ignore guidance


# ---------------------------------------------------------------- against fake runtimes


@pytest.fixture(scope="module")
def sample_mp4(tmp_path_factory) -> bytes:
    path = tmp_path_factory.mktemp("media") / "sample.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x176:rate=24:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1", "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True,
    )
    return path.read_bytes()


@pytest.fixture
def fake_sglang(sample_mp4):
    """Stands in for the official SGLang video server."""
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            self._send(200, json.dumps({"id": "vid-1"}).encode(), "application/json")

        def do_GET(self):
            if self.path.endswith("/content"):
                self._send(200, sample_mp4, "video/mp4")
            else:
                self._send(200, json.dumps({"id": "vid-1", "status": "completed", "progress": 100}).encode(), "application/json")

        def _send(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", received
    server.shutdown()


def test_h3_backend_generates_and_cleans_up(fake_sglang, tmp_path):
    url, received = fake_sglang
    workdir = tmp_path / "work"
    backend = H3SglangBackend(url, url, workdir)
    task = task_for("h3", Mode.IMAGE_TO_VIDEO, tmp_path / "inputs", duration_s=5)
    result = backend.generate(task, NOOP)

    assert result.data[4:8] == b"ftyp"
    assert received[0]["task"] == "fl2va"
    assert result.info.frames == 124 and result.info.fps == 24  # H3's 17n+5 grid
    assert result.info.audio is True
    assert not (workdir / task.job_id).exists()  # customer media never lingers on disk


def test_h3_backend_strips_audio_when_the_customer_turned_it_off(fake_sglang, tmp_path):
    url, _ = fake_sglang
    backend = H3SglangBackend(url, url, tmp_path / "work")
    task = task_for("h3", Mode.TEXT_TO_VIDEO, tmp_path / "inputs", audio=False)
    result = backend.generate(task, NOOP)
    assert result.info.audio is False
    probe = subprocess.run(
        [ffmpeg_exe().replace("ffmpeg", "ffprobe"), "-v", "error", "-show_streams", "-of", "json", "-"],
        input=result.data, capture_output=True,
    )
    if probe.returncode == 0:  # ffprobe ships with system ffmpeg, not with imageio-ffmpeg
        assert not any(s["codec_type"] == "audio" for s in json.loads(probe.stdout)["streams"])


class InProcessTurbo(Backend):
    """Stands in for H3ResidentBackend: records what reaches it."""

    name = "in-process"

    def __init__(self, hardware_class: str | None):
        self.hardware_class = hardware_class
        self.jobs: list[str] = []
        self.warmed: list[str] = []

    def warm(self, profile):
        self.warmed.append(profile.id)

    def generate(self, task, progress):
        self.jobs.append(task.profile.id)
        return "in-process result"


DEAD_URL = "http://127.0.0.1:9"  # nothing listens: a job sent here fails


def test_h3_turbo_jobs_go_to_the_turbo_server(fake_sglang, tmp_path):
    url, received = fake_sglang
    backend = H3SglangBackend(DEAD_URL, DEAD_URL, tmp_path / "work", turbo_url=url + "/")
    result = backend.generate(task_for("h3-turbo", Mode.IMAGE_TO_VIDEO, tmp_path / "inputs", duration_s=5), NOOP)
    assert result.data[4:8] == b"ftyp" and result.info.frames == 124
    assert received[0]["flow_shift"] == 6.0 and received[0]["num_inference_steps"] == 9
    assert not backend.verified_enabled(PROFILES["h3-turbo"])


def test_verified_h3_turbo_runs_in_process_and_everything_else_on_the_servers(fake_sglang, tmp_path):
    url, received = fake_sglang
    turbo = InProcessTurbo("C2.h200-141gb.x1")
    backend = H3SglangBackend(url, url, tmp_path / "work", turbo_url=DEAD_URL, turbo=turbo)
    for profile_id in ("h3-turbo", "h3", "h3-reference"):
        backend.warm(PROFILES[profile_id])
    assert backend.generate(task_for("h3-turbo", Mode.TEXT_TO_VIDEO, tmp_path / "turbo"), NOOP) == "in-process result"
    backend.generate(task_for("h3", Mode.TEXT_TO_VIDEO, tmp_path / "h3"), NOOP)
    assert turbo.jobs == ["h3-turbo"] and turbo.warmed == ["h3-turbo"] and len(received) == 1
    assert [backend.verified_enabled(PROFILES[p]) for p in ("h3-turbo", "h3", "h3-reference")] == [True, False, False]


def test_an_in_process_pipeline_on_a_class_turbo_does_not_pin_leaves_turbo_on_its_server(fake_sglang, tmp_path):
    url, received = fake_sglang
    turbo = InProcessTurbo("C4.h200-141gb.x4.ulysses4")  # h3 and h3-reference's class; h3-turbo pins the x1 ones
    backend = H3SglangBackend(DEAD_URL, DEAD_URL, tmp_path / "work", turbo_url=url, turbo=turbo)
    backend.warm(PROFILES["h3-turbo"])
    backend.generate(task_for("h3-turbo", Mode.TEXT_TO_VIDEO, tmp_path / "inputs"), NOOP)
    assert turbo.jobs == [] and turbo.warmed == [] and len(received) == 1


def test_ltx_backend_runs_the_pipeline_module_and_returns_its_video(tmp_path, monkeypatch, sample_mp4):
    """A stub `ltx_pipelines` package writes the file the real CLI would write."""
    stub = tmp_path / "stub"
    (stub / "ltx_pipelines").mkdir(parents=True)
    (stub / "ltx_pipelines" / "__init__.py").write_text("")
    (stub / "ltx_pipelines" / "distilled.py").write_text(
        "import sys, shutil\n"
        "argv = sys.argv[1:]\n"
        "out = argv[argv.index('--output-path') + 1]\n"
        f"shutil.copy({str(tmp_path / 'sample.mp4')!r}, out)\n"
        "open(out + '.argv', 'w').write('\\n'.join(argv))\n"
    )
    (tmp_path / "sample.mp4").write_bytes(sample_mp4)
    monkeypatch.setenv("PYTHONPATH", str(stub))

    from kuno_worker.backends.ltx import LtxPipelinesBackend

    backend = LtxPipelinesBackend(Path("/models"), tmp_path / "work")
    task = task_for("ltx-2.5-fast", Mode.TEXT_TO_VIDEO, tmp_path / "inputs", duration_s=3, fps=24)
    result = backend.generate(task, NOOP)

    assert result.data[4:8] == b"ftyp"
    assert result.info.frames == 73 and result.info.duration_s == pytest.approx(73 / 24, abs=0.01)
    assert not (tmp_path / "work" / task.job_id).exists()


def test_ltx_backend_reports_a_failing_pipeline_without_leaking_the_prompt(tmp_path, monkeypatch):
    stub = tmp_path / "stub"
    (stub / "ltx_pipelines").mkdir(parents=True)
    (stub / "ltx_pipelines" / "__init__.py").write_text("")
    (stub / "ltx_pipelines" / "distilled.py").write_text("import sys; print(sys.argv); sys.exit(3)\n")
    monkeypatch.setenv("PYTHONPATH", str(stub))

    from kuno_worker.backends.ltx import LtxPipelinesBackend
    from kuno_worker.backends.media_tools import BackendError

    backend = LtxPipelinesBackend(Path("/models"), tmp_path / "work")
    task = task_for("ltx-2.5-fast", Mode.TEXT_TO_VIDEO, tmp_path / "inputs", prompt="secret client campaign")
    with pytest.raises(BackendError) as exc:
        backend.generate(task, NOOP)
    assert "secret client campaign" not in str(exc.value)
    assert "3" in str(exc.value)


def test_every_profile_and_mode_builds_a_command(tmp_path):
    """No profile/mode pair the gateway would accept can crash a backend builder."""
    for profile in PROFILES.values():
        for mode in profile.modes:
            task = task_for(profile.id, mode, tmp_path / profile.id / mode.value)
            if profile.family == "minimax-h3":
                _, body = build_sglang_request(task, tmp_path)
                assert body["prompt"] and body["target"]["duration_seconds"] > 0
            else:
                argv, frames, fps = build_command(task, LtxPaths(Path("/models")), tmp_path, tmp_path / "out.mp4")
                assert argv[0] == sys.executable and frames > 0 and fps > 0
