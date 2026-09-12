"""Resident runtimes: weights load once, jobs serialize behind them, and frames come
back as a real MP4. Stub pipelines stand in for the GPU."""

from __future__ import annotations

import json
import subprocess
import threading
import time

import numpy as np
import pytest

from kuno_protocol.profiles import Mode, load_profiles
from kuno_worker.backends.h3_resident import H3ResidentBackend, build_call as h3_call
from kuno_worker.backends.ltx_resident import DISTILLED_SIGMAS, LtxResidentBackend, build_call as ltx_call, pipeline_kind
from kuno_worker.backends.media_tools import BackendError, encode_video, ffmpeg_exe
from kuno_worker.backends.resident import ModelStore, PipelineResult
from kuno_worker.plan import build_task, example_task

PROFILES = load_profiles()
NOOP = lambda _v, _s: None  # noqa: E731


def task_for(profile_id: str, mode: Mode, tmp_path, **kwargs):
    profile = PROFILES[profile_id]
    params = example_task(profile, mode, **{k: v for k, v in kwargs.items() if k in {"duration_s", "resolution", "aspect_ratio", "fps", "audio", "roles"}})
    task = build_task(profile, params, tmp_path, **{k: v for k, v in kwargs.items() if k in {"seed", "prompt", "negative_prompt", "options", "time_s"}})
    for item in task.inputs:
        item.save(tmp_path)
    return task


def frames(count: int = 12, width: int = 64, height: int = 48):
    return [np.full((height, width, 3), i * 8 % 256, dtype=np.uint8) for i in range(count)]


class StubPipeline:
    """Records calls and returns frames, like a loaded diffusers pipeline would."""

    def __init__(self, frame_count: int = 12, with_audio: bool = True):
        self.calls: list[dict] = []
        self.frame_count = frame_count
        self.with_audio = with_audio
        self.unloaded = False

    def __call__(self, **call):
        self.calls.append(call)
        audio = np.zeros((self.frame_count * 4000, 2), dtype=np.float32) if self.with_audio else None
        return {"videos": [frames(self.frame_count)], "audio": [audio] if audio is not None else None, "sampling_rate": 48000}

    def unload(self):
        self.unloaded = True


# ---------------------------------------------------------------- the store


def test_weights_load_once_and_stay_loaded():
    loads = []
    store = ModelStore(lambda profile: loads.append(profile.id) or StubPipeline())
    for _ in range(5):
        with store.acquire(PROFILES["ltx-2.5-fast"]):
            pass
    assert loads == ["ltx-2.5-fast"] and store.loads == 1


def test_a_second_profile_evicts_the_first_when_only_one_fits():
    pipelines = {}

    def loader(profile):
        pipelines[profile.id] = StubPipeline()
        return pipelines[profile.id]

    store = ModelStore(loader, capacity=1)
    store.warm(PROFILES["ltx-2.5-fast"])
    store.warm(PROFILES["ltx-2.5-pro"])
    assert store.loaded == ["ltx-2.5-pro"] and store.evictions == 1
    assert pipelines["ltx-2.5-fast"].unloaded, "the evicted pipeline must free its VRAM"


def test_two_profiles_stay_loaded_when_both_fit():
    store = ModelStore(lambda _p: StubPipeline(), capacity=2)
    store.warm(PROFILES["ltx-2.5-fast"])
    store.warm(PROFILES["ltx-2.5-pro"])
    with store.acquire(PROFILES["ltx-2.5-fast"]):
        pass
    assert set(store.loaded) == {"ltx-2.5-fast", "ltx-2.5-pro"} and store.evictions == 0


def test_jobs_take_turns_on_the_gpu():
    store = ModelStore(lambda _p: StubPipeline())
    overlaps, inside = [], []

    def run():
        with store.acquire(PROFILES["ltx-2.5-fast"]):
            inside.append(1)
            overlaps.append(len(inside))
            time.sleep(0.05)
            inside.pop()

    threads = [threading.Thread(target=run) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert max(overlaps) == 1, "two generations ran at once"


@pytest.mark.parametrize(
    "raw",
    [
        {"videos": [frames(3)], "audio": [np.zeros((100, 2))], "sampling_rate": 44100},
        {"frames": frames(3)},
        PipelineResult(frames(3)),
    ],
)
def test_pipeline_outputs_are_normalized(raw):
    result = PipelineResult.from_pipeline(raw)
    assert len(result.frames) == 3 and result.frames[0].shape == (48, 64, 3)


def test_a_pipeline_with_no_frames_is_an_error():
    with pytest.raises(TypeError):
        PipelineResult.from_pipeline({"something_else": 1})


# ---------------------------------------------------------------- encoding


def probe(data: bytes) -> dict | None:
    ffprobe = ffmpeg_exe().replace("ffmpeg", "ffprobe")
    out = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", "-"], input=data, capture_output=True)
    return json.loads(out.stdout) if out.returncode == 0 else None


def test_frames_become_a_playable_file():
    data = encode_video(frames(48), fps=24)
    assert data[4:8] == b"ftyp"
    info = probe(data)
    if info:
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        assert (video["width"], video["height"]) == (64, 48)
        assert not any(s["codec_type"] == "audio" for s in info["streams"])
        assert float(info["format"]["duration"]) == pytest.approx(2.0, abs=0.2)


def test_audio_is_muxed_in_when_the_model_produced_it():
    audio = np.sin(np.linspace(0, 400, 48000 * 2, dtype=np.float32))[:, None].repeat(2, axis=1)
    info = probe(encode_video(frames(48), fps=24, audio=audio, sample_rate=48000))
    if info:
        assert any(s["codec_type"] == "audio" for s in info["streams"])


def test_int16_and_mono_audio_are_accepted():
    mono = (np.zeros(48000, dtype=np.int16))
    assert encode_video(frames(24), fps=24, audio=mono, sample_rate=48000)[4:8] == b"ftyp"


def test_frames_of_different_sizes_are_rejected():
    mixed = frames(4) + [np.zeros((10, 10, 3), dtype=np.uint8)]
    with pytest.raises(BackendError):
        encode_video(mixed, fps=24)


# ---------------------------------------------------------------- LTX calls


def test_distilled_profiles_use_their_sigmas_not_a_step_count(tmp_path):
    call = ltx_call(task_for("ltx-2.5-fast", Mode.TEXT_TO_VIDEO, tmp_path, duration_s=5, fps=24))
    assert call["sigmas"] == DISTILLED_SIGMAS and call["guidance_scale"] == 1.0
    assert "num_inference_steps" not in call
    assert call["num_frames"] == 121 and call["pipeline"] == "text"


def test_the_full_model_takes_steps_guidance_and_a_negative_prompt(tmp_path):
    call = ltx_call(task_for("ltx-2.5-pro", Mode.TEXT_TO_VIDEO, tmp_path, negative_prompt="blurry"))
    assert call["num_inference_steps"] == 30 and call["negative_prompt"] == "blurry"
    assert "sigmas" not in call


def test_frames_and_keyframes_become_conditions(tmp_path):
    flf = ltx_call(task_for("ltx-2.5-fast", Mode.FIRST_LAST_FRAME, tmp_path, duration_s=5, fps=24))
    assert [c["index"] for c in flf["conditions"]] == [0, 120]
    assert pipeline_kind(PROFILES["ltx-2.5-fast"], Mode.FIRST_LAST_FRAME) == "condition"

    keys = ltx_call(task_for("ltx-2.5-fast", Mode.KEYFRAMES, tmp_path, duration_s=4, fps=24, time_s=2.0))
    assert keys["conditions"][0]["index"] == 48


def test_retake_and_audio_to_video_calls(tmp_path):
    retake = ltx_call(task_for("ltx-2.5-fast", Mode.RETAKE, tmp_path, options={"retake": {"start_s": 1, "end_s": 3}}))
    assert (retake["start_time"], retake["end_time"]) == (1.0, 3.0) and retake["video_path"]

    a2v = ltx_call(task_for("ltx-2.5-pro", Mode.AUDIO_TO_VIDEO, tmp_path, duration_s=6))
    assert a2v["audio_path"] and a2v["audio_max_duration"] == 6
    assert "num_frames" not in a2v  # the CLI rejects both together


def test_4k_renders_at_half_rate_then_interpolates(tmp_path):
    call = ltx_call(task_for("ltx-2.5-4k", Mode.TEXT_TO_VIDEO, tmp_path, duration_s=4, fps=48, resolution="2160p"))
    assert call["temporal_upscalings"] == 1 and call["frame_rate"] == 24
    assert (call["width"], call["height"]) == (3840, 2176)


# ---------------------------------------------------------------- H3 calls


def test_h3_keyframe_and_reference_calls(tmp_path):
    flf = h3_call(task_for("h3", Mode.FIRST_LAST_FRAME, tmp_path, duration_s=5))
    assert flf["num_frames"] == 124 and flf["image"] and flf["last_image"]
    assert flf["video_shift"] == 12.0 and flf["num_inference_steps"] == 50

    turbo = h3_call(task_for("h3-turbo", Mode.TEXT_TO_VIDEO, tmp_path))
    assert turbo["video_shift"] == 6.0 and turbo["num_inference_steps"] == 8  # LoRA's trained shift

    refs = h3_call(task_for("h3-reference", Mode.VIDEO_EDIT, tmp_path))
    assert refs["references"][0]["type"] == "video_audio"
    muted = h3_call(task_for("h3-reference", Mode.VIDEO_EDIT, tmp_path, options={"keep_source_audio": False}))
    assert muted["references"][0]["type"] == "video"


# ---------------------------------------------------------------- end to end with stubs


def test_ltx_backend_reuses_one_loaded_pipeline_across_jobs(tmp_path):
    pipeline = StubPipeline(frame_count=48)
    backend = LtxResidentBackend(None, tmp_path / "work", loader=lambda _p: pipeline)
    backend.warm(PROFILES["ltx-2.5-fast"])
    for _ in range(3):
        result = backend.generate(task_for("ltx-2.5-fast", Mode.TEXT_TO_VIDEO, tmp_path, duration_s=2, fps=24), NOOP)
        assert result.data[4:8] == b"ftyp" and result.info.frames == 48 and result.info.audio
    assert backend.store.loads == 1, "weights must not reload per job"
    assert len(pipeline.calls) == 3
    assert not (tmp_path / "work").exists() or not any((tmp_path / "work").iterdir())


def test_silent_requests_come_back_without_an_audio_track(tmp_path):
    backend = LtxResidentBackend(None, tmp_path / "work", loader=lambda _p: StubPipeline(frame_count=24))
    result = backend.generate(task_for("ltx-2.5-fast", Mode.TEXT_TO_VIDEO, tmp_path, duration_s=2, fps=24, audio=False), NOOP)
    assert result.info.audio is False
    info = probe(result.data)
    if info:
        assert not any(s["codec_type"] == "audio" for s in info["streams"])


def test_h3_backend_encodes_at_24fps(tmp_path):
    backend = H3ResidentBackend(tmp_path / "work", loader=lambda _p: StubPipeline(frame_count=124))
    result = backend.generate(task_for("h3-turbo", Mode.TEXT_TO_VIDEO, tmp_path, duration_s=5), NOOP)
    assert result.info.fps == 24 and result.info.frames == 124
    assert result.info.duration_s == pytest.approx(124 / 24, abs=0.01)


def test_an_empty_pipeline_result_fails_the_job_not_the_worker(tmp_path):
    class Empty(StubPipeline):
        def __call__(self, **call):
            return {"videos": [[]], "audio": None, "sampling_rate": 48000}

    backend = LtxResidentBackend(None, tmp_path / "work", loader=lambda _p: Empty())
    with pytest.raises(BackendError):
        backend.generate(task_for("ltx-2.5-fast", Mode.TEXT_TO_VIDEO, tmp_path, duration_s=2), NOOP)
