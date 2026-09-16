"""Storyboard jobs through the worker without a GPU: the resident backend renders shot after shot through a stand-in
renderer and stitches the protocol's frame count, reports `shot i/N`, stops between shots on cancel, admits memory by the
longest shot and never commits steps; the mock backend makes the same video for dev networks; the worker refuses a
malformed shot list and runs the safety gate on every shot; kuno-plan shows the joins."""

from __future__ import annotations

import json
import uuid

import pytest

from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import SenderSession
from kuno_protocol.mp4 import probe
from kuno_protocol.profiles import Mode, load_profiles, shot_prompt, storyboard_duration_s, storyboard_frames
from kuno_protocol.schemas import GenerationParams, MinerJob, SealedPayload, ShotPrompt, ShotSpec, job_aad
from kuno_worker import safety, worker as worker_module
from kuno_worker.backends.base import Backend, GenerationTask
from kuno_worker.backends.ltx_resident import LtxResidentBackend
from kuno_worker.backends.ltx_storyboard import StoryboardError
from kuno_worker.backends.media_tools import CapacityRefused, ffmpeg_exe
from kuno_worker.backends.mock import MockBackend
from kuno_worker.backends.quantized import longest_duration
from kuno_worker.plan import build_task, example_task, parse_shots, storyboard_plan
from kuno_worker.safety import SafetyGate, SafetyViolation
from kuno_worker.safety_frames import FRAMES_PER_SHOT, RequestSignals, plan_frame_indices
from kuno_worker.verified import RetentionStore
from kuno_worker.worker import JobCanceled, JobRejected

from ltx_storyboard_doubles import StubRenderer
from test_worker_provenance import make_worker

pytestmark = pytest.mark.skipif(ffmpeg_exe() is None, reason="ffmpeg encodes the stitched video")

FAST = load_profiles()["ltx-2.5-fast"]
LTX_CLASS = "C1.rtx-pro-6000-bw-se.x1"
RTX5090 = "O1.rtx-5090-32gb.x1.fp8-cast"
SCENE = "An old harbor at golden hour, a blue fishing boat by the dock"
SHOT_PROMPTS = ["The boat pulls away from the dock", "It rounds the breakwater", "A gull lands on a post"]


def shots(*spec: tuple[float, str]) -> list[ShotSpec]:
    return [ShotSpec(duration_s=duration, join=join) for duration, join in spec]


def board(shot_list: list[ShotSpec], fps: int = 24, resolution: str = "720p", aspect_ratio: str = "16:9", audio: bool = True) -> GenerationParams:
    return GenerationParams(
        profile_id=FAST.id, mode=Mode.STORYBOARD, duration_s=storyboard_duration_s(FAST, shot_list, fps), resolution=resolution,
        aspect_ratio=aspect_ratio, fps=fps, audio=audio, shots=shot_list,
    )


def storyboard_task(params: GenerationParams, seed: int = 7, width: int = 320, height: int = 192, prompts: list[str] | None = None) -> GenerationTask:
    """A small frame size keeps the stitched video cheap; build_call takes the task's size."""
    prompts = prompts or [shot_prompt(SCENE, p) for p in SHOT_PROMPTS[: len(params.shots)]]
    return GenerationTask(
        job_id=str(uuid.uuid4()), profile=FAST, params=params, prompt=SCENE, negative_prompt=None, seed=seed, width=width,
        height=height, shot_prompts=prompts,
    )


def backend_with(tmp_path, renderer, **kwargs) -> LtxResidentBackend:
    loads = []
    backend = LtxResidentBackend(
        None, tmp_path / "work", loader=lambda profile: loads.append(profile.id) or object(),
        storyboard_renderer=lambda loaded, profile: renderer, **kwargs,
    )
    backend.loads = loads
    return backend


# ---------------------------------------------------------------- the resident backend


def test_a_storyboard_renders_every_shot_and_stitches_the_protocols_frame_count(tmp_path):
    renderer, store = StubRenderer(), RetentionStore()
    # A class that runs verified mode for this profile's other jobs: storyboards still carry no commitment.
    backend = backend_with(tmp_path, renderer, hardware_class=LTX_CLASS, retention=store, model_digest="d" * 64)
    params = board(shots((2, "fresh"), (3, "continue"), (2, "cut")))
    task = storyboard_task(params, seed=2**31 - 1)
    reports = []
    result = backend.generate(task, lambda value, stage: reports.append((value, stage)))

    frames = storyboard_frames(FAST, params.shots, 24)
    assert frames == 49 + (73 - 17) + (49 - 17)
    assert (result.info.frames, result.info.duration_s, result.info.fps, result.info.audio) == (frames, round(frames / 24, 3), 24.0, True)
    assert (result.info.width, result.info.height) == (320, 192)
    assert probe(result.data).frames == frames
    assert result.step_commitment is None and result.openings is None and task.job_id not in store

    assert [stage for _, stage in reports] == ["shot 1/3", "shot 2/3", "shot 3/3", "encoding", "encoded"]
    values = [value for value, _ in reports]
    assert values == sorted(values) and values[-1] == 1.0

    # Each shot is its own text-to-video call: its length, its model prompt, seed + index mod 2^31, both distilled passes.
    assert [c["num_frames"] for c in renderer.calls] == [49, 73, 49]
    assert [c["prompt"] for c in renderer.calls] == task.shot_prompts
    assert [c["seed"] for c in renderer.calls] == [2**31 - 1, 0, 1]
    assert all(c["pipeline"] == "text" and c["second_stage_sigmas"] and "kuno_trajectory_tap" not in c for c in renderer.calls)
    assert backend.loads == ["ltx-2.5-fast"]
    assert not (tmp_path / "work" / task.job_id).exists()

    # Pins: the stub labels every tail token with the shot that made it.
    first, second, third = renderer.pins
    assert all(pin.video is None and pin.audio is None for pin in first.values())
    for stage in ("half", "full"):
        assert second[stage].video.shape[1] == 3 * 6 and (second[stage].video == 0).all()  # the previous shot's last 3 latent frames
        assert second[stage].audio.shape[1] == 18 and (second[stage].audio == 0).all()
        # A cut pins sound only, from the first shot of the run (the protocol's anchor), not from shot 2.
        assert third[stage].video is None
        assert third[stage].audio.shape[1] == 18 and (third[stage].audio == 0).all()


def test_a_silent_storyboard_has_no_audio_track(tmp_path):
    result = backend_with(tmp_path, StubRenderer()).generate(storyboard_task(board(shots((2, "fresh"), (2, "cut")), audio=False)), lambda *_: None)
    assert result.info.audio is False and not probe(result.data).audio
    assert result.info.frames == probe(result.data).frames == 49 + 32


def test_a_cancel_between_shots_stops_the_storyboard(tmp_path):
    renderer = StubRenderer()
    backend = backend_with(tmp_path, renderer)
    task = storyboard_task(board(shots((2, "fresh"), (2, "continue"), (2, "continue"))))

    def progress(value, stage):
        if stage == "shot 2/3":  # the worker's progress raises once the gateway reports the job canceled
            raise JobCanceled()

    with pytest.raises(JobCanceled):
        backend.generate(task, progress)
    assert len(renderer.calls) == 1
    assert not (tmp_path / "work" / task.job_id).exists()


def test_a_shot_that_fails_fails_the_job_and_cleans_up(tmp_path):
    renderer = StubRenderer(fail_on_shot=1)
    backend = backend_with(tmp_path, renderer)
    task = storyboard_task(board(shots((2, "fresh"), (2, "continue"))))
    with pytest.raises(RuntimeError, match="out of memory"):
        backend.generate(task, lambda *_: None)
    assert len(renderer.calls) == 2 and not (tmp_path / "work" / task.job_id).exists()


def test_a_join_whose_pins_did_not_hold_fails_the_job(tmp_path):
    class Drifting(StubRenderer):
        def render(self, call, pins):
            rendered = super().render(call, pins)
            if "full" in rendered.pins_exact and rendered.pins_exact["full"]:
                rendered.pins_exact["full"] = False
            return rendered

    with pytest.raises(StoryboardError, match="shot 2: the pinned full-pass tokens changed"):
        backend_with(tmp_path, Drifting()).generate(storyboard_task(board(shots((2, "fresh"), (2, "continue")))), lambda *_: None)


def test_memory_admission_looks_at_the_longest_shot_before_anything_loads(tmp_path):
    renderer = StubRenderer(fail_on_shot=0)
    backend = backend_with(tmp_path, renderer, hardware_class=RTX5090, host_ram_gib=128)
    width, height = FAST.size_for("1080p", "21:9")
    longest = longest_duration(backend.memory_plan(FAST), FAST, width, height, 24)
    assert longest is not None and longest < FAST.limits.max_duration_s

    too_long = storyboard_task(board(shots((2, "fresh"), (longest + 1, "continue")), resolution="1080p", aspect_ratio="21:9"), width=width, height=height)
    with pytest.raises(CapacityRefused):
        backend.generate(too_long, lambda *_: None)
    assert backend.loads == [] and renderer.calls == []

    # Twice as long stitched as one clip could be, but no shot is longer than the class serves: admitted, and rendered.
    fits = storyboard_task(board(shots((longest, "fresh"), (longest, "continue")), resolution="1080p", aspect_ratio="21:9"), width=width, height=height)
    with pytest.raises(RuntimeError, match="out of memory"):
        backend.generate(fits, lambda *_: None)
    assert backend.loads == ["ltx-2.5-fast"] and len(renderer.calls) == 1


def test_a_storyboard_task_splits_into_shot_tasks():
    task = storyboard_task(board(shots((2, "fresh"), (5, "cut"))), seed=40)
    assert task.shot_frames == [49, 121 - 17] and task.num_frames == storyboard_frames(FAST, task.params.shots, 24)
    shot = task.shot_task(1)
    assert (shot.params.mode, shot.params.duration_s, shot.params.shots, shot.prompt, shot.seed) == (Mode.TEXT_TO_VIDEO, 5, None, task.shot_prompts[1], 41)
    assert shot.shot_frames is None and shot.num_frames == 121
    with pytest.raises(ValueError, match="one model prompt per shot"):
        storyboard_task(task.params, prompts=["only one"]).shot_task(0)


# ---------------------------------------------------------------- the mock backend


@pytest.mark.parametrize("fps", [24, 25, 50])
def test_the_mock_backend_makes_the_stitched_video_without_a_commitment(fps):
    params = board(shots((2, "fresh"), (3, "continue"), (2, "cut")), fps=fps)
    task = storyboard_task(params)
    stages = []
    result = MockBackend().generate(task, lambda value, stage: stages.append(stage))  # the dev hardware class runs verified mode
    frames = storyboard_frames(FAST, params.shots, fps)
    assert probe(result.data).frames == result.info.frames == frames
    assert (result.info.fps, result.info.duration_s) == (fps, round(frames / fps, 3))
    assert stages[:3] == ["shot 1/3", "shot 2/3", "shot 3/3"]
    assert result.step_commitment is None and result.openings is None


# ---------------------------------------------------------------- the worker


class SpyBackend(MockBackend):
    def __init__(self):
        super().__init__()
        self.tasks = []

    def generate(self, task, progress):
        self.tasks.append(task)
        return super().generate(task, progress)


class NoStoryboards(Backend):
    name = "cold"

    def generate(self, task, progress):
        raise AssertionError("a backend without storyboards must never see one")


@pytest.fixture(autouse=True)
def fresh_gate():
    safety.configure(SafetyGate())
    yield
    safety.configure(None)


def sealed(worker, params: GenerationParams, payload: SealedPayload | bytes) -> MinerJob:
    job_id = str(uuid.uuid4())
    session = SenderSession(worker.identity.hpke_public)
    plaintext = payload if isinstance(payload, bytes) else payload.model_dump_json().encode()
    ciphertext = session.seal(plaintext, job_aad(job_id, worker.identity.enclave_id, params, []))
    return MinerJob(job_id=job_id, params=params, enc=b64e(session.enc), ciphertext=b64e(ciphertext), input_blob_ids=[])


def storyboard_payload(prompts: list[str], scene: str = SCENE) -> SealedPayload:
    return SealedPayload(prompt=scene, seed=11, shots=[ShotPrompt(prompt=p) for p in prompts])


def rejected(worker, job) -> str:
    with pytest.raises(JobRejected) as exc:
        worker.process(job)
    return exc.value.code


def test_the_worker_hands_the_backend_every_shots_model_prompt_and_checks_each(tmp_path, monkeypatch):
    backend = SpyBackend()
    worker = make_worker(tmp_path, backend)
    checked, sampled = [], []
    monkeypatch.setattr(worker_module, "check_request", lambda prompt, negative=None: checked.append(prompt))
    monkeypatch.setattr(worker_module, "check_output", lambda video, signals, shot_frames=None: sampled.append(shot_frames))
    params = board(shots((2, "fresh"), (3, "continue"), (2, "cut")))
    receipt = worker.process(sealed(worker, params, storyboard_payload(SHOT_PROMPTS)))

    [task] = backend.tasks
    models = [f"{SCENE}\n\n{prompt}" for prompt in SHOT_PROMPTS]
    assert task.shot_prompts == models and task.prompt == SCENE and task.seed == 11
    assert checked == models
    assert sampled == [task.shot_frames] and sum(task.shot_frames) == storyboard_frames(FAST, params.shots, 24)
    assert receipt.body.video.frames == storyboard_frames(FAST, params.shots, 24)


def test_a_storyboard_with_an_empty_scene_prompts_each_shot_alone(tmp_path):
    backend = SpyBackend()
    worker = make_worker(tmp_path, backend)
    worker.process(sealed(worker, board(shots((2, "fresh"), (2, "cut"))), storyboard_payload(["  A gull  ", "A boat"], scene="")))
    assert backend.tasks[0].shot_prompts == ["A gull", "A boat"]


def _raw_shots(*prompts: str) -> bytes:
    return json.dumps({"v": 1, "prompt": SCENE, "shots": [{"prompt": p} for p in prompts]}).encode()


@pytest.mark.parametrize(
    "payload",
    [
        storyboard_payload(SHOT_PROMPTS[:2]),
        storyboard_payload(SHOT_PROMPTS + ["one too many"]),
        SealedPayload(prompt=SCENE),
        _raw_shots("The boat pulls away", "   ", "A gull lands"),  # a blank prompt no longer parses as a ShotPrompt
        _raw_shots("The boat pulls away", "", "A gull lands"),
    ],
    ids=["too few shot prompts", "too many", "no shot prompts", "a blank shot prompt", "an empty shot prompt"],
)
def test_a_shot_list_that_does_not_match_the_params_is_a_bad_payload(tmp_path, payload):
    backend = SpyBackend()
    worker = make_worker(tmp_path, backend)
    assert rejected(worker, sealed(worker, board(shots((2, "fresh"), (2, "continue"), (2, "cut"))), payload)) == "bad_payload"
    assert backend.tasks == []


def test_shot_prompts_on_an_ordinary_job_are_a_bad_payload(tmp_path):
    backend = SpyBackend()
    worker = make_worker(tmp_path, backend)
    params = GenerationParams(profile_id=FAST.id, mode=Mode.TEXT_TO_VIDEO, duration_s=2, resolution="720p", aspect_ratio="16:9", fps=24)
    assert rejected(worker, sealed(worker, params, storyboard_payload(SHOT_PROMPTS[:1]))) == "bad_payload"
    assert backend.tasks == []


def test_every_shots_model_prompt_must_fit_the_profiles_limit(tmp_path):
    backend = SpyBackend()
    worker = make_worker(tmp_path, backend)
    limit = FAST.limits.max_prompt_chars
    scene = "s" * (limit // 2)
    fits, too_long = "a" * (limit - len(scene) - 2), "a" * (limit - len(scene) - 1)  # the model sees scene + "\n\n" + shot
    params = board(shots((2, "fresh"), (2, "cut")))
    assert rejected(worker, sealed(worker, params, storyboard_payload([fits, too_long], scene=scene))) == "prompt_too_long"
    worker.process(sealed(worker, params, storyboard_payload([fits, fits], scene=scene)))
    assert [len(p) for p in backend.tasks[0].shot_prompts] == [limit, limit]


def test_one_blocked_shot_blocks_the_storyboard_before_it_renders(tmp_path, monkeypatch):
    backend = SpyBackend()
    worker = make_worker(tmp_path, backend)

    def check(prompt, negative=None):
        if prompt.endswith(SHOT_PROMPTS[2]):
            raise SafetyViolation("violence")

    monkeypatch.setattr(worker_module, "check_request", check)
    assert rejected(worker, sealed(worker, board(shots((2, "fresh"), (2, "continue"), (2, "cut"))), storyboard_payload(SHOT_PROMPTS))) == "safety_blocked"
    assert backend.tasks == []


def test_a_backend_that_cannot_render_storyboards_refuses_before_decrypting(tmp_path):
    worker = make_worker(tmp_path, NoStoryboards())
    job = sealed(worker, board(shots((2, "fresh"), (2, "continue"))), b"not even decryptable")
    job = job.model_copy(update={"ciphertext": b64e(b"\x00" * 64)})
    assert rejected(worker, job) == "internal_error"


# ---------------------------------------------------------------- the output safety check


def test_every_shot_is_sampled_however_short():
    # Eleven 20 s shots and a 2 s one: an even spread of 36 frames over the stitched video steps over the short shot.
    kept = [481] + [481 - 17] * 5 + [49 - 17] + [481 - 17] * 5
    total = sum(kept)
    even = plan_frame_indices(total, 3 * len(kept))
    start = sum(kept[:6])
    assert not [i for i in even if start <= i < start + kept[6]]
    planned = plan_frame_indices(total, 10, kept)
    offset = 0
    for frames in kept:
        inside = [i for i in planned if offset <= i < offset + frames]
        assert len(inside) >= FRAMES_PER_SHOT and inside[0] == offset and inside[-1] == offset + frames - 1
        offset += frames
    assert len(planned) >= max(10, FRAMES_PER_SHOT * len(kept)) and planned[0] == 0 and planned[-1] == total - 1
    with pytest.raises(ValueError):
        plan_frame_indices(total + 1, 10, kept)


def test_the_gate_samples_a_storyboard_per_shot_and_at_least_ten_frames():
    class Scorer:
        name, input_size = "fake-scorer", 64

        def __init__(self):
            self.seen = []

        def score_frames(self, frames):
            self.seen = list(frames)
            return [{"sexual": 0.0, "minor": 0.0} for _ in frames]

    task = storyboard_task(board(shots((2, "fresh"), (2, "continue"))))
    video = MockBackend().generate(task, lambda *_: None).data
    scorer = Scorer()
    SafetyGate(frame_classifiers=[scorer], frames_to_sample=2).check_output(video, None, shot_frames=task.shot_frames)
    assert len(scorer.seen) == len(plan_frame_indices(sum(task.shot_frames), 10, task.shot_frames)) >= 10


def test_signals_from_several_prompts_keep_whatever_any_raised():
    assert RequestSignals.combine([RequestSignals(), RequestSignals(mentions_minor=True)]).mentions_minor
    assert not RequestSignals.combine([RequestSignals(), RequestSignals()]).mentions_minor


# ---------------------------------------------------------------- kuno-plan


def test_kuno_plan_shows_each_shots_call_and_join(tmp_path):
    params = example_task(FAST, Mode.STORYBOARD, shots=parse_shots("5:fresh,5:continue,3:cut"))
    assert params.duration_s == storyboard_duration_s(FAST, params.shots, 24)
    plan = storyboard_plan(build_task(FAST, params, tmp_path, seed=42))
    assert plan["stitched_frames"] == storyboard_frames(FAST, params.shots, 24) == 121 + 104 + 56
    assert [(s["join"], s["video_pin_latent_frames"], s["audio_pin_latents"], s["audio_source_shot"], s["trim_frames"]) for s in plan["shots"]] == [
        ("fresh", 0, 0, None, 0), ("continue", 3, 18, 1, 17), ("cut", 0, 18, 1, 17),
    ]
    assert [s["call"]["seed"] for s in plan["shots"]] == [42, 43, 44]
    assert example_task(FAST, Mode.STORYBOARD).shots == shots((2, "fresh"), (2, "continue"))
