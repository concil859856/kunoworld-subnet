"""Storyboard pins through real tensors: pinned tokens never move, the model is shown them at timestep 0, and continuous
seams run through to the frame and the sample. The fake renderer needs torch; the tiny one needs diffusers too, and runs
the resident backend's whole storyboard path through diffusers' real LTX-2 classes. Both skip where torch is not
installed; they run inside the LTX worker image:

    S=/video/.venv/lib/python3.12/site-packages; docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=kuno \\
      -v /video/subnet:/src:ro $(for m in pytest _pytest pluggy iniconfig py.py; do printf -- '-v %s/%s:/pt/%s:ro ' "$S" "$m" "$m"; done) \\
      -e PYTHONPATH=/src/worker/src:/src/protocol/src:/pt -e PYTHONDONTWRITEBYTECODE=1 -w /tmp --entrypoint python kuno-worker:ltx \\
      -m pytest /src/worker/tests/test_ltx_storyboard_render.py -q -p no:cacheprovider
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from kuno_protocol.mp4 import probe
from kuno_protocol.profiles import Mode, load_profiles, ltx_num_frames, storyboard_duration_s, storyboard_frames
from kuno_protocol.schemas import GenerationParams, ShotSpec
from kuno_worker.backends.base import GenerationTask
from kuno_worker.backends.ltx_resident import DISTILLED_SIGMAS, SECOND_STAGE_SIGMAS, LtxResidentBackend, build_call
from kuno_worker.backends.ltx_storyboard import (
    Geometry,
    Timeline,
    assemble_audio,
    clean_head,
    extend_classes,
    pins_for,
    pinned_timesteps,
    render_storyboard,
)

from ltx_storyboard_doubles import FakeRenderer

torch = pytest.importorskip("torch")

PROFILES = load_profiles()
FAST = PROFILES["ltx-2.5-fast"]
G24 = Geometry(fps=24.0)


def fake_call(seed: int, prompt: str = "harbor", frames: int = 49, second: bool = True) -> dict:
    call = {  # the shape of ltx_resident.build_call for ltx-2.5-fast
        "pipeline": "text", "prompt": prompt, "width": 320, "height": 192, "num_frames": frames, "frame_rate": 24.0, "seed": seed,
        "generate_audio": True, "sigmas": DISTILLED_SIGMAS,
    }
    if second:
        call["second_stage_sigmas"] = SECOND_STAGE_SIGMAS
    return call


def test_pinned_timesteps_are_zero_on_the_head_only():
    t = torch.tensor([909.375, 909.375])  # a CFG batch of two
    out = pinned_timesteps(t, 51, 18)
    assert out.shape == (2, 51)
    assert torch.equal(out[:, :18], torch.zeros(2, 18)) and torch.equal(out[:, 18:], torch.full((2, 33), 909.375))
    assert clean_head(out, torch.zeros(1, 18, 128)) is True
    assert clean_head(t, torch.zeros(1, 18, 128)) is False  # one timestep per row: the model sees the head as noise


def test_fake_pins_stay_bit_identical_through_every_step():
    renderer = FakeRenderer(overlap=3, trace=True)
    timeline = Timeline(G24, 3)
    tails = {}
    for index, join_mode in enumerate(["fresh", "continue"]):
        join = timeline.plan(join_mode, 49)
        call = fake_call(1234 + index)
        pins = {stage: pins_for(join, tails, stage) for stage in renderer.stages(call)}
        renderer.head_trace.clear()
        rendered = renderer.render(call, pins)
        timeline.rendered(index, len(rendered.frames), rendered.audio.shape[1])
        tails[index] = rendered.tails
    assert rendered.pins_exact == {"half": True, "full": True}
    assert rendered.seen_clean == {"half": {"video": True, "audio": True}, "full": {"video": True, "audio": True}}
    assert len(renderer.head_trace) == len(DISTILLED_SIGMAS) + len(SECOND_STAGE_SIGMAS)
    for stage, video_head, audio_head in renderer.head_trace:
        assert torch.equal(video_head, pins[stage].video)
        assert torch.equal(audio_head, pins[stage].audio)
    assert pins["half"].video.shape == (1, 3 * 5 * 3, 128)  # 3 latent frames of the 160x96 pass's 5x3 grid
    assert pins["full"].video.shape == (1, 3 * 10 * 6, 128)  # and of the 320x192 pass's 10x6 grid
    assert pins["full"].audio.shape == (1, join.audio_pin, 128)


def test_pinned_scheduler_holds_the_head_and_changes_nothing_else():
    pytest.importorskip("diffusers")
    from diffusers import FlowMatchEulerDiscreteScheduler

    Scheduler, _ = extend_classes()
    pinned, stock = Scheduler.from_config(FlowMatchEulerDiscreteScheduler().config), FlowMatchEulerDiscreteScheduler()
    for scheduler in (pinned, stock):
        scheduler.set_timesteps(sigmas=DISTILLED_SIGMAS, device="cpu")
        scheduler.set_begin_index(0)
    generator = torch.Generator("cpu").manual_seed(0)
    head = torch.randn(1, 18, 128, generator=generator)
    pinned.start(head)
    sample_pinned = torch.randn(1, 51, 128, generator=generator)
    sample_pinned[:, :18] = head
    sample_stock = sample_pinned.clone()
    for t in pinned.timesteps:
        velocity = torch.randn(1, 51, 128, generator=generator)
        sample_pinned = pinned.step(velocity, t, sample_pinned, return_dict=False)[0]
        sample_stock = stock.step(velocity, t, sample_stock, return_dict=False)[0]
        assert torch.equal(sample_pinned[:, :18], head)
        assert torch.equal(sample_pinned[:, 18:], sample_stock[:, 18:])  # the step itself is the stock one
    assert not torch.equal(sample_stock[:, :18], head)  # without the pin the head would have moved
    assert torch.equal(pinned.final, sample_pinned)


class Recording:
    """Keeps what each shot rendered, for checks the stitched MP4 can't show."""

    def __init__(self, renderer):
        self.renderer, self.results = renderer, []

    def geometry(self, fps):
        return self.renderer.geometry(fps)

    def stages(self, call):
        return self.renderer.stages(call)

    def render(self, call, pins):
        self.results.append(self.renderer.render(call, pins))
        return self.results[-1]


@pytest.mark.parametrize("anchor", ["first", "previous"])
@pytest.mark.parametrize("two_stage", [True, False])
def test_fake_chain_runs_through_every_continuous_seam(two_stage, anchor):
    """The fake decodes clocks through the causal VAEs' own frame and mel-frame layout, independently of the timeline's
    arithmetic: stitched, a continue seam advances the picture by one frame and a continuous audio join by one sample."""
    renderer = FakeRenderer(overlap=3)
    joins = ["fresh", "continue", "continue", "cut", "fresh", "cut", "continue"]
    durations = [2, 3, 2, 3, 2, 2, 3]
    timeline = Timeline(G24, 3, anchor)
    tails, frame_clocks, sample_clocks = {}, [], []
    for index, (join_mode, duration) in enumerate(zip(joins, durations)):
        join = timeline.plan(join_mode, ltx_num_frames(duration, 24))
        call = fake_call(1234 + index, prompt=f"p{index}", frames=join.frames, second=two_stage)
        rendered = renderer.render(call, {stage: pins_for(join, tails, stage) for stage in renderer.stages(call)})
        timeline.rendered(index, len(rendered.frames), rendered.audio.shape[1])
        tails[index] = rendered.tails
        frame_clocks.append(rendered.clocks[0])
        sample_clocks.append(rendered.clocks[1])
    total_frames, total_samples = timeline.finish()
    video = np.concatenate([c[j.video_trim :] for j, c in zip(timeline.joins, frame_clocks)])
    audio = assemble_audio(timeline.joins, [c[None] for c in sample_clocks], total_samples)[0].astype(np.float64)
    assert len(video) == total_frames
    frame, sample = 1 / 24, 1 / 48_000
    assert np.allclose(np.diff(video[: timeline.joins[1].video_start]), frame, atol=1e-4)
    continuous = 0
    for join in timeline.joins[1:]:
        f, s = join.video_start, join.audio_start
        if join.join == "continue":
            assert video[f] - video[f - 1] == pytest.approx(frame, abs=1e-4)
        if join.audio_continuous:
            continuous += 1
            assert audio[s] - audio[s - 1] == pytest.approx(sample, abs=1e-3)
            assert audio[s + 480] - audio[s - 1] == pytest.approx(481 * sample, abs=1e-3)
    assert continuous == (2 if anchor == "first" else 5)


def test_a_fake_storyboard_stitches_through_the_product_loop(tmp_path):
    joins, durations = ["fresh", "continue", "cut", "fresh"], [2, 3, 2, 2]
    calls = [fake_call(7 + i, prompt=f"p{i}", frames=ltx_num_frames(d, 24)) for i, d in enumerate(durations)]
    renderer = Recording(FakeRenderer(overlap=3))
    stages = []
    data, frames = render_storyboard(renderer, joins, calls, tmp_path, lambda v, s: stages.append(s), overlap=3)
    shots = [ShotSpec(duration_s=d, join=j) for j, d in zip(joins, durations)]
    info = probe(data)
    assert frames == info.frames == storyboard_frames(FAST, shots, 24) and info.audio
    assert info.duration_s == pytest.approx(frames / 24, abs=0.05)
    assert stages == ["shot 1/4", "shot 2/4", "shot 3/4", "shot 4/4", "encoding"]
    assert [r.pins_exact for r in renderer.results] == [{"half": None, "full": None}, {"half": True, "full": True}, {"half": True, "full": True}, {"half": None, "full": None}]


# ---------------------------------------------------------------- tiny diffusers


def tiny_task(shot_list: list[ShotSpec], profile_id: str = "ltx-2.5-fast") -> GenerationTask:
    params = GenerationParams(
        profile_id=profile_id, mode=Mode.STORYBOARD, duration_s=storyboard_duration_s(FAST, shot_list, 24), resolution="720p",
        aspect_ratio="16:9", fps=24, shots=shot_list,
    )
    prompts = [f"A calm lake at dawn, shot {i + 1}" for i in range(len(shot_list))]
    return GenerationTask(
        job_id=str(uuid.uuid4()), profile=PROFILES[profile_id], params=params, prompt="A calm lake", negative_prompt=None, seed=3,
        width=320, height=192, shot_prompts=prompts,
    )


def test_tiny_diffusers_storyboard_holds_the_pins_in_both_passes_through_the_backend(tmp_path):
    pytest.importorskip("diffusers")
    from ltx_storyboard_doubles import TinyRenderer, tiny_pipelines

    recorders = []

    def renderer(loaded, profile):
        recorders.append(Recording(TinyRenderer(loaded.pipelines, profile.limits.storyboard.overlap_latent_frames)))
        return recorders[-1]

    backend = LtxResidentBackend(
        None, tmp_path, loader=lambda profile: SimpleNamespace(pipelines=tiny_pipelines(), device="cpu"), storyboard_renderer=renderer,
    )
    shot_list = [ShotSpec(duration_s=2, join="fresh"), ShotSpec(duration_s=2, join="continue"), ShotSpec(duration_s=2, join="cut")]
    task = tiny_task(shot_list)
    result = backend.generate(task, lambda *_: None)

    assert result.info.frames == probe(result.data).frames == storyboard_frames(FAST, shot_list, 24) == 3 * 49 - 2 * 17
    assert probe(result.data).audio and result.step_commitment is None
    [recorder] = recorders
    rendered = recorder.results
    assert [r.pins_exact for r in rendered] == [{"half": None, "full": None}, {"half": True, "full": True}, {"half": True, "full": True}]
    assert rendered[1].seen_clean == {"half": {"video": True, "audio": True}, "full": {"video": True, "audio": True}}
    assert rendered[2].seen_clean == {"half": {"video": None, "audio": True}, "full": {"video": None, "audio": True}}
    geometry = recorder.renderer.geometry(24.0)
    assert (geometry.sample_rate, geometry.audio_latents_per_second) == (48_000, 25.0)
    # The real audio VAE and vocoder decode to exactly what the causal rule the planner uses predicts.
    assert all(r.audio.shape[1] == geometry.audio_samples(geometry.audio_latents(49)) for r in rendered)


def test_tiny_diffusers_single_pass_with_guidance_holds_the_pins(tmp_path):
    """ltx-2.5-pro's call (one pass, CFG, cross timesteps) through the same renderer: the guidance batch of two gets
    per-token audio timesteps too. Steps cut to keep the CPU run short."""
    pytest.importorskip("diffusers")
    from ltx_storyboard_doubles import TinyRenderer, tiny_pipelines

    pro = PROFILES["ltx-2.5-pro"]
    shot_list = [ShotSpec(duration_s=2, join="fresh"), ShotSpec(duration_s=2, join="continue")]
    task = tiny_task(shot_list)
    calls = []
    for index in range(len(shot_list)):
        call = build_call(replace(task.shot_task(index), profile=pro))
        assert "second_stage_sigmas" not in call and call["guidance_scale"] > 1
        calls.append({**call, "num_inference_steps": 3})
    renderer = Recording(TinyRenderer(tiny_pipelines(), overlap=3))
    data, frames = render_storyboard(renderer, [s.join for s in shot_list], calls, tmp_path, lambda *_: None, overlap=3)
    assert frames == probe(data).frames == 49 + 32
    assert [r.pins_exact for r in renderer.results] == [{"full": None}, {"full": True}]
    assert renderer.results[1].seen_clean == {"full": {"video": True, "audio": True}}
