"""Audio-to-video and retake through diffusers' real LTX-2 classes with tiny random weights (backends/ltx_edit.py on
ltx_pinning.PinnedRenderer): the held tokens are what the pipeline itself would hold, they never move, the transformer sees
them at timestep 0, and the job returns the source's own sound wherever it is held. The pictures are noise; the shapes,
hooks, passes and lengths are the real code's. Skips where torch and diffusers are not installed; runs in the LTX worker
image:

    S=/video/.venv/lib/python3.12/site-packages; docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=kuno \\
      -v /video/subnet:/src:ro $(for m in pytest _pytest pluggy iniconfig py.py; do printf -- '-v %s/%s:/pt/%s:ro ' "$S" "$m" "$m"; done) \\
      -e PYTHONPATH=/src/worker/src:/src/protocol/src:/pt:/src/worker/tests -e PYTHONDONTWRITEBYTECODE=1 -w /tmp --entrypoint python \\
      kuno-worker:ltx -m pytest /src/worker/tests/test_ltx_edit_render.py -q -p no:cacheprovider
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import numpy as np
import pytest

from kuno_protocol.mp4 import probe
from kuno_protocol.profiles import InputRole, Mode, load_profiles
from kuno_protocol.schemas import GenerationParams, InputRef
from kuno_worker.backends import ltx_resident
from kuno_worker.backends.base import GenerationTask, InputFile
from kuno_worker.backends.ltx_edit import EditError, decode_audio, fit_samples, render_edit
from kuno_worker.backends.ltx_pinning import Pins, Rendered
from kuno_worker.backends.ltx_resident import LtxResidentBackend, build_call
from kuno_worker.backends.media_tools import ffmpeg_exe

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("torchaudio")

from ltx_storyboard_doubles import TinyPinnedRenderer, tiny_pipelines  # noqa: E402

PROFILES = load_profiles()
FAST, PRO = PROFILES["ltx-2.5-fast"], PROFILES["ltx-2.5-pro"]
WIDTH, HEIGHT = 320, 192


def ffmpeg(*args: str) -> None:
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("edit-media")
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3", "-f", "lavfi", "-i", "sine=frequency=660:sample_rate=48000:duration=3",
           "-filter_complex", "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]", "-map", "[a]", str(out / "tone.wav"))
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate=24:duration=2.1", "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=2.1",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out / "clip.mp4"))
    ffmpeg("-f", "lavfi", "-i", f"color=orange:s={WIDTH}x{HEIGHT}", "-frames:v", "1", str(out / "first.png"))
    return {"audio": out / "tone.wav", "clip": out / "clip.mp4", "image": out / "first.png"}


@pytest.fixture(scope="module")
def pipelines():
    return tiny_pipelines(audio_ch_mult=(1, 1, 1))


def job(profile, mode: Mode, inputs: list[tuple[InputRole, Path, dict]], *, options=None, seed: int = 5, audio: bool = True) -> GenerationTask:
    mimes = {".wav": "audio/wav", ".mp4": "video/mp4", ".png": "image/png"}
    files = []
    for index, (role, path, ref) in enumerate(inputs):
        data = path.read_bytes()
        files.append(InputFile(
            ref=InputRef(index=index, role=role, mime=mimes[path.suffix], sha256="0" * 64, size=len(data), **ref), data=data, mime=mimes[path.suffix],
        ))
    params = GenerationParams(
        profile_id=profile.id, mode=mode, duration_s=2, resolution="720p", aspect_ratio="16:9", fps=24, audio=audio,
        input_roles=[role for role, _, _ in inputs],
    )
    return GenerationTask(
        job_id=str(uuid.uuid4()), profile=profile, params=params, prompt="An orange room hums", negative_prompt=None, seed=seed,
        width=WIDTH, height=HEIGHT, inputs=files, options=options or {},
    )


class Recording:
    """The loaded adapter, keeping what each render returned before it was encoded."""

    def __init__(self, adapter):
        self.adapter, self.results = adapter, []

    def __call__(self, **call):
        self.results.append(self.adapter(**call))
        return self.results[-1]


def backend_for(tmp_path, pipelines) -> tuple[LtxResidentBackend, Recording]:
    from kuno_worker.backends.runtimes import LtxAdapter

    recording = Recording(LtxAdapter(pipelines, device="cpu", renderer=lambda p, device: TinyPinnedRenderer(p)))
    return LtxResidentBackend(None, tmp_path / "work", loader=lambda profile: recording), recording


# ---------------------------------------------------------------- the encoders hold what the pipeline holds


def test_a_source_frame_encodes_to_the_tokens_the_pipeline_holds_for_a_first_frame(pipelines, media):
    from diffusers.pipelines.ltx2 import LTX2VideoCondition
    from PIL import Image

    renderer = TinyPinnedRenderer(pipelines)
    frame = np.asarray(Image.open(media["image"]).convert("RGB"))
    tokens = renderer.encode_video(frame[None], WIDTH, HEIGHT)
    assert tokens.shape == (1, (HEIGHT // 32) * (WIDTH // 32), 128) and tokens.dtype == torch.float32
    # crf=0: LTX2ConditionPipeline re-compresses a single image with H.264 otherwise, which a video's frames never are.
    _, mask, clean, _ = renderer.pipeline.prepare_latents(
        conditions=[LTX2VideoCondition(frames=Image.open(media["image"]).convert("RGB"), index=0, strength=1.0, crf=0)],
        height=HEIGHT, width=WIDTH, num_frames=49, dtype=torch.float32, device=torch.device("cpu"), generator=torch.Generator("cpu").manual_seed(0),
    )
    per = tokens.shape[1]
    assert torch.equal(mask[0, :per, 0], torch.ones(per)) and torch.equal(clean[:, :per], tokens)


def test_a_sound_track_encodes_to_the_latents_the_pipeline_would_normalize_and_pack(pipelines, media):
    import torchaudio

    renderer = TinyPinnedRenderer(pipelines)
    pipeline = renderer.pipeline
    sound = decode_audio(media["audio"], sample_rate=48_000, duration_s=49 / 24).astype(np.float32) / 32768
    tokens = renderer.encode_audio(sound, 48_000, 51)
    assert tokens.shape == (1, 51, 128)
    # The same log-mel and posterior mode by hand, then the pipeline's own packing and normalization of given latents
    # (prepare_audio_latents with noise_scale 0 adds no noise).
    waveform = torchaudio.functional.resample(torch.from_numpy(sound), 48_000, 16_000)
    waveform = torch.nn.functional.pad(waveform, (0, (4 * 51 + 2) * 160 + 1024 - waveform.shape[1]))
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16_000, n_fft=1024, win_length=1024, hop_length=160, f_min=0.0, f_max=8000.0, n_mels=64, window_fn=torch.hann_window,
        center=True, pad_mode="reflect", power=1.0, mel_scale="slaney", norm="slaney",
    )(waveform[None])
    mel = torch.log(torch.clamp(mel, min=1e-5)).permute(0, 1, 3, 2)
    latents = pipeline.audio_vae.encode(mel).latent_dist.mode()[:, :, :51]
    expected = pipeline.prepare_audio_latents(latents=latents, noise_scale=0.0, dtype=torch.float32, device=torch.device("cpu"))
    assert torch.equal(tokens, expected)


# ---------------------------------------------------------------- audio-to-video


def test_audio_to_video_holds_every_audio_token_and_returns_the_source_sound(tmp_path, pipelines, media, monkeypatch):
    monkeypatch.setattr(ltx_resident, "FULL_STEPS", 3)  # ltx-2.5-pro's 30 guided steps, cut to keep the CPU run short
    backend, recording = backend_for(tmp_path, pipelines)
    task = job(PRO, Mode.AUDIO_TO_VIDEO, [(InputRole.SOURCE_AUDIO, media["audio"], {"start_s": 0.25}), (InputRole.FIRST_FRAME, media["image"], {})])
    result = backend.generate(task, lambda value, stage: None)

    info = probe(result.data)
    assert result.info.frames == info.frames == 49 and info.audio and result.step_commitment is None
    [raw] = recording.results
    report = raw["edit"]
    assert report["held_audio_latents"] == 51 and report["held_latent_frames"] == 0
    assert report["pins_exact"] == {"full": True}
    assert report["seen_clean"] == {"full": {"video": None, "audio": True}}  # every audio token at t = 0, from the first step
    source = fit_samples(decode_audio(media["audio"], sample_rate=48_000, start_s=0.25, duration_s=49 / 24), 98_000)
    assert raw["audio"].dtype == np.int16 and np.array_equal(raw["audio"], source)


def test_audio_to_video_pads_a_short_sound_with_silence(pipelines, media):
    renderer = TinyPinnedRenderer(pipelines)
    task = job(PRO, Mode.AUDIO_TO_VIDEO, [(InputRole.SOURCE_AUDIO, media["audio"], {"start_s": 2.5})])
    for item in task.inputs:
        item.save(media["audio"].parent / "short")
    call = {**build_call(task), "num_inference_steps": 2}
    out = render_edit(renderer, call, call.pop("edit"))
    assert out["edit"]["held_audio_latents"] == 51 and out["edit"]["pins_exact"] == {"full": True}
    assert np.array_equal(out["audio"][:, 24_000:], np.zeros((2, 98_000 - 24_000), np.int16))  # 0.5 s of sound, then silence
    assert np.abs(out["audio"][:, :23_000]).max() > 1000


# ---------------------------------------------------------------- retake


def test_a_retake_holds_everything_outside_its_window(tmp_path, pipelines, media):
    backend, recording = backend_for(tmp_path, pipelines)
    task = job(FAST, Mode.RETAKE, [(InputRole.SOURCE_VIDEO, media["clip"], {"start_s": 0.5, "end_s": 1.5})])
    result = backend.generate(task, lambda value, stage: None)

    assert result.info.frames == probe(result.data).frames == 49 and result.step_commitment is None
    [raw] = recording.results
    report = raw["edit"]
    # 0.5 s to 1.5 s at 24 fps: latent frames 2..5 (pixel frames 9..40) and audio latents 13..38 (0.49 s to 1.53 s).
    assert (report["regenerated_latent_frames"], report["regenerated_frames"]) == ([2, 6], [9, 41])
    assert report["held_latent_frames"] == 3 and report["held_audio_latents"] == 51 - 26
    assert report["regenerated_samples"] == [23_520, 73_440]
    assert report["pins_exact"] == {"full": True} and report["seen_clean"] == {"full": {"video": True, "audio": True}}
    source = fit_samples(decode_audio(media["clip"], sample_rate=48_000, duration_s=49 / 24), 98_000)
    s0, s1 = report["regenerated_samples"]
    assert np.array_equal(raw["audio"][:, :s0], source[:, :s0]) and np.array_equal(raw["audio"][:, s1:], source[:, s1:])
    assert not np.array_equal(raw["audio"][:, s0 + 960 : s1 - 960], source[:, s0 + 960 : s1 - 960])


def test_a_retake_of_the_sound_alone_holds_every_video_token(pipelines, media, tmp_path):
    renderer = TinyPinnedRenderer(pipelines)
    task = job(FAST, Mode.RETAKE, [(InputRole.SOURCE_VIDEO, media["clip"], {"start_s": 0.5, "end_s": 1.5})], options={"regenerate_video": False})
    for item in task.inputs:
        item.save(tmp_path)
    call = build_call(task)
    out = render_edit(renderer, call, call.pop("edit"))
    report = out["edit"]
    assert report["held_latent_frames"] == 7 and report["regenerated_latent_frames"] == [0, 0]
    assert report["pins_exact"] == {"full": True} and report["seen_clean"] == {"full": {"video": True, "audio": True}}


def test_two_retakes_of_the_same_clip_share_their_held_latents(pipelines, media, tmp_path):
    """Different seeds regenerate the window differently, but the held latent frame 0 is the source's in both, and the
    tiny VAE's causal decoder makes frame 0 from it alone."""
    renderer = TinyPinnedRenderer(pipelines)
    frames = []
    for seed in (1, 2):
        task = job(FAST, Mode.RETAKE, [(InputRole.SOURCE_VIDEO, media["clip"], {"start_s": 1.0, "end_s": 1.5})], seed=seed)
        for item in task.inputs:
            item.save(tmp_path / str(seed))
        call = build_call(task)
        frames.append([np.asarray(f) for f in render_edit(renderer, call, call.pop("edit"))["videos"]])
    assert np.array_equal(frames[0][0], frames[1][0])
    assert not all(np.array_equal(a, b) for a, b in zip(frames[0][17:41], frames[1][17:41]))


def test_retake_pins_hold_in_both_passes_of_the_two_stage_recipe(pipelines, media, tmp_path):
    renderer = TinyPinnedRenderer(pipelines)
    task = job(FAST, Mode.RETAKE, [(InputRole.SOURCE_VIDEO, media["clip"], {"start_s": 0.5, "end_s": 1.5})])
    for item in task.inputs:
        item.save(tmp_path)
    call = {**build_call(task), "second_stage_sigmas": ltx_resident.SECOND_STAGE_SIGMAS}
    assert renderer.stages(call) == ("half", "full")
    out = render_edit(renderer, call, call.pop("edit"))
    report = out["edit"]
    assert report["pins_exact"] == {"half": True, "full": True}
    assert report["seen_clean"] == {"half": {"video": True, "audio": True}, "full": {"video": True, "audio": True}}
    assert len(out["videos"]) == 49


def test_held_tokens_that_move_fail_the_job(pipelines, media, tmp_path):
    class Drifting(TinyPinnedRenderer):
        def render(self, call, pins: dict[str, Pins]) -> Rendered:
            rendered = super().render(call, pins)
            rendered.pins_exact = {stage: False for stage in rendered.pins_exact}
            return rendered

    task = job(FAST, Mode.RETAKE, [(InputRole.SOURCE_VIDEO, media["clip"], {"start_s": 0.5, "end_s": 1.5})])
    for item in task.inputs:
        item.save(tmp_path)
    call = build_call(task)
    with pytest.raises(EditError, match="changed during denoising"):
        render_edit(Drifting(pipelines), call, call.pop("edit"))


def test_a_source_much_shorter_than_the_retake_is_refused(pipelines, tmp_path):
    short = tmp_path / "short.mp4"
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate=24:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(short))
    task = job(FAST, Mode.RETAKE, [(InputRole.SOURCE_VIDEO, short, {"start_s": 0.2, "end_s": 0.6})])
    for item in task.inputs:
        item.save(tmp_path)
    call = build_call(task)
    with pytest.raises(EditError, match="has 24 frames at 24 fps; this retake renders 49"):
        render_edit(TinyPinnedRenderer(pipelines), call, call.pop("edit"))
