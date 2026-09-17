"""Audio-to-video and retake without a GPU (backends/ltx_edit.py): which latents a window regenerates and which are held,
the customer's media decoded as the job renders it, the sound a job returns, the calls build_call makes for both modes,
memory admission for encoding the source, and no step commitment on a verified class. The render itself, through
diffusers' real classes with tiny weights, is test_ltx_edit_render.py."""

from __future__ import annotations

import math
import random
import subprocess
import uuid
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kuno_protocol.profiles import InputRole, Mode, load_profiles, ltx_num_frames
from kuno_protocol.schemas import GenerationParams, InputRef
from kuno_worker.backends.base import GenerationTask, InputFile
from kuno_worker.backends.ltx_edit import (
    AUDIO_TO_VIDEO,
    RETAKE,
    EditError,
    audio_latent_times,
    decode_audio,
    decode_frames,
    fit_samples,
    plan_edit,
    splice_audio,
    to_int16,
    video_latent_times,
)
from kuno_worker.backends.ltx_pinning import Geometry
from kuno_worker.backends.ltx_resident import LtxResidentBackend, build_call
from kuno_worker.backends.media_tools import BackendError, CapacityRefused, _write_wav, ffmpeg_exe
from kuno_worker.backends.quantized import MemoryPlan, admit, latent_tokens, source_encode_gib, source_held_gib

PROFILES = load_profiles()
FAST, PRO = PROFILES["ltx-2.5-fast"], PROFILES["ltx-2.5-pro"]
G24 = Geometry(fps=24.0)


def retake(start_s=0.0, end_s=None, video=True, audio=True) -> dict:
    return {"mode": RETAKE, "video_path": "clip.mp4", "start_s": start_s, "end_s": end_s, "regenerate_video": video, "regenerate_audio": audio}


# ---------------------------------------------------------------- the plan


def test_a_window_rounds_out_to_whole_latents():
    # ltx_edit's docstring example: 1 s to 3 s of a 5 s clip at 24 fps.
    plan = plan_edit(G24, 121, retake(1.0, 3.0))
    assert (plan.latent_frames, plan.audio_latents, plan.samples) == (16, 126, 242_000)
    assert plan.video_span == (3, 10) and plan.frame_span == (17, 73)  # 0.708 s to 3.042 s
    assert plan.audio_span == (25, 76) and plan.sample_span == (46_560, 144_480)  # 0.97 s to 3.01 s
    assert plan.held_video == [(0, 3), (10, 16)] and plan.held_audio == [(0, 25), (76, 126)]


def test_the_latent_times_follow_the_causal_vaes():
    assert video_latent_times(G24, 3) == [(0.0, 1 / 24), (1 / 24, 9 / 24), (9 / 24, 17 / 24)]  # frame 0 alone, then 8 at a time
    times = audio_latent_times(G24, 3)
    assert times[0] == (0.0, 0.01) and times[1] == pytest.approx((0.01, 0.05)) and times[2] == pytest.approx((0.05, 0.09))


@pytest.mark.parametrize("fps", [24, 25, 48, 50])
def test_what_is_regenerated_covers_the_window_and_nothing_held_touches_it(fps):
    g = Geometry(fps=float(fps))
    rng = random.Random(fps)
    for seconds in (2, 5, 10):
        frames = ltx_num_frames(seconds, fps)
        for _ in range(40):
            start = rng.uniform(0, frames / fps - 0.05)
            end = rng.uniform(start + 0.01, frames / fps + 0.5)
            plan = plan_edit(g, frames, retake(start, end))
            p0, p1 = plan.frame_span
            for f in range(frames):
                overlaps = (f + 1) / fps > start + 1e-9 and f / fps < min(end, frames / fps) - 1e-9
                assert not overlaps or p0 <= f < p1, (seconds, start, end, f)
            # Whole latent frames: the span starts on a latent's first frame and ends after one's last.
            assert p0 == 0 or (p0 - 1) % 8 == 0
            assert (p1 - 1) % 8 == 0
            s0, s1 = plan.sample_span
            assert s0 <= start * g.sample_rate + 1e-6
            assert s1 == plan.samples or min(end, frames / fps) * g.sample_rate <= s1 + 1e-6
            a, b = plan.audio_span
            assert s0 == (0 if a == 0 else (4 * a - 3) * 480)


def test_a_modality_not_regenerated_is_held_entirely():
    audio_only = plan_edit(G24, 49, retake(0.5, 1.5, video=False))
    assert audio_only.video_span == (0, 0) and audio_only.held_video == [(0, 7)] and audio_only.frame_span == (0, 0)
    assert audio_only.audio_span != (0, 0)
    video_only = plan_edit(G24, 49, retake(0.5, 1.5, audio=False))
    assert video_only.held_audio == [(0, 51)] and video_only.sample_span == (0, 0)
    whole = plan_edit(G24, 49, retake())
    assert whole.held_video == [] and whole.held_audio == [] and whole.sample_span == (0, whole.samples)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (retake(2.1, 3.0), "starts at or after the clip's end"),
        (retake(1.0, 1.0), "end_s must be after"),
        (retake(-1.0, 1.0), "non-negative"),
        (retake(0.0, math.nan), "non-negative"),
        (retake(0.5, 1.0, video=False, audio=False), "must regenerate"),
        ({"mode": "dub"}, "unknown edit mode"),
        ({"mode": AUDIO_TO_VIDEO, "audio_path": "a.wav", "start_s": 2.0, "end_s": 1.0}, "end_s must be after"),
    ],
)
def test_impossible_edits_are_refused(edit, message):
    with pytest.raises(EditError, match=message):
        plan_edit(G24, 49, edit)


def test_a_window_past_the_clip_is_cut_at_its_end():
    plan = plan_edit(G24, 49, retake(1.0, 9.0))
    assert plan.end_s == pytest.approx(49 / 24) and plan.video_span[1] == 7 and plan.sample_span[1] == plan.samples


def test_audio_to_video_holds_every_audio_latent_and_generates_every_frame():
    plan = plan_edit(G24, 49, {"mode": AUDIO_TO_VIDEO, "audio_path": "a.wav", "start_s": 0.5, "end_s": None})
    assert plan.held_audio == [(0, 51)] and plan.held_video == [] and plan.frame_span == (0, 49)
    assert (plan.start_s, plan.end_s) == (0.5, pytest.approx(0.5 + 49 / 24))
    short = plan_edit(G24, 49, {"mode": AUDIO_TO_VIDEO, "audio_path": "a.wav", "start_s": 0.5, "end_s": 1.0})
    assert short.end_s == 1.0  # the rest is silence, still held


# ---------------------------------------------------------------- the customer's media


def ffmpeg(*args: str) -> None:
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def write_pcm(path: Path, samples: np.ndarray, rate: int) -> None:
    """(channels, n) int16 as a plain 16-bit WAV."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(samples.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.ascontiguousarray(samples.T).tobytes())


def test_source_frames_are_conformed_to_the_render(tmp_path):
    clip = tmp_path / "clip.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2.1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip))
    frames = decode_frames(clip, fps=24, width=128, height=64, count=49)
    assert frames.shape == (49, 64, 128, 3) and frames.dtype == np.uint8
    assert not np.array_equal(frames[0], frames[24])  # testsrc2 moves: these are different moments
    short = decode_frames(clip, fps=24, width=128, height=64, count=200)
    assert 50 <= len(short) <= 51  # 2.1 s at 24 fps: what the clip has (the fps filter rounds the end), never padded here


def test_source_frames_are_scaled_to_cover_and_centre_cropped(tmp_path):
    # A 4:1 picture, left half red and right half blue, into a square: the crop keeps the middle, so both colours meet at
    # the centre and fill the frame top to bottom.
    image = tmp_path / "wide.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=red:s=200x100:r=24:d=1", "-f", "lavfi", "-i", "color=blue:s=200x100:r=24:d=1",
           "-filter_complex", "[0:v][1:v]hstack,format=yuv444p", "-c:v", "libx264", "-crf", "0", str(image))
    [frame] = decode_frames(image, fps=24, width=64, height=64, count=1)
    assert frame[32, 5, 0] > 200 and frame[32, 5, 2] < 60  # red on the left
    assert frame[32, 58, 2] > 200 and frame[32, 58, 0] < 60  # blue on the right


def test_source_audio_is_sample_exact_from_its_start(tmp_path):
    rng = np.random.default_rng(1)
    stereo = rng.integers(-20000, 20000, size=(2, 96_000), dtype=np.int16)
    path = tmp_path / "stereo.wav"
    write_pcm(path, stereo, 48_000)
    audio = decode_audio(path, sample_rate=48_000, start_s=0.5, duration_s=1.0)
    assert audio.dtype == np.int16 and np.array_equal(audio, stereo[:, 24_000:72_000])
    assert decode_audio(path, sample_rate=48_000, start_s=5.0).shape == (2, 0)


def test_mono_is_doubled_and_surround_downmixed_and_silence_is_none(tmp_path):
    mono = np.random.default_rng(2).integers(-9000, 9000, size=(1, 48_000), dtype=np.int16)
    write_pcm(tmp_path / "mono.wav", mono, 48_000)
    doubled = decode_audio(tmp_path / "mono.wav", sample_rate=48_000)
    assert doubled.shape == (2, 48_000) and np.array_equal(doubled[0], mono[0]) and np.array_equal(doubled[1], mono[0])

    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1", "-ac", "1", str(tmp_path / "cd.wav"))
    resampled = decode_audio(tmp_path / "cd.wav", sample_rate=48_000)
    assert resampled.shape[0] == 2 and abs(resampled.shape[1] - 48_000) <= 48 and np.array_equal(resampled[0], resampled[1])

    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1", "-filter_complex",
           "[0:a]pan=5.1|c0=c0|c1=c0|c2=c0|c3=c0|c4=c0|c5=c0[a]", "-map", "[a]", str(tmp_path / "six.wav"))
    assert decode_audio(tmp_path / "six.wav", sample_rate=48_000).shape == (2, 48_000)

    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp_path / "silent.mp4"))
    assert decode_audio(tmp_path / "silent.mp4", sample_rate=48_000) is None


def test_the_sound_a_job_returns_is_what_the_encoder_writes(tmp_path):
    floats = np.random.default_rng(3).uniform(-1.2, 1.2, size=(2, 1000)).astype(np.float32)
    _write_wav(tmp_path / "out.wav", floats, 48_000)
    with wave.open(str(tmp_path / "out.wav")) as handle:
        written = np.frombuffer(handle.readframes(1000), dtype="<i2").reshape(-1, 2).T
    assert np.array_equal(written, to_int16(floats))  # so int16 samples pass media_tools untouched, as the source's must
    assert fit_samples(np.ones((2, 5), np.int16), 8).tolist() == [[1] * 5 + [0] * 3] * 2
    assert fit_samples(np.ones((2, 5), np.int16), 3).shape == (2, 3)


def test_a_splice_keeps_the_held_sound_exactly_and_fades_inside_the_span():
    rng = np.random.default_rng(4)
    source = rng.integers(-30000, 30000, size=(2, 10_000), dtype=np.int16)
    render = rng.uniform(-0.5, 0.5, size=(2, 9_000)).astype(np.float32)  # a vocoder's output ends short of the video
    out = splice_audio(source, render, (2_000, 6_000), 100)
    assert np.array_equal(out[:, :2_000], source[:, :2_000]) and np.array_equal(out[:, 6_000:], source[:, 6_000:])
    assert np.array_equal(out[:, 2_100:5_900], to_int16(render)[:, 2_100:5_900])
    assert not np.array_equal(out[:, 2_000:2_100], to_int16(render)[:, 2_000:2_100])  # crossfaded
    tail = splice_audio(source, render, (8_000, 10_000), 100)
    assert np.array_equal(tail[:, 9_000:], np.zeros((2, 1_000), np.int16))  # past the render: silence, no fade at the end
    whole = splice_audio(source, render[:1], (0, 10_000), 100)
    assert np.array_equal(whole[:, :9_000], np.repeat(to_int16(render[:1]), 2, axis=0))  # no held sound to fade from


# ---------------------------------------------------------------- the calls


def edit_task(profile, mode: Mode, tmp_path, *, roles, options=None, refs=None, audio=True, duration_s=2.0) -> GenerationTask:
    params = GenerationParams(
        profile_id=profile.id, mode=mode, duration_s=duration_s, resolution="720p", aspect_ratio="16:9", fps=24, audio=audio, input_roles=roles,
    )
    mimes = {InputRole.SOURCE_VIDEO: "video/mp4", InputRole.SOURCE_AUDIO: "audio/wav", InputRole.FIRST_FRAME: "image/png"}
    inputs = []
    for index, role in enumerate(roles):
        ref = InputRef(index=index, role=role, mime=mimes[role], sha256="0" * 64, size=1, **((refs or {}).get(role) or {}))
        inputs.append(InputFile(ref=ref, data=b"x", mime=mimes[role]))
        inputs[-1].save(tmp_path)
    return GenerationTask(
        job_id=str(uuid.uuid4()), profile=profile, params=params, prompt="p", negative_prompt=None, seed=1, width=1280, height=704,
        inputs=inputs, options=options or {},
    )


def test_a_retake_window_comes_from_the_options_else_the_source_input(tmp_path):
    roles = [InputRole.SOURCE_VIDEO]
    from_ref = build_call(edit_task(FAST, Mode.RETAKE, tmp_path, roles=roles, refs={InputRole.SOURCE_VIDEO: {"start_s": 0.5, "end_s": 1.25}}))
    assert from_ref["edit"]["start_s"] == 0.5 and from_ref["edit"]["end_s"] == 1.25 and from_ref["edit"]["video_path"].endswith(".mp4")
    overridden = build_call(edit_task(
        FAST, Mode.RETAKE, tmp_path, roles=roles, refs={InputRole.SOURCE_VIDEO: {"start_s": 0.5, "end_s": 1.25}},
        options={"retake": {"start_s": 1, "end_s": 2}, "regenerate_audio": False},
    ))
    assert overridden["edit"] | {"video_path": None} == {
        "mode": "retake", "video_path": None, "start_s": 1.0, "end_s": 2.0, "regenerate_video": True, "regenerate_audio": False,
    }
    silent = build_call(edit_task(FAST, Mode.RETAKE, tmp_path, roles=roles, audio=False))
    assert (silent["edit"]["start_s"], silent["edit"]["end_s"], silent["edit"]["regenerate_audio"]) == (0.0, None, False)


@pytest.mark.parametrize(
    "options", [{"retake": "1-2"}, {"retake": {"start_s": "1"}}, {"retake": {"end_s": math.inf}}, {"regenerate_video": "no"}, {"regenerate_audio": 1}],
)
def test_malformed_retake_options_fail_the_job_without_echoing_them(tmp_path, options):
    with pytest.raises(BackendError) as refused:
        build_call(edit_task(FAST, Mode.RETAKE, tmp_path, roles=[InputRole.SOURCE_VIDEO], options=options))
    assert "no" not in str(refused.value).split() and "1-2" not in str(refused.value)


def test_audio_to_video_holds_the_sound_and_conditions_the_first_frame(tmp_path):
    task = edit_task(
        PRO, Mode.AUDIO_TO_VIDEO, tmp_path, roles=[InputRole.SOURCE_AUDIO, InputRole.FIRST_FRAME],
        refs={InputRole.SOURCE_AUDIO: {"start_s": 1.5, "end_s": 4.0}}, duration_s=3.0,
    )
    call = build_call(task)
    assert call["edit"] | {"audio_path": None} == {"mode": "audio_to_video", "audio_path": None, "start_s": 1.5, "end_s": 4.0}
    assert call["conditions"] == [{"path": task.inputs[1].path, "index": 0, "strength": 1.0}]
    assert call["pipeline"] == "condition" and call["num_frames"] == 73 and call["num_inference_steps"] == 30


# ---------------------------------------------------------------- memory


def plan_with(max_tokens: int) -> MemoryPlan:
    """A plan with the distilled bf16 recipe's measured activation slope, 4.6 GiB per 10k latent tokens, capped at `max_tokens`."""
    per_token = 4.6 / 10_000
    return MemoryPlan(
        recipe_id="synthetic", hardware_class="O1.synthetic", offload="none", usable_gib=10 + 1 + max_tokens * per_token + 1e-9,
        floor_gib=0.0, token_base_gib=10.0, per_token_gib=per_token, overhead_gib=1.0, host_ram_gib=0.0,
    )


def text_call(duration_s: float) -> dict:
    return build_call(GenerationTask(
        job_id="j", profile=FAST, params=GenerationParams(profile_id=FAST.id, mode=Mode.TEXT_TO_VIDEO, duration_s=duration_s, resolution="720p",
                                                          aspect_ratio="16:9", fps=24), prompt="p", negative_prompt=None, seed=1, width=1280, height=704,
    ))


def test_a_retake_source_encodes_in_memory_set_by_its_frame_size_not_its_length():
    # The chunked encode: 2,900 bytes per source pixel beside the weights, whatever the clip's length; audio-to-video's
    # sound fits in the held tokens' margin, which both modes keep through the render.
    assert source_encode_gib("retake", 1280, 704) == pytest.approx(2900 * 1280 * 704 / 2**30) == pytest.approx(2.43, abs=0.005)
    assert source_encode_gib("retake", 1920, 1088) == pytest.approx(5.64, abs=0.005)
    assert source_encode_gib("audio_to_video", 1280, 704) == 0.0 and source_encode_gib(None, 1280, 704) == 0.0
    assert source_held_gib("retake") == source_held_gib("audio_to_video") == 0.25 and source_held_gib(None) == 0.0


def test_a_retake_holds_its_tokens_through_the_render(tmp_path):
    roles = [InputRole.SOURCE_VIDEO]
    tokens = latent_tokens(1280, 704, 121)
    plan = plan_with(tokens)
    assert admit(plan, FAST, text_call(5), 1280, 704, 24) == tokens  # a clip of this length fits exactly
    with pytest.raises(CapacityRefused) as refused:
        admit(plan, FAST, build_call(edit_task(FAST, Mode.RETAKE, tmp_path, roles=roles, duration_s=5.0)), 1280, 704, 24)
    # The held tokens' 0.25 GiB costs 543 of the 5 s clip's 14,080 tokens; 4 s needs 11,440.
    message = str(refused.value)
    assert "2.4 GiB to encode its source" in message and "a retake job serves up to 4 s" in message
    assert admit(plan, FAST, build_call(edit_task(FAST, Mode.RETAKE, tmp_path, roles=roles, duration_s=4.0)), 1280, 704, 24)


def test_the_encode_peaks_beside_the_weights_not_beside_the_render(tmp_path):
    # The encode runs and frees before the pipeline is called, so a job needs the larger of the two peaks, not their sum:
    # here the render leaves no room for the encode's 2.43 GiB, and the retake still fits.
    retake = build_call(edit_task(FAST, Mode.RETAKE, tmp_path, roles=[InputRole.SOURCE_VIDEO], duration_s=5.0))
    tokens = latent_tokens(1280, 704, 121)
    plan = plan_with(tokens + 544)
    assert plan.usable_gib - plan.estimate_gib(tokens) - 0.25 < source_encode_gib("retake", 1280, 704)
    assert admit(plan, FAST, retake, 1280, 704, 24) == tokens
    # With 2.5 GiB free beside the resident weights, the 2.43 GiB encode and 0.25 GiB of held tokens don't fit at any length.
    crowded = replace(plan, resident_gib=plan.usable_gib - plan.overhead_gib - 2.5)
    assert admit(crowded, FAST, text_call(5), 1280, 704, 24) == tokens
    with pytest.raises(CapacityRefused, match=r"2\.4 GiB to encode its source\): about 17\.9 GiB .* a retake job cannot serve 1280x704 at any duration"):
        admit(crowded, FAST, build_call(edit_task(FAST, Mode.RETAKE, tmp_path, roles=[InputRole.SOURCE_VIDEO], duration_s=2.0)), 1280, 704, 24)


def test_a_card_plans_the_encode_beside_the_weights_a_render_keeps(tmp_path):
    # On an RTX PRO 6000 without offload the enhancer waits in host RAM: 66.18 GiB of weights, 1.5 GiB of overhead.
    backend = LtxResidentBackend(None, tmp_path, loader=lambda profile: None, hardware_class="C1.rtx-pro-6000-bw-se.x1", host_ram_gib=512, device_gib=94.97)
    plan = backend.memory_plan(FAST)
    assert plan.offload == "none" and plan.resident_gib == pytest.approx(66.18)
    assert plan.encode_gib(source_encode_gib("retake", 1920, 1088)) == pytest.approx(66.18 + 5.64 + 1.5, abs=0.01)
    # Retakes serve what text-to-video does at 720p (18 s at 24 fps; 15 s when the encode was counted whole and beside the
    # render), and 7 s of 1080p's 8, where the measured line leaves 0.01 GiB.
    from kuno_worker.backends.quantized import _longest_fitting, longest_duration

    assert longest_duration(plan, FAST, 1280, 704, 24) == _longest_fitting(plan, FAST, 1280, 704, 24, "retake") == 18
    assert (longest_duration(plan, FAST, 1920, 1088, 24), _longest_fitting(plan, FAST, 1920, 1088, 24, "retake")) == (8, 7)


def test_a_source_clip_encodes_in_chunks_of_whole_latent_frames():
    from kuno_worker.backends.ltx_chunked_encode import CHUNK_LATENT_FRAMES, chunk_bounds

    # Frame 0 is latent frame 0 alone; every later chunk is whole latent frames of 8, so every chunk folds evenly at each of
    # the encoder's 2x temporal downsamplings. The last chunk ends with the clip.
    assert CHUNK_LATENT_FRAMES == 1 and chunk_bounds(25, 8) == [(0, 1), (1, 9), (9, 17), (17, 25)]
    assert chunk_bounds(49, 8, 2) == [(0, 1), (1, 17), (17, 33), (33, 49)] and chunk_bounds(41, 8, 3) == [(0, 1), (1, 25), (25, 41)]
    assert chunk_bounds(1, 8) == [(0, 1)]
    for frames in (0, 24, 120):
        with pytest.raises(BackendError, match="needs 1 \\+ 8k frames"):
            chunk_bounds(frames, 8)


def test_a_card_that_fits_every_clip_still_admits_retakes_by_memory(tmp_path, monkeypatch):
    # A 288 GiB card holds every ltx-2.5-fast clip, so text-to-video has no plan and no admission. A retake still needs its
    # source encoded, so it is checked against that card's whole-GPU plan: with the source's cost inflated a hundredfold
    # here, that plan refuses it.
    from kuno_worker.backends import quantized

    backend = LtxResidentBackend(None, tmp_path, loader=lambda profile: None, host_ram_gib=512, device_gib=288.0)
    assert backend.memory_plan(FAST) is None

    def job(mode, roles):
        params = GenerationParams(profile_id=FAST.id, mode=mode, duration_s=20, resolution="1080p", aspect_ratio="16:9", fps=25, input_roles=roles)
        task = GenerationTask(job_id="j", profile=FAST, params=params, prompt="p", negative_prompt=None, seed=1, width=1920, height=1088)
        if roles:
            task.inputs = edit_task(FAST, Mode.RETAKE, tmp_path, roles=roles).inputs
        return task

    text, retake_job = job(Mode.TEXT_TO_VIDEO, []), job(Mode.RETAKE, [InputRole.SOURCE_VIDEO])
    backend.admit(retake_job, build_call(retake_job))  # 130 GiB for the tokens, and 5.6 GiB beside the weights to encode the source
    monkeypatch.setattr(quantized, "SOURCE_ENCODE_BYTES_PER_PIXEL", 290_000)
    backend.admit(text, build_call(text))
    with pytest.raises(CapacityRefused, match="to encode its source"):
        backend.admit(retake_job, build_call(retake_job))


# ---------------------------------------------------------------- verified mode


def test_edits_carry_no_step_commitment_on_a_verified_class(tmp_path):
    backend = LtxResidentBackend(None, tmp_path / "work", loader=lambda profile: None, hardware_class="C1.rtx-pro-6000-bw-se.x1", device_gib=94.97, host_ram_gib=512)
    text = GenerationTask(
        job_id=str(uuid.uuid4()), profile=FAST,
        params=GenerationParams(profile_id=FAST.id, mode=Mode.TEXT_TO_VIDEO, duration_s=2, resolution="720p", aspect_ratio="16:9", fps=24),
        prompt="p", negative_prompt=None, seed=1, width=1280, height=704,
    )
    recorder = backend.step_recorder(text)
    assert recorder is not None
    recorder.abort()
    assert backend.step_recorder(edit_task(FAST, Mode.RETAKE, tmp_path, roles=[InputRole.SOURCE_VIDEO])) is None
    assert backend.step_recorder(edit_task(PRO, Mode.AUDIO_TO_VIDEO, tmp_path, roles=[InputRole.SOURCE_AUDIO])) is None


def test_a_retake_on_a_verified_class_renders_without_a_tap(tmp_path):
    calls = []

    def pipeline(**call):
        calls.append(call)
        return {"videos": [[np.zeros((64, 64, 3), dtype=np.uint8)] * 49], "audio": [np.zeros((2, 98_000), dtype=np.int16)], "sampling_rate": 48_000}

    backend = LtxResidentBackend(None, tmp_path / "work", loader=lambda profile: pipeline, hardware_class="C1.rtx-pro-6000-bw-se.x1", device_gib=94.97, host_ram_gib=512)
    task = edit_task(FAST, Mode.RETAKE, tmp_path, roles=[InputRole.SOURCE_VIDEO])
    task.width = task.height = 64
    result = backend.generate(task, lambda value, stage: None)
    assert result.step_commitment is None and result.openings is None and result.info.frames == 49
    [call] = calls
    assert "kuno_trajectory_tap" not in call and call["edit"]["mode"] == "retake"
