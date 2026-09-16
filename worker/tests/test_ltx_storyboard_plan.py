"""The storyboard join plan (backends/ltx_storyboard.py), which a GPU run can't afford to get wrong: which latents are
pinned, how many frames and samples each join trims, where every shot lands in the stitched video, and that its length is
the protocol's. Plain data: no torch."""

from __future__ import annotations

import numpy as np
import pytest

from kuno_protocol.profiles import load_profiles, ltx_num_frames, storyboard_frames
from kuno_protocol.schemas import ShotSpec
from kuno_worker.backends.ltx_storyboard import ANCHORS, Geometry, StoryboardError, Timeline, assemble_audio, predicted_timeline

G24 = Geometry(fps=24.0)
FAST = load_profiles()["ltx-2.5-fast"]


def plan(joins: list[str], durations: list[float], fps: int = 24, overlap: int = 3, anchor: str = "previous") -> Timeline:
    frames = [ltx_num_frames(d, fps) for d in durations]
    return predicted_timeline(Geometry(fps=float(fps)), joins, frames, overlap, anchor)


# ---------------------------------------------------------------- geometry


@pytest.mark.parametrize("fps", [24, 25, 48, 50])
@pytest.mark.parametrize("duration", [2, 3, 5, 8, 10])
def test_latent_frames_decode_back_to_the_clip(fps, duration):
    g = Geometry(fps=float(fps))
    frames = ltx_num_frames(duration, fps)
    assert (frames - 1) % 8 == 0
    assert g.pixel_frames(g.latent_frames(frames)) == frames


def test_pinned_head_frame_counts():
    # The causal VAE's first latent frame is one frame, every later one eight.
    assert [G24.pixel_frames(k) for k in (1, 2, 3, 4, 6)] == [1, 9, 17, 25, 41]
    assert G24.latent_frames(49) == 7 and G24.latent_frames(121) == 16


def test_audio_geometry_matches_the_gpu_measurement():
    # LTX-2.5 on the RTX PRO 6000, 2026-09-15: 49 frames at 24 fps came back with 96,480 samples at 48 kHz.
    assert G24.audio_latents_per_second == 25.0
    assert G24.samples_per_mel == 480 and G24.samples_per_frame == 2000
    assert G24.audio_latents(49) == 51
    assert G24.mel_frames(51) == 201
    assert G24.audio_samples(51) == 96_480


@pytest.mark.parametrize("fps", [24, 25, 48, 50])
def test_audio_latent_count_is_the_pipelines_own_rounding(fps):
    g = Geometry(fps=float(fps))
    for frames in range(9, 1000, 8):
        duration_s = frames / float(fps)  # pipeline_ltx2_condition.py: round(duration_s * audio_latents_per_second)
        assert g.audio_latents(frames) == round(duration_s * (16000 / 160 / float(4)))


# ---------------------------------------------------------------- tails and trims


def test_audio_tail_for_two_second_shots():
    join = plan(["fresh", "continue"], [2, 2]).joins[1]
    assert (join.video_pin, join.video_trim) == (3, 17)
    assert join.audio_pin == 18 and join.audio_trim == 69 * 480 == 33_120
    assert join.audio_continuous and join.audio_source == 0
    assert join.sync_error_samples == -640  # -13.3 ms


def test_a_proportional_audio_tail_would_be_far_off():
    """JoyLTX25's round(audio_latents * overlap / latent_frames) against the count the timeline picks, for 2 s shots."""
    g, overlap, latents, audio = G24, 3, 7, 51
    proportional = round(audio * overlap / latents)
    assert proportional == 22
    video_head = g.pixel_frames(overlap) * g.samples_per_frame  # what the join trims from the picture
    shift_video = (g.pixel_frames(latents) - g.pixel_frames(overlap)) * g.samples_per_frame  # shot N+1's frame 0 in shot N
    for k, replayed_ms, skew_ms in ((proportional, 141.7, -173.3), (plan(["fresh", "continue"], [2, 2]).joins[1].audio_pin, None, -13.3)):
        shift_audio = g.audio_ratio * (audio - k) * g.samples_per_mel  # shot N+1's first sample in shot N
        assert round(float(shift_audio - shift_video) / 48, 1) == skew_ms
        if replayed_ms is not None:  # trimming only the picture's span would leave this much of shot N's sound in again
            assert round(float(g.audio_samples(k) - video_head) / 48, 1) == replayed_ms


def test_trims_and_kept_counts_are_exact():
    timeline = plan(["fresh", "continue", "continue", "continue"], [2, 2, 2, 2])
    assert [j.frames for j in timeline.joins] == [49, 49, 49, 49]
    assert [j.video_trim for j in timeline.joins] == [0, 17, 17, 17]
    assert [j.kept_frames for j in timeline.joins] == [49, 32, 32, 32]
    assert [j.video_start for j in timeline.joins] == [0, 49, 81, 113]
    for before, after in zip(timeline.joins, timeline.joins[1:]):
        assert after.video_start == before.video_start + before.kept_frames
        assert after.audio_start == before.audio_start + before.audio_keep
        if after.audio_continuous:
            assert before.audio_keep == before.decoded_samples - before.audio_trim  # nothing padded, nothing cut
            assert before.audio_pad == 0
    assert timeline.total_frames == 145
    assert timeline.total_samples == 145 * 2000
    last = timeline.joins[-1]
    assert last.audio_start + last.audio_keep == timeline.total_samples


@pytest.mark.parametrize("overlap", [1, 3, 5])
@pytest.mark.parametrize("fps", [24, 25, 48, 50])
def test_stitched_duration_is_the_shots_minus_the_overlaps(fps, overlap):
    joins = ["fresh", "continue", "cut", "continue", "fresh", "cut", "continue"]
    durations = [2, 3, 5, 2, 4, 6, 3]
    timeline = plan(joins, durations, fps=fps, overlap=overlap)
    g = timeline.geometry
    overlaps = sum(g.pixel_frames(overlap) for j in joins if j != "fresh")
    assert timeline.total_frames == sum(ltx_num_frames(d, fps) for d in durations) - overlaps
    assert timeline.total_samples == timeline.total_frames * g.samples_per_frame


@pytest.mark.parametrize("fps", [24, 25, 48, 50])
def test_the_plan_stitches_exactly_the_frames_the_protocol_bills(fps):
    """The planner at the profile's overlap and `profiles.storyboard_frames`, which prices the job and sets its duration_s,
    must agree on every storyboard."""
    overlap = FAST.limits.storyboard.overlap_latent_frames
    for joins, durations in (
        (["fresh"] + ["continue"] * 7, [5] * 8),
        (["fresh", "continue", "cut", "fresh"], [3, 3, 3, 3]),
        (["fresh", "cut", "continue", "fresh", "continue", "cut"], [2, 10, 4, 7, 2, 9]),
    ):
        shots = [ShotSpec(duration_s=d, join=j) for j, d in zip(joins, durations)]
        timeline = plan(joins, durations, fps=fps, overlap=overlap, anchor="first")
        assert timeline.total_frames == storyboard_frames(FAST, shots, fps)


@pytest.mark.parametrize("anchor", ANCHORS)
@pytest.mark.parametrize("fps", [24, 25, 48, 50])
@pytest.mark.parametrize("overlap", [1, 2, 3, 5])
def test_sync_error_stays_within_half_an_audio_latent_and_never_adds_up(anchor, fps, overlap):
    joins = ["fresh"] + ["continue", "continue", "cut", "continue", "cut", "cut", "continue"] * 4
    durations = [2, 3, 2, 5, 4, 2, 7, 3] * 3 + [2, 3, 4, 5, 6]
    timeline = plan(joins, durations[: len(joins)], fps=fps, overlap=overlap, anchor=anchor)
    half_latent = timeline.geometry.audio_ratio * timeline.geometry.samples_per_mel / 2  # 960 samples, 20 ms
    for join in timeline.joins:
        if join.audio_continuous:
            assert abs(join.sync_error_samples) <= half_latent
        else:  # nothing to run through: padded or cut into sync, to the sample
            assert abs(join.sync_error_samples) <= 0.5
    # Continuous joins only ever correct toward zero, so the last shot is no further off than the first join.
    assert abs(timeline.joins[-1].sync_error_samples) <= half_latent


def test_an_overlap_too_short_for_the_sound_pads_instead_of_drifting():
    # 2 s at 50 fps: 97 frames (93,120 samples of picture) but 48 audio latents (90,720 samples), 50 ms short. A one-frame
    # head trims 20 ms; even a one-latent audio pin would start the next shot's sound 40 ms early, and again at every join.
    timeline = plan(["fresh", "continue", "continue", "cut"], [2, 2, 2, 2], fps=50, overlap=1, anchor="previous")
    for join in timeline.joins[1:]:
        assert not join.audio_continuous and join.audio_source == join.index - 1
        assert join.audio_pin >= 1 and abs(join.sync_error_samples) <= 0.5
        assert "padded into sync" in join.notes[0]
    assert plan(["fresh", "continue"], [2, 2], fps=50, overlap=2).joins[1].audio_continuous  # 9 frames is enough


def test_audio_pins_come_from_the_first_shot_of_each_run_by_default():
    joins = ["fresh", "continue", "cut", "fresh", "cut", "continue"]
    first = predicted_timeline(G24, joins, [49] * 6, 3)  # the protocol's anchor
    previous = plan(joins, [2] * 6, anchor="previous")
    assert first.anchor == "first"
    assert [j.audio_source for j in first.joins] == [None, 0, 0, None, 3, 3]
    assert [j.audio_source for j in previous.joins] == [None, 0, 1, None, 3, 4]
    assert [j.video_source for j in first.joins] == [None, 0, None, None, None, 4]  # the picture always continues the previous shot
    assert [j.audio_continuous for j in first.joins] == [False, True, False, False, True, False]
    assert all(j.audio_pin == 0 and j.audio_trim == 0 and j.video_trim == 0 for j in first.joins if j.join == "fresh")


def test_anchored_and_fresh_joins_pad_the_previous_audio_into_sync():
    # Shot 2's pin is shot 1's own tail (continuous); shot 3's is anchored to shot 1 and shot 4 is fresh, so neither can run
    # through and the audio before each is padded to put the shot's audio clock exactly on its video clock.
    timeline = plan(["fresh", "cut", "cut", "fresh"], [2, 2, 2, 2], anchor="first")
    assert [j.audio_continuous for j in timeline.joins] == [False, True, False, False]
    assert [j.audio_pin for j in timeline.joins] == [0, 18, 18, 0]
    assert [j.sync_error_samples for j in timeline.joins] == [0, -640, 0, 0]
    # 49 frames are 98,000 samples of picture and the vocoder returns 96,480: about 32 ms of silence where a join can't run through.
    assert [j.audio_pad for j in timeline.joins] == [0, 1280, 1520, 1520]
    assert [j.audio_start for j in timeline.joins] == [0, 96_480, 161_120, 226_000]


def test_stitched_audio_and_video_run_through_continuous_seams():
    """Clock arrays standing in for pictures and sound: shot N+1's own frame f is shot N's frame F - trim + f, and its
    sample s is shot N's sample S - trim + s. Stitched, a continuous seam must not repeat or skip a single one."""
    joins = ["fresh", "continue", "continue", "cut", "fresh", "cut", "continue"]
    timeline = plan(joins, [2, 3, 2, 4, 2, 3, 2], overlap=3, anchor="previous")
    video_clocks, audio_clocks = [], []
    for join in timeline.joins:
        video_base = video_clocks[-1][-1] + 1 - join.video_trim if join.video_pin else 100_000 * (join.index + 1)
        video_clocks.append(video_base + np.arange(join.decoded_frames))
        audio_base = audio_clocks[-1][-1] + 1 - join.audio_trim if join.audio_continuous else 1_000_000 * (join.index + 1)
        audio_clocks.append(audio_base + np.arange(join.decoded_samples, dtype=np.float64))
    stitched_video = np.concatenate([clock[j.video_trim :] for j, clock in zip(timeline.joins, video_clocks)])
    assert len(stitched_video) == timeline.total_frames
    audio = assemble_audio(timeline.joins, [c[None].astype(np.float32) for c in audio_clocks], timeline.total_samples)
    assert audio.shape == (1, timeline.total_samples)
    for join in timeline.joins[1:]:
        if join.video_pin:
            assert stitched_video[join.video_start] - stitched_video[join.video_start - 1] == 1
        if join.audio_continuous:
            assert audio[0, join.audio_start] - audio[0, join.audio_start - 1] == 1


def test_the_declick_fade_touches_only_seams_the_sound_cannot_run_through():
    timeline = plan(["fresh", "continue", "cut", "fresh", "cut"], [2, 2, 2, 2, 2], anchor="first")
    assert [j.audio_continuous for j in timeline.joins] == [False, True, False, False, True]
    audios = [np.ones((2, j.decoded_samples), dtype=np.float32) for j in timeline.joins]
    plain = assemble_audio(timeline.joins, audios, timeline.total_samples)
    faded = assemble_audio(timeline.joins, audios, timeline.total_samples, fade_samples=240)
    changed = np.flatnonzero((plain != faded).any(axis=0))
    for join in timeline.joins[1:]:
        s = join.audio_start
        if join.audio_continuous:
            assert np.array_equal(plain[:, s - 300 : s + 300], faded[:, s - 300 : s + 300])
        else:
            assert 0 < faded[0, s] < 0.01 and faded[0, s + 239] > 0.99  # ramped in
            before = timeline.joins[join.index - 1]
            end = before.audio_start + before.decoded_samples - before.audio_trim  # the last real sample before the pad
            assert faded[0, end - 1] < 0.01 and faded[0, end - 241] == 1.0  # ramped out
    assert len(changed) == 2 * 240 * 2  # two seams, both sides


def test_a_decoder_that_breaks_the_causal_rule_rescales_the_trim():
    timeline = Timeline(G24, 3, "previous")
    for index, join_mode in enumerate(["fresh", "continue"]):
        planned = timeline.plan(join_mode, 49)
        timeline.rendered(index, 49, 96_000)  # 480 fewer samples than the rule predicts
    join = timeline.joins[1]
    assert join.notes and "96000" in join.notes[0]
    assert join.audio_trim == round(96_000 * G24.mel_frames(planned.audio_pin) / G24.mel_frames(51))
    assert join.audio_start == timeline.joins[0].audio_start + timeline.joins[0].audio_keep


# ---------------------------------------------------------------- refusals


def test_the_first_shot_must_be_fresh_and_joins_are_named():
    with pytest.raises(StoryboardError, match="must be fresh"):
        Timeline(G24, 3).plan("continue", 49)
    timeline = Timeline(G24, 3)
    timeline.plan("fresh", 49)
    with pytest.raises(StoryboardError, match="dissolve"):
        timeline.plan("dissolve", 49)


def test_an_overlap_the_shot_cannot_hold_is_refused():
    with pytest.raises(StoryboardError, match="overlap of 7"):
        plan(["fresh", "continue"], [2, 2], overlap=7)  # 2 s is 7 latent frames: at most 6 can be pinned


def test_a_shot_is_planned_only_after_the_one_before_it_rendered():
    timeline = Timeline(G24, 3)
    timeline.plan("fresh", 49)
    with pytest.raises(RuntimeError, match="render shot 1 first"):
        timeline.plan("continue", 49)
