"""Storyboard jobs in the protocol: the stitched length every party computes, what `validate_params` accepts, how the
shot list is serialized (and left out of every other job's bytes), and what a storyboard pays and costs."""

from __future__ import annotations

import pytest

from kuno_protocol.envelope import EnvelopeQuery, fits
from kuno_protocol.profiles import (
    Mode,
    ParamError,
    load_profiles,
    shot_prompt,
    storyboard_duration_s,
    storyboard_frames,
    storyboard_trim_frames,
    validate_params,
)
from kuno_protocol.schemas import GenerationParams, SealedPayload, ShotPrompt, ShotSpec, job_aad

PROFILES = load_profiles()
FAST = PROFILES["ltx-2.5-fast"]


def shots(*spec: tuple[float, str]) -> list[ShotSpec]:
    return [ShotSpec(duration_s=duration, join=join) for duration, join in spec]


def board(shot_list: list[ShotSpec], fps: int = 24, **change) -> GenerationParams:
    base = dict(profile_id=FAST.id, mode=Mode.STORYBOARD, duration_s=storyboard_duration_s(FAST, shot_list, fps),
                resolution="720p", aspect_ratio="16:9", fps=fps, shots=shot_list)
    return GenerationParams(**{**base, **change})


def test_the_stitched_length_matches_the_gpu_run():
    # research/long-video_ltx-av-extend_2026-09-16.md: harbor-long and mixed-joins, stitched on an RTX PRO 6000.
    assert storyboard_trim_frames(FAST) == 17
    long_take = shots((5, "fresh"), *[(5, "continue")] * 7)
    assert (storyboard_frames(FAST, long_take, 24), storyboard_duration_s(FAST, long_take, 24)) == (849, 35.375)
    mixed = shots((3, "fresh"), (3, "continue"), (3, "cut"), (3, "fresh"))
    assert (storyboard_frames(FAST, mixed, 24), storyboard_duration_s(FAST, mixed, 24)) == (258, 10.75)


def test_a_valid_storyboard_passes_and_prices_by_its_stitched_seconds():
    params = board(shots((5, "fresh"), (5, "continue"), (5, "cut")))
    validate_params(FAST, params)
    assert params.duration_s == pytest.approx(13.708, abs=1e-3) and params.render_duration_s == 5
    assert FAST.price_usd(params) == round(0.05 * params.duration_s, 4)
    assert FAST.price_usd(params, "standard") == round(0.04 * params.duration_s, 4)


def test_miners_are_paid_for_every_rendered_shot_overlaps_included():
    params = board(shots((5, "fresh"), (10, "continue"), (5, "cut")))
    rendered = FAST.vcu_at("720p", 24, 5) * 2 + FAST.vcu_at("720p", 24, 10)
    assert FAST.vcu_for(params) == pytest.approx(rendered)
    # Every rendered second is paid (20 s here, against 18.7 s delivered), but each shot at its own, shorter duration
    # factor: a storyboard costs less per second to render than one clip of its whole length would.
    assert FAST.vcu_weights.per_output_second["720p"] * 20 < FAST.vcu_for(params) < FAST.vcu_at("720p", 24, 20)
    assert FAST.vcu_for(params, seconds=params.duration_s / 2) == pytest.approx(rendered / 2)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (dict(shots=shots((5, "fresh"))), "between 2 and 12 shots"),
        (dict(shots=shots(*[(2, "fresh")] * 13)), "between 2 and 12 shots"),
        (dict(shots=shots((5, "continue"), (5, "continue"))), "first shot must be fresh"),
        (dict(shots=shots((5, "fresh"), (21, "continue"))), "shot 2's duration must be between 2 and 20 seconds"),
        (dict(shots=shots((5, "fresh"), (2.5, "cut"))), "shot 2's duration must be in 1-second steps"),
        (dict(shots=shots((5, "fresh"), (11, "cut")), fps=48), "at 48 fps, shot 2's duration must be at most 10 seconds"),
        (dict(shots=shots(*[(20, "fresh")] * 7)), "at most 120 seconds"),
    ],
)
def test_bad_storyboards_are_refused(change, message):
    shot_list = change.pop("shots")
    fps = change.pop("fps", 24)
    with pytest.raises(ParamError, match=message):
        validate_params(FAST, board(shot_list, fps=fps, **change))


def test_duration_s_must_be_exactly_the_stitched_length():
    params = board(shots((5, "fresh"), (5, "continue")))
    with pytest.raises(ParamError, match="must be its stitched length"):
        validate_params(FAST, params.model_copy(update={"duration_s": 10.0}))


def test_a_joined_shot_must_keep_frames_after_its_overlap():
    # At 48 fps a 2 s shot is 97 frames, well past the 17-frame trim; a profile with a larger overlap would refuse it.
    wide = FAST.model_copy(update={"limits": FAST.limits.model_copy(update={
        "storyboard": FAST.limits.storyboard.model_copy(update={"overlap_latent_frames": 7})})})
    shot_list = shots((2, "fresh"), (2, "continue"))
    params = GenerationParams(profile_id=FAST.id, mode=Mode.STORYBOARD, duration_s=storyboard_duration_s(wide, shot_list, 24),
                              resolution="720p", aspect_ratio="16:9", fps=24, shots=shot_list)
    with pytest.raises(ParamError, match="too short to join"):
        validate_params(wide, params)


def test_shots_are_refused_outside_storyboard_mode_and_storyboards_outside_their_profiles():
    params = GenerationParams(profile_id=FAST.id, mode=Mode.TEXT_TO_VIDEO, duration_s=5, resolution="720p", aspect_ratio="16:9",
                              fps=24, shots=shots((5, "fresh"), (5, "continue")))
    with pytest.raises(ParamError, match="only for storyboard mode"):
        validate_params(FAST, params)
    pro = PROFILES["ltx-2.5-pro"]
    assert Mode.STORYBOARD not in pro.modes and pro.limits.storyboard is None
    with pytest.raises(ParamError, match="does not support storyboard"):
        validate_params(pro, board(shots((5, "fresh"), (5, "continue")), profile_id=pro.id))


def test_other_jobs_serialize_exactly_as_before_storyboards():
    params = GenerationParams(profile_id=FAST.id, mode=Mode.TEXT_TO_VIDEO, duration_s=5, resolution="720p", aspect_ratio="16:9", fps=24)
    assert "shots" not in params.model_dump(mode="json")
    assert job_aad("j", "e", params, []) == (
        b'{"enclave_id":"e","inputs":[],"job_id":"j","params":{"aspect_ratio":"16:9","audio":true,"duration_s":5,"fps":24,'
        b'"input_roles":[],"mode":"text_to_video","profile_id":"ltx-2.5-fast","resolution":"720p"},"v":1}'
    )
    assert "shots" not in SealedPayload(prompt="a fox").model_dump_json()
    storyboard = board(shots((5, "fresh"), (5, "continue")))
    assert storyboard.model_dump(mode="json")["shots"] == [{"duration_s": 5.0, "join": "fresh"}, {"duration_s": 5.0, "join": "continue"}]
    assert GenerationParams.model_validate(storyboard.model_dump(mode="json")) == storyboard


def test_the_sealed_payload_carries_one_non_empty_prompt_per_shot_and_the_scene_goes_first():
    payload = SealedPayload(prompt="A small blue fishing boat.", shots=[ShotPrompt(prompt="It leaves the harbor."), ShotPrompt(prompt="Night falls.")])
    assert SealedPayload.model_validate_json(payload.model_dump_json()) == payload
    for blank in ("", "  \n\t"):
        with pytest.raises(ValueError):
            ShotPrompt(prompt=blank)
    with pytest.raises(ValueError):
        GenerationParams(profile_id=FAST.id, mode=Mode.STORYBOARD, duration_s=1, resolution="720p", aspect_ratio="16:9", fps=24,
                         shots=[ShotSpec(duration_s=2, join="fresh")] * 65)
    assert shot_prompt("A small blue fishing boat. ", " It leaves the harbor.") == "A small blue fishing boat.\n\nIt leaves the harbor."
    assert shot_prompt("", "Night falls.") == "Night falls."


def test_envelopes_and_the_long_clip_rule_look_at_the_longest_shot():
    params = board(shots((8, "fresh"), (8, "continue"), (8, "continue")))
    assert params.duration_s > 20 and params.render_duration_s == 8
    assert fits({"720p": {"16:9": {24: 10}}}, params) and not fits({"720p": {"16:9": {24: 7}}}, params)
    assert EnvelopeQuery.of(params).duration_s == 8
