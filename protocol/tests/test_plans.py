"""Plans (Director): parsing and repairing what the planner writes, fitting durations, validation, brief quotes, the
sealed output, and the profile, receipt and registration additions. The planner's replies are the raw outputs of the
2026-09-16 GPU spike (data/plan_spike_2026-09-16.json) and the defects the CPU experiment observed."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from kuno_protocol import plans
from kuno_protocol.blobs import encrypt_blob
from kuno_protocol.canonical import canonical_json, sha256_hex
from kuno_protocol.attestation import MockTEE, build_evidence
from kuno_protocol.canonical import b64d
from kuno_protocol.crypto import DecryptionError, generate_hpke_keypair, generate_signing_key, public_key_bytes
from kuno_protocol.envelope import fits
from kuno_protocol.plans import Plan, PlanContext, PlannedShot, PlanOptions, PlanRevision
from kuno_protocol.profiles import InputRole, Mode, ParamError, load_profiles, shot_prompt, storyboard_duration_s, validate_params
from kuno_protocol.receipts import PlanInfo, ReceiptBody, VideoInfo, receipt_message, sign_receipt, verify_receipt
from kuno_protocol.schemas import GenerationParams, MinerRegistration, ShotSpec
from kuno_protocol.sealed_payload import MIN_PADDED, pad_payload

HERE = Path(__file__).parent
SPIKE = {(r["planner"], r["case"]): r for r in json.loads((HERE / "data" / "plan_spike_2026-09-16.json").read_text())["replies"]}
PROFILES = load_profiles()
FAST = PROFILES["ltx-2.5-fast"]
PLANNER = "ltx-2.5-distilled/bf16/1:prompt_enhancer"


def params(target: float = 30, resolution: str = "720p", aspect_ratio: str = "16:9", fps: int = 24, audio: bool = True) -> GenerationParams:
    return GenerationParams(profile_id=FAST.id, mode=Mode.PLAN, duration_s=target, resolution=resolution, aspect_ratio=aspect_ratio, fps=fps, audio=audio)


def context(target: float = 30, max_shot_s: float | None = 11, **kwargs) -> PlanContext:
    options = PlanOptions(max_shot_s=max_shot_s, **{k: v for k, v in kwargs.items() if k in ("min_shots", "max_shots", "style")})
    frame = {k: v for k, v in kwargs.items() if k in ("resolution", "aspect_ratio", "fps", "audio")}
    return plans.plan_context(FAST, params(target, **frame), options)


def spike(case: str, planner: str = "e2b") -> tuple[dict, PlanContext]:
    record = SPIKE[(planner, case)]
    return record, context(record["target_s"], aspect_ratio=record["aspect_ratio"], resolution=record["resolution"])


def reply(shots: list[dict], **fields) -> str:
    return json.dumps({"title": "A title", "scene": "A quiet harbor at dawn.", "shots": shots, "notes": "", **fields})


def shot(prompt: str = "Wide shot; a boat leaves the dock.", duration_s=5, join="cut", beat="A beat") -> dict:
    return {"beat": beat, "prompt": prompt, "duration_s": duration_s, "join": join}


def plan_shots(*spec: tuple[float, str]) -> list[PlannedShot]:
    return [PlannedShot(beat=f"Beat {i}", prompt=f"Medium shot; moment {i}.", duration_s=d, join=j) for i, (d, j) in enumerate(spec)]


# ------------------------------------------------------------------ parsing: the defects the planner showed


def test_an_object_closed_early_and_continued_with_a_comma_is_reopened():
    record, ctx = spike("roastery")
    with pytest.raises(json.JSONDecodeError, match="Extra data"):
        json.loads(record["raw"])
    data, fixes = plans.parse_object(record["raw"])
    assert fixes == ["reopened an object closed early"]
    assert len(data["shots"]) == 5 and data["notes"].startswith("I focused on tactile, warm visuals")


def test_an_object_closed_early_and_continued_without_a_comma_is_reopened():
    record, _ = spike("lighthouse")
    assert ']} "notes"' in record["raw"]
    data, fixes = plans.parse_object(record["raw"])
    assert fixes == ["reopened an object closed early"] and len(data["shots"]) == 8 and "melancholic" in data["notes"]


def test_fences_thinking_trailing_commas_and_text_after_the_object_are_removed():
    raw = '<think>plan {it}</think>Here you go:\n```json\n{"title": "T", "shots": [{"prompt": "a, ]"},],}\n```\nHope it helps! {"x": 1}'
    data, fixes = plans.parse_object(raw)
    assert data == {"title": "T", "shots": [{"prompt": "a, ]"}]}
    assert fixes == ["removed trailing commas", "dropped text after the object"]


@pytest.mark.parametrize(("raw", "message"), [
    ("SZGGASSACZGDGORKVBSFOSBUEKUMNIOYSIlKSGVVRLLYSUSVUYPDEGKBCKYD", "no JSON object"),  # the 12B text encoder, on the GPU
    # A reply cut off at max_new_tokens: everything up to its last brace is still an unclosed object.
    ('{"title": "T", "shots": [{"prompt": "a"}, {"prompt": "the camera', "stopped before its JSON object was complete"),
    ('{"title": "T", "shots": [{"prompt": "a"}, {"prompt": "b"}', "stopped before its JSON object was complete"),
    ('{"title": NaN, "shots": []}', "not one valid JSON object"),
    ('{"title": "T" "shots": []}', "not one valid JSON object"),
])
def test_what_no_repair_can_parse_is_named_for_the_planner(raw, message):
    with pytest.raises(plans.PlanParseError, match=message):
        plans.parse_object(raw)


# ------------------------------------------------------------------ repair on the GPU spike's replies


@pytest.mark.parametrize("case", ["roastery", "lighthouse", "water", "bakery_de", "ebike"])
def test_every_e2b_reply_from_the_gpu_spike_becomes_a_valid_plan_on_target(case):
    record, ctx = spike(case)
    result = plans.repair(record["raw"], ctx, planner=PLANNER, brief=record["brief"])
    plan = result.plan
    assert plan is not None and not result.refusal
    plans.validate(plan, FAST, context=ctx)
    assert abs(plan.duration_s - record["target_s"]) <= plans.TARGET_TOLERANCE_S
    assert plan.duration_s == storyboard_duration_s(FAST, plan.shot_specs(), 24)
    assert plan.shots[0].join == "fresh" and all(shot.duration_s <= 11 for shot in plan.shots)
    assert plan.planner.model == PLANNER and plan.planner.prompt_version == "plan/1"
    validate_params(FAST, plan.storyboard_params())


def test_the_lighthouse_reply_loses_its_join_words_and_its_resized_continues():
    record, ctx = spike("lighthouse")
    result = plans.repair(record["raw"], ctx, planner=PLANNER, brief=record["brief"])
    plan = result.plan
    # E2B ended prompts with "Continue." / "Cut." / "This is a fresh moment.", which the video model would read as instructions.
    assert not any(p.prompt.endswith(("Cut.", "Continue.", "This is a fresh moment.")) for p in plan.shots)
    assert plan.shots[-1].prompt.endswith("speaks softly, 'It is time.'")
    # Wide -> medium -> close-up: a continued take can't change shot size.
    assert [s.join for s in plan.shots[:3]] == ["fresh", "cut", "cut"]
    assert plan.repairs == [
        "removed a join word written at the end of the prompt of shots 1, 2, 3, 4, 5, 6 and 7",
        "shots 2 and 3 continued the take before at a different shot size; made cuts",
        "the shots ran 43.375 s; shots 1 and 2 lengthened to reach 45.375 s",
    ]
    assert result.syntax == ["reopened an object closed early"] and result.problems == []


def test_the_water_reply_that_dropped_the_slogan_is_sent_back_and_delivered_with_a_notice():
    record, ctx = spike("water")
    assert plans.brief_quotes(record["brief"]) == ["Sip. Stay fresh."]
    result = plans.repair(record["raw"], ctx, planner=PLANNER, brief=record["brief"])
    [problem] = result.problems
    assert problem.code == "missing_quote" and not problem.fatal and '"Sip. Stay fresh."' in problem.message
    delivered = result.deliverable()
    assert delivered.repairs[-1] == 'the brief\'s quoted words "Sip. Stay fresh." are in no shot'
    assert delivered.aspect_ratio == "9:16"


def test_the_ebike_reply_is_lengthened_to_its_90_second_target_within_the_shot_cap():
    record, ctx = spike("ebike")
    result = plans.repair(record["raw"], ctx, planner=PLANNER, brief=record["brief"])
    assert result.model_duration_s == pytest.approx(82.083, abs=1e-3)
    assert result.plan.duration_s == pytest.approx(90.083, abs=1e-3) and result.plan.notes == ""  # it wrote no notes
    assert any(r.startswith("the shots ran 82.083 s;") for r in result.plan.repairs)


def test_the_12b_text_encoders_reply_is_unparseable():
    record, ctx = spike("roastery", planner="te12b")
    result = plans.repair(record["raw"], ctx, planner="ltx-2.5-distilled/bf16/1:text_encoder")
    assert result.plan is None and [p.code for p in result.problems] == ["unparseable"] and result.problems[0].fatal


# ------------------------------------------------------------------ repair rules


def test_a_refusal_is_a_refusal_and_nothing_else():
    result = plans.repair('{"refusal": "cannot plan this brief"}', context(), planner=PLANNER)
    assert result.refusal and result.plan is None and result.problems == []
    # A refusal key beside real shots is not a refusal.
    raw = reply([shot(join="fresh"), shot()], refusal="no")
    assert not plans.repair(raw, context(), planner=PLANNER).refusal


def test_shots_are_cleaned_labelled_and_joined_sensibly():
    raw = json.dumps({
        "title": "**Harbor** at dawn",
        "scene": "  A quiet harbor   at dawn.\n",
        "shots": [
            "not a shot",
            {"beat": "", "prompt": "Shot 1: Prompt: “Wide shot”; a boat leaves the dock — slowly.", "duration_s": "6 s", "join": "cut"},
            {"beat": "Beat: Gulls", "prompt": "   ", "duration_s": 5, "join": "cut"},
            {"beat": "Gulls", "prompt": "Close-up shot; a gull lands on a post.", "duration_s": 5.4, "join": "dissolve"},
            {"beat": "Rope", "prompt": "Close-up shot; a hand coils a rope. Cut.", "duration_s": None, "join": "continue"},
        ],
    })
    result = plans.repair(raw, context(target=16), planner=PLANNER)
    plan = result.plan
    assert plan.title == "Harbor at dawn" and plan.scene == "A quiet harbor at dawn." and plan.notes == ""
    first, second, third = plan.shots
    assert first.prompt == '"Wide shot"; a boat leaves the dock - slowly.' and first.beat == '"Wide shot"; a boat leaves the'
    assert (first.join, second.join, third.join) == ("fresh", "cut", "continue")  # both close-ups: the take may continue
    assert third.prompt == "Close-up shot; a hand coils a rope."
    assert plan.repairs[:5] == [
        "shot 1 had no beat; named from the prompt",
        "removed a join word written at the end of the prompt of shot 3",
        "shot 1's join became fresh: nothing comes before it",
        "shot 2 had no valid join; made a cut",
        "shot 3 had no length; given 6 s",
    ]


def test_fewer_than_two_usable_shots_is_fatal_and_more_than_the_maximum_are_cut():
    one = plans.repair(reply([shot(join="fresh"), {"prompt": ""}]), context(), planner=PLANNER)
    assert one.plan is None and one.problems[0].code == "too_few_shots" and one.problems[0].fatal
    many = plans.repair(reply([shot(duration_s=2) for _ in range(15)]), context(target=30, max_shots=10), planner=PLANNER)
    assert len(many.plan.shots) == 10 and many.plan.repairs[0] == "kept the first 10 of the 15 shots"


def test_a_reply_far_short_of_the_target_is_a_problem_even_when_the_fit_reaches_it():
    result = plans.repair(reply([shot(duration_s=4, join="fresh"), shot(duration_s=4), shot(duration_s=4)]), context(target=30), planner=PLANNER)
    assert result.model_duration_s < 30 * 0.75 and [p.code for p in result.problems] == ["short"]
    assert result.problems[0].message == "Your plan runs 10.708 s; the target is 30 s. Add shots or make them longer."
    assert abs(result.plan.duration_s - 30) <= 0.5 and result.deliverable().repairs == result.plan.repairs  # the fit notes it


def test_text_is_cut_to_its_limits_at_a_sentence_end():
    sentence = "The keeper climbs the stair and lights the lamp while waves break on the rocks below. "
    long_prompt = "Medium shot; " + sentence * 60
    scene = "A lighthouse at dusk. " * 60
    raw = reply([shot(long_prompt, join="fresh"), shot()], title="A very long title " * 8, scene=scene, notes="Notes. " * 80)
    plan = plans.repair(raw, context(), planner=PLANNER).plan
    assert len(plan.title) <= 80 and len(plan.scene) <= 1000 and plan.scene.endswith("dusk.") and len(plan.notes) <= 400
    assert len(shot_prompt(plan.scene, plan.shots[0].prompt)) <= FAST.limits.max_prompt_chars and plan.shots[0].prompt.endswith("below.")
    assert any(r.startswith("shot 1's prompt was cut to ") for r in plan.repairs)


def test_shot_sizes_are_read_longest_words_first():
    assert plans.shot_size("Medium close-up shot; a hand.") == "medium close-up"
    assert plans.shot_size("Extreme close up of an eye, then a wide shot") == "extreme close-up"
    assert plans.shot_size("A bird's-eye view of the square") == "overhead"
    assert plans.shot_size("The mediumship reading") is None and plans.shot_size("A slow push-in") is None


# ------------------------------------------------------------------ fit


def test_the_fit_lengthens_the_shortest_shots_first_until_the_target():
    ctx = context(target=30)
    shots, repairs = plans.fit(plan_shots((4, "fresh"), (6, "cut"), (4, "cut")), ctx)
    # 4 + 6 + 4 s less two 17-frame trims is 12.708 s; the shortest shot gains a second at a time, the first on ties.
    assert [s.duration_s for s in shots] == [11, 10, 10]
    stitched = storyboard_duration_s(FAST, [ShotSpec(duration_s=s.duration_s, join=s.join) for s in shots], 24)
    assert abs(stitched - 30) <= 0.5
    assert repairs == [f"the shots ran 12.708 s; shots 1, 2 and 3 lengthened to reach {plans._seconds(stitched)} s"]


def test_the_fit_snaps_to_the_grid_and_caps_and_says_why():
    ctx = context(target=10, max_shot_s=6)
    shots, repairs = plans.fit(plan_shots((30, "fresh"), (1, "cut"), (4.6, "cut")), ctx)
    assert repairs[:3] == [
        "shot 1 shortened from 30 s to 6 s, the longest shot this plan allows",
        "shot 2 lengthened from 1 s to 2 s, the shortest shot LTX-2.5 Fast renders",
        "shot 3's length rounded from 4.6 s to 5 s",
    ]
    assert all(2 <= s.duration_s <= 6 for s in shots)


def test_a_target_the_caps_cannot_reach_is_named():
    shots, repairs = plans.fit(plan_shots((5, "fresh"), (5, "cut")), context(target=60, max_shot_s=11))
    assert [s.duration_s for s in shots] == [11, 11]
    assert repairs[-1] == "the plan runs 21.375 s of its 60 s target: the shots that may change are at the longest this plan allows, 11 s"


def test_the_fit_keeps_a_storyboard_within_its_longest_total():
    ctx = context(target=120, max_shot_s=20)
    shots, _ = plans.fit(plan_shots(*[(20, "fresh")] + [(20, "cut")] * 11), ctx)
    stitched = storyboard_duration_s(FAST, [ShotSpec(duration_s=s.duration_s, join=s.join) for s in shots], 24)
    assert 119.5 <= stitched <= 120


@pytest.mark.parametrize("fps", [24, 25, 48, 50])
def test_the_fit_ends_at_every_frame_rate(fps):
    ctx = context(target=37, max_shot_s=None, fps=fps)
    shots, _ = plans.fit(plan_shots((2, "fresh"), (3, "cut"), (9, "continue"), (2, "cut")), ctx)
    stitched = storyboard_duration_s(FAST, [ShotSpec(duration_s=s.duration_s, join=s.join) for s in shots], fps)
    assert abs(stitched - 37) <= 1.0


def test_on_a_revision_only_the_rewritten_shots_move():
    ctx = context(target=30)
    before = plan_shots((5, "fresh"), (5, "cut"), (5, "cut"))
    shots, _ = plans.fit(before, ctx, movable=[1])
    assert shots[0] == before[0] and shots[2] == before[2] and shots[1].duration_s == 11


# ------------------------------------------------------------------ revisions


def base_plan(ctx: PlanContext) -> Plan:
    raw = reply([shot("Wide shot; a boat leaves the dock.", 7, "fresh"), shot("Close-up shot; a gull lands.", 7), shot("Medium shot; the harbor empties.", 7)])
    return plans.repair(raw, ctx, planner=PLANNER).plan


def test_a_revision_of_listed_shots_keeps_everything_else_byte_identical():
    ctx = context(target=20)
    earlier = base_plan(ctx)
    revise = PlanRevision(plan=earlier, instruction="make shot 2 darker", shots=[2])
    plans.check_revision(revise, ctx)
    rewritten = reply([shot("Changed.", 3, "fresh"), shot("Close-up shot; a gull lands at dusk, in shadow.", 9, "continue"), shot("Changed.")],
                      title="Changed", scene="Changed", notes="Changed")
    plan = plans.repair(rewritten, ctx, planner=PLANNER, revise=revise).plan
    assert (plan.title, plan.scene, plan.notes) == (earlier.title, earlier.scene, earlier.notes)
    assert plan.shots[0] == earlier.shots[0] and plan.shots[2] == earlier.shots[2]
    assert plan.shots[1].prompt == "Close-up shot; a gull lands at dusk, in shadow." and plan.shots[1].join == "cut"

    missing = plans.repair(reply([shot(join="fresh")]), ctx, planner=PLANNER, revise=revise)
    assert missing.plan is None and missing.problems[0].code == "unrevised"
    messages = plans.plan_messages("A harbor film", ctx, PlanOptions(revise=revise))
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert json.loads(messages[2]["content"])["shots"][0]["duration_s"] == earlier.shots[0].duration_s
    assert messages[3]["content"].startswith("Rewrite shot 2 of this plan: make shot 2 darker. Keep the title")


def test_a_revision_must_match_the_job():
    ctx = context(target=20)
    earlier = base_plan(ctx)
    with pytest.raises(plans.PlanError, match="different profile, size"):
        plans.check_revision(PlanRevision(plan=earlier), context(target=20, aspect_ratio="9:16"))
    with pytest.raises(plans.PlanError, match="has 3 shots"):
        plans.check_revision(PlanRevision(plan=earlier, shots=[4]), ctx)
    with pytest.raises(plans.PlanError, match="stays as it is, is longer"):
        plans.check_revision(PlanRevision(plan=earlier, shots=[1]), context(target=20, max_shot_s=5))
    with pytest.raises(ValidationError):
        PlanRevision(plan=earlier, shots=[0])


# ------------------------------------------------------------------ validation


def valid_plan(ctx: PlanContext | None = None) -> Plan:
    return base_plan(ctx or context(target=20))


@pytest.mark.parametrize(("change", "message"), [
    (lambda p: p.model_copy(update={"shots": p.shots[:1]}), "between 2 and 12 shots"),
    (lambda p: p.model_copy(update={"shots": [p.shots[0].model_copy(update={"join": "cut"}), *p.shots[1:]]}), "first shot must be fresh"),
    (lambda p: p.model_copy(update={"duration_s": p.duration_s + 1}), "stitched length"),
    (lambda p: p.model_copy(update={"title": "x" * 81}), "title is longer than 80"),
    (lambda p: p.model_copy(update={"shots": [p.shots[0].model_copy(update={"beat": "x" * 61}), *p.shots[1:]]}), "beat is longer than 60"),
    (lambda p: p.model_copy(update={"shots": [p.shots[0].model_copy(update={"prompt": "x" * 4000}), *p.shots[1:]]}), "with the scene, is longer"),
    (lambda p: p.model_copy(update={"shots": [p.shots[0].model_copy(update={"duration_s": 6.5}), *p.shots[1:]]}), "1-second steps"),
    (lambda p: p.model_copy(update={"profile_id": "ltx-2.5-pro"}), "not ltx-2.5-fast"),
])
def test_validate_names_the_first_rule_a_plan_breaks(change, message):
    plan = valid_plan()
    plans.validate(plan, FAST)
    with pytest.raises(plans.PlanError, match=message):
        plans.validate(change(plan), FAST)


def test_validate_with_a_context_checks_the_frame_target_and_longest_shot():
    ctx = context(target=20)
    plan = valid_plan(ctx)
    plans.validate(plan, FAST, context=ctx)
    with pytest.raises(plans.PlanError, match="frame or target"):
        plans.validate(plan, FAST, context=context(target=25))
    with pytest.raises(plans.PlanError, match="longer than 6 s"):
        plans.validate(plan, FAST, context=context(target=20, max_shot_s=6))


def test_the_context_takes_the_longest_shot_from_the_options_else_the_envelope_capped_by_the_profile():
    assert context(max_shot_s=11.7).max_shot_s == 11
    assert plans.plan_context(FAST, params(), PlanOptions(), served_max_s=18).max_shot_s == 18
    assert plans.plan_context(FAST, params(), PlanOptions(max_shot_s=9), served_max_s=18).max_shot_s == 9
    assert plans.plan_context(FAST, params(fps=50), PlanOptions(max_shot_s=40)).max_shot_s == 10  # 50 fps caps at 10 s
    assert plans.plan_context(FAST, params(), PlanOptions(max_shots=40)).max_shots == 12
    with pytest.raises(plans.PlanError, match="shortest"):
        plans.plan_context(FAST, params(), PlanOptions(max_shot_s=1.5))
    with pytest.raises(plans.PlanError, match="min_shots"):
        plans.plan_context(FAST, params(), PlanOptions(min_shots=5, max_shots=4))
    with pytest.raises(plans.PlanError, match="does not make storyboards"):
        plans.plan_context(PROFILES["ltx-2.5-pro"], params(), PlanOptions())


def test_the_options_refuse_unknown_fields_and_publish_a_schema():
    with pytest.raises(ValidationError):
        PlanOptions.model_validate({"style": "warm", "temperature": 2})
    with pytest.raises(ValidationError):
        PlanOptions.model_validate({"min_shots": 1})
    schema = plans.plan_options_schema()
    assert set(schema["properties"]) == {"v", "style", "max_shot_s", "min_shots", "max_shots", "revise"}
    assert plans.model_output_schema(context())["properties"]["shots"]["items"]["properties"]["duration_s"]["maximum"] == 11


# ------------------------------------------------------------------ brief quotes


@pytest.mark.parametrize(("brief", "quotes"), [
    ("end on the slogan 'Sip. Stay fresh.'", ["Sip. Stay fresh."]),
    ("an old lighthouse keeper's last night; the keeper's line", []),
    ('He says "Guten Morgen" and she answers “Hallo, du!”', ["Guten Morgen", "Hallo, du!"]),
    ("Die Bäckerin sagt „Frisch jeden Morgen.“", ["Frisch jeden Morgen."]),
    ("Le slogan «Toujours frais», puis »Noch einmal«", ["Toujours frais", "Noch einmal"]),
    ("the ‘Rock’n’roll’ band plays ‘Encore’", ["Encore"]),
    ('"Sip." and "sip!" twice, and a lone "x"', ["Sip."]),
])
def test_brief_quotes_find_quoted_phrases_but_not_apostrophes(brief, quotes):
    assert plans.brief_quotes(brief) == quotes


def test_a_quote_matches_whatever_its_case_quotes_and_end_punctuation():
    brief = "end on the slogan 'Sip. Stay fresh.'"
    assert plans.missing_quotes(brief, ['A voice says, “SIP.  Stay fresh!”']) == []
    assert plans.missing_quotes(brief, ["A voice says, Sip, stay fresh."]) == ["Sip. Stay fresh."]


# ------------------------------------------------------------------ the prompt


def test_the_system_prompt_fills_every_placeholder_and_keeps_its_json_example():
    text = plans.system_prompt(context(target=90, style="35mm film, warm."))
    assert "{" not in text.replace('{"title": "...", "scene": "...", "shots": [{"beat": "...", "prompt": "...", "duration_s": 6, "join": "fresh"}], "notes": "..."}', "").replace('{"refusal": "cannot plan this brief"}', "")
    assert "about 90 seconds. Use about 12 shots of about 8 seconds each." in text  # 15 suggested, capped at 12
    assert "whole number from 2 to 11. Use between 2 and 12 shots." in text
    assert "Frame: 16:9 at 720p. Sound: on. Visual style: 35mm film, warm." in text
    assert plans.suggested_shots(context(target=30)) == (5, 7)
    messages = plans.plan_messages("  A 30-second ad\nfor a roastery.  ", context())
    assert messages[1] == {"role": "user", "content": "Brief: A 30-second ad for a roastery."}


def test_the_retry_turn_carries_the_reply_and_its_problems():
    problems = [plans.Problem("short", "Your plan runs 49 s; the target is 90 s. Add shots or make them longer.", "")]
    chat = plans.retry_messages([{"role": "system", "content": "s"}], "{bad", problems)
    assert chat[-2] == {"role": "assistant", "content": "{bad"}
    assert chat[-1]["content"] == ("Your plan runs 49 s; the target is 90 s. Add shots or make them longer. "
                                   "Output the whole corrected plan as one JSON object, with nothing before or after it.")


def test_choose_prefers_a_plan_then_fewer_problems_then_the_retry():
    plan = valid_plan()
    good, soft, none = plans.Repaired(plan=plan), plans.Repaired(plan=plan, problems=[plans.Problem("short", "", "")]), plans.Repaired()
    assert plans.choose(soft, none) is soft and plans.choose(soft, good) is good and plans.choose(good, soft) is good
    assert plans.choose(none, plans.Repaired()) is not none


# ------------------------------------------------------------------ output


def test_a_plan_is_sealed_as_padded_canonical_json_under_its_own_label():
    plan = valid_plan()
    key, job_id = os.urandom(32), "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
    data, blob = plans.seal_plan(key, job_id, plan)
    assert data == canonical_json(plan.model_dump(mode="json")) and b'"duration_s":7,' in data
    opened, raw = plans.open_plan(key, job_id, blob)
    assert opened == plan and raw == data and sha256_hex(raw) == sha256_hex(plans.encode_plan(plan))
    assert plans.plan_output_label(job_id) == f"{job_id}/output/plan" and len(plans.pad_plan(data)) == MIN_PADDED
    with pytest.raises(DecryptionError):
        plans.open_plan(key, "00000000-0000-4000-8000-000000000000", blob)
    with pytest.raises(plans.PlanError, match="not padded"):
        plans.open_plan(key, job_id, encrypt_blob(key, plans.plan_output_label(job_id), data))
    with pytest.raises(plans.PlanError, match="not a Plan v1"):
        plans.open_plan(key, job_id, encrypt_blob(key, plans.plan_output_label(job_id), pad_payload(b'{"v": 1}')))


def test_seconds_are_written_the_same_in_every_language():
    assert [plans._seconds(v) for v in (30, 29.708333333333332, 10.0625, 0.0005, 45.375, 82.08333333333333)] == [
        "30", "29.708", "10.063", "0.001", "45.375", "82.083",
    ]


# ------------------------------------------------------------------ profiles, receipts, registration


def test_the_fast_profile_offers_plans_with_flat_prices_and_vcu():
    assert Mode.PLAN in FAST.modes and FAST.limits.plan.prompt_version == plans.PROMPT_VERSION
    for fps in (24, 50):
        job = params(target=90, resolution="1080p", fps=fps)
        validate_params(FAST, job)
        # Flat: not per second, no fps multiplier, no minimum charge.
        assert FAST.price_usd(job) == 0.10 and FAST.price_usd(job, "standard") == 0.08
        assert FAST.vcu_for(job) == 27 and FAST.vcu_for(job, seconds=3) == 27
        assert job.render_duration_s == 0
    assert FAST.price_usd(params(target=4)) == 0.10  # below min_job_usd would not be raised either way
    no_price = FAST.model_copy(update={"pricing": FAST.pricing.model_copy(update={"standard_plan_usd": None})})
    with pytest.raises(ParamError, match="no standard price for plans"):
        no_price.price_usd(params(), "standard")


@pytest.mark.parametrize(("change", "message"), [
    ({"duration_s": 3}, "between 4 and 120"),
    ({"duration_s": 121}, "between 4 and 120"),
    ({"input_roles": [InputRole.FIRST_FRAME]}, "does not accept"),
    ({"shots": [ShotSpec(duration_s=5, join="fresh"), ShotSpec(duration_s=5, join="cut")]}, "only for storyboard"),
    ({"resolution": "4k"}, "does not support 4k"),
    ({"profile_id": "ltx-2.5-pro"}, "does not match"),
])
def test_plan_params_are_validated(change, message):
    with pytest.raises(ParamError, match=message):
        validate_params(FAST, params().model_copy(update=change))
    with pytest.raises(ParamError, match="does not support plan"):
        validate_params(PROFILES["ltx-2.5-pro"], params().model_copy(update={"profile_id": "ltx-2.5-pro"}))


def test_a_plan_fits_any_envelope_that_serves_its_size_and_frame_rate():
    assert fits({"720p": {"16:9": {"24": 2}}}, params(target=120))
    assert not fits({"720p": {"16:9": {"25": 20}}}, params(target=10))


def receipt_body(**update) -> ReceiptBody:
    fields = dict(
        job_id="3f2504e0-4f89-41d3-9a0c-0305e82c3301", enclave_id="0" * 32, profile_id=FAST.id, image_digest="sha256:example",
        params_digest="0" * 64, input_digest="1" * 64, output_digest="2" * 64, output_bytes=4352, content_digest="3" * 64,
        attestation_digest="4" * 64, started_at=1_800_000_000.0, finished_at=1_800_000_012.5, gpu_seconds=12.5,
    )
    return ReceiptBody(**(fields | update))


def test_video_receipts_sign_the_same_bytes_as_before_plans():
    video = VideoInfo(duration_s=5.166, width=1344, height=768, fps=24, frames=124, audio=True)
    body = receipt_body(video=video)
    dumped = body.model_dump(mode="json")
    assert "plan" not in dumped and "step_commitment" not in dumped and dumped["video"]["frames"] == 124
    vector = json.loads((HERE / "vectors.json").read_text())["receipt"]
    assert receipt_message(ReceiptBody.model_validate(vector["body"])) == b64d(vector["message_b64"])


def test_a_plan_receipt_has_a_plan_and_no_video_and_verifies():
    info = PlanInfo(shots=5, duration_s=30.375, planner=PLANNER, prompt_version="plan/1", output_tokens=451)
    body = receipt_body(plan=info)
    dumped = body.model_dump(mode="json")
    assert "video" not in dumped and dumped["plan"] == info.model_dump(mode="json")
    key = generate_signing_key()
    receipt = sign_receipt(key, body)
    assert verify_receipt(receipt, public_key_bytes(key))
    assert ReceiptBody.model_validate_json(body.model_dump_json()) == body
    with pytest.raises(ValidationError, match="exactly one output"):
        receipt_body()
    with pytest.raises(ValidationError, match="exactly one output"):
        receipt_body(plan=info, video=VideoInfo(duration_s=1, width=1, height=1, fps=24, frames=24, audio=False))


def test_registration_features_are_left_out_when_absent():
    _, hpke = generate_hpke_keypair()
    signing = generate_signing_key()
    evidence = build_evidence(MockTEE(signing, "sha256:img"), os.urandom(32), hpke, public_key_bytes(signing), "sha256:img", [FAST.id])
    without = MinerRegistration(evidence=evidence)
    assert "features" not in without.model_dump(mode="json") and "features" not in json.loads(without.model_dump_json())
    body = MinerRegistration(evidence=evidence, features=[plans.PLAN_FEATURE]).model_dump(mode="json")
    assert body["features"] == ["plan/1"] and MinerRegistration.model_validate(body).features == ["plan/1"]
