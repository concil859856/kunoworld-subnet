"""Plans (Director): a storyboard written from a customer's brief inside the enclave (PROTOCOL.md "Plans (Director)").

A plan is a job in mode `plan`. The public params are the storyboard-to-be's frame (profile, resolution, aspect ratio,
fps, audio) and its target length in `duration_s`; the sealed payload carries the brief as `prompt` and this module's
`PlanOptions` as `options.plan`. The worker has the LTX-2.5 pipeline's bundled prompt enhancer (Gemma-4-E2B) write a
JSON shot list under the `plan/1` system prompt, and the code here does everything else:

    parse_object   finds the JSON object in the planner's reply and fixes the syntax E2B gets wrong (an object closed
                   early, trailing commas, text after the object)
    repair         turns the object into a `Plan`: shots cleaned, joins sanity-checked, durations fitted, text cut to
                   its limits, and what still needs the planner (unparseable, too few shots, far too short, a brief quote
                   missing) reported as `Problem`s for the worker's single retry
    fit            snaps durations to the profile's grid and caps, then moves them until the stitched length is within
                   0.5 s of the target; on a revision only the rewritten shots move
    validate       every rule a delivered plan keeps, the same in the worker, the gateway (Standard) and the SDKs
    seal_plan      the output: canonical plan JSON, padded like a sealed request, sealed as `<job_id>/output/plan`

Every step is deterministic, so a client can reproduce a plan from the planner's reply, and the shared vectors
(`protocol/tests/vectors.json`, group `plans`) pin cases the JS SDK must match.

Portability rules, so other implementations match character for character: text lengths count Unicode code points;
whitespace is exactly ECMAScript's `\\s` set (`_SPACE_CHARS`); seconds in `repairs` have at most three decimals, rounded
half up on the double (`_seconds`); shot sizes are matched on lower-cased text with ASCII letter and digit boundaries.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .blobs import decrypt_blob, encrypt_blob
from .canonical import canonical_json
from .envelope import profile_max_duration
from .profiles import Mode, ModelProfile, ParamError, shot_prompt, storyboard_duration_s, storyboard_trim_frames, validate_params
from .schemas import GenerationParams, ShotJoin, ShotSpec
from .sealed_payload import PAYLOAD_V2, MalformedPayload, pad_payload, payload_version, unpad_payload

PLAN_VERSION = 1
# The system prompt a plan is written under (plan_prompts/). A new wording is a new version, never an edit in place:
# receipts name it, and every prompt change needs a regression run on the fixed brief set.
PROMPT_VERSION = "plan/1"
# What a worker lists in `MinerRegistration.features` once it serves plan jobs under this module's contract.
PLAN_FEATURE = "plan/1"
# The key of `PlanOptions` in `SealedPayload.options`.
PLAN_OPTION = "plan"
# The failure code for a reply that no repair or retry turned into a plan: refunded, and not the miner's fault.
PLAN_FAILED = "plan_failed"
# Decoding for plan/1, as the 2026-09-16 GPU spike ran it (research/director-design_2026-09-16.md §10). diffusers' own
# enhancement settings can't write JSON: greedy with no_repeat_ngram_size=5 bans the keys every shot repeats.
PLAN_SAMPLING: dict[str, Any] = {"do_sample": True, "temperature": 0.7, "top_p": 0.95, "top_k": 64}

TITLE_MAX_CHARS = 80
BEAT_MAX_CHARS = 60
NOTES_MAX_CHARS = 400
SCENE_MAX_CHARS = 1000
# The fit stops once the stitched length is this close to the target.
TARGET_TOLERANCE_S = 0.5
# A reply more than this share short of the target (before the fit) is sent back once.
SHORT_FRACTION = 0.25
# A missing beat is named from its prompt's first words.
BEAT_WORDS = 6
QUOTE_MIN_CHARS, QUOTE_MAX_CHARS = 2, 200
# Syntax fixes tried before a reply counts as unparseable.
PARSE_ATTEMPTS = 3
_EPSILON = 1e-6

_PROMPT_FILES = {"plan/1": "plan-1.txt"}


class PlanError(ValueError):
    """A plan, or the options for one, breaks a rule of this module or of the profile."""


class PlanParseError(PlanError):
    """The planner's reply holds no JSON object this module can repair. The message is written to the planner."""


# ------------------------------------------------------------------ models


class PlannedShot(BaseModel):
    """One shot of a plan: a label for the storyboard card, what the model renders after the scene, its length and join.
    `prompt` is the storyboard's `ShotPrompt`; `duration_s` and `join` are its `ShotSpec`."""

    model_config = ConfigDict(extra="forbid")

    beat: str
    prompt: str = Field(min_length=1)
    duration_s: float
    join: ShotJoin


class PlannerInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # "<precision recipe id>:<component>", e.g. "ltx-2.5-distilled/bf16/1:prompt_enhancer".
    model: str
    prompt_version: str


class Plan(BaseModel):
    """Plan v1: what the enclave delivers, and what an SDK loads into a storyboard."""

    model_config = ConfigDict(extra="forbid")

    v: Literal[1] = 1
    profile_id: str
    resolution: str
    aspect_ratio: str
    fps: int
    audio: bool
    # The job's `duration_s`: the length asked for.
    target_s: float
    # The stitched length of `shots`, exactly (`profiles.storyboard_duration_s`): the storyboard's `duration_s`.
    duration_s: float
    title: str
    # What every shot shares; the video model sees `shot_prompt(scene, shot.prompt)`.
    scene: str
    shots: list[PlannedShot] = Field(max_length=64)
    notes: str = ""
    # Every change code made to what the planner wrote, in words, and what the retry couldn't fix. Nothing changes silently.
    repairs: list[str] = Field(default_factory=list)
    planner: PlannerInfo

    def shot_specs(self) -> list[ShotSpec]:
        return [ShotSpec(duration_s=shot.duration_s, join=shot.join) for shot in self.shots]

    def model_prompts(self) -> list[str]:
        """What the video model sees for each shot."""
        return [shot_prompt(self.scene, shot.prompt) for shot in self.shots]

    def storyboard_params(self) -> GenerationParams:
        """The public params of the storyboard this plan renders as."""
        return GenerationParams(
            profile_id=self.profile_id, mode=Mode.STORYBOARD, duration_s=self.duration_s, resolution=self.resolution,
            aspect_ratio=self.aspect_ratio, fps=self.fps, audio=self.audio, shots=self.shot_specs(),
        )


class PlanRevision(BaseModel):
    """Rewrite an earlier plan. With `shots` (numbered from 1), only those shots are rewritten: the title, the scene, the
    notes and every other shot come back byte-identical, and the fit moves only the rewritten shots' durations. Without
    it, the whole plan is rewritten under the instruction."""

    model_config = ConfigDict(extra="forbid")

    plan: Plan
    instruction: str = ""
    shots: list[int] | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("shots")
    @classmethod
    def _numbers(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and any(number < 1 for number in value):
            raise ValueError("shots are numbered from 1")
        return sorted(set(value)) if value is not None else None


class PlanOptions(BaseModel):
    """`SealedPayload.options["plan"]`. Every field is optional; an absent `options.plan` is all defaults."""

    model_config = ConfigDict(extra="forbid")

    v: Literal[1] = 1
    # A look to keep to, e.g. "35mm film, warm". Checked like the brief, at most `limits.plan.max_style_chars`.
    style: str | None = None
    # The longest shot to plan. Clients send the longest duration a routable enclave's envelope serves at this size and
    # frame rate, the rule the storyboard will be routed by. Unset: this worker's own envelope. Always capped by the profile.
    max_shot_s: float | None = Field(default=None, gt=0)
    min_shots: int = Field(default=2, ge=2)
    max_shots: int | None = Field(default=None, ge=2)
    revise: PlanRevision | None = None


def plan_options_schema() -> dict[str, Any]:
    """The JSON Schema of `options.plan`."""
    return PlanOptions.model_json_schema()


# ------------------------------------------------------------------ context


@dataclass(frozen=True)
class PlanContext:
    """What a plan is written and fitted to: the job's frame and target, and the shot limits after the options."""

    profile: ModelProfile
    resolution: str
    aspect_ratio: str
    fps: int
    audio: bool
    target_s: float
    min_shot_s: float
    max_shot_s: float
    min_shots: int
    max_shots: int
    style: str | None = None
    prompt_version: str = PROMPT_VERSION

    @property
    def step_s(self) -> float:
        return self.profile.limits.duration_step_s

    @property
    def max_total_s(self) -> float:
        board = self.profile.limits.storyboard
        assert board is not None  # plan_context refuses a profile without storyboards
        return board.max_total_s

    def matches(self, plan: Plan) -> bool:
        return (plan.profile_id, plan.resolution, plan.aspect_ratio, plan.fps, plan.audio) == (
            self.profile.id, self.resolution, self.aspect_ratio, self.fps, self.audio,
        )


def plan_context(
    profile: ModelProfile, params: GenerationParams, options: PlanOptions | None = None, *, served_max_s: float | None = None
) -> PlanContext:
    """The context for a plan job's params. The longest shot is the profile's limit at this frame rate, lowered to
    `options.max_shot_s` when the client sent one, or else to `served_max_s` (the worker's own serving envelope), then
    snapped down to the profile's duration grid. Raises PlanError for options that leave no plan possible."""
    lim, board = profile.limits, profile.limits.storyboard
    if board is None:
        raise PlanError(f"{profile.name} does not make storyboards")
    options = options or PlanOptions()
    cap = profile_max_duration(profile, params.fps)
    if options.max_shot_s is not None:
        cap = min(cap, options.max_shot_s)
    elif served_max_s is not None:
        cap = min(cap, served_max_s)
    cap = _grid_floor(cap, profile)
    if cap < lim.min_duration_s - _EPSILON:
        raise PlanError(f"the longest shot can't be shorter than {profile.name}'s shortest, {lim.min_duration_s:g} s")
    max_shots = board.max_shots if options.max_shots is None else min(board.max_shots, options.max_shots)
    if options.min_shots > max_shots:
        raise PlanError(f"min_shots is more than the {max_shots} shots a plan may have")
    version = lim.plan.prompt_version if lim.plan is not None else PROMPT_VERSION
    style = _clean_space(options.style) if options.style else ""
    return PlanContext(
        profile=profile, resolution=params.resolution, aspect_ratio=params.aspect_ratio, fps=params.fps, audio=params.audio,
        target_s=params.duration_s, min_shot_s=lim.min_duration_s, max_shot_s=cap, min_shots=options.min_shots,
        max_shots=max_shots, style=style or None, prompt_version=version,
    )


def check_revision(revise: PlanRevision, context: PlanContext) -> None:
    """A revision's earlier plan must be a valid plan with this job's frame, its shot numbers must exist, and the shots
    that stay as they are must fit this job's longest shot. Raises PlanError."""
    plan = revise.plan
    if not context.matches(plan):
        raise PlanError("the plan to revise has a different profile, size, frame rate or sound than this job")
    validate(plan, context.profile)
    if revise.shots is None:
        return
    if revise.shots[-1] > len(plan.shots):
        raise PlanError(f"the plan to revise has {len(plan.shots)} shots")
    for number, shot in enumerate(plan.shots, start=1):
        if number not in revise.shots and shot.duration_s > context.max_shot_s + _EPSILON:
            raise PlanError(f"shot {number}, which stays as it is, is longer than this plan's longest shot, {context.max_shot_s:g} s")


# ------------------------------------------------------------------ what the planner is sent


@lru_cache(maxsize=None)
def prompt_template(version: str = PROMPT_VERSION) -> str:
    name = _PROMPT_FILES.get(version)
    if name is None:
        raise PlanError(f"unknown plan prompt version {version!r}")
    return resources.files(__package__).joinpath("plan_prompts", name).read_text(encoding="utf-8")


def suggested_shots(context: PlanContext) -> tuple[int, float]:
    """The system prompt's hint: about one shot per 6 s, each long enough that the shots stitch to the target (every
    join trims `storyboard_trim_frames` / fps seconds). The hint cut E2B's miss on a 90 s brief from 41 s to 10 s
    (research/director-design_2026-09-16.md §6.2)."""
    count = min(max(math.floor(context.target_s / 6 + 0.5), context.min_shots), context.max_shots)
    trim_s = storyboard_trim_frames(context.profile) / context.fps
    length = math.floor((context.target_s + trim_s * (count - 1)) / count + 0.5)
    return count, snap_duration(length, context)


def system_prompt(context: PlanContext) -> str:
    """The versioned system prompt with its placeholders filled. Only the named placeholders are replaced, so the
    prompt's own JSON example keeps its braces."""
    count, length = suggested_shots(context)
    values = {
        "target_s": _seconds(context.target_s),
        "suggested_shots": str(count),
        "suggested_shot_s": _seconds(length),
        "min_shot_s": _seconds(context.min_shot_s),
        "max_shot_s": _seconds(context.max_shot_s),
        "min_shots": str(context.min_shots),
        "max_shots": str(context.max_shots),
        "aspect_ratio": context.aspect_ratio,
        "resolution": context.resolution,
        "sound": "on" if context.audio else "off",
        "style_line": f" Visual style: {context.style.rstrip('.')}." if context.style else "",
    }
    return re.sub(r"\{([a-z_]+)\}", lambda match: values.get(match.group(1), match.group(0)), prompt_template(context.prompt_version))


def model_json(plan: Plan) -> str:
    """A plan in the shape the planner writes (title, scene, shots, notes): the planner's own turn in a revision."""
    shots = [
        {"beat": shot.beat, "prompt": shot.prompt, "duration_s": _json_number(shot.duration_s), "join": shot.join}
        for shot in plan.shots
    ]
    return json.dumps({"title": plan.title, "scene": plan.scene, "shots": shots, "notes": plan.notes}, ensure_ascii=False)


def plan_messages(brief: str, context: PlanContext, options: PlanOptions | None = None) -> list[dict[str, str]]:
    """The chat the planner answers: the system prompt and the brief; for a revision, then the earlier plan as the
    planner's own turn and what to change."""
    revise = (options or PlanOptions()).revise
    messages = [{"role": "system", "content": system_prompt(context)}, {"role": "user", "content": f"Brief: {_clean_space(brief)}"}]
    if revise is not None:
        messages += [{"role": "assistant", "content": model_json(revise.plan)}, {"role": "user", "content": revision_request(revise)}]
    return messages


def revision_request(revise: PlanRevision) -> str:
    instruction = _clean_space(revise.instruction)
    if instruction and instruction[-1] not in ".!?":
        instruction += "."
    if revise.shots:
        ask = f"Rewrite {_shots_text([number - 1 for number in revise.shots])} of this plan"
        ask = f"{ask}: {instruction}" if instruction else f"{ask} with a different take."
        return f"{ask} Keep the title, the scene, the notes and every other shot exactly as they are. Output the whole plan as one JSON object."
    ask = f"Rewrite this plan: {instruction}" if instruction else "Rewrite this plan with a different take."
    return f"{ask} Output the whole plan as one JSON object."


def retry_messages(messages: list[dict[str, str]], reply: str, problems: Iterable[Problem]) -> list[dict[str, str]]:
    """The single retry: the planner's reply as its own turn, then what is wrong with it."""
    asks = " ".join(problem.message for problem in problems)
    correction = f"{asks} Output the whole corrected plan as one JSON object, with nothing before or after it."
    return [*messages, {"role": "assistant", "content": reply}, {"role": "user", "content": correction}]


def model_output_schema(context: PlanContext) -> dict[str, Any]:
    """The JSON Schema of what the planner writes (research/director-design_2026-09-16.md §5.2). plan/1 decodes freely
    and repairs instead; the schema is here for a runtime that constrains decoding with a grammar."""
    return {
        "type": "object", "additionalProperties": False, "required": ["title", "scene", "shots", "notes"],
        "properties": {
            "title": {"type": "string"}, "scene": {"type": "string"}, "notes": {"type": "string"},
            "shots": {
                "type": "array", "minItems": context.min_shots, "maxItems": context.max_shots,
                "items": {
                    "type": "object", "additionalProperties": False, "required": ["beat", "prompt", "duration_s", "join"],
                    "properties": {
                        "beat": {"type": "string"}, "prompt": {"type": "string"},
                        "duration_s": {"type": "integer", "minimum": _json_number(context.min_shot_s), "maximum": _json_number(context.max_shot_s)},
                        "join": {"enum": ["fresh", "continue", "cut"]},
                    },
                },
            },
        },
    }


# ------------------------------------------------------------------ parsing the reply

_THINK = re.compile(r"<think>.*?</think>", re.S)


def extract_object(raw_text: str) -> str:
    """The reply from its first `{` to its last `}`, after dropping `<think>…</think>` blocks. Code fences and prose
    around the object go with the rest."""
    text = _THINK.sub("", raw_text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise PlanParseError("Your reply had no JSON object.")
    return text[start : end + 1]


def parse_object(raw_text: str) -> tuple[dict[str, Any], list[str]]:
    """The JSON object in a reply, and the syntax fixes it needed (at most PARSE_ATTEMPTS), each tried in turn when
    parsing fails: trailing commas removed; an object closed early reopened (E2B closed the plan after its shots array and
    went on with `, "notes": …` or `"notes": …`, in 3 of 5 CPU replies and 2 of 5 GPU replies); text after a complete
    object dropped. NaN and infinities are not JSON and are refused."""
    text = extract_object(raw_text)
    fixes: list[str] = []
    while True:
        try:
            value = json.loads(text, parse_constant=_no_constants)
        except ValueError:
            fixed, note = _fix_json(text)
            if fixed is None or len(fixes) >= PARSE_ATTEMPTS:
                if _object_end(text) is None:
                    raise PlanParseError("Your reply stopped before its JSON object was complete; write shorter prompts.") from None
                raise PlanParseError("Your reply was not one valid JSON object.") from None
            text = fixed
            fixes.append(note)
            continue
        if not isinstance(value, dict):
            raise PlanParseError("Your reply was not one JSON object.")
        return value, fixes


def _no_constants(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _fix_json(text: str) -> tuple[str | None, str]:
    dropped = _drop_trailing_commas(text)
    if dropped != text:
        return dropped, "removed trailing commas"
    end = _object_end(text)
    if end is not None and end < len(text):
        rest = text[end:]
        after = rest.lstrip(" \t\n\r")
        if after.startswith((",", '"')):
            # `text[end - 1]` is the brace that closed the object too soon; the reply's last brace closes it instead.
            return text[: end - 1] + ("" if after.startswith(",") else ",") + rest, "reopened an object closed early"
        return text[:end], "dropped text after the object"
    return None, ""


def _drop_trailing_commas(text: str) -> str:
    """Removes each comma outside strings that only JSON whitespace separates from a closing `}` or `]`."""
    out: list[str] = []
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "," and text[index + 1 :].lstrip(" \t\n\r").startswith(("}", "]")):
            continue
        out.append(char)
    return "".join(out)


def _object_end(text: str) -> int | None:
    """The index just past the bracket that closes the value `text` starts with, outside strings; None if it never closes."""
    depth = 0
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


# ------------------------------------------------------------------ repair


@dataclass(frozen=True)
class Problem:
    """Something the planner is asked to fix in its single retry. `message` is written to the planner. `notice` is what a
    delivered plan's `repairs` says when the retry didn't fix it ("" says nothing); None when no plan can be delivered."""

    code: str
    message: str
    notice: str | None = None

    @property
    def fatal(self) -> bool:
        return self.notice is None


@dataclass
class Repaired:
    """What `repair` made of a reply. `plan` is None when nothing usable came back; `refusal` when the planner refused."""

    plan: Plan | None = None
    refusal: bool = False
    problems: list[Problem] = field(default_factory=list)
    # Syntax fixes (`parse_object`). They change no content, so they aren't in the plan's `repairs`.
    syntax: list[str] = field(default_factory=list)
    # The stitched length of the shots as the planner wrote them, snapped to the grid and caps, before the fit.
    model_duration_s: float | None = None

    def deliverable(self) -> Plan:
        """The plan, with a notice in `repairs` for each problem it still has."""
        if self.plan is None:
            raise PlanError("there is no plan to deliver")
        notices = [problem.notice for problem in self.problems if problem.notice]
        return self.plan.model_copy(update={"repairs": [*self.plan.repairs, *notices]}) if notices else self.plan


def choose(first: Repaired, retry: Repaired) -> Repaired:
    """Which of a reply and its retry to deliver: one with a plan over one without, then fewer problems, then the retry."""
    rank = lambda result: (result.plan is not None, -len(result.problems))  # noqa: E731
    return retry if rank(retry) >= rank(first) else first


@dataclass
class _Draft:
    beat: str
    prompt: str
    duration: float | None
    join: str | None


def repair(
    raw_text: str, context: PlanContext, *, planner: str, brief: str = "", revise: PlanRevision | None = None
) -> Repaired:
    """The planner's reply as a Plan for `context`, with every change recorded (research/director-design §5.3).

    1. Parse (`parse_object`). No object is the fatal problem `unparseable`.
    2. A `{"refusal": …}` object without shots is a refusal.
    3. Shots: non-objects and blank prompts are dropped, and at most `max_shots` kept; under 2 is fatal (`too_few_shots`).
       Text is tidied (`_clean_text`), leading labels ("Shot 3:", "Prompt:") are stripped, and a join word the planner
       wrote as a prompt's last sentence ("Cut.") is removed, since the video model would read it as an instruction. A
       missing beat is the prompt's first words.
    4. Joins: the first is `fresh`; an unknown one becomes `cut`; a `continue` whose prompt names a different shot size
       than the shot before becomes `cut`, since a continued take starts from the previous shot's last frames.
    5. Durations: a missing one takes the suggested shot length; then `fit`.
    6. Text is cut to its limits at the last sentence end (or word) that fits: title, scene, notes, beats, and each
       prompt so that `shot_prompt(scene, prompt)` fits `max_prompt_chars`.
    7. The result must pass `validate`; if it can't, that is fatal (`invalid`).
    8. Problems a delivered plan can live with, for the retry: fewer than `min_shots` (`few_shots`), the planner's own
       stitched length more than 25% short (`short`), a phrase the brief puts in quotes that no prompt contains
       (`missing_quote`).

    A revision with `shots` takes only those shots from the reply, by number; everything else is the earlier plan's."""
    result = Repaired()
    try:
        data, result.syntax = parse_object(raw_text)
    except PlanParseError as exc:
        result.problems.append(Problem("unparseable", str(exc)))
        return result
    items = data.get("shots") if isinstance(data.get("shots"), list) else []
    if isinstance(data.get("refusal"), str) and not items:
        result.refusal = True
        return result

    repairs: list[str] = []
    rewrite = revise is not None and revise.shots is not None
    if rewrite:
        assert revise is not None and revise.shots is not None
        earlier = revise.plan
        chosen: dict[int, _Draft] = {}
        for number in revise.shots:
            draft = _clean_shot(items[number - 1]) if number <= len(items) else None
            if draft is None:
                result.problems.append(Problem("unrevised", f"Your plan had no usable shot {number}; output every shot, with shot {number} rewritten."))
            else:
                chosen[number - 1] = draft
        if result.problems:
            return result
        title, scene, notes = earlier.title, earlier.scene, earlier.notes
        movable = sorted(chosen)
        drafts = [chosen.get(index) or _Draft(shot.beat, shot.prompt, shot.duration_s, shot.join) for index, shot in enumerate(earlier.shots)]
    else:
        drafts = [draft for draft in map(_clean_shot, items) if draft is not None]
        if len(drafts) > context.max_shots:
            repairs.append(f"kept the first {context.max_shots} of the {len(drafts)} shots")
            drafts = drafts[: context.max_shots]
        if len(drafts) < 2:
            count = len(drafts)
            result.problems.append(Problem(
                "too_few_shots",
                f"Your plan had {count} usable shot{'' if count == 1 else 's'}; write between {context.min_shots} and {context.max_shots} shots, each with a prompt.",
            ))
            return result
        title, scene, notes = (_clean_text(data.get(key)) for key in ("title", "scene", "notes"))
        movable = list(range(len(drafts)))

    unnamed = [index for index in movable if not drafts[index].beat]
    for index in unnamed:
        drafts[index].beat = _beat_from(drafts[index].prompt)
    if unnamed:
        repairs.append(f"{_shots_text(unnamed)} had no beat; named from the prompt")
    tails = [index for index in movable if _strip_join_words(drafts[index])]
    if tails:
        repairs.append(f"removed a join word written at the end of the prompt of {_shots_text(tails)}")
    _repair_joins(drafts, set(movable), repairs)
    suggested = suggested_shots(context)[1]
    untimed = [index for index in movable if drafts[index].duration is None]
    for index in untimed:
        drafts[index].duration = suggested
    if untimed:
        repairs.append(f"{_shots_text(untimed)} had no length; given {_seconds(suggested)} s")

    shots = [PlannedShot(beat=d.beat, prompt=d.prompt, duration_s=float(d.duration or suggested), join=d.join or "cut") for d in drafts]  # type: ignore[arg-type]
    written = [ShotSpec(duration_s=snap_duration(s.duration_s, context) if i in movable else s.duration_s, join=s.join) for i, s in enumerate(shots)]
    result.model_duration_s = storyboard_duration_s(context.profile, written, context.fps)
    shots, fitted = fit(shots, context, movable=movable)
    repairs += fitted

    if not rewrite:
        title = _cut_noted(title, TITLE_MAX_CHARS, "the title", repairs)
        scene = _cut_noted(scene, SCENE_MAX_CHARS, "the scene", repairs)
        notes = _cut_noted(notes, NOTES_MAX_CHARS, "the notes", repairs)
    budget = max(1, context.profile.limits.max_prompt_chars - (len(scene) + 2 if scene else 0))
    for index in movable:
        shot = shots[index]
        beat = _cut_noted(shot.beat, BEAT_MAX_CHARS, f"shot {index + 1}'s beat", repairs)
        prompt = _cut_noted(shot.prompt, budget, f"shot {index + 1}'s prompt", repairs)
        shots[index] = shot.model_copy(update={"beat": beat, "prompt": prompt})

    specs = [ShotSpec(duration_s=shot.duration_s, join=shot.join) for shot in shots]
    plan = Plan(
        profile_id=context.profile.id, resolution=context.resolution, aspect_ratio=context.aspect_ratio, fps=context.fps,
        audio=context.audio, target_s=context.target_s, duration_s=storyboard_duration_s(context.profile, specs, context.fps),
        title=title, scene=scene, shots=shots, notes=notes, repairs=repairs,
        planner=PlannerInfo(model=planner, prompt_version=context.prompt_version),
    )
    try:
        validate(plan, context.profile, context=context)
    except PlanError as exc:
        result.problems.append(Problem("invalid", f"Your plan doesn't fit the limits: {exc}."))
        return result
    result.plan = plan

    if len(shots) < context.min_shots:
        result.problems.append(Problem(
            "few_shots", f"Your plan has {len(shots)} shots; use between {context.min_shots} and {context.max_shots}.",
            f"the plan has {len(shots)} shots, fewer than the {context.min_shots} asked for",
        ))
    if result.model_duration_s < context.target_s * (1 - SHORT_FRACTION) - _EPSILON:
        result.problems.append(Problem(
            "short", f"Your plan runs {_seconds(result.model_duration_s)} s; the target is {_seconds(context.target_s)} s. Add shots or make them longer.", "",
        ))
    for quote in missing_quotes(brief, (shot.prompt for shot in shots)):
        result.problems.append(Problem(
            "missing_quote", f'Put the words the brief quotes, "{quote}", exactly and in double quotes into one of the prompts.',
            f'the brief\'s quoted words "{quote}" are in no shot',
        ))
    return result


_LABEL = re.compile(r"^(?:shot[\t\n\v\f\r ]*\d+[\t\n\v\f\r ]*[:.)-]|prompt[\t\n\v\f\r ]*:|beat[\t\n\v\f\r ]*:)[\t\n\v\f\r ]*", re.I)
_JOIN_TAIL = re.compile(r"(?<=[.!?])[\t\n\v\f\r ]*(?:join[\t\n\v\f\r ]*:[\t\n\v\f\r ]*)?(?:fresh|continue|cut|this is a fresh moment)[.!]?[\t\n\v\f\r ]*$", re.I)
_DURATION_TEXT = re.compile(r"^[\t\n\v\f\r ]*(\d+(?:\.\d+)?)[\t\n\v\f\r ]*(?:s|sec|secs|second|seconds)?[\t\n\v\f\r ]*$", re.I)


def _clean_shot(item: Any) -> _Draft | None:
    if not isinstance(item, dict):
        return None
    prompt = _strip_labels(_clean_text(item.get("prompt")))
    if not prompt:
        return None
    return _Draft(beat=_strip_labels(_clean_text(item.get("beat"))), prompt=prompt, duration=_duration_value(item.get("duration_s")), join=_join_value(item.get("join")))


def _strip_labels(text: str) -> str:
    previous = None
    while previous != text:
        previous, text = text, _LABEL.sub("", text, count=1)
    return text


def _strip_join_words(draft: _Draft) -> bool:
    stripped = False
    while (match := _JOIN_TAIL.search(draft.prompt)) is not None:
        draft.prompt = _clean_space(draft.prompt[: match.start()])
        stripped = True
    return stripped


def _duration_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return float(value)
    if isinstance(value, str) and (match := _DURATION_TEXT.match(value)) and float(match.group(1)) > 0:
        return float(match.group(1))
    return None


def _join_value(value: Any) -> str | None:
    text = _clean_space(value).lower() if isinstance(value, str) else None
    return text if text in ("fresh", "continue", "cut") else None


def _repair_joins(drafts: list[_Draft], moving: set[int], repairs: list[str]) -> None:
    unknown, resized = [], []
    for index, draft in enumerate(drafts):
        if index not in moving:
            continue
        if index == 0:
            if draft.join not in (None, "fresh"):
                repairs.append("shot 1's join became fresh: nothing comes before it")
            draft.join = "fresh"
            continue
        if draft.join is None:
            draft.join = "cut"
            unknown.append(index)
        elif draft.join == "continue":
            before, now = shot_size(drafts[index - 1].prompt), shot_size(draft.prompt)
            if before is not None and now is not None and before != now:
                draft.join = "cut"
                resized.append(index)
    if unknown:
        repairs.append(f"{_shots_text(unknown)} had no valid join; made {'a cut' if len(unknown) == 1 else 'cuts'}")
    if resized:
        repairs.append(f"{_shots_text(resized)} continued the take before at a different shot size; made {'a cut' if len(resized) == 1 else 'cuts'}")


# Shot sizes a continued take keeps, each with the words that name it.
SHOT_SIZES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("extreme close-up", ("extreme close-up", "extreme close up", "extreme closeup")),
    ("medium close-up", ("medium close-up", "medium close up", "medium closeup")),
    ("close-up", ("close-up", "close up", "closeup")),
    ("extreme wide", ("extreme wide shot", "extreme long shot")),
    ("medium wide", ("medium wide shot", "medium long shot", "medium full shot", "cowboy shot")),
    ("medium", ("medium shot", "mid shot", "mid-shot")),
    ("wide", ("wide shot", "long shot", "establishing shot", "wide-angle shot", "wide angle shot")),
    ("full", ("full shot", "full-body shot", "full body shot")),
    ("overhead", ("overhead shot", "top-down shot", "top down shot", "bird's-eye", "bird's eye", "birds-eye", "birds eye")),
    ("aerial", ("aerial shot", "drone shot")),
    ("insert", ("insert shot", "detail shot", "macro shot")),
    ("over the shoulder", ("over-the-shoulder", "over the shoulder")),
    ("point of view", ("point-of-view", "point of view", "pov shot")),
)
# Longest words first, so "medium close-up" is found before "close-up" at the same place.
_SIZE_WORDS = sorted(((word, size) for size, words in SHOT_SIZES for word in words), key=lambda pair: (-len(pair[0]), pair[0]))
_SIZE_PATTERN = re.compile("(?<![a-z0-9])(" + "|".join(re.escape(word) for word, _ in _SIZE_WORDS) + ")(?![a-z0-9])")
_SIZE_OF = dict(_SIZE_WORDS)


def shot_size(prompt: str) -> str | None:
    """The first shot size a prompt names, or None."""
    match = _SIZE_PATTERN.search(prompt.lower())
    return _SIZE_OF[match.group(1)] if match else None


# ------------------------------------------------------------------ durations


def snap_duration(value: float, context: PlanContext) -> float:
    """The nearest duration on the profile's grid (min_duration_s + k × duration_step_s, half up), within the context's
    shortest and longest shot."""
    lim = context.profile.limits
    steps = math.floor((value - lim.min_duration_s) / lim.duration_step_s + 0.5)
    snapped = round(lim.min_duration_s + steps * lim.duration_step_s, 6)
    return min(max(snapped, context.min_shot_s), context.max_shot_s)


def fit(shots: list[PlannedShot], context: PlanContext, *, movable: Iterable[int] | None = None) -> tuple[list[PlannedShot], list[str]]:
    """Durations that keep to the grid and caps and bring the stitched length within TARGET_TOLERANCE_S of the target,
    changing only the `movable` shots (default: all), and what changed, in words.

    Each movable duration is snapped (`snap_duration`). Then, while the plan is short and a movable shot is below the
    longest, the shortest (the first on ties) gains a step; while it is long, or over the storyboard's `max_total_s`, the
    longest (the first on ties) loses one, unless that would make it short while within `max_total_s`. At 24 fps a step
    changes the stitched length by exactly 1 s, so a reachable target is always met."""
    count = len(shots)
    moving = sorted({index for index in (range(count) if movable is None else movable) if 0 <= index < count})
    joins = [shot.join for shot in shots]
    durations = [float(shot.duration_s) for shot in shots]
    repairs: list[str] = []
    for index in moving:
        snapped = snap_duration(durations[index], context)
        if abs(snapped - durations[index]) > _EPSILON:
            repairs.append(_snap_note(index, durations[index], snapped, context))
        durations[index] = snapped

    def stitched(values: list[float]) -> float:
        return storyboard_duration_s(context.profile, [ShotSpec(duration_s=d, join=j) for d, j in zip(values, joins)], context.fps)

    start = list(durations)
    before = current = stitched(durations)
    low, high, ceiling, step = context.target_s - TARGET_TOLERANCE_S, context.target_s + TARGET_TOLERANCE_S, context.max_total_s, context.step_s
    while current < low - _EPSILON:
        growable = [index for index in moving if durations[index] + step <= context.max_shot_s + _EPSILON]
        if not growable:
            break
        index = min(growable, key=lambda i: (durations[i], i))
        durations[index] = round(durations[index] + step, 6)
        current = stitched(durations)
    while current > high + _EPSILON or current > ceiling + _EPSILON:
        shrinkable = [index for index in moving if durations[index] - step >= context.min_shot_s - _EPSILON]
        if not shrinkable:
            break
        index = min(shrinkable, key=lambda i: (-durations[i], i))
        trial = list(durations)
        trial[index] = round(trial[index] - step, 6)
        after = stitched(trial)
        if after < low - _EPSILON and current <= ceiling + _EPSILON:
            break
        durations, current = trial, after

    longer = [index for index in moving if durations[index] > start[index] + _EPSILON]
    shorter = [index for index in moving if durations[index] < start[index] - _EPSILON]
    changes = [f"{_shots_text(indices)} {verb}" for indices, verb in ((longer, "lengthened"), (shorter, "shortened")) if indices]
    if changes:
        repairs.append(f"the shots ran {_seconds(before)} s; {' and '.join(changes)} to reach {_seconds(current)} s")
    if current < low - _EPSILON:
        repairs.append(
            f"the plan runs {_seconds(current)} s of its {_seconds(context.target_s)} s target: the shots that may change are at "
            f"the longest this plan allows, {_seconds(context.max_shot_s)} s"
        )
    elif current > high + _EPSILON:
        repairs.append(
            f"the plan runs {_seconds(current)} s against its {_seconds(context.target_s)} s target: the shots that may change "
            f"are at the shortest, {_seconds(context.min_shot_s)} s"
        )
    fitted = [shot.model_copy(update={"duration_s": durations[index]}) if index in moving else shot for index, shot in enumerate(shots)]
    return fitted, repairs


def _snap_note(index: int, before: float, after: float, context: PlanContext) -> str:
    number = index + 1
    if before > context.max_shot_s + _EPSILON:
        return f"shot {number} shortened from {_seconds(before)} s to {_seconds(after)} s, the longest shot this plan allows"
    if before < context.min_shot_s - _EPSILON:
        return f"shot {number} lengthened from {_seconds(before)} s to {_seconds(after)} s, the shortest shot {context.profile.name} renders"
    return f"shot {number}'s length rounded from {_seconds(before)} s to {_seconds(after)} s"


def _grid_floor(value: float, profile: ModelProfile) -> float:
    lim = profile.limits
    steps = math.floor((value - lim.min_duration_s) / lim.duration_step_s + 1e-9)
    return round(lim.min_duration_s + steps * lim.duration_step_s, 6)


# ------------------------------------------------------------------ validation


def validate(plan: Plan, profile: ModelProfile, *, context: PlanContext | None = None) -> None:
    """Every rule a delivered plan keeps (research/director-design §5.2). Raises PlanError naming the first it breaks.

    2 to `storyboard.max_shots` shots with non-blank prompts; the first join `fresh`; each duration on the grid within the
    profile's limits at this fps; `duration_s` exactly the stitched length and within `max_total_s`; every
    `shot_prompt(scene, prompt)` within `max_prompt_chars`; title, scene, notes and beats within their limits; so the
    storyboard params pass `validate_params`. With a context, also its frame, target, longest shot and shot count."""
    lim, board = profile.limits, profile.limits.storyboard
    if board is None:
        raise PlanError(f"{profile.name} does not make storyboards")
    if plan.profile_id != profile.id:
        raise PlanError(f"the plan is for {plan.profile_id}, not {profile.id}")
    if not 2 <= len(plan.shots) <= board.max_shots:
        raise PlanError(f"a plan needs between 2 and {board.max_shots} shots")
    cap = profile_max_duration(profile, plan.fps)
    if context is not None:
        if not context.matches(plan) or abs(plan.target_s - context.target_s) > _EPSILON:
            raise PlanError("the plan's frame or target differs from the job's")
        if len(plan.shots) > context.max_shots:
            raise PlanError(f"the plan has more than {context.max_shots} shots")
        cap = min(cap, context.max_shot_s)
    for name, value, limit in (("title", plan.title, TITLE_MAX_CHARS), ("scene", plan.scene, SCENE_MAX_CHARS), ("notes", plan.notes, NOTES_MAX_CHARS)):
        if len(value) > limit:
            raise PlanError(f"the {name} is longer than {limit} characters")
    for number, shot in enumerate(plan.shots, start=1):
        if not shot.prompt.strip():
            raise PlanError(f"shot {number} has no prompt")
        if len(shot.beat) > BEAT_MAX_CHARS:
            raise PlanError(f"shot {number}'s beat is longer than {BEAT_MAX_CHARS} characters")
        if shot.duration_s > cap + _EPSILON:
            raise PlanError(f"shot {number} is longer than {cap:g} s")
        if len(shot_prompt(plan.scene, shot.prompt)) > lim.max_prompt_chars:
            raise PlanError(f"shot {number}'s prompt, with the scene, is longer than {lim.max_prompt_chars} characters")
    try:
        validate_params(profile, plan.storyboard_params())
    except (ParamError, ValidationError) as exc:
        raise PlanError(str(exc)) from None


# ------------------------------------------------------------------ brief quotes

# Paired quotation marks: ASCII double, curly double, German low-high, guillemets both ways, curly single, and ASCII
# single quotes only where they can't be an apostrophe (not after or before an ASCII letter or digit).
_QUOTE_PATTERNS = (
    re.compile(r'"([^"\n]+)"'),
    re.compile("“([^“”\n]+)”"),
    re.compile("„([^“”„\n]+)[“”]"),
    re.compile("«([^«»\n]+)»"),
    re.compile("»([^«»\n]+)«"),
    re.compile("(?<![A-Za-z0-9])‘([^‘’\n]+)’(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9'])'(?![\t\n\v\f\r ])([^'\n]+?)(?<![\t\n\v\f\r ])'(?![A-Za-z0-9])"),
)
# diffusers' LTX-2 `_UNICODE_REPLACEMENTS` (prompt_enhancement.py): curly quotes, dashes, no-break space, prime and minus.
_UNICODE_REPLACEMENTS = str.maketrans("‘’“”—– ′−", "''\"\"-- '-")
_QUOTE_TRIM = " .,!?;:\"'"


def brief_quotes(brief: str) -> list[str]:
    """The phrases a brief puts in quotation marks, in order of appearance, each once: a slogan, or a line to be spoken."""
    found: list[tuple[int, str]] = []
    for pattern in _QUOTE_PATTERNS:
        for match in pattern.finditer(brief):
            phrase = _clean_space(match.group(1))
            if QUOTE_MIN_CHARS <= len(_quote_key(phrase)) <= QUOTE_MAX_CHARS:
                found.append((match.start(), phrase))
    phrases: list[str] = []
    for _, phrase in sorted(found, key=lambda item: item[0]):
        if all(_quote_key(phrase) != _quote_key(seen) for seen in phrases):
            phrases.append(phrase)
    return phrases


def missing_quotes(brief: str, prompts: Iterable[str]) -> list[str]:
    """The brief's quoted phrases that appear in none of the prompts, compared after NFKC, the diffusers quote mapping,
    lower-casing, collapsed whitespace and trimmed end punctuation ("Sip. Stay fresh." matches `"sip. stay fresh!"`)."""
    keys = [_quote_key(prompt) for prompt in prompts]
    return [phrase for phrase in brief_quotes(brief) if not any(_quote_key(phrase) in key for key in keys)]


def _quote_key(text: str) -> str:
    return _trim(_clean_space(unicodedata.normalize("NFKC", text).translate(_UNICODE_REPLACEMENTS).lower()), _QUOTE_TRIM)


# ------------------------------------------------------------------ output


def plan_output_label(job_id: str) -> str:
    return f"{job_id}/output/plan"


def encode_plan(plan: Plan) -> bytes:
    """The plan JSON a job delivers: canonical JSON (PROTOCOL.md "Encodings"). The receipt's `content_digest` is its SHA-256."""
    return canonical_json(plan.model_dump(mode="json"))


def pad_plan(plan_json: bytes) -> bytes:
    """The sealed plaintext: the request padding framing, `0x02 | length:u32be | JSON | 0x00 …` to a power-of-two bucket
    from 4 KiB. A plan is 2-5 KB, where blob PADMÉ alone would give its length away to within a few percent, and with it
    roughly how many shots it has."""
    return pad_payload(plan_json)


def seal_plan(output_key: bytes, job_id: str, plan: Plan) -> tuple[bytes, bytes]:
    """(plan JSON, sealed output blob) for a plan job: `encrypt_blob(output key, "<job_id>/output/plan", pad_plan(JSON))`."""
    data = encode_plan(plan)
    return data, encrypt_blob(output_key, plan_output_label(job_id), pad_plan(data))


def open_plan(output_key: bytes, job_id: str, blob: bytes) -> tuple[Plan, bytes]:
    """A delivered plan and its JSON, whose SHA-256 is the receipt's `content_digest`. Raises DecryptionError when the
    blob fails authentication, PlanError when it isn't a padded Plan v1."""
    framed = decrypt_blob(output_key, plan_output_label(job_id), blob)
    if payload_version(framed) != PAYLOAD_V2:
        raise PlanError("the plan output is not padded plan JSON")
    try:
        data = unpad_payload(framed)
    except MalformedPayload as exc:
        raise PlanError(f"the plan output's padding is invalid: {exc}") from None
    try:
        return Plan.model_validate_json(data), data
    except ValidationError:
        raise PlanError("the plan output is not a Plan v1") from None


# ------------------------------------------------------------------ text helpers

# ECMAScript's `\s`: whitespace everywhere in this module, so every language splits and trims alike.
_SPACE_CHARS = "\t\n\v\f\r                  　﻿"
_SPACE = re.compile(f"[{_SPACE_CHARS}]+")
_SENTENCE_END = re.compile(f"[.!?][\"')\\]]*(?=[{_SPACE_CHARS}]|$)")
_MARKDOWN = re.compile(r"\*\*|__|`")
_LEADING_MARK = re.compile(r"^(?:#{1,6}|[-*•])(?=[\t\n\v\f\r ])")


def _clean_space(text: str) -> str:
    return _trim(_SPACE.sub(" ", text), " ")


def _trim(text: str, chars: str) -> str:
    start, end = 0, len(text)
    while start < end and text[start] in chars:
        start += 1
    while end > start and text[end - 1] in chars:
        end -= 1
    return text[start:end]


def _clean_text(value: Any) -> str:
    """A string field tidied: curly quotes and dashes mapped as diffusers maps them, Markdown emphasis and backticks and a
    leading heading or bullet mark removed, whitespace collapsed. Anything but a string is empty."""
    if not isinstance(value, str):
        return ""
    text = _clean_space(_MARKDOWN.sub("", value.translate(_UNICODE_REPLACEMENTS)))
    return _clean_space(_LEADING_MARK.sub("", text))


def _beat_from(prompt: str) -> str:
    return _trim(" ".join(prompt.split(" ")[:BEAT_WORDS]), " ,;:")


def cut_text(text: str, limit: int) -> str:
    """`text` cut to at most `limit` code points: at the last sentence end that fits, else the last space, else hard."""
    if len(text) <= limit:
        return text
    ends = [match.end() for match in _SENTENCE_END.finditer(text) if match.end() <= limit]
    if ends:
        return text[: ends[-1]]
    head = text[:limit]
    space = head.rfind(" ")
    return head[:space].rstrip(" ") if space > 0 else head


def _cut_noted(text: str, limit: int, what: str, repairs: list[str]) -> str:
    cut = cut_text(text, limit)
    if cut != text:
        repairs.append(f"{what} was cut to {len(cut)} characters")
    return cut


def _seconds(value: float) -> str:
    """Seconds with at most three decimals, rounded half up on the double as it is: the same digits in every language."""
    whole, fraction = divmod(math.floor(value * 1000 + 0.5), 1000)
    return str(whole) if fraction == 0 else f"{whole}.{fraction:03d}".rstrip("0")


def _shots_text(indices: Iterable[int]) -> str:
    numbers = [str(index + 1) for index in sorted(set(indices))]
    if len(numbers) == 1:
        return f"shot {numbers[0]}"
    return f"shots {', '.join(numbers[:-1])} and {numbers[-1]}"


def _json_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value
