"""Model profiles: the unit miners serve, validators audit and customers buy.

A profile pins one model family + checkpoint + step count + hardware class and
declares exactly which generation modes and parameters it accepts. Profiles are
data (profiles.json) so capability updates don't require code changes.

Capabilities follow the official inference code (see research/research_model_capabilities.md):
  MiniMax H3  FL2VA checkpoint: text, first frame, last frame, first+last frame.
              Ref2VA checkpoint: up to 9 images / 3 videos / 3 audio (12 total),
              which also covers video edit, extend and audio-driven video.
              24 fps, 5–14 s (345-frame cap), frames = 17n+5, ≤ 1,032,192 px, audio always on.
  LTX-2.5     distilled / full / DFR pipelines: text, first/last frame, any number
              of keyframes, retake of a time window, audio-to-video (full), 4K (DFR).
              frames = 8k+1, sizes divisible by 64 (128 for DFR).
"""

from __future__ import annotations

import json
import math
from enum import Enum
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .schemas import GenerationParams

FAMILY_H3 = "minimax-h3"
FAMILY_LTX = "ltx-2.5"


class Mode(str, Enum):
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    LAST_FRAME = "last_frame"
    FIRST_LAST_FRAME = "first_last_frame"
    KEYFRAMES = "keyframes"
    REFERENCE_TO_VIDEO = "reference_to_video"
    VIDEO_EDIT = "video_edit"
    EXTEND_VIDEO = "extend_video"
    AUDIO_TO_VIDEO = "audio_to_video"
    RETAKE = "retake"


class InputRole(str, Enum):
    FIRST_FRAME = "first_frame"
    LAST_FRAME = "last_frame"
    KEYFRAME = "keyframe"
    REFERENCE_IMAGE = "reference_image"
    REFERENCE_VIDEO = "reference_video"
    REFERENCE_AUDIO = "reference_audio"
    SOURCE_VIDEO = "source_video"
    SOURCE_AUDIO = "source_audio"


R = InputRole

# (required roles, allowed roles) per mode.
MODE_ROLES: dict[Mode, tuple[frozenset[InputRole], frozenset[InputRole]]] = {
    Mode.TEXT_TO_VIDEO: (frozenset(), frozenset()),
    Mode.IMAGE_TO_VIDEO: (frozenset({R.FIRST_FRAME}), frozenset({R.FIRST_FRAME})),
    Mode.LAST_FRAME: (frozenset({R.LAST_FRAME}), frozenset({R.LAST_FRAME})),
    Mode.FIRST_LAST_FRAME: (frozenset({R.FIRST_FRAME, R.LAST_FRAME}), frozenset({R.FIRST_FRAME, R.LAST_FRAME})),
    Mode.KEYFRAMES: (frozenset({R.KEYFRAME}), frozenset({R.KEYFRAME})),
    Mode.REFERENCE_TO_VIDEO: (frozenset(), frozenset({R.REFERENCE_IMAGE, R.REFERENCE_VIDEO, R.REFERENCE_AUDIO})),
    Mode.VIDEO_EDIT: (frozenset({R.SOURCE_VIDEO}), frozenset({R.SOURCE_VIDEO, R.REFERENCE_IMAGE})),
    Mode.EXTEND_VIDEO: (frozenset({R.SOURCE_VIDEO}), frozenset({R.SOURCE_VIDEO, R.REFERENCE_IMAGE})),
    Mode.AUDIO_TO_VIDEO: (frozenset({R.SOURCE_AUDIO}), frozenset({R.SOURCE_AUDIO, R.FIRST_FRAME, R.REFERENCE_IMAGE})),
    Mode.RETAKE: (frozenset({R.SOURCE_VIDEO}), frozenset({R.SOURCE_VIDEO})),
}

VISUAL_ROLES = frozenset({R.FIRST_FRAME, R.LAST_FRAME, R.KEYFRAME, R.REFERENCE_IMAGE, R.REFERENCE_VIDEO, R.SOURCE_VIDEO})
AUDIO_ROLES = frozenset({R.REFERENCE_AUDIO, R.SOURCE_AUDIO})


class InputGroup(BaseModel):
    """A shared cap across several roles, e.g. H3's 3 videos in total."""

    roles: list[InputRole]
    max: int


class Limits(BaseModel):
    min_duration_s: float
    max_duration_s: float
    duration_step_s: float = 1.0
    # resolution label -> aspect ratio -> [width, height]
    sizes: dict[str, dict[str, tuple[int, int]]]
    fps: list[int]
    default_fps: int
    audio: bool
    max_inputs: dict[InputRole, int] = Field(default_factory=dict)
    input_groups: list[InputGroup] = Field(default_factory=list)
    max_total_inputs: int | None = None
    visual_required_with_audio: bool = False
    max_prompt_chars: int = 2000
    negative_prompt: bool = False
    prompt_enhancer: bool = False
    seed: bool = True


class LicenseInfo(BaseModel):
    name: str
    url: str
    attribution: str | None = None
    region_policy: str | None = None


class Pricing(BaseModel):
    usd_per_second: dict[str, float]


class ModelProfile(BaseModel):
    id: str
    family: str
    name: str
    tagline: str
    variant: str
    checkpoint: str
    runtime: str
    modes: list[Mode]
    limits: Limits
    hardware_class: str
    gpus_per_worker: int
    steps: int
    license: LicenseInfo
    pricing: Pricing
    vcu_per_output_second: float
    timeout_s: int = 1800
    provisional: bool = False

    def size_for(self, resolution: str, aspect_ratio: str) -> tuple[int, int]:
        try:
            width, height = self.limits.sizes[resolution][aspect_ratio]
        except KeyError:
            raise ParamError(f"{self.name} does not support {resolution} at {aspect_ratio}") from None
        return width, height

    def price_usd(self, params: GenerationParams) -> float:
        rate = self.pricing.usd_per_second.get(params.resolution)
        if rate is None:
            raise ParamError(f"{self.name} has no price for {params.resolution}")
        return round(rate * params.duration_s, 4)

    def vcu(self, duration_s: float) -> float:
        return self.vcu_per_output_second * duration_s

    def num_frames(self, duration_s: float, fps: int) -> int:
        """Frame count the model actually renders for a requested duration."""
        if self.family == FAMILY_H3:
            return h3_num_frames(duration_s)
        return ltx_num_frames(duration_s, fps)


def h3_num_frames(duration_s: float) -> int:
    """H3 renders 17n+5 frames at 24 fps (diffusers rounds up to this grid), capped at 345."""
    return min(345, 17 * math.ceil((24 * duration_s - 5) / 17) + 5)


def ltx_num_frames(duration_s: float, fps: int) -> int:
    """LTX requires 8k+1 frames."""
    return 8 * max(1, round(duration_s * fps / 8)) + 1


class ParamError(ValueError):
    """A request doesn't fit the chosen profile."""


def validate_params(profile: ModelProfile, params: GenerationParams) -> None:
    """Checks the public (gateway-visible) parameters against a profile."""
    lim = profile.limits
    if params.profile_id != profile.id:
        raise ParamError("params.profile_id does not match profile")
    if params.mode not in profile.modes:
        raise ParamError(f"{profile.name} does not support {params.mode.value}")
    if not lim.min_duration_s <= params.duration_s <= lim.max_duration_s:
        raise ParamError(f"duration must be between {lim.min_duration_s:g} and {lim.max_duration_s:g} seconds")
    steps_from_min = (params.duration_s - lim.min_duration_s) / lim.duration_step_s
    if abs(steps_from_min - round(steps_from_min)) > 1e-6:
        raise ParamError(f"duration must be in {lim.duration_step_s:g}-second steps")
    profile.size_for(params.resolution, params.aspect_ratio)
    if params.fps not in lim.fps:
        raise ParamError(f"fps must be one of {lim.fps}")
    if params.audio and not lim.audio:
        raise ParamError(f"{profile.name} cannot generate audio")
    validate_roles(profile, params.mode, params.input_roles)


def validate_roles(profile: ModelProfile, mode: Mode, roles: list[InputRole]) -> None:
    lim = profile.limits
    required, allowed = MODE_ROLES[mode]
    present = set(roles)
    if missing := required - present:
        raise ParamError(f"{mode.value} needs: {', '.join(sorted(r.value for r in missing))}")
    if extra := present - allowed:
        raise ParamError(f"{mode.value} does not accept: {', '.join(sorted(r.value for r in extra))}")
    if mode is Mode.REFERENCE_TO_VIDEO and not present:
        raise ParamError("reference_to_video needs at least one reference image, video or audio clip")
    for role in present:
        limit = lim.max_inputs.get(role, 0)
        if roles.count(role) > limit:
            raise ParamError(f"{profile.name} accepts at most {limit} {role.value} input(s)")
    for group in lim.input_groups:
        if sum(roles.count(r) for r in group.roles) > group.max:
            names = " + ".join(r.value for r in group.roles)
            raise ParamError(f"{profile.name} accepts at most {group.max} inputs across {names}")
    if lim.max_total_inputs is not None and len(roles) > lim.max_total_inputs:
        raise ParamError(f"{profile.name} accepts at most {lim.max_total_inputs} inputs in total")
    if lim.visual_required_with_audio and present & AUDIO_ROLES and not present & VISUAL_ROLES:
        raise ParamError(f"{profile.name} needs an image or video alongside audio references")


@lru_cache(maxsize=1)
def load_profiles() -> dict[str, ModelProfile]:
    """Profiles in catalog order (the order is used for defaults and fallbacks)."""
    raw = json.loads(resources.files(__package__).joinpath("profiles.json").read_text())
    return {p["id"]: ModelProfile.model_validate(p) for p in raw["profiles"]}
