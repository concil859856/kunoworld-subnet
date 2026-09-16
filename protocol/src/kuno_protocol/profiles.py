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
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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


def example_roles(mode: Mode) -> list[InputRole]:
    """The inputs a mode is normally called with: everything it requires, or one reference image."""
    required, _ = MODE_ROLES[mode]
    if required:
        return sorted(required, key=lambda r: r.value)
    return [InputRole.REFERENCE_IMAGE] if mode is Mode.REFERENCE_TO_VIDEO else []


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
    # fps -> a lower duration cap at that frame rate, e.g. LTX-2.5 Fast renders over 10 s only at 24/25 fps.
    max_duration_s_by_fps: dict[int, float] = Field(default_factory=dict)


class LicenseInfo(BaseModel):
    name: str
    url: str
    attribution: str | None = None
    region_policy: str | None = None


PRIVACY_MODES = ("private", "standard")


class LongClip(BaseModel):
    """A multiplier on a whole Private job once its duration exceeds `over_s`. Standard prices follow the market's list
    prices, which are flat per second, so it never applies to them."""

    over_s: float
    multiplier: float


class Pricing(BaseModel):
    # The Private price per resolution. Private is the default mode, so a client that reads only this sees what a
    # default job costs.
    usd_per_second: dict[str, float]
    # The Standard price per resolution. None: the profile is offered in Private mode only.
    standard_usd_per_second: dict[str, float] | None = None
    # No job costs less than this, after multipliers.
    min_job_usd: float = 0.0
    long_clip: LongClip | None = None
    # fps -> a multiplier on the whole job.
    fps_multipliers: dict[int, float] = Field(default_factory=dict)


class HardwareClass(BaseModel):
    """One reproducibility domain. Bits match only within a class: same GPU SKU and form
    factor, same GPU count, same parallel layout, same pinned image."""

    id: str
    tier: str  # the README's C1/C2/C4/C8 serving class this belongs to ("dev" for simulated)
    gpu_sku: str
    gpu_count: int
    # Parallel degrees the runtime is pinned to, e.g. {"ulysses": 4}. Changing any changes bits.
    parallel: dict[str, int] = Field(default_factory=dict)
    interconnect: str | None = None
    # Simulated classes exist for dev networks; production validators refuse them.
    dev: bool = False
    # "bitwise": replayed steps must match exactly (one reproducibility domain, the confidential tier).
    # "tolerance": within a calibrated distance (kuno_protocol.tolerance; open-tier hardware).
    comparison: Literal["bitwise", "tolerance"] = "bitwise"
    # Weight precision the class runs ("bf16", "fp8-cast", "int8"...); part of the class, since it changes the model.
    precision: str | None = None
    # VRAM per GPU on this class, for operators choosing hardware.
    vram_gb: float | None = None


class DeterminismSettings(BaseModel):
    """What a verified-mode runtime pins before its first step (see VERIFIED_MODE.md)."""

    use_deterministic_algorithms: bool = True
    cublas_workspace_config: str = ":4096:8"
    allow_tf32: bool = False
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    float32_matmul_precision: str = "highest"
    torch_compile: bool = False
    kernel_autotuning: bool = False
    # TeaCache / MagCache / Cache-DiT style step skipping; always off in verified mode.
    data_dependent_caching: bool = False
    attention_backend: str = "sdpa"
    # The initial latent comes from torch.Generator("cpu") seeded with the job seed.
    noise: str = "torch-cpu-generator"
    latent_dtype: str = "bfloat16"
    env: dict[str, str] = Field(default_factory=dict)


class VerifiedMode(BaseModel):
    """The deterministic variant of a profile that commits to every denoising step."""

    variant: str
    runtime: str
    scheduler: str
    # Denoising steps per stage, e.g. [8, 3] for distilled + refine; leaves = Σ (steps + 1).
    stage_steps: list[int]
    # Stages a single-step re-executor supports (stage transitions such as upsampling are not steps).
    replayable_stages: list[int] = Field(default_factory=lambda: [0])
    hardware_classes: list[HardwareClass]
    determinism: DeterminismSettings = Field(default_factory=DeterminismSettings)
    # Retain every k-th latent and recompute the rest on audit (1 = retain every step).
    retention_checkpoint_every: int = 1
    # Share of a validator's own canaries to step-audit.
    audit_rate: float = 0.03

    def hardware_class(self, class_id: str) -> HardwareClass | None:
        return next((h for h in self.hardware_classes if h.id == class_id), None)

    @property
    def leaves(self) -> int:
        return sum(steps + 1 for steps in self.stage_steps)


class VcuWeights(BaseModel):
    """Verified video compute units (VCU): what a job's output costs in GPU time, comparable across profiles.

        VCU = per_output_second[resolution] × fps_multiplier[fps] × (1 + duration_slope × max(0, seconds − duration_base_s)) × seconds

    Weights follow GPU cost (research/pricing/costs.md §6.2, §8.4), anchored at `h3` 5 s = 60 per second, so a VCU is about
    the same GPU time in every profile and one USD rate per VCU pays every profile alike (rate_card.py). Every weight is a
    PLACEHOLDER until the benchmarks in research/research_pricing.md §8 have run.
    """

    # resolution label -> VCU per output second at 24/25 fps, for clips up to duration_base_s
    per_output_second: dict[str, float]
    # Longer clips cost more per second (attention grows with the frame count): the share added per second past the base.
    duration_slope: float = 0.0
    duration_base_s: float = 5.0
    # fps -> multiplier on the weight; 48 and 50 fps render twice the frames of 24 and 25. An fps not listed counts once.
    fps_multiplier: dict[int, float] = Field(default_factory=dict)
    note: str = ""

    @field_validator("per_output_second")
    @classmethod
    def _check_weights(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("a profile needs a VCU weight for at least one resolution")
        for resolution, weight in value.items():
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"{resolution}: a VCU weight must be a finite, positive number")
        return value

    @field_validator("fps_multiplier")
    @classmethod
    def _check_multipliers(cls, value: dict[int, float]) -> dict[int, float]:
        for fps, multiplier in value.items():
            if not math.isfinite(multiplier) or multiplier <= 0:
                raise ValueError(f"{fps} fps: a VCU multiplier must be a finite, positive number")
        return value

    @field_validator("duration_slope", "duration_base_s")
    @classmethod
    def _check_duration(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("the VCU duration slope and base must be finite and non-negative")
        return value

    def per_second(self, resolution: str, fps: int | None, seconds: float) -> float | None:
        """VCU per output second of a job, or None when there is no weight for its resolution. fps None counts once."""
        weight = self.per_output_second.get(resolution)
        if weight is None:
            return None
        multiplier = self.fps_multiplier.get(fps, 1.0) if fps is not None else 1.0
        return weight * multiplier * (1.0 + self.duration_slope * max(0.0, seconds - self.duration_base_s))


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
    # Per-GPU VRAM the official bf16 recipe needs; quantized paths need less.
    min_vram_gb: float = 0
    steps: int
    license: LicenseInfo
    pricing: Pricing
    # What a verified output second is worth in GPU time, by resolution, fps and duration. Profiles written before these
    # weights carry one `vcu_per_output_second` instead, read as that weight everywhere.
    vcu_weights: VcuWeights
    timeout_s: int = 1800
    provisional: bool = False
    # Deterministic variant with per-step commitments; None where no verified mode is defined.
    verified: VerifiedMode | None = None

    @model_validator(mode="before")
    @classmethod
    def _read_one_vcu_weight(cls, data):
        """A profile from before VcuWeights: its one `vcu_per_output_second` weighs every resolution, fps and duration
        alike, exactly as it was paid then."""
        if isinstance(data, dict) and "vcu_weights" not in data and "vcu_per_output_second" in data:
            data = dict(data)
            weight = data.pop("vcu_per_output_second")
            limits = data.get("limits")
            sizes = limits.sizes if isinstance(limits, Limits) else (limits.get("sizes") if isinstance(limits, dict) else None)
            data["vcu_weights"] = {"per_output_second": {resolution: weight for resolution in sizes or {}}}
        return data

    def vcu_for(self, params: GenerationParams, seconds: float | None = None) -> float:
        """The job's VCU from its public params: its resolution's weight × its fps multiplier × the duration factor ×
        `seconds` (the billable seconds, by default the requested duration). ParamError for a resolution without a weight."""
        return self.vcu_at(params.resolution, params.fps, params.duration_s if seconds is None else seconds)

    def vcu_at(self, resolution: str, fps: int | None, seconds: float) -> float:
        """VCU for `seconds` of output at this resolution and fps (None counts as 24/25 fps)."""
        per_second = self.vcu_weights.per_second(resolution, fps, seconds)
        if per_second is None:
            raise ParamError(f"{self.name} has no VCU weight for {resolution}")
        return per_second * seconds

    @property
    def base_vcu_resolution(self) -> str:
        """The lowest resolution with a VCU weight: fewest pixels, then the lowest weight."""
        weights = self.vcu_weights.per_output_second

        def pixels(resolution: str) -> float:
            return max((width * height for width, height in self.limits.sizes.get(resolution, {}).values()), default=math.inf)

        return min(weights, key=lambda resolution: (pixels(resolution), weights[resolution]))

    def size_for(self, resolution: str, aspect_ratio: str) -> tuple[int, int]:
        try:
            width, height = self.limits.sizes[resolution][aspect_ratio]
        except KeyError:
            raise ParamError(f"{self.name} does not support {resolution} at {aspect_ratio}") from None
        return width, height

    @property
    def privacy_modes(self) -> list[str]:
        """The privacy modes this profile is sold in: Private always, Standard where it has a Standard price."""
        return list(PRIVACY_MODES) if self.pricing.standard_usd_per_second is not None else ["private"]

    def offers(self, privacy: str) -> bool:
        return privacy in self.privacy_modes

    def price_usd(self, params: GenerationParams, privacy: str = "private") -> float:
        """Per-second rate x duration x the fps multiplier (and, in Private mode, the long-clip multiplier), never below
        the profile's minimum charge."""
        if privacy not in PRIVACY_MODES:
            raise ParamError(f"unknown privacy mode {privacy!r}")
        if not self.offers(privacy):
            raise PrivacyModeUnavailable(f"{self.name} is offered in Private mode only")
        pricing = self.pricing
        rates = pricing.usd_per_second if privacy == "private" else pricing.standard_usd_per_second
        rate = rates.get(params.resolution)
        if rate is None:
            raise ParamError(f"{self.name} has no {privacy} price for {params.resolution}")
        usd = rate * params.duration_s * pricing.fps_multipliers.get(params.fps, 1.0)
        if privacy == "private" and pricing.long_clip is not None and params.duration_s > pricing.long_clip.over_s:
            usd *= pricing.long_clip.multiplier
        return round(max(pricing.min_job_usd, usd), 4)

    def vcu(self, duration_s: float) -> float:
        """VCU for a job known only by its duration: the lowest resolution's weight (`base_vcu_resolution`) at the default
        fps, with the duration factor. It underpays higher resolutions and frame rates; callers with the params use `vcu_for`."""
        return self.vcu_at(self.base_vcu_resolution, self.limits.default_fps, duration_s)

    def num_frames(self, duration_s: float, fps: int) -> int:
        """Frame count the model actually renders for a requested duration."""
        if self.family == FAMILY_H3:
            return h3_num_frames(duration_s)
        return ltx_num_frames(duration_s, fps)


def h3_num_frames(duration_s: float) -> int:
    """H3 renders 17n+5 frames at 24 fps (diffusers rounds up to this grid), capped at 345."""
    return min(345, 17 * math.ceil((24 * duration_s - 5) / 17) + 5)


def h3_schedule_points(steps: int) -> int:
    """The `num_inference_steps` to send H3's runtimes for `steps` model passes. SGLang and diffusers both build the
    sigma grid as `linspace(1, 0, num_inference_steps)`, and its terminal 0 is not a pass, so N points run N - 1 passes.
    A profile's `steps` (and verified mode's `stage_steps`) count passes, which is also how LightX2V names its Turbo
    LoRAs: the 8-step LoRA is trained on the 9-point grid (ModelTC/Minimax-H3-Turbo, "Note on shift")."""
    return steps + 1


def ltx_num_frames(duration_s: float, fps: int) -> int:
    """LTX requires 8k+1 frames."""
    return 8 * max(1, round(duration_s * fps / 8)) + 1


class ParamError(ValueError):
    """A request doesn't fit the chosen profile."""


class PrivacyModeUnavailable(ParamError):
    """The profile isn't sold in the requested privacy mode (a profile with no Standard price is Private-only)."""


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
    fps_max = lim.max_duration_s_by_fps.get(params.fps)
    if fps_max is not None and params.duration_s > fps_max:
        raise ParamError(f"at {params.fps} fps, duration must be at most {fps_max:g} seconds")
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
