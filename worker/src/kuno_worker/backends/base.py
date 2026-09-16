from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from kuno_protocol.media import EXTENSIONS
from kuno_protocol.profiles import InputRole, Mode, ModelProfile, storyboard_trim_frames
from kuno_protocol.receipts import VideoInfo
from kuno_protocol.schemas import GenerationParams, InputRef
from kuno_protocol.verified import StepCommitment, Tensor

from ..verified import OpeningsHandle, RetentionStore, StepRecorder, shared_retention

ProgressFn = Callable[[float, str], None]

# The sealed option (SealedPayload.options) asking a profile with `limits.prompt_enhancer` to rewrite the prompt first.
ENHANCE_PROMPT_OPTION = "enhance_prompt"


@dataclass
class InputFile:
    ref: InputRef
    data: bytes
    mime: str
    path: str | None = None  # set by save(), for runtimes that take file paths

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"input-{self.ref.index}{EXTENSIONS.get(self.mime, '.bin')}"
        path.write_bytes(self.data)
        self.path = str(path)
        return path


@dataclass
class GenerationTask:
    job_id: str
    profile: ModelProfile
    params: GenerationParams
    prompt: str
    negative_prompt: str | None
    seed: int
    width: int
    height: int
    inputs: list[InputFile] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    # Storyboards: what the model sees for each shot, `profiles.shot_prompt(scene, shot)`, in order. `prompt` is the scene.
    shot_prompts: list[str] | None = None

    def first(self, role: InputRole) -> InputFile | None:
        return next((i for i in self.inputs if i.ref.role is role), None)

    def all(self, role: InputRole) -> list[InputFile]:
        return [i for i in self.inputs if i.ref.role is role]

    @property
    def num_frames(self) -> int:
        if self.params.shots:
            return sum(self.shot_frames)
        return self.profile.num_frames(self.params.duration_s, self.params.fps)

    @property
    def shot_frames(self) -> list[int] | None:
        """A storyboard's frames per shot in the stitched video, in order: each shot's rendered frames less the head a
        joined shot repeats (they sum to `profiles.storyboard_frames`). None for any other job."""
        if not self.params.shots:
            return None
        trim = storyboard_trim_frames(self.profile)
        fps = self.params.fps
        return [self.profile.num_frames(shot.duration_s, fps) - (0 if shot.join == "fresh" else trim) for shot in self.params.shots]

    def shot_task(self, index: int) -> GenerationTask:
        """Storyboard shot `index` as a text-to-video task of its own: its duration, its model prompt, and seed
        (seed + index) mod 2^31 (PROTOCOL.md)."""
        shots, prompts = self.params.shots or [], self.shot_prompts or []
        if len(prompts) != len(shots) or not 0 <= index < len(shots):
            raise ValueError("a storyboard task needs one model prompt per shot")
        params = self.params.model_copy(update={"mode": Mode.TEXT_TO_VIDEO, "duration_s": shots[index].duration_s, "shots": None})
        return replace(self, params=params, prompt=prompts[index], seed=(self.seed + index) % 2**31, inputs=[], shot_prompts=None)


@dataclass
class VideoResult:
    data: bytes
    info: VideoInfo
    # Verified mode: the per-step commitment to sign into the receipt (ReceiptBody.step_commitment)
    # and the handle to the retained trajectory audit openings are produced from.
    step_commitment: StepCommitment | None = None
    openings: OpeningsHandle | None = None


class StepSink(Protocol):
    """The step hook: a verified-mode backend reports the latent state after every step.

    Leaf 0 of each stage is the stage's initial latent (kind "init"); every later leaf is
    the state after one denoising step (kind "denoise"), with the sigma it has reached.
    Tensors are (TensorSpec, little-endian bytes) pairs, e.g. from `tensor_from_array`.
    """

    def report(self, index: int, stage: int, kind: str, sigma: float, tensors: list[Tensor]) -> None: ...


class Backend(ABC):
    name: str = "backend"
    # Verified mode is on for a profile when the backend knows its hardware class and the
    # profile pins a deterministic variant for that class.
    hardware_class: str | None = None
    retention: RetentionStore | None = None
    # Renders storyboard jobs (PROTOCOL.md, "Storyboards"). A backend without it would render one clip of the stitched
    # length, so the worker refuses the job instead.
    storyboards: bool = False
    # Rewrites a prompt with a language model the backend already holds (LTX-2.5's bundled prompt enhancer), as a step of
    # its own: the worker checks the text `enhance_prompt` returns exactly as it checks a customer's prompt, then renders
    # it (worker.Worker._enhance). `generate` never rewrites a prompt itself, since nothing would check what it wrote.
    # Without it the option has no effect, and the prompt renders as sent.
    prompt_enhancement: bool = False

    def warm(self, profile: ModelProfile) -> None:
        """Load weights ahead of the first job. TEE model loads are slow; do it once."""

    def serving_envelope(self, profile: ModelProfile) -> dict[str, dict[str, dict[int, float]]]:
        """The longest duration this backend serves at each resolution, aspect ratio and fps of `profile`
        (kuno_protocol.envelope). A backend whose hardware holds the whole profile serves its full limits;
        one that plans memory against a smaller card (backends/quantized.py) overrides this."""
        from kuno_protocol.envelope import full_table

        return full_table(profile)

    def verified_enabled(self, profile: ModelProfile) -> bool:
        return (
            profile.verified is not None
            and self.hardware_class is not None
            and profile.verified.hardware_class(self.hardware_class) is not None
        )

    def step_recorder(self, task: GenerationTask, context: bytes = b"") -> StepRecorder | None:
        """A recorder for this job when verified mode applies, else None. Storyboards carry no step commitment, whatever
        the hardware class."""
        if task.params.mode is Mode.STORYBOARD or not self.verified_enabled(task.profile):
            return None
        from kuno_protocol.verified import AUDIT_BINDING_OPTION

        binding = task.options.get(AUDIT_BINDING_OPTION)
        return StepRecorder(
            self.retention or shared_retention(),
            task.job_id,
            checkpoint_every=task.profile.verified.retention_checkpoint_every,
            context=context,
            audit_binding=binding if isinstance(binding, str) else None,
        )

    def enhance_prompt(self, task: GenerationTask) -> str:
        """`task.prompt` rewritten by the backend's prompt enhancer for the render `generate(task)` would make. Only
        called when `prompt_enhancement` is set, never for a storyboard; errors must not carry either prompt."""
        raise NotImplementedError(f"{self.name} has no prompt enhancer")

    @abstractmethod
    def generate(self, task: GenerationTask, progress: ProgressFn) -> VideoResult: ...
