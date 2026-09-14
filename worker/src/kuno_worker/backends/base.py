from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from kuno_protocol.media import EXTENSIONS
from kuno_protocol.profiles import InputRole, ModelProfile
from kuno_protocol.receipts import VideoInfo
from kuno_protocol.schemas import GenerationParams, InputRef
from kuno_protocol.verified import StepCommitment, Tensor

from ..verified import OpeningsHandle, RetentionStore, StepRecorder, shared_retention

ProgressFn = Callable[[float, str], None]


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

    def first(self, role: InputRole) -> InputFile | None:
        return next((i for i in self.inputs if i.ref.role is role), None)

    def all(self, role: InputRole) -> list[InputFile]:
        return [i for i in self.inputs if i.ref.role is role]

    @property
    def num_frames(self) -> int:
        return self.profile.num_frames(self.params.duration_s, self.params.fps)


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
        """A recorder for this job when verified mode applies, else None."""
        if not self.verified_enabled(task.profile):
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

    @abstractmethod
    def generate(self, task: GenerationTask, progress: ProgressFn) -> VideoResult: ...
