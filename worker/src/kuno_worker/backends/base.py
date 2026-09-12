from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from kuno_protocol.media import EXTENSIONS
from kuno_protocol.profiles import InputRole, ModelProfile
from kuno_protocol.receipts import VideoInfo
from kuno_protocol.schemas import GenerationParams, InputRef

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


class Backend(ABC):
    name: str = "backend"

    def warm(self, profile: ModelProfile) -> None:
        """Load weights ahead of the first job. TEE model loads are slow; do it once."""

    @abstractmethod
    def generate(self, task: GenerationTask, progress: ProgressFn) -> VideoResult: ...
