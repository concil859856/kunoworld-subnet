"""MiniMax H3 kept resident through the diffusers modular pipeline.

The SGLang server (backends/h3.py) is already resident and is the officially documented
serving path; use it when it is running. This backend covers the case SGLang does not:
the LightX2V Turbo LoRA, whose only documented entry point reloads ~124 GB per job.

As everywhere else, the call is plain data and tested without a GPU; only the loader in
runtimes.py needs hardware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from kuno_protocol.profiles import InputRole, Mode, ModelProfile, h3_num_frames
from kuno_protocol.receipts import VideoInfo

from .base import Backend, GenerationTask, ProgressFn, VideoResult
from .media_tools import BackendError, encode_video
from .resident import ModelStore, PipelineResult

FPS = 24
REFERENCE_TYPES = {
    InputRole.REFERENCE_IMAGE: "image",
    InputRole.FIRST_FRAME: "image",
    InputRole.REFERENCE_VIDEO: "video",
    InputRole.SOURCE_VIDEO: "video_audio",
    InputRole.REFERENCE_AUDIO: "audio",
    InputRole.SOURCE_AUDIO: "audio",
}
# LightX2V's 768p distilled LoRA is trained for these shifts.
TURBO_SHIFTS = {"video_shift": 6.0, "audio_shift": 3.0}
FULL_SHIFTS = {"video_shift": 12.0, "audio_shift": 3.0}


def build_call(task: GenerationTask) -> dict[str, Any]:
    profile, params = task.profile, task.params
    turbo = profile.runtime == "lightx2v"
    call: dict[str, Any] = {
        "prompt": task.prompt,
        "width": task.width,
        "height": task.height,
        "aspect_ratio": params.aspect_ratio,
        "num_frames": h3_num_frames(params.duration_s),
        "num_inference_steps": profile.steps,
        "seed": task.seed,
        **(TURBO_SHIFTS if turbo else FULL_SHIFTS),
    }
    if params.mode is Mode.REFERENCE_TO_VIDEO or profile.id == "h3-reference":
        # Order matters: it sets the <Picture n> / <Video n> / <Audio n> labels in the prompt.
        references = []
        for item in task.inputs:
            kind = REFERENCE_TYPES[item.ref.role]
            if kind == "video_audio" and not task.options.get("keep_source_audio", True):
                kind = "video"
            entry: dict[str, Any] = {"type": kind, "path": item.path}
            if item.ref.start_s is not None:
                entry["start_time_seconds"] = item.ref.start_s
            references.append(entry)
        call["references"] = references
        return call

    for role, key in ((InputRole.FIRST_FRAME, "image"), (InputRole.LAST_FRAME, "last_image")):
        item = task.first(role)
        if item is not None:
            call[key] = item.path
    return call


class H3ResidentBackend(Backend):
    name = "minimax-h3/resident"

    def __init__(
        self,
        workdir: Path,
        model_id: str = "MiniMaxAI/MiniMax-H3",
        loader: Callable[[ModelProfile], Any] | None = None,
        turbo_lora: str | None = None,
        capacity: int = 1,
    ):
        if loader is None:
            from .runtimes import h3_loader

            loader = h3_loader(model_id, turbo_lora=turbo_lora)
        self.store = ModelStore(loader, capacity=capacity)
        self.workdir = Path(workdir)

    def warm(self, profile: ModelProfile) -> None:
        self.store.warm(profile)

    def generate(self, task: GenerationTask, progress: ProgressFn) -> VideoResult:
        directory = self.workdir / task.job_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            for item in task.inputs:
                item.save(directory)
            call = build_call(task)
            progress(0.05, "denoising")
            with self.store.acquire(task.profile) as pipeline:
                raw = pipeline(**call)
            result = PipelineResult.from_pipeline(raw)
            if not len(result.frames):
                raise BackendError("pipeline returned no frames")
            progress(0.9, "encoding")
            # H3 always renders audio; drop the track when the customer asked for silence.
            audio = result.audio if task.params.audio else None
            data = encode_video(result.frames, FPS, audio, result.sample_rate)
            info = VideoInfo(
                duration_s=round(len(result.frames) / FPS, 3),
                width=task.width,
                height=task.height,
                fps=FPS,
                frames=len(result.frames),
                audio=task.params.audio and result.audio is not None,
            )
            progress(1.0, "encoded")
            return VideoResult(data=data, info=info)
        finally:
            import shutil

            shutil.rmtree(directory, ignore_errors=True)
