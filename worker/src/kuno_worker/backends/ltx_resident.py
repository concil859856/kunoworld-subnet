"""LTX-2.5 kept resident through the diffusers pipelines.

The official `ltx_pipelines` CLI (backends/ltx.py) reloads ~66 GB per job; it stays as the
`cold` backend for first-run validation against the model's own documentation. For serving,
the diffusers classes are designed to be called repeatedly on a loaded pipeline.

Everything except the loader is plain data, so each mode's call is tested without a GPU.
The loader itself (runtimes.py) is the only part that needs hardware to verify.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from kuno_protocol.profiles import InputRole, Mode, ModelProfile, ltx_num_frames
from kuno_protocol.receipts import VideoInfo

from ..verified import RetentionStore, context_bytes
from .base import Backend, GenerationTask, ProgressFn, VideoResult
from .media_tools import BackendError, encode_video
from .resident import ModelStore, PipelineResult

# The distilled transformer is trained for these sigmas; passing a step count instead
# silently degrades quality (LTX documents this explicitly).
DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
SECOND_STAGE_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]
FULL_STEPS = 30


def pipeline_kind(profile: ModelProfile, mode: Mode) -> str:
    """Which diffusers pipeline class the loader should hand us."""
    if mode is Mode.RETAKE:
        return "condition"
    if mode is Mode.AUDIO_TO_VIDEO:
        return "audio"
    if profile.variant == "dfr":
        return "dfr"
    if mode in (Mode.IMAGE_TO_VIDEO, Mode.LAST_FRAME, Mode.FIRST_LAST_FRAME, Mode.KEYFRAMES):
        return "condition"
    return "text"


def build_call(task: GenerationTask) -> dict[str, Any]:
    """The keyword arguments for one generation. Paths, not loaded media: the loader reads them."""
    profile, params = task.profile, task.params
    distilled = profile.variant != "pro"
    frames = ltx_num_frames(params.duration_s, params.fps)
    call: dict[str, Any] = {
        "pipeline": pipeline_kind(profile, params.mode),
        "prompt": task.prompt,
        "width": task.width,
        "height": task.height,
        "num_frames": frames,
        "frame_rate": float(params.fps),
        "seed": task.seed,
        "generate_audio": params.audio,
    }
    if distilled:
        call["sigmas"] = DISTILLED_SIGMAS
        call["second_stage_sigmas"] = SECOND_STAGE_SIGMAS
        call["guidance_scale"] = 1.0
    else:
        call["num_inference_steps"] = FULL_STEPS
        call["guidance_scale"] = 3.0
        call["audio_guidance_scale"] = 7.0
        if task.negative_prompt:
            call["negative_prompt"] = task.negative_prompt
    if profile.limits.prompt_enhancer and task.options.get("enhance_prompt"):
        call["enable_prompt_enhancement"] = True
    if profile.variant == "dfr":
        call["spatial_upscalings"] = 1
        call["temporal_upscalings"] = 1 if params.fps >= 48 else 0
        if call["temporal_upscalings"]:
            # Rendered at half rate, then interpolated to the requested playback rate.
            call["frame_rate"] = params.fps / 2
            call["num_frames"] = ltx_num_frames(params.duration_s, params.fps // 2)

    conditions = []
    last_index = call["num_frames"] - 1
    for item in task.inputs:
        role = item.ref.role
        if role is InputRole.FIRST_FRAME:
            index = 0
        elif role is InputRole.LAST_FRAME:
            index = last_index
        elif role is InputRole.KEYFRAME:
            index = min(max(round((item.ref.time_s or 0.0) * call["frame_rate"]), 0), last_index)
        elif role is InputRole.SOURCE_VIDEO:
            window = task.options.get("retake", {})
            call["video_path"] = item.path
            call["start_time"] = float(window.get("start_s", item.ref.start_s or 0.0))
            call["end_time"] = float(window.get("end_s", item.ref.end_s or params.duration_s))
            call["regenerate_video"] = bool(task.options.get("regenerate_video", True))
            call["regenerate_audio"] = bool(task.options.get("regenerate_audio", params.audio))
            continue
        elif role is InputRole.SOURCE_AUDIO:
            call["audio_path"] = item.path
            call["audio_start_time"] = float(item.ref.start_s or 0.0)
            call["audio_max_duration"] = params.duration_s
            call.pop("num_frames", None)  # mutually exclusive with audio_max_duration
            continue
        else:
            continue
        conditions.append({"path": item.path, "index": index, "strength": float(item.ref.strength or 1.0)})
    if conditions:
        call["conditions"] = conditions
    return call


class LtxResidentBackend(Backend):
    name = "ltx-2.5/resident"

    def __init__(
        self,
        models_dir: Path | None,
        workdir: Path,
        loader: Callable[[ModelProfile], Any] | None = None,
        capacity: int = 1,
        hardware_class: str | None = None,
        retention: RetentionStore | None = None,
        model_digest: str | None = None,
        offload: str = "auto",
        weights_verify: str = "full",
        allow_unpinned_weights: bool = False,
        host_ram_gib: float | None = None,
    ):
        """`hardware_class` turns on verified mode for profiles that pin it (see VERIFIED_MODE.md) and picks
        the weights precision (backends/quantized.py); `model_digest` is the weights identity from the
        owner-signed manifest. `offload` is auto | none | model | group."""
        if loader is None:
            if models_dir is None:
                raise ValueError("KUNO_LTX_MODELS_DIR must point at the LTX-2.5 weights")
            from .runtimes import ltx_loader

            loader = ltx_loader(
                Path(models_dir), hardware_class=hardware_class, model_digest=model_digest, offload=offload,
                weights_verify=weights_verify, allow_unpinned_weights=allow_unpinned_weights, host_ram_gib=host_ram_gib,
            )
        self.store = ModelStore(loader, capacity=capacity)
        self.workdir = Path(workdir)
        self.hardware_class = hardware_class
        self.retention = retention
        self.model_digest = model_digest
        self.offload = offload
        self.host_ram_gib = host_ram_gib
        self._plans: dict[str, Any] = {}
        self._determinism: dict[str, Any] | None = None

    def memory_plan(self, profile: ModelProfile):
        """The class's memory plan for this profile (None for classes that declare no VRAM). Raises
        PrecisionError when the class cannot serve the profile at all."""
        if profile.id not in self._plans:
            from .quantized import host_memory_gib, plan_for_class

            ram = host_memory_gib() if self.host_ram_gib is None else self.host_ram_gib
            self._plans[profile.id] = plan_for_class(profile, self.hardware_class, host_ram_gib=ram, mode=self.offload)
        return self._plans[profile.id]

    def admit(self, task: GenerationTask, call: dict[str, Any]) -> None:
        """Refuses, before any GPU work, a request larger than this class's memory plan allows."""
        plan = self.memory_plan(task.profile)
        if plan is not None:
            from .quantized import admit

            admit(plan, task.profile, call, task.width, task.height, task.params.fps)

    def _pin(self, profile: ModelProfile) -> None:
        """Determinism must be pinned before weights touch the GPU (cuBLAS reads its workspace config once)."""
        if self.verified_enabled(profile) and self._determinism is None:
            from kuno_protocol.torch_verified import apply_determinism

            self._determinism = apply_determinism(profile.verified.determinism)

    def warm(self, profile: ModelProfile) -> None:
        self._pin(profile)
        self.memory_plan(profile)
        self.store.warm(profile)

    def generate(self, task: GenerationTask, progress: ProgressFn) -> VideoResult:
        directory = self.workdir / task.job_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            for item in task.inputs:
                item.save(directory)
            call = build_call(task)
            self.admit(task, call)
            recorder = self.step_recorder(task, context_bytes(prompt=task.prompt, negative_prompt=task.negative_prompt))
            tap = None
            if recorder is not None:
                from .verified_gpu import TrajectoryTap

                self._pin(task.profile)
                tap = call["kuno_trajectory_tap"] = TrajectoryTap(recorder)
            progress(0.05, "denoising")
            try:
                with self.store.acquire(task.profile) as pipeline:
                    raw = pipeline(**call)
                if recorder is not None:
                    from .verified_gpu import finish_trajectory

                    commitment, openings = finish_trajectory(
                        task, recorder, tap, model_digest=self.model_digest, hardware_class=self.hardware_class, determinism=self._determinism
                    )
                else:
                    commitment, openings = None, None
            except BaseException:
                if recorder is not None:
                    recorder.abort()
                raise
            result = PipelineResult.from_pipeline(raw)
            if not len(result.frames):
                raise BackendError("pipeline returned no frames")
            progress(0.9, "encoding")
            fps = float(task.params.fps)
            audio = result.audio if task.params.audio else None
            data = encode_video(result.frames, fps, audio, result.sample_rate)
            info = VideoInfo(
                duration_s=round(len(result.frames) / fps, 3),
                width=task.width,
                height=task.height,
                fps=fps,
                frames=len(result.frames),
                audio=task.params.audio and result.audio is not None,
            )
            progress(1.0, "encoded")
            return VideoResult(data=data, info=info, step_commitment=commitment, openings=openings)
        finally:
            import shutil

            shutil.rmtree(directory, ignore_errors=True)
