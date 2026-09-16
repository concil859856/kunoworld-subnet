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

from kuno_protocol.profiles import InputRole, Mode, ModelProfile, ltx_num_frames, storyboard_frames
from kuno_protocol.receipts import VideoInfo

from ..verified import RetentionStore, context_bytes
from .base import Backend, GenerationTask, ProgressFn, VideoResult
from .media_tools import BackendError, encode_video
from .resident import ModelStore, PipelineResult

# The distilled transformer is trained for these sigmas; passing a step count instead
# silently degrades quality (LTX documents this explicitly). They equal diffusers'
# DISTILLED_SIGMA_VALUES and STAGE_2_DISTILLED_SIGMA_VALUES: the scheduler appends the final 0.0
# itself, so listing it here would add a zero-length step.
DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
SECOND_STAGE_SIGMAS = [0.909375, 0.725, 0.421875]
# The distilled model runs without guidance; diffusers' two-stage example turns every guide off.
DISTILLED_GUIDANCE = {
    "guidance_scale": 1.0, "audio_guidance_scale": 1.0, "stg_scale": 0.0, "audio_stg_scale": 0.0,
    "modality_scale": 1.0, "audio_modality_scale": 1.0,
}
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
        call.update(DISTILLED_GUIDANCE)
        if call["pipeline"] == "text":
            # Stage one at half size, the latents upsampled x2, then these sigmas at full size (runtimes.LtxAdapter).
            call["second_stage_sigmas"] = SECOND_STAGE_SIGMAS
    else:
        call["num_inference_steps"] = FULL_STEPS
        call["guidance_scale"] = 3.0
        call["audio_guidance_scale"] = 7.0
        call["use_cross_timestep"] = True  # as the LTX-2.5-Diffusers card runs transformer_full
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
    storyboards = True

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
        storyboard_renderer: Callable[[Any, ModelProfile], Any] | None = None,
        device_gib: float | None = None,
    ):
        """`hardware_class` turns on verified mode for profiles that pin it (see VERIFIED_MODE.md) and picks
        the weights precision (backends/quantized.py); `model_digest` is the weights identity from the
        owner-signed manifest. `offload` is auto | none | model | group. `storyboard_renderer(loaded, profile)` replaces
        the one storyboards render through (ltx_storyboard.ExtendRenderer on the loaded pipelines) in tests. `device_gib`
        replaces reading the GPU's memory (quantized.probe_device) for the memory plan."""
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
        self.storyboard_renderer = storyboard_renderer
        self.device_gib = device_gib
        self._plans: dict[str, Any] = {}
        self._determinism: dict[str, Any] | None = None

    def memory_plan(self, profile: ModelProfile):
        """The memory plan for this profile on this GPU (None when nothing limits it, or when neither the class nor the
        device gives the VRAM, e.g. without CUDA). Raises PrecisionError when the GPU cannot serve the profile at all."""
        if profile.id not in self._plans:
            from .quantized import host_memory_gib, plan_for_class

            ram = host_memory_gib() if self.host_ram_gib is None else self.host_ram_gib
            self._plans[profile.id] = plan_for_class(
                profile, self.hardware_class, host_ram_gib=ram, mode=self.offload, device_gib=self._device_gib(),
            )
        return self._plans[profile.id]

    def _device_gib(self) -> float | None:
        if self.device_gib is None:
            try:
                from .quantized import probe_device

                self.device_gib = probe_device().total_gib
            except Exception:  # noqa: BLE001 - no torch or no CUDA (CPU tests, the mock network): nothing to plan against
                return None
        return self.device_gib

    def serving_envelope(self, profile: ModelProfile):
        """What the gateway may route here: the profile's limits, or on a class with a memory plan, the longest
        duration the plan fits at each size and frame rate (the same rule `admit` refuses by)."""
        plan = self.memory_plan(profile)
        if plan is None:
            return super().serving_envelope(profile)
        from .quantized import envelope_for_plan

        return envelope_for_plan(plan, profile)

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
        if task.params.mode is Mode.STORYBOARD:
            return self._storyboard(task, progress)
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

    def _storyboard(self, task: GenerationTask, progress: ProgressFn) -> VideoResult:
        """Shot after shot on the loaded pipelines, each joined to the ones before it from their final latents
        (backends/ltx_storyboard.py), into one stitched video. Never verified: no step recorder and no commitment, even
        on a class that runs verified mode for this profile's other jobs."""
        import shutil

        from .ltx_storyboard import render_storyboard

        shots = task.params.shots or []
        tasks = [task.shot_task(index) for index in range(len(shots))]
        calls = [build_call(shot) for shot in tasks]
        # Shots render one at a time, so the longest decides whether this class's memory holds the storyboard: refused
        # before any GPU work, as a single clip of that length would be.
        longest = max(range(len(calls)), key=lambda index: calls[index]["num_frames"])
        self.admit(tasks[longest], calls[longest])
        directory = self.workdir / task.job_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with self.store.acquire(task.profile) as loaded:
                renderer = self._storyboard_renderer(loaded, task.profile)
                data, frames = render_storyboard(
                    renderer, [shot.join for shot in shots], calls, directory, progress,
                    overlap=task.profile.limits.storyboard.overlap_latent_frames, audio=task.params.audio,
                )
            expected = storyboard_frames(task.profile, shots, task.params.fps)
            if frames != expected:
                raise BackendError(f"the stitched video has {frames} frames, the storyboard's params say {expected}")
            fps = float(task.params.fps)
            info = VideoInfo(
                duration_s=round(frames / fps, 3), width=task.width, height=task.height, fps=fps, frames=frames, audio=task.params.audio,
            )
            progress(1.0, "encoded")
            return VideoResult(data=data, info=info)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def _storyboard_renderer(self, loaded: Any, profile: ModelProfile) -> Any:
        if self.storyboard_renderer is not None:
            return self.storyboard_renderer(loaded, profile)
        pipelines = getattr(loaded, "pipelines", None)
        if not isinstance(pipelines, dict):
            raise BackendError("the loaded LTX-2.5 runtime has no diffusers pipelines to render a storyboard with")
        from .ltx_storyboard import ExtendRenderer

        return ExtendRenderer(pipelines, device=getattr(loaded, "device", "cuda"), overlap=profile.limits.storyboard.overlap_latent_frames)
