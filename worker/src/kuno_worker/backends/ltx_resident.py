"""LTX-2.5 kept resident through the diffusers pipelines.

The official `ltx_pipelines` CLI (backends/ltx.py) reloads ~66 GB per job; it stays as the
`cold` backend for first-run validation against the model's own documentation. For serving,
the diffusers classes are designed to be called repeatedly on a loaded pipeline.

Everything except the loader is plain data, so each mode's call is tested without a GPU.
The loader itself (runtimes.py) is the only part that needs hardware to verify.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable

from kuno_protocol.plans import PLAN_SAMPLING
from kuno_protocol.profiles import InputRole, Mode, ModelProfile, ltx_num_frames, storyboard_frames
from kuno_protocol.receipts import VideoInfo

from ..verified import RetentionStore, context_bytes
from .base import Backend, GenerationTask, PlanText, ProgressFn, VideoResult
from .media_tools import BackendError, encode_video
from .resident import ModelStore, PipelineResult

log = logging.getLogger("kuno.worker.ltx")

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
# The keys of a `build_call` output that are the worker's, not diffusers': runtimes.LtxAdapter and
# ltx_pinning.PinnedRenderer consume them. Every other key must be a keyword the diffusers pipeline's __call__ accepts
# (tests/test_ltx_diffusers_signatures.py checks every call build_call can make against diffusers 0.40's signatures).
WORKER_KEYS = ("pipeline", "seed", "conditions", "generate_audio", "second_stage_sigmas", "kuno_trajectory_tap", "edit", "video_decoder")
# Modes that hold tokens encoded from the customer's own media while the rest is generated (backends/ltx_edit.py).
EDIT_MODES = (Mode.AUDIO_TO_VIDEO, Mode.RETAKE)
# `video_decoder` of a call whose latents LTX-2.5's diffusion decoder turns into frames (ltx-2.5-4k): the loaded
# LTX2VideoDiffusionDecodePipeline, not the video VAE the pipeline would decode with (runtimes.LtxAdapter._diffusion_decode).
DIFFUSION_DECODER = "diffusion"


def pipeline_kind(profile: ModelProfile, mode: Mode) -> str:
    """Which diffusers pipeline class the loader should hand us. Audio-to-video and retake render through the condition
    pipeline's pinning subclass (ltx_pinning.LTX2PinnedPipeline): no diffusers LTX-2 pipeline takes a sound track or a
    source clip to keep. ltx-2.5-4k renders through the same classes as ltx-2.5-fast; only its decoder differs."""
    if mode in EDIT_MODES:
        return "condition"
    if mode in (Mode.IMAGE_TO_VIDEO, Mode.LAST_FRAME, Mode.FIRST_LAST_FRAME, Mode.KEYFRAMES):
        return "condition"
    return "text"


def _seconds(value: Any, what: str) -> float | None:
    """A time from the sealed options: None, or a finite number of seconds. Options are the customer's JSON, so anything
    else fails the job with the option's name, never its value."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BackendError(f"{what} must be a number of seconds")
    return float(value)


def _flag(options: dict[str, Any], key: str, default: bool) -> bool:
    value = options.get(key, default)
    if not isinstance(value, bool):
        raise BackendError(f"options.{key} must be true or false")
    return value


def build_call(task: GenerationTask) -> dict[str, Any]:
    """The keyword arguments for one generation. Paths, not loaded media: the loader reads them.

    Audio-to-video and retake carry `edit`, what ltx_edit.render_edit holds from the customer's media:
      audio_to_video  {"mode", "audio_path", "start_s", "end_s"}: the sound from the source_audio input's start_s (0 by
                      default) to its end_s (the clip's end by default), held under the whole render
      retake          {"mode", "video_path", "start_s", "end_s", "regenerate_video", "regenerate_audio"}: the source_video
                      clip from its first frame, regenerated in [start_s, end_s) (options.retake's, else the input's; the
                      whole clip by default); `regenerate_video` (default true) and `regenerate_audio` (default: whether
                      the job has audio) come from the options, and a modality not regenerated is held entirely
    Both render `num_frames` from `duration_s`, like every other mode, so the receipt's length is the job's."""
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
        # Never inside the render: diffusers would rewrite the prompt between the worker's safety check and the text
        # encoder. The worker enhances as a step of its own and checks the result (LtxResidentBackend.enhance_prompt),
        # so `prompt` here is already the text to condition on. Explicit, so no diffusers default can turn it back on.
        "enable_prompt_enhancement": False,
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
    if profile.variant == "dfr":
        # ltx-2.5-4k: ltx-2.5-fast's render (text: 8 sigmas at half size, the latents upsampled x2, 3 at full size; frames
        # and keyframes: 8 at full size) at 1440p or 2160p, decoded by LTX-2.5's diffusion decoder. That decoder has the
        # VAE's 8x temporal ratio and interpolates nothing, so 48 and 50 fps render every frame at that rate, as they do on
        # ltx-2.5-fast; nothing renders at half rate.
        call["video_decoder"] = DIFFUSION_DECODER

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
            window = task.options.get("retake")
            if window is None:
                window = {}
            if not isinstance(window, dict):
                raise BackendError("options.retake must be an object with start_s and end_s")
            start = _seconds(window["start_s"], "options.retake.start_s") if "start_s" in window else item.ref.start_s
            end = _seconds(window["end_s"], "options.retake.end_s") if "end_s" in window else item.ref.end_s
            call["edit"] = {
                "mode": Mode.RETAKE.value, "video_path": item.path, "start_s": start or 0.0, "end_s": end,
                "regenerate_video": _flag(task.options, "regenerate_video", True),
                "regenerate_audio": _flag(task.options, "regenerate_audio", params.audio),
            }
            continue
        elif role is InputRole.SOURCE_AUDIO:
            call["edit"] = {"mode": Mode.AUDIO_TO_VIDEO.value, "audio_path": item.path, "start_s": item.ref.start_s or 0.0, "end_s": item.ref.end_s}
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
    prompt_enhancement = True
    plans = True

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
        the one storyboards render through (ltx_storyboard.ExtendRenderer on the loaded pipelines) in tests; audio-to-video
        and retake render through the loaded runtime's own (runtimes.LtxAdapter.pinned_renderer). `device_gib` replaces
        reading the GPU's memory (quantized.probe_device) for the memory plan."""
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
        """Refuses, before any GPU work, a request larger than this class's memory plan allows. Audio-to-video and retake
        also encode the customer's media on the GPU first (quantized.source_encode_gib), so on a card whose plan holds every
        clip the profile allows, they are checked against the whole-GPU plan that `memory_plan` leaves out."""
        plan = self.memory_plan(task.profile)
        if plan is None and call.get("edit") is not None:
            plan = self._edit_plan(task.profile)
        if plan is not None:
            from .quantized import admit

            admit(plan, task.profile, call, task.width, task.height, task.params.fps)

    def _edit_plan(self, profile: ModelProfile):
        key = f"{profile.id}#edit"
        if key not in self._plans:
            from .quantized import host_memory_gib, plan_for_class

            ram = host_memory_gib() if self.host_ram_gib is None else self.host_ram_gib
            self._plans[key] = plan_for_class(
                profile, self.hardware_class, host_ram_gib=ram, mode=self.offload, device_gib=self._device_gib(), keep_covering=True,
            )
        return self._plans[key]

    def step_recorder(self, task: GenerationTask, context: bytes = b""):
        """None for audio-to-video and retake, as for storyboards: no step commitment on any class. Their held tokens are
        encoded from the customer's own media, which the transcript's conditioning digest (the prompt embeddings) doesn't
        describe and a validator can't replay: validators step-audit only their text-to-video canaries and Standard jobs
        without inputs (kuno_validator.audits.standard_record), so a commitment here would never be opened."""
        if task.params.mode in EDIT_MODES:
            return None
        return super().step_recorder(task, context)

    def _pin(self, profile: ModelProfile) -> None:
        """Determinism must be pinned before weights touch the GPU (cuBLAS reads its workspace config once)."""
        if self.verified_enabled(profile) and self._determinism is None:
            from kuno_protocol.torch_verified import apply_determinism

            self._determinism = apply_determinism(profile.verified.determinism)

    def warm(self, profile: ModelProfile) -> None:
        self._pin(profile)
        self.memory_plan(profile)
        self.store.warm(profile)

    def enhance_prompt(self, task: GenerationTask) -> str:
        """The loaded pipeline's prompt enhancer (Gemma-4-E2B, `prompt_enhancer/` with `processor/`'s chat template) run
        on `task.prompt` for the call `generate(task)` will make, as diffusers would have run it inside that call
        (runtimes.LtxAdapter.enhance_prompt). Memory is admitted first, so a request this class refuses costs no
        enhancement."""
        if task.params.mode is Mode.STORYBOARD:
            raise BackendError("storyboard shots are rendered as written, never enhanced")
        import shutil

        directory = self.workdir / task.job_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            for item in task.inputs:  # an image-to-video call enhances with its first condition's image
                item.save(directory)
            call = build_call(task)
            self.admit(task, call)
            self._pin(task.profile)  # before the first load, as generate does, if no warm-up loaded the weights
            with self.store.acquire(task.profile) as loaded:
                enhance = getattr(loaded, "enhance_prompt", None)
                if not callable(enhance):
                    raise BackendError("the loaded LTX-2.5 runtime cannot enhance prompts")
                return enhance(call)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def write_plan(self, task: GenerationTask, messages: list[dict[str, str]], *, seed: int, max_new_tokens: int) -> PlanText:
        """The reply of the bundled prompt enhancer (Gemma-4-E2B) on whichever LTX-2.5 pipeline is loaded: every recipe
        includes the same enhancer, so a plan never forces a reload (ModelStore.acquire_loaded). The enhancer is on the
        GPU only while it writes (runtimes.LtxAdapter.enhancer), under the store's lock, so a plan never overlaps a
        render. Measured 2026-09-16 on an RTX PRO 6000: 49-51 tokens/s, 7.6-18.6 s per plan, 0.12 GiB extra."""
        self._pin(task.profile)  # before the first load, as generate does, if no warm-up loaded the weights
        with self.store.acquire_loaded(task.profile) as loaded:
            write = getattr(loaded, "write_text", None)
            if not callable(write):
                raise BackendError("the loaded LTX-2.5 runtime cannot write text")
            text, tokens = write(messages, seed=seed, max_new_tokens=max_new_tokens, **PLAN_SAMPLING)
            recipe = getattr(getattr(loaded, "load_plan", None), "recipe", None)
        if recipe is None:
            from .quantized import resolve_recipe

            recipe, _ = resolve_recipe(task.profile, self.hardware_class)
        component = task.profile.limits.plan.planner if task.profile.limits.plan is not None else "prompt_enhancer"
        return PlanText(text=text, output_tokens=int(tokens), planner=f"{recipe.id}:{component}")

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
            # `task.prompt` is the text the model conditions on: the enhanced prompt when the worker enhanced one
            # (worker.Worker._enhance), since the call no longer enhances. So the retained replay context and the
            # transcript's conditioning digest (prompt_embeds, verified_gpu.ltx_step_callback) describe the same text.
            # Validators replay only their canaries and option-free Standard jobs (kuno_validator.audits.standard_record),
            # and canaries never send enhance_prompt, so every replayed job's conditioning is the prompt the validator
            # holds. A canary that asked for enhancement would fail the conditioning check against an executor that
            # encodes the raw prompt, so one may only be sent once the executor enhances the same way first.
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
            edit = raw.get("edit") if isinstance(raw, dict) else None
            if isinstance(edit, dict):  # counts and timings only
                log.info(
                    "%s: held %d of %d latent frames and %d of %d audio latents, pins exact %s, %s",
                    edit.get("mode"), edit.get("held_latent_frames", 0), edit.get("latent_frames", 0), edit.get("held_audio_latents", 0),
                    edit.get("audio_latents", 0), edit.get("pins_exact"), edit.get("timings"),
                )
            if call.get("video_decoder") is not None and isinstance(raw, dict):  # seconds (and GiB) per phase only
                log.info("diffusion decode of %dx%d: %s %s", task.width, task.height, raw.get("timings"), raw.get("memory_gib") or "")
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
