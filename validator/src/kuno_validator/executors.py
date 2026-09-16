"""GPU step executors for verified LTX-2.5 and MiniMax H3 audits.

NOT RUN ON A GPU. Written against the same documented hook points as the worker
(see kuno_protocol.torch_verified and kuno_worker.backends.verified_gpu). An executor must run
on the hardware class it certifies, with the profile's determinism pins and the same pinned
weights; it only ever compares bit for bit. Before trusting one, compute a golden set with a
worker image on the same class and replay every step of it: any hook that feeds or captures
the wrong tensor shows up as a mismatch on an honest trajectory.

LTX replays step k without re-running steps 1..k-1: it calls the pipeline with the pinned
(pre-shift) schedule [σ_{k-1}, σ_{k-1}, σ_k, …]. The first scheduler step has dt = 0 and its
output is overridden with the committed latents at leaf k-1; the second step is exactly
σ_{k-1} → σ_k, after which the callback interrupts the loop (`pipe._interrupt = True`) and
`output_type="latent"` skips decoding. The schedule the scheduler actually produced is checked
against the transcript's post-shift sigmas before the result counts.

H3's modular loop has no sigmas argument, so its replay skips the "denoiser" and "update"
blocks for i < k-1, injects the committed latents at i = k-1, and stops after that update.

Tolerance mode: `tolerance_classes` lists open-tier miner classes (profiles.json, `comparison:
"tolerance"`) this executor also replays; the auditor then compares within the calibrated distance
instead of bit for bit. A class that runs different weights (fp8-cast, int8) needs its own pinned
digest in `model_digests` under "<profile_id>@<hardware_class>", and the loader must load that recipe.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kuno_protocol.profiles import ModelProfile
from kuno_protocol.schemas import GenerationParams
from kuno_protocol.torch_verified import (
    SchedulerTrap,
    apply_determinism,
    cpu_generator,
    find_denoise_loop,
    loop_block,
    tensor_from_record,
    tensor_record,
)
from kuno_protocol.verified import StepCommitment, StepTranscript, Tensor, expected_layout, f64_value, latent_digest

from .audits import CanaryRecord, TranscriptMismatch

# Must equal kuno_worker.backends.ltx_resident.DISTILLED_SIGMAS (tests/test_verified_audit_flow.py checks).
# The worker's stage-0 sigmas (ltx_resident.DISTILLED_SIGMAS) plus the terminal 0.0 the scheduler appends: a replay is
# checked against what the transcript recorded, which is scheduler.sigmas, not the shorter list passed to the call.
LTX_DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
LTX_FULL_STEPS = 30


class ReplayScheduleError(RuntimeError):
    """The replay schedule did not reproduce the transcript's sigmas: an executor problem, not the miner's."""


class _DiffusersExecutor:
    runtime = ""

    def __init__(
        self,
        loader: Callable[[ModelProfile], Any],
        hardware_class: str,
        model_digests: dict[str, str],
        device: str = "cuda",
        tolerance_classes: tuple[str, ...] | list[str] = (),
    ):
        """`loader(profile)` returns the loaded pipeline (loaded after determinism is applied);
        `model_digests[profile_id]` is the pinned weights identity from the owner-signed manifest."""
        self.loader = loader
        self.hardware_class = hardware_class
        self.model_digests = model_digests
        self.device = device
        self.tolerance_classes = tuple(tolerance_classes)
        self._pipelines: dict[str, Any] = {}
        self._pinned = False

    def pipeline(self, profile: ModelProfile) -> Any:
        if not self._pinned:
            apply_determinism(profile.verified.determinism)
            self._pinned = True
        if profile.id not in self._pipelines:
            self._pipelines[profile.id] = self.loader(profile)
        return self._pipelines[profile.id]

    def candidate_steps(self, commitment: StepCommitment, profile: ModelProfile) -> list[int]:
        verified = profile.verified
        if verified is None or not self.replays(commitment.hardware_class):
            return []
        steps, first = [], 0
        for stage, count in enumerate(verified.stage_steps):
            if stage in verified.replayable_stages:
                steps.extend(range(first + 1, first + count + 1))
            first += count + 1
        return steps

    def replays(self, hardware_class: str) -> bool:
        return hardware_class == self.hardware_class or hardware_class in self.tolerance_classes

    def check_transcript(self, transcript: StepTranscript, canary: CanaryRecord, profile: ModelProfile) -> str | None:
        verified = profile.verified
        if not self.replays(transcript.hardware_class):
            return None  # another class: this executor can't judge it, and candidate_steps is empty
        if [stage.steps for stage in transcript.stages] != verified.stage_steps:
            return f"transcript schedules {[s.steps for s in transcript.stages]} steps per stage, not {verified.stage_steps}"
        pinned = self.model_digests.get(f"{profile.id}@{transcript.hardware_class}") or self.model_digests.get(profile.id)
        if transcript.model_digest != pinned:
            return "transcript model_digest is not the pinned weights"
        pins = verified.determinism.model_dump(mode="json")
        if any(transcript.determinism.get(key) != value for key, value in pins.items()):
            return "transcript determinism settings differ from the profile's pins"
        if transcript.noise != verified.determinism.noise:
            return "transcript noise source is not the pinned one"
        return None

    def initial_state(self, transcript: StepTranscript, canary: CanaryRecord) -> list[Tensor] | None:
        # Recomputing leaf 0 needs the pipeline's own noise ordering for video and audio; enable after Phase 0.
        return None

    def trajectory(self, transcript: StepTranscript, canary: CanaryRecord) -> list[str] | None:
        # Full re-runs on reference hardware go through golden.worker_backend_runner with the reference image.
        return None

    @staticmethod
    def _local(transcript: StepTranscript, step: int) -> tuple[int, int]:
        layout = expected_layout(transcript)
        stage = layout[step].stage
        first = next(slot.index for slot in layout if slot.stage == stage)
        return stage, step - first

    def _check_conditioning(self, transcript: StepTranscript, embeds: Any) -> None:
        if embeds is not None and latent_digest([tensor_record("prompt_embeds", embeds)]) != transcript.conditioning_digest:
            raise TranscriptMismatch("transcript conditioning does not match the canary prompt")


class LtxStepExecutor(_DiffusersExecutor):
    runtime = "diffusers-ltx2/1"

    def pinned_sigmas(self, profile: ModelProfile) -> list[float]:
        if profile.variant == "pro":
            # diffusers' default when only num_inference_steps is given: linspace(1.0, 1/N, N).
            n = LTX_FULL_STEPS
            return [1.0 - i * (1.0 - 1.0 / n) / (n - 1) for i in range(n)]
        return list(LTX_DISTILLED_SIGMAS)

    def execute(self, transcript: StepTranscript, canary: CanaryRecord, step: int, state: list[Tensor]) -> list[Tensor]:
        from kuno_protocol.profiles import load_profiles

        profile = load_profiles()[transcript.profile_id]
        loaded = self.pipeline(profile)
        pipe = loaded["text"] if isinstance(loaded, dict) else loaded
        stage, local = self._local(transcript, step)
        if stage != 0:
            raise ReplayScheduleError("only the first stage is replayable")
        committed = [f64_value(s) for s in transcript.stages[0].sigmas]
        pinned = self.pinned_sigmas(profile)
        schedule = [pinned[local - 1]] + pinned[local - 1 :]

        params = GenerationParams.model_validate(canary.params)
        width, height = profile.size_for(params.resolution, params.aspect_ratio)
        tensors = {spec.name: tensor_from_record((spec, data), self.device) for spec, data in state}
        trap = SchedulerTrap(pipe.scheduler)
        trap.overrides[("video", 0)] = tensors["video"]
        if "audio" in tensors:
            trap.overrides[("audio", 0)] = tensors["audio"]
        captured: dict[str, Any] = {}

        def callback(p, i, timestep, kwargs):
            if i == 0:
                actual = [float(s) for s in p.scheduler.sigmas.tolist()]
                if actual[1 : 2] != committed[local - 1 : local] or actual[2:3] != committed[local : local + 1]:
                    raise ReplayScheduleError("replay schedule does not reproduce the committed sigmas")
                self._check_conditioning(transcript, kwargs.get("prompt_embeds"))
            elif i == 1:
                captured["video"] = kwargs["latents"]
                captured["audio"] = trap.outputs.get("audio")
                p._interrupt = True
            return kwargs

        with trap.installed():
            pipe(
                prompt=canary.prompt, negative_prompt=canary.negative_prompt, width=width, height=height,
                num_frames=profile.num_frames(params.duration_s, params.fps), frame_rate=float(params.fps), sigmas=schedule,
                guidance_scale=1.0 if profile.variant != "pro" else 3.0, generator=cpu_generator(canary.seed), output_type="latent",
                callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
            )
        if "video" not in captured:
            raise ReplayScheduleError("the pipeline finished before the replayed step")
        return [tensor_record(name, value) for name, value in captured.items() if value is not None]


class _StopReplay(Exception):
    pass


class H3StepExecutor(_DiffusersExecutor):
    runtime = "diffusers-modular-h3/1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._armed: dict[str, Any] | None = None

    def pipeline(self, profile: ModelProfile) -> Any:
        pipe = super().pipeline(profile)
        loop = find_denoise_loop(pipe.blocks)
        if loop is not None and "kuno_replay_inject" not in loop.sub_blocks:
            self._install(loop)
        return pipe

    def _install(self, loop: Any) -> None:
        denoiser, update = loop.sub_blocks["denoiser"], loop.sub_blocks["update"]

        def inject(components, block_state, i, t):
            armed = self._armed
            if armed is not None and i == armed["local"] - 1:
                sigmas = [float(s) for s in components.scheduler.sigmas.tolist()]
                if sigmas[armed["local"] - 1 : armed["local"] + 1] != armed["sigmas"]:
                    raise ReplayScheduleError("replay schedule does not reproduce the committed sigmas")
                block_state.latents = armed["video"]
                if "audio" in armed:
                    block_state.audio_latents = armed["audio"]

        def skip_before_target(block):
            def run(components, block_state, i, t=None, **kwargs):
                armed = self._armed
                if armed is not None and i < armed["local"] - 1:
                    return components, block_state
                return block(components, block_state, i, t, **kwargs) if t is not None else block(components, block_state, i, **kwargs)

            return run

        def capture(components, block_state, i, t):
            armed = self._armed
            if armed is not None and i == armed["local"] - 1:
                armed["out"] = {"video": block_state.latents, "audio": getattr(block_state, "audio_latents", None)}
                raise _StopReplay()

        loop.sub_blocks.insert("kuno_replay_inject", loop_block(inject), 0)
        loop.sub_blocks["denoiser"] = skip_before_target(denoiser)
        loop.sub_blocks["update"] = skip_before_target(update)
        loop.sub_blocks.insert("kuno_replay_capture", loop_block(capture), len(loop.sub_blocks))

    def execute(self, transcript: StepTranscript, canary: CanaryRecord, step: int, state: list[Tensor]) -> list[Tensor]:
        from kuno_protocol.profiles import load_profiles

        profile = load_profiles()[transcript.profile_id]
        pipe = self.pipeline(profile)
        stage, local = self._local(transcript, step)
        if stage != 0:
            raise ReplayScheduleError("H3 has a single stage")
        params = GenerationParams.model_validate(canary.params)
        width, height = profile.size_for(params.resolution, params.aspect_ratio)
        committed = [f64_value(s) for s in transcript.stages[0].sigmas]
        tensors = {spec.name: tensor_from_record((spec, data), self.device) for spec, data in state}
        self._armed = {"local": local, "sigmas": committed[local - 1 : local + 1], **tensors}
        try:
            pipe(
                prompt=canary.prompt, width=width, height=height, num_frames=profile.num_frames(params.duration_s, params.fps),
                num_inference_steps=profile.steps, generator=cpu_generator(canary.seed), output=["videos"],
            )
            raise ReplayScheduleError("the pipeline finished before the replayed step")
        except _StopReplay:
            out = self._armed["out"]
        finally:
            self._armed = None
        return [tensor_record(name, value) for name, value in out.items() if value is not None]
