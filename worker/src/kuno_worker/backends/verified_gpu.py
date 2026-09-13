"""Verified mode for the real LTX-2.5 and MiniMax H3 pipelines: report every step's latent
state to the step sink while the resident pipeline runs.

NOT RUN ON A GPU. Written against the documented hook points; Phase 0 validates it by
computing golden sets on each hardware class and checking a fresh image against them.

LTX-2.5 (diffusers `LTX2Pipeline` / `LTX2ConditionPipeline`)
  `callback_on_step_end(pipe, step, timestep, callback_kwargs) -> callback_kwargs` with
  `callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"]`
  (https://huggingface.co/docs/diffusers/using-diffusers/callback). The loop's latents are
  packed [B, tokens, C] and leaves commit to exactly that representation. The audio latents
  are not callback inputs (`_callback_tensor_inputs` = latents, prompt_embeds,
  negative_prompt_embeds; audio is stepped by a deepcopy of the scheduler), so they and each
  stage's initial latent come from `SchedulerTrap`
  (https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx2/pipeline_ltx2.py).
  User sigmas are time-shifted inside `FlowMatchEulerDiscreteScheduler.set_timesteps`, so the
  transcript records `pipe.scheduler.sigmas` as actually used.

MiniMax H3 (Modular Diffusers `MiniMaxH3ModularPipeline`)
  The denoise loop (`LoopSequentialPipelineBlocks`, sub-blocks ["denoiser", "update"]) has no
  callback; blocks are inserted into its `sub_blocks` InsertableDict: one before "denoiser"
  (records the initial latent at i == 0) and one after "update" (records each step)
  (https://huggingface.co/docs/diffusers/main/en/modular_diffusers/loop_sequential_pipeline_blocks,
  https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3). Documented noise
  shapes: latents (1, 24, F, H, W), audio_latents (2, 32, N).

Not hooked: SGLang-Diffusion exposes no per-step callback, but `SamplingParams.return_trajectory_latents`
returns every step's latents (python/sglang/multimodal_gen/configs/sample/sampling_params.py), which
could feed commitments once the server API exposes it. LightX2V's runner loop
(lightx2v/models/runners/default_runner.py: step_pre → infer → step_post) holds the state in
`scheduler.latents` right after `step_post()`. Verified H3 profiles use the diffusers modular path.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from kuno_protocol.canonical import canonical_json, sha256_hex
from kuno_protocol.torch_verified import SchedulerTrap, find_denoise_loop, loop_block, tensor_record
from kuno_protocol.verified import StageTranscript, StepTranscript, Tensor, VerifiedModeError, f64_hex, latent_digest, tensor_from_array

from .base import GenerationTask, StepSink


def _records(state: dict[str, Any]) -> list[Tensor]:
    """torch tensors on a GPU image; numpy arrays from stub pipelines in tests."""
    out = []
    for name, value in state.items():
        if value is None:
            continue
        out.append(tensor_record(name, value) if type(value).__module__.startswith("torch") else tensor_from_array(name, value))
    return out


class TrajectoryTap:
    """Turns per-step observations from a running pipeline into leaves, stage by stage."""

    def __init__(self, sink: StepSink):
        self.sink = sink
        self.index = 0
        self.stages: list[dict[str, Any]] = []
        self.conditioning_digest: str | None = None

    def begin_stage(self, sigmas: list[float], state: dict[str, Any]) -> None:
        tensors = _records(state)
        self.stages.append({"sigmas": [float(s) for s in sigmas], "tensors": [spec for spec, _ in tensors], "steps": 0})
        self.sink.report(self.index, len(self.stages) - 1, "init", float(sigmas[0]), tensors)
        self.index += 1

    def end_step(self, step: int, state: dict[str, Any]) -> None:
        if not self.stages:
            raise VerifiedModeError("a step was reported before its stage began")
        stage = self.stages[-1]
        tensors = _records(state)
        if [spec for spec, _ in tensors] != stage["tensors"]:
            raise VerifiedModeError("the latent layout changed inside a stage")
        if step != stage["steps"]:
            raise VerifiedModeError(f"step {step} reported after {stage['steps']} steps of this stage")
        stage["steps"] += 1
        self.sink.report(self.index, len(self.stages) - 1, "denoise", stage["sigmas"][step + 1], tensors)
        self.index += 1

    def note_conditioning(self, **tensors: Any) -> None:
        if self.conditioning_digest is None and tensors:
            self.conditioning_digest = latent_digest(_records(tensors))

    def transcript(
        self, task: GenerationTask, *, runtime: str, model_digest: str, hardware_class: str, scheduler: str, determinism: dict[str, Any]
    ) -> StepTranscript:
        verified = task.profile.verified
        return StepTranscript(
            job_id=task.job_id,
            params_digest=sha256_hex(canonical_json(task.params.model_dump(mode="json"))),
            profile_id=task.profile.id,
            family=task.profile.family,
            runtime=runtime,
            model_digest=model_digest,
            hardware_class=hardware_class,
            seed=task.seed,
            noise=verified.determinism.noise if verified else "torch-cpu-generator",
            conditioning_digest=self.conditioning_digest or "0" * 64,
            stages=[
                StageTranscript(
                    name=f"stage{i}", scheduler=scheduler, tensors=stage["tensors"],
                    sigmas=[f64_hex(s) for s in stage["sigmas"][: stage["steps"] + 1]],
                )
                for i, stage in enumerate(self.stages)
            ],
            determinism=determinism,
        )


def unpinned_model_digest(checkpoint: str) -> str:
    """Stand-in weights identity when the backend was given none. Validators pin the real one
    from the owner-signed manifest, so trajectories carrying this fail their transcript check."""
    return sha256_hex(b"kuno/v1/unpinned-model\n" + checkpoint.encode())


def finish_trajectory(task: GenerationTask, recorder, tap: TrajectoryTap | None, *, model_digest: str | None, hardware_class: str, determinism: dict[str, Any] | None):
    """(commitment, openings) for a verified run, or (None, None) when verified mode was off."""
    if recorder is None or tap is None:
        return None, None
    verified = task.profile.verified
    transcript = tap.transcript(
        task, runtime=verified.runtime, model_digest=model_digest or unpinned_model_digest(task.profile.checkpoint),
        hardware_class=hardware_class, scheduler=verified.scheduler, determinism=determinism or {},
    )
    return recorder.finish(transcript)


# ------------------------------------------------------------------ LTX-2.5


def ltx_step_callback(tap: TrajectoryTap, trap: SchedulerTrap) -> Callable:
    def callback(pipe, step: int, timestep, callback_kwargs: dict) -> dict:
        if step == 0:
            tap.begin_stage(pipe.scheduler.sigmas.tolist(), {"video": trap.inputs.get("video"), "audio": trap.inputs.get("audio")})
            if callback_kwargs.get("prompt_embeds") is not None:
                tap.note_conditioning(prompt_embeds=callback_kwargs["prompt_embeds"])
        tap.end_step(step, {"video": callback_kwargs["latents"], "audio": trap.outputs.get("audio")})
        return callback_kwargs

    return callback


@contextmanager
def ltx_verified(pipeline: Any, call: dict[str, Any], tap: TrajectoryTap):
    """Adds the step callback to `call` and traps the scheduler for the duration of the pipeline call."""
    trap = SchedulerTrap(pipeline.scheduler)
    call["callback_on_step_end"] = ltx_step_callback(tap, trap)
    call["callback_on_step_end_tensor_inputs"] = ["latents", "prompt_embeds"]
    with trap.installed():
        yield


# ------------------------------------------------------------------ MiniMax H3


def install_h3_commit_blocks(pipeline: Any, current_tap: Callable[[], TrajectoryTap | None]) -> None:
    """Inserts the commit blocks into H3's denoise loop once; they report to whichever tap is active."""
    loop = find_denoise_loop(pipeline.blocks)
    if loop is None:
        raise VerifiedModeError("MiniMax H3 pipeline has no ['denoiser', 'update'] loop to hook")
    if "kuno_commit_init" in loop.sub_blocks:
        return

    def before(components, block_state, i, t):
        tap = current_tap()
        if tap is not None and i == 0:
            tap.begin_stage(
                components.scheduler.sigmas.tolist(),
                {"video": block_state.latents, "audio": getattr(block_state, "audio_latents", None)},
            )
            embeds = getattr(block_state, "prompt_embeds", None)
            if embeds is not None:
                tap.note_conditioning(prompt_embeds=embeds)

    def after(components, block_state, i, t):
        tap = current_tap()
        if tap is not None:
            tap.end_step(i, {"video": block_state.latents, "audio": getattr(block_state, "audio_latents", None)})

    loop.sub_blocks.insert("kuno_commit_init", loop_block(before), 0)
    loop.sub_blocks.insert("kuno_commit_step", loop_block(after), len(loop.sub_blocks))
