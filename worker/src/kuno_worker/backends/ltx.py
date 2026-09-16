"""LTX-2.5 through the official `ltx_pipelines` modules, inside the confidential VM.

Pipelines by profile and mode:
  ltx-2.5-fast  distilled (8+3 steps)          text, first/last frame, keyframes; retake
  ltx-2.5-pro   ti2vid_two_stages (dev, 30+3)  text, frames; keyframe_interpolation for
                                               first+last/keyframes; a2vid_two_stage for audio
  ltx-2.5-4k    dfr_pipeline                   text, first frame, keyframes; x2 temporal for 48/50 fps

Files follow the official layout under KUNO_LTX_MODELS_DIR (diffusion_models/, text_encoders/,
vae/, latent_upscale_models/, loras/).

Status: flags follow the official CLI docs, not yet run on GPUs. Each job currently starts
a fresh process that reloads ~66 GB of weights, which is fine for validation and far too
slow for serving; replace with a resident pipeline process before launch.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from kuno_protocol.profiles import InputRole, Mode, ltx_num_frames
from kuno_protocol.receipts import VideoInfo

from .base import Backend, GenerationTask, ProgressFn, VideoResult
from .media_tools import BackendError, strip_audio

DEV_PIPELINES = {"ti2vid_two_stages", "keyframe_interpolation", "a2vid_two_stage"}


@dataclass(frozen=True)
class LtxPaths:
    root: Path

    def _p(self, *parts: str) -> str:
        return str(self.root.joinpath(*parts))

    @property
    def distilled_transformer(self) -> str:
        return self._p("diffusion_models", "ltx-2.5-22b-distilled-transformer-bf16.safetensors")

    @property
    def dev_transformer(self) -> str:
        return self._p("diffusion_models", "ltx-2.5-22b-dev-transformer-bf16.safetensors")

    @property
    def text_encoder(self) -> str:
        return self._p("text_encoders", "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")

    @property
    def video_vae(self) -> str:
        return self._p("vae", "ltx-2.5-video-vae-bf16.safetensors")

    @property
    def audio_vae(self) -> str:
        return self._p("vae", "ltx-2.5-audio-vae-bf16.safetensors")

    @property
    def spatial_upsampler(self) -> str:
        return self._p("latent_upscale_models", "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")

    @property
    def temporal_upsampler(self) -> str:
        return self._p("latent_upscale_models", "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors")

    @property
    def distilled_lora(self) -> str:
        return self._p("loras", "ltx-2.5-22b-distilled-lora-450-bf16.safetensors")

    @property
    def detailing_lora(self) -> str:
        return self._p("loras", "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors")


def pick_pipeline(task: GenerationTask) -> str:
    mode, variant = task.params.mode, task.profile.variant
    if mode is Mode.RETAKE:
        return "retake"
    if mode is Mode.AUDIO_TO_VIDEO:
        return "a2vid_two_stage"
    if variant == "dfr":
        return "dfr_pipeline"
    if variant == "pro":
        return "keyframe_interpolation" if mode in (Mode.KEYFRAMES, Mode.FIRST_LAST_FRAME) else "ti2vid_two_stages"
    return "distilled"


def build_command(task: GenerationTask, paths: LtxPaths, directory: Path, output: Path) -> tuple[list[str], int, float]:
    """Returns (argv, frames in the output file, output fps)."""
    params = task.params
    pipeline = pick_pipeline(task)
    temporal = pipeline == "dfr_pipeline" and params.fps >= 48
    gen_fps = params.fps / 2 if temporal else float(params.fps)
    gen_frames = ltx_num_frames(params.duration_s, int(gen_fps))
    out_frames = (gen_frames - 1) * 2 + 1 if temporal else gen_frames

    argv = [
        sys.executable, "-m", f"ltx_pipelines.{pipeline}",
        "--transformer-path", paths.dev_transformer if pipeline in DEV_PIPELINES else paths.distilled_transformer,
        "--text-encoder-path", paths.text_encoder,
        "--video-vae-path", paths.video_vae,
        "--audio-vae-path", paths.audio_vae,
        "--seed", str(task.seed),
        "--width", str(task.width),
        "--height", str(task.height),
        "--frame-rate", f"{gen_fps:g}",
        "--output-path", str(output),
        "--prompt", task.prompt,
    ]
    if pipeline != "retake":
        argv += ["--spatial-upsampler-path", paths.spatial_upsampler]
    if pipeline in DEV_PIPELINES:
        argv += ["--distilled-lora", paths.distilled_lora, "1.0", "--num-inference-steps", "30"]
        if task.negative_prompt:
            argv += ["--negative-prompt", task.negative_prompt]
    if pipeline == "dfr_pipeline":
        argv += [
            "--detailing-lora", paths.detailing_lora,
            "--temporal-upsampler-path", paths.temporal_upsampler,
            "--temporal-upscalings", "1" if temporal else "0",
        ]
    # No --enhance-prompt: the CLI would rewrite the prompt inside its own process, where the worker cannot check the
    # result before rendering. This backend has no separate enhancement step, so the option has no effect here.

    if pipeline == "a2vid_two_stage":
        audio = task.first(InputRole.SOURCE_AUDIO)
        argv += [
            "--audio-path", str(audio.save(directory)),
            "--audio-start-time", f"{audio.ref.start_s or 0:g}",
            "--audio-max-duration", f"{params.duration_s:g}",
        ]
    else:
        argv += ["--num-frames", str(gen_frames)]

    if pipeline == "retake":
        source = task.first(InputRole.SOURCE_VIDEO)
        window = task.options.get("retake", {})
        start = float(window.get("start_s", source.ref.start_s or 0.0))
        end = float(window.get("end_s", source.ref.end_s or params.duration_s))
        argv += ["--video-path", str(source.save(directory)), "--start-time", f"{start:g}", "--end-time", f"{end:g}"]

    last_index = gen_frames - 1
    for item in task.inputs:
        role = item.ref.role
        if role is InputRole.FIRST_FRAME or (role is InputRole.REFERENCE_IMAGE and pipeline == "a2vid_two_stage"):
            index = 0
        elif role is InputRole.LAST_FRAME:
            index = last_index
        elif role is InputRole.KEYFRAME:
            index = min(max(round((item.ref.time_s or 0.0) * gen_fps), 0), last_index)
        else:
            continue
        argv += ["--image", str(item.save(directory)), str(index), f"{item.ref.strength or 1.0:g}"]
    return argv, out_frames, gen_fps * (2 if temporal else 1)


class LtxPipelinesBackend(Backend):
    name = "ltx-2.5"

    def __init__(self, models_dir: Path | None, workdir: Path):
        if models_dir is None:
            raise ValueError("KUNO_LTX_MODELS_DIR must point at the LTX-2.5 weights")
        self.paths = LtxPaths(Path(models_dir))
        self.workdir = Path(workdir)

    def generate(self, task: GenerationTask, progress: ProgressFn) -> VideoResult:
        directory = self.workdir / task.job_id
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "out.mp4"
        try:
            argv, frames, fps = build_command(task, self.paths, directory, output)
            progress(0.05, "denoising")
            # stdout/stderr are discarded: pipelines may echo the prompt.
            result = subprocess.run(argv, cwd=directory, capture_output=True, timeout=task.profile.timeout_s)
            if result.returncode != 0 or not output.exists():
                raise BackendError(f"LTX pipeline exited with code {result.returncode}")
            data = output.read_bytes()
            if not task.params.audio:
                data = strip_audio(data)
            progress(1.0, "decoded")
            info = VideoInfo(
                duration_s=round(frames / fps, 3), width=task.width, height=task.height, fps=fps, frames=frames, audio=task.params.audio
            )
            return VideoResult(data=data, info=info)
        finally:
            shutil.rmtree(directory, ignore_errors=True)
