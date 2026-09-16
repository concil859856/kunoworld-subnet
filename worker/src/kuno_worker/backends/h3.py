"""MiniMax H3 through its official runtimes, running inside the same confidential VM.

Media reaches the runtime as TEE-local file paths and is deleted after each job.
Never enable MiniMax's hosted prompt rewriter (H3-Context-IR) or 2K regenerator:
both are API-only and would send customer content outside the enclave.

Every profile goes to one of three SGLang servers (official recipe, 4 GPUs, Ulysses SP), which
h3_servers.py starts beside the worker:

  h3            fl2va    sglang serve --model-path MiniMaxAI/MiniMax-H3 --num-gpus 4 --ulysses-degree 4 \
                           --performance-mode speed --port 30010 --model-variant fl2va
  h3-reference  ref2va   sglang serve ... --port 30011 --model-variant ref2va
  h3-turbo      turbo    sglang serve ... --port 30012 --model-variant fl2va \
                           --lora-path <KUNO_H3_TURBO_LORA> --lora-nickname turbo
                         and each request carries the LoRA's shifts (video 6, audio 3)

POST /v1/videos → GET /v1/videos/{id} → GET /v1/videos/{id}/content

The exception is verified mode for h3-turbo. Its profile pins the diffusers modular pipeline
(`diffusers-modular-h3/1`), whose every step the worker commits to, so when the worker's hardware
class is one the profile pins, `real` runs it in process (h3_resident.py) and starts no Turbo
server. `h3` and `h3-reference` pin that runtime too, but SGLang exposes no step hook: they stay on
SGLang in either mode, and their receipts carry no step commitment.

Status: `h3` and `h3-reference` ran through this backend on 4x H200 without confidential computing
(2026-09-15, one profile per worker). Turbo ran only straight against `sglang serve --lora-path`
with these shifts (2026-09-16); the Turbo server and requests as built here have not run on GPUs.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import httpx

from kuno_protocol.profiles import InputRole, Mode, ModelProfile, h3_schedule_points
from kuno_protocol.receipts import VideoInfo

from .base import Backend, GenerationTask, ProgressFn, VideoResult
from .h3_resident import TURBO_SHIFTS
from .media_tools import BackendError, strip_audio

FPS = 24
FL2VA_MODES = {Mode.TEXT_TO_VIDEO, Mode.IMAGE_TO_VIDEO, Mode.LAST_FRAME, Mode.FIRST_LAST_FRAME}
REF_TYPES = {
    InputRole.REFERENCE_IMAGE: "image",
    InputRole.FIRST_FRAME: "image",
    InputRole.REFERENCE_VIDEO: "video",
    InputRole.SOURCE_VIDEO: "video_audio",
    InputRole.REFERENCE_AUDIO: "audio",
    InputRole.SOURCE_AUDIO: "audio",
}
# The file KUNO_H3_TURBO_LORA names: lightx2v/Minimax-h3-Turbo's 8-step 768p LoRA.
TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors"
TURBO_NICKNAME = "turbo"
# SGLang's names for the shifts the LoRA was trained with; `audio_flow_shift` is an extra field its H3 pipeline reads.
SGLANG_TURBO_SHIFTS = {"flow_shift": TURBO_SHIFTS["video_shift"], "audio_flow_shift": TURBO_SHIFTS["audio_shift"]}


def is_turbo(profile: ModelProfile) -> bool:
    """LightX2V's distilled LoRA on the fl2va checkpoint (profile runtime `lightx2v`, which names where the LoRA comes from)."""
    return profile.runtime == "lightx2v"


def server_for(profile: ModelProfile, mode: Mode) -> str:
    """The SGLang server a job goes to: fl2va or ref2va by checkpoint variant, or turbo for the LoRA profile."""
    if is_turbo(profile):
        return "turbo"
    return "fl2va" if mode in FL2VA_MODES else "ref2va"


def turbo_in_process(profile: ModelProfile, hardware_class: str | None) -> bool:
    """True when `real` runs `profile` in the worker process instead of on SGLang: h3-turbo in verified mode, on a
    hardware class the profile pins (the same test as Backend.verified_enabled)."""
    return (
        is_turbo(profile)
        and hardware_class is not None
        and profile.verified is not None
        and profile.verified.hardware_class(hardware_class) is not None
    )


def _uri(path: Path) -> str:
    return f"file://{path.resolve()}"


def build_sglang_request(task: GenerationTask, directory: Path) -> tuple[str, dict]:
    """Returns (server, request body) for the official SGLang video API; `server_for` names the server."""
    params = task.params
    body: dict = {
        "prompt": task.prompt,
        "target": {
            "short_edge": min(task.width, task.height),
            "aspect_ratio": params.aspect_ratio,
            "duration_seconds": params.duration_s,
        },
        "seed": task.seed,
        "num_inference_steps": h3_schedule_points(task.profile.steps),
    }
    if params.mode in FL2VA_MODES:
        conditions = []
        for role, frame_index in ((InputRole.FIRST_FRAME, 0), (InputRole.LAST_FRAME, -1)):
            item = task.first(role)
            if item is not None:
                conditions.append({"type": "image", "uri": _uri(item.save(directory)), "role": "keyframe", "frame_index": frame_index})
        body.update(task="fl2va" if conditions else "t2va", conditions=conditions)
        if is_turbo(task.profile):
            body.update(SGLANG_TURBO_SHIFTS)
        return server_for(task.profile, params.mode), body

    # Reference order sets the <Picture n> / <Video n> / <Audio n> labels the prompt refers to.
    conditions = []
    for item in task.inputs:
        kind = REF_TYPES[item.ref.role]
        if kind == "video_audio" and not task.options.get("keep_source_audio", True):
            kind = "video"
        condition = {"type": kind, "uri": _uri(item.save(directory)), "role": "reference"}
        if item.ref.start_s is not None:
            condition["start_time_seconds"] = item.ref.start_s
        conditions.append(condition)
    body.update(task="ref2va", conditions=conditions)
    return server_for(task.profile, params.mode), body


def _finish(task: GenerationTask, data: bytes) -> VideoResult:
    if not task.params.audio:
        data = strip_audio(data)
    frames = task.num_frames
    info = VideoInfo(duration_s=round(frames / FPS, 3), width=task.width, height=task.height, fps=FPS, frames=frames, audio=task.params.audio)
    return VideoResult(data=data, info=info)


class H3SglangBackend(Backend):
    name = "minimax-h3"

    def __init__(self, fl2va_url: str, ref2va_url: str, workdir: Path, poll_s: float = 1.0, http: httpx.Client | None = None,
                 turbo_url: str = "http://127.0.0.1:30012", turbo: Backend | None = None):
        """`turbo` is the in-process pipeline for verified h3-turbo (build_backends sets it only where a verified
        class could use it); every other job, h3-turbo in performance mode included, goes to the servers."""
        self.urls = {"fl2va": fl2va_url.rstrip("/"), "ref2va": ref2va_url.rstrip("/"), "turbo": turbo_url.rstrip("/")}
        self.workdir = Path(workdir)
        self.poll_s = poll_s
        self.http = http or httpx.Client(timeout=60.0)
        self.turbo = turbo

    def _in_process(self, profile: ModelProfile) -> bool:
        return self.turbo is not None and turbo_in_process(profile, self.turbo.hardware_class)

    def verified_enabled(self, profile: ModelProfile) -> bool:
        # Only the in-process Turbo pipeline commits to its steps; the SGLang servers have no step hook.
        return self._in_process(profile)

    def warm(self, profile: ModelProfile) -> None:
        if self._in_process(profile):
            self.turbo.warm(profile)  # the servers were ready before the worker started (h3_servers.py)

    def generate(self, task: GenerationTask, progress: ProgressFn) -> VideoResult:
        if self._in_process(task.profile):
            return self.turbo.generate(task, progress)
        directory = self.workdir / task.job_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            server, body = build_sglang_request(task, directory)
            base = self.urls[server]
            response = self.http.post(f"{base}/v1/videos", json=body)
            if response.status_code >= 400:
                raise BackendError(f"H3 runtime rejected the request (HTTP {response.status_code})")
            video_id = response.json()["id"]
            deadline = time.time() + task.profile.timeout_s
            while True:
                status = self.http.get(f"{base}/v1/videos/{video_id}").json()
                state = str(status.get("status", "")).lower()
                if state in ("completed", "succeeded"):
                    break
                if state in ("failed", "error", "cancelled", "canceled"):
                    raise BackendError("H3 runtime reported a failed generation")
                if time.time() > deadline:
                    raise BackendError("H3 runtime timed out")
                if "progress" in status:
                    value = float(status["progress"])
                    progress(min(value / 100 if value > 1 else value, 0.99), "denoising")
                time.sleep(self.poll_s)
            data = self.http.get(f"{base}/v1/videos/{video_id}/content", timeout=300).content
            progress(1.0, "decoded")
            return _finish(task, data)
        finally:
            shutil.rmtree(directory, ignore_errors=True)
