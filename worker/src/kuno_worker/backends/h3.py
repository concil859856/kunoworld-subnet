"""MiniMax H3 through its official runtimes, running inside the same confidential VM.

Media reaches the runtime as TEE-local file paths and is deleted after each job.
Never enable MiniMax's hosted prompt rewriter (H3-Context-IR) or 2K regenerator:
both are API-only and would send customer content outside the enclave.

Every profile goes to one of three SGLang servers, on as many GPUs as the profile needs (h3 and
h3-reference four, with Ulysses sequence parallelism; h3-turbo one), which h3_servers.py starts
beside the worker:

  h3            fl2va    sglang serve --model-path MiniMaxAI/MiniMax-H3 --num-gpus 4 --ulysses-degree 4 \
                           --performance-mode speed --port 30010 --model-variant fl2va
  h3-reference  ref2va   sglang serve ... --port 30011 --model-variant ref2va
  h3-turbo      turbo    sglang serve ... --num-gpus 1 --ulysses-degree 1 --port 30012 --model-variant fl2va \
                           --lora-path <KUNO_H3_TURBO_LORA> --lora-nickname turbo
                         and each request carries the LoRA's shifts (video 6, audio 3)

One GPU serves Turbo at 9.92 GPU-seconds per output second against 15.6 on four through the worker
(2026-09-17), which is the difference between selling it above and below cost, and a single-GPU
confidential VM is far easier to rent than a whole 8-GPU server. What one card cannot do is the
profile's longest clips: see the serving envelope below.

POST /v1/videos → GET /v1/videos/{id} → GET /v1/videos/{id}/content

The exception is verified mode for h3-turbo. Its profile pins the diffusers modular pipeline
(`diffusers-modular-h3/1`), whose every step the worker commits to, so when the worker's hardware
class is one the profile pins, `real` runs it in process (h3_resident.py) and starts no Turbo
server. `h3` and `h3-reference` pin that runtime too, but SGLang exposes no step hook: they stay on
SGLang in either mode, and their receipts carry no step commitment.

Status: `h3` and `h3-reference` ran through this backend on 4x H200 without confidential computing
(2026-09-15 and 2026-09-17, one profile per worker), and so did `h3-turbo` on four GPUs through a real
gateway (2026-09-17). One-GPU Turbo serving, which is what the profile now asks for, has run only
straight against `sglang serve --lora-path` (2026-09-16 and -17), so the envelope below has never been
checked against a real 10 s render.
"""

from __future__ import annotations

import contextlib
import logging
import math
import shutil
import time
from pathlib import Path

import httpx

from kuno_protocol.profiles import InputRole, Mode, ModelProfile, h3_schedule_points
from kuno_protocol.receipts import VideoInfo

from .base import Backend, GenerationTask, ProgressFn, VideoResult
from .h3_resident import TURBO_SHIFTS
from .media_tools import BackendError, strip_audio

log = logging.getLogger("kuno.worker.h3")

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


# ------------------------------------------------------------------ the one-GPU serving envelope
#
# h3-turbo is served on one GPU (profiles.json, gpus_per_worker 1), and one GPU's memory decides how long a clip it
# can render. Measured peaks of one H200 (nominally 141 GB) at 1344x768 with the 8-step LoRA:
#
#     5 s   126.6 GB with FlashAttention, 128.9 with SageAttention   2026-09-17, research/h3-image-check_2026-09-17.md §3
#    10 s   134,935 MiB of the card's 143,771 (nvidia-smi), FlashAttention, through kuno-h3-worker and a real gateway,
#           2026-09-18 (SGLang's own figure: 133,674 MB); research/h3-image-check_2026-09-17.md, "One-GPU Turbo"
#    14 s   137.6 GB with the 8-step LoRA, 138.9 with the 4-step one 2026-09-16, research/pricing/measured_2026-09-16_h3-turbo.md
#
# so the peak grows by about 1.1 GB per second of clip, and 14 s leaves 2-3 GB of the card: too tight to advertise.
# 10 s leaves about 8.8 GB, 6.5 with SageAttention's extra 2.3, and that is what a 141 GB card advertises (interpolating
# 5 s and 14 s had predicted about 134 GB). A card of 180 GB or more (B200, B300) has 40 GB of headroom at 14 s and
# serves the profile in full; no card above 141 GB has been measured at all. Rungs, smallest card first: (the GPU's total memory, the longest clip it serves). The rungs sit
# far from every real card's size, so it makes no difference whether a card's memory is counted in GB or GiB.
TURBO_ONE_GPU_RUNGS = ((120.0, 10.0), (160.0, 14.0))


def one_gpu_max_duration_s(profile: ModelProfile, memory_gb: float) -> float | None:
    """The longest clip a single GPU of `memory_gb` serves for `profile`, or None when it cannot serve the profile's
    shortest clip. Kept to the profile's own limits and duration step."""
    longest = max((serves for at_least, serves in TURBO_ONE_GPU_RUNGS if memory_gb + 0.5 >= at_least), default=None)
    if longest is None:
        return None
    limits = profile.limits
    longest = min(longest, limits.max_duration_s)
    if limits.duration_step_s:  # only whole steps above the minimum are requestable
        steps = math.floor((longest - limits.min_duration_s) / limits.duration_step_s + 1e-6)
        longest = limits.min_duration_s + max(steps, 0) * limits.duration_step_s
    return longest if longest + 1e-6 >= limits.min_duration_s else None


def one_gpu_envelope(profile: ModelProfile, memory_gb: float) -> dict[str, dict[str, dict[int, float]]]:
    """`profile`'s serving envelope on one GPU of `memory_gb` (kuno_protocol.envelope): its full table capped at the
    longest clip that GPU holds, or {} — serving nothing — for a GPU too small for the profile at all."""
    from kuno_protocol.envelope import full_table

    longest = one_gpu_max_duration_s(profile, memory_gb)
    if longest is None:
        return {}
    return {
        resolution: {aspect: {fps: min(serves, longest) for fps, serves in by_fps.items()} for aspect, by_fps in ratios.items()}
        for resolution, ratios in full_table(profile).items()
    }


def visible_gpu_memory_gb() -> float | None:
    """The total memory of the smallest GPU this process can see, read through NVML (no CUDA context, so nothing is
    taken from the SGLang servers), or None where NVML is missing or answers nothing."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - no driver (a development machine, or the mock network)
        return None
    try:
        totals = [
            pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index)).total / 2**30
            for index in range(pynvml.nvmlDeviceGetCount())
        ]
    except Exception:  # noqa: BLE001 - NVML is there but will not answer
        return None
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()
    return min(totals, default=None) if totals else None


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
                 turbo_url: str = "http://127.0.0.1:30012", turbo: Backend | None = None, memory_gb: float | None = None):
        """`turbo` is the in-process pipeline for verified h3-turbo (build_backends sets it only where a verified
        class could use it); every other job, h3-turbo in performance mode included, goes to the servers.
        `memory_gb` is this worker's GPU memory, for the one-GPU serving envelope; None reads it from NVML."""
        self.urls = {"fl2va": fl2va_url.rstrip("/"), "ref2va": ref2va_url.rstrip("/"), "turbo": turbo_url.rstrip("/")}
        self.workdir = Path(workdir)
        self.poll_s = poll_s
        self.http = http or httpx.Client(timeout=60.0)
        self.turbo = turbo
        self.memory_gb = memory_gb

    def _in_process(self, profile: ModelProfile) -> bool:
        return self.turbo is not None and turbo_in_process(profile, self.turbo.hardware_class)

    def _memory_gb(self) -> float | None:
        if self.memory_gb is None:
            self.memory_gb = visible_gpu_memory_gb()
        return self.memory_gb

    def serving_envelope(self, profile: ModelProfile):
        """What the gateway may route here. A profile served on several GPUs (h3, h3-reference) keeps its full limits:
        four H200s hold every clip it allows. A one-GPU profile (h3-turbo) is capped at what one card of this worker's
        GPU memory holds, the way LtxResidentBackend caps a profile at its memory plan. Without a memory reading
        (no NVML, no driver: the mock network and the tests) nothing is capped, exactly as before envelopes."""
        memory_gb = self._memory_gb() if profile.gpus_per_worker == 1 else None
        if memory_gb is None:
            return super().serving_envelope(profile)
        table = one_gpu_envelope(profile, memory_gb)
        log.info("%s on one %.0f GB GPU serves clips up to %s", profile.id, memory_gb,
                 f"{one_gpu_max_duration_s(profile, memory_gb):g} s" if table else "nothing: the card is too small")
        return table

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
