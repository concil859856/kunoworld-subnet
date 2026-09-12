"""MiniMax H3 through its official runtimes, running inside the same confidential VM.

Media reaches the runtime as TEE-local file paths and is deleted after each job.
Never enable MiniMax's hosted prompt rewriter (H3-Context-IR) or 2K regenerator:
both are API-only and would send customer content outside the enclave.

sglang   profiles `h3` and `h3-reference` (official recipe, 4 GPUs, Ulysses SP):
           sglang serve --model-path MiniMaxAI/MiniMax-H3 --num-gpus 4 --ulysses-degree 4 \
             --performance-mode speed --port 30010 --model-variant fl2va
           sglang serve ... --port 30011 --model-variant ref2va
         POST /v1/videos → GET /v1/videos/{id} → GET /v1/videos/{id}/content

lightx2v profile `h3-turbo`: ModelTC/Minimax-H3-Turbo `inference_minimax_h3.py` with the
         8-step 768p LoRA (video shift 6, audio shift 3). The script loads weights per
         invocation; production needs a resident runtime before this profile goes live.

Status: written against the official docs, not yet run on GPUs. Validate every mode
against the reference outputs in Phase 0 before enabling the profiles.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

from kuno_protocol.profiles import InputRole, Mode
from kuno_protocol.receipts import VideoInfo

from .base import Backend, GenerationTask, ProgressFn, VideoResult
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
TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors"


def _uri(path: Path) -> str:
    return f"file://{path.resolve()}"


def build_sglang_request(task: GenerationTask, directory: Path) -> tuple[str, dict]:
    """Returns (model variant, request body) for the official SGLang video API."""
    params = task.params
    body: dict = {
        "prompt": task.prompt,
        "target": {
            "short_edge": min(task.width, task.height),
            "aspect_ratio": params.aspect_ratio,
            "duration_seconds": params.duration_s,
        },
        "seed": task.seed,
        "num_inference_steps": task.profile.steps,
    }
    if params.mode in FL2VA_MODES:
        conditions = []
        for role, frame_index in ((InputRole.FIRST_FRAME, 0), (InputRole.LAST_FRAME, -1)):
            item = task.first(role)
            if item is not None:
                conditions.append({"type": "image", "uri": _uri(item.save(directory)), "role": "keyframe", "frame_index": frame_index})
        body.update(task="fl2va" if conditions else "t2va", conditions=conditions)
        return "fl2va", body

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
    return "ref2va", body


def _finish(task: GenerationTask, data: bytes) -> VideoResult:
    if not task.params.audio:
        data = strip_audio(data)
    frames = task.num_frames
    info = VideoInfo(duration_s=round(frames / FPS, 3), width=task.width, height=task.height, fps=FPS, frames=frames, audio=task.params.audio)
    return VideoResult(data=data, info=info)


class H3SglangBackend(Backend):
    name = "minimax-h3"

    def __init__(self, fl2va_url: str, ref2va_url: str, workdir: Path, poll_s: float = 1.0, http: httpx.Client | None = None):
        self.urls = {"fl2va": fl2va_url.rstrip("/"), "ref2va": ref2va_url.rstrip("/")}
        self.workdir = Path(workdir)
        self.poll_s = poll_s
        self.http = http or httpx.Client(timeout=60.0)
        self.turbo = H3TurboBackend(self.workdir)

    def generate(self, task: GenerationTask, progress: ProgressFn) -> VideoResult:
        if task.profile.runtime == "lightx2v":
            return self.turbo.generate(task, progress)
        directory = self.workdir / task.job_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            variant, body = build_sglang_request(task, directory)
            base = self.urls[variant]
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


class H3TurboBackend(Backend):
    """LightX2V turbo script. Jobs-JSON keys beyond duration/megapixels/aspect_ratio
    follow the repo's examples and must be checked against examples/prompts_t2va_test.json."""

    name = "minimax-h3-turbo"

    def __init__(self, workdir: Path, script: str = "inference_minimax_h3.py", lora_path: str = TURBO_LORA):
        self.workdir = Path(workdir)
        self.script = script
        self.lora_path = lora_path

    def generate(self, task: GenerationTask, progress: ProgressFn) -> VideoResult:
        directory = self.workdir / task.job_id
        out_dir = directory / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            job: dict = {
                "prompt": task.prompt,
                "duration": task.params.duration_s,
                "megapixels": round(task.width * task.height / 1_000_000, 4),
                "aspect_ratio": task.params.aspect_ratio,
            }
            images = []
            for role, frame_index in ((InputRole.FIRST_FRAME, 0), (InputRole.LAST_FRAME, -1)):
                item = task.first(role)
                if item is not None:
                    images.append({"path": str(item.save(directory)), "frame_index": frame_index})
            if images:
                job["images"] = images
            jobs_file = directory / "jobs.json"
            jobs_file.write_text(json.dumps([job]))
            command = [
                sys.executable, self.script,
                "--jobs-json", str(jobs_file),
                "--lora-path", self.lora_path,
                "--inference-steps", str(task.profile.steps),
                "--video-shift", "6", "--audio-shift", "3", "--lora-alpha", "128",
                "--seed", str(task.seed),
                "--output-dir", str(out_dir),
                "--no-cpu-offload",
            ]
            progress(0.05, "denoising")
            result = subprocess.run(command, cwd=directory, capture_output=True, timeout=task.profile.timeout_s)
            if result.returncode != 0:
                raise BackendError(f"H3 turbo runtime exited with code {result.returncode}")
            outputs = sorted(out_dir.glob("*.mp4"))
            if not outputs:
                raise BackendError("H3 turbo runtime produced no video")
            return _finish(task, outputs[0].read_bytes())
        finally:
            shutil.rmtree(directory, ignore_errors=True)
