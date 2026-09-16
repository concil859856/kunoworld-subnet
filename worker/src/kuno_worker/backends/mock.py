"""Placeholder renderer for development and tests. It exercises every mode with
the real input media (frames, clips, audio) using ffmpeg, so the full encrypted
pipeline can run on a laptop without GPUs.

It also runs verified mode for real: a tiny deterministic toy denoiser
(`kuno_protocol.toy_denoiser`) produces a genuine per-step trajectory, so dev networks
commit to steps, retain them, answer audits and get re-executed exactly like GPU miners.
The rendered video does not depend on the toy latents."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from kuno_protocol.canonical import canonical_json, sha256_hex
from kuno_protocol.profiles import InputRole, Mode
from kuno_protocol.receipts import VideoInfo
from kuno_protocol.toy_denoiser import DEV_HARDWARE_CLASS, TOY_RUNTIME, run_toy_trajectory, toy_model_digest, toy_replay_step, toy_transcript
from kuno_protocol.verified import StepCommitment, StepTranscript, Tensor

from ..verified import OpeningsHandle, RetentionStore, context_bytes, shared_retention
from .base import Backend, GenerationTask, ProgressFn, VideoResult
from .media_tools import ffmpeg_exe as _ffmpeg


def toy_replayer(transcript: StepTranscript, context: bytes, target: int, state: list[Tensor]) -> list[Tensor]:
    values = json.loads(context)
    return toy_replay_step(transcript, values["prompt"], values.get("negative_prompt"), target, state)


class MockBackend(Backend):
    name = "mock"
    storyboards = True

    def __init__(self, ffmpeg: str | None = None, retention: RetentionStore | None = None, hardware_class: str | None = DEV_HARDWARE_CLASS):
        self.ffmpeg = ffmpeg or _ffmpeg()
        self.retention = retention
        self.hardware_class = hardware_class
        (retention or shared_retention()).register_replayer(TOY_RUNTIME, toy_replayer)

    def generate(self, task: GenerationTask, progress: ProgressFn) -> VideoResult:
        if task.params.mode is Mode.STORYBOARD:
            return self._storyboard(task, progress)
        params = task.params
        w, h, fps, d = task.width, task.height, params.fps, params.duration_s
        progress(0.05, "denoising")
        verified = self.verified_trajectory(task)
        progress(0.1, "rendering")
        with tempfile.TemporaryDirectory(prefix="kuno-mock-") as tmp:
            tmpdir = Path(tmp)
            inputs, graph = self._video_graph(task, tmpdir, w, h, fps, d)
            audio_args: list[str] = []
            if params.audio:
                index = inputs.count("-i")
                source = task.first(InputRole.SOURCE_AUDIO) or task.first(InputRole.REFERENCE_AUDIO)
                if source is not None:
                    inputs += ["-stream_loop", "-1", "-t", f"{d:.3f}", "-i", str(source.save(tmpdir))]
                else:
                    inputs += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", f"sine=frequency={220 + task.seed % 440}:sample_rate=48000"]
                graph += f";[{index}:a]aresample=48000,aformat=channel_layouts=stereo[a]"
                audio_args = ["-map", "[a]", "-c:a", "aac", "-b:a", "128k"]
            out = tmpdir / "out.mp4"
            cmd = [
                self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs,
                "-filter_complex", graph, "-map", "[v]", *audio_args,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p",
                "-r", str(fps), "-t", f"{d:.3f}", "-movflags", "+faststart", str(out),
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            data = out.read_bytes()
        progress(1.0, "rendered")
        info = VideoInfo(duration_s=d, width=w, height=h, fps=fps, frames=round(d * fps), audio=params.audio)
        commitment, openings = verified if verified is not None else (None, None)
        return VideoResult(data=data, info=info, step_commitment=commitment, openings=openings)

    def _storyboard(self, task: GenerationTask, progress: ProgressFn) -> VideoResult:
        """The stitched video's exact frame count, fps and duration, each shot's kept frames in a hue of its own, with
        the real backend's `shot i/N` stages. Storyboards are never verified, so there is no toy trajectory."""
        params = task.params
        w, h, fps = task.width, task.height, params.fps
        kept = task.shot_frames or []
        frames, count = sum(kept), len(kept)
        graph = []
        for index, shot_frames in enumerate(kept):
            progress(0.05 + 0.8 * index / count, f"shot {index + 1}/{count}")
            graph.append(f"testsrc2=size={w}x{h}:rate={fps},trim=end_frame={shot_frames},hue=h={(task.seed + 47 * index) % 360},format=yuv420p[s{index}]")
        graph.append("".join(f"[s{index}]" for index in range(count)) + f"concat=n={count}:v=1:a=0[v]")
        progress(0.9, "encoding")
        with tempfile.TemporaryDirectory(prefix="kuno-mock-") as tmp:
            inputs: list[str] = []
            audio_args: list[str] = []
            if params.audio:
                inputs = ["-f", "lavfi", "-t", f"{frames / fps:.6f}", "-i", f"sine=frequency={220 + task.seed % 440}:sample_rate=48000"]
                graph.append("[0:a]aresample=48000,aformat=channel_layouts=stereo[a]")
                audio_args = ["-map", "[a]", "-c:a", "aac", "-b:a", "128k"]
            out = Path(tmp) / "out.mp4"
            cmd = [
                self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(graph), "-map", "[v]",
                *audio_args, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p", "-r", str(fps),
                "-frames:v", str(frames), "-movflags", "+faststart", str(out),
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            data = out.read_bytes()
        progress(1.0, "rendered")
        info = VideoInfo(duration_s=round(frames / fps, 3), width=w, height=h, fps=fps, frames=frames, audio=params.audio)
        return VideoResult(data=data, info=info)

    # ------------------------------------------------------------ verified mode

    def model_digest(self, task: GenerationTask) -> str:
        return toy_model_digest(task.profile.id, task.profile.checkpoint)

    def transcript(self, task: GenerationTask) -> StepTranscript:
        return toy_transcript(
            job_id=task.job_id,
            params_digest=sha256_hex(canonical_json(task.params.model_dump(mode="json"))),
            profile_id=task.profile.id,
            family=task.profile.family,
            model_digest=self.model_digest(task),
            seed=task.seed,
            prompt=task.prompt,
            negative_prompt=task.negative_prompt,
            frames=task.num_frames,
            steps=task.profile.steps,
            hardware_class=self.hardware_class or DEV_HARDWARE_CLASS,
        )

    def trajectory(self, transcript: StepTranscript, task: GenerationTask):
        """The denoising loop. Every leaf it yields goes through the step hook."""
        return run_toy_trajectory(transcript, task.prompt, task.negative_prompt)

    def verified_trajectory(self, task: GenerationTask) -> tuple[StepCommitment, OpeningsHandle] | None:
        recorder = self.step_recorder(task, context_bytes(prompt=task.prompt, negative_prompt=task.negative_prompt))
        if recorder is None:
            return None
        transcript = self.transcript(task)
        try:
            for index, stage, kind, sigma, tensors in self.trajectory(transcript, task):
                recorder.report(index, stage, kind, sigma, tensors)
            return recorder.finish(transcript)
        except BaseException:
            recorder.abort()
            raise

    def _video_graph(self, task: GenerationTask, tmp: Path, w: int, h: int, fps: int, d: float) -> tuple[list[str], str]:
        fit = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1"
        keyframes = task.all(InputRole.KEYFRAME)
        first = task.first(InputRole.FIRST_FRAME) or (keyframes[0] if keyframes else None) or task.first(InputRole.REFERENCE_IMAGE)
        last = task.first(InputRole.LAST_FRAME) or (keyframes[-1] if len(keyframes) > 1 else None)
        source = task.first(InputRole.SOURCE_VIDEO) or task.first(InputRole.REFERENCE_VIDEO)

        if first is not None and last is not None:
            a, b = first.save(tmp), last.save(tmp)
            inputs = ["-loop", "1", "-t", f"{d:.3f}", "-i", str(a), "-loop", "1", "-t", f"{d * 0.8:.3f}", "-i", str(b)]
            graph = (
                f"[0:v]{fit},fps={fps},format=yuv420p[a0];[1:v]{fit},fps={fps},format=yuv420p[b0];"
                f"[a0][b0]xfade=transition=fade:duration={d * 0.6:.3f}:offset={d * 0.2:.3f}[v]"
            )
            return inputs, graph

        image = first or last
        if image is not None:
            frames = max(1, round(d * fps))
            step = 0.25 / frames
            zoom = f"1.25-on*{step:.6f}" if task.params.mode is Mode.LAST_FRAME else f"1+on*{step:.6f}"
            graph = (
                f"[0:v]scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},"
                f"zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps},"
                f"setsar=1,format=yuv420p[v]"
            )
            return ["-i", str(image.save(tmp))], graph

        if source is not None:
            graph = f"[0:v]{fit},fps={fps},hue=h={task.seed % 360},format=yuv420p[v]"
            return ["-stream_loop", "-1", "-t", f"{d:.3f}", "-i", str(source.save(tmp))], graph

        graph = f"[0:v]hue=h={task.seed % 360},format=yuv420p[v]"
        return ["-f", "lavfi", "-t", f"{d:.3f}", "-i", f"testsrc2=size={w}x{h}:rate={fps}"], graph
