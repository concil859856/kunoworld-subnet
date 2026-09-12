"""Placeholder renderer for development and tests. It exercises every mode with
the real input media (frames, clips, audio) using ffmpeg, so the full encrypted
pipeline can run on a laptop without GPUs."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from kuno_protocol.profiles import InputRole, Mode
from kuno_protocol.receipts import VideoInfo

from .base import Backend, GenerationTask, ProgressFn, VideoResult
from .media_tools import ffmpeg_exe as _ffmpeg


class MockBackend(Backend):
    name = "mock"

    def __init__(self, ffmpeg: str | None = None):
        self.ffmpeg = ffmpeg or _ffmpeg()

    def generate(self, task: GenerationTask, progress: ProgressFn) -> VideoResult:
        params = task.params
        w, h, fps, d = task.width, task.height, params.fps, params.duration_s
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
        return VideoResult(data=data, info=info)

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
