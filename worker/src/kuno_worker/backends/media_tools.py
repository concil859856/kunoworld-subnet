from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class BackendError(Exception):
    """The model runtime failed. Messages must never include request content."""


class CapacityRefused(BackendError):
    """This hardware class cannot fit the request; the message says what it can serve. The worker reports it as
    `capacity_refused` (kuno_protocol.envelope.CAPACITY_REFUSED), not `internal_error`."""


def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def encode_video(frames, fps: float, audio=None, sample_rate: int = 48000, crf: int = 18) -> bytes:
    """Encode raw frames (and optional audio) to MP4.

    Resident pipelines return frames, not files. `frames` may be PIL images or HxWx3
    uint8 arrays; `audio` may be mono or stereo, float in [-1, 1] or int16.
    """
    import numpy as np

    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise BackendError(f"expected HxWx3 RGB frames, got shape {first.shape}")
    height, width = first.shape[:2]

    with tempfile.TemporaryDirectory(prefix="kuno-encode-") as tmp:
        out = Path(tmp) / "out.mp4"
        args = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-"]
        if audio is not None:
            wav = Path(tmp) / "audio.wav"
            _write_wav(wav, audio, sample_rate)
            # The video decides the length: LTX-2.5's vocoder returns 2.010 s for 49 frames (2.042 s), and a bare
            # -shortest cut the last frame. apad extends the audio with silence, so -shortest only trims longer audio.
            args += ["-i", str(wav), "-c:a", "aac", "-b:a", "192k", "-af", "apad", "-shortest"]
        args += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(out)]

        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for frame in frames:
                array = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
                if array.shape[:2] != (height, width):
                    raise BackendError("frames must all be the same size")
                process.stdin.write(array.tobytes())
            process.stdin.close()
        except BrokenPipeError:
            pass
        if process.wait(timeout=600) != 0:
            raise BackendError(f"encoding failed (ffmpeg exit {process.returncode})")
        return out.read_bytes()


def _write_wav(path: Path, audio, sample_rate: int) -> None:
    import wave

    import numpy as np

    if hasattr(audio, "detach"):  # a torch tensor: diffusers vocoders return it on the GPU, often bfloat16, which numpy rejects
        audio = audio.detach().to("cpu").float().numpy()
    samples = np.asarray(audio)
    if samples.ndim == 2 and samples.shape[0] in (1, 2) and samples.shape[0] < samples.shape[1]:
        samples = samples.T  # (channels, n) -> (n, channels)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.dtype.kind == "f":
        samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    else:
        samples = samples.astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(samples.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def strip_audio(video: bytes) -> bytes:
    """Drops the audio track without re-encoding video (H3 always renders audio)."""
    with tempfile.TemporaryDirectory(prefix="kuno-strip-") as tmp:
        src, dst = Path(tmp) / "in.mp4", Path(tmp) / "out.mp4"
        src.write_bytes(video)
        subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-c:v", "copy", "-an", "-movflags", "+faststart", str(dst)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        return dst.read_bytes()
