from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class BackendError(Exception):
    """The model runtime failed. Messages must never include request content."""


def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


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
