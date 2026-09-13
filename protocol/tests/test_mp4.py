"""The MP4 probe is what validators trust instead of a miner's description of its output."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from kuno_protocol.mp4 import Mp4Error, probe


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    return imageio_ffmpeg.get_ffmpeg_exe()


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> bytes:
    out = Path(tmp_path_factory.mktemp("mp4")) / "clip.mp4"
    subprocess.run(
        [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=24",
         "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000", "-t", "2", "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(out)],
        check=True, capture_output=True, timeout=60,
    )
    return out.read_bytes()


def test_probe_reads_what_the_file_contains(clip):
    info = probe(clip)
    assert info.duration_s == pytest.approx(2.0, abs=0.05)
    assert (info.width, info.height, info.frames, info.audio) == (160, 96, 48, True)


@pytest.mark.parametrize("data", [b"", b"not a video at all", b"\x00\x00\x00\x08ftyp"])
def test_non_mp4_input_is_rejected(data):
    with pytest.raises(Mp4Error):
        probe(data)


def test_a_truncated_file_is_rejected_not_crashed(clip):
    with pytest.raises(Mp4Error):
        probe(clip[: len(clip) // 3])
