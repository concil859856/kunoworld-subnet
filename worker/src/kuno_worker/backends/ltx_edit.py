"""Audio-to-video and retake on LTX-2.5: renders that hold tokens encoded from the customer's own media.

No diffusers 0.40 LTX-2 pipeline takes a sound track or a clip to keep (only the HDR LoRA pipeline takes a video, as a
reference), so both render through ltx_pinning's pinned condition pipeline, the mechanism storyboards join shots with:

  audio_to_video  the source sound from the input's start_s, cut to the clip's length and padded with silence past its
                  end, encoded to audio latents and held under the whole render (both passes of a two-pass recipe). The
                  picture is generated to match it; a first_frame input conditions frame 0 as in image-to-video.
  retake          the source clip from its first frame, conformed to the job's size, frame rate and length, encoded to
                  video latents (and its sound to audio latents). The tokens outside [start_s, end_s) are held; with
                  regenerate_video false every video token is, with regenerate_audio false every audio token is.

Time to latents, LTX-2.5's causal VAEs (ltx_pinning.Geometry; ltx-core's TemporalRegionMask draws the same bounds):
  video  latent frame 0 is pixel frame 0 alone, latent frame k >= 1 is pixel frames 8k - 7 .. 8k. A pixel frame f is shown
         over [f / fps, (f + 1) / fps).
  audio  latent 0 is mel frame 0, latent j >= 1 is mel frames 4j - 3 .. 4j, and mel frame m covers [m, m + 1) x 10 ms.
A latent is regenerated when any part of the time it covers overlaps the window, so the window is rounded out to whole
latents, never in: a 1 s to 3 s retake at 24 fps regenerates latent frames 3..9, pixel frames 17..72 (0.708 s to
3.042 s), and audio latents 25..75 (0.97 s to 3.01 s). Everything held is exactly the source's encoding.

What comes back: the render's frames (outside the window they are the held latents decoded, so near-identical to the
source, not identical), and the sound as 16-bit samples at the vocoder's rate (48 kHz), cut or padded to frames / fps as
media_tools.encode_video would:
  audio_to_video  the source's own samples. The held latents never change, so decoding them would only return the audio
                  VAE and vocoder's reconstruction of the source (a 16 kHz mel, bandwidth-extended); ltx-pipelines'
                  A2VidPipelineTwoStage returns the original waveform for the same reason. 16-bit because
                  media_tools._write_wav writes 16-bit PCM, which passes int16 through untouched and would round floats.
  retake          the source's samples wherever audio is held, and the render's inside the regenerated span, crossfaded
                  over EDIT_CROSSFADE_S inside that span so nothing held is touched; the render's sound alone when the
                  source has none.

Never verified: no step commitment on any class (LtxResidentBackend.step_recorder), as for storyboards.

Plain data first (EditPlan, the decoders, the splice: numpy and ffmpeg), then `render_edit`, which needs torch.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ltx_pinning import Geometry, Pins, pin_failure
from .media_tools import BackendError, ffmpeg_exe

AUDIO_TO_VIDEO, RETAKE = "audio_to_video", "retake"
# The crossfade from held source sound into a retake's regenerated sound, and back, inside the regenerated span: the render
# continues the held latents, but its vocoder's waveform is not the source's, so a hard cut could click.
EDIT_CROSSFADE_S = 0.02
# A source clip this many frames shorter than the render is held with its last frame repeated: LTX renders 8k + 1 frames,
# so a whole-second clip is up to 5 frames short of its own duration on that grid (12 s at 25 fps: 300 frames, 305 rendered).
SOURCE_FRAME_SLACK = 8
_EPS = 1e-9


class EditError(BackendError):
    """An audio-to-video or retake job can't be rendered as asked. Messages carry counts only, never request content."""


# ------------------------------------------------------------------ the plan (plain data)


def video_latent_times(g: Geometry, latent_frames: int) -> list[tuple[float, float]]:
    """[start, end) seconds each video latent frame covers."""
    out = []
    for k in range(latent_frames):
        first, last = (0, 0) if k == 0 else (g.temporal_ratio * k - (g.temporal_ratio - 1), g.temporal_ratio * k)
        out.append((first / g.fps, (last + 1) / g.fps))
    return out


def audio_latent_times(g: Geometry, latents: int) -> list[tuple[float, float]]:
    """[start, end) seconds each audio latent covers."""
    mel_s = g.mel_hop / g.mel_sample_rate
    out = []
    for j in range(latents):
        first, last = (0, 0) if j == 0 else (g.audio_ratio * j - (g.audio_ratio - 1), g.audio_ratio * j)
        out.append((first * mel_s, (last + 1) * mel_s))
    return out


def overlapping(times: list[tuple[float, float]], start_s: float, end_s: float) -> tuple[int, int]:
    """The latents [a, b) whose time overlaps [start_s, end_s); (0, 0) when none does. Latent times only move forward, so
    the overlapping ones are consecutive."""
    hits = [i for i, (t0, t1) in enumerate(times) if t1 > start_s + _EPS and t0 < end_s - _EPS]
    return (hits[0], hits[-1] + 1) if hits else (0, 0)


def _held(count: int, span: tuple[int, int]) -> list[tuple[int, int]]:
    """The ranges of [0, count) outside `span`."""
    a, b = span
    return [(lo, hi) for lo, hi in ((0, a), (b, count)) if hi > lo]


@dataclass(frozen=True)
class EditPlan:
    """Which latents a render holds and which it regenerates. Spans are [start, end); an empty span (a == b) regenerates
    nothing of that modality, a full one holds nothing."""

    mode: str
    frames: int
    latent_frames: int
    audio_latents: int
    samples: int  # the output sound track's length at the vocoder's rate
    video_span: tuple[int, int]  # regenerated video latent frames
    audio_span: tuple[int, int]  # regenerated audio latents (retake: before knowing whether the source has sound)
    frame_span: tuple[int, int]  # the pixel frames the regenerated latent frames decode to
    sample_span: tuple[int, int]  # the output samples the regenerated audio latents decode to (to the end if they reach it)
    start_s: float = 0.0
    end_s: float = 0.0

    @property
    def held_video(self) -> list[tuple[int, int]]:
        return _held(self.latent_frames, self.video_span)

    @property
    def held_audio(self) -> list[tuple[int, int]]:
        return _held(self.audio_latents, self.audio_span)


def _span_frames(g: Geometry, span: tuple[int, int]) -> tuple[int, int]:
    a, b = span
    if a == b:
        return (0, 0)
    first = 0 if a == 0 else g.temporal_ratio * a - (g.temporal_ratio - 1)
    return (first, g.pixel_frames(b))


def _span_samples(g: Geometry, span: tuple[int, int], latents: int, samples: int) -> tuple[int, int]:
    a, b = span
    if a == b:
        return (0, 0)
    first = 0 if a == 0 else round((g.audio_ratio * a - (g.audio_ratio - 1)) * g.samples_per_mel)
    # A span reaching the last latent owns the tail past the decoded sound too: a render's vocoder output ends a few ms
    # short of frames / fps, and encode_video pads that with silence.
    last = samples if b == latents else round(g.mel_frames(b) * g.samples_per_mel)
    return (first, min(last, samples))


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise EditError(f"{what} must be a non-negative number of seconds")
    return float(value)


def plan_edit(g: Geometry, frames: int, edit: dict[str, Any]) -> EditPlan:
    """The held and regenerated latents of a `build_call` edit spec for a render of `frames` frames."""
    mode = edit.get("mode")
    latent_frames, audio_latents, samples = g.latent_frames(frames), g.audio_latents(frames), g.output_samples(frames)
    duration = frames / g.fps
    if mode == AUDIO_TO_VIDEO:
        start = _finite(edit.get("start_s") or 0.0, "the source audio's start_s")
        end = edit.get("end_s")
        if end is not None and _finite(end, "the source audio's end_s") <= start:
            raise EditError("the source audio's end_s must be after its start_s")
        return EditPlan(
            mode=mode, frames=frames, latent_frames=latent_frames, audio_latents=audio_latents, samples=samples,
            video_span=(0, latent_frames), audio_span=(0, 0), frame_span=(0, frames), sample_span=(0, 0), start_s=start,
            end_s=start + duration if end is None else min(float(end), start + duration),
        )
    if mode != RETAKE:
        raise EditError(f"unknown edit mode {mode!r}")
    start = _finite(edit.get("start_s") or 0.0, "the retake window's start_s")
    end = duration if edit.get("end_s") is None else min(_finite(edit["end_s"], "the retake window's end_s"), duration)
    if start >= duration:
        raise EditError(f"the retake window starts at or after the clip's end ({duration:.3f} s)")
    if end <= start:
        raise EditError("the retake window's end_s must be after its start_s")
    regenerate_video, regenerate_audio = edit.get("regenerate_video", True), edit.get("regenerate_audio", True)
    if not regenerate_video and not regenerate_audio:
        raise EditError("a retake must regenerate the video, the audio or both")
    video = overlapping(video_latent_times(g, latent_frames), start, end) if regenerate_video else (0, 0)
    audio = overlapping(audio_latent_times(g, audio_latents), start, end) if regenerate_audio else (0, 0)
    return EditPlan(
        mode=mode, frames=frames, latent_frames=latent_frames, audio_latents=audio_latents, samples=samples, video_span=video,
        audio_span=audio, frame_span=_span_frames(g, video), sample_span=_span_samples(g, audio, audio_latents, samples),
        start_s=start, end_s=end,
    )


# ------------------------------------------------------------------ the customer's media (numpy, ffmpeg)


def decode_frames(path: str | Path, *, fps: float, width: int, height: int, count: int):
    """Up to `count` frames from the start of a clip as the job renders them, RGB uint8 [frames, height, width, 3]: at `fps`
    (ffmpeg's fps filter repeats or drops frames), scaled to cover width x height and centre-cropped, as
    LTX2ConditionPipeline fits a condition to the render. The frames are read straight into one array."""
    import numpy as np

    graph = f"fps={fps:g},scale={width}:{height}:force_original_aspect_ratio=increase:flags=bicubic,crop={width}:{height},setsar=1,format=rgb24"
    command = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", graph, "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    out = np.empty((count, height, width, 3), dtype=np.uint8)
    view = memoryview(out).cast("B")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    filled = 0
    try:
        while filled < len(view):
            read = process.stdout.readinto(view[filled:])
            if not read:
                break
            filled += read
    finally:
        process.stdout.close()
        code = process.wait(timeout=600)
    if code != 0:
        raise EditError(f"the source video could not be decoded (ffmpeg exit {code})")
    return out[: filled // (width * height * 3)]


def _read_wav(path: Path):
    """(channels, samples) int16 from a 16-bit PCM WAV, plain or WAVE_FORMAT_EXTENSIBLE (ffmpeg writes the latter for more
    than two channels, which Python's wave module reads only from 3.12)."""
    import struct

    import numpy as np

    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise EditError("the source audio could not be decoded (not a WAV)")
    position, channels, samples = 12, None, None
    while position + 8 <= len(data):
        kind, size = data[position : position + 4], struct.unpack("<I", data[position + 4 : position + 8])[0]
        body = position + 8
        if kind == b"fmt ":
            channels, bits = struct.unpack("<H", data[body + 2 : body + 4])[0], struct.unpack("<H", data[body + 14 : body + 16])[0]
            if bits != 16:
                raise EditError("the source audio could not be decoded (not 16-bit)")
        elif kind == b"data":
            size = min(size, len(data) - body)
            samples = np.frombuffer(data[body : body + size - size % 2], dtype="<i2")
            break
        position = body + size + size % 2
    if channels is None or samples is None:
        raise EditError("the source audio could not be decoded (no samples)")
    return samples[: len(samples) - len(samples) % channels].reshape(-1, channels).T.astype(np.int16)


def decode_audio(path: str | Path, *, sample_rate: int, start_s: float = 0.0, duration_s: float | None = None):
    """(2, samples) int16 at `sample_rate` from `start_s`, at most `duration_s` long, or None when the file has no audio
    stream. Mono is doubled into both channels, stereo kept sample for sample, and more channels downmixed by ffmpeg."""
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="kuno-edit-audio-") as tmp:
        out = Path(tmp) / "audio.wav"

        def run(channels: list[str]) -> subprocess.CompletedProcess:
            command = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
            if start_s:
                command += ["-ss", f"{start_s:.6f}"]
            command += ["-i", str(path), "-map", "0:a:0?", "-vn", "-sn", "-dn"]
            if duration_s is not None:
                command += ["-t", f"{duration_s:.6f}"]
            command += [*channels, "-c:a", "pcm_s16le", "-ar", str(int(sample_rate)), "-f", "wav", str(out)]
            return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600, check=False)

        done = run([])
        if done.returncode != 0:
            if b"does not contain any stream" in done.stderr:
                return None
            raise EditError(f"the source audio could not be decoded (ffmpeg exit {done.returncode})")
        audio = _read_wav(out)
        if audio.shape[0] > 2:
            done = run(["-ac", "2"])
            if done.returncode != 0:
                raise EditError(f"the source audio could not be downmixed (ffmpeg exit {done.returncode})")
            audio = _read_wav(out)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    return np.ascontiguousarray(audio)


def fit_samples(audio, length: int):
    """`audio` ((channels, samples)) cut, or padded with silence, to exactly `length` samples."""
    import numpy as np

    if audio.shape[1] >= length:
        return np.ascontiguousarray(audio[:, :length])
    return np.concatenate([audio, np.zeros((audio.shape[0], length - audio.shape[1]), dtype=audio.dtype)], axis=1)


def to_int16(audio):
    """Float samples in [-1, 1] as 16-bit, exactly as media_tools._write_wav converts them."""
    import numpy as np

    return (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16)


def splice_audio(source, render, span: tuple[int, int], fade: int):
    """The source's samples outside `span`, the render's inside it, with `fade` samples of crossfade just inside each end
    of the span that meets held sound. `source` is (2, n) int16; `render` (channels, m) float in [-1, 1]; the result is
    (2, n) int16, identical to `source` outside the span."""
    import numpy as np

    out = source.copy()
    s0, s1 = span
    if s1 <= s0:
        return out
    new = to_int16(fit_samples(np.asarray(render, dtype=np.float32), source.shape[1]))
    if new.shape[0] == 1:
        new = np.repeat(new, 2, axis=0)
    new = new[:2].astype(np.float32)
    held = source.astype(np.float32)
    piece = new[:, s0:s1].copy()
    n = min(fade, (s1 - s0) // 2)
    if n > 0:
        ramp = (np.arange(1, n + 1, dtype=np.float32) / (n + 1))[None]
        if s0 > 0:
            piece[:, :n] = held[:, s0 : s0 + n] * (1 - ramp) + piece[:, :n] * ramp
        if s1 < source.shape[1]:
            piece[:, -n:] = piece[:, -n:] * ramp[:, ::-1] + held[:, s1 - n : s1] * (1 - ramp[:, ::-1])
    out[:, s0:s1] = np.clip(np.rint(piece), -32768, 32767).astype(np.int16)
    return out


# ------------------------------------------------------------------ the render


def _token_positions(held: list[tuple[int, int]], per: int):
    """Sequence positions of the held ranges, `per` tokens a latent, ascending."""
    import torch

    return torch.cat([torch.arange(lo * per, hi * per, dtype=torch.long) for lo, hi in held])


def render_edit(renderer: Any, call: dict[str, Any], edit: dict[str, Any]) -> dict[str, Any]:
    """An audio-to-video or retake render on `renderer` (ltx_pinning.PinnedRenderer) for a `build_call` output and its edit
    spec: {"videos", "audio", "sampling_rate", "edit"}, where `audio` is the output sound track ((2, samples) int16, or the
    render's float samples when a retake's source has no sound) and `edit` what was held (counts and spans only)."""
    import numpy as np

    fps, frames = float(call["frame_rate"]), int(call["num_frames"])
    width, height = int(call["width"]), int(call["height"])
    g = renderer.geometry(fps)
    plan = plan_edit(g, frames, edit)
    timings: dict[str, float] = {}
    started = time.perf_counter()

    source_frames = None
    if plan.mode == RETAKE:
        sound = decode_audio(edit["video_path"], sample_rate=g.sample_rate, duration_s=frames / fps)
        if plan.held_video:
            source_frames = decode_frames(edit["video_path"], fps=fps, width=width, height=height, count=frames)
            decoded = len(source_frames)
            if decoded < frames - SOURCE_FRAME_SLACK or decoded == 0:
                raise EditError(f"the source video has {decoded} frames at {fps:g} fps; this retake renders {frames}")
            if decoded < frames:  # the 8k + 1 grid: hold the last frame over the few the source lacks
                source_frames = np.concatenate([source_frames, np.repeat(source_frames[-1:], frames - decoded, axis=0)])
    else:
        sound = decode_audio(edit["audio_path"], sample_rate=g.sample_rate, start_s=plan.start_s, duration_s=plan.end_s - plan.start_s)
        if sound is None or sound.shape[1] == 0:
            raise EditError("the source audio has no sound from its start_s")
    if sound is not None:
        sound = fit_samples(sound, plan.samples)
    # Audio-to-video holds every audio latent (its span is empty); a retake those outside its span, if its source has sound.
    held_audio = plan.held_audio if sound is not None else []
    timings["decode_s"] = round(time.perf_counter() - started, 3)

    started = time.perf_counter()
    stages = renderer.stages(call)
    audio_tokens = audio_at = None
    if held_audio:
        encoded = renderer.encode_audio(sound.astype(np.float32) / 32768.0, g.sample_rate, plan.audio_latents)
        audio_at = _token_positions(held_audio, encoded.shape[1] // plan.audio_latents)
        audio_tokens = encoded[:, audio_at.to(encoded.device)].contiguous()
        del encoded
    pins: dict[str, Pins] = {}
    for stage in stages:
        video_tokens = video_at = None
        if source_frames is not None:
            stage_width, stage_height = renderer.stage_size(stage, call)
            encoded = renderer.encode_video(source_frames, stage_width, stage_height)
            per = encoded.shape[1] // plan.latent_frames
            if per * plan.latent_frames != encoded.shape[1]:
                raise EditError(f"the source encoded to {encoded.shape[1]} tokens, not {plan.latent_frames} latent frames of equal size")
            video_at = _token_positions(plan.held_video, per)
            video_tokens = encoded[:, video_at.to(encoded.device)].contiguous()
            del encoded
        pins[stage] = Pins(video=video_tokens, audio=audio_tokens, video_at=video_at, audio_at=audio_at)
    del source_frames
    timings["encode_s"] = round(time.perf_counter() - started, 3)

    rendered = renderer.render(call, pins)
    failure = pin_failure(stages, pins, rendered)
    if failure is not None:
        raise EditError(failure)
    if len(rendered.frames) != frames:
        raise EditError(f"the render decoded {len(rendered.frames)} frames, not the {frames} planned")
    if plan.mode == AUDIO_TO_VIDEO:
        track = sound
    elif sound is None:
        track = rendered.audio
    elif plan.sample_span[1] <= plan.sample_span[0]:
        track = sound
    else:
        if rendered.audio is None or rendered.sample_rate != g.sample_rate:
            raise EditError(f"the render came back without audio at the vocoder's {g.sample_rate} Hz")
        track = splice_audio(sound, rendered.audio, plan.sample_span, round(EDIT_CROSSFADE_S * g.sample_rate))
    report = {
        "mode": plan.mode, "frames": frames, "latent_frames": plan.latent_frames, "audio_latents": plan.audio_latents,
        "window_s": [plan.start_s, plan.end_s], "regenerated_latent_frames": list(plan.video_span), "regenerated_frames": list(plan.frame_span),
        "held_latent_frames": sum(hi - lo for lo, hi in plan.held_video),
        "held_audio_latents": sum(hi - lo for lo, hi in held_audio), "regenerated_samples": list(plan.sample_span) if sound is not None else None,
        "source_audio": sound is not None, "pins_exact": rendered.pins_exact, "seen_clean": rendered.seen_clean,
        "timings": {**timings, **rendered.timings},
    }
    return {"videos": rendered.frames, "audio": track, "sampling_rate": g.sample_rate, "edit": report}
