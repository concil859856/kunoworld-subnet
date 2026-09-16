"""Storyboards on LTX-2.5: one long video from shots rendered one after another and joined inside the enclave.

PROTOCOL.md, "Storyboards", is the contract. Every shot is an ordinary text-to-video call (`ltx_resident.build_call`). A
joined shot holds the tail of an earlier shot's final latents fixed at the head of its own, the AV-extend technique of
ComfyUI-JoyLTX25 (MIT):

  continue  the previous shot's last `overlap` video latent frames, and the matching audio latents: one unbroken take
  cut       the audio latents only: a new picture over the same voice and room tone
  fresh     nothing

The held head is trimmed from the shot's decoded frames and samples before it joins the stitched video. The latents
never leave the process, so a storyboard can't be assembled from separate jobs.

How the pin is held, from diffusers 0.40 (pipelines/ltx2/pipeline_ltx2_condition.py, transformer_ltx2.py):
  video  LTX2ConditionPipeline's native conditioning. prepare_latents returns a per-token conditioning_mask; the loop
         passes the transformer `timestep * (1 - mask)`, so masked tokens are clean context at t = 0, and blends
         x0 = denoised * (1 - mask) + clean * mask, so their Euler step is exactly zero. It is how the pipeline holds an
         image-to-video first frame; LTX2ExtendPipeline puts an earlier shot's tokens there instead of encoded pixels.
  audio  no pipeline has an audio mask, but the transformer accepts `audio_timestep` per token. A forward pre-hook makes
         it `t * (1 - audio_mask)`, prepare_audio_latents writes the tail, and since nothing blends the audio x0, the
         `audio_scheduler` (a PinnedScheduler) writes the pinned tokens back after every step.
  both   PinnedScheduler keeps each pass's final tokens: the pipeline's own normalized, packed space, so nothing is
         decoded, re-encoded or re-normalized between shots. The distilled recipe's two passes (half size, x2 latent
         upsampler, refine; runtimes.LtxAdapter._two_stage) are each pinned from the same pass of the earlier shot, so
         the refine can't redraw the join.

Geometry, read from the loaded pipeline (these are LTX-2.5's): the video VAE is causal, 8x in time and 32x in space, so
n latent frames decode to 1 + 8(n - 1) frames. Audio is 16 kHz mel at hop 160, 4x in time (25 latents a second); the
audio VAE is causal too (n latents decode to 4n - 3 mel frames), and the vocoder gives 480 samples per mel frame at
48 kHz.

Plain data first (Geometry, Timeline, assemble_audio, StitchWriter, render_storyboard: no torch), then the torch parts,
imported only when a real storyboard renders.

Status: the experiment this is ported from rendered a seamless 35.375 s take from 8 x 5 s shots on an RTX PRO 6000
(dev repo, research/long-video_ltx-av-extend_2026-09-16.md). This module has run on the CPU only, against diffusers'
real classes with tiny random weights.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from .base import ProgressFn
from .media_tools import BackendError, _write_wav, ffmpeg_exe

log = logging.getLogger("kuno.worker.storyboard")

JOINS = ("fresh", "continue", "cut")
# Where a joined shot's audio pin comes from. The protocol fixes `first`: the first shot of the current run of joined
# shots, which keeps a voice from drifting across many joins. `previous` (the shot just before, so the sound runs straight
# through every seam) is the experiment's other setting, kept so the planner is tested under both.
ANCHORS = ("first", "previous")
# A fade in and out this long where a join's sound can't run through (fresh, or an anchored pin): no click.
DECLICK_S = 0.005
# The share of the backend's progress the shots take; encoding the stitched video follows.
SHOTS_FROM, SHOTS_TO = 0.05, 0.85


class StoryboardError(BackendError):
    """A storyboard can't be planned or rendered as asked. Messages carry counts only, never request content."""


# ------------------------------------------------------------------ geometry (plain data)


@dataclass(frozen=True)
class Geometry:
    """Where latents sit in time. The defaults are LTX-2.5's; `from_pipeline` reads a loaded pipeline."""

    fps: float = 24.0
    temporal_ratio: int = 8  # video VAE: frames per latent frame after the first (the first latent frame is one frame)
    spatial_ratio: int = 32
    mel_sample_rate: int = 16000
    mel_hop: int = 160
    audio_ratio: int = 4  # audio VAE: mel frames per latent after the first
    sample_rate: int = 48000  # the vocoder's output

    @classmethod
    def from_pipeline(cls, pipeline: Any, fps: float) -> Geometry:
        config = getattr(getattr(pipeline, "vocoder", None), "config", None)
        return cls(
            fps=float(fps),
            temporal_ratio=int(pipeline.vae_temporal_compression_ratio),
            spatial_ratio=int(pipeline.vae_spatial_compression_ratio),
            mel_sample_rate=int(pipeline.audio_sampling_rate),
            mel_hop=int(pipeline.audio_hop_length),
            audio_ratio=int(pipeline.audio_vae_temporal_compression_ratio),
            sample_rate=int(getattr(config, "output_sampling_rate", None) or 48000),
        )

    @property
    def audio_latents_per_second(self) -> float:
        return self.mel_sample_rate / self.mel_hop / float(self.audio_ratio)

    @property
    def samples_per_mel(self) -> Fraction:
        return Fraction(self.mel_hop * self.sample_rate, self.mel_sample_rate)

    @property
    def samples_per_frame(self) -> Fraction:
        return Fraction(self.sample_rate) / Fraction(self.fps).limit_denominator(1001)

    def latent_frames(self, frames: int) -> int:
        return (frames - 1) // self.temporal_ratio + 1

    def pixel_frames(self, latent_frames: int) -> int:
        """Frames `latent_frames` decode to when they start a clip: the causal first one is a single frame."""
        return 1 + self.temporal_ratio * (latent_frames - 1) if latent_frames > 0 else 0

    def audio_latents(self, frames: int) -> int:
        # The pipelines' own arithmetic, float then round, so the count is always theirs.
        return round(frames / float(self.fps) * self.audio_latents_per_second)

    def mel_frames(self, audio_latents: int) -> int:
        return self.audio_ratio * audio_latents - (self.audio_ratio - 1) if audio_latents > 0 else 0

    def audio_samples(self, audio_latents: int) -> int:
        return round(self.mel_frames(audio_latents) * self.samples_per_mel)


# ------------------------------------------------------------------ the join plan (plain data)


@dataclass
class Join:
    """How one shot attaches to the shots before it, and what of it the stitched video keeps. Counts are the shot's own;
    `video_start` and `audio_start` are positions in the stitched output."""

    index: int
    join: str
    frames: int
    latent_frames: int
    audio_latents: int
    video_pin: int = 0  # latent frames of video_source's tail held at the head
    video_source: int | None = None
    audio_pin: int = 0  # audio latents of audio_source's tail held at the head
    audio_source: int | None = None
    audio_continuous: bool = False  # the audio pin is the previous shot's own tail: the sound runs straight through
    video_trim: int = 0  # decoded frames dropped from the head
    audio_trim: int = 0  # decoded samples dropped from the head
    video_start: int = 0
    audio_start: int = 0
    decoded_frames: int | None = None
    decoded_samples: int | None = None
    audio_keep: int | None = None  # samples kept from audio_trim on (past the decoded end: silence); set by the next join
    video_zero: Fraction = Fraction(0)  # where this shot's own frame 0 falls in the output, in samples
    sync_error_samples: float = 0.0  # this shot's audio clock start minus its video clock start, in the output
    notes: list[str] = field(default_factory=list)

    @property
    def kept_frames(self) -> int | None:
        return None if self.decoded_frames is None else self.decoded_frames - self.video_trim

    @property
    def audio_pad(self) -> int | None:
        """Silence appended after the decoded audio (negative: samples cut from its end)."""
        if self.audio_keep is None or self.decoded_samples is None:
            return None
        return self.audio_keep - (self.decoded_samples - self.audio_trim)


class Timeline:
    """Plans every join from what the shots before it actually decoded to.

    Video: a continue or cut join drops the frames its first `overlap` latent frames decode to, 1 + 8(overlap - 1). For
    continue those are the pinned ones, so the first kept frame is the one right after the previous shot's last. Cut drops
    the same span: those frames were drawn under the previous shot's replayed sound.

    Audio: k pinned latents decode to 4k - 3 mel frames, exactly what is trimmed, so a shot whose pin is the previous
    shot's own tail continues its sound sample for sample. Only k is a choice. A tail proportional to the overlap
    (JoyLTX25: round(audio_latents * overlap / latent_frames)) ignores that both VAEs are causal: for 2 s shots at 24 fps
    it replays about 140 ms of sound and shifts the pinned sound 170 ms against the pinned picture. Here k is the count
    that best lines the shot's audio clock up with its video clock in the output, given every earlier choice, so the error
    stays within half an audio latent (20 ms) and never adds up. Where the sound can't run through (a fresh shot, or a pin
    anchored to an earlier shot) the previous shot's audio is padded with silence or cut at its end so the next shot
    starts in sync.
    """

    def __init__(self, geometry: Geometry, overlap: int, anchor: str = "first"):
        if anchor not in ANCHORS:
            raise StoryboardError(f"audio anchor must be one of {', '.join(ANCHORS)}, not {anchor!r}")
        if overlap < 1:
            raise StoryboardError("overlap must be at least one latent frame")
        self.geometry, self.overlap, self.anchor = geometry, overlap, anchor
        self.joins: list[Join] = []
        self.total_frames: int | None = None
        self.total_samples: int | None = None

    def chain_start(self, index: int) -> int:
        """The shot that began the current run of joined shots: the latest fresh one before `index`."""
        return max(j.index for j in self.joins[:index] if j.join == "fresh")

    def plan(self, join_mode: str, frames: int) -> Join:
        g, index = self.geometry, len(self.joins)
        if join_mode not in JOINS:
            raise StoryboardError(f"shot {index + 1}: unknown join {join_mode!r}")
        join = Join(index=index, join=join_mode, frames=frames, latent_frames=g.latent_frames(frames), audio_latents=g.audio_latents(frames))
        if index == 0:
            if join_mode != "fresh":
                raise StoryboardError("shot 1 has nothing before it to join; its join must be fresh")
            self.joins.append(join)
            return join
        prev = self.joins[-1]
        if prev.decoded_frames is None or prev.decoded_samples is None:
            raise RuntimeError(f"shot {index + 1} is planned from what shot {index} decoded to; render shot {index} first")
        join.video_start = prev.video_start + prev.kept_frames
        if join_mode != "fresh":
            if not 1 <= self.overlap < min(join.latent_frames, prev.latent_frames):
                raise StoryboardError(
                    f"shot {index + 1}: an overlap of {self.overlap} latent frames needs shots of at least {self.overlap + 1} latent "
                    f"frames ({g.pixel_frames(self.overlap + 1)} frames); shots {index} and {index + 1} have {prev.latent_frames} and "
                    f"{join.latent_frames}"
                )
            join.video_trim = g.pixel_frames(self.overlap)
        if join_mode == "continue":
            join.video_pin, join.video_source = self.overlap, index - 1
        join.video_zero = (join.video_start - join.video_trim) * g.samples_per_frame
        if join_mode != "fresh":
            source = index - 1 if self.anchor == "previous" else self.chain_start(index)
            join.audio_source, join.audio_continuous = source, source == index - 1
            limit = min(self.joins[source].audio_latents, join.audio_latents) - 1
            if join.audio_continuous:
                start = prev.audio_start + (prev.decoded_samples - prev.audio_trim)
                join.audio_pin = self._audio_pin(start - join.video_zero, limit, index)
                miss = abs(start - g.audio_samples(join.audio_pin) - join.video_zero)
                if miss > g.audio_ratio * g.samples_per_mel / 2:
                    # Even the shortest pin starts the sound too early: the previous shot's audio falls short of its
                    # picture by more than this overlap trims (a 1-latent-frame overlap at 50 fps). Running it through
                    # would drift further at every join, so this join pads into sync instead.
                    join.audio_continuous = False
                    join.notes.append(
                        f"an overlap of {self.overlap} trims {join.video_trim} frames, less than shot {index}'s audio falls short of its "
                        f"picture: its sound is padded into sync instead of running through ({1000 * float(miss) / g.sample_rate:.0f} ms off otherwise)"
                    )
            if not join.audio_continuous:
                # The pinned sound spans the time of the trimmed frames.
                join.audio_pin = self._audio_pin(join.video_trim * g.samples_per_frame, limit, index)
            join.audio_trim = g.audio_samples(join.audio_pin)
        self.joins.append(join)
        self._place_audio(join)
        return join

    def _audio_pin(self, target: Fraction, limit: int, index: int) -> int:
        """The audio latent count whose decoded span is closest to `target` samples."""
        g = self.geometry
        if limit < 1:
            raise StoryboardError(f"shot {index + 1}: too short to hold an audio pin")
        guess = round((target / g.samples_per_mel + (g.audio_ratio - 1)) / g.audio_ratio)
        candidates = {max(1, min(limit, guess + d)) for d in (-1, 0, 1)}
        return min(candidates, key=lambda k: (abs(g.audio_samples(k) - target), k))

    def _place_audio(self, join: Join) -> None:
        if join.index == 0:
            return
        prev = self.joins[join.index - 1]
        if join.audio_continuous:
            prev.audio_keep = prev.decoded_samples - prev.audio_trim  # all of it: this shot's first kept sample follows its last
            join.audio_start = prev.audio_start + prev.audio_keep
        else:
            join.audio_start = round(join.video_zero + join.audio_trim)
            prev.audio_keep = join.audio_start - prev.audio_start
        join.sync_error_samples = float(join.audio_start - join.audio_trim - join.video_zero)

    def rendered(self, index: int, frames: int, samples: int) -> Join:
        """Records what shot `index` decoded to. The plan assumes LTX-2.5's causal audio decoder; if the samples differ,
        the trim becomes the same share of what did come back and the difference is noted."""
        g, join = self.geometry, self.joins[index]
        join.decoded_frames, join.decoded_samples = int(frames), int(samples)
        if frames != join.frames:
            join.notes.append(f"decoded {frames} frames, planned {join.frames}")
        if frames <= join.video_trim:
            raise StoryboardError(f"shot {index + 1} decoded {frames} frames, no more than the {join.video_trim} its join trims")
        predicted = g.audio_samples(join.audio_latents)
        if samples != predicted:
            join.notes.append(f"decoded {samples} audio samples, the causal decoder rule predicts {predicted}")
            if join.audio_pin:
                join.audio_trim = round(samples * g.mel_frames(join.audio_pin) / g.mel_frames(join.audio_latents))
                self._place_audio(join)
        return join

    def finish(self) -> tuple[int, int]:
        """Closes the last shot's audio at the end of the video: (stitched frames, stitched samples)."""
        last = self.joins[-1]
        if last.decoded_frames is None:
            raise RuntimeError("finish() needs every shot rendered")
        self.total_frames = last.video_start + last.kept_frames
        self.total_samples = round(self.total_frames * self.geometry.samples_per_frame)
        last.audio_keep = self.total_samples - last.audio_start
        return self.total_frames, self.total_samples


def predicted_timeline(geometry: Geometry, joins: list[str], frames: list[int], overlap: int, anchor: str = "first") -> Timeline:
    """The whole plan up front, assuming every shot decodes to what the geometry predicts (kuno-plan, tests)."""
    timeline = Timeline(geometry, overlap, anchor)
    for join_mode, count in zip(joins, frames):
        join = timeline.plan(join_mode, count)
        timeline.rendered(join.index, count, geometry.audio_samples(join.audio_latents))
    timeline.finish()
    return timeline


def assemble_audio(joins: list[Join], audios: list, total_samples: int, fade_samples: int = 0):
    """The stitched track: each shot's kept samples at its audio_start, silence where a join pads. `audios` are
    (channels, samples) arrays. Where a join's sound doesn't run through, `fade_samples` ramps the sound out before it and
    in after it, so two unrelated waveforms don't click; a continuous join is never touched."""
    import numpy as np

    channels = max(a.shape[0] for a in audios)
    out = np.zeros((channels, total_samples), dtype=np.float32)
    for index, (join, audio) in enumerate(zip(joins, audios)):
        keep = max(0, join.audio_keep or 0)
        piece = audio[:, join.audio_trim : join.audio_trim + keep]
        if fade_samples and piece.shape[1]:
            n = min(fade_samples, piece.shape[1])
            ramp = (np.arange(1, n + 1, dtype=np.float32) / (n + 1))[None]
            if index > 0 and not join.audio_continuous:
                piece = piece.copy()
                piece[:, :n] *= ramp
            if index + 1 < len(joins) and not joins[index + 1].audio_continuous:
                piece = piece.copy()
                piece[:, -n:] *= ramp[:, ::-1]
        end = min(total_samples, join.audio_start + piece.shape[1])
        if end > join.audio_start:
            out[: piece.shape[0], join.audio_start : end] = piece[:, : end - join.audio_start]
    return out


def audio_array(audio: Any):
    """(channels, samples) float32 from what a pipeline returns: diffusers' vocoder gives a bfloat16 tensor on the GPU."""
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().to("cpu").float().numpy()
    array = np.asarray(audio, dtype=np.float32)
    while array.ndim > 2:
        array = array[0]
    if array.ndim == 1:
        array = array[None]
    if array.shape[0] > array.shape[1]:
        array = array.T
    return np.ascontiguousarray(array)


# ------------------------------------------------------------------ pins and tails


@dataclass
class Pins:
    """Tokens held at the head of one pass: [1, tokens, features], in the pipeline's normalized, packed space."""

    video: Any = None
    audio: Any = None


@dataclass
class Tail:
    """The end of one shot's latents after one pass, kept on the CPU for later shots: the last `overlap` video latent
    frames, and all of the audio (each join takes the count it planned)."""

    video: Any
    audio: Any


def pins_for(join: Join, tails: dict[int, dict[str, Tail]], stage: str) -> Pins:
    video = tails[join.video_source][stage].video if join.video_pin else None
    audio = tails[join.audio_source][stage].audio[:, -join.audio_pin :] if join.audio_pin else None
    return Pins(video=video, audio=audio)


@dataclass
class Rendered:
    frames: Any  # the shot's frames in order: PIL images or HxWx3 uint8 arrays
    audio: Any  # (channels, samples) float32
    sample_rate: int
    tails: dict[str, Tail]  # per pass
    pins_exact: dict[str, bool | None]  # per pass: the pinned tokens came out of denoising bit-identical
    # per pass and modality: the transformer's first call saw the pinned head at timestep 0 and the rest above it
    seen_clean: dict[str, dict[str, bool | None]]
    timings: dict[str, float] = field(default_factory=dict)


class Renderer(Protocol):
    def geometry(self, fps: float) -> Geometry: ...

    def stages(self, call: dict[str, Any]) -> tuple[str, ...]: ...

    def render(self, call: dict[str, Any], pins: dict[str, Pins]) -> Rendered: ...


# ------------------------------------------------------------------ the stitched video


class StitchWriter:
    """The stitched MP4, encoded as shots arrive (a 120 s take doesn't fit in memory as raw frames), with the sound muxed
    in at the end, once every join has placed it. The same H.264 settings as media_tools.encode_video."""

    def __init__(self, directory: Path, fps: float, width: int, height: int, crf: int = 18):
        self.directory, self.size, self.frames = Path(directory), (width, height), 0
        self.video_only = self.directory / "stitched-video.mp4"
        self.process = subprocess.Popen(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(self.video_only)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def add(self, frames) -> None:
        import numpy as np

        width, height = self.size
        for frame in frames:
            array = np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame, dtype=np.uint8)
            if array.shape != (height, width, 3):
                raise StoryboardError(f"a shot's frame is {array.shape}, the stitched video is {width}x{height}")
            try:
                self.process.stdin.write(np.ascontiguousarray(array).tobytes())
            except BrokenPipeError:
                raise StoryboardError(f"encoding the stitched video failed (ffmpeg exit {self.process.wait()})") from None
            self.frames += 1

    def finish(self, audio, sample_rate: int) -> bytes:
        """The MP4: the video as encoded, with `audio` ((channels, samples)) muxed in unless it is None."""
        self.process.stdin.close()
        if self.process.wait(timeout=3600) != 0:
            raise StoryboardError(f"encoding the stitched video failed (ffmpeg exit {self.process.returncode})")
        if audio is None:
            return self.video_only.read_bytes()
        wav, out = self.directory / "stitched.wav", self.directory / "stitched.mp4"
        _write_wav(wav, audio, sample_rate)
        mux = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(self.video_only), "-i", str(wav), "-map", "0:v:0",
             "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600,
        )
        if mux.returncode != 0:
            raise StoryboardError(f"muxing the stitched video's audio failed (ffmpeg exit {mux.returncode})")
        return out.read_bytes()

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()


def render_storyboard(
    renderer: Renderer,
    joins: list[str],
    calls: list[dict[str, Any]],
    directory: Path,
    progress: ProgressFn,
    *,
    overlap: int,
    audio: bool = True,
) -> tuple[bytes, int]:
    """Renders every shot in order, each pinned from the tails of the shots before it, streams its kept frames into the
    stitched video and returns (MP4, frames). `calls` are the shots' `build_call` outputs, `joins` their joins."""
    count = len(calls)
    fps = float(calls[0]["frame_rate"])
    geometry = renderer.geometry(fps)
    timeline = Timeline(geometry, overlap)
    tails: dict[int, dict[str, Tail]] = {}
    sounds: list[Any] = []
    writer = StitchWriter(directory, fps, calls[0]["width"], calls[0]["height"])
    try:
        for index, (join_mode, call) in enumerate(zip(joins, calls)):
            # Between shots: the worker's progress raises once the job is canceled, so a cancel stops the storyboard here.
            progress(SHOTS_FROM + (SHOTS_TO - SHOTS_FROM) * index / count, f"shot {index + 1}/{count}")
            started = time.perf_counter()
            join = timeline.plan(join_mode, int(call["num_frames"]))
            stages = renderer.stages(call)
            pins = {stage: pins_for(join, tails, stage) for stage in stages}
            rendered = renderer.render(call, pins)
            _check_pins(index, stages, pins, rendered)
            if len(rendered.frames) != join.frames:
                # Every later join, and the stitched length the customer pays for, assume the planned count.
                raise StoryboardError(f"shot {index + 1} decoded {len(rendered.frames)} frames, not the {join.frames} planned")
            if rendered.audio is None or rendered.sample_rate != geometry.sample_rate:
                raise StoryboardError(f"shot {index + 1} came back without audio at the vocoder's {geometry.sample_rate} Hz")
            timeline.rendered(index, len(rendered.frames), rendered.audio.shape[1])
            tails[index] = rendered.tails
            writer.add(rendered.frames[join.video_trim :])
            sounds.append(rendered.audio)
            for note in join.notes:  # counts only
                log.warning("storyboard shot %d/%d: %s", index + 1, count, note)
            log.info(
                "storyboard shot %d/%d (%s, %d frames, video pin %d, audio pin %d) rendered in %.1fs %s",
                index + 1, count, join_mode, join.frames, join.video_pin, join.audio_pin, time.perf_counter() - started, rendered.timings,
            )
        progress(0.9, "encoding")
        total_frames, total_samples = timeline.finish()
        track = None
        if audio:
            track = assemble_audio(timeline.joins, sounds, total_samples, fade_samples=round(geometry.sample_rate * DECLICK_S))
        data = writer.finish(track, geometry.sample_rate)
    except BaseException:
        writer.abort()
        raise
    if writer.frames != total_frames:
        raise StoryboardError(f"the stitched video has {writer.frames} frames; its joins make {total_frames}")
    return data, writer.frames


def _check_pins(index: int, stages: tuple[str, ...], pins: dict[str, Pins], rendered: Rendered) -> None:
    """A join whose pinned tokens moved, or that the model didn't see as clean context, can show at the seam. It means the
    runtime changed under this module (a diffusers upgrade), so the job fails rather than deliver it."""
    for stage in stages:
        pin = pins[stage]
        if (pin.video is not None or pin.audio is not None) and rendered.pins_exact.get(stage) is not True:
            raise StoryboardError(f"shot {index + 1}: the pinned {stage}-pass tokens changed during denoising")
        for modality, head in (("video", pin.video), ("audio", pin.audio)):
            if head is not None and (rendered.seen_clean.get(stage) or {}).get(modality) is not True:
                raise StoryboardError(f"shot {index + 1}: the model was not shown the pinned {modality} head at timestep 0 in the {stage} pass")


# ------------------------------------------------------------------ torch


def pinned_timesteps(timestep: Any, tokens: int, head: int):
    """Per-token timesteps for one modality: 0 on the pinned head, so the transformer reads it as clean context."""
    import torch

    mask = torch.zeros(tokens, device=timestep.device, dtype=timestep.dtype)
    mask[:head] = 1
    return timestep.reshape(-1, 1) * (1 - mask)[None]


def take_tail(video_tokens: Any, audio_tokens: Any, latent_frames: int, tokens_per_frame: int, overlap: int) -> Tail:
    if video_tokens.shape[1] != latent_frames * tokens_per_frame:
        raise StoryboardError(
            f"the final video latents have {video_tokens.shape[1]} tokens, not {latent_frames} latent frames x {tokens_per_frame}: "
            "the pipeline's patching is not what the storyboard assumes"
        )
    keep = min(overlap, latent_frames)
    video = video_tokens[:, (latent_frames - keep) * tokens_per_frame :].detach().to("cpu", copy=True)
    return Tail(video=video, audio=audio_tokens.detach().to("cpu", copy=True))


def head_matches(tokens: Any, head: Any) -> bool | None:
    import torch

    if head is None:
        return None
    return bool(torch.equal(tokens[:, : head.shape[1]].to("cpu"), head.to("cpu", tokens.dtype)))


def clean_head(timestep: Any, head: Any) -> bool | None:
    """Whether per-token timesteps show the transformer the pinned head at t = 0 and everything after it as noise."""
    if head is None:
        return None
    if timestep is None or timestep.ndim != 2:
        return False
    n = head.shape[1]
    return bool((timestep[:, :n] == 0).all()) and bool((timestep[:, n:] > 0).all())


def all_exact(*checks: bool | None) -> bool | None:
    present = [c for c in checks if c is not None]
    return all(present) if present else None


_EXTEND_CLASSES: tuple | None = None


def extend_classes() -> tuple:
    """PinnedScheduler and LTX2ExtendPipeline, defined on first use so the plain-data parts import without diffusers."""
    global _EXTEND_CLASSES
    if _EXTEND_CLASSES is not None:
        return _EXTEND_CLASSES
    from diffusers import FlowMatchEulerDiscreteScheduler, LTX2ConditionPipeline
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput

    class PinnedScheduler(FlowMatchEulerDiscreteScheduler):
        """The Euler step, then the pinned head written back exactly. Keeps the pass's final tokens in `final`."""

        pin = None
        final = None

        def start(self, pin: Any) -> None:
            self.pin, self.final = pin, None

        def step(self, model_output, timestep, sample, *args, return_dict: bool = True, **kwargs):
            prev = super().step(model_output, timestep, sample, *args, return_dict=False, **kwargs)[0]
            if self.pin is not None:
                prev[:, : self.pin.shape[1]] = self.pin.to(prev.device, prev.dtype)
            if self.step_index is not None and self.step_index >= len(self.timesteps):
                self.final = prev.detach().clone()
            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev) if return_dict else (prev,)

    class LTX2ExtendPipeline(LTX2ConditionPipeline):
        """LTX2ConditionPipeline with `pins` written at the head, where it would put an encoded first frame."""

        pins: Pins | None = None

        def prepare_latents(self, *args, **kwargs):
            latents, mask, clean, keyframe_coords = super().prepare_latents(*args, **kwargs)
            head = getattr(self.pins, "video", None)
            if head is not None:
                n = head.shape[1]
                head = head.to(latents.device, latents.dtype)
                # mask 1: the loop's timestep * (1 - mask) shows these to the transformer at t = 0, and its x0 blend keeps
                # them, so the Euler step moves them by exactly nothing.
                latents[:, :n] = head
                clean[:, :n] = head
                mask[:, :n] = 1.0
            return latents, mask, clean, keyframe_coords

        def prepare_audio_latents(self, *args, **kwargs):
            latents = super().prepare_audio_latents(*args, **kwargs)
            head = getattr(self.pins, "audio", None)
            if head is not None:
                latents[:, : head.shape[1]] = head.to(latents.device, latents.dtype)
            return latents

    _EXTEND_CLASSES = (PinnedScheduler, LTX2ExtendPipeline)
    return _EXTEND_CLASSES


def _clock(cuda: bool) -> float:
    if cuda:
        import torch

        torch.cuda.synchronize()
    return time.perf_counter()


class ExtendRenderer:
    """One shot at a time on the pipelines the worker's loader built (runtimes.ltx_loader), through an LTX2ExtendPipeline
    that shares their loaded modules: no second copy of the weights."""

    def __init__(self, pipelines: dict[str, Any], device: str, overlap: int):
        Scheduler, Pipeline = extend_classes()
        source = pipelines.get("condition") or pipelines["text"]
        components = dict(source.components)
        # Its own schedulers, so the pins never touch the pipelines other jobs use. from_config keeps the recipe's schedule
        # (build_ltx_pipelines gives the full model dynamic shifting).
        components["scheduler"] = Scheduler.from_config(source.scheduler.config)
        components["audio_scheduler"] = Scheduler.from_config(source.scheduler.config)
        self.pipeline = Pipeline(**components)
        self.upsample = pipelines.get("upsample")
        self.device = device
        self.overlap = overlap

    def geometry(self, fps: float) -> Geometry:
        return Geometry.from_pipeline(self.pipeline, fps)

    def stages(self, call: dict[str, Any]) -> tuple[str, ...]:
        return ("half", "full") if call.get("second_stage_sigmas") and self.upsample is not None else ("full",)

    def prepare_call(self, call: dict[str, Any]) -> dict[str, Any]:
        """The pipeline's keyword arguments; the CPU tests replace the text encoder here."""
        return call

    def render(self, call: dict[str, Any], pins: dict[str, Pins]) -> Rendered:
        import torch

        from .resident import PipelineResult
        from .runtimes import _audio_rate

        call = dict(call)
        for key in ("pipeline", "kuno_trajectory_tap", "conditions", "generate_audio"):  # as runtimes.LtxAdapter does
            call.pop(key, None)
        generator = torch.Generator(device=self.device).manual_seed(int(call.pop("seed")))
        second = call.pop("second_stage_sigmas", None)
        call = self.prepare_call(call)
        cuda = str(self.device).startswith("cuda") and torch.cuda.is_available()
        record: dict[str, dict] = {"timings": {}, "tails": {}, "exact": {}, "seen": {}}
        width, height = call.pop("width"), call.pop("height")
        if second and self.upsample is not None:
            # runtimes.LtxAdapter._two_stage, with the pins held in both passes.
            latents, audio_latents = self._pass(
                "half", call, pins, generator, record, cuda, width=width // 2, height=height // 2, output_type="latent", return_dict=False,
            )
            started = _clock(cuda)
            upsampled = self.upsample(latents=latents, output_type="latent", return_dict=False)[0]
            record["timings"]["upsample_s"] = round(_clock(cuda) - started, 3)
            result = self._pass(
                "full", {**call, "sigmas": second}, pins, generator, record, cuda, width=width, height=height,
                latents=upsampled, audio_latents=audio_latents, noise_scale=second[0],
            )
        else:
            result = self._pass("full", call, pins, generator, record, cuda, width=width, height=height)
        normalized = PipelineResult.from_pipeline(
            {"videos": result.frames, "audio": result.audio, "sampling_rate": _audio_rate(self.pipeline, result)}
        )
        audio = None if normalized.audio is None else audio_array(normalized.audio)
        return Rendered(normalized.frames, audio, normalized.sample_rate, record["tails"], record["exact"], record["seen"], record["timings"])

    def _pass(self, stage: str, call: dict[str, Any], pins: dict[str, Pins], generator: Any, record: dict, cuda: bool, **extra) -> Any:
        pipeline = self.pipeline
        pin = pins.get(stage) or Pins()
        pin = Pins(*(None if t is None else t.to(self.device) for t in (pin.video, pin.audio)))  # once, not every step
        pipeline.pins = pin
        pipeline.scheduler.start(pin.video)
        pipeline.audio_scheduler.start(pin.audio)
        seen: dict[str, bool | None] = {}

        def timesteps(module, args, kwargs):
            audio_t, audio_tokens = kwargs.get("audio_timestep"), kwargs.get("audio_hidden_states")
            if pin.audio is not None and audio_t is not None and audio_t.ndim == 1 and audio_tokens is not None:
                # One timestep per batch row becomes one per audio token, 0 on the pinned head. audio_sigma (prompt AdaLN,
                # cross-modal modulation) stays per row, as it does for video.
                kwargs["audio_timestep"] = audio_t = pinned_timesteps(audio_t, audio_tokens.shape[1], pin.audio.shape[1])
            if not seen:  # the pass's first transformer call: what the model is actually shown
                seen.update(video=clean_head(kwargs.get("timestep"), pin.video), audio=clean_head(audio_t, pin.audio))
            return args, kwargs

        hook = pipeline.transformer.register_forward_pre_hook(timesteps, with_kwargs=True)
        started = _clock(cuda)
        try:
            output = pipeline(generator=generator, **call, **extra)
            video, audio = pipeline.scheduler.final, pipeline.audio_scheduler.final
        finally:
            hook.remove()
            pipeline.pins = None
            pipeline.scheduler.start(None)
            pipeline.audio_scheduler.start(None)
        record["timings"][f"{stage}_s"] = round(_clock(cuda) - started, 3)
        if video is None or audio is None:
            raise StoryboardError(f"the {stage} pass finished without a final scheduler step")
        record["seen"][stage] = seen
        record["exact"][stage] = all_exact(head_matches(video, pin.video), head_matches(audio, pin.audio))
        p = int(pipeline.transformer_spatial_patch_size)
        ratio = pipeline.vae_spatial_compression_ratio
        tokens_per_frame = (extra["height"] // ratio // p) * (extra["width"] // ratio // p)
        latent_frames = (call["num_frames"] - 1) // pipeline.vae_temporal_compression_ratio + 1
        record["tails"][stage] = take_tail(video, audio, latent_frames, tokens_per_frame, self.overlap)
        return output
