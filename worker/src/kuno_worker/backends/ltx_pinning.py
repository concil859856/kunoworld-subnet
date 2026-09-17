"""Tokens held fixed while LTX-2.5 denoises the rest: the one mechanism storyboards, audio-to-video and retake render with.

  storyboards     (ltx_storyboard.py) the tail of an earlier shot's final latents, held at the head of the next shot
  audio_to_video  (ltx_edit.py) every audio token, encoded from the customer's sound; the picture is generated to match
  retake          (ltx_edit.py) the video and audio tokens outside a time window, encoded from the customer's clip

How a token is held, from diffusers 0.40 (pipelines/ltx2/pipeline_ltx2_condition.py, transformer_ltx2.py):
  video  LTX2ConditionPipeline's native conditioning. prepare_latents returns a per-token conditioning_mask; the loop
         passes the transformer `timestep * (1 - mask)`, so masked tokens are clean context at t = 0, and blends
         x0 = denoised * (1 - mask) + clean * mask, so their Euler step is exactly zero. It is how the pipeline holds an
         image-to-video first frame; LTX2PinnedPipeline puts held tokens there instead of an encoded image.
  audio  no pipeline has an audio mask, but the transformer accepts `audio_timestep` per token. A forward pre-hook makes
         it `t * (1 - audio_mask)`, prepare_audio_latents writes the held tokens, and since nothing blends the audio x0,
         the `audio_scheduler` (a PinnedScheduler) writes them back after every step. `audio_sigma` (prompt AdaLN,
         cross-modal modulation) stays one per row, as ltx-core's own audio-to-video and retake pipelines keep `sigma`.
  both   PinnedScheduler keeps each pass's final tokens: the pipeline's own normalized, packed space. The distilled
         recipe's two passes (half size, x2 latent upsampler, refine; runtimes.LtxAdapter._two_stage) are each pinned
         from tokens of that pass's own size.

Every render reports, per pass, whether the held tokens came out of denoising bit-identical (`pins_exact`) and whether
the transformer's first call saw them at timestep 0 and everything else above it (`seen_clean`). Proven on an RTX PRO
6000 for storyboard heads (research/long-video_ltx-av-extend_2026-09-16.md): pins bit-exact, t = 0 on held tokens. Held
tokens anywhere in the sequence (retake) and a fully held audio track (audio-to-video) use the same writes at other
positions, and have run on the CPU only, against diffusers' real classes with tiny random weights.

Plain data first (Geometry, Pins, Rendered: no torch), then the torch parts, imported only when a real render runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Protocol

from .media_tools import BackendError

# The audio VAE's STFT size. diffusers' AutoencoderKLLTX2Audio config records the mel rate, hop and bins but not this;
# ltx-core's AudioEncoderConfigurator reads it from the checkpoint's `preprocessing.stft.filter_length` with 1024 as the
# default, and every LTX-2 family config we have seen (JoyAI Echo's, built on LTX-2) says 1024 with win_length 1024 and
# mel_fmax 8000. Unconfirmed for LTX-2.5's own checkpoint: the GPU driver's round trip (encode, decode, vocoder) checks it.
AUDIO_N_FFT = 1024
# ltx-core's AudioProcessor.waveform_to_mel: log(max(mel, 1e-5)).
AUDIO_LOG_FLOOR = 1e-5


# ------------------------------------------------------------------ geometry (plain data)


@dataclass(frozen=True)
class Geometry:
    """Where latents sit in time. The defaults are LTX-2.5's; `from_pipeline` reads a loaded pipeline.

    The video VAE is causal, 8x in time and 32x in space, so n latent frames decode to 1 + 8(n - 1) frames. Audio is 16 kHz
    mel at hop 160, 4x in time (25 latents a second); the audio VAE is causal too (n latents decode to 4n - 3 mel frames),
    and the vocoder gives 480 samples per mel frame at 48 kHz."""

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

    def output_samples(self, frames: int) -> int:
        """Samples of a clip's sound track: its frames' duration at the vocoder's rate, as media_tools.encode_video cuts it."""
        return round(frames / float(self.fps) * self.sample_rate)


# ------------------------------------------------------------------ pins (plain data)


@dataclass
class Pins:
    """Tokens held fixed in one pass, in the pipeline's normalized, packed space ([1, n, features]).

    `video_at` and `audio_at` are the n sequence positions the tokens hold, ascending ([n] integer indices); None holds
    them at the head, positions 0..n-1, which is what a storyboard join pins."""

    video: Any = None
    audio: Any = None
    video_at: Any = None
    audio_at: Any = None

    def to(self, device: Any) -> Pins:
        move = lambda value: None if value is None else value.to(device)
        return Pins(move(self.video), move(self.audio), move(self.video_at), move(self.audio_at))


@dataclass
class Rendered:
    frames: Any  # the render's frames in order: PIL images or HxWx3 uint8 arrays
    audio: Any  # (channels, samples) float32
    sample_rate: int
    tails: dict[str, Any]  # per pass: what a storyboard keeps for later shots (ltx_storyboard.Tail); empty otherwise
    pins_exact: dict[str, bool | None]  # per pass: the held tokens came out of denoising bit-identical
    # per pass and modality: the transformer's first call saw the held tokens at timestep 0 and the rest above it
    seen_clean: dict[str, dict[str, bool | None]]
    timings: dict[str, float] = field(default_factory=dict)


class Renderer(Protocol):
    def geometry(self, fps: float) -> Geometry: ...

    def stages(self, call: dict[str, Any]) -> tuple[str, ...]: ...

    def render(self, call: dict[str, Any], pins: dict[str, Pins]) -> Rendered: ...


def pin_failure(stages: tuple[str, ...], pins: dict[str, Pins], rendered: Rendered) -> str | None:
    """Why held tokens can't be trusted, or None. Held tokens that moved, or that the model wasn't shown as clean context,
    show in the output: at a storyboard seam, or as a retake that changed what it should keep. Either means the runtime
    changed under this module (a diffusers upgrade), so the job fails rather than deliver it."""
    for stage in stages:
        pin = pins.get(stage) or Pins()
        if (pin.video is not None or pin.audio is not None) and rendered.pins_exact.get(stage) is not True:
            return f"the pinned {stage}-pass tokens changed during denoising"
        for modality, held, at in (("video", pin.video, pin.video_at), ("audio", pin.audio, pin.audio_at)):
            if held is not None and (rendered.seen_clean.get(stage) or {}).get(modality) is not True:
                where = "head" if at is None else "tokens"
                return f"the model was not shown the pinned {modality} {where} at timestep 0 in the {stage} pass"
    return None


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


# ------------------------------------------------------------------ torch


def held_mask(tokens: int, held: Any, device: Any = None):
    """A [tokens] bool mask of the held positions: `held` is a head length (int) or an index tensor."""
    import torch

    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    if isinstance(held, int):
        mask[:held] = True
    elif held is not None:
        mask[held.to(mask.device)] = True
    return mask


def pinned_timesteps(timestep: Any, tokens: int, head: Any):
    """Per-token timesteps for one modality: 0 on the held tokens (a head length or index tensor), so the transformer reads
    them as clean context."""
    mask = held_mask(tokens, head, timestep.device).to(timestep.dtype)
    return timestep.reshape(-1, 1) * (1 - mask)[None]


def _positions(values: Any, at: Any):
    return values.shape[1] if at is None else at


def held_matches(tokens: Any, values: Any, at: Any = None) -> bool | None:
    """Whether `tokens` hold exactly `values` at `at` (None: the head)."""
    import torch

    if values is None:
        return None
    found = tokens[:, : values.shape[1]] if at is None else tokens[:, at.to(tokens.device)]
    return bool(torch.equal(found.to("cpu"), values.to("cpu", tokens.dtype)))


def head_matches(tokens: Any, head: Any) -> bool | None:
    return held_matches(tokens, head)


def clean_at(timestep: Any, values: Any, at: Any = None) -> bool | None:
    """Whether per-token timesteps show the transformer the held tokens at t = 0 and every other token above it."""
    if values is None:
        return None
    if timestep is None or timestep.ndim != 2:
        return False
    mask = held_mask(timestep.shape[1], _positions(values, at), timestep.device)
    return bool((timestep[:, mask] == 0).all()) and bool((timestep[:, ~mask] > 0).all())


def clean_head(timestep: Any, head: Any) -> bool | None:
    """Whether per-token timesteps show the transformer the pinned head at t = 0 and everything after it as noise."""
    return clean_at(timestep, head)


def all_exact(*checks: bool | None) -> bool | None:
    present = [c for c in checks if c is not None]
    return all(present) if present else None


_PINNED_CLASSES: tuple | None = None


def pinned_classes() -> tuple:
    """PinnedScheduler and LTX2PinnedPipeline, defined on first use so the plain-data parts import without diffusers."""
    global _PINNED_CLASSES
    if _PINNED_CLASSES is not None:
        return _PINNED_CLASSES
    from diffusers import FlowMatchEulerDiscreteScheduler, LTX2ConditionPipeline
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput

    class PinnedScheduler(FlowMatchEulerDiscreteScheduler):
        """The Euler step, then the held tokens written back exactly. Keeps the pass's final tokens in `final`."""

        pin = None
        at = None
        final = None

        def start(self, pin: Any, at: Any = None) -> None:
            self.pin, self.at, self.final = pin, at, None

        def step(self, model_output, timestep, sample, *args, return_dict: bool = True, **kwargs):
            prev = super().step(model_output, timestep, sample, *args, return_dict=False, **kwargs)[0]
            if self.pin is not None:
                held = self.pin.to(prev.device, prev.dtype)
                if self.at is None:
                    prev[:, : held.shape[1]] = held
                else:
                    prev[:, self.at] = held
            if self.step_index is not None and self.step_index >= len(self.timesteps):
                self.final = prev.detach().clone()
            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev) if return_dict else (prev,)

    class LTX2PinnedPipeline(LTX2ConditionPipeline):
        """LTX2ConditionPipeline with `pins` written where it would put an encoded first frame."""

        pins: Pins | None = None

        def prepare_latents(self, *args, **kwargs):
            latents, mask, clean, keyframe_coords = super().prepare_latents(*args, **kwargs)
            held = getattr(self.pins, "video", None)
            if held is not None:
                held = held.to(latents.device, latents.dtype)
                at = self.pins.video_at
                where = slice(0, held.shape[1]) if at is None else at.to(latents.device)
                # mask 1: the loop's timestep * (1 - mask) shows these to the transformer at t = 0, and its x0 blend keeps
                # them, so the Euler step moves them by exactly nothing.
                latents[:, where] = held
                clean[:, where] = held
                mask[:, where] = 1.0
            return latents, mask, clean, keyframe_coords

        def prepare_audio_latents(self, *args, **kwargs):
            latents = super().prepare_audio_latents(*args, **kwargs)
            held = getattr(self.pins, "audio", None)
            if held is not None:
                at = self.pins.audio_at
                where = slice(0, held.shape[1]) if at is None else at.to(latents.device)
                latents[:, where] = held.to(latents.device, latents.dtype)
            return latents

    _PINNED_CLASSES = (PinnedScheduler, LTX2PinnedPipeline)
    return _PINNED_CLASSES


def _clock(cuda: bool) -> float:
    if cuda:
        import torch

        torch.cuda.synchronize()
    return time.perf_counter()


class PinnedRenderer:
    """One render at a time on the pipelines the worker's loader built (runtimes.ltx_loader), through an LTX2PinnedPipeline
    that shares their loaded modules: no second copy of the weights. `render(call, pins)` takes a `build_call` output."""

    def __init__(self, pipelines: dict[str, Any], device: str):
        Scheduler, Pipeline = pinned_classes()
        source = pipelines.get("condition") or pipelines["text"]
        components = dict(source.components)
        # Its own schedulers, so the pins never touch the pipelines other jobs use. from_config keeps the recipe's schedule
        # (build_ltx_pipelines gives the full model dynamic shifting).
        components["scheduler"] = Scheduler.from_config(source.scheduler.config)
        components["audio_scheduler"] = Scheduler.from_config(source.scheduler.config)
        self.pipeline = Pipeline(**components)
        self.upsample = pipelines.get("upsample")
        self.device = device

    def geometry(self, fps: float) -> Geometry:
        return Geometry.from_pipeline(self.pipeline, fps)

    def stages(self, call: dict[str, Any]) -> tuple[str, ...]:
        return ("half", "full") if call.get("second_stage_sigmas") and self.upsample is not None else ("full",)

    @staticmethod
    def stage_size(stage: str, call: dict[str, Any]) -> tuple[int, int]:
        """(width, height) a pass renders at: the distilled recipe's first pass is half size."""
        width, height = int(call["width"]), int(call["height"])
        return (width // 2, height // 2) if stage == "half" else (width, height)

    def prepare_call(self, call: dict[str, Any]) -> dict[str, Any]:
        """The pipeline's keyword arguments; the CPU tests replace the text encoder here."""
        return call

    def render(self, call: dict[str, Any], pins: dict[str, Pins]) -> Rendered:
        import torch

        from .ltx_resident import WORKER_KEYS
        from .resident import PipelineResult
        from .runtimes import _audio_rate, video_conditions

        call = dict(call)
        conditions = call.get("conditions")
        second = call.get("second_stage_sigmas")
        generator = torch.Generator(device=self.device).manual_seed(int(call["seed"]))
        for key in WORKER_KEYS:  # as runtimes.LtxAdapter does: nothing the diffusers call doesn't accept
            call.pop(key, None)
        if conditions:
            call["conditions"] = video_conditions(conditions)
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
        pin = (pins.get(stage) or Pins()).to(self.device)  # once, not every step
        pipeline.pins = pin
        pipeline.scheduler.start(pin.video, pin.video_at)
        pipeline.audio_scheduler.start(pin.audio, pin.audio_at)
        seen: dict[str, bool | None] = {}

        def timesteps(module, args, kwargs):
            audio_t, audio_tokens = kwargs.get("audio_timestep"), kwargs.get("audio_hidden_states")
            if pin.audio is not None and audio_t is not None and audio_t.ndim == 1 and audio_tokens is not None:
                # One timestep per batch row becomes one per audio token, 0 on the held ones. audio_sigma (prompt AdaLN,
                # cross-modal modulation) stays per row, as it does for video.
                kwargs["audio_timestep"] = audio_t = pinned_timesteps(audio_t, audio_tokens.shape[1], _positions(pin.audio, pin.audio_at))
            if not seen:  # the pass's first transformer call: what the model is actually shown
                seen.update(video=clean_at(kwargs.get("timestep"), pin.video, pin.video_at), audio=clean_at(audio_t, pin.audio, pin.audio_at))
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
            raise BackendError(f"the {stage} pass finished without a final scheduler step")
        record["seen"][stage] = seen
        record["exact"][stage] = all_exact(held_matches(video, pin.video, pin.video_at), held_matches(audio, pin.audio, pin.audio_at))
        self.finished_pass(stage, video, audio, call, extra, record)
        return output

    def finished_pass(self, stage: str, video: Any, audio: Any, call: dict[str, Any], extra: dict[str, Any], record: dict) -> None:
        """A pass's final tokens, for renderers that keep some (ltx_storyboard.ExtendRenderer keeps tails)."""

    # -------------------------------------------------------------- sources, encoded as the pipeline's own tokens

    def encode_video(self, frames: Any, width: int, height: int):
        """[1, latent frames x h x w, 128] tokens of RGB uint8 frames ([frames, H, W, 3], 8k + 1 of them), in the space
        prepare_latents holds an encoded condition in: the VAE's posterior mode ("argmax"), normalized by the VAE's latent
        statistics without a scaling factor, then packed. Frames of another size are resized to width x height first (the
        distilled recipe's half-size pass). The pixels are written to the GPU a frame at a time, so the only full-clip
        tensors there are the VAE's input and what it makes of it (quantized.source_encode_gib counts them)."""
        import torch
        import torch.nn.functional as F

        pipeline = self.pipeline
        vae = pipeline.vae
        device = pipeline._execution_device  # where offload hooks run the VAE, as the pipeline itself encodes conditions
        dtype = vae.dtype
        count = len(frames)
        pixels = torch.empty((1, 3, count, height, width), dtype=dtype, device=device)
        with torch.no_grad():
            for index in range(count):
                frame = torch.from_numpy(frames[index]).to(device).permute(2, 0, 1).to(torch.float32)
                if frame.shape[1:] != (height, width):
                    frame = F.interpolate(frame[None], size=(height, width), mode="bilinear", align_corners=False, antialias=True)[0]
                pixels[0, :, index] = (frame / 127.5 - 1.0).to(dtype)
            latent = vae.encode(pixels).latent_dist.mode()
            del pixels
            latent = pipeline._normalize_latents(latent, vae.latents_mean, vae.latents_std).to(torch.float32)
            return pipeline._pack_latents(latent, pipeline.transformer_spatial_patch_size, pipeline.transformer_temporal_patch_size)

    def encode_audio(self, samples: Any, sample_rate: int, latents: int):
        """[1, latents, 128] tokens of a sound track ((channels, samples) float32 in [-1, 1] at `sample_rate`), in the space
        prepare_audio_latents holds them in. As ltx-core's encode_audio: stereo (mono is doubled), resampled to the audio
        VAE's rate with torchaudio, log-mel (AUDIO_N_FFT, Hann, centred, magnitude, Slaney scale and norm, 0 Hz to Nyquist),
        the VAE's posterior mode; then packed and normalized as LTX2Pipeline._pack_audio_latents and
        _normalize_audio_latents do. The audio VAE is causal (latent k reads mel frames up to 4k), so the track is padded
        with silence past its end and the latents cut to `latents`."""
        import torch
        import torchaudio

        pipeline = self.pipeline
        audio_vae = pipeline.audio_vae
        config = audio_vae.config
        rate, hop, bins = int(config.sample_rate), int(config.mel_hop_length), int(config.mel_bins)
        device = pipeline._execution_device
        with torch.no_grad():
            waveform = torch.as_tensor(samples, dtype=torch.float32)
            if waveform.ndim == 1:
                waveform = waveform[None]
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            waveform = waveform[:2].to(device)
            if int(sample_rate) != rate:
                waveform = torchaudio.functional.resample(waveform, int(sample_rate), rate)
            ratio = int(pipeline.audio_vae_temporal_compression_ratio)
            needed = (ratio * latents + 2) * hop + AUDIO_N_FFT  # the mel frames `latents` read, and the STFT's reach past them
            if waveform.shape[1] < needed:
                waveform = torch.nn.functional.pad(waveform, (0, needed - waveform.shape[1]))
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=rate, n_fft=AUDIO_N_FFT, win_length=AUDIO_N_FFT, hop_length=hop, f_min=0.0, f_max=rate / 2.0, n_mels=bins,
                window_fn=torch.hann_window, center=True, pad_mode="reflect", power=1.0, mel_scale="slaney", norm="slaney",
            ).to(device)(waveform[None])
            mel = torch.log(torch.clamp(mel, min=AUDIO_LOG_FLOOR)).permute(0, 1, 3, 2).contiguous()  # [1, channels, time, bins]
            latent = audio_vae.encode(mel.to(audio_vae.dtype)).latent_dist.mode()[:, :, :latents].to(torch.float32)
            if latent.shape[2] != latents:
                raise BackendError(f"the audio VAE encoded {latent.shape[2]} latents, {latents} were needed")
            packed = pipeline._pack_audio_latents(latent)
            return pipeline._normalize_audio_latents(packed, audio_vae.latents_mean, audio_vae.latents_std)
