"""Stand-ins for LTX-2.5 that storyboard tests render through, from the GPU experiment's fake and tiny backends.

  StubRenderer  numpy only: the right frame and sample counts, tails labelled with the shot that made them. For the
                backend and worker flow on any machine.
  FakeRenderer  plain torch with LTX-2.5's latent geometry and a stand-in model that continues whatever clean context it
                is shown: proves the tail, pin, trim and stitch arithmetic end to end.
  TinyRenderer  ExtendRenderer on diffusers' real LTX-2 classes with tiny random weights: proves the integration with
                diffusers (hooks, shapes, both passes, the real audio VAE and vocoder), not the pictures.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from kuno_worker.backends.ltx_storyboard import (
    ExtendRenderer,
    Geometry,
    Pins,
    Rendered,
    Tail,
    all_exact,
    clean_head,
    head_matches,
    pinned_timesteps,
    take_tail,
)


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def stages_of(call: dict[str, Any]) -> tuple[str, ...]:
    return ("half", "full") if call.get("second_stage_sigmas") else ("full",)


class StubRenderer:
    """Frames and sound of the right length. Every tail token holds the index of the shot that rendered it, so a test can
    see which shot each pin came from."""

    features = 4

    def __init__(self, overlap: int = 3, fail_on_shot: int | None = None):
        self.overlap, self.fail_on_shot = overlap, fail_on_shot
        self.calls: list[dict[str, Any]] = []
        self.pins: list[dict[str, Pins]] = []

    def geometry(self, fps: float) -> Geometry:
        return Geometry(fps=float(fps))

    def stages(self, call: dict[str, Any]) -> tuple[str, ...]:
        return stages_of(call)

    def render(self, call: dict[str, Any], pins: dict[str, Pins]) -> Rendered:
        index = len(self.calls)
        self.calls.append(call)
        self.pins.append(pins)
        if index == self.fail_on_shot:
            raise RuntimeError("CUDA out of memory")
        g = self.geometry(call["frame_rate"])
        frames, width, height = int(call["num_frames"]), call["width"], call["height"]
        pixels = np.empty((frames, height, width, 3), dtype=np.uint8)
        pixels[:] = np.array([(37 * index) % 256, 128, (int(call["seed"]) * 11) % 256], dtype=np.uint8)
        audio_latents = g.audio_latents(frames)
        wave = 0.1 * np.sin(np.arange(g.audio_samples(audio_latents)) * 2 * np.pi * 220 / g.sample_rate).astype(np.float32)
        tails = {
            stage: Tail(video=np.full((1, self.overlap * 6, self.features), float(index)), audio=np.full((1, audio_latents, self.features), float(index)))
            for stage in self.stages(call)
        }
        present = lambda pin: True if pin is not None else None  # noqa: E731
        exact = {stage: all_exact(present(pins[stage].video), present(pins[stage].audio)) for stage in tails}
        seen = {stage: {"video": present(pins[stage].video), "audio": present(pins[stage].audio)} for stage in tails}
        return Rendered(pixels, np.stack([wave, wave]), g.sample_rate, tails, exact, seen)


class FakeRenderer:
    """LTX-2.5's latent geometry in plain torch on the CPU, with a stand-in for the transformer.

    Tokens are packed like the pipeline's ([1, latent frames x h x w, 128] and [1, audio latents, 128]) and denoised by the
    same Euler steps over the call's sigmas, with the same per-token timesteps, x0 blend and re-imposition as the real
    path, in both passes of the two-stage recipe. The stand-in writes a clock into channel 0: the time of each latent's
    last frame (or mel frame), counted on from the first latent of whatever head it is shown at timestep 0 (a fresh
    clock otherwise), plus a colour in channels 1-3. Decoding follows the causal VAEs, so the stitched video's clock
    advances by exactly one frame, and its audio clock by one sample, across every seam the plan says is continuous.
    Each result carries `clocks`: (per-frame clock, per-sample clock).
    """

    features = 128

    def __init__(self, overlap: int, trace: bool = False):
        self.overlap = overlap
        self.trace = trace
        self.head_trace: list[tuple[str, Any, Any]] = []

    def geometry(self, fps: float) -> Geometry:
        return Geometry(fps=float(fps))

    def stages(self, call: dict[str, Any]) -> tuple[str, ...]:
        return stages_of(call)

    def render(self, call: dict[str, Any], pins: dict[str, Pins]) -> Rendered:
        import torch

        g = self.geometry(call["frame_rate"])
        frames = int(call["num_frames"])
        latent_frames, audio_latents = g.latent_frames(frames), g.audio_latents(frames)
        generator = torch.Generator("cpu").manual_seed(int(call["seed"]))
        fresh = 5.0 * (int(call["seed"]) % 20 + 1)
        colour = torch.tensor([((stable_seed(call["prompt"]) >> s) % 200) / 100.0 - 1.0 for s in (0, 8, 16)])
        width, height, second = call["width"], call["height"], call.get("second_stage_sigmas")
        tails: dict[str, Tail] = {}
        exact: dict[str, bool | None] = {}
        seen: dict[str, dict[str, bool | None]] = {}
        started = time.perf_counter()
        if second:
            h, w = height // 2 // g.spatial_ratio, width // 2 // g.spatial_ratio
            video, audio, seen["half"] = self._pass("half", g, pins, generator, call["sigmas"], latent_frames, h * w, audio_latents, fresh, colour)
            tails["half"] = take_tail(video, audio, latent_frames, h * w, self.overlap)
            half = pins.get("half") or Pins()
            exact["half"] = all_exact(head_matches(video, half.video), head_matches(audio, half.audio))
            # The x2 latent upsampler, then the refine starts from it noised to its first sigma, as the pipeline does.
            grid = video.reshape(1, latent_frames, h, w, -1).repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
            init = (grid.reshape(1, latent_frames * 4 * h * w, -1), audio)
            video, audio, seen["full"] = self._pass("full", g, pins, generator, second, latent_frames, 4 * h * w, audio_latents, fresh, colour, init)
            tokens_per_frame = 4 * h * w
        else:
            steps = call.get("sigmas") or list(np.linspace(1.0, 1.0 / call["num_inference_steps"], call["num_inference_steps"]))
            tokens_per_frame = (height // g.spatial_ratio) * (width // g.spatial_ratio)
            video, audio, seen["full"] = self._pass("full", g, pins, generator, steps, latent_frames, tokens_per_frame, audio_latents, fresh, colour)
        full = pins.get("full") or Pins()
        exact["full"] = all_exact(head_matches(video, full.video), head_matches(audio, full.audio))
        tails["full"] = take_tail(video, audio, latent_frames, tokens_per_frame, self.overlap)
        timings = {"generation_s": round(time.perf_counter() - started, 3)}
        frame_pixels, frame_clock = self._decode_video(g, video, latent_frames, tokens_per_frame, frames, width, height)
        samples, sample_clock = self._decode_audio(g, audio)
        rendered = Rendered(frame_pixels, samples, g.sample_rate, tails, exact, seen, timings)
        rendered.clocks = (frame_clock, sample_clock)
        return rendered

    def _pass(self, stage, g, pins, generator, sigmas, latent_frames, tokens_per_frame, audio_latents, fresh, colour, init=None):
        import torch

        pin = pins.get(stage) or Pins()
        sigmas = [float(s) for s in sigmas] + [0.0]
        video = torch.randn(1, latent_frames * tokens_per_frame, self.features, generator=generator)
        audio = torch.randn(1, audio_latents, self.features, generator=generator)
        if init is not None:
            video = sigmas[0] * video + (1 - sigmas[0]) * init[0]
            audio = sigmas[0] * audio + (1 - sigmas[0]) * init[1]
        video_head = 0 if pin.video is None else pin.video.shape[1]
        audio_head = 0 if pin.audio is None else pin.audio.shape[1]
        clean = torch.zeros_like(video)
        mask = torch.zeros(1, video.shape[1], 1)
        if video_head:
            video[:, :video_head] = clean[:, :video_head] = pin.video
            mask[:, :video_head] = 1.0
        if audio_head:
            audio[:, :audio_head] = pin.audio
        seen: dict[str, bool | None] = {}
        for i in range(len(sigmas) - 1):
            sigma, following = sigmas[i], sigmas[i + 1]
            t = torch.tensor([sigma * 1000.0])
            t_video, t_audio = pinned_timesteps(t, video.shape[1], video_head), pinned_timesteps(t, audio.shape[1], audio_head)
            if i == 0:
                seen = {"video": clean_head(t_video, pin.video), "audio": clean_head(t_audio, pin.audio)}
            x0_video, x0_audio = self._model(g, video, audio, t_video, t_audio, latent_frames, tokens_per_frame, fresh, colour)
            x0_video = x0_video * (1 - mask) + clean * mask  # the pipeline's blend
            video = video + (following - sigma) * ((video - x0_video) / sigma)  # FlowMatchEulerDiscreteScheduler.step
            audio = audio + (following - sigma) * ((audio - x0_audio) / sigma)
            if video_head:
                video[:, :video_head] = pin.video  # PinnedScheduler.step
            if audio_head:
                audio[:, :audio_head] = pin.audio
            if self.trace:
                self.head_trace.append((stage, video[:, :video_head].clone(), audio[:, :audio_head].clone()))
        return video, audio, seen

    def _model(self, g, video, audio, t_video, t_audio, latent_frames, tokens_per_frame, fresh, colour):
        """x0 for every token. Context is only what arrives at timestep 0: a head re-imposed after the step but shown
        noised would not count."""
        import torch

        video_context = bool(t_video[0, :tokens_per_frame].eq(0).all())
        audio_context = bool(t_audio[0, :1].eq(0).all())
        v0 = float(video[0, 0, 0]) if video_context else float(audio[0, 0, 0]) if audio_context else fresh
        a0 = float(audio[0, 0, 0]) if audio_context else v0
        tint = video[0, 0, 1:4] if video_context else colour
        x0_video = torch.zeros_like(video)
        clock = v0 + g.temporal_ratio * torch.arange(latent_frames, dtype=torch.float64) / g.fps
        x0_video[0, :, 0] = clock.repeat_interleave(tokens_per_frame).to(video.dtype)
        x0_video[0, :, 1:4] = tint
        x0_audio = torch.zeros_like(audio)
        step = g.audio_ratio * g.mel_hop / g.mel_sample_rate
        x0_audio[0, :, 0] = (a0 + step * torch.arange(audio.shape[1], dtype=torch.float64)).to(audio.dtype)
        return x0_video, x0_audio

    @staticmethod
    def _decode_video(g, tokens, latent_frames, tokens_per_frame, frames, width, height):
        grid = tokens[0].reshape(latent_frames, tokens_per_frame, -1).double().mean(dim=1).numpy()
        clocks, tints = grid[:, 0], grid[:, 1:4]
        f = np.arange(frames)
        block = np.where(f == 0, 0, (f + g.temporal_ratio - 1) // g.temporal_ratio)
        share = np.where(f == 0, 1.0, (f - (g.temporal_ratio * block - (g.temporal_ratio - 1)) + 1) / g.temporal_ratio)
        before = clocks[np.maximum(block - 1, 0)]
        frame_clock = np.where(f == 0, clocks[0], before + share * (clocks[block] - before))
        pixels = np.empty((frames, height, width, 3), dtype=np.uint8)
        pixels[:] = np.clip(127.5 * (tints[block] + 1.0), 0, 255).astype(np.uint8)[:, None, None, :]
        bar = ((frame_clock * 0.25) % 1.0 * (width - 8)).astype(int)  # a bar crossing the frame every 4 s of clock
        for i, x in enumerate(bar):
            pixels[i, :, x : x + 8] = 255
        return pixels, frame_clock

    @staticmethod
    def _decode_audio(g, tokens):
        clocks = tokens[0, :, 0].double().numpy()
        spm = int(g.samples_per_mel)
        s = np.arange(g.audio_samples(len(clocks)))
        mel = s // spm
        block = np.where(mel == 0, 0, (mel + g.audio_ratio - 1) // g.audio_ratio)
        share = np.where(mel == 0, 1.0, (mel - (g.audio_ratio * block - (g.audio_ratio - 1)) + 1) / g.audio_ratio)
        before = clocks[np.maximum(block - 1, 0)]
        mel_clock = np.where(mel == 0, clocks[0], before + share * (clocks[block] - before))
        sample_clock = mel_clock + (s % spm) / g.sample_rate
        wave = (0.2 * np.sin(2 * np.pi * 220.0 * sample_clock)).astype(np.float32)
        return np.stack([wave, wave]), sample_clock


def tiny_pipelines(seed: int = 0) -> dict[str, Any]:
    """LTX2ConditionPipeline and the latent upsampler with tiny random weights, LTX-2.5's geometry (32x/8x video VAE, 16 kHz
    mel at hop 160, 4x audio VAE, 48 kHz vocoder with bandwidth extension) and 128-feature tokens for both modalities. No
    text encoder: TinyRenderer passes prompt embeddings."""
    import torch
    from diffusers import (
        AutoencoderKLLTX2Audio,
        AutoencoderKLLTX2Video,
        FlowMatchEulerDiscreteScheduler,
        LTX2ConditionPipeline,
        LTX2LatentUpsamplePipeline,
        LTX2VideoTransformer3DModel,
    )
    from diffusers.pipelines.ltx2 import LTX2TextConnectors
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

    torch.manual_seed(seed)
    text = TinyRenderer.text_channels
    vae = AutoencoderKLLTX2Video(
        latent_channels=128, block_out_channels=(8, 16, 32, 64), decoder_block_out_channels=(8, 16, 32),
        layers_per_block=(1, 1, 1, 1, 1), decoder_layers_per_block=(1, 1, 1, 1), patch_size=4, patch_size_t=1,
        spatial_compression_ratio=32, temporal_compression_ratio=8,
    )
    audio_vae = AutoencoderKLLTX2Audio(base_channels=128, ch_mult=(1,), num_res_blocks=1, latent_channels=8, mel_bins=64)
    # Non-trivial normalization, so a token taken or written in the wrong space would show.
    for module in (vae, audio_vae):
        module.latents_mean.copy_(torch.randn_like(module.latents_mean) * 0.1)
        module.latents_std.copy_(torch.rand_like(module.latents_std) + 0.5)
    vocoder = LTX2VocoderWithBWE(
        in_channels=128, hidden_channels=128, out_channels=2, resnet_kernel_sizes=[3], resnet_dilations=[[1]],
        bwe_in_channels=128, bwe_hidden_channels=64, bwe_out_channels=2, bwe_resnet_kernel_sizes=[3], bwe_resnet_dilations=[[1]],
    )
    connectors = LTX2TextConnectors(
        caption_channels=text, text_proj_in_factor=TinyRenderer.text_layers + 1, video_connector_num_attention_heads=2,
        video_connector_attention_head_dim=8, video_connector_num_layers=1, video_connector_num_learnable_registers=None,
        audio_connector_num_attention_heads=2, audio_connector_attention_head_dim=8, audio_connector_num_layers=1,
        audio_connector_num_learnable_registers=None, connector_rope_base_seq_len=32, rope_double_precision=False, rope_type="split",
    )
    transformer = LTX2VideoTransformer3DModel(
        in_channels=128, out_channels=128, num_attention_heads=2, attention_head_dim=8, cross_attention_dim=16,
        audio_in_channels=128, audio_out_channels=128, audio_num_attention_heads=2, audio_attention_head_dim=4,
        audio_cross_attention_dim=8, num_layers=2, caption_channels=text, rope_double_precision=False, rope_type="split",
        cross_attn_mod=True, audio_cross_attn_mod=True,  # the LTX-2.3+ branch: prompt AdaLN and 9 modulation parameters
    )
    condition = LTX2ConditionPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae, audio_vae=audio_vae, text_encoder=None, tokenizer=None,
        connectors=connectors, transformer=transformer, vocoder=vocoder,
    )
    condition.set_progress_bar_config(disable=True)
    upsampler = LTX2LatentUpsamplerModel(in_channels=128, mid_channels=32, num_blocks_per_stage=1)
    return {"condition": condition, "upsample": LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=upsampler)}


class TinyRenderer(ExtendRenderer):
    """ExtendRenderer on tiny_pipelines(), on the CPU. The pictures are noise; the shapes, hooks and passes are the real ones."""

    text_channels, text_layers, text_tokens = 16, 2, 8

    def __init__(self, pipelines: dict[str, Any], overlap: int):
        super().__init__(pipelines, device="cpu", overlap=overlap)
        self.pipeline.set_progress_bar_config(disable=True)

    def prepare_call(self, call: dict[str, Any]) -> dict[str, Any]:
        import torch

        prompt = call.pop("prompt")
        call.pop("negative_prompt", None)
        shape = (1, self.text_tokens, self.text_channels * (self.text_layers + 1))
        generator = torch.Generator("cpu").manual_seed(stable_seed(prompt))
        call["prompt_embeds"] = torch.randn(shape, generator=generator)
        call["prompt_attention_mask"] = torch.ones(1, self.text_tokens, dtype=torch.long)
        if call.get("guidance_scale", 3.0) > 1.0 or (call.get("audio_guidance_scale") or 0) > 1.0:
            call["negative_prompt_embeds"] = torch.zeros(shape)
            call["negative_prompt_attention_mask"] = torch.ones(1, self.text_tokens, dtype=torch.long)
        return call
