"""LTX-2.5's video VAE encoder run over a clip a few frames at a time, giving the latents of encoding the clip whole.

Why. A retake encodes its source clip on the GPU before denoising (ltx_pinning.PinnedRenderer.encode_video). diffusers'
AutoencoderKLLTX2Video.encode holds the encoder's activations for every frame at once: 12.07 GiB over the weights for
121 frames of 1280x704 on an RTX PRO 6000 (2026-09-17), growing with the clip, so a 15 s 720p retake would have needed
about 36 GiB for the encode alone beside 66.18 GiB of weights. Here memory depends on the frame size, not the clip's
length.

Why not diffusers' tiling (autoencoder_kl_ltx2.py, 0.40). `enable_tiling` tiles the encode in space only: 512 px tiles
every 448 px, encoded separately and blended where they overlap. Temporal tiling (`_temporal_tiled_encode`) is reached
only through `use_framewise_decoding` (`use_framewise_encoding` is read nowhere) and encodes overlapping 17-frame windows,
each from its own causally padded first frame, then blends the overlapping latent frames. Both change the latents, and a
retake's held tokens are the source's encoding: every frame outside the window is decoded from them. The model and group
offload modes enable tiling (quantized._apply_offload), so `vae.encode` tiled there; this never does. Decoding is left
as it is: every render in a mode decodes the same way (whole on an RTX PRO 6000, spatially tiled with offload), held
frames of a retake included.

How. Every operation of the encoder that looks across time is causal and local, so it can be streamed exactly:
  LTX2VideoCausalConv3d   pads the front of its input with (kernel - 1) copies of its first frame and convolves with no
                          temporal padding, so output t reads inputs up to t. The first chunk is padded as the module
                          pads it; every later chunk is preceded by the frames of the input so far that the next output
                          still reads (the last kernel - 1 at stride 1).
  LTX2VideoDownsampler3d  prepends (stride - 1) copies of its input's first frame, then folds every `stride` frames into
                          one. Only the first chunk gets the copies.
  everything else         per frame: the spatial patchify (patch_size_t 1), RMS and layer norms over channels, SiLU,
                          1x1x1 shortcuts, the output's repeated last channel and the posterior's split.
Chunks are frame 0 alone (latent frame 0), then CHUNK_LATENT_FRAMES x the temporal ratio frames at a time: every chunk is
a whole number of folds at every downsampler, so chunk j's latents are the whole clip's latents for its frames.

Equality (test_ltx_edit_render.py, diffusers' real classes with tiny random weights, float32 on the CPU). One chunk
through the streamed layers is bit-identical to `vae.encode`, so the layers are reproduced exactly. Split into chunks the
latents differ by at most 2.5e-6 against a spread (standard deviation) of 0.27: a convolution computes the same sums in
another order for an input of another length (one Conv3d applied to 3 frames and then 19 already differs from it applied
to 20 by 1.8e-6). The tests allow 1e-4 of the spread. A GPU picks convolution algorithms per shape too, so bf16 there may
differ by more; the GPU driver (scripts/gpu-test/long_video/run_edit_modes_worker.py) reports the largest difference from
the whole-clip encode of its 5 s source against that spread.

Memory (quantized.source_encode_gib models it from these). Counted on the CPU as live tensors, with LTX-2's encoder
layout at its real widths (diffusers' defaults: 694M parameters) and bf16-sized values, per source pixel: the whole clip
takes 86.0 bytes a frame plus 65; chunked, 1,797 bytes for 8-frame chunks and 2,485 for 16, whatever the clip's length.
1,044 of those are what the stream keeps between chunks: kernel - 1 input frames at every causal convolution
(`encoder_cache_values`, 522 activations per pixel for that layout).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

from .media_tools import BackendError

# Latent frames a chunk after the first encodes: 8 frames at LTX-2.5's 8x. Its peak is 72% of a 16-frame chunk's (the
# stream's cached frames are the same for any chunk), which matters on the offloading cards; the cost is a pass through
# every layer per chunk, small beside a render (121 frames encoded whole in 1.1 s on an RTX PRO 6000).
CHUNK_LATENT_FRAMES = 1


def chunk_bounds(frames: int, temporal_ratio: int, latent_frames_per_chunk: int = CHUNK_LATENT_FRAMES) -> list[tuple[int, int]]:
    """[start, end) frames of each chunk of a clip of `frames` = 1 + ratio x k frames: frame 0 alone, then whole latent
    frames. Aligning chunks with latent frames makes every chunk a whole number of folds at every temporal downsampler."""
    if frames < 1 or (frames - 1) % temporal_ratio:
        raise BackendError(f"a clip to encode needs 1 + {temporal_ratio}k frames, not {frames}")
    step = temporal_ratio * max(1, int(latent_frames_per_chunk))
    return [(0, 1)] + [(start, min(start + step, frames)) for start in range(1, frames, step)]


def encoder_cache_values(encoder: Any) -> float:
    """Activations the stream holds between chunks per source pixel: (kernel - 1) input frames at every causal
    convolution, at that layer's channels and spatial size. Works on an encoder built on the meta device."""
    from diffusers.models.autoencoders.autoencoder_kl_ltx2 import LTX2VideoCausalConv3d, LTX2VideoDownsampler3d

    total = 0.0

    def walk(module: Any, factor: int) -> int:
        """`factor`: the module input's spatial downsampling from the source. Returns its output's."""
        nonlocal total
        if isinstance(module, LTX2VideoCausalConv3d):
            total += (module.kernel_size[0] - 1) * module.in_channels / factor**2
            return factor * module.conv.stride[1]
        for child in module.children():  # registration order is the order forward runs them
            factor = walk(child, factor)
        return factor * module.stride[1] if isinstance(module, LTX2VideoDownsampler3d) else factor

    walk(encoder, int(encoder.patch_size))
    return total


@contextmanager
def streaming(encoder: Any) -> Iterator[Callable[[Any], Any]]:
    """`encoder` (LTX2VideoEncoder3d) with its temporal layers streaming: calling the yielded function on consecutive
    chunks returns each chunk's share of the whole clip's encoder output. The layers are restored on exit."""
    from diffusers.models.autoencoders.autoencoder_kl_ltx2 import LTX2VideoCausalConv3d, LTX2VideoDownsampler3d
    import torch

    if not getattr(encoder, "is_causal", False) or int(getattr(encoder, "patch_size_t", 1)) != 1:
        raise BackendError("the video VAE's encoder is not causal frame by frame; it can't be encoded in chunks")
    held: dict[Any, Any] = {}  # causal convolution -> the input frames its next output still reads
    started: set[Any] = set()  # downsamplers that have had the first chunk
    patched = []

    def causal_conv(self, hidden_states, causal: bool = True):
        if not causal:
            raise BackendError("the video VAE's encoder ran a non-causal convolution; it can't be encoded in chunks")
        conv = self.conv
        kernel, stride, dilation = self.kernel_size[0], conv.stride[0], conv.dilation[0]
        span = dilation * (kernel - 1) + 1
        before = held.get(self)
        if before is None:  # the module's own causal padding, from the clip's first frame
            before = hidden_states[:, :, :1].repeat((1, 1, kernel - 1, 1, 1))
        stream = torch.cat([before, hidden_states], dim=2)
        count = (stream.shape[2] - span) // stride + 1 if stream.shape[2] >= span else 0
        if count < 1:
            raise BackendError("a chunk of the source was too short for the video VAE's convolutions")
        # A copy: a view would keep the whole padded chunk alive until the next one.
        held[self] = stream[:, :, count * stride :].clone()
        return conv(stream[:, :, : (count - 1) * stride + span])

    def downsampler(self, hidden_states, causal: bool = True):
        # LTX2VideoDownsampler3d.forward (diffusers 0.40), with the leading copies on the first chunk only.
        s_t, s_h, s_w = self.stride
        if self not in started:
            started.add(self)
            hidden_states = torch.cat([hidden_states[:, :, : s_t - 1], hidden_states], dim=2)
        if hidden_states.shape[2] % s_t:
            raise BackendError("a chunk of the source did not fold evenly at the video VAE's temporal downsampling")
        residual = hidden_states.unflatten(4, (-1, s_w)).unflatten(3, (-1, s_h)).unflatten(2, (-1, s_t))
        residual = residual.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        residual = residual.unflatten(1, (-1, self.group_size)).mean(dim=2)
        hidden_states = self.conv(hidden_states, causal=causal)
        hidden_states = hidden_states.unflatten(4, (-1, s_w)).unflatten(3, (-1, s_h)).unflatten(2, (-1, s_t))
        hidden_states = hidden_states.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        return hidden_states + residual

    inner = {module.conv for module in encoder.modules() if isinstance(module, LTX2VideoCausalConv3d)}
    try:
        for module in encoder.modules():
            if isinstance(module, (LTX2VideoCausalConv3d, LTX2VideoDownsampler3d)):
                step = causal_conv if isinstance(module, LTX2VideoCausalConv3d) else downsampler
            elif isinstance(module, torch.nn.Conv3d) and module not in inner and (module.kernel_size[0], module.stride[0]) != (1, 1):
                raise BackendError("the video VAE's encoder has a temporal convolution the chunked encode doesn't know")
            else:
                continue
            if "forward" in module.__dict__:  # a hook (offload, layerwise casting) already wraps this layer
                raise BackendError("the video VAE's encoder layers are hooked; it can't be encoded in chunks")
            module.forward = step.__get__(module)
            patched.append(module)
        yield lambda chunk: encoder(chunk, causal=True)
    finally:
        for module in patched:
            del module.forward
        held.clear()


def encode_chunked(vae: Any, pixels: Callable[[int, int], Any], frames: int, latent_frames_per_chunk: int = CHUNK_LATENT_FRAMES):
    """The posterior mode of `vae` (AutoencoderKLLTX2Video) for a clip of `frames` frames, [1, 128, latent frames, h, w],
    as `vae.encode(clip).latent_dist.mode()` without spatial or temporal tiling. `pixels(start, end)` returns frames
    [start, end) as the VAE's input ([1, 3, end - start, H, W] in [-1, 1], its dtype, on its device), so only one chunk's
    pixels are there at a time. Offload hooks run first, as they do for `vae.encode`."""
    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.utils.accelerate_utils import apply_forward_hook

    def run(vae):
        latents = []
        with torch.no_grad(), streaming(vae.encoder) as encode:
            for start, end in chunk_bounds(frames, int(vae.temporal_compression_ratio), latent_frames_per_chunk):
                chunk = pixels(start, end)
                moments = encode(chunk)
                del chunk
                # A copy of the mean's half, so the chunk's full moments (and their log-variance) free now.
                latents.append(DiagonalGaussianDistribution(moments).mode().clone())
                del moments
        return torch.cat(latents, dim=2)

    return apply_forward_hook(run)(vae)
