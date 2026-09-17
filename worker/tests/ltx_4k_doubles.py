"""ltx-2.5-4k's runtime with tiny random weights, for the CPU: diffusers' real LTX2Pipeline and LTX2ConditionPipeline (from
ltx_storyboard_doubles.tiny_pipelines), its latent upsampler, and a real LTX2VideoDiffusionDecoderModel shrunk to widths
(128, 64, 32, 32, 16) with smaller windows, behind diffusers' LTX2VideoDiffusionDecodePipeline, as
quantized.build_ltx_pipelines assembles them. The pictures are noise; the passes, shapes, hooks, tiling and lengths are the
real code's.

No text encoder: the pipelines' `encode_prompt` returns embeddings seeded from the prompt, as TinyRenderer's `prepare_call`
does for the storyboard doubles, so `runtimes.LtxAdapter` sends them exactly the call it sends the real pipelines.
"""

from __future__ import annotations

import inspect
from typing import Any

from ltx_storyboard_doubles import TinyRenderer, stable_seed, tiny_pipelines

# The tiny decoder's windows: LTX-2.5's are 3x7x7, 3x7x7, 3x5x5, 3x5x5 and 11x11x11, which need grids of at least 7 latent
# rows and 11 frames; these keep a 320x192 clip of 49 frames tiled in every direction.
TINY_KERNELS = ((3, 5, 5), (3, 5, 5), (3, 3, 3), (3, 3, 3))
TINY_STAGE5_KERNEL = (5, 5, 5)
# Tiles of 128 px every 96 px and 16 frames every 8: a 320x192 clip of 49 frames is cut into 5 temporal groups of 2 x 3 tiles.
TINY_TILING = {"tile_sample_min_height": 128, "tile_sample_min_width": 128, "tile_sample_min_num_frames": 16,
               "tile_sample_stride_height": 96, "tile_sample_stride_width": 96, "tile_sample_stride_num_frames": 8}


def _embeddings(prompt: str | None, device: Any):
    import torch

    shape = (1, TinyRenderer.text_tokens, TinyRenderer.text_channels * (TinyRenderer.text_layers + 1))
    generator = torch.Generator("cpu").manual_seed(stable_seed(prompt or ""))
    return torch.randn(shape, generator=generator).to(device), torch.ones(1, TinyRenderer.text_tokens, dtype=torch.long, device=device)


def _encode_prompt(self, prompt=None, negative_prompt=None, do_classifier_free_guidance=True, num_videos_per_prompt=1, prompt_embeds=None,
                   negative_prompt_embeds=None, prompt_attention_mask=None, negative_prompt_attention_mask=None, max_sequence_length=1024,
                   scale_factor=8, device=None, dtype=None):
    """A stand-in for Gemma: embeddings seeded from the prompt, zeros for the negative one."""
    import torch

    embeds, mask = _embeddings(prompt, device or "cpu")
    return embeds, mask, torch.zeros_like(embeds), torch.ones_like(mask)


def tiny_decoder(seed: int = 0, kernels=TINY_KERNELS, stage5_kernel=TINY_STAGE5_KERNEL, dtype=None):
    """A real LTX2VideoDiffusionDecoderModel with LTX-2.5's geometry (128 latent channels, 32x and 8x, 4x4 patches, one x0
    step) at tiny widths. Head dimension 16 splits its rotary embedding (4, 6, 6) across time, height and width."""
    import torch
    from diffusers import LTX2VideoDiffusionDecoderModel

    torch.manual_seed(seed)
    decoder = LTX2VideoDiffusionDecoderModel(
        decoder_head_dim=16, decoder_stage_channels=(128, 64, 32, 32, 16), decoder_stage_depths=(1, 1, 1, 1, 2),
        decoder_stage_kernels=kernels, decoder_stage5_kernel=stage5_kernel, decoder_t_emb_dim=32,
    ).eval()
    return decoder if dtype is None else decoder.to(dtype)


def tiny_4k_pipelines(seed: int = 0, budget_bytes: int = 1 << 20) -> dict[str, Any]:
    """The pipelines dict `quantized.build_ltx_pipelines` returns for ltx-2.5-dfr/bf16/1, tiny: text, condition, upsample and
    the diffusion decode pipeline, with the chunked attention (a 1 MiB budget, so calls split into batches) and tiling."""
    from diffusers import FlowMatchEulerDiscreteScheduler, LTX2ConditionPipeline, LTX2Pipeline, LTX2VideoDiffusionDecodePipeline

    from kuno_worker.backends.ltx_diffusion_decode import prepare_decoder

    base = tiny_pipelines(seed=seed)
    components = base["condition"].components
    accepted = inspect.signature(LTX2Pipeline.__init__).parameters

    class TinyTextPipeline(LTX2Pipeline):
        encode_prompt = _encode_prompt

    class TinyConditionPipeline(LTX2ConditionPipeline):
        encode_prompt = _encode_prompt

    # The loader builds the condition pipeline from the text pipeline's components; here the other way round, less what
    # LTX2Pipeline doesn't take (the condition pipeline's own audio scheduler).
    text = TinyTextPipeline(**{name: module for name, module in components.items() if name in accepted})
    condition = TinyConditionPipeline(**components)
    for pipeline in (text, condition):
        pipeline.set_progress_bar_config(disable=True)
    decoder = prepare_decoder(tiny_decoder(seed), budget_bytes)
    decoder.enable_tiling(**TINY_TILING)
    decode = LTX2VideoDiffusionDecodePipeline(diffusion_decoder=decoder, scheduler=FlowMatchEulerDiscreteScheduler())
    return {"text": text, "condition": condition, "audio": text, "upsample": base["upsample"], "decode": decode}
