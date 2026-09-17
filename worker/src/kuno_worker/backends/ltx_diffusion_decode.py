"""LTX-2.5's diffusion decoder (diffusers 0.40 `LTX2VideoDiffusionDecoderModel`) for ltx-2.5-4k: latents to 1440p and 2160p
frames, with attention that fits a GPU, and the decode's memory as plain arithmetic for the planner.

What the decoder is (read from diffusers 0.40's `models/autoencoders/ltx2_diffusion_decoder.py` in the worker image). A
417M-parameter model (0.777 GiB in bf16, the checkpoint's `diffusion_decoder/` file) that replaces the convolutional VAE's
decoder, with the same geometry: 32x in space and 8x in time, so it upscales nothing beyond what the VAE decodes and does
not change the frame rate. Four deterministic stages of neighborhood attention and pixel-shuffle upsampling build a context
volume from the latents (widths 2048, 1024, 512, 512; depths 4, 6, 4, 2; windows 3x7x7, 3x7x7, 3x5x5, 3x5x5), then a fifth
stage denoises patchified pixels (4x4 patches, width 256, 8 blocks, 11x11x11 windows) conditioned on it. As LTX-2.5 ships
it (`decoder_model_output_type` "x0", `decoder_num_inference_steps` 1) stage 5 runs once from Gaussian noise and its
prediction is the frame, so the decode draws noise and needs a generator to repeat. Tiling (`enable_tiling`, defaults 768 px
tiles every 704 px and 80 frames every 56) runs the last deterministic stage and stage 5 per tile; stages 1-3 always see the
whole volume.

Why not diffusers' attention processors.
  LTX2VideoVaeNeighborhoodAttnProcessor (the default)  FlexAttention. Its mask is built with `create_block_mask`, which
      evaluates the window rule on a dense query x key grid, and without torch.compile `flex_attention` "materializes the
      full scores matrix" (torch 2.11's own warning). A 256x256 clip of 17 frames was killed for memory on a 61 GiB CPU host
      with a tiny decoder (2026-09-17); the model's own comment puts one production stage's mask at 167 GiB. Compiling it
      needs Triton to build its launcher with a C compiler, and the worker image has none.
  LTX2VideoVaeNeighborhoodNattenProcessor  downloads NATTEN from the Hub through the `kernels` package at load. The image
      has no `kernels`, runs with HF_HUB_OFFLINE=1, and code fetched at run time would not be the attested image's.

ChunkedNeighborhoodAttnProcessor computes the same attention exactly with PyTorch's scaled_dot_product_attention. Every
query attends to the `kernel_size` window centred on it, shifted inward at the grid's borders so it always holds exactly
`kernel_size` positions (what NATTEN's na3d and the flex mask define). Queries are cut into chunks of one window's size along
each axis; a chunk's keys are the box its queries' windows span (at most chunk + kernel - 1 per axis), and a boolean mask
keeps each query to its own window inside the box. Interior chunks share one mask, so a run of them is a view (`unfold`) of
the key slab and one batched call. Query, key and value projections run per temporal chunk, with rotary positions counted
from the grid's origin, so each element is computed by the same operations as the whole-volume path. Results equal
diffusers' flex processor's to rounding (test_ltx_4k_render.py). Memory: one temporal chunk's projections and one batch of
gathered keys (`budget_bytes`), not the grid squared. Speed is unmeasured: each query's box holds about 7x its window at
stage 5, and the boxes are copied, so expect it slower than NATTEN (scripts/gpu-test/long_video/run_4k_worker.py times it).

Memory (`decode_activation_bytes`, used by quantized.MemoryPlan). The decode's live tensors beside the weights, in bf16
bytes, followed through `tiled_decode` in the order it allocates and frees them (`decode_phases`), for F = 1 + 8(T - 1)
frames of W x H (T latent frames):
  stages 1-3   the whole volume. The peak is the last upsample: its input (512 channels on (2T + 3) frames of H/16 x W/16)
               with its projection and the permuted copy (512 channels on (4T + 6) frames of H/8 x W/8 each), so
               4(2T + 3)HW + 32(4T + 6)HW.
  features     that output, `16(4T + 6)HW`, alive until the decode returns.
  each tile    stage 4 and stage 5 on one tile of the stage-4 grid (a cell is 8 x 8 px and 2 frames; the defaults cut tiles of
               96 x 96 x 40 cells, the trailing one in time carrying 8 ghost cells more): TILE_BYTES_PER_TOKEN for each token
               stage 5 denoises (4 x 4 px of a frame) and TILE_BYTES_PER_GHOST_CELL for each ghost cell. That includes the
               previous tile's context, which lives until this tile's stage 4 returns.
  pixels       3 channels of bf16, 6 bytes a pixel-frame. A temporal group's tiles are kept until the group is joined; the row
               joins and the group join are copies; the previous group's rows live until this group's tiles are done; every
               group lives until the final join, which copies them all once more. Then the pipeline's `pt` output holds the
               decode, the [0, 1] rescale and its stack.
Counted as live tensors (2026-09-17), with this processor and the default tiling, at the decoder's real widths in bf16:
  on the meta device (shapes only, nothing allocated; SDPA as a fused kernel that keeps no score matrix, as the GPU's
  memory-efficient and cuDNN kernels do), whole decodes and the pipeline's `pt` output, peak GiB:
      frames       49      97     241
      1440p     10.45   16.42   27.00
      2160p     14.93   24.41   47.06
  the stage 1-3 peaks and the features within 0.06 GiB of the formulas above, and the plan's arithmetic with 5,300 bytes
  a stage-5 token and no workspace above every peak, by 4-16%;
  on the CPU, allocated and run, at 256 px for 17 to 81 frames: the formulas to the byte, a production-depth tile (79
  frames) 3,900-4,700 bytes a token, and the whole decode within the plan's estimate.
test_ltx_4k_render.py repeats the meta count at 1440p for 49 frames. Beside the counted activations, the attention's
per-chunk projections and gathered keys and diffusers' 16,384-token MLP tiles (ATTENTION_WORKSPACE_BYTES,
PROJECTION_BYTES_PER_PIXEL), and a fifth more per token for the GPU. Never run on a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

# ---------------------------------------------------------------- the decoder's geometry and memory (LTX-2.5's config)

SPATIAL_RATIO = 32
TEMPORAL_RATIO = 8
# Two trailing latent frames are replicated through stages 1-3 so NATTEN's inward-shifted windows keep the last real frames
# away from the border (`trailing_pad_latent_frames`): 8 cells of the stage-4 grid.
GHOST_CELLS = 8
# The tiling grid is stage 4's input: 8 pixels a cell in space, 2 frames in time.
CELL_PIXELS, CELL_FRAMES = 8, 2
# The smallest tile the remaining neighborhood kernels accept, in cells (tiled_decode's `min_sizes`): max(stage 4's
# kernel, ceil(stage 5's 11 / the last upsample's stride of 2)).
TILE_MIN_CELLS = 6
# Pixel frames the decoder needs at least: stage 5's window is 11 frames (a clip of 2 latent frames, 9 frames, is refused).
MIN_FRAMES = 11


@dataclass(frozen=True)
class Tiling:
    """`enable_tiling`'s sizes, in pixels and frames; the defaults are diffusers 0.40's, which the worker uses."""

    tile_px: int = 768
    stride_px: int = 704
    tile_frames: int = 80
    stride_frames: int = 56


DEFAULT_TILING = Tiling()
# A tile's stage 4 and stage 5, bf16 bytes: TILE_BYTES_PER_TOKEN for each token stage 5 denoises (4 x 4 pixels of a frame)
# and TILE_BYTES_PER_GHOST_CELL for each ghost cell only stage 4 sees (its blocks on 512 channels, then
# the last upsample's projection and copy). Counted at 1440p and 2160p (module docstring): with 5,300 bytes a token the
# plan covers each count's peak by 4-16%. The figure adds a fifth for what a count of tensors can't see, the GPU
# allocator's rounding and cuBLAS and SDPA workspaces. (The edit modes' unchunked VAE encode measured 1.374x its CPU count
# on a GPU, cuDNN's convolution workspaces included; the decoder has no convolutions.)
TILE_BYTES_PER_TOKEN = 6_400
TILE_BYTES_PER_GHOST_CELL = 16_000
PIXEL_BYTES = 6
# Keys and values one attention call gathers, at most (bf16 bytes).
ATTENTION_BUDGET_BYTES = 1 << 30
# Beside the counted activations: the gathered keys and values (the budget), SDPA's output, the MLP's hidden-width tile
# (16,384 tokens x 8,192 x 3 in stage 1, 0.75 GiB) and slack.
ATTENTION_WORKSPACE_BYTES = 2 << 30
# The per-temporal-chunk query, key and value projections of stages 1-3 and their rotary transients, bytes a pixel of the
# frame size: at most stage 2's 13 frames of H/16 x W/16 x 1,024 channels, and the float32 rotation of a key slab.
PROJECTION_BYTES_PER_PIXEL = 400
# The latent render's result held through the decode: the latents, the audio track and its decode (a 10 s track is under
# 0.1 GiB at every stage). A round figure.
DECODE_HELD_BYTES = 1 << 28


def tile_intervals(length: int, tile: int, stride: int, min_size: int) -> list[tuple[int, int]]:
    """diffusers' `_tile_intervals`: overlapping [start, end) tiles covering [0, length), a trailing remnant shorter than
    `min_size` merged into the tile before it."""
    if length <= tile:
        return [(0, length)]
    starts = list(range(0, length, stride))
    while len(starts) > 1 and length - starts[-1] < min_size:
        starts.pop()
    return [(start, min(start + tile, length)) for start in starts[:-1]] + [(starts[-1], length)]


def decode_phases(width: int, height: int, frames: int, tiling: Tiling = DEFAULT_TILING, per_token: int = TILE_BYTES_PER_TOKEN,
                  per_ghost: int = TILE_BYTES_PER_GHOST_CELL, workspace: bool = True) -> dict[str, int]:
    """The peak live bytes beside the weights in each phase of `LTX2VideoDiffusionDecoderModel.decode` and the decode
    pipeline's `pt` output, followed tile by tile (module docstring). `workspace=False` leaves out the attention's and the
    MLP's bounded workspaces, which a CPU count with a small budget doesn't reach."""
    latent_frames = (frames - 1) // TEMPORAL_RATIO + 1
    pixels = width * height
    extra = ATTENTION_WORKSPACE_BYTES if workspace else 0
    stages = 4 * (2 * latent_frames + 3) * pixels + 32 * (4 * latent_frames + 6) * pixels
    features = 16 * (4 * latent_frames + 6) * pixels
    output = PIXEL_BYTES * frames * pixels
    cell_frames, cell_rows, cell_cols = 4 * latent_frames - 3, height // CELL_PIXELS, width // CELL_PIXELS
    phases = {"stages_1_3": stages + (PROJECTION_BYTES_PER_PIXEL * pixels + extra if workspace else 0), "postprocess": 3 * output}
    tiled = latent_frames > tiling.tile_frames // TEMPORAL_RATIO or height // SPATIAL_RATIO > tiling.tile_px // SPATIAL_RATIO or (
        width // SPATIAL_RATIO > tiling.tile_px // SPATIAL_RATIO
    )
    if not tiled:
        tokens = frames * (height // 4) * (width // 4)
        phases["tiles"] = features + per_token * tokens + per_ghost * GHOST_CELLS * cell_rows * cell_cols + extra
        return phases
    temporal = tile_intervals(cell_frames, tiling.tile_frames // CELL_FRAMES, tiling.stride_frames // CELL_FRAMES, TILE_MIN_CELLS)
    rows = tile_intervals(cell_rows, tiling.tile_px // CELL_PIXELS, tiling.stride_px // CELL_PIXELS, TILE_MIN_CELLS)
    cols = tile_intervals(cell_cols, tiling.tile_px // CELL_PIXELS, tiling.stride_px // CELL_PIXELS, TILE_MIN_CELLS)
    done = previous_rows = tiles_peak = join_peak = 0
    tiles_of_group = 0
    for t0, t1 in temporal:
        group_frames = CELL_FRAMES * (t1 - t0) - (1 if t0 == 0 else 0)
        ghost = GHOST_CELLS if t1 == cell_frames else 0
        tiles_of_group = 0  # `rows = []`: the previous group's tiles are freed
        for h0, h1 in rows:
            for w0, w1 in cols:
                area = (h1 - h0) * (w1 - w0)
                work = per_token * group_frames * 4 * area + per_ghost * ghost * area
                tiles_peak = max(tiles_peak, features + done + previous_rows + tiles_of_group + work)
                tiles_of_group += PIXEL_BYTES * group_frames * area * CELL_PIXELS * CELL_PIXELS
        group = PIXEL_BYTES * group_frames * pixels
        # `result_rows = []` frees the previous group's rows; this group's row joins, then the group join, while its tiles live.
        join_peak = max(join_peak, features + done + tiles_of_group + 2 * group)
        done += group
        previous_rows = group
    # The final join copies every group once more, while the last group's tiles and rows are still referenced.
    join_peak = max(join_peak, features + done + tiles_of_group + previous_rows + output)
    phases["tiles"] = tiles_peak + extra
    phases["join"] = join_peak
    return phases


def decode_activation_bytes(width: int, height: int, frames: int) -> int:
    """The decode's peak beside the weights, bf16 bytes: its largest phase, plus the render's result it holds."""
    return max(decode_phases(width, height, frames).values()) + DECODE_HELD_BYTES


# ---------------------------------------------------------------- exact neighborhood attention in chunks


@dataclass(frozen=True)
class Segment:
    """A run of equal query chunks along one axis that share a mask. Chunk j holds queries [start + j*size, + size) and
    attends inside the key box [key_start + j*size, + keys); `offsets[a]` is where query a's window starts in its box."""

    start: int
    size: int
    count: int
    key_start: int
    keys: int
    offsets: tuple[int, ...]


def window_start(index: int, length: int, kernel: int) -> int:
    """Where the window of the query at `index` starts: centred, shifted inward at the borders (NATTEN's na3d)."""
    return min(max(index - kernel // 2, 0), length - kernel)


def axis_segments(length: int, kernel: int, chunk: int | None = None) -> list[Segment]:
    """Chunks covering [0, length) exactly once, merged into runs that share a mask and are evenly spaced (every interior
    chunk: one run)."""
    kernel = min(kernel, length)
    size = min(chunk or kernel, length)
    keys = min(length, size + kernel - 1)
    segments: list[Segment] = []
    for start in range(0, length, size):
        end = min(start + size, length)
        # The box holds every window of the chunk: window starts only grow along the axis, and span at most size - 1.
        key_start = max(0, min(window_start(start, length, kernel), length - keys))
        offsets = tuple(window_start(i, length, kernel) - key_start for i in range(start, end))
        last = segments[-1] if segments else None
        if (
            last is not None and last.size == end - start and last.offsets == offsets
            and start == last.start + last.count * last.size and key_start == last.key_start + last.count * last.size
        ):
            segments[-1] = replace(last, count=last.count + 1)
        else:
            segments.append(Segment(start, end - start, 1, key_start, keys, offsets))
    return segments


def _axis_mask(segment: Segment, kernel: int, device: Any):
    import torch

    keys = torch.arange(segment.keys, device=device)
    offsets = torch.tensor(segment.offsets, device=device)[:, None]
    return (keys[None, :] >= offsets) & (keys[None, :] < offsets + kernel)


def rotate(rope: Any, x: Any, origin: tuple[int, int, int]) -> Any:
    """`LTX2VideoVaeRotaryPosEmbed3D.forward` for a slab of a grid whose first cell is at `origin`: the same operations on
    the same positions as the whole grid's rotation, so the slab's values are the whole grid's."""
    import torch

    dim_t, dim_h, _ = rope.rope_dim_split
    frames, height, width = x.shape[1:4]
    device = x.device
    inv_t, inv_h, inv_w = (rope._inv_freqs(dim, device) for dim in rope.rope_dim_split)
    t0, h0, w0 = origin
    positions_t = torch.arange(t0, t0 + frames, dtype=torch.float32, device=device)
    positions_h = torch.arange(h0, h0 + height, dtype=torch.float32, device=device)
    positions_w = torch.arange(w0, w0 + width, dtype=torch.float32, device=device)
    rotated_t = rope._rotate_axis(x[..., :dim_t], positions_t, inv_t, axis=1)
    rotated_h = rope._rotate_axis(x[..., dim_t : dim_t + dim_h], positions_h, inv_h, axis=2)
    rotated_w = rope._rotate_axis(x[..., dim_t + dim_h :], positions_w, inv_w, axis=3)
    return torch.cat([rotated_t, rotated_h, rotated_w], dim=-1)


def _queries(attn: Any, x: Any, origin: tuple[int, int, int]) -> Any:
    """`project_qkv`'s query for the slab `x`: projected, RMS-normed, pre-scaled, rotated."""
    shape = (*x.shape[:4], attn.heads, attn.head_dim)
    return rotate(attn.rope, attn.norm_q(attn.to_q(x).view(shape)) * attn.scale, origin)


def _keys_values(attn: Any, x: Any, origin: tuple[int, int, int]) -> tuple[Any, Any]:
    shape = (*x.shape[:4], attn.heads, attn.head_dim)
    return rotate(attn.rope, attn.norm_k(attn.to_k(x).view(shape)), origin), attn.to_v(x).view(shape)


class ChunkedNeighborhoodAttnProcessor:
    """Neighborhood attention for `LTX2VideoVaeNeighborhoodAttention`, exact and in bounded memory (module docstring).

    Not a subclass of diffusers' flex processor, so the decoder never builds a flex mask for it (`build_block_mask` returns
    None for any other processor). `budget_bytes` caps the keys and values one batched call gathers."""

    def __init__(self, budget_bytes: int = ATTENTION_BUDGET_BYTES):
        self.budget_bytes = int(budget_bytes)

    def __call__(self, attn: Any, hidden_states: Any, block_mask: Any = None) -> Any:
        import torch.nn.functional as F

        batch, frames, height, width, _ = hidden_states.shape
        kernel = tuple(min(k, n) for k, n in zip(attn.kernel_size, (frames, height, width)))
        heads, head_dim = attn.heads, attn.head_dim
        device, element = hidden_states.device, hidden_states.element_size()
        out = hidden_states.new_empty((batch, frames, height, width, attn.to_out[0].out_features))
        along_t, along_h, along_w = (axis_segments(n, k) for n, k in zip((frames, height, width), kernel))
        masks_h = [_axis_mask(s, kernel[1], device) for s in along_h]
        masks_w = [_axis_mask(s, kernel[2], device) for s in along_w]
        for seg_t in along_t:
            mask_t = _axis_mask(seg_t, kernel[0], device)
            for j in range(seg_t.count):
                t0, k0 = seg_t.start + j * seg_t.size, seg_t.key_start + j * seg_t.size
                q = _queries(attn, hidden_states[:, t0 : t0 + seg_t.size], (t0, 0, 0))
                k, v = _keys_values(attn, hidden_states[:, k0 : k0 + seg_t.keys], (k0, 0, 0))
                for seg_h, mask_h in zip(along_h, masks_h):
                    for seg_w, mask_w in zip(along_w, masks_w):
                        mask = (
                            mask_t[:, None, None, :, None, None] & mask_h[None, :, None, None, :, None] & mask_w[None, None, :, None, None, :]
                        ).reshape(seg_t.size * seg_h.size * seg_w.size, seg_t.keys * seg_h.keys * seg_w.keys)
                        per_row = seg_w.count * 2 * (mask.shape[0] + mask.shape[1]) * heads * head_dim * element
                        rows = max(1, min(seg_h.count, self.budget_bytes // max(per_row, 1)))
                        for r0 in range(0, seg_h.count, rows):
                            self._attend(attn, F, out, q, k, v, mask, seg_t, seg_h, seg_w, t0, r0, min(rows, seg_h.count - r0))
        return out

    @staticmethod
    def _attend(attn, F, out, q, k, v, mask, seg_t, seg_h, seg_w, t0, r0, rows) -> None:
        """One batched call: `rows` chunks along h (from chunk r0) times every chunk of `seg_w`, for temporal chunk t0."""
        batch, heads, head_dim = q.shape[0], attn.heads, attn.head_dim
        n = batch * rows * seg_w.count
        h_q = seg_h.start + r0 * seg_h.size
        h_k = seg_h.key_start + r0 * seg_h.size

        def windows(x, h, size, stride, w, w_size):
            span_h, span_w = (rows - 1) * stride + size, (seg_w.count - 1) * seg_w.size + w_size
            # (B, t, rows, chunks_w, heads, head_dim, size, w_size) -> (n, heads, t x size x w_size, head_dim)
            view = x[:, :, h : h + span_h, w : w + span_w].unfold(2, size, stride).unfold(3, w_size, seg_w.size)
            return view.permute(0, 2, 3, 4, 1, 6, 7, 5).reshape(n, heads, -1, head_dim)

        queries = windows(q, h_q, seg_h.size, seg_h.size, seg_w.start, seg_w.size)
        keys = windows(k, h_k, seg_h.keys, seg_h.size, seg_w.key_start, seg_w.keys)
        values = windows(v, h_k, seg_h.keys, seg_h.size, seg_w.key_start, seg_w.keys)
        # `scale=1.0`: the query was scaled before its rotation, as in the reference.
        attended = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask, scale=1.0)
        del queries, keys, values
        attended = attn.to_out[0](attended.transpose(1, 2).reshape(n, -1, heads * head_dim))
        channels = attended.shape[-1]
        attended = attended.view(batch, rows, seg_w.count, seg_t.size, seg_h.size, seg_w.size, channels)
        attended = attended.permute(0, 3, 1, 4, 2, 5, 6).reshape(batch, seg_t.size, rows * seg_h.size, seg_w.count * seg_w.size, channels)
        out[:, t0 : t0 + seg_t.size, h_q : h_q + rows * seg_h.size, seg_w.start : seg_w.start + seg_w.count * seg_w.size] = attended


def use_chunked_attention(decoder: Any, budget_bytes: int = ATTENTION_BUDGET_BYTES) -> int:
    """Gives every neighborhood attention of `decoder` (the model or its pipeline's) the chunked processor; returns how many."""
    from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoVaeNeighborhoodAttention

    processor = ChunkedNeighborhoodAttnProcessor(budget_bytes)
    count = 0
    for module in decoder.modules():
        if isinstance(module, LTX2VideoVaeNeighborhoodAttention):
            module.set_processor(processor)
            count += 1
    return count


def prepare_decoder(decoder: Any, budget_bytes: int = ATTENTION_BUDGET_BYTES) -> Any:
    """The decoder as ltx-2.5-4k runs it: chunked attention everywhere and diffusers' default tiling."""
    if not use_chunked_attention(decoder, budget_bytes):
        raise ValueError("the diffusion decoder has no neighborhood attention to replace")
    decoder.enable_tiling()
    return decoder


# ---------------------------------------------------------------- decoding a render


def decode_frames(pipeline: Any, latents: Any, generator: Any) -> list:
    """RGB uint8 frames (H x W x 3 numpy arrays) of the first video in `latents`, as LTX-2 pipelines return them from
    `output_type="latent"` (already denormalized). `pipeline` is diffusers' LTX2VideoDiffusionDecodePipeline. Its `pt` output
    is [0, 1] in the decoder's dtype; each frame becomes uint8 on the device as diffusers' `pil` output does on the host
    ((x * 255).round() in float32), so the host holds 3 bytes a pixel rather than float32 copies of the whole clip."""
    import torch

    (video,) = pipeline(latents=latents, generator=generator, output_type="pt", return_dict=False, denormalize=False)
    frames = []
    with torch.no_grad():
        for index in range(video.shape[1]):
            pixel = (video[0, index].float() * 255).round_().to(torch.uint8)
            frames.append(pixel.permute(1, 2, 0).contiguous().cpu().numpy())
    del video
    return frames


def decode_audio(pipeline: Any, audio_latents: Any) -> Any:
    """The waveform of the audio latents an LTX-2 pipeline returns from `output_type="latent"` (denormalized, unpacked):
    what its own non-latent path does, the audio VAE and then the vocoder."""
    mel = pipeline.audio_vae.decode(audio_latents.to(pipeline.audio_vae.dtype), return_dict=False)[0]
    return pipeline.vocoder(mel)
