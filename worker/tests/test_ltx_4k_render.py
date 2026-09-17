"""ltx-2.5-4k through diffusers' real classes with tiny random weights: the chunked neighborhood attention is diffusers' own to
rounding, the diffusion decoder decodes the same frames with it as with diffusers' FlexAttention processor, a 4K job renders
end to end through LtxResidentBackend.generate (latents from the distilled passes, frames from LTX2VideoDiffusionDecodePipeline,
sound from the audio VAE and vocoder), a seed repeats its frames, and the loader builds the decode pipeline only from a
recipe that verified the decoder's files. The pictures are noise; the passes, shapes, tiling and lengths are the real code's.
Skips where torch and diffusers are not installed; runs in the LTX worker image:

    S=/video/.venv/lib/python3.12/site-packages; docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=kuno \\
      -v /video/subnet:/src:ro $(for m in pytest _pytest pluggy iniconfig py.py; do printf -- '-v %s/%s:/pt/%s:ro ' "$S" "$m" "$m"; done) \\
      -e PYTHONPATH=/src/worker/src:/src/protocol/src:/pt:/src/worker/tests -e PYTHONDONTWRITEBYTECODE=1 -w /tmp --entrypoint python \\
      kuno-worker:ltx -m pytest /src/worker/tests/test_ltx_4k_render.py -q -p no:cacheprovider
"""

from __future__ import annotations

import inspect
import subprocess
import uuid
from pathlib import Path

import numpy as np
import pytest

from kuno_protocol.mp4 import probe
from kuno_protocol.precision import PrecisionError, load_recipes
from kuno_protocol.profiles import InputRole, Mode, load_profiles, ltx_num_frames
from kuno_protocol.schemas import GenerationParams, InputRef
from kuno_worker.backends.base import GenerationTask, InputFile
from kuno_worker.backends.ltx_resident import DIFFUSION_DECODER, LtxResidentBackend, build_call
from kuno_worker.backends.media_tools import ffmpeg_exe

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from ltx_4k_doubles import TINY_TILING, tiny_4k_pipelines, tiny_decoder  # noqa: E402

PROFILES = load_profiles()
FOUR_K = PROFILES["ltx-2.5-4k"]
WIDTH, HEIGHT = 320, 192  # a 4K job's call, at a size the CPU renders in seconds (divisible by 64 for the half-size pass)


def attention(dim: int, kernel, head_dim: int = 8):
    from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoVaeNeighborhoodAttention

    torch.manual_seed(3)
    return LTX2VideoVaeNeighborhoodAttention(dim=dim, kernel_size=kernel, head_dim=head_dim).eval()


# ---------------------------------------------------------------- the attention


@pytest.mark.parametrize(
    ("grid", "kernel", "budget", "batch"),
    [
        ((5, 9, 13), (3, 5, 5), 1 << 30, 1),      # interior runs along both spatial axes
        ((4, 7, 7), (3, 7, 7), 1 << 30, 2),       # a window as large as the grid, and a batch of two
        ((11, 12, 23), (11, 11, 11), 1 << 30, 1),  # stage 5's window: a short last chunk and a clipped key box
        ((6, 17, 10), (3, 5, 5), 1, 1),           # a budget of one byte: one row of chunks per call
        ((13, 30, 31), (3, 7, 7), 200_000, 1),    # several batched calls per temporal chunk
        ((3, 5, 5), (3, 5, 5), 1 << 20, 1),       # the smallest grid the kernel accepts
    ],
)
def test_the_chunked_attention_is_diffusers_neighborhood_attention(grid, kernel, budget, batch):
    from kuno_worker.backends.ltx_diffusion_decode import ChunkedNeighborhoodAttnProcessor

    attn = attention(32, kernel)
    x = torch.randn(batch, *grid, 32, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        reference = attn(x)  # diffusers' default: FlexAttention over its neighborhood block mask
        attn.set_processor(ChunkedNeighborhoodAttnProcessor(budget))
        chunked = attn(x)
    assert chunked.shape == reference.shape
    assert float((chunked - reference).abs().max()) <= 1e-5 * float(reference.abs().max())


def test_the_segments_cover_every_query_once_inside_its_windows_box():
    from kuno_worker.backends.ltx_diffusion_decode import axis_segments, window_start

    for length in range(1, 40):
        for kernel in (3, 5, 7, 11):
            seen = []
            for seg in axis_segments(length, kernel):
                for j in range(seg.count):
                    box = seg.key_start + j * seg.size
                    assert 0 <= box and box + seg.keys <= length
                    for a in range(seg.size):
                        query = seg.start + j * seg.size + a
                        start = window_start(query, length, min(kernel, length))
                        assert box + seg.offsets[a] == start and start + min(kernel, length) <= box + seg.keys
                        seen.append(query)
            assert seen == list(range(length)), (length, kernel)
    # Interior chunks share one mask and are evenly spaced: one run each, however long the axis.
    assert len(axis_segments(544, 11)) <= 4 and len(axis_segments(960, 11)) <= 4


# ---------------------------------------------------------------- the decoder


@pytest.mark.parametrize("tiled", [False, True])
def test_the_decoder_decodes_the_same_frames_with_chunked_attention_as_with_flex(tiled, monkeypatch):
    from diffusers.models.autoencoders import ltx2_diffusion_decoder

    from kuno_worker.backends.ltx_diffusion_decode import use_chunked_attention

    decoder = tiny_decoder(kernels=((3, 3, 3),) * 4, stage5_kernel=(3, 3, 3))
    if tiled:
        decoder.enable_tiling(tile_sample_min_height=64, tile_sample_min_width=64, tile_sample_min_num_frames=16,
                              tile_sample_stride_height=48, tile_sample_stride_width=48, tile_sample_stride_num_frames=8)
    latents = torch.randn(1, 128, 3, 3, 3, generator=torch.Generator().manual_seed(1))  # 17 frames of 96x96
    with torch.no_grad():
        reference = decoder.decode(latents, generator=torch.Generator().manual_seed(9), return_dict=False)[0]
        assert use_chunked_attention(decoder) == 6  # one attention in each block of the five stages (depths 1, 1, 1, 1, 2)
        # The chunked processor never has the decoder build a flex block mask, whose size is the grid squared.
        monkeypatch.setattr(ltx2_diffusion_decoder, "_neighborhood_block_mask", lambda *a, **k: pytest.fail("built a flex mask"))
        chunked = decoder.decode(latents, generator=torch.Generator().manual_seed(9), return_dict=False)[0]
    assert chunked.shape == reference.shape == (1, 3, 17, 96, 96)
    assert float((chunked - reference).abs().max()) <= 1e-4 * float(reference.abs().max())


def test_prepare_decoder_gives_every_attention_the_chunked_processor_and_tiles():
    from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoVaeNeighborhoodAttention

    from kuno_worker.backends.ltx_diffusion_decode import ChunkedNeighborhoodAttnProcessor, prepare_decoder

    decoder = prepare_decoder(tiny_decoder(), budget_bytes=12345)
    attentions = [m for m in decoder.modules() if isinstance(m, LTX2VideoVaeNeighborhoodAttention)]
    assert attentions and all(isinstance(m.processor, ChunkedNeighborhoodAttnProcessor) and m.processor.budget_bytes == 12345 for m in attentions)
    assert decoder.use_tiling and (decoder.tile_sample_min_height, decoder.tile_sample_stride_num_frames) == (768, 56)


def test_the_tiling_arithmetic_is_diffusers():
    from diffusers.models.autoencoders.ltx2_diffusion_decoder import _tile_intervals

    from kuno_worker.backends.ltx_diffusion_decode import TILE_MIN_CELLS, tile_intervals

    for length in range(1, 400, 3):
        for tile, stride in ((40, 28), (96, 88), (12, 8)):
            assert tile_intervals(length, tile, stride, TILE_MIN_CELLS) == _tile_intervals(length, tile, stride, TILE_MIN_CELLS)


def test_the_decode_plan_covers_the_decoders_live_tensors_at_4k(monkeypatch):
    """LTX-2.5's decoder at its real widths on the meta device (nothing is allocated), at 1440p for 49 frames with the
    default tiling: the live tensors' peak, with SDPA as a fused kernel (no score matrix), stays within the plan's
    estimate of the same decode without its workspace allowance."""
    import torch.nn.functional as F
    from diffusers import FlowMatchEulerDiscreteScheduler, LTX2VideoDiffusionDecodePipeline, LTX2VideoDiffusionDecoderModel
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    from kuno_worker.backends.ltx_diffusion_decode import DECODE_HELD_BYTES, decode_activation_bytes, decode_phases, prepare_decoder

    class Live(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.live = self.peak = 0
            self.sizes, self.refs, self.seen = {}, {}, set()

        def _drop(self, key, ident):
            self.seen.discard(ident)
            self.refs[key] -= 1
            if not self.refs[key]:
                self.live -= self.sizes.pop(key)
                del self.refs[key]

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            import weakref

            out = func(*args, **(kwargs or {}))
            for tensor in tree_flatten(out)[0]:
                if isinstance(tensor, torch.Tensor) and id(tensor) not in self.seen:
                    storage = tensor.untyped_storage()
                    key = storage._cdata
                    self.seen.add(id(tensor))
                    if key not in self.sizes:
                        self.sizes[key] = storage.nbytes()
                        self.live += storage.nbytes()
                        self.peak = max(self.peak, self.live)
                    self.refs[key] = self.refs.get(key, 0) + 1
                    weakref.finalize(tensor, self._drop, key, id(tensor))
            return out

    monkeypatch.setattr(F, "scaled_dot_product_attention", lambda q, k, v, **kw: q.new_empty((*q.shape[:-1], v.shape[-1])))
    width, height, frames = FOUR_K.size_for("1440p", "16:9")[0], FOUR_K.size_for("1440p", "16:9")[1], 49
    with torch.device("meta"):
        decoder = LTX2VideoDiffusionDecoderModel().to(torch.bfloat16)
    pipeline = LTX2VideoDiffusionDecodePipeline(diffusion_decoder=prepare_decoder(decoder), scheduler=FlowMatchEulerDiscreteScheduler())
    latents = torch.empty(1, 128, 7, height // 32, width // 32, dtype=torch.bfloat16, device="meta")
    with torch.no_grad(), Live() as live:
        (video,) = pipeline(latents=latents, generator=None, output_type="pt", return_dict=False, denormalize=False)
    assert tuple(video.shape) == (1, frames, 3, height, width)
    estimate = max(decode_phases(width, height, frames, workspace=False).values())
    assert live.peak <= estimate <= 2 * live.peak  # covered, and not wildly above
    assert decode_activation_bytes(width, height, frames) >= estimate + DECODE_HELD_BYTES


# ---------------------------------------------------------------- a 4K job through the worker backend


def ffmpeg(*args: str) -> None:
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def pipelines():
    return tiny_4k_pipelines()


@pytest.fixture(scope="module")
def image(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("frames") / "first.png"
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}", "-frames:v", "1", str(path))
    return path


def job(mode: Mode = Mode.TEXT_TO_VIDEO, *, seed: int = 11, fps: int = 24, duration: float = 2, inputs=(), resolution="1440p") -> GenerationTask:
    files = []
    for index, (role, path, ref) in enumerate(inputs):
        data = path.read_bytes()
        files.append(InputFile(ref=InputRef(index=index, role=role, mime="image/png", sha256="0" * 64, size=len(data), **ref), data=data, mime="image/png"))
    params = GenerationParams(
        profile_id=FOUR_K.id, mode=mode, duration_s=duration, resolution=resolution, aspect_ratio="16:9", fps=fps, audio=True,
        input_roles=[role for role, _, _ in inputs],
    )
    return GenerationTask(job_id=str(uuid.uuid4()), profile=FOUR_K, params=params, prompt="A lighthouse in a storm", negative_prompt=None,
                          seed=seed, width=WIDTH, height=HEIGHT, inputs=files, options={})


class Watched:
    """A pipeline whose calls are recorded and whose keywords are checked against its class's own `__call__`."""

    def __init__(self, pipeline):
        self.pipeline, self.calls = pipeline, []
        self.accepted = set(list(inspect.signature(type(pipeline).__call__).parameters)[1:])

    def __call__(self, **kwargs):
        assert set(kwargs) <= self.accepted, sorted(set(kwargs) - self.accepted)
        self.calls.append({k: v for k, v in kwargs.items() if k in ("width", "height", "output_type", "return_dict", "denormalize", "num_frames", "frame_rate")})
        return self.pipeline(**kwargs)

    def __getattr__(self, name):
        return getattr(self.pipeline, name)


class Recording:
    def __init__(self, adapter):
        self.adapter, self.results = adapter, []

    def __call__(self, **call):
        self.results.append(self.adapter(**call))
        return self.results[-1]

    def __getattr__(self, name):
        return getattr(self.adapter, name)


def backend_for(tmp_path, pipelines):
    from kuno_worker.backends.runtimes import LtxAdapter

    watched = {name: Watched(p) for name, p in pipelines.items()}
    recording = Recording(LtxAdapter(watched, device="cpu"))
    return LtxResidentBackend(None, tmp_path / "work", loader=lambda profile: recording), recording, watched


def test_a_4k_text_to_video_job_renders_latents_and_decodes_them_with_the_diffusion_decoder(tmp_path, pipelines, monkeypatch):
    # The video VAE never decodes a 4K frame: the pipelines stop at latents, and the diffusion decoder makes the frames.
    monkeypatch.setattr(pipelines["text"].vae, "decode", lambda *a, **k: pytest.fail("the video VAE decoded a 4K job"))
    backend, recording, watched = backend_for(tmp_path, pipelines)
    task = job()
    assert build_call(task)["video_decoder"] == DIFFUSION_DECODER
    result = backend.generate(task, lambda value, stage: None)

    info = probe(result.data)
    frames = ltx_num_frames(2, 24)
    assert result.info.frames == info.frames == frames == 49 and (info.width, info.height) == (WIDTH, HEIGHT) and info.audio
    [raw] = recording.results
    [video] = raw["videos"]
    assert len(video) == frames and all(f.dtype == np.uint8 and f.shape == (HEIGHT, WIDTH, 3) for f in video)
    # The two-stage distilled render: half size, then full size, both returning latents; then one decode of the latents.
    assert [(c["width"], c["height"], c["output_type"]) for c in watched["text"].calls] == [(160, 96, "latent"), (320, 192, "latent")]
    assert len(watched["upsample"].calls) == 1
    assert watched["decode"].calls == [{"output_type": "pt", "return_dict": False, "denormalize": False}]
    assert set(raw["timings"]) == {"latent_render", "video_decode", "audio_decode"} and "memory_gib" not in raw
    # The sound, decoded as the pipeline decodes it: 48 kHz stereo about as long as the picture (the vocoder's own length).
    audio = raw["audio"]
    assert audio.shape[0] == 1 and audio.shape[1] == 2 and abs(audio.shape[-1] / raw["sampling_rate"] - frames / 24) < 0.1


def test_a_seed_repeats_its_4k_frames_and_another_seed_does_not(tmp_path, pipelines):
    backend, recording, _ = backend_for(tmp_path, pipelines)
    for seed in (11, 11, 12):
        backend.generate(job(seed=seed), lambda value, stage: None)
    first, again, other = (raw["videos"][0] for raw in recording.results)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not all(np.array_equal(a, b) for a, b in zip(first, other))


def test_the_decoder_draws_its_own_noise_from_the_seed(pipelines):
    from kuno_worker.backends.ltx_diffusion_decode import decode_frames

    latents = torch.randn(1, 128, 3, HEIGHT // 32, WIDTH // 32, generator=torch.Generator().manual_seed(2))
    one = decode_frames(pipelines["decode"], latents, torch.Generator().manual_seed(1))
    same = decode_frames(pipelines["decode"], latents, torch.Generator().manual_seed(1))
    two = decode_frames(pipelines["decode"], latents, torch.Generator().manual_seed(2))
    assert len(one) == 17 and all(np.array_equal(a, b) for a, b in zip(one, same)) and not all(np.array_equal(a, b) for a, b in zip(one, two))


@pytest.mark.parametrize("fps", [25, 48])
def test_4k_image_to_video_and_keyframes_render_one_full_size_pass_at_the_requested_rate(tmp_path, pipelines, image, fps):
    backend, _, watched = backend_for(tmp_path, pipelines)
    for mode, inputs in (
        (Mode.IMAGE_TO_VIDEO, [(InputRole.FIRST_FRAME, image, {})]),
        (Mode.KEYFRAMES, [(InputRole.KEYFRAME, image, {"time_s": 0.0}), (InputRole.KEYFRAME, image, {"time_s": 1.0})]),
    ):
        result = backend.generate(job(mode, fps=fps, inputs=inputs), lambda value, stage: None)
        frames = ltx_num_frames(2, fps)  # every frame rendered at the requested rate: 49 at 25 fps, 97 at 48
        assert result.info.frames == probe(result.data).frames == frames and result.info.fps == fps
    calls = watched["condition"].calls
    assert [(c["width"], c["output_type"], c["frame_rate"], c["num_frames"]) for c in calls] == [(WIDTH, "latent", float(fps), ltx_num_frames(2, fps))] * 2
    assert not watched["text"].calls and len(watched["decode"].calls) == 2


def test_a_4k_call_without_a_loaded_decoder_is_refused_before_rendering(tmp_path, pipelines):
    from kuno_worker.backends.media_tools import BackendError
    from kuno_worker.backends.runtimes import LtxAdapter

    without = {name: p for name, p in pipelines.items() if name != "decode"}
    watched = {name: Watched(p) for name, p in without.items()}
    with pytest.raises(BackendError, match="no diffusion decoder"):
        LtxAdapter(watched, device="cpu")(**build_call(job()))
    assert not any(p.calls for p in watched.values())


# ---------------------------------------------------------------- loading


def test_the_loader_builds_the_decode_pipeline_only_for_a_recipe_that_verified_the_decoder(tmp_path):
    from kuno_worker.backends.ltx_diffusion_decode import ChunkedNeighborhoodAttnProcessor
    from kuno_worker.backends.quantized import LoadPlan, diffusion_decode_pipeline
    from kuno_protocol.precision import WeightsCheck

    tiny_decoder().save_pretrained(tmp_path / "diffusion_decoder")
    recipes = load_recipes()
    weights = WeightsCheck(recipe_id="x", mode="size", model_digest="0" * 64, files=[])
    dfr = LoadPlan(profile_id=FOUR_K.id, recipe=recipes["ltx-2.5-dfr/bf16/1"], hardware=None, memory=None, weights=weights, offload="none")
    pipeline = diffusion_decode_pipeline(tmp_path, dfr, device="cpu")
    decoder = pipeline.diffusion_decoder
    assert decoder.dtype == torch.bfloat16 and decoder.use_tiling
    processors = [m.processor for m in decoder.modules() if hasattr(m, "kernel_size") and hasattr(m, "processor")]
    assert len(processors) == 6 and all(isinstance(p, ChunkedNeighborhoodAttnProcessor) for p in processors)
    fast = LoadPlan(profile_id="ltx-2.5-fast", recipe=recipes["ltx-2.5-distilled/bf16/1"], hardware=None, memory=None, weights=weights, offload="none")
    with pytest.raises(PrecisionError, match="does not include diffusion_decoder"):
        diffusion_decode_pipeline(tmp_path, fast, device="cpu")
    with pytest.raises(PrecisionError, match="missing"):
        diffusion_decode_pipeline(tmp_path / "elsewhere", dfr, device="cpu")
    assert TINY_TILING["tile_sample_min_height"] < 768  # the doubles tile smaller than the loader's defaults, to tile a tiny clip
