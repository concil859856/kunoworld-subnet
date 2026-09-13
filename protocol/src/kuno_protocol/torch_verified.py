"""The PyTorch side of verified mode, shared by GPU workers (committing) and GPU validators
(replaying). torch and diffusers are imported lazily.

NOT RUN ON A GPU in this repository. Everything here follows documented behaviour and is
meant to be validated in Phase 0 with the golden-set tool (`kuno_validator.golden`), which
fails loudly on any hook that captures the wrong tensor:

* PyTorch reproducibility: `torch.use_deterministic_algorithms`, `cudnn.benchmark = False`,
  `cudnn.deterministic = True`, TF32 off, `CUBLAS_WORKSPACE_CONFIG` (`:4096:8` or `:16:8`,
  read when cuBLAS initializes, so set before the first CUDA call)
  https://docs.pytorch.org/docs/stable/notes/randomness.html,
  https://docs.pytorch.org/docs/stable/notes/cuda.html,
  https://docs.nvidia.com/cuda/cublas/index.html ("Results reproducibility": bitwise only for
  the same architecture and number of SMs, which is why hardware classes pin the SKU).
* Inductor: `TORCHINDUCTOR_DETERMINISTIC=1` skips on-device benchmarking; `max_autotune*` and
  `coordinate_descent_tuning` off (torch/_inductor/config.py). Verified profiles don't compile.
* CPU-seeded noise: diffusers recommends `torch.Generator(device="cpu")` because GPU RNG
  streams differ across devices (https://huggingface.co/docs/diffusers/using-diffusers/reusing_seeds).
* FlashAttention's forward pass is deterministic ("The forward pass is always deterministic",
  flash_attn_interface.py); NCCL documents no determinism knob, so multi-GPU classes pin
  `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple` as our choice (https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).
  Ulysses all-to-all only moves bytes; all-reduce order is the numeric risk.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .profiles import DeterminismSettings
from .verified import Tensor, TensorSpec, VerifiedModeError

_DTYPE_NAMES = ("bfloat16", "float16", "float32", "float64")


def determinism_env(settings: DeterminismSettings) -> dict[str, str]:
    env = dict(settings.env)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", settings.cublas_workspace_config)
    if not settings.kernel_autotuning:
        env.setdefault("TORCHINDUCTOR_DETERMINISTIC", "1")
    return env


def apply_determinism(settings: DeterminismSettings) -> dict[str, Any]:
    """Pins the process. Call before any CUDA work. Returns the record that goes into transcripts."""
    os.environ.update(determinism_env(settings))
    import torch

    torch.use_deterministic_algorithms(settings.use_deterministic_algorithms)
    torch.backends.cudnn.deterministic = settings.cudnn_deterministic
    torch.backends.cudnn.benchmark = settings.cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = settings.allow_tf32
    torch.backends.cudnn.allow_tf32 = settings.allow_tf32
    torch.set_float32_matmul_precision(settings.float32_matmul_precision)
    if not settings.kernel_autotuning:
        try:
            import torch._inductor.config as inductor

            inductor.max_autotune = False
            inductor.max_autotune_pointwise = False
            inductor.max_autotune_gemm = False
            inductor.coordinate_descent_tuning = False
        except ImportError:
            pass
    return determinism_record(settings)


def determinism_record(settings: DeterminismSettings) -> dict[str, Any]:
    import torch

    return {
        **settings.model_dump(mode="json"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def determinism_problems(settings: DeterminismSettings) -> list[str]:
    """What differs between the running process and the profile's pins (for preflight)."""
    import torch

    problems = []
    if torch.are_deterministic_algorithms_enabled() != settings.use_deterministic_algorithms:
        problems.append("torch.use_deterministic_algorithms")
    if torch.backends.cudnn.benchmark != settings.cudnn_benchmark:
        problems.append("cudnn.benchmark")
    if torch.backends.cudnn.deterministic != settings.cudnn_deterministic:
        problems.append("cudnn.deterministic")
    if torch.backends.cuda.matmul.allow_tf32 != settings.allow_tf32 or torch.backends.cudnn.allow_tf32 != settings.allow_tf32:
        problems.append("TF32")
    for key, value in determinism_env(settings).items():
        if os.environ.get(key) != value:
            problems.append(f"env {key}")
    return problems


def cpu_generator(seed: int):
    import torch

    return torch.Generator(device="cpu").manual_seed(int(seed))


def tensor_record(name: str, tensor) -> Tensor:
    """A torch tensor as (spec, little-endian C-order bytes), bfloat16 included."""
    import torch

    if sys.byteorder != "little":
        raise VerifiedModeError("verified mode needs a little-endian host")
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    if dtype_name not in _DTYPE_NAMES:
        raise VerifiedModeError(f"unsupported latent dtype {tensor.dtype}")
    cpu = tensor.detach().to("cpu").contiguous()
    data = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return TensorSpec(name=name, dtype=dtype_name, shape=tuple(int(d) for d in cpu.shape)), data


def tensor_from_record(tensor: Tensor, device: str = "cpu"):
    import torch

    spec, data = tensor
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(getattr(torch, spec.dtype)).reshape(spec.shape).to(device)


class SchedulerTrap:
    """Observes (and optionally overrides) scheduler steps during one pipeline call.

    diffusers' LTX-2 callback exposes only `latents`/`prompt_embeds`; the audio latents are
    stepped by `copy.deepcopy(pipe.scheduler)` and never reach the callback. Wrapping the
    scheduler class's `step` for the duration of the call sees both streams: the pipeline's own
    scheduler instance is "video", any other instance of the class is "audio". The first
    `sample` a stream steps after `set_timesteps` is that stage's initial latent.
    """

    def __init__(self, main_scheduler: Any):
        self.main = main_scheduler
        self.inputs: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        self.steps: dict[str, int] = {}
        # (stream, step index within the stage) -> tensor returned as prev_sample instead.
        self.overrides: dict[tuple[str, int], Any] = {}
        self._fresh = {"video": True, "audio": True}

    def stream(self, scheduler: Any) -> str:
        return "video" if scheduler is self.main else "audio"

    @contextmanager
    def installed(self) -> Iterator[SchedulerTrap]:
        cls = type(self.main)
        step, set_timesteps = cls.step, cls.set_timesteps
        trap = self

        def wrapped_set_timesteps(scheduler, *args, **kwargs):
            out = set_timesteps(scheduler, *args, **kwargs)
            if scheduler is trap.main:
                trap._fresh = {"video": True, "audio": True}
                trap.steps = {}
            return out

        def wrapped_step(scheduler, model_output, timestep, sample, *args, **kwargs):
            name = trap.stream(scheduler)
            if trap._fresh.get(name, True):
                trap.inputs[name] = sample.detach().clone()
                trap._fresh[name] = False
            out = step(scheduler, model_output, timestep, sample, *args, **kwargs)
            index = trap.steps.get(name, 0)
            trap.steps[name] = index + 1
            override = trap.overrides.get((name, index))
            if override is not None:
                override = override.to(device=sample.device, dtype=sample.dtype)
                out = (override, *out[1:]) if isinstance(out, tuple) else type(out)(prev_sample=override)
            trap.outputs[name] = out[0] if isinstance(out, tuple) else out.prev_sample
            return out

        cls.step, cls.set_timesteps = wrapped_step, wrapped_set_timesteps
        try:
            yield self
        finally:
            cls.step, cls.set_timesteps = step, set_timesteps


def find_denoise_loop(blocks: Any) -> Any:
    """The loop block whose sub-blocks are ["denoiser", "update"] (MiniMax H3's denoise step)."""
    sub_blocks = getattr(blocks, "sub_blocks", None) or {}
    if {"denoiser", "update"} <= set(sub_blocks):
        return blocks
    for child in sub_blocks.values():
        found = find_denoise_loop(child)
        if found is not None:
            return found
    return None


def loop_block(fn):
    """Wraps `fn(components, block_state, i, t)` as a sub-block of a Modular Diffusers loop."""
    try:
        from diffusers.modular_pipelines import ModularPipelineBlocks
    except ImportError:  # pragma: no cover - only reached on a GPU image
        ModularPipelineBlocks = object

    class _KunoLoopBlock(ModularPipelineBlocks):  # type: ignore[misc, valid-type]
        model_name = "kuno-verified"

        @property
        def inputs(self):
            return []

        @property
        def intermediate_outputs(self):
            return []

        def __call__(self, components, block_state, i, t=None, **kwargs):
            fn(components, block_state, i, t)
            return components, block_state

    return _KunoLoopBlock()
