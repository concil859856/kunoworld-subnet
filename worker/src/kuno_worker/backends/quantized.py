"""LTX-2.5 in the precision a hardware class declares, on cards that cannot hold the bf16 pipeline.

Selection. `kuno_protocol.precision.select_recipe` maps (profile, hardware class) to one recipe:

    O1.rtx-5090-32gb.x1.fp8-cast  fp8-cast  transformer weights stored float8_e4m3fn, upcast to bf16 per
                                            layer (diffusers enable_layerwise_casting; the same plain cast
                                            as ltx-pipelines' --quantization fp8-cast, the only FP8 route
                                            for LTX-2.5, which has no FP8 checkpoint)
    O1.rtx-4090-24gb.x1.int8      int8-wo   transformer and text encoder quantized at load with torchao
                                            Int8WeightOnlyConfig(group_size=128, version=2); Lightricks'
                                            comfy-int8-convrot file is ComfyUI-only
    every other class, or none    bf16      the pipeline as published

A class the profile does not list runs bf16 with verified mode off, as `Backend.verified_enabled` does.

Refusals, all before a GPU is touched where possible:
  * the weights are not the pinned ones (`precision.verify_weights`), or no digest pins them;
  * the GPU is not the class's SKU, has less memory than the class declares, or its compute
    capability or PyTorch build cannot run the recipe;
  * the class cannot fit even the profile's smallest request in any offload mode the host RAM allows;
  * at job time, the request needs more memory than the loaded plan (`CapacityRefused`).

Memory (GiB). A linear model per recipe (precision_recipes.json) over the stage with the most latent
tokens, `((frames - 1) // 8 + 1) × (width // 32) × (height // 32)`:

    none   every component on the GPU but the prompt enhancer, which waits in host RAM and moves to the GPU
           only to write text (an enhanced prompt or a plan), between renders (runtimes.LtxAdapter.enhancer)
    model  one model on the GPU at a time (diffusers enable_model_cpu_offload): the text encoder and
           prompt enhancer alone, then the transformer with the VAEs and activations
    group  transformer, text encoder and prompt enhancer streamed a block at a time from pinned host memory
           (diffusers apply_group_offloading, block_level, use_stream); slowest, smallest

`auto` takes the lightest mode whose envelope covers the profile's largest request; if none does,
the mode with the largest envelope, and requests beyond it are refused. The estimates are unmeasured
until `worker/scripts/benchmark_ltx_quantized.py` runs on the class.

Verified mode. Quantization changes how weights are stored, not what the denoising loop carries: the
latents stay bfloat16, so the step callback, `SchedulerTrap` and `tensor_record` are unchanged. The
class id (which encodes the precision) and the recipe's weights digest go into the transcript, and
open-tier classes are compared within a tolerance.

NOT RUN ON A GPU. Everything above the torch section is plain data and tested without hardware.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kuno_protocol.envelope import EnvelopeTable, profile_max_duration
from kuno_protocol.precision import (
    ComponentPrecision,
    PrecisionError,
    PrecisionRecipe,
    WeightFile,
    WeightsCheck,
    select_recipe,
    verify_weights,
    weight_files,
)
from kuno_protocol.profiles import HardwareClass, ModelProfile, ltx_num_frames

from .media_tools import CapacityRefused

log = logging.getLogger("kuno.worker.quantized")

OFFLOAD_MODES = ("none", "model", "group")
# Left free on every card: the display, other processes, allocator fragmentation.
VRAM_RESERVE_GIB = 0.5
# Host memory kept for the OS, the worker and the CVM's own buffers when weights live in RAM.
HOST_RAM_MARGIN_GIB = 8.0
# Vendor rounding between the class's nominal size and what the driver reports.
GPU_MEMORY_TOLERANCE_GIB = 1.0
# Below this share of parameters in the storage dtype, a recipe did not apply.
MIN_QUANTIZED_FRACTION = 0.5
# CapacityRefused is defined with BackendError (media_tools), so the worker loop needn't import this module; importing
# it from here still works.


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    total_gib: float
    capability: tuple[int, int]
    torch_version: str = ""


# ------------------------------------------------------------------ selection


def resolve_recipe(profile: ModelProfile, hardware_class: str | None) -> tuple[PrecisionRecipe, HardwareClass | None]:
    """The recipe this worker loads. A class the profile does not list is ignored (bf16, verified mode off)."""
    listed = profile.verified.hardware_class(hardware_class) if profile.verified and hardware_class else None
    return select_recipe(profile, hardware_class if listed is not None else None)


def _model_tokens(text: str) -> set[str]:
    """GPU model tokens such as "4090", "h200", "6000": words with a digit that are not a memory size."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if any(c.isdigit() for c in t) and not t.endswith("gb")}


def _version(text: str) -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", text or "")
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def check_device(recipe: PrecisionRecipe, hardware: HardwareClass | None, device: DeviceInfo) -> None:
    if hardware is not None:
        wanted = _model_tokens(hardware.gpu_sku)
        if wanted and not wanted <= _model_tokens(device.name):
            raise PrecisionError(
                f"this GPU is a {device.name}, but {hardware.id} is {hardware.gpu_sku}: "
                "set KUNO_VERIFIED_HARDWARE_CLASS to the class of the GPU you have"
            )
        if hardware.vram_gb and device.total_gib + GPU_MEMORY_TOLERANCE_GIB < hardware.vram_gb:
            raise PrecisionError(f"{device.name} reports {device.total_gib:.1f} GiB; {hardware.id} needs {hardware.vram_gb:g} GiB")
    if recipe.min_compute_capability and device.capability < tuple(recipe.min_compute_capability):
        need = ".".join(map(str, recipe.min_compute_capability))
        raise PrecisionError(f"{recipe.id} needs compute capability {need}; {device.name} is {device.capability[0]}.{device.capability[1]}")
    if device.capability >= (12, 0) and device.torch_version and _version(device.torch_version) < (2, 7):
        raise PrecisionError(f"{device.name} (Blackwell, sm_120) needs PyTorch 2.7 or newer built for CUDA 12.8+; this is {device.torch_version}")


# ------------------------------------------------------------------ memory


def latent_tokens(width: int, height: int, num_frames: int) -> int:
    return ((num_frames - 1) // 8 + 1) * (width // 32) * (height // 32)


# Audio-to-video and retake encode the customer's media on the GPU before the render (ltx_edit.render_edit), and hold
# the tokens through it. Two costs, which never add up:
#
#   the encode   a retake's source clip through the video VAE's encoder, in chunks of 8 frames that give the whole-clip
#                encode's latents (ltx_chunked_encode). It runs before the pipeline is called, and its activations free when
#                it returns, so its peak is beside the weights alone, not the render's activations: a job peaks at the
#                larger of the two. The first GPU run showed it: a 5 s 720p retake peaked at 78.26 GiB, which is the
#                weights' 66.18 plus the (then unchunked) encode's 12.07, above the same clip's 77.22 GiB as text-to-video.
#   the held     the tokens encoded (a 20 s 1080p clip's 128,520 tokens x 128 features in float32 are 62 MiB) and the
#                sound's log-mel through the audio VAE (tens of MiB), on the GPU through both. A round margin above both.
#
# The encode per source pixel, which the clip's length doesn't change:
#   measured  (GPU, 2026-09-17, unchunked) 12.07 GiB for 121 frames of 1280x704: 118.9 bytes per pixel per frame.
#   counted   (CPU, live tensors, LTX-2's encoder layout at its real widths, bf16-sized) unchunked 86.0 bytes per pixel per
#             frame plus 65, so 10,470 at 121 frames; chunked into 8-frame chunks 1,797 bytes per pixel at any length, of
#             which 1,044 are the stream's cached frames.
#   estimate  the GPU measured 1.374x the CPU count unchunked (14,387 against 10,470 bytes per pixel: the allocator's
#             rounding and cuDNN's workspace are not tensors), so the chunked encode should take 1.374 x 1,797 = 2,469
#             bytes per pixel there. Admission counts 2,900, 17% more for what the CPU count can't see: 2.43 GiB at
#             1280x704, 5.64 GiB at 1920x1088 and 7.52 GiB at 2560x1088.
# Unmeasured: the chunked encode on a GPU, and whether LTX-2.5's encoder has diffusers' default layout (its cached frames
# grow with its layer count). The GPU driver (scripts/gpu-test/long_video/run_edit_modes_worker.py) records the encode's
# peak against this at 5 s and near the retake cap, and the loaded encoder's cached activations per pixel against 522.
SOURCE_ENCODE_BYTES_PER_PIXEL = 2900
SOURCE_HELD_GIB = 0.25


def source_encode_gib(mode: str | None, width: int, height: int) -> float:
    """GPU memory beside the weights an audio-to-video or retake job's encode peaks at, before its render: a retake's
    source clip through the chunked video encoder. Audio-to-video's sound is within SOURCE_HELD_GIB."""
    if mode == "retake":
        return SOURCE_ENCODE_BYTES_PER_PIXEL * width * height / 2**30
    return 0.0


def source_held_gib(mode: str | None) -> float:
    """GPU memory an audio-to-video or retake job holds through both its encode and its render: the encoded tokens."""
    return SOURCE_HELD_GIB if mode in ("retake", "audio_to_video") else 0.0


def _edit_mode(call: dict[str, Any]) -> str | None:
    edit = call.get("edit")
    return edit.get("mode") if isinstance(edit, dict) else None


def _render_frames(profile: ModelProfile, duration_s: float, fps: int) -> int:
    """Frames the transformer renders (DFR renders 48/50 fps requests at half rate, as build_call does)."""
    if profile.variant == "dfr" and fps >= 48:
        return ltx_num_frames(duration_s, fps // 2)
    return ltx_num_frames(duration_s, fps)


def call_tokens(call: dict[str, Any], width: int, height: int) -> int:
    frames = call.get("num_frames")
    if frames is None:  # a call shaped for the ltx-pipelines CLI's audio-to-video; build_call always sets num_frames now
        frames = ltx_num_frames(float(call.get("audio_max_duration", 0)), round(float(call["frame_rate"])))
    return latent_tokens(width, height, int(frames))


def profile_token_range(profile: ModelProfile) -> tuple[int, int]:
    lim = profile.limits
    sizes = [tuple(size) for ratios in lim.sizes.values() for size in ratios.values()]
    low = min(latent_tokens(w, h, _render_frames(profile, lim.min_duration_s, fps)) for w, h in sizes for fps in lim.fps)
    high = max(latent_tokens(w, h, _render_frames(profile, lim.max_duration_s, fps)) for w, h in sizes for fps in lim.fps)
    return low, high


@dataclass(frozen=True)
class MemoryPlan:
    recipe_id: str
    hardware_class: str
    offload: str
    usable_gib: float
    # peak(tokens) = max(floor_gib, token_base_gib + per_token_gib × tokens) + overhead_gib
    floor_gib: float
    token_base_gib: float
    per_token_gib: float
    overhead_gib: float
    host_ram_gib: float
    measured: bool = False
    # The weights on the GPU while a job encodes its source, before the pipeline is called: the render's base without the
    # activation line's fixed part. None counts the whole base.
    resident_gib: float | None = None

    def estimate_gib(self, tokens: int) -> float:
        return max(self.floor_gib, self.token_base_gib + self.per_token_gib * tokens) + self.overhead_gib

    @property
    def max_tokens(self) -> int:
        """The largest request this plan fits; -1 when nothing fits."""
        return self.max_tokens_with(0.0)

    def max_tokens_with(self, extra_gib: float) -> int:
        """The largest request that fits with `extra_gib` more in use through the render (source_held_gib); -1 when
        nothing fits."""
        room = self.usable_gib - self.overhead_gib - extra_gib
        if self.floor_gib > room or self.token_base_gib > room:
            return -1
        return int((room - self.token_base_gib) / self.per_token_gib) if self.per_token_gib > 0 else 1 << 40

    def encode_gib(self, extra_gib: float) -> float:
        """The peak while a job encodes its source: the resident weights, the encode's `extra_gib` and the overhead."""
        resident = self.token_base_gib if self.resident_gib is None else self.resident_gib
        return resident + extra_gib + self.overhead_gib

    def fits_encode(self, extra_gib: float) -> bool:
        return self.encode_gib(extra_gib) <= self.usable_gib


def _plan(recipe: PrecisionRecipe, hardware: HardwareClass, mode: str, usable: float) -> MemoryPlan:
    memory = recipe.memory
    c = memory.components_gib
    transformer, encoder, enhancer, other = (c.get(k, 0.0) for k in ("transformer", "text_encoder", "prompt_enhancer", "other"))
    if mode == "none":
        # A render never runs the enhancer, so it stays in host RAM and a render's peak counts every other weight. Text
        # generation is the other peak: every weight, enhancer included, and a KV cache under 0.2 GiB (2026-09-16).
        floor, base, host = memory.weights_gib, memory.weights_gib - enhancer + memory.activation_fixed_gib, enhancer
    elif mode == "model":
        floor, base, host = max(encoder, enhancer), transformer + other + memory.activation_fixed_gib, memory.weights_gib
    elif mode == "group":
        floor, base, host = memory.group_onload_gib, memory.group_onload_gib + other + memory.activation_fixed_gib, memory.weights_gib
    else:
        raise PrecisionError(f"KUNO_LTX_OFFLOAD must be auto, none, model or group, not {mode!r}")
    return MemoryPlan(
        recipe_id=recipe.id, hardware_class=hardware.id, offload=mode, usable_gib=usable, floor_gib=floor, token_base_gib=base,
        per_token_gib=memory.activation_gib_per_10k_tokens / 10_000, overhead_gib=memory.overhead_gib, host_ram_gib=host,
        measured=memory.measured, resident_gib=base - memory.activation_fixed_gib,
    )


def plan_memory(
    profile: ModelProfile, recipe: PrecisionRecipe, hardware: HardwareClass, *, host_ram_gib: float | None, mode: str = "auto"
) -> MemoryPlan:
    if not hardware.vram_gb:
        raise PrecisionError(f"{hardware.id} declares no vram_gb to plan memory against")
    usable = hardware.vram_gb - VRAM_RESERVE_GIB
    low, high = profile_token_range(profile)
    modes = OFFLOAD_MODES if mode == "auto" else (mode,)
    viable, notes = [], []
    for candidate in modes:
        plan = _plan(recipe, hardware, candidate, usable)
        if plan.host_ram_gib and host_ram_gib is not None and host_ram_gib < plan.host_ram_gib + HOST_RAM_MARGIN_GIB:
            notes.append(f"{candidate} offload needs {plan.host_ram_gib + HOST_RAM_MARGIN_GIB:.0f} GiB of host RAM, this host has {host_ram_gib:.0f}")
            continue
        if plan.max_tokens < low:
            notes.append(f"{candidate} offload needs about {plan.estimate_gib(low):.1f} GiB for the smallest request")
            continue
        viable.append(plan)
    if not viable:
        raise PrecisionError(
            f"{hardware.id} cannot serve {profile.id} with {recipe.precision} weights on {usable:.1f} GiB usable: " + "; ".join(notes)
        )
    covering = [p for p in viable if p.max_tokens >= high]
    chosen = covering[0] if covering else max(viable, key=lambda p: p.max_tokens)
    if not covering:
        log.warning(
            "%s on %s: %s offload serves up to %d latent tokens of the profile's %d; larger requests are refused",
            profile.id, hardware.id, chosen.offload, chosen.max_tokens, high,
        )
    return chosen


def host_memory_gib() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return None


def plan_for_class(
    profile: ModelProfile, hardware_class: str | None, *, host_ram_gib: float | None, mode: str = "auto", device_gib: float | None = None,
    keep_covering: bool = False,
) -> MemoryPlan | None:
    """The plan admission uses, or None: nothing to plan against (neither the class nor a device reading gives the VRAM),
    or, with `auto`, a card that fits every request of the profile with every component on the GPU. `keep_covering`
    returns that card's plan instead of None, for the jobs that need memory beyond their tokens (source_encode_gib).

    `device_gib` is the GPU's own total (probe_device): it plans the classes that declare no VRAM (the confidential ones)
    and a worker with no class, and caps a class's figure, since a "96 GB" card reports 94.97 GiB.

    A card that holds every component but not the profile's largest requests keeps them all on the GPU and gets a token
    cap. Offloading would fit those requests, but on an RTX PRO 6000 (2026-09-16) it made every 5 s shot three times
    slower (41-44 s against 13.9 s), so they go to larger cards instead."""
    recipe, hardware = resolve_recipe(profile, hardware_class)
    vram = hardware.vram_gb if hardware is not None and hardware.vram_gb else None
    if device_gib is not None:
        vram = device_gib if vram is None else min(vram, device_gib)
    if vram is None:
        return None
    if hardware is None:
        hardware = HardwareClass(id="unclassified", tier="unclassified", gpu_sku="this GPU", gpu_count=1, vram_gb=vram)
    elif hardware.vram_gb != vram:
        hardware = hardware.model_copy(update={"vram_gb": vram})
    if mode == "auto" and vram >= recipe.memory.weights_gib + recipe.memory.overhead_gib:
        whole = _plan(recipe, hardware, "none", vram - VRAM_RESERVE_GIB)
        low, high = profile_token_range(profile)
        if whole.max_tokens >= high:
            return whole if keep_covering else None
        if whole.max_tokens >= low:
            return whole
    return plan_memory(profile, recipe, hardware, host_ram_gib=host_ram_gib, mode=mode)


def longest_duration(plan: MemoryPlan, profile: ModelProfile, width: int, height: int, fps: int, held_gib: float = 0.0) -> float | None:
    """The longest duration on the profile's grid (min_duration_s + k × duration_step_s, up to its limit at this fps) whose
    latent tokens fit the plan with `held_gib` more in use; None when not even the shortest does. Tokens only grow with
    duration at a fixed size and frame rate, so every shorter duration fits too."""
    lim = profile.limits
    cap = profile_max_duration(profile, fps)
    steps = math.floor((cap - lim.min_duration_s) / lim.duration_step_s + 1e-9)
    most = plan.max_tokens_with(held_gib)
    for k in range(steps, -1, -1):
        duration = round(lim.min_duration_s + k * lim.duration_step_s, 6)
        if latent_tokens(width, height, _render_frames(profile, duration, fps)) <= most:
            return duration
    return None


def _longest_fitting(plan: MemoryPlan, profile: ModelProfile, width: int, height: int, fps: int, edit: str | None = None) -> float | None:
    """longest_duration for a job of `edit`'s mode: its encode must fit beside the weights, which is the same at every
    duration, and its render must fit with the held tokens."""
    if edit is None:
        return longest_duration(plan, profile, width, height, fps)
    held = source_held_gib(edit)
    if not plan.fits_encode(source_encode_gib(edit, width, height) + held):
        return None
    return longest_duration(plan, profile, width, height, fps, held)


def envelope_for_plan(plan: MemoryPlan, profile: ModelProfile) -> EnvelopeTable:
    """The serving envelope this plan advertises (kuno_protocol.envelope): resolution -> aspect ratio -> fps -> the longest
    duration `admit` accepts. A size and frame rate nothing fits is left out."""
    table: EnvelopeTable = {}
    for resolution, ratios in profile.limits.sizes.items():
        for aspect, (width, height) in ratios.items():
            for fps in profile.limits.fps:
                longest = longest_duration(plan, profile, width, height, fps)
                if longest is not None:
                    table.setdefault(resolution, {}).setdefault(aspect, {})[fps] = longest
    return table


def admit(plan: MemoryPlan, profile: ModelProfile, call: dict[str, Any], width: int, height: int, fps: int) -> int:
    """Raises CapacityRefused when the request cannot fit; returns its latent tokens otherwise. An audio-to-video or retake
    call must also fit its source's encode beside the weights (source_encode_gib) and hold its tokens through the render
    (source_held_gib), so near the top of the serving envelope, which is per size and frame rate, not per mode, one can be
    refused where a text-to-video job of the same length is not."""
    tokens = call_tokens(call, width, height)
    edit = _edit_mode(call)
    encode, held = source_encode_gib(edit, width, height), source_held_gib(edit)
    if tokens <= plan.max_tokens_with(held) and plan.fits_encode(encode + held):
        return tokens
    longest = _longest_fitting(plan, profile, width, height, fps, edit)
    what = f"a {edit} job" if edit else "it"
    hint = f"at {width}x{height} and {fps} fps {what} serves up to {longest:g} s" if longest else f"{what} cannot serve {width}x{height} at any duration"
    need = max(plan.estimate_gib(tokens), plan.encode_gib(encode) if edit else 0.0) + held
    source = f" and {encode:.1f} GiB to encode its source" if encode else ""
    raise CapacityRefused(
        f"{plan.hardware_class} cannot fit this request ({tokens} latent tokens{source}): about {need:.1f} GiB "
        f"with {plan.offload} offload, {plan.usable_gib:.1f} GiB usable ({'measured' if plan.measured else 'estimated'}); {hint}"
    )


# ------------------------------------------------------------------ load plan


@dataclass(frozen=True)
class LoadPlan:
    profile_id: str
    recipe: PrecisionRecipe
    hardware: HardwareClass | None
    memory: MemoryPlan | None
    weights: WeightsCheck
    offload: str


def prepare_load(
    models_dir: Path,
    profile: ModelProfile,
    *,
    hardware_class: str | None,
    model_digest: str | None,
    offload: str = "auto",
    verify: str = "full",
    allow_unpinned: bool = False,
    device: DeviceInfo | None = None,
    host_ram_gib: float | None = None,
) -> LoadPlan:
    """Everything decided before torch loads a byte. Raises PrecisionError with the reason otherwise."""
    recipe, hardware = resolve_recipe(profile, hardware_class)
    if device is not None:
        check_device(recipe, hardware, device)
    memory = plan_for_class(profile, hardware_class, host_ram_gib=host_ram_gib, mode=offload,
                            device_gib=device.total_gib if device is not None else None)
    if memory is not None:
        mode = memory.offload
    elif offload in ("auto", "none"):
        mode = "none"
    elif offload in OFFLOAD_MODES:
        mode = offload
    else:
        raise PrecisionError(f"KUNO_LTX_OFFLOAD must be auto, none, model or group, not {offload!r}")
    if hardware is None and model_digest is None and not allow_unpinned:
        # Performance mode with nothing to pin: no transcript carries a digest, so hashing 66 GB buys nothing.
        files = [WeightFile(path=p, size=(Path(models_dir) / p).stat().st_size) for p in weight_files(Path(models_dir), recipe)]
        weights = WeightsCheck(recipe_id=recipe.id, mode="size", model_digest="unpinned", files=files)
    else:
        weights = verify_weights(Path(models_dir), recipe, expected_digest=model_digest, mode=verify, allow_unpinned=allow_unpinned)  # type: ignore[arg-type]
    return LoadPlan(profile_id=profile.id, recipe=recipe, hardware=hardware, memory=memory, weights=weights, offload=mode)


# ------------------------------------------------------------------ torch (GPU image only)


def probe_device(device: str = "cuda") -> DeviceInfo:
    import torch

    if not torch.cuda.is_available():
        raise PrecisionError("no CUDA device is visible to PyTorch")
    index = torch.device(device).index or 0
    props = torch.cuda.get_device_properties(index)
    return DeviceInfo(name=props.name, total_gib=props.total_memory / 2**30, capability=(props.major, props.minor), torch_version=torch.__version__)


def dtype_census(module: Any) -> dict[str, int]:
    """Parameter elements by "<tensor type>:<dtype>". torchao keeps the original dtype on its tensor
    subclasses, so the type name is what shows a weight was quantized."""
    counts: dict[str, int] = {}
    for _name, param in module.named_parameters():
        data = getattr(param, "data", param)
        key = f"{type(data).__name__}:{str(param.dtype).removeprefix('torch.')}"
        counts[key] = counts.get(key, 0) + int(param.numel())
    return counts


def quantized_fraction(counts: dict[str, int], spec: ComponentPrecision) -> float:
    total = sum(counts.values())
    if spec.method == "none" or total == 0:
        return 1.0 if spec.method == "none" else 0.0
    if spec.method == "layerwise-cast":
        stored = sum(n for key, n in counts.items() if key.endswith(":" + spec.storage_dtype))
    else:
        stored = sum(n for key, n in counts.items() if not key.startswith(("Tensor:", "Parameter:")) or key.endswith(":int8"))
    return stored / total


def check_quantized(name: str, module: Any, spec: ComponentPrecision) -> float:
    """A recipe that silently did not apply would load bf16 weights under a quantized class's name."""
    fraction = quantized_fraction(dtype_census(module), spec)
    if fraction < MIN_QUANTIZED_FRACTION:
        raise PrecisionError(
            f"{name}: only {fraction:.0%} of its parameters are stored as {spec.storage_dtype} after {spec.method}; "
            "the quantization did not apply (check the diffusers, transformers and torchao versions)"
        )
    return fraction


def _torchao_config(spec: ComponentPrecision):
    import torchao.quantization as quantization

    factory = getattr(quantization, spec.torchao_config or "", None)
    if factory is None:
        raise PrecisionError(f"torchao has no {spec.torchao_config}: install torchao>=0.15")
    return factory(**spec.torchao_kwargs)


def _apply_cast(module: Any, spec: ComponentPrecision) -> None:
    if module is None or spec.method != "layerwise-cast":
        return
    import torch

    storage, compute = getattr(torch, spec.storage_dtype), getattr(torch, spec.compute_dtype)
    pattern = tuple(spec.skip_modules_pattern) or None
    if hasattr(module, "enable_layerwise_casting"):
        module.enable_layerwise_casting(storage_dtype=storage, compute_dtype=compute, skip_modules_pattern=pattern)
    else:  # transformers models (the text encoder)
        from diffusers.hooks import apply_layerwise_casting

        apply_layerwise_casting(module, storage_dtype=storage, compute_dtype=compute, skip_modules_pattern=pattern)


def _apply_offload(pipeline: Any, mode: str, device: str) -> None:
    import torch

    if mode == "none":
        # Everything but the prompt enhancer: no render uses it, and its 9.51 GiB in host RAM lets an RTX PRO 6000 render
        # 720p 16 s and 1080p 8 s instead of 12 s and 4 s (2026-09-16). It visits the GPU only to write text, under the
        # model store's lock (runtimes.LtxAdapter.enhancer). diffusers' `pipeline.device` prefers a component that is not
        # on the CPU, so the render still runs on `device`.
        for name, component in pipeline.components.items():
            if isinstance(component, torch.nn.Module) and name != "prompt_enhancer":
                component.to(device)
        return
    if mode == "model":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        from diffusers.hooks import apply_group_offloading

        streamed = ("transformer", "text_encoder", "prompt_enhancer")
        for name in streamed:
            module = getattr(pipeline, name, None)
            if module is not None:
                apply_group_offloading(
                    module, onload_device=torch.device(device), offload_device=torch.device("cpu"),
                    offload_type="block_level", num_blocks_per_group=1, use_stream=True,
                )
        for name, component in pipeline.components.items():
            if name not in streamed and isinstance(component, torch.nn.Module):
                component.to(device)
    vae = getattr(pipeline, "vae", None)
    if vae is not None and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()


def build_ltx_pipelines(models_dir: Path, plan: LoadPlan, device: str = "cuda") -> dict[str, Any]:
    """Loads one recipe: diffusers LTX2Pipeline with the transformer (and text encoder) in its precision."""
    import torch
    from diffusers import LTX2ConditionPipeline, LTX2Pipeline, LTX2VideoTransformer3DModel

    recipe, dtype = plan.recipe, torch.bfloat16
    specs = recipe.components
    transformer_spec = specs.get("transformer", ComponentPrecision())
    quantization = None
    if transformer_spec.method == "torchao":
        from diffusers import TorchAoConfig

        quantization = TorchAoConfig(_torchao_config(transformer_spec))
    transformer = LTX2VideoTransformer3DModel.from_pretrained(
        str(models_dir), subfolder=recipe.transformer_subfolder, torch_dtype=dtype, quantization_config=quantization
    )
    kwargs: dict[str, Any] = {"transformer": transformer, "torch_dtype": dtype}
    encoder_spec = specs.get("text_encoder")
    if encoder_spec is not None and encoder_spec.method == "torchao":
        from diffusers.quantizers import PipelineQuantizationConfig
        from transformers import TorchAoConfig as TransformersTorchAoConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_mapping={"text_encoder": TransformersTorchAoConfig(quant_type=_torchao_config(encoder_spec))}
        )
    base = LTX2Pipeline.from_pretrained(str(models_dir), **kwargs)
    _apply_cast(base.transformer, transformer_spec)
    if encoder_spec is not None:
        _apply_cast(getattr(base, "text_encoder", None), encoder_spec)
    for name, spec in specs.items():
        module = getattr(base, name, None)
        if module is not None and spec.method != "none":
            log.info("%s: %.0f%% of parameters in %s", name, 100 * check_quantized(name, module, spec), spec.storage_dtype)
    if recipe.transformer_subfolder == "transformer_full":
        # The full model's schedule, as the LTX-2.5-Diffusers card configures it for transformer_full.
        from diffusers import FlowMatchEulerDiscreteScheduler

        base.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            base.scheduler.config, use_dynamic_shifting=True, shift_terminal=0.1
        )
    _apply_offload(base, plan.offload, device)
    # The condition pipeline shares the loaded (and offload-hooked) modules rather than loading a second copy.
    condition = LTX2ConditionPipeline(**base.components)
    pipelines = {"text": base, "condition": condition, "audio": base, "dfr": base}
    if (Path(models_dir) / "latent_upsampler").is_dir():
        # The distilled two-stage recipe's x2 latent upsampler (runtimes.LtxAdapter), sharing the loaded VAE.
        from diffusers import LTX2LatentUpsamplePipeline
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

        upsampler = LTX2LatentUpsamplerModel.from_pretrained(str(models_dir), subfolder="latent_upsampler", torch_dtype=dtype)
        pipelines["upsample"] = LTX2LatentUpsamplePipeline(vae=base.vae, latent_upsampler=upsampler.to(device))
    return pipelines
