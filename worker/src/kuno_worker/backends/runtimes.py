"""The only code here that needs a GPU: loading the real pipelines.

Everything above this file works on plain data and is tested without hardware. These
loaders are written from the official documentation and have **not been run on a GPU**;
validate them in Phase 0 with `kuno-plan` alongside the model's own docs, then treat any
correction here as a one-file change.

torch and diffusers are imported inside the functions so a worker running the mock or
cold backends never pays for them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from kuno_protocol.profiles import ModelProfile

log = logging.getLogger("kuno.worker.runtimes")


def _load_image(path: str):
    from PIL import Image

    with Image.open(path) as handle:
        return handle.convert("RGB").copy()


class LtxAdapter:
    """Turns `ltx_resident.build_call` output into diffusers calls on loaded pipelines."""

    def __init__(self, pipelines: dict[str, Any], device: str = "cuda"):
        self.pipelines = pipelines
        self.device = device

    def __call__(self, **call: Any) -> dict[str, Any]:
        import torch

        kind = call.pop("pipeline")
        pipeline = self.pipelines.get(kind) or self.pipelines["text"]
        tap = call.pop("kuno_trajectory_tap", None)
        # Verified mode draws noise on the CPU, so every GPU of a hardware class starts from identical latents.
        generator = torch.Generator(device="cpu" if tap is not None else self.device).manual_seed(int(call.pop("seed")))

        conditions = call.pop("conditions", None)
        if conditions:
            from diffusers.pipelines.ltx2 import LTX2VideoCondition

            call["conditions"] = [
                LTX2VideoCondition(frames=_load_image(c["path"]), index=c["index"], strength=c["strength"])
                for c in conditions
            ]
        for key in ("video_path", "audio_path"):
            if key in call:
                call[key] = str(call[key])
        call.pop("generate_audio", None)  # these pipelines always produce their audio track
        if tap is not None:
            from .verified_gpu import ltx_verified

            with ltx_verified(pipeline, call, tap):
                result = pipeline(generator=generator, **call)
        else:
            result = pipeline(generator=generator, **call)
        return {
            "videos": getattr(result, "frames", None),
            "audio": getattr(result, "audio", None),
            "sampling_rate": int(getattr(result, "sampling_rate", 48000)),
        }

    def unload(self) -> None:
        import torch

        self.pipelines.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def ltx_loader(
    models_dir: Path,
    device: str = "cuda",
    *,
    hardware_class: str | None = None,
    model_digest: str | None = None,
    offload: str = "auto",
    weights_verify: str = "full",
    allow_unpinned_weights: bool = False,
    host_ram_gib: float | None = None,
    device_probe: Callable[[str], Any] | None = None,
    builder: Callable[..., dict[str, Any]] | None = None,
) -> Callable[[ModelProfile], Any]:
    """Loads LTX-2.5 once, in the precision the hardware class declares (backends/quantized.py), and
    shares its components across pipeline classes. The weights must hash to `model_digest` (the
    owner-signed manifest's entry for this profile and class) before anything reaches the GPU.
    `device_probe` and `builder` replace the torch parts in tests."""

    def load(profile: ModelProfile) -> LtxAdapter:
        from .quantized import build_ltx_pipelines, host_memory_gib, prepare_load, probe_device

        plan = prepare_load(
            Path(models_dir), profile, hardware_class=hardware_class, model_digest=model_digest, offload=offload,
            verify=weights_verify, allow_unpinned=allow_unpinned_weights, device=(device_probe or probe_device)(device),
            host_ram_gib=host_memory_gib() if host_ram_gib is None else host_ram_gib,
        )
        pipelines = (builder or build_ltx_pipelines)(Path(models_dir), plan, device)
        log.info(
            "LTX-2.5 resident for %s: %s, %s offload, weights %s", profile.id, plan.recipe.id, plan.offload, plan.weights.model_digest[:16]
        )
        adapter = LtxAdapter(pipelines, device=device)
        adapter.load_plan = plan
        return adapter

    return load


class H3Adapter:
    """Turns `h3_resident.build_call` output into MiniMax H3 modular-pipeline calls."""

    def __init__(self, pipeline: Any, device: str = "cuda"):
        self.pipeline = pipeline
        self.device = device
        self._tap = None
        self._hooked = False

    def __call__(self, **call: Any) -> dict[str, Any]:
        import torch

        tap = call.pop("kuno_trajectory_tap", None)
        generator = torch.Generator(device="cpu" if tap is not None else self.device).manual_seed(int(call.pop("seed")))
        if tap is not None and not self._hooked:
            from .verified_gpu import install_h3_commit_blocks

            install_h3_commit_blocks(self.pipeline, lambda: self._tap)
            self._hooked = True
        for key in ("image", "last_image"):
            if call.get(key):
                call[key] = _load_image(call[key])
        references = call.pop("references", None)
        if references:
            from diffusers.modular_pipelines.minimax_h3 import (
                MiniMaxH3AudioReference,
                MiniMaxH3ImageReference,
                MiniMaxH3VideoReference,
            )

            builders = {
                "image": MiniMaxH3ImageReference,
                "video": MiniMaxH3VideoReference,
                "video_audio": MiniMaxH3VideoReference,
                "audio": MiniMaxH3AudioReference,
            }
            call["references"] = [builders[r["type"]].from_file(r["path"]) for r in references]
        call.pop("aspect_ratio", None)  # size comes from width/height
        self._tap = tap  # the commit blocks report to it only for this call
        try:
            result = self.pipeline(generator=generator, output=["videos", "audio", "sampling_rate"], **call)
        finally:
            self._tap = None
        return dict(result) if isinstance(result, dict) else {"videos": result}

    def unload(self) -> None:
        import torch

        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def h3_loader(model_id: str = "MiniMaxAI/MiniMax-H3", device: str = "cuda", turbo_lora: str | None = None) -> Callable[[ModelProfile], Any]:
    """Loads H3 through the modular pipeline; `h3-reference` uses the Ref2VA workflow."""

    def load(profile: ModelProfile) -> H3Adapter:
        import torch
        from diffusers import ModularPipeline

        workflow = "ref2va" if profile.id == "h3-reference" else "fl2va"
        pipeline = ModularPipeline.from_pretrained(model_id, workflow=workflow)
        pipeline.load_components(dtype=torch.bfloat16)
        if profile.runtime == "lightx2v" and turbo_lora:
            # LightX2V's distilled LoRA: 8 steps at 1344x768, video shift 6.
            pipeline.load_lora_weights(turbo_lora)
        log.info("MiniMax H3 resident for %s (%s workflow)", profile.id, workflow)
        return H3Adapter(pipeline, device=device)

    return load
