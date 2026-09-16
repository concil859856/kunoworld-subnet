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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from kuno_protocol.profiles import ModelProfile

log = logging.getLogger("kuno.worker.runtimes")


def _load_image(path: str):
    from PIL import Image

    with Image.open(path) as handle:
        return handle.convert("RGB").copy()


def _audio_rate(pipeline: Any, result: Any) -> int:
    """diffusers' LTX-2 output carries no sample rate; the vocoder's config has it (24 kHz for LTX-2.5)."""
    rate = getattr(result, "sampling_rate", None)
    if rate is None:
        config = getattr(getattr(pipeline, "vocoder", None), "config", None)
        rate = getattr(config, "output_sampling_rate", None)
    return int(rate or 48000)


def _empty_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LtxAdapter:
    """Turns `ltx_resident.build_call` output into diffusers calls on loaded pipelines, and writes text with the pipeline's
    bundled prompt enhancer.

    `offload` is the load plan's mode. Without offload the enhancer waits in host RAM (quantized._apply_offload) and
    `enhancer` brings it to the GPU for one piece of text; with offload, diffusers' hooks place it."""

    def __init__(self, pipelines: dict[str, Any], device: str = "cuda", offload: str = "none"):
        self.pipelines = pipelines
        self.device = device
        self.offload = offload

    @contextmanager
    def enhancer(self, pipeline: Any | None = None) -> Iterator[tuple[Any, Any]]:
        """(prompt enhancer, processor) with the enhancer on the GPU, and back in host RAM as soon as the text is written:
        about 0.6 s there and 2.7 s back for its 9.51 GiB on an RTX PRO 6000 without confidential computing (2026-09-16).
        Callers hold the model store's lock (resident.ModelStore), so it never shares the GPU with a render, whose memory
        plan doesn't count it. On the CPU again, the cache it used is released for the next render."""
        pipeline = pipeline if pipeline is not None else self.pipelines["text"]
        enhancer, processor = getattr(pipeline, "prompt_enhancer", None), getattr(pipeline, "processor", None)
        if enhancer is None or processor is None:
            # diffusers would fall back to the text encoder, which LTX-2.5 did not train for enhancement; on 2026-09-16 it
            # wrote random capital letters when asked for a plan.
            raise RuntimeError("the loaded LTX-2.5 pipeline has no prompt_enhancer and processor")
        if self.offload != "none":
            yield enhancer, processor
            return
        enhancer.to(self.device)
        try:
            yield enhancer, processor
        finally:
            enhancer.to("cpu")
            _empty_cache()

    def write_text(self, messages: list[dict[str, str]], *, seed: int, max_new_tokens: int, **sampling: Any) -> tuple[str, int]:
        """The enhancer's reply to a chat and the tokens it generated, as the 2026-09-16 GPU spike ran plan/1: the
        processor tokenizer's chat template with thinking off, `generate` seeded through `torch.manual_seed` (generate
        takes no torch.Generator), and the new tokens decoded without special tokens. Unlike diffusers' `enhance_prompt`,
        the reply is not passed through `clean_response`, which drops everything before the first letter, a JSON
        object's opening brace included."""
        import torch

        with self.enhancer() as (model, processor):
            tokenizer = processor.tokenizer
            chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(chat, return_tensors="pt").to(self.device)
            prompt_tokens = inputs["input_ids"].shape[1]
            torch.manual_seed(int(seed))
            with torch.no_grad():
                sequences = model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
            generated = sequences[0, prompt_tokens:]
            return tokenizer.decode(generated, skip_special_tokens=True), int(generated.shape[0])

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
        second_stage = call.pop("second_stage_sigmas", None)
        upsample = self.pipelines.get("upsample")
        if tap is not None:
            from .verified_gpu import ltx_verified

            # Verified mode commits a single denoising pass, so it renders in one stage at full size.
            with ltx_verified(pipeline, call, tap):
                result = pipeline(generator=generator, **call)
        elif second_stage and upsample is not None:
            result = self._two_stage(pipeline, upsample, generator, second_stage, call)
        else:
            result = pipeline(generator=generator, **call)
        return {
            "videos": getattr(result, "frames", None),
            "audio": getattr(result, "audio", None),
            "sampling_rate": _audio_rate(pipeline, result),
        }

    def enhance_prompt(self, call: dict[str, Any]) -> str:
        """The prompt `call` would condition on had it passed `enable_prompt_enhancement=True`, computed apart from the
        render so the worker can check it first. As diffusers 0.40 does inside the call (LTX2Pipeline.__call__ and
        LTX2ConditionPipeline.__call__ into `enhance_prompt`): the dedicated `prompt_enhancer` through `processor`'s chat
        template, greedy (GEMMA4_PROMPT_ENHANCEMENT_CONFIG), seeded with the job seed its generator would carry; on the
        condition pipeline with the first condition's image and LTX-2.5's image-to-video instructions, otherwise with
        the text-to-video ones. Once per job: inside the call, the distilled recipe's second pass enhanced again."""
        from diffusers.pipelines.ltx2.utils import LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT, LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT

        pipeline = self.pipelines.get(call["pipeline"]) or self.pipelines["text"]
        # diffusers' enhance_prompt moves the enhancer to the execution device itself; `enhancer` returns it to host RAM.
        with self.enhancer(pipeline):
            conditions = call.get("conditions") or []
            image = _load_image(conditions[0]["path"]) if conditions and pipeline is self.pipelines.get("condition") else None
            instructions = LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT if image is not None else LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT
            [enhanced] = pipeline.enhance_prompt(prompt=call["prompt"], system_prompt=instructions, seed=int(call["seed"]), image=image)
        return enhanced

    @staticmethod
    def _two_stage(pipeline: Any, upsample: Any, generator: Any, second_stage: list[float], call: dict[str, Any]) -> Any:
        """The distilled recipe diffusers documents for LTX-2.5: the first sigmas at half size, the video latents
        upsampled x2, then the second-stage sigmas at full size, continuing from the same audio latents."""
        width, height = call.pop("width"), call.pop("height")
        latents, audio_latents = pipeline(
            generator=generator, width=width // 2, height=height // 2, output_type="latent", return_dict=False, **call
        )
        upsampled = upsample(latents=latents, output_type="latent", return_dict=False)[0]
        stage_two = {**call, "sigmas": second_stage}
        return pipeline(
            generator=generator, width=width, height=height, latents=upsampled, audio_latents=audio_latents,
            noise_scale=second_stage[0], **stage_two,
        )

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
        adapter = LtxAdapter(pipelines, device=device, offload=plan.offload)
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
        turbo = profile.runtime == "lightx2v"
        if turbo and not turbo_lora:
            # Without it the pipeline would run the full model for 8 passes: refused before ~124 GB loads.
            raise ValueError(f"{profile.id} needs KUNO_H3_TURBO_LORA, the path of LightX2V's 8-step 768p LoRA")
        import torch
        from diffusers import ModularPipeline

        workflow = "ref2va" if profile.id == "h3-reference" else "fl2va"
        pipeline = ModularPipeline.from_pretrained(model_id, workflow=workflow)
        pipeline.load_components(dtype=torch.bfloat16)
        if turbo:
            # LightX2V's distilled LoRA: 8 passes at 1344x768, video shift 6. diffusers loads it through peft.
            pipeline.load_lora_weights(turbo_lora)
        log.info("MiniMax H3 resident for %s (%s workflow)", profile.id, workflow)
        return H3Adapter(pipeline, device=device)

    return load
