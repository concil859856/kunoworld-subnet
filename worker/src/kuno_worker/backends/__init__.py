"""Generation backends.

    mock   placeholder video rendered with ffmpeg; no GPU, used by tests and dev networks
    real   resident pipelines: weights load once and stay loaded (what serving needs)
    cold   the models' own CLI / server entry points, reloading per job; slow, but it is
           the officially documented path — use it to validate a new GPU box before
           trusting the resident runtimes
"""

from __future__ import annotations

from .base import Backend, GenerationTask, InputFile, VideoResult


def build_backends(kind: str, config) -> dict[str, Backend]:
    if kind == "mock":
        from .mock import MockBackend

        return {"*": MockBackend()}

    if kind == "real":
        from .h3 import H3SglangBackend
        from .h3_resident import H3ResidentBackend
        from .ltx_resident import LtxResidentBackend

        # SGLang is the documented H3 serving path and is already resident; it forwards
        # the Turbo profile to the LoRA runtime, which SGLang does not support.
        h3 = H3SglangBackend(config.h3_fl2va_url, config.h3_ref2va_url, config.workdir)
        verified = {"hardware_class": config.verified_hardware_class, "model_digest": config.model_digest}
        # KUNO_H3_MODEL_ID names the same weights the SGLang servers load (a local path, or a Hub id resolved
        # offline from HF_HUB_CACHE), so the Turbo pipeline never falls back to downloading the default id.
        model_id = getattr(config, "h3_model_id", None) or "MiniMaxAI/MiniMax-H3"
        h3.turbo = H3ResidentBackend(config.workdir, model_id=model_id, turbo_lora=config.h3_turbo_lora, **verified)
        ltx = LtxResidentBackend(
            config.ltx_models_dir,
            config.workdir,
            offload=getattr(config, "ltx_offload", "auto"),
            weights_verify=getattr(config, "weights_verify", "full"),
            allow_unpinned_weights=getattr(config, "allow_unpinned_weights", False),
            **verified,
        )
        return {"minimax-h3": h3, "ltx-2.5": ltx}

    if kind == "cold":
        from .h3 import H3SglangBackend
        from .ltx import LtxPipelinesBackend

        return {
            "minimax-h3": H3SglangBackend(config.h3_fl2va_url, config.h3_ref2va_url, config.workdir),
            "ltx-2.5": LtxPipelinesBackend(config.ltx_models_dir, config.workdir),
        }

    raise ValueError(f"unknown backend kind {kind!r}; use mock, real or cold")


__all__ = ["Backend", "GenerationTask", "InputFile", "VideoResult", "build_backends"]
