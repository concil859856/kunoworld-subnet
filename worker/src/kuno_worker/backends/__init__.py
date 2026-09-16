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
        from kuno_protocol.profiles import load_profiles

        from .h3 import H3SglangBackend, turbo_in_process
        from .h3_resident import H3ResidentBackend
        from .ltx_resident import LtxResidentBackend

        # MiniMax H3 is served by SGLang's servers, which are resident and which kuno-h3-worker starts beside the
        # worker: h3 on fl2va, h3-reference on ref2va, h3-turbo on fl2va with the Turbo LoRA. Verified mode for
        # h3-turbo is the one exception: its profile pins the diffusers pipeline, whose steps the worker commits to,
        # so a hardware class it pins gets that pipeline in process (and no Turbo server). SGLang has no step hook,
        # so h3 and h3-reference stay on SGLang, without step commitments, whatever the class.
        h3 = H3SglangBackend(config.h3_fl2va_url, config.h3_ref2va_url, config.workdir, turbo_url=config.h3_turbo_url)
        verified = {"hardware_class": config.verified_hardware_class, "model_digest": config.model_digest}
        if any(turbo_in_process(profile, config.verified_hardware_class) for profile in load_profiles().values()):
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
            # SGLang's servers for every H3 profile, h3-turbo included; cold has no in-process pipeline.
            "minimax-h3": H3SglangBackend(config.h3_fl2va_url, config.h3_ref2va_url, config.workdir, turbo_url=config.h3_turbo_url),
            "ltx-2.5": LtxPipelinesBackend(config.ltx_models_dir, config.workdir),
        }

    raise ValueError(f"unknown backend kind {kind!r}; use mock, real or cold")


__all__ = ["Backend", "GenerationTask", "InputFile", "VideoResult", "build_backends"]
