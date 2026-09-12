"""Generation backends. `mock` renders placeholder video with ffmpeg for development
and tests; `real` wires each model family to its official inference runtime."""

from __future__ import annotations

from .base import Backend, GenerationTask, InputFile, VideoResult


def build_backends(kind: str, config) -> dict[str, Backend]:
    if kind == "mock":
        from .mock import MockBackend

        return {"*": MockBackend()}
    if kind == "real":
        from .h3 import H3SglangBackend
        from .ltx import LtxPipelinesBackend

        return {
            "minimax-h3": H3SglangBackend(config.h3_fl2va_url, config.h3_ref2va_url, config.workdir),
            "ltx-2.5": LtxPipelinesBackend(config.ltx_models_dir, config.workdir),
        }
    raise ValueError(f"unknown backend kind {kind!r}")


__all__ = ["Backend", "GenerationTask", "InputFile", "VideoResult", "build_backends"]
